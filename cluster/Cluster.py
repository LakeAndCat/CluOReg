# -*- encoding: utf-8 -*-
'''
@File    :   Cluster.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/10/8 15:26   guzhouweihu      1.0         None
'''


import numpy as np
import math
import random
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from collections import defaultdict, deque
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
from itertools import cycle
from helper.util import accuracy, AverageMeter


class Cluster(nn.Module):
    def __init__(self, centers, cluster_lr, m=2):
        super(Cluster, self).__init__()
        self.centers = nn.Parameter(centers)
        self.m = m
        self.cluster_lr = cluster_lr
        self.cluster_now_lr = cluster_lr
        self.eps = 1e-6

    def replace_centers(self, new_centers):
        self.centers = nn.Parameter(new_centers.data.clone().detach())

    def _membership(self, features):
        mu = 1.0 / (torch.sum(torch.abs(features.unsqueeze(1) -
                                        self.centers) ** (2.0 / (self.m - 1.0)), dim=2))
        mu = mu / torch.sum(mu, dim=1, keepdim=True)

        return mu

    def _membership_new(self, features):
        k = self.centers.size(0)

        d_ic = torch.sum(torch.abs(features.unsqueeze(1) - self.centers), dim=2)

        d_ic = k * torch.sum(torch.abs(features.unsqueeze(1) - self.centers) ** (2.0 / (self.m - 1.0)), dim=2)
        d_ij = torch.sum(torch.sum(torch.abs(features.unsqueeze(1) - self.centers) ** (2.0 / (self.m - 1.0)), dim=2), dim=1, keepdim=True)

        mu = d_ij / d_ic

        return mu

    def computeLoss(self, features, mu):
        cluster_loss = (torch.pow(features.unsqueeze(1) - self.centers, 2)).mean(dim=2)
        cluster_loss = cluster_loss.mul(mu) * 0.5
        cluster_loss = cluster_loss.sum(dim=1).mean()
        return cluster_loss

    def computeLoss_m(self, features, mu):
        cluster_loss = (torch.pow(features.unsqueeze(1) - self.centers, 2)).mean(dim=2)
        cluster_loss = cluster_loss.mul(torch.pow(mu, self.m)) * 0.5
        cluster_loss = cluster_loss.sum(dim=1).mean()
        return cluster_loss

    def _membership_maximumEntropy(self, features, gama):
        mu = torch.sum(torch.abs(features.unsqueeze(1) - self.centers) ** 2, dim=2)
        mu = torch.exp(-mu / gama) / (torch.sum(torch.exp(-mu / gama), dim=1, keepdim=True))
        return mu

    def _membership_use_softmax(self, features, gama):
        mu = torch.sum(torch.abs(features.unsqueeze(1) - self.centers) ** 2, dim=2)
        mu = F.softmax(-mu / gama, dim=1)
        return mu

    def computeLoss_maximumEntropy(self, features, mu):
        cluster_loss = (torch.abs(features.unsqueeze(1) - self.centers)).mean(dim=2)
        cluster_loss = cluster_loss.mul(mu)
        cluster_loss = cluster_loss.sum(dim=1).mean()
        return cluster_loss

    def forward(self, features):
        return self._membership(features)

    def __str__(self):
        return f'centers shape: {self.centers.size()}, m: {self.m}, cluster_lr: {self.cluster_lr}'


def cluster_unsup_init_CRKD(model, data_loader, c, m, lr, cluster_init_batches, args):
    batch_size = 128
    batches_features = None

    start = time.time()
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.cuda(args.gpu, non_blocking=True), targets.cuda(args.gpu, non_blocking=True)
            ouputs, features = model(data)

            if batch_idx == 0:
                batches_features = features.cpu().numpy()
                batch_size = batches_features.shape[0]
            else:
                batches_features = np.concatenate((batches_features, features.cpu().numpy()), axis=0)
            if batch_idx >= cluster_init_batches:
                break

    init_offsets = random.sample(range(1, batch_size * cluster_init_batches + 1), c)
    centers = np.copy(batches_features[init_offsets])

    centers_torch = torch.tensor(centers, dtype=torch.float32)
    centers_torch = centers_torch.cuda(args.gpu)

    end = time.time()
    print('init_time:', end - start)

    cluster = Cluster(centers_torch, lr, m)
    return cluster


def cluster_train_CRKDT(train_loader, model, cluster, optimizer,
                  max_train_epochs, args, print_freq=100):
    model.train()

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for ep in range(max_train_epochs):
        NMI_score = AverageMeter()
        epoch_time = AverageMeter()

        start = time.time()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.cuda(args.gpu, non_blocking=True), targets.cuda(args.gpu, non_blocking=True)

            with torch.no_grad():
                outputs, features = model(data)
                outputs, features = outputs.detach(), features.detach()
            mu = cluster(features)
            mu = mu.detach()
            loss = cluster.computeLoss_m(features, mu)

            # NMI_score.update(normalized_mutual_info_score(targets.cpu().numpy(), mu.max(1)[1].cpu().numpy()),
            #                  data.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % print_freq == 0:
                print('\tTrain cluster: [{0:<3}/{1}]\t'
                      'NMI@1 {NMI.val:.6f} ({NMI.avg:.6f})\t'.format(
                    batch_idx, len(train_loader), NMI=NMI_score))

        scheduler.step()

        epoch_time.update(time.time() - start)

        for param_group in optimizer.param_groups:
            now_lr = param_group['lr']

        print('Train cluster: [epoch:{0}/{1}, {2}]\t'
              'Time {epoch_time.val:.3f} ({epoch_time.avg:.3f})\t'
              'NMI@1 {NMI.val:.6f} ({NMI.avg:.6f})\t'
              'cluster_lr {cluster_lr}'.format(
            ep, max_train_epochs, len(train_loader), epoch_time=epoch_time,
            NMI=NMI_score, cluster_lr=now_lr))

    print('train finished!')
