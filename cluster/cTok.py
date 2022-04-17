# -*- encoding: utf-8 -*-
'''
@File    :   cTok.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/10/8 16:43   guzhouweihu      1.0         None
'''


import time

import torch
from torch import nn
from torch import optim
from helper.util import accuracy, AverageMeter


class cToK_net(nn.Module):

    def __init__(self, c, k):
        super(cToK_net, self).__init__()

        self.fc = nn.Linear(c, k, bias=False)
        self.bn = nn.BatchNorm1d(k)

    def forward(self, features):
        x = self.fc(features)
        x = self.bn(x)
        return x


class cToK_net_relu(nn.Module):

    def __init__(self, c, k):
        super(cToK_net_relu, self).__init__()

        self.fc = nn.Linear(c, k, bias=False)
        self.bn = nn.BatchNorm1d(k)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        x = self.fc(features)
        x = self.bn(x)
        x = self.relu(x)
        return x


class cToK_net_MLP(nn.Module):

    def __init__(self, c, k):
        super(cToK_net_MLP, self).__init__()

        self.fc1 = nn.Linear(c, c)
        self.bn = nn.BatchNorm1d(c)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(c, k)

    def forward(self, features):
        x = self.fc1(features)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class cToK_net_replaceCluster(nn.Module):
    def __init__(self, features_size, c, k):
        super(cToK_net_replaceCluster, self).__init__()

        self.fc1 = nn.Linear(features_size, c, bias=False)
        self.bn1 = nn.BatchNorm1d(c)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(c, k, bias=False)
        self.bn2 = nn.BatchNorm1d(k)

    def forward(self, features):
        x = self.fc1(features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)

        return x



def train_and_eval_cTok_SGD_CRKD(cTok, model, cluster, cfc_optimizer, train_loader, eval_loader, train_epochs, criterion, args):

    epoch_time = AverageMeter()
    for ep in range(train_epochs):
        start = time.time()

        adjust_learning_rate_CRKD(cfc_optimizer, ep)

        top1 = train_CRKD(cTok, model, cluster, train_loader, args, criterion, cfc_optimizer, ep)

        top1_eval = test_CRKD(cTok, model, cluster, eval_loader, args)

        print('\teval fc: [epoch: {0}]\t'
              'top1@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(ep, top1=top1_eval))

        epoch_time.update(time.time() - start)
    print(f'epoch train mean time: {epoch_time.avg}')


def train_CRKD(cTok, model, cluster, train_loader, args, criterion, optimizer, now_epoch):
    top1 = AverageMeter()
    top1_net = AverageMeter()
    losses = AverageMeter()

    model.train()
    cTok.train()
    with torch.enable_grad():
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.cuda(args.gpu, non_blocking=True), targets.cuda(args.gpu, non_blocking=True)

            with torch.no_grad():
                outputs, features = model(data)
                outputs, features = outputs.detach(), features.detach()
                mu = cluster(features)
                mu = mu.detach()

            cTok_out = cTok(mu)

            loss = criterion(cTok_out, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            top1_net_acc = accuracy(outputs, targets, topk=(1,))
            top1_net.update(top1_net_acc[0], data.size(0))
            top1_acc = accuracy(cTok_out, targets, topk=(1,))
            top1.update(top1_acc[0], data.size(0))
            losses.update(loss, data.size(0))

            if batch_idx % 100 == 0:
                print('\tTrain fc: [{0} {1:<3}/{2}]\t'
                      'top1_net@1 {top1_net.val:.3f} ({top1_net.avg:.3f})\t'
                      'top1@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                    now_epoch, batch_idx, len(train_loader), top1=top1, top1_net=top1_net, loss=losses))

    return top1


def test_CRKD(cTok, model, cluster, eval_loader, args):

    model.eval()
    cTok.eval()
    top1 = AverageMeter()
    with torch.no_grad():

        for batch_idx, (data, targets) in enumerate(eval_loader):
            data, targets = data.cuda(args.gpu, non_blocking=True), targets.cuda(args.gpu, non_blocking=True)

            with torch.no_grad():
                outputs_val, features_val = model(data)
                outputs_val, features_val = outputs_val.detach(), features_val.detach()
            mu_val = cluster(features_val)
            mu_val = mu_val.detach()

            cTok_out_val = cTok(mu_val)

            top1_acc = accuracy(cTok_out_val, targets, topk=(1,))
            top1.update(top1_acc[0], data.size(0))

            # if batch_idx % 40 == 0:
            #     print('\teval fc: [{0} {1:<3}/{2}]\t'
            #           'top1@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
            #         now_epoch, batch_idx, len(eval_loader), top1=top1))

    return top1

def adjust_learning_rate_CRKD(optimizer, now_epoch):
    """decrease the learning rate at 5 and 8 epoch"""
    lr = 0.1
    if now_epoch > 2:
        lr = 0.01
    if now_epoch > 3:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
