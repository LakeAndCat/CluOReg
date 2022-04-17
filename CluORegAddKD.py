# -*- encoding: utf-8 -*-
'''
@File    :   CluORegAddKD.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2021/1/3 21:52   guzhouweihu      1.0         None
'''


import os
import sys
import csv
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from helper.config import parse_commandline_args
from helper.cifarDatasets import load
from helper.util import AverageMeter, accuracy, moving_average
from helper.adjustLr import adjust_learning_rate
from cluster.Cluster import cluster_unsup_init, cluster_train
from cluster.cTok import train_and_eval_cTok_SGD, cToK_net
from models import model_dict
from helper.losses import softmax_kl_loss_kd_diff_t, KDLoss, kl_loss


def create_loaders(train_set, test_set, config):

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=config.workers)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=config.workers)

    return train_loader, test_loader


def get_soft_pseudo_labels(cTok):
    cTok_W = torch.zeros_like(cTok.fc.weight.data.detach())
    cTok_W.copy_(cTok.fc.weight.data.detach())

    WtW = cTok_W @ cTok_W.t()
    return WtW


def train_distill(epoch, model, ema_model, teacher_model, cluster, cluster_optimizer, cTok, cTok_optimizer, train_loader):
    model.train()
    ema_model.train()
    teacher_model.train()
    if cluster is not None and cTok is not None:
        cluster.train()
        cTok.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    ema_top1 = AverageMeter()
    ema_losses = AverageMeter()
    pseudo_losses = AverageMeter()
    teacher_top1 = AverageMeter()
    cons_losses = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        input, target = input.to(device), target.to(device)

        # ===================forward=====================
        outputs, features = model(input)
        t_outputs, t_features = teacher_model(input)

        loss = criterion(outputs, target)
        acc1 = accuracy(outputs, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        # _simpleKD
        kd_loss = kl_loss(outputs, t_outputs.detach(), args.temperature)
        loss += kd_loss * args.temperature * args.temperature

        # _simpleKD
        teacher_acc1 = accuracy(t_outputs, target, topk=(1,))
        cons_losses.update(kd_loss.item(), input.size(0))
        teacher_top1.update(teacher_acc1[0], input.size(0))

        now_lr = 0
        for param_group in optimizer.param_groups:
            now_lr = param_group['lr']

        if cluster is not None:
            # print(cluster)
            with torch.no_grad():
                ema_outputs, ema_features = ema_model(input)
                ema_outputs, ema_features = ema_outputs.detach(), ema_features.detach()
            cluster_mu = cluster(ema_features.detach())

            cTok_outputs = cTok(cluster_mu.detach())

            # compute loss
            MEncoder_KD_loss = kl_loss(outputs, ema_outputs, args.temperature) * args.temperature * args.temperature

            pseudo_labels = get_soft_pseudo_labels(cTok)
            pseudo_targets = pseudo_labels[target]
            pseudo_loss = pseudo_criterion(outputs, pseudo_targets, args.outputs_temperature,
                                           args.pseudo_temperature) * args.outputs_temperature * args.pseudo_temperature

            # loss information
            ema_acc1 = accuracy(ema_outputs, target, topk=(1,))
            ema_losses.update(MEncoder_KD_loss.item(), input.size(0))
            ema_top1.update(ema_acc1[0], input.size(0))
            pseudo_losses.update(pseudo_loss.item(), input.size(0))

            loss = loss + MEncoder_KD_loss * args.MEncoder_weight
            loss = loss + pseudo_loss * args.pseudo_weight

            # update cluster
            if epoch % args.cluster_train_interval == 0:
                for param_group in cluster_optimizer.param_groups:
                    param_group['lr'] = args.cluster_fine_tune_lr
                cluster_loss = cluster.computeLoss(ema_features, cluster_mu.detach())
                cluster_optimizer.zero_grad()
                cluster_loss.backward()
                cluster_optimizer.step()

            # update cTok
            if epoch % args.cTok_train_interval == 0:

                for param_group in cTok_optimizer.param_groups:
                    param_group['lr'] = args.cTok_train_lr
                cTok_loss = criterion(cTok_outputs, target)
                cTok_optimizer.zero_grad()
                cTok_loss.backward()
                cTok_optimizer.step()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                         'learning rate: {lr}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'cons_loss@1 {cons_loss.val:.3f} ({cons_loss.avg:.3f})\t'
                         'teacher_Acc@1 {teacher_top1.val:.3f} ({teacher_top1.avg:.3f})\t'.format(
                epoch, idx, len(train_loader), lr=now_lr, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1,
                cons_loss=cons_losses, teacher_top1=teacher_top1), end=' ')
            # sys.stdout.flush()
            if cluster is not None:
                print('ema_loss@1 {ema_loss.val:.3f} ({ema_loss.avg:.3f})\t'
                      'pseudo_loss@1 {pseudo_loss.val:.3f} ({pseudo_loss.avg:.3f})\t'
                      'ema_Acc@1 {ema_top1.val:.3f} ({ema_top1.avg:.3f})\t'.format(ema_loss=ema_losses, ema_top1=ema_top1, pseudo_loss=pseudo_losses), end=' ')
            print()
            sys.stdout.flush()
    print(' * Acc@1 {top1.avg:.3f} ema_Acc@5 {ema_top1.avg:.3f} teacher_Acc@1 {teacher_top1.avg:.3f}'.format(top1=top1, ema_top1=ema_top1, teacher_top1=teacher_top1))

    if cluster is not None:
        return top1.avg, losses.avg, ema_top1.avg, ema_losses.avg, pseudo_losses.avg
    else:
        return top1.avg, losses.avg, 0.0, 0.0, 0.0


def validate(val_loader, model, ema_model, teacher_model, cluster):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    ema_top1 = AverageMeter()
    teacher_top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    ema_model.eval()
    teacher_model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            input, target = input.to(device), target.to(device)

            # compute output
            output, features = model(input)

            t_outputs, t_features = teacher_model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            teacher_acc1 = accuracy(t_outputs, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            teacher_top1.update(teacher_acc1[0], input.size(0))

            if cluster is not None:
                ema_outputs, ema_features = ema_model(input)
                ema_acc1 = accuracy(ema_outputs, target, topk=(1,))
                ema_top1.update(ema_acc1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'ema_Acc@1 {ema_top1.val:.3f} ({ema_top1.avg:.3f})\t'
                      'teacher_Acc@1 {teacher_top1.val:.3f} ({teacher_top1.avg:.3f})\t'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, ema_top1=ema_top1, teacher_top1=teacher_top1))
                if cluster is not None:
                    print('ema_Acc@1 {ema_top1.val:.3f} ({ema_top1.avg:.3f})\t'.format(ema_top1=ema_top1))

        print(' * Acc@1 {top1.avg:.3f} ema_Acc@1 {ema_top1.avg:.3f} teacher_Acc@1 {teacher_top1.avg:.3f}'.format(top1=top1, ema_top1=ema_top1, teacher_top1=teacher_top1))

    return top1.avg, losses.avg, ema_top1.avg


def initC(model, train_loader, eval_loader):
    model.train()
    with torch.no_grad():
        print('=======cluster init start========')
        cluster = cluster_unsup_init(model, train_loader, args.c, args.cluster_m, args.cluster_lr, args.cluster_init_batches, device)
        cluster_optimizer = optim.SGD(cluster.parameters(), args.cluster_lr,
                                      momentum=0.9,
                                      weight_decay=args.weight_decay,
                                      nesterov=True)
        print('=======cluster init end========')
    with torch.enable_grad():
        print('=======cluster train start========')
        cluster_train(train_loader, model, cluster, cluster_optimizer, args.cluster_init_train_epochs, device)
        print('=======cluster train end========')

    cTok = cToK_net(args.c, args.num_labels).to(device)
    cfc_optimizer = optim.SGD(cTok.parameters(), args.cTok_lr,
                              momentum=0.9,
                              weight_decay=args.weight_decay,
                              nesterov=True)
    train_and_eval_cTok_SGD(cTok, model, cluster, cfc_optimizer, train_loader, eval_loader, args.cTok_init_train_epochs, criterion, device)

    return cluster, cluster_optimizer, cTok, cfc_optimizer


def train_and_evaluate(model, ema_model, teacher_model, train_loader, test_loader, start_epoch):

    best_acc, ema_best_acc = 0., 0.
    cluster, cluster_optimizer, cTok, cTok_optimizer = None, None, None, None
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.epochs, args.lr)
        if epoch == (args.cluster_init_epoch - 1):
            # 初始化ema与cluster与cTok
            moving_average(ema_model, model)
            print('====== ema model init end ======')
            cluster, cluster_optimizer, cTok, cTok_optimizer = initC(model, train_loader, test_loader)

        time1 = time.time()
        train_acc, train_loss, ema_acc, ema_loss, pseudo_loss = train_distill(epoch, model, ema_model, teacher_model, cluster, cluster_optimizer, cTok, cTok_optimizer, train_loader)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # ema
        if epoch > (args.cluster_init_epoch - 1):
            moving_average(ema_model, model, alpha=args.ema_weight)

        test_acc, test_loss, ema_test_acc = validate(test_loader, model, ema_model, teacher_model, cluster)

        lr = None
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        with open(csv_logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, lr, train_loss, ema_loss, pseudo_loss, train_acc, ema_acc,
                                test_acc, ema_test_acc])

        if test_acc > best_acc:
            model_out_path = Path(save_dir)
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model_weight': model.state_dict(),
                # 'ema_weight': ema_model.state_dict(),
                'cTok_weight': cTok.state_dict() if cTok is not None else None,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            if not model_out_path.exists():
                model_out_path.mkdir()
            save_file = model_out_path / '{}_best.pth'.format(args.arch)
            torch.save(state, save_file)
            print('save the best model!')

        if epoch > (args.cluster_init_epoch - 1) and ema_test_acc > ema_best_acc:
            model_out_path = Path(save_dir)
            ema_best_acc = ema_test_acc
            state = {
                'epoch': epoch,
                # 'model_weight': model.state_dict(),
                'ema_weight': ema_model.state_dict(),
                'cTok_weight': cTok.state_dict() if cTok is not None else None,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            if not model_out_path.exists():
                model_out_path.mkdir()
            save_file = model_out_path / '{}_ema_best.pth'.format(args.arch)
            torch.save(state, save_file)
            print('save the ema best model!')

        if (epoch+1) % args.save_freq == 0:
            model_out_path = Path(save_dir)
            print('==> Saveing...')
            state = {
                'epoch': epoch,
                'model_weight': model.state_dict(),
                'ema_wieght': ema_model.state_dict(),
                'cTok_weight': cTok.state_dict() if cTok is not None else None,
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            if not model_out_path.exists():
                model_out_path.mkdir()
            save_file = model_out_path / 'ckpt_epoch_{}.pth'.format(epoch)
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc, 'ema best accuracy:', ema_best_acc)

    # save model
    model_out_path = Path(save_dir)
    state = {
        'model_weight': model.state_dict(),
        'ema_wieght': ema_model.state_dict(),
        'cTok_weight': cTok.state_dict() if cTok is not None else None,
        'accuracy': test_acc,
        'optimizer': optimizer.state_dict(),
    }
    save_file = model_out_path / '{}_last_pth'.format(args.arch)
    torch.save(state, save_file)

args = parse_commandline_args()

# logging init
save_dir = '{}-{}_{}-{}_ClusterInKD'.format(args.arch, args.dataset, args.num_labels,
                                         datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
save_dir = os.path.join(args.save_dir, save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


csv_logname = os.path.join(save_dir, 'results/log_{}_{}_{}_simpleKD.csv'.format(args.student_model, args.teacher_model, args.dataset))
if not os.path.exists(csv_logname):
    os.makedirs(os.path.join(save_dir, 'results'))
    with open(csv_logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'learning rate', 'train loss', 'KD loss', 'pseudo loss', 'train acc', 'ema acc',
                            'test acc', 'test ema acc'])


# load data
data_root = os.path.join(args.data_root, args.dataset)
data_config = load[args.dataset](data_root=data_root)

train_loader, test_loader = create_loaders(**data_config, config=args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# create net
model = model_dict[args.arch](num_classes=args.num_labels).to(device)
ema_model = model_dict[args.arch](num_classes=args.num_labels).to(device)
teacher_model = model_dict[args.teacher_model](num_classes=args.num_labels).to(device)

# load teahcer model
checkpoint = torch.load(args.teacher_model_dir)
teacher_model.load_state_dict(checkpoint['model_weight'])

pytorch_total_params = sum(p.numel() for p in model.parameters())
print("模型参数量：", pytorch_total_params)

# create optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      nesterov=args.nesterov)

# criterion
criterion = nn.CrossEntropyLoss()
# kdLoss = KDLoss(args.temperature)
pseudo_criterion = softmax_kl_loss_kd_diff_t

# load pre train
start_epoch = 0
# if args.pre_train:
#     start_epoch = load_pre_train_model(model, optimizer, args.pre_train_path)


# loop
train_and_evaluate(model, ema_model, teacher_model, train_loader, test_loader, start_epoch)
