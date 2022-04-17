# -*- encoding: utf-8 -*-
'''
@File    :   CluORegfinal.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2021/1/4 16:15   guzhouweihu      1.0         None
'''


import os
import sys
import csv
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from helper.config import parse_commandline_args
from helper.cifarDatasets import load
from helper.util import AverageMeter, accuracy, moving_average, update_ema
from helper.adjustLr import adjust_learning_rate, adjust_learning_rate_imagenet
from cluster.Cluster import cluster_unsup_init_CRKD, cluster_train_CRKDT
from cluster.cTok import train_and_eval_cTok_SGD_CRKD, cToK_net
from models import model_dict
from helper.losses import softmax_kl_loss_kd_diff_t, kl_loss


def main():
    args = parse_commandline_args()

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def get_soft_pseudo_labels(cTok):
    cTok_module = cTok
    if isinstance(cTok, torch.nn.parallel.DistributedDataParallel) or isinstance(cTok, torch.nn.DataParallel):
        cTok_module = cTok.module
    cTok_W = torch.zeros_like(cTok_module.fc.weight.data.detach())
    cTok_W.copy_(cTok_module.fc.weight.data.detach())

    WtW = cTok_W @ cTok_W.t()
    return WtW


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # build model
    model = model_dict[args.arch](num_classes=args.num_labels)
    ema_model = model_dict[args.arch](num_classes=args.num_labels)

    for param in ema_model.parameters():
        param.detach_()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            ema_model.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            ema_model = torch.nn.parallel.DistributedDataParallel(ema_model, device_ids=[args.gpu])
        else:
            model.cuda()
            ema_model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            ema_model = torch.nn.parallel.DistributedDataParallel(ema_model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        ema_model = ema_model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            ema_model.features = torch.nn.DataParallel(ema_model.features)
            model.cuda()
            ema_model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            ema_model = torch.nn.DataParallel(ema_model).cuda()

    # create optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)

    # criterion
    criterion = nn.CrossEntropyLoss()
    pseudo_criterion = softmax_kl_loss_kd_diff_t

    cudnn.benchmark = True

    # Data loading code
    data_root = os.path.join(args.data_root, args.dataset)
    data_config = load[args.dataset](data_root=data_root)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_config['train_set'])
    else:
        train_sampler = None
    # train_loader = torch.utils.data.DataLoader(
    #     data_config['train_set'], batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    #
    # test_loader = torch.utils.data.DataLoader(
    #     data_config['test_set'],
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        data_config['train_set'], batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        data_config['test_set'],
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print(args)

    # logging init
    save_dir = '{}-{}_{}-{}_ClusterInKD'.format(args.arch, args.dataset, args.num_labels,
                                                datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    save_dir = os.path.join(args.save_dir, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    csv_logname = os.path.join(save_dir, 'results/log_{}_{}_ClusterInKD.csv'.format(args.arch, args.dataset))
    if not os.path.exists(csv_logname):
        os.makedirs(os.path.join(save_dir, 'results'))
        with open(csv_logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                ['epoch', 'learning rate', 'train loss', 'KD loss', 'pseudo loss', 'train acc', 'ema acc',
                 'test acc', 'test ema acc'])

    start_epoch = 0
    # loop
    train_and_evaluate(model, ema_model, train_loader, test_loader, start_epoch, optimizer, csv_logname, save_dir, criterion, pseudo_criterion, args)


def train_and_evaluate(model, ema_model, train_loader, test_loader, start_epoch, optimizer, csv_logname, save_dir, criterion, pseudo_criterion, args):
    best_acc, ema_best_acc = 0., 0.
    cluster, cluster_optimizer, cTok, cTok_optimizer = None, None, None, None
    for epoch in range(start_epoch, args.epochs):
        if args.epochs == 90:
            adjust_learning_rate_imagenet(optimizer, epoch, args.epochs, args.lr)
        else:
            adjust_learning_rate(optimizer, epoch, args.epochs, args.lr)
        if epoch == (args.cluster_init_epoch - 1):
            # init ema, cluster, cTok
            moving_average(ema_model, model)
            # update_ema(ema_model, model)
            print('====== ema model init end ======')
            cluster, cluster_optimizer, cTok, cTok_optimizer = initC(model, train_loader, test_loader, criterion, args)

        time1 = time.time()
        train_acc, train_loss, ema_acc, ema_loss, pseudo_loss = train_distill(epoch, model, ema_model, cluster, cluster_optimizer, cTok, cTok_optimizer, optimizer, train_loader, criterion, pseudo_criterion, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # ema
        if epoch > (args.cluster_init_epoch - 1) and epoch < args.epochs * 0.75:
            print('ema model update')
            moving_average(ema_model, model, alpha=args.ema_weight)
            # update_ema(ema_model, model, alpha=args.ema_weight)
        elif epoch >= args.epochs * 0.75 and (epoch - 2) % args.ema_interval == 0:
            print('ema model update')
            moving_average(ema_model, model, alpha=args.ema_weight)
            # update_ema(ema_model, model, alpha=args.ema_weight)

        test_acc, test_loss, ema_test_acc = validate(test_loader, model, ema_model, cluster, criterion, args)

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
                'best_acc': ema_best_acc,
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
                'ema_weight': ema_model.state_dict(),
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
        'ema_weight': ema_model.state_dict(),
        'cTok_weight': cTok.state_dict() if cTok is not None else None,
        'accuracy': test_acc,
        'optimizer': optimizer.state_dict(),
    }
    save_file = model_out_path / '{}_last_pth'.format(args.arch)
    torch.save(state, save_file)


def train_distill(epoch, model, ema_model, cluster, cluster_optimizer, cTok, cTok_optimizer, optimizer, train_loader, criterion, pseudo_criterion, args):
    model.train()
    ema_model.train()
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

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        input, target = input.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)

        # ===================forward=====================
        outputs, features = model(input)
        print(features.shape)
        loss = criterion(outputs, target)
        acc1 = accuracy(outputs, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        now_lr = 0
        for param_group in optimizer.param_groups:
            now_lr = param_group['lr']

        if cluster is not None:
            # print(cluster)
            with torch.no_grad():
                ema_outputs, ema_features = ema_model(input)
                ema_outputs, ema_features = ema_outputs.detach(), ema_features.detach()
            cluster_mu = cluster(ema_features)

            cTok_outputs = cTok(cluster_mu.detach())

            # compute loss
            MEncoder_KD_loss = kl_loss(outputs, ema_outputs, args.temperature) * args.temperature * args.temperature

            pseudo_labels = get_soft_pseudo_labels(cTok)
            pseudo_targets = pseudo_labels[target]
            pseudo_loss = pseudo_criterion(outputs, pseudo_targets.detach(), args.outputs_temperature,
                                           args.pseudo_temperature) * args.outputs_temperature * args.pseudo_temperature

            # loss information
            ema_acc1 = accuracy(ema_outputs, target, topk=(1,))
            ema_losses.update(MEncoder_KD_loss.item(), input.size(0))
            ema_top1.update(ema_acc1[0], input.size(0))
            pseudo_losses.update(pseudo_loss.item(), input.size(0))

            # loss = loss + MEncoder_KD_loss * args.MEncoder_weight * args.temperature * args.temperature
            # loss = loss + pseudo_loss * args.pseudo_weight

            loss = loss + MEncoder_KD_loss * args.MEncoder_weight + pseudo_loss * args.pseudo_weight

            # update cluster
            if epoch % args.cluster_train_interval == 0:
                for param_group in cluster_optimizer.param_groups:
                    param_group['lr'] = args.cluster_fine_tune_lr
                cluster_module = cluster
                if isinstance(cluster, torch.nn.parallel.DistributedDataParallel) or \
                        isinstance(cluster, torch.nn.DataParallel):
                    cluster_module = cluster.module
                cluster_loss = cluster_module.computeLoss_m(ema_features.detach(), cluster_mu.detach())
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
                         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, idx, len(train_loader), lr=now_lr, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1,), end=' ')
            # sys.stdout.flush()
            if cluster is not None:
                print('ema_loss@1 {ema_loss.val:.3f} ({ema_loss.avg:.3f})\t'
                      'pseudo_loss@1 {pseudo_loss.val:.3f} ({pseudo_loss.avg:.3f})\t'
                      'ema_Acc@1 {ema_top1.val:.3f} ({ema_top1.avg:.3f})\t'.format(ema_loss=ema_losses, ema_top1=ema_top1, pseudo_loss=pseudo_losses), end=' ')
            print()
            sys.stdout.flush()
    print(' * Acc@1 {top1.avg:.3f} ema_Acc@1 {ema_top1.avg:.3f}'.format(top1=top1, ema_top1=ema_top1))

    if cluster is not None:
        return top1.avg, losses.avg, ema_top1.avg, ema_losses.avg, pseudo_losses.avg
    else:
        return top1.avg, losses.avg, 0.0, 0.0, 0.0


def validate(val_loader, model, ema_model, cluster, criterion, args):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    ema_top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    ema_model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            input, target = input.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)

            # compute output
            output, features = model(input)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

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
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, ema_top1=ema_top1), end=' ')
                if cluster is not None:
                    print('ema_Acc@1 {ema_top1.val:.3f} ({ema_top1.avg:.3f})\t'.format(ema_top1=ema_top1), end=' ')
                print()
                sys.stdout.flush()

        print(' * Acc@1 {top1.avg:.3f} ema_Acc@1 {ema_top1.avg:.3f}'.format(top1=top1, ema_top1=ema_top1))

    return top1.avg, losses.avg, ema_top1.avg


def initC(model, train_loader, eval_loader, criterion, args):
    model.train()
    with torch.no_grad():
        print('=======cluster init start========')
        cluster = cluster_unsup_init_CRKD(model, train_loader, args.c, args.cluster_m, args.cluster_lr, args.cluster_init_batches, args)
        cluster_optimizer = optim.SGD(cluster.parameters(), args.cluster_lr,
                                      momentum=0.9,
                                      weight_decay=args.weight_decay,
                                      nesterov=True)
        print('=======cluster init end========')
    with torch.enable_grad():
        print('=======cluster train start========')
        cluster_train_CRKDT(train_loader, model, cluster, cluster_optimizer, args.cluster_init_train_epochs, args)
        print('=======cluster train end========')

    cTok = cToK_net(args.c, args.num_labels)


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:

            torch.cuda.set_device(args.gpu)
            cTok = cTok.cuda(args.gpu)
            cluster = cluster.cuda(args.gpu)
            cTok = torch.nn.parallel.DistributedDataParallel(cTok, device_ids=[args.gpu])
            cluster = torch.nn.parallel.DistributedDataParallel(cluster, device_ids=[args.gpu])

            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            # cTok = torch.nn.parallel.DistributedDataParallel(cTok)
            # cluster = torch.nn.parallel.DistributedDataParallel(cluster)

        else:
            cTok.cuda()
            cluster.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            cTok = torch.nn.parallel.DistributedDataParallel(cTok)
            cluster = torch.nn.parallel.DistributedDataParallel(cluster)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        cTok = cTok.cuda(args.gpu)
        cluster = cluster.cuda(args.gpu)
    else:
        cTok = torch.nn.DataParallel(cTok).cuda()
        cluster = torch.nn.DataParallel(cluster).cuda()
    # cTok.cuda()
    # cluster.cuda()

    cfc_optimizer = optim.SGD(cTok.parameters(), args.cTok_lr,
                              momentum=0.9,
                              weight_decay=args.weight_decay,
                              nesterov=True)
    train_and_eval_cTok_SGD_CRKD(cTok, model, cluster, cfc_optimizer, train_loader, eval_loader, args.cTok_init_train_epochs, criterion, args)

    return cluster, cluster_optimizer, cTok, cfc_optimizer


if __name__ == '__main__':
    main()
