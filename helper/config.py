# -*- encoding: utf-8 -*-
'''
@File    :   config.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/7/13 14:14   guzhouweihu      1.0         None
'''

import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='cluster kd Training --PyTorch')

    # Log and save
    parser.add_argument('--print-freq', default=70, type=int, metavar='N', help='display frequence (default: 70)')
    parser.add_argument('--save-freq', default=100, type=int, metavar='EPOCHS', help='checkpoint frequency(default: 75)')
    parser.add_argument('--save-dir', default='./save_checkpoints', type=str, metavar='DIR')
    parser.add_argument('--logging-save-dir', default='./result', type=str, metavar='DIR')

    # Parallel setting
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH')

    # Data
    parser.add_argument('--dataset', default='cifar100', type=str, metavar='DATASET')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--num-labels', default=100, type=int, metavar='N', help='number of labeled samples')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                        help='batch size for supervised training form batch size (default: 128)')
    parser.add_argument('--data-root', default='./datasets/cifar', type=str, metavar='DIR')

    # Optimization
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total training epochs')
    parser.add_argument('--optim', default="sgd", type=str, metavar='TYPE', choices=['sgd', 'adam'])
    parser.add_argument('--min-lr', default=1e-4, type=float, metavar='LR',
                        help='minimum learning rate (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--nesterov', default=False, type=str2bool, metavar='BOOL',
                        help='use nesterov momentum (default: False)')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='max learning rate (default: 0.1)')

    # KD parameters
    parser.add_argument('--temperature', default=3, type=int, metavar='T', help='KD models temperature')
    parser.add_argument('--ema-weight', default=1.0, type=float, metavar='W', help='MEncoder loss weight')
    parser.add_argument('--moving-interval', default=500, type=int, metavar='N')
    parser.add_argument('--usp-weight', default=1.0, type=float, metavar='W',
                        help='the upper of unsuperivsed weight (default: 1.0)')
    parser.add_argument('--MEncoder-weight', default=1.0, type=float, metavar='W', help='MEncoder loss weight')

    parser.add_argument('--cons-loss', default='kl', type=str, metavar='LOSS', choices=['kl', 'mse'])

    # load pretrain models
    parser.add_argument('--pre-train', default=False, type=str2bool, metavar='pre train models',
                        help='Whether to use pretrain models')
    parser.add_argument('--pre-train-path', default='./checkpoint', type=str, metavar='', help='pretrain models path')
    parser.add_argument('--load-optimizer', default=False, type=str2bool, metavar='load opt',
                        help='load pre_train optimizer')

    # cluster
    parser.add_argument('--cluster-lr', default=0.01, type=float, metavar='', help='SGD cluster learning rate')
    parser.add_argument('--cluster-train-epochs', default=10, type=int, metavar='', help='SGD cluster train epochs')
    parser.add_argument('--cluster-init-epoch', default=50, type=int, help='cluster centers init epoch')
    parser.add_argument('--cluster-train-interval', default=10, type=int, metavar='t_i',
                        help='Start training after t_i epochs trained with the classifier')
    parser.add_argument('--cluster-m', default=1.2, type=float, metavar='m', help='Clustering fuzziness')
    parser.add_argument('--cluster-temperature', default=1, type=int, metavar='T', help='cluster KD temperature')
    parser.add_argument('--c', type=int, default=200, metavar='N', help='cluster numbers')
    parser.add_argument('--cluster-init-batches', type=int, default=15, metavar='N', help='')
    parser.add_argument('--cluster-init-train-epochs', type=int, default=15, help='SGD cluster init train epochs')
    parser.add_argument('--cluster-fine-tune-lr', type=float, default=0.01, help='cluster fine tuning learning rate')
    parser.add_argument('--cluster-error', type=float, default=0.001)
    parser.add_argument('--cluster-maxiter', type=float, default=7)
    parser.add_argument('--feat-dim', type=int, default=512)
    parser.add_argument('--cluster-init-interval', default=1, type=int, metavar='t_i',
                        help='Start training after t_i epochs trained with the classifier')

    # cTok param
    parser.add_argument('--cTok-lr', default=0.1, type=float, metavar='', help='cTok net learning rate')
    parser.add_argument('--cTok-init-train-epochs', default=10, type=int, metavar='N',
                        help='cTok net init train epochs')
    parser.add_argument('--cTok-train-epochs', default=20, type=int, metavar='N', help='cTok net train epochs')
    parser.add_argument('--pseudo-weight', default=1.0, type=float, metavar='T', help='')
    parser.add_argument('--pseudo-temperature', default=3, type=float, metavar='T', help='')
    parser.add_argument('--outputs-temperature', default=3, type=float, metavar='T', help='')
    parser.add_argument('--cTok-train-interval', default=1, type=int, metavar='t_i',
                        help='Start training after t_i epochs trained with the classifier')
    parser.add_argument('--cTok-train-lr', default=0.01, type=float, metavar='', help='')
    parser.add_argument('--cTok-mlp-hidden', default=512, type=int, metavar='', help='')

    # pseudo labels train
    parser.add_argument('--pseudo-train-epoch', default=40, type=int, metavar='', help='')

    # ema models
    # parser.add_argument('--ema-weight', default=0.9, type=float, metavar='W', help='Momentum smoothing encoder weight')
    parser.add_argument('--last-lr', default=0.001, type=float, metavar='N', help='set last learning rate')
    parser.add_argument('--middle-lr', default=0.01, type=float, metavar='N', help='set middle learning rate')

    # test acc model
    parser.add_argument('--trained-model-path', default="./", type=str, metavar='N')

    # Simple kd
    parser.add_argument('--teacher-model', default='resnet101', type=str)
    parser.add_argument('--teacher-model-dir', default='./', type=str)
    parser.add_argument('--student-model', default='resnet18', type=str)


    #label smoothing parameter
    parser.add_argument('--label-smoothing', default=False, type=str2bool, help='Label Smoothing use')
    parser.add_argument('--label-smoothing-alpha', default=0.1, type=float, metavar='')

    parser.add_argument('--feature-size', default=512, type=int, metavar='F', help='')

    parser.add_argument('--alpha', default=0.4, type=float)


    parser.add_argument('--ema-interval', default=3, type=int, help='')
    # mixup
    parser.add_argument('--mixup-alpha', default=1.0, type=float)

    # Maximum Entropy beta
    parser.add_argument('--max-entropy-beta', default=1.0, type=float)




    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
