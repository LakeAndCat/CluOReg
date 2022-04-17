# -*- encoding: utf-8 -*-
'''
@File    :   mixupUtils.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/12/22 20:07   guzhouweihu      1.0         None
'''

import numpy as np
import torch


def mixup_data(x, y, args, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    # if use_cuda:
    #     index = torch.randperm(batch_size).cuda()
    # else:
    index = torch.randperm(batch_size).cuda(args.gpu, non_blocking=True)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_criterion_distillation(criterion, pred, y_a, y_b, lam, temperature):
    return lam * criterion(pred, y_a, temperature) + (1 - lam) * criterion(pred, y_b, temperature)


def mixup_criterion_pseudo_distillation(criterion, pred, y_a, y_b, lam, pseudo_labels, output_temperature,
                                        pseudo_temperature):
    T_2 = output_temperature * pseudo_temperature
    return lam * criterion(pred, pseudo_labels[y_a], output_temperature, pseudo_temperature) * T_2 + (
                1 - lam) * criterion(
        pred, pseudo_labels[y_b], output_temperature, pseudo_temperature) * T_2
