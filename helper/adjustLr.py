# -*- encoding: utf-8 -*-
'''
@File    :   adjustLr.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/5/21 14:43   guzhouweihu      1.0         None
'''

# import lib


def adjust_learning_rate(optimizer, now_epoch, total_epoch, now_lr):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = now_lr
    # if now_epoch >= 0.25 * total_epoch:
    #     lr /= 10
    if now_epoch >= 0.5 * total_epoch:
        lr /= 10
    if now_epoch >= 0.75 * total_epoch:
        lr /= 10

    print('Epoch [{}] learning rate = {}'.format(now_epoch, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_imagenet(optimizer, now_epoch, total_epoch, now_lr):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = now_lr
    # if now_epoch >= 0.25 * total_epoch:
    #     lr /= 10
    if now_epoch >= 30:
        lr /= 10
    if now_epoch >= 60:
        lr /= 10

    print('Epoch [{}] learning rate = {}'.format(now_epoch, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
