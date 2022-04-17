# -*- encoding: utf-8 -*-
'''
@File    :   losses.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/7/13 15:48   guzhouweihu      1.0         None
'''

import torch.nn.functional as F
import torch
from torch import nn


def kl_loss(input_logits, target_logits, temperature):
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits / float(temperature), dim=1)
    target_softmax = F.softmax(target_logits / float(temperature), dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction='batchmean')


def mse_with_softmax(logit1, logit2):
    assert logit1.size() == logit2.size()
    return F.mse_loss(F.softmax(logit1, 1), F.softmax(logit2, 1))


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss


class KL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(KL, self).__init__()
        self.T = T

    def forward(self, y_s, p_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / p_s.shape[0]
        return loss


class Softmax_T(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(Softmax_T, self).__init__()
        self.T = T

    def forward(self, y):
        p = F.softmax(y/self.T, dim=1)
        return p


def softmax_kl_loss_kd_diff_t(input_logits, target_logits, outputs_temperature=1, pseudo_temperature=1):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits / float(outputs_temperature), dim=1)
    target_softmax = F.softmax(target_logits / float(pseudo_temperature), dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction='batchmean')


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    # input is probability distribution of output classes
    def forward(self, input):
        prob = F.softmax(input, dim=1)
        if (prob < 0).any() or (prob > 1).any():
            raise Exception('Entropy Loss takes probabilities 0<=input<=1')

        prob = prob + 1e-16  # for numerical stability while taking log
        H = torch.mean(torch.sum(prob * torch.log(prob), dim=1))

        return H


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))