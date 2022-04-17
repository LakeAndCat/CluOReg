# -*- encoding: utf-8 -*-
'''
@File    :   cifarDatasets.py    
@Contact :   guzhouweihu@163.com

@Modify Time      @Author        @Version    @Desciption
------------      -----------    --------    -----------
2020/10/7 19:18   guzhouweihu      1.0         None
'''
import numpy as np
from PIL import Image

import torchvision
import torchvision as tv
import torchvision.transforms as transforms
from .stanfordDogs import Dogs
from .cub2011 import Cub2011
from .tinyImagenet import TinyImageNet
from .mit67 import MITDataloder
import os


load = {}


def register_dataset(dataset):
    def warpper(f):
        load[dataset] = f
        return f

    return warpper


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class TransformNTimes:
    def __init__(self, transform, n):
        self.transform = transform
        self.n = n

    def __call__(self, inp):
        outs = []
        for i in range(0, self.n):
            outs.append(self.transform(inp))
        return outs


@register_dataset('cifar10')
def get_cifar10(data_root='../data-local/cifar/cifar10/',
                    noisy_norm=False):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    trainset = tv.datasets.CIFAR10(data_root, train=True, download=True,
                                   transform=train_transform)
    evalset = tv.datasets.CIFAR10(data_root, train=False, download=True,
                                  transform=eval_transform)
    return {
        'train_set': trainset,
        'test_set': evalset
    }


@register_dataset('cifar100')
def get_cifar100(
        data_root='../data-local/cifar/cifar100/'):
    channel_stats = dict(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = tv.datasets.CIFAR100(data_root, train=True, download=True,
                                    transform=train_transform)
    evalset = tv.datasets.CIFAR100(data_root, train=False, download=True,
                                   transform=eval_transform)
    return {
        'train_set': trainset,
        'test_set': evalset
    }


@register_dataset('cifar100_twice')
def get_cifar100(
        data_root='../data-local/cifar/cifar100/'):
    channel_stats = dict(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = tv.datasets.CIFAR100(data_root, train=True, download=True,
                                    transform=TransformTwice(train_transform))
    evalset = tv.datasets.CIFAR100(data_root, train=False, download=True,
                                   transform=eval_transform)
    return {
        'train_set': trainset,
        'test_set': evalset
    }

@register_dataset('stanfordDogs')
def get_stanfordDogs(data_root):
    data_transforms = {
        'Training': transforms.Compose([
            transforms.Resize(256),
            # transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Testing': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = Dogs(data_root, train=True, transform=data_transforms['Training'], download=True)
    test_dataset = Dogs(data_root, train=False, transform=data_transforms['Testing'], download=True)

    return {
        'train_set': train_dataset,
        'test_set': test_dataset
    }


@register_dataset('cub2011')
def get_cub2011(data_root):
    data_transforms = {
        'Training': transforms.Compose([
            transforms.Resize(256),
            # transforms.RandomCrop(256, padding=4),
            # transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Testing': transforms.Compose([
            transforms.Resize(256),
            # transforms.RandomCrop(256, padding=4),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = Cub2011(data_root, train=True, transform=data_transforms['Training'], download=True)
    test_dataset = Cub2011(data_root, train=False, transform=data_transforms['Testing'], download=True)

    return {
        'train_set': train_dataset,
        'test_set': test_dataset
    }


@register_dataset('mit67')
def get_cub2011(data_root):
    data_transforms = {
        'Training': transforms.Compose([
            transforms.Resize(256),
            # transforms.RandomCrop(256, padding=4),
            # transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Testing': transforms.Compose([
            transforms.Resize(256),
            # transforms.RandomCrop(256, padding=4),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_source = os.path.join(data_root, "labels/TrainImages.txt")
    test_source = os.path.join(data_root, "labels/TestImages.txt")

    train_dataset = MITDataloder(data_root, train_source, transform=data_transforms['Training'])
    test_dataset = MITDataloder(data_root, test_source, transform=data_transforms['Testing'])

    return {
        'train_set': train_dataset,
        'test_set': test_dataset
    }


@register_dataset('tinyImagenet')
def get_tinyImagenet(data_root):
    data_transforms = {
        'Training': transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Testing': transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = TinyImageNet(data_root, split='train', transform=data_transforms['Training'], download=True)
    test_dataset = TinyImageNet(data_root, split='val', transform=data_transforms['Testing'], download=True)

    return {
        'train_set': train_dataset,
        'test_set': test_dataset
    }


@register_dataset('imagenet')
def get_Imagenet(data_root):
    data_transforms = {
        'Training': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Testing': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'val')

    trainset = tv.datasets.ImageFolder(traindir, transform=data_transforms['Training'])
    testset = tv.datasets.ImageFolder(valdir, transform=data_transforms['Testing'])

    return {
        'train_set': trainset,
        'test_set': testset
    }

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x



@register_dataset('index_cifar100')
def get_cifar100(
        data_root='../data-local/cifar/cifar100/'):
    channel_stats = dict(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = CIFAR100_index(data_root, train=True,
                                    transform=train_transform, download=True)
    evalset = CIFAR100_index(data_root, train=False, download=True,
                                   transform=eval_transform)
    return {
        'train_set': trainset,
        'test_set': evalset
    }

class CIFAR100_index(torchvision.datasets.CIFAR100):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_index, self).__init__(root, train=train, transform=transform, target_transform=target_transform,
                                             download=download)
        # if indexs is not None:
        #     self.data = self.data[indexs]
        #     self.targets = np.array(self.targets)[indexs]
        # self.data = transpose(normalise(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target
