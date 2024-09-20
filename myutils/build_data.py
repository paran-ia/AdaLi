import torchvision.transforms as transforms
import torchvision.datasets as datasets
from myutils.transform import Cutout
import torch.utils.data as data
import os
import torch
from data_process.cifar10_dvs import CIFAR10DVS
from data_process.augmentation import ToPILImage, Resize, ToTensor
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


def build_cifar(dataset, data_dir, batch_size, num_workers=0):
    if dataset == 'CIFAR10':
        dataloader = datasets.CIFAR10
        normalization_mean = (0.4914, 0.4822, 0.4465)
        normalization_std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'CIFAR100':
        dataloader = datasets.CIFAR100
        normalization_mean = (0.5071, 0.4867, 0.4408)
        normalization_std = (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        Cutout(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
    ])

    trainset = dataloader(root=data_dir, train=True, download=True, transform=transform_train)
    train_data_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = dataloader(root=data_dir, train=False, download=False, transform=transform_test)
    test_data_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_data_loader, test_data_loader


def build_tinyimagenet(data_dir, batch_size, num_workers=0):
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_data_loader, test_data_loader


def build_cifar10dvs(data_dir, batch_size, frames_num, num_workers=0):
    transform_train = transforms.Compose([
        ToPILImage(),
        Resize(48),
        ToTensor(),
    ])

    transform_test = transforms.Compose([
        ToPILImage(),
        Resize(48),
        ToTensor(),
    ])

    trainset = CIFAR10DVS(data_dir, train=True, use_frame=True, frames_num=frames_num, split_by='number',
                          normalization=None, transform=transform_train)
    train_data_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = CIFAR10DVS(data_dir, train=False, use_frame=True, frames_num=frames_num, split_by='number',
                         normalization=None, transform=transform_test)
    test_data_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_data_loader, test_data_loader


def build_dvsgesture(data_dir, batch_size, frames_num, num_workers=0):
    trainset = DVS128Gesture(root=data_dir, train=True, data_type='frame', frames_number=frames_num, split_by='number')
    train_data_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                        drop_last=True,
                                        pin_memory=True)

    testset = DVS128Gesture(root=data_dir, train=False, data_type='frame', frames_number=frames_num, split_by='number')
    test_data_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                       drop_last=False,
                                       pin_memory=True)
    return train_data_loader, test_data_loader