import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from data_process.autoaugment import CIFAR10Policy, Cutout
from torch.utils.data import DataLoader
from data_process.datasets.dvs128_gesture import DVS128Gesture
from data_process.datasets.cifar10_dvs import CIFAR10DVS
from data_process.datasets import split_to_train_test_set
import torch

def build_cifar(batch_size=128, cutout=False, workers=0, use_cifar10=True, auto_aug=False):

    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root='./data/cifar10/',
                                train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root='./data/cifar10/',
                              train=False, download=True, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='./data/cifar100/',
                                 train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root='./data/cifar100/',
                               train=False, download=False, transform=transform_test)

    #train_sampler = DistributedSampler(train_dataset)
    #val_sampler = DistributedSampler(val_dataset, round_up=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader

def build_dvs_gesture(step,batch_size,num_workers=0):
    root_dir=f'./data/DVS128Gesture'
    train_set_path = f'{root_dir}/train_set_T{step}.pt'
    test_set_path = f'{root_dir}/test_set_T{step}.pt'
    try:
        train_set = torch.load(train_set_path)
        test_set = torch.load(test_set_path)
    except:
        train_set = DVS128Gesture(root=root_dir, train=True, data_type='frame', frames_number=step,
                                  split_by='number')
        test_set = DVS128Gesture(root=root_dir, train=False, data_type='frame', frames_number=step,
                                 split_by='number')
        torch.save(train_set, train_set_path)
        torch.save(test_set, test_set_path)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_data_loader,test_data_loader

def bulid_cifar10_dvs(step,batch_size,num_workers=0):
    root_dir = './data/CIFAR-10DVS'
    train_set_path=f'{root_dir}/train_set_T{step}.pt'
    test_set_path=f'{root_dir}/test_set_T{step}.pt'

    try:
        train_set = torch.load(train_set_path)
        test_set = torch.load(test_set_path)
    except:
        dataset = CIFAR10DVS(root=root_dir, data_type='frame', frames_number=step, split_by='number')
        train_set, test_set = split_to_train_test_set(0.9,dataset,10)
        torch.save(train_set, train_set_path)
        torch.save(test_set, test_set_path)

    train_set.dataset.transform = transforms.Compose([
        transforms.Resize([48, 48]),
        transforms.RandomCrop(48, padding=4),
    ])

    test_set.dataset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([48, 48]),
    ])

    train_data_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True)

    test_data_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True)

    return  train_data_loader,test_data_loader

