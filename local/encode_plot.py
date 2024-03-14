import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models.resnet import resnet18_cifar
from models.vgg import vgg11,vgg16
from modules.neuron import IF
index=[0]
def input_hook(module, input, output):
    raw_img = input[0].clone().squeeze().mean(0).squeeze().detach().cpu()
    #raw_img = np.transpose(raw_img.numpy(), (1, 2, 0))
    np.save(f'feature/vgg16/before_{module._get_name()}{index[0]}.npy',raw_img)
    index[0] += 1
def output_hook(module, input, output):
    raw_img = output[0].clone().squeeze().mean(0).squeeze().detach().cpu().numpy()
    #raw_img = np.transpose(raw_img.numpy(), (1, 2, 0))
    np.save(f'feature/vgg16/after_{module._get_name()}{index[0]}.npy', raw_img)
    index[0] += 1
if __name__ == '__main__':
    #data=torch.randn((1,1,3,32,32))
    device = 'cuda:0'
    #model=resnet18_cifar(neuron=IF,num_classes=10,input_c=3)
    model = vgg11(IF, 3)
    model.to(device)
    for name, module in model.named_modules():
        if name == 'layer1.0':
            module.register_forward_hook(input_hook)
            print('layer1.0 hooked')
        if name == 'layer1.2':
            module.register_forward_hook(output_hook)
            print('layer1.2 hooked')
        if name == 'layer2.0':
            module.register_forward_hook(output_hook)
            print('layer2.0 hooked')
        if name == 'layer3.0':
            module.register_forward_hook(output_hook)
            print('layer3.0 hooked')
        if name == 'layer4.0':
            module.register_forward_hook(output_hook)
            print('layer4.0 hooked')
        if name == 'layer5.0':
            module.register_forward_hook(output_hook)
            print('layer5.0 hooked')
    # for name, module in model.named_modules():
    #     if name == 'conv1':
    #         module.register_forward_hook(input_hook)
    #         print('conv1 hooked')
    #     if name == 'relu':
    #         module.register_forward_hook(output_hook)
    #         print('relu hooked')
    #     if name == 'layer1.1.conv2':
    #         module.register_forward_hook(output_hook)
    #         print('layer1.1.conv2 hooked')
    #     if name == 'layer2.1.conv2':
    #         module.register_forward_hook(output_hook)
    #         print('layer2.1.conv2 hooked')
    #     if name == 'layer3.1.conv2':
    #         module.register_forward_hook(output_hook)
    #         print('layer3.1.conv2 hooked')
    #     if name == 'layer4.1.conv2':
    #         module.register_forward_hook(output_hook)
    #         print('layer4.1.conv2 hooked')
    print('register hook done')
    # 获取CIFAR-10数据集中各类别的索引
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    car_class_index = classes.index('car')

    # 定义数据转换
    transform = transforms.ToTensor()

    # 加载 CIFAR-10 数据集
    trainset = torchvision.datasets.CIFAR10(root='data\cifar10', train=True, download=True, transform=transform)

    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    for images, labels in trainloader:
        if labels != car_class_index:
            continue
        print('find car')
        images=images.to(device).unsqueeze(0)
        model(images)
        break
