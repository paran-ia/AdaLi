import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


# 定义数据转换
transform = transforms.ToTensor()

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='data\cifar10', train=True, download=True, transform=transform)

# 创建数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)


# 从数据加载器中获取一张图片和标签


# 显示图像
import matplotlib.pyplot as plt
import numpy as np
from modules.layer import Spike_Conv2d,Spike_BatchNorm2d,Spike_AdaptiveAvgPool2d
from modules.neuron import LIF
conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
device='cuda:0'
for images, labels in trainloader:
    raw_img=images.clone().detach()
    raw_img = np.transpose(raw_img.squeeze().numpy(), (1, 2, 0))
    plt.imshow(raw_img)
    plt.axis('off')
    plt.show()
    images.to(device)
# 将张量转换为图像
    #
    images=conv1(images)
    conv_img =images.clone().detach()
    conv_img = conv_img.squeeze().numpy()
    print(conv_img.shape)
    print(conv_img.max(),conv_img.min())
    plt.imshow(conv_img, cmap='jet')
    # plt.imshow(conv_img.squeeze().detach().numpy(), cmap='jet')#viridis
    plt.axis('off')
    plt.show()
    print(f'max:{torch.max(images)},min:{torch.min(images)}')
    spike = torch.gt(images, -0.1)
    plt.imshow(spike.squeeze().detach().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()

    break

# 显示图像
#plt.imshow(image)






# bn1 = Spike_BatchNorm2d(1)
# relu = LIF()


