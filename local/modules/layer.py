import torch
import torch.nn as nn


class Spike_Conv2d(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=0,bias=False):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)

    def forward(self, x):
        T,B,C,H,W = x.shape
        x = x.reshape(T*B,C,H,W)
        x =self.conv(x)
        N,c,h,w = x.shape
        x = x.reshape(T,B,c,h,w)
        return x

class Spike_BatchNorm2d(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.bn=nn.BatchNorm2d(channels)

    def forward(self,x):
        T, B, C, H, W = x.shape
        x = x.reshape(T * B, C, H, W)
        x = self.bn(x)
        x = x.reshape(T, B, C, H, W)
        return x

class Spike_AdaptiveAvgPool2d(nn.Module):
    def __init__(self,output_size):
        super().__init__()
        self.avg=nn.AdaptiveAvgPool2d(output_size)

    def forward(self,x):
        T, B, C, H, W = x.shape
        x = x.reshape(T * B, C, H, W)
        x = self.avg(x)
        N, C, h, w = x.shape
        x = x.reshape(T, B, C, h, w)
        return x

class Spike_AvgPool2d(nn.Module):
    def __init__(self,kernel_size, stride):
        super().__init__()
        self.avg=nn.AvgPool2d(kernel_size, stride)

    def forward(self,x):
        T, B, C, H, W = x.shape
        x = x.reshape(T * B, C, H, W)
        x = self.avg(x)
        N, C, h, w = x.shape
        x = x.reshape(T, B, C, h, w)
        return x






