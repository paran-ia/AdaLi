import torch
from torch import nn
from copy import deepcopy
from spikingjelly.activation_based import layer, functional


class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

def dvsgestureNet(neuron: callable = None, **kwargs):
    model = DVSGestureNet(neuron=neuron,**kwargs)
    functional.set_step_mode(model,'m')
    return model