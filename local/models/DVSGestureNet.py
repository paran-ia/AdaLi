import torch
import torch.nn as nn
from spikingjelly.activation_based import layer,functional


class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**kwargs))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(start_dim=1,end_dim=-1),

            layer.Dropout(0.5),

            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**kwargs),

            layer.Dropout(0.5),
            layer.Linear(512, 110),
            spiking_neuron(**kwargs),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

def DvsgestureNet(neuron,**kwargs):
    model = DVSGestureNet(spiking_neuron=neuron, **kwargs)
    return model


