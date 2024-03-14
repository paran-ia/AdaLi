import torch
import torch.nn as nn
from modules.layer import Spike_BatchNorm2d,Spike_Conv2d,Spike_AvgPool2d,Spike_AdaptiveAvgPool2d
from modules.neuron import LIF,IF

import torch
cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class VGG(nn.Module):
    def __init__(self, vgg_name, in_channels,neuron = None, dropout=0.0, num_classes=10, **kwargs):
        super().__init__()
        self.whether_bias = True
        self.init_channels = in_channels
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, **kwargs)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, **kwargs)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, **kwargs)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, **kwargs)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, **kwargs)

        self.avgpool = Spike_AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(2,-1),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, dropout, neuron, **kwargs):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(Spike_AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Spike_Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
                layers.append(Spike_BatchNorm2d(x))
                layers.append(neuron(**kwargs))
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = self.classifier(out)
        return out.mean(0)


def vgg11(neuron, input_c,num_classes=10, neuron_dropout=0.0, **kwargs):
    return VGG('VGG11', neuron=neuron, in_channels=input_c,dropout=neuron_dropout, num_classes=num_classes, **kwargs)
def vgg16(neuron, input_c,num_classes=10, neuron_dropout=0.0, **kwargs):
    return VGG('VGG16', neuron=neuron, in_channels=input_c,dropout=neuron_dropout, num_classes=num_classes, **kwargs)
if __name__ == '__main__':
    model=vgg11(LIF,2)
    print(model)
    # for name, module in model.named_modules():
    #     if name == 'layer1.2':
    #         print(name,module)
    #     if name == 'layer5.2':
    #         print(name, module)

