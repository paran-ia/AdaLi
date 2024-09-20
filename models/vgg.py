import torch
import torch.nn as nn
from modules.neuron import IF
from modules.sgAdaLi import AdaLi
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.layer import Conv2d,AdaptiveAvgPool2d,AvgPool2d,Dropout,BatchNorm2d,Flatten,Linear

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


class SpikingVGGBN(nn.Module):
    def __init__(self, vgg_name, input_channels = 2,neuron: callable = None, dropout=0.0, num_classes=10, **kwargs):
        super(SpikingVGGBN, self).__init__()
        self.whether_bias = True
        self.init_channels = input_channels
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, **kwargs)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, **kwargs)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, **kwargs)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, **kwargs)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, **kwargs)
        self.avgpool = AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            Flatten(),
            Linear(512*7*7, num_classes),
        )

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, dropout, neuron, **kwargs):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
                layers.append(BatchNorm2d(x))
                layers.append(neuron(**kwargs))
                layers.append(Dropout(dropout))
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
        return out


def spiking_vgg11_bn(neuron: callable = None, input_channels = 2,num_classes=10, neuron_dropout=0.0, **kwargs):
    model = SpikingVGGBN('VGG11', input_channels = input_channels,neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)
    functional.set_step_mode(model, 'm')
    return model
def spiking_vgg13_bn(neuron: callable = None, input_channels = 2,num_classes=10, neuron_dropout=0.0, **kwargs):
    model = SpikingVGGBN('VGG13', input_channels = input_channels,neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)
    functional.set_step_mode(model, 'm')
    return model
def spiking_vgg16_bn(neuron: callable = None, input_channels = 2,num_classes=10, neuron_dropout=0.0, **kwargs):
    model = SpikingVGGBN('VGG16', input_channels = input_channels,neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)
    functional.set_step_mode(model, 'm')
    return model
def spiking_vgg19_bn(neuron: callable = None, input_channels = 2,num_classes=10, neuron_dropout=0.0, **kwargs):
    model = SpikingVGGBN('VGG19', input_channels = input_channels,neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)
    functional.set_step_mode(model, 'm')
    return model

if __name__ == '__main__':
    data = torch.randn(2,2,3,32,32)
    model = spiking_vgg11_bn(neuron=IF,input_channels=3,num_classes=10,neuron_dropout=0,surrogate_function=AdaLi)

    print(model(data).shape)