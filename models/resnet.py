import torch
import torch.nn as nn
from modules.neuron import IF
from modules.sgAdaLi import AdaLi
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.layer import Conv2d,AvgPool2d,Dropout,BatchNorm2d,Flatten,Linear

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, dropout, neuron: callable = None, **kwargs):
        super(PreActBlock, self).__init__()
        whether_bias = True
        self.bn1 = BatchNorm2d(in_channels)

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=whether_bias)
        self.bn2 = BatchNorm2d(out_channels)

        self.dropout = Dropout(dropout)
        self.conv2 = Conv2d(out_channels, self.expansion * out_channels, kernel_size=3, stride=1, padding=1, bias=whether_bias)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0, bias=whether_bias)
        else:
            self.shortcut = nn.Sequential()

        self.relu1 = neuron(**kwargs)
        self.relu2 = neuron(**kwargs)

    def forward(self, x):
        x = self.relu1(self.bn1(x))
        out = self.conv1(x)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        out = out + self.shortcut(x)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, dropout, neuron: callable = None, **kwargs):
        super(PreActBottleneck, self).__init__()
        whether_bias = True

        self.bn1 = BatchNorm2d(in_channels)

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=whether_bias)
        self.bn2 = BatchNorm2d(out_channels)

        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=whether_bias)
        self.bn3 = BatchNorm2d(out_channels)
        self.dropout = Dropout(dropout)
        self.conv3 = Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, stride=1, padding=0, bias=whether_bias)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0, bias=whether_bias)
        else:
            self.shortcut = nn.Sequential()

        self.relu1 = neuron(**kwargs)
        self.relu2 = neuron(**kwargs)
        self.relu3 = neuron(**kwargs)

    def forward(self, x):
        x = self.relu1(self.bn1(x))
        out = self.conv1(x)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.dropout(self.relu3(self.bn3(out))))
        out = out + self.shortcut(x)
        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, input_channels,num_classes, dropout, neuron: callable = None, **kwargs):
        super(PreActResNet, self).__init__()
        self.num_blocks = num_blocks

        self.data_channels = input_channels
        self.init_channels = 64
        self.conv1 = Conv2d(self.data_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, dropout, neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, dropout, neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, dropout, neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, dropout, neuron, **kwargs)

        self.bn1 = BatchNorm2d(512 * block.expansion)
        self.pool = AvgPool2d(4)
        self.flat = Flatten()
        self.drop = Dropout(dropout)
        self.linear = Linear(512 * block.expansion, num_classes)

        self.relu1 = neuron(**kwargs)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, Linear):
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout, neuron, **kwargs):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.init_channels, out_channels, stride, dropout, neuron, **kwargs))
            self.init_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(self.relu1(self.bn1(out)))
        out = self.drop(self.flat(out))
        out = self.linear(out)
        return out


def spiking_resnet18(neuron: callable = None, input_channels=3,num_classes=10,  neuron_dropout=0, **kwargs):
    model = PreActResNet(PreActBlock, [2, 2, 2, 2],input_channels, num_classes, neuron_dropout, neuron=neuron, **kwargs)
    functional.set_step_mode(model, 'm')
    return model


def spiking_resnet34(neuron: callable = None, input_channels=3,num_classes=10,  neuron_dropout=0, **kwargs):
    model = PreActResNet(PreActBlock, [3, 4, 6, 3],input_channels, num_classes, neuron_dropout, neuron=neuron, **kwargs)
    functional.set_step_mode(model, 'm')
    return model

if __name__ == '__main__':
    data = torch.randn(3,2,3,32,32)
    model  = spiking_resnet18(IF,surrogate_function=AdaLi)
    print(model)
    #print(model(data).shape)