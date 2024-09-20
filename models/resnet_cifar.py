import torch
import torch.nn as nn
import math

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.layer import Conv2d, BatchNorm2d, AdaptiveAvgPool2d, AvgPool2d,Linear
from copy import deepcopy
from modules.neuron import LIF,IF
from modules.sgAdaLi import AdaLi


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,neuron=None,**kwargs):
        super().__init__()
        self.conv1=Conv2d(inplanes,planes,stride=stride,kernel_size=3,padding=1,bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.relu1 = neuron(**deepcopy(kwargs))
        self.conv2 = Conv2d(planes,planes,kernel_size=3,padding=1,bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = neuron(**deepcopy(kwargs))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu1(out)

        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu2(out)
        return out
class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, input_c=3, num_classes=10 ,neuron = None, **kwargs):
        super().__init__()
        inplanes = 64
        self.inplanes = 64
        self.conv1 = Conv2d(input_c, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(self.inplanes)
        self.relu = neuron(**deepcopy(kwargs))
        self.layer1 = self._make_layer(block, inplanes, layers[0],neuron=neuron,**deepcopy(kwargs))
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2,neuron=neuron,**deepcopy(kwargs))
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2,neuron=neuron,**deepcopy(kwargs))
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2,neuron=neuron,**deepcopy(kwargs))
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc = Linear(inplanes * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,neuron= None,**kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,neuron,**deepcopy(kwargs)))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,neuron=neuron,**deepcopy(kwargs)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.fc(x)
        return x


class ResNet_Cifar_Modified(nn.Module):

    def __init__(self, block, layers, num_classes=10, input_c=3, rp=False,neuron = None, **kwargs):
        super(ResNet_Cifar_Modified, self).__init__()

        self.rp = rp

        self.inplanes = 64
        self.conv1 = nn.Sequential(
            Conv2d(input_c, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            neuron(**deepcopy(kwargs)),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            neuron(**deepcopy(kwargs)),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            neuron(**deepcopy(kwargs)),
        )
        self.avgpool = AvgPool2d(2)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,neuron=neuron,**deepcopy(kwargs))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,neuron=neuron,**deepcopy(kwargs))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,neuron=neuron,**deepcopy(kwargs))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,neuron=neuron,**deepcopy(kwargs))
        self.avgpool2 = AdaptiveAvgPool2d(1)
        self.fc = Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

        # zero_init_residual:
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1,neuron= None,**kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # AvgDown Layer
            downsample = nn.Sequential(
                AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,neuron,**deepcopy(kwargs)))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,neuron=neuron,**deepcopy(kwargs)))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.avgpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool2(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.fc(x)
        return x
def resnet18_cifar(neuron: callable = None, input_channels=3,num_classes=10, **kwargs):
    model = ResNet_Cifar(BasicBlock, [2, 2, 2, 2],input_c=input_channels,num_classes=num_classes,neuron=neuron, **kwargs)
    functional.set_step_mode(model,'m')
    return model
def resnet19_cifar(neuron: callable = None, input_channels=3,num_classes=10, **kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 2],input_c=input_channels,num_classes=num_classes,neuron=neuron, **kwargs)
    functional.set_step_mode(model, 'm')
    return model
def resnet20_cifar_modified(neuron: callable = None, input_channels=3,num_classes=10, **kwargs):
    model = ResNet_Cifar_Modified(block=BasicBlock,layers= [2, 2, 2, 2], num_classes=num_classes,input_c=input_channels,neuron=neuron,**kwargs)
    functional.set_step_mode(model, 'm')
    return model

if __name__ == '__main__':
    data =torch.rand(2,5,3,32,32)
    model=resnet20_cifar_modified(neuron=IF,num_classes=10,input_channels=3,surrogate_function = AdaLi)
    print(model(data).shape)
