import torch
import torch.nn as nn
import math
from modules.layer import Spike_Conv2d,Spike_BatchNorm2d,Spike_AdaptiveAvgPool2d
from copy import deepcopy
from modules.neuron import LIF
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,spiking_neuron=None,**kwargs):
        super().__init__()
        self.conv1=Spike_Conv2d(inplanes,planes,stride=stride,kernel_size=3,padding=1,bias=False)
        self.bn1 = Spike_BatchNorm2d(planes)
        self.relu1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = Spike_Conv2d(planes,planes,kernel_size=3,padding=1,bias=False)
        self.bn2 = Spike_BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = spiking_neuron(**deepcopy(kwargs))

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

    def __init__(self, block, layers, input_c=3, num_classes=10 ,spiking_neuron = None, **kwargs):
        super().__init__()
        inplanes = 64
        self.inplanes = 64
        self.conv1 = Spike_Conv2d(input_c, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = Spike_BatchNorm2d(self.inplanes)
        self.relu = spiking_neuron(**deepcopy(kwargs))
        self.layer1 = self._make_layer(block, inplanes, layers[0],spiking_neuron=spiking_neuron,**deepcopy(kwargs))
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2,spiking_neuron=spiking_neuron,**deepcopy(kwargs))
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2,spiking_neuron=spiking_neuron,**deepcopy(kwargs))
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2,spiking_neuron=spiking_neuron,**deepcopy(kwargs))
        self.avgpool = Spike_AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inplanes * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,spiking_neuron= None,**kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Spike_Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                Spike_BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,spiking_neuron,**deepcopy(kwargs)))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,spiking_neuron=spiking_neuron,**deepcopy(kwargs)))

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
        return x.mean(0)

def resnet18_cifar(neuron,input_c,num_classes,**kwargs):
    model = ResNet_Cifar(BasicBlock, [2, 2, 2, 2],input_c=input_c,num_classes=num_classes,spiking_neuron=neuron, **kwargs)
    return model

if __name__ == '__main__':
    model=resnet18_cifar(neuron=LIF,num_classes=10,input_c=2)
    # print(model)
    # t=torch.randn((4,15,2,32,32))
    # print(model(t).shape)
    print(model)
    #print(list(model.named_modules()['layer4'])[0])
    #
    # for name, module in model.named_modules():
    #     if name == 'layer1.1.relu2':
    #         print(module._get_name())
