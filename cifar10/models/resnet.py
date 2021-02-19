import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fat_quantization import *


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def FAT_Quantconv3x3(in_planes, out_planes, stride=1, bit=8):
    " 3x3 quantized convolution with padding"
    return FAT_QuantConv2d(in_planes, out_planes, kernel_size=3,
                           bit=bit,
                           stride=stride, padding=1,
                           bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bit=8):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.regs = 0.
        self.bit = bit
        if bit == 32:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv1 = FAT_Quantconv3x3(inplanes, planes, stride, bit=bit, )
            self.conv2 = FAT_Quantconv3x3(planes, planes, bit=bit, )

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # self.shortcut = nn.Sequential()
        # if stride != 1 or inplanes != planes:
        #     if stride != 1:
        #         self.shortcut = LambdaLayer(
        #             lambda x: F.pad(x[:, :, ::2, ::2],
        #                             (
        #                             0, 0, 0, 0, (planes - inplanes) // 2, planes - inplanes - (planes - inplanes) // 2),
        #                             "constant", 0))
        #     else:
        #         self.shortcut = LambdaLayer(
        #             lambda x: F.pad(x[:, :, :, :],
        #                             (
        #                             0, 0, 0, 0, (planes - inplanes) // 2, planes - inplanes - (planes - inplanes) // 2),
        #                             "constant", 0))

    def forward(self, x):
        self.regs = 0.
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
        if self.bit != 32:
            self.regs = self.conv1.reg + self.conv2.reg
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_layers, bit, num_classes=10):
        super(ResNet, self).__init__()
        self.regs = 0.
        self.inplanes = 16
        self.bit = bit

        self.layer_num = 0
        self.conv1 = first_conv(3, 16, kernel_size=3, stride=1,
                                padding=1, bias=False, )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList()
        self.layer_num += 1

        # self.layers = nn.ModuleList()
        self.layer1 = self._make_layer(block, 16, blocks_num=num_layers[0], stride=1, bit=bit, )
        self.layer2 = self._make_layer(block, 32, blocks_num=num_layers[1], stride=2, bit=bit, )
        self.layer3 = self._make_layer(block, 64, blocks_num=num_layers[2], stride=2, bit=bit, )

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = last_fc(64 * BasicBlock.expansion, num_classes, )

    def _make_layer(self, block, planes, blocks_num, stride, bit):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                FAT_QuantConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bit=bit,
                                )
                if bit != 32 else nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                                            stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, bit))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks_num):
            layers.append(block(self.inplanes, planes, 1, None, bit))

        return nn.Sequential(*layers)

    def forward(self, x):
        # import pdb;
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for i, block in enumerate(self.layer1):
            x = block(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        for i, block in enumerate(self.layer3):
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, FAT_QuantConv2d):
                m.show_params()


def resnet_20(bit):
    return ResNet(BasicBlock, [3, 3, 3],
                  bit=bit, )


def resnet_32(bit):
    return ResNet(BasicBlock, [5, 5, 5],
                  bit=bit, )


def resnet_44(bit):
    return ResNet(BasicBlock, [7, 7, 7],
                  bit=bit, )


def resnet_56(bit):
    return ResNet(BasicBlock, [9, 9, 9],
                  bit=bit, )


def resnet_110(bit):
    return ResNet(BasicBlock, [18, 18, 18],
                  bit=bit, )


if __name__ == "__main__":
    x = torch.randn(10, 3, 32, 32)
    label = torch.randint(0, 5, (10,))
    print(label)
    net = resnet_20(4)
    x = x.cuda()
    label = label.cuda()
    net = net.cuda()
    # for n,p in net.named_parameters():
    #     print(n)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    net.train()
    for _ in range(100):
        o = net(x)
        loss = nn.CrossEntropyLoss()(o, label)
        print(net.regs, loss)
        loss.backward()
        optimizer.step()
