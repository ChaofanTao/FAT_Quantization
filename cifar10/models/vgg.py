import torch.nn as nn
from collections import OrderedDict
from .fat_quantization import *

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
small_cfg = [128, 128, 'M', 256, 256, 'M', 512, 512, 'M']


# relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]

class VGG(nn.Module):
    def __init__(self, bit, cfg=None, num_classes=10):
        super(VGG, self).__init__()
        self.bit = bit
        if cfg is None:
            cfg = small_cfg
        self.regs = 0.

        # self.compress_rate = compress_rate[:]
        # self.compress_rate.append(0.0)

        self.features = self._make_layers(cfg)
        if cfg == defaultcfg:
            self.classifier = nn.Sequential(OrderedDict([
                ('linear1', last_fc(cfg[-2], cfg[-1])),
                ('norm1', nn.BatchNorm1d(cfg[-1])),
                ('relu1', nn.ReLU(inplace=True)),
                ('linear2', last_fc(cfg[-1], num_classes)),
            ]))
        else:
            self.classifier = nn.Sequential(OrderedDict([
                ('dropout', nn.Dropout(0.5)),
                ('linear2', last_fc(2048, num_classes)),
            ]))

    def _make_layers(self, cfg):

        layers = nn.Sequential()
        in_channels = 3
        cnt = 0

        for i, x in enumerate(cfg):
            if x == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # x = int(x * (1-self.compress_rate[cnt]))
                cnt += 1
                if self.bit == 32:
                    conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                else:
                    if cnt == 1:
                        conv2d = first_conv(in_channels, x, kernel_size=3, padding=1)
                    else:
                        conv2d = FAT_QuantConv2d(in_channels, x, kernel_size=3, padding=1, bit=self.bit)
                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(x))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = x

        return layers

    def forward(self, x):
        self.regs = 0.
        for i, layer in enumerate(self.features):
            x = layer(x)
            if hasattr(layer, "reg"):
                self.regs += layer.reg

        x = nn.AvgPool2d(2)(x)
        # import pdb;pdb.set_trace()
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if hasattr(layer, "reg"):
                self.regs += layer.reg
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, FAT_QuantConv2d):
                m.show_params()


def vgg_7_bn(bit):
    return VGG(cfg=small_cfg, bit=bit)


def vgg_16_bn(bit):
    return VGG(cfg=defaultcfg, bit=bit)
