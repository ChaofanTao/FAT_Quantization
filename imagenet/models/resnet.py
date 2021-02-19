import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
from .fat_quantization import *
import matplotlib.pyplot as plt

__all__ = ['ResNet', 'resnet_18', 'resnet_34', 'resnet_50', 'resnet_101',
           'resnet_152', ]

model_urls = {
    'resnet_18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet_34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet_50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet_101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet_152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,
            bit=4):
    """3x3 convolution with padding"""
    return FAT_QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=dilation, groups=groups, bias=False, dilation=dilation,
                           bit=bit, )


def conv1x1(in_planes, out_planes, stride=1,
            bit=4):
    """1x1 convolution"""
    return FAT_QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                           bit=bit, )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 bit=4):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,
                             bit=bit)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             bit=bit)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 bit=4):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width,
                             bit=bit)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation,
                             bit=bit)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion,
                             bit=bit)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, bit=4):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.bit = bit

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = first_conv(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,
                                       dilate=False,
                                       bit=bit)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       bit=bit)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       bit=bit)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       bit=bit)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = last_fc(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    bit=4):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride,
                        bit=bit),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            bit=bit))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                bit=bit))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, FAT_QuantConv2d):
                m.show_params()


def _resnet(arch, block, layers, bit):
    model = ResNet(block, layers, bit=bit)
    # default setting: use the torchvision pretrained weights for resnet.
    state_dict = load_url(model_urls[arch])
    model.load_state_dict(state_dict, strict=False)
    return model


def resnet_18(bit=4):
    return _resnet('resnet_18', BasicBlock, [2, 2, 2, 2],
                   bit)


def resnet_34(bit=4):
    return _resnet('resnet_34', BasicBlock, [3, 4, 6, 3],
                   bit)


def resnet_50(bit=4):
    return _resnet('resnet_50', Bottleneck, [3, 4, 6, 3],
                   bit)


def resnet_101(bit=4):
    return _resnet('resnet_101', BasicBlock, [3, 4, 23, 3],
                   bit)


def resnet_152(bit=4):
    return _resnet('resnet_152', Bottleneck, [3, 8, 36, 3],
                   bit)


def vis_freq_spectrum(model):
    cnt = 0  # number of conv
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            cnt += 1
    plt.figure(figsize=(cnt, 100))
    cur = 0
    for i, (name, m) in enumerate(model.named_modules()):
        if isinstance(m, nn.Conv2d):
            cur += 1
            weight = m.weight.data
            cout, cin, kh, kw = m.weight.size()
            # print(name, weight.size(),)

            stack_w = torch.stack([weight, torch.zeros_like(weight)], dim=-1)  # [cout,cin,kh,kw,2]
            stack_w = stack_w.view(cout, cin * kh * kw, 2)
            fft_w = torch.fft(stack_w, 1)
            mag_x = torch.norm(fft_w, dim=-1)
            plt.subplot(cnt, 1, cur)
            # plt.imshow(mag_x.cpu().detach().numpy())
            plt.hist(weight.view(cout * cin * kh * kw).cpu().detach().numpy(), bins=30)
            plt.title("{}".format(name))
            # plt.axis('off')

    plt.savefig('./res18_weight.png')
    plt.show()


# do not add regularization term, until training quantized model to convergence.
if __name__ == "__main__":
    x = torch.randn(10, 3, 224, 224)
    label = torch.randint(0, 5, (10,))
    model = resnet_18(bit=4)
    x = x.cuda()
    label = label.cuda()
    model = model.cuda()
    vis_freq_spectrum(model)

    # for n,p in net.named_parameters():
    #     print(n)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #
    # model.train()
    # for _ in range(100):
    #     o = model(x)
    #     loss = nn.CrossEntropyLoss()(o, label)
    #     print(loss)
    #     loss.backward()
    #     optimizer.step()

# resnet-18
# conv1 torch.Size([64, 3, 7, 7])
# layer1.0.conv1 torch.Size([64, 64, 3, 3])
# layer1.0.conv2 torch.Size([64, 64, 3, 3])
# layer1.1.conv1 torch.Size([64, 64, 3, 3])
# layer1.1.conv2 torch.Size([64, 64, 3, 3])
# layer2.0.conv1 torch.Size([128, 64, 3, 3])
# layer2.0.conv2 torch.Size([128, 128, 3, 3])
# layer2.0.downsample.0 torch.Size([128, 64, 1, 1])
# layer2.1.conv1 torch.Size([128, 128, 3, 3])
# layer2.1.conv2 torch.Size([128, 128, 3, 3])
# layer3.0.conv1 torch.Size([256, 128, 3, 3])
# layer3.0.conv2 torch.Size([256, 256, 3, 3])
# layer3.0.downsample.0 torch.Size([256, 128, 1, 1])
# layer3.1.conv1 torch.Size([256, 256, 3, 3])
# layer3.1.conv2 torch.Size([256, 256, 3, 3])
# layer4.0.conv1 torch.Size([512, 256, 3, 3])
# layer4.0.conv2 torch.Size([512, 512, 3, 3])
# layer4.0.downsample.0 torch.Size([512, 256, 1, 1])
# layer4.1.conv1 torch.Size([512, 512, 3, 3])
# layer4.1.conv2 torch.Size([512, 512, 3, 3])
