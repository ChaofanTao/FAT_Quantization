"""
https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py
"""
import torch
from torch import nn
from torch import Tensor
from torch.utils.model_zoo import load_url
# import sys
# sys.path.append("../")
from .fat_quantization import *
import matplotlib.pyplot as plt

__all__ = ['MobileNetV2', 'mobilenet_v2']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
            self,
            bit,
            in_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            groups=1,
            norm_layer=None,
            activation_layer=None,
            first=False,
    ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        # first conv
        if first:
            super(ConvBNReLU, self).__init__(
                first_conv(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                           padding=padding, groups=groups, bias=False),
                norm_layer(out_planes),
                activation_layer(inplace=True)
            )
        else:
            super(ConvBNReLU, self).__init__(
                UniQuantConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, groups=groups, bias=False, bit=bit,
                               ),
                norm_layer(out_planes),
                activation_layer(inplace=True)
            )


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
            self,
            bit,
            inp,
            oup,
            stride,
            expand_ratio,
            norm_layer=None,

    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvBNReLU(bit, inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(bit, hidden_dim, hidden_dim, stride=stride, groups=hidden_dim,
                       norm_layer=norm_layer),
            # pw-linear
            FAT_QuantConv2d(hidden_dim, oup, kernel_size=1, stride=1,
                            padding=0, bias=False,
                            bit=bit),
            # QuantConv2d(hidden_dim, oup, kernel_size=1, stride=1,padding=0, bias=False,
            #             bit=bit),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
            self,
            bit,
            num_classes=1000,
            width_mult=1.0,
            inverted_residual_setting=None,
            round_nearest=8,
            block=None,
            norm_layer=None
    ):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.bit = bit
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features = [ConvBNReLU(bit, 3, input_channel, stride=2, norm_layer=norm_layer,
                               first=True)]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(bit, input_channel, output_channel, stride, expand_ratio=t,
                          norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(bit, input_channel, self.last_channel, kernel_size=1,
                                   norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            last_fc(self.last_channel, num_classes)
        )

    # weight initialization
    def initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    def show_params(self):
        for m in self.modules():
            if isinstance(m, FAT_QuantConv2d) or isinstance(m, UniQuantConv2d):
                m.show_params()


def mobilenet_v2(bit=4, pretrained=True, progress=True,
                 **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(bit, **kwargs)
    if pretrained:
        state_dict = load_url(model_urls['mobilenet_v2'],
                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def vis_freq_spectrum(model):
    cnt = 0  # number of conv
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            cnt += 1
    plt.figure(figsize=(cnt, 240))

    cur = 0
    for i, (name, m) in enumerate(model.named_modules()):
        # 3conv conv.0.0, conv.1.0, conv.2
        if isinstance(m, FAT_QuantConv2d):
            cur += 1
            weight = m.weight.data
            cout, cin, kh, kw = m.weight.size()

            transform = nn.Linear(cout, cout, bias=False).cuda()
            transform.weight.data.fill_(5 / cout)

            stack_w = torch.stack([weight, torch.zeros_like(weight)], dim=-1)  # [cout,cin,kh,kw,2]
            stack_w = stack_w.view(cout, cin * kh * kw, 2)
            fft_w = torch.fft(stack_w, 1)
            mag_w = torch.norm(fft_w, dim=-1)
            gate = torch.sigmoid(transform(torch.transpose(mag_w, 0, 1))).permute(1, 0)
            learned_ratio = gate.sum() / weight.numel()
            print(name, "->", weight.size(), "->", learned_ratio.item())
        #     plt.imshow(mag_w.cpu().detach().numpy())
        #
        #     plt.title("{}".format(name))
        #     plt.axis('off')

    # plt.savefig('./mobilev2_freq.png')
    # plt.show()


if __name__ == "__main__":
    model = mobilenet_v2(bit=4, pretrained=True)
    model.cuda()

# features.0.0 torch.Size([32, 3, 3, 3])
# features.1.conv.0.0 torch.Size([32, 1, 3, 3])
# features.1.conv.1 torch.Size([16, 32, 1, 1])
# features.2.conv.0.0 torch.Size([96, 16, 1, 1])
# features.2.conv.1.0 torch.Size([96, 1, 3, 3])
# features.5.conv.2 torch.Size([32, 192, 1, 1])
# features.6.conv.0.0 torch.Size([192, 32, 1, 1])
# features.6.conv.1.0 torch.Size([192, 1, 3, 3])
# features.6.conv.2 torch.Size([32, 192, 1, 1])
# features.7.conv.0.0 torch.Size([192, 32, 1, 1])
# features.7.conv.1.0 torch.Size([192, 1, 3, 3])
# features.7.conv.2 torch.Size([64, 192, 1, 1])
# features.8.conv.0.0 torch.Size([384, 64, 1, 1])
# features.8.conv.1.0 torch.Size([384, 1, 3, 3])
# features.8.conv.2 torch.Size([64, 384, 1, 1])
# features.9.conv.0.0 torch.Size([384, 64, 1, 1])
# features.9.conv.1.0 torch.Size([384, 1, 3, 3])
# features.9.conv.2 torch.Size([64, 384, 1, 1])
# features.10.conv.0.0 torch.Size([384, 64, 1, 1])
# features.10.conv.1.0 torch.Size([384, 1, 3, 3])
# features.10.conv.2 torch.Size([64, 384, 1, 1])
# features.11.conv.0.0 torch.Size([384, 64, 1, 1])
# features.11.conv.1.0 torch.Size([384, 1, 3, 3])
# features.11.conv.2 torch.Size([96, 384, 1, 1])
# features.12.conv.0.0 torch.Size([576, 96, 1, 1])
# features.12.conv.1.0 torch.Size([576, 1, 3, 3])
# features.12.conv.2 torch.Size([96, 576, 1, 1])
# features.13.conv.0.0 torch.Size([576, 96, 1, 1])
# features.13.conv.1.0 torch.Size([576, 1, 3, 3])
# features.13.conv.2 torch.Size([96, 576, 1, 1])
# features.14.conv.0.0 torch.Size([576, 96, 1, 1])
# features.14.conv.1.0 torch.Size([576, 1, 3, 3])
# features.14.conv.2 torch.Size([160, 576, 1, 1])
# features.15.conv.0.0 torch.Size([960, 160, 1, 1])
# features.15.conv.1.0 torch.Size([960, 1, 3, 3])
# features.15.conv.2 torch.Size([160, 960, 1, 1])
# features.16.conv.0.0 torch.Size([960, 160, 1, 1])
# features.16.conv.1.0 torch.Size([960, 1, 3, 3])
# features.16.conv.2 torch.Size([160, 960, 1, 1])
# features.17.conv.0.0 torch.Size([960, 160, 1, 1])
# features.17.conv.1.0 torch.Size([960, 1, 3, 3])
# features.17.conv.2 torch.Size([320, 960, 1, 1])
# features.18.0 torch.Size([1280, 320, 1, 1])
