import numpy as np
import logging
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pdb
import math


def build_power_value(B=4):
    base_a = [0.]
    for i in range(2 ** B - 1):
        base_a.append(2 ** (-i - 1))
    values = torch.Tensor(list(set(base_a)))
    values = values.mul(1.0 / torch.max(values))
    return values


def weight_quantization(b, grids, power=False):
    # b is bit-width
    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, value):
        shape = x.shape
        xhard = x.view(-1)
        value = value.type_as(x)
        idxs = (xhard.unsqueeze(0) - value.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
        xhard = value[idxs].view(shape)
        # xout = (xhard - x).detach() + x
        return xhard

    class _weight_q(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)  # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)  # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            if power:
                input_q = power_quant(input_abs, grids).mul(sign)  # project to Q^a(alpha, B)
            else:
                input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)  # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()  # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs() > 1.).float()
            sign = input.sign()
            grad_alpha = (grad_output * (sign * i + (input_q - input) * (1 - i))).sum()
            # grad_input = grad_input * (1 - i)

            # grad_alpha = (grad_output * sign * i ).sum()
            return grad_input, grad_alpha

    return _weight_q().apply


def act_quantization(b, grids, power):
    def uniform_quant(x, b=3):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, grids):
        shape = x.shape
        xhard = x.view(-1)
        value_s = grids.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        xhard = value_s[idxs].view(shape)
        return xhard

    class _act_q(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input = input.div(alpha)
            input_c = input.clamp(max=1)
            if power:
                input_q = power_quant(input_c, grids)
            else:
                input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            grad_alpha = (grad_output * i).sum()
            grad_input = grad_input * (1 - i)
            return grad_input, grad_alpha

    return _act_q().apply


def weight_deactivate_freq(weight, transform=None, fc=False):
    """
    weight:
        for conv layer (without bias), weight is a 4-D tensor with shape [Cout, Cin, H, W]
        for fc layer, weight is a 2-D tensor with shape [Cout, Cin]

    transform:
        The fully-connected layer used for transform the magnitude.
    fc:
        whether a conv layer or a fc layer.
    """

    if fc:
        cout, cin = weight.size()
    else:
        cout, cin, kh, kw = weight.size()

    device = weight.device
    # flatten weight
    reshape_weight = weight.reshape(cout, -1)
    stack_w = torch.stack([reshape_weight, torch.zeros_like(reshape_weight).to(device)], dim=-1)
    # map weight to frequency domain
    fft_w = torch.fft(stack_w, 1)
    # compute the norm in the frequency domain
    mag_w = torch.norm(fft_w, dim=-1)
    assert transform is not None
    freq_score = transform(torch.transpose(mag_w, 0, 1))

    # generate element-wise mask for the weight
    mask = torch.sigmoid(freq_score)
    mask = mask.permute(1, 0)
    restore_ffw = fft_w * mask.unsqueeze(2)

    # map weight back to spatial domain
    restore_w = torch.ifft(restore_ffw, 1)[..., 0]

    if fc:
        restore_w = restore_w.view(cout, cin)
    else:
        restore_w = restore_w.view(cout, cin, kh, kw)
    return restore_w, (mask.max().item(), mask.min().item())


# [optional]   sort the norm in the freq domain, just learn a threshold.
def weight_deactivate_sort(fc=False):
    def get_norm(weight, fc=fc):
        if fc:
            cout, cin = weight.size()
        else:
            cout, cin, kh, kw = weight.size()
        device = weight.device
        reshape_weight = weight.reshape(cout, -1)
        stack_w = torch.stack([reshape_weight, torch.zeros_like(reshape_weight).to(device)], dim=-1)
        fft_w = torch.fft(stack_w, 1)
        mag_w = torch.norm(fft_w, dim=-1)  # [cout, cin*kh*kw]
        return fft_w, mag_w

    class _learn_threshold(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, threshold):
            cout, cin, kh, kw = input.size()
            fft_w, mag_w = get_norm(input)
            one_idx = (mag_w >= threshold).float()
            # zero_idx = mag_w < threshold
            fft_w = fft_w * one_idx.unsqueeze(2)
            restore_w = torch.ifft(fft_w, 1)[..., 0]
            restore_w = restore_w.view(cout, cin, kh, kw)

            one_idx = one_idx.view(cout, cin, kh, kw)
            ctx.save_for_backward(one_idx)
            return restore_w

        @staticmethod
        def backward(ctx, grad_output):
            one_idx = ctx.saved_tensors
            # import pdb;
            # pdb.set_trace()

            grad_input = grad_output * one_idx[0]
            grad_threshold = (1 - one_idx[0]).sum()
            return grad_input, grad_threshold

    return _learn_threshold().apply


# [optional] compute the signal-to-noise-ratio as a regularizer
def compute_snr(weight, weight_q, log=False):
    """
    snr (signal-noise-ratio) is (variance of signal) / (variance of noise)

    Basedon the shnanon formula, C = bit * log(1 + snr) in a unit time, where C is the max speed
    of information transmission in the weight.

    psnr = 10 log_10 ((2**n-1)**2/mse)  (dB) 20~30
    snr = 10 log_10 ((weight**2/mse)  (dB) 20~30
    Therefore, given the bit, the larger C, the more informative the quantizated network.

    sqnr = E[x^2] / E[(Q(x)-x)^2],,

    the sqnr for uniform quantization in [-alpha, alpha] is :
    E[(Q(x)-x)^2]= (interval)^2 / 12,    interval = alpha / (2**(bit-1))
    E[x^2] = (alpha)^2 / 3
    sqnr = (alpha)^2 * 4 / (interval)^2

    our quantizer should have sqnr more than this
    """
    mse = (weight - weight_q).pow(2).mean()
    weight_pow = weight.pow(2).mean()
    return 10 * torch.log10(weight_pow / mse)


class weight_quantize_fn(nn.Module):
    def __init__(self, out_channels, w_bit, power):
        super(weight_quantize_fn, self).__init__()
        assert (w_bit <= 8 and w_bit > 0) or w_bit == 32
        self.w_bit = w_bit - 1
        self.transform = None
        self.transform = nn.Linear(out_channels, out_channels, bias=False)
        self.transform.weight.data.fill_(1 / out_channels)
        self.grids = build_power_value(self.w_bit)
        self.weight_quantizor = weight_quantization(self.w_bit, self.grids, power=power)
        self.register_parameter('wgt_alpha', Parameter(torch.tensor(3.0)))
        # self.register_parameter('threshold', Parameter(torch.tensor(10.0)))
        self.mask = None

    def forward(self, weight):
        if self.w_bit == 32:
            weight_q = weight
        else:
            # weights normalization
            mean = weight.data.mean()
            std = weight.data.std()
            weight = weight.add(-mean).div(std)
            # restore_w = weight
            restore_w, mask_value = weight_deactivate_freq(weight, self.transform)
            self.mask = mask_value  # print(self, learned_ratio)
            weight_q = self.weight_quantizor(restore_w, self.wgt_alpha)
        return weight_q


class uniform_weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, power):
        super(uniform_weight_quantize_fn, self).__init__()
        assert (w_bit <= 8 and w_bit > 0) or w_bit == 32
        self.w_bit = w_bit - 1
        self.grids = build_power_value(self.w_bit)
        self.weight_quantizor = weight_quantization(self.w_bit, self.grids, power=power)
        self.register_parameter('wgt_alpha', Parameter(torch.tensor(3.0)))

    def forward(self, weight):
        if self.w_bit == 32:
            weight_q = weight
        else:
            # weights normalization
            mean = weight.data.mean()
            # std = weight.data.std()
            weight = weight.add(-mean)
            weight_q = self.weight_quantizor(weight, self.wgt_alpha)

        return weight_q


class FAT_QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, bit=8):
        super(FAT_QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              bias)
        self.layer_type = 'FAT_QuantConv2d'
        self.bit = bit
        self.weight_quant = weight_quantize_fn(out_channels, self.bit, power=False)  # quantization module
        self.act_grid = build_power_value(self.bit)
        self.act_quant = act_quantization(self.bit, self.act_grid, power=False)  # quantization module
        self.act_alpha = torch.nn.Parameter(torch.tensor(8.0))

    def forward(self, x):
        weight_q = self.weight_quant(self.weight)
        xq = self.act_quant(x, self.act_alpha)
        return F.conv2d(xq, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def show_params(self):
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        weight_q = self.weight_quant(self.weight)
        self.mask = self.weight_quant.mask
        act_alpha = round(self.act_alpha.data.item(), 3)
        logging.info('frequency mask, max: {:.3f}, min: {:.3f};  weight alpha: {:.3f}, activation alpha: {:.3f}'.format(
            self.mask[0], self.mask[1], wgt_alpha, act_alpha))


class UniQuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, bit=8):
        super(UniQuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             bias)
        self.layer_type = 'UniQuantConv2d'
        self.bit = bit
        self.weight_quant = uniform_weight_quantize_fn(self.bit, power=False)  # quantization module
        self.act_grid = build_power_value(self.bit)
        self.act_quant = act_quantization(self.bit, self.act_grid, power=False)
        self.act_alpha = torch.nn.Parameter(torch.tensor(5.0))

    def forward(self, x):
        weight_q = self.weight_quant(self.weight)
        xq = self.act_quant(x, self.act_alpha)
        return F.conv2d(xq, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def show_params(self):
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        act_alpha = round(self.act_alpha.data.item(), 3)
        logging.info('weight alpha : {:2f}  , activation alpha: {:2f}'.format(
            wgt_alpha, act_alpha))


# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                         bias)
        self.layer_type = 'FConv2d'
        self.transform = None
        self.transform = nn.Linear(out_channels, out_channels, bias=False)
        self.transform.weight.data.fill_(1 / out_channels)

    def forward(self, x):
        restore_w = self.weight
        max = restore_w.data.max()
        weight_q = restore_w.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - restore_w).detach() + restore_w
        # self.reg = compute_snr(restore_w, weight_q)

        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'
        self.transform = nn.Linear(out_features, out_features, bias=False)
        self.transform.weight.data.fill_(1 / out_features)

    def forward(self, x):
        restore_w = self.weight
        max = restore_w.data.max()
        weight_q = restore_w.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - restore_w).detach() + restore_w

        return F.linear(x, weight_q, self.bias)
