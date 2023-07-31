# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-May
# @Address  : Time Lab @ SDU
# @FileName : basic_module_unformer_v2.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

# basic sampling modules for our method, LGTEUN

import torch.nn as nn


def point_conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=1)


def dep_conv(in_channels, kernel_size):
    return nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size // 2, groups=in_channels)


def sampling_(x, s_factor, mode_='bicubic'):
    return nn.functional.interpolate(x, scale_factor=s_factor, mode=mode_, align_corners=False,
                                     recompute_scale_factor=False)


class sampling_unit_(nn.Module):
    def __init__(self, s_factor, mode='bicubic'):
        super(sampling_unit_, self).__init__()
        self.s_factor = s_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.s_factor, mode=self.mode, align_corners=False,
                                         recompute_scale_factor=False)


class depthwise_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(depthwise_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.point_conv = point_conv(self.in_channels, self.out_channels)
        self.depth_conv = dep_conv(in_channels=self.out_channels,
                                   kernel_size=self.kernel_size)

    def forward(self, x):
        out = self.point_conv(x)
        out = self.depth_conv(out)

        return out


class span_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(span_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.point_conv_1 = point_conv(self.in_channels, self.out_channels)
        self.depth_conv_1 = dep_conv(in_channels=self.out_channels,
                                     kernel_size=self.kernel_size)

        self.point_conv_2 = point_conv(self.in_channels, self.out_channels)
        self.depth_conv_2 = dep_conv(in_channels=self.out_channels,
                                     kernel_size=self.kernel_size)

    def forward(self, x):
        out1 = self.point_conv_1(x)
        out1 = self.depth_conv_1(out1)

        out2 = self.point_conv_2(x)
        out2 = self.depth_conv_2(out2)

        out = out1 + out2

        return out
