# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Jul
# @Address  : Time Lab @ SDU
# @FileName : lightnet.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

# https://github.com/zhi-xuan-chen/IJCAI-2022_SpanConv/blob/main/codes/model_wv3.py
# SpanConv: A New Convolution via Spanning Kernel Space for Lightweight Pansharpening, IJCAI 2022

import torch
import torch.nn as nn

from .base.builder import MODELS
from .base.base_model import Base_model


# --------------------------------SpanConv Block -----------------------------------#
class SpanConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpanConv, self).__init__()
        self.in_planes = in_channels
        self.out_planes = out_channels
        self.kernel_size = kernel_size

        self.point_wise_1 = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      bias=True)

        self.depth_wise_1 = nn.Conv2d(in_channels=out_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding=(kernel_size - 1) // 2,
                                      groups=out_channels,
                                      bias=True)

        self.point_wise_2 = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=1,
                                      bias=True)

        self.depth_wise_2 = nn.Conv2d(in_channels=out_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=1,
                                      padding=(kernel_size - 1) // 2,
                                      groups=out_channels,
                                      bias=True)

    def forward(self, x):  #
        out_tmp_1 = self.point_wise_1(x)  #
        out_tmp_1 = self.depth_wise_1(out_tmp_1)  #

        out_tmp_2 = self.point_wise_2(x)  #
        out_tmp_2 = self.depth_wise_2(out_tmp_2)  #

        out = out_tmp_1 + out_tmp_2

        return out


# --------------------------------Belly Block -----------------------------------#
class Belly_Block(nn.Module):
    def __init__(self, in_planes):
        super(Belly_Block, self).__init__()
        self.conv1 = SpanConv(in_planes, in_planes, 3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = SpanConv(in_planes, in_planes, 3)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu1(res)
        res = self.conv2(res)
        return res


class LightNet(nn.Module):
    def __init__(self, cfg, logger):
        super(LightNet, self).__init__()
        self.in_channels = cfg.ms_chans
        self.channels = self.in_channels + 1

        self.head_conv = nn.Sequential(
            SpanConv(self.channels, self.channels, 3),
            SpanConv(self.channels, 20, 3),
            # nn.Conv2d(9,32,3,1,1),
            SpanConv(20, 32, 3),
            nn.ReLU(inplace=True)
        )

        self.belly_conv = nn.Sequential(
            Belly_Block(32),
            Belly_Block(32)

        )

        self.tail_conv = nn.Sequential(
            # nn.Conv2d(32,8,3,1,1),
            SpanConv(32, 16, 3),
            SpanConv(16, 8, 3),
            SpanConv(8, self.in_channels, 3)
        )

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, lms, pan):
        lms = nn.functional.interpolate(nn.functional.interpolate(
            lms, scale_factor=2, mode='bicubic', align_corners=False,
            recompute_scale_factor=False), scale_factor=2, mode='bicubic',
            align_corners=False, recompute_scale_factor=False)
        x = torch.cat([pan, lms], 1)
        x = self.head_conv(x)
        x = self.belly_conv(x)
        x = self.tail_conv(x)
        sr = lms + x
        return sr


@MODELS.register_module()
class lightnet(Base_model):
    def __init__(self, cfg, logger, train_data_loader, test_data_loader0, test_data_loader1):
        super().__init__(cfg, logger, train_data_loader, test_data_loader0, test_data_loader1)

        model_cfg = cfg.get('model_cfg', dict())
        G_cfg = model_cfg.get('core_module', dict())

        self.add_module('core_module', LightNet(cfg=cfg, logger=logger, **G_cfg))

    def get_model_output(self, input_batch):
        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']
        output = self.module_dict['core_module'](input_lr, input_pan)
        return output

    def train_iter(self, iter_id, input_batch, log_freq=10):
        G = self.module_dict['core_module']
        G_optim = self.optim_dict['core_module']

        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']

        output = G(input_lr, input_pan)

        loss_g = 0
        loss_res = dict()
        loss_cfg = self.cfg.get('loss_cfg', {})
        if 'rec_loss' in self.loss_module:
            target = input_batch['target']
            rec_loss = self.loss_module['rec_loss'](
                out=output, gt=target
            )
            loss_g = loss_g + rec_loss * loss_cfg['rec_loss'].w
            loss_res['rec_loss'] = rec_loss.item()

        loss_res['full_loss'] = loss_g.item()

        G_optim.zero_grad()
        loss_g.backward()
        G_optim.step()

        self.print_train_log(iter_id, loss_res, log_freq)
