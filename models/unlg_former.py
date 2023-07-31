# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-May
# @Address  : Time Lab @ SDU
# @FileName : unlg_former.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

# Local-Global Transformer Enhanced Unfolding Network for Pan-sharpening, IJCAI 2023

import torch
import torch.nn as nn

import models.common.basic_module_unformer_v2 as bmu
import models.common.LGT as LGT

from .base.builder import MODELS
from .base.base_model import Base_model



class Pansharpening(nn.Module):
    def __init__(self, cfg, logger, stage=5):
        super(Pansharpening, self).__init__()
        self.in_channels = cfg.ms_chans  # spectral bands of MS
        self.stage = stage
        self.up_factor = 4

        # spatial domain
        self.D = nn.Sequential(bmu.sampling_unit_(s_factor=1 / 2), bmu.dep_conv(self.in_channels, 3),
                               bmu.sampling_unit_(s_factor=1 / 2), bmu.dep_conv(self.in_channels, 3))  # down_sampling

        self.DT = nn.Sequential(bmu.sampling_unit_(s_factor=2), bmu.dep_conv(self.in_channels, 3),
                                bmu.sampling_unit_(s_factor=2), bmu.dep_conv(self.in_channels, 3))  # up_sampling

        # spectral domain
        self.R = bmu.point_conv(self.in_channels, 1)  # HrMS Z->Pan P
        self.RT = bmu.point_conv(1, self.in_channels)  # Pan P-> HrMS Z

        # para
        self.eta = nn.ParameterList([nn.Parameter(torch.tensor(0.1), requires_grad=True) for _ in range(self.stage)])

        self.prior_module = nn.ModuleList([])

        for _ in range(self.stage):
            ratio = 0
            self.prior_module.append(
                LGT.LGT(in_channels=self.in_channels, embed_channels=self.in_channels * 4, patch_size=1, window_size=8,
                        num_block=[2, 1], num_heads=2, channel_ratio=ratio))

    def forward(self, ms, pan):
        outs_list = []

        Z = bmu.sampling_(ms, s_factor=4)
        outs_list.append(Z)

        for i in range(self.stage):
            # data module
            ms_term = self.DT(self.D(Z) - ms)
            pan_term = self.RT(self.R(Z) - pan)

            Z = Z - self.eta[i] * (ms_term + pan_term)

            Z_ = self.prior_module[i](Z)

            outs_list.append(Z_)

        return outs_list[-1]


@MODELS.register_module()
class UnlgFormer(Base_model):
    def __init__(self, cfg, logger, train_data_loader, test_data_loader0, test_data_loader1):
        super().__init__(cfg, logger, train_data_loader, test_data_loader0, test_data_loader1)

        model_cfg = cfg.get('model_cfg', dict())
        G_cfg = model_cfg.get('core_module', dict())

        self.add_module('core_module', Pansharpening(cfg=cfg, logger=logger, **G_cfg))

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
