# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Jul
# @Address  : Time Lab @ SDU
# @FileName : SFIM.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

# https://github.com/manman1995/Awaresome-pansharpening/blob/main/py-tra/methods/SFIM.py
# Smoothing Filter-based Intensity Modulation: A spectral preserve image fusion technique for improving spatial details, IJRS 2000


import numpy as np

from scipy import signal
from models.common.model_based_utils import upsample_interp23

from .base.builder import MODELS
from .base.base_model import Base_model


def SFIM_(hs, pan):
    pan = pan.squeeze(0).cpu().detach().numpy()
    hs = hs.squeeze(0).cpu().detach().numpy()

    pan, hs = pan.transpose(1, 2, 0), hs.transpose(1, 2, 0)

    M, N, c = pan.shape
    m, n, C = hs.shape

    ratio = int(np.round(M / m))

    assert int(np.round(M / m)) == int(np.round(N / n))

    # upsample
    u_hs = upsample_interp23(hs, ratio)

    if np.mod(ratio, 2) == 0:
        ratio = ratio + 1

    pan = np.tile(pan, (1, 1, C))

    pan = (pan - np.mean(pan, axis=(0, 1))) * (
            np.std(u_hs, axis=(0, 1), ddof=1) / np.std(pan, axis=(0, 1), ddof=1)) + np.mean(u_hs, axis=(0, 1))

    kernel = np.ones((ratio, ratio))
    kernel = kernel / np.sum(kernel)

    I_SFIM = np.zeros((M, N, C))
    for i in range(C):
        lrpan = signal.convolve2d(pan[:, :, i], kernel, mode='same', boundary='wrap')
        I_SFIM[:, :, i] = u_hs[:, :, i] * pan[:, :, i] / (lrpan + 1e-8)

    I_SFIM[I_SFIM < 0] = 0
    I_SFIM[I_SFIM > 1] = 1

    I_SFIM = np.expand_dims(I_SFIM, axis=0)

    return I_SFIM


@MODELS.register_module()
class SFIM(Base_model):
    def __init__(self, cfg, logger, train_data_loader, test_data_loader0, test_data_loader1):
        super().__init__(cfg, logger, train_data_loader, test_data_loader0, test_data_loader1)

    def get_model_output(self, input_batch):
        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']
        output = SFIM_(input_lr, input_pan)
        return output
