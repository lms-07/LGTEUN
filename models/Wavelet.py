# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Jul
# @Address  : Time Lab @ SDU
# @FileName : Wavelet.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

# https://github.com/manman1995/Awaresome-pansharpening/blob/main/py-tra/methods/Wavelet.py
# A wavelet based algorithm for pan sharpening Landsat 7 imagery, IGARSS 2001


import pywt
import numpy as np

from models.common.model_based_utils import upsample_interp23

from .base.builder import MODELS
from .base.base_model import Base_model


def Wavelet_(hs, pan):
    pan = pan.squeeze(0).cpu().detach().numpy()
    hs = hs.squeeze(0).cpu().detach().numpy()

    pan, hs = pan.transpose(1, 2, 0), hs.transpose(1, 2, 0)

    M, N, c = pan.shape
    m, n, C = hs.shape

    ratio = int(np.round(M / m))

    assert int(np.round(M / m)) == int(np.round(N / n))

    # upsample
    u_hs = upsample_interp23(hs, ratio)

    pan = np.squeeze(pan)
    pc = pywt.wavedec2(pan, 'haar', level=2)

    rec = []
    for i in range(C):
        temp_dec = pywt.wavedec2(u_hs[:, :, i], 'haar', level=2)

        pc[0] = temp_dec[0]

        temp_rec = pywt.waverec2(pc, 'haar')
        temp_rec = np.expand_dims(temp_rec, -1)
        rec.append(temp_rec)

    I_Wavelet = np.concatenate(rec, axis=-1)

    # adjustment
    I_Wavelet[I_Wavelet < 0] = 0
    I_Wavelet[I_Wavelet > 1] = 1

    I_Wavelet = np.expand_dims(I_Wavelet, axis=0)

    return I_Wavelet


@MODELS.register_module()
class Wavelet(Base_model):
    def __init__(self, cfg, logger, train_data_loader, test_data_loader0, test_data_loader1):
        super().__init__(cfg, logger, train_data_loader, test_data_loader0, test_data_loader1)

    def get_model_output(self, input_batch):
        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']
        output = Wavelet_(input_lr, input_pan)
        return output
