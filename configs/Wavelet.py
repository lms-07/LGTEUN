# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Jul
# @Address  : Time Lab @ SDU
# @FileName : Wavelet.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

# A wavelet based algorithm for pan sharpening Landsat 7 imagery, IGARSS 2001

# ---> GENERAL CONFIG <---
name = 'Wavelet'
description = 'test panformer on PSData3/GF-2 dataset'
dataset = ['GF-2', 'WV-2', 'WV-3']
ms_chans_list = [4, 4, 8]
index = 1

device_id = '0'

datas = dataset[index]
ms_chans = ms_chans_list[index]

currentPath = '/media/lms/LMS/Strore_Space_ws/Pan-sharpening'

model_type = 'Wavelet'
work_dir = f'data/model_out/{name}'
log_dir = f'logs/{model_type.lower()}/{datas}'
log_file = f'{log_dir}/{name}.log'
log_level = 'INFO'

only_test = True  #

# ---> DATASET CONFIG <---
aug_dict = {'lr_flip': 0.5, 'ud_flip': 0.5}

bit_depth = 11
test_set0_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=[currentPath + f'/data/PSData3/Dataset/{datas}/test_full_res'],
        bit_depth=bit_depth),
    batch_size=1,
    num_workers=0,
    shuffle=False)
test_set1_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=[currentPath + f'/data/PSData3/Dataset/{datas}/test_reduce_res'],
        bit_depth=bit_depth),
    batch_size=1,
    num_workers=0,
    shuffle=False)
seed = 19971118
cuda = True
max_iter_list = [0, 0, 0]
max_iter = max_iter_list[index]

step_list = [25900, 25300, 22750]
step = step_list[index]

save_freq = 5000
test_freq = 5000
eval_freq = 5000

norm_input = True

# ---> SPECIFIC CONFIG <---
optim_cfg = {
    'core_module': dict(type='Adam', betas=(0.9, 0.999), lr=1.25e-3),
}
sched_cfg = dict(step_size=step, gamma=0.85)

loss_cfg = {
    'rec_loss': dict(type='l1', w=1.)
}
model_cfg = {
}
