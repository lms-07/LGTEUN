# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Jul
# @Address  : Time Lab @ SDU
# @FileName : SFIIN.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

# Spatial-Frequency Domain Information Integration for Pan-Sharpening, ECCV 2022

# ---> GENERAL CONFIG <---
name = 'SFIIN'
description = 'test panformer on PSData3/GF-2 dataset'
dataset = ['GF-2', 'WV-2', 'WV-3']
ms_chans_list = [4, 4, 8]
index = 2

datas = dataset[index]
ms_chans = ms_chans_list[index]

currentPath = '/media/lms/LMS/Strore_Space_ws/Pan-sharpening'

model_type = 'SFIIN'
work_dir = f'data/model_out/{name}'
log_dir = f'logs/{model_type.lower()}/{datas}'
log_file = f'{log_dir}/{name}.log'
log_level = 'INFO'

only_test = True  #

checkpoint_list = [f'data/PSData3/model_out/{name}/{datas}/model_iter_518000.pth',
                   f'data/PSData3/model_out/{name}/{datas}/model_iter_506000.pth',
                   f'data/PSData3/model_out/{name}/{datas}/model_iter_455000.pth']
checkpoint = checkpoint_list[index]

checkpoint = currentPath + '/' + checkpoint  # for original pretrain.pth

# ---> DATASET CONFIG <---
aug_dict = {'lr_flip': 0.5, 'ud_flip': 0.5}

bit_depth = 11
train_set_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=[currentPath + f'/data/PSData3/Dataset/{datas}/train_reduce_res'],
        bit_depth=bit_depth),
    num_workers=4,
    batch_size=4,
    shuffle=True)
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
max_iter_list = [518000, 506000, 455000]
max_iter = max_iter_list[index]  # max inter 2000 epoch

step_list = [51800, 50600, 45500]
step = step_list[index]

save_freq = 10000
test_freq = 10000
eval_freq = 10000

norm_input = True

# ---> SPECIFIC CONFIG <---

optim_cfg = {
    'core_module': dict(type='Adam', betas=(0.9, 0.999), lr=8e-4),
}
sched_cfg = dict(step_size=step, gamma=0.5)

loss_cfg = {
    'rec_loss': dict(type='l1', w=1.),
    'fre_amp_rec_loss': dict(type='l1', w=0.1),
    'fre_pha_rec_loss': dict(type='l1', w=0.1)
}

model_cfg = {
    'core_module': dict(),
}
