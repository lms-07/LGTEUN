# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Jul
# @Address  : Time Lab @ SDU
# @FileName : MDCUN.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

# Memory-Augmented Deep Conditional Unfolding Network for Pan-Sharpening, CVPR 2022

# ---> GENERAL CONFIG <---
name = 'MDCUN'
description = 'test panformer on PSData3/GF-2 dataset'
dataset = ['GF-2', 'WV-2', 'WV-3']
ms_chans_list = [4, 4, 8]
index = 2

datas = dataset[index]
ms_chans = ms_chans_list[index]

currentPath = '/media/lms/LMS/Strore_Space_ws/Pan-sharpening/PanFormer'

model_type = 'MDCUN'
work_dir = f'data/model_out/{name}'
log_dir = f'logs/{model_type.lower()}'
log_file = f'{log_dir}/{name}.log'
log_level = 'INFO'

only_test = True  #

checkpoint_list = [f'data/PSData3/model_out/{name}/{datas}/model_iter_259000.pth',  # 0
                   f'data/PSData3/model_out/{name}/{datas}/model_iter_253000.pth',  # 1
                   f'data/PSData3/model_out/{name}/{datas}/model_iter_227500.pth']  # 0
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
max_iter_list = [259000, 253000, 227500]
max_iter = max_iter_list[index]  # max inter 1000 epoch

step_list = [51800, 50600, 45500]
step = step_list[index]

save_freq = 10000
test_freq = 10000
eval_freq = 10000

norm_input = True

# ---> SPECIFIC CONFIG <---
optim_cfg = {
    'core_module': dict(type='Adam', betas=(0.9, 0.999), lr=5e-4),
}
sched_cfg = dict(step_size=step, gamma=0.5)

loss_cfg = {
    'rec_loss': dict(type='l1', w=1.)
}

model_cfg = {
    'core_module': dict(),
}
