# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Jul
# @Address  : Time Lab @ SDU
# @FileName : SFIM.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

# Smoothing Filter-based Intensity Modulation: A spectral preserve image fusion technique for improving spatial details, IJRS 2000


# ---> GENERAL CONFIG <---
name = 'SFIM'
description = 'test panformer on PSData3/GF-2 dataset'
dataset = ['GF-2', 'WV-2', 'WV-3']
ms_chans_list = [4, 4, 8]
index = 1

device_id = '0'

datas = dataset[index]
ms_chans = ms_chans_list[index]

currentPath = '/media/lms/LMS/Strore_Space_ws/Pan-sharpening/PanFormer'

model_type = 'SFIM'
work_dir = f'data/model_out/{name}'
log_dir = f'logs/{model_type.lower()}/{datas}'
log_file = f'{log_dir}/{name}.log'
log_level = 'INFO'

# only_test = False  #
only_test = True #
# checkpoint = currentPath + '/' + checkpoint # for original pretrain.pth

# ---> DATASET CONFIG <---
aug_dict = {'lr_flip': 0.5, 'ud_flip': 0.5}

bit_depth = 11
# train_set_cfg = dict(
#     dataset=dict(
#         type='PSDataset',
#         image_dirs=[currentPath + f'/data/PSData3/Dataset/{datas}/train_reduce_res'],
#         # image_dirs=[currentPath +f'/data/PSData3/Dataset/{datas}/original/train_low_res'],
#         bit_depth=bit_depth),
#     num_workers=4,
#     # num_workers=0,
#     batch_size=4,
#     # batch_size=64,
#     # shuffle=True)
#     shuffle=False)
test_set0_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=[currentPath + f'/data/PSData3/Dataset/{datas}/test_full_res'],
        # image_dirs=[currentPath +f'/data/PSData3/Dataset/{datas}/original/test_full_res'],
        bit_depth=bit_depth),
    # num_workers=4,
    batch_size=1,
    num_workers=0,
    # batch_size=64,
    # num_workers=0,
    # batch_size=12,
    shuffle=False)
test_set1_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=[currentPath + f'/data/PSData3/Dataset/{datas}/test_reduce_res'],
        # image_dirs=[currentPath +f'/data/PSData3/Dataset/{datas}/original/test_low_res'],
        bit_depth=bit_depth),
    # num_workers=4,
    batch_size=1,
    num_workers=0,
    # batch_size=64,
    shuffle=False)
seed = 19971118
cuda = True
max_iter_list = [0, 0, 0]
# max_iter_list=[0,100000,20000]
# max_iter_list=[0,0,45000]
# max_iter_list = [0, 0, 30000]
max_iter = max_iter_list[index]  # max inter 1000 epoch
# max_iter = 200 # 7000, 15000
# max_iter = 200 # 7000, 15000
# max_iter = 20 # 7000, 15000
step_list = [25900, 25300, 22750]
step = step_list[index]

save_freq = 5000
test_freq = 5000
eval_freq = 5000
# save_freq = 10000
# test_freq = 10000
# eval_freq = 10000

# save_freq = 20
# test_freq = 20
# eval_freq = 20
# norm_input = False
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
    # 'core_module': dict(stage=2),
}
