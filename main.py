# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-May
# @Address  : Time Lab @ SDU
# @FileName : main.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

import mmcv
import torch
import random
import argparse
import traceback
import numpy as np

from mmcv import Config
from logging import Logger
from mmcv.utils import get_logger
from torch.utils.data import DataLoader

from dataset.builder import build_dataset
from models.base.builder import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='pan-sharpening implementation')

    # parser.add_argument('-c', '--config',  default="./configs/GSA.py", help='config file path')
    # parser.add_argument('-c', '--config',  default="./configs/SFIM.py", help='config file path')
    # parser.add_argument('-c', '--config',  default="./configs/Wavelet.py", help='config file path')

    # parser.add_argument('-c', '--config', default="./configs/PanFormer.py", help='config file path')
    # parser.add_argument('-c', '--config',  default="./configs/INNT.py", help='config file path')
    # parser.add_argument('-c', '--config',  default="./configs/lightnet.py", help='config file path')
    # parser.add_argument('-c', '--config',  default="./configs/SFIIN.py", help='config file path')
    # parser.add_argument('-c', '--config',  default="./configs/MutInf.py", help='config file path')
    # parser.add_argument('-c', '--config',  default="./configs/MDCUN.py", help='config file path')
    parser.add_argument('-c', '--config', default="./configs/unlg_former.py", help='config file path')

    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(cfg, logger):
    # type: (mmcv.Config, Logger) -> None

    # Setting Random Seed
    if 'seed' in cfg:
        logger.info('===> Setting Random Seed')
        set_random_seed(cfg.seed, True)

    # Loading Datasets
    logger.info('===> Loading Datasets')
    if 'train_set_cfg' in cfg:
        train_set_cfg = cfg.train_set_cfg.copy()
        train_set_cfg['dataset'] = build_dataset(cfg.train_set_cfg['dataset'])
        train_data_loader = DataLoader(**train_set_cfg)
    else:
        train_data_loader = None

    # data set for full-resolution test
    test_set0_cfg = cfg.test_set0_cfg.copy()
    test_set0_cfg['dataset'] = build_dataset(cfg.test_set0_cfg['dataset'])
    test_data_loader0 = DataLoader(**test_set0_cfg)

    # data set for reduced-resolution test
    test_set1_cfg = cfg.test_set1_cfg.copy()
    test_set1_cfg['dataset'] = build_dataset(cfg.test_set1_cfg['dataset'])
    test_data_loader1 = DataLoader(**test_set1_cfg)

    # Building Model
    logger.info('===> Building Model')
    runner = build_model(cfg.model_type, cfg, logger, train_data_loader, test_data_loader0, test_data_loader1)
    # if cfg.dataset=='WV-3':
    #     runner = build_model(cfg.model_type, cfg, logger, train_data_loader, test_data_loader0, test_data_loader1)
    # else:
    #     runner = build_model(cfg.model_type, cfg, logger, train_data_loader, test_data_loader1, test_data_loader1)

    # Setting GPU
    if 'cuda' in cfg and cfg.cuda:
        logger.info("===> Setting GPU")
        runner.set_cuda()

    # Weight Initialization
    if 'checkpoint' not in cfg:
        logger.info("===> Weight Initializing")
        runner.init()

    #  Resume from a Checkpoint (Optionally)
    if 'checkpoint' in cfg:
        logger.info("===> Loading Checkpoint")
        runner.load_checkpoint(cfg.checkpoint)

    # Copy Weights from a Checkpoint (Optionally)
    if 'pretrained' in cfg:
        logger.info("===> Loading Pretrained")
        runner.load_pretrained(cfg.pretrained)

    # Setting Optimizer
    logger.info("===> Setting Optimizer")
    runner.set_optim()

    # Setting Scheduler for learning_rate Decay
    logger.info("===> Setting Scheduler")
    runner.set_sched()

    # Print Params Count
    logger.info("===> Params Count")
    runner.print_total_params()
    runner.print_total_trainable_params()

    if ('only_test' not in cfg) or (not cfg.only_test):
        # Training
        logger.info("===> Training Start")
        runner.train()

        # Saving
        logger.info("===> Final Saving Weights")
        runner.save(iter_id=cfg.max_iter)

    # Testing
    logger.info("===> Final Testing")
    runner.test(iter_id=cfg.max_iter, save=True, ref=True)  # low-resolution testing
    # runner.test(iter_id=cfg.max_iter, save=True, ref=False)  # full-resolution testing

    logger.info("===> Finish !!!")


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    mmcv.mkdir_or_exist(cfg.log_dir)
    logger = get_logger('mmFusion', cfg.log_file, cfg.log_level)
    logger.info(f'Config:\n{cfg.pretty_text}')

    try:
        main(cfg, logger)
    except:
        logger.error(str(traceback.format_exc()))
