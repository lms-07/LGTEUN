# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Jul
# @Address  : Time Lab @ SDU
# @FileName : losses.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mmcv import Config
from logging import Logger

from models.base.metrics import D_lambda_torch, D_s_torch
from models.base.utils import down_sample


class ReconstructionLoss(nn.Module):
    def __init__(self, cfg, logger, loss_type='l1'):
        # type: (Config, Logger, str) -> None
        r"""
            loss_type in ['l1', 'l2']
        """
        super(ReconstructionLoss, self).__init__()
        self.cfg = cfg
        self.loss_type = loss_type
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            logger.error(f'No such type of ReconstructionLoss: \"{loss_type}\"')
            raise SystemExit(f'No such type of ReconstructionLoss: \"{loss_type}\"')

    def get_type(self):
        return self.loss_type

    def forward(self, out, gt):
        return self.loss(out, gt)


class AdversarialLoss(nn.Module):
    def __init__(self, cfg, logger, cuda, gan_type='GAN'):
        # type: (Config, Logger, bool, str) -> None
        r"""
            cfg.soft_label: whether or not use soft label in LSGAN
                Default: False
            cfg.gp_w: the weight of gradient penalty in WGAN-GP
                Default: 10
            gan_type: in ['GAN', 'LSGAN', 'WGAN-GP']
        """
        super(AdversarialLoss, self).__init__()
        self.cfg = cfg
        self.gan_type = gan_type
        self.device = torch.device('cuda' if cuda else 'cpu')
        if gan_type not in ['GAN', 'LSGAN', 'WGAN-GP']:
            logger.error(f'No such type of GAN: \"{gan_type}\"')
            raise SystemExit(f'No such type of GAN: \"{gan_type}\"')
        if gan_type == 'GAN':
            self.bce_loss = nn.BCELoss().to(self.device)
        if gan_type == 'LSGAN':
            self.mse_loss = nn.MSELoss().to(self.device)

    def get_type(self):
        return self.gan_type

    def forward(self, fake, real, D, D_optim):
        r""" calculate the loss of D and G, the optim of D has been done

        Args:
            fake (torch.Tensor): fake input
            real (torch.Tensor): real input
            D (nn.Module): Discriminator
            D_optim (optim.Optimizer): optim of D
        Returns:
            (torch.Tensor, torch.Tensor): loss of G, loss of D
        """
        fake_detach = fake.detach()
        real_detach = real.detach()

        D_optim.zero_grad()
        # calculate d_loss
        d_fake = D(fake_detach)
        d_real = D(real_detach)
        if self.gan_type == 'GAN':
            valid_score = torch.ones(d_real.shape).to(self.device)
            fake_score = torch.zeros(d_fake.shape).to(self.device)
            real_loss = self.bce_loss(torch.sigmoid(d_real), fake_score)
            fake_loss = self.bce_loss(torch.sigmoid(d_fake), valid_score)
            loss_d = - (real_loss + fake_loss)
        elif self.gan_type == 'LSGAN':
            soft_label = self.cfg.get('soft_label', False)
            if not soft_label:
                valid_score = torch.ones(d_real.shape).to(self.device)
                fake_score = torch.zeros(d_fake.shape).to(self.device)
            else:
                valid_score = .7 + np.float32(np.random.rand(1)) * .5  # rand in [0.7, 1.2]
                fake_score = .0 + np.float32(np.random.rand(1)) * .3  # rand in [0, 0.3]
                valid_score = torch.ones(d_real.shape) * valid_score
                fake_score = torch.ones(d_real.shape) * fake_score
                valid_score = valid_score.to(self.device)
                fake_score = fake_score.to(self.device)
            real_loss = self.mse_loss(d_real, valid_score)
            fake_loss = self.mse_loss(d_fake, fake_score)
            loss_d = (real_loss + fake_loss) / 2.
        elif self.gan_type == 'WGAN-GP':
            gp_w = self.cfg.get('gp_w', 10)

            loss_d = (d_fake - d_real).mean()
            epsilon = torch.rand(real_detach.size(0), 1, 1, 1).to(self.device)
            epsilon = epsilon.expand(real_detach.size())
            hat = fake_detach.mul(1 - epsilon) + real_detach.mul(epsilon)
            hat.requires_grad = True
            d_hat = D(hat)
            gradients = torch.autograd.grad(
                outputs=d_hat.sum(), inputs=hat,
                retain_graph=True, create_graph=True, only_inputs=True
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_norm = gradients.norm(2, dim=1)
            gradient_penalty = gp_w * gradient_norm.sub(1).pow(2).mean()
            loss_d = loss_d + gradient_penalty

        # Discriminator update
        loss_d.backward()
        D_optim.step()

        # calculate g_loss
        d_fake_for_g = D(fake)
        if self.gan_type == 'GAN':
            loss_g = self.bce_loss(torch.sigmoid(d_fake_for_g), valid_score)
        elif self.gan_type == 'LSGAN':
            loss_g = self.mse_loss(d_fake_for_g, valid_score)
        elif self.gan_type == 'WGAN-GP':
            loss_g = -d_fake_for_g.mean()

        return loss_g, loss_d


class QNRLoss(nn.Module):
    def __init__(self, cfg, logger):
        # type: (Config, Logger) -> None
        super(QNRLoss, self).__init__()
        self.cfg = cfg
        self.logger = logger

    def forward(self, pan, ms, out, pan_l=None):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        D_lambda = D_lambda_torch(l_ms=ms, ps=out)
        D_s = D_s_torch(l_ms=ms, pan=pan, l_pan=pan_l if pan_l is not None else down_sample(pan), ps=out)
        QNR = (1 - D_lambda) * (1 - D_s)
        return 1 - QNR


from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable

CE = torch.nn.BCELoss(reduction='sum')


class Mutual_info_reg(nn.Module):
    def __init__(self, cfg, logger, input_channels=4, channels=4, latent_size=4):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        self.channel = channels

        self.fc1_rgb3 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc2_rgb3 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc1_depth3 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc2_depth3 = nn.Linear(channels * 1 * 32 * 32, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.layer3(self.leakyrelu(self.layer1(rgb_feat)))
        depth_feat = self.layer4(self.leakyrelu(self.layer2(depth_feat)))
        rgb_feat = rgb_feat.view(-1, self.channel * 1 * 32 * 32)
        depth_feat = depth_feat.view(-1, self.channel * 1 * 32 * 32)
        mu_rgb = self.fc1_rgb3(rgb_feat)
        logvar_rgb = self.fc2_rgb3(rgb_feat)
        mu_depth = self.fc1_depth3(depth_feat)
        logvar_depth = self.fc2_depth3(depth_feat)

        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm, z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth + ce_depth_rgb - bi_di_kld

        return latent_loss


def get_loss_module(full_cfg, logger):
    r""" get the loss dictionary, mapping from loss_name to loss_instance

    Args:
        full_cfg (Config): full config, use 'cuda' and 'loss_cfg' in it
        logger (Logger)
    Returns:
        dict[str, nn.Module]: loss mapping
    """
    loss_cfg = full_cfg.get('loss_cfg')
    loss_module = dict()

    for loss_name in loss_cfg:
        cfg = loss_cfg[loss_name]
        if 'rec_loss' in loss_name:
            if abs(cfg.w - 0) > 1e-8:
                loss_module[loss_name] = ReconstructionLoss(cfg, logger, loss_type=cfg.type)
        if 'adv_loss' in loss_name:
            if abs(cfg.w - 0) > 1e-8:
                loss_module[loss_name] = AdversarialLoss(cfg, logger, gan_type=cfg.type, cuda=full_cfg.cuda)
        if 'QNR_loss' in loss_name:
            if abs(cfg.w - 0) > 1e-8:
                loss_module[loss_name] = QNRLoss(cfg, logger)
        if 'MI_loss' in loss_name:
            if abs(cfg.w - 0) > 1e-8:
                loss_module[loss_name] = Mutual_info_reg(cfg, logger)

    return loss_module
