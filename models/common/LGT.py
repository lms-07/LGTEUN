# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-May
# @Address  : Time Lab @ SDU
# @FileName : LGT.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

# local-global transformer, LGT

import math
import warnings
import torch
import torch.nn as nn

import models.common.basic_module_unformer_v2 as bmu

from torch import einsum
from einops import rearrange


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class pre_norm(nn.Module):
    def __init__(self, channels, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # HMSA or MLP or FFN object


class patch_embedding(nn.Module):
    def __init__(self, in_channels, embed_channels, patch_size, norm_layer=None):
        super(patch_embedding, self).__init__()

        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.patch_size = patch_size

        self.proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.patch_size, stride=self.patch_size,
                      padding=0, groups=self.in_channels),
            nn.Conv2d(self.in_channels, self.embed_channels, kernel_size=1, stride=1, padding=0, groups=1)
        )

        if norm_layer is not None:
            self.norm = norm_layer(self.embed_channels)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)  # B Ph,Pw C
        if self.norm is not None:
            x = self.norm(x)

        return x


class feed_forward(nn.Module):
    def __init__(self, channels, ratio=4):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            bmu.point_conv(self.channels, self.channels * ratio),
            nn.GELU(),
            bmu.depthwise_conv(self.channels * ratio, self.channels * ratio),
            nn.GELU(),
            bmu.point_conv(self.channels * ratio, self.channels)
        )

    def forward(self, x):
        '''
        x: [b,h,w,c]
        return out: [b,h,w,c]
        '''
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class local_mixer(nn.Module):
    def __init__(self, channels, win_size, num_heads):
        super(local_mixer, self).__init__()

        self.channels = channels
        self.win_size = win_size
        self.num_heads = num_heads
        self.head_channel = self.channels // self.num_heads
        self.scale = self.head_channel ** -0.5

        self.to_qkv = bmu.point_conv(self.channels, self.channels * 3)
        self.softmax = nn.Softmax(dim=-1)

        # position embedding
        seq_l = self.win_size * self.win_size
        self.pos_emb = nn.Parameter(torch.Tensor(1, self.num_heads, seq_l, seq_l))
        trunc_normal_(self.pos_emb)

    def forward(self, x):
        # patch size as token is 2
        # here, h, w after patch embedding as h/2, w/2
        b, h, w, c = x.size()

        x_win = rearrange(x, 'b (h i) (w j) c -> b c (h w) (i j)', i=self.win_size, j=self.win_size)  ######33
        q, k, v = self.to_qkv(x_win).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) m n  -> (b m) h n c', h=self.num_heads), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i c, b h j c -> b h i j', q, k)
        sim = sim + self.pos_emb
        atten_map = self.softmax(sim)
        out = einsum('b h i j, b h j c -> b h i c', atten_map, v)
        out = rearrange(out, 'b h m c -> b m (h c)')

        return out


class global_mixer(nn.Module):
    def __init__(self, channels):
        super(global_mixer, self).__init__()
        self.channels = channels

        self.conv_amp = nn.Sequential(
            bmu.dep_conv(self.channels, kernel_size=1),  # dep_conv group=self.channels
        )

        self.conv_pha = nn.Sequential(
            bmu.dep_conv(self.channels, kernel_size=1),  # dep_conv group=self.channels
        )

    def forward(self, x):
        b, h, w, c = x.size()
        x = x.permute(0, 3, 1, 2)  # [b c h w]

        fre = torch.fft.rfft2(x, norm='backward')

        amp = torch.abs(fre)
        pha = torch.angle(fre)

        amp_fea = self.conv_amp(amp)
        pha_fea = self.conv_pha(pha)

        real = amp_fea * torch.cos(pha_fea) + 1e-8
        imag = amp_fea * torch.sin(pha_fea) + 1e-8

        out = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(h, w), norm='backward'))

        return out.permute(0, 2, 3, 1)


class LGMixer(nn.Module):
    def __init__(self, channels, win_size, num_heads, channel_ratio):
        super(LGMixer, self).__init__()
        self.channels = channels
        self.win_size = win_size
        self.num_heads = num_heads
        self.channel_ratio = channel_ratio

        # half channles local and global
        self.half_chan = self.channels // 2

        self.local_mixer = local_mixer(channels=self.half_chan, win_size=self.win_size, num_heads=self.num_heads)
        self.global_mixer = global_mixer(self.half_chan)

        self.proj = bmu.point_conv(self.channels, self.channels)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        b, h, w, c = x.size()

        x1 = x[:, :, :, :self.half_chan].contiguous()
        x2 = x[:, :, :, self.half_chan:].contiguous()

        x1 = self.local_mixer(x1)
        x1 = rearrange(x1, '(b h w) (i j) c ->  b (h i) (w j) c', h=h // self.win_size, w=w // self.win_size,
                       i=self.win_size)

        x2 = self.global_mixer(x2)

        out = torch.cat((x1, x2), dim=-1).permute(0, 3, 1, 2)

        out = self.proj(out)
        out = self.proj_drop(out)

        out = out.permute(0, 2, 3, 1)

        return out


class LGB(nn.Module):
    def __init__(self, channels, num_blocks, win_size, num_heads, channel_ratio):
        super(LGB, self).__init__()
        self.channels = channels
        self.blocks = nn.ModuleList([])
        self.window_size = win_size
        self.num_heads = num_heads
        self.channel_ratio = channel_ratio

        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                residual(
                    fn=pre_norm(channels=self.channels,
                                fn=LGMixer(channels=self.channels, win_size=self.window_size, num_heads=self.num_heads,
                                           channel_ratio=self.channel_ratio))),
                residual(fn=pre_norm(channels=self.channels, fn=feed_forward(channels=self.channels)))
            ]))

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,c,h,w]
        """
        for (mixer, ffn) in self.blocks:
            x = mixer(x)
            x = ffn(x)
        return x.permute(0, 3, 1, 2)


class LGT(nn.Module):
    def __init__(self, in_channels=8, embed_channels=16, patch_size=1, window_size=8,
                 num_block=[2, 1], num_heads=2, channel_ratio=0, patch_norm=True, ):
        super(LGT, self).__init__()

        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.patch_size = patch_size
        self.window_size = window_size

        self.patch_norm = patch_norm
        self.norm_layer = nn.LayerNorm if self.patch_norm else None

        self.scales = len(num_block)
        self.layer_channels = embed_channels

        self.num_heads = num_heads
        self.channel_ratio = channel_ratio

        self.patch_embed = patch_embedding(in_channels=self.in_channels, embed_channels=self.embed_channels,
                                           patch_size=self.patch_size, norm_layer=self.norm_layer)

        # encoder
        self.encoder_layers = nn.ModuleList([])
        for i in range(self.scales - 1):
            self.encoder_layers.append(nn.ModuleList([
                LGB(channels=self.layer_channels, num_blocks=num_block[i], win_size=self.window_size,
                    num_heads=self.num_heads,
                    channel_ratio=self.channel_ratio),
                nn.Sequential(bmu.sampling_unit_(1 / 2),
                              bmu.point_conv(self.layer_channels, self.layer_channels * 2)),
            ]))
            self.layer_channels *= 2

        # bottleneck
        self.bottleneck = LGB(channels=self.layer_channels, num_blocks=num_block[-1], win_size=self.window_size,
                              num_heads=self.num_heads,
                              channel_ratio=self.channel_ratio)

        # decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales - 1):
            self.decoder_layers.append(nn.ModuleList([
                nn.Sequential(bmu.sampling_unit_(2), bmu.point_conv(self.layer_channels, self.layer_channels // 2)),
                bmu.point_conv(self.layer_channels, self.layer_channels // 2),
                LGB(channels=self.layer_channels // 2, num_blocks=num_block[self.scales - 2 - i],
                    win_size=self.window_size, num_heads=self.num_heads, channel_ratio=self.channel_ratio)
            ]))
            self.layer_channels //= 2

        # reconstruction
        self.tail = nn.Sequential(bmu.sampling_unit_(s_factor=patch_size),
                                  bmu.point_conv(self.layer_channels, self.in_channels))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h, w = x.size()

        fea = self.patch_embed(x)

        # encoder
        fea_encoder = []
        for (LGB, fea_down_s) in self.encoder_layers:
            fea = LGB(fea)  # [b h w c]
            fea_encoder.append(fea)  # [b c h w]
            fea = fea_down_s(fea)
            fea = fea.permute(0, 2, 3, 1)  # [b h w c]

        # bottleneck
        fea = self.bottleneck(fea)

        # decoder
        for i, (fea_up_s, fusion_conv, LGB) in enumerate(self.decoder_layers):
            fea = fea_up_s(fea)
            fea = fusion_conv(
                torch.cat([fea, fea_encoder[self.scales - 2 - i]], dim=1))  # test scale
            fea = LGB(fea.permute(0, 2, 3, 1))

        # reconstruction
        out = self.tail(fea) + x

        return out
