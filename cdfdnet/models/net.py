# -*- coding: utf-8 -*-
"""CDFDNet: a three-stage U-shaped network with CDB blocks in stages 1-2 and
CDFB blocks (with frequency modulation) in stage 3."""
import torch
import torch.nn as nn

from .blocks import CDFDBlock, DetailConv, CDB_STEPS, CDFB_STEPS


class CDFDNet(nn.Module):
    """
    Args:
        in_chans, out_chans: image channels.
        embed_dims: channels of the three stages (default 32, 64, 128).
        depths: number of blocks per stage (default 8, 12, 16).
        detail_conv: True = training form (DetailConv at the entrance, in the
            fine-scale branch and in the tri-scale dilated branches);
            False = deployed form (every DetailConv folded into a standard convolution).
        fm_stages: stage indices that use CDFB (frequency modulation); default (2,).
        big_kernel, strip: kernel sizes of the large-kernel and strip convolution.
    """

    def __init__(self, in_chans=3, out_chans=3, embed_dims=(32, 64, 128), depths=(8, 12, 16),
                 detail_conv=True, fm_stages=(2,), big_kernel=7, strip=11):
        super().__init__()
        embed_dims, depths = list(embed_dims), list(depths)
        assert len(embed_dims) == len(depths)
        self.detail_conv = bool(detail_conv)
        modules_cfg = {
            'CRA': {'residual': True},
            'TSDW': {'groups': 'depthwise'},
            'TSDD': {'detail_conv': self.detail_conv},
            'DSD': {'detail_conv': self.detail_conv},
            'LKS': {'big': big_kernel, 'strip': strip},
        }
        if self.detail_conv:
            self.patch_embed = DetailConv(in_chans, embed_dims[0])
        else:
            self.patch_embed = nn.Conv2d(in_chans, embed_dims[0], 3, padding=1)
        self.layers = nn.ModuleList([
            nn.Sequential(*[CDFDBlock(embed_dims[i], CDFB_STEPS if i in fm_stages else CDB_STEPS,
                                      modules_cfg) for _ in range(depths[i])])
            for i in range(len(embed_dims))])
        self.transition = nn.ModuleList([
            nn.Conv2d(embed_dims[i], embed_dims[i + 1], 3, stride=2, padding=1)
            for i in range(len(embed_dims) - 1)])
        self.upscale = nn.ModuleList([
            nn.ConvTranspose2d(embed_dims[i + 1], embed_dims[i], 3, stride=2, padding=1,
                               output_padding=1)
            for i in range(len(embed_dims) - 1)])
        self.skip_adapters = nn.ModuleList([
            nn.Conv2d(embed_dims[i], embed_dims[i], 1) for i in range(len(embed_dims) - 1)])
        self.final_conv = nn.Conv2d(embed_dims[0], out_chans, 3, padding=1)
        self.n_down = len(embed_dims) - 1

    def forward(self, x):
        H, W = x.shape[-2:]
        div = 2 ** self.n_down
        if H % div or W % div:
            raise ValueError(f'input size {H}x{W} must be divisible by {div}; pad first '
                             f'(see cdfdnet.inference.dehaze).')
        x = self.patch_embed(x)
        skips = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                skips.append(x)
                x = self.transition[i](x)
        for i in range(len(skips) - 1, -1, -1):
            x = self.upscale[i](x)
            x = x + self.skip_adapters[i](skips[i])
        return self.final_conv(x)


DEFAULT_MODEL_CFG = {'in_chans': 3, 'out_chans': 3, 'embed_dims': [32, 64, 128],
                     'depths': [8, 12, 16], 'detail_conv': True, 'fm_stages': [2],
                     'big_kernel': 7, 'strip': 11}


def build_model(cfg=None) -> nn.Module:
    """Build CDFDNet from a (possibly partial) model config dict."""
    m = dict(DEFAULT_MODEL_CFG)
    m.update({k: v for k, v in (cfg or {}).items() if k in DEFAULT_MODEL_CFG})
    return CDFDNet(**m)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
