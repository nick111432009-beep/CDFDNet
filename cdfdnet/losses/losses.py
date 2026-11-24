# -*- coding: utf-8 -*-
"""Training losses: L1 + frequency-domain L1 (L_freq) and the uniform contrastive
loss (L_ucr) on ResNet-152 features.

L = L1 + lambda_f * L_freq + lambda_c * L_ucr
"""
import torch
import torch.nn as nn
from torchvision import models


class Resnet152(nn.Module):
    """Frozen ImageNet ResNet-152; returns the outputs of its four stages."""

    def __init__(self, requires_grad=False):
        super().__init__()
        try:
            from torchvision.models import ResNet152_Weights
            feats = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        except Exception:
            feats = models.resnet152(pretrained=True)
        ch = list(feats.children())
        self.slice1 = nn.Sequential(*ch[:-5])
        self.slice2 = nn.Sequential(*ch[-5:-4])
        self.slice3 = nn.Sequential(*ch[-4:-3])
        self.slice4 = nn.Sequential(*ch[-3:-2])
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]


class ContrastLoss(nn.Module):
    """sum_i w_i * L1(phi_i(out), phi_i(gt)) / (L1(phi_i(out), phi_i(hazy)) + 1e-7).

    The feature extractor keeps the behavior of the base implementation (its
    BatchNorm layers are not switched to eval mode during training).
    """

    def __init__(self, device='cuda', layer_weights=None):
        super().__init__()
        self.net = Resnet152().to(device)
        self.l1 = nn.L1Loss()
        self.weights = list(layer_weights) if layer_weights else [0.359375] * 4

    def forward(self, a, p, n):
        a_f, p_f, n_f = self.net(a), self.net(p), self.net(n)
        loss = 0
        for i in range(len(a_f)):
            d_ap = self.l1(a_f[i], p_f[i].detach())
            d_an = self.l1(a_f[i], n_f[i].detach())
            loss = loss + self.weights[i] * (d_ap / (d_an + 1e-7))
        return loss


class L1WithFFT(nn.Module):
    """Pixel L1 plus fft_weight * L1 between the orthonormal rFFT2 spectra
    (computed in fp32)."""

    def __init__(self, fft_weight=0.0):
        super().__init__()
        self.base = nn.L1Loss()
        self.fft_weight = float(fft_weight)

    def forward(self, out, tgt):
        loss = self.base(out, tgt)
        if self.fft_weight > 0:
            with torch.autocast(device_type=out.device.type, enabled=False):
                fo = torch.fft.rfft2(out.float(), norm='ortho')
                ft = torch.fft.rfft2(tgt.float(), norm='ortho')
                loss = loss + self.fft_weight * (fo - ft).abs().mean()
        return loss


def build_criterion(cfg, device):
    """Return (pixel_loss, l1_weight, contrast_loss, contrast_weight)."""
    lc = cfg.get('loss', {})
    fft_w = float(lc.get('fft_l1_weight', 0.0))
    l1 = L1WithFFT(fft_w) if fft_w > 0 else nn.L1Loss()
    cr = ContrastLoss(device=device, layer_weights=lc.get('contrast_layer_weights')).to(device)
    return l1, float(lc.get('l1_weight', 1.0)), cr, float(lc.get('contrast_weight', 0.1))
