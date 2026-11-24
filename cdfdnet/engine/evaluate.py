# -*- coding: utf-8 -*-
"""Evaluation protocol.

Value chain: network output in [-1,1] -> clamp -> *0.5+0.5 -> [0,1] -> metrics.
PSNR is computed at full resolution on RGB.
SSIM follows the common convention of Wang et al. (2004): the images are first
downsampled by max(1, round(min(H, W) / 256)) (620x460 -> 310x230), then an
11x11 Gaussian window (sigma = 1.5) is applied with 'valid' convolution.
"""
import torch
import torch.nn.functional as F


def to01(x):
    return x.clamp(-1, 1) * 0.5 + 0.5


def psnr(pred01, gt01):
    mse = F.mse_loss(pred01, gt01, reduction='none').mean((1, 2, 3))
    return 10 * torch.log10(1.0 / mse)


def _gaussian(window_size, sigma, device, dtype):
    g = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def ssim(pred01, gt01, down_ratio=True, window_size=11, sigma=1.5,
         C1=0.01 ** 2, C2=0.03 ** 2):
    x, y = pred01, gt01
    if down_ratio:
        H, W = x.shape[-2:]
        r = max(1, int(round(min(H, W) / 256.0)))
        if r > 1:
            x = F.avg_pool2d(x, r)
            y = F.avg_pool2d(y, r)
    C = x.shape[1]
    w1 = _gaussian(window_size, sigma, x.device, x.dtype)
    win = (w1[:, None] @ w1[None, :]).expand(C, 1, window_size, window_size).contiguous()
    mu_x = F.conv2d(x, win, groups=C)
    mu_y = F.conv2d(y, win, groups=C)
    mx2, my2, mxy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx = F.conv2d(x * x, win, groups=C) - mx2
    sy = F.conv2d(y * y, win, groups=C) - my2
    sxy = F.conv2d(x * y, win, groups=C) - mxy
    m = ((2 * mxy + C1) * (2 * sxy + C2)) / ((mx2 + my2 + C1) * (sx + sy + C2))
    return m.flatten(1).mean(1)


@torch.no_grad()
def evaluate(net, loader, device, ssim_down_ratio=True):
    """Return {'psnr': mean, 'ssim': mean, 'n': count} over a paired loader."""
    net.eval()
    ps, ss, n = 0.0, 0.0, 0
    for batch in loader:
        src = batch['source'].to(device)
        tgt = batch['target'].to(device)
        out = to01(net(src))
        gt = to01(tgt)
        ps += psnr(out, gt).sum().item()
        ss += ssim(out, gt, down_ratio=ssim_down_ratio).sum().item()
        n += src.size(0)
    return {'psnr': ps / n, 'ssim': ss / n, 'n': n}
