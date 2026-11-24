#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure PSNR/SSIM of saved dehazed images against GT with the same metric
implementation used for training and inference.

  python scripts/evaluate.py --pred out/ --gt data/RESIDE-IN/test/GT --metrics out.csv
  python scripts/evaluate.py --pred out_ihaze/ --gt data/I-HAZE/test/GT --center-crop 256

Saved images are 8-bit, so the numbers are lower than the floating-point
metrics of scripts/infer.py: with the released weights the mean PSNR is about
0.31 dB lower on SOTS-Indoor (47.36 vs. 47.67 dB) and nearly identical on
I-HAZE (24.219 vs. 24.220 dB; SSIM 0.7299 vs. 0.7306). The numbers reported
in the paper are the floating-point ones (scripts/infer.py --gt ... --metrics).
"""
import argparse
import csv
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdfdnet.engine import psnr, ssim                  # noqa: E402
from cdfdnet.inference import find_gt, IMG_EXTS        # noqa: E402

EXTS = set(IMG_EXTS)


def load01(path, device):
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(path).convert('RGB'), dtype='uint8').copy()
    return (torch.from_numpy(a).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)


def center_crop(t, size):
    H, W = t.shape[-2:]
    ys, xs = (H - size) // 2, (W - size) // 2
    return t[..., ys:ys + size, xs:xs + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--metrics', default=None)
    ap.add_argument('--label', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--center-crop', type=int, default=0)
    ap.add_argument('--no-ssim-down-ratio', dest='sdr', action='store_false', default=True)
    args = ap.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    pred_dir, gt_dir = pathlib.Path(args.pred), pathlib.Path(args.gt)
    files = sorted(f for f in pred_dir.rglob('*') if f.suffix.lower() in EXTS)
    if not files:
        sys.exit(f'no images under {pred_dir}')
    rows, skipped = [], []
    ps = ss = 0.0
    with torch.no_grad():
        for f in files:
            g = find_gt(gt_dir, f.stem, f.suffix)
            if g is None:
                skipped.append(f.name)
                continue
            x, y = load01(f, device), load01(g, device)
            if x.shape != y.shape:
                H, W = y.shape[-2:]
                h, w = x.shape[-2:]
                if h >= H and w >= W:
                    t, l = (h - H) // 2, (w - W) // 2
                    x = x[..., t:t + H, l:l + W]
                else:
                    skipped.append(f.name)
                    continue
            if args.center_crop:
                x, y = center_crop(x, args.center_crop), center_crop(y, args.center_crop)
            p = psnr(x, y).item()
            s = ssim(x, y, down_ratio=args.sdr).item()
            ps += p
            ss += s
            rows.append({'file': f.name, 'gt': g.name, 'psnr': round(p, 4), 'ssim': round(s, 6)})
    if not rows:
        sys.exit('no image matched a GT file')
    n = len(rows)
    print(f'{args.label or pred_dir.name} | PSNR {ps / n:.3f} | SSIM {ss / n:.4f} | n={n}')
    if args.metrics:
        mp = pathlib.Path(args.metrics)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with open(mp, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=['file', 'gt', 'psnr', 'ssim'])
            w.writeheader()
            w.writerows(rows)
            w.writerow({'file': 'MEAN', 'gt': f'n={n}', 'psnr': round(ps / n, 4), 'ssim': round(ss / n, 6)})
        print(f'metrics -> {mp}')
    if skipped:
        print(f'warning: skipped {len(skipped)}: {", ".join(skipped[:6])}')


if __name__ == '__main__':
    main()
