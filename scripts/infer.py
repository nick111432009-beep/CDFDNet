#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dehaze a folder (or a single image) and optionally measure PSNR/SSIM against GT.

  # dehaze only
  python scripts/infer.py --weights cdfdnet_sots_indoor.pth --input hazy/ --output out/

  # dehaze + metrics (SOTS-Indoor protocol)
  python scripts/infer.py --weights cdfdnet_sots_indoor.pth \
      --input data/RESIDE-IN/test/hazy --gt data/RESIDE-IN/test/GT --output out/ --metrics out/metrics.csv

  # I-HAZE protocol: tiled full-resolution inference, metrics on the central 256x256 region
  python scripts/infer.py --weights cdfdnet_ihaze.pth --input data/I-HAZE/test/hazy --gt data/I-HAZE/test/GT \
      --output out_ihaze/ --tile 1024 --overlap 128 --center-crop 256 --metrics out_ihaze/metrics.csv

  # a training checkpoint (training form) can be used directly
  python scripts/infer.py --ckpt ckpt/cdfdnet_sots_indoor.pth --input ... --output ...

Metrics are computed on the floating-point output clipped to [0, 1], before it is
stored as an 8-bit image. GT matching: the same file name first, then the RESIDE
convention ("1400_1.png" -> "1400.png") and the I-HAZE convention
("31_indoor_hazy.jpg" -> "31_indoor_GT.jpg"), see cdfdnet.inference.find_gt.
"""
import argparse
import csv
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdfdnet.models import build_model, count_params, DEFAULT_MODEL_CFG      # noqa: E402
from cdfdnet.engine import to01, psnr, ssim                                   # noqa: E402
from cdfdnet.inference import (load_package, load_image, save_image, infer_whole,  # noqa: E402
                               infer_tiled, find_gt, IMG_EXTS)

EXTS = set(IMG_EXTS)


def center_crop(t, size):
    H, W = t.shape[-2:]
    ys, xs = (H - size) // 2, (W - size) // 2
    return t[..., ys:ys + size, xs:xs + size]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--weights', help='weight package (model_cfg + state_dict)')
    src.add_argument('--ckpt', help='training checkpoint (training form, default model config)')
    ap.add_argument('--input', required=True)
    ap.add_argument('--gt', default=None, help='GT folder; metrics are computed when given')
    ap.add_argument('--output', default=None)
    ap.add_argument('--metrics', default=None, help='per-image CSV')
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--device', default=None)
    ap.add_argument('--amp', action='store_true', help='fp16 autocast inference')
    ap.add_argument('--tile', type=int, default=0, help='tile size for tiled inference (0 = whole image)')
    ap.add_argument('--overlap', type=int, default=128)
    ap.add_argument('--center-crop', type=int, default=0,
                    help='compute metrics on the central NxN region of output and GT (I-HAZE: 256)')
    ap.add_argument('--no-ssim-down-ratio', dest='sdr', action='store_false', default=True)
    args = ap.parse_args()

    if not args.output and not args.no_save:
        sys.exit('give --output, or --no-save to only compute metrics')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    if args.weights:
        pkg = load_package(args.weights)
        model_cfg, sd, srcname = pkg['model_cfg'], pkg['state_dict'], pathlib.Path(args.weights).name
    else:
        # training checkpoints also hold optimizer state; load only files you created yourself
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model_cfg = dict(DEFAULT_MODEL_CFG)
        model_cfg.update({k: v for k, v in ck.get('config', {}).get('model', {}).items() if k in DEFAULT_MODEL_CFG})
        sd = {(k[7:] if k.startswith('module.') else k): v for k, v in ck['state_dict'].items()}
        srcname = pathlib.Path(args.ckpt).name
    net = build_model(model_cfg).to(device).eval()
    net.load_state_dict(sd, strict=True)
    form = 'training' if model_cfg.get('detail_conv', True) else 'deployed'
    print(f'{srcname} | {form} form | params {count_params(net):,} | {device}')

    ip = pathlib.Path(args.input)
    files = [ip] if ip.is_file() else sorted(f for f in ip.rglob('*') if f.suffix.lower() in EXTS)
    if not files:
        sys.exit(f'no images under {ip}')
    gt_dir = pathlib.Path(args.gt) if args.gt else None
    out_dir = pathlib.Path(args.output) if args.output else None

    rows, skipped = [], []
    ps_sum = ss_sum = 0.0
    t0 = time.time()
    with torch.no_grad():
        for i, f in enumerate(files, 1):
            x = load_image(f).to(device)
            y = infer_tiled(net, x, args.tile, args.overlap, args.amp).to(device) if args.tile \
                else infer_whole(net, x, args.amp)
            y01 = to01(y)
            if out_dir and not args.no_save:
                save_image(y01, out_dir / f'{f.stem}.png')
            if gt_dir:
                g = find_gt(gt_dir, f.stem, f.suffix)
                if g is None:
                    skipped.append(f.name)
                else:
                    gt01 = to01(load_image(g).to(device))
                    a, b = (center_crop(y01, args.center_crop), center_crop(gt01, args.center_crop)) \
                        if args.center_crop else (y01, gt01)
                    p = psnr(a, b).item()
                    s = ssim(a, b, down_ratio=args.sdr).item()
                    ps_sum += p
                    ss_sum += s
                    rows.append({'file': f.name, 'gt': g.name, 'psnr': round(p, 4), 'ssim': round(s, 6)})
            if i % 50 == 0 or i == len(files):
                print(f'  {i}/{len(files)}', end='\r', flush=True)
    dt = time.time() - t0
    print(f'\n{len(files)} images in {dt:.1f}s')
    if rows:
        n = len(rows)
        print(f'PSNR {ps_sum / n:.3f} | SSIM {ss_sum / n:.4f} | n={n}'
              + (f' | central {args.center_crop}x{args.center_crop}' if args.center_crop else ''))
        if args.metrics:
            mp = pathlib.Path(args.metrics)
            mp.parent.mkdir(parents=True, exist_ok=True)
            with open(mp, 'w', newline='', encoding='utf-8') as fh:
                w = csv.DictWriter(fh, fieldnames=['file', 'gt', 'psnr', 'ssim'])
                w.writeheader()
                w.writerows(rows)
                w.writerow({'file': 'MEAN', 'gt': f'n={n}', 'psnr': round(ps_sum / n, 4), 'ssim': round(ss_sum / n, 6)})
            print(f'metrics -> {mp}')
    elif gt_dir:
        print('warning: no GT matched')
    if skipped:
        print(f'warning: {len(skipped)} images without GT: {", ".join(skipped[:8])}')


if __name__ == '__main__':
    main()
