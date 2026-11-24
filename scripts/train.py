#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train CDFDNet.

  python scripts/train.py --config configs/cdfdnet_sots_indoor.yaml
  python scripts/train.py --config configs/cdfdnet_sots_indoor.yaml --max-hours 12   # resumable segments
  python scripts/train.py --config configs/cdfdnet_ihaze.yaml --override data.root=/path/to/I-HAZE

Running the same command again resumes from <ckpt_dir>/<name>_last.pth.
"""
import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdfdnet.utils import load_config, apply_overrides, describe, append_row   # noqa: E402
from cdfdnet.models import build_model, count_params                           # noqa: E402
from cdfdnet.data import build_loaders                                          # noqa: E402
from cdfdnet.losses import build_criterion                                      # noqa: E402
from cdfdnet.engine import train, evaluate, find_resume                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--max-hours', type=float, default=None, help='stop cleanly after this many hours')
    ap.add_argument('--override', nargs='*', default=None, help='e.g. train.batch_size=8 data.root=/data/RESIDE-IN')
    ap.add_argument('--fresh', action='store_true', help='ignore existing checkpoints')
    ap.add_argument('--eval-only', action='store_true', help='evaluate the best checkpoint on the test split')
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    if args.fresh:
        cfg['train']['fresh_start'] = True
    print(describe(cfg))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = int(cfg['train'].get('seed', 42))
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    net = build_model(cfg['model']).to(device)
    n_param = count_params(net)
    gpu = torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'
    print(f'parameters {n_param:,} | device {gpu}')

    loaders = build_loaders(cfg)
    print(f'train {len(loaders[0])} pairs | val {len(loaders[1])} pairs | {len(loaders[2])} steps/epoch')
    ckpt_dir = pathlib.Path(cfg['io']['ckpt_dir'])

    if args.eval_only:
        best = ckpt_dir / f"{cfg['name']}.pth"
        src_ck = best if best.exists() else find_resume(ckpt_dir, cfg['name'])
        if src_ck is None:
            sys.exit(f'no checkpoint for {cfg["name"]} under {ckpt_dir}')
        ck = torch.load(str(src_ck), map_location=device, weights_only=False)
        net.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in ck['state_dict'].items()})
        m = evaluate(net, loaders[3], device, ssim_down_ratio=cfg['eval'].get('ssim_down_ratio', True))
        print(f"{src_ck.name}: PSNR {m['psnr']:.4f} | SSIM {m['ssim']:.4f} | n={m['n']}")
        return

    resume = find_resume(ckpt_dir, cfg['name']) if not args.fresh else None
    print(f'resuming from {resume.name}' if resume else 'training from scratch')
    criterion = build_criterion(cfg, device)
    result = train(cfg, net, loaders, criterion, device, max_hours=args.max_hours)

    best_ck = ckpt_dir / f"{cfg['name']}.pth"
    if best_ck.exists():
        ck = torch.load(str(best_ck), map_location=device, weights_only=False)
        net.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in ck['state_dict'].items()})
    metrics = evaluate(net, loaders[3], device, ssim_down_ratio=cfg['eval'].get('ssim_down_ratio', True))
    print(f"best PSNR {result['best_psnr']:.3f} | SSIM {metrics['ssim']:.4f} | "
          f"{result['it_s']:.2f} it/s | peak {result['peak_gb']:.2f} GB | {result['hours']:.2f} h")
    log = append_row(ROOT / cfg['io'].get('run_log', 'runs.csv'), cfg, result,
                     extra={'params': n_param, 'gpu': gpu, 'ssim': round(metrics['ssim'], 4)})
    print(f'logged to {log}')


if __name__ == '__main__':
    main()
