#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package a training checkpoint (weights + model config, no optimizer state)
in the training form. Use scripts/fold_deploy.py for the deployed form.

  python scripts/export_model.py --ckpt ckpt/cdfdnet_sots_indoor.pth --out cdfdnet_sots_indoor_train.pth
"""
import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdfdnet import __version__                                           # noqa: E402
from cdfdnet.models import build_model, count_params, DEFAULT_MODEL_CFG   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    ck_path, out = pathlib.Path(args.ckpt), pathlib.Path(args.out)
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    if 'state_dict' not in ck:
        sys.exit('not a training checkpoint (no state_dict)')
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in ck['state_dict'].items()}
    mcfg = dict(DEFAULT_MODEL_CFG)
    mcfg.update({k: v for k, v in ck.get('config', {}).get('model', {}).items() if k in DEFAULT_MODEL_CFG})
    net = build_model(mcfg)
    net.load_state_dict(sd, strict=True)
    pkg = {'model_cfg': mcfg, 'state_dict': sd,
           # meta holds no source file names, paths or validation scores
           'meta': {'name': out.stem, 'version': __version__, 'form': 'training',
                    'params_training': count_params(net)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pkg, out)
    print(f'saved {out} ({out.stat().st_size / 2 ** 20:.1f} MiB) | params {count_params(net):,}')


if __name__ == '__main__':
    main()
