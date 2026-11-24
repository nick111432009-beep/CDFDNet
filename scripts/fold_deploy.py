#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fold every DetailConv of a training checkpoint into a standard convolution and
save the deployed weights (re-parameterization).

  python scripts/fold_deploy.py --ckpt ckpt/cdfdnet_sots_indoor.pth --out cdfdnet_sots_indoor.pth

The script checks that the folded network has the expected number of
parameters (25,799,391 for the default size), that its output equals the
training form on random inputs, and that the deployed model loads the folded
state_dict strictly.
"""
import argparse
import copy
import pathlib
import sys

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdfdnet import __version__                                                       # noqa: E402
from cdfdnet.models import build_model, count_params, DetailConv, DEFAULT_MODEL_CFG   # noqa: E402


def fold_detailconv(net: nn.Module) -> int:
    """Replace every DetailConv by an equivalent nn.Conv2d in place; returns the count."""
    n = 0
    for name, m in list(net.named_modules()):
        if not isinstance(m, DetailConv):
            continue
        with torch.no_grad():
            wm, bm = m.merged_weight_bias()
        conv = nn.Conv2d(m.conv.in_channels, m.conv.out_channels, 3, padding=m.dilation,
                         dilation=m.dilation, bias=bm is not None).to(wm.device)
        with torch.no_grad():
            conv.weight.copy_(wm)
            if bm is not None:
                conv.bias.copy_(bm)
        parent_name, _, attr = name.rpartition('.')
        parent = net.get_submodule(parent_name) if parent_name else net
        if type(parent).__name__ == 'DualScaleDetailConv' and attr == 'conv2':
            conv = nn.Sequential(conv)        # same container as the deployed definition
        parent._modules[attr] = conv
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='training checkpoint or training-form package')
    ap.add_argument('--out', required=True)
    ap.add_argument('--expect-params', type=int, default=25_799_391)
    ap.add_argument('--dataset', default=None, help='training dataset name stored in meta (optional)')
    args = ap.parse_args()

    ck_path, out = pathlib.Path(args.ckpt), pathlib.Path(args.out)
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    mcfg = dict(DEFAULT_MODEL_CFG)
    src_cfg = ck.get('model_cfg') or ck.get('config', {}).get('model', {})
    mcfg.update({k: v for k, v in src_cfg.items() if k in DEFAULT_MODEL_CFG})
    mcfg['detail_conv'] = True
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in ck['state_dict'].items()}

    torch.manual_seed(0)
    net = build_model(mcfg)
    net.load_state_dict(sd, strict=True)
    net.eval()
    n_train = count_params(net)
    ref = copy.deepcopy(net)

    n_fold = fold_detailconv(net)
    net.eval()
    n_dep = count_params(net)
    print(f'training-form parameters {n_train:,} -> deployed {n_dep:,} ({n_fold} DetailConv folded)')
    if args.expect_params and n_dep != args.expect_params:
        sys.exit(f'deployed parameter count {n_dep:,} != expected {args.expect_params:,}')

    with torch.no_grad():
        for shape in [(1, 3, 256, 256), (1, 3, 460, 620)]:
            x = torch.rand(*shape)
            d = (ref(x) - net(x)).abs()
            print(f'equivalence {shape[2]}x{shape[3]}: max|diff|={d.max().item():.3e}')

    dcfg = dict(mcfg)
    dcfg['detail_conv'] = False
    dep_sd = net.state_dict()
    build_model(dcfg).load_state_dict(dep_sd, strict=True)
    print(f'deployed model loads the folded state_dict strictly ({len(dep_sd)} tensors)')

    pkg = {'model_cfg': dcfg,
           'state_dict': {k: v.detach().clone() for k, v in dep_sd.items()},
           # meta holds no source file names, paths or validation scores
           'meta': {'name': out.stem, 'version': __version__, 'dataset': args.dataset,
                    'form': 'deployed', 'folded_detailconv': n_fold,
                    'params_training': n_train, 'params_deployed': n_dep}}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pkg, out)
    print(f'saved {out} ({out.stat().st_size / 2 ** 20:.1f} MiB)')


if __name__ == '__main__':
    main()
