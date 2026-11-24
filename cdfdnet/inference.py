# -*- coding: utf-8 -*-
"""Loading released weights and running inference.

    from cdfdnet.inference import load_model, dehaze
    net = load_model('cdfdnet_sots_indoor.pth')          # deployed (folded) weights
    out = dehaze(net, 'hazy.png')                        # PIL image in return

Weight packages are dicts with 'model_cfg', 'state_dict' and 'meta'
(produced by scripts/export_model.py or scripts/fold_deploy.py). They contain
only tensors and plain Python values and are loaded with weights_only=True.
"""
import pathlib
import re

import torch
import torch.nn.functional as F

from .models import build_model


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def gt_candidates(stem):
    """Possible GT file stems for a hazy file stem, in order of preference:
    the identical name; RESIDE ITS / SOTS names ('1400_1' or '1_1_0.90179' -> '1400'
    or '1'); I-HAZE / O-HAZE names ('31_indoor_hazy' -> '31_indoor_GT')."""
    cands = [stem]
    m = re.fullmatch(r'(\d+)_\d+(_[0-9.]+)?', stem)
    if m:
        cands.append(m.group(1))
    elif '_' in stem and stem.rsplit('_', 1)[1].isdigit():
        cands.append(stem.rsplit('_', 1)[0])
    if stem.lower().endswith('_hazy'):
        base = stem[:-5]
        cands += [base + '_GT', base + '_gt', base]
    return cands


def find_gt(gt_dir, stem, ext=''):
    """Return the GT file in `gt_dir` that pairs with a hazy file, or None."""
    gt_dir = pathlib.Path(gt_dir)
    for c in gt_candidates(stem):
        for e in (ext, *IMG_EXTS):
            p = gt_dir / f'{c}{e}'
            if e and p.exists():
                return p
    return None


def load_package(path):
    """Load a released weight package (safe loading: tensors and plain values only)."""
    pkg = torch.load(str(path), map_location='cpu', weights_only=True)
    if 'model_cfg' not in pkg or 'state_dict' not in pkg:
        raise ValueError(f'{path} is not a CDFDNet weight package (needs model_cfg and state_dict)')
    return pkg


def load_model(weights, device=None, verbose=True):
    """Build the model described by the package and load its weights (strict)."""
    path = pathlib.Path(weights)
    pkg = load_package(path)
    net = build_model(pkg['model_cfg'])
    net.load_state_dict(pkg['state_dict'], strict=True)
    net.eval()
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    net = net.to(dev)
    net.meta = pkg.get('meta', {})
    if verbose:
        form = 'training' if pkg['model_cfg'].get('detail_conv', True) else 'deployed'
        print(f'loaded {path.name} | {form} form | {dev}')
    return net


def load_image(path):
    """File -> (1,3,H,W) tensor in [-1,1]."""
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert('RGB')
    a = torch.from_numpy(np.asarray(im, dtype='uint8').copy()).permute(2, 0, 1)
    return (a.float() / 255.0 * 2 - 1).unsqueeze(0)


def save_image(t01, path):
    """(1,3,H,W) tensor in [0,1] -> PNG/JPEG file."""
    from PIL import Image
    a = (t01.clamp(0, 1)[0] * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


def pad_to(x, div):
    """Reflect-pad to a multiple of `div`; returns (padded, H, W)."""
    H, W = x.shape[-2:]
    ph, pw = (-H) % div, (-W) % div
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    return x, H, W


def _run(net, x, amp):
    if amp and x.device.type == 'cuda':
        with torch.autocast('cuda', dtype=torch.float16):
            y = net(x)
        return y.float()
    return net(x)


@torch.no_grad()
def infer_whole(net, x, amp=False):
    """Whole-image inference with reflect padding; x is (1,3,H,W) in [-1,1]."""
    div = 2 ** getattr(net, 'n_down', 2)
    xp, H, W = pad_to(x, div)
    return _run(net, xp, amp)[..., :H, :W]


def _ramp(L, ov):
    r = torch.ones(L)
    if ov > 0:
        for i in range(min(ov, L)):
            v = (i + 1) / (ov + 1)
            r[i] = min(float(r[i]), v)
            r[L - 1 - i] = min(float(r[L - 1 - i]), v)
    return r.clamp(min=1e-3)


def _starts(total, tile, stride):
    ss = [0]
    while ss[-1] + tile < total:
        ss.append(min(ss[-1] + stride, total - tile))
    return sorted(set(ss))


@torch.no_grad()
def infer_tiled(net, x, tile=1024, overlap=128, amp=False):
    """Overlapping tiled inference with linear feathering (for very large images,
    e.g. I-HAZE). Returns a CPU tensor of the same size as x."""
    div = 2 ** getattr(net, 'n_down', 2)
    _, _, H, W = x.shape
    th, tw = min(tile, H), min(tile, W)
    stride = tile - overlap
    acc = torch.zeros(1, 3, H, W)
    wac = torch.zeros(1, 1, H, W)
    for ys in _starts(H, th, stride):
        for xs in _starts(W, tw, stride):
            patch = x[:, :, ys:ys + th, xs:xs + tw]
            pp, h0, w0 = pad_to(patch, div)
            y = _run(net, pp, amp)[..., :h0, :w0].cpu()
            wy = _ramp(h0, overlap if th < H else 0)
            wx = _ramp(w0, overlap if tw < W else 0)
            w2 = wy.view(1, 1, h0, 1) * wx.view(1, 1, 1, w0)
            acc[..., ys:ys + h0, xs:xs + w0] += y * w2
            wac[..., ys:ys + h0, xs:xs + w0] += w2
    return acc / wac


@torch.no_grad()
def dehaze(net, img, tile=0, overlap=128, amp=False):
    """Dehaze one image (file path, PIL image or (1,3,H,W) tensor in [-1,1]);
    returns a PIL image."""
    from PIL import Image
    import numpy as np
    if isinstance(img, (str, pathlib.Path)):
        x = load_image(img)
    elif torch.is_tensor(img):
        x = img if img.dim() == 4 else img.unsqueeze(0)
    else:
        a = np.asarray(img.convert('RGB'), dtype='uint8').copy()
        x = (torch.from_numpy(a).permute(2, 0, 1).float() / 255.0 * 2 - 1).unsqueeze(0)
    dev = next(net.parameters()).device
    x = x.to(dev)
    y = infer_tiled(net, x, tile, overlap, amp) if tile else infer_whole(net, x, amp)
    y01 = y.clamp(-1, 1) * 0.5 + 0.5
    a = (y01[0] * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)
