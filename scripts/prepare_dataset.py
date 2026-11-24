#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset preparation.

`pair` converts an official hazy/clear folder pair into the layout used for
training (<dst>/hazy and <dst>/GT with identical file names). GT files are
matched with the same rules as scripts/infer.py (RESIDE: "1_1_0.90179.png" or
"1400_1.png" -> "1.png" / "1400.png"; I-HAZE: "31_indoor_hazy.jpg" ->
"31_indoor_GT.jpg"). Files are hard-linked when possible, otherwise copied.

  python scripts/prepare_dataset.py pair --hazy <ITS>/hazy --gt <ITS>/clear --dst data/RESIDE-IN/train
  python scripts/prepare_dataset.py pair --hazy <SOTS>/indoor/hazy --gt <SOTS>/indoor/gt --dst data/RESIDE-IN/test
  python scripts/prepare_dataset.py pair --hazy <I-HAZE>/hazy --gt <I-HAZE>/GT --dst data/I-HAZE/train --ids 1-25
  python scripts/prepare_dataset.py pair --hazy <I-HAZE>/hazy --gt <I-HAZE>/GT --dst data/I-HAZE/test --ids 31-35

`pack` / `restore` (optional): RESIDE ITS in the paired layout stores one GT
copy per hazy image (13,990 copies of 1,399 clear images). `pack` keeps one
copy of each unique GT (by content hash) and writes a manifest; `restore`
recreates the layout with hard links.

  python scripts/prepare_dataset.py pack --src data/RESIDE-IN --out RESIDE-IN-slim.zip
  python scripts/prepare_dataset.py restore --zip RESIDE-IN-slim.zip --dst data/RESIDE-IN
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdfdnet.inference import find_gt          # noqa: E402

EXTS = ('.jpg', '.png', '.jpeg', '.bmp')


def _link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _parse_ids(spec):
    ids = set()
    for part in spec.split(','):
        a, _, b = part.partition('-')
        ids.update(range(int(a), int(b or a) + 1))
    return ids


def pair(hazy_dir, gt_dir, dst, ids=None):
    hazy_dir, gt_dir, dst = pathlib.Path(hazy_dir), pathlib.Path(gt_dir), pathlib.Path(dst)
    keep = _parse_ids(ids) if ids else None
    (dst / 'hazy').mkdir(parents=True, exist_ok=True)
    (dst / 'GT').mkdir(parents=True, exist_ok=True)
    n, missing = 0, []
    for f in sorted(p for p in hazy_dir.iterdir() if p.suffix.lower() in EXTS):
        if keep is not None:
            m = re.match(r'(\d+)', f.stem)
            if not m or int(m.group(1)) not in keep:
                continue
        g = find_gt(gt_dir, f.stem, f.suffix)
        if g is None:
            missing.append(f.name)
            continue
        stem = f.stem[:-5] if f.stem.lower().endswith('_hazy') else f.stem
        name = stem + f.suffix.lower()
        _link_or_copy(f, dst / 'hazy' / name)
        _link_or_copy(g, dst / 'GT' / name)
        n += 1
    print(f'{n} pairs -> {dst}' + (f' | {len(missing)} without GT, e.g. {missing[:3]}' if missing else ''))
    if not n:
        sys.exit('no pairs written')


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pack(src, out):
    src = pathlib.Path(src)
    manifest, uniques, plain = {}, {}, []
    for sub in ('train', 'test'):
        gt_dir = src / sub / 'GT'
        if not gt_dir.is_dir():
            continue
        for f in sorted(p for p in gt_dir.iterdir() if p.suffix.lower() in EXTS):
            h = _md5(f)
            if h not in uniques:
                uniques[h] = (f'_unique/{h}{f.suffix.lower()}', f)
            manifest[f'{sub}/GT/{f.name}'] = uniques[h][0]
        hz = src / sub / 'hazy'
        if hz.is_dir():
            plain += [(f, f'{sub}/hazy/{f.name}') for f in sorted(hz.iterdir()) if f.suffix.lower() in EXTS]
    print(f'GT: {len(manifest):,} files -> {len(uniques):,} unique; hazy: {len(plain):,} files')
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) as z:
        z.writestr('manifest.json', json.dumps(manifest))
        for uname, fpath in uniques.values():
            z.write(fpath, uname)
        for fpath, arc in plain:
            z.write(fpath, arc)
    print(f'wrote {out} ({out.stat().st_size / 1024 ** 3:.2f} GB)')


def restore(zip_path, dst):
    dst = pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        if 'manifest.json' not in z.namelist():
            sys.exit('not a packed dataset (no manifest.json); a plain zip can simply be extracted')
        z.extractall(dst)
        manifest = json.loads((dst / 'manifest.json').read_text())

    def _safe(rel):
        n = os.path.normpath(rel)
        if os.path.isabs(n) or n.startswith('..'):
            sys.exit(f'unsafe path in manifest: {rel!r}')
        return n

    linked = copied = 0
    for orig, uname in manifest.items():
        target, source = dst / _safe(orig), dst / _safe(uname)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        try:
            os.link(source, target)
            linked += 1
        except OSError:
            shutil.copy2(source, target)
            copied += 1
    missing = [o for o in manifest if not (dst / _safe(o)).exists()]
    if missing:
        sys.exit(f'restore failed for {len(missing)} files, e.g. {missing[:3]}')
    print(f'restored {len(manifest):,} GT files (hard links {linked:,}, copies {copied:,})')


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p0 = sp.add_parser('pair')
    p0.add_argument('--hazy', required=True, help='folder of hazy images')
    p0.add_argument('--gt', required=True, help='folder of clear (GT) images')
    p0.add_argument('--dst', required=True, help='output folder; hazy/ and GT/ are created inside')
    p0.add_argument('--ids', default=None, help='keep only these leading image numbers, e.g. 1-25 or 31-35')
    p1 = sp.add_parser('pack')
    p1.add_argument('--src', required=True)
    p1.add_argument('--out', required=True)
    p2 = sp.add_parser('restore')
    p2.add_argument('--zip', required=True)
    p2.add_argument('--dst', required=True)
    a = ap.parse_args()
    if a.cmd == 'pair':
        pair(a.hazy, a.gt, a.dst, a.ids)
    elif a.cmd == 'pack':
        pack(a.src, a.out)
    else:
        restore(a.zip, a.dst)


if __name__ == '__main__':
    main()
