# -*- coding: utf-8 -*-
"""Paired image loader (GT / hazy folders with identical file names).

Value chain: file [0,255] -> /255 -> [0,1] -> *2-1 -> [-1,1] -> network.
"""
import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

VALID_EXTS = ('.jpg', '.png', '.jpeg', '.bmp')


def read_img(filename):
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f'cannot read image: {filename}')
    return img[:, :, ::-1].astype('float32') / 255.0


def write_img(filename, img):
    cv2.imwrite(filename, np.round(img[:, :, ::-1].copy() * 255.0).astype('uint8'))


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).copy()


def augment(imgs=(), size=256, edge_decay=0., only_h_flip=False):
    """Random crop (with a small probability of snapping to the image border),
    horizontal flip and 90-degree rotations, applied identically to all images."""
    imgs = list(imgs)
    H, W, _ = imgs[0].shape
    Hc = Wc = size
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H - Hc)
    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W - Wc)
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:Hs + Hc, Ws:Ws + Wc, :]
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)
    if not only_h_flip:
        rot = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot, (0, 1))
    return imgs


def align(imgs=(), size=256):
    """Central crop (used for validation on very large images)."""
    imgs = list(imgs)
    H, W, _ = imgs[0].shape
    Hs, Ws = (H - size) // 2, (W - size) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:Hs + size, Ws:Ws + size, :]
    return imgs


class PairLoader(Dataset):
    """Loads <root>/<sub_dir>/GT and <root>/<sub_dir>/hazy; file names are matched."""

    def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False,
                 cache=False):
        assert mode in ('train', 'valid', 'test')
        self.mode, self.size = mode, size
        self.edge_decay, self.only_h_flip = edge_decay, only_h_flip
        self._cache = {} if cache else None
        self.root_dir = os.path.join(data_dir, sub_dir)
        gt = set(f for f in os.listdir(os.path.join(self.root_dir, 'GT'))
                 if f.lower().endswith(VALID_EXTS))
        hz = set(f for f in os.listdir(os.path.join(self.root_dir, 'hazy'))
                 if f.lower().endswith(VALID_EXTS))
        self.img_names = sorted(gt & hz)
        if not self.img_names:
            raise RuntimeError(f'no paired images found under {self.root_dir}')

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        n = self.img_names[idx]
        if self._cache is not None:
            if n not in self._cache:
                self._cache[n] = (read_img(os.path.join(self.root_dir, 'hazy', n)),
                                  read_img(os.path.join(self.root_dir, 'GT', n)))
            c_src, c_tgt = self._cache[n]
            src, tgt = c_src * 2 - 1, c_tgt * 2 - 1
        else:
            src = read_img(os.path.join(self.root_dir, 'hazy', n)) * 2 - 1
            tgt = read_img(os.path.join(self.root_dir, 'GT', n)) * 2 - 1
        if self.mode == 'train':
            src, tgt = augment([src, tgt], self.size, self.edge_decay, self.only_h_flip)
        if self.mode == 'valid':
            src, tgt = align([src, tgt], self.size)
        return {'source': hwc_to_chw(src), 'target': hwc_to_chw(tgt), 'filename': n}


def make_synthetic_dataset(root, n_train=16, n_test=4, H=460, W=620, seed=0):
    """Write a tiny synthetic paired dataset (atmospheric scattering model) for
    smoke tests; the folder layout is the same as the real datasets."""
    rng = np.random.default_rng(seed)
    for sub, n in (('train', n_train), ('test', n_test)):
        for d in ('GT', 'hazy'):
            os.makedirs(os.path.join(root, sub, d), exist_ok=True)
        for i in range(n):
            small = rng.random((max(H // 20, 2), max(W // 20, 2), 3), dtype=np.float32)
            gt = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)
            t_small = rng.random((max(H // 40, 2), max(W // 40, 2), 1), dtype=np.float32) * 0.6 + 0.3
            t = cv2.resize(t_small, (W, H), interpolation=cv2.INTER_CUBIC)[..., None].clip(0.1, 1)
            A = float(rng.random() * 0.2 + 0.8)
            hazy = (gt * t + A * (1 - t)).clip(0, 1)
            name = f'{i:05d}.png'
            write_img(os.path.join(root, sub, 'GT', name), gt)
            write_img(os.path.join(root, sub, 'hazy', name), hazy)
    return root


def _worker_init(worker_id):
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_loaders(cfg):
    """Build (train_set, val_set, train_loader, val_loader) from the config."""
    import torch
    from torch.utils.data import DataLoader
    d, t = cfg['data'], cfg['train']
    root = d['root']
    cache = bool(d.get('cache', False))
    train_set = PairLoader(root, d.get('train_sub', 'train'), 'train',
                           d['patch_size'], d['edge_decay'], d['only_h_flip'], cache=cache)
    # Very large images (I-HAZE): validate on a central crop during training.
    val_mode = 'valid' if d.get('val_center_crop', False) else 'test'
    val_set = PairLoader(root, d.get('test_sub', 'test'), val_mode, d['patch_size'], cache=cache)
    nw = d.get('num_workers', 2)
    gen = torch.Generator()
    gen.manual_seed(int(t.get('seed', 42)))
    train_loader = DataLoader(train_set, batch_size=t['batch_size'], shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=nw > 0, generator=gen,
                              worker_init_fn=_worker_init)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=0, pin_memory=True)
    return train_set, val_set, train_loader, val_loader
