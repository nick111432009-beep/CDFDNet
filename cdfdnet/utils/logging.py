# -*- coding: utf-8 -*-
"""Append one summary row per training run to a CSV file."""
import csv
import time
from pathlib import Path

COLUMNS = ['date', 'name', 'config', 'best_psnr', 'ssim', 'epochs', 'epochs_done', 'batch',
           'patch', 'lr', 'seed', 'params', 'gpu', 'it_s', 'peak_gb', 'hours', 'note']


def append_row(csv_path, cfg, result, extra=None):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    t, d = cfg.get('train', {}), cfg.get('data', {})
    extra = extra or {}
    row = {'date': time.strftime('%Y-%m-%d %H:%M'), 'name': cfg.get('name', ''),
           'config': Path(cfg.get('_config_path', '')).name,
           'best_psnr': round(result.get('best_psnr', 0), 3), 'ssim': extra.get('ssim', ''),
           'epochs': t.get('epochs'), 'epochs_done': result.get('last_epoch', ''),
           'batch': t.get('batch_size'), 'patch': d.get('patch_size'), 'lr': t.get('lr'),
           'seed': t.get('seed'), 'params': extra.get('params', ''), 'gpu': extra.get('gpu', ''),
           'it_s': round(result.get('it_s', 0), 3), 'peak_gb': round(result.get('peak_gb', 0), 2),
           'hours': round(result.get('hours', 0), 3),
           'note': extra.get('note', 'stopped early' if result.get('stopped_early') else '')}
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            w.writeheader()
        w.writerow(row)
    return csv_path
