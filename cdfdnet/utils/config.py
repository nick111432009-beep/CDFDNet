# -*- coding: utf-8 -*-
"""YAML config loading with single-parent inheritance (`_base: file.yaml`)."""
import copy
import pathlib
import yaml


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path, _seen=None):
    path = pathlib.Path(path).resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f'circular config inheritance: {path}')
    _seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    base_ref = raw.pop('_base', None)
    cfg = load_config(path.parent / base_ref, _seen) if base_ref else {}
    cfg = _merge(cfg, raw)
    cfg['_config_path'] = str(path)
    return cfg


def apply_overrides(cfg, pairs):
    """Apply `a.b.c=value` overrides (values parsed as YAML)."""
    for p in pairs or []:
        k, v = p.split('=', 1)
        cur = cfg
        parts = k.split('.')
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = yaml.safe_load(v)
    return cfg


def describe(cfg):
    m, t, d = cfg.get('model', {}), cfg.get('train', {}), cfg.get('data', {})
    return '\n'.join([
        f"run      : {cfg.get('name', '(unnamed)')}",
        f"model    : dims={m.get('embed_dims')} depths={m.get('depths')} detail_conv={m.get('detail_conv', True)}",
        f"data     : {d.get('name')} root={d.get('root')} patch={d.get('patch_size')} batch={t.get('batch_size')}",
        f"train    : epochs={t.get('epochs')} lr={t.get('lr')} cosine T_max={t.get('epochs')} amp={t.get('amp', True)}",
        f"init     : {t.get('init_weights') or 'random'}",
    ])
