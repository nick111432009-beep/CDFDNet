# -*- coding: utf-8 -*-
"""Training loop: AdamW + cosine schedule, mixed precision, resumable checkpoints."""
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def build_optimizer(net, cfg):
    t = cfg['train']
    opt = torch.optim.AdamW(net.parameters(), lr=float(t['lr']))
    # T_max is bound to the actual budget so that the cosine schedule completes.
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(t['epochs']),
        eta_min=float(t['lr']) * float(t.get('eta_min_ratio', 0.01)))
    return opt, sch


def find_resume(ckpt_dir: Path, name: str):
    for cand in (ckpt_dir / f'{name}_last.pth', ckpt_dir / f'{name}.pth'):
        if cand.exists():
            return cand
    return None


def load_checkpoint(path, net, optimizer, scheduler, scaler, device):
    ck = torch.load(str(path), map_location=device, weights_only=False)
    net.load_state_dict({k[7:] if k.startswith('module.') else k: v
                         for k, v in ck['state_dict'].items()})
    optimizer.load_state_dict(ck['optimizer'])
    if 'lr_scheduler' in ck:
        old_tmax = ck['lr_scheduler'].get('T_max')
        new_tmax = getattr(scheduler, 'T_max', None)
        if old_tmax is not None and new_tmax is not None and old_tmax != new_tmax:
            raise RuntimeError(f'T_max mismatch: checkpoint {old_tmax} vs config epochs {new_tmax}. '
                               f'The cosine schedule cannot change budget mid-run; use --fresh.')
        scheduler.load_state_dict(ck['lr_scheduler'])
    else:
        for _ in range(ck['epoch'] + 1):
            scheduler.step()
    if 'scaler' in ck and scaler is not None and ck['scaler'] is not None:
        scaler.load_state_dict(ck['scaler'])
    return ck['epoch'] + 1, ck.get('best_psnr', 0.0), ck.get('history', None)


def load_init_weights(path, net, device, log_fn=print):
    """Initialize from pretrained weights (state_dict only; non-strict)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'init_weights not found: {path}')
    ck = torch.load(str(path), map_location=device, weights_only=False)
    sd = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    own = net.state_dict()
    fit = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    net.load_state_dict(fit, strict=False)
    log_fn(f'init_weights: {path.name} | loaded {len(fit)}/{len(own)} tensors')
    if not fit:
        raise RuntimeError('init_weights: no tensor matched the model')
    return {'file': path.name, 'loaded': len(fit), 'total': len(own)}


def _plain(cfg):
    return {k: v for k, v in cfg.items() if not str(k).startswith('_')}


def save_checkpoint(path, net, optimizer, scheduler, scaler, epoch, best_psnr, history, cfg):
    torch.save({'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None,
                'epoch': epoch, 'best_psnr': best_psnr,
                'history': history, 'config': _plain(cfg)}, str(path))


@torch.no_grad()
def validate(net, val_loader, device):
    """Validation PSNR at full resolution (values mapped from [-1,1] to [0,1])."""
    net.eval()
    meter = AverageMeter()
    for batch in val_loader:
        src = batch['source'].to(device)
        tgt = batch['target'].to(device)
        out = net(src).clamp_(-1, 1)
        mse = F.mse_loss(out * 0.5 + 0.5, tgt * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        meter.update((10 * torch.log10(1 / mse)).mean().item(), src.size(0))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meter.avg


def train(cfg, net, loaders, criterion, device, log_fn=print, max_hours=None):
    """Main training loop; returns a summary dict."""
    t = cfg['train']
    _, _, train_loader, val_loader = loaders
    l1, l1_w, cr, cr_w = criterion
    use_amp = bool(t.get('amp', True)) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    optimizer, scheduler = build_optimizer(net, cfg)

    ckpt_dir = Path(cfg['io']['ckpt_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = cfg['name']
    start_epoch, best_psnr = 0, 0.0
    history = {'epoch': [], 'loss': [], 'psnr': []}

    resume = find_resume(ckpt_dir, name) if not t.get('fresh_start', False) else None
    if resume:
        start_epoch, best_psnr, h = load_checkpoint(resume, net, optimizer, scheduler, scaler, device)
        history = h or history
        log_fn(f'resumed from epoch {start_epoch} (best PSNR {best_psnr:.2f})')
    elif t.get('init_weights'):
        load_init_weights(t['init_weights'], net, device, log_fn)

    total_epochs = int(t['epochs'])
    t0 = time.time()
    it_rates, stopped_early = [], False
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, total_epochs):
        net.train()
        meter, n_step, ts = AverageMeter(), 0, time.time()
        nonfinite_steps = 0
        buf_snapshot = {k: v.detach().clone() for k, v in net.named_buffers()}
        for batch in train_loader:
            src = batch['source'].to(device, non_blocking=True)
            tgt = batch['target'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = net(src)
                loss = l1(out, tgt) * l1_w + cr(out, tgt, src) * cr_w
            if not torch.isfinite(loss):
                # Skip the step and roll back BN statistics; stop after 50 in a row.
                nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for k, v in net.named_buffers():
                        v.copy_(buf_snapshot[k])
                log_fn(f'warning: non-finite loss at epoch {epoch} step {n_step} (skipped, {nonfinite_steps} in a row)')
                if nonfinite_steps >= 50:
                    raise RuntimeError('training diverged: 50 consecutive non-finite losses')
                continue
            nonfinite_steps = 0
            meter.update(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n_step += 1
        it_s = n_step / max(time.time() - ts, 1e-9)
        it_rates.append(it_s)
        scheduler.step()
        history['epoch'].append(epoch)
        history['loss'].append(meter.avg)

        line = (f'epoch {epoch:4d}/{total_epochs} | loss {meter.avg:.4f}'
                f' | lr {optimizer.param_groups[0]["lr"]:.2e} | {it_s:.2f} it/s')
        if epoch % int(t.get('eval_freq', 1)) == 0:
            psnr = validate(net, val_loader, device)
            if not math.isfinite(psnr):
                raise RuntimeError(f'training diverged: non-finite validation PSNR at epoch {epoch}')
            history['psnr'].append((epoch, psnr))
            line += f' | val PSNR {psnr:.2f}'
            if psnr > best_psnr:
                best_psnr = psnr
                save_checkpoint(ckpt_dir / f'{name}.pth', net, optimizer, scheduler,
                                scaler, epoch, best_psnr, history, cfg)
                line += '  (best)'
        if t.get('save_every_epoch', True):
            save_checkpoint(ckpt_dir / f'{name}_last.pth', net, optimizer, scheduler,
                            scaler, epoch, best_psnr, history, cfg)
        if torch.cuda.is_available():
            line += f' | peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB'
        log_fn(line)

        if max_hours and (time.time() - t0) / 3600 >= max_hours:
            log_fn(f'reached the {max_hours} h limit at epoch {epoch}; rerun the same command to resume.')
            stopped_early = True
            break

    elapsed = time.time() - t0
    return {'name': name, 'best_psnr': best_psnr, 'hours': elapsed / 3600,
            'it_s': sum(it_rates) / max(len(it_rates), 1),
            'peak_gb': torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0,
            'last_epoch': (history['epoch'][-1] + 1) if history['epoch'] else start_epoch,
            'total_epochs': total_epochs, 'stopped_early': stopped_early, 'history': history}
