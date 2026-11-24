# -*- coding: utf-8 -*-
"""Building blocks of CDFDNet.

Module naming (paper, Section III -> step name -> class):
  Detail convolution (DConv)                 -          DetailConv
  Batch normalization (BN)                   'BN'       nn.BatchNorm2d
  Composite residual attention (CRA)         'CRA'      CompositeResidualAttention
  Tri-scale depthwise convolution (TSDW)     'TSDW'     TriScaleDepthwiseConv
  Tri-scale dilated detail convolution (TSDD) 'TSDD'    TriScaleDilatedDetailConv
  Large-kernel and strip convolution (LKS)   'LKS'      LargeKernelStripConv
  Dual-scale detail convolution (DSD)        'DSD'      DualScaleDetailConv
  Frequency modulation (FM)                  'FM'       FrequencyModulation
  Feed-forward network (FFN)                 'FFN'      1x1 -> GELU -> 1x1
  Block (CDB / CDFB)                         -          CDFDBlock

The attribute names inside the modules (see STEP_TABLE) define the state_dict
keys of the released weights; do not rename them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad(kernel: int, dilation: int = 1) -> int:
    """Padding that keeps the spatial size for a (dilated) kernel."""
    return dilation * (kernel - 1) // 2


def _act(name):
    if name in ('gelu', None):
        return nn.GELU()
    if name == 'relu':
        return nn.ReLU()
    if name == 'prelu':
        return nn.PReLU()
    raise ValueError(f'unknown activation: {name}')


def _seq_convs(in_ch, out_ch, kernel, dilation=1, act='prelu', groups=1):
    """Conv2d (+ activation) as an nn.Sequential; `groups='depthwise'` is allowed."""
    g = in_ch if groups == 'depthwise' else groups
    layers = [nn.Conv2d(in_ch, out_ch, kernel, dilation=dilation,
                        padding=_pad(kernel, dilation), groups=g)]
    if act not in (None, 'none'):
        layers.append(_act(act))
    return nn.Sequential(*layers)


def _mlp(dim, expansion=2, act='gelu'):
    """C -> expansion*C -> C with 1x1 convolutions (per-pixel channel mixing)."""
    return nn.Sequential(nn.Conv2d(dim, dim * expansion, 1), _act(act),
                         nn.Conv2d(dim * expansion, dim, 1))


def _mlp_wide(in_ch, out_ch, expansion=2, act='gelu'):
    """Fusion after concatenation: in_ch -> expansion*out_ch -> out_ch."""
    return nn.Sequential(nn.Conv2d(in_ch, out_ch * expansion, 1), _act(act),
                         nn.Conv2d(out_ch * expansion, out_ch, 1))


# ---- detail convolution ----

class DetailConv(nn.Module):
    """Detail convolution (re-parameterizable; derived from the DEConv of DEA-Net).

    Training form: one vanilla 3x3 convolution in parallel with four difference
    convolutions (central, angular, horizontal, vertical). The difference kernels
    are obtained from raw parameters by fixed linear maps, so the five kernels can
    be summed into a single 3x3 kernel; this implementation always convolves with
    the merged kernel, which is mathematically identical to summing the five
    branch outputs. A dilation rate d > 1 applies the same kernel on a dilated
    sampling grid (the extension used by the tri-scale dilated branches).

    All difference parameters and biases are zero-initialized, so at
    initialization the layer equals a standard 3x3 convolution. `merged_weight_bias`
    returns the deployed kernel; `scripts/fold_deploy.py` replaces every
    DetailConv by an equivalent nn.Conv2d.
    """
    PERM_AD = [3, 0, 1, 6, 4, 2, 7, 8, 5]

    def __init__(self, in_ch, out_ch, dilation=1, bias=True):
        super().__init__()
        self.dilation = int(dilation)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=self.dilation,
                              dilation=self.dilation, bias=bias)      # vanilla branch
        self.w_cd = nn.Parameter(torch.zeros(out_ch, in_ch, 3, 3))    # central difference
        self.w_ad = nn.Parameter(torch.zeros(out_ch, in_ch, 3, 3))    # angular difference
        self.w_hd = nn.Parameter(torch.zeros(out_ch, in_ch, 3))       # horizontal difference
        self.w_vd = nn.Parameter(torch.zeros(out_ch, in_ch, 3))       # vertical difference
        self.has_bias = bool(bias)
        if bias:
            self.b_cd = nn.Parameter(torch.zeros(out_ch))
            self.b_ad = nn.Parameter(torch.zeros(out_ch))
            self.b_hd = nn.Parameter(torch.zeros(out_ch))
            self.b_vd = nn.Parameter(torch.zeros(out_ch))

    def merged_weight_bias(self):
        o, i = self.w_cd.shape[:2]
        w = self.conv.weight.reshape(o, i, 9)
        cd_raw = self.w_cd.reshape(o, i, 9)
        cd = cd_raw.clone()
        cd[:, :, 4] = cd_raw[:, :, 4] - cd_raw.sum(2)            # central difference
        ad_raw = self.w_ad.reshape(o, i, 9)
        ad = ad_raw - ad_raw[:, :, self.PERM_AD]                  # angular difference
        hd = self.w_hd.new_zeros(o, i, 9)                         # horizontal difference
        hd[:, :, [0, 3, 6]] = self.w_hd
        hd[:, :, [2, 5, 8]] = -self.w_hd
        vd = self.w_vd.new_zeros(o, i, 9)                         # vertical difference
        vd[:, :, [0, 1, 2]] = self.w_vd
        vd[:, :, [6, 7, 8]] = -self.w_vd
        wm = (w + cd + ad + hd + vd).reshape(o, i, 3, 3)
        bm = None
        if self.has_bias:
            bm = self.conv.bias + self.b_cd + self.b_ad + self.b_hd + self.b_vd
        return wm, bm

    def forward(self, x):
        wm, bm = self.merged_weight_bias()
        return F.conv2d(x, wm, bm, padding=self.dilation, dilation=self.dilation)


# ---- block modules ----

class CompositeResidualAttention(nn.Module):
    """y = x + x * A_c(x) * A_s(x) with channel and spatial score maps."""

    def __init__(self, dim, reduction=8, spatial_kernel=3, residual=True):
        super().__init__()
        self.residual = residual
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1), nn.ReLU(),
            nn.Conv2d(dim // reduction, dim, 1), nn.Sigmoid())
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, dim, spatial_kernel, padding=_pad(spatial_kernel),
                      groups=dim, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(dim, 1, 1), nn.Sigmoid())

    def forward(self, x):
        ca = self.channel_attn(x)
        sa = self.spatial_attn(x)
        out = x * ca * sa          # keep this left-to-right order (bit-exact with the weights)
        return x + out if self.residual else out


class TriScaleDepthwiseConv(nn.Module):
    """Chained 3x3 -> 5x5 -> 7x7 depthwise convolutions, each with a residual,
    followed by concatenation of the four tensors and a 1x1 -> GELU -> 1x1 fusion."""

    def __init__(self, dim, kernels=(3, 5, 7), expansion=2, groups='depthwise', act='prelu'):
        super().__init__()
        k = list(kernels)
        self.conv1 = _seq_convs(dim, dim, k[0], 1, act=act, groups=groups)
        self.conv2 = _seq_convs(dim, dim, k[1], 1, act=act, groups=groups)
        self.conv3 = _seq_convs(dim, dim, k[2], 1, act=act, groups=groups)
        self.fusion = _mlp_wide(dim * 4, dim, expansion)

    def forward(self, x):
        x1 = self.conv1(x) + x
        x2 = self.conv2(x1) + x1
        x3 = self.conv3(x2) + x2
        return x + self.fusion(torch.cat([x, x1, x2, x3], dim=1))


class TriScaleDilatedDetailConv(nn.Module):
    """Same topology as TriScaleDepthwiseConv, but the three branches are 3x3
    convolutions with dilation rates 5, 1 and 3. With `detail_conv=True` each
    branch is a DetailConv (training form); with `detail_conv=False` each branch
    is the folded standard convolution (deployed form)."""

    def __init__(self, dim, dilations=(5, 1, 3), expansion=2, act='prelu', detail_conv=True):
        super().__init__()
        ds = list(dilations)
        if detail_conv:
            def branch(d):
                return nn.Sequential(DetailConv(dim, dim, dilation=d), _act(act))
            self.conv1, self.conv2, self.conv3 = branch(ds[0]), branch(ds[1]), branch(ds[2])
        else:
            self.conv1 = _seq_convs(dim, dim, 3, ds[0], act=act)
            self.conv2 = _seq_convs(dim, dim, 3, ds[1], act=act)
            self.conv3 = _seq_convs(dim, dim, 3, ds[2], act=act)
        self.merge = _mlp_wide(dim * 4, dim, expansion)

    def forward(self, x):
        x1 = self.conv1(x) + x
        x2 = self.conv2(x1) + x1
        x3 = self.conv3(x2) + x2
        return x + self.merge(torch.cat([x, x1, x2, x3], dim=1))


class LargeKernelStripConv(nn.Module):
    """Parallel depthwise 3x3, kxk (default 7x7), 1xs and sx1 (default s=11)
    kernels, summed and passed through GELU."""

    def __init__(self, dim, kernel=3, big=7, strip=11, zero_init=True):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel, padding=_pad(kernel), groups=dim)
        self.conv_big = nn.Conv2d(dim, dim, big, padding=_pad(big), groups=dim)
        self.conv_h = nn.Conv2d(dim, dim, (1, strip), padding=(0, strip // 2), groups=dim)
        self.conv_v = nn.Conv2d(dim, dim, (strip, 1), padding=(strip // 2, 0), groups=dim)
        if zero_init:
            for m in (self.conv_big, self.conv_h, self.conv_v):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.conv_big(x) + self.conv_h(x) + self.conv_v(x))


class DualScaleDetailConv(nn.Module):
    """A dilated (d=4) standard 3x3 convolution in parallel with a d=1 detail
    convolution; the sum is fused by 1x1 -> GELU -> 1x1 and added to the input."""

    def __init__(self, dim, dilations=(4, 1), expansion=2, detail_conv=True):
        super().__init__()
        d0, d1 = dilations
        self.conv1 = _seq_convs(dim, dim, 3, d0, act=None)
        if detail_conv:
            self.conv2 = DetailConv(dim, dim, dilation=d1)
        else:
            self.conv2 = _seq_convs(dim, dim, 3, d1, act=None)
        self.merge = _mlp(dim, expansion)

    def forward(self, x):
        return x + self.merge(self.conv1(x) + self.conv2(x))


class FrequencyModulation(nn.Module):
    """Frequency modulation (stage 3 only).

    x_b = BN(x); X = rFFT2(x_b) (orthonormal, fp32); the real and imaginary parts
    are mixed independently by 1x1 -> GELU -> 1x1; x_f = irFFT2; the output is
    x + GELU(1x1([x_b, x_f])).
    """

    def __init__(self, dim, expansion=1):
        super().__init__()
        e = int(dim * expansion)
        self.norm = nn.BatchNorm2d(dim)
        self.re = nn.Sequential(nn.Conv2d(dim, e, 1), nn.GELU(), nn.Conv2d(e, dim, 1))
        self.im = nn.Sequential(nn.Conv2d(dim, e, 1), nn.GELU(), nn.Conv2d(e, dim, 1))
        self.fuse = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GELU())

    def forward(self, x):
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            xb = self.norm(xf)
            H, W = xb.shape[-2:]
            X = torch.fft.rfft2(xb, norm='ortho')
            Xm = torch.complex(self.re(X.real), self.im(X.imag))
            x_freq = torch.fft.irfft2(Xm, s=(H, W), norm='ortho')
            out = xf + self.fuse(torch.cat([xb, x_freq], dim=1))
        return out.to(in_dtype)


# ---- block ----

# step name (paper abbreviation) -> (attribute name, constructor).  The step
# names follow Section III-B of the paper; the attribute names are part of the
# state_dict key layout of the released weights and must not be changed.
STEP_TABLE = {
    'BN':   ('norm1',       lambda dim, c: nn.BatchNorm2d(dim)),
    'CRA':  ('attn',        lambda dim, c: CompositeResidualAttention(dim, **c)),
    'TSDW': ('multi_scale', lambda dim, c: TriScaleDepthwiseConv(dim, **c)),
    'TSDD': ('tri_scale',   lambda dim, c: TriScaleDilatedDetailConv(dim, **c)),
    'LKS':  ('mslk_conv',   lambda dim, c: LargeKernelStripConv(dim, **c)),
    'DSD':  ('dual_scale',  lambda dim, c: DualScaleDetailConv(dim, **c)),
    'FM':   ('hafm_freq',   lambda dim, c: FrequencyModulation(dim, **c)),
    'FFN':  ('mlp',         lambda dim, c: _mlp(dim, **c)),
}

CDB_STEPS = ['BN', 'CRA', 'TSDW', 'TSDD', 'LKS', 'DSD', 'FFN']
CDFB_STEPS = ['BN', 'CRA', 'TSDW', 'TSDD', 'LKS', 'DSD', 'FM', 'FFN']


class CDFDBlock(nn.Module):
    """Composite detail block (CDB) or composite detail-frequency block (CDFB):
    the modules in `steps` are applied in order with a block-level residual."""

    def __init__(self, dim, steps=None, modules_cfg=None):
        super().__init__()
        steps = list(steps) if steps else list(CDB_STEPS)
        modules_cfg = modules_cfg or {}
        for s in steps:
            if s not in STEP_TABLE:
                raise ValueError(f'unknown step {s!r}; available: {list(STEP_TABLE)}')
            attr, ctor = STEP_TABLE[s]
            setattr(self, attr, ctor(dim, dict(modules_cfg.get(s, {}))))
        self._attrs = [STEP_TABLE[s][0] for s in steps]

    def forward(self, x):
        identity = x
        for a in self._attrs:
            x = getattr(self, a)(x)
        return x + identity
