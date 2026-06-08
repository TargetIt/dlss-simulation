"""
DLSS-Style Temporal Super Resolution — Core Building Blocks

Three architectures mapping to DLSS evolution:
  Phase 1 — Early Fusion (DLSS 2 早期): naive stacking of adjacent frames
  Phase 2 — Warp-then-Fuse (DLSS 2/3): backward-warp alignment before fusion  
  Phase 3 — Recurrent Feedback (DLSS 4 风格): HR-space recurrent feedback loop

Based on egan-wu/Super-Resolution, restructured for clarity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# ICNR Initialization — prevents PixelShuffle checkerboard artifacts
# ═══════════════════════════════════════════════════════════════════════════

def icnr_init(conv, scale_factor=2):
    """Initialize conv before PixelShuffle with ICNR to prevent checkerboard."""
    out_ch, in_ch, kH, kW = conv.weight.shape
    sub_out = out_ch // (scale_factor ** 2)
    tmp = torch.empty(sub_out, in_ch, kH, kW)
    nn.init.kaiming_normal_(tmp, nonlinearity="relu")
    kernel = tmp.repeat_interleave(scale_factor ** 2, dim=0)
    conv.weight.data.copy_(kernel)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


# ═══════════════════════════════════════════════════════════════════════════
# Shared blocks
# ═══════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Standard residual block with PReLU activation."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        r = self.conv1(x)
        r = self.bn1(r)
        r = self.prelu(r)
        r = self.conv2(r)
        r = self.bn2(r)
        return x + r


def make_pixelshuffle_upsampler(in_ch, scale_factor=4, smooth=True):
    """Build PixelShuffle upsampler with anti-checkerboard smoothing convs.
    
    For scale_factor=4: two 2× PixelShuffle stages.
    Each stage: pre-shuffle conv (ICNR) → PixelShuffle → smooth conv → PReLU
    """
    stages = []
    num_stages = int(torch.log2(torch.tensor(scale_factor)).item())
    for _ in range(num_stages):
        stages.append(nn.Conv2d(in_ch, in_ch * 4, 3, padding=1))  # pre-shuffle
        stages.append(nn.PixelShuffle(2))
        if smooth:
            stages.append(nn.Conv2d(in_ch, in_ch, 3, padding=1))  # post-smooth
            stages.append(nn.PReLU())
    return nn.Sequential(*stages)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Early Fusion (最简单：直接拼接两帧)
# ═══════════════════════════════════════════════════════════════════════════

class EarlyFusionTSR(nn.Module):
    """Phase 1: Naively concatenate LR(t-1) + LR(t) as 6-channel input.
    
    This is the simplest temporal approach — no motion compensation.
    The network must learn to handle misalignment on its own.
    """
    def __init__(self, scale_factor=4, num_res_blocks=16, hidden_channels=64):
        super().__init__()
        ch = hidden_channels
        
        self.conv1 = nn.Conv2d(6, ch, 9, padding=4)  # 6ch = 2 frames × RGB
        self.prelu1 = nn.PReLU()
        self.res_blocks = nn.Sequential(*[ResidualBlock(ch) for _ in range(num_res_blocks)])
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.upsample = make_pixelshuffle_upsampler(ch, scale_factor)
        self.conv3 = nn.Conv2d(ch, 3, 9, padding=4)
        self._init_icnr()

    def _init_icnr(self):
        modules = list(self.upsample)
        for i, m in enumerate(modules):
            if isinstance(m, nn.Conv2d) and i+1 < len(modules) and isinstance(modules[i+1], nn.PixelShuffle):
                icnr_init(m, scale_factor=2)

    def forward(self, x_prev, x_curr):
        """x_prev, x_curr: (B, 3, H, W) — adjacent LR frames"""
        x = torch.cat([x_prev, x_curr], dim=1)   # (B, 6, H, W)
        out1 = self.prelu1(self.conv1(x))
        res = self.res_blocks(out1)
        out = out1 + self.bn2(self.conv2(res))
        out = self.upsample(out)
        return self.conv3(out)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Warp-then-Fuse (DLSS 核心技巧：光流对齐后融合)
# ═══════════════════════════════════════════════════════════════════════════

class WarpFuseTSR(nn.Module):
    """Phase 2: Backward-warp LR(t-1) using known flow, then fuse.
    
    Input: [warped_prev (3ch), lr_curr (3ch), occlusion_mask (1ch)] = 7ch.
    The mask tells the network where the warp was invalid (out-of-bounds).
    
    This is the core DLSS trick: geometrically-aligned history is much more
    useful than raw stacking.
    """
    def __init__(self, scale_factor=4, num_res_blocks=16, hidden_channels=64, pad_mode="border"):
        super().__init__()
        self.pad_mode = pad_mode
        ch = hidden_channels
        
        self.conv1 = nn.Conv2d(7, ch, 9, padding=4)  # 7ch = warped(3) + curr(3) + mask(1)
        self.prelu1 = nn.PReLU()
        self.res_blocks = nn.Sequential(*[ResidualBlock(ch) for _ in range(num_res_blocks)])
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.upsample = make_pixelshuffle_upsampler(ch, scale_factor)
        self.conv3 = nn.Conv2d(ch, 3, 9, padding=4)
        self._init_icnr()

    def _init_icnr(self):
        modules = list(self.upsample)
        for i, m in enumerate(modules):
            if isinstance(m, nn.Conv2d) and i+1 < len(modules) and isinstance(modules[i+1], nn.PixelShuffle):
                icnr_init(m, scale_factor=2)

    def forward(self, x_prev, x_curr, flow, warp_fn):
        """x_prev, x_curr: (B,3,H,W) | flow: (B,2) or (B,2,H,W) | warp_fn: backward_warp"""
        warped_prev, mask = warp_fn(x_prev, flow, padding_mode=self.pad_mode)
        x = torch.cat([warped_prev, x_curr, mask], dim=1)  # (B, 7, H, W)
        out1 = self.prelu1(self.conv1(x))
        res = self.res_blocks(out1)
        out = out1 + self.bn2(self.conv2(res))
        out = self.upsample(out)
        return self.conv3(out)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Recurrent Feedback (DLSS 4 风格：HR空间循环反馈)
# ═══════════════════════════════════════════════════════════════════════════

class RecurrentTSR(nn.Module):
    """Phase 3: True recurrent feedback at HR resolution.
    
    Architecture:
      LR Branch  → extract features at LR → PixelShuffle → 64ch @ HR
      Hist Branch → warp HR(t-1) in HR space → extract features → 64ch @ HR
      Fusion     → concat (128ch) → refine → HR output
    
    At inference, HR(t-1) is the model's own previous output (true recurrence).
    During training, HR(t-1) is the GT frame (teacher forcing).
    Halton jitter gives each frame a different sub-pixel view, and the
    recurrent loop accumulates these into sharper HR images.
    """
    def __init__(self, scale_factor=4, lr_res_blocks=8, hist_res_blocks=4,
                 fuse_res_blocks=4, hidden_channels=64, pad_mode="border"):
        super().__init__()
        self.scale_factor = scale_factor
        self.pad_mode = pad_mode
        ch = hidden_channels

        # LR branch (at LR resolution)
        self.lr_entry = nn.Sequential(nn.Conv2d(3, ch, 9, padding=4), nn.PReLU())
        self.lr_res = nn.Sequential(*[ResidualBlock(ch) for _ in range(lr_res_blocks)])
        self.lr_post = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch))
        self.lr_up = make_pixelshuffle_upsampler(ch, scale_factor)

        # History branch (at HR resolution) — input: warped HR(t-1) + mask = 4ch
        self.hist_entry = nn.Sequential(nn.Conv2d(4, ch, 3, padding=1), nn.PReLU())
        self.hist_res = nn.Sequential(*[ResidualBlock(ch) for _ in range(hist_res_blocks)])

        # Fusion (at HR resolution)
        self.fuse_entry = nn.Sequential(nn.Conv2d(ch * 2, ch, 3, padding=1), nn.PReLU())
        self.fuse_res = nn.Sequential(*[ResidualBlock(ch) for _ in range(fuse_res_blocks)])
        self.output = nn.Conv2d(ch, 3, 9, padding=4)
        self._init_icnr()

    def _init_icnr(self):
        modules = list(self.lr_up)
        for i, m in enumerate(modules):
            if isinstance(m, nn.Conv2d) and i+1 < len(modules) and isinstance(modules[i+1], nn.PixelShuffle):
                icnr_init(m, scale_factor=2)

    def forward(self, lr_curr, hr_prev, flow_lr, warp_fn):
        """
        lr_curr  : (B, 3, H_lr, W_lr) — current low-res frame
        hr_prev  : (B, 3, H_hr, W_hr) — previous HR frame
        flow_lr  : (B,2) rigid or (B,2,H_lr,W_lr) dense — backward flow in LR space
        """
        s = self.scale_factor

        # Scale flow from LR space to HR space
        if flow_lr.dim() == 2:
            flow_hr = flow_lr * s
        else:
            flow_hr = F.interpolate(flow_lr, scale_factor=s, mode="bilinear", align_corners=False) * s

        # Warp HR(t-1) in HR space
        warped_hr, mask = warp_fn(hr_prev, flow_hr, padding_mode=self.pad_mode)

        # LR branch → upsample to HR feature space
        x = self.lr_entry(lr_curr)
        x = x + self.lr_post(self.lr_res(x))
        x = self.lr_up(x)  # (B, ch, H_hr, W_hr)

        # History branch
        hist_in = torch.cat([warped_hr, mask], dim=1)  # (B, 4, H_hr, W_hr)
        h = self.hist_entry(hist_in)
        h = self.hist_res(h)  # (B, ch, H_hr, W_hr)

        # Fusion
        f = self.fuse_entry(torch.cat([x, h], dim=1))  # (B, ch, H_hr, W_hr)
        f = self.fuse_res(f)
        return self.output(f)  # (B, 3, H_hr, W_hr)
