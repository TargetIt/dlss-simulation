"""
Frame Generation Module — Optical Flow-based Frame Interpolation

Inspired by RIFE (Real-time Intermediate Flow Estimation).
For DLSS Frame Generation simulation: takes two rendered frames,
estimates optical flow, warps, and synthesises intermediate frames.

Two modes:
  1. Farneback (OpenCV) — for real video, no training needed
  2. IFNet-lite — small CNN flow estimator (learnable)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Backward warp (reused from super_resolution, duplicated for independence)
# ═══════════════════════════════════════════════════════════════════════════

def backward_warp(frame, flow, padding_mode="border"):
    """Backward-warp `frame` using flow field.
    
    flow: (B, 2, H, W) dense per-pixel displacement in pixel units.
    """
    B, C, H, W = frame.shape
    device = frame.device

    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    dx_norm = (2.0 * flow[:, 0] / max(W - 1, 1)).unsqueeze(-1)
    dy_norm = (2.0 * flow[:, 1] / max(H - 1, 1)).unsqueeze(-1)
    flow_norm = torch.cat([dx_norm, dy_norm], dim=-1)

    sample_grid = base_grid + flow_norm

    in_x = sample_grid[..., 0].abs() <= 1.0
    in_y = sample_grid[..., 1].abs() <= 1.0
    mask = (in_x & in_y).float().unsqueeze(1)

    warped = F.grid_sample(frame, sample_grid, mode="bilinear",
                           padding_mode=padding_mode, align_corners=True)
    return warped, mask


# ═══════════════════════════════════════════════════════════════════════════
# FlowNet-lite — small CNN for bidirectional flow estimation
# ═══════════════════════════════════════════════════════════════════════════

class FlowNetLite(nn.Module):
    """Lightweight flow estimator: 6ch input → bidirectional flow + fusion weights.
    
    Input: concat(img0, img1) = 6 channels
    Output: flow_0→1, flow_1→0, fusion_map (per-pixel blend weight)
    
    Uses a simple encoder-decoder with skip connections.
    """
    def __init__(self, base_ch=16):
        super().__init__()
        c = base_ch

        # Encoder (downsample)
        self.enc1 = nn.Sequential(nn.Conv2d(6, c, 3, padding=1), nn.PReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(c, c*2, 3, stride=2, padding=1), nn.PReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(c*2, c*4, 3, stride=2, padding=1), nn.PReLU())

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c*4, c*4, 3, padding=1), nn.PReLU(),
            nn.Conv2d(c*4, c*4, 3, padding=1), nn.PReLU(),
        )

        # Decoder (upsample)
        self.dec3 = nn.Sequential(nn.Conv2d(c*4 + c*2, c*2, 3, padding=1), nn.PReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(c*2 + c, c, 3, padding=1), nn.PReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.PReLU())

        # Output heads
        self.flow_out = nn.Conv2d(c, 4, 3, padding=1)    # 4ch = flow_0→1 (2ch) + flow_1→0 (2ch)
        self.fusion_out = nn.Conv2d(c, 1, 3, padding=1)  # 1ch = per-pixel blend weight

    def forward(self, img0, img1):
        """img0, img1: (B, 3, H, W) — input frame pair"""
        x = torch.cat([img0, img1], dim=1)  # (B, 6, H, W)

        # Encode
        e1 = self.enc1(x)    # (B, c, H, W)
        e2 = self.enc2(e1)   # (B, c*2, H/2, W/2)
        e3 = self.enc3(e2)   # (B, c*4, H/4, W/4)

        # Bottleneck
        b = self.bottleneck(e3)  # (B, c*4, H/4, W/4)

        # Decode with skip connections from encoder layers
        # d3: upsample bottleneck → concat with e2 (same spatial: H/2)
        d3 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))  # (B, c*4 + c*2) → (B, c*2, H/2, W/2)

        # d2: upsample → concat with e1 (same spatial: H)
        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))   # (B, c*2 + c) → (B, c, H, W)

        d1 = self.dec1(d2)   # (B, c, H, W)

        # Outputs
        flows = self.flow_out(d1)                     # (B, 4, H, W)
        flow_01 = flows[:, :2]                         # flow 0→1
        flow_10 = flows[:, 2:]                         # flow 1→0
        fusion = torch.sigmoid(self.fusion_out(d1))   # (B, 1, H, W) ∈ [0, 1]

        return flow_01, flow_10, fusion


# ═══════════════════════════════════════════════════════════════════════════
# Frame Generator — full pipeline
# ═══════════════════════════════════════════════════════════════════════════

class FrameGenerator(nn.Module):
    """Complete frame generation pipeline.
    
    Input:  img0, img1 (two consecutive rendered frames)
    Output: interpolated frame at timestep t ∈ [0, 1]
    
    Pipeline:
      1. Estimate bidirectional flow ± fusion weights (FlowNetLite)
      2. Scale flow to target timestep
      3. Backward-warp both frames
      4. Blend with learned fusion weights
    """
    def __init__(self, base_ch=16):
        super().__init__()
        self.flownet = FlowNetLite(base_ch)

    def forward(self, img0, img1, t=0.5):
        """
        img0, img1: (B, 3, H, W) — frame at t=0 and t=1
        t: float ∈ [0,1] — target interpolation time
        Returns: interpolated frame (B, 3, H, W)
        """
        # Estimate flow and fusion weights
        flow_01, flow_10, fusion = self.flownet(img0, img1)

        # Scale flow to target timestep
        # flow_0→t = t * flow_0→1  (linear motion assumption)
        flow_0t = flow_01 * t
        flow_1t = flow_10 * (1 - t)

        # Warp both frames towards t
        warped_0, mask_0 = backward_warp(img0, flow_0t)
        warped_1, mask_1 = backward_warp(img1, flow_1t)

        # Blend: fusion weighted, with occlusion handling
        # Where frame 0 is occluded (mask_0 ≈ 0), use more of frame 1
        alpha = fusion * mask_0 / (fusion * mask_0 + (1 - fusion) * mask_1 + 1e-8)
        output = alpha * warped_0 + (1 - alpha) * warped_1

        return output

    @torch.no_grad()
    def generate_n_frames(self, img0, img1, n=3):
        """Generate n intermediate frames between img0 and img1.
        
        Returns list of n frames (including img0 and img1: n+2 total if inclusive).
        """
        frames = [img0]
        for i in range(1, n + 1):
            t = i / (n + 1)
            frames.append(self.forward(img0, img1, t))
        frames.append(img1)
        return frames
