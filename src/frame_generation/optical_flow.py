"""
Optical flow estimation using OpenCV Farneback (no GPU training needed).
"""
import cv2
import numpy as np
import torch


def estimate_flow_farneback(img0, img1):
    """Estimate dense optical flow between two images using Farneback algorithm.
    
    Args:
        img0, img1: torch.Tensor (1, 3, H, W) or (3, H, W), values in [0, 1]
    
    Returns:
        flow_01: torch.Tensor (1, 2, H, W) — backward flow from frame 1→0 (pixel units)
        flow_10: torch.Tensor (1, 2, H, W) — backward flow from frame 0→1 (pixel units)
    """
    # Convert to numpy uint8
    def to_np(t):
        if t.dim() == 4:
            t = t[0]
        t = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(t, cv2.COLOR_RGB2GRAY)

    g0 = to_np(img0)
    g1 = to_np(img1)

    # Farneback optical flow
    flow_01 = cv2.calcOpticalFlowFarneback(g1, g0, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_10 = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert to torch (H, W, 2) → (1, 2, H, W)
    flow_01 = torch.from_numpy(flow_01).float().permute(2, 0, 1).unsqueeze(0)
    flow_10 = torch.from_numpy(flow_10).float().permute(2, 0, 1).unsqueeze(0)

    return flow_01, flow_10


def interpolate_frame_farneback(img0, img1, t=0.5):
    """Interpolate a frame at timestep t using Farneback optical flow.
    
    This is the simplest possible frame generation — no neural network,
    just optical flow + warp + blend. Good baseline to compare against.
    """
    B, C, H, W = img0.shape
    flow_01, flow_10 = estimate_flow_farneback(img0, img1)

    # Scale flow
    flow_0t = flow_01.to(img0.device) * t
    flow_1t = flow_10.to(img0.device) * (1 - t)

    # Simple bilinear warp (using grid_sample)
    def warp(img, flow_px):
        ys = torch.linspace(-1, 1, H, device=img.device)
        xs = torch.linspace(-1, 1, W, device=img.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        dx = (2.0 * flow_px[:, 0] / max(W-1, 1)).unsqueeze(-1)
        dy = (2.0 * flow_px[:, 1] / max(H-1, 1)).unsqueeze(-1)
        sample_grid = base_grid + torch.cat([dx, dy], dim=-1)
        return torch.nn.functional.grid_sample(img, sample_grid, mode="bilinear",
                                                padding_mode="border", align_corners=True)

    w0 = warp(img0, flow_0t)
    w1 = warp(img1, flow_1t)

    # Simple blend
    return (1 - t) * w0 + t * w1
