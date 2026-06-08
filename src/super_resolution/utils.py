"""
Utilities: device detection, PSNR, SSIM.
"""
import torch
import torch.nn.functional as F
from math import log10


def get_device():
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio (higher = better)."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 20 * log10(max_val) - 10 * log10(mse.item())


def compute_ssim(pred, target, window_size=11):
    """Structural Similarity Index (0-1, higher = better).
    
    Simplified implementation using Gaussian-filtered luminance/contrast/structure.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    window_2d = window_1d.unsqueeze(0).unsqueeze(0).expand(pred.size(1), 1, -1, -1)

    mu1 = F.conv2d(pred, window_2d, padding=window_size//2, groups=pred.size(1))
    mu2 = F.conv2d(target, window_2d, padding=window_size//2, groups=target.size(1))
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window_2d, padding=window_size//2, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window_2d, padding=window_size//2, groups=target.size(1)) - mu2_sq
    sigma12 = F.conv2d(pred * target, window_2d, padding=window_size//2, groups=target.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()
