"""
Denoising Module — AI-based single-model denoiser

Simulates DLSS 3.5 Ray Reconstruction: replaces multiple hand-tuned denoisers
with a single learned model. For path-traced images, ray reconstruction cleans
up noise from sparse ray sampling.

Our simplified version: U-Net denoiser that works on general image noise.
For DLSS integration: the denoiser could be applied after super-resolution
to clean up artifacts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv → BN → PReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.PReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UNetDenoiser(nn.Module):
    """Lightweight U-Net for image denoising.

    Architecture:
      Encoder: 4 levels of DoubleConv + MaxPool (channels: 3 → 32 → 64 → 128 → 256)
      Bottleneck: DoubleConv (256)
      Decoder: 4 levels of Upsample + DoubleConv with skip connections
      Output: 3ch residual (added to input for residual learning)

    Residual learning (predict noise, not clean image) helps training stability.
    """
    def __init__(self, base_ch=32):
        super().__init__()
        c = base_ch

        # Encoder
        self.enc1 = DoubleConv(3, c)        # H×W
        self.enc2 = DoubleConv(c, c*2)       # H/2×W/2
        self.enc3 = DoubleConv(c*2, c*4)     # H/4×W/4
        self.enc4 = DoubleConv(c*4, c*8)     # H/8×W/8

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(c*8, c*8)  # H/16×W/16

        # Decoder (upsample + concat skip → DoubleConv)
        # Channel math: skip from encN has c*2^(N-1), upsampled has c*2^(5-N)
        self.up4 = nn.ConvTranspose2d(c*8, c*4, 2, stride=2)
        self.dec4 = DoubleConv(c*4 + c*8, c*4)   # up4 out (c*4) + skip e4 (c*8)

        self.up3 = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec3 = DoubleConv(c*2 + c*4, c*2)   # up3 out (c*2) + skip e3 (c*4)

        self.up2 = nn.ConvTranspose2d(c*2, c, 2, stride=2)
        self.dec2 = DoubleConv(c + c*2, c)       # up2 out (c) + skip e2 (c*2)

        self.up1 = nn.ConvTranspose2d(c, c, 2, stride=2)
        self.dec1 = DoubleConv(c + c, c)         # up1 out (c) + skip e1 (c)

        # Output: predict residual (noise), not clean image
        self.output = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, x):
        """x: (B, 3, H, W) noisy image → (B, 3, H, W) denoised image"""
        # Encoder
        e1 = self.enc1(x)                     # (B, c, H, W)
        e2 = self.enc2(self.pool(e1))         # (B, c*2, H/2, W/2)
        e3 = self.enc3(self.pool(e2))         # (B, c*4, H/4, W/4)
        e4 = self.enc4(self.pool(e3))         # (B, c*8, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))    # (B, c*8, H/16, W/16)

        # Decoder with skip connections
        d4 = self.up4(b)                      # (B, c*4, H/8, W/8)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))  # (B, c*4, H/8, W/8)

        d3 = self.up3(d4)                     # (B, c*2, H/4, W/4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # (B, c*2, H/4, W/4)

        d2 = self.up2(d3)                     # (B, c, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, c, H/2, W/2)

        d1 = self.up1(d2)                     # (B, c, H, W)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, c, H, W)

        # Residual learning: predict noise, return x - noise
        residual = self.output(d1)            # (B, 3, H, W)
        return x - residual


def add_gaussian_noise(image, sigma=0.1):
    """Add Gaussian noise to an image tensor.
    
    Args:
        image: (B, 3, H, W) clean image in [0, 1]
        sigma: noise standard deviation
    """
    noise = torch.randn_like(image) * sigma
    noisy = torch.clamp(image + noise, 0, 1)
    return noisy, noise
