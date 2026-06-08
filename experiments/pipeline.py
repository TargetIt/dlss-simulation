"""
DLSS 4.5 Simulation — End-to-End Integration Pipeline

Pipeline:
  1. Load HR source image → generate synthetic LR video (pan + jitter)
  2. Super Resolution: LR → HR (WarpFuseTSR with Farneback flow)
  3. Frame Generation: interpolate between SR frames (FlowNetLite)
  4. Denoising: clean up artifacts (UNetDenoiser)
  5. Metrics: PSNR, SSIM, latency per stage

Usage:
  python experiments/pipeline.py --image data/samples/sample_00.jpg
"""
import argparse, time, sys, os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_skimage

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.super_resolution import WarpFuseTSR, backward_warp, get_device, compute_psnr, compute_ssim
from src.frame_generation import FrameGenerator, interpolate_frame_farneback
from src.denoising import UNetDenoiser


# ═══════════════════════════════════════════════════════════════════════════
# Data Generation — synthetic LR video from a single HR image
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_frames(hr_image, num_frames=30, lr_size=(80, 80), offset_range=8):
    """Generate LR frames simulating camera pan across an HR image.

    Each frame is a different crop of the HR image, then downsampled to LR.
    Halton-like jitter adds sub-pixel diversity between frames.
    """
    h_hr, w_hr = hr_image.shape[:2]
    frames_lr = []
    frames_hr = []  # ground truth HR crops
    offsets = []

    for i in range(num_frames):
        # Deterministic offset (simulating smooth pan + jitter)
        t = i / max(num_frames - 1, 1)
        ox = int(np.sin(t * np.pi * 2) * offset_range + offset_range)
        oy = int(np.cos(t * np.pi * 3) * offset_range + offset_range)
        ox = max(0, min(ox, w_hr - lr_size[0] * 4))
        oy = max(0, min(oy, h_hr - lr_size[1] * 4))

        # HR crop (ground truth)
        hr_crop = hr_image[oy:oy + lr_size[1]*4, ox:ox + lr_size[0]*4]
        if hr_crop.shape[0] != lr_size[1]*4 or hr_crop.shape[1] != lr_size[0]*4:
            hr_crop = cv2.resize(hr_image, (lr_size[0]*4, lr_size[1]*4))

        # LR frame (downsample)
        lr_crop = cv2.resize(hr_crop, lr_size, interpolation=cv2.INTER_AREA)

        frames_lr.append(lr_crop)
        frames_hr.append(hr_crop)
        offsets.append((ox, oy))

    return frames_lr, frames_hr, offsets


def to_torch(img, device="cpu"):
    """numpy H×W×C [0,255] → torch B×C×H×W [0,1]"""
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.to(device)


def to_numpy(t, normalize=False):
    """torch B×C×H×W [0,1] → numpy H×W×C [0,255]"""
    x = t[0].permute(1, 2, 0).detach().cpu()
    if normalize:
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-8:
            x = (x - x_min) / (x_max - x_min)
    return (x.clip(0, 1) * 255).to(torch.uint8).numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(image_path, output_dir="assets/pipeline", num_frames=30, lr_size=(80, 80)):
    """Run the full DLSS 4.5 simulation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    dev = get_device()
    print(f"Device: {dev}")
    print(f"Input:  {image_path}")
    print(f"Frames: {num_frames}, LR size: {lr_size}")
    print("=" * 65)

    # ── Load image ──
    hr_img = cv2.imread(image_path)
    if hr_img is None:
        print(f"ERROR: Cannot load {image_path}")
        return
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
    print(f"HR source: {hr_img.shape[1]}×{hr_img.shape[0]}")

    # ── Generate synthetic frames ──
    print("\n[1/4] Generating synthetic LR frames...")
    t0 = time.perf_counter()
    lr_frames, hr_frames, offsets = generate_synthetic_frames(hr_img, num_frames, lr_size)
    dt_gen = time.perf_counter() - t0
    print(f"  Generated {len(lr_frames)} frames in {dt_gen:.2f}s")

    # ── Stage 1: Super Resolution ──
    print("\n[2/4] Super Resolution (WarpFuseTSR)...")
    sr_model = WarpFuseTSR(hidden_channels=64).to(dev).eval()

    # Try loading trained checkpoint
    checkpoint_path = "references/Super-Resolution/checkpoints/p2_best.pth"
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=dev, weights_only=True)
        sr_model.load_state_dict(state)
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print("  ⚠️  No checkpoint found, using random weights")
    sr_frames = []
    sr_times = []

    prev_frame = None
    for i, lr_np in enumerate(lr_frames):
        lr_t = to_torch(lr_np, dev)

        if prev_frame is None:
            # First frame: no history, use self as prev (no warp benefit)
            sr_t = sr_model(lr_t, lr_t, torch.tensor([[0., 0.]], device=dev), backward_warp)
        else:
            # Estimate flow using Farneback (returns H×W×2 numpy)
            flow_np, _ = estimate_flow_farneback_cv(prev_frame, lr_np)
            # Convert: H×W×2 → 2×H×W → 1×2×H×W
            flow_t = torch.from_numpy(flow_np).float().permute(2, 0, 1).unsqueeze(0).to(dev)
            t_start = time.perf_counter()
            sr_t = sr_model(prev_t, lr_t, flow_t, backward_warp)
            sr_times.append(time.perf_counter() - t_start)

        sr_frames.append(sr_t)
        prev_frame = lr_np
        prev_t = lr_t

    sr_np = [to_numpy(f) for f in sr_frames]  # trained model, no normalize needed
    avg_sr_time = np.mean(sr_times) * 1000 if sr_times else 0
    print(f"  {len(sr_frames)} frames upscaled | avg {avg_sr_time:.1f} ms/frame")

    # ── Stage 2: Frame Generation ──
    print("\n[3/4] Frame Generation (FlowNetLite)...")
    fg_model = FrameGenerator(base_ch=16).to(dev).eval()
    all_frames = []  # interleaved: SR0, FG0.25, FG0.5, FG0.75, SR1, ...
    fg_times = []

    for i in range(len(sr_frames) - 1):
        all_frames.append(sr_frames[i])  # original SR frame

        # Generate 3 intermediate frames (simulating 4× frame gen)
        for t_ratio in [0.25, 0.5, 0.75]:
            t_start = time.perf_counter()
            mid = fg_model(sr_frames[i], sr_frames[i + 1], t=t_ratio)
            fg_times.append(time.perf_counter() - t_start)
            all_frames.append(mid)

    all_frames.append(sr_frames[-1])
    all_np = [to_numpy(f) for f in all_frames]
    avg_fg_time = np.mean(fg_times) * 1000 if fg_times else 0
    print(f"  {len(sr_frames)} SR → {len(all_frames)} frames (4× frame gen)")
    print(f"  avg {avg_fg_time:.1f} ms/interpolated frame")

    # ── Stage 3: Denoising ──
    print("\n[4/4] Denoising (UNetDenoiser)...")
    dn_model = UNetDenoiser(base_ch=32).to(dev).eval()
    dn_frames = []
    dn_times = []

    for frame_t in all_frames:
        t_start = time.perf_counter()
        dn_t = dn_model(frame_t)
        dn_times.append(time.perf_counter() - t_start)
        dn_frames.append(dn_t)

    dn_np = [to_numpy(f) for f in dn_frames]
    avg_dn_time = np.mean(dn_times) * 1000 if dn_times else 0
    print(f"  {len(dn_frames)} frames denoised | avg {avg_dn_time:.1f} ms/frame")

    # ── Metrics ──
    print("\n" + "=" * 65)
    print("METRICS")
    print("=" * 65)

    # Compare SR frames with ground truth HR crops
    sr_psnrs, sr_ssims = [], []
    for i, (sr_np_i, hr_np_i) in enumerate(zip(sr_np, hr_frames)):
        psnr_val = cv2.PSNR(hr_np_i, sr_np_i) if sr_np_i.shape == hr_np_i.shape else 0
        ssim_val = ssim_skimage(hr_np_i, sr_np_i, channel_axis=2,
                                data_range=255) if sr_np_i.shape == hr_np_i.shape else 0
        sr_psnrs.append(psnr_val)
        sr_ssims.append(ssim_val)

    print(f"Super Resolution:")
    print(f"  PSNR: {np.mean(sr_psnrs):.1f} dB (±{np.std(sr_psnrs):.1f})")
    print(f"  SSIM: {np.mean(sr_ssims):.4f} (±{np.std(sr_ssims):.4f})")

    print(f"\nLatency breakdown (per frame):")
    print(f"  Super Resolution:    {avg_sr_time:6.1f} ms")
    print(f"  Frame Generation:    {avg_fg_time:6.1f} ms (per interpolated frame)")
    print(f"  Denoising:           {avg_dn_time:6.1f} ms")
    total_ms = avg_sr_time + avg_fg_time * 3 + avg_dn_time  # 3 interpolated per SR pair
    print(f"  Total (1 SR + 3 FG + 1 DN): {total_ms:.0f} ms")
    fps = 1000 / total_ms if total_ms > 0 else 0
    print(f"  Effective FPS:       {fps:.1f}")

    # ── Save comparison grid ──
    print(f"\nSaving outputs to {output_dir}/")
    save_comparison_grid(hr_img, lr_frames[0], sr_np[0], all_np[1], output_dir)

    # Save as video
    save_video(all_np, f"{output_dir}/pipeline_output.mp4", fps=10)

    print("\nDone! ✅")
    return {
        "sr_psnr": np.mean(sr_psnrs),
        "sr_ssim": np.mean(sr_ssims),
        "sr_ms": avg_sr_time,
        "fg_ms": avg_fg_time,
        "dn_ms": avg_dn_time,
        "total_ms": total_ms,
    }


def estimate_flow_farneback_cv(img0_np, img1_np):
    """Estimate dense optical flow (numpy in, numpy out).
    
    Returns flow as (H, W, 2) numpy array in OpenCV convention.
    """
    g0 = cv2.cvtColor(img0_np, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g0, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow, None


def save_comparison_grid(hr_original, lr_sample, sr_sample, fg_sample, output_dir):
    """Save a 2×2 comparison: Original | LR | SR | FG"""
    # Resize for visualization
    h, w = sr_sample.shape[:2]
    lr_vis = cv2.resize(lr_sample, (w, h), interpolation=cv2.INTER_NEAREST)
    hr_vis = cv2.resize(hr_original, (w, h))

    top = np.hstack([hr_vis, lr_vis])
    bottom = np.hstack([sr_sample, fg_sample])

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for img, label, col in [(hr_vis, "Original HR", (0, 255, 0)),
                              (lr_vis, "Input LR", (255, 0, 0)),
                              (sr_sample, "Super Resolution", (0, 255, 255)),
                              (fg_sample, "Frame Generation", (255, 255, 0))]:
        pass  # labels drawn on the composite below

    composite = np.vstack([top, bottom])
    composite = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    # Add text labels
    labels = ["Original HR", "Input LR (upscaled)", "Super Resolution", "Frame Generation"]
    x_positions = [10, w + 10, 10, w + 10]
    y_positions = [30, 30, h + 30, h + 30]
    for label, x, y in zip(labels, x_positions, y_positions):
        cv2.putText(composite, label, (x, y), font, 0.7, (255, 255, 255), 2)

    cv2.imwrite(f"{output_dir}/comparison_grid.png", composite)
    print(f"  → {output_dir}/comparison_grid.png")


def save_video(frames_np, path, fps=10):
    """Save list of numpy frames as MP4 video."""
    if not frames_np:
        return
    h, w = frames_np[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames_np:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  → {path} ({len(frames_np)} frames @ {fps} fps)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLSS 4.5 Simulation Pipeline")
    parser.add_argument("--image", default="references/Super-Resolution/data/samples/sample_00.jpg",
                        help="Path to HR source image")
    parser.add_argument("--frames", type=int, default=30, help="Number of synthetic LR frames")
    parser.add_argument("--lr-size", type=int, default=80, help="LR frame width (height auto)")
    parser.add_argument("--output", default="assets/pipeline", help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.image, args.output, args.frames, (args.lr_size, args.lr_size))
