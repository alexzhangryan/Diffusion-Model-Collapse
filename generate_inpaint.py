#!/usr/bin/env python3
"""
Replacement-based conditional inpainting using a pretrained EDM model.

At every denoising step the known region (left half of each image) is
enforced by replacing those pixels with the real image plus Gaussian noise
at the current noise level.  The unknown region (right half) is generated
freely by the diffusion model.  Final output is a composite:

    output = real_left_half  |  model_generated_right_half

This implements the "RePaint" strategy and works with any pretrained EDM
checkpoint without retraining or architectural changes.

Usage (called automatically by inpainting.py):
    python generate_inpaint.py \
        --network  path/to/network.pkl \
        --images   path/to/source_images/ \
        --outdir   path/to/output/ \
        --n        50000 \
        --lam      0.5 \
        --batch    64
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import tqdm


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_png_batch(paths, device: torch.device) -> torch.Tensor:
    """Load a list of PNG paths -> [B, C, H, W] float32 in [-1, 1]."""
    arrs = [np.asarray(PIL.Image.open(p).convert("RGB"), dtype=np.float32)
            for p in paths]
    t = torch.from_numpy(np.stack(arrs)).to(device)   # [B, H, W, C]
    return t.permute(0, 3, 1, 2) / 127.5 - 1.0        # [B, C, H, W]


def save_composite(tensor: torch.Tensor, path: Path):
    """Save a single [C, H, W] tensor in [-1, 1] as a PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = ((tensor.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5)
    PIL.Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------------------
# Inpainting sampler
# ---------------------------------------------------------------------------

def build_mask(B: int, C: int, H: int, W: int,
               lam: float, device: torch.device) -> torch.Tensor:
    """Float mask: 1 = known (left half), 0 = unknown (right half)."""
    mask = torch.ones(B, C, H, W, device=device)
    mask[:, :, :, int(W * (1.0 - lam)):] = 0.0
    return mask


@torch.no_grad()
def heun_inpaint(
    net,
    real: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    class_labels=None,
) -> torch.Tensor:
    """
    EDM Heun sampler with per-step replacement (RePaint).

    Parameters
    ----------
    net          : EDM EMA network (callable: net(x, sigma, class_labels))
    real         : [B, C, H, W] real images in [-1, 1]
    mask         : [B, C, H, W] float, 1 = known, 0 = unknown
    steps        : number of denoising steps
    class_labels : [B, label_dim] one-hot or None

    Returns
    -------
    Composite [B, C, H, W]: known pixels from `real`, unknown from model.
    """
    device = real.device
    B = real.shape[0]

    # EDM sigma schedule (Eq. 5 in Karras et al. 2022)
    idx = torch.arange(steps, dtype=torch.float64, device=device)
    sigmas = (
        sigma_max ** (1 / rho)
        + idx / max(steps - 1, 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])  # append ?=0

    # Start from pure noise
    x = torch.randn_like(real) * sigmas[0]

    for i in range(steps):
        t_cur  = sigmas[i].float()
        t_next = sigmas[i + 1].float()

        # -- Replacement: paste known region with correctly-noised real image --
        x = mask * (real + torch.randn_like(real) * t_cur) + (1 - mask) * x

        # -- First-order Euler step --------------------------------------------
        denoised_cur = net(x, t_cur.expand(B), class_labels)
        d_cur = (x - denoised_cur) / t_cur
        x_next = x + (t_next - t_cur) * d_cur

        # -- Heun correction (skip at final step where t_next = 0) ------------
        if t_next > 0:
            x_next = (mask * (real + torch.randn_like(real) * t_next)
                      + (1 - mask) * x_next)
            denoised_next = net(x_next, t_next.expand(B), class_labels)
            d_next = (x_next - denoised_next) / t_next
            x_next = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_next)

        x = x_next

    # Final composite: real left half pasted over generated output
    return mask * real + (1 - mask) * x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="EDM replacement-based inpainting.")
    ap.add_argument("--network", required=True,
                    help="Path to EDM .pkl checkpoint.")
    ap.add_argument("--images",  required=True,
                    help="Directory of source (conditioning) images.")
    ap.add_argument("--outdir",  required=True,
                    help="Output directory for composite images.")
    ap.add_argument("--n",       type=int, default=None,
                    help="Max number of images to process (default: all).")
    ap.add_argument("--lam",     type=float, default=0.5,
                    help="Fraction of width to inpaint (right side). Default 0.5.")
    ap.add_argument("--steps",   type=int,   default=18,
                    help="Denoising steps. Default 18.")
    ap.add_argument("--batch",   type=int,   default=64,
                    help="Batch size. Default 64.")
    ap.add_argument("--edm-dir", default="edm",
                    help="Path to EDM repo root (for imports).")
    args = ap.parse_args()

    # Make EDM utilities importable (needed for pickle deserialization)
    sys.path.insert(0, str(Path(args.edm_dir).resolve()))
    import dnnlib       # noqa: F401
    import torch_utils  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inpaint] device={device}", flush=True)

    # Load network
    print(f"[inpaint] loading {args.network} ...", flush=True)
    with open(args.network, "rb") as f:
        net = pickle.load(f)["ema"].to(device)
    net.eval()
    label_dim = getattr(net, "label_dim", 0)

    # Collect source images
    src_paths = sorted(Path(args.images).rglob("*.png"))
    if args.n:
        src_paths = src_paths[: args.n]
    print(f"[inpaint] {len(src_paths)} source images -> {args.outdir}",
          flush=True)

    out_root = Path(args.outdir)

    for b_start in tqdm.trange(0, len(src_paths), args.batch,
                                desc="inpainting"):
        batch_paths = src_paths[b_start: b_start + args.batch]
        real = load_png_batch(batch_paths, device)
        B, C, H, W = real.shape
        mask = build_mask(B, C, H, W, args.lam, device)

        # Random class labels (matches unconditional generation behaviour)
        class_labels = None
        if label_dim:
            rand_idx = torch.randint(label_dim, (B,), device=device)
            class_labels = torch.zeros(B, label_dim, device=device)
            class_labels.scatter_(1, rand_idx.unsqueeze(1), 1)

        composites = heun_inpaint(net, real, mask,
                                   steps=args.steps,
                                   class_labels=class_labels)

        for i, src_p in enumerate(batch_paths):
            rel = Path(src_p).relative_to(Path(args.images))
            save_composite(composites[i], out_root / rel)

    print(f"[inpaint] done ? {len(src_paths)} composites saved.", flush=True)


if __name__ == "__main__":
    main()
