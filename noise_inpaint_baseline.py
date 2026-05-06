#!/usr/bin/env python3
"""
Pure-noise inpainting baseline.

Takes N real CIFAR-10 training images, randomly masks 50% of pixels
(same locations across all channels, independently per image), replaces
masked pixels with pure Gaussian noise, saves the results, then computes
FID and MSE against the original images.

Usage:
    python noise_inpaint_baseline.py
    python noise_inpaint_baseline.py --n 100 --lam 0.5
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import PIL.Image

REAL_IMAGES_DIR = Path("output/real_images_train")
FID_REF_NPZ     = Path("cifar10-32x32.npz")
EDM_DIR         = Path("edm")
DEFAULT_OUT     = Path("output/noise_inpaint_baseline")


def apply_noise_mask(arr: np.ndarray, lam: float, rng: np.random.Generator) -> np.ndarray:
    """Replace lam fraction of pixels with Gaussian noise (per-image random mask)."""
    arr = arr.copy().astype(np.float32)
    H, W, C = arr.shape
    n_pixels = H * W
    n_masked = int(n_pixels * lam)
    idx = rng.permutation(n_pixels)[:n_masked]
    flat = arr.reshape(n_pixels, C)
    # Gaussian noise matching typical pixel range: N(128, 60^2)
    flat[idx] = rng.normal(128.0, 60.0, size=(n_masked, C))
    flat[idx] = flat[idx].clip(0, 255)
    return flat.reshape(H, W, C).astype(np.uint8)


def compute_mse(corrupted_paths: list, original_paths: list) -> float:
    errors = []
    for cp, op in zip(corrupted_paths, original_paths):
        gen_arr  = np.asarray(PIL.Image.open(cp).convert("RGB"),  dtype=np.float32)
        orig_arr = np.asarray(PIL.Image.open(op).convert("RGB"), dtype=np.float32)
        errors.append(float(np.mean((gen_arr - orig_arr) ** 2)))
    return float(np.mean(errors))


def compute_fid(images_dir: Path, ref_npz: Path, n: int, batch: int) -> float:
    env = dict(os.environ)
    env.update({"USE_LIBUV": "0", "KMP_DUPLICATE_LIB_OK": "TRUE"})
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "HOSTNAME"):
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, str(EDM_DIR / "fid.py"), "calc",
         "--images", str(images_dir),
         "--ref",    str(ref_npz),
         "--num",    str(n),
         "--batch",  str(batch)],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"fid.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    print(result.stdout, flush=True)
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    raise RuntimeError(f"Could not parse FID:\n{result.stdout}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int,   default=10,   help="Number of images (default: 10)")
    ap.add_argument("--lam",     type=float, default=0.5,  help="Fraction of pixels to corrupt (default: 0.5)")
    ap.add_argument("--seed",    type=int,   default=0)
    ap.add_argument("--outdir",  type=Path,  default=DEFAULT_OUT)
    ap.add_argument("--skip-fid", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Collect source images
    src_paths = sorted(REAL_IMAGES_DIR.rglob("*.png"))[: args.n]
    if len(src_paths) < args.n:
        raise RuntimeError(f"Only {len(src_paths)} images found in {REAL_IMAGES_DIR}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_paths = []

    print(f"Masking {int(args.lam * 100)}% of pixels in {args.n} images -> {args.outdir}", flush=True)
    for i, src in enumerate(src_paths):
        arr = np.asarray(PIL.Image.open(src).convert("RGB"))
        corrupted = apply_noise_mask(arr, args.lam, rng)
        out_p = args.outdir / f"{i:05d}.png"
        PIL.Image.fromarray(corrupted).save(out_p)
        out_paths.append(out_p)
        print(f"  [{i+1}/{args.n}] {src.name} -> {out_p.name}", flush=True)

    mse = compute_mse(out_paths, src_paths)
    print(f"\n--- Results (n={args.n}, lam={args.lam}) ---")
    print(f"MSE (corrupted vs original): {mse:.4f}")

    if args.skip_fid:
        print("FID: skipped")
        return

    if not FID_REF_NPZ.exists():
        print(f"WARNING: {FID_REF_NPZ} not found — skipping FID")
        return

    sys.path.insert(0, str(EDM_DIR.resolve()))
    print("\nComputing FID ...", flush=True)
    fid = compute_fid(args.outdir, FID_REF_NPZ, args.n, batch=min(args.n, 64))
    print(f"FID (corrupted vs CIFAR-10 ref): {fid:.4f}")


if __name__ == "__main__":
    main()
