#!/usr/bin/env python3
"""
Baseline: generate N pure Gaussian noise images and compute FID + MSE
against real CIFAR-10 images.

Usage:
    python gaussian_noise_baseline.py          # 10 samples
    python gaussian_noise_baseline.py --n 100  # more samples
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import PIL.Image

EDM_DIR = Path("edm")
FID_REF_NPZ = Path("cifar10-32x32.npz")
REAL_IMAGES_DIR = Path("output/real_images_train")
IMAGE_SIZE = 32


def generate_noise_images(out_dir: Path, n: int, seed: int = 0) -> list[Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        # N(128, 60^2) clipped to [0, 255] — matches typical image pixel statistics
        noise = rng.normal(128.0, 60.0, (IMAGE_SIZE, IMAGE_SIZE, 3))
        arr = noise.clip(0, 255).astype(np.uint8)
        p = out_dir / f"{i:05d}.png"
        PIL.Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def compute_mse(noise_dir: Path, real_dir: Path, n: int) -> float:
    noise_paths = sorted(noise_dir.rglob("*.png"))[:n]
    real_paths  = sorted(real_dir.rglob("*.png"))[:n]
    pairs = min(len(noise_paths), len(real_paths))
    if pairs == 0:
        return float("nan")
    errors = []
    for np_path, rp in zip(noise_paths[:pairs], real_paths[:pairs]):
        gen_arr  = np.asarray(PIL.Image.open(np_path).convert("RGB"), dtype=np.float32)
        real_arr = np.asarray(PIL.Image.open(rp).convert("RGB"),  dtype=np.float32)
        errors.append(float(np.mean((gen_arr - real_arr) ** 2)))
    return float(np.mean(errors))


def compute_fid(images_dir: Path, ref_npz: Path, n: int, batch: int = 10) -> float:
    dist_env = dict(os.environ)
    dist_env.update({"USE_LIBUV": "0", "KMP_DUPLICATE_LIB_OK": "TRUE"})
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "HOSTNAME"):
        dist_env.pop(key, None)

    result = subprocess.run(
        [sys.executable, str(EDM_DIR / "fid.py"), "calc",
         "--images", str(images_dir),
         "--ref",    str(ref_npz),
         "--num",    str(n),
         "--batch",  str(batch)],
        capture_output=True, text=True, env=dist_env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"fid.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    print(result.stdout)
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    raise RuntimeError(f"Could not parse FID:\n{result.stdout}\n{result.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="Number of noise samples (default: 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("output/gaussian_noise_baseline"))
    ap.add_argument("--skip-fid", action="store_true", help="Skip FID (needs edm + inception model)")
    args = ap.parse_args()

    sys.path.insert(0, str(EDM_DIR.resolve()))

    print(f"Generating {args.n} pure Gaussian noise images -> {args.outdir}", flush=True)
    generate_noise_images(args.outdir, args.n, args.seed)

    if not REAL_IMAGES_DIR.exists():
        print(f"WARNING: real images dir not found at {REAL_IMAGES_DIR}, skipping MSE", flush=True)
        mse = float("nan")
    else:
        mse = compute_mse(args.outdir, REAL_IMAGES_DIR, args.n)

    print(f"\n--- Results (n={args.n}) ---")
    print(f"MSE (noise vs real CIFAR-10): {mse:.4f}")

    if args.skip_fid:
        print("FID: skipped")
    else:
        if not FID_REF_NPZ.exists():
            print(f"WARNING: FID reference not found at {FID_REF_NPZ}, skipping FID")
        else:
            print("Computing FID (note: unreliable with n<2048) ...", flush=True)
            try:
                fid = compute_fid(args.outdir, FID_REF_NPZ, args.n)
                print(f"FID (noise vs real CIFAR-10): {fid:.4f}")
            except Exception as e:
                print(f"FID computation failed: {e}")


if __name__ == "__main__":
    main()
