#!/usr/bin/env python3
"""
Inpainting model-collapse experiment using EDM on CIFAR-10.

Each generation:
  1. Train an EDM model on the previous generation's composite images
     (real CIFAR-10 for generation 0).  Training uses COMPLETE images —
     no masking — so the model learns the full image distribution.
  2. Run conditional inpainting: given the real left half of each CIFAR-10
     image, the trained model fills in the right half using replacement-based
     sampling (RePaint style).  Output = real_left | synthetic_right.
  3. Measure quality of the inpainted region:
       - MSE between generated and original images (right half only)
       - Pixel variance of the composite images
       - FID vs. real CIFAR-10
  4. The composite images become the training set for the next generation.

LAMBDA controls what fraction of each image width is inpainted (right side).
  LAMBDA = 0.5  -> right half is inpainted  (default)

Usage:
  python inpainting.py                  # server run
  python inpainting.py --local          # quick local test
  python inpainting.py --lambda 0.3     # override inpaint fraction
  python inpainting.py --untar experiment.tar.gz
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration  — change LAMBDA here or pass --lambda on the command line
# ---------------------------------------------------------------------------
LAMBDA            = 0.5      # fraction of image width to mask (right side)

N_GENERATIONS     = 2

# Full-scale (server)
N_IMAGES_FULL     = 10_000
DURATION_FULL     = 25.0     # Mimg
BATCH_FULL        = 512

# Local test
N_IMAGES_LOCAL    = 200
DURATION_LOCAL    = 1.0
BATCH_LOCAL       = 32

# Paths
EDM_DIR           = Path("edm")
CIFAR10_ZIP       = Path("dataset/cifar10-32x32.zip")        # 50k train images
FID_REF_NPZ       = Path("cifar10-32x32.npz")
OUTPUT_DIR        = Path("output")
RESULTS_JSON      = OUTPUT_DIR / "results.json"

IMAGE_SIZE        = 32   # CIFAR-10 images are 32×32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, **kwargs):
    printable = " ".join(str(c) for c in cmd)
    print(f"\n>>> {printable}\n", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def unpack_tar(tar_path: str):
    print(f"Unpacking {tar_path}...", flush=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(".")


def find_latest_snapshot(train_dir: Path) -> Path:
    snaps = sorted(train_dir.glob("network-snapshot-*.pkl"))
    if not snaps:
        raise FileNotFoundError(f"No snapshot in {train_dir}")
    return snaps[-1]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def copy_n_images(src_dir: Path, dst_dir: Path, n: int, offset: int = 0):
    """Copy n PNGs from src_dir starting at offset, preserving structure."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    for img_path in sorted(src_dir.rglob("*.png"))[offset: offset + n]:
        rel = img_path.relative_to(src_dir)
        out_path = dst_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, out_path)


def compute_masked_mse(generated_dir: Path, original_dir: Path, lam: float) -> float:
    """
    MSE between generated and original images, evaluated only on the masked region.
    Pairs images by sorted filename. Returns NaN if no pairs found.
    """
    gen_paths  = sorted(generated_dir.rglob("*.png"))
    orig_paths = sorted(original_dir.rglob("*.png"))
    n = min(len(gen_paths), len(orig_paths), 1000)   # cap at 1000 for speed
    if n == 0:
        return float("nan")

    squared_errors = []
    for gp, op in zip(gen_paths[:n], orig_paths[:n]):
        gen_arr  = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32)
        orig_arr = np.asarray(Image.open(op).convert("RGB"), dtype=np.float32)
        mask_start = int(IMAGE_SIZE * (1.0 - lam))
        diff = gen_arr[:, mask_start:, :] - orig_arr[:, mask_start:, :]
        squared_errors.append(float(np.mean(diff ** 2)))

    return float(np.mean(squared_errors))


def compute_pixel_variance(images_dir: Path) -> float:
    pixel_lists = []
    for img_path in images_dir.rglob("*.png"):
        arr = np.asarray(Image.open(img_path), dtype=np.float32)
        pixel_lists.append(arr.ravel())
    if not pixel_lists:
        return float("nan")
    return float(np.concatenate(pixel_lists).var())


def compute_fid(images_dir: Path, ref_npz: Path, batch: int, local: bool = False) -> float:
    launcher = [sys.executable] if local else ["torchrun", "--standalone", "--nproc_per_node=1"]
    result = subprocess.run(
        launcher + [
            EDM_DIR / "fid.py", "calc",
            "--images", images_dir,
            "--ref", ref_npz,
            "--batch", str(batch),
        ],
        capture_output=True, text=True, check=True,
    )
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    raise RuntimeError(f"Could not parse FID:\n{result.stdout}\n{result.stderr}")


# ---------------------------------------------------------------------------
# Core pipeline steps
# ---------------------------------------------------------------------------

def train_model(data_zip: Path, out_dir: Path, duration: float, batch: int, local: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    launcher = [sys.executable] if local else ["torchrun", "--standalone", "--nproc_per_node=1"]
    tick = "1" if local else "10"
    snap = "1" if local else "500"
    cmd = launcher + [
        EDM_DIR / "train.py",
        "--outdir", out_dir,
        "--data",   data_zip,
        "--cond=1",
        "--arch=ddpmpp",
        "--precond=vp",
        "--duration", str(duration),
        "--batch",    str(batch),
        "--nosubdir",
        f"--tick={tick}",
        f"--snap={snap}",
    ]
    if not local:
        cmd += ["--fp16=1", "--workers=4"]
    run(cmd)


def generate_inpainted_images(
    network_pkl: Path, cond_dir: Path, out_dir: Path,
    n_images: int, lam: float, batch: int,
):
    """Run replacement-based inpainting: real left half + model right half."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    run([
        sys.executable,
        Path(__file__).parent / "generate_inpaint.py",
        "--network", network_pkl,
        "--images",  cond_dir,
        "--outdir",  out_dir,
        "--n",       str(n_images),
        "--lam",     str(lam),
        "--batch",   str(batch),
        "--edm-dir", str(EDM_DIR),
    ])


def pack_images_to_zip(images_dir: Path, out_zip: Path):
    if out_zip.exists():
        out_zip.unlink()
    run([sys.executable, EDM_DIR / "dataset_tool.py",
         "--source", images_dir, "--dest", out_zip])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDM inpainting collapse experiment.")
    parser.add_argument("--local",    action="store_true",
                        help="Local test mode: fewer images, shorter training.")
    parser.add_argument("--untar",    metavar="TAR", default=None,
                        help="Unpack this tar.gz before starting (CHTC use).")
    parser.add_argument("--lambda",   dest="lam", type=float, default=LAMBDA,
                        help=f"Fraction of image width to mask (default: {LAMBDA}).")
    parser.add_argument("--generations", type=int, default=N_GENERATIONS)
    args = parser.parse_args()

    if args.untar:
        unpack_tar(args.untar)

    local    = args.local
    lam      = args.lam
    n_images = N_IMAGES_LOCAL if local else N_IMAGES_FULL
    duration = DURATION_LOCAL if local else DURATION_FULL
    batch    = BATCH_LOCAL    if local else BATCH_FULL

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(EDM_DIR.resolve()))

    print(f"\n{'='*60}", flush=True)
    print(f"EDM Inpainting Experiment")
    print(f"  Lambda (mask fraction) : {lam:.2f}  ({int(lam*100)}% of width masked)")
    print(f"  Generations            : {args.generations}")
    print(f"  Images/gen             : {n_images}")
    print(f"  Train duration         : {duration} Mimg")
    print(f"  Mode                   : {'LOCAL TEST' if local else 'SERVER'}")
    print(f"{'='*60}\n", flush=True)

    # Extract CIFAR-10 train images (used as gen-0 training data)
    real_images_dir = OUTPUT_DIR / "real_images_train"
    if not real_images_dir.exists():
        print("  Extracting CIFAR-10 train images...", flush=True)
        real_images_dir.mkdir(parents=True)
        run([sys.executable, EDM_DIR / "dataset_tool.py",
             "--source", CIFAR10_ZIP, "--dest", real_images_dir])

    # Conditioning images: second 10k slice from the train zip — never used
    # for training, so the model cannot have memorised their right halves.
    test_images_dir = OUTPUT_DIR / "real_images_cond"
    if not test_images_dir.exists():
        print("  Slicing conditioning images from CIFAR-10 train set...", flush=True)
        copy_n_images(real_images_dir, test_images_dir, n_images, offset=n_images)

    results = []
    current_synth_dir = None   # None in gen 0 → use real images

    for gen in range(args.generations):
        print(f"\n{'─'*50}", flush=True)
        print(f"Generation {gen} / {args.generations - 1}  (λ={lam:.2f})")
        print(f"{'─'*50}", flush=True)

        gen_dir   = OUTPUT_DIR / f"gen_{gen:03d}"
        train_dir = gen_dir / "training"
        stage_dir = gen_dir / "train_staging"   # n_images copied here for packing
        synth_dir = gen_dir / "synthetic_images"

        # --- 1. Select source images for training ---
        # Gen 0: real CIFAR-10.  Gen 1+: composites from previous generation.
        source_dir = current_synth_dir if current_synth_dir else real_images_dir
        print(f"  [1/4] Training source  : {source_dir}", flush=True)

        # --- 2. Pack n_images complete (unmasked) images for EDM training ---
        train_zip = gen_dir / "train_data.zip"
        print(f"  [2/4] Staging {n_images} images → {train_zip}", flush=True)
        copy_n_images(source_dir, stage_dir, n_images)
        pack_images_to_zip(stage_dir, train_zip)
        shutil.rmtree(stage_dir, ignore_errors=True)

        # --- 3. Train on complete images ---
        print(f"  [3/4] Training (duration={duration} Mimg, batch={batch})...", flush=True)
        train_model(train_zip, train_dir, duration, batch, local)
        network_pkl = find_latest_snapshot(train_dir)
        print(f"  Checkpoint: {network_pkl}", flush=True)

        # --- 4. Conditional inpainting: real left half + model right half ---
        print(f"  [4/4] Inpainting {n_images} images (λ={lam:.2f})...", flush=True)
        generate_inpainted_images(network_pkl, test_images_dir, synth_dir,
                                   n_images, lam, batch)

        # --- 6. Metrics ---
        print(f"  [6/6] Computing metrics...", flush=True)

        pixel_var = compute_pixel_variance(synth_dir)
        print(f"  Pixel variance : {pixel_var:.4f}", flush=True)

        # MSE on the masked region vs real images
        masked_mse = compute_masked_mse(synth_dir, test_images_dir, lam)
        print(f"  Masked MSE     : {masked_mse:.4f}", flush=True)

        fid = compute_fid(synth_dir, FID_REF_NPZ, batch, local)
        print(f"  FID            : {fid:.4f}", flush=True)

        record = {
            "generation":  gen,
            "lambda":      lam,
            "fid":         fid,
            "pixel_variance": pixel_var,
            "masked_mse":  masked_mse,
        }
        results.append(record)
        print(f"\n  ✓ Gen {gen}: FID={fid:.2f}  var={pixel_var:.4f}  masked_mse={masked_mse:.4f}",
              flush=True)

        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2)

        # Clean up large files
        train_zip.unlink(missing_ok=True)
        for pt in train_dir.glob("training-state-*.pt"):
            if pt.exists():
                pt.unlink()

        current_synth_dir = synth_dir

    # ---------------------------------------------------------------------------
    # Final metrics summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}", flush=True)
    print(f"{'Gen':<5} {'Source':<20} {'FID':>8} {'PixelVar':>10} {'MaskedMSE':>12}")
    print(f"{'-'*5} {'-'*20} {'-'*8} {'-'*10} {'-'*12}")
    for r in results:
        source = "real CIFAR-10" if r["generation"] == 0 else f"gen_{r['generation']-1:03d} synth"
        print(f"{r['generation']:<5} {source:<20} {r['fid']:>8.2f} {r['pixel_variance']:>10.4f} {r['masked_mse']:>12.4f}")
    if len(results) == 2:
        delta_fid = results[1]["fid"] - results[0]["fid"]
        delta_var = results[1]["pixel_variance"] - results[0]["pixel_variance"]
        delta_mse = results[1]["masked_mse"] - results[0]["masked_mse"]
        print(f"{'Δ':<5} {'gen0 → gen1':<20} {delta_fid:>+8.2f} {delta_var:>+10.4f} {delta_mse:>+12.4f}")
    print(f"{'='*60}\n", flush=True)

    # ---------------------------------------------------------------------------
    # Summary plot
    # ---------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gens      = [r["generation"]    for r in results]
        fids      = [r["fid"]           for r in results]
        vars_     = [r["pixel_variance"] for r in results]
        mses      = [r["masked_mse"]    for r in results]

        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        axes[0].plot(gens, fids,   marker="o", color="steelblue")
        axes[0].set_ylabel("FID")
        axes[0].set_title(f"Inpainting Experiment  (λ={lam:.2f}, {int(lam*100)}% masked)\n"
                          "Gen 0 = real data → Gen 1 = synthetic data")
        axes[0].grid(True)

        axes[1].plot(gens, vars_,  marker="o", color="darkorange")
        axes[1].set_ylabel("Pixel Variance")
        axes[1].grid(True)

        axes[2].plot(gens, mses,   marker="o", color="crimson")
        axes[2].set_ylabel("Masked Region MSE")
        axes[2].set_xlabel("Generation")
        axes[2].grid(True)

        fig.tight_layout()
        plot_path = OUTPUT_DIR / f"inpainting_lambda{int(lam*100):03d}.png"
        fig.savefig(plot_path, dpi=150)
        print(f"\n  -> Saved {plot_path}")

    except ImportError:
        print("matplotlib not available; skipping plot.")

    print(f"\nDone. Results: {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
