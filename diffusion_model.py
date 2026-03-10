#!/usr/bin/env python3
"""
Model collapse experiment using an EDM diffusion model on CIFAR-10.

Lambda = 1: each generation fully replaces the training dataset with synthetic
images (no real data retained).  The model trained on generation N becomes the
data source for generation N+1.

Metrics tracked per generation:
  - Pixel variance  (mean variance across all pixel values of generated images)
  - FID             (Frechet Inception Distance vs. real CIFAR-10)

Trains the cifar10-32x32-cond-vp configuration:
  --arch=ddpmpp  --precond=vp  --cond=1

--- CHTC / HTCondor packaging ---
Before submitting your job, build a tar that contains this script, the EDM
source tree, the pre-processed CIFAR-10 zip, and the FID reference file:

    # One-time: convert raw CIFAR-10 to EDM zip format (run locally)
    python edm/dataset_tool.py --source=dataset/ --dest=cifar10-32x32.zip

    # One-time: download FID reference stats (run locally, requires internet)
    torchrun --standalone --nproc_per_node=1 edm/fid.py ref \\
        --data=cifar10-32x32.zip --dest=cifar10-32x32.npz

    # Bundle everything
    tar -czf experiment.tar.gz edm/ cifar10-32x32.zip cifar10-32x32.npz diffusion_model.py

In your HTCondor .sub file:
    executable            = diffusion_model.py
    arguments             = --untar experiment.tar.gz
    transfer_input_files  = experiment.tar.gz
    transfer_output_files = output/

The --untar flag unpacks the tarball into the job's working directory before
the experiment loop begins.

--- Local testing ---
    python diffusion_model.py --local

Local mode uses 200 images per generation and 1 Mimg training duration so
the loop can be verified end-to-end on a laptop in minutes.
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
# Default configuration
# ---------------------------------------------------------------------------
N_GENERATIONS     = 10

# Full-scale (server)
N_IMAGES_FULL     = 50_000   # images to generate per generation
DURATION_FULL     = 50.0     # training duration in million images (Mimg)
BATCH_FULL        = 512

# Local test
N_IMAGES_LOCAL    = 200
DURATION_LOCAL    = 1.0
BATCH_LOCAL       = 32

# Paths (relative to the working directory where the script is launched)
EDM_DIR           = Path("edm")
CIFAR10_ZIP       = Path("dataset/cifar10-32x32.zip")   # pre-processed real dataset
FID_REF_NPZ       = Path("cifar10-32x32.npz")   # FID reference statistics
OUTPUT_DIR        = Path("output")
RESULTS_JSON      = OUTPUT_DIR / "results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, **kwargs):
    """Run a command, printing it first, and raise on non-zero exit."""
    printable = " ".join(str(c) for c in cmd)
    print(f"\n>>> {printable}\n", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def unpack_tar(tar_path: str):
    print(f"Unpacking {tar_path} into working directory...", flush=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(".")


def find_latest_snapshot(train_dir: Path) -> Path:
    """Return the most recently saved network-snapshot-*.pkl."""
    snaps = sorted(train_dir.glob("network-snapshot-*.pkl"))
    if not snaps:
        raise FileNotFoundError(f"No network snapshot found in {train_dir}")
    return snaps[-1]


def compute_pixel_variance(images_dir: Path) -> float:
    """
    Compute the mean pixel variance of all PNG images under images_dir.

    Each image is flattened to a 1-D array of uint8 values [0, 255].
    All pixel values are concatenated and the overall variance is returned.
    This is the image-domain analogue of the variance tracked in gmm.py.
    """
    pixel_lists = []
    for img_path in images_dir.rglob("*.png"):
        arr = np.asarray(Image.open(img_path), dtype=np.float32)
        pixel_lists.append(arr.ravel())
    if not pixel_lists:
        return float("nan")
    return float(np.concatenate(pixel_lists).var())


def compute_fid(images_dir: Path, ref_npz: Path, batch: int, local: bool = False) -> float:
    """
    Run edm/fid.py calc as a subprocess and parse the FID from stdout.

    fid.py prints one float (the FID score) on stdout at the end.
    All other log lines are emitted via dist.print0 and go to stdout too,
    so we collect stdout and return the last line parseable as a float.
    """
    launcher = [sys.executable] if local else ["torchrun", "--standalone", "--nproc_per_node=1"]
    result = subprocess.run(
        launcher + [
            EDM_DIR / "fid.py", "calc",
            "--images", images_dir,
            "--ref", ref_npz,
            "--batch", str(batch),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # The FID value is printed as the final numeric line
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        try:
            return float(line)
        except ValueError:
            continue
    raise RuntimeError(
        f"Could not parse FID from fid.py output:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def train_model(
    data_zip: Path,
    out_dir: Path,
    duration: float,
    batch: int,
    local: bool,
):
    """
    Train the cifar10-32x32-cond-vp EDM model on data_zip.

    Architecture flags:
      --arch=ddpmpp   DDPM++ U-Net (matches the cond-vp configuration)
      --precond=vp    VP (variance-preserving) preconditioning
      --cond=1        Class-conditional generation (10 CIFAR-10 classes)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    launcher = [sys.executable] if local else ["torchrun", "--standalone", "--nproc_per_node=1"]
    tick = "1" if local else "10"
    snap = "1" if local else "500"
    cmd = launcher + [
        EDM_DIR / "train.py",
        "--outdir", out_dir,
        "--data", data_zip,
        "--cond=1",
        "--arch=ddpmpp",
        "--precond=vp",
        "--duration", str(duration),
        "--batch", str(batch),
        "--nosubdir",
        f"--tick={tick}",
        f"--snap={snap}",
    ]
    if not local:
        cmd += ["--fp16=1", "--workers=4"]
    run(cmd)


def generate_images(
    network_pkl: Path,
    out_dir: Path,
    n_images: int,
    batch: int,
    local: bool = False,
):
    """Generate n_images synthetic images with class labels sampled uniformly."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    seeds = f"0-{n_images - 1}"
    launcher = [sys.executable] if local else ["torchrun", "--standalone", "--nproc_per_node=1"]
    run(launcher + [
        EDM_DIR / "generate.py",
        "--outdir", out_dir,
        "--seeds", seeds,
        "--subdirs",            # save images in per-class subdirectories
        "--network", network_pkl,
        "--batch", str(batch),
    ])


def pack_images_to_zip(images_dir: Path, out_zip: Path):
    """Re-pack the directory of generated PNG images into an EDM dataset zip."""
    if out_zip.exists():
        out_zip.unlink()
    run([
        sys.executable, EDM_DIR / "dataset_tool.py",
        "--source", images_dir,
        "--dest", out_zip,
    ])


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EDM diffusion model collapse experiment (lambda=1)."
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Local test mode: fewer images, shorter training (verify pipeline).",
    )
    parser.add_argument(
        "--untar", metavar="TAR", default=None,
        help="Unpack this tar.gz into the working directory before starting "
             "(used on CHTC to unpack experiment.tar.gz).",
    )
    parser.add_argument(
        "--generations", type=int, default=N_GENERATIONS,
        help=f"Number of collapse generations (default: {N_GENERATIONS}).",
    )
    parser.add_argument(
        "--n-images", type=int, default=None,
        help="Override number of images generated per generation.",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Override training duration in Mimg.",
    )
    args = parser.parse_args()

    # --- Unpack tarball if running on CHTC ---
    if args.untar:
        unpack_tar(args.untar)

    # --- Resolve configuration ---
    local    = args.local
    n_images = args.n_images or (N_IMAGES_LOCAL if local else N_IMAGES_FULL)
    duration = args.duration or (DURATION_LOCAL if local else DURATION_FULL)
    batch    = BATCH_LOCAL if local else BATCH_FULL

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure EDM is importable (fid.py / dataset_tool.py may import from it)
    sys.path.insert(0, str(EDM_DIR.resolve()))

    print(f"\n{'='*60}", flush=True)
    print(f"Diffusion Model Collapse Experiment")
    print(f"  Lambda         : 1.0  (fully synthetic each generation)")
    print(f"  Generations    : {args.generations}")
    print(f"  Images/gen     : {n_images}")
    print(f"  Train duration : {duration} Mimg")
    print(f"  Mode           : {'LOCAL TEST' if local else 'SERVER'}")
    print(f"{'='*60}\n", flush=True)

    results = []

    # Generation 0 trains on the real CIFAR-10 dataset.
    # Generations 1-N train on synthetic images from the previous generation.
    current_data_zip = CIFAR10_ZIP

    for gen in range(args.generations):
        print(f"\n{'─'*50}", flush=True)
        print(f"Generation {gen} / {args.generations - 1}")
        print(f"  Training data  : {current_data_zip}")
        print(f"{'─'*50}", flush=True)

        gen_dir   = OUTPUT_DIR / f"gen_{gen:03d}"
        train_dir = gen_dir / "training"
        synth_dir = gen_dir / "synthetic_images"

        # 1. Train model on current dataset
        print(f"  [1/5] Training model (duration={duration} Mimg, batch={batch})...", flush=True)
        train_model(current_data_zip, train_dir, duration, batch, local)
        network_pkl = find_latest_snapshot(train_dir)
        print(f"  Checkpoint     : {network_pkl}", flush=True)

        # 2. Generate synthetic images from trained model
        print(f"  [2/5] Generating {n_images} synthetic images...", flush=True)
        generate_images(network_pkl, synth_dir, n_images, batch, local)
        print(f"  Generated images saved to: {synth_dir}", flush=True)

        # 3. Pixel variance of generated images
        print(f"  [3/5] Computing pixel variance...", flush=True)
        variance = compute_pixel_variance(synth_dir)
        print(f"  Pixel variance : {variance:.4f}", flush=True)

        # 4. FID vs. real CIFAR-10 reference statistics
        print(f"  [4/5] Computing FID...", flush=True)
        fid = compute_fid(synth_dir, FID_REF_NPZ, batch, local)
        print(f"  FID            : {fid:.4f}", flush=True)

        # 5. Pack synthetic images into a zip for the next generation (lambda=1)
        print(f"  [5/5] Packing synthetic images to zip...", flush=True)
        next_data_zip = OUTPUT_DIR / f"gen_{gen:03d}_synthetic.zip"
        pack_images_to_zip(synth_dir, next_data_zip)

        # 6. Record result
        record = {"generation": gen, "fid": fid, "pixel_variance": variance}
        results.append(record)
        print(f"\n  ✓ Generation {gen} complete: FID={fid:.4f}, pixel_variance={variance:.4f}", flush=True)

        # 7. Save incremental results so partial runs are not lost
        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2)

        # 8. Free disk space: remove synthetic images dir (data is in the zip)
        #    and training checkpoints (keep only the .pkl used for generation)
        shutil.rmtree(synth_dir, ignore_errors=True)
        for pt_file in train_dir.glob("training-state-*.pt"):
            pt_file.unlink(missing_ok=True)

        # Rotate dataset: next generation trains on this generation's output
        current_data_zip = next_data_zip

    # ---------------------------------------------------------------------------
    # Summary plot
    # ---------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gens  = [r["generation"]     for r in results]
        fids  = [r["fid"]            for r in results]
        vars_ = [r["pixel_variance"] for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        ax1.plot(gens, fids, marker="o", color="steelblue")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("FID")
        ax1.set_title("FID vs. Generation  (λ=1, full synthetic replacement)")
        ax1.grid(True)

        ax2.plot(gens, vars_, marker="o", color="darkorange")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Pixel Variance")
        ax2.set_title("Pixel Variance vs. Generation  (λ=1)")
        ax2.grid(True)

        fig.tight_layout()
        plot_path = OUTPUT_DIR / "collapse_metrics.png"
        fig.savefig(plot_path, dpi=150)
        print(f"\n  -> Saved {plot_path}")

    except ImportError:
        print("matplotlib not available; skipping plot.")

    print(f"\nDone. Full results: {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
