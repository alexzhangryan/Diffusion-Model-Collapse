#!/usr/bin/env python3
"""
Inpainting model-collapse experiment using EDM on CIFAR-10.

Each generation:
  1. Train an EDM model on the previous generation's composite images
     (real CIFAR-10 for generation 0).  Training uses COMPLETE images —
     no masking — so the model learns the full image distribution.
  2. Run conditional inpainting: a randomly-placed window of width lam*W
     is masked in each conditioning image.  The model fills in the masked
     region; the rest is replaced with the real pixel values (RePaint style).
     Output = real_unmasked_region | model_generated_masked_region.
  3. Measure quality:
       - MSE between generated composites and original images (full image)
       - Pixel variance of the composite images
       - FID vs. real CIFAR-10
  4. The composite images become the training set for the next generation.

All 50k CIFAR-10 training images are used for both training and conditioning
each generation.  No data-leak concern: the distribution shifts from real to
synthetic, so the model cannot reproduce the exact originals regardless.

LAMBDA controls the fraction of image width that is randomly masked.
  LAMBDA = 0.5  -> half the image is inpainted  (default)

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
# Configuration  ? change LAMBDA here or pass --lambda on the command line
# ---------------------------------------------------------------------------
LAMBDA = 0.5  # fraction of image width to mask (right side)

N_GENERATIONS = 10

# Full-scale (server)
N_IMAGES_FULL = 50_000   # use all 50k CIFAR-10 training images
DURATION_FULL = 25.0     # Mimg
BATCH_FULL    = 512

# Local: same parameters as server; snapshots saved every ~1 hr for easy resume
SNAP_LOCAL  = 150   # ticks between EDM snapshots (~1 hr on RTX 4080 @ ~28 s/tick)
SNAP_SERVER = 500

# Paths
EDM_DIR = Path("edm")
CIFAR10_ZIP = Path("dataset/cifar10-32x32.zip")  # 50k train images
FID_REF_NPZ = Path("cifar10-32x32.npz")
OUTPUT_DIR = Path("output")
SNAPSHOTS_DIR = Path("snapshots")
RESULTS_JSON = OUTPUT_DIR / "results.json"

IMAGE_SIZE = 32  # CIFAR-10 images are 32?32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd: list, **kwargs):
    printable = " ".join(str(c) for c in cmd)
    print(f"\n>>> {printable}\n", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def assert_cuda_available():
    """Fail fast if no GPU is visible — prevents silent CPU fallback and OOM."""
    probe = subprocess.run(
        [sys.executable, "-c",
         "import torch, sys; ok = torch.cuda.is_available(); "
         "print(f'CUDA={ok} devices={torch.cuda.device_count()}'); "
         "sys.exit(0 if ok else 1)"],
        capture_output=True, text=True,
    )
    print(f"  [GPU check] {probe.stdout.strip()}", flush=True)
    if probe.returncode != 0:
        raise RuntimeError(
            f"No GPU visible before training — aborting to avoid CPU OOM.\n"
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}\n"
            f"{probe.stderr.strip()}"
        )


def unpack_tar(tar_path: str):
    print(f"Unpacking {tar_path}...", flush=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(".")


def find_latest_snapshot(train_dir: Path) -> Path:
    snaps = sorted(train_dir.glob("network-snapshot-*.pkl"))
    if not snaps:
        raise FileNotFoundError(f"No snapshot in {train_dir}")
    return snaps[-1]


def save_generation_snapshot(gen: int, synth_dir: Path, results: list):
    """Save synthetic images and results after a completed generation."""
    snap_dir = SNAPSHOTS_DIR / f"gen_{gen:03d}"
    snap_images = snap_dir / "synthetic_images"
    if snap_images.exists():
        shutil.rmtree(snap_images)
    shutil.copytree(synth_dir, snap_images)
    (snap_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generation": gen,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"  [snapshot] Saved gen {gen} -> {snap_dir}", flush=True)


def load_latest_generation_snapshot():
    """Return (last_completed_gen, synth_dir, results) or (-1, None, [])."""
    if not SNAPSHOTS_DIR.exists():
        return -1, None, []
    completed = sorted(SNAPSHOTS_DIR.glob("gen_???"))
    if not completed:
        return -1, None, []
    latest = completed[-1]
    ckpt = latest / "checkpoint.json"
    if not ckpt.exists():
        return -1, None, []
    data = json.loads(ckpt.read_text())
    gen = data["generation"]
    synth_dir = latest / "synthetic_images"
    print(f"  [resume] Found snapshot at gen {gen} -> {latest}", flush=True)
    return gen, synth_dir, data["results"]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def copy_n_images(src_dir: Path, dst_dir: Path, n: int, offset: int = 0):
    """Copy n PNGs from src_dir starting at offset, preserving structure."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    selected = sorted(src_dir.rglob("*.png"))[offset : offset + n]
    selected_rels = set()
    for img_path in selected:
        rel = img_path.relative_to(src_dir)
        out_path = dst_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, out_path)
        selected_rels.add(str(rel).replace("\\", "/"))

    # Copy dataset.json so --cond=1 has labels.
    # Assign labels positionally (by index) since filenames may not match
    # (e.g. after inpainting renames files).
    src_json = src_dir / "dataset.json"
    if src_json.exists():
        meta = json.loads(src_json.read_text())
        if meta.get("labels"):
            all_labels = meta["labels"]
            selected_list = sorted(src_dir.rglob("*.png"))[offset : offset + n]
            new_labels = []
            for i, img_path in enumerate(selected_list):
                rel = str(img_path.relative_to(src_dir)).replace("\\", "/")
                label_idx = offset + i
                class_label = (
                    all_labels[label_idx][1] if label_idx < len(all_labels) else 0
                )
                new_labels.append([rel, class_label])
            meta["labels"] = new_labels
        (dst_dir / "dataset.json").write_text(json.dumps(meta))


def compute_image_mse(generated_dir: Path, original_dir: Path) -> float:
    """
    MSE between generated composites and original images over the full image.
    Pairs images by sorted filename. Returns NaN if no pairs found.
    """
    gen_paths = sorted(generated_dir.rglob("*.png"))
    orig_paths = sorted(original_dir.rglob("*.png"))
    n = min(len(gen_paths), len(orig_paths), 1000)  # cap at 1000 for speed
    if n == 0:
        return float("nan")

    squared_errors = []
    for gp, op in zip(gen_paths[:n], orig_paths[:n]):
        gen_arr = np.asarray(Image.open(gp).convert("RGB"), dtype=np.float32)
        orig_arr = np.asarray(Image.open(op).convert("RGB"), dtype=np.float32)
        squared_errors.append(float(np.mean((gen_arr - orig_arr) ** 2)))

    return float(np.mean(squared_errors))


def compute_pixel_variance(images_dir: Path) -> float:
    pixel_lists = []
    for img_path in images_dir.rglob("*.png"):
        arr = np.asarray(Image.open(img_path), dtype=np.float32)
        pixel_lists.append(arr.ravel())
    if not pixel_lists:
        return float("nan")
    return float(np.concatenate(pixel_lists).var())


def compute_fid(
    images_dir: Path,
    ref_npz: Path,
    batch: int,
    n_images: int = 50000,
) -> float:
    # Use plain python — same reasoning as train_model (avoid NCCL in containers).
    # Do NOT set RANK/WORLD_SIZE so EDM skips distributed init entirely.
    dist_env = dict(os.environ)
    dist_env.update({
        "USE_LIBUV": "0",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
    })
    dist_env.pop("RANK", None)
    dist_env.pop("WORLD_SIZE", None)
    dist_env.pop("LOCAL_RANK", None)
    dist_env.pop("HOSTNAME", None)  # prevent docker hostname being used as MASTER_ADDR
    result = subprocess.run(
        [sys.executable]
        + [
            EDM_DIR / "fid.py",
            "calc",
            "--images",
            images_dir,
            "--ref",
            ref_npz,
            "--num",
            str(n_images),
            "--batch",
            str(batch),
        ],
        capture_output=True,
        text=True,
        env=dist_env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"fid.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
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


def train_model(
    data_zip: Path, out_dir: Path, duration: float, batch: int, local: bool
):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Always use plain python — torchrun sets RANK/WORLD_SIZE which forces EDM
    # to init NCCL, and NCCL fails inside HTCondor containers.  With plain
    # python those env-vars are absent so EDM runs single-process without
    # touching NCCL at all.
    snap = str(SNAP_LOCAL if local else SNAP_SERVER)
    cmd = [sys.executable, EDM_DIR / "train.py",
        "--outdir", out_dir,
        "--data",   data_zip,
        "--cond=1",
        "--arch=ddpmpp",
        "--precond=vp",
        "--duration", str(duration),
        "--batch",    str(batch),
        "--nosubdir",
        "--tick=10",
        f"--snap={snap}",
        "--fp16=1",
        "--workers=4",
    ]
    # Resume mid-generation if a training state file exists (local restarts).
    resume_states = sorted(out_dir.glob("training-state-*.pt"))
    if resume_states:
        cmd += ["--resume", str(resume_states[-1])]
        print(f"  [resume] Resuming training from {resume_states[-1]}", flush=True)
    assert_cuda_available()
    run(cmd)


def generate_inpainted_images(
    network_pkl: Path,
    cond_dir: Path,
    out_dir: Path,
    n_images: int,
    lam: float,
    batch: int,
):
    """Run replacement-based inpainting: real left half + model right half."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    run(
        [
            sys.executable,
            Path(__file__).parent / "generate_inpaint.py",
            "--network",
            network_pkl,
            "--images",
            cond_dir,
            "--outdir",
            out_dir,
            "--n",
            str(n_images),
            "--lam",
            str(lam),
            "--batch",
            str(batch),
            "--edm-dir",
            str(EDM_DIR),
        ]
    )


def pack_images_to_zip(images_dir: Path, out_zip: Path):
    if out_zip.exists():
        out_zip.unlink()
    run(
        [
            sys.executable,
            EDM_DIR / "dataset_tool.py",
            "--source",
            images_dir,
            "--dest",
            out_zip,
        ]
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="EDM inpainting collapse experiment.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Local test mode: fewer images, shorter training.",
    )
    parser.add_argument(
        "--untar",
        metavar="TAR",
        default=None,
        help="Unpack this tar.gz before starting (CHTC use).",
    )
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=LAMBDA,
        help=f"Fraction of image width to mask (default: {LAMBDA}).",
    )
    parser.add_argument("--generations", type=int, default=N_GENERATIONS)
    args = parser.parse_args()

    if args.untar:
        unpack_tar(args.untar)

    local = args.local
    lam = args.lam
    n_images = N_IMAGES_FULL
    duration = DURATION_FULL
    batch = BATCH_FULL

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(EDM_DIR.resolve()))

    print(f"\n{'='*60}", flush=True)
    print(f"EDM Inpainting Experiment")
    print(f"  Lambda (mask fraction) : {lam:.2f}  ({int(lam*100)}% of width masked)")
    print(f"  Generations            : {args.generations}")
    print(f"  Images/gen             : {n_images}")
    print(f"  Train duration         : {duration} Mimg")
    print(f"  Mode                   : {'LOCAL (snap every ~1 hr)' if local else 'SERVER'}")
    print(f"{'='*60}\n", flush=True)

    # Extract all 50k CIFAR-10 training images once.
    # The same set is used for both training (gen 0) and conditioning every gen.
    # No data-leak concern: the distribution shifts each generation, so the
    # model cannot reproduce originals exactly regardless of overlap.
    real_images_dir = OUTPUT_DIR / "real_images_train"
    if not real_images_dir.exists():
        print("  Extracting CIFAR-10 train images...", flush=True)
        real_images_dir.mkdir(parents=True)
        run([sys.executable, EDM_DIR / "dataset_tool.py",
             "--source", CIFAR10_ZIP, "--dest", real_images_dir])

    last_done, current_synth_dir, results = load_latest_generation_snapshot()
    start_gen = last_done + 1
    if start_gen > 0:
        print(f"  [resume] Resuming from generation {start_gen}", flush=True)

    for gen in range(start_gen, args.generations):
        print(f"\n{'-'*50}", flush=True)
        print(f"Generation {gen} / {args.generations - 1}  (lam={lam:.2f})")
        print(f"{'-'*50}", flush=True)

        gen_dir   = OUTPUT_DIR / f"gen_{gen:03d}"
        train_dir = gen_dir / "training"
        stage_dir = gen_dir / "train_staging"
        synth_dir = gen_dir / "synthetic_images"

        # Conditioning always uses the full 50k real images every generation.
        cond_dir = real_images_dir

        # --- 1. Select source images for training ---
        # Gen 0: real images.  Gen 1+: composites from previous generation.
        source_dir = current_synth_dir if current_synth_dir else real_images_dir
        print(f"  [1/4] Training source  : {source_dir}", flush=True)
        print(f"  [1/4] Conditioning on  : {cond_dir}", flush=True)

        # --- 2. Pack n_images complete (unmasked) images for EDM training ---
        train_zip = gen_dir / "train_data.zip"
        print(f"  [2/4] Staging {n_images} images -> {train_zip}", flush=True)
        copy_n_images(source_dir, stage_dir, n_images)
        pack_images_to_zip(stage_dir, train_zip)
        shutil.rmtree(stage_dir, ignore_errors=True)

        # --- 3. Train on complete images ---
        print(f"  [3/4] Training (duration={duration} Mimg, batch={batch})...", flush=True)
        train_model(train_zip, train_dir, duration, batch, local)
        network_pkl = find_latest_snapshot(train_dir)
        print(f"  Checkpoint: {network_pkl}", flush=True)

        # --- 4. Conditional inpainting: real left half + model right half ---
        print(f"  [4/4] Inpainting {n_images} images (lam={lam:.2f})...", flush=True)
        generate_inpainted_images(network_pkl, cond_dir, synth_dir, n_images, lam, batch)
        # Propagate labels so next generation can train with --cond=1
        src_json = cond_dir / "dataset.json"
        if src_json.exists():
            shutil.copy2(src_json, synth_dir / "dataset.json")

        # --- 5. Metrics ---
        print(f"  [5/5] Computing metrics...", flush=True)

        pixel_var = compute_pixel_variance(synth_dir)
        image_mse = compute_image_mse(synth_dir, cond_dir)
        fid       = compute_fid(synth_dir, FID_REF_NPZ, batch, n_images)

        print(f"  Pixel variance : {pixel_var:.4f}", flush=True)
        print(f"  Image MSE      : {image_mse:.4f}", flush=True)
        print(f"  FID            : {fid:.4f}", flush=True)

        record = {
            "generation":     gen,
            "lambda":         lam,
            "fid":            fid,
            "pixel_variance": pixel_var,
            "image_mse":      image_mse,
        }
        results.append(record)
        print(
            f"\n  OK Gen {gen}: FID={fid:.2f}  var={pixel_var:.4f}  image_mse={image_mse:.4f}",
            flush=True,
        )

        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2)

        save_generation_snapshot(gen, synth_dir, results)

        # Clean up large files; keep training-state on local for mid-gen resume
        train_zip.unlink(missing_ok=True)
        if not local:
            for pt in train_dir.glob("training-state-*.pt"):
                if pt.exists():
                    pt.unlink()

        current_synth_dir = synth_dir

    # ---------------------------------------------------------------------------
    # Final metrics summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}", flush=True)
    print(f"{'Gen':<5} {'Source':<20} {'FID':>8} {'PixelVar':>10} {'ImageMSE':>10}")
    print(f"{'-'*5} {'-'*20} {'-'*8} {'-'*10} {'-'*10}")
    for r in results:
        source = "real CIFAR-10" if r["generation"] == 0 else f"gen_{r['generation']-1:03d} synth"
        print(f"{r['generation']:<5} {source:<20} {r['fid']:>8.2f} {r['pixel_variance']:>10.4f} {r['image_mse']:>10.4f}")
    print(f"{'='*60}\n", flush=True)

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
        mses  = [r["image_mse"]      for r in results]

        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        axes[0].plot(gens, fids, marker="o", color="steelblue")
        axes[0].set_ylabel("FID")
        axes[0].set_title(
            f"Inpainting Experiment  (lam={lam:.2f}, {int(lam*100)}% masked)\n"
            "Gen 0 = real data -> synthetic data"
        )
        axes[0].grid(True)

        axes[1].plot(gens, vars_, marker="o", color="darkorange")
        axes[1].set_ylabel("Pixel Variance")
        axes[1].grid(True)

        axes[2].plot(gens, mses, marker="o", color="crimson")
        axes[2].set_ylabel("Image MSE")
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
