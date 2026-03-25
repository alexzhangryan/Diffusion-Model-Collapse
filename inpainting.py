#!/usr/bin/env python3
"""
Inpainting model-collapse experiment using EDM on CIFAR-10.

Each generation:
  1. Train an EDM model on the previous generation's generated images
     (real CIFAR-10 for generation 0).  Training uses COMPLETE images —
     no masking — so the model learns the full image distribution.
  2. Generate new images conditioned on the fully real CIFAR-10 data:
     a randomly selected lam fraction of pixels is masked per image.
     At every denoising step the unmasked pixels are replaced with the
     real image + noise at the current noise level (RePaint strategy),
     guiding the model.  The stored output is the model's full generation
     — real pixels are NOT pasted back, so training data is purely synthetic.
  3. Measure quality:
       - MSE between generated images and original real images
       - Pixel variance of the generated images
       - FID vs. real CIFAR-10
  4. The generated images become the training set for the next generation.

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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration  ? change LAMBDA here or pass --lambda on the command line
# ---------------------------------------------------------------------------
LAMBDA = 0.5  # fraction of image width to mask (right side)

N_GENERATIONS = 20

# Full-scale (server)
N_IMAGES_FULL = 50_000   # use all 50k CIFAR-10 training images
DURATION_FULL = 25.0     # Mimg
BATCH_FULL    = 512

# Checkpoint frequency: .pkl network snapshots + .pt training-state files.
# Keep only the 2 most recent .pkl per generation; older ones are pruned.
SNAP_LOCAL  = 50    # ticks between checkpoints
SNAP_SERVER = 50    # ticks between checkpoints
MAX_PKLS_PER_GEN = 1  # keep only the latest .pkl; delete when a new one is saved

# Quick smoke-test (--test): completes 5 generations in ~2-3 minutes
N_IMAGES_TEST  = 100
DURATION_TEST  = 0.001   # Mimg (~100 gradient steps)
BATCH_TEST     = 32
SNAP_TEST      = 1

# Paths
EDM_DIR = Path("edm")
CIFAR10_ZIP = Path("dataset/cifar10-32x32.zip")  # 50k train images
FID_REF_NPZ = Path("cifar10-32x32.npz")
OUTPUT_DIR = Path("output")
SNAPSHOTS_DIR = Path("snapshots")
CHECKPOINT_TAR = Path("checkpoint.tar.gz")  # single-file spool checkpoint
SAMPLES_DIR = OUTPUT_DIR / "samples"        # 100-image gallery, one subdir per gen
N_CHECKPOINT_SAMPLES = 100                  # images saved per generation
RESULTS_JSON = OUTPUT_DIR / "results.json"
WANDB_PROJECT = "inpainting-collapse"
WANDB_RUN_ID_FILE = OUTPUT_DIR / "wandb_run_id.txt"

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
    """Record completed generation — metadata only, no image copy.
    Synthetic images stay in output/gen_NNN/synthetic_images/ and survive
    eviction via transfer_output_files = output/."""
    snap_dir = SNAPSHOTS_DIR / f"gen_{gen:03d}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "checkpoint.json").write_text(
        json.dumps({"generation": gen, "synth_dir": str(synth_dir), "results": results}, indent=2)
    )
    print(f"  [snapshot] Saved gen {gen} -> {snap_dir / 'checkpoint.json'}", flush=True)


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
    synth_dir = Path(data["synth_dir"])
    if not synth_dir.exists():
        print(f"  [resume] Snapshot found for gen {gen} but synth_dir missing: {synth_dir}", flush=True)
        return -1, None, []
    print(f"  [resume] Resuming from gen {gen}, synth_dir={synth_dir}", flush=True)
    return gen, synth_dir, data["results"]


def save_generation_samples(gen: int, synth_dir: Path, n: int = N_CHECKPOINT_SAMPLES):
    """Copy up to n randomly selected images from synth_dir into the samples gallery.

    Stored at output/samples/gen_NNN/ so the gallery accumulates across all
    generations and survives eviction inside checkpoint.tar.gz.
    """
    sample_dir = SAMPLES_DIR / f"gen_{gen:03d}"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True)

    all_images = sorted(synth_dir.rglob("*.png"))
    chosen = all_images[:n] if len(all_images) <= n else [
        all_images[i] for i in sorted(
            np.random.choice(len(all_images), n, replace=False)
        )
    ]
    for img_path in chosen:
        shutil.copy2(img_path, sample_dir / img_path.name)
    print(f"  [samples] Saved {len(chosen)} samples -> {sample_dir}", flush=True)


def save_checkpoint(gen: int, synth_dir: Path, results: list, train_dir: Path = None):
    """Pack the minimum state needed to resume into a single checkpoint.tar.gz.

    Contains:
    - snapshots/           (generation metadata, tiny)
    - synth_dir            (latest completed gen's synthetic images; skipped for
                            real_images_train since it is re-extracted from the zip)
    - train_dir (optional) (current in-progress training dir: latest .pt + .pkl,
                            so EDM can resume mid-generation via --resume)
    - output/samples/      (100-image gallery from every completed generation)
    - output/results.json  (accumulated metrics)

    A single tar transfers cleanly through the HTCondor spool without the
    directory-stripping that affects folder entries in transfer_output_files.
    Written atomically via a .tmp file to avoid a partial tar on eviction.
    """
    real_images_dir = OUTPUT_DIR / "real_images_train"
    tmp = CHECKPOINT_TAR.with_suffix(".tmp.gz")
    with tarfile.open(tmp, "w:gz") as tf:
        if SNAPSHOTS_DIR.exists():
            tf.add(SNAPSHOTS_DIR)
        # Skip packing real_images_train — it is re-extracted from cifar zip on restart.
        if synth_dir is not None and synth_dir.exists() and synth_dir != real_images_dir:
            tf.add(synth_dir)
        # Pack latest training-state .pt and network .pkl for mid-generation resume.
        if train_dir is not None and train_dir.exists():
            for pt in sorted(train_dir.glob("training-state-*.pt"))[-1:]:
                tf.add(pt)
            for pkl in sorted(train_dir.glob("network-snapshot-*.pkl"))[-1:]:
                tf.add(pkl)
        if SAMPLES_DIR.exists():
            tf.add(SAMPLES_DIR)
        results_file = OUTPUT_DIR / "results.json"
        if results_file.exists():
            tf.add(results_file)
        if WANDB_RUN_ID_FILE.exists():
            tf.add(WANDB_RUN_ID_FILE)
    tmp.replace(CHECKPOINT_TAR)
    mid = " (mid-training)" if train_dir else ""
    print(f"  [checkpoint] Saved {CHECKPOINT_TAR} (gen {gen}{mid})", flush=True)


def load_checkpoint():
    """Extract checkpoint.tar.gz if present (delivered here by HTCondor spool on restart)."""
    if not CHECKPOINT_TAR.exists():
        return
    print("  [resume] Found checkpoint.tar.gz — extracting...", flush=True)
    try:
        with tarfile.open(CHECKPOINT_TAR) as tf:
            tf.extractall(".")
        print("  [resume] Checkpoint restored.", flush=True)
    except Exception as e:
        print(f"  [resume] WARNING: checkpoint extraction failed ({e}), starting fresh.", flush=True)


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
    gen: int, data_zip: Path, out_dir: Path, duration: float, batch: int, local: bool, test: bool = False,
    checkpoint_callback=None,
):
    import re
    out_dir.mkdir(parents=True, exist_ok=True)
    # Always use plain python — torchrun sets RANK/WORLD_SIZE which forces EDM
    # to init NCCL, and NCCL fails inside HTCondor containers.  With plain
    # python those env-vars are absent so EDM runs single-process without
    # touching NCCL at all.
    snap = str(SNAP_TEST if test else SNAP_LOCAL if local else SNAP_SERVER)
    cmd = [str(c) for c in [sys.executable, EDM_DIR / "train.py",
        "--outdir", out_dir,
        "--data",   data_zip,
        "--cond=1",
        "--arch=ddpmpp",
        "--precond=edm",
        "--duration", str(duration),
        "--batch",    str(batch),
        "--nosubdir",
        "--tick=10",
        f"--snap={snap}",
        f"--dump={snap}",
    ]]
    if not test:
        cmd += ["--fp16=1", "--workers=4"]
    # Resume mid-generation if a training state file exists (local restarts).
    resume_states = sorted(out_dir.glob("training-state-*.pt"))
    if resume_states:
        cmd += ["--resume", str(resume_states[-1])]
        print(f"  [resume] Resuming training from {resume_states[-1]}", flush=True)
    if not test:
        assert_cuda_available()

    # Stream output line-by-line so we can parse tick stats for W&B.
    # tick 0   kimg 0.0   time 5s   sec/tick 0.9   sec/kimg 27.66   gpumem 6.88 ...
    tick_re = re.compile(
        r"tick\s+(\d+)\s+kimg\s+([\d.]+).*?sec/tick\s+([\d.]+).*?sec/kimg\s+([\d.]+)"
        r"(?:.*?gpumem\s+([\d.]+))?"
    )
    printable = " ".join(cmd)
    print(f"\n>>> {printable}\n", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        m = tick_re.search(line)
        if m:
            tick, kimg, sec_per_tick, sec_per_kimg, gpumem = m.groups()
            tick_num = int(tick)
            loss = None

            # Read all stats from stats.jsonl (EDM writes it after each tick).
            stats_file = out_dir / "stats.jsonl"
            parsed_stats = {}
            if stats_file.exists():
                try:
                    last_line = stats_file.read_text().strip().splitlines()[-1]
                    raw = json.loads(last_line)
                    for key, val in raw.items():
                        if isinstance(val, dict) and "mean" in val:
                            parsed_stats[key] = val["mean"]
                    loss = parsed_stats.get("Loss/loss")
                    if loss is not None:
                        print(f"  [loss] {loss:.6f}", flush=True)
                except Exception:
                    pass

            # Prune mid-training to bound disk use.
            # Keep only the latest .pt (only ever need one for resume).
            for old_pt in sorted(out_dir.glob("training-state-*.pt"))[:-1]:
                old_pt.unlink()
            # Keep only the latest MAX_PKLS_PER_GEN .pkl snapshots.
            for old_pkl in sorted(out_dir.glob("network-snapshot-*.pkl"))[:-MAX_PKLS_PER_GEN]:
                old_pkl.unlink()

            # Save a mid-training checkpoint one tick AFTER EDM writes its state
            # files.  EDM prints the tick line first, then writes .pt/.pkl —
            # so at tick N the files from tick N are not yet on disk.  We fire
            # at tick N+1 (i.e. when tick_num-1 is a multiple of snap) so the
            # state from the previous snap boundary is safely on disk.
            if checkpoint_callback is not None and tick_num > 1 and (tick_num - 1) % int(snap) == 0:
                checkpoint_callback(out_dir)

            if WANDB_AVAILABLE:
                log_dict = {
                    "train/tick":         int(tick),
                    "train/kimg":         float(kimg),
                    "train/sec_per_tick": float(sec_per_tick),
                    "train/sec_per_kimg": float(sec_per_kimg),
                    "train/gpumem_gb":    float(gpumem) if gpumem else None,
                    "generation":         gen,
                }
                # Log all stats from stats.jsonl under train/ namespace.
                for key, val in parsed_stats.items():
                    wandb_key = "train/" + key.replace("/", "_").lower()
                    log_dict[wandb_key] = val
                wandb.log(log_dict)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def generate_inpainted_images(
    network_pkl: Path,
    cond_dir: Path,
    out_dir: Path,
    n_images: int,
    lam: float,
    batch: int,
):
    """Generate images conditioned on real data with random pixel masking.

    lam fraction of pixels are randomly masked per image.  The real image
    guides the denoising at each step (RePaint), but the stored output is
    the model's complete generation — not a composite with real pixels pasted in.
    """
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
        help="Local mode: full server params, snapshots every ~1 hr.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Smoke-test: 100 images, 0.001 Mimg, 5 gens, no FID. Completes in ~2-3 min.",
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
    test  = args.test
    lam   = args.lam

    if test:
        n_images = N_IMAGES_TEST
        duration = DURATION_TEST
        batch    = BATCH_TEST
        if args.generations == N_GENERATIONS:
            args.generations = 5
    else:
        n_images = N_IMAGES_FULL
        duration = DURATION_FULL
        batch    = BATCH_FULL
        if local and args.generations == N_GENERATIONS:
            args.generations = 5

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(EDM_DIR.resolve()))

    # Restore output/ and snapshots/ from the spool checkpoint if present.
    # Must happen before load_latest_generation_snapshot() reads snapshots/.
    load_checkpoint()

    # Guarantee checkpoint.tar.gz exists so HTCondor can always transfer it
    # back even if the job is interrupted before the first generation completes.
    if not CHECKPOINT_TAR.exists():
        with tarfile.open(CHECKPOINT_TAR, "w:gz") as _tf:
            pass

    # Initialise W&B.  Reuse the same run ID across evictions so all
    # generations appear in one continuous run on the dashboard.
    if WANDB_AVAILABLE:
        if WANDB_RUN_ID_FILE.exists():
            _wandb_id = WANDB_RUN_ID_FILE.read_text().strip()
        else:
            _wandb_id = wandb.util.generate_id()
            WANDB_RUN_ID_FILE.write_text(_wandb_id)
        wandb.init(
            project=WANDB_PROJECT,
            id=_wandb_id,
            resume="allow",
            config={
                "lambda": lam,
                "n_generations": args.generations,
                "n_images": n_images,
                "duration_mimg": duration,
                "batch": batch,
            },
        )

    print(f"\n{'='*60}", flush=True)
    print(f"EDM Inpainting Experiment")
    print(f"  Lambda (mask fraction) : {lam:.2f}  ({int(lam*100)}% of width masked)")
    print(f"  Generations            : {args.generations}")
    print(f"  Images/gen             : {n_images}")
    print(f"  Train duration         : {duration} Mimg")
    mode_str = "TEST (no FID, 0.001 Mimg)" if test else "LOCAL (snap every ~1 hr)" if local else "SERVER"
    print(f"  Mode                   : {mode_str}")
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
        # Skip training if a network snapshot already exists (e.g. recovered
        # from a previous run) — go straight to generation with the saved pkl.
        existing_pkl = find_latest_snapshot(train_dir) if train_dir.exists() else None
        if existing_pkl is not None:
            print(f"  [3/4] Skipping training — existing pkl found: {existing_pkl}", flush=True)
        else:
            print(f"  [3/4] Training (duration={duration} Mimg, batch={batch})...", flush=True)
            # Mid-training checkpoint callback: saves training-state .pt + .pkl so
            # the job can resume from within this generation after an eviction.
            _mid_synth = current_synth_dir  # prev gen's synth dir (None for gen 0)
            _mid_results = list(results)    # copy so closure captures current state
            def _ckpt_cb(td):
                save_checkpoint(gen, _mid_synth, _mid_results, train_dir=td)
            train_model(gen, train_zip, train_dir, duration, batch, local, test,
                        checkpoint_callback=_ckpt_cb)
        network_pkl = find_latest_snapshot(train_dir)
        print(f"  Checkpoint: {network_pkl}", flush=True)

        # --- 4. Generate from real data with random pixel masking ---
        print(f"  [4/4] Generating {n_images} images from real data (lam={lam:.2f} masked)...", flush=True)
        generate_inpainted_images(network_pkl, cond_dir, synth_dir, n_images, lam, batch)
        # Propagate labels so next generation can train with --cond=1
        src_json = cond_dir / "dataset.json"
        if src_json.exists():
            shutil.copy2(src_json, synth_dir / "dataset.json")

        # --- 5. Metrics ---
        print(f"  [5/5] Computing metrics...", flush=True)

        pixel_var = compute_pixel_variance(synth_dir)
        image_mse = compute_image_mse(synth_dir, cond_dir)
        fid       = -1.0 if test else compute_fid(synth_dir, FID_REF_NPZ, batch, n_images)

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

        print_summary_and_plot(results, lam)
        save_generation_samples(gen, synth_dir)
        save_generation_snapshot(gen, synth_dir, results)

        if WANDB_AVAILABLE:
            log = {
                "fid":            fid,
                "pixel_variance": pixel_var,
                "image_mse":      image_mse,
            }
            sample_dir = SAMPLES_DIR / f"gen_{gen:03d}"
            if sample_dir.exists():
                imgs = sorted(sample_dir.glob("*.png"))[:32]
                log["samples"] = [wandb.Image(str(p)) for p in imgs]
            wandb.log(log, step=gen)

        # --- Checkpoint housekeeping ---

        # Remove training data zip (large, no longer needed).
        if train_zip.exists():
            train_zip.unlink()

        # Prune this generation's mid-training .pkl files, keeping only the
        # final snapshot (highest kimg) for later evaluation.
        all_pkls = sorted(train_dir.glob("network-snapshot-*.pkl"))
        for pkl in all_pkls[:-1]:
            pkl.unlink()

        # Training-state .pt files are only needed for mid-generation resume.
        # The generation is now fully complete, so they can be removed.
        for pt in train_dir.glob("training-state-*.pt"):
            pt.unlink()

        # Pack the minimum resume state into a single tar for HTCondor spool.
        # Must be last so deleted files above are not included.
        save_checkpoint(gen, synth_dir, results)

        current_synth_dir = synth_dir

    print(f"\nDone. Results: {RESULTS_JSON}", flush=True)


def print_summary_and_plot(results: list, lam: float):
    """Print metrics table and save/overwrite the summary plot."""
    print(f"\n{'='*60}", flush=True)
    print(f"{'Gen':<5} {'Source':<20} {'FID':>8} {'PixelVar':>10} {'ImageMSE':>10}")
    print(f"{'-'*5} {'-'*20} {'-'*8} {'-'*10} {'-'*10}")
    for r in results:
        source = "real CIFAR-10" if r["generation"] == 0 else f"gen_{r['generation']-1:03d} synth"
        print(f"{r['generation']:<5} {source:<20} {r['fid']:>8.2f} {r['pixel_variance']:>10.4f} {r['image_mse']:>10.4f}")
    print(f"{'='*60}\n", flush=True)

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
        plt.close(fig)
        print(f"  -> Saved {plot_path}", flush=True)

    except ImportError:
        print("matplotlib not available; skipping plot.", flush=True)


if __name__ == "__main__":
    main()
