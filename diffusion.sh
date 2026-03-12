#!/usr/bin/env bash
# diffusion.sh — Run the full EDM model-collapse experiment.
#
# Usage:
#   ./diffusion.sh              # full server run (50 generations, 200 Mimg)
#   ./diffusion.sh --local      # quick local test (200 images, 1 Mimg)
#
# On CHTC (HTCondor), this script is the job executable.
# The submit file transfers experiment.tar.gz; this script unpacks it first.

set -euo pipefail

# Force unbuffered Python output so logs appear in HTCondor's .out file in real time
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

LOCAL_FLAG="${1:-}"

# ---------------------------------------------------------------------------
# Step 0: Unpack experiment bundle if running inside an HTCondor job
# ---------------------------------------------------------------------------
if [ -f "experiment.tar.gz" ] && [ ! -d "edm" ]; then
    echo ">>> [0/3] Unpacking experiment.tar.gz..."
    tar -xzf experiment.tar.gz
fi

# ---------------------------------------------------------------------------
# Step 1: Convert raw CIFAR-10 dataset -> EDM zip format
# ---------------------------------------------------------------------------
CIFAR_ZIP="dataset/cifar10-32x32.zip"

if [ ! -f "$CIFAR_ZIP" ]; then
    if [ ! -d "dataset" ]; then
        echo "ERROR: dataset/ directory not found."
        echo "       Place the raw CIFAR-10 images under dataset/ and re-run."
        exit 1
    fi
    echo ">>> [1/3] Converting raw CIFAR-10 to EDM zip format..."
    python edm/dataset_tool.py --source=dataset/ --dest="$CIFAR_ZIP"
else
    echo ">>> [1/3] CIFAR-10 zip already exists: $CIFAR_ZIP  (skipping)"
fi

# ---------------------------------------------------------------------------
# Step 2: Generate FID reference statistics
# ---------------------------------------------------------------------------
FID_REF="cifar10-32x32.npz"

if [ ! -f "$FID_REF" ]; then
    echo ">>> [2/3] Generating FID reference statistics..."
    torchrun --standalone --nproc_per_node=1 edm/fid.py ref \
        --data="$CIFAR_ZIP" \
        --dest="$FID_REF"
else
    echo ">>> [2/3] FID reference stats already exist: $FID_REF  (skipping)"
fi

# ---------------------------------------------------------------------------
# Step 3: Run the model-collapse experiment
# ---------------------------------------------------------------------------
mkdir -p output
echo ">>> [3/3] Starting diffusion model collapse experiment..."

if [ "$LOCAL_FLAG" = "--local" ]; then
    echo "    Mode: LOCAL TEST (200 images, 1 Mimg training)"
    python -u diffusion_model.py --local
else
    echo "    Mode: SERVER (50 000 images/gen, 50 Mimg training, 10 generations, ~12-15h)"
    python -u diffusion_model.py
fi

echo ""
echo "Done. Results saved to output/"
