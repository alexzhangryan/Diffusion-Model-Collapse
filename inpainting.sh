#!/usr/bin/env bash
# inpainting.sh — Run the EDM inpainting experiment on CHTC.
#
# Usage:
#   ./inpainting.sh              # full server run
#   ./inpainting.sh --local      # quick local test
#   ./inpainting.sh --lambda 0.3 # override mask fraction

set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

LOCAL_FLAG="${1:-}"

# ---------------------------------------------------------------------------
# Step 0: Unpack experiment bundle if running inside an HTCondor job
# ---------------------------------------------------------------------------
if [ -f "inpainting.tar.gz" ] && [ ! -d "edm" ]; then
    echo ">>> [0/3] Unpacking inpainting.tar.gz..."
    tar -xzf inpainting.tar.gz
    echo "    PWD: $(pwd)"
    echo "    Contents after extraction:"
    ls -la
fi

# Always ensure output dir exists before HTCondor tries to transfer it
mkdir -p output

# ---------------------------------------------------------------------------
# Step 1: Convert raw CIFAR-10 dataset -> EDM zip format
# ---------------------------------------------------------------------------
CIFAR_ZIP="dataset/cifar10-32x32.zip"

if [ ! -f "$CIFAR_ZIP" ]; then
    if [ ! -d "dataset" ]; then
        echo "ERROR: dataset/ directory not found."
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
# Step 3: Run the inpainting experiment
# ---------------------------------------------------------------------------
mkdir -p output
echo ">>> [3/3] Starting EDM inpainting experiment..."

if [ "$LOCAL_FLAG" = "--local" ]; then
    echo "    Mode: LOCAL TEST"
    python -u inpainting.py --local
else
    echo "    Mode: SERVER (10 000 images, 25 Mimg training, lambda=0.5)"
    python -u inpainting.py
fi

echo ""
echo "Done. Results saved to output/"
