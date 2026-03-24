# DiffusionCollapse

> Investigating model collapse in generative models — how iterative training on synthetic data degrades output distributions.

![GMM Variance Collapse](gmm_variance_comparison.png)

## Overview

This project studies **model collapse**: the phenomenon where generative models progressively lose fidelity when trained on their own outputs. We start with a Gaussian Mixture Model (GMM) toy problem to build intuition, then scale to diffusion models ([EDM](https://github.com/NVlabs/edm)) using a conditional inpainting approach on CIFAR-10.

### Key Research Questions

- What is the **minimum proportion of real data** needed to prevent collapse?
- Does **data accumulation** (keeping old data) prevent collapse, while **data replacement** causes it?
- How does the synthetic-to-real data ratio affect distribution degradation over generations?

## Project Structure

```
├── inpainting.py              # Main EDM inpainting collapse experiment
├── generate_inpaint.py        # Replacement-based inpainting sampler (RePaint)
├── inpainting.sh              # CHTC job entry point
├── inpainting.sub             # HTCondor submit file
├── submit.sh                  # Submit wrapper (creates required dirs automatically)
├── Dockerfile                 # Container definition (PyTorch 1.12.1 + CUDA 11.3)
├── gmm.py                     # GMM model collapse experiment
├── generate_gmm_samples.py    # GMM sample generation utility
├── gmm_variance_comparison.png # GMM experiment output plot
├── edm/                       # NVIDIA EDM codebase (gitignored)
├── dataset/                   # CIFAR-10 training data (gitignored)
├── output/                    # Generated images, metrics, plots (gitignored)
├── snapshots/                 # Generation-level resume checkpoints (gitignored)
└── gmm_out/                   # GMM samples and statistics (gitignored)
```

## Getting Started

### Prerequisites

- Python 3.8–3.9
- [Conda](https://docs.conda.io/en/latest/) or Docker (for CHTC)
- CUDA-capable GPU (recommended for full-scale runs)

### Installation

```bash
git clone https://github.com/alexzhangryan/DiffusionCollapse
cd DiffusionCollapse
git clone https://github.com/NVlabs/edm.git

conda create -n edm python=3.9
conda activate edm
pip install torch==1.12.1 numpy scipy pillow tqdm scikit-learn matplotlib imageio pyspng
```

### Dataset

Download CIFAR-10 in EDM zip format and generate FID reference statistics (one-time setup):

```bash
# Convert CIFAR-10 to EDM format
python edm/dataset_tool.py --source=dataset/ --dest=dataset/cifar10-32x32.zip

# Generate FID reference statistics
python edm/fid.py ref \
    --data=dataset/cifar10-32x32.zip \
    --dest=cifar10-32x32.npz
```

## Running the Inpainting Experiment

### Smoke test (~2-3 min, verifies everything works)

```bash
python inpainting.py --test
# 100 images, 0.001 Mimg training, 5 generations, no FID
```

### Local full-scale run

```bash
python inpainting.py --local
# Full server parameters (50k images, 25 Mimg), 5 generations
# Snapshots every ~1 hr; supports mid-generation resume on restart
```

### On CHTC (HTCondor)

**Step 1 — Bundle code and data** (run as one line locally):
```bash
tar -czf inpainting.tar.gz edm/ dataset/ cifar10-32x32.npz inpainting.py generate_inpaint.py inpainting.sh
```

**Step 2 — Upload to CHTC:**
```bash
scp inpainting.tar.gz inpainting.sub submit.sh <user>@ap2001.chtc.wisc.edu:~/
```

**Step 3 — Submit:**
```bash
# On CHTC — use submit.sh instead of condor_submit directly;
# it creates output/, snapshots/, and logs/ if they don't exist
bash submit.sh
```

**Step 4 — Monitor:**
```bash
condor_q                                        # check job status
tail -f logs/inpainting_<ClusterID>_0.out       # stream stdout live
```

**Resume after eviction:** handled automatically. On eviction, HTCondor transfers `output/` and `snapshots/` back to the submit node. On restart, the job resumes from the last completed generation.

### Output

Results are written after **every generation** (not just at the end):
- `output/results.json` — FID, pixel variance, image MSE per generation
- `output/inpainting_lambda050.png` — live-updating summary plot

## Experiment Design

### Inpainting Collapse (`inpainting.py`)

Each generation:
1. **Train** an EDM model (VP precond, ddpmpp arch) on the previous generation's composite images. Generation 0 trains on real CIFAR-10.
2. **Inpaint**: for each conditioning image, randomly mask 50% of pixels. The trained model fills in the masked region; real pixels are preserved (RePaint-style replacement sampling). Output = real unmasked region | model-generated masked region.
3. **Measure**: pixel variance, image MSE vs. originals, FID vs. real CIFAR-10.
4. Composite images become the next generation's training data.

All 50k CIFAR-10 training images are used for both training and conditioning each generation. No train/condition separation is needed — the distribution shifts each generation regardless of overlap.

**Key parameters:**

| Parameter | Value |
|---|---|
| Images per generation | 50,000 |
| Training duration | 25 Mimg |
| Batch size | 512 |
| Mask fraction (λ) | 0.5 (random position) |
| Generations | 10 |
| Architecture | EDM ddpmpp, VP preconditioning |

### GMM Collapse Experiment (`gmm.py`)

Simulates model collapse on a 1-D Gaussian:

1. Start with `N = 1000` real samples from `N(5.0, 2.0)`
2. At each generation, fit a GMM to the current dataset
3. Replace fraction **λ** of the data with synthetic samples
4. Track variance over 5,000 generations

| λ | Synthetic % | Behavior |
|---|---|---|
| `0.1` | 10% | Variance stable near true value |
| `0.9` | 90% | Mean preserved, variance unstable |
| `1.0` | 100% | Complete collapse to a point mass |

```bash
python gmm.py
```

## Key Findings

From the GMM experiment:
- **Pure replacement (λ = 1.0)** causes rapid, complete collapse — variance drops to zero.
- **High synthetic ratio (λ = 0.9)** preserves the mean but variance becomes unstable.
- **Low synthetic ratio (λ = 0.1)** maintains a stable distribution close to the true data.
- Even a **small fraction of real data** can anchor the distribution and prevent total collapse.

## Literature

1. **"AI Models Collapse When Trained on Recursively Generated Data"** — Shumailov et al. Models amplify their own errors through recursive training.
2. **"Strong Model Collapse"** — Dohmatob et al. ([arXiv:2410.04840](https://arxiv.org/abs/2410.04840)) — Even 1% synthetic data can induce collapse; larger models may amplify it.
3. **"Is Model Collapse Inevitable?"** — Dohmatob et al. ([arXiv:2404.01413](https://arxiv.org/abs/2404.01413)) — Data accumulation prevents collapse; data replacement causes it.
4. **"Theoretical Perspective on Mitigating Model Collapse"** — ([arXiv:2502.18865](https://arxiv.org/abs/2502.18865)) — Recursive stability and architecture choice matter for collapse resistance.

## Roadmap

- [ ] Sweep over λ values (mask fraction) and compare collapse rates
- [ ] Implement data accumulation scenario (vs. current replacement)
- [ ] Add bias-variance decomposition metrics
- [ ] Scale to higher-resolution datasets

## License

This project is for research purposes. The EDM codebase is subject to [NVIDIA's license](https://github.com/NVlabs/edm/blob/main/LICENSE.txt).
