#!/opt/miniconda3/envs/edm/bin/python
"""Re-runs GMM simulation and saves plots with Times New Roman font."""

import os
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
from pathlib import Path
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman"],
    "font.size":       14,
    "axes.titlesize":  15,
    "axes.labelsize":  14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

def sliced_wasserstein(X, Y, n_projections=50):
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(n_projections, X.shape[1]))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(X @ v, Y @ v) for v in vecs]))

def count_covered_modes(fitted_means, true_means, threshold):
    return sum(
        np.min(np.linalg.norm(fitted_means - tm, axis=1)) < threshold
        for tm in true_means
    )

N = 5000
N_GENERATIONS = 1000
LAMBDA_VALUES = [0, 0.1, 0.3, 0.5, 0.9, 1]
MODE_THRESHOLD = 2.0

TRUE_MEANS_2D = np.array([[-4.0, -4.0], [0.0, 4.0], [4.0, -4.0]])
TRUE_COVS_2D  = np.tile(np.eye(2) * 0.5, (3, 1, 1))
TRUE_WEIGHTS_2D = np.array([1/3, 1/3, 1/3])
N_COMPONENTS_2D = 3

_mu_bar = TRUE_WEIGHTS_2D @ TRUE_MEANS_2D
_true_cov = sum(
    TRUE_WEIGHTS_2D[k] * (TRUE_COVS_2D[k] + np.outer(TRUE_MEANS_2D[k], TRUE_MEANS_2D[k]))
    for k in range(N_COMPONENTS_2D)
) - np.outer(_mu_bar, _mu_bar)
TRUE_TRACE = float(np.trace(_true_cov))

np.random.seed(0)
comp_idx = np.random.choice(N_COMPONENTS_2D, size=N, p=TRUE_WEIGHTS_2D)
parts = [
    np.random.multivariate_normal(TRUE_MEANS_2D[k], TRUE_COVS_2D[k], max(1, int(np.sum(comp_idx == k))))
    for k in range(N_COMPONENTS_2D)
]
real_data = np.vstack(parts)[:N]
np.random.shuffle(real_data)

all_vars, all_mode_cov, all_wasserstein = {}, {}, {}

for lam in LAMBDA_VALUES:
    n_synthetic = int(N * lam)
    n_keep = N - n_synthetic
    vars_over_gen, cov_over_gen, wass_over_gen = [], [], []
    current_data = real_data.copy()

    for gen in range(N_GENERATIONS):
        vars_over_gen.append(float(np.trace(np.cov(current_data.T))))
        gmm = GaussianMixture(n_components=N_COMPONENTS_2D, max_iter=200, tol=1e-6, covariance_type="full")
        gmm.fit(current_data)
        cov_over_gen.append(count_covered_modes(gmm.means_, TRUE_MEANS_2D, MODE_THRESHOLD))
        n_sub = min(1000, len(current_data))
        idx = np.random.choice(len(current_data), n_sub, replace=False)
        wass_over_gen.append(sliced_wasserstein(current_data[idx], real_data[:n_sub]))
        if n_synthetic > 0:
            keep_idx = np.random.choice(len(current_data), n_keep, replace=False)
            synthetic_data, _ = gmm.sample(n_synthetic)
            current_data = np.vstack([current_data[keep_idx], synthetic_data])
        if gen % 100 == 0:
            print(f"lambda={lam}  gen={gen}  cov_trace={vars_over_gen[-1]:.4f}", flush=True)

    all_vars[lam] = vars_over_gen
    all_mode_cov[lam] = cov_over_gen
    all_wasserstein[lam] = wass_over_gen

out_dir = Path("multivariateout")
out_dir.mkdir(exist_ok=True)
gens = list(range(N_GENERATIONS))
suffix = "lambda=0-0.1-0.3-0.5-0.9-1_N=5000_D=2_sub.png"

# Variance
fig, ax = plt.subplots(figsize=(8, 5))
for lam in LAMBDA_VALUES:
    ax.plot(gens, all_vars[lam], label=f"λ = {lam}", zorder=1 - lam)
ax.axhline(TRUE_TRACE, color="black", linestyle="--", alpha=0.5, label="True cov trace")
margin = TRUE_TRACE * 0.6
ax.set_ylim(max(0, TRUE_TRACE - margin), TRUE_TRACE + margin)
ax.set_xlabel("Generation")
ax.set_ylabel("Covariance Trace  (tr Σ)")
ax.set_title(f"Covariance Trace over Generations  (N = {N})")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / f"variance_{suffix}", dpi=150)
print("Saved variance plot")

# Mode coverage
fig, ax = plt.subplots(figsize=(8, 5))
for lam in LAMBDA_VALUES:
    ax.plot(gens, all_mode_cov[lam], label=f"λ = {lam}", zorder=1 - lam)
ax.axhline(N_COMPONENTS_2D, color="black", linestyle="--", alpha=0.5, label="All modes")
ax.set_xlabel("Generation")
ax.set_ylabel("Modes Covered")
ax.set_ylim(0, N_COMPONENTS_2D + 0.5)
ax.set_title(f"Mode Coverage over Generations  (N = {N})")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / f"mode_coverage_{suffix}", dpi=150)
print("Saved mode coverage plot")

# Wasserstein
fig, ax = plt.subplots(figsize=(8, 5))
for lam in LAMBDA_VALUES:
    ax.plot(gens, all_wasserstein[lam], label=f"λ = {lam}", zorder=1 - lam)
ax.set_xlabel("Generation")
ax.set_ylabel("Sliced Wasserstein Distance")
ax.set_title(f"Wasserstein Distance over Generations  (N = {N})")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / f"wasserstein_{suffix}", dpi=150)
print("Saved wasserstein plot")
print("DONE")
