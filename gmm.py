#!/opt/miniconda3/envs/edm/bin/python
"""
GMM-based model collapse experiment using scikit-learn's GaussianMixture.
Supports 1D Gaussian and 2D multi-modal GMM.
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def sliced_wasserstein(X, Y, n_projections=50):
    """Approximate 2-Wasserstein distance via random 1D projections."""
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(n_projections, X.shape[1]))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(X @ v, Y @ v) for v in vecs]))


def count_covered_modes(fitted_means, true_means, threshold):
    """Count true modes with at least one fitted mean within threshold distance."""
    return sum(
        np.min(np.linalg.norm(fitted_means - tm, axis=1)) < threshold
        for tm in true_means
    )


if __name__ == "__main__":

    # --- Configuration ---
    N = 1000  # Number of real data points
    N_GENERATIONS = 5000
    LAMBDA_VALUES = [0.1, 0.3, 0.5, 0.9, 1.0]
    ACCUMULATE = False  # True: append λ*N synthetic points; False: substitute
    DIM = 2  # 1: 1D Gaussian;  2: 2D multi-modal GMM

    # --- 1D true distribution (used when DIM=1) ---
    TRUE_MEAN_1D = np.array([5.0])
    TRUE_VAR_1D = np.array([[2.0]])

    # --- 2D true distribution (used when DIM=2) ---
    TRUE_MEANS_2D = np.array([[-4.0, -4.0], [0.0, 4.0], [4.0, -4.0]])
    TRUE_COVS_2D = np.tile(np.eye(2) * 0.5, (3, 1, 1))
    TRUE_WEIGHTS_2D = np.array([1 / 3, 1 / 3, 1 / 3])
    N_COMPONENTS_2D = len(TRUE_MEANS_2D)
    MODE_THRESHOLD = 2.0  # distance within which a mode counts as "covered"

    # --- Generate real data ---
    if DIM == 1:
        real_data = np.random.multivariate_normal(TRUE_MEAN_1D, TRUE_VAR_1D, N)
        n_fit = 1
    else:
        comp_idx = np.random.choice(N_COMPONENTS_2D, size=N, p=TRUE_WEIGHTS_2D)
        parts = [
            np.random.multivariate_normal(
                TRUE_MEANS_2D[k], TRUE_COVS_2D[k], max(1, int(np.sum(comp_idx == k)))
            )
            for k in range(N_COMPONENTS_2D)
        ]
        real_data = np.vstack(parts)[:N]
        np.random.shuffle(real_data)
        n_fit = N_COMPONENTS_2D

    # --- Run experiment for each lambda ---
    all_vars = {}
    all_mode_cov = {}  # 2D only
    all_wasserstein = {}  # 2D only
    final_data = {}

    for lam in LAMBDA_VALUES:
        n_synthetic = int(N * lam)
        real_subset = real_data[: N - n_synthetic].copy() if not ACCUMULATE else None

        vars_over_gen = []
        cov_over_gen = []
        wass_over_gen = []

        current_data = real_data.copy()

        for gen in range(N_GENERATIONS):
            # Track variance (trace/d for multivariate)
            if DIM == 1:
                vars_over_gen.append(float(current_data.var()))
            else:
                vars_over_gen.append(float(np.trace(np.cov(current_data.T)) / DIM))

            # Fit GMM
            gmm = GaussianMixture(
                n_components=n_fit,
                max_iter=200,
                tol=1e-6,
                covariance_type="full",
            )
            gmm.fit(current_data)

            # 2D-only metrics
            if DIM == 2:
                cov_over_gen.append(
                    count_covered_modes(gmm.means_, TRUE_MEANS_2D, MODE_THRESHOLD)
                )
                # Subsample for Wasserstein to keep computation tractable
                n_sub = min(1000, len(current_data))
                idx = np.random.choice(len(current_data), n_sub, replace=False)
                wass_over_gen.append(
                    sliced_wasserstein(current_data[idx], real_data[:n_sub])
                )

            # Generate and update dataset
            synthetic_data, _ = gmm.sample(n_synthetic)
            if ACCUMULATE:
                current_data = np.vstack([current_data, synthetic_data])
            else:
                current_data = np.vstack([real_subset, synthetic_data])

            if gen % 10 == 0 or gen == N_GENERATIONS - 1:
                print(
                    f"lambda={lam:.1f}  gen={gen:3d}  "
                    f"n={len(current_data):6d}  var={vars_over_gen[-1]:.4f}"
                )

        all_vars[lam] = vars_over_gen
        all_mode_cov[lam] = cov_over_gen
        all_wasserstein[lam] = wass_over_gen
        final_data[lam] = current_data.copy()

    generations = list(range(N_GENERATIONS))

    # --- Figure 1: Variance over generations ---
    fig_var, ax_var = plt.subplots(figsize=(8, 5))
    for lam in LAMBDA_VALUES:
        ax_var.plot(generations, all_vars[lam], label=f"λ = {lam}", zorder=1 - lam)
    if DIM == 1:
        ax_var.axhline(
            TRUE_VAR_1D[0, 0],
            color="black",
            linestyle="--",
            alpha=0.5,
            label="True variance",
        )
    ax_var.set_xlabel("Generation")
    ax_var.set_ylabel("Variance" if DIM == 1 else "Mean variance (trace / d)")
    ax_var.set_ylim(bottom=0)
    ax_var.set_title(f"Variance over Generations  (N = {N} data points)")
    ax_var.legend()
    fig_var.tight_layout()
    fig_var.savefig("gmm_variance.png", dpi=150)
    print("  -> Saved gmm_variance.png")

    if DIM == 2:
        # --- Figure 1b: Mode coverage ---
        fig_cov, ax_cov = plt.subplots(figsize=(8, 5))
        for lam in LAMBDA_VALUES:
            ax_cov.plot(
                generations, all_mode_cov[lam], label=f"λ = {lam}", zorder=1 - lam
            )
        ax_cov.axhline(
            N_COMPONENTS_2D, color="black", linestyle="--", alpha=0.5, label="All modes"
        )
        ax_cov.set_xlabel("Generation")
        ax_cov.set_ylabel("Modes covered")
        ax_cov.set_ylim(0, N_COMPONENTS_2D + 0.5)
        ax_cov.set_title(f"Mode Coverage over Generations  (N = {N} data points)")
        ax_cov.legend()
        fig_cov.tight_layout()
        fig_cov.savefig("gmm_mode_coverage.png", dpi=150)
        print("  -> Saved gmm_mode_coverage.png")

        # --- Figure 1c: Wasserstein distance ---
        fig_wass, ax_wass = plt.subplots(figsize=(8, 5))
        for lam in LAMBDA_VALUES:
            ax_wass.plot(
                generations, all_wasserstein[lam], label=f"λ = {lam}", zorder=1 - lam
            )
        ax_wass.set_xlabel("Generation")
        ax_wass.set_ylabel("Sliced Wasserstein distance")
        ax_wass.set_title(
            f"Wasserstein Distance over Generations  (N = {N} data points)"
        )
        ax_wass.legend()
        fig_wass.tight_layout()
        fig_wass.savefig("gmm_wasserstein.png", dpi=150)
        print("  -> Saved gmm_wasserstein.png")

    # --- Figure 2: Distribution comparisons per lambda ---
    n_lambdas = len(LAMBDA_VALUES)
    if DIM == 1:
        fig_dist, axes_dist = plt.subplots(1, n_lambdas, figsize=(4 * n_lambdas, 5))
        fig_dist.suptitle(
            f"Final Distributions after {N_GENERATIONS} Generations  (N = {N} data points)"
        )
        for i, lam in enumerate(LAMBDA_VALUES):
            ax = axes_dist[i]
            ax.hist(
                real_data.flatten(),
                bins=50,
                density=True,
                alpha=0.5,
                color="blue",
                label="Real data",
            )
            ax.hist(
                final_data[lam].flatten(),
                bins=50,
                density=True,
                alpha=0.5,
                color="red",
                label=f"Gen {N_GENERATIONS}",
            )
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.set_title(f"λ = {lam}")
            ax.legend()
    else:
        fig_dist, axes_dist = plt.subplots(1, n_lambdas, figsize=(4 * n_lambdas, 4))
        fig_dist.suptitle(
            f"Final 2D Distributions after {N_GENERATIONS} Generations  (N = {N} data points)"
        )
        for i, lam in enumerate(LAMBDA_VALUES):
            ax = axes_dist[i]
            ax.scatter(
                TRUE_MEANS_2D[:, 0],
                TRUE_MEANS_2D[:, 1],
                marker="*",
                s=60,
                color="black",
                zorder=5,
                label="True modes",
            )
            n_plot = min(500, len(real_data))
            idx_r = np.random.choice(len(real_data), n_plot, replace=False)
            ax.scatter(
                real_data[idx_r, 0],
                real_data[idx_r, 1],
                alpha=0.3,
                s=5,
                color="blue",
                label="Real",
            )
            n_plot_f = min(500, len(final_data[lam]))
            idx_f = np.random.choice(len(final_data[lam]), n_plot_f, replace=False)
            ax.scatter(
                final_data[lam][idx_f, 0],
                final_data[lam][idx_f, 1],
                alpha=0.3,
                s=5,
                color="red",
                label=f"Gen {N_GENERATIONS}",
            )
            ax.set_title(f"λ = {lam}")
            ax.legend(markerscale=3, fontsize=7)
    fig_dist.tight_layout()
    fig_dist.savefig("gmm_distributions.png", dpi=150)
    print("  -> Saved gmm_distributions.png")

    plt.show()
