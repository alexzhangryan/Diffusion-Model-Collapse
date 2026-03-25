#!/opt/miniconda3/envs/edm/bin/python
"""
GMM-based model collapse experiment using scikit-learn's GaussianMixture.
Supports 1D Gaussian and 2D multi-modal GMM.
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
from pathlib import Path
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def out_path(description: str, lam_values: list, N: int, accumulate: bool, dim: int) -> Path:
    """Build output path and create directory if needed.

    1D  → 1doutput/Description_lambda=X-Y_N=N_acc|sub.png
    nD  → multivariateout/Description_lambda=X-Y_N=N_D=d_acc|sub.png
    """
    lam_str = "-".join(str(l) for l in lam_values)
    acc_str = "acc" if accumulate else "sub"
    if dim == 1:
        directory = Path("1doutput")
        fname = f"{description}_lambda={lam_str}_N={N}_{acc_str}.png"
    else:
        directory = Path("multivariateout")
        fname = f"{description}_lambda={lam_str}_N={N}_D={dim}_{acc_str}.png"
    directory.mkdir(exist_ok=True)
    return directory / fname


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
    N = 5000  # Number of real data points
    N_GENERATIONS = 1000
    LAMBDA_VALUES = [0, 0.1, 0.3, 0.5, 0.9, 1]
    ACCUMULATE = False  # True: append λ*N synthetic points; False: substitute
    DIM = 2  # 1: 1D Gaussian;  2: 2D multi-modal GMM
    TUNE_LAMBDA = False  # False: skip lambda search entirely

    # --- Recursive lambda search (only used if TUNE_LAMBDA = True) ---
    # Round 1: evaluate 4 candidates evenly spaced in [0, 1].
    # Each candidate runs for TUNE_EVAL_GENS generations; score = lambda * (mean_var / initial_var).
    # Best candidate becomes the centre of a narrower range for the next round.
    # Repeats for TUNE_ROUNDS rounds, converging on the highest lambda with least collapse.
    TUNE_ROUNDS = 5
    TUNE_N_CANDIDATES = 4
    TUNE_EVAL_GENS = 200    # generations to evaluate each candidate
    TUNE_THRESHOLD = 0.90   # variance must stay >= this fraction of initial variance

    # --- 1D true distribution (used when DIM=1) ---
    TRUE_MEAN_1D = np.array([5.0])
    TRUE_VAR_1D = np.array([[2.0]])

    # --- 2D true distribution (used when DIM=2) ---
    TRUE_MEANS_2D = np.array([[-4.0, -4.0], [0.0, 4.0], [4.0, -4.0]])
    TRUE_COVS_2D = np.tile(np.eye(2) * 0.5, (3, 1, 1))
    TRUE_WEIGHTS_2D = np.array([1 / 3, 1 / 3, 1 / 3])
    N_COMPONENTS_2D = len(TRUE_MEANS_2D)
    MODE_THRESHOLD = 2.0  # distance within which a mode counts as "covered"

    # --- True population covariance trace for the 2D GMM ---
    # Cov = sum_k w_k*(Sigma_k + mu_k mu_k^T) - mu_bar mu_bar^T
    _mu_bar = TRUE_WEIGHTS_2D @ TRUE_MEANS_2D
    _true_cov_2d = sum(
        TRUE_WEIGHTS_2D[k] * (TRUE_COVS_2D[k] + np.outer(TRUE_MEANS_2D[k], TRUE_MEANS_2D[k]))
        for k in range(N_COMPONENTS_2D)
    ) - np.outer(_mu_bar, _mu_bar)
    TRUE_TRACE_2D = float(np.trace(_true_cov_2d))

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
        n_keep = N - n_synthetic

        vars_over_gen = []
        cov_over_gen = []
        wass_over_gen = []

        current_data = real_data.copy()

        for gen in range(N_GENERATIONS):
            # Track variance (1D) or covariance trace (2D)
            if DIM == 1:
                vars_over_gen.append(float(current_data.var()))
            else:
                vars_over_gen.append(float(np.trace(np.cov(current_data.T))))

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

            # Generate and update dataset (inpainting-style)
            # λ*N points are replaced by GMM samples; the rest survive from
            # the previous generation's data — no re-anchoring to real_data.
            if n_synthetic == 0:
                pass  # nothing changes; current_data carries forward unchanged
            elif ACCUMULATE:
                synthetic_data, _ = gmm.sample(n_synthetic)
                current_data = np.vstack([current_data, synthetic_data])
            else:
                keep_idx = np.random.choice(len(current_data), n_keep, replace=False)
                synthetic_data, _ = gmm.sample(n_synthetic)
                current_data = np.vstack([current_data[keep_idx], synthetic_data])

            if gen % 10 == 0 or gen == N_GENERATIONS - 1:
                print(
                    f"lambda={lam:.1f}  gen={gen:3d}  "
                    f"n={len(current_data):6d}  {'var' if DIM == 1 else 'cov_trace'}={vars_over_gen[-1]:.4f}"
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
        true_ref = TRUE_VAR_1D[0, 0]
        ax_var.axhline(true_ref, color="black", linestyle="--", alpha=0.5, label="True variance")
        ax_var.set_ylabel("Variance")
        ax_var.set_ylim(0, true_ref * 2)
    else:
        true_ref = TRUE_TRACE_2D
        ax_var.axhline(true_ref, color="black", linestyle="--", alpha=0.5, label="True cov trace")
        ax_var.set_ylabel("Covariance trace  (tr Σ)")
        margin = true_ref * 0.6
        ax_var.set_ylim(max(0, true_ref - margin), true_ref + margin)
    ax_var.set_xlabel("Generation")
    ax_var.set_title(f"Covariance Trace over Generations  (N = {N} data points)")
    ax_var.legend()
    fig_var.tight_layout()
    p = out_path("variance", LAMBDA_VALUES, N, ACCUMULATE, DIM)
    fig_var.savefig(p, dpi=150)
    print(f"  -> Saved {p}")

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
        p = out_path("mode_coverage", LAMBDA_VALUES, N, ACCUMULATE, DIM)
        fig_cov.savefig(p, dpi=150)
        print(f"  -> Saved {p}")

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
        p = out_path("wasserstein", LAMBDA_VALUES, N, ACCUMULATE, DIM)
        fig_wass.savefig(p, dpi=150)
        print(f"  -> Saved {p}")

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
    p = out_path("distributions", LAMBDA_VALUES, N, ACCUMULATE, DIM)
    fig_dist.savefig(p, dpi=150)
    print(f"  -> Saved {p}")

    # ---------------------------------------------------------------------------
    # Recursive lambda search
    # ---------------------------------------------------------------------------
    if TUNE_LAMBDA:
        print("\n--- Recursive lambda search ---", flush=True)

        initial_var = float(real_data.var()) if DIM == 1 else float(np.trace(np.cov(real_data.T)) / DIM)

        def eval_lambda(cand_lam):
            """Run cand_lam for TUNE_EVAL_GENS generations; return mean variance."""
            data = real_data.copy()
            n_syn = int(N * cand_lam)
            real_sub = real_data[: N - n_syn].copy() if n_syn > 0 and not ACCUMULATE else None
            vars_run = []
            for _ in range(TUNE_EVAL_GENS):
                if DIM == 1:
                    vars_run.append(float(data.var()))
                else:
                    vars_run.append(float(np.trace(np.cov(data.T)) / DIM))
                g = GaussianMixture(n_components=n_fit, max_iter=200, tol=1e-6, covariance_type="full")
                g.fit(data)
                if n_syn == 0:
                    data = real_data.copy()
                elif ACCUMULATE:
                    syn, _ = g.sample(n_syn)
                    data = np.vstack([data, syn])
                else:
                    syn, _ = g.sample(n_syn)
                    data = np.vstack([real_sub, syn])
            return vars_run  # return full trace, not just mean

        lo, hi = 0.0, 1.0
        round_history = []  # list of (candidates, scores, best_lam) per round

        for round_i in range(TUNE_ROUNDS):
            candidates = list(np.linspace(lo, hi, TUNE_N_CANDIDATES))
            scores = []
            for cand_lam in candidates:
                vars_run = eval_lambda(cand_lam)
                mse = float(np.mean((np.array(vars_run) - initial_var) ** 2))
                # Score: highest lambda with lowest variance MSE; avoid div-by-zero
                score = cand_lam / (mse + 1e-9)
                scores.append(score)
                print(f"  round={round_i+1}  λ={cand_lam:.4f}  mse={mse:.6f}  score={score:.4f}", flush=True)

            best_idx = int(np.argmax(scores))
            best_lam = candidates[best_idx]
            round_history.append((candidates, scores, best_lam))

            # Narrow range to neighbours of best candidate
            lo = candidates[max(0, best_idx - 1)]
            hi = candidates[min(TUNE_N_CANDIDATES - 1, best_idx + 1)]

            print(f"  -> round {round_i+1} best: λ={best_lam:.4f}  new range=[{lo:.4f}, {hi:.4f}]\n", flush=True)

        print(f"\n  -> Final best lambda = {best_lam:.4f}")

        # --- Plot: score vs lambda for each round ---
        colors = plt.cm.viridis(np.linspace(0.15, 0.9, TUNE_ROUNDS))
        fig_tune, ax_tune = plt.subplots(figsize=(8, 5))
        for round_i, (cands, scores, best) in enumerate(round_history):
            ax_tune.plot(cands, scores, marker="o", color=colors[round_i], label=f"Round {round_i+1}")
            ax_tune.axvline(best, color=colors[round_i], linestyle=":", alpha=0.5)
        ax_tune.set_xlabel("λ")
        ax_tune.set_ylabel("Score  (λ / MSE(sample variance))")
        ax_tune.set_xlim(0, 1)
        ax_tune.set_title(
            f"Recursive λ Search  (N={N}, {TUNE_EVAL_GENS} eval gens/candidate)\n"
            f"Best λ = {best_lam:.4f}"
        )
        ax_tune.legend(fontsize=8)
        fig_tune.tight_layout()
        p = out_path("tune_lambda", LAMBDA_VALUES, N, ACCUMULATE, DIM)
        fig_tune.savefig(p, dpi=150)
        print(f"  -> Saved {p}")

    plt.show()
