#!/opt/miniconda3/envs/edm/bin/python
"""
GMM-based model collapse experiment using scikit-learn's GaussianMixture.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # --- Configuration ---
    N = 1000                       # Total number of data points
    TRUE_MEAN = np.array([5.0])     # 1-D Gaussian
    TRUE_VAR = np.array([[2.0]])    # Variance of real data
    N_GENERATIONS = 5000
    LAMBDA_VALUES = [0.1, 0.9, 1]

    # Generate original real data (fixed throughout all experiments)
    real_data = np.random.multivariate_normal(TRUE_MEAN, TRUE_VAR, N)

    # Store variance traces and final generation data for combined plot
    all_vars = {}
    final_data = {}

    for lam in LAMBDA_VALUES:
        n_real = int(N * (1 - lam))     # Number of real points kept every generation
        n_synthetic = N - n_real         # Number of synthetic points per generation

        # Always keep the same subset of real data
        real_subset = real_data[:n_real].copy()

        vars_over_gen = []

        # Current dataset starts as all real data
        current_data = real_data.copy()

        for gen in range(N_GENERATIONS):
            vars_over_gen.append(current_data.var())

            # Fit a 1-component GMM to the current dataset
            gmm = GaussianMixture(
                n_components=1,
                max_iter=200,
                tol=1e-6,
                covariance_type='full',
            )
            gmm.fit(current_data)

            # Generate synthetic data from the fitted GMM
            synthetic_data, _ = gmm.sample(n_synthetic)

            # Build next generation: original real subset + new synthetic data
            current_data = np.vstack([real_subset, synthetic_data])

            if gen % 10 == 0 or gen == N_GENERATIONS - 1:
                print(f"lambda={lam:.1f}  gen={gen:3d}  var={vars_over_gen[-1]:.4f}")

        all_vars[lam] = vars_over_gen
        final_data[lam] = current_data.copy()

    # --- Figure with variance plot + distribution comparisons ---
    n_lambdas = len(LAMBDA_VALUES)
    fig, axes = plt.subplots(1, 1 + n_lambdas, figsize=(6 + 4 * n_lambdas, 5))

    # Left panel: variance over generations
    ax_var = axes[0]
    generations = list(range(N_GENERATIONS))
    for i, lam in enumerate(LAMBDA_VALUES):
        ax_var.plot(generations, all_vars[lam], label=f"λ = {lam}", zorder=len(LAMBDA_VALUES) - i)
    ax_var.axhline(TRUE_VAR[0, 0], color="black", linestyle="--", alpha=0.5, label="True variance")
    ax_var.set_xlabel("Generation")
    ax_var.set_ylabel("Variance")
    ax_var.set_ylim(0, 4)
    ax_var.set_title("Variance over Generations")
    ax_var.legend()

    # Right panels: distribution comparison for each lambda
    x_range = np.linspace(-2, 12, 300)
    for i, lam in enumerate(LAMBDA_VALUES):
        ax = axes[1 + i]
        ax.hist(real_data.flatten(), bins=50, density=True, alpha=0.5, color="blue", label="Real data")
        ax.hist(final_data[lam].flatten(), bins=50, density=True, alpha=0.5, color="red", label=f"Gen {N_GENERATIONS}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(f"λ = {lam}")
        ax.legend()

    fig.tight_layout()
    fig.savefig("gmm_variance_comparison.png", dpi=150)
    print(f"\n  -> Saved gmm_variance_comparison.png")

    plt.show()
