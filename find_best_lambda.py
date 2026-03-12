#!/opt/miniconda3/envs/edm/bin/python
"""
Iterative lambda search for GMM model collapse.

Each iteration:
  1. Evaluate N_CANDIDATES lambda values (initially spread across [0, 1]).
  2. Score each: lambda / MSE(sample_variance, initial_var).
  3. Pick the best lambda.
  4. Place N_CANDIDATES new candidates in a shrinking window around the best.
  5. Repeat for N_ITERATIONS iterations.

Shares the same GMM setup as gmm.py:
  N=1000, DIM=1, ACCUMULATE=False, TRUE_MEAN=[5], TRUE_VAR=[[2]]
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N              = 1000       # real data points
EVAL_GENS      = 5000       # generations to run per candidate evaluation
N_CANDIDATES   = 4          # lambdas to test per iteration
N_ITERATIONS   = 5          # how many times to zoom in
WINDOW_SHRINK  = 0.5        # each iteration the search window multiplies by this
N_SEEDS        = 3          # independent runs per candidate; MSE is averaged to reduce noise

TRUE_MEAN = np.array([5.0])
TRUE_VAR  = np.array([[2.0]])

# ---------------------------------------------------------------------------
# Generate real data (fixed seed for reproducibility)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
real_data = rng.multivariate_normal(TRUE_MEAN, TRUE_VAR, N)
initial_var = float(real_data.var())

print(f"Initial sample variance: {initial_var:.4f}  (true: {TRUE_VAR[0,0]})")

# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def _run_one_seed(lam: float, seed: int) -> float:
    """Single seeded run; returns MSE of sample variance over EVAL_GENS generations."""
    rng_s = np.random.default_rng(seed)
    data = real_data.copy()
    n_syn = int(N * lam)
    real_sub = real_data[: N - n_syn].copy() if n_syn > 0 else None

    vars_trace = []
    for _ in range(EVAL_GENS):
        vars_trace.append(float(data.var()))
        g = GaussianMixture(n_components=1, max_iter=200, tol=1e-6, covariance_type="full",
                            random_state=rng_s.integers(1 << 31).item())
        g.fit(data)
        if n_syn == 0:
            data = real_data.copy()
        else:
            syn, _ = g.sample(n_syn)
            data = np.vstack([real_sub, syn])

    return float(np.mean((np.array(vars_trace) - initial_var) ** 2))


def evaluate(lam: float) -> tuple[float, float]:
    """
    Average MSE over N_SEEDS independent runs to get a stable score.
    Returns (mean_mse, score) where score = lam / (mean_mse + 1e-9).
    Higher score is better: rewards high lambda with low variance drift.
    """
    mses = [_run_one_seed(lam, seed=s) for s in range(N_SEEDS)]
    mean_mse = float(np.mean(mses))
    score = lam / (mean_mse + 1e-9)
    return mean_mse, score

# ---------------------------------------------------------------------------
# Iterative search
# ---------------------------------------------------------------------------
lo, hi = 0.0, 1.0
best_lam = None
history = []   # list of dicts per iteration

print(f"\n{'='*55}")
print(f"Iterative lambda search  ({N_ITERATIONS} iterations × {N_CANDIDATES} candidates)")
print(f"{'='*55}")

for it in range(N_ITERATIONS):
    candidates = list(np.linspace(lo, hi, N_CANDIDATES))
    results = []

    print(f"\n--- Iteration {it+1}  search range [{lo:.4f}, {hi:.4f}] ---")
    for lam in candidates:
        mse, score = evaluate(lam)
        results.append({"lam": lam, "mse": mse, "score": score})
        print(f"  λ={lam:.4f}  mse={mse:.6f}  score={score:.4f}", flush=True)

    best = max(results, key=lambda r: r["score"])
    best_lam = best["lam"]
    print(f"  -> best: λ={best_lam:.4f}  (score={best['score']:.4f})")

    history.append({"iteration": it + 1, "candidates": results, "best_lam": best_lam, "lo": lo, "hi": hi})

    # Shrink window around best lambda for next iteration
    half = (hi - lo) * WINDOW_SHRINK / 2
    lo = max(0.0, best_lam - half)
    hi = min(1.0, best_lam + half)

print(f"\n{'='*55}")
print(f"Final best lambda: {best_lam:.4f}")
print(f"{'='*55}\n")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Iterative λ Search  (N={N}, {EVAL_GENS} eval gens/candidate)\nFinal best λ = {best_lam:.4f}")

colors = plt.cm.plasma(np.linspace(0.1, 0.9, N_ITERATIONS))

# Left: score vs lambda per iteration
ax_score = axes[0]
for entry in history:
    lams   = [r["lam"]   for r in entry["candidates"]]
    scores = [r["score"] for r in entry["candidates"]]
    it_num = entry["iteration"]
    ax_score.plot(lams, scores, marker="o", color=colors[it_num - 1], label=f"Iter {it_num}")
    ax_score.axvline(entry["best_lam"], color=colors[it_num - 1], linestyle=":", alpha=0.5)
ax_score.set_xlabel("λ")
ax_score.set_ylabel("Score  (λ / MSE)")
ax_score.set_xlim(0, 1)
ax_score.legend(fontsize=8)
ax_score.set_title("Score vs λ  (each iteration)")

# Right: MSE vs lambda per iteration
ax_mse = axes[1]
for entry in history:
    lams = [r["lam"] for r in entry["candidates"]]
    mses = [r["mse"] for r in entry["candidates"]]
    it_num = entry["iteration"]
    ax_mse.plot(lams, mses, marker="o", color=colors[it_num - 1], label=f"Iter {it_num}")
    ax_mse.axvline(entry["best_lam"], color=colors[it_num - 1], linestyle=":", alpha=0.5)
ax_mse.set_xlabel("λ")
ax_mse.set_ylabel("MSE(sample variance)")
ax_mse.set_xlim(0, 1)
ax_mse.legend(fontsize=8)
ax_mse.set_title("MSE vs λ  (each iteration)")

fig.tight_layout()
fig.savefig("find_best_lambda.png", dpi=150)
print("  -> Saved find_best_lambda.png")
plt.show()
