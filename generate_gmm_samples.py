#!/opt/miniconda3/envs/edm/bin/python
"""Generate samples from GMM and save to gmm_out directory."""

import numpy as np
import os
from gmm import GaussianMixtureModel

# ============================================================================
# HYPERPARAMETERS - Edit these
# ============================================================================
NUM_SAMPLES = 1000  # Number of samples to generate
N_COMPONENTS = 3    # Number of GMM components
RANDOM_SEED = 42

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'gmm_out')

# ============================================================================

def main():
    print("=" * 70)
    print("GMM Sample Generation")
    print("=" * 70)

    # Generate synthetic training data (same as gmm.py example)
    np.random.seed(RANDOM_SEED)

    print("\nGenerating synthetic training data...")

    # Component 1
    n1 = 300
    mean1 = np.array([0, 0])
    cov1 = np.array([[1, 0.5], [0.5, 1]])
    X1 = np.random.multivariate_normal(mean1, cov1, n1)

    # Component 2
    n2 = 200
    mean2 = np.array([5, 5])
    cov2 = np.array([[1, -0.3], [-0.3, 1]])
    X2 = np.random.multivariate_normal(mean2, cov2, n2)

    # Component 3
    n3 = 250
    mean3 = np.array([5, 0])
    cov3 = np.array([[0.5, 0], [0, 0.5]])
    X3 = np.random.multivariate_normal(mean3, cov3, n3)

    # Combine data
    X_train = np.vstack([X1, X2, X3])
    print(f"Training data shape: {X_train.shape}")

    # Fit GMM
    print(f"\nFitting GMM with {N_COMPONENTS} components...")
    gmm = GaussianMixtureModel(
        n_components=N_COMPONENTS,
        max_iter=100,
        tol=1e-4,
        init_method='kmeans',
        random_state=RANDOM_SEED
    )
    gmm.fit(X_train)

    print(f"Converged: {gmm.converged_}")
    print(f"Number of iterations: {gmm.n_iter_}")
    print(f"Final log-likelihood: {gmm.log_likelihood_history_[-1]:.2f}")

    # Generate samples
    print(f"\nGenerating {NUM_SAMPLES} samples from fitted GMM...")
    samples, labels = gmm.sample(NUM_SAMPLES)

    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample component distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for comp, count in zip(unique, counts):
        print(f"  Component {comp}: {count} samples ({count/len(labels)*100:.1f}%)")

    # Save samples
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    samples_path = os.path.join(OUTPUT_DIR, 'samples.npy')
    labels_path = os.path.join(OUTPUT_DIR, 'labels.npy')
    training_data_path = os.path.join(OUTPUT_DIR, 'training_data.npy')

    np.save(samples_path, samples)
    np.save(labels_path, labels)
    np.save(training_data_path, X_train)

    print(f"\n✓ Samples saved to: {samples_path}")
    print(f"✓ Labels saved to: {labels_path}")
    print(f"✓ Training data saved to: {training_data_path}")

    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'gmm_model.pkl')
    gmm.save(model_path)
    print(f"✓ Model saved to: {model_path}")

    # Save statistics
    stats_path = os.path.join(OUTPUT_DIR, 'statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GMM Sample Generation Statistics\n")
        f.write("=" * 70 + "\n\n")

        f.write("TRAINING DATA\n")
        f.write("-" * 70 + "\n")
        f.write(f"Shape: {X_train.shape}\n")
        f.write(f"Mean: {X_train.mean(axis=0)}\n")
        f.write(f"Std: {X_train.std(axis=0)}\n\n")

        f.write("MODEL PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of components: {N_COMPONENTS}\n")
        f.write(f"Converged: {gmm.converged_}\n")
        f.write(f"Iterations: {gmm.n_iter_}\n")
        f.write(f"Final log-likelihood: {gmm.log_likelihood_history_[-1]:.4f}\n\n")

        for k in range(gmm.n_components):
            f.write(f"Component {k}:\n")
            f.write(f"  Weight: {gmm.weights_[k]:.4f}\n")
            f.write(f"  Mean: {gmm.means_[k]}\n")
            f.write(f"  Covariance:\n")
            f.write(f"    {gmm.covariances_[k]}\n\n")

        f.write("GENERATED SAMPLES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of samples: {NUM_SAMPLES}\n")
        f.write(f"Shape: {samples.shape}\n")
        f.write(f"Mean: {samples.mean(axis=0)}\n")
        f.write(f"Std: {samples.std(axis=0)}\n\n")

        f.write("Component Distribution:\n")
        for comp, count in zip(unique, counts):
            f.write(f"  Component {comp}: {count} ({count/len(labels)*100:.1f}%)\n")

    print(f"✓ Statistics saved to: {stats_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
