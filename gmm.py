#!/opt/miniconda3/envs/edm/bin/python
"""
Gaussian Mixture Model (GMM) Implementation
Using Expectation-Maximization (EM) algorithm
"""

import numpy as np
import pickle
from scipy.stats import multivariate_normal
from typing import Optional, Tuple, Literal


class GaussianMixtureModel:
    """
    Gaussian Mixture Model for clustering and density estimation.

    Fits a mixture of K Gaussian distributions to data using EM algorithm.

    Parameters:
    -----------
    n_components : int
        Number of Gaussian components in the mixture
    max_iter : int, default=100
        Maximum number of EM iterations
    tol : float, default=1e-4
        Convergence threshold for log-likelihood change
    init_method : {'random', 'kmeans'}, default='random'
        Method for initializing parameters
    random_state : int or None, default=None
        Random seed for reproducibility
    reg_covar : float, default=1e-6
        Regularization added to diagonal of covariance matrices
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        init_method: Literal['random', 'kmeans'] = 'random',
        random_state: Optional[int] = None,
        reg_covar: float = 1e-6
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.random_state = random_state
        self.reg_covar = reg_covar

        # Model parameters (fitted during training)
        self.weights_ = None      # Mixture weights (K,)
        self.means_ = None         # Component means (K, D)
        self.covariances_ = None   # Component covariances (K, D, D)
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_history_ = []

        if random_state is not None:
            np.random.seed(random_state)

    def _initialize_parameters(self, X: np.ndarray) -> None:
        """Initialize GMM parameters."""
        n_samples, n_features = X.shape

        if self.init_method == 'random':
            # Random initialization
            indices = np.random.choice(n_samples, self.n_components, replace=False)
            self.means_ = X[indices]

            # Initialize covariances as identity matrices scaled by data variance
            data_var = np.var(X, axis=0).mean()
            self.covariances_ = np.array([
                np.eye(n_features) * data_var for _ in range(self.n_components)
            ])

        elif self.init_method == 'kmeans':
            # K-means++ initialization
            self.means_ = self._kmeans_plusplus_init(X)

            # Compute initial covariances based on k-means assignments
            labels = self._assign_clusters(X)
            self.covariances_ = np.zeros((self.n_components, n_features, n_features))
            for k in range(self.n_components):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    self.covariances_[k] = np.cov(cluster_points.T) + self.reg_covar * np.eye(n_features)
                else:
                    self.covariances_[k] = np.eye(n_features)

        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components

    def _kmeans_plusplus_init(self, X: np.ndarray) -> np.ndarray:
        """K-means++ initialization for better starting centroids."""
        n_samples, n_features = X.shape
        centers = np.zeros((self.n_components, n_features))

        # Choose first center randomly
        centers[0] = X[np.random.randint(n_samples)]

        # Choose remaining centers
        for k in range(1, self.n_components):
            distances = np.min([np.sum((X - c)**2, axis=1) for c in centers[:k]], axis=0)
            probabilities = distances / distances.sum()
            centers[k] = X[np.random.choice(n_samples, p=probabilities)]

        return centers

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to nearest mean (for initialization)."""
        distances = np.array([np.sum((X - mean)**2, axis=1) for mean in self.means_])
        return np.argmin(distances, axis=0)

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E-step: Compute responsibilities (posterior probabilities).

        Returns:
        --------
        responsibilities : ndarray of shape (n_samples, n_components)
            Probability that sample i belongs to component k
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        # Compute weighted likelihood for each component
        for k in range(self.n_components):
            try:
                rv = multivariate_normal(
                    mean=self.means_[k],
                    cov=self.covariances_[k],
                    allow_singular=True
                )
                responsibilities[:, k] = self.weights_[k] * rv.pdf(X)
            except np.linalg.LinAlgError:
                # Handle singular covariance
                responsibilities[:, k] = 0

        # Normalize to get probabilities
        total = responsibilities.sum(axis=1, keepdims=True)
        total[total == 0] = 1  # Avoid division by zero
        responsibilities /= total

        return responsibilities

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """
        M-step: Update parameters based on responsibilities.
        """
        n_samples, n_features = X.shape

        # Effective number of points assigned to each component
        n_k = responsibilities.sum(axis=0)

        # Update weights
        self.weights_ = n_k / n_samples

        # Update means
        self.means_ = (responsibilities.T @ X) / n_k[:, np.newaxis]

        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            self.covariances_[k] = (diff.T @ weighted_diff) / n_k[k]

            # Add regularization to prevent singular matrices
            self.covariances_[k] += self.reg_covar * np.eye(n_features)

    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """Compute log-likelihood of data under current model."""
        n_samples = X.shape[0]
        log_likelihood = 0

        for i in range(n_samples):
            sample_likelihood = 0
            for k in range(self.n_components):
                try:
                    rv = multivariate_normal(
                        mean=self.means_[k],
                        cov=self.covariances_[k],
                        allow_singular=True
                    )
                    sample_likelihood += self.weights_[k] * rv.pdf(X[i])
                except np.linalg.LinAlgError:
                    continue

            if sample_likelihood > 0:
                log_likelihood += np.log(sample_likelihood)

        return log_likelihood

    def fit(self, X: np.ndarray) -> 'GaussianMixtureModel':
        """
        Fit GMM to data using EM algorithm.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data

        Returns:
        --------
        self : GaussianMixtureModel
            Fitted model
        """
        # Initialize parameters
        self._initialize_parameters(X)

        # EM iterations
        prev_log_likelihood = -np.inf
        self.log_likelihood_history_ = []

        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)

            # M-step
            self._m_step(X, responsibilities)

            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_history_.append(log_likelihood)

            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration + 1
                break

            prev_log_likelihood = log_likelihood

        if not self.converged_:
            self.n_iter_ = self.max_iter
            print(f"Warning: EM did not converge after {self.max_iter} iterations")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict component labels for data.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Data to predict

        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Component labels (0 to n_components-1)
        """
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict posterior probabilities for each component.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Data to predict

        Returns:
        --------
        probabilities : ndarray of shape (n_samples, n_components)
            Probability of each sample belonging to each component
        """
        return self._e_step(X)

    def sample(self, n_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from the fitted GMM.

        Parameters:
        -----------
        n_samples : int, default=1
            Number of samples to generate

        Returns:
        --------
        samples : ndarray of shape (n_samples, n_features)
            Generated samples
        labels : ndarray of shape (n_samples,)
            Component labels for each sample
        """
        if self.means_ is None:
            raise ValueError("Model must be fitted before sampling")

        # Sample component labels according to weights
        labels = np.random.choice(
            self.n_components,
            size=n_samples,
            p=self.weights_
        )

        # Sample from each component
        n_features = self.means_.shape[1]
        samples = np.zeros((n_samples, n_features))

        for k in range(self.n_components):
            mask = labels == k
            n_k = mask.sum()
            if n_k > 0:
                samples[mask] = np.random.multivariate_normal(
                    self.means_[k],
                    self.covariances_[k],
                    size=n_k
                )

        return samples, labels

    def score(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of data under the model.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Data to score

        Returns:
        --------
        log_likelihood : float
            Log-likelihood of data
        """
        return self._compute_log_likelihood(X)

    def bic(self, X: np.ndarray) -> float:
        """
        Compute Bayesian Information Criterion (BIC).

        Lower BIC is better.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Data

        Returns:
        --------
        bic : float
            BIC score
        """
        n_samples, n_features = X.shape
        n_params = (self.n_components - 1) + \
                   self.n_components * n_features + \
                   self.n_components * n_features * (n_features + 1) // 2

        log_likelihood = self.score(X)
        return -2 * log_likelihood + n_params * np.log(n_samples)

    def aic(self, X: np.ndarray) -> float:
        """
        Compute Akaike Information Criterion (AIC).

        Lower AIC is better.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Data

        Returns:
        --------
        aic : float
            AIC score
        """
        n_features = X.shape[1]
        n_params = (self.n_components - 1) + \
                   self.n_components * n_features + \
                   self.n_components * n_features * (n_features + 1) // 2

        log_likelihood = self.score(X)
        return -2 * log_likelihood + 2 * n_params

    def save(self, filepath: str) -> None:
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'GaussianMixtureModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# Example usage
if __name__ == "__main__":
    # Generate synthetic data from a mixture of 3 Gaussians
    np.random.seed(42)

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
    X = np.vstack([X1, X2, X3])
    true_labels = np.hstack([np.zeros(n1), np.ones(n2), np.ones(n3) * 2])

    print("=" * 70)
    print("Gaussian Mixture Model Example")
    print("=" * 70)
    print(f"\nData shape: {X.shape}")
    print(f"True number of components: 3")

    # Fit GMM
    print("\nFitting GMM with 3 components...")
    gmm = GaussianMixtureModel(
        n_components=3,
        max_iter=100,
        tol=1e-4,
        init_method='kmeans',
        random_state=42
    )
    gmm.fit(X)

    print(f"Converged: {gmm.converged_}")
    print(f"Number of iterations: {gmm.n_iter_}")
    print(f"Final log-likelihood: {gmm.log_likelihood_history_[-1]:.2f}")

    # Model selection
    print(f"\nModel Selection Metrics:")
    print(f"BIC: {gmm.bic(X):.2f}")
    print(f"AIC: {gmm.aic(X):.2f}")

    # Predictions
    labels = gmm.predict(X)
    print(f"\nPredicted component distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for comp, count in zip(unique, counts):
        print(f"  Component {comp}: {count} samples ({count/len(X)*100:.1f}%)")

    # Component parameters
    print(f"\nLearned component parameters:")
    for k in range(gmm.n_components):
        print(f"\nComponent {k}:")
        print(f"  Weight: {gmm.weights_[k]:.3f}")
        print(f"  Mean: {gmm.means_[k]}")
        print(f"  Covariance:\n{gmm.covariances_[k]}")

    # Sample from fitted model
    print(f"\nGenerating 5 samples from fitted GMM:")
    samples, sample_labels = gmm.sample(5)
    for i, (sample, label) in enumerate(zip(samples, sample_labels)):
        print(f"  Sample {i+1}: {sample} (component {label})")

    # Save model
    print("\n" + "=" * 70)
    print("Model can be saved/loaded:")
    print("  gmm.save('gmm_model.pkl')")
    print("  gmm_loaded = GaussianMixtureModel.load('gmm_model.pkl')")
    print("=" * 70)
