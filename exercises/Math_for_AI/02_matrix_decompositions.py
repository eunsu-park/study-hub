"""
Exercises for Lesson 02: Matrix Decompositions
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy.linalg import expm


# === Exercise 1: Eigendecomposition ===
# Problem: Calculate eigenvalues and eigenvectors of A = [[5,2],[2,2]].
# Determine if positive definite. Compute A^10 using eigendecomposition.

def exercise_1():
    """Eigendecomposition, positive definiteness, and matrix powers."""
    A = np.array([[5, 2],
                  [2, 2]])
    print(f"Matrix A:\n{A}")

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")

    # Hand calculation verification:
    # det(A - lambda*I) = 0
    # (5-l)(2-l) - 4 = 0
    # l^2 - 7l + 6 = 0
    # (l-6)(l-1) = 0 => l = 1, 6
    print(f"\nHand calculation: eigenvalues should be 1 and 6")
    print(f"Computed: {sorted(eigenvalues)}")

    # Check positive definiteness
    is_pd = np.all(eigenvalues > 0)
    print(f"\nAll eigenvalues positive: {is_pd}")
    print(f"Matrix is positive definite: {is_pd}")

    # Compute A^10 using eigendecomposition: A^10 = V * Lambda^10 * V^T
    V = eigenvectors
    Lambda_10 = np.diag(eigenvalues ** 10)
    A_10_eigen = V @ Lambda_10 @ V.T
    A_10_direct = np.linalg.matrix_power(A, 10)

    print(f"\nA^10 (eigendecomposition):\n{A_10_eigen}")
    print(f"A^10 (direct):\n{A_10_direct}")
    print(f"Results match: {np.allclose(A_10_eigen, A_10_direct)}")


# === Exercise 2: SVD and Low-Rank Approximation ===
# Problem: Generate a 4x3 matrix, compute SVD. Find rank 1 and rank 2
# approximations. Calculate Frobenius norm error for each.

def exercise_2():
    """SVD computation and low-rank approximation with error analysis."""
    np.random.seed(42)
    A = np.random.randn(4, 3)
    print(f"Original matrix A (4x3):\n{np.round(A, 4)}")
    print(f"Matrix rank: {np.linalg.matrix_rank(A)}")

    # Full SVD
    U, sigma, VT = np.linalg.svd(A, full_matrices=False)
    print(f"\nSingular values: {sigma}")

    # Rank-1 approximation
    k = 1
    A_rank1 = U[:, :k] @ np.diag(sigma[:k]) @ VT[:k, :]
    error_rank1 = np.linalg.norm(A - A_rank1, 'fro')
    # Theoretical error: sqrt(sum of squared remaining singular values)
    theoretical_error_1 = np.sqrt(np.sum(sigma[k:] ** 2))
    print(f"\nRank-1 approximation:")
    print(f"  Frobenius norm error: {error_rank1:.6f}")
    print(f"  Theoretical error:    {theoretical_error_1:.6f}")
    print(f"  Match: {np.isclose(error_rank1, theoretical_error_1)}")

    # Rank-2 approximation
    k = 2
    A_rank2 = U[:, :k] @ np.diag(sigma[:k]) @ VT[:k, :]
    error_rank2 = np.linalg.norm(A - A_rank2, 'fro')
    theoretical_error_2 = np.sqrt(np.sum(sigma[k:] ** 2))
    print(f"\nRank-2 approximation:")
    print(f"  Frobenius norm error: {error_rank2:.6f}")
    print(f"  Theoretical error:    {theoretical_error_2:.6f}")
    print(f"  Match: {np.isclose(error_rank2, theoretical_error_2)}")

    # Summary
    original_norm = np.linalg.norm(A, 'fro')
    print(f"\nSummary:")
    print(f"  Original Frobenius norm: {original_norm:.6f}")
    print(f"  Rank-1 relative error:   {error_rank1/original_norm:.4%}")
    print(f"  Rank-2 relative error:   {error_rank2/original_norm:.4%}")


# === Exercise 3: PCA Implementation ===
# Problem: For the Iris dataset:
# 1. PCA using eigendecomposition of covariance matrix
# 2. PCA using SVD
# 3. Verify both give identical results
# 4. Minimum components for 95% cumulative explained variance

def exercise_3():
    """PCA via eigendecomposition and SVD on Iris dataset."""
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data  # (150, 4)
    print(f"Iris data shape: {X.shape}")

    # Center the data
    X_centered = X - X.mean(axis=0)

    # --- Method 1: Eigendecomposition of covariance matrix ---
    cov_matrix = np.cov(X_centered.T)
    eigenvalues_cov, eigenvectors_cov = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    idx = eigenvalues_cov.argsort()[::-1]
    eigenvalues_cov = eigenvalues_cov[idx]
    eigenvectors_cov = eigenvectors_cov[:, idx]

    X_pca_eigen = X_centered @ eigenvectors_cov[:, :2]
    print("\nMethod 1: Eigendecomposition of covariance matrix")
    print(f"  Eigenvalues (variances): {eigenvalues_cov}")

    # --- Method 2: SVD ---
    U, sigma, VT = np.linalg.svd(X_centered, full_matrices=False)
    explained_variance_svd = (sigma ** 2) / (len(X) - 1)

    X_pca_svd = X_centered @ VT.T[:, :2]
    print("\nMethod 2: SVD")
    print(f"  Explained variance: {explained_variance_svd}")

    # --- Verify both methods give same results ---
    # Note: eigenvectors may differ by sign
    match = np.allclose(np.abs(X_pca_eigen), np.abs(X_pca_svd))
    print(f"\nBoth methods give identical results (up to sign): {match}")

    # --- Minimum components for 95% explained variance ---
    total_variance = eigenvalues_cov.sum()
    cumulative_ratio = np.cumsum(eigenvalues_cov) / total_variance
    n_components_95 = np.argmax(cumulative_ratio >= 0.95) + 1

    print(f"\nExplained variance ratios: {eigenvalues_cov / total_variance}")
    print(f"Cumulative ratios: {cumulative_ratio}")
    print(f"Minimum components for 95% variance: {n_components_95}")


# === Exercise 4: Cholesky Decomposition and Sampling ===
# Problem: Generate 1000 samples from 2D Gaussian with
# Sigma = [[4,2],[2,3]], mu = [0,0]^T using Cholesky decomposition.

def exercise_4():
    """Cholesky-based multivariate Gaussian sampling."""
    mu = np.array([0, 0])
    Sigma = np.array([[4, 2],
                      [2, 3]])

    # Check positive definiteness
    eigenvalues = np.linalg.eigvalsh(Sigma)
    print(f"Covariance matrix:\n{Sigma}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Positive definite: {np.all(eigenvalues > 0)}")

    # Cholesky decomposition: Sigma = L @ L^T
    L = np.linalg.cholesky(Sigma)
    print(f"\nCholesky factor L:\n{L}")
    print(f"L @ L^T:\n{L @ L.T}")
    print(f"Reconstruction correct: {np.allclose(Sigma, L @ L.T)}")

    # Generate samples: x = mu + L @ z, z ~ N(0, I)
    np.random.seed(42)
    n_samples = 1000
    Z = np.random.randn(n_samples, 2)
    samples = mu + (L @ Z.T).T

    # Verify empirical statistics
    empirical_mean = samples.mean(axis=0)
    empirical_cov = np.cov(samples.T)

    print(f"\n=== Verification (n={n_samples}) ===")
    print(f"True mean: {mu}")
    print(f"Empirical mean: {np.round(empirical_mean, 4)}")
    print(f"\nTrue covariance:\n{Sigma}")
    print(f"Empirical covariance:\n{np.round(empirical_cov, 4)}")
    print(f"\nCovariance close: {np.allclose(Sigma, empirical_cov, atol=0.3)}")


# === Exercise 5: Recommender System ===
# Problem: Predict missing values in user-movie rating matrix using SVD.
# R = [[5,?,4,?],[?,3,?,2],[4,?,5,?],[?,2,?,3]]
# Use rank 2 approximation and verify ratings in 1-5 range.

def exercise_5():
    """SVD-based recommender system for missing rating prediction."""
    # Rating matrix (0 = missing)
    R = np.array([
        [5, 0, 4, 0],
        [0, 3, 0, 2],
        [4, 0, 5, 0],
        [0, 2, 0, 3]
    ], dtype=float)
    print("Original rating matrix (0 = missing):")
    print(R)

    # Fill missing values with row means for initial SVD
    R_filled = R.copy()
    for i in range(R.shape[0]):
        row = R[i]
        known = row[row > 0]
        if len(known) > 0:
            R_filled[i][row == 0] = known.mean()

    print(f"\nMean-filled matrix:\n{np.round(R_filled, 2)}")

    # SVD with rank-2 approximation
    U, sigma, VT = np.linalg.svd(R_filled, full_matrices=False)
    k = 2
    R_approx = U[:, :k] @ np.diag(sigma[:k]) @ VT[:k, :]

    # Clip to valid range [1, 5]
    R_clipped = np.clip(R_approx, 1, 5)

    print(f"\nRank-2 approximation:\n{np.round(R_approx, 2)}")
    print(f"\nClipped to [1,5]:\n{np.round(R_clipped, 2)}")

    # Show predicted values for missing entries
    print("\nPredicted missing ratings:")
    mask = R == 0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if mask[i, j]:
                print(f"  User {i}, Movie {j}: {R_clipped[i, j]:.2f}")

    # Verify all predictions in [1, 5]
    all_in_range = np.all((R_clipped >= 1) & (R_clipped <= 5))
    print(f"\nAll predicted ratings in [1, 5]: {all_in_range}")


if __name__ == "__main__":
    print("=== Exercise 1: Eigendecomposition ===")
    exercise_1()
    print("\n=== Exercise 2: SVD and Low-Rank Approximation ===")
    exercise_2()
    print("\n=== Exercise 3: PCA Implementation ===")
    exercise_3()
    print("\n=== Exercise 4: Cholesky Decomposition and Sampling ===")
    exercise_4()
    print("\n=== Exercise 5: Recommender System ===")
    exercise_5()
    print("\nAll exercises completed!")
