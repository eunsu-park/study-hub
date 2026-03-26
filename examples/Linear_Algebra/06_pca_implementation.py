"""
Principal Component Analysis

Demonstrates PCA concepts:
- PCA from scratch: centering, covariance, eigendecomposition
- PCA via SVD (more numerically stable)
- Comparison with sklearn PCA
- Explained variance ratio and scree plot
- Dimensionality reduction on synthetic and real data

Dependencies: numpy, matplotlib, sklearn
"""

import numpy as np
import matplotlib.pyplot as plt


def pca_from_scratch():
    """Implement PCA step by step using eigendecomposition."""
    print("=" * 60)
    print("PCA FROM SCRATCH (EIGENDECOMPOSITION)")
    print("=" * 60)

    # Generate 2D data with clear principal direction
    np.random.seed(42)
    n = 200
    theta = np.pi / 4  # 45 degrees
    # Data stretched along direction theta
    data = np.column_stack([
        np.random.randn(n) * 3,
        np.random.randn(n) * 0.5
    ])
    # Rotate
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    X = data @ R.T + np.array([5, 3])  # Translate

    print(f"Data shape: {X.shape}")
    print(f"Data mean: {np.round(np.mean(X, axis=0), 4)}")

    # Step 1: Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    print(f"\nStep 1: Center data (subtract mean)")
    print(f"Centered mean: {np.round(np.mean(X_centered, axis=0), 10)}")

    # Step 2: Compute covariance matrix
    cov = (X_centered.T @ X_centered) / (n - 1)
    print(f"\nStep 2: Covariance matrix")
    print(f"C = X^T X / (n-1):\n{np.round(cov, 4)}")
    print(f"np.cov verification:\n{np.round(np.cov(X.T), 4)}")

    # Step 3: Eigendecomposition of covariance
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"\nStep 3: Eigendecomposition")
    print(f"Eigenvalues (variance along PCs): {np.round(eigenvalues, 4)}")
    print(f"PC1 direction: {np.round(eigenvectors[:, 0], 4)}")
    print(f"PC2 direction: {np.round(eigenvectors[:, 1], 4)}")

    # Step 4: Project data
    X_pca = X_centered @ eigenvectors
    print(f"\nStep 4: Project data onto principal components")
    print(f"Projected data shape: {X_pca.shape}")

    # Explained variance
    total_var = np.sum(eigenvalues)
    explained = eigenvalues / total_var
    print(f"\nExplained variance ratio: {np.round(explained, 4)}")
    print(f"PC1 explains {explained[0]*100:.1f}% of variance")

    return X, X_centered, X_pca, eigenvectors, eigenvalues, mean


def pca_via_svd():
    """Implement PCA using SVD (more numerically stable)."""
    print("\n" + "=" * 60)
    print("PCA VIA SVD")
    print("=" * 60)

    np.random.seed(42)
    n, d = 100, 5
    X = np.random.randn(n, d) @ np.diag([5, 3, 1, 0.5, 0.1]) + np.random.randn(n, d) * 0.1

    # Center data
    X_centered = X - np.mean(X, axis=0)

    # SVD of centered data matrix
    U, sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Principal components are rows of V^T (columns of V)
    # Eigenvalues of covariance = sigma^2 / (n-1)
    eigenvalues_svd = sigma ** 2 / (n - 1)
    components_svd = Vt  # Principal component directions

    # Compare with eigendecomposition
    cov = np.cov(X.T)
    eigenvalues_eig, eigenvectors_eig = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues_eig)[::-1]
    eigenvalues_eig = eigenvalues_eig[idx]

    print(f"Eigenvalues via SVD:  {np.round(eigenvalues_svd, 4)}")
    print(f"Eigenvalues via eigh: {np.round(eigenvalues_eig, 4)}")
    print(f"Match: {np.allclose(eigenvalues_svd, eigenvalues_eig, atol=1e-10)}")

    # Explained variance
    explained = eigenvalues_svd / np.sum(eigenvalues_svd)
    cumulative = np.cumsum(explained)
    print(f"\nExplained variance ratio: {np.round(explained, 4)}")
    print(f"Cumulative: {np.round(cumulative, 4)}")

    # Number of components for 95% variance
    k_95 = np.searchsorted(cumulative, 0.95) + 1
    print(f"\nComponents for 95% variance: {k_95} out of {d}")

    return eigenvalues_svd


def compare_with_sklearn():
    """Compare custom PCA with sklearn."""
    print("\n" + "=" * 60)
    print("COMPARISON WITH SKLEARN PCA")
    print("=" * 60)

    try:
        from sklearn.decomposition import PCA
        from sklearn.datasets import load_iris

        # Load Iris dataset
        iris = load_iris()
        X = iris.data  # (150, 4)
        y = iris.target

        print(f"Iris dataset: {X.shape}")

        # sklearn PCA
        pca = PCA(n_components=2)
        X_sklearn = pca.fit_transform(X)
        print(f"\nsklearn PCA:")
        print(f"Explained variance ratio: {np.round(pca.explained_variance_ratio_, 4)}")
        print(f"Components:\n{np.round(pca.components_, 4)}")

        # Manual PCA
        X_centered = X - np.mean(X, axis=0)
        U, sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)
        eigenvalues = sigma ** 2 / (X.shape[0] - 1)
        explained = eigenvalues / np.sum(eigenvalues)

        print(f"\nManual PCA via SVD:")
        print(f"Explained variance ratio: {np.round(explained[:2], 4)}")
        print(f"Components:\n{np.round(Vt[:2], 4)}")

        # Projections match (up to sign)
        X_manual = X_centered @ Vt[:2].T
        # Fix sign ambiguity
        for i in range(2):
            if np.dot(X_sklearn[:, i], X_manual[:, i]) < 0:
                X_manual[:, i] *= -1

        print(f"\nProjections match (up to sign): {np.allclose(np.abs(X_sklearn), np.abs(X_manual))}")

    except ImportError:
        print("sklearn not available, skipping comparison")


def visualize_pca(X, X_centered, X_pca, eigenvectors, eigenvalues, mean):
    """Visualize PCA results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Original data with principal components
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=10, color='blue')
    scale = 3
    for i in range(2):
        ev = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * scale
        ax.annotate('', xy=mean + ev, xytext=mean,
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.text(mean[0] + ev[0] * 1.1, mean[1] + ev[1] * 1.1,
                f'PC{i+1}', fontsize=10, color='red', fontweight='bold')
    ax.scatter(*mean, color='red', s=100, marker='+', linewidths=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Original Data with PCs')

    # Plot 2: Projected data
    ax = axes[1]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=10, color='green')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Data in PC Space')

    # Plot 3: Scree plot
    ax = axes[2]
    explained = eigenvalues / np.sum(eigenvalues)
    cumulative = np.cumsum(explained)
    components = range(1, len(eigenvalues) + 1)
    ax.bar(components, explained, alpha=0.7, label='Individual')
    ax.plot(components, cumulative, 'ro-', label='Cumulative')
    ax.axhline(y=0.95, color='k', linestyle='--', alpha=0.3, label='95% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_xticks(list(components))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Scree Plot')

    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: pca_visualization.png")


def dimensionality_reduction():
    """Demonstrate dimensionality reduction with PCA."""
    print("\n" + "=" * 60)
    print("DIMENSIONALITY REDUCTION")
    print("=" * 60)

    # High-dimensional data with intrinsic low dimension
    np.random.seed(42)
    n = 300
    d_intrinsic = 3
    d_ambient = 20

    # Generate intrinsic coordinates
    t = np.column_stack([np.random.randn(n) for _ in range(d_intrinsic)])

    # Embed in higher dimensions via random projection
    W = np.random.randn(d_intrinsic, d_ambient)
    X = t @ W + np.random.randn(n, d_ambient) * 0.1

    print(f"Ambient dimension: {d_ambient}")
    print(f"Intrinsic dimension: {d_intrinsic}")
    print(f"Samples: {n}")

    # Compute all singular values
    X_centered = X - np.mean(X, axis=0)
    sigma = np.linalg.svd(X_centered, compute_uv=False)
    eigenvalues = sigma ** 2 / (n - 1)
    explained = eigenvalues / np.sum(eigenvalues)
    cumulative = np.cumsum(explained)

    print(f"\nSingular values: {np.round(sigma[:8], 4)} ...")
    print(f"Explained variance (first 6): {np.round(explained[:6], 4)}")
    print(f"Cumulative (first 6): {np.round(cumulative[:6], 4)}")

    # Clear gap between intrinsic dimensions and noise
    print(f"\nGap analysis:")
    print(f"  sigma[{d_intrinsic-1}] / sigma[{d_intrinsic}] = "
          f"{sigma[d_intrinsic-1] / sigma[d_intrinsic]:.2f} (large gap = low intrinsic dim)")

    # Reconstruction error
    for k in [1, 2, 3, 5, 10, 20]:
        X_k = X_centered @ np.linalg.svd(X_centered, full_matrices=False)[2][:k].T
        X_recon = X_k @ np.linalg.svd(X_centered, full_matrices=False)[2][:k]
        error = np.linalg.norm(X_centered - X_recon, 'fro') / np.linalg.norm(X_centered, 'fro')
        print(f"  k={k:2d}: relative reconstruction error = {error:.6f}")


if __name__ == "__main__":
    X, X_centered, X_pca, eigvecs, eigvals, mean = pca_from_scratch()
    pca_via_svd()
    compare_with_sklearn()
    visualize_pca(X, X_centered, X_pca, eigvecs, eigvals, mean)
    dimensionality_reduction()
    print("\nAll examples completed!")
