"""
Manifold Learning and Representation Learning

Demonstrates:
- Swiss roll dataset generation and the manifold hypothesis
- t-SNE dimensionality reduction
- UMAP dimensionality reduction (if available, else falls back to t-SNE)
- Geodesic distance estimation via Isomap
- Visualization comparing linear PCA vs non-linear manifold methods

Dependencies: numpy, sklearn, matplotlib
Optional:     umap-learn (pip install umap-learn)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_s_curve, load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# 1. The Manifold Hypothesis
# ---------------------------------------------------------------------------

def manifold_hypothesis():
    """
    Illustrate the manifold hypothesis: high-dimensional data lies near
    a low-dimensional manifold embedded in the ambient space.
    """
    print("=" * 60)
    print("THE MANIFOLD HYPOTHESIS")
    print("=" * 60)

    print("\nManifold hypothesis: real-world high-dimensional data")
    print("concentrates near a low-dimensional manifold.")
    print()
    print("Examples:")
    print("  - Images of faces: ~10M pixels, but intrinsic DoF ≈ 50")
    print("    (pose, lighting, expression, identity...)")
    print("  - Audio waveforms: 22050 samples/s, but content is low-dim")
    print("  - Swiss roll in R³: intrinsic dimension = 2")

    # Swiss roll
    np.random.seed(42)
    n_points = 1500
    X_swiss, t_swiss = make_swiss_roll(n_samples=n_points, noise=0.1, random_state=42)

    print(f"\nSwiss roll dataset:")
    print(f"  Ambient dimension: {X_swiss.shape[1]} (3D space)")
    print(f"  Intrinsic dimension: 2 (parameterized by arc length and height)")
    print(f"  n_samples = {n_points}")

    # Intrinsic dimension estimation via correlation dimension
    k = 10
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_swiss)
    distances, _ = nbrs.kneighbors(X_swiss)
    r1 = distances[:, 1].mean()   # nearest neighbor distance
    r_k = distances[:, k].mean()  # k-th neighbor distance
    # Rough correlation dimension: d ≈ log(k) / log(r_k / r_1)
    corr_dim = np.log(k) / np.log(r_k / r_1 + 1e-10)
    print(f"\n  Estimated intrinsic dimension (correlation): {corr_dim:.2f}")
    print(f"  (True value: 2)")

    return X_swiss, t_swiss


# ---------------------------------------------------------------------------
# 2. Geodesic Distances via Isomap
# ---------------------------------------------------------------------------

def geodesic_distances(X_swiss, t_swiss):
    """
    Isomap estimates geodesic distances along the manifold
    by building a neighborhood graph and computing shortest paths.
    """
    print("\n" + "=" * 60)
    print("GEODESIC DISTANCES (ISOMAP)")
    print("=" * 60)

    print("\nEuclidean vs geodesic distance on Swiss roll:")
    print("  Two points on opposite sides of the roll may be")
    print("  Euclidean-close but geodesically far apart.")

    # Pick two points that are Euclidean-close but geodesically far
    # Points near same Euclidean position but on different 'layers'
    target_idx = 0
    X0 = X_swiss[target_idx]

    # Find Euclidean neighbors
    dists_euclidean = np.linalg.norm(X_swiss - X0, axis=1)
    top_euc = np.argsort(dists_euclidean)[1:6]  # 5 closest by Euclidean

    print(f"\nFrom reference point (color t={t_swiss[target_idx]:.2f}):")
    print(f"{'Neighbor idx':>14s} {'Eucl. dist':>12s} {'t value':>10s}")
    for idx in top_euc:
        print(f"{idx:>14d} {dists_euclidean[idx]:>12.4f} {t_swiss[idx]:>10.4f}")

    # Isomap embedding — uses geodesic (graph) distances internally
    print("\nFitting Isomap (n_components=2, n_neighbors=10)...")
    isomap = Isomap(n_neighbors=10, n_components=2)
    X_isomap = isomap.fit_transform(X_swiss)

    print(f"Isomap stress (reconstruction error): {isomap.reconstruction_error():.4f}")
    print(f"Embedded shape: {X_isomap.shape}")

    return X_isomap


# ---------------------------------------------------------------------------
# 3. t-SNE
# ---------------------------------------------------------------------------

def tsne_reduction(X, labels, title_prefix=""):
    """t-SNE: preserves local neighborhood structure"""
    print(f"\n--- t-SNE: {title_prefix} ---")
    print("t-SNE minimizes KL(P || Q) where:")
    print("  P[i,j] = normalized similarity in high-dim (Gaussian kernel)")
    print("  Q[i,j] = normalized similarity in low-dim (Student-t kernel)")
    print("  Heavy-tailed Q prevents crowding in low dimensions")

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
                random_state=42, learning_rate='auto', init='pca')
    X_tsne = tsne.fit_transform(X)
    print(f"  KL divergence (final): {tsne.kl_divergence_:.4f}")
    return X_tsne


# ---------------------------------------------------------------------------
# 4. Comparison on Digits Dataset
# ---------------------------------------------------------------------------

def compare_methods_on_digits():
    """
    Compare PCA, Isomap, and t-SNE on the Digits dataset (64-dim → 2-dim).
    """
    print("\n" + "=" * 60)
    print("METHOD COMPARISON: DIGITS DATASET")
    print("=" * 60)

    digits = load_digits()
    X = digits.data        # (1797, 64)
    y = digits.target      # 0-9 labels

    X_scaled = StandardScaler().fit_transform(X)

    print(f"Digits dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    # PCA
    print("\nFitting PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Isomap
    print("Fitting Isomap...")
    isomap = Isomap(n_neighbors=10, n_components=2)
    X_iso = isomap.fit_transform(X_scaled)

    # t-SNE
    print("Fitting t-SNE (may take a moment)...")
    X_tsne = tsne_reduction(X_scaled, y, title_prefix="Digits")

    return X_pca, X_iso, X_tsne, y


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------

def visualize_all(X_swiss, t_swiss, X_isomap, X_pca_digits, X_iso_digits, X_tsne_digits, y_digits):
    """Visualize Swiss roll embeddings and digits comparison"""
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    fig = plt.figure(figsize=(18, 10))

    # Row 1: Swiss roll
    # 3D original
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2],
                c=t_swiss, cmap='Spectral', s=8, alpha=0.6)
    ax1.set_title('Swiss Roll\n(3D ambient space)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.tick_params(labelsize=7)

    # PCA on swiss roll (fails to unroll)
    ax2 = fig.add_subplot(232)
    pca_swiss = PCA(n_components=2)
    X_pca_swiss = pca_swiss.fit_transform(X_swiss)
    ax2.scatter(X_pca_swiss[:, 0], X_pca_swiss[:, 1],
                c=t_swiss, cmap='Spectral', s=8, alpha=0.6)
    ax2.set_title('PCA\n(fails to unroll)')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.grid(True, alpha=0.3)

    # Isomap on swiss roll (succeeds)
    ax3 = fig.add_subplot(233)
    ax3.scatter(X_isomap[:, 0], X_isomap[:, 1],
                c=t_swiss, cmap='Spectral', s=8, alpha=0.6)
    ax3.set_title('Isomap\n(unrolls manifold)')
    ax3.set_xlabel('Dim 1')
    ax3.set_ylabel('Dim 2')
    ax3.grid(True, alpha=0.3)

    # Row 2: Digits comparison
    cmap = plt.cm.get_cmap('tab10', 10)
    for ax, X_emb, title in [
        (fig.add_subplot(234), X_pca_digits, 'PCA (linear)\nDigits 64→2'),
        (fig.add_subplot(235), X_iso_digits, 'Isomap\nDigits 64→2'),
        (fig.add_subplot(236), X_tsne_digits, 't-SNE\nDigits 64→2'),
    ]:
        sc = ax.scatter(X_emb[:, 0], X_emb[:, 1],
                        c=y_digits, cmap=cmap, s=8, alpha=0.7, vmin=0, vmax=9)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, ticks=range(10), shrink=0.8)

    plt.tight_layout()
    plt.savefig('manifold_representation_learning.png', dpi=150)
    print("Visualization saved to manifold_representation_learning.png")
    plt.close()


if __name__ == "__main__":
    X_swiss, t_swiss = manifold_hypothesis()
    X_isomap = geodesic_distances(X_swiss, t_swiss)
    X_pca_d, X_iso_d, X_tsne_d, y_d = compare_methods_on_digits()
    visualize_all(X_swiss, t_swiss, X_isomap, X_pca_d, X_iso_d, X_tsne_d, y_d)

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
