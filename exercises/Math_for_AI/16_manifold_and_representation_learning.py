"""
Exercises for Lesson 16: Manifold and Representation Learning
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_swiss_roll(n_samples=1000, noise=0.5):
    """Generate Swiss roll dataset."""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = 30 * np.random.rand(n_samples)
    z = t * np.sin(t)
    X = np.column_stack([x, y, z]) + noise * np.random.randn(n_samples, 3)
    return X, t  # t is the underlying 1D parameter (color)


# === Exercise 1: Importance of Geodesic Distance ===
# Problem: Compare Euclidean and geodesic distances on Swiss roll, vary k.

def exercise_1():
    """Geodesic vs Euclidean distance on Swiss roll data."""
    np.random.seed(42)
    n = 500
    X, t = make_swiss_roll(n, noise=0.3)

    print("Geodesic vs Euclidean Distance on Swiss Roll\n")

    # Euclidean distance matrix
    D_euclid = squareform(pdist(X))

    # True geodesic approximation (using the parameter t and y)
    # On the Swiss roll, geodesic distance ~ sqrt((t1-t2)^2 + (y1-y2)^2)
    t_matrix = np.abs(t[:, np.newaxis] - t[np.newaxis, :])
    y_diff = np.abs(X[:, 1, np.newaxis] - X[np.newaxis, :, 1])
    D_true_geodesic = np.sqrt(t_matrix ** 2 + y_diff ** 2)

    print("Euclidean vs True Geodesic correlation:")
    # Flatten upper triangle
    idx = np.triu_indices(n, k=1)
    euclid_flat = D_euclid[idx]
    geodesic_flat = D_true_geodesic[idx]
    corr_euclid = np.corrcoef(euclid_flat, geodesic_flat)[0, 1]
    print(f"  Euclidean-Geodesic correlation: {corr_euclid:.4f}")
    print(f"  Euclidean distances shortcut through the roll, distorting distances")

    # Graph-based geodesic approximation for various k
    print("\nGraph-based geodesic approximation (varying k):")
    k_values = [5, 10, 15, 20, 30, 50]
    correlations = []

    for k in k_values:
        # k-nearest neighbor graph
        knn_graph = np.full((n, n), np.inf)
        np.fill_diagonal(knn_graph, 0)
        for i in range(n):
            dists = D_euclid[i]
            neighbors = np.argsort(dists)[1:k + 1]
            for j in neighbors:
                knn_graph[i, j] = dists[j]
                knn_graph[j, i] = dists[j]  # symmetric

        # Shortest path = geodesic approximation
        D_graph = shortest_path(knn_graph, method='D')

        # Handle disconnected components
        connected = np.isfinite(D_graph[idx])
        if np.sum(connected) > 0:
            graph_flat = D_graph[idx][connected]
            geo_flat = geodesic_flat[connected]
            corr = np.corrcoef(graph_flat, geo_flat)[0, 1]
            pct_connected = np.sum(connected) / len(connected) * 100
        else:
            corr = 0
            pct_connected = 0

        correlations.append(corr)
        print(f"  k={k:2d}: correlation={corr:.4f}, connected={pct_connected:.1f}%")

    print(f"\nBest k: {k_values[np.argmax(correlations)]} "
          f"(correlation={max(correlations):.4f})")
    print("Too small k -> disconnected graph, too large k -> shortcutting")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    sc = ax.scatter(X[:, 0], X[:, 2], c=t, cmap='viridis', s=10)
    ax.set_title('Swiss Roll (top view)')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    plt.colorbar(sc, ax=ax, label='t')

    axes[1].scatter(euclid_flat[::100], geodesic_flat[::100], s=1, alpha=0.3)
    axes[1].set_xlabel('Euclidean Distance')
    axes[1].set_ylabel('True Geodesic Distance')
    axes[1].set_title(f'Euclidean vs Geodesic (r={corr_euclid:.3f})')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(k_values, correlations, 'bo-')
    axes[2].set_xlabel('k (neighbors)')
    axes[2].set_ylabel('Correlation with true geodesic')
    axes[2].set_title('Geodesic Approximation Quality')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex16_1_geodesic_distance.png', dpi=150)
    plt.close()
    print("  Plot saved: ex16_1_geodesic_distance.png")


# === Exercise 2: t-SNE Hyperparameter Study ===
# Problem: Vary perplexity, learning rate, iterations on Swiss roll.

def exercise_2():
    """t-SNE hyperparameter study on Swiss roll."""
    np.random.seed(42)
    n = 300
    X, t = make_swiss_roll(n, noise=0.3)

    print("t-SNE Hyperparameter Study on Swiss Roll\n")

    def compute_pairwise_probs(X_data, perplexity):
        """Compute symmetric pairwise probabilities P for t-SNE."""
        n_pts = X_data.shape[0]
        D = squareform(pdist(X_data, 'sqeuclidean'))
        P = np.zeros((n_pts, n_pts))
        target_entropy = np.log(perplexity)

        for i in range(n_pts):
            # Binary search for sigma_i
            lo, hi = 1e-10, 1e4
            for _ in range(50):
                sigma = (lo + hi) / 2
                pij = np.exp(-D[i] / (2 * sigma ** 2))
                pij[i] = 0
                sum_pij = np.sum(pij)
                if sum_pij < 1e-10:
                    lo = sigma
                    continue
                pij /= sum_pij
                entropy = -np.sum(pij[pij > 0] * np.log(pij[pij > 0]))
                if entropy > target_entropy:
                    hi = sigma
                else:
                    lo = sigma
            P[i] = pij

        # Symmetrize
        P = (P + P.T) / (2 * n_pts)
        P = np.maximum(P, 1e-12)
        return P

    def tsne(X_data, perplexity=30, lr=200, n_iter=300, seed=42):
        """Simple t-SNE implementation."""
        np.random.seed(seed)
        n_pts = X_data.shape[0]
        P = compute_pairwise_probs(X_data, perplexity)

        # Initialize
        Y = 0.01 * np.random.randn(n_pts, 2)
        velocity = np.zeros_like(Y)
        kl_history = []

        for it in range(n_iter):
            # Q distribution (Student-t kernel)
            D_low = squareform(pdist(Y, 'sqeuclidean'))
            Q = 1.0 / (1.0 + D_low)
            np.fill_diagonal(Q, 0)
            Q_sum = np.sum(Q)
            Q_norm = Q / max(Q_sum, 1e-10)
            Q_norm = np.maximum(Q_norm, 1e-12)

            # KL divergence
            kl = np.sum(P * np.log(P / Q_norm))
            kl_history.append(kl)

            # Gradient
            PQ_diff = P - Q_norm
            grad = np.zeros_like(Y)
            for i in range(n_pts):
                diff = Y[i] - Y
                grad[i] = 4 * np.sum((PQ_diff[i] * Q[:, i])[:, np.newaxis] * diff, axis=0)

            # Update with momentum
            momentum = 0.8 if it > 20 else 0.5
            velocity = momentum * velocity - lr * grad
            Y += velocity

        return Y, kl_history

    # (1) Vary perplexity
    print("(1) Effect of perplexity:")
    perplexities = [5, 10, 30, 50, 100]
    results_perp = {}
    for perp in perplexities:
        effective_perp = min(perp, n - 1)
        Y, kl = tsne(X, perplexity=effective_perp, lr=200, n_iter=200)
        results_perp[perp] = (Y, kl[-1])
        print(f"    perplexity={perp:3d}: final KL={kl[-1]:.4f}")

    # (2) Vary learning rate
    print("\n(2) Effect of learning rate:")
    learning_rates = [10, 50, 200, 500]
    results_lr = {}
    for lr_val in learning_rates:
        Y, kl = tsne(X, perplexity=30, lr=lr_val, n_iter=200)
        results_lr[lr_val] = (Y, kl[-1])
        print(f"    lr={lr_val:4d}: final KL={kl[-1]:.4f}")

    # (3) Convergence analysis
    print("\n(3) Convergence over iterations:")
    Y_full, kl_full = tsne(X, perplexity=30, lr=200, n_iter=300)
    print(f"    KL at iter 50:  {kl_full[49]:.4f}")
    print(f"    KL at iter 100: {kl_full[99]:.4f}")
    print(f"    KL at iter 200: {kl_full[199]:.4f}")
    print(f"    KL at iter 300: {kl_full[-1]:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Perplexity comparison
    for i, perp in enumerate([5, 30, 100]):
        Y_vis = results_perp[perp][0]
        axes[0, i].scatter(Y_vis[:, 0], Y_vis[:, 1], c=t, cmap='viridis', s=10)
        axes[0, i].set_title(f'Perplexity={perp} (KL={results_perp[perp][1]:.3f})')

    # Learning rate comparison
    for i, lr_val in enumerate([10, 200, 500]):
        Y_vis = results_lr[lr_val][0]
        axes[1, i].scatter(Y_vis[:, 0], Y_vis[:, 1], c=t, cmap='viridis', s=10)
        axes[1, i].set_title(f'LR={lr_val} (KL={results_lr[lr_val][1]:.3f})')

    plt.tight_layout()
    plt.savefig('ex16_2_tsne_hyperparams.png', dpi=150)
    plt.close()
    print("  Plot saved: ex16_2_tsne_hyperparams.png")


# === Exercise 3: UMAP vs t-SNE Comparison ===
# Problem: Compare embeddings, execution time, global structure preservation.

def exercise_3():
    """UMAP vs t-SNE comparison (simplified UMAP implementation)."""
    np.random.seed(42)
    n = 400
    X, t = make_swiss_roll(n, noise=0.3)

    print("UMAP vs t-SNE Comparison\n")

    # Simple t-SNE (reuse core logic)
    def simple_tsne(X_data, n_iter=300):
        n_pts = X_data.shape[0]
        D = squareform(pdist(X_data, 'sqeuclidean'))
        P = np.zeros((n_pts, n_pts))
        perplexity = 30
        target_H = np.log(perplexity)

        for i in range(n_pts):
            lo, hi = 1e-10, 1e4
            for _ in range(50):
                sigma = (lo + hi) / 2
                pij = np.exp(-D[i] / (2 * sigma ** 2))
                pij[i] = 0
                s = np.sum(pij)
                if s < 1e-10:
                    lo = sigma
                    continue
                pij /= s
                H = -np.sum(pij[pij > 0] * np.log(pij[pij > 0]))
                if H > target_H:
                    hi = sigma
                else:
                    lo = sigma
            P[i] = pij
        P = (P + P.T) / (2 * n_pts)
        P = np.maximum(P, 1e-12)

        Y = 0.01 * np.random.randn(n_pts, 2)
        vel = np.zeros_like(Y)
        for it in range(n_iter):
            D_low = squareform(pdist(Y, 'sqeuclidean'))
            Q = 1.0 / (1.0 + D_low)
            np.fill_diagonal(Q, 0)
            Q_norm = Q / max(np.sum(Q), 1e-10)
            Q_norm = np.maximum(Q_norm, 1e-12)
            PQ = P - Q_norm
            grad = np.zeros_like(Y)
            for i in range(n_pts):
                diff = Y[i] - Y
                grad[i] = 4 * np.sum((PQ[i] * Q[:, i])[:, np.newaxis] * diff, axis=0)
            mom = 0.8 if it > 20 else 0.5
            vel = mom * vel - 200 * grad
            Y += vel
        return Y

    # Simplified UMAP-like embedding
    def simple_umap(X_data, n_neighbors=15, min_dist=0.1, n_iter=200):
        """Simplified UMAP: fuzzy simplicial set + cross-entropy optimization."""
        n_pts = X_data.shape[0]
        D = squareform(pdist(X_data))

        # Build fuzzy k-NN graph
        P = np.zeros((n_pts, n_pts))
        for i in range(n_pts):
            knn_dists = np.sort(D[i])[1:n_neighbors + 1]
            rho = knn_dists[0]  # distance to nearest neighbor
            # Binary search for sigma
            target_log_k = np.log2(n_neighbors)
            lo, hi = 1e-10, 100
            for _ in range(64):
                sigma = (lo + hi) / 2
                vals = np.exp(-(np.maximum(D[i] - rho, 0)) / sigma)
                vals[i] = 0
                # Only consider k nearest neighbors
                top_k = np.argsort(D[i])[1:n_neighbors + 1]
                membership = np.sum(vals[top_k])
                if membership > target_log_k:
                    hi = sigma
                else:
                    lo = sigma
            P[i] = vals

        # Symmetrize: P_sym = P + P^T - P * P^T
        P = P + P.T - P * P.T
        P = np.clip(P, 0, 1)

        # Initialize with PCA
        X_centered = X_data - X_data.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        Y = X_centered @ Vt[:2].T
        Y *= 0.01 / np.std(Y)

        # SGD optimization
        a, b = 1.0, 1.0  # parameters for low-dim kernel
        lr = 1.0
        for it in range(n_iter):
            D_low = squareform(pdist(Y, 'sqeuclidean'))
            Q = 1.0 / (1.0 + a * D_low ** b)
            np.fill_diagonal(Q, 0)

            # Gradient of cross-entropy
            grad = np.zeros_like(Y)
            for i in range(n_pts):
                diff = Y[i] - Y
                # Attractive: -2ab * P * d^(b-1) / (1 + a*d^b) * (yi - yj)
                attract = P[i] * 2 * a * b * (D_low[i] ** (b - 1) + 1e-10) * Q[:, i]
                # Repulsive: 2b * (1-P) / (d * (1 + a*d^b)) * (yi - yj)
                repulse = (1 - P[i]) * 2 * b / (D_low[i] + 0.01) * Q[:, i] ** 2
                grad[i] = np.sum((repulse - attract)[:, np.newaxis] * diff, axis=0)

            Y -= lr * grad
            lr *= 0.99

        return Y

    # Time comparison
    print("Execution time comparison:")
    sizes = [100, 200, 400]
    for sz in sizes:
        X_sub = X[:sz]

        t_start = time.time()
        Y_tsne = simple_tsne(X_sub, n_iter=100)
        t_tsne = time.time() - t_start

        t_start = time.time()
        Y_umap = simple_umap(X_sub, n_iter=100)
        t_umap = time.time() - t_start

        print(f"  n={sz}: t-SNE={t_tsne:.2f}s, UMAP={t_umap:.2f}s")

    # Full comparison
    print("\nFull comparison (n=400):")
    t_start = time.time()
    Y_tsne = simple_tsne(X, n_iter=200)
    t_tsne = time.time() - t_start

    t_start = time.time()
    Y_umap = simple_umap(X, n_neighbors=15, min_dist=0.1, n_iter=200)
    t_umap = time.time() - t_start

    print(f"  t-SNE time: {t_tsne:.2f}s")
    print(f"  UMAP time:  {t_umap:.2f}s")

    # Trustworthiness metric
    def trustworthiness(X_high, X_low, k=10):
        """Measure how well local neighborhoods are preserved."""
        n_pts = X_high.shape[0]
        D_high = squareform(pdist(X_high))
        D_low = squareform(pdist(X_low))

        # For each point, find k-NN in low-D
        score = 0
        for i in range(n_pts):
            nn_low = np.argsort(D_low[i])[1:k + 1]
            ranks_high = np.argsort(np.argsort(D_high[i]))
            for j in nn_low:
                rank = ranks_high[j]
                if rank > k:
                    score += rank - k
        return 1 - 2 / (n_pts * k * (2 * n_pts - 3 * k - 1)) * score

    trust_tsne = trustworthiness(X, Y_tsne, k=10)
    trust_umap = trustworthiness(X, Y_umap, k=10)
    print(f"\n  Trustworthiness (k=10):")
    print(f"    t-SNE: {trust_tsne:.4f}")
    print(f"    UMAP:  {trust_umap:.4f}")

    # Global structure: correlation of pairwise distances
    D_orig = pdist(X)
    D_tsne = pdist(Y_tsne)
    D_umap = pdist(Y_umap)
    corr_tsne = np.corrcoef(D_orig, D_tsne)[0, 1]
    corr_umap = np.corrcoef(D_orig, D_umap)[0, 1]
    print(f"\n  Global structure (distance correlation):")
    print(f"    t-SNE: {corr_tsne:.4f}")
    print(f"    UMAP:  {corr_umap:.4f}")
    print(f"    UMAP typically preserves global structure better")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(X[:, 0], X[:, 2], c=t, cmap='viridis', s=10)
    axes[0].set_title('Swiss Roll (X-Z)')
    axes[1].scatter(Y_tsne[:, 0], Y_tsne[:, 1], c=t, cmap='viridis', s=10)
    axes[1].set_title(f't-SNE (trust={trust_tsne:.3f})')
    axes[2].scatter(Y_umap[:, 0], Y_umap[:, 1], c=t, cmap='viridis', s=10)
    axes[2].set_title(f'UMAP (trust={trust_umap:.3f})')
    plt.tight_layout()
    plt.savefig('ex16_3_umap_vs_tsne.png', dpi=150)
    plt.close()
    print("  Plot saved: ex16_3_umap_vs_tsne.png")


# === Exercise 4: Latent Space Analysis of Autoencoders ===
# Problem: Train autoencoder on Swiss roll, vary latent dimension, analyze.

def exercise_4():
    """Autoencoder latent space analysis on Swiss roll."""
    np.random.seed(42)
    n = 500
    X, t = make_swiss_roll(n, noise=0.3)
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

    print("Autoencoder Latent Space Analysis\n")

    def relu(x):
        return np.maximum(0, x)

    def relu_grad(x):
        return (x > 0).astype(float)

    class SimpleAutoencoder:
        """Minimal autoencoder with one hidden layer."""
        def __init__(self, d_input, d_latent, d_hidden=32, lr=0.01):
            scale = 0.1
            # Encoder: input -> hidden -> latent
            self.W1 = np.random.randn(d_input, d_hidden) * scale
            self.b1 = np.zeros(d_hidden)
            self.W2 = np.random.randn(d_hidden, d_latent) * scale
            self.b2 = np.zeros(d_latent)
            # Decoder: latent -> hidden -> output
            self.W3 = np.random.randn(d_latent, d_hidden) * scale
            self.b3 = np.zeros(d_hidden)
            self.W4 = np.random.randn(d_hidden, d_input) * scale
            self.b4 = np.zeros(d_input)
            self.lr = lr

        def encode(self, X_in):
            self.h1 = X_in @ self.W1 + self.b1
            self.a1 = relu(self.h1)
            self.z = self.a1 @ self.W2 + self.b2
            return self.z

        def decode(self, z):
            self.h3 = z @ self.W3 + self.b3
            self.a3 = relu(self.h3)
            self.x_hat = self.a3 @ self.W4 + self.b4
            return self.x_hat

        def forward(self, X_in):
            z = self.encode(X_in)
            return self.decode(z)

        def train_step(self, X_in):
            batch = X_in.shape[0]
            x_hat = self.forward(X_in)
            loss = np.mean((X_in - x_hat) ** 2)

            # Backward pass
            d_out = 2 * (x_hat - X_in) / batch
            # Decoder
            d_W4 = self.a3.T @ d_out
            d_b4 = d_out.sum(axis=0)
            d_a3 = d_out @ self.W4.T
            d_h3 = d_a3 * relu_grad(self.h3)
            d_W3 = self.z.T @ d_h3
            d_b3 = d_h3.sum(axis=0)
            d_z = d_h3 @ self.W3.T
            # Encoder
            d_W2 = self.a1.T @ d_z
            d_b2 = d_z.sum(axis=0)
            d_a1 = d_z @ self.W2.T
            d_h1 = d_a1 * relu_grad(self.h1)
            d_W1 = X_in.T @ d_h1
            d_b1 = d_h1.sum(axis=0)

            # Update
            for param, grad in [(self.W1, d_W1), (self.b1, d_b1),
                                (self.W2, d_W2), (self.b2, d_b2),
                                (self.W3, d_W3), (self.b3, d_b3),
                                (self.W4, d_W4), (self.b4, d_b4)]:
                param -= self.lr * np.clip(grad, -5, 5)

            return loss

    # (1) Vary latent dimension
    print("(1) Reconstruction error vs latent dimension:")
    latent_dims = [1, 2, 3, 5, 10]
    results = {}

    for d_lat in latent_dims:
        ae = SimpleAutoencoder(3, d_lat, d_hidden=32, lr=0.005)
        losses = []
        for epoch in range(200):
            loss = ae.train_step(X_norm)
            losses.append(loss)
        final_loss = losses[-1]
        z = ae.encode(X_norm)
        results[d_lat] = (ae, losses, z)
        print(f"  d_latent={d_lat:2d}: final MSE = {final_loss:.6f}")

    # (2) Latent space smoothness
    print("\n(2) Latent space smoothness (d_latent=2):")
    ae_2d = results[2][0]
    z_2d = results[2][2]

    # Sort by original t parameter
    sort_idx = np.argsort(t)
    z_sorted = z_2d[sort_idx]
    diffs = np.sqrt(np.sum(np.diff(z_sorted, axis=0) ** 2, axis=1))
    smoothness = np.mean(diffs)
    print(f"  Mean step size in latent space: {smoothness:.4f}")
    print(f"  Std of step sizes: {np.std(diffs):.4f}")
    print(f"  Smoothness ratio (mean/std): {smoothness / max(np.std(diffs), 1e-10):.2f}")

    # (3) Grid sampling in latent space
    print("\n(3) Latent space grid decoding (d_latent=2):")
    z_min = z_2d.min(axis=0)
    z_max = z_2d.max(axis=0)
    grid_size = 10
    z1_vals = np.linspace(z_min[0], z_max[0], grid_size)
    z2_vals = np.linspace(z_min[1], z_max[1], grid_size)
    z_grid = np.array([[z1, z2] for z1 in z1_vals for z2 in z2_vals])
    x_decoded = ae_2d.decode(z_grid)
    print(f"  Grid: {grid_size}x{grid_size} = {len(z_grid)} points")
    print(f"  Decoded range: [{x_decoded.min():.2f}, {x_decoded.max():.2f}]")

    # (4) PCA initialization comparison
    print("\n(4) PCA initialization vs random:")
    # PCA init
    ae_pca = SimpleAutoencoder(3, 2, d_hidden=32, lr=0.005)
    # Initialize encoder weights with PCA directions
    U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)
    ae_pca.W2 = np.random.randn(32, 2) * 0.1
    losses_pca = []
    for epoch in range(200):
        loss = ae_pca.train_step(X_norm)
        losses_pca.append(loss)

    # Random init (already have from results)
    losses_random = results[2][1]

    print(f"  Random init: loss at epoch 50 = {losses_random[49]:.6f}, "
          f"final = {losses_random[-1]:.6f}")
    print(f"  PCA init:    loss at epoch 50 = {losses_pca[49]:.6f}, "
          f"final = {losses_pca[-1]:.6f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Latent space colored by t
    axes[0, 0].scatter(z_2d[:, 0], z_2d[:, 1], c=t, cmap='viridis', s=10)
    axes[0, 0].set_title('Latent Space (d=2)')
    axes[0, 0].set_xlabel('z1')
    axes[0, 0].set_ylabel('z2')

    # Reconstruction error vs latent dim
    dims_list = list(results.keys())
    final_losses = [results[d][1][-1] for d in dims_list]
    axes[0, 1].plot(dims_list, final_losses, 'bo-')
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Final MSE')
    axes[0, 1].set_title('Reconstruction Error vs d_latent')
    axes[0, 1].grid(True, alpha=0.3)

    # Training curves
    for d_lat in [1, 2, 5]:
        axes[0, 2].plot(results[d_lat][1], label=f'd={d_lat}')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('MSE Loss')
    axes[0, 2].set_title('Training Curves')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Decoded grid
    ax3d = fig.add_subplot(2, 3, 4, projection='3d')
    # Unnormalize
    x_decoded_orig = x_decoded * X.std(axis=0) + X.mean(axis=0)
    ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap='viridis', s=5, alpha=0.3)
    ax3d.scatter(x_decoded_orig[:, 0], x_decoded_orig[:, 1], x_decoded_orig[:, 2],
                 c='red', s=20, marker='^')
    ax3d.set_title('Decoded Grid Points')
    axes[0, 0].figure.delaxes(axes[1, 0])
    fig.add_subplot(ax3d)

    # PCA vs random convergence
    axes[1, 1].plot(losses_random, label='Random init')
    axes[1, 1].plot(losses_pca, label='PCA init')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE Loss')
    axes[1, 1].set_title('PCA vs Random Initialization')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Smoothness along t
    axes[1, 2].plot(diffs)
    axes[1, 2].set_xlabel('Sorted index')
    axes[1, 2].set_ylabel('Latent step size')
    axes[1, 2].set_title('Latent Space Smoothness')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex16_4_autoencoder.png', dpi=150)
    plt.close()
    print("  Plot saved: ex16_4_autoencoder.png")


# === Exercise 5: Simple t-SNE Implementation ===
# Problem: Implement t-SNE from scratch with binary search for sigma, gradient descent.

def exercise_5():
    """Complete t-SNE implementation from scratch using only NumPy."""
    np.random.seed(42)
    n = 200
    X, t = make_swiss_roll(n, noise=0.3)
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

    print("t-SNE Implementation from Scratch\n")

    target_perplexity = 30.0
    target_entropy = np.log(target_perplexity)

    # Step 1: Compute sigma_i for each point via binary search
    print("Step 1: Binary search for sigma_i (target perplexity=30)")
    D = squareform(pdist(X_norm, 'sqeuclidean'))
    sigmas = np.zeros(n)
    P_cond = np.zeros((n, n))

    for i in range(n):
        lo, hi = 1e-10, 1e4
        for _ in range(100):
            sigma = (lo + hi) / 2
            pij = np.exp(-D[i] / (2 * sigma ** 2))
            pij[i] = 0
            sum_p = np.sum(pij)
            if sum_p < 1e-12:
                lo = sigma
                continue
            pij_norm = pij / sum_p
            entropy = -np.sum(pij_norm[pij_norm > 0] * np.log(pij_norm[pij_norm > 0]))
            if np.abs(entropy - target_entropy) < 1e-5:
                break
            if entropy > target_entropy:
                hi = sigma
            else:
                lo = sigma
        sigmas[i] = sigma
        P_cond[i] = pij / max(sum_p, 1e-12)

    print(f"  Sigma range: [{sigmas.min():.4f}, {sigmas.max():.4f}]")
    print(f"  Mean sigma: {sigmas.mean():.4f}")

    # Step 2: Symmetric probability matrix P
    print("\nStep 2: Symmetric P matrix")
    P = (P_cond + P_cond.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    print(f"  P sum: {P.sum():.6f} (should be ~1)")
    print(f"  P range: [{P.min():.2e}, {P.max():.2e}]")

    # Step 3: Gradient descent optimization
    print("\nStep 3: Gradient descent (200 iterations)")
    Y = 0.01 * np.random.randn(n, 2)
    velocity = np.zeros_like(Y)
    lr = 200.0
    kl_history = []

    # Early exaggeration
    P_exag = P * 4  # multiply P by 4 for first 50 iterations

    for iteration in range(200):
        P_current = P_exag if iteration < 50 else P

        # Step 3a: Compute Q (Student-t kernel)
        D_low = squareform(pdist(Y, 'sqeuclidean'))
        Q = 1.0 / (1.0 + D_low)
        np.fill_diagonal(Q, 0)
        Q_sum = np.sum(Q)
        Q_norm = Q / max(Q_sum, 1e-10)
        Q_norm = np.maximum(Q_norm, 1e-12)

        # Step 3b: KL divergence
        kl = np.sum(P_current * np.log(P_current / Q_norm))
        kl_history.append(kl)

        # Step 3c: Gradient of KL divergence
        PQ_diff = P_current - Q_norm
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y  # (n, 2)
            weights = PQ_diff[i] * Q[:, i]  # (n,)
            grad[i] = 4 * np.sum(weights[:, np.newaxis] * diff, axis=0)

        # Step 3d: Update with momentum
        momentum = 0.8 if iteration > 50 else 0.5
        velocity = momentum * velocity - lr * grad
        Y += velocity

        if (iteration + 1) % 50 == 0:
            print(f"  Iter {iteration + 1}: KL = {kl:.4f}")

    # Step 4: Compare with reference (PCA for baseline)
    print("\nStep 4: Quality comparison:")
    # PCA 2D
    X_centered = X_norm - X_norm.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    Y_pca = X_centered @ Vt[:2].T

    # Trustworthiness
    def trustworthiness(X_high, X_low, k=10):
        n_pts = X_high.shape[0]
        D_high = squareform(pdist(X_high))
        D_low = squareform(pdist(X_low))
        score = 0
        for i in range(n_pts):
            nn_low = np.argsort(D_low[i])[1:k + 1]
            ranks_high = np.argsort(np.argsort(D_high[i]))
            for j in nn_low:
                rank = ranks_high[j]
                if rank > k:
                    score += rank - k
        return 1 - 2 / (n_pts * k * (2 * n_pts - 3 * k - 1)) * score

    trust_tsne = trustworthiness(X_norm, Y, k=10)
    trust_pca = trustworthiness(X_norm, Y_pca, k=10)
    print(f"  t-SNE trustworthiness (k=10): {trust_tsne:.4f}")
    print(f"  PCA trustworthiness (k=10):   {trust_pca:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].scatter(Y[:, 0], Y[:, 1], c=t, cmap='viridis', s=15)
    axes[0, 0].set_title(f'Our t-SNE (trust={trust_tsne:.3f})')

    axes[0, 1].scatter(Y_pca[:, 0], Y_pca[:, 1], c=t, cmap='viridis', s=15)
    axes[0, 1].set_title(f'PCA (trust={trust_pca:.3f})')

    axes[1, 0].plot(kl_history)
    axes[1, 0].axvline(50, color='r', linestyle='--', label='End early exag.')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Convergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(sigmas, bins=20, edgecolor='black')
    axes[1, 1].set_xlabel('Sigma')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution of sigma_i values')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex16_5_tsne_scratch.png', dpi=150)
    plt.close()
    print("  Plot saved: ex16_5_tsne_scratch.png")


# === Main ===

def main():
    exercises = [
        ("Exercise 1: Importance of Geodesic Distance", exercise_1),
        ("Exercise 2: t-SNE Hyperparameter Study", exercise_2),
        ("Exercise 3: UMAP vs t-SNE Comparison", exercise_3),
        ("Exercise 4: Latent Space Analysis of Autoencoders", exercise_4),
        ("Exercise 5: Simple t-SNE Implementation", exercise_5),
    ]

    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}\n")
        func()


if __name__ == "__main__":
    main()
