"""
Exercises for Lesson 15: Graph Theory and Spectral Methods
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy.linalg import eigh
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Proof of Quadratic Form of Laplacian ===
# Problem: Prove x^T L x = (1/2) sum_ij A_ij (x_i - x_j)^2
#          and show L is positive semidefinite.

def exercise_1():
    """Proof and numerical verification of Laplacian quadratic form."""
    print("Proof: x^T L x = (1/2) sum_{i,j} A_{ij} (x_i - x_j)^2\n")

    print("Step 1: Expand the right side")
    print("  (1/2) sum_{i,j} A_{ij} (x_i - x_j)^2")
    print("  = (1/2) sum_{i,j} A_{ij} (x_i^2 - 2 x_i x_j + x_j^2)")
    print("  = (1/2) [sum_{i,j} A_{ij} x_i^2 - 2 sum_{i,j} A_{ij} x_i x_j + sum_{i,j} A_{ij} x_j^2]")
    print()
    print("Step 2: Simplify each term")
    print("  sum_{i,j} A_{ij} x_i^2 = sum_i x_i^2 sum_j A_{ij} = sum_i d_i x_i^2 = x^T D x")
    print("  sum_{i,j} A_{ij} x_j^2 = sum_j x_j^2 sum_i A_{ij} = sum_j d_j x_j^2 = x^T D x")
    print("  sum_{i,j} A_{ij} x_i x_j = x^T A x")
    print()
    print("Step 3: Combine")
    print("  = (1/2) [x^T D x - 2 x^T A x + x^T D x]")
    print("  = x^T D x - x^T A x")
    print("  = x^T (D - A) x")
    print("  = x^T L x  (since L = D - A)  QED")
    print()
    print("Positive semidefiniteness:")
    print("  x^T L x = (1/2) sum_{i,j} A_{ij} (x_i - x_j)^2 >= 0")
    print("  since A_{ij} >= 0 and (x_i - x_j)^2 >= 0 for all i,j")
    print("  Therefore L is positive semidefinite (PSD).")

    # Numerical verification
    print("\nNumerical verification:")
    np.random.seed(42)

    # Create a random graph
    n = 6
    A = np.random.randint(0, 2, (n, n))
    A = (A + A.T) // 2  # make symmetric
    np.fill_diagonal(A, 0)

    D = np.diag(A.sum(axis=1))
    L = D - A

    x = np.random.randn(n)

    # Method 1: x^T L x
    quadratic = x @ L @ x

    # Method 2: (1/2) sum A_ij (x_i - x_j)^2
    sum_form = 0
    for i in range(n):
        for j in range(n):
            sum_form += A[i, j] * (x[i] - x[j]) ** 2
    sum_form *= 0.5

    print(f"  x^T L x = {quadratic:.6f}")
    print(f"  (1/2) sum A_ij (x_i-x_j)^2 = {sum_form:.6f}")
    print(f"  Match: {np.allclose(quadratic, sum_form)}")

    # Verify PSD via eigenvalues
    eigenvalues = np.linalg.eigvalsh(L)
    print(f"\n  Eigenvalues of L: {eigenvalues}")
    print(f"  All non-negative: {np.all(eigenvalues >= -1e-10)}")
    print(f"  Smallest eigenvalue: {eigenvalues[0]:.6f} (should be ~0 for connected graph)")


# === Exercise 2: Spectral Clustering Implementation ===
# Problem: Implement spectral clustering with k-means from scratch (no sklearn).

def exercise_2():
    """Spectral clustering: normalized Laplacian, eigen-decomposition, k-means."""
    np.random.seed(42)

    # Generate clustered data: 3 clusters
    n_per_cluster = 50
    centers = [(0, 0), (5, 0), (2.5, 4)]
    X = np.vstack([
        np.random.randn(n_per_cluster, 2) * 0.7 + center
        for center in centers
    ])
    true_labels = np.repeat(np.arange(3), n_per_cluster)
    n = len(X)
    k = 3

    print("Spectral Clustering Implementation (from scratch)")
    print(f"Data: {n} points, {k} clusters\n")

    # Step 1: Compute similarity matrix (Gaussian kernel)
    sigma = 1.5
    dists = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2)
    W = np.exp(-dists / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)
    print(f"Step 1: Gaussian similarity matrix (sigma={sigma})")

    # Step 2: Normalized Laplacian (L_sym = I - D^{-1/2} W D^{-1/2})
    D_vec = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_vec + 1e-10))
    L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    print("Step 2: Normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}")

    # Step 3: Eigenvalue decomposition (smallest k eigenvectors)
    eigenvalues, eigenvectors = eigh(L_sym)
    U = eigenvectors[:, :k]  # first k eigenvectors
    # Row-normalize
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    U_norm = U / norms
    print(f"Step 3: Smallest {k} eigenvalues: {eigenvalues[:k]}")

    # Step 4: k-means clustering (Lloyd's algorithm)
    def kmeans(data, k_clusters, max_iter=100):
        """Simple k-means implementation."""
        n_samples = len(data)
        # Initialize centroids randomly
        idx = np.random.choice(n_samples, k_clusters, replace=False)
        centroids = data[idx].copy()
        labels = np.zeros(n_samples, dtype=int)

        for iteration in range(max_iter):
            # Assign to nearest centroid
            new_labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                dists_to_c = np.sum((centroids - data[i]) ** 2, axis=1)
                new_labels[i] = np.argmin(dists_to_c)

            # Check convergence
            if np.all(new_labels == labels):
                break
            labels = new_labels

            # Update centroids
            for c in range(k_clusters):
                mask = labels == c
                if np.sum(mask) > 0:
                    centroids[c] = data[mask].mean(axis=0)

        return labels, centroids, iteration + 1

    labels_pred, centroids, n_iter = kmeans(U_norm, k)
    print(f"Step 4: k-means converged in {n_iter} iterations")

    # Step 5: Evaluate clustering quality (silhouette score)
    def silhouette_score(data, labels_s):
        """Compute mean silhouette score."""
        n_samples = len(data)
        scores = np.zeros(n_samples)
        unique_labels = np.unique(labels_s)

        for i in range(n_samples):
            own_cluster = labels_s[i]
            # a(i): mean intra-cluster distance
            same = data[labels_s == own_cluster]
            if len(same) > 1:
                a_i = np.mean(np.sqrt(np.sum((same - data[i]) ** 2, axis=1)))
            else:
                a_i = 0

            # b(i): min mean distance to other clusters
            b_i = np.inf
            for c in unique_labels:
                if c == own_cluster:
                    continue
                other = data[labels_s == c]
                if len(other) > 0:
                    mean_dist = np.mean(np.sqrt(np.sum((other - data[i]) ** 2, axis=1)))
                    b_i = min(b_i, mean_dist)

            scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

        return np.mean(scores)

    sil_spectral = silhouette_score(X, labels_pred)

    # Compare with direct k-means on original data
    labels_direct, _, _ = kmeans(X, k)
    sil_direct = silhouette_score(X, labels_direct)

    # Accuracy (with best permutation matching)
    def cluster_accuracy(true, pred):
        from itertools import permutations
        best_acc = 0
        for perm in permutations(range(k)):
            remapped = np.array([perm[l] for l in pred])
            acc = np.mean(remapped == true)
            best_acc = max(best_acc, acc)
        return best_acc

    acc_spectral = cluster_accuracy(true_labels, labels_pred)
    acc_direct = cluster_accuracy(true_labels, labels_direct)

    print(f"\nResults:")
    print(f"  Spectral clustering: silhouette={sil_spectral:.3f}, accuracy={acc_spectral:.3f}")
    print(f"  Direct k-means:      silhouette={sil_direct:.3f}, accuracy={acc_direct:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for label in range(k):
        mask = true_labels == label
        axes[0].scatter(X[mask, 0], X[mask, 1], s=20, alpha=0.7)
    axes[0].set_title('Ground Truth')

    for label in range(k):
        mask = labels_pred == label
        axes[1].scatter(X[mask, 0], X[mask, 1], s=20, alpha=0.7)
    axes[1].set_title(f'Spectral Clustering (sil={sil_spectral:.3f})')

    axes[2].plot(eigenvalues[:10], 'bo-')
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel('Eigenvalue')
    axes[2].set_title('Laplacian Eigenvalues (eigengap at k=3)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex15_2_spectral_clustering.png', dpi=150)
    plt.close()
    print("  Plot saved: ex15_2_spectral_clustering.png")


# === Exercise 3: Eigenvalue Interpretation of PageRank ===
# Problem: Transform PageRank equation into eigenvector problem, solve with power iteration.

def exercise_3():
    """PageRank as eigenvector problem solved via power iteration."""
    np.random.seed(42)

    print("PageRank as Eigenvector Problem\n")

    # Create a small web graph (10 pages)
    n = 10
    # Adjacency: page i links to pages in adj_list[i]
    adj_list = {
        0: [1, 2, 3],
        1: [0, 3],
        2: [0, 4, 5],
        3: [1, 4],
        4: [0, 2, 5, 6],
        5: [3, 7],
        6: [4, 7, 8],
        7: [5, 8, 9],
        8: [6, 9],
        9: [0, 7],
    }

    # Build transition matrix P (column-stochastic -> row-stochastic for our formulation)
    P = np.zeros((n, n))
    for i, links in adj_list.items():
        for j in links:
            P[i, j] = 1.0 / len(links)

    print("Mathematical Derivation:")
    print("  PageRank equation: r = (1-d)*e/n + d * P^T * r")
    print("  Rearrange: r = [(1-d)/n * e*1^T + d*P^T] * r")
    print("  Define: M = (1-d)/n * E + d*P^T  (where E = e*1^T)")
    print("  Then: r = M * r")
    print("  This is an eigenvector equation: M*r = 1*r")
    print("  r is the eigenvector of M with eigenvalue 1.\n")

    d = 0.85  # damping factor
    e = np.ones((n, n)) / n
    M = (1 - d) * e + d * P.T

    # Power iteration
    print("Power iteration:")
    r = np.ones(n) / n  # initial uniform
    for iteration in range(100):
        r_new = M @ r
        r_new /= np.sum(r_new)
        diff = np.linalg.norm(r_new - r)
        if diff < 1e-10:
            print(f"  Converged in {iteration + 1} iterations (diff={diff:.2e})")
            break
        r = r_new

    r_power = r_new

    # Verify: compare with direct eigenvector computation
    eigenvalues, eigenvectors = np.linalg.eig(M)
    # Find eigenvector for eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    r_eigen = np.real(eigenvectors[:, idx])
    r_eigen = np.abs(r_eigen) / np.sum(np.abs(r_eigen))

    print(f"\nPageRank scores:")
    print(f"  {'Page':>4s}  {'Power Iter':>10s}  {'Eigenvector':>11s}  {'Match':>5s}")
    for i in range(n):
        match = np.isclose(r_power[i], r_eigen[i], atol=1e-6)
        print(f"  {i:4d}  {r_power[i]:10.6f}  {r_eigen[i]:11.6f}  {'Yes' if match else 'No':>5s}")

    # Verify dominant eigenvalue = 1
    print(f"\n  Dominant eigenvalue: {eigenvalues[idx]:.6f} (should be ~1.0)")
    print(f"  Power iteration matches eigenvector: {np.allclose(r_power, r_eigen, atol=1e-5)}")

    # Top-ranked pages
    ranking = np.argsort(r_power)[::-1]
    print(f"\n  Top 3 pages: {ranking[:3]} (scores: {r_power[ranking[:3]]})")
    print(f"  Page 0 has many incoming links -> high PageRank")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(range(n), r_power)
    axes[0].set_xlabel('Page')
    axes[0].set_ylabel('PageRank')
    axes[0].set_title('PageRank Scores (Power Iteration)')
    axes[0].grid(True, alpha=0.3)

    # Convergence plot
    r = np.ones(n) / n
    diffs = []
    for _ in range(50):
        r_new = M @ r
        r_new /= np.sum(r_new)
        diffs.append(np.linalg.norm(r_new - r))
        r = r_new
    axes[1].semilogy(diffs)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('||r_{k+1} - r_k||')
    axes[1].set_title('Power Iteration Convergence')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex15_3_pagerank.png', dpi=150)
    plt.close()
    print("  Plot saved: ex15_3_pagerank.png")


# === Exercise 4: Graph Fourier Transform Application ===
# Problem: Ring graph Laplacian, GFT of low/high frequency signals, graph filtering.

def exercise_4():
    """Graph Fourier transform on ring graph with low-pass filtering."""
    n = 20  # ring graph vertices

    # Build ring graph adjacency
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[i, (i - 1) % n] = 1

    D = np.diag(A.sum(axis=1))
    L = D - A

    # Step 1: Eigendecomposition
    eigenvalues, eigenvectors = eigh(L)
    print("Graph Fourier Transform on Ring Graph (n=20)\n")
    print("Step 1: Laplacian eigenvalues:")
    print(f"  {eigenvalues}")
    print(f"  Range: [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")
    print(f"  Theoretical for ring: lambda_k = 2 - 2*cos(2*pi*k/n)")

    # Step 2: GFT of signals
    print("\nStep 2: Graph Fourier Transform of signals:")

    # Low-frequency signal: cos(2*pi*i/20)
    f_low = np.cos(2 * np.pi * np.arange(n) / n)
    f_low_hat = eigenvectors.T @ f_low

    # High-frequency signal: (-1)^i
    f_high = np.array([(-1) ** i for i in range(n)], dtype=float)
    f_high_hat = eigenvectors.T @ f_high

    print(f"  Low-frequency signal (cos): energy concentrated at small eigenvalues")
    print(f"    Top 3 GFT coefficients at indices: {np.argsort(np.abs(f_low_hat))[-3:][::-1]}")
    print(f"  High-frequency signal (alternating): energy at large eigenvalues")
    print(f"    Top 3 GFT coefficients at indices: {np.argsort(np.abs(f_high_hat))[-3:][::-1]}")

    # Step 3: Gaussian low-pass filter
    print("\nStep 3: Gaussian low-pass graph filter:")
    sigma_vals = [0.5, 1.0, 2.0]

    # Create a noisy signal (low freq + high freq noise)
    np.random.seed(42)
    f_noisy = f_low + 0.5 * f_high + 0.3 * np.random.randn(n)
    f_noisy_hat = eigenvectors.T @ f_noisy

    for sigma_g in sigma_vals:
        # Filter in spectral domain: h(lambda) = exp(-lambda^2 / (2*sigma^2))
        h = np.exp(-eigenvalues ** 2 / (2 * sigma_g ** 2))
        f_filtered_hat = h * f_noisy_hat
        f_filtered = eigenvectors @ f_filtered_hat

        error = np.linalg.norm(f_filtered - f_low) / np.linalg.norm(f_low)
        print(f"  sigma={sigma_g}: relative error to clean signal = {error:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Eigenvectors
    for k in [0, 1, 2, n // 2]:
        axes[0, 0].plot(eigenvectors[:, k], 'o-', markersize=4, label=f'k={k}')
    axes[0, 0].set_title('Laplacian Eigenvectors')
    axes[0, 0].set_xlabel('Vertex')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # GFT of low-freq signal
    axes[0, 1].stem(eigenvalues, np.abs(f_low_hat), markerfmt='bo', basefmt='k-')
    axes[0, 1].set_title('GFT of Low-freq Signal')
    axes[0, 1].set_xlabel('Eigenvalue (frequency)')
    axes[0, 1].set_ylabel('|coefficient|')
    axes[0, 1].grid(True, alpha=0.3)

    # GFT of high-freq signal
    axes[0, 2].stem(eigenvalues, np.abs(f_high_hat), markerfmt='ro', basefmt='k-')
    axes[0, 2].set_title('GFT of High-freq Signal')
    axes[0, 2].set_xlabel('Eigenvalue (frequency)')
    axes[0, 2].set_ylabel('|coefficient|')
    axes[0, 2].grid(True, alpha=0.3)

    # Noisy and filtered signals
    axes[1, 0].plot(f_noisy, 'k--', label='Noisy', alpha=0.5)
    axes[1, 0].plot(f_low, 'b-', linewidth=2, label='Clean')
    axes[1, 0].set_title('Noisy vs Clean Signal')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    for sigma_g in sigma_vals:
        h = np.exp(-eigenvalues ** 2 / (2 * sigma_g ** 2))
        f_filt = eigenvectors @ (h * f_noisy_hat)
        axes[1, 1].plot(f_filt, 'o-', markersize=4, label=f'sigma={sigma_g}')
    axes[1, 1].plot(f_low, 'k--', linewidth=2, label='Clean')
    axes[1, 1].set_title('Low-pass Filtered Signals')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # Filter responses
    for sigma_g in sigma_vals:
        h = np.exp(-eigenvalues ** 2 / (2 * sigma_g ** 2))
        axes[1, 2].plot(eigenvalues, h, 'o-', markersize=4, label=f'sigma={sigma_g}')
    axes[1, 2].set_title('Gaussian Filter Response')
    axes[1, 2].set_xlabel('Eigenvalue (frequency)')
    axes[1, 2].set_ylabel('h(lambda)')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex15_4_graph_fourier.png', dpi=150)
    plt.close()
    print("  Plot saved: ex15_4_graph_fourier.png")


# === Exercise 5: GCN vs GAT Comparison ===
# Problem: Compare GCN and GAT layers on a small graph, visualize attention.

def exercise_5():
    """GCN vs GAT layer comparison on a small graph."""
    np.random.seed(42)

    n = 15  # vertices
    d_in = 4  # input feature dimension
    d_out = 2  # output dimension
    n_heads = 2

    # Generate random graph
    A = np.zeros((n, n))
    # Erdos-Renyi with p=0.3
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < 0.3:
                A[i, j] = 1
                A[j, i] = 1
    # Ensure connected: add edges from isolated nodes
    for i in range(n):
        if A[i].sum() == 0:
            j = (i + 1) % n
            A[i, j] = 1
            A[j, i] = 1

    # Add self-loops
    A_hat = A + np.eye(n)

    # Random node features
    X = np.random.randn(n, d_in)

    print("GCN vs GAT Comparison\n")
    print(f"Graph: {n} vertices, {int(A.sum() / 2)} edges, features: {d_in}D -> {d_out}D\n")

    # GCN Layer: H' = D_hat^{-1/2} A_hat D_hat^{-1/2} X W
    print("1. GCN Layer:")
    D_hat = np.diag(A_hat.sum(axis=1))
    D_hat_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_hat)))
    A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

    W_gcn = np.random.randn(d_in, d_out) * 0.5
    H_gcn = A_norm @ X @ W_gcn

    print(f"  A_hat = A + I (self-loops)")
    print(f"  H = D_hat^{{-1/2}} A_hat D_hat^{{-1/2}} X W")
    print(f"  Output shape: {H_gcn.shape}")
    print(f"  Output range: [{H_gcn.min():.3f}, {H_gcn.max():.3f}]")

    # GCN effectively averages neighbor features weighted by degree
    # All neighbors get equal weight (normalized by degree)
    gcn_weights = A_norm.copy()
    np.fill_diagonal(gcn_weights, 0)  # remove self
    print(f"  Neighbor weights (node 0): {gcn_weights[0, gcn_weights[0] > 0]}")
    print(f"  GCN treats all neighbors equally (up to degree normalization)")

    # GAT Layer: attention mechanism
    print("\n2. GAT Layer:")

    def gat_layer(X_in, A_adj, W_gat, a_vec, n_heads_gat):
        """Single GAT layer with multi-head attention."""
        n_nodes = X_in.shape[0]
        d_out_head = W_gat.shape[1] // n_heads_gat
        all_outputs = []
        all_attentions = []

        for h in range(n_heads_gat):
            # Linear projection for this head
            W_h = W_gat[:, h * d_out_head:(h + 1) * d_out_head]
            a_h = a_vec[h]  # attention vector for head h
            Z = X_in @ W_h  # (n, d_out_head)

            # Compute attention coefficients
            # e_ij = LeakyReLU(a^T [z_i || z_j])
            alpha_raw = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if A_adj[i, j] > 0:  # only compute for neighbors
                        concat = np.concatenate([Z[i], Z[j]])
                        e_ij = np.dot(a_h, concat)
                        # LeakyReLU
                        alpha_raw[i, j] = e_ij if e_ij > 0 else 0.2 * e_ij

            # Softmax over neighbors
            alpha = np.zeros_like(alpha_raw)
            for i in range(n_nodes):
                neighbors = np.where(A_adj[i] > 0)[0]
                if len(neighbors) > 0:
                    e_vals = alpha_raw[i, neighbors]
                    e_vals = e_vals - np.max(e_vals)  # stability
                    exp_vals = np.exp(e_vals)
                    alpha[i, neighbors] = exp_vals / np.sum(exp_vals)

            # Weighted aggregation
            H_head = alpha @ Z  # (n, d_out_head)
            all_outputs.append(H_head)
            all_attentions.append(alpha)

        # Concatenate heads
        H_out = np.concatenate(all_outputs, axis=1)
        return H_out, all_attentions

    W_gat = np.random.randn(d_in, d_out * n_heads) * 0.5
    a_vecs = [np.random.randn(d_out * 2) * 0.5 for _ in range(n_heads)]

    t_start = time.time()
    for _ in range(100):
        # GCN
        _ = A_norm @ X @ W_gcn
    gcn_time = (time.time() - t_start) / 100

    t_start = time.time()
    for _ in range(10):
        # GAT
        H_gat, attentions = gat_layer(X, A_hat, W_gat, a_vecs, n_heads)
    gat_time = (time.time() - t_start) / 10

    print(f"  Multi-head attention with {n_heads} heads")
    print(f"  Output shape: {H_gat.shape}")

    # Visualize attention weights
    print(f"\n3. Attention weight analysis (head 0):")
    attn = attentions[0]
    node_0_neighbors = np.where(A_hat[0] > 0)[0]
    print(f"  Node 0 neighbors: {node_0_neighbors}")
    print(f"  Attention weights: {attn[0, node_0_neighbors]}")
    print(f"  GAT learns different importance for each neighbor")
    print(f"  Entropy of attention (node 0): {-np.sum(attn[0, node_0_neighbors] * np.log(attn[0, node_0_neighbors] + 1e-10)):.3f}")

    # Compare uniformity
    uniform_entropy = np.log(len(node_0_neighbors))
    actual_entropy = -np.sum(attn[0, node_0_neighbors] * np.log(attn[0, node_0_neighbors] + 1e-10))
    print(f"  Uniform entropy: {uniform_entropy:.3f}")
    print(f"  Higher deviation from uniform = more selective attention")

    # Complexity analysis
    n_edges = int(A_hat.sum())
    print(f"\n4. Computational complexity:")
    print(f"  GCN: O(|E| * d_in * d_out) = O({n_edges} * {d_in} * {d_out}) = O({n_edges * d_in * d_out})")
    print(f"  GAT: O(|E| * d_out + |V| * d_in * d_out) = O({n_edges * d_out} + {n * d_in * d_out})")
    print(f"  GCN time: {gcn_time * 1000:.3f} ms")
    print(f"  GAT time: {gat_time * 1000:.3f} ms")
    print(f"  GAT is ~{gat_time / max(gcn_time, 1e-10):.1f}x slower (attention computation overhead)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # GCN output
    axes[0].scatter(H_gcn[:, 0], H_gcn[:, 1], c=np.arange(n), cmap='viridis', s=60)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                axes[0].plot([H_gcn[i, 0], H_gcn[j, 0]], [H_gcn[i, 1], H_gcn[j, 1]],
                             'k-', alpha=0.2)
    axes[0].set_title('GCN Embeddings')
    axes[0].set_xlabel('Dim 0')
    axes[0].set_ylabel('Dim 1')

    # GAT output
    H_gat_vis = H_gat[:, :2]  # first head output
    axes[1].scatter(H_gat_vis[:, 0], H_gat_vis[:, 1], c=np.arange(n), cmap='viridis', s=60)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                w = (attentions[0][i, j] + attentions[0][j, i]) / 2
                axes[1].plot([H_gat_vis[i, 0], H_gat_vis[j, 0]],
                             [H_gat_vis[i, 1], H_gat_vis[j, 1]],
                             'k-', alpha=w * 5, linewidth=w * 10)
    axes[1].set_title('GAT Embeddings (edge=attention)')
    axes[1].set_xlabel('Dim 0')
    axes[1].set_ylabel('Dim 1')

    # Attention heatmap
    im = axes[2].imshow(attentions[0], cmap='hot', aspect='auto')
    axes[2].set_title('GAT Attention Weights (Head 0)')
    axes[2].set_xlabel('Target')
    axes[2].set_ylabel('Source')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig('ex15_5_gcn_vs_gat.png', dpi=150)
    plt.close()
    print("  Plot saved: ex15_5_gcn_vs_gat.png")


# === Main ===

def main():
    exercises = [
        ("Exercise 1: Proof of Quadratic Form of Laplacian", exercise_1),
        ("Exercise 2: Spectral Clustering Implementation", exercise_2),
        ("Exercise 3: Eigenvalue Interpretation of PageRank", exercise_3),
        ("Exercise 4: Graph Fourier Transform Application", exercise_4),
        ("Exercise 5: GCN vs GAT Comparison", exercise_5),
    ]

    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}\n")
        func()


if __name__ == "__main__":
    main()
