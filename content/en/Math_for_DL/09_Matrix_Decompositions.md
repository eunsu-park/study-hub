# Lesson 9: Matrix Decompositions

## Learning Objectives

- Review eigendecomposition and interpret eigenvalues/eigenvectors in a DL context
- Derive and apply singular value decomposition (SVD) to rectangular matrices
- Connect SVD to low-rank approximation via the Eckart-Young theorem
- Understand how PCA uses eigendecomposition/SVD for dimensionality reduction
- Apply low-rank factorization techniques (LoRA) to neural network weight matrices
- Use SVD for weight initialization and analyzing trained networks
- Compute spectral norms using SVD for spectral normalization in GANs
- Implement truncated SVD for model compression

---

## 1. Eigendecomposition Review

### 1.1 Definition

A square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ has eigenvalue $\lambda$ and eigenvector $\mathbf{v} \neq \mathbf{0}$ if:

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

If $\mathbf{A}$ has $n$ linearly independent eigenvectors, it can be decomposed as:

$$\mathbf{A} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^{-1}$$

where $\mathbf{V} = [\mathbf{v}_1, \ldots, \mathbf{v}_n]$ and $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

### 1.2 Symmetric Matrices (Spectral Theorem)

If $\mathbf{A}$ is symmetric ($\mathbf{A} = \mathbf{A}^\top$):
- All eigenvalues are **real**
- Eigenvectors are **orthogonal**
- $\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^\top$ where $\mathbf{Q}$ is orthogonal ($\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$)

### 1.3 DL Context

The Hessian $\mathbf{H}$ of the loss function is symmetric. Its eigendecomposition reveals:
- **Eigenvalues**: curvature along each principal direction
- **Eigenvectors**: the principal directions of curvature
- **Largest eigenvalue**: determines maximum safe learning rate ($\eta < 2/\lambda_\max$)

```python
import numpy as np
import matplotlib.pyplot as plt

# Eigendecomposition of a symmetric matrix
A = np.array([[3, 1],
              [1, 2]])

eigenvalues, eigenvectors = np.linalg.eigh(A)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Visualize: A maps the unit circle to an ellipse
theta = np.linspace(0, 2*np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])
ellipse = A @ circle

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Unit circle and eigenvectors
axes[0].plot(circle[0], circle[1], 'b-', linewidth=2)
for i in range(2):
    axes[0].arrow(0, 0, eigenvectors[0, i]*0.8, eigenvectors[1, i]*0.8,
                  head_width=0.08, color='red', linewidth=2)
axes[0].set_aspect('equal')
axes[0].set_title('Unit circle with eigenvectors')
axes[0].grid(True, alpha=0.3)

# Transformed ellipse
axes[1].plot(ellipse[0], ellipse[1], 'b-', linewidth=2)
for i in range(2):
    scaled = eigenvalues[i] * eigenvectors[:, i]
    axes[1].arrow(0, 0, scaled[0]*0.8, scaled[1]*0.8,
                  head_width=0.15, color='red', linewidth=2)
axes[1].set_aspect('equal')
axes[1].set_title('After transformation: $A \\cdot$ circle')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 2. Singular Value Decomposition (SVD)

### 2.1 Motivation

Eigendecomposition works only for square matrices. But weight matrices in DL are typically rectangular ($\mathbf{W} \in \mathbb{R}^{m \times n}$ with $m \neq n$). SVD generalizes eigendecomposition to any matrix.

### 2.2 Definition

Any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$$

where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$: orthogonal matrix (left singular vectors)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$: diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $\mathbf{V} \in \mathbb{R}^{n \times n}$: orthogonal matrix (right singular vectors)

### 2.3 Compact (Thin) SVD

If $r = \text{rank}(\mathbf{A}) \leq \min(m, n)$, the compact SVD keeps only non-zero singular values:

$$\mathbf{A} = \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top$$

where $\mathbf{U}_r \in \mathbb{R}^{m \times r}$, $\boldsymbol{\Sigma}_r \in \mathbb{R}^{r \times r}$, $\mathbf{V}_r \in \mathbb{R}^{n \times r}$.

### 2.4 SVD and Eigendecomposition

- $\mathbf{A}^\top \mathbf{A} = \mathbf{V} \boldsymbol{\Sigma}^2 \mathbf{V}^\top$ -- eigendecomposition of $\mathbf{A}^\top \mathbf{A}$
- $\mathbf{A} \mathbf{A}^\top = \mathbf{U} \boldsymbol{\Sigma}^2 \mathbf{U}^\top$ -- eigendecomposition of $\mathbf{A} \mathbf{A}^\top$
- Singular values = square roots of eigenvalues of $\mathbf{A}^\top \mathbf{A}$

```python
# SVD of a rectangular matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])

U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

print(f"A shape: {A.shape}")
print(f"U shape: {U.shape}")
print(f"Sigma: {sigma}")
print(f"V^T shape: {Vt.shape}")

# Reconstruct
A_reconstructed = U @ np.diag(sigma) @ Vt
print(f"\nReconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")

# Verify singular values = sqrt(eigenvalues of A^T A)
AtA = A.T @ A
eigvals_AtA = np.linalg.eigvalsh(AtA)
print(f"\nSingular values: {sigma}")
print(f"sqrt(eigvals of A^T A): {np.sqrt(np.sort(eigvals_AtA)[::-1][:2])}")
```

---

## 3. Low-Rank Approximation

### 3.1 Eckart-Young Theorem

The best rank-$k$ approximation to $\mathbf{A}$ (in Frobenius or spectral norm) is obtained by truncating the SVD:

$$\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^\top$$

The approximation error:

$$\|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$

### 3.2 Compression Ratio

Storing $\mathbf{A} \in \mathbb{R}^{m \times n}$ requires $mn$ numbers. The rank-$k$ approximation requires $k(m + n + 1)$ numbers (for $\mathbf{U}_k$, $\boldsymbol{\Sigma}_k$, $\mathbf{V}_k$).

**Compression ratio**: $\frac{mn}{k(m + n + 1)} \approx \frac{mn}{k(m+n)}$

For a $1000 \times 1000$ matrix with rank-10 approximation: $\frac{10^6}{10 \cdot 2001} \approx 50\times$ compression.

```python
# Low-rank approximation of an image-like matrix
np.random.seed(42)

# Create a matrix with rapidly decaying singular values
m, n = 100, 80
U_true, _ = np.linalg.qr(np.random.randn(m, m))
V_true, _ = np.linalg.qr(np.random.randn(n, n))
sigmas = np.exp(-np.arange(min(m, n)) * 0.1)
Sigma = np.zeros((m, n))
for i in range(min(m, n)):
    Sigma[i, i] = sigmas[i]
A = U_true @ Sigma @ V_true.T

# Compute SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Approximations at different ranks
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

ranks = [1, 3, 5, 10, 30, min(m, n)]
for ax, k in zip(axes.flat, ranks):
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    error = np.linalg.norm(A - A_k, 'fro') / np.linalg.norm(A, 'fro')
    compression = (m * n) / (k * (m + n + 1))
    ax.imshow(A_k, cmap='viridis', aspect='auto')
    ax.set_title(f'Rank {k}\nError: {error:.3f}, Compress: {compression:.1f}x')
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Low-rank approximation via truncated SVD')
plt.tight_layout()
plt.show()

# Plot singular value spectrum
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(s, 'bo-', markersize=4)
ax.set_xlabel('Index')
ax.set_ylabel('Singular value')
ax.set_title('Singular value spectrum')
ax.grid(True, alpha=0.3)
plt.show()
```

---

## 4. SVD in Deep Learning

### 4.1 LoRA (Low-Rank Adaptation)

Instead of fine-tuning a full weight matrix $\mathbf{W}_0 \in \mathbb{R}^{d \times d}$, LoRA learns a low-rank update:

$$\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$$

where $\mathbf{B} \in \mathbb{R}^{d \times r}$ and $\mathbf{A} \in \mathbb{R}^{r \times d}$ with $r \ll d$.

**Parameter savings**: $\frac{d^2}{2dr} = \frac{d}{2r}$. For $d = 4096$ and $r = 8$: 256x fewer parameters.

```python
# LoRA simulation
d = 512
r = 8  # LoRA rank

# Original weight matrix (frozen)
W0 = np.random.randn(d, d) * 0.01

# LoRA matrices (trainable)
B = np.zeros((d, r))  # Initialized to zero
A = np.random.randn(r, d) * 0.01  # Random init

# Effective weight
W_effective = W0 + B @ A

# Parameter count comparison
params_full = d * d
params_lora = d * r * 2  # B and A

print(f"Full fine-tuning parameters: {params_full:,}")
print(f"LoRA parameters: {params_lora:,}")
print(f"Reduction: {params_full / params_lora:.0f}x")
```

### 4.2 Spectral Normalization

For GANs, spectral normalization constrains the spectral norm (largest singular value) of each layer's weight matrix:

$$\bar{\mathbf{W}} = \frac{\mathbf{W}}{\sigma_1(\mathbf{W})}$$

This ensures each layer has a Lipschitz constant of 1, stabilizing discriminator training.

```python
def spectral_norm(W, n_power_iterations=10):
    """Estimate the spectral norm of W using power iteration."""
    u = np.random.randn(W.shape[0])
    u = u / np.linalg.norm(u)

    for _ in range(n_power_iterations):
        v = W.T @ u
        v = v / np.linalg.norm(v)
        u = W @ v
        u = u / np.linalg.norm(u)

    sigma = u @ W @ v
    return sigma

# Compare with SVD
W = np.random.randn(100, 80)
sigma_power = spectral_norm(W)
sigma_svd = np.linalg.svd(W, compute_uv=False)[0]

print(f"Spectral norm (power iteration): {sigma_power:.6f}")
print(f"Spectral norm (SVD): {sigma_svd:.6f}")
print(f"Error: {abs(sigma_power - sigma_svd):.2e}")

# Spectral normalization
W_normalized = W / sigma_svd
print(f"\nSpectral norm after normalization: {np.linalg.svd(W_normalized, compute_uv=False)[0]:.6f}")
```

### 4.3 SVD for Weight Analysis

After training, SVD reveals the effective dimensionality of learned transformations:

```python
# Analyze singular value spectrum of random vs structured weight matrices
np.random.seed(42)

fig, ax = plt.subplots(figsize=(8, 5))

# Random matrix (Marchenko-Pastur distribution)
W_random = np.random.randn(256, 128) / np.sqrt(128)
s_random = np.linalg.svd(W_random, compute_uv=False)

# Low-rank structure (simulating a trained layer)
U_struct = np.random.randn(256, 20)
V_struct = np.random.randn(20, 128)
W_struct = U_struct @ V_struct / np.sqrt(128) + np.random.randn(256, 128) * 0.01
s_struct = np.linalg.svd(W_struct, compute_uv=False)

ax.plot(s_random / s_random[0], 'b-', linewidth=2, label='Random init')
ax.plot(s_struct / s_struct[0], 'r-', linewidth=2, label='Trained (low-rank)')
ax.set_xlabel('Index')
ax.set_ylabel('Normalized singular value')
ax.set_title('Singular value spectra of weight matrices')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

---

## 5. PCA via SVD

### 5.1 Connection

PCA finds the directions of maximum variance in data. For centered data matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$ (rows = samples):

The covariance matrix is $\mathbf{C} = \frac{1}{N-1}\mathbf{X}^\top \mathbf{X}$.

Let $\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$ be the SVD. Then:

$$\mathbf{C} = \frac{1}{N-1}\mathbf{V} \boldsymbol{\Sigma}^2 \mathbf{V}^\top$$

The principal components are the columns of $\mathbf{V}$, and the variances along them are $\sigma_i^2 / (N-1)$.

### 5.2 Dimensionality Reduction

Project data onto the top-$k$ principal components:

$$\mathbf{X}_k = \mathbf{X} \mathbf{V}_k \in \mathbb{R}^{N \times k}$$

This preserves the maximum possible variance in $k$ dimensions.

### 5.3 DL Applications

- **Embedding visualization**: Project high-dimensional embeddings to 2D/3D for inspection
- **Data preprocessing**: Reduce feature dimensionality before feeding to a network
- **Weight compression**: Approximate weight matrices by projecting onto their top singular vectors

```python
# PCA via SVD on synthetic data
np.random.seed(42)

# Generate 2D data with correlation
N = 300
mean = np.array([2, 3])
cov = np.array([[2, 1.5],
                [1.5, 1.5]])
data = np.random.multivariate_normal(mean, cov, N)

# Center the data
data_centered = data - data.mean(axis=0)

# SVD
U, s, Vt = np.linalg.svd(data_centered, full_matrices=False)

# Principal components (rows of Vt)
pc1 = Vt[0]
pc2 = Vt[1]

# Variance explained
var_explained = s**2 / (N - 1)
var_ratio = var_explained / var_explained.sum()

print(f"Singular values: {s[:2].round(2)}")
print(f"Variance explained: {var_explained.round(3)}")
print(f"Variance ratio: {var_ratio.round(3)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original data with PCs
axes[0].scatter(data_centered[:, 0], data_centered[:, 1], alpha=0.3, s=10)
scale = 3
axes[0].arrow(0, 0, pc1[0]*scale, pc1[1]*scale, head_width=0.15, color='red', linewidth=2)
axes[0].arrow(0, 0, pc2[0]*scale*0.5, pc2[1]*scale*0.5, head_width=0.15, color='blue', linewidth=2)
axes[0].set_aspect('equal')
axes[0].set_title('Data with principal components')
axes[0].grid(True, alpha=0.3)

# Projected onto PC1
projected = data_centered @ pc1
axes[1].hist(projected, bins=30, density=True, alpha=0.7)
axes[1].set_xlabel('PC1 projection')
axes[1].set_title(f'Projection onto PC1 ({var_ratio[0]*100:.1f}% variance)')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 6. Cholesky Decomposition

### 6.1 Definition

For a symmetric positive definite matrix $\mathbf{A}$:

$$\mathbf{A} = \mathbf{L}\mathbf{L}^\top$$

where $\mathbf{L}$ is lower triangular with positive diagonal entries.

### 6.2 DL Applications

- **Sampling from multivariate Gaussian**: $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$ where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$
- **Solving linear systems**: $\mathbf{A}\mathbf{x} = \mathbf{b}$ via two triangular solves (2x faster than general solvers)
- **Natural gradient**: Approximating the Fisher information matrix

```python
# Sampling from multivariate Gaussian using Cholesky
mu = np.array([1.0, 2.0])
Sigma = np.array([[2.0, 0.8],
                   [0.8, 1.0]])

# Cholesky decomposition
L = np.linalg.cholesky(Sigma)
print(f"L:\n{L}")
print(f"L @ L^T:\n{L @ L.T}")
print(f"Sigma:\n{Sigma}")

# Sample
n_samples = 1000
z = np.random.randn(n_samples, 2)
samples = mu + (L @ z.T).T

print(f"\nSample mean: {samples.mean(axis=0).round(3)}")
print(f"Sample cov:\n{np.cov(samples.T).round(3)}")
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Eigendecomposition | $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ for symmetric matrices; reveals curvature |
| SVD | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$; works for any matrix |
| Low-rank approximation | Truncated SVD gives best rank-$k$ fit (Eckart-Young) |
| LoRA | Fine-tune with $\Delta W = BA$ where $r \ll d$; 100x+ parameter savings |
| Spectral normalization | $\bar{W} = W / \sigma_1(W)$; stabilizes GAN training |
| PCA | Eigendecomposition of covariance = SVD of centered data |
| Cholesky | $A = LL^\top$ for PD matrices; fast sampling and solving |

---

## Exercises

1. Implement power iteration to find the largest singular value and corresponding singular vectors of a matrix.
2. Compress a weight matrix to rank $k$ using truncated SVD and measure the reconstruction error vs. $k$.
3. Implement LoRA: freeze a weight matrix, learn low-rank $B$ and $A$, and verify the forward pass.
4. Apply PCA to a high-dimensional dataset (e.g., 100D) and plot the variance explained curve (scree plot).
5. Implement spectral normalization for a weight matrix using power iteration and verify the Lipschitz bound.

---

**Next**: [10. Numerical Stability](10_Numerical_Stability.md)
