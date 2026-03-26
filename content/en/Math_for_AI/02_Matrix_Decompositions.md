# 02. Matrix Decompositions

## Learning Objectives

- Understand the concept of eigenvalues and eigenvectors and be able to calculate them
- Explain the spectral theorem for symmetric matrices and properties of positive definite matrices
- Understand the geometric meaning of singular value decomposition (SVD) and implement it
- Explain the mathematical principles of low-rank approximation using SVD and PCA
- Understand the purpose and computation methods of LU, QR, and Cholesky decompositions
- Give concrete examples of how matrix decompositions are used in machine learning

---

## 1. Eigenvalues and Eigenvectors

### 1.1 Definition and Meaning

An eigenvector $\mathbf{v}$ of matrix $A$ is a non-zero vector that satisfies:

$$A\mathbf{v} = \lambda\mathbf{v}$$

where $\lambda$ is the eigenvalue.

**Geometric interpretation**: An eigenvector is a special direction that doesn't change direction under linear transformation $A$, only changes magnitude by a factor of $\lambda$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Eigenvalues and eigenvectors of a 2x2 matrix
A = np.array([[3, 1],
              [0, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verification: Av = λv
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    Av = A @ v
    lam_v = lam * v
    print(f"\nEigenvector {i+1}:")
    print(f"Av = {Av}")
    print(f"λv = {lam_v}")
    print(f"Equal? {np.allclose(Av, lam_v)}")
```

### 1.2 Characteristic Equation

Eigenvalues are solutions to the characteristic equation:

$$\det(A - \lambda I) = 0$$

This is an $n$-th degree polynomial with $n$ eigenvalues (including multiplicities).

```python
# Manual computation of characteristic equation (2x2 example)
A = np.array([[4, 2],
              [1, 3]])

# det(A - λI) = 0
# (4-λ)(3-λ) - 2*1 = 0
# λ² - 7λ + 10 = 0
# (λ-5)(λ-2) = 0

eigenvalues = np.linalg.eigvals(A)
print(f"Eigenvalues: {eigenvalues}")  # [5, 2]

# Verification: characteristic equation
for lam in eigenvalues:
    det = np.linalg.det(A - lam * np.eye(2))
    print(f"det(A - {lam}I) = {det:.10f}")  # ~0
```

### 1.3 Eigendecomposition

If an $n \times n$ matrix $A$ has $n$ linearly independent eigenvectors:

$$A = V\Lambda V^{-1}$$

where:
- $V$: Matrix with eigenvectors as columns
- $\Lambda$: Diagonal matrix with eigenvalues on diagonal

```python
# Eigendecomposition
A = np.array([[3, 1],
              [0, 2]])

eigenvalues, V = np.linalg.eig(A)
Lambda = np.diag(eigenvalues)

print(f"V (eigenvectors):\n{V}")
print(f"Λ (eigenvalues):\n{Lambda}")

# Reconstruction: A = VΛV^(-1)
A_reconstructed = V @ Lambda @ np.linalg.inv(V)
print(f"A reconstructed:\n{A_reconstructed}")
print(f"Original A:\n{A}")
print(f"Equal? {np.allclose(A, A_reconstructed)}")
```

### 1.4 Matrix Powers and Functions

Matrix powers become easier using eigendecomposition:

$$A^k = V\Lambda^k V^{-1}$$

Matrix functions can also be defined:

$$f(A) = Vf(\Lambda)V^{-1}$$

```python
# Matrix powers
A = np.array([[2, 1],
              [1, 2]])

eigenvalues, V = np.linalg.eig(A)
Lambda = np.diag(eigenvalues)

# Compute A^10 (two methods)
# Method 1: direct multiplication
A_10_direct = np.linalg.matrix_power(A, 10)

# Method 2: using eigendecomposition
Lambda_10 = np.diag(eigenvalues**10)
A_10_eigen = V @ Lambda_10 @ np.linalg.inv(V)

print(f"A^10 (direct):\n{A_10_direct}")
print(f"A^10 (eigendecomposition):\n{A_10_eigen.real}")
print(f"Equal? {np.allclose(A_10_direct, A_10_eigen.real)}")

# Matrix exponential function
from scipy.linalg import expm
exp_A_scipy = expm(A)
exp_Lambda = np.diag(np.exp(eigenvalues))
exp_A_eigen = V @ exp_Lambda @ np.linalg.inv(V)
print(f"exp(A) match? {np.allclose(exp_A_scipy, exp_A_eigen.real)}")
```

## 2. Spectral Theorem for Symmetric Matrices

### 2.1 Spectral Theorem

For a real symmetric matrix $A = A^T$:
1. **All eigenvalues are real**
2. **Eigenvectors corresponding to different eigenvalues are orthogonal**
3. **Orthogonally diagonalizable**: $A = Q\Lambda Q^T$ (where $Q$ is orthogonal)

```python
# Eigendecomposition of a symmetric matrix
A_sym = np.array([[4, 2, 0],
                  [2, 3, 1],
                  [0, 1, 2]])

# Check if symmetric
print(f"Is symmetric? {np.allclose(A_sym, A_sym.T)}")

eigenvalues, Q = np.linalg.eigh(A_sym)  # function dedicated to symmetric matrices
Lambda = np.diag(eigenvalues)

print(f"Eigenvalues: {eigenvalues}")
print(f"Q (orthogonal matrix):\n{Q}")

# Check orthogonality
print(f"Q^T Q = I?\n{Q.T @ Q}")
print(f"Is orthogonal? {np.allclose(Q.T @ Q, np.eye(3))}")

# Reconstruction: A = QΛQ^T
A_reconstructed = Q @ Lambda @ Q.T
print(f"Reconstruction error: {np.linalg.norm(A_sym - A_reconstructed)}")
```

### 2.2 Positive Definite Matrix

A symmetric matrix $A$ is positive definite if:

$$\mathbf{x}^T A \mathbf{x} > 0 \quad \forall \mathbf{x} \neq \mathbf{0}$$

**Equivalent conditions**:
- All eigenvalues are positive
- All leading principal minors are positive
- Cholesky decomposition exists

Positive definite matrices are very important in machine learning (covariance matrices, Hessians, etc.).

```python
# Positive definite matrix example
A_pd = np.array([[2, -1, 0],
                 [-1, 2, -1],
                 [0, -1, 2]])

eigenvalues = np.linalg.eigvalsh(A_pd)
print(f"Eigenvalues: {eigenvalues}")
print(f"All positive? {np.all(eigenvalues > 0)}")

# Verify x^T A x > 0
x = np.random.randn(3)
quadratic_form = x.T @ A_pd @ x
print(f"x^T A x = {quadratic_form}")
print(f"Positive? {quadratic_form > 0}")

# Positive semidefinite (PSD) example
A_psd = np.array([[1, 1],
                  [1, 1]])
eigenvalues_psd = np.linalg.eigvalsh(A_psd)
print(f"PSD eigenvalues: {eigenvalues_psd}")  # [0, 2]
print(f"All non-negative? {np.all(eigenvalues_psd >= 0)}")
```

### 2.3 Geometric Interpretation of Symmetric Matrices

Symmetric matrices define ellipsoids. Eigenvectors are the directions of the principal axes, and eigenvalues correspond to the length of each axis.

```python
# 2D ellipse visualization
A = np.array([[3, 1],
              [1, 2]])

eigenvalues, eigenvectors = np.linalg.eigh(A)

# Transform the unit circle by A
theta = np.linspace(0, 2*np.pi, 100)
unit_circle = np.array([np.cos(theta), np.sin(theta)])
ellipse = A @ unit_circle

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Unit circle
ax1.plot(unit_circle[0], unit_circle[1], 'b-', label='Unit circle')
ax1.set_aspect('equal')
ax1.grid(True)
ax1.legend()
ax1.set_title('Before transformation')

# Ellipse
ax2.plot(ellipse[0], ellipse[1], 'r-', label='Ellipse')
# Draw eigenvectors
for i in range(2):
    scale = np.sqrt(eigenvalues[i])
    v = eigenvectors[:, i] * scale
    ax2.arrow(0, 0, v[0], v[1], head_width=0.2, head_length=0.2,
              fc=f'C{i}', ec=f'C{i}', label=f'√λ{i+1} * v{i+1}')
ax2.set_aspect('equal')
ax2.grid(True)
ax2.legend()
ax2.set_title('After transformation by A')

plt.tight_layout()
plt.show()
```

## 3. Singular Value Decomposition (SVD)

### 3.1 SVD Definition

Every $m \times n$ matrix $A$ can be decomposed as:

$$A = U\Sigma V^T$$

where:
- $U$: $m \times m$ orthogonal matrix (left singular vectors)
- $\Sigma$: $m \times n$ diagonal matrix (singular values, $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$)
- $V$: $n \times n$ orthogonal matrix (right singular vectors)

```python
# SVD computation
A = np.array([[3, 1, 1],
              [-1, 3, 1]])

U, sigma, VT = np.linalg.svd(A, full_matrices=True)

print(f"A shape: {A.shape}")
print(f"U shape: {U.shape}")
print(f"sigma: {sigma}")  # 1D array
print(f"V^T shape: {VT.shape}")

# Reconstruct Σ matrix (m x n)
Sigma = np.zeros_like(A, dtype=float)
Sigma[:min(A.shape), :min(A.shape)] = np.diag(sigma)

print(f"Sigma shape: {Sigma.shape}")
print(f"Sigma:\n{Sigma}")

# Verify reconstruction
A_reconstructed = U @ Sigma @ VT
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed)}")
```

### 3.2 Geometric Meaning of SVD

SVD breaks down any linear transformation into three steps:
1. $V^T$: Rotation/reflection
2. $\Sigma$: Axis-aligned scaling
3. $U$: Rotation/reflection

```
  V^T          Σ           U
 ───────    ───────    ───────
 Rotate  →  Scale   →  Rotate
```

```python
# Geometric interpretation of SVD
A = np.array([[3, 1],
              [1, 2]])

U, sigma, VT = np.linalg.svd(A)

# Unit circle
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

# Apply each step
step1 = VT @ circle  # first rotation
Sigma_2d = np.diag(sigma)
step2 = Sigma_2d @ step1  # scaling
step3 = U @ step2  # second rotation

# Final result (all at once)
final = A @ circle

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].plot(circle[0], circle[1], 'b-')
axes[0, 0].set_title('1. Original circle')
axes[0, 0].set_aspect('equal')
axes[0, 0].grid(True)

axes[0, 1].plot(step1[0], step1[1], 'g-')
axes[0, 1].set_title('2. After V^T (rotation)')
axes[0, 1].set_aspect('equal')
axes[0, 1].grid(True)

axes[0, 2].plot(step2[0], step2[1], 'm-')
axes[0, 2].set_title('3. After Σ (scaling)')
axes[0, 2].set_aspect('equal')
axes[0, 2].grid(True)

axes[1, 0].plot(step3[0], step3[1], 'r-')
axes[1, 0].set_title('4. After U (rotation)')
axes[1, 0].set_aspect('equal')
axes[1, 0].grid(True)

axes[1, 1].plot(final[0], final[1], 'k-', linewidth=2)
axes[1, 1].set_title('5. Final: A @ circle')
axes[1, 1].set_aspect('equal')
axes[1, 1].grid(True)

# Verification
axes[1, 2].plot(step3[0], step3[1], 'r-', alpha=0.5, label='SVD steps')
axes[1, 2].plot(final[0], final[1], 'k--', label='Direct')
axes[1, 2].set_title('Verification')
axes[1, 2].set_aspect('equal')
axes[1, 2].grid(True)
axes[1, 2].legend()

plt.tight_layout()
plt.show()
```

### 3.3 Relationship Between SVD and Eigendecomposition

SVD can be obtained from eigendecomposition of $A^TA$ and $AA^T$:

$$A^TA = V\Sigma^T\Sigma V^T = V\Sigma^2 V^T$$
$$AA^T = U\Sigma\Sigma^T U^T = U\Sigma^2 U^T$$

That is:
- $V$ is the eigenvectors of $A^TA$
- $U$ is the eigenvectors of $AA^T$
- $\sigma_i^2$ are the eigenvalues of $A^TA$ (or $AA^T$)

```python
# Relationship between SVD and eigendecomposition
A = np.array([[3, 1, 1],
              [-1, 3, 1]])

# Method 1: direct SVD
U_svd, sigma_svd, VT_svd = np.linalg.svd(A)

# Method 2: eigendecomposition of A^T A
ATA = A.T @ A
eigenvalues_ATA, V_eigen = np.linalg.eigh(ATA)
# Sort eigenvalues in descending order
idx = eigenvalues_ATA.argsort()[::-1]
eigenvalues_ATA = eigenvalues_ATA[idx]
V_eigen = V_eigen[:, idx]

sigma_from_eigen = np.sqrt(eigenvalues_ATA)

print(f"Singular values (SVD): {sigma_svd}")
print(f"sqrt(eigenvalues of A^T A): {sigma_from_eigen}")
print(f"V from SVD:\n{VT_svd.T}")
print(f"V from eigendecomposition:\n{V_eigen}")
```

### 3.4 Low-Rank Approximation

**Eckart-Young-Mirsky theorem**: The best rank $k$ approximation in the Frobenius norm sense is:

$$A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T = U_k \Sigma_k V_k^T$$

```python
# Low-rank approximation example: image compression
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

# Load sample image
china = load_sample_image("china.jpg")
# Convert to grayscale
china_gray = china.mean(axis=2)

# SVD
U, sigma, VT = np.linalg.svd(china_gray, full_matrices=False)

# Approximate with various ranks
ranks = [5, 20, 50, 100]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

axes[0].imshow(china_gray, cmap='gray')
axes[0].set_title(f'Original (rank={np.linalg.matrix_rank(china_gray)})')
axes[0].axis('off')

for i, k in enumerate(ranks, 1):
    # Rank k approximation
    A_k = U[:, :k] @ np.diag(sigma[:k]) @ VT[:k, :]

    axes[i].imshow(A_k, cmap='gray')

    # Compute compression ratio
    original_size = china_gray.size
    compressed_size = k * (U.shape[0] + VT.shape[1] + 1)
    compression_ratio = original_size / compressed_size

    axes[i].set_title(f'Rank {k} (compression: {compression_ratio:.1f}x)')
    axes[i].axis('off')

# Singular value distribution
axes[-1].plot(sigma, 'b-')
axes[-1].set_xlabel('Index')
axes[-1].set_ylabel('Singular value')
axes[-1].set_title('Singular value spectrum')
axes[-1].set_yscale('log')
axes[-1].grid(True)

plt.tight_layout()
plt.show()
```

## 4. PCA: From SVD to Principal Component Analysis

### 4.1 Covariance Matrix

For a data matrix $X$ ($n$ samples × $d$ features), the covariance matrix is:

$$C = \frac{1}{n-1}X^TX \quad \text{(assuming data is centered)}$$

The covariance matrix is symmetric and positive semidefinite.

```python
# Covariance matrix
np.random.seed(42)
X = np.random.randn(100, 3)

# Center the data
X_centered = X - X.mean(axis=0)

# Covariance matrix (two methods)
cov_manual = (X_centered.T @ X_centered) / (len(X) - 1)
cov_numpy = np.cov(X.T)

print(f"Manual covariance:\n{cov_manual}")
print(f"NumPy covariance:\n{cov_numpy}")
print(f"Equal? {np.allclose(cov_manual, cov_numpy)}")

# Check symmetry
print(f"Is symmetric? {np.allclose(cov_manual, cov_manual.T)}")
```

### 4.2 Mathematical Derivation of PCA

**Goal**: Find the direction (principal component) that maximizes data variance

The first principal component $\mathbf{w}_1$ solves:

$$\max_{\mathbf{w}} \mathbf{w}^T C \mathbf{w} \quad \text{subject to } \|\mathbf{w}\| = 1$$

**Solution**: The eigenvector corresponding to the largest eigenvalue of covariance matrix $C$

The $k$-th principal component is the direction orthogonal to the previous $k-1$ components that maximizes variance.

```python
# Manual PCA implementation
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

# 1. Center the data
X_centered = X - X.mean(axis=0)

# 2. Covariance matrix
cov = np.cov(X_centered.T)

# 3. Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# 4. Sort in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Eigenvalues (variances): {eigenvalues}")
print(f"Principal components (eigenvectors):\n{eigenvectors}")

# 5. Transform (project onto first 2 principal components)
k = 2
X_pca = X_centered @ eigenvectors[:, :k]

print(f"Transformed data shape: {X_pca.shape}")
```

### 4.3 PCA Using SVD

SVD can be used without explicitly computing the covariance matrix:

$$X = U\Sigma V^T \quad \Rightarrow \quad X^TX = V\Sigma^2 V^T$$

Therefore, the columns of $V$ are the principal components, and $\sigma_i^2/(n-1)$ is the variance.

```python
# PCA using SVD
X_centered = X - X.mean(axis=0)

# SVD
U, sigma, VT = np.linalg.svd(X_centered, full_matrices=False)

# Principal components = column vectors of V
principal_components = VT.T
print(f"Principal components (from SVD):\n{principal_components}")

# Explained variance
explained_variance = (sigma**2) / (len(X) - 1)
print(f"Explained variance: {explained_variance}")

# Variance ratio
explained_variance_ratio = explained_variance / explained_variance.sum()
print(f"Explained variance ratio: {explained_variance_ratio}")

# Transform
X_pca_svd = X_centered @ principal_components[:, :2]
print(f"Same as eigendecomposition? {np.allclose(np.abs(X_pca), np.abs(X_pca_svd))}")
```

### 4.4 PCA Visualization and Interpretation

```python
from sklearn.decomposition import PCA as SklearnPCA

# scikit-learn PCA
pca = SklearnPCA(n_components=2)
X_sklearn = pca.fit_transform(X)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Transformed data
for i, target_name in enumerate(iris.target_names):
    mask = iris.target == i
    axes[0].scatter(X_sklearn[mask, 0], X_sklearn[mask, 1],
                   label=target_name, alpha=0.7)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
axes[0].set_title('PCA of Iris Dataset')
axes[0].legend()
axes[0].grid(True)

# Explained variance
axes[1].bar(range(1, len(pca.explained_variance_ratio_)+1),
           pca.explained_variance_ratio_)
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Explained Variance Ratio')
axes[1].set_title('Scree Plot')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Principal component loadings (correlation with original features)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
print("\nFeature loadings:")
for i, feature in enumerate(iris.feature_names):
    print(f"{feature:20s}: PC1={loadings[i,0]:6.3f}, PC2={loadings[i,1]:6.3f}")
```

### 4.5 Dimensionality Reduction and Reconstruction

PCA can compress data to lower dimensions and reconstruct it.

```python
# Dimensionality reduction and reconstruction
pca_full = SklearnPCA(n_components=4)
X_transformed = pca_full.fit_transform(X)

# Reconstruct with various numbers of components
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, n_components in enumerate([1, 2, 3, 4]):
    # Use only the first n components
    X_reduced = X_transformed[:, :n_components]

    # Reconstruction
    X_reconstructed = pca_full.inverse_transform(
        np.column_stack([X_reduced,
                        np.zeros((len(X), 4-n_components))])
    )

    # Reconstruction error
    reconstruction_error = np.mean((X - X_reconstructed)**2)
    explained_var = pca_full.explained_variance_ratio_[:n_components].sum()

    # Visualize first two features only
    axes[idx].scatter(X[:, 0], X[:, 1], alpha=0.3, label='Original')
    axes[idx].scatter(X_reconstructed[:, 0], X_reconstructed[:, 1],
                     alpha=0.3, label='Reconstructed')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')
    axes[idx].set_title(f'{n_components} PCs (Var: {explained_var:.2%}, MSE: {reconstruction_error:.3f})')
    axes[idx].legend()
    axes[idx].grid(True)

plt.tight_layout()
plt.show()
```

## 5. Other Matrix Decompositions

### 5.1 LU Decomposition

Decompose square matrix $A$ into lower triangular $L$ and upper triangular $U$:

$$A = LU$$

**Uses**: Solving systems of equations $Ax = b$, computing determinants

```python
from scipy.linalg import lu

# LU decomposition
A = np.array([[2, 1, 1],
              [4, -6, 0],
              [-2, 7, 2]])

P, L, U = lu(A)

print(f"P (permutation matrix):\n{P}")
print(f"L (lower triangular):\n{L}")
print(f"U (upper triangular):\n{U}")

# Reconstruction
A_reconstructed = P @ L @ U
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed)}")

# Solving equations using LU decomposition
b = np.array([4, 2, 6])

# Convert Ax = b to LUx = b
# 1. Solve Ly = Pb (forward substitution)
# 2. Solve Ux = y (back substitution)

from scipy.linalg import solve_triangular

y = solve_triangular(L, P @ b, lower=True)
x = solve_triangular(U, y, lower=False)

print(f"Solution: {x}")
print(f"Verification: Ax = {A @ x}")
```

### 5.2 QR Decomposition

Decompose matrix $A$ into orthogonal matrix $Q$ and upper triangular matrix $R$:

$$A = QR$$

**Uses**: Least squares problems, Gram-Schmidt orthogonalization

```python
# QR decomposition
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]])

Q, R = np.linalg.qr(A)

print(f"Q (orthogonal):\n{Q}")
print(f"R (upper triangular):\n{R}")

# Check orthogonality
print(f"Q^T Q:\n{Q.T @ Q}")

# Reconstruction
A_reconstructed = Q @ R
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed)}")

# Solve least squares problem: Ax = b (overdetermined)
A_over = np.array([[1, 1],
                   [1, 2],
                   [1, 3],
                   [1, 4]])
b = np.array([2, 3, 5, 6])

Q, R = np.linalg.qr(A_over)
# R is upper triangular for only the first 2 rows
R_square = R[:2, :]
Q_reduced = Q[:, :2]

# x = R^(-1) Q^T b
x = np.linalg.solve(R_square, Q_reduced.T @ b)
print(f"Least squares solution: {x}")
```

### 5.3 Cholesky Decomposition

Decompose positive definite symmetric matrix $A$ into lower triangular $L$:

$$A = LL^T$$

**Uses**: Gaussian sampling, numerically stable computations

```python
# Cholesky decomposition
A_pd = np.array([[4, 2, 2],
                 [2, 5, 1],
                 [2, 1, 6]])

# Check positive definiteness
eigenvalues = np.linalg.eigvalsh(A_pd)
print(f"Eigenvalues: {eigenvalues}")
print(f"Is positive definite? {np.all(eigenvalues > 0)}")

# Cholesky decomposition
L = np.linalg.cholesky(A_pd)
print(f"L:\n{L}")

# Reconstruction
A_reconstructed = L @ L.T
print(f"Reconstruction error: {np.linalg.norm(A_pd - A_reconstructed)}")

# Application: multivariate Gaussian sampling
# To sample from N(μ, Σ): μ + L @ z (z ~ N(0, I))
mu = np.array([1, 2, 3])
Sigma = A_pd
L = np.linalg.cholesky(Sigma)

# Sample from standard normal distribution
n_samples = 1000
Z = np.random.randn(n_samples, 3)

# Transform
samples = mu + (L @ Z.T).T

# Verification
print(f"Sample mean: {samples.mean(axis=0)}")
print(f"True mean: {mu}")
print(f"Sample covariance:\n{np.cov(samples.T)}")
print(f"True covariance:\n{Sigma}")
```

### 5.4 Comparison of Decomposition Methods

| Decomposition | Condition | Computational Complexity | Main Uses |
|---------------|-----------|-------------------------|-----------|
| Eigendecomposition | Square matrix | $O(n^3)$ | Matrix powers, dynamical systems |
| SVD | Any matrix | $O(mn^2)$ | Low-rank approximation, PCA, recommender systems |
| LU | Square matrix | $O(n^3)$ | Systems of equations, determinants |
| QR | Any matrix | $O(mn^2)$ | Least squares, orthogonalization |
| Cholesky | Positive definite | $O(n^3)$ | Sampling, numerical stability |

## 6. Applications of Matrix Decompositions in ML

### 6.1 Recommender Systems: Matrix Factorization

Approximate user-item rating matrix $R$ with low rank:

$$R \approx UV^T$$

```python
# Simple recommender system
from sklearn.decomposition import TruncatedSVD

# User-movie rating matrix (5 users × 6 movies)
R = np.array([
    [5, 3, 0, 1, 0, 0],
    [4, 0, 0, 1, 0, 0],
    [1, 1, 0, 5, 0, 0],
    [0, 0, 0, 4, 4, 5],
    [0, 0, 5, 0, 4, 4]
])

# 0 represents missing values
mask = R > 0

# SVD (rank 2 approximation)
svd = TruncatedSVD(n_components=2)
U = svd.fit_transform(R)
VT = svd.components_

# Reconstruction
R_approx = U @ VT

print("Original ratings:")
print(R)
print("\nApproximated ratings:")
print(R_approx)

# Predict missing values
print("\nPredicted ratings for missing values:")
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i, j] == 0:
            print(f"User {i}, Movie {j}: {R_approx[i, j]:.2f}")
```

### 6.2 Image Compression

Image compression using SVD was shown earlier. SVD can be applied independently to each color channel.

### 6.3 Spectral Clustering

Clustering using eigenvectors of graph Laplacian matrix:

```python
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons

# Nonlinearly separable data
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# Spectral clustering
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                               random_state=42)
labels = spectral.fit_predict(X)

# Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('True labels')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Spectral clustering')

plt.tight_layout()
plt.show()
```

### 6.4 Feature Extraction: Kernel PCA

Combining kernel trick with PCA for nonlinear dimensionality reduction:

```python
from sklearn.decomposition import KernelPCA

# Circular data
theta = np.linspace(0, 2*np.pi, 100)
r = 1 + 0.3 * np.random.randn(100)
X_circle = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

# Linear PCA
pca_linear = SklearnPCA(n_components=1)
X_pca_linear = pca_linear.fit_transform(X_circle)

# Kernel PCA (RBF kernel)
kpca = KernelPCA(n_components=1, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X_circle)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(X_circle[:, 0], X_circle[:, 1], c=theta, cmap='viridis')
axes[0].set_title('Original data')
axes[0].set_aspect('equal')

axes[1].scatter(X_pca_linear, np.zeros_like(X_pca_linear),
               c=theta, cmap='viridis')
axes[1].set_title('Linear PCA')
axes[1].set_ylim(-0.5, 0.5)

axes[2].scatter(X_kpca, np.zeros_like(X_kpca), c=theta, cmap='viridis')
axes[2].set_title('Kernel PCA')
axes[2].set_ylim(-0.5, 0.5)

plt.tight_layout()
plt.show()
```

## Practice Problems

### Problem 1: Eigendecomposition
Calculate eigenvalues and eigenvectors of the following matrix by hand and verify with NumPy:

$$A = \begin{bmatrix} 5 & 2 \\ 2 & 2 \end{bmatrix}$$

Determine if this matrix is positive definite, and compute $A^{10}$ using eigendecomposition.

### Problem 2: SVD and Low-Rank Approximation
Generate a $4 \times 3$ matrix and compute its SVD. Find rank 1 and rank 2 approximations, and calculate the Frobenius norm error for each.

### Problem 3: PCA Implementation
For the Iris dataset:
1. Perform PCA using eigendecomposition of covariance matrix
2. Perform PCA using SVD
3. Verify that both methods give identical results
4. What is the minimum number of components for 95% cumulative explained variance?

### Problem 4: Cholesky Decomposition and Sampling
Generate 1000 samples from a 2D Gaussian distribution with covariance matrix $\Sigma = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$ and mean $\mu = [0, 0]^T$. Use Cholesky decomposition and verify that the empirical covariance of samples is close to $\Sigma$.

### Problem 5: Recommender System
Predict missing values in the following user-movie rating matrix using SVD:

$$R = \begin{bmatrix} 5 & ? & 4 & ? \\ ? & 3 & ? & 2 \\ 4 & ? & 5 & ? \\ ? & 2 & ? & 3 \end{bmatrix}$$

Use rank 2 approximation and verify that predicted ratings are in the 1-5 range.

## References

### Textbooks
1. **Strang, G.** (2016). *Introduction to Linear Algebra*. Chapter 6: Eigenvalues and Eigenvectors.
2. **Trefethen, L. N., & Bau III, D.** (1997). *Numerical Linear Algebra*. SIAM.
   - In-depth discussion of SVD and numerical stability
3. **Golub, G. H., & Van Loan, C. F.** (2013). *Matrix Computations* (4th ed.). Johns Hopkins.
   - The bible of matrix decomposition algorithms

### Papers
1. **Eckart, C., & Young, G.** (1936). "The approximation of one matrix by another of lower rank." *Psychometrika*.
2. **Jolliffe, I. T.** (2002). *Principal Component Analysis* (2nd ed.). Springer.

### Online Resources
1. **3Blue1Brown - Eigenvectors and eigenvalues**: https://www.youtube.com/watch?v=PFDu9oVAE-g
2. **Stanford CS229 - Linear Algebra Review**: http://cs229.stanford.edu/section/cs229-linalg.pdf
3. **scikit-learn PCA Guide**: https://scikit-learn.org/stable/modules/decomposition.html

### Practice Tools
1. **NumPy Linear Algebra**: https://numpy.org/doc/stable/reference/routines.linalg.html
2. **SciPy Linear Algebra**: https://docs.scipy.org/doc/scipy/reference/linalg.html

---

**Next lesson**: [03. Matrix Calculus](03_Matrix_Calculus.md) to learn the mathematical foundation of backpropagation.
