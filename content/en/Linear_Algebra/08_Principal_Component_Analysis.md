# Lesson 8: Principal Component Analysis

## Learning Objectives

- Derive PCA from both the eigendecomposition and SVD perspectives
- Explain what "variance explained" means and use scree plots to choose the number of components
- Implement PCA from scratch and compare with scikit-learn
- Apply PCA to real-world dimensionality reduction tasks
- Understand the limitations and assumptions of PCA

---

## 1. Motivation

### 1.1 The Problem of High Dimensionality

Many datasets have far more features than needed. Redundant features waste storage, slow computation, and can degrade model performance (the "curse of dimensionality"). **Dimensionality reduction** finds a lower-dimensional representation that retains as much useful information as possible.

### 1.2 PCA in One Sentence

PCA finds the directions of maximum variance in the data and projects the data onto those directions.

### 1.3 Assumptions

PCA assumes that:
- The most important structure in the data lies along directions of **highest variance**
- The data is centered (mean-subtracted)
- Linear projections are sufficient to capture the structure

---

## 2. PCA via Eigendecomposition

### 2.1 The Covariance Matrix

Given a centered data matrix $X \in \mathbb{R}^{n \times d}$ (where $n$ is the number of samples and $d$ is the number of features), the **covariance matrix** is:

$$C = \frac{1}{n-1} X^T X \in \mathbb{R}^{d \times d}$$

$C$ is symmetric and positive semi-definite.

### 2.2 Derivation

PCA seeks a unit vector $\mathbf{w}$ that maximizes the variance of the projected data:

$$\max_{\|\mathbf{w}\| = 1} \mathrm{Var}(X\mathbf{w}) = \max_{\|\mathbf{w}\| = 1} \mathbf{w}^T C \mathbf{w}$$

By the spectral theorem, the solution is the eigenvector of $C$ corresponding to the **largest eigenvalue**. The $k$-th principal component is the eigenvector corresponding to the $k$-th largest eigenvalue.

### 2.3 Algorithm

1. **Center** the data: $X \leftarrow X - \bar{X}$
2. Compute the covariance matrix: $C = \frac{1}{n-1} X^T X$
3. Compute the eigendecomposition: $C = Q \Lambda Q^T$
4. Sort eigenvalues in decreasing order
5. Select the top $k$ eigenvectors as principal components
6. Project: $Z = X Q_k$ where $Q_k$ is the matrix of top $k$ eigenvectors

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate 2D data with clear principal axes
mean = [2, 3]
cov = [[3, 2],
       [2, 2]]
X = np.random.multivariate_normal(mean, cov, 200)

# Step 1: Center the data
X_centered = X - X.mean(axis=0)

# Step 2: Covariance matrix
C = np.cov(X_centered, rowvar=False)
print(f"Covariance matrix:\n{np.round(C, 4)}")

# Step 3: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(C)

# Step 4: Sort in decreasing order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{np.round(eigenvectors, 4)}")

# Step 5: Project onto first principal component
Z = X_centered @ eigenvectors

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.4, s=10)
for i in range(2):
    ax1.arrow(0, 0, eigenvectors[0, i] * eigenvalues[i]**0.5 * 2,
              eigenvectors[1, i] * eigenvalues[i]**0.5 * 2,
              head_width=0.15, color=['red', 'blue'][i], linewidth=2,
              label=f'PC{i+1} (var={eigenvalues[i]:.2f})')
ax1.set_xlabel('Feature 1'); ax1.set_ylabel('Feature 2')
ax1.set_title('Original Data with Principal Components')
ax1.legend(); ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)

ax2.scatter(Z[:, 0], Z[:, 1], alpha=0.4, s=10)
ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')
ax2.set_title('Data in Principal Component Space')
ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 3. PCA via SVD

### 3.1 Connection to SVD

For the centered data matrix $X \in \mathbb{R}^{n \times d}$ with SVD $X = U\Sigma V^T$:

$$C = \frac{1}{n-1} X^T X = V \frac{\Sigma^2}{n-1} V^T$$

Therefore:
- The **right singular vectors** $V$ are the principal component directions
- The eigenvalues of $C$ are $\lambda_i = \sigma_i^2 / (n-1)$
- The projected data is $Z = XV = U\Sigma$

### 3.2 Why SVD Is Preferred

The SVD approach is numerically superior because:
1. It avoids explicitly forming $X^TX$ (which can amplify floating-point errors)
2. It works directly with $X$, which may be sparse
3. It naturally handles the case $n < d$ (more features than samples)

```python
# PCA via SVD
X_centered = X - X.mean(axis=0)

U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Principal directions
V = Vt.T
print(f"Principal directions (SVD):\n{np.round(V, 4)}")
print(f"Principal directions (eig):\n{np.round(eigenvectors, 4)}")

# Eigenvalues from singular values
n = X_centered.shape[0]
eigenvalues_svd = s**2 / (n - 1)
print(f"\nEigenvalues (SVD): {eigenvalues_svd}")
print(f"Eigenvalues (eig): {eigenvalues}")
print(f"Match? {np.allclose(eigenvalues_svd, eigenvalues)}")

# Projected data
Z_svd = X_centered @ V
Z_eig = X_centered @ eigenvectors
# Note: signs may differ (eigenvectors are defined up to sign)
print(f"Projections match (up to sign)? {np.allclose(np.abs(Z_svd), np.abs(Z_eig))}")
```

---

## 4. Variance Explained

### 4.1 Definition

The **variance explained** by the $i$-th principal component is:

$$\text{Variance explained ratio}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$

The **cumulative variance explained** tells us how much total variance is captured by the first $k$ components:

$$\text{Cumulative}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}$$

### 4.2 Scree Plot

A **scree plot** displays eigenvalues (or variance ratios) against component index. The "elbow" suggests a natural cutoff for the number of components.

```python
# Higher-dimensional example
np.random.seed(42)
d = 10
n_samples = 500

# Data with some correlated features
true_factors = np.random.randn(n_samples, 3)  # 3 latent factors
mixing = np.random.randn(3, d)
X_high = true_factors @ mixing + 0.5 * np.random.randn(n_samples, d)

# PCA
X_c = X_high - X_high.mean(axis=0)
U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
var_explained = s**2 / (n_samples - 1)
var_ratio = var_explained / var_explained.sum()
cumulative = np.cumsum(var_ratio)

# Scree plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(range(1, d + 1), var_ratio, alpha=0.7, label='Individual')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Variance Explained Ratio')
ax1.set_title('Scree Plot')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, d + 1), cumulative, 'ro-', linewidth=2)
ax2.axhline(y=0.9, color='gray', linestyle='--', label='90% threshold')
ax2.axhline(y=0.95, color='gray', linestyle=':', label='95% threshold')
k_90 = np.searchsorted(cumulative, 0.9) + 1
ax2.axvline(x=k_90, color='green', linestyle='--', alpha=0.5)
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Variance Explained')
ax2.set_title(f'Cumulative Variance ({k_90} components for 90%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

for i, (vr, cum) in enumerate(zip(var_ratio, cumulative)):
    print(f"PC{i+1}: {vr:.4f} ({cum:.4f} cumulative)")
```

---

## 5. PCA Implementation from Scratch

```python
class PCA:
    """Principal Component Analysis from scratch."""

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None       # principal directions (rows)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """Fit PCA on data X (n_samples x n_features)."""
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # SVD
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Eigenvalues (variance along each PC)
        self.explained_variance_ = s**2 / (n - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()

        # Select components
        if self.n_components is None:
            self.n_components = min(n, d)
        self.components_ = Vt[:self.n_components]

        return self

    def transform(self, X):
        """Project X onto the principal components."""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def inverse_transform(self, Z):
        """Reconstruct X from the projected data Z."""
        return Z @ self.components_ + self.mean_

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

# Test with the Iris dataset
from sklearn.datasets import load_iris

iris = load_iris()
X_iris = iris.data      # (150, 4)
y_iris = iris.target

pca = PCA(n_components=2)
Z = pca.fit_transform(X_iris)

print(f"Original shape: {X_iris.shape}")
print(f"Projected shape: {Z.shape}")
print(f"Variance explained: {pca.explained_variance_ratio_[:2]}")
print(f"Total: {pca.explained_variance_ratio_[:2].sum():.4f}")

# Visualize
plt.figure(figsize=(8, 6))
for label, name in zip([0, 1, 2], iris.target_names):
    mask = y_iris == label
    plt.scatter(Z[mask, 0], Z[mask, 1], label=name, alpha=0.6, s=30)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Iris Dataset -- PCA Projection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 6. Reconstruction and Reconstruction Error

### 6.1 Projection and Reconstruction

When we project data onto $k$ components, we lose information. The **reconstruction** maps back to the original space:

$$\hat{X} = Z Q_k^T + \bar{X}$$

The **reconstruction error** is:

$$\|X - \hat{X}\|_F^2 = \sum_{i=k+1}^{d} \lambda_i$$

```python
# Reconstruction with varying number of components
pca_full = PCA()
pca_full.fit(X_iris)

errors = []
for k in range(1, 5):
    pca_k = PCA(n_components=k)
    Z_k = pca_k.fit_transform(X_iris)
    X_reconstructed = pca_k.inverse_transform(Z_k)
    error = np.linalg.norm(X_iris - X_reconstructed, 'fro')
    errors.append(error)
    print(f"k={k}: reconstruction error = {error:.4f}, "
          f"variance retained = {sum(pca_full.explained_variance_ratio_[:k]):.4f}")

plt.figure(figsize=(8, 4))
plt.bar(range(1, 5), errors, alpha=0.7)
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error (Frobenius)')
plt.title('Reconstruction Error vs Number of Components')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 7. Dimensionality Reduction Workflow

### 7.1 Complete PCA Pipeline

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA

# Step 1: Load and prepare data
np.random.seed(42)
n, d = 300, 20
X = np.random.randn(n, 5) @ np.random.randn(5, d) + np.random.randn(n, d) * 0.5

# Step 2: Standardize (important when features have different scales!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Fit PCA and examine variance
pca = SklearnPCA()
pca.fit(X_scaled)

# Step 4: Choose number of components (e.g., 95% variance)
cumulative = np.cumsum(pca.explained_variance_ratio_)
k = np.searchsorted(cumulative, 0.95) + 1
print(f"Components for 95% variance: {k}")

# Step 5: Transform
pca_final = SklearnPCA(n_components=k)
Z = pca_final.fit_transform(X_scaled)

print(f"Original: {X.shape[1]} features")
print(f"Reduced:  {Z.shape[1]} features")
print(f"Variance retained: {pca_final.explained_variance_ratio_.sum():.4f}")

# Step 6: Use Z for downstream tasks (classification, clustering, etc.)
```

### 7.2 When to Standardize

- **Always standardize** when features have different units or vastly different scales
- If features are already on the same scale (e.g., pixel values), standardization is optional
- PCA on unscaled data is dominated by high-variance features regardless of their importance

```python
# Demonstration of standardization effect
X_mixed = np.column_stack([
    np.random.randn(200) * 1,        # small scale
    np.random.randn(200) * 1000,     # large scale
    np.random.randn(200) * 1,        # small scale
])

# Without standardization: PC1 dominated by feature 2
pca_no_scale = SklearnPCA()
pca_no_scale.fit(X_mixed)
print("Without standardization:")
print(f"  Variance ratio: {np.round(pca_no_scale.explained_variance_ratio_, 6)}")
print(f"  PC1 loadings: {np.round(pca_no_scale.components_[0], 4)}")

# With standardization: balanced
X_std = StandardScaler().fit_transform(X_mixed)
pca_scaled = SklearnPCA()
pca_scaled.fit(X_std)
print("With standardization:")
print(f"  Variance ratio: {np.round(pca_scaled.explained_variance_ratio_, 4)}")
print(f"  PC1 loadings: {np.round(pca_scaled.components_[0], 4)}")
```

---

## 8. Comparison with scikit-learn

```python
from sklearn.decomposition import PCA as SklearnPCA

# Our implementation
pca_ours = PCA(n_components=2)
Z_ours = pca_ours.fit_transform(X_iris)

# scikit-learn
pca_sk = SklearnPCA(n_components=2)
Z_sk = pca_sk.fit_transform(X_iris)

print("Variance explained (ours):    ", np.round(pca_ours.explained_variance_ratio_[:2], 6))
print("Variance explained (sklearn): ", np.round(pca_sk.explained_variance_ratio_, 6))

# Projections should match (up to sign flips)
sign_match = np.sign(Z_ours[0]) * np.sign(Z_sk[0])
Z_ours_aligned = Z_ours * sign_match
print(f"Projections match (after sign alignment)? {np.allclose(Z_ours_aligned, Z_sk, atol=1e-10)}")
```

---

## 9. Limitations of PCA

| Limitation | Description |
|-----------|-------------|
| Linear only | Cannot capture non-linear structure (use kernel PCA, t-SNE, UMAP instead) |
| Variance bias | Assumes variance = importance, which is not always true |
| Sensitivity to outliers | Outliers disproportionately affect the covariance matrix |
| Interpretability | Principal components are linear combinations of all features, hard to interpret |
| Scale dependence | Results depend on feature scaling |

```python
# Example: PCA fails on non-linear structure
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)

pca_moon = SklearnPCA(n_components=1)
Z_moon = pca_moon.fit_transform(X_moons)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], label='Class 0', alpha=0.6)
ax1.scatter(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], label='Class 1', alpha=0.6)
ax1.set_title('Original 2D Data (Two Moons)')
ax1.legend(); ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)

ax2.scatter(Z_moon[y_moons == 0], np.zeros(sum(y_moons == 0)), label='Class 0', alpha=0.6)
ax2.scatter(Z_moon[y_moons == 1], np.zeros(sum(y_moons == 1)), label='Class 1', alpha=0.6)
ax2.set_title('PCA Projection to 1D\n(Classes overlap!)')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 10. Summary

| Concept | Description |
|---------|-------------|
| PCA goal | Find directions of maximum variance |
| Covariance matrix | $C = \frac{1}{n-1}X^TX$ |
| PCA via eigendecomp | Eigenvectors of $C$ are principal directions |
| PCA via SVD | Right singular vectors of centered $X$ |
| Variance explained | $\lambda_i / \sum \lambda_j$ |
| Scree plot | Eigenvalues vs component index |
| Reconstruction | $\hat{X} = ZQ_k^T + \bar{X}$ |
| NumPy | `np.linalg.svd()` or `np.linalg.eigh()` |
| scikit-learn | `sklearn.decomposition.PCA` |

---

## Exercises

### Exercise 1: PCA from Scratch

Implement PCA from scratch using only NumPy (both the eigendecomposition and SVD approaches). Apply both to the Iris dataset and verify they produce the same results.

### Exercise 2: Scree Plot Analysis

Generate a dataset with known intrinsic dimensionality (e.g., 5 latent factors in 50-dimensional space). Perform PCA and create a scree plot. Can you recover the true dimensionality from the plot?

### Exercise 3: Face Eigenfaces

Download a face dataset (e.g., Olivetti faces via `sklearn.datasets.fetch_olivetti_faces`). Apply PCA and visualize the top 10 principal components as "eigenfaces". How many components are needed to capture 90% of the variance?

### Exercise 4: Effect of Standardization

Create a dataset where one feature has much larger variance than others. Run PCA with and without standardization. Compare and explain the results.

### Exercise 5: Reconstruction Quality

Using the Iris dataset, for $k = 1, 2, 3, 4$:
1. Project to $k$ dimensions using PCA
2. Reconstruct back to 4D
3. Compute the mean squared reconstruction error
4. Plot the error as a function of $k$

---

[<< Previous: Lesson 7 - Singular Value Decomposition](07_Singular_Value_Decomposition.md) | [Overview](00_Overview.md) | [Next: Lesson 9 - Orthogonality and Projections >>](09_Orthogonality_and_Projections.md)

**License**: CC BY-NC 4.0
