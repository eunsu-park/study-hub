# Lesson 16: Linear Algebra in Machine Learning

[Previous: Lesson 15](./15_Tensors_and_Multilinear_Algebra.md) | [Overview](./00_Overview.md) | [Next: Lesson 17](./17_Linear_Algebra_in_Deep_Learning.md)

---

## Learning Objectives

- Derive the normal equation for linear regression and understand its geometric meaning
- Explain kernel methods and feature spaces through the lens of linear algebra
- Compute and interpret covariance matrices and their eigenstructure
- Apply the Mahalanobis distance for multivariate statistical analysis
- Understand embeddings as linear maps from high-dimensional to low-dimensional spaces
- Connect fundamental ML algorithms to their underlying linear algebra

---

## 1. Linear Regression: The Normal Equation

### 1.1 Problem Formulation

Given a data matrix $X \in \mathbb{R}^{n \times d}$ (n samples, d features) and target vector $\mathbf{y} \in \mathbb{R}^n$, linear regression finds the weight vector $\mathbf{w} \in \mathbb{R}^d$ that minimizes:

$$\min_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|^2 = \min_{\mathbf{w}} (X\mathbf{w} - \mathbf{y})^T(X\mathbf{w} - \mathbf{y})$$

### 1.2 Derivation of the Normal Equation

Setting the gradient to zero:

$$\nabla_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|^2 = 2X^T(X\mathbf{w} - \mathbf{y}) = \mathbf{0}$$

yields the **normal equation**:

$$X^T X \mathbf{w} = X^T \mathbf{y} \quad \Rightarrow \quad \mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$$

The matrix $X^+ = (X^T X)^{-1} X^T$ is the **Moore-Penrose pseudoinverse** of $X$.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Generate synthetic data
np.random.seed(42)
n_samples, n_features = 200, 3
X, y = make_regression(n_samples=n_samples, n_features=n_features,
                       noise=10, random_state=42)

# Add bias column
X_bias = np.column_stack([np.ones(n_samples), X])

# Method 1: Normal equation (direct)
w_normal = np.linalg.solve(X_bias.T @ X_bias, X_bias.T @ y)
print(f"Normal equation weights: {w_normal}")

# Method 2: Pseudoinverse
w_pinv = np.linalg.pinv(X_bias) @ y
print(f"Pseudoinverse weights:   {w_pinv}")

# Method 3: QR decomposition (numerically better)
Q, R = np.linalg.qr(X_bias)
w_qr = np.linalg.solve(R, Q.T @ y)
print(f"QR decomposition weights: {w_qr}")

# Method 4: SVD (most robust)
U, S, Vt = np.linalg.svd(X_bias, full_matrices=False)
w_svd = Vt.T @ np.diag(1.0 / S) @ U.T @ y
print(f"SVD weights: {w_svd}")

# All methods should agree
print(f"\nAll close? {np.allclose(w_normal, w_pinv) and np.allclose(w_normal, w_qr)}")

# Prediction and residual
y_pred = X_bias @ w_normal
residual = y - y_pred
print(f"Residual norm: {np.linalg.norm(residual):.4f}")
print(f"R-squared: {1 - np.sum(residual**2) / np.sum((y - y.mean())**2):.4f}")
```

### 1.3 Geometric Interpretation: Projection

The predicted values $\hat{\mathbf{y}} = X\mathbf{w}$ are the **orthogonal projection** of $\mathbf{y}$ onto the column space of $X$. The residual $\mathbf{y} - \hat{\mathbf{y}}$ is orthogonal to every column of $X$:

$$X^T(\mathbf{y} - X\mathbf{w}) = \mathbf{0}$$

```python
# Geometric visualization (2D example)
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 2.5 * x + 3 + np.random.randn(50) * 3

X_2d = np.column_stack([np.ones_like(x), x])
w = np.linalg.lstsq(X_2d, y, rcond=None)[0]
y_pred = X_2d @ w
residuals = y - y_pred

# Verify orthogonality: X^T r = 0
print(f"X^T * residual = {X_2d.T @ residuals}")
print(f"(Should be near zero)")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(x, y, alpha=0.6, label='Data')
ax1.plot(x, y_pred, 'r-', linewidth=2, label='Fit')
for i in range(0, len(x), 5):
    ax1.plot([x[i], x[i]], [y[i], y_pred[i]], 'g--', alpha=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression as Projection')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Residual')
ax2.set_ylabel('Count')
ax2.set_title('Residual Distribution')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 1.4 Ridge Regression: Regularized Normal Equation

When $X^TX$ is ill-conditioned (nearly singular), adding $\ell_2$ regularization stabilizes the solution:

$$\mathbf{w}_{\text{ridge}} = (X^TX + \alpha I)^{-1} X^T \mathbf{y}$$

The regularization parameter $\alpha > 0$ shifts all eigenvalues of $X^TX$ away from zero, improving the condition number.

```python
# Ridge regression and the effect on condition number
from sklearn.linear_model import Ridge

# Create collinear features (ill-conditioned)
np.random.seed(42)
n = 100
x1 = np.random.randn(n)
x2 = x1 + 0.001 * np.random.randn(n)  # Nearly identical to x1
x3 = np.random.randn(n)
X_ill = np.column_stack([x1, x2, x3])
y = 3 * x1 + 2 * x3 + np.random.randn(n) * 0.5

print(f"Condition number of X^T X: {np.linalg.cond(X_ill.T @ X_ill):.2e}")

# OLS solution (unstable)
w_ols = np.linalg.solve(X_ill.T @ X_ill, X_ill.T @ y)
print(f"OLS weights: {w_ols}")

# Ridge solutions
for alpha in [0.01, 0.1, 1.0, 10.0]:
    XtX_reg = X_ill.T @ X_ill + alpha * np.eye(3)
    w_ridge = np.linalg.solve(XtX_reg, X_ill.T @ y)
    cond = np.linalg.cond(XtX_reg)
    print(f"alpha={alpha:5.2f}: cond={cond:.2e}, weights={w_ridge}")
```

---

## 2. Kernel Methods and Feature Spaces

### 2.1 The Kernel Trick

Many ML algorithms only access the data through inner products $\langle \mathbf{x}_i, \mathbf{x}_j \rangle$. The **kernel trick** replaces this with a kernel function $k(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$ that implicitly maps data to a high-dimensional (possibly infinite-dimensional) feature space $\phi$.

```python
from scipy.spatial.distance import cdist

# Feature map for polynomial kernel of degree 2
def phi_poly2(x):
    """Explicit feature map for degree-2 polynomial kernel in 2D."""
    # phi(x) = [x1^2, x2^2, sqrt(2)*x1*x2, sqrt(2)*x1, sqrt(2)*x2, 1]
    x1, x2 = x[:, 0], x[:, 1]
    return np.column_stack([
        x1**2, x2**2, np.sqrt(2) * x1 * x2,
        np.sqrt(2) * x1, np.sqrt(2) * x2, np.ones_like(x1)
    ])

# Kernel function: k(x, y) = (x . y + 1)^2
def polynomial_kernel(X, Y, degree=2, c=1):
    return (X @ Y.T + c) ** degree

# Verify: phi(x)^T phi(y) = k(x, y)
np.random.seed(42)
X = np.random.randn(10, 2)

# Explicit feature map
Phi = phi_poly2(X)
K_explicit = Phi @ Phi.T

# Kernel function
K_kernel = polynomial_kernel(X, X, degree=2, c=1)

print(f"Feature map dimension: {Phi.shape[1]}")
print(f"K_explicit == K_kernel? {np.allclose(K_explicit, K_kernel)}")

# RBF kernel: infinite-dimensional feature space
def rbf_kernel(X, Y, gamma=1.0):
    sq_dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-gamma * sq_dists)

K_rbf = rbf_kernel(X, X, gamma=0.5)
eigenvalues = np.linalg.eigvalsh(K_rbf)
print(f"\nRBF kernel eigenvalues (all non-negative):")
print(f"  min: {eigenvalues.min():.6e}, max: {eigenvalues.max():.4f}")
```

### 2.2 Kernel Ridge Regression

In the dual formulation, kernel ridge regression solves:

$$\boldsymbol{\alpha} = (K + \lambda I)^{-1} \mathbf{y}$$

where $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$. Prediction: $\hat{y}(\mathbf{x}) = \sum_i \alpha_i k(\mathbf{x}_i, \mathbf{x})$.

```python
from sklearn.datasets import make_moons

# Non-linear classification with kernel ridge regression
X_train, y_train = make_moons(n_samples=200, noise=0.15, random_state=42)
y_train = 2 * y_train - 1  # Convert to +1/-1

# Kernel matrix
gamma = 2.0
K = rbf_kernel(X_train, X_train, gamma=gamma)

# Solve dual problem
lam = 0.01
alpha = np.linalg.solve(K + lam * np.eye(len(X_train)), y_train)

# Predict on grid
xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
X_grid = np.column_stack([xx.ravel(), yy.ravel()])
K_grid = rbf_kernel(X_grid, X_train, gamma=gamma)
y_pred_grid = K_grid @ alpha

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, y_pred_grid.reshape(xx.shape), levels=50, cmap='RdBu', alpha=0.7)
plt.contour(xx, yy, y_pred_grid.reshape(xx.shape), levels=[0], colors='black', linewidths=2)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            c='blue', edgecolors='k', label='+1')
plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1],
            c='red', edgecolors='k', label='-1')
plt.title('Kernel Ridge Regression (RBF kernel)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 2.3 Mercer's Theorem

A kernel $k(x, y)$ corresponds to a valid feature map if and only if the kernel matrix is **positive semidefinite** for any finite set of points (Mercer's condition). This connects kernel methods directly to positive definite matrices (Lesson 11).

```python
# Verify Mercer's condition for common kernels
np.random.seed(42)
X = np.random.randn(50, 3)

kernels = {
    "Linear": lambda X, Y: X @ Y.T,
    "Polynomial (d=3)": lambda X, Y: (X @ Y.T + 1) ** 3,
    "RBF": lambda X, Y: rbf_kernel(X, Y, gamma=0.5),
    "Laplacian": lambda X, Y: np.exp(-0.5 * cdist(X, Y, 'cityblock')),
}

for name, kernel_fn in kernels.items():
    K = kernel_fn(X, X)
    eigenvalues = np.linalg.eigvalsh(K)
    is_psd = np.all(eigenvalues >= -1e-10)
    print(f"{name:20s}: PSD={is_psd}, min_eig={eigenvalues.min():.6e}")
```

---

## 3. Covariance Matrices

### 3.1 Estimation and Properties

The covariance matrix captures the second-order structure of a dataset. For centered data $X$ with $n$ samples and $d$ features:

$$\Sigma = \frac{1}{n-1} X^T X$$

```python
from sklearn.datasets import load_iris

# Load and center data
iris = load_iris()
X = iris.data
X_centered = X - X.mean(axis=0)

# Covariance matrix
cov = np.cov(X.T)
print(f"Covariance matrix:\n{np.round(cov, 3)}")
print(f"Shape: {cov.shape}")

# Properties
print(f"\nSymmetric: {np.allclose(cov, cov.T)}")
eigenvalues = np.linalg.eigvalsh(cov)
print(f"Eigenvalues: {eigenvalues}")
print(f"All non-negative (PSD): {np.all(eigenvalues >= -1e-10)}")
print(f"Determinant (generalized variance): {np.linalg.det(cov):.6f}")
print(f"Trace (total variance): {np.trace(cov):.4f}")
print(f"Sum of individual variances: {np.var(X, axis=0, ddof=1).sum():.4f}")
```

### 3.2 Eigenstructure of the Covariance Matrix

The eigenvectors of the covariance matrix are the **principal components** (directions of maximum variance), and the eigenvalues are the **variances** along those directions.

```python
# Eigendecomposition of covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Explained variance
explained_variance_ratio = eigenvalues / eigenvalues.sum()
cumulative = np.cumsum(explained_variance_ratio)

print("Principal Components Analysis:")
for i in range(len(eigenvalues)):
    print(f"  PC{i+1}: variance={eigenvalues[i]:.4f}, "
          f"explained={explained_variance_ratio[i]:.2%}, "
          f"cumulative={cumulative[i]:.2%}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot of first two PCs
X_pca = X_centered @ eigenvectors[:, :2]
for i, target_name in enumerate(iris.target_names):
    mask = iris.target == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=target_name, alpha=0.7)
axes[0].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%})')
axes[0].set_title('PCA of Iris Dataset')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Scree plot
axes[1].bar(range(1, 5), explained_variance_ratio, alpha=0.7, label='Individual')
axes[1].plot(range(1, 5), cumulative, 'ro-', label='Cumulative')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Variance Explained')
axes[1].set_title('Scree Plot')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.3 Whitening Transform

**Whitening** (or sphering) transforms data so that the covariance is the identity matrix. If $\Sigma = Q \Lambda Q^T$, the whitening transform is:

$$\mathbf{z} = \Lambda^{-1/2} Q^T (\mathbf{x} - \boldsymbol{\mu})$$

```python
# Whitening transform
def whiten(X):
    """Whiten data so that covariance becomes identity."""
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Whitening transform
    W = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues))
    return X_centered @ W, W

X_white, W = whiten(X)

# Verify
cov_white = np.cov(X_white.T)
print(f"Whitened covariance:\n{np.round(cov_white, 4)}")
print(f"Close to identity: {np.allclose(cov_white, np.eye(4), atol=0.1)}")
```

---

## 4. Mahalanobis Distance

### 4.1 Definition

The **Mahalanobis distance** accounts for correlations between variables by using the inverse covariance matrix:

$$d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

When $\Sigma = I$, this reduces to the Euclidean distance. The Mahalanobis distance transforms the space so that the data has spherical symmetry.

```python
from scipy.spatial.distance import mahalanobis

# Generate correlated 2D data
np.random.seed(42)
mean = np.array([2, 3])
cov_2d = np.array([[2, 1.5],
                   [1.5, 3]])
X_2d = np.random.multivariate_normal(mean, cov_2d, 500)

# Compare Euclidean and Mahalanobis distances
test_points = np.array([
    [4, 3],    # Along major axis (far in Euclidean)
    [2, 6],    # Along minor axis (closer in Euclidean)
    [0, 0],    # Off-center
])

cov_inv = np.linalg.inv(cov_2d)

print("Point comparison:")
for p in test_points:
    d_eucl = np.linalg.norm(p - mean)
    d_maha = mahalanobis(p, mean, cov_inv)
    print(f"  {p}: Euclidean={d_eucl:.3f}, Mahalanobis={d_maha:.3f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Euclidean distance contours
for ax, title, metric in zip(axes, ['Euclidean Distance', 'Mahalanobis Distance'],
                              ['euclidean', 'mahalanobis']):
    ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.3, s=10)

    # Draw contours
    xx, yy = np.meshgrid(np.linspace(-3, 7, 200), np.linspace(-3, 9, 200))
    points = np.column_stack([xx.ravel(), yy.ravel()])

    if metric == 'euclidean':
        dists = np.sqrt(np.sum((points - mean)**2, axis=1))
    else:
        dists = np.sqrt(np.sum((points - mean) @ cov_inv * (points - mean), axis=1))

    ax.contour(xx, yy, dists.reshape(xx.shape), levels=[1, 2, 3],
               colors=['green', 'orange', 'red'], linewidths=2)
    ax.plot(*mean, 'k*', markersize=15)
    for p in test_points:
        ax.plot(*p, 'rs', markersize=10)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4.2 Applications: Outlier Detection

```python
from scipy.stats import chi2

# Mahalanobis-based outlier detection
np.random.seed(42)
n = 500
X_normal = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n)
X_outliers = np.random.uniform(-5, 5, (20, 2))
X_all = np.vstack([X_normal, X_outliers])
labels = np.array([0]*n + [1]*20)  # 0=normal, 1=outlier

# Robust estimation of mean and covariance
mu = np.mean(X_normal, axis=0)
cov = np.cov(X_normal.T)
cov_inv = np.linalg.inv(cov)

# Compute Mahalanobis distances
dists = np.array([mahalanobis(x, mu, cov_inv) for x in X_all])

# Threshold based on chi-squared distribution (2 degrees of freedom)
threshold = np.sqrt(chi2.ppf(0.975, df=2))  # 97.5% quantile
detected = dists > threshold

print(f"Threshold (97.5%): {threshold:.3f}")
print(f"True outliers: {labels.sum()}")
print(f"Detected outliers: {detected.sum()}")
print(f"True positives: {(detected & (labels == 1)).sum()}")
print(f"False positives: {(detected & (labels == 0)).sum()}")
```

---

## 5. Embeddings and Dimensionality Reduction

### 5.1 Linear Embeddings

An **embedding** maps high-dimensional data to a lower-dimensional space while preserving important structure. Linear embeddings are simply matrix multiplications:

$$\mathbf{z} = W^T \mathbf{x}, \quad W \in \mathbb{R}^{d \times k}, \quad k \ll d$$

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# High-dimensional data: 20 features
np.random.seed(42)
n = 500
# Three clusters in a 3D subspace, embedded in 20D
W_true = np.random.randn(20, 3)
means_3d = [np.array([3, 0, 0]), np.array([0, 3, 0]), np.array([-3, -3, 0])]
clusters = []
labels = []
for i, mu in enumerate(means_3d):
    cluster = np.random.randn(n // 3, 3) * 0.5 + mu
    clusters.append(cluster @ W_true.T)  # Project to 20D
    labels.extend([i] * (n // 3))
X_high = np.vstack(clusters)
labels = np.array(labels)

print(f"Original data shape: {X_high.shape}")

# PCA embedding (linear)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_high)

# Compare: random projection (also linear)
W_random = np.random.randn(20, 2) / np.sqrt(2)
X_random = X_high @ W_random

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, X_emb, title in zip(axes,
                              [X_pca, X_random],
                              ['PCA (optimal linear)', 'Random Projection']):
    for i in range(3):
        mask = labels == i
        ax.scatter(X_emb[mask, 0], X_emb[mask, 1], alpha=0.6, label=f'Cluster {i}')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Explained variance
print(f"\nPCA explained variance: {pca.explained_variance_ratio_}")
print(f"PCA total explained: {pca.explained_variance_ratio_.sum():.2%}")
```

### 5.2 Word Embeddings as Linear Algebra

Word embeddings (Word2Vec, GloVe) map words to vectors in $\mathbb{R}^d$ such that semantic relationships are captured by linear operations:

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

```python
# Simulated word embeddings (in practice, use pre-trained vectors)
np.random.seed(42)
d = 50  # Embedding dimension

# Create word vectors with structured relationships
words = {
    'king': np.random.randn(d),
    'queen': None,  # Derived
    'man': np.random.randn(d),
    'woman': None,   # Derived
    'prince': None,
    'princess': None,
}

# Create gender direction
gender = np.random.randn(d) * 0.3
royalty = np.random.randn(d) * 0.3

words['woman'] = words['man'] + gender
words['queen'] = words['king'] + gender
words['prince'] = words['king'] - royalty
words['princess'] = words['queen'] - royalty

# Test analogy: king - man + woman = ?
analogy = words['king'] - words['man'] + words['woman']

# Find closest word
def cosine_similarity(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

print("Analogy: king - man + woman = ?")
for word, vec in words.items():
    sim = cosine_similarity(analogy, vec)
    print(f"  {word:10s}: cosine similarity = {sim:.4f}")
```

### 5.3 Matrix Factorization for Embeddings

Many embedding methods can be cast as matrix factorization. For example, GloVe factorizes a co-occurrence matrix, and collaborative filtering factorizes a user-item matrix.

```python
# Non-negative Matrix Factorization (NMF) for topic modeling
from sklearn.decomposition import NMF

# Simulated document-term matrix (n_docs x n_terms)
np.random.seed(42)
n_docs, n_terms = 100, 50
n_topics = 5

# Generate synthetic data from a topic model
W_true = np.random.exponential(0.3, (n_docs, n_topics))
H_true = np.random.exponential(0.3, (n_topics, n_terms))
V = W_true @ H_true + 0.01 * np.random.rand(n_docs, n_terms)

# NMF decomposition: V ~ W H
nmf = NMF(n_components=n_topics, max_iter=500, random_state=42)
W = nmf.fit_transform(V)
H = nmf.components_

print(f"Original matrix: {V.shape}")
print(f"W (doc embeddings): {W.shape}")
print(f"H (topic-term): {H.shape}")
print(f"Reconstruction error: {np.linalg.norm(V - W @ H) / np.linalg.norm(V):.4f}")

# Each row of W is a document's "embedding" in topic space
print(f"\nDocument 0 topic weights: {np.round(W[0], 3)}")
print(f"Most prominent topic: {np.argmax(W[0])}")
```

---

## 6. Linear Discriminant Analysis (LDA)

LDA finds the projection that maximizes between-class variance relative to within-class variance:

$$\mathbf{w}^* = \arg\max_{\mathbf{w}} \frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$$

This is a **generalized eigenvalue problem**: $S_B \mathbf{w} = \lambda S_W \mathbf{w}$.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA from scratch
iris = load_iris()
X, y = iris.data, iris.target
classes = np.unique(y)
n_features = X.shape[1]

# Within-class scatter matrix
S_W = np.zeros((n_features, n_features))
for c in classes:
    X_c = X[y == c]
    S_W += (X_c - X_c.mean(axis=0)).T @ (X_c - X_c.mean(axis=0))

# Between-class scatter matrix
overall_mean = X.mean(axis=0)
S_B = np.zeros((n_features, n_features))
for c in classes:
    X_c = X[y == c]
    n_c = len(X_c)
    mean_diff = (X_c.mean(axis=0) - overall_mean).reshape(-1, 1)
    S_B += n_c * mean_diff @ mean_diff.T

# Solve generalized eigenvalue problem
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W) @ S_B)
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real

# Sort by eigenvalue
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Project to 2D
W_lda = eigenvectors[:, :2]
X_lda = X @ W_lda

# Compare with sklearn
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda_sklearn = lda.fit_transform(X, y)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, X_proj, title in zip(axes,
                              [X_lda, X_lda_sklearn],
                              ['Manual LDA', 'sklearn LDA']):
    for i, name in enumerate(iris.target_names):
        mask = y == i
        ax.scatter(X_proj[mask, 0], X_proj[mask, 1], alpha=0.7, label=name)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 7. Singular Value Decomposition in ML

### 7.1 Low-Rank Approximation for Data Compression

```python
from sklearn.datasets import fetch_olivetti_faces

# Load face dataset
faces = fetch_olivetti_faces()
X_faces = faces.data  # (400 faces, 4096 pixels)
print(f"Data shape: {X_faces.shape}")

# SVD
U, S, Vt = np.linalg.svd(X_faces, full_matrices=False)

# Reconstruct with different ranks
ranks = [5, 20, 50, 100, 200]
face_idx = 0

fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(18, 3))
axes[0].imshow(X_faces[face_idx].reshape(64, 64), cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

for i, k in enumerate(ranks):
    X_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    axes[i + 1].imshow(X_approx[face_idx].reshape(64, 64), cmap='gray')
    error = np.linalg.norm(X_faces - X_approx) / np.linalg.norm(X_faces)
    axes[i + 1].set_title(f'Rank {k}\nerr={error:.3f}')
    axes[i + 1].axis('off')

plt.tight_layout()
plt.show()
```

### 7.2 Latent Semantic Analysis (LSA)

LSA applies SVD to a term-document matrix to discover latent semantic structure:

```python
# Simple LSA example
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "machine learning is a branch of artificial intelligence",
    "deep learning uses neural networks",
    "natural language processing analyzes text",
    "computer vision processes images and video",
    "reinforcement learning optimizes decisions",
    "neural networks learn from data",
    "AI systems can understand language",
    "image recognition uses deep neural networks",
]

# TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(documents).toarray()
terms = vectorizer.get_feature_names_out()

print(f"TF-IDF matrix shape: {tfidf.shape} (docs x terms)")

# SVD for LSA
U, S, Vt = np.linalg.svd(tfidf, full_matrices=False)

# Reduce to 2 dimensions
k = 2
doc_embeddings = U[:, :k] @ np.diag(S[:k])
term_embeddings = np.diag(S[:k]) @ Vt[:k, :]

# Find similar documents
from scipy.spatial.distance import cosine
print("\nDocument similarity (cosine in LSA space):")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        sim = 1 - cosine(doc_embeddings[i], doc_embeddings[j])
        if sim > 0.5:
            print(f"  Doc {i} <-> Doc {j}: {sim:.3f}")
```

---

## Exercises

### Exercise 1: Linear Regression Methods

For a dataset with 1000 samples and 50 features (some highly correlated):

1. Solve using the normal equation, QR decomposition, and SVD
2. Compare residual norms and weights
3. Add $\ell_2$ regularization with $\alpha \in \{0.01, 0.1, 1, 10, 100\}$ and plot the weight magnitude $\|\mathbf{w}\|$ vs $\alpha$

### Exercise 2: Kernel Methods

1. Implement kernel ridge regression with RBF kernel from scratch
2. Apply it to a synthetic 2D classification problem (make_circles)
3. Visualize the decision boundary for different values of $\gamma$ in the RBF kernel

### Exercise 3: Mahalanobis Outlier Detection

Generate a 3D Gaussian dataset with 500 normal points and inject 20 outliers. Implement Mahalanobis-based outlier detection and report precision, recall, and F1 score. Compare with Euclidean distance-based detection.

### Exercise 4: PCA vs LDA

For the Iris dataset:

1. Apply PCA to reduce to 2D
2. Apply LDA to reduce to 2D
3. Compare the class separability of each projection (measure using between-class / within-class variance ratio)
4. Which method better separates the classes and why?

### Exercise 5: Matrix Factorization

Create a synthetic user-item rating matrix (50 users, 100 items) with known low-rank structure (rank 5). Remove 50% of entries. Use SVD-based matrix factorization to predict the missing entries and compute RMSE.

---

[Previous: Lesson 15](./15_Tensors_and_Multilinear_Algebra.md) | [Overview](./00_Overview.md) | [Next: Lesson 17](./17_Linear_Algebra_in_Deep_Learning.md)

**License**: CC BY-NC 4.0
