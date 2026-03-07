# 16강: 머신러닝에서의 선형대수 (Linear Algebra in Machine Learning)

[이전: 15강](./15_Tensors_and_Multilinear_Algebra.md) | [개요](./00_Overview.md) | [다음: 17강](./17_Linear_Algebra_in_Deep_Learning.md)

---

## 학습 목표

- 선형 회귀의 정규방정식을 유도하고 기하학적 의미를 이해할 수 있다
- 커널 방법과 특징 공간을 선형대수의 관점에서 설명할 수 있다
- 공분산 행렬과 그 고유 구조를 계산하고 해석할 수 있다
- 다변량 통계 분석에 Mahalanobis 거리를 적용할 수 있다
- 임베딩을 고차원에서 저차원으로의 선형 사상으로 이해할 수 있다
- 핵심 ML 알고리즘과 그 기반 선형대수를 연결할 수 있다

---

## 1. 선형 회귀: 정규방정식

### 1.1 문제 정의

데이터 행렬 $X \in \mathbb{R}^{n \times d}$ (n개 샘플, d개 특징)와 목표 벡터 $\mathbf{y} \in \mathbb{R}^n$이 주어졌을 때, 선형 회귀는 다음을 최소화하는 가중치 벡터 $\mathbf{w} \in \mathbb{R}^d$를 찾습니다:

$$\min_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|^2 = \min_{\mathbf{w}} (X\mathbf{w} - \mathbf{y})^T(X\mathbf{w} - \mathbf{y})$$

### 1.2 정규방정식의 유도

그래디언트를 0으로 설정하면:

$$\nabla_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|^2 = 2X^T(X\mathbf{w} - \mathbf{y}) = \mathbf{0}$$

**정규방정식**이 도출됩니다:

$$X^T X \mathbf{w} = X^T \mathbf{y} \quad \Rightarrow \quad \mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$$

행렬 $X^+ = (X^T X)^{-1} X^T$는 $X$의 **Moore-Penrose 유사역행렬**입니다.

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

### 1.3 기하학적 해석: 투영

예측값 $\hat{\mathbf{y}} = X\mathbf{w}$는 $\mathbf{y}$를 $X$의 열공간에 **직교 투영**한 것입니다. 잔차 $\mathbf{y} - \hat{\mathbf{y}}$는 $X$의 모든 열에 대해 직교합니다:

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

### 1.4 릿지 회귀: 정규화된 정규방정식

$X^TX$가 조건 불량(거의 특이행렬)일 때, $\ell_2$ 정규화를 추가하면 해가 안정화됩니다:

$$\mathbf{w}_{\text{ridge}} = (X^TX + \alpha I)^{-1} X^T \mathbf{y}$$

정규화 매개변수 $\alpha > 0$는 $X^TX$의 모든 고유값을 0에서 멀리 이동시켜 조건수를 개선합니다.

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

## 2. 커널 방법과 특징 공간

### 2.1 커널 트릭

많은 ML 알고리즘은 데이터에 내적 $\langle \mathbf{x}_i, \mathbf{x}_j \rangle$을 통해서만 접근합니다. **커널 트릭**은 이를 커널 함수 $k(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$로 대체하여 데이터를 고차원(잠재적으로 무한 차원) 특징 공간 $\phi$로 묵시적으로 매핑합니다.

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

### 2.2 커널 릿지 회귀

쌍대 정식에서 커널 릿지 회귀는 다음을 풉니다:

$$\boldsymbol{\alpha} = (K + \lambda I)^{-1} \mathbf{y}$$

여기서 $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$입니다. 예측: $\hat{y}(\mathbf{x}) = \sum_i \alpha_i k(\mathbf{x}_i, \mathbf{x})$.

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

### 2.3 Mercer의 정리

커널 $k(x, y)$가 유효한 특징 맵에 대응하려면, 임의의 유한 점 집합에 대해 커널 행렬이 **양의 준정치 행렬**이어야 합니다(Mercer 조건). 이는 커널 방법을 양의 정치 행렬(11강)과 직접적으로 연결합니다.

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

## 3. 공분산 행렬

### 3.1 추정과 성질

공분산 행렬은 데이터셋의 2차 구조를 포착합니다. 중심화된 데이터 $X$가 $n$개의 샘플과 $d$개의 특징을 가질 때:

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

### 3.2 공분산 행렬의 고유 구조

공분산 행렬의 고유벡터는 **주성분**(최대 분산 방향)이며, 고유값은 해당 방향의 **분산**입니다.

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

### 3.3 백색화 변환

**백색화**(또는 구형화)는 공분산이 단위행렬이 되도록 데이터를 변환합니다. $\Sigma = Q \Lambda Q^T$일 때, 백색화 변환은 다음과 같습니다:

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

## 4. Mahalanobis 거리

### 4.1 정의

**Mahalanobis 거리**는 역공분산 행렬을 사용하여 변수 간 상관관계를 고려합니다:

$$d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

$\Sigma = I$일 때 유클리드 거리로 축소됩니다. Mahalanobis 거리는 데이터가 구형 대칭을 갖도록 공간을 변환합니다.

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

### 4.2 응용: 이상치 탐지

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

## 5. 임베딩과 차원 축소

### 5.1 선형 임베딩

**임베딩**은 중요한 구조를 보존하면서 고차원 데이터를 저차원 공간으로 매핑합니다. 선형 임베딩은 단순히 행렬 곱셈입니다:

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

### 5.2 단어 임베딩의 선형대수

단어 임베딩(Word2Vec, GloVe)은 단어를 $\mathbb{R}^d$의 벡터로 매핑하여 의미적 관계가 선형 연산으로 포착되도록 합니다:

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

### 5.3 임베딩을 위한 행렬 인수분해

많은 임베딩 방법은 행렬 인수분해로 표현할 수 있습니다. 예를 들어, GloVe는 동시 출현 행렬을 인수분해하고, 협업 필터링은 사용자-아이템 행렬을 인수분해합니다.

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

## 6. 선형 판별 분석 (LDA)

LDA는 클래스 간 분산 대비 클래스 내 분산을 최대화하는 투영을 찾습니다:

$$\mathbf{w}^* = \arg\max_{\mathbf{w}} \frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$$

이는 **일반화된 고유값 문제**입니다: $S_B \mathbf{w} = \lambda S_W \mathbf{w}$.

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

## 7. 머신러닝에서의 특이값 분해

### 7.1 데이터 압축을 위한 저랭크 근사

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

### 7.2 잠재 의미 분석 (LSA)

LSA는 SVD를 단어-문서 행렬에 적용하여 잠재 의미 구조를 발견합니다:

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

## 연습 문제

### 문제 1: 선형 회귀 방법 비교

1000개 샘플과 50개 특징(일부 높은 상관관계)을 가진 데이터셋에 대해:

1. 정규방정식, QR 분해, SVD를 사용하여 풀기
2. 잔차 노름과 가중치 비교
3. $\alpha \in \{0.01, 0.1, 1, 10, 100\}$으로 $\ell_2$ 정규화를 추가하고 가중치 크기 $\|\mathbf{w}\|$ 대 $\alpha$를 그래프로 나타내기

### 문제 2: 커널 방법

1. RBF 커널을 사용한 커널 릿지 회귀를 처음부터 구현하기
2. 합성 2D 분류 문제(make_circles)에 적용하기
3. RBF 커널의 다양한 $\gamma$ 값에 대해 결정 경계를 시각화하기

### 문제 3: Mahalanobis 이상치 탐지

3차원 가우시안 데이터셋(500개 정상 포인트)을 생성하고 20개의 이상치를 주입하시오. Mahalanobis 기반 이상치 탐지를 구현하고 정밀도, 재현율, F1 점수를 보고하시오. 유클리드 거리 기반 탐지와 비교하시오.

### 문제 4: PCA vs LDA

Iris 데이터셋에 대해:

1. PCA를 적용하여 2D로 축소
2. LDA를 적용하여 2D로 축소
3. 각 투영의 클래스 분리도를 비교 (클래스 간 / 클래스 내 분산 비율로 측정)
4. 어떤 방법이 클래스를 더 잘 분리하며, 그 이유는?

### 문제 5: 행렬 인수분해

알려진 저랭크 구조(랭크 5)를 가진 합성 사용자-아이템 평점 행렬(50명 사용자, 100개 아이템)을 생성하시오. 50%의 항목을 제거하시오. SVD 기반 행렬 인수분해를 사용하여 누락된 항목을 예측하고 RMSE를 계산하시오.

---

[이전: 15강](./15_Tensors_and_Multilinear_Algebra.md) | [개요](./00_Overview.md) | [다음: 17강](./17_Linear_Algebra_in_Deep_Learning.md)

**License**: CC BY-NC 4.0
