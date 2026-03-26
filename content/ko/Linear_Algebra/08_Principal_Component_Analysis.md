# 레슨 8: 주성분 분석 (PCA)

## 학습 목표

- 고유분해와 SVD 관점 모두에서 PCA를 유도할 수 있습니다
- "설명된 분산"의 의미를 설명하고 스크리 도표(scree plot)를 사용하여 성분 수를 선택할 수 있습니다
- PCA를 처음부터 구현하고 scikit-learn과 비교할 수 있습니다
- 실제 차원 축소 작업에 PCA를 적용할 수 있습니다
- PCA의 한계와 가정을 이해할 수 있습니다

---

## 1. 동기

### 1.1 고차원의 문제

많은 데이터셋은 필요한 것보다 훨씬 많은 특성을 가지고 있습니다. 중복된 특성은 저장 공간을 낭비하고, 계산을 느리게 하며, 모델 성능을 저하시킬 수 있습니다("차원의 저주"). **차원 축소**(dimensionality reduction)는 가능한 한 많은 유용한 정보를 유지하는 저차원 표현을 찾습니다.

### 1.2 PCA를 한 문장으로

PCA는 데이터에서 최대 분산 방향을 찾아 데이터를 그 방향으로 투영합니다.

### 1.3 가정

PCA는 다음을 가정합니다:
- 데이터에서 가장 중요한 구조는 **최대 분산** 방향을 따라 존재합니다
- 데이터는 중심화(평균 제거)되어 있습니다
- 선형 투영만으로 구조를 포착하기에 충분합니다

---

## 2. 고유분해를 통한 PCA

### 2.1 공분산 행렬

중심화된 데이터 행렬 $X \in \mathbb{R}^{n \times d}$ ($n$은 샘플 수, $d$는 특성 수)가 주어졌을 때, **공분산 행렬**은 다음과 같습니다:

$$C = \frac{1}{n-1} X^T X \in \mathbb{R}^{d \times d}$$

$C$는 대칭이며 양의 준정부호(positive semi-definite)입니다.

### 2.2 유도

PCA는 투영된 데이터의 분산을 최대화하는 단위 벡터 $\mathbf{w}$를 찾습니다:

$$\max_{\|\mathbf{w}\| = 1} \mathrm{Var}(X\mathbf{w}) = \max_{\|\mathbf{w}\| = 1} \mathbf{w}^T C \mathbf{w}$$

스펙트럼 정리에 의해, 해는 $C$의 **최대 고유값**에 대응하는 고유벡터입니다. $k$번째 주성분은 $k$번째로 큰 고유값에 대응하는 고유벡터입니다.

### 2.3 알고리즘

1. 데이터를 **중심화**합니다: $X \leftarrow X - \bar{X}$
2. 공분산 행렬을 계산합니다: $C = \frac{1}{n-1} X^T X$
3. 고유분해를 계산합니다: $C = Q \Lambda Q^T$
4. 고유값을 내림차순으로 정렬합니다
5. 상위 $k$개 고유벡터를 주성분으로 선택합니다
6. 투영합니다: $Z = X Q_k$ ($Q_k$는 상위 $k$개 고유벡터 행렬)

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

## 3. SVD를 통한 PCA

### 3.1 SVD와의 연결

중심화된 데이터 행렬 $X \in \mathbb{R}^{n \times d}$의 SVD $X = U\Sigma V^T$에 대해:

$$C = \frac{1}{n-1} X^T X = V \frac{\Sigma^2}{n-1} V^T$$

따라서:
- **오른쪽 특이벡터** $V$가 주성분 방향입니다
- $C$의 고유값은 $\lambda_i = \sigma_i^2 / (n-1)$입니다
- 투영된 데이터는 $Z = XV = U\Sigma$입니다

### 3.2 SVD가 선호되는 이유

SVD 접근법이 수치적으로 우수한 이유는 다음과 같습니다:
1. $X^TX$를 명시적으로 구성하지 않습니다(부동소수점 오차를 증폭시킬 수 있음)
2. 희소(sparse)할 수 있는 $X$를 직접 다룹니다
3. $n < d$(샘플보다 특성이 많은 경우)를 자연스럽게 처리합니다

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

## 4. 설명된 분산

### 4.1 정의

$i$번째 주성분이 **설명하는 분산**은 다음과 같습니다:

$$\text{Variance explained ratio}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$

**누적 설명된 분산**은 처음 $k$개 성분이 포착하는 총 분산의 비율을 나타냅니다:

$$\text{Cumulative}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}$$

### 4.2 스크리 도표 (Scree Plot)

**스크리 도표**는 고유값(또는 분산 비율)을 성분 인덱스에 대해 표시합니다. "팔꿈치"(elbow)는 성분 수의 자연스러운 절단점을 제안합니다.

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

## 5. PCA 직접 구현

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

## 6. 재구성과 재구성 오차

### 6.1 투영과 재구성

데이터를 $k$개 성분으로 투영하면 정보가 손실됩니다. **재구성**은 원래 공간으로 다시 매핑합니다:

$$\hat{X} = Z Q_k^T + \bar{X}$$

**재구성 오차**는 다음과 같습니다:

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

## 7. 차원 축소 워크플로

### 7.1 완전한 PCA 파이프라인

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

### 7.2 표준화 시점

- 특성이 서로 다른 단위나 매우 다른 스케일을 가질 때 **항상 표준화**합니다
- 특성이 이미 같은 스케일인 경우(예: 픽셀 값), 표준화는 선택 사항입니다
- 스케일링하지 않은 데이터에 대한 PCA는 중요도에 관계없이 고분산 특성에 의해 지배됩니다

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

## 8. scikit-learn과의 비교

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

## 9. PCA의 한계

| 한계 | 설명 |
|------|------|
| 선형만 가능 | 비선형 구조를 포착할 수 없습니다 (대신 kernel PCA, t-SNE, UMAP 사용) |
| 분산 편향 | 분산 = 중요도를 가정하지만 항상 그렇지는 않습니다 |
| 이상치 민감도 | 이상치가 공분산 행렬에 불균형적으로 영향을 미칩니다 |
| 해석 가능성 | 주성분은 모든 특성의 선형 결합으로 해석이 어렵습니다 |
| 스케일 의존성 | 결과가 특성 스케일링에 의존합니다 |

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

## 10. 요약

| 개념 | 설명 |
|------|------|
| PCA 목표 | 최대 분산 방향을 찾습니다 |
| 공분산 행렬 | $C = \frac{1}{n-1}X^TX$ |
| 고유분해를 통한 PCA | $C$의 고유벡터가 주성분 방향입니다 |
| SVD를 통한 PCA | 중심화된 $X$의 오른쪽 특이벡터 |
| 설명된 분산 | $\lambda_i / \sum \lambda_j$ |
| 스크리 도표 | 고유값 vs 성분 인덱스 |
| 재구성 | $\hat{X} = ZQ_k^T + \bar{X}$ |
| NumPy | `np.linalg.svd()` 또는 `np.linalg.eigh()` |
| scikit-learn | `sklearn.decomposition.PCA` |

---

## 연습 문제

### 연습 문제 1: PCA 직접 구현

NumPy만 사용하여 PCA를 처음부터 구현하세요 (고유분해 및 SVD 접근법 모두). 두 방법을 Iris 데이터셋에 적용하고 동일한 결과를 생성하는지 검증하세요.

### 연습 문제 2: 스크리 도표 분석

알려진 고유 차원을 가진 데이터셋을 생성하세요 (예: 50차원 공간에 5개의 잠재 인자). PCA를 수행하고 스크리 도표를 생성하세요. 도표에서 실제 차원을 복원할 수 있습니까?

### 연습 문제 3: 고유 얼굴 (Eigenfaces)

얼굴 데이터셋을 다운로드하세요 (예: `sklearn.datasets.fetch_olivetti_faces`를 통한 Olivetti 얼굴). PCA를 적용하고 상위 10개 주성분을 "고유 얼굴"로 시각화하세요. 분산의 90%를 포착하려면 몇 개의 성분이 필요합니까?

### 연습 문제 4: 표준화의 효과

한 특성이 다른 특성보다 훨씬 큰 분산을 가진 데이터셋을 생성하세요. 표준화 유무에 따라 PCA를 실행하세요. 결과를 비교하고 설명하세요.

### 연습 문제 5: 재구성 품질

Iris 데이터셋을 사용하여 $k = 1, 2, 3, 4$에 대해:
1. PCA를 사용하여 $k$차원으로 투영합니다
2. 4차원으로 다시 재구성합니다
3. 평균 제곱 재구성 오차를 계산합니다
4. $k$의 함수로 오차를 도표로 그립니다

---

[<< 이전: 레슨 7 - 특이값 분해](07_Singular_Value_Decomposition.md) | [개요](00_Overview.md) | [다음: 레슨 9 - 직교성과 투영 >>](09_Orthogonality_and_Projections.md)

**License**: CC BY-NC 4.0
