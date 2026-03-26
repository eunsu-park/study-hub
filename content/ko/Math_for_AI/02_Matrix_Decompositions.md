# 02. 행렬 분해 (Matrix Decompositions)

## 학습 목표

- 고유값과 고유벡터의 개념을 이해하고 계산할 수 있다
- 대칭 행렬의 스펙트럼 정리와 양정치 행렬의 성질을 설명할 수 있다
- 특이값 분해(SVD)의 기하학적 의미를 이해하고 구현할 수 있다
- SVD를 이용한 저랭크 근사와 PCA의 수학적 원리를 설명할 수 있다
- LU, QR, Cholesky 분해의 용도와 계산 방법을 이해한다
- 행렬 분해가 머신러닝에서 어떻게 활용되는지 구체적인 예를 들 수 있다

---

## 1. 고유값과 고유벡터

### 1.1 정의와 의미

행렬 $A$의 고유벡터(eigenvector) $\mathbf{v}$는 다음을 만족하는 0이 아닌 벡터입니다:

$$A\mathbf{v} = \lambda\mathbf{v}$$

여기서 $\lambda$는 고유값(eigenvalue)입니다.

**기하학적 해석**: 고유벡터는 선형 변환 $A$에 의해 방향은 바뀌지 않고 크기만 $\lambda$배 변하는 특별한 방향입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 2x2 행렬의 고유값과 고유벡터
A = np.array([[3, 1],
              [0, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# 검증: Av = λv
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

### 1.2 특성방정식 (Characteristic Equation)

고유값은 특성방정식의 해입니다:

$$\det(A - \lambda I) = 0$$

이는 $n$차 다항식이며, $n$개의 고유값을 가집니다 (중근 포함).

```python
# 특성방정식 수동 계산 (2x2 예제)
A = np.array([[4, 2],
              [1, 3]])

# det(A - λI) = 0
# (4-λ)(3-λ) - 2*1 = 0
# λ² - 7λ + 10 = 0
# (λ-5)(λ-2) = 0

eigenvalues = np.linalg.eigvals(A)
print(f"Eigenvalues: {eigenvalues}")  # [5, 2]

# 검증: 특성방정식
for lam in eigenvalues:
    det = np.linalg.det(A - lam * np.eye(2))
    print(f"det(A - {lam}I) = {det:.10f}")  # ~0
```

### 1.3 고유값 분해 (Eigendecomposition)

$n \times n$ 행렬 $A$가 $n$개의 선형 독립인 고유벡터를 가지면:

$$A = V\Lambda V^{-1}$$

여기서:
- $V$: 고유벡터를 열로 하는 행렬
- $\Lambda$: 대각선에 고유값을 가지는 대각 행렬

```python
# 고유값 분해
A = np.array([[3, 1],
              [0, 2]])

eigenvalues, V = np.linalg.eig(A)
Lambda = np.diag(eigenvalues)

print(f"V (eigenvectors):\n{V}")
print(f"Λ (eigenvalues):\n{Lambda}")

# 재구성: A = VΛV^(-1)
A_reconstructed = V @ Lambda @ np.linalg.inv(V)
print(f"A reconstructed:\n{A_reconstructed}")
print(f"Original A:\n{A}")
print(f"Equal? {np.allclose(A, A_reconstructed)}")
```

### 1.4 거듭제곱과 행렬 함수

고유값 분해를 이용하면 행렬 거듭제곱이 쉬워집니다:

$$A^k = V\Lambda^k V^{-1}$$

행렬 함수도 정의할 수 있습니다:

$$f(A) = Vf(\Lambda)V^{-1}$$

```python
# 행렬 거듭제곱
A = np.array([[2, 1],
              [1, 2]])

eigenvalues, V = np.linalg.eig(A)
Lambda = np.diag(eigenvalues)

# A^10 계산 (두 가지 방법)
# 방법 1: 직접 곱셈
A_10_direct = np.linalg.matrix_power(A, 10)

# 방법 2: 고유값 분해 이용
Lambda_10 = np.diag(eigenvalues**10)
A_10_eigen = V @ Lambda_10 @ np.linalg.inv(V)

print(f"A^10 (direct):\n{A_10_direct}")
print(f"A^10 (eigendecomposition):\n{A_10_eigen.real}")
print(f"Equal? {np.allclose(A_10_direct, A_10_eigen.real)}")

# 행렬 지수 함수
from scipy.linalg import expm
exp_A_scipy = expm(A)
exp_Lambda = np.diag(np.exp(eigenvalues))
exp_A_eigen = V @ exp_Lambda @ np.linalg.inv(V)
print(f"exp(A) match? {np.allclose(exp_A_scipy, exp_A_eigen.real)}")
```

## 2. 대칭 행렬의 스펙트럼 정리

### 2.1 스펙트럼 정리 (Spectral Theorem)

실수 대칭 행렬 $A = A^T$에 대해:
1. **모든 고유값이 실수**
2. **서로 다른 고유값에 대응하는 고유벡터들은 직교**
3. **직교 대각화 가능**: $A = Q\Lambda Q^T$ (여기서 $Q$는 직교 행렬)

```python
# 대칭 행렬의 고유값 분해
A_sym = np.array([[4, 2, 0],
                  [2, 3, 1],
                  [0, 1, 2]])

# 대칭 행렬인지 확인
print(f"Is symmetric? {np.allclose(A_sym, A_sym.T)}")

eigenvalues, Q = np.linalg.eigh(A_sym)  # 대칭 행렬 전용 함수
Lambda = np.diag(eigenvalues)

print(f"Eigenvalues: {eigenvalues}")
print(f"Q (orthogonal matrix):\n{Q}")

# 직교성 확인
print(f"Q^T Q = I?\n{Q.T @ Q}")
print(f"Is orthogonal? {np.allclose(Q.T @ Q, np.eye(3))}")

# 재구성: A = QΛQ^T
A_reconstructed = Q @ Lambda @ Q.T
print(f"Reconstruction error: {np.linalg.norm(A_sym - A_reconstructed)}")
```

### 2.2 양정치 행렬 (Positive Definite Matrix)

대칭 행렬 $A$가 양정치(positive definite)이려면:

$$\mathbf{x}^T A \mathbf{x} > 0 \quad \forall \mathbf{x} \neq \mathbf{0}$$

**동치 조건**:
- 모든 고유값이 양수
- 모든 주 소행렬식(leading principal minor)이 양수
- Cholesky 분해 가능

양정치 행렬은 머신러닝에서 매우 중요합니다 (공분산 행렬, 헤시안 등).

```python
# 양정치 행렬 예제
A_pd = np.array([[2, -1, 0],
                 [-1, 2, -1],
                 [0, -1, 2]])

eigenvalues = np.linalg.eigvalsh(A_pd)
print(f"Eigenvalues: {eigenvalues}")
print(f"All positive? {np.all(eigenvalues > 0)}")

# x^T A x > 0 확인
x = np.random.randn(3)
quadratic_form = x.T @ A_pd @ x
print(f"x^T A x = {quadratic_form}")
print(f"Positive? {quadratic_form > 0}")

# 반양정치(positive semidefinite) 예제
A_psd = np.array([[1, 1],
                  [1, 1]])
eigenvalues_psd = np.linalg.eigvalsh(A_psd)
print(f"PSD eigenvalues: {eigenvalues_psd}")  # [0, 2]
print(f"All non-negative? {np.all(eigenvalues_psd >= 0)}")
```

### 2.3 대칭 행렬의 기하학적 해석

대칭 행렬은 타원체(ellipsoid)를 정의합니다. 고유벡터는 주축(principal axes)의 방향이고, 고유값은 각 축의 길이에 대응됩니다.

```python
# 2D 타원 시각화
A = np.array([[3, 1],
              [1, 2]])

eigenvalues, eigenvectors = np.linalg.eigh(A)

# 단위원을 A로 변환
theta = np.linspace(0, 2*np.pi, 100)
unit_circle = np.array([np.cos(theta), np.sin(theta)])
ellipse = A @ unit_circle

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 단위원
ax1.plot(unit_circle[0], unit_circle[1], 'b-', label='Unit circle')
ax1.set_aspect('equal')
ax1.grid(True)
ax1.legend()
ax1.set_title('Before transformation')

# 타원
ax2.plot(ellipse[0], ellipse[1], 'r-', label='Ellipse')
# 고유벡터 그리기
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

## 3. 특이값 분해 (Singular Value Decomposition, SVD)

### 3.1 SVD의 정의

모든 $m \times n$ 행렬 $A$는 다음과 같이 분해됩니다:

$$A = U\Sigma V^T$$

여기서:
- $U$: $m \times m$ 직교 행렬 (left singular vectors)
- $\Sigma$: $m \times n$ 대각 행렬 (singular values, $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$)
- $V$: $n \times n$ 직교 행렬 (right singular vectors)

```python
# SVD 계산
A = np.array([[3, 1, 1],
              [-1, 3, 1]])

U, sigma, VT = np.linalg.svd(A, full_matrices=True)

print(f"A shape: {A.shape}")
print(f"U shape: {U.shape}")
print(f"sigma: {sigma}")  # 1D 배열
print(f"V^T shape: {VT.shape}")

# Σ 행렬 재구성 (m x n)
Sigma = np.zeros_like(A, dtype=float)
Sigma[:min(A.shape), :min(A.shape)] = np.diag(sigma)

print(f"Sigma shape: {Sigma.shape}")
print(f"Sigma:\n{Sigma}")

# 재구성 확인
A_reconstructed = U @ Sigma @ VT
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed)}")
```

### 3.2 SVD의 기하학적 의미

SVD는 모든 선형 변환을 세 단계로 분해합니다:
1. $V^T$: 회전/반사
2. $\Sigma$: 축 방향 스케일링
3. $U$: 회전/반사

```
  V^T          Σ           U
 ───────    ───────    ───────
 Rotate  →  Scale   →  Rotate
```

```python
# SVD의 기하학적 해석
A = np.array([[3, 1],
              [1, 2]])

U, sigma, VT = np.linalg.svd(A)

# 단위원
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

# 각 단계 적용
step1 = VT @ circle  # 첫 번째 회전
Sigma_2d = np.diag(sigma)
step2 = Sigma_2d @ step1  # 스케일링
step3 = U @ step2  # 두 번째 회전

# 최종 결과 (한번에)
final = A @ circle

# 시각화
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

# 검증
axes[1, 2].plot(step3[0], step3[1], 'r-', alpha=0.5, label='SVD steps')
axes[1, 2].plot(final[0], final[1], 'k--', label='Direct')
axes[1, 2].set_title('Verification')
axes[1, 2].set_aspect('equal')
axes[1, 2].grid(True)
axes[1, 2].legend()

plt.tight_layout()
plt.show()
```

### 3.3 SVD와 고유값 분해의 관계

$A^TA$와 $AA^T$의 고유값 분해로부터 SVD를 얻을 수 있습니다:

$$A^TA = V\Sigma^T\Sigma V^T = V\Sigma^2 V^T$$
$$AA^T = U\Sigma\Sigma^T U^T = U\Sigma^2 U^T$$

즉:
- $V$는 $A^TA$의 고유벡터
- $U$는 $AA^T$의 고유벡터
- $\sigma_i^2$는 $A^TA$ (또는 $AA^T$)의 고유값

```python
# SVD와 고유값 분해의 관계
A = np.array([[3, 1, 1],
              [-1, 3, 1]])

# 방법 1: 직접 SVD
U_svd, sigma_svd, VT_svd = np.linalg.svd(A)

# 방법 2: A^T A의 고유값 분해
ATA = A.T @ A
eigenvalues_ATA, V_eigen = np.linalg.eigh(ATA)
# 고유값을 내림차순으로 정렬
idx = eigenvalues_ATA.argsort()[::-1]
eigenvalues_ATA = eigenvalues_ATA[idx]
V_eigen = V_eigen[:, idx]

sigma_from_eigen = np.sqrt(eigenvalues_ATA)

print(f"Singular values (SVD): {sigma_svd}")
print(f"sqrt(eigenvalues of A^T A): {sigma_from_eigen}")
print(f"V from SVD:\n{VT_svd.T}")
print(f"V from eigendecomposition:\n{V_eigen}")
```

### 3.4 저랭크 근사 (Low-Rank Approximation)

**Eckart-Young-Mirsky 정리**: 랭크 $k$ 근사 중 Frobenius 노름 의미에서 최선의 근사는:

$$A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T = U_k \Sigma_k V_k^T$$

```python
# 저랭크 근사 예제: 이미지 압축
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

# 샘플 이미지 로드
china = load_sample_image("china.jpg")
# 그레이스케일로 변환
china_gray = china.mean(axis=2)

# SVD
U, sigma, VT = np.linalg.svd(china_gray, full_matrices=False)

# 다양한 랭크로 근사
ranks = [5, 20, 50, 100]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

axes[0].imshow(china_gray, cmap='gray')
axes[0].set_title(f'Original (rank={np.linalg.matrix_rank(china_gray)})')
axes[0].axis('off')

for i, k in enumerate(ranks, 1):
    # 랭크 k 근사
    A_k = U[:, :k] @ np.diag(sigma[:k]) @ VT[:k, :]

    axes[i].imshow(A_k, cmap='gray')

    # 압축률 계산
    original_size = china_gray.size
    compressed_size = k * (U.shape[0] + VT.shape[1] + 1)
    compression_ratio = original_size / compressed_size

    axes[i].set_title(f'Rank {k} (compression: {compression_ratio:.1f}x)')
    axes[i].axis('off')

# 특이값 분포
axes[-1].plot(sigma, 'b-')
axes[-1].set_xlabel('Index')
axes[-1].set_ylabel('Singular value')
axes[-1].set_title('Singular value spectrum')
axes[-1].set_yscale('log')
axes[-1].grid(True)

plt.tight_layout()
plt.show()
```

## 4. PCA: SVD에서 주성분 분석으로

### 4.1 공분산 행렬 (Covariance Matrix)

데이터 행렬 $X$ ($n$ 샘플 × $d$ 피처)에 대해, 공분산 행렬은:

$$C = \frac{1}{n-1}X^TX \quad \text{(데이터가 중심화되어 있다고 가정)}$$

공분산 행렬은 대칭이고 반양정치입니다.

```python
# 공분산 행렬
np.random.seed(42)
X = np.random.randn(100, 3)

# 데이터 중심화
X_centered = X - X.mean(axis=0)

# 공분산 행렬 (두 가지 방법)
cov_manual = (X_centered.T @ X_centered) / (len(X) - 1)
cov_numpy = np.cov(X.T)

print(f"Manual covariance:\n{cov_manual}")
print(f"NumPy covariance:\n{cov_numpy}")
print(f"Equal? {np.allclose(cov_manual, cov_numpy)}")

# 대칭성 확인
print(f"Is symmetric? {np.allclose(cov_manual, cov_manual.T)}")
```

### 4.2 PCA의 수학적 유도

**목표**: 데이터의 분산을 최대화하는 방향(주성분)을 찾기

첫 번째 주성분 $\mathbf{w}_1$은 다음 문제의 해:

$$\max_{\mathbf{w}} \mathbf{w}^T C \mathbf{w} \quad \text{subject to } \|\mathbf{w}\| = 1$$

**해**: 공분산 행렬 $C$의 가장 큰 고유값에 대응하는 고유벡터

$k$번째 주성분은 이전 $k-1$개에 직교하면서 분산을 최대화하는 방향입니다.

```python
# PCA 수동 구현
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

# 1. 데이터 중심화
X_centered = X - X.mean(axis=0)

# 2. 공분산 행렬
cov = np.cov(X_centered.T)

# 3. 고유값 분해
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# 4. 내림차순 정렬
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Eigenvalues (variances): {eigenvalues}")
print(f"Principal components (eigenvectors):\n{eigenvectors}")

# 5. 변환 (처음 2개 주성분으로 투영)
k = 2
X_pca = X_centered @ eigenvectors[:, :k]

print(f"Transformed data shape: {X_pca.shape}")
```

### 4.3 SVD를 이용한 PCA

공분산 행렬을 명시적으로 계산하지 않고 SVD를 이용할 수 있습니다:

$$X = U\Sigma V^T \quad \Rightarrow \quad X^TX = V\Sigma^2 V^T$$

따라서 $V$의 열벡터들이 주성분이고, $\sigma_i^2/(n-1)$이 분산입니다.

```python
# SVD를 이용한 PCA
X_centered = X - X.mean(axis=0)

# SVD
U, sigma, VT = np.linalg.svd(X_centered, full_matrices=False)

# 주성분 = V의 열벡터
principal_components = VT.T
print(f"Principal components (from SVD):\n{principal_components}")

# 설명된 분산
explained_variance = (sigma**2) / (len(X) - 1)
print(f"Explained variance: {explained_variance}")

# 분산 비율
explained_variance_ratio = explained_variance / explained_variance.sum()
print(f"Explained variance ratio: {explained_variance_ratio}")

# 변환
X_pca_svd = X_centered @ principal_components[:, :2]
print(f"Same as eigendecomposition? {np.allclose(np.abs(X_pca), np.abs(X_pca_svd))}")
```

### 4.4 PCA 시각화와 해석

```python
from sklearn.decomposition import PCA as SklearnPCA

# scikit-learn PCA
pca = SklearnPCA(n_components=2)
X_sklearn = pca.fit_transform(X)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 변환된 데이터
for i, target_name in enumerate(iris.target_names):
    mask = iris.target == i
    axes[0].scatter(X_sklearn[mask, 0], X_sklearn[mask, 1],
                   label=target_name, alpha=0.7)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
axes[0].set_title('PCA of Iris Dataset')
axes[0].legend()
axes[0].grid(True)

# 설명된 분산
axes[1].bar(range(1, len(pca.explained_variance_ratio_)+1),
           pca.explained_variance_ratio_)
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Explained Variance Ratio')
axes[1].set_title('Scree Plot')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 주성분 로딩 (원래 피처와의 상관관계)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
print("\nFeature loadings:")
for i, feature in enumerate(iris.feature_names):
    print(f"{feature:20s}: PC1={loadings[i,0]:6.3f}, PC2={loadings[i,1]:6.3f}")
```

### 4.5 차원 축소와 재구성

PCA는 데이터를 저차원으로 압축하고 다시 복원할 수 있습니다.

```python
# 차원 축소와 재구성
pca_full = SklearnPCA(n_components=4)
X_transformed = pca_full.fit_transform(X)

# 다양한 성분 개수로 재구성
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, n_components in enumerate([1, 2, 3, 4]):
    # 처음 n개 성분만 사용
    X_reduced = X_transformed[:, :n_components]

    # 재구성
    X_reconstructed = pca_full.inverse_transform(
        np.column_stack([X_reduced,
                        np.zeros((len(X), 4-n_components))])
    )

    # 재구성 오차
    reconstruction_error = np.mean((X - X_reconstructed)**2)
    explained_var = pca_full.explained_variance_ratio_[:n_components].sum()

    # 첫 두 피처만 시각화
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

## 5. 기타 행렬 분해

### 5.1 LU 분해

정방행렬 $A$를 하삼각행렬 $L$과 상삼각행렬 $U$의 곱으로 분해:

$$A = LU$$

**용도**: 연립방정식 $Ax = b$ 풀기, 행렬식 계산

```python
from scipy.linalg import lu

# LU 분해
A = np.array([[2, 1, 1],
              [4, -6, 0],
              [-2, 7, 2]])

P, L, U = lu(A)

print(f"P (permutation matrix):\n{P}")
print(f"L (lower triangular):\n{L}")
print(f"U (upper triangular):\n{U}")

# 재구성
A_reconstructed = P @ L @ U
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed)}")

# LU 분해를 이용한 방정식 풀이
b = np.array([4, 2, 6])

# Ax = b를 LUx = b로 변환
# 1. Ly = Pb 풀기 (전방 대입)
# 2. Ux = y 풀기 (후방 대입)

from scipy.linalg import solve_triangular

y = solve_triangular(L, P @ b, lower=True)
x = solve_triangular(U, y, lower=False)

print(f"Solution: {x}")
print(f"Verification: Ax = {A @ x}")
```

### 5.2 QR 분해

행렬 $A$를 직교행렬 $Q$와 상삼각행렬 $R$의 곱으로 분해:

$$A = QR$$

**용도**: 최소자승 문제, 그람-슈미트 직교화

```python
# QR 분해
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]])

Q, R = np.linalg.qr(A)

print(f"Q (orthogonal):\n{Q}")
print(f"R (upper triangular):\n{R}")

# 직교성 확인
print(f"Q^T Q:\n{Q.T @ Q}")

# 재구성
A_reconstructed = Q @ R
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed)}")

# 최소자승 문제 풀기: Ax = b (overdetermined)
A_over = np.array([[1, 1],
                   [1, 2],
                   [1, 3],
                   [1, 4]])
b = np.array([2, 3, 5, 6])

Q, R = np.linalg.qr(A_over)
# R은 처음 2행만 상삼각
R_square = R[:2, :]
Q_reduced = Q[:, :2]

# x = R^(-1) Q^T b
x = np.linalg.solve(R_square, Q_reduced.T @ b)
print(f"Least squares solution: {x}")
```

### 5.3 Cholesky 분해

양정치 대칭행렬 $A$를 하삼각행렬 $L$로 분해:

$$A = LL^T$$

**용도**: 가우시안 샘플링, 수치적으로 안정적인 계산

```python
# Cholesky 분해
A_pd = np.array([[4, 2, 2],
                 [2, 5, 1],
                 [2, 1, 6]])

# 양정치 확인
eigenvalues = np.linalg.eigvalsh(A_pd)
print(f"Eigenvalues: {eigenvalues}")
print(f"Is positive definite? {np.all(eigenvalues > 0)}")

# Cholesky 분해
L = np.linalg.cholesky(A_pd)
print(f"L:\n{L}")

# 재구성
A_reconstructed = L @ L.T
print(f"Reconstruction error: {np.linalg.norm(A_pd - A_reconstructed)}")

# 응용: 다변량 가우시안 샘플링
# N(μ, Σ)에서 샘플링하려면: μ + L @ z (z ~ N(0, I))
mu = np.array([1, 2, 3])
Sigma = A_pd
L = np.linalg.cholesky(Sigma)

# 표준 정규분포에서 샘플링
n_samples = 1000
Z = np.random.randn(n_samples, 3)

# 변환
samples = mu + (L @ Z.T).T

# 검증
print(f"Sample mean: {samples.mean(axis=0)}")
print(f"True mean: {mu}")
print(f"Sample covariance:\n{np.cov(samples.T)}")
print(f"True covariance:\n{Sigma}")
```

### 5.4 분해 방법 비교

| 분해 | 조건 | 계산 복잡도 | 주요 용도 |
|------|------|------------|-----------|
| Eigendecomposition | 정방행렬 | $O(n^3)$ | 행렬 거듭제곱, 동역학 시스템 |
| SVD | 모든 행렬 | $O(mn^2)$ | 저랭크 근사, PCA, 추천 시스템 |
| LU | 정방행렬 | $O(n^3)$ | 연립방정식, 행렬식 |
| QR | 모든 행렬 | $O(mn^2)$ | 최소자승, 직교화 |
| Cholesky | 양정치 | $O(n^3)$ | 샘플링, 수치 안정성 |

## 6. ML에서의 행렬 분해 응용

### 6.1 추천 시스템: Matrix Factorization

사용자-아이템 평점 행렬 $R$을 저랭크로 근사:

$$R \approx UV^T$$

```python
# 간단한 추천 시스템
from sklearn.decomposition import TruncatedSVD

# 사용자-영화 평점 행렬 (5 users × 6 movies)
R = np.array([
    [5, 3, 0, 1, 0, 0],
    [4, 0, 0, 1, 0, 0],
    [1, 1, 0, 5, 0, 0],
    [0, 0, 0, 4, 4, 5],
    [0, 0, 5, 0, 4, 4]
])

# 0은 결측값
mask = R > 0

# SVD (랭크 2 근사)
svd = TruncatedSVD(n_components=2)
U = svd.fit_transform(R)
VT = svd.components_

# 재구성
R_approx = U @ VT

print("Original ratings:")
print(R)
print("\nApproximated ratings:")
print(R_approx)

# 결측값 예측
print("\nPredicted ratings for missing values:")
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i, j] == 0:
            print(f"User {i}, Movie {j}: {R_approx[i, j]:.2f}")
```

### 6.2 이미지 압축

SVD를 이용한 이미지 압축은 앞서 보았습니다. 각 색상 채널마다 독립적으로 SVD를 적용할 수 있습니다.

### 6.3 스펙트럼 군집화 (Spectral Clustering)

그래프 라플라시안 행렬의 고유벡터를 이용한 군집화:

```python
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons

# 비선형 분리 가능한 데이터
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# 스펙트럼 군집화
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                               random_state=42)
labels = spectral.fit_predict(X)

# 시각화
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

### 6.4 특성 추출: Kernel PCA

비선형 차원 축소를 위해 커널 트릭과 PCA를 결합:

```python
from sklearn.decomposition import KernelPCA

# 원형 데이터
theta = np.linspace(0, 2*np.pi, 100)
r = 1 + 0.3 * np.random.randn(100)
X_circle = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

# 선형 PCA
pca_linear = SklearnPCA(n_components=1)
X_pca_linear = pca_linear.fit_transform(X_circle)

# Kernel PCA (RBF 커널)
kpca = KernelPCA(n_components=1, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X_circle)

# 시각화
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

## 연습 문제

### 문제 1: 고유값 분해
다음 행렬의 고유값과 고유벡터를 손으로 계산하고, NumPy로 검증하세요:

$$A = \begin{bmatrix} 5 & 2 \\ 2 & 2 \end{bmatrix}$$

이 행렬이 양정치인지 판별하고, $A^{10}$을 고유값 분해를 이용해 계산하세요.

### 문제 2: SVD와 저랭크 근사
$4 \times 3$ 행렬을 생성하고 SVD를 계산하세요. 랭크 1과 랭크 2 근사를 구하고, 각각의 Frobenius 노름 오차를 계산하세요.

### 문제 3: PCA 구현
붓꽃 데이터셋에 대해:
1. 공분산 행렬의 고유값 분해로 PCA 수행
2. SVD로 PCA 수행
3. 두 방법의 결과가 동일한지 확인
4. 누적 설명 분산이 95%가 되는 최소 성분 개수는?

### 문제 4: Cholesky 분해와 샘플링
공분산 행렬 $\Sigma = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$와 평균 $\mu = [0, 0]^T$를 가지는 2차원 가우시안 분포에서 1000개 샘플을 생성하세요. Cholesky 분해를 이용하고, 샘플의 경험적 공분산이 $\Sigma$와 가까운지 확인하세요.

### 문제 5: 추천 시스템
다음 사용자-영화 평점 행렬에서 결측값을 SVD로 예측하세요:

$$R = \begin{bmatrix} 5 & ? & 4 & ? \\ ? & 3 & ? & 2 \\ 4 & ? & 5 & ? \\ ? & 2 & ? & 3 \end{bmatrix}$$

랭크 2 근사를 사용하고, 예측된 평점이 1-5 범위에 있는지 확인하세요.

## 참고 자료

### 교재
1. **Strang, G.** (2016). *Introduction to Linear Algebra*. Chapter 6: Eigenvalues and Eigenvectors.
2. **Trefethen, L. N., & Bau III, D.** (1997). *Numerical Linear Algebra*. SIAM.
   - SVD와 수치 안정성에 대한 깊이 있는 논의
3. **Golub, G. H., & Van Loan, C. F.** (2013). *Matrix Computations* (4th ed.). Johns Hopkins.
   - 행렬 분해 알고리즘의 바이블

### 논문
1. **Eckart, C., & Young, G.** (1936). "The approximation of one matrix by another of lower rank." *Psychometrika*.
2. **Jolliffe, I. T.** (2002). *Principal Component Analysis* (2nd ed.). Springer.

### 온라인 자료
1. **3Blue1Brown - Eigenvectors and eigenvalues**: https://www.youtube.com/watch?v=PFDu9oVAE-g
2. **Stanford CS229 - Linear Algebra Review**: http://cs229.stanford.edu/section/cs229-linalg.pdf
3. **scikit-learn PCA Guide**: https://scikit-learn.org/stable/modules/decomposition.html

### 실습 도구
1. **NumPy Linear Algebra**: https://numpy.org/doc/stable/reference/routines.linalg.html
2. **SciPy Linear Algebra**: https://docs.scipy.org/doc/scipy/reference/linalg.html

---

**다음 레슨**: [03. 행렬 미적분](03_Matrix_Calculus.md)에서 역전파의 수학적 기반을 배웁니다.
