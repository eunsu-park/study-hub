# 19강: 랜덤화 선형대수 (Randomized Linear Algebra)

[이전: 18강](./18_Linear_Algebra_in_Computer_Graphics.md) | [개요](./00_Overview.md) | [다음: 20강](./20_Advanced_Topics.md)

---

## 학습 목표

- 선형대수에서 계산 도구로서 무작위성의 역할을 이해할 수 있다
- Johnson-Lindenstrauss 보조정리를 진술하고 해석할 수 있다
- 랜덤 투영을 구현하고 거리 보존 특성을 분석할 수 있다
- 효율적인 저랭크 근사를 위한 랜덤화 SVD를 유도하고 구현할 수 있다
- 스트리밍 및 대규모 데이터에 대한 행렬 스케치를 적용할 수 있다
- 랜덤화 알고리즘에서 정확도, 속도, 메모리 간의 트레이드오프를 평가할 수 있다

---

## 1. 왜 랜덤화 알고리즘인가?

### 1.1 동기

고전적 선형대수 알고리즘(SVD, 고유 분해, LU)은 결정론적이며 정확도 면에서 최적이지만, $O(n^3)$ 또는 $O(mn^2)$ 비용이 현대 데이터 크기에서는 감당할 수 없게 됩니다. 랜덤화 알고리즘은 작고 제어 가능한 양의 정확도를 교환하여 극적인 속도 향상을 얻습니다.

**핵심 통찰**: 실제로 접하는 대부분의 대규모 행렬은 근사적으로 저랭크입니다. 랜덤화 방법은 "중요한" 부분공간을 무작위로 샘플링하고 나머지를 버림으로써 이를 활용합니다.

| 방법 | 결정론적 | 랜덤화 |
|---|---|---|
| $m \times n$ 전체 SVD | $O(mn^2)$ | 해당 없음 |
| 랭크-$k$ SVD | $O(mn^2)$ | $O(mn \log k + (m + n)k^2)$ |
| 최소자승 | $O(mn^2)$ | $O(mn \log n)$ (스케치) |
| PCA | $O(mn^2)$ | $O(mn \log k)$ |

```python
import numpy as np
import time
import matplotlib.pyplot as plt

# Demonstrate the speedup of randomized vs deterministic SVD
from scipy.linalg import svd as full_svd

def create_low_rank_matrix(m, n, rank, noise=0.01):
    """Create a noisy low-rank matrix."""
    U = np.random.randn(m, rank)
    V = np.random.randn(n, rank)
    # Singular values with geometric decay
    sigma = np.logspace(0, -3, rank)
    A = U @ np.diag(sigma) @ V.T
    A += noise * np.random.randn(m, n)
    return A

# Compare timing for different sizes
print(f"{'Size':>12s}  {'Full SVD (s)':>12s}  {'Rank-10 rSVD (s)':>16s}  {'Speedup':>8s}")

for n in [500, 1000, 2000, 4000]:
    A = create_low_rank_matrix(n, n, rank=50, noise=0.001)

    start = time.time()
    U, S, Vt = full_svd(A, full_matrices=False)
    t_full = time.time() - start

    start = time.time()
    # We will implement this properly in Section 4
    # For now, use a simple version
    Omega = np.random.randn(n, 20)
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A
    Ub, Sb, Vtb = np.linalg.svd(B, full_matrices=False)
    t_rand = time.time() - start

    print(f"{n:>5d}x{n:<5d}  {t_full:>12.4f}  {t_rand:>16.4f}  {t_full/t_rand:>7.1f}x")
```

---

## 2. 랜덤 투영

### 2.1 아이디어

**랜덤 투영**은 고차원 데이터를 랜덤 행렬 $\Omega \in \mathbb{R}^{d \times k}$ (여기서 $k \ll d$)를 사용하여 저차원 공간으로 매핑합니다:

$$\mathbf{z} = \Omega^T \mathbf{x} \in \mathbb{R}^k$$

놀라운 사실은 이것이 높은 확률로 쌍별 거리를 보존한다는 것입니다.

```python
# Random projection types
def gaussian_projection(d, k):
    """Gaussian random projection matrix (dense)."""
    return np.random.randn(d, k) / np.sqrt(k)

def sparse_projection(d, k, density=1/3):
    """Sparse random projection (Achlioptas, 2003).

    Each entry is +1, 0, or -1 with probabilities 1/6, 2/3, 1/6.
    """
    s = 1.0 / density
    choices = np.random.choice([-1, 0, 0, 0, 0, 1], size=(d, k))
    return choices * np.sqrt(s / k)

def srht_projection(d, k):
    """Subsampled Randomized Hadamard Transform (simplified).

    Faster than Gaussian but harder to implement exactly.
    Here we use a random sign flip + random column selection.
    """
    # Random sign flips
    D = np.diag(np.random.choice([-1, 1], size=d))
    # Random subsampling
    idx = np.random.choice(d, k, replace=False)
    S = np.eye(d)[idx].T  # Subsampling matrix
    return D @ S / np.sqrt(k)

# Compare projection types
d, k = 1000, 50
X = np.random.randn(200, d)

projections = {
    'Gaussian': gaussian_projection(d, k),
    'Sparse': sparse_projection(d, k),
}

for name, Omega in projections.items():
    X_proj = X @ Omega
    print(f"{name:10s}: projection shape = {X_proj.shape}, "
          f"nonzeros = {np.count_nonzero(Omega)} / {Omega.size} "
          f"({np.count_nonzero(Omega)/Omega.size:.1%})")
```

### 2.2 거리 보존

```python
from scipy.spatial.distance import pdist, squareform

# Demonstrate distance preservation
np.random.seed(42)
n_points = 100
d_original = 500
X = np.random.randn(n_points, d_original)

# Original pairwise distances
dists_original = pdist(X, 'euclidean')

# Project to various dimensions and measure distortion
k_values = [10, 25, 50, 100, 200]
fig, axes = plt.subplots(1, len(k_values), figsize=(20, 4))

for ax, k in zip(axes, k_values):
    Omega = gaussian_projection(d_original, k)
    X_proj = X @ Omega
    dists_projected = pdist(X_proj, 'euclidean')

    # Distortion ratio
    ratios = dists_projected / dists_original

    ax.hist(ratios, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=1.0, color='r', linestyle='--', linewidth=2)
    ax.set_title(f'k={k}\nmean={ratios.mean():.3f}, std={ratios.std():.3f}')
    ax.set_xlabel('Distance ratio')

plt.suptitle('Distance Preservation Under Random Projection', y=1.02)
plt.tight_layout()
plt.show()
```

---

## 3. Johnson-Lindenstrauss 보조정리

### 3.1 정리

**Johnson-Lindenstrauss (JL) 보조정리**: $\mathbb{R}^d$의 임의의 $n$개 점 집합과 임의의 $\epsilon \in (0, 1)$에 대해, 다음을 만족하는 선형 사상 $f: \mathbb{R}^d \to \mathbb{R}^k$가 존재합니다:

$$k = O\left(\frac{\log n}{\epsilon^2}\right)$$

모든 쌍 $i, j$에 대해:

$$(1 - \epsilon) \|\mathbf{x}_i - \mathbf{x}_j\|^2 \leq \|f(\mathbf{x}_i) - f(\mathbf{x}_j)\|^2 \leq (1 + \epsilon) \|\mathbf{x}_i - \mathbf{x}_j\|^2$$

핵심: 목표 차원 $k$는 $\log n$과 $\epsilon$에만 의존하며, **원래 차원 $d$에는 의존하지 않습니다**.

```python
# Verify JL lemma empirically
def jl_min_dimension(n_points, epsilon):
    """Minimum dimension from JL lemma: k >= 8 * ln(n) / epsilon^2."""
    return int(np.ceil(8 * np.log(n_points) / epsilon**2))

# Print required dimensions
print("JL minimum dimensions (k):")
print(f"{'n_points':>10s}  {'eps=0.5':>8s}  {'eps=0.3':>8s}  {'eps=0.1':>8s}")
for n in [100, 1000, 10000, 100000, 1000000]:
    dims = [jl_min_dimension(n, eps) for eps in [0.5, 0.3, 0.1]]
    print(f"{n:>10d}  {dims[0]:>8d}  {dims[1]:>8d}  {dims[2]:>8d}")

print("\nNote: k depends on log(n), so doubling n barely changes k")
```

### 3.2 경험적 검증

```python
# Verify distortion bounds
def verify_jl(X, k, epsilon, n_trials=10):
    """Verify JL bounds for random projections."""
    n, d = X.shape
    dists_original = squareform(pdist(X, 'sqeuclidean'))

    violations = []
    for trial in range(n_trials):
        Omega = np.random.randn(d, k) / np.sqrt(k)
        X_proj = X @ Omega
        dists_proj = squareform(pdist(X_proj, 'sqeuclidean'))

        # Check bounds for all pairs
        mask = dists_original > 0  # Exclude self-distances
        ratios = dists_proj[mask] / dists_original[mask]
        n_violations = np.sum((ratios < 1 - epsilon) | (ratios > 1 + epsilon))
        total_pairs = np.sum(mask)
        violations.append(n_violations / total_pairs)

    return np.mean(violations), np.std(violations)

np.random.seed(42)
n_points = 200
d = 1000
X = np.random.randn(n_points, d)

epsilon = 0.3
k_jl = jl_min_dimension(n_points, epsilon)
print(f"n={n_points}, d={d}, epsilon={epsilon}")
print(f"JL minimum k: {k_jl}")

print(f"\n{'k':>6s}  {'Violation rate':>16s}")
for k in [10, 25, 50, k_jl, 200, 500]:
    rate, std = verify_jl(X, k, epsilon, n_trials=5)
    marker = " <-- JL bound" if k == k_jl else ""
    print(f"{k:>6d}  {rate:>14.4%} +/- {std:.4%}{marker}")
```

---

## 4. 랜덤화 SVD

### 4.1 알고리즘

**랜덤화 SVD** (Halko, Martinsson, Tropp, 2011)는 근사 랭크-$k$ SVD를 두 단계로 계산합니다:

**단계 A: $A$의 치역에 대한 근사 기저 구성**
1. 랜덤 행렬 $\Omega \in \mathbb{R}^{n \times (k+p)}$를 생성 (여기서 $p$는 오버샘플링, 일반적으로 5-10)
2. $Y = A\Omega$를 계산 (단지 행렬-행렬 곱셈)
3. QR 분해를 계산: $Y = QR$ (이제 $Q$가 상위 $k+p$ 특이벡터를 근사적으로 생성)

**단계 B: 작은 행렬 $B$를 형성하고 SVD를 계산**
1. $B = Q^T A$ ($A$를 근사 기저에 투영)
2. $B$의 SVD를 계산: $B = \hat{U}\Sigma V^T$
3. $U = Q\hat{U}$를 복원

```python
def randomized_svd(A, k, p=10, n_iter=2):
    """Randomized SVD algorithm.

    Args:
        A: (m, n) matrix
        k: target rank
        p: oversampling parameter (default 10)
        n_iter: number of power iterations for accuracy (default 2)

    Returns:
        U: (m, k) left singular vectors
        S: (k,) singular values
        Vt: (k, n) right singular vectors
    """
    m, n = A.shape
    l = k + p  # Oversampled rank

    # Stage A: random sampling
    Omega = np.random.randn(n, l)
    Y = A @ Omega

    # Power iterations for better accuracy (optional but recommended)
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    Q, _ = np.linalg.qr(Y)

    # Stage B: form and factorize small matrix
    B = Q.T @ A  # (l, n)
    U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)

    # Recover full left singular vectors
    U = Q @ U_hat

    # Truncate to rank k
    return U[:, :k], S[:k], Vt[:k, :]

# Test on a low-rank matrix
np.random.seed(42)
m, n = 2000, 1500
rank_true = 20
A = create_low_rank_matrix(m, n, rank_true, noise=0.001)

# Deterministic SVD (truncated)
start = time.time()
U_full, S_full, Vt_full = np.linalg.svd(A, full_matrices=False)
t_full = time.time() - start

# Randomized SVD
k = 20
start = time.time()
U_rand, S_rand, Vt_rand = randomized_svd(A, k, p=10, n_iter=2)
t_rand = time.time() - start

# Compare
A_approx_full = U_full[:, :k] @ np.diag(S_full[:k]) @ Vt_full[:k, :]
A_approx_rand = U_rand @ np.diag(S_rand) @ Vt_rand

err_full = np.linalg.norm(A - A_approx_full) / np.linalg.norm(A)
err_rand = np.linalg.norm(A - A_approx_rand) / np.linalg.norm(A)

print(f"Matrix shape: {A.shape}, true rank: {rank_true}")
print(f"Target rank k: {k}")
print(f"\nDeterministic SVD: time={t_full:.4f}s, error={err_full:.6f}")
print(f"Randomized SVD:    time={t_rand:.4f}s, error={err_rand:.6f}")
print(f"Speedup: {t_full/t_rand:.1f}x")
```

### 4.2 오버샘플링과 멱급수 반복의 효과

```python
# Analyze the effect of oversampling parameter p and power iterations
A = create_low_rank_matrix(1000, 800, rank=30, noise=0.01)
k = 30

results = {}

# Vary oversampling
for p in [0, 5, 10, 20, 50]:
    U, S, Vt = randomized_svd(A, k, p=p, n_iter=0)
    A_approx = U @ np.diag(S) @ Vt
    error = np.linalg.norm(A - A_approx) / np.linalg.norm(A)
    results[f'p={p}'] = error

print("Effect of oversampling (no power iterations):")
for key, err in results.items():
    print(f"  {key}: relative error = {err:.6f}")

# Vary power iterations
results_power = {}
for n_iter in [0, 1, 2, 3, 5]:
    U, S, Vt = randomized_svd(A, k, p=10, n_iter=n_iter)
    A_approx = U @ np.diag(S) @ Vt
    error = np.linalg.norm(A - A_approx) / np.linalg.norm(A)
    results_power[f'q={n_iter}'] = error

print("\nEffect of power iterations (p=10):")
for key, err in results_power.items():
    print(f"  {key}: relative error = {err:.6f}")
```

### 4.3 특이값 비교

```python
# Compare singular values from full vs randomized SVD
A = create_low_rank_matrix(1000, 800, rank=30, noise=0.005)
k = 40

U_full, S_full, Vt_full = np.linalg.svd(A, full_matrices=False)
U_rand, S_rand, Vt_rand = randomized_svd(A, k, p=10, n_iter=2)

plt.figure(figsize=(10, 6))
plt.semilogy(S_full[:50], 'b-o', markersize=4, label='Full SVD')
plt.semilogy(range(k), S_rand, 'r-s', markersize=4, label='Randomized SVD')
plt.axvline(x=30, color='green', linestyle='--', alpha=0.5, label='True rank')
plt.xlabel('Index')
plt.ylabel('Singular value')
plt.title('Singular Values: Full vs Randomized SVD')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 5. 행렬 스케치

### 5.1 개요

**스케칭**은 행렬 $A \in \mathbb{R}^{m \times n}$을 핵심 성질을 근사적으로 보존하면서 훨씬 작은 행렬 $SA \in \mathbb{R}^{s \times n}$ (여기서 $s \ll m$)으로 압축합니다.

스케치 $S$는 다음이 될 수 있습니다:
- **가우시안**: $S_{ij} \sim \mathcal{N}(0, 1/s)$
- **CountSketch**: $A$의 각 행이 랜덤 부호와 함께 $SA$의 랜덤 행에 매핑
- **SRHT**: 서브샘플링된 랜덤화 Hadamard 변환

```python
def count_sketch(A, s):
    """CountSketch: hash-based sketching.

    Each row of A is hashed to one of s buckets with a random sign.
    Very fast: O(nnz(A)) time.
    """
    m, n = A.shape
    # Hash function: which bucket
    h = np.random.randint(0, s, m)
    # Sign function
    sigma = np.random.choice([-1, 1], m)

    SA = np.zeros((s, n))
    for i in range(m):
        SA[h[i]] += sigma[i] * A[i]
    return SA

# Compare sketching methods
np.random.seed(42)
m, n = 10000, 100
A = np.random.randn(m, n)
b = np.random.randn(m)

# True least squares solution
x_true = np.linalg.lstsq(A, b, rcond=None)[0]

sketch_sizes = [200, 500, 1000, 2000]
print(f"Original size: {m} x {n}")
print(f"{'Sketch size':>12s}  {'Gaussian error':>16s}  {'CountSketch error':>18s}")

for s in sketch_sizes:
    # Gaussian sketch
    S_gauss = np.random.randn(s, m) / np.sqrt(s)
    SA_gauss = S_gauss @ A
    Sb_gauss = S_gauss @ b
    x_gauss = np.linalg.lstsq(SA_gauss, Sb_gauss, rcond=None)[0]
    err_gauss = np.linalg.norm(x_gauss - x_true) / np.linalg.norm(x_true)

    # CountSketch
    SA_cs = count_sketch(A, s)
    # Need to apply same sketch to b
    h = np.random.randint(0, s, m)
    sigma = np.random.choice([-1, 1], m)
    Sb_cs = np.zeros(s)
    for i in range(m):
        Sb_cs[h[i]] += sigma[i] * b[i]
    x_cs = np.linalg.lstsq(SA_cs, Sb_cs, rcond=None)[0]
    err_cs = np.linalg.norm(x_cs - x_true) / np.linalg.norm(x_true)

    print(f"{s:>12d}  {err_gauss:>16.6f}  {err_cs:>18.6f}")
```

### 5.2 스트리밍 스케치 (Frequent Directions)

스트림으로 도착하는 데이터에 대해 **Frequent Directions** 알고리즘은 공분산 행렬을 근사하는 스케치를 유지합니다:

```python
def frequent_directions(stream, l):
    """Frequent Directions algorithm for streaming covariance estimation.

    Maintains an l x d sketch that approximates the top-l singular vectors.

    Args:
        stream: iterator yielding (d,) vectors one at a time
        l: sketch size (number of rows to maintain)
    """
    first = next(stream)
    d = len(first)
    B = np.zeros((l, d))
    B[0] = first

    row_idx = 1
    for x in stream:
        if row_idx < l:
            B[row_idx] = x
            row_idx += 1
        else:
            # Sketch is full: compress
            B[l - 1] = x
            U, S, Vt = np.linalg.svd(B, full_matrices=False)
            # Shrink: subtract the smallest squared singular value
            delta = S[l - 1]**2
            S_new = np.sqrt(np.maximum(S**2 - delta, 0))
            B = np.diag(S_new) @ Vt
            row_idx = np.sum(S_new > 0)

    return B

# Test: compare streaming sketch with full covariance
np.random.seed(42)
n_samples = 5000
d = 50
X = np.random.randn(n_samples, d) @ np.random.randn(d, d)  # Correlated data

# Full covariance
cov_full = X.T @ X / n_samples

# Streaming sketch
l = 15
sketch = frequent_directions(iter(X), l)
cov_sketch = sketch.T @ sketch / n_samples

# Compare top singular values
sv_full = np.linalg.svd(X, compute_uv=False)[:l]
sv_sketch = np.linalg.svd(sketch, compute_uv=False)

print(f"Top-{l} singular values comparison:")
print(f"{'Full':>10s}  {'Sketch':>10s}  {'Ratio':>8s}")
for i in range(min(l, len(sv_sketch))):
    if sv_sketch[i] > 0:
        print(f"{sv_full[i]:>10.2f}  {sv_sketch[i]:>10.2f}  {sv_sketch[i]/sv_full[i]:>8.3f}")
```

---

## 6. 랜덤 특징

### 6.1 랜덤 푸리에 특징

랜덤 푸리에 특징(Rahimi & Recht, 2007)은 커널 함수를 명시적 저차원 특징 맵으로 근사합니다:

$$k(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{y}\|^2}{2\sigma^2}\right) \approx \frac{1}{D} \sum_{i=1}^{D} \cos(\omega_i^T \mathbf{x} + b_i) \cos(\omega_i^T \mathbf{y} + b_i)$$

여기서 $\omega_i \sim \mathcal{N}(0, \sigma^{-2}I)$이고 $b_i \sim \text{Uniform}(0, 2\pi)$입니다.

```python
def random_fourier_features(X, D, sigma=1.0):
    """Approximate RBF kernel using random Fourier features.

    Args:
        X: (n, d) input data
        D: number of random features
        sigma: kernel bandwidth

    Returns:
        Z: (n, D) feature matrix such that Z Z^T ~ K
    """
    d = X.shape[1]
    omega = np.random.randn(d, D) / sigma
    b = np.random.uniform(0, 2 * np.pi, D)
    Z = np.sqrt(2.0 / D) * np.cos(X @ omega + b)
    return Z

# Compare: exact kernel vs random features approximation
from scipy.spatial.distance import cdist

np.random.seed(42)
n = 500
d = 10
sigma = 2.0
X = np.random.randn(n, d)

# Exact RBF kernel
K_exact = np.exp(-cdist(X, X, 'sqeuclidean') / (2 * sigma**2))

# Approximate with varying D
Ds = [10, 50, 100, 500, 2000]
fig, axes = plt.subplots(1, len(Ds), figsize=(20, 4))

for ax, D in zip(axes, Ds):
    Z = random_fourier_features(X, D, sigma)
    K_approx = Z @ Z.T
    error = np.linalg.norm(K_exact - K_approx) / np.linalg.norm(K_exact)
    ax.imshow(K_approx[:50, :50], cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f'D={D}\nerror={error:.4f}')
    ax.axis('off')

plt.suptitle('Kernel Approximation with Random Fourier Features', y=1.02)
plt.tight_layout()
plt.show()
```

### 6.2 대규모 커널 SVM을 위한 랜덤 특징

```python
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a nonlinearly separable dataset
X, y = make_classification(n_samples=5000, n_features=20,
                           n_informative=10, n_redundant=5,
                           n_clusters_per_class=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42)

# Linear SVM (baseline)
start = time.time()
svm_linear = LinearSVC(max_iter=5000, random_state=42)
svm_linear.fit(X_train, y_train)
t_linear = time.time() - start
acc_linear = accuracy_score(y_test, svm_linear.predict(X_test))

# SVM with random Fourier features
results = []
for D in [50, 100, 500, 1000]:
    start = time.time()
    # Create random features
    omega = np.random.randn(20, D) / 2.0
    b = np.random.uniform(0, 2 * np.pi, D)
    Z_train = np.sqrt(2.0 / D) * np.cos(X_train @ omega + b)
    Z_test = np.sqrt(2.0 / D) * np.cos(X_test @ omega + b)

    # Train linear SVM on features
    svm_rff = LinearSVC(max_iter=5000, random_state=42)
    svm_rff.fit(Z_train, y_train)
    t_rff = time.time() - start
    acc_rff = accuracy_score(y_test, svm_rff.predict(Z_test))
    results.append((D, acc_rff, t_rff))

print(f"Linear SVM: accuracy={acc_linear:.4f}, time={t_linear:.4f}s")
for D, acc, t in results:
    print(f"RFF (D={D:4d}): accuracy={acc:.4f}, time={t:.4f}s")
```

---

## 7. 확장성과 트레이드오프

### 7.1 오차-속도-메모리 트레이드오프

```python
# Comprehensive comparison: randomized SVD accuracy vs speed vs memory
m, n = 5000, 3000
rank = 50
A = create_low_rank_matrix(m, n, rank, noise=0.001)

# Full SVD (baseline)
start = time.time()
U_f, S_f, Vt_f = np.linalg.svd(A, full_matrices=False)
t_full = time.time() - start
A_best = U_f[:, :rank] @ np.diag(S_f[:rank]) @ Vt_f[:rank, :]
err_best = np.linalg.norm(A - A_best) / np.linalg.norm(A)

print(f"Full SVD: time={t_full:.2f}s, best rank-{rank} error={err_best:.8f}")
print(f"\n{'k':>4s}  {'p':>4s}  {'q':>4s}  {'Time (s)':>10s}  {'Rel error':>12s}  {'Speedup':>8s}")

for k in [20, 50, 80]:
    for p in [5, 10]:
        for q in [0, 1, 2]:
            start = time.time()
            U_r, S_r, Vt_r = randomized_svd(A, k, p=p, n_iter=q)
            t_r = time.time() - start
            A_r = U_r @ np.diag(S_r) @ Vt_r
            err_r = np.linalg.norm(A - A_r) / np.linalg.norm(A)
            print(f"{k:>4d}  {p:>4d}  {q:>4d}  {t_r:>10.4f}  {err_r:>12.8f}  {t_full/t_r:>7.1f}x")
```

### 7.2 랜덤화 방법의 사용 시기

**랜덤화 알고리즘을 사용해야 할 때**:
- 행렬이 전체 분해에 비해 너무 클 때
- 근사 랭크-$k$ 결과로 충분할 때
- 행렬의 특이값이 빠르게 감쇠할 때
- 스트리밍 또는 온라인 계산이 필요할 때
- 많은 행렬을 빠르게 처리해야 할 때 (예: 여러 데이터셋에 대한 PCA)

**결정론적 방법을 유지해야 할 때**:
- 높은 정밀도가 중요할 때 (예: 수치적 PDE 풀이기)
- 행렬이 정확한 계산에 충분히 작을 때
- 행렬이 근사적으로 저랭크가 아닐 때
- 모든 특이값/벡터가 필요할 때

```python
# Demonstrate when randomized methods struggle: flat spectrum
np.random.seed(42)
n = 1000

# Good case: rapid decay
sigma_good = np.logspace(0, -6, n)
A_good = np.random.randn(n, n) @ np.diag(sigma_good) @ np.random.randn(n, n)

# Bad case: flat spectrum
sigma_flat = np.ones(n) + 0.01 * np.random.randn(n)
A_flat = np.random.randn(n, n) @ np.diag(sigma_flat) @ np.random.randn(n, n)

k = 50
for name, A_test in [("Rapid decay", A_good), ("Flat spectrum", A_flat)]:
    U_f, S_f, Vt_f = np.linalg.svd(A_test, full_matrices=False)
    U_r, S_r, Vt_r = randomized_svd(A_test, k, p=10, n_iter=2)

    err_full = np.linalg.norm(A_test - U_f[:,:k] @ np.diag(S_f[:k]) @ Vt_f[:k,:]) / np.linalg.norm(A_test)
    err_rand = np.linalg.norm(A_test - U_r @ np.diag(S_r) @ Vt_r) / np.linalg.norm(A_test)

    print(f"{name}: full_err={err_full:.6f}, rand_err={err_rand:.6f}, "
          f"excess_err={err_rand - err_full:.2e}")
```

---

## 연습 문제

### 문제 1: 랜덤 투영 검증

1. $\mathbb{R}^{500}$에서 1000개의 점을 생성하시오
2. 원래 공간에서 모든 쌍별 거리를 계산하시오
3. 가우시안 랜덤 행렬을 사용하여 차원 $k = 10, 20, 50, 100, 200$으로 투영하시오
4. 최대 왜곡 $\max_{i,j} |d'_{ij}/d_{ij} - 1|$ 대 $k$를 그래프로 나타내시오
5. JL 바운드를 오버레이하고 성립하는지 검증하시오

### 문제 2: 랜덤화 SVD 구현

다음을 포함하여 랜덤화 SVD를 처음부터 구현하시오:
1. 가우시안 랜덤 샘플링
2. 선택적 멱급수 반복
3. 오버샘플링 매개변수

알려진 랭크-20 구조를 가진 $5000 \times 3000$ 행렬에서 테스트하시오. 특이값, 재구성 오차, 실행 시간을 `np.linalg.svd`와 비교하시오.

### 문제 3: 랜덤화 PCA

랜덤화 SVD를 사용하여 랜덤화 PCA를 구현하시오:
1. 데이터를 중심화하시오
2. 근사 상위-$k$ 특이벡터를 계산하시오
3. Olivetti 얼굴 데이터셋(400개 얼굴, 4096 픽셀)에 적용하시오
4. $k = 10, 50, 100$에 대해 전체 PCA와 재구성 품질을 비교하시오

### 문제 4: 랜덤 푸리에 특징

1. RBF 커널에 대한 랜덤 푸리에 특징을 구현하시오
2. $D = 10, 50, 200, 1000$에 대해 커널 행렬 근사 오차를 비교하시오
3. 랜덤 특징에 대한 선형 분류기를 훈련하고 커널 SVM과 정확도를 비교하시오

### 문제 5: 스트리밍 공분산

Frequent Directions 알고리즘을 구현하여 스트리밍 공분산을 추정하시오:
1. 50차원의 100,000개 샘플 데이터셋을 한 번에 하나씩 처리하시오
2. 크기 $l = 10, 20, 50$의 스케치를 유지하시오
3. 스케치의 상위 고유값을 전체 공분산 행렬의 고유값과 비교하시오

---

[이전: 18강](./18_Linear_Algebra_in_Computer_Graphics.md) | [개요](./00_Overview.md) | [다음: 20강](./20_Advanced_Topics.md)

**License**: CC BY-NC 4.0
