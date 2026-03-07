# 레슨 7: 특이값 분해 (Singular Value Decomposition)

## 학습 목표

- $A^T A$와 $A A^T$의 고유 분해로부터 SVD를 유도할 수 있다
- SVD를 회전-스케일-회전으로 기하학적으로 해석할 수 있다
- 최적 저랭크 근사에 대한 Eckart-Young 정리를 설명하고 적용할 수 있다
- 절단된 SVD를 사용하여 이미지 압축을 구현할 수 있다
- SVD를 유사역행렬, 랭크, 노름, 조건수와 연결할 수 있다

---

## 1. 정의와 유도

### 1.1 SVD 정리

모든 행렬 $A \in \mathbb{R}^{m \times n}$ (임의의 모양, 임의의 랭크)에 대해 **특이값 분해**가 존재합니다:

$$A = U \Sigma V^T$$

여기서:
- $U \in \mathbb{R}^{m \times m}$는 직교 행렬 (열은 **좌특이벡터**)
- $\Sigma \in \mathbb{R}^{m \times n}$는 **특이값** $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_{\min(m,n)} \ge 0$을 가진 대각 행렬
- $V \in \mathbb{R}^{n \times n}$는 직교 행렬 (열은 **우특이벡터**)

### 1.2 고유 분해로부터의 유도

대칭 양반정부호 행렬 $A^T A$와 $A A^T$를 고려합니다:

- $A^T A = V \Sigma^T \Sigma V^T$ -- 고유 분해로 $V$와 $\sigma_i^2$를 얻음
- $A A^T = U \Sigma \Sigma^T U^T$ -- 고유 분해로 $U$를 얻음

특이값은 $A^T A$ (또는 $A A^T$)의 고유값의 제곱근입니다:

$$\sigma_i = \sqrt{\lambda_i(A^T A)}$$

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[3, 2, 2],
              [2, 3, -2]], dtype=float)

# Full SVD
U, s, Vt = np.linalg.svd(A)
Sigma = np.zeros_like(A)
np.fill_diagonal(Sigma, s)

print(f"U ({U.shape}):\n{np.round(U, 4)}")
print(f"Singular values: {s}")
print(f"V^T ({Vt.shape}):\n{np.round(Vt, 4)}")

# Verify A = U Sigma V^T
A_reconstructed = U @ Sigma @ Vt
print(f"\nU @ Sigma @ V^T:\n{np.round(A_reconstructed, 10)}")
print(f"Matches A? {np.allclose(A, A_reconstructed)}")

# Verify: singular values = sqrt(eigenvalues of A^T A)
eigvals_ATA = np.linalg.eigvalsh(A.T @ A)
print(f"\neigenvalues of A^T A: {np.sort(eigvals_ATA)[::-1]}")
print(f"sigma^2:              {s**2}")
```

### 1.3 축약(경제적) SVD

$m > n$일 때, **축약 SVD** $A = U_r \Sigma_r V_r^T$를 사용할 수 있습니다. 여기서 $U_r \in \mathbb{R}^{m \times r}$, $\Sigma_r \in \mathbb{R}^{r \times r}$, $V_r \in \mathbb{R}^{n \times r}$이고 $r = \mathrm{rank}(A)$입니다.

```python
# Economy SVD
U_econ, s_econ, Vt_econ = np.linalg.svd(A, full_matrices=False)
print(f"Economy U shape: {U_econ.shape}")    # (2, 2) instead of (2, 2)
print(f"Economy V^T shape: {Vt_econ.shape}")  # (2, 3) instead of (3, 3)
```

---

## 2. 기하학적 해석

### 2.1 세 가지 변환으로서의 SVD

SVD는 임의의 선형 변환이 세 단계로 분해될 수 있음을 보여줍니다:

1. 입력 공간을 **회전** ($V^T$로)
2. 축을 따라 **스케일링** ($\Sigma$로)
3. 출력 공간으로 **회전** ($U$로)

$\mathbb{R}^n$의 단위 구는 $\mathbb{R}^m$의 타원체로 매핑됩니다. 특이값은 타원체의 반축 길이를 나타냅니다.

```python
# Visualize SVD of a 2x2 matrix
A = np.array([[2, 1],
              [1, 3]], dtype=float)

U, s, Vt = np.linalg.svd(A)

# Unit circle
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Step 0: Original unit circle
axes[0].plot(circle[0], circle[1], 'b-')
axes[0].set_title('Unit circle')

# Step 1: After V^T rotation
step1 = Vt @ circle
axes[1].plot(step1[0], step1[1], 'g-')
axes[1].set_title(r'After $V^T$ (rotation)')

# Step 2: After Sigma scaling
step2 = np.diag(s) @ step1
axes[2].plot(step2[0], step2[1], 'orange')
axes[2].set_title(r'After $\Sigma$ (scaling)')

# Step 3: After U rotation = A @ circle
step3 = U @ step2
axes[3].plot(step3[0], step3[1], 'r-')
axes[3].set_title(r'After $U$ (rotation) = $A \cdot$ circle')

for ax in axes:
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

print(f"Semi-axis lengths: {s}")
```

### 2.2 주축으로서의 특이벡터

- **우특이벡터** ($V$의 열)는 입력 공간의 주방향
- **좌특이벡터** ($U$의 열)는 출력 공간의 주방향
- 특이값 $\sigma_i$는 각 주방향을 따른 늘임 배율

---

## 3. 저랭크 근사 (Low-Rank Approximation)

### 3.1 절단된 SVD (Truncated SVD)

**랭크-$k$ 절단 SVD**는 가장 큰 $k$개의 특이값만 유지합니다:

$$A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

여기서 $U_k$는 $U$의 처음 $k$개 열 등입니다.

### 3.2 Eckart-Young 정리

절단된 SVD는 Frobenius 노름과 연산자 2-노름 모두에서 $A$에 대한 **최적의 랭크-$k$ 근사**를 제공합니다:

$$A_k = \arg\min_{\mathrm{rank}(B) \le k} \|A - B\|_F$$

근사 오차는 다음과 같습니다:

$$\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2}$$

$$\|A - A_k\|_2 = \sigma_{k+1}$$

```python
# Create a matrix and approximate it
np.random.seed(42)
A = np.random.randn(10, 8)

U, s, Vt = np.linalg.svd(A, full_matrices=False)
print(f"Singular values: {np.round(s, 3)}")

# Rank-k approximations
for k in [1, 2, 3, 5]:
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    error_fro = np.linalg.norm(A - A_k, 'fro')
    error_theory = np.sqrt(np.sum(s[k:]**2))
    print(f"Rank-{k}: ||A - A_k||_F = {error_fro:.4f} (theory: {error_theory:.4f})")

# Cumulative energy retained
energy = np.cumsum(s**2) / np.sum(s**2)
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(s)+1), s**2 / np.sum(s**2), alpha=0.5, label='Individual')
plt.plot(range(1, len(s)+1), energy, 'ro-', label='Cumulative')
plt.xlabel('Singular value index')
plt.ylabel('Fraction of total energy')
plt.title('Singular Value Energy Distribution')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

---

## 4. SVD를 이용한 이미지 압축

SVD의 가장 직관적인 응용 중 하나입니다: 더 적은 수의 특이값으로 이미지를 근사합니다.

```python
# Create a simple test image (or use a real one)
from scipy.datasets import face

# Use a grayscale image
try:
    img = face(gray=True).astype(float)
except:
    # Fallback: create a synthetic image
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    img = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2*X + Y)

print(f"Image shape: {img.shape}")

# SVD of the image
U, s, Vt = np.linalg.svd(img, full_matrices=False)
print(f"Number of singular values: {len(s)}")

# Reconstruct with different ranks
ranks = [1, 5, 20, 50, 100]
fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(20, 4))

axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'Original\n({img.shape[0]}x{img.shape[1]})')
axes[0].axis('off')

for ax, k in zip(axes[1:], ranks):
    img_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    compression = (k * (img.shape[0] + img.shape[1] + 1)) / (img.shape[0] * img.shape[1])
    ax.imshow(img_k, cmap='gray')
    ax.set_title(f'Rank {k}\n({compression:.1%} storage)')
    ax.axis('off')

plt.suptitle('Image Compression via Truncated SVD', fontsize=14)
plt.tight_layout()
plt.show()

# Storage comparison
m, n = img.shape
for k in ranks:
    original_entries = m * n
    compressed_entries = k * (m + n + 1)
    ratio = compressed_entries / original_entries
    print(f"Rank {k}: {compressed_entries} entries ({ratio:.1%} of original)")
```

---

## 5. SVD와 유사역행렬 (Pseudoinverse)

### 5.1 Moore-Penrose 유사역행렬

**유사역행렬** (또는 Moore-Penrose 역행렬) $A^+$는 SVD를 통해 정의됩니다:

$$A^+ = V \Sigma^+ U^T$$

여기서 $\Sigma^+$는 각 0이 아닌 특이값의 역수를 취하여 형성됩니다:

$$\Sigma^+ = \mathrm{diag}(1/\sigma_1, \ldots, 1/\sigma_r, 0, \ldots, 0)^T$$

### 5.2 성질

- $A$가 가역이면: $A^+ = A^{-1}$
- $A A^+ A = A$
- $A^+ A A^+ = A^+$
- $A^+ \mathbf{b}$는 $A\mathbf{x} = \mathbf{b}$의 최소 노름 최소자승해를 줌

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=float)

# Pseudoinverse via NumPy
A_pinv = np.linalg.pinv(A)
print(f"A^+ =\n{np.round(A_pinv, 4)}")

# Verify via SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1 / s)
A_pinv_svd = Vt.T @ S_inv @ U.T
print(f"A^+ (via SVD) =\n{np.round(A_pinv_svd, 4)}")
print(f"Match? {np.allclose(A_pinv, A_pinv_svd)}")

# Verify properties
print(f"\nA @ A^+ @ A == A? {np.allclose(A @ A_pinv @ A, A)}")
print(f"A^+ @ A @ A^+ == A^+? {np.allclose(A_pinv @ A @ A_pinv, A_pinv)}")

# Least squares solution via pseudoinverse
b = np.array([1, 2, 3], dtype=float)
x_pinv = A_pinv @ b
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"\nx (pseudoinverse): {x_pinv}")
print(f"x (lstsq):         {x_lstsq}")
print(f"Match? {np.allclose(x_pinv, x_lstsq)}")
```

---

## 6. SVD와 행렬 성질

### 6.1 랭크

$A$의 랭크는 0이 아닌 특이값의 개수와 같습니다:

$$\mathrm{rank}(A) = \#\{i : \sigma_i > 0\}$$

### 6.2 노름

- **연산자 2-노름**: $\|A\|_2 = \sigma_1$ (최대 특이값)
- **Frobenius 노름**: $\|A\|_F = \sqrt{\sigma_1^2 + \cdots + \sigma_r^2}$

### 6.3 조건수 (Condition Number)

**조건수**는 $A\mathbf{x} = \mathbf{b}$가 섭동에 얼마나 민감한지를 측정합니다:

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{\sigma_1}{\sigma_r}$$

조건수가 크면 행렬이 **악조건(ill-conditioned)**이라 합니다.

```python
# Well-conditioned matrix
A_good = np.array([[1, 0], [0, 1]], dtype=float)

# Ill-conditioned matrix
A_bad = np.array([[1, 1], [1, 1.0001]], dtype=float)

for name, M in [("Well-conditioned", A_good), ("Ill-conditioned", A_bad)]:
    s = np.linalg.svd(M, compute_uv=False)
    print(f"\n{name}:")
    print(f"  Singular values: {s}")
    print(f"  Rank: {np.sum(s > 1e-10)}")
    print(f"  ||A||_2 = {s[0]:.6f}")
    print(f"  ||A||_F = {np.sqrt(np.sum(s**2)):.6f}")
    print(f"  Condition number: {s[0]/s[-1]:.2f}")

# Demonstration: ill-conditioning amplifies errors
b = np.array([2, 2.0001])
x = np.linalg.solve(A_bad, b)
print(f"\nSolution to ill-conditioned system: {x}")

b_perturbed = b + np.array([0.0001, 0])
x_perturbed = np.linalg.solve(A_bad, b_perturbed)
print(f"Perturbed solution: {x_perturbed}")
print(f"Relative change in b: {np.linalg.norm(b - b_perturbed) / np.linalg.norm(b):.6f}")
print(f"Relative change in x: {np.linalg.norm(x - x_perturbed) / np.linalg.norm(x):.6f}")
```

---

## 7. 랭크-1 행렬의 합으로서의 SVD

SVD는 $A$를 가중된 랭크-1 외적의 합으로 표현합니다:

$$A = \sum_{i=1}^r \sigma_i \, \mathbf{u}_i \mathbf{v}_i^T$$

각 항 $\sigma_i \mathbf{u}_i \mathbf{v}_i^T$는 랭크-1 행렬이며, 특이값이 각 성분의 "중요도"를 결정합니다.

```python
A = np.array([[3, 2, 2],
              [2, 3, -2]], dtype=float)

U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Build A as sum of rank-1 terms
A_sum = np.zeros_like(A)
r = len(s)

fig, axes = plt.subplots(1, r + 1, figsize=(4 * (r + 1), 3))

for i in range(r):
    rank1 = s[i] * np.outer(U[:, i], Vt[i, :])
    A_sum += rank1
    axes[i].imshow(rank1, cmap='RdBu', vmin=-4, vmax=4)
    axes[i].set_title(f'sigma_{i+1}={s[i]:.2f}\nRank-1 term')
    axes[i].axis('off')

axes[r].imshow(A_sum, cmap='RdBu', vmin=-4, vmax=4)
axes[r].set_title('Sum = A')
axes[r].axis('off')

plt.suptitle('SVD: A as sum of rank-1 matrices')
plt.tight_layout()
plt.show()

print(f"Reconstruction matches A? {np.allclose(A, A_sum)}")
```

---

## 8. 실용적 고려사항

### 8.1 수치적 랭크

실제로는 부동소수점 연산 때문에 특이값이 정확히 0이 되지 않습니다. **수치적 랭크**는 임계값 (보통 $\epsilon \cdot \sigma_1$, 여기서 $\epsilon$은 기계 엡실론) 이상인 특이값의 수입니다.

```python
# Matrix that is "nearly" rank-deficient
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9.0001]])  # almost singular

s = np.linalg.svd(A, compute_uv=False)
print(f"Singular values: {s}")
print(f"Rank (tol=1e-10): {np.sum(s > 1e-10)}")
print(f"Rank (tol=1e-3):  {np.sum(s > 1e-3)}")
print(f"NumPy rank:       {np.linalg.matrix_rank(A)}")
```

### 8.2 대규모 행렬을 위한 랜덤화 SVD

매우 큰 행렬의 경우, 전체 SVD 계산은 비용이 높습니다 ($O(mn \min(m,n))$). 랜덤화 SVD는 훨씬 빠르게 근사적인 랭크-$k$ 분해를 제공합니다.

```python
from scipy.linalg import svd as scipy_svd

def randomized_svd(A, k, n_oversamples=10, n_iter=2):
    """Compute approximate rank-k SVD using randomized algorithm."""
    m, n = A.shape

    # Step 1: Random projection
    Omega = np.random.randn(n, k + n_oversamples)
    Y = A @ Omega

    # Step 2: Power iteration for better approximation
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    # Step 3: Orthonormalize
    Q, _ = np.linalg.qr(Y)

    # Step 4: Project and compute SVD of small matrix
    B = Q.T @ A
    U_hat, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat

    return U[:, :k], s[:k], Vt[:k, :]

# Test on a large-ish matrix
np.random.seed(42)
A_large = np.random.randn(500, 300)

# Full SVD
U_full, s_full, Vt_full = np.linalg.svd(A_large, full_matrices=False)

# Randomized SVD (rank 20)
k = 20
U_rand, s_rand, Vt_rand = randomized_svd(A_large, k)

print(f"Top {k} singular values (full):       {np.round(s_full[:k], 3)}")
print(f"Top {k} singular values (randomized): {np.round(s_rand, 3)}")
print(f"Relative error: {np.linalg.norm(s_full[:k] - s_rand) / np.linalg.norm(s_full[:k]):.6f}")
```

---

## 9. 요약

| 개념 | 설명 |
|------|------|
| SVD | $A = U\Sigma V^T$ (임의의 행렬) |
| 특이값 | $\sigma_i = \sqrt{\lambda_i(A^TA)}$, $\sigma_1 \ge \sigma_2 \ge \cdots$ 순서 |
| 기하학적 의미 | 회전 ($V^T$) -- 스케일링 ($\Sigma$) -- 회전 ($U$) |
| 절단 SVD | 최적 랭크-$k$ 근사 (Eckart-Young) |
| 유사역행렬 | $A^+ = V\Sigma^+ U^T$ |
| 조건수 | $\kappa = \sigma_1 / \sigma_r$ |
| NumPy | `np.linalg.svd(A)` 또는 `np.linalg.svd(A, full_matrices=False)` |

---

## 연습 문제

### 연습 문제 1: 수동 SVD 계산

$A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$의 SVD를 손으로 계산하세요. NumPy로 검증하세요.

### 연습 문제 2: 저랭크 근사

$100 \times 100$이고 랭크가 5인 행렬을 생성하세요 ($100 \times 5$와 $5 \times 100$ 행렬의 곱에 작은 노이즈를 더하여). SVD를 사용하여 랭크-5 구조를 복원하세요. 특이값과 근사 오차를 랭크의 함수로 플롯하세요.

### 연습 문제 3: 이미지 압축

$256 \times 256$ 회색조 이미지를 불러오거나 생성하세요. 절단된 SVD를 사용하여 랭크 1, 5, 10, 25, 50, 100에서 압축하세요. 각 랭크에 대해 압축률과 상대 Frobenius 오차를 계산하세요. 두 값을 플롯하세요.

### 연습 문제 4: 조건수

$n = 10, 50, 100, 200$에 대해 랜덤 $n \times n$ 행렬을 생성하고 조건수를 계산하세요. 같은 크기의 힐베르트 행렬 (`scipy.linalg.hilbert(n)`)도 생성하세요. 조건수를 비교하고 연립방정식 풀이에 대한 시사점을 논의하세요.

### 연습 문제 5: 유사역행렬

행렬 $A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$에 대해, SVD를 사용하여 $A^+$를 계산하세요. 이를 사용하여 $A\mathbf{x} = [1, 2, 2]^T$의 최소자승해를 구하세요. 이 해가 $\|A\mathbf{x} - \mathbf{b}\|_2$를 최소화하는지 검증하세요.

---

[<< 이전: 레슨 6 - 고유값과 고유벡터](06_Eigenvalues_and_Eigenvectors.md) | [개요](00_Overview.md) | [다음: 레슨 8 - 주성분 분석 >>](08_Principal_Component_Analysis.md)

**License**: CC BY-NC 4.0
