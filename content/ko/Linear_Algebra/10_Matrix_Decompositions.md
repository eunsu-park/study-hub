# 레슨 10: 행렬 분해

## 학습 목표

- 대칭 양정부호 행렬에 대한 Cholesky 분해를 계산하고 적용할 수 있습니다
- Cholesky의 변형인 LDL^T 분해를 이해하고 계산할 수 있습니다
- Schur 분해와 고유값과의 관계를 설명할 수 있습니다
- 극분해(polar decomposition)와 그 기하학적 의미를 설명할 수 있습니다
- 비용, 안정성, 적용 가능성 및 사용 사례별로 분해를 비교할 수 있습니다

---

## 1. 분해의 개요

행렬 분해(또는 인수분해)는 수치 선형대수의 핵심 도구입니다. 각 분해는 서로 다른 구조적 성질을 드러내며 특정 작업에 적합합니다.

| 분해 | 형태 | 요구 조건 | 주요 용도 |
|------|------|----------|----------|
| LU | $PA = LU$ | 정방 (일반) | 선형 시스템 풀기 |
| QR | $A = QR$ | 완전 열랭크 | 최소제곱, 고유값 |
| 고유분해 | $A = P\Lambda P^{-1}$ | 정방, 대각화 가능 | 스펙트럼 분석 |
| SVD | $A = U\Sigma V^T$ | 모든 행렬 | 랭크, 유사역행렬, 압축 |
| Cholesky | $A = LL^T$ | 대칭 양정부호 | 빠른 SPD 풀기, 샘플링 |
| LDL^T | $A = LDL^T$ | 대칭 | SPD 검사, 부정부호 시스템 |
| Schur | $A = QTQ^T$ | 정방 | 고유값, 행렬 함수 |
| 극분해 | $A = UP$ | 모든 행렬 | 회전 추출 |

---

## 2. Cholesky 분해

### 2.1 정의

**대칭 양정부호**(SPD) 행렬 $A$에 대해 **Cholesky 분해**는 다음과 같습니다:

$$A = LL^T$$

여기서 $L$은 **양의 대각 원소**를 가진 **하삼각** 행렬입니다. 이것은 유일합니다.

### 2.2 Cholesky를 사용하는 이유

- LU 분해보다 **2배 빠릅니다** (약 $\frac{1}{3}n^3$ vs $\frac{2}{3}n^3$ 플롭)
- 피봇팅 없이 **수치적으로 안정적**입니다 (SPD 행렬의 경우)
- **SPD 검사**: $A$가 SPD인 것은 Cholesky 분해가 성공하는 것과 동치입니다

### 2.3 알고리즘

```python
import numpy as np
from scipy import linalg

def cholesky_manual(A):
    """Compute the Cholesky decomposition A = L L^T."""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    for j in range(n):
        # Diagonal entry
        val = A[j, j] - np.sum(L[j, :j]**2)
        if val <= 0:
            raise ValueError("Matrix is not positive definite")
        L[j, j] = np.sqrt(val)

        # Below-diagonal entries
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

    return L

# Create an SPD matrix
np.random.seed(42)
B = np.random.randn(4, 4)
A = B.T @ B + 0.1 * np.eye(4)  # guarantee SPD

print(f"A is symmetric? {np.allclose(A, A.T)}")
print(f"Eigenvalues: {np.linalg.eigvalsh(A)}")  # all positive

# Manual implementation
L_manual = cholesky_manual(A)
print(f"\nL (manual):\n{np.round(L_manual, 4)}")
print(f"L L^T == A? {np.allclose(L_manual @ L_manual.T, A)}")

# NumPy implementation
L_numpy = np.linalg.cholesky(A)
print(f"\nL (numpy):\n{np.round(L_numpy, 4)}")
print(f"Match? {np.allclose(L_manual, L_numpy)}")
```

### 2.4 Cholesky를 사용한 시스템 풀기

$A$가 SPD인 $A\mathbf{x} = \mathbf{b}$에 대해:

1. $A = LL^T$ 계산
2. $L\mathbf{y} = \mathbf{b}$ 풀기 (전진 대입)
3. $L^T\mathbf{x} = \mathbf{y}$ 풀기 (후진 대입)

```python
b = np.array([1, 2, 3, 4], dtype=float)

# Solve using Cholesky
L = np.linalg.cholesky(A)
y = linalg.solve_triangular(L, b, lower=True)
x = linalg.solve_triangular(L.T, y, lower=False)

print(f"Solution: {x}")
print(f"Verification: Ax = {A @ x}")
print(f"Match b? {np.allclose(A @ x, b)}")

# Compare with direct solve
x_direct = np.linalg.solve(A, b)
print(f"Direct solve: {x_direct}")
print(f"Match? {np.allclose(x, x_direct)}")
```

### 2.5 응용: 다변량 정규분포에서 샘플링

$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$에서 샘플링하려면:

1. $\Sigma = LL^T$이 되는 $L$을 계산합니다
2. $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$를 샘플링합니다
3. $\mathbf{x} = \boldsymbol{\mu} + L\mathbf{z}$로 설정합니다

```python
import matplotlib.pyplot as plt

# Desired distribution parameters
mu = np.array([2, 3])
Sigma = np.array([[2, 1.5],
                   [1.5, 3]])

# Cholesky decomposition
L = np.linalg.cholesky(Sigma)

# Sample
n_samples = 1000
Z = np.random.randn(n_samples, 2)
X = Z @ L.T + mu  # equivalent to mu + L @ z for each sample

print(f"Sample mean: {X.mean(axis=0)} (expected: {mu})")
print(f"Sample covariance:\n{np.round(np.cov(X.T), 3)} (expected:\n{Sigma})")

plt.figure(figsize=(7, 7))
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, s=5)
plt.plot(mu[0], mu[1], 'r+', markersize=15, markeredgewidth=3, label='True mean')
plt.xlabel('x1'); plt.ylabel('x2')
plt.title('Samples from Multivariate Normal via Cholesky')
plt.legend(); plt.grid(True, alpha=0.3); plt.axis('equal')
plt.show()
```

---

## 3. LDL^T 분해

### 3.1 정의

**대칭** 행렬 $A$ (양정부호가 아니어도 됨)에 대해:

$$A = LDL^T$$

여기서 $L$은 단위 하삼각 행렬(대각에 1)이고 $D$는 대각 행렬입니다.

### 3.2 Cholesky와의 관계

$A$가 SPD이면 $D$의 모든 원소가 양수이고:

$$A = LDL^T = L\sqrt{D}\sqrt{D}L^T = (L\sqrt{D})(L\sqrt{D})^T$$

따라서 Cholesky 인자는 $\tilde{L} = L\sqrt{D}$입니다.

### 3.3 장점

- Cholesky와 달리 **부정부호** 대칭 행렬에도 작동합니다
- Cholesky와 달리 제곱근을 피합니다
- $D$의 부호를 확인하여 $A$가 양정부호인지 판별할 수 있습니다

```python
def ldlt_decomposition(A):
    """Compute A = L D L^T for symmetric A."""
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros(n)

    for j in range(n):
        D[j] = A[j, j] - sum(L[j, k]**2 * D[k] for k in range(j))

        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - sum(L[i, k] * L[j, k] * D[k] for k in range(j))) / D[j]

    return L, D

# SPD matrix
A_spd = np.array([[4, 2, 1],
                   [2, 5, 3],
                   [1, 3, 6]], dtype=float)

L, D = ldlt_decomposition(A_spd)
print(f"L:\n{np.round(L, 4)}")
print(f"D: {D}")
print(f"L D L^T:\n{np.round(L @ np.diag(D) @ L.T, 10)}")
print(f"Matches A? {np.allclose(L @ np.diag(D) @ L.T, A_spd)}")
print(f"All D > 0 (SPD)? {np.all(D > 0)}")

# Indefinite symmetric matrix
A_indef = np.array([[ 2,  1, -1],
                     [ 1, -3,  2],
                     [-1,  2,  1]], dtype=float)
A_indef = (A_indef + A_indef.T) / 2  # ensure symmetric

L_i, D_i = ldlt_decomposition(A_indef)
print(f"\nIndefinite matrix D: {D_i}")
print(f"Positive definite? {np.all(D_i > 0)}")
print(f"Reconstruction matches? {np.allclose(L_i @ np.diag(D_i) @ L_i.T, A_indef)}")
```

### 3.4 SciPy의 LDL^T

```python
# SciPy provides LDL^T with pivoting
from scipy.linalg import ldl

L_sp, D_sp, perm = ldl(A_spd)
print(f"SciPy L:\n{np.round(L_sp, 4)}")
print(f"SciPy D:\n{np.round(D_sp, 4)}")

# Reconstruction (perm is a permutation)
print(f"Matches A? {np.allclose(L_sp @ D_sp @ L_sp.T, A_spd)}")
```

---

## 4. Schur 분해

### 4.1 정의

모든 정방 행렬 $A \in \mathbb{R}^{n \times n}$는 **Schur 분해**를 가집니다:

$$A = QTQ^T \quad \text{(실수 Schur 형)}$$

여기서 $Q$는 직교 행렬이고 $T$는 **준상삼각** 행렬입니다 (복소 고유값 쌍에 대한 $2 \times 2$ 대각 블록이 있을 수 있는 상삼각).

복소수를 허용하면:

$$A = QTQ^* \quad \text{(복소 Schur 형)}$$

여기서 $T$는 상삼각이고 대각 원소가 $A$의 고유값입니다.

### 4.2 Schur를 사용하는 이유

- 모든 정방 행렬에 대해 **항상 존재**합니다 (대각화와 달리)
- 고유값이 $T$의 대각에 나타납니다
- 수치적으로 안정적입니다 (직교 변환 사용)
- 고유값, 행렬 함수, 행렬 방정식 계산의 기초입니다

```python
from scipy.linalg import schur

A = np.array([[1, 2, 3],
              [0, 4, 5],
              [0, 0, 6]], dtype=float)

# Real Schur decomposition
T, Q = schur(A, output='real')
print(f"T (quasi-upper triangular):\n{np.round(T, 4)}")
print(f"Q (orthogonal):\n{np.round(Q, 4)}")
print(f"Q^T Q = I? {np.allclose(Q.T @ Q, np.eye(3))}")
print(f"Q T Q^T = A? {np.allclose(Q @ T @ Q.T, A)}")

# Eigenvalues on the diagonal
print(f"Diagonal of T: {np.diag(T)}")
print(f"Eigenvalues:    {np.linalg.eigvals(A)}")

# Complex Schur for a matrix with complex eigenvalues
B = np.array([[0, -1],
              [1,  0]], dtype=float)  # 90-degree rotation

T_c, Q_c = schur(B, output='complex')
print(f"\nComplex Schur T:\n{np.round(T_c, 4)}")
print(f"Eigenvalues: {np.diag(T_c)}")
```

### 4.3 행렬 함수를 위한 Schur

Schur 분해는 행렬 함수(지수, 로그, 제곱근)를 계산하는 데 사용됩니다:

$$f(A) = Q \, f(T) \, Q^T$$

$T$가 (준)삼각이므로 $f(T)$의 계산이 $f(A)$를 직접 계산하는 것보다 훨씬 간단합니다.

```python
from scipy.linalg import expm, sqrtm

A = np.array([[1, 0.5],
              [0.5, 2]], dtype=float)

# Matrix exponential via Schur
T, Q = schur(A)
# For a diagonal T, exp(T) is just exp of diagonal entries
# For triangular T, use Parlett's algorithm (what scipy does internally)
exp_A = expm(A)
print(f"exp(A) =\n{np.round(exp_A, 4)}")

# Matrix square root
sqrt_A = sqrtm(A)
print(f"\nsqrt(A) =\n{np.round(sqrt_A, 4)}")
print(f"sqrt(A)^2 = A? {np.allclose(sqrt_A @ sqrt_A, A)}")
```

---

## 5. 극분해 (Polar Decomposition)

### 5.1 정의

모든 행렬 $A \in \mathbb{R}^{m \times n}$ ($m \ge n$)는 다음과 같이 쓸 수 있습니다:

$$A = UP$$

여기서:
- $U \in \mathbb{R}^{m \times n}$는 정규직교 열을 가집니다 ($U^TU = I_n$)
- $P \in \mathbb{R}^{n \times n}$는 대칭 양의 준정부호 행렬입니다

### 5.2 기하학적 해석

극분해는 선형 변환을 다음으로 분리합니다:
1. **늘림**(stretch, $P$에 의한) -- 대칭이며, 모양을 변경하지만 방향은 변경하지 않습니다
2. **회전/반사**(rotation/reflection, $U$에 의한) -- 직교이며, 모양을 보존합니다

이것은 극좌표 $z = re^{i\theta}$에서 크기와 위상을 분리하는 것과 유사합니다.

### 5.3 SVD를 통한 계산

$A = U_s \Sigma V_s^T$가 주어졌을 때:

$$U = U_s V_s^T, \quad P = V_s \Sigma V_s^T$$

```python
A = np.array([[2, 1],
              [1, 3]], dtype=float)

# Compute via SVD
U_svd, s, Vt_svd = np.linalg.svd(A)

U_polar = U_svd @ Vt_svd
P_polar = Vt_svd.T @ np.diag(s) @ Vt_svd

print(f"U (orthogonal part):\n{np.round(U_polar, 4)}")
print(f"P (SPD part):\n{np.round(P_polar, 4)}")

# Verify
print(f"\nU @ P = A? {np.allclose(U_polar @ P_polar, A)}")
print(f"U^T U = I? {np.allclose(U_polar.T @ U_polar, np.eye(2))}")
print(f"P = P^T? {np.allclose(P_polar, P_polar.T)}")
print(f"P eigenvalues: {np.linalg.eigvalsh(P_polar)}")  # all >= 0

# SciPy polar decomposition
from scipy.linalg import polar
U_sp, P_sp = polar(A)
print(f"\nSciPy U:\n{np.round(U_sp, 4)}")
print(f"SciPy P:\n{np.round(P_sp, 4)}")
```

### 5.4 극분해 시각화

```python
# Show how a transformation decomposes into stretch + rotation
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Original unit circle
axes[0].plot(circle[0], circle[1], 'b-', linewidth=2)
axes[0].set_title('Unit circle')

# After P (stretch)
stretched = P_polar @ circle
axes[1].plot(stretched[0], stretched[1], 'g-', linewidth=2)
axes[1].set_title('After P (stretch)')

# After U rotation of stretched
final = U_polar @ stretched
axes[2].plot(final[0], final[1], 'r-', linewidth=2)
axes[2].set_title('After U (rotation) = A @ circle')

# Direct A @ circle (should match)
direct = A @ circle
axes[3].plot(direct[0], direct[1], 'r--', linewidth=2)
axes[3].set_title('A @ circle (direct)')

for ax in axes:
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5); ax.axvline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()
```

---

## 6. 분해의 비교

### 6.1 용도별 선택

| 작업 | 최적 분해 | 이유 |
|------|----------|------|
| $A\mathbf{x} = \mathbf{b}$ 풀기 (일반) | LU | 표준 접근법 |
| $A\mathbf{x} = \mathbf{b}$ 풀기 (SPD) | Cholesky | 2배 빠름, 항상 안정적 |
| 최소제곱 | QR 또는 SVD | 수치적으로 안정적 |
| 고유값 | Schur 후 추출 | 수치적으로 견고 |
| 저랭크 근사 | SVD | 최적 (Eckart-Young) |
| 양정부호 판별 | Cholesky 또는 LDL^T | SPD가 아니면 실패 |
| 샘플링 (MVN) | Cholesky | $\Sigma = LL^T$, 그런 다음 $x = \mu + Lz$ |
| 회전 추출 | 극분해 | $A = UP$로 회전과 늘림을 분리 |
| 행렬 함수 | Schur | $f(A) = Qf(T)Q^T$ |
| 유사역행렬 | SVD | $A^+ = V\Sigma^+ U^T$ |

### 6.2 계산 비용

| 분해 | 비용 (플롭) | 요구 조건 |
|------|------------|----------|
| LU | $\frac{2}{3}n^3$ | 정방 |
| Cholesky | $\frac{1}{3}n^3$ | SPD |
| QR (Householder) | $\frac{2}{3}n^3$ | 모든 행렬 |
| SVD | $\sim 4n^3$ (전체) | 모든 행렬 |
| 고유분해 | $\sim 9n^3$ | 정방 |
| Schur | $\sim 9n^3$ | 정방 |

```python
import time

sizes = [100, 200, 500, 1000]
results = {name: [] for name in ['LU', 'Cholesky', 'QR', 'SVD', 'Eig']}

for n in sizes:
    np.random.seed(42)
    B = np.random.randn(n, n)
    A = B.T @ B + np.eye(n)  # SPD

    # LU
    t0 = time.perf_counter()
    linalg.lu(A)
    results['LU'].append(time.perf_counter() - t0)

    # Cholesky
    t0 = time.perf_counter()
    np.linalg.cholesky(A)
    results['Cholesky'].append(time.perf_counter() - t0)

    # QR
    t0 = time.perf_counter()
    np.linalg.qr(A)
    results['QR'].append(time.perf_counter() - t0)

    # SVD
    t0 = time.perf_counter()
    np.linalg.svd(A)
    results['SVD'].append(time.perf_counter() - t0)

    # Eigendecomposition
    t0 = time.perf_counter()
    np.linalg.eigh(A)
    results['Eig'].append(time.perf_counter() - t0)

# Plot
plt.figure(figsize=(10, 6))
for name, times in results.items():
    plt.plot(sizes, times, 'o-', label=name, linewidth=2)
plt.xlabel('Matrix size n')
plt.ylabel('Time (seconds)')
plt.title('Decomposition Timing Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()
```

---

## 7. 안정성 및 수치적 고려사항

### 7.1 조건수와 분해 선택

```python
# Create matrices with different condition numbers
cond_numbers = [1, 10, 1e4, 1e8, 1e12]

print(f"{'Condition':>12} {'LU error':>12} {'Cholesky err':>14} {'QR error':>12} {'SVD error':>12}")
print("-" * 64)

for kappa in cond_numbers:
    n = 50
    np.random.seed(42)

    # Create SPD matrix with desired condition number
    U, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.logspace(0, -np.log10(kappa), n)
    A = U @ np.diag(s) @ U.T

    x_true = np.random.randn(n)
    b = A @ x_true

    # Solve with different methods
    x_lu = linalg.solve(A, b)
    L_chol = np.linalg.cholesky(A)
    y = linalg.solve_triangular(L_chol, b, lower=True)
    x_chol = linalg.solve_triangular(L_chol.T, y, lower=False)
    Q, R = np.linalg.qr(A)
    x_qr = linalg.solve_triangular(R, Q.T @ b)
    x_svd = np.linalg.lstsq(A, b, rcond=None)[0]

    err_lu = np.linalg.norm(x_lu - x_true) / np.linalg.norm(x_true)
    err_chol = np.linalg.norm(x_chol - x_true) / np.linalg.norm(x_true)
    err_qr = np.linalg.norm(x_qr - x_true) / np.linalg.norm(x_true)
    err_svd = np.linalg.norm(x_svd - x_true) / np.linalg.norm(x_true)

    print(f"{kappa:12.0e} {err_lu:12.2e} {err_chol:14.2e} {err_qr:12.2e} {err_svd:12.2e}")
```

### 7.2 모범 사례

1. **행렬을 파악하세요**: 구조(대칭성, 양정부호, 희소성)를 활용하여 올바른 분해를 선택합니다
2. 가능하면 **$A^TA$ 형성을 피하세요**: 최소제곱에는 정규방정식 대신 QR이나 SVD를 사용합니다
3. **한 번 분해하고 여러 번 풀기**: LU, Cholesky, QR은 여러 우변에 재사용할 수 있습니다
4. 풀기 전에 **조건수를 확인하세요**: $\kappa(A) > 10^{12}$이면 결과가 신뢰할 수 없을 수 있습니다

---

## 8. 응용: 일반화 고유값 문제

SPD $B$에 대한 일반화 고유값 문제 $A\mathbf{x} = \lambda B\mathbf{x}$는 Cholesky를 사용하여 표준 문제로 변환할 수 있습니다:

1. $B = LL^T$ 계산
2. $C = L^{-1} A L^{-T}$ 형성
3. 표준 고유값 문제 $C\mathbf{y} = \lambda\mathbf{y}$ 풀기
4. $\mathbf{x} = L^{-T}\mathbf{y}$ 복원

```python
# Generalized eigenvalue problem
A = np.array([[5, 2], [2, 3]], dtype=float)
B = np.array([[2, 1], [1, 2]], dtype=float)  # SPD

# Via Cholesky reduction
L = np.linalg.cholesky(B)
L_inv = np.linalg.inv(L)
C = L_inv @ A @ L_inv.T

eigvals_gen, Y = np.linalg.eigh(C)
X = linalg.solve_triangular(L.T, Y, lower=False)

print(f"Generalized eigenvalues: {eigvals_gen}")

# Verify: A x = lambda B x
for i in range(2):
    lhs = A @ X[:, i]
    rhs = eigvals_gen[i] * B @ X[:, i]
    print(f"lambda={eigvals_gen[i]:.4f}: Ax={lhs}, lambda*Bx={rhs}, match={np.allclose(lhs, rhs)}")

# Compare with scipy
from scipy.linalg import eigh
eigvals_sp, eigvecs_sp = eigh(A, B)
print(f"\nSciPy generalized eigenvalues: {eigvals_sp}")
```

---

## 9. 결정 흐름도

분해를 선택하려면 다음 논리를 따릅니다:

```
A가 정방인가?
  |
  |-- 예 --> A가 대칭인가?
  |             |
  |             |-- 예 --> A가 양정부호인가?
  |             |             |
  |             |             |-- 예 --> Cholesky (가장 빠름)
  |             |             |-- 아니오 --> LDL^T 또는 고유분해
  |             |
  |             |-- 아니오 --> 고유값이 필요한가?
  |                           |
  |                           |-- 예 --> Schur (가장 안정적)
  |                           |-- 아니오 --> LU (Ax = b 풀기용)
  |
  |-- 아니오 --> 최소제곱이 필요한가?
                |
                |-- 예 --> QR (안정적) 또는 SVD (가장 견고)
                |-- 아니오 --> SVD (랭크, 유사역행렬 등)
```

---

## 10. 요약

| 분해 | 형태 | 핵심 성질 | NumPy / SciPy |
|------|------|----------|---------------|
| Cholesky | $A = LL^T$ | SPD, 빠름, 유일 | `np.linalg.cholesky` |
| LDL^T | $A = LDL^T$ | 대칭, 제곱근 불필요 | `scipy.linalg.ldl` |
| Schur | $A = QTQ^T$ | 항상 존재, 대각에 고유값 | `scipy.linalg.schur` |
| 극분해 | $A = UP$ | 회전 + 늘림 | `scipy.linalg.polar` |
| LU | $PA = LU$ | 일반 풀기 | `scipy.linalg.lu` |
| QR | $A = QR$ | 최소제곱 | `np.linalg.qr` |
| SVD | $A = U\Sigma V^T$ | 범용 | `np.linalg.svd` |
| 고유분해 | $A = P\Lambda P^{-1}$ | 스펙트럼 분석 | `np.linalg.eig`, `eigh` |

---

## 연습 문제

### 연습 문제 1: Cholesky

1. 양의 원소를 가진 대각 행렬의 Cholesky 분해가 단순히 해당 원소의 제곱근 행렬임을 보이세요.
2. $A = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$의 Cholesky 분해를 수작업으로 계산하세요. NumPy로 검증하세요.
3. $\begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$의 Cholesky 분해를 시도하세요. 무슨 일이 일어나며 그 이유는 무엇입니까?

### 연습 문제 2: 정부호성 검사를 위한 LDL^T

100개의 무작위 $5 \times 5$ 대칭 행렬을 생성하세요. 각각에 대해 LDL^T 분해를 계산하고 $D$의 대각 원소의 부호에 기반하여 행렬을 양정부호, 양의 준정부호, 부정부호, 음정부호로 분류하세요. 고유값을 사용하여 분류를 검증하세요.

### 연습 문제 3: Schur 분해

$A = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 2 \end{bmatrix}$에 대해:
1. 실수 Schur 분해를 계산하세요.
2. 복소 Schur 분해를 계산하세요.
3. $T$의 대각에서 고유값을 식별하세요.

### 연습 문제 4: 극분해

$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$에 대해:
1. 극분해 $A = UP$를 계산하세요.
2. $U$가 직교이고 $P$가 SPD인지 검증하세요.
3. $P$만, $U$만, 그리고 $A = UP$에 의해 단위원이 어떻게 변환되는지 시각화하세요.

### 연습 문제 5: 분해 속도 경쟁

크기 $n = 500$의 SPD 행렬에 대해 다음 방법으로 $A\mathbf{x} = \mathbf{b}$를 푸는 데 걸리는 시간을 측정하세요:
1. `np.linalg.solve` (내부적으로 LU)
2. Cholesky 인수분해 + 두 번의 삼각 풀기
3. QR 분해
4. SVD + 유사역행렬

벽시계 시간을 비교하세요. 어느 것이 가장 빠릅니까? 같은 시스템을 (다른 $\mathbf{b}$로) 100번 풀고 각 방법의 총 시간을 측정하세요. 사전 인수분해가 어떻게 도움이 됩니까?

---

[<< 이전: 레슨 9 - 직교성과 투영](09_Orthogonality_and_Projections.md) | [개요](00_Overview.md) | [다음: 레슨 11 - 이차형식과 정부호성 >>](11_Quadratic_Forms_and_Definiteness.md)

**License**: CC BY-NC 4.0
