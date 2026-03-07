# 레슨 9: 직교성과 투영

## 학습 목표

- 벡터, 직선, 부분공간에 대한 직교 투영을 계산할 수 있습니다
- 최소제곱 문제에 대한 정규방정식을 유도하고 적용할 수 있습니다
- 그람-슈미트 과정을 구현하고 QR 분해와의 관계를 이해할 수 있습니다
- QR 분해를 계산하고 최소제곱 문제를 푸는 데 사용할 수 있습니다
- 최소제곱법을 선형 회귀와 다항식 피팅에 적용할 수 있습니다

---

## 1. 벡터에 대한 직교 투영

### 1.1 투영 공식

$\mathbf{b}$를 $\mathbf{a}$에 대해 직교 투영한 것은 $\mathbf{a}$가 생성하는 직선 위의 점 중 $\mathbf{b}$에 가장 가까운 점입니다:

$$\mathrm{proj}_{\mathbf{a}}(\mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{a} \cdot \mathbf{a}} \mathbf{a} = \frac{\mathbf{a}^T \mathbf{b}}{\mathbf{a}^T \mathbf{a}} \mathbf{a}$$

$\mathbf{a}$에 투영하는 **투영 행렬**은 다음과 같습니다:

$$P = \frac{\mathbf{a}\mathbf{a}^T}{\mathbf{a}^T \mathbf{a}}$$

### 1.2 투영 행렬의 성질

투영 행렬 $P$는 다음을 만족합니다:
- **멱등성**: $P^2 = P$ (두 번 투영해도 같은 결과)
- **대칭성**: $P = P^T$ (직교 투영의 경우)
- **오차**(잔차) $\mathbf{e} = \mathbf{b} - P\mathbf{b}$는 투영에 직교합니다: $P\mathbf{b} \perp \mathbf{e}$

```python
import numpy as np
import matplotlib.pyplot as plt

a = np.array([2, 1], dtype=float)
b = np.array([1, 3], dtype=float)

# Projection of b onto a
proj = (np.dot(a, b) / np.dot(a, a)) * a
error = b - proj

print(f"proj_a(b) = {proj}")
print(f"error = {error}")
print(f"proj . error = {np.dot(proj, error):.10f}")  # ~0 (orthogonal)

# Projection matrix
P = np.outer(a, a) / np.dot(a, a)
print(f"\nProjection matrix P:\n{P}")
print(f"P^2 == P? {np.allclose(P @ P, P)}")
print(f"P == P^T? {np.allclose(P, P.T)}")
print(f"P @ b = {P @ b}")  # same as proj

# Visualize
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1,
          color='red', width=0.008, label='a')
ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1,
          color='blue', width=0.008, label='b')
ax.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1,
          color='green', width=0.008, label='proj_a(b)')
ax.plot([proj[0], b[0]], [proj[1], b[1]], 'k--', linewidth=1, label='error (perpendicular)')
ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend()
ax.set_title('Orthogonal Projection onto a Vector')
plt.show()
```

---

## 2. 부분공간에 대한 투영

### 2.1 일반 공식

$\mathbf{b}$를 행렬 $A$의 열공간에 투영하려면 ($A$의 열이 선형 독립):

$$\hat{\mathbf{b}} = A(A^T A)^{-1} A^T \mathbf{b}$$

**투영 행렬**은 다음과 같습니다:

$$P = A(A^T A)^{-1} A^T$$

### 2.2 유도

$\hat{\mathbf{b}} = A\hat{\mathbf{x}}$를 찾되, 오차 $\mathbf{e} = \mathbf{b} - A\hat{\mathbf{x}}$가 $A$의 열공간에 직교하도록 합니다:

$$A^T(\mathbf{b} - A\hat{\mathbf{x}}) = \mathbf{0}$$

이로부터 **정규방정식**(normal equations)을 얻습니다:

$$A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$$

풀면: $\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$, 그리고 $\hat{\mathbf{b}} = A\hat{\mathbf{x}}$.

```python
# Project b onto the plane spanned by a1 and a2
a1 = np.array([1, 0, 0], dtype=float)
a2 = np.array([0, 1, 0], dtype=float)
b = np.array([1, 2, 3], dtype=float)

A = np.column_stack([a1, a2])

# Projection
P = A @ np.linalg.inv(A.T @ A) @ A.T
b_hat = P @ b
error = b - b_hat

print(f"Projection: {b_hat}")
print(f"Error: {error}")
print(f"Error orthogonal to a1? {np.isclose(np.dot(error, a1), 0)}")
print(f"Error orthogonal to a2? {np.isclose(np.dot(error, a2), 0)}")
print(f"P is idempotent? {np.allclose(P @ P, P)}")
print(f"P is symmetric? {np.allclose(P, P.T)}")
```

### 2.3 보완 투영

$P$가 부분공간 $S$에 대한 투영이면, $I - P$는 $S^{\perp}$(직교 보공간)에 대한 투영입니다:

```python
# Complementary projection
P_perp = np.eye(3) - P

b_perp = P_perp @ b
print(f"Projection onto complement: {b_perp}")
print(f"b = proj + proj_perp? {np.allclose(b, b_hat + b_perp)}")
print(f"proj . proj_perp = {np.dot(b_hat, b_perp):.10f}")  # ~0
```

---

## 3. 최소제곱법

### 3.1 문제

**과잉결정**(overdetermined) 시스템 $A\mathbf{x} = \mathbf{b}$ (방정식 수가 미지수보다 많은 경우)에는 일반적으로 정확한 해가 없습니다. **최소제곱** 해는 다음을 최소화합니다:

$$\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2$$

### 3.2 정규방정식

기울기를 0으로 놓으면 **정규방정식**을 얻습니다:

$$A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$$

$A$가 완전 열랭크(full column rank)이면, 유일한 해는:

$$\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$$

### 3.3 선형 회귀 예제

```python
# Linear regression: fit y = ax + b
np.random.seed(42)
x_data = np.linspace(0, 10, 50)
y_data = 2.5 * x_data + 1.0 + np.random.randn(50) * 2

# Design matrix (column of ones + column of x values)
A = np.column_stack([np.ones_like(x_data), x_data])
b = y_data

# Normal equation
x_hat = np.linalg.solve(A.T @ A, A.T @ b)
print(f"Intercept: {x_hat[0]:.4f}, Slope: {x_hat[1]:.4f}")
print(f"True: Intercept = 1.0, Slope = 2.5")

# Prediction and residual
y_pred = A @ x_hat
residual = b - y_pred
print(f"Residual norm: {np.linalg.norm(residual):.4f}")

# Compare with np.linalg.lstsq
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"lstsq result: {x_lstsq}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(x_data, y_data, alpha=0.5, s=20, label='Data')
ax1.plot(x_data, y_pred, 'r-', linewidth=2,
         label=f'Fit: y = {x_hat[1]:.2f}x + {x_hat[0]:.2f}')
ax1.set_xlabel('x'); ax1.set_ylabel('y')
ax1.set_title('Least Squares Linear Regression')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.stem(range(len(residual)), residual, linefmt='b-', markerfmt='bo', basefmt='r-')
ax2.set_xlabel('Sample index'); ax2.set_ylabel('Residual')
ax2.set_title('Residuals')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.4 다항식 피팅

```python
# Polynomial regression: fit y = a0 + a1*x + a2*x^2 + a3*x^3
np.random.seed(42)
x = np.linspace(-2, 2, 60)
y_true = 0.5 * x**3 - 2 * x + 1
y = y_true + np.random.randn(60) * 0.5

degrees = [1, 3, 10]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, deg in zip(axes, degrees):
    # Vandermonde matrix
    A = np.vander(x, N=deg+1, increasing=True)
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]

    x_fine = np.linspace(-2.2, 2.2, 200)
    A_fine = np.vander(x_fine, N=deg+1, increasing=True)
    y_fit = A_fine @ coeffs

    ax.scatter(x, y, alpha=0.4, s=15, label='Data')
    ax.plot(x_fine, y_fit, 'r-', linewidth=2, label=f'Degree {deg} fit')
    ax.set_ylim(-8, 8)
    ax.set_title(f'Polynomial Degree {deg}')
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 4. 그람-슈미트 과정

### 4.1 알고리즘

선형 독립 벡터 $\mathbf{a}_1, \ldots, \mathbf{a}_n$이 주어졌을 때, 그람-슈미트 과정은 같은 부분공간을 생성하는 **정규직교** 집합 $\mathbf{q}_1, \ldots, \mathbf{q}_n$을 생성합니다.

**단계**:

$$\mathbf{u}_1 = \mathbf{a}_1, \quad \mathbf{q}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|}$$

$$\mathbf{u}_k = \mathbf{a}_k - \sum_{j=1}^{k-1} (\mathbf{q}_j^T \mathbf{a}_k) \mathbf{q}_j, \quad \mathbf{q}_k = \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|}$$

### 4.2 구현

```python
def gram_schmidt(A):
    """Classical Gram-Schmidt orthonormalization.

    A: matrix whose columns are the input vectors.
    Returns Q (orthonormal columns) and R (upper triangular).
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        # Subtract projections onto previous basis vectors
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-10:
            raise ValueError("Columns are linearly dependent")
        Q[:, j] = v / R[j, j]

    return Q, R

# Test
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

Q, R = gram_schmidt(A)

print(f"Q (orthonormal):\n{np.round(Q, 4)}")
print(f"R (upper triangular):\n{np.round(R, 4)}")
print(f"Q^T Q (should be I):\n{np.round(Q.T @ Q, 10)}")
print(f"QR == A? {np.allclose(Q @ R, A)}")
```

### 4.3 수정 그람-슈미트 (Modified Gram-Schmidt)

고전적 알고리즘은 수치적 불안정성을 겪을 수 있습니다. **수정** 그람-슈미트 알고리즘이 더 안정적입니다:

```python
def modified_gram_schmidt(A):
    """Modified Gram-Schmidt -- more numerically stable."""
    m, n = A.shape
    Q = A.astype(float).copy()
    R = np.zeros((n, n))

    for j in range(n):
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / R[j, j]

        for k in range(j + 1, n):
            R[j, k] = np.dot(Q[:, j], Q[:, k])
            Q[:, k] -= R[j, k] * Q[:, j]

    return Q, R

# Compare classical vs modified on a mildly ill-conditioned matrix
np.random.seed(42)
A_ill = np.random.randn(100, 50) + 1e-8

Q_classical, R_classical = gram_schmidt(A_ill)
Q_modified, R_modified = modified_gram_schmidt(A_ill)

# Orthogonality quality
orth_error_classical = np.linalg.norm(Q_classical.T @ Q_classical - np.eye(50))
orth_error_modified = np.linalg.norm(Q_modified.T @ Q_modified - np.eye(50))

print(f"Orthogonality error (classical): {orth_error_classical:.2e}")
print(f"Orthogonality error (modified):  {orth_error_modified:.2e}")
```

---

## 5. QR 분해

### 5.1 정의

모든 $m \times n$ 행렬 $A$ ($m \ge n$이고 완전 열랭크)는 다음과 같이 분해됩니다:

$$A = QR$$

여기서:
- $Q \in \mathbb{R}^{m \times n}$는 정규직교 열을 가집니다 ($Q^TQ = I_n$)
- $R \in \mathbb{R}^{n \times n}$는 양의 대각 원소를 가진 상삼각 행렬입니다

### 5.2 NumPy로 QR 계산

```python
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

Q, R = np.linalg.qr(A)

print(f"Q:\n{np.round(Q, 4)}")
print(f"R:\n{np.round(R, 4)}")
print(f"Q^T Q:\n{np.round(Q.T @ Q, 10)}")
print(f"QR == A? {np.allclose(Q @ R, A)}")
```

### 5.3 QR을 사용한 최소제곱 풀기

QR 분해는 최소제곱을 수치적으로 안정하게 푸는 방법을 제공합니다:

$$A\mathbf{x} = \mathbf{b} \implies QR\mathbf{x} = \mathbf{b} \implies R\mathbf{x} = Q^T\mathbf{b}$$

이 방법은 $A^TA$를 형성하지 않으므로 조건수의 제곱 증가를 피합니다.

```python
# Least squares via QR
np.random.seed(42)
m, n = 100, 3
A = np.random.randn(m, n)
b = np.random.randn(m)

# Method 1: Normal equations (less stable)
x_normal = np.linalg.solve(A.T @ A, A.T @ b)

# Method 2: QR decomposition (more stable)
Q, R = np.linalg.qr(A)
x_qr = np.linalg.solve(R, Q.T @ b)

# Method 3: NumPy lstsq (uses SVD internally)
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]

print(f"Normal equations: {x_normal}")
print(f"QR decomposition: {x_qr}")
print(f"NumPy lstsq:      {x_lstsq}")
print(f"All match? {np.allclose(x_normal, x_qr) and np.allclose(x_qr, x_lstsq)}")

# Condition number comparison
print(f"\ncond(A):     {np.linalg.cond(A):.4f}")
print(f"cond(A^T A): {np.linalg.cond(A.T @ A):.4f}")  # squared!
```

---

## 6. 직교 보공간

### 6.1 정의

$\mathbb{R}^n$에서 부분공간 $S$의 **직교 보공간**은 다음과 같습니다:

$$S^{\perp} = \{\mathbf{v} \in \mathbb{R}^n : \mathbf{v} \cdot \mathbf{s} = 0 \text{ for all } \mathbf{s} \in S\}$$

주요 사실:
- $\dim(S) + \dim(S^{\perp}) = n$
- $(S^{\perp})^{\perp} = S$
- 모든 $\mathbf{v} \in \mathbb{R}^n$은 $\mathbf{v} = \mathbf{v}_S + \mathbf{v}_{S^{\perp}}$로 유일하게 분해됩니다

### 6.2 기본 부분공간 관계

행렬 $A$에 대해:
- $\mathrm{col}(A)^{\perp} = \mathrm{null}(A^T)$ ($\mathbb{R}^m$에서)
- $\mathrm{row}(A)^{\perp} = \mathrm{null}(A)$ ($\mathbb{R}^n$에서)

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=float)

# Column space (via QR)
Q_col, _ = np.linalg.qr(A)

# Null space of A^T (left null space)
from scipy.linalg import null_space
left_null = null_space(A.T)

print(f"Column space basis (Q):\n{np.round(Q_col, 4)}")
print(f"Left null space basis:\n{np.round(left_null, 4)}")

# Verify orthogonality
print(f"\nQ^T @ left_null:\n{np.round(Q_col.T @ left_null, 10)}")  # should be ~0
```

---

## 7. 응용: 가중 최소제곱법

서로 다른 관측치가 서로 다른 신뢰도를 가질 때, **가중 최소제곱법**(weighted least squares)을 사용합니다:

$$\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} (A\mathbf{x} - \mathbf{b})^T W (A\mathbf{x} - \mathbf{b})$$

여기서 $W$는 대각 가중 행렬입니다. 해는 다음과 같습니다:

$$\hat{\mathbf{x}} = (A^T W A)^{-1} A^T W \mathbf{b}$$

```python
# Weighted least squares
np.random.seed(42)
x_data = np.linspace(0, 10, 50)
y_true = 2 * x_data + 1

# Add heteroscedastic noise (more noise for larger x)
noise_std = 0.5 + 0.5 * x_data
y_data = y_true + np.random.randn(50) * noise_std

# Design matrix
A = np.column_stack([np.ones_like(x_data), x_data])

# Weights: inverse of noise variance
weights = 1.0 / noise_std**2
W = np.diag(weights)

# Ordinary least squares
x_ols = np.linalg.solve(A.T @ A, A.T @ y_data)

# Weighted least squares
x_wls = np.linalg.solve(A.T @ W @ A, A.T @ W @ y_data)

print(f"OLS: intercept = {x_ols[0]:.4f}, slope = {x_ols[1]:.4f}")
print(f"WLS: intercept = {x_wls[0]:.4f}, slope = {x_wls[1]:.4f}")
print(f"True: intercept = 1.0, slope = 2.0")

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(x_data, y_data, yerr=noise_std, fmt='o', alpha=0.3, markersize=3)
plt.plot(x_data, A @ x_ols, 'b-', linewidth=2, label=f'OLS: {x_ols[1]:.2f}x + {x_ols[0]:.2f}')
plt.plot(x_data, A @ x_wls, 'r-', linewidth=2, label=f'WLS: {x_wls[1]:.2f}x + {x_wls[0]:.2f}')
plt.plot(x_data, y_true, 'k--', linewidth=1, label='True: 2.0x + 1.0')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.grid(True, alpha=0.3)
plt.title('Ordinary vs Weighted Least Squares')
plt.show()
```

---

## 8. 하우스홀더 반사 (고급)

NumPy는 내부적으로 그람-슈미트 대신 **하우스홀더 반사**(Householder reflections)를 사용하여 QR을 계산합니다. 하우스홀더 반사는 다음과 같습니다:

$$H = I - 2\mathbf{v}\mathbf{v}^T \quad (\|\mathbf{v}\| = 1)$$

이것은 벡터를 $\mathbf{v}$에 수직인 초평면을 기준으로 반사합니다.

```python
def householder_qr(A):
    """QR decomposition using Householder reflections."""
    m, n = A.shape
    R = A.astype(float).copy()
    Q = np.eye(m)

    for j in range(min(m - 1, n)):
        # Select the column below the diagonal
        x = R[j:, j]

        # Householder vector
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * (-1 if x[0] < 0 else 1)
        v = x + e1
        v = v / np.linalg.norm(v)

        # Apply reflection to R
        R[j:, j:] -= 2 * np.outer(v, v @ R[j:, j:])

        # Accumulate Q
        Q[j:, :] -= 2 * np.outer(v, v @ Q[j:, :])

    return Q.T, R

# Test
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

Q_hh, R_hh = householder_qr(A)
print(f"Householder Q:\n{np.round(Q_hh, 4)}")
print(f"Householder R:\n{np.round(R_hh, 4)}")
print(f"QR == A? {np.allclose(Q_hh @ R_hh, A)}")
```

---

## 9. 요약

| 개념 | 공식 / 설명 |
|------|-------------|
| 벡터에 대한 투영 | $\frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}}\mathbf{a}$ |
| 투영 행렬 (부분공간) | $P = A(A^TA)^{-1}A^T$ |
| 정규방정식 | $A^TA\hat{\mathbf{x}} = A^T\mathbf{b}$ |
| 그람-슈미트 | 독립 벡터로부터 정규직교 기저를 생성합니다 |
| QR 분해 | $A = QR$, $Q$ 정규직교, $R$ 상삼각 |
| QR을 통한 최소제곱 | $R\hat{\mathbf{x}} = Q^T\mathbf{b}$ (정규방정식보다 안정적) |
| NumPy | `np.linalg.qr(A)`, `np.linalg.lstsq(A, b)` |

---

## 연습 문제

### 연습 문제 1: 투영 행렬

$\mathbf{a}_1 = [1, 0, 1]^T$과 $\mathbf{a}_2 = [0, 1, 1]^T$이 생성하는 평면에 $\mathbb{R}^3$를 투영하는 투영 행렬 $P$를 계산하세요. $P$가 멱등성과 대칭성을 만족하는지 검증하세요. $\mathbf{b} = [1, 2, 3]^T$를 투영하고 오차가 $\mathbf{a}_1$과 $\mathbf{a}_2$ 모두에 직교하는지 확인하세요.

### 연습 문제 2: 최소제곱 피팅

다음 데이터에 이차식 $y = ax^2 + bx + c$를 피팅하세요:

| x | -2 | -1 | 0 | 1 | 2 | 3 |
|---|----|----|---|---|---|---|
| y | 7  | 2  | 1 | 2 | 7 | 14|

정규방정식을 세우고 풀어보세요. 잔차를 계산하세요.

### 연습 문제 3: 그람-슈미트

$\{\mathbf{a}_1 = [1, 1, 1]^T, \mathbf{a}_2 = [1, 1, 0]^T, \mathbf{a}_3 = [1, 0, 0]^T\}$에 그람-슈미트 과정을 적용하세요 (수작업과 코드 모두). 결과가 정규직교이고 $A = QR$인지 검증하세요.

### 연습 문제 4: QR vs 정규방정식

큰 조건수를 가진 $200 \times 10$ 행렬 $A$를 생성하세요 (`np.linalg.svd`를 사용하여 특정 특이값 설정). 정규방정식과 QR을 모두 사용하여 $A\mathbf{x} = \mathbf{b}$를 푸세요. 알려진 참 해와 비교하여 두 해의 정확도를 비교하세요. 정규방정식 접근법이 실패하는 때는 언제입니까?

### 연습 문제 5: 가중 최소제곱

센서가 서로 다른 시간에 알려진 측정 불확실도 $\sigma_i$로 측정을 수행합니다. 가중 최소제곱법을 사용하여 선형 모델을 피팅하고, 잡음이 이분산(heteroscedastic)일 때 통상 최소제곱법보다 더 나은 매개변수 추정을 제공함을 보이세요.

---

[<< 이전: 레슨 8 - 주성분 분석](08_Principal_Component_Analysis.md) | [개요](00_Overview.md) | [다음: 레슨 10 - 행렬 분해 >>](10_Matrix_Decompositions.md)

**License**: CC BY-NC 4.0
