# 레슨 6: 고유값과 고유벡터 (Eigenvalues and Eigenvectors)

## 학습 목표

- 고유값과 고유벡터를 정의하고 그 기하학적 의미를 이해할 수 있다
- 특성 다항식의 근을 구하여 고유값을 계산할 수 있다
- 행렬을 대각화하고 대각화가 가능한 조건을 이해할 수 있다
- 대칭 행렬에 대한 스펙트럼 정리를 설명하고 적용할 수 있다
- 지배적 고유값을 구하기 위한 거듭제곱법을 구현할 수 있다
- 안정성 분석, 마르코프 체인, PCA에서의 고유값 응용을 인식할 수 있다

---

## 1. 정의와 기하학적 의미

### 1.1 고유값과 고유벡터란 무엇인가?

정방행렬 $A \in \mathbb{R}^{n \times n}$에 대해, 영이 아닌 벡터 $\mathbf{v}$가 다음을 만족하면 $A$의 **고유벡터**입니다:

$$A\mathbf{v} = \lambda\mathbf{v}$$

스칼라 $\lambda$가 대응하는 **고유값**입니다.

**기하학적 의미**: 고유벡터는 변환 $A$가 회전 없이 단순히 늘이거나(또는 뒤집는) 방향입니다. 고유값은 그 늘임 배율을 나타냅니다.

- $|\lambda| > 1$: 늘임 (stretching)
- $|\lambda| < 1$: 압축 (compression)
- $\lambda < 0$: 방향 반전
- $\lambda = 0$: 축소 (고유벡터가 0으로 매핑)

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[2, 1],
              [1, 2]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

# Verify: A @ v = lambda * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    lhs = A @ v
    rhs = lam * v
    print(f"\nlambda_{i} = {lam:.4f}")
    print(f"  A @ v = {lhs}")
    print(f"  lambda * v = {rhs}")
    print(f"  Match? {np.allclose(lhs, rhs)}")
```

### 1.2 고유벡터 시각화

```python
fig, ax = plt.subplots(figsize=(8, 8))

# Draw transformed and original vectors along many directions
theta_vals = np.linspace(0, 2 * np.pi, 60)
for t in theta_vals:
    v = np.array([np.cos(t), np.sin(t)])
    Av = A @ v
    ax.arrow(0, 0, v[0], v[1], head_width=0.03, color='blue', alpha=0.2)
    ax.arrow(0, 0, Av[0], Av[1], head_width=0.03, color='red', alpha=0.2)

# Highlight eigenvectors
for i in range(2):
    v = eigenvectors[:, i]
    Av = A @ v
    ax.arrow(0, 0, v[0], v[1], head_width=0.05, color='blue', linewidth=2,
             label=f'v{i+1}' if i == 0 else f'v{i+1}')
    ax.arrow(0, 0, Av[0], Av[1], head_width=0.05, color='red', linewidth=2,
             label=f'Av{i+1} (lambda={eigenvalues[i]:.1f})' if i == 0 else f'Av{i+1}')

ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Blue = original, Red = transformed\nEigenvectors stay on their line')
plt.show()
```

---

## 2. 특성 다항식 (Characteristic Polynomial)

### 2.1 유도

$A\mathbf{v} = \lambda\mathbf{v}$를 재배열하면:

$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

영이 아닌 해 $\mathbf{v}$가 존재하려면, 행렬 $(A - \lambda I)$가 특이여야 합니다:

$$\det(A - \lambda I) = 0$$

이것이 **특성 방정식**입니다. 좌변은 $\lambda$에 대한 $n$차 다항식으로, **특성 다항식**이라 합니다:

$$p(\lambda) = \det(A - \lambda I) = (-1)^n \lambda^n + \cdots$$

### 2.2 예시: 2x2 행렬

$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$에 대해:

$$p(\lambda) = \lambda^2 - (a+d)\lambda + (ad - bc) = \lambda^2 - \mathrm{tr}(A)\lambda + \det(A)$$

```python
import sympy

# Symbolic computation of characteristic polynomial
A_sym = sympy.Matrix([[2, 1],
                       [1, 2]])

lam = sympy.Symbol('lambda')
char_poly = (A_sym - lam * sympy.eye(2)).det()
print(f"Characteristic polynomial: {sympy.expand(char_poly)}")

# Find eigenvalues symbolically
eigenvals = sympy.solve(char_poly, lam)
print(f"Eigenvalues: {eigenvals}")
```

### 2.3 성질

고유값이 $\lambda_1, \ldots, \lambda_n$인 $n \times n$ 행렬에 대해 (중복도 포함):

| 성질 | 공식 |
|------|------|
| 고유값의 합 | $\lambda_1 + \cdots + \lambda_n = \mathrm{tr}(A)$ |
| 고유값의 곱 | $\lambda_1 \cdots \lambda_n = \det(A)$ |
| $A^k$의 고유값 | $\lambda_1^k, \ldots, \lambda_n^k$ |
| $A^{-1}$의 고유값 | $1/\lambda_1, \ldots, 1/\lambda_n$ |
| $A + cI$의 고유값 | $\lambda_1 + c, \ldots, \lambda_n + c$ |

```python
A = np.array([[3, 1, 0],
              [0, 2, 1],
              [0, 0, 1]], dtype=float)

eigenvalues = np.linalg.eigvals(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Sum = {np.sum(eigenvalues):.4f}, tr(A) = {np.trace(A):.4f}")
print(f"Product = {np.prod(eigenvalues):.4f}, det(A) = {np.linalg.det(A):.4f}")

# Eigenvalues of A^2
eig_A2 = np.linalg.eigvals(A @ A)
print(f"\nEigenvalues of A^2: {np.sort(eig_A2)}")
print(f"lambda^2:           {np.sort(eigenvalues**2)}")
```

---

## 3. 고유 분해 (Eigendecomposition, Diagonalization)

### 3.1 대각화가 가능한 조건

행렬 $A$가 **대각화 가능**하다는 것은 가역 행렬 $P$와 대각 행렬 $\Lambda$가 존재하여 다음을 만족하는 것입니다:

$$A = P \Lambda P^{-1}$$

여기서 $P$의 열은 고유벡터이고 $\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)$입니다.

**충분 조건**:
- $A$가 $n$개의 서로 다른 고유값을 가짐 (항상 대각화 가능)
- $A$가 대칭 (중복 고유값이 있어도 항상 대각화 가능)

**대각화 불가능**: 어떤 고유값의 기하학적 중복도가 대수적 중복도보다 작은 경우 (결함 행렬).

### 3.2 고유 분해 계산

```python
A = np.array([[4, 1],
              [2, 3]], dtype=float)

eigenvalues, P = np.linalg.eig(A)
Lambda = np.diag(eigenvalues)

print(f"P (eigenvectors):\n{P}")
print(f"Lambda (eigenvalues):\n{Lambda}")

# Verify: A = P Lambda P^{-1}
A_reconstructed = P @ Lambda @ np.linalg.inv(P)
print(f"\nP @ Lambda @ P^(-1):\n{A_reconstructed}")
print(f"Matches A? {np.allclose(A, A_reconstructed)}")
```

### 3.3 대각화 가능한 행렬의 거듭제곱

$A = P\Lambda P^{-1}$이면:

$$A^k = P \Lambda^k P^{-1}$$

이를 통해 행렬의 높은 거듭제곱을 효율적으로 계산할 수 있습니다.

```python
# Compute A^10 efficiently via diagonalization
k = 10

# Direct computation
A_power_direct = np.linalg.matrix_power(A, k)

# Via eigendecomposition
A_power_eig = P @ np.diag(eigenvalues**k) @ np.linalg.inv(P)

print(f"A^{k} (direct):\n{np.round(A_power_direct, 4)}")
print(f"A^{k} (eigen):\n{np.round(A_power_eig, 4)}")
print(f"Match? {np.allclose(A_power_direct, A_power_eig)}")
```

---

## 4. 스펙트럼 정리 (Spectral Theorem)

### 4.1 식

**실수 대칭** 행렬 $A = A^T$에 대해:

1. 모든 고유값이 **실수**
2. 서로 다른 고유값에 대응하는 고유벡터는 **직교**
3. $A$는 **직교** 행렬 $Q$로 대각화 가능 (즉, $Q^{-1} = Q^T$):

$$A = Q \Lambda Q^T$$

이것을 **스펙트럼 분해** (또는 대칭 행렬의 고유 분해)라 합니다.

### 4.2 외적의 합으로 표현된 스펙트럼 분해

$$A = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T$$

각 $\mathbf{q}_i \mathbf{q}_i^T$는 고유 공간으로의 랭크-1 투영 행렬입니다.

```python
# Symmetric matrix
A = np.array([[4, 2, 0],
              [2, 5, 3],
              [0, 3, 6]], dtype=float)

print(f"Symmetric? {np.allclose(A, A.T)}")

eigenvalues, Q = np.linalg.eigh(A)  # eigh for symmetric matrices
print(f"Eigenvalues: {eigenvalues}")
print(f"All real? {np.all(np.isreal(eigenvalues))}")

# Verify orthogonality
print(f"Q^T @ Q:\n{np.round(Q.T @ Q, 10)}")

# Verify spectral decomposition
A_reconstructed = Q @ np.diag(eigenvalues) @ Q.T
print(f"\nQ Lambda Q^T:\n{np.round(A_reconstructed, 10)}")
print(f"Matches A? {np.allclose(A, A_reconstructed)}")

# Outer product sum
A_outer = sum(eigenvalues[i] * np.outer(Q[:, i], Q[:, i]) for i in range(3))
print(f"Outer product sum matches? {np.allclose(A, A_outer)}")
```

### 4.3 `eig` vs `eigh`

| 함수 | 사용 대상 | 고유값 | 고유벡터 |
|------|----------|--------|---------|
| `np.linalg.eig` | 일반 행렬 | 복소수일 수 있음 | 직교가 아닐 수 있음 |
| `np.linalg.eigh` | 대칭/에르미트 행렬 | 항상 실수, 정렬됨 | 항상 정규 직교 |

대칭 행렬에는 항상 `eigh`를 사용하세요 -- 더 빠르고 수치적으로 더 안정적입니다.

---

## 5. 거듭제곱법 (Power Method)

### 5.1 알고리즘

**거듭제곱법**은 **지배적 고유값** (절댓값이 가장 큰)과 그 고유벡터를 찾는 반복 알고리즘입니다.

초기 벡터 $\mathbf{v}_0$이 주어지면, 다음을 반복합니다:

$$\mathbf{v}_{k+1} = \frac{A \mathbf{v}_k}{\|A \mathbf{v}_k\|}$$

벡터는 지배적 고유값에 대응하는 고유벡터로 수렴합니다.

### 5.2 구현

```python
def power_method(A, num_iterations=100, tol=1e-10):
    """Find the dominant eigenvalue and eigenvector using the power method."""
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    eigenvalue_estimates = []

    for i in range(num_iterations):
        Av = A @ v
        eigenvalue = np.dot(v, Av)  # Rayleigh quotient
        eigenvalue_estimates.append(eigenvalue)

        v_new = Av / np.linalg.norm(Av)

        if np.linalg.norm(v_new - v) < tol:
            print(f"Converged after {i+1} iterations")
            break

        v = v_new

    return eigenvalue, v, eigenvalue_estimates

# Test
A = np.array([[4, 1, 0],
              [1, 3, 1],
              [0, 1, 2]], dtype=float)

lam, v, history = power_method(A)
print(f"Dominant eigenvalue: {lam:.6f}")
print(f"Eigenvector: {v}")

# Compare with NumPy
eigenvalues_np = np.linalg.eigvals(A)
print(f"\nAll eigenvalues (NumPy): {np.sort(np.abs(eigenvalues_np))[::-1]}")

# Plot convergence
plt.figure(figsize=(8, 4))
plt.plot(history, 'b.-')
plt.axhline(y=max(np.abs(eigenvalues_np)), color='r', linestyle='--', label='True dominant eigenvalue')
plt.xlabel('Iteration')
plt.ylabel('Eigenvalue estimate')
plt.title('Power Method Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 5.3 역거듭제곱법 (Inverse Power Method)

**가장 작은** 고유값을 찾으려면 $A^{-1}$에 거듭제곱법을 적용합니다:

```python
def inverse_power_method(A, num_iterations=100, tol=1e-10):
    """Find the smallest eigenvalue using the inverse power method."""
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    lu_piv = linalg.lu_factor(A)  # Factor once for efficiency

    for i in range(num_iterations):
        from scipy import linalg as sp_linalg
        w = sp_linalg.lu_solve(lu_piv, v)
        eigenvalue = np.dot(v, w)
        v_new = w / np.linalg.norm(w)

        if np.linalg.norm(v_new - v) < tol:
            print(f"Converged after {i+1} iterations")
            break
        v = v_new

    return 1.0 / eigenvalue, v

from scipy import linalg

lam_min, v_min = inverse_power_method(A)
print(f"Smallest eigenvalue: {lam_min:.6f}")
print(f"All eigenvalues: {np.sort(eigenvalues_np)}")
```

---

## 6. 응용

### 6.1 안정성 분석

이산 시간 시스템 $\mathbf{x}_{k+1} = A\mathbf{x}_k$가 **안정**한 것은 모든 고유값이 $|\lambda_i| < 1$을 만족하는 것과 같습니다.

```python
# Stable system
A_stable = np.array([[0.5, 0.1],
                      [0.1, 0.5]])
eigs_stable = np.linalg.eigvals(A_stable)
print(f"Stable system eigenvalues: {eigs_stable}")
print(f"All |lambda| < 1? {np.all(np.abs(eigs_stable) < 1)}")

# Simulate
x = np.array([1.0, 0.5])
trajectory = [x.copy()]
for _ in range(50):
    x = A_stable @ x
    trajectory.append(x.copy())
trajectory = np.array(trajectory)

plt.figure(figsize=(8, 4))
plt.plot(trajectory[:, 0], label='x1')
plt.plot(trajectory[:, 1], label='x2')
plt.xlabel('Time step'); plt.ylabel('Value')
plt.title('Stable System (eigenvalues inside unit circle)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

### 6.2 마르코프 체인 (Markov Chains)

전이 행렬 $P$를 가진 **마르코프 체인**의 **정상 분포** $\boldsymbol{\pi}$는 $P^T \boldsymbol{\pi} = \boldsymbol{\pi}$를 만족합니다. 이것은 $P^T$의 고유값 $\lambda = 1$에 대응하는 고유벡터입니다.

```python
# Weather Markov chain: Sunny, Cloudy, Rainy
P = np.array([[0.7, 0.2, 0.1],   # Sunny -> ...
              [0.3, 0.4, 0.3],   # Cloudy -> ...
              [0.2, 0.3, 0.5]])  # Rainy -> ...

# Find stationary distribution
eigenvalues, eigenvectors = np.linalg.eig(P.T)
print(f"Eigenvalues of P^T: {eigenvalues}")

# Eigenvector with eigenvalue 1
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi = np.abs(eigenvectors[:, idx])
pi = pi / np.sum(pi)  # normalize

print(f"Stationary distribution: {pi}")
print(f"Sunny: {pi[0]:.3f}, Cloudy: {pi[1]:.3f}, Rainy: {pi[2]:.3f}")

# Verify: P^T pi = pi
print(f"P^T @ pi = {P.T @ pi}")
```

### 6.3 Google PageRank

PageRank는 근본적으로 고유값 문제입니다: 순위 벡터는 수정된 웹 링크 행렬의 지배적 고유벡터입니다.

```python
# Simplified PageRank example
# 4 web pages with link structure
# Page 0 links to 1, 2
# Page 1 links to 2
# Page 2 links to 0
# Page 3 links to 0, 1, 2

n_pages = 4
L = np.zeros((n_pages, n_pages))
L[1, 0] = 1; L[2, 0] = 1     # page 0 links to 1, 2
L[2, 1] = 1                    # page 1 links to 2
L[0, 2] = 1                    # page 2 links to 0
L[0, 3] = 1; L[1, 3] = 1; L[2, 3] = 1  # page 3 links to 0, 1, 2

# Column-normalize (each column sums to 1)
col_sums = L.sum(axis=0)
col_sums[col_sums == 0] = 1  # avoid division by zero
M = L / col_sums

# Damping factor
d = 0.85
G = d * M + (1 - d) / n_pages * np.ones((n_pages, n_pages))

# Find dominant eigenvector
eigenvalues, eigenvectors = np.linalg.eig(G)
idx = np.argmax(np.abs(eigenvalues))
rank = np.abs(eigenvectors[:, idx])
rank = rank / np.sum(rank)

print(f"PageRank: {rank}")
for i, r in enumerate(rank):
    print(f"  Page {i}: {r:.4f}")
```

---

## 7. 복소 고유값 (Complex Eigenvalues)

비대칭 실수 행렬은 **복소 고유값**을 가질 수 있으며, 항상 켤레 쌍으로 나타납니다: $\lambda = a \pm bi$.

복소 고유값은 변환에서 **회전** 동작에 대응합니다.

```python
# Rotation matrix has complex eigenvalues
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

eigs = np.linalg.eigvals(R)
print(f"Rotation eigenvalues: {eigs}")
print(f"Magnitude: {np.abs(eigs)}")  # both = 1

# Non-symmetric example with complex eigenvalues
A = np.array([[0, -1],
              [1,  0]], dtype=float)  # 90-degree rotation
eigs = np.linalg.eigvals(A)
print(f"\n90-degree rotation eigenvalues: {eigs}")
```

---

## 8. 요약

| 개념 | 설명 |
|------|------|
| 고유값 방정식 | $A\mathbf{v} = \lambda\mathbf{v}$ |
| 특성 다항식 | $\det(A - \lambda I) = 0$ |
| 대각화 | $A = P\Lambda P^{-1}$ |
| 스펙트럼 정리 | 대칭 $A = Q\Lambda Q^T$ (직교 $Q$) |
| 거듭제곱법 | 반복적 지배 고유값 탐색기 |
| 대각합 = 고유값의 합 | $\mathrm{tr}(A) = \sum \lambda_i$ |
| 행렬식 = 고유값의 곱 | $\det(A) = \prod \lambda_i$ |
| NumPy | `np.linalg.eig(A)` 또는 `np.linalg.eigh(A)` |

---

## 연습 문제

### 연습 문제 1: 수동 고유값 계산

$A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$의 고유값과 고유벡터를 특성 다항식을 이용하여 손으로 구하세요. NumPy로 검증하세요.

### 연습 문제 2: 대각화

$A = \begin{bmatrix} 5 & 4 \\ 1 & 2 \end{bmatrix}$를 대각화하세요:
1. 고유값과 고유벡터를 구하세요.
2. $P$와 $\Lambda$를 구성하세요.
3. $A = P\Lambda P^{-1}$을 검증하세요.
4. 이 분해를 사용하여 $A^{100}$을 계산하세요.

### 연습 문제 3: 스펙트럼 정리

대칭 행렬 $A = \begin{bmatrix} 2 & 1 & 0 \\ 1 & 3 & 1 \\ 0 & 1 & 2 \end{bmatrix}$에 대해:
1. 스펙트럼 분해 $A = Q\Lambda Q^T$를 구하세요.
2. $Q$가 직교 행렬임을 확인하세요.
3. $A$를 랭크-1 외적의 합으로 표현하세요.

### 연습 문제 4: 거듭제곱법

시프트를 적용한 거듭제곱법을 구현하세요: 목표값 $\sigma$에 가장 가까운 고유값을 찾으려면 $(A - \sigma I)$에 역거듭제곱법을 적용합니다. $4 \times 4$ 행렬에서 테스트하세요.

### 연습 문제 5: 마르코프 체인

개구리가 3개의 연잎 사이를 전이 확률 $P = \begin{bmatrix} 0.5 & 0.25 & 0.25 \\ 0.25 & 0.5 & 0.25 \\ 0.25 & 0.25 & 0.5 \end{bmatrix}$로 뛰어다닙니다. 정상 분포를 구하세요. 10,000번의 시뮬레이션을 수행하고 경험적 빈도를 이론적 분포와 비교하세요.

---

[<< 이전: 레슨 5 - 선형 변환](05_Linear_Transformations.md) | [개요](00_Overview.md) | [다음: 레슨 7 - 특이값 분해 >>](07_Singular_Value_Decomposition.md)

**License**: CC BY-NC 4.0
