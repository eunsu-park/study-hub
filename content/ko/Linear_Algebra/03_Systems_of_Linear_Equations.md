# 레슨 3: 연립일차방정식 (Systems of Linear Equations)

## 학습 목표

- 연립일차방정식을 행렬 형태 $A\mathbf{x} = \mathbf{b}$로 표현할 수 있다
- 가우스 소거법을 손으로, 그리고 코드로 수행하여 행 사다리꼴(REF)과 기약 행 사다리꼴(RREF)로 변환할 수 있다
- 랭크 조건을 사용하여 시스템의 해가 없는지, 하나인지, 무한히 많은지 판별할 수 있다
- LU 분해를 계산하고 이를 사용하여 시스템을 효율적으로 풀 수 있다
- 이러한 기법을 회로 분석과 보간법 등의 실용적인 문제에 적용할 수 있다

---

## 1. 연립방정식의 행렬 형태

$n$개의 미지수에 대한 $m$개의 일차방정식 시스템:

$$\begin{aligned}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n &= b_2 \\
&\;\;\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m
\end{aligned}$$

은 다음과 같이 간결하게 표현할 수 있습니다:

$$A\mathbf{x} = \mathbf{b}$$

여기서 $A \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{b} \in \mathbb{R}^m$입니다.

```python
import numpy as np
from scipy import linalg

# Example: 3 equations, 3 unknowns
# 2x + y - z = 8
# -3x - y + 2z = -11
# -2x + y + 2z = -3
A = np.array([[ 2,  1, -1],
              [-3, -1,  2],
              [-2,  1,  2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

x = np.linalg.solve(A, b)
print(f"Solution: x = {x}")
print(f"Verification: Ax = {A @ x}")
```

---

## 2. 가우스 소거법 (Gaussian Elimination)

### 2.1 알고리즘

가우스 소거법은 세 가지 기본 행 연산을 사용하여 확대 행렬 $[A \mid \mathbf{b}]$를 **행 사다리꼴(REF)**로 변환합니다:

1. 두 행을 **교환**
2. 하나의 행에 0이 아닌 스칼라를 **곱함**
3. 한 행의 배수를 다른 행에 **더함**

### 2.2 행 사다리꼴 (Row Echelon Form, REF)

행렬이 REF에 있으려면:
- 모든 영행이 맨 아래에 위치
- 각 비영행의 선행 원소(피벗)가 윗행의 피벗보다 오른쪽에 위치
- 각 피벗 아래의 모든 원소가 0

### 2.3 구현

```python
def gaussian_elimination(A, b, verbose=False):
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix
    Ab = np.column_stack([A.astype(float), b.astype(float)])

    if verbose:
        print("Augmented matrix:")
        print(Ab)
        print()

    # Forward elimination
    for col in range(n):
        # Partial pivoting: find row with largest absolute value in column
        max_row = col + np.argmax(np.abs(Ab[col:, col]))
        if max_row != col:
            Ab[[col, max_row]] = Ab[[max_row, col]]
            if verbose:
                print(f"Swap rows {col} and {max_row}")

        if abs(Ab[col, col]) < 1e-12:
            raise ValueError(f"Zero pivot encountered at column {col}")

        # Eliminate entries below the pivot
        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]
            if verbose:
                print(f"R{row} <- R{row} - ({factor:.4f}) * R{col}")

        if verbose:
            print(Ab)
            print()

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]

    return x

# Test
A = np.array([[ 2,  1, -1],
              [-3, -1,  2],
              [-2,  1,  2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

x = gaussian_elimination(A, b, verbose=True)
print(f"Solution: {x}")
```

---

## 3. 기약 행 사다리꼴 (Reduced Row Echelon Form, RREF)

### 3.1 정의

RREF는 REF에 두 가지 추가 조건이 더해진 형태입니다:
- 각 피벗이 1
- 각 피벗이 해당 열에서 유일한 0이 아닌 원소

### 3.2 구현

```python
def rref(A, tol=1e-12):
    """Compute the reduced row echelon form of matrix A."""
    M = A.astype(float).copy()
    m, n = M.shape
    pivot_row = 0

    pivot_cols = []

    for col in range(n):
        if pivot_row >= m:
            break

        # Find pivot
        max_row = pivot_row + np.argmax(np.abs(M[pivot_row:, col]))
        if abs(M[max_row, col]) < tol:
            continue  # skip this column

        # Swap rows
        M[[pivot_row, max_row]] = M[[max_row, pivot_row]]

        # Scale pivot to 1
        M[pivot_row] = M[pivot_row] / M[pivot_row, col]

        # Eliminate all other entries in this column
        for row in range(m):
            if row != pivot_row and abs(M[row, col]) > tol:
                M[row] -= M[row, col] * M[pivot_row]

        pivot_cols.append(col)
        pivot_row += 1

    return M, pivot_cols

# Example
A_aug = np.array([[ 1,  2, -1,  3],
                   [ 2,  4,  1,  6],
                   [ 3,  6,  0,  9]], dtype=float)

R, pivots = rref(A_aug)
print(f"RREF:\n{R}")
print(f"Pivot columns: {pivots}")
```

### 3.3 SymPy를 사용한 정확한 RREF

```python
import sympy

# SymPy gives exact rational arithmetic
M = sympy.Matrix([[1, 2, -1, 3],
                   [2, 4,  1, 6],
                   [3, 6,  0, 9]])

R, pivots = M.rref()
print(f"RREF:\n{R}")
print(f"Pivot columns: {pivots}")
```

---

## 4. 해의 존재성과 유일성

### 4.1 랭크 조건

$A \in \mathbb{R}^{m \times n}$인 시스템 $A\mathbf{x} = \mathbf{b}$에 대해:

| 조건 | 결과 |
|------|------|
| $\mathrm{rank}(A) < \mathrm{rank}([A \mid \mathbf{b}])$ | **해 없음** (비일관적) |
| $\mathrm{rank}(A) = \mathrm{rank}([A \mid \mathbf{b}]) = n$ | **유일해** |
| $\mathrm{rank}(A) = \mathrm{rank}([A \mid \mathbf{b}]) < n$ | **무한히 많은 해** |

### 4.2 기하학적 해석

- **유일해**: $n$개의 초평면이 정확히 한 점에서 교차
- **해 없음**: 적어도 두 초평면이 평행 (비일관적)
- **무한 해**: 초평면이 직선, 평면 또는 더 높은 차원의 평탄한 집합을 따라 교차

```python
# Case 1: Unique solution
A1 = np.array([[1, 1], [1, -1]], dtype=float)
b1 = np.array([4, 2], dtype=float)
Ab1 = np.column_stack([A1, b1])
print("Case 1 (unique):")
print(f"  rank(A) = {np.linalg.matrix_rank(A1)}, rank([A|b]) = {np.linalg.matrix_rank(Ab1)}, n = {A1.shape[1]}")
print(f"  Solution: {np.linalg.solve(A1, b1)}")

# Case 2: No solution
A2 = np.array([[1, 1], [1, 1]], dtype=float)
b2 = np.array([2, 3], dtype=float)
Ab2 = np.column_stack([A2, b2])
print("\nCase 2 (no solution):")
print(f"  rank(A) = {np.linalg.matrix_rank(A2)}, rank([A|b]) = {np.linalg.matrix_rank(Ab2)}")

# Case 3: Infinitely many solutions
A3 = np.array([[1, 2], [2, 4]], dtype=float)
b3 = np.array([3, 6], dtype=float)
Ab3 = np.column_stack([A3, b3])
print("\nCase 3 (infinite solutions):")
print(f"  rank(A) = {np.linalg.matrix_rank(A3)}, rank([A|b]) = {np.linalg.matrix_rank(Ab3)}, n = {A3.shape[1]}")
```

### 4.3 2D 시스템 시각화

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

x_vals = np.linspace(-1, 5, 100)

# Unique solution: x + y = 4, x - y = 2
ax = axes[0]
ax.plot(x_vals, 4 - x_vals, label='x + y = 4')
ax.plot(x_vals, x_vals - 2, label='x - y = 2')
ax.plot(3, 1, 'ro', markersize=8)
ax.set_title('Unique solution (3, 1)')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

# No solution: x + y = 2, x + y = 3 (parallel lines)
ax = axes[1]
ax.plot(x_vals, 2 - x_vals, label='x + y = 2')
ax.plot(x_vals, 3 - x_vals, label='x + y = 3')
ax.set_title('No solution (parallel)')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

# Infinite solutions: x + 2y = 3, 2x + 4y = 6 (same line)
ax = axes[2]
ax.plot(x_vals, (3 - x_vals) / 2, label='x + 2y = 3', linewidth=3, alpha=0.5)
ax.plot(x_vals, (6 - 2*x_vals) / 4, label='2x + 4y = 6', linewidth=1, linestyle='--')
ax.set_title('Infinite solutions (same line)')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

for ax in axes:
    ax.set_xlim(-1, 5); ax.set_ylim(-2, 4)
    ax.set_xlabel('x'); ax.set_ylabel('y')

plt.tight_layout()
plt.show()
```

---

## 5. LU 분해

### 5.1 개념

LU 분해는 행렬 $A$를 다음과 같이 인수분해합니다:

$$A = LU$$

여기서 $L$은 **하삼각 행렬** (대각선에 1)이고 $U$는 **상삼각 행렬**입니다.

부분 피벗팅을 사용하면 다음과 같이 됩니다:

$$PA = LU$$

여기서 $P$는 치환 행렬입니다.

### 5.2 LU 분해를 사용하는 이유

$A = LU$가 구해지면, $A\mathbf{x} = \mathbf{b}$의 풀이는 두 번의 삼각 시스템 풀이로 귀결됩니다:

1. **전방 대입**: $L\mathbf{y} = \mathbf{b}$에서 $\mathbf{y}$를 구함
2. **후방 대입**: $U\mathbf{x} = \mathbf{y}$에서 $\mathbf{x}$를 구함

이 방법은 같은 시스템을 **여러 우변 벡터**에 대해 풀 때 특히 효율적입니다. 분해를 한 번만 계산하면 되기 때문입니다.

### 5.3 구현

```python
from scipy.linalg import lu, lu_factor, lu_solve

A = np.array([[ 2,  1, -1],
              [-3, -1,  2],
              [-2,  1,  2]], dtype=float)

# Full LU decomposition
P, L, U = lu(A)
print(f"P =\n{P}")
print(f"L =\n{L}")
print(f"U =\n{U}")

# Verify: PA = LU  =>  A = P^T L U
print(f"P^T L U =\n{P.T @ L @ U}")
print(f"Matches A? {np.allclose(A, P.T @ L @ U)}")

# Solve using LU
b = np.array([8, -11, -3], dtype=float)
lu_piv = lu_factor(A)
x = lu_solve(lu_piv, b)
print(f"Solution: {x}")

# Solve with multiple right-hand sides
B = np.array([[8, 1],
              [-11, 2],
              [-3, 3]], dtype=float)
X = lu_solve(lu_piv, B)
print(f"Solutions for two RHS:\n{X}")
```

### 5.4 수동 LU 분해 (피벗팅 없이)

```python
def lu_no_pivot(A):
    """LU decomposition without pivoting (for educational purposes)."""
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()

    for col in range(n - 1):
        for row in range(col + 1, n):
            factor = U[row, col] / U[col, col]
            L[row, col] = factor
            U[row, col:] -= factor * U[col, col:]

    return L, U

A = np.array([[2, 1, -1],
              [4, 5, -3],
              [6, 10, 1]], dtype=float)

L, U = lu_no_pivot(A)
print(f"L =\n{L}")
print(f"U =\n{U}")
print(f"L @ U =\n{L @ U}")
print(f"Matches A? {np.allclose(A, L @ U)}")
```

### 5.5 계산 비용

| 연산 | 비용 (flops) |
|------|-------------|
| LU 분해 | $\frac{2}{3}n^3$ |
| 전방 대입 | $n^2$ |
| 후방 대입 | $n^2$ |
| 1회 풀이 총 비용 | $\frac{2}{3}n^3 + 2n^2$ |
| 추가 우변 1개당 | $2n^2$ |

$A^{-1}$을 명시적으로 계산하는 것과 비교: 약 $2n^3$ flops. LU가 훨씬 효율적이며, 수치적으로도 더 안정적입니다.

---

## 6. 동차 시스템 (Homogeneous Systems)

$\mathbf{b} = \mathbf{0}$인 시스템을 **동차 시스템**이라 합니다:

$$A\mathbf{x} = \mathbf{0}$$

동차 시스템은 항상 적어도 **자명해** $\mathbf{x} = \mathbf{0}$을 가집니다.

비자명해가 존재하는 것은 $\mathrm{rank}(A) < n$ (미지수의 개수)인 것과 동치입니다.

모든 해의 집합은 $A$의 **영공간** (또는 **핵**)을 형성합니다:

$$\mathrm{null}(A) = \{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0}\}$$

```python
# Matrix with rank < n => non-trivial null space
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

print(f"rank(A) = {np.linalg.matrix_rank(A)}")  # 2 < 3

# Find null space using SVD
from scipy.linalg import null_space
ns = null_space(A)
print(f"Null space basis:\n{ns}")
print(f"A @ null_vector =\n{A @ ns}")  # should be ~0
```

---

## 7. 과대결정 시스템과 과소결정 시스템

### 7.1 과대결정 시스템 ($m > n$)

방정식이 미지수보다 많습니다. 일반적으로 정확한 해가 존재하지 않습니다. **최소자승해**는 $\|A\mathbf{x} - \mathbf{b}\|^2$를 최소화합니다:

$$\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$$

```python
# Overdetermined: 4 equations, 2 unknowns
A = np.array([[1, 1],
              [1, 2],
              [1, 3],
              [1, 4]], dtype=float)
b = np.array([2.1, 2.9, 4.2, 4.8], dtype=float)

# Least squares solution
x_lstsq, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
print(f"Least squares solution: {x_lstsq}")
print(f"Residual norm: {np.linalg.norm(A @ x_lstsq - b):.6f}")

# Via normal equation
x_normal = np.linalg.solve(A.T @ A, A.T @ b)
print(f"Normal equation solution: {x_normal}")
```

### 7.2 과소결정 시스템 ($m < n$)

방정식이 미지수보다 적습니다. 해가 존재하면 무한히 많습니다. **최소 노름 해**는 $\|\mathbf{x}\|$가 가장 작은 해입니다:

```python
# Underdetermined: 2 equations, 3 unknowns
A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=float)
b = np.array([6, 15], dtype=float)

# Minimum norm solution via pseudoinverse
x_min_norm = np.linalg.pinv(A) @ b
print(f"Minimum norm solution: {x_min_norm}")
print(f"Verification: Ax = {A @ x_min_norm}")
print(f"||x|| = {np.linalg.norm(x_min_norm):.6f}")
```

---

## 8. 응용: 다항식 보간법

$n+1$개의 데이터 점 $(x_0, y_0), \ldots, (x_n, y_n)$이 주어졌을 때, 모든 점을 지나는 다항식 $p(x) = c_0 + c_1 x + \cdots + c_n x^n$을 구합니다. 이것은 **반데르몽드 시스템**으로 이어집니다:

$$\begin{bmatrix} 1 & x_0 & x_0^2 & \cdots & x_0^n \\ 1 & x_1 & x_1^2 & \cdots & x_1^n \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_n & x_n^2 & \cdots & x_n^n \end{bmatrix} \begin{bmatrix} c_0 \\ c_1 \\ \vdots \\ c_n \end{bmatrix} = \begin{bmatrix} y_0 \\ y_1 \\ \vdots \\ y_n \end{bmatrix}$$

```python
import matplotlib.pyplot as plt

# Data points
x_data = np.array([0, 1, 2, 3, 4], dtype=float)
y_data = np.array([1, 3, 2, 5, 4], dtype=float)

# Build Vandermonde matrix
n = len(x_data) - 1
V = np.vander(x_data, increasing=True)
print(f"Vandermonde matrix:\n{V}")

# Solve for coefficients
coeffs = np.linalg.solve(V, y_data)
print(f"Polynomial coefficients: {coeffs}")

# Plot
x_fine = np.linspace(-0.5, 4.5, 200)
V_fine = np.vander(x_fine, N=n+1, increasing=True)
y_fine = V_fine @ coeffs

plt.figure(figsize=(8, 5))
plt.plot(x_data, y_data, 'ro', markersize=8, label='Data points')
plt.plot(x_fine, y_fine, 'b-', label=f'Degree-{n} polynomial')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.grid(True, alpha=0.3)
plt.title('Polynomial Interpolation via Vandermonde System')
plt.show()
```

---

## 9. 피벗팅 전략

### 9.1 왜 피벗팅이 필요한가?

피벗팅 없이는, 피벗이 0이거나 다른 원소에 비해 매우 작을 때 가우스 소거법이 실패하거나 매우 부정확한 결과를 낼 수 있습니다.

### 9.2 부분 피벗팅 (Partial Pivoting)

각 단계에서 피벗 열의 절댓값이 가장 큰 행과 현재 행을 교환합니다. 이것은 실제로 사용되는 표준 전략이며 NumPy/SciPy에서 구현하는 방식입니다.

### 9.3 완전 피벗팅 (Complete Pivoting)

남은 부분 행렬 전체에서 절댓값이 가장 큰 원소를 찾아 행과 열을 모두 교환합니다. 더 안정적이지만 비용이 더 듭니다. 실제로는 거의 필요하지 않습니다.

```python
def gaussian_partial_pivot(A, b):
    """Gaussian elimination with partial pivoting -- production quality."""
    n = len(b)
    Ab = np.column_stack([A.astype(float), b.astype(float)])

    for col in range(n):
        # Partial pivoting
        max_idx = col + np.argmax(np.abs(Ab[col:, col]))
        Ab[[col, max_idx]] = Ab[[max_idx, col]]

        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:]) / Ab[i, i]
    return x

# Test with a matrix that needs pivoting
A = np.array([[1e-20, 1],
              [1,     1]], dtype=float)
b = np.array([1, 2], dtype=float)

x_pivot = gaussian_partial_pivot(A, b)
x_numpy = np.linalg.solve(A, b)
print(f"With pivoting: {x_pivot}")
print(f"NumPy:         {x_numpy}")
```

---

## 10. 요약

| 개념 | 설명 |
|------|------|
| $A\mathbf{x} = \mathbf{b}$ | 연립방정식의 행렬 형태 |
| 가우스 소거법 | 행 축소를 통한 사다리꼴 변환 |
| REF / RREF | 행 사다리꼴 및 기약 행 사다리꼴 |
| 랭크 조건 | 해의 존재성/유일성 판별 |
| LU 분해 | $PA = LU$로 반복 풀이의 효율화 |
| 영공간 | $\{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ |
| 최소자승법 | $\hat{\mathbf{x}} = (A^TA)^{-1}A^T\mathbf{b}$ (과대결정 시스템) |
| 부분 피벗팅 | 수치적 안정성을 위한 행 교환 |

---

## 연습 문제

### 연습 문제 1: 가우스 소거법

다음 시스템을 가우스 소거법으로 손으로 풀고, NumPy로 검증하세요:

$$\begin{cases} x + 2y + z = 9 \\ 2x - y + 3z = 8 \\ 3x + y - z = 3 \end{cases}$$

### 연습 문제 2: RREF 분석

$\begin{bmatrix} 1 & 2 & 3 & 4 \\ 2 & 4 & 7 & 9 \\ 3 & 6 & 10 & 13 \end{bmatrix}$의 RREF를 구하고, 랭크, 피벗 열, 자유 변수를 결정하세요.

### 연습 문제 3: LU 분해

1. $A = \begin{bmatrix} 2 & 4 & -2 \\ 4 & 9 & -3 \\ -2 & -3 & 7 \end{bmatrix}$의 LU 분해를 계산하세요.
2. 이 분해를 사용하여 $\mathbf{b} = [2, 8, 10]^T$와 $\mathbf{b} = [4, 8, -2]^T$에 대해 $A\mathbf{x} = \mathbf{b}$를 풀으세요.

### 연습 문제 4: 최소자승 직선 피팅

정규방정식을 사용하여 데이터 점 $(1, 2.1)$, $(2, 3.9)$, $(3, 6.2)$, $(4, 7.8)$, $(5, 10.1)$에 직선 $y = ax + b$를 피팅하세요. 데이터와 피팅된 직선을 그래프로 나타내세요.

### 연습 문제 5: 영공간

$A = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 2 & 4 & 6 & 8 \\ 1 & 2 & 4 & 6 \end{bmatrix}$의 영공간에 대한 기저를 구하세요. 영공간의 차원은 무엇입니까?

---

[<< 이전: 레슨 2 - 행렬과 연산](02_Matrices_and_Operations.md) | [개요](00_Overview.md) | [다음: 레슨 4 - 벡터 노름과 내적 >>](04_Vector_Norms_and_Inner_Products.md)

**License**: CC BY-NC 4.0
