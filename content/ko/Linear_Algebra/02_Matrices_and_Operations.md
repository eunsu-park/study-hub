# 레슨 2: 행렬과 연산 (Matrices and Operations)

## 학습 목표

- 행렬을 정의하고 NumPy로 덧셈, 스칼라 곱, 행렬 곱셈을 수행할 수 있다
- 행렬의 전치, 역행렬, 행렬식, 대각합을 계산할 수 있다
- 특수 행렬(단위 행렬, 대각 행렬, 대칭 행렬, 반대칭 행렬, 직교 행렬, 삼각 행렬)을 인식하고 구성할 수 있다
- 행렬 대수의 규칙(결합법칙, 비교환성 등)을 이해할 수 있다
- 행렬 연산을 실용적인 문제 해결에 적용할 수 있다

---

## 1. 행렬의 기초

### 1.1 정의

$m \times n$ 행렬 $A$는 $m$개의 행과 $n$개의 열로 배열된 실수들의 직사각형 배열입니다:

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

$a_{ij}$ 또는 $(A)_{ij}$는 $i$행 $j$열의 원소를 나타냅니다.

```python
import numpy as np

# Creating matrices
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"A =\n{A}")
print(f"Shape: {A.shape}")      # (2, 3)
print(f"Element (1,2): {A[0, 1]}")  # 2 (0-indexed)

# Useful constructors
Z = np.zeros((3, 4))      # 3x4 zero matrix
O = np.ones((2, 3))       # 2x3 all-ones matrix
R = np.random.randn(3, 3) # 3x3 random matrix
print(f"Random 3x3:\n{R}")
```

### 1.2 행렬 덧셈과 스칼라 곱

**같은 크기**의 행렬끼리 원소별로 더할 수 있으며, 임의의 행렬에 스칼라를 곱할 수 있습니다:

$$C = A + B \implies c_{ij} = a_{ij} + b_{ij}$$
$$D = \alpha A \implies d_{ij} = \alpha \, a_{ij}$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"A + B =\n{A + B}")
print(f"3A =\n{3 * A}")
print(f"A - B =\n{A - B}")
```

---

## 2. 행렬 곱셈

### 2.1 정의

$A \in \mathbb{R}^{m \times n}$이고 $B \in \mathbb{R}^{n \times p}$일 때, 곱 $C = AB \in \mathbb{R}^{m \times p}$는 다음과 같이 정의됩니다:

$$c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

$A$의 **열** 수가 $B$의 **행** 수와 같아야 합니다.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])     # 2x3
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])      # 3x2

C = A @ B   # preferred syntax; also np.matmul(A, B) or np.dot(A, B)
print(f"A @ B =\n{C}")       # 2x2
print(f"Shape: {C.shape}")

# Manual verification of C[0,0]
print(f"C[0,0] = 1*7 + 2*9 + 3*11 = {1*7 + 2*9 + 3*11}")
```

### 2.2 행렬 곱셈의 성질

| 성질 | 식 |
|------|-----|
| 결합법칙 | $(AB)C = A(BC)$ |
| 좌분배법칙 | $A(B + C) = AB + AC$ |
| 우분배법칙 | $(A + B)C = AC + BC$ |
| 스칼라 호환성 | $\alpha(AB) = (\alpha A)B = A(\alpha B)$ |
| **비교환성** | $AB \neq BA$ (일반적으로) |

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"AB =\n{A @ B}")
print(f"BA =\n{B @ A}")
print(f"AB == BA? {np.allclose(A @ B, B @ A)}")  # False

# Associativity check
C = np.array([[1, 0], [0, 1]])
print(f"(AB)C == A(BC)? {np.allclose((A @ B) @ C, A @ (B @ C))}")  # True
```

### 2.3 행렬 곱셈의 해석

$C = AB$를 보는 네 가지 동등한 관점이 있습니다:

1. **행-열 내적**: $c_{ij} = (A\text{의 행 } i) \cdot (B\text{의 열 } j)$
2. **열 관점**: $C$의 각 열은 $A$의 열들의 선형 결합
3. **행 관점**: $C$의 각 행은 $B$의 행들의 선형 결합
4. **외적 합**: $C = \sum_k (A\text{의 열 } k)(B\text{의 행 } k)$

```python
# Column picture: column j of C is A times column j of B
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A @ B
print(f"Column 0 of C: {C[:, 0]}")
print(f"A @ col 0 of B: {A @ B[:, 0]}")  # same

# Outer product sum
outer_sum = np.outer(A[:, 0], B[0, :]) + np.outer(A[:, 1], B[1, :])
print(f"Outer product sum:\n{outer_sum}")
print(f"Matches C? {np.allclose(C, outer_sum)}")
```

### 2.4 원소별 곱 (Hadamard Product)

원소별 곱 $A \odot B$는 대응하는 원소끼리 곱합니다. 이것은 행렬 곱셈과 **다릅니다**.

$$(\mathbf{A} \odot \mathbf{B})_{ij} = a_{ij} \, b_{ij}$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

hadamard = A * B   # element-wise in NumPy
matmul = A @ B     # matrix multiplication

print(f"Hadamard product:\n{hadamard}")
print(f"Matrix product:\n{matmul}")
```

---

## 3. 전치 (Transpose)

$A \in \mathbb{R}^{m \times n}$의 **전치** $A^T \in \mathbb{R}^{n \times m}$는 다음과 같이 정의됩니다:

$$(A^T)_{ij} = A_{ji}$$

### 성질

| 성질 | 식 |
|------|-----|
| 이중 전치 | $(A^T)^T = A$ |
| 합 | $(A + B)^T = A^T + B^T$ |
| 스칼라 | $(\alpha A)^T = \alpha A^T$ |
| 곱 | $(AB)^T = B^T A^T$ (순서 반전!) |

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A_T = A.T
print(f"A =\n{A}")
print(f"A^T =\n{A_T}")
print(f"Shape: {A.shape} -> {A_T.shape}")

# Product transpose rule
B = np.array([[1, 0],
              [0, 1],
              [1, 1]])
lhs = (A @ B).T
rhs = B.T @ A.T
print(f"(AB)^T == B^T A^T? {np.allclose(lhs, rhs)}")
```

---

## 4. 행렬식 (Determinant)

### 4.1 정의

**행렬식**은 정방행렬에 대해 정의되는 스칼라 값 함수입니다. $2 \times 2$ 행렬의 경우:

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

더 큰 행렬의 경우 여인수 전개 또는 행 축소를 통해 계산할 수 있습니다.

### 4.2 기하학적 해석

$|\det(A)|$는 $A$가 나타내는 선형 변환의 **부피 변화율**을 나타냅니다:
- $|\det(A)| = 0$: 변환이 공간을 축소 (행렬이 특이)
- $\det(A) < 0$: 변환이 방향을 반전

### 4.3 성질

| 성질 | 식 |
|------|-----|
| 곱 법칙 | $\det(AB) = \det(A) \det(B)$ |
| 전치 | $\det(A^T) = \det(A)$ |
| 역행렬 | $\det(A^{-1}) = 1 / \det(A)$ |
| 스칼라 | $\det(\alpha A) = \alpha^n \det(A)$ ($n \times n$ 행렬) |
| 삼각 행렬 | 행렬식 = 대각 원소의 곱 |

```python
# 2x2 determinant
A = np.array([[3, 1],
              [2, 4]])
det_A = np.linalg.det(A)
print(f"det(A) = {det_A:.4f}")  # 3*4 - 1*2 = 10

# 3x3 determinant
B = np.array([[1, 2, 3],
              [0, 4, 5],
              [0, 0, 6]])
print(f"det(B) = {np.linalg.det(B):.4f}")  # 1*4*6 = 24 (triangular)

# Singular matrix
C = np.array([[1, 2],
              [2, 4]])
print(f"det(C) = {np.linalg.det(C):.10f}")  # ~0

# Product rule
print(f"det(A)*det(B[:2,:2]) = {det_A * np.linalg.det(B[:2,:2]):.4f}")
```

### 4.4 행렬식 시각화

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Unit square vertices
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

matrices = [
    (np.array([[2, 0], [0, 1]]), "det = 2 (stretch)"),
    (np.array([[1, 1], [0, 1]]), "det = 1 (shear)"),
    (np.array([[1, 2], [0.5, 1]]), "det = 0 (singular)"),
]

for ax, (M, title) in zip(axes, matrices):
    d = np.linalg.det(M)
    transformed = (M @ square.T).T

    ax.fill(*square.T, alpha=0.3, color='blue', label='Original')
    ax.fill(*transformed.T, alpha=0.3, color='red', label='Transformed')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"{title}\n|det| = {abs(d):.2f}")

plt.tight_layout()
plt.show()
```

---

## 5. 역행렬 (Inverse Matrix)

### 5.1 정의

정방행렬 $A$의 **역행렬** $A^{-1}$은 다음을 만족합니다:

$$AA^{-1} = A^{-1}A = I$$

역행렬은 $\det(A) \neq 0$ (행렬이 **비특이** 또는 **가역**)인 경우에만 존재합니다.

### 5.2 성질

| 성질 | 식 |
|------|-----|
| 유일성 | 역행렬이 존재하면 유일함 |
| 대합 | $(A^{-1})^{-1} = A$ |
| 곱 | $(AB)^{-1} = B^{-1}A^{-1}$ (순서 반전!) |
| 전치 | $(A^T)^{-1} = (A^{-1})^T$ |

### 5.3 역행렬 계산

$2 \times 2$ 행렬의 경우:

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

```python
A = np.array([[4, 7],
              [2, 6]])

A_inv = np.linalg.inv(A)
print(f"A^(-1) =\n{A_inv}")
print(f"A @ A^(-1) =\n{A @ A_inv}")       # identity
print(f"A^(-1) @ A =\n{A_inv @ A}")       # identity

# Verify with 2x2 formula
det_A = np.linalg.det(A)
A_inv_formula = (1 / det_A) * np.array([[6, -7], [-2, 4]])
print(f"Formula matches? {np.allclose(A_inv, A_inv_formula)}")

# Singular matrix -- no inverse
B = np.array([[1, 2], [2, 4]])
try:
    np.linalg.inv(B)
except np.linalg.LinAlgError as e:
    print(f"Cannot invert singular matrix: {e}")
```

---

## 6. 대각합 (Trace)

정방행렬의 **대각합**은 대각 원소의 합입니다:

$$\mathrm{tr}(A) = \sum_{i=1}^n a_{ii}$$

### 성질

| 성질 | 식 |
|------|-----|
| 선형성 | $\mathrm{tr}(A + B) = \mathrm{tr}(A) + \mathrm{tr}(B)$ |
| 스칼라 | $\mathrm{tr}(\alpha A) = \alpha \, \mathrm{tr}(A)$ |
| 전치 | $\mathrm{tr}(A^T) = \mathrm{tr}(A)$ |
| 순환 성질 | $\mathrm{tr}(ABC) = \mathrm{tr}(CAB) = \mathrm{tr}(BCA)$ |
| 고유값 연결 | $\mathrm{tr}(A) = \sum_i \lambda_i$ |

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(f"tr(A) = {np.trace(A)}")  # 1 + 5 + 9 = 15

# Cyclic property
B = np.random.randn(3, 3)
C = np.random.randn(3, 3)
print(f"tr(ABC) = {np.trace(A @ B @ C):.6f}")
print(f"tr(CAB) = {np.trace(C @ A @ B):.6f}")
print(f"tr(BCA) = {np.trace(B @ C @ A):.6f}")

# Trace equals sum of eigenvalues
eigenvalues = np.linalg.eigvals(A)
print(f"Sum of eigenvalues = {np.sum(eigenvalues).real:.6f}")
print(f"tr(A) = {np.trace(A)}")
```

---

## 7. 특수 행렬

### 7.1 단위 행렬 (Identity Matrix)

$n \times n$ 단위 행렬 $I_n$은 대각선에 1이 있고 나머지는 0입니다. 곱셈의 항등원입니다: $AI = IA = A$.

```python
I3 = np.eye(3)
print(f"I_3 =\n{I3}")

A = np.random.randn(3, 3)
print(f"A @ I == A? {np.allclose(A @ I3, A)}")
```

### 7.2 대각 행렬 (Diagonal Matrix)

대각 행렬 $D$는 주대각선에만 0이 아닌 원소를 가집니다:

$$D = \mathrm{diag}(d_1, d_2, \ldots, d_n)$$

대각 행렬은 역행렬 계산, 거듭제곱, 곱셈이 쉽습니다.

```python
d = np.array([2, 3, 5])
D = np.diag(d)
print(f"D =\n{D}")

# Inverse of diagonal matrix
D_inv = np.diag(1.0 / d)
print(f"D^(-1) =\n{D_inv}")
print(f"D @ D^(-1) =\n{D @ D_inv}")

# Powers of diagonal matrix
D_cubed = np.diag(d**3)
print(f"D^3 =\n{D_cubed}")
```

### 7.3 대칭 행렬 (Symmetric Matrix)

행렬이 $A = A^T$, 즉 $a_{ij} = a_{ji}$이면 **대칭 행렬**입니다.

대칭 행렬은 여러 중요한 성질을 가집니다:
- 모든 고유값이 실수
- 서로 다른 고유값에 대응하는 고유벡터는 직교
- 직교 행렬로 대각화 가능

```python
# Creating a symmetric matrix
A = np.random.randn(3, 3)
S = (A + A.T) / 2   # guaranteed symmetric
print(f"S =\n{S}")
print(f"S == S^T? {np.allclose(S, S.T)}")

# Eigenvalues are real
eigenvalues = np.linalg.eigvals(S)
print(f"Eigenvalues: {eigenvalues}")
print(f"All real? {np.all(np.isreal(eigenvalues))}")
```

### 7.4 반대칭 행렬 (Skew-Symmetric Matrix)

$A = -A^T$이면 **반대칭 행렬**입니다. 반대칭 행렬의 대각 원소는 항상 0입니다.

```python
A = np.random.randn(3, 3)
K = (A - A.T) / 2   # guaranteed skew-symmetric
print(f"K =\n{np.round(K, 4)}")
print(f"K == -K^T? {np.allclose(K, -K.T)}")
print(f"Diagonal: {np.diag(K)}")  # all ~0
```

### 7.5 직교 행렬 (Orthogonal Matrix)

정방행렬 $Q$가 $Q^T Q = Q Q^T = I$를 만족하면 **직교 행렬**입니다. 즉 $Q^{-1} = Q^T$입니다.

직교 행렬은 길이와 각도를 보존합니다 -- 회전과 반사를 나타냅니다.

```python
# 2D rotation matrix
theta = np.pi / 4  # 45 degrees
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print(f"Q =\n{Q}")
print(f"Q^T @ Q =\n{np.round(Q.T @ Q, 10)}")   # identity
print(f"det(Q) = {np.linalg.det(Q):.6f}")        # +1 (rotation)

# Orthogonal matrices preserve vector norms
v = np.array([3, 4])
Qv = Q @ v
print(f"||v|| = {np.linalg.norm(v):.4f}")
print(f"||Qv|| = {np.linalg.norm(Qv):.4f}")  # same
```

### 7.6 삼각 행렬 (Triangular Matrices)

**상삼각 행렬**은 대각선 아래의 모든 원소가 0이고, **하삼각 행렬**은 대각선 위의 모든 원소가 0입니다.

$$U = \begin{bmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{bmatrix}, \quad L = \begin{bmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{bmatrix}$$

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

U = np.triu(A)  # upper triangular
L = np.tril(A)  # lower triangular

print(f"Upper triangular:\n{U}")
print(f"Lower triangular:\n{L}")

# Determinant of triangular matrix = product of diagonal
print(f"det(U) = {np.linalg.det(U):.4f}")
print(f"Product of diagonal: {np.prod(np.diag(U))}")
```

---

## 8. 행렬의 랭크 (Rank)

행렬의 **랭크**는 선형 독립인 행(또는 동등하게 열)의 최대 개수입니다.

$$\mathrm{rank}(A) \le \min(m, n) \quad \text{for } A \in \mathbb{R}^{m \times n}$$

$\mathrm{rank}(A) = \min(m, n)$이면 행렬이 **full rank**라 합니다.

```python
# Full rank matrix
A = np.array([[1, 2],
              [3, 4]])
print(f"rank(A) = {np.linalg.matrix_rank(A)}")  # 2

# Rank-deficient matrix
B = np.array([[1, 2, 3],
              [4, 5, 6],
              [5, 7, 9]])  # row 3 = row 1 + row 2
print(f"rank(B) = {np.linalg.matrix_rank(B)}")  # 2

# Rank properties
C = np.random.randn(3, 5)
print(f"C is 3x5, rank(C) = {np.linalg.matrix_rank(C)}")  # at most 3
print(f"rank(C) = rank(C^T)? {np.linalg.matrix_rank(C) == np.linalg.matrix_rank(C.T)}")
```

---

## 9. 블록 행렬 (Block Matrices)

행렬을 **블록** (부분 행렬)으로 분할할 수 있습니다. 블록의 크기가 호환되면 블록 곱셈은 스칼라 곱셈과 동일한 규칙을 따릅니다.

$$\begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix} \begin{bmatrix} B_{11} \\ B_{21} \end{bmatrix} = \begin{bmatrix} A_{11}B_{11} + A_{12}B_{21} \\ A_{21}B_{11} + A_{22}B_{21} \end{bmatrix}$$

```python
# Block matrix construction
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
Z = np.zeros((2, 2))
I = np.eye(2)

# Build a 4x4 block matrix
M = np.block([[A, B],
              [Z, I]])
print(f"Block matrix M =\n{M}")
print(f"Shape: {M.shape}")
```

---

## 10. 요약

| 연산 | 표기법 | NumPy |
|------|--------|-------|
| 행렬 곱셈 | $AB$ | `A @ B` |
| 원소별 곱 | $A \odot B$ | `A * B` |
| 전치 | $A^T$ | `A.T` |
| 역행렬 | $A^{-1}$ | `np.linalg.inv(A)` |
| 행렬식 | $\det(A)$ | `np.linalg.det(A)` |
| 대각합 | $\mathrm{tr}(A)$ | `np.trace(A)` |
| 랭크 | $\mathrm{rank}(A)$ | `np.linalg.matrix_rank(A)` |
| 단위 행렬 | $I_n$ | `np.eye(n)` |
| 대각 행렬 | $\mathrm{diag}(d)$ | `np.diag(d)` |

---

## 연습 문제

### 연습 문제 1: 행렬 산술

$A = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$와 $B = \begin{bmatrix} 5 & 7 \\ 6 & 8 \end{bmatrix}$가 주어졌을 때,

$AB$, $BA$, $A^T B$를 계산하고 $AB \neq BA$임을 확인하세요.

### 연습 문제 2: 행렬식과 가역성

행렬 $M = \begin{bmatrix} 2 & 1 & 0 \\ 1 & 3 & 1 \\ 0 & 1 & 2 \end{bmatrix}$에 대해:

1. $\det(M)$을 계산하세요.
2. $M$이 가역인지 판별하세요.
3. 가역이면 $M^{-1}$을 계산하고 $M M^{-1} = I$를 확인하세요.

### 연습 문제 3: 특수 행렬

1. 임의의 정방행렬 $A$를 $A = S + K$ (여기서 $S$는 대칭, $K$는 반대칭)로 표현할 수 있음을 보이세요.
2. 임의의 $4 \times 4$ 행렬에 대해 이 분해를 수치적으로 검증하세요.

### 연습 문제 4: 대각합의 성질

임의의 $3 \times 3$ 행렬 $A$, $B$, $C$에 대해 다음을 수치적으로 검증하세요:

1. $\mathrm{tr}(AB) = \mathrm{tr}(BA)$
2. $\mathrm{tr}(ABC) = \mathrm{tr}(BCA) = \mathrm{tr}(CAB)$
3. $\mathrm{tr}(A)$가 $A$의 고유값의 합과 같음

### 연습 문제 5: 랭크 탐구

다음 성질을 가진 행렬을 구성하고 NumPy로 검증하세요:

1. 랭크가 1인 $3 \times 3$ 행렬
2. 랭크가 2인 $4 \times 2$ 행렬
3. 랭크가 2이고 행렬식이 0인 $3 \times 3$ 행렬

---

[<< 이전: 레슨 1 - 벡터와 벡터 공간](01_Vectors_and_Vector_Spaces.md) | [개요](00_Overview.md) | [다음: 레슨 3 - 연립일차방정식 >>](03_Systems_of_Linear_Equations.md)

**License**: CC BY-NC 4.0
