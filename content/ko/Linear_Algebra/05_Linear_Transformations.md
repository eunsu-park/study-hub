# 레슨 5: 선형 변환 (Linear Transformations)

## 학습 목표

- 선형 변환을 정의하고 선형성 조건을 검증할 수 있다
- 선형 변환을 행렬로 표현하고 행렬 곱셈을 통해 변환을 합성할 수 있다
- 변환의 핵(영공간)과 상(열공간)을 계산할 수 있다
- 차원 정리(rank-nullity theorem)를 설명하고 적용할 수 있다
- 주요 2D 변환(회전, 스케일링, 반사, 전단, 투영)을 시각화할 수 있다

---

## 1. 선형 변환의 정의

### 1.1 선형 변환이란 무엇인가?

함수 $T : \mathbb{R}^n \to \mathbb{R}^m$이 모든 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$과 $c \in \mathbb{R}$에 대해 다음 두 성질을 만족하면 **선형 변환**입니다:

1. **가법성**: $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **동차성**: $T(c\mathbf{v}) = c \, T(\mathbf{v})$

동등하게, $T$가 선형인 것은 다음과 동치입니다:

$$T(\alpha\mathbf{u} + \beta\mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})$$

선형 변환은 항상 영벡터를 영벡터로 보냅니다: $T(\mathbf{0}) = \mathbf{0}$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Example: T(x, y) = (2x + y, x - y)
def T(v):
    return np.array([2*v[0] + v[1], v[0] - v[1]])

# Verify linearity
u = np.array([1, 3])
v = np.array([2, -1])
alpha, beta = 3, -2

lhs = T(alpha * u + beta * v)
rhs = alpha * T(u) + beta * T(v)
print(f"T(alpha*u + beta*v) = {lhs}")
print(f"alpha*T(u) + beta*T(v) = {rhs}")
print(f"Linear? {np.allclose(lhs, rhs)}")

# Counter-example: T(v) = v + [1, 0] is NOT linear
def T_affine(v):
    return v + np.array([1, 0])

lhs = T_affine(u + v)
rhs = T_affine(u) + T_affine(v)
print(f"\nAffine T(u+v) = {lhs}")
print(f"T(u) + T(v) = {rhs}")
print(f"Linear? {np.allclose(lhs, rhs)}")  # False
```

---

## 2. 행렬 표현

### 2.1 모든 선형 변환에는 행렬이 있다

임의의 선형 변환 $T : \mathbb{R}^n \to \mathbb{R}^m$에 대해, 다음을 만족하는 유일한 $m \times n$ 행렬 $A$가 존재합니다:

$$T(\mathbf{v}) = A\mathbf{v}$$

$A$의 열은 표준 기저 벡터의 상(image)입니다:

$$A = \begin{bmatrix} T(\mathbf{e}_1) & T(\mathbf{e}_2) & \cdots & T(\mathbf{e}_n) \end{bmatrix}$$

```python
# Find the matrix of T(x, y) = (2x + y, x - y)
e1 = np.array([1, 0])
e2 = np.array([0, 1])

col1 = T(e1)  # [2, 1]
col2 = T(e2)  # [1, -1]

A = np.column_stack([col1, col2])
print(f"Matrix A:\n{A}")

# Verify
v = np.array([3, 5])
print(f"T(v) = {T(v)}")
print(f"A @ v = {A @ v}")
```

### 2.2 주요 2D 변환 행렬

| 변환 | 행렬 | 효과 |
|------|------|------|
| $\theta$ 회전 | $\begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$ | 반시계 방향 회전 |
| 스케일링 | $\begin{bmatrix}s_x & 0 \\ 0 & s_y\end{bmatrix}$ | 축을 따른 크기 변환 |
| 반사 (x축) | $\begin{bmatrix}1 & 0 \\ 0 & -1\end{bmatrix}$ | 상하 반전 |
| 반사 (y축) | $\begin{bmatrix}-1 & 0 \\ 0 & 1\end{bmatrix}$ | 좌우 반전 |
| 전단 (x방향) | $\begin{bmatrix}1 & k \\ 0 & 1\end{bmatrix}$ | 수평 방향 기울임 |
| x축 투영 | $\begin{bmatrix}1 & 0 \\ 0 & 0\end{bmatrix}$ | y성분 제거 |

```python
theta = np.pi / 6  # 30 degrees

transforms = {
    'Rotation (30 deg)': np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta),  np.cos(theta)]]),
    'Scale (2x, 0.5y)': np.array([[2, 0],
                                   [0, 0.5]]),
    'Reflect (x-axis)':  np.array([[1,  0],
                                   [0, -1]]),
    'Shear (k=0.5)':     np.array([[1, 0.5],
                                   [0, 1]]),
    'Projection (x)':    np.array([[1, 0],
                                   [0, 0]]),
}

# Original shape: unit square
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, (name, M) in zip(axes, transforms.items()):
    transformed = M @ square

    ax.fill(square[0], square[1], alpha=0.3, color='blue', label='Original')
    ax.fill(transformed[0], transformed[1], alpha=0.3, color='red', label='Transformed')
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(name, fontsize=9)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.show()
```

---

## 3. 변환의 합성

### 3.1 두 변환의 합성

$T_1(\mathbf{v}) = A\mathbf{v}$이고 $T_2(\mathbf{v}) = B\mathbf{v}$이면, 합성 $T_2 \circ T_1$은 다음과 같습니다:

$$(T_2 \circ T_1)(\mathbf{v}) = B(A\mathbf{v}) = (BA)\mathbf{v}$$

**순서에 주의**: $A$를 먼저 적용하고 $B$를 적용하지만, 결합 행렬은 $BA$입니다 (오른쪽에서 왼쪽으로).

```python
# Rotate 45 degrees, then scale by 2 in x
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
S = np.array([[2, 0],
              [0, 1]])

# Composition: first R, then S
M_compose = S @ R

# Verify
v = np.array([1, 0])
step1 = R @ v
step2 = S @ step1
combined = M_compose @ v

print(f"Step-by-step: R@v = {step1}, S@(R@v) = {step2}")
print(f"Combined:     (S@R)@v = {combined}")
print(f"Match? {np.allclose(step2, combined)}")

# Visualize
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].fill(square[0], square[1], alpha=0.3, color='blue')
axes[0].set_title('Original')

rotated = R @ square
axes[1].fill(rotated[0], rotated[1], alpha=0.3, color='green')
axes[1].set_title('After Rotation (45 deg)')

composed = M_compose @ square
axes[2].fill(composed[0], composed[1], alpha=0.3, color='red')
axes[2].set_title('After Rotation + Scale')

for ax in axes:
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1, 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.2 순서의 중요성

행렬 곱셈이 교환법칙을 만족하지 않으므로 합성의 순서가 중요합니다:

$$T_1 \circ T_2 \neq T_2 \circ T_1 \quad \text{(일반적으로)}$$

```python
# Rotation then scale vs scale then rotation
SR = S @ R  # rotate first, then scale
RS = R @ S  # scale first, then rotate

print(f"S @ R =\n{np.round(SR, 4)}")
print(f"R @ S =\n{np.round(RS, 4)}")
print(f"Same? {np.allclose(SR, RS)}")
```

---

## 4. 핵 (Null Space)

### 4.1 정의

선형 변환 $T(\mathbf{v}) = A\mathbf{v}$의 **핵** (또는 **영공간**)은 다음과 같습니다:

$$\ker(T) = \mathrm{null}(A) = \{\mathbf{v} \in \mathbb{R}^n : A\mathbf{v} = \mathbf{0}\}$$

핵은 항상 $\mathbb{R}^n$의 부분공간입니다. 그 차원을 $A$의 **퇴화 차수(nullity)**라 합니다.

### 4.2 해석

- $\ker(T) = \{\mathbf{0}\}$이면, 변환은 **단사(injective, one-to-one)**입니다: 서로 다른 입력이 서로 다른 출력으로 매핑됩니다.
- $\ker(T)$가 비자명이면, 변환이 정의역의 일부를 "축소"시킵니다.

```python
from scipy.linalg import null_space

# Projection onto x-axis: kills the y-component
P = np.array([[1, 0],
              [0, 0]], dtype=float)

ns = null_space(P)
print(f"Kernel of projection:\n{ns}")
print(f"Nullity = {ns.shape[1]}")

# Rank-2 matrix in R^3 -> R^3 (has non-trivial kernel)
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

ns_A = null_space(A)
print(f"\nKernel of A:\n{ns_A}")
print(f"Nullity = {ns_A.shape[1]}")

# Verify kernel vector is in null space
if ns_A.shape[1] > 0:
    print(f"A @ kernel_vec = {A @ ns_A[:, 0]}")  # should be ~0
```

---

## 5. 상 (Column Space)

### 5.1 정의

선형 변환 $T(\mathbf{v}) = A\mathbf{v}$의 **상** (또는 **열공간**, **치역**)은 다음과 같습니다:

$$\mathrm{im}(T) = \mathrm{col}(A) = \{A\mathbf{v} : \mathbf{v} \in \mathbb{R}^n\}$$

이것은 $A$의 열들의 생성(span)입니다. 그 차원이 $A$의 **랭크**입니다.

### 5.2 열공간의 기저 계산

$A$를 사다리꼴로 축소하고 피벗 열을 식별합니다. **원래** 행렬 $A$의 해당 열이 $\mathrm{col}(A)$의 기저를 이룹니다.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

rank = np.linalg.matrix_rank(A)
print(f"Rank of A: {rank}")

# Use SVD to find an orthonormal basis for the column space
U, s, Vt = np.linalg.svd(A)
col_space_basis = U[:, :rank]
print(f"Orthonormal basis for col(A):\n{col_space_basis}")

# Verify: any column of A should be in the column space
# Project column 2 onto the column space
col2 = A[:, 2]
proj = col_space_basis @ (col_space_basis.T @ col2)
print(f"\nColumn 2 of A: {col2}")
print(f"Projection onto col(A): {proj}")
print(f"In column space? {np.allclose(col2, proj)}")
```

---

## 6. 차원 정리 (Rank-Nullity Theorem)

### 6.1 식

$m \times n$ 행렬 $A$에 대해:

$$\mathrm{rank}(A) + \mathrm{nullity}(A) = n$$

동등하게:

$$\dim(\mathrm{im}(T)) + \dim(\ker(T)) = \dim(\text{정의역})$$

### 6.2 직관

변환의 정의역은 두 부분으로 "분리"됩니다:
- 서로 다른 출력으로 매핑되는 부분 (차원 = 랭크)
- 0으로 축소되는 부분 (차원 = 퇴화 차수)

### 6.3 검증

```python
def verify_rank_nullity(A):
    """Verify the rank-nullity theorem for matrix A."""
    n = A.shape[1]  # number of columns = dimension of domain
    rank = np.linalg.matrix_rank(A)
    ns = null_space(A)
    nullity = ns.shape[1]

    print(f"Matrix shape: {A.shape}")
    print(f"n (domain dim): {n}")
    print(f"Rank:           {rank}")
    print(f"Nullity:        {nullity}")
    print(f"Rank + Nullity = {rank + nullity} == n = {n}: {rank + nullity == n}")
    print()

# Full rank square matrix
A1 = np.array([[1, 2], [3, 4]], dtype=float)
verify_rank_nullity(A1)

# Rank-deficient square matrix
A2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
verify_rank_nullity(A2)

# Rectangular matrix (wide)
A3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
verify_rank_nullity(A3)

# Rectangular matrix (tall)
A4 = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
verify_rank_nullity(A4)
```

---

## 7. 네 가지 기본 부분공간

모든 $m \times n$ 행렬 $A$는 네 가지 기본 부분공간을 정의합니다 (Gilbert Strang):

| 부분공간 | 정의 | 차원 |
|---------|------|------|
| 열공간 $\mathrm{col}(A)$ | $\{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}$ | $r$ |
| 영공간 $\mathrm{null}(A)$ | $\{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ | $n - r$ |
| 행공간 $\mathrm{col}(A^T)$ | $\{A^T\mathbf{y} : \mathbf{y} \in \mathbb{R}^m\}$ | $r$ |
| 좌영공간 $\mathrm{null}(A^T)$ | $\{\mathbf{y} : A^T\mathbf{y} = \mathbf{0}\}$ | $m - r$ |

여기서 $r = \mathrm{rank}(A)$입니다.

**핵심 직교 관계**:
- $\mathrm{col}(A) \perp \mathrm{null}(A^T)$ ($\mathbb{R}^m$ 내)
- $\mathrm{col}(A^T) \perp \mathrm{null}(A)$ ($\mathbb{R}^n$ 내)

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=float)

r = np.linalg.matrix_rank(A)
m, n = A.shape

# Compute all four subspaces via SVD
U, s, Vt = np.linalg.svd(A)

col_space = U[:, :r]           # Column space basis
left_null = U[:, r:]           # Left null space basis
row_space = Vt[:r, :].T       # Row space basis
null_space_A = Vt[r:, :].T    # Null space basis

print(f"Matrix A ({m}x{n}), rank = {r}")
print(f"Column space dim:    {col_space.shape[1]} (in R^{m})")
print(f"Left null space dim: {left_null.shape[1]} (in R^{m})")
print(f"Row space dim:       {row_space.shape[1]} (in R^{n})")
print(f"Null space dim:      {null_space_A.shape[1]} (in R^{n})")

# Verify orthogonality
print(f"\ncol(A)^T @ null(A^T) =\n{np.round(col_space.T @ left_null, 10)}")
print(f"row(A)^T @ null(A) =\n{np.round(row_space.T @ null_space_A, 10)}")
```

---

## 8. 가역 변환

선형 변환 $T : \mathbb{R}^n \to \mathbb{R}^n$이 **가역**인 것은 다음과 동치입니다:

- $\ker(T) = \{\mathbf{0}\}$ (단사)
- $\mathrm{im}(T) = \mathbb{R}^n$ (전사)
- $\mathrm{rank}(A) = n$ (full rank)
- $\det(A) \neq 0$ (비특이)

이 모든 조건은 정방행렬에서 동치입니다.

```python
# Invertible transformation
A = np.array([[2, 1],
              [1, 3]], dtype=float)
print(f"det(A) = {np.linalg.det(A):.4f}")
print(f"rank(A) = {np.linalg.matrix_rank(A)}")

# Apply and then invert
v = np.array([3, 5])
Tv = A @ v
v_recovered = np.linalg.inv(A) @ Tv
print(f"v = {v}")
print(f"T(v) = {Tv}")
print(f"T^(-1)(T(v)) = {v_recovered}")
print(f"Recovered original? {np.allclose(v, v_recovered)}")
```

---

## 9. 기저 변환과 변환 (Change of Basis as a Transformation)

$B = [\mathbf{b}_1 | \cdots | \mathbf{b}_n]$이 기저 변환 행렬이면, 같은 선형 변환 $T$는 다른 기저에서 다른 행렬 표현을 가집니다:

$$A' = B^{-1} A B$$

이것을 **닮음 변환(similarity transformation)**이라 합니다. $A$와 $A'$는 **닮은 행렬(similar matrices)**입니다.

```python
# Same transformation in two different bases
A = np.array([[3, 1],
              [0, 2]], dtype=float)

# Change of basis
B = np.array([[1, 1],
              [1, -1]], dtype=float)

A_prime = np.linalg.inv(B) @ A @ B
print(f"A in standard basis:\n{A}")
print(f"A in new basis:\n{A_prime}")

# Verify: eigenvalues are preserved under similarity
eig_A = np.sort(np.linalg.eigvals(A))
eig_A_prime = np.sort(np.linalg.eigvals(A_prime))
print(f"\nEigenvalues of A:  {eig_A}")
print(f"Eigenvalues of A': {eig_A_prime}")
print(f"Same? {np.allclose(eig_A, eig_A_prime)}")
```

---

## 10. 요약

| 개념 | 설명 |
|------|------|
| 선형 변환 | $T(\alpha\mathbf{u} + \beta\mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})$ |
| 행렬 표현 | $T(\mathbf{v}) = A\mathbf{v}$, $A$의 열은 $T(\mathbf{e}_i)$ |
| 합성 | $(T_2 \circ T_1)(\mathbf{v}) = (B A)\mathbf{v}$ |
| 핵 (영공간) | $\ker(T) = \{\mathbf{v} : A\mathbf{v} = \mathbf{0}\}$ |
| 상 (열공간) | $\mathrm{im}(T) = \{A\mathbf{v} : \mathbf{v} \in \mathbb{R}^n\}$ |
| 차원 정리 | $\mathrm{rank}(A) + \mathrm{nullity}(A) = n$ |
| 닮음 변환 | $A' = B^{-1}AB$는 고유값을 보존 |

---

## 연습 문제

### 연습 문제 1: 변환의 행렬

$T : \mathbb{R}^3 \to \mathbb{R}^2$가 $T(x, y, z) = (x + 2y - z, \; 3x - y + z)$로 정의될 때, 행렬 표현을 구하세요. $T(1, 2, 3)$을 직접 계산한 것과 행렬을 통해 계산한 것이 일치하는지 확인하세요.

### 연습 문제 2: 핵과 상

행렬 $A = \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 3 & 6 & 3 \end{bmatrix}$에 대해:

1. $\ker(A)$의 기저를 구하세요.
2. $\mathrm{im}(A)$의 기저를 구하세요.
3. 차원 정리를 검증하세요.

### 연습 문제 3: 합성

먼저 반시계 방향으로 90도 회전한 다음 x축에 대해 반사하는 하나의 행렬을 구하세요. 이 행렬을 단위 정사각형에 적용하고 결과를 플롯하세요.

### 연습 문제 4: 네 가지 기본 부분공간

$A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}$에 대해, 네 가지 기본 부분공간의 기저를 구하고 직교 관계를 검증하세요.

### 연습 문제 5: 가역성

다음 변환 중 가역인 것을 판별하세요. 가역인 경우 역변환을 구하세요.

(a) $T(x, y) = (2x - y, \; x + 3y)$

(b) $T(x, y) = (x + 2y, \; 2x + 4y)$

(c) $T(x, y, z) = (x + y, \; y + z, \; x + z)$

---

[<< 이전: 레슨 4 - 벡터 노름과 내적](04_Vector_Norms_and_Inner_Products.md) | [개요](00_Overview.md) | [다음: 레슨 6 - 고유값과 고유벡터 >>](06_Eigenvalues_and_Eigenvectors.md)

**License**: CC BY-NC 4.0
