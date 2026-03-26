# 03. 선형대수 (Linear Algebra)

> **Boas Chapter 3** — 물리과학에서 선형대수는 역학의 관성 텐서, 양자역학의 행렬 형식, 연성 진동 분석 등 거의 모든 분야의 기초를 이룹니다.

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

- **행렬 연산**(덧셈, 곱셈, 전치, 역행렬)을 수행하고 행렬식(determinant)을 계산할 수 있다
- **가우스 소거법**과 **크래머 법칙**을 이용하여 연립일차방정식을 풀 수 있다
- **고유값/고유벡터**를 구하고 행렬의 대각화(diagonalization)를 수행할 수 있다
- **대칭 행렬과 에르미트 행렬**의 스펙트럼 정리를 이해하고, 직교/유니터리 행렬의 성질을 활용할 수 있다
- **이차형식**(quadratic form)의 양정치/음정치 판별을 수행할 수 있다
- **물리학 응용**: 관성 텐서의 주축 변환, 연성 진동의 고유 모드, 양자역학의 행렬 역학을 다룰 수 있다

---

## 1. 행렬의 기본

### 1.1 행렬과 기본 연산

**행렬**(matrix)은 수를 직사각형으로 배열한 것입니다. $m \times n$ 행렬 $A$의 원소를 $a_{ij}$로 표기합니다:

$$
A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}
$$

**행렬 덧셈**: 같은 크기의 행렬에 대해 원소별 덧셈

$$
(A + B)_{ij} = a_{ij} + b_{ij}
$$

**스칼라 곱**: 모든 원소에 스칼라를 곱함

$$
(cA)_{ij} = c \cdot a_{ij}
$$

**행렬 곱셈**: $A$가 $m \times n$, $B$가 $n \times p$일 때, 결과는 $m \times p$

$$
(AB)_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}
$$

> **주의**: 행렬 곱셈은 일반적으로 교환 법칙이 성립하지 않습니다: $AB \neq BA$

```python
import numpy as np

# 행렬 기본 연산
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A =\n", A)
print("B =\n", B)

# 행렬 곱셈
print("\nAB =\n", A @ B)
print("BA =\n", B @ A)
print("AB ≠ BA:", not np.allclose(A @ B, B @ A))

# 행렬 곱의 성질
C = np.array([[1, 0], [2, 3]])
print("\n(AB)C =\n", (A @ B) @ C)
print("A(BC) =\n", A @ (B @ C))
print("결합법칙 성립:", np.allclose((A @ B) @ C, A @ (B @ C)))
```

### 1.2 전치 행렬과 켤레 전치

**전치 행렬**(transpose) $A^T$: 행과 열을 교환

$$
(A^T)_{ij} = a_{ji}
$$

**성질**:
- $(AB)^T = B^T A^T$ (순서 반전)
- $(A^T)^T = A$
- $(A + B)^T = A^T + B^T$

**켤레 전치**(conjugate transpose, adjoint) $A^\dagger$: 전치 + 복소 켤레

$$
(A^\dagger)_{ij} = \overline{a_{ji}}
$$

```python
# 전치 행렬
A = np.array([[1, 2, 3], [4, 5, 6]])
print("A =\n", A)
print("A^T =\n", A.T)

# 복소 행렬의 켤레 전치
Z = np.array([[1+2j, 3-1j], [4j, 2+3j]])
print("\nZ =\n", Z)
print("Z† =\n", Z.conj().T)

# (AB)^T = B^T A^T 확인
B = np.array([[1, 2], [3, 4], [5, 6]])
print("\n(AB)^T =\n", (A @ B).T)
print("B^T A^T =\n", B.T @ A.T)
```

### 1.3 특수 행렬

**단위 행렬**(identity matrix) $I$: 대각 원소가 모두 1

$$
I_{ij} = \delta_{ij} = \begin{cases} 1 & (i = j) \\ 0 & (i \neq j) \end{cases}
$$

**대각 행렬**(diagonal matrix): 대각 원소만 0이 아닌 행렬

**대칭 행렬**(symmetric matrix): $A^T = A$, 즉 $a_{ij} = a_{ji}$

**반대칭 행렬**(antisymmetric): $A^T = -A$

**에르미트 행렬**(Hermitian matrix): $A^\dagger = A$ (대칭 행렬의 복소 확장)

**직교 행렬**(orthogonal matrix): $A^T A = AA^T = I$, 즉 $A^{-1} = A^T$

**유니터리 행렬**(unitary matrix): $A^\dagger A = AA^\dagger = I$ (직교 행렬의 복소 확장)

```python
# 직교 행렬 예: 2D 회전 행렬
theta = np.pi / 4  # 45도 회전
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print("R =\n", R)
print("R^T R =\n", np.round(R.T @ R, 10))
print("det(R) =", np.linalg.det(R))  # det = +1 (proper rotation)

# 에르미트 행렬 예
H = np.array([[2, 1-1j], [1+1j, 3]])
print("\nH =\n", H)
print("H† =\n", H.conj().T)
print("에르미트:", np.allclose(H, H.conj().T))
```

---

## 2. 행렬식 (Determinants)

### 2.1 정의와 기본 성질

$n \times n$ 정방행렬 $A$의 **행렬식**(determinant) $\det(A)$는 스칼라값입니다.

**2×2 행렬식**:

$$
\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc
$$

**3×3 행렬식** (사뤼스 법칙 또는 여인수 전개):

$$
\det\begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}
= a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})
$$

**주요 성질**:

1. $\det(AB) = \det(A) \cdot \det(B)$
2. $\det(A^T) = \det(A)$
3. $\det(cA) = c^n \det(A)$ ($n \times n$ 행렬)
4. 행(또는 열)을 교환하면 부호가 바뀜
5. 한 행이 다른 행의 상수배이면 $\det = 0$
6. $\det(A^{-1}) = 1/\det(A)$

### 2.2 여인수 전개 (Cofactor Expansion)

**소행렬식**(minor) $M_{ij}$: $i$번째 행과 $j$번째 열을 제거한 $(n-1) \times (n-1)$ 행렬의 행렬식

**여인수**(cofactor): $C_{ij} = (-1)^{i+j} M_{ij}$

**행렬식의 여인수 전개** (제$i$행으로 전개):

$$
\det(A) = \sum_{j=1}^{n} a_{ij} C_{ij} = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} M_{ij}
$$

```python
import numpy as np
from numpy.linalg import det

# 행렬식 계산
A = np.array([[2, 1, 3],
              [0, -1, 2],
              [4, 3, 1]])

print("det(A) =", det(A))

# 여인수 전개를 수동으로 계산 (1행 기준)
# det = 2*(-1-6) - 1*(0-8) + 3*(0+4)
manual = 2*(-1*1 - 2*3) - 1*(0*1 - 2*4) + 3*(0*3 - (-1)*4)
print("수동 계산:", manual)

# 행렬식의 성질 확인
B = np.array([[1, 2, 0],
              [3, 1, -1],
              [2, 0, 4]])

print(f"\ndet(A) = {det(A):.4f}")
print(f"det(B) = {det(B):.4f}")
print(f"det(AB) = {det(A @ B):.4f}")
print(f"det(A)*det(B) = {det(A)*det(B):.4f}")
```

### 2.3 행렬식의 기하학적 의미

$n \times n$ 행렬 $A$의 행렬식의 절댓값 $|\det(A)|$는 행벡터(또는 열벡터)로 이루어진 **평행다면체의 부피**입니다.

- 2D: $|\det(A)|$ = 두 벡터가 이루는 평행사변형의 넓이
- 3D: $|\det(A)|$ = 세 벡터가 이루는 평행육면체의 부피

$\det(A)$의 **부호**는 방향(orientation)을 나타냅니다:
- $\det > 0$: 오른손 좌표계 보존
- $\det < 0$: 좌표계 반전 (거울 반사 포함)

```python
import matplotlib.pyplot as plt

# 2D에서의 행렬식 = 넓이
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 변환 전: 단위 정사각형
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T

A = np.array([[2, 1], [0, 3]])  # det = 6

# 변환 후
transformed = A @ square

for ax, shape, title in [(axes[0], square, '단위 정사각형 (넓이=1)'),
                          (axes[1], transformed, f'변환 후 (넓이=|det|={abs(det(A)):.0f})')]:
    ax.fill(shape[0], shape[1], alpha=0.3, color='blue')
    ax.plot(shape[0], shape[1], 'b-', linewidth=2)
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('determinant_area.png', dpi=100, bbox_inches='tight')
plt.close()
print(f"det(A) = {det(A):.0f}: 넓이가 {abs(det(A)):.0f}배 확대")
```

---

## 3. 역행렬과 연립방정식

### 3.1 역행렬 (Inverse Matrix)

정방행렬 $A$의 **역행렬** $A^{-1}$는 다음을 만족합니다:

$$
AA^{-1} = A^{-1}A = I
$$

역행렬이 존재하기 위한 **필요충분조건**: $\det(A) \neq 0$ (비특이행렬, non-singular)

**여인수 행렬을 이용한 역행렬**:

$$
A^{-1} = \frac{1}{\det(A)} \text{adj}(A)
$$

여기서 $\text{adj}(A)$는 **수반 행렬**(adjugate matrix)로, 여인수 행렬의 전치입니다: $\text{adj}(A)_{ij} = C_{ji}$

**2×2 역행렬**:

$$
\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1}
= \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$

```python
A = np.array([[2, 1], [5, 3]])
A_inv = np.linalg.inv(A)

print("A =\n", A)
print("A^{-1} =\n", A_inv)
print("A @ A^{-1} =\n", np.round(A @ A_inv, 10))

# 수반 행렬을 이용한 수동 계산
d = det(A)  # 2*3 - 1*5 = 1
adj_A = np.array([[3, -1], [-5, 2]])  # 여인수 전치
manual_inv = adj_A / d
print("\n수동 계산:\n", manual_inv)
```

### 3.2 가우스 소거법 (Gaussian Elimination)

연립일차방정식 $A\mathbf{x} = \mathbf{b}$를 **확대 행렬** $(A | \mathbf{b})$에 대해 행 연산을 수행하여 풉니다.

**허용되는 기본 행 연산**:
1. 두 행의 교환
2. 한 행에 0이 아닌 상수를 곱함
3. 한 행의 상수배를 다른 행에 더함

**목표**: 상삼각 행렬(또는 기약 행 사다리꼴)로 변환 후 후진 대입

```python
def gauss_eliminate(A_aug, verbose=True):
    """가우스 소거법으로 연립방정식을 풉니다.

    A_aug: 확대 행렬 [A | b]
    """
    A = A_aug.astype(float).copy()
    n = A.shape[0]

    # 전진 소거 (forward elimination)
    for col in range(n):
        # 피봇 선택 (부분 피봇팅)
        max_row = col + np.argmax(np.abs(A[col:, col]))
        if max_row != col:
            A[[col, max_row]] = A[[max_row, col]]

        if abs(A[col, col]) < 1e-12:
            print(f"경고: 피봇이 0에 가깝습니다 (열 {col})")
            continue

        # 아래 행들 소거
        for row in range(col + 1, n):
            factor = A[row, col] / A[col, col]
            A[row] -= factor * A[col]

    if verbose:
        print("상삼각 행렬:\n", np.round(A, 4))

    # 후진 대입 (back substitution)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (A[i, -1] - A[i, i+1:n] @ x[i+1:n]) / A[i, i]

    return x

# 예제: 3x3 연립방정식
# 2x + y - z = 8
# -3x - y + 2z = -11
# -2x + y + 2z = -3
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

A_aug = np.column_stack([A, b])
x = gauss_eliminate(A_aug)
print(f"\n해: x = {x}")
print(f"검증 Ax = {A @ x}")

# NumPy 내장 함수와 비교
x_np = np.linalg.solve(A, b)
print(f"NumPy solve: {x_np}")
```

### 3.3 크래머 법칙 (Cramer's Rule)

연립방정식 $A\mathbf{x} = \mathbf{b}$에서, $\det(A) \neq 0$이면 각 미지수는:

$$
x_i = \frac{\det(A_i)}{\det(A)}
$$

여기서 $A_i$는 $A$의 $i$번째 열을 $\mathbf{b}$로 대체한 행렬입니다.

> **참고**: 크래머 법칙은 이론적으로 중요하지만, 계산 효율성은 가우스 소거법보다 낮습니다. $n$이 크면 가우스 소거법($O(n^3)$)이 크래머 법칙($O(n \cdot n!)$)보다 훨씬 빠릅니다.

```python
def cramer(A, b):
    """크래머 법칙으로 Ax = b를 풉니다."""
    n = len(b)
    d = det(A)
    if abs(d) < 1e-12:
        raise ValueError("행렬식이 0: 유일한 해가 존재하지 않음")

    x = np.zeros(n)
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        x[i] = det(A_i) / d
    return x

# 예제
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

x = cramer(A, b)
print(f"크래머 법칙 해: {x}")
```

### 3.4 행렬의 계수 (Rank)

행렬의 **계수**(rank)는 선형 독립인 행(또는 열)의 최대 개수입니다.

$$
\text{rank}(A) = \text{dim}(\text{Col}(A)) = \text{dim}(\text{Row}(A))
$$

**연립방정식의 해의 존재성**:
- $\text{rank}(A) = \text{rank}(A|b) = n$: 유일한 해
- $\text{rank}(A) = \text{rank}(A|b) < n$: 무한히 많은 해
- $\text{rank}(A) < \text{rank}(A|b)$: 해 없음 (불능)

```python
# 계수 예제
A_full = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])  # rank 3
A_deficient = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # rank 2 (3행 = 행1 + 2*행2 아님, 하지만 1+3=2*2)

print(f"rank(A_full) = {np.linalg.matrix_rank(A_full)}")
print(f"rank(A_deficient) = {np.linalg.matrix_rank(A_deficient)}")
print(f"det(A_full) = {det(A_full):.4f}")
print(f"det(A_deficient) = {det(A_deficient):.4f}")
```

---

## 4. 고유값과 고유벡터 (Eigenvalues and Eigenvectors)

### 4.1 정의와 특성 방정식

정방행렬 $A$에 대해, 다음을 만족하는 0이 아닌 벡터 $\mathbf{v}$와 스칼라 $\lambda$를 찾는 문제:

$$
A\mathbf{v} = \lambda\mathbf{v}
$$

- $\lambda$: **고유값**(eigenvalue)
- $\mathbf{v}$: **고유벡터**(eigenvector)

기하학적 의미: $A$에 의한 선형 변환에서, 고유벡터는 **방향이 변하지 않는** 특별한 벡터입니다. 고유값은 그 방향으로의 늘림/줄임 비율입니다.

**특성 방정식**(characteristic equation):

$$
\det(A - \lambda I) = 0
$$

이것은 $\lambda$에 대한 $n$차 다항식으로, **특성 다항식**(characteristic polynomial)이라 합니다.

```python
import numpy as np
from numpy.linalg import eig

# 2x2 행렬의 고유값/고유벡터
A = np.array([[4, 2], [1, 3]])

eigenvalues, eigenvectors = eig(A)
print("고유값:", eigenvalues)
print("고유벡터 (열벡터):\n", eigenvectors)

# 검증: Av = λv
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    Av = A @ v
    lam_v = lam * v
    print(f"\nλ_{i+1} = {lam:.4f}")
    print(f"v_{i+1} = {v}")
    print(f"Av = {Av}")
    print(f"λv = {lam_v}")
    print(f"Av ≈ λv: {np.allclose(Av, lam_v)}")
```

### 4.2 수동 계산 예제

$A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$의 고유값/고유벡터:

**특성 방정식**:
$$
\det(A - \lambda I) = \det\begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix} = (3-\lambda)(2-\lambda) = 0
$$

$\lambda_1 = 3$, $\lambda_2 = 2$

**고유벡터** ($\lambda_1 = 3$):
$$
(A - 3I)\mathbf{v} = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}\mathbf{v} = 0 \implies v_2 = 0 \implies \mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}
$$

**고유벡터** ($\lambda_2 = 2$):
$$
(A - 2I)\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\mathbf{v} = 0 \implies v_1 + v_2 = 0 \implies \mathbf{v}_2 = \begin{pmatrix} -1 \\ 1 \end{pmatrix}
$$

```python
import sympy as sp

# SymPy를 이용한 해석적 계산
A_sym = sp.Matrix([[3, 1], [0, 2]])
print("특성 다항식:", A_sym.charpoly().as_expr())
print("고유값:", A_sym.eigenvals())  # {3: 1, 2: 1} (고유값: 중복도)
print("고유벡터:", A_sym.eigenvects())
```

### 4.3 행렬의 대각화 (Diagonalization)

$n \times n$ 행렬 $A$가 $n$개의 선형 독립인 고유벡터를 가지면 **대각화 가능**(diagonalizable)합니다.

고유벡터를 열로 나열한 행렬 $P = [\mathbf{v}_1 | \mathbf{v}_2 | \cdots | \mathbf{v}_n]$을 구성하면:

$$
P^{-1}AP = D = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)
$$

**대각화의 유용성**:

- **거듭제곱**: $A^k = P D^k P^{-1}$이므로 $D^k$는 대각 원소를 $k$제곱하면 됨
- **지수함수**: $e^{At} = P \, e^{Dt} \, P^{-1}$, 여기서 $e^{Dt} = \text{diag}(e^{\lambda_1 t}, \ldots, e^{\lambda_n t})$
- **연립 ODE**: $\dot{\mathbf{x}} = A\mathbf{x}$를 독립적인 스칼라 ODE로 분리

```python
# 대각화 예제
A = np.array([[4, 2], [1, 3]])
eigenvalues, P = eig(A)
D = np.diag(eigenvalues)

print("P (고유벡터 행렬):\n", P)
print("D (대각 행렬):\n", D)

# 검증: A = P D P^{-1}
P_inv = np.linalg.inv(P)
A_reconstructed = P @ D @ P_inv
print("\nP D P^{-1} =\n", np.round(A_reconstructed, 10))
print("A와 일치:", np.allclose(A, A_reconstructed))

# 응용: A^10 계산
A_power_10 = P @ np.diag(eigenvalues**10) @ P_inv
print(f"\nA^10 =\n{np.round(A_power_10.real, 2)}")
print(f"직접 계산:\n{np.round(np.linalg.matrix_power(A, 10).astype(float), 2)}")
```

### 4.4 대칭 행렬과 에르미트 행렬의 성질

**스펙트럼 정리**(Spectral Theorem):

1. **실대칭 행렬** ($A^T = A$):
   - 모든 고유값이 **실수**
   - 서로 다른 고유값에 대응하는 고유벡터는 **직교**
   - 직교 행렬 $Q$로 대각화 가능: $Q^T A Q = D$, $Q^T = Q^{-1}$

2. **에르미트 행렬** ($A^\dagger = A$):
   - 모든 고유값이 **실수**
   - 서로 다른 고유값에 대응하는 고유벡터는 **직교**
   - 유니터리 행렬 $U$로 대각화 가능: $U^\dagger A U = D$, $U^\dagger = U^{-1}$

```python
# 실대칭 행렬의 성질
A_sym = np.array([[4, 1, 2],
                   [1, 3, 1],
                   [2, 1, 5]])

eigenvalues, eigenvectors = eig(A_sym)
print("대칭 행렬의 고유값:", eigenvalues)
print("모두 실수:", np.allclose(eigenvalues.imag, 0))

# 고유벡터의 직교성 확인
print("\n고유벡터 내적:")
for i in range(3):
    for j in range(i+1, 3):
        dot = eigenvectors[:, i] @ eigenvectors[:, j]
        print(f"  v{i+1} · v{j+1} = {dot:.6e}")

# 직교 행렬로 대각화
Q = eigenvectors
print(f"\nQ^T Q =\n{np.round(Q.T @ Q, 10)}")
print("Q는 직교 행렬:", np.allclose(Q.T @ Q, np.eye(3)))
```

### 4.5 주요 행렬 분해법 요약

| 분해법 | 형태 | 조건 | 물리적 응용 |
|--------|------|------|------------|
| 고유값 분해 | $A = PDP^{-1}$ | $n$개 독립 고유벡터 | 진동 모드, 안정성 |
| 직교 대각화 | $A = QDQ^T$ | 대칭 행렬 | 주축 정리, 관성 텐서 |
| SVD | $A = U\Sigma V^T$ | 임의의 $m \times n$ | 데이터 축소, 최소자승 |
| LU 분해 | $A = LU$ | 정방행렬 | 연립방정식 효율 풀이 |
| 촐레스키 | $A = LL^T$ | 양정치 대칭 | 통계, 최적화 |

---

## 5. 이차형식과 양정치성 (Quadratic Forms)

### 5.1 이차형식의 정의

벡터 $\mathbf{x} \in \mathbb{R}^n$에 대한 **이차형식**(quadratic form):

$$
Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i,j} a_{ij} x_i x_j
$$

여기서 $A$는 항상 대칭 행렬로 취급할 수 있습니다 (비대칭 부분은 이차형식에 기여하지 않으므로 $A \to (A + A^T)/2$).

**2변수 예**:

$$
Q(x, y) = \begin{pmatrix} x & y \end{pmatrix} \begin{pmatrix} a & b \\ b & c \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = ax^2 + 2bxy + cy^2
$$

### 5.2 양정치성 판별

대칭 행렬 $A$의 **정치성**(definiteness):

| 분류 | 조건 | 고유값 조건 |
|------|------|------------|
| 양정치 (positive definite) | $\mathbf{x}^T A \mathbf{x} > 0$ (모든 $\mathbf{x} \neq 0$) | 모든 $\lambda_i > 0$ |
| 양반정치 (positive semidefinite) | $\mathbf{x}^T A \mathbf{x} \geq 0$ | 모든 $\lambda_i \geq 0$ |
| 음정치 (negative definite) | $\mathbf{x}^T A \mathbf{x} < 0$ (모든 $\mathbf{x} \neq 0$) | 모든 $\lambda_i < 0$ |
| 부정치 (indefinite) | 양수와 음수 모두 가능 | 양수/음수 고유값 혼재 |

**실벡스터 기준**(Sylvester's criterion): 대칭 행렬이 양정치일 필요충분조건은 모든 **선행 주 소행렬식**(leading principal minors)이 양수:

$$
a_{11} > 0, \quad \det\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix} > 0, \quad \det\begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix} > 0, \quad \ldots
$$

```python
def check_definiteness(A):
    """대칭 행렬의 정치성을 판별합니다."""
    eigenvalues = np.linalg.eigvalsh(A)  # 대칭 행렬 전용
    print(f"고유값: {eigenvalues}")

    if all(eigenvalues > 0):
        return "양정치 (positive definite)"
    elif all(eigenvalues >= 0):
        return "양반정치 (positive semidefinite)"
    elif all(eigenvalues < 0):
        return "음정치 (negative definite)"
    elif all(eigenvalues <= 0):
        return "음반정치 (negative semidefinite)"
    else:
        return "부정치 (indefinite)"

# 예제
matrices = {
    "양정치": np.array([[2, -1], [-1, 2]]),
    "부정치": np.array([[1, 3], [3, 1]]),
    "양반정치": np.array([[1, 1], [1, 1]]),
}

for name, M in matrices.items():
    result = check_definiteness(M)
    print(f"{name} 행렬: {result}\n")
```

### 5.3 주축 정리 (Principal Axis Theorem)

이차형식을 **주축 좌표**로 변환하면 교차항이 사라집니다.

대칭 행렬 $A = Q D Q^T$이면:

$$
Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \mathbf{y}^T D \mathbf{y} = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \cdots + \lambda_n y_n^2
$$

여기서 $\mathbf{y} = Q^T \mathbf{x}$ (주축 좌표).

> 이것은 물리학에서 **관성 텐서의 주축 변환**, **응력 텐서의 주응력 방향** 등에 직접 응용됩니다.

```python
import matplotlib.pyplot as plt

# 이차곡면과 주축 변환 시각화
A = np.array([[5, 2], [2, 2]])
eigenvalues, Q = np.linalg.eigh(A)

print(f"원래 이차형식: 5x² + 4xy + 2y²")
print(f"고유값: λ₁={eigenvalues[0]:.2f}, λ₂={eigenvalues[1]:.2f}")
print(f"주축 좌표: {eigenvalues[0]:.2f}y₁² + {eigenvalues[1]:.2f}y₂²")

# 타원 시각화
theta = np.linspace(0, 2*np.pi, 200)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 원래 좌표계에서의 타원 (5x² + 4xy + 2y² = 1)
# 매개변수 표현 이용
t = np.linspace(0, 2*np.pi, 200)
# 주축 좌표에서 타원: λ₁y₁² + λ₂y₂² = 1
y1 = np.cos(t) / np.sqrt(eigenvalues[0])
y2 = np.sin(t) / np.sqrt(eigenvalues[1])
# 원래 좌표로 변환: x = Q y
xy = Q @ np.array([y1, y2])

axes[0].plot(xy[0], xy[1], 'b-', linewidth=2)
# 주축 방향
for i in range(2):
    v = Q[:, i] / np.sqrt(eigenvalues[i])
    axes[0].arrow(0, 0, v[0], v[1], head_width=0.03, color=['red', 'green'][i],
                  linewidth=2, label=f'주축 {i+1} (λ={eigenvalues[i]:.2f})')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].set_title('원래 좌표계: 5x² + 4xy + 2y² = 1')

# 주축 좌표에서의 타원 (축 정렬)
y1 = np.cos(t) / np.sqrt(eigenvalues[0])
y2 = np.sin(t) / np.sqrt(eigenvalues[1])
axes[1].plot(y1, y2, 'b-', linewidth=2)
axes[1].axhline(y=0, color='green', linewidth=1.5, label=f'y₂축 (λ={eigenvalues[1]:.2f})')
axes[1].axvline(x=0, color='red', linewidth=1.5, label=f'y₁축 (λ={eigenvalues[0]:.2f})')
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].set_title(f'주축 좌표: {eigenvalues[0]:.2f}y₁² + {eigenvalues[1]:.2f}y₂² = 1')

plt.tight_layout()
plt.savefig('principal_axes.png', dpi=100, bbox_inches='tight')
plt.close()
```

---

## 6. 물리학 응용

### 6.1 관성 텐서와 주축 (Moment of Inertia Tensor)

강체의 **관성 텐서**(moment of inertia tensor)는 $3 \times 3$ 대칭 행렬입니다:

$$
I = \begin{pmatrix} I_{xx} & -I_{xy} & -I_{xz} \\ -I_{yx} & I_{yy} & -I_{yz} \\ -I_{zx} & -I_{zy} & I_{zz} \end{pmatrix}
$$

여기서:
- $I_{xx} = \sum m_i (y_i^2 + z_i^2)$ (관성 모멘트)
- $I_{xy} = \sum m_i x_i y_i$ (관성곱)

**주축 변환**: $I$를 대각화하면 **주관성 모멘트** $I_1, I_2, I_3$와 **주축**(principal axes)을 얻습니다.

$$
Q^T I Q = \text{diag}(I_1, I_2, I_3)
$$

주축 좌표계에서 각운동량은 $L_i = I_i \omega_i$로 단순화됩니다.

```python
# 관성 텐서 예제: 정육면체 꼭짓점에 질량이 있는 경우
# 정육면체 꼭짓점 좌표 (변의 길이 2, 중심 원점)
vertices = np.array([
    [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
    [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1]
], dtype=float)

m = 1.0  # 각 꼭짓점의 질량

# 관성 텐서 계산
I_tensor = np.zeros((3, 3))
for r in vertices:
    I_tensor += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

print("관성 텐서:\n", I_tensor)

# 대각화 → 주관성 모멘트
eigenvalues, eigenvectors = np.linalg.eigh(I_tensor)
print(f"\n주관성 모멘트: I₁={eigenvalues[0]:.2f}, I₂={eigenvalues[1]:.2f}, I₃={eigenvalues[2]:.2f}")
print("주축 방향:\n", eigenvectors)
```

### 6.2 연성 진동 (Coupled Oscillations)

두 개의 질량이 스프링으로 연결된 시스템:

$$
M\ddot{\mathbf{x}} = -K\mathbf{x}
$$

여기서 $M$은 질량 행렬, $K$는 강성 행렬(stiffness matrix).

**고유진동수**(normal frequencies)와 **고유모드**(normal modes)는 일반화 고유값 문제입니다:

$$
K\mathbf{v} = \omega^2 M\mathbf{v}
$$

```python
# 연성 진동 예제: 두 질량-스프링 계
# m₁ = m₂ = m, 스프링 상수 k₁ = k₂ = k₃ = k
# 운동방정식: m*x₁'' = -k*x₁ + k*(x₂-x₁) = -2k*x₁ + k*x₂
#             m*x₂'' = -k*(x₂-x₁) - k*x₂ = k*x₁ - 2k*x₂

m, k = 1.0, 1.0

M_mat = m * np.eye(2)
K_mat = np.array([[2*k, -k],
                   [-k, 2*k]])

# 일반화 고유값 문제: K v = ω² M v
# M = mI이므로 (1/m)K v = ω² v
eigenvalues, eigenvectors = np.linalg.eigh(K_mat / m)
omega = np.sqrt(eigenvalues)

print("강성 행렬 K:\n", K_mat)
print(f"\n고유진동수: ω₁ = {omega[0]:.4f}, ω₂ = {omega[1]:.4f}")
print(f"주기: T₁ = {2*np.pi/omega[0]:.4f}, T₂ = {2*np.pi/omega[1]:.4f}")
print(f"\n고유모드 1 (동상): {eigenvectors[:, 0]}")
print(f"고유모드 2 (역상): {eigenvectors[:, 1]}")

# 시간 발전 시각화
t = np.linspace(0, 20, 500)
# 초기 조건: x₁(0) = 1, x₂(0) = 0, 속도 0
# 모드 좌표로 분해
eta = eigenvectors.T @ np.array([1, 0])
x = np.zeros((2, len(t)))
for i in range(2):
    x += np.outer(eigenvectors[:, i], eta[i] * np.cos(omega[i] * t))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, x[0], label='x₁(t)', linewidth=1.5)
ax.plot(t, x[1], label='x₂(t)', linewidth=1.5)
ax.set_xlabel('시간 t')
ax.set_ylabel('변위')
ax.set_title('연성 진동: 맥놀이(beat) 현상')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('coupled_oscillations.png', dpi=100, bbox_inches='tight')
plt.close()
```

### 6.3 양자역학에서의 행렬 (Matrices in Quantum Mechanics)

양자역학에서 관측 가능량(observable)은 **에르미트 연산자**로 표현되며, 유한 차원에서는 에르미트 행렬입니다.

**스핀-1/2 입자의 파울리 행렬**:

$$
\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

이들은 에르미트($\sigma_i^\dagger = \sigma_i$)이고, 각각의 고유값은 $\pm 1$입니다.

```python
# 파울리 행렬
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

for name, sigma in [('σ_x', sigma_x), ('σ_y', sigma_y), ('σ_z', sigma_z)]:
    vals, vecs = np.linalg.eigh(sigma)
    print(f"{name}:")
    print(f"  에르미트: {np.allclose(sigma, sigma.conj().T)}")
    print(f"  고유값: {vals}")
    print(f"  tr(σ) = {np.trace(sigma):.0f}, det(σ) = {np.linalg.det(sigma):.0f}")
    print()

# 교환 관계: [σ_i, σ_j] = 2i ε_ijk σ_k
comm_xy = sigma_x @ sigma_y - sigma_y @ sigma_x
print("[σ_x, σ_y] = 2i σ_z:")
print(np.round(comm_xy, 10))
print("2i σ_z:")
print(2j * sigma_z)
print("일치:", np.allclose(comm_xy, 2j * sigma_z))
```

---

## 7. Python으로 하는 선형대수

### 7.1 NumPy와 SciPy 활용

```python
import numpy as np
from scipy import linalg

# ===== 기본 연산 =====
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])

# 행렬식, 역행렬
print(f"det(A) = {np.linalg.det(A):.4f}")
print(f"A^(-1) =\n{np.linalg.inv(A)}")

# 고유값 분해
vals, vecs = np.linalg.eig(A)
print(f"\n고유값: {vals}")

# ===== 다양한 행렬 분해 =====

# LU 분해
P, L, U = linalg.lu(A)
print(f"\nLU 분해:")
print(f"P =\n{P}")
print(f"L =\n{np.round(L, 4)}")
print(f"U =\n{np.round(U, 4)}")
print(f"PLU = A: {np.allclose(P @ L @ U, A)}")

# SVD (Singular Value Decomposition)
U_svd, s, Vt = np.linalg.svd(A)
print(f"\n특이값: {s}")

# QR 분해
Q, R = np.linalg.qr(A)
print(f"\nQR 분해: Q 직교성 {np.allclose(Q.T @ Q, np.eye(3))}")

# 연립방정식 풀기
b = np.array([1, 2, 3])
x = np.linalg.solve(A, b)
print(f"\nAx = b의 해: {x}")
print(f"검증: Ax = {A @ x}")
```

### 7.2 SymPy를 이용한 해석적 계산

```python
import sympy as sp

# 기호 행렬
a, b, c, d = sp.symbols('a b c d')
M = sp.Matrix([[a, b], [c, d]])

print("일반 2x2 행렬:")
print(f"  det = {M.det()}")
print(f"  trace = {M.trace()}")
print(f"  특성 다항식: {M.charpoly().as_expr()}")

# 케일리-해밀턴 정리: 모든 행렬은 자신의 특성 다항식을 만족
# A² - (a+d)A + (ad-bc)I = 0
lam = sp.Symbol('lambda')
char_poly = M.charpoly(lam)
print(f"\n특성 다항식 p(λ) = {char_poly.as_expr()}")
print("케일리-해밀턴: p(M) = 0 확인")
result = M**2 - (a+d)*M + (a*d - b*c)*sp.eye(2)
print(f"  p(M) = {sp.simplify(result)}")
```

---

## 연습문제

### 기본 문제

**1.** 다음 행렬의 행렬식과 역행렬을 구하시오:
$$
A = \begin{pmatrix} 1 & 2 & 1 \\ 3 & 1 & -1 \\ 2 & 0 & 3 \end{pmatrix}
$$

**2.** 다음 연립방정식을 가우스 소거법으로 푸시오:
$$
x + 2y + z = 4, \quad 3x + y - z = 2, \quad 2x + 3z = 7
$$

**3.** 다음 행렬의 고유값과 고유벡터를 구하시오:
$$
B = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}
$$

### 응용 문제

**4.** 이차형식 $Q(x,y) = 3x^2 + 2xy + 3y^2$의 양정치/음정치/부정치 여부를 판별하고, 주축 변환하시오.

**5.** 질량 $m_1 = 1$, $m_2 = 2$인 두 물체가 스프링 상수 $k_1 = 3$, $k_2 = 4$, $k_3 = 2$인 세 스프링으로 연결되어 있습니다 (벽–k₁–m₁–k₂–m₂–k₃–벽). 고유진동수와 고유모드를 구하시오.

**6.** 자기장 방향 $\hat{n} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$에 대한 스핀-1/2 측정 연산자 $\hat{S}_n = \frac{\hbar}{2}(\sigma_x \sin\theta\cos\phi + \sigma_y \sin\theta\sin\phi + \sigma_z \cos\theta)$의 고유값과 고유벡터를 구하시오.

---

## 참고 자료

- Boas, *Mathematical Methods in the Physical Sciences*, Chapter 3
- Strang, *Introduction to Linear Algebra*
- Arfken & Weber, *Mathematical Methods for Physicists*, Chapters 3-4
- 이 레슨의 코드: `examples/Math_for_AI/01_vector_matrix_ops.py`, `02_svd_pca.py` 참조

---

## 다음 레슨

[04. 편미분 (Partial Differentiation)](./04_Partial_Differentiation.md)에서 다변수 함수의 미적분을 다룹니다.
