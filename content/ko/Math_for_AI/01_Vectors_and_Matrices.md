# 01. 벡터와 행렬 (Vectors and Matrices)

## 학습 목표

- 벡터의 기하학적 의미와 대수적 연산을 이해하고 Python으로 구현할 수 있다
- 벡터 공간, 기저, 차원의 개념을 이해하고 선형 독립성을 판별할 수 있다
- 행렬을 선형 변환으로 해석하고 기하학적 의미를 시각화할 수 있다
- 행렬식, 역행렬, 랭크의 개념을 이해하고 연립방정식의 해를 구할 수 있다
- 머신러닝에서 벡터와 행렬이 어떻게 사용되는지 구체적인 예를 들 수 있다

---

## 1. 벡터의 정의와 연산

### 1.1 벡터란 무엇인가?

벡터는 크기와 방향을 가진 양입니다. 벡터는 두 가지 관점에서 이해할 수 있습니다:

1. **기하학적 관점**: 화살표. 시작점에서 끝점으로의 변위
2. **대수적 관점**: 순서가 있는 숫자들의 리스트

n차원 벡터 $\mathbf{v}$는 다음과 같이 표현됩니다:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

```python
import numpy as np
import matplotlib.pyplot as plt

# 2D 벡터 생성
v = np.array([3, 2])
w = np.array([1, 3])

# 벡터 시각화
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', width=0.006, label='v')
ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='b', width=0.006, label='w')
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title('2D Vectors')
plt.show()
```

### 1.2 벡터 연산

**벡터 덧셈**: 두 벡터를 더하면 평행사변형 법칙을 따릅니다.

$$\mathbf{v} + \mathbf{w} = \begin{bmatrix} v_1 + w_1 \\ v_2 + w_2 \\ \vdots \\ v_n + w_n \end{bmatrix}$$

**스칼라 곱**: 스칼라 $c$를 벡터에 곱하면 방향은 유지되고 크기만 변합니다.

$$c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$$

```python
# 벡터 덧셈과 스칼라 곱
v_plus_w = v + w
c = 2
cv = c * v

# 시각화
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', width=0.006, label='v')
ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='b', width=0.006, label='w')
ax.quiver(0, 0, v_plus_w[0], v_plus_w[1], angles='xy', scale_units='xy', scale=1,
          color='purple', width=0.006, label='v+w')
ax.quiver(v[0], v[1], w[0], w[1], angles='xy', scale_units='xy', scale=1,
          color='gray', width=0.004, linestyle='--', alpha=0.5)
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title('Vector Addition (Parallelogram Law)')
plt.show()
```

### 1.3 내적 (Dot Product)

두 벡터의 내적은 스칼라 값을 반환합니다:

$$\mathbf{v} \cdot \mathbf{w} = \sum_{i=1}^n v_i w_i = v_1w_1 + v_2w_2 + \cdots + v_nw_n$$

기하학적 해석:

$$\mathbf{v} \cdot \mathbf{w} = \|\mathbf{v}\| \|\mathbf{w}\| \cos\theta$$

여기서 $\theta$는 두 벡터 사이의 각도입니다.

**내적의 의미**:
- $\mathbf{v} \cdot \mathbf{w} > 0$: 예각 (같은 방향)
- $\mathbf{v} \cdot \mathbf{w} = 0$: 직각 (직교)
- $\mathbf{v} \cdot \mathbf{w} < 0$: 둔각 (반대 방향)

```python
# 내적 계산
dot_product = np.dot(v, w)
print(f"v · w = {dot_product}")

# 각도 계산
norm_v = np.linalg.norm(v)
norm_w = np.linalg.norm(w)
cos_theta = dot_product / (norm_v * norm_w)
theta = np.arccos(cos_theta)
print(f"Angle between v and w: {np.degrees(theta):.2f} degrees")

# 직교 벡터 예제
u = np.array([1, 0])
v_orth = np.array([0, 1])
print(f"u · v_orth = {np.dot(u, v_orth)}")  # 0 (직교)
```

### 1.4 외적 (Cross Product) - 3D 전용

3차원 벡터에서만 정의되며, 결과는 두 벡터에 수직인 벡터입니다:

$$\mathbf{v} \times \mathbf{w} = \begin{bmatrix} v_2w_3 - v_3w_2 \\ v_3w_1 - v_1w_3 \\ v_1w_2 - v_2w_1 \end{bmatrix}$$

크기는 평행사변형의 넓이: $\|\mathbf{v} \times \mathbf{w}\| = \|\mathbf{v}\| \|\mathbf{w}\| \sin\theta$

```python
# 3D 외적
v3d = np.array([1, 2, 3])
w3d = np.array([4, 5, 6])
cross_product = np.cross(v3d, w3d)
print(f"v × w = {cross_product}")

# 외적이 두 벡터에 수직인지 확인
print(f"(v × w) · v = {np.dot(cross_product, v3d)}")  # ~0
print(f"(v × w) · w = {np.dot(cross_product, w3d)}")  # ~0
```

## 2. 벡터 공간 (Vector Spaces)

### 2.1 벡터 공간의 정의

벡터 공간 $V$는 다음 두 연산이 정의된 집합입니다:
1. **벡터 덧셈**: $\mathbf{u} + \mathbf{v} \in V$
2. **스칼라 곱**: $c\mathbf{v} \in V$

그리고 8개의 공리를 만족해야 합니다 (교환법칙, 결합법칙, 항등원, 역원, 분배법칙 등).

**예시**:
- $\mathbb{R}^n$: 모든 n차원 실수 벡터
- 다항식 공간: 차수가 $n$ 이하인 모든 다항식
- 함수 공간: 구간 $[a, b]$에서 연속인 모든 함수

### 2.2 부분공간 (Subspace)

벡터 공간 $V$의 부분집합 $W$가 부분공간이 되려면:
1. 영벡터 포함: $\mathbf{0} \in W$
2. 덧셈에 닫혀있음: $\mathbf{u}, \mathbf{v} \in W \Rightarrow \mathbf{u} + \mathbf{v} \in W$
3. 스칼라 곱에 닫혀있음: $\mathbf{v} \in W, c \in \mathbb{R} \Rightarrow c\mathbf{v} \in W$

**예시**: $\mathbb{R}^3$에서 원점을 지나는 평면이나 직선은 부분공간입니다.

```python
# 부분공간 예제: R^3에서 z=0인 평면 (xy-평면)
# 이 평면은 R^3의 부분공간

# 평면 위의 두 벡터
u = np.array([1, 2, 0])
v = np.array([3, -1, 0])

# 덧셈에 닫혀있는지 확인
print(f"u + v = {u + v}")  # [4, 1, 0] - 여전히 z=0 평면 위
print(f"2*u = {2*u}")      # [2, 4, 0] - 여전히 z=0 평면 위
```

### 2.3 생성 (Span)과 선형 결합

벡터들 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$의 선형 결합:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$$

생성 (span): 모든 가능한 선형 결합의 집합

$$\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \{c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k : c_i \in \mathbb{R}\}$$

```python
# Span 시각화: 두 벡터의 span은 평면을 형성
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

# 다양한 선형 결합 생성
combinations = []
for c1 in np.linspace(-2, 2, 20):
    for c2 in np.linspace(-2, 2, 20):
        combinations.append(c1 * v1 + c2 * v2)

combinations = np.array(combinations)

# 3D 플롯
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(combinations[:, 0], combinations[:, 1], combinations[:, 2],
           alpha=0.3, s=1)
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', arrow_length_ratio=0.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Span of two vectors in R^3')
plt.show()
```

### 2.4 선형 독립 (Linear Independence)

벡터들 $\mathbf{v}_1, \ldots, \mathbf{v}_k$가 선형 독립이려면:

$$c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \Rightarrow c_1 = \cdots = c_k = 0$$

즉, 자명하지 않은 선형 결합으로는 영벡터를 만들 수 없어야 합니다.

**판별법**: 벡터들을 열로 하는 행렬의 랭크가 벡터 개수와 같으면 선형 독립입니다.

```python
# 선형 독립성 검사
vectors_independent = np.array([[1, 2], [2, 3], [3, 4]]).T  # 각 열이 벡터
vectors_dependent = np.array([[1, 2], [2, 4], [3, 6]]).T

rank_indep = np.linalg.matrix_rank(vectors_independent)
rank_dep = np.linalg.matrix_rank(vectors_dependent)

print(f"Independent vectors rank: {rank_indep} (# vectors: 3)")
print(f"Dependent vectors rank: {rank_dep} (# vectors: 3)")
print(f"First set is linearly independent: {rank_indep == 3}")
print(f"Second set is linearly independent: {rank_dep == 3}")
```

### 2.5 기저 (Basis)와 차원 (Dimension)

**기저**: 벡터 공간 $V$를 생성하는 선형 독립인 벡터들의 집합

**차원**: 기저의 벡터 개수 (모든 기저는 같은 개수)

$\mathbb{R}^n$의 표준 기저:

$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix}, \quad \ldots, \quad \mathbf{e}_n = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$

```python
# 표준 기저
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

# 임의의 벡터를 기저로 표현
v = np.array([5, 3, -2])
print(f"v = {v[0]}*e1 + {v[1]}*e2 + {v[2]}*e3")
print(f"Verification: {v[0]*e1 + v[1]*e2 + v[2]*e3}")

# 다른 기저 사용
b1 = np.array([1, 1, 0])
b2 = np.array([0, 1, 1])
b3 = np.array([1, 0, 1])

# v를 새로운 기저로 표현하려면 연립방정식 풀기
B = np.column_stack([b1, b2, b3])
coords = np.linalg.solve(B, v)
print(f"v in new basis: {coords}")
print(f"Verification: {coords[0]*b1 + coords[1]*b2 + coords[2]*b3}")
```

## 3. 행렬의 기초

### 3.1 행렬 정의와 연산

$m \times n$ 행렬 $A$는 $m$개의 행과 $n$개의 열을 가진 숫자 배열입니다:

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

**행렬 덧셈**: 같은 크기의 행렬만 가능

$$C = A + B \quad \Rightarrow \quad c_{ij} = a_{ij} + b_{ij}$$

**행렬 곱셈**: $A$가 $m \times n$, $B$가 $n \times p$일 때, $C = AB$는 $m \times p$

$$(AB)_{ij} = \sum_{k=1}^n a_{ik}b_{kj}$$

```python
# 행렬 생성과 연산
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

print(f"A shape: {A.shape}")  # (2, 3)
print(f"B shape: {B.shape}")  # (3, 2)

# 행렬 곱셈
C = A @ B  # 또는 np.matmul(A, B) 또는 np.dot(A, B)
print(f"C = A @ B:\n{C}")
print(f"C shape: {C.shape}")  # (2, 2)

# 원소별 곱셈 (element-wise, Hadamard product)
D = np.array([[1, 2], [3, 4]])
E = np.array([[5, 6], [7, 8]])
F = D * E  # element-wise
print(f"D * E (element-wise):\n{F}")
```

### 3.2 전치 (Transpose), 역행렬 (Inverse), 행렬식 (Determinant)

**전치**: 행과 열을 바꾼 행렬

$$A^T_{ij} = A_{ji}$$

**역행렬**: $AA^{-1} = A^{-1}A = I$ (정방행렬이고 full rank일 때만 존재)

**행렬식**: 정방행렬에 대해 정의되는 스칼라 값
- $\det(A) \neq 0 \Leftrightarrow A$가 역행렬을 가짐

```python
# 전치
A = np.array([[1, 2, 3],
              [4, 5, 6]])
A_T = A.T
print(f"A^T:\n{A_T}")

# 역행렬
B = np.array([[1, 2],
              [3, 4]])
B_inv = np.linalg.inv(B)
print(f"B inverse:\n{B_inv}")
print(f"B @ B_inv:\n{B @ B_inv}")  # 단위 행렬

# 행렬식
det_B = np.linalg.det(B)
print(f"det(B) = {det_B}")

# 특이 행렬 (역행렬 없음)
C = np.array([[1, 2],
              [2, 4]])
print(f"det(C) = {np.linalg.det(C):.10f}")  # ~0
# C_inv = np.linalg.inv(C)  # LinAlgError 발생
```

### 3.3 특수 행렬들

**단위 행렬 (Identity matrix)**: $I_n$

$$I_n = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}$$

**대칭 행렬 (Symmetric)**: $A = A^T$

**직교 행렬 (Orthogonal)**: $Q^TQ = QQ^T = I$ (열들이 정규 직교 기저)

**대각 행렬 (Diagonal)**: 대각선 이외 모두 0

**삼각 행렬 (Triangular)**: 상삼각 또는 하삼각

```python
# 특수 행렬 생성
I = np.eye(3)  # 3x3 단위 행렬
print(f"Identity matrix:\n{I}")

# 대각 행렬
D = np.diag([1, 2, 3])
print(f"Diagonal matrix:\n{D}")

# 대칭 행렬
S = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])
print(f"Is symmetric? {np.allclose(S, S.T)}")

# 직교 행렬 (회전 행렬)
theta = np.pi / 4
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
print(f"Q^T @ Q:\n{Q.T @ Q}")  # 단위 행렬
```

## 4. 선형 변환 (Linear Transformations)

### 4.1 행렬 = 선형 변환

행렬 $A$는 벡터를 다른 벡터로 변환하는 함수로 볼 수 있습니다:

$$T(\mathbf{v}) = A\mathbf{v}$$

선형 변환의 두 가지 성질:
1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. $T(c\mathbf{v}) = cT(\mathbf{v})$

```python
# 선형 변환 예제
def linear_transform(A, v):
    return A @ v

# 변환 행렬
A = np.array([[2, 0],
              [0, 0.5]])

# 단위 정사각형
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

# 변환 후
transformed = A @ square

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(square[0], square[1], 'b-o')
ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.5, 2.5)
ax1.set_aspect('equal')
ax1.grid(True)
ax1.set_title('Original')

ax2.plot(transformed[0], transformed[1], 'r-o')
ax2.set_xlim(-0.5, 2.5)
ax2.set_ylim(-0.5, 2.5)
ax2.set_aspect('equal')
ax2.grid(True)
ax2.set_title(f'After transformation by A')

plt.show()
```

### 4.2 기하학적 변환들

**스케일링 (Scaling)**:

$$\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

**회전 (Rotation)**:

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**반사 (Reflection)**: x축에 대한 반사

$$\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$

**전단 (Shear)**: x 방향 전단

$$\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$$

```python
# 다양한 변환 시각화
theta = np.pi / 6  # 30도

transformations = {
    'Scaling': np.array([[2, 0], [0, 0.5]]),
    'Rotation': np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]]),
    'Reflection': np.array([[1, 0], [0, -1]]),
    'Shear': np.array([[1, 0.5], [0, 1]])
}

# 원본 도형
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()

for idx, (name, A) in enumerate(transformations.items()):
    transformed = A @ square

    axes[idx].plot(square[0], square[1], 'b-o', label='Original', alpha=0.5)
    axes[idx].plot(transformed[0], transformed[1], 'r-o', label='Transformed')
    axes[idx].set_xlim(-1.5, 2.5)
    axes[idx].set_ylim(-1.5, 2.5)
    axes[idx].set_aspect('equal')
    axes[idx].grid(True)
    axes[idx].legend()
    axes[idx].set_title(name)

plt.tight_layout()
plt.show()
```

### 4.3 투영 (Projection)

벡터 $\mathbf{v}$를 벡터 $\mathbf{u}$에 투영:

$$\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\mathbf{v} \cdot \mathbf{u}}{\mathbf{u} \cdot \mathbf{u}} \mathbf{u}$$

부분공간으로의 투영은 행렬로 표현 가능합니다.

```python
# 벡터의 투영
def project(v, u):
    """벡터 v를 u에 투영"""
    return (np.dot(v, u) / np.dot(u, u)) * u

v = np.array([3, 2])
u = np.array([1, 0])

proj_v = project(v, u)

# 시각화
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
          color='blue', width=0.006, label='v')
ax.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1,
          color='red', width=0.006, label='u')
ax.quiver(0, 0, proj_v[0], proj_v[1], angles='xy', scale_units='xy', scale=1,
          color='green', width=0.006, label='proj_u(v)')
ax.plot([proj_v[0], v[0]], [proj_v[1], v[1]], 'k--', alpha=0.5)
ax.set_xlim(-0.5, 4)
ax.set_ylim(-0.5, 3)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title('Vector Projection')
plt.show()
```

## 5. 랭크와 연립방정식

### 5.1 행렬의 랭크 (Rank)

행렬의 랭크는 선형 독립인 행(또는 열)의 최대 개수입니다.

**성질**:
- $\text{rank}(A) \leq \min(m, n)$ for $m \times n$ matrix
- $\text{rank}(A) = \text{rank}(A^T)$
- Full rank: $\text{rank}(A) = \min(m, n)$

```python
# 랭크 계산
A_full = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(f"rank(A_full) = {np.linalg.matrix_rank(A_full)}")  # 2 (not full rank)

A_rank2 = np.array([[1, 2],
                    [3, 4],
                    [5, 6]])
print(f"rank(A_rank2) = {np.linalg.matrix_rank(A_rank2)}")  # 2 (full rank)
```

### 5.2 연립방정식 $A\mathbf{x} = \mathbf{b}$

**해의 존재성**:
- 해가 존재 $\Leftrightarrow$ $\mathbf{b} \in \text{col}(A)$ (A의 열공간)
- $\text{rank}(A) = \text{rank}([A|\mathbf{b}])$이면 해 존재

**해의 유일성**:
- 유일해 $\Leftrightarrow$ A가 full column rank
- 무수히 많은 해 $\Leftrightarrow$ A가 not full column rank

| $\text{rank}(A)$ | $\text{rank}([A\|\mathbf{b}])$ | 해의 개수 |
|------------------|--------------------------------|-----------|
| $r$ | $r$ | 무수히 많음 (if $r < n$) 또는 유일 (if $r = n$) |
| $r$ | $r+1$ | 해 없음 |

```python
# 유일해가 있는 경우
A = np.array([[2, 1],
              [1, 3]])
b = np.array([5, 8])
x = np.linalg.solve(A, b)
print(f"Solution: x = {x}")
print(f"Verification: Ax = {A @ x}")

# 해가 없는 경우 (최소자승해 구하기)
A_overdetermined = np.array([[1, 1],
                             [1, 2],
                             [1, 3]])
b_overdetermined = np.array([1, 2, 4])  # 해가 정확히 없음

# 최소자승해
x_lstsq = np.linalg.lstsq(A_overdetermined, b_overdetermined, rcond=None)[0]
print(f"Least squares solution: {x_lstsq}")
print(f"Residual: {A_overdetermined @ x_lstsq - b_overdetermined}")
```

### 5.3 가우스 소거법 (Gaussian Elimination)

연립방정식을 푸는 알고리즘. 행렬을 행 사다리꼴(row echelon form)로 변환합니다.

```python
def gaussian_elimination(A, b):
    """가우스 소거법으로 Ax = b 풀기"""
    n = len(b)
    # 증강 행렬 생성
    Ab = np.column_stack([A.astype(float), b.astype(float)])

    # 전방 소거
    for i in range(n):
        # 피벗 선택 (부분 피벗팅)
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        Ab[[i, max_row]] = Ab[[max_row, i]]

        # 소거
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # 후방 대입
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]

    return x

A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

x_gauss = gaussian_elimination(A, b)
print(f"Solution by Gaussian elimination: {x_gauss}")
print(f"Verification: {A @ x_gauss}")
```

## 6. ML에서의 벡터와 행렬

### 6.1 피처 벡터 (Feature Vectors)

머신러닝에서 각 데이터 포인트는 벡터로 표현됩니다.

```python
# 예: 붓꽃 데이터
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # (150, 4) - 150개 샘플, 4개 피처
y = iris.target

print(f"Data shape: {X.shape}")
print(f"First sample (feature vector):\n{X[0]}")
print(f"Feature names: {iris.feature_names}")
```

### 6.2 데이터 행렬과 배치 처리

**데이터 행렬**: 각 행이 하나의 샘플 (관례)

$$X = \begin{bmatrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_m^T \end{bmatrix} = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1n} \\ x_{21} & x_{22} & \cdots & x_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{m1} & x_{m2} & \cdots & x_{mn} \end{bmatrix}$$

행렬 연산을 통해 전체 배치를 한번에 처리할 수 있습니다.

```python
# 배치 처리 예제: 선형 회귀
np.random.seed(42)
X = np.random.randn(100, 3)  # 100 샘플, 3 피처
true_w = np.array([2, -1, 0.5])
y = X @ true_w + np.random.randn(100) * 0.1

# 정규방정식으로 해 구하기: w = (X^T X)^{-1} X^T y
w_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"True weights: {true_w}")
print(f"Estimated weights: {w_hat}")

# 예측 (배치 처리)
y_pred = X @ w_hat
print(f"Predictions for first 5 samples:\n{y_pred[:5]}")
```

### 6.3 가중치 행렬 (Weight Matrices)

신경망의 각 층은 가중치 행렬로 표현됩니다.

```python
# 간단한 신경망 층
class DenseLayer:
    def __init__(self, input_dim, output_dim):
        # He 초기화
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2/input_dim)
        self.b = np.zeros(output_dim)

    def forward(self, X):
        """X: (batch_size, input_dim)"""
        return X @ self.W + self.b

# 예제
layer = DenseLayer(input_dim=10, output_dim=5)
X_batch = np.random.randn(32, 10)  # 배치 크기 32
output = layer.forward(X_batch)
print(f"Input shape: {X_batch.shape}")
print(f"Weight matrix shape: {layer.W.shape}")
print(f"Output shape: {output.shape}")
```

### 6.4 코사인 유사도와 거리

벡터 간 유사도를 측정하는 방법:

**코사인 유사도**:

$$\text{similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \cos\theta$$

**유클리드 거리**:

$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\| = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}$$

```python
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# 문서 벡터 예제 (TF-IDF)
doc1 = np.array([1, 2, 0, 1])
doc2 = np.array([2, 1, 1, 0])
doc3 = np.array([0, 0, 1, 1])

docs = np.array([doc1, doc2, doc3])

# 코사인 유사도
cos_sim = cosine_similarity(docs)
print(f"Cosine similarity matrix:\n{cos_sim}")

# 유클리드 거리
eucl_dist = euclidean_distances(docs)
print(f"Euclidean distance matrix:\n{eucl_dist}")
```

## 연습 문제

### 문제 1: 벡터 공간과 기저
다음 벡터들이 $\mathbb{R}^3$의 기저를 이루는지 판별하고, 기저를 이룬다면 벡터 $\mathbf{v} = [1, 2, 3]^T$를 이 기저로 표현하세요.

$$\mathbf{b}_1 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \quad \mathbf{b}_2 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}, \quad \mathbf{b}_3 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$$

### 문제 2: 선형 변환 시각화
다음 변환을 순서대로 적용했을 때의 최종 변환 행렬을 구하고 시각화하세요:
1. 45도 회전
2. x방향으로 2배 스케일링
3. x축에 대한 반사

### 문제 3: 투영 행렬
xy-평면(즉, $z=0$인 평면)으로의 투영 행렬 $P$를 구하세요. 즉, $P\mathbf{v}$가 $\mathbf{v}$를 xy-평면에 투영한 결과가 되도록 하는 $3 \times 3$ 행렬 $P$를 찾으세요.

### 문제 4: 최소자승 문제
다음 과대결정 시스템(overdetermined system)의 최소자승해를 구하세요:

$$\begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 2 \\ 3 \\ 5 \\ 6 \end{bmatrix}$$

### 문제 5: ML 응용
100개의 샘플과 5개의 피처를 가진 데이터셋이 있습니다. 선형 회귀 모델을 학습하려 합니다.
1. 데이터 행렬 $X$와 가중치 벡터 $\mathbf{w}$의 차원을 명시하세요
2. 예측값 $\hat{\mathbf{y}} = X\mathbf{w}$의 차원은?
3. 정규방정식 $\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$에서 각 행렬의 차원을 확인하세요

## 참고 자료

### 교재
1. **Strang, G.** (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.
   - 선형대수의 고전. 기하학적 직관이 뛰어남.
2. **Axler, S.** (2015). *Linear Algebra Done Right* (3rd ed.). Springer.
   - 추상적이지만 엄밀한 접근.
3. **Boyd, S., & Vandenberghe, L.** (2018). *Introduction to Applied Linear Algebra*. Cambridge University Press.
   - 응용 중심의 현대적 접근.

### 온라인 자료
1. **3Blue1Brown - Essence of Linear Algebra**: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
   - 선형대수의 결정적인 시각적 입문서
2. **MIT 18.06 - Linear Algebra (Gilbert Strang)**: https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/
3. **Khan Academy - Linear Algebra**: https://www.khanacademy.org/math/linear-algebra

### 실습 도구
1. **NumPy Documentation**: https://numpy.org/doc/stable/
2. **Matrix Calculator**: https://matrixcalc.org/
3. **Wolfram Alpha**: https://www.wolframalpha.com/

---

**다음 레슨**: [02. 행렬 분해](02_Matrix_Decompositions.md)에서 고유값 분해와 SVD를 배웁니다.
