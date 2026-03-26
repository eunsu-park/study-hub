# 레슨 1: 벡터와 벡터 공간 (Vectors and Vector Spaces)

## 학습 목표

- 벡터를 기하학적, 대수적으로 정의하고 Python으로 덧셈, 스칼라 곱, 내적을 수행할 수 있다
- 랭크 검사를 사용하여 벡터 집합의 선형 독립성을 판별할 수 있다
- 기저, 생성, 차원의 개념을 설명하고 벡터를 다른 기저로 표현할 수 있다
- $\mathbb{R}^n$의 부분공간을 식별하고 검증할 수 있다
- $\mathbb{R}^3$에서 외적을 계산하고 그 결과를 기하학적으로 해석할 수 있다

---

## 1. 벡터란 무엇인가?

벡터는 **크기**와 **방향**을 모두 가진 양입니다. 벡터는 두 가지 상호 보완적인 관점에서 이해할 수 있습니다:

1. **기하학적 관점** -- 한 위치에서 다른 위치를 가리키는 공간 속의 화살표
2. **대수적 관점** -- 순서가 있는 숫자들의 목록 (튜플)

$n$차원 실수 벡터 $\mathbf{v}$는 열(column)로 다음과 같이 표기합니다:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

NumPy에서는 벡터를 1차원 배열로 표현합니다:

```python
import numpy as np
import matplotlib.pyplot as plt

# Create 2D and 3D vectors
v2 = np.array([3, 2])
v3 = np.array([1, -4, 7])

print(f"2D vector: {v2}, shape: {v2.shape}")
print(f"3D vector: {v3}, shape: {v3.shape}")
```

### 2D 벡터 시각화

```python
fig, ax = plt.subplots(figsize=(6, 6))

v = np.array([3, 2])
w = np.array([-1, 4])

ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
          color='tab:red', width=0.008, label=r'$\mathbf{v} = (3,2)$')
ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1,
          color='tab:blue', width=0.008, label=r'$\mathbf{w} = (-1,4)$')

ax.set_xlim(-3, 5)
ax.set_ylim(-1, 6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_title('Two vectors in the plane')
plt.show()
```

---

## 2. 벡터 연산

### 2.1 벡터 덧셈

같은 차원의 두 벡터는 성분별로 더합니다. 기하학적으로는 **평행사변형 법칙**을 따릅니다 -- 첫 번째 벡터의 끝점에 두 번째 벡터의 시작점을 놓습니다.

$$\mathbf{v} + \mathbf{w} = \begin{bmatrix} v_1 + w_1 \\ v_2 + w_2 \\ \vdots \\ v_n + w_n \end{bmatrix}$$

```python
v = np.array([3, 2])
w = np.array([1, 4])
s = v + w
print(f"v + w = {s}")  # [4, 6]

# Visualize addition
fig, ax = plt.subplots(figsize=(7, 7))
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
          color='tab:red', width=0.007, label='v')
ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1,
          color='tab:blue', width=0.007, label='w')
ax.quiver(0, 0, s[0], s[1], angles='xy', scale_units='xy', scale=1,
          color='purple', width=0.007, label='v + w')
# Parallelogram sides
ax.quiver(v[0], v[1], w[0], w[1], angles='xy', scale_units='xy', scale=1,
          color='gray', width=0.004, alpha=0.5, linestyle='--')
ax.quiver(w[0], w[1], v[0], v[1], angles='xy', scale_units='xy', scale=1,
          color='gray', width=0.004, alpha=0.5, linestyle='--')
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Vector Addition (Parallelogram Law)')
plt.show()
```

### 2.2 스칼라 곱

스칼라 $c \in \mathbb{R}$를 벡터에 곱하면 크기가 변하고, $c < 0$이면 방향이 반전됩니다.

$$c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$$

```python
v = np.array([2, 1])
scales = [0.5, 1, 2, -1]

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:orange']
for c, col in zip(scales, colors):
    cv = c * v
    ax.quiver(0, 0, cv[0], cv[1], angles='xy', scale_units='xy', scale=1,
              color=col, width=0.006, label=f'{c}v = {cv}')
ax.set_xlim(-3, 5)
ax.set_ylim(-2, 3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Scalar Multiplication')
plt.show()
```

### 2.3 내적 (Dot Product)

두 벡터의 내적은 스칼라 값을 반환합니다:

$$\mathbf{v} \cdot \mathbf{w} = \sum_{i=1}^{n} v_i w_i = v_1 w_1 + v_2 w_2 + \cdots + v_n w_n$$

**기하학적 해석**:

$$\mathbf{v} \cdot \mathbf{w} = \|\mathbf{v}\| \, \|\mathbf{w}\| \cos\theta$$

여기서 $\theta$는 두 벡터 사이의 각도입니다.

| 내적 값 | 각도 관계 |
|---|---|
| $\mathbf{v} \cdot \mathbf{w} > 0$ | 예각 (90도 미만) |
| $\mathbf{v} \cdot \mathbf{w} = 0$ | 직각 (직교) |
| $\mathbf{v} \cdot \mathbf{w} < 0$ | 둔각 (90도 초과) |

```python
v = np.array([3, 1])
w = np.array([1, 3])

# Compute dot product
dot = np.dot(v, w)
print(f"v . w = {dot}")  # 6

# Compute angle between vectors
norm_v = np.linalg.norm(v)
norm_w = np.linalg.norm(w)
cos_theta = dot / (norm_v * norm_w)
theta_rad = np.arccos(np.clip(cos_theta, -1, 1))
theta_deg = np.degrees(theta_rad)
print(f"||v|| = {norm_v:.4f}")
print(f"||w|| = {norm_w:.4f}")
print(f"Angle = {theta_deg:.2f} degrees")

# Orthogonal check
u = np.array([1, 0])
u_perp = np.array([0, 1])
print(f"u . u_perp = {np.dot(u, u_perp)}")  # 0
```

### 2.4 외적 (Cross Product) - 3D 전용

외적은 $\mathbb{R}^3$에서만 정의되며, 두 입력 벡터에 모두 **수직인** 벡터를 반환합니다:

$$\mathbf{v} \times \mathbf{w} = \begin{bmatrix} v_2 w_3 - v_3 w_2 \\ v_3 w_1 - v_1 w_3 \\ v_1 w_2 - v_2 w_1 \end{bmatrix}$$

외적의 크기는 $\mathbf{v}$와 $\mathbf{w}$가 이루는 평행사변형의 넓이와 같습니다:

$$\|\mathbf{v} \times \mathbf{w}\| = \|\mathbf{v}\| \, \|\mathbf{w}\| \sin\theta$$

```python
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])

cross = np.cross(v, w)
print(f"v x w = {cross}")

# Verify perpendicularity
print(f"(v x w) . v = {np.dot(cross, v)}")  # 0
print(f"(v x w) . w = {np.dot(cross, w)}")  # 0

# Parallelogram area
area = np.linalg.norm(cross)
print(f"Parallelogram area = {area:.4f}")
```

---

## 3. 벡터 공간 (Vector Spaces)

### 3.1 정의

**벡터 공간** $V$는 $\mathbb{R}$ 위의 집합으로, 벡터 덧셈과 스칼라 곱이라는 두 가지 연산이 정의되어 있으며, 모든 $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$와 $a, b \in \mathbb{R}$에 대해 다음 여덟 가지 공리를 만족합니다:

1. 교환법칙: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
2. 덧셈의 결합법칙: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
3. 덧셈의 항등원: $\mathbf{v} + \mathbf{0} = \mathbf{v}$인 $\mathbf{0}$이 존재
4. 덧셈의 역원: 각 $\mathbf{v}$에 대해 $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$인 $-\mathbf{v}$가 존재
5. 곱셈의 항등원: $1 \cdot \mathbf{v} = \mathbf{v}$
6. 호환성: $a(b\mathbf{v}) = (ab)\mathbf{v}$
7. 벡터 덧셈에 대한 분배법칙: $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$
8. 스칼라 덧셈에 대한 분배법칙: $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$

**대표적인 예시**:
- $\mathbb{R}^n$ -- 모든 $n$-튜플 실수의 공간
- $P_n$ -- 차수가 $n$ 이하인 모든 다항식
- $C[a, b]$ -- $[a, b]$에서 연속인 모든 함수

### 3.2 부분공간 (Subspaces)

공집합이 아닌 부분집합 $W \subseteq V$가 같은 연산 하에서 자체적으로 벡터 공간을 이루면 **부분공간**이라 합니다. 부분공간 판정에는 세 가지 조건만 확인하면 됩니다:

1. $\mathbf{0} \in W$
2. 덧셈에 닫혀있음: $\mathbf{u}, \mathbf{v} \in W \implies \mathbf{u} + \mathbf{v} \in W$
3. 스칼라 곱에 닫혀있음: $\mathbf{v} \in W,\, c \in \mathbb{R} \implies c\mathbf{v} \in W$

```python
# Example: the plane z = 0 in R^3 is a subspace
u = np.array([2, -1, 0])
v = np.array([3, 5, 0])

# Check closure
print(f"u + v = {u + v}")      # z-component is still 0
print(f"3 * u = {3 * u}")      # z-component is still 0
print(f"Contains zero? {np.allclose(0 * u, np.zeros(3))}")

# Counter-example: the plane z = 1 is NOT a subspace
w = np.array([1, 0, 1])
print(f"2 * w = {2 * w}")  # z-component is 2, not 1 => not closed
```

---

## 4. 선형 결합과 생성 (Span)

### 4.1 선형 결합

벡터 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$의 **선형 결합**은 다음과 같은 형태의 모든 표현식입니다:

$$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_k \mathbf{v}_k, \quad c_i \in \mathbb{R}$$

```python
v1 = np.array([1, 0])
v2 = np.array([0, 1])

# Some linear combinations
combo1 = 3 * v1 + 2 * v2   # [3, 2]
combo2 = -1 * v1 + 4 * v2  # [-1, 4]
print(f"3*v1 + 2*v2 = {combo1}")
print(f"-1*v1 + 4*v2 = {combo2}")
```

### 4.2 생성 (Span)

벡터 집합의 **생성(span)**은 모든 가능한 선형 결합의 집합입니다:

$$\mathrm{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \left\{ \sum_{i=1}^k c_i \mathbf{v}_i \;\middle|\; c_i \in \mathbb{R} \right\}$$

생성은 항상 주어진 벡터 공간의 부분공간입니다.

```python
from mpl_toolkits.mplot3d import Axes3D

# Span of two non-parallel vectors in R^3 forms a plane
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

# Sample many linear combinations
t = np.linspace(-2, 2, 30)
points = []
for c1 in t:
    for c2 in t:
        points.append(c1 * v1 + c2 * v2)
points = np.array(points)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.4)
ax.quiver(0, 0, 0, *v1, color='r', arrow_length_ratio=0.1, label='v1')
ax.quiver(0, 0, 0, *v2, color='b', arrow_length_ratio=0.1, label='v2')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('span(v1, v2) = the xy-plane')
ax.legend()
plt.show()
```

---

## 5. 선형 독립 (Linear Independence)

### 5.1 정의

벡터 $\mathbf{v}_1, \ldots, \mathbf{v}_k$가 **선형 독립**이려면 다음을 만족하는 유일한 해가 $c_1 = c_2 = \cdots = c_k = 0$이어야 합니다:

$$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_k \mathbf{v}_k = \mathbf{0}$$

자명하지 않은 해 (모든 $c_i$가 0이 아닌)가 존재하면, 벡터들은 **선형 종속**입니다 -- 적어도 하나의 벡터가 나머지의 선형 결합으로 표현될 수 있습니다.

### 5.2 실용적 판정법

벡터들을 행렬 $A$의 열로 배치합니다. 그러면:

- **선형 독립** $\iff$ $\mathrm{rank}(A) = k$ (벡터의 개수)

```python
# Independent set
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
A_indep = np.column_stack([v1, v2, v3])
print(f"Rank = {np.linalg.matrix_rank(A_indep)}, #vectors = 3 => independent")

# Dependent set (v3 = v1 + v2)
w1 = np.array([1, 0, 0])
w2 = np.array([0, 1, 0])
w3 = np.array([1, 1, 0])
A_dep = np.column_stack([w1, w2, w3])
print(f"Rank = {np.linalg.matrix_rank(A_dep)}, #vectors = 3 => dependent")
```

### 5.3 기하학적 직관

- $\mathbb{R}^2$에서 두 벡터가 독립인 것은 원점을 지나는 같은 직선 위에 있지 않은 것과 동치입니다.
- $\mathbb{R}^3$에서 세 벡터가 독립인 것은 원점을 지나는 같은 평면 위에 있지 않은 것과 동치입니다.
- $\mathbb{R}^n$에서 $n$개를 초과하는 선형 독립 벡터는 존재할 수 없습니다.

```python
# Visual: two independent vs two dependent vectors in R^2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Independent
v = np.array([2, 1])
w = np.array([1, 3])
ax1.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
           color='r', width=0.008, label='v')
ax1.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1,
           color='b', width=0.008, label='w')
ax1.set_xlim(-1, 4); ax1.set_ylim(-1, 4)
ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)
ax1.legend(); ax1.set_title('Linearly Independent')

# Dependent (w = 2v)
v = np.array([1, 2])
w = np.array([2, 4])
ax2.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
           color='r', width=0.008, label='v')
ax2.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1,
           color='b', width=0.008, label='w = 2v')
ax2.set_xlim(-1, 5); ax2.set_ylim(-1, 5)
ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
ax2.legend(); ax2.set_title('Linearly Dependent')

plt.tight_layout()
plt.show()
```

---

## 6. 기저와 차원 (Basis and Dimension)

### 6.1 기저 (Basis)

벡터 공간 $V$의 **기저**는 다음 두 조건을 만족하는 벡터 집합 $\{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$입니다:

1. **선형 독립**
2. $V$를 **생성**

$V$의 모든 벡터는 기저 벡터의 **유일한** 선형 결합으로 표현할 수 있습니다.

### 6.2 표준 기저 (Standard Basis)

$\mathbb{R}^n$의 **표준 기저**는 단위 벡터들로 구성됩니다:

$$\mathbf{e}_1 = \begin{bmatrix}1\\0\\\vdots\\0\end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix}0\\1\\\vdots\\0\end{bmatrix}, \quad \ldots, \quad \mathbf{e}_n = \begin{bmatrix}0\\0\\\vdots\\1\end{bmatrix}$$

### 6.3 차원 (Dimension)

벡터 공간의 **차원**은 기저를 이루는 벡터의 개수입니다. 주어진 벡터 공간의 모든 기저는 같은 개수의 원소를 가집니다.

- $\dim(\mathbb{R}^n) = n$
- $\dim(P_k) = k + 1$ (차수 $k$ 이하의 다항식)

### 6.4 기저 변환 (Change of Basis)

$B = [\mathbf{b}_1 \mid \cdots \mid \mathbf{b}_n]$이 새로운 기저의 열로 구성된 행렬이면, $\mathbf{v}$의 새 기저에서의 좌표는 다음과 같습니다:

$$[\mathbf{v}]_B = B^{-1} \mathbf{v}$$

```python
# Standard basis representation
v = np.array([5, 3, -2])
e1, e2, e3 = np.eye(3)
print(f"v = {v[0]}*e1 + {v[1]}*e2 + {v[2]}*e3")

# Change of basis
b1 = np.array([1, 1, 0])
b2 = np.array([0, 1, 1])
b3 = np.array([1, 0, 1])
B = np.column_stack([b1, b2, b3])

# Coordinates in the new basis
coords_B = np.linalg.solve(B, v)
print(f"Coordinates in new basis: {coords_B}")

# Verify: reconstruct from new basis
v_reconstructed = coords_B[0] * b1 + coords_B[1] * b2 + coords_B[2] * b3
print(f"Reconstructed: {v_reconstructed}")
print(f"Match: {np.allclose(v, v_reconstructed)}")
```

---

## 7. 핵심 개념 요약

| 개념 | 정의 | NumPy |
|------|------|-------|
| 벡터 덧셈 | $\mathbf{v} + \mathbf{w}$ | `v + w` |
| 스칼라 곱 | $c\mathbf{v}$ | `c * v` |
| 내적 | $\mathbf{v} \cdot \mathbf{w}$ | `np.dot(v, w)` |
| 외적 | $\mathbf{v} \times \mathbf{w}$ | `np.cross(v, w)` |
| 노름 | $\|\mathbf{v}\|$ | `np.linalg.norm(v)` |
| 선형 독립 판정 | $\mathrm{rank}(A) = k$ | `np.linalg.matrix_rank(A)` |
| 기저 변환 | $B^{-1}\mathbf{v}$ | `np.linalg.solve(B, v)` |

---

## 연습 문제

### 연습 문제 1: 벡터 산술

$\mathbf{u} = [2, -1, 3]^T$와 $\mathbf{v} = [-1, 4, 2]^T$가 주어졌을 때:

1. $\mathbf{u} + \mathbf{v}$, $3\mathbf{u} - 2\mathbf{v}$, 그리고 $\mathbf{u} \cdot \mathbf{v}$를 계산하세요.
2. $\mathbf{u}$와 $\mathbf{v}$ 사이의 각도를 도(degrees)로 구하세요.
3. $\mathbf{u} \times \mathbf{v}$를 계산하고 두 입력 벡터에 수직인지 검증하세요.

### 연습 문제 2: 선형 독립

다음 벡터 집합이 선형 독립인지 판별하세요. 이유를 설명하세요.

(a) $\{[1, 2]^T, [3, 6]^T\}$

(b) $\{[1, 0, 1]^T, [0, 1, 1]^T, [1, 1, 0]^T\}$

(c) $\{[1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T, [1, 1, 1]^T\}$

### 연습 문제 3: 기저와 좌표

$\mathbf{b}_1 = [1, 1]^T$과 $\mathbf{b}_2 = [1, -1]^T$이 $\mathbb{R}^2$의 기저를 이루는 것을 보이세요. 그런 다음 $\mathbf{v} = [5, 3]^T$를 이 기저로 표현하세요.

### 연습 문제 4: 부분공간 검증

다음 중 $\mathbb{R}^3$의 부분공간인 것은 어느 것입니까?

(a) $W_1 = \{[x, y, z]^T : x + y + z = 0\}$

(b) $W_2 = \{[x, y, z]^T : x^2 + y^2 + z^2 \le 1\}$

(c) $W_3 = \{[x, y, z]^T : x = 2y\}$

### 연습 문제 5: 생성 (Span)

$\{[1, 2, 3]^T, [4, 5, 6]^T, [7, 8, 9]^T\}$의 생성을 구하세요. 이 생성의 차원은 무엇입니까? (힌트: 이 벡터들로 구성된 행렬의 랭크를 확인하세요.)

---

[개요](00_Overview.md) | [다음: 레슨 2 - 행렬과 연산 >>](02_Matrices_and_Operations.md)

**License**: CC BY-NC 4.0
