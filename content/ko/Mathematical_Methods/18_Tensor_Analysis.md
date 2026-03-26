# 18. 텐서 해석 (Tensor Analysis)

## 학습 목표
- 텐서의 정의를 좌표 변환 법칙의 관점에서 이해하고, 스칼라·벡터·행렬을 텐서의 특수한 경우로 분류할 수 있다
- 아인슈타인 합산 규약과 인덱스 표기법을 사용하여 텐서 방정식을 간결하게 표현하고 조작할 수 있다
- 반변(contravariant)·공변(covariant) 텐서의 변환 법칙을 구분하고, 계량 텐서를 통한 인덱스 올림/내림을 수행할 수 있다
- 크리스토펠 기호와 공변 미분의 개념을 이해하고, 곡선 좌표계에서 텐서의 미분을 올바르게 계산할 수 있다
- 리만 곡률 텐서의 정의와 기하학적 의미를 이해하고, 간단한 공간에서 곡률을 계산할 수 있다
- 텐서 해석의 물리적 응용(응력 텐서, 전자기장 텐서, 아인슈타인 방정식)을 서술하고 Python으로 계산할 수 있다

> **텐서는 왜 필요한가?** 자연 법칙은 좌표계의 선택에 무관해야 한다. 스칼라(rank-0)와 벡터(rank-1)만으로는 응력, 관성 모멘트, 전자기장 같은 물리량을 기술할 수 없다. 텐서는 임의의 좌표 변환에서 명확한 변환 법칙을 따르는 기하학적 대상으로, 물리 법칙을 좌표계에 독립적인 형태로 표현하는 자연스러운 언어이다.

---

## 1. 텐서의 기본 개념

### 1.1 동기: 왜 스칼라와 벡터만으로는 부족한가

물리학에서 많은 양은 스칼라(온도, 에너지)나 벡터(힘, 속도)로 충분히 기술된다. 그러나 다음과 같은 물리량은 하나의 방향이 아니라 **두 개 이상의 방향** 정보를 필요로 한다:

- **응력 텐서(stress tensor)** $\sigma_{ij}$: 면의 방향($j$)과 그 면에 작용하는 힘의 방향($i$)
- **관성 모멘트 텐서(moment of inertia tensor)** $I_{ij}$: 각속도 방향과 각운동량 방향의 관계
- **전자기장 텐서(electromagnetic field tensor)** $F_{\mu\nu}$: 전기장과 자기장을 통합하는 반대칭 텐서

이들은 모두 **rank-2 텐서**로, $n$차원 공간에서 $n^2$개의 성분을 가진다.

### 1.2 좌표 변환과 텐서의 정의

좌표 변환 $x^i \to x'^i(x^1, x^2, \ldots, x^n)$에서, **rank-$k$ 텐서**는 그 성분이 다음과 같은 특정한 변환 법칙을 따르는 대상이다:

- **Rank-0 (스칼라)**: $\phi' = \phi$ (불변)
- **Rank-1 (벡터)**: $A'^i = \frac{\partial x'^i}{\partial x^j}A^j$ (반변 벡터의 경우)
- **Rank-2 텐서**: $T'^{ij} = \frac{\partial x'^i}{\partial x^k}\frac{\partial x'^j}{\partial x^l}T^{kl}$

일반적으로, **텐서는 다중선형 사상(multilinear map)**으로 정의할 수 있다. rank-$(p,q)$ 텐서는 $p$개의 공변 벡터(covector)와 $q$개의 반변 벡터를 받아 실수를 내놓는 다중선형 함수이다.

### 1.3 텐서의 계수(Rank/Order)

| Rank | 성분 수 ($n$차원) | 물리적 예시 |
|------|-------------------|-------------|
| 0 | $1$ | 온도, 질량, 에너지 |
| 1 | $n$ | 힘, 속도, 전기장 |
| 2 | $n^2$ | 응력, 관성 모멘트, 계량 텐서 |
| 3 | $n^3$ | 압전(piezoelectric) 텐서 |
| 4 | $n^4$ | 리만 곡률 텐서, 탄성 텐서 |

### 1.4 Python: 좌표 변환 예제

```python
import numpy as np

# === 2차원 회전 변환에서 텐서 변환 법칙 검증 ===

# 회전 각도
theta = np.pi / 6  # 30도

# 변환 행렬: x'^i = R^i_j x^j
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)]
])
print(f"회전 행렬 R (θ = {np.degrees(theta):.0f}°):")
print(R)

# --- Rank-1 텐서 (벡터) 변환 ---
A = np.array([3.0, 4.0])  # 원래 좌표계의 벡터
A_prime = R @ A            # A'^i = R^i_j A^j
print(f"\n원래 벡터: A = {A}")
print(f"변환된 벡터: A' = {A_prime}")
print(f"|A| = {np.linalg.norm(A):.4f}, |A'| = {np.linalg.norm(A_prime):.4f}")
# 크기 보존 확인

# --- Rank-2 텐서 변환 ---
# 응력 텐서 예시
T = np.array([
    [10.0, 3.0],
    [3.0,  5.0]
])  # 대칭 텐서

# T'^{ij} = R^i_k R^j_l T^{kl} = R T R^T
T_prime = R @ T @ R.T
print(f"\n원래 텐서:\n{T}")
print(f"변환된 텐서:\n{T_prime}")

# 텐서의 불변량(trace, determinant) 확인
print(f"\ntr(T) = {np.trace(T):.4f}, tr(T') = {np.trace(T_prime):.4f}")
print(f"det(T) = {np.linalg.det(T):.4f}, det(T') = {np.linalg.det(T_prime):.4f}")
# 대각합과 행렬식은 좌표 변환에 불변
```

---

## 2. 인덱스 표기법과 아인슈타인 합산 규약

### 2.1 아인슈타인 합산 규약

아인슈타인(Einstein)의 합산 규약은 텐서 표현을 간결하게 만드는 표기법이다:

> **같은 항에서 위 인덱스와 아래 인덱스가 동일한 문자로 반복되면, 그 인덱스에 대해 합산한다.**

$$
A^i B_i \equiv \sum_{i=1}^{n} A^i B_i
$$

**자유 인덱스(free index)**: 항의 양변에 공통으로 나타나는 인덱스 (방정식의 각 성분을 표현)

**더미 인덱스(dummy index)**: 합산 규약에 의해 합산되는 반복 인덱스 (문자 교체 가능)

$$
A^i B_i = A^j B_j \quad (\text{더미 인덱스 } i \text{를 } j\text{로 교체 가능})
$$

### 2.2 크로네커 델타 $\delta_{ij}$

**크로네커 델타(Kronecker delta)**는 단위 행렬의 성분에 해당한다:

$$
\delta_{ij} = \begin{cases} 1 & (i = j) \\ 0 & (i \ne j) \end{cases}
$$

주요 성질:
- $\delta_{ij} A^j = A^i$ (인덱스 치환 역할)
- $\delta_{ii} = n$ ($n$차원에서의 대각합)
- $\delta_{ij}\delta_{jk} = \delta_{ik}$

### 2.3 레비-치비타 기호 $\varepsilon_{ijk}$

**레비-치비타 기호(Levi-Civita symbol)**는 완전 반대칭 기호이다:

$$
\varepsilon_{ijk} = \begin{cases} +1 & (ijk) \text{가 }(123)\text{의 짝수 치환} \\ -1 & (ijk) \text{가 }(123)\text{의 홀수 치환} \\ 0 & \text{인덱스 중 반복이 있으면} \end{cases}
$$

**외적과 행렬식의 표현:**

$$
(\mathbf{A} \times \mathbf{B})_i = \varepsilon_{ijk} A_j B_k
$$

$$
\det(M) = \varepsilon_{ijk} M_{1i} M_{2j} M_{3k}
$$

### 2.4 $\varepsilon$-$\delta$ 항등식

텐서 계산에서 매우 유용한 항등식:

$$
\varepsilon_{ijk}\varepsilon_{imn} = \delta_{jm}\delta_{kn} - \delta_{jn}\delta_{km}
$$

이 항등식으로부터 벡터 삼중곱 등 다양한 항등식을 유도할 수 있다:

$$
\mathbf{A} \times (\mathbf{B} \times \mathbf{C}) = \mathbf{B}(\mathbf{A} \cdot \mathbf{C}) - \mathbf{C}(\mathbf{A} \cdot \mathbf{B})
$$

### 2.5 Python: numpy.einsum을 이용한 텐서 연산

```python
import numpy as np

# === numpy.einsum: 아인슈타인 합산 규약의 Python 구현 ===

A = np.array([1.0, 2.0, 3.0])
B = np.array([4.0, 5.0, 6.0])

# 내적: A^i B_i
dot = np.einsum('i,i->', A, B)
print(f"내적 A·B = {dot}")  # 32.0

# 외적 (텐서곱): C^{ij} = A^i B^j
outer = np.einsum('i,j->ij', A, B)
print(f"\n외적 A⊗B:\n{outer}")

# 행렬-벡터 곱: (Mv)^i = M^{ij} v_j
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
v = np.array([1.0, 0.0, -1.0])
Mv = np.einsum('ij,j->i', M, v)
print(f"\nMv = {Mv}")

# 행렬 곱: (AB)^{ik} = A^{ij} B^{jk}
N = np.random.rand(3, 3)
MN = np.einsum('ij,jk->ik', M, N)
print(f"\n행렬 곱 MN (einsum):\n{MN}")
print(f"행렬 곱 MN (numpy):  \n{M @ N}")

# 대각합(trace): T^i_i
trace = np.einsum('ii->', M)
print(f"\ntr(M) = {trace}")

# 이중 축약(double contraction): A_{ij} B_{ij}
A2 = np.random.rand(3, 3)
B2 = np.random.rand(3, 3)
double_contract = np.einsum('ij,ij->', A2, B2)
print(f"A:B (이중 축약) = {double_contract:.4f}")

# --- 레비-치비타 기호와 외적 ---
# 3차원 레비-치비타 기호 생성
def levi_civita_3d():
    """3차원 레비-치비타 기호 ε_{ijk}를 생성"""
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
    return eps

eps = levi_civita_3d()

# 외적: (A × B)_i = ε_{ijk} A_j B_k
cross_einsum = np.einsum('ijk,j,k->i', eps, A, B)
cross_numpy = np.cross(A, B)
print(f"\nA × B (einsum): {cross_einsum}")
print(f"A × B (numpy):  {cross_numpy}")

# ε-δ 항등식 검증: ε_{ijk} ε_{imn} = δ_{jm}δ_{kn} - δ_{jn}δ_{km}
lhs = np.einsum('ijk,imn->jkmn', eps, eps)
delta = np.eye(3)
rhs = np.einsum('jm,kn->jkmn', delta, delta) - np.einsum('jn,km->jkmn', delta, delta)
print(f"\nε-δ 항등식 검증: {np.allclose(lhs, rhs)}")
```

---

## 3. 반변·공변 텐서

### 3.1 반변 벡터 (Contravariant Vector)

좌표 변환 $x^i \to x'^i$에서, **반변 벡터(contravariant vector)**의 성분은 좌표의 미분처럼 변환한다:

$$
A'^i = \frac{\partial x'^i}{\partial x^j}A^j
$$

위 인덱스(upper index)로 표기하며, 변위 벡터 $dx^i$가 대표적인 예이다.

### 3.2 공변 벡터 (Covariant Vector)

**공변 벡터(covariant vector, 1-form)**의 성분은 기울기(gradient)처럼 변환한다:

$$
A'_i = \frac{\partial x^j}{\partial x'^i}A_j
$$

아래 인덱스(lower index)로 표기하며, 스칼라 함수의 편미분 $\partial_i \phi = \frac{\partial \phi}{\partial x^i}$가 대표적인 예이다.

### 3.3 혼합 텐서와 인덱스 올림/내림

**혼합 텐서(mixed tensor)**: 위·아래 인덱스를 모두 가진 텐서. 예를 들어 rank-$(1,1)$ 텐서 $T^i{}_j$는:

$$
T'^i{}_j = \frac{\partial x'^i}{\partial x^k}\frac{\partial x^l}{\partial x'^j}T^k{}_l
$$

**인덱스 올림/내림**은 계량 텐서를 통해 수행한다:

$$
A^i = g^{ij}A_j \quad (\text{올림}), \qquad A_i = g_{ij}A^j \quad (\text{내림})
$$

### 3.4 Python: 반변/공변 변환 예제

```python
import numpy as np
import sympy as sp

# === 극좌표 ↔ 직교좌표 변환에서 반변/공변 벡터 ===

r_val, theta_val = 2.0, np.pi / 4  # (r, θ) = (2, 45°)

# 직교 좌표 → 극좌표 변환의 야코비안
# x = r cosθ, y = r sinθ
# ∂x^i/∂x'^j 계산 (x' = 극좌표, x = 직교좌표)

# ∂(x,y)/∂(r,θ) : 직교를 극좌표로 미분
J = np.array([
    [np.cos(theta_val), -r_val * np.sin(theta_val)],  # ∂x/∂r, ∂x/∂θ
    [np.sin(theta_val),  r_val * np.cos(theta_val)]   # ∂y/∂r, ∂y/∂θ
])

# 역야코비안: ∂(r,θ)/∂(x,y)
J_inv = np.linalg.inv(J)

print("야코비안 ∂(x,y)/∂(r,θ):")
print(J)
print(f"\n역야코비안 ∂(r,θ)/∂(x,y):")
print(J_inv)

# 직교좌표에서의 벡터 (반변 성분)
A_cart = np.array([1.0, 1.0])  # (Ax, Ay)

# 반변 변환: A'^i = (∂x'^i/∂x^j) A^j
# 극좌표의 반변 성분 = J_inv @ A_cart
A_polar_contra = J_inv @ A_cart
print(f"\n직교좌표 벡터 A = {A_cart}")
print(f"극좌표 반변 성분 (A^r, A^θ) = {A_polar_contra}")

# 공변 변환: A'_i = (∂x^j/∂x'^i) A_j
# 극좌표의 공변 성분 = J^T @ A_cart
A_polar_cov = J.T @ A_cart
print(f"극좌표 공변 성분 (A_r, A_θ) = {A_polar_cov}")

# 극좌표의 계량 텐서로 검증: g_{ij} = diag(1, r²)
g = np.diag([1.0, r_val**2])
A_cov_from_contra = g @ A_polar_contra
print(f"\ng_{ij} A^j = {A_cov_from_contra}")
print(f"직접 계산한 공변 성분과 일치: {np.allclose(A_polar_cov, A_cov_from_contra)}")
```

---

## 4. 계량 텐서 (Metric Tensor)

### 4.1 선소(Line Element)와 계량 텐서의 정의

좌표계에서 두 인접한 점 사이의 거리(선소, line element)는 계량 텐서 $g_{ij}$에 의해 정의된다:

$$
ds^2 = g_{ij} \, dx^i \, dx^j
$$

계량 텐서는 내적을 정의하는 **대칭 rank-2 공변 텐서**이다: $g_{ij} = g_{ji}$.

### 4.2 다양한 좌표계에서의 계량 텐서

**직교 좌표** $(x, y, z)$:

$$
ds^2 = dx^2 + dy^2 + dz^2 \quad \Rightarrow \quad g_{ij} = \delta_{ij} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}
$$

**원통 좌표** $(\rho, \phi, z)$:

$$
ds^2 = d\rho^2 + \rho^2 d\phi^2 + dz^2 \quad \Rightarrow \quad g_{ij} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \rho^2 & 0 \\ 0 & 0 & 1 \end{pmatrix}
$$

**구면 좌표** $(r, \theta, \phi)$:

$$
ds^2 = dr^2 + r^2 d\theta^2 + r^2 \sin^2\theta \, d\phi^2 \quad \Rightarrow \quad g_{ij} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & r^2 & 0 \\ 0 & 0 & r^2\sin^2\theta \end{pmatrix}
$$

### 4.3 곡면 위의 계량 텐서

**2차원 구면** ($r = R$ 고정):

$$
ds^2 = R^2 d\theta^2 + R^2 \sin^2\theta \, d\phi^2 \quad \Rightarrow \quad g_{ij} = R^2\begin{pmatrix} 1 & 0 \\ 0 & \sin^2\theta \end{pmatrix}
$$

### 4.4 역계량 텐서와 체적 요소

**역계량 텐서(inverse metric)** $g^{ij}$는 $g^{ik}g_{kj} = \delta^i{}_j$를 만족한다.

**체적 요소**: 좌표계의 체적 요소는 계량 텐서의 행렬식으로 결정된다:

$$
dV = \sqrt{|g|} \, dx^1 \, dx^2 \cdots dx^n, \quad g = \det(g_{ij})
$$

### 4.5 Python: 계량 텐서 계산

```python
import sympy as sp

# === 다양한 좌표계에서 계량 텐서 계산 ===

def compute_metric(coords, transform):
    """
    좌표 변환으로부터 계량 텐서를 계산한다.

    Parameters:
        coords: 곡선좌표 변수 리스트
        transform: 직교좌표 표현 리스트 [x(...), y(...), z(...)]
    Returns:
        계량 텐서 행렬 (SymPy Matrix)
    """
    n = len(coords)
    r = sp.Matrix(transform)
    g = sp.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            g[i, j] = sp.trigsimp(r.diff(coords[i]).dot(r.diff(coords[j])))
            g[j, i] = g[i, j]
    return g

# 구면 좌표
r, theta, phi = sp.symbols('r theta phi', positive=True)
g_sph = compute_metric(
    [r, theta, phi],
    [r * sp.sin(theta) * sp.cos(phi),
     r * sp.sin(theta) * sp.sin(phi),
     r * sp.cos(theta)]
)
print("구면 좌표 계량 텐서:")
sp.pprint(g_sph)
print(f"det(g) = {sp.trigsimp(g_sph.det())}")
print(f"sqrt(|g|) = {sp.sqrt(sp.trigsimp(g_sph.det()))}")
# r^4 sin^2(theta) → sqrt = r^2 sin(theta)

# 2차원 구면 (r = R 고정)
R = sp.Symbol('R', positive=True)
g_sphere = compute_metric(
    [theta, phi],
    [R * sp.sin(theta) * sp.cos(phi),
     R * sp.sin(theta) * sp.sin(phi),
     R * sp.cos(theta)]
)
print("\n2차원 구면 계량 텐서:")
sp.pprint(g_sphere)

# 역계량 텐서
g_sph_inv = g_sph.inv()
print("\n구면 좌표 역계량 텐서 g^{ij}:")
sp.pprint(sp.simplify(g_sph_inv))

# 검증: g^{ik} g_{kj} = δ^i_j
identity_check = sp.simplify(g_sph_inv * g_sph)
print(f"\ng^{{ik}} g_{{kj}} = I 검증: {identity_check == sp.eye(3)}")
```

---

## 5. 텐서 대수

### 5.1 텐서의 덧셈과 스칼라 곱

같은 type의 텐서끼리만 더할 수 있다:

$$
(A + B)^{ij} = A^{ij} + B^{ij}, \quad (\alpha A)^{ij} = \alpha A^{ij}
$$

### 5.2 텐서곱 (Tensor Product, Outer Product)

rank-$(p,q)$ 텐서와 rank-$(r,s)$ 텐서의 텐서곱은 rank-$(p+r, q+s)$ 텐서를 생성한다:

$$
(A \otimes B)^{ij}{}_{kl} = A^i{}_k \, B^j{}_l
$$

### 5.3 축약 (Contraction)

위 인덱스와 아래 인덱스 하나씩을 같은 문자로 놓고 합산하면 rank가 2만큼 줄어든다:

$$
T^i{}_{ij} = \sum_i T^i{}_{ij} \quad (\text{rank-}(2,1) \to \text{rank-}(1,0))
$$

대표적인 예: 대각합(trace) $T^i{}_i = \text{tr}(T)$

### 5.4 대칭화와 반대칭화

**대칭 텐서**: $T_{ij} = T_{ji}$

$$
T_{(ij)} = \frac{1}{2}(T_{ij} + T_{ji}) \quad (\text{대칭화})
$$

**반대칭 텐서**: $T_{ij} = -T_{ji}$

$$
T_{[ij]} = \frac{1}{2}(T_{ij} - T_{ji}) \quad (\text{반대칭화})
$$

임의의 rank-2 텐서는 대칭 부분과 반대칭 부분으로 분해된다:

$$
T_{ij} = T_{(ij)} + T_{[ij]}
$$

### 5.5 Python: 텐서 대수 연산

```python
import numpy as np

# === 텐서 대수 연산 ===

# rank-2 텐서 (3×3)
T = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=float)

# 대칭 부분과 반대칭 부분 분해
T_sym = 0.5 * (T + T.T)       # T_{(ij)}
T_antisym = 0.5 * (T - T.T)   # T_{[ij]}
print("원래 텐서 T:")
print(T)
print("\n대칭 부분 T_{(ij)}:")
print(T_sym)
print("\n반대칭 부분 T_{[ij]}:")
print(T_antisym)
print(f"\n복원 검증: T = T_sym + T_antisym? {np.allclose(T, T_sym + T_antisym)}")

# 텐서곱 (outer product)
A = np.array([1, 2, 3], dtype=float)
B = np.array([4, 5, 6], dtype=float)
AB_outer = np.einsum('i,j->ij', A, B)  # A^i B^j
print(f"\n텐서곱 A⊗B:\n{AB_outer}")

# 축약 (contraction)
# rank-2 텐서의 trace: T^i_i
trace_T = np.einsum('ii->', T)
print(f"\n축약 (trace): T^i_i = {trace_T}")

# rank-4 텐서에서 축약
R = np.random.rand(3, 3, 3, 3)
# R^i_{jkl} → 첫째와 셋째 인덱스 축약 → R^i_{jil} = S_{jl}
S = np.einsum('ijil->jl', R)
print(f"\nrank-4 텐서 축약 결과 (rank-2): shape = {S.shape}")
```

---

## 6. 공변 미분과 크리스토펠 기호

### 6.1 일반 미분의 문제점

곡선좌표계에서 텐서의 일반 편미분 $\partial_i A^j$는 **텐서가 아니다**. 이는 기저 벡터 자체가 점마다 변하기 때문이다. 올바른 미분 연산자가 필요하며, 이것이 **공변 미분(covariant derivative)**이다.

### 6.2 크리스토펠 기호의 정의

**크리스토펠 기호(Christoffel symbol of the second kind)** $\Gamma^k{}_{ij}$는 계량 텐서로부터 계산된다:

$$
\Gamma^k{}_{ij} = \frac{1}{2}g^{kl}\left(\frac{\partial g_{jl}}{\partial x^i} + \frac{\partial g_{il}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^l}\right)
$$

크리스토펠 기호는 텐서가 **아니다** (비균질 변환 법칙을 따름). 이는 좌표계의 "비관성" 효과를 보정하는 역할을 한다.

**대칭성**: $\Gamma^k{}_{ij} = \Gamma^k{}_{ji}$ (계량 텐서가 비틀림이 없는 경우)

### 6.3 공변 미분

**반변 벡터의 공변 미분**:

$$
\nabla_i A^j = \partial_i A^j + \Gamma^j{}_{ik} A^k
$$

**공변 벡터의 공변 미분**:

$$
\nabla_i A_j = \partial_i A_j - \Gamma^k{}_{ij} A_k
$$

**일반 텐서의 공변 미분** (위 인덱스마다 $+\Gamma$, 아래 인덱스마다 $-\Gamma$):

$$
\nabla_i T^j{}_k = \partial_i T^j{}_k + \Gamma^j{}_{il} T^l{}_k - \Gamma^l{}_{ik} T^j{}_l
$$

**계량 텐서의 공변 미분**: $\nabla_i g_{jk} = 0$ (메트릭 적합성 조건)

### 6.4 평행 이동과 측지선 방정식

벡터 $A^i$를 곡선 $x^i(\tau)$를 따라 **평행 이동(parallel transport)**하는 조건:

$$
\frac{DA^i}{D\tau} = \frac{dA^i}{d\tau} + \Gamma^i{}_{jk}\frac{dx^j}{d\tau}A^k = 0
$$

**측지선(geodesic)**은 접선 벡터 자체를 평행 이동하는 곡선이다:

$$
\frac{d^2x^k}{d\tau^2} + \Gamma^k{}_{ij}\frac{dx^i}{d\tau}\frac{dx^j}{d\tau} = 0
$$

평탄 공간에서 측지선은 직선이고, 구면 위에서는 대원(great circle)이다.

### 6.5 Python: 크리스토펠 기호와 측지선 계산

```python
import sympy as sp

# === 크리스토펠 기호 계산 함수 ===

def christoffel_symbols(g, coords):
    """
    계량 텐서로부터 크리스토펠 기호 Γ^k_{ij}를 계산한다.

    Parameters:
        g: 계량 텐서 (SymPy Matrix)
        coords: 좌표 변수 리스트
    Returns:
        Γ[k][i][j] 형태의 3차원 리스트
    """
    n = len(coords)
    g_inv = g.inv()

    Gamma = [[[sp.Integer(0) for _ in range(n)]
              for _ in range(n)]
             for _ in range(n)]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                s = sp.Integer(0)
                for l in range(n):
                    s += sp.Rational(1, 2) * g_inv[k, l] * (
                        sp.diff(g[j, l], coords[i]) +
                        sp.diff(g[i, l], coords[j]) -
                        sp.diff(g[i, j], coords[l])
                    )
                Gamma[k][i][j] = sp.simplify(s)
    return Gamma

# --- 구면 좌표에서 크리스토펠 기호 ---
r, theta, phi = sp.symbols('r theta phi', positive=True)
g_sph = sp.diag(1, r**2, r**2 * sp.sin(theta)**2)
coords_sph = [r, theta, phi]

Gamma_sph = christoffel_symbols(g_sph, coords_sph)

print("구면 좌표의 0이 아닌 크리스토펠 기호:")
names = ['r', 'θ', 'φ']
for k in range(3):
    for i in range(3):
        for j in range(i, 3):
            if Gamma_sph[k][i][j] != 0:
                print(f"  Γ^{names[k]}_{{{names[i]}{names[j]}}} = {Gamma_sph[k][i][j]}")

# --- 2차원 구면 위의 측지선 (수치 계산) ---
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def geodesic_sphere(tau, y, R_val=1.0):
    """
    단위 구면 위의 측지선 방정식.
    y = [θ, φ, dθ/dτ, dφ/dτ]
    """
    th, ph, dth, dph = y

    # 2차원 구면의 크리스토펠 기호
    # Γ^θ_{φφ} = -sinθ cosθ
    # Γ^φ_{θφ} = Γ^φ_{φθ} = cosθ/sinθ

    d2th = np.sin(th) * np.cos(th) * dph**2
    d2ph = -2.0 * np.cos(th) / np.sin(th) * dth * dph if np.sin(th) > 1e-10 else 0.0

    return [dth, dph, d2th, d2ph]

# 초기 조건: 적도에서 북동 방향으로 출발
th0, ph0 = np.pi / 2, 0.0
dth0, dph0 = -0.3, 1.0  # 북쪽으로 약간 + 동쪽으로

y0 = [th0, ph0, dth0, dph0]
sol = solve_ivp(geodesic_sphere, [0, 8], y0, max_step=0.01, dense_output=True)

# 구면 위에 측지선 표시
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 구면 그리기
u = np.linspace(0, np.pi, 40)
v = np.linspace(0, 2*np.pi, 40)
U, V = np.meshgrid(u, v)
X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)
ax.plot_surface(X, Y, Z, alpha=0.15, color='lightblue')

# 측지선 (대원)
th_geo = sol.y[0]
ph_geo = sol.y[1]
xg = np.sin(th_geo) * np.cos(ph_geo)
yg = np.sin(th_geo) * np.sin(ph_geo)
zg = np.cos(th_geo)
ax.plot(xg, yg, zg, 'r-', linewidth=2.5, label='측지선 (대원)')
ax.plot([xg[0]], [yg[0]], [zg[0]], 'go', markersize=8, label='출발점')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('구면 위의 측지선')
ax.legend()
plt.tight_layout()
plt.savefig('geodesic_sphere.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 7. 곡률 텐서 (Curvature)

### 7.1 리만 곡률 텐서

**리만 곡률 텐서(Riemann curvature tensor)** $R^l{}_{kij}$는 공변 미분의 비교환성으로 정의된다:

$$
(\nabla_i \nabla_j - \nabla_j \nabla_i)A^l = R^l{}_{kij}A^k
$$

크리스토펠 기호를 사용한 명시적 표현:

$$
R^l{}_{kij} = \partial_i \Gamma^l{}_{jk} - \partial_j \Gamma^l{}_{ik} + \Gamma^l{}_{im}\Gamma^m{}_{jk} - \Gamma^l{}_{jm}\Gamma^m{}_{ik}
$$

### 7.2 기하학적 의미

리만 곡률 텐서는 **닫힌 경로를 따라 벡터를 평행 이동했을 때 원래 벡터와 달라지는 정도**를 측정한다.

- 평탄 공간: $R^l{}_{kij} = 0$ (평행 이동이 경로에 무관)
- 곡률이 있는 공간: $R^l{}_{kij} \ne 0$ (경로 의존적)

### 7.3 리치 텐서, 스칼라 곡률, 아인슈타인 텐서

**리치 텐서(Ricci tensor)**: 리만 텐서의 축약

$$
R_{ij} = R^k{}_{ikj}
$$

**스칼라 곡률(Ricci scalar)**: 리치 텐서의 축약

$$
R = g^{ij}R_{ij}
$$

**2차원 곡면의 가우스 곡률(Gaussian curvature)**:

$$
K = \frac{R}{2} \quad (\text{2차원에서})
$$

- 구면 ($r = a$): $K = 1/a^2 > 0$
- 평면: $K = 0$
- 쌍곡면: $K < 0$

**아인슈타인 텐서(Einstein tensor)**:

$$
G_{ij} = R_{ij} - \frac{1}{2}g_{ij}R
$$

$G_{ij}$는 $\nabla_i G^{ij} = 0$ (비안키 항등식에 의해)을 만족하며, 이는 에너지-운동량 보존과 직결된다.

### 7.4 Python: 곡률 계산

```python
import sympy as sp

# === 리만 곡률 텐서 계산 함수 ===

def riemann_tensor(Gamma, coords):
    """
    크리스토펠 기호로부터 리만 곡률 텐서 R^l_{kij}를 계산한다.
    """
    n = len(coords)
    R = [[[[sp.Integer(0) for _ in range(n)]
           for _ in range(n)]
          for _ in range(n)]
         for _ in range(n)]

    for l in range(n):
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    # ∂_i Γ^l_{jk} - ∂_j Γ^l_{ik}
                    term = sp.diff(Gamma[l][j][k], coords[i]) - \
                           sp.diff(Gamma[l][i][k], coords[j])
                    # + Γ^l_{im} Γ^m_{jk} - Γ^l_{jm} Γ^m_{ik}
                    for m in range(n):
                        term += Gamma[l][i][m] * Gamma[m][j][k] - \
                                Gamma[l][j][m] * Gamma[m][i][k]
                    R[l][k][i][j] = sp.simplify(term)
    return R

def ricci_tensor(R_riem, n):
    """리치 텐서 R_{ij} = R^k_{ikj}"""
    Ric = sp.zeros(n, n)
    for i in range(n):
        for j in range(n):
            s = sp.Integer(0)
            for k in range(n):
                s += R_riem[k][i][k][j]
            Ric[i, j] = sp.simplify(s)
    return Ric

def scalar_curvature(Ric, g_inv, n):
    """스칼라 곡률 R = g^{ij} R_{ij}"""
    R_scalar = sp.Integer(0)
    for i in range(n):
        for j in range(n):
            R_scalar += g_inv[i, j] * Ric[i, j]
    return sp.simplify(R_scalar)

# --- 2차원 구면 (r = a)의 가우스 곡률 계산 ---
theta, phi = sp.symbols('theta phi', positive=True)
a = sp.Symbol('a', positive=True)

g_sphere = sp.diag(a**2, a**2 * sp.sin(theta)**2)
coords_sphere = [theta, phi]

# 크리스토펠 기호
Gamma_sp = christoffel_symbols(g_sphere, coords_sphere)
print("2차원 구면의 0이 아닌 크리스토펠 기호:")
snames = ['θ', 'φ']
for k in range(2):
    for i in range(2):
        for j in range(i, 2):
            if Gamma_sp[k][i][j] != 0:
                print(f"  Γ^{snames[k]}_{{{snames[i]}{snames[j]}}} = {Gamma_sp[k][i][j]}")

# 리만 텐서
R_riem_sp = riemann_tensor(Gamma_sp, coords_sphere)

# 리치 텐서
Ric_sp = ricci_tensor(R_riem_sp, 2)
print(f"\n리치 텐서 R_{{ij}}:")
sp.pprint(Ric_sp)

# 스칼라 곡률
g_sp_inv = g_sphere.inv()
R_sc = scalar_curvature(Ric_sp, g_sp_inv, 2)
print(f"\n스칼라 곡률 R = {R_sc}")

# 가우스 곡률 (2차원에서 K = R/2)
K = sp.simplify(R_sc / 2)
print(f"가우스 곡률 K = R/2 = {K}")
# 출력: K = 1/a^2 (양의 일정한 곡률 → 구면)
```

---

## 8. 물리적 응용

### 8.1 연속체 역학: 응력 텐서

**코시 응력 텐서(Cauchy stress tensor)** $\sigma_{ij}$는 연속체 내부의 임의의 면에 작용하는 응력을 기술한다. 면의 법선 방향이 $\hat{n}$일 때, 그 면에 작용하는 단위 면적당 힘(견인력, traction)은:

$$
t_i = \sigma_{ij} n_j
$$

$\sigma_{ij}$는 대칭 텐서($\sigma_{ij} = \sigma_{ji}$, 각운동량 보존에 의해)이며, 대각 성분은 법선 응력(normal stress), 비대각 성분은 전단 응력(shear stress)이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# === 2D 응력 텐서의 모어 원 (Mohr's Circle) ===
# 응력 텐서 σ = [[σ_xx, τ_xy], [τ_xy, σ_yy]]
sigma_xx, sigma_yy, tau_xy = 50.0, 20.0, 15.0
sigma = np.array([[sigma_xx, tau_xy],
                   [tau_xy, sigma_yy]])

# 주응력 (고유값)
eigenvalues, eigenvectors = np.linalg.eigh(sigma)
sigma_1 = eigenvalues[1]  # 최대 주응력
sigma_2 = eigenvalues[0]  # 최소 주응력
print(f"주응력: σ₁ = {sigma_1:.2f} MPa, σ₂ = {sigma_2:.2f} MPa")
print(f"주응력 방향:\n{eigenvectors}")

# 모어 원 그리기
center = (sigma_1 + sigma_2) / 2
radius = (sigma_1 - sigma_2) / 2

fig, ax = plt.subplots(figsize=(8, 6))
circle = plt.Circle((center, 0), radius, fill=False, color='blue', linewidth=2)
ax.add_patch(circle)

# 원래 응력 상태 표시
ax.plot(sigma_xx, tau_xy, 'ro', markersize=8, label=f'(σ_xx, τ_xy) = ({sigma_xx}, {tau_xy})')
ax.plot(sigma_yy, -tau_xy, 'go', markersize=8, label=f'(σ_yy, -τ_xy) = ({sigma_yy}, {-tau_xy})')
ax.plot([sigma_1, sigma_2], [0, 0], 'k^', markersize=10, label=f'주응력 σ₁={sigma_1:.1f}, σ₂={sigma_2:.1f}')

ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('법선 응력 σ (MPa)')
ax.set_ylabel('전단 응력 τ (MPa)')
ax.set_title('모어 원 (Mohr\'s Circle)')
ax.set_aspect('equal')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mohr_circle.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 8.2 전자기학: 전자기장 텐서 $F_{\mu\nu}$

특수 상대론적 전자기학에서, 전기장 $\mathbf{E}$와 자기장 $\mathbf{B}$는 하나의 **반대칭 rank-2 텐서** $F_{\mu\nu}$로 통합된다:

$$
F_{\mu\nu} = \begin{pmatrix} 0 & -E_x/c & -E_y/c & -E_z/c \\ E_x/c & 0 & -B_z & B_y \\ E_y/c & B_z & 0 & -B_x \\ E_z/c & -B_y & B_x & 0 \end{pmatrix}
$$

맥스웰 방정식의 텐서 형태:

$$
\partial_\mu F^{\mu\nu} = \mu_0 J^\nu \quad (\text{비균질 방정식: 가우스 법칙 + 앙페르-맥스웰 법칙})
$$

$$
\partial_{[\lambda} F_{\mu\nu]} = 0 \quad (\text{균질 방정식: 패러데이 법칙 + 자기 가우스 법칙})
$$

이 표현은 로렌츠 변환에서의 공변성을 자명하게 만든다.

```python
import numpy as np

# === 전자기장 텐서 구성 및 로렌츠 변환 ===

c = 1.0  # 자연 단위 (c = 1)

def em_field_tensor(E, B):
    """전기장 E와 자기장 B로부터 전자기장 텐서 F_μν를 구성한다."""
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    F = np.array([
        [0,      -Ex/c,  -Ey/c,  -Ez/c],
        [Ex/c,    0,     -Bz,     By  ],
        [Ey/c,    Bz,     0,     -Bx  ],
        [Ez/c,   -By,     Bx,     0   ]
    ])
    return F

# x 방향 전기장만 있는 경우
E = np.array([1.0, 0.0, 0.0])
B = np.array([0.0, 0.0, 0.0])
F = em_field_tensor(E, B)
print("전자기장 텐서 F_μν (순수 전기장):")
print(F)

# 로렌츠 변환 (x 방향, 속도 v = 0.6c)
v = 0.6
gamma = 1.0 / np.sqrt(1 - v**2)
beta = v

# 로렌츠 변환 행렬 Λ^μ'_ν
Lambda = np.array([
    [gamma,      -gamma*beta, 0, 0],
    [-gamma*beta, gamma,      0, 0],
    [0,           0,          1, 0],
    [0,           0,          0, 1]
])

# 텐서 변환: F'^{μν} = Λ^μ_α Λ^ν_β F^{αβ}
# 먼저 F^{μν} = η^{μα} η^{νβ} F_{αβ} (민코프스키 계량으로 인덱스 올림)
eta = np.diag([-1, 1, 1, 1])  # 민코프스키 계량 (-,+,+,+)
F_up = eta @ F @ eta           # F^{μν}

F_prime_up = Lambda @ F_up @ Lambda.T
F_prime = eta @ F_prime_up @ eta  # F'_{μν}로 내림

print(f"\n로렌츠 변환 후 (v = {v}c):")
print(f"F'_μν:")
print(np.round(F_prime, 4))

# 변환된 전기장과 자기장 추출
E_prime = np.array([F_prime[0, 1], F_prime[0, 2], F_prime[0, 3]]) * (-c)
B_prime = np.array([F_prime[2, 3], F_prime[3, 1], F_prime[1, 2]])
print(f"\n변환된 전기장: E' = {np.round(E_prime, 4)}")
print(f"변환된 자기장: B' = {np.round(B_prime, 4)}")
print("순수 전기장이 로렌츠 변환에 의해 자기장 성분을 획득함!")

# 로렌츠 불변량 검증
inv1 = -0.5 * np.einsum('ij,ij->', F, F)   # F_{μν}F^{μν}/2
inv1_prime = -0.5 * np.einsum('ij,ij->', F_prime, F_prime)
print(f"\n로렌츠 불변량 F_μν F^μν: 원래 = {inv1:.4f}, 변환 후 = {inv1_prime:.4f}")
```

### 8.3 일반 상대론: 아인슈타인 장방정식

일반 상대론의 핵심인 **아인슈타인 장방정식(Einstein field equations)**은 시공간의 기하학(곡률)과 물질-에너지 분포를 연결한다:

$$
G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}
$$

여기서:
- $G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R$: 아인슈타인 텐서 (기하학)
- $\Lambda$: 우주상수
- $T_{\mu\nu}$: 에너지-운동량 텐서 (물질)

**슈바르츠실트 계량(Schwarzschild metric)**: 구대칭 진공 해

$$
ds^2 = -\left(1 - \frac{r_s}{r}\right)c^2 dt^2 + \left(1 - \frac{r_s}{r}\right)^{-1}dr^2 + r^2 d\theta^2 + r^2\sin^2\theta \, d\phi^2
$$

여기서 $r_s = 2GM/c^2$는 슈바르츠실트 반지름이다.

```python
import sympy as sp

# === 슈바르츠실트 계량의 크리스토펠 기호 계산 ===
t, r, theta, phi = sp.symbols('t r theta phi')
r_s, c_sym = sp.symbols('r_s c', positive=True)

# 슈바르츠실트 계량 텐서 (대각)
f = 1 - r_s / r  # f(r) = 1 - r_s/r

g_schw = sp.diag(
    -f * c_sym**2,   # g_{tt}
    1 / f,           # g_{rr}
    r**2,            # g_{θθ}
    r**2 * sp.sin(theta)**2  # g_{φφ}
)
coords_schw = [t, r, theta, phi]

print("슈바르츠실트 계량 텐서:")
sp.pprint(g_schw)

# 크리스토펠 기호 계산 (시간 소요 가능)
Gamma_schw = christoffel_symbols(g_schw, coords_schw)

print("\n슈바르츠실트 계량의 0이 아닌 크리스토펠 기호:")
coord_names = ['t', 'r', 'θ', 'φ']
for k in range(4):
    for i in range(4):
        for j in range(i, 4):
            val = Gamma_schw[k][i][j]
            if val != 0:
                print(f"  Γ^{coord_names[k]}_{{{coord_names[i]}{coord_names[j]}}} = {val}")
```

---

## 연습 문제

### 문제 1: 텐서 변환

2차원에서 좌표 변환 $x' = x\cosh\alpha + y\sinh\alpha$, $y' = x\sinh\alpha + y\cosh\alpha$ (로렌츠 부스트)가 주어졌을 때:

(a) 변환 행렬 $\frac{\partial x'^i}{\partial x^j}$를 구하라.

(b) 벡터 $A^i = (3, 4)$를 새 좌표계로 변환하라 ($\alpha = 0.5$).

(c) 텐서 $T^{ij} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$를 새 좌표계로 변환하라.

### 문제 2: 아인슈타인 합산 규약

다음 텐서 표현을 합산 기호 없이 아인슈타인 합산 규약으로 쓰고, 3차원에서 전개하라:

(a) 벡터의 내적 $\sum_i A^i B_i$

(b) 행렬 곱 $\sum_k M^i{}_k N^k{}_j$

(c) $\varepsilon_{ijk}\varepsilon_{imn}$을 $\delta$-항등식으로 환원하라.

### 문제 3: 계량 텐서와 인덱스 올림/내림

2차원 극좌표 $(r, \theta)$에서 $ds^2 = dr^2 + r^2 d\theta^2$이다.

(a) 계량 텐서 $g_{ij}$와 역계량 텐서 $g^{ij}$를 구하라.

(b) 반변 벡터 $A^i = (A^r, A^\theta) = (2, 1/r)$의 공변 성분 $A_i = g_{ij}A^j$를 구하라.

(c) $A^i A_i$를 계산하여 벡터 크기의 제곱이 좌표에 무관한 불변량임을 확인하라.

### 문제 4: 크리스토펠 기호

2차원 극좌표의 계량 텐서 $g = \text{diag}(1, r^2)$에 대해:

(a) 모든 크리스토펠 기호 $\Gamma^k{}_{ij}$를 계산하라.

(b) 벡터장 $A^r = \cos\theta$, $A^\theta = -\sin\theta / r$의 공변 발산 $\nabla_i A^i$를 계산하라.

(c) 이 결과가 직교 좌표에서의 발산과 일치함을 확인하라.

### 문제 5: 측지선

2차원 쌍곡면(pseudosphere, $K = -1$)의 계량이 $ds^2 = du^2 + e^{-2u}dv^2$로 주어진다.

(a) 크리스토펠 기호를 구하라.

(b) 가우스 곡률 $K$를 계산하여 $K = -1$임을 확인하라.

(c) 측지선 방정식을 쓰고, $u = \text{const}$ 곡선이 측지선인지 판별하라.

### 문제 6: 곡률 텐서

토러스(torus) 표면의 계량이 매개변수 $(\theta, \phi)$로 다음과 같이 주어진다:

$$
ds^2 = a^2 d\theta^2 + (R + a\cos\theta)^2 d\phi^2
$$

여기서 $R$은 토러스의 큰 반지름, $a$는 작은 반지름이다.

(a) 가우스 곡률 $K(\theta)$를 구하라.

(b) $K > 0$, $K = 0$, $K < 0$인 영역을 $\theta$ 값으로 구분하라.

(c) 가우스-보네 정리 $\int K \, dA = 2\pi\chi$에서 $\chi = 0$ (토러스의 오일러 수)임을 확인하라.

### 문제 7: 전자기장 텐서

전기장 $\mathbf{E} = E_0 \hat{x}$, 자기장 $\mathbf{B} = B_0 \hat{z}$가 주어졌을 때:

(a) 전자기장 텐서 $F_{\mu\nu}$를 구성하라.

(b) 로렌츠 불변량 $F_{\mu\nu}F^{\mu\nu}$와 $\frac{1}{2}\varepsilon^{\mu\nu\rho\sigma}F_{\mu\nu}F_{\rho\sigma}$를 계산하라.

(c) $x$-방향 로렌츠 부스트($v = 0.8c$)로 변환한 후의 $\mathbf{E}'$, $\mathbf{B}'$를 구하라.

### 문제 8: 응력 텐서의 주축

3차원 응력 텐서가 다음과 같이 주어진다:

$$
\sigma_{ij} = \begin{pmatrix} 100 & 30 & 0 \\ 30 & 50 & 20 \\ 0 & 20 & 80 \end{pmatrix} \text{ (MPa)}
$$

(a) 주응력(principal stresses)과 주응력 방향을 구하라 (고유값/고유벡터).

(b) 최대 전단 응력을 구하라.

(c) 폰 미제스 응력(von Mises stress)을 계산하라: $\sigma_v = \sqrt{\frac{1}{2}[(\sigma_1-\sigma_2)^2 + (\sigma_2-\sigma_3)^2 + (\sigma_3-\sigma_1)^2]}$.

---

## 심화 학습

### 미분 형식 (Differential Forms)

텐서 해석의 현대적 언어인 **미분 형식(differential forms)**은 반대칭 공변 텐서를 체계적으로 다룬다. 외대수(exterior algebra)와 외미분(exterior derivative) $d$를 사용하여:

- 0-형식 = 스칼라 함수
- 1-형식 = $\omega = A_i dx^i$
- 2-형식 = $F = \frac{1}{2}F_{ij}dx^i \wedge dx^j$

맥스웰 방정식은 미분 형식으로 극도로 간결해진다: $dF = 0$, $d{*F} = J$

### 리 미분 (Lie Derivative)

**리 미분(Lie derivative)** $\mathcal{L}_X T$는 벡터장 $X$를 따라 텐서 $T$의 변화를 좌표에 무관하게 정의한다. 이는 대칭과 보존 법칙(킬링 벡터장)을 다루는 핵심 도구이다.

### 올다발과 게이지 이론 (Fiber Bundles and Gauge Theory)

양-밀스 이론(Yang-Mills theory)에서 게이지 결합(gauge connection)은 크리스토펠 기호의 일반화이며, 곡률은 장 강도 텐서(field strength tensor)에 대응한다. 이는 표준 모형의 수학적 기초를 이룬다.

### 계산 도구

| 도구 | 언어 | 설명 |
|------|------|------|
| `sympy.diffgeom` | Python | 미분 기하 계산 |
| `einsteinpy` | Python | 일반 상대론 텐서 계산 |
| `xAct` / `xTensor` | Mathematica | 전문적 텐서 대수 시스템 |
| `SageManifolds` | SageMath | 미분 다양체 계산 |
| `cadabra2` | Python/C++ | 장론(field theory) 텐서 계산 |

### 추천 참고 자료

1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 10. Wiley.
2. **Carroll, S. M.** (2019). *Spacetime and Geometry*, 2nd ed. Cambridge University Press.
   - 일반 상대론에서의 텐서 해석을 위한 최고의 교재
3. **Schutz, B.** (2009). *A First Course in General Relativity*, 2nd ed. Cambridge University Press.
4. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapter 3. Academic Press.
5. **Nakahara, M.** (2003). *Geometry, Topology and Physics*, 2nd ed. CRC Press.
   - 미분 형식, 올다발, 게이지 이론의 수학적 기초

---

**이전**: [17. 변분법](17_Calculus_of_Variations.md)
**다음**: 과정 완료! [00. 개요](00_Overview.md)로 돌아가기
