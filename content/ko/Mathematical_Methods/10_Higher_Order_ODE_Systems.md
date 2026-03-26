# 10. 고차 ODE와 연립계 (Higher-Order ODE and Systems)

## 학습 목표

- **n차 상수계수 선형 ODE**의 특성방정식을 세우고 일반해를 구할 수 있다
- **미정계수법**과 **매개변수 변환법**으로 비제차 ODE의 특수해를 구할 수 있다
- 연립 ODE를 **벡터-행렬 형태**로 표현하고, **고유값/고유벡터** 및 **행렬 지수**로 풀 수 있다
- **위상 평면(phase plane)**에서 평형점을 분류하고 **안정성**을 판별할 수 있다
- **비선형 시스템**의 선형화 기법과 대표적인 물리/생물 모델을 분석할 수 있다
- **결합 진동자**의 정규 모드(normal modes)를 구하고, 라그랑주 역학과의 연결을 이해한다

---

## 1. 고차 선형 ODE

### 1.1 n차 상수계수 ODE

n차 상수계수 선형 ODE의 일반적 형태는 다음과 같다:

$$a_n y^{(n)} + a_{n-1} y^{(n-1)} + \cdots + a_1 y' + a_0 y = f(x)$$

여기서 $a_0, a_1, \ldots, a_n$은 상수이며, $f(x) = 0$이면 **제차(homogeneous)**, $f(x) \neq 0$이면 **비제차(non-homogeneous)** 방정식이다.

**핵심 원리 - 중첩(superposition)**: 제차 방정식의 해는 **선형 공간**을 이루므로, $n$개의 일차독립인 해 $y_1, y_2, \ldots, y_n$의 **일반해**는:

$$y_h = c_1 y_1 + c_2 y_2 + \cdots + c_n y_n$$

비제차 방정식의 일반해는:

$$y = y_h + y_p$$

여기서 $y_p$는 하나의 특수해(particular solution)이다.

### 1.2 특성방정식과 일반해

제차 방정식의 해를 $y = e^{rx}$ 형태로 가정하면, **특성방정식(characteristic equation)**을 얻는다:

$$a_n r^n + a_{n-1} r^{n-1} + \cdots + a_1 r + a_0 = 0$$

특성근의 종류에 따라 일반해가 결정된다:

| 특성근의 종류 | 해의 형태 |
|---|---|
| 서로 다른 실근 $r_1, r_2, \ldots, r_n$ | $c_1 e^{r_1 x} + c_2 e^{r_2 x} + \cdots$ |
| 중복근 $r$ (중복도 $m$) | $(c_1 + c_2 x + \cdots + c_m x^{m-1}) e^{rx}$ |
| 복소근 $\alpha \pm i\beta$ | $e^{\alpha x}(c_1 \cos\beta x + c_2 \sin\beta x)$ |
| 복소 중복근 ($\alpha \pm i\beta$, 중복도 $m$) | $e^{\alpha x}\sum_{k=0}^{m-1} x^k (a_k \cos\beta x + b_k \sin\beta x)$ |

**예제**: 4차 ODE $y^{(4)} - 5y'' + 4y = 0$

```python
import numpy as np
import sympy as sp

# --- SymPy로 특성방정식 풀기 ---
r = sp.Symbol('r')
char_eq = r**4 - 5*r**2 + 4  # 특성방정식
roots = sp.solve(char_eq, r)
print(f"특성근: {roots}")
# 출력: 특성근: [-2, -1, 1, 2]

# --- 일반해 구하기 ---
x = sp.Symbol('x')
y = sp.Function('y')
ode = sp.Eq(y(x).diff(x, 4) - 5*y(x).diff(x, 2) + 4*y(x), 0)
general_sol = sp.dsolve(ode, y(x))
print(f"일반해: {general_sol}")
# 출력: y(x) = C1*exp(-2*x) + C2*exp(-x) + C3*exp(x) + C4*exp(2*x)
```

**중복근이 있는 예제**: $y''' - 3y'' + 3y' - y = 0$

특성방정식 $r^3 - 3r^2 + 3r - 1 = (r-1)^3 = 0$이므로 $r = 1$ (중복도 3)

$$y = (c_1 + c_2 x + c_3 x^2) e^x$$

```python
# 중복근이 있는 3차 ODE
ode2 = sp.Eq(y(x).diff(x, 3) - 3*y(x).diff(x, 2) + 3*y(x).diff(x) - y(x), 0)
sol2 = sp.dsolve(ode2, y(x))
print(f"중복근 일반해: {sol2}")
# 출력: y(x) = (C1 + C2*x + C3*x**2)*exp(x)
```

### 1.3 비제차 문제의 특수해

#### 미정계수법 (Method of Undetermined Coefficients)

$f(x)$가 다항식, 지수함수, 삼각함수 또는 이들의 곱으로 이루어진 경우 적용 가능하다.

| $f(x)$의 형태 | $y_p$의 추정 형태 |
|---|---|
| $P_n(x)$ (n차 다항식) | $A_n x^n + A_{n-1} x^{n-1} + \cdots + A_0$ |
| $e^{\alpha x}$ | $A e^{\alpha x}$ |
| $\cos\beta x$ 또는 $\sin\beta x$ | $A \cos\beta x + B \sin\beta x$ |
| $e^{\alpha x} P_n(x)$ | $e^{\alpha x}(A_n x^n + \cdots + A_0)$ |

$\alpha$가 특성근이면 $x^s$를 곱한다 ($s$는 $\alpha$의 중복도).

#### 매개변수 변환법 (Variation of Parameters)

$f(x)$가 어떤 형태든 적용 가능한 일반적 방법이다. 2차 ODE $y'' + p(x)y' + q(x)y = f(x)$에 대해:

$$y_p = -y_1 \int \frac{y_2 f}{W} dx + y_2 \int \frac{y_1 f}{W} dx$$

여기서 $W = y_1 y_2' - y_2 y_1'$은 **론스키안(Wronskian)**이다.

```python
# --- 비제차 ODE: y'' + y = sec(x) ---
# 미정계수법으로는 풀 수 없음 → 매개변수 변환법 사용
x = sp.Symbol('x')
y = sp.Function('y')

ode_nh = sp.Eq(y(x).diff(x, 2) + y(x), sp.sec(x))
sol_nh = sp.dsolve(ode_nh, y(x))
print(f"매개변수 변환법 결과: {sol_nh}")

# 론스키안 직접 계산
y1 = sp.cos(x)
y2 = sp.sin(x)
W = y1 * sp.diff(y2, x) - y2 * sp.diff(y1, x)
print(f"론스키안: W = {sp.simplify(W)}")
# 출력: W = 1

# 특수해 계산
f_x = sp.sec(x)
yp = -y1 * sp.integrate(y2 * f_x / W, x) + y2 * sp.integrate(y1 * f_x / W, x)
yp_simplified = sp.simplify(yp)
print(f"특수해: y_p = {yp_simplified}")
```

---

## 2. 연립 ODE (Systems of ODEs)

### 2.1 벡터-행렬 표현

연립 1차 ODE는 벡터-행렬 형태로 간결하게 표현된다:

$$\frac{d\mathbf{x}}{dt} = A\mathbf{x} + \mathbf{f}(t)$$

여기서 $\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}$, $A$는 $n \times n$ 계수 행렬이다.

**중요**: 모든 **n차 ODE**는 연립 1차 ODE로 변환할 수 있다. 예를 들어, $y'' + 3y' + 2y = 0$은:

$$x_1 = y, \quad x_2 = y'$$

$$\frac{d}{dt}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 0 & 1 \\ -2 & -3 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- 2차 ODE → 연립 1차 ODE 변환 및 수치 풀이 ---
# y'' + 3y' + 2y = 0, y(0) = 1, y'(0) = 0
def system(t, x):
    """x[0] = y, x[1] = y'"""
    return [x[1], -2*x[0] - 3*x[1]]

t_span = (0, 5)
x0 = [1.0, 0.0]
sol = solve_ivp(system, t_span, x0, t_eval=np.linspace(0, 5, 200), method='RK45')

# 해석해와 비교
t_exact = np.linspace(0, 5, 200)
# 특성근: r = -1, -2 → y = 2e^{-t} - e^{-2t}
y_exact = 2*np.exp(-t_exact) - np.exp(-2*t_exact)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(sol.t, sol.y[0], 'b-', label='수치해 (RK45)')
axes[0].plot(t_exact, y_exact, 'r--', label='해석해')
axes[0].set_xlabel('t')
axes[0].set_ylabel('y(t)')
axes[0].set_title('해 비교')
axes[0].legend()

axes[1].plot(sol.y[0], sol.y[1], 'b-')
axes[1].set_xlabel('y')
axes[1].set_ylabel("y'")
axes[1].set_title('위상 평면 궤적')
axes[1].plot(x0[0], x0[1], 'ro', markersize=8, label='초기값')
axes[1].legend()

plt.tight_layout()
plt.savefig('second_order_ode_solution.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2.2 고유값/고유벡터를 이용한 풀이

제차 연립 $\mathbf{x}' = A\mathbf{x}$의 해는 $\mathbf{x} = \mathbf{v} e^{\lambda t}$로 가정하면:

$$A\mathbf{v} = \lambda \mathbf{v}$$

즉, $A$의 **고유값(eigenvalue)** $\lambda$와 **고유벡터(eigenvector)** $\mathbf{v}$를 구하는 문제로 귀결된다.

**경우 1: 서로 다른 실수 고유값** $\lambda_1, \lambda_2$

$$\mathbf{x}(t) = c_1 \mathbf{v}_1 e^{\lambda_1 t} + c_2 \mathbf{v}_2 e^{\lambda_2 t}$$

**경우 2: 복소 고유값** $\lambda = \alpha \pm i\beta$

$$\mathbf{x}(t) = e^{\alpha t}\left[c_1(\mathbf{a}\cos\beta t - \mathbf{b}\sin\beta t) + c_2(\mathbf{a}\sin\beta t + \mathbf{b}\cos\beta t)\right]$$

여기서 $\mathbf{v} = \mathbf{a} + i\mathbf{b}$이다.

**경우 3: 중복 고유값** ($\lambda$, 중복도 2, 고유벡터 1개) - 일반화된 고유벡터 $\mathbf{w}$ 필요:

$$(A - \lambda I)\mathbf{w} = \mathbf{v}$$

$$\mathbf{x}(t) = c_1 \mathbf{v} e^{\lambda t} + c_2 (\mathbf{v} t + \mathbf{w}) e^{\lambda t}$$

```python
import numpy as np
from scipy.linalg import eig

# --- 연립 ODE의 고유값/고유벡터 풀이 ---
# x' = Ax, A = [[1, 1], [4, -2]]
A = np.array([[1, 1], [4, -2]])
eigenvalues, eigenvectors = eig(A)

print("고유값:", eigenvalues)
print("고유벡터 (열벡터):")
print(eigenvectors)
# 고유값: [2, -3]
# 고유벡터: v1 = [1, 1], v2 = [1, -4] (정규화됨)

# 일반해 구성
t = np.linspace(0, 3, 200)
# 초기조건 x(0) = [3, 2]에서 c1, c2 결정
x0 = np.array([3, 2])
# c1*v1 + c2*v2 = x0 → [c1, c2] = V^{-1} x0
V = eigenvectors
c = np.linalg.solve(V, x0)
print(f"계수: c1 = {c[0]:.4f}, c2 = {c[1]:.4f}")

# 해석해 계산
x1_sol = c[0] * V[0, 0] * np.exp(eigenvalues[0].real * t) + \
         c[1] * V[0, 1] * np.exp(eigenvalues[1].real * t)
x2_sol = c[0] * V[1, 0] * np.exp(eigenvalues[0].real * t) + \
         c[1] * V[1, 1] * np.exp(eigenvalues[1].real * t)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 시간에 따른 해
axes[0].plot(t, x1_sol.real, 'b-', label='$x_1(t)$')
axes[0].plot(t, x2_sol.real, 'r-', label='$x_2(t)$')
axes[0].set_xlabel('t')
axes[0].set_title('연립 ODE의 해')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 위상 평면
axes[1].plot(x1_sol.real, x2_sol.real, 'b-', linewidth=2)
axes[1].plot(x0[0], x0[1], 'ro', markersize=8, label='초기값')
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
axes[1].set_title('위상 평면 궤적')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('system_ode_eigenvalue.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2.3 행렬 지수 (Matrix Exponential)

연립 ODE $\mathbf{x}' = A\mathbf{x}$, $\mathbf{x}(0) = \mathbf{x}_0$의 해는 다음과 같이 **행렬 지수(matrix exponential)**로 표현된다:

$$\mathbf{x}(t) = e^{At} \mathbf{x}_0$$

행렬 지수는 스칼라 지수함수의 급수 정의를 행렬로 확장한 것이다:

$$e^{At} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots = \sum_{k=0}^{\infty} \frac{(At)^k}{k!}$$

**성질:**
- $e^{0} = I$ (항등 행렬)
- $\frac{d}{dt} e^{At} = A e^{At}$
- $A$가 대각화 가능하면: $A = PDP^{-1}$ → $e^{At} = P e^{Dt} P^{-1}$

```python
from scipy.linalg import expm

# --- 행렬 지수를 이용한 연립 ODE 풀이 ---
A = np.array([[1, 1], [4, -2]])
x0 = np.array([3, 2])

t_vals = np.linspace(0, 3, 200)
solutions = np.array([expm(A * t) @ x0 for t in t_vals])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_vals, solutions[:, 0], 'b-', linewidth=2, label='$x_1(t)$')
ax.plot(t_vals, solutions[:, 1], 'r-', linewidth=2, label='$x_2(t)$')
ax.set_xlabel('t', fontsize=12)
ax.set_ylabel('x(t)', fontsize=12)
ax.set_title('행렬 지수를 이용한 해', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('matrix_exponential.png', dpi=150, bbox_inches='tight')
plt.show()

# --- SymPy로 심볼릭 행렬 지수 계산 ---
t_sym = sp.Symbol('t')
A_sym = sp.Matrix([[1, 1], [4, -2]])
exp_At = sp.exp(A_sym * t_sym)  # 심볼릭 행렬 지수
print("e^{At} =")
sp.pprint(exp_At)
```

**비제차 연립 ODE** $\mathbf{x}' = A\mathbf{x} + \mathbf{f}(t)$의 해:

$$\mathbf{x}(t) = e^{At}\mathbf{x}_0 + \int_0^t e^{A(t-s)} \mathbf{f}(s) \, ds$$

이것은 **듀아멜 적분(Duhamel's integral)** 또는 **변수 변환 공식(variation of constants formula)**으로 불린다.

---

## 3. 위상 평면 분석 (Phase Plane Analysis)

2차원 자율 연립 $\mathbf{x}' = A\mathbf{x}$에서 $A$의 고유값이 시스템의 정성적 거동을 결정한다.

### 3.1 평형점과 분류 (노드, 안장점, 소용돌이, 중심)

**평형점(equilibrium point)** $\mathbf{x}^*$은 $A\mathbf{x}^* = \mathbf{0}$을 만족하는 점이다. $A$가 가역이면 원점 $\mathbf{x}^* = \mathbf{0}$이 유일한 평형점이다.

$A$의 고유값 $\lambda_1, \lambda_2$에 따른 분류:

| 고유값의 종류 | 평형점 유형 | 안정성 |
|---|---|---|
| $\lambda_1 < \lambda_2 < 0$ (실수, 음수) | **안정 노드** (stable node) | 점근 안정 |
| $\lambda_1 > \lambda_2 > 0$ (실수, 양수) | **불안정 노드** (unstable node) | 불안정 |
| $\lambda_1 < 0 < \lambda_2$ (실수, 이부호) | **안장점** (saddle point) | 불안정 |
| $\alpha \pm i\beta$, $\alpha < 0$ | **안정 소용돌이** (stable spiral) | 점근 안정 |
| $\alpha \pm i\beta$, $\alpha > 0$ | **불안정 소용돌이** (unstable spiral) | 불안정 |
| $\pm i\beta$ (순허수) | **중심** (center) | 안정 (점근 안정이 아님) |

**트레이스-행렬식 평면**: $\tau = \text{tr}(A) = \lambda_1 + \lambda_2$, $\Delta = \det(A) = \lambda_1 \lambda_2$를 이용하면:
- $\Delta < 0$: 안장점
- $\Delta > 0$, $\tau < 0$: 안정 (노드 또는 소용돌이)
- $\Delta > 0$, $\tau > 0$: 불안정 (노드 또는 소용돌이)
- $\Delta > 0$, $\tau = 0$: 중심
- $\tau^2 - 4\Delta > 0$: 노드, $\tau^2 - 4\Delta < 0$: 소용돌이

### 3.2 안정성 판별

**정의**: 평형점 $\mathbf{x}^*$은
- **안정(stable)**: 모든 $\epsilon > 0$에 대해, $\|\mathbf{x}(0) - \mathbf{x}^*\| < \delta$이면 모든 $t > 0$에서 $\|\mathbf{x}(t) - \mathbf{x}^*\| < \epsilon$
- **점근 안정(asymptotically stable)**: 안정이고, $t \to \infty$일 때 $\mathbf{x}(t) \to \mathbf{x}^*$
- **불안정(unstable)**: 안정이 아닌 경우

**선형 시스템의 안정성 판별 기준**:
- 모든 고유값의 실수부가 음수 → **점근 안정**
- 하나라도 실수부가 양수인 고유값 → **불안정**
- 모든 고유값의 실수부가 0 이하이고, 0인 고유값이 있으면 → 추가 분석 필요

### 3.3 위상 초상화 (Phase Portrait) 그리기

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def plot_phase_portrait(A, title, ax, xlim=(-3, 3), ylim=(-3, 3)):
    """2D 선형 시스템의 위상 초상화를 그린다."""
    # 벡터장 (streamplot)
    x1 = np.linspace(xlim[0], xlim[1], 20)
    x2 = np.linspace(ylim[0], ylim[1], 20)
    X1, X2 = np.meshgrid(x1, x2)
    U = A[0, 0] * X1 + A[0, 1] * X2
    V = A[1, 0] * X1 + A[1, 1] * X2

    ax.streamplot(X1, X2, U, V, color='steelblue', density=1.5, linewidth=0.8,
                  arrowsize=1.2)

    # 여러 초기조건에서 궤적 추가
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        r0 = 2.5
        ic = [r0 * np.cos(angle), r0 * np.sin(angle)]

        def rhs(t, x):
            return A @ x

        sol = solve_ivp(rhs, [0, 10], ic, t_eval=np.linspace(0, 10, 500),
                        method='RK45')
        ax.plot(sol.y[0], sol.y[1], 'b-', alpha=0.4, linewidth=0.8)

    # 고유값/고유벡터
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigval_str = ", ".join([f"{ev:.2f}" for ev in eigenvalues])
    ax.set_title(f"{title}\n$\\lambda = {eigval_str}$", fontsize=11)

    # 고유벡터 방향 표시 (실수인 경우)
    for i in range(2):
        if np.isreal(eigenvalues[i]):
            v = eigenvectors[:, i].real
            ax.arrow(0, 0, v[0]*1.5, v[1]*1.5, head_width=0.15,
                     head_length=0.1, fc='red', ec='red', linewidth=1.5)
            ax.arrow(0, 0, -v[0]*1.5, -v[1]*1.5, head_width=0.15,
                     head_length=0.1, fc='red', ec='red', linewidth=1.5)

    ax.plot(0, 0, 'ko', markersize=6)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


# --- 4가지 평형점 유형의 위상 초상화 ---
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# (a) 안정 노드: 고유값 모두 음수
A_stable_node = np.array([[-2, 0], [0, -1]])
plot_phase_portrait(A_stable_node, '안정 노드 (Stable Node)', axes[0, 0])

# (b) 안장점: 고유값이 이부호
A_saddle = np.array([[1, 0], [0, -2]])
plot_phase_portrait(A_saddle, '안장점 (Saddle Point)', axes[0, 1])

# (c) 안정 소용돌이: 복소 고유값, 실수부 음수
A_spiral = np.array([[-0.5, 2], [-2, -0.5]])
plot_phase_portrait(A_spiral, '안정 소용돌이 (Stable Spiral)', axes[1, 0])

# (d) 중심: 순허수 고유값
A_center = np.array([[0, 1], [-4, 0]])
plot_phase_portrait(A_center, '중심 (Center)', axes[1, 1])

plt.tight_layout()
plt.savefig('phase_portraits.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 4. 비선형 시스템 개론

### 4.1 선형화 (Linearization)

비선형 자율 시스템 $\mathbf{x}' = \mathbf{F}(\mathbf{x})$의 평형점 $\mathbf{x}^*$ 근방에서의 거동은 **야코비 행렬(Jacobian matrix)**을 이용한 선형화로 분석한다.

$$\mathbf{x}' \approx J(\mathbf{x}^*) (\mathbf{x} - \mathbf{x}^*)$$

여기서 야코비 행렬은:

$$J = \begin{pmatrix} \frac{\partial F_1}{\partial x_1} & \frac{\partial F_1}{\partial x_2} \\ \frac{\partial F_2}{\partial x_1} & \frac{\partial F_2}{\partial x_2} \end{pmatrix}_{\mathbf{x} = \mathbf{x}^*}$$

**하트만-그로브만 정리(Hartman-Grobman theorem)**: 평형점이 **쌍곡적(hyperbolic)**이면 (즉, 야코비 행렬의 모든 고유값의 실수부가 0이 아니면), 비선형 시스템의 위상적 거동은 선형화된 시스템과 **위상동형(topologically equivalent)**이다.

> **주의**: 중심(center)은 쌍곡적이지 않으므로, 선형화만으로는 비선형 시스템의 정확한 거동을 결정할 수 없다.

### 4.2 로트카-볼테라 (Lotka-Volterra) 방정식

포식자-피식자(predator-prey) 상호작용을 모델링하는 고전적인 비선형 시스템:

$$\frac{dx}{dt} = \alpha x - \beta x y \quad \text{(피식자, prey)}$$

$$\frac{dy}{dt} = -\gamma y + \delta x y \quad \text{(포식자, predator)}$$

**평형점**:
1. $(0, 0)$ - 멸종 (trivial)
2. $(\gamma/\delta, \alpha/\beta)$ - 공존 (coexistence)

공존 평형점에서의 야코비 행렬:

$$J = \begin{pmatrix} 0 & -\beta\gamma/\delta \\ \delta\alpha/\beta & 0 \end{pmatrix}$$

고유값이 $\lambda = \pm i\sqrt{\alpha\gamma}$ (순허수)이므로 선형 분석에서는 **중심(center)**이다. 비선형 분석에서도 이 경우 보존량 $V = \delta x - \gamma \ln x + \beta y - \alpha \ln y$가 존재하여 닫힌 궤도(주기 운동)가 확인된다.

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- 로트카-볼테라 시뮬레이션 ---
alpha, beta, gamma, delta = 1.0, 0.5, 0.75, 0.25

def lotka_volterra(t, z):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = -gamma * y + delta * x * y
    return [dxdt, dydt]

# 여러 초기조건
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

initial_conditions = [[2, 1], [4, 2], [1, 3], [6, 1]]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for ic, color in zip(initial_conditions, colors):
    sol = solve_ivp(lotka_volterra, [0, 30], ic,
                    t_eval=np.linspace(0, 30, 1000), method='RK45',
                    rtol=1e-10, atol=1e-12)

    # 시간 영역
    axes[0].plot(sol.t, sol.y[0], '-', color=color, alpha=0.8,
                 label=f'피식자 ({ic[0]},{ic[1]})')
    axes[0].plot(sol.t, sol.y[1], '--', color=color, alpha=0.8,
                 label=f'포식자 ({ic[0]},{ic[1]})')

    # 위상 평면
    axes[1].plot(sol.y[0], sol.y[1], '-', color=color, linewidth=1.5,
                 label=f'IC=({ic[0]},{ic[1]})')
    axes[1].plot(ic[0], ic[1], 'o', color=color, markersize=6)

# 평형점 표시
x_eq, y_eq = gamma / delta, alpha / beta
axes[1].plot(x_eq, y_eq, 'k*', markersize=15, zorder=5, label='평형점')

axes[0].set_xlabel('시간 t')
axes[0].set_ylabel('개체수')
axes[0].set_title('로트카-볼테라: 시간 영역')
axes[0].legend(fontsize=7, ncol=2)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('피식자 x')
axes[1].set_ylabel('포식자 y')
axes[1].set_title('로트카-볼테라: 위상 평면')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# 벡터장 (streamplot)
x_range = np.linspace(0.2, 8, 20)
y_range = np.linspace(0.2, 5, 20)
X, Y = np.meshgrid(x_range, y_range)
U = alpha * X - beta * X * Y
V = -gamma * Y + delta * X * Y
speed = np.sqrt(U**2 + V**2)

axes[2].streamplot(X, Y, U, V, color=speed, cmap='coolwarm', density=1.5,
                   linewidth=0.8)
axes[2].plot(x_eq, y_eq, 'k*', markersize=15, label='평형점')
axes[2].set_xlabel('피식자 x')
axes[2].set_ylabel('포식자 y')
axes[2].set_title('로트카-볼테라: 벡터장')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lotka_volterra.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.3 반데르폴 (Van der Pol) 진동자

비선형 감쇠를 가진 진동자로, 전자공학과 생체 리듬 모델링에 널리 사용된다:

$$\ddot{x} - \mu(1 - x^2)\dot{x} + x = 0$$

- $|x| < 1$일 때: 음의 감쇠 (에너지 공급) → 진폭 증가
- $|x| > 1$일 때: 양의 감쇠 (에너지 소산) → 진폭 감소

이 경쟁으로 인해 $\mu > 0$일 때 **한계 주기(limit cycle)**가 존재한다 ($|x| \approx 2$ 주위).

연립 형태: $x_1 = x$, $x_2 = \dot{x}$

$$\dot{x}_1 = x_2, \quad \dot{x}_2 = \mu(1 - x_1^2) x_2 - x_1$$

원점 $(0, 0)$에서의 야코비 행렬:

$$J = \begin{pmatrix} 0 & 1 \\ -1 & \mu \end{pmatrix}$$

$\mu > 0$이면 $\text{tr}(J) = \mu > 0$이므로 원점은 **불안정**하다.

```python
# --- 반데르폴 진동자 시뮬레이션 ---
def van_der_pol(t, z, mu):
    x, xdot = z
    return [xdot, mu * (1 - x**2) * xdot - x]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
mu_values = [0.1, 1.0, 5.0]

for idx, mu in enumerate(mu_values):
    # 여러 초기조건
    ics = [[0.1, 0], [4, 0], [0, 5], [2, -3]]

    for ic in ics:
        sol = solve_ivp(van_der_pol, [0, 50], ic,
                        args=(mu,), t_eval=np.linspace(0, 50, 2000),
                        method='RK45', rtol=1e-10, atol=1e-12)

        # 시간 영역
        axes[0, idx].plot(sol.t, sol.y[0], linewidth=0.8, alpha=0.8)
        # 위상 평면
        axes[1, idx].plot(sol.y[0], sol.y[1], linewidth=0.8, alpha=0.8)

    axes[0, idx].set_xlabel('t')
    axes[0, idx].set_ylabel('x(t)')
    axes[0, idx].set_title(f'Van der Pol ($\\mu$ = {mu}): 시간 영역')
    axes[0, idx].grid(True, alpha=0.3)

    axes[1, idx].set_xlabel('x')
    axes[1, idx].set_ylabel('$\\dot{x}$')
    axes[1, idx].set_title(f'Van der Pol ($\\mu$ = {mu}): 위상 평면')
    axes[1, idx].plot(0, 0, 'ro', markersize=5)
    axes[1, idx].grid(True, alpha=0.3)
    axes[1, idx].set_aspect('equal')

plt.tight_layout()
plt.savefig('van_der_pol.png', dpi=150, bbox_inches='tight')
plt.show()
```

**관찰 결과**:
- $\mu \to 0$: 거의 정현파적 진동 (약한 비선형성)
- $\mu = 1$: 한계 주기로의 수렴이 뚜렷
- $\mu \gg 1$: **이완 진동(relaxation oscillation)** - 느린/빠른 구간이 교대

---

## 5. 물리학 응용

### 5.1 결합 진동자 (Coupled Oscillators)

스프링으로 연결된 두 질량의 운동을 고려하자:

```
벽 ─── k ─── [m₁] ─── k_c ─── [m₂] ─── k ─── 벽
```

뉴턴의 운동 법칙:

$$m_1 \ddot{x}_1 = -k x_1 - k_c (x_1 - x_2)$$

$$m_2 \ddot{x}_2 = -k x_2 - k_c (x_2 - x_1)$$

$m_1 = m_2 = m$인 대칭 경우, 행렬 형태로:

$$m \begin{pmatrix} \ddot{x}_1 \\ \ddot{x}_2 \end{pmatrix} = -\begin{pmatrix} k + k_c & -k_c \\ -k_c & k + k_c \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

### 5.2 정규 모드 (Normal Modes)

정규 모드는 모든 입자가 **같은 진동수**로 진동하는 특별한 운동 패턴이다.

$\mathbf{x}(t) = \mathbf{u} e^{i\omega t}$를 대입하면 **고유값 문제**:

$$K\mathbf{u} = \omega^2 M\mathbf{u}$$

또는 $M^{-1}K\mathbf{u} = \omega^2 \mathbf{u}$

대칭 경우의 정규 모드:

| 모드 | 주파수 | 패턴 | 물리적 의미 |
|---|---|---|---|
| 모드 1 (동위상) | $\omega_1 = \sqrt{k/m}$ | $\mathbf{u}_1 = (1, 1)^T$ | 두 질량이 같은 방향으로 |
| 모드 2 (역위상) | $\omega_2 = \sqrt{(k + 2k_c)/m}$ | $\mathbf{u}_2 = (1, -1)^T$ | 두 질량이 반대 방향으로 |

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- 결합 진동자와 정규 모드 ---
m = 1.0     # 질량
k = 1.0     # 벽 스프링 상수
kc = 0.5    # 결합 스프링 상수

# 강성 행렬과 질량 행렬
K = np.array([[k + kc, -kc],
              [-kc, k + kc]])
M = np.array([[m, 0],
              [0, m]])

# 정규 모드 (일반화된 고유값 문제)
from scipy.linalg import eigh
omega_sq, modes = eigh(K, M)
omega = np.sqrt(omega_sq)

print("정규 주파수:")
for i, w in enumerate(omega):
    print(f"  omega_{i+1} = {w:.4f} rad/s (f = {w/(2*np.pi):.4f} Hz)")
print(f"\n정규 모드 벡터:")
print(f"  모드 1 (동위상): {modes[:, 0]}")
print(f"  모드 2 (역위상): {modes[:, 1]}")

# 시뮬레이션: 초기에 질량 1만 변위
def coupled_oscillator(t, z):
    x1, x2, v1, v2 = z
    a1 = (-k * x1 - kc * (x1 - x2)) / m
    a2 = (-k * x2 - kc * (x2 - x1)) / m
    return [v1, v2, a1, a2]

# 초기조건: x1(0) = 1, x2(0) = 0 (비트 현상 관찰)
sol = solve_ivp(coupled_oscillator, [0, 60], [1, 0, 0, 0],
                t_eval=np.linspace(0, 60, 2000), method='RK45',
                rtol=1e-12, atol=1e-14)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 각 질량의 변위
axes[0].plot(sol.t, sol.y[0], 'b-', linewidth=1, label='$x_1(t)$ (질량 1)')
axes[0].plot(sol.t, sol.y[1], 'r-', linewidth=1, label='$x_2(t)$ (질량 2)')
axes[0].set_xlabel('시간 t')
axes[0].set_ylabel('변위')
axes[0].set_title('결합 진동자: 에너지 전달 (비트 현상)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 정규 좌표
q1 = (sol.y[0] + sol.y[1]) / np.sqrt(2)  # 동위상 모드
q2 = (sol.y[0] - sol.y[1]) / np.sqrt(2)  # 역위상 모드

axes[1].plot(sol.t, q1, 'g-', linewidth=1, label='$q_1(t)$ (동위상 모드)')
axes[1].plot(sol.t, q2, 'm-', linewidth=1, label='$q_2(t)$ (역위상 모드)')
axes[1].set_xlabel('시간 t')
axes[1].set_ylabel('정규 좌표')
axes[1].set_title('정규 좌표: 각 모드의 독립적 진동')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 에너지 교환
E1 = 0.5 * m * sol.y[2]**2 + 0.5 * k * sol.y[0]**2
E2 = 0.5 * m * sol.y[3]**2 + 0.5 * k * sol.y[1]**2
E_coupling = 0.5 * kc * (sol.y[0] - sol.y[1])**2

axes[2].plot(sol.t, E1, 'b-', linewidth=1, label='질량 1 에너지')
axes[2].plot(sol.t, E2, 'r-', linewidth=1, label='질량 2 에너지')
axes[2].plot(sol.t, E1 + E2 + E_coupling, 'k--', linewidth=0.8,
             label='총 에너지', alpha=0.5)
axes[2].set_xlabel('시간 t')
axes[2].set_ylabel('에너지')
axes[2].set_title('에너지 교환: 비트 진동수 = $|\\omega_2 - \\omega_1|/2$')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coupled_oscillators.png', dpi=150, bbox_inches='tight')
plt.show()

# 비트 진동수 계산
omega_beat = abs(omega[1] - omega[0]) / 2
T_beat = 2 * np.pi / omega_beat if omega_beat > 0 else float('inf')
print(f"\n비트 진동수: {omega_beat:.4f} rad/s")
print(f"비트 주기: {T_beat:.2f} s")
```

### 5.3 라그랑주 역학에서의 연립 ODE

**이중 진자(double pendulum)**는 라그랑주 역학의 대표적인 비선형 연립 ODE 예제이다.

라그랑지안 $L = T - V$에서 오일러-라그랑주 방정식:

$$\frac{d}{dt} \frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0$$

**소진폭(small angle) 근사**에서 이중 진자의 선형화된 운동 방정식:

$$\begin{pmatrix} (m_1 + m_2) l_1 & m_2 l_2 \\ l_1 & l_2 \end{pmatrix} \begin{pmatrix} \ddot{\theta}_1 \\ \ddot{\theta}_2 \end{pmatrix} = -g \begin{pmatrix} (m_1 + m_2) \theta_1 \\ \theta_2 \end{pmatrix}$$

이것은 일반화된 고유값 문제 $K\mathbf{u} = \omega^2 M\mathbf{u}$로 귀결된다.

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- 이중 진자 (비선형 전체 방정식) ---
g = 9.81
m1, m2 = 1.0, 1.0
l1, l2 = 1.0, 1.0

def double_pendulum(t, z):
    th1, th2, w1, w2 = z
    delta = th1 - th2

    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta)**2
    den2 = l2 / l1 * den1

    dw1 = (-m2 * l1 * w1**2 * np.sin(delta) * np.cos(delta)
            - m2 * l2 * w2**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(th1)
            + m2 * g * np.sin(th2) * np.cos(delta)) / den1

    dw2 = (m2 * l2 * w2**2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * l1 * w1**2 * np.sin(delta)
            + (m1 + m2) * g * np.sin(th1) * np.cos(delta)
            - (m1 + m2) * g * np.sin(th2)) / den2

    return [w1, w2, dw1, dw2]

# 두 가지 약간 다른 초기조건 → 카오스 민감성
t_span = (0, 20)
t_eval = np.linspace(0, 20, 5000)

ic1 = [np.pi/2, np.pi/2, 0, 0]
ic2 = [np.pi/2 + 0.001, np.pi/2, 0, 0]  # theta1을 0.001 rad 차이

sol1 = solve_ivp(double_pendulum, t_span, ic1, t_eval=t_eval,
                 method='RK45', rtol=1e-12, atol=1e-14)
sol2 = solve_ivp(double_pendulum, t_span, ic2, t_eval=t_eval,
                 method='RK45', rtol=1e-12, atol=1e-14)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# theta1 비교
axes[0, 0].plot(sol1.t, sol1.y[0], 'b-', linewidth=0.8, label='IC 1')
axes[0, 0].plot(sol2.t, sol2.y[0], 'r--', linewidth=0.8, label='IC 2')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('$\\theta_1$')
axes[0, 0].set_title('$\\theta_1(t)$: 초기조건 민감성 (카오스)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# theta2 비교
axes[0, 1].plot(sol1.t, sol1.y[1], 'b-', linewidth=0.8, label='IC 1')
axes[0, 1].plot(sol2.t, sol2.y[1], 'r--', linewidth=0.8, label='IC 2')
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel('$\\theta_2$')
axes[0, 1].set_title('$\\theta_2(t)$: 초기조건 민감성 (카오스)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 위상 공간 (theta1 vs omega1)
axes[1, 0].plot(sol1.y[0], sol1.y[2], 'b-', linewidth=0.3, alpha=0.7)
axes[1, 0].set_xlabel('$\\theta_1$')
axes[1, 0].set_ylabel('$\\dot{\\theta}_1$')
axes[1, 0].set_title('위상 공간: $(\\theta_1, \\dot{\\theta}_1)$')
axes[1, 0].grid(True, alpha=0.3)

# 진자 끝 궤적
x1 = l1 * np.sin(sol1.y[0])
y1 = -l1 * np.cos(sol1.y[0])
x2 = x1 + l2 * np.sin(sol1.y[1])
y2 = y1 - l2 * np.cos(sol1.y[1])

axes[1, 1].plot(x2, y2, 'b-', linewidth=0.2, alpha=0.5)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')
axes[1, 1].set_title('이중 진자 끝점 궤적')
axes[1, 1].set_aspect('equal')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('이중 진자 (Double Pendulum) - 카오스적 역학', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('double_pendulum.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 소진폭 정규 모드 분석 ---
M_mat = np.array([[(m1 + m2) * l1, m2 * l2],
                  [l1, l2]])
K_mat = np.array([[(m1 + m2) * g, 0],
                  [0, g]])

# 일반화된 고유값 문제: K u = omega^2 M u
from scipy.linalg import eig
eigenvalues, eigenvectors = eig(K_mat, M_mat)
omega_normal = np.sqrt(eigenvalues.real)
omega_normal = np.sort(omega_normal)

print("\n이중 진자 (소진폭) 정규 주파수:")
print(f"  omega_1 = {omega_normal[0]:.4f} rad/s (동위상 모드)")
print(f"  omega_2 = {omega_normal[1]:.4f} rad/s (역위상 모드)")
print(f"  비율 omega_2/omega_1 = {omega_normal[1]/omega_normal[0]:.4f}")
```

---

## 연습 문제

### 기본 문제

**문제 1**: 다음 ODE의 일반해를 구하시오.
- (a) $y^{(4)} + 4y'' = 0$
- (b) $y''' - y = 0$
- (c) $y'' + 4y' + 13y = 0$

<details>
<summary>풀이 힌트</summary>

(a) 특성방정식: $r^4 + 4r^2 = r^2(r^2 + 4) = 0$ → $r = 0, 0, \pm 2i$

일반해: $y = c_1 + c_2 x + c_3 \cos 2x + c_4 \sin 2x$

(b) $r^3 - 1 = (r-1)(r^2+r+1) = 0$ → $r = 1, -\frac{1}{2} \pm \frac{\sqrt{3}}{2}i$

(c) $r^2 + 4r + 13 = 0$ → $r = -2 \pm 3i$

```python
# 검증
import sympy as sp
x = sp.Symbol('x')
y = sp.Function('y')

# (a)
sol_a = sp.dsolve(y(x).diff(x, 4) + 4*y(x).diff(x, 2), y(x))
print(f"(a): {sol_a}")

# (b)
sol_b = sp.dsolve(y(x).diff(x, 3) - y(x), y(x))
print(f"(b): {sol_b}")

# (c)
sol_c = sp.dsolve(y(x).diff(x, 2) + 4*y(x).diff(x) + 13*y(x), y(x))
print(f"(c): {sol_c}")
```

</details>

**문제 2**: 다음 연립 ODE의 일반해를 고유값/고유벡터를 이용하여 구하시오.

$$\frac{d}{dt}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 3 & -2 \\ 2 & -2 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

<details>
<summary>풀이 힌트</summary>

특성방정식: $\lambda^2 - \lambda - 2 = 0$ → $\lambda_1 = 2$, $\lambda_2 = -1$

각 고유값에 대한 고유벡터를 구하고 일반해를 구성하면:

$$\mathbf{x}(t) = c_1 \begin{pmatrix} 2 \\ 1 \end{pmatrix} e^{2t} + c_2 \begin{pmatrix} 1 \\ 2 \end{pmatrix} e^{-t}$$

</details>

### 응용 문제

**문제 3** (결합 진동자): 세 개의 질량 $m$이 스프링 상수 $k$인 동일한 스프링으로 연결되어 있다 (양 끝은 벽에 고정). 정규 주파수와 정규 모드를 구하시오.

<details>
<summary>풀이 힌트</summary>

강성 행렬:

$$K = k \begin{pmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{pmatrix}$$

고유값: $\omega_n^2 = \frac{k}{m}(2 - 2\cos\frac{n\pi}{4})$, $n = 1, 2, 3$

```python
import numpy as np
from scipy.linalg import eigh

k, m_val = 1.0, 1.0
K = k * np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
M = m_val * np.eye(3)

omega_sq, modes = eigh(K, M)
print("정규 주파수:", np.sqrt(omega_sq))
print("정규 모드:\n", modes)
```

</details>

**문제 4** (비선형 분석): 다음 비선형 시스템의 모든 평형점을 찾고, 각 평형점에서의 안정성을 분류하시오.

$$\dot{x} = x(3 - x - 2y), \quad \dot{y} = y(2 - x - y)$$

<details>
<summary>풀이 힌트</summary>

평형점: $(0,0)$, $(3,0)$, $(0,2)$, $(1,1)$ — 시스템이 $\dot{x}=0$, $\dot{y}=0$을 동시에 만족하는 점들.

각 평형점에서 야코비 행렬:

$$J = \begin{pmatrix} 3 - 2x - 2y & -2x \\ -y & 2 - x - 2y \end{pmatrix}$$

```python
import sympy as sp

x, y = sp.symbols('x y')
F1 = x * (3 - x - 2*y)
F2 = y * (2 - x - y)

# 평형점
eq_pts = sp.solve([F1, F2], [x, y])
print("평형점:", eq_pts)

# 야코비 행렬
J = sp.Matrix([[sp.diff(F1, x), sp.diff(F1, y)],
               [sp.diff(F2, x), sp.diff(F2, y)]])

for pt in eq_pts:
    J_at = J.subs([(x, pt[0]), (y, pt[1])])
    eigenvals = J_at.eigenvals()
    print(f"\n평형점 {pt}: 고유값 = {eigenvals}")
```

</details>

**문제 5** (위상 초상화): 매개변수 $\mu$에 따라 다음 시스템의 위상 초상화가 어떻게 변하는지 분석하시오 (**호프 분기, Hopf bifurcation**):

$$\dot{x} = \mu x - y - x(x^2 + y^2), \quad \dot{y} = x + \mu y - y(x^2 + y^2)$$

<details>
<summary>풀이 힌트</summary>

원점에서의 야코비 행렬의 고유값: $\lambda = \mu \pm i$

- $\mu < 0$: 안정 소용돌이
- $\mu = 0$: 분기점 (비쌍곡적)
- $\mu > 0$: 불안정 소용돌이 + 안정 한계 주기 (반지름 $r = \sqrt{\mu}$)

극좌표 $(r, \theta)$로 변환하면 $\dot{r} = r(\mu - r^2)$, $\dot{\theta} = 1$

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def hopf_system(t, z, mu):
    x, y = z
    r_sq = x**2 + y**2
    return [mu*x - y - x*r_sq, x + mu*y - y*r_sq]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
mu_vals = [-0.5, 0.0, 0.5]

for idx, mu in enumerate(mu_vals):
    for r0 in [0.1, 0.3, 0.8, 1.2]:
        for theta0 in [0, np.pi/2, np.pi]:
            ic = [r0*np.cos(theta0), r0*np.sin(theta0)]
            sol = solve_ivp(hopf_system, [0, 30], ic, args=(mu,),
                            t_eval=np.linspace(0, 30, 2000),
                            method='RK45', rtol=1e-10)
            axes[idx].plot(sol.y[0], sol.y[1], linewidth=0.5, alpha=0.6)

    axes[idx].set_title(f'$\\mu = {mu}$')
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('y')
    axes[idx].set_aspect('equal')
    axes[idx].grid(True, alpha=0.3)
    if mu > 0:
        theta = np.linspace(0, 2*np.pi, 100)
        axes[idx].plot(np.sqrt(mu)*np.cos(theta), np.sqrt(mu)*np.sin(theta),
                       'r-', linewidth=2, label=f'한계 주기 $r=\\sqrt{{{mu}}}$')
        axes[idx].legend()

plt.suptitle('호프 분기 (Hopf Bifurcation)', fontsize=14)
plt.tight_layout()
plt.savefig('hopf_bifurcation.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

---

## 참고 자료

### 교재
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 8. Wiley.
2. **Strogatz, S. H.** (2018). *Nonlinear Dynamics and Chaos*, 2nd ed. CRC Press.
   - 비선형 동역학과 위상 평면 분석의 표준 교재
3. **Hirsch, M. W., Smale, S., & Devaney, R. L.** (2013). *Differential Equations, Dynamical Systems, and an Introduction to Chaos*, 3rd ed. Academic Press.
4. **Arfken, G. B. et al.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapter 10. Academic Press.

### 온라인 자료
1. **MIT OCW 18.03**: Differential Equations (Arthur Mattuck)
2. **3Blue1Brown**: Differential equations, studying the unsolvable
3. **Steve Brunton (YouTube)**: Data-Driven Dynamical Systems 시리즈

### 핵심 라이브러리 문서
1. **SciPy `solve_ivp`**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
2. **SymPy `dsolve`**: https://docs.sympy.org/latest/modules/solvers/ode.html
3. **Matplotlib `streamplot`**: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.streamplot.html

---

## 다음 레슨

[11. 급수해와 특수함수 (Series Solutions and Special Functions)](11_Series_Solutions_Special_Functions.md)에서는 **프로베니우스 방법(Frobenius method)**을 이용한 ODE의 급수해와, 물리학에서 가장 중요한 **특수함수들** (베셀 함수, 르장드르 다항식, 에르미트 함수, 라게르 함수)의 성질과 응용을 다룹니다.
