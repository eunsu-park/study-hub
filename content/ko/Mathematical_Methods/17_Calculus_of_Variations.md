# 17. 변분법 (Calculus of Variations)

## 학습 목표

- 범함수(functional)의 개념과 일반 함수와의 차이를 이해한다
- 오일러-라그랑주 방정식을 유도하고 다양한 특수 형태에 적용한다
- 최속 강하선, 측지선, 현수선 등 고전적 변분 문제를 풀 수 있다
- 구속 조건이 있는 변분 문제에 라그랑주 승수법을 적용한다
- 라그랑주 역학과 해밀턴 역학의 변분법적 기초를 이해한다
- 변분법의 현대적 응용(유한요소법, 최적제어, 레일리-리츠법)을 파악한다

> **물리학에서의 중요성**: 변분법은 "자연은 작용을 최소화한다"는 최소작용 원리의 수학적 토대이다. 뉴턴 역학을 라그랑주·해밀턴 형식으로 재구성하고, 페르마 원리(광학), 최소 곡면(비누막), 최적 경로 문제 등 물리학과 공학 전반에서 핵심적 역할을 한다. 양자역학의 경로적분, 일반상대론의 아인슈타인 방정식, 고전장론의 라그랑지안 밀도 모두 변분법에 기초한다.

---

## 1. 변분법의 기본 개념

### 1.1 범함수와 함수의 구분

일반 함수 $f: \mathbb{R} \to \mathbb{R}$는 **수**를 입력받아 **수**를 반환한다. 반면 **범함수(functional)** $J: \mathcal{F} \to \mathbb{R}$는 **함수**를 입력받아 **수**를 반환한다.

$$J[y] = \int_a^b F(x, y(x), y'(x))\, dx$$

여기서 $F$는 **피적분함수(integrand)**이고, $y(x)$는 **허용 함수(admissible function)**의 공간 $\mathcal{F}$에 속한다. 허용 함수는 다음 조건을 만족한다:

- $y(x) \in C^2[a, b]$ (두 번 연속 미분 가능)
- 경계조건: $y(a) = y_a$, $y(b) = y_b$ (고정 끝점)

### 1.2 변분 $\delta y$와 제1변분

허용 함수 $y(x)$에 대한 **변분(variation)**은 경계에서 사라지는 임의의 함수 $\eta(x)$에 대해 다음과 같이 정의된다:

$$\delta y(x) = \epsilon \eta(x), \quad \eta(a) = \eta(b) = 0$$

**제1변분(first variation)**은 범함수의 미분에 해당한다:

$$\delta J = \left.\frac{d}{d\epsilon}J[y + \epsilon\eta]\right|_{\epsilon=0} = \int_a^b \left(\frac{\partial F}{\partial y}\eta + \frac{\partial F}{\partial y'}\eta'\right) dx$$

범함수가 극값을 가지려면, 모든 허용 변분 $\eta$에 대해 $\delta J = 0$이어야 한다.

```python
import numpy as np
import matplotlib.pyplot as plt

def evaluate_functional(F, y_func, x_range, N=1000):
    """범함수 J[y] = ∫F(x, y, y')dx를 수치적으로 계산"""
    a, b = x_range
    x = np.linspace(a, b, N)
    h = (b - a) / (N - 1)

    y = y_func(x)
    # 중심 차분으로 y' 계산
    yp = np.gradient(y, x)

    # 사다리꼴 적분
    integrand = F(x, y, yp)
    return np.trapz(integrand, x)

# 예: 호의 길이 범함수 J[y] = ∫√(1 + y'²)dx
F_arclength = lambda x, y, yp: np.sqrt(1 + yp**2)

# 직선 y = x vs 포물선 y = x²  (구간 [0, 1])
y_line = lambda x: x
y_parabola = lambda x: x**2

J_line = evaluate_functional(F_arclength, y_line, (0, 1))
J_parab = evaluate_functional(F_arclength, y_parabola, (0, 1))

print(f"직선의 호의 길이: J[y=x]     = {J_line:.6f}")
print(f"포물선의 호의 길이: J[y=x²]  = {J_parab:.6f}")
print(f"직선이 더 짧음 (최단 경로): {J_line < J_parab}")
```

---

## 2. 오일러-라그랑주 방정식

### 2.1 유도

$\delta J = 0$ 조건에 부분적분을 적용한다:

$$\delta J = \int_a^b \left(\frac{\partial F}{\partial y}\eta + \frac{\partial F}{\partial y'}\eta'\right) dx$$

두 번째 항을 부분적분하면:

$$\int_a^b \frac{\partial F}{\partial y'}\eta'\, dx = \left[\frac{\partial F}{\partial y'}\eta\right]_a^b - \int_a^b \frac{d}{dx}\frac{\partial F}{\partial y'}\eta\, dx$$

경계 항은 $\eta(a) = \eta(b) = 0$이므로 사라진다. 따라서:

$$\delta J = \int_a^b \left(\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'}\right)\eta\, dx = 0$$

**변분법의 기본 보조정리**: 모든 $\eta$에 대해 위 적분이 0이면 피적분함수가 0이어야 한다. 따라서 **오일러-라그랑주 방정식**이 도출된다:

$$\boxed{\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} = 0}$$

### 2.2 특수한 경우

**경우 1**: $F$가 $y$에 의존하지 않을 때 ($F = F(x, y')$):

$$\frac{\partial F}{\partial y'} = C \quad \text{(상수)}$$

**경우 2**: $F$가 $x$에 명시적으로 의존하지 않을 때 ($F = F(y, y')$) -- **벨트라미 항등식**:

$$\boxed{F - y'\frac{\partial F}{\partial y'} = C}$$

이는 $\frac{d}{dx}\left(F - y'F_{y'}\right) = -xF_x = 0$으로부터 도출된다.

```python
import sympy as sp

x = sp.Symbol('x')
y = sp.Function('y')(x)
yp = y.diff(x)

def euler_lagrange(F, y_func, x_var):
    """오일러-라그랑주 방정식을 심볼릭으로 도출"""
    yp = y_func.diff(x_var)
    # ∂F/∂y
    dF_dy = sp.diff(F, y_func)
    # d/dx(∂F/∂y')
    dF_dyp = sp.diff(F, yp)
    d_dx_dF_dyp = sp.diff(dF_dyp, x_var)

    el_eq = sp.simplify(dF_dy - d_dx_dF_dyp)
    return sp.Eq(el_eq, 0)

# 예 1: 최단 거리 — F = √(1 + y'²)
F1 = sp.sqrt(1 + yp**2)
el1 = euler_lagrange(F1, y, x)
print("=== 최단 거리 문제 ===")
print(f"오일러-라그랑주: {el1}")
print("해: y'' = 0  →  y = ax + b (직선)")

# 예 2: 최소 곡면 (회전면) — F = y√(1 + y'²)
F2 = y * sp.sqrt(1 + yp**2)
el2 = euler_lagrange(F2, y, x)
print(f"\n=== 최소 회전면 ===")
print(f"오일러-라그랑주: {el2}")
print("해: y = c₁ cosh((x - c₂)/c₁) (현수선)")
```

---

## 3. 고전적 변분 문제들

### 3.1 최속 강하선 (Brachistochrone Problem)

중력장에서 점 $A(0,0)$에서 점 $B(x_1, y_1)$까지 마찰 없이 미끄러지는 가장 빠른 경로를 구하라. ($y$축이 아래로 향한다.)

에너지 보존에 의해 속력은 $v = \sqrt{2gy}$이고, 이동 시간은:

$$T[y] = \int_0^{x_1} \frac{\sqrt{1 + y'^2}}{\sqrt{2gy}}\, dx$$

$F = \sqrt{(1 + y'^2)/(2gy)}$는 $x$에 명시적으로 의존하지 않으므로 벨트라미 항등식을 적용한다:

$$F - y'F_{y'} = C \implies \frac{1}{\sqrt{2gy(1 + y'^2)}} = C$$

정리하면 $y(1 + y'^2) = \frac{1}{2gC^2} = 2R$ (상수). 이를 매개변수 $\theta$로 풀면 **사이클로이드** 해가 된다:

$$\boxed{x = R(\theta - \sin\theta), \quad y = R(1 - \cos\theta)}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def brachistochrone(x1, y1, N=500):
    """최속 강하선 (사이클로이드) 계산"""
    def equations(params):
        R, theta1 = params
        eq1 = R * (theta1 - np.sin(theta1)) - x1
        eq2 = R * (1 - np.cos(theta1)) - y1
        return [eq1, eq2]

    R, theta1 = fsolve(equations, [1.0, np.pi])

    theta = np.linspace(0, theta1, N)
    x_cyc = R * (theta - np.sin(theta))
    y_cyc = R * (1 - np.cos(theta))
    return x_cyc, y_cyc, R, theta1

# 사이클로이드 vs 직선 vs 포물선 비교
x1, y1 = 1.0, 0.6
x_cyc, y_cyc, R, th1 = brachistochrone(x1, y1)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# 직선
ax.plot([0, x1], [0, y1], 'b--', label='직선', linewidth=2)

# 포물선 (y = ax²)
x_par = np.linspace(0, x1, 200)
a_par = y1 / x1**2
y_par = a_par * x_par**2
ax.plot(x_par, y_par, 'g-.', label='포물선', linewidth=2)

# 사이클로이드
ax.plot(x_cyc, y_cyc, 'r-', label='사이클로이드 (최속)', linewidth=2.5)

ax.set_xlabel('x')
ax.set_ylabel('y (아래 방향)')
ax.set_title('최속 강하선 문제 (Brachistochrone)')
ax.legend()
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('brachistochrone.png', dpi=150)
plt.show()

# 각 경로의 이동 시간 비교
g = 9.81
def travel_time(x_path, y_path):
    """경로를 따라 이동하는 시간 계산"""
    ds = np.sqrt(np.diff(x_path)**2 + np.diff(y_path)**2)
    v = np.sqrt(2 * g * (y_path[:-1] + y_path[1:]) / 2)
    v = np.where(v < 1e-10, 1e-10, v)  # 0 나눗셈 방지
    return np.sum(ds / v)

t_line = travel_time(np.linspace(0, x1, 500), np.linspace(0, y1, 500))
t_par = travel_time(x_par, y_par)
t_cyc = travel_time(x_cyc, y_cyc)

print(f"직선 이동 시간:        {t_line:.4f}초")
print(f"포물선 이동 시간:      {t_par:.4f}초")
print(f"사이클로이드 이동 시간: {t_cyc:.4f}초 (최소)")
```

### 3.2 측지선 (Geodesics)

**평면에서의 측지선**: 두 점 사이의 최단 거리. $F = \sqrt{1 + y'^2}$로부터 $y'' = 0$, 즉 **직선**이 해이다.

**구면에서의 측지선**: 구면 좌표 $(\theta, \phi)$에서 호의 길이는:

$$L = R\int \sqrt{d\theta^2 + \sin^2\theta\, d\phi^2} = R\int_{\phi_1}^{\phi_2} \sqrt{\theta'^2 + \sin^2\theta}\, d\phi$$

오일러-라그랑주 방정식을 풀면 **대원(great circle)**이 된다.

### 3.3 현수선 (Catenary)

밀도가 균일한 체인이 양 끝이 고정된 상태로 매달려 있을 때, 위치에너지를 최소화하는 형태를 구하라:

$$U[y] = \rho g \int_{-a}^{a} y\sqrt{1 + y'^2}\, dx$$

체인의 길이 $L = \int \sqrt{1 + y'^2}\, dx$가 일정하다는 구속 조건 하에서, $F = y\sqrt{1 + y'^2}$에 벨트라미 항등식을 적용하면:

$$y = c_1 \cosh\left(\frac{x - c_2}{c_1}\right)$$

이것이 **현수선(catenary)**이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 현수선 그리기
def catenary(x, c1, c2=0):
    """현수선: y = c₁ cosh((x - c₂)/c₁)"""
    return c1 * np.cosh((x - c2) / c1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) 다양한 매개변수의 현수선
x = np.linspace(-2, 2, 300)
for c in [0.5, 1.0, 1.5, 2.0]:
    axes[0].plot(x, catenary(x, c), label=f'$c_1 = {c}$')
axes[0].set_title('다양한 매개변수의 현수선')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# (b) 현수선 vs 포물선 비교
c1 = 1.0
y_cat = catenary(x, c1) - c1  # 최저점을 0으로 이동
y_par = 0.5 * x**2            # 비슷한 모양의 포물선
axes[1].plot(x, y_cat, 'r-', linewidth=2, label='현수선')
axes[1].plot(x, y_par, 'b--', linewidth=2, label='포물선')
axes[1].set_title('현수선 vs 포물선')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('catenary.png', dpi=150)
plt.show()
```

### 3.4 등주 문제 (Isoperimetric Problem)

주어진 둘레 $L$을 가진 폐곡선 중 면적을 최대화하는 형태를 구하라. 해는 **원**이다.

면적: $A = \frac{1}{2}\oint (x\, dy - y\, dx)$, 둘레 구속: $L = \oint \sqrt{dx^2 + dy^2}$

### 3.5 최소 곡면 (비누막 문제)

두 동축 원형 고리 사이에 형성되는 비누막의 최소 회전면. 범함수:

$$A[y] = 2\pi\int_a^b y\sqrt{1 + y'^2}\, dx$$

벨트라미 항등식으로 풀면 **현수면(catenoid)**: $y = c\cosh\left(\frac{x - x_0}{c}\right)$

---

## 4. 다변수와 고차 도함수

### 4.1 다변수 오일러-라그랑주 방정식

여러 종속변수 $y_1(x), y_2(x), \ldots, y_n(x)$에 대한 범함수:

$$J[y_1, \ldots, y_n] = \int_a^b F(x, y_1, \ldots, y_n, y_1', \ldots, y_n')\, dx$$

각 $y_i$에 대해 독립적인 오일러-라그랑주 방정식이 성립한다:

$$\frac{\partial F}{\partial y_i} - \frac{d}{dx}\frac{\partial F}{\partial y_i'} = 0, \quad i = 1, 2, \ldots, n$$

### 4.2 고차 도함수: 오일러-푸아송 방정식

$F$가 $y''$까지 포함할 때:

$$J[y] = \int_a^b F(x, y, y', y'')\, dx$$

**오일러-푸아송 방정식**:

$$\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} + \frac{d^2}{dx^2}\frac{\partial F}{\partial y''} = 0$$

경계조건으로 $y$와 $y'$이 양 끝에서 고정되어야 한다 (4개의 경계조건).

**예**: 탄성 빔의 굽힘 에너지 $J[y] = \int_0^L \frac{EI}{2}(y'')^2\, dx$에서 $y'''' = 0$, 즉 3차 다항식이 해이다.

### 4.3 다중 독립변수

$u(x, y)$에 대한 범함수:

$$J[u] = \iint_D F(x, y, u, u_x, u_y)\, dx\, dy$$

오일러-라그랑주 방정식:

$$\frac{\partial F}{\partial u} - \frac{\partial}{\partial x}\frac{\partial F}{\partial u_x} - \frac{\partial}{\partial y}\frac{\partial F}{\partial u_y} = 0$$

**예**: 디리클레 범함수 $J[u] = \iint (u_x^2 + u_y^2)\, dx\, dy$를 최소화하면 $\nabla^2 u = 0$ (라플라스 방정식)이 된다.

```python
import sympy as sp

x = sp.Symbol('x')
y1 = sp.Function('y1')(x)
y2 = sp.Function('y2')(x)

# 2변수 라그랑지안 예: 결합 진동자
# L = ½m(ẏ₁² + ẏ₂²) - ½k(y₁² + y₂²) - ½k'(y₁ - y₂)²
m, k, kp = sp.symbols('m k k_prime', positive=True)
y1p, y2p = y1.diff(x), y2.diff(x)

T = sp.Rational(1, 2) * m * (y1p**2 + y2p**2)
V = sp.Rational(1, 2) * k * (y1**2 + y2**2) + sp.Rational(1, 2) * kp * (y1 - y2)**2
L = T - V

# 각 변수에 대한 오일러-라그랑주
for yi, name in [(y1, 'y₁'), (y2, 'y₂')]:
    yip = yi.diff(x)
    dL_dyi = sp.diff(L, yi)
    dL_dyip = sp.diff(L, yip)
    d_dx = sp.diff(dL_dyip, x)
    eq = sp.simplify(dL_dyi - d_dx)
    print(f"E-L for {name}: {eq} = 0")
```

---

## 5. 구속 조건이 있는 변분 문제

### 5.1 등주 구속 (Isoperimetric Constraints)

범함수 $J[y] = \int_a^b F(x, y, y')\, dx$를 다음 구속 조건 하에서 극값화하라:

$$K[y] = \int_a^b G(x, y, y')\, dx = \ell \quad \text{(주어진 상수)}$$

**라그랑주 승수법**: 보조 범함수 $\bar{J} = J + \lambda K$를 정의하고, $\bar{F} = F + \lambda G$에 대해 오일러-라그랑주 방정식을 적용한다:

$$\frac{\partial \bar{F}}{\partial y} - \frac{d}{dx}\frac{\partial \bar{F}}{\partial y'} = 0$$

미지수 $\lambda$는 구속 조건 $K = \ell$로 결정된다.

### 5.2 홀로노믹 구속 (Holonomic Constraints)

$g(x, y_1, y_2) = 0$ 형태의 대수적 구속:

$$\frac{\partial F}{\partial y_i} - \frac{d}{dx}\frac{\partial F}{\partial y_i'} + \lambda(x)\frac{\partial g}{\partial y_i} = 0$$

여기서 $\lambda(x)$는 점마다 다를 수 있는 승수 함수이다.

### 5.3 구속 변분 문제 예: 최대 면적 곡선

길이 $L$이 주어진 곡선 아래의 면적을 최대화하라:

$$J[y] = \int_0^a y\, dx, \quad K[y] = \int_0^a \sqrt{1 + y'^2}\, dx = L$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def isoperimetric_circle(a, L):
    """등주 문제: 길이 L, 구간 [0, a]에서 최대 면적 곡선 (원호)"""
    # 원호: 반지름 R, 중심 (a/2, y_c)
    # L = 2R·arcsin(a/(2R))로부터 R 결정
    def eq(R):
        if R < a / 2:
            return 1e10
        return 2 * R * np.arcsin(a / (2 * R)) - L

    R = fsolve(eq, L / np.pi)[0]

    theta = np.linspace(-np.arcsin(a / (2 * R)), np.arcsin(a / (2 * R)), 300)
    x_arc = R * np.sin(theta) + a / 2
    y_arc = R * np.cos(theta) - R * np.cos(np.arcsin(a / (2 * R)))

    return x_arc, y_arc, R

# 구간 길이 2, 곡선 길이 π
a, L = 2.0, np.pi
x_arc, y_arc, R = isoperimetric_circle(a, L)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_arc, y_arc, 'r-', linewidth=2, label=f'원호 (R={R:.3f})')
ax.fill_between(x_arc, 0, y_arc, alpha=0.2, color='red')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_title('등주 문제: 주어진 길이에서 최대 면적')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('isoperimetric.png', dpi=150)
plt.show()
```

---

## 6. 라그랑주 역학

### 6.1 최소작용 원리

**해밀턴의 원리**: 역학계의 실제 운동 경로는 **작용(action)**을 정류(stationarize)하는 경로이다.

$$S[q] = \int_{t_1}^{t_2} L(q, \dot{q}, t)\, dt, \quad \delta S = 0$$

**라그랑지안**: $L = T - V$ (운동에너지 - 위치에너지)

**일반화 좌표(generalized coordinates)** $q_i$: 계의 자유도를 기술하는 독립 변수. 구속 조건을 자동으로 처리한다.

각 일반화 좌표에 대한 **라그랑주 운동방정식**:

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0, \quad i = 1, 2, \ldots, n$$

### 6.2 단진자

일반화 좌표: 각도 $\theta$

$$T = \frac{1}{2}ml^2\dot{\theta}^2, \quad V = -mgl\cos\theta$$

$$L = \frac{1}{2}ml^2\dot{\theta}^2 + mgl\cos\theta$$

오일러-라그랑주 방정식: $ml^2\ddot{\theta} + mgl\sin\theta = 0$, 즉 $\ddot{\theta} + \frac{g}{l}\sin\theta = 0$

```python
import sympy as sp

t = sp.Symbol('t')
m, l, g = sp.symbols('m l g', positive=True)
theta = sp.Function('theta')(t)
theta_dot = theta.diff(t)

# 라그랑지안
T = sp.Rational(1, 2) * m * l**2 * theta_dot**2
V = -m * g * l * sp.cos(theta)
Lag = T - V

# 오일러-라그랑주 방정식 도출
dL_dtheta = sp.diff(Lag, theta)
dL_dthetadot = sp.diff(Lag, theta_dot)
d_dt = sp.diff(dL_dthetadot, t)

eq = sp.simplify(d_dt - dL_dtheta)
print("=== 단진자 운동방정식 ===")
print(f"  {eq} = 0")
# 정리: ml²θ̈ + mgl·sin(θ) = 0
```

### 6.3 이중 진자

일반화 좌표: $\theta_1, \theta_2$

$$x_1 = l_1\sin\theta_1, \quad y_1 = -l_1\cos\theta_1$$

$$x_2 = l_1\sin\theta_1 + l_2\sin\theta_2, \quad y_2 = -l_1\cos\theta_1 - l_2\cos\theta_2$$

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def double_pendulum_eom(t, state, m1, m2, l1, l2, g=9.81):
    """이중 진자의 운동방정식 (라그랑주 역학으로 유도)
    state = [theta1, theta2, omega1, omega2]
    """
    th1, th2, w1, w2 = state
    dth = th1 - th2
    cos_d, sin_d = np.cos(dth), np.sin(dth)
    M = m1 + m2

    # 질량 행렬의 역을 이용한 각가속도 계산
    den = M * l1 - m2 * l1 * cos_d**2

    alpha1 = (-m2 * l2 * w2**2 * sin_d
              - m2 * g * np.sin(th2) * cos_d
              + M * g * np.sin(th1)) / (-den)

    den2 = (l2 / l1) * den

    alpha2 = (M * l1 * w1**2 * sin_d
              + M * g * np.sin(th1) * cos_d
              - M * g * np.sin(th2)) / den2

    return [w1, w2, alpha1, alpha2]

# 시뮬레이션
m1, m2, l1, l2 = 1.0, 1.0, 1.0, 1.0
state0 = [np.pi / 2, np.pi / 2, 0, 0]  # 초기 각도 90°, 각속도 0
t_span = (0, 20)
t_eval = np.linspace(*t_span, 2000)

sol = solve_ivp(double_pendulum_eom, t_span, state0, t_eval=t_eval,
                args=(m1, m2, l1, l2), method='RK45', rtol=1e-10)

# 데카르트 좌표 변환
x1 = l1 * np.sin(sol.y[0])
y1_pos = -l1 * np.cos(sol.y[0])
x2 = x1 + l2 * np.sin(sol.y[1])
y2_pos = y1_pos - l2 * np.cos(sol.y[1])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 끝점 궤적
axes[0].plot(x2, y2_pos, 'b-', linewidth=0.3, alpha=0.7)
axes[0].set_title('이중 진자: 끝점 궤적')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# 각도 시간 그래프
axes[1].plot(sol.t, sol.y[0], label=r'$\theta_1$')
axes[1].plot(sol.t, sol.y[1], label=r'$\theta_2$')
axes[1].set_title('이중 진자: 각도 vs 시간')
axes[1].set_xlabel('시간 (s)')
axes[1].set_ylabel('각도 (rad)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('double_pendulum.png', dpi=150)
plt.show()
```

### 6.4 중심력 문제

일반화 좌표: $(r, \theta)$ (극좌표)

$$L = \frac{1}{2}m(\dot{r}^2 + r^2\dot{\theta}^2) - V(r)$$

$\theta$가 순환 좌표($L$이 $\theta$에 명시적으로 의존하지 않음)이므로:

$$p_\theta = \frac{\partial L}{\partial \dot{\theta}} = mr^2\dot{\theta} = \text{const} \quad \text{(각운동량 보존)}$$

### 6.5 애트우드 기계 (Atwood Machine)

두 질량 $m_1, m_2$가 도르래에 줄로 연결되어 있다. 일반화 좌표: $x$ (한 쪽 질량의 변위).

$$T = \frac{1}{2}(m_1 + m_2)\dot{x}^2, \quad V = -(m_1 - m_2)gx$$

$$L = \frac{1}{2}(m_1 + m_2)\dot{x}^2 + (m_1 - m_2)gx$$

오일러-라그랑주 방정식: $(m_1 + m_2)\ddot{x} = (m_1 - m_2)g$, 즉 $\ddot{x} = \frac{m_1 - m_2}{m_1 + m_2}g$

### 6.6 뇌터 정리 (Noether's Theorem)

라그랑지안이 어떤 연속 변환에 대해 불변이면, 그에 대응하는 보존량이 존재한다:

| 대칭 (변환) | 보존량 |
|-------------|--------|
| 시간 병진 ($t \to t + \epsilon$) | 에너지 |
| 공간 병진 ($x \to x + \epsilon$) | 운동량 |
| 회전 ($\theta \to \theta + \epsilon$) | 각운동량 |

---

## 7. 해밀턴 역학 기초

### 7.1 르장드르 변환

라그랑지안 $L(q, \dot{q}, t)$에서 **해밀토니안** $H(q, p, t)$으로의 변환:

일반화 운동량: $p_i = \frac{\partial L}{\partial \dot{q}_i}$

$$\boxed{H(q, p, t) = \sum_i p_i \dot{q}_i - L(q, \dot{q}, t)}$$

여기서 $\dot{q}_i$는 $p_i = \partial L / \partial \dot{q}_i$를 역으로 풀어 $q, p$로 표현한다.

보존계(시간 독립)에서 $H = T + V$ (전체 에너지)이다.

### 7.2 해밀턴의 정준 방정식

$$\boxed{\dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}}$$

라그랑주 방정식이 $n$개의 2차 ODE인 반면, 해밀턴 방정식은 $2n$개의 1차 ODE이다.

### 7.3 위상 공간과 푸아송 괄호

**위상 공간(phase space)**: $(q_1, \ldots, q_n, p_1, \ldots, p_n)$로 이루어진 $2n$차원 공간.

**푸아송 괄호**:

$$\{A, B\} = \sum_i \left(\frac{\partial A}{\partial q_i}\frac{\partial B}{\partial p_i} - \frac{\partial A}{\partial p_i}\frac{\partial B}{\partial q_i}\right)$$

시간 발전: $\dot{A} = \{A, H\} + \frac{\partial A}{\partial t}$. 보존량이면 $\{A, H\} = 0$.

### 7.4 양자역학과의 연결

양자역학에서 푸아송 괄호는 교환자로 대체된다:

$$\{A, B\}_{\text{Poisson}} \to \frac{1}{i\hbar}[\hat{A}, \hat{B}]$$

정준 교환 관계 $\{q, p\} = 1$은 양자역학의 $[\hat{q}, \hat{p}] = i\hbar$에 대응한다. 이것이 **정준 양자화(canonical quantization)**의 출발점이다.

```python
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# === 심볼릭: 1차원 조화진동자의 해밀턴 역학 ===
t = sp.Symbol('t')
m_sym, k_sym = sp.symbols('m k', positive=True)
q_sym = sp.Function('q')(t)
p_sym = sp.Function('p')(t)

# 해밀토니안: H = p²/(2m) + kq²/2
H = p_sym**2 / (2 * m_sym) + k_sym * q_sym**2 / 2

# 해밀턴의 정준 방정식
q_dot = sp.diff(H, p_sym)      # dq/dt = ∂H/∂p = p/m
p_dot = -sp.diff(H, q_sym)     # dp/dt = -∂H/∂q = -kq

print("=== 조화진동자 해밀턴 방정식 ===")
print(f"  dq/dt = {q_dot}")
print(f"  dp/dt = {p_dot}")

# === 수치 해: 위상 공간 궤적 ===
def harmonic_hamiltonian(t, state, m=1.0, k=1.0):
    q, p = state
    dqdt = p / m       # ∂H/∂p
    dpdt = -k * q      # -∂H/∂q
    return [dqdt, dpdt]

# 여러 초기 조건으로 위상 공간 궤적
fig, ax = plt.subplots(figsize=(7, 7))
for E0 in [0.5, 1.0, 2.0, 4.0]:
    q0 = np.sqrt(2 * E0)  # p0 = 0일 때 최대 변위
    sol = solve_ivp(harmonic_hamiltonian, (0, 10), [q0, 0],
                    t_eval=np.linspace(0, 10, 1000))
    ax.plot(sol.y[0], sol.y[1], label=f'E = {E0}')

ax.set_xlabel('q (위치)')
ax.set_ylabel('p (운동량)')
ax.set_title('조화진동자 위상 공간')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('phase_space.png', dpi=150)
plt.show()
```

---

## 8. 변분법의 현대적 응용

### 8.1 페르마 원리 (광학)

빛은 두 점 사이를 **광학 경로 길이**가 정류(stationary)인 경로로 진행한다:

$$\delta \int_A^B n(x, y)\, ds = 0$$

여기서 $n$은 굴절률. 이로부터 스넬의 법칙이 도출된다:

$$n_1 \sin\theta_1 = n_2 \sin\theta_2$$

```python
import numpy as np
import matplotlib.pyplot as plt

def snell_law_demo():
    """페르마 원리와 스넬의 법칙 시각화"""
    n1, n2 = 1.0, 1.5  # 굴절률 (공기 → 유리)
    y_source, y_target = 2.0, -2.0
    x_source, x_target = -1.0, 1.5

    # 경계면에서의 입사점 x를 변화시키며 광학 경로 길이 계산
    x_boundary = np.linspace(-2, 3, 1000)
    optical_path = []
    for xb in x_boundary:
        d1 = np.sqrt((xb - x_source)**2 + y_source**2)
        d2 = np.sqrt((x_target - xb)**2 + y_target**2)
        opl = n1 * d1 + n2 * d2
        optical_path.append(opl)

    optical_path = np.array(optical_path)
    x_opt = x_boundary[np.argmin(optical_path)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) 광학 경로 길이 vs 입사점
    axes[0].plot(x_boundary, optical_path, 'b-')
    axes[0].axvline(x_opt, color='r', linestyle='--', label=f'최소점 x={x_opt:.3f}')
    axes[0].set_xlabel('경계면 입사점 x')
    axes[0].set_ylabel('광학 경로 길이')
    axes[0].set_title('페르마 원리: 광학 경로 길이 최소화')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (b) 빛의 경로
    axes[1].fill_between([-3, 4], 0, 3, alpha=0.1, color='blue', label=f'매질 1 (n={n1})')
    axes[1].fill_between([-3, 4], -3, 0, alpha=0.1, color='orange', label=f'매질 2 (n={n2})')
    axes[1].plot([x_source, x_opt], [y_source, 0], 'r-', linewidth=2)
    axes[1].plot([x_opt, x_target], [0, y_target], 'r-', linewidth=2)
    axes[1].axhline(y=0, color='k', linewidth=1)
    axes[1].plot(x_source, y_source, 'ko', markersize=8)
    axes[1].plot(x_target, y_target, 'ko', markersize=8)
    axes[1].set_title('스넬의 법칙 (페르마 원리)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('fermat_snell.png', dpi=150)
    plt.show()

    # 스넬의 법칙 검증
    theta1 = np.arctan2(abs(x_opt - x_source), y_source)
    theta2 = np.arctan2(abs(x_target - x_opt), abs(y_target))
    print(f"입사각: {np.degrees(theta1):.2f}°, 굴절각: {np.degrees(theta2):.2f}°")
    print(f"n₁ sin θ₁ = {n1 * np.sin(theta1):.4f}")
    print(f"n₂ sin θ₂ = {n2 * np.sin(theta2):.4f}")

snell_law_demo()
```

### 8.2 레일리-리츠 방법 (Rayleigh-Ritz Method)

변분 문제의 근사해를 구하는 **직접법(direct method)**. 해를 기저함수의 선형 결합으로 가정한다:

$$y(x) \approx \sum_{i=1}^{N} c_i \phi_i(x)$$

범함수를 $c_i$의 함수로 바꾸고, $\partial J / \partial c_i = 0$을 풀어 계수를 결정한다.

```python
import numpy as np
import matplotlib.pyplot as plt

def rayleigh_ritz_beam(N_basis=5, L_beam=1.0, q0=1.0, EI=1.0):
    """
    레일리-리츠법으로 단순 지지 보의 처짐 계산
    범함수: J[y] = ∫[EI/2·(y'')² - q₀·y]dx
    경계조건: y(0) = y(L) = 0, y''(0) = y''(L) = 0
    기저함수: φₙ(x) = sin(nπx/L)
    """
    x = np.linspace(0, L_beam, 500)

    # 정확해: y = q₀/(24EI) · x(L³ - 2Lx² + x³)
    y_exact = q0 / (24 * EI) * x * (L_beam**3 - 2 * L_beam * x**2 + x**3)

    # 레일리-리츠 근사
    y_approx = np.zeros_like(x)
    coefficients = []

    for n in range(1, N_basis + 1):
        kn = n * np.pi / L_beam
        # 강성 행렬 대각 성분
        K_nn = EI * kn**4 * L_beam / 2
        # 하중 벡터: 홀수 n만 기여
        if n % 2 == 1:
            f_n = 2 * q0 * L_beam / (n * np.pi)
        else:
            f_n = 0
        c_n = f_n / K_nn
        coefficients.append(c_n)
        y_approx += c_n * np.sin(n * np.pi * x / L_beam)

    # 그래프
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 처짐 비교
    axes[0].plot(x, y_exact, 'r-', linewidth=2, label='정확해')
    axes[0].plot(x, y_approx, 'b--', linewidth=2, label=f'R-R ({N_basis}항)')
    axes[0].set_title('레일리-리츠법: 보의 처짐')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y(x)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 수렴 분석
    errors = []
    N_range = range(1, 20)
    for N in N_range:
        y_rr = np.zeros_like(x)
        for n in range(1, N + 1):
            kn = n * np.pi / L_beam
            K_nn = EI * kn**4 * L_beam / 2
            f_n = 2 * q0 * L_beam / (n * np.pi) if n % 2 == 1 else 0
            y_rr += (f_n / K_nn) * np.sin(n * np.pi * x / L_beam)
        err = np.max(np.abs(y_rr - y_exact))
        errors.append(err)

    axes[1].semilogy(list(N_range), errors, 'ko-')
    axes[1].set_title('기저함수 수에 따른 오차 수렴')
    axes[1].set_xlabel('기저함수 수 N')
    axes[1].set_ylabel('최대 오차')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rayleigh_ritz.png', dpi=150)
    plt.show()

    return coefficients

coeffs = rayleigh_ritz_beam(N_basis=7)
print("레일리-리츠 계수:", [f"{c:.6e}" for c in coeffs])
```

### 8.3 최적 제어 이론 (개요)

**문제**: 상태 $\mathbf{x}(t)$와 제어 $\mathbf{u}(t)$에 대해 비용 범함수를 최소화:

$$J = \int_{t_0}^{t_f} L(\mathbf{x}, \mathbf{u}, t)\, dt$$

구속: $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u}, t)$ (상태 방정식)

**폰트랴긴의 최대 원리**: 해밀토니안 $H = L + \boldsymbol{\lambda}^T \mathbf{f}$를 정의하고, $H$를 $\mathbf{u}$에 대해 최소화하는 것이 최적 제어이다.

### 8.4 유한요소법과의 연결

변분 문제의 약형(weak form)은 유한요소법(FEM)의 출발점이다. 디리클레 문제:

$$\text{최소화: } J[u] = \frac{1}{2}\int_\Omega |\nabla u|^2\, d\Omega - \int_\Omega fu\, d\Omega$$

의 오일러-라그랑주 방정식은 $-\nabla^2 u = f$ (포아송 방정식)이다. FEM은 이 변분 문제를 유한 차원 기저함수로 이산화하여 근사 해를 구한다.

---

## 연습 문제

### 기초 문제

1. 다음 범함수의 오일러-라그랑주 방정식을 구하라:
   - (a) $J[y] = \int_0^1 (y'^2 + 2yy')\, dx$, $y(0)=0$, $y(1)=1$
   - (b) $J[y] = \int_0^{\pi} (y'^2 - y^2)\, dx$, $y(0)=0$, $y(\pi)=0$
   - (c) $J[y] = \int_1^2 \frac{\sqrt{1 + y'^2}}{x}\, dx$

2. 벨트라미 항등식을 이용하여 $J[y] = \int_0^1 (y'^2 + y^2)\, dx$의 극값 곡선을 구하라. ($y(0)=0$, $y(1)=1$)

3. 호의 길이 $L$이 주어진 두 점 $(0, 0)$, $(a, 0)$을 잇는 곡선 중 $x$축과의 면적이 최대인 곡선을 구하라.

### 중급 문제

4. 구면 위의 측지선이 대원임을 오일러-라그랑주 방정식으로 증명하라. (힌트: $\phi$를 독립변수로 사용)

5. **최속 강하선**: 점 $(0, 0)$에서 $(1, 1)$까지의 사이클로이드를 매개변수로 구하고, 이동 시간을 직선 경로와 비교하라.

6. 이중 진자의 라그랑지안을 구하고, $\theta_1$, $\theta_2$에 대한 운동방정식을 유도하라.

7. 1차원 조화진동자 $L = \frac{1}{2}m\dot{x}^2 - \frac{1}{2}kx^2$에 대해:
   - (a) 라그랑주 운동방정식을 구하라
   - (b) 해밀토니안으로 변환하라
   - (c) 해밀턴의 정준 방정식이 (a)의 결과와 일치함을 보여라

### 심화 문제

8. **탄성 빔**: 캔틸레버 빔의 굽힘을 $J[y] = \int_0^L \left[\frac{EI}{2}(y'')^2 - qy\right] dx$로 모델링하라. 경계조건 $y(0) = y'(0) = 0$, $y''(L) = y'''(L) = 0$에서 해를 구하라.

9. **전기장의 변분 원리**: 전하 분포 $\rho(\mathbf{r})$가 주어질 때, 범함수 $J[\phi] = \int \left[\frac{\epsilon_0}{2}|\nabla\phi|^2 + \rho\phi\right] d^3r$의 오일러-라그랑주 방정식이 포아송 방정식임을 보여라.

10. 레일리-리츠법으로 $y'' + y = 1$ ($y(0) = y(\pi) = 0$)의 근사해를 $y \approx c_1\sin x + c_2\sin 2x + c_3\sin 3x$로 구하고, 정확해와 비교하라.

---

## 심화 학습

### 제2변분과 충분 조건

제1변분 $\delta J = 0$은 **필요 조건**이다. 실제로 극소인지 확인하려면 **제2변분**을 조사한다:

$$\delta^2 J = \int_a^b \left(P\eta^2 + Q\eta'^2\right) dx$$

여기서 $P = F_{yy} - \frac{d}{dx}F_{yy'}$, $Q = F_{y'y'}$. $\delta^2 J > 0$이면 극소, $\delta^2 J < 0$이면 극대.

**르장드르 조건**: $F_{y'y'} > 0$이면 약한 극소의 필요 조건.

**야코비 조건**: 야코비 방정식 $(Q\eta')' - P\eta = 0$의 해가 구간 $(a, b)$에서 영점을 갖지 않으면 강한 극소의 충분 조건.

### 직접법

- **리츠법 (Ritz method)**: 범함수를 유한 차원 부분공간에서 최소화
- **갈레르킨법 (Galerkin method)**: 잔차(residual)를 시험함수와 직교하게 설정
- **유한요소법 (FEM)**: 구간별(piecewise) 기저함수를 사용하는 리츠/갈레르킨법

### 변분 부등식

구속 $y(x) \geq \psi(x)$ (장애물 문제) 등, 등호가 아닌 부등호 조건이 있는 문제. 자유 경계 문제와 밀접히 관련된다.

### 참고 문헌

1. **Boas, M. L.** *Mathematical Methods in the Physical Sciences*, 3rd ed., Ch. 9
2. **Gelfand, I. M. & Fomin, S. V.** *Calculus of Variations* (Dover) -- 변분법의 고전적 교과서
3. **Goldstein, Poole, Safko** *Classical Mechanics*, 3rd ed. -- 라그랑주/해밀턴 역학
4. **Lanczos, C.** *The Variational Principles of Mechanics* (Dover) -- 변분 원리의 물리학적 해석
5. **Arfken, Weber** *Mathematical Methods for Physicists*, 7th ed., Ch. 22

### 핵심 공식 요약

| 공식 | 설명 |
|------|------|
| $\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} = 0$ | 오일러-라그랑주 방정식 |
| $F - y'F_{y'} = C$ | 벨트라미 항등식 ($F$가 $x$에 무관) |
| $F_{y'} = C$ | $F$가 $y$에 무관한 경우 |
| $\frac{d}{dt}\frac{\partial L}{\partial \dot{q}} - \frac{\partial L}{\partial q} = 0$ | 라그랑주 운동방정식 |
| $H = \sum p_i\dot{q}_i - L$ | 해밀토니안 정의 |
| $\dot{q} = \partial H/\partial p$, $\dot{p} = -\partial H/\partial q$ | 해밀턴 정준 방정식 |
| $x = R(\theta - \sin\theta)$, $y = R(1-\cos\theta)$ | 사이클로이드 (최속 강하선) |

---

**이전**: [16. 그린 함수](16_Greens_Functions.md)
**다음**: [18. 텐서 해석](18_Tensor_Analysis.md)
