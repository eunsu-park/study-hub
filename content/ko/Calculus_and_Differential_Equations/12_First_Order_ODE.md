# 12. 1계 상미분방정식(First-Order Ordinary Differential Equations)

## 학습 목표

- ODE를 차수, 선형성, 유형별로 분류하고, 해가 무엇을 의미하는지 설명한다
- 방향장(direction field)을 통해 해의 거동을 시각화하고 오일러 방법(Euler's method)으로 근사 해를 구한다
- 분리변수법(separable), 선형(linear), 완전(exact) 1계 ODE를 해석적 기법으로 푼다
- 피카르-린델뢰프 정리(Picard-Lindelof theorem)를 서술하고 유일성이 실패하는 경우를 식별한다
- 실세계 현상(인구 성장, 혼합, 냉각)을 1계 ODE로 모델링하고 수치적으로 해를 검증한다

---

## 1. ODE란 무엇인가?

**상미분방정식(ordinary differential equation, ODE)**은 함수 $y(x)$와 그 도함수를 포함하는 방정식이다:

$$F\!\left(x,\, y,\, y',\, y'',\, \ldots,\, y^{(n)}\right) = 0$$

- **차수(Order):** 나타나는 가장 높은 도함수. $y' + y = 0$은 1계; $y'' + y = 0$은 2계이다.
- **선형(Linear):** 미지 함수 $y$와 그 도함수가 1차 거듭제곱으로 나타나고 서로 곱해지지 않는다. $y' + xy = x^2$은 선형; $y' + y^2 = 0$은 비선형이다.
- **해(Solution):** 어떤 구간에서 방정식을 만족하는 함수 $y(x)$. **초기값 문제(Initial Value Problem, IVP)**는 조건 $y(x_0) = y_0$을 추가한다.

**비유:** ODE는 "변화율에 관한 규칙"이다. $y$가 은행 잔고라면, $y' = 0.05y$는 "성장률이 현재 잔고의 5%"라고 말한다 -- 이것은 연속 복리의 ODE이다.

**왜 ODE를 공부하는가?** 과학의 거의 모든 동역학 시스템은 미분방정식으로 기술된다: 인구 성장, 방사성 붕괴, 전기 회로, 행성 궤도, 화학 반응, 신경망 등.

---

## 2. 방향장과 오일러 방법(Direction Fields and Euler's Method)

### 2.1 방향장

ODE를 해석적으로 풀기 전에 해를 **시각화**할 수 있다. $y' = f(x, y)$에서, 모든 점 $(x, y)$에서 방정식이 기울기를 알려준다. 이러한 기울기를 가진 짧은 선분을 그리면 **방향장(direction field)**(또는 기울기장)이 만들어진다.

**방향장 읽는 법:** 장 위의 아무 곳에 공을 놓으면, 기울기를 따라 "굴러가면서" 해 곡선을 그린다. 서로 다른 시작점은 서로 다른 해를 주며 -- 장은 모든 해를 동시에 보여준다.

### 2.2 오일러 방법

가장 간단한 수치적 방법: 접선을 따라 앞으로 진행한다.

$y' = f(x, y)$, $y(x_0) = y_0$, 보폭 $h$가 주어졌을 때:

$$y_{n+1} = y_n + h \cdot f(x_n, y_n), \quad x_{n+1} = x_n + h$$

- $h$: 보폭(step size) (작을수록 정확하지만 계산량이 많다)
- $f(x_n, y_n)$: 현재 점에서의 기울기
- $y_{n+1}$: 다음 단계에서의 예측값

**직관:** 접선을 따라 작은 걸음으로 걷고, 그런 다음 기울기를 다시 평가한다. 이것은 현재 위치의 방향만 보여주는 나침반으로 항해하는 것과 같다 -- 계속 확인하고 조정한다.

**오차:** 오일러 방법은 **1차 정확도(first-order accuracy)**를 가진다: 전역 오차는 $O(h)$이다. 보폭을 반으로 줄이면 오차도 반으로 줄어든다. 룽게-쿠타(Runge-Kutta)와 같은 더 정교한 방법은 $O(h^4)$를 달성한다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Direction field and Euler's method ---

def f(x, y):
    """RHS of ODE: y' = x - y (a simple linear first-order ODE)."""
    return x - y

# Direction field
x_grid = np.linspace(-2, 4, 25)
y_grid = np.linspace(-2, 4, 25)
X, Y = np.meshgrid(x_grid, y_grid)
slopes = f(X, Y)

# Normalize arrows for uniform length (direction only)
magnitude = np.sqrt(1 + slopes**2)
U = 1 / magnitude
V = slopes / magnitude

fig, ax = plt.subplots(figsize=(10, 8))
ax.quiver(X, Y, U, V, color='lightblue', alpha=0.6, scale=30)

# Euler's method with different step sizes
def euler_method(f, x0, y0, h, x_end):
    """Solve y' = f(x, y) using Euler's method."""
    xs = [x0]
    ys = [y0]
    x, y = x0, y0
    while x < x_end - 1e-12:
        y = y + h * f(x, y)
        x = x + h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Exact solution for y' = x - y, y(0) = 0: y = x - 1 + e^(-x)
x_exact = np.linspace(0, 4, 200)
y_exact = x_exact - 1 + np.exp(-x_exact)

ax.plot(x_exact, y_exact, 'k-', linewidth=2.5, label='Exact solution')

for h, color in [(0.5, 'red'), (0.2, 'orange'), (0.05, 'green')]:
    xe, ye = euler_method(f, 0, 0, h, 4)
    ax.plot(xe, ye, 'o-', color=color, markersize=3, label=f'Euler h={h}')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Direction Field and Euler's Method for $y' = x - y$")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 4)
ax.set_ylim(-2, 4)
plt.tight_layout()
plt.show()
```

---

## 3. 분리변수 방정식(Separable Equations)

1계 ODE가 **분리 가능(separable)**하다 함은 다음과 같이 쓸 수 있는 것이다:

$$\frac{dy}{dx} = g(x)\,h(y)$$

**풀이 방법:** 변수를 분리하고 양변을 적분한다:

$$\int \frac{dy}{h(y)} = \int g(x)\, dx + C$$

**생각하는 방법:** 모든 $y$를 한쪽으로, 모든 $x$를 다른 쪽으로 옮기는 것이다.

### 3.1 예제: 지수 성장과 감소

$$\frac{dy}{dx} = ky$$

분리: $\frac{dy}{y} = k\,dx$. 적분: $\ln|y| = kx + C$, 따라서 $y = y_0 e^{kx}$.

- $k > 0$: 지수 성장 (인구, 복리)
- $k < 0$: 지수 감소 (방사성 붕괴, 뉴턴 냉각)

### 3.2 예제: 로지스틱 성장(Logistic Growth)

$$\frac{dP}{dt} = rP\!\left(1 - \frac{P}{K}\right)$$

- $P$: 인구
- $r$: 내재적 성장률
- $K$: 환경 수용력(carrying capacity, 지속 가능한 최대 인구)

분리하고 부분 분수를 사용하면:

$$P(t) = \frac{K}{1 + \left(\frac{K}{P_0} - 1\right)e^{-rt}}$$

인구는 처음에 지수적으로 성장하다가 $K$에서 평탄해진다 -- S자 형태(시그모이드) 곡선이다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, Function, dsolve, Eq, exp

# --- Logistic growth: analytical and numerical ---
t_sym = symbols('t')
P = Function('P')
r, K, P0 = 0.5, 1000, 10

# Analytical solution via SymPy
ode = Eq(P(t_sym).diff(t_sym), r * P(t_sym) * (1 - P(t_sym) / K))
sol = dsolve(ode, P(t_sym), ics={P(0): P0})
print(f"Analytical solution: {sol}")

# Numerical solution via SciPy
t_span = (0, 25)
t_eval = np.linspace(0, 25, 300)

def logistic(t, y):
    """Logistic growth ODE: dP/dt = r*P*(1 - P/K)."""
    return r * y[0] * (1 - y[0] / K)

result = solve_ivp(logistic, t_span, [P0], t_eval=t_eval)

# Analytical formula
P_exact = K / (1 + (K / P0 - 1) * np.exp(-r * t_eval))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_eval, P_exact, 'b-', linewidth=2, label='Analytical')
ax.plot(result.t, result.y[0], 'r--', linewidth=2, label='Numerical (solve_ivp)')
ax.axhline(y=K, color='gray', linestyle=':', label=f'Carrying capacity K={K}')
ax.set_xlabel('Time t')
ax.set_ylabel('Population P(t)')
ax.set_title('Logistic Growth Model')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
```

---

## 4. 1계 선형 ODE(Linear First-Order ODEs)

### 4.1 표준형과 적분인자

**1계 선형 ODE**의 표준형:

$$\frac{dy}{dx} + P(x)\,y = Q(x)$$

**적분인자(integrating factor)** 방법은 체계적이며 항상 작동한다:

1. 적분인자 계산: $\mu(x) = e^{\int P(x)\,dx}$
2. 양변에 $\mu$를 곱한다: $\frac{d}{dx}[\mu\,y] = \mu\,Q$
3. 적분: $\mu\,y = \int \mu\,Q\, dx + C$
4. $y$에 대해 풀기: $y = \frac{1}{\mu}\left[\int \mu\,Q\, dx + C\right]$

**왜 이것이 작동하는가?** $\mu$를 곱하면 좌변이 **완전 도함수**(곱의 법칙의 역)가 된다. $\mu = e^{\int P\,dx}$의 마법은 $\mu' = P\mu$인데, 이것이 바로 필요한 것이다.

### 4.2 예제: 혼합 문제

탱크에 처음 100 L의 순수한 물이 있다. 2 g/L의 소금물이 3 L/분의 속도로 유입되고, 잘 혼합된 용액이 3 L/분의 속도로 유출된다.

$y(t)$ = 시간 $t$에서의 소금 그램수라 하자. 부피는 100 L로 유지된다.

$$\frac{dy}{dt} = \underbrace{2 \cdot 3}_{\text{소금 유입}} - \underbrace{\frac{y}{100} \cdot 3}_{\text{소금 유출}} = 6 - \frac{3y}{100}$$

표준형: $y' + 0.03y = 6$

적분인자: $\mu = e^{0.03t}$

$$y(t) = 200(1 - e^{-0.03t})$$

$t \to \infty$이면, $y \to 200$ g (평형: 농도가 2 g/L에 접근).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Mixing problem ---
def mixing(t, y):
    """dy/dt = rate_in - rate_out = 6 - 0.03*y."""
    return 6 - 0.03 * y[0]

t_span = (0, 200)
t_eval = np.linspace(0, 200, 500)
result = solve_ivp(mixing, t_span, [0], t_eval=t_eval)

# Analytical solution
y_exact = 200 * (1 - np.exp(-0.03 * t_eval))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Salt amount over time
ax1.plot(t_eval, y_exact, 'b-', linewidth=2, label='Salt amount y(t)')
ax1.axhline(y=200, color='red', linestyle='--', label='Equilibrium (200 g)')
ax1.set_xlabel('Time (min)')
ax1.set_ylabel('Salt (g)')
ax1.set_title('Mixing Problem: Salt Amount')
ax1.legend()
ax1.grid(True)

# Concentration over time
conc = y_exact / 100  # g/L
ax2.plot(t_eval, conc, 'g-', linewidth=2, label='Concentration (g/L)')
ax2.axhline(y=2.0, color='red', linestyle='--', label='Inflow concentration (2 g/L)')
ax2.set_xlabel('Time (min)')
ax2.set_ylabel('Concentration (g/L)')
ax2.set_title('Mixing Problem: Concentration')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

---

## 5. 완전 방정식(Exact Equations)

### 5.1 정의

방정식 $M(x,y)\,dx + N(x,y)\,dy = 0$이 **완전(exact)**하다 함은 함수 $\Psi(x,y)$가 존재하여:

$$\frac{\partial\Psi}{\partial x} = M, \quad \frac{\partial\Psi}{\partial y} = N$$

해는 음함수 방정식 $\Psi(x, y) = C$이다.

### 5.2 완전성 판정

클레로 정리(Clairaut's theorem)에 의해, 방정식이 완전할 필요충분조건은:

$$\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}$$

### 5.3 풀이 방법

완전한 경우:
1. $M$을 $x$에 대해 적분: $\Psi = \int M\,dx + g(y)$
2. $y$에 대해 미분하고 $N$과 같다고 놓기: $\frac{\partial}{\partial y}\int M\,dx + g'(y) = N$
3. $g'(y)$를 구하고 적분하여 $g(y)$를 찾기

**예제:** $(2xy + 3)\,dx + (x^2 + 4y)\,dy = 0$

확인: $M_y = 2x = N_x$ -- 완전.

$\Psi = \int (2xy + 3)\,dx = x^2 y + 3x + g(y)$

$\Psi_y = x^2 + g'(y) = x^2 + 4y \implies g'(y) = 4y \implies g(y) = 2y^2$

해: $x^2 y + 3x + 2y^2 = C$

```python
from sympy import symbols, Function, Eq, dsolve, classify_ode

x = symbols('x')
y = Function('y')

# --- Exact equation solved with SymPy ---
# (2xy + 3) + (x^2 + 4y) y' = 0
# Rewrite as: y' = -(2xy + 3) / (x^2 + 4y)

ode = Eq(y(x).diff(x), -(2*x*y(x) + 3) / (x**2 + 4*y(x)))

# SymPy can identify and solve exact equations
classifications = classify_ode(ode, y(x))
print(f"ODE classification: {classifications}")

sol = dsolve(ode, y(x))
print(f"Solution: {sol}")
```

---

## 6. 존재성과 유일성(Existence and Uniqueness)

### 6.1 피카르-린델뢰프 정리(The Picard-Lindelof Theorem)

**정리:** IVP $y' = f(x, y)$, $y(x_0) = y_0$에 대해:

$f$와 $\partial f / \partial y$가 $(x_0, y_0)$를 포함하는 직사각형 $R$에서 연속이면, $x_0$의 어떤 근방에서 **유일한** 해가 존재한다.

핵심 조건은 **립시츠 조건(Lipschitz condition)**이다:

$$|f(x, y_1) - f(x, y_2)| \le L |y_1 - y_2|$$

어떤 상수 $L$에 대해. $\partial f/\partial y$가 $R$에서 유계이면, 립시츠 조건은 자동으로 성립한다 (평균값 정리에 의해).

### 6.2 유일성이 실패하는 경우

**고전적 반례:** $y' = y^{1/3}$, $y(0) = 0$

여기서 $f(x, y) = y^{1/3}$이고, $\partial f / \partial y = \frac{1}{3}y^{-2/3} \to \infty$ ($y \to 0$일 때).

$y(x) = 0$과 $y(x) = \left(\frac{2x}{3}\right)^{3/2}$ 모두 $(0, 0)$을 지나는 유효한 해이다 -- 유일성이 실패하는 이유는 정확히 립시츠 조건이 초기점에서 깨지기 때문이다.

### 6.3 이것이 중요한 이유

존재성과 유일성은 단순히 이론적 호기심이 아니다:
- **수치적 방법**은 수렴을 보장하기 위해 유일성에 의존한다
- **물리적 모델**은 유일한 해를 가져야 한다 (공은 두 곳에 있을 수 없다)
- 유일성이 실패하면, 모델을 다듬을 필요가 있을 수 있다 (예: 특이점에서의 물리가 올바르게 포착되지 않았다)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Uniqueness failure: y' = y^(1/3), y(0) = 0 ---
# Two valid solutions: y=0 and y = (2x/3)^(3/2)

x_vals = np.linspace(-0.5, 3, 200)

# Solution 1: trivial
y_trivial = np.zeros_like(x_vals)

# Solution 2: nontrivial (only valid for x >= 0)
y_nontrivial = np.where(x_vals >= 0,
                         (2 * x_vals / 3)**1.5,
                         0)

# What does a numerical solver do?
def ode_func(x, y):
    """y' = y^(1/3), handling y=0 carefully."""
    return np.sign(y) * np.abs(y)**(1/3)

# Starting exactly at y=0, the solver might find either solution
result = solve_ivp(ode_func, (0, 3), [0.0], t_eval=np.linspace(0, 3, 200),
                   method='RK45', max_step=0.01)

# Starting slightly above 0
result2 = solve_ivp(ode_func, (0, 3), [1e-10], t_eval=np.linspace(0, 3, 200),
                    method='RK45', max_step=0.01)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_vals, y_trivial, 'b-', linewidth=2, label='$y = 0$ (trivial)')
ax.plot(x_vals, y_nontrivial, 'r-', linewidth=2,
        label='$y = (2x/3)^{3/2}$ (nontrivial)')
ax.plot(result.t, result.y[0], 'g--', linewidth=2,
        label='Numerical from $y(0)=0$')
ax.plot(result2.t, result2.y[0], 'm:', linewidth=2,
        label='Numerical from $y(0)=10^{-10}$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Uniqueness Failure: $y' = y^{1/3}$, $y(0) = 0$\n"
             "Two valid solutions through the same point!")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
```

---

## 7. 응용

### 7.1 뉴턴의 냉각 법칙(Newton's Law of Cooling)

$$\frac{dT}{dt} = -k(T - T_{\text{env}})$$

- $T(t)$: 물체 온도
- $T_{\text{env}}$: 주변 온도 (상수)
- $k > 0$: 냉각 상수

해: $T(t) = T_{\text{env}} + (T_0 - T_{\text{env}})e^{-kt}$

온도는 주변 온도에 지수적으로 접근한다.

### 7.2 RC 회로

$$R\frac{dQ}{dt} + \frac{Q}{C} = V(t)$$

- $Q$: 축전기(capacitor)의 전하
- $R$: 저항, $C$: 전기용량
- $V(t)$: 전원 전압

이것은 시간 상수 $\tau = RC$를 가진 1계 선형 ODE이다.

### 7.3 종단 속도(Terminal Velocity)

항력이 있는 물체의 낙하:

$$m\frac{dv}{dt} = mg - bv$$

여기서 $b$는 항력 계수이다. 해: $v(t) = \frac{mg}{b}(1 - e^{-bt/m})$

종단 속도: $v_{\text{term}} = mg/b$ (항력이 중력과 같아질 때).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Three first-order ODE applications ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
t_eval = np.linspace(0, 10, 200)

# 1. Newton's Cooling: hot coffee cooling from 90C in 20C room
T_env, T0, k = 20, 90, 0.3
T_cool = T_env + (T0 - T_env) * np.exp(-k * t_eval)

axes[0].plot(t_eval, T_cool, 'r-', linewidth=2)
axes[0].axhline(y=T_env, color='blue', linestyle='--', label=f'$T_{{env}}={T_env}$°C')
axes[0].set_xlabel('Time (min)')
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title("Newton's Law of Cooling")
axes[0].legend()
axes[0].grid(True)

# 2. RC circuit: charging capacitor (V=5V, R=1kΩ, C=1mF)
R_val, C_val, V0 = 1000, 1e-3, 5.0
tau = R_val * C_val  # time constant = 1 s
t_rc = np.linspace(0, 5 * tau, 200)
Q_charge = C_val * V0 * (1 - np.exp(-t_rc / tau))
V_cap = Q_charge / C_val

axes[1].plot(t_rc, V_cap, 'b-', linewidth=2)
axes[1].axhline(y=V0, color='red', linestyle='--', label=f'$V_0={V0}$ V')
axes[1].axvline(x=tau, color='gray', linestyle=':', label=f'$\\tau={tau}$ s')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Capacitor Voltage (V)')
axes[1].set_title('RC Circuit Charging')
axes[1].legend()
axes[1].grid(True)

# 3. Terminal velocity: skydiver (m=80kg, b=15 kg/s)
m, b_drag, g = 80, 15, 9.8
v_term = m * g / b_drag
t_fall = np.linspace(0, 40, 200)
v_fall = v_term * (1 - np.exp(-b_drag * t_fall / m))

axes[2].plot(t_fall, v_fall, 'g-', linewidth=2)
axes[2].axhline(y=v_term, color='red', linestyle='--',
                label=f'$v_{{term}}={v_term:.1f}$ m/s')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Velocity (m/s)')
axes[2].set_title('Terminal Velocity (Skydiver)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
```

---

## 8. 교차 참조

- **Mathematical Methods 레슨 09**는 더 고급의 1계 ODE 기법(베르누이 방정식, 치환법)과 존재 정리의 심화 다룸을 제공한다.
- **레슨 13 (2계 ODE)**은 여기서의 방법을 2계 방정식으로 확장하며, 진동에 대한 물리적 응용을 다룬다.
- **Numerical Simulation 레슨 01**은 더 정교한 수치적 ODE 풀이기(룽게-쿠타, 적응적 방법)를 다룬다.

---

## 연습 문제

**1.** 분리 가능한 ODE $\frac{dy}{dx} = \frac{x(1 + y^2)}{y(1 + x^2)}$, $y(0) = 1$을 풀어라.

**2.** 탱크에 500 L의 물과 10 kg의 용해된 소금이 있다. 순수한 물이 5 L/분으로 유입되고 잘 혼합된 용액이 5 L/분으로 유출된다. 시간 $t$에서의 소금량 $y(t)$를 구하고, 소금 농도가 0.01 kg/L 이하로 떨어지는 데 걸리는 시간을 결정하라.

**3.** 방정식 $(3x^2 + y\cos x)\,dx + (\sin x - 4y^3)\,dy = 0$이 완전한지 판별하라. 완전하다면, 해를 구하라.

**4.** $y' = y - x^2 + 1$, $y(0) = 0.5$를 $[0, 2]$에서 보폭 $h = 0.2$와 $h = 0.1$로 오일러 방법을 적용하라. 정확한 해 $y = (x + 1)^2 - 0.5e^x$과 비교하고 $x = 2$에서의 오차를 계산하라.

**5.** 95°C의 커피 한 잔이 22°C의 방에 놓인다. 5분 후 온도는 70°C이다.
   - (a) 냉각 상수 $k$를 구하라.
   - (b) 커피가 40°C(마시기 적합한 온도)에 도달하는 시점은?
   - (c) `solve_ivp`를 사용하여 냉각 곡선을 그리고 해석적으로 검증하라.

---

## 참고 자료

- **William E. Boyce & Richard C. DiPrima**, *Elementary Differential Equations and Boundary Value Problems*, 11th Edition, Chapters 1-2
- **James Stewart**, *Calculus: Early Transcendentals*, 9th Edition, Chapter 9
- **Erwin Kreyszig**, *Advanced Engineering Mathematics*, 10th Edition, Chapter 1
- **SciPy solve_ivp**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- **3Blue1Brown**, "Differential Equations" (시각적 직관)

---

[이전: 벡터 미적분](./11_Vector_Calculus.md) | [다음: 2계 상미분방정식](./13_Second_Order_ODE.md)
