# 상미분방정식 기초

## 개요

상미분방정식(ODE)은 하나의 독립변수에 대한 미분을 포함하는 방정식입니다. 물리적 시스템의 시간 변화를 기술하는 데 널리 사용됩니다.

---

## 1. ODE의 기본 개념

### 1.1 정의와 분류

```
일반적인 ODE:
F(t, y, y', y'', ..., y⁽ⁿ⁾) = 0

1차 ODE:
dy/dt = f(t, y)

n차 ODE:
d^n y/dt^n = f(t, y, y', ..., y^(n-1))
```

### 1.2 분류

```python
"""
1. 차수(Order): 가장 높은 미분의 차수
   - 1차: y' = f(t, y)
   - 2차: y'' = f(t, y, y')

2. 선형성(Linearity):
   - 선형: y'' + p(t)y' + q(t)y = g(t)
   - 비선형: y' = y²

3. 자율성(Autonomous):
   - 자율: y' = f(y) (t가 명시적으로 없음)
   - 비자율: y' = f(t, y)

4. 문제 유형:
   - 초기값 문제(IVP): y(t₀) = y₀ 주어짐
   - 경계값 문제(BVP): y(a) = α, y(b) = β 주어짐
"""
```

### 1.3 예시 ODE

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 인구 성장 모델 (지수 성장)
# dy/dt = ky, y(0) = y₀
# 해: y(t) = y₀ * e^(kt)

def population_growth():
    k = 0.1  # 성장률
    y0 = 100  # 초기 인구

    t = np.linspace(0, 50, 100)
    y = y0 * np.exp(k * t)

    plt.figure(figsize=(10, 4))
    plt.plot(t, y)
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(f'인구 성장 모델 (k={k})')
    plt.grid(True)
    plt.show()

population_growth()

# 2. 방사성 붕괴
# dN/dt = -λN, N(0) = N₀
# 해: N(t) = N₀ * e^(-λt)

def radioactive_decay():
    lambda_ = 0.05
    N0 = 1000
    half_life = np.log(2) / lambda_

    t = np.linspace(0, 100, 100)
    N = N0 * np.exp(-lambda_ * t)

    plt.figure(figsize=(10, 4))
    plt.plot(t, N)
    plt.axhline(y=N0/2, color='r', linestyle='--', label=f'반감기: {half_life:.1f}')
    plt.xlabel('Time')
    plt.ylabel('Number of atoms')
    plt.title('방사성 붕괴')
    plt.legend()
    plt.grid(True)
    plt.show()

radioactive_decay()
```

---

## 2. 해석적 해법

### 2.1 변수분리법

```python
"""
dy/dt = g(t)h(y) 형태

1/h(y) dy = g(t) dt
∫ 1/h(y) dy = ∫ g(t) dt

예시: dy/dt = ty
1/y dy = t dt
ln|y| = t²/2 + C
y = Ae^(t²/2)
"""

def separable_ode_example():
    # dy/dt = ty, y(0) = 1
    t = np.linspace(-2, 2, 100)
    y = np.exp(t**2 / 2)  # 해석적 해

    plt.figure(figsize=(8, 5))
    plt.plot(t, y)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("dy/dt = ty 의 해")
    plt.grid(True)
    plt.show()

separable_ode_example()
```

### 2.2 1차 선형 ODE

```python
"""
dy/dt + p(t)y = q(t)

적분인자: μ(t) = e^(∫p(t)dt)

해: y = (1/μ) * ∫ μ*q dt + C/μ

예시: dy/dt + 2y = e^(-t)
μ = e^(2t)
y = e^(-2t) * ∫ e^(2t) * e^(-t) dt
y = e^(-2t) * (e^t + C)
y = e^(-t) + Ce^(-2t)
"""

def linear_ode_example():
    t = np.linspace(0, 5, 100)
    C = 1  # 초기조건에서 결정
    y = np.exp(-t) + C * np.exp(-2*t)

    plt.figure(figsize=(8, 5))
    plt.plot(t, y)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("dy/dt + 2y = e^(-t) 의 해")
    plt.grid(True)
    plt.show()

linear_ode_example()
```

### 2.3 2차 상수계수 선형 ODE

```python
"""
ay'' + by' + cy = 0

특성방정식: ar² + br + c = 0

경우 1: 서로 다른 두 실근 r₁, r₂
  y = C₁e^(r₁t) + C₂e^(r₂t)

경우 2: 중근 r
  y = (C₁ + C₂t)e^(rt)

경우 3: 복소근 α ± βi
  y = e^(αt)(C₁cos(βt) + C₂sin(βt))
"""

def second_order_examples():
    t = np.linspace(0, 10, 200)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 경우 1: y'' - 3y' + 2y = 0 (r=1, 2)
    y1 = np.exp(t) - np.exp(2*t)
    axes[0].plot(t, y1)
    axes[0].set_title("서로 다른 실근")
    axes[0].set_xlabel('t')

    # 경우 2: y'' - 2y' + y = 0 (r=1 중근)
    y2 = (1 + t) * np.exp(t)
    axes[1].plot(t, y2)
    axes[1].set_title("중근")
    axes[1].set_xlabel('t')

    # 경우 3: y'' + y = 0 (r=±i)
    y3 = np.cos(t) + np.sin(t)
    axes[2].plot(t, y3)
    axes[2].set_title("복소근 (진동)")
    axes[2].set_xlabel('t')

    plt.tight_layout()
    plt.show()

second_order_examples()
```

---

## 3. 초기값 문제 (IVP)

### 3.1 존재성과 유일성

```python
"""
리프시츠 조건 (Lipschitz Condition):

|f(t, y₁) - f(t, y₂)| ≤ L|y₁ - y₂|

이 조건이 만족되면 해가 존재하고 유일함.

예외 케이스:
dy/dt = 3y^(2/3), y(0) = 0
해: y = 0 (자명해) 또는 y = t³ (유일하지 않음)
"""

def non_unique_solution():
    t = np.linspace(-2, 2, 100)

    # 두 가지 해가 모두 초기조건 y(0) = 0을 만족
    y1 = np.zeros_like(t)  # y = 0
    y2 = t**3  # y = t³

    plt.figure(figsize=(8, 5))
    plt.plot(t, y1, label='y = 0')
    plt.plot(t, y2, label='y = t³')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("dy/dt = 3y^(2/3) - 유일하지 않은 해")
    plt.legend()
    plt.grid(True)
    plt.show()

non_unique_solution()
```

### 3.2 고차 ODE를 1차 시스템으로 변환

```python
"""
n차 ODE를 n개의 1차 ODE 시스템으로 변환

예시: y'' + 4y' + 3y = 0

변환:
y₁ = y
y₂ = y' = y₁'

시스템:
y₁' = y₂
y₂' = -3y₁ - 4y₂

행렬 형태:
[y₁']   [0   1 ] [y₁]
[y₂'] = [-3 -4] [y₂]
"""

def convert_to_system():
    # 2차 ODE: y'' + 4y' + 3y = 0
    # 초기조건: y(0) = 1, y'(0) = 0

    # 1차 시스템으로 해석
    # 특성 방정식: r² + 4r + 3 = 0 → r = -1, -3
    # 해: y = Ae^(-t) + Be^(-3t)
    # 초기조건 적용: A + B = 1, -A - 3B = 0
    # A = 3/2, B = -1/2

    t = np.linspace(0, 5, 100)
    y = 1.5 * np.exp(-t) - 0.5 * np.exp(-3*t)
    y_prime = -1.5 * np.exp(-t) + 1.5 * np.exp(-3*t)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t, y, label='y(t)')
    axes[0].plot(t, y_prime, label="y'(t)")
    axes[0].set_xlabel('t')
    axes[0].set_title('시간 도메인')
    axes[0].legend()
    axes[0].grid(True)

    # 위상 평면
    axes[1].plot(y, y_prime)
    axes[1].set_xlabel('y')
    axes[1].set_ylabel("y'")
    axes[1].set_title('위상 평면')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

convert_to_system()
```

---

## 4. 위상 평면 분석

### 4.1 평형점과 안정성

```python
"""
dy/dt = f(y)에서 평형점: f(y*) = 0

안정성:
- f'(y*) < 0: 안정 (수렴)
- f'(y*) > 0: 불안정 (발산)
- f'(y*) = 0: 추가 분석 필요
"""

def equilibrium_analysis():
    # dy/dt = y(1 - y) (로지스틱 성장)
    # 평형점: y* = 0 또는 y* = 1

    y = np.linspace(-0.5, 1.5, 100)
    f = y * (1 - y)

    plt.figure(figsize=(10, 5))

    # dy/dt vs y
    plt.subplot(1, 2, 1)
    plt.plot(y, f)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.scatter([0, 1], [0, 0], color='red', s=100, zorder=5)
    plt.xlabel('y')
    plt.ylabel('dy/dt')
    plt.title('dy/dt = y(1-y)')
    plt.grid(True)

    # f'(y) = 1 - 2y
    # f'(0) = 1 > 0: 불안정
    # f'(1) = -1 < 0: 안정

    # 시간 진화
    plt.subplot(1, 2, 2)
    from scipy.integrate import solve_ivp

    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 100)

    for y0 in [-0.1, 0.1, 0.5, 0.9, 1.1, 1.5]:
        sol = solve_ivp(lambda t, y: y*(1-y), t_span, [y0], t_eval=t_eval)
        plt.plot(sol.t, sol.y[0], label=f'y₀={y0}')

    plt.axhline(y=1, color='r', linestyle='--', label='안정 평형')
    plt.axhline(y=0, color='b', linestyle='--', label='불안정 평형')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend(loc='right')
    plt.title('다양한 초기조건에서의 해')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

equilibrium_analysis()
```

### 4.2 2D 위상 평면

```python
def phase_plane_2d():
    """2차원 시스템의 위상 평면"""
    # 단순 조화 진동자: x'' + x = 0
    # 시스템: x' = v, v' = -x

    def harmonic_oscillator(t, state):
        x, v = state
        return [v, -x]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 벡터장
    x_range = np.linspace(-2, 2, 15)
    v_range = np.linspace(-2, 2, 15)
    X, V = np.meshgrid(x_range, v_range)
    dX = V
    dV = -X

    axes[0].quiver(X, V, dX, dV, alpha=0.5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('v')
    axes[0].set_title('벡터장')
    axes[0].set_aspect('equal')

    # 궤적
    from scipy.integrate import solve_ivp

    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 200)

    for x0, v0 in [(1, 0), (0, 1), (1.5, 0.5)]:
        sol = solve_ivp(harmonic_oscillator, t_span, [x0, v0], t_eval=t_eval)
        axes[1].plot(sol.y[0], sol.y[1], label=f'({x0}, {v0})')

    axes[1].set_xlabel('x')
    axes[1].set_ylabel('v')
    axes[1].set_title('위상 궤적')
    axes[1].legend()
    axes[1].set_aspect('equal')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

phase_plane_2d()
```

---

## 5. 물리적 예제

### 5.1 자유 낙하

```python
def free_fall():
    """중력 하의 자유 낙하 (공기 저항 포함)"""
    from scipy.integrate import solve_ivp

    # 파라미터
    g = 9.8  # 중력 가속도
    k = 0.1  # 공기 저항 계수
    m = 1.0  # 질량

    # 운동 방정식: m*dv/dt = mg - kv²
    # dv/dt = g - (k/m)v²

    def fall_with_drag(t, state):
        v = state[0]
        return [g - (k/m) * v * abs(v)]

    # 종단 속도: v_terminal = sqrt(mg/k)
    v_terminal = np.sqrt(m * g / k)

    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 200)

    sol = solve_ivp(fall_with_drag, t_span, [0], t_eval=t_eval)

    plt.figure(figsize=(10, 5))
    plt.plot(sol.t, sol.y[0])
    plt.axhline(y=v_terminal, color='r', linestyle='--',
                label=f'종단 속도: {v_terminal:.2f} m/s')
    plt.xlabel('시간 (s)')
    plt.ylabel('속도 (m/s)')
    plt.title('공기 저항이 있는 자유 낙하')
    plt.legend()
    plt.grid(True)
    plt.show()

free_fall()
```

### 5.2 RC 회로

```python
def rc_circuit():
    """RC 회로의 과도 응답"""
    from scipy.integrate import solve_ivp

    # 파라미터
    R = 1000  # 저항 (Ω)
    C = 1e-6  # 커패시턴스 (F)
    V_source = 5  # 전원 전압 (V)
    tau = R * C  # 시정수

    # 충전: V_C' = (V_source - V_C) / (RC)
    def charging(t, V_C):
        return [(V_source - V_C[0]) / (R * C)]

    t_span = (0, 5 * tau)
    t_eval = np.linspace(0, 5*tau, 200)

    sol = solve_ivp(charging, t_span, [0], t_eval=t_eval)

    # 해석적 해
    t_analytical = np.linspace(0, 5*tau, 200)
    V_analytical = V_source * (1 - np.exp(-t_analytical / tau))

    plt.figure(figsize=(10, 5))
    plt.plot(sol.t * 1000, sol.y[0], 'b-', label='수치 해')
    plt.plot(t_analytical * 1000, V_analytical, 'r--', label='해석적 해')
    plt.axhline(y=V_source * 0.632, color='g', linestyle=':',
                label=f'τ = {tau*1000:.3f} ms')
    plt.xlabel('시간 (ms)')
    plt.ylabel('커패시터 전압 (V)')
    plt.title('RC 회로 충전')
    plt.legend()
    plt.grid(True)
    plt.show()

rc_circuit()
```

---

## 연습 문제

### 문제 1
냉각의 법칙: dT/dt = -k(T - T_ambient)
초기 온도 90°C, 주변 온도 20°C, k=0.1일 때 시간에 따른 온도 변화를 그리세요.

```python
def exercise_1():
    from scipy.integrate import solve_ivp

    T_ambient = 20
    k = 0.1
    T0 = 90

    def cooling(t, T):
        return [-k * (T[0] - T_ambient)]

    sol = solve_ivp(cooling, (0, 50), [T0], t_eval=np.linspace(0, 50, 100))

    plt.figure(figsize=(10, 5))
    plt.plot(sol.t, sol.y[0])
    plt.axhline(y=T_ambient, color='r', linestyle='--')
    plt.xlabel('시간')
    plt.ylabel('온도 (°C)')
    plt.title('뉴턴의 냉각 법칙')
    plt.grid(True)
    plt.show()

exercise_1()
```

### 문제 2
감쇠 진동: x'' + 2γx' + ω₀²x = 0
ω₀ = 2, γ = 0.5일 때 위상 평면과 시간 응답을 그리세요.

```python
def exercise_2():
    from scipy.integrate import solve_ivp

    omega0 = 2
    gamma = 0.5

    def damped_oscillator(t, state):
        x, v = state
        return [v, -2*gamma*v - omega0**2*x]

    sol = solve_ivp(damped_oscillator, (0, 10), [1, 0],
                    t_eval=np.linspace(0, 10, 200))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(sol.t, sol.y[0])
    axes[0].set_xlabel('시간')
    axes[0].set_ylabel('x')
    axes[0].set_title('시간 응답')
    axes[0].grid(True)

    axes[1].plot(sol.y[0], sol.y[1])
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('v')
    axes[1].set_title('위상 평면')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

exercise_2()
```

---

## 요약

| 개념 | 내용 |
|------|------|
| ODE 분류 | 차수, 선형성, 자율성 |
| 해석적 해법 | 변수분리, 적분인자, 특성방정식 |
| 고차→1차 변환 | n차 ODE → n개 1차 시스템 |
| 위상 평면 | 평형점, 안정성, 궤적 분석 |
| 존재성/유일성 | 리프시츠 조건 |
