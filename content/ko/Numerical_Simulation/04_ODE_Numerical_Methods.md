# ODE 수치해법

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 전진 오일러 방법(Forward Euler Method)을 구현하고 테일러 전개로부터의 유도 과정을 설명할 수 있습니다.
2. 4차 룽게-쿠타(RK4, 4th-order Runge-Kutta) 방법을 구현하고 오일러 방법보다 정확도가 높은 이유를 설명할 수 있습니다.
3. 명시적 방법(Explicit Method)(오일러, RK4)을 국소 절단 오차(Local Truncation Error), 전역 오차(Global Error), 계산 비용 측면에서 비교할 수 있습니다.
4. ODE 솔버에서 정확도와 효율성을 균형 있게 조절하는 적응적 스텝 크기 제어(Adaptive Step-size Control)를 적용할 수 있습니다.
5. 표준 ODE 초기값 문제에 대해 적절한 솔버 설정으로 `scipy.integrate.solve_ivp`를 사용할 수 있습니다.

---

## 왜 수치적 ODE 방법이 필요한가?

행성 궤도에서 화학 반응 속도론에 이르기까지 과학과 공학에서 발생하는 대부분의 미분 방정식은 해석적 해(closed-form solution)가 없습니다. 수치적 방법은 작은 시간 증분으로 전진하면서 각 단계에서 기울기를 추정하고 이를 사용하여 다음 값을 예측함으로써 해를 근사합니다.

### 전진 오일러: 테일러 전개로부터의 유도

가장 간단한 접근법은 y(t+h)의 테일러 전개(Taylor expansion)에서 시작합니다:

```
y(t + h) = y(t) + h·y'(t) + (h²/2)·y''(ξ)     ξ ∈ [t, t+h]인 어떤 값
```

O(h²) 나머지 항을 버리면 **전진 오일러 공식(forward Euler formula)**이 됩니다:

```
y_{n+1} = y_n + h·f(t_n, y_n)
```

국소 절단 오차(local truncation error)는 O(h²)(버린 항)이지만, N = T/h 단계에 걸쳐 오차가 누적되므로 **전역 오차(global error)는 O(h)**입니다 — 스텝 크기를 반으로 줄이면 오차도 반으로 줄어듭니다.

### RK4: 왜 1/6, 2/6, 2/6, 1/6 가중치인가?

4차 룽게-쿠타(Runge-Kutta) 방법은 각 단계 내 네 지점에서 기울기를 샘플링합니다 — 시작점(k1), 두 중간점(k2, k3), 끝점(k4) — 그리고 가중 평균을 취합니다:

```
y_{n+1} = y_n + (h/6)·(k1 + 2·k2 + 2·k3 + k4)
```

1-2-2-1 가중치는 수치 적분의 **심프슨 법칙(Simpson's rule)**과 동일합니다: 끝점은 가중치 1, 내부 점은 가중치 2. 이것이 O(h⁴) 정확도를 달성합니다 — 10배 작은 스텝이 10,000배 작은 오차를 의미합니다 — 이것이 RK4가 과학 계산의 주력 방법인 이유입니다.

### 암시적 방법과 강성(Stiffness)

시스템에 빠른 동역학과 느린 동역학이 모두 있을 때(예: 급속 반응과 점진적 반응이 있는 화학 반응), 전진 오일러 같은 양시적(explicit) 방법은 안정성을 유지하기 위해 극도로 작은 시간 단계를 요구합니다. **암시적 방법(implicit method)**(역방향 오일러 등)은 *다음* 시간 단계에서 f를 평가하므로 각 단계마다 비선형 풀이(일반적으로 뉴턴-랩슨)가 필요합니다. 단계당 추가 비용은 훨씬 큰 단계를 취할 수 있는 능력으로 상쇄됩니다 — 이것이 **강성(stiff)** 문제의 핵심 트레이드오프입니다.

---

## 개요

상미분방정식의 수치해법은 해석적 해를 구할 수 없거나 어려운 경우에 근사해를 계산합니다. 오일러 방법부터 4차 룽게-쿠타까지 주요 방법들을 학습합니다.

---

## 1. 오일러 방법 (Euler Method)

### 1.1 전진 오일러 (Forward Euler)

```python
import numpy as np
import matplotlib.pyplot as plt

def forward_euler(f, y0, t_span, n_steps):
    """
    전진 오일러 방법

    dy/dt = f(t, y)
    y_{n+1} = y_n + h * f(t_n, y_n)
    """
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]

    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        y[i+1] = y[i] + h * f(t[i], y[i])

    return t, y

# 테스트: dy/dt = -y, y(0) = 1
# 해석적 해: y = e^(-t)
f = lambda t, y: -y
y0 = 1.0
t_span = (0, 5)

# 다양한 스텝 수로 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

t_exact = np.linspace(0, 5, 200)
y_exact = np.exp(-t_exact)

for n in [10, 20, 50, 100]:
    t, y = forward_euler(f, y0, t_span, n)
    axes[0].plot(t, y, 'o-', markersize=3, label=f'n={n}')

axes[0].plot(t_exact, y_exact, 'k-', linewidth=2, label='정확해')
axes[0].set_xlabel('t')
axes[0].set_ylabel('y')
axes[0].set_title('전진 오일러 방법')
axes[0].legend()
axes[0].grid(True)

# 오차 분석
errors = []
n_values = [10, 20, 50, 100, 200, 500]
for n in n_values:
    t, y = forward_euler(f, y0, t_span, n)
    error = np.abs(y[-1] - np.exp(-5))
    errors.append(error)

axes[1].loglog(n_values, errors, 'bo-', label='실제 오차')
axes[1].loglog(n_values, [5/n for n in n_values], 'r--', label='O(h)')
axes[1].set_xlabel('스텝 수 n')
axes[1].set_ylabel('오차')
axes[1].set_title('오차 vs 스텝 수 (O(h) 수렴)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

### 1.2 후진 오일러 (Backward Euler)

```python
def backward_euler(f, df_dy, y0, t_span, n_steps, tol=1e-10):
    """
    후진 오일러 방법 (암시적)

    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})

    뉴턴-랩슨으로 암시적 방정식 풀이
    """
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]

    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        # 뉴턴-랩슨: g(y_{n+1}) = y_{n+1} - y_n - h*f(t_{n+1}, y_{n+1}) = 0
        y_guess = y[i]  # 초기 추측

        for _ in range(100):  # 최대 반복 횟수
            g = y_guess - y[i] - h * f(t[i+1], y_guess)
            dg = 1 - h * df_dy(t[i+1], y_guess)

            y_new = y_guess - g / dg

            if abs(y_new - y_guess) < tol:
                break
            y_guess = y_new

        y[i+1] = y_new

    return t, y

# 테스트: dy/dt = -y
f = lambda t, y: -y
df_dy = lambda t, y: -1  # ∂f/∂y

t_fw, y_fw = forward_euler(f, 1.0, (0, 5), 20)
t_bw, y_bw = backward_euler(f, df_dy, 1.0, (0, 5), 20)
t_exact = np.linspace(0, 5, 200)
y_exact = np.exp(-t_exact)

plt.figure(figsize=(10, 5))
plt.plot(t_fw, y_fw, 'bo-', label='전진 오일러')
plt.plot(t_bw, y_bw, 'rs-', label='후진 오일러')
plt.plot(t_exact, y_exact, 'k-', linewidth=2, label='정확해')
plt.xlabel('t')
plt.ylabel('y')
plt.title('전진 vs 후진 오일러')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 2. 룽게-쿠타 방법 (Runge-Kutta Methods)

### 2.1 RK2 (중점법, Heun)

```python
def rk2_midpoint(f, y0, t_span, n_steps):
    """RK2 중점법"""
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]

    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        y[i+1] = y[i] + h * k2

    return t, y

def rk2_heun(f, y0, t_span, n_steps):
    """RK2 Heun 방법 (수정 오일러)"""
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]

    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h, y[i] + h * k1)
        y[i+1] = y[i] + h/2 * (k1 + k2)

    return t, y

# 비교
f = lambda t, y: -y
t_span = (0, 5)
n = 20

t_euler, y_euler = forward_euler(f, 1.0, t_span, n)
t_mid, y_mid = rk2_midpoint(f, 1.0, t_span, n)
t_heun, y_heun = rk2_heun(f, 1.0, t_span, n)
t_exact = np.linspace(0, 5, 200)
y_exact = np.exp(-t_exact)

plt.figure(figsize=(10, 5))
plt.plot(t_euler, y_euler, 'o-', label='Euler', alpha=0.7)
plt.plot(t_mid, y_mid, 's-', label='RK2 중점법', alpha=0.7)
plt.plot(t_heun, y_heun, '^-', label='RK2 Heun', alpha=0.7)
plt.plot(t_exact, y_exact, 'k-', linewidth=2, label='정확해')
plt.xlabel('t')
plt.ylabel('y')
plt.title('오일러 vs RK2 비교')
plt.legend()
plt.grid(True)
plt.show()
```

### 2.2 RK4 (고전적 4차)

```python
def rk4(f, y0, t_span, n_steps):
    """
    고전적 4차 룽게-쿠타 방법

    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h*k1/2)
    k3 = f(t_n + h/2, y_n + h*k2/2)
    k4 = f(t_n + h, y_n + h*k3)

    y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]

    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)

        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    return t, y

# 정확도 비교
f = lambda t, y: -y
t_span = (0, 5)

methods = [
    ('Euler', forward_euler),
    ('RK2', rk2_heun),
    ('RK4', rk4)
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 해 비교
n = 10
for name, method in methods:
    t, y = method(f, 1.0, t_span, n)
    axes[0].plot(t, y, 'o-', label=name)

t_exact = np.linspace(0, 5, 200)
axes[0].plot(t_exact, np.exp(-t_exact), 'k-', linewidth=2, label='정확해')
axes[0].set_xlabel('t')
axes[0].set_ylabel('y')
axes[0].set_title(f'n={n} 스텝')
axes[0].legend()
axes[0].grid(True)

# 수렴 차수
n_values = [10, 20, 50, 100, 200]
for name, method in methods:
    errors = []
    for n in n_values:
        t, y = method(f, 1.0, t_span, n)
        error = np.abs(y[-1] - np.exp(-5))
        errors.append(error)
    axes[1].loglog(n_values, errors, 'o-', label=name)

# 이론적 수렴 차수
axes[1].loglog(n_values, [1/n for n in n_values], 'k--', alpha=0.5, label='O(h)')
axes[1].loglog(n_values, [1/n**2 for n in n_values], 'k:', alpha=0.5, label='O(h²)')
axes[1].loglog(n_values, [1/n**4 for n in n_values], 'k-.', alpha=0.5, label='O(h⁴)')

axes[1].set_xlabel('스텝 수 n')
axes[1].set_ylabel('오차')
axes[1].set_title('수렴 차수 비교')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

---

## 3. 시스템에 대한 적용

### 3.1 벡터화된 RK4

```python
def rk4_system(f, y0, t_span, n_steps):
    """
    연립 ODE 시스템을 위한 RK4

    y: 벡터
    f: 벡터 함수
    """
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]

    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    for i in range(n_steps):
        k1 = np.array(f(t[i], y[i]))
        k2 = np.array(f(t[i] + h/2, y[i] + h/2 * k1))
        k3 = np.array(f(t[i] + h/2, y[i] + h/2 * k2))
        k4 = np.array(f(t[i] + h, y[i] + h * k3))

        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    return t, y

# 단순 조화 진동자: x'' + ω²x = 0
# 시스템: y₁' = y₂, y₂' = -ω²y₁
omega = 2.0

def harmonic_oscillator(t, y):
    return [y[1], -omega**2 * y[0]]

t_span = (0, 10)
y0 = [1.0, 0.0]  # x(0) = 1, x'(0) = 0

t, y = rk4_system(harmonic_oscillator, y0, t_span, 200)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 시간 응답
axes[0].plot(t, y[:, 0], label='x(t)')
axes[0].plot(t, y[:, 1], label="x'(t)")
axes[0].plot(t, np.cos(omega * t), 'k--', alpha=0.5, label='정확해')
axes[0].set_xlabel('t')
axes[0].set_ylabel('값')
axes[0].set_title('조화 진동자')
axes[0].legend()
axes[0].grid(True)

# 위상 평면
axes[1].plot(y[:, 0], y[:, 1])
axes[1].set_xlabel('x')
axes[1].set_ylabel("x'")
axes[1].set_title('위상 공간')
axes[1].set_aspect('equal')
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

### 3.2 감쇠 진동자

```python
def damped_oscillator_example():
    """감쇠 진동자: x'' + 2γx' + ω₀²x = 0"""
    omega0 = 2.0
    gamma = 0.3  # 감쇠 계수

    def damped_osc(t, y):
        return [y[1], -2*gamma*y[1] - omega0**2 * y[0]]

    t_span = (0, 15)
    y0 = [1.0, 0.0]

    t, y = rk4_system(damped_osc, y0, t_span, 300)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 시간 응답
    omega_d = np.sqrt(omega0**2 - gamma**2)  # 감쇠 진동수
    envelope = np.exp(-gamma * t)

    axes[0].plot(t, y[:, 0], label='x(t)')
    axes[0].plot(t, envelope, 'r--', label='감쇠 envelope')
    axes[0].plot(t, -envelope, 'r--')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x')
    axes[0].set_title(f'감쇠 진동자 (γ={gamma}, ω₀={omega0})')
    axes[0].legend()
    axes[0].grid(True)

    # 위상 공간 (나선형)
    axes[1].plot(y[:, 0], y[:, 1])
    axes[1].scatter([0], [0], color='red', s=100, zorder=5, label='안정 평형')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel("x'")
    axes[1].set_title('위상 공간')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

damped_oscillator_example()
```

---

## 4. 적응형 스텝 크기

### 4.1 오차 추정

```python
def rk45_step(f, t, y, h):
    """
    RK4-5 (Dormand-Prince) 한 스텝

    4차와 5차 근사를 동시에 계산하여 오차 추정
    """
    # Dormand-Prince 계수
    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    a = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ]
    b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]

    # k 계산
    k = [np.array(f(t, y))]
    for i in range(1, 7):
        yi = y.copy()
        for j in range(i):
            yi = yi + h * a[i][j] * k[j]
        k.append(np.array(f(t + c[i]*h, yi)))

    # 5차 근사
    y5 = y.copy()
    for i in range(7):
        y5 = y5 + h * b5[i] * k[i]

    # 4차 근사 (오차 추정용)
    y4 = y.copy()
    for i in range(7):
        y4 = y4 + h * b4[i] * k[i]

    # 오차 추정
    error = np.max(np.abs(y5 - y4))

    return y5, error

def adaptive_rk45(f, y0, t_span, tol=1e-6, h_init=0.1):
    """적응형 RK4-5 솔버"""
    t = [t_span[0]]
    y = [np.array(y0)]
    h = h_init

    while t[-1] < t_span[1]:
        # 스텝 크기가 범위를 벗어나지 않도록
        if t[-1] + h > t_span[1]:
            h = t_span[1] - t[-1]

        # RK4-5 스텝
        y_new, error = rk45_step(f, t[-1], y[-1], h)

        # 오차에 따른 스텝 조절
        if error < tol:
            t.append(t[-1] + h)
            y.append(y_new)

            # 스텝 크기 증가 (안전 계수 0.9)
            if error > 0:
                h = min(h * 0.9 * (tol / error) ** 0.2, 2 * h)
        else:
            # 스텝 크기 감소
            h = max(h * 0.9 * (tol / error) ** 0.25, h / 4)

    return np.array(t), np.array(y)

# 테스트
def stiff_like(t, y):
    """급격한 변화가 있는 함수"""
    return [-10 * y[0] if t < 1 else -0.1 * y[0]]

t_adapt, y_adapt = adaptive_rk45(stiff_like, [1.0], (0, 5), tol=1e-6)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(t_adapt, y_adapt[:, 0], 'o-', markersize=3)
plt.xlabel('t')
plt.ylabel('y')
plt.title('적응형 RK4-5')
plt.grid(True)

plt.subplot(1, 2, 2)
dt = np.diff(t_adapt)
plt.plot(t_adapt[:-1], dt, 'o-')
plt.xlabel('t')
plt.ylabel('스텝 크기 h')
plt.title('스텝 크기 변화')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 5. SciPy를 이용한 ODE 풀이

### 5.1 solve_ivp

```python
from scipy.integrate import solve_ivp

# 로렌츠 시스템
def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

t_span = (0, 50)
y0 = [1.0, 1.0, 1.0]

# 다양한 솔버 비교
solvers = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF']

fig = plt.figure(figsize=(15, 10))

for idx, method in enumerate(solvers, 1):
    sol = solve_ivp(lorenz, t_span, y0, method=method,
                    dense_output=True, max_step=0.01)

    ax = fig.add_subplot(2, 3, idx, projection='3d')
    ax.plot(sol.y[0], sol.y[1], sol.y[2], linewidth=0.5)
    ax.set_title(f'{method} (n={len(sol.t)})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

plt.tight_layout()
plt.show()
```

### 5.2 이벤트 검출

```python
def projectile_motion():
    """포물선 운동 - 지면 충돌 검출"""

    g = 9.8

    def projectile(t, state):
        x, y, vx, vy = state
        return [vx, vy, 0, -g]

    def hit_ground(t, state):
        return state[1]  # y = 0일 때 이벤트

    hit_ground.terminal = True  # 이벤트 발생 시 중단
    hit_ground.direction = -1   # 감소할 때만 (낙하 중)

    # 초기 조건: 45도 각도, 속도 20 m/s
    v0 = 20
    angle = 45 * np.pi / 180
    y0 = [0, 0, v0 * np.cos(angle), v0 * np.sin(angle)]

    sol = solve_ivp(projectile, (0, 10), y0, events=hit_ground,
                    dense_output=True, max_step=0.01)

    print(f"비행 시간: {sol.t_events[0][0]:.4f} s")
    print(f"비행 거리: {sol.y_events[0][0][0]:.4f} m")

    plt.figure(figsize=(10, 5))
    plt.plot(sol.y[0], sol.y[1])
    plt.scatter(sol.y_events[0][0][0], 0, color='red', s=100, label='착지점')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('포물선 운동')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

projectile_motion()
```

---

## 6. 안정성 분석

### 6.1 안정 영역

```python
def stability_regions():
    """수치 방법의 안정 영역"""

    # 테스트 방정식: dy/dt = λy
    # 정확해: y(nh) = e^(λnh)
    # 수치해: y_n = R(λh)^n * y_0

    # 복소 평면에서 |R(z)| ≤ 1인 영역

    x = np.linspace(-4, 2, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y  # z = λh

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 전진 오일러: R(z) = 1 + z
    R_euler = np.abs(1 + Z)
    axes[0, 0].contour(X, Y, R_euler, levels=[1], colors='blue')
    axes[0, 0].contourf(X, Y, R_euler, levels=[0, 1], alpha=0.3)
    axes[0, 0].set_title('전진 오일러: |1 + z| ≤ 1')
    axes[0, 0].set_xlabel('Re(λh)')
    axes[0, 0].set_ylabel('Im(λh)')
    axes[0, 0].grid(True)
    axes[0, 0].set_aspect('equal')

    # 후진 오일러: R(z) = 1/(1-z)
    R_backward = np.abs(1 / (1 - Z))
    axes[0, 1].contour(X, Y, R_backward, levels=[1], colors='blue')
    axes[0, 1].contourf(X, Y, R_backward, levels=[0, 1], alpha=0.3)
    axes[0, 1].set_title('후진 오일러: |1/(1-z)| ≤ 1')
    axes[0, 1].set_xlabel('Re(λh)')
    axes[0, 1].set_ylabel('Im(λh)')
    axes[0, 1].grid(True)
    axes[0, 1].set_aspect('equal')

    # RK2: R(z) = 1 + z + z²/2
    R_rk2 = np.abs(1 + Z + Z**2/2)
    axes[1, 0].contour(X, Y, R_rk2, levels=[1], colors='blue')
    axes[1, 0].contourf(X, Y, R_rk2, levels=[0, 1], alpha=0.3)
    axes[1, 0].set_title('RK2: |1 + z + z²/2| ≤ 1')
    axes[1, 0].set_xlabel('Re(λh)')
    axes[1, 0].set_ylabel('Im(λh)')
    axes[1, 0].grid(True)
    axes[1, 0].set_aspect('equal')

    # RK4: R(z) = 1 + z + z²/2 + z³/6 + z⁴/24
    R_rk4 = np.abs(1 + Z + Z**2/2 + Z**3/6 + Z**4/24)
    axes[1, 1].contour(X, Y, R_rk4, levels=[1], colors='blue')
    axes[1, 1].contourf(X, Y, R_rk4, levels=[0, 1], alpha=0.3)
    axes[1, 1].set_title('RK4')
    axes[1, 1].set_xlabel('Re(λh)')
    axes[1, 1].set_ylabel('Im(λh)')
    axes[1, 1].grid(True)
    axes[1, 1].set_aspect('equal')

    plt.tight_layout()
    plt.show()

stability_regions()
```

---

## 연습 문제

### 문제 1
RK4로 Van der Pol 진동자를 풀어보세요:
x'' - μ(1 - x²)x' + x = 0, μ = 2

```python
def exercise_1():
    mu = 2.0

    def van_der_pol(t, y):
        return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

    sol = solve_ivp(van_der_pol, (0, 30), [0.1, 0],
                    method='RK45', max_step=0.01)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(sol.t, sol.y[0])
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x')
    axes[0].set_title('Van der Pol 진동자')

    axes[1].plot(sol.y[0], sol.y[1])
    axes[1].set_xlabel('x')
    axes[1].set_ylabel("x'")
    axes[1].set_title('위상 공간 (리밋 사이클)')

    plt.tight_layout()
    plt.show()

exercise_1()
```

---

## 요약

| 방법 | 차수 | 특징 |
|------|------|------|
| 전진 오일러 | O(h) | 간단, 제한적 안정성 |
| 후진 오일러 | O(h) | 암시적, A-안정 |
| RK2 | O(h²) | 중점법, Heun |
| RK4 | O(h⁴) | 가장 널리 사용 |
| RK4-5 | O(h⁵) | 적응형 스텝 |

| SciPy 솔버 | 유형 | 용도 |
|-----------|------|------|
| RK45 | 명시적 | 일반 문제 (기본) |
| DOP853 | 명시적 | 고정밀도 |
| Radau | 암시적 | 강성 문제 |
| BDF | 암시적 | 강성 문제 |
