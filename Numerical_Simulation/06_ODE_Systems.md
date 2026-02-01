# 연립 ODE와 시스템

## 개요

실제 물리적 시스템은 여러 변수가 상호작용하는 연립 ODE로 기술됩니다. 생태계 모델, 진자 운동, 혼돈 시스템 등 다양한 예제를 통해 연립 ODE의 수치해법을 학습합니다.

---

## 1. Lotka-Volterra (포식자-피식자)

### 1.1 모델 설명

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
Lotka-Volterra 방정식:

dx/dt = αx - βxy    (피식자: 토끼)
dy/dt = δxy - γy    (포식자: 여우)

α: 피식자 성장률
β: 포식률
γ: 포식자 사망률
δ: 포식자 성장 효율

평형점:
- (0, 0): 멸종 (새들포인트)
- (γ/δ, α/β): 공존 (중심)
"""

def lotka_volterra():
    alpha = 1.0  # 토끼 성장률
    beta = 0.1   # 포식률
    gamma = 1.5  # 여우 사망률
    delta = 0.075  # 여우 성장 효율

    def lv(t, y):
        x, y_pred = y
        dx = alpha*x - beta*x*y_pred
        dy = delta*x*y_pred - gamma*y_pred
        return [dx, dy]

    # 초기 조건
    y0 = [40, 9]  # 토끼 40마리, 여우 9마리
    t_span = (0, 50)
    t_eval = np.linspace(0, 50, 1000)

    sol = solve_ivp(lv, t_span, y0, t_eval=t_eval, method='RK45')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 시간에 따른 개체수
    axes[0].plot(sol.t, sol.y[0], 'b-', label='토끼 (피식자)')
    axes[0].plot(sol.t, sol.y[1], 'r-', label='여우 (포식자)')
    axes[0].set_xlabel('시간')
    axes[0].set_ylabel('개체수')
    axes[0].set_title('Lotka-Volterra 개체 동역학')
    axes[0].legend()
    axes[0].grid(True)

    # 위상 평면
    axes[1].plot(sol.y[0], sol.y[1], 'g-')
    axes[1].scatter([y0[0]], [y0[1]], color='red', s=100, zorder=5, label='시작점')
    axes[1].scatter([gamma/delta], [alpha/beta], color='black', s=100,
                    marker='x', zorder=5, label='평형점')
    axes[1].set_xlabel('토끼')
    axes[1].set_ylabel('여우')
    axes[1].set_title('위상 공간')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

lotka_volterra()
```

### 1.2 벡터장과 궤적

```python
def lv_phase_portrait():
    """Lotka-Volterra 벡터장과 여러 초기조건"""
    alpha, beta, gamma, delta = 1.0, 0.1, 1.5, 0.075

    def lv(t, y):
        return [alpha*y[0] - beta*y[0]*y[1],
                delta*y[0]*y[1] - gamma*y[1]]

    # 벡터장
    x_range = np.linspace(0.1, 80, 20)
    y_range = np.linspace(0.1, 30, 20)
    X, Y = np.meshgrid(x_range, y_range)

    U = alpha*X - beta*X*Y
    V = delta*X*Y - gamma*Y

    # 정규화
    N = np.sqrt(U**2 + V**2)
    U, V = U/N, V/N

    plt.figure(figsize=(12, 8))
    plt.quiver(X, Y, U, V, alpha=0.5, color='gray')

    # 다양한 초기조건
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for i, (x0, y0) in enumerate([(10, 5), (30, 5), (50, 10), (70, 15), (20, 20)]):
        sol = solve_ivp(lv, (0, 50), [x0, y0], max_step=0.1)
        plt.plot(sol.y[0], sol.y[1], color=colors[i], linewidth=1.5)
        plt.scatter([x0], [y0], color=colors[i], s=50)

    # 평형점
    plt.scatter([gamma/delta], [alpha/beta], color='red', s=150,
                marker='*', zorder=10, label='평형점')

    plt.xlabel('피식자 (x)')
    plt.ylabel('포식자 (y)')
    plt.title('Lotka-Volterra 위상 초상화')
    plt.legend()
    plt.xlim(0, 80)
    plt.ylim(0, 30)
    plt.grid(True)
    plt.show()

lv_phase_portrait()
```

---

## 2. 진자 운동

### 2.1 단순 진자

```python
def simple_pendulum():
    """
    단순 진자: θ'' + (g/L)sin(θ) = 0

    작은 각도 근사: θ'' + (g/L)θ = 0
    해: θ = θ₀cos(ωt), ω = √(g/L)
    """
    g = 9.8  # 중력 가속도
    L = 1.0  # 진자 길이

    def pendulum(t, y):
        theta, omega = y
        return [omega, -(g/L) * np.sin(theta)]

    # 다양한 초기 각도
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, theta0 in zip(axes.flat, [0.1, 0.5, np.pi/2, np.pi - 0.1]):
        sol = solve_ivp(pendulum, (0, 10), [theta0, 0], max_step=0.01)

        ax.plot(sol.y[0], sol.y[1])
        ax.scatter([theta0], [0], color='red', s=100, label='시작')
        ax.set_xlabel('θ (rad)')
        ax.set_ylabel('ω (rad/s)')
        ax.set_title(f'θ₀ = {theta0:.2f} rad ({np.degrees(theta0):.1f}°)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

simple_pendulum()
```

### 2.2 감쇠 강제 진자

```python
def driven_pendulum():
    """
    감쇠 강제 진자:
    θ'' + γθ' + (g/L)sin(θ) = A*cos(ωt)

    비선형 → 혼돈 가능성
    """
    g, L = 9.8, 1.0
    gamma = 0.5  # 감쇠 계수
    A = 1.5      # 구동력 진폭
    omega_d = 2/3 * np.sqrt(g/L)  # 구동 주파수

    def driven(t, y):
        theta, w = y
        return [w, -gamma*w - (g/L)*np.sin(theta) + A*np.cos(omega_d*t)]

    t_span = (0, 200)
    sol = solve_ivp(driven, t_span, [0.1, 0], max_step=0.01)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 시간 응답
    axes[0, 0].plot(sol.t, sol.y[0])
    axes[0, 0].set_xlabel('시간')
    axes[0, 0].set_ylabel('θ')
    axes[0, 0].set_title('시간 응답')
    axes[0, 0].grid(True)

    # 위상 공간
    axes[0, 1].plot(sol.y[0], sol.y[1], linewidth=0.5)
    axes[0, 1].set_xlabel('θ')
    axes[0, 1].set_ylabel('ω')
    axes[0, 1].set_title('위상 공간')
    axes[0, 1].grid(True)

    # 포앙카레 단면 (구동력 주기마다 샘플링)
    T_drive = 2 * np.pi / omega_d
    t_poincare = np.arange(0, t_span[1], T_drive)

    from scipy.interpolate import interp1d
    theta_interp = interp1d(sol.t, sol.y[0])
    omega_interp = interp1d(sol.t, sol.y[1])

    valid_t = t_poincare[(t_poincare >= sol.t[0]) & (t_poincare <= sol.t[-1])]
    theta_p = theta_interp(valid_t)
    omega_p = omega_interp(valid_t)

    axes[1, 0].scatter(theta_p[100:], omega_p[100:], s=1)  # 과도 상태 제외
    axes[1, 0].set_xlabel('θ')
    axes[1, 0].set_ylabel('ω')
    axes[1, 0].set_title('포앙카레 단면')
    axes[1, 0].grid(True)

    # 에너지
    E = 0.5 * sol.y[1]**2 + (g/L) * (1 - np.cos(sol.y[0]))
    axes[1, 1].plot(sol.t, E)
    axes[1, 1].set_xlabel('시간')
    axes[1, 1].set_ylabel('에너지')
    axes[1, 1].set_title('역학적 에너지')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

driven_pendulum()
```

### 2.3 이중 진자

```python
def double_pendulum():
    """
    이중 진자: 혼돈적 시스템

    θ₁, θ₂: 각 진자의 각도
    ω₁, ω₂: 각속도
    """
    g = 9.8
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0

    def double_pend(t, y):
        t1, t2, w1, w2 = y

        delta = t2 - t1
        denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
        denom2 = (L2 / L1) * denom1

        dw1 = (m2 * L1 * w1**2 * np.sin(delta) * np.cos(delta) +
               m2 * g * np.sin(t2) * np.cos(delta) +
               m2 * L2 * w2**2 * np.sin(delta) -
               (m1 + m2) * g * np.sin(t1)) / denom1

        dw2 = (-m2 * L2 * w2**2 * np.sin(delta) * np.cos(delta) +
               (m1 + m2) * g * np.sin(t1) * np.cos(delta) -
               (m1 + m2) * L1 * w1**2 * np.sin(delta) -
               (m1 + m2) * g * np.sin(t2)) / denom2

        return [w1, w2, dw1, dw2]

    # 초기 조건에 민감
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    initial_conditions = [
        [np.pi/2, np.pi/2, 0, 0],
        [np.pi/2 + 0.001, np.pi/2, 0, 0],  # 미세한 차이
    ]

    for ic, color in zip(initial_conditions, ['blue', 'red']):
        sol = solve_ivp(double_pend, (0, 20), ic, max_step=0.01)

        # 위치 계산
        x1 = L1 * np.sin(sol.y[0])
        y1 = -L1 * np.cos(sol.y[0])
        x2 = x1 + L2 * np.sin(sol.y[1])
        y2 = y1 - L2 * np.cos(sol.y[1])

        axes[0, 0].plot(sol.t, sol.y[0], color=color, alpha=0.7)
        axes[0, 1].plot(sol.t, sol.y[1], color=color, alpha=0.7)
        axes[1, 0].plot(x2, y2, color=color, linewidth=0.5, alpha=0.7)

    axes[0, 0].set_xlabel('시간')
    axes[0, 0].set_ylabel('θ₁')
    axes[0, 0].set_title('첫 번째 진자')
    axes[0, 0].grid(True)

    axes[0, 1].set_xlabel('시간')
    axes[0, 1].set_ylabel('θ₂')
    axes[0, 1].set_title('두 번째 진자')
    axes[0, 1].grid(True)

    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('끝점 궤적 (초기조건 민감도)')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].grid(True)

    # 초기조건 차이 시각화
    sol1 = solve_ivp(double_pend, (0, 20), initial_conditions[0], max_step=0.01)
    sol2 = solve_ivp(double_pend, (0, 20), initial_conditions[1], max_step=0.01)

    from scipy.interpolate import interp1d
    t_common = np.linspace(0, 20, 1000)
    theta1_1 = interp1d(sol1.t, sol1.y[0])(t_common)
    theta1_2 = interp1d(sol2.t, sol2.y[0])(t_common)

    axes[1, 1].semilogy(t_common, np.abs(theta1_1 - theta1_2))
    axes[1, 1].set_xlabel('시간')
    axes[1, 1].set_ylabel('|Δθ₁|')
    axes[1, 1].set_title('궤적 분리 (지수적 증가)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

double_pendulum()
```

---

## 3. 로렌츠 시스템 (혼돈)

### 3.1 기본 시뮬레이션

```python
def lorenz_system():
    """
    로렌츠 시스템 (1963):

    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

    표준 파라미터: σ=10, ρ=28, β=8/3
    → 이상한 끌개 (strange attractor)
    """
    sigma, rho, beta = 10, 28, 8/3

    def lorenz(t, state):
        x, y, z = state
        return [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ]

    sol = solve_ivp(lorenz, (0, 50), [1, 1, 1], max_step=0.01)

    fig = plt.figure(figsize=(15, 5))

    # 3D 궤적
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(sol.y[0], sol.y[1], sol.y[2], linewidth=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('로렌츠 끌개')

    # x-z 투영
    ax2 = fig.add_subplot(132)
    ax2.plot(sol.y[0], sol.y[2], linewidth=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    ax2.set_title('x-z 투영')
    ax2.grid(True)

    # 시간 응답
    ax3 = fig.add_subplot(133)
    ax3.plot(sol.t[:500], sol.y[0][:500])
    ax3.set_xlabel('시간')
    ax3.set_ylabel('x')
    ax3.set_title('x(t)')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

lorenz_system()
```

### 3.2 초기조건 민감도

```python
def lorenz_sensitivity():
    """로렌츠 시스템의 초기조건 민감도"""
    sigma, rho, beta = 10, 28, 8/3

    def lorenz(t, state):
        x, y, z = state
        return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

    # 미세하게 다른 초기조건
    eps = 1e-10
    sol1 = solve_ivp(lorenz, (0, 50), [1, 1, 1], max_step=0.01)
    sol2 = solve_ivp(lorenz, (0, 50), [1+eps, 1, 1], max_step=0.01)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 시간에 따른 x
    axes[0, 0].plot(sol1.t, sol1.y[0], 'b-', alpha=0.7, label='IC 1')
    axes[0, 0].plot(sol2.t, sol2.y[0], 'r-', alpha=0.7, label='IC 2')
    axes[0, 0].set_xlabel('시간')
    axes[0, 0].set_ylabel('x')
    axes[0, 0].set_title(f'x(t), 초기차이 = {eps}')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 궤적 분리
    from scipy.interpolate import interp1d
    t_common = np.linspace(0, 50, 5000)
    x1 = interp1d(sol1.t, sol1.y[0])(t_common)
    x2 = interp1d(sol2.t, sol2.y[0])(t_common)
    y1 = interp1d(sol1.t, sol1.y[1])(t_common)
    y2 = interp1d(sol2.t, sol2.y[1])(t_common)
    z1 = interp1d(sol1.t, sol1.y[2])(t_common)
    z2 = interp1d(sol2.t, sol2.y[2])(t_common)

    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

    axes[0, 1].semilogy(t_common, dist)
    axes[0, 1].set_xlabel('시간')
    axes[0, 1].set_ylabel('거리')
    axes[0, 1].set_title('궤적 간 거리 (로그 스케일)')
    axes[0, 1].grid(True)

    # 리아푸노프 지수 추정
    # 초기 지수적 성장 구간에서
    early = (t_common > 5) & (t_common < 20)
    from scipy.stats import linregress
    slope, _, _, _, _ = linregress(t_common[early], np.log(dist[early] + 1e-20))
    print(f"추정 리아푸노프 지수: {slope:.3f}")

    axes[1, 0].plot(t_common[early], np.log(dist[early]))
    axes[1, 0].set_xlabel('시간')
    axes[1, 0].set_ylabel('log(거리)')
    axes[1, 0].set_title(f'리아푸노프 지수 추정: λ ≈ {slope:.3f}')
    axes[1, 0].grid(True)

    # 3D 비교
    ax = fig.add_subplot(224, projection='3d')
    ax.plot(sol1.y[0][::10], sol1.y[1][::10], sol1.y[2][::10],
            'b-', alpha=0.5, linewidth=0.5)
    ax.plot(sol2.y[0][::10], sol2.y[1][::10], sol2.y[2][::10],
            'r-', alpha=0.5, linewidth=0.5)
    ax.set_title('두 궤적 비교')

    plt.tight_layout()
    plt.show()

lorenz_sensitivity()
```

### 3.3 분기 다이어그램

```python
def lorenz_bifurcation():
    """ρ에 따른 로렌츠 시스템 분기"""
    sigma, beta = 10, 8/3

    rho_values = np.linspace(0, 50, 100)
    bifurcation_points = []

    for rho in rho_values:
        def lorenz(t, state):
            x, y, z = state
            return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

        sol = solve_ivp(lorenz, (0, 200), [1, 1, 1], max_step=0.01)

        # 과도 상태 제거 후 z의 극값 수집
        z = sol.y[2][len(sol.t)//2:]

        # 극값 찾기
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(z)
        if len(peaks) > 0:
            z_peaks = z[peaks[-min(50, len(peaks)):]]  # 마지막 50개
            for zp in z_peaks:
                bifurcation_points.append((rho, zp))

    if bifurcation_points:
        rhos, zs = zip(*bifurcation_points)
        plt.figure(figsize=(12, 6))
        plt.scatter(rhos, zs, s=0.5, c='black')
        plt.xlabel('ρ')
        plt.ylabel('z 극값')
        plt.title('로렌츠 시스템 분기 다이어그램')
        plt.grid(True)
        plt.show()

# 계산 시간이 오래 걸릴 수 있음
# lorenz_bifurcation()
print("분기 다이어그램은 계산 시간이 오래 걸립니다.")
```

---

## 4. 기타 시스템

### 4.1 SIR 전염병 모델

```python
def sir_model():
    """
    SIR 모델:
    dS/dt = -βSI
    dI/dt = βSI - γI
    dR/dt = γI

    S: 감수성(Susceptible)
    I: 감염(Infected)
    R: 회복(Recovered)
    """
    beta = 0.3   # 감염률
    gamma = 0.1  # 회복률

    def sir(t, y):
        S, I, R = y
        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I
        return [dS, dI, dR]

    # 초기 조건 (인구의 비율)
    S0, I0, R0 = 0.99, 0.01, 0.0
    t_span = (0, 100)

    sol = solve_ivp(sir, t_span, [S0, I0, R0],
                    t_eval=np.linspace(0, 100, 1000))

    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, sol.y[0], 'b-', label='감수성 (S)')
    plt.plot(sol.t, sol.y[1], 'r-', label='감염 (I)')
    plt.plot(sol.t, sol.y[2], 'g-', label='회복 (R)')
    plt.xlabel('시간')
    plt.ylabel('인구 비율')
    plt.title(f'SIR 모델 (β={beta}, γ={gamma}, R₀={beta/gamma:.1f})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 기본 재생산 수 R₀ = β/γ
    R0 = beta / gamma
    print(f"기본 재생산 수 R₀ = {R0:.2f}")
    print(f"임계 면역률 = 1 - 1/R₀ = {1 - 1/R0:.2%}")

sir_model()
```

### 4.2 뢰슬러 끌개

```python
def rossler_attractor():
    """
    뢰슬러 끌개:
    dx/dt = -y - z
    dy/dt = x + ay
    dz/dt = b + z(x - c)
    """
    a, b, c = 0.2, 0.2, 5.7

    def rossler(t, state):
        x, y, z = state
        return [-y - z, x + a*y, b + z*(x - c)]

    sol = solve_ivp(rossler, (0, 200), [0, 1, 0], max_step=0.01)

    fig = plt.figure(figsize=(15, 5))

    # 3D
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(sol.y[0], sol.y[1], sol.y[2], linewidth=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('뢰슬러 끌개')

    # x-y 투영
    ax2 = fig.add_subplot(132)
    ax2.plot(sol.y[0], sol.y[1], linewidth=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('x-y 투영')
    ax2.grid(True)

    # 시계열
    ax3 = fig.add_subplot(133)
    ax3.plot(sol.t[2000:4000], sol.y[0][2000:4000])
    ax3.set_xlabel('시간')
    ax3.set_ylabel('x')
    ax3.set_title('x(t)')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

rossler_attractor()
```

### 4.3 N체 문제 (제한적)

```python
def three_body_simplified():
    """단순화된 3체 문제 (평면, 동일 질량)"""
    G = 1  # 중력 상수 (단위계 조정)
    m = 1  # 모든 질량 동일

    def three_body(t, y):
        # y = [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
        x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = y

        # 거리
        r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
        r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)

        # 가속도
        ax1 = G*m*((x2-x1)/r12**3 + (x3-x1)/r13**3)
        ay1 = G*m*((y2-y1)/r12**3 + (y3-y1)/r13**3)
        ax2 = G*m*((x1-x2)/r12**3 + (x3-x2)/r23**3)
        ay2 = G*m*((y1-y2)/r12**3 + (y3-y2)/r23**3)
        ax3 = G*m*((x1-x3)/r13**3 + (x2-x3)/r23**3)
        ay3 = G*m*((y1-y3)/r13**3 + (y2-y3)/r23**3)

        return [vx1, vy1, vx2, vy2, vx3, vy3,
                ax1, ay1, ax2, ay2, ax3, ay3]

    # 8자형 해 초기조건 (Chenciner & Montgomery, 2000)
    # 특별한 주기해
    x0 = 0.97000436
    y0 = -0.24308753
    vx0 = 0.4662036850
    vy0 = 0.4323657300

    y0_state = [-x0, -y0, x0, y0, 0, 0,
                vx0, vy0, vx0, vy0, -2*vx0, -2*vy0]

    sol = solve_ivp(three_body, (0, 6.3), y0_state,
                    method='DOP853', max_step=0.01)

    plt.figure(figsize=(10, 8))
    plt.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.8, label='물체 1')
    plt.plot(sol.y[2], sol.y[3], 'r-', linewidth=0.8, label='물체 2')
    plt.plot(sol.y[4], sol.y[5], 'g-', linewidth=0.8, label='물체 3')

    # 초기 위치
    plt.scatter([sol.y[0][0], sol.y[2][0], sol.y[4][0]],
                [sol.y[1][0], sol.y[3][0], sol.y[5][0]],
                s=100, c=['blue', 'red', 'green'])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('3체 문제: 8자형 주기해')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

three_body_simplified()
```

---

## 연습 문제

### 문제 1
Lotka-Volterra 시스템의 평형점 주변에서 선형화하고 고유값을 분석하세요.

```python
def exercise_1():
    alpha, beta, gamma, delta = 1.0, 0.1, 1.5, 0.075

    # 평형점
    x_eq = gamma / delta
    y_eq = alpha / beta

    print(f"평형점: ({x_eq:.2f}, {y_eq:.2f})")

    # 야코비안
    # J = [[α - βy, -βx], [δy, δx - γ]]
    # 평형점에서:
    J = np.array([[0, -beta * x_eq],
                  [delta * y_eq, 0]])

    eigenvalues = np.linalg.eigvals(J)
    print(f"고유값: {eigenvalues}")
    print(f"순허수 → 중심 (주기 궤도)")

exercise_1()
```

---

## 요약

| 시스템 | 특징 | 주요 현상 |
|--------|------|----------|
| Lotka-Volterra | 2D, 보존계 | 주기 진동 |
| 단순 진자 | 2D, 비선형 | 작은 각도: 주기, 큰 각도: 비선형 |
| 이중 진자 | 4D, 혼돈 | 초기조건 민감도 |
| 로렌츠 | 3D, 혼돈 | 이상한 끌개 |
| SIR | 3D, 소산 | 전염병 동역학 |
| 뢰슬러 | 3D, 혼돈 | 띠 끌개 |

| 분석 도구 | 용도 |
|----------|------|
| 위상 초상화 | 궤적 시각화 |
| 포앙카레 단면 | 주기성/혼돈 구분 |
| 리아푸노프 지수 | 혼돈 정량화 |
| 분기 다이어그램 | 파라미터 민감도 |
