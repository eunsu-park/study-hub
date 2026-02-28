# 응용과 모델링(Applications and Modeling)

## 학습 목표

- 미분방정식을 사용하여 개체군 동역학, 전기 회로, 기계 시스템의 수학적 모델을 수립할 수 있다
- 비선형 시스템의 정성적 거동(평형점, 안정성, 위상 초상)을 분석할 수 있다
- 연립 ODE를 수치적으로 풀고 결과를 물리적으로 해석할 수 있다
- PDE 모델(열전도, 확산)을 실제 경계값 문제에 적용할 수 있다
- 전체 과정의 기법을 종합하여 완전한 물리적 시나리오를 모델링하고 시뮬레이션할 수 있다

## 선수 과목

이 종합 레슨은 전체 과정을 활용합니다. 주요 선수 과목:
- 1계 ODE와 분리 가능 방정식 (레슨 8-9)
- 2계 선형 ODE (레슨 10-12)
- ODE 연립 (레슨 14)
- 수치 해법 (레슨 19)
- 기본 PDE 개념 (레슨 17)

## 동기: 방정식에서 이해로

이 과정을 통해 우리는 적분, 미분방정식, 급수 해법, 변환, 푸리에 방법, 수치 알고리즘이라는 도구 상자를 구축해 왔습니다. 이 마지막 레슨에서는 생물학, 공학, 물리학의 실제 시스템을 모델링하여 모든 것을 하나로 모읍니다.

모델링 주기는 다음과 같습니다:

```
Real-world problem
       │
       ▼
 Identify variables
 and assumptions
       │
       ▼
  Write DE model    ←─── Physics / Biology / Engineering laws
       │
       ▼
  Solve (analytical
  or numerical)
       │
       ▼
  Interpret and     ←─── Does it match observations?
  validate               If not, refine the model.
```

## 개체군 동역학(Population Dynamics)

### 로지스틱 모델(Logistic Model)

가장 간단한 개체군 모델은 지수적 성장입니다: $P' = rP$, 이것은 $P(t) = P_0 e^{rt}$를 줍니다. 자원이 유한하므로 대규모 개체군에서는 비현실적입니다.

**로지스틱 방정식**(logistic equation)은 환경 수용 능력(carrying capacity) $K$를 추가합니다:

$$\frac{dP}{dt} = rP\left(1 - \frac{P}{K}\right)$$

여기서:
- $P(t)$: 시각 $t$에서의 개체군
- $r$: 내재적 성장률 ($P \ll K$일 때 1인당 출생 빼기 사망)
- $K$: 환경 수용 능력 (최대 지속 가능 개체군)
- $(1 - P/K)$ 인자: $P$가 $K$에 접근할수록 성장을 둔화

**평형점**: $P^* = 0$ (멸종, 불안정)과 $P^* = K$ (안정적 환경 수용 능력).

**정확한 해** (분리 가능 방정식):

$$P(t) = \frac{K}{1 + \left(\frac{K}{P_0} - 1\right)e^{-rt}}$$

이것이 유명한 **로지스틱 곡선**(logistic curve) 또는 S-곡선입니다: 느린 초기 성장, 빠른 중간 성장, 그리고 포화. 세균 군집에서 기술 보급에 이르기까지 모든 것을 모델링합니다.

### 포식자-피식자 관계: 로트카-볼테라 시스템(Lotka-Volterra System)

두 종이 상호작용할 때 -- 하나가 다른 하나를 잡아먹을 때 -- 연립 시스템을 얻습니다:

$$\frac{dx}{dt} = \alpha x - \beta xy \qquad \text{(피식자)}$$

$$\frac{dy}{dt} = \delta xy - \gamma y \qquad \text{(포식자)}$$

여기서:
- $x(t)$: 피식자 개체군 (예: 토끼)
- $y(t)$: 포식자 개체군 (예: 여우)
- $\alpha$: 피식자 출생률 (포식자 부재 시)
- $\beta$: 포식률 (포식자와 피식자의 조우)
- $\delta$: 포식자 번식률 (섭취한 피식자에 비례)
- $\gamma$: 포식자 사망률 (피식자 부재 시)

**평형점**:
1. $(0, 0)$: 두 종 모두 멸종
2. $(\gamma/\delta, \alpha/\beta)$: 공존 평형

놀라운 특징: 해는 공존 평형 주위의 **주기적 궤도**(periodic orbit)입니다. 피식자가 풍부하면 포식자가 번성하고 증가합니다; 그러면 포식자가 피식자를 과도하게 소비하고, 피식자가 감소하고, 포식자가 기아로 감소하고, 피식자가 회복되며, 주기가 반복됩니다. 이것은 허드슨만 회사(Hudson's Bay Company)의 역사적인 스라소니-산토끼 개체군 데이터에서 처음 관찰되었습니다.

## 전기 회로: RLC 회로

### 회로 법칙에서의 2계 ODE

전압원 $E(t)$가 있는 RLC 직렬 회로는 키르히호프의 전압 법칙(Kirchhoff's voltage law)에 의해 지배됩니다:

$$L\frac{d^2q}{dt^2} + R\frac{dq}{dt} + \frac{1}{C}q = E(t)$$

여기서:
- $q(t)$: 커패시터의 전하 (쿨롱)
- $L$: 인덕턴스 (헨리) -- 전류 변화에 저항 (전기적 관성)
- $R$: 저항 (옴) -- 에너지를 열로 소산 (마찰)
- $C$: 커패시턴스 (패럿) -- 전기장에 에너지 저장 (용수철)
- $E(t)$: 인가 전압 (구동력)

전류는 $I = dq/dt$이므로 동등하게:

$$L\frac{dI}{dt} + RI + \frac{1}{C}\int_0^t I(\tau) \, d\tau = E(t)$$

**역학적 유사**(mechanical analogy): 이 방정식은 감쇠 강제 조화 진동자와 형태가 동일합니다:

| 역학적 | 전기적 |
|--------|--------|
| 질량 $m$ | 인덕턴스 $L$ |
| 감쇠 $b$ | 저항 $R$ |
| 용수철 상수 $k$ | $1/C$ |
| 변위 $x$ | 전하 $q$ |
| 외력 $F(t)$ | 전압 $E(t)$ |

### 거동 영역

특성 방정식 $Ls^2 + Rs + 1/C = 0$의 근은:

$$s = \frac{-R \pm \sqrt{R^2 - 4L/C}}{2L}$$

- **과감쇠**(Overdamped, $R^2 > 4L/C$): 두 개의 실수 음근. 전하가 진동 없이 감쇠.
- **임계 감쇠**(Critically damped, $R^2 = 4L/C$): 중복근. 진동 없이 가장 빠른 감쇠.
- **부족 감쇠**(Underdamped, $R^2 < 4L/C$): 복소 켤레근. 감쇠 진동.

$R = 0$ (저항 없음)일 때: **공진 진동수**(resonant frequency) $\omega_0 = 1/\sqrt{LC}$에서 비감쇠 진동.

## 역학 시스템(Mechanical Systems)

### 결합 질량-용수철 시스템(Coupled Spring-Mass System)

용수철로 연결된 두 질량:

```
   Wall ──|/\/\/|── m₁ ──|/\/\/|── m₂ ──|/\/\/|── Wall
           k₁              k₂              k₃
```

$x_1, x_2$를 평형으로부터의 변위라 합시다. 뉴턴의 제2법칙:

$$m_1 x_1'' = -k_1 x_1 + k_2(x_2 - x_1)$$
$$m_2 x_2'' = -k_2(x_2 - x_1) - k_3 x_2$$

행렬 형태: $\mathbf{M}\mathbf{x}'' = -\mathbf{K}\mathbf{x}$, 여기서 $\mathbf{M}$과 $\mathbf{K}$는 질량 행렬과 강성 행렬입니다. $\mathbf{M}^{-1}\mathbf{K}$의 고유값이 **정규 모드 진동수**(normal mode frequency)를 주고, 고유벡터가 **정규 모드**(normal mode, 모든 부분이 같은 진동수로 운동하는 진동 패턴)를 줍니다.

### 비선형 진자(Nonlinear Pendulum)

완전한 (비선형) 진자 방정식:

$$\frac{d^2\theta}{dt^2} + \frac{g}{L}\sin\theta = 0$$

작은 각도에서 $\sin\theta \approx \theta$이므로 주기 $T_0 = 2\pi\sqrt{L/g}$의 단순 조화 운동을 줍니다. 큰 각도에서는 주기가 증가합니다 -- 거의 수직에서 놓아진 진자는 살짝 흔들리는 진자보다 한 주기를 완성하는 데 더 오래 걸립니다.

정확한 주기는 **타원 적분**(elliptic integral)을 포함합니다:

$$T = 4\sqrt{\frac{L}{g}} \int_0^{\pi/2} \frac{d\phi}{\sqrt{1 - \sin^2(\theta_0/2)\sin^2\phi}}$$

이것은 선형 근사가 깔끔한 답을 주지만, 완전한 물리학이 수치 방법이나 특수 함수를 필요로 하는 아름다운 예입니다.

## 열전도: 1D 막대

### 문제 설정

길이 $L$인 금속 막대의 왼쪽 끝이 온도 $T_1$, 오른쪽 끝이 $T_2$로 유지되고, 초기 온도 분포가 $f(x)$인 경우:

$$u_t = \alpha u_{xx}, \quad 0 < x < L, \quad t > 0$$
$$u(0, t) = T_1, \quad u(L, t) = T_2, \quad u(x, 0) = f(x)$$

### 정상 상태 해

$t \to \infty$에서 $u_t \to 0$이므로, 정상 상태는 $u_{xx} = 0$을 만족합니다:

$$u_{\infty}(x) = T_1 + (T_2 - T_1)\frac{x}{L}$$

선형 온도 프로파일 -- 열이 고온 끝에서 저온 끝으로 균일하게 흐릅니다.

### 과도 해(Transient Solution)

$u(x, t) = u_{\infty}(x) + v(x, t)$로 쓰면, $v$는 **동차**(homogeneous) 경계 조건($v(0,t) = v(L,t) = 0$)을 가진 열 방정식을 만족합니다. 그러면 $v$는 푸리에 급수(레슨 18)로 풀립니다.

## 확산: 피크의 법칙(Fick's Law)

물질의 확산(물에서의 잉크, 공기 중의 오염물)은 열전도와 동일한 방정식을 따릅니다:

$$\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2}$$

여기서:
- $c(x, t)$: 물질의 농도
- $D$: 확산 계수 ($\text{m}^2/\text{s}$)

피크의 법칙($J = -D \, \partial c / \partial x$)은 푸리에의 열전도 법칙의 물질 수송 유사입니다. 수학은 동일합니다.

**기본 해**(fundamental solution): 단위량의 물질이 $t = 0$에 $x = 0$에서 방출된 경우:

$$c(x, t) = \frac{1}{\sqrt{4\pi D t}} \exp\left(-\frac{x^2}{4Dt}\right)$$

이 가우시안은 시간이 지남에 따라 퍼지고 평평해집니다: 폭은 $\sqrt{Dt}$로 성장합니다 (이 제곱근은 확산 시간 척도를 이해하는 데 핵심입니다).

## 파이썬 구현

```python
"""
Applications and Modeling: Capstone Simulations.

This script demonstrates complete modeling workflows for:
1. Population dynamics (logistic + Lotka-Volterra)
2. RLC circuit response
3. Coupled spring-mass system
4. Heat conduction in a rod with non-homogeneous BCs
5. 1D diffusion
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ── 1. Population Dynamics ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1a. Logistic growth
r = 0.5    # growth rate
K = 1000   # carrying capacity

def logistic(t, P):
    """
    Logistic growth: dP/dt = r*P*(1 - P/K)

    The (1 - P/K) factor models resource competition:
    - When P << K: growth is nearly exponential
    - When P = K/2: growth rate is maximum
    - When P -> K: growth slows to zero (saturation)
    """
    return r * P * (1 - P / K)

# Multiple initial conditions show convergence to K
for P0 in [10, 50, 200, 800, 1200]:
    sol = solve_ivp(logistic, (0, 30), [P0], dense_output=True,
                    max_step=0.1)
    t_plot = np.linspace(0, 30, 300)
    axes[0, 0].plot(t_plot, sol.sol(t_plot)[0], linewidth=2,
                    label=f'P(0) = {P0}')

axes[0, 0].axhline(y=K, color='gray', linestyle='--', label=f'K = {K}')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Population')
axes[0, 0].set_title('Logistic Growth: Convergence to Carrying Capacity')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# 1b. Lotka-Volterra predator-prey
alpha_lv = 1.0    # prey birth rate
beta_lv = 0.1     # predation rate
delta_lv = 0.075  # predator growth from eating
gamma_lv = 1.5    # predator death rate

def lotka_volterra(t, state):
    """
    Lotka-Volterra predator-prey model.

    Key insight: the system produces closed orbits because it has
    a conserved quantity (the Lotka-Volterra invariant). In reality,
    stochastic effects and more complex interactions break this
    perfect periodicity.
    """
    x, y = state  # x = prey, y = predators
    dx = alpha_lv * x - beta_lv * x * y
    dy = delta_lv * x * y - gamma_lv * y
    return [dx, dy]

sol_lv = solve_ivp(lotka_volterra, (0, 40), [40, 9],
                   dense_output=True, max_step=0.05)
t_lv = np.linspace(0, 40, 1000)
state_lv = sol_lv.sol(t_lv)

# Time series
axes[0, 1].plot(t_lv, state_lv[0], 'b-', linewidth=2, label='Prey (rabbits)')
axes[0, 1].plot(t_lv, state_lv[1], 'r-', linewidth=2, label='Predators (foxes)')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Population')
axes[0, 1].set_title('Lotka-Volterra: Predator-Prey Oscillations')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)


# ── 2. RLC Circuit ───────────────────────────────────────────
L_circ = 0.5       # inductance (henrys)
R_circ = 2.0       # resistance (ohms)
C_circ = 0.01      # capacitance (farads)

def rlc_circuit(t, state, E_func):
    """
    RLC series circuit: L*q'' + R*q' + q/C = E(t)

    Converted to system:
        q' = I          (current is rate of charge change)
        I' = (E(t) - R*I - q/C) / L  (from Kirchhoff's voltage law)
    """
    q, I = state
    E = E_func(t)
    dq = I
    dI = (E - R_circ * I - q / C_circ) / L_circ
    return [dq, dI]

# Step voltage input: E(t) = 10V for t >= 0
E_step = lambda t: 10.0

# Vary R to show overdamped, critically damped, underdamped
R_critical = 2 * np.sqrt(L_circ / C_circ)  # critical damping value
print(f"Critical resistance: R_c = {R_critical:.2f} ohms")

for R_val, label in [(0.5, 'Underdamped'),
                      (R_critical, 'Critically damped'),
                      (30.0, 'Overdamped')]:
    R_circ = R_val
    sol_rlc = solve_ivp(lambda t, s: rlc_circuit(t, s, E_step),
                        (0, 0.5), [0, 0], dense_output=True,
                        max_step=0.001)
    t_rlc = np.linspace(0, 0.5, 1000)
    q_rlc = sol_rlc.sol(t_rlc)[0]
    # Voltage across capacitor: V_C = q / C
    V_C = q_rlc / C_circ
    axes[1, 0].plot(t_rlc, V_C, linewidth=2,
                    label=f'{label} (R={R_val:.1f})')

R_circ = 2.0  # restore default
axes[1, 0].axhline(y=10, color='gray', linestyle='--', alpha=0.5,
                    label='Steady state (10V)')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Capacitor Voltage (V)')
axes[1, 0].set_title('RLC Circuit: Step Response')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)


# ── 3. Coupled Spring-Mass System ────────────────────────────
m1, m2 = 1.0, 1.0     # masses (kg)
k1, k2, k3 = 4.0, 2.0, 4.0  # spring constants (N/m)

def coupled_springs(t, state):
    """
    Two masses connected by three springs (wall-m1-m2-wall).

    x1'' = (-k1*x1 + k2*(x2-x1)) / m1
    x2'' = (-k2*(x2-x1) - k3*x2) / m2

    Normal modes:
    - Symmetric: both masses move in same direction
    - Antisymmetric: masses move in opposite directions
    """
    x1, v1, x2, v2 = state
    a1 = (-k1 * x1 + k2 * (x2 - x1)) / m1
    a2 = (-k2 * (x2 - x1) - k3 * x2) / m2
    return [v1, a1, v2, a2]

# Initial condition: pull m1, release
sol_springs = solve_ivp(coupled_springs, (0, 15), [1.0, 0, 0, 0],
                        dense_output=True, max_step=0.01)
t_springs = np.linspace(0, 15, 1000)
state_springs = sol_springs.sol(t_springs)

axes[1, 1].plot(t_springs, state_springs[0], 'b-', linewidth=1.5,
                label='Mass 1 (x₁)')
axes[1, 1].plot(t_springs, state_springs[2], 'r-', linewidth=1.5,
                label='Mass 2 (x₂)')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Displacement')
axes[1, 1].set_title('Coupled Springs: Energy Exchange Between Masses')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig('applications_ode.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to applications_ode.png")


# ── 4. Heat Conduction in a Rod ──────────────────────────────
print("\n=== Heat Conduction: Non-homogeneous BCs ===")

L_rod = 1.0       # rod length (m)
alpha_rod = 0.01   # thermal diffusivity
T_left = 100.0    # left boundary temperature
T_right = 25.0    # right boundary temperature
Nx = 50
Nt = 10000
dx = L_rod / Nx
dt = 0.4 * dx**2 / alpha_rod  # CFL condition with safety factor

x_rod = np.linspace(0, L_rod, Nx + 1)
r_heat = alpha_rod * dt / dx**2
print(f"dx={dx:.4f}, dt={dt:.6f}, r={r_heat:.4f} (must be <= 0.5)")

# Initial condition: room temperature everywhere
u = np.ones(Nx + 1) * 20.0

# Steady-state solution (linear profile)
u_steady = T_left + (T_right - T_left) * x_rod / L_rod

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Store snapshots at selected times
snapshot_indices = [0, 50, 200, 500, 2000, 10000]
colors = plt.cm.hot(np.linspace(0.1, 0.9, len(snapshot_indices)))

for n in range(Nt + 1):
    if n in snapshot_indices:
        idx = snapshot_indices.index(n)
        axes2[0].plot(x_rod, u, color=colors[idx], linewidth=2,
                      label=f't = {n*dt:.3f}s')

    if n < Nt:
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r_heat * (u[2:] - 2*u[1:-1] + u[:-2])
        u_new[0] = T_left
        u_new[-1] = T_right
        u = u_new

axes2[0].plot(x_rod, u_steady, 'k--', linewidth=2, label='Steady state')
axes2[0].set_xlabel('Position x (m)')
axes2[0].set_ylabel('Temperature (°C)')
axes2[0].set_title('Heat Conduction: Rod with T(0)=100°C, T(L)=25°C')
axes2[0].legend(fontsize=8)
axes2[0].grid(True, alpha=0.3)


# ── 5. 1D Diffusion ─────────────────────────────────────────
print("\n=== 1D Diffusion (Fick's Law) ===")

D_diff = 0.001   # diffusion coefficient
L_diff = 2.0     # domain length
Nx_diff = 100
dx_diff = L_diff / Nx_diff
dt_diff = 0.4 * dx_diff**2 / D_diff
Nt_diff = 8000

x_diff = np.linspace(-L_diff/2, L_diff/2, Nx_diff + 1)
r_diff = D_diff * dt_diff / dx_diff**2

# Initial condition: concentrated blob (approximating a delta function)
c = np.exp(-x_diff**2 / (2 * 0.01**2))
c = c / (np.sum(c) * dx_diff)  # normalize to unit total mass

snapshot_times_diff = [0, 0.01, 0.05, 0.1, 0.5, 2.0]
colors_diff = plt.cm.Blues(np.linspace(0.3, 1.0, len(snapshot_times_diff)))
snap_idx = 0
current_t = 0.0

for n in range(Nt_diff + 1):
    if snap_idx < len(snapshot_times_diff) and current_t >= snapshot_times_diff[snap_idx]:
        # Also plot analytical Gaussian for comparison
        if current_t > 0:
            c_exact = (1.0 / np.sqrt(4 * np.pi * D_diff * current_t)) * \
                      np.exp(-x_diff**2 / (4 * D_diff * current_t))
            axes2[1].plot(x_diff, c_exact, '--', color=colors_diff[snap_idx],
                          alpha=0.5, linewidth=1)
        axes2[1].plot(x_diff, c, color=colors_diff[snap_idx], linewidth=2,
                      label=f't = {current_t:.3f}')
        snap_idx += 1

    if n < Nt_diff:
        c_new = c.copy()
        c_new[1:-1] = c[1:-1] + r_diff * (c[2:] - 2*c[1:-1] + c[:-2])
        # Zero-flux boundary conditions (no substance escapes)
        c_new[0] = c_new[1]
        c_new[-1] = c_new[-2]
        c = c_new
        current_t += dt_diff

axes2[1].set_xlabel('Position x')
axes2[1].set_ylabel('Concentration c(x, t)')
axes2[1].set_title('Diffusion: Spreading of Concentrated Substance')
axes2[1].legend(fontsize=8)
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('applications_pde.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to applications_pde.png")


# ── 6. Phase Portrait: Lotka-Volterra ────────────────────────
print("\n=== Phase Portrait ===")

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

# Lotka-Volterra phase portrait with multiple trajectories
alpha_lv, beta_lv, delta_lv, gamma_lv = 1.0, 0.1, 0.075, 1.5

for x0, y0 in [(20, 5), (30, 8), (40, 9), (50, 12), (60, 15)]:
    sol = solve_ivp(lotka_volterra, (0, 30), [x0, y0],
                    dense_output=True, max_step=0.05)
    t_ph = np.linspace(0, 30, 2000)
    state_ph = sol.sol(t_ph)
    axes3[0].plot(state_ph[0], state_ph[1], linewidth=1.5, alpha=0.8)
    # Mark starting point
    axes3[0].plot(x0, y0, 'ko', markersize=5)

# Mark equilibrium
x_eq = gamma_lv / delta_lv
y_eq = alpha_lv / beta_lv
axes3[0].plot(x_eq, y_eq, 'r*', markersize=15, label=f'Equilibrium ({x_eq:.0f}, {y_eq:.0f})')
axes3[0].set_xlabel('Prey Population')
axes3[0].set_ylabel('Predator Population')
axes3[0].set_title('Lotka-Volterra Phase Portrait')
axes3[0].legend()
axes3[0].grid(True, alpha=0.3)

# Pendulum phase portrait
def pendulum_sys(t, state, g=9.81, L_pend=1.0):
    theta, omega = state
    return [omega, -(g / L_pend) * np.sin(theta)]

# Multiple energy levels show the transition from oscillation to rotation
for E_level in np.linspace(0.5, 25, 12):
    # Initial condition: theta=0, omega chosen to give desired energy
    # Energy: E = (1/2)*omega^2 + g/L*(1-cos(theta))
    omega0 = np.sqrt(2 * E_level)
    if omega0 < 0.1:
        continue
    sol_p = solve_ivp(pendulum_sys, (0, 10), [0, omega0],
                      dense_output=True, max_step=0.01)
    t_p = np.linspace(0, 10, 2000)
    state_p = sol_p.sol(t_p)
    axes3[1].plot(state_p[0], state_p[1], linewidth=1, alpha=0.7)

# Also plot the separatrix (the orbit separating oscillation from rotation)
# Energy at separatrix: E = 2*g/L
omega_sep = np.sqrt(2 * 9.81 * 2)  # energy = 2*g/L for separatrix
sol_sep = solve_ivp(pendulum_sys, (0, 20), [0.001, omega_sep],
                    dense_output=True, max_step=0.005)
t_sep = np.linspace(0, 20, 5000)
state_sep = sol_sep.sol(t_sep)
axes3[1].plot(state_sep[0], state_sep[1], 'r-', linewidth=2, alpha=0.8,
              label='Separatrix')

axes3[1].set_xlabel(r'$\theta$ (rad)')
axes3[1].set_ylabel(r'$\omega$ (rad/s)')
axes3[1].set_title('Pendulum Phase Portrait')
axes3[1].set_xlim(-2*np.pi, 2*np.pi)
axes3[1].legend()
axes3[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase_portraits.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to phase_portraits.png")
```

## 미니 프로젝트 제안

이 프로젝트들은 여러 레슨의 개념을 종합하며 심화 탐구에 적합합니다:

### 프로젝트 1: 전염병 모델링 (SIR 모델)

SIR 모델은 개체군을 감수성(Susceptible), 감염(Infected), 회복(Recovered)으로 나눕니다:

$$\frac{dS}{dt} = -\beta SI, \quad \frac{dI}{dt} = \beta SI - \gamma I, \quad \frac{dR}{dt} = \gamma I$$

- 기본 재생산수 $R_0 = \beta S_0 / \gamma$ 분석
- 백신 접종, 공간적 확산(PDE 버전), 또는 확률적 효과 추가
- 실제 전염병 데이터와 비교

### 프로젝트 2: 궤도 역학(Orbital Mechanics)

뉴턴의 만유인력 법칙을 사용한 행성 궤도 시뮬레이션:

$$\ddot{\mathbf{r}} = -\frac{GM}{|\mathbf{r}|^3}\mathbf{r}$$

- 케플러의 법칙을 수치적으로 검증
- 삼체 문제를 시뮬레이션하고 혼돈적 거동 관찰
- 장기 에너지 보존을 위한 심플렉틱 적분기(symplectic integrator) 구현

### 프로젝트 3: 진동하는 막(Vibrating Membrane)

원형 드럼에서의 2D 파동 방정식 풀기:

$$u_{tt} = c^2(u_{xx} + u_{yy})$$

- 해석적 해를 위해 베셀 함수(레슨 16) 사용
- 2D 유한 차분 풀이기 구현
- 드럼 모드를 애니메이션 표면으로 시각화

### 프로젝트 4: 화학 반응 네트워크(Chemical Reaction Networks)

강성 ODE를 가진 결합 화학 반응 모델링:

$$A \xrightarrow{k_1} B \xrightarrow{k_2} C$$

- 강성이 발생하는 조건 탐구 ($k_1 \gg k_2$ 또는 그 반대)
- 양해법과 음해법 구현 및 비교
- 가역 반응 추가 및 평형 농도 구하기

## 요약

이 종합 레슨은 미분방정식이 실제 시스템을 어떻게 모델링하는지 보여주었습니다:

| 응용 | 방정식 유형 | 핵심 개념 |
|------|-----------|----------|
| 로지스틱 성장 | 1계 ODE | 환경 수용 능력, S-곡선 |
| 포식자-피식자 | 연립 ODE | 주기적 궤도, 위상 초상 |
| RLC 회로 | 2계 ODE | 감쇠 영역, 공진 |
| 결합 용수철 | ODE 연립 | 정규 모드, 에너지 교환 |
| 비선형 진자 | 비선형 2계 ODE | 주기가 진폭에 의존 |
| 열전도 | 포물형 PDE | 확산, 정상 상태 |
| 피크 법칙 확산 | 포물형 PDE | 가우시안 퍼짐, $\sqrt{t}$ 법칙 |

모델링 과정 -- 물리를 방정식으로 번역하고, (해석적 또는 수치적으로) 풀고, 결과를 해석하기 -- 는 응용 수학의 핵심 기술입니다.

## 연습 문제

1. **수정된 로지스틱 모델**: 수확이 있는 로지스틱 방정식은 $P' = rP(1-P/K) - H$이며 여기서 $H$는 일정한 수확률입니다. $H$의 함수로 평형점을 구하세요. $H > rK/4$이면 초기 조건에 관계없이 개체군이 0으로 붕괴됨을 보이세요. 파이썬으로 시뮬레이션하고 분기(bifurcation)를 시각화하세요.

2. **경쟁하는 종**: 두 종이 같은 자원을 놓고 경쟁합니다: $x' = x(3 - x - 2y)$, $y' = y(2 - y - x)$. 모든 평형점을 구하고 야코비 행렬(Jacobian matrix)을 사용하여 안정성을 결정하세요. 위상 초상을 그리세요. 어떤 종이 승리하나요?

3. **RLC 공진**: $L = 1$, $R = 0.1$, $C = 0.04$인 RLC 회로에서 공진 진동수를 구하세요. 정현파 전압 $E(t) = \sin(\omega t)$를 인가하고 $\omega$의 함수로 정상 상태 진폭을 그리세요. $\omega$가 공진 진동수와 같을 때 무슨 일이 일어나나요? $R$을 증가시키면 피크에 어떤 영향이 있나요?

4. **열원이 있는 열 방정식**: $[0,1]$에서 $u(0,t) = u(1,t) = 0$이고 $u(x,0) = 0$인 $u_t = 0.01 u_{xx} + \sin(\pi x)$를 수치적으로 풀으세요. 열원 항은 균일한 내부 가열을 나타냅니다. 막대가 어떤 정상 상태 온도 프로파일에 접근하나요? $u_t = 0$으로 놓아 해석적으로 구하세요.

5. **확산 시간 척도**: 오염물이 1 km 호수의 중심에서 방출됩니다. 확산 계수가 $D = 10^{-5}$ m$^2$/s이면, $\sigma \approx \sqrt{2Dt}$ 관계를 사용하여 오염물이 100 m 거리까지 퍼지는 데 걸리는 시간을 추정하세요. 1D 확산 방정식을 시뮬레이션하고 100 m에서의 농도가 최대값의 10%에 도달하는 시점을 측정하여 수치적으로 검증하세요.

---

*이전: [미분방정식의 수치 해법](./19_Numerical_Methods_for_DE.md) | 다음: 이것은 미적분과 미분방정식 과정의 마지막 레슨입니다.*
