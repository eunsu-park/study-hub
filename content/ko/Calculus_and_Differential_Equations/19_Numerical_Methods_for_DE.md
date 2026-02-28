# 미분방정식의 수치 해법(Numerical Methods for Differential Equations)

## 학습 목표

- 테일러 전개로부터 오일러 방법을 유도하고 국소 및 전역 절단 오차를 분석할 수 있다
- 정확도 향상을 위해 룽게-쿠타 방법(RK2, RK4)을 구현하고 비교할 수 있다
- 적응적 단계 크기 제어를 적용하여 정확도와 계산 비용의 균형을 맞출 수 있다
- 강성 방정식(stiff equation)을 식별하고 양해법이 왜 실패하는지 설명할 수 있다
- ODE 연립을 수치적으로 풀고 실제 동역학 시스템(로렌츠 끌개, 진자)에 적용할 수 있다

## 선수 과목

이 레슨을 학습하기 전에 다음 내용을 숙지해야 합니다:
- 1계 및 2계 ODE (레슨 8-12)
- 테일러 급수 전개 (레슨 3-4)
- NumPy를 이용한 기본적인 파이썬 프로그래밍

## 왜 수치 해법인가?

대부분의 미분방정식은 해석적으로 풀 수 없습니다. 실제로:

- 삼체 문제, 난류 모델, 화학 반응 네트워크와 같은 **비선형 방정식**은 닫힌 형태의 해가 없습니다
- 기상 모델, 신경망, 생태 모델과 같은 많은 연립 방정식을 가진 **복잡한 시스템**은 계산이 필요합니다
- **실제 공학**에서는 공식이 아닌 숫자가 필요합니다 -- $y = C_1 e^t + C_2 te^t$이 아니라 $y(3.7) = 2.451$이 필요합니다

수치 방법은 이산 시간 단계에서 해를 근사합니다. 근본적인 트레이드오프는 **정확도 대 비용**입니다: 고차 방법은 단계당 더 정확하지만 더 많은 함수 평가를 필요로 합니다.

```
Exact solution: continuous curve y(t)
                  ╱╲
                ╱    ╲         ╱
              ╱        ╲     ╱
            ╱            ╲ ╱

Numerical: discrete points (t_n, y_n)
            •
          •   •
        •       •   •
      •           •
```

## 오일러 방법(Euler's Method)

### 전진 오일러: 유도

$y' = f(t, y)$, $y(t_0) = y_0$이 주어졌을 때, 이산점에서 $y(t)$를 근사하고자 합니다.

$t$ 주위에서 $y(t + h)$를 테일러 전개하면:

$$y(t + h) = y(t) + h \cdot y'(t) + \frac{h^2}{2} y''(t) + \cdots$$

1차 도함수 이후의 모든 항을 버리면:

$$y(t + h) \approx y(t) + h \cdot f(t, y(t))$$

이것이 **전진(양해) 오일러 방법**(forward/explicit Euler method)입니다:

$$\boxed{y_{n+1} = y_n + h \cdot f(t_n, y_n)}$$

여기서 $h$는 단계 크기(step size)이고 $y_n \approx y(t_n)$입니다.

**기하학적 해석**: 각 점에서 접선(기울기장 방향)을 따라 크기 $h$의 단계를 밟습니다. 이 방법은 "접선을 따라 미끄러지기"로 작동합니다 -- 작은 단계에서는 정확하지만 많은 단계에 걸쳐 오차가 누적됩니다.

### 오차 분석

- **국소 절단 오차**(local truncation error, 단계당 오차): $O(h^2)$ -- 테일러에서 $h^2$ 항을 버렸으므로
- **전역 오차**(global error, $[t_0, T]$에 걸친 누적 오차): $O(h)$ -- 모든 $N = (T - t_0)/h$ 단계의 오차가 누적

오일러 방법은 **1차**(first-order) 방법입니다. 오차를 절반으로 줄이려면 단계 크기를 절반으로 줄여야 하며, 계산량은 두 배가 됩니다.

### 후진 오일러: 음해법(Implicit Method)

$$y_{n+1} = y_n + h \cdot f(t_{n+1}, y_{n+1})$$

미지수 $y_{n+1}$이 양변에 나타납니다 -- 각 단계에서 방정식을 풀어야 합니다(보통 뉴턴 방법을 사용). 이 추가 비용으로 **무조건적 안정성**(unconditional stability)을 얻으며, 강성 방정식(아래에서 논의)에 필수적입니다.

## 룽게-쿠타 방법(Runge-Kutta Methods)

### 아이디어

오일러 방법은 구간 시작점에서 기울기를 한 번 취합니다. $[t_n, t_{n+1}]$ 내의 여러 점에서 기울기를 취하고 가중 평균을 구하면 어떨까요? 이것이 룽게-쿠타 접근법입니다.

언덕의 평균 높이를 추정하는 것에 비유할 수 있습니다. 오일러는 한쪽 끝에서만 측정하고, 룽게-쿠타는 길을 따라 여러 지점에서 샘플링하여 더 좋은 추정을 합니다.

### RK2: 중점법(Midpoint Method)

**전략**: 오일러를 사용하여 중점을 추정한 다음, 중점에서의 기울기로 전체 단계를 밟습니다.

$$k_1 = f(t_n, y_n)$$
$$k_2 = f\left(t_n + \frac{h}{2}, \; y_n + \frac{h}{2} k_1\right)$$
$$y_{n+1} = y_n + h \cdot k_2$$

- **국소 오차**: $O(h^3)$, **전역 오차**: $O(h^2)$
- 비용: 단계당 2회 함수 평가 (오일러는 1회)
- 정확도 향상: 같은 단계 크기에서 오일러보다 훨씬 좋음

### RK4: 고전적 4차 방법

수치 ODE 풀이의 주력입니다. 네 점에서 기울기를 평가합니다:

$$k_1 = f(t_n, y_n)$$
$$k_2 = f\left(t_n + \frac{h}{2}, \; y_n + \frac{h}{2}k_1\right)$$
$$k_3 = f\left(t_n + \frac{h}{2}, \; y_n + \frac{h}{2}k_2\right)$$
$$k_4 = f(t_n + h, \; y_n + h \cdot k_3)$$

$$\boxed{y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)}$$

여기서:
- $k_1$: 시작점에서의 기울기
- $k_2$: $k_1$로 $y$를 추정한 중점에서의 기울기
- $k_3$: 개선된 $k_2$ 추정치를 사용한 중점에서의 기울기
- $k_4$: $k_3$으로 $y$를 추정한 끝점에서의 기울기
- 최종 공식: 심프슨 법칙 가중치 $(1:2:2:1)/6$을 사용한 가중 평균

**오차**: 국소 $O(h^5)$, 전역 $O(h^4)$. 이는 $h$를 절반으로 줄이면 오차가 16분의 1로 줄어든다는 의미입니다. RK4는 단계당 4회 함수 평가가 필요하지만 매끄러운 문제에서 매우 정확합니다.

### 방법 비교

| 방법 | 차수 | 단계당 평가 | 국소 오차 | 전역 오차 |
|------|------|-----------|----------|----------|
| 전진 오일러(Forward Euler) | 1 | 1 | $O(h^2)$ | $O(h)$ |
| RK2 (중점법) | 2 | 2 | $O(h^3)$ | $O(h^2)$ |
| RK4 (고전적) | 4 | 4 | $O(h^5)$ | $O(h^4)$ |

## 적응적 단계 크기 제어(Adaptive Step Size Control)

### 왜 적응하는가?

고정 단계 크기는 낭비적입니다: 해가 어떤 영역에서는 느리게 변하고(큰 단계 허용) 다른 영역에서는 빠르게 변합니다(작은 단계 필요). 적응적 방법은 목표 정확도를 유지하기 위해 $h$를 자동으로 조정합니다.

### 내장 룽게-쿠타 방법(Embedded Runge-Kutta Methods)

가장 일반적인 접근법은 (대부분) 동일한 함수 평가를 사용하여 서로 다른 차수의 두 추정치를 계산합니다:

1. $p$차와 $(p+1)$차 해를 모두 계산
2. 국소 오차 추정: $\text{err} \approx |y_{p+1} - y_p|$
3. $\text{err} < \text{tol}$이면: 단계를 수용하고 더 큰 $h$를 시도
4. $\text{err} > \text{tol}$이면: 단계를 거부하고 $h$를 줄여 재시도

**도르만-프린스**(Dormand-Prince) 방법(`scipy.integrate.solve_ivp`에서 `method='RK45'`로 사용)은 비강성 문제의 현대적 표준인 내장 RK4(5) 쌍입니다.

새로운 단계 크기는 다음과 같이 추정됩니다:

$$h_{\text{new}} = h \cdot \left(\frac{\text{tol}}{\text{err}}\right)^{1/(p+1)}$$

수용과 거부 사이의 진동을 방지하기 위한 안전 계수(보통 0.9)가 포함됩니다.

## 강성 방정식(Stiff Equations)

### 강성이란 무엇인가?

**강성**(stiff) 방정식은 매우 다른 시간 척도로 진화하는 성분들을 가집니다. 고전적 예:

$$y' = -1000y + 1000, \quad y(0) = 0$$

정확한 해는 $y(t) = 1 - e^{-1000t}$입니다. 지수 함수는 거의 즉시 감쇠(시간 척도 $\sim 0.001$)하지만, 정상 상태는 영원히 지속됩니다. 양해법은 안정성을 위해 $h \ll 0.002$를 사용해야 하며, $t > 0.01$ 이후에는 해가 본질적으로 일정한데도 그렇습니다.

**비유**: 공이 가파른 계곡으로 굴러 내려간 다음 평평한 바닥을 따라 가는 것을 상상하세요. 양해법은 가파른 하강 중에 (안정성을 위해) 미세한 단계를 밟아야 하며, 그 동일한 미세한 단계가 길고 지루한 평평한 구간에서도 강제됩니다. 음해법은 정상 상태를 직접 "보기" 때문에 큰 단계를 취할 수 있습니다.

### 양해법이 실패하는 이유

$y' = \lambda y$에서 $\lambda < 0$인 경우, 전진 오일러는 $y_{n+1} = (1 + h\lambda) y_n$을 줍니다. 안정성을 위해 $|1 + h\lambda| < 1$이 필요하며, 이는 $h < 2/|\lambda|$를 요구합니다. $|\lambda| = 1000$이면 정확도 요구에 관계없이 $h < 0.002$가 강제됩니다.

### 강성 문제를 위한 음해법

$y' = \lambda y$에 적용된 후진 오일러 방법은 $y_{n+1} = y_n / (1 - h\lambda)$를 주며, $\lambda < 0$일 때 모든 $h > 0$에서 안정합니다. 이것이 **A-안정성**(A-stability) 또는 **무조건적 안정성**(unconditional stability)입니다.

실제로 `scipy.integrate.solve_ivp`에서 `method='Radau'`나 `method='BDF'`는 강성 문제에 최적화된 고차 음해법을 사용합니다.

## ODE 연립(Systems of ODE)

스칼라 ODE에 대한 모든 방법은 자연스럽게 연립으로 확장됩니다:

$$\mathbf{y}' = \mathbf{f}(t, \mathbf{y}), \quad \mathbf{y}(t_0) = \mathbf{y}_0$$

여기서 $\mathbf{y} = [y_1, y_2, \ldots, y_m]^T$입니다. RK4 공식은 벡터 $\mathbf{k}_i$ 값으로 적용됩니다.

고계 ODE는 연립으로 변환됩니다. 예를 들어 $y'' + y = 0$은 다음이 됩니다:

$$\begin{cases} y_1' = y_2 \\ y_2' = -y_1 \end{cases} \quad \text{여기서 } y_1 = y, \; y_2 = y'$$

## PDE 수치 방법 간단 소개

이 레슨은 ODE에 초점을 맞추지만, 많은 PDE는 공간을 이산화하여 큰 ODE 연립을 얻은 다음 위의 방법을 적용하여 풀립니다. **방법선법**(method of lines) 접근법:

1. 유한 차분으로 공간 도함수를 이산화 (중심 차분: $u_{xx} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$)
2. 이것이 PDE를 시간에 대한 ODE 연립으로 변환
3. 오일러, RK4, 또는 음해법을 적용하여 시간을 전진

PDE 수치 방법의 포괄적 내용은 [수치 시뮬레이션 - 유한 차분 PDE](../Numerical_Simulation/06_Finite_Difference_PDE.md)를 참조하세요.

## 파이썬 구현

```python
"""
Numerical Methods for Differential Equations.

This script implements and compares:
1. Forward Euler, RK2, and RK4 on a test problem
2. Error convergence analysis
3. Lorenz attractor (chaotic system)
4. Stiff equation comparison (explicit vs implicit)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp


# ── Core Methods ─────────────────────────────────────────────
def euler_forward(f, t_span, y0, h):
    """
    Forward Euler method: y_{n+1} = y_n + h * f(t_n, y_n).

    The simplest ODE solver. First-order accurate.
    Use only as a baseline for comparison — RK4 is almost
    always a better choice for the same computational cost.

    Parameters:
        f: function(t, y) -> dy/dt (can be scalar or vector)
        t_span: (t_start, t_end)
        y0: initial condition (scalar or array)
        h: step size

    Returns:
        t_vals, y_vals (arrays)
    """
    t_start, t_end = t_span
    t_vals = np.arange(t_start, t_end + h/2, h)
    y0 = np.atleast_1d(np.array(y0, dtype=float))
    y_vals = np.zeros((len(t_vals), len(y0)))
    y_vals[0] = y0

    for i in range(len(t_vals) - 1):
        y_vals[i + 1] = y_vals[i] + h * np.array(f(t_vals[i], y_vals[i]))

    return t_vals, y_vals


def rk2_midpoint(f, t_span, y0, h):
    """
    RK2 Midpoint method. Second-order accurate.

    Idea: Use Euler to get to the midpoint, then use the slope
    there for the full step. This cancels the leading error term.
    """
    t_start, t_end = t_span
    t_vals = np.arange(t_start, t_end + h/2, h)
    y0 = np.atleast_1d(np.array(y0, dtype=float))
    y_vals = np.zeros((len(t_vals), len(y0)))
    y_vals[0] = y0

    for i in range(len(t_vals) - 1):
        k1 = np.array(f(t_vals[i], y_vals[i]))
        k2 = np.array(f(t_vals[i] + h/2, y_vals[i] + h/2 * k1))
        y_vals[i + 1] = y_vals[i] + h * k2

    return t_vals, y_vals


def rk4_classic(f, t_span, y0, h):
    """
    Classic RK4 method. Fourth-order accurate.

    The "gold standard" of explicit ODE solvers.
    Four function evaluations per step, but the O(h^4)
    accuracy makes it extremely efficient for smooth problems.
    """
    t_start, t_end = t_span
    t_vals = np.arange(t_start, t_end + h/2, h)
    y0 = np.atleast_1d(np.array(y0, dtype=float))
    y_vals = np.zeros((len(t_vals), len(y0)))
    y_vals[0] = y0

    for i in range(len(t_vals) - 1):
        t = t_vals[i]
        y = y_vals[i]

        k1 = np.array(f(t, y))
        k2 = np.array(f(t + h/2, y + h/2 * k1))
        k3 = np.array(f(t + h/2, y + h/2 * k2))
        k4 = np.array(f(t + h, y + h * k3))

        # Simpson's rule weights: 1/6 * (1 + 2 + 2 + 1)
        y_vals[i + 1] = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return t_vals, y_vals


# ── 1. Comparison on a test problem ──────────────────────────
# y' = -2ty, y(0) = 1  =>  exact solution: y = exp(-t^2)
print("=== Method Comparison: y' = -2ty, y(0) = 1 ===\n")

f_test = lambda t, y: -2 * t * y
exact = lambda t: np.exp(-t**2)
t_span = (0, 3)
y0 = 1.0
h = 0.2

t_euler, y_euler = euler_forward(f_test, t_span, y0, h)
t_rk2, y_rk2 = rk2_midpoint(f_test, t_span, y0, h)
t_rk4, y_rk4 = rk4_classic(f_test, t_span, y0, h)

t_exact = np.linspace(0, 3, 300)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Solution comparison
axes[0].plot(t_exact, exact(t_exact), 'k-', linewidth=2, label='Exact')
axes[0].plot(t_euler, y_euler[:, 0], 'ro-', markersize=4, label=f'Euler (h={h})')
axes[0].plot(t_rk2, y_rk2[:, 0], 'gs-', markersize=4, label=f'RK2 (h={h})')
axes[0].plot(t_rk4, y_rk4[:, 0], 'b^-', markersize=4, label=f'RK4 (h={h})')
axes[0].set_xlabel('t')
axes[0].set_ylabel('y(t)')
axes[0].set_title('Solution Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Error comparison
axes[1].semilogy(t_euler, np.abs(y_euler[:, 0] - exact(t_euler)),
                 'ro-', markersize=4, label='Euler')
axes[1].semilogy(t_rk2, np.abs(y_rk2[:, 0] - exact(t_rk2)),
                 'gs-', markersize=4, label='RK2')
axes[1].semilogy(t_rk4, np.abs(y_rk4[:, 0] - exact(t_rk4)),
                 'b^-', markersize=4, label='RK4')
axes[1].set_xlabel('t')
axes[1].set_ylabel('|error|')
axes[1].set_title('Absolute Error (log scale)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# ── 2. Convergence Study ────────────────────────────────────
h_values = [0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
errors_euler = []
errors_rk2 = []
errors_rk4 = []

for h_val in h_values:
    # Measure error at t=2
    _, y_e = euler_forward(f_test, t_span, y0, h_val)
    _, y_r2 = rk2_midpoint(f_test, t_span, y0, h_val)
    _, y_r4 = rk4_classic(f_test, t_span, y0, h_val)

    # Find index closest to t=2
    idx = int(2.0 / h_val)
    errors_euler.append(abs(y_e[idx, 0] - exact(2.0)))
    errors_rk2.append(abs(y_r2[idx, 0] - exact(2.0)))
    errors_rk4.append(abs(y_r4[idx, 0] - exact(2.0)))

axes[2].loglog(h_values, errors_euler, 'ro-', label='Euler (slope~1)')
axes[2].loglog(h_values, errors_rk2, 'gs-', label='RK2 (slope~2)')
axes[2].loglog(h_values, errors_rk4, 'b^-', label='RK4 (slope~4)')
# Reference slopes
axes[2].loglog(h_values, [h**1 * 0.5 for h in h_values], 'r:', alpha=0.5)
axes[2].loglog(h_values, [h**2 * 0.5 for h in h_values], 'g:', alpha=0.5)
axes[2].loglog(h_values, [h**4 * 0.5 for h in h_values], 'b:', alpha=0.5)
axes[2].set_xlabel('Step size h')
axes[2].set_ylabel('Error at t=2')
axes[2].set_title('Convergence (error vs step size)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('numerical_methods_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to numerical_methods_comparison.png")


# ── 3. Lorenz Attractor (Chaotic System) ────────────────────
print("\n=== Lorenz Attractor ===")

def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    The Lorenz system — the first discovered example of deterministic chaos.

    dx/dt = sigma * (y - x)       # convective heat transfer
    dy/dt = x * (rho - z) - y     # temperature difference driving
    dz/dt = x * y - beta * z      # nonlinear interaction

    For sigma=10, rho=28, beta=8/3, the system exhibits the famous
    "butterfly" attractor — sensitive dependence on initial conditions.
    """
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]


# Solve using our RK4 implementation
t_span_lorenz = (0, 50)
y0_lorenz = [1.0, 1.0, 1.0]
h_lorenz = 0.01  # Small step needed for chaotic system

t_lorenz, y_lorenz = rk4_classic(lorenz, t_span_lorenz, y0_lorenz, h_lorenz)

# 3D plot of the attractor
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(y_lorenz[:, 0], y_lorenz[:, 1], y_lorenz[:, 2],
        linewidth=0.5, alpha=0.8, color='steelblue')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor (solved with RK4, h=0.01)')

plt.tight_layout()
plt.savefig('lorenz_attractor.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to lorenz_attractor.png")


# ── 4. Nonlinear Pendulum ───────────────────────────────────
print("\n=== Nonlinear Pendulum ===")

def pendulum(t, state, g=9.81, L=1.0):
    """
    Nonlinear pendulum: theta'' + (g/L) sin(theta) = 0

    Converted to first-order system:
        theta' = omega
        omega' = -(g/L) sin(theta)

    For small angles, sin(theta) ~ theta gives SHM.
    For large angles, the nonlinearity matters significantly.
    """
    theta, omega = state
    return [omega, -(g / L) * np.sin(theta)]


fig, ax = plt.subplots(figsize=(10, 6))

# Compare different initial angles
# Small angle gives nearly perfect sinusoidal motion
# Large angle shows period elongation and anharmonicity
for theta0_deg in [10, 45, 90, 135, 170]:
    theta0 = np.radians(theta0_deg)
    t_pend, y_pend = rk4_classic(pendulum, (0, 10), [theta0, 0.0], 0.01)
    ax.plot(t_pend, np.degrees(y_pend[:, 0]),
            linewidth=1.5, label=f'{theta0_deg} degrees')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (degrees)')
ax.set_title('Nonlinear Pendulum: Period Depends on Amplitude')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig('nonlinear_pendulum.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to nonlinear_pendulum.png")


# ── 5. Stiff Equation ───────────────────────────────────────
print("\n=== Stiff Equation: y' = -1000(y - sin(t)) + cos(t) ===")

def stiff_rhs(t, y):
    """
    A stiff ODE with exact solution y(t) = sin(t).

    The -1000 coefficient creates a fast transient that decays
    in ~0.003 seconds, but explicit methods need h < 0.002
    for stability — even when the solution is barely changing.
    """
    return -1000 * (y - np.sin(t)) + np.cos(t)


# Forward Euler with different step sizes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
t_exact_stiff = np.linspace(0, 0.1, 1000)

for h_stiff in [0.0018, 0.0015, 0.001]:
    t_s, y_s = euler_forward(stiff_rhs, (0, 0.1), [0.0], h_stiff)
    axes[0].plot(t_s, y_s[:, 0], '-o', markersize=2,
                 label=f'Euler h={h_stiff}')

axes[0].plot(t_exact_stiff, np.sin(t_exact_stiff), 'k-', linewidth=2,
             label='Exact: sin(t)')
axes[0].set_xlabel('t')
axes[0].set_ylabel('y')
axes[0].set_title('Stiff ODE: Forward Euler (stability issues)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.5, 0.5)

# SciPy implicit solver handles stiffness easily
sol_radau = solve_ivp(stiff_rhs, (0, 2), [0.0], method='Radau',
                      max_step=0.1, rtol=1e-8)
sol_rk45 = solve_ivp(stiff_rhs, (0, 2), [0.0], method='RK45',
                      max_step=0.01, rtol=1e-8)

t_ex = np.linspace(0, 2, 500)
axes[1].plot(t_ex, np.sin(t_ex), 'k-', linewidth=2, label='Exact')
axes[1].plot(sol_radau.t, sol_radau.y[0], 'b.-', label=f'Radau ({len(sol_radau.t)} steps)')
axes[1].plot(sol_rk45.t, sol_rk45.y[0], 'r.-', alpha=0.5,
             label=f'RK45 ({len(sol_rk45.t)} steps)')
axes[1].set_xlabel('t')
axes[1].set_ylabel('y')
axes[1].set_title('Stiff ODE: Implicit (Radau) vs Explicit (RK45)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stiff_equation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to stiff_equation.png")
print(f"\nRadau used {len(sol_radau.t)} steps, RK45 used {len(sol_rk45.t)} steps")
```

## 요약

| 방법 | 차수 | 적합한 용도 | 한계 |
|------|------|-----------|------|
| 전진 오일러(Forward Euler) | 1 | 교육, 간단한 문제 | 큰 오차, 안정성 문제 |
| 후진 오일러(Backward Euler) | 1 | 강성 문제 (기본) | 각 단계마다 비선형 풀이 필요 |
| RK2 (중점법) | 2 | 중간 정확도 요구 | RK4만큼 효율적이지 않음 |
| RK4 (고전적) | 4 | 일반 비강성 문제 | 강성 방정식에 실패 |
| RK45 (도르만-프린스) | 4-5 | 적응적 비강성 | 강성 문제에 부적합 |
| Radau / BDF | 고차 | 강성 문제 | 단계당 비용이 높음 |

**실용적 조언**: 대부분의 문제에서 `scipy.integrate.solve_ivp`를 `method='RK45'`로 시작하세요. 극도로 작은 단계가 필요하거나 예기치 않은 결과가 나오면 강성을 의심하고 `method='Radau'`로 전환하세요.

ODE 및 PDE 수치 방법의 포괄적 내용(고차 기법 포함)은 [수치 시뮬레이션](../Numerical_Simulation/03_ODE_Solvers.md)과 이어지는 레슨을 참조하세요.

## 연습 문제

1. **오일러 vs RK4**: $[0, 10]$에서 $y' = y \sin(t)$, $y(0) = 1$을 오일러와 RK4 모두 $h = 0.1$로 풀으세요. 정확한 해는 $y = e^{1-\cos t}$입니다. 두 방법의 오차를 그리세요. 오일러가 $t = 10$에서 RK4의 $h = 0.1$과 같은 정확도를 달성하려면 어떤 단계 크기가 필요한가요?

2. **수렴 속도**: $y' = -y + \sin(t)$, $y(0) = 0$ 문제에서 단계 크기 $h = 0.5, 0.25, 0.1, 0.05, 0.01$에 대해 $t = 5$에서의 전역 오차를 오일러, RK2, RK4로 계산하세요. 로그-로그 스케일에서 오차 대 $h$를 그리고 기울기를 측정하여 이론적 수렴 속도를 검증하세요.

3. **로렌츠 민감도**: $x$ 성분이 $10^{-10}$만큼 다른 두 초기 조건으로 로렌츠 시스템을 풀으세요. 두 궤적과 그 사이의 거리를 시간의 함수로 그리세요. 거리의 지수적 성장률로부터 리아푸노프 지수(Lyapunov exponent)를 추정하세요.

4. **강성 탐지**: 반 데어 폴 진동자(Van der Pol oscillator) $y'' - \mu(1 - y^2)y' + y = 0$은 큰 $\mu$에서 강성이 됩니다. 연립으로 변환하고 $\mu = 1, 10, 100, 1000$에 대해 `RK45`와 `Radau`로 풀으세요. 함수 평가 횟수를 비교하세요. 어떤 $\mu$ 값에서 `RK45`가 비실용적으로 느려지나요?

5. **적응적 단계 크기**: 국소 오차가 $\text{tol}/10$ 미만이면 $h$를 두 배로 하고 $\text{tol}$ 초과이면 $h$를 절반으로 하는 간단한 적응적 오일러 방법을 구현하세요. $y' = -50(y - \cos t)$, $y(0) = 0$에서 $\text{tol} = 10^{-4}$로 테스트하세요. 시간의 함수로 단계 크기를 그리고 패턴을 설명하세요.

---

*이전: [푸리에 급수와 PDE](./18_Fourier_Series_and_PDE.md) | 다음: [응용과 모델링](./20_Applications_and_Modeling.md)*
