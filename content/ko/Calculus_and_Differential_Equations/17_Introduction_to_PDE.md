# 편미분방정식 입문(Introduction to Partial Differential Equations)

## 학습 목표

- 편미분방정식을 상미분방정식과 구별하고 그 차수와 선형성을 식별할 수 있다
- 판별식 방법을 사용하여 2계 선형 PDE를 타원형, 포물형, 쌍곡형으로 분류할 수 있다
- 물리적 원리로부터 열 방정식, 파동 방정식, 라플라스 방정식을 유도할 수 있다
- 적정 조건(well-posed) 문제를 위한 적절한 경계 조건(디리클레, 노이만, 로빈)과 초기 조건을 지정할 수 있다
- 기본적인 유한 차분법을 구현하여 열 확산과 파동 전파를 시각화할 수 있다

## 선수 과목

이 레슨을 학습하기 전에 다음 내용을 숙지해야 합니다:
- 다변수 미적분: 편미분, 연쇄 법칙 (레슨 3-4)
- ODE 풀이 기법 (레슨 10-15)
- 기본적인 선형대수 개념

## 편미분방정식이란 무엇인가?

**상미분방정식**(ODE, Ordinary Differential Equation)은 단일 변수의 함수와 그 도함수를 포함합니다. **편미분방정식**(PDE, Partial Differential Equation)은 **여러 변수**의 함수와 그 **편도함수**를 포함합니다.

비교:

| | ODE | PDE |
|---|-----|-----|
| 미지 함수 | $y(t)$ | $u(x, t)$, $u(x, y)$, $u(x, y, z, t)$ |
| 도함수 | $y', y'', \ldots$ | $u_t, u_x, u_{xx}, u_{xy}, \ldots$ |
| 변수 | 하나의 독립 변수 | 둘 이상의 독립 변수 |
| 해 | 곡선 | 곡면 또는 고차원 대상 |

이렇게 생각할 수 있습니다: ODE는 단일 축을 따라 무언가가 어떻게 변하는지(예: 시간에 따라 이동하는 입자)를 기술합니다. PDE는 공간 *그리고* 시간에 걸쳐 무언가가 동시에 어떻게 변하는지를 기술합니다 -- 예를 들어 금속 막대 안의 온도 분포나 진동하는 줄의 형상 같은 것입니다.

### 물리학의 예

PDE는 연속체 물리학의 언어입니다. 공간적 변화를 포함하는 거의 모든 물리 현상은 PDE로 귀결됩니다:

| PDE | 물리적 상황 |
|-----|------------|
| 열 방정식(Heat equation): $u_t = \alpha u_{xx}$ | 막대의 온도 |
| 파동 방정식(Wave equation): $u_{tt} = c^2 u_{xx}$ | 진동하는 줄, 음파 |
| 라플라스 방정식(Laplace equation): $u_{xx} + u_{yy} = 0$ | 정상 상태 온도, 정전기학 |
| 슈뢰딩거 방정식(Schrodinger equation): $i\hbar \psi_t = -\frac{\hbar^2}{2m}\nabla^2 \psi + V\psi$ | 양자 역학 |
| 나비에-스토크스 방정식(Navier-Stokes): $\rho(\mathbf{v}_t + \mathbf{v} \cdot \nabla \mathbf{v}) = -\nabla p + \mu \nabla^2 \mathbf{v}$ | 유체 역학 |
| 맥스웰 방정식(Maxwell's equations) | 전자기장 |

## 표기법

전체에 걸쳐 편도함수에 아래 첨자 표기법을 사용합니다:

$$u_t = \frac{\partial u}{\partial t}, \quad u_x = \frac{\partial u}{\partial x}, \quad u_{xx} = \frac{\partial^2 u}{\partial x^2}, \quad u_{xy} = \frac{\partial^2 u}{\partial x \partial y}$$

PDE의 **차수**(order)는 가장 높은 편도함수의 차수입니다. 미지 함수 $u$와 그 도함수가 1차 거듭제곱으로만 나타나고 서로 곱해지지 않으면 PDE는 **선형**(linear)입니다.

## 2계 선형 PDE의 분류

두 변수에서의 일반적인 2계 선형 PDE는 다음과 같습니다:

$$Au_{xx} + 2Bu_{xy} + Cu_{yy} + Du_x + Eu_y + Fu = G$$

여기서 $A, B, C, D, E, F, G$는 $x$와 $y$에 의존할 수 있습니다(선형성을 위해 $u$나 그 도함수에는 의존하지 않습니다).

**판별식**(discriminant) $\Delta = B^2 - AC$이 유형을 결정합니다:

| 판별식 | 유형 | 물리적 원형 | 거동 |
|--------|------|------------|------|
| $B^2 - AC < 0$ | **타원형**(Elliptic) | 라플라스 방정식 | 매끄러운 정상 상태 |
| $B^2 - AC = 0$ | **포물형**(Parabolic) | 열 방정식 | 확산적, 평활화 |
| $B^2 - AC > 0$ | **쌍곡형**(Hyperbolic) | 파동 방정식 | 전파, 보존적 |

이 분류는 원뿔 곡선(conic section)과 유사합니다: 동일한 판별식 $B^2 - AC$이 타원, 포물선, 쌍곡선을 구별합니다. 이것은 우연이 아닙니다 -- 분류는 PDE의 특성 곡선(characteristic curve)으로부터 나오며, 이것이 원뿔 곡선입니다.

### 분류가 중요한 이유

각 유형은 근본적으로 다른 물리적 거동을 가지며 다른 풀이 기법과 경계 조건을 필요로 합니다:

- **타원형**: 정보가 모든 방향으로 동시에 전파됩니다. 전체 경계에서 경계 조건이 필요합니다(경계값 문제).
- **포물형**: 정보가 무한한 속도로 시간 순방향으로 전파됩니다(확산). 초기 조건 + 경계 조건이 필요합니다(초기-경계값 문제).
- **쌍곡형**: 정보가 특성 곡선을 따라 유한한 속도로 전파됩니다. 위치와 속도에 대한 초기 조건이 필요합니다(초기값 문제).

### 분류 예제

**열 방정식**: $u_t = \alpha u_{xx}$. $\alpha u_{xx} - u_t = 0$으로 다시 쓰면 변수 $(x, t)$에 대해: $A = \alpha$, $B = 0$, $C = 0$. $\Delta = 0 - \alpha \cdot 0 = 0$. **포물형**.

**파동 방정식**: $u_{tt} = c^2 u_{xx}$. $c^2 u_{xx} - u_{tt} = 0$으로 다시 쓰면 $A = c^2$, $B = 0$, $C = -1$. $\Delta = 0 - c^2(-1) = c^2 > 0$. **쌍곡형**.

**라플라스 방정식**: $u_{xx} + u_{yy} = 0$. $A = 1$, $B = 0$, $C = 1$. $\Delta = 0 - 1 = -1 < 0$. **타원형**.

## 열 방정식(Heat Equation)

### 물리적 유도

길이 $L$인 가는 막대의 측면이 단열되어 열이 $x$축을 따라서만 흐르는 경우를 생각합니다.

$u(x, t)$를 위치 $x$, 시각 $t$에서의 온도라 합시다. 두 가지 물리 법칙으로부터 열 방정식을 유도합니다:

**에너지 보존**: 작은 구간 $[x, x + \Delta x]$에서 열 에너지의 변화율은 구간으로의 순 열 유속과 같습니다:

$$\rho c \frac{\partial u}{\partial t} \Delta x = q(x, t) - q(x + \Delta x, t)$$

여기서 $\rho$는 밀도, $c$는 비열, $q$는 열 유속(단위 면적당 단위 시간당 에너지)입니다.

**푸리에의 열전도 법칙**(Fourier's law): 열은 온도 기울기에 비례하여 고온에서 저온으로 흐릅니다:

$$q = -\kappa \frac{\partial u}{\partial x}$$

여기서 $\kappa > 0$은 열전도율입니다. 음의 부호는 열이 온도가 감소하는 방향으로 흐르도록 보장합니다.

이들을 결합하고 $\Delta x \to 0$으로 보내면:

$$\rho c \frac{\partial u}{\partial t} = \kappa \frac{\partial^2 u}{\partial x^2}$$

**열 확산율**(thermal diffusivity) $\alpha = \kappa / (\rho c)$를 정의하면:

$$\boxed{u_t = \alpha u_{xx}}$$

**직관적 해석**: 한 점에서의 온도 변화율은 온도 프로파일의 **곡률**(curvature)에 비례합니다. 온도 프로파일이 위로 볼록($u_{xx} > 0$)이면 그 점은 이웃보다 차가우므로 가열됩니다. 아래로 볼록하면 냉각됩니다. 이것은 "열은 고온에서 저온으로 흐른다"는 것의 수학적 표현입니다.

## 파동 방정식(Wave Equation)

### 물리적 유도

장력 $T$ 하에 있는 길이 $L$인 진동하는 줄을 생각합니다. 선밀도(단위 길이당 질량)는 $\mu$입니다. $u(x, t)$를 수직 변위라 합시다.

줄의 작은 구간에 대해 뉴턴의 제2법칙을 적용하면:

$$\mu \Delta x \frac{\partial^2 u}{\partial t^2} = T \sin\theta_2 - T \sin\theta_1$$

작은 변위에서 $\sin\theta \approx \tan\theta = \partial u / \partial x$이므로:

$$\mu \frac{\partial^2 u}{\partial t^2} = T \frac{\partial^2 u}{\partial x^2}$$

**파동 속도**(wave speed) $c = \sqrt{T/\mu}$를 정의하면:

$$\boxed{u_{tt} = c^2 u_{xx}}$$

**직관적 해석**: 각 점에서의 가속도는 줄의 곡률에 비례합니다. 이웃을 잇는 선 아래에 있는 점은 위로 당겨집니다(양의 곡률, 양의 가속도). 열 방정식과 달리 파동 방정식은 에너지를 보존합니다 -- 파동은 소산 없이 전파됩니다.

## 라플라스 방정식(Laplace Equation)

$$\boxed{u_{xx} + u_{yy} = 0} \qquad \text{(2D 라플라스 방정식)}$$

또는 3D에서: $u_{xx} + u_{yy} + u_{zz} = 0$, 흔히 $\nabla^2 u = 0$으로 씁니다.

라플라스 방정식은 **정상 상태**(steady-state) 상황을 기술합니다: 평형에 도달한 후의 판 안 온도 분포, 전하가 없는 영역의 전위, 또는 빈 공간에서의 중력 퍼텐셜 등입니다.

라플라스 방정식을 만족하는 함수를 **조화 함수**(harmonic function)라 합니다. 이들은 놀라운 **평균값 성질**(mean value property)을 가집니다: 어떤 점에서의 값은 그 점을 둘러싸는 임의의 구(또는 2D에서 원) 위의 평균과 같습니다.

## 경계 조건과 초기 조건

PDE만으로는 무한히 많은 해가 있습니다. 유일하고 물리적으로 의미 있는 해를 선택하기 위해 추가 조건이 필요합니다.

### 경계 조건(Boundary Conditions, BCs)

영역의 공간적 경계에 적용됩니다:

| 종류 | 이름 | 형태 | 물리적 의미 |
|------|------|------|------------|
| 제1종 | **디리클레**(Dirichlet) | $u = g$ (경계에서) | 지정된 온도/변위 |
| 제2종 | **노이만**(Neumann) | $\frac{\partial u}{\partial n} = h$ (경계에서) | 지정된 열 유속 ($h=0$이면 단열) |
| 제3종 | **로빈**(Robin) | $\alpha u + \beta \frac{\partial u}{\partial n} = g$ | 대류 냉각 (뉴턴의 냉각 법칙) |

여기서 $\partial u / \partial n$은 경계에서의 외향 법선 도함수입니다.

**예제**: 왼쪽 끝이 $100°$C로 유지되고 오른쪽 끝이 단열된 금속 막대:
- $u(0, t) = 100$ ($x = 0$에서 디리클레)
- $u_x(L, t) = 0$ ($x = L$에서 노이만; 유속이 0이면 단열)

### 초기 조건(Initial Conditions, ICs)

$t = 0$에서 적용됩니다:

- **열 방정식** ($t$에 대해 1계): $u(x, 0) = f(x)$ (초기 온도)가 필요
- **파동 방정식** ($t$에 대해 2계): $u(x, 0) = f(x)$ (초기 변위)와 $u_t(x, 0) = g(x)$ (초기 속도) 모두 필요

## 적정 조건(Well-Posedness, 아다마르 조건)

PDE 문제가 아다마르(Hadamard) 의미에서 **적정**(well-posed)이려면:

1. **존재성**(Existence): 해가 존재한다
2. **유일성**(Uniqueness): 해가 유일하다
3. **데이터에 대한 연속 의존성**(Continuous dependence on data): 데이터의 작은 변화가 해의 작은 변화를 야기한다

어느 조건이라도 실패하면 문제는 **부적정**(ill-posed)입니다. 예를 들어, 경계 조건이 너무 적으면 유일성이 없을 수 있고, 너무 많으면 해가 없을 수 있습니다. 역방향 열 방정식(backward heat equation)은 데이터의 미세한 섭동이 해의 거대한 변화를 야기하므로 부적정입니다.

## 파이썬 구현

```python
"""
Introduction to PDE: Classification and Basic Numerical Solutions.

This script demonstrates:
1. PDE classification using the discriminant
2. Heat equation simulation with finite differences
3. Wave equation simulation with finite differences
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ── 1. PDE Classification ────────────────────────────────────
def classify_pde(A, B, C):
    """
    Classify a second-order PDE: A*u_xx + 2B*u_xy + C*u_yy + ... = 0

    The discriminant Delta = B^2 - A*C determines the type.
    This is exactly the same discriminant that classifies conic sections,
    which is no coincidence: the classification comes from the shape of
    the characteristic curves.
    """
    discriminant = B**2 - A * C

    if discriminant < 0:
        return "Elliptic", discriminant
    elif discriminant == 0:
        return "Parabolic", discriminant
    else:
        return "Hyperbolic", discriminant


# Test with our three prototype equations
print("=== PDE Classification ===\n")
equations = [
    ("Heat equation (u_t = alpha*u_xx)", 1, 0, 0),
    ("Wave equation (u_tt = c^2*u_xx)", 1, 0, -1),
    ("Laplace equation (u_xx + u_yy = 0)", 1, 0, 1),
    ("Mixed PDE (u_xx + 2u_xy + u_yy = 0)", 1, 1, 1),
    ("Tricomi equation (y*u_xx + u_yy = 0, y>0)", 1, 0, 1),  # y > 0
]

for name, A, B, C in equations:
    pde_type, disc = classify_pde(A, B, C)
    print(f"{name}")
    print(f"  A={A}, B={B}, C={C} -> Delta = {disc} -> {pde_type}\n")


# ── 2. Heat Equation: u_t = alpha * u_xx ─────────────────────
def solve_heat_equation(L=1.0, T_final=0.5, alpha=0.01,
                        Nx=50, Nt=5000, u_left=0.0, u_right=0.0):
    """
    Solve the 1D heat equation using the explicit forward-time,
    central-space (FTCS) finite difference scheme.

    u_t = alpha * u_xx  on [0, L] x [0, T_final]

    The FTCS scheme:
        u[i,n+1] = u[i,n] + r * (u[i+1,n] - 2*u[i,n] + u[i-1,n])
    where r = alpha * dt / dx^2.

    STABILITY REQUIREMENT: r <= 0.5 (the CFL condition for heat equation).
    If violated, the numerical solution will oscillate and blow up.
    """
    dx = L / Nx
    dt = T_final / Nt
    r = alpha * dt / dx**2  # Stability parameter

    print(f"\nHeat equation: dx={dx:.4f}, dt={dt:.6f}, r={r:.4f}")
    if r > 0.5:
        print(f"WARNING: r={r:.4f} > 0.5! Scheme is UNSTABLE.")
    else:
        print(f"Stability OK: r={r:.4f} <= 0.5")

    x = np.linspace(0, L, Nx + 1)

    # Initial condition: a sine wave bump
    # This is a natural choice because sine functions are eigenfunctions
    # of the heat operator with Dirichlet BCs
    u = np.sin(np.pi * x / L)

    # Store snapshots at selected times for visualization
    snapshots = [(0, u.copy())]
    snapshot_times = [0.05, 0.1, 0.2, 0.5]
    next_snap_idx = 0

    for n in range(Nt):
        # FTCS update: explicit forward Euler in time, central differences in space
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])

        # Apply boundary conditions (Dirichlet)
        u_new[0] = u_left
        u_new[-1] = u_right

        u = u_new
        current_time = (n + 1) * dt

        # Capture snapshot if we've passed a snapshot time
        if (next_snap_idx < len(snapshot_times) and
                current_time >= snapshot_times[next_snap_idx]):
            snapshots.append((current_time, u.copy()))
            next_snap_idx += 1

    return x, snapshots


# Run heat equation simulation
x_heat, heat_snapshots = solve_heat_equation()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot heat equation evolution
colors = plt.cm.coolwarm(np.linspace(0, 1, len(heat_snapshots)))
for i, (time, u_snap) in enumerate(heat_snapshots):
    axes[0].plot(x_heat, u_snap, color=colors[i], linewidth=2,
                 label=f't = {time:.2f}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('u(x, t)')
axes[0].set_title('Heat Equation: Diffusion of Temperature')
axes[0].legend()
axes[0].grid(True, alpha=0.3)


# ── 3. Wave Equation: u_tt = c^2 * u_xx ─────────────────────
def solve_wave_equation(L=1.0, T_final=2.0, c=1.0, Nx=100, Nt=500):
    """
    Solve the 1D wave equation using an explicit finite difference scheme.

    u_tt = c^2 * u_xx  on [0, L] x [0, T_final]

    The scheme (second-order in both time and space):
        u[i,n+1] = 2*u[i,n] - u[i,n-1] + r^2*(u[i+1,n] - 2*u[i,n] + u[i-1,n])
    where r = c * dt / dx (the Courant number).

    STABILITY: r <= 1 (the CFL condition for the wave equation).
    """
    dx = L / Nx
    dt = T_final / Nt
    r = c * dt / dx  # Courant number

    print(f"\nWave equation: dx={dx:.4f}, dt={dt:.6f}, Courant r={r:.4f}")
    if r > 1.0:
        print(f"WARNING: Courant number r={r:.4f} > 1! Scheme is UNSTABLE.")
    else:
        print(f"Stability OK: Courant r={r:.4f} <= 1.0")

    x = np.linspace(0, L, Nx + 1)

    # Initial displacement: Gaussian bump centered at L/3
    # This asymmetric placement makes the wave dynamics more interesting
    u_prev = np.exp(-200 * (x - L/3)**2)  # u at time n-1
    u_curr = u_prev.copy()  # u at time n (zero initial velocity => same as u_prev)

    # Boundary conditions: fixed ends (Dirichlet)
    u_prev[0] = u_prev[-1] = 0
    u_curr[0] = u_curr[-1] = 0

    snapshots = [(0, u_curr.copy())]
    snapshot_times = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
    next_snap_idx = 0

    for n in range(Nt):
        # Central difference in both space and time
        u_next = np.zeros_like(u_curr)
        u_next[1:-1] = (2*u_curr[1:-1] - u_prev[1:-1] +
                        r**2 * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]))

        # Fixed boundary conditions
        u_next[0] = 0
        u_next[-1] = 0

        u_prev = u_curr
        u_curr = u_next
        current_time = (n + 1) * dt

        if (next_snap_idx < len(snapshot_times) and
                current_time >= snapshot_times[next_snap_idx]):
            snapshots.append((current_time, u_curr.copy()))
            next_snap_idx += 1

    return x, snapshots


# Run wave equation simulation
x_wave, wave_snapshots = solve_wave_equation()

colors_wave = plt.cm.viridis(np.linspace(0, 1, len(wave_snapshots)))
for i, (time, u_snap) in enumerate(wave_snapshots):
    axes[1].plot(x_wave, u_snap, color=colors_wave[i], linewidth=1.5,
                 label=f't = {time:.1f}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('u(x, t)')
axes[1].set_title('Wave Equation: Propagating Pulse')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('intro_pde_solutions.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to intro_pde_solutions.png")
```

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| PDE vs ODE | PDE는 여러 독립 변수를 포함 |
| 분류 | 판별식 $B^2 - AC$: 타원형(<0), 포물형(=0), 쌍곡형(>0) |
| 열 방정식 | $u_t = \alpha u_{xx}$; 확산적, 평활화 거동 |
| 파동 방정식 | $u_{tt} = c^2 u_{xx}$; 전파, 에너지 보존 |
| 라플라스 방정식 | $\nabla^2 u = 0$; 정상 상태, 평균값 성질 |
| 경계 조건 | 디리클레(값), 노이만(유속), 로빈(혼합) |
| 적정 조건 | 존재성 + 유일성 + 데이터에 대한 연속 의존성 |

세 가지 원형 방정식 -- 열, 파동, 라플라스 -- 은 PDE 이론의 기초를 형성합니다. 다음 레슨에서는 푸리에 급수를 사용하여 이들을 해석적으로 풀 것입니다. 포괄적인 수치 방법에 대해서는 [수치 시뮬레이션 - 유한 차분법](../Numerical_Simulation/06_Finite_Difference_PDE.md)과 이어지는 레슨을 참조하세요. 변수 분리법의 수학적 프레임워크에 대해서는 [물리수학 - PDE와 변수 분리법](../Mathematical_Methods/13_PDE_Separation_of_Variables.md)을 참조하세요.

## 연습 문제

1. **분류**: 각 PDE를 타원형, 포물형, 쌍곡형으로 분류하세요:
   - (a) $u_{xx} + 4u_{xy} + 4u_{yy} = 0$
   - (b) $u_{xx} - 4u_{xy} + 4u_{yy} = 0$
   - (c) $y \cdot u_{xx} + u_{yy} = 0$ ($y$의 부호에 따라 유형이 어떻게 달라지는지 논의하세요)

2. **유도**: 열원이 있는 1D 열 방정식을 유도하세요. 막대에 단위 체적당 $q(x, t)$ 와트의 내부 열원이 있다면, 방정식이 $u_t = \alpha u_{xx} + \frac{q}{\rho c}$가 됨을 보이세요. $q = q_0 \sin(\pi x / L)$을 생성하는 물리적 시나리오는 무엇인가요?

3. **경계 조건**: 길이 $L = 1$인 막대의 왼쪽 끝은 $0°$C이고 오른쪽 끝은 $20°$C인 주변 매질로 뉴턴의 냉각 법칙($-\kappa u_x(L,t) = h(u(L,t) - 20)$)을 통해 열을 잃습니다. 경계 조건을 형식적으로 쓰고 디리클레, 노이만, 로빈 중 어느 것인지 식별하세요.

4. **적정 조건**: 역방향 열 방정식 $u_t = -\alpha u_{xx}$ (부호에 주의)이 시간 순방향 전개에서 왜 부적정인지 설명하세요. 힌트: 푸리에 모드 $u(x,t) = e^{\alpha k^2 t} \sin(kx)$를 고려하고 고주파 섭동에 무슨 일이 일어나는지 살펴보세요.

5. **수치 실험**: 열 방정식 코드를 $r = 0.6$으로 수정하여(CFL 조건 위반) 시뮬레이션을 실행하고 무슨 일이 일어나는지 서술하세요. 그런 다음 무조건적으로 안정한 음해적 후진 오일러 기법(implicit backward Euler scheme)을 구현하고 $r = 0.6$에서도 작동하는지 확인하세요.

---

*이전: [멱급수 해법](./16_Power_Series_Solutions.md) | 다음: [푸리에 급수와 PDE](./18_Fourier_Series_and_PDE.md)*
