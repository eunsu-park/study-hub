# Applications and Modeling

## Learning Objectives

- Formulate mathematical models for population dynamics, electrical circuits, and mechanical systems using differential equations
- Analyze the qualitative behavior of nonlinear systems (equilibria, stability, phase portraits)
- Solve coupled ODE systems numerically and interpret the results physically
- Apply PDE models (heat conduction, diffusion) to real-world boundary value problems
- Synthesize techniques from the entire course to model and simulate complete physical scenarios

## Prerequisites

This capstone lesson draws on the full course. Key prerequisites:
- First-order ODE and separable equations (Lessons 8-9)
- Second-order linear ODE (Lessons 10-12)
- Systems of ODE (Lesson 14)
- Numerical methods (Lesson 19)
- Basic PDE concepts (Lesson 17)

## Motivation: From Equations to Understanding

Throughout this course, we have built a toolkit: integration, differential equations, series solutions, transforms, Fourier methods, and numerical algorithms. In this final lesson, we bring everything together by modeling real systems from biology, engineering, and physics.

The modeling cycle is:

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

## Population Dynamics

### The Logistic Model

The simplest population model is exponential growth: $P' = rP$, giving $P(t) = P_0 e^{rt}$. This is unrealistic for large populations because resources are finite.

The **logistic equation** adds a carrying capacity $K$:

$$\frac{dP}{dt} = rP\left(1 - \frac{P}{K}\right)$$

where:
- $P(t)$: population at time $t$
- $r$: intrinsic growth rate (births minus deaths per capita when $P \ll K$)
- $K$: carrying capacity (maximum sustainable population)
- The factor $(1 - P/K)$: slows growth as $P$ approaches $K$

**Equilibria**: $P^* = 0$ (extinction, unstable) and $P^* = K$ (stable carrying capacity).

**Exact solution** (separable equation):

$$P(t) = \frac{K}{1 + \left(\frac{K}{P_0} - 1\right)e^{-rt}}$$

This is the famous **logistic curve** or S-curve: slow initial growth, rapid middle growth, then saturation. It models everything from bacterial colonies to technology adoption.

### Predator-Prey: The Lotka-Volterra System

When two species interact -- one eating the other -- we get a coupled system:

$$\frac{dx}{dt} = \alpha x - \beta xy \qquad \text{(prey)}$$

$$\frac{dy}{dt} = \delta xy - \gamma y \qquad \text{(predator)}$$

where:
- $x(t)$: prey population (e.g., rabbits)
- $y(t)$: predator population (e.g., foxes)
- $\alpha$: prey birth rate (in the absence of predators)
- $\beta$: predation rate (encounters between predator and prey)
- $\delta$: predator reproduction rate (proportional to prey consumed)
- $\gamma$: predator death rate (in the absence of prey)

**Equilibria**:
1. $(0, 0)$: both species extinct
2. $(\gamma/\delta, \alpha/\beta)$: coexistence equilibrium

The remarkable feature: solutions are **periodic orbits** around the coexistence equilibrium. When prey is abundant, predators thrive and multiply; then predators over-consume the prey, prey declines, predators decline from starvation, prey recovers, and the cycle repeats. This was first observed in the historical lynx-hare population data from the Hudson's Bay Company.

## Electrical Circuits: The RLC Circuit

### Second-Order ODE from Circuit Laws

An RLC series circuit with a voltage source $E(t)$ is governed by Kirchhoff's voltage law:

$$L\frac{d^2q}{dt^2} + R\frac{dq}{dt} + \frac{1}{C}q = E(t)$$

where:
- $q(t)$: charge on the capacitor (coulombs)
- $L$: inductance (henrys) -- opposes changes in current (electrical inertia)
- $R$: resistance (ohms) -- dissipates energy as heat (friction)
- $C$: capacitance (farads) -- stores energy in the electric field (spring)
- $E(t)$: applied voltage (driving force)

The current is $I = dq/dt$, so equivalently:

$$L\frac{dI}{dt} + RI + \frac{1}{C}\int_0^t I(\tau) \, d\tau = E(t)$$

**Mechanical analogy**: This equation is identical in form to a damped driven harmonic oscillator:

| Mechanical | Electrical |
|-----------|-----------|
| Mass $m$ | Inductance $L$ |
| Damping $b$ | Resistance $R$ |
| Spring constant $k$ | $1/C$ |
| Displacement $x$ | Charge $q$ |
| Applied force $F(t)$ | Voltage $E(t)$ |

### Behavior Regimes

The characteristic equation $Ls^2 + Rs + 1/C = 0$ gives roots:

$$s = \frac{-R \pm \sqrt{R^2 - 4L/C}}{2L}$$

- **Overdamped** ($R^2 > 4L/C$): Two real negative roots. Charge decays without oscillation.
- **Critically damped** ($R^2 = 4L/C$): Repeated root. Fastest decay without oscillation.
- **Underdamped** ($R^2 < 4L/C$): Complex conjugate roots. Decaying oscillations.

When $R = 0$ (no resistance): undamped oscillations at the **resonant frequency** $\omega_0 = 1/\sqrt{LC}$.

## Mechanical Systems

### Coupled Spring-Mass System

Two masses connected by springs:

```
   Wall ──|/\/\/|── m₁ ──|/\/\/|── m₂ ──|/\/\/|── Wall
           k₁              k₂              k₃
```

Let $x_1, x_2$ be the displacements from equilibrium. Newton's second law:

$$m_1 x_1'' = -k_1 x_1 + k_2(x_2 - x_1)$$
$$m_2 x_2'' = -k_2(x_2 - x_1) - k_3 x_2$$

In matrix form: $\mathbf{M}\mathbf{x}'' = -\mathbf{K}\mathbf{x}$, where $\mathbf{M}$ and $\mathbf{K}$ are the mass and stiffness matrices. The eigenvalues of $\mathbf{M}^{-1}\mathbf{K}$ give the **normal mode frequencies**, and the eigenvectors give the **normal modes** (patterns of oscillation where all parts move at the same frequency).

### The Nonlinear Pendulum

The full (nonlinear) pendulum equation:

$$\frac{d^2\theta}{dt^2} + \frac{g}{L}\sin\theta = 0$$

For small angles, $\sin\theta \approx \theta$ gives simple harmonic motion with period $T_0 = 2\pi\sqrt{L/g}$. For large angles, the period increases -- a pendulum released from near-vertical takes longer to complete one cycle than a gently swinging one.

The exact period involves an **elliptic integral**:

$$T = 4\sqrt{\frac{L}{g}} \int_0^{\pi/2} \frac{d\phi}{\sqrt{1 - \sin^2(\theta_0/2)\sin^2\phi}}$$

This is a beautiful example where the linear approximation gives a clean answer, but the full physics requires either numerical methods or special functions.

## Heat Conduction: 1D Rod

### Problem Setup

A metal rod of length $L$ with the left end held at temperature $T_1$, the right end at $T_2$, and an initial temperature distribution $f(x)$:

$$u_t = \alpha u_{xx}, \quad 0 < x < L, \quad t > 0$$
$$u(0, t) = T_1, \quad u(L, t) = T_2, \quad u(x, 0) = f(x)$$

### Steady-State Solution

As $t \to \infty$, $u_t \to 0$, so the steady state satisfies $u_{xx} = 0$:

$$u_{\infty}(x) = T_1 + (T_2 - T_1)\frac{x}{L}$$

A linear temperature profile -- heat flows uniformly from the hot end to the cold end.

### Transient Solution

Write $u(x, t) = u_{\infty}(x) + v(x, t)$ where $v$ satisfies the heat equation with **homogeneous** BCs ($v(0,t) = v(L,t) = 0$). Then $v$ is solved by Fourier series (Lesson 18).

## Diffusion: Fick's Law

Diffusion of a substance (ink in water, pollutant in air) follows the same equation as heat conduction:

$$\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2}$$

where:
- $c(x, t)$: concentration of the substance
- $D$: diffusion coefficient ($\text{m}^2/\text{s}$)

Fick's law ($J = -D \, \partial c / \partial x$) is the mass-transport analog of Fourier's law of heat conduction. The mathematics is identical.

**Fundamental solution**: If a unit amount of substance is released at $x = 0$ at $t = 0$:

$$c(x, t) = \frac{1}{\sqrt{4\pi D t}} \exp\left(-\frac{x^2}{4Dt}\right)$$

This Gaussian spreads and flattens over time: the width grows as $\sqrt{Dt}$ (the square root is key to understanding diffusion time scales).

## Python Implementation

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

## Mini-Project Suggestions

These projects synthesize concepts from multiple lessons and are suitable for extended exploration:

### Project 1: Epidemic Modeling (SIR Model)

The SIR model divides a population into Susceptible, Infected, and Recovered:

$$\frac{dS}{dt} = -\beta SI, \quad \frac{dI}{dt} = \beta SI - \gamma I, \quad \frac{dR}{dt} = \gamma I$$

- Analyze the basic reproduction number $R_0 = \beta S_0 / \gamma$
- Add vaccination, spatial diffusion (PDE version), or stochastic effects
- Compare with real epidemic data

### Project 2: Orbital Mechanics

Simulate planetary orbits using Newton's gravitational law:

$$\ddot{\mathbf{r}} = -\frac{GM}{|\mathbf{r}|^3}\mathbf{r}$$

- Verify Kepler's laws numerically
- Simulate the three-body problem and observe chaotic behavior
- Implement a symplectic integrator for long-term energy conservation

### Project 3: Vibrating Membrane

Solve the 2D wave equation on a circular drum:

$$u_{tt} = c^2(u_{xx} + u_{yy})$$

- Use Bessel functions (Lesson 16) for the analytical solution
- Implement a 2D finite difference solver
- Visualize the drum modes as animated surfaces

### Project 4: Chemical Reaction Networks

Model coupled chemical reactions with stiff ODE:

$$A \xrightarrow{k_1} B \xrightarrow{k_2} C$$

- Explore when stiffness arises ($k_1 \gg k_2$ or vice versa)
- Implement and compare explicit and implicit solvers
- Add reversible reactions and find equilibrium concentrations

## Summary

This capstone lesson demonstrated how differential equations model real systems:

| Application | Equation Type | Key Concept |
|------------|--------------|-------------|
| Logistic growth | 1st-order ODE | Carrying capacity, S-curve |
| Predator-prey | Coupled ODE system | Periodic orbits, phase portrait |
| RLC circuit | 2nd-order ODE | Damping regimes, resonance |
| Coupled springs | ODE system | Normal modes, energy exchange |
| Nonlinear pendulum | Nonlinear 2nd-order ODE | Period depends on amplitude |
| Heat conduction | Parabolic PDE | Diffusion, steady state |
| Fick's law diffusion | Parabolic PDE | Gaussian spreading, $\sqrt{t}$ law |

The modeling process -- translating physics into equations, solving (analytically or numerically), and interpreting results -- is the core skill of applied mathematics.

## Practice Problems

1. **Modified logistic model**: The logistic equation with harvesting is $P' = rP(1-P/K) - H$ where $H$ is a constant harvest rate. Find the equilibria as a function of $H$. Show that if $H > rK/4$, the population collapses to zero regardless of initial conditions. Simulate this in Python and visualize the bifurcation.

2. **Competing species**: Two species compete for the same resource: $x' = x(3 - x - 2y)$, $y' = y(2 - y - x)$. Find all equilibria and determine their stability using the Jacobian matrix. Draw the phase portrait. Which species wins?

3. **Resonance in RLC**: For an RLC circuit with $L = 1$, $R = 0.1$, $C = 0.04$, find the resonant frequency. Apply a sinusoidal voltage $E(t) = \sin(\omega t)$ and plot the steady-state amplitude as a function of $\omega$. What happens when $\omega$ equals the resonant frequency? How does increasing $R$ affect the peak?

4. **Heat equation with source**: Solve $u_t = 0.01 u_{xx} + \sin(\pi x)$ on $[0,1]$ with $u(0,t) = u(1,t) = 0$ and $u(x,0) = 0$ numerically. The source term represents uniform internal heating. What steady-state temperature profile does the rod approach? Find it analytically by setting $u_t = 0$.

5. **Diffusion time scales**: A pollutant is released at the center of a 1 km lake. If the diffusion coefficient is $D = 10^{-5}$ m$^2$/s, estimate how long it takes for the pollutant to spread to a distance of 100 m using the relation $\sigma \approx \sqrt{2Dt}$. Verify this numerically by simulating the 1D diffusion equation and measuring when the concentration at 100 m reaches 10% of the peak value.

---

*Previous: [Numerical Methods for Differential Equations](./19_Numerical_Methods_for_DE.md) | Next: This is the final lesson of the Calculus and Differential Equations course.*
