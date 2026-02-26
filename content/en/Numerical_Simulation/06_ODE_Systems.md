# ODE Systems

## Learning Objectives

After completing this lesson, you will be able to:

1. Formulate coupled ODE systems and convert higher-order ODEs to first-order systems for numerical integration.
2. Implement and analyze the Lotka-Volterra predator-prey model, interpreting the ecological dynamics from the solution.
3. Simulate nonlinear pendulum motion and compare it to the linearized small-angle approximation.
4. Implement the Lorenz system and explain how sensitive dependence on initial conditions leads to chaotic behavior.
5. Apply phase plane analysis and trajectory visualization to interpret the qualitative behavior of ODE systems.

---

## Overview

Real physical systems are described by coupled ODE systems where multiple variables interact. We learn numerical solutions for coupled ODEs through various examples including ecosystem models, pendulum motion, and chaotic systems.

**Why This Lesson Matters:** Single-variable ODEs can model isolated processes, but the natural world is full of interacting quantities: predator and prey populations, position and velocity of a pendulum, temperature and concentration in a reactor. Coupled ODE systems capture these interactions, and their solutions reveal emergent phenomena -- oscillations, chaos, pattern formation -- that no single equation can produce.

---

## 1. Lotka-Volterra (Predator-Prey)

The Lotka-Volterra model is one of the simplest systems that produces sustained oscillations from pure interaction dynamics. Two species with no intrinsic oscillation create periodic boom-bust cycles purely through their coupling. This makes it a cornerstone example for understanding how nonlinear feedback generates complex dynamics.

### 1.1 Model Description

The Lotka-Volterra equations model predator-prey interaction through two coupled ODEs:

$$\frac{dx}{dt} = \alpha x - \beta x y \qquad \text{(prey growth minus predation)}$$
$$\frac{dy}{dt} = \delta x y - \gamma y \qquad \text{(predator growth minus natural death)}$$

where $x$ is prey population, $y$ is predator population, $\alpha$ is prey birth rate, $\beta$ is predation rate, $\gamma$ is predator death rate, and $\delta$ is predator conversion efficiency. The equilibrium point $(\gamma/\delta, \alpha/\beta)$ is a center, meaning nearby orbits are closed curves -- the system oscillates perpetually without damping or amplification.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
Lotka-Volterra equations:

dx/dt = αx - βxy    (Prey: rabbits)
dy/dt = δxy - γy    (Predator: foxes)

α: Prey growth rate
β: Predation rate
γ: Predator death rate
δ: Predator growth efficiency

Equilibrium points:
- (0, 0): Extinction (saddle point)
- (γ/δ, α/β): Coexistence (center)
"""

def lotka_volterra():
    alpha = 1.0  # Rabbit growth rate
    beta = 0.1   # Predation rate
    gamma = 1.5  # Fox death rate
    delta = 0.075  # Fox growth efficiency

    def lv(t, y):
        x, y_pred = y
        dx = alpha*x - beta*x*y_pred
        dy = delta*x*y_pred - gamma*y_pred
        return [dx, dy]

    # Initial conditions
    y0 = [40, 9]  # 40 rabbits, 9 foxes
    t_span = (0, 50)
    t_eval = np.linspace(0, 50, 1000)

    sol = solve_ivp(lv, t_span, y0, t_eval=t_eval, method='RK45')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Population over time
    axes[0].plot(sol.t, sol.y[0], 'b-', label='Rabbits (prey)')
    axes[0].plot(sol.t, sol.y[1], 'r-', label='Foxes (predator)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Population')
    axes[0].set_title('Lotka-Volterra Population Dynamics')
    axes[0].legend()
    axes[0].grid(True)

    # Phase plane
    axes[1].plot(sol.y[0], sol.y[1], 'g-')
    axes[1].scatter([y0[0]], [y0[1]], color='red', s=100, zorder=5, label='Start')
    axes[1].scatter([gamma/delta], [alpha/beta], color='black', s=100,
                    marker='x', zorder=5, label='Equilibrium')
    axes[1].set_xlabel('Rabbits')
    axes[1].set_ylabel('Foxes')
    axes[1].set_title('Phase Space')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

lotka_volterra()
```

### 1.2 Vector Field and Trajectories

```python
def lv_phase_portrait():
    """Lotka-Volterra vector field with multiple initial conditions"""
    alpha, beta, gamma, delta = 1.0, 0.1, 1.5, 0.075

    def lv(t, y):
        return [alpha*y[0] - beta*y[0]*y[1],
                delta*y[0]*y[1] - gamma*y[1]]

    # Vector field
    x_range = np.linspace(0.1, 80, 20)
    y_range = np.linspace(0.1, 30, 20)
    X, Y = np.meshgrid(x_range, y_range)

    U = alpha*X - beta*X*Y
    V = delta*X*Y - gamma*Y

    # Normalize
    N = np.sqrt(U**2 + V**2)
    U, V = U/N, V/N

    plt.figure(figsize=(12, 8))
    plt.quiver(X, Y, U, V, alpha=0.5, color='gray')

    # Various initial conditions
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for i, (x0, y0) in enumerate([(10, 5), (30, 5), (50, 10), (70, 15), (20, 20)]):
        sol = solve_ivp(lv, (0, 50), [x0, y0], max_step=0.1)
        plt.plot(sol.y[0], sol.y[1], color=colors[i], linewidth=1.5)
        plt.scatter([x0], [y0], color=colors[i], s=50)

    # Equilibrium point
    plt.scatter([gamma/delta], [alpha/beta], color='red', s=150,
                marker='*', zorder=10, label='Equilibrium')

    plt.xlabel('Prey (x)')
    plt.ylabel('Predator (y)')
    plt.title('Lotka-Volterra Phase Portrait')
    plt.legend()
    plt.xlim(0, 80)
    plt.ylim(0, 30)
    plt.grid(True)
    plt.show()

lv_phase_portrait()
```

---

## 2. Pendulum Motion

**Why the pendulum?** The simple pendulum bridges linear and nonlinear dynamics. At small angles, $\sin\theta \approx \theta$ and the motion is perfectly harmonic. As the amplitude grows, the nonlinearity $\sin\theta$ causes the period to depend on amplitude -- a phenomenon impossible in linear systems. At angles near $\pi$, the pendulum approaches its separatrix and the dynamics become qualitatively different (rotation vs. oscillation).

### 2.1 Simple Pendulum

The equation of motion for a simple pendulum of length $L$ in gravitational field $g$ is:

$$\ddot{\theta} + \frac{g}{L}\sin\theta = 0$$

Converting to a first-order system with $\omega = \dot{\theta}$: the state $(\theta, \omega)$ traces closed orbits in phase space for oscillatory motion, but open curves for rotational motion (when the pendulum swings over the top).

```python
def simple_pendulum():
    """
    Simple pendulum: θ'' + (g/L)sin(θ) = 0

    Small angle approximation: θ'' + (g/L)θ = 0
    Solution: θ = θ₀cos(ωt), ω = √(g/L)
    """
    g = 9.8  # Gravitational acceleration
    L = 1.0  # Pendulum length

    def pendulum(t, y):
        theta, omega = y
        return [omega, -(g/L) * np.sin(theta)]

    # Various initial angles
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, theta0 in zip(axes.flat, [0.1, 0.5, np.pi/2, np.pi - 0.1]):
        sol = solve_ivp(pendulum, (0, 10), [theta0, 0], max_step=0.01)

        ax.plot(sol.y[0], sol.y[1])
        ax.scatter([theta0], [0], color='red', s=100, label='Start')
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

### 2.2 Damped Driven Pendulum

Adding damping ($\gamma\dot{\theta}$) and periodic driving ($A\cos\omega_d t$) transforms the pendulum into one of the simplest systems that exhibits chaos. The competition between energy injection (driving), energy dissipation (damping), and nonlinear restoring force ($\sin\theta$) creates a rich parameter space where periodic, quasiperiodic, and chaotic motions coexist.

```python
def driven_pendulum():
    """
    Damped driven pendulum:
    θ'' + γθ' + (g/L)sin(θ) = A*cos(ωt)

    Nonlinear → possibility of chaos
    """
    g, L = 9.8, 1.0
    gamma = 0.5  # Damping coefficient
    A = 1.5      # Driving force amplitude
    omega_d = 2/3 * np.sqrt(g/L)  # Driving frequency

    def driven(t, y):
        theta, w = y
        return [w, -gamma*w - (g/L)*np.sin(theta) + A*np.cos(omega_d*t)]

    t_span = (0, 200)
    sol = solve_ivp(driven, t_span, [0.1, 0], max_step=0.01)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time response
    axes[0, 0].plot(sol.t, sol.y[0])
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('θ')
    axes[0, 0].set_title('Time Response')
    axes[0, 0].grid(True)

    # Phase space
    axes[0, 1].plot(sol.y[0], sol.y[1], linewidth=0.5)
    axes[0, 1].set_xlabel('θ')
    axes[0, 1].set_ylabel('ω')
    axes[0, 1].set_title('Phase Space')
    axes[0, 1].grid(True)

    # Poincaré section (sampling at each driving period)
    T_drive = 2 * np.pi / omega_d
    t_poincare = np.arange(0, t_span[1], T_drive)

    from scipy.interpolate import interp1d
    theta_interp = interp1d(sol.t, sol.y[0])
    omega_interp = interp1d(sol.t, sol.y[1])

    valid_t = t_poincare[(t_poincare >= sol.t[0]) & (t_poincare <= sol.t[-1])]
    theta_p = theta_interp(valid_t)
    omega_p = omega_interp(valid_t)

    axes[1, 0].scatter(theta_p[100:], omega_p[100:], s=1)  # Exclude transient
    axes[1, 0].set_xlabel('θ')
    axes[1, 0].set_ylabel('ω')
    axes[1, 0].set_title('Poincaré Section')
    axes[1, 0].grid(True)

    # Energy
    E = 0.5 * sol.y[1]**2 + (g/L) * (1 - np.cos(sol.y[0]))
    axes[1, 1].plot(sol.t, E)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Energy')
    axes[1, 1].set_title('Mechanical Energy')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

driven_pendulum()
```

### 2.3 Double Pendulum

The double pendulum is a 4-dimensional system ($\theta_1, \theta_2, \omega_1, \omega_2$) that exhibits chaos even without external driving. The Lagrangian mechanics yields coupled, highly nonlinear equations of motion. Tiny differences in initial conditions lead to exponentially diverging trajectories -- the hallmark of deterministic chaos.

```python
def double_pendulum():
    """
    Double pendulum: Chaotic system

    θ₁, θ₂: Angles of each pendulum
    ω₁, ω₂: Angular velocities
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

    # Sensitive to initial conditions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    initial_conditions = [
        [np.pi/2, np.pi/2, 0, 0],
        [np.pi/2 + 0.001, np.pi/2, 0, 0],  # Tiny difference
    ]

    for ic, color in zip(initial_conditions, ['blue', 'red']):
        sol = solve_ivp(double_pend, (0, 20), ic, max_step=0.01)

        # Calculate positions
        x1 = L1 * np.sin(sol.y[0])
        y1 = -L1 * np.cos(sol.y[0])
        x2 = x1 + L2 * np.sin(sol.y[1])
        y2 = y1 - L2 * np.cos(sol.y[1])

        axes[0, 0].plot(sol.t, sol.y[0], color=color, alpha=0.7)
        axes[0, 1].plot(sol.t, sol.y[1], color=color, alpha=0.7)
        axes[1, 0].plot(x2, y2, color=color, linewidth=0.5, alpha=0.7)

    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('θ₁')
    axes[0, 0].set_title('First Pendulum')
    axes[0, 0].grid(True)

    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('θ₂')
    axes[0, 1].set_title('Second Pendulum')
    axes[0, 1].grid(True)

    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('End Point Trajectory (IC Sensitivity)')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].grid(True)

    # Initial condition difference visualization
    sol1 = solve_ivp(double_pend, (0, 20), initial_conditions[0], max_step=0.01)
    sol2 = solve_ivp(double_pend, (0, 20), initial_conditions[1], max_step=0.01)

    from scipy.interpolate import interp1d
    t_common = np.linspace(0, 20, 1000)
    theta1_1 = interp1d(sol1.t, sol1.y[0])(t_common)
    theta1_2 = interp1d(sol2.t, sol2.y[0])(t_common)

    axes[1, 1].semilogy(t_common, np.abs(theta1_1 - theta1_2))
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('|Δθ₁|')
    axes[1, 1].set_title('Trajectory Divergence (Exponential Growth)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

double_pendulum()
```

---

## 3. Lorenz System (Chaos)

Edward Lorenz discovered this system in 1963 while studying a simplified model of atmospheric convection. It demonstrated that deterministic systems can produce apparently random behavior, launching the field of chaos theory. The Lorenz attractor became the iconic image of "the butterfly effect."

### 3.1 Basic Simulation

The Lorenz system consists of three coupled ODEs derived from a truncated Fourier expansion of the Rayleigh-Benard convection equations:

$$\dot{x} = \sigma(y - x), \quad \dot{y} = x(\rho - z) - y, \quad \dot{z} = xy - \beta z$$

Here $\sigma$ (Prandtl number) controls the ratio of viscous to thermal diffusion, $\rho$ (Rayleigh number) measures the strength of convective driving, and $\beta$ relates to the aspect ratio of convection cells. At the standard parameters $\sigma=10, \rho=28, \beta=8/3$, the system has a strange attractor with fractal dimension $\approx 2.06$.

```python
def lorenz_system():
    """
    Lorenz system (1963):

    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

    Standard parameters: σ=10, ρ=28, β=8/3
    → Strange attractor
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

    # 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(sol.y[0], sol.y[1], sol.y[2], linewidth=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Lorenz Attractor')

    # x-z projection
    ax2 = fig.add_subplot(132)
    ax2.plot(sol.y[0], sol.y[2], linewidth=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    ax2.set_title('x-z Projection')
    ax2.grid(True)

    # Time response
    ax3 = fig.add_subplot(133)
    ax3.plot(sol.t[:500], sol.y[0][:500])
    ax3.set_xlabel('Time')
    ax3.set_ylabel('x')
    ax3.set_title('x(t)')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

lorenz_system()
```

### 3.2 Initial Condition Sensitivity

```python
def lorenz_sensitivity():
    """Lorenz system initial condition sensitivity"""
    sigma, rho, beta = 10, 28, 8/3

    def lorenz(t, state):
        x, y, z = state
        return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

    # Slightly different initial conditions
    eps = 1e-10
    sol1 = solve_ivp(lorenz, (0, 50), [1, 1, 1], max_step=0.01)
    sol2 = solve_ivp(lorenz, (0, 50), [1+eps, 1, 1], max_step=0.01)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # x over time
    axes[0, 0].plot(sol1.t, sol1.y[0], 'b-', alpha=0.7, label='IC 1')
    axes[0, 0].plot(sol2.t, sol2.y[0], 'r-', alpha=0.7, label='IC 2')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('x')
    axes[0, 0].set_title(f'x(t), initial difference = {eps}')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Trajectory divergence
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
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Distance')
    axes[0, 1].set_title('Distance Between Trajectories (Log Scale)')
    axes[0, 1].grid(True)

    # Lyapunov exponent estimation
    # From initial exponential growth region
    early = (t_common > 5) & (t_common < 20)
    from scipy.stats import linregress
    slope, _, _, _, _ = linregress(t_common[early], np.log(dist[early] + 1e-20))
    print(f"Estimated Lyapunov exponent: {slope:.3f}")

    axes[1, 0].plot(t_common[early], np.log(dist[early]))
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('log(distance)')
    axes[1, 0].set_title(f'Lyapunov Exponent Estimate: λ ≈ {slope:.3f}')
    axes[1, 0].grid(True)

    # 3D comparison
    ax = fig.add_subplot(224, projection='3d')
    ax.plot(sol1.y[0][::10], sol1.y[1][::10], sol1.y[2][::10],
            'b-', alpha=0.5, linewidth=0.5)
    ax.plot(sol2.y[0][::10], sol2.y[1][::10], sol2.y[2][::10],
            'r-', alpha=0.5, linewidth=0.5)
    ax.set_title('Two Trajectories Comparison')

    plt.tight_layout()
    plt.show()

lorenz_sensitivity()
```

### 3.3 Bifurcation Diagram

```python
def lorenz_bifurcation():
    """Lorenz system bifurcation with ρ"""
    sigma, beta = 10, 8/3

    rho_values = np.linspace(0, 50, 100)
    bifurcation_points = []

    for rho in rho_values:
        def lorenz(t, state):
            x, y, z = state
            return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

        sol = solve_ivp(lorenz, (0, 200), [1, 1, 1], max_step=0.01)

        # Collect z extrema after removing transient
        z = sol.y[2][len(sol.t)//2:]

        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(z)
        if len(peaks) > 0:
            z_peaks = z[peaks[-min(50, len(peaks)):]]  # Last 50 peaks
            for zp in z_peaks:
                bifurcation_points.append((rho, zp))

    if bifurcation_points:
        rhos, zs = zip(*bifurcation_points)
        plt.figure(figsize=(12, 6))
        plt.scatter(rhos, zs, s=0.5, c='black')
        plt.xlabel('ρ')
        plt.ylabel('z extrema')
        plt.title('Lorenz System Bifurcation Diagram')
        plt.grid(True)
        plt.show()

# Computation may take long time
# lorenz_bifurcation()
print("Bifurcation diagram computation takes a long time.")
```

---

## 4. Other Systems

### 4.1 SIR Epidemic Model

The SIR model partitions a population into three compartments and describes epidemic dynamics with two parameters:

$$\frac{dS}{dt} = -\beta SI, \quad \frac{dI}{dt} = \beta SI - \gamma I, \quad \frac{dR}{dt} = \gamma I$$

The basic reproduction number $R_0 = \beta/\gamma$ determines whether an epidemic occurs: if $R_0 > 1$, each infected individual infects more than one susceptible, and the epidemic grows. The herd immunity threshold $1 - 1/R_0$ gives the fraction that must be immune to prevent outbreaks.

```python
def sir_model():
    """
    SIR model:
    dS/dt = -βSI
    dI/dt = βSI - γI
    dR/dt = γI

    S: Susceptible
    I: Infected
    R: Recovered
    """
    beta = 0.3   # Infection rate
    gamma = 0.1  # Recovery rate

    def sir(t, y):
        S, I, R = y
        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I
        return [dS, dI, dR]

    # Initial conditions (fractions of population)
    S0, I0, R0 = 0.99, 0.01, 0.0
    t_span = (0, 100)

    sol = solve_ivp(sir, t_span, [S0, I0, R0],
                    t_eval=np.linspace(0, 100, 1000))

    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, sol.y[0], 'b-', label='Susceptible (S)')
    plt.plot(sol.t, sol.y[1], 'r-', label='Infected (I)')
    plt.plot(sol.t, sol.y[2], 'g-', label='Recovered (R)')
    plt.xlabel('Time')
    plt.ylabel('Population Fraction')
    plt.title(f'SIR Model (β={beta}, γ={gamma}, R₀={beta/gamma:.1f})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Basic reproduction number R₀ = β/γ
    R0 = beta / gamma
    print(f"Basic reproduction number R₀ = {R0:.2f}")
    print(f"Critical immunity threshold = 1 - 1/R₀ = {1 - 1/R0:.2%}")

sir_model()
```

### 4.2 Rössler Attractor

```python
def rossler_attractor():
    """
    Rössler attractor:
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
    ax1.set_title('Rössler Attractor')

    # x-y projection
    ax2 = fig.add_subplot(132)
    ax2.plot(sol.y[0], sol.y[1], linewidth=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('x-y Projection')
    ax2.grid(True)

    # Time series
    ax3 = fig.add_subplot(133)
    ax3.plot(sol.t[2000:4000], sol.y[0][2000:4000])
    ax3.set_xlabel('Time')
    ax3.set_ylabel('x')
    ax3.set_title('x(t)')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

rossler_attractor()
```

### 4.3 N-Body Problem (Simplified)

```python
def three_body_simplified():
    """Simplified 3-body problem (planar, equal masses)"""
    G = 1  # Gravitational constant (normalized units)
    m = 1  # All masses equal

    def three_body(t, y):
        # y = [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
        x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = y

        # Distances
        r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
        r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)

        # Accelerations
        ax1 = G*m*((x2-x1)/r12**3 + (x3-x1)/r13**3)
        ay1 = G*m*((y2-y1)/r12**3 + (y3-y1)/r13**3)
        ax2 = G*m*((x1-x2)/r12**3 + (x3-x2)/r23**3)
        ay2 = G*m*((y1-y2)/r12**3 + (y3-y2)/r23**3)
        ax3 = G*m*((x1-x3)/r13**3 + (x2-x3)/r23**3)
        ay3 = G*m*((y1-y3)/r13**3 + (y2-y3)/r23**3)

        return [vx1, vy1, vx2, vy2, vx3, vy3,
                ax1, ay1, ax2, ay2, ax3, ay3]

    # Figure-8 solution initial conditions (Chenciner & Montgomery, 2000)
    # Special periodic solution
    x0 = 0.97000436
    y0 = -0.24308753
    vx0 = 0.4662036850
    vy0 = 0.4323657300

    y0_state = [-x0, -y0, x0, y0, 0, 0,
                vx0, vy0, vx0, vy0, -2*vx0, -2*vy0]

    sol = solve_ivp(three_body, (0, 6.3), y0_state,
                    method='DOP853', max_step=0.01)

    plt.figure(figsize=(10, 8))
    plt.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.8, label='Body 1')
    plt.plot(sol.y[2], sol.y[3], 'r-', linewidth=0.8, label='Body 2')
    plt.plot(sol.y[4], sol.y[5], 'g-', linewidth=0.8, label='Body 3')

    # Initial positions
    plt.scatter([sol.y[0][0], sol.y[2][0], sol.y[4][0]],
                [sol.y[1][0], sol.y[3][0], sol.y[5][0]],
                s=100, c=['blue', 'red', 'green'])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('3-Body Problem: Figure-8 Periodic Solution')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

three_body_simplified()
```

---

## Exercises

### Problem 1
Linearize the Lotka-Volterra system around the equilibrium point and analyze the eigenvalues.

```python
def exercise_1():
    alpha, beta, gamma, delta = 1.0, 0.1, 1.5, 0.075

    # Equilibrium point
    x_eq = gamma / delta
    y_eq = alpha / beta

    print(f"Equilibrium: ({x_eq:.2f}, {y_eq:.2f})")

    # Jacobian
    # J = [[α - βy, -βx], [δy, δx - γ]]
    # At equilibrium:
    J = np.array([[0, -beta * x_eq],
                  [delta * y_eq, 0]])

    eigenvalues = np.linalg.eigvals(J)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Pure imaginary → center (periodic orbits)")

exercise_1()
```

---

## Summary

| System | Characteristics | Key Phenomena |
|--------|----------------|---------------|
| Lotka-Volterra | 2D, conservative | Periodic oscillations |
| Simple pendulum | 2D, nonlinear | Small angle: periodic, large angle: nonlinear |
| Double pendulum | 4D, chaotic | Initial condition sensitivity |
| Lorenz | 3D, chaotic | Strange attractor |
| SIR | 3D, dissipative | Epidemic dynamics |
| Rössler | 3D, chaotic | Band attractor |

| Analysis Tool | Purpose |
|---------------|---------|
| Phase portrait | Trajectory visualization |
| Poincaré section | Distinguish periodicity/chaos |
| Lyapunov exponent | Quantify chaos |
| Bifurcation diagram | Parameter sensitivity |
