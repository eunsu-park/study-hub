# Numerical Methods for Differential Equations

## Learning Objectives

- Derive Euler's method from Taylor expansion and analyze its local and global truncation errors
- Implement and compare Runge-Kutta methods (RK2, RK4) for improved accuracy
- Apply adaptive step size control to balance accuracy and computational cost
- Identify stiff equations and explain why explicit methods fail on them
- Solve systems of ODE numerically and apply the methods to real-world dynamical systems (Lorenz attractor, pendulum)

## Prerequisites

Before studying this lesson, you should be comfortable with:
- First-order and second-order ODE (Lessons 8-12)
- Taylor series expansion (Lessons 3-4)
- Basic Python programming with NumPy

## Why Numerical Methods?

Most differential equations cannot be solved analytically. In practice:

- **Nonlinear equations** like the three-body problem, turbulence models, or chemical reaction networks have no closed-form solutions
- **Complex systems** with many coupled equations (weather models, neural networks, ecological models) require computation
- **Real engineering** demands numbers, not formulas -- we need $y(3.7) = 2.451$, not $y = C_1 e^t + C_2 te^t$

Numerical methods approximate the solution at discrete time steps. The fundamental trade-off is **accuracy vs cost**: higher-order methods are more accurate per step but require more function evaluations.

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

## Euler's Method

### Forward Euler: Derivation

Given $y' = f(t, y)$, $y(t_0) = y_0$, we want to approximate $y(t)$ at discrete points.

Taylor expand $y(t + h)$ around $t$:

$$y(t + h) = y(t) + h \cdot y'(t) + \frac{h^2}{2} y''(t) + \cdots$$

Drop all terms beyond the first derivative:

$$y(t + h) \approx y(t) + h \cdot f(t, y(t))$$

This gives the **forward (explicit) Euler method**:

$$\boxed{y_{n+1} = y_n + h \cdot f(t_n, y_n)}$$

where $h$ is the step size and $y_n \approx y(t_n)$.

**Geometric interpretation**: At each point, we follow the tangent line (slope field direction) for a step of size $h$. The method works by "sliding along tangent lines" -- accurate for small steps but accumulates error over many steps.

### Error Analysis

- **Local truncation error** (error per step): $O(h^2)$ -- we dropped the $h^2$ term from Taylor
- **Global error** (accumulated error over $[t_0, T]$): $O(h)$ -- errors from all $N = (T - t_0)/h$ steps accumulate

Euler's method is a **first-order** method. To halve the error, you must halve the step size, doubling the computation.

### Backward Euler: Implicit Method

$$y_{n+1} = y_n + h \cdot f(t_{n+1}, y_{n+1})$$

The unknown $y_{n+1}$ appears on both sides -- we must solve an equation at each step (typically using Newton's method). This extra cost buys **unconditional stability**, making it essential for stiff equations (discussed below).

## Runge-Kutta Methods

### The Idea

Euler's method samples the slope at one point (the beginning of the interval). What if we sample at multiple points within $[t_n, t_{n+1}]$ and take a weighted average? This is the Runge-Kutta approach.

Think of it like estimating the average height of a hill. Euler measures at one end; Runge-Kutta samples at several points along the way for a better estimate.

### RK2: The Midpoint Method

**Strategy**: Use Euler to estimate the midpoint, then use the slope at the midpoint for the full step.

$$k_1 = f(t_n, y_n)$$
$$k_2 = f\left(t_n + \frac{h}{2}, \; y_n + \frac{h}{2} k_1\right)$$
$$y_{n+1} = y_n + h \cdot k_2$$

- **Local error**: $O(h^3)$, **Global error**: $O(h^2)$
- Cost: 2 function evaluations per step (vs 1 for Euler)
- Accuracy gain: much better than Euler for the same step size

### RK4: The Classic Fourth-Order Method

The workhorse of numerical ODE solving. It evaluates the slope at four points:

$$k_1 = f(t_n, y_n)$$
$$k_2 = f\left(t_n + \frac{h}{2}, \; y_n + \frac{h}{2}k_1\right)$$
$$k_3 = f\left(t_n + \frac{h}{2}, \; y_n + \frac{h}{2}k_2\right)$$
$$k_4 = f(t_n + h, \; y_n + h \cdot k_3)$$

$$\boxed{y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)}$$

where:
- $k_1$: slope at the beginning
- $k_2$: slope at the midpoint, using $k_1$ to estimate $y$ there
- $k_3$: slope at the midpoint again, using the improved $k_2$ estimate
- $k_4$: slope at the end, using $k_3$ to estimate $y$ there
- Final formula: weighted average with Simpson's rule weights $(1:2:2:1)/6$

**Error**: Local $O(h^5)$, Global $O(h^4)$. This means halving $h$ reduces the error by a factor of 16. RK4 costs 4 function evaluations per step but is extremely accurate for smooth problems.

### Comparison of Methods

| Method | Order | Evaluations/step | Local Error | Global Error |
|--------|-------|------------------|-------------|-------------|
| Forward Euler | 1 | 1 | $O(h^2)$ | $O(h)$ |
| RK2 (Midpoint) | 2 | 2 | $O(h^3)$ | $O(h^2)$ |
| RK4 (Classic) | 4 | 4 | $O(h^5)$ | $O(h^4)$ |

## Adaptive Step Size Control

### Why Adapt?

A fixed step size is wasteful: the solution may change slowly in some regions (allowing large steps) and rapidly in others (requiring small steps). Adaptive methods automatically adjust $h$ to maintain a target accuracy.

### Embedded Runge-Kutta Methods

The most common approach computes two estimates of different order using (mostly) the same function evaluations:

1. Compute both an order-$p$ and an order-$(p+1)$ solution
2. Estimate the local error: $\text{err} \approx |y_{p+1} - y_p|$
3. If $\text{err} < \text{tol}$: accept the step, try a larger $h$
4. If $\text{err} > \text{tol}$: reject the step, reduce $h$, try again

The **Dormand-Prince** method (used by `scipy.integrate.solve_ivp` with `method='RK45'`) is an embedded RK4(5) pair that is the modern standard for non-stiff problems.

The new step size is estimated as:

$$h_{\text{new}} = h \cdot \left(\frac{\text{tol}}{\text{err}}\right)^{1/(p+1)}$$

with a safety factor (typically 0.9) to avoid oscillating between accepting and rejecting.

## Stiff Equations

### What Is Stiffness?

A **stiff** equation has components evolving on vastly different time scales. The classic example:

$$y' = -1000y + 1000, \quad y(0) = 0$$

The exact solution is $y(t) = 1 - e^{-1000t}$. The exponential decays almost instantly (time scale $\sim 0.001$), but the steady state persists forever. An explicit method must use $h \ll 0.002$ for stability, even though the solution is essentially constant for $t > 0.01$.

**Analogy**: Imagine a ball rolling into a steep valley and then along a flat floor. An explicit method must take tiny steps during the steep descent (for stability), and those same tiny steps are forced on the long, boring flat section. An implicit method can take large steps because it "sees" the steady state directly.

### Why Explicit Methods Fail

For $y' = \lambda y$ with $\lambda < 0$, forward Euler gives $y_{n+1} = (1 + h\lambda) y_n$. For stability, we need $|1 + h\lambda| < 1$, which requires $h < 2/|\lambda|$. When $|\lambda| = 1000$, this forces $h < 0.002$ regardless of accuracy needs.

### Implicit Methods for Stiff Problems

The backward Euler method applied to $y' = \lambda y$ gives $y_{n+1} = y_n / (1 - h\lambda)$, which is stable for all $h > 0$ when $\lambda < 0$. This is **A-stability** or **unconditional stability**.

In practice, `scipy.integrate.solve_ivp` with `method='Radau'` or `method='BDF'` uses high-order implicit methods optimized for stiff problems.

## Systems of ODE

Any method for scalar ODE extends naturally to systems. Write:

$$\mathbf{y}' = \mathbf{f}(t, \mathbf{y}), \quad \mathbf{y}(t_0) = \mathbf{y}_0$$

where $\mathbf{y} = [y_1, y_2, \ldots, y_m]^T$. The RK4 formulas apply with vector $\mathbf{k}_i$ values.

Higher-order ODE are converted to systems. For example, $y'' + y = 0$ becomes:

$$\begin{cases} y_1' = y_2 \\ y_2' = -y_1 \end{cases} \quad \text{with } y_1 = y, \; y_2 = y'$$

## Brief Introduction to PDE Numerical Methods

While this lesson focuses on ODE, many PDE are solved by discretizing space to get a large system of ODE, then applying the methods above. The **method of lines** approach:

1. Discretize spatial derivatives using finite differences (central difference: $u_{xx} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$)
2. This converts the PDE into a system of ODE in time
3. Apply Euler, RK4, or an implicit method to advance in time

For comprehensive coverage of numerical PDE methods, see [Numerical Simulation - Finite Difference PDE](../Numerical_Simulation/06_Finite_Difference_PDE.md).

## Python Implementation

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

## Summary

| Method | Order | Best For | Limitation |
|--------|-------|----------|------------|
| Forward Euler | 1 | Teaching, simple problems | Large errors, stability issues |
| Backward Euler | 1 | Stiff problems (basic) | Requires nonlinear solve each step |
| RK2 (Midpoint) | 2 | Moderate accuracy needs | Not as efficient as RK4 |
| RK4 (Classic) | 4 | General non-stiff problems | Fails on stiff equations |
| RK45 (Dormand-Prince) | 4-5 | Adaptive non-stiff | Not for stiff problems |
| Radau / BDF | High | Stiff problems | Higher per-step cost |

**Practical advice**: For most problems, start with `scipy.integrate.solve_ivp` using `method='RK45'`. If it requires extremely small steps or produces unexpected results, suspect stiffness and switch to `method='Radau'`.

For comprehensive coverage of numerical ODE and PDE methods including higher-order schemes, see [Numerical Simulation](../Numerical_Simulation/03_ODE_Solvers.md) and subsequent lessons.

## Practice Problems

1. **Euler vs RK4**: Solve $y' = y \sin(t)$, $y(0) = 1$ on $[0, 10]$ using both Euler and RK4 with $h = 0.1$. The exact solution is $y = e^{1-\cos t}$. Plot the error for both methods. At what step size does Euler achieve the same accuracy at $t = 10$ as RK4 with $h = 0.1$?

2. **Convergence rates**: For the problem $y' = -y + \sin(t)$, $y(0) = 0$, compute the global error at $t = 5$ for step sizes $h = 0.5, 0.25, 0.1, 0.05, 0.01$ using Euler, RK2, and RK4. Plot error vs $h$ on a log-log scale and verify the theoretical convergence rates by measuring the slopes.

3. **Lorenz sensitivity**: Solve the Lorenz system with two initial conditions that differ by $10^{-10}$ in the $x$-component. Plot both trajectories and the distance between them as a function of time. Estimate the Lyapunov exponent from the exponential growth rate of the distance.

4. **Stiffness detection**: The Van der Pol oscillator $y'' - \mu(1 - y^2)y' + y = 0$ becomes stiff for large $\mu$. Convert to a system and solve with `RK45` and `Radau` for $\mu = 1, 10, 100, 1000$. Compare the number of function evaluations. At what value of $\mu$ does `RK45` become impractically slow?

5. **Adaptive step size**: Implement a simple adaptive Euler method that doubles $h$ when the local error is below $\text{tol}/10$ and halves $h$ when above $\text{tol}$. Test on $y' = -50(y - \cos t)$, $y(0) = 0$ with $\text{tol} = 10^{-4}$. Plot the step size as a function of time and explain the pattern.

---

*Previous: [Fourier Series and PDE](./18_Fourier_Series_and_PDE.md) | Next: [Applications and Modeling](./20_Applications_and_Modeling.md)*
