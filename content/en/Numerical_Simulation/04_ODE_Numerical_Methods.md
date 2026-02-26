# ODE Numerical Methods

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement the forward Euler method and explain its derivation from the Taylor expansion.
2. Implement the 4th-order Runge-Kutta (RK4) method and explain how it achieves higher accuracy than Euler methods.
3. Compare explicit methods (Euler, RK4) in terms of local truncation error, global error, and computational cost.
4. Apply adaptive step-size control to balance accuracy and efficiency in ODE solvers.
5. Use `scipy.integrate.solve_ivp` with appropriate solver settings for standard ODE initial value problems.

---

## Why Numerical ODE Methods?

Most differential equations arising in science and engineering — from planetary orbits to chemical kinetics — have no closed-form solution. Numerical methods approximate the solution by stepping forward in small increments of time, estimating the slope at each step and using it to predict the next value.

### Forward Euler: Derivation from Taylor Expansion

The simplest approach starts from the Taylor expansion of y(t+h):

```
y(t + h) = y(t) + h·y'(t) + (h²/2)·y''(ξ)     for some ξ ∈ [t, t+h]
```

Dropping the O(h²) remainder gives the **forward Euler formula**:

```
y_{n+1} = y_n + h·f(t_n, y_n)
```

The local truncation error is O(h²) (the dropped term), but errors accumulate over N = T/h steps, so the **global error is O(h)** — halving the step size halves the error.

### RK4: Why the 1/6, 2/6, 2/6, 1/6 Weights?

The 4th-order Runge-Kutta method samples the slope at four points within each step — the beginning (k1), two midpoints (k2, k3), and the end (k4) — then takes a weighted average:

```
y_{n+1} = y_n + (h/6)·(k1 + 2·k2 + 2·k3 + k4)
```

The 1-2-2-1 weighting is the same as **Simpson's rule** for numerical integration: endpoints weighted 1, interior points weighted 2. This achieves O(h⁴) accuracy — meaning a 10× smaller step gives 10,000× smaller error — which is why RK4 is the workhorse of scientific computing.

### Implicit Methods and Stiffness

When a system has both fast and slow dynamics (e.g., a chemical reaction with rapid and gradual components), explicit methods like forward Euler require extremely small time steps to remain stable. **Implicit methods** (like backward Euler) evaluate f at the *next* time step, requiring a nonlinear solve (typically Newton-Raphson) at each step. The extra cost per step is offset by the ability to take much larger steps — this is the core trade-off for **stiff** problems.

---

## Overview

Numerical methods for ordinary differential equations calculate approximate solutions when analytical solutions cannot be found or are difficult to obtain. We will learn the major methods from Euler's method to 4th-order Runge-Kutta.

---

## 1. Euler Method

### 1.1 Forward Euler

```python
import numpy as np
import matplotlib.pyplot as plt

def forward_euler(f, y0, t_span, n_steps):
    """
    Forward Euler method

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

# Test: dy/dt = -y, y(0) = 1
# Analytical solution: y = e^(-t)
f = lambda t, y: -y
y0 = 1.0
t_span = (0, 5)

# Compare with various step numbers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

t_exact = np.linspace(0, 5, 200)
y_exact = np.exp(-t_exact)

for n in [10, 20, 50, 100]:
    t, y = forward_euler(f, y0, t_span, n)
    axes[0].plot(t, y, 'o-', markersize=3, label=f'n={n}')

axes[0].plot(t_exact, y_exact, 'k-', linewidth=2, label='Exact solution')
axes[0].set_xlabel('t')
axes[0].set_ylabel('y')
axes[0].set_title('Forward Euler Method')
axes[0].legend()
axes[0].grid(True)

# Error analysis
errors = []
n_values = [10, 20, 50, 100, 200, 500]
for n in n_values:
    t, y = forward_euler(f, y0, t_span, n)
    error = np.abs(y[-1] - np.exp(-5))
    errors.append(error)

axes[1].loglog(n_values, errors, 'bo-', label='Actual error')
axes[1].loglog(n_values, [5/n for n in n_values], 'r--', label='O(h)')
axes[1].set_xlabel('Number of steps n')
axes[1].set_ylabel('Error')
axes[1].set_title('Error vs Steps (O(h) convergence)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

### 1.2 Backward Euler

```python
def backward_euler(f, df_dy, y0, t_span, n_steps, tol=1e-10):
    """
    Backward Euler method (implicit)

    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})

    Solve implicit equation using Newton-Raphson
    """
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]

    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        # Newton-Raphson: g(y_{n+1}) = y_{n+1} - y_n - h*f(t_{n+1}, y_{n+1}) = 0
        y_guess = y[i]  # Initial guess

        for _ in range(100):  # Maximum iterations
            g = y_guess - y[i] - h * f(t[i+1], y_guess)
            dg = 1 - h * df_dy(t[i+1], y_guess)

            y_new = y_guess - g / dg

            if abs(y_new - y_guess) < tol:
                break
            y_guess = y_new

        y[i+1] = y_new

    return t, y

# Test: dy/dt = -y
f = lambda t, y: -y
df_dy = lambda t, y: -1  # ∂f/∂y

t_fw, y_fw = forward_euler(f, 1.0, (0, 5), 20)
t_bw, y_bw = backward_euler(f, df_dy, 1.0, (0, 5), 20)
t_exact = np.linspace(0, 5, 200)
y_exact = np.exp(-t_exact)

plt.figure(figsize=(10, 5))
plt.plot(t_fw, y_fw, 'bo-', label='Forward Euler')
plt.plot(t_bw, y_bw, 'rs-', label='Backward Euler')
plt.plot(t_exact, y_exact, 'k-', linewidth=2, label='Exact solution')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Forward vs Backward Euler')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 2. Runge-Kutta Methods

### 2.1 RK2 (Midpoint, Heun)

```python
def rk2_midpoint(f, y0, t_span, n_steps):
    """RK2 midpoint method"""
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
    """RK2 Heun method (modified Euler)"""
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]

    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h, y[i] + h * k1)
        y[i+1] = y[i] + h/2 * (k1 + k2)

    return t, y

# Comparison
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
plt.plot(t_mid, y_mid, 's-', label='RK2 Midpoint', alpha=0.7)
plt.plot(t_heun, y_heun, '^-', label='RK2 Heun', alpha=0.7)
plt.plot(t_exact, y_exact, 'k-', linewidth=2, label='Exact solution')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Euler vs RK2 Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

### 2.2 RK4 (Classical 4th Order)

```python
def rk4(f, y0, t_span, n_steps):
    """
    Classical 4th-order Runge-Kutta method

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

# Accuracy comparison
f = lambda t, y: -y
t_span = (0, 5)

methods = [
    ('Euler', forward_euler),
    ('RK2', rk2_heun),
    ('RK4', rk4)
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Solution comparison
n = 10
for name, method in methods:
    t, y = method(f, 1.0, t_span, n)
    axes[0].plot(t, y, 'o-', label=name)

t_exact = np.linspace(0, 5, 200)
axes[0].plot(t_exact, np.exp(-t_exact), 'k-', linewidth=2, label='Exact solution')
axes[0].set_xlabel('t')
axes[0].set_ylabel('y')
axes[0].set_title(f'n={n} steps')
axes[0].legend()
axes[0].grid(True)

# Convergence order
n_values = [10, 20, 50, 100, 200]
for name, method in methods:
    errors = []
    for n in n_values:
        t, y = method(f, 1.0, t_span, n)
        error = np.abs(y[-1] - np.exp(-5))
        errors.append(error)
    axes[1].loglog(n_values, errors, 'o-', label=name)

# Theoretical convergence orders
axes[1].loglog(n_values, [1/n for n in n_values], 'k--', alpha=0.5, label='O(h)')
axes[1].loglog(n_values, [1/n**2 for n in n_values], 'k:', alpha=0.5, label='O(h²)')
axes[1].loglog(n_values, [1/n**4 for n in n_values], 'k-.', alpha=0.5, label='O(h⁴)')

axes[1].set_xlabel('Number of steps n')
axes[1].set_ylabel('Error')
axes[1].set_title('Convergence Order Comparison')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

---

## 3. Application to Systems

### 3.1 Vectorized RK4

```python
def rk4_system(f, y0, t_span, n_steps):
    """
    RK4 for systems of ODEs

    y: vector
    f: vector function
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

# Simple harmonic oscillator: x'' + ω²x = 0
# System: y₁' = y₂, y₂' = -ω²y₁
omega = 2.0

def harmonic_oscillator(t, y):
    return [y[1], -omega**2 * y[0]]

t_span = (0, 10)
y0 = [1.0, 0.0]  # x(0) = 1, x'(0) = 0

t, y = rk4_system(harmonic_oscillator, y0, t_span, 200)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time response
axes[0].plot(t, y[:, 0], label='x(t)')
axes[0].plot(t, y[:, 1], label="x'(t)")
axes[0].plot(t, np.cos(omega * t), 'k--', alpha=0.5, label='Exact solution')
axes[0].set_xlabel('t')
axes[0].set_ylabel('Value')
axes[0].set_title('Harmonic Oscillator')
axes[0].legend()
axes[0].grid(True)

# Phase plane
axes[1].plot(y[:, 0], y[:, 1])
axes[1].set_xlabel('x')
axes[1].set_ylabel("x'")
axes[1].set_title('Phase Space')
axes[1].set_aspect('equal')
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

### 3.2 Damped Oscillator

```python
def damped_oscillator_example():
    """Damped oscillator: x'' + 2γx' + ω₀²x = 0"""
    omega0 = 2.0
    gamma = 0.3  # Damping coefficient

    def damped_osc(t, y):
        return [y[1], -2*gamma*y[1] - omega0**2 * y[0]]

    t_span = (0, 15)
    y0 = [1.0, 0.0]

    t, y = rk4_system(damped_osc, y0, t_span, 300)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time response
    omega_d = np.sqrt(omega0**2 - gamma**2)  # Damped frequency
    envelope = np.exp(-gamma * t)

    axes[0].plot(t, y[:, 0], label='x(t)')
    axes[0].plot(t, envelope, 'r--', label='Damping envelope')
    axes[0].plot(t, -envelope, 'r--')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x')
    axes[0].set_title(f'Damped Oscillator (γ={gamma}, ω₀={omega0})')
    axes[0].legend()
    axes[0].grid(True)

    # Phase space (spiral)
    axes[1].plot(y[:, 0], y[:, 1])
    axes[1].scatter([0], [0], color='red', s=100, zorder=5, label='Stable equilibrium')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel("x'")
    axes[1].set_title('Phase Space')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

damped_oscillator_example()
```

---

## 4. Adaptive Step Size

### 4.1 Error Estimation

```python
def rk45_step(f, t, y, h):
    """
    RK4-5 (Dormand-Prince) one step

    Calculate 4th and 5th order approximations simultaneously for error estimation
    """
    # Dormand-Prince coefficients
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

    # Calculate k
    k = [np.array(f(t, y))]
    for i in range(1, 7):
        yi = y.copy()
        for j in range(i):
            yi = yi + h * a[i][j] * k[j]
        k.append(np.array(f(t + c[i]*h, yi)))

    # 5th order approximation
    y5 = y.copy()
    for i in range(7):
        y5 = y5 + h * b5[i] * k[i]

    # 4th order approximation (for error estimation)
    y4 = y.copy()
    for i in range(7):
        y4 = y4 + h * b4[i] * k[i]

    # Error estimation
    error = np.max(np.abs(y5 - y4))

    return y5, error

def adaptive_rk45(f, y0, t_span, tol=1e-6, h_init=0.1):
    """Adaptive RK4-5 solver"""
    t = [t_span[0]]
    y = [np.array(y0)]
    h = h_init

    while t[-1] < t_span[1]:
        # Ensure step size doesn't exceed range
        if t[-1] + h > t_span[1]:
            h = t_span[1] - t[-1]

        # RK4-5 step
        y_new, error = rk45_step(f, t[-1], y[-1], h)

        # Adjust step based on error
        if error < tol:
            t.append(t[-1] + h)
            y.append(y_new)

            # Increase step size (safety factor 0.9)
            if error > 0:
                h = min(h * 0.9 * (tol / error) ** 0.2, 2 * h)
        else:
            # Decrease step size
            h = max(h * 0.9 * (tol / error) ** 0.25, h / 4)

    return np.array(t), np.array(y)

# Test
def stiff_like(t, y):
    """Function with rapid changes"""
    return [-10 * y[0] if t < 1 else -0.1 * y[0]]

t_adapt, y_adapt = adaptive_rk45(stiff_like, [1.0], (0, 5), tol=1e-6)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(t_adapt, y_adapt[:, 0], 'o-', markersize=3)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Adaptive RK4-5')
plt.grid(True)

plt.subplot(1, 2, 2)
dt = np.diff(t_adapt)
plt.plot(t_adapt[:-1], dt, 'o-')
plt.xlabel('t')
plt.ylabel('Step size h')
plt.title('Step Size Variation')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 5. Solving ODEs with SciPy

### 5.1 solve_ivp

```python
from scipy.integrate import solve_ivp

# Lorenz system
def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

t_span = (0, 50)
y0 = [1.0, 1.0, 1.0]

# Compare various solvers
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

### 5.2 Event Detection

```python
def projectile_motion():
    """Projectile motion - ground impact detection"""

    g = 9.8

    def projectile(t, state):
        x, y, vx, vy = state
        return [vx, vy, 0, -g]

    def hit_ground(t, state):
        return state[1]  # Event when y = 0

    hit_ground.terminal = True  # Stop when event occurs
    hit_ground.direction = -1   # Only when decreasing (falling)

    # Initial condition: 45 degree angle, velocity 20 m/s
    v0 = 20
    angle = 45 * np.pi / 180
    y0 = [0, 0, v0 * np.cos(angle), v0 * np.sin(angle)]

    sol = solve_ivp(projectile, (0, 10), y0, events=hit_ground,
                    dense_output=True, max_step=0.01)

    print(f"Flight time: {sol.t_events[0][0]:.4f} s")
    print(f"Range: {sol.y_events[0][0][0]:.4f} m")

    plt.figure(figsize=(10, 5))
    plt.plot(sol.y[0], sol.y[1])
    plt.scatter(sol.y_events[0][0][0], 0, color='red', s=100, label='Landing point')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Projectile Motion')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

projectile_motion()
```

---

## 6. Stability Analysis

### 6.1 Stability Regions

```python
def stability_regions():
    """Stability regions of numerical methods"""

    # Test equation: dy/dt = λy
    # Exact solution: y(nh) = e^(λnh)
    # Numerical solution: y_n = R(λh)^n * y_0

    # Region where |R(z)| ≤ 1 in complex plane

    x = np.linspace(-4, 2, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y  # z = λh

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Forward Euler: R(z) = 1 + z
    R_euler = np.abs(1 + Z)
    axes[0, 0].contour(X, Y, R_euler, levels=[1], colors='blue')
    axes[0, 0].contourf(X, Y, R_euler, levels=[0, 1], alpha=0.3)
    axes[0, 0].set_title('Forward Euler: |1 + z| ≤ 1')
    axes[0, 0].set_xlabel('Re(λh)')
    axes[0, 0].set_ylabel('Im(λh)')
    axes[0, 0].grid(True)
    axes[0, 0].set_aspect('equal')

    # Backward Euler: R(z) = 1/(1-z)
    R_backward = np.abs(1 / (1 - Z))
    axes[0, 1].contour(X, Y, R_backward, levels=[1], colors='blue')
    axes[0, 1].contourf(X, Y, R_backward, levels=[0, 1], alpha=0.3)
    axes[0, 1].set_title('Backward Euler: |1/(1-z)| ≤ 1')
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

## Practice Problems

### Problem 1
Solve the Van der Pol oscillator using RK4:
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
    axes[0].set_title('Van der Pol Oscillator')

    axes[1].plot(sol.y[0], sol.y[1])
    axes[1].set_xlabel('x')
    axes[1].set_ylabel("x'")
    axes[1].set_title('Phase Space (Limit Cycle)')

    plt.tight_layout()
    plt.show()

exercise_1()
```

---

## Summary

| Method | Order | Characteristics |
|------|------|------|
| Forward Euler | O(h) | Simple, limited stability |
| Backward Euler | O(h) | Implicit, A-stable |
| RK2 | O(h²) | Midpoint, Heun |
| RK4 | O(h⁴) | Most widely used |
| RK4-5 | O(h⁵) | Adaptive step |

| SciPy Solver | Type | Purpose |
|-----------|------|------|
| RK45 | Explicit | General problems (default) |
| DOP853 | Explicit | High precision |
| Radau | Implicit | Stiff problems |
| BDF | Implicit | Stiff problems |

## Exercises

### Exercise 1: Euler Method Step Size and Global Error

Solve `dy/dt = -2y + 2`, `y(0) = 0` (exact solution: `y(t) = 1 - e^(-2t)`) using forward Euler with step sizes `h = 0.5, 0.25, 0.1, 0.05` over `t ∈ [0, 3]`. For each h compute the global error at `t = 3`, verify that doubling h doubles the error (O(h) convergence), and determine what h is needed to achieve a global error below 10⁻³.

<details>
<summary>Show Answer</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def forward_euler(f, y0, t_span, n_steps):
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]
    y = np.zeros(n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        y[i+1] = y[i] + h * f(t[i], y[i])
    return t, y

f = lambda t, y: -2*y + 2
y_exact = lambda t: 1 - np.exp(-2*t)
t_end = 3.0

print(f"{'h':>8}  {'Global error at t=3':>22}  {'Ratio':>8}")
print("-" * 46)
prev_err = None
for h in [0.5, 0.25, 0.1, 0.05, 0.01]:
    n = int(t_end / h)
    t, y = forward_euler(f, 0.0, (0, t_end), n)
    err = abs(y[-1] - y_exact(t_end))
    ratio = prev_err / err if prev_err else float('nan')
    print(f"{h:>8.3f}  {err:>22.6e}  {ratio:>8.2f}")
    prev_err = err

# Required h for error < 1e-3
# For forward Euler: global error ~ C*h
# Estimate C from known errors
t_tmp, y_tmp = forward_euler(f, 0.0, (0, t_end), 100)
C = abs(y_tmp[-1] - y_exact(t_end)) / 0.03   # approx C
h_needed = 1e-3 / C
print(f"\nEstimated h needed for error < 1e-3: h ≈ {h_needed:.4f}")
print("(try h=0.005 to verify)")
```

The ratio column should be close to **2.0** each time h halves, confirming O(h) convergence. For `h = 0.01` the error is already below `10⁻³`.

</details>

### Exercise 2: RK4 vs. RK2 Convergence Order Verification

For the ODE `dy/dt = cos(t)·y`, `y(0) = 1` (exact solution: `y(t) = e^(sin(t))`), numerically verify the convergence orders of RK2 (Heun) and RK4 by computing the error at `t = 2π` for n = 10, 20, 40, 80, 160 steps. Estimate the convergence order from successive log-ratios and confirm they match O(h²) and O(h⁴) respectively.

<details>
<summary>Show Answer</summary>

```python
import numpy as np

def rk2_heun(f, y0, t_span, n):
    t = np.linspace(t_span[0], t_span[1], n + 1)
    h = t[1] - t[0]
    y = np.zeros(n + 1); y[0] = y0
    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h, y[i] + h * k1)
        y[i+1] = y[i] + h/2 * (k1 + k2)
    return t, y

def rk4(f, y0, t_span, n):
    t = np.linspace(t_span[0], t_span[1], n + 1)
    h = t[1] - t[0]
    y = np.zeros(n + 1); y[0] = y0
    for i in range(n):
        k1 = f(t[i],       y[i])
        k2 = f(t[i] + h/2, y[i] + h/2*k1)
        k3 = f(t[i] + h/2, y[i] + h/2*k2)
        k4 = f(t[i] + h,   y[i] + h*k3)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, y

f = lambda t, y: np.cos(t) * y
y_exact_end = np.exp(np.sin(2 * np.pi))   # y(2π) = e^0 = 1
t_span = (0, 2 * np.pi)
n_values = [10, 20, 40, 80, 160]

print(f"{'n':>5}  {'RK2 error':>12}  {'RK2 order':>10}  {'RK4 error':>12}  {'RK4 order':>10}")
print("-" * 60)
prev2 = prev4 = None
for n in n_values:
    _, y2 = rk2_heun(f, 1.0, t_span, n)
    _, y4 = rk4(f, 1.0, t_span, n)
    e2 = abs(y2[-1] - y_exact_end)
    e4 = abs(y4[-1] - y_exact_end)
    ord2 = np.log2(prev2/e2) if prev2 else float('nan')
    ord4 = np.log2(prev4/e4) if prev4 else float('nan')
    print(f"{n:>5}  {e2:>12.3e}  {ord2:>10.2f}  {e4:>12.3e}  {ord4:>10.2f}")
    prev2, prev4 = e2, e4
```

The RK2 order column converges to **~2.0** and the RK4 order column converges to **~4.0**, confirming the theoretical convergence orders. At n = 160, RK4 is already near machine precision while RK2 still has significant error.

</details>

### Exercise 3: Stiffness and Solver Choice

The stiff ODE `dy/dt = -100(y - cos(t)) - sin(t)`, `y(0) = 0` has exact solution `y(t) = cos(t)`. Solve it using (a) `RK45` and (b) `Radau` from `scipy.integrate.solve_ivp` with `rtol=1e-6`. Compare the number of function evaluations (`sol.nfev`) and explain why RK45 requires far more evaluations than Radau on this problem.

<details>
<summary>Show Answer</summary>

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def stiff_ode(t, y):
    return [-100 * (y[0] - np.cos(t)) - np.sin(t)]

t_span = (0, 2 * np.pi)
y0 = [0.0]
t_eval = np.linspace(0, 2*np.pi, 200)
y_exact = np.cos(np.linspace(0, 2*np.pi, 200))

results = {}
for method in ['RK45', 'Radau', 'BDF']:
    sol = solve_ivp(stiff_ode, t_span, y0, method=method,
                    t_eval=t_eval, rtol=1e-6, atol=1e-8)
    results[method] = sol
    err = np.max(np.abs(sol.y[0] - y_exact))
    print(f"{method:>6}: nfev={sol.nfev:>6}  nsteps={len(sol.t):>5}  "
          f"max_error={err:.2e}  success={sol.success}")

# Plot
plt.figure(figsize=(9, 4))
for method, sol in results.items():
    plt.plot(sol.t, sol.y[0], label=f'{method} (nfev={sol.nfev})', alpha=0.8)
plt.plot(t_eval, y_exact, 'k--', linewidth=2, label='Exact cos(t)')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Stiff ODE: y\' = -100(y - cos t) - sin t')
plt.legend()
plt.grid(True)
plt.show()
```

**Explanation:** RK45 is an explicit method with stability region limited to `Re(λh) > -3.5`. For this problem `λ = -100`, so stability requires `h < 0.035` regardless of accuracy needs. Radau is an implicit L-stable method that remains stable for any step size, so it can take large steps dictated by accuracy rather than stability.

</details>

### Exercise 4: Energy Conservation in Hamiltonian Systems

Solve the simple harmonic oscillator `x'' + ω²x = 0` (ω = 1) with `x(0) = 1, x'(0) = 0` using forward Euler and RK4 with h = 0.1 over 50 periods. Compute the total mechanical energy `E(t) = ½(x')² + ½ω²x²` at each time step and plot it alongside the exact constant energy E = 0.5. Compare how well each method conserves energy.

<details>
<summary>Show Answer</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

omega = 1.0
h = 0.1
T = 2 * np.pi / omega          # one period
n_steps = int(50 * T / h)      # 50 periods

def harmonic(t, y):
    return [y[1], -omega**2 * y[0]]

y0 = [1.0, 0.0]   # x=1, v=0

# Forward Euler
t_eu = [0.0]; y_eu = [np.array(y0)]
for _ in range(n_steps):
    yi = y_eu[-1]; ti = t_eu[-1]
    dy = np.array(harmonic(ti, yi))
    y_eu.append(yi + h * dy)
    t_eu.append(ti + h)
t_eu = np.array(t_eu)
y_eu = np.array(y_eu)

# RK4
def rk4_step(f, t, y, h):
    k1 = np.array(f(t,       y))
    k2 = np.array(f(t + h/2, y + h/2*k1))
    k3 = np.array(f(t + h/2, y + h/2*k2))
    k4 = np.array(f(t + h,   y + h*k3))
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)

t_rk = [0.0]; y_rk = [np.array(y0)]
for _ in range(n_steps):
    y_rk.append(rk4_step(harmonic, t_rk[-1], y_rk[-1], h))
    t_rk.append(t_rk[-1] + h)
t_rk = np.array(t_rk)
y_rk = np.array(y_rk)

# Energy: E = 0.5*v² + 0.5*ω²*x²
E_exact = 0.5   # ½·1² + ½·0² = 0.5
E_eu = 0.5 * y_eu[:, 1]**2 + 0.5 * omega**2 * y_eu[:, 0]**2
E_rk = 0.5 * y_rk[:, 1]**2 + 0.5 * omega**2 * y_rk[:, 0]**2

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(t_eu, E_eu, 'b-', linewidth=0.8, label='Forward Euler')
axes[0].plot(t_rk, E_rk, 'r-', linewidth=0.8, label='RK4')
axes[0].axhline(E_exact, color='k', linestyle='--', label=f'Exact E={E_exact}')
axes[0].set_xlabel('t')
axes[0].set_ylabel('Energy')
axes[0].set_title('Energy vs Time (50 periods)')
axes[0].legend()
axes[0].grid(True)

axes[1].semilogy(t_eu, np.abs(E_eu - E_exact), 'b-', linewidth=0.8, label='Euler |ΔE|')
axes[1].semilogy(t_rk, np.abs(E_rk - E_exact) + 1e-17, 'r-', linewidth=0.8, label='RK4 |ΔE|')
axes[1].set_xlabel('t')
axes[1].set_ylabel('|Energy error|')
axes[1].set_title('Energy Error (log scale)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

print(f"Euler: energy grows ~ linearly (method not symplectic)")
print(f"RK4:   energy error ~ {np.max(np.abs(E_rk - E_exact)):.2e} (small but non-zero)")
```

**Key observation:** Forward Euler's energy grows monotonically (the method adds artificial energy). RK4 conserves energy much better but still has a small drift over 50 periods. For long-time Hamiltonian simulations, symplectic integrators (e.g., leapfrog/Störmer-Verlet) are preferred because they conserve a modified energy exactly.

</details>
