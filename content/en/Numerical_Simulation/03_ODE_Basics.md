# ODE Basics

## Learning Objectives

After completing this lesson, you will be able to:

1. Define ordinary differential equations (ODE) and classify them by order, linearity, and homogeneity.
2. Solve first-order ODEs analytically (separation of variables, integrating factor) for simple physical models.
3. Formulate initial value problems (IVP) and boundary value problems (BVP) from physical system descriptions.
4. Explain the concept of solution existence and uniqueness and identify conditions under which they hold.
5. Model and analyze physical phenomena (e.g., exponential growth, harmonic oscillator) using ODEs.

---

## Overview

Ordinary Differential Equations (ODE) are equations that contain derivatives with respect to a single independent variable. They are widely used to describe temporal changes in physical systems.

**Why This Lesson Matters:** ODEs are the simplest differential equations, yet they describe an astonishing range of phenomena: radioactive decay, population dynamics, electrical circuits, mechanical oscillators, chemical reactions, and orbital mechanics. Mastering ODEs -- both their analytical structure and numerical solution -- is the essential first step before tackling the partial differential equations that dominate the rest of this course.

---

## 1. Basic Concepts of ODE

### 1.1 Definition and Classification

```
General ODE:
F(t, y, y', y'', ..., y⁽ⁿ⁾) = 0

First-order ODE:
dy/dt = f(t, y)

nth-order ODE:
d^n y/dt^n = f(t, y, y', ..., y^(n-1))
```

### 1.2 Classification

```python
"""
1. Order: The order of the highest derivative
   - 1st order: y' = f(t, y)
   - 2nd order: y'' = f(t, y, y')

2. Linearity:
   - Linear: y'' + p(t)y' + q(t)y = g(t)
   - Nonlinear: y' = y²

3. Autonomy:
   - Autonomous: y' = f(y) (t not explicitly present)
   - Non-autonomous: y' = f(t, y)

4. Problem type:
   - Initial Value Problem (IVP): y(t₀) = y₀ given
   - Boundary Value Problem (BVP): y(a) = α, y(b) = β given
"""
```

### 1.3 Example ODEs

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Population growth model (exponential growth)
# dy/dt = ky, y(0) = y₀
# Solution: y(t) = y₀ * e^(kt)

def population_growth():
    k = 0.1  # Growth rate
    y0 = 100  # Initial population

    t = np.linspace(0, 50, 100)
    y = y0 * np.exp(k * t)

    plt.figure(figsize=(10, 4))
    plt.plot(t, y)
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(f'Population Growth Model (k={k})')
    plt.grid(True)
    plt.show()

population_growth()

# 2. Radioactive decay
# dN/dt = -λN, N(0) = N₀
# Solution: N(t) = N₀ * e^(-λt)

def radioactive_decay():
    lambda_ = 0.05
    N0 = 1000
    half_life = np.log(2) / lambda_

    t = np.linspace(0, 100, 100)
    N = N0 * np.exp(-lambda_ * t)

    plt.figure(figsize=(10, 4))
    plt.plot(t, N)
    plt.axhline(y=N0/2, color='r', linestyle='--', label=f'Half-life: {half_life:.1f}')
    plt.xlabel('Time')
    plt.ylabel('Number of atoms')
    plt.title('Radioactive Decay')
    plt.legend()
    plt.grid(True)
    plt.show()

radioactive_decay()
```

---

## 2. Analytical Solutions

### 2.1 Separation of Variables

```python
"""
Form: dy/dt = g(t)h(y)

1/h(y) dy = g(t) dt
∫ 1/h(y) dy = ∫ g(t) dt

Example: dy/dt = ty
1/y dy = t dt
ln|y| = t²/2 + C
y = Ae^(t²/2)
"""

def separable_ode_example():
    # dy/dt = ty, y(0) = 1
    t = np.linspace(-2, 2, 100)
    y = np.exp(t**2 / 2)  # Analytical solution

    plt.figure(figsize=(8, 5))
    plt.plot(t, y)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("Solution of dy/dt = ty")
    plt.grid(True)
    plt.show()

separable_ode_example()
```

### 2.2 First-Order Linear ODE

```python
"""
dy/dt + p(t)y = q(t)

Integrating factor: μ(t) = e^(∫p(t)dt)

Solution: y = (1/μ) * ∫ μ*q dt + C/μ

Example: dy/dt + 2y = e^(-t)
μ = e^(2t)
y = e^(-2t) * ∫ e^(2t) * e^(-t) dt
y = e^(-2t) * (e^t + C)
y = e^(-t) + Ce^(-2t)
"""

def linear_ode_example():
    t = np.linspace(0, 5, 100)
    C = 1  # Determined from initial condition
    y = np.exp(-t) + C * np.exp(-2*t)

    plt.figure(figsize=(8, 5))
    plt.plot(t, y)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("Solution of dy/dt + 2y = e^(-t)")
    plt.grid(True)
    plt.show()

linear_ode_example()
```

### 2.3 Second-Order Linear ODE with Constant Coefficients

```python
"""
ay'' + by' + cy = 0

Characteristic equation: ar² + br + c = 0

Case 1: Two distinct real roots r₁, r₂
  y = C₁e^(r₁t) + C₂e^(r₂t)

Case 2: Repeated root r
  y = (C₁ + C₂t)e^(rt)

Case 3: Complex roots α ± βi
  y = e^(αt)(C₁cos(βt) + C₂sin(βt))
"""

def second_order_examples():
    t = np.linspace(0, 10, 200)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Case 1: y'' - 3y' + 2y = 0 (r=1, 2)
    y1 = np.exp(t) - np.exp(2*t)
    axes[0].plot(t, y1)
    axes[0].set_title("Distinct Real Roots")
    axes[0].set_xlabel('t')

    # Case 2: y'' - 2y' + y = 0 (r=1 repeated)
    y2 = (1 + t) * np.exp(t)
    axes[1].plot(t, y2)
    axes[1].set_title("Repeated Root")
    axes[1].set_xlabel('t')

    # Case 3: y'' + y = 0 (r=±i)
    y3 = np.cos(t) + np.sin(t)
    axes[2].plot(t, y3)
    axes[2].set_title("Complex Roots (Oscillation)")
    axes[2].set_xlabel('t')

    plt.tight_layout()
    plt.show()

second_order_examples()
```

---

## 3. Initial Value Problem (IVP)

### 3.1 Existence and Uniqueness

Before solving an IVP numerically, we should ask: does a solution even exist, and is it unique? The Picard-Lindelof theorem provides the answer through the Lipschitz condition. If $f$ is Lipschitz continuous in $y$ near the initial point, the solution exists and is unique. This matters practically because if uniqueness fails, a numerical solver may converge to any of the multiple solutions depending on implementation details.

```python
"""
Lipschitz Condition:

|f(t, y₁) - f(t, y₂)| ≤ L|y₁ - y₂|

If this condition is satisfied, the solution exists and is unique.

Exception case:
dy/dt = 3y^(2/3), y(0) = 0
Solutions: y = 0 (trivial) or y = t³ (not unique)
"""

def non_unique_solution():
    t = np.linspace(-2, 2, 100)

    # Both solutions satisfy initial condition y(0) = 0
    y1 = np.zeros_like(t)  # y = 0
    y2 = t**3  # y = t³

    plt.figure(figsize=(8, 5))
    plt.plot(t, y1, label='y = 0')
    plt.plot(t, y2, label='y = t³')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("dy/dt = 3y^(2/3) - Non-unique Solutions")
    plt.legend()
    plt.grid(True)
    plt.show()

non_unique_solution()
```

### 3.2 Converting Higher-Order ODE to First-Order System

```python
"""
Convert nth-order ODE to system of n first-order ODEs

Example: y'' + 4y' + 3y = 0

Transformation:
y₁ = y
y₂ = y' = y₁'

System:
y₁' = y₂
y₂' = -3y₁ - 4y₂

Matrix form:
[y₁']   [0   1 ] [y₁]
[y₂'] = [-3 -4] [y₂]
"""

def convert_to_system():
    # 2nd order ODE: y'' + 4y' + 3y = 0
    # Initial conditions: y(0) = 1, y'(0) = 0

    # Analytical solution for first-order system
    # Characteristic equation: r² + 4r + 3 = 0 → r = -1, -3
    # Solution: y = Ae^(-t) + Be^(-3t)
    # Apply initial conditions: A + B = 1, -A - 3B = 0
    # A = 3/2, B = -1/2

    t = np.linspace(0, 5, 100)
    y = 1.5 * np.exp(-t) - 0.5 * np.exp(-3*t)
    y_prime = -1.5 * np.exp(-t) + 1.5 * np.exp(-3*t)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t, y, label='y(t)')
    axes[0].plot(t, y_prime, label="y'(t)")
    axes[0].set_xlabel('t')
    axes[0].set_title('Time Domain')
    axes[0].legend()
    axes[0].grid(True)

    # Phase plane
    axes[1].plot(y, y_prime)
    axes[1].set_xlabel('y')
    axes[1].set_ylabel("y'")
    axes[1].set_title('Phase Plane')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

convert_to_system()
```

---

## 4. Phase Plane Analysis

Phase plane analysis provides a geometric view of the solution without actually solving the ODE. By plotting the vector field $(\dot{y}_1, \dot{y}_2)$ at every point, we can visualize all possible trajectories at once, identify equilibrium points, and determine their stability. This qualitative understanding is often more valuable than a single numerical solution.

### 4.1 Equilibrium Points and Stability

```python
"""
For dy/dt = f(y), equilibrium points: f(y*) = 0

Stability:
- f'(y*) < 0: Stable (converging)
- f'(y*) > 0: Unstable (diverging)
- f'(y*) = 0: Further analysis required
"""

def equilibrium_analysis():
    # dy/dt = y(1 - y) (logistic growth)
    # Equilibrium points: y* = 0 or y* = 1

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
    # f'(0) = 1 > 0: Unstable
    # f'(1) = -1 < 0: Stable

    # Time evolution
    plt.subplot(1, 2, 2)
    from scipy.integrate import solve_ivp

    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 100)

    for y0 in [-0.1, 0.1, 0.5, 0.9, 1.1, 1.5]:
        sol = solve_ivp(lambda t, y: y*(1-y), t_span, [y0], t_eval=t_eval)
        plt.plot(sol.t, sol.y[0], label=f'y₀={y0}')

    plt.axhline(y=1, color='r', linestyle='--', label='Stable equilibrium')
    plt.axhline(y=0, color='b', linestyle='--', label='Unstable equilibrium')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend(loc='right')
    plt.title('Solutions from Various Initial Conditions')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

equilibrium_analysis()
```

### 4.2 2D Phase Plane

```python
def phase_plane_2d():
    """Phase plane for 2D system"""
    # Simple harmonic oscillator: x'' + x = 0
    # System: x' = v, v' = -x

    def harmonic_oscillator(t, state):
        x, v = state
        return [v, -x]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Vector field
    x_range = np.linspace(-2, 2, 15)
    v_range = np.linspace(-2, 2, 15)
    X, V = np.meshgrid(x_range, v_range)
    dX = V
    dV = -X

    axes[0].quiver(X, V, dX, dV, alpha=0.5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('v')
    axes[0].set_title('Vector Field')
    axes[0].set_aspect('equal')

    # Trajectories
    from scipy.integrate import solve_ivp

    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 200)

    for x0, v0 in [(1, 0), (0, 1), (1.5, 0.5)]:
        sol = solve_ivp(harmonic_oscillator, t_span, [x0, v0], t_eval=t_eval)
        axes[1].plot(sol.y[0], sol.y[1], label=f'({x0}, {v0})')

    axes[1].set_xlabel('x')
    axes[1].set_ylabel('v')
    axes[1].set_title('Phase Trajectories')
    axes[1].legend()
    axes[1].set_aspect('equal')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

phase_plane_2d()
```

---

## 5. Physical Examples

### 5.1 Free Fall

```python
def free_fall():
    """Free fall under gravity (with air resistance)"""
    from scipy.integrate import solve_ivp

    # Parameters
    g = 9.8  # Gravitational acceleration
    k = 0.1  # Air resistance coefficient
    m = 1.0  # Mass

    # Equation of motion: m*dv/dt = mg - kv²
    # dv/dt = g - (k/m)v²

    def fall_with_drag(t, state):
        v = state[0]
        return [g - (k/m) * v * abs(v)]

    # Terminal velocity: v_terminal = sqrt(mg/k)
    v_terminal = np.sqrt(m * g / k)

    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 200)

    sol = solve_ivp(fall_with_drag, t_span, [0], t_eval=t_eval)

    plt.figure(figsize=(10, 5))
    plt.plot(sol.t, sol.y[0])
    plt.axhline(y=v_terminal, color='r', linestyle='--',
                label=f'Terminal velocity: {v_terminal:.2f} m/s')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Free Fall with Air Resistance')
    plt.legend()
    plt.grid(True)
    plt.show()

free_fall()
```

### 5.2 RC Circuit

```python
def rc_circuit():
    """Transient response of RC circuit"""
    from scipy.integrate import solve_ivp

    # Parameters
    R = 1000  # Resistance (Ω)
    C = 1e-6  # Capacitance (F)
    V_source = 5  # Source voltage (V)
    tau = R * C  # Time constant

    # Charging: V_C' = (V_source - V_C) / (RC)
    def charging(t, V_C):
        return [(V_source - V_C[0]) / (R * C)]

    t_span = (0, 5 * tau)
    t_eval = np.linspace(0, 5*tau, 200)

    sol = solve_ivp(charging, t_span, [0], t_eval=t_eval)

    # Analytical solution
    t_analytical = np.linspace(0, 5*tau, 200)
    V_analytical = V_source * (1 - np.exp(-t_analytical / tau))

    plt.figure(figsize=(10, 5))
    plt.plot(sol.t * 1000, sol.y[0], 'b-', label='Numerical solution')
    plt.plot(t_analytical * 1000, V_analytical, 'r--', label='Analytical solution')
    plt.axhline(y=V_source * 0.632, color='g', linestyle=':',
                label=f'τ = {tau*1000:.3f} ms')
    plt.xlabel('Time (ms)')
    plt.ylabel('Capacitor Voltage (V)')
    plt.title('RC Circuit Charging')
    plt.legend()
    plt.grid(True)
    plt.show()

rc_circuit()
```

---

## Practice Problems

### Problem 1
Newton's law of cooling: dT/dt = -k(T - T_ambient)
Plot temperature vs time for initial temperature 90°C, ambient temperature 20°C, k=0.1.

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
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title("Newton's Law of Cooling")
    plt.grid(True)
    plt.show()

exercise_1()
```

### Problem 2
Damped oscillation: x'' + 2γx' + ω₀²x = 0
Plot phase plane and time response for ω₀ = 2, γ = 0.5.

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
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('x')
    axes[0].set_title('Time Response')
    axes[0].grid(True)

    axes[1].plot(sol.y[0], sol.y[1])
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('v')
    axes[1].set_title('Phase Plane')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

exercise_2()
```

---

## Summary

| Concept | Content |
|------|------|
| ODE classification | Order, linearity, autonomy |
| Analytical solutions | Separation of variables, integrating factor, characteristic equation |
| Higher→First conversion | nth-order ODE → n first-order systems |
| Phase plane | Equilibrium points, stability, trajectory analysis |
| Existence/Uniqueness | Lipschitz condition |

## Exercises

### Exercise 1: Classify and Solve a First-Order Separable ODE

Classify the ODE `dy/dt = -2ty²` (order, linearity, autonomy) and find the general solution by separation of variables. Then apply the initial condition `y(0) = 1/3` to obtain the particular solution and verify it satisfies the ODE.

<details>
<summary>Show Answer</summary>

**Classification:**
- Order: 1st (highest derivative is dy/dt)
- Linearity: Nonlinear (contains y²)
- Autonomy: Non-autonomous (t appears explicitly)

**Separation of variables:**

```
dy/dt = -2ty²
dy/y² = -2t dt
∫ y⁻² dy = ∫ -2t dt
-1/y = -t² + C
y = 1/(t² - C) = 1/(t² + K)  where K = -C
```

Applying `y(0) = 1/3`: `1/3 = 1/K` → `K = 3`

Particular solution: `y(t) = 1/(t² + 3)`

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Analytical solution
t_vals = np.linspace(0, 3, 200)
y_exact = 1 / (t_vals**2 + 3)

# Numerical verification
def ode(t, y):
    return [-2 * t * y[0]**2]

sol = solve_ivp(ode, (0, 3), [1/3], t_eval=t_vals, rtol=1e-10)

plt.figure(figsize=(8, 4))
plt.plot(t_vals, y_exact, 'b-', linewidth=2, label='Analytical: 1/(t²+3)')
plt.plot(sol.t, sol.y[0], 'r--', linewidth=1.5, label='Numerical (RK45)')
plt.xlabel('t')
plt.ylabel('y')
plt.title('dy/dt = -2ty², y(0) = 1/3')
plt.legend()
plt.grid(True)
plt.show()

# Verify: dy/dt = -2t * y² should equal -2ty²
dy_dt = np.gradient(y_exact, t_vals)
rhs   = -2 * t_vals * y_exact**2
print(f"Max residual |dy/dt - f(t,y)|: {np.max(np.abs(dy_dt - rhs)):.2e}")
```

</details>

### Exercise 2: Characteristic Equation and Damped Oscillation

Solve the 2nd-order ODE `y'' + 4y' + 5y = 0` with initial conditions `y(0) = 2, y'(0) = 0` by:
1. Finding the characteristic equation and its roots.
2. Writing the general solution using the complex root formula.
3. Applying initial conditions to find constants C₁ and C₂.
4. Plotting the solution and identifying the damped frequency and decay rate.

<details>
<summary>Show Answer</summary>

**Step 1:** Characteristic equation: `r² + 4r + 5 = 0`

Roots: `r = (-4 ± √(16 - 20))/2 = -2 ± i`

**Step 2:** General solution (complex roots `α ± βi` → `α = -2, β = 1`):
```
y(t) = e^(-2t)(C₁ cos(t) + C₂ sin(t))
```

**Step 3:** Apply initial conditions:
- `y(0) = C₁ = 2` → `C₁ = 2`
- `y'(t) = e^(-2t)[(-2C₁ + C₂)cos(t) + (-2C₂ - C₁)sin(t)]`
- `y'(0) = -2C₁ + C₂ = 0` → `C₂ = 2·2 = 4`

Particular solution: `y(t) = e^(-2t)(2cos(t) + 4sin(t))`

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

t = np.linspace(0, 5, 300)

# Analytical solution
C1, C2 = 2, 4
y_exact = np.exp(-2 * t) * (C1 * np.cos(t) + C2 * np.sin(t))
envelope = np.sqrt(C1**2 + C2**2) * np.exp(-2 * t)  # amplitude envelope

# Numerical check
def ode(t, y):
    return [y[1], -4*y[1] - 5*y[0]]

sol = solve_ivp(ode, (0, 5), [2, 0], t_eval=t, rtol=1e-10)

plt.figure(figsize=(9, 4))
plt.plot(t, y_exact, 'b-', label='Analytical')
plt.plot(sol.t, sol.y[0], 'r--', alpha=0.7, label='Numerical')
plt.plot(t,  envelope, 'g:', label='Envelope ±√(C₁²+C₂²)·e^(-2t)')
plt.plot(t, -envelope, 'g:')
plt.xlabel('t')
plt.ylabel('y')
plt.title("y'' + 4y' + 5y = 0,  α = -2, ω_d = 1 rad/s")
plt.legend()
plt.grid(True)
plt.show()

print(f"Decay rate α = 2,  Damped frequency ω_d = 1 rad/s")
print(f"Max error vs numerical: {np.max(np.abs(y_exact - sol.y[0])):.2e}")
```

</details>

### Exercise 3: Stability of Equilibrium Points in a Nonlinear ODE

For the autonomous ODE `dy/dt = y³ - y` (a.k.a. `dy/dt = y(y-1)(y+1)`):
1. Find all equilibrium points analytically.
2. Determine the stability of each equilibrium using linear stability analysis (`f'(y*)`).
3. Simulate trajectories from initial conditions `y₀ = -1.5, -0.5, 0.5, 1.5` and confirm each converges or diverges as predicted.

<details>
<summary>Show Answer</summary>

**Step 1:** Equilibrium points where `f(y*) = 0`:
```
y³ - y = y(y² - 1) = y(y-1)(y+1) = 0
y* = -1, 0, +1
```

**Step 2:** Linear stability: `f'(y) = 3y² - 1`
- `y* = -1`: `f'(-1) = 3 - 1 = 2 > 0` → **Unstable**
- `y* = 0`:  `f'(0)  = 0 - 1 = -1 < 0` → **Stable**
- `y* = +1`: `f'(+1) = 3 - 1 = 2 > 0` → **Unstable**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f(y):
    return y**3 - y

t_span = (0, 8)
t_eval = np.linspace(0, 8, 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Phase portrait
y_phase = np.linspace(-1.8, 1.8, 300)
axes[0].plot(y_phase, f(y_phase), 'b-', linewidth=2)
axes[0].axhline(0, color='k', linewidth=0.8)
for y_eq, stable in [(-1, False), (0, True), (1, False)]:
    color = 'green' if stable else 'red'
    label = 'Stable' if stable else 'Unstable'
    axes[0].scatter([y_eq], [0], s=100, color=color, zorder=5, label=f'y*={y_eq} ({label})')
axes[0].set_xlabel('y')
axes[0].set_ylabel("dy/dt = y³ - y")
axes[0].set_title("Phase Portrait")
axes[0].legend()
axes[0].grid(True)

# Trajectories
for y0 in [-1.5, -0.5, 0.5, 1.5]:
    sol = solve_ivp(lambda t, y: [f(y[0])], t_span, [y0], t_eval=t_eval)
    axes[1].plot(sol.t, sol.y[0], label=f'y₀={y0}')

for y_eq, stable in [(-1, False), (0, True), (1, False)]:
    ls = '--' if stable else ':'
    axes[1].axhline(y_eq, color='gray', linestyle=ls, alpha=0.6)
axes[1].set_xlabel('t')
axes[1].set_ylabel('y(t)')
axes[1].set_title('Solution Trajectories')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

Trajectories starting between ±1 converge to y* = 0 (stable), while those outside diverge (unstable equilibria repel). This matches the linear stability analysis.

</details>

### Exercise 4: Converting a 3rd-Order ODE to a First-Order System

Convert the 3rd-order ODE `y''' - y'' + 4y' - 4y = 0` to a system of three first-order ODEs. Write the system in matrix form, find the characteristic equation of the coefficient matrix, and use `scipy.integrate.solve_ivp` to numerically solve with initial conditions `y(0) = 1, y'(0) = 0, y''(0) = -1`.

<details>
<summary>Show Answer</summary>

**State vector:** Let `u₁ = y, u₂ = y', u₃ = y''`

**System:**
```
u₁' = u₂
u₂' = u₃
u₃' = 4u₁ - 4u₂ + u₃
```

**Matrix form:**
```
d/dt [u₁]   [ 0  1  0] [u₁]
     [u₂] = [ 0  0  1] [u₂]
     [u₃]   [ 4 -4  1] [u₃]
```

**Characteristic equation:** `r³ - r² + 4r - 4 = 0` → `r²(r-1) + 4(r-1) = 0` → `(r-1)(r²+4) = 0`

Roots: `r = 1, ±2i` → general solution: `y = C₁eᵗ + C₂cos(2t) + C₃sin(2t)`

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def system(t, u):
    u1, u2, u3 = u
    return [u2, u3, 4*u1 - 4*u2 + u3]

# Initial conditions: y(0)=1, y'(0)=0, y''(0)=-1
u0 = [1.0, 0.0, -1.0]
t_span = (0, 4)
t_eval = np.linspace(0, 4, 400)

sol = solve_ivp(system, t_span, u0, t_eval=t_eval, method='RK45', rtol=1e-10)

# Find constants from initial conditions
# y(0) = C1 + C2 = 1
# y'(0) = C1 + 2*C3 = 0  (derivative of C2*cos + C3*sin is -2C2*sin + 2C3*cos)
# y''(0) = C1 - 4*C2 = -1
# Solve: C1+C2=1, C1+2C3=0, C1-4C2=-1
# From eq1-eq3: 5C2=2 -> C2=0.4, C1=0.6; C3=-C1/2=-0.3
C1, C2, C3 = 0.6, 0.4, -0.3
y_analytic = C1*np.exp(t_eval) + C2*np.cos(2*t_eval) + C3*np.sin(2*t_eval)

plt.figure(figsize=(9, 4))
plt.plot(t_eval, sol.y[0], 'b-', linewidth=2, label='Numerical y(t)')
plt.plot(t_eval, y_analytic, 'r--', linewidth=1.5, label='Analytical y(t)')
plt.xlabel('t')
plt.ylabel('y')
plt.title("y''' - y'' + 4y' - 4y = 0")
plt.legend()
plt.grid(True)
plt.show()

print(f"Max error: {np.max(np.abs(sol.y[0] - y_analytic)):.2e}")
```

</details>
