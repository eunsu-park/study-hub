# ODE Advanced

## Learning Objectives

After completing this lesson, you will be able to:

1. Define stiffness in ODE systems and explain why explicit methods fail for stiff problems.
2. Implement implicit methods (backward Euler, trapezoidal rule) and explain how they achieve A-stability.
3. Apply `scipy.integrate.solve_ivp` with stiff solvers (Radau, BDF) and configure appropriate tolerances.
4. Analyze the eigenvalue structure of a system to diagnose stiffness and select an appropriate solver.
5. Solve practical stiff ODE problems from chemical kinetics and electrical circuits.

---

## Overview

Learn about stiff problems, implicit methods, and advanced usage of scipy.integrate. We'll tackle difficult ODE problems that frequently appear in real-world applications.

**Why This Lesson Matters:** Many real-world systems -- chemical reactions, electrical circuits, biological networks -- contain processes that evolve on vastly different time scales. A chemical reaction may have a fast transient that settles in microseconds alongside a slow equilibrium that takes hours. Naive explicit solvers would need microsecond-sized steps for the entire simulation, making them impractical. Implicit methods and stiff solvers are the essential tools that make such problems tractable.

---

## 1. Stiff Problems

### 1.1 Definition of Stiffness

A system is **stiff** when it contains widely separated time scales. Formally, stiffness is characterized by the ratio of the largest to smallest eigenvalue magnitudes of the Jacobian matrix:

$$\text{Stiffness ratio} = \frac{|\lambda_{\max}|}{|\lambda_{\min}|}$$

When this ratio is large (e.g., 500 or more), explicit methods must use a time step dictated by the fastest mode ($\Delta t \lesssim 2/|\lambda_{\max}|$ for stability), even though the solution of interest evolves on the slow time scale ($1/|\lambda_{\min}|$). This wastes enormous computational effort on a component that has already decayed.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
Characteristics of stiff problems:
1. Solution components have vastly different time scales
2. Some components decay rapidly, others change slowly
3. Explicit methods require very small steps for stability
"""

def stiff_example():
    """Example of a stiff system"""
    # dy₁/dt = -500*y₁ + 500*y₂
    # dy₂/dt = y₁ - y₂
    #
    # Eigenvalues: λ₁ ≈ -500.002, λ₂ ≈ -0.998
    # Ratio: |λ₁/λ₂| ≈ 500 (stiffness ratio)

    # The fast mode (λ₁ ≈ -500) decays in ~0.002 time units, but the slow mode
    # (λ₂ ≈ -1) needs ~5 time units to evolve. RK45 must track the fast mode's
    # stability limit the entire time, while Radau's implicit formulation is
    # unconditionally stable and can take large steps once the transient decays.
    def stiff_system(t, y):
        return [-500*y[0] + 500*y[1], y[0] - y[1]]

    y0 = [1.0, 0.0]
    t_span = (0, 5)

    # Explicit RK45: stability region limits dt to ~2/500 = 0.004, forcing many steps
    sol_rk45 = solve_ivp(stiff_system, t_span, y0, method='RK45',
                         rtol=1e-6, atol=1e-9)

    # Implicit Radau: A-stable, so dt is limited only by accuracy, not stability
    sol_radau = solve_ivp(stiff_system, t_span, y0, method='Radau',
                          rtol=1e-6, atol=1e-9)

    print(f"RK45 steps: {len(sol_rk45.t)}")
    print(f"Radau steps: {len(sol_radau.t)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(sol_rk45.t, np.abs(sol_rk45.y[0]), 'b-', label='y₁')
    axes[0].semilogy(sol_rk45.t, np.abs(sol_rk45.y[1]), 'r-', label='y₂')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('|y|')
    axes[0].set_title(f'RK45 ({len(sol_rk45.t)} steps)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].semilogy(sol_radau.t, np.abs(sol_radau.y[0]), 'b-', label='y₁')
    axes[1].semilogy(sol_radau.t, np.abs(sol_radau.y[1]), 'r-', label='y₂')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('|y|')
    axes[1].set_title(f'Radau ({len(sol_radau.t)} steps)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

stiff_example()
```

### 1.2 Chemical Reaction Kinetics

The Robertson problem is a classic benchmark for stiff solvers. The three reaction rate constants span 9 orders of magnitude ($k_1 = 0.04$, $k_2 = 3 \times 10^7$, $k_3 = 10^4$), creating extreme stiffness. Species B reaches a quasi-steady-state almost instantly, while A and C evolve over millions of time units.

```python
def chemical_kinetics():
    """Stiff chemical reaction system (Robertson problem)"""
    # A → B (k₁ = 0.04)
    # B + B → C + B (k₂ = 3e7)
    # B + C → A + C (k₃ = 1e4)

    k1, k2, k3 = 0.04, 3e7, 1e4

    def robertson(t, y):
        A, B, C = y
        dA = -k1*A + k3*B*C
        dB = k1*A - k2*B*B - k3*B*C
        dC = k2*B*B
        return [dA, dB, dC]

    y0 = [1.0, 0.0, 0.0]
    t_span = (0, 1e7)
    t_eval = np.logspace(-5, 7, 200)

    # BDF is chosen because it damps high-frequency transients efficiently.
    # Logarithmic time spacing matches the solution's multi-scale behavior:
    # rapid early transient (t < 1) followed by slow equilibration (t ~ 10^7).
    sol = solve_ivp(robertson, t_span, y0, method='BDF',
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    axes[0].semilogx(sol.t, sol.y[0], label='A')
    axes[0].semilogx(sol.t, sol.y[2], label='C')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('Concentration')
    axes[0].set_title('Robertson Chemical Reaction (A, C)')
    axes[0].legend()
    axes[0].grid(True)

    # B is very small, so display separately
    axes[1].semilogx(sol.t, sol.y[1] * 1e4, label='B × 10⁴')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('Concentration × 10⁴')
    axes[1].set_title('Robertson Chemical Reaction (B)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

chemical_kinetics()
```

---

## 2. Implicit Methods

**Why Implicit Methods?** Explicit methods evaluate $f$ at the current time to predict the future. Implicit methods evaluate $f$ at the future time, requiring the solution of a (generally nonlinear) equation at each step. This extra cost buys unconditional stability: the method remains stable regardless of step size, so $\Delta t$ can be chosen based on accuracy rather than stability constraints.

### 2.1 Backward Euler (Revisited)

The Backward Euler method is the simplest implicit scheme. At each step we must solve:

$$y_{n+1} = y_n + h \cdot f(t_{n+1}, y_{n+1})$$

Since $y_{n+1}$ appears on both sides, we use Newton-Raphson iteration. Define the residual $F(y_{n+1}) = y_{n+1} - y_n - h \cdot f(t_{n+1}, y_{n+1})$ and its Jacobian $J_F = I - h \cdot J_f$, where $J_f = \partial f / \partial y$. Newton's update is $\delta = -J_F^{-1} F$, converging quadratically when the initial guess is close.

```python
def backward_euler_system(f, jacobian, y0, t_span, n_steps, tol=1e-10):
    """
    Backward Euler + Newton-Raphson for systems

    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
    """
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]
    n = len(y0)

    y = np.zeros((n_steps + 1, n))
    y[0] = y0

    for i in range(n_steps):
        # Use current solution as initial guess for Newton iteration
        y_guess = y[i].copy()

        for _ in range(100):
            # Residual: should be zero when y_guess satisfies the implicit equation
            F = y_guess - y[i] - h * np.array(f(t[i+1], y_guess))
            # Newton Jacobian: I - h*J_f; solving J*delta = -F is the key cost
            J = np.eye(n) - h * np.array(jacobian(t[i+1], y_guess))

            delta = np.linalg.solve(J, -F)
            y_guess = y_guess + delta

            # Quadratic convergence: once close, each iteration squares the error
            if np.linalg.norm(delta) < tol:
                break

        y[i+1] = y_guess

    return t, y

# Test: stiff system
def stiff_f(t, y):
    return [-500*y[0] + 500*y[1], y[0] - y[1]]

def stiff_jacobian(t, y):
    return [[-500, 500], [1, -1]]

t, y = backward_euler_system(stiff_f, stiff_jacobian, [1.0, 0.0], (0, 1), 50)

plt.figure(figsize=(10, 5))
plt.plot(t, y[:, 0], 'b-o', label='y₁')
plt.plot(t, y[:, 1], 'r-o', label='y₂')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Backward Euler (50 steps)')
plt.legend()
plt.grid(True)
plt.show()
```

### 2.2 Crank-Nicolson Method

The Crank-Nicolson method averages the explicit and implicit Euler methods, yielding second-order accuracy while remaining A-stable:

$$y_{n+1} = y_n + \frac{h}{2}\bigl[f(t_n, y_n) + f(t_{n+1}, y_{n+1})\bigr]$$

This is equivalent to applying the trapezoidal quadrature rule to the integral form of the ODE. The trade-off: it is A-stable but not L-stable, meaning it does not damp high-frequency oscillations as aggressively as Backward Euler. For stiff problems with oscillatory transients, this can produce small spurious oscillations.

```python
def crank_nicolson(f, jacobian, y0, t_span, n_steps, tol=1e-10):
    """
    Crank-Nicolson (Trapezoidal Rule)

    y_{n+1} = y_n + h/2 * (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))

    2nd order accuracy, A-stable
    """
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]
    n = len(y0)

    y = np.zeros((n_steps + 1, n))
    y[0] = y0

    for i in range(n_steps):
        f_n = np.array(f(t[i], y[i]))
        y_guess = y[i] + h * f_n  # Initial guess (forward Euler)

        for _ in range(100):
            f_new = np.array(f(t[i+1], y_guess))
            F = y_guess - y[i] - h/2 * (f_n + f_new)
            J = np.eye(n) - h/2 * np.array(jacobian(t[i+1], y_guess))

            delta = np.linalg.solve(J, -F)
            y_guess = y_guess + delta

            if np.linalg.norm(delta) < tol:
                break

        y[i+1] = y_guess

    return t, y

# Comparison
t_be, y_be = backward_euler_system(stiff_f, stiff_jacobian, [1.0, 0.0], (0, 1), 20)
t_cn, y_cn = crank_nicolson(stiff_f, stiff_jacobian, [1.0, 0.0], (0, 1), 20)

# Reference solution
sol_ref = solve_ivp(stiff_f, (0, 1), [1.0, 0.0], method='Radau',
                    t_eval=np.linspace(0, 1, 200))

plt.figure(figsize=(10, 5))
plt.plot(sol_ref.t, sol_ref.y[0], 'k-', linewidth=2, label='Reference')
plt.plot(t_be, y_be[:, 0], 'bo-', label='Backward Euler')
plt.plot(t_cn, y_cn[:, 0], 'rs-', label='Crank-Nicolson')
plt.xlabel('t')
plt.ylabel('y₁')
plt.title('Comparison of Implicit Methods')
plt.legend()
plt.grid(True)
plt.show()
```

### 2.3 BDF Methods

BDF methods approximate the derivative using backward differences of the solution history, then set it equal to $f$ at the new time. Higher-order BDF uses more past points for a better derivative approximation:

$$\text{BDF}k: \quad \sum_{j=0}^{k} \alpha_j \, y_{n+1-j} = h \cdot f(t_{n+1}, y_{n+1})$$

The key property is **L-stability** (for BDF1-2): stiff components are damped to zero in one step, unlike Crank-Nicolson which merely keeps them bounded. BDF3-6 lose A-stability but remain useful with adaptive order selection (as in scipy's `BDF` solver).

```python
"""
BDF (Backward Differentiation Formula) Methods

BDF1 (Backward Euler):
    y_{n+1} - y_n = h * f(t_{n+1}, y_{n+1})

BDF2:
    (3y_{n+1} - 4y_n + y_{n-1}) / 2 = h * f(t_{n+1}, y_{n+1})

BDF3:
    (11y_{n+1} - 18y_n + 9y_{n-1} - 2y_{n-2}) / 6 = h * f(t_{n+1}, y_{n+1})

Characteristics:
- A-stable (BDF1, BDF2)
- Effective for stiff problems
- scipy's 'BDF' solver implements variable-order BDF
"""

def bdf2_solver(f, jacobian, y0, t_span, n_steps, tol=1e-10):
    """BDF2 method"""
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]
    n = len(y0)

    y = np.zeros((n_steps + 1, n))
    y[0] = y0

    # BDF2 is a 2-step method, so the first step must use a 1-step method (BDF1).
    # This "bootstrap" is unavoidable for any multistep method.
    y_guess = y[0].copy()
    for _ in range(100):
        F = y_guess - y[0] - h * np.array(f(t[1], y_guess))
        J = np.eye(n) - h * np.array(jacobian(t[1], y_guess))
        delta = np.linalg.solve(J, -F)
        y_guess = y_guess + delta
        if np.linalg.norm(delta) < tol:
            break
    y[1] = y_guess

    # Subsequent steps using BDF2
    for i in range(1, n_steps):
        y_guess = y[i].copy()

        for _ in range(100):
            f_new = np.array(f(t[i+1], y_guess))
            # BDF2: (3y_{n+1} - 4y_n + y_{n-1})/2 = h*f(t_{n+1}, y_{n+1})
            F = 1.5*y_guess - 2*y[i] + 0.5*y[i-1] - h * f_new
            J = 1.5*np.eye(n) - h * np.array(jacobian(t[i+1], y_guess))

            delta = np.linalg.solve(J, -F)
            y_guess = y_guess + delta

            if np.linalg.norm(delta) < tol:
                break

        y[i+1] = y_guess

    return t, y

t_bdf2, y_bdf2 = bdf2_solver(stiff_f, stiff_jacobian, [1.0, 0.0], (0, 1), 20)

plt.figure(figsize=(10, 5))
plt.plot(sol_ref.t, sol_ref.y[0], 'k-', linewidth=2, label='Reference')
plt.plot(t_bdf2, y_bdf2[:, 0], 'go-', label='BDF2')
plt.xlabel('t')
plt.ylabel('y₁')
plt.title('BDF2 Method')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 3. scipy.integrate Advanced

### 3.1 Providing Jacobian

Implicit solvers must solve a nonlinear system at each step, which requires the Jacobian $J = \partial f / \partial y$. If not provided, the solver estimates it by finite differences (requiring $n$ extra function evaluations per Jacobian update). Providing an analytical Jacobian can significantly reduce computation time, especially for large systems.

```python
def with_jacobian():
    """Performance improvement when providing Jacobian"""

    def system(t, y):
        return [
            -100*y[0] + y[1],
            y[0] - 100*y[1] + y[2],
            y[1] - 100*y[2]
        ]

    def jacobian(t, y):
        return [
            [-100, 1, 0],
            [1, -100, 1],
            [0, 1, -100]
        ]

    y0 = [1.0, 0.0, 0.0]
    t_span = (0, 0.5)

    import time

    # Without Jacobian
    start = time.time()
    sol1 = solve_ivp(system, t_span, y0, method='Radau')
    time1 = time.time() - start

    # With Jacobian
    start = time.time()
    sol2 = solve_ivp(system, t_span, y0, method='Radau', jac=jacobian)
    time2 = time.time() - start

    print(f"Without Jacobian: {time1:.4f}s, {len(sol1.t)} steps")
    print(f"With Jacobian: {time2:.4f}s, {len(sol2.t)} steps")

with_jacobian()
```

### 3.2 Dense Output

Adaptive solvers take variable-sized steps, so the solution is naturally available only at solver-chosen time points. Dense output provides a continuous polynomial interpolant between steps, allowing evaluation at arbitrary times without re-running the solver. This is essential for event detection (e.g., finding zero-crossings) and smooth visualization.

```python
def dense_output_example():
    """Obtaining continuous solution with dense output"""

    def oscillator(t, y):
        return [y[1], -y[0]]

    sol = solve_ivp(oscillator, (0, 10), [1, 0],
                    method='RK45', dense_output=True)

    # Evaluate solution at arbitrary times
    t_dense = np.linspace(0, 10, 1000)
    y_dense = sol.sol(t_dense)

    plt.figure(figsize=(10, 5))
    plt.plot(t_dense, y_dense[0], 'b-', label='Dense output')
    plt.plot(sol.t, sol.y[0], 'ro', markersize=5, label='Solver steps')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Dense Output')
    plt.legend()
    plt.grid(True)
    plt.show()

dense_output_example()
```

### 3.3 Mass Matrix (DAE)

A Differential-Algebraic Equation (DAE) has the form $M \cdot y' = f(t, y)$ where $M$ may be singular. Rows of $M$ that are zero correspond to algebraic constraints (equations with no derivatives), which couple the differential equations. DAEs arise naturally in constrained mechanical systems (pendulums, robots) and electrical circuits (Kirchhoff's laws impose algebraic constraints on voltages and currents).

```python
def dae_example():
    """
    Differential-Algebraic Equations (DAE)

    M * y' = f(t, y)

    If M is a singular matrix, includes algebraic constraints
    """

    # Example: simple pendulum (constraint: x² + y² = L²)
    # x'' = λx
    # y'' = λy - g
    # x² + y² = L²

    # Converted to index-1 form
    g = 9.8
    L = 1.0

    def pendulum(t, y):
        x, y_pos, vx, vy, lam = y

        # Acceleration
        ax = lam * x
        ay = lam * y_pos - g

        # Constraint (determines λ)
        # Actually this requires more complex handling...
        # Just showing a simple example

        return [vx, vy, ax, ay, 0]

    # In scipy, DAEs can be solved using Radau solver with mass_matrix option
    # Here we just explain the concept

    print("DAEs can be solved using scipy.integrate.solve_ivp's Radau solver")
    print("with the mass_matrix option.")

dae_example()
```

---

## 4. Boundary Value Problems (BVP)

**Why BVPs are different from IVPs:** Initial value problems specify all conditions at one point and march forward in time. Boundary value problems specify conditions at two (or more) separate points, so we cannot simply march forward -- the solution at the left boundary depends on what happens at the right boundary. This requires either iterative "shooting" or solving the entire domain simultaneously.

### 4.1 Shooting Method

The shooting method converts a BVP into a sequence of IVPs. We guess the unknown initial conditions, integrate forward, check whether the boundary condition at the far end is satisfied, and adjust the guess. It is analogous to aiming a cannon: adjust the launch angle (initial slope) until the projectile hits the target (far boundary).

```python
from scipy.optimize import brentq

def shooting_method():
    """
    Solving BVP using shooting method

    y'' = -y, y(0) = 0, y(π) = 0
    Exact solution: y = sin(x)
    """

    def ode(t, y):
        return [y[1], -y[0]]

    def shoot(initial_slope):
        """Solve IVP with given initial slope"""
        sol = solve_ivp(ode, (0, np.pi), [0, initial_slope],
                        dense_output=True)
        return sol.sol(np.pi)[0]  # Return y(π)

    # Find correct initial slope
    slope = brentq(shoot, 0.5, 1.5)
    print(f"Found initial slope: {slope:.6f} (exact: 1.0)")

    # Final solution
    sol = solve_ivp(ode, (0, np.pi), [0, slope], dense_output=True)
    t = np.linspace(0, np.pi, 100)

    plt.figure(figsize=(10, 5))
    plt.plot(t, sol.sol(t)[0], 'b-', label='Numerical')
    plt.plot(t, np.sin(t), 'r--', label='Exact sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Shooting Method')
    plt.legend()
    plt.grid(True)
    plt.show()

shooting_method()
```

### 4.2 scipy.integrate.solve_bvp

```python
from scipy.integrate import solve_bvp

def bvp_example():
    """
    y'' + y = 0, y(0) = 0, y(π) = 0

    Convert to first-order system:
    y₁' = y₂
    y₂' = -y₁
    """

    def ode(x, y):
        return np.vstack([y[1], -y[0]])

    def bc(ya, yb):
        return np.array([ya[0], yb[0]])  # y(0) = 0, y(π) = 0

    # Initial guess
    x = np.linspace(0, np.pi, 10)
    y = np.zeros((2, x.size))
    y[0] = np.sin(x)  # Initial guess

    sol = solve_bvp(ode, bc, x, y)

    x_plot = np.linspace(0, np.pi, 100)
    y_plot = sol.sol(x_plot)

    plt.figure(figsize=(10, 5))
    plt.plot(x_plot, y_plot[0], 'b-', label='Numerical')
    plt.plot(x_plot, np.sin(x_plot), 'r--', label='Exact')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('solve_bvp')
    plt.legend()
    plt.grid(True)
    plt.show()

bvp_example()
```

### 4.3 Nonlinear BVP

```python
def nonlinear_bvp():
    """
    Nonlinear BVP: y'' = y² - 1
    y(0) = 0, y(1) = 1
    """

    def ode(x, y):
        return np.vstack([y[1], y[0]**2 - 1])

    def bc(ya, yb):
        return np.array([ya[0], yb[0] - 1])

    x = np.linspace(0, 1, 10)
    y = np.zeros((2, x.size))
    y[0] = x  # Linear initial guess

    sol = solve_bvp(ode, bc, x, y)

    if sol.success:
        x_plot = np.linspace(0, 1, 100)
        y_plot = sol.sol(x_plot)

        plt.figure(figsize=(10, 5))
        plt.plot(x_plot, y_plot[0], 'b-', linewidth=2)
        plt.scatter([0, 1], [0, 1], color='red', s=100, zorder=5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("Nonlinear BVP: y'' = y² - 1")
        plt.grid(True)
        plt.show()
    else:
        print("Convergence failed")

nonlinear_bvp()
```

---

## 5. Special Problems

### 5.1 Finding Periodic Orbits

The Van der Pol oscillator $x'' - \mu(1 - x^2)x' + x = 0$ models self-sustaining oscillations: the nonlinear damping term pumps energy in when the amplitude is small and dissipates energy when it is large, driving all trajectories toward a unique **limit cycle**. Poincare sections (sampling the trajectory once per period) reveal whether the system has settled onto this periodic orbit.

```python
def find_periodic_orbit():
    """Periodic orbit of Van der Pol oscillator"""
    mu = 1.0

    def vdp(t, y):
        return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

    # Poincaré section
    def poincare(t, y):
        return y[1]  # When y' = 0

    poincare.direction = -1  # Crossing from above to below

    # Integrate for sufficiently long time
    sol = solve_ivp(vdp, (0, 100), [2, 0], events=poincare,
                    dense_output=True, max_step=0.01)

    # Crossing points after converging to limit cycle
    crossings = sol.y_events[0][-5:, 0]  # x values of last 5 crossings
    print(f"Limit cycle amplitude: {np.mean(crossings):.6f}")

    # Visualize last period
    t_last = sol.t_events[0][-2:]
    period = t_last[1] - t_last[0]
    print(f"Period: {period:.6f}")

    t_plot = np.linspace(t_last[0], t_last[1], 200)
    y_plot = sol.sol(t_plot)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(y_plot[0], y_plot[1])
    axes[0].set_xlabel('x')
    axes[0].set_ylabel("x'")
    axes[0].set_title('Limit Cycle')
    axes[0].grid(True)

    axes[1].plot(t_plot - t_last[0], y_plot[0])
    axes[1].set_xlabel('t (within period)')
    axes[1].set_ylabel('x')
    axes[1].set_title(f'One Period (T = {period:.4f})')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

find_periodic_orbit()
```

### 5.2 Bifurcation Analysis

```python
def bifurcation_analysis():
    """Change in steady state with parameter variation"""
    # dy/dt = r*y - y³
    # Equilibria: y* = 0 or y* = ±√r (r > 0)

    r_values = np.linspace(-1, 2, 100)

    plt.figure(figsize=(10, 6))

    # Stable branches
    r_pos = r_values[r_values >= 0]
    plt.plot(r_pos, np.sqrt(r_pos), 'b-', linewidth=2, label='Stable')
    plt.plot(r_pos, -np.sqrt(r_pos), 'b-', linewidth=2)

    # Unstable branches
    plt.plot(r_values[r_values <= 0], np.zeros_like(r_values[r_values <= 0]),
             'r--', linewidth=2, label='Unstable')
    plt.plot(r_values[r_values > 0], np.zeros_like(r_values[r_values > 0]),
             'r--', linewidth=2)

    plt.xlabel('r')
    plt.ylabel('y*')
    plt.title("Pitchfork Bifurcation: y' = ry - y³")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.show()

bifurcation_analysis()
```

---

## Exercises

### Problem 1
Solve the following stiff system using both BDF and Radau, and compare the results:
dy₁/dt = -1000*y₁ + y₂
dy₂/dt = 999*y₁ - 2*y₂
y₁(0) = 1, y₂(0) = 0

```python
def exercise_1():
    def system(t, y):
        return [-1000*y[0] + y[1], 999*y[0] - 2*y[1]]

    y0 = [1.0, 0.0]
    t_span = (0, 1)

    sol_bdf = solve_ivp(system, t_span, y0, method='BDF',
                        t_eval=np.linspace(0, 1, 100))
    sol_radau = solve_ivp(system, t_span, y0, method='Radau',
                          t_eval=np.linspace(0, 1, 100))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].semilogy(sol_bdf.t, np.abs(sol_bdf.y[0]), label='y₁')
    axes[0].semilogy(sol_bdf.t, np.abs(sol_bdf.y[1]), label='y₂')
    axes[0].set_title('BDF')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].semilogy(sol_radau.t, np.abs(sol_radau.y[0]), label='y₁')
    axes[1].semilogy(sol_radau.t, np.abs(sol_radau.y[1]), label='y₂')
    axes[1].set_title('Radau')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

exercise_1()
```

---

## Summary

| Problem Type | Recommended Solver | Characteristics |
|--------------|-------------------|-----------------|
| General non-stiff | RK45, DOP853 | Explicit, fast |
| Stiff problems | Radau, BDF | Implicit, stable |
| DAE | Radau + mass_matrix | Algebraic constraints |
| BVP | solve_bvp, shooting | Boundary conditions |

| Implicit Method | Order | A-stable |
|----------------|-------|----------|
| Backward Euler | 1 | O |
| Crank-Nicolson | 2 | O |
| BDF1-2 | 1-2 | O |
| BDF3-6 | 3-6 | X |
