# Introduction to Partial Differential Equations

## Learning Objectives

- Distinguish partial differential equations from ordinary differential equations and identify their order and linearity
- Classify second-order linear PDE as elliptic, parabolic, or hyperbolic using the discriminant method
- Derive the heat equation, wave equation, and Laplace equation from physical principles
- Specify appropriate boundary conditions (Dirichlet, Neumann, Robin) and initial conditions for well-posed problems
- Implement basic finite difference schemes to visualize heat diffusion and wave propagation

## Prerequisites

Before studying this lesson, you should be comfortable with:
- Multivariable calculus: partial derivatives, chain rule (Lessons 3-4)
- ODE solution techniques (Lessons 10-15)
- Basic linear algebra concepts

## What Is a Partial Differential Equation?

An **ordinary differential equation** (ODE) involves functions of a single variable and their derivatives. A **partial differential equation** (PDE) involves functions of **multiple variables** and their **partial derivatives**.

Compare:

| | ODE | PDE |
|---|-----|-----|
| Unknown | $y(t)$ | $u(x, t)$, $u(x, y)$, $u(x, y, z, t)$ |
| Derivatives | $y', y'', \ldots$ | $u_t, u_x, u_{xx}, u_{xy}, \ldots$ |
| Variables | One independent variable | Two or more independent variables |
| Solution | A curve | A surface or higher-dimensional object |

Think of it this way: an ODE describes how something changes along a single axis (like a particle moving through time). A PDE describes how something changes across space *and* time simultaneously -- like the temperature distribution in a metal rod, or the shape of a vibrating string.

### Examples from Physics

PDE are the language of continuum physics. Nearly every physical phenomenon involving spatial variation leads to a PDE:

| PDE | Physical Setting |
|-----|-----------------|
| Heat equation: $u_t = \alpha u_{xx}$ | Temperature in a rod |
| Wave equation: $u_{tt} = c^2 u_{xx}$ | Vibrating string, sound waves |
| Laplace equation: $u_{xx} + u_{yy} = 0$ | Steady-state temperature, electrostatics |
| Schrodinger equation: $i\hbar \psi_t = -\frac{\hbar^2}{2m}\nabla^2 \psi + V\psi$ | Quantum mechanics |
| Navier-Stokes: $\rho(\mathbf{v}_t + \mathbf{v} \cdot \nabla \mathbf{v}) = -\nabla p + \mu \nabla^2 \mathbf{v}$ | Fluid dynamics |
| Maxwell's equations | Electromagnetic fields |

## Notation

We use subscript notation for partial derivatives throughout:

$$u_t = \frac{\partial u}{\partial t}, \quad u_x = \frac{\partial u}{\partial x}, \quad u_{xx} = \frac{\partial^2 u}{\partial x^2}, \quad u_{xy} = \frac{\partial^2 u}{\partial x \partial y}$$

The **order** of a PDE is the order of the highest partial derivative. A PDE is **linear** if the unknown function $u$ and its derivatives appear only to the first power and are not multiplied together.

## Classification of Second-Order Linear PDE

The general second-order linear PDE in two variables is:

$$Au_{xx} + 2Bu_{xy} + Cu_{yy} + Du_x + Eu_y + Fu = G$$

where $A, B, C, D, E, F, G$ may depend on $x$ and $y$ (but not on $u$ or its derivatives for linearity).

The **discriminant** $\Delta = B^2 - AC$ determines the type:

| Discriminant | Type | Physical Prototype | Behavior |
|-------------|------|-------------------|----------|
| $B^2 - AC < 0$ | **Elliptic** | Laplace equation | Smooth, steady-state |
| $B^2 - AC = 0$ | **Parabolic** | Heat equation | Diffusive, smoothing |
| $B^2 - AC > 0$ | **Hyperbolic** | Wave equation | Propagating, preserving |

This classification is analogous to conic sections: the same discriminant $B^2 - AC$ distinguishes ellipses, parabolas, and hyperbolas. This is not a coincidence -- the classification comes from the characteristic curves of the PDE, which are conic sections.

### Why Classification Matters

Each type has fundamentally different physical behavior and requires different solution techniques and boundary conditions:

- **Elliptic**: Information propagates in all directions simultaneously. Need boundary conditions on the entire boundary (boundary value problem).
- **Parabolic**: Information propagates forward in time with infinite speed (diffusion). Need initial condition + boundary conditions (initial-boundary value problem).
- **Hyperbolic**: Information propagates at finite speed along characteristic curves. Need initial conditions on position and velocity (initial value problem).

### Classification Examples

**Heat equation**: $u_t = \alpha u_{xx}$. Rewrite as $\alpha u_{xx} - u_t = 0$. With variables $(x, t)$: $A = \alpha$, $B = 0$, $C = 0$. $\Delta = 0 - \alpha \cdot 0 = 0$. **Parabolic**.

**Wave equation**: $u_{tt} = c^2 u_{xx}$. Rewrite as $c^2 u_{xx} - u_{tt} = 0$. $A = c^2$, $B = 0$, $C = -1$. $\Delta = 0 - c^2(-1) = c^2 > 0$. **Hyperbolic**.

**Laplace equation**: $u_{xx} + u_{yy} = 0$. $A = 1$, $B = 0$, $C = 1$. $\Delta = 0 - 1 = -1 < 0$. **Elliptic**.

## The Heat Equation

### Physical Derivation

Consider a thin rod of length $L$ with insulated sides, so heat flows only along the $x$-axis.

Let $u(x, t)$ be the temperature at position $x$ and time $t$. We derive the heat equation from two physical laws:

**Conservation of energy**: The rate of change of heat energy in a small segment $[x, x + \Delta x]$ equals the net heat flux into the segment:

$$\rho c \frac{\partial u}{\partial t} \Delta x = q(x, t) - q(x + \Delta x, t)$$

where $\rho$ is density, $c$ is specific heat, and $q$ is heat flux (energy per unit time per unit area).

**Fourier's law of heat conduction**: Heat flows from hot to cold proportionally to the temperature gradient:

$$q = -\kappa \frac{\partial u}{\partial x}$$

where $\kappa > 0$ is the thermal conductivity. The negative sign ensures heat flows in the direction of decreasing temperature.

Combining these and taking $\Delta x \to 0$:

$$\rho c \frac{\partial u}{\partial t} = \kappa \frac{\partial^2 u}{\partial x^2}$$

Defining the **thermal diffusivity** $\alpha = \kappa / (\rho c)$:

$$\boxed{u_t = \alpha u_{xx}}$$

**Intuition**: The temperature at a point changes at a rate proportional to the **curvature** of the temperature profile. If the temperature profile is concave up ($u_{xx} > 0$), the point is cooler than its neighbors, so it heats up. If concave down, it cools down. This is the mathematical expression of "heat flows from hot to cold."

## The Wave Equation

### Physical Derivation

Consider a vibrating string of length $L$ under tension $T$, with mass per unit length $\mu$. Let $u(x, t)$ be the vertical displacement.

For a small segment of string, Newton's second law gives:

$$\mu \Delta x \frac{\partial^2 u}{\partial t^2} = T \sin\theta_2 - T \sin\theta_1$$

For small displacements, $\sin\theta \approx \tan\theta = \partial u / \partial x$, so:

$$\mu \frac{\partial^2 u}{\partial t^2} = T \frac{\partial^2 u}{\partial x^2}$$

Defining the **wave speed** $c = \sqrt{T/\mu}$:

$$\boxed{u_{tt} = c^2 u_{xx}}$$

**Intuition**: Acceleration at each point is proportional to the curvature of the string. A point below the line connecting its neighbors gets pulled up (positive curvature, positive acceleration). Unlike the heat equation, the wave equation preserves energy -- waves propagate without dissipation.

## The Laplace Equation

$$\boxed{u_{xx} + u_{yy} = 0} \qquad \text{(2D Laplace equation)}$$

or in 3D: $u_{xx} + u_{yy} + u_{zz} = 0$, often written as $\nabla^2 u = 0$.

The Laplace equation describes **steady-state** situations: the temperature distribution in a plate after it has reached equilibrium, the electric potential in a charge-free region, or the gravitational potential in empty space.

Functions satisfying the Laplace equation are called **harmonic functions**. They have the remarkable **mean value property**: the value at any point equals the average over any surrounding sphere (or circle in 2D).

## Boundary and Initial Conditions

A PDE alone has infinitely many solutions. We need additional conditions to select a unique, physically meaningful solution.

### Boundary Conditions (BCs)

Applied at the spatial boundaries of the domain:

| Type | Name | Form | Physical Meaning |
|------|------|------|-----------------|
| First kind | **Dirichlet** | $u = g$ on boundary | Prescribed temperature/displacement |
| Second kind | **Neumann** | $\frac{\partial u}{\partial n} = h$ on boundary | Prescribed heat flux (insulation if $h=0$) |
| Third kind | **Robin** | $\alpha u + \beta \frac{\partial u}{\partial n} = g$ | Convective cooling (Newton's law of cooling) |

where $\partial u / \partial n$ is the outward normal derivative at the boundary.

**Example**: A metal rod with the left end held at $100°$C and the right end insulated:
- $u(0, t) = 100$ (Dirichlet at $x = 0$)
- $u_x(L, t) = 0$ (Neumann at $x = L$; zero flux means insulated)

### Initial Conditions (ICs)

Applied at $t = 0$:

- **Heat equation** (first order in $t$): needs $u(x, 0) = f(x)$ (initial temperature)
- **Wave equation** (second order in $t$): needs both $u(x, 0) = f(x)$ (initial displacement) and $u_t(x, 0) = g(x)$ (initial velocity)

## Well-Posedness (Hadamard Conditions)

A PDE problem is **well-posed** in the sense of Hadamard if:

1. **Existence**: A solution exists
2. **Uniqueness**: The solution is unique
3. **Continuous dependence on data**: Small changes in the data produce small changes in the solution

If any condition fails, the problem is **ill-posed**. For instance, specifying too few boundary conditions may give non-uniqueness; specifying too many may give no solution; the backward heat equation is ill-posed because tiny perturbations in data lead to enormous changes in the solution.

## Python Implementation

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

## Summary

| Concept | Key Point |
|---------|-----------|
| PDE vs ODE | PDE involve multiple independent variables |
| Classification | Discriminant $B^2 - AC$: elliptic (<0), parabolic (=0), hyperbolic (>0) |
| Heat equation | $u_t = \alpha u_{xx}$; diffusive, smoothing behavior |
| Wave equation | $u_{tt} = c^2 u_{xx}$; propagating, energy-preserving |
| Laplace equation | $\nabla^2 u = 0$; steady-state, mean value property |
| Boundary conditions | Dirichlet (value), Neumann (flux), Robin (mixed) |
| Well-posedness | Existence + uniqueness + continuous dependence on data |

The three prototype equations -- heat, wave, and Laplace -- form the foundation of PDE theory. In the next lesson, we will solve them analytically using Fourier series. For comprehensive numerical methods, see [Numerical Simulation - Finite Difference Methods](../Numerical_Simulation/06_Finite_Difference_PDE.md) and subsequent lessons. For the mathematical framework of separation of variables, see [Mathematical Methods - PDE and Separation of Variables](../Mathematical_Methods/13_PDE_Separation_of_Variables.md).

## Practice Problems

1. **Classification**: Classify each PDE as elliptic, parabolic, or hyperbolic:
   - (a) $u_{xx} + 4u_{xy} + 4u_{yy} = 0$
   - (b) $u_{xx} - 4u_{xy} + 4u_{yy} = 0$
   - (c) $y \cdot u_{xx} + u_{yy} = 0$ (discuss how the type depends on the sign of $y$)

2. **Derivation**: Derive the 1D heat equation with a source term. If the rod has an internal heat source of $q(x, t)$ watts per unit volume, show that the equation becomes $u_t = \alpha u_{xx} + \frac{q}{\rho c}$. What physical scenario would produce $q = q_0 \sin(\pi x / L)$?

3. **Boundary conditions**: A rod of length $L = 1$ has its left end at $0°$C and its right end losing heat to a surrounding medium at $20°$C via Newton's law of cooling ($-\kappa u_x(L,t) = h(u(L,t) - 20)$). Write the boundary conditions formally, identifying them as Dirichlet, Neumann, or Robin.

4. **Well-posedness**: Explain why the backward heat equation $u_t = -\alpha u_{xx}$ (note the minus sign) is ill-posed for forward-in-time evolution. Hint: consider the Fourier mode $u(x,t) = e^{\alpha k^2 t} \sin(kx)$ and what happens to high-frequency perturbations.

5. **Numerical experiment**: Modify the heat equation code to use $r = 0.6$ (violating the CFL condition). Run the simulation and describe what happens. Then implement the implicit backward Euler scheme (which is unconditionally stable) and verify that it works with $r = 0.6$.

---

*Previous: [Power Series Solutions](./16_Power_Series_Solutions.md) | Next: [Fourier Series and PDE](./18_Fourier_Series_and_PDE.md)*
