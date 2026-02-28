"""
Exercises for Lesson 03: ODE Basics

Topics: ODE classification, separation of variables, characteristic equations,
        stability analysis of equilibria, system conversion.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Exercise 1: Classify and Solve a First-Order Separable ODE
# ---------------------------------------------------------------------------
# Classify dy/dt = -2ty^2 (order, linearity, autonomy). Find the general
# solution by separation of variables. Apply y(0) = 1/3 and verify
# numerically.
# ---------------------------------------------------------------------------

def exercise_1():
    """Classify and solve a first-order separable ODE."""
    print("Classification of dy/dt = -2ty^2:")
    print("  Order:    1st (highest derivative is dy/dt)")
    print("  Linear:   Nonlinear (contains y^2)")
    print("  Autonomy: Non-autonomous (t appears explicitly)")
    print()
    print("Separation of variables:")
    print("  dy/y^2 = -2t dt")
    print("  -1/y = -t^2 + C")
    print("  y = 1/(t^2 + K)")
    print("  y(0) = 1/3 => K = 3")
    print("  Particular solution: y(t) = 1/(t^2 + 3)")

    # Numerical verification
    t_vals = np.linspace(0, 3, 200)
    y_exact = 1 / (t_vals**2 + 3)

    def ode(t, y):
        return [-2 * t * y[0]**2]

    sol = solve_ivp(ode, (0, 3), [1 / 3], t_eval=t_vals, rtol=1e-10)

    plt.figure(figsize=(8, 4))
    plt.plot(t_vals, y_exact, 'b-', linewidth=2, label='Analytical: 1/(t^2+3)')
    plt.plot(sol.t, sol.y[0], 'r--', linewidth=1.5, label='Numerical (RK45)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('dy/dt = -2ty^2, y(0) = 1/3')
    plt.legend()
    plt.grid(True)
    plt.savefig('ex03_separable_ode.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to ex03_separable_ode.png")

    # Verify residual
    dy_dt = np.gradient(y_exact, t_vals)
    rhs = -2 * t_vals * y_exact**2
    print(f"Max residual |dy/dt - f(t,y)|: {np.max(np.abs(dy_dt - rhs)):.2e}")


# ---------------------------------------------------------------------------
# Exercise 2: Characteristic Equation and Damped Oscillation
# ---------------------------------------------------------------------------
# Solve y'' + 4y' + 5y = 0 with y(0)=2, y'(0)=0.
# Find characteristic roots, general solution, and plot.
# ---------------------------------------------------------------------------

def exercise_2():
    """Characteristic equation and damped oscillation."""
    print("Characteristic equation: r^2 + 4r + 5 = 0")
    print("Roots: r = (-4 +/- sqrt(16-20))/2 = -2 +/- i")
    print()
    print("General solution: y(t) = e^(-2t)(C1 cos(t) + C2 sin(t))")
    print("y(0) = C1 = 2")
    print("y'(0) = -2*C1 + C2 = 0 => C2 = 4")
    print("Particular solution: y(t) = e^(-2t)(2cos(t) + 4sin(t))")

    t = np.linspace(0, 5, 300)
    C1, C2 = 2, 4
    y_exact = np.exp(-2 * t) * (C1 * np.cos(t) + C2 * np.sin(t))
    envelope = np.sqrt(C1**2 + C2**2) * np.exp(-2 * t)

    # Numerical check
    def ode(t, y):
        return [y[1], -4 * y[1] - 5 * y[0]]

    sol = solve_ivp(ode, (0, 5), [2, 0], t_eval=t, rtol=1e-10)

    plt.figure(figsize=(9, 4))
    plt.plot(t, y_exact, 'b-', label='Analytical')
    plt.plot(sol.t, sol.y[0], 'r--', alpha=0.7, label='Numerical')
    plt.plot(t, envelope, 'g:', label='Envelope')
    plt.plot(t, -envelope, 'g:')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("y'' + 4y' + 5y = 0,  alpha=-2, omega_d=1 rad/s")
    plt.legend()
    plt.grid(True)
    plt.savefig('ex03_damped_oscillation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to ex03_damped_oscillation.png")
    print(f"Decay rate alpha = 2,  Damped frequency omega_d = 1 rad/s")
    print(f"Max error vs numerical: {np.max(np.abs(y_exact - sol.y[0])):.2e}")


# ---------------------------------------------------------------------------
# Exercise 3: Stability of Equilibrium Points in a Nonlinear ODE
# ---------------------------------------------------------------------------
# For dy/dt = y^3 - y = y(y-1)(y+1):
# 1. Find all equilibrium points.
# 2. Determine stability via f'(y*).
# 3. Simulate and confirm.
# ---------------------------------------------------------------------------

def exercise_3():
    """Stability analysis of equilibrium points."""
    print("Equilibrium points: y^3 - y = y(y-1)(y+1) = 0")
    print("  y* = -1, 0, +1")
    print()
    print("Linear stability: f'(y) = 3y^2 - 1")
    print("  y*=-1: f'(-1) = 2 > 0  => Unstable")
    print("  y*= 0: f'(0)  = -1 < 0 => Stable")
    print("  y*=+1: f'(+1) = 2 > 0  => Unstable")

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
        axes[0].scatter([y_eq], [0], s=100, color=color, zorder=5,
                        label=f'y*={y_eq} ({label})')
    axes[0].set_xlabel('y')
    axes[0].set_ylabel("dy/dt = y^3 - y")
    axes[0].set_title("Phase Portrait")
    axes[0].legend()
    axes[0].grid(True)

    # Trajectories
    for y0 in [-1.5, -0.5, 0.5, 1.5]:
        sol = solve_ivp(lambda t, y: [f(y[0])], t_span, [y0], t_eval=t_eval)
        axes[1].plot(sol.t, sol.y[0], label=f'y0={y0}')
    for y_eq, stable in [(-1, False), (0, True), (1, False)]:
        ls = '--' if stable else ':'
        axes[1].axhline(y_eq, color='gray', linestyle=ls, alpha=0.6)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('y(t)')
    axes[1].set_title('Solution Trajectories')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('ex03_equilibrium_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to ex03_equilibrium_stability.png")


# ---------------------------------------------------------------------------
# Exercise 4: Converting a 3rd-Order ODE to a First-Order System
# ---------------------------------------------------------------------------
# Convert y''' - y'' + 4y' - 4y = 0 to a system of three first-order ODEs.
# Write in matrix form, find characteristic equation, solve numerically
# with y(0)=1, y'(0)=0, y''(0)=-1.
# ---------------------------------------------------------------------------

def exercise_4():
    """Convert 3rd-order ODE to first-order system and solve."""
    print("State vector: u1=y, u2=y', u3=y''")
    print("System:")
    print("  u1' = u2")
    print("  u2' = u3")
    print("  u3' = 4*u1 - 4*u2 + u3")
    print()
    print("Characteristic equation: r^3 - r^2 + 4r - 4 = 0")
    print("  (r-1)(r^2+4) = 0")
    print("  Roots: r = 1, +/-2i")
    print("  General solution: y = C1*e^t + C2*cos(2t) + C3*sin(2t)")

    def system(t, u):
        u1, u2, u3 = u
        return [u2, u3, 4 * u1 - 4 * u2 + u3]

    u0 = [1.0, 0.0, -1.0]
    t_span = (0, 4)
    t_eval = np.linspace(0, 4, 400)

    sol = solve_ivp(system, t_span, u0, t_eval=t_eval, method='RK45', rtol=1e-10)

    # Analytical: C1=0.6, C2=0.4, C3=-0.3
    C1, C2, C3 = 0.6, 0.4, -0.3
    y_analytic = C1 * np.exp(t_eval) + C2 * np.cos(2 * t_eval) + C3 * np.sin(2 * t_eval)

    plt.figure(figsize=(9, 4))
    plt.plot(t_eval, sol.y[0], 'b-', linewidth=2, label='Numerical y(t)')
    plt.plot(t_eval, y_analytic, 'r--', linewidth=1.5, label='Analytical y(t)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("y''' - y'' + 4y' - 4y = 0")
    plt.legend()
    plt.grid(True)
    plt.savefig('ex03_3rd_order_system.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to ex03_3rd_order_system.png")
    print(f"Max error: {np.max(np.abs(sol.y[0] - y_analytic)):.2e}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 1: Classify and Solve a Separable ODE")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("Exercise 2: Characteristic Equation and Damped Oscillation")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("Exercise 3: Stability of Equilibrium Points")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("Exercise 4: Converting 3rd-Order ODE to System")
    print("=" * 60)
    exercise_4()
