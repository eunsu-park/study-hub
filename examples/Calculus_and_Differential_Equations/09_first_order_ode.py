"""
First-Order Ordinary Differential Equations

Demonstrates:
  - Direction (slope) field plotting
  - Euler's method with error analysis
  - Separable ODE analytical solution
  - Integrating factor method
  - Mixing problem physical simulation

Dependencies: numpy, matplotlib, sympy, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# 1. Direction Field (Slope Field)
# ---------------------------------------------------------------------------
def plot_direction_field(f, x_range=(-3, 3), y_range=(-3, 3), n_grid=20,
                         solutions=None, title="Direction Field"):
    """Plot a slope field for dy/dx = f(x, y).

    At each grid point (x, y) we draw a small line segment with slope
    f(x, y).  This gives a visual guide for how solutions "flow" —
    any solution curve must be tangent to the nearby segments.
    """
    x = np.linspace(*x_range, n_grid)
    y = np.linspace(*y_range, n_grid)
    X, Y = np.meshgrid(x, y)
    slopes = f(X, Y)

    # Normalize arrows to unit length so the field is easy to read
    # dx = 1, dy = slope  =>  length = sqrt(1 + slope^2)
    norm = np.sqrt(1 + slopes ** 2)
    U = 1.0 / norm
    V = slopes / norm

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(X, Y, U, V, angles="xy", alpha=0.5, color="gray",
              headwidth=0, headlength=0, headaxislength=0, pivot="middle")

    # Overlay particular solutions if provided
    if solutions is not None:
        colors = plt.cm.Set1(np.linspace(0, 0.8, len(solutions)))
        for (t_sol, y_sol, label), color in zip(solutions, colors):
            ax.plot(t_sol, y_sol, "-", color=color, lw=2, label=label)

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel("x (or t)")
    ax.set_ylabel("y")
    ax.set_title(title)
    if solutions:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("09_direction_field.png", dpi=100)
    plt.close()
    print("[Saved] 09_direction_field.png")


# ---------------------------------------------------------------------------
# 2. Euler's Method
# ---------------------------------------------------------------------------
def euler_method(f, t0, y0, t_end, h):
    """Forward Euler method: y_{n+1} = y_n + h * f(t_n, y_n).

    First-order accurate (global error O(h)).  Simple but requires very
    small step sizes for stiff problems.  We track every step for
    visualization and error analysis.
    """
    t_vals = [t0]
    y_vals = [y0]
    t, y = t0, y0

    while t < t_end - 1e-12:
        y = y + h * f(t, y)
        t = t + h
        t_vals.append(t)
        y_vals.append(y)

    return np.array(t_vals), np.array(y_vals)


def euler_error_analysis():
    """Compare Euler's method at different step sizes against the exact solution.

    ODE: dy/dt = -2y,  y(0) = 1  =>  y(t) = exp(-2t).
    We expect the global error to halve when h halves (first order).
    """
    f = lambda t, y: -2 * y
    exact = lambda t: np.exp(-2 * t)
    t_end = 3.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    step_sizes = [0.5, 0.2, 0.1, 0.05, 0.01]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(step_sizes)))
    final_errors = []

    t_fine = np.linspace(0, t_end, 500)
    axes[0].plot(t_fine, exact(t_fine), "k-", lw=2, label="Exact")

    for h, color in zip(step_sizes, colors):
        t_e, y_e = euler_method(f, 0, 1, t_end, h)
        axes[0].plot(t_e, y_e, "o-", color=color, ms=2, lw=1,
                     label=f"h = {h}")
        final_errors.append(abs(y_e[-1] - exact(t_e[-1])))

    axes[0].set_xlabel("t")
    axes[0].set_ylabel("y")
    axes[0].set_title("Euler's Method: dy/dt = -2y")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Error vs step size (should be roughly linear on log-log)
    axes[1].loglog(step_sizes, final_errors, "ro-", ms=8)
    axes[1].loglog(step_sizes, step_sizes, "k--", alpha=0.4, label="O(h)")
    axes[1].set_xlabel("Step size h")
    axes[1].set_ylabel("Final error |y(3) - y_euler(3)|")
    axes[1].set_title("Euler Global Error (first order)")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("09_euler_error.png", dpi=100)
    plt.close()
    print("[Saved] 09_euler_error.png")
    return final_errors


# ---------------------------------------------------------------------------
# 3. Separable ODE (Symbolic Solution)
# ---------------------------------------------------------------------------
def solve_separable_ode():
    """Solve a separable ODE symbolically: dy/dx = y * cos(x).

    Separable means we can write g(y) dy = h(x) dx and integrate both sides.
    Here: dy/y = cos(x) dx  =>  ln|y| = sin(x) + C  =>  y = A * exp(sin(x)).
    """
    x = sp.Symbol("x")
    y = sp.Function("y")
    ode = sp.Eq(y(x).diff(x), y(x) * sp.cos(x))
    solution = sp.dsolve(ode, y(x))

    print("Separable ODE: dy/dx = y * cos(x)")
    print(f"  General solution: {solution}")

    # Particular solution with y(0) = 2
    C1 = sp.Symbol("C1")
    particular = solution.subs(C1, 2)  # C1 * exp(sin(x)), need C1 s.t. y(0)=2
    ics = {y(0): 2}
    sol_ivp = sp.dsolve(ode, y(x), ics=ics)
    print(f"  IVP y(0) = 2   : {sol_ivp}")
    return sol_ivp


# ---------------------------------------------------------------------------
# 4. Integrating Factor Method
# ---------------------------------------------------------------------------
def integrating_factor_demo():
    """Solve a linear first-order ODE using the integrating factor.

    ODE: dy/dx + 2y = 4x   (standard form: y' + P(x)y = Q(x))

    Integrating factor: mu(x) = exp(integral P(x) dx) = exp(2x).
    Multiply through: d/dx [mu * y] = mu * Q  =>  integrate both sides.
    """
    x = sp.Symbol("x")
    y = sp.Function("y")
    ode = sp.Eq(y(x).diff(x) + 2 * y(x), 4 * x)

    # Step-by-step manual approach
    P = 2
    mu = sp.exp(sp.integrate(P, x))  # = exp(2x)
    print("\nIntegrating Factor Method: dy/dx + 2y = 4x")
    print(f"  P(x) = {P}")
    print(f"  Integrating factor mu(x) = {mu}")

    # SymPy's general solver
    general = sp.dsolve(ode, y(x))
    print(f"  General solution: {general}")

    # IVP: y(0) = 1
    particular = sp.dsolve(ode, y(x), ics={y(0): 1})
    print(f"  IVP y(0) = 1   : {particular}")
    return particular


# ---------------------------------------------------------------------------
# 5. Mixing Problem Simulation
# ---------------------------------------------------------------------------
def mixing_problem():
    """Simulate a classic mixing tank problem.

    A 100-gallon tank initially holds 50 lb of dissolved salt.
    Brine with 2 lb/gal enters at 3 gal/min.
    Well-mixed solution drains at 3 gal/min (volume stays constant).

    ODE: dA/dt = rate_in - rate_out = 6 - 3A/100,  A(0) = 50.
    Exact: A(t) = 200 - 150 * exp(-3t/100).
    """
    # Parameters
    V = 100.0       # tank volume (gallons)
    c_in = 2.0      # inlet concentration (lb/gal)
    r_in = 3.0      # inlet flow rate (gal/min)
    r_out = 3.0     # outlet flow rate (gal/min)
    A0 = 50.0       # initial salt (lb)

    rate_in = c_in * r_in                 # 6 lb/min
    # rate_out = (A/V) * r_out            # concentration * flow

    f = lambda t, A: rate_in - (r_out / V) * A
    exact = lambda t: 200 - 150 * np.exp(-3 * t / 100)

    t_end = 200  # minutes

    # Solve with scipy (high accuracy reference)
    sol = solve_ivp(f, [0, t_end], [A0], dense_output=True,
                    max_step=0.5)

    # Euler for comparison
    t_euler, A_euler = euler_method(f, 0, A0, t_end, h=2.0)

    t_fine = np.linspace(0, t_end, 500)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_fine, exact(t_fine), "k-", lw=2, label="Exact")
    ax.plot(t_fine, sol.sol(t_fine)[0], "b--", lw=1.5, label="RK45 (scipy)")
    ax.plot(t_euler, A_euler, "r.", ms=3, label="Euler (h=2)")

    ax.axhline(200, color="gray", ls=":", alpha=0.5, label="Steady state = 200 lb")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Salt in tank (lb)")
    ax.set_title("Mixing Problem: dA/dt = 6 - 0.03A")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("09_mixing_problem.png", dpi=100)
    plt.close()
    print("[Saved] 09_mixing_problem.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("First-Order ODE Demonstrations")
    print("=" * 60)

    # --- Demo 1: Direction field for dy/dx = x - y ---
    print("\nDemo 1: Direction field for dy/dx = x - y")
    f_slope = lambda x, y: x - y

    # Generate a few solution curves using scipy
    solutions = []
    for y0 in [-2, 0, 1, 3]:
        sol = solve_ivp(lambda t, y: t - y, [0, 3], [y0],
                        dense_output=True, t_eval=np.linspace(0, 3, 200))
        solutions.append((sol.t, sol.y[0], f"y(0) = {y0}"))

    plot_direction_field(f_slope, x_range=(0, 3), y_range=(-3, 4),
                         solutions=solutions, title="dy/dx = x - y")

    # --- Demo 2: Euler's method error analysis ---
    print("\nDemo 2: Euler's method error analysis")
    errors = euler_error_analysis()

    # --- Demo 3: Separable ODE ---
    print("\nDemo 3: Separable ODE solution")
    solve_separable_ode()

    # --- Demo 4: Integrating factor ---
    integrating_factor_demo()

    # --- Demo 5: Mixing problem ---
    print("\nDemo 5: Mixing tank simulation")
    mixing_problem()
