"""
Fourier Series and PDE Solutions

Demonstrates:
  - Fourier coefficient computation (analytical and numerical)
  - Gibbs phenomenon visualization
  - Solving the heat equation via Fourier series
  - Comparing analytical Fourier solution with numerical FTCS

Dependencies: numpy, matplotlib, sympy
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# ---------------------------------------------------------------------------
# 1. Fourier Coefficient Computation
# ---------------------------------------------------------------------------
def fourier_coefficients_numerical(f, L, n_terms, n_quad=2000):
    """Compute Fourier sine series coefficients numerically.

    For f(x) on [0, L] with Dirichlet BCs (f(0) = f(L) = 0):
      f(x) = sum_{n=1}^{inf} b_n * sin(n*pi*x/L)
      b_n  = (2/L) * integral_0^L f(x) * sin(n*pi*x/L) dx

    We use the sine series because it naturally satisfies homogeneous
    Dirichlet boundary conditions — each sin(n*pi*x/L) is zero at x=0 and x=L.
    """
    x = np.linspace(0, L, n_quad)
    dx = x[1] - x[0]
    fx = f(x)

    coeffs = []
    for n in range(1, n_terms + 1):
        basis = np.sin(n * np.pi * x / L)
        bn = (2.0 / L) * np.trapz(fx * basis, x)
        coeffs.append(bn)

    return coeffs


def fourier_coefficients_symbolic(f_sym, x_sym, L, n_terms):
    """Compute Fourier sine series coefficients symbolically with SymPy."""
    coeffs = []
    for n in range(1, n_terms + 1):
        integrand = f_sym * sp.sin(n * sp.pi * x_sym / L)
        bn = (2 / L) * sp.integrate(integrand, (x_sym, 0, L))
        coeffs.append(float(bn))
    return coeffs


def fourier_partial_sum(coeffs, x, L):
    """Evaluate the Fourier sine series partial sum."""
    result = np.zeros_like(x, dtype=float)
    for n, bn in enumerate(coeffs, start=1):
        result += bn * np.sin(n * np.pi * x / L)
    return result


# ---------------------------------------------------------------------------
# 2. Gibbs Phenomenon
# ---------------------------------------------------------------------------
def plot_gibbs_phenomenon():
    """Demonstrate the Gibbs phenomenon: overshoot at discontinuities.

    The square wave has a jump discontinuity.  No matter how many Fourier
    terms we include, the partial sum overshoots by about 9% near the
    discontinuity.  This is a fundamental limitation of Fourier series
    at points of discontinuity.
    """
    L = 1.0
    # Square wave: f(x) = 1 for 0 < x < L/2, f(x) = -1 for L/2 < x < L
    # Modified to be symmetric: f(x) = 1 on (0, L) (constant)
    # Actually, let's use the classic: f(x) = x on [0, 1]
    # which has the sawtooth-like Fourier behavior.

    # Use a step function for clearest Gibbs effect
    def step_func(x):
        return np.where(x < L / 2, 1.0, 0.0)

    x = np.linspace(0, L, 1000)
    y_exact = step_func(x)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y_exact, "k--", lw=2, label="Exact (step)")

    for n_terms in [5, 15, 50, 150]:
        coeffs = fourier_coefficients_numerical(step_func, L, n_terms)
        y_fourier = fourier_partial_sum(coeffs, x, L)
        ax.plot(x, y_fourier, lw=1.5, label=f"{n_terms} terms")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Gibbs Phenomenon: Fourier Approximation of a Step Function")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("14_gibbs_phenomenon.png", dpi=100)
    plt.close()
    print("[Saved] 14_gibbs_phenomenon.png")


# ---------------------------------------------------------------------------
# 3. Heat Equation via Fourier Series
# ---------------------------------------------------------------------------
def heat_fourier_analytical(alpha, L, coeffs, x, t):
    """Analytical solution of the heat equation using Fourier series.

    u(x, t) = sum_{n=1}^N b_n * exp(-alpha*(n*pi/L)^2 * t) * sin(n*pi*x/L)

    Each Fourier mode decays exponentially with rate proportional to n^2.
    Higher modes (shorter wavelengths) decay faster — this is why the
    heat equation smooths out sharp features quickly.
    """
    result = np.zeros_like(x, dtype=float)
    for n, bn in enumerate(coeffs, start=1):
        decay = np.exp(-alpha * (n * np.pi / L) ** 2 * t)
        result += bn * decay * np.sin(n * np.pi * x / L)
    return result


def plot_heat_fourier_solution(alpha, L, u0_func, n_terms=50):
    """Solve and plot the heat equation using the Fourier series method."""
    x = np.linspace(0, L, 500)
    coeffs = fourier_coefficients_numerical(u0_func, L, n_terms)

    # Show how coefficients decay
    print(f"  First 10 Fourier coefficients:")
    for n, bn in enumerate(coeffs[:10], start=1):
        print(f"    b_{n:2d} = {bn:+.6f}")

    times = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.hot(np.linspace(0.1, 0.9, len(times)))

    for t, color in zip(times, colors):
        u = heat_fourier_analytical(alpha, L, coeffs, x, t)
        ax.plot(x, u, "-", color=color, lw=2, label=f"t = {t:.3f}")

    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"Heat Equation (Fourier Series, {n_terms} terms)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("14_heat_fourier.png", dpi=100)
    plt.close()
    print("[Saved] 14_heat_fourier.png")
    return coeffs


# ---------------------------------------------------------------------------
# 4. Comparison: Analytical vs Numerical
# ---------------------------------------------------------------------------
def heat_ftcs_solve(alpha, L, T, Nx, Nt, u0_func):
    """Minimal FTCS solver for comparison purposes."""
    dx = L / Nx
    dt = T / Nt
    r = alpha * dt / dx ** 2
    x = np.linspace(0, L, Nx + 1)
    u = u0_func(x).copy()
    u[0] = 0.0
    u[-1] = 0.0

    for _ in range(Nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u = u_new

    return x, u


def compare_analytical_numerical(alpha, L, T_compare):
    """Compare Fourier series (analytical) vs FTCS (numerical) solutions.

    The Fourier solution is "exact" (up to truncation of the series),
    while FTCS introduces discretization error.  The comparison shows
    that both methods converge to the same solution.
    """
    # Initial condition: triangular pulse
    def u0_func(x):
        return np.where(x < L / 2, 2 * x / L, 2 * (1 - x / L))

    n_terms = 100
    coeffs = fourier_coefficients_numerical(u0_func, L, n_terms)

    Nx_values = [20, 50, 100]

    fig, axes = plt.subplots(1, len(Nx_values), figsize=(15, 5))

    for ax, Nx in zip(axes, Nx_values):
        dx = L / Nx
        dt_max = 0.5 * dx ** 2 / alpha  # max stable dt
        dt = 0.8 * dt_max
        Nt = int(T_compare / dt) + 1

        # Numerical solution
        x_num, u_num = heat_ftcs_solve(alpha, L, T_compare, Nx, Nt, u0_func)

        # Analytical solution on same grid
        u_ana = heat_fourier_analytical(alpha, L, coeffs, x_num, T_compare)

        # Fine grid analytical for reference
        x_fine = np.linspace(0, L, 500)
        u_fine = heat_fourier_analytical(alpha, L, coeffs, x_fine, T_compare)

        ax.plot(x_fine, u_fine, "b-", lw=2, label="Fourier (analytical)")
        ax.plot(x_num, u_num, "r--o", ms=3, lw=1.5, label=f"FTCS (Nx={Nx})")

        error = np.max(np.abs(u_num - u_ana))
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"Nx={Nx}, max error={error:.2e}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Fourier vs FTCS at t = {T_compare}", fontsize=13)
    plt.tight_layout()
    plt.savefig("14_comparison.png", dpi=100)
    plt.close()
    print("[Saved] 14_comparison.png")


# ---------------------------------------------------------------------------
# 5. Convergence of Fourier Series for Heat Equation
# ---------------------------------------------------------------------------
def fourier_convergence():
    """Show how the number of Fourier terms affects solution accuracy.

    At t = 0, many terms are needed to capture sharp features.
    At later times, high-frequency modes have decayed, so fewer terms suffice.
    """
    L, alpha = 1.0, 1.0
    # Sharp initial condition
    u0 = lambda x: np.where((x > 0.3) & (x < 0.7), 1.0, 0.0)

    x = np.linspace(0, L, 500)
    times = [0.0, 0.005, 0.02]
    n_terms_list = [5, 20, 100]

    fig, axes = plt.subplots(len(times), 1, figsize=(9, 10))

    for ax, t in zip(axes, times):
        ax.plot(x, u0(x) if t == 0 else np.nan * x, "k--", lw=1.5,
                label="Exact IC" if t == 0 else "")

        for n_terms in n_terms_list:
            coeffs = fourier_coefficients_numerical(u0, L, n_terms)
            u = heat_fourier_analytical(alpha, L, coeffs, x, t)
            ax.plot(x, u, lw=1.5, label=f"{n_terms} terms")

        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"t = {t:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Fourier Series Convergence at Different Times", fontsize=13)
    plt.tight_layout()
    plt.savefig("14_fourier_convergence.png", dpi=100)
    plt.close()
    print("[Saved] 14_fourier_convergence.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Fourier Series and PDE Solutions")
    print("=" * 60)

    L = 1.0
    alpha = 1.0

    # --- Demo 1: Fourier coefficients ---
    print("\nDemo 1: Fourier coefficient computation")
    u0 = lambda x: np.sin(np.pi * x)
    coeffs = fourier_coefficients_numerical(u0, L, 10)
    print("  Coefficients for sin(pi*x) (expect b_1=1, rest~0):")
    for n, bn in enumerate(coeffs, start=1):
        if abs(bn) > 1e-10:
            print(f"    b_{n} = {bn:.6f}")

    # Symbolic comparison
    x_sym = sp.Symbol("x")
    coeffs_sym = fourier_coefficients_symbolic(sp.sin(sp.pi * x_sym), x_sym, 1, 5)
    print(f"  Symbolic coefficients: {coeffs_sym}")

    # --- Demo 2: Gibbs phenomenon ---
    print("\nDemo 2: Gibbs phenomenon")
    plot_gibbs_phenomenon()

    # --- Demo 3: Heat equation via Fourier ---
    print("\nDemo 3: Heat equation via Fourier series")
    # Triangular initial condition
    u0_tri = lambda x: np.where(x < L / 2, 2 * x / L, 2 * (1 - x / L))
    coeffs_heat = plot_heat_fourier_solution(alpha, L, u0_tri, n_terms=50)

    # --- Demo 4: Analytical vs numerical comparison ---
    print("\nDemo 4: Fourier (analytical) vs FTCS (numerical) comparison")
    compare_analytical_numerical(alpha, L, T_compare=0.02)

    # --- Demo 5: Convergence with number of terms ---
    print("\nDemo 5: Fourier series convergence")
    fourier_convergence()
