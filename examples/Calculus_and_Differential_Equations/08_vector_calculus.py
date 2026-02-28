"""
Vector Calculus — Fields, Line Integrals, and Integral Theorems

Demonstrates:
  - Vector field plotting (2D)
  - Line integral computation (work done by a force field)
  - Green's theorem verification
  - Divergence theorem verification (2D form)

Dependencies: numpy, matplotlib, sympy, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sympy as sp


# ---------------------------------------------------------------------------
# 1. Vector Field Visualization
# ---------------------------------------------------------------------------
def plot_vector_field(Fx_np, Fy_np, x_range=(-3, 3), y_range=(-3, 3),
                      title="Vector Field F(x,y)", filename="08_vector_field.png"):
    """Plot a 2D vector field using quiver arrows with color-coded magnitude.

    The arrow color represents |F|: brighter = stronger field.
    We use a coarser grid than the domain to keep arrows readable.
    """
    # Coarse grid for arrows
    x_q = np.linspace(*x_range, 20)
    y_q = np.linspace(*y_range, 20)
    XQ, YQ = np.meshgrid(x_q, y_q)
    U = Fx_np(XQ, YQ)
    V = Fy_np(XQ, YQ)
    magnitude = np.sqrt(U ** 2 + V ** 2)

    fig, ax = plt.subplots(figsize=(8, 7))
    quiv = ax.quiver(XQ, YQ, U, V, magnitude, cmap="plasma", alpha=0.8,
                     scale=40)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.colorbar(quiv, ax=ax, label="|F|")
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"[Saved] {filename}")


# ---------------------------------------------------------------------------
# 2. Line Integral
# ---------------------------------------------------------------------------
def line_integral_work(Fx_np, Fy_np, path_x, path_y, t_range=(0, 2 * np.pi),
                       n_points=2000):
    """Compute the line integral of F . dr along a parametric curve.

    Work = integral_C F . dr = integral_a^b [Fx * dx/dt + Fy * dy/dt] dt

    This measures the total work done by the force field F along the path.
    For a conservative field, the result depends only on the endpoints.
    """
    t = np.linspace(*t_range, n_points)
    dt = t[1] - t[0]
    x = path_x(t)
    y = path_y(t)

    # Compute dr/dt using central differences
    dx_dt = np.gradient(x, dt)
    dy_dt = np.gradient(y, dt)

    # Evaluate field along the path
    Fx_vals = Fx_np(x, y)
    Fy_vals = Fy_np(x, y)

    # Dot product and integrate
    integrand = Fx_vals * dx_dt + Fy_vals * dy_dt
    work = np.trapz(integrand, t)
    return work


# ---------------------------------------------------------------------------
# 3. Green's Theorem Verification
# ---------------------------------------------------------------------------
def verify_greens_theorem():
    """Verify Green's theorem: oint_C (P dx + Q dy) = iint_D (dQ/dx - dP/dy) dA.

    We choose P = -y, Q = x (rotation field).
    The line integral around the unit circle should equal the double
    integral of (dQ/dx - dP/dy) = 1 - (-1) = 2 over the unit disk.

    Line integral: oint_C (-y dx + x dy) = integral_0^{2pi} (sin^2 + cos^2) dt = 2 pi
    Area integral: iint_D 2 dA = 2 * pi * 1^2 = 2 pi
    """
    print("Green's Theorem Verification")
    print("  F = (-y, x),  C = unit circle (CCW)")

    # --- Line integral ---
    # Parametrize: x = cos(t), y = sin(t), t in [0, 2pi]
    P_np = lambda x, y: -y  # F_x = -y
    Q_np = lambda x, y: x   # F_y = x

    # Work integral: integral_0^{2pi} [P * dx/dt + Q * dy/dt] dt
    t = np.linspace(0, 2 * np.pi, 5000)
    x = np.cos(t)
    y = np.sin(t)
    dx_dt = -np.sin(t)
    dy_dt = np.cos(t)
    integrand = P_np(x, y) * dx_dt + Q_np(x, y) * dy_dt
    line_int = np.trapz(integrand, t)

    # --- Double integral ---
    # iint_D (dQ/dx - dP/dy) dA = iint_D (1 - (-1)) dA = 2 * area
    # For unit disk: area = pi, so double_int = 2*pi
    def integrand_area(r, theta):
        return 2.0 * r  # 2 * Jacobian(r)

    area_int, _ = integrate.dblquad(integrand_area, 0, 2 * np.pi, 0, 1)

    exact = 2 * np.pi
    print(f"  Line integral   : {line_int:.10f}")
    print(f"  Area integral   : {area_int:.10f}")
    print(f"  Exact (2*pi)    : {exact:.10f}")
    print(f"  Agreement       : {abs(line_int - area_int):.2e}")

    return line_int, area_int


# ---------------------------------------------------------------------------
# 4. Divergence Theorem Verification (2D)
# ---------------------------------------------------------------------------
def verify_divergence_theorem():
    """Verify the 2D divergence theorem: oint_C F . n ds = iint_D div(F) dA.

    We use F = (x^2, y^2).
    div(F) = 2x + 2y.
    Domain: unit disk.

    The flux through the boundary should equal the integral of the divergence.
    """
    print("\nDivergence Theorem Verification (2D)")
    print("  F = (x^2, y^2),  D = unit disk")

    # --- Flux integral (line integral of F . n around boundary) ---
    # On the unit circle: n = (cos(t), sin(t)), ds = dt
    # F . n = x^2 cos(t) + y^2 sin(t) = cos^3(t) + sin^3(t)
    t = np.linspace(0, 2 * np.pi, 5000)
    integrand_flux = np.cos(t) ** 3 + np.sin(t) ** 3
    flux_int = np.trapz(integrand_flux, t)

    # --- Divergence integral ---
    # iint_D (2x + 2y) dA  in polar: int_0^{2pi} int_0^1 (2r cos + 2r sin) r dr dtheta
    def integrand_div(r, theta):
        return (2 * r * np.cos(theta) + 2 * r * np.sin(theta)) * r

    div_int, _ = integrate.dblquad(integrand_div, 0, 2 * np.pi, 0, 1)

    print(f"  Flux integral   : {flux_int:.10f}")
    print(f"  Divergence int  : {div_int:.10f}")
    print(f"  Agreement       : {abs(flux_int - div_int):.2e}")

    return flux_int, div_int


# ---------------------------------------------------------------------------
# 5. Visualization: Path + Field
# ---------------------------------------------------------------------------
def plot_field_and_path(Fx_np, Fy_np, path_x, path_y,
                        t_range=(0, 2 * np.pi)):
    """Plot a vector field with an overlaid curve (path of integration)."""
    x_range = (-2, 2)
    y_range = (-2, 2)

    x_q = np.linspace(*x_range, 15)
    y_q = np.linspace(*y_range, 15)
    XQ, YQ = np.meshgrid(x_q, y_q)
    U = Fx_np(XQ, YQ)
    V = Fy_np(XQ, YQ)

    t = np.linspace(*t_range, 500)
    cx = path_x(t)
    cy = path_y(t)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.quiver(XQ, YQ, U, V, alpha=0.5, color="gray")
    ax.plot(cx, cy, "r-", lw=2.5, label="Path C")
    ax.plot(cx[0], cy[0], "go", ms=10, label="Start")
    ax.plot(cx[-1], cy[-1], "rs", ms=10, label="End")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Vector Field with Integration Path")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("08_field_path.png", dpi=100)
    plt.close()
    print("[Saved] 08_field_path.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Vector Calculus Demonstrations")
    print("=" * 60)

    # Define a vortex-like field: F = (-y, x) / (x^2 + y^2 + 0.1)
    Fx = lambda x, y: -y / (x ** 2 + y ** 2 + 0.1)
    Fy = lambda x, y: x / (x ** 2 + y ** 2 + 0.1)

    # --- Demo 1: Vector field plot ---
    print("\nDemo 1: Vector field visualization")
    plot_vector_field(Fx, Fy, title="Vortex Field")

    # --- Demo 2: Line integral (work) ---
    print("\nDemo 2: Line integral (work along circular path)")
    path_x = lambda t: np.cos(t)
    path_y = lambda t: np.sin(t)
    work = line_integral_work(Fx, Fy, path_x, path_y)
    print(f"  Work = integral_C F . dr = {work:.6f}")

    # For comparison: F = (-y, x) on unit circle (no denominator)
    work_simple = line_integral_work(
        lambda x, y: -y, lambda x, y: x, path_x, path_y
    )
    print(f"  Work (simple rotation F=(-y,x)): {work_simple:.6f} (expect 2*pi = {2 * np.pi:.6f})")

    # --- Demo 3: Green's theorem ---
    print()
    verify_greens_theorem()

    # --- Demo 4: Divergence theorem ---
    verify_divergence_theorem()

    # --- Demo 5: Field + path visualization ---
    print("\nDemo 5: Field with integration path")
    plot_field_and_path(Fx, Fy, path_x, path_y)
