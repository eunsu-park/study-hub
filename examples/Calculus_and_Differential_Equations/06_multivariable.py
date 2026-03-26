"""
Multivariable Calculus Visualization

Demonstrates:
  - 3D surface plotting
  - Contour plots
  - Gradient vector field visualization
  - Directional derivative computation
  - Tangent plane at a point

Dependencies: numpy, matplotlib, sympy
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
import sympy as sp


# ---------------------------------------------------------------------------
# 1. 3D Surface and Contour Plots
# ---------------------------------------------------------------------------
def plot_surface_and_contour(f_np, x_range=(-3, 3), y_range=(-3, 3),
                             title="f(x,y)"):
    """Create side-by-side 3D surface and 2D contour plots.

    The contour plot is the "top view" of the surface — level curves
    where f(x,y) = constant.  These are essential for understanding
    optimization landscapes.
    """
    x = np.linspace(*x_range, 200)
    y = np.linspace(*y_range, 200)
    X, Y = np.meshgrid(x, y)
    Z = f_np(X, Y)

    fig = plt.figure(figsize=(14, 5))

    # 3D surface
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85,
                            edgecolor="none")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x,y)")
    ax1.set_title(f"Surface: {title}")
    fig.colorbar(surf, ax=ax1, shrink=0.5, pad=0.1)

    # Contour plot
    ax2 = fig.add_subplot(122)
    cs = ax2.contourf(X, Y, Z, levels=30, cmap="viridis")
    ax2.contour(X, Y, Z, levels=15, colors="k", linewidths=0.5, alpha=0.4)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(f"Contour: {title}")
    ax2.set_aspect("equal")
    fig.colorbar(cs, ax=ax2)

    plt.tight_layout()
    plt.savefig("06_surface_contour.png", dpi=100)
    plt.close()
    print("[Saved] 06_surface_contour.png")


# ---------------------------------------------------------------------------
# 2. Gradient Vector Field
# ---------------------------------------------------------------------------
def compute_gradient_symbolic(f_sym, x_sym, y_sym):
    """Compute the gradient symbolically: grad f = (df/dx, df/dy).

    The gradient points in the direction of steepest ascent and its
    magnitude equals the maximum rate of change of f at that point.
    """
    df_dx = sp.diff(f_sym, x_sym)
    df_dy = sp.diff(f_sym, y_sym)
    return df_dx, df_dy


def plot_gradient_field(f_np, grad_x_np, grad_y_np,
                        x_range=(-3, 3), y_range=(-3, 3)):
    """Overlay the gradient vector field on a contour plot.

    Note how the gradient arrows are perpendicular to the contour lines
    everywhere — this is a fundamental property of the gradient.
    """
    x = np.linspace(*x_range, 200)
    y = np.linspace(*y_range, 200)
    X, Y = np.meshgrid(x, y)
    Z = f_np(X, Y)

    # Coarser grid for arrows (dense arrows are unreadable)
    x_q = np.linspace(*x_range, 20)
    y_q = np.linspace(*y_range, 20)
    XQ, YQ = np.meshgrid(x_q, y_q)
    U = grad_x_np(XQ, YQ)
    V = grad_y_np(XQ, YQ)

    fig, ax = plt.subplots(figsize=(8, 7))
    cs = ax.contourf(X, Y, Z, levels=30, cmap="coolwarm", alpha=0.7)
    ax.contour(X, Y, Z, levels=15, colors="k", linewidths=0.3, alpha=0.5)
    ax.quiver(XQ, YQ, U, V, color="black", alpha=0.7, scale=80)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Gradient Field (arrows) on Contour Plot")
    ax.set_aspect("equal")
    fig.colorbar(cs, ax=ax)
    plt.tight_layout()
    plt.savefig("06_gradient_field.png", dpi=100)
    plt.close()
    print("[Saved] 06_gradient_field.png")


# ---------------------------------------------------------------------------
# 3. Directional Derivative
# ---------------------------------------------------------------------------
def directional_derivative(grad_x_np, grad_y_np, x0, y0, direction):
    """Compute the directional derivative D_u f at (x0, y0).

    D_u f = grad f . u_hat  (dot product with unit direction vector).
    This tells us the rate of change of f in the specified direction.
    """
    # Normalize the direction to a unit vector
    u = np.array(direction, dtype=float)
    u_hat = u / np.linalg.norm(u)

    grad = np.array([grad_x_np(x0, y0), grad_y_np(x0, y0)])
    return float(np.dot(grad, u_hat)), u_hat, grad


# ---------------------------------------------------------------------------
# 4. Tangent Plane
# ---------------------------------------------------------------------------
def plot_tangent_plane(f_np, grad_x_np, grad_y_np, x0, y0):
    """Plot the surface and its tangent plane at (x0, y0).

    Tangent plane equation:
      z = f(x0,y0) + f_x(x0,y0)(x-x0) + f_y(x0,y0)(y-y0)

    The tangent plane is the best linear approximation to the surface
    near the point of tangency.
    """
    x = np.linspace(x0 - 2, x0 + 2, 100)
    y = np.linspace(y0 - 2, y0 + 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = f_np(X, Y)

    z0 = f_np(x0, y0)
    fx0 = grad_x_np(x0, y0)
    fy0 = grad_y_np(x0, y0)
    Z_plane = z0 + fx0 * (X - x0) + fy0 * (Y - y0)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6, edgecolor="none")
    ax.plot_surface(X, Y, Z_plane, color="orange", alpha=0.4,
                    edgecolor="none")
    ax.scatter([x0], [y0], [z0], color="red", s=80, zorder=5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Tangent Plane at ({x0}, {y0})")
    plt.tight_layout()
    plt.savefig("06_tangent_plane.png", dpi=100)
    plt.close()
    print("[Saved] 06_tangent_plane.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Function: f(x,y) = sin(x) * cos(y) — a smooth, periodic landscape
    x_sym, y_sym = sp.symbols("x y")
    f_sym = sp.sin(x_sym) * sp.cos(y_sym)
    f_np = lambda x, y: np.sin(x) * np.cos(y)

    # --- Demo 1: Surface and contour ---
    print("=" * 60)
    print("Demo 1: 3D Surface and Contour Plots")
    print("=" * 60)
    plot_surface_and_contour(f_np, title="sin(x) cos(y)")

    # --- Demo 2: Gradient field ---
    print("\nDemo 2: Gradient vector field")
    gx_sym, gy_sym = compute_gradient_symbolic(f_sym, x_sym, y_sym)
    print(f"  grad f = ({gx_sym}, {gy_sym})")

    gx_np = sp.lambdify((x_sym, y_sym), gx_sym, "numpy")
    gy_np = sp.lambdify((x_sym, y_sym), gy_sym, "numpy")
    plot_gradient_field(f_np, gx_np, gy_np)

    # --- Demo 3: Directional derivative ---
    print("\nDemo 3: Directional derivative at (pi/4, pi/4)")
    x0, y0 = np.pi / 4, np.pi / 4
    direction = (1, 1)  # direction toward (1,1)
    dd, u_hat, grad = directional_derivative(gx_np, gy_np, x0, y0, direction)
    print(f"  Point     : ({x0:.4f}, {y0:.4f})")
    print(f"  Direction : {direction}  (unit: ({u_hat[0]:.4f}, {u_hat[1]:.4f}))")
    print(f"  Gradient  : ({grad[0]:.4f}, {grad[1]:.4f})")
    print(f"  |grad f|  : {np.linalg.norm(grad):.4f} (max rate of change)")
    print(f"  D_u f     : {dd:.4f}")

    # --- Demo 4: Tangent plane ---
    print("\nDemo 4: Tangent plane visualization")
    plot_tangent_plane(f_np, gx_np, gy_np, x0=1.0, y0=0.5)
