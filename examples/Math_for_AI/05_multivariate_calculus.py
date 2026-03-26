"""
Multivariate Calculus for AI/ML

Demonstrates:
- Partial derivatives with symbolic and numerical computation
- Gradient vectors and gradient fields
- Directional derivatives
- Second-order Taylor expansion around a point
- Visualization of gradient fields and level curves

Dependencies: numpy, sympy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def partial_derivatives_symbolic():
    """Compute partial derivatives symbolically with SymPy"""
    print("=" * 60)
    print("PARTIAL DERIVATIVES (SYMBOLIC)")
    print("=" * 60)

    # Define symbolic variables
    x, y = sp.symbols('x y')

    # Example 1: f(x, y) = x^2 * y + sin(x) * e^y
    f = x**2 * y + sp.sin(x) * sp.exp(y)
    print(f"\nf(x, y) = {f}")

    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)
    print(f"\n∂f/∂x = {df_dx}")
    print(f"∂f/∂y = {df_dy}")

    # Evaluate at a point
    point = {x: 1.0, y: 0.0}
    print(f"\nAt (x, y) = (1, 0):")
    print(f"  ∂f/∂x = {float(df_dx.subs(point)):.4f}")
    print(f"  ∂f/∂y = {float(df_dy.subs(point)):.4f}")

    # Second-order partial derivatives
    print("\n--- Second-Order Partial Derivatives ---")
    d2f_dx2 = sp.diff(f, x, 2)
    d2f_dy2 = sp.diff(f, y, 2)
    d2f_dxdy = sp.diff(f, x, y)

    print(f"∂²f/∂x² = {d2f_dx2}")
    print(f"∂²f/∂y² = {d2f_dy2}")
    print(f"∂²f/∂x∂y = {d2f_dxdy}")
    print(f"\nClairaut's theorem (symmetry): ∂²f/∂x∂y == ∂²f/∂y∂x")
    d2f_dydx = sp.diff(f, y, x)
    print(f"Equal: {d2f_dxdy == d2f_dydx}")

    # Example 2: Chain rule — g(t) = f(cos(t), sin(t))
    print("\n--- Chain Rule ---")
    t = sp.Symbol('t')
    x_t = sp.cos(t)
    y_t = sp.sin(t)
    g = f.subs({x: x_t, y: y_t})
    dg_dt = sp.diff(g, t)
    dg_dt_simplified = sp.simplify(dg_dt)
    print(f"g(t) = f(cos(t), sin(t))")
    print(f"dg/dt = {dg_dt_simplified}")

    # Verify via chain rule: dg/dt = ∂f/∂x * dx/dt + ∂f/∂y * dy/dt
    chain_rule = (df_dx.subs({x: x_t, y: y_t}) * sp.diff(x_t, t) +
                  df_dy.subs({x: x_t, y: y_t}) * sp.diff(y_t, t))
    chain_rule_simplified = sp.simplify(chain_rule)
    print(f"Chain rule result: {chain_rule_simplified}")
    print(f"Match: {sp.simplify(dg_dt_simplified - chain_rule_simplified) == 0}")


def gradient_vector():
    """Demonstrate gradient vectors and their properties"""
    print("\n" + "=" * 60)
    print("GRADIENT VECTORS")
    print("=" * 60)

    # f(x, y) = x^2 + 2y^2  (elliptic paraboloid)
    def f(x, y):
        return x**2 + 2*y**2

    def grad_f(x, y):
        """Gradient: [∂f/∂x, ∂f/∂y] = [2x, 4y]"""
        return np.array([2*x, 4*y])

    # Key properties of gradient
    print("\nf(x, y) = x² + 2y²")
    print("∇f(x, y) = [2x, 4y]")

    points = [(1, 1), (2, 0), (0, 1), (-1, -1)]
    print("\nGradient at various points:")
    for px, py in points:
        g = grad_f(px, py)
        magnitude = np.linalg.norm(g)
        print(f"  ∇f({px}, {py}) = {g},  ||∇f|| = {magnitude:.4f}")

    # Gradient points in direction of steepest ascent
    print("\n--- Key Property: Gradient is perpendicular to level curves ---")
    # Level curve f = c: x^2 + 2y^2 = c
    # At point (1, 1), f = 3. Tangent to level curve has direction [-4, 2] (from implicit diff)
    p = np.array([1.0, 1.0])
    g = grad_f(*p)
    tangent = np.array([-4*p[1], 2*p[0]])  # tangent to ellipse at (1,1)
    dot_product = np.dot(g, tangent)
    print(f"At point (1, 1): ∇f = {g}")
    print(f"Level curve tangent direction: {tangent}")
    print(f"Dot product (should be ~0): {dot_product:.6f}")


def directional_derivative():
    """Demonstrate directional derivatives"""
    print("\n" + "=" * 60)
    print("DIRECTIONAL DERIVATIVES")
    print("=" * 60)

    print("\nD_u f(x) = ∇f(x) · u  (unit vector u gives rate of change in direction u)")

    # f(x, y) = x^2 + xy - y^2
    def f(x, y):
        return x**2 + x*y - y**2

    def grad_f(x, y):
        return np.array([2*x + y, x - 2*y])

    # Point and various directions
    point = np.array([2.0, 1.0])
    g = grad_f(*point)
    f_val = f(*point)

    print(f"\nf(x, y) = x² + xy - y²")
    print(f"At point {point}: f = {f_val},  ∇f = {g}")

    directions = {
        "x-axis [1, 0]": np.array([1.0, 0.0]),
        "y-axis [0, 1]": np.array([0.0, 1.0]),
        "diagonal [1, 1]/√2": np.array([1.0, 1.0]) / np.sqrt(2),
        "steepest ascent (∇f dir)": g / np.linalg.norm(g),
        "steepest descent (-∇f dir)": -g / np.linalg.norm(g),
    }

    print("\nDirectional derivatives at this point:")
    for name, u in directions.items():
        D_u = np.dot(g, u)
        print(f"  D_{{{name}}} f = {D_u:.4f}")

    max_rate = np.linalg.norm(g)
    print(f"\nMax rate of increase (||∇f||) = {max_rate:.4f}")
    print(f"Max rate of decrease = -{max_rate:.4f}")

    # Numerical verification
    print("\n--- Numerical Verification ---")
    u = np.array([1.0, 1.0]) / np.sqrt(2)
    h = 1e-6
    D_u_numerical = (f(*(point + h*u)) - f(*(point - h*u))) / (2*h)
    D_u_analytical = np.dot(g, u)
    print(f"Direction u = {u}")
    print(f"Numerical D_u f  = {D_u_numerical:.8f}")
    print(f"Analytical D_u f = {D_u_analytical:.8f}")
    print(f"Difference: {abs(D_u_numerical - D_u_analytical):.2e}")


def taylor_expansion():
    """Second-order Taylor expansion for multivariate functions"""
    print("\n" + "=" * 60)
    print("TAYLOR EXPANSION (MULTIVARIATE)")
    print("=" * 60)

    print("\nSecond-order Taylor: f(x) ≈ f(a) + ∇f(a)ᵀ(x-a) + ½(x-a)ᵀH(a)(x-a)")

    # f(x, y) = exp(-(x^2 + y^2)/2)  — Gaussian bump, expand around (0, 0)
    def f(x, y):
        return np.exp(-(x**2 + y**2) / 2)

    def grad_f(x, y):
        return np.array([-x * np.exp(-(x**2 + y**2) / 2),
                         -y * np.exp(-(x**2 + y**2) / 2)])

    def hessian_f(x, y):
        e = np.exp(-(x**2 + y**2) / 2)
        return e * np.array([[x**2 - 1, x*y],
                              [x*y, y**2 - 1]])

    # Expand around a = (0, 0)
    a = np.array([0.0, 0.0])
    f_a = f(*a)
    g_a = grad_f(*a)
    H_a = hessian_f(*a)

    print(f"\nf(x, y) = exp(-(x² + y²)/2)")
    print(f"Expanding around a = {a}")
    print(f"f(a) = {f_a:.4f}")
    print(f"∇f(a) = {g_a}")
    print(f"H(a) =\n{H_a}")

    def taylor_approx(x, y):
        dx = np.array([x - a[0], y - a[1]])
        return f_a + g_a @ dx + 0.5 * dx @ H_a @ dx

    # Compare at various points
    print("\nf(x, y) vs Taylor approximation:")
    test_points = [(0.2, 0.1), (0.5, 0.3), (1.0, 0.5), (1.5, 1.0)]
    print(f"{'Point':15s} {'True f':12s} {'Taylor approx':15s} {'Error':12s}")
    for px, py in test_points:
        true_val = f(px, py)
        approx_val = taylor_approx(px, py)
        error = abs(true_val - approx_val)
        print(f"({px}, {py}):       {true_val:10.6f}   {approx_val:13.6f}   {error:.2e}")

    print("\nNote: Taylor approximation accuracy degrades farther from expansion point")


def visualize_gradient_field():
    """Visualize gradient field and level curves"""
    print("\n" + "=" * 60)
    print("GRADIENT FIELD VISUALIZATION")
    print("=" * 60)

    # f(x, y) = sin(x) * cos(y)
    def f(x, y):
        return np.sin(x) * np.cos(y)

    x_range = np.linspace(-np.pi, np.pi, 200)
    y_range = np.linspace(-np.pi, np.pi, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    # Gradient: ∂f/∂x = cos(x)*cos(y), ∂f/∂y = -sin(x)*sin(y)
    x_quiver = np.linspace(-np.pi, np.pi, 15)
    y_quiver = np.linspace(-np.pi, np.pi, 15)
    Xq, Yq = np.meshgrid(x_quiver, y_quiver)
    Gx = np.cos(Xq) * np.cos(Yq)   # ∂f/∂x
    Gy = -np.sin(Xq) * np.sin(Yq)  # ∂f/∂y

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: level curves and gradient field
    ax1 = axes[0]
    contour = ax1.contourf(X, Y, Z, levels=20, cmap='RdBu_r', alpha=0.7)
    ax1.contour(X, Y, Z, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    plt.colorbar(contour, ax=ax1)
    # Normalize arrows for display
    magnitude = np.sqrt(Gx**2 + Gy**2) + 1e-10
    ax1.quiver(Xq, Yq, Gx/magnitude, Gy/magnitude,
               color='yellow', scale=20, width=0.004, alpha=0.8)
    ax1.set_title('f(x,y) = sin(x)cos(y)\nLevel curves + gradient field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')

    # Right: Taylor approximation accuracy as function of distance
    ax2 = axes[1]
    def g(x, y):
        return x**2 + 2*y**2

    def g_taylor(x, y, a_x, a_y):
        # Second-order Taylor around (a_x, a_y)
        dx = x - a_x
        dy = y - a_y
        f0 = a_x**2 + 2*a_y**2
        gx = 2*a_x
        gy = 4*a_y
        # Hessian is constant: [[2, 0], [0, 4]]
        return f0 + gx*dx + gy*dy + 0.5*(2*dx**2 + 4*dy**2)

    # Error along x-axis from expansion point (1, 0.5)
    a_x, a_y = 1.0, 0.5
    t_vals = np.linspace(0, 2, 100)
    true_vals = g(a_x + t_vals, a_y)
    taylor_vals = g_taylor(a_x + t_vals, a_y, a_x, a_y)

    ax2.plot(t_vals, true_vals, 'b-', linewidth=2, label='True f')
    ax2.plot(t_vals, taylor_vals, 'r--', linewidth=2, label='2nd-order Taylor')
    ax2.axvline(0, color='gray', linestyle=':', alpha=0.7, label='Expansion point')
    ax2.set_xlabel('Distance from expansion point along x')
    ax2.set_ylabel('f value')
    ax2.set_title('Taylor Approximation Accuracy\ng(x, y) = x² + 2y²')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('multivariate_calculus.png', dpi=150)
    print("Visualization saved to multivariate_calculus.png")
    plt.close()


if __name__ == "__main__":
    partial_derivatives_symbolic()
    gradient_vector()
    directional_derivative()
    taylor_expansion()
    visualize_gradient_field()

    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
