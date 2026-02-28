"""
Exercise Solutions: Lesson 02 - Derivatives Fundamentals
Calculus and Differential Equations

Topics covered:
- Differentiation rules (power, product, quotient, chain)
- Implicit differentiation
- Numerical accuracy investigation (forward vs central difference)
- Derivative from first principles (sin x)
- Chain rule in machine learning (sigmoid backpropagation)
"""

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Applying Differentiation Rules
# ============================================================
def exercise_1():
    """
    Compute the derivative of each function:
    (a) f(x) = 3x^4 - 2x^3 + 7x - 9
    (b) g(x) = x^2 * e^x * sin(x)  (product rule twice)
    (c) h(x) = ln(x) / (x^2 + 1)
    (d) k(x) = cos(sqrt(x^2 + 1))  (chain rule, three nested functions)
    """
    print("=" * 60)
    print("Problem 1: Applying Differentiation Rules")
    print("=" * 60)

    x = sp.Symbol('x')

    # (a) Power rule
    f_a = 3*x**4 - 2*x**3 + 7*x - 9
    df_a = sp.diff(f_a, x)
    print(f"\n(a) f(x) = 3x^4 - 2x^3 + 7x - 9")
    print(f"    f'(x) = {df_a}")
    print(f"    (Apply power rule to each term: 12x^3 - 6x^2 + 7)")

    # (b) Product rule applied twice: d/dx[x^2 * e^x * sin(x)]
    # Let u = x^2, v = e^x, w = sin(x)
    # d/dx[uvw] = u'vw + uv'w + uvw'
    f_b = x**2 * sp.exp(x) * sp.sin(x)
    df_b = sp.diff(f_b, x)
    df_b_simplified = sp.simplify(df_b)
    print(f"\n(b) g(x) = x^2 * e^x * sin(x)")
    print(f"    g'(x) = d/dx[x^2]*e^x*sin(x) + x^2*d/dx[e^x]*sin(x) + x^2*e^x*d/dx[sin(x)]")
    print(f"         = 2x*e^x*sin(x) + x^2*e^x*sin(x) + x^2*e^x*cos(x)")
    print(f"    SymPy result: {df_b}")
    print(f"    Simplified: {df_b_simplified}")

    # (c) Quotient rule: d/dx[ln(x)/(x^2+1)]
    f_c = sp.ln(x) / (x**2 + 1)
    df_c = sp.diff(f_c, x)
    df_c_simplified = sp.simplify(df_c)
    print(f"\n(c) h(x) = ln(x) / (x^2 + 1)")
    print(f"    h'(x) = [(1/x)(x^2+1) - ln(x)(2x)] / (x^2+1)^2")
    print(f"    SymPy result: {df_c}")
    print(f"    Simplified: {df_c_simplified}")

    # (d) Chain rule: d/dx[cos(sqrt(x^2+1))]
    # Outer: cos(u), u = sqrt(v), v = x^2+1
    # d/dx = -sin(sqrt(x^2+1)) * 1/(2*sqrt(x^2+1)) * 2x
    #       = -x*sin(sqrt(x^2+1)) / sqrt(x^2+1)
    f_d = sp.cos(sp.sqrt(x**2 + 1))
    df_d = sp.diff(f_d, x)
    df_d_simplified = sp.simplify(df_d)
    print(f"\n(d) k(x) = cos(sqrt(x^2 + 1))")
    print(f"    Three nested functions:")
    print(f"      Outer: cos(u)  =>  -sin(u)")
    print(f"      Middle: u = sqrt(v)  =>  1/(2*sqrt(v))")
    print(f"      Inner: v = x^2+1  =>  2x")
    print(f"    k'(x) = -sin(sqrt(x^2+1)) * (1/(2*sqrt(x^2+1))) * 2x")
    print(f"          = -x*sin(sqrt(x^2+1)) / sqrt(x^2+1)")
    print(f"    SymPy result: {df_d_simplified}")


# ============================================================
# Problem 2: Implicit Differentiation
# ============================================================
def exercise_2():
    """
    Find dy/dx for the ellipse x^2/9 + y^2/4 = 1.
    Find points where the tangent is horizontal and vertical.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Implicit Differentiation")
    print("=" * 60)

    x, y = sp.symbols('x y')

    # Implicit differentiation of x^2/9 + y^2/4 = 1
    # d/dx: 2x/9 + 2y/4 * dy/dx = 0
    # dy/dx = -(2x/9) / (2y/4) = -(4x) / (9*2y) = -4x/(9y) ... wait
    # Let's be careful: 2x/9 + (2y*y')/4 = 0  =>  y' = -(2x/9)*(4/(2y)) = -4x/(9y)
    ellipse = x**2 / 9 + y**2 / 4 - 1
    dy_dx = sp.idiff(ellipse, y, x)
    print(f"\n  Ellipse: x^2/9 + y^2/4 = 1")
    print(f"  Implicit differentiation:")
    print(f"    2x/9 + (2y/4)*dy/dx = 0")
    print(f"    dy/dx = -4x/(9y)")
    print(f"    SymPy result: dy/dx = {dy_dx}")

    # Horizontal tangent: dy/dx = 0 => -4x/(9y) = 0 => x = 0
    # When x=0: y^2/4 = 1 => y = +/- 2
    print(f"\n  Horizontal tangent (dy/dx = 0):")
    print(f"    -4x/(9y) = 0  =>  x = 0")
    print(f"    At x=0: y^2/4 = 1  =>  y = +/-2")
    print(f"    Points: (0, 2) and (0, -2)")

    # Vertical tangent: dy/dx undefined (denominator = 0) => y = 0
    # When y=0: x^2/9 = 1 => x = +/- 3
    print(f"\n  Vertical tangent (dy/dx undefined, y = 0):")
    print(f"    At y=0: x^2/9 = 1  =>  x = +/-3")
    print(f"    Points: (3, 0) and (-3, 0)")

    # Plot the ellipse with tangent points
    theta = np.linspace(0, 2*np.pi, 500)
    x_ell = 3 * np.cos(theta)
    y_ell = 2 * np.sin(theta)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_ell, y_ell, 'b-', linewidth=2, label='Ellipse $x^2/9 + y^2/4 = 1$')

    # Mark horizontal tangent points
    ax.plot([0, 0], [2, -2], 'ro', markersize=10, label='Horizontal tangent')
    ax.axhline(y=2, color='r', linestyle='--', alpha=0.3, xmin=0.3, xmax=0.7)
    ax.axhline(y=-2, color='r', linestyle='--', alpha=0.3, xmin=0.3, xmax=0.7)

    # Mark vertical tangent points
    ax.plot([3, -3], [0, 0], 'gs', markersize=10, label='Vertical tangent')
    ax.axvline(x=3, color='g', linestyle='--', alpha=0.3, ymin=0.2, ymax=0.8)
    ax.axvline(x=-3, color='g', linestyle='--', alpha=0.3, ymin=0.2, ymax=0.8)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Ellipse with Tangent Points', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex02_ellipse_tangents.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: ex02_ellipse_tangents.png]")


# ============================================================
# Problem 3: Numerical Accuracy Investigation
# ============================================================
def exercise_3():
    """
    Compute derivative of f(x) = e^x at x = 0 using:
    - Forward difference with h = 10^-k for k = 1..16
    - Central difference with same h values
    Plot absolute error vs h on log-log scale.
    Explain the error behavior.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Numerical Accuracy Investigation")
    print("=" * 60)

    x0 = 0.0
    exact = np.exp(x0)  # f'(0) = e^0 = 1

    k_values = np.arange(1, 17)
    h_values = 10.0 ** (-k_values)

    forward_errors = []
    central_errors = []

    print(f"\n  f(x) = e^x, f'(0) = 1 (exact)")
    print(f"  {'h':>12s}  {'Forward err':>14s}  {'Central err':>14s}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*14}")

    for h in h_values:
        # Forward difference: [f(x+h) - f(x)] / h
        fwd = (np.exp(x0 + h) - np.exp(x0)) / h
        fwd_err = abs(fwd - exact)

        # Central difference: [f(x+h) - f(x-h)] / (2h)
        ctr = (np.exp(x0 + h) - np.exp(x0 - h)) / (2 * h)
        ctr_err = abs(ctr - exact)

        forward_errors.append(fwd_err)
        central_errors.append(ctr_err)
        print(f"  {h:>12.1e}  {fwd_err:>14.6e}  {ctr_err:>14.6e}")

    forward_errors = np.array(forward_errors)
    central_errors = np.array(central_errors)

    print(f"\n  Explanation:")
    print(f"  - Forward difference has truncation error O(h), so error decreases linearly with h.")
    print(f"  - Central difference has truncation error O(h^2), so error decreases quadratically.")
    print(f"  - For very small h (h < ~1e-8 forward, ~1e-5 central), roundoff error dominates")
    print(f"    because floating-point subtraction of nearly equal numbers loses significant digits.")
    print(f"  - Optimal h: ~1e-8 for forward (sqrt(machine_eps)), ~1e-5 for central (eps^(1/3)).")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(h_values, forward_errors, 'bo-', linewidth=2, markersize=6, label='Forward difference')
    ax.loglog(h_values, central_errors, 'rs-', linewidth=2, markersize=6, label='Central difference')

    # Reference lines for theoretical slopes
    h_ref = h_values[h_values > 1e-8]
    ax.loglog(h_ref, h_ref, 'b--', alpha=0.4, label='O(h) reference')
    ax.loglog(h_ref, h_ref**2, 'r--', alpha=0.4, label='O(h^2) reference')

    ax.set_xlabel('Step size h', fontsize=12)
    ax.set_ylabel('Absolute error', fontsize=12)
    ax.set_title('Numerical Differentiation: Truncation vs Roundoff Error', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig('ex02_numerical_diff_error.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex02_numerical_diff_error.png]")


# ============================================================
# Problem 4: Derivative from First Principles
# ============================================================
def exercise_4():
    """
    Prove d/dx[sin(x)] = cos(x) using only the limit definition.
    Verify symbolically and numerically.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Derivative from First Principles")
    print("=" * 60)

    print("\n  Proof:")
    print("  d/dx[sin(x)] = lim_{h->0} [sin(x+h) - sin(x)] / h")
    print("")
    print("  Using angle addition: sin(x+h) = sin(x)cos(h) + cos(x)sin(h)")
    print("")
    print("  = lim_{h->0} [sin(x)cos(h) + cos(x)sin(h) - sin(x)] / h")
    print("  = lim_{h->0} [sin(x)(cos(h)-1) + cos(x)sin(h)] / h")
    print("  = sin(x) * lim_{h->0} (cos(h)-1)/h  +  cos(x) * lim_{h->0} sin(h)/h")
    print("  = sin(x) * 0  +  cos(x) * 1")
    print("  = cos(x)  QED")

    # Symbolic verification with SymPy
    x, h = sp.symbols('x h')

    # Compute the limit directly
    diff_quotient = (sp.sin(x + h) - sp.sin(x)) / h
    limit_result = sp.limit(diff_quotient, h, 0)
    print(f"\n  SymPy verification:")
    print(f"    lim_{{h->0}} [sin(x+h) - sin(x)] / h = {limit_result}")

    # Also verify the two key limits
    lim1 = sp.limit(sp.sin(h)/h, h, 0)
    lim2 = sp.limit((sp.cos(h) - 1)/h, h, 0)
    print(f"    lim_{{h->0}} sin(h)/h = {lim1}")
    print(f"    lim_{{h->0}} (cos(h)-1)/h = {lim2}")

    # Numerical verification at specific point
    x0 = 1.0  # test at x = 1
    h_vals = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    exact = np.cos(x0)
    print(f"\n  Numerical check at x = {x0}:")
    print(f"  Exact cos({x0}) = {exact:.15f}")
    for hv in h_vals:
        approx = (np.sin(x0 + hv) - np.sin(x0)) / hv
        print(f"    h = {hv:.0e}: [sin({x0}+h)-sin({x0})]/h = {approx:.15f}, error = {abs(approx-exact):.2e}")


# ============================================================
# Problem 5: Chain Rule in Machine Learning
# ============================================================
def exercise_5():
    """
    Neural network loss: L = (y - y_hat)^2 where y_hat = sigma(w*x + b),
    sigma(z) = 1/(1 + e^(-z)).
    (a) Compute dL/dw using chain rule.
    (b) Verify with SymPy.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Chain Rule in Machine Learning")
    print("=" * 60)

    w, x_s, b, y_s = sp.symbols('w x b y', real=True)

    # Define the sigmoid
    z = w * x_s + b
    sigma = 1 / (1 + sp.exp(-z))
    y_hat = sigma
    L = (y_s - y_hat)**2

    print("\n  (a) Chain Rule derivation:")
    print("  L = (y - y_hat)^2")
    print("  y_hat = sigma(z) = 1/(1 + e^(-z))")
    print("  z = w*x + b")
    print("")
    print("  dL/dw = dL/dy_hat * dy_hat/dz * dz/dw")
    print("  dL/dy_hat = -2(y - y_hat)")
    print("  dy_hat/dz = sigma(z)*(1 - sigma(z))  [sigmoid derivative]")
    print("  dz/dw = x")
    print("")
    print("  Therefore:")
    print("  dL/dw = -2(y - y_hat) * sigma(z)*(1 - sigma(z)) * x")

    # (b) SymPy verification
    dL_dw = sp.diff(L, w)
    dL_dw_simplified = sp.simplify(dL_dw)
    print(f"\n  (b) SymPy verification:")
    print(f"  dL/dw = {dL_dw_simplified}")

    # Numerical example
    w_val, x_val, b_val, y_val = 0.5, 2.0, -0.3, 1.0
    z_val = w_val * x_val + b_val
    sig_val = 1.0 / (1.0 + np.exp(-z_val))
    y_hat_val = sig_val
    L_val = (y_val - y_hat_val)**2

    # Manual chain rule
    dL_manual = -2 * (y_val - y_hat_val) * sig_val * (1 - sig_val) * x_val

    # SymPy evaluation
    dL_sympy = float(dL_dw.subs([(w, w_val), (x_s, x_val), (b, b_val), (y_s, y_val)]))

    print(f"\n  Numerical check (w={w_val}, x={x_val}, b={b_val}, y={y_val}):")
    print(f"    z = {z_val}, sigma(z) = {sig_val:.6f}")
    print(f"    y_hat = {y_hat_val:.6f}, L = {L_val:.6f}")
    print(f"    dL/dw (chain rule) = {dL_manual:.6f}")
    print(f"    dL/dw (SymPy)      = {dL_sympy:.6f}")
    print(f"    Agreement: {abs(dL_manual - dL_sympy):.2e}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("\n" + "=" * 60)
    print("All exercises for Lesson 02 completed.")
    print("=" * 60)
