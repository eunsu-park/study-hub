"""
Exercise Solutions: Lesson 03 - Applications of Derivatives
Calculus and Differential Equations

Topics covered:
- Finding and classifying extrema
- Optimization (surface area minimization)
- Related rates (expanding balloon)
- L'Hopital's Rule
- Taylor polynomial approximation
"""

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Finding and Classifying Extrema
# ============================================================
def exercise_1():
    """
    Find all critical points of f(x) = x^4 - 4x^3 + 4x^2
    and classify each as local max, local min, or neither.
    """
    print("=" * 60)
    print("Problem 1: Finding and Classifying Extrema")
    print("=" * 60)

    x = sp.Symbol('x')
    f = x**4 - 4*x**3 + 4*x**2

    # Find critical points: f'(x) = 0
    f_prime = sp.diff(f, x)
    f_double_prime = sp.diff(f, x, 2)

    critical_points = sp.solve(f_prime, x)
    print(f"\n  f(x) = x^4 - 4x^3 + 4x^2")
    print(f"  f'(x) = {f_prime}")
    print(f"  f'(x) = {sp.factor(f_prime)}")
    print(f"  f''(x) = {f_double_prime}")
    print(f"\n  Critical points (f'(x) = 0): {critical_points}")

    # Second derivative test
    print(f"\n  Classification using the second derivative test:")
    for cp in critical_points:
        f2_val = f_double_prime.subs(x, cp)
        f_val = f.subs(x, cp)
        if f2_val > 0:
            classification = "LOCAL MINIMUM"
        elif f2_val < 0:
            classification = "LOCAL MAXIMUM"
        else:
            classification = "INCONCLUSIVE (need higher-order test)"
        print(f"    x = {cp}: f({cp}) = {f_val}, f''({cp}) = {f2_val} => {classification}")

    # For x=2 where second derivative test is inconclusive,
    # use first derivative test or factor analysis
    print(f"\n  For x = 2 (f''(2) = 0), use the first derivative test:")
    print(f"    f'(x) = 4x(x-2)^2 = 4x(x^2 - 4x + 4)")
    print(f"    Near x=2: (x-2)^2 >= 0 always, and 4x > 0 for x near 2")
    print(f"    So f'(x) >= 0 on both sides of x=2 => neither max nor min")
    print(f"    x = 2 is an INFLECTION POINT")

    # Visualization
    x_vals = np.linspace(-1, 3.5, 500)
    y_vals = x_vals**4 - 4*x_vals**3 + 4*x_vals**2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label=r'$f(x) = x^4 - 4x^3 + 4x^2$')

    # Mark critical points
    for cp in critical_points:
        cp_f = float(cp)
        fv = float(f.subs(x, cp))
        ax.plot(cp_f, fv, 'ro', markersize=10)
        ax.annotate(f'({cp_f}, {fv})', xy=(cp_f, fv), xytext=(cp_f + 0.2, fv + 0.5),
                    fontsize=11, color='red')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Critical Points of $f(x) = x^4 - 4x^3 + 4x^2$', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex03_extrema.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: ex03_extrema.png]")


# ============================================================
# Problem 2: Optimization
# ============================================================
def exercise_2():
    """
    Rectangular box with square base and open top, volume = 32000 cm^3.
    Minimize surface area: S = x^2 + 4xh, subject to V = x^2 * h = 32000.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Optimization")
    print("=" * 60)

    x, h = sp.symbols('x h', positive=True)

    # Constraint: V = x^2 * h = 32000  =>  h = 32000/x^2
    # Surface area (open top): S = x^2 + 4xh = x^2 + 4x*(32000/x^2) = x^2 + 128000/x
    V = 32000
    h_expr = V / x**2
    S = x**2 + 4*x*h_expr  # = x^2 + 128000/x

    print(f"\n  Constraint: V = x^2 * h = 32000  =>  h = 32000/x^2")
    print(f"  Surface area (open top): S = x^2 + 4xh")
    print(f"  Substituting h: S = {sp.simplify(S)}")

    # Find critical point
    dS = sp.diff(S, x)
    critical_x = sp.solve(dS, x)
    print(f"\n  dS/dx = {dS}")
    print(f"  Setting dS/dx = 0: x = {critical_x}")

    # Since x must be positive, take the real positive root
    x_opt = [c for c in critical_x if c.is_real and c > 0][0]
    h_opt = V / x_opt**2
    S_opt = S.subs(x, x_opt)

    # Verify it's a minimum with second derivative
    d2S = sp.diff(S, x, 2)
    d2S_val = d2S.subs(x, x_opt)

    print(f"\n  Optimal dimensions:")
    print(f"    x (base side) = {x_opt} = {float(x_opt):.4f} cm")
    print(f"    h (height)    = {sp.simplify(h_opt)} = {float(h_opt):.4f} cm")
    print(f"    Minimum surface area = {sp.simplify(S_opt)} = {float(S_opt):.4f} cm^2")
    print(f"\n  d^2S/dx^2 at x_opt = {float(d2S_val):.4f} > 0 => confirmed minimum")

    # Note: the optimal box has h = x/2 (height is half the base side)
    print(f"\n  Insight: h = 32000/x^2 = 32000/({float(x_opt):.2f})^2 = {float(h_opt):.2f}")
    print(f"           h/x = {float(h_opt/x_opt):.4f} (height = half the base side)")


# ============================================================
# Problem 3: Related Rates
# ============================================================
def exercise_3():
    """
    Spherical balloon inflated at dV/dt = 100 cm^3/s.
    Find dr/dt when diameter = 50 cm (r = 25 cm).
    V = (4/3)*pi*r^3
    """
    print("\n" + "=" * 60)
    print("Problem 3: Related Rates")
    print("=" * 60)

    t, r = sp.symbols('t r', positive=True)

    # V = (4/3)*pi*r^3
    # dV/dt = 4*pi*r^2 * dr/dt
    # dr/dt = dV/dt / (4*pi*r^2)

    dV_dt = 100  # cm^3/s
    r_val = 25   # cm (diameter = 50 cm)

    dr_dt = dV_dt / (4 * sp.pi * r_val**2)

    print(f"\n  Given: dV/dt = {dV_dt} cm^3/s, diameter = 50 cm => r = {r_val} cm")
    print(f"\n  V = (4/3)*pi*r^3")
    print(f"  Differentiating with respect to t:")
    print(f"  dV/dt = 4*pi*r^2 * dr/dt")
    print(f"  dr/dt = dV/dt / (4*pi*r^2)")
    print(f"        = {dV_dt} / (4*pi*{r_val}^2)")
    print(f"        = {dV_dt} / (4*pi*{r_val**2})")
    print(f"        = {dr_dt}")
    print(f"        = {float(dr_dt):.6f} cm/s")
    print(f"        ~ {float(dr_dt):.4e} cm/s")
    print(f"\n  The radius is increasing at approximately {float(dr_dt)*100:.4f} x 10^-2 cm/s")
    print(f"  (about 1/(25*pi) cm/s)")


# ============================================================
# Problem 4: L'Hopital's Rule
# ============================================================
def exercise_4():
    """
    (a) lim_{x->0} (e^x - 1 - x - x^2/2) / x^3
    (b) lim_{x->0+} x*ln(x)
    (c) lim_{x->inf} x^(1/x)
    """
    print("\n" + "=" * 60)
    print("Problem 4: L'Hopital's Rule")
    print("=" * 60)

    x = sp.Symbol('x')

    # (a) lim_{x->0} (e^x - 1 - x - x^2/2) / x^3
    # This is 0/0 form. Apply L'Hopital three times:
    expr_a = (sp.exp(x) - 1 - x - x**2/2) / x**3
    limit_a = sp.limit(expr_a, x, 0)
    print(f"\n(a) lim_{{x->0}} (e^x - 1 - x - x^2/2) / x^3")
    print(f"    Form: 0/0")
    print(f"    Apply L'Hopital 1st time: (e^x - 1 - x) / (3x^2)  -- still 0/0")
    print(f"    Apply L'Hopital 2nd time: (e^x - 1) / (6x)  -- still 0/0")
    print(f"    Apply L'Hopital 3rd time: e^x / 6  ->  1/6")
    print(f"    Answer: {limit_a}")

    # (b) lim_{x->0+} x*ln(x)
    # This is 0 * (-inf) form. Rewrite as ln(x) / (1/x)  =>  -inf/inf
    expr_b = x * sp.ln(x)
    limit_b = sp.limit(expr_b, x, 0, '+')
    print(f"\n(b) lim_{{x->0+}} x*ln(x)")
    print(f"    Form: 0 * (-inf)")
    print(f"    Rewrite: ln(x) / (1/x)  =>  -inf/inf form")
    print(f"    L'Hopital: (1/x) / (-1/x^2) = -x  ->  0")
    print(f"    Answer: {limit_b}")

    # (c) lim_{x->inf} x^(1/x)
    # This is inf^0 form. Let y = x^(1/x), ln(y) = ln(x)/x
    # lim_{x->inf} ln(x)/x = inf/inf => L'Hopital: (1/x)/1 = 0
    # So ln(y) -> 0, hence y -> e^0 = 1
    expr_c = x ** (1/x)
    limit_c = sp.limit(expr_c, x, sp.oo)
    print(f"\n(c) lim_{{x->inf}} x^(1/x)")
    print(f"    Form: inf^0")
    print(f"    Let y = x^(1/x), so ln(y) = ln(x)/x")
    print(f"    lim_{{x->inf}} ln(x)/x  [inf/inf form]")
    print(f"    L'Hopital: (1/x)/1 = 1/x  ->  0")
    print(f"    Since ln(y) -> 0, we get y -> e^0 = 1")
    print(f"    Answer: {limit_c}")


# ============================================================
# Problem 5: Taylor Polynomial Approximation
# ============================================================
def exercise_5():
    """
    (a) Write the 4th-degree Maclaurin polynomial for e^x. Estimate e^0.5.
    (b) Plot sin(x) with Taylor polynomials of degree 1, 3, 5, 7, 9 on [-2pi, 2pi].
    """
    print("\n" + "=" * 60)
    print("Problem 5: Taylor Polynomial Approximation")
    print("=" * 60)

    x = sp.Symbol('x')

    # (a) Maclaurin polynomial for e^x
    # e^x = 1 + x + x^2/2! + x^3/3! + x^4/4! + ...
    T4 = sp.series(sp.exp(x), x, 0, 5).removeO()
    print(f"\n(a) 4th-degree Maclaurin polynomial for e^x:")
    print(f"    T_4(x) = {T4}")

    # Estimate e^0.5
    approx = float(T4.subs(x, sp.Rational(1, 2)))
    exact = np.exp(0.5)
    error = abs(approx - exact)
    print(f"\n    Estimating e^0.5:")
    print(f"    T_4(0.5) = 1 + 0.5 + 0.125 + 0.020833... + 0.002604...")
    print(f"             = {approx:.10f}")
    print(f"    Exact e^0.5 = {exact:.10f}")
    print(f"    Error = {error:.2e}")

    # (b) Taylor polynomials of sin(x)
    print(f"\n(b) Taylor polynomials of sin(x) on [-2pi, 2pi]:")

    x_vals = np.linspace(-2*np.pi, 2*np.pi, 1000)
    y_sin = np.sin(x_vals)

    degrees = [1, 3, 5, 7, 9]
    colors = ['red', 'orange', 'green', 'blue', 'purple']

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_vals, y_sin, 'k-', linewidth=3, label='sin(x)')

    for deg, color in zip(degrees, colors):
        # Taylor polynomial of degree 'deg' centered at 0
        T_n = sp.series(sp.sin(x), x, 0, deg + 1).removeO()
        T_func = sp.lambdify(x, T_n, 'numpy')
        y_taylor = T_func(x_vals)

        # Clip for visualization (Taylor polynomials diverge)
        y_taylor = np.clip(y_taylor, -3, 3)
        ax.plot(x_vals, y_taylor, color=color, linewidth=1.5, linestyle='--',
                label=f'$T_{{{deg}}}(x) = {sp.latex(T_n)}$' if deg <= 3
                else f'$T_{{{deg}}}(x)$')
        print(f"    T_{deg}(x) = {T_n}")

    ax.set_ylim(-3, 3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Taylor Polynomial Approximations of sin(x)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('ex03_taylor_sin.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: ex03_taylor_sin.png]")
    print("  Observation: Higher-degree polynomials approximate sin(x)")
    print("  over a wider range. T_9 is nearly exact on [-pi, pi].")


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
    print("All exercises for Lesson 03 completed.")
    print("=" * 60)
