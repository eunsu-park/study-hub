"""
Exercise Solutions: Lesson 05 - Integration Techniques
Calculus and Differential Equations

Topics covered:
- u-Substitution
- Integration by parts
- Partial fractions
- Trigonometric substitution
- Improper integral convergence
"""

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Substitution
# ============================================================
def exercise_1():
    """
    Evaluate:
    (a) integral e^(sqrt(x)) / sqrt(x) dx
    (b) integral_0^{pi/2} cos(x) * e^(sin(x)) dx
    (c) integral x / (x^2+1)^3 dx
    """
    print("=" * 60)
    print("Problem 1: Substitution")
    print("=" * 60)

    x = sp.Symbol('x')

    # (a) integral e^(sqrt(x)) / sqrt(x) dx
    # Let u = sqrt(x), du = 1/(2*sqrt(x)) dx  =>  2 du = dx/sqrt(x)
    # integral = 2 * integral e^u du = 2*e^u + C = 2*e^(sqrt(x)) + C
    expr_a = sp.exp(sp.sqrt(x)) / sp.sqrt(x)
    result_a = sp.integrate(expr_a, x)
    print(f"\n(a) integral e^(sqrt(x)) / sqrt(x) dx")
    print(f"    Let u = sqrt(x), du = 1/(2*sqrt(x)) dx")
    print(f"    => integral = 2 * integral e^u du = 2*e^(sqrt(x)) + C")
    print(f"    SymPy result: {result_a}")

    # (b) integral_0^{pi/2} cos(x) * e^(sin(x)) dx
    # Let u = sin(x), du = cos(x) dx
    # When x=0: u=0; when x=pi/2: u=1
    # integral = integral_0^1 e^u du = e^1 - e^0 = e - 1
    expr_b = sp.cos(x) * sp.exp(sp.sin(x))
    result_b = sp.integrate(expr_b, (x, 0, sp.pi/2))
    print(f"\n(b) integral_0^{{pi/2}} cos(x) * e^(sin(x)) dx")
    print(f"    Let u = sin(x), du = cos(x) dx")
    print(f"    Limits: x=0 -> u=0, x=pi/2 -> u=1")
    print(f"    = integral_0^1 e^u du = e - 1")
    print(f"    = {result_b} = {float(result_b):.10f}")

    # (c) integral x / (x^2+1)^3 dx
    # Let u = x^2+1, du = 2x dx  =>  x dx = du/2
    # integral = (1/2) * integral u^(-3) du = (1/2)*(-1/2)*u^(-2) + C
    # = -1/(4*(x^2+1)^2) + C
    expr_c = x / (x**2 + 1)**3
    result_c = sp.integrate(expr_c, x)
    print(f"\n(c) integral x / (x^2+1)^3 dx")
    print(f"    Let u = x^2+1, du = 2x dx")
    print(f"    = (1/2) integral u^(-3) du = -1/(4*u^2) + C")
    print(f"    = -1/(4*(x^2+1)^2) + C")
    print(f"    SymPy result: {result_c}")


# ============================================================
# Problem 2: Integration by Parts
# ============================================================
def exercise_2():
    """
    Evaluate:
    (a) integral x^2 * e^(-x) dx
    (b) integral e^x * cos(x) dx
    (c) integral arctan(x) dx
    """
    print("\n" + "=" * 60)
    print("Problem 2: Integration by Parts")
    print("=" * 60)

    x = sp.Symbol('x')

    # (a) integral x^2 * e^(-x) dx
    # Apply IBP twice. u=x^2, dv=e^(-x)dx => du=2x dx, v=-e^(-x)
    # = -x^2*e^(-x) + 2*integral x*e^(-x) dx
    # Apply again: u=x, dv=e^(-x)dx => du=dx, v=-e^(-x)
    # = -x^2*e^(-x) + 2*(-x*e^(-x) + integral e^(-x) dx)
    # = -x^2*e^(-x) - 2x*e^(-x) - 2*e^(-x) + C
    # = -e^(-x)*(x^2 + 2x + 2) + C
    expr_a = x**2 * sp.exp(-x)
    result_a = sp.integrate(expr_a, x)
    print(f"\n(a) integral x^2 * e^(-x) dx")
    print(f"    IBP 1st: u=x^2, dv=e^(-x)dx => v=-e^(-x)")
    print(f"    = -x^2*e^(-x) + 2*integral x*e^(-x) dx")
    print(f"    IBP 2nd: u=x, dv=e^(-x)dx")
    print(f"    = -x^2*e^(-x) + 2*(-x*e^(-x) + integral e^(-x) dx)")
    print(f"    = -e^(-x)*(x^2 + 2x + 2) + C")
    print(f"    SymPy result: {result_a}")

    # (b) integral e^x * cos(x) dx
    # Apply IBP twice, then solve algebraically:
    # Let I = integral e^x cos(x) dx
    # u=cos(x), dv=e^x dx => I = e^x*cos(x) + integral e^x*sin(x) dx
    # For the second integral, u=sin(x), dv=e^x dx:
    # = e^x*cos(x) + e^x*sin(x) - integral e^x*cos(x) dx
    # = e^x*cos(x) + e^x*sin(x) - I
    # 2I = e^x*(cos(x) + sin(x))
    # I = e^x*(cos(x) + sin(x))/2 + C
    expr_b = sp.exp(x) * sp.cos(x)
    result_b = sp.integrate(expr_b, x)
    print(f"\n(b) integral e^x * cos(x) dx")
    print(f"    Let I = integral e^x*cos(x) dx")
    print(f"    IBP twice gives: I = e^x*cos(x) + e^x*sin(x) - I")
    print(f"    2I = e^x*(cos(x) + sin(x))")
    print(f"    I = e^x*(cos(x) + sin(x))/2 + C")
    print(f"    SymPy result: {result_b}")

    # (c) integral arctan(x) dx
    # u = arctan(x), dv = dx => du = 1/(1+x^2) dx, v = x
    # = x*arctan(x) - integral x/(1+x^2) dx
    # = x*arctan(x) - (1/2)*ln(1+x^2) + C
    expr_c = sp.atan(x)
    result_c = sp.integrate(expr_c, x)
    print(f"\n(c) integral arctan(x) dx")
    print(f"    u = arctan(x), dv = dx")
    print(f"    du = 1/(1+x^2) dx, v = x")
    print(f"    = x*arctan(x) - integral x/(1+x^2) dx")
    print(f"    = x*arctan(x) - (1/2)*ln(1+x^2) + C")
    print(f"    SymPy result: {result_c}")


# ============================================================
# Problem 3: Partial Fractions
# ============================================================
def exercise_3():
    """
    Evaluate:
    (a) integral (3x+1)/(x^2-5x+6) dx
    (b) integral (x^2+1)/(x*(x-1)^2) dx
    """
    print("\n" + "=" * 60)
    print("Problem 3: Partial Fractions")
    print("=" * 60)

    x = sp.Symbol('x')

    # (a) integral (3x+1)/(x^2-5x+6) dx
    # Factor: x^2-5x+6 = (x-2)(x-3)
    # (3x+1)/((x-2)(x-3)) = A/(x-2) + B/(x-3)
    # 3x+1 = A(x-3) + B(x-2)
    # x=2: 7 = A(-1) => A = -7
    # x=3: 10 = B(1) => B = 10
    expr_a = (3*x + 1) / (x**2 - 5*x + 6)
    pf_a = sp.apart(expr_a, x)
    result_a = sp.integrate(expr_a, x)
    print(f"\n(a) integral (3x+1)/(x^2-5x+6) dx")
    print(f"    Factor: x^2-5x+6 = (x-2)(x-3)")
    print(f"    Partial fractions: {pf_a}")
    print(f"    = integral [-7/(x-2) + 10/(x-3)] dx")
    print(f"    = -7*ln|x-2| + 10*ln|x-3| + C")
    print(f"    SymPy result: {result_a}")

    # (b) integral (x^2+1)/(x*(x-1)^2) dx
    # Partial fractions: A/x + B/(x-1) + C/(x-1)^2
    # x^2+1 = A(x-1)^2 + Bx(x-1) + Cx
    # x=0: 1 = A(1) => A = 1
    # x=1: 2 = C(1) => C = 2
    # Compare x^2 coeff: 1 = A + B => B = 0
    expr_b = (x**2 + 1) / (x * (x - 1)**2)
    pf_b = sp.apart(expr_b, x)
    result_b = sp.integrate(expr_b, x)
    print(f"\n(b) integral (x^2+1)/(x*(x-1)^2) dx")
    print(f"    Partial fractions: {pf_b}")
    print(f"    A=1, B=0, C=2")
    print(f"    = integral [1/x + 2/(x-1)^2] dx")
    print(f"    = ln|x| - 2/(x-1) + C")
    print(f"    SymPy result: {result_b}")


# ============================================================
# Problem 4: Trigonometric Substitution
# ============================================================
def exercise_4():
    """
    Evaluate integral x^2 / sqrt(9 - x^2) dx using x = 3*sin(theta).
    """
    print("\n" + "=" * 60)
    print("Problem 4: Trigonometric Substitution")
    print("=" * 60)

    x, theta = sp.symbols('x theta')

    print(f"\n  integral x^2 / sqrt(9 - x^2) dx")
    print(f"\n  Substitution: x = 3*sin(theta), dx = 3*cos(theta) d(theta)")
    print(f"  sqrt(9 - x^2) = sqrt(9 - 9*sin^2(theta)) = 3*cos(theta)")
    print(f"\n  Substituting:")
    print(f"  integral (9*sin^2(theta)) / (3*cos(theta)) * 3*cos(theta) d(theta)")
    print(f"  = integral 9*sin^2(theta) d(theta)")
    print(f"  = 9 * integral (1 - cos(2*theta))/2 d(theta)")
    print(f"  = (9/2) * (theta - sin(2*theta)/2) + C")
    print(f"  = (9/2) * (theta - sin(theta)*cos(theta)) + C")
    print(f"\n  Back-substitute: sin(theta) = x/3, cos(theta) = sqrt(9-x^2)/3")
    print(f"  theta = arcsin(x/3)")
    print(f"  = (9/2)*arcsin(x/3) - (x/2)*sqrt(9-x^2) + C")

    # SymPy verification
    expr = x**2 / sp.sqrt(9 - x**2)
    result = sp.integrate(expr, x)
    print(f"\n  SymPy result: {result}")
    print(f"  Simplified:   {sp.simplify(result)}")

    # Numerical verification: definite integral from 0 to 3/2
    from scipy import integrate as sci_int
    num_result, _ = sci_int.quad(lambda xv: xv**2 / np.sqrt(9 - xv**2), 0, 1.5)
    sym_def = sp.integrate(expr, (x, 0, sp.Rational(3, 2)))
    print(f"\n  Verification on [0, 3/2]:")
    print(f"    SymPy definite: {float(sym_def):.10f}")
    print(f"    Numerical:      {num_result:.10f}")


# ============================================================
# Problem 5: Improper Integral Convergence
# ============================================================
def exercise_5():
    """
    Determine convergence and evaluate:
    (a) integral_0^inf x * e^(-x) dx
    (b) integral_0^1 1/x^(2/3) dx
    (c) integral_2^inf 1/(x * ln^2(x)) dx
    """
    print("\n" + "=" * 60)
    print("Problem 5: Improper Integral Convergence")
    print("=" * 60)

    x = sp.Symbol('x')

    # (a) integral_0^inf x * e^(-x) dx
    # This is Gamma(2) = 1! = 1
    # Or IBP: u=x, dv=e^(-x)dx => = -x*e^(-x)|_0^inf + integral_0^inf e^(-x) dx
    # = 0 + [-e^(-x)]_0^inf = 0 - (-1) = 1
    expr_a = x * sp.exp(-x)
    result_a = sp.integrate(expr_a, (x, 0, sp.oo))
    print(f"\n(a) integral_0^inf x * e^(-x) dx")
    print(f"    IBP: u = x, dv = e^(-x) dx")
    print(f"    = [-x*e^(-x)]_0^inf + integral_0^inf e^(-x) dx")
    print(f"    = 0 + [-e^(-x)]_0^inf = 0 + 1 = 1")
    print(f"    (Also = Gamma(2) = 1! = 1)")
    print(f"    SymPy result: {result_a}")
    print(f"    CONVERGES to 1")

    # (b) integral_0^1 1/x^(2/3) dx
    # This is a Type II improper integral (integrand singular at x=0)
    # integral_0^1 x^(-2/3) dx = [x^(1/3) / (1/3)]_0^1 = 3*x^(1/3)|_0^1 = 3 - 0 = 3
    expr_b = 1 / x**sp.Rational(2, 3)
    result_b = sp.integrate(expr_b, (x, 0, 1))
    print(f"\n(b) integral_0^1 1/x^(2/3) dx")
    print(f"    Type II: integrand singular at x = 0")
    print(f"    integral x^(-2/3) dx = x^(1/3)/(1/3) = 3*x^(1/3)")
    print(f"    [3*x^(1/3)]_0^1 = 3*1 - 3*0 = 3")
    print(f"    SymPy result: {result_b}")
    print(f"    CONVERGES to 3")

    # (c) integral_2^inf 1/(x * ln^2(x)) dx
    # Let u = ln(x), du = 1/x dx
    # = integral_{ln2}^inf 1/u^2 du = [-1/u]_{ln2}^inf = 0 - (-1/ln2) = 1/ln2
    expr_c = 1 / (x * sp.ln(x)**2)
    result_c = sp.integrate(expr_c, (x, 2, sp.oo))
    print(f"\n(c) integral_2^inf 1/(x * ln^2(x)) dx")
    print(f"    Let u = ln(x), du = dx/x")
    print(f"    = integral_{{ln2}}^inf u^(-2) du")
    print(f"    = [-1/u]_{{ln2}}^inf = 0 + 1/ln(2)")
    print(f"    = 1/ln(2) = {float(1/sp.ln(2)):.10f}")
    print(f"    SymPy result: {result_c} = {float(result_c):.10f}")
    print(f"    CONVERGES to 1/ln(2)")

    # Numerical verification
    from scipy import integrate as sci_int
    num_a, _ = sci_int.quad(lambda xv: xv * np.exp(-xv), 0, np.inf)
    num_b, _ = sci_int.quad(lambda xv: xv**(-2/3), 1e-15, 1)
    num_c, _ = sci_int.quad(lambda xv: 1/(xv * np.log(xv)**2), 2, np.inf)
    print(f"\n  Numerical verification (scipy.quad):")
    print(f"    (a) {num_a:.10f}")
    print(f"    (b) {num_b:.10f}")
    print(f"    (c) {num_c:.10f}")


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
    print("All exercises for Lesson 05 completed.")
    print("=" * 60)
