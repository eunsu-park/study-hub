"""
Exercise Solutions: Lesson 07 - Sequences and Series
Calculus and Differential Equations

Topics covered:
- Sequence convergence analysis
- Series convergence tests (ratio, integral, alternating)
- Radius and interval of convergence
- Taylor series derivation (arctan, pi via Leibniz)
- Taylor polynomial error bound
"""

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Sequence Convergence
# ============================================================
def exercise_1():
    """
    (a) a_n = (n^2 + 3n) / (2n^2 - 1)
    (b) a_n = (-1)^n * n / (n+1)
    (c) a_n = (1 + 3/n)^n
    """
    print("=" * 60)
    print("Problem 1: Sequence Convergence")
    print("=" * 60)

    n = sp.Symbol('n', positive=True)

    # (a) a_n = (n^2 + 3n) / (2n^2 - 1)
    a_n_a = (n**2 + 3*n) / (2*n**2 - 1)
    lim_a = sp.limit(a_n_a, n, sp.oo)
    print(f"\n(a) a_n = (n^2 + 3n) / (2n^2 - 1)")
    print(f"    Divide by n^2: (1 + 3/n) / (2 - 1/n^2)")
    print(f"    As n -> inf: 1/2")
    print(f"    SymPy: lim = {lim_a}")
    print(f"    CONVERGES to 1/2")

    # Numerical check
    vals_a = [(n_val, float(a_n_a.subs(n, n_val))) for n_val in [10, 100, 1000, 10000]]
    for nv, av in vals_a:
        print(f"    a_{nv} = {av:.8f}")

    # (b) a_n = (-1)^n * n / (n+1)
    a_n_b = (-1)**n * n / (n + 1)
    lim_b = sp.limit(a_n_b, n, sp.oo)
    print(f"\n(b) a_n = (-1)^n * n / (n+1)")
    print(f"    |a_n| = n/(n+1) -> 1, but the sign alternates")
    print(f"    SymPy limit: {lim_b}")
    print(f"    The sequence oscillates between values near +1 and -1")
    print(f"    DIVERGES (limit does not exist)")

    # Numerical check
    for nv in [10, 11, 100, 101, 1000, 1001]:
        val = (-1)**nv * nv / (nv + 1)
        print(f"    a_{nv} = {val:.8f}")

    # (c) a_n = (1 + 3/n)^n
    a_n_c = (1 + sp.Rational(3, 1)/n)**n
    lim_c = sp.limit(a_n_c, n, sp.oo)
    print(f"\n(c) a_n = (1 + 3/n)^n")
    print(f"    This is of the form (1 + k/n)^n -> e^k as n -> inf")
    print(f"    With k = 3: limit = e^3")
    print(f"    SymPy: lim = {lim_c} = {float(lim_c):.10f}")
    print(f"    CONVERGES to e^3")

    vals_c = [float((1 + 3.0/nv)**nv) for nv in [10, 100, 1000, 10000]]
    for nv, vc in zip([10, 100, 1000, 10000], vals_c):
        print(f"    a_{nv} = {vc:.10f}")


# ============================================================
# Problem 2: Series Convergence Tests
# ============================================================
def exercise_2():
    """
    (a) sum n^2/3^n (ratio test)
    (b) sum 1/(n*ln(n)) for n>=2 (integral test)
    (c) sum (-1)^n / sqrt(n) (alternating series test)
    (d) sum n!/n^n (ratio test)
    """
    print("\n" + "=" * 60)
    print("Problem 2: Series Convergence Tests")
    print("=" * 60)

    n = sp.Symbol('n', positive=True, integer=True)

    # (a) sum n^2/3^n: Ratio test
    a_n = n**2 / 3**n
    ratio = sp.simplify((n+1)**2 / 3**(n+1) * 3**n / n**2)
    ratio_limit = sp.limit(ratio, n, sp.oo)
    print(f"\n(a) sum_{{n=1}}^inf n^2/3^n")
    print(f"    Ratio test: a_{{n+1}}/a_n = (n+1)^2 / (3*n^2)")
    print(f"    lim = {ratio_limit}")
    print(f"    Since 1/3 < 1, the series CONVERGES (by ratio test)")

    # Compute partial sums numerically
    partial = sum(k**2 / 3**k for k in range(1, 200))
    # Exact sum (using sympy): sum = 3/2 (can be derived from geometric series derivatives)
    exact_sum = sp.summation(n**2 / 3**n, (n, 1, sp.oo))
    print(f"    Sum = {exact_sum} = {float(exact_sum):.10f}")
    print(f"    Partial sum (200 terms) = {partial:.10f}")

    # (b) sum 1/(n*ln(n)) for n>=2: Integral test
    x = sp.Symbol('x', positive=True)
    integral_test = sp.integrate(1/(x*sp.ln(x)), (x, 2, sp.oo))
    print(f"\n(b) sum_{{n=2}}^inf 1/(n*ln(n))")
    print(f"    Integral test: integral_2^inf 1/(x*ln(x)) dx")
    print(f"    Let u = ln(x), du = dx/x")
    print(f"    = integral_{{ln2}}^inf 1/u du = ln(u)|_{{ln2}}^inf = inf")
    print(f"    SymPy: {integral_test}")
    print(f"    The integral DIVERGES, so the series DIVERGES")

    # (c) sum (-1)^n / sqrt(n): Alternating series test
    print(f"\n(c) sum_{{n=1}}^inf (-1)^n / sqrt(n)")
    print(f"    Alternating series test:")
    print(f"    b_n = 1/sqrt(n)")
    print(f"    (i)  b_n > 0 for all n >= 1  [YES]")
    print(f"    (ii) b_{{n+1}} = 1/sqrt(n+1) < 1/sqrt(n) = b_n  [YES, decreasing]")
    print(f"    (iii) lim b_n = lim 1/sqrt(n) = 0  [YES]")
    print(f"    All conditions satisfied => CONVERGES (conditionally)")
    print(f"    Note: sum 1/sqrt(n) diverges (p-series, p=1/2 < 1),")
    print(f"    so convergence is conditional, not absolute.")

    # (d) sum n!/n^n: Ratio test
    print(f"\n(d) sum_{{n=1}}^inf n!/n^n")
    print(f"    Ratio test: a_{{n+1}}/a_n = (n+1)!/(n+1)^(n+1) * n^n/n!")
    print(f"    = n^n / (n+1)^n = (n/(n+1))^n = (1 - 1/(n+1))^n")
    print(f"    lim = e^(-1) = 1/e")
    ratio_d = sp.limit((1 - 1/(n+1))**n, n, sp.oo)
    print(f"    SymPy: lim = {ratio_d} = {float(ratio_d):.10f}")
    print(f"    Since 1/e < 1, the series CONVERGES (by ratio test)")


# ============================================================
# Problem 3: Radius of Convergence
# ============================================================
def exercise_3():
    """
    (a) sum (x-3)^n / (n * 2^n) for n>=1
    (b) sum n! * x^n / n^n for n>=1
    """
    print("\n" + "=" * 60)
    print("Problem 3: Radius of Convergence")
    print("=" * 60)

    n, x = sp.symbols('n x')

    # (a) sum (x-3)^n / (n * 2^n)
    # Ratio test: |a_{n+1}/a_n| = |x-3|/2 * n/(n+1) -> |x-3|/2
    # Converges when |x-3|/2 < 1 => |x-3| < 2 => R = 2
    # Interval: 1 < x < 5
    # At x=1: sum (-1)^n/n, alternating harmonic => converges
    # At x=5: sum 1/n, harmonic => diverges
    print(f"\n(a) sum_{{n=1}}^inf (x-3)^n / (n * 2^n)")
    print(f"    Ratio test: |a_{{n+1}}/a_n| = |x-3|/2 * n/(n+1)")
    print(f"    lim = |x-3|/2")
    print(f"    Converges when |x-3|/2 < 1  =>  |x-3| < 2")
    print(f"    Radius of convergence R = 2")
    print(f"    Center = 3, so interval candidates: (1, 5)")
    print(f"\n    Check endpoints:")
    print(f"    x = 1: sum (-1)^n/n = -ln(2) (alternating harmonic, CONVERGES)")
    print(f"    x = 5: sum 1/n (harmonic series, DIVERGES)")
    print(f"    Interval of convergence: [1, 5)")

    # (b) sum n! * x^n / n^n
    # Ratio test: |a_{n+1}/a_n| = |x| * (n+1)! * n^n / ((n+1)^(n+1) * n!)
    # = |x| * (n+1) * n^n / (n+1)^(n+1) = |x| * (n/(n+1))^n -> |x|/e
    # Converges when |x|/e < 1 => |x| < e => R = e
    print(f"\n(b) sum_{{n=1}}^inf n! * x^n / n^n")
    print(f"    Ratio test: |a_{{n+1}}/a_n| = |x| * (n/(n+1))^n")
    print(f"    lim = |x| * 1/e = |x|/e")
    print(f"    Converges when |x|/e < 1  =>  |x| < e")
    print(f"    Radius of convergence R = e = {float(sp.E):.10f}")
    print(f"    Interval of convergence: (-e, e)")
    print(f"    (Endpoint behavior requires careful analysis with Stirling's approximation)")


# ============================================================
# Problem 4: Taylor Series Derivation
# ============================================================
def exercise_4():
    """
    (a) Derive Maclaurin series for 1/(1+x^2)
    (b) Integrate to get arctan(x) series
    (c) Derive Leibniz formula for pi/4
    (d) Compute pi using the series
    """
    print("\n" + "=" * 60)
    print("Problem 4: Taylor Series Derivation")
    print("=" * 60)

    x = sp.Symbol('x')

    # (a) 1/(1+x^2) from geometric series
    print(f"\n(a) Maclaurin series for 1/(1+x^2):")
    print(f"    Geometric series: 1/(1-u) = sum u^n for |u| < 1")
    print(f"    Substitute u = -x^2:")
    print(f"    1/(1+x^2) = sum_{{n=0}}^inf (-x^2)^n = sum (-1)^n * x^(2n)")
    print(f"              = 1 - x^2 + x^4 - x^6 + ...  for |x| < 1")

    # Verify with SymPy
    series_verify = sp.series(1/(1+x**2), x, 0, 10)
    print(f"    SymPy verification: {series_verify}")

    # (b) Integrate term by term to get arctan(x)
    print(f"\n(b) Integrating term by term:")
    print(f"    arctan(x) = integral 1/(1+x^2) dx")
    print(f"              = sum_{{n=0}}^inf (-1)^n * x^(2n+1) / (2n+1)")
    print(f"              = x - x^3/3 + x^5/5 - x^7/7 + ...")

    series_arctan = sp.series(sp.atan(x), x, 0, 12)
    print(f"    SymPy verification: {series_arctan}")

    # (c) Leibniz formula
    print(f"\n(c) Leibniz formula for pi:")
    print(f"    arctan(1) = pi/4")
    print(f"    pi/4 = 1 - 1/3 + 1/5 - 1/7 + 1/9 - ...")
    print(f"    pi = 4 * (1 - 1/3 + 1/5 - 1/7 + ...)")

    # (d) Compute pi
    print(f"\n(d) Computing pi using the Leibniz series:")
    print(f"    {'N terms':>10s}  {'pi approx':>18s}  {'error':>14s}")
    print(f"    {'-'*10}  {'-'*18}  {'-'*14}")

    target_accuracy = 1e-6
    found_n = None

    for N in [10, 100, 1000, 10000, 100000, 1000000]:
        pi_approx = 4.0 * sum((-1)**k / (2*k + 1) for k in range(N))
        err = abs(pi_approx - np.pi)
        print(f"    {N:>10d}  {pi_approx:>18.12f}  {err:>14.2e}")
        if err < target_accuracy and found_n is None:
            found_n = N

    if found_n:
        print(f"\n    First 6-digit accuracy at N = {found_n} terms")
    else:
        # The Leibniz series converges very slowly
        # Need about 500000 terms for 6 digits
        # Let's find it more precisely
        pi_est = 0.0
        for k in range(2000000):
            pi_est += 4.0 * (-1)**k / (2*k + 1)
            if abs(pi_est - np.pi) < target_accuracy and found_n is None:
                found_n = k + 1
                break
        print(f"\n    6-digit accuracy requires approximately {found_n} terms")
        print(f"    (The Leibniz series converges very slowly -- O(1/N) error)")


# ============================================================
# Problem 5: Taylor Polynomial Error Bound
# ============================================================
def exercise_5():
    """
    Find degree n so that T_n(0.5) for e^x has error < 10^-8.
    Verify computationally.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Taylor Polynomial Error Bound")
    print("=" * 60)

    x_val = 0.5

    # Taylor remainder: |R_n(x)| <= M * |x|^(n+1) / (n+1)!
    # where M = max |f^(n+1)(c)| on [0, x]
    # For e^x on [0, 0.5]: M = e^0.5 < 2 (since e^0.5 ~ 1.6487)
    print(f"\n  Taylor remainder theorem:")
    print(f"  |R_n(x)| <= M * |x-a|^(n+1) / (n+1)!")
    print(f"  For e^x centered at a=0, evaluated at x=0.5:")
    print(f"  M = max e^c on [0, 0.5] = e^0.5 < 2")
    print(f"  |R_n(0.5)| <= 2 * (0.5)^(n+1) / (n+1)!")
    print(f"             = 2 / (2^(n+1) * (n+1)!)")
    print(f"             = 1 / (2^n * (n+1)!)")

    # Find n such that 1/(2^n * (n+1)!) < 10^-8
    print(f"\n  Finding n such that 1/(2^n * (n+1)!) < 10^-8:")
    for n in range(1, 25):
        bound = 1.0 / (2**n * np.math.factorial(n + 1))
        sufficient = bound < 1e-8
        print(f"    n = {n:2d}: bound = {bound:.2e}  {'<-- sufficient!' if sufficient else ''}")
        if sufficient:
            n_required = n
            break

    print(f"\n  Need n >= {n_required} for guaranteed error < 10^-8")

    # Verify computationally
    print(f"\n  Computational verification:")
    exact = np.exp(x_val)
    print(f"  Exact e^0.5 = {exact:.15f}")
    print(f"\n  {'n':>4s}  {'T_n(0.5)':>20s}  {'|error|':>14s}")
    print(f"  {'----':>4s}  {'--------------------':>20s}  {'--------------':>14s}")

    for n in range(1, n_required + 5):
        # T_n(0.5) = sum_{k=0}^{n} (0.5)^k / k!
        T_n = sum(x_val**k / np.math.factorial(k) for k in range(n + 1))
        error = abs(T_n - exact)
        marker = " <--" if n == n_required else ""
        print(f"  {n:>4d}  {T_n:>20.15f}  {error:>14.2e}{marker}")


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
    print("All exercises for Lesson 07 completed.")
    print("=" * 60)
