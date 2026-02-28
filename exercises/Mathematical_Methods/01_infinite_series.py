"""
Exercise Solutions: Lesson 01 - Infinite Series
Mathematical Methods for Physical Sciences

Covers: convergence tests, radius of convergence, Taylor series,
        elliptic integrals, Stirling's approximation, relativistic energy
"""

import numpy as np
from scipy import special, integrate


def exercise_1_convergence_tests():
    """
    Problem 1: Convergence Tests
    Test convergence of:
    (a) sum 1/n^2     (b) sum n/(2^n)     (c) sum (-1)^n / sqrt(n)
    """
    print("=" * 60)
    print("Problem 1: Convergence Tests")
    print("=" * 60)

    # (a) sum_{n=1}^{inf} 1/n^2  -- p-series with p=2 > 1 => converges
    print("\n(a) sum 1/n^2:")
    print("  p-series test: p = 2 > 1 => CONVERGES")
    partial_sums_a = np.cumsum(1.0 / np.arange(1, 10001)**2)
    print(f"  S_100   = {partial_sums_a[99]:.10f}")
    print(f"  S_1000  = {partial_sums_a[999]:.10f}")
    print(f"  S_10000 = {partial_sums_a[9999]:.10f}")
    print(f"  Exact   = pi^2/6 = {np.pi**2 / 6:.10f}")

    # (b) sum_{n=1}^{inf} n/2^n  -- ratio test
    print("\n(b) sum n/2^n:")
    ns = np.arange(1, 101)
    terms_b = ns / 2.0**ns
    # Ratio test: a_{n+1}/a_n = (n+1)/(2n) -> 1/2 < 1 => converges
    ratios = terms_b[1:] / terms_b[:-1]
    print(f"  Ratio test: lim a_{{n+1}}/a_n = {ratios[-1]:.6f} (-> 1/2)")
    print(f"  Since 1/2 < 1, the series CONVERGES")
    partial_sums_b = np.cumsum(terms_b)
    print(f"  S_20  = {partial_sums_b[19]:.10f}")
    print(f"  S_100 = {partial_sums_b[99]:.10f}")
    # Exact: sum n*x^n = x/(1-x)^2, at x=1/2 => (1/2)/(1/2)^2 = 2
    print(f"  Exact = 2.0000000000")

    # (c) sum_{n=1}^{inf} (-1)^n / sqrt(n)  -- alternating series test
    print("\n(c) sum (-1)^n / sqrt(n):")
    print("  Alternating series test:")
    print("    b_n = 1/sqrt(n) is positive, decreasing, and -> 0")
    print("    => CONVERGES (conditionally)")
    ns = np.arange(1, 10001)
    terms_c = (-1.0)**ns / np.sqrt(ns)
    partial_sums_c = np.cumsum(terms_c)
    print(f"  S_100   = {partial_sums_c[99]:.8f}")
    print(f"  S_1000  = {partial_sums_c[999]:.8f}")
    print(f"  S_10000 = {partial_sums_c[9999]:.8f}")
    # Exact: -(1 - sqrt(2)) * zeta(1/2) ~ related to Dirichlet eta function
    # eta(1/2) = (1 - 2^{1-1/2}) * zeta(1/2)
    # Numerically:
    print(f"  (Note: Not absolutely convergent since sum 1/sqrt(n) diverges)")


def exercise_2_radius_of_convergence():
    """
    Problem 2: Radius of Convergence
    Find R for: (a) sum n*x^n  (b) sum x^n/n!  (c) sum n! * x^n
    """
    print("\n" + "=" * 60)
    print("Problem 2: Radius of Convergence")
    print("=" * 60)

    # (a) sum n*x^n: ratio test |a_{n+1}/a_n| = |(n+1)/n| * |x| -> |x|
    print("\n(a) sum n*x^n:")
    print("  Ratio test: |(n+1)x^{n+1}| / |n*x^n| = (n+1)/n * |x| -> |x|")
    print("  Converges when |x| < 1, so R = 1")
    # Verify numerically
    for x_val in [0.5, 0.9, 0.99]:
        ns = np.arange(1, 201)
        partial = np.cumsum(ns * x_val**ns)
        exact = x_val / (1 - x_val)**2
        print(f"  x={x_val}: S_200 = {partial[-1]:.6f}, exact = {exact:.6f}")

    # (b) sum x^n/n!: ratio test |x|/(n+1) -> 0 for all x
    print("\n(b) sum x^n/n!:")
    print("  Ratio test: |x^{n+1}/(n+1)!| / |x^n/n!| = |x|/(n+1) -> 0")
    print("  Converges for all x, so R = infinity")
    print("  (This is the Taylor series for e^x)")
    for x_val in [1.0, 5.0, 10.0]:
        ns = np.arange(0, 51)
        terms = x_val**ns / special.factorial(ns, exact=False)
        partial = np.cumsum(terms)
        print(f"  x={x_val}: S_50 = {partial[-1]:.10f}, e^x = {np.exp(x_val):.10f}")

    # (c) sum n!*x^n: ratio test (n+1)|x| -> infinity for any x != 0
    print("\n(c) sum n!*x^n:")
    print("  Ratio test: |(n+1)!*x^{n+1}| / |n!*x^n| = (n+1)|x| -> infinity")
    print("  Diverges for all x != 0, so R = 0")
    print("  (Converges only at x = 0)")


def exercise_3_taylor_sqrt():
    """
    Problem 3: Taylor Series
    Find Taylor series for sqrt(1+x) around x=0 up to x^4 term.
    Estimate sqrt(1.1) and compare.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Taylor Series of sqrt(1+x)")
    print("=" * 60)

    # sqrt(1+x) = (1+x)^{1/2}
    # Binomial series: (1+x)^p = sum C(p,n) x^n
    # C(p,n) = p(p-1)...(p-n+1)/n!

    p = 0.5
    print("\nBinomial series coefficients for (1+x)^{1/2}:")
    coeffs = []
    for n in range(5):
        if n == 0:
            c = 1.0
        else:
            c = 1.0
            for k in range(n):
                c *= (p - k) / (k + 1)
        coeffs.append(c)
        print(f"  a_{n} = {c:+.6f}")

    print(f"\nsqrt(1+x) = {coeffs[0]:.1f} + {coeffs[1]:.4f}*x "
          f"+ ({coeffs[2]:.6f})*x^2 + ({coeffs[3]:.6f})*x^3 "
          f"+ ({coeffs[4]:.6f})*x^4 + ...")

    # Estimate sqrt(1.1)
    x = 0.1
    estimates = []
    for order in range(1, 6):
        est = sum(coeffs[n] * x**n for n in range(order))
        estimates.append(est)

    exact = np.sqrt(1.1)
    print(f"\nEstimates for sqrt(1.1):")
    for order, est in enumerate(estimates, 1):
        err = abs(est - exact)
        print(f"  Order {order}: {est:.10f}  (error = {err:.2e})")
    print(f"  Exact:   {exact:.10f}")


def exercise_4_pendulum_period():
    """
    Problem 4: Pendulum Period
    The exact period of a simple pendulum: T = 4*sqrt(L/g) * K(sin(theta_0/2))
    where K is the complete elliptic integral of the first kind.
    Compare with the small-angle approximation T_0 = 2*pi*sqrt(L/g).
    """
    print("\n" + "=" * 60)
    print("Problem 4: Pendulum Period with Elliptic Integrals")
    print("=" * 60)

    # T/T_0 = (2/pi) * K(sin(theta_0/2))
    # Series expansion: T/T_0 = 1 + (1/16)*theta_0^2 + (11/3072)*theta_0^4 + ...
    theta_0_degrees = np.array([5, 10, 15, 30, 45, 60, 90])
    theta_0_rad = np.radians(theta_0_degrees)

    print(f"\n{'theta_0 (deg)':>13} | {'T/T_0 (exact)':>14} | {'T/T_0 (2nd order)':>18} | {'Error':>10}")
    print("-" * 65)

    for deg, rad in zip(theta_0_degrees, theta_0_rad):
        k = np.sin(rad / 2)
        # Exact: using complete elliptic integral
        K_val = special.ellipk(k**2)  # scipy uses m = k^2
        T_ratio_exact = (2 / np.pi) * K_val

        # 2nd order series approximation
        T_ratio_approx = 1 + (1/16) * rad**2 + (11/3072) * rad**4

        error = abs(T_ratio_exact - T_ratio_approx) / T_ratio_exact * 100
        print(f"  {deg:10d}   | {T_ratio_exact:14.8f} | {T_ratio_approx:18.8f} | {error:8.4f}%")

    print("\nConclusion: Small-angle approximation error is < 1% for theta_0 < 30 deg")


def exercise_5_stirling():
    """
    Problem 5: Stirling's Approximation
    Use Stirling's formula to estimate log10(100!).
    Stirling: ln(n!) ~ n*ln(n) - n + 0.5*ln(2*pi*n)
    """
    print("\n" + "=" * 60)
    print("Problem 5: Stirling's Approximation for log10(100!)")
    print("=" * 60)

    n = 100

    # Exact value using scipy
    exact_ln = special.gammaln(n + 1)  # ln(n!) = ln(Gamma(n+1))
    exact_log10 = exact_ln / np.log(10)

    # Stirling approximations of increasing order
    # Order 0: ln(n!) ~ n*ln(n) - n
    s0 = n * np.log(n) - n
    # Order 1: + 0.5*ln(2*pi*n)
    s1 = s0 + 0.5 * np.log(2 * np.pi * n)
    # Order 2: + 1/(12n)
    s2 = s1 + 1 / (12 * n)
    # Order 3: - 1/(360*n^3)
    s3 = s2 - 1 / (360 * n**3)

    print(f"\nExact: ln(100!) = {exact_ln:.10f}")
    print(f"Exact: log10(100!) = {exact_log10:.10f}")

    approxs = [
        ("n*ln(n) - n", s0),
        ("+ 0.5*ln(2*pi*n)", s1),
        ("+ 1/(12n)", s2),
        ("- 1/(360n^3)", s3),
    ]

    print(f"\n{'Stirling Order':<25} | {'ln(100!)':<18} | {'log10(100!)':<15} | {'Rel Error':>10}")
    print("-" * 75)
    for label, val in approxs:
        log10_val = val / np.log(10)
        rel_err = abs(val - exact_ln) / exact_ln * 100
        print(f"  {label:<23} | {val:16.8f} | {log10_val:13.8f} | {rel_err:8.6f}%")


def exercise_6_relativistic_energy():
    """
    Problem 6: Relativistic Kinetic Energy
    E_kinetic = mc^2 * (gamma - 1) where gamma = 1/sqrt(1 - v^2/c^2)
    Show that for v << c, E_kinetic ~ (1/2)mv^2 + (3/8)mv^4/c^2 + ...
    """
    print("\n" + "=" * 60)
    print("Problem 6: Relativistic Kinetic Energy Expansion")
    print("=" * 60)

    # gamma = (1 - beta^2)^{-1/2} where beta = v/c
    # Binomial expansion: (1-x)^{-1/2} = 1 + x/2 + 3x^2/8 + 5x^3/16 + ...
    # So gamma - 1 = beta^2/2 + 3*beta^4/8 + 5*beta^6/16 + ...
    # E_kinetic = mc^2 * (gamma - 1)
    #           = (1/2)mv^2 + (3/8)mv^4/c^2 + (5/16)mv^6/c^4 + ...

    print("\nBinomial expansion of gamma - 1:")
    print("  gamma = (1 - beta^2)^{-1/2}")
    print("        = 1 + (1/2)beta^2 + (3/8)beta^4 + (5/16)beta^6 + ...")
    print("  gamma - 1 = (1/2)beta^2 + (3/8)beta^4 + (5/16)beta^6 + ...")
    print("\n  E_kinetic = mc^2*(gamma - 1)")
    print("            = (1/2)mv^2 [1 + (3/4)beta^2 + (5/8)beta^4 + ...]")

    # Numerical comparison
    betas = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99])
    print(f"\n{'beta=v/c':>10} | {'Exact gamma-1':>15} | {'1st order':>12} | {'2nd order':>12} | {'Err 1st':>10} | {'Err 2nd':>10}")
    print("-" * 80)

    for beta in betas:
        exact = 1.0 / np.sqrt(1 - beta**2) - 1
        approx1 = beta**2 / 2
        approx2 = beta**2 / 2 + 3 * beta**4 / 8

        err1 = abs(exact - approx1) / exact * 100 if exact > 0 else 0
        err2 = abs(exact - approx2) / exact * 100 if exact > 0 else 0

        print(f"  {beta:8.4f} | {exact:15.8f} | {approx1:12.8f} | {approx2:12.8f} | {err1:8.4f}% | {err2:8.4f}%")

    print("\nConclusion: Classical approximation (1/2)mv^2 is accurate for v/c < 0.1")


if __name__ == "__main__":
    exercise_1_convergence_tests()
    exercise_2_radius_of_convergence()
    exercise_3_taylor_sqrt()
    exercise_4_pendulum_period()
    exercise_5_stirling()
    exercise_6_relativistic_energy()
