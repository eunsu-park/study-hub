"""
Exercises for Lesson 05: Stability Analysis
Topic: Control_Theory
Solutions to practice problems from the lesson.
"""

import numpy as np


def routh_array(coeffs):
    """
    Construct the Routh array for a polynomial with given coefficients.
    coeffs: list of coefficients [a_n, a_{n-1}, ..., a_1, a_0]
    Returns the Routh array as a list of rows.
    """
    n = len(coeffs) - 1  # degree
    # Number of columns needed
    ncols = (n + 2) // 2

    # Initialize array
    array = np.zeros((n + 1, ncols))

    # Fill first two rows
    array[0, :len(coeffs[0::2])] = coeffs[0::2]
    array[1, :len(coeffs[1::2])] = coeffs[1::2]

    # Compute remaining rows
    for i in range(2, n + 1):
        for j in range(ncols - 1):
            if array[i-1, 0] == 0:
                array[i-1, 0] = 1e-10  # epsilon replacement
            a = array[i-1, 0]
            b = array[i-2, j+1]
            c = array[i-2, 0]
            d = array[i-1, j+1]
            array[i, j] = (a * b - c * d) / a

    return array


def print_routh(coeffs, label=""):
    """Print the Routh array with row labels."""
    n = len(coeffs) - 1
    array = routh_array(coeffs)

    if label:
        print(f"  {label}")
    for i in range(n + 1):
        row_label = f"s^{n-i}"
        row_vals = [f"{v:10.4f}" for v in array[i] if abs(v) > 1e-15 or i < 2]
        print(f"    {row_label:5s}: {' '.join(row_vals)}")

    # Count sign changes in first column
    first_col = array[:, 0]
    sign_changes = sum(1 for i in range(len(first_col)-1)
                       if first_col[i] * first_col[i+1] < 0)

    return array, sign_changes


def exercise_1():
    """
    Exercise 1: Routh-Hurwitz Application
    Determine stability for three characteristic polynomials.
    """
    # 1. s^4 + 3s^3 + 5s^2 + 4s + 2
    print("Polynomial 1: s^4 + 3s^3 + 5s^2 + 4s + 2")
    coeffs1 = [1, 3, 5, 4, 2]
    arr1, sc1 = print_routh(coeffs1)
    print(f"  First column: {arr1[:, 0]}")
    print(f"  Sign changes: {sc1}")
    print(f"  Stability: {'STABLE' if sc1 == 0 else f'UNSTABLE ({sc1} RHP poles)'}")

    # Verify with numpy
    roots1 = np.roots(coeffs1)
    print(f"  Verification - roots: {np.round(roots1, 4)}")

    # 2. s^4 + s^3 + 2s^2 + 2s + 1
    print("\nPolynomial 2: s^4 + s^3 + 2s^2 + 2s + 1")
    coeffs2 = [1, 1, 2, 2, 1]
    arr2, sc2 = print_routh(coeffs2)
    print(f"  First column: {arr2[:, 0]}")
    print(f"  Sign changes: {sc2}")
    # Note: may have zero row issue or imaginary axis roots
    roots2 = np.roots(coeffs2)
    print(f"  Verification - roots: {np.round(roots2, 4)}")
    has_rhp = any(r.real > 1e-10 for r in roots2)
    has_imag = any(abs(r.real) < 1e-6 and abs(r.imag) > 1e-6 for r in roots2)
    if has_rhp:
        print("  UNSTABLE (poles in RHP)")
    elif has_imag:
        print("  MARGINALLY STABLE (poles on imaginary axis)")
    else:
        print("  STABLE")

    # 3. s^5 + 2s^4 + 3s^3 + 6s^2 + 2s + 1
    print("\nPolynomial 3: s^5 + 2s^4 + 3s^3 + 6s^2 + 2s + 1")
    coeffs3 = [1, 2, 3, 6, 2, 1]
    arr3, sc3 = print_routh(coeffs3)
    print(f"  First column: {arr3[:, 0]}")
    print(f"  Sign changes: {sc3}")
    print(f"  Stability: {'STABLE' if sc3 == 0 else f'UNSTABLE ({sc3} RHP poles)'}")
    roots3 = np.roots(coeffs3)
    print(f"  Verification - roots: {np.round(roots3, 4)}")


def exercise_2():
    """
    Exercise 2: Gain Range
    G(s) = K(s+2) / [s(s+1)(s+3)(s+4)], unity feedback.
    """
    print("G(s) = K(s+2) / [s(s+1)(s+3)(s+4)]")

    # Characteristic equation: s(s+1)(s+3)(s+4) + K(s+2) = 0
    # Expand s(s+1)(s+3)(s+4):
    # s(s+1) = s^2 + s
    # (s+3)(s+4) = s^2 + 7s + 12
    # (s^2 + s)(s^2 + 7s + 12) = s^4 + 7s^3 + 12s^2 + s^3 + 7s^2 + 12s
    #                           = s^4 + 8s^3 + 19s^2 + 12s
    # Add K(s+2) = Ks + 2K
    # Char eq: s^4 + 8s^3 + 19s^2 + (12+K)s + 2K = 0

    print("\nCharacteristic equation:")
    print("  s^4 + 8s^3 + 19s^2 + (12+K)s + 2K = 0")

    print("\nPart 1: Routh array")
    print("  s^4:  1          19          2K")
    print("  s^3:  8          12+K")
    print("  s^2:  (8*19 - 1*(12+K))/8 = (140-K)/8      2K")
    print("  s^1:  [(140-K)/8 * (12+K) - 8*2K] / [(140-K)/8]")
    print("  s^0:  2K")

    print("\nFor stability, all first-column entries > 0:")
    print("  1 > 0: always true")
    print("  8 > 0: always true")
    print("  (140-K)/8 > 0  =>  K < 140")
    print("  s^1 entry > 0: need to compute")
    print("  2K > 0  =>  K > 0")

    # s^1 entry: [(140-K)(12+K) - 128K] / (140-K)
    # Numerator: (140-K)(12+K) - 128K
    # = 1680 + 140K - 12K - K^2 - 128K
    # = 1680 + 0K - K^2
    # = 1680 - K^2
    # Wait, let me redo: (140-K)(12+K) = 1680 + 140K - 12K - K^2 = 1680 + 128K - K^2
    # Then subtract 128K: 1680 + 128K - K^2 - 128K = 1680 - K^2

    # Actually the s^1 computation:
    # row s^2: b1 = (140-K)/8, b2 = 2K
    # row s^3: a = 8, b = 12+K
    # row s^1: c1 = [b1*(12+K) - 8*2K] / b1
    #            = [(140-K)/8 * (12+K) - 16K] / [(140-K)/8]
    #            = [(140-K)(12+K) - 128K] / (140-K)
    #            = [1680 + 128K - K^2 - 128K] / (140-K)
    #            = (1680 - K^2) / (140-K)

    print("\n  s^1 entry = (1680 - K^2) / (140 - K)")
    print("  For K > 0 and K < 140:")
    print("  Need 1680 - K^2 > 0  =>  K^2 < 1680  =>  K < sqrt(1680)")

    K_crit = np.sqrt(1680)
    print(f"  K < sqrt(1680) = {K_crit:.4f}")
    print(f"\n  Stability range: 0 < K < {K_crit:.4f}")

    # Part 2: Frequency of oscillation at critical K
    print(f"\nPart 2: Frequency of oscillation at K = {K_crit:.4f}")
    print("  At the critical K, the s^1 row becomes zero.")
    print("  Form auxiliary polynomial from s^2 row:")
    K = K_crit
    b1 = (140 - K) / 8
    b2 = 2 * K
    print(f"  Auxiliary polynomial: {b1:.4f}*s^2 + {b2:.4f} = 0")
    omega_sq = b2 / b1
    omega = np.sqrt(omega_sq)
    print(f"  s^2 = -{b2:.4f}/{b1:.4f} = -{omega_sq:.4f}")
    print(f"  s = +/- j*{omega:.4f}")
    print(f"  Frequency of oscillation: omega = {omega:.4f} rad/s")

    # Verify
    print(f"\n  Verification: roots at K = {K_crit:.2f}")
    coeffs = [1, 8, 19, 12 + K_crit, 2 * K_crit]
    roots = np.roots(coeffs)
    print(f"  Roots: {np.round(roots, 4)}")


def exercise_3():
    """
    Exercise 3: Relative Stability
    Characteristic polynomial: s^3 + 10s^2 + 31s + 30
    """
    coeffs = [1, 10, 31, 30]

    print("Characteristic polynomial: s^3 + 10s^2 + 31s + 30")

    # Part 1: Verify stability
    print("\nPart 1: Routh-Hurwitz stability check")
    arr, sc = print_routh(coeffs)
    print(f"  First column: {arr[:, 0]}")
    print(f"  Sign changes: {sc}")
    print(f"  System is {'STABLE' if sc == 0 else 'UNSTABLE'}")

    # Also check necessary condition: all coefficients positive
    print(f"  All coefficients positive: {all(c > 0 for c in coeffs)}")

    # Part 2: Check if all poles have Re(p) < -1
    print("\nPart 2: Relative stability (Re(p) < -1)")
    print("  Substitute s = s_hat - 1 into the polynomial:")
    print("  (s_hat-1)^3 + 10(s_hat-1)^2 + 31(s_hat-1) + 30")
    print("  = s_hat^3 - 3s_hat^2 + 3s_hat - 1")
    print("    + 10s_hat^2 - 20s_hat + 10")
    print("    + 31s_hat - 31 + 30")
    print("  = s_hat^3 + 7s_hat^2 + 14s_hat + 8")

    shifted_coeffs = [1, 7, 14, 8]
    print(f"\n  Shifted polynomial: s_hat^3 + 7s_hat^2 + 14s_hat + 8")
    arr_shifted, sc_shifted = print_routh(shifted_coeffs)
    print(f"  First column: {arr_shifted[:, 0]}")
    print(f"  Sign changes: {sc_shifted}")

    if sc_shifted == 0:
        print("  All poles of shifted polynomial in LHP => all original poles have Re(p) < -1")
    else:
        print("  Some poles NOT satisfying Re(p) < -1")

    # Part 3: Find actual poles
    print("\nPart 3: Actual poles")
    roots = np.roots(coeffs)
    print(f"  Roots: {np.round(roots, 6)}")
    for r in roots:
        print(f"    s = {r:.6f}, Re(s) = {r.real:.6f} {'< -1' if r.real < -1 else '>= -1'}")

    # The polynomial factors as (s+1)(s+2)(s+5)... let's check
    # 1*2*5 = 10 != 30; try (s+1)(s+3)(s+10)?  1*3*10=30, 1+3+10=14 != 10
    # try (s+1)(s+5)(s+6)? 1*5*6=30, 1+5+6=12!=10
    # try (s+2)(s+3)(s+5)? 2*3*5=30, 2+3+5=10, 2*3+2*5+3*5=6+10+15=31. Yes!
    print(f"  Factored form: (s+2)(s+3)(s+5)")
    print(f"  All poles at s = -2, -3, -5 satisfy Re(p) < -1")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Routh-Hurwitz Application ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Gain Range ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Relative Stability ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
