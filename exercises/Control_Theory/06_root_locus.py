"""
Exercises for Lesson 06: Root Locus Method
Topic: Control_Theory
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Exercise 1: Root Locus Sketch
    G(s) = K / [s(s+2)(s+4)]
    """
    print("G(s) = K / [s(s+2)(s+4)]")

    # Open-loop poles and zeros
    poles = np.array([0, -2, -4])
    n = len(poles)  # 3
    m = 0  # no finite zeros

    print(f"\nOpen-loop poles: {poles}")
    print(f"Number of poles n = {n}, zeros m = {m}")

    # Rule 1: Branches start at poles, end at zeros (3 at infinity)
    print("\nRule 1: 3 branches start at s = 0, -2, -4")
    print("  All 3 branches go to infinity (no finite zeros)")

    # Rule 4: Real-axis segments
    print("\nRule 4: Real-axis segments")
    print("  s < -4: 3 poles to the right => odd => ON locus")
    print("  -4 < s < -2: 2 poles to the right => even => NOT on locus")
    print("  -2 < s < 0: 1 pole to the right => odd => ON locus")
    print("  s > 0: 0 poles to the right => even => NOT on locus")
    print("  Segments: (-inf, -4) and (-2, 0)")

    # Rule 5: Asymptotes
    sigma_a = sum(poles) / (n - m)
    angles = [(2*k + 1) * 180 / (n - m) for k in range(n - m)]
    print(f"\nRule 5: Asymptotes")
    print(f"  Centroid: sigma_a = (0 + (-2) + (-4)) / 3 = {sigma_a:.4f}")
    print(f"  Angles: {angles} degrees")

    # Rule 6: Breakaway points
    # K = -s(s+2)(s+4) = -(s^3 + 6s^2 + 8s)
    # dK/ds = -(3s^2 + 12s + 8) = 0
    # 3s^2 + 12s + 8 = 0
    disc = 144 - 96
    s_break = (-12 + np.array([np.sqrt(disc), -np.sqrt(disc)])) / 6
    print(f"\nRule 6: Breakaway/break-in points")
    print(f"  dK/ds = 0 => 3s^2 + 12s + 8 = 0")
    print(f"  s = {s_break}")

    for sb in s_break:
        on_locus = (-2 < sb < 0) or (sb < -4)
        K_at_point = -sb * (sb + 2) * (sb + 4)
        print(f"  s = {sb:.4f}: on locus = {on_locus}, K = {K_at_point:.4f}")

    # Rule 8: Imaginary axis crossings (Routh criterion)
    # Characteristic equation: s^3 + 6s^2 + 8s + K = 0
    print("\nRule 8: Imaginary axis crossings")
    print("  Char. eq: s^3 + 6s^2 + 8s + K = 0")
    print("  Routh array:")
    print("    s^3:  1     8")
    print("    s^2:  6     K")
    print("    s^1:  (48-K)/6")
    print("    s^0:  K")
    print("  For stability: K > 0 and (48-K)/6 > 0 => 0 < K < 48")

    K_crit = 48
    print(f"\n  Critical K = {K_crit}")
    print(f"  At K = {K_crit}: auxiliary polynomial from s^2 row: 6s^2 + {K_crit} = 0")
    omega = np.sqrt(K_crit / 6)
    print(f"  s = +/- j*sqrt({K_crit}/6) = +/- j*{omega:.4f}")

    # Part 2: Value of K at imaginary axis crossing
    print(f"\nPart 2: K at imaginary axis crossing = {K_crit}")
    print(f"  Crossing frequency: omega = {omega:.4f} rad/s")

    # Part 3: Range of K for stability
    print(f"\nPart 3: Stability range: 0 < K < {K_crit}")

    # Plot root locus
    num = [1]
    den = np.polymul([1, 0], np.polymul([1, 2], [1, 4]))
    sys = signal.TransferFunction(num, den)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute root locus manually for various K values
    K_values = np.concatenate([
        np.linspace(0.01, 48, 500),
        np.linspace(48, 200, 200)
    ])

    roots_list = []
    for K in K_values:
        char_poly = np.polyadd(den, K * np.array(num))
        char_poly = np.pad(num, (len(den) - len(num), 0)) * K
        char_poly = den.copy().astype(float)
        char_poly[-1] += K
        roots_list.append(np.roots(char_poly))

    roots_array = np.array(roots_list)

    for i in range(n):
        ax.plot(roots_array[:, i].real, roots_array[:, i].imag, 'b.', markersize=1)

    ax.plot(poles.real, np.zeros(n), 'rx', markersize=12, markeredgewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Real Axis')
    ax.set_ylabel('Imaginary Axis')
    ax.set_title('Root Locus: G(s) = K/[s(s+2)(s+4)]')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-10, 2])
    ax.set_ylim([-6, 6])

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex06_root_locus.png',
                dpi=100)
    plt.close()
    print("\n  Root locus plot saved to 'ex06_root_locus.png'")


def exercise_2():
    """
    Exercise 2: Design with Root Locus
    G(s) = K(s+3) / [s(s+1)(s+5)]
    """
    print("G(s) = K(s+3) / [s(s+1)(s+5)]")
    poles = np.array([0, -1, -5])
    zeros = np.array([-3])
    n, m = 3, 1

    print(f"\nPoles: {poles}")
    print(f"Zeros: {zeros}")
    print(f"n - m = {n-m} = 2 branches go to infinity")

    # Asymptotes
    sigma_a = (sum(poles) - sum(zeros)) / (n - m)
    angles_a = [(2*k + 1) * 180 / (n - m) for k in range(n - m)]
    print(f"\nAsymptotes:")
    print(f"  Centroid: ({sum(poles)} - {sum(zeros)}) / {n-m} = {sigma_a:.2f}")
    print(f"  Angles: {angles_a}")

    # Real-axis segments
    print("\nReal-axis segments:")
    print("  s < -5: 3 poles + 1 zero = 4 to right => even => NOT on locus")
    print("  -5 < s < -3: 2 poles + 1 zero = 3 to right => odd => ON locus")
    print("  -3 < s < -1: 2 poles + 0 zeros = 2 to right => even => NOT on locus")
    print("  -1 < s < 0: 1 pole to right => odd => ON locus")
    print("  s > 0: 0 to right => even => NOT on locus")

    # Part 2: Find K for closed-loop pole at s = -2
    print("\nPart 2: K for closed-loop pole at s = -2")
    s_d = -2
    # Angle condition: sum(angles from zeros) - sum(angles from poles) = 180 + 360k
    # Check if s = -2 is on the locus
    # s = -2 is between -3 and -1, which is NOT on the real-axis locus
    # So s = -2 is NOT on the real axis locus.
    # But the problem says "at s = -2", let's check with magnitude condition.

    # Actually, re-checking: s=-2 is between -3 and -1, count poles+zeros to right:
    # poles at 0 and -1 (2 poles), zero at -3 is to the LEFT. So 2 to the right => even => NOT on locus
    # This means s = -2 is not on the root locus.
    # BUT the problem asks to find K -- perhaps it's asking for the characteristic equation:
    # s(s+1)(s+5) + K(s+3) = 0, plug s=-2
    # (-2)(-1)(3) + K(1) = 0
    # -6 + K = 0 => K = 6

    # Wait, (-2)(-2+1)(-2+5) + K(-2+3) = 0
    # (-2)(-1)(3) + K(1) = 0
    # 6 + K = 0 => K = -6
    # Negative K? Let me recheck.
    # s(s+1)(s+5) = (-2)(-1)(3) = 6
    # K(s+3) = K(1)
    # 6 + K = 0 => K = -6

    # This means s = -2 is on the COMPLEMENTARY root locus (K < 0), not the standard one.
    # Let me check the angle condition more carefully for s = -2.
    # Actually, the problem might still be valid -- the characteristic equation is:
    # 1 + KG(s) = 0 => K = -1/G(s)
    # G(-2) = (-2+3)/[(-2)(-2+1)(-2+5)] = 1/[(-2)(-1)(3)] = 1/6
    # K = -1/G(-2) = -6

    # For standard root locus K > 0, s = -2 is not on the locus.
    # The problem likely intends us to verify this. But let's solve as stated.

    # Re-reading: Perhaps the characteristic equation approach:
    # delta(s) = s(s+1)(s+5) + K(s+3) = 0
    # At s = -2: (-2)(-1)(3) + K(1) = 6 + K = 0 => K = -6
    # Since K must be positive for standard root locus, let me reconsider.

    # Actually for the equation s^3 + 6s^2 + 5s + K(s+3) = 0:
    # s^3 + 6s^2 + (5+K)s + 3K = 0
    # At s = -2: -8 + 24 + (-2)(5+K) + 3K = 0
    # -8 + 24 - 10 - 2K + 3K = 0
    # 6 + K = 0 => K = -6

    # So the problem has no positive K solution at exactly s = -2.
    # But perhaps the problem means we should find K that places a pole
    # closest to s = -2. Let's just solve analytically and note the result.

    # Let me just use the magnitude condition approach:
    K_val = -(-2) * (-2+1) * (-2+5) / (-2+3)
    print(f"  Characteristic equation: s(s+1)(s+5) + K(s+3) = 0")
    print(f"  At s = -2: (-2)(-1)(3) + K(1) = 0")
    print(f"  6 + K = 0 => K = -6")
    print(f"  Since K = -6 < 0, s = -2 is NOT on the standard root locus (K > 0).")
    print(f"  s = -2 lies on the complementary root locus (K < 0).")
    print()
    print(f"  For the standard locus, let us find the K that places a pole")
    print(f"  at s = -2 + j*omega for some omega.")
    print(f"  Using magnitude condition: K = |s(s+1)(s+5)| / |s+3|")

    # Let's find a real pole on the locus segment (-1, 0)
    # At breakaway near s = -0.5 (example)
    s_test = -0.5
    K_test = abs(s_test * (s_test + 1) * (s_test + 5)) / abs(s_test + 3)
    print(f"\n  Example: At s = {s_test}: K = {K_test:.4f}")

    # Part 3: All closed-loop poles at K = -6
    print(f"\nPart 3: All closed-loop poles at K = -6 (complementary locus)")
    char_coeffs = [1, 6, 5 + (-6), 3 * (-6)]
    # = [1, 6, -1, -18]
    cl_poles = np.roots(char_coeffs)
    print(f"  Char. polynomial: s^3 + 6s^2 - s - 18 = 0")
    print(f"  Closed-loop poles: {np.round(cl_poles, 4)}")

    # For positive K, let's also show poles
    print(f"\n  For reference, closed-loop poles at K = 6 (positive):")
    char_coeffs_pos = [1, 6, 5 + 6, 3 * 6]
    cl_poles_pos = np.roots(char_coeffs_pos)
    print(f"  Char. polynomial: s^3 + 6s^2 + 11s + 18 = 0")
    print(f"  Closed-loop poles: {np.round(cl_poles_pos, 4)}")


def exercise_3():
    """
    Exercise 3: Effect of Compensation
    G(s) = K/[s(s+2)] with zero at s=-1 or s=-5
    """
    print("Base system: G(s) = K/[s(s+2)]")
    print("Comparing addition of zero at s = -1 vs s = -5")

    # Case 1: zero at s = -1 (between poles)
    print("\nCase 1: G(s) = K(s+1)/[s(s+2)]")
    print("  Poles: s = 0, -2; Zero: s = -1")
    print("  n = 2, m = 1, n-m = 1")
    print("  One branch goes to infinity")
    print("  Asymptote angle: 180 degrees")
    print("  Real-axis: (-inf, -2) and (-1, 0) are on locus")
    print("  The zero between the poles 'pulls' the locus toward the LHP")
    print("  Since n-m = 1, the locus extends along the negative real axis to -inf")
    print("  Effect: System is stable for ALL K > 0 (no imaginary axis crossing)")

    # Case 2: zero at s = -5 (left of both poles)
    print("\nCase 2: G(s) = K(s+5)/[s(s+2)]")
    print("  Poles: s = 0, -2; Zero: s = -5")
    print("  n = 2, m = 1, n-m = 1")
    print("  One branch goes to infinity")
    print("  Asymptote angle: 180 degrees")
    print("  Real-axis: (-5, -2) and origin to beyond? Let's count:")
    print("    s > 0: 0 to right => even => NOT on locus")
    print("    -2 < s < 0: 1 pole to right => odd => ON locus")
    print("    -5 < s < -2: 2 poles to right => even => NOT on locus")
    print("    s < -5: 2 poles + 1 zero = 3 to right => odd => ON locus")
    print("  One branch starts at s=0, goes left to breakaway, then returns")
    print("  One branch starts at s=-2, goes to zero at s=-5")
    print("  Effect: Also stable for all K > 0, but the zero being farther away")
    print("  provides less 'pull' toward the LHP near the dominant region")

    print("\nComparison:")
    print("  Zero at s = -1 (between poles):")
    print("    - Stronger stabilizing effect near the poles")
    print("    - The breakaway angle pushes branches into the LHP more effectively")
    print("    - Better transient response achievable")
    print("  Zero at s = -5 (left of poles):")
    print("    - Weaker stabilizing effect (zero is far from dominant poles)")
    print("    - Less influence on the root locus near the origin")
    print("    - May cause less overshoot due to zero being far from closed-loop poles")

    # Plot both root loci
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (zero_loc, ax) in enumerate(zip([-1, -5], axes)):
        K_values = np.linspace(0.01, 50, 500)
        den = np.polymul([1, 0], [1, 2])  # s(s+2) = s^2 + 2s

        for K in K_values:
            num_k = np.array([K, K * abs(zero_loc)])  # K(s + |zero|)
            char_poly = np.polyadd(den, num_k)
            roots = np.roots(char_poly)
            ax.plot(roots.real, roots.imag, 'b.', markersize=1)

        ax.plot([0, -2], [0, 0], 'rx', markersize=12, markeredgewidth=2, label='Poles')
        ax.plot([zero_loc], [0], 'bo', markersize=10, markeredgewidth=2,
                fillstyle='none', label='Zero')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_xlabel('Real Axis')
        ax.set_ylabel('Imaginary Axis')
        ax.set_title(f'Root Locus: zero at s = {zero_loc}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-8, 2])
        ax.set_ylim([-5, 5])

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex06_compensation.png',
                dpi=100)
    plt.close()
    print("\n  Comparison plot saved to 'ex06_compensation.png'")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Root Locus Sketch ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Design with Root Locus ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Effect of Compensation ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
