"""
Exercises for Lesson 08: Nyquist Stability Criterion
Topic: Control_Theory
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Exercise 1: Nyquist Plot and Stability
    G(s) = K / [s(s+1)(s+2)]
    """
    print("G(s) = K / [s(s+1)(s+2)]")
    print("Open-loop poles: s = 0, -1, -2 (all in LHP or on boundary)")
    print("P = 0 (no RHP poles)")

    # Part 1: Sketch Nyquist plot key features
    print("\nPart 1: Key features of Nyquist plot")
    print("  - At w -> 0+: |G| -> infinity, angle -> -90 (integrator)")
    print("  - As w increases: magnitude decreases, phase becomes more negative")
    print("  - At w -> infinity: |G| -> 0, angle -> -270")
    print("  - The plot starts from -j*infinity (coming from the origin detour)")
    print("  - Crosses real axis at some frequency")
    print("  - Approaches origin as w -> infinity")
    print("  - Lower half is the mirror image (conjugate symmetry)")

    # Part 2: Real-axis crossing
    print("\nPart 2: Real-axis crossing")
    print("  G(jw) = K / [jw(jw+1)(jw+2)]")
    print("  Denominator: jw(jw+1)(jw+2) = jw[(jw)^2 + 3(jw) + 2]")
    print("             = jw[-w^2 + 3jw + 2] = jw(2-w^2) + j*jw*3w")
    print("             = jw(2-w^2) - 3w^2")
    print("             = -3w^2 + jw(2-w^2)")
    print()
    print("  G(jw) = K / [-3w^2 + jw(2-w^2)]")
    print("  Multiply by conjugate:")
    print("  G(jw) = K[-3w^2 - jw(2-w^2)] / [9w^4 + w^2(2-w^2)^2]")
    print()
    print("  For real-axis crossing: Im[G(jw)] = 0")
    print("  -Kw(2-w^2) = 0")
    print("  w = 0 (trivial) or w^2 = 2 => w = sqrt(2)")

    w_cross = np.sqrt(2)
    print(f"\n  Crossing frequency: w = sqrt(2) = {w_cross:.4f} rad/s")

    # Real part at crossing
    # G(jw_cross) real part:
    # = K * (-3w^2) / [9w^4 + w^2(2-w^2)^2]
    # At w^2 = 2: 2-w^2 = 0, so denominator = 9*4 + 2*0 = 36
    # Real part = K * (-3*2) / 36 = -K/6
    real_at_cross = -1.0 / 6  # for K = 1
    print(f"  At w = sqrt(2), denominator = 9w^4 + w^2(2-w^2)^2 = 9*4 + 0 = 36")
    print(f"  Real part of G(jw) = K * (-6) / 36 = -K/6")
    print(f"  For K = 1: crossing point = ({real_at_cross:.4f}, 0)")

    # Part 3: Maximum K for stability
    print("\nPart 3: Maximum K for closed-loop stability")
    print("  P = 0 (no open-loop RHP poles)")
    print("  For stability: N = 0 (no encirclements of -1)")
    print("  The Nyquist plot crosses the real axis at -K/6")
    print("  For no encirclement: -K/6 > -1 => K/6 < 1 => K < 6")
    print(f"  Maximum K for stability: K_max = 6")

    # Part 4: Verify with Routh
    print("\nPart 4: Verification using Routh-Hurwitz")
    print("  Char. eq: s^3 + 3s^2 + 2s + K = 0")
    print("  Routh array:")
    print("    s^3:  1      2")
    print("    s^2:  3      K")
    print("    s^1:  (6-K)/3")
    print("    s^0:  K")
    print("  Stability: K > 0 and (6-K)/3 > 0 => 0 < K < 6")
    print("  Confirmed: K_max = 6")

    # Plot Nyquist for K = 3 (stable)
    K = 3
    num = [K]
    den = np.polymul([1, 0], np.polymul([1, 1], [1, 2]))
    sys = signal.TransferFunction(num, den)

    w = np.logspace(-2, 3, 10000)
    w, H = signal.freqresp(sys, w=w)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(H.real, H.imag, 'b-', linewidth=2, label=f'K = {K} (positive w)')
    ax.plot(H.real, -H.imag, 'b--', linewidth=1, alpha=0.5, label='Negative w (mirror)')
    ax.plot(-1, 0, 'r+', markersize=15, markeredgewidth=2, label='Critical point (-1, 0)')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(f'Nyquist Plot: G(s) = {K}/[s(s+1)(s+2)]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([-2, 1])
    ax.set_ylim([-2, 2])

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex08_nyquist.png',
                dpi=100)
    plt.close()
    print("  Nyquist plot saved to 'ex08_nyquist.png'")


def exercise_2():
    """
    Exercise 2: Non-Minimum Phase
    G(s) = K(s-1) / [s(s+2)] with K = 5
    """
    print("G(s) = K(s-1) / [s(s+2)]")
    K = 5

    # Part 1: Open-loop RHP poles
    print("\nPart 1: Open-loop RHP poles")
    print("  Poles: s = 0, s = -2")
    print("  P = 0 (no RHP poles; pole at origin is on the boundary, handled by indentation)")
    print("  Note: The RHP zero at s = +1 makes this a non-minimum phase system")

    # Part 2: Nyquist plot for K = 5
    print(f"\nPart 2: Nyquist plot for K = {K}")
    num = np.array([K, -K])  # K(s-1) = Ks - K
    den = np.polymul([1, 0], [1, 2])  # s(s+2) = s^2 + 2s
    sys = signal.TransferFunction(num, den)

    w = np.logspace(-3, 3, 50000)
    w_arr, H = signal.freqresp(sys, w=w)

    # Find real-axis crossing
    # G(jw) = K(jw - 1) / [jw(jw+2)]
    # = K(jw-1) / [-w^2 + 2jw]
    # Multiply by conjugate:
    # = K(jw-1)(-w^2 - 2jw) / [w^4 + 4w^2]
    # Numerator: (jw-1)(-w^2-2jw) = -jw^3 - 2j^2w^2 + w^2 + 2jw
    #          = -jw^3 + 2w^2 + w^2 + 2jw = 3w^2 + j(-w^3 + 2w)
    # Hmm, let me redo carefully:
    # (jw - 1)(-w^2 - 2jw) = jw*(-w^2) + jw*(-2jw) + (-1)(-w^2) + (-1)(-2jw)
    #                       = -jw^3 + 2w^2 + w^2 + 2jw
    #                       = 3w^2 + j(2w - w^3)

    # G(jw) = K[3w^2 + j(2w - w^3)] / [w^4 + 4w^2]
    # = K[3w^2 + jw(2 - w^2)] / [w^2(w^2 + 4)]

    # Im[G] = Kw(2-w^2) / [w^2(w^2+4)] = K(2-w^2)/[w(w^2+4)]
    # Im[G] = 0 when w^2 = 2 => w = sqrt(2)

    w_cross = np.sqrt(2)
    # Re[G] at w = sqrt(2):
    # Re = K * 3*2 / [4*(2+4)] = 6K/24 = K/4
    re_val = K / 4
    print(f"  Real-axis crossing at w = sqrt(2):")
    print(f"  G(j*sqrt(2)) = K/4 = {re_val:.2f} (positive real axis)")

    # As w -> 0+: |G| -> infinity
    # As w -> infinity: G -> 0 with phase approaching 0 degrees (one zero, 2 poles)
    # Actually phase: K(s-1)/[s(s+2)], at high w:
    # angle = angle(s-1) - angle(s) - angle(s+2) = 90 - 90 - 90 = -90
    # But (s-1) at high freq has angle ~90 (since jw dominates)
    # Phase at DC is tricky due to integrator and RHP zero.

    print(f"\n  Low frequency behavior:")
    print(f"  At w -> 0+: G ~ K(-1)/(jw*2) = -K/(2jw) = K/(2w) * e^(j*90)")
    print(f"  So starts from +j*infinity (positive imaginary)")

    # Part 3 & 4: Encirclements and stability
    print(f"\nPart 3: Does the Nyquist plot encircle (-1, 0)?")

    # Closed-loop char. eq: s(s+2) + K(s-1) = 0
    # s^2 + 2s + Ks - K = 0
    # s^2 + (2+K)s - K = 0
    char_coeffs = [1, 2+K, -K]
    cl_poles = np.roots(char_coeffs)
    print(f"  Closed-loop char. eq: s^2 + {2+K}s + {-K} = 0")
    print(f"  Closed-loop poles: {np.round(cl_poles, 4)}")
    n_rhp = sum(1 for p in cl_poles if p.real > 0)
    print(f"  Number of RHP closed-loop poles: Z = {n_rhp}")
    print(f"  Since P = 0, N = Z - P = {n_rhp}")
    print(f"  The Nyquist plot encircles (-1, 0) {n_rhp} time(s) clockwise")

    print(f"\nPart 4: Is the closed-loop system stable?")
    print(f"  Z = {n_rhp}, so the system is {'STABLE' if n_rhp == 0 else 'UNSTABLE'}")
    print(f"  The constant term of char. eq is -K = {-K} < 0,")
    print(f"  which by Routh's necessary condition means the system is unstable.")
    print(f"  (One coefficient is negative => at least one RHP pole)")


def exercise_3():
    """
    Exercise 3: Time Delay
    G(s) = 2*exp(-0.5s) / (s+1)
    """
    print("G(s) = 2*exp(-0.5s) / (s+1)")
    K = 2.0
    T_delay = 0.5

    # Part 1: Phase crossover frequency
    print("\nPart 1: Phase crossover frequency")
    print("  Phase of G(jw) = -atan(w) - w*T_delay (in radians)")
    print("                  = -atan(w) - 0.5w")
    print("  Set phase = -pi (-180 degrees):")
    print("  -atan(w) - 0.5w = -pi")
    print("  atan(w) + 0.5w = pi")

    def phase_eq(w):
        return np.arctan(w) + 0.5 * w - np.pi

    w_pc = optimize.brentq(phase_eq, 0.01, 20)
    print(f"\n  Solving numerically: w_pc = {w_pc:.4f} rad/s")
    print(f"  Verification: atan({w_pc:.4f}) + 0.5*{w_pc:.4f} = "
          f"{np.arctan(w_pc) + 0.5*w_pc:.4f} (pi = {np.pi:.4f})")

    # Part 2: Gain margin
    print("\nPart 2: Gain margin")
    # |G(jw_pc)| = K / sqrt(1 + w_pc^2) (delay has unity magnitude)
    mag_at_pc = K / np.sqrt(1 + w_pc**2)
    GM = 1 / mag_at_pc
    GM_dB = 20 * np.log10(GM)
    print(f"  |G(jw_pc)| = {K} / sqrt(1 + {w_pc:.4f}^2) = {mag_at_pc:.4f}")
    print(f"  GM = 1 / {mag_at_pc:.4f} = {GM:.4f} = {GM_dB:.2f} dB")

    # Part 3: Maximum additional time delay
    print("\nPart 3: Maximum additional time delay before instability")
    print("  The current gain crossover frequency is where |G(jw)| = 1")
    print("  |G(jw)| = 2/sqrt(1+w^2) = 1 => w^2 = 3 => w_gc = sqrt(3)")

    w_gc = np.sqrt(3)
    current_phase = -np.arctan(w_gc) - T_delay * w_gc
    current_PM = np.pi + current_phase  # in radians
    print(f"  w_gc = sqrt(3) = {w_gc:.4f} rad/s")
    print(f"  Current phase at w_gc = -atan(sqrt(3)) - 0.5*sqrt(3)")
    print(f"                        = {np.degrees(current_phase):.2f} degrees")
    print(f"  Current PM = 180 + ({np.degrees(current_phase):.2f}) = {np.degrees(current_PM):.2f} degrees")

    # For instability: PM = 0, meaning total phase = -180
    # -atan(w_gc) - (T_delay + T_extra)*w_gc = -pi
    # T_extra = (pi - atan(w_gc)) / w_gc - T_delay
    # But w_gc also changes with additional delay... however, delay doesn't change magnitude
    # so w_gc stays at sqrt(3).
    T_extra = (np.pi - np.arctan(w_gc)) / w_gc - T_delay
    print(f"\n  Since delay doesn't affect magnitude, w_gc remains sqrt(3)")
    print(f"  For PM = 0: -atan(sqrt(3)) - (0.5 + T_extra)*sqrt(3) = -pi")
    print(f"  T_total = (pi - atan(sqrt(3))) / sqrt(3) = "
          f"{(np.pi - np.arctan(w_gc)) / w_gc:.4f} s")
    print(f"  T_extra = {T_extra:.4f} s")
    print(f"  Maximum additional delay: {T_extra:.4f} s")
    print(f"  Total delay at instability: {T_delay + T_extra:.4f} s")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Nyquist Plot and Stability ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Non-Minimum Phase ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Time Delay ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
