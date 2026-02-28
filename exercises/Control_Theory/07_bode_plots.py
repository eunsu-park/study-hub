"""
Exercises for Lesson 07: Frequency Response - Bode Plots
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
    Exercise 1: Bode Plot Construction
    G(s) = 50(s+5) / [s(s+2)(s+50)]
    """
    print("G(s) = 50(s+5) / [s(s+2)(s+50)]")

    # Part 1: Time-constant form
    # Factor out constants to get (1 + s/w) form:
    # G(s) = 50 * 5 * (1 + s/5) / [s * 2 * (1 + s/2) * 50 * (1 + s/50)]
    # G(s) = (50*5) / (2*50) * (1+s/5) / [s * (1+s/2) * (1+s/50)]
    # G(s) = 2.5 * (1+s/5) / [s * (1+s/2) * (1+s/50)]
    K_tc = 50 * 5 / (2 * 50)
    print(f"\nPart 1: Time-constant form")
    print(f"  G(s) = {K_tc} * (1 + s/5) / [s * (1 + s/2) * (1 + s/50)]")

    # Part 2: Corner frequencies
    print(f"\nPart 2: Corner frequencies")
    print(f"  omega_1 = 2 rad/s (pole)")
    print(f"  omega_2 = 5 rad/s (zero)")
    print(f"  omega_3 = 50 rad/s (pole)")

    # Part 3: Asymptotic Bode plot
    print(f"\nPart 3: Asymptotic magnitude slopes")
    print(f"  omega < 2: -20 dB/dec (integrator + constant)")
    print(f"  At omega = 1: |G| = 2.5/1 = 2.5 => {20*np.log10(2.5):.1f} dB")
    print(f"  2 < omega < 5: -40 dB/dec (integrator + pole)")
    print(f"  5 < omega < 50: -20 dB/dec (integrator + pole + zero cancel one slope)")
    print(f"  omega > 50: -40 dB/dec (integrator + 2 poles + zero)")

    # Part 4: Gain and phase margins
    num = [50, 250]       # 50(s+5) = 50s + 250
    den = np.polymul([1, 0], np.polymul([1, 2], [1, 50]))  # s(s+2)(s+50)
    sys = signal.TransferFunction(num, den)

    w = np.logspace(-2, 3, 10000)
    w, mag, phase = signal.bode(sys, w=w)

    # Find gain crossover frequency (where mag = 0 dB)
    gc_idx = np.argmin(np.abs(mag))
    w_gc = w[gc_idx]
    phase_at_gc = phase[gc_idx]
    PM = 180 + phase_at_gc

    # Find phase crossover frequency (where phase = -180)
    pc_idx = np.argmin(np.abs(phase + 180))
    w_pc = w[pc_idx]
    mag_at_pc = mag[pc_idx]
    GM = -mag_at_pc

    print(f"\nPart 4: Stability margins")
    print(f"  Gain crossover frequency: omega_gc = {w_gc:.2f} rad/s")
    print(f"  Phase at gain crossover: {phase_at_gc:.1f} degrees")
    print(f"  Phase Margin (PM) = 180 + ({phase_at_gc:.1f}) = {PM:.1f} degrees")
    print(f"  Phase crossover frequency: omega_pc = {w_pc:.2f} rad/s")
    print(f"  Magnitude at phase crossover: {mag_at_pc:.1f} dB")
    print(f"  Gain Margin (GM) = {GM:.1f} dB")

    if PM > 0 and GM > 0:
        print(f"  System is STABLE (PM > 0 and GM > 0)")
    else:
        print(f"  System is UNSTABLE")

    # Plot Bode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.semilogx(w, mag, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.axvline(x=w_gc, color='g', linestyle='--', alpha=0.5, label=f'w_gc = {w_gc:.1f}')
    ax1.axvline(x=w_pc, color='r', linestyle='--', alpha=0.5, label=f'w_pc = {w_pc:.1f}')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Bode Plot: G(s) = 50(s+5)/[s(s+2)(s+50)]')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    ax2.semilogx(w, phase, 'b-', linewidth=2)
    ax2.axhline(y=-180, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=w_gc, color='g', linestyle='--', alpha=0.5)
    ax2.axvline(x=w_pc, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_xlabel('Frequency (rad/s)')
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex07_bode.png',
                dpi=100)
    plt.close()
    print("  Bode plot saved to 'ex07_bode.png'")


def exercise_2():
    """
    Exercise 2: System Identification from Bode Plot
    Bode magnitude shows:
    - Slope -20 dB/dec for w < 2
    - Slope -40 dB/dec for 2 < w < 10
    - Slope -60 dB/dec for w > 10
    - Magnitude is 20 dB at w = 1
    """
    print("Given Bode magnitude characteristics:")
    print("  Slope -20 dB/dec for w < 2")
    print("  Slope -40 dB/dec for 2 < w < 10")
    print("  Slope -60 dB/dec for w > 10")
    print("  Magnitude is 20 dB at w = 1")

    print("\nAnalysis:")
    print("  Initial slope -20 dB/dec => one integrator (1/s)")
    print("  Slope changes to -40 dB/dec at w = 2 => pole at s = -2")
    print("  Slope changes to -60 dB/dec at w = 10 => pole at s = -10")

    print("\n  Form: G(s) = K / [s(s/2 + 1)(s/10 + 1)] = K / [s(1+s/2)(1+s/10)]")

    # Find K from magnitude at w = 1:
    # |G(j1)| in dB = 20
    # At w = 1 (below all break frequencies for asymptotic):
    # |G(j1)| approx = K / (1 * 1 * 1) = K (asymptotic)
    # 20 log10(K/1) = 20 => K/1 = 10 => K = 10
    # But we need to account for the integrator: |G(jw)| = K/w * 1/(sqrt(1+w^2/4)) * ...
    # At w=1: |G(j1)| = K/1 * 1/sqrt(1+1/4) * 1/sqrt(1+1/100)
    # Asymptotically (w << 2): |G(j1)| approx = K/1 = K
    # 20 log10(K) = 20 => K = 10

    K = 10
    print(f"\n  At w = 1 (below break frequencies):")
    print(f"  |G(j1)| (asymptotic) = K/w = K = 10 => 20 dB")
    print(f"  K = {K}")

    print(f"\n  Transfer function:")
    print(f"  G(s) = {K} / [s(1 + s/2)(1 + s/10)]")
    print(f"       = {K} * 20 / [s(s+2)(s+10)]")
    print(f"       = {K*20} / [s(s+2)(s+10)]")
    # Actually: K/[s * (1+s/2) * (1+s/10)]
    # = K / [s * (s+2)/2 * (s+10)/10]
    # = K * 20 / [s(s+2)(s+10)]
    # = 200 / [s(s+2)(s+10)]

    # Verify
    num = [200]
    den = np.polymul([1, 0], np.polymul([1, 2], [1, 10]))
    sys = signal.TransferFunction(num, den)

    w_test = np.array([1.0])
    _, mag_test, _ = signal.bode(sys, w=w_test)
    print(f"\n  Verification: |G(j1)| = {mag_test[0]:.2f} dB (target: 20 dB)")


def exercise_3():
    """
    Exercise 3: Stability Margin Analysis
    G(s) = K / [s(0.1s+1)(0.01s+1)]
    """
    print("G(s) = K / [s(0.1s+1)(0.01s+1)]")
    print("     = K / [s(1+s/10)(1+s/100)]")

    # Part 1: K = 10, find GM and PM
    K = 10
    print(f"\nPart 1: K = {K}")

    # G(s) = 10 / [s(0.1s+1)(0.01s+1)]
    # = 10 * 1000 / [s(s+10)(s+100)]
    # = 10000 / [s(s+10)(s+100)]
    num = [K]
    den_tc = np.polymul([0.1, 1], [0.01, 1])  # (0.1s+1)(0.01s+1)
    den = np.polymul([1, 0], den_tc)  # s * ...

    sys = signal.TransferFunction(num, den)
    w = np.logspace(-1, 4, 100000)
    w, mag, phase = signal.bode(sys, w=w)

    # Gain crossover
    gc_idx = np.argmin(np.abs(mag))
    w_gc = w[gc_idx]
    PM = 180 + phase[gc_idx]

    # Phase crossover
    pc_idx = np.argmin(np.abs(phase + 180))
    w_pc = w[pc_idx]
    GM = -mag[pc_idx]

    print(f"  Gain crossover: w_gc = {w_gc:.2f} rad/s")
    print(f"  Phase margin: PM = {PM:.1f} degrees")
    print(f"  Phase crossover: w_pc = {w_pc:.2f} rad/s")
    print(f"  Gain margin: GM = {GM:.1f} dB")

    # Part 2: Maximum K for stability
    print(f"\nPart 2: Maximum K for stability")
    # At phase crossover, |G(jw_pc)| must equal 0 dB for marginal stability
    # Phase crossover: angle(G) = -180
    # G(jw) = K / [jw * (1 + jw/10) * (1 + jw/100)]
    # Phase = -90 - atan(w/10) - atan(w/100) = -180
    # => atan(w/10) + atan(w/100) = 90
    # Using tan(A+B) = (tanA + tanB)/(1 - tanA*tanB) = infinity
    # => 1 - (w/10)(w/100) = 0 => w^2 = 1000 => w = sqrt(1000)

    w_pc_exact = np.sqrt(1000)
    print(f"  Phase crossover frequency (exact): w_pc = sqrt(1000) = {w_pc_exact:.4f} rad/s")

    # |G(jw_pc)| at w_pc:
    # |G| = K / [w * sqrt(1 + w^2/100) * sqrt(1 + w^2/10000)]
    w = w_pc_exact
    mag_at_pc = K / (w * np.sqrt(1 + w**2/100) * np.sqrt(1 + w**2/10000))
    print(f"  |G(jw_pc)| at K={K}: {mag_at_pc:.6f} = {20*np.log10(mag_at_pc):.2f} dB")

    # For marginal stability: |G(jw_pc)| = 1
    K_max = 1 / (mag_at_pc / K)
    print(f"  For marginal stability: K_max * |G(jw_pc)/K| = 1")
    print(f"  K_max = {K_max:.2f}")

    # Verify: K_max = w_pc * sqrt(1 + w_pc^2/100) * sqrt(1 + w_pc^2/10000)
    K_max_v2 = w_pc_exact * np.sqrt(1 + w_pc_exact**2/100) * np.sqrt(1 + w_pc_exact**2/10000)
    print(f"  K_max (direct) = {K_max_v2:.2f}")

    # Part 3: K for PM = 45 degrees
    print(f"\nPart 3: K for PM = 45 degrees")
    print(f"  Need: phase at gain crossover = -180 + 45 = -135 degrees")
    print(f"  -90 - atan(w/10) - atan(w/100) = -135")
    print(f"  atan(w/10) + atan(w/100) = 45")

    # Solve numerically
    from scipy.optimize import brentq

    def phase_eq(w):
        return np.arctan(w/10) + np.arctan(w/100) - np.radians(45)

    w_45 = brentq(phase_eq, 0.1, 100)
    print(f"  Solving: w_gc for PM=45 is {w_45:.4f} rad/s")

    # At this frequency, |G| must equal 1 (0 dB)
    K_45 = w_45 * np.sqrt(1 + w_45**2/100) * np.sqrt(1 + w_45**2/10000)
    print(f"  K for PM=45: K = w * |1+jw/10| * |1+jw/100| = {K_45:.4f}")

    # Verify
    num_v = [K_45]
    den_v = np.polymul([1, 0], np.polymul([0.1, 1], [0.01, 1]))
    sys_v = signal.TransferFunction(num_v, den_v)
    w_check = np.logspace(-1, 4, 100000)
    _, mag_v, phase_v = signal.bode(sys_v, w=w_check)
    gc_idx_v = np.argmin(np.abs(mag_v))
    PM_v = 180 + phase_v[gc_idx_v]
    print(f"  Verification: PM = {PM_v:.1f} degrees (target: 45 degrees)")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Bode Plot Construction ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: System Identification from Bode Plot ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Stability Margin Analysis ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
