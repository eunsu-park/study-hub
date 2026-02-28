"""
Exercises for Lesson 04: Time-Domain Analysis
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
    Exercise 1: Second-Order Specifications
    Unity-feedback system with G(s) = 50 / [s(s+5)]
    """
    print("Open-loop: G(s) = 50 / [s(s+5)]")

    # Part 1: Closed-loop transfer function
    # T(s) = G/(1+G) = 50/(s^2 + 5s + 50)
    print("\nPart 1: Closed-loop transfer function")
    print("  T(s) = G(s)/(1 + G(s)) = 50 / (s^2 + 5s + 50)")

    # Part 2: omega_n and zeta
    # s^2 + 2*zeta*wn*s + wn^2 = s^2 + 5s + 50
    wn = np.sqrt(50)
    zeta = 5 / (2 * wn)
    print(f"\nPart 2: Identifying parameters")
    print(f"  omega_n^2 = 50  =>  omega_n = {wn:.4f} rad/s")
    print(f"  2*zeta*omega_n = 5  =>  zeta = 5/(2*{wn:.4f}) = {zeta:.4f}")
    print(f"  Since zeta = {zeta:.4f} < 1, system is underdamped")

    # Part 3: Compute Mp, tp, ts
    wd = wn * np.sqrt(1 - zeta**2)  # damped natural frequency

    # Peak overshoot
    Mp = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
    Mp_pct = Mp * 100

    # Peak time
    tp = np.pi / wd

    # Settling time (2% criterion)
    ts = 4 / (zeta * wn)

    print(f"\nPart 3: Time-domain specifications")
    print(f"  omega_d = omega_n * sqrt(1 - zeta^2) = {wd:.4f} rad/s")
    print(f"  M_p = exp(-pi*zeta/sqrt(1-zeta^2)) = {Mp_pct:.2f}%")
    print(f"  t_p = pi/omega_d = {tp:.4f} s")
    print(f"  t_s (2%) = 4/(zeta*omega_n) = {ts:.4f} s")

    # Part 4: Steady-state error for unit ramp
    # System type 1 (one integrator in G(s))
    # Kv = lim s->0 s*G(s) = lim s->0 s*50/(s(s+5)) = 50/5 = 10
    Kv = 50 / 5
    ess = 1 / Kv
    print(f"\nPart 4: Steady-state error for unit ramp")
    print(f"  System type = 1 (one integrator)")
    print(f"  Kv = lim(s->0) s*G(s) = 50/5 = {Kv:.1f}")
    print(f"  e_ss = 1/Kv = {ess:.4f}")

    # Verification with scipy
    num_cl = [50]
    den_cl = [1, 5, 50]
    sys_cl = signal.TransferFunction(num_cl, den_cl)
    t = np.linspace(0, 3, 1000)
    t_step, y_step = signal.step(sys_cl, T=t)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_step, y_step, 'b-', linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Final value')
    ax.axhline(y=1+Mp, color='r', linestyle='--', alpha=0.5, label=f'Overshoot ({Mp_pct:.1f}%)')
    ax.axvline(x=tp, color='g', linestyle='--', alpha=0.5, label=f'Peak time ({tp:.3f}s)')
    ax.axvline(x=ts, color='m', linestyle='--', alpha=0.5, label=f'Settling time ({ts:.3f}s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Response')
    ax.set_title('Step Response: T(s) = 50/(s^2 + 5s + 50)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex04_step_response.png',
                dpi=100)
    plt.close()
    print("  Step response plot saved to 'ex04_step_response.png'")


def exercise_2():
    """
    Exercise 2: Dominant Poles
    Closed-loop poles at s = -2 +/- j3 and s = -20.
    """
    dominant_poles = np.array([-2 + 3j, -2 - 3j])
    extra_pole = -20.0
    sigma = 2.0  # real part magnitude of dominant poles

    print("Closed-loop poles: s = -2 +/- j3 and s = -20")

    # Part 1: Can the third pole be neglected?
    ratio = abs(extra_pole) / sigma
    print(f"\nPart 1: Can the third pole be neglected?")
    print(f"  |third pole| / sigma = {abs(extra_pole)}/{sigma} = {ratio:.1f}")
    print(f"  Rule of thumb: neglect if ratio > 5")
    print(f"  {ratio:.1f} > 5 => YES, the third pole can be neglected")
    print(f"  The pole at s = -20 is 10x farther from the imaginary axis")
    print(f"  than the dominant pair, so it decays much faster.")

    # Part 2: Estimate Mp and ts using dominant second-order approximation
    wn = np.sqrt(sigma**2 + 3**2)  # sqrt(4 + 9) = sqrt(13)
    zeta_dom = sigma / wn

    Mp_dom = np.exp(-np.pi * zeta_dom / np.sqrt(1 - zeta_dom**2)) * 100
    ts_dom = 4 / (zeta_dom * wn)  # 2% criterion

    print(f"\nPart 2: Dominant second-order approximation")
    print(f"  omega_n = sqrt(2^2 + 3^2) = sqrt(13) = {wn:.4f} rad/s")
    print(f"  zeta = sigma/omega_n = {sigma}/{wn:.4f} = {zeta_dom:.4f}")
    print(f"  M_p = exp(-pi*zeta/sqrt(1-zeta^2)) * 100 = {Mp_dom:.2f}%")
    print(f"  t_s (2%) = 4/(zeta*omega_n) = 4/{sigma} = {ts_dom:.4f} s")

    # Verify with full 3rd order system
    # T(s) = wn^2 * 20 / [(s^2 + 4s + 13)(s + 20)]
    # Normalize for DC gain = 1: T(s) = 260 / [(s^2 + 4s + 13)(s + 20)]
    num_full = [260.0]
    den_full = np.polymul([1, 4, 13], [1, 20])
    sys_full = signal.TransferFunction(num_full, den_full)
    t = np.linspace(0, 4, 1000)
    t_out, y_out = signal.step(sys_full, T=t)

    print(f"\n  Verification with full 3rd-order system:")
    print(f"  Peak value: {np.max(y_out):.4f}, overshoot: {(np.max(y_out)-1)*100:.2f}%")


def exercise_3():
    """
    Exercise 3: System Type Design
    Design Gc(s) = K(s+a)/s with plant Gp(s) = 1/(s+2)
    Requirements: zero e_ss for step, e_ss <= 0.02 for unit ramp
    """
    print("Plant: Gp(s) = 1/(s+2)")
    print("Controller: Gc(s) = K(s+a)/s")
    print("Open-loop: G(s) = Gc(s)*Gp(s) = K(s+a) / [s(s+2)]")

    print("\nStep 1: System type analysis")
    print("  G(s) = K(s+a)/[s(s+2)] has one integrator => Type 1 system")
    print("  Type 1 => zero steady-state error for step input (automatic)")

    print("\nStep 2: Ramp error requirement")
    print("  Kv = lim(s->0) s*G(s) = lim(s->0) s*K(s+a)/[s(s+2)]")
    print("     = lim(s->0) K(s+a)/(s+2) = K*a/2")
    print("  e_ss_ramp = 1/Kv = 2/(K*a)")
    print("  Requirement: e_ss <= 0.02")
    print("  2/(K*a) <= 0.02")
    print("  K*a >= 100")

    # Minimum K depends on choice of a
    # For a given 'a', K_min = 100/a
    # If we choose a = 1 (simple), K_min = 100
    # If we choose a = 2 (cancels plant pole), K_min = 50

    print("\nStep 3: Minimum K")
    print("  K*a >= 100, so K_min = 100/a")
    print()
    print("  Example choices:")

    for a in [1, 2, 5, 10]:
        K_min = 100 / a
        Kv = K_min * a / 2
        ess = 1 / Kv
        print(f"    a = {a:2d}: K_min = {K_min:6.1f}, Kv = {Kv:.1f}, e_ss = {ess:.4f}")

    print("\n  Note: choosing a = 2 cancels the plant pole at s = -2,")
    print("  simplifying the closed-loop but losing the natural damping.")
    print("  The minimum value of K*a is 100.")

    # Verify with a=2, K=50
    print("\n  Verification with a=2, K=50:")
    K, a = 50, 2
    # G(s) = 50(s+2)/[s(s+2)] = 50/s
    # T(s) = 50/(s+50)
    print(f"  G(s) = {K}(s+{a})/[s(s+{a})] = {K}/s")
    print(f"  T(s) = {K}/(s+{K})")
    print(f"  Kv = K*a/2 = {K*a/2}")
    print(f"  e_ss (ramp) = 1/{K*a/2} = {2/(K*a):.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Second-Order Specifications ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Dominant Poles ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: System Type Design ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
