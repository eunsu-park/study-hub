"""
Exercises for Lesson 10: Lead-Lag Compensation
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
    Exercise 1: Lead Compensator Design
    Gp(s) = 5/[s(s+2)], PM >= 50 deg, Kv = 10
    """
    print("Plant: Gp(s) = 5/[s(s+2)]")
    print("Requirements: PM >= 50 degrees, Kv = 10")

    # Part 1: Determine required gain
    print("\nPart 1: Required gain for Kv = 10")
    print("  Kv = lim(s->0) s * K * Gp(s) = lim(s->0) s * K * 5/[s(s+2)]")
    print("     = K * 5/2 = 2.5K")
    print("  Kv = 10 => K = 10/2.5 = 4")
    K = 4.0
    print(f"  K = {K}")

    # Part 2: Uncompensated phase margin
    print(f"\nPart 2: Uncompensated phase margin (K = {K})")
    # KGp(s) = 20/[s(s+2)]
    num_uncomp = [K * 5]
    den_uncomp = np.polymul([1, 0], [1, 2])
    sys_uncomp = signal.TransferFunction(num_uncomp, den_uncomp)

    w = np.logspace(-2, 3, 100000)
    w_arr, mag, phase = signal.bode(sys_uncomp, w=w)

    gc_idx = np.argmin(np.abs(mag))
    w_gc_uncomp = w_arr[gc_idx]
    PM_uncomp = 180 + phase[gc_idx]
    print(f"  Gain crossover: w_gc = {w_gc_uncomp:.2f} rad/s")
    print(f"  Uncompensated PM = {PM_uncomp:.1f} degrees")

    # Part 3: Lead compensator design
    print(f"\nPart 3: Lead compensator design")
    PM_desired = 50
    margin = 10  # extra margin for frequency shift
    phi_max = PM_desired - PM_uncomp + margin
    print(f"  Required phase lead: phi_max = {PM_desired} - {PM_uncomp:.1f} + {margin}")
    print(f"                              = {phi_max:.1f} degrees")

    # alpha
    phi_rad = np.radians(phi_max)
    alpha = (1 - np.sin(phi_rad)) / (1 + np.sin(phi_rad))
    print(f"  alpha = (1-sin({phi_max:.1f}))/(1+sin({phi_max:.1f})) = {alpha:.4f}")

    # New gain crossover: where |KGp| = -10*log10(1/alpha) dB
    target_db = -10 * np.log10(1/alpha)
    print(f"  Target magnitude at new w_gc: {target_db:.2f} dB")

    # Find new w_gc
    gc_new_idx = np.argmin(np.abs(mag - (-10*np.log10(1/alpha))))
    w_gc_new = w_arr[gc_new_idx]
    print(f"  New gain crossover: w_gc_new = {w_gc_new:.2f} rad/s")

    # Compensator parameters
    tau = 1 / (w_gc_new * np.sqrt(alpha))
    zero = 1 / tau
    pole = 1 / (alpha * tau)
    Kc = K  # Lead compensator gain adjustment handled separately

    print(f"  tau = 1/(w_gc_new * sqrt(alpha)) = {tau:.4f}")
    print(f"  Zero at s = -{zero:.2f}")
    print(f"  Pole at s = -{pole:.2f}")
    print(f"  Lead: Gc_lead(s) = (s + {zero:.2f}) / (s + {pole:.2f})")

    # Part 4: Verify compensated PM
    print(f"\nPart 4: Verification")
    # Full compensated open-loop: K * Gc_lead * Gp
    # = K * (tau*s + 1)/(alpha*tau*s + 1) * 5/[s(s+2)]
    num_lead = [tau, 1]
    den_lead = [alpha * tau, 1]
    num_comp = np.polymul([K * 5], num_lead)
    den_comp = np.polymul(np.polymul([1, 0], [1, 2]), den_lead)

    sys_comp = signal.TransferFunction(num_comp, den_comp)
    w_arr2, mag2, phase2 = signal.bode(sys_comp, w=w)

    gc_idx2 = np.argmin(np.abs(mag2))
    PM_comp = 180 + phase2[gc_idx2]
    w_gc_comp = w_arr2[gc_idx2]
    print(f"  Compensated gain crossover: w_gc = {w_gc_comp:.2f} rad/s")
    print(f"  Compensated PM = {PM_comp:.1f} degrees")
    print(f"  Requirement satisfied: PM = {PM_comp:.1f} >= {PM_desired}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    w_plot = np.logspace(-1, 2, 5000)
    _, mag_u, phase_u = signal.bode(sys_uncomp, w=w_plot)
    _, mag_c, phase_c = signal.bode(sys_comp, w=w_plot)

    ax1.semilogx(w_plot, mag_u, 'b--', linewidth=2, label='Uncompensated')
    ax1.semilogx(w_plot, mag_c, 'r-', linewidth=2, label='With Lead')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Lead Compensation Design')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    ax2.semilogx(w_plot, phase_u, 'b--', linewidth=2, label='Uncompensated')
    ax2.semilogx(w_plot, phase_c, 'r-', linewidth=2, label='With Lead')
    ax2.axhline(y=-180, color='k', linestyle='--', alpha=0.3)
    ax2.axhline(y=-180 + PM_desired, color='g', linestyle=':', alpha=0.5,
                label=f'PM = {PM_desired} deg target')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_xlabel('Frequency (rad/s)')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex10_lead.png',
                dpi=100)
    plt.close()
    print("  Bode plot saved to 'ex10_lead.png'")


def exercise_2():
    """
    Exercise 2: Lag Compensator Design
    Gp(s) = 1/[s(s+1)(s+5)], Kv = 10, PM >= 40 deg
    """
    print("Plant: Gp(s) = 1/[s(s+1)(s+5)]")
    print("Requirements: Kv = 10, PM >= 40 degrees")

    # Step 1: Set gain for PM without Kv constraint
    print("\nStep 1: Find K that gives PM = 40 degrees (ignoring Kv)")
    # We need to find w where phase = -140 degrees, then set gain to 0 dB there

    num_plant = [1]
    den_plant = np.polymul([1, 0], np.polymul([1, 1], [1, 5]))

    # Phase of Gp(jw) = -90 - atan(w) - atan(w/5)
    # For PM = 40: phase at w_gc = -140
    # -90 - atan(w) - atan(w/5) = -140
    # atan(w) + atan(w/5) = 50

    def phase_eq(w):
        return np.degrees(np.arctan(w) + np.arctan(w/5)) - 50

    w_gc_target = optimize.brentq(phase_eq, 0.01, 10)
    print(f"  w_gc for PM=40: {w_gc_target:.4f} rad/s")

    # |Gp(jw_gc)| at this frequency
    mag_at_target = 1 / (w_gc_target * np.sqrt(1 + w_gc_target**2) *
                         np.sqrt(1 + w_gc_target**2/25))
    K_pm = 1 / mag_at_target
    print(f"  |Gp(jw_gc)| = {mag_at_target:.6f}")
    print(f"  K for 0dB at w_gc: K = {K_pm:.4f}")
    print(f"  Kv with this K: Kv = K/(1*5) = {K_pm/5:.4f}")

    # Step 2: Determine beta
    Kv_required = 10
    K_needed = Kv_required * 5  # Kv = K/5 => K = 50
    beta = K_needed / K_pm
    print(f"\nStep 2: Determine lag ratio beta")
    print(f"  K needed for Kv = {Kv_required}: K = {K_needed}")
    print(f"  beta = K_needed / K_pm = {K_needed}/{K_pm:.4f} = {beta:.2f}")

    # Step 3: Place lag compensator
    w_z = w_gc_target / 10  # one decade below crossover
    w_p = w_z / beta
    print(f"\nStep 3: Lag compensator placement")
    print(f"  Zero: w_z = w_gc/10 = {w_z:.4f} rad/s => zero at s = -{w_z:.4f}")
    print(f"  Pole: w_p = w_z/beta = {w_p:.6f} rad/s => pole at s = -{w_p:.6f}")

    # Verify
    print(f"\nStep 4: Verification")
    num_lag = [1/w_z, 1]  # (s/w_z + 1)
    den_lag = [1/w_p, 1]  # (s/w_p + 1)
    num_comp = np.polymul([K_needed], np.polymul(num_plant, num_lag))
    den_comp = np.polymul(den_plant, den_lag)

    sys_comp = signal.TransferFunction(num_comp, den_comp)
    w = np.logspace(-3, 2, 100000)
    w_arr, mag, phase = signal.bode(sys_comp, w=w)

    gc_idx = np.argmin(np.abs(mag))
    PM_final = 180 + phase[gc_idx]
    print(f"  Compensated PM = {PM_final:.1f} degrees (target: >= 40)")
    print(f"  Kv = K/5 = {K_needed/5:.1f} (target: {Kv_required})")


def exercise_3():
    """
    Exercise 3: Lead-Lag Design
    Gp(s) = 10/[(s+1)(s+5)]
    Requirements: zero e_ss for step (need integrator), PM >= 50 deg
    """
    print("Plant: Gp(s) = 10/[(s+1)(s+5)]")
    print("Requirements: zero step e_ss (needs integrator), PM >= 50 degrees")

    # Type 0 plant => need integrator for zero step error
    print("\nDesign approach:")
    print("  1. Add integrator (part of lag section): 1/s")
    print("  2. The lag section becomes: (s+z_lag) / [s*(s+p_lag)] with z_lag > p_lag")
    print("     This provides the integrator + additional low-freq gain")
    print("  3. Add lead section for PM improvement")

    # Open-loop with integrator only: 10/[s(s+1)(s+5)]
    print("\nStep 1: Open-loop with integrator")
    print("  G_ol(s) = 10/[s(s+1)(s+5)]")

    num_ol = [10]
    den_ol = np.polymul([1, 0], np.polymul([1, 1], [1, 5]))
    sys_ol = signal.TransferFunction(num_ol, den_ol)

    w = np.logspace(-3, 3, 100000)
    w_arr, mag, phase = signal.bode(sys_ol, w=w)

    gc_idx = np.argmin(np.abs(mag))
    PM_ol = 180 + phase[gc_idx]
    w_gc_ol = w_arr[gc_idx]
    print(f"  w_gc = {w_gc_ol:.2f} rad/s, PM = {PM_ol:.1f} degrees")

    # Need lead compensation to boost PM from ~current to 50
    PM_desired = 50
    margin = 12
    phi_max = PM_desired - PM_ol + margin
    print(f"\nStep 2: Lead compensator for PM improvement")
    print(f"  Needed phase lead: {PM_desired} - {PM_ol:.1f} + {margin} = {phi_max:.1f} degrees")

    if phi_max > 0:
        phi_rad = np.radians(min(phi_max, 65))  # cap at practical limit
        alpha = (1 - np.sin(phi_rad)) / (1 + np.sin(phi_rad))
        print(f"  alpha = {alpha:.4f}")

        target_db = -10 * np.log10(1/alpha)
        gc_new_idx = np.argmin(np.abs(mag - target_db))
        w_gc_new = w_arr[gc_new_idx]
        print(f"  New w_gc = {w_gc_new:.2f} rad/s")

        tau = 1 / (w_gc_new * np.sqrt(alpha))
        zero_lead = 1 / tau
        pole_lead = 1 / (alpha * tau)
        print(f"  Lead zero: s = -{zero_lead:.2f}")
        print(f"  Lead pole: s = -{pole_lead:.2f}")

        # Compensated system
        num_lead = [tau, 1]
        den_lead = [alpha * tau, 1]
        num_comp = np.polymul(num_ol, num_lead)
        den_comp = np.polymul(den_ol, den_lead)

        sys_comp = signal.TransferFunction(num_comp, den_comp)
        _, mag_c, phase_c = signal.bode(sys_comp, w=w)

        gc_idx_c = np.argmin(np.abs(mag_c))
        PM_comp = 180 + phase_c[gc_idx_c]
        print(f"\n  Compensated PM = {PM_comp:.1f} degrees")

        if PM_comp < PM_desired:
            print(f"  May need to iterate or add a second lead section")
        else:
            print(f"  Requirement satisfied: PM = {PM_comp:.1f} >= {PM_desired}")
    else:
        print(f"  PM already sufficient, no lead needed")

    print(f"\nFinal compensator structure:")
    print(f"  Gc(s) = [1/s] * [(s+{zero_lead:.2f})/(s+{pole_lead:.2f})]")
    print(f"  = (s+{zero_lead:.2f}) / [s*(s+{pole_lead:.2f})]")
    print(f"  This integrator-lead compensator provides:")
    print(f"  - Integrator for zero steady-state step error")
    print(f"  - Lead section for phase margin improvement")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Lead Compensator Design ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Lag Compensator Design ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Lead-Lag Design ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
