"""
Exercises for Lesson 09: Solar Dynamo and Cycle
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np


# --- Physical constants ---
R_sun = 6.957e8        # solar radius [m]


def exercise_1():
    """
    Problem 1: Differential Rotation Shear

    Omega_eq = 14.7 deg/day, Omega_pole = 10.0 deg/day.
    Profile: Omega(theta) = Omega_eq - Delta_Omega * sin^2(theta)
    where theta is colatitude.
    Calculate time for a radial field line at 45 deg latitude to wind
    into one complete toroidal loop.
    """
    Omega_eq = 14.7     # deg/day
    Omega_pole = 10.0   # deg/day
    latitude = 45.0     # degrees

    # Delta_Omega = Omega_eq - Omega_pole
    # At the pole (theta=0, latitude=90): Omega = Omega_eq - Delta_Omega * sin^2(0) = Omega_eq (wrong!)
    # Wait: theta = colatitude, so pole = theta=0, equator = theta=90
    # At pole (theta=0): Omega = Omega_eq - Delta_Omega * 0 = Omega_eq -- that's wrong
    # The convention used: Omega(theta) = Omega_eq - Delta_Omega * sin^2(theta)
    # If theta is colatitude: sin(0) = 0 at pole, sin(90) = 1 at equator
    # So this gives Omega(equator) = Omega_eq - Delta_Omega = Omega_pole
    # That's the wrong convention. Let's check...
    # Actually the standard solar convention is often written with latitude:
    # Omega(lat) = Omega_eq - Delta_Omega * sin^2(lat)
    # At lat=90 (pole): Omega = Omega_eq - Delta_Omega
    # This matches: Omega_pole = Omega_eq - Delta_Omega

    Delta_Omega = Omega_eq - Omega_pole

    # At latitude 45 deg:
    lat = np.radians(latitude)
    Omega_45 = Omega_eq - Delta_Omega * np.sin(lat)**2

    print(f"  Omega_eq = {Omega_eq} deg/day")
    print(f"  Omega_pole = {Omega_pole} deg/day")
    print(f"  Delta_Omega = {Delta_Omega:.1f} deg/day")
    print(f"  At latitude {latitude} deg:")
    print(f"    Omega(45) = {Omega_eq} - {Delta_Omega} * sin^2(45)")
    print(f"              = {Omega_eq} - {Delta_Omega} * {np.sin(lat)**2:.3f}")
    print(f"              = {Omega_45:.2f} deg/day")

    # Shear rate between equator and 45 deg
    dOmega = Omega_eq - Omega_45
    print(f"\n  Angular velocity difference (equator - 45 deg): {dOmega:.2f} deg/day")

    # Time to wind one complete loop (360 degrees of differential rotation)
    t_wind = 360.0 / dOmega  # days
    print(f"  Time for 360 deg of differential winding:")
    print(f"    t = 360 / {dOmega:.2f} = {t_wind:.0f} days = {t_wind/365.25:.1f} years")

    # Also compute the shear between 45 deg and the equator more precisely
    # The question says "initially radial field line at 45 deg latitude"
    # One end at latitude 45, the other can be thought of as anchored
    # The relevant differential is between the field line's latitude and itself
    # across longitude -- essentially how fast the footpoint moves relative to
    # a reference frame rotating at some rate.

    print(f"\n  Physical interpretation:")
    print(f"  A field line anchored at the equator and at 45 deg latitude")
    print(f"  will be wound by the {dOmega:.1f} deg/day shear.")
    print(f"  After {t_wind:.0f} days, the equatorial end has lapped the 45-deg end")
    print(f"  by one full rotation, creating one complete toroidal loop.")


def exercise_2():
    """
    Problem 2: Dynamo Number Estimate

    N_D = alpha_0 * Delta_Omega * R^3 / eta_t^2
    alpha_0 ~ 1 m/s, Delta_Omega ~ 1e-6 rad/s, R ~ 5e10 cm,
    eta_t ~ 1e12 cm^2/s.
    """
    alpha_0 = 1.0       # m/s = 100 cm/s
    Delta_Omega = 1.0e-6  # rad/s
    R = 5.0e10           # cm
    eta_t = 1.0e12       # cm^2/s

    # Convert alpha_0 to CGS
    alpha_0_cgs = alpha_0 * 100.0  # cm/s

    # Dynamo number
    N_D = alpha_0_cgs * Delta_Omega * R**3 / eta_t**2
    print(f"  Parameters:")
    print(f"    alpha_0 = {alpha_0} m/s = {alpha_0_cgs} cm/s")
    print(f"    Delta_Omega = {Delta_Omega:.1e} rad/s")
    print(f"    R = {R:.1e} cm")
    print(f"    eta_t = {eta_t:.1e} cm^2/s")
    print(f"")
    print(f"  Dynamo number: N_D = alpha_0 * Delta_Omega * R^3 / eta_t^2")
    print(f"                    = {alpha_0_cgs} * {Delta_Omega:.1e} * ({R:.1e})^3 / ({eta_t:.1e})^2")
    print(f"                    = {N_D:.1f}")
    print(f"")

    # Typical critical dynamo numbers
    N_D_crit = 10.0  # rough order of magnitude
    print(f"  Typical critical dynamo number: N_D_crit ~ {N_D_crit:.0f}")
    if abs(N_D) > N_D_crit:
        print(f"  |N_D| = {abs(N_D):.1f} > {N_D_crit:.0f} => ABOVE critical => dynamo action possible")
    else:
        print(f"  |N_D| = {abs(N_D):.1f} < {N_D_crit:.0f} => BELOW critical => marginal")

    print(f"\n  Note: The exact critical dynamo number depends on the boundary")
    print(f"  conditions and geometry. For an alpha-Omega dynamo, N_D_crit ~ 1-10.")
    print(f"  Our estimate suggests the solar convection zone is capable of")
    print(f"  sustaining dynamo action, consistent with the observed cycle.")


def exercise_3():
    """
    Problem 3: Meridional Flow and Cycle Period

    T_cyc ~ 1/v_m. For T = 11 years at v_m = 12 m/s,
    find T for v_m = 8 m/s and v_m = 18 m/s.
    """
    T_ref = 11.0    # years (reference cycle period)
    v_ref = 12.0    # m/s (reference flow speed)

    print(f"  Flux transport dynamo: T_cyc proportional to 1/v_m")
    print(f"  Reference: T = {T_ref:.0f} years at v_m = {v_ref:.0f} m/s")
    print(f"  => T_cyc = {T_ref} * {v_ref} / v_m = {T_ref * v_ref:.0f} / v_m")

    for v_m in [8.0, 12.0, 18.0]:
        T = T_ref * v_ref / v_m
        print(f"\n    v_m = {v_m:.0f} m/s:")
        print(f"    T_cyc = {T_ref * v_ref:.0f} / {v_m:.0f} = {T:.1f} years")

    print(f"\n  Discussion:")
    print(f"    - Slower meridional flow (8 m/s) => longer cycle ({T_ref*v_ref/8:.0f} years)")
    print(f"    - Faster meridional flow (18 m/s) => shorter cycle ({T_ref*v_ref/18:.1f} years)")
    print(f"    - This is consistent with observations: weaker cycles tend to be")
    print(f"      longer, suggesting a link between flow speed and cycle strength.")
    print(f"    - Amplitude implications: slower transport allows more flux to")
    print(f"      accumulate at the poles before reversal, potentially producing")
    print(f"      a stronger poloidal field for the NEXT cycle, but the current")
    print(f"      cycle is weaker/longer. This is the basis of cycle prediction")
    print(f"      models using polar field as a precursor.")


def exercise_4():
    """
    Problem 4: Babcock-Leighton Source

    BMR at latitude 20 deg, tilt 10 deg, separation d=1e10 cm, Phi=1e22 Mx.
    Dipole moment: m = Phi * d * sin(gamma).
    How many such regions to reverse the polar dipole?
    """
    latitude = 20.0      # degrees
    gamma = 10.0         # tilt angle [degrees]
    d = 1.0e10           # pole separation [cm]
    Phi = 1.0e22         # flux per polarity [Mx]

    gamma_rad = np.radians(gamma)

    # Dipole moment from a single BMR
    m_single = Phi * d * np.sin(gamma_rad)
    print(f"  Single BMR parameters:")
    print(f"    Latitude: {latitude} deg")
    print(f"    Tilt (Joy's law): gamma = {gamma} deg")
    print(f"    Pole separation: d = {d:.0e} cm")
    print(f"    Flux per polarity: Phi = {Phi:.0e} Mx")
    print(f"")
    print(f"  Dipole moment: m = Phi * d * sin(gamma)")
    print(f"                   = {Phi:.0e} * {d:.0e} * sin({gamma} deg)")
    print(f"                   = {Phi:.0e} * {d:.0e} * {np.sin(gamma_rad):.4f}")
    print(f"                   = {m_single:.2e} Mx cm")

    # Total from N_cycle ~ 2000 BMRs
    N_cycle = 2000
    m_total = N_cycle * m_single
    print(f"\n  Total from {N_cycle} BMRs per cycle:")
    print(f"    m_total = {N_cycle} * {m_single:.2e} = {m_total:.2e} Mx cm")

    # Polar dipole moment estimate
    # The Sun's dipole moment: m_sun ~ B_pole * R_sun^2 * (4pi/3) * ...
    # More simply: polar flux ~ B_pole * 2*pi*R^2*(1-cos(theta_cap))
    # For polar cap of 15 deg: flux ~ B_pole * 2*pi*R^2*(1-cos(75 deg))
    # Typical B_pole ~ 5-10 G at solar minimum
    B_pole = 10.0  # G
    R_cm = R_sun * 100.0  # cm
    # Polar cap flux (for one pole, cap above ~75 deg = within 15 deg of pole)
    theta_cap = np.radians(15.0)
    Phi_pole = B_pole * 2.0 * np.pi * R_cm**2 * (1.0 - np.cos(theta_cap))
    m_pole = Phi_pole * R_cm  # rough dipole moment scale

    print(f"\n  Polar field for reference:")
    print(f"    B_pole ~ {B_pole:.0f} G")
    print(f"    Polar cap flux (within 15 deg): Phi ~ {Phi_pole:.2e} Mx")
    print(f"    Polar dipole moment scale: m ~ {m_pole:.2e} Mx cm")
    print(f"\n  Ratio m_total / m_pole ~ {m_total / m_pole:.1f}")
    print(f"  The collective effect of ~{N_cycle} BMRs with systematic tilt is")
    print(f"  sufficient to reverse and rebuild the polar magnetic dipole")
    print(f"  each half-cycle, as required by the Babcock-Leighton mechanism.")


def exercise_5():
    """
    Problem 5: Grand Minimum Probability

    Probability of entering a grand minimum in any 11-yr cycle: p = 0.02.
    Model as geometric distribution.
    P(at least one in N cycles) = 1 - (1-p)^N.
    """
    p = 0.02  # probability per cycle

    # Within next 50 years => approximately 50/11 ~ 4.5 cycles
    # Let's use integer cycles
    t_years = 50.0
    N_cycles = t_years / 11.0  # approximate number of cycles

    # Use N_cycles rounded (both floor and ceil for illustration)
    print(f"  Probability per 11-year cycle: p = {p}")
    print(f"  Time horizon: {t_years:.0f} years")
    print(f"  Number of cycles: {t_years:.0f} / 11 = {N_cycles:.2f}")
    print(f"  (Using approximately {int(round(N_cycles))} cycles)")

    # P(at least one) = 1 - (1-p)^N
    for N in [4, 5]:
        P_enter = 1.0 - (1.0 - p)**N
        print(f"\n    For N = {N} cycles:")
        print(f"    P(grand minimum) = 1 - (1-{p})^{N}")
        print(f"                     = 1 - {(1-p)**N:.4f}")
        print(f"                     = {P_enter:.4f} = {P_enter*100:.1f}%")

    # Most precise estimate using fractional cycles
    P_enter_exact = 1.0 - (1.0 - p)**(N_cycles)
    print(f"\n    Interpolated for {N_cycles:.2f} cycles:")
    print(f"    P(grand minimum) = {P_enter_exact:.4f} = {P_enter_exact*100:.1f}%")

    # Consistency check: 17% occurrence rate
    print(f"\n  Consistency check:")
    print(f"    If p = 0.02 per cycle and average duration = 70 years (~6 cycles),")
    print(f"    steady-state fraction of time in grand minimum:")
    print(f"    = p * duration / (1 + p * duration)")
    avg_duration_cycles = 70.0 / 11.0
    frac_in_minimum = p * avg_duration_cycles / (1.0 + p * avg_duration_cycles)
    print(f"    = {p} * {avg_duration_cycles:.1f} / (1 + {p} * {avg_duration_cycles:.1f})")
    print(f"    = {frac_in_minimum:.3f} = {frac_in_minimum*100:.1f}%")
    print(f"    This is roughly consistent with the ~17% occurrence rate,")
    print(f"    though the exact calculation depends on the distribution model.")

    # Broader context
    print(f"\n  The ~{P_enter_exact*100:.0f}% probability over 50 years is LOW,")
    print(f"  but not negligible. Given the significant climate and space weather")
    print(f"  implications of a Maunder-like minimum, this probability warrants")
    print(f"  continued monitoring and dynamo modeling efforts.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Differential Rotation Shear ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Dynamo Number Estimate ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Meridional Flow and Cycle Period ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Babcock-Leighton Source ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Grand Minimum Probability ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
