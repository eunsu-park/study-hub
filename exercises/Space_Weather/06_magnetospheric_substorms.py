"""
Exercise Solutions for Lesson 06: Magnetospheric Substorms

Topics covered:
  - Growth phase magnetic energy storage in magnetotail lobes
  - Auroral emission altitude from precipitating electron energy
  - Electrojet current estimation from AL index
  - Injection drift dispersion (electrons vs protons)
  - Substorm vs storm energy comparison
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Growth Phase Energy

    Tail lobe field increases from 20 nT to 35 nT over 45 minutes.
    Lobe volume = 5e23 m^3 (one lobe).
    (a) Magnetic energy in both lobes before and after.
    (b) Net energy added.
    (c) Average power input.
    """
    print("=" * 70)
    print("Exercise 1: Growth Phase Magnetic Energy")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7

    B_before = 20e-9   # T
    B_after = 35e-9    # T
    V_lobe = 5e23      # m^3 (one lobe)
    dt = 45 * 60       # 45 minutes in seconds

    # (a) Magnetic energy: E = B^2 / (2*mu0) * V (both lobes = 2 * V_lobe)
    V_total = 2 * V_lobe

    E_before = B_before**2 / (2 * mu0) * V_total
    E_after = B_after**2 / (2 * mu0) * V_total

    print(f"\n    B_before = {B_before*1e9:.0f} nT, B_after = {B_after*1e9:.0f} nT")
    print(f"    V_lobe = {V_lobe:.1e} m^3, V_total (both lobes) = {V_total:.1e} m^3")

    print(f"\n(a) Magnetic energy in both lobes:")
    print(f"    E = B^2/(2*mu0) * V")
    print(f"    Before: E = ({B_before*1e9:.0f}e-9)^2 / (2*{mu0:.3e}) * {V_total:.1e}")
    print(f"    E_before = {E_before:.3e} J")
    print(f"    After:  E = ({B_after*1e9:.0f}e-9)^2 / (2*{mu0:.3e}) * {V_total:.1e}")
    print(f"    E_after  = {E_after:.3e} J")

    # (b) Net energy added
    Delta_E = E_after - E_before
    print(f"\n(b) Net energy added:")
    print(f"    Delta_E = E_after - E_before = {Delta_E:.3e} J")
    print(f"    = {Delta_E*1e-15:.2f} PJ (petajoules)")

    # (c) Average power
    P_avg = Delta_E / dt
    print(f"\n(c) Average power input:")
    print(f"    P = Delta_E / dt = {Delta_E:.3e} / {dt:.0f}")
    print(f"    P = {P_avg:.3e} W = {P_avg*1e-9:.1f} GW")
    print(f"    For comparison, typical Akasofu epsilon during moderate driving:")
    print(f"    ~100-500 GW. This is consistent with the growth phase loading rate.")


def exercise_2():
    """
    Exercise 2: Auroral Emission Altitude

    Precipitating electron E = 5 keV.
    Empirical relation: h_peak ~ 130 - 10*ln(E/keV) km.
    Determine dominant emission color (green 557.7 nm vs red 630.0 nm).
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Auroral Emission Altitude")
    print("=" * 70)

    E_keV = 5.0  # keV

    # Peak emission altitude
    h_peak = 130 - 10 * np.log(E_keV)

    print(f"\n    Precipitating electron energy: E = {E_keV} keV")
    print(f"    h_peak = 130 - 10*ln(E/keV)")
    print(f"    = 130 - 10*ln({E_keV})")
    print(f"    = 130 - 10*{np.log(E_keV):.3f}")
    print(f"    = {h_peak:.1f} km")

    print(f"\n    Dominant emission analysis:")
    print(f"    At {h_peak:.0f} km altitude:")
    print(f"    - Green line (557.7 nm, O I ^1S -> ^1D): ")
    print(f"      Radiative lifetime ~0.7 s. At ~{h_peak:.0f} km, the atmosphere")
    print(f"      is dense enough that collisions are frequent, but the ^1S state")
    print(f"      still has time to radiate before being collisionally deactivated.")
    print(f"    - Red line (630.0 nm, O I ^1D -> ^3P):")
    print(f"      Radiative lifetime ~110 s. At {h_peak:.0f} km, the neutral")
    print(f"      density is too high — the ^1D state is collisionally deactivated")
    print(f"      before it can radiate.")
    print(f"\n    CONCLUSION: GREEN (557.7 nm) dominates at {h_peak:.0f} km")
    print(f"    The red line dominates only above ~200-250 km where the")
    print(f"    atmosphere is tenuous enough for the long-lived ^1D state")
    print(f"    to survive to radiative decay.")

    # Show altitudes for different energies
    print(f"\n    Peak emission altitude for various electron energies:")
    print(f"    {'E (keV)':>10} {'h_peak (km)':>14} {'Dominant Color':>16}")
    print(f"    {'-'*42}")
    for E in [0.5, 1, 2, 5, 10, 20, 50]:
        h = 130 - 10 * np.log(E)
        color = "RED" if h > 200 else "GREEN"
        print(f"    {E:>10.1f} {h:>14.1f} {color:>16}")


def exercise_3():
    """
    Exercise 3: Electrojet Current from AL Index

    AL = -1200 nT. Infinite line current at h = 110 km.
    Estimate total electrojet current.
    Compare with typical cross-tail current disrupted during substorm.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Electrojet Current from AL Index")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7
    AL = -1200e-9  # T
    h = 110e3      # m

    # Infinite line current: B = mu0 * I / (2*pi*h)
    # |AL| = mu0 * I / (2*pi*h)
    # I = |AL| * 2*pi*h / mu0
    I = abs(AL) * 2 * np.pi * h / mu0

    print(f"\n    AL index = {AL*1e9:.0f} nT")
    print(f"    Electrojet altitude = {h/1e3:.0f} km")

    print(f"\n    From infinite line current model:")
    print(f"    |Delta_H| = mu0 * I / (2*pi*h)")
    print(f"    I = |AL| * 2*pi*h / mu0")
    print(f"    I = {abs(AL):.1e} * 2*pi*{h:.1e} / {mu0:.3e}")
    print(f"    I = {I:.3e} A = {I/1e6:.2f} MA")

    print(f"\n    Comparison with cross-tail current:")
    print(f"    Typical cross-tail current: ~5-10 MA over ~20-30 R_E width")
    print(f"    Current disrupted during substorm: ~1-3 MA")
    print(f"    Estimated electrojet current: {I/1e6:.1f} MA")
    print(f"    This is comparable to the disrupted tail current,")
    print(f"    consistent with the current wedge diversion model")
    print(f"    (tail current diverts through the ionosphere)")

    print(f"\n    Caveat: The infinite line current model overestimates I")
    print(f"    because the actual electrojet is a finite, distributed current.")
    print(f"    Typical correction factor: ~0.5-0.7")
    print(f"    More realistic estimate: {I*0.6/1e6:.1f} MA")


def exercise_4():
    """
    Exercise 4: Injection Drift Dispersion

    Dispersionless injection at midnight. Calculate drift period for:
    (a) 50 keV electron at L = 6.6
    (b) 50 keV proton at L = 6.6
    Then find detection time delay at dawn (6 hours = 90 degrees from midnight).
    T_d = 2*pi*q*B0*R_E^2 / (3*L*E)
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Injection Drift Dispersion")
    print("=" * 70)

    eV = 1.602e-19
    q = 1.602e-19   # C (magnitude)
    B0 = 3.1e-5     # T
    R_E = 6.371e6   # m
    L = 6.6
    E_keV = 50
    E_J = E_keV * 1e3 * eV

    # Drift period: T_d = 2*pi*q*B0*R_E^2 / (3*L*E)
    T_d = 2 * np.pi * q * B0 * R_E**2 / (3 * L * E_J)
    T_d_min = T_d / 60

    print(f"\n    L = {L}, E = {E_keV} keV")
    print(f"    T_d = 2*pi*q*B0*R_E^2 / (3*L*E)")
    print(f"    = 2*pi*{q:.3e}*{B0:.1e}*({R_E:.3e})^2 / (3*{L}*{E_J:.3e})")
    print(f"    T_d = {T_d:.1f} s = {T_d_min:.1f} minutes")

    # (a) 50 keV electron
    print(f"\n(a) 50 keV electron at L = {L}:")
    print(f"    T_d = {T_d_min:.1f} minutes")
    print(f"    Electrons drift eastward (from midnight toward dawn)")

    # (b) 50 keV proton - same formula, same result for drift period!
    # But protons drift WESTWARD (from midnight toward dusk)
    # The formula gives the same magnitude since it uses |q| and E
    T_d_proton = T_d  # same value
    print(f"\n(b) 50 keV proton at L = {L}:")
    print(f"    T_d = {T_d_min:.1f} minutes (same magnitude as electron)")
    print(f"    BUT protons drift WESTWARD (from midnight toward dusk)")

    # Detection at dawn (90 degrees from midnight)
    # Electron drifts eastward: midnight -> dawn = 1/4 of orbit
    t_detect_e = T_d / 4

    # Proton drifts westward: midnight -> dusk -> noon -> dawn = 3/4 of orbit
    t_detect_p = 3 * T_d / 4

    print(f"\n    Detection at dawn meridian (90 deg east of midnight):")
    print(f"    Electron (eastward drift): t = T_d/4 = {t_detect_e/60:.1f} min")
    print(f"    Proton (westward drift):   t = 3*T_d/4 = {t_detect_p/60:.1f} min")
    print(f"    Time difference: {(t_detect_p - t_detect_e)/60:.1f} min")

    print(f"\n    This energy-dependent time dispersion is the hallmark of a")
    print(f"    substorm injection. Satellites at different local times see the")
    print(f"    injection at different times, with higher energy particles")
    print(f"    arriving first (shorter drift period).")

    # Show drift periods for different energies
    print(f"\n    Drift periods at L={L} for various energies:")
    print(f"    {'E (keV)':>10} {'T_d (min)':>12} {'Dawn delay (min)':>18}")
    print(f"    {'-'*42}")
    for E in [10, 30, 50, 100, 200, 500]:
        T = 2 * np.pi * q * B0 * R_E**2 / (3 * L * E * 1e3 * eV)
        print(f"    {E:>10} {T/60:>12.1f} {T/4/60:>18.1f}")


def exercise_5():
    """
    Exercise 5: Substorm vs Storm Energy

    Storm: 3 days, avg Dst = -80 nT, 15 substorms each releasing 5e14 J.
    (a) Ring current energy from DPS.
    (b) Total substorm energy.
    (c) Compare and discuss.
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Substorm vs Storm Energy")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7
    B0 = 3.1e-5
    R_E = 6.371e6

    Dst_avg = -80e-9   # T
    n_substorms = 15
    E_substorm = 5e14  # J per substorm

    # (a) Ring current energy from DPS
    E_rc = abs(Dst_avg) * B0 * R_E**3 / (2 * mu0)

    print(f"\n    Storm parameters: 3 days, avg Dst = {Dst_avg*1e9:.0f} nT")
    print(f"    Substorms: {n_substorms} events, each {E_substorm:.1e} J")

    print(f"\n(a) Ring current energy (DPS relation):")
    print(f"    E_rc = |Dst| * B0 * R_E^3 / (2*mu0)")
    print(f"    E_rc = {abs(Dst_avg):.1e} * {B0:.1e} * ({R_E:.3e})^3 / "
          f"(2*{mu0:.3e})")
    print(f"    E_rc = {E_rc:.3e} J")

    # (b) Total substorm energy
    E_total_substorms = n_substorms * E_substorm
    print(f"\n(b) Total energy from all substorms:")
    print(f"    E_substorms = {n_substorms} * {E_substorm:.1e} = "
          f"{E_total_substorms:.3e} J")

    # (c) Comparison
    ratio = E_total_substorms / E_rc
    print(f"\n(c) Comparison:")
    print(f"    E_substorms / E_ring_current = {ratio:.2f}")
    print(f"    Total substorm energy = {E_total_substorms:.2e} J")
    print(f"    Ring current energy = {E_rc:.2e} J")

    if ratio > 1:
        print(f"\n    Substorms release ~{ratio:.1f}x MORE energy than the ring current")
    else:
        print(f"\n    Substorms release ~{ratio:.1f}x the ring current energy")

    print(f"\n    Discussion:")
    print(f"    - Substorms dissipate ~70-80% of their energy in the ionosphere")
    print(f"      (Joule heating + particle precipitation), not in the ring current")
    print(f"    - Energy going to ring current from substorms: ~20-30% of total")
    print(f"      = {0.25 * E_total_substorms:.2e} J")
    print(f"    - The ring current also receives energy from enhanced convection")
    print(f"      (direct injection from plasma sheet) independent of substorms")
    print(f"    - Both channels contribute: sustained convection provides a")
    print(f"      baseline injection, and substorms provide impulsive enhancements")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
