"""
Exercises for Lesson 13: Solar Spectroscopy and Instruments
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np


# --- Physical constants ---
c = 3.0e8              # speed of light [m/s]
R_sun = 6.957e8        # solar radius [m]
AU = 1.496e11          # astronomical unit [m]


def exercise_1():
    """
    Problem 1: Doppler Shift Velocity

    Fe I 6173 A line shifted by +0.021 A in sunspot penumbra.
    Calculate LOS velocity, direction, and Evershed flow interpretation.
    """
    lambda_0 = 6173.0e-10  # rest wavelength [m]
    delta_lambda = 0.021e-10  # shift [m] (0.021 A)

    # Doppler formula: delta_lambda / lambda_0 = v / c
    v_LOS = c * delta_lambda / lambda_0
    v_LOS_kms = v_LOS / 1e3

    print(f"  Fe I line: lambda_0 = 6173.0 A")
    print(f"  Observed shift: +0.021 A (positive = redshift)")
    print(f"  ")
    print(f"  v_LOS = c * delta_lambda / lambda_0")
    print(f"        = {c:.0e} * {delta_lambda*1e10:.3f}e-10 / {lambda_0*1e10:.1f}e-10")
    print(f"        = {v_LOS:.0f} m/s = {v_LOS_kms:.2f} km/s")
    print(f"  ")
    print(f"  Direction: Positive shift = REDSHIFT = motion AWAY from observer")
    print(f"  ")
    print(f"  Evershed flow interpretation:")
    print(f"  Near the solar limb, the line-of-sight is approximately tangent")
    print(f"  to the solar surface. The Evershed flow is a radial outflow")
    print(f"  in the penumbra, parallel to the photospheric surface.")
    print(f"  Near the limb, this horizontal flow has a large LOS component.")
    print(f"  A redshift on the limb-side penumbra indicates outward (away from")
    print(f"  sunspot center) flow directed away from us -- consistent with the")
    print(f"  Evershed effect (horizontal outflow in the penumbra at ~{v_LOS_kms:.1f} km/s).")


def exercise_2():
    """
    Problem 2: Coronal Loop Temperature from AIA Channel Ratio

    171/193 intensity ratio = 0.8.
    171 A (Fe IX, peak at 0.7 MK), 193 A (Fe XII, peak at 1.5 MK).
    Qualitative temperature assessment.
    If bright in 94 A (Fe XVIII, ~7 MK), what does it mean?
    """
    print(f"  SDO/AIA channel responses:")
    print(f"    171 A: Fe IX, peak at ~0.7 MK (cooler coronal plasma)")
    print(f"    193 A: Fe XII, peak at ~1.5 MK (typical coronal temperature)")
    print(f"    94 A:  Fe XVIII, peak at ~7 MK (hot, flare-temperature plasma)")
    print(f"  ")
    print(f"  Measured 171/193 ratio = 0.8")
    print(f"  ")
    print(f"  Qualitative interpretation:")
    print(f"  - If the loop were at 0.7 MK (peak of 171), the 171/193 ratio")
    print(f"    would be >> 1 (strong in 171, weak in 193)")
    print(f"  - If the loop were at 1.5 MK (peak of 193), the ratio would be < 1")
    print(f"    (moderate in 171, strong in 193)")
    print(f"  - A ratio of 0.8 (slightly below 1) suggests the loop temperature")
    print(f"    is between 0.7 and 1.5 MK, closer to ~1-1.2 MK")
    print(f"  - Both channels have significant response, indicating intermediate T")
    print(f"  ")
    print(f"  If ALSO bright in 94 A (Fe XVIII, ~7 MK):")
    print(f"  - This indicates a HOT component in addition to the ~1 MK plasma")
    print(f"  - Possible scenarios:")
    print(f"    1. Multi-thermal loop: unresolved hot and cool strands")
    print(f"    2. Recently flared loop with cooling hot plasma")
    print(f"    3. Nanoflare-heated plasma creating a hot component")
    print(f"  - Note: AIA 94 A also has a secondary cool response at ~1 MK,")
    print(f"    so need to check if the brightness is from Fe XVIII or Fe X")


def exercise_3():
    """
    Problem 3: Type II Radio Burst

    Burst: 150 MHz -> 30 MHz over 10 minutes.
    Newkirk model: n_e(r) = n_0 * 10^(4.32 * R_sun / r), n_0 = 4.2e4 cm^-3.
    Estimate heights and shock speed.
    """
    f_start = 150.0e6   # Hz
    f_end = 30.0e6       # Hz
    dt = 10.0 * 60.0     # seconds

    # Plasma frequency: f_p = 9000 * sqrt(n_e) Hz (n_e in cm^-3)
    # => n_e = (f_p / 9000)^2 cm^-3
    # For fundamental emission: f = f_p
    # For harmonic emission: f = 2*f_p (use fundamental here)

    n0 = 4.2e4  # cm^-3

    # Electron densities at start and end
    n_start = (f_start / 9000.0)**2  # cm^-3
    n_end = (f_end / 9000.0)**2

    print(f"  Type II burst: {f_start/1e6:.0f} MHz -> {f_end/1e6:.0f} MHz in {dt/60:.0f} min")
    print(f"  Assuming fundamental plasma emission (f = f_p)")
    print(f"  ")
    print(f"  Plasma frequency: f_p = 9000 sqrt(n_e) Hz  (n_e in cm^-3)")
    print(f"  => n_e = (f/9000)^2")
    print(f"  ")
    print(f"  At start ({f_start/1e6:.0f} MHz): n_e = {n_start:.2e} cm^-3")
    print(f"  At end ({f_end/1e6:.0f} MHz):   n_e = {n_end:.2e} cm^-3")

    # Newkirk model: n_e(r) = n_0 * 10^(4.32 * R_sun / r)
    # Solve for r: r = 4.32 * R_sun / log10(n_e / n_0)
    r_start = 4.32 * R_sun / np.log10(n_start / n0)
    r_end = 4.32 * R_sun / np.log10(n_end / n0)

    h_start = r_start - R_sun  # height above surface
    h_end = r_end - R_sun

    print(f"\n  Newkirk model: n_e(r) = {n0:.1e} * 10^(4.32 R_sun/r)")
    print(f"  Heights:")
    print(f"    Start: r = {r_start/R_sun:.2f} R_sun, h = {h_start/R_sun:.2f} R_sun = {h_start/1e6:.0f} Mm")
    print(f"    End:   r = {r_end/R_sun:.2f} R_sun, h = {h_end/R_sun:.2f} R_sun = {h_end/1e6:.0f} Mm")

    # Average shock speed
    dr = r_end - r_start
    v_shock = dr / dt
    v_shock_kms = v_shock / 1e3

    print(f"\n  Radial distance traveled: dr = {dr/1e6:.0f} Mm")
    print(f"  Time: dt = {dt:.0f} s = {dt/60:.0f} min")
    print(f"  Average shock speed: v = dr/dt = {v_shock_kms:.0f} km/s")
    print(f"  This is typical of a CME-driven shock in the low corona.")


def exercise_4():
    """
    Problem 4: Density Diagnostic with Si X Line Ratio

    Si X: 356 A / 347 A ratio.
    Region A: ratio = 1.5, Region B: ratio = 3.0.
    Critical density n_crit ~ 1e10 cm^-3.
    """
    ratio_A = 1.5
    ratio_B = 3.0
    n_crit = 1.0e10  # cm^-3

    print(f"  Si X density-sensitive line ratio (356 A / 347 A):")
    print(f"  Region A: ratio = {ratio_A}")
    print(f"  Region B: ratio = {ratio_B}")
    print(f"  ")
    print(f"  The 356/347 ratio INCREASES with electron density.")
    print(f"  Therefore:")
    print(f"  Region B (ratio = {ratio_B}) has HIGHER density than Region A (ratio = {ratio_A})")
    print(f"  ")
    print(f"  Critical density analysis:")
    print(f"  n_crit ~ {n_crit:.0e} cm^-3 for the 356 A transition")
    print(f"  ")
    print(f"  - At n_e << n_crit: the upper level of the 356 A transition is")
    print(f"    primarily depopulated by radiative decay. The ratio is relatively")
    print(f"    insensitive to density (low-density limit). Region A with ratio = {ratio_A}")
    print(f"    may be in or near this regime.")
    print(f"  ")
    print(f"  - At n_e ~ n_crit: collisional de-excitation becomes important,")
    print(f"    and the ratio is most sensitive to density. Region B with ratio = {ratio_B}")
    print(f"    suggests n_e is near or above n_crit.")
    print(f"  ")
    print(f"  - At n_e >> n_crit: the ratio saturates at the high-density limit")
    print(f"    (Boltzmann distribution between levels). The ratio becomes")
    print(f"    insensitive to density again.")
    print(f"  ")
    print(f"  Density estimates:")
    print(f"    Region A: n_e ~ few x 10^9 cm^-3 (below n_crit)")
    print(f"    Region B: n_e ~ 10^10 cm^-3 or above (near/above n_crit)")
    print(f"  ")
    print(f"  Region B could be a coronal loop footpoint, flare plasma,")
    print(f"  or a denser structure like a prominence-corona interface.")


def exercise_5():
    """
    Problem 5: DKIST Resolution

    4m mirror at 1074.7 nm (Fe XIII coronal line).
    Calculate diffraction limit, compare with SDO/AIA.
    """
    D = 4.0              # mirror diameter [m]
    lambda_vis = 500.0e-9  # visible wavelength for comparison [m]
    lambda_IR = 1074.7e-9  # Fe XIII coronal line [m]
    arcsec_to_km = 725.0   # 1 arcsec = 725 km on the Sun

    # Diffraction limit: theta = 1.22 * lambda / D [radians]
    theta_vis = 1.22 * lambda_vis / D  # rad
    theta_IR = 1.22 * lambda_IR / D    # rad

    # Convert to arcsec: 1 rad = 206265 arcsec
    theta_vis_arcsec = theta_vis * 206265.0
    theta_IR_arcsec = theta_IR * 206265.0

    # Spatial resolution on the Sun
    res_vis_km = theta_vis_arcsec * arcsec_to_km
    res_IR_km = theta_IR_arcsec * arcsec_to_km

    print(f"  DKIST: D = {D:.0f} m mirror")
    print(f"  ")
    print(f"  At 500 nm (visible):")
    print(f"    theta = 1.22 * lambda/D = 1.22 * {lambda_vis*1e9:.0f} nm / {D:.0f} m")
    print(f"          = {theta_vis_arcsec:.3f} arcsec")
    print(f"    Spatial resolution: {res_vis_km:.0f} km on the Sun")
    print(f"  ")
    print(f"  At 1074.7 nm (Fe XIII coronal line):")
    print(f"    theta = 1.22 * {lambda_IR*1e9:.1f} nm / {D:.0f} m")
    print(f"          = {theta_IR_arcsec:.3f} arcsec")
    print(f"    Spatial resolution: {res_IR_km:.0f} km on the Sun")

    # SDO/AIA comparison
    AIA_res = 1.2  # arcsec
    AIA_km = AIA_res * arcsec_to_km

    print(f"\n  SDO/AIA comparison:")
    print(f"    AIA resolution: {AIA_res} arcsec = {AIA_km:.0f} km at 17.1 nm")
    print(f"    DKIST at 500 nm: {theta_vis_arcsec:.3f} arcsec = {res_vis_km:.0f} km")
    print(f"    DKIST at 1074.7 nm: {theta_IR_arcsec:.3f} arcsec = {res_IR_km:.0f} km")

    print(f"\n  Granule size: ~1000 km")
    print(f"  DKIST at 500 nm: {1000/res_vis_km:.0f} resolution elements across a granule")
    print(f"  ")
    print(f"  Bright points (100-200 km):")
    bright_min, bright_max = 100.0, 200.0
    print(f"  DKIST at 500 nm can resolve these: {res_vis_km:.0f} km < {bright_min:.0f} km")

    print(f"\n  Why DKIST at 1074.7 nm is still valuable:")
    print(f"    - It observes CORONAL magnetic fields (Fe XIII forbidden line)")
    print(f"    - No other facility can measure coronal B fields with this resolution")
    print(f"    - The ~{res_IR_km:.0f} km resolution resolves coronal loops (~1-10 Mm)")
    print(f"    - Spectropolarimetry gives both temperature and magnetic field")
    print(f"    - Ground-based IR avoids the EUV/X-ray requirement of space")
    print(f"    - Complements AIA's EUV imaging with magnetic field measurements")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Doppler Shift Velocity ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Coronal Loop Temperature ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Type II Radio Burst ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Density Diagnostic ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: DKIST Resolution ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
