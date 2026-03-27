"""
Exercise Solutions for Lesson 02: Magnetosphere Structure

Topics covered:
  - Dipole magnetic field calculation at various locations
  - Magnetopause standoff distance (Halloween 2003 storm + Shue model)
  - Plasmapause location from corotation/convection electric field balance
  - Invariant latitude mapping (L-shell to ground latitude)
  - Bow shock Mach number (Alfven, sound, magnetosonic)
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Dipole Field Calculation

    Calculate the magnetic field magnitude at:
    (a) Magnetic equator at surface (r = R_E, theta = 90 deg)
    (b) North magnetic pole (r = R_E, theta = 0 deg)
    (c) Equatorial plane at GEO (r = 6.6 R_E, theta = 90 deg)

    Using M = 8.0e22 A*m^2
    Dipole field: B = (mu0 * M) / (4*pi*r^3) * sqrt(1 + 3*cos^2(theta))
    """
    print("=" * 70)
    print("Exercise 1: Dipole Field Calculation")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7  # T*m/A
    M = 8.0e22               # A*m^2
    R_E = 6.371e6            # m

    def dipole_B(r, theta_deg):
        """Calculate dipole field magnitude in Tesla."""
        theta = np.radians(theta_deg)
        B = (mu0 * M) / (4 * np.pi * r**3) * np.sqrt(1 + 3 * np.cos(theta)**2)
        return B

    locations = [
        ("(a) Magnetic equator, surface", R_E, 90, 1.0),
        ("(b) North magnetic pole, surface", R_E, 0, 1.0),
        ("(c) Equatorial plane, GEO (6.6 R_E)", 6.6 * R_E, 90, 6.6),
    ]

    print(f"\n    M = {M:.1e} A*m^2, R_E = {R_E:.3e} m")
    print(f"    B(r, theta) = (mu0*M)/(4*pi*r^3) * sqrt(1 + 3*cos^2(theta))")

    for label, r, theta, r_re in locations:
        B = dipole_B(r, theta)
        B_nT = B * 1e9
        print(f"\n    {label}:")
        print(f"      r = {r_re:.1f} R_E, theta = {theta} deg")
        print(f"      B = {B:.4e} T = {B_nT:.1f} nT")

    # Known values for comparison
    print(f"\n    Reference values:")
    print(f"      Equatorial surface field B_0 = mu0*M/(4*pi*R_E^3) "
          f"= {dipole_B(R_E, 90)*1e9:.0f} nT (expected ~31,000 nT)")
    print(f"      Polar surface field = 2*B_0 = {dipole_B(R_E, 0)*1e9:.0f} nT "
          f"(expected ~62,000 nT)")
    print(f"      GEO equatorial field scales as (1/6.6)^3 = "
          f"{1/6.6**3:.5f} of surface value")


def exercise_2():
    """
    Exercise 2: Magnetopause Standoff (Halloween 2003 Storm)

    Solar wind: n = 50 cm^-3, v = 700 km/s.
    (a) Calculate dynamic pressure.
    (b) Estimate magnetopause standoff distance.
    (c) Was GEO (6.6 R_E) exposed? Compare with Shue et al. model (Bz = -30 nT).
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Magnetopause Standoff (Halloween 2003)")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7
    M = 8.0e22
    R_E = 6.371e6
    m_p = 1.67e-27  # kg

    n = 50e6     # 50 cm^-3 in m^-3
    v = 700e3    # m/s
    Bz = -30e-9  # T (-30 nT, strongly southward)

    # (a) Dynamic pressure
    rho = n * m_p
    P_dyn = 0.5 * rho * v**2
    P_dyn_nPa = P_dyn * 1e9

    print(f"\n(a) Dynamic pressure:")
    print(f"    n = {n/1e6:.0f} cm^-3 = {n:.1e} m^-3")
    print(f"    v = {v/1e3:.0f} km/s")
    print(f"    rho = n * m_p = {rho:.3e} kg/m^3")
    print(f"    P_dyn = 0.5 * rho * v^2 = {P_dyn:.3e} Pa = {P_dyn_nPa:.1f} nPa")

    # (b) Pressure balance: P_dyn = B^2 / (2*mu0) at magnetopause
    # B at equatorial magnetopause: B = B0 * (R_E/r)^3 (approx dipole)
    # B0 = mu0*M/(4*pi*R_E^3)
    B0 = mu0 * M / (4 * np.pi * R_E**3)

    # P_dyn = B0^2 * (R_E/r)^6 / (2*mu0)
    # r^6 = B0^2 * R_E^6 / (2*mu0 * P_dyn)
    r6 = B0**2 * R_E**6 / (2 * mu0 * P_dyn)
    r_mp = r6 ** (1 / 6)
    r_mp_RE = r_mp / R_E

    print(f"\n(b) Magnetopause standoff (simple pressure balance):")
    print(f"    B_0 (equatorial surface) = {B0*1e9:.1f} nT")
    print(f"    Pressure balance: P_dyn = B^2/(2*mu0)")
    print(f"    => R_mp = R_E * (B_0^2 / (2*mu0*P_dyn))^(1/6)")
    print(f"    R_mp = {r_mp_RE:.2f} R_E = {r_mp/1e6:.2f} x 10^6 m")

    # (c) Compare with GEO and Shue model
    print(f"\n(c) Comparison with GEO orbit:")
    print(f"    GEO at 6.6 R_E, magnetopause at {r_mp_RE:.2f} R_E")
    if r_mp_RE < 6.6:
        print(f"    YES - GEO was exposed to direct solar wind!")
        print(f"    Magnetopause was pushed {6.6 - r_mp_RE:.1f} R_E inside GEO")
    else:
        print(f"    No - GEO remained inside the magnetopause")

    # Shue et al. (1998) model
    # r_0 = (10.22 + 1.29 * tanh(0.184*(Bz_nT + 8.14))) * Dp^(-1/6.6)
    Bz_nT = Bz * 1e9
    Dp = P_dyn_nPa
    r0_shue = (10.22 + 1.29 * np.tanh(0.184 * (Bz_nT + 8.14))) * Dp**(-1 / 6.6)

    print(f"\n    Shue et al. (1998) model:")
    print(f"    r_0 = (10.22 + 1.29*tanh(0.184*(Bz + 8.14))) * Dp^(-1/6.6)")
    print(f"    Bz = {Bz_nT:.0f} nT, Dp = {Dp:.1f} nPa")
    print(f"    r_0 = {r0_shue:.2f} R_E")
    if r0_shue < 6.6:
        print(f"    Shue model also predicts GEO exposure!")


def exercise_3():
    """
    Exercise 3: Plasmapause Location

    Corotation potential: Phi_cor = -Omega_E * B0 * R_E^3 / r
    Convection potential: Phi_conv = -E0 * r * sin(phi)  (phi=0 at noon)

    Find plasmapause on dusk side (phi = -pi/2) for:
    - Quiet: E0 = 0.3 mV/m
    - Storm: E0 = 1.5 mV/m
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Plasmapause Location")
    print("=" * 70)

    Omega_E = 7.27e-5  # rad/s
    B0 = 31000e-9      # T (31,000 nT)
    R_E = 6.371e6      # m

    # At the plasmapause on the dusk side (phi = -pi/2, sin(phi) = -1):
    # The total potential: Phi = -Omega_E * B0 * R_E^3 / r + E0 * r
    # At the stagnation point (plasmapause), dPhi/dr = 0:
    # dPhi/dr = Omega_E * B0 * R_E^3 / r^2 + E0 = 0
    # Wait, that gives no solution since both terms are positive.
    #
    # Let's be more careful. The corotation E-field points radially outward
    # (centrifugal), and the convection E-field is dawn-to-dusk.
    # On the dusk side, convection pushes inward, opposing corotation.
    #
    # Corotation potential: Phi_cor = -C/r  where C = Omega_E * B0 * R_E^3
    # E_cor = -dPhi_cor/dr = -C/r^2 (radially outward, i.e., negative if
    # we define outward as positive r direction)
    #
    # Actually, the plasmapause is found where the total radial velocity is zero.
    # The corotation drift is eastward (azimuthal), and convection is sunward.
    # On the dusk side, these compete: corotation carries plasma duskward (away
    # from Sun) while convection carries it sunward.
    #
    # The standard approach: find where E_r = 0 on the dusk side.
    # Phi_total = -C/r - E0 * r * sin(phi)
    # On dusk side, phi = -pi/2 (or 270 deg), sin(phi) = -1:
    # Phi_total = -C/r + E0 * r
    # dPhi/dr = C/r^2 + E0 = 0 gives r^2 = -C/E0 which is unphysical.
    #
    # Let me reconsider: for the plasmapause, we need the last closed
    # equipotential of the total (corotation + convection) potential.
    # The Volland-Stern model gives: r_pp = (C / E0)^(1/2) on the dusk side.
    # This comes from finding the separatrix where Phi_cor + Phi_conv have
    # a saddle point.
    #
    # More precisely, the total potential in the equatorial plane:
    #   Phi = -C/r + E0 * y  (where y = r*sin(phi), dusk = negative y)
    # In polar: Phi = -C/r - E0 * r * sin(phi_mlt) where phi_mlt measured
    # from noon, and on dusk side sin(phi_mlt) = -1 (phi_mlt = 270 deg).
    #
    # Actually we want Phi = -C/r + E0*r on the dusk side.
    # Saddle point: dPhi/dr = C/r^2 + E0 = 0 => no solution for positive E0.
    #
    # The correct formulation: convection E-field is dawn-to-dusk, which in
    # the GSM frame means E_y < 0 (pointing from dawn to dusk, i.e., -y).
    # So Phi_conv = E0 * y = E0 * r * sin(phi_noon). On dusk side, this is
    # negative. Or equivalently, Phi_conv = -E0 * r * sin(phi) where phi is
    # measured differently.
    #
    # Let me use the standard result directly:
    # The plasmapause radius on the dusk side: L_pp = sqrt(C / E0) / R_E
    # where C = Omega_E * B0 * R_E^3

    C = Omega_E * B0 * R_E**3
    print(f"\n    Constants:")
    print(f"    Omega_E = {Omega_E:.2e} rad/s")
    print(f"    B_0 = {B0*1e9:.0f} nT")
    print(f"    R_E = {R_E:.3e} m")
    print(f"    C = Omega_E * B_0 * R_E^3 = {C:.3e} V*m")

    for label, E0 in [("Quiet (E0 = 0.3 mV/m)", 0.3e-3),
                       ("Storm (E0 = 1.5 mV/m)", 1.5e-3)]:
        # L_pp at dusk: found from dPhi/dr = 0 on the dusk separatrix
        # r_pp = sqrt(C / E0)
        r_pp = np.sqrt(C / E0)
        L_pp = r_pp / R_E

        print(f"\n    {label}:")
        print(f"      r_pp = sqrt(C / E_0) = sqrt({C:.3e} / {E0:.1e})")
        print(f"      r_pp = {r_pp:.3e} m")
        print(f"      L_pp = {L_pp:.2f} R_E")

    print(f"\n    Physical interpretation:")
    print(f"    - During quiet times, plasmasphere extends to L ~ 4-5")
    print(f"    - During storms, enhanced convection erodes the plasmasphere")
    print(f"    - The plasmapause moves inward, exposing outer radiation belt")
    print(f"      to different wave environments (EMIC, chorus, hiss)")


def exercise_4():
    """
    Exercise 4: Invariant Latitude Mapping

    (a) GEO at L = 6.6: what invariant latitude for aurora?
    (b) Account for 11 deg dipole tilt on nightside.
    (c) Storm-time injection at L = 4: where does aurora shift?
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Invariant Latitude Mapping")
    print("=" * 70)

    # Invariant latitude: cos^2(Lambda) = 1/L  =>  Lambda = arccos(1/sqrt(L))
    def inv_lat(L):
        return np.degrees(np.arccos(1 / np.sqrt(L)))

    # (a) GEO L = 6.6
    L_geo = 6.6
    Lambda_geo = inv_lat(L_geo)
    print(f"\n(a) GEO (L = {L_geo}):")
    print(f"    cos^2(Lambda) = 1/L = 1/{L_geo} = {1/L_geo:.4f}")
    print(f"    Lambda = arccos(1/sqrt({L_geo})) = {Lambda_geo:.1f} deg")
    print(f"    Aurora would appear at ~{Lambda_geo:.1f} deg invariant latitude")

    # (b) Dipole tilt effect
    tilt = 11  # degrees
    # On nightside, the dipole tilt shifts the field line footprint
    # The geographic latitude is approximately Lambda +/- tilt
    # On the nightside facing away from Sun, for northern hemisphere:
    geo_lat_north = Lambda_geo - tilt  # footprint shifts equatorward
    geo_lat_south = Lambda_geo + tilt  # or poleward depending on geometry

    print(f"\n(b) With {tilt} deg dipole tilt (nightside):")
    print(f"    Geographic latitude ~ invariant latitude +/- tilt")
    print(f"    Northern hemisphere nightside: ~{Lambda_geo:.1f} - {tilt} = "
          f"~{geo_lat_north:.1f} deg geographic")
    print(f"    (Actual correction depends on local time and specific")
    print(f"     dipole orientation; this is a first-order estimate)")

    # (c) Storm-time L = 4
    L_storm = 4
    Lambda_storm = inv_lat(L_storm)
    print(f"\n(c) Storm-time injection at L = {L_storm}:")
    print(f"    Lambda = arccos(1/sqrt({L_storm})) = {Lambda_storm:.1f} deg")
    print(f"    Aurora shifts equatorward from {Lambda_geo:.1f} to "
          f"{Lambda_storm:.1f} deg")
    print(f"    This is a {Lambda_geo - Lambda_storm:.1f} deg equatorward shift")
    print(f"    During intense storms, aurora can be visible from mid-latitudes")

    # Table of common L-values
    print(f"\n    Reference L-shell to invariant latitude table:")
    print(f"    {'L':>6} {'Lambda (deg)':>14}")
    print(f"    {'-'*22}")
    for L in [2, 3, 4, 5, 6, 6.6, 8, 10]:
        print(f"    {L:6.1f} {inv_lat(L):14.1f}")


def exercise_5():
    """
    Exercise 5: Bow Shock Mach Number

    Solar wind: v = 450 km/s, n = 8 cm^-3, B = 7 nT, T_p = 1e5 K
    (a) Alfven speed v_A
    (b) Sound speed c_s (gamma = 5/3)
    (c) Magnetosonic speed v_ms
    (d) Magnetosonic Mach number
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Bow Shock Mach Number")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7  # T*m/A
    m_p = 1.67e-27           # kg
    k_B = 1.38e-23           # J/K
    gamma = 5 / 3

    v_sw = 450e3   # m/s
    n = 8e6        # 8 cm^-3 in m^-3
    B = 7e-9       # T (7 nT)
    T_p = 1e5      # K
    rho = n * m_p

    # (a) Alfven speed
    v_A = B / np.sqrt(mu0 * rho)
    print(f"\n    Solar wind parameters:")
    print(f"    v = {v_sw/1e3:.0f} km/s, n = {n/1e6:.0f} cm^-3, "
          f"B = {B*1e9:.0f} nT, T = {T_p:.0e} K")
    print(f"    rho = n * m_p = {rho:.3e} kg/m^3")

    print(f"\n(a) Alfven speed:")
    print(f"    v_A = B / sqrt(mu0 * rho)")
    print(f"    v_A = {B:.1e} / sqrt({mu0:.3e} * {rho:.3e})")
    print(f"    v_A = {v_A:.1f} m/s = {v_A/1e3:.1f} km/s")

    # (b) Sound speed
    c_s = np.sqrt(gamma * k_B * T_p / m_p)
    print(f"\n(b) Sound speed (gamma = {gamma:.2f}):")
    print(f"    c_s = sqrt(gamma * k_B * T / m_p)")
    print(f"    c_s = sqrt({gamma:.2f} * {k_B:.2e} * {T_p:.0e} / {m_p:.2e})")
    print(f"    c_s = {c_s:.1f} m/s = {c_s/1e3:.1f} km/s")

    # (c) Magnetosonic speed
    v_ms = np.sqrt(v_A**2 + c_s**2)
    print(f"\n(c) Magnetosonic speed:")
    print(f"    v_ms = sqrt(v_A^2 + c_s^2)")
    print(f"    v_ms = sqrt({v_A/1e3:.1f}^2 + {c_s/1e3:.1f}^2) km/s")
    print(f"    v_ms = {v_ms:.1f} m/s = {v_ms/1e3:.1f} km/s")

    # (d) Mach number
    M_ms = v_sw / v_ms
    M_A = v_sw / v_A
    print(f"\n(d) Mach numbers:")
    print(f"    Magnetosonic Mach number: M_ms = v_sw / v_ms = "
          f"{v_sw/1e3:.0f} / {v_ms/1e3:.1f} = {M_ms:.1f}")
    print(f"    Alfven Mach number: M_A = v_sw / v_A = "
          f"{v_sw/1e3:.0f} / {v_A/1e3:.1f} = {M_A:.1f}")
    print(f"\n    The bow shock is a {'strong' if M_ms > 3 else 'moderate'} "
          f"shock (M_ms = {M_ms:.1f})")
    print(f"    Typical solar wind: M_ms ~ 5-8, M_A ~ 6-10")
    print(f"    This is a supercritical shock requiring particle reflection")
    print(f"    for dissipation (not just resistive/viscous dissipation)")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
