"""
Exercises for Lesson 08: Active Regions and Sunspots
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np


# --- Physical constants ---
k_B = 1.381e-23        # Boltzmann constant [J/K]
sigma_SB = 5.670e-8    # Stefan-Boltzmann constant [W m^-2 K^-4]
R_sun = 6.957e8        # solar radius [m]
mu_0 = 4.0e-7 * np.pi  # vacuum permeability [T m/A]

# 1 MSH = 3.04e12 m^2 (millionths of solar hemisphere)
MSH = 3.04e12  # m^2


def exercise_1():
    """
    Problem 1: Sunspot Brightness and Temperature

    Umbra T_u = 3800 K, photosphere T_p = 5778 K.
    (a) Brightness ratio I_u/I_p (Stefan-Boltzmann).
    (b) Missing luminosity for A_u = 100 MSH.
    (c) Where does the missing energy go?
    """
    T_u = 3800.0     # umbral temperature [K]
    T_p = 5778.0     # photospheric temperature [K]

    # (a) For blackbodies: I ~ sigma T^4 => I_u/I_p = (T_u/T_p)^4
    ratio = (T_u / T_p)**4
    print(f"  (a) T_umbra = {T_u:.0f} K, T_photosphere = {T_p:.0f} K")
    print(f"      I_u/I_p = (T_u/T_p)^4 = ({T_u}/{T_p})^4 = {ratio:.4f}")
    print(f"      The umbra radiates only {ratio*100:.1f}% of the photospheric intensity.")

    # (b) Missing luminosity
    A_u = 100.0 * MSH   # area in m^2
    F_p = sigma_SB * T_p**4   # photospheric flux [W/m^2]
    F_u = sigma_SB * T_u**4   # umbral flux [W/m^2]
    L_missing = (F_p - F_u) * A_u

    print(f"\n  (b) Umbral area: A_u = 100 MSH = {A_u:.3e} m^2")
    print(f"      Photospheric flux: F_p = sigma T_p^4 = {F_p:.3e} W/m^2")
    print(f"      Umbral flux: F_u = sigma T_u^4 = {F_u:.3e} W/m^2")
    print(f"      Missing luminosity: (F_p - F_u) * A_u = {L_missing:.3e} W")
    print(f"      As fraction of L_sun: {L_missing/3.828e26:.2e}")

    # (c) Where does it go?
    print(f"\n  (c) The 'missing' energy does NOT disappear. It is redistributed:")
    print(f"      - The strong magnetic field suppresses convection in the sunspot")
    print(f"      - Energy that would have been carried to the surface by convection")
    print(f"        is diverted to the surrounding photosphere ('bright ring')")
    print(f"      - Observations show a slight brightening around sunspot groups")
    print(f"      - The total solar irradiance actually INCREASES during solar maximum")
    print(f"        because faculae (bright magnetic features) overcompensate for")
    print(f"        the sunspot deficit on timescales longer than ~days")


def exercise_2():
    """
    Problem 2: Wilson Depression

    B = 2500 G at umbral edge. Pressure balance.
    (a) Magnetic pressure in dyn/cm^2 and Pa.
    (b) Fraction of external pressure.
    (c) Wilson depression estimate.
    """
    B = 2500.0e-4      # T (2500 G)
    B_G = 2500.0       # Gauss
    p_ext_cgs = 1.2e5  # dyn/cm^2
    H_p = 150.0e3      # pressure scale height [m] = 150 km

    # (a) Magnetic pressure: P_mag = B^2 / (2 mu_0) [SI] = B^2 / (8 pi) [CGS]
    P_mag_SI = B**2 / (2.0 * mu_0)  # Pa
    P_mag_cgs = B_G**2 / (8.0 * np.pi)  # dyn/cm^2

    print(f"  (a) B = {B_G:.0f} G = {B:.4f} T")
    print(f"      Magnetic pressure (CGS): B^2/(8 pi) = {P_mag_cgs:.2e} dyn/cm^2")
    print(f"      Magnetic pressure (SI):  B^2/(2 mu_0) = {P_mag_SI:.2e} Pa")
    print(f"      Cross-check: {P_mag_cgs} dyn/cm^2 * 0.1 = {P_mag_cgs * 0.1:.2e} Pa")

    # (b) Fraction of external pressure
    frac = P_mag_cgs / p_ext_cgs
    print(f"\n  (b) External photospheric pressure: p_ext = {p_ext_cgs:.1e} dyn/cm^2")
    print(f"      P_mag / p_ext = {frac:.3f}")
    print(f"      Magnetic pressure is {frac*100:.1f}% of the external gas pressure.")
    print(f"      The internal gas pressure must be reduced by this amount.")

    # (c) Wilson depression
    # Pressure balance: p_ext(z) = p_int(z) + B^2/(8pi)
    # At the external tau=1 surface: p_ext = p_ext_surface
    # Inside the sunspot at the same geometric height:
    #   p_int = p_ext - P_mag < p_ext
    # The internal tau=1 surface is BELOW the external one because
    # we need to go deeper to reach the same gas pressure.
    # The depression: delta_z = H_p * ln(p_ext / (p_ext - P_mag))
    # Approximately: delta_z ~ H_p * P_mag / p_ext for P_mag << p_ext

    depression_exact = H_p * np.log(p_ext_cgs / (p_ext_cgs - P_mag_cgs))
    depression_approx = H_p * frac
    depression_km = depression_exact / 1e3

    print(f"\n  (c) Wilson depression estimate:")
    print(f"      delta_z = H_p * ln(p_ext / (p_ext - P_mag))")
    print(f"             = {H_p/1e3:.0f} km * ln({p_ext_cgs:.1e} / {p_ext_cgs - P_mag_cgs:.1e})")
    print(f"             = {depression_km:.0f} km")
    print(f"      Approximate: delta_z ~ H_p * P_mag/p_ext = {depression_approx/1e3:.0f} km")
    print(f"      Observed Wilson depressions are typically 400-800 km,")
    print(f"      consistent with this estimate.")


def exercise_3():
    """
    Problem 3: Magnetic Flux Emergence

    dPhi/dt = 5e19 Mx/hour for 48 hours.
    (a) Total flux after 48 hours.
    (b) Average field strength for R = 20 Mm.
    (c) Rise speed from tachocline in ~2 months.
    """
    dPhi_dt = 5.0e19     # Mx/hour
    t_emerge = 48.0       # hours
    R_AR = 20.0e8         # 20 Mm in cm (CGS for Mx)

    # (a) Total flux
    Phi_total = dPhi_dt * t_emerge
    print(f"  (a) Flux emergence rate: dPhi/dt = {dPhi_dt:.1e} Mx/hour")
    print(f"      Duration: {t_emerge:.0f} hours")
    print(f"      Total unsigned flux: Phi = {Phi_total:.2e} Mx")

    # (b) Average vertical field
    R_AR_m = 20.0e6   # 20 Mm in meters
    A_AR = np.pi * R_AR_m**2  # m^2

    # 1 Mx = 1 G cm^2 = 1e-8 T m^2 = 1e-8 Wb
    Phi_Wb = Phi_total * 1.0e-8  # Wb
    B_avg = Phi_Wb / A_AR  # T
    B_avg_G = B_avg * 1.0e4  # Gauss

    print(f"\n  (b) AR radius: R = {R_AR_m/1e6:.0f} Mm")
    print(f"      AR area: A = pi R^2 = {A_AR:.3e} m^2")
    print(f"      Total flux: {Phi_Wb:.2e} Wb")
    print(f"      Average |B_z| = Phi / A = {B_avg_G:.0f} G")

    # (c) Rise speed from tachocline
    depth = 0.3 * R_sun  # from 0.7 R_sun to surface
    t_rise = 2.0 * 30.0 * 24.0 * 3600.0  # 2 months in seconds
    v_rise = depth / t_rise

    print(f"\n  (c) Rise from tachocline (0.7 R_sun) to surface:")
    print(f"      Distance: 0.3 R_sun = {depth:.3e} m = {depth/1e6:.0f} Mm")
    print(f"      Time: ~2 months = {t_rise:.3e} s")
    print(f"      Average rise speed: v = {v_rise:.0f} m/s = {v_rise/1e3:.2f} km/s")
    print(f"      This is much slower than the convective velocity (~100 m/s)")
    print(f"      and much slower than the sound speed (~100 km/s).")
    print(f"      The rise is controlled by magnetic buoyancy and drag.")


def exercise_4():
    """
    Problem 4: Active Region Decay

    A_0 = 400 MSH, decay rate dA/dt = -25 MSH/day.
    (a) Sunspot lifetime.
    (b) Flux removal rate.
    (c) Compare with quiet Sun flux recycling.
    """
    A_0 = 400.0         # MSH
    dA_dt = -25.0        # MSH/day
    Phi = 5.0e21         # Mx

    # (a) Lifetime (linear decay to zero)
    t_life = A_0 / abs(dA_dt)
    print(f"  (a) Initial area: A_0 = {A_0:.0f} MSH")
    print(f"      Decay rate: dA/dt = {dA_dt:.0f} MSH/day")
    print(f"      Lifetime: t = A_0 / |dA/dt| = {t_life:.0f} days")

    # (b) Flux removal rate (proportional to area)
    # dPhi/dt = Phi * (dA/dt) / A_0  (if flux proportional to area)
    dPhi_dt = Phi * abs(dA_dt) / A_0  # Mx/day
    dPhi_dt_per_s = dPhi_dt / (24.0 * 3600.0)  # Mx/s

    print(f"\n  (b) Total flux: Phi = {Phi:.1e} Mx")
    print(f"      If flux proportional to area:")
    print(f"      dPhi/dt = Phi * |dA/dt| / A_0")
    print(f"             = {dPhi_dt:.2e} Mx/day")
    print(f"             = {dPhi_dt_per_s:.2e} Mx/s")

    # (c) Compare with quiet Sun flux recycling
    Phi_QS = 1.0e24     # total quiet Sun unsigned flux [Mx]
    t_recycle_hr = 14.0  # hours
    t_recycle_s = t_recycle_hr * 3600.0

    dPhi_QS = Phi_QS / t_recycle_s  # Mx/s
    ratio = dPhi_dt_per_s / dPhi_QS

    print(f"\n  (c) Quiet Sun flux recycling:")
    print(f"      Total QS flux: ~{Phi_QS:.0e} Mx")
    print(f"      Recycling time: {t_recycle_hr:.0f} hours")
    print(f"      QS flux recycling rate: {dPhi_QS:.2e} Mx/s")
    print(f"      Sunspot flux removal: {dPhi_dt_per_s:.2e} Mx/s")
    print(f"      Ratio (sunspot/QS): {ratio:.2e}")
    print(f"      The quiet Sun flux recycling rate far exceeds the individual")
    print(f"      sunspot decay rate -- the 'magnetic carpet' is continuously")
    print(f"      replacing itself on a ~14-hour timescale.")


def exercise_5():
    """
    Problem 5: Quiet Sun Magnetic Energy

    Internetwork: <|B|> ~ 20 G over 90% of surface, height 500 km.
    Network: <|B|> ~ 200 G over 10% of surface, height 2 Mm.
    (a) Magnetic energy density for each.
    (b) Total magnetic energy over the full solar surface.
    (c) Energy dissipation rate if recycled every 14 hours.
    """
    B_inet = 20.0       # G (internetwork)
    B_net = 200.0        # G (network)
    f_net = 0.10         # network area fraction
    f_inet = 0.90        # internetwork area fraction
    h_inet = 500.0e3     # m (height extent of internetwork field)
    h_net = 2.0e6        # m (height extent of network field)
    A_sun = 6.08e22      # cm^2 (solar surface area in CGS)
    A_sun_m2 = 6.08e18   # m^2

    # (a) Magnetic energy density: u_B = B^2 / (8 pi) [CGS, erg/cm^3]
    u_inet = B_inet**2 / (8.0 * np.pi)   # erg/cm^3
    u_net = B_net**2 / (8.0 * np.pi)     # erg/cm^3

    print(f"  (a) Internetwork: <|B|> = {B_inet:.0f} G")
    print(f"      u_B = B^2/(8 pi) = {u_inet:.2f} erg/cm^3")
    print(f"      Network: <|B|> = {B_net:.0f} G")
    print(f"      u_B = B^2/(8 pi) = {u_net:.1f} erg/cm^3")

    # (b) Total magnetic energy
    # E_inet = u_inet * f_inet * A_sun * h_inet_cm
    h_inet_cm = h_inet * 100.0   # m to cm
    h_net_cm = h_net * 100.0

    E_inet = u_inet * f_inet * A_sun * h_inet_cm
    E_net = u_net * f_net * A_sun * h_net_cm
    E_total = E_inet + E_net

    print(f"\n  (b) Total magnetic energy:")
    print(f"      Internetwork: E = u_B * f * A_sun * h")
    print(f"        = {u_inet:.2f} * {f_inet} * {A_sun:.2e} * {h_inet_cm:.2e}")
    print(f"        = {E_inet:.2e} erg")
    print(f"      Network: E = u_B * f * A_sun * h")
    print(f"        = {u_net:.1f} * {f_net} * {A_sun:.2e} * {h_net_cm:.2e}")
    print(f"        = {E_net:.2e} erg")
    print(f"      Total: E = {E_total:.2e} erg")

    # (c) Dissipation rate
    t_recycle = 14.0 * 3600.0  # 14 hours in seconds
    P_total_erg = E_total / t_recycle  # erg/s
    P_total_W = P_total_erg * 1.0e-7  # W

    # Per unit area (in W/m^2)
    P_per_area = P_total_W / A_sun_m2

    print(f"\n  (c) If recycled every 14 hours:")
    print(f"      Total dissipation rate: P = E/t = {P_total_erg:.2e} erg/s")
    print(f"                             = {P_total_W:.2e} W")
    print(f"      Per unit area: P/A = {P_per_area:.0f} W/m^2")
    print(f"      Coronal heating requirement: ~300 W/m^2")
    ratio = P_per_area / 300.0
    print(f"      Ratio: {ratio:.2f}")
    if ratio > 1:
        print(f"      The magnetic carpet energy dissipation ({P_per_area:.0f} W/m^2)")
        print(f"      exceeds the coronal heating requirement -- sufficient in principle!")
    else:
        print(f"      The magnetic carpet provides ~{ratio*100:.0f}% of the requirement.")
        print(f"      It could be a significant contributor to quiet Sun heating.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Sunspot Brightness and Temperature ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Wilson Depression ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Magnetic Flux Emergence ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Active Region Decay ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Quiet Sun Magnetic Energy ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
