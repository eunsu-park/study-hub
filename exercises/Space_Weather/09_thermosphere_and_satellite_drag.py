"""
Exercise Solutions for Lesson 09: Thermosphere and Satellite Drag

Topics covered:
  - Scale height and density ratio
  - Drag force comparison (ISS vs Starlink)
  - CubeSat orbital lifetime estimation
  - Starlink February 2022 storm loss analysis
  - Joule heating power estimation
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Scale Height and Density Ratio

    At 400 km, T_inf = 800 K, dominant species: atomic oxygen (m_O = 16 u).
    (a) Scale height H.
    (b) Density ratio between 400 km and 500 km.
    (c) Storm: T_inf = 1200 K; density at 500 km vs pre-storm 400 km.
    """
    print("=" * 70)
    print("Exercise 1: Scale Height and Density Ratio")
    print("=" * 70)

    k_B = 1.38e-23   # J/K
    u = 1.66e-27      # kg (atomic mass unit)
    g = 8.7            # m/s^2
    m_O = 16 * u       # kg

    T_quiet = 800      # K
    T_storm = 1200     # K

    # (a) Scale height H = k_B * T / (m * g)
    H_quiet = k_B * T_quiet / (m_O * g)
    H_km = H_quiet / 1e3

    print(f"\n    T_inf = {T_quiet} K, m_O = 16 u = {m_O:.3e} kg, g = {g} m/s^2")

    print(f"\n(a) Scale height at quiet time:")
    print(f"    H = k_B * T / (m * g)")
    print(f"    = {k_B:.2e} * {T_quiet} / ({m_O:.3e} * {g})")
    print(f"    = {H_quiet:.0f} m = {H_km:.1f} km")

    # (b) Density ratio 400 -> 500 km (Dz = 100 km)
    Dz = 100e3  # m
    ratio_quiet = np.exp(-Dz / H_quiet)

    print(f"\n(b) Density ratio between 400 km and 500 km:")
    print(f"    rho(500)/rho(400) = exp(-Dz/H) = exp(-{Dz/1e3:.0f} km / {H_km:.1f} km)")
    print(f"    = exp(-{Dz/H_quiet:.2f}) = {ratio_quiet:.4f}")
    print(f"    Density drops by factor {1/ratio_quiet:.1f} over 100 km")

    # (c) Storm conditions: T = 1200 K
    H_storm = k_B * T_storm / (m_O * g)
    H_storm_km = H_storm / 1e3
    ratio_storm = np.exp(-Dz / H_storm)

    print(f"\n(c) Storm conditions (T_inf = {T_storm} K):")
    print(f"    H_storm = k_B * {T_storm} / (m*g) = {H_storm:.0f} m = {H_storm_km:.1f} km")
    print(f"    rho(500)/rho(400)_storm = exp(-{Dz/1e3:.0f}/{H_storm_km:.1f})")
    print(f"    = {ratio_storm:.4f}")

    # Compare storm 500 km density to quiet 400 km
    # During storm, density at same altitude also increases.
    # rho_storm(400) / rho_quiet(400) ~ exp(Dz * (1/H_q - 1/H_s) * something)
    # Simplified: if the base density scales with T (which it doesn't exactly),
    # but a reasonable approach: the density at 500 km during storm vs 400 km quiet:
    # rho_storm(500) / rho_quiet(400) = [rho_storm(400)/rho_quiet(400)] * ratio_storm
    # The scale height increase means less exponential falloff.
    # Using the standard barometric formula from a reference altitude h0:
    # Increased T -> larger H -> more density at altitude
    # A reasonable estimate: rho(500,storm)/rho(400,quiet)
    # = exp(- Dz/H_storm) / exp(0) = ratio_storm (if base density unchanged)
    # But actually the base density increases too. Approximately:
    # rho(h) ~ rho_0 * exp(-(h-h0)/H)
    # The total density at 500 km during storm is higher by the "lifting" factor

    print(f"\n    Comparison: storm 500 km density vs quiet 400 km density:")
    print(f"    Quiet: rho(500)/rho(400) = {ratio_quiet:.4f}")
    print(f"    Storm: rho(500)/rho(400) = {ratio_storm:.4f}")
    print(f"    The storm density at 500 km = {ratio_storm:.4f} * rho_storm(400)")
    print(f"    Since rho_storm(400) ~ rho_quiet(400) * (T_storm/T_quiet)^n (n~1-2)")
    print(f"    Factor from base increase ~ {T_storm/T_quiet:.2f} (at least)")
    print(f"    Net: rho_storm(500) ~ {ratio_storm * T_storm/T_quiet:.3f} * rho_quiet(400)")
    print(f"    So storm density at 500 km can approach or exceed quiet density at 400 km")
    print(f"    This is why storms cause dramatically increased drag at higher altitudes")


def exercise_2():
    """
    Exercise 2: Drag Force Comparison (ISS vs Starlink)

    At 400 km: rho = 5e-12 kg/m^3, v = 7.67 km/s.
    ISS: CdA = 3520 m^2, mass = 420000 kg.
    Starlink: CdA = 22 m^2, mass = 306 kg.
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Drag Force Comparison")
    print("=" * 70)

    rho = 5e-12    # kg/m^3
    v = 7.67e3     # m/s
    T_orbit = 92 * 60  # 92 minutes

    # ISS
    CdA_iss = 3520  # m^2
    m_iss = 420000  # kg

    # Starlink
    CdA_sl = 22     # m^2
    m_sl = 306      # kg

    # (a) Drag force: F = 0.5 * rho * v^2 * CdA
    F_iss = 0.5 * rho * v**2 * CdA_iss
    F_sl = 0.5 * rho * v**2 * CdA_sl

    print(f"\n    At 400 km: rho = {rho:.1e} kg/m^3, v = {v/1e3:.2f} km/s")

    print(f"\n(a) Drag force on ISS (CdA = {CdA_iss} m^2):")
    print(f"    F = 0.5 * rho * v^2 * CdA")
    print(f"    = 0.5 * {rho:.1e} * ({v:.0f})^2 * {CdA_iss}")
    print(f"    = {F_iss:.4f} N")

    print(f"\n(b) Drag force on Starlink (CdA = {CdA_sl} m^2):")
    print(f"    F = 0.5 * {rho:.1e} * ({v:.0f})^2 * {CdA_sl}")
    print(f"    = {F_sl:.5f} N")

    # (c) Drag acceleration
    a_iss = F_iss / m_iss
    a_sl = F_sl / m_sl

    print(f"\n(c) Drag acceleration:")
    print(f"    ISS: a = F/m = {F_iss:.4f}/{m_iss} = {a_iss:.3e} m/s^2")
    print(f"    Starlink: a = F/m = {F_sl:.5f}/{m_sl} = {a_sl:.3e} m/s^2")
    print(f"    Ratio: a_Starlink / a_ISS = {a_sl/a_iss:.1f}")
    print(f"    Starlink experiences {a_sl/a_iss:.1f}x greater deceleration!")

    # Ballistic coefficients
    BC_iss = m_iss / CdA_iss
    BC_sl = m_sl / CdA_sl
    print(f"\n    Ballistic coefficients:")
    print(f"    ISS: B_C = m/(CdA) = {BC_iss:.1f} kg/m^2")
    print(f"    Starlink: B_C = m/(CdA) = {BC_sl:.1f} kg/m^2")
    print(f"    Lower B_C = more affected by drag")

    # (d) Velocity loss per orbit
    Dv_iss = a_iss * T_orbit
    Dv_sl = a_sl * T_orbit

    print(f"\n(d) Velocity loss per orbit (T = {T_orbit/60:.0f} min):")
    print(f"    ISS: Dv = a * T = {a_iss:.3e} * {T_orbit} = {Dv_iss:.3f} m/s")
    print(f"    Starlink: Dv = a * T = {Dv_sl:.3f} m/s")
    print(f"    ISS must reboost regularly; Starlink uses ion thrusters")


def exercise_3():
    """
    Exercise 3: Orbital Lifetime Estimation

    CubeSat: mass 4 kg, CdA = 0.066 m^2, at 350 km.
    rho = 2e-11 kg/m^3, H = 50 km.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Orbital Lifetime Estimation")
    print("=" * 70)

    m = 4           # kg
    CdA = 0.066     # m^2
    h = 350e3       # m
    rho = 2e-11     # kg/m^3
    H = 50e3        # m (scale height)
    R_E = 6.371e6   # m
    r = R_E + h

    # (a) Ballistic coefficient
    BC = m / CdA

    print(f"\n    CubeSat: m = {m} kg, CdA = {CdA} m^2")
    print(f"    At 350 km: rho = {rho:.1e} kg/m^3, H = {H/1e3:.0f} km")

    print(f"\n(a) Ballistic coefficient:")
    print(f"    B_C = m / CdA = {m} / {CdA} = {BC:.1f} kg/m^2")

    # (b) Orbital decay rate
    # da/dt ~ -2*pi*a^2*rho/(BC) (approximate, a ~ r for circular orbit)
    # Or: dr/dt ~ -rho * v * r / BC (simplified)
    v = np.sqrt(3.986e14 / r)  # orbital velocity
    dr_dt = -rho * v * r / BC  # approximate m/s
    dr_dt_km_day = dr_dt * 86400 / 1e3

    print(f"\n(b) Orbital decay rate:")
    print(f"    v = sqrt(GM/r) = {v:.0f} m/s")
    print(f"    dr/dt ~ -rho * v * r / BC")
    print(f"    = -{rho:.1e} * {v:.0f} * {r:.3e} / {BC:.1f}")
    print(f"    = {dr_dt:.4f} m/s = {dr_dt_km_day:.2f} km/day")

    # (c) Lifetime approximation
    # tau ~ H * BC / (2*pi*r^2*rho)
    # or equivalently: tau ~ H / |dr/dt|
    tau = H * BC / (2 * np.pi * r**2 * rho)
    tau_days = tau / 86400

    # Alternative: using the simpler formula
    tau_alt = H / abs(dr_dt)
    tau_alt_days = tau_alt / 86400

    print(f"\n(c) Orbital lifetime estimation:")
    print(f"    tau ~ H * BC / (2*pi*r^2*rho)")
    print(f"    = {H:.0e} * {BC:.1f} / (2*pi*({r:.3e})^2*{rho:.1e})")
    print(f"    = {tau:.0f} s = {tau_days:.0f} days = {tau_days/365:.1f} years")

    print(f"\n    Alternative: tau ~ H / |dr/dt| = {H:.0e} / {abs(dr_dt):.4f}")
    print(f"    = {tau_alt:.0f} s = {tau_alt_days:.0f} days")

    # Time to descend from 350 to 250 km
    dh = 100e3  # m
    t_250 = dh / abs(dr_dt)
    t_250_days = t_250 / 86400

    print(f"\n    Time to descend from 350 to 250 km (rough estimate):")
    print(f"    dt ~ dh / |dr/dt| = {dh/1e3:.0f} km / {abs(dr_dt_km_day):.2f} km/day")
    print(f"    ~ {t_250_days:.0f} days")
    print(f"    Note: This is approximate. As altitude decreases, density increases")
    print(f"    exponentially, so the actual final descent accelerates rapidly.")


def exercise_4():
    """
    Exercise 4: Starlink Storm Loss Analysis

    210 km deployment: rho_0 = 1.5e-10 kg/m^3, v = 7.79 km/s.
    Starlink: mass 306 kg, CdA = 22 m^2.
    Storm: 50% density enhancement. Thruster: 0.08 N.
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Starlink Storm Loss Analysis")
    print("=" * 70)

    rho_0 = 1.5e-10   # kg/m^3
    v = 7.79e3         # m/s (at 210 km)
    m = 306            # kg
    CdA = 22           # m^2
    F_thrust = 0.08    # N
    T_orbit = 88.5 * 60  # ~88.5 min at 210 km

    # (a) Baseline drag force
    F_drag_base = 0.5 * rho_0 * v**2 * CdA

    print(f"\n    Starlink at 210 km: rho_0 = {rho_0:.1e} kg/m^3, v = {v/1e3:.2f} km/s")
    print(f"    Mass = {m} kg, CdA = {CdA} m^2, Thruster = {F_thrust} N")

    print(f"\n(a) Baseline drag force:")
    print(f"    F_drag = 0.5 * rho * v^2 * CdA")
    print(f"    = 0.5 * {rho_0:.1e} * ({v:.0f})^2 * {CdA}")
    print(f"    = {F_drag_base:.4f} N")

    # (b) Storm-enhanced drag (50% increase)
    rho_storm = 1.5 * rho_0
    F_drag_storm = 0.5 * rho_storm * v**2 * CdA

    print(f"\n(b) Storm-enhanced drag (+50%):")
    print(f"    rho_storm = 1.5 * rho_0 = {rho_storm:.2e} kg/m^3")
    print(f"    F_drag_storm = {F_drag_storm:.4f} N")

    # (c) Can thruster overcome drag?
    print(f"\n(c) Thruster vs drag comparison:")
    print(f"    Thruster force: {F_thrust:.2f} N")
    print(f"    Baseline drag: {F_drag_base:.4f} N")
    print(f"    Storm drag:    {F_drag_storm:.4f} N")
    margin_base = F_thrust - F_drag_base
    margin_storm = F_thrust - F_drag_storm
    print(f"    Baseline margin: {margin_base:.4f} N "
          f"({'sufficient' if margin_base > 0 else 'INSUFFICIENT'})")
    print(f"    Storm margin:    {margin_storm:.4f} N "
          f"({'sufficient' if margin_storm > 0 else 'INSUFFICIENT'})")

    if margin_storm > 0:
        print(f"    Thruster can barely overcome storm drag, but margin is thin")
        print(f"    Satellites not yet oriented for thrusting would lose the battle")
    else:
        print(f"    Thruster CANNOT overcome storm drag at 210 km!")

    # (d) Altitude loss per orbit without thrust
    a_drag = F_drag_storm / m
    Dv = a_drag * T_orbit
    # Altitude loss ~ 2 * a_orbit / v * Dv ~ (Dv / v) * a_orbit approximately
    R_E = 6.371e6
    r = R_E + 210e3
    # dh per orbit ~ rho * CdA * pi * a / (m) * period-related
    # Simple estimate: dh ~ v * T * a_drag / v = a_drag * T^2 * v / (2*r)
    # Or from orbital mechanics: da ~ -2*a*Dv/v
    da_orbit = 2 * r * Dv / v  # altitude drop per orbit (approximate)

    print(f"\n(d) Altitude loss per orbit during storm (no thrust):")
    print(f"    Drag deceleration = F/m = {a_drag:.4e} m/s^2")
    print(f"    Velocity loss per orbit: Dv = a * T = {Dv:.2f} m/s")
    print(f"    Altitude drop per orbit: da ~ 2*r*Dv/v = {da_orbit:.0f} m")
    print(f"    = {da_orbit/1e3:.1f} km per orbit")
    print(f"    At {da_orbit/1e3:.1f} km/orbit and ~16 orbits/day:")
    print(f"    ~{da_orbit/1e3 * 16:.0f} km altitude loss per day")
    print(f"    At 210 km, this leads to reentry within days")


def exercise_5():
    """
    Exercise 5: Joule Heating Power

    E = 50 mV/m, Sigma_P = 10 S.
    (a) Local Joule heating rate per unit area.
    (b) Total heating over auroral oval (A = 5e12 m^2).
    (c) Compare with solar EUV heating (~3e11 W).
    (d) Temperature increase rate.
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Joule Heating Power")
    print("=" * 70)

    E = 50e-3       # V/m
    Sigma_P = 10    # S
    A_oval = 5e12   # m^2
    P_EUV = 3e11    # W
    m_col = 1e-3    # kg/m^2 (thermospheric column mass)
    cp = 1000       # J/(kg*K)

    # (a) Local Joule heating rate
    q_J = Sigma_P * E**2

    print(f"\n    E = {E*1e3:.0f} mV/m, Sigma_P = {Sigma_P} S")

    print(f"\n(a) Local Joule heating rate per unit area:")
    print(f"    q_J = Sigma_P * E^2")
    print(f"    = {Sigma_P} * ({E:.3e})^2")
    print(f"    = {q_J:.3e} W/m^2 = {q_J*1e3:.1f} mW/m^2")

    # (b) Total heating
    P_total = q_J * A_oval
    P_total_GW = P_total * 1e-9

    print(f"\n(b) Total Joule heating power:")
    print(f"    P = q_J * A = {q_J:.3e} * {A_oval:.1e}")
    print(f"    = {P_total:.3e} W = {P_total_GW:.0f} GW")

    # (c) Compare with EUV
    ratio = P_total / P_EUV
    print(f"\n(c) Comparison with solar EUV heating:")
    print(f"    P_EUV = {P_EUV:.1e} W = {P_EUV*1e-9:.0f} GW")
    print(f"    P_Joule / P_EUV = {ratio:.1f}")
    if ratio > 1:
        print(f"    YES, Joule heating ({P_total_GW:.0f} GW) EXCEEDS EUV heating "
              f"({P_EUV*1e-9:.0f} GW)")
        print(f"    by a factor of {ratio:.1f}!")
    else:
        print(f"    Joule heating is {ratio:.1f}x of EUV heating")

    # (d) Temperature increase rate
    # dT/dt = q_J / (m_col * cp)
    dT_dt = q_J / (m_col * cp)  # K/s
    dT_dt_hr = dT_dt * 3600  # K/hour

    print(f"\n(d) Temperature increase rate:")
    print(f"    dT/dt = q_J / (m_col * cp)")
    print(f"    = {q_J:.3e} / ({m_col:.0e} * {cp})")
    print(f"    = {dT_dt:.3e} K/s = {dT_dt_hr:.0f} K/hour")
    print(f"    This rapid heating rate explains why thermospheric temperatures")
    print(f"    can increase by hundreds of K within a few hours during storms,")
    print(f"    leading to dramatic increases in atmospheric density and satellite drag.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
