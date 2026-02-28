"""
Exercise Solutions for Lesson 08: Ionosphere

Topics covered:
  - Chapman layer peak density and altitude
  - GPS ionospheric range error with mapping function
  - Pedersen and Hall conductivity calculation
  - Scintillation frequency scaling
  - TEC, plasma frequency, and maximum usable frequency (MUF)
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Chapman Layer Calculation

    E-region: n_m = 2e5 cm^-3 at h_m = 110 km, H = 10 km (at chi = 0).
    (a) Peak density at chi = 60 deg (alpha-Chapman).
    (b) Altitude of peak production at chi = 60 deg.
    (c) Time for density to halve after sunset (alpha = 3e-7 cm^3/s).
    """
    print("=" * 70)
    print("Exercise 1: Chapman Layer Calculation")
    print("=" * 70)

    n_m0 = 2e5       # cm^-3 (at chi = 0)
    h_m = 110        # km
    H = 10           # km (scale height)
    chi = 60         # degrees
    alpha_rec = 3e-7  # cm^3/s (recombination coefficient)

    # (a) Alpha-Chapman layer: n_m(chi) = n_m(0) * (cos(chi))^(1/2)
    # For alpha-Chapman (quadratic recombination), the peak density scales as:
    # n_m(chi) = n_m(0) * (cos(chi))^(1/2)
    cos_chi = np.cos(np.radians(chi))
    n_m_chi = n_m0 * cos_chi**(0.5)

    print(f"\n    E-region parameters: n_m = {n_m0:.0e} cm^-3 at h_m = {h_m} km")
    print(f"    Scale height H = {H} km, chi = {chi} deg")

    print(f"\n(a) Peak density at chi = {chi} deg (alpha-Chapman):")
    print(f"    n_m(chi) = n_m(0) * cos(chi)^(1/2)")
    print(f"    = {n_m0:.0e} * cos({chi} deg)^(1/2)")
    print(f"    = {n_m0:.0e} * {cos_chi:.4f}^(1/2)")
    print(f"    = {n_m0:.0e} * {cos_chi**0.5:.4f}")
    print(f"    = {n_m_chi:.3e} cm^-3")

    # (b) Peak production altitude shifts upward
    # z_m(chi) = h_m + H * ln(sec(chi)) = h_m + H * ln(1/cos(chi))
    dh = H * np.log(1 / cos_chi)
    h_m_chi = h_m + dh

    print(f"\n(b) Peak production altitude at chi = {chi} deg:")
    print(f"    h_m(chi) = h_m + H * ln(sec(chi))")
    print(f"    = {h_m} + {H} * ln(1/{cos_chi:.4f})")
    print(f"    = {h_m} + {H} * {np.log(1/cos_chi):.4f}")
    print(f"    = {h_m} + {dh:.1f} = {h_m_chi:.1f} km")
    print(f"    The peak moves up by {dh:.1f} km at higher zenith angles")

    # (c) After sunset: dn/dt = -alpha * n^2 (quadratic recombination)
    # Solution: 1/n(t) = 1/n_0 + alpha*t
    # At n = n_0/2: 1/(n_0/2) = 1/n_0 + alpha*t
    # 2/n_0 = 1/n_0 + alpha*t
    # t = 1/(alpha * n_0)
    n0 = n_m_chi  # use the chi=60 density at sunset
    t_half = 1 / (alpha_rec * n0)  # seconds
    # convert n0 to m^-3 for display, but alpha is in cm^3/s so keep n in cm^-3

    print(f"\n(c) Density halving time after sunset:")
    print(f"    After sunset, q = 0: dn/dt = -alpha * n^2")
    print(f"    Solution: 1/n(t) = 1/n_0 + alpha*t")
    print(f"    For n = n_0/2: t = 1/(alpha * n_0)")
    print(f"    n_0 = {n0:.3e} cm^-3 (at sunset, chi=60 deg)")
    print(f"    t = 1/({alpha_rec:.1e} * {n0:.3e})")
    print(f"    t = {t_half:.1f} s = {t_half/60:.1f} min")
    print(f"    The E-region decays rapidly after sunset because alpha is large,")
    print(f"    which is why the E-region largely disappears at night.")


def exercise_2():
    """
    Exercise 2: Ionospheric Range Error

    L1 GPS (1575.42 MHz), elevation 30 deg, VTEC = 80 TECU.
    Ionosphere shell at 350 km.
    (a) Slant TEC using thin-shell mapping.
    (b) Range error in meters.
    (c) L1/L2 pseudorange difference (L2 = 1227.60 MHz).
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Ionospheric Range Error")
    print("=" * 70)

    f_L1 = 1575.42e6  # Hz
    f_L2 = 1227.60e6  # Hz
    elev = 30          # degrees
    VTEC = 80          # TECU (1 TECU = 1e16 el/m^2)
    VTEC_m2 = VTEC * 1e16  # electrons/m^2
    R_E = 6371e3       # m
    h_iono = 350e3     # m

    # (a) Slant TEC with mapping function
    # sin(chi_i) = R_E * cos(elev) / (R_E + h_iono)
    sin_chi = R_E * np.cos(np.radians(elev)) / (R_E + h_iono)
    chi_i = np.degrees(np.arcsin(sin_chi))
    cos_chi = np.cos(np.radians(chi_i))
    STEC = VTEC / cos_chi
    STEC_m2 = STEC * 1e16

    print(f"\n    GPS L1 = {f_L1/1e6:.2f} MHz, elevation = {elev} deg")
    print(f"    VTEC = {VTEC} TECU, ionosphere shell at {h_iono/1e3:.0f} km")

    print(f"\n(a) Slant TEC:")
    print(f"    sin(chi_i) = R_E*cos(elev) / (R_E + h_iono)")
    print(f"    = {R_E:.0f}*cos({elev} deg) / ({R_E:.0f} + {h_iono:.0f})")
    print(f"    = {sin_chi:.4f}")
    print(f"    chi_i = {chi_i:.1f} deg")
    print(f"    Obliquity factor = 1/cos(chi_i) = {1/cos_chi:.3f}")
    print(f"    STEC = VTEC / cos(chi_i) = {VTEC} / {cos_chi:.4f}")
    print(f"    STEC = {STEC:.1f} TECU")

    # (b) Range error
    # Delta_rho = 40.3 * TEC / f^2 (TEC in el/m^2, f in Hz, rho in m)
    Delta_rho = 40.3 * STEC_m2 / f_L1**2

    print(f"\n(b) Ionospheric range error on L1:")
    print(f"    Delta_rho = 40.3 * STEC / f^2")
    print(f"    = 40.3 * {STEC_m2:.3e} / ({f_L1:.5e})^2")
    print(f"    = {Delta_rho:.2f} m")
    print(f"    This is a significant error for single-frequency GPS users!")

    # (c) L1/L2 pseudorange difference
    Delta_L1 = 40.3 * STEC_m2 / f_L1**2
    Delta_L2 = 40.3 * STEC_m2 / f_L2**2
    diff = Delta_L2 - Delta_L1

    print(f"\n(c) Dual-frequency pseudorange difference:")
    print(f"    Range error on L1: {Delta_L1:.2f} m")
    print(f"    Range error on L2: {Delta_L2:.2f} m")
    print(f"    Difference (P1 - P2): {diff:.2f} m")
    print(f"    This difference allows dual-frequency receivers to measure TEC")
    print(f"    and remove the ionospheric delay to first order.")


def exercise_3():
    """
    Exercise 3: Pedersen and Hall Conductivity

    At 120 km: nu_in = 500 s^-1, Omega_i = 200 rad/s, n_e = 1e5 cm^-3,
    B = 5e-5 T.
    (a) Pedersen conductivity.
    (b) Hall conductivity.
    (c) Ratio sigma_H / sigma_P.
    (d) Joule heating rate for E_perp = 50 mV/m.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Pedersen and Hall Conductivity")
    print("=" * 70)

    e = 1.602e-19     # C
    m_e = 9.109e-31   # kg
    m_i = 1.67e-27 * 30.5  # approximate effective ion mass (NO+/O2+)
    # Actually we don't need masses directly since we're given frequencies

    nu_in = 500       # s^-1 (ion-neutral collision frequency)
    Omega_i = 200     # rad/s (ion cyclotron frequency)
    n_e = 1e5 * 1e6   # 1e5 cm^-3 = 1e11 m^-3
    B = 5e-5          # T

    # For electrons at 120 km, typically nu_en >> Omega_e (electron magnetized)
    # Omega_e = eB/m_e
    Omega_e = e * B / m_e
    # Electron-neutral collision frequency at 120 km: ~1e4 s^-1
    nu_en = 1e4  # approximate; not given, but we can derive conductivities

    # Pedersen conductivity: sigma_P = (n_e * e / B) * (kappa_i/(1+kappa_i^2)
    #                                                    + kappa_e/(1+kappa_e^2))
    # where kappa_i = Omega_i / nu_in, kappa_e = Omega_e / nu_en
    # BUT since the problem gives specific values, let's use the standard formulas
    # directly with the given parameters.

    # Standard formulas:
    # sigma_P = (n_e * e / B) * [nu_in*Omega_i/(nu_in^2 + Omega_i^2)
    #           + nu_en*Omega_e/(nu_en^2 + Omega_e^2)]
    # sigma_H = (n_e * e / B) * [Omega_i^2/(nu_in^2 + Omega_i^2)
    #           - Omega_e^2/(nu_en^2 + Omega_e^2)]

    # At 120 km, the electron term is usually small compared to ion term
    # for Pedersen, and the electron Hall term dominates.
    # For simplicity, use the dominant terms:

    # Ion Pedersen conductivity (dominant at E-region):
    kappa_i = Omega_i / nu_in
    ion_P = kappa_i / (1 + kappa_i**2)
    sigma_P_ion = n_e * e * ion_P / B

    # Electron contribution to Pedersen (usually smaller):
    kappa_e = Omega_e / nu_en
    elec_P = kappa_e / (1 + kappa_e**2)
    sigma_P_elec = n_e * e * elec_P / B

    sigma_P = sigma_P_ion + sigma_P_elec

    print(f"\n    Parameters at 120 km:")
    print(f"    nu_in = {nu_in} s^-1, Omega_i = {Omega_i} rad/s")
    print(f"    n_e = {n_e/1e6:.0e} cm^-3 = {n_e:.1e} m^-3, B = {B:.1e} T")
    print(f"    Omega_e = eB/m_e = {Omega_e:.3e} rad/s")
    print(f"    kappa_i = Omega_i/nu_in = {kappa_i:.2f}")
    print(f"    kappa_e = Omega_e/nu_en = {kappa_e:.1f} (using nu_en ~ {nu_en:.0e})")

    print(f"\n(a) Pedersen conductivity:")
    print(f"    sigma_P = (n_e*e/B) * [kappa_i/(1+kappa_i^2) + kappa_e/(1+kappa_e^2)]")
    print(f"    Ion term:  kappa_i/(1+kappa_i^2) = {ion_P:.4f}")
    print(f"    Elec term: kappa_e/(1+kappa_e^2) = {elec_P:.6f}")
    print(f"    n_e*e/B = {n_e*e/B:.3e} S/m")
    print(f"    sigma_P = {sigma_P:.3e} S/m")

    # (b) Hall conductivity
    ion_H = kappa_i**2 / (1 + kappa_i**2)
    elec_H = kappa_e**2 / (1 + kappa_e**2)
    sigma_H = n_e * e * (elec_H - ion_H) / B
    # Actually the signs: sigma_H = (n_e*e/B) * [Omega_e^2/(nu_en^2+Omega_e^2)
    #                                             - Omega_i^2/(nu_in^2+Omega_i^2)]
    # Since electrons are magnetized, the electron term dominates

    print(f"\n(b) Hall conductivity:")
    print(f"    sigma_H = (n_e*e/B) * [kappa_e^2/(1+kappa_e^2) - kappa_i^2/(1+kappa_i^2)]")
    print(f"    Elec term: kappa_e^2/(1+kappa_e^2) = {elec_H:.6f}")
    print(f"    Ion term:  kappa_i^2/(1+kappa_i^2) = {ion_H:.4f}")
    print(f"    sigma_H = {sigma_H:.3e} S/m")

    # (c) Ratio
    ratio = sigma_H / sigma_P
    print(f"\n(c) sigma_H / sigma_P = {ratio:.2f}")
    print(f"    At 120 km, Hall conductivity exceeds Pedersen conductivity")
    print(f"    This is typical of the E-region where ions are unmagnetized")
    print(f"    (kappa_i < 1) but electrons are magnetized (kappa_e >> 1)")

    # (d) Joule heating
    E_perp = 50e-3  # V/m
    Q_J = sigma_P * E_perp**2

    print(f"\n(d) Joule heating rate for E_perp = {E_perp*1e3:.0f} mV/m:")
    print(f"    Q_J = sigma_P * E_perp^2")
    print(f"    = {sigma_P:.3e} * ({E_perp:.3e})^2")
    print(f"    = {Q_J:.3e} W/m^3")
    print(f"    For comparison, solar EUV heating rate: ~{1e-7:.0e} W/m^3")
    print(f"    Joule heating is {Q_J/1e-7:.0f}x stronger during storm conditions!")


def exercise_4():
    """
    Exercise 4: Scintillation Frequency Scaling

    S4 = 0.8 on L1 (1575.42 MHz). Weak scintillation: S4 ~ f^(-1.5).
    (a) Predict S4 on L2 and L5.
    (b) Which frequencies risk loss of lock (S4 > 1)?
    (c) Predict S4 at 250 MHz (VHF communications).
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Scintillation Frequency Scaling")
    print("=" * 70)

    S4_L1 = 0.8
    f_L1 = 1575.42e6  # Hz
    f_L2 = 1227.60e6
    f_L5 = 1176.45e6
    f_VHF = 250e6

    # S4 ~ f^(-1.5) => S4_2 = S4_1 * (f_1/f_2)^1.5

    print(f"\n    S4 on L1 ({f_L1/1e6:.2f} MHz) = {S4_L1}")
    print(f"    Scaling: S4 ~ f^(-1.5)")

    freqs = [
        ("L1", f_L1),
        ("L2", f_L2),
        ("L5", f_L5),
        ("VHF (250 MHz)", f_VHF),
    ]

    print(f"\n(a)-(c) S4 predictions at various frequencies:")
    print(f"    {'Frequency':>20} {'f (MHz)':>10} {'S4':>10} {'Risk':>15}")
    print(f"    {'-'*57}")

    for name, f in freqs:
        S4 = S4_L1 * (f_L1 / f)**1.5
        risk = "LOSS OF LOCK" if S4 > 1.0 else "OK"
        S4_display = f"{S4:.2f}" if S4 < 10 else f"{S4:.1f}"
        # Note saturation
        if S4 > 1.0 and f == f_VHF:
            S4_display += " (saturated)"
            risk = "SEVERE"
        print(f"    {name:>20} {f/1e6:>10.2f} {S4_display:>10} {risk:>15}")

    S4_L2 = S4_L1 * (f_L1 / f_L2)**1.5
    S4_L5 = S4_L1 * (f_L1 / f_L5)**1.5
    S4_VHF = S4_L1 * (f_L1 / f_VHF)**1.5

    print(f"\n(b) Loss of lock analysis (threshold S4 > 1.0):")
    print(f"    L2 ({f_L2/1e6:.2f} MHz): S4 = {S4_L2:.2f} "
          f"{'- AT RISK' if S4_L2 > 1.0 else '- safe'}")
    print(f"    L5 ({f_L5/1e6:.2f} MHz): S4 = {S4_L5:.2f} "
          f"{'- AT RISK' if S4_L5 > 1.0 else '- safe'}")

    print(f"\n(c) VHF at 250 MHz:")
    print(f"    Predicted S4 = {S4_VHF:.1f}")
    print(f"    NOTE: S4 > 1 means the weak scintillation approximation breaks down.")
    print(f"    In the strong scintillation regime, S4 saturates near ~1-1.5.")
    print(f"    At 250 MHz during these conditions, expect:")
    print(f"    - Saturated scintillation (S4 ~ 1.0-1.5)")
    print(f"    - Deep signal fades (>20 dB)")
    print(f"    - VHF communication systems would be severely disrupted")


def exercise_5():
    """
    Exercise 5: TEC and Plasma Frequency / MUF

    F2-layer: n_m = 5e5 cm^-3 at h_m = 300 km.
    (a) Peak plasma frequency f_0F2.
    (b) Critical frequency (max vertically reflected).
    (c) MUF for phi = 75 deg incidence.
    (d) Slab thickness and vertical TEC (Chapman, H = 50 km).
    """
    print("\n" + "=" * 70)
    print("Exercise 5: TEC and Plasma Frequency / MUF")
    print("=" * 70)

    n_m_cm3 = 5e5       # cm^-3
    n_m = n_m_cm3 * 1e6  # m^-3
    h_m = 300            # km
    H = 50               # km (scale height)
    phi = 75             # degrees (incidence angle)

    # (a) Plasma frequency: f_p = 9*sqrt(n_e) (n_e in m^-3, f in Hz)
    f_0F2 = 9 * np.sqrt(n_m)  # Hz

    print(f"\n    F2-layer: n_m = {n_m_cm3:.0e} cm^-3 = {n_m:.1e} m^-3")
    print(f"    h_m = {h_m} km, H = {H} km")

    print(f"\n(a) Peak plasma frequency:")
    print(f"    f_0F2 = 9 * sqrt(n_e)  (with n_e in m^-3)")
    print(f"    = 9 * sqrt({n_m:.1e})")
    print(f"    = 9 * {np.sqrt(n_m):.3e}")
    print(f"    = {f_0F2:.3e} Hz = {f_0F2/1e6:.2f} MHz")

    # (b) Critical frequency = plasma frequency at peak = f_0F2
    print(f"\n(b) Critical frequency (max reflected at vertical incidence):")
    print(f"    f_c = f_0F2 = {f_0F2/1e6:.2f} MHz")
    print(f"    Frequencies above {f_0F2/1e6:.2f} MHz pass through the ionosphere")
    print(f"    at vertical incidence (including GPS at 1575 MHz)")

    # (c) MUF using secant law
    f_MUF = f_0F2 * 1 / np.cos(np.radians(phi))

    print(f"\n(c) Maximum Usable Frequency (MUF) at phi = {phi} deg:")
    print(f"    f_MUF = f_0F2 * sec(phi)")
    print(f"    = {f_0F2/1e6:.2f} MHz * sec({phi} deg)")
    print(f"    = {f_0F2/1e6:.2f} * {1/np.cos(np.radians(phi)):.3f}")
    print(f"    = {f_MUF/1e6:.1f} MHz")
    print(f"    HF radio at oblique incidence can reflect up to {f_MUF/1e6:.0f} MHz")

    # (d) Slab thickness and TEC
    # For Chapman profile: TEC = n_m * tau_s where tau_s (slab thickness)
    # tau_s = H * sqrt(2*pi*e) ~ H * 4.13 for a Chapman layer
    # More precisely: tau_s = integral n(z) dz / n_m
    # For Chapman: tau_s = H * sqrt(2*pi) * exp(0.5) ~ 4.13 * H
    tau_s = H * np.sqrt(2 * np.pi) * np.exp(0.5)  # km (slab thickness)

    # Convert to meters for TEC
    tau_s_m = tau_s * 1e3  # m
    TEC = n_m * tau_s_m  # electrons/m^2
    TEC_TECU = TEC / 1e16

    print(f"\n(d) Slab thickness and vertical TEC:")
    print(f"    Chapman slab thickness: tau_s = H * sqrt(2*pi) * exp(0.5)")
    print(f"    = {H} km * {np.sqrt(2*np.pi)*np.exp(0.5):.3f}")
    print(f"    = {tau_s:.1f} km")
    print(f"\n    Vertical TEC = n_m * tau_s")
    print(f"    = {n_m:.1e} m^-3 * {tau_s_m:.3e} m")
    print(f"    = {TEC:.3e} el/m^2")
    print(f"    = {TEC_TECU:.1f} TECU")
    print(f"\n    This is a moderate TEC value, typical of mid-latitude daytime")
    print(f"    during moderate solar activity.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
