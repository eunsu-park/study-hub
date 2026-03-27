"""
Exercise Solutions for Lesson 05: Geomagnetic Storms

Topics covered:
  - Ring current energy from Dst via DPS relation
  - Pressure-corrected Dst* (Burton correction)
  - Charge exchange lifetime of ring current protons
  - CME vs CIR storm energy comparison
  - Extreme event probability from power-law distribution
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Ring Current Energy

    Minimum Dst = -250 nT. Calculate ring current energy using DPS relation.
    Compare with solar wind KE impacting magnetosphere over 12 hours
    (n=10 cm^-3, v=500 km/s, cross-section diameter = 30 R_E).
    """
    print("=" * 70)
    print("Exercise 1: Ring Current Energy (DPS)")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7
    B0 = 3.1e-5     # T
    R_E = 6.371e6    # m
    m_p = 1.67e-27   # kg
    Dst = -250e-9    # T

    # DPS: E_rc = |Dst| * B0 * R_E^3 / (2 * mu0)
    E_rc = abs(Dst) * B0 * R_E**3 / (2 * mu0)

    print(f"\n    Dst = {Dst*1e9:.0f} nT")
    print(f"    E_rc = |Dst| * B0 * R_E^3 / (2*mu0)")
    print(f"    E_rc = {abs(Dst):.1e} * {B0:.1e} * ({R_E:.3e})^3 / (2*{mu0:.3e})")
    print(f"    E_rc = {E_rc:.3e} J")

    # Solar wind KE
    n = 10e6        # m^-3
    v = 500e3       # m/s
    d = 30 * R_E    # diameter
    r = d / 2       # radius
    A = np.pi * r**2
    rho = n * m_p
    dt = 12 * 3600  # 12 hours

    E_sw = 0.5 * rho * v**2 * v * A * dt  # energy = 0.5*rho*v^2 * (v*A*dt)

    print(f"\n    Solar wind KE over 12 hours:")
    print(f"    n = {n/1e6:.0f} cm^-3, v = {v/1e3:.0f} km/s")
    print(f"    Cross-section: diameter = 30 R_E, A = pi*({r/R_E:.0f} R_E)^2 "
          f"= {A:.3e} m^2")
    print(f"    E_sw = 0.5 * rho * v^3 * A * dt")
    print(f"    E_sw = {E_sw:.3e} J")

    print(f"\n    Comparison:")
    print(f"    E_rc / E_sw = {E_rc / E_sw:.4f}")
    print(f"    => Only ~{E_rc / E_sw * 100:.2f}% of solar wind KE is captured")
    print(f"    as ring current energy during this major storm")


def exercise_2():
    """
    Exercise 2: Pressure Correction Dst*

    Measured Dst = -180 nT, P_dyn = 25 nPa.
    Burton correction: Dst* = Dst - b*sqrt(P_dyn) + c
    b = 7.26 nT/sqrt(nPa), c = 11 nT.
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Pressure-Corrected Dst*")
    print("=" * 70)

    Dst = -180    # nT
    P_dyn = 25    # nPa
    b = 7.26      # nT/sqrt(nPa)
    c = 11        # nT

    # Dst* = Dst - b*sqrt(P_dyn) + c
    correction = b * np.sqrt(P_dyn)
    Dst_star = Dst - correction + c

    print(f"\n    Measured Dst = {Dst} nT")
    print(f"    P_dyn = {P_dyn} nPa")
    print(f"    b = {b} nT/sqrt(nPa), c = {c} nT")

    print(f"\n    Magnetopause compression contribution:")
    print(f"    b * sqrt(P_dyn) = {b} * sqrt({P_dyn}) = {b} * {np.sqrt(P_dyn):.2f}")
    print(f"    = {correction:.1f} nT (positive contribution from compression)")

    print(f"\n    Pressure-corrected Dst*:")
    print(f"    Dst* = Dst - b*sqrt(P_dyn) + c")
    print(f"    = {Dst} - {correction:.1f} + {c}")
    print(f"    = {Dst_star:.1f} nT")

    compression_frac = correction / abs(Dst)
    print(f"\n    How much of measured Dst is from compression:")
    print(f"    Compression effect = +{correction:.1f} nT out of |Dst| = {abs(Dst)} nT")
    print(f"    => {compression_frac*100:.1f}% of the measured depression is due to")
    print(f"       magnetopause compression, not the ring current")
    print(f"    The 'true' ring current contribution: Dst* = {Dst_star:.1f} nT")
    print(f"    This is stronger than the measured Dst after removing the")
    print(f"    positive compression contribution and quiet-time offset")


def exercise_3():
    """
    Exercise 3: Charge Exchange Lifetime

    50 keV proton at L=4, n_H = 100 cm^-3.
    sigma_cx = 2e-19 m^2.
    Ring current energy = 4e15 J.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Charge Exchange Lifetime")
    print("=" * 70)

    m_p = 1.67e-27    # kg
    eV = 1.602e-19    # J
    E_kin = 50e3 * eV  # 50 keV in Joules
    n_H = 100e6       # 100 cm^-3 in m^-3
    sigma_cx = 2e-19  # m^2
    E_rc = 4e15       # J

    # Proton speed from kinetic energy
    v = np.sqrt(2 * E_kin / m_p)

    # Charge exchange rate: R = n_H * sigma_cx * v
    R_cx = n_H * sigma_cx * v

    # Lifetime: tau = 1 / R
    tau = 1 / R_cx
    tau_hr = tau / 3600
    tau_days = tau / 86400

    print(f"\n    50 keV proton at L = 4:")
    print(f"    E = 50 keV = {E_kin:.3e} J")
    print(f"    v = sqrt(2E/m_p) = sqrt(2*{E_kin:.3e}/{m_p:.3e})")
    print(f"    v = {v:.3e} m/s = {v/1e3:.0f} km/s")

    print(f"\n    Geocoronal hydrogen: n_H = {n_H/1e6:.0f} cm^-3")
    print(f"    Charge exchange cross-section: sigma = {sigma_cx:.1e} m^2")

    print(f"\n    Charge exchange rate:")
    print(f"    R = n_H * sigma * v = {n_H:.1e} * {sigma_cx:.1e} * {v:.3e}")
    print(f"    R = {R_cx:.3e} s^-1")

    print(f"\n    Charge exchange lifetime:")
    print(f"    tau = 1/R = {tau:.3e} s = {tau_hr:.1f} hours = {tau_days:.1f} days")

    # Ring current 1/e decay
    print(f"\n    Ring current energy decay:")
    print(f"    E_rc = {E_rc:.1e} J")
    print(f"    Time to 1/e: tau_CX = {tau_hr:.1f} hours = {tau_days:.1f} days")
    print(f"    This is consistent with the 'fast' decay component of ~5-10 hours")
    print(f"    Note: Other loss mechanisms (wave-particle scattering,")
    print(f"    Coulomb drag) also contribute, making the actual decay faster.")


def exercise_4():
    """
    Exercise 4: CME vs CIR Storm Comparison

    CME storm: 8 hr main phase, avg dDst/dt = -30 nT/hr, min Dst = -240 nT.
    CIR storm: 24 hr main phase, avg dDst/dt = -4 nT/hr, min Dst = -96 nT.
    Injection rate ~ |dDst/dt|.
    """
    print("\n" + "=" * 70)
    print("Exercise 4: CME vs CIR Storm Comparison")
    print("=" * 70)

    # CME storm
    dt_cme = 8       # hours
    dDst_dt_cme = 30  # |dDst/dt| in nT/hr
    Dst_min_cme = -240  # nT

    # CIR storm
    dt_cir = 24      # hours
    dDst_dt_cir = 4   # |dDst/dt| in nT/hr
    Dst_min_cir = -96   # nT

    # Total "injection" proportional to integral of |dDst/dt| * dt
    # If injection rate Q ~ |dDst/dt|, then total injection ~ |dDst/dt| * dt
    # But we also need to account for decay: dDst*/dt = Q - Dst*/tau
    # Total injection energy ~ integral(Q dt) = integral(dDst*/dt + Dst*/tau) dt
    # For simplicity, use the proxy: total injection ~ |dDst/dt| * duration

    inject_cme = dDst_dt_cme * dt_cme  # nT (total Dst change without decay)
    inject_cir = dDst_dt_cir * dt_cir

    print(f"\n    CME-driven storm:")
    print(f"    Main phase: {dt_cme} hours, avg |dDst/dt| = {dDst_dt_cme} nT/hr")
    print(f"    Min Dst = {Dst_min_cme} nT")
    print(f"    Total injection proxy = |dDst/dt| * dt = {inject_cme} nT")

    print(f"\n    CIR-driven storm:")
    print(f"    Main phase: {dt_cir} hours, avg |dDst/dt| = {dDst_dt_cir} nT/hr")
    print(f"    Min Dst = {Dst_min_cir} nT")
    print(f"    Total injection proxy = |dDst/dt| * dt = {inject_cir} nT")

    # Better estimate accounting for decay (using Burton equation)
    # Total injected = integral(Q) dt = Delta_Dst + integral(Dst/tau) dt
    # Approximate: if Dst linearly decreases from 0 to Dst_min:
    # integral(Dst/tau) ~ |Dst_min| * dt / (2 * tau)
    tau = 8  # hours (typical)

    decay_cme = abs(Dst_min_cme) * dt_cme / (2 * tau)
    total_cme = abs(Dst_min_cme) + decay_cme

    decay_cir = abs(Dst_min_cir) * dt_cir / (2 * tau)
    total_cir = abs(Dst_min_cir) + decay_cir

    print(f"\n    Better estimate (accounting for decay, tau = {tau} hr):")
    print(f"    Total injected ~ |Dst_min| + |Dst_min|*dt/(2*tau)")
    print(f"    CME: {abs(Dst_min_cme)} + {decay_cme:.0f} = {total_cme:.0f} nT-equivalent")
    print(f"    CIR: {abs(Dst_min_cir)} + {decay_cir:.0f} = {total_cir:.0f} nT-equivalent")

    print(f"\n    The CIR storm injects {total_cir:.0f}/{total_cme:.0f} = "
          f"{total_cir/total_cme:.2f} times as much energy as the CME storm")
    print(f"    Despite having a much weaker minimum Dst!")
    print(f"\n    Key insight: CIR storms inject energy more slowly but over much")
    print(f"    longer periods. The weaker Dst reflects the balance between slow")
    print(f"    injection and continuous decay, not necessarily less total energy.")


def exercise_5():
    """
    Exercise 5: Extreme Event Probability (Power Law)

    P(Dst < -x) = C * x^(-4.5), calibrated: P(Dst < -200) = 0.1 per year.
    (a) P(Dst < -600) per year.
    (b) Return period for Dst < -600 events.
    (c) P(at least one in 25 years).
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Extreme Event Probability (Power Law)")
    print("=" * 70)

    alpha = 4.5
    # Calibration: P(Dst < -200) = C * 200^(-4.5) = 0.1
    x_cal = 200
    P_cal = 0.1
    C = P_cal / x_cal**(-alpha)
    # C = 0.1 * 200^4.5

    print(f"\n    Power-law: P(Dst < -x) = C * x^(-{alpha})")
    print(f"    Calibration: P(Dst < -{x_cal}) = {P_cal} per year")
    print(f"    C = {P_cal} / {x_cal}^(-{alpha}) = {P_cal} * {x_cal}^{alpha}")
    print(f"    C = {C:.3e}")

    # (a) P(Dst < -600)
    x_target = 600
    P_600 = C * x_target**(-alpha)

    print(f"\n(a) P(Dst < -{x_target}) per year:")
    print(f"    P = C * {x_target}^(-{alpha})")
    print(f"    P = {C:.3e} * {x_target}^(-{alpha})")
    print(f"    P = {P_600:.6e} per year")

    # Alternative: use ratio directly
    ratio = (x_cal / x_target)**alpha
    P_600_alt = P_cal * ratio
    print(f"    Alternatively: P = {P_cal} * ({x_cal}/{x_target})^{alpha}")
    print(f"    = {P_cal} * {ratio:.6e} = {P_600_alt:.6e}")

    # (b) Return period
    T_return = 1 / P_600
    print(f"\n(b) Return period:")
    print(f"    T = 1 / P = 1 / {P_600:.6e}")
    print(f"    T = {T_return:.0f} years")

    # (c) P(at least one in 25 years)
    t_window = 25
    P_zero = (1 - P_600)**t_window
    P_at_least_one = 1 - P_zero
    # Also using Poisson approximation
    mu = P_600 * t_window
    P_poisson = 1 - np.exp(-mu)

    print(f"\n(c) P(at least one in {t_window} years):")
    print(f"    Exact: P = 1 - (1-p)^{t_window} = 1 - {P_zero:.8f}")
    print(f"    = {P_at_least_one:.6e}")
    print(f"    Poisson approx: P = 1 - exp(-mu) where mu = {mu:.6e}")
    print(f"    = {P_poisson:.6e}")
    print(f"    => About {P_at_least_one*100:.4f}% chance in {t_window} years")

    print(f"\n    Note: The power-law exponent ({alpha}) is critical.")
    print(f"    With alpha={alpha}, extreme events are extremely rare.")
    print(f"    Some studies suggest shallower power laws (alpha ~ 2-3),")
    print(f"    which would make extreme events much more probable.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
