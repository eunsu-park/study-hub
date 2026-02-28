"""
Exercise Solutions for Lesson 10: Solar Energetic Particle Events

Topics covered:
  - Diffusive shock acceleration (DSA) spectral index
  - Velocity dispersion onset analysis (path length, release time)
  - Radiation dose from SEP proton spectrum
  - Aviation dose estimation (polar route, rerouting)
  - Proton shielding effectiveness (aluminum range-energy)
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: DSA Spectral Index

    CME shock: upstream wind 400 km/s, shock speed 1600 km/s (solar frame).
    (a) Upstream/downstream speeds in shock frame.
    (b) Compression ratio for strong (r=4) and moderate (r=3) shock.
    (c) Spectral index gamma of f(p) ~ p^(-gamma).
    (d) Convert to differential intensity index: j(E) ~ E^(-alpha).
    """
    print("=" * 70)
    print("Exercise 1: DSA Spectral Index")
    print("=" * 70)

    v_sw = 400     # km/s (upstream solar wind in solar frame)
    v_shock = 1600  # km/s (shock speed in solar frame)

    # (a) In shock frame:
    # u1 = shock speed - upstream wind = relative speed of upstream plasma
    u1 = v_shock - v_sw

    print(f"\n    Solar wind: {v_sw} km/s, Shock speed: {v_shock} km/s")

    print(f"\n(a) In the shock rest frame:")
    print(f"    u1 (upstream) = v_shock - v_sw = {v_shock} - {v_sw} = {u1} km/s")
    print(f"    u2 (downstream) = u1 / r (depends on compression ratio)")

    # (b) & (c) For two compression ratios
    for r in [4, 3]:
        u2 = u1 / r

        # DSA spectral index for momentum: gamma = 3r/(r-1)
        gamma = 3 * r / (r - 1)

        print(f"\n(b,c) Compression ratio r = {r}:")
        print(f"    u2 = u1/r = {u1}/{r} = {u2:.0f} km/s")
        print(f"    Spectral index: gamma = 3r/(r-1) = 3*{r}/({r}-1) = {gamma:.2f}")
        print(f"    f(p) ~ p^(-{gamma:.2f})")

        # (d) Non-relativistic: E = p^2/(2m), so dE ~ p dp
        # j(E) ~ p^2 f(p) dp/dE ~ p^2 * p^(-gamma) * (1/p) = p^(2-gamma+1-1) = p^(2-gamma)
        # Since E ~ p^2: p ~ E^(1/2), so j(E) ~ E^((2-gamma)/2)
        # => alpha = (gamma-2)/2
        # Or from the problem: alpha = (gamma-1)/2 for non-relativistic
        # Let me verify: j(E) dE = 4*pi*p^2*f(p)*dp, f(p) ~ p^(-gamma)
        # j(E) = 4*pi*p^2*f(p)*dp/dE; dp/dE = m/p for NR
        # j(E) ~ p^2 * p^(-gamma) * p^(-1) = p^(1-gamma) = E^((1-gamma)/2)
        # alpha = (gamma-1)/2
        alpha = (gamma - 1) / 2

        print(f"\n(d) Differential intensity spectral index (non-relativistic):")
        print(f"    j(E) ~ E^(-alpha) where alpha = (gamma-1)/2")
        print(f"    alpha = ({gamma:.2f}-1)/2 = {alpha:.2f}")
        print(f"    j(E) ~ E^(-{alpha:.2f})")

    print(f"\n    Summary:")
    print(f"    Strong shock (r=4): gamma=4.00, alpha=1.50 (harder spectrum)")
    print(f"    Moderate shock (r=3): gamma=4.50, alpha=1.75 (softer spectrum)")
    print(f"    Stronger shocks produce harder (flatter) energy spectra")


def exercise_2():
    """
    Exercise 2: Velocity Dispersion

    Onset times at 1 AU for different proton energies.
    (a) Convert v/c to km/s.
    (b) Plot/tabulate t_onset vs 1/v; find path length and release time.
    (c) Compare path length to Parker spiral (~1.15 AU).
    (d) Compare release time with flare peak at 11:05 UT.
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Velocity Dispersion Analysis")
    print("=" * 70)

    c = 3e5  # km/s

    # Data
    energies = [100, 50, 20, 10]  # MeV
    v_over_c = [0.428, 0.314, 0.203, 0.145]
    onset_UT = ["11:30", "11:42", "12:06", "12:32"]

    # Convert onset to minutes from 11:00
    onset_min = []
    for t in onset_UT:
        h, m = map(int, t.split(':'))
        onset_min.append((h - 11) * 60 + m)

    # (a) Convert to km/s
    v_kms = [beta * c for beta in v_over_c]

    print(f"\n(a) Velocity conversion:")
    print(f"    {'E (MeV)':>10} {'v/c':>8} {'v (km/s)':>12} {'Onset (UT)':>12} "
          f"{'t (min from 11:00)':>20}")
    print(f"    {'-'*64}")
    for i in range(len(energies)):
        print(f"    {energies[i]:>10} {v_over_c[i]:>8.3f} {v_kms[i]:>12.0f} "
              f"{onset_UT[i]:>12} {onset_min[i]:>20}")

    # (b) Linear regression: t = d/v + t_release
    # t = (d * 1/v) + t0
    inv_v = [1 / v for v in v_kms]

    # Simple linear regression
    x = np.array(inv_v)
    y = np.array(onset_min)

    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    intercept = (np.sum(y) - slope * np.sum(x)) / n

    # slope = d in km, since t is in minutes and 1/v is in s/km -> need consistent units
    # t [min] = d [km] * (1/v) [s/km] / 60 + t0 [min]
    # So slope = d / 60 (d in km)
    d_km = slope * 60  # km
    d_AU = d_km / 1.496e8

    t0_min = intercept
    t0_UT_h = 11 + int(t0_min // 60)
    t0_UT_m = t0_min % 60

    print(f"\n(b) Linear regression: t_onset = (d/v)/60 + t_release")
    print(f"    {'1/v (s/km)':>15} {'t (min)':>10}")
    print(f"    {'-'*27}")
    for i in range(len(x)):
        print(f"    {x[i]:>15.6e} {y[i]:>10.0f}")

    print(f"\n    Slope = {slope:.4e} min * km/s = d/60")
    print(f"    Intercept = {intercept:.2f} min")
    print(f"    Path length: d = slope * 60 = {d_km:.3e} km = {d_AU:.2f} AU")
    print(f"    Release time: t_0 = {t0_min:.1f} min after 11:00")
    print(f"    = {t0_UT_h:02d}:{t0_UT_m:04.1f} UT")

    # (c) Comparison with Parker spiral
    d_parker = 1.15  # AU
    print(f"\n(c) Path length comparison:")
    print(f"    Inferred: {d_AU:.2f} AU")
    print(f"    Parker spiral: ~{d_parker} AU")
    if d_AU > d_parker * 1.2:
        print(f"    Path length is LONGER than Parker spiral by "
              f"{(d_AU/d_parker - 1)*100:.0f}%")
        print(f"    This implies significant scattering during transport,")
        print(f"    or a non-nominal connection to the acceleration site")
    else:
        print(f"    Consistent with near-scatter-free transport along Parker spiral")

    # (d) Comparison with flare
    flare_UT = "11:05"
    flare_min = 5  # 5 min after 11:00
    delay = t0_min - flare_min

    print(f"\n(d) Release time vs flare peak:")
    print(f"    Flare peak: {flare_UT} UT")
    print(f"    Particle release: ~{t0_UT_h:02d}:{t0_UT_m:04.1f} UT")
    print(f"    Delay: ~{delay:.0f} minutes")
    print(f"    This delay suggests the particles were NOT accelerated")
    print(f"    directly by the flare but by the CME-driven shock,")
    print(f"    which takes ~{delay:.0f} min to develop and connect to Earth.")


def exercise_3():
    """
    Exercise 3: Radiation Dose from SEP Spectrum

    Phi(E) = 1e9 * E^(-2) protons/(cm^2 MeV) for 10 < E < 1000 MeV.
    (a) Total fluence.
    (b) Absorbed dose.
    (c) Dose equivalent with w_R = 2.
    (d) Compare to NASA 30-day limit.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Radiation Dose from SEP Spectrum")
    print("=" * 70)

    A = 1e9       # normalization (protons/(cm^2 MeV))
    alpha = 2     # spectral index
    E_min = 10    # MeV
    E_max = 1000  # MeV

    # (a) Total fluence: F = integral(Phi(E) dE) = integral(A * E^(-2) dE)
    # = A * [-E^(-1)] from E_min to E_max = A * (1/E_min - 1/E_max)
    F = A * (1/E_min - 1/E_max)

    print(f"\n    Spectrum: Phi(E) = {A:.0e} * E^(-{alpha}) protons/(cm^2 MeV)")
    print(f"    Energy range: {E_min} - {E_max} MeV")

    print(f"\n(a) Total fluence:")
    print(f"    F = integral(A * E^(-2) dE) from {E_min} to {E_max}")
    print(f"    = A * (1/{E_min} - 1/{E_max})")
    print(f"    = {A:.0e} * ({1/E_min:.4f} - {1/E_max:.6f})")
    print(f"    = {F:.3e} protons/cm^2")

    # (b) Absorbed dose
    # D = <dE/dx> * F * conversion
    dEdx = 5  # MeV*cm^2/g (mean stopping power)
    # D [rad] = dEdx [MeV*cm^2/g] * F [protons/cm^2] * 1.6e-6 [erg/MeV] * 1e-4 [J/erg]
    #           / 1e-3 [kg/g]
    # = dEdx * F * 1.6e-6 * 1e-4 / 1e-3
    # = dEdx * F * 1.6e-7
    # Actually in CGS: D [rad] = dEdx * F * 1.6e-6 [erg/MeV] * 1 [cm^2/(g)] / 100
    # Wait, let's be careful:
    # D [Gy] = dEdx [MeV*cm^2/g] * F [/cm^2] * 1.602e-13 [J/MeV] * 1000 [g/kg]
    # D [Gy] = dEdx * F * 1.602e-10
    D_Gy = dEdx * F * 1.602e-13 * 1000  # Gy

    # In rad: 1 Gy = 100 rad
    D_rad = D_Gy * 100

    print(f"\n(b) Absorbed dose:")
    print(f"    <dE/dx> = {dEdx} MeV*cm^2/g")
    print(f"    D = <dE/dx> * F * (1.602e-13 J/MeV) * (1000 g/kg)")
    print(f"    = {dEdx} * {F:.3e} * 1.602e-10")
    print(f"    = {D_Gy:.4f} Gy = {D_rad:.2f} rad")

    # (c) Dose equivalent with w_R = 2
    w_R = 2
    H = D_Gy * w_R  # Sv
    H_mSv = H * 1000

    print(f"\n(c) Dose equivalent (w_R = {w_R} for protons):")
    print(f"    H = D * w_R = {D_Gy:.4f} * {w_R} = {H:.4f} Sv = {H_mSv:.1f} mSv")

    # (d) Compare to NASA limit
    NASA_30day = 250  # mSv

    print(f"\n(d) Comparison with NASA 30-day BFO limit:")
    print(f"    Dose equivalent: {H_mSv:.1f} mSv")
    print(f"    NASA 30-day limit: {NASA_30day} mSv")
    print(f"    This is {H_mSv/NASA_30day*100:.1f}% of the 30-day limit")
    if H_mSv > NASA_30day:
        print(f"    EXCEEDS the 30-day limit! Dangerous for unshielded astronaut.")
    elif H_mSv > 100:
        print(f"    Significant fraction of the limit; dangerous for EVA")
    else:
        print(f"    Below the 30-day limit but still significant for EVA conditions")
    print(f"    Note: This is the unshielded dose. Behind spacecraft shielding,")
    print(f"    the dose would be reduced significantly for the lower energies.")


def exercise_4():
    """
    Exercise 4: Aviation Dose Estimate

    Polar flight: 12 km, 10 hours. GCR = 5 uSv/hr.
    (a) Total GCR dose.
    (b) With SEP: +20 uSv/hr additional.
    (c) Rerouted: equatorial, +3 hr, GCR = 2 uSv/hr, no SEP.
    (d) Annual crew dose for 80 polar flights.
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Aviation Dose Estimate")
    print("=" * 70)

    t_polar = 10     # hours
    GCR_polar = 5    # uSv/hr
    SEP_add = 20     # uSv/hr (additional)
    GCR_equat = 2    # uSv/hr
    t_equat = 13     # hours (10 + 3)
    n_flights = 80   # per year
    occ_limit = 20e3  # uSv/year (20 mSv)

    # (a) GCR dose for polar flight
    D_GCR = GCR_polar * t_polar

    print(f"\n    Polar flight: {t_polar} hr at 12 km, GCR rate = {GCR_polar} uSv/hr")

    print(f"\n(a) Total GCR dose for polar flight:")
    print(f"    D = {GCR_polar} uSv/hr * {t_polar} hr = {D_GCR} uSv")

    # (b) With SEP event
    D_total = (GCR_polar + SEP_add) * t_polar

    print(f"\n(b) During moderate SEP event (+{SEP_add} uSv/hr):")
    print(f"    D = ({GCR_polar} + {SEP_add}) * {t_polar} = {D_total} uSv")
    print(f"    = {D_total/1000:.2f} mSv")

    # (c) Rerouted equatorial flight
    D_rerouted = GCR_equat * t_equat

    print(f"\n(c) Rerouted equatorial flight ({t_equat} hr, no SEP):")
    print(f"    D = {GCR_equat} uSv/hr * {t_equat} hr = {D_rerouted} uSv")
    print(f"    Dose savings vs SEP polar: {D_total - D_rerouted} uSv")
    print(f"    = {(D_total - D_rerouted)/1000:.2f} mSv saved")

    # (d) Annual crew dose
    D_annual = n_flights * D_GCR

    print(f"\n(d) Annual dose for crew ({n_flights} polar flights, no SEP):")
    print(f"    D_annual = {n_flights} * {D_GCR} uSv = {D_annual} uSv")
    print(f"    = {D_annual/1000:.1f} mSv")
    print(f"    Occupational limit: {occ_limit/1000:.0f} mSv/year")
    print(f"    Fraction of limit: {D_annual/occ_limit*100:.0f}%")
    if D_annual > occ_limit:
        print(f"    EXCEEDS occupational limit!")
    else:
        print(f"    Within limit, but {D_annual/occ_limit*100:.0f}% is significant")
        print(f"    Airlines must monitor cumulative crew doses carefully")


def exercise_5():
    """
    Exercise 5: Proton Shielding Effectiveness

    Range in aluminum: R(E) ~ 0.0022 * E^1.77 g/cm^2 (E in MeV).
    rho_Al = 2.7 g/cm^3.
    (a) Thickness for 30, 100, 500 MeV protons.
    (b) Min energy penetrating 10 g/cm^2 (ISS).
    (c) Min energy penetrating 0.3 g/cm^2 (EVA suit).
    (d) Why GLE events are dangerous even inside ISS.
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Proton Shielding Effectiveness")
    print("=" * 70)

    rho_Al = 2.7  # g/cm^3

    def proton_range(E):
        """Range in g/cm^2 for proton of energy E (MeV)."""
        return 0.0022 * E**1.77

    def range_to_mm(R_gcm2):
        """Convert range from g/cm^2 to mm of aluminum."""
        return R_gcm2 / rho_Al * 10  # cm -> mm

    def energy_for_range(R_gcm2):
        """Find proton energy that has given range."""
        return (R_gcm2 / 0.0022)**(1 / 1.77)

    # (a) Shielding thickness for various energies
    print(f"\n    Range formula: R(E) = 0.0022 * E^1.77 g/cm^2 (E in MeV)")
    print(f"    rho_Al = {rho_Al} g/cm^3")

    energies = [30, 100, 500]
    print(f"\n(a) Aluminum thickness to stop protons:")
    print(f"    {'E (MeV)':>10} {'R (g/cm^2)':>14} {'Thickness (mm)':>16}")
    print(f"    {'-'*42}")
    for E in energies:
        R = proton_range(E)
        t_mm = range_to_mm(R)
        print(f"    {E:>10} {R:>14.2f} {t_mm:>16.1f}")

    # (b) ISS shielding: 10 g/cm^2
    R_ISS = 10  # g/cm^2
    E_min_ISS = energy_for_range(R_ISS)

    print(f"\n(b) Minimum energy penetrating ISS (10 g/cm^2):")
    print(f"    E = (R/0.0022)^(1/1.77) = ({R_ISS}/0.0022)^({1/1.77:.4f})")
    print(f"    E = {E_min_ISS:.0f} MeV")
    print(f"    Protons above ~{E_min_ISS:.0f} MeV penetrate ISS shielding")

    # (c) EVA suit: 0.3 g/cm^2
    R_EVA = 0.3  # g/cm^2
    E_min_EVA = energy_for_range(R_EVA)

    print(f"\n(c) Minimum energy penetrating EVA suit (0.3 g/cm^2):")
    print(f"    E = (0.3/0.0022)^(1/1.77) = {E_min_EVA:.0f} MeV")
    print(f"    Protons above ~{E_min_EVA:.0f} MeV penetrate the suit")
    print(f"    This means most SEP protons (>10 MeV) penetrate the suit!")

    # (d) Discussion of GLE events
    print(f"\n(d) Why GLE events are particularly dangerous:")
    print(f"    GLE events have significant flux above 500 MeV (even >1 GeV)")
    print(f"    At 500 MeV, range = {proton_range(500):.1f} g/cm^2")
    print(f"    This exceeds ISS shielding of 10 g/cm^2 by a factor of "
          f"{proton_range(500)/R_ISS:.1f}")
    print(f"    Even the most heavily shielded areas (~20 g/cm^2) are penetrated")
    print(f"    by protons above ~{energy_for_range(20):.0f} MeV")
    print(f"    GLE-class events cannot be fully shielded with practical")
    print(f"    spacecraft materials; the only protection is advance warning")
    print(f"    and retreat to the most shielded area available")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
