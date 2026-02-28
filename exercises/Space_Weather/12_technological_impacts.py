"""
Exercise Solutions for Lesson 12: Technological Impacts

Topics covered:
  - Surface charging at GEO from substorm injection
  - Internal charging timescale (dielectric relaxation)
  - Single-event upset (SEU) rate estimation
  - Solar cell degradation from radiation fluence
  - GPS range error during geomagnetic storm
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Surface Charging at GEO

    Substorm injection: electron flux = 1e8 el/(cm^2 s sr), E = 10 keV.
    Surface element: 1 m^2.
    (a) Charging current (isotropic flux over 2pi sr).
    (b) Charging rate with C = 100 pF; time to 1000 V.
    (c) Photoelectron emission: 1e10 el/(cm^2 s) on sunlit side.
    (d) Differential charging discussion.
    """
    print("=" * 70)
    print("Exercise 1: Surface Charging at GEO")
    print("=" * 70)

    e = 1.602e-19   # C
    J = 1e8          # electrons/(cm^2 s sr) - directional flux
    A_cm2 = 1e4      # 1 m^2 in cm^2
    C = 100e-12      # F (100 pF)
    J_photo = 1e10   # photoelectrons/(cm^2 s)

    # (a) Charging current
    # For isotropic flux integrated over hemisphere (2*pi sr):
    # The current to a flat surface from isotropic flux is J * pi * A (not 2*pi*J*A)
    # because of the cosine factor: integral over hemisphere of J*cos(theta)*d(Omega)
    # = J * pi steradians (effective)
    I_electron = e * J * np.pi * A_cm2  # using pi factor for hemisphere with cosine

    print(f"\n    Electron flux: J = {J:.0e} el/(cm^2 s sr)")
    print(f"    Surface area: A = 1 m^2 = {A_cm2:.0e} cm^2")

    print(f"\n(a) Electron charging current:")
    print(f"    For isotropic flux over 2pi hemisphere:")
    print(f"    I = e * J * pi * A (cosine-weighted hemisphere integral)")
    print(f"    = {e:.3e} * {J:.0e} * pi * {A_cm2:.0e}")
    print(f"    = {I_electron:.3e} A = {I_electron*1e6:.1f} uA")

    # Simpler estimate using 2*pi (problem's formula, ignoring cosine factor)
    I_simple = e * J * A_cm2 * 2 * np.pi
    print(f"    Using 2*pi (as in problem): I = {I_simple:.3e} A = {I_simple*1e6:.1f} uA")

    # (b) Charging rate
    dVdt = I_simple / C
    t_1kV = 1000 / dVdt  # seconds to reach 1000 V

    print(f"\n(b) Charging rate (C = {C*1e12:.0f} pF):")
    print(f"    dV/dt = I/C = {I_simple:.3e} / {C:.0e}")
    print(f"    = {dVdt:.1f} V/s")
    print(f"    Time to 1000 V: {1000}/{dVdt:.1f} = {t_1kV:.2f} s")
    print(f"    Surface charges to kilovolt levels in seconds!")

    # (c) Photoelectron emission
    I_photo = e * J_photo * A_cm2  # photoelectron current (leaving surface)

    print(f"\n(c) Photoelectron emission (sunlit side):")
    print(f"    J_photo = {J_photo:.0e} el/(cm^2 s)")
    print(f"    I_photo = e * J_photo * A = {I_photo:.3e} A = {I_photo*1e6:.1f} uA")
    print(f"    Compared to electron current: {I_simple*1e6:.1f} uA")

    if I_photo > I_simple:
        print(f"    Photoelectron current EXCEEDS electron current on sunlit side")
        print(f"    => Sunlit surface will charge to low positive potential (few V)")
        print(f"    The photoemission limits sunlit surface charging")
    else:
        print(f"    Photoelectron current is insufficient to prevent charging")

    print(f"\n    SHADOWED side: No photoelectrons! Charges to -{t_1kV*dVdt:.0f}+ V")

    # (d) Differential charging
    print(f"\n(d) Why differential charging is more dangerous:")
    print(f"    - Absolute charging (entire spacecraft at same potential) is")
    print(f"      relatively benign: all surfaces at same voltage, no E-fields")
    print(f"    - Differential charging: sunlit side ~+5 V, shadowed side ~-kV")
    print(f"    - This creates large electric fields across the spacecraft surface")
    print(f"    - E-fields across insulating surfaces can exceed dielectric")
    print(f"      breakdown threshold -> electrostatic discharge (ESD)")
    print(f"    - ESD generates electromagnetic interference that can cause")
    print(f"      logic upsets, phantom commands, and component damage")
    print(f"    - Most spacecraft charging anomalies are from differential charging")


def exercise_2():
    """
    Exercise 2: Internal Charging Timescale

    Teflon: eps_r = 2.1, sigma_dc = 1e-18 S/m.
    Beam current density: J_beam = 1e-11 A/cm^2.
    (a) Charge relaxation time.
    (b) Steady-state internal E-field.
    (c) Compare to breakdown field (100 kV/mm = 1e8 V/m).
    (d) Carbon-loaded Teflon (sigma = 1e-14 S/m).
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Internal Charging Timescale")
    print("=" * 70)

    eps0 = 8.854e-12     # F/m
    eps_r = 2.1
    sigma_dc = 1e-18     # S/m
    J_beam = 1e-11 * 1e4  # A/cm^2 -> A/m^2
    E_breakdown = 1e8    # V/m (100 kV/mm)

    # (a) Charge relaxation time
    tau = eps0 * eps_r / sigma_dc

    print(f"\n    Teflon: eps_r = {eps_r}, sigma = {sigma_dc:.0e} S/m")
    print(f"    Beam current: J = {J_beam:.0e} A/m^2")

    print(f"\n(a) Charge relaxation time:")
    print(f"    tau = eps0 * eps_r / sigma")
    print(f"    = {eps0:.3e} * {eps_r} / {sigma_dc:.0e}")
    print(f"    = {tau:.3e} s")
    print(f"    = {tau/3600:.0f} hours = {tau/86400:.0f} days")
    print(f"    Charge dissipates EXTREMELY slowly in pure Teflon!")

    # (b) Steady-state internal E-field
    E_ss = J_beam / sigma_dc  # V/m

    print(f"\n(b) Steady-state internal electric field:")
    print(f"    E_ss = J_beam / sigma")
    print(f"    = {J_beam:.0e} / {sigma_dc:.0e}")
    print(f"    = {E_ss:.1e} V/m")
    print(f"    = {E_ss/1e6:.0f} MV/m = {E_ss/1e8:.1f} * 10^8 V/m")

    # (c) Compare to breakdown
    ratio = E_ss / E_breakdown
    print(f"\n(c) Comparison to breakdown field ({E_breakdown:.0e} V/m = 100 kV/mm):")
    print(f"    E_ss / E_breakdown = {ratio:.1f}")
    if ratio > 1:
        print(f"    E_ss EXCEEDS breakdown by factor {ratio:.0f}!")
        print(f"    Dielectric breakdown WILL occur before steady state is reached")
        print(f"    The result: electrostatic discharge (ESD) through the insulation")
        print(f"    This can damage cables, connectors, and nearby electronics")
    else:
        print(f"    Below breakdown threshold")

    # (d) Carbon-loaded Teflon
    sigma_carbon = 1e-14  # S/m
    tau_carbon = eps0 * eps_r / sigma_carbon
    E_ss_carbon = J_beam / sigma_carbon

    print(f"\n(d) Carbon-loaded Teflon (sigma = {sigma_carbon:.0e} S/m):")
    print(f"    tau = {tau_carbon:.1f} s = {tau_carbon/3600:.2f} hours")
    print(f"    E_ss = {J_beam:.0e} / {sigma_carbon:.0e} = {E_ss_carbon:.1e} V/m")
    print(f"    = {E_ss_carbon/1e3:.0f} kV/m = {E_ss_carbon/1e6:.1f} MV/m")
    print(f"    E_ss / E_breakdown = {E_ss_carbon/E_breakdown:.4f}")
    if E_ss_carbon < E_breakdown:
        print(f"    Well below breakdown! Carbon loading is sufficient to prevent ESD")
        print(f"    The 10,000x higher conductivity reduces E_ss by 10,000x")


def exercise_3():
    """
    Exercise 3: SEU Rate Estimation

    SEU cross-section: 0 below LET_th = 5, saturates at sigma_sat = 1e-7 cm^2/bit
    above LET = 20 MeV*cm^2/mg.
    GCR flux above LET_th: ~1e2 particles/(cm^2 day).
    (a) SEU rate per bit per day.
    (b) SEU per day for 1e9 bits.
    (c) Double-bit error probability.
    (d) Effect of memory scrubbing every 10 s.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: SEU Rate Estimation")
    print("=" * 70)

    sigma_sat = 1e-7    # cm^2/bit
    Phi = 1e2           # particles/(cm^2 day) above LET_th
    N_bits = 1e9        # total bits
    word_size = 32      # bits per word

    # (a) SEU rate per bit per day (upper bound)
    R_bit = sigma_sat * Phi  # per bit per day

    print(f"\n    sigma_sat = {sigma_sat:.0e} cm^2/bit")
    print(f"    Phi(>LET_th) = {Phi:.0e} particles/(cm^2 day)")

    print(f"\n(a) Upper bound SEU rate per bit per day:")
    print(f"    R <= sigma_sat * Phi = {sigma_sat:.0e} * {Phi:.0e}")
    print(f"    R = {R_bit:.0e} SEU/(bit*day)")

    # (b) Total SEU for satellite
    R_total_day = R_bit * N_bits
    R_total_year = R_total_day * 365

    print(f"\n(b) For {N_bits:.0e} bits of memory:")
    print(f"    SEU per day: {R_bit:.0e} * {N_bits:.0e} = {R_total_day:.0f}")
    print(f"    SEU per year: {R_total_day:.0f} * 365 = {R_total_year:.0f}")
    print(f"    That is ~{R_total_day:.0f} single-bit errors per day!")

    # (c) Double-bit error probability
    # For a 32-bit word, the probability of 2 SEUs in the same word
    # before scrubbing is approximately:
    # P(double) ~ R_single^2 * (word_size/2) * T_scrub
    # where R_single is the rate per word
    # Rate per word per second: R_word = R_bit * word_size / 86400
    R_word_s = R_bit * word_size / 86400  # per word per second
    N_words = N_bits / word_size

    # Without scrubbing: probability of double-bit in a word over 1 day
    # P(2 in 1 word in 1 day) ~ (R_bit * T_day)^2 * C(32,2) ... complex
    # Simpler: the number of double-bit events per day
    # N_double ~ N_words * (R_word_day)^2 / 2  where R_word_day = R_bit * 32 * 1 day
    R_word_day = R_bit * word_size  # SEU per word per day

    # Expected number of words with >= 2 hits in 1 day (Poisson)
    # P(>=2) ~ (R_word_day)^2/2 per word
    P_double_per_word = R_word_day**2 / 2
    N_double_day = N_words * P_double_per_word

    print(f"\n(c) Double-bit error probability (no scrubbing, 1-day window):")
    print(f"    Words: {N_words:.0e} ({word_size}-bit)")
    print(f"    SEU rate per word per day: {R_word_day:.1e}")
    print(f"    P(double-bit in one word per day) ~ lambda^2/2 = {P_double_per_word:.1e}")
    print(f"    Expected double-bit errors per day across all memory:")
    print(f"    = {N_words:.0e} * {P_double_per_word:.1e} = {N_double_day:.3e}")
    print(f"    Very unlikely per day without scrubbing")

    # (d) With scrubbing every 10 seconds
    T_scrub = 10  # seconds
    R_word_scrub = R_bit * word_size * T_scrub / 86400  # per word per scrub interval
    P_double_scrub = R_word_scrub**2 / 2
    N_scrubs = 86400 / T_scrub  # scrubs per day
    N_double_scrub_day = N_words * P_double_scrub * N_scrubs

    print(f"\n(d) With memory scrubbing every {T_scrub} seconds:")
    print(f"    SEU per word per scrub interval: {R_word_scrub:.1e}")
    print(f"    P(double-bit per word per interval): {P_double_scrub:.1e}")
    print(f"    Number of scrub intervals per day: {N_scrubs:.0f}")
    print(f"    Expected double-bit errors per day: {N_double_scrub_day:.1e}")
    print(f"    Scrubbing reduces the double-bit error rate by correcting")
    print(f"    single-bit errors before a second hit can occur in the same word.")
    print(f"    The rate scales as T_scrub^2, so faster scrubbing is very effective.")


def exercise_4():
    """
    Exercise 4: Solar Cell Degradation

    GEO triple-junction cells, 150 um cover glass.
    15-year mission: Phi_eq = 5e14 e/cm^2.
    P/P0 = 1 - k*ln(1 + Phi/Phi0), k=0.04, Phi0=1e13.
    (a) EOL power fraction.
    (b) BOL power for 10 kW EOL.
    (c) After SEP event adding 2e14 e/cm^2.
    (d) Discussion.
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Solar Cell Degradation")
    print("=" * 70)

    k = 0.04
    Phi0 = 1e13     # e/cm^2
    Phi_15yr = 5e14  # e/cm^2 (15-year fluence)
    P_EOL_req = 10   # kW

    # Degradation function
    def power_fraction(Phi):
        return 1 - k * np.log(1 + Phi / Phi0)

    # (a) EOL power fraction
    frac_15yr = power_fraction(Phi_15yr)

    print(f"\n    Model: P/P0 = 1 - {k}*ln(1 + Phi/{Phi0:.0e})")
    print(f"    15-year fluence: Phi = {Phi_15yr:.0e} e/cm^2")

    print(f"\n(a) EOL power fraction after 15 years:")
    print(f"    P/P0 = 1 - {k}*ln(1 + {Phi_15yr:.0e}/{Phi0:.0e})")
    print(f"    = 1 - {k}*ln({1 + Phi_15yr/Phi0:.0f})")
    print(f"    = 1 - {k}*{np.log(1 + Phi_15yr/Phi0):.4f}")
    print(f"    = {frac_15yr:.4f} = {frac_15yr*100:.1f}%")

    # (b) BOL power
    P_BOL = P_EOL_req / frac_15yr

    print(f"\n(b) Required BOL power for {P_EOL_req} kW at EOL:")
    print(f"    P_BOL = P_EOL / (P/P0) = {P_EOL_req} / {frac_15yr:.4f}")
    print(f"    = {P_BOL:.2f} kW")
    print(f"    Need {P_BOL - P_EOL_req:.2f} kW ({(P_BOL/P_EOL_req-1)*100:.1f}%) "
          f"margin for degradation")

    # (c) After SEP event
    Phi_SEP = 2e14
    Phi_total = Phi_15yr + Phi_SEP
    frac_after_SEP = power_fraction(Phi_total)

    print(f"\n(c) After large SEP event (additional {Phi_SEP:.0e} e/cm^2):")
    print(f"    Total fluence: {Phi_total:.0e} e/cm^2")
    print(f"    P/P0 = 1 - {k}*ln(1 + {Phi_total:.0e}/{Phi0:.0e})")
    print(f"    = {frac_after_SEP:.4f} = {frac_after_SEP*100:.1f}%")
    print(f"    Additional degradation from SEP: "
          f"{(frac_15yr - frac_after_SEP)*100:.1f} percentage points")

    P_after_SEP = P_BOL * frac_after_SEP
    print(f"    Actual power after SEP: {P_BOL:.2f} * {frac_after_SEP:.4f} "
          f"= {P_after_SEP:.2f} kW")
    if P_after_SEP < P_EOL_req:
        print(f"    WARNING: Power ({P_after_SEP:.2f} kW) falls below "
              f"requirement ({P_EOL_req} kW)!")
    else:
        print(f"    Still above {P_EOL_req} kW requirement")

    # (d) Discussion
    print(f"\n(d) Why a single SEP event can exceed years of trapped radiation:")
    print(f"    - Trapped belt particles are relatively low energy (<few MeV)")
    print(f"    - Cover glass stops most trapped particles")
    print(f"    - SEP events produce high-energy protons (10-100+ MeV)")
    print(f"    - These penetrate the cover glass and damage the cell junction")
    print(f"    - A single large SEP can deliver equivalent fluence of years")
    print(f"      of trapped radiation in just hours to days")
    print(f"    - This is why SEP events are the dominant radiation concern")
    print(f"      for GEO satellite solar arrays")


def exercise_5():
    """
    Exercise 5: GPS Range Error During Storm

    Quiet TEC = 30 TECU, storm TEC = 120 TECU.
    (a) L1 range error for both conditions.
    (b) Residual error with dual-frequency (2nd order ~0.1% of 1st order).
    (c) TEC gradient interpolation error (20 TECU/100 km, 300 km grid).
    (d) WAAS precision approach assessment.
    """
    print("\n" + "=" * 70)
    print("Exercise 5: GPS Range Error During Storm")
    print("=" * 70)

    f_L1 = 1.575e9   # Hz (1.575 GHz)
    TEC_quiet = 30    # TECU
    TEC_storm = 120   # TECU
    f_L1_Hz = f_L1

    # (a) Range error: DR = 40.3 * TEC / f^2
    # TEC in el/m^2: TECU * 1e16
    def range_error(TEC_TECU, f_Hz):
        return 40.3 * TEC_TECU * 1e16 / f_Hz**2

    DR_quiet = range_error(TEC_quiet, f_L1_Hz)
    DR_storm = range_error(TEC_storm, f_L1_Hz)

    print(f"\n    f_L1 = {f_L1/1e9:.3f} GHz")
    print(f"    TEC quiet = {TEC_quiet} TECU, storm = {TEC_storm} TECU")

    print(f"\n(a) Single-frequency range error at L1:")
    print(f"    DR = 40.3 * TEC * 1e16 / f^2")
    print(f"    Quiet:  DR = 40.3 * {TEC_quiet}e16 / ({f_L1:.3e})^2 = {DR_quiet:.1f} m")
    print(f"    Storm:  DR = 40.3 * {TEC_storm}e16 / ({f_L1:.3e})^2 = {DR_storm:.1f} m")
    print(f"    Storm error is {DR_storm/DR_quiet:.0f}x quiet error")

    # (b) Dual-frequency residual
    frac_2nd = 0.001  # 0.1% of first-order
    DR_2nd_quiet = frac_2nd * DR_quiet
    DR_2nd_storm = frac_2nd * DR_storm

    print(f"\n(b) Dual-frequency residual (2nd order ~ 0.1% of 1st):")
    print(f"    Quiet: {frac_2nd} * {DR_quiet:.1f} m = {DR_2nd_quiet*100:.1f} cm")
    print(f"    Storm: {frac_2nd} * {DR_storm:.1f} m = {DR_2nd_storm*100:.1f} cm")
    print(f"    2nd-order residual is cm-level — significant only for")
    print(f"    high-precision applications (geodesy, PPP)")

    # (c) TEC gradient interpolation error
    grad_TEC = 20     # TECU per 100 km
    grid_spacing = 300  # km

    # Maximum interpolation error occurs at the midpoint between grid points
    # for a linear interpolation over a non-linear TEC gradient
    # Worst case: the gradient is unresolved by the grid
    # Max TEC error ~ gradient * (grid_spacing / 2) - correction
    # If TEC varies linearly, interpolation is exact.
    # But sharp gradients mean TEC is not linear between grid points.
    # Max error ~ grad * grid_spacing / 2 (if gradient changes sign)
    # More conservatively: max TEC error ~ grad_TECU_per_km * grid_spacing/4
    grad_per_km = grad_TEC / 100  # TECU/km
    max_TEC_error = grad_per_km * grid_spacing / 2  # TECU (linear approx mismatch)

    DR_interp = range_error(max_TEC_error, f_L1_Hz)

    print(f"\n(c) TEC gradient interpolation error:")
    print(f"    TEC gradient: {grad_TEC} TECU / 100 km = {grad_per_km:.2f} TECU/km")
    print(f"    WAAS grid spacing: {grid_spacing} km")
    print(f"    Maximum TEC interpolation error between grid points:")
    print(f"    ~ gradient * (grid_spacing/2) = {grad_per_km:.2f} * {grid_spacing/2:.0f}")
    print(f"    = {max_TEC_error:.0f} TECU")
    print(f"    Corresponding range error: {DR_interp:.1f} m")

    # (d) WAAS precision approach assessment
    threshold = 40  # m (horizontal accuracy requirement)

    print(f"\n(d) WAAS precision approach assessment:")
    print(f"    Required horizontal accuracy: < {threshold} m")
    print(f"    Storm range error (single-freq): {DR_storm:.1f} m")
    print(f"    WAAS correction interpolation error: up to {DR_interp:.1f} m")
    if DR_interp > threshold:
        print(f"    WAAS correction error ({DR_interp:.1f} m) EXCEEDS "
              f"the {threshold} m threshold!")
    else:
        print(f"    WAAS correction error ({DR_interp:.1f} m) is within threshold,")
        print(f"    but integrity bounds may still be exceeded")
    print(f"    Operational consequence:")
    print(f"    - WAAS will likely declare the ionospheric correction unreliable")
    print(f"    - Protection levels will inflate beyond approach limits")
    print(f"    - Aircraft would need to revert to ILS or non-precision approaches")
    print(f"    - This can cause delays, diversions, and congestion at airports")
    print(f"      without ILS capability")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
