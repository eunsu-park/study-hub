"""
Exercises for Lesson 12: Nonlinear Optics
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Second Harmonic Generation (SHG) Efficiency in KTP
    Calculate the SHG conversion efficiency for a KTP crystal,
    including phase matching bandwidth and acceptance angle.
    """
    # KTP crystal parameters for SHG of 1064 nm
    lam_fund = 1064e-9   # Fundamental wavelength (m)
    lam_shg = 532e-9     # SHG wavelength (m)
    c = 3.0e8
    eps0 = 8.854e-12

    # Crystal parameters
    d_eff = 3.18e-12     # Effective nonlinear coefficient (m/V) for KTP Type II
    L = 10e-3            # Crystal length (m)
    n_fund = 1.830       # Refractive index at 1064 nm
    n_shg = 1.889        # Refractive index at 532 nm

    print("SHG in KTP Crystal:")
    print(f"Fundamental: {lam_fund*1e9:.0f} nm")
    print(f"SHG: {lam_shg*1e9:.0f} nm")
    print(f"Crystal length: L = {L*1e3:.0f} mm")
    print(f"d_eff = {d_eff*1e12:.2f} pm/V")

    # Phase mismatch
    k_fund = 2 * np.pi * n_fund / lam_fund
    k_shg = 2 * np.pi * n_shg / lam_shg
    delta_k = k_shg - 2 * k_fund

    print(f"\nPhase mismatch:")
    print(f"  k(omega) = {k_fund:.2f} rad/m")
    print(f"  k(2omega) = {k_shg:.2f} rad/m")
    print(f"  Delta_k = k(2w) - 2k(w) = {delta_k:.2f} rad/m")

    # Coherence length
    L_coh = np.pi / abs(delta_k) if delta_k != 0 else np.inf
    print(f"  Coherence length: L_coh = {L_coh*1e6:.1f} um")

    # SHG efficiency (low conversion, CW)
    # eta = (2 * d_eff^2 * omega^2 * L^2 * I) / (n^2 * n_2w * eps0 * c^3) * sinc^2(dkL/2)
    omega = 2 * np.pi * c / lam_fund

    # For perfect phase matching (delta_k = 0)
    print(f"\n--- Conversion Efficiency (Perfect Phase Matching) ---")

    beam_diameters = [100e-6, 50e-6, 20e-6]  # Beam diameter
    P_input = 1.0  # 1 W input power

    for d_beam in beam_diameters:
        A_beam = np.pi * (d_beam/2)**2
        I_fund = P_input / A_beam  # Intensity (W/m^2)

        # Normalized efficiency
        eta_norm = 2 * omega**2 * d_eff**2 * L**2 / (
            n_fund**2 * n_shg * eps0 * c**3
        )
        eta = eta_norm * I_fund

        P_shg = eta * P_input
        print(f"  Beam diameter {d_beam*1e6:.0f} um: "
              f"I = {I_fund/1e9:.1f} GW/m^2, "
              f"eta = {eta*100:.4f}%, "
              f"P_SHG = {P_shg*1e3:.3f} mW")

    # Sinc^2 factor for imperfect phase matching
    print(f"\n--- Phase Matching Sensitivity ---")
    print(f"{'Delta_k*L/2':>14} {'sinc^2':>10} {'Efficiency':>12}")
    print("-" * 38)

    for dkL_half in [0, 0.5, 1.0, np.pi/2, np.pi, 2*np.pi, 3*np.pi]:
        sinc2 = np.sinc(dkL_half/np.pi)**2  # np.sinc(x) = sin(pi*x)/(pi*x)
        print(f"{dkL_half:>14.3f} {sinc2:>10.4f} {sinc2*100:>11.2f}%")

    # Temperature bandwidth (for NCPM)
    # dT ~ 0.4425 * lambda / (L * |dn/dT_shg - dn/dT_fund|)
    dn_dT_diff = 1.5e-5  # /K (typical)
    delta_T = 0.4425 * lam_fund / (L * dn_dT_diff)
    print(f"\nTemperature acceptance bandwidth: dT ~ {delta_T:.1f} K")


def exercise_2():
    """
    Exercise 2: Quasi-Phase Matching (QPM) in PPLN
    Design a periodically poled lithium niobate (PPLN) crystal for
    efficient frequency conversion.
    """
    lam_fund = 1064e-9  # Fundamental wavelength (m)
    lam_shg = 532e-9
    c = 3.0e8

    print("QPM Design: Periodically Poled LiNbO3 (PPLN):")
    print(f"Fundamental: {lam_fund*1e9:.0f} nm -> SHG: {lam_shg*1e9:.0f} nm")

    # LiNbO3 refractive indices (Sellmeier at 25 C)
    # Simplified values for extraordinary polarization
    n_e_fund = 2.1560    # ne at 1064 nm
    n_e_shg = 2.2340     # ne at 532 nm

    # Phase mismatch without QPM
    k_fund = 2 * np.pi * n_e_fund / lam_fund
    k_shg = 2 * np.pi * n_e_shg / lam_shg
    delta_k = k_shg - 2 * k_fund

    print(f"\nn_e(1064 nm) = {n_e_fund}")
    print(f"n_e(532 nm)  = {n_e_shg}")
    print(f"Phase mismatch: Delta_k = {delta_k:.2f} rad/m")
    print(f"Coherence length: L_coh = {np.pi/abs(delta_k)*1e6:.2f} um")

    # QPM period: Lambda_QPM = 2*pi / |delta_k| (first order)
    Lambda_QPM = 2 * np.pi / abs(delta_k)
    print(f"\nQPM poling period (m=1): Lambda = {Lambda_QPM*1e6:.2f} um")

    # Higher-order QPM
    print(f"\nHigher-order QPM periods:")
    print(f"{'Order m':>10} {'Period (um)':>14} {'d_eff factor':>14}")
    print("-" * 40)
    d33_LN = 25.0e-12  # d33 for LiNbO3 (pm/V)

    for m in [1, 3, 5, 7]:
        Lambda_m = m * Lambda_QPM
        # d_eff_QPM = (2/m*pi) * d33
        d_eff_m = (2 / (m * np.pi)) * d33_LN
        factor = 2 / (m * np.pi)
        print(f"{m:>10} {Lambda_m*1e6:>14.2f} {factor:>14.4f}")

    # QPM vs BPM efficiency comparison
    print(f"\n--- Efficiency Comparison: QPM vs BPM ---")
    d_eff_QPM = (2/np.pi) * d33_LN  # First-order QPM
    d_eff_BPM = 4.7e-12  # Typical BPM d_eff for LiNbO3

    print(f"d_eff (QPM, m=1): {d_eff_QPM*1e12:.2f} pm/V")
    print(f"d_eff (BPM):      {d_eff_BPM*1e12:.2f} pm/V")
    print(f"Efficiency ratio (QPM/BPM): {(d_eff_QPM/d_eff_BPM)**2:.1f}x")
    print(f"  (QPM uses largest d33 coefficient, not angle-constrained)")

    # Temperature tuning
    print(f"\n--- Temperature Tuning ---")
    # dn/dT for LiNbO3: ~4e-5 /K
    dn_dT = 4e-5  # /K
    # d(delta_k)/dT = 2*pi/lam * (dn_shg/dT - 2*dn_fund/dT) -- simplified
    # QPM period shifts: d(Lambda)/dT
    T_range = np.arange(25, 225, 25)
    print(f"{'Temp (C)':>10} {'Delta_k shift':>14} {'Phase match?':>14}")
    print("-" * 40)
    for T in T_range:
        dT = T - 25  # From room temp
        # Approximate shift in delta_k
        dk_shift = 2 * np.pi * dn_dT * dT * (1/lam_shg - 2/lam_fund)
        dk_total = delta_k + dk_shift
        # Phase matched when dk_total = 2*pi/Lambda_QPM
        dk_residual = abs(dk_total) - 2*np.pi/Lambda_QPM
        matched = "Yes" if abs(dk_residual) < 100 else "No"
        print(f"{T:>10} {dk_shift:>14.1f} {matched:>14}")


def exercise_3():
    """
    Exercise 3: Self-Phase Modulation (SPM)
    Calculate the spectral broadening of an ultrashort pulse due to
    self-phase modulation in an optical fiber.
    """
    c = 3.0e8
    lam = 800e-9       # Wavelength (m)
    omega0 = 2 * np.pi * c / lam

    # Pulse parameters
    P_peak = 10e3      # Peak power (W)
    tau = 100e-15      # Pulse duration (FWHM, s)
    n2 = 2.6e-20       # Nonlinear refractive index of silica (m^2/W)
    A_eff = 50e-12     # Effective mode area (m^2)
    L = 0.10           # Fiber length (m)

    print("Self-Phase Modulation in Optical Fiber:")
    print(f"Wavelength: {lam*1e9:.0f} nm")
    print(f"Peak power: {P_peak/1e3:.0f} kW")
    print(f"Pulse duration (FWHM): {tau*1e15:.0f} fs")
    print(f"n2 = {n2:.1e} m^2/W")
    print(f"A_eff = {A_eff*1e12:.0f} um^2")
    print(f"Fiber length: {L*100:.0f} cm")

    # Nonlinear coefficient
    gamma = 2 * np.pi * n2 / (lam * A_eff)  # 1/(W*m)
    print(f"Nonlinear coefficient: gamma = {gamma:.4f} /(W*m)")

    # Maximum nonlinear phase shift
    phi_NL_max = gamma * P_peak * L
    print(f"\nMaximum nonlinear phase shift: phi_max = {phi_NL_max:.2f} rad")
    print(f"  = {phi_NL_max/(2*np.pi):.2f} * 2*pi")

    # B-integral
    B_integral = phi_NL_max
    print(f"B-integral: B = {B_integral:.2f} rad")

    # Spectral broadening factor
    # For Gaussian pulse: delta_omega_out / delta_omega_in ~ sqrt(1 + (4/3*sqrt(3)) * phi_max^2)
    BF = np.sqrt(1 + (4/(3*np.sqrt(3))) * phi_NL_max**2)
    delta_lam_in = lam**2 / (c * tau * np.pi)  # Transform-limited bandwidth
    delta_lam_out = BF * delta_lam_in

    print(f"\nSpectral broadening:")
    print(f"  Input bandwidth: {delta_lam_in*1e9:.2f} nm (transform-limited)")
    print(f"  Broadening factor: {BF:.2f}x")
    print(f"  Output bandwidth: {delta_lam_out*1e9:.2f} nm")

    # Number of spectral peaks ~ phi_max / pi
    N_peaks = int(phi_NL_max / np.pi) + 1
    print(f"  Expected spectral peaks: ~{N_peaks}")

    # Instantaneous frequency shift
    print(f"\n--- Instantaneous Frequency (Chirp) ---")
    # For Gaussian pulse: I(t) = I0 * exp(-2*(t/tau_0)^2)
    # phi_NL(t) = gamma * L * I(t)
    # delta_omega(t) = -d(phi_NL)/dt
    tau_0 = tau / (2 * np.sqrt(np.log(2)))  # 1/e^2 half-width

    t_points = np.linspace(-2*tau, 2*tau, 9)
    print(f"{'t/tau':>8} {'I/I_peak':>10} {'phi_NL (rad)':>14} {'d_omega/omega0':>16}")
    print("-" * 50)

    for t in t_points:
        I_t = P_peak * np.exp(-2 * (t/tau_0)**2)
        phi_t = gamma * L * I_t
        # d_omega = -d(phi)/dt = gamma * L * 4*t/tau_0^2 * I(t)
        d_omega = gamma * L * 4 * t / tau_0**2 * I_t
        print(f"{t/tau:>8.2f} {I_t/P_peak:>10.4f} {phi_t:>14.4f} {d_omega/omega0:>16.2e}")


def exercise_4():
    """
    Exercise 4: Phase Matching Angles for BBO Crystal
    Calculate the phase matching angles for Type I and Type II SHG
    in BBO (beta-BaB2O4) crystal.
    """
    c = 3.0e8

    print("Phase Matching in BBO Crystal:")

    # BBO Sellmeier equations (simplified)
    def n_o_bbo(lam_um):
        """Ordinary refractive index of BBO."""
        l2 = lam_um**2
        return np.sqrt(2.7359 + 0.01878 / (l2 - 0.01822) - 0.01354 * l2)

    def n_e_bbo(lam_um):
        """Extraordinary refractive index of BBO (along optic axis)."""
        l2 = lam_um**2
        return np.sqrt(2.3753 + 0.01224 / (l2 - 0.01667) - 0.01516 * l2)

    # Print refractive indices
    wavelengths = [0.400, 0.532, 0.800, 1.064]
    print(f"\n{'Lambda (nm)':>12} {'n_o':>10} {'n_e':>10} {'Biref.':>10}")
    print("-" * 44)
    for lam_um in wavelengths:
        no = n_o_bbo(lam_um)
        ne = n_e_bbo(lam_um)
        print(f"{lam_um*1000:>12.0f} {no:>10.4f} {ne:>10.4f} {no-ne:>10.4f}")

    # Type I SHG: o + o -> e
    # Phase matching: n_e(2w, theta) = n_o(w)
    # n_e(theta)^-2 = cos^2(theta)/n_o^2 + sin^2(theta)/n_e^2

    print(f"\n--- Type I SHG (o + o -> e) ---")

    fund_wavelengths = [0.800, 1.064, 0.600]
    for lam_fund_um in fund_wavelengths:
        lam_shg_um = lam_fund_um / 2

        n_o_w = n_o_bbo(lam_fund_um)
        n_o_2w = n_o_bbo(lam_shg_um)
        n_e_2w = n_e_bbo(lam_shg_um)

        # Phase matching: n_eff(theta) at 2w = n_o at w
        # 1/n_eff^2 = cos^2(theta)/n_o_2w^2 + sin^2(theta)/n_e_2w^2 = 1/n_o_w^2
        # sin^2(theta) = (1/n_o_w^2 - 1/n_o_2w^2) / (1/n_e_2w^2 - 1/n_o_2w^2)

        num = 1/n_o_w**2 - 1/n_o_2w**2
        den = 1/n_e_2w**2 - 1/n_o_2w**2

        if 0 <= num/den <= 1:
            sin2_theta = num / den
            theta_pm = np.degrees(np.arcsin(np.sqrt(sin2_theta)))
            print(f"  {lam_fund_um*1000:.0f} nm -> {lam_shg_um*1000:.0f} nm: "
                  f"theta_PM = {theta_pm:.2f} deg")
        else:
            print(f"  {lam_fund_um*1000:.0f} nm -> {lam_shg_um*1000:.0f} nm: "
                  f"No phase matching possible")

    # Walk-off angle
    print(f"\n--- Walk-Off Angle ---")
    lam_fund_um = 0.800
    lam_shg_um = 0.400
    n_o_w = n_o_bbo(lam_fund_um)
    n_o_2w = n_o_bbo(lam_shg_um)
    n_e_2w = n_e_bbo(lam_shg_um)

    # Phase matching angle
    num = 1/n_o_w**2 - 1/n_o_2w**2
    den = 1/n_e_2w**2 - 1/n_o_2w**2
    sin2_theta = num / den
    theta_pm = np.arcsin(np.sqrt(sin2_theta))

    # Walk-off: rho = arctan(n_e^2/n_o^2 * tan(theta)) - theta (for e-ray)
    # More precisely: tan(rho) = sin(2*theta)/2 * (1/n_e^2 - 1/n_o^2) / (sin^2(theta)/n_e^2 + cos^2(theta)/n_o^2)
    rho_num = np.sin(2*theta_pm) * (1/n_e_2w**2 - 1/n_o_2w**2)
    rho_den = 2 * (np.sin(theta_pm)**2/n_e_2w**2 + np.cos(theta_pm)**2/n_o_2w**2)
    rho = np.arctan(abs(rho_num / rho_den))

    print(f"At 800 nm -> 400 nm SHG:")
    print(f"  Phase matching angle: {np.degrees(theta_pm):.2f} deg")
    print(f"  Walk-off angle: {np.degrees(rho):.2f} deg ({np.degrees(rho)*1000:.1f} mrad)")

    # Aperture length (effective interaction length limited by walk-off)
    w_beam = 100e-6  # Beam waist
    L_a = w_beam * np.sqrt(np.pi) / np.tan(rho)
    print(f"  Beam waist: {w_beam*1e6:.0f} um")
    print(f"  Aperture length: L_a = {L_a*1e3:.1f} mm")
    print(f"  (Crystal should be shorter than L_a for efficient conversion)")

    # Angular acceptance bandwidth
    # delta_theta * L ~ lambda / (n_o * sin(2*theta_PM))
    L_crystal = 5e-3  # 5 mm crystal
    delta_theta = lam_fund_um * 1e-6 / (n_o_w * np.sin(2*theta_pm) * L_crystal)
    print(f"\n  Angular acceptance (L={L_crystal*1e3:.0f} mm): "
          f"{np.degrees(delta_theta)*60:.2f} arcmin ({np.degrees(delta_theta):.3f} deg)")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: SHG Efficiency in KTP", exercise_1),
        ("Exercise 2: QPM Design (PPLN)", exercise_2),
        ("Exercise 3: Self-Phase Modulation", exercise_3),
        ("Exercise 4: Phase Matching Angles (BBO)", exercise_4),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")
