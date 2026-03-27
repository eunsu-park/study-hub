"""
Plasma Physics - Lesson 12: Wave Heating and Instabilities
Exercise Solutions

Topics covered:
- ECRH system design with density profile considerations
- Two-stream instability growth rate
- Weibel instability in laser-heated plasma
- Solar wind anisotropy stability analysis
- Laser-plasma instabilities (SRS) in ICF
"""

import numpy as np

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
m_p = 1.673e-27
epsilon_0 = 8.854e-12
k_B = 1.381e-23
mu_0 = 4 * np.pi * 1e-7
c = 3e8
eV_to_J = e


def exercise_1():
    """
    Exercise 1: ECRH System Design
    Tokamak: B_0 = 2.5 T, n(r) = n_0*(1-r^2/a^2)^2, n_0 = 8e19, a = 0.5 m.
    """
    print("--- Exercise 1: ECRH System Design ---")

    B_0 = 2.5
    n_0 = 8e19
    a = 0.5

    # (a) Electron cyclotron frequency on axis
    omega_ce = e * B_0 / m_e
    f_ce = omega_ce / (2 * np.pi)

    print(f"(a) B_0 = {B_0} T on axis")
    print(f"    f_ce = {f_ce/1e9:.2f} GHz")

    # (b) 2nd harmonic
    f_2ce = 2 * f_ce
    omega_2ce = 2 * omega_ce

    print(f"\n(b) 2nd harmonic ECRH:")
    print(f"    f_gyrotron = 2*f_ce = {f_2ce/1e9:.1f} GHz")

    # (c) O-mode cutoff density
    n_cutoff = epsilon_0 * m_e * omega_2ce**2 / e**2

    print(f"\n(c) O-mode cutoff density at 2*f_ce:")
    print(f"    n_cutoff = {n_cutoff:.2e} m^-3")
    print(f"    Core density n_0 = {n_0:.0e} m^-3")
    if n_0 < n_cutoff:
        print(f"    n_0 < n_cutoff: O-mode CAN reach the core")
    else:
        print(f"    n_0 > n_cutoff: O-mode CANNOT reach the core")
        # Find where n(r) = n_cutoff
        # n_0 * (1 - r^2/a^2)^2 = n_cutoff
        # (1 - r^2/a^2) = (n_cutoff/n_0)^(1/2)
        # r/a = sqrt(1 - (n_cutoff/n_0)^(1/2))
        ratio = np.sqrt(n_cutoff / n_0)
        if ratio <= 1:
            r_cutoff = a * np.sqrt(1 - ratio)
            print(f"    O-mode cutoff at r = {r_cutoff*100:.1f} cm (r/a = {r_cutoff/a:.3f})")
        else:
            print(f"    O-mode can propagate everywhere")

    # (d) X-mode: upper hybrid resonance location
    # f_UH^2 = f_pe^2 + f_ce^2 = (2*f_ce)^2
    # f_pe^2 = 4*f_ce^2 - f_ce^2 = 3*f_ce^2
    # n_UH such that omega_pe^2 = 3*omega_ce^2
    n_UH = 3 * epsilon_0 * m_e * omega_ce**2 / e**2

    print(f"\n(d) X-mode UH resonance:")
    print(f"    n_UH = {n_UH:.2e} m^-3")
    # Find location
    ratio_uh = np.sqrt(n_UH / n_0)
    if ratio_uh <= 1:
        r_UH = a * np.sqrt(1 - ratio_uh)
        print(f"    UH resonance at r = {r_UH*100:.1f} cm (r/a = {r_UH/a:.3f})")
    else:
        print(f"    UH resonance density exceeds profile everywhere")
    print()


def exercise_2():
    """
    Exercise 2: Two-Stream Instability
    Electron beam: n_b = 10^17, v_0 = 10^7 m/s through n_0 = 10^19 plasma.
    """
    print("--- Exercise 2: Two-Stream Instability ---")

    n_0 = 1e19     # Background density
    n_b = 1e17     # Beam density
    v_0 = 1e7      # Beam velocity [m/s]

    # (a) Background plasma frequency
    omega_p0 = np.sqrt(n_0 * e**2 / (epsilon_0 * m_e))
    f_p0 = omega_p0 / (2 * np.pi)

    print(f"(a) Background plasma frequency:")
    print(f"    omega_p0 = {omega_p0:.4e} rad/s")
    print(f"    f_p0 = {f_p0/1e9:.2f} GHz")

    # (b) Growth rate: gamma ~ omega_p0 * (n_b/n_0)^(1/3)
    density_ratio = n_b / n_0
    gamma = omega_p0 * density_ratio**(1.0 / 3)

    print(f"\n(b) Two-stream growth rate:")
    print(f"    n_b/n_0 = {density_ratio:.0e}")
    print(f"    gamma ~ omega_p0 * (n_b/n_0)^(1/3)")
    print(f"    gamma = {gamma:.4e} rad/s")
    print(f"    gamma/omega_p0 = {gamma/omega_p0:.4f}")

    # (c) Resonant wavenumber
    k_res = omega_p0 / v_0
    lambda_res = 2 * np.pi / k_res

    print(f"\n(c) Resonant wavenumber:")
    print(f"    k_res = omega_p0/v_0 = {k_res:.4e} m^-1")
    print(f"    Wavelength = {lambda_res*1e3:.2f} mm")

    # (d) E-folding analysis
    factor_1000 = np.log(1000)
    n_efold = factor_1000  # ln(1000) ~ 6.9
    t_efold = 1.0 / gamma
    t_1000 = n_efold * t_efold

    print(f"\n(d) Growth to factor 1000:")
    print(f"    Number of e-foldings: ln(1000) = {factor_1000:.1f}")
    print(f"    e-folding time: tau = 1/gamma = {t_efold:.4e} s")
    print(f"    Time for 1000x growth: {t_1000:.4e} s")

    # Beam transit time
    L = 1.0  # Assume 1 m plasma
    t_transit = 1e-6  # Given as 1 us
    n_efold_transit = gamma * t_transit

    print(f"    Beam transit time: {t_transit*1e6:.0f} us")
    print(f"    e-foldings during transit: {n_efold_transit:.1f}")
    print(f"    Growth factor: exp({n_efold_transit:.1f}) = {np.exp(n_efold_transit):.2e}")
    if n_efold_transit > n_efold:
        print(f"    >> 1000x growth: YES, instability is strongly driven")
    else:
        print(f"    < 1000x: instability may not fully develop")
    print()


def exercise_3():
    """
    Exercise 3: Weibel Instability in Laser Plasmas
    T_perp = 500 eV, T_par = 50 eV, n = 10^21 m^-3.
    """
    print("--- Exercise 3: Weibel Instability ---")

    T_perp_eV = 500
    T_par_eV = 50
    T_perp = T_perp_eV * eV_to_J
    T_par = T_par_eV * eV_to_J
    n = 1e21

    # (a) Anisotropy parameter
    A = (T_perp - T_par) / T_par

    print(f"(a) Temperature anisotropy:")
    print(f"    T_perp = {T_perp_eV} eV, T_par = {T_par_eV} eV")
    print(f"    Anisotropy: A = (T_perp - T_par)/T_par = {A:.1f}")

    # (b) Maximum Weibel growth rate
    # gamma_max ~ omega_pe * sqrt(A/(1+A)) * (v_th_perp/c)
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    v_th_perp = np.sqrt(T_perp / m_e)
    v_th_par = np.sqrt(T_par / m_e)

    gamma_max = omega_pe * np.sqrt(A / (1 + A)) * (v_th_perp / c)

    print(f"\n(b) Maximum Weibel growth rate:")
    print(f"    omega_pe = {omega_pe:.4e} rad/s")
    print(f"    v_th,perp/c = {v_th_perp/c:.4f}")
    print(f"    gamma_max ~ omega_pe * sqrt(A/(1+A)) * v_th_perp/c")
    print(f"    gamma_max = {gamma_max:.4e} rad/s")
    print(f"    gamma_max/omega_pe = {gamma_max/omega_pe:.4e}")

    # (c) Generated magnetic field
    # B^2/(2*mu_0) ~ n * k_B * (T_perp - T_par) [energy equipartition estimate]
    B_weibel = np.sqrt(2 * mu_0 * n * (T_perp - T_par))

    print(f"\n(c) Self-generated magnetic field:")
    print(f"    B^2/(2*mu_0) ~ n*(T_perp - T_par)")
    print(f"    B_Weibel ~ {B_weibel:.2f} T = {B_weibel*1e4:.0f} Gauss")

    # (d) Electron magnetization
    # Larmor radius in self-generated field
    omega_ce = e * B_weibel / m_e
    rho_e = v_th_perp / omega_ce
    k_max = omega_pe / c * np.sqrt(A / (1 + A))
    lambda_max = 2 * np.pi / k_max

    print(f"\n(d) Electron magnetization by self-generated field:")
    print(f"    omega_ce = {omega_ce:.4e} rad/s")
    print(f"    rho_e = {rho_e:.4e} m")
    print(f"    Instability scale: 1/k_max ~ {1/k_max:.4e} m")
    print(f"    rho_e / (1/k_max) = {rho_e * k_max:.2f}")
    if rho_e * k_max < 1:
        print(f"    rho_e < 1/k_max: electrons ARE magnetized by self-field")
    else:
        print(f"    rho_e > 1/k_max: electrons are NOT magnetized")
    print()


def exercise_4():
    """
    Exercise 4: Solar Wind Anisotropy
    beta_par = 0.8, beta_perp = 1.5.
    Check firehose and mirror instability.
    """
    print("--- Exercise 4: Solar Wind Anisotropy ---")

    beta_par = 0.8
    beta_perp = 1.5
    v_A = 50e3   # Alfven speed [m/s] = 50 km/s
    L = 1e9      # Scale length [m] = 10^6 km

    print(f"Solar wind at 1 AU:")
    print(f"  beta_par = {beta_par}, beta_perp = {beta_perp}")
    print(f"  v_A = {v_A/1e3:.0f} km/s")

    # (a) Firehose instability check
    # Threshold: beta_par - beta_perp > 2
    firehose_param = beta_par - beta_perp
    firehose_unstable = firehose_param > 2

    print(f"\n(a) Firehose instability:")
    print(f"    Threshold: beta_par - beta_perp > 2")
    print(f"    beta_par - beta_perp = {firehose_param:.1f}")
    print(f"    Firehose unstable? {firehose_unstable}")

    # (b) Mirror instability check
    # Threshold: beta_perp * (beta_perp/beta_par - 1) > 1
    # Equivalently: T_perp/T_par - 1 > 1/beta_perp
    mirror_param = beta_perp * (beta_perp / beta_par - 1)
    mirror_unstable = mirror_param > 1

    print(f"\n(b) Mirror instability:")
    print(f"    Threshold: beta_perp * (beta_perp/beta_par - 1) > 1")
    print(f"    beta_perp * (beta_perp/beta_par - 1) = {mirror_param:.3f}")
    print(f"    Mirror unstable? {mirror_unstable}")

    # (c) Growth rate if unstable
    if mirror_unstable:
        # Mirror growth rate: gamma ~ beta_perp * v_A / L * (mirror_param - 1)
        # More precisely: gamma ~ omega_ci * (mirror_param - 1) / beta_perp
        # Use approximate: gamma ~ k * v_A * sqrt(mirror_param - 1)
        # At the most unstable k ~ 1/rho_i

        omega_ci_approx = v_A / (50e3 / (e * 5e-9 / m_p))  # rough estimate
        # Simpler: gamma_max ~ omega_ci * (mirror_param - 1) / (2 * beta_perp)
        # Let's use dimensional estimate:
        gamma_estimate = v_A / L * np.sqrt(max(0, mirror_param - 1))

        print(f"\n(c) Mirror instability growth rate (estimate):")
        print(f"    gamma ~ v_A/L * sqrt(M - 1) where M = {mirror_param:.3f}")
        print(f"    gamma ~ {gamma_estimate:.4e} rad/s")
        print(f"    Growth time: {1/gamma_estimate:.0f} s = {1/gamma_estimate/3600:.1f} hours")
    elif firehose_unstable:
        gamma_estimate = v_A / L * np.sqrt(max(0, firehose_param - 2))
        print(f"\n(c) Firehose growth rate (estimate):")
        print(f"    gamma ~ {gamma_estimate:.4e} rad/s")
    else:
        print(f"\n(c) Plasma is stable to both instabilities")
        print(f"    Firehose margin: {2 - firehose_param:.1f}")
        print(f"    Mirror margin: {1 - mirror_param:.3f}")

    # (d) Marginal stability mechanism
    print(f"\n(d) Marginal stability in solar wind:")
    print(f"    The observed quasi-steady anisotropy suggests:")
    print(f"    1. Solar wind expansion increases anisotropy (CGL: p_perp/B = const)")
    print(f"    2. When threshold is exceeded, instability generates waves")
    print(f"    3. Waves scatter particles, reducing anisotropy")
    print(f"    4. System relaxes back to near-marginal state")
    print(f"    This 'regulation' keeps beta_perp/beta_par near the stability boundary")
    print(f"    Evidence: satellite data shows clustering along instability thresholds")
    print()


def exercise_5():
    """
    Exercise 5: Laser-Plasma Instabilities in ICF
    I = 3e15 W/cm^2, lambda = 351 nm, n = 0.1*n_c, T_e = 3 keV.
    """
    print("--- Exercise 5: Laser-Plasma Instabilities in ICF ---")

    I = 3e15 * 1e4   # Convert W/cm^2 to W/m^2
    lambda_L = 351e-9  # Laser wavelength [m]
    T_e_eV = 3e3       # 3 keV
    T_e = T_e_eV * eV_to_J

    omega_0 = 2 * np.pi * c / lambda_L  # Laser frequency

    # (a) Critical density
    n_c = epsilon_0 * m_e * omega_0**2 / e**2
    n = 0.1 * n_c

    print(f"(a) Laser parameters:")
    print(f"    lambda = {lambda_L*1e9:.0f} nm, omega_0 = {omega_0:.4e} rad/s")
    print(f"    f_0 = {omega_0/(2*np.pi)/1e14:.2f} x 10^14 Hz")
    print(f"    Critical density: n_c = {n_c:.2e} m^-3")
    print(f"    Plasma density: n = 0.1*n_c = {n:.2e} m^-3")

    # (b) Quiver velocity
    # v_osc = e * E_0 / (m_e * omega_0)
    # E_0 = sqrt(2*I/(c*epsilon_0))
    E_0 = np.sqrt(2 * I / (c * epsilon_0))
    v_osc = e * E_0 / (m_e * omega_0)

    print(f"\n(b) Quiver velocity:")
    print(f"    I = {I/1e4:.0e} W/cm^2")
    print(f"    E_0 = {E_0:.4e} V/m")
    print(f"    v_osc = eE_0/(m_e*omega_0) = {v_osc:.4e} m/s")
    print(f"    v_osc / c = {v_osc/c:.4f}")
    v_th_e = np.sqrt(T_e / m_e)
    print(f"    v_osc / v_th,e = {v_osc/v_th_e:.4f}")

    # (c) SRS growth rate
    # gamma_SRS ~ (k_L * v_osc / 4) * sqrt(omega_0 / omega_s)
    # For backscatter: k_L ~ 2*omega_0/c
    # omega_s = omega_0 - omega_pe (frequency of scattered light)
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    omega_s = omega_0 - omega_pe  # Scattered light frequency
    k_L = 2 * omega_0 / c  # Backscatter wavenumber

    gamma_SRS = (k_L * v_osc / 4) * np.sqrt(omega_0 / omega_s)

    print(f"\n(c) SRS growth rate (backscatter):")
    print(f"    omega_pe = {omega_pe:.4e} rad/s")
    print(f"    omega_pe/omega_0 = {omega_pe/omega_0:.4f} (sqrt(n/n_c) = {np.sqrt(n/n_c):.4f})")
    print(f"    k_L = 2*omega_0/c = {k_L:.4e} m^-1")
    print(f"    gamma_SRS = {gamma_SRS:.4e} rad/s")
    print(f"    gamma_SRS/omega_0 = {gamma_SRS/omega_0:.4e}")

    # (d) Compare to Landau damping
    lambda_D = np.sqrt(epsilon_0 * T_e / (n * e**2))
    k_epw = k_L  # The Langmuir wave has similar k
    k_lD = k_epw * lambda_D

    # Landau damping of EPW (electron plasma wave)
    gamma_L = np.sqrt(np.pi / 8) * omega_pe / k_lD**3 * np.exp(-1 / (2 * k_lD**2) - 1.5)

    print(f"\n(d) Landau damping comparison:")
    print(f"    k_EPW * lambda_D = {k_lD:.4f}")
    print(f"    Landau damping rate: |gamma_L| = {abs(gamma_L):.4e} rad/s")
    print(f"    gamma_SRS / |gamma_L| = {gamma_SRS/abs(gamma_L):.2f}")

    above_threshold = gamma_SRS > abs(gamma_L)
    print(f"    SRS above threshold? {above_threshold}")

    print(f"\n    Strategies to reduce SRS:")
    print(f"    1. Shorter wavelength laser (351 nm already optimal)")
    print(f"    2. Lower intensity (but reduces implosion efficiency)")
    print(f"    3. Beam smoothing (SSD, RPP) to reduce coherence")
    print(f"    4. CBET (cross-beam energy transfer) management")
    print(f"    5. Broadband laser concepts (reduces coherent growth)")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
