"""
Plasma Physics - Lesson 15: Plasma Diagnostics
Exercise Solutions

Topics covered:
- Langmuir probe analysis (Bohm velocity, density, floating potential)
- Thomson scattering spectrum analysis
- Interferometry Abel inversion
- Doppler spectroscopy (rotation, ion temperature, Stark broadening)
- Magnetic diagnostics (Rogowski coil, diamagnetic loop)
"""

import numpy as np
from scipy.optimize import curve_fit

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
m_p = 1.673e-27
epsilon_0 = 8.854e-12
k_B = 1.381e-23
mu_0 = 4 * np.pi * 1e-7
c = 3e8
eV_to_J = e
amu = 1.661e-27  # Atomic mass unit [kg]


def exercise_1():
    """
    Exercise 1: Langmuir Probe in Edge Plasma
    I_sat,i = -5 mA, A_p = 2 mm^2, T_e = 20 eV.
    """
    print("--- Exercise 1: Langmuir Probe Analysis ---")

    I_sat_i = 5e-3      # Ion saturation current magnitude [A]
    A_p = 2e-6           # Probe area [m^2]
    T_e_eV = 20.0        # Electron temperature [eV]
    T_e = T_e_eV * eV_to_J
    m_i = 2 * m_p        # Deuterium

    # (a) Bohm velocity: u_B = sqrt(T_e / m_i)
    u_B = np.sqrt(T_e / m_i)

    print(f"(a) Bohm velocity:")
    print(f"    u_B = sqrt(T_e/m_i) = {u_B:.4e} m/s = {u_B/1e3:.1f} km/s")
    print(f"    (T_e = {T_e_eV} eV, deuterium)")

    # (b) Ion density: I_sat,i = n_i * e * u_B * A_p
    n_i = I_sat_i / (e * u_B * A_p)

    print(f"\n(b) Ion density:")
    print(f"    I_sat,i = n_i * e * u_B * A_p")
    print(f"    n_i = I_sat,i / (e * u_B * A_p)")
    print(f"    n_i = {n_i:.4e} m^-3")
    print(f"    = {n_i/1e18:.2f} x 10^18 m^-3")

    # (c) Floating potential
    # V_f = -(T_e/(2*e)) * ln(2*pi*m_e/m_i)  (relative to plasma potential)
    # For Te = Ti, singly charged:
    V_f = -(T_e_eV / 2) * np.log(2 * np.pi * m_e / m_i)

    print(f"\n(c) Floating potential (relative to plasma potential):")
    print(f"    V_f = -(T_e/2) * ln(2*pi*m_e/m_i)")
    print(f"    V_f = {V_f:.1f} V")
    print(f"    (Floating potential is negative: electrons repelled)")
    print(f"    The factor ln(2*pi*m_e/m_i) = {np.log(2*np.pi*m_e/m_i):.2f}")
    print(f"    For D+: V_f ~ {-T_e_eV * abs(np.log(2*np.pi*m_e/m_i))/2:.1f} V")

    # Additional: Debye length and sheath
    lambda_D = np.sqrt(epsilon_0 * T_e / (n_i * e**2))
    print(f"\n    Debye length: lambda_D = {lambda_D*1e6:.1f} um")
    print(f"    Probe size should be >> lambda_D for valid measurement")
    print(f"    Probe area: sqrt(A_p) ~ {np.sqrt(A_p)*1e3:.2f} mm >> {lambda_D*1e3:.4f} mm: OK")
    print()


def exercise_2():
    """
    Exercise 2: Thomson Scattering Spectrum
    Nd:YAG laser (1064 nm), 90 degree scattering, Gaussian width 2 nm.
    """
    print("--- Exercise 2: Thomson Scattering ---")

    lambda_0 = 1064e-9   # Laser wavelength [m]
    theta = 90.0          # Scattering angle [degrees]
    Delta_lambda = 2e-9   # Gaussian width [m]
    P_s = 1e-9            # Scattered power [W]
    P_i = 1.0             # Incident energy [J]
    Delta_V = 1e-9        # Scattering volume [m^3]
    Delta_Omega = 0.01    # Collection solid angle [sr]
    sigma_T = 6.652e-29   # Thomson cross section [m^2]

    # (a) Electron temperature from spectral width
    # For 90 degree scattering:
    # Delta_lambda / lambda_0 = (v_th/c) * sin(theta/2) * 2
    # More precisely: Delta_lambda = lambda_0 * sqrt(2*T_e/(m_e*c^2)) * 2*sin(theta/2)
    # Spectral width: Delta_lambda = lambda_0 * sqrt(8*T_e*sin^2(theta/2)/(m_e*c^2))

    theta_rad = np.radians(theta)
    # Delta_lambda / lambda_0 = sqrt(8*T_e*sin^2(theta/2)/(m_e*c^2))
    # T_e = m_e * c^2 * (Delta_lambda/lambda_0)^2 / (8*sin^2(theta/2))

    T_e = m_e * c**2 * (Delta_lambda / lambda_0)**2 / (8 * np.sin(theta_rad / 2)**2)
    T_e_eV = T_e / eV_to_J

    print(f"(a) Temperature from Thomson scattering spectrum:")
    print(f"    lambda_0 = {lambda_0*1e9:.0f} nm, theta = {theta}°")
    print(f"    Delta_lambda = {Delta_lambda*1e9:.0f} nm")
    print(f"    Delta_lambda/lambda_0 = {Delta_lambda/lambda_0:.4e}")
    print(f"    T_e = m_e*c^2*(Delta_lambda/lambda_0)^2 / (8*sin^2(theta/2))")
    print(f"    T_e = {T_e_eV:.1f} eV = {T_e_eV/1e3:.3f} keV")

    # (b) Electron density from scattered power
    # P_s = n_e * sigma_T * P_i * Delta_V * Delta_Omega / (4*pi)
    # (for incoherent scattering, alpha << 1)
    n_e = P_s * 4 * np.pi / (sigma_T * P_i * Delta_V * Delta_Omega)

    print(f"\n(b) Density from scattered power:")
    print(f"    P_s = {P_s:.0e} W, P_i = {P_i} J")
    print(f"    Delta_V = {Delta_V*1e9:.0f} mm^3, Delta_Omega = {Delta_Omega} sr")
    print(f"    sigma_T = {sigma_T:.3e} m^2")
    print(f"    n_e = P_s * 4*pi / (sigma_T * P_i * DeltaV * DeltaOmega)")
    print(f"    n_e = {n_e:.4e} m^-3")

    # Scattering parameter
    lambda_D = np.sqrt(epsilon_0 * T_e / (n_e * e**2))
    k_s = 4 * np.pi * np.sin(theta_rad / 2) / lambda_0  # Scattering wave vector
    alpha = 1 / (k_s * lambda_D)

    print(f"\n    Scattering parameter: alpha = 1/(k_s*lambda_D) = {alpha:.3f}")
    print(f"    alpha << 1: incoherent scattering (individual electrons)")
    print(f"    alpha >> 1: collective scattering (density fluctuations)")
    print()


def exercise_3():
    """
    Exercise 3: Interferometry Abel Inversion
    5 chords through cylindrical plasma, radius a = 10 cm.
    """
    print("--- Exercise 3: Interferometry Abel Inversion ---")

    a = 0.10  # Plasma radius [m]

    # Data: impact parameter r [cm], line integral [10^18 m^-2]
    r_data = np.array([0, 3, 5, 7, 9]) * 1e-2  # Convert to m
    NL_data = np.array([10.0, 9.5, 8.0, 5.0, 2.0]) * 1e18  # m^-2

    print(f"Data: cylindrical plasma, a = {a*100:.0f} cm")
    print(f"  {'r [cm]':>8} {'NL [10^18 m^-2]':>18}")
    print("  " + "-" * 30)
    for r, NL in zip(r_data, NL_data):
        print(f"  {r*100:>8.0f} {NL/1e18:>18.1f}")

    # (a) Fit parabolic profile: n_e(r) = n_0 * (1 - r^2/a^2)^alpha
    # Line integral: NL(p) = 2 * integral_p^a n_e(r) * r*dr / sqrt(r^2 - p^2)
    # For parabolic profile: NL(p) = 2*n_0 * integral_p^a (1-r^2/a^2)^alpha * r/sqrt(r^2-p^2) dr
    # For alpha = 1: NL(p) = (pi/2) * n_0 * a * (1 - p^2/a^2)

    # Let's try alpha = 1 first
    # NL(0) = (pi/2) * n_0 * a -> n_0 = 2*NL(0) / (pi*a)
    from scipy.integrate import quad

    def line_integral(p, n_0, alpha, a_val):
        """Calculate line integral through parabolic profile."""
        def integrand(r):
            if r <= abs(p) + 1e-10:
                return 0
            return n_0 * (1 - r**2 / a_val**2)**alpha * r / np.sqrt(r**2 - p**2)
        if abs(p) >= a_val:
            return 0
        result, _ = quad(integrand, abs(p) + 1e-10, a_val * 0.999)
        return 2 * result

    # Fit n_0 and alpha
    def model(r_arr, n_0, alpha):
        return np.array([line_integral(p, n_0, alpha, a) for p in r_arr])

    # Initial guess
    n_0_guess = 2 * NL_data[0] / (np.pi * a)
    alpha_guess = 1.0

    try:
        popt, pcov = curve_fit(model, r_data, NL_data, p0=[n_0_guess, alpha_guess],
                               bounds=([0, 0.1], [1e21, 5.0]))
        n_0_fit, alpha_fit = popt
    except Exception:
        # Fallback: simple estimate
        n_0_fit = n_0_guess
        alpha_fit = 1.0

    print(f"\n(a) Fitting n_e(r) = n_0*(1 - r^2/a^2)^alpha:")
    print(f"    n_0 = {n_0_fit:.2e} m^-3")
    print(f"    alpha = {alpha_fit:.2f}")

    # Verify fit
    print(f"\n    Fit quality:")
    print(f"    {'r [cm]':>8} {'NL_data':>12} {'NL_fit':>12} {'Residual':>12}")
    print("    " + "-" * 48)
    for r, NL in zip(r_data, NL_data):
        NL_fit = line_integral(r, n_0_fit, alpha_fit, a)
        print(f"    {r*100:>8.0f} {NL/1e18:>12.1f} {NL_fit/1e18:>12.2f} {(NL-NL_fit)/1e18:>12.2f}")

    # (b) Abel inversion reconstruction
    print(f"\n(b) Reconstructed density profile:")
    r_eval = np.array([0, 0.05, 0.10])  # r = 0, 5, 10 cm
    print(f"    {'r [cm]':>8} {'n_e [10^18 m^-3]':>20}")
    print("    " + "-" * 32)
    for r in r_eval:
        if r < a:
            n_e_r = n_0_fit * (1 - r**2 / a**2)**alpha_fit
        else:
            n_e_r = 0
        print(f"    {r*100:>8.0f} {n_e_r/1e18:>20.2f}")
    print()


def exercise_4():
    """
    Exercise 4: Doppler Spectroscopy
    C6+ line at 529.0 nm. Peak at 529.05 nm, FWHM = 0.02 nm.
    """
    print("--- Exercise 4: Doppler Spectroscopy ---")

    lambda_0 = 529.0e-9   # Rest wavelength [m]
    lambda_peak = 529.05e-9  # Observed peak [m]
    FWHM = 0.02e-9        # FWHM of line [m]
    m_C = 12 * amu         # Carbon mass [kg]

    # (a) Toroidal rotation velocity
    # Doppler shift: delta_lambda / lambda_0 = v / c
    delta_lambda = lambda_peak - lambda_0
    v_rot = c * delta_lambda / lambda_0

    print(f"(a) Toroidal rotation velocity:")
    print(f"    lambda_0 = {lambda_0*1e9:.1f} nm")
    print(f"    lambda_peak = {lambda_peak*1e9:.2f} nm")
    print(f"    delta_lambda = {delta_lambda*1e9:.2f} nm")
    print(f"    v_rot = c * delta_lambda / lambda_0 = {v_rot/1e3:.1f} km/s")

    # (b) Ion temperature from Doppler broadening
    # FWHM = lambda_0 * sqrt(8*ln(2)*T_i/(m*c^2))
    # T_i = m*c^2 * (FWHM/lambda_0)^2 / (8*ln(2))

    T_i = m_C * c**2 * (FWHM / lambda_0)**2 / (8 * np.log(2))
    T_i_eV = T_i / eV_to_J
    T_i_keV = T_i_eV / 1e3

    print(f"\n(b) Ion temperature from Doppler broadening:")
    print(f"    FWHM = {FWHM*1e9:.3f} nm")
    print(f"    T_i = m_C*c^2*(FWHM/lambda_0)^2 / (8*ln(2))")
    print(f"    T_i = {T_i_eV:.0f} eV = {T_i_keV:.2f} keV")

    # (c) Stark broadening correction
    Stark_width = 0.005e-9  # Stark contribution [m]

    # Total broadening: FWHM_total^2 = FWHM_Doppler^2 + FWHM_Stark^2
    # (for Gaussian convolution; for Lorentzian Stark + Gaussian Doppler, use Voigt profile)
    # Simple quadrature deconvolution:
    FWHM_Doppler = np.sqrt(FWHM**2 - Stark_width**2)

    T_i_corrected = m_C * c**2 * (FWHM_Doppler / lambda_0)**2 / (8 * np.log(2))
    T_i_corr_eV = T_i_corrected / eV_to_J

    print(f"\n(c) Correcting for Stark broadening:")
    print(f"    Stark width: {Stark_width*1e9:.3f} nm")
    print(f"    FWHM_Doppler = sqrt(FWHM^2 - Stark^2) = {FWHM_Doppler*1e9:.4f} nm")
    print(f"    Corrected T_i = {T_i_corr_eV:.0f} eV = {T_i_corr_eV/1e3:.2f} keV")
    print(f"    Correction: {(T_i_eV - T_i_corr_eV)/T_i_eV*100:.1f}% reduction")

    print(f"\n    Note: C6+ is used as impurity diagnostic")
    print(f"    Assumes C6+ has same temperature and rotation as main ions")
    print(f"    Valid if carbon equilibration time << confinement time")
    print()


def exercise_5():
    """
    Exercise 5: Magnetic Diagnostics
    Rogowski coil: N = 1000 turns, A = 1 cm^2, V_coil = -50 mV during 1 s ramp.
    """
    print("--- Exercise 5: Magnetic Diagnostics ---")

    N = 1000         # Number of turns
    A = 1e-4         # Cross-sectional area [m^2]
    V_coil = -50e-3  # Induced voltage [V]
    dt = 1.0         # Ramp duration [s]

    # (a) Rate of change of plasma current
    # V_coil = -N * A * mu_0 * dI_p/dt / (2*pi*R_coil)
    # For a Rogowski coil wrapped around the plasma:
    # V_coil = -mu_0 * N * A / (2*pi*R) * dI_p/dt
    # But more simply: V = M * dI_p/dt where M = mu_0 * N * A / l
    # For a coil of length l encircling the plasma:
    # Simplification: V = mu_0 * n_coil * A * dI_p/dt
    # where n_coil = N / (2*pi*R)

    # Using Ampere's law directly:
    # integral B . dl = mu_0 * I_p
    # Each turn links flux: B * A = mu_0 * I_p * A / (2*pi*R)
    # Total flux: Phi = N * mu_0 * A * I_p / (2*pi*R)
    # V = -dPhi/dt = -N * mu_0 * A / (2*pi*R) * dI_p/dt

    # For simplicity, use the mutual inductance form:
    # V = M * dI/dt, M = mu_0 * N * A / (2*pi*R_torus)
    # Let R_torus = 1 m (typical)
    R_torus = 1.0  # Assume 1 m major radius

    M = mu_0 * N * A / (2 * np.pi * R_torus)
    dIdt = abs(V_coil) / M

    print(f"(a) Rate of change of plasma current:")
    print(f"    Rogowski coil: N = {N}, A = {A*1e4:.0f} cm^2")
    print(f"    V_coil = {V_coil*1e3:.0f} mV")
    print(f"    Mutual inductance: M = mu_0*N*A/(2*pi*R) = {M:.4e} H")
    print(f"    dI_p/dt = |V_coil| / M = {dIdt:.4e} A/s = {dIdt/1e6:.2f} MA/s")

    # (b) Final plasma current
    I_p = dIdt * dt

    print(f"\n(b) Plasma current after {dt} s ramp (constant dI/dt):")
    print(f"    I_p = dI/dt * dt = {I_p:.4e} A = {I_p/1e6:.2f} MA")

    # (c) Diamagnetic loop - stored energy
    W = 10e6         # Stored energy [J] = 10 MJ
    V_plasma = 100.0  # Plasma volume [m^3]

    # W = (3/2) * <p> * V  (for isotropic pressure)
    # <p> = (2/3) * W / V
    p_avg = (2.0 / 3.0) * W / V_plasma

    print(f"\n(c) Diamagnetic loop measurement:")
    print(f"    Stored energy: W = {W/1e6:.0f} MJ")
    print(f"    Plasma volume: V = {V_plasma} m^3")
    print(f"    Average pressure: <p> = (2/3)*W/V = {p_avg:.4e} Pa")
    print(f"    <p> = {p_avg/1e3:.1f} kPa")

    # Convert to beta
    B_t = 5.0  # Typical toroidal field
    beta = 2 * mu_0 * p_avg / B_t**2
    print(f"    For B = {B_t} T: beta = 2*mu_0*<p>/B^2 = {beta:.4f} = {beta*100:.2f}%")

    # Additional info
    print(f"\n    Diagnostic summary:")
    print(f"    - Rogowski coil: measures dI_p/dt (need integrator for I_p)")
    print(f"    - Diamagnetic loop: measures change in toroidal flux")
    print(f"      -> relates to stored energy W and beta")
    print(f"    - Mirnov coils: measure magnetic fluctuations (MHD modes)")
    print(f"    - Equilibrium: need Rogowski + diamagnetic + external coils")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
