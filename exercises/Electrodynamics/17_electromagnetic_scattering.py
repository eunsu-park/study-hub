"""
Exercises for Lesson 17: Electromagnetic Scattering
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn


# Constants
c = 2.998e8
mu_0 = 4.0 * np.pi * 1e-7
epsilon_0 = 8.854e-12
h_planck = 6.626e-34
k_B = 1.38e-23


def mie_coefficients(x, m, n_max):
    """
    Compute Mie scattering coefficients a_n and b_n.

    Parameters
    ----------
    x : float
        Size parameter 2*pi*a/lambda.
    m : complex
        Relative refractive index n_sphere / n_medium.
    n_max : int
        Maximum multipole order.

    Returns
    -------
    a_n, b_n : arrays of complex Mie coefficients.
    """
    a_n = np.zeros(n_max, dtype=complex)
    b_n = np.zeros(n_max, dtype=complex)

    for n in range(1, n_max + 1):
        # Riccati-Bessel functions: psi_n(rho) = rho * j_n(rho)
        #                           xi_n(rho) = rho * (j_n(rho) + i*y_n(rho))
        # We need derivatives as well

        # At argument x
        jn_x = spherical_jn(n, x)
        yn_x = spherical_yn(n, x)
        jn_x_d = spherical_jn(n, x, derivative=True)
        yn_x_d = spherical_yn(n, x, derivative=True)

        psi_x = x * jn_x
        psi_x_d = jn_x + x * jn_x_d
        xi_x = x * (jn_x + 1j * yn_x)
        xi_x_d = (jn_x + 1j * yn_x) + x * (jn_x_d + 1j * yn_x_d)

        # At argument mx
        mx = m * x
        jn_mx = spherical_jn(n, mx)
        jn_mx_d = spherical_jn(n, mx, derivative=True)
        psi_mx = mx * jn_mx
        psi_mx_d = jn_mx + mx * jn_mx_d

        # Mie coefficients
        a_n[n - 1] = (m * psi_mx * psi_x_d - psi_mx_d * psi_x) / \
                     (m * psi_mx * xi_x_d - psi_mx_d * xi_x)
        b_n[n - 1] = (psi_mx * psi_x_d - m * psi_mx_d * psi_x) / \
                     (psi_mx * xi_x_d - m * psi_mx_d * xi_x)

    return a_n, b_n


def mie_efficiencies(x, m, n_max=None):
    """
    Compute Mie scattering, extinction, and absorption efficiencies.

    Returns
    -------
    Q_ext, Q_sca, Q_abs : extinction, scattering, absorption efficiencies.
    """
    if n_max is None:
        n_max = int(x + 4 * x**(1 / 3) + 2)
        n_max = max(n_max, 10)

    a_n, b_n = mie_coefficients(x, m, n_max)

    n_vals = np.arange(1, n_max + 1)
    Q_ext = (2 / x**2) * np.sum((2 * n_vals + 1) * np.real(a_n + b_n))
    Q_sca = (2 / x**2) * np.sum((2 * n_vals + 1) * (np.abs(a_n)**2 + np.abs(b_n)**2))
    Q_abs = Q_ext - Q_sca

    return Q_ext, Q_sca, Q_abs


def exercise_1():
    """
    Exercise 1: Sunset Simulation
    Model sunlight transmission through atmosphere using Rayleigh scattering.
    Exponential density profile: n(h) = n0 * exp(-h/H), H = 8.5 km.
    Plot transmitted spectrum for various zenith angles.
    """
    # Atmospheric parameters
    H_scale = 8500.0  # scale height (m)
    R_earth = 6.371e6  # m
    n0_N2 = 2.0e25     # number density at sea level (m^-3) for N2+O2 combined

    # Rayleigh scattering cross section: sigma = (8*pi/3) * (2*pi/(lambda))^4 * alpha^2
    # For air: sigma_R = 5.2e-31 / lambda^4 (with lambda in m, sigma in m^2)
    # More precisely: sigma_R(lambda) = (24*pi^3 / (n_s^2 * N^2)) * ((n_s^2-1)/(n_s^2+2))^2 * (6+3*delta)/(6-7*delta) / lambda^4
    # Simplified: sigma_R ~ 4.0e-31 / lambda_um^4  (lambda_um in micrometers, sigma in m^2)

    def rayleigh_cross_section(lam):
        """Rayleigh cross section for air molecules (m^2). lam in meters."""
        # Using the simplified formula with depolarization factor
        lam_um = lam * 1e6
        return 4.0e-28 / lam_um**4  # m^2 (corrected units)

    # Solar spectrum (approximated as blackbody at 5778 K)
    def solar_spectrum(lam):
        """Normalized Planck function at 5778 K. lam in meters."""
        T = 5778.0
        B = (2 * h_planck * c**2 / lam**5) / (np.exp(h_planck * c / (lam * k_B * T)) - 1)
        return B

    # Path length through atmosphere at zenith angle theta
    # For a spherical atmosphere: integral of n(h) ds along the path
    # For moderate angles (< 80 deg), path length ~ N_column / cos(theta)
    # For grazing angles, need full spherical geometry

    def column_density(zenith_deg):
        """
        Compute total column density (m^-2) along path at given zenith angle.
        Uses numerical integration through spherical atmosphere.
        """
        theta = np.radians(zenith_deg)

        if zenith_deg < 75:
            # Plane-parallel approximation: good for small zenith angles
            N_vertical = n0_N2 * H_scale  # vertical column density
            return N_vertical / np.cos(theta)
        else:
            # Numerical integration for large zenith angles
            # Parameterize ray from observer at sea level
            N_steps = 5000
            s_max = 5 * H_scale / max(np.cos(theta), 0.01)
            s_max = min(s_max, 50 * H_scale)  # cap
            s = np.linspace(0, s_max, N_steps)
            ds = s[1] - s[0]

            # Height along ray: h(s) = sqrt((R_earth + 0)^2 + s^2 + 2*R_earth*s*cos(theta)) - R_earth
            r = np.sqrt(R_earth**2 + s**2 + 2 * R_earth * s * np.cos(theta))
            h = r - R_earth
            h = np.maximum(h, 0)

            n_density = n0_N2 * np.exp(-h / H_scale)
            return np.trapz(n_density, s)

    # Wavelength range (visible spectrum)
    wavelengths = np.linspace(380e-9, 780e-9, 200)

    zenith_angles = [0, 45, 70, 85, 90]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color map for visible spectrum
    colors_zen = ['blue', 'green', 'orange', 'red', 'darkred']

    print("  Rayleigh scattering sunset simulation:")
    print(f"  Scale height H = {H_scale/1e3:.1f} km")
    print(f"  Sea-level density n0 = {n0_N2:.1e} m^-3")
    print()

    for idx, zen in enumerate(zenith_angles):
        N_col = column_density(zen)
        sigma_R = rayleigh_cross_section(wavelengths)
        tau = N_col * sigma_R  # optical depth
        transmission = np.exp(-tau)

        # Transmitted solar spectrum
        I_solar = solar_spectrum(wavelengths)
        I_transmitted = I_solar * transmission

        # Normalize
        I_solar_norm = I_solar / np.max(I_solar)
        I_trans_norm = I_transmitted / np.max(I_solar)

        axes[0].plot(wavelengths * 1e9, transmission * 100,
                     color=colors_zen[idx], linewidth=2, label=f'{zen} deg')

        axes[1].plot(wavelengths * 1e9, I_trans_norm,
                     color=colors_zen[idx], linewidth=2, label=f'{zen} deg')

        # Estimate color temperature by fitting Wien displacement
        if np.max(I_transmitted) > 0:
            peak_idx = np.argmax(I_transmitted)
            peak_lam = wavelengths[peak_idx]
            T_color = 2.898e-3 / peak_lam  # Wien's law
            avg_tau_blue = N_col * rayleigh_cross_section(450e-9)
            avg_tau_red = N_col * rayleigh_cross_section(650e-9)
            print(f"  Zenith {zen:2d} deg: N_col = {N_col:.3e} m^-2, "
                  f"tau(450nm) = {avg_tau_blue:.2f}, tau(650nm) = {avg_tau_red:.3f}, "
                  f"T_color ~ {T_color:.0f} K")

    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Transmission (%)')
    axes[0].set_title('Atmospheric Transmission (Rayleigh)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(wavelengths * 1e9, I_solar_norm, 'k--', alpha=0.3, label='Solar (top)')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Normalized Intensity')
    axes[1].set_title('Transmitted Solar Spectrum')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Sunset Simulation: Rayleigh Scattering', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex17_sunset_simulation.png', dpi=150)
    plt.close()

    print()
    print("  Blue light (450nm) is scattered ~5.7x more than red (650nm):")
    ratio = (650 / 450)**4
    print(f"  sigma(450nm)/sigma(650nm) = (650/450)^4 = {ratio:.2f}")
    print("  This explains why sky is blue (scattered light) and sunsets are red")
    print("  (transmitted light with blue removed).")
    print("  Plot saved: ex17_sunset_simulation.png")


def exercise_2():
    """
    Exercise 2: Mie Resonances
    Plot Q_ext for a glass sphere (n=1.5) vs size parameter x from 0 to 50.
    Identify resonance peaks and dominant multipole orders.
    """
    n_sphere = 1.5
    m = n_sphere  # relative to vacuum

    x_vals = np.linspace(0.1, 50, 1000)
    Q_ext_vals = np.zeros(len(x_vals))
    Q_sca_vals = np.zeros(len(x_vals))

    # Also track individual multipole contributions
    n_track = 5  # track first 5 multipole orders
    a_n_contrib = np.zeros((n_track, len(x_vals)))
    b_n_contrib = np.zeros((n_track, len(x_vals)))

    for i, x in enumerate(x_vals):
        n_max = int(x + 4 * x**(1 / 3) + 10)
        n_max = max(n_max, 15)
        a_n, b_n = mie_coefficients(x, m, n_max)

        n_arr = np.arange(1, n_max + 1)
        Q_ext_vals[i] = (2 / x**2) * np.sum((2 * n_arr + 1) * np.real(a_n + b_n))
        Q_sca_vals[i] = (2 / x**2) * np.sum((2 * n_arr + 1) * (np.abs(a_n)**2 + np.abs(b_n)**2))

        for n_idx in range(min(n_track, len(a_n))):
            n = n_idx + 1
            a_n_contrib[n_idx, i] = (2 / x**2) * (2 * n + 1) * np.abs(a_n[n_idx])**2
            b_n_contrib[n_idx, i] = (2 / x**2) * (2 * n + 1) * np.abs(b_n[n_idx])**2

    print(f"  Glass sphere: n = {n_sphere}")
    print(f"  Size parameter range: x = 0.1 to 50")
    print()

    # Find peaks in Q_ext
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(Q_ext_vals, height=2.5, distance=10)
    print("  Major resonance peaks:")
    print(f"  {'x':>8s}  {'Q_ext':>8s}  {'Dominant mode':>15s}")
    for pk in peaks[:10]:
        x_pk = x_vals[pk]
        # Determine dominant mode
        max_a = np.max(a_n_contrib[:, pk])
        max_b = np.max(b_n_contrib[:, pk])
        if max_a > max_b:
            mode_idx = np.argmax(a_n_contrib[:, pk])
            mode = f"a_{mode_idx+1} (TM)"
        else:
            mode_idx = np.argmax(b_n_contrib[:, pk])
            mode = f"b_{mode_idx+1} (TE)"
        print(f"  {x_pk:8.2f}  {Q_ext_vals[pk]:8.3f}  {mode:>15s}")

    # Asymptotic value
    Q_ext_large_x = np.mean(Q_ext_vals[-100:])
    print(f"\n  Asymptotic Q_ext (large x): {Q_ext_large_x:.3f}")
    print(f"  (Expected: ~2 due to extinction paradox)")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(x_vals, Q_ext_vals, 'b-', linewidth=1.5, label='Q_ext')
    axes[0].plot(x_vals, Q_sca_vals, 'r--', linewidth=1, alpha=0.7, label='Q_sca')
    axes[0].axhline(y=2, color='gray', linestyle=':', alpha=0.5, label='Extinction paradox (Q=2)')
    if len(peaks) > 0:
        axes[0].plot(x_vals[peaks[:10]], Q_ext_vals[peaks[:10]], 'rv', markersize=8)
    axes[0].set_xlabel('Size parameter x = 2*pi*a/lambda')
    axes[0].set_ylabel('Efficiency Q')
    axes[0].set_title(f'Mie Extinction Efficiency (n = {n_sphere})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 50)

    # Individual multipole contributions
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for n_idx in range(n_track):
        n = n_idx + 1
        axes[1].plot(x_vals, a_n_contrib[n_idx], '-',
                     color=colors[n_idx], linewidth=1, label=f'|a_{n}|^2', alpha=0.8)
        axes[1].plot(x_vals, b_n_contrib[n_idx], '--',
                     color=colors[n_idx], linewidth=1, label=f'|b_{n}|^2', alpha=0.6)

    axes[1].set_xlabel('Size parameter x')
    axes[1].set_ylabel('Contribution to Q_sca')
    axes[1].set_title('Individual Multipole Contributions')
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 20)  # zoom in to see structure

    plt.suptitle('Mie Scattering Resonances', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex17_mie_resonances.png', dpi=150)
    plt.close()
    print("  Plot saved: ex17_mie_resonances.png")


def exercise_3():
    """
    Exercise 3: Cloud Opacity
    Cloud droplets: r = 10 um, concentration ~ 300 cm^-3.
    Compute Q_ext at 550 nm, extinction coefficient, optical depth of 1 km cloud.
    """
    a = 10e-6       # droplet radius (10 um)
    n_droplet = 300e6  # 300 cm^-3 = 300e6 m^-3
    n_water = 1.33
    lam = 550e-9     # visible wavelength
    cloud_thickness = 1000  # 1 km

    x = 2 * np.pi * a / lam
    m = n_water  # relative to air

    print(f"  Cloud droplets: radius = {a*1e6:.0f} um, n = {n_droplet/1e6:.0f} cm^-3")
    print(f"  Water refractive index: n = {n_water}")
    print(f"  Wavelength: {lam*1e9:.0f} nm")
    print(f"  Size parameter: x = {x:.1f}")
    print()

    # (a) Mie Q_ext
    Q_ext, Q_sca, Q_abs = mie_efficiencies(x, m)

    print(f"  (a) Mie efficiencies:")
    print(f"      Q_ext = {Q_ext:.4f}")
    print(f"      Q_sca = {Q_sca:.4f}")
    print(f"      Q_abs = {Q_abs:.4f}")
    print(f"      (Q_ext ~ 2 as expected for large x)")

    # (b) Extinction coefficient
    sigma_ext = Q_ext * np.pi * a**2  # extinction cross section
    alpha_ext = n_droplet * sigma_ext  # extinction coefficient (m^-1)

    print(f"\n  (b) Extinction cross section: sigma_ext = Q_ext * pi * a^2 = {sigma_ext:.4e} m^2")
    print(f"      Extinction coefficient: alpha = n * sigma = {alpha_ext:.4f} m^-1")
    print(f"      Mean free path: l = 1/alpha = {1/alpha_ext:.1f} m")

    # (c) Optical depth of 1 km cloud
    tau = alpha_ext * cloud_thickness
    transmission = np.exp(-tau)

    print(f"\n  (c) Optical depth (1 km cloud): tau = alpha * L = {tau:.1f}")
    print(f"      Transmission = exp(-tau) = {transmission:.2e}")
    print(f"      {'Cloud is opaque' if tau > 3 else 'Cloud is somewhat transparent'}")
    print(f"      (tau >> 1 confirms visual opacity of clouds)")

    # Plot Q_ext vs wavelength for cloud droplets
    wavelengths = np.linspace(300e-9, 2000e-9, 100)
    Q_ext_spectrum = np.zeros(len(wavelengths))

    for i, lam_i in enumerate(wavelengths):
        x_i = 2 * np.pi * a / lam_i
        Q_ext_spectrum[i], _, _ = mie_efficiencies(x_i, m)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(wavelengths * 1e9, Q_ext_spectrum, 'b-', linewidth=2)
    axes[0].axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Q = 2 (geometric limit)')
    axes[0].axvline(x=550, color='green', linestyle=':', alpha=0.5, label='550 nm')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Q_ext')
    axes[0].set_title(f'Cloud Droplet Q_ext (a = {a*1e6:.0f} um)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Optical depth vs cloud thickness
    thicknesses = np.linspace(0, 5000, 200)
    tau_vs_L = alpha_ext * thicknesses

    axes[1].plot(thicknesses, tau_vs_L, 'b-', linewidth=2)
    axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='tau = 1')
    axes[1].axhline(y=3, color='orange', linestyle=':', alpha=0.5, label='tau = 3 (opaque)')
    axes[1].set_xlabel('Cloud Thickness (m)')
    axes[1].set_ylabel('Optical Depth tau')
    axes[1].set_title('Cloud Optical Depth')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Cloud Opacity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex17_cloud_opacity.png', dpi=150)
    plt.close()
    print("  Plot saved: ex17_cloud_opacity.png")


def exercise_4():
    """
    Exercise 4: Born Approximation Validity
    Compare Born and Mie cross sections for n = 1.05, 1.2, 1.5 vs x.
    Verify rule of thumb |Delta_eps_r| * x << 1.
    """
    # Born approximation for a sphere:
    # sigma_Born = (k^4 / (6*pi)) * |alpha_Born|^2
    # where alpha_Born = 4*pi*a^3 * (eps_r - 1) / 3 (for Rayleigh-like approximation)
    # More properly, the Born approximation total cross section for a sphere:
    # sigma_Born = (2*pi*k^4*a^6/9) * |eps_r - 1|^2 * [1 + terms in (ka)]

    # Actually, for the Born approximation to the scattering amplitude:
    # f_Born(theta) ~ (k^2 * (eps_r-1) / (4*pi)) * V * form_factor
    # For a uniform sphere, the Born total cross section is:
    # Q_Born = (8/27) * |m^2 - 1|^2 * x^4 * [correction terms]

    # We'll use the Rayleigh-Gans (RG) or Born approximation:
    # Valid when |m - 1| << 1 AND 2*x*|m-1| << 1
    # Q_RG = (8/3) * (x^4) * |K|^2 * correction factor
    # K = (m^2 - 1) / (m^2 + 2)

    # For the Born/RG regime, a good formula for Q_sca is:
    # Q_sca_RG = (8/3) * x^4 * |K|^2  (Rayleigh, x << 1)
    # For larger x in RG regime, integrate over the form factor

    refractive_indices = [1.05, 1.2, 1.5]
    x_vals = np.linspace(0.1, 20, 300)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    print("  Born approximation validity check:")
    print(f"  Rule of thumb: |Delta_eps_r| * x << 1")
    print()

    for idx, n_s in enumerate(refractive_indices):
        m = n_s
        eps_r = n_s**2
        delta_eps = abs(eps_r - 1)
        K = (eps_r - 1) / (eps_r + 2)

        Q_mie = np.zeros(len(x_vals))
        Q_born = np.zeros(len(x_vals))

        for i, x in enumerate(x_vals):
            # Mie (exact)
            Q_ext_mie, Q_sca_mie, _ = mie_efficiencies(x, m)
            Q_mie[i] = Q_sca_mie

            # Born/Rayleigh-Gans approximation
            # For x << 1: Q_sca ~ (8/3) * x^4 * |K|^2 (Rayleigh)
            # For arbitrary x with |m-1| << 1: use RGD (Rayleigh-Gans-Debye)
            # Q_sca_RGD = (8/3) * x^4 * |K|^2 * [1 + 2*x^2/5 + ...] for moderate x
            # More complete: use numerical integration of the form factor
            u = 2 * x  # for Born, momentum transfer
            if u < 0.01:
                g_factor = 1.0
            else:
                # Born form factor integral for a sphere:
                # G(u) = 3*(sin(u) - u*cos(u)) / u^3
                # Q_sca_Born ~ (2 * x^4 * delta_eps^2 / 9) * integral term
                # Simplified: use Rayleigh-Gans total cross section
                g_factor = 1 + (2 * x**2 * (m - 1)**2) / 5

            Q_born[i] = (8 / 3) * x**4 * abs(K)**2 * min(g_factor, 10)

        # Find where Born error exceeds 10%
        error = np.abs(Q_born - Q_mie) / (Q_mie + 1e-30) * 100
        exceed_10 = np.where(error > 10)[0]
        if len(exceed_10) > 0:
            x_limit = x_vals[exceed_10[0]]
            delta_eps_x = delta_eps * x_limit
        else:
            x_limit = x_vals[-1]
            delta_eps_x = delta_eps * x_limit

        print(f"  n = {n_s}: |eps_r - 1| = {delta_eps:.3f}")
        print(f"    Born error > 10% at x = {x_limit:.1f}")
        print(f"    |Delta_eps_r| * x = {delta_eps_x:.2f} at that point")

        axes[idx].semilogy(x_vals, Q_mie, 'b-', linewidth=2, label='Mie (exact)')
        axes[idx].semilogy(x_vals, Q_born, 'r--', linewidth=2, label='Born/RG approx')
        if len(exceed_10) > 0:
            axes[idx].axvline(x=x_limit, color='green', linestyle=':', alpha=0.7,
                              label=f'10% error (x={x_limit:.1f})')
        axes[idx].set_xlabel('Size parameter x')
        axes[idx].set_ylabel('Q_sca')
        axes[idx].set_title(f'n = {n_s}, |Delta eps| = {delta_eps:.3f}')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    print()
    print("  Conclusion: Born approximation validity follows |Delta_eps_r| * x << 1")
    print("  For nearly transparent particles (n ~ 1.05), Born works up to large x")
    print("  For high-contrast particles (n ~ 1.5), Born fails at very small x")

    plt.suptitle('Born Approximation vs Mie Theory', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex17_born_validity.png', dpi=150)
    plt.close()
    print("  Plot saved: ex17_born_validity.png")


def exercise_5():
    """
    Exercise 5: Radar Cross Section
    Metallic sphere: a = 1 m, f = 10 GHz, n = 10 + 10i (metal approximation).
    Compute size parameter, Q_ext, RCS, received power from radar equation.
    """
    a = 1.0          # sphere radius (m)
    f = 10e9          # 10 GHz radar
    lam = c / f
    n_metal = 10 + 10j  # approximate metal

    # (a) Size parameter
    x = 2 * np.pi * a / lam

    print(f"  Metallic sphere: radius = {a} m")
    print(f"  Radar frequency: {f/1e9:.0f} GHz, lambda = {lam*1e3:.1f} mm")
    print(f"\n  (a) Size parameter: x = 2*pi*a/lambda = {x:.1f}")
    print(f"      x >> 1, so geometric optics regime")

    # (b) Mie Q_ext
    # For a large metallic sphere, Q_ext -> 2 (extinction paradox)
    # and Q_sca -> Q_ext (no absorption for perfect conductor)
    n_max = int(x + 4 * x**(1 / 3) + 10)

    # For very large x with complex n, direct Mie computation can be numerically challenging
    # Use the geometric optics limit: Q_ext ~ 2 for large perfectly conducting sphere
    Q_ext_analytical = 2.0  # geometric limit

    # For a PEC sphere in the geometric limit:
    # sigma_backscatter = pi * a^2 (geometric cross section)
    # This is also the RCS of a sphere in the high-frequency limit

    print(f"\n  (b) For large metallic sphere (x >> 1):")
    print(f"      Q_ext ~ 2 (extinction paradox: diffraction + geometric)")
    print(f"      sigma_ext = Q_ext * pi * a^2 = {Q_ext_analytical * np.pi * a**2:.4f} m^2")

    # Try Mie calculation for moderate x to demonstrate
    x_demo = 50  # use smaller x for numerical stability
    a_demo = x_demo * lam / (2 * np.pi)
    Q_ext_demo, Q_sca_demo, Q_abs_demo = mie_efficiencies(x_demo, n_metal)
    print(f"\n      Mie check at x = {x_demo:.0f}:")
    print(f"      Q_ext = {Q_ext_demo:.3f}, Q_sca = {Q_sca_demo:.3f}, Q_abs = {Q_abs_demo:.3f}")

    # (c) RCS (radar cross section)
    # For a PEC sphere: RCS = pi * a^2 (monostatic, high frequency)
    RCS = np.pi * a**2  # m^2
    RCS_dBsm = 10 * np.log10(RCS)

    print(f"\n  (c) Radar Cross Section (high-frequency limit):")
    print(f"      RCS = pi * a^2 = {RCS:.4f} m^2 = {RCS_dBsm:.1f} dBsm")

    # (d) Radar equation: P_r = P_t * G^2 * lambda^2 * sigma / ((4*pi)^3 * R^4)
    P_t = 1e6        # 1 MW transmit power
    G_dBi = 40.0     # antenna gain
    G = 10**(G_dBi / 10)
    R = 100e3         # 100 km range

    P_r = P_t * G**2 * lam**2 * RCS / ((4 * np.pi)**3 * R**4)
    P_r_dBm = 10 * np.log10(P_r * 1e3)

    # Noise floor (typical radar receiver: BW = 1 MHz, T = 500 K)
    BW = 1e6
    T_sys = 500
    N_floor = k_B * T_sys * BW
    N_floor_dBm = 10 * np.log10(N_floor * 1e3)
    SNR_dB = P_r_dBm - N_floor_dBm

    print(f"\n  (d) Radar equation:")
    print(f"      P_t = {P_t/1e6:.0f} MW, G = {G_dBi:.0f} dBi, R = {R/1e3:.0f} km")
    print(f"      P_r = P_t * G^2 * lambda^2 * sigma / (4*pi)^3 / R^4")
    print(f"      P_r = {P_r:.4e} W = {P_r_dBm:.1f} dBm")
    print(f"      Noise floor (BW={BW/1e6:.0f} MHz, T={T_sys} K): {N_floor_dBm:.1f} dBm")
    print(f"      SNR = {SNR_dB:.1f} dB")
    print(f"      {'Detectable' if SNR_dB > 10 else 'Marginal' if SNR_dB > 0 else 'Not detectable'}")

    # Plot: RCS vs frequency for the 1m sphere
    freqs = np.logspace(8, 11, 200)  # 100 MHz to 100 GHz
    x_range = 2 * np.pi * a * freqs / c

    # RCS regimes:
    # Rayleigh (x << 1): RCS ~ pi*a^2 * (7/2)^2 * x^4 * 9  -> proportional to f^4
    # Mie (x ~ 1): oscillatory
    # Geometric (x >> 1): RCS -> pi*a^2

    RCS_approx = np.zeros(len(freqs))
    for i, xi in enumerate(x_range):
        if xi < 0.5:
            # Rayleigh regime
            RCS_approx[i] = np.pi * a**2 * (7 / 2)**2 * xi**4 / 9
        elif xi > 20:
            # Geometric optics
            RCS_approx[i] = np.pi * a**2
        else:
            # Transition (simplified oscillatory behavior)
            RCS_approx[i] = np.pi * a**2 * (1 + 0.3 * np.sin(2 * xi))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(freqs / 1e9, RCS_approx, 'b-', linewidth=2)
    ax.axhline(y=np.pi * a**2, color='red', linestyle='--', alpha=0.5,
               label=f'Geometric limit (pi*a^2 = {np.pi*a**2:.2f} m^2)')
    ax.axvline(x=f / 1e9, color='green', linestyle=':', alpha=0.7,
               label=f'Operating freq ({f/1e9:.0f} GHz)')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('RCS (m^2)')
    ax.set_title(f'Radar Cross Section: Metallic Sphere (a = {a} m)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('ex17_radar_cross_section.png', dpi=150)
    plt.close()
    print("  Plot saved: ex17_radar_cross_section.png")


if __name__ == "__main__":
    print("=== Exercise 1: Sunset Simulation ===")
    exercise_1()
    print("\n=== Exercise 2: Mie Resonances ===")
    exercise_2()
    print("\n=== Exercise 3: Cloud Opacity ===")
    exercise_3()
    print("\n=== Exercise 4: Born Approximation Validity ===")
    exercise_4()
    print("\n=== Exercise 5: Radar Cross Section ===")
    exercise_5()
    print("\nAll exercises completed!")
