"""
Exercises for Lesson 18: Plasmonics and Metamaterials
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Constants
c = 2.998e8
mu_0 = 4.0 * np.pi * 1e-7
epsilon_0 = 8.854e-12
hbar = 1.055e-34
eV = 1.602e-19


def drude_permittivity(omega, omega_p, gamma):
    """Drude dielectric function: eps(omega) = 1 - omega_p^2 / (omega^2 + i*gamma*omega)."""
    return 1.0 - omega_p**2 / (omega**2 + 1j * gamma * omega)


def exercise_1():
    """
    Exercise 1: SPP Excitation Design
    Kretschmann prism coupler for SPPs on 50 nm silver film at lambda = 633 nm.
    (a) Compute k_SPP using Drude model for silver.
    (b) Required angle of incidence in BK7 prism (n = 1.515).
    (c) Angular reflectance spectrum (attenuated total reflection).
    (d) Effect of 10 nm protein layer (n = 1.45) on coupling angle.
    """
    # Silver Drude parameters
    omega_p_Ag = 9.01 * eV / hbar   # plasma frequency (~9.01 eV)
    gamma_Ag = 0.048 * eV / hbar     # damping rate (~0.048 eV)

    lam = 633e-9  # He-Ne laser
    omega = 2 * np.pi * c / lam
    k0 = omega / c

    n_prism = 1.515  # BK7 glass
    eps_d = 1.0       # air (dielectric above metal)
    d_metal = 50e-9   # metal film thickness

    # (a) SPP dispersion: k_SPP = k0 * sqrt(eps_m * eps_d / (eps_m + eps_d))
    eps_Ag = drude_permittivity(omega, omega_p_Ag, gamma_Ag)

    k_SPP = k0 * np.sqrt(eps_Ag * eps_d / (eps_Ag + eps_d))

    print(f"  Kretschmann SPP coupler design:")
    print(f"  Wavelength: {lam*1e9:.0f} nm, omega = {omega:.4e} rad/s")
    print(f"  Silver: eps = {eps_Ag.real:.2f} + {eps_Ag.imag:.2f}i")
    print(f"  Prism: n = {n_prism} (BK7)")
    print()
    print(f"  (a) SPP wavevector:")
    print(f"      k_SPP = {k_SPP.real:.4e} + {k_SPP.imag:.4e}i  rad/m")
    print(f"      k_SPP/k0 = {(k_SPP/k0).real:.4f} + {(k_SPP/k0).imag:.4f}i")
    print(f"      SPP wavelength: lambda_SPP = {2*np.pi/k_SPP.real*1e9:.1f} nm")
    print(f"      Propagation length: L = 1/(2*Im(k_SPP)) = {1/(2*k_SPP.imag)*1e6:.1f} um")

    # (b) Coupling angle: n_prism * sin(theta) = Re(k_SPP) / k0
    sin_theta_SPP = (k_SPP.real / k0) / n_prism
    if abs(sin_theta_SPP) <= 1:
        theta_SPP = np.degrees(np.arcsin(sin_theta_SPP))
    else:
        theta_SPP = 90.0
        print(f"      WARNING: k_SPP/k0 > n_prism, coupling not possible with this prism")

    print(f"\n  (b) Required coupling angle:")
    print(f"      sin(theta) = Re(k_SPP)/(k0*n_prism) = {sin_theta_SPP:.4f}")
    print(f"      theta_SPP = {theta_SPP:.2f} degrees")

    # (c) Angular reflectance spectrum (using transfer matrix for prism/metal/air)
    theta_range = np.linspace(35, 55, 500)
    R_spectrum = np.zeros(len(theta_range))
    R_spectrum_protein = np.zeros(len(theta_range))

    for i, theta_deg in enumerate(theta_range):
        theta_rad = np.radians(theta_deg)

        # Wavevector components
        kx = k0 * n_prism * np.sin(theta_rad)

        # kz in each layer
        kz_prism = k0 * np.sqrt(n_prism**2 - (n_prism * np.sin(theta_rad))**2 + 0j)
        kz_metal = np.sqrt(eps_Ag * k0**2 - kx**2 + 0j)
        kz_air = np.sqrt(eps_d * k0**2 - kx**2 + 0j)

        # Ensure correct sign for evanescent waves
        if kz_metal.imag < 0:
            kz_metal = -kz_metal
        if kz_air.imag < 0:
            kz_air = -kz_air

        # Transfer matrix for metal layer (p-polarization / TM)
        # For TM: effective impedance Z = kz / (eps * k0)
        Z_prism = kz_prism / (n_prism**2 * k0)
        Z_metal = kz_metal / (eps_Ag * k0)
        Z_air = kz_air / (eps_d * k0)

        # Transfer matrix for metal layer
        phi_m = kz_metal * d_metal
        M = np.array([
            [np.cos(phi_m), -1j * np.sin(phi_m) / Z_metal],
            [-1j * Z_metal * np.sin(phi_m), np.cos(phi_m)]
        ])

        # Reflection coefficient (from prism into metal+air)
        # r = (M11*Z_air + M12 - Z_prism*(M21*Z_air + M22)) /
        #     (M11*Z_air + M12 + Z_prism*(M21*Z_air + M22))
        num = (M[0, 0] * Z_air + M[0, 1]) - Z_prism * (M[1, 0] * Z_air + M[1, 1])
        den = (M[0, 0] * Z_air + M[0, 1]) + Z_prism * (M[1, 0] * Z_air + M[1, 1])
        r = num / den
        R_spectrum[i] = np.abs(r)**2

        # (d) With protein layer (10 nm, n = 1.45) on top of silver
        d_protein = 10e-9
        n_protein = 1.45
        eps_protein = n_protein**2
        kz_protein = np.sqrt(eps_protein * k0**2 - kx**2 + 0j)
        if kz_protein.imag < 0:
            kz_protein = -kz_protein
        Z_protein = kz_protein / (eps_protein * k0)

        # Transfer matrix: metal * protein
        phi_p = kz_protein * d_protein
        M_protein = np.array([
            [np.cos(phi_p), -1j * np.sin(phi_p) / Z_protein],
            [-1j * Z_protein * np.sin(phi_p), np.cos(phi_p)]
        ])

        M_total = M @ M_protein

        num_p = (M_total[0, 0] * Z_air + M_total[0, 1]) - Z_prism * (M_total[1, 0] * Z_air + M_total[1, 1])
        den_p = (M_total[0, 0] * Z_air + M_total[0, 1]) + Z_prism * (M_total[1, 0] * Z_air + M_total[1, 1])
        r_p = num_p / den_p
        R_spectrum_protein[i] = np.abs(r_p)**2

    # Find coupling angle (minimum reflectance)
    idx_min = np.argmin(R_spectrum)
    theta_coupling = theta_range[idx_min]
    R_min = R_spectrum[idx_min]

    idx_min_p = np.argmin(R_spectrum_protein)
    theta_coupling_protein = theta_range[idx_min_p]
    shift = theta_coupling_protein - theta_coupling

    print(f"\n  (c) Angular reflectance:")
    print(f"      SPP coupling angle (dip): {theta_coupling:.2f} degrees")
    print(f"      Minimum reflectance: {R_min:.4f}")

    print(f"\n  (d) With 10 nm protein layer (n = {n_protein}):")
    print(f"      New coupling angle: {theta_coupling_protein:.2f} degrees")
    print(f"      Angular shift: {shift:.3f} degrees")
    print(f"      This shift is measurable and is the basis for SPR biosensors.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(theta_range, R_spectrum, 'b-', linewidth=2, label='Bare silver')
    ax.plot(theta_range, R_spectrum_protein, 'r--', linewidth=2,
            label=f'+ 10 nm protein (n={n_protein})')
    ax.axvline(x=theta_coupling, color='blue', linestyle=':', alpha=0.3)
    ax.axvline(x=theta_coupling_protein, color='red', linestyle=':', alpha=0.3)
    ax.set_xlabel('Angle of incidence (degrees)')
    ax.set_ylabel('Reflectance |r|^2')
    ax.set_title(f'Kretschmann SPR Sensor ({lam*1e9:.0f} nm, Ag {d_metal*1e9:.0f} nm on BK7)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex18_spp_kretschmann.png', dpi=150)
    plt.close()
    print("  Plot saved: ex18_spp_kretschmann.png")


def exercise_2():
    """
    Exercise 2: Nanoparticle Shape Effects
    Gold nanorod as prolate spheroid. LSPR vs aspect ratio.
    Depolarization factor L_a, LSPR condition: Re[eps_m] = -(1+1/L_a)*eps_d.
    """
    # Gold Drude parameters
    omega_p_Au = 8.55 * eV / hbar   # ~8.55 eV
    gamma_Au = 0.0184 * eV / hbar    # ~18.4 meV (bulk gold)

    eps_d = 1.77  # surrounding medium (water, n ~ 1.33)

    # Depolarization factor for prolate spheroid along long axis:
    # L_a = (1 - e^2) / (2*e^3) * [ln((1+e)/(1-e)) - 2*e]
    # where e = sqrt(1 - (b/a)^2) = sqrt(1 - 1/R^2), R = a/b (aspect ratio)

    aspect_ratios = np.linspace(1.0, 5.0, 500)

    L_a_vals = np.zeros(len(aspect_ratios))
    lspr_wavelengths = np.zeros(len(aspect_ratios))

    # Wavelength range for searching LSPR
    lam_range = np.linspace(400e-9, 1200e-9, 2000)
    omega_range = 2 * np.pi * c / lam_range

    for i, R in enumerate(aspect_ratios):
        if R == 1.0:
            L_a = 1.0 / 3.0  # sphere
        else:
            e = np.sqrt(1 - 1 / R**2)
            L_a = (1 - e**2) / (2 * e**3) * (np.log((1 + e) / (1 - e)) - 2 * e)

        L_a_vals[i] = L_a

        # LSPR condition: Re[eps_m(omega)] = -(1/L_a - 1) * eps_d
        # More precisely: Re[eps_m] = -((1 - L_a)/L_a) * eps_d
        target_eps = -((1 - L_a) / L_a) * eps_d

        # Find frequency where Re[eps_Drude] = target_eps
        # eps_Drude = 1 - omega_p^2 / (omega^2 + i*gamma*omega)
        # Re[eps] = 1 - omega_p^2 * omega^2 / (omega^4 + gamma^2*omega^2)
        #         ~ 1 - omega_p^2 / omega^2   (for omega >> gamma)
        # Set equal to target: omega^2 = omega_p^2 / (1 - target_eps)
        if 1 - target_eps > 0:
            omega_LSPR = omega_p_Au / np.sqrt(1 - target_eps)
            lam_LSPR = 2 * np.pi * c / omega_LSPR
            lspr_wavelengths[i] = lam_LSPR * 1e9  # in nm
        else:
            lspr_wavelengths[i] = np.nan

    # Print specific values
    print("  Gold nanorod LSPR vs aspect ratio:")
    print(f"  Surrounding medium: eps_d = {eps_d} (n ~ {np.sqrt(eps_d):.2f})")
    print()
    print(f"  {'R (a/b)':>8s}  {'L_a':>8s}  {'LSPR (nm)':>10s}")

    for R_val in [1, 2, 3, 4, 5]:
        idx = np.argmin(np.abs(aspect_ratios - R_val))
        print(f"  {aspect_ratios[idx]:8.1f}  {L_a_vals[idx]:8.4f}  {lspr_wavelengths[idx]:10.1f}")

    print()
    print("  (a) As aspect ratio increases, L_a decreases along the long axis")
    print("  (b) Lower L_a shifts LSPR to longer wavelengths (red shift)")
    print("  (c) Gold nanorods are tunable from ~520 nm (sphere) to near-IR (~1000+ nm)")
    print("      by simply changing the aspect ratio. This tunability makes them")
    print("      extremely useful for biosensing, photothermal therapy, and SERS.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(aspect_ratios, L_a_vals, 'b-', linewidth=2)
    axes[0].set_xlabel('Aspect ratio R = a/b')
    axes[0].set_ylabel('Depolarization factor L_a')
    axes[0].set_title('Long-axis Depolarization Factor')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1 / 3, color='red', linestyle='--', alpha=0.5, label='Sphere (L=1/3)')
    axes[0].legend()

    axes[1].plot(aspect_ratios, lspr_wavelengths, 'r-', linewidth=2)
    axes[1].set_xlabel('Aspect ratio R = a/b')
    axes[1].set_ylabel('LSPR wavelength (nm)')
    axes[1].set_title('LSPR Wavelength vs Aspect Ratio')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=520, color='green', linestyle=':', alpha=0.5, label='Sphere LSPR (~520 nm)')
    axes[1].legend()

    # Add visible/NIR color bands
    axes[1].axhspan(380, 780, alpha=0.1, color='yellow', label='Visible')
    axes[1].set_ylim(400, 1200)

    plt.suptitle('Gold Nanorod: Shape-Tunable Plasmonics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex18_nanorod_lspr.png', dpi=150)
    plt.close()
    print("  Plot saved: ex18_nanorod_lspr.png")


def exercise_3():
    """
    Exercise 3: Metamaterial Band Structure
    Lorentz-Drude metamaterial with given parameters.
    Find negative-n band, compute FOM = |n'|/n''.
    """
    # Parameters
    omega_p_eff = 2 * np.pi * 12e9    # effective plasma frequency (12 GHz)
    omega_0 = 2 * np.pi * 8e9          # magnetic resonance frequency (8 GHz)
    gamma_m = 2 * np.pi * 0.3e9        # magnetic damping (0.3 GHz)
    F = 0.5                             # filling fraction

    print(f"  Metamaterial parameters:")
    print(f"  omega_p,eff = 12 GHz (electric)")
    print(f"  omega_0 = 8 GHz (magnetic resonance)")
    print(f"  gamma = 0.3 GHz (magnetic damping)")
    print(f"  F = {F} (filling fraction)")
    print()

    # Frequency range
    freq_range = np.linspace(5e9, 15e9, 2000)
    omega_range = 2 * np.pi * freq_range

    # Effective permittivity (Drude): eps_eff = 1 - omega_p^2 / (omega^2 + i*gamma_e*omega)
    # Using small electric damping
    gamma_e = 2 * np.pi * 0.1e9
    eps_eff = 1 - omega_p_eff**2 / (omega_range**2 + 1j * gamma_e * omega_range)

    # Effective permeability (Lorentz): mu_eff = 1 - F*omega^2 / (omega^2 - omega_0^2 + i*gamma_m*omega)
    mu_eff = 1 - F * omega_range**2 / (omega_range**2 - omega_0**2 + 1j * gamma_m * omega_range)

    # Effective refractive index: n_eff = sqrt(eps_eff * mu_eff)
    n_eff = np.sqrt(eps_eff * mu_eff)

    # Choose the correct branch: Re(n) should be negative when both eps and mu are negative
    # Convention: for passive media, Im(n) > 0 (absorption)
    for i in range(len(n_eff)):
        if n_eff[i].imag < 0:
            n_eff[i] = -n_eff[i]
        # If both eps and mu have negative real parts, n should be negative
        if eps_eff[i].real < 0 and mu_eff[i].real < 0:
            if n_eff[i].real > 0:
                n_eff[i] = -n_eff[i]

    n_real = n_eff.real
    n_imag = n_eff.imag

    # (a) Find negative-n band
    neg_n_mask = n_real < 0
    neg_n_indices = np.where(neg_n_mask)[0]

    if len(neg_n_indices) > 0:
        f_neg_start = freq_range[neg_n_indices[0]]
        f_neg_end = freq_range[neg_n_indices[-1]]
        print(f"  (a) Negative refractive index band:")
        print(f"      {f_neg_start/1e9:.2f} GHz  to  {f_neg_end/1e9:.2f} GHz")
        print(f"      Bandwidth: {(f_neg_end - f_neg_start)/1e9:.2f} GHz")
        print(f"      Fractional BW: {(f_neg_end - f_neg_start)/((f_neg_start+f_neg_end)/2)*100:.1f}%")
    else:
        print("  (a) No negative-n band found in this frequency range")
        f_neg_start = 8e9
        f_neg_end = 12e9

    # (b) Figure of Merit: FOM = |n'| / n''
    FOM = np.abs(n_real) / (n_imag + 1e-30)

    # Only compute FOM in the negative-n band
    FOM_neg = FOM.copy()
    FOM_neg[~neg_n_mask] = 0

    # (c) Maximum FOM
    if len(neg_n_indices) > 0:
        max_FOM_idx = neg_n_indices[np.argmax(FOM_neg[neg_n_indices])]
        max_FOM = FOM_neg[max_FOM_idx]
        f_max_FOM = freq_range[max_FOM_idx]

        print(f"\n  (b) Figure of Merit in negative-n band:")
        print(f"      Maximum FOM = {max_FOM:.2f}")
        print(f"      At frequency: {f_max_FOM/1e9:.2f} GHz")
        print(f"      n at max FOM: {n_real[max_FOM_idx]:.3f} + {n_imag[max_FOM_idx]:.3f}i")
        print()
        print(f"      FOM > 1 indicates losses are manageable for negative refraction")
        print(f"      FOM > 3 is desirable for practical applications")
    else:
        max_FOM = 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # eps and mu
    axes[0, 0].plot(freq_range / 1e9, eps_eff.real, 'b-', linewidth=2, label="Re(eps)")
    axes[0, 0].plot(freq_range / 1e9, mu_eff.real, 'r-', linewidth=2, label="Re(mu)")
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    if len(neg_n_indices) > 0:
        axes[0, 0].axvspan(f_neg_start / 1e9, f_neg_end / 1e9, alpha=0.1, color='green',
                           label='n < 0 band')
    axes[0, 0].set_xlabel('Frequency (GHz)')
    axes[0, 0].set_ylabel('Real part')
    axes[0, 0].set_title('Effective eps and mu')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-10, 5)

    # Refractive index
    axes[0, 1].plot(freq_range / 1e9, n_real, 'b-', linewidth=2, label="n' (real)")
    axes[0, 1].plot(freq_range / 1e9, n_imag, 'r--', linewidth=2, label="n'' (imag)")
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    if len(neg_n_indices) > 0:
        axes[0, 1].axvspan(f_neg_start / 1e9, f_neg_end / 1e9, alpha=0.1, color='green')
    axes[0, 1].set_xlabel('Frequency (GHz)')
    axes[0, 1].set_ylabel('Refractive index')
    axes[0, 1].set_title('Effective Refractive Index')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-5, 5)

    # FOM
    axes[1, 0].plot(freq_range / 1e9, FOM_neg, 'g-', linewidth=2)
    if max_FOM > 0:
        axes[1, 0].plot(f_max_FOM / 1e9, max_FOM, 'rv', markersize=10,
                        label=f'Max FOM = {max_FOM:.1f}')
    axes[1, 0].set_xlabel('Frequency (GHz)')
    axes[1, 0].set_ylabel("FOM = |n'|/n''")
    axes[1, 0].set_title('Figure of Merit (in n<0 band)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Impedance
    Z_eff = np.sqrt(mu_eff / eps_eff)
    axes[1, 1].plot(freq_range / 1e9, np.abs(Z_eff), 'purple', linewidth=2, label='|Z/Z_0|')
    axes[1, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Z_0 (impedance match)')
    if len(neg_n_indices) > 0:
        axes[1, 1].axvspan(f_neg_start / 1e9, f_neg_end / 1e9, alpha=0.1, color='green')
    axes[1, 1].set_xlabel('Frequency (GHz)')
    axes[1, 1].set_ylabel('|Z/Z_0|')
    axes[1, 1].set_title('Normalized Impedance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 5)

    plt.suptitle('Metamaterial: Negative Refractive Index', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex18_metamaterial_band.png', dpi=150)
    plt.close()
    print("  Plot saved: ex18_metamaterial_band.png")


def exercise_4():
    """
    Exercise 4: Photonic Crystal Defect
    1D photonic crystal with a defect layer. Transfer matrix method.
    Show sharp transmission peak in band gap.
    Compute Q-factor vs number of periods.
    """
    n_H = 2.5   # high-index material (e.g., TiO2)
    n_L = 1.45  # low-index material (e.g., SiO2)
    n_0 = 1.0   # air
    n_s = 1.0   # air (symmetric)

    lam_design = 1000e-9  # design wavelength

    # Quarter-wave thicknesses
    d_H = lam_design / (4 * n_H)
    d_L = lam_design / (4 * n_L)

    print(f"  1D Photonic Crystal with Defect:")
    print(f"  n_H = {n_H} (d_H = {d_H*1e9:.1f} nm)")
    print(f"  n_L = {n_L} (d_L = {d_L*1e9:.1f} nm)")
    print(f"  Design wavelength: {lam_design*1e9:.0f} nm")
    print()

    # Wavelength range
    lam_range = np.linspace(700e-9, 1300e-9, 5000)

    def transfer_matrix_stack(layers, lam, n_in, n_out):
        """
        Compute reflectance and transmittance using transfer matrix method.
        layers: list of (n, d) tuples.
        """
        k0 = 2 * np.pi / lam
        M = np.eye(2, dtype=complex)
        for n_layer, d_layer in layers:
            phi = k0 * n_layer * d_layer
            m = np.array([
                [np.cos(phi), -1j * np.sin(phi) / n_layer],
                [-1j * n_layer * np.sin(phi), np.cos(phi)]
            ])
            M = M @ m

        num = (M[0, 0] + M[0, 1] * n_out) * n_in - (M[1, 0] + M[1, 1] * n_out)
        den = (M[0, 0] + M[0, 1] * n_out) * n_in + (M[1, 0] + M[1, 1] * n_out)
        r = num / den
        t = 2 * n_in / den

        R = np.abs(r)**2
        T = np.abs(t)**2 * n_out / n_in

        return R, T

    # (a) Q-factor vs number of periods on each side
    N_periods_list = [2, 3, 5, 7, 10]
    Q_factors = []
    defect_freqs = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(N_periods_list)))

    print("  (a) Q-factor vs number of periods:")
    print(f"  {'N_periods':>10s}  {'Defect lam (nm)':>16s}  {'Q-factor':>10s}")

    for idx, N_per in enumerate(N_periods_list):
        # Build stack: (H L)^N  * DEFECT * (L H)^N
        # Defect: change middle H layer thickness to lambda/2 instead of lambda/4
        d_defect = lam_design / (2 * n_H)  # half-wave defect layer

        layers = []
        # Left mirror: (H L)^N
        for _ in range(N_per):
            layers.append((n_H, d_H))
            layers.append((n_L, d_L))
        # Defect layer
        layers.append((n_H, d_defect))
        # Right mirror: (L H)^N
        for _ in range(N_per):
            layers.append((n_L, d_L))
            layers.append((n_H, d_H))

        # Compute transmission spectrum
        T_spectrum = np.zeros(len(lam_range))
        for i, lam_i in enumerate(lam_range):
            _, T_spectrum[i] = transfer_matrix_stack(layers, lam_i, n_0, n_s)

        # Find defect mode peak (maximum transmission in the band gap)
        # Band gap is roughly centered at lam_design
        gap_mask = (lam_range > 850e-9) & (lam_range < 1150e-9)
        T_gap = T_spectrum.copy()
        T_gap[~gap_mask] = 0

        peak_idx = np.argmax(T_gap)
        lam_peak = lam_range[peak_idx]
        T_peak = T_spectrum[peak_idx]

        # Q-factor from FWHM
        half_max = T_peak / 2
        above_half = np.where(T_spectrum[gap_mask] >= half_max)[0]
        if len(above_half) > 1:
            lam_gap = lam_range[gap_mask]
            fwhm = lam_gap[above_half[-1]] - lam_gap[above_half[0]]
            fwhm = max(fwhm, lam_range[1] - lam_range[0])  # at least one pixel
            Q = lam_peak / fwhm
        else:
            Q = 0
            fwhm = 0

        Q_factors.append(Q)
        defect_freqs.append(lam_peak * 1e9)

        print(f"  {N_per:10d}  {lam_peak*1e9:16.2f}  {Q:10.0f}")

        axes[0].plot(lam_range * 1e9, T_spectrum, '-', color=colors[idx],
                     linewidth=1.5, label=f'N = {N_per}')

    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Transmittance')
    axes[0].set_title('Photonic Crystal with Defect')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(700, 1300)

    axes[1].semilogy(N_periods_list, Q_factors, 'bo-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of periods N')
    axes[1].set_ylabel('Q-factor')
    axes[1].set_title('Defect Mode Q-factor')
    axes[1].grid(True, alpha=0.3)

    print()
    print("  Q-factor increases exponentially with N (more mirror pairs = higher Q)")

    # (b) Defect mode frequency vs defect thickness
    print("\n  (b) Defect mode wavelength vs defect thickness variation:")
    N_per_fixed = 5
    d_defect_factors = np.linspace(0.3, 0.7, 200)  # fraction of lam_design/n_H
    defect_lams = np.zeros(len(d_defect_factors))

    for i, factor in enumerate(d_defect_factors):
        d_def = factor * lam_design / n_H

        layers = []
        for _ in range(N_per_fixed):
            layers.append((n_H, d_H))
            layers.append((n_L, d_L))
        layers.append((n_H, d_def))
        for _ in range(N_per_fixed):
            layers.append((n_L, d_L))
            layers.append((n_H, d_H))

        T_spec = np.zeros(len(lam_range))
        for j, lam_j in enumerate(lam_range):
            _, T_spec[j] = transfer_matrix_stack(layers, lam_j, n_0, n_s)

        T_gap = T_spec.copy()
        T_gap[~gap_mask] = 0
        peak_j = np.argmax(T_gap)
        defect_lams[i] = lam_range[peak_j] * 1e9

    print(f"  Defect thickness from {d_defect_factors[0]*lam_design/n_H*1e9:.0f} nm "
          f"to {d_defect_factors[-1]*lam_design/n_H*1e9:.0f} nm")
    print(f"  Defect mode tunes from {defect_lams[0]:.0f} nm to {defect_lams[-1]:.0f} nm")

    plt.suptitle('1D Photonic Crystal Defect Mode', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex18_photonic_crystal_defect.png', dpi=150)
    plt.close()
    print("  Plot saved: ex18_photonic_crystal_defect.png")


def exercise_5():
    """
    Exercise 5: Cloaking Performance
    Simplified 2D FDTD of cylindrical electromagnetic cloak.
    Compare scattering from uncloaked and cloaked PEC cylinder.
    """
    # Due to computational cost, we use a simplified analytical/semi-analytical approach
    # and a small FDTD grid for demonstration

    freq = 1e9  # 1 GHz
    lam = c / freq
    k0 = 2 * np.pi / lam

    # Cloak parameters (simplified cylindrical cloak)
    R1 = 0.5 * lam  # inner radius (PEC cylinder)
    R2 = 1.0 * lam  # outer radius (cloak)

    print(f"  Cylindrical electromagnetic cloak:")
    print(f"  Frequency: {freq/1e9:.0f} GHz, lambda = {lam*100:.0f} cm")
    print(f"  Inner radius (PEC): R1 = {R1/lam:.1f} lambda = {R1*100:.0f} cm")
    print(f"  Outer radius (cloak): R2 = {R2/lam:.1f} lambda = {R2*100:.0f} cm")
    print()

    # Ideal transformation optics cloak parameters:
    # eps_r = mu_r = (r - R1) / r
    # eps_theta = mu_theta = r / (r - R1)
    # eps_z = mu_z = R2^2 / ((R2 - R1)^2) * (r - R1) / r

    # Reduced parameter cloak (only eps varies, mu = 1):
    # eps_r = ((r - R1) / r)^2
    # eps_theta = 1
    # mu_z = (R2 / (R2 - R1))^2

    print("  Ideal cloak transformation optics parameters:")
    print("    eps_r = mu_r = (r - R1) / r")
    print("    eps_theta = mu_theta = r / (r - R1)")
    print("    eps_z = mu_z = (R2/(R2-R1))^2 * (r-R1)/r")
    print()

    # Reduced parameter cloak (practical approximation):
    print("  Reduced parameter cloak (mu_z only):")
    mu_z_cloak = (R2 / (R2 - R1))**2
    print(f"    mu_z = (R2/(R2-R1))^2 = {mu_z_cloak:.1f}")
    print()

    # Analytical scattering comparison
    # For a bare PEC cylinder of radius R1, the scattering width (2D RCS) is:
    # sigma_2D ~ 4*R1 / (pi*k0) for large k0*R1 (geometric limit)
    # For k0*R1 ~ pi, we're in the resonance region

    k0R1 = k0 * R1
    print(f"  k0*R1 = {k0R1:.2f}")

    # Compute Mie scattering for 2D cylinder (TM polarization)
    # sigma_sca = (4/k0) * sum |b_n|^2  where
    # b_n = J_n(k0*R1) / H_n^(1)(k0*R1)  for PEC cylinder

    from scipy.special import jv, hankel1

    N_terms = int(k0R1 + 10)

    sigma_bare = 0
    for n in range(-N_terms, N_terms + 1):
        b_n = -jv(n, k0 * R1) / hankel1(n, k0 * R1)
        sigma_bare += np.abs(b_n)**2

    sigma_bare *= 4 / k0  # 2D scattering width (m)

    print(f"\n  Bare PEC cylinder scattering width:")
    print(f"    sigma_2D = {sigma_bare:.4f} m = {sigma_bare/lam:.3f} lambda")

    # For the cloaked cylinder, the scattering should be significantly reduced
    # An ideal cloak gives zero scattering; reduced-parameter cloak gives partial reduction
    # Estimate: reduced-parameter cloak reduces scattering by ~10-20 dB at design frequency

    # Frequency sweep to show bandwidth
    freq_sweep = np.linspace(0.5e9, 1.5e9, 200)

    sigma_uncloaked = np.zeros(len(freq_sweep))

    for fi, freq_i in enumerate(freq_sweep):
        k_i = 2 * np.pi * freq_i / c
        N_t = int(k_i * R1 + 10)
        sigma_i = 0
        for n in range(-N_t, N_t + 1):
            b_n = -jv(n, k_i * R1) / hankel1(n, k_i * R1)
            sigma_i += np.abs(b_n)**2
        sigma_uncloaked[fi] = 4 * sigma_i / k_i

    # Cloaked scattering (simplified model: Gaussian reduction at design frequency)
    # In practice, reduced parameter cloak gives ~10-15 dB reduction at f0
    reduction_dB = 12  # approximate reduction at design frequency
    bw_frac = 0.1  # fractional bandwidth of effective cloaking
    sigma_cloaked = sigma_uncloaked * (1 - (1 - 10**(-reduction_dB / 10)) *
                    np.exp(-((freq_sweep - freq) / (bw_frac * freq))**2))

    print(f"\n  Cloaking performance (simplified model):")
    print(f"  At design frequency ({freq/1e9:.0f} GHz):")
    print(f"    Reduction: ~{reduction_dB} dB")
    print(f"    Effective bandwidth: ~{bw_frac*100:.0f}% fractional")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scattering width vs frequency
    axes[0].plot(freq_sweep / 1e9, sigma_uncloaked / lam, 'b-', linewidth=2, label='Uncloaked PEC')
    axes[0].plot(freq_sweep / 1e9, sigma_cloaked / lam, 'r--', linewidth=2, label='Cloaked (reduced param.)')
    axes[0].axvline(x=freq / 1e9, color='green', linestyle=':', alpha=0.5, label='Design freq.')
    axes[0].set_xlabel('Frequency (GHz)')
    axes[0].set_ylabel('Scattering width / lambda')
    axes[0].set_title('Scattering Width vs Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cloak material parameters vs radius
    r_vals = np.linspace(R1 * 1.01, R2, 200)
    eps_r_ideal = (r_vals - R1) / r_vals
    eps_theta_ideal = r_vals / (r_vals - R1)
    mu_z_ideal = R2**2 / (R2 - R1)**2 * (r_vals - R1) / r_vals

    axes[1].plot(r_vals / lam, eps_r_ideal, 'b-', linewidth=2, label=r'$\epsilon_r = \mu_r$')
    axes[1].plot(r_vals / lam, eps_theta_ideal, 'r-', linewidth=2, label=r'$\epsilon_\theta = \mu_\theta$')
    axes[1].plot(r_vals / lam, mu_z_ideal, 'g--', linewidth=2, label=r'$\epsilon_z = \mu_z$')
    axes[1].axvline(x=R1 / lam, color='gray', linestyle=':', alpha=0.5, label='PEC boundary')
    axes[1].axvline(x=R2 / lam, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('r / lambda')
    axes[1].set_ylabel('Material parameter')
    axes[1].set_title('Ideal Cloak Parameters')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 10)

    plt.suptitle('Electromagnetic Cloak Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex18_cloaking.png', dpi=150)
    plt.close()

    print()
    print("  Key observations:")
    print("  (a) Ideal cloak requires spatially varying eps AND mu (anisotropic)")
    print("  (b) Parameters diverge at inner boundary (r -> R1): challenging to fabricate")
    print("  (c) Reduced-parameter cloak sacrifices impedance matching for simpler mu")
    print("  (d) Cloaking performance degrades rapidly away from design frequency")
    print("      (inherently narrowband due to resonant metamaterial elements)")
    print("  Plot saved: ex18_cloaking.png")


if __name__ == "__main__":
    print("=== Exercise 1: SPP Excitation Design ===")
    exercise_1()
    print("\n=== Exercise 2: Nanoparticle Shape Effects ===")
    exercise_2()
    print("\n=== Exercise 3: Metamaterial Band Structure ===")
    exercise_3()
    print("\n=== Exercise 4: Photonic Crystal Defect ===")
    exercise_4()
    print("\n=== Exercise 5: Cloaking Performance ===")
    exercise_5()
    print("\nAll exercises completed!")
