#!/usr/bin/env python3
"""Spectroscopy: Line Profiles, Spectrometer Simulation, and Absorption.

This module demonstrates fundamental spectroscopic concepts:

1. Spectral line profiles — Lorentzian (natural/pressure broadening),
   Gaussian (Doppler broadening), and Voigt (combined) profiles
2. Diffraction grating spectrometer — angular dispersion maps wavelengths
   to positions on a focal plane detector
3. Fabry-Pérot interferometer — the Airy function produces sharp
   transmission peaks with resolving power R = m * F
4. Beer-Lambert absorption — exponential attenuation relates absorbance
   to concentration and path length

Physics background:
- Spectral line shapes arise from different broadening mechanisms:
    Natural: Heisenberg uncertainty → Lorentzian, FWHM ~ 1/(2*pi*tau)
    Doppler: thermal velocity distribution → Gaussian, FWHM ~ (nu0/c)*sqrt(8kT*ln2/m)
    Pressure: collisional interruption → Lorentzian, FWHM ~ N*v*sigma
    Combined: Voigt profile = convolution of Gaussian and Lorentzian
- Diffraction grating equation: d*(sin(theta_i) + sin(theta_m)) = m*lambda
  Resolving power R = m*N, where N is the number of illuminated grooves
- Fabry-Pérot transmission: T = 1 / (1 + F*sin^2(delta/2))
  where F = 4R/(1-R)^2 and delta = 4*pi*n*d*cos(theta)/lambda
- Beer-Lambert law: A = epsilon * c * l; T = 10^(-A)
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Spectral line profiles
# ---------------------------------------------------------------------------

def lorentzian(nu: np.ndarray, nu0: float, gamma: float) -> np.ndarray:
    """Lorentzian line profile (natural or pressure broadening).

    L(nu) = (1/pi) * (gamma/2) / ((nu - nu0)^2 + (gamma/2)^2)

    The Lorentzian arises from the Fourier transform of an exponentially
    damped oscillator, which is the classical model for a radiating atom.
    The FWHM equals gamma.

    Args:
        nu: Frequency array.
        nu0: Line center frequency.
        gamma: Full width at half maximum (FWHM).

    Returns:
        Normalized profile values (integral = 1).
    """
    return (1.0 / np.pi) * (gamma / 2.0) / ((nu - nu0)**2 + (gamma / 2.0)**2)


def gaussian_profile(nu: np.ndarray, nu0: float,
                     sigma: float) -> np.ndarray:
    """Gaussian line profile (Doppler broadening).

    G(nu) = (1/(sigma*sqrt(2*pi))) * exp(-(nu-nu0)^2 / (2*sigma^2))

    Doppler broadening arises because emitting atoms have a Maxwell-Boltzmann
    velocity distribution. The Doppler width (1-sigma) is:
        sigma_D = (nu0/c) * sqrt(kT/m)
    where m is the atomic mass.

    Args:
        nu: Frequency array.
        nu0: Line center frequency.
        sigma: Standard deviation (Doppler width parameter).

    Returns:
        Normalized profile values (integral = 1).
    """
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((nu - nu0) / sigma)**2
    )


def voigt(nu: np.ndarray, nu0: float, sigma: float,
          gamma: float) -> np.ndarray:
    """Voigt profile via numerical convolution.

    The Voigt profile is the convolution of a Gaussian (Doppler) and
    Lorentzian (natural/pressure) profile. It has a Gaussian core
    (dominated by Doppler near line center) and Lorentzian wings
    (dominated by pressure broadening far from center).

    This implementation uses direct convolution (no scipy dependency).
    For production use, scipy.special.wofz (Faddeeva function) is faster.

    Args:
        nu: Frequency array (must be uniformly spaced).
        nu0: Line center frequency.
        sigma: Gaussian standard deviation.
        gamma: Lorentzian FWHM.

    Returns:
        Normalized Voigt profile values.
    """
    dnu = nu[1] - nu[0]
    # Evaluate Lorentzian on the grid
    L = lorentzian(nu, nu0, gamma)
    # Gaussian convolution kernel (extend to ±5 sigma)
    n_kernel = int(5 * sigma / dnu)
    if n_kernel < 1:
        n_kernel = 1
    nu_k = np.arange(-n_kernel, n_kernel + 1) * dnu
    G_kernel = gaussian_profile(nu_k + nu0, nu0, sigma) * dnu
    # Convolve
    V = np.convolve(L, G_kernel, mode='same')
    # Normalize
    V /= np.trapezoid(V, nu)
    return V


# ---------------------------------------------------------------------------
# Diffraction grating spectrometer
# ---------------------------------------------------------------------------

def grating_angles(wavelengths: np.ndarray, d: float, theta_i: float,
                   m: int = 1) -> np.ndarray:
    """Compute diffraction angles from the grating equation.

    d * (sin(theta_i) + sin(theta_m)) = m * lambda

    Args:
        wavelengths: Array of wavelengths (same units as d).
        d: Groove spacing.
        theta_i: Incidence angle (radians).
        m: Diffraction order (positive integer).

    Returns:
        Diffraction angles in radians (NaN where no solution exists).
    """
    sin_theta_m = m * wavelengths / d - np.sin(theta_i)
    theta_m = np.full_like(wavelengths, np.nan, dtype=float)
    valid = np.abs(sin_theta_m) <= 1.0
    theta_m[valid] = np.arcsin(sin_theta_m[valid])
    return theta_m


def focal_plane_positions(wavelengths: np.ndarray, d: float,
                          theta_i: float, f: float,
                          m: int = 1) -> np.ndarray:
    """Map wavelengths to detector positions in a grating spectrometer.

    Uses a camera lens/mirror of focal length f to convert diffraction
    angles to positions: x = f * tan(theta_m - theta_center).

    Args:
        wavelengths: Wavelength array.
        d: Groove spacing (same units as wavelengths).
        theta_i: Incidence angle (radians).
        f: Camera focal length (output units).
        m: Diffraction order.

    Returns:
        Position array on the detector (same units as f).
    """
    theta_m = grating_angles(wavelengths, d, theta_i, m)
    theta_c = np.nanmedian(theta_m)
    x = f * np.tan(theta_m - theta_c)
    return x


# ---------------------------------------------------------------------------
# Fabry-Pérot interferometer
# ---------------------------------------------------------------------------

def fabry_perot_transmission(wavelength: np.ndarray, d: float,
                              R: float, n: float = 1.0,
                              theta: float = 0.0) -> np.ndarray:
    """Compute Fabry-Pérot etalon transmission (Airy function).

    T(lambda) = 1 / (1 + F * sin^2(delta/2))

    where:
        delta = 4*pi*n*d*cos(theta) / lambda
        F = 4*R / (1 - R)^2  (coefficient of finesse)

    The transmission peaks occur at delta = 2*m*pi, i.e., when
    2*n*d*cos(theta) = m*lambda (constructive interference).

    Args:
        wavelength: Wavelength array.
        d: Mirror separation (same units as wavelength).
        R: Mirror reflectance (0 to 1).
        n: Refractive index of gap medium.
        theta: Angle of incidence (radians).

    Returns:
        Transmission values (0 to 1).
    """
    delta = 4 * np.pi * n * d * np.cos(theta) / wavelength
    F = 4 * R / (1 - R)**2
    T = 1.0 / (1.0 + F * np.sin(delta / 2.0)**2)
    return T


def fp_finesse(R: float) -> float:
    """Reflectance finesse of a Fabry-Pérot interferometer.

    F = pi * sqrt(R) / (1 - R)
    """
    return np.pi * np.sqrt(R) / (1.0 - R)


def fp_fsr(wavelength: float, d: float, n: float = 1.0) -> float:
    """Free spectral range of a Fabry-Pérot in wavelength units.

    FSR = lambda^2 / (2*n*d)
    """
    return wavelength**2 / (2 * n * d)


# ---------------------------------------------------------------------------
# Beer-Lambert absorption
# ---------------------------------------------------------------------------

def beer_lambert(epsilon: np.ndarray | float, c: float,
                 l: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute transmittance and absorbance via Beer-Lambert law.

    A = epsilon * c * l
    T = 10^(-A)

    Args:
        epsilon: Molar absorption coefficient (L/(mol*cm)), scalar or array.
        c: Concentration (mol/L).
        l: Path length (cm).

    Returns:
        Tuple of (transmittance, absorbance) arrays.
    """
    epsilon = np.asarray(epsilon, dtype=float)
    A = epsilon * c * l
    T = 10.0**(-A)
    return T, A


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_line_broadening() -> None:
    """Compare Lorentzian, Gaussian, and Voigt line profiles.

    Simulates the H-alpha line at different conditions to show how
    each broadening mechanism shapes the spectral line.
    """
    # Parameters for H-alpha (656.28 nm, nu0 ≈ 4.568e14 Hz)
    nu0 = 4.568e14  # Hz
    # Doppler width at T = 10,000 K for hydrogen
    k_B = 1.381e-23
    m_H = 1.673e-27
    T = 10_000  # K
    sigma_D = (nu0 / 3e8) * np.sqrt(k_B * T / m_H)
    # Pressure broadening (Stark effect in plasma, ~1 GHz)
    gamma_P = 1e9  # Hz
    # Natural broadening (very small, ~100 MHz)
    gamma_N = 1e8  # Hz
    # Frequency grid (±20 sigma around line center)
    dnu = sigma_D / 10
    nu = np.arange(nu0 - 20 * sigma_D, nu0 + 20 * sigma_D, dnu)
    # Compute profiles
    L_nat = lorentzian(nu, nu0, gamma_N)
    L_press = lorentzian(nu, nu0, gamma_P)
    G = gaussian_profile(nu, nu0, sigma_D)
    V = voigt(nu, nu0, sigma_D, gamma_P)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Linear scale
    ax = axes[0]
    delta_nu = (nu - nu0) / 1e9  # GHz from line center
    ax.plot(delta_nu, G / G.max(), 'b-', label='Gaussian (Doppler)', linewidth=2)
    ax.plot(delta_nu, L_press / L_press.max(), 'r--',
            label='Lorentzian (pressure)', linewidth=2)
    ax.plot(delta_nu, V / V.max(), 'k-', label='Voigt (combined)',
            linewidth=2, alpha=0.8)
    ax.set_xlabel("Δν from line center (GHz)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title("Line Profiles — Linear Scale")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-15, 15)
    # Log scale (shows Lorentzian wings)
    ax = axes[1]
    ax.semilogy(delta_nu, G / G.max(), 'b-', label='Gaussian', linewidth=2)
    ax.semilogy(delta_nu, L_press / L_press.max(), 'r--',
                label='Lorentzian', linewidth=2)
    ax.semilogy(delta_nu, V / V.max(), 'k-', label='Voigt',
                linewidth=2, alpha=0.8)
    ax.set_xlabel("Δν from line center (GHz)")
    ax.set_ylabel("Normalized intensity (log)")
    ax.set_title("Line Profiles — Log Scale\n(Lorentzian wings visible)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-15, 15)
    ax.set_ylim(1e-4, 2)
    fig.suptitle(f"H-α Line Broadening (T={T:,} K, γ_P={gamma_P/1e9:.1f} GHz)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig("13_line_broadening.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[Saved] 13_line_broadening.png")
    # Print widths
    fwhm_D = 2 * np.sqrt(2 * np.log(2)) * sigma_D
    print(f"  Doppler FWHM: {fwhm_D/1e9:.2f} GHz ({fwhm_D * 656.28e-9**2 / 3e8 * 1e9:.4f} nm)")
    print(f"  Pressure FWHM: {gamma_P/1e9:.2f} GHz")


def plot_grating_spectrum() -> None:
    """Simulate a grating spectrometer observing an emission line spectrum.

    Models several emission lines (e.g., from a mercury lamp) dispersed
    by a 1200 grooves/mm grating in first order.
    """
    # Mercury emission lines (nm)
    hg_lines = {
        404.66: ("Violet", 0.5),
        435.83: ("Blue", 0.8),
        546.07: ("Green", 1.0),
        576.96: ("Yellow-1", 0.6),
        579.07: ("Yellow-2", 0.5),
    }
    # Grating parameters — all lengths in mm for consistency
    d = 1.0 / 1200  # groove spacing in mm (1/1200 mm ≈ 833 nm)
    theta_i = np.radians(30)  # incidence angle
    f = 500  # focal length in mm
    N_grooves = 60_000  # 50 mm illuminated width
    # Resolving power
    R_power = N_grooves  # first order
    # Wavelength grid for the "detector"
    lam = np.linspace(390, 590, 10000)  # nm
    # Compute detector positions (convert wavelength nm → mm)
    x = focal_plane_positions(lam * 1e-6, d, theta_i, f, m=1)  # mm

    # Build simulated spectrum (sum of instrument-broadened lines)
    spectrum = np.zeros_like(lam)
    for lam0, (name, intensity) in hg_lines.items():
        # Instrument linewidth: delta_lambda = lambda / R
        delta_lam = lam0 / R_power
        # Gaussian instrument profile
        sigma_inst = delta_lam / (2 * np.sqrt(2 * np.log(2)))
        line = intensity * np.exp(-0.5 * ((lam - lam0) / sigma_inst)**2)
        spectrum += line

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    # Spectrum vs wavelength
    axes[0].plot(lam, spectrum, 'k-', linewidth=0.8)
    for lam0, (name, intensity) in hg_lines.items():
        axes[0].axvline(lam0, color='gray', linewidth=0.5, alpha=0.5)
        axes[0].annotate(f"{lam0:.1f} nm\n{name}",
                         xy=(lam0, intensity * 1.05),
                         fontsize=7, ha='center')
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Relative Intensity")
    axes[0].set_title(f"Mercury Emission Spectrum (R = {R_power:,})")
    axes[0].grid(True, alpha=0.3)

    # Spectrum vs detector position
    axes[1].plot(x, spectrum, 'k-', linewidth=0.8)
    axes[1].set_xlabel("Detector position (mm)")
    axes[1].set_ylabel("Relative Intensity")
    axes[1].set_title(f"Focal Plane ({d*1e6:.0f} nm grating, f = {f} mm)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Diffraction Grating Spectrometer Simulation", fontsize=14)
    fig.tight_layout()
    fig.savefig("13_grating_spectrum.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[Saved] 13_grating_spectrum.png")
    print(f"  Grating: 1200 gr/mm, R = {R_power:,} (1st order)")
    print(f"  Resolution: {546.07/R_power:.4f} nm at 546 nm")


def plot_fabry_perot() -> None:
    """Plot Fabry-Pérot transmission for different reflectances.

    Shows how increasing mirror reflectance sharpens the transmission
    peaks (higher finesse), trading spectral range for resolution.
    """
    d = 5e-3  # 5 mm gap (in mm for wavelength in nm: use consistent units)
    n = 1.0
    # Wavelength range near 500 nm (show ~2 FSR)
    lam_center = 500.0  # nm
    fsr = fp_fsr(lam_center, d * 1e6, n)  # convert d to nm
    lam = np.linspace(lam_center - 1.5 * fsr, lam_center + 1.5 * fsr, 10000)
    reflectances = [0.5, 0.8, 0.95, 0.99]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Transmission curves
    ax = axes[0]
    for R_val in reflectances:
        T = fabry_perot_transmission(lam, d * 1e6, R_val, n)
        F = fp_finesse(R_val)
        ax.plot(lam - lam_center, T,
                label=f'R={R_val}, F={F:.0f}', linewidth=1.5)
    ax.set_xlabel(f"Δλ from {lam_center:.0f} nm (nm)")
    ax.set_ylabel("Transmission")
    ax.set_title(f"Fabry-Pérot Transmission (d = {d*1e3:.0f} mm)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Finesse vs reflectance
    ax = axes[1]
    R_arr = np.linspace(0.1, 0.999, 200)
    F_arr = [fp_finesse(r) for r in R_arr]
    ax.semilogy(R_arr, F_arr, 'b-', linewidth=2)
    ax.set_xlabel("Mirror Reflectance R")
    ax.set_ylabel("Finesse F")
    ax.set_title("Finesse vs Reflectance")
    ax.grid(True, alpha=0.3)
    # Mark the example reflectances
    for R_val in reflectances:
        F = fp_finesse(R_val)
        ax.plot(R_val, F, 'ro', markersize=8)
        ax.annotate(f'F={F:.0f}', xy=(R_val, F), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')

    fig.suptitle("Fabry-Pérot Interferometer", fontsize=14)
    fig.tight_layout()
    fig.savefig("13_fabry_perot.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[Saved] 13_fabry_perot.png")
    print(f"  FSR at {lam_center:.0f} nm: {fsr:.4f} nm")
    print(f"  Resolution (R=0.95): {fsr / fp_finesse(0.95):.6f} nm "
          f"(R_power = {int(lam_center / (fsr / fp_finesse(0.95)))})")


def plot_absorption() -> None:
    """Demonstrate Beer-Lambert absorption at varying concentrations.

    Simulates absorption spectroscopy of a dye with a Gaussian absorption
    band, showing how transmittance and absorbance change with concentration.
    """
    # Model absorption spectrum: Gaussian peak at 525 nm
    wavelengths = np.linspace(400, 700, 1000)  # nm
    # Molar absorption coefficient (L/(mol*cm))
    epsilon_peak = 2455  # KMnO4 at 525 nm
    lam_peak = 525
    sigma_abs = 30  # bandwidth (nm)
    epsilon = epsilon_peak * np.exp(-0.5 * ((wavelengths - lam_peak) / sigma_abs)**2)

    concentrations = [0.01, 0.02, 0.05, 0.1, 0.2]  # mmol/L
    path_length = 1.0  # cm

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(concentrations)))

    for c_mmol, color in zip(concentrations, colors):
        c_mol = c_mmol * 1e-3  # convert mmol/L to mol/L
        T, A = beer_lambert(epsilon, c_mol, path_length)
        axes[0].plot(wavelengths, T, color=color,
                     label=f'{c_mmol} mM', linewidth=1.5)
        axes[1].plot(wavelengths, A, color=color,
                     label=f'{c_mmol} mM', linewidth=1.5)

    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Transmittance")
    axes[0].set_title("Transmittance Spectra")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Absorbance")
    axes[1].set_title("Absorbance Spectra")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Calibration curve at peak wavelength
    c_range = np.linspace(0, 0.3, 100)  # mmol/L
    A_peak = epsilon_peak * c_range * 1e-3 * path_length
    axes[2].plot(c_range, A_peak, 'b-', linewidth=2)
    # Mark the example concentrations
    for c_mmol in concentrations:
        A_val = epsilon_peak * c_mmol * 1e-3 * path_length
        axes[2].plot(c_mmol, A_val, 'ro', markersize=8)
    axes[2].set_xlabel("Concentration (mmol/L)")
    axes[2].set_ylabel(f"Absorbance at {lam_peak} nm")
    axes[2].set_title("Beer-Lambert Calibration Curve")
    axes[2].grid(True, alpha=0.3)
    # Mark the linear range limit (A ≈ 2)
    axes[2].axhline(2.0, color='red', linewidth=1, linestyle='--', alpha=0.7)
    axes[2].annotate("A = 2 (1% T)\nNonlinear above", xy=(0.15, 2.05),
                     fontsize=8, color='red')

    fig.suptitle(f"Beer-Lambert Absorption (ε = {epsilon_peak} L/(mol·cm) "
                 f"at {lam_peak} nm, l = {path_length} cm)", fontsize=13)
    fig.tight_layout()
    fig.savefig("13_absorption.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[Saved] 13_absorption.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Spectroscopy: Line Profiles, Gratings, Fabry-Pérot, Absorption")
    print("=" * 60)

    print("\n1. Spectral line broadening comparison...")
    plot_line_broadening()

    print("\n2. Diffraction grating spectrometer simulation...")
    plot_grating_spectrum()

    print("\n3. Fabry-Pérot interferometer...")
    plot_fabry_perot()

    print("\n4. Beer-Lambert absorption spectroscopy...")
    plot_absorption()

    print("\nDone. All plots saved.")
