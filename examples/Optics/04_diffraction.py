#!/usr/bin/env python3
"""Diffraction Patterns: Single Slit, Circular Aperture, and Gratings.

This module computes and visualizes Fraunhofer (far-field) diffraction patterns
for various aperture geometries:

1. Single slit — the sinc^2 intensity pattern
2. Circular aperture — the Airy pattern (Bessel function)
3. Diffraction grating — sharp principal maxima with resolving power
4. Comparison of single slit, double slit, and N-slit patterns

Physics background:
- Fraunhofer diffraction is the far-field limit where the observation screen
  is at infinity (or at the focal plane of a lens). The diffraction pattern
  is the squared modulus of the Fourier transform of the aperture function.
- Single slit: I(theta) = I0 * [sin(beta)/beta]^2, beta = pi*a*sin(theta)/lambda
- Circular aperture: I(theta) = I0 * [2*J1(x)/x]^2, x = pi*D*sin(theta)/lambda
  This produces the Airy disk, which sets the diffraction limit of imaging systems.
- Diffraction grating: N equally spaced slits produce sharp maxima whose
  angular positions satisfy d*sin(theta) = m*lambda (grating equation).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1 as bessel_j1  # First-order Bessel function


# ---------------------------------------------------------------------------
# Single slit diffraction
# ---------------------------------------------------------------------------

def single_slit_intensity(theta: np.ndarray, a: float, wavelength: float) -> np.ndarray:
    """Compute the Fraunhofer diffraction pattern of a single slit.

    The intensity is proportional to [sin(beta)/beta]^2 where
    beta = pi * a * sin(theta) / lambda.

    The sinc pattern arises because a rectangular aperture has a sinc-shaped
    Fourier transform. The first zeros occur at sin(theta) = m*lambda/a.

    Args:
        theta: Array of angles from the optical axis (radians).
        a: Slit width.
        wavelength: Wavelength.

    Returns:
        Normalized intensity (peak = 1).
    """
    beta = np.pi * a * np.sin(theta) / wavelength
    # Handle the central maximum (beta=0) using numpy's sinc
    # np.sinc(x) = sin(pi*x)/(pi*x), so sinc(beta/pi) = sin(beta)/beta
    intensity = np.sinc(beta / np.pi)**2
    return intensity


def plot_single_slit():
    """Plot single slit diffraction for different slit widths.

    Wider slits produce narrower central maxima, following the uncertainty
    principle: confining light to a smaller aperture increases its angular spread.
    """
    wavelength = 550e-9  # meters (green light)
    theta = np.linspace(-0.05, 0.05, 2000)  # radians

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: vary slit width ---
    for a_um in [10, 20, 50, 100]:
        a = a_um * 1e-6
        I = single_slit_intensity(theta, a, wavelength)
        ax1.plot(np.degrees(theta), I, label=f'a = {a_um} um')

    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Normalized intensity')
    ax1.set_title(f'Single Slit Diffraction (lambda = {wavelength*1e9:.0f} nm)')
    ax1.legend()
    ax1.set_xlim(-3, 3)
    ax1.grid(True, alpha=0.3)

    # --- Right: log scale to show secondary maxima ---
    a = 50e-6
    I = single_slit_intensity(theta, a, wavelength)
    ax2.semilogy(np.degrees(theta), I, 'b-', linewidth=1.5)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Intensity (log scale)')
    ax2.set_title(f'Single Slit (a = {a*1e6:.0f} um) - Log Scale')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(1e-6, 1.5)
    ax2.grid(True, alpha=0.3)

    # Mark first few minima
    for m in [1, 2, 3]:
        theta_min = np.arcsin(m * wavelength / a)
        ax2.axvline(np.degrees(theta_min), color='red', linestyle=':', alpha=0.5)
        ax2.axvline(-np.degrees(theta_min), color='red', linestyle=':', alpha=0.5)
        if m == 1:
            ax2.text(np.degrees(theta_min) + 0.05, 0.5, f'm={m}', fontsize=8, color='red')

    # Print first minimum angle
    theta_first_min = np.degrees(np.arcsin(wavelength / a))
    print(f"\n  Single slit (a={a*1e6:.0f} um):")
    print(f"  First minimum at theta = {theta_first_min:.4f} deg = {theta_first_min*60:.2f} arcmin")

    fig.tight_layout()
    fig.savefig('04_diffraction_single_slit.png', dpi=150)
    plt.close(fig)
    print("[Saved] 04_diffraction_single_slit.png")


# ---------------------------------------------------------------------------
# Circular aperture (Airy pattern)
# ---------------------------------------------------------------------------

def airy_intensity(theta: np.ndarray, D: float, wavelength: float) -> np.ndarray:
    """Compute the Airy diffraction pattern for a circular aperture.

    The intensity is proportional to [2*J1(x)/x]^2 where
    x = pi * D * sin(theta) / lambda.

    The Airy pattern determines the diffraction-limited resolution of
    telescopes and microscopes. The first dark ring (Airy disk radius)
    occurs at sin(theta) = 1.22 * lambda / D (Rayleigh criterion).

    Args:
        theta: Array of angles from the optical axis (radians).
        D: Aperture diameter.
        wavelength: Wavelength.

    Returns:
        Normalized intensity (peak = 1).
    """
    x = np.pi * D * np.sin(theta) / wavelength

    # Handle x=0 (central maximum): lim_{x->0} 2*J1(x)/x = 1
    intensity = np.ones_like(x)
    nonzero = np.abs(x) > 1e-10
    intensity[nonzero] = (2 * bessel_j1(x[nonzero]) / x[nonzero])**2
    return intensity


def airy_2d_pattern(N: int, D: float, wavelength: float, max_angle: float) -> np.ndarray:
    """Compute the 2D Airy pattern on a grid.

    Creates a 2D intensity map by computing the Airy function as a function
    of radial distance (the pattern is circularly symmetric).

    Args:
        N: Grid size (N x N pixels).
        D: Aperture diameter.
        wavelength: Wavelength.
        max_angle: Maximum angle from axis (radians).

    Returns:
        2D normalized intensity array.
    """
    theta_x = np.linspace(-max_angle, max_angle, N)
    Tx, Ty = np.meshgrid(theta_x, theta_x)
    theta_r = np.sqrt(Tx**2 + Ty**2)
    return airy_intensity(theta_r, D, wavelength)


def plot_airy_pattern():
    """Visualize the Airy diffraction pattern for a circular aperture.

    Shows both the 2D pattern and the radial intensity profile. Also
    demonstrates the Rayleigh criterion for resolving two point sources.
    """
    wavelength = 550e-9
    D = 10e-3  # 10 mm aperture

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: 2D Airy pattern ---
    ax = axes[0]
    max_angle = 3 * 1.22 * wavelength / D  # Show ~3 Airy rings
    pattern = airy_2d_pattern(500, D, wavelength, max_angle)

    # Use log scale for display to show rings clearly
    pattern_log = np.log10(pattern + 1e-8)
    ax.imshow(pattern_log,
              extent=[-max_angle * 1e6, max_angle * 1e6,
                      -max_angle * 1e6, max_angle * 1e6],
              cmap='inferno', vmin=-4, vmax=0)
    ax.set_xlabel('theta_x (urad)')
    ax.set_ylabel('theta_y (urad)')
    ax.set_title(f'Airy Pattern (D = {D*1e3:.0f} mm, log scale)')

    # --- Panel 2: Radial profile ---
    ax = axes[1]
    theta = np.linspace(0, max_angle, 1000)
    I = airy_intensity(theta, D, wavelength)

    # Airy disk radius (Rayleigh criterion)
    theta_airy = 1.22 * wavelength / D

    ax.plot(theta * 1e6, I, 'b-', linewidth=2)
    ax.axvline(theta_airy * 1e6, color='red', linestyle='--',
               label=f'Airy radius = {theta_airy*1e6:.2f} urad')
    ax.set_xlabel('Angle (urad)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title('Radial Profile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Rayleigh criterion ---
    ax = axes[2]
    # Two point sources separated by the Airy radius
    separations = [0.5, 1.0, 1.5]  # in units of theta_airy
    theta_range = np.linspace(-3 * theta_airy, 3 * theta_airy, 1000)

    for sep_factor in separations:
        sep = sep_factor * theta_airy
        # Sum intensity from two incoherent point sources
        I1 = airy_intensity(theta_range - sep / 2, D, wavelength)
        I2 = airy_intensity(theta_range + sep / 2, D, wavelength)
        I_total = I1 + I2
        I_total /= I_total.max()

        label = f'sep = {sep_factor:.1f} x theta_Airy'
        if sep_factor == 1.0:
            label += ' (Rayleigh)'
        ax.plot(theta_range / theta_airy, I_total, label=label, linewidth=1.5)

    ax.set_xlabel('Angle / theta_Airy')
    ax.set_ylabel('Total intensity (normalized)')
    ax.set_title('Rayleigh Resolution Criterion')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    print(f"\n  Circular aperture (D={D*1e3:.0f} mm):")
    print(f"  Airy disk angular radius = {theta_airy*1e6:.3f} urad = {np.degrees(theta_airy)*3600:.3f} arcsec")

    fig.tight_layout()
    fig.savefig('04_diffraction_airy.png', dpi=150)
    plt.close(fig)
    print("[Saved] 04_diffraction_airy.png")


# ---------------------------------------------------------------------------
# Diffraction grating
# ---------------------------------------------------------------------------

def n_slit_intensity(theta: np.ndarray, N: int, d: float, a: float,
                     wavelength: float) -> np.ndarray:
    """Compute the diffraction pattern of N equally spaced slits.

    The intensity combines the single-slit diffraction envelope with the
    N-slit interference pattern:

        I = I0 * [sin(beta)/beta]^2 * [sin(N*gamma)/sin(gamma)]^2 / N^2

    where:
        beta = pi * a * sin(theta) / lambda  (single slit diffraction)
        gamma = pi * d * sin(theta) / lambda (grating interference)

    The factor [sin(N*gamma)/sin(gamma)]^2 produces sharp principal maxima
    at d*sin(theta) = m*lambda, with N-2 subsidiary maxima between them.

    Args:
        theta: Angles from axis (radians).
        N: Number of slits.
        d: Slit spacing (center to center).
        a: Slit width.
        wavelength: Wavelength.

    Returns:
        Normalized intensity (peak = 1).
    """
    beta = np.pi * a * np.sin(theta) / wavelength
    gamma = np.pi * d * np.sin(theta) / wavelength

    # Single-slit envelope
    envelope = np.sinc(beta / np.pi)**2

    # N-slit interference factor
    # Handle gamma = m*pi (principal maxima) where sin(gamma) = 0
    numerator = np.sin(N * gamma)
    denominator = np.sin(gamma)

    # Use L'Hopital at singularities: sin(N*gamma)/sin(gamma) -> N at gamma = m*pi
    grating_factor = np.ones_like(theta) * N**2
    valid = np.abs(denominator) > 1e-12
    grating_factor[valid] = (numerator[valid] / denominator[valid])**2

    intensity = envelope * grating_factor / N**2
    return intensity


def plot_grating():
    """Visualize diffraction grating patterns and resolving power.

    Shows how the number of slits affects the sharpness of the diffraction maxima
    and the resolving power of the grating.
    """
    wavelength = 550e-9
    d = 2e-6  # 2 um slit spacing (500 lines/mm grating)
    a = 0.5e-6  # 0.5 um slit width

    theta = np.linspace(-0.3, 0.3, 10000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Vary number of slits ---
    ax = axes[0, 0]
    for N in [2, 5, 10, 50]:
        I = n_slit_intensity(theta, N, d, a, wavelength)
        ax.plot(np.degrees(theta), I, label=f'N = {N}', linewidth=1)
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title('Effect of Number of Slits')
    ax.legend()
    ax.set_xlim(-15, 15)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: High-resolution grating ---
    ax = axes[0, 1]
    N = 1000
    # Zoom into the first order
    theta_zoom = np.linspace(0.25, 0.30, 5000)
    I = n_slit_intensity(theta_zoom, N, d, a, wavelength)
    ax.plot(np.degrees(theta_zoom), I, 'b-', linewidth=1)

    # Show two close wavelengths (sodium doublet as example)
    wl1 = 589.0e-9
    wl2 = 589.6e-9
    I1 = n_slit_intensity(theta_zoom, N, d, a, wl1)
    I2 = n_slit_intensity(theta_zoom, N, d, a, wl2)
    ax.plot(np.degrees(theta_zoom), I1 * 0.5, 'r-', linewidth=1, alpha=0.7, label=f'{wl1*1e9:.1f} nm')
    ax.plot(np.degrees(theta_zoom), I2 * 0.5, 'g-', linewidth=1, alpha=0.7, label=f'{wl2*1e9:.1f} nm')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title(f'Grating Resolving Power (N={N}, d={d*1e6:.1f} um)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Resolving power: R = m * N (for order m)
    R = 1 * N  # first order
    min_delta_lambda = wavelength / R
    print(f"\n  Grating: N={N} slits, d={d*1e6:.1f} um")
    print(f"  Resolving power (1st order): R = {R}")
    print(f"  Minimum resolvable wavelength difference: {min_delta_lambda*1e9:.3f} nm")

    # --- Panel 3: Compare single, double, N-slit ---
    ax = axes[1, 0]
    theta_comp = np.linspace(-0.15, 0.15, 5000)

    # Single slit
    I_single = single_slit_intensity(theta_comp, a, wavelength)
    ax.plot(np.degrees(theta_comp), I_single, 'k--', linewidth=1, label='Single slit', alpha=0.5)

    # Double slit (N=2)
    I_double = n_slit_intensity(theta_comp, 2, d, a, wavelength)
    ax.plot(np.degrees(theta_comp), I_double, 'b-', linewidth=1, label='Double slit (N=2)')

    # N=10 slit
    I_10 = n_slit_intensity(theta_comp, 10, d, a, wavelength)
    ax.plot(np.degrees(theta_comp), I_10, 'r-', linewidth=1, label='N = 10 slits')

    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title('Single Slit vs Double Slit vs Grating')
    ax.legend(fontsize=8)
    ax.set_xlim(-8, 8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Grating equation and angular dispersion ---
    ax = axes[1, 1]
    # For a 600 lines/mm grating: d = 1/600 mm = 1.667 um
    d_grating = 1.0 / 600 * 1e-3  # meters
    wavelengths = np.linspace(380, 780, 200) * 1e-9

    for m in [1, 2, 3]:
        sin_theta = m * wavelengths / d_grating
        # Only plot where sin(theta) <= 1
        valid = np.abs(sin_theta) <= 1
        theta_diff = np.degrees(np.arcsin(sin_theta[valid]))
        ax.plot(wavelengths[valid] * 1e9, theta_diff, label=f'Order m={m}', linewidth=1.5)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Diffraction angle (degrees)')
    ax.set_title(f'Grating Equation (600 lines/mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Angular dispersion: d(theta)/d(lambda) = m / (d * cos(theta))
    theta_550 = np.arcsin(1 * 550e-9 / d_grating)
    ang_disp = 1 / (d_grating * np.cos(theta_550))  # rad/m
    print(f"  Angular dispersion (1st order, 550 nm): {ang_disp*1e-6:.1f} rad/um")

    fig.suptitle('Diffraction Gratings and N-Slit Patterns', fontsize=14)
    fig.tight_layout()
    fig.savefig('04_diffraction_grating.png', dpi=150)
    plt.close(fig)
    print("[Saved] 04_diffraction_grating.png")


# ---------------------------------------------------------------------------
# Comparison: Airy disk vs slit for different aperture sizes
# ---------------------------------------------------------------------------

def plot_resolution_comparison():
    """Compare diffraction-limited resolution for different aperture sizes.

    This demonstrates that larger apertures produce smaller diffraction spots,
    enabling higher spatial resolution -- the fundamental motivation for
    building large telescopes and using short wavelengths in microscopy.
    """
    wavelengths_nm = [400, 550, 700]
    diameters_mm = np.linspace(1, 100, 200)

    fig, ax = plt.subplots(figsize=(10, 6))

    for wl_nm in wavelengths_nm:
        wl = wl_nm * 1e-9
        # Rayleigh criterion: theta = 1.22 * lambda / D
        theta_airy = 1.22 * wl / (diameters_mm * 1e-3)
        theta_arcsec = np.degrees(theta_airy) * 3600
        ax.loglog(diameters_mm, theta_arcsec, label=f'{wl_nm} nm', linewidth=2)

    # Mark some reference points
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.text(1.5, 1.1, '1 arcsec', fontsize=8, color='gray')

    # Hubble (2.4 m)
    theta_hubble = np.degrees(1.22 * 550e-9 / 2.4) * 3600
    ax.plot(2400, theta_hubble, 'r*', markersize=15, label=f'Hubble (0.05 arcsec)')

    # Human eye (~6 mm pupil)
    theta_eye = np.degrees(1.22 * 550e-9 / 6e-3) * 3600
    ax.plot(6, theta_eye, 'go', markersize=10, label=f'Human eye (~1 arcmin)')

    ax.set_xlabel('Aperture diameter (mm)')
    ax.set_ylabel('Angular resolution (arcsec)')
    ax.set_title('Diffraction-Limited Angular Resolution')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig('04_diffraction_resolution.png', dpi=150)
    plt.close(fig)
    print("[Saved] 04_diffraction_resolution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Diffraction: Single Slit, Airy Pattern, and Gratings")
    print("=" * 60)

    print("\n--- Single Slit Diffraction ---")
    plot_single_slit()

    print("\n--- Circular Aperture (Airy Pattern) ---")
    plot_airy_pattern()

    print("\n--- Diffraction Grating ---")
    plot_grating()

    print("\n--- Resolution Comparison ---")
    plot_resolution_comparison()

    print("\nDone. All plots saved.")
