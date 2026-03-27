#!/usr/bin/env python3
"""Interference Phenomena: Double Slit, Thin Film, and Michelson Interferometer.

This module simulates three fundamental interference experiments:

1. Young's double-slit experiment — the canonical demonstration of wave
   interference. Two coherent point sources produce an intensity pattern
   governed by the path difference between the slits.

2. Thin-film interference — light reflected from the top and bottom
   surfaces of a thin dielectric layer interferes constructively or
   destructively depending on film thickness and wavelength. Used in
   anti-reflection coatings and decorative coatings.

3. Michelson interferometer — a beam-splitting device that creates
   circular fringe patterns. The fringe spacing depends on the optical
   path difference between the two arms.

Key physics:
- Constructive interference: path difference = m * lambda (integer m)
- Destructive interference: path difference = (m + 1/2) * lambda
- Phase shifts: reflection from a denser medium adds pi phase shift
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Young's double slit
# ---------------------------------------------------------------------------

def double_slit_intensity(y: np.ndarray, d: float, wavelength: float,
                          L: float, slit_width: float = 0.0) -> np.ndarray:
    """Compute the intensity pattern of Young's double-slit experiment.

    The intensity at position y on a screen at distance L from the slits is:
        I(y) = I0 * cos^2(pi * d * y / (lambda * L))

    When slit width a > 0, we include the single-slit envelope:
        I(y) = I0 * [sinc(a*y / (lambda*L))]^2 * cos^2(pi*d*y / (lambda*L))

    The sinc envelope modulates the double-slit fringes because each slit
    has finite width and produces its own diffraction pattern.

    Args:
        y: Screen positions (array).
        d: Slit separation (center to center).
        wavelength: Wavelength of light.
        L: Distance from slits to screen.
        slit_width: Width of each slit (0 for ideal point sources).

    Returns:
        Normalized intensity array.
    """
    # Double-slit interference term
    # Phase difference: delta = 2*pi*d*sin(theta)/lambda ≈ 2*pi*d*y/(lambda*L)
    phase = np.pi * d * y / (wavelength * L)
    interference = np.cos(phase)**2

    if slit_width > 0:
        # Single-slit diffraction envelope
        beta = np.pi * slit_width * y / (wavelength * L)
        # np.sinc(x) = sin(pi*x)/(pi*x), so we need sinc(beta/pi) = sinc(a*y/(lambda*L))
        envelope = np.sinc(beta / np.pi)**2
        return envelope * interference
    return interference


def plot_double_slit():
    """Visualize Young's double-slit experiment for varying parameters.

    We show how the fringe spacing changes with slit separation and wavelength,
    and how a finite slit width creates a diffraction envelope.
    """
    L = 1.0  # screen distance (meters)
    y = np.linspace(-0.01, 0.01, 2000)  # screen positions (meters)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Vary slit separation ---
    ax = axes[0, 0]
    wavelength = 550e-9  # green light
    for d_um in [50, 100, 200, 500]:
        d = d_um * 1e-6
        I = double_slit_intensity(y, d, wavelength, L)
        ax.plot(y * 1e3, I, label=f'd = {d_um} um')
    ax.set_xlabel('Screen position (mm)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title(f'Vary Slit Separation (lambda = {wavelength*1e9:.0f} nm)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Vary wavelength ---
    ax = axes[0, 1]
    d = 100e-6
    colors = ['violet', 'blue', 'green', 'red']
    for wl_nm, color in zip([400, 470, 550, 650], colors):
        wl = wl_nm * 1e-9
        I = double_slit_intensity(y, d, wl, L)
        ax.plot(y * 1e3, I, color=color, label=f'{wl_nm} nm')
    ax.set_xlabel('Screen position (mm)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title(f'Vary Wavelength (d = {d*1e6:.0f} um)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Effect of slit width (diffraction envelope) ---
    ax = axes[1, 0]
    d = 200e-6
    wavelength = 550e-9
    for a_um in [0, 10, 30, 50]:
        a = a_um * 1e-6
        I = double_slit_intensity(y, d, wavelength, L, slit_width=a)
        label = f'a = {a_um} um' if a_um > 0 else 'Point slits'
        ax.plot(y * 1e3, I, label=label)
    ax.set_xlabel('Screen position (mm)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title('Effect of Finite Slit Width (Diffraction Envelope)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Fringe spacing formula ---
    ax = axes[1, 1]
    # Fringe spacing: Dy = lambda * L / d
    d_range = np.linspace(20, 500, 200) * 1e-6
    for wl_nm in [400, 550, 700]:
        wl = wl_nm * 1e-9
        fringe_spacing = wl * L / d_range * 1e3  # in mm
        ax.plot(d_range * 1e6, fringe_spacing, label=f'{wl_nm} nm')
    ax.set_xlabel('Slit separation d (um)')
    ax.set_ylabel('Fringe spacing (mm)')
    ax.set_title('Fringe Spacing = lambda*L/d')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Print key results
    d = 100e-6
    wl = 550e-9
    fringe_sp = wl * L / d
    print(f"\n  Double slit: d={d*1e6:.0f} um, lambda={wl*1e9:.0f} nm, L={L:.1f} m")
    print(f"  Fringe spacing = {fringe_sp*1e3:.3f} mm")

    fig.suptitle("Young's Double-Slit Experiment", fontsize=14)
    fig.tight_layout()
    fig.savefig('03_interference_double_slit.png', dpi=150)
    plt.close(fig)
    print("[Saved] 03_interference_double_slit.png")


# ---------------------------------------------------------------------------
# Thin film interference
# ---------------------------------------------------------------------------

def thin_film_reflectance(wavelength: np.ndarray, thickness: float,
                          n_film: float, n_substrate: float = 1.52,
                          n_medium: float = 1.0) -> np.ndarray:
    """Compute reflectance of a single thin film on a substrate.

    Uses the Fresnel equations for normal incidence. The reflected amplitude
    from a thin film is the sum of reflections from both surfaces, including
    the phase accumulated during the round trip through the film.

    For normal incidence:
        r_total = (r12 + r23 * exp(-i*delta)) / (1 + r12*r23 * exp(-i*delta))

    where delta = 4*pi*n_film*t/lambda is the round-trip phase, and
    r12, r23 are the Fresnel reflection coefficients at each interface.

    The phase shift on reflection from a denser medium is automatically
    included in the sign of the Fresnel coefficient.

    Args:
        wavelength: Array of wavelengths (same units as thickness).
        thickness: Film thickness.
        n_film: Refractive index of the thin film.
        n_substrate: Refractive index of the substrate.
        n_medium: Refractive index of the incident medium (usually air).

    Returns:
        Reflectance R (fraction of reflected intensity) at each wavelength.
    """
    # Fresnel reflection coefficients (normal incidence, amplitude)
    r12 = (n_medium - n_film) / (n_medium + n_film)
    r23 = (n_film - n_substrate) / (n_film + n_substrate)

    # Round-trip phase in the film
    # Factor of 2 for the round trip, 2*pi/lambda for wave number
    delta = 4 * np.pi * n_film * thickness / wavelength

    # Total reflected amplitude (Airy formula for a single layer)
    r_total = (r12 + r23 * np.exp(-1j * delta)) / (1 + r12 * r23 * np.exp(-1j * delta))

    return np.abs(r_total)**2


def multilayer_reflectance(wavelength: np.ndarray,
                           layers: list[tuple[float, float]],
                           n_substrate: float, n_medium: float = 1.0) -> np.ndarray:
    """Compute reflectance of a multilayer thin film stack using transfer matrices.

    The transfer matrix method (TMM) is the standard approach for multilayer
    thin film calculations. Each layer is represented by a 2x2 matrix that
    relates the forward and backward propagating fields.

    Args:
        wavelength: Array of wavelengths.
        layers: List of (n, thickness) tuples for each layer, ordered from
                incident medium side to substrate side.
        n_substrate: Refractive index of the substrate.
        n_medium: Refractive index of the incident medium.

    Returns:
        Reflectance R at each wavelength.
    """
    R = np.zeros_like(wavelength, dtype=float)

    for i, wl in enumerate(wavelength):
        # Build the system transfer matrix as the product of layer matrices
        M = np.eye(2, dtype=complex)
        for n_layer, t_layer in layers:
            # Phase accumulated in this layer
            phi = 2 * np.pi * n_layer * t_layer / wl
            # Layer transfer matrix (normal incidence)
            layer_matrix = np.array([
                [np.cos(phi), -1j * np.sin(phi) / n_layer],
                [-1j * n_layer * np.sin(phi), np.cos(phi)]
            ])
            M = M @ layer_matrix

        # Extract reflectance from the transfer matrix
        # Using boundary conditions at medium and substrate
        # r = (M00*ns + M01*nm*ns - M10 - M11*nm) / (M00*ns + M01*nm*ns + M10 + M11*nm)
        # Simplified for normal incidence:
        num = M[0, 0] * n_substrate + M[0, 1] * n_medium * n_substrate - M[1, 0] - M[1, 1] * n_medium
        den = M[0, 0] * n_substrate + M[0, 1] * n_medium * n_substrate + M[1, 0] + M[1, 1] * n_medium
        r = num / den
        R[i] = np.abs(r)**2

    return R


def plot_thin_film():
    """Visualize thin film interference for single and multilayer coatings.

    Panel 1: Reflectance of a single MgF2 layer on glass vs film thickness
    Panel 2: Anti-reflection coating: reflectance vs wavelength for optimal thickness
    Panel 3: Multilayer high-reflector (quarter-wave stack)
    """
    wavelengths = np.linspace(300, 900, 1000)  # nm

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Single layer reflectance vs thickness ---
    ax = axes[0, 0]
    n_MgF2 = 1.38  # MgF2, common AR coating material
    n_glass = 1.52
    wl_center = 550  # nm
    thicknesses = np.linspace(0, 600, 500)

    R_vs_t = [thin_film_reflectance(np.array([wl_center]), t, n_MgF2, n_glass)[0]
              for t in thicknesses]
    ax.plot(thicknesses, R_vs_t, 'b-', linewidth=2)

    # Mark the quarter-wave thickness (minimum reflectance for AR)
    t_qw = wl_center / (4 * n_MgF2)
    R_min = thin_film_reflectance(np.array([wl_center]), t_qw, n_MgF2, n_glass)[0]
    ax.axvline(t_qw, color='red', linestyle='--', label=f't_QW = {t_qw:.1f} nm')
    ax.plot(t_qw, R_min, 'ro', markersize=8)
    ax.text(t_qw + 10, R_min + 0.005, f'R = {R_min:.4f}', fontsize=9)

    # Bare glass reflectance for comparison
    R_bare = ((1 - n_glass) / (1 + n_glass))**2
    ax.axhline(R_bare, color='gray', linestyle=':', label=f'Bare glass R = {R_bare:.4f}')

    ax.set_xlabel('Film thickness (nm)')
    ax.set_ylabel('Reflectance')
    ax.set_title(f'Single MgF2 Layer on Glass (lambda = {wl_center} nm)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: AR coating reflectance vs wavelength ---
    ax = axes[0, 1]
    t_ar = wl_center / (4 * n_MgF2)  # optimized for 550 nm
    R_ar = thin_film_reflectance(wavelengths, t_ar, n_MgF2, n_glass)
    ax.plot(wavelengths, R_ar * 100, 'b-', linewidth=2, label='MgF2 AR coating')
    ax.axhline(R_bare * 100, color='gray', linestyle=':', label='Bare glass')
    ax.fill_between([380, 780], 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5,
                     alpha=0.1, color='yellow', label='Visible range')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title(f'AR Coating: Quarter-Wave MgF2 (t = {t_ar:.1f} nm)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3)

    print(f"\n  AR coating (MgF2): t = {t_ar:.1f} nm")
    print(f"  R at {wl_center} nm = {R_min*100:.3f}% (bare glass = {R_bare*100:.2f}%)")

    # --- Panel 3: Multilayer high-reflector (quarter-wave stack) ---
    ax = axes[1, 0]
    n_H = 2.35  # TiO2 (high index)
    n_L = 1.38  # MgF2 (low index)
    t_H = wl_center / (4 * n_H)
    t_L = wl_center / (4 * n_L)

    # Build stacks of increasing layer count: (HL)^N H
    for N in [1, 3, 5, 7]:
        layers = []
        for _ in range(N):
            layers.append((n_H, t_H))
            layers.append((n_L, t_L))
        layers.append((n_H, t_H))  # End with H layer for maximum reflectance

        R_ml = multilayer_reflectance(wavelengths, layers, n_glass)
        ax.plot(wavelengths, R_ml * 100, label=f'(HL)^{N}H ({2*N+1} layers)')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title('Quarter-Wave Stack High Reflector (TiO2/MgF2)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Color of thin films (reflectance spectrum -> perceived color) ---
    ax = axes[1, 1]
    # Show reflectance spectra for different film thicknesses (like soap bubble colors)
    n_soap = 1.33  # soap film in air
    thicknesses_soap = [100, 200, 300, 400, 500]
    for t in thicknesses_soap:
        R = thin_film_reflectance(wavelengths, t, n_soap, n_substrate=1.0, n_medium=1.0)
        ax.plot(wavelengths, R * 100, label=f't = {t} nm')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%)')
    ax.set_title('Soap Film Interference (varying thickness)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Thin Film Interference', fontsize=14)
    fig.tight_layout()
    fig.savefig('03_interference_thin_film.png', dpi=150)
    plt.close(fig)
    print("[Saved] 03_interference_thin_film.png")


# ---------------------------------------------------------------------------
# Michelson interferometer
# ---------------------------------------------------------------------------

def michelson_fringe_pattern(N: int, delta_d: float, wavelength: float,
                             beam_radius: float) -> np.ndarray:
    """Compute the Michelson interferometer fringe pattern.

    The Michelson interferometer produces circular fringes (fringes of equal
    inclination) when the mirrors are slightly tilted, or concentric rings
    when the path difference varies with angle of incidence.

    For a collimated beam with a path difference 2*delta_d between the arms,
    the intensity at angle theta from the axis is:
        I(theta) = I0 * [1 + cos(2*pi * 2*delta_d * cos(theta) / lambda)]

    For small angles (paraxial), theta ≈ r/f where r is the radial distance
    on the screen and f is the focal length of the imaging lens.

    Args:
        N: Grid size (N x N pixels).
        delta_d: Mirror displacement from balanced position.
        wavelength: Wavelength of the light source.
        beam_radius: Radius of the collimated beam.

    Returns:
        2D intensity array.
    """
    # Create a grid of angles (paraxial approximation)
    x = np.linspace(-beam_radius, beam_radius, N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)

    # Angle from axis: theta ≈ r / focal_length
    # We normalize so the edge of the beam corresponds to a small angle
    f_eff = beam_radius / 0.02  # effective focal length (max angle ~20 mrad)
    theta = R / f_eff

    # Phase difference between the two arms
    # The factor of 2 accounts for the round trip
    phase = 2 * np.pi * 2 * delta_d * np.cos(theta) / wavelength

    # Intensity: two-beam interference
    intensity = 0.5 * (1 + np.cos(phase))

    # Apply circular aperture (beam profile)
    aperture = R <= beam_radius
    intensity *= aperture

    return intensity


def plot_michelson():
    """Visualize Michelson interferometer fringe patterns.

    Show how the fringe pattern changes as the mirror displacement varies.
    Also show the intensity profile along the diameter.
    """
    wavelength = 632.8e-9  # HeNe laser (meters)
    beam_radius = 5e-3  # 5 mm beam
    N = 500

    # Different mirror displacements to show fringe evolution
    displacements_um = [0, 0.5, 1.0, 2.0, 5.0, 10.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, delta_um in zip(axes.flat, displacements_um):
        delta_d = delta_um * 1e-6
        pattern = michelson_fringe_pattern(N, delta_d, wavelength, beam_radius)

        ax.imshow(pattern, extent=[-beam_radius * 1e3, beam_radius * 1e3,
                                    -beam_radius * 1e3, beam_radius * 1e3],
                   cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'delta_d = {delta_um} um')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')

    fig.suptitle(f'Michelson Interferometer Fringes (lambda = {wavelength*1e9:.1f} nm)', fontsize=14)
    fig.tight_layout()
    fig.savefig('03_interference_michelson.png', dpi=150)
    plt.close(fig)
    print("[Saved] 03_interference_michelson.png")

    # Print fringe count
    for delta_um in displacements_um:
        delta_d = delta_um * 1e-6
        n_fringes = 2 * delta_d / wavelength
        print(f"  delta_d = {delta_um:.1f} um -> {n_fringes:.1f} fringes")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Interference: Double Slit, Thin Film, Michelson")
    print("=" * 60)

    print("\n--- Young's Double-Slit Experiment ---")
    plot_double_slit()

    print("\n--- Thin Film Interference ---")
    plot_thin_film()

    print("\n--- Michelson Interferometer ---")
    plot_michelson()

    print("\nDone. All plots saved.")
