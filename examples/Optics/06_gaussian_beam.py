#!/usr/bin/env python3
"""Gaussian Beam Propagation: Waist, Rayleigh Range, ABCD Matrix, and M-squared.

This module explores the properties and propagation of Gaussian laser beams:

1. Beam waist, Rayleigh range, and divergence — the fundamental parameters
2. ABCD ray transfer matrix method for beam propagation through optical systems
3. Beam quality factor M^2 — quantifying deviation from an ideal Gaussian
4. Focused beam spot size — optimizing lens systems for tight focusing

Physics background:
- A Gaussian beam is the fundamental mode (TEM00) of a laser cavity.
  Its transverse intensity profile is a Gaussian function.
- The beam radius w(z) expands as: w(z) = w0 * sqrt(1 + (z/z_R)^2)
  where w0 is the beam waist and z_R = pi * w0^2 / lambda is the Rayleigh range.
- The ABCD matrix method extends ray transfer matrices to Gaussian beams
  by tracking the complex beam parameter q(z) = z + i*z_R.
  After propagation through an ABCD system: q' = (A*q + B) / (C*q + D).
- The M^2 factor describes real beams: a beam with M^2 > 1 diverges M^2
  times faster than the ideal diffraction limit.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Gaussian beam parameters
# ---------------------------------------------------------------------------

class GaussianBeam:
    """Represents a Gaussian beam with its fundamental parameters.

    The complex beam parameter q encodes both the beam radius and wavefront
    curvature at any position z:
        1/q(z) = 1/R(z) - i*lambda/(pi*w(z)^2)

    where R(z) is the radius of curvature and w(z) is the beam radius.

    Attributes:
        wavelength: Wavelength of the beam.
        w0: Beam waist radius (1/e^2 intensity).
        z_R: Rayleigh range.
        theta_div: Far-field half-angle divergence.
        M2: Beam quality factor.
    """

    def __init__(self, wavelength: float, w0: float, M2: float = 1.0):
        """Initialize Gaussian beam from wavelength and beam waist.

        Args:
            wavelength: Wavelength in the same units as w0.
            w0: Beam waist (minimum radius, 1/e^2 intensity).
            M2: Beam quality factor (1.0 for ideal TEM00).
        """
        self.wavelength = wavelength
        self.w0 = w0
        self.M2 = M2

        # Rayleigh range: distance from waist where beam area doubles
        # For real beams with M^2 > 1, z_R is reduced by M^2
        self.z_R = np.pi * w0**2 / (M2 * wavelength)

        # Far-field divergence half-angle (small angle approximation)
        # theta = M^2 * lambda / (pi * w0)
        self.theta_div = M2 * wavelength / (np.pi * w0)

    def beam_radius(self, z: np.ndarray) -> np.ndarray:
        """Beam radius w(z) at position z (measured from waist).

        The beam expands hyperbolically from the waist. At z = z_R,
        the beam radius is sqrt(2) * w0.

        Args:
            z: Array of positions along the propagation axis.

        Returns:
            Beam radius at each position.
        """
        return self.w0 * np.sqrt(1 + (z / self.z_R)**2)

    def wavefront_curvature(self, z: np.ndarray) -> np.ndarray:
        """Radius of curvature R(z) of the wavefront.

        R is infinite at the waist (planar) and at z >> z_R (planar far field).
        Minimum R occurs at z = z_R.

        Args:
            z: Positions along the axis (avoid z=0).

        Returns:
            Radius of curvature at each position.
        """
        # Avoid division by zero at the waist
        z_safe = np.where(np.abs(z) < 1e-15, 1e-15, z)
        return z_safe * (1 + (self.z_R / z_safe)**2)

    def gouy_phase(self, z: np.ndarray) -> np.ndarray:
        """Gouy phase shift at position z.

        The Gouy phase is an extra phase shift that a Gaussian beam acquires
        relative to a plane wave. It changes by pi/2 from the waist to the
        far field, and by pi total through a focus.

        Args:
            z: Positions along the axis.

        Returns:
            Gouy phase in radians.
        """
        return np.arctan(z / self.z_R)

    def complex_beam_parameter(self, z: float) -> complex:
        """Complex beam parameter q at position z.

        q(z) = z + i*z_R (at the waist, q = i*z_R).

        Args:
            z: Position along the axis.

        Returns:
            Complex beam parameter.
        """
        return z + 1j * self.z_R

    def intensity_profile(self, r: np.ndarray, z: float) -> np.ndarray:
        """Transverse intensity profile at position z.

        I(r, z) = (w0/w(z))^2 * exp(-2*r^2/w(z)^2)

        The factor (w0/w)^2 ensures energy conservation as the beam expands.

        Args:
            r: Radial distances from beam center.
            z: Position along the axis.

        Returns:
            Normalized intensity (peak at waist = 1).
        """
        w = self.beam_radius(np.array([z]))[0]
        return (self.w0 / w)**2 * np.exp(-2 * r**2 / w**2)


# ---------------------------------------------------------------------------
# ABCD matrix utilities
# ---------------------------------------------------------------------------

def abcd_free_space(d: float) -> np.ndarray:
    """ABCD matrix for free-space propagation over distance d.

    A ray at height y with slope y' maps to:
        y_new = y + d * y'
        y'_new = y'
    """
    return np.array([[1, d], [0, 1]], dtype=float)


def abcd_thin_lens(f: float) -> np.ndarray:
    """ABCD matrix for a thin lens with focal length f.

    A lens changes the ray slope without changing its height:
        y_new = y
        y'_new = y' - y/f
    """
    return np.array([[1, 0], [-1/f, 1]], dtype=float)


def abcd_curved_mirror(R: float) -> np.ndarray:
    """ABCD matrix for a curved mirror with radius of curvature R.

    Equivalent to a thin lens with f = R/2 (in reflection).
    """
    return np.array([[1, 0], [-2/R, 1]], dtype=float)


def propagate_q(q: complex, abcd: np.ndarray) -> complex:
    """Propagate the complex beam parameter through an ABCD system.

    The ABCD law for Gaussian beams:
        q' = (A*q + B) / (C*q + D)

    This is the central result connecting ray optics (ABCD matrices) to
    Gaussian beam optics. It allows us to trace a beam through any
    paraxial optical system.

    Args:
        q: Complex beam parameter at the input.
        abcd: 2x2 ABCD matrix.

    Returns:
        Complex beam parameter at the output.
    """
    A, B = abcd[0, 0], abcd[0, 1]
    C, D = abcd[1, 0], abcd[1, 1]
    return (A * q + B) / (C * q + D)


def q_to_beam_params(q: complex, wavelength: float) -> tuple[float, float]:
    """Extract beam radius and curvature from complex beam parameter.

    From 1/q = 1/R - i*lambda/(pi*w^2), we get:
        R = 1 / Re(1/q)
        w = sqrt(-lambda / (pi * Im(1/q)))

    Args:
        q: Complex beam parameter.
        wavelength: Wavelength.

    Returns:
        Tuple of (beam radius w, wavefront curvature R).
    """
    inv_q = 1 / q
    R = 1 / np.real(inv_q) if abs(np.real(inv_q)) > 1e-15 else float('inf')
    w = np.sqrt(-wavelength / (np.pi * np.imag(inv_q)))
    return w, R


# ---------------------------------------------------------------------------
# Visualization 1: Gaussian beam fundamentals
# ---------------------------------------------------------------------------

def plot_beam_propagation():
    """Visualize Gaussian beam propagation showing beam profile, divergence, and Gouy phase."""
    wavelength = 1064e-9  # Nd:YAG laser (meters)
    w0 = 0.5e-3  # 0.5 mm beam waist

    beam = GaussianBeam(wavelength, w0)
    print(f"\n  Beam parameters (lambda={wavelength*1e9:.0f} nm, w0={w0*1e3:.1f} mm):")
    print(f"  Rayleigh range z_R = {beam.z_R:.4f} m = {beam.z_R*100:.1f} cm")
    print(f"  Far-field divergence = {beam.theta_div*1e3:.3f} mrad")
    print(f"  Beam parameter product w0*theta = {w0 * beam.theta_div * 1e6:.3f} um*mrad")

    z = np.linspace(-3 * beam.z_R, 3 * beam.z_R, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Beam envelope ---
    ax = axes[0, 0]
    w = beam.beam_radius(z)
    ax.fill_between(z * 100, -w * 1e3, w * 1e3, alpha=0.2, color='red')
    ax.plot(z * 100, w * 1e3, 'r-', linewidth=2, label='w(z)')
    ax.plot(z * 100, -w * 1e3, 'r-', linewidth=2)

    # Asymptotic divergence lines
    z_far = np.array([0, z.max()])
    w_asymp = beam.theta_div * z_far
    ax.plot(z_far * 100, w_asymp * 1e3, 'b--', linewidth=1, alpha=0.5, label='Asymptote')
    ax.plot(z_far * 100, -w_asymp * 1e3, 'b--', linewidth=1, alpha=0.5)

    # Mark Rayleigh range
    w_zR = beam.beam_radius(np.array([beam.z_R]))[0]
    ax.axvline(beam.z_R * 100, color='green', linestyle=':', alpha=0.5,
               label=f'z_R = {beam.z_R*100:.1f} cm')
    ax.axvline(-beam.z_R * 100, color='green', linestyle=':', alpha=0.5)

    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Beam radius (mm)')
    ax.set_title('Gaussian Beam Envelope')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Intensity profiles at different z ---
    ax = axes[0, 1]
    r = np.linspace(-3 * w0, 3 * w0, 500)
    z_positions = [0, 0.5 * beam.z_R, beam.z_R, 2 * beam.z_R]
    for z_pos in z_positions:
        I = beam.intensity_profile(r, z_pos)
        label = f'z = {z_pos/beam.z_R:.1f} z_R'
        ax.plot(r * 1e3, I, linewidth=1.5, label=label)

    ax.set_xlabel('Radial position r (mm)')
    ax.set_ylabel('Intensity (normalized)')
    ax.set_title('Transverse Intensity Profiles')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Wavefront curvature ---
    ax = axes[1, 0]
    # Avoid z=0 (infinite R)
    z_curv = np.concatenate([np.linspace(-3 * beam.z_R, -0.01 * beam.z_R, 300),
                              np.linspace(0.01 * beam.z_R, 3 * beam.z_R, 300)])
    R_curv = beam.wavefront_curvature(z_curv)
    ax.plot(z_curv / beam.z_R, R_curv / beam.z_R, 'b-', linewidth=2)

    # Minimum R at z = z_R
    R_min = beam.wavefront_curvature(np.array([beam.z_R]))[0]
    ax.plot(1, R_min / beam.z_R, 'ro', markersize=8, label=f'Min R at z_R = {R_min/beam.z_R:.1f} z_R')

    ax.set_xlabel('z / z_R')
    ax.set_ylabel('R / z_R')
    ax.set_title('Wavefront Radius of Curvature')
    ax.set_ylim(-20, 20)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Gouy phase ---
    ax = axes[1, 1]
    gouy = beam.gouy_phase(z)
    ax.plot(z / beam.z_R, np.degrees(gouy), 'purple', linewidth=2)
    ax.axhline(90, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(-90, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('z / z_R')
    ax.set_ylabel('Gouy phase (degrees)')
    ax.set_title('Gouy Phase Shift')
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.9, 'Total shift through focus = 180 deg',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat'))

    fig.suptitle(f'Gaussian Beam (lambda={wavelength*1e9:.0f} nm, w0={w0*1e3:.1f} mm)', fontsize=14)
    fig.tight_layout()
    fig.savefig('06_gaussian_beam_propagation.png', dpi=150)
    plt.close(fig)
    print("[Saved] 06_gaussian_beam_propagation.png")


# ---------------------------------------------------------------------------
# Visualization 2: ABCD matrix beam tracing
# ---------------------------------------------------------------------------

def plot_abcd_beam_tracing():
    """Trace a Gaussian beam through a two-lens optical system using ABCD matrices.

    System: [waist] --d1--> [lens f1] --d2--> [lens f2] --d3--> [screen]

    This demonstrates how the ABCD method naturally handles the beam parameter
    transformation through multiple elements.
    """
    wavelength = 632.8e-9  # HeNe laser
    w0 = 0.3e-3  # 300 um waist at the start

    # Optical system: beam expander (f1=50mm, f2=200mm)
    f1 = 50e-3  # 50 mm
    f2 = 200e-3  # 200 mm
    d_between = f1 + f2  # Telescopic configuration (afocal)
    d_before = 100e-3  # 100 mm before first lens
    d_after = 300e-3  # 300 mm after second lens

    beam = GaussianBeam(wavelength, w0)
    q0 = beam.complex_beam_parameter(0)  # Start at waist

    # Trace beam through the system at many z positions
    n_steps = 2000
    z_total = d_before + d_between + d_after
    z_positions = np.linspace(0, z_total, n_steps)

    w_trace = []
    R_trace = []

    for z in z_positions:
        if z <= d_before:
            # Free space before first lens
            M = abcd_free_space(z)
        elif z <= d_before + d_between:
            # Through first lens + some free space
            dz = z - d_before
            M = abcd_free_space(dz) @ abcd_thin_lens(f1) @ abcd_free_space(d_before)
        else:
            # Through both lenses + remaining free space
            dz = z - d_before - d_between
            M = (abcd_free_space(dz) @ abcd_thin_lens(f2) @
                 abcd_free_space(d_between) @ abcd_thin_lens(f1) @
                 abcd_free_space(d_before))

        q = propagate_q(q0, M)
        w, R = q_to_beam_params(q, wavelength)
        w_trace.append(w)
        R_trace.append(R)

    w_trace = np.array(w_trace)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # --- Beam radius through system ---
    ax1.fill_between(z_positions * 1e3, -w_trace * 1e3, w_trace * 1e3,
                      alpha=0.2, color='red')
    ax1.plot(z_positions * 1e3, w_trace * 1e3, 'r-', linewidth=1.5)
    ax1.plot(z_positions * 1e3, -w_trace * 1e3, 'r-', linewidth=1.5)

    # Mark lens positions
    lens1_z = d_before * 1e3
    lens2_z = (d_before + d_between) * 1e3
    ax1.axvline(lens1_z, color='blue', linewidth=2, label=f'Lens 1 (f={f1*1e3:.0f} mm)')
    ax1.axvline(lens2_z, color='green', linewidth=2, label=f'Lens 2 (f={f2*1e3:.0f} mm)')

    ax1.set_xlabel('z (mm)')
    ax1.set_ylabel('Beam radius (mm)')
    ax1.set_title('Gaussian Beam Through Beam Expander (ABCD Method)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Print beam magnification
    w_input = w_trace[0]
    w_output = w_trace[-1]
    magnification = w_output / w_input
    print(f"\n  Beam expander: f1={f1*1e3:.0f} mm, f2={f2*1e3:.0f} mm")
    print(f"  Input beam radius: {w_input*1e3:.3f} mm")
    print(f"  Output beam radius: {w_output*1e3:.3f} mm")
    print(f"  Magnification: {magnification:.2f}x (expected: {f2/f1:.2f}x)")

    # --- Different input waists ---
    ax2 = ax2
    for w0_mm in [0.1, 0.3, 0.5, 1.0]:
        w0_test = w0_mm * 1e-3
        beam_test = GaussianBeam(wavelength, w0_test)
        q0_test = beam_test.complex_beam_parameter(0)

        w_test = []
        for z in z_positions:
            if z <= d_before:
                M = abcd_free_space(z)
            elif z <= d_before + d_between:
                M = (abcd_free_space(z - d_before) @ abcd_thin_lens(f1) @
                     abcd_free_space(d_before))
            else:
                M = (abcd_free_space(z - d_before - d_between) @ abcd_thin_lens(f2) @
                     abcd_free_space(d_between) @ abcd_thin_lens(f1) @
                     abcd_free_space(d_before))
            q = propagate_q(q0_test, M)
            w, _ = q_to_beam_params(q, wavelength)
            w_test.append(w)

        ax2.plot(z_positions * 1e3, np.array(w_test) * 1e3,
                 label=f'w0 = {w0_mm} mm')

    ax2.axvline(lens1_z, color='blue', linewidth=1.5, alpha=0.5)
    ax2.axvline(lens2_z, color='green', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('z (mm)')
    ax2.set_ylabel('Beam radius (mm)')
    ax2.set_title('Different Input Beam Waists')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('06_gaussian_beam_abcd.png', dpi=150)
    plt.close(fig)
    print("[Saved] 06_gaussian_beam_abcd.png")


# ---------------------------------------------------------------------------
# Visualization 3: M-squared and focused spot size
# ---------------------------------------------------------------------------

def plot_m_squared_and_focusing():
    """Demonstrate M^2 beam quality and its effect on focused spot size.

    A real laser beam with M^2 > 1 cannot be focused as tightly as an ideal
    Gaussian beam. The minimum focused spot size is:
        w_focus = M^2 * lambda * f / (pi * w_input)

    This is M^2 times larger than the diffraction limit.
    """
    wavelength = 1064e-9
    w0 = 1e-3  # 1 mm input beam

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Beam divergence for different M^2 ---
    z = np.linspace(0, 2, 1000)  # meters

    for M2 in [1.0, 1.2, 1.5, 2.0, 3.0]:
        beam = GaussianBeam(wavelength, w0, M2=M2)
        w = beam.beam_radius(z)
        ax1.plot(z * 100, w * 1e3, label=f'M^2 = {M2:.1f}', linewidth=1.5)

    ax1.set_xlabel('Distance z (cm)')
    ax1.set_ylabel('Beam radius (mm)')
    ax1.set_title('Beam Divergence for Different M^2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Focused spot size ---
    f_lens = 100e-3  # 100 mm focusing lens
    M2_range = np.linspace(1, 5, 100)

    # Minimum focused spot radius: w_focus = M^2 * lambda * f / (pi * w_input)
    # This comes from the ABCD analysis of a lens at distance f from the waist
    w_focus = M2_range * wavelength * f_lens / (np.pi * w0)

    ax2.plot(M2_range, w_focus * 1e6, 'b-', linewidth=2)
    ax2.set_xlabel('M^2')
    ax2.set_ylabel('Focused spot radius (um)')
    ax2.set_title(f'Focused Spot Size (f={f_lens*1e3:.0f} mm, w_in={w0*1e3:.0f} mm)')
    ax2.grid(True, alpha=0.3)

    # Mark diffraction limit
    w_dl = wavelength * f_lens / (np.pi * w0)
    ax2.axhline(w_dl * 1e6, color='red', linestyle='--',
                label=f'Diffraction limit = {w_dl*1e6:.1f} um')
    ax2.legend()

    # Also show as a function of input beam size
    w_input_range = np.linspace(0.2, 5, 200) * 1e-3
    w_foc_vs_input = wavelength * f_lens / (np.pi * w_input_range)

    # Print key values
    print(f"\n  Focusing: f={f_lens*1e3:.0f} mm, w_input={w0*1e3:.1f} mm")
    print(f"  Diffraction-limited spot (M^2=1): {w_dl*1e6:.2f} um")
    print(f"  f-number: f/# = {f_lens / (2 * w0):.1f}")

    # Depth of focus (confocal parameter of the focused beam)
    z_R_focus = np.pi * w_dl**2 / wavelength
    dof = 2 * z_R_focus
    print(f"  Depth of focus (2*z_R): {dof*1e6:.1f} um")

    fig.tight_layout()
    fig.savefig('06_gaussian_beam_m2.png', dpi=150)
    plt.close(fig)
    print("[Saved] 06_gaussian_beam_m2.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Gaussian Beam: Propagation, ABCD Matrix, and M-squared")
    print("=" * 60)

    print("\n--- Beam Propagation Fundamentals ---")
    plot_beam_propagation()

    print("\n--- ABCD Matrix Beam Tracing ---")
    plot_abcd_beam_tracing()

    print("\n--- M^2 and Focused Spot Size ---")
    plot_m_squared_and_focusing()

    print("\nDone. All plots saved.")
