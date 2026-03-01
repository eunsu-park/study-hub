#!/usr/bin/env python3
"""Polarization: Jones Calculus, Malus's Law, Wave Plates, and Brewster Angle.

This module explores the polarization of light using the Jones vector formalism:

1. Jones vectors and matrices — representing polarization states and optical elements
2. Malus's law — intensity transmitted through crossed polarizers
3. Wave plates — quarter-wave and half-wave plates that transform polarization states
4. Polarization ellipse — visualizing the electric field trajectory
5. Brewster angle — the angle at which reflected light is perfectly polarized

Physics background:
- Light is a transverse electromagnetic wave. The electric field oscillates
  perpendicular to the propagation direction, and the direction of oscillation
  defines the polarization state.
- The Jones vector [Ex; Ey] represents a fully polarized plane wave.
  Each component is complex-valued to encode both amplitude and phase.
- Optical elements (polarizers, wave plates, rotators) are represented
  by 2x2 Jones matrices that act on the Jones vector.
- Malus's law: I = I0 * cos^2(theta) for light passing through two polarizers
  whose transmission axes are at angle theta.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# ---------------------------------------------------------------------------
# Jones vectors for common polarization states
# ---------------------------------------------------------------------------

def jones_horizontal() -> np.ndarray:
    """Horizontal linear polarization: E-field along x-axis."""
    return np.array([1, 0], dtype=complex)


def jones_vertical() -> np.ndarray:
    """Vertical linear polarization: E-field along y-axis."""
    return np.array([0, 1], dtype=complex)


def jones_linear(theta: float) -> np.ndarray:
    """Linear polarization at angle theta from horizontal.

    Args:
        theta: Polarization angle in radians.

    Returns:
        Normalized Jones vector.
    """
    return np.array([np.cos(theta), np.sin(theta)], dtype=complex)


def jones_rcp() -> np.ndarray:
    """Right circular polarization.

    The y-component leads the x-component by pi/2 (convention: exp(-i*omega*t)).
    The E-field vector rotates clockwise when viewed along the propagation direction.
    """
    return np.array([1, -1j], dtype=complex) / np.sqrt(2)


def jones_lcp() -> np.ndarray:
    """Left circular polarization.

    The y-component lags the x-component by pi/2.
    The E-field vector rotates counter-clockwise.
    """
    return np.array([1, 1j], dtype=complex) / np.sqrt(2)


def jones_elliptical(a: float, b: float, theta: float, handedness: int = 1) -> np.ndarray:
    """General elliptical polarization.

    Args:
        a: Semi-major axis amplitude.
        b: Semi-minor axis amplitude.
        theta: Orientation angle of the ellipse (radians).
        handedness: +1 for right-handed, -1 for left-handed.

    Returns:
        Normalized Jones vector.
    """
    # Build the ellipse in the rotated frame, then rotate
    phase = handedness * np.arctan2(b, a)
    Ex = a * np.cos(theta) - 1j * handedness * b * np.sin(theta)
    Ey = a * np.sin(theta) + 1j * handedness * b * np.cos(theta)
    J = np.array([Ex, Ey], dtype=complex)
    return J / np.linalg.norm(J)


# ---------------------------------------------------------------------------
# Jones matrices for optical elements
# ---------------------------------------------------------------------------

def jones_matrix_polarizer(theta: float) -> np.ndarray:
    """Jones matrix for a linear polarizer with transmission axis at angle theta.

    A polarizer transmits only the component of E along its axis.
    The matrix is the outer product of the unit vector along the axis.

    Args:
        theta: Angle of transmission axis from horizontal (radians).

    Returns:
        2x2 Jones matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c**2, c * s],
                     [c * s, s**2]], dtype=complex)


def jones_matrix_hwp(theta: float) -> np.ndarray:
    """Jones matrix for a half-wave plate with fast axis at angle theta.

    A half-wave plate introduces a phase retardation of pi between the
    fast and slow axis components. It rotates the polarization direction
    by 2*theta (if input is linear polarization) and converts between
    left and right circular polarization.

    Args:
        theta: Angle of fast axis from horizontal (radians).

    Returns:
        2x2 Jones matrix.
    """
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)
    return np.array([[c2, s2],
                     [s2, -c2]], dtype=complex)


def jones_matrix_qwp(theta: float) -> np.ndarray:
    """Jones matrix for a quarter-wave plate with fast axis at angle theta.

    A quarter-wave plate introduces a phase retardation of pi/2 between
    the fast and slow axis components. It converts linear polarization
    to circular (and vice versa) when the input is at 45 degrees to the axes.

    Args:
        theta: Angle of fast axis from horizontal (radians).

    Returns:
        2x2 Jones matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    # General QWP matrix: rotation to fast-axis frame, retardation, rotate back
    # M = R(-theta) * diag(1, exp(i*pi/2)) * R(theta)
    phase = np.exp(1j * np.pi / 2)
    return np.array([
        [c**2 + phase * s**2, (1 - phase) * c * s],
        [(1 - phase) * c * s, s**2 + phase * c**2]
    ], dtype=complex)


def jones_matrix_retarder(theta: float, delta: float) -> np.ndarray:
    """General Jones matrix for a wave plate with retardation delta.

    Args:
        theta: Fast axis angle from horizontal (radians).
        delta: Phase retardation between fast and slow axes (radians).

    Returns:
        2x2 Jones matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    phase = np.exp(1j * delta)
    return np.array([
        [c**2 + phase * s**2, (1 - phase) * c * s],
        [(1 - phase) * c * s, s**2 + phase * c**2]
    ], dtype=complex)


# ---------------------------------------------------------------------------
# Polarization ellipse computation
# ---------------------------------------------------------------------------

def compute_polarization_ellipse(jones: np.ndarray, n_points: int = 200) -> tuple:
    """Compute the electric field trajectory (polarization ellipse).

    The time evolution of the real E-field traces out an ellipse:
        Ex(t) = Re[Jx * exp(-i*omega*t)]
        Ey(t) = Re[Jy * exp(-i*omega*t)]

    Args:
        jones: Jones vector [Jx, Jy].
        n_points: Number of points along the trajectory.

    Returns:
        Tuple of (Ex_array, Ey_array).
    """
    t = np.linspace(0, 2 * np.pi, n_points)
    # E(t) = Re[J * exp(-i*omega*t)] — we use omega*t as the parameter
    Ex = np.real(jones[0] * np.exp(-1j * t))
    Ey = np.real(jones[1] * np.exp(-1j * t))
    return Ex, Ey


def polarization_ellipse_params(jones: np.ndarray) -> dict:
    """Extract ellipse parameters from a Jones vector.

    Returns the semi-axes, orientation angle, and ellipticity.

    Args:
        jones: Jones vector.

    Returns:
        Dictionary with 'a' (semi-major), 'b' (semi-minor),
        'theta' (orientation), 'ellipticity' (b/a), 'handedness' (R or L).
    """
    Ex, Ey = compute_polarization_ellipse(jones, 1000)

    # Compute via the Stokes parameters
    S0 = np.abs(jones[0])**2 + np.abs(jones[1])**2
    S1 = np.abs(jones[0])**2 - np.abs(jones[1])**2
    S2 = 2 * np.real(jones[0] * np.conj(jones[1]))
    S3 = 2 * np.imag(jones[0] * np.conj(jones[1]))

    # Orientation angle
    psi = 0.5 * np.arctan2(S2, S1)

    # Ellipticity angle
    chi = 0.5 * np.arcsin(S3 / S0)

    # Semi-axes
    a = np.sqrt(S0) * np.cos(np.abs(chi)) if np.abs(chi) < np.pi / 4 else np.sqrt(S0)
    b = np.sqrt(S0) * np.abs(np.sin(chi))

    handedness = 'R' if S3 < 0 else 'L'  # Convention dependent

    return {
        'a': a,
        'b': b,
        'theta': psi,
        'ellipticity': b / a if a > 0 else 0,
        'handedness': handedness,
        'S0': S0, 'S1': S1, 'S2': S2, 'S3': S3,
    }


# ---------------------------------------------------------------------------
# Visualization 1: Polarization states and ellipses
# ---------------------------------------------------------------------------

def plot_polarization_states():
    """Visualize common polarization states as electric field trajectories.

    Shows horizontal, vertical, 45-degree linear, RCP, LCP, and general
    elliptical polarization.
    """
    states = [
        (jones_horizontal(), 'Horizontal'),
        (jones_vertical(), 'Vertical'),
        (jones_linear(np.pi / 4), 'Linear 45 deg'),
        (jones_rcp(), 'Right Circular'),
        (jones_lcp(), 'Left Circular'),
        (jones_elliptical(1, 0.5, np.pi / 6, 1), 'Elliptical (RH)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for ax, (jones, name) in zip(axes.flat, states):
        Ex, Ey = compute_polarization_ellipse(jones)
        params = polarization_ellipse_params(jones)

        # Color code by time to show propagation direction
        t = np.linspace(0, 1, len(Ex))
        for i in range(len(Ex) - 1):
            ax.plot(Ex[i:i+2], Ey[i:i+2], color=plt.cm.viridis(t[i]),
                    linewidth=2, alpha=0.8)

        # Start and direction marker
        ax.plot(Ex[0], Ey[0], 'ro', markersize=8, label='Start')
        # Arrow showing direction of rotation
        mid = len(Ex) // 4
        ax.annotate('', xy=(Ex[mid+1], Ey[mid+1]),
                     xytext=(Ex[mid], Ey[mid]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Ex')
        ax.set_ylabel('Ey')
        ax.set_title(f'{name}\n(e={params["ellipticity"]:.2f}, {params["handedness"]})')

    fig.suptitle('Polarization States: Electric Field Trajectories', fontsize=14)
    fig.tight_layout()
    fig.savefig('05_polarization_states.png', dpi=150)
    plt.close(fig)
    print("[Saved] 05_polarization_states.png")


# ---------------------------------------------------------------------------
# Visualization 2: Malus's law
# ---------------------------------------------------------------------------

def plot_malus_law():
    """Demonstrate Malus's law: I = I0 * cos^2(theta).

    Shows the transmitted intensity as the analyzer angle is rotated,
    and extends to three-polarizer configurations.
    """
    theta = np.linspace(0, 2 * np.pi, 500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Two polarizers ---
    I_malus = np.cos(theta)**2
    ax1.plot(np.degrees(theta), I_malus, 'b-', linewidth=2, label="Malus's law: cos^2(theta)")
    ax1.set_xlabel('Analyzer angle (degrees)')
    ax1.set_ylabel('Transmitted intensity (fraction)')
    ax1.set_title("Malus's Law: Two Polarizers")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Verify with Jones calculus
    J_in = jones_horizontal()
    I_jones = []
    for th in theta:
        P = jones_matrix_polarizer(th)
        J_out = P @ J_in
        I_jones.append(np.abs(J_out[0])**2 + np.abs(J_out[1])**2)
    ax1.plot(np.degrees(theta), I_jones, 'r--', linewidth=1, label='Jones calc')
    ax1.legend()

    # --- Panel 2: Three polarizers (crossed with intermediate) ---
    # First and third polarizers are crossed (0 and 90 degrees).
    # The middle polarizer at angle phi allows some light through.
    # This demonstrates that a polarizer between crossed polarizers can
    # increase transmission -- counterintuitive at first!
    phi_range = np.linspace(0, np.pi / 2, 200)
    I_three = []

    for phi in phi_range:
        P1 = jones_matrix_polarizer(0)
        P2 = jones_matrix_polarizer(phi)
        P3 = jones_matrix_polarizer(np.pi / 2)
        J_out = P3 @ P2 @ P1 @ jones_horizontal()
        I_three.append(np.abs(J_out[0])**2 + np.abs(J_out[1])**2)

    I_three = np.array(I_three)
    ax2.plot(np.degrees(phi_range), I_three, 'b-', linewidth=2)
    ax2.set_xlabel('Middle polarizer angle (degrees)')
    ax2.set_ylabel('Transmitted intensity')
    ax2.set_title('Three Polarizers: Crossed First & Last')

    # Maximum occurs at phi=45 degrees
    phi_max = np.degrees(phi_range[np.argmax(I_three)])
    I_max = I_three.max()
    ax2.axvline(45, color='red', linestyle='--', alpha=0.5)
    ax2.text(46, I_max, f'Max at 45 deg\nI = {I_max:.4f}', fontsize=9)
    ax2.grid(True, alpha=0.3)

    print(f"\n  Three-polarizer max transmission: {I_max:.4f} at phi = {phi_max:.1f} deg")
    print(f"  (Without middle polarizer: I = 0 -- crossed polarizers block all light)")

    fig.tight_layout()
    fig.savefig('05_polarization_malus.png', dpi=150)
    plt.close(fig)
    print("[Saved] 05_polarization_malus.png")


# ---------------------------------------------------------------------------
# Visualization 3: Wave plate effects
# ---------------------------------------------------------------------------

def plot_wave_plate_effects():
    """Show how wave plates transform polarization states.

    Demonstrates:
    - QWP on linear -> circular
    - QWP on circular -> linear
    - HWP rotating linear polarization
    - HWP flipping circular handedness
    """
    transformations = [
        # (input, element, description)
        (jones_linear(np.pi / 4), jones_matrix_qwp(0),
         'QWP(0) on Linear(45)\n-> RCP'),
        (jones_rcp(), jones_matrix_qwp(0),
         'QWP(0) on RCP\n-> Linear'),
        (jones_horizontal(), jones_matrix_hwp(np.pi / 8),
         'HWP(22.5) on H-pol\n-> V-pol'),
        (jones_rcp(), jones_matrix_hwp(0),
         'HWP(0) on RCP\n-> LCP'),
        (jones_linear(np.pi / 6), jones_matrix_retarder(0, np.pi / 3),
         'Retarder(60 deg) on\nLinear(30) -> Elliptical'),
        (jones_horizontal(), jones_matrix_qwp(np.pi / 4),
         'QWP(45) on H-pol\n-> Linear(45)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, (J_in, M, desc) in zip(axes.flat, transformations):
        J_out = M @ J_in

        # Plot input (dashed) and output (solid)
        Ex_in, Ey_in = compute_polarization_ellipse(J_in)
        Ex_out, Ey_out = compute_polarization_ellipse(J_out)

        ax.plot(Ex_in, Ey_in, 'b--', linewidth=1.5, alpha=0.5, label='Input')
        ax.plot(Ex_out, Ey_out, 'r-', linewidth=2, label='Output')

        # Arrow for output direction
        mid = len(Ex_out) // 4
        ax.annotate('', xy=(Ex_out[mid + 1], Ey_out[mid + 1]),
                     xytext=(Ex_out[mid], Ey_out[mid]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlabel('Ex')
        ax.set_ylabel('Ey')
        ax.set_title(desc, fontsize=9)

    fig.suptitle('Wave Plate Effects on Polarization States', fontsize=14)
    fig.tight_layout()
    fig.savefig('05_polarization_waveplates.png', dpi=150)
    plt.close(fig)
    print("[Saved] 05_polarization_waveplates.png")


# ---------------------------------------------------------------------------
# Visualization 4: Brewster angle
# ---------------------------------------------------------------------------

def fresnel_reflectance(theta_i: np.ndarray, n1: float, n2: float) -> tuple:
    """Compute Fresnel reflectance for s- and p-polarization.

    At the Brewster angle, the p-polarized reflectance goes to zero.
    This occurs when: tan(theta_B) = n2/n1

    The reflected light is then purely s-polarized, which is the basis
    of polarization by reflection (used in some laser designs).

    Args:
        theta_i: Array of incidence angles (radians).
        n1: Refractive index of incident medium.
        n2: Refractive index of transmitted medium.

    Returns:
        Tuple of (Rs, Rp) reflectance arrays.
    """
    sin_theta_t = n1 * np.sin(theta_i) / n2
    # Clip to handle TIR gracefully
    cos_theta_t = np.sqrt(np.maximum(0, 1 - sin_theta_t**2))
    cos_theta_i = np.cos(theta_i)

    # Fresnel coefficients (amplitude)
    rs = (n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)
    rp = (n2 * cos_theta_i - n1 * cos_theta_t) / (n2 * cos_theta_i + n1 * cos_theta_t)

    return rs**2, rp**2


def plot_brewster_angle():
    """Visualize Fresnel reflectance and the Brewster angle.

    At the Brewster angle, reflected light is 100% s-polarized.
    This principle is used in Brewster windows for laser cavities.
    """
    n1 = 1.0  # air
    n2 = 1.52  # glass

    theta_B = np.arctan(n2 / n1)  # Brewster angle
    theta_B_deg = np.degrees(theta_B)

    theta = np.linspace(0, np.pi / 2, 1000)
    Rs, Rp = fresnel_reflectance(theta, n1, n2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Reflectance vs angle ---
    ax1.plot(np.degrees(theta), Rs, 'b-', linewidth=2, label='Rs (s-pol)')
    ax1.plot(np.degrees(theta), Rp, 'r-', linewidth=2, label='Rp (p-pol)')
    ax1.plot(np.degrees(theta), 0.5 * (Rs + Rp), 'k--', linewidth=1,
             label='Unpolarized avg')
    ax1.axvline(theta_B_deg, color='green', linestyle='--', linewidth=1.5,
                label=f'Brewster angle = {theta_B_deg:.1f} deg')
    ax1.set_xlabel('Incidence angle (degrees)')
    ax1.set_ylabel('Reflectance')
    ax1.set_title(f'Fresnel Reflectance (n1={n1}, n2={n2})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Degree of polarization of reflected light ---
    # DOP = (Rs - Rp) / (Rs + Rp) for unpolarized incident light
    R_total = Rs + Rp
    # Avoid division by zero
    DOP = np.where(R_total > 1e-10, (Rs - Rp) / R_total, 0)

    ax2.plot(np.degrees(theta), DOP, 'purple', linewidth=2)
    ax2.axvline(theta_B_deg, color='green', linestyle='--', linewidth=1.5,
                label=f'Brewster angle = {theta_B_deg:.1f} deg')
    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Incidence angle (degrees)')
    ax2.set_ylabel('Degree of polarization')
    ax2.set_title('Polarization of Reflected Light')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    print(f"\n  Brewster angle (air -> glass): {theta_B_deg:.2f} deg")
    print(f"  At Brewster angle: Rs = {Rs[np.argmin(Rp)]:.4f}, Rp = {Rp.min():.2e}")
    print(f"  Normal incidence: R = {((n1 - n2)/(n1 + n2))**2:.4f}")

    fig.tight_layout()
    fig.savefig('05_polarization_brewster.png', dpi=150)
    plt.close(fig)
    print("[Saved] 05_polarization_brewster.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Polarization: Jones Calculus, Malus's Law, Wave Plates, Brewster Angle")
    print("=" * 60)

    # Jones vector demonstrations
    print("\n--- Polarization States ---")
    states = {
        'Horizontal': jones_horizontal(),
        'Vertical': jones_vertical(),
        '45 deg linear': jones_linear(np.pi / 4),
        'RCP': jones_rcp(),
        'LCP': jones_lcp(),
    }
    for name, J in states.items():
        params = polarization_ellipse_params(J)
        print(f"  {name}: J = [{J[0]:.3f}, {J[1]:.3f}], "
              f"ellipticity = {params['ellipticity']:.3f}, "
              f"handedness = {params['handedness']}")

    plot_polarization_states()

    print("\n--- Malus's Law ---")
    plot_malus_law()

    print("\n--- Wave Plate Effects ---")
    plot_wave_plate_effects()

    print("\n--- Brewster Angle ---")
    plot_brewster_angle()

    print("\nDone. All plots saved.")
