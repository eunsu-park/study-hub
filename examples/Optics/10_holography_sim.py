#!/usr/bin/env python3
"""Holography Simulation: Recording, Reconstruction, and Phase Retrieval.

This module simulates the fundamental processes of holography:

1. Hologram recording — interference between an object wave and a reference wave
   creates a fringe pattern encoding both amplitude and phase information
2. Numerical reconstruction — the Angular Spectrum Method (ASM) propagates
   the wavefield to reconstruct the 3D object
3. Off-axis vs inline holography — different geometries and their trade-offs
4. Phase retrieval — the Gerchberg-Saxton algorithm recovers phase from
   intensity-only measurements

Physics background:
- A hologram records the interference pattern between a known reference wave
  and an unknown object wave. Unlike photography, which records only intensity,
  holography preserves the phase information.
- The hologram intensity is:
      H = |R + O|^2 = |R|^2 + |O|^2 + R*O + RO*
  where R is the reference wave and O is the object wave.
- Reconstruction illuminates the hologram with the reference wave:
      R * H = R|R|^2 + R|O|^2 + |R|^2 * O + R^2 * O*
  The third term is the reconstructed object wave (virtual image).
  The fourth term is the conjugate wave (real image, twin image).
- Off-axis holography separates these terms in angular space.
- The Angular Spectrum Method propagates a wavefield by multiplying its
  Fourier transform by a propagation transfer function.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq


# ---------------------------------------------------------------------------
# Wave propagation: Angular Spectrum Method
# ---------------------------------------------------------------------------

def angular_spectrum_propagate(field: np.ndarray, wavelength: float,
                                dx: float, dz: float) -> np.ndarray:
    """Propagate a complex optical field using the Angular Spectrum Method.

    The ASM is exact (within the scalar diffraction approximation) for
    propagation over any distance. It works by:
    1. Fourier transforming the input field to get the angular spectrum
    2. Multiplying by the free-space propagation transfer function
    3. Inverse Fourier transforming to get the output field

    The transfer function is: H(fx,fy) = exp(i*2*pi*dz*sqrt(1/lambda^2 - fx^2 - fy^2))
    where fx, fy are spatial frequencies.

    Evanescent waves (fx^2 + fy^2 > 1/lambda^2) decay exponentially
    and are filtered out to prevent numerical artifacts.

    Args:
        field: 2D complex field (amplitude and phase).
        wavelength: Wavelength.
        dx: Pixel size (same units as wavelength and dz).
        dz: Propagation distance (positive = forward).

    Returns:
        Propagated complex field.
    """
    N = field.shape[0]

    # Spatial frequencies
    fx = fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fx)

    # Propagation transfer function
    # The argument of the square root must be non-negative (propagating waves only)
    arg = 1 / wavelength**2 - FX**2 - FY**2
    propagating = arg >= 0  # Mask out evanescent waves

    # Phase accumulated during propagation
    kz = np.zeros_like(arg)
    kz[propagating] = 2 * np.pi * np.sqrt(arg[propagating])

    # Transfer function (complex exponential for phase advance)
    H = np.exp(1j * kz * dz) * propagating

    # Propagate: FT -> multiply -> IFT
    spectrum = fft2(field)
    propagated_spectrum = spectrum * H
    output = ifft2(propagated_spectrum)

    return output


# ---------------------------------------------------------------------------
# Object and reference wave generation
# ---------------------------------------------------------------------------

def create_object_field(N: int, dx: float, wavelength: float) -> np.ndarray:
    """Create a complex object wave from a simple test pattern.

    The object is a combination of geometric shapes with varying amplitude
    and phase. This simulates what a real object would look like as a
    scattered wavefield.

    Args:
        N: Grid size.
        dx: Pixel size.
        wavelength: Wavelength.

    Returns:
        2D complex field (object wave at the hologram plane after propagation).
    """
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)

    # Object amplitude: geometric pattern
    amplitude = np.zeros((N, N))

    # Letter-like shapes
    # Horizontal bar
    amplitude[N//4:N//4+10, N//4:3*N//4] = 1.0
    # Vertical bars
    amplitude[N//4:3*N//4, N//4:N//4+10] = 1.0
    amplitude[N//4:3*N//4, 3*N//4-10:3*N//4] = 1.0
    # Circle
    R = np.sqrt((X - 0)**2 + (Y - 0)**2)
    amplitude += (R < 15 * dx).astype(float) * 0.5
    # Small dots
    for xp, yp in [(30, 30), (-30, 30), (30, -30), (-30, -30)]:
        R_dot = np.sqrt((X - xp * dx)**2 + (Y - yp * dx)**2)
        amplitude += (R_dot < 5 * dx).astype(float) * 0.8

    # Object phase: smooth random phase (simulates surface roughness)
    rng = np.random.default_rng(42)
    # Low-pass filtered random phase for realistic speckle
    random_phase = rng.uniform(0, 2 * np.pi, (N, N))
    # Smooth the phase with a Gaussian filter in Fourier domain
    fx = fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fx)
    gauss_filter = np.exp(-(FX**2 + FY**2) / (2 * (1 / (20 * dx))**2))
    smooth_phase = np.real(ifft2(fft2(random_phase) * gauss_filter))
    smooth_phase = smooth_phase / smooth_phase.max() * np.pi  # Scale to [0, pi]

    # Complex object field
    obj = amplitude * np.exp(1j * smooth_phase)
    return obj


def plane_wave_reference(N: int, dx: float, wavelength: float,
                          theta_x: float = 0, theta_y: float = 0) -> np.ndarray:
    """Generate a tilted plane wave reference beam.

    For off-axis holography, the reference beam is tilted with respect to
    the optical axis. The tilt angle determines the carrier frequency in the
    hologram, which allows spatial separation of the image terms.

    Args:
        N: Grid size.
        dx: Pixel size.
        wavelength: Wavelength.
        theta_x: Tilt angle in x-direction (radians).
        theta_y: Tilt angle in y-direction (radians).

    Returns:
        2D complex reference field.
    """
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)

    # Wave vector components from the tilt angles
    kx = 2 * np.pi * np.sin(theta_x) / wavelength
    ky = 2 * np.pi * np.sin(theta_y) / wavelength

    return np.exp(1j * (kx * X + ky * Y))


# ---------------------------------------------------------------------------
# Hologram recording and reconstruction
# ---------------------------------------------------------------------------

def record_hologram(object_wave: np.ndarray, reference_wave: np.ndarray) -> np.ndarray:
    """Record a hologram as the intensity of the interference pattern.

    The hologram records H = |R + O|^2, which encodes both the amplitude
    and phase of the object wave through its interference with the reference.

    This is the fundamental principle of holography invented by Dennis Gabor:
    by recording the interference, we preserve phase information that would
    be lost in a simple intensity measurement.

    Args:
        object_wave: Complex object wavefield.
        reference_wave: Complex reference wavefield.

    Returns:
        2D real-valued hologram (intensity pattern).
    """
    total_field = object_wave + reference_wave
    return np.abs(total_field)**2


def reconstruct_hologram(hologram: np.ndarray, reference_wave: np.ndarray,
                          wavelength: float, dx: float,
                          reconstruction_distance: float) -> np.ndarray:
    """Reconstruct a hologram using the Angular Spectrum Method.

    The reconstruction process simulates illuminating the hologram with
    the reference wave and propagating the transmitted field to the
    reconstruction plane.

    The transmitted field through the hologram is: T = R * H
    This field is then propagated to the image plane.

    Args:
        hologram: 2D hologram intensity.
        reference_wave: Complex reference wave used for illumination.
        wavelength: Wavelength.
        dx: Pixel size.
        reconstruction_distance: Distance to propagate (negative for virtual image).

    Returns:
        Complex reconstructed field at the image plane.
    """
    # Illuminate hologram with reference wave
    # The hologram acts as a transmittance function
    transmitted = reference_wave * hologram

    # Propagate to the reconstruction plane
    reconstructed = angular_spectrum_propagate(transmitted, wavelength, dx,
                                               reconstruction_distance)
    return reconstructed


# ---------------------------------------------------------------------------
# Visualization 1: Off-axis holography
# ---------------------------------------------------------------------------

def plot_offaxis_holography():
    """Demonstrate off-axis holography: recording and reconstruction.

    In off-axis holography, the reference beam is tilted relative to the
    object beam. This creates a carrier frequency that separates the
    real image, virtual image, and DC terms in the Fourier domain.

    This separation eliminates the twin-image problem that plagues
    inline (Gabor) holography.
    """
    N = 512
    dx = 1e-6  # 1 um pixels
    wavelength = 632.8e-9  # HeNe laser
    propagation_distance = 5e-3  # 5 mm object-to-hologram distance

    print(f"\n  Grid: {N}x{N}, dx={dx*1e6:.1f} um")
    print(f"  Wavelength: {wavelength*1e9:.1f} nm")
    print(f"  Propagation distance: {propagation_distance*1e3:.1f} mm")

    # Create object and propagate to hologram plane
    obj = create_object_field(N, dx, wavelength)
    obj_at_hologram = angular_spectrum_propagate(obj, wavelength, dx,
                                                  propagation_distance)

    # Reference wave: tilted plane wave
    # The tilt angle must satisfy: sin(theta) > lambda / (2*dx)
    # to separate the sideband from the DC term (Nyquist criterion)
    max_freq = 1 / (2 * dx)
    min_tilt = np.arcsin(wavelength * max_freq / 3)  # Use 1/3 of max for safety
    theta_ref = min_tilt * 1.5  # Comfortable offset

    reference = plane_wave_reference(N, dx, wavelength, theta_x=theta_ref)
    print(f"  Reference tilt angle: {np.degrees(theta_ref):.2f} deg")

    # Record hologram
    hologram = record_hologram(obj_at_hologram, reference)

    # Reconstruct (propagate backward to the object plane)
    reconstructed = reconstruct_hologram(hologram, reference, wavelength, dx,
                                          -propagation_distance)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --- Top row: Recording ---
    # Object amplitude
    axes[0, 0].imshow(np.abs(obj), cmap='gray',
                       extent=[-N*dx/2*1e3, N*dx/2*1e3, -N*dx/2*1e3, N*dx/2*1e3])
    axes[0, 0].set_title('Object Amplitude')
    axes[0, 0].set_xlabel('x (mm)')
    axes[0, 0].set_ylabel('y (mm)')

    # Hologram (zoom to show fringes)
    axes[0, 1].imshow(hologram, cmap='gray',
                       extent=[-N*dx/2*1e3, N*dx/2*1e3, -N*dx/2*1e3, N*dx/2*1e3])
    axes[0, 1].set_title('Hologram (Full)')
    axes[0, 1].set_xlabel('x (mm)')

    # Hologram zoom
    center = N // 2
    zoom = 50
    axes[0, 2].imshow(hologram[center-zoom:center+zoom, center-zoom:center+zoom],
                       cmap='gray')
    axes[0, 2].set_title('Hologram (Zoomed - Carrier Fringes)')

    # --- Bottom row: Reconstruction ---
    # Hologram spectrum (shows DC and sideband separation)
    spectrum = np.abs(fftshift(fft2(hologram)))
    log_spectrum = np.log10(spectrum + 1)
    axes[1, 0].imshow(log_spectrum, cmap='inferno')
    axes[1, 0].set_title('Hologram Spectrum (log)\nSidebands separated from DC')

    # Reconstructed amplitude
    recon_amp = np.abs(reconstructed)
    axes[1, 1].imshow(recon_amp, cmap='gray',
                       extent=[-N*dx/2*1e3, N*dx/2*1e3, -N*dx/2*1e3, N*dx/2*1e3])
    axes[1, 1].set_title('Reconstructed Amplitude')
    axes[1, 1].set_xlabel('x (mm)')

    # Reconstructed phase
    recon_phase = np.angle(reconstructed)
    # Mask phase to show only where there is significant amplitude
    phase_masked = np.where(recon_amp > recon_amp.max() * 0.1, recon_phase, np.nan)
    axes[1, 2].imshow(phase_masked, cmap='hsv',
                       extent=[-N*dx/2*1e3, N*dx/2*1e3, -N*dx/2*1e3, N*dx/2*1e3])
    axes[1, 2].set_title('Reconstructed Phase')
    axes[1, 2].set_xlabel('x (mm)')

    fig.suptitle('Off-Axis Digital Holography', fontsize=14)
    fig.tight_layout()
    fig.savefig('10_holography_offaxis.png', dpi=150)
    plt.close(fig)
    print("[Saved] 10_holography_offaxis.png")


# ---------------------------------------------------------------------------
# Visualization 2: Inline (Gabor) holography and twin image problem
# ---------------------------------------------------------------------------

def plot_inline_holography():
    """Demonstrate inline holography and the twin-image problem.

    In Gabor's original scheme, the reference and object waves propagate
    along the same axis (theta = 0). This is simpler optically but
    produces the twin-image artifact: the real and virtual images overlap
    in the reconstruction.

    The twin image appears as an out-of-focus conjugate superimposed on
    the focused reconstruction. This was the main limitation of Gabor's
    method until off-axis holography was invented by Leith and Upatnieks.
    """
    N = 512
    dx = 1e-6
    wavelength = 632.8e-9
    prop_dist = 5e-3

    # Create object (sparse for inline holography -- needs mostly transparent)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)

    # Simple object: small opaque dots
    obj_amp = np.ones((N, N))
    # Subtract small circles (opaque regions)
    for xp, yp, r in [(20, 20, 8), (-25, 15, 6), (10, -30, 10), (-15, -20, 5)]:
        R = np.sqrt((X - xp * dx)**2 + (Y - yp * dx)**2)
        obj_amp -= 0.8 * (R < r * dx).astype(float)

    obj = obj_amp.astype(complex)

    # Propagate to hologram plane
    obj_at_holo = angular_spectrum_propagate(obj, wavelength, dx, prop_dist)

    # Inline reference: on-axis plane wave (theta=0)
    reference = plane_wave_reference(N, dx, wavelength, theta_x=0, theta_y=0)

    # Record hologram
    hologram = record_hologram(obj_at_holo, reference)

    # Reconstruct at different distances
    distances = [-prop_dist, 0, prop_dist]
    labels = ['Virtual image\n(focused)', 'Hologram plane', 'Real image\n(twin, defocused)']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: hologram and its spectrum
    axes[0, 0].imshow(np.abs(obj), cmap='gray')
    axes[0, 0].set_title('Object Amplitude')

    axes[0, 1].imshow(hologram, cmap='gray')
    axes[0, 1].set_title('Inline Hologram')

    spectrum = np.abs(fftshift(fft2(hologram)))
    axes[0, 2].imshow(np.log10(spectrum + 1), cmap='inferno')
    axes[0, 2].set_title('Spectrum (DC and sidebands overlap!)')

    # Bottom row: reconstructions at different distances
    for i, (dist, label) in enumerate(zip(distances, labels)):
        recon = reconstruct_hologram(hologram, reference, wavelength, dx, dist)
        axes[1, i].imshow(np.abs(recon), cmap='gray')
        axes[1, i].set_title(label)

    fig.suptitle('Inline (Gabor) Holography: Twin Image Problem', fontsize=14)
    fig.tight_layout()
    fig.savefig('10_holography_inline.png', dpi=150)
    plt.close(fig)
    print("[Saved] 10_holography_inline.png")


# ---------------------------------------------------------------------------
# Visualization 3: Gerchberg-Saxton phase retrieval
# ---------------------------------------------------------------------------

def gerchberg_saxton(target_amplitude: np.ndarray, source_amplitude: np.ndarray,
                     wavelength: float, dx: float, dz: float,
                     n_iterations: int = 50) -> tuple[np.ndarray, list]:
    """Gerchberg-Saxton algorithm for phase retrieval.

    The GS algorithm iteratively recovers the phase of a wavefield when
    only the amplitude is known at two planes. It alternates between:
    1. Propagating forward and replacing amplitude with the measured target
    2. Propagating backward and replacing amplitude with the measured source

    The phase is preserved at each step, allowing it to converge to a
    solution that is consistent with both amplitude measurements.

    This algorithm is widely used in:
    - Holographic displays (computing phase-only holograms)
    - Adaptive optics (wavefront sensing)
    - Electron microscopy (exit wave reconstruction)

    Args:
        target_amplitude: Desired amplitude at the target plane.
        source_amplitude: Known amplitude at the source plane.
        wavelength: Wavelength.
        dx: Pixel size.
        dz: Distance between source and target planes.
        n_iterations: Number of GS iterations.

    Returns:
        Tuple of (recovered phase at source plane, error history).
    """
    N = target_amplitude.shape[0]

    # Initial guess: random phase at the source plane
    rng = np.random.default_rng(123)
    phase = rng.uniform(0, 2 * np.pi, (N, N))
    field = source_amplitude * np.exp(1j * phase)

    errors = []

    for iteration in range(n_iterations):
        # Forward propagation to target plane
        field_target = angular_spectrum_propagate(field, wavelength, dx, dz)

        # Replace amplitude with target, keep phase
        target_phase = np.angle(field_target)
        field_target = target_amplitude * np.exp(1j * target_phase)

        # Compute error: how well does the current solution match?
        current_amp = np.abs(angular_spectrum_propagate(
            source_amplitude * np.exp(1j * np.angle(field)),
            wavelength, dx, dz
        ))
        error = np.sqrt(np.mean((current_amp - target_amplitude)**2))
        errors.append(error)

        # Backward propagation to source plane
        field = angular_spectrum_propagate(field_target, wavelength, dx, -dz)

        # Replace amplitude with source, keep phase
        source_phase = np.angle(field)
        field = source_amplitude * np.exp(1j * source_phase)

    recovered_phase = np.angle(field)
    return recovered_phase, errors


def plot_gerchberg_saxton():
    """Demonstrate the Gerchberg-Saxton phase retrieval algorithm.

    We define a target intensity pattern and use GS to find the phase
    that a uniform-amplitude source must have to produce that pattern
    after propagation. This is the core problem in holographic display
    design (phase-only spatial light modulators).
    """
    N = 256
    dx = 2e-6
    wavelength = 532e-9  # green laser
    dz = 10e-3  # 10 mm

    # Source: uniform amplitude (like a laser beam)
    source_amp = np.ones((N, N))

    # Target: desired intensity pattern (a simple image)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)

    target = np.zeros((N, N))
    # Create a star pattern
    for angle in np.linspace(0, np.pi, 5, endpoint=False):
        mask = np.abs(X * np.cos(angle) + Y * np.sin(angle)) < 3 * dx
        mask &= (np.sqrt(X**2 + Y**2) < 60 * dx)
        target += mask.astype(float)

    # Add a circle
    R = np.sqrt(X**2 + Y**2)
    target += (np.abs(R - 50 * dx) < 3 * dx).astype(float)
    target = np.clip(target, 0, 1)
    target_amp = np.sqrt(target)  # Amplitude (not intensity)

    # Run GS algorithm
    print(f"\n  Running Gerchberg-Saxton ({50} iterations)...")
    recovered_phase, errors = gerchberg_saxton(
        target_amp, source_amp, wavelength, dx, dz, n_iterations=50
    )

    # Verify reconstruction
    optimized_field = source_amp * np.exp(1j * recovered_phase)
    reconstructed = angular_spectrum_propagate(optimized_field, wavelength, dx, dz)
    reconstructed_intensity = np.abs(reconstructed)**2

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --- Top row: GS inputs and outputs ---
    axes[0, 0].imshow(target, cmap='gray')
    axes[0, 0].set_title('Target Intensity')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(recovered_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title('Recovered Phase (source plane)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(reconstructed_intensity, cmap='gray')
    axes[0, 2].set_title('Reconstructed Intensity')
    axes[0, 2].axis('off')

    # --- Bottom row: convergence and comparison ---
    axes[1, 0].semilogy(errors, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('RMS Error')
    axes[1, 0].set_title('GS Convergence')
    axes[1, 0].grid(True, alpha=0.3)

    # Cross-section comparison
    mid = N // 2
    axes[1, 1].plot(target[mid, :], 'b-', linewidth=2, label='Target')
    axes[1, 1].plot(reconstructed_intensity[mid, :], 'r--', linewidth=1.5,
                     label='Reconstructed')
    axes[1, 1].set_xlabel('Pixel')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].set_title('Cross-section Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Phase histogram
    axes[1, 2].hist(recovered_phase.ravel(), bins=100, color='purple', alpha=0.7)
    axes[1, 2].set_xlabel('Phase (radians)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Phase Distribution')
    axes[1, 2].grid(True, alpha=0.3)

    # Compute quality metrics
    # Efficiency: fraction of light in the target area
    target_mask = target > 0.1
    efficiency = np.sum(reconstructed_intensity[target_mask]) / np.sum(reconstructed_intensity)
    # Uniformity in target area
    values_in_target = reconstructed_intensity[target_mask]
    uniformity = 1 - (values_in_target.max() - values_in_target.min()) / (
        values_in_target.max() + values_in_target.min())

    print(f"  Efficiency: {efficiency*100:.1f}%")
    print(f"  Uniformity: {uniformity*100:.1f}%")
    print(f"  Final RMS error: {errors[-1]:.6f}")

    fig.suptitle('Gerchberg-Saxton Phase Retrieval Algorithm', fontsize=14)
    fig.tight_layout()
    fig.savefig('10_holography_gerchberg_saxton.png', dpi=150)
    plt.close(fig)
    print("[Saved] 10_holography_gerchberg_saxton.png")


# ---------------------------------------------------------------------------
# Visualization 4: Propagation through focus
# ---------------------------------------------------------------------------

def plot_propagation_through_focus():
    """Visualize wavefield propagation through focus using ASM.

    This shows how a focused beam contracts, reaches minimum size at the
    focal plane, and then expands -- demonstrating the angular spectrum
    method's ability to accurately propagate fields through a focus.
    """
    N = 256
    dx = 1e-6
    wavelength = 532e-9

    # Create a converging spherical wave (simulated by a lens)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)

    # Aperture with a converging phase (thin lens phase)
    f_lens = 5e-3  # 5 mm focal length
    aperture_radius = 80 * dx

    amplitude = (R <= aperture_radius).astype(float)
    # Thin lens phase: exp(-i*k*r^2/(2f))
    k = 2 * np.pi / wavelength
    lens_phase = -k * R**2 / (2 * f_lens)
    field_after_lens = amplitude * np.exp(1j * lens_phase)

    # Propagate to multiple planes around the focus
    z_positions = np.linspace(f_lens * 0.8, f_lens * 1.2, 9)

    fig, axes = plt.subplots(1, len(z_positions), figsize=(18, 3))

    for ax, z in zip(axes, z_positions):
        prop_field = angular_spectrum_propagate(field_after_lens, wavelength, dx, z)
        intensity = np.abs(prop_field)**2
        intensity /= intensity.max()

        # Show central region
        c = N // 2
        zoom = 30
        ax.imshow(intensity[c-zoom:c+zoom, c-zoom:c+zoom],
                   cmap='inferno', vmin=0, vmax=1,
                   extent=[-zoom*dx*1e6, zoom*dx*1e6, -zoom*dx*1e6, zoom*dx*1e6])
        rel_z = (z - f_lens) / f_lens
        ax.set_title(f'z/f = {z/f_lens:.2f}', fontsize=8)
        ax.set_xlabel('um', fontsize=7)
        if ax == axes[0]:
            ax.set_ylabel('um', fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle('Wavefield Propagation Through Focus (Angular Spectrum Method)', fontsize=12)
    fig.tight_layout()
    fig.savefig('10_holography_propagation.png', dpi=150)
    plt.close(fig)
    print("[Saved] 10_holography_propagation.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Holography: Recording, Reconstruction, and Phase Retrieval")
    print("=" * 60)

    print("\n--- Off-Axis Holography ---")
    plot_offaxis_holography()

    print("\n--- Inline (Gabor) Holography ---")
    plot_inline_holography()

    print("\n--- Gerchberg-Saxton Phase Retrieval ---")
    plot_gerchberg_saxton()

    print("\n--- Propagation Through Focus ---")
    plot_propagation_through_focus()

    print("\nDone. All plots saved.")
