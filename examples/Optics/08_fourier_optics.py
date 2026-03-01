#!/usr/bin/env python3
"""Fourier Optics: Spatial Filtering, PSF, OTF, and Image Formation.

This module explores the Fourier-transform relationship between apertures,
diffraction patterns, and imaging:

1. 2D Fourier transforms of apertures — the far-field diffraction pattern
   is the Fourier transform of the aperture transmittance function
2. Lens as Fourier transformer — a lens performs an optical Fourier transform
   in its back focal plane
3. 4f spatial filtering system — using the Fourier plane for low-pass,
   high-pass, and edge detection filtering
4. Point spread function (PSF) and Optical Transfer Function (OTF)
5. Image formation simulation — convolution with the PSF

Physics background:
- Fraunhofer diffraction: The far-field pattern of an aperture is proportional
  to the 2D Fourier transform of the aperture's transmittance function.
- A lens placed one focal length behind an aperture produces the Fourier
  transform at its back focal plane (without the far-field requirement).
- The 4f system uses two lenses to perform Fourier filtering: the first
  lens transforms to the frequency domain, a mask filters frequencies,
  and the second lens transforms back to the spatial domain.
- The PSF is the image of a point source. The OTF is its Fourier transform
  and describes the system's frequency response.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift


# ---------------------------------------------------------------------------
# Aperture functions
# ---------------------------------------------------------------------------

def circular_aperture(N: int, radius: float, pixel_size: float = 1.0) -> np.ndarray:
    """Create a circular aperture transmittance function.

    A circular aperture is the most common pupil shape in optical systems.
    Its Fourier transform gives the Airy pattern.

    Args:
        N: Grid size (N x N pixels).
        radius: Radius of the aperture (in pixel units or physical units).
        pixel_size: Physical size of each pixel.

    Returns:
        2D binary array (1 inside, 0 outside).
    """
    x = (np.arange(N) - N / 2) * pixel_size
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    return (R <= radius).astype(float)


def rectangular_aperture(N: int, width: float, height: float,
                          pixel_size: float = 1.0) -> np.ndarray:
    """Create a rectangular aperture transmittance function.

    Its Fourier transform is a 2D sinc function (product of sinc in x and y).

    Args:
        N: Grid size.
        width: Width of the aperture.
        height: Height of the aperture.
        pixel_size: Physical size of each pixel.

    Returns:
        2D binary array.
    """
    x = (np.arange(N) - N / 2) * pixel_size
    X, Y = np.meshgrid(x, x)
    return ((np.abs(X) <= width / 2) & (np.abs(Y) <= height / 2)).astype(float)


def double_slit_aperture(N: int, slit_width: float, separation: float,
                          slit_height: float, pixel_size: float = 1.0) -> np.ndarray:
    """Create a double-slit aperture function.

    Args:
        N: Grid size.
        slit_width: Width of each slit.
        separation: Center-to-center distance between slits.
        slit_height: Height of the slits.
        pixel_size: Physical size per pixel.

    Returns:
        2D binary array.
    """
    x = (np.arange(N) - N / 2) * pixel_size
    X, Y = np.meshgrid(x, x)

    slit1 = ((np.abs(X - separation / 2) <= slit_width / 2) &
             (np.abs(Y) <= slit_height / 2))
    slit2 = ((np.abs(X + separation / 2) <= slit_width / 2) &
             (np.abs(Y) <= slit_height / 2))
    return (slit1 | slit2).astype(float)


# ---------------------------------------------------------------------------
# Fourier transform and diffraction
# ---------------------------------------------------------------------------

def compute_diffraction_pattern(aperture: np.ndarray) -> np.ndarray:
    """Compute the Fraunhofer diffraction pattern of an aperture.

    The far-field diffraction pattern is |FT(aperture)|^2.
    We use fftshift to center the zero-frequency component.

    The intensity is normalized to unit peak for display purposes.

    Args:
        aperture: 2D aperture transmittance function.

    Returns:
        2D intensity pattern (normalized).
    """
    # The Fourier transform gives the complex amplitude in the far field
    field = fftshift(fft2(ifftshift(aperture)))
    intensity = np.abs(field)**2
    # Normalize to peak = 1
    intensity /= intensity.max()
    return intensity


# ---------------------------------------------------------------------------
# Visualization 1: Aperture Fourier transforms
# ---------------------------------------------------------------------------

def plot_aperture_transforms():
    """Show apertures and their Fraunhofer diffraction patterns.

    Demonstrates the FT relationship for circular, rectangular, and
    double-slit apertures.
    """
    N = 512

    apertures = [
        (circular_aperture(N, 40), 'Circular'),
        (rectangular_aperture(N, 60, 30), 'Rectangular'),
        (double_slit_aperture(N, 5, 40, 80), 'Double Slit'),
        (circular_aperture(N, 40) - circular_aperture(N, 35), 'Annular'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, (aperture, name) in enumerate(apertures):
        # Top row: aperture
        axes[0, i].imshow(aperture, cmap='gray', extent=[-N//2, N//2, -N//2, N//2])
        axes[0, i].set_title(f'{name} Aperture')
        axes[0, i].set_xlabel('x (pixels)')
        axes[0, i].set_ylabel('y (pixels)')

        # Bottom row: diffraction pattern (log scale for visibility)
        pattern = compute_diffraction_pattern(aperture)
        log_pattern = np.log10(pattern + 1e-8)
        axes[1, i].imshow(log_pattern, cmap='inferno',
                           extent=[-N//2, N//2, -N//2, N//2],
                           vmin=-4, vmax=0)
        axes[1, i].set_title(f'{name} Diffraction')
        axes[1, i].set_xlabel('fx (pixels)')
        axes[1, i].set_ylabel('fy (pixels)')
        axes[1, i].set_xlim(-N//4, N//4)
        axes[1, i].set_ylim(-N//4, N//4)

    fig.suptitle('Apertures and Their Fraunhofer Diffraction Patterns', fontsize=14)
    fig.tight_layout()
    fig.savefig('08_fourier_apertures.png', dpi=150)
    plt.close(fig)
    print("[Saved] 08_fourier_apertures.png")


# ---------------------------------------------------------------------------
# Visualization 2: 4f spatial filtering system
# ---------------------------------------------------------------------------

def create_test_image(N: int) -> np.ndarray:
    """Create a test image with features at multiple spatial frequencies.

    Contains a mix of sharp edges, gradual gradients, and fine details
    to demonstrate the effects of spatial filtering.

    Args:
        N: Image size (N x N).

    Returns:
        2D grayscale image (0 to 1).
    """
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)

    # Combine multiple frequency components
    image = np.zeros((N, N))

    # Low frequency: smooth gradient
    image += 0.3 * (1 + np.cos(2 * np.pi * X))

    # Medium frequency: bars
    image += 0.2 * (np.sign(np.sin(10 * np.pi * X)) + 1) / 2

    # High frequency: fine grid
    image += 0.1 * (np.cos(40 * np.pi * X) * np.cos(40 * np.pi * Y) + 1) / 2

    # Sharp feature: circle
    R = np.sqrt(X**2 + Y**2)
    image += 0.3 * (R < 0.3).astype(float)

    # Normalize
    image = (image - image.min()) / (image.max() - image.min())
    return image


def apply_spatial_filter(image: np.ndarray, filter_mask: np.ndarray) -> np.ndarray:
    """Apply a spatial filter in the Fourier (frequency) domain.

    This simulates the 4f system: the first lens computes the FT of the image,
    the mask multiplies the spectrum, and the second lens computes the inverse FT.

    The 4f system is elegant because it performs coherent optical processing
    at the speed of light -- no computation required (in principle).

    Args:
        image: Input 2D image.
        filter_mask: 2D filter in the frequency domain (centered).

    Returns:
        Filtered image (real part of inverse FT).
    """
    # Forward Fourier transform
    spectrum = fftshift(fft2(image))

    # Apply filter in frequency domain
    filtered_spectrum = spectrum * filter_mask

    # Inverse Fourier transform
    filtered_image = np.real(ifft2(ifftshift(filtered_spectrum)))
    return filtered_image


def make_lowpass_filter(N: int, cutoff_radius: float) -> np.ndarray:
    """Create a circular low-pass filter in the frequency domain.

    Passes only spatial frequencies below the cutoff, removing fine details.
    This is equivalent to blurring the image.

    Args:
        N: Grid size.
        cutoff_radius: Radius in frequency pixels.

    Returns:
        2D filter mask (0 to 1).
    """
    x = np.arange(N) - N / 2
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    return (R <= cutoff_radius).astype(float)


def make_highpass_filter(N: int, cutoff_radius: float) -> np.ndarray:
    """Create a circular high-pass filter (complement of low-pass).

    Passes only high spatial frequencies, emphasizing edges and fine details.
    This is the basis of edge detection in the frequency domain.

    Args:
        N: Grid size.
        cutoff_radius: Radius in frequency pixels.

    Returns:
        2D filter mask.
    """
    return 1.0 - make_lowpass_filter(N, cutoff_radius)


def make_bandpass_filter(N: int, inner: float, outer: float) -> np.ndarray:
    """Create an annular band-pass filter.

    Passes only spatial frequencies between inner and outer radii.
    Useful for isolating specific frequency bands.

    Args:
        N: Grid size.
        inner: Inner radius.
        outer: Outer radius.

    Returns:
        2D filter mask.
    """
    x = np.arange(N) - N / 2
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    return ((R >= inner) & (R <= outer)).astype(float)


def make_directional_filter(N: int, angle: float, width: float) -> np.ndarray:
    """Create a directional (oriented) frequency filter.

    Passes frequencies along a specific direction. This can be used
    to detect features oriented in a particular direction.

    Args:
        N: Grid size.
        angle: Direction angle (radians, from x-axis).
        width: Angular width of the passband (radians).

    Returns:
        2D filter mask.
    """
    x = np.arange(N) - N / 2
    X, Y = np.meshgrid(x, x)
    theta = np.arctan2(Y, X)
    diff = np.abs(np.angle(np.exp(1j * (theta - angle))))
    mask = (diff <= width / 2) | (np.abs(diff - np.pi) <= width / 2)
    return mask.astype(float)


def plot_4f_filtering():
    """Demonstrate 4f spatial filtering with different filter types."""
    N = 256
    image = create_test_image(N)

    filters = [
        (make_lowpass_filter(N, 20), 'Low-pass (r=20)'),
        (make_lowpass_filter(N, 50), 'Low-pass (r=50)'),
        (make_highpass_filter(N, 10), 'High-pass (r=10)'),
        (make_bandpass_filter(N, 10, 40), 'Band-pass (10-40)'),
        (make_directional_filter(N, 0, np.pi / 6), 'Directional (0 deg)'),
    ]

    fig, axes = plt.subplots(3, len(filters) + 1, figsize=(18, 10))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Original spectrum
    spectrum = np.abs(fftshift(fft2(image)))
    log_spec = np.log10(spectrum + 1)
    axes[1, 0].imshow(log_spec, cmap='inferno')
    axes[1, 0].set_title('Spectrum (log)')
    axes[1, 0].axis('off')

    axes[2, 0].axis('off')

    for i, (filt, name) in enumerate(filters):
        col = i + 1

        # Filter mask
        axes[0, col].imshow(filt, cmap='gray')
        axes[0, col].set_title(name)
        axes[0, col].axis('off')

        # Filtered spectrum
        filtered_spec = spectrum * filt
        log_filt_spec = np.log10(filtered_spec + 1)
        axes[1, col].imshow(log_filt_spec, cmap='inferno')
        axes[1, col].set_title('Filtered Spectrum')
        axes[1, col].axis('off')

        # Filtered image
        filtered = apply_spatial_filter(image, filt)
        axes[2, col].imshow(filtered, cmap='gray')
        axes[2, col].set_title('Filtered Image')
        axes[2, col].axis('off')

    fig.suptitle('4f Spatial Filtering System', fontsize=14)
    fig.tight_layout()
    fig.savefig('08_fourier_4f_filtering.png', dpi=150)
    plt.close(fig)
    print("[Saved] 08_fourier_4f_filtering.png")


# ---------------------------------------------------------------------------
# Visualization 3: PSF and OTF
# ---------------------------------------------------------------------------

def compute_psf(pupil: np.ndarray) -> np.ndarray:
    """Compute the Point Spread Function from the pupil function.

    The PSF is |FT(pupil)|^2 -- the intensity pattern produced by
    a point source after passing through the optical system.

    The PSF describes the fundamental imaging capability of the system.
    A perfect system (no aberrations) produces the diffraction-limited PSF
    (Airy pattern for a circular pupil).

    Args:
        pupil: 2D pupil function (complex-valued to include aberrations).

    Returns:
        2D PSF (normalized to unit sum for energy conservation).
    """
    field = fftshift(fft2(ifftshift(pupil)))
    psf = np.abs(field)**2
    psf /= psf.sum()  # Normalize so total energy = 1
    return psf


def compute_otf(psf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Optical Transfer Function from the PSF.

    The OTF is the Fourier transform of the PSF. Its magnitude is the
    Modulation Transfer Function (MTF), which describes how well the
    system preserves contrast at each spatial frequency.

    MTF = 1 means perfect contrast transfer.
    MTF = 0 means the frequency is completely suppressed.

    Args:
        psf: 2D Point Spread Function.

    Returns:
        Tuple of (MTF, phase transfer function).
    """
    otf = fftshift(fft2(ifftshift(psf)))
    otf /= otf[otf.shape[0] // 2, otf.shape[1] // 2]  # Normalize DC to 1
    mtf = np.abs(otf)
    ptf = np.angle(otf)
    return mtf, ptf


def plot_psf_and_otf():
    """Visualize PSF and MTF for different pupil configurations.

    Shows how aberrations (represented by phase errors in the pupil)
    degrade the PSF and reduce the MTF.
    """
    N = 256
    radius = 50

    # Different pupil functions
    x = (np.arange(N) - N / 2)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    aperture = (R <= radius).astype(float)

    pupils = [
        (aperture.astype(complex), 'Perfect (no aberration)'),
        # Defocus: quadratic phase
        (aperture * np.exp(1j * 2 * (R / radius)**2 * aperture), 'Defocus'),
        # Spherical aberration: r^4 phase
        (aperture * np.exp(1j * 3 * (R / radius)**4 * aperture), 'Spherical aberration'),
        # Coma: r^3 * cos(theta)
        (aperture * np.exp(1j * 4 * (R / radius)**3 * (X / (radius + 1e-10)) * aperture), 'Coma'),
    ]

    fig, axes = plt.subplots(3, len(pupils), figsize=(16, 12))

    for i, (pupil, name) in enumerate(pupils):
        # PSF
        psf = compute_psf(pupil)
        log_psf = np.log10(psf / psf.max() + 1e-8)

        axes[0, i].imshow(log_psf, cmap='inferno', vmin=-4, vmax=0,
                           extent=[-N//2, N//2, -N//2, N//2])
        axes[0, i].set_title(f'PSF: {name}')
        axes[0, i].set_xlim(-30, 30)
        axes[0, i].set_ylim(-30, 30)

        # MTF
        mtf, _ = compute_otf(psf)

        axes[1, i].imshow(mtf, cmap='viridis', vmin=0, vmax=1,
                           extent=[-N//2, N//2, -N//2, N//2])
        axes[1, i].set_title(f'MTF: {name}')
        axes[1, i].set_xlim(-N//4, N//4)
        axes[1, i].set_ylim(-N//4, N//4)

        # MTF cross-section
        mtf_1d = mtf[N // 2, N // 2:]
        freq = np.arange(len(mtf_1d))
        axes[2, i].plot(freq, mtf_1d, 'b-', linewidth=2)
        axes[2, i].set_xlabel('Spatial frequency')
        axes[2, i].set_ylabel('MTF')
        axes[2, i].set_title('MTF Cross-section')
        axes[2, i].set_xlim(0, N // 4)
        axes[2, i].set_ylim(0, 1.1)
        axes[2, i].grid(True, alpha=0.3)

        # Compute Strehl ratio (ratio of peak PSF to diffraction-limited peak)
        psf_perfect = compute_psf(aperture.astype(complex))
        strehl = psf.max() / psf_perfect.max()
        axes[0, i].text(0.05, 0.95, f'Strehl = {strehl:.3f}',
                         transform=axes[0, i].transAxes, fontsize=9,
                         color='white', va='top',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    fig.suptitle('Point Spread Function and Modulation Transfer Function', fontsize=14)
    fig.tight_layout()
    fig.savefig('08_fourier_psf_otf.png', dpi=150)
    plt.close(fig)
    print("[Saved] 08_fourier_psf_otf.png")


# ---------------------------------------------------------------------------
# Visualization 4: Image formation as convolution
# ---------------------------------------------------------------------------

def plot_image_formation():
    """Simulate coherent image formation as convolution with the PSF.

    The image I(x,y) of an object O(x,y) through an optical system with
    PSF h(x,y) is:
        I(x,y) = |O(x,y) * h(x,y)|^2  (coherent)
    or
        I(x,y) = O_intensity(x,y) ** |h(x,y)|^2  (incoherent)

    where * denotes convolution and ** denotes correlation.

    We demonstrate incoherent image formation (more common in practice),
    which is a convolution of the object intensity with the PSF intensity.
    """
    N = 256

    # Create object: simple resolution target pattern
    obj = np.zeros((N, N))

    # Bars at different spacings (like a USAF resolution target)
    for k, spacing in enumerate([20, 14, 10, 7, 5]):
        y_start = 30 + k * 40
        for j in range(5):
            x_start = j * spacing * 2
            if x_start + spacing < N:
                obj[y_start:y_start + 30, x_start:x_start + spacing] = 1.0

    # Add some point sources
    for y, x in [(220, 50), (220, 70), (220, 85), (220, 95)]:
        if y < N and x < N:
            obj[y, x] = 5.0

    # Create PSFs for different aperture sizes
    aperture_radii = [50, 30, 15]

    fig, axes = plt.subplots(2, len(aperture_radii) + 1, figsize=(16, 8))

    # Original object
    axes[0, 0].imshow(obj, cmap='gray')
    axes[0, 0].set_title('Object')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    for i, r in enumerate(aperture_radii):
        col = i + 1
        pupil = circular_aperture(N, r).astype(complex)
        psf = compute_psf(pupil)

        # Incoherent image formation: convolve object intensity with PSF
        # Using FFT-based convolution for efficiency
        obj_spectrum = fft2(obj)
        psf_spectrum = fft2(ifftshift(psf))  # PSF needs to be centered at origin for FFT convolution
        image = np.real(ifft2(obj_spectrum * psf_spectrum))
        image = np.maximum(image, 0)  # Ensure non-negative

        # Show PSF
        log_psf = np.log10(psf / psf.max() + 1e-8)
        axes[0, col].imshow(log_psf, cmap='inferno', vmin=-4, vmax=0,
                             extent=[-N//2, N//2, -N//2, N//2])
        axes[0, col].set_title(f'PSF (aperture r={r})')
        axes[0, col].set_xlim(-30, 30)
        axes[0, col].set_ylim(-30, 30)

        # Show image
        axes[1, col].imshow(image, cmap='gray')
        axes[1, col].set_title(f'Image (r={r})')
        axes[1, col].axis('off')

    fig.suptitle('Image Formation: Object Convolved with PSF', fontsize=14)
    fig.tight_layout()
    fig.savefig('08_fourier_image_formation.png', dpi=150)
    plt.close(fig)
    print("[Saved] 08_fourier_image_formation.png")

    print("\n  Larger aperture -> sharper PSF -> better resolution")
    print("  Smaller aperture -> broader PSF -> more blurring")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Fourier Optics: Spatial Filtering, PSF, OTF, Image Formation")
    print("=" * 60)

    print("\n--- Aperture Fourier Transforms ---")
    plot_aperture_transforms()

    print("\n--- 4f Spatial Filtering ---")
    plot_4f_filtering()

    print("\n--- PSF and OTF ---")
    plot_psf_and_otf()

    print("\n--- Image Formation ---")
    plot_image_formation()

    print("\nDone. All plots saved.")
