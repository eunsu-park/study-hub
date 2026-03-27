#!/usr/bin/env python3
"""Zernike Polynomials: Mode Generation, Wavefront Fitting, and Turbulence.

This module implements Zernike polynomials for optical wavefront analysis:

1. Zernike mode generation — evaluates the Noll-indexed Zernike polynomials
   on the unit circle using the explicit radial polynomial formula
2. Wavefront fitting — decomposes a measured wavefront into Zernike
   coefficients via least-squares fitting
3. Kolmogorov phase screens — generates random atmospheric turbulence
   phase screens with the correct spatial statistics
4. Strehl ratio estimation — uses the Maréchal approximation to estimate
   image quality from wavefront error

Physics background:
- Zernike polynomials form a complete, orthonormal basis on the unit circle.
  Each mode corresponds to a classical optical aberration (tilt, defocus,
  coma, astigmatism, spherical, etc.).
- The polynomial Z_j(rho, theta) = N * R_n^|m|(rho) * {cos, sin}(|m|*theta)
  where R is the radial polynomial and (n, m) are determined by the Noll
  index j (Noll, 1976).
- Atmospheric turbulence follows Kolmogorov statistics with a phase power
  spectral density Phi(kappa) = 0.023 * r0^(-5/3) * kappa^(-11/3), where
  r0 is the Fried parameter (coherence length).
- The Maréchal approximation: Strehl ratio S ≈ exp(-(2*pi*sigma)^2) where
  sigma is the RMS wavefront error in waves.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import factorial


# ---------------------------------------------------------------------------
# Noll index conversion
# ---------------------------------------------------------------------------

def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll single-index j (>= 1) to radial order n and azimuthal m.

    The Noll convention (Noll, 1976) orders Zernike modes by increasing
    radial order n. Within each n, even j corresponds to cosine (m >= 0)
    and odd j to sine (m < 0). This mapping is the standard in adaptive
    optics and optical testing.

    Args:
        j: Noll index, starting at 1.

    Returns:
        Tuple (n, m) where n is the radial order and m the azimuthal
        frequency (positive for cosine, negative for sine).
    """
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")
    # Find the radial order n
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    # Position within this radial order
    k = j - n * (n + 1) // 2
    # Determine |m|
    if n % 2 == 0:
        m_abs = 2 * ((k + 1) // 2)
    else:
        m_abs = 2 * (k // 2) + 1
    # Sign: even j -> cosine (m >= 0), odd j -> sine (m < 0)
    if m_abs == 0:
        m = 0
    elif j % 2 == 0:
        m = m_abs
    else:
        m = -m_abs
    return n, m


def nm_to_noll(n: int, m: int) -> int:
    """Convert radial order n and azimuthal frequency m to Noll index j.

    Inverse of noll_to_nm. Useful for looking up specific modes by their
    (n, m) quantum numbers.
    """
    m_abs = abs(m)
    # Starting index for this radial order
    j_start = n * (n + 1) // 2 + 1
    # Count through modes in this order
    for j in range(j_start, j_start + n + 1):
        nn, mm = noll_to_nm(j)
        if nn == n and mm == m:
            return j
    raise ValueError(f"No Noll index for (n={n}, m={m})")


# ---------------------------------------------------------------------------
# Zernike polynomial evaluation
# ---------------------------------------------------------------------------

def zernike_radial(n: int, m_abs: int, rho: np.ndarray) -> np.ndarray:
    """Compute radial Zernike polynomial R_n^|m|(rho).

    Uses the explicit factorial formula:
        R_n^m(rho) = sum_s (-1)^s * (n-s)! / (s! * ((n+m)/2-s)! * ((n-m)/2-s)!)
                     * rho^(n-2s)

    This direct formula is numerically stable for orders up to ~30.
    For higher orders, a recurrence relation would be preferred.

    Args:
        n: Radial order (non-negative integer).
        m_abs: Absolute value of azimuthal frequency.
        rho: Radial coordinate(s), should be in [0, 1].

    Returns:
        Evaluated radial polynomial, same shape as rho.
    """
    if (n - m_abs) % 2 != 0:
        return np.zeros_like(rho, dtype=float)
    R = np.zeros_like(rho, dtype=float)
    for s in range((n - m_abs) // 2 + 1):
        coeff = ((-1) ** s * factorial(n - s)
                 / (factorial(s)
                    * factorial((n + m_abs) // 2 - s)
                    * factorial((n - m_abs) // 2 - s)))
        R = R + coeff * rho ** (n - 2 * s)
    return R


def zernike(j: int, rho: np.ndarray, theta: np.ndarray,
            mask: np.ndarray | None = None) -> np.ndarray:
    """Evaluate Noll-indexed Zernike polynomial Z_j at polar coordinates.

    Z_j(rho, theta) = N * R_n^|m|(rho) * cos(|m|*theta)  if m >= 0
                    = N * R_n^|m|(rho) * sin(|m|*theta)  if m < 0

    where N = sqrt(2(n+1)) for m != 0, sqrt(n+1) for m = 0.

    Points outside the unit circle (rho > 1) are set to 0 (or NaN).

    Args:
        j: Noll index (>= 1).
        rho: Radial coordinate array.
        theta: Angular coordinate array (same shape as rho).
        mask: Optional boolean mask (True inside pupil). If None, uses rho <= 1.

    Returns:
        Zernike polynomial values, same shape as rho.
    """
    n, m = noll_to_nm(j)
    m_abs = abs(m)
    # Normalization factor
    norm = np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)
    R = zernike_radial(n, m_abs, rho)
    if m >= 0:
        Z = norm * R * np.cos(m_abs * theta)
    else:
        Z = norm * R * np.sin(m_abs * theta)
    # Mask outside unit circle
    if mask is not None:
        Z = np.where(mask, Z, np.nan)
    else:
        Z = np.where(rho <= 1.0, Z, np.nan)
    return Z


# ---------------------------------------------------------------------------
# Wavefront fitting
# ---------------------------------------------------------------------------

def zernike_fit(wavefront: np.ndarray, n_modes: int,
                rho: np.ndarray, theta: np.ndarray,
                mask: np.ndarray) -> np.ndarray:
    """Fit Zernike coefficients to a wavefront via least-squares.

    Builds the design matrix [Z_1(r_k), Z_2(r_k), ..., Z_J(r_k)] at
    valid (in-pupil) pixel locations and solves using numpy's lstsq
    (SVD-based for numerical stability).

    Args:
        wavefront: 2D measured wavefront array.
        n_modes: Number of Zernike modes to fit (j = 1..n_modes).
        rho: 2D radial coordinate array (same shape as wavefront).
        theta: 2D angular coordinate array.
        mask: 2D boolean array (True inside pupil).

    Returns:
        1D array of fitted Zernike coefficients (length n_modes).
    """
    w = wavefront[mask].ravel()
    r = rho[mask].ravel()
    t = theta[mask].ravel()
    # Design matrix: each column is Z_j at valid pixels
    A = np.column_stack([
        zernike(j, r, t, mask=np.ones_like(r, dtype=bool))
        for j in range(1, n_modes + 1)
    ])
    # Handle any NaN from zernike (shouldn't occur inside mask, but safety)
    A = np.nan_to_num(A, nan=0.0)
    coeffs, _, _, _ = np.linalg.lstsq(A, w, rcond=None)
    return coeffs


def reconstruct_wavefront(coeffs: np.ndarray, rho: np.ndarray,
                          theta: np.ndarray,
                          mask: np.ndarray) -> np.ndarray:
    """Reconstruct a wavefront from Zernike coefficients.

    W(rho, theta) = sum_j a_j * Z_j(rho, theta)
    """
    wf = np.zeros_like(rho, dtype=float)
    for j, a in enumerate(coeffs, start=1):
        Z = zernike(j, rho, theta, mask)
        wf += a * np.nan_to_num(Z, nan=0.0)
    wf[~mask] = np.nan
    return wf


# ---------------------------------------------------------------------------
# Kolmogorov turbulence phase screens
# ---------------------------------------------------------------------------

def kolmogorov_phase_screen(N: int, r0: float, L: float,
                            seed: int | None = None) -> np.ndarray:
    """Generate a Kolmogorov turbulence phase screen (FFT method).

    Produces a random phase screen with the correct spatial statistics
    for Kolmogorov turbulence: D_phi(r) = 6.88 * (r/r0)^(5/3).

    The method filters white noise by the square root of the Kolmogorov
    power spectral density in Fourier space, then inverse-transforms.

    Args:
        N: Grid size (N x N pixels).
        r0: Fried parameter in physical units (e.g., meters).
        L: Physical side length of the screen (same units as r0).
        seed: Random seed for reproducibility.

    Returns:
        2D phase screen in radians, shape (N, N).
    """
    rng = np.random.default_rng(seed)
    # Spatial frequency grids
    fx = np.fft.fftfreq(N, d=L / N)
    fy = np.fft.fftfreq(N, d=L / N)
    Fx, Fy = np.meshgrid(fx, fy)
    f_mag = np.sqrt(Fx**2 + Fy**2)
    # Avoid division by zero at DC
    f_mag[0, 0] = 1.0
    # Kolmogorov PSD: Phi(f) = 0.023 * r0^(-5/3) * (2*pi*f)^(-11/3)
    psd = 0.023 * r0**(-5.0 / 3.0) * (2 * np.pi * f_mag)**(-11.0 / 3.0)
    psd[0, 0] = 0.0  # remove piston
    # Random complex field weighted by sqrt(PSD)
    cn = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    cn *= np.sqrt(psd) * (2 * np.pi / L)
    # Inverse FFT to spatial domain
    phi = np.real(np.fft.ifft2(cn)) * N**2
    return phi


# ---------------------------------------------------------------------------
# Strehl ratio utilities
# ---------------------------------------------------------------------------

def rms_from_zernike(coeffs: np.ndarray,
                     exclude_piston: bool = True) -> float:
    """Compute RMS wavefront error from Zernike coefficients.

    By orthonormality, sigma = sqrt(sum(a_j^2)) for j >= 2.
    Piston (j=1) is excluded by default since it does not degrade imaging.
    """
    start = 1 if exclude_piston else 0
    return float(np.sqrt(np.sum(coeffs[start:]**2)))


def strehl_marechal(rms_waves: float) -> float:
    """Estimate Strehl ratio via the Maréchal approximation.

    S ≈ exp(-(2*pi*sigma)^2), valid for sigma < lambda/4 (Strehl > ~0.1).

    Args:
        rms_waves: RMS wavefront error in units of wavelength.

    Returns:
        Estimated Strehl ratio (0 to 1).
    """
    return float(np.exp(-(2 * np.pi * rms_waves)**2))


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def polar_grid(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a polar coordinate grid on the unit circle.

    Returns rho, theta, and a boolean mask (True inside unit circle).
    The grid spans [-1, 1] in both x and y.
    """
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    mask = rho <= 1.0
    return rho, theta, mask


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_zernike_gallery(n_modes: int = 21, grid_size: int = 256) -> None:
    """Plot a gallery of the first n_modes Zernike polynomials.

    Displays each mode on the unit circle with its Noll index,
    (n, m) quantum numbers, and common name.
    """
    # Standard aberration names for the first 21 modes
    names = {
        1: "Piston", 2: "Tilt X", 3: "Tilt Y",
        4: "Defocus", 5: "Astig. (oblique)", 6: "Astig. (vert.)",
        7: "Coma Y", 8: "Coma X", 9: "Trefoil Y", 10: "Trefoil X",
        11: "Spherical", 12: "2nd Astig. V", 13: "2nd Astig. O",
        14: "Quadrafoil V", 15: "Quadrafoil O",
        16: "2nd Coma X", 17: "2nd Coma Y",
        18: "2nd Trefoil X", 19: "2nd Trefoil Y",
        20: "Pentafoil X", 21: "Pentafoil Y",
    }
    rho, theta, mask = polar_grid(grid_size)
    # Determine grid layout
    ncols = 5
    nrows = (n_modes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.ravel()
    for idx in range(n_modes):
        j = idx + 1
        n, m = noll_to_nm(j)
        Z = zernike(j, rho, theta, mask)
        ax = axes[idx]
        im = ax.imshow(Z, cmap='RdBu_r', extent=[-1, 1, -1, 1],
                       origin='lower', vmin=-3, vmax=3)
        name = names.get(j, "")
        ax.set_title(f"j={j}  ({n},{m})\n{name}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    # Hide unused axes
    for idx in range(n_modes, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Zernike Polynomial Gallery (Noll indexing)", fontsize=14)
    fig.tight_layout()
    fig.savefig("11_zernike_gallery.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[Saved] 11_zernike_gallery.png")


def plot_wavefront_fitting(grid_size: int = 256) -> None:
    """Demonstrate wavefront fitting: synthesize → fit → compare.

    Creates a synthetic aberrated wavefront from known Zernike coefficients,
    fits it, and shows the original, fitted, and residual wavefronts.
    """
    rho, theta, mask = polar_grid(grid_size)
    # Synthetic wavefront with known coefficients (in waves)
    true_coeffs = np.zeros(15)
    true_coeffs[3] = 0.30   # j=4: defocus
    true_coeffs[4] = -0.15  # j=5: astigmatism (oblique)
    true_coeffs[6] = 0.12   # j=7: coma Y
    true_coeffs[10] = 0.20  # j=11: spherical
    # Generate wavefront
    wf_true = reconstruct_wavefront(true_coeffs, rho, theta, mask)
    # Add small noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.02, wf_true.shape)
    noise[~mask] = 0
    wf_noisy = wf_true + noise
    wf_noisy[~mask] = np.nan
    # Fit Zernike coefficients
    fitted_coeffs = zernike_fit(np.nan_to_num(wf_noisy, nan=0.0),
                                15, rho, theta, mask)
    wf_fitted = reconstruct_wavefront(fitted_coeffs, rho, theta, mask)
    residual = wf_noisy - wf_fitted
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    vmin, vmax = -0.5, 0.5
    for ax, data, title in zip(
        axes[:3],
        [wf_noisy, wf_fitted, residual],
        ["Input (noisy)", "Fitted (15 modes)", "Residual"]
    ):
        im = ax.imshow(data, cmap='RdBu_r', extent=[-1, 1, -1, 1],
                       origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, label='waves')
    # Coefficient comparison bar chart
    ax = axes[3]
    j_indices = np.arange(1, 16)
    width = 0.35
    ax.bar(j_indices - width / 2, true_coeffs, width, label='True', alpha=0.7)
    ax.bar(j_indices + width / 2, fitted_coeffs, width, label='Fitted', alpha=0.7)
    ax.set_xlabel("Noll index j")
    ax.set_ylabel("Coefficient (waves)")
    ax.legend(fontsize=8)
    ax.set_title("Coefficient Comparison")
    ax.grid(True, alpha=0.3)
    fig.suptitle("Wavefront Fitting with Zernike Polynomials", fontsize=14)
    fig.tight_layout()
    fig.savefig("11_wavefront_fitting.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[Saved] 11_wavefront_fitting.png")
    # Print metrics
    rms_true = rms_from_zernike(true_coeffs)
    rms_fitted = rms_from_zernike(fitted_coeffs)
    print(f"  True RMS: {rms_true:.4f} waves  →  Strehl: {strehl_marechal(rms_true):.3f}")
    print(f"  Fitted RMS: {rms_fitted:.4f} waves  →  Strehl: {strehl_marechal(rms_fitted):.3f}")


def plot_phase_screen(N: int = 512, r0: float = 0.15,
                      L: float = 4.0) -> None:
    """Generate and display a Kolmogorov turbulence phase screen.

    Also fits Zernike modes and shows the coefficient distribution.
    """
    phi = kolmogorov_phase_screen(N, r0, L, seed=2024)
    # Extract the circular aperture (telescope of diameter D = L)
    rho, theta, mask = polar_grid(N)
    phi_pupil = phi.copy()
    phi_pupil[~mask] = np.nan
    # Fit Zernike modes
    n_modes = 36
    coeffs = zernike_fit(np.nan_to_num(phi_pupil, nan=0.0),
                         n_modes, rho, theta, mask)
    # Reconstruct and compute residual
    wf_fitted = reconstruct_wavefront(coeffs, rho, theta, mask)
    residual = phi_pupil - wf_fitted
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # Phase screen
    im0 = axes[0].imshow(phi_pupil, cmap='RdBu_r',
                         extent=[-L/2, L/2, -L/2, L/2], origin='lower')
    axes[0].set_title(f"Kolmogorov Phase Screen\n(r₀={r0*100:.0f} cm, D={L:.0f} m)")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axes[0], label='Phase (rad)')
    # Zernike coefficients
    j_arr = np.arange(2, n_modes + 1)  # skip piston
    axes[1].bar(j_arr, np.abs(coeffs[1:]), color='steelblue', alpha=0.8)
    axes[1].set_xlabel("Noll index j")
    axes[1].set_ylabel("|a_j| (rad)")
    axes[1].set_title(f"Zernike Coefficients ({n_modes} modes)")
    axes[1].grid(True, alpha=0.3)
    # Residual after removing modes
    rms_total = np.nanstd(phi_pupil)
    rms_residual = np.nanstd(residual)
    im2 = axes[2].imshow(residual, cmap='RdBu_r',
                         extent=[-L/2, L/2, -L/2, L/2], origin='lower')
    axes[2].set_title(f"Residual after {n_modes} modes\n"
                      f"RMS: {rms_total:.2f} → {rms_residual:.2f} rad")
    axes[2].set_xlabel("x (m)")
    fig.colorbar(im2, ax=axes[2], label='Phase (rad)')
    fig.tight_layout()
    fig.savefig("11_phase_screen.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[Saved] 11_phase_screen.png")
    # Variance breakdown
    var_tiptilt = coeffs[1]**2 + coeffs[2]**2
    var_total = np.sum(coeffs[1:]**2)
    print(f"  Total phase variance (from Zernike): {var_total:.2f} rad²")
    print(f"  Tip-tilt fraction: {var_tiptilt/var_total*100:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Zernike Polynomials: Modes, Fitting, and Turbulence")
    print("=" * 60)

    print("\n1. Generating Zernike mode gallery (21 modes)...")
    plot_zernike_gallery(n_modes=21)

    print("\n2. Wavefront fitting demonstration...")
    plot_wavefront_fitting()

    print("\n3. Kolmogorov phase screen simulation...")
    plot_phase_screen()

    print("\nDone. All plots saved.")
