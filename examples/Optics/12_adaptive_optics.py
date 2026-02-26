#!/usr/bin/env python3
"""Adaptive Optics Simulation: Wavefront Sensing, Reconstruction, and Correction.

This module simulates a complete adaptive optics (AO) system:

1. Shack-Hartmann wavefront sensor — divides the pupil into subapertures
   and measures local wavefront slopes from spot displacements
2. Deformable mirror — models a continuous facesheet DM with Gaussian
   influence functions driven by an array of actuators
3. Interaction matrix — maps DM actuator commands to WFS slope responses,
   calibrated by poking each actuator
4. Closed-loop control — runs a temporal integrator that iteratively
   corrects the atmospheric turbulence

Physics background:
- Atmospheric turbulence (Kolmogorov model) limits telescope resolution to
  lambda/r0 instead of the diffraction limit lambda/D. The Fried parameter
  r0 is typically 10-20 cm at visible wavelengths.
- The Greenwood frequency f_G = 0.427 * v_eff / r0 sets the required AO
  bandwidth. Typical values: 20-50 Hz.
- A Shack-Hartmann sensor measures average wavefront slopes over each
  subaperture. The reconstructor converts slopes to DM commands.
- The integrator controller updates: c_{k+1} = c_k + g * R @ s_k
  where g is the loop gain (0 < g <= 1), R the command matrix, s_k slopes.
- The Strehl ratio S ≈ exp(-(2*pi*sigma)^2) measures how close the
  corrected PSF peak is to the diffraction limit.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Grid utilities (shared with 11_zernike_polynomials.py)
# ---------------------------------------------------------------------------

def polar_grid(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a polar coordinate grid on the unit circle."""
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    mask = rho <= 1.0
    return rho, theta, mask


def circular_mask(N: int) -> np.ndarray:
    """Boolean mask: True inside the unit circle on an NxN grid."""
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    return (X**2 + Y**2) <= 1.0


# ---------------------------------------------------------------------------
# Kolmogorov phase screen (reproduced here for self-containment)
# ---------------------------------------------------------------------------

def kolmogorov_phase_screen(N: int, r0: float, L: float,
                            seed: int | None = None) -> np.ndarray:
    """Generate a Kolmogorov turbulence phase screen via FFT filtering.

    The phase screen has spatial statistics matching:
        D_phi(r) = 6.88 * (r / r0)^(5/3)

    Args:
        N: Grid size (N x N pixels).
        r0: Fried parameter (same units as L).
        L: Physical side length of the screen.
        seed: Random seed for reproducibility.

    Returns:
        Phase screen in radians, shape (N, N).
    """
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(N, d=L / N)
    fy = np.fft.fftfreq(N, d=L / N)
    Fx, Fy = np.meshgrid(fx, fy)
    f_mag = np.sqrt(Fx**2 + Fy**2)
    f_mag[0, 0] = 1.0
    psd = 0.023 * r0**(-5.0 / 3.0) * (2 * np.pi * f_mag)**(-11.0 / 3.0)
    psd[0, 0] = 0.0
    cn = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    cn *= np.sqrt(psd) * (2 * np.pi / L)
    phi = np.real(np.fft.ifft2(cn)) * N**2
    return phi


# ---------------------------------------------------------------------------
# Shack-Hartmann wavefront sensor
# ---------------------------------------------------------------------------

class ShackHartmann:
    """Shack-Hartmann wavefront sensor simulation.

    The pupil is divided into n_sub x n_sub subapertures. For each
    subaperture, the average wavefront slope (x and y) is measured
    using finite differences — simulating what real centroiding
    algorithms compute from spot displacements.

    Attributes:
        n_sub: Number of subapertures across the diameter.
        valid: Boolean mask of which subapertures are inside the pupil.
        n_valid: Number of valid (illuminated) subapertures.
    """

    def __init__(self, n_sub: int, pupil_diameter_px: int):
        """Initialize the WFS geometry.

        Args:
            n_sub: Subapertures across the pupil diameter.
            pupil_diameter_px: Wavefront grid size in pixels.
        """
        self.n_sub = n_sub
        self.sub_size = pupil_diameter_px // n_sub
        self.N = pupil_diameter_px
        # Determine which subapertures are inside the circular pupil
        # (center of each subaperture must be within 90% of the radius)
        centers = (np.arange(n_sub) + 0.5) / n_sub * 2 - 1  # [-1, 1]
        cx, cy = np.meshgrid(centers, centers)
        self.valid = (cx**2 + cy**2) <= 0.9**2
        self.n_valid = int(np.sum(self.valid))

    def measure_slopes(self, wavefront: np.ndarray) -> np.ndarray:
        """Measure wavefront slopes in each valid subaperture.

        Returns a 1D array: [sx_1, sx_2, ..., sy_1, sy_2, ...] for
        the n_valid subapertures. Slopes are in radians per pixel.

        Args:
            wavefront: 2D phase array (radians), shape (N, N).

        Returns:
            Slope vector of length 2 * n_valid.
        """
        sx_list = []
        sy_list = []
        for i in range(self.n_sub):
            for j in range(self.n_sub):
                if not self.valid[i, j]:
                    continue
                r0 = i * self.sub_size
                r1 = r0 + self.sub_size
                c0 = j * self.sub_size
                c1 = c0 + self.sub_size
                sub = wavefront[r0:r1, c0:c1]
                # Average slope via finite differences
                sx = np.mean(np.diff(sub, axis=1))
                sy = np.mean(np.diff(sub, axis=0))
                sx_list.append(sx)
                sy_list.append(sy)
        return np.concatenate([sx_list, sy_list])


# ---------------------------------------------------------------------------
# Deformable mirror
# ---------------------------------------------------------------------------

class DeformableMirror:
    """Continuous facesheet deformable mirror with Gaussian influence functions.

    Each actuator produces a Gaussian bump on the mirror surface. The
    total surface is the sum of all actuator contributions:
        W_DM(x, y) = sum_k c_k * exp(-ln2 * |r - r_k|^2 / w^2)

    Attributes:
        n_act: Actuators across the diameter.
        coupling: Fraction of stroke seen by nearest neighbor.
    """

    def __init__(self, n_act: int, N: int, coupling: float = 0.15):
        """Initialize DM geometry.

        Args:
            n_act: Actuators across the pupil diameter.
            N: Wavefront grid size in pixels.
            coupling: Inter-actuator coupling (0 to 1).
        """
        self.n_act = n_act
        self.N = N
        self.coupling = coupling
        self.n_actuators = n_act * n_act
        # Actuator positions in pixel coordinates
        act_pos = np.linspace(0, N - 1, n_act)
        ax, ay = np.meshgrid(act_pos, act_pos)
        self.act_x = ax.ravel()
        self.act_y = ay.ravel()
        # Influence function width from coupling
        pitch = N / (n_act - 1) if n_act > 1 else N
        self.w = pitch / np.sqrt(-np.log(coupling) / np.log(2))
        # Precompute pupil mask for valid actuators
        cx = self.act_x / (N - 1) * 2 - 1
        cy = self.act_y / (N - 1) * 2 - 1
        self.act_valid = (cx**2 + cy**2) <= 1.1**2
        self.commands = np.zeros(self.n_actuators)

    def surface(self, commands: np.ndarray | None = None) -> np.ndarray:
        """Compute the DM surface from actuator commands.

        Args:
            commands: 1D array of actuator commands. If None, use stored.

        Returns:
            2D surface array, shape (N, N).
        """
        if commands is not None:
            self.commands = commands.copy()
        y, x = np.mgrid[0:self.N, 0:self.N].astype(float)
        surf = np.zeros((self.N, self.N))
        for k in range(self.n_actuators):
            if abs(self.commands[k]) < 1e-15:
                continue
            r2 = (x - self.act_x[k])**2 + (y - self.act_y[k])**2
            surf += self.commands[k] * np.exp(-np.log(2) * r2 / self.w**2)
        return surf


# ---------------------------------------------------------------------------
# Interaction matrix and reconstructor
# ---------------------------------------------------------------------------

def build_interaction_matrix(wfs: ShackHartmann,
                             dm: DeformableMirror,
                             poke_amplitude: float = 1.0) -> np.ndarray:
    """Build the interaction matrix by poking each DM actuator.

    For each actuator k, apply a unit poke, measure the WFS response,
    and record it as column k of the matrix M.
    The command matrix R = pinv(M) converts slopes to commands.

    Args:
        wfs: ShackHartmann sensor instance.
        dm: DeformableMirror instance.
        poke_amplitude: Actuator poke amplitude.

    Returns:
        Interaction matrix M of shape (2 * n_valid, n_actuators).
    """
    n_slopes = 2 * wfs.n_valid
    M = np.zeros((n_slopes, dm.n_actuators))
    for k in range(dm.n_actuators):
        # Poke actuator k
        cmd = np.zeros(dm.n_actuators)
        cmd[k] = poke_amplitude
        surface = dm.surface(cmd)
        # Measure WFS response
        slopes = wfs.measure_slopes(surface)
        M[:, k] = slopes / poke_amplitude
    return M


def compute_reconstructor(M: np.ndarray,
                          n_modes_cut: int = 0) -> np.ndarray:
    """Compute the command matrix (reconstructor) via SVD pseudo-inverse.

    Optionally truncates the smallest singular values to reduce noise
    amplification.

    Args:
        M: Interaction matrix, shape (n_slopes, n_actuators).
        n_modes_cut: Number of lowest singular values to truncate.

    Returns:
        Command matrix R, shape (n_actuators, n_slopes).
    """
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    # Truncate small singular values (noise suppression)
    s_inv = np.zeros_like(s)
    n_keep = len(s) - n_modes_cut
    s_inv[:n_keep] = 1.0 / s[:n_keep]
    R = (Vt.T * s_inv) @ U.T
    return R


# ---------------------------------------------------------------------------
# Closed-loop simulation
# ---------------------------------------------------------------------------

def ao_closed_loop(n_steps: int, wfs: ShackHartmann, dm: DeformableMirror,
                   R: np.ndarray, gain: float, r0: float, L: float,
                   base_seed: int = 1000) -> dict:
    """Run a closed-loop AO simulation.

    At each time step, a new atmospheric phase screen is generated
    (simulating frozen-flow turbulence), the DM correction is applied,
    and the WFS measures the residual. The integrator updates the DM.

    Args:
        n_steps: Number of time steps.
        wfs: Shack-Hartmann sensor.
        dm: Deformable mirror.
        R: Command matrix (reconstructor).
        gain: Integrator gain (0 < gain <= 1).
        r0: Fried parameter.
        L: Physical aperture diameter.
        base_seed: Base random seed.

    Returns:
        Dict with 'strehls', 'rms_open', 'rms_closed' arrays.
    """
    N = dm.N
    mask = circular_mask(N)
    dm.commands = np.zeros(dm.n_actuators)
    strehls = []
    rms_open_list = []
    rms_closed_list = []

    for step in range(n_steps):
        # Generate atmospheric phase screen for this time step
        atm = kolmogorov_phase_screen(N, r0, L, seed=base_seed + step)
        # Apply DM correction
        dm_surf = dm.surface()
        residual = atm - dm_surf
        # Metrics
        rms_open = np.std(atm[mask])
        rms_closed = np.std(residual[mask])
        # Strehl ratio (in radians, so sigma is already in rad)
        strehl = np.exp(-rms_closed**2) if rms_closed < 3.0 else 0.0
        strehls.append(strehl)
        rms_open_list.append(rms_open)
        rms_closed_list.append(rms_closed)
        # WFS measurement on residual wavefront
        slopes = wfs.measure_slopes(residual)
        # Integrator update
        correction = R @ slopes
        dm.commands += gain * correction

    return {
        'strehls': np.array(strehls),
        'rms_open': np.array(rms_open_list),
        'rms_closed': np.array(rms_closed_list),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_sh_spots(wavefront: np.ndarray, wfs: ShackHartmann) -> None:
    """Plot the Shack-Hartmann spot pattern for an aberrated wavefront.

    Shows the subaperture grid overlaid on the wavefront, with arrows
    indicating measured slopes (spot displacements).
    """
    slopes = wfs.measure_slopes(wavefront)
    n_valid = wfs.n_valid
    sx = slopes[:n_valid]
    sy = slopes[n_valid:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Wavefront with subaperture grid
    mask = circular_mask(wfs.N)
    wf_display = wavefront.copy()
    wf_display[~mask] = np.nan
    axes[0].imshow(wf_display, cmap='RdBu_r', origin='lower')
    axes[0].set_title("Wavefront with Subaperture Grid")
    # Draw subaperture grid
    for i in range(wfs.n_sub + 1):
        pos = i * wfs.sub_size
        axes[0].axhline(pos, color='gray', linewidth=0.5, alpha=0.5)
        axes[0].axvline(pos, color='gray', linewidth=0.5, alpha=0.5)

    # Slope quiver plot
    centers = (np.arange(wfs.n_sub) + 0.5) / wfs.n_sub * 2 - 1
    cx, cy = np.meshgrid(centers, centers)
    valid_cx = cx[wfs.valid]
    valid_cy = cy[wfs.valid]
    # Scale arrows for visibility
    scale = 0.3 / (np.max(np.abs(slopes)) + 1e-10)
    axes[1].quiver(valid_cx, valid_cy, sx * scale, sy * scale,
                   angles='xy', scale_units='xy', scale=1,
                   color='steelblue', width=0.008)
    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    axes[1].plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
    axes[1].set_xlim(-1.2, 1.2)
    axes[1].set_ylim(-1.2, 1.2)
    axes[1].set_aspect('equal')
    axes[1].set_title("WFS Slope Measurements")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Shack-Hartmann Wavefront Sensor", fontsize=14)
    fig.tight_layout()
    fig.savefig("12_sh_spots.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[Saved] 12_sh_spots.png")


def plot_ao_correction(r0: float = 0.15, L: float = 2.0,
                       N: int = 128, n_sub: int = 10,
                       n_act: int = 12) -> None:
    """Show before/after wavefront and PSF for AO correction.

    A single atmospheric phase screen is corrected in one step (open-loop)
    to illustrate the effect of AO.
    """
    # Generate turbulence
    atm = kolmogorov_phase_screen(N, r0, L, seed=42)
    mask = circular_mask(N)
    # Setup AO system
    wfs = ShackHartmann(n_sub, N)
    dm = DeformableMirror(n_act, N, coupling=0.15)
    M = build_interaction_matrix(wfs, dm)
    R = compute_reconstructor(M, n_modes_cut=2)
    # Measure and correct (single step, high gain for demonstration)
    slopes = wfs.measure_slopes(atm)
    dm.commands = R @ slopes
    dm_surf = dm.surface()
    corrected = atm - dm_surf

    # Compute PSFs
    pupil = mask.astype(float)
    psf_diff = np.abs(np.fft.fftshift(np.fft.fft2(pupil)))**2
    psf_open = np.abs(np.fft.fftshift(
        np.fft.fft2(pupil * np.exp(1j * atm))))**2
    psf_closed = np.abs(np.fft.fftshift(
        np.fft.fft2(pupil * np.exp(1j * corrected))))**2
    # Normalize
    psf_diff /= psf_diff.max()
    psf_open /= psf_diff.max()  # normalize to diffraction limit
    psf_closed /= psf_diff.max()

    rms_open = np.std(atm[mask])
    rms_closed = np.std(corrected[mask])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # Top row: wavefronts
    vmax = max(np.nanmax(np.abs(atm[mask])), 1.0)
    for ax, data, title in zip(
        axes[0],
        [atm, dm_surf, corrected],
        [f"Atmospheric (RMS={rms_open:.2f} rad)",
         "DM Surface",
         f"Corrected (RMS={rms_closed:.2f} rad)"]
    ):
        display = data.copy()
        display[~mask] = np.nan
        im = ax.imshow(display, cmap='RdBu_r', origin='lower',
                       vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, label='rad')

    # Bottom row: PSFs (log scale)
    # Crop central region for visibility
    c = N // 2
    hw = N // 8  # half-width of zoom window
    for ax, data, title in zip(
        axes[1],
        [psf_diff[c-hw:c+hw, c-hw:c+hw],
         psf_open[c-hw:c+hw, c-hw:c+hw],
         psf_closed[c-hw:c+hw, c-hw:c+hw]],
        ["Diffraction Limit",
         f"Seeing-Limited (Strehl={psf_open.max():.3f})",
         f"AO-Corrected (Strehl={psf_closed.max():.3f})"]
    ):
        ax.imshow(data**0.3, cmap='inferno', origin='lower')
        ax.set_title(title)

    fig.suptitle(f"Adaptive Optics Correction ({n_sub}×{n_sub} WFS, "
                 f"{n_act}×{n_act} DM, r₀={r0*100:.0f} cm)", fontsize=14)
    fig.tight_layout()
    fig.savefig("12_ao_correction.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[Saved] 12_ao_correction.png")


def plot_strehl_convergence(r0: float = 0.15, L: float = 2.0,
                            N: int = 128, n_sub: int = 10,
                            n_act: int = 12, n_steps: int = 50) -> None:
    """Plot Strehl ratio convergence for different loop gains."""
    gains = [0.3, 0.5, 0.7, 0.9]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for g in gains:
        wfs = ShackHartmann(n_sub, N)
        dm = DeformableMirror(n_act, N, coupling=0.15)
        M = build_interaction_matrix(wfs, dm)
        R = compute_reconstructor(M, n_modes_cut=2)
        result = ao_closed_loop(n_steps, wfs, dm, R, g, r0, L)
        axes[0].plot(result['strehls'], label=f'g={g}')
        axes[1].plot(result['rms_closed'], label=f'g={g}')

    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Strehl ratio")
    axes[0].set_title("Strehl Ratio Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("RMS wavefront error (rad)")
    axes[1].set_title("Residual RMS vs Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"AO Loop Convergence (r₀={r0*100:.0f} cm, "
                 f"{n_sub}×{n_sub} WFS, {n_act}×{n_act} DM)", fontsize=14)
    fig.tight_layout()
    fig.savefig("12_strehl_convergence.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[Saved] 12_strehl_convergence.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Adaptive Optics Simulation")
    print("=" * 60)

    # Parameters
    N = 128       # Grid size (small for speed; increase for accuracy)
    r0 = 0.15     # Fried parameter (meters)
    L = 2.0       # Telescope diameter (meters)
    n_sub = 10    # WFS subapertures across diameter
    n_act = 12    # DM actuators across diameter

    print(f"\nSystem: {L:.0f} m telescope, r₀={r0*100:.0f} cm")
    print(f"  D/r₀ = {L/r0:.1f}")
    print(f"  WFS: {n_sub}×{n_sub} subapertures")
    print(f"  DM: {n_act}×{n_act} actuators")

    print("\n1. Shack-Hartmann spot simulation...")
    atm = kolmogorov_phase_screen(N, r0, L, seed=42)
    wfs = ShackHartmann(n_sub, N)
    plot_sh_spots(atm, wfs)

    print("\n2. AO correction (before/after)...")
    plot_ao_correction(r0, L, N, n_sub, n_act)

    print("\n3. Closed-loop convergence for different gains...")
    plot_strehl_convergence(r0, L, N, n_sub, n_act, n_steps=50)

    print("\nDone. All plots saved.")
