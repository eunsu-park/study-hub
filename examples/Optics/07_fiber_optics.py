#!/usr/bin/env python3
"""Fiber Optics: Modes, Dispersion, Attenuation, and Pulse Broadening.

This module explores the physics and engineering of optical fibers:

1. Step-index fiber parameters — Numerical Aperture (NA), V-number, mode count
2. Modal dispersion — pulse broadening from different ray paths in multimode fiber
3. Chromatic dispersion — material and waveguide contributions to group velocity dispersion
4. Attenuation budget — power loss calculation for a fiber link
5. Pulse broadening simulation — how an input pulse spreads during propagation

Physics background:
- Optical fibers guide light by total internal reflection (TIR). The core has
  a higher refractive index than the cladding: n_core > n_clad.
- The Numerical Aperture NA = sqrt(n_core^2 - n_clad^2) determines the
  maximum acceptance angle for light entering the fiber.
- The V-number V = 2*pi*a*NA/lambda determines the number of guided modes.
  Single-mode operation requires V < 2.405 (the first zero of J0).
- Chromatic dispersion has two components:
  (a) Material dispersion: n depends on wavelength (Sellmeier equation)
  (b) Waveguide dispersion: mode confinement depends on wavelength
  These can be engineered to partially cancel at specific wavelengths.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn_zeros  # Zeros of Bessel functions


# ---------------------------------------------------------------------------
# Fiber parameters
# ---------------------------------------------------------------------------

class StepIndexFiber:
    """Model of a step-index optical fiber.

    A step-index fiber has a uniform core refractive index n_core surrounded
    by a uniform cladding n_clad. This is the simplest fiber type and
    illustrates all fundamental fiber-optic concepts.

    Attributes:
        n_core: Core refractive index.
        n_clad: Cladding refractive index.
        core_radius: Core radius (meters).
        NA: Numerical aperture.
        delta: Relative index difference.
    """

    def __init__(self, n_core: float, n_clad: float, core_radius: float):
        """Initialize fiber with core/cladding indices and core radius.

        Args:
            n_core: Core refractive index.
            n_clad: Cladding refractive index.
            core_radius: Core radius in meters.
        """
        self.n_core = n_core
        self.n_clad = n_clad
        self.core_radius = core_radius

        # Numerical aperture: determines the cone of light the fiber can accept
        # Derived from the condition for TIR at the core-cladding interface
        self.NA = np.sqrt(n_core**2 - n_clad**2)

        # Relative index difference (weak-guidance parameter)
        # Most practical fibers have delta << 1 (weakly guiding)
        self.delta = (n_core - n_clad) / n_core

    def v_number(self, wavelength: float) -> float:
        """Compute the V-number (normalized frequency) at a given wavelength.

        The V-number determines how many modes the fiber supports.
        V < 2.405: single-mode operation (only the fundamental LP01 mode)
        V > 2.405: multimode operation

        The V-number is analogous to the Fresnel number in free-space optics.

        Args:
            wavelength: Wavelength in meters.

        Returns:
            V-number (dimensionless).
        """
        return 2 * np.pi * self.core_radius * self.NA / wavelength

    def cutoff_wavelength(self) -> float:
        """Compute the single-mode cutoff wavelength.

        Below this wavelength, the fiber supports multiple modes.
        Above it, only the fundamental LP01 mode propagates.

        Returns:
            Cutoff wavelength in meters.
        """
        # V = 2.405 at cutoff
        return 2 * np.pi * self.core_radius * self.NA / 2.405

    def num_modes(self, wavelength: float) -> int:
        """Estimate the number of guided modes at a given wavelength.

        For a step-index fiber with large V-number, the approximate
        number of modes is V^2 / 2 (for each polarization).

        For small V, we count the LP modes by finding zeros of Bessel functions.

        Args:
            wavelength: Wavelength in meters.

        Returns:
            Approximate number of guided modes (including degeneracies).
        """
        V = self.v_number(wavelength)
        if V < 2.405:
            return 1  # Single mode
        # Approximate formula for large V (step-index)
        return int(V**2 / 2)

    def acceptance_angle(self) -> float:
        """Compute the maximum acceptance half-angle in air.

        Light entering at angles greater than this is not guided.

        Returns:
            Acceptance half-angle in radians.
        """
        return np.arcsin(self.NA)

    def modal_dispersion_delay(self, length: float) -> float:
        """Compute the modal dispersion (maximum time delay between modes).

        The fastest mode travels along the fiber axis, while the slowest
        mode bounces at the critical angle. The time difference per unit length
        is: delta_t = (n_core * L / c) * delta

        This limits the bandwidth of multimode fibers.

        Args:
            length: Fiber length in meters.

        Returns:
            Time spread in seconds.
        """
        c = 3e8  # speed of light
        return self.n_core * length * self.delta / c


# ---------------------------------------------------------------------------
# Chromatic dispersion model
# ---------------------------------------------------------------------------

def sellmeier_silica(wavelength_um: float) -> float:
    """Refractive index of fused silica using the Sellmeier equation.

    Fused silica (SiO2) is the primary material in optical fibers.
    These coefficients are well-established from spectroscopic measurements.

    Args:
        wavelength_um: Wavelength in micrometers.

    Returns:
        Refractive index.
    """
    # Sellmeier coefficients for fused silica
    B = [0.6961663, 0.4079426, 0.8974794]
    C = [0.0684043**2, 0.1162414**2, 9.896161**2]
    lam2 = wavelength_um**2
    n_sq = 1.0
    for b, c in zip(B, C):
        n_sq += b * lam2 / (lam2 - c)
    return np.sqrt(n_sq)


def material_dispersion(wavelength_um: np.ndarray) -> np.ndarray:
    """Compute material dispersion parameter D_M for fused silica.

    Material dispersion arises because n(lambda) varies with wavelength.
    The dispersion parameter is:
        D_M = -(lambda/c) * d^2n/d(lambda)^2  [ps/(nm*km)]

    We compute d^2n/dlambda^2 numerically.

    Args:
        wavelength_um: Array of wavelengths in micrometers.

    Returns:
        Material dispersion D_M in ps/(nm*km).
    """
    c_light = 3e5  # km/s (for ps/nm/km units)
    dlam = 1e-4  # small increment for numerical differentiation (um)

    D_M = np.zeros_like(wavelength_um)
    for i, lam in enumerate(wavelength_um):
        n_plus = sellmeier_silica(lam + dlam)
        n_center = sellmeier_silica(lam)
        n_minus = sellmeier_silica(lam - dlam)
        d2n_dlam2 = (n_plus - 2 * n_center + n_minus) / dlam**2
        D_M[i] = -lam / c_light * d2n_dlam2 * 1e6  # convert to ps/(nm*km)
    return D_M


def group_index(wavelength_um: float) -> float:
    """Compute the group index n_g = n - lambda * dn/dlambda.

    The group index determines the speed of a pulse (group velocity = c/n_g),
    as opposed to the phase velocity (c/n). They differ because of dispersion.

    Args:
        wavelength_um: Wavelength in micrometers.

    Returns:
        Group index.
    """
    dlam = 1e-4
    n = sellmeier_silica(wavelength_um)
    dn_dlam = (sellmeier_silica(wavelength_um + dlam) -
               sellmeier_silica(wavelength_um - dlam)) / (2 * dlam)
    return n - wavelength_um * dn_dlam


# ---------------------------------------------------------------------------
# Visualization 1: Fiber parameters
# ---------------------------------------------------------------------------

def plot_fiber_parameters():
    """Visualize fundamental fiber parameters: V-number, mode count, NA."""
    # Standard single-mode fiber (SMF-28)
    smf = StepIndexFiber(n_core=1.4504, n_clad=1.4447, core_radius=4.1e-6)
    # Standard multimode fiber (OM3)
    mmf = StepIndexFiber(n_core=1.4620, n_clad=1.4470, core_radius=25e-6)

    print(f"\n  SMF-28:")
    print(f"    NA = {smf.NA:.4f}")
    print(f"    Acceptance angle = {np.degrees(smf.acceptance_angle()):.2f} deg")
    print(f"    V @ 1310 nm = {smf.v_number(1310e-9):.3f}")
    print(f"    V @ 1550 nm = {smf.v_number(1550e-9):.3f}")
    print(f"    Cutoff wavelength = {smf.cutoff_wavelength()*1e9:.0f} nm")

    print(f"\n  OM3 Multimode:")
    print(f"    NA = {mmf.NA:.4f}")
    print(f"    V @ 850 nm = {mmf.v_number(850e-9):.1f}")
    print(f"    Modes @ 850 nm = {mmf.num_modes(850e-9)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: V-number vs wavelength ---
    ax = axes[0, 0]
    wavelengths = np.linspace(600, 2000, 500)  # nm
    V_smf = [smf.v_number(w * 1e-9) for w in wavelengths]
    V_mmf = [mmf.v_number(w * 1e-9) for w in wavelengths]

    ax.plot(wavelengths, V_smf, 'b-', linewidth=2, label='SMF-28 (a=4.1 um)')
    ax.axhline(2.405, color='red', linestyle='--', label='Single-mode cutoff V=2.405')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('V-number')
    ax.set_title('V-Number vs Wavelength')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 6)

    # --- Panel 2: Number of modes vs core radius ---
    ax = axes[0, 1]
    radii_um = np.linspace(1, 50, 200)
    wavelengths_test = [850, 1310, 1550]
    for wl in wavelengths_test:
        modes = []
        for r in radii_um:
            fiber = StepIndexFiber(1.4504, 1.4447, r * 1e-6)
            modes.append(fiber.num_modes(wl * 1e-9))
        ax.plot(radii_um, modes, label=f'{wl} nm', linewidth=1.5)

    ax.set_xlabel('Core radius (um)')
    ax.set_ylabel('Number of modes')
    ax.set_title('Guided Modes vs Core Radius')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # --- Panel 3: Fiber cross-section and NA ---
    ax = axes[1, 0]
    theta_range = np.linspace(0, np.pi/2, 200)

    fibers = [
        (StepIndexFiber(1.450, 1.445, 4e-6), 'Low NA (0.060)'),
        (StepIndexFiber(1.460, 1.440, 25e-6), 'Medium NA (0.172)'),
        (StepIndexFiber(1.480, 1.440, 50e-6), 'High NA (0.249)'),
    ]

    for fiber, label in fibers:
        # Acceptance cone visualization: plot sin(theta) vs 1 for theta < acceptance
        theta_accept = fiber.acceptance_angle()
        theta_arr = np.linspace(0, theta_accept, 100)
        # Power coupling efficiency (simplified as cos^2 for Lambertian source)
        efficiency = np.cos(theta_arr)**2
        ax.plot(np.degrees(theta_arr), efficiency, linewidth=2,
                label=f'{label}, NA={fiber.NA:.3f}')
        ax.axvline(np.degrees(theta_accept), linestyle=':', alpha=0.3)

    ax.set_xlabel('Launch angle (degrees)')
    ax.set_ylabel('Coupling efficiency')
    ax.set_title('Fiber Acceptance Cone (NA)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Refractive index profiles ---
    ax = axes[1, 1]
    r = np.linspace(-50, 50, 1000)  # um

    # Step-index profile
    n_step = np.where(np.abs(r) <= 25, 1.462, 1.447)
    ax.plot(r, n_step, 'b-', linewidth=2, label='Step-index')

    # Graded-index profile (parabolic)
    alpha = 2.0  # parabolic
    a = 25  # core radius
    n_core, n_clad = 1.462, 1.447
    delta = (n_core - n_clad) / n_core
    n_graded = np.where(
        np.abs(r) <= a,
        n_core * np.sqrt(1 - 2 * delta * (np.abs(r) / a)**alpha),
        n_clad
    )
    ax.plot(r, n_graded, 'r-', linewidth=2, label='Graded-index (alpha=2)')

    ax.set_xlabel('Radial position (um)')
    ax.set_ylabel('Refractive index')
    ax.set_title('Fiber Index Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Optical Fiber Parameters', fontsize=14)
    fig.tight_layout()
    fig.savefig('07_fiber_parameters.png', dpi=150)
    plt.close(fig)
    print("[Saved] 07_fiber_parameters.png")


# ---------------------------------------------------------------------------
# Visualization 2: Chromatic dispersion
# ---------------------------------------------------------------------------

def plot_chromatic_dispersion():
    """Plot material and total dispersion of fused silica fiber.

    The zero-dispersion wavelength (~1310 nm for standard silica) is where
    material dispersion crosses zero. This is important for telecommunications
    because pulses do not broaden at this wavelength.
    """
    wavelengths = np.linspace(0.6, 1.8, 500)  # um

    # Material dispersion
    D_mat = material_dispersion(wavelengths)

    # Group index
    ng = np.array([group_index(w) for w in wavelengths])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Refractive index and group index ---
    ax = axes[0, 0]
    n_phase = np.array([sellmeier_silica(w) for w in wavelengths])
    ax.plot(wavelengths * 1000, n_phase, 'b-', linewidth=2, label='n (phase)')
    ax.plot(wavelengths * 1000, ng, 'r-', linewidth=2, label='n_g (group)')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Refractive index')
    ax.set_title('Phase and Group Index of Fused Silica')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark telecom windows
    for wl, name in [(850, 'O'), (1310, 'S'), (1550, 'C')]:
        ax.axvline(wl, color='gray', linestyle=':', alpha=0.3)
        ax.text(wl, ax.get_ylim()[1] * 0.999, name, ha='center', fontsize=8)

    # --- Panel 2: Material dispersion ---
    ax = axes[0, 1]
    ax.plot(wavelengths * 1000, D_mat, 'b-', linewidth=2)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('D (ps/(nm*km))')
    ax.set_title('Material Dispersion of Fused Silica')
    ax.grid(True, alpha=0.3)

    # Find zero-dispersion wavelength
    zero_cross_idx = np.where(np.diff(np.sign(D_mat)))[0]
    if len(zero_cross_idx) > 0:
        zd_wl = wavelengths[zero_cross_idx[0]] * 1000
        ax.axvline(zd_wl, color='red', linestyle='--',
                    label=f'Zero-dispersion: {zd_wl:.0f} nm')
        ax.legend()
        print(f"\n  Zero-dispersion wavelength: {zd_wl:.0f} nm")

    # --- Panel 3: Pulse broadening ---
    ax = axes[1, 0]
    # Time broadening: delta_t = |D| * delta_lambda * L
    L_km = 10  # 10 km fiber
    source_widths_nm = [0.1, 1.0, 5.0]  # source spectral widths

    for dw in source_widths_nm:
        delta_t = np.abs(D_mat) * dw * L_km  # ps
        ax.plot(wavelengths * 1000, delta_t, label=f'delta_lambda = {dw} nm')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Pulse broadening (ps)')
    ax.set_title(f'Chromatic Pulse Broadening ({L_km} km fiber)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Bandwidth-distance product ---
    ax = axes[1, 1]
    # BW ~ 0.44 / delta_t  (for Gaussian pulses)
    D_1550 = np.interp(1.55, wavelengths, D_mat)
    lengths_km = np.linspace(0.1, 100, 200)
    for dw in [0.01, 0.1, 1.0]:
        delta_t_ns = abs(D_1550) * dw * lengths_km * 1e-3  # convert ps to ns
        bw_ghz = 0.44 / np.maximum(delta_t_ns, 1e-6)
        ax.loglog(lengths_km, bw_ghz, label=f'delta_lambda = {dw} nm')

    ax.set_xlabel('Fiber length (km)')
    ax.set_ylabel('Bandwidth (GHz)')
    ax.set_title(f'Bandwidth vs Distance (at 1550 nm, D={D_1550:.1f} ps/nm/km)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    print(f"  Dispersion at 1550 nm: D = {D_1550:.2f} ps/(nm*km)")
    print(f"  Group index at 1550 nm: n_g = {group_index(1.55):.6f}")

    fig.suptitle('Chromatic Dispersion in Optical Fiber', fontsize=14)
    fig.tight_layout()
    fig.savefig('07_fiber_dispersion.png', dpi=150)
    plt.close(fig)
    print("[Saved] 07_fiber_dispersion.png")


# ---------------------------------------------------------------------------
# Visualization 3: Attenuation and link budget
# ---------------------------------------------------------------------------

def plot_attenuation_and_link():
    """Compute fiber attenuation spectrum and a simple link budget.

    The attenuation spectrum of silica fiber has:
    - High loss at short wavelengths (Rayleigh scattering ~ 1/lambda^4)
    - OH absorption peaks near 1383 nm (water peak)
    - Infrared absorption at long wavelengths
    - Minimum around 1550 nm (~0.2 dB/km) -- the C-band telecom window
    """
    wavelengths = np.linspace(800, 1700, 500)  # nm

    # Simplified attenuation model
    # Rayleigh scattering: alpha_R = A / lambda^4
    A_rayleigh = 0.7  # dB/km at 1 um (typical for silica)
    alpha_rayleigh = A_rayleigh * (1000 / wavelengths)**4

    # IR absorption (exponential tail)
    alpha_ir = 0.01 * np.exp((wavelengths - 1000) / 400)

    # OH absorption peak (simplified Lorentzian at 1383 nm)
    alpha_oh = 0.5 * (20**2) / ((wavelengths - 1383)**2 + 20**2)

    # Total attenuation
    alpha_total = alpha_rayleigh + alpha_ir + alpha_oh

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Attenuation spectrum ---
    ax = axes[0]
    ax.plot(wavelengths, alpha_total, 'k-', linewidth=2, label='Total')
    ax.plot(wavelengths, alpha_rayleigh, 'b--', linewidth=1, alpha=0.5, label='Rayleigh')
    ax.plot(wavelengths, alpha_ir, 'r--', linewidth=1, alpha=0.5, label='IR absorption')
    ax.plot(wavelengths, alpha_oh, 'g--', linewidth=1, alpha=0.5, label='OH peak')

    # Mark telecom windows
    windows = {'O': (1260, 1360), 'S': (1460, 1530), 'C': (1530, 1565), 'L': (1565, 1625)}
    colors_w = {'O': 'blue', 'S': 'green', 'C': 'red', 'L': 'orange'}
    for name, (w1, w2) in windows.items():
        ax.axvspan(w1, w2, alpha=0.1, color=colors_w[name])
        ax.text((w1 + w2) / 2, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 3,
                name, ha='center', fontsize=10, fontweight='bold', color=colors_w[name])

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Attenuation (dB/km)')
    ax.set_title('Silica Fiber Attenuation Spectrum')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # --- Panel 2: Link budget ---
    ax = axes[1]

    # Typical link parameters
    fiber_loss = 0.22  # dB/km at 1550 nm
    splice_loss = 0.05  # dB per splice
    connector_loss = 0.3  # dB per connector
    tx_power_dBm = 0  # transmitter power (1 mW)
    rx_sensitivity_dBm = -28  # receiver sensitivity
    margin_dB = 3  # safety margin

    n_connectors = 2
    splice_per_km = 1  # one splice every km (approx.)

    distances_km = np.linspace(0, 120, 500)

    # Power at receiver
    total_loss = (fiber_loss * distances_km +
                  connector_loss * n_connectors +
                  splice_loss * splice_per_km * distances_km)
    rx_power = tx_power_dBm - total_loss

    ax.plot(distances_km, rx_power, 'b-', linewidth=2, label='Received power')
    ax.axhline(rx_sensitivity_dBm, color='red', linestyle='--',
               label=f'Rx sensitivity ({rx_sensitivity_dBm} dBm)')
    ax.axhline(rx_sensitivity_dBm + margin_dB, color='orange', linestyle='--',
               label=f'With {margin_dB} dB margin')
    ax.fill_between(distances_km, rx_sensitivity_dBm, rx_power,
                     where=rx_power >= rx_sensitivity_dBm,
                     alpha=0.1, color='green')

    # Find maximum reach
    max_budget = tx_power_dBm - rx_sensitivity_dBm - margin_dB
    max_distance = max_budget / (fiber_loss + splice_loss * splice_per_km)
    ax.axvline(max_distance, color='green', linestyle=':', alpha=0.5)
    ax.text(max_distance + 1, tx_power_dBm - 5, f'Max: {max_distance:.0f} km',
            fontsize=9, color='green')

    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Power (dBm)')
    ax.set_title('Fiber Link Budget (1550 nm)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    print(f"\n  Link budget:")
    print(f"  Tx power: {tx_power_dBm} dBm")
    print(f"  Fiber loss: {fiber_loss} dB/km")
    print(f"  Total budget: {max_budget:.1f} dB")
    print(f"  Maximum reach: {max_distance:.0f} km")

    fig.tight_layout()
    fig.savefig('07_fiber_attenuation.png', dpi=150)
    plt.close(fig)
    print("[Saved] 07_fiber_attenuation.png")


# ---------------------------------------------------------------------------
# Visualization 4: Pulse broadening simulation
# ---------------------------------------------------------------------------

def plot_pulse_broadening():
    """Simulate Gaussian pulse propagation through a dispersive fiber.

    A transform-limited Gaussian pulse broadens as it propagates due to
    group velocity dispersion (GVD). The pulse width evolves as:
        tau(z) = tau_0 * sqrt(1 + (z/L_D)^2)

    where L_D = tau_0^2 / |beta_2| is the dispersion length, and
    beta_2 is the group velocity dispersion parameter.

    The pulse also acquires a frequency chirp (linear phase across the pulse),
    meaning different frequency components arrive at different times.
    """
    # Fiber parameters
    D = 17  # ps/(nm*km) at 1550 nm (standard SMF)
    wavelength = 1550e-9  # meters
    c = 3e8

    # Convert D to beta_2 (GVD parameter in s^2/m)
    # beta_2 = -D * lambda^2 / (2*pi*c)
    beta_2 = -D * 1e-6 * wavelength**2 / (2 * np.pi * c)  # s^2/m
    print(f"\n  GVD parameter beta_2 = {beta_2*1e24:.3f} ps^2/km")

    # Input pulse: Gaussian with FWHM = 10 ps
    fwhm_ps = 10
    tau_0 = fwhm_ps / (2 * np.sqrt(np.log(2))) * 1e-12  # 1/e half-width in seconds

    # Dispersion length
    L_D = tau_0**2 / abs(beta_2)
    print(f"  Input pulse FWHM = {fwhm_ps} ps")
    print(f"  Dispersion length L_D = {L_D*1e-3:.2f} km")

    # Time axis
    t = np.linspace(-50e-12, 50e-12, 1000)  # seconds
    t_ps = t * 1e12  # for plotting

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Pulse shape at different distances ---
    distances_km = [0, 1, 5, 10, 20]

    for L_km in distances_km:
        L = L_km * 1e3  # meters

        # Analytical solution for Gaussian pulse in dispersive medium
        # (frequency-domain propagation with exp(i*beta_2*omega^2*z/2))
        # Result: still Gaussian but with broadened width and added chirp
        tau_z = tau_0 * np.sqrt(1 + (L / L_D)**2)
        fwhm_z = tau_z * 2 * np.sqrt(np.log(2)) * 1e12  # ps

        # Intensity profile (normalize peak to 1)
        I = np.exp(-t**2 / tau_z**2)

        ax1.plot(t_ps, I, label=f'z = {L_km} km (FWHM={fwhm_z:.1f} ps)', linewidth=1.5)

    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Normalized intensity')
    ax1.set_title('Pulse Broadening Due to GVD')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Pulse width vs distance ---
    z_range = np.linspace(0, 50, 200)  # km
    z_m = z_range * 1e3  # meters
    tau_z_array = tau_0 * np.sqrt(1 + (z_m / L_D)**2)
    fwhm_array = tau_z_array * 2 * np.sqrt(np.log(2)) * 1e12  # ps

    ax2.plot(z_range, fwhm_array, 'b-', linewidth=2)
    ax2.axhline(fwhm_ps, color='gray', linestyle=':', alpha=0.5, label='Input FWHM')
    ax2.axvline(L_D * 1e-3, color='red', linestyle='--',
                label=f'L_D = {L_D*1e-3:.1f} km')

    # Asymptotic linear broadening at z >> L_D
    fwhm_asymp = fwhm_ps * z_m / L_D
    ax2.plot(z_range, fwhm_asymp * 1e12 * 2 * np.sqrt(np.log(2)),
             'g--', linewidth=1, alpha=0.5, label='Linear asymptote')

    ax2.set_xlabel('Propagation distance (km)')
    ax2.set_ylabel('FWHM pulse width (ps)')
    ax2.set_title('Pulse Width vs Propagation Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bit rate limit: B * delta_t < 1 -> B < 1/delta_t
    for L_km in [1, 10, 50]:
        L = L_km * 1e3
        tau_z = tau_0 * np.sqrt(1 + (L / L_D)**2)
        fwhm_z = tau_z * 2 * np.sqrt(np.log(2))
        max_bitrate = 0.44 / fwhm_z  # Hz
        print(f"  At {L_km} km: FWHM = {fwhm_z*1e12:.1f} ps, "
              f"max bitrate ~ {max_bitrate*1e-9:.1f} Gbps")

    fig.tight_layout()
    fig.savefig('07_fiber_pulse_broadening.png', dpi=150)
    plt.close(fig)
    print("[Saved] 07_fiber_pulse_broadening.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Fiber Optics: Modes, Dispersion, Attenuation, and Pulses")
    print("=" * 60)

    print("\n--- Fiber Parameters ---")
    plot_fiber_parameters()

    print("\n--- Chromatic Dispersion ---")
    plot_chromatic_dispersion()

    print("\n--- Attenuation and Link Budget ---")
    plot_attenuation_and_link()

    print("\n--- Pulse Broadening ---")
    plot_pulse_broadening()

    print("\nDone. All plots saved.")
