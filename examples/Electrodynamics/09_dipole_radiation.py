"""
Dipole Radiation: Oscillating Dipole, Far Field, and Antenna Arrays
===================================================================

Topics covered:
  1. Oscillating electric dipole radiation pattern
  2. Far-field E and B computation
  3. 3D radiation pattern (polar plot)
  4. Antenna array factor (2-element and N-element)

Why dipole radiation is foundational:
  The oscillating electric dipole is the simplest radiating system.
  Every antenna, every atom emitting light, and every radio transmitter
  can be decomposed (in a multipole expansion) into dipole, quadrupole,
  and higher-order terms. The dipole term dominates in most cases,
  so understanding it is essential for both engineering and physics.

Physics background:
  - Oscillating dipole: p(t) = p0 * cos(omega*t) * z_hat
  - Far-field (r >> lambda):
      E_theta = -(mu0 * p0 * omega^2 / 4*pi) * sin(theta) * cos(omega*t - kr) / r
      B_phi = E_theta / c
  - Radiation pattern: P(theta) ~ sin^2(theta)  (donut shape)
  - Total radiated power: P = (mu0 * p0^2 * omega^4) / (12 * pi * c)
  - Antenna array: multiply element pattern by array factor AF
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
C = 2.998e8
MU0 = 4 * np.pi * 1e-7
EPS0 = 8.854e-12


# ===========================
# 1. Dipole Radiation Pattern (2D Polar Plot)
# ===========================

def dipole_radiation_2d():
    """
    Plot the radiation pattern of an oscillating electric dipole.

    The pattern is P(theta) ~ sin^2(theta), independent of phi (azimuthal symmetry).

    Why sin^2(theta)?
      The oscillating dipole accelerates charges along the z-axis.
      Accelerated charges radiate perpendicular to their acceleration.
      - Along z (theta = 0, pi): no radiation (looking end-on at the dipole)
      - In the x-y plane (theta = pi/2): maximum radiation (broadside)
      The sin(theta) in E_theta gives sin^2(theta) in power.
    """
    theta = np.linspace(0, 2 * np.pi, 500)

    # Radiation pattern (power): P ~ sin^2(theta)
    P = np.sin(theta)**2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Polar plot
    ax = axes[0]
    ax = fig.add_subplot(121, projection='polar')
    ax.plot(theta, P, 'b-', linewidth=2)
    ax.set_title(r'Dipole Radiation: $P(\theta) \sim \sin^2\theta$', pad=20)
    ax.set_rlabel_position(0)

    # Cartesian plot for clarity
    ax = axes[1]
    theta_deg = np.linspace(0, 180, 200)
    P_half = np.sin(np.radians(theta_deg))**2
    ax.plot(theta_deg, P_half, 'b-', linewidth=2)
    ax.fill_between(theta_deg, 0, P_half, alpha=0.2, color='blue')
    ax.set_xlabel(r'$\theta$ (degrees)')
    ax.set_ylabel(r'$P(\theta) / P_{max}$')
    ax.set_title('Radiation Pattern (linear scale)')
    ax.grid(True, alpha=0.3)

    # Mark important angles
    ax.axvline(x=0, color='red', linestyle=':', alpha=0.5, label=r'$\theta$ = 0 (null)')
    ax.axvline(x=90, color='green', linestyle=':', alpha=0.5, label=r'$\theta$ = 90 (max)')
    ax.axvline(x=180, color='red', linestyle=':', alpha=0.5, label=r'$\theta$ = 180 (null)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('09_dipole_2d.png', dpi=150)
    plt.close()
    print("[Saved] 09_dipole_2d.png")


# ===========================
# 2. Far-Field E and B Computation
# ===========================

def far_field_computation():
    """
    Compute the far-field E and B for an oscillating dipole at various
    distances, verifying the 1/r dependence and E/B = c relationship.

    Why 1/r and not 1/r^2?
      In the near field, there are static-like terms that fall off as
      1/r^2 and 1/r^3. But in the far field (radiation zone, r >> lambda),
      only the 1/r term survives. This is the radiation field -- it carries
      energy to infinity. Energy flux S ~ E*B ~ 1/r^2, and since the
      sphere area grows as r^2, the total power P = S * 4*pi*r^2 is
      constant -- energy is conserved as the wave spreads.
    """
    # Dipole parameters
    p0 = 1e-10       # C*m (dipole moment amplitude)
    f = 1e9           # 1 GHz
    omega = 2 * np.pi * f
    k = omega / C
    lam = C / f

    print(f"  Frequency: {f/1e9:.1f} GHz, Wavelength: {lam*100:.1f} cm")

    # Radial distances (in units of wavelength)
    r_over_lam = np.linspace(1, 20, 500)
    r = r_over_lam * lam
    theta = np.pi / 2  # Maximum radiation direction

    # Far-field E_theta at theta = pi/2 (peak), t = 0
    # E_theta = -(mu0 * p0 * omega^2 * sin(theta)) / (4*pi*r)
    # (ignoring the oscillating phase factor for amplitude)
    E_amplitude = MU0 * p0 * omega**2 * np.sin(theta) / (4 * np.pi * r)
    B_amplitude = E_amplitude / C

    # Verify 1/r dependence
    E_times_r = E_amplitude * r  # should be constant

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(r_over_lam, E_amplitude * 1e6, 'b-', linewidth=2)
    ax.set_xlabel(r'$r / \lambda$')
    ax.set_ylabel(r'$|E_\theta|$ ($\mu$V/m)')
    ax.set_title(r'Far-field $E_\theta$ amplitude')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(r_over_lam, B_amplitude * 1e12, 'r-', linewidth=2)
    ax.set_xlabel(r'$r / \lambda$')
    ax.set_ylabel(r'$|B_\phi|$ (pT)')
    ax.set_title(r'Far-field $B_\phi$ amplitude')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(r_over_lam, E_times_r * 1e3, 'g-', linewidth=2)
    ax.set_xlabel(r'$r / \lambda$')
    ax.set_ylabel(r'$r \cdot |E_\theta|$ (mV)')
    ax.set_title(r'$r \cdot E$ = constant (1/r verified)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('09_far_field.png', dpi=150)
    plt.close()
    print("[Saved] 09_far_field.png")

    # Total radiated power
    # P = (mu0 * p0^2 * omega^4) / (12 * pi * c)
    P_total = MU0 * p0**2 * omega**4 / (12 * np.pi * C)
    print(f"  Total radiated power: P = {P_total:.4e} W")


# ===========================
# 3. 3D Radiation Pattern
# ===========================

def radiation_3d():
    """
    Visualize the 3D dipole radiation pattern (donut/toroid shape).

    Why 3D visualization?
      The 2D polar plot shows a cross-section, but the full pattern is
      a toroid (donut) obtained by rotating sin^2(theta) around the
      z-axis. The 3D view reveals the complete null along z and the
      ring of maximum radiation in the x-y plane.
    """
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi)

    # Radiation pattern as a function of (theta, phi)
    # For a z-directed dipole: P(theta, phi) = sin^2(theta)
    R = np.sin(THETA)**2

    # Convert to Cartesian for 3D plot
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Why use facecolors based on R?
    #   This maps the radiation intensity to color, making the
    #   pattern quantitatively readable (hot = strong radiation).
    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.hot(R / R.max()),
                           alpha=0.8, rstride=2, cstride=2)

    # Draw dipole axis
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', linewidth=2, label='Dipole axis (z)')
    ax.plot([0], [0], [0], 'ko', markersize=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(r'3D Dipole Radiation Pattern: $P \sim \sin^2\theta$')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.tight_layout()
    plt.savefig('09_radiation_3d.png', dpi=150)
    plt.close()
    print("[Saved] 09_radiation_3d.png")


# ===========================
# 4. Antenna Array Factor
# ===========================

def array_factor_2element():
    """
    Compute and plot the array factor for a 2-element dipole array.

    Two identical dipoles separated by distance d along the x-axis,
    observed in the x-z plane (phi = 0).

    Array factor: AF = 2 * cos(k*d*sin(theta)*cos(phi)/2 + delta/2)
    where delta is the phase difference between elements.

    Why array factor?
      Real antennas often use multiple elements to shape the radiation
      pattern. The total pattern is the product of the single-element
      pattern and the array factor:
        P_total(theta, phi) = P_element(theta, phi) * |AF(theta, phi)|^2
      This factorization is valid when all elements are identical and
      have the same orientation (pattern multiplication theorem).
    """
    theta = np.linspace(-np.pi, np.pi, 1000)
    f = 1e9
    lam = C / f
    k = 2 * np.pi / lam

    # Different spacings and phase shifts
    configs = [
        (lam / 2, 0, r'd=$\lambda$/2, $\delta$=0 (broadside)'),
        (lam / 2, np.pi, r'd=$\lambda$/2, $\delta$=$\pi$ (endfire)'),
        (lam / 4, np.pi / 2, r'd=$\lambda$/4, $\delta$=$\pi$/2 (cardioid)'),
        (lam, 0, r'd=$\lambda$, $\delta$=0 (grating lobes!)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': 'polar'})
    axes = axes.flatten()

    for ax, (d, delta, label) in zip(axes, configs):
        # Array factor for 2-element array
        # Why k*d*cos(theta)/2?
        #   For elements along x-axis and observation in x-z plane,
        #   the path difference is d*sin(theta_from_broadside) = d*cos(theta_from_axis).
        #   With theta measured from z-axis: AF = cos(k*d*sin(theta)/2 + delta/2).
        psi = k * d * np.sin(theta) + delta
        AF = np.cos(psi / 2)

        # Element pattern: sin^2(theta) for z-directed dipoles
        P_element = np.sin(theta)**2
        P_total = P_element * AF**2

        # Normalize
        P_total_norm = P_total / (np.max(P_total) + 1e-20)

        ax.plot(theta, P_total_norm, 'b-', linewidth=2)
        ax.plot(theta, np.abs(AF**2) / (np.max(np.abs(AF**2)) + 1e-20),
                'r--', linewidth=1, alpha=0.5, label='AF only')
        ax.set_title(label, fontsize=10, pad=15)
        ax.set_rlabel_position(0)
        ax.legend(loc='lower right', fontsize=8)

    plt.suptitle('2-Element Antenna Array Patterns', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('09_array_2element.png', dpi=150)
    plt.close()
    print("[Saved] 09_array_2element.png")


def array_factor_N_element():
    """
    N-element uniform linear array with varying N.

    Array factor: AF_N = sin(N*psi/2) / sin(psi/2)
    where psi = k*d*sin(theta) + delta

    Why does increasing N sharpen the beam?
      More elements means more constructive/destructive interference
      conditions. The main beam gets narrower (better directivity),
      but side lobes appear. The half-power beamwidth scales as ~1/N
      for large N.

    This is the fundamental principle behind phased array radar,
    5G beamforming, and radio telescope interferometry.
    """
    theta = np.linspace(-np.pi / 2, np.pi / 2, 1000)
    f = 1e9
    lam = C / f
    k = 2 * np.pi / lam
    d = lam / 2  # half-wavelength spacing
    delta = 0    # broadside array

    N_values = [2, 4, 8, 16]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, N in zip(axes, N_values):
        psi = k * d * np.sin(theta) + delta

        # Array factor magnitude (handle 0/0 at psi = 0)
        # Why use L'Hopital limit?
        #   At psi = 0, both numerator and denominator are zero.
        #   The limit sin(N*x)/sin(x) -> N as x -> 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            AF = np.sin(N * psi / 2) / np.sin(psi / 2)
            AF = np.where(np.abs(np.sin(psi / 2)) < 1e-10, N, AF)

        AF_norm = np.abs(AF) / N  # Normalize to max = 1
        AF_dB = 20 * np.log10(AF_norm + 1e-10)

        ax.plot(np.degrees(theta), AF_dB, 'b-', linewidth=2)
        ax.set_xlabel(r'$\theta$ (degrees from broadside)')
        ax.set_ylabel('Array Factor (dB)')
        ax.set_title(f'N = {N} elements, d = lambda/2')
        ax.set_ylim(-30, 5)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=-3, color='red', linestyle='--', alpha=0.5, label='-3 dB')

        # Estimate half-power beamwidth
        above_3dB = np.where(AF_dB >= -3)[0]
        if len(above_3dB) > 0:
            bw = np.degrees(theta[above_3dB[-1]] - theta[above_3dB[0]])
            ax.set_title(f'N = {N}, HPBW ~ {bw:.1f} deg')
        ax.legend(fontsize=9)

    plt.suptitle('N-Element Uniform Array: Beam Narrowing', fontsize=14)
    plt.tight_layout()
    plt.savefig('09_array_N_element.png', dpi=150)
    plt.close()
    print("[Saved] 09_array_N_element.png")


# ===========================
# Main
# ===========================

if __name__ == '__main__':
    print("=== Dipole Radiation Pattern (2D) ===")
    dipole_radiation_2d()

    print("\n=== Far-Field E and B ===")
    far_field_computation()

    print("\n=== 3D Radiation Pattern ===")
    radiation_3d()

    print("\n=== 2-Element Array ===")
    array_factor_2element()

    print("\n=== N-Element Array ===")
    array_factor_N_element()
