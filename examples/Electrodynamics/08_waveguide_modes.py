"""
Rectangular Waveguide: TE and TM Modes, Cutoff, and Dispersion
==============================================================

Topics covered:
  1. TE_mn and TM_mn mode field patterns for a rectangular waveguide
  2. Cutoff frequency computation for each mode
  3. Dispersion diagram (omega vs k_z)
  4. Mode field visualization (2D contour plots)

Why study waveguides?
  Waveguides are hollow metal tubes that guide EM waves without radiation
  loss. Unlike free-space propagation, waveguides support only discrete
  modes, each with a minimum (cutoff) frequency below which it cannot
  propagate. This is the foundation of microwave engineering, radar,
  and particle accelerators.

Physics background:
  - For a rectangular waveguide of width a (x) and height b (y):
  - TE_mn: E_z = 0, H_z = H0 * cos(m*pi*x/a) * cos(n*pi*y/b) * exp(i*k_z*z)
  - TM_mn: H_z = 0, E_z = E0 * sin(m*pi*x/a) * sin(n*pi*y/b) * exp(i*k_z*z)
  - Cutoff frequency: f_c(m,n) = c/(2*pi) * sqrt((m*pi/a)^2 + (n*pi/b)^2)
                                = (c/2) * sqrt((m/a)^2 + (n/b)^2)
  - Dispersion: k_z = sqrt(omega^2/c^2 - k_c^2)  where k_c = sqrt((m*pi/a)^2 + (n*pi/b)^2)
  - Dominant mode: TE_10 (lowest cutoff for a > b)
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
C = 2.998e8  # m/s


# ===========================
# 1. Cutoff Frequencies
# ===========================

def cutoff_frequency(m, n, a, b):
    """
    Compute the cutoff frequency for mode TE_mn or TM_mn.

    f_c = (c/2) * sqrt((m/a)^2 + (n/b)^2)

    Why does a cutoff frequency exist?
      In a waveguide, the wave must "fit" transversely between the walls.
      This constrains the transverse wavenumber: k_x = m*pi/a, k_y = n*pi/b.
      The total wavenumber must satisfy k^2 = k_x^2 + k_y^2 + k_z^2 = omega^2/c^2.
      For propagation, k_z must be real, requiring omega > omega_c.
      Below cutoff, k_z is imaginary and the wave decays exponentially (evanescent).
    """
    return (C / 2.0) * np.sqrt((m / a)**2 + (n / b)**2)


def print_cutoff_table(a, b, max_modes=8):
    """Print cutoff frequencies for the lowest modes."""
    modes = []
    for m in range(5):
        for n in range(5):
            if m == 0 and n == 0:
                continue  # TE_00 doesn't exist (trivial), TM_00 doesn't exist
            fc = cutoff_frequency(m, n, a, b)
            # Why distinguish TE and TM?
            #   TM modes require m >= 1 AND n >= 1 (sin*sin pattern).
            #   TE modes allow m=0 or n=0 (but not both).
            mode_types = []
            if not (m == 0 and n == 0):
                mode_types.append(f'TE_{m}{n}')
            if m >= 1 and n >= 1:
                mode_types.append(f'TM_{m}{n}')
            for mt in mode_types:
                modes.append((fc, mt))

    modes.sort(key=lambda x: x[0])

    print(f"\n  Waveguide: a = {a*100:.1f} cm, b = {b*100:.1f} cm")
    print(f"  {'Mode':<10} {'f_c (GHz)':>12}")
    print(f"  {'-'*24}")
    for fc, name in modes[:max_modes]:
        print(f"  {name:<10} {fc/1e9:>12.3f}")

    return modes[:max_modes]


# ===========================
# 2. Mode Field Patterns
# ===========================

def te_mode_fields(m, n, a, b, Nx=100, Ny=100):
    """
    Compute the transverse field pattern for TE_mn mode.

    For TE modes, E_z = 0 and the transverse fields are derived from H_z:
      H_z = H0 * cos(m*pi*x/a) * cos(n*pi*y/b)
      E_x = (j*omega*mu0 / k_c^2) * (n*pi/b) * H0 * cos(m*pi*x/a) * (-sin(n*pi*y/b))
      E_y = -(j*omega*mu0 / k_c^2) * (m*pi/a) * (-sin(m*pi*x/a)) * cos(n*pi*y/b)

    Why cosines for H_z in TE modes?
      The boundary condition on a perfect conductor is that the tangential
      E field vanishes at the wall. For TE modes, this translates to
      dH_z/dn = 0 at the walls (Neumann BC), which cosines satisfy naturally.

    For visualization, we compute normalized patterns (set H0=1, drop constants).
    """
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    X, Y = np.meshgrid(x, y)

    kx = m * np.pi / a
    ky = n * np.pi / b
    kc2 = kx**2 + ky**2

    # H_z pattern
    Hz = np.cos(kx * X) * np.cos(ky * Y)

    # Transverse E fields (proportional, ignoring prefactors)
    # Ex ~ -ky * cos(kx*x) * sin(ky*y)
    # Ey ~  kx * sin(kx*x) * cos(ky*y)
    if kc2 > 0:
        Ex = -(ky / kc2) * np.cos(kx * X) * np.sin(ky * Y)
        Ey = (kx / kc2) * np.sin(kx * X) * np.cos(ky * Y)
    else:
        Ex = np.zeros_like(X)
        Ey = np.zeros_like(Y)

    return X, Y, Ex, Ey, Hz


def tm_mode_fields(m, n, a, b, Nx=100, Ny=100):
    """
    Compute the transverse field pattern for TM_mn mode.

    For TM modes, H_z = 0 and the transverse fields are derived from E_z:
      E_z = E0 * sin(m*pi*x/a) * sin(n*pi*y/b)
      E_x = -(j*k_z / k_c^2) * (m*pi/a) * cos(m*pi*x/a) * sin(n*pi*y/b)
      E_y = -(j*k_z / k_c^2) * (n*pi/b) * sin(m*pi*x/a) * cos(n*pi*y/b)

    Why sines for E_z in TM modes?
      E_z is tangential to the walls, so it must vanish there (Dirichlet BC).
      sin(m*pi*x/a) = 0 at x=0 and x=a -- exactly what we need.
      This is why TM modes require m >= 1 AND n >= 1.
    """
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    X, Y = np.meshgrid(x, y)

    kx = m * np.pi / a
    ky = n * np.pi / b
    kc2 = kx**2 + ky**2

    # E_z pattern
    Ez = np.sin(kx * X) * np.sin(ky * Y)

    # Transverse E fields (proportional)
    # Ex ~ -kx * cos(kx*x) * sin(ky*y)
    # Ey ~ -ky * sin(kx*x) * cos(ky*y)
    if kc2 > 0:
        Ex = -(kx / kc2) * np.cos(kx * X) * np.sin(ky * Y)
        Ey = -(ky / kc2) * np.sin(kx * X) * np.cos(ky * Y)
    else:
        Ex = np.zeros_like(X)
        Ey = np.zeros_like(Y)

    return X, Y, Ex, Ey, Ez


# ===========================
# 3. Dispersion Diagram
# ===========================

def dispersion_diagram(a, b, modes_list):
    """
    Plot the dispersion relation omega(k_z) for several modes.

    omega^2 = c^2 * (k_z^2 + k_c^2)
    -->  omega = c * sqrt(k_z^2 + k_c^2)

    Why is the dispersion hyperbolic (not linear)?
      In free space, omega = c*k (linear -- all frequencies travel at c).
      In a waveguide, part of k is "used up" transversely (k_c), so
      the axial component k_z is smaller: k_z = sqrt(k^2 - k_c^2).
      This makes the dispersion relation hyperbolic, with important
      consequences:
        - Phase velocity v_p = omega/k_z > c  (superluminal but not physical)
        - Group velocity v_g = d(omega)/d(k_z) < c  (subluminal, carries energy)
        - v_p * v_g = c^2  (a beautiful relationship)
    """
    kz = np.linspace(0, 200, 500)  # rad/m

    fig, ax = plt.subplots(figsize=(10, 7))

    # Light line (free space)
    omega_light = C * kz
    ax.plot(kz, omega_light / (2 * np.pi * 1e9), 'k--', linewidth=1,
            alpha=0.5, label='Light line ($\\omega = ck$)')

    colors = plt.cm.tab10(np.linspace(0, 1, len(modes_list)))

    for (fc, name), color in zip(modes_list, colors):
        kc = 2 * np.pi * fc / C
        # omega = c * sqrt(kz^2 + kc^2)
        omega = C * np.sqrt(kz**2 + kc**2)
        f_GHz = omega / (2 * np.pi * 1e9)
        ax.plot(kz, f_GHz, linewidth=2, color=color, label=name)

        # Mark cutoff (kz = 0)
        ax.plot(0, fc / 1e9, 'o', color=color, markersize=6)

    ax.set_xlabel(r'$k_z$ (rad/m)')
    ax.set_ylabel('Frequency (GHz)')
    ax.set_title(f'Waveguide Dispersion Diagram (a={a*100:.1f} cm, b={b*100:.1f} cm)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, max(fc for fc, _ in modes_list) / 1e9 * 1.5)

    plt.tight_layout()
    plt.savefig('08_dispersion.png', dpi=150)
    plt.close()
    print("[Saved] 08_dispersion.png")


# ===========================
# 4. Mode Field Visualization
# ===========================

def plot_mode_fields(a, b):
    """
    Create 2D contour plots of field patterns for several waveguide modes.

    Why plot multiple modes side by side?
      Seeing TE_10, TE_20, TE_11, and TM_11 together reveals the
      pattern: higher mode indices mean more nodes (zero-crossings),
      shorter transverse wavelength, and higher cutoff frequency.
    """
    modes = [
        ('TE', 1, 0),
        ('TE', 2, 0),
        ('TE', 1, 1),
        ('TM', 1, 1),
        ('TE', 2, 1),
        ('TM', 2, 1),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (mode_type, m, n) in enumerate(modes):
        ax = axes[idx]

        if mode_type == 'TE':
            X, Y, Ex, Ey, pattern = te_mode_fields(m, n, a, b)
            pattern_name = r'$H_z$'
        else:
            X, Y, Ex, Ey, pattern = tm_mode_fields(m, n, a, b)
            pattern_name = r'$E_z$'

        fc = cutoff_frequency(m, n, a, b)

        # Why both contourf and quiver?
        #   contourf shows the longitudinal field component (Hz for TE, Ez for TM).
        #   quiver shows the transverse E field direction. Together they give
        #   the full picture of the mode structure.
        cf = ax.contourf(X * 100, Y * 100, pattern, levels=30, cmap='RdBu_r')
        fig.colorbar(cf, ax=ax, shrink=0.8)

        # Overlay transverse E field arrows
        # Why subsample?
        #   Plotting arrows at every grid point would be unreadable.
        #   Every 8th point gives a clear picture.
        skip = 8
        E_mag = np.sqrt(Ex**2 + Ey**2)
        E_mag_max = np.max(E_mag) + 1e-20
        ax.quiver(X[::skip, ::skip] * 100, Y[::skip, ::skip] * 100,
                  Ex[::skip, ::skip] / E_mag_max,
                  Ey[::skip, ::skip] / E_mag_max,
                  color='black', alpha=0.6, scale=20)

        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f'{mode_type}$_{{{m}{n}}}$ | {pattern_name} | '
                     f'$f_c$ = {fc/1e9:.2f} GHz')
        ax.set_aspect('equal')

    plt.suptitle(f'Waveguide Mode Patterns (a={a*100:.1f} cm, b={b*100:.1f} cm)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('08_mode_patterns.png', dpi=150)
    plt.close()
    print("[Saved] 08_mode_patterns.png")


def plot_dominant_mode_detail(a, b):
    """
    Detailed view of the dominant TE_10 mode.

    Why focus on TE_10?
      TE_10 has the lowest cutoff frequency (for a > b), making it the
      dominant mode. In practice, waveguides are operated in the
      frequency range between TE_10 cutoff and the next mode's cutoff.
      This "single-mode" operation ensures clean signal transmission.
    """
    X, Y, Ex, Ey, Hz = te_mode_fields(1, 0, a, b, Nx=100, Ny=100)
    E_mag = np.sqrt(Ex**2 + Ey**2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Hz pattern
    ax = axes[0]
    cf = ax.contourf(X * 100, Y * 100, Hz, levels=30, cmap='RdBu_r')
    fig.colorbar(cf, ax=ax, label=r'$H_z$ (normalized)')
    ax.set_title(r'$H_z$ pattern (TE$_{10}$)')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_aspect('equal')

    # |E| pattern
    ax = axes[1]
    cf = ax.contourf(X * 100, Y * 100, E_mag, levels=30, cmap='hot')
    fig.colorbar(cf, ax=ax, label=r'$|E_t|$ (normalized)')
    ax.set_title(r'$|E_{transverse}|$ (TE$_{10}$)')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_aspect('equal')

    # E field arrows
    ax = axes[2]
    skip = 5
    ax.quiver(X[::skip, ::skip] * 100, Y[::skip, ::skip] * 100,
              Ex[::skip, ::skip], Ey[::skip, ::skip],
              E_mag[::skip, ::skip], cmap='viridis')
    ax.set_title(r'$\vec{E}_{transverse}$ (TE$_{10}$)')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_aspect('equal')

    plt.suptitle(f'Dominant Mode TE_10 Detail (a={a*100:.1f} cm, b={b*100:.1f} cm)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('08_te10_detail.png', dpi=150)
    plt.close()
    print("[Saved] 08_te10_detail.png")


# ===========================
# Main
# ===========================

if __name__ == '__main__':
    # WR-90 standard waveguide (X-band, 8-12 GHz)
    # Why WR-90?
    #   It's one of the most common waveguide sizes, used extensively
    #   in radar and satellite communications. The dimensions are
    #   standardized: a = 22.86 mm, b = 10.16 mm.
    a = 0.02286  # m (width)
    b = 0.01016  # m (height)

    print("=== Cutoff Frequencies ===")
    modes = print_cutoff_table(a, b, max_modes=8)

    print("\n=== Dispersion Diagram ===")
    dispersion_diagram(a, b, modes)

    print("\n=== Mode Field Patterns ===")
    plot_mode_fields(a, b)

    print("\n=== Dominant Mode (TE_10) Detail ===")
    plot_dominant_mode_detail(a, b)

    # Single-mode bandwidth
    fc_10 = cutoff_frequency(1, 0, a, b)
    fc_20 = cutoff_frequency(2, 0, a, b)
    fc_01 = cutoff_frequency(0, 1, a, b)
    fc_next = min(fc_20, fc_01)
    print(f"\n  Single-mode bandwidth: {fc_10/1e9:.2f} - {fc_next/1e9:.2f} GHz")
    print(f"  Bandwidth ratio: {fc_next/fc_10:.2f}")
