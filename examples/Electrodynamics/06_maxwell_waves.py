"""
Maxwell's Equations and Electromagnetic Waves
==============================================

Topics covered:
  1. Plane wave propagation animation (snapshots)
  2. E and B field relationship (perpendicular, in-phase, E/B = c)
  3. Poynting vector computation (energy flow)
  4. Energy density visualization

Why this is the culmination of classical electrodynamics:
  Maxwell's genius was adding the displacement current term to Ampere's
  law, completing the set of four equations. This single addition allows
  electromagnetic waves to exist: oscillating E creates oscillating B,
  which creates oscillating E, propagating through empty space at
  c = 1/sqrt(mu0*eps0). Light, radio, X-rays -- all electromagnetic waves.

Physics background:
  - Maxwell's equations in vacuum (differential form):
      div E = 0,        div B = 0
      curl E = -dB/dt,  curl B = mu0*eps0 * dE/dt
  - Wave equation: nabla^2 E = mu0*eps0 * d^2E/dt^2
  - Plane wave: E = E0 * cos(kz - omega*t) * x_hat
                B = (E0/c) * cos(kz - omega*t) * y_hat
  - Phase velocity: c = omega/k = 1/sqrt(mu0*eps0) ~ 3e8 m/s
  - Poynting vector: S = (1/mu0) * E x B  (power per area, W/m^2)
  - Energy density: u = (1/2)*(eps0*E^2 + B^2/mu0)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C = 2.998e8          # m/s, speed of light
EPS0 = 8.854e-12     # F/m
MU0 = 4 * np.pi * 1e-7  # T*m/A

# Verify: c = 1/sqrt(mu0*eps0)
c_computed = 1.0 / np.sqrt(MU0 * EPS0)
print(f"Speed of light: c = 1/sqrt(mu0*eps0) = {c_computed:.4e} m/s")
print(f"Known value:    c = {C:.4e} m/s")
print(f"Relative error: {abs(c_computed - C)/C:.2e}")


# ===========================
# 1. Plane Wave Propagation (Snapshots)
# ===========================

def plane_wave_snapshots():
    """
    Visualize a plane electromagnetic wave at several time instants.

    Why snapshots instead of animation?
      Animations require a GUI or video export. Snapshots on a single
      figure let us see the wave at multiple times simultaneously,
      making it easy to verify that the pattern moves to the right
      at speed c.
    """
    # Wave parameters
    f = 1e9               # 1 GHz (microwave)
    omega = 2 * np.pi * f
    k = omega / C          # wavenumber
    lam = 2 * np.pi / k   # wavelength
    E0 = 1.0              # V/m (amplitude)

    print(f"\nPlane wave: f = {f/1e9:.1f} GHz, lambda = {lam*100:.2f} cm")

    z = np.linspace(0, 5 * lam, 1000)
    times = [0, 0.25 / f, 0.5 / f, 0.75 / f]  # 0, T/4, T/2, 3T/4

    fig, axes = plt.subplots(len(times), 1, figsize=(12, 10), sharex=True)

    for ax, t in zip(axes, times):
        # Why E in x-direction and B in y-direction?
        #   For a wave propagating in z, E and B must be transverse
        #   (perpendicular to z). The choice of x for E and y for B
        #   is conventional. E x B = S must point in the +z direction
        #   (direction of propagation), which x_hat x y_hat = z_hat does.
        Ex = E0 * np.cos(k * z - omega * t)
        By = (E0 / C) * np.cos(k * z - omega * t)

        ax.plot(z / lam, Ex, 'b-', linewidth=2, label=r'$E_x$ (V/m)')
        ax.plot(z / lam, By * C, 'r--', linewidth=2, label=r'$cB_y$ (V/m)')
        ax.set_ylabel('Field')
        ax.set_title(f't = {t*f:.2f} T')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.3, 1.3)

    axes[-1].set_xlabel(r'z / $\lambda$')
    plt.suptitle('Plane EM Wave Propagation', fontsize=14)
    plt.tight_layout()
    plt.savefig('06_plane_wave_snapshots.png', dpi=150)
    plt.close()
    print("[Saved] 06_plane_wave_snapshots.png")


# ===========================
# 2. E and B Field Relationship
# ===========================

def eb_relationship():
    """
    Demonstrate the key relationships between E and B in a plane wave:
      1. E and B are perpendicular to each other and to propagation direction
      2. E and B are in phase (peaks coincide)
      3. |E| / |B| = c  at all times

    Why is E/B = c so important?
      This ratio is fixed by Maxwell's equations and equals the speed of
      light. It means the electric and magnetic energies are equal:
        (1/2)*eps0*E^2 = (1/2)*B^2/mu0  -->  E/B = 1/sqrt(mu0*eps0) = c
      This is not a coincidence but a fundamental consequence of the
      wave equation derived from Maxwell's equations.
    """
    E0 = 100.0  # V/m
    f = 1e6      # 1 MHz
    omega = 2 * np.pi * f
    k = omega / C

    z = np.linspace(0, 3 * 2 * np.pi / k, 500)
    t = 0

    Ex = E0 * np.cos(k * z - omega * t)
    By = (E0 / C) * np.cos(k * z - omega * t)

    # Verify E/B = c at every point
    ratio = Ex / (By + 1e-30)  # avoid division by zero at nodes

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    ax = axes[0]
    ax.plot(z * k / (2 * np.pi), Ex, 'b-', linewidth=2)
    ax.set_ylabel(r'$E_x$ (V/m)')
    ax.set_title('Electric Field')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(z * k / (2 * np.pi), By * 1e6, 'r-', linewidth=2)
    ax.set_ylabel(r'$B_y$ ($\mu$T)')
    ax.set_title('Magnetic Field')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # Why mask near nodes?
    #   At nodes (where E=B=0), the ratio E/B is 0/0, which is
    #   numerically unstable. We mask these points.
    mask = np.abs(By) > 1e-10
    ax.plot(z[mask] * k / (2 * np.pi), ratio[mask] / 1e8, 'g-', linewidth=2)
    ax.axhline(y=C / 1e8, color='black', linestyle='--',
               label=f'c = {C:.3e} m/s')
    ax.set_xlabel(r'$z / \lambda$')
    ax.set_ylabel(r'$E/B$ ($\times 10^8$ m/s)')
    ax.set_title(r'Ratio $|E|/|B|$ = c everywhere')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('06_eb_relationship.png', dpi=150)
    plt.close()
    print("[Saved] 06_eb_relationship.png")


# ===========================
# 3. Poynting Vector (Energy Flow)
# ===========================

def poynting_vector():
    """
    Compute and visualize the Poynting vector S = (1/mu0) * E x B.

    S represents the power per unit area (W/m^2) carried by the wave.
    For a plane wave: S = (E0^2 / (2*mu0)) * z_hat  (time-averaged)

    Why time-average?
      The instantaneous Poynting vector oscillates at 2*omega (double the
      wave frequency). Detectors (eyes, antennas, solar cells) respond
      to the time-averaged intensity, not the instantaneous value.
      <S> = (1/2) * (E0^2 / mu0*c) = (1/2) * eps0 * c * E0^2
    """
    E0 = 100.0  # V/m
    f = 1e9
    omega = 2 * np.pi * f
    k = omega / C
    lam = C / f

    z = np.linspace(0, 3 * lam, 500)
    t_values = np.linspace(0, 1 / f, 200)

    # Time-averaged Poynting vector (analytic)
    S_avg = E0**2 / (2 * MU0 * C)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Instantaneous S at several times
    ax = axes[0]
    for t in [0, 0.125 / f, 0.25 / f, 0.375 / f]:
        Ex = E0 * np.cos(k * z - omega * t)
        By = (E0 / C) * np.cos(k * z - omega * t)
        # S_z = (1/mu0) * Ex * By  (for x-polarized E, y-polarized B)
        Sz = (1.0 / MU0) * Ex * By
        ax.plot(z / lam, Sz, linewidth=1.5, alpha=0.7,
                label=f't = {t*f:.3f} T')

    ax.axhline(y=S_avg, color='black', linestyle='--', linewidth=2,
               label=f'<S> = {S_avg:.2f} W/m^2')
    ax.set_xlabel(r'z / $\lambda$')
    ax.set_ylabel(r'$S_z$ (W/m$^2$)')
    ax.set_title('Instantaneous Poynting Vector')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Why does S oscillate at 2*omega?
    #   S = E*B/mu0 = (E0^2/mu0*c) * cos^2(kz - wt)
    #   Using cos^2(x) = [1 + cos(2x)]/2, we see the oscillation
    #   is at frequency 2f, with a positive DC offset (no negative S).
    #   Negative S would mean energy flowing backward -- impossible
    #   for a plane wave propagating in one direction.

    # Time-averaged at each z
    ax = axes[1]
    S_at_z = np.zeros_like(z)
    for t in t_values:
        Ex = E0 * np.cos(k * z - omega * t)
        By = (E0 / C) * np.cos(k * z - omega * t)
        S_at_z += (1.0 / MU0) * Ex * By
    S_at_z /= len(t_values)

    ax.plot(z / lam, S_at_z, 'b-', linewidth=2, label='Numerical time average')
    ax.axhline(y=S_avg, color='red', linestyle='--', linewidth=2,
               label=f'Analytic <S> = {S_avg:.2f} W/m^2')
    ax.set_xlabel(r'z / $\lambda$')
    ax.set_ylabel(r'$\langle S_z \rangle$ (W/m$^2$)')
    ax.set_title('Time-Averaged Poynting Vector (uniform, as expected)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('06_poynting_vector.png', dpi=150)
    plt.close()
    print("[Saved] 06_poynting_vector.png")
    print(f"  Time-averaged intensity: <S> = {S_avg:.4f} W/m^2")


# ===========================
# 4. Energy Density Visualization
# ===========================

def energy_density():
    """
    Visualize electric and magnetic energy densities in a plane wave.

    u_E = (1/2) * eps0 * E^2  (electric energy density)
    u_B = (1/2) * B^2 / mu0   (magnetic energy density)
    u_total = u_E + u_B

    Why are u_E and u_B equal?
      For a plane wave, E = c*B. Substituting:
        u_E = (1/2)*eps0*E^2 = (1/2)*eps0*(cB)^2 = (1/2)*eps0*c^2*B^2
            = (1/2)*B^2/mu0 = u_B    (using c^2 = 1/(mu0*eps0))
      Electric and magnetic fields carry equal energy. This equipartition
      is specific to EM waves (in a capacitor, u_E dominates; in an
      inductor, u_B dominates).
    """
    E0 = 100.0
    f = 1e9
    omega = 2 * np.pi * f
    k = omega / C
    lam = C / f

    z = np.linspace(0, 3 * lam, 500)
    t = 0

    Ex = E0 * np.cos(k * z - omega * t)
    By = (E0 / C) * np.cos(k * z - omega * t)

    u_E = 0.5 * EPS0 * Ex**2
    u_B = 0.5 * By**2 / MU0
    u_total = u_E + u_B

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(z / lam, u_E * 1e6, 'b-', linewidth=2, label=r'$u_E = \frac{1}{2}\varepsilon_0 E^2$')
    ax.plot(z / lam, u_B * 1e6, 'r--', linewidth=2, label=r'$u_B = \frac{1}{2}B^2/\mu_0$')
    ax.plot(z / lam, u_total * 1e6, 'k-', linewidth=2, alpha=0.5,
            label=r'$u_{total} = u_E + u_B$')

    # Time-averaged energy density
    u_avg = EPS0 * E0**2 / 2  # = eps0 * E0^2 / 2 (time-avg of cos^2 = 1/2)
    ax.axhline(y=u_avg * 1e6, color='green', linestyle=':', linewidth=2,
               label=rf'$\langle u \rangle$ = {u_avg*1e6:.2f} $\mu$J/m$^3$')

    ax.set_xlabel(r'z / $\lambda$')
    ax.set_ylabel(r'Energy density ($\mu$J/m$^3$)')
    ax.set_title('Energy Density in a Plane EM Wave (E and B contribute equally)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('06_energy_density.png', dpi=150)
    plt.close()
    print("[Saved] 06_energy_density.png")

    # Verify equipartition
    print(f"  max(u_E) = {np.max(u_E):.6e} J/m^3")
    print(f"  max(u_B) = {np.max(u_B):.6e} J/m^3")
    print(f"  Ratio max(u_E)/max(u_B) = {np.max(u_E)/np.max(u_B):.6f} (should be 1.0)")


# ===========================
# Main
# ===========================

if __name__ == '__main__':
    print("\n=== Plane Wave Snapshots ===")
    plane_wave_snapshots()

    print("\n=== E-B Relationship ===")
    eb_relationship()

    print("\n=== Poynting Vector ===")
    poynting_vector()

    print("\n=== Energy Density ===")
    energy_density()
