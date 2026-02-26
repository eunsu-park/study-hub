"""
Magnetostatics: Biot-Savart Law, Magnetic Dipoles, and Solenoids
================================================================

Topics covered:
  1. Biot-Savart law for a circular current loop
  2. B field visualization using streamplot
  3. Magnetic dipole field (far-field approximation)
  4. Solenoid field computation (finite-length)

Why magnetostatics?
  Just as electrostatics deals with fields from stationary charges,
  magnetostatics deals with fields from *steady* (DC) currents.
  The Biot-Savart law is the magnetic analogue of Coulomb's law:
  it tells us the B field from each current element dI x r_hat / r^2.
  Understanding loops and solenoids is essential for electromagnets,
  inductors, MRI machines, and particle accelerators.

Physics background:
  - Biot-Savart:  dB = (mu0 / 4*pi) * I * dl x r_hat / r^2
  - Magnetic dipole moment: m = I * A * n_hat  (current x area)
  - Far field of a dipole: B = (mu0/4*pi) * [3(m.r_hat)r_hat - m] / r^3
  - Solenoid (infinite): B = mu0 * n * I  (uniform inside)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU0 = 4 * np.pi * 1e-7  # T*m/A, vacuum permeability


# ===========================
# 1. Biot-Savart Law for a Circular Loop
# ===========================

def biot_savart_loop(I, R_loop, N_seg=500):
    """
    Compute B field of a circular current loop using the Biot-Savart law.

    The loop lies in the x-y plane, centered at the origin, with current I
    and radius R_loop.

    Parameters
    ----------
    I : float
        Current in Amperes.
    R_loop : float
        Loop radius in meters.
    N_seg : int
        Number of segments to discretize the loop.

    Returns
    -------
    Callable: field_func(x, y, z) -> (Bx, By, Bz)

    Why discretize the loop into segments?
      The Biot-Savart integral can be evaluated analytically only on the
      axis of the loop (using elliptic integrals off-axis). By breaking
      the loop into N small straight segments, we can compute the field
      at ANY point in space by summing contributions from each segment.
      This is the numerical equivalent of the integral.
    """
    # Parameterize the loop
    phi = np.linspace(0, 2 * np.pi, N_seg, endpoint=False)
    dphi = 2 * np.pi / N_seg

    # Why compute dl as tangent vectors?
    #   Each segment has a position r' on the loop and a current element
    #   dl = R * dphi * (-sin(phi), cos(phi), 0). This is the tangent
    #   direction times the arc length.
    x_loop = R_loop * np.cos(phi)
    y_loop = R_loop * np.sin(phi)
    z_loop = np.zeros_like(phi)

    dlx = -R_loop * np.sin(phi) * dphi
    dly = R_loop * np.cos(phi) * dphi
    dlz = np.zeros_like(phi)

    def field_func(xp, yp, zp):
        """Compute B at point (xp, yp, zp)."""
        Bx = 0.0
        By = 0.0
        Bz = 0.0

        for i in range(N_seg):
            # Displacement from source to field point
            rx = xp - x_loop[i]
            ry = yp - y_loop[i]
            rz = zp - z_loop[i]
            r_mag = np.sqrt(rx**2 + ry**2 + rz**2 + 1e-30)
            r_cubed = r_mag**3

            # Biot-Savart: dB = (mu0*I / 4*pi) * (dl x r_hat) / r^2
            #                 = (mu0*I / 4*pi) * (dl x r) / r^3
            # Why dl x r instead of dl x r_hat / r^2?
            #   Both are equivalent. Using r (not r_hat) and dividing by r^3
            #   avoids computing r_hat separately -- one fewer division.
            cross_x = dly[i] * rz - dlz[i] * ry
            cross_y = dlz[i] * rx - dlx[i] * rz
            cross_z = dlx[i] * ry - dly[i] * rx

            factor = (MU0 * I) / (4 * np.pi * r_cubed)
            Bx += factor * cross_x
            By += factor * cross_y
            Bz += factor * cross_z

        return Bx, By, Bz

    return field_func


def compute_loop_field_on_grid(I, R_loop, x_range, z_range, Nx=60, Nz=60):
    """
    Evaluate B field on a 2D grid in the x-z plane (y=0).

    Why the x-z plane?
      For a loop in the x-y plane, the field is axially symmetric about z.
      The x-z plane captures both the axial component (Bz, dominant on axis)
      and the radial component (Bx, dominant off-axis). By symmetry, By=0
      in this plane.
    """
    field_func = biot_savart_loop(I, R_loop)

    x = np.linspace(*x_range, Nx)
    z = np.linspace(*z_range, Nz)
    X, Z = np.meshgrid(x, z)

    Bx = np.zeros_like(X)
    Bz = np.zeros_like(Z)

    for i in range(Nz):
        for j in range(Nx):
            bx, _, bz = field_func(X[i, j], 0.0, Z[i, j])
            Bx[i, j] = bx
            Bz[i, j] = bz

    return X, Z, Bx, Bz


# ===========================
# 2. Magnetic Dipole Field (Analytic)
# ===========================

def magnetic_dipole_field(m, X, Z):
    """
    Compute the magnetic dipole field in the x-z plane.

    B = (mu0 / 4*pi*r^5) * [3*z*(x*x_hat + z*z_hat) - r^2 * z_hat] * m
      (simplified for m = m*z_hat, in x-z plane)

    More precisely:
      Br = (mu0 * m * 2*cos(theta)) / (4*pi*r^3)
      Btheta = (mu0 * m * sin(theta)) / (4*pi*r^3)

    Why compare with the exact Biot-Savart result?
      The dipole approximation is valid only when r >> R_loop.
      Comparing shows where the approximation breaks down and builds
      intuition for when to use it.
    """
    r = np.sqrt(X**2 + Z**2 + 1e-30)
    r5 = r**5

    # Why express in Cartesian rather than spherical?
    #   The streamplot function needs Cartesian components.
    #   We convert the standard dipole formula:
    #     Bx = (mu0*m / 4*pi) * 3*x*z / r^5
    #     Bz = (mu0*m / 4*pi) * (3*z^2 - r^2) / r^5
    prefactor = MU0 * m / (4 * np.pi)
    Bx = prefactor * 3 * X * Z / r5
    Bz = prefactor * (3 * Z**2 - r**2) / r5

    return Bx, Bz


# ===========================
# 3. Solenoid Field (Superposition of Loops)
# ===========================

def solenoid_field(I, R_sol, L_sol, N_turns, Nx=50, Nz=80):
    """
    Compute the B field of a finite solenoid by superposing circular loops.

    Parameters
    ----------
    I : float
        Current (A).
    R_sol : float
        Solenoid radius (m).
    L_sol : float
        Solenoid length (m).
    N_turns : int
        Total number of turns.

    Why superposition of loops?
      A solenoid is just N circular loops stacked along the z-axis.
      By the superposition principle, B_total = sum of B from each loop.
      This is brute-force but physically transparent.

    Why not just use B = mu0*n*I?
      That formula is for an *infinite* solenoid. A finite solenoid has
      weaker field near the ends and non-zero field outside. Our numerical
      approach captures these real-world features.
    """
    print(f"  Computing solenoid field ({N_turns} loops)...")

    # Position each loop along z-axis
    z_positions = np.linspace(-L_sol / 2, L_sol / 2, N_turns)

    x = np.linspace(-R_sol * 3, R_sol * 3, Nx)
    z = np.linspace(-L_sol, L_sol, Nz)
    X, Z = np.meshgrid(x, z)

    Bx_total = np.zeros_like(X)
    Bz_total = np.zeros_like(Z)

    # Why only 100 segments per loop for the solenoid?
    #   With N_turns loops x N_seg segments, total computation scales as
    #   N_turns * N_seg * Nx * Nz. Using 100 segments (vs 500 for a single
    #   loop) keeps runtime manageable while maintaining adequate accuracy.
    field_func = biot_savart_loop(I, R_sol, N_seg=100)

    for z0 in z_positions:
        for i in range(Nz):
            for j in range(Nx):
                bx, _, bz = field_func(X[i, j], 0.0, Z[i, j] - z0)
                Bx_total[i, j] += bx
                Bz_total[i, j] += bz

    return X, Z, Bx_total, Bz_total, x, z


# ===========================
# 4. Visualization
# ===========================

def plot_loop_field(X, Z, Bx, Bz, R_loop, title="Current Loop B Field"):
    """Plot magnetic field of a current loop using streamplot."""
    fig, ax = plt.subplots(figsize=(8, 8))

    B_mag = np.sqrt(Bx**2 + Bz**2)
    B_log = np.log10(B_mag + 1e-15)

    strm = ax.streamplot(X, Z, Bx, Bz, color=B_log, cmap='plasma',
                         density=2, linewidth=1)
    fig.colorbar(strm.lines, ax=ax, label=r'$\log_{10}|B|$ (T)')

    # Mark the loop cross-section (ring in x-z plane at y=0)
    ax.plot([-R_loop, R_loop], [0, 0], 'ro', markersize=8, label='Loop cross-section')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('04_loop_field.png', dpi=150)
    plt.close()
    print("[Saved] 04_loop_field.png")


def plot_dipole_comparison(X, Z, Bx_exact, Bz_exact, Bx_dipole, Bz_dipole, R_loop):
    """Compare Biot-Savart (exact) vs dipole approximation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, Bx, Bz, title in [
        (axes[0], Bx_exact, Bz_exact, 'Biot-Savart (exact)'),
        (axes[1], Bx_dipole, Bz_dipole, 'Dipole approximation'),
    ]:
        B_mag = np.sqrt(Bx**2 + Bz**2)
        B_log = np.log10(B_mag + 1e-15)
        strm = ax.streamplot(X, Z, Bx, Bz, color=B_log, cmap='plasma',
                             density=2, linewidth=1)
        fig.colorbar(strm.lines, ax=ax, label=r'$\log_{10}|B|$ (T)')
        ax.plot([-R_loop, R_loop], [0, 0], 'ro', markersize=6)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('z (m)')
        ax.set_title(title)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('04_dipole_comparison.png', dpi=150)
    plt.close()
    print("[Saved] 04_dipole_comparison.png")


def plot_solenoid(X, Z, Bx, Bz, R_sol, L_sol):
    """Plot solenoid magnetic field."""
    fig, ax = plt.subplots(figsize=(10, 7))

    B_mag = np.sqrt(Bx**2 + Bz**2)
    B_log = np.log10(B_mag + 1e-15)

    strm = ax.streamplot(X, Z, Bx, Bz, color=B_log, cmap='plasma',
                         density=2.5, linewidth=1)
    fig.colorbar(strm.lines, ax=ax, label=r'$\log_{10}|B|$ (T)')

    # Draw solenoid outline
    ax.plot([-R_sol, -R_sol], [-L_sol / 2, L_sol / 2], 'c-', linewidth=2)
    ax.plot([R_sol, R_sol], [-L_sol / 2, L_sol / 2], 'c-', linewidth=2)
    ax.plot([-R_sol, R_sol], [-L_sol / 2, -L_sol / 2], 'c-', linewidth=2)
    ax.plot([-R_sol, R_sol], [L_sol / 2, L_sol / 2], 'c-', linewidth=2)

    # Compare with ideal solenoid field on axis
    n = 50 / L_sol  # turns per length (N_turns=50 default)
    B_ideal = MU0 * n * 1.0  # for I=1A
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_title(f'Finite Solenoid B Field (ideal B_z = {B_ideal:.2e} T inside)')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('04_solenoid_field.png', dpi=150)
    plt.close()
    print("[Saved] 04_solenoid_field.png")


# ===========================
# Main
# ===========================

if __name__ == '__main__':
    I = 1.0          # Current (A)
    R_loop = 0.5     # Loop radius (m)

    # --- 1. Current loop B field (Biot-Savart) ---
    print("=== Current Loop (Biot-Savart) ===")
    X, Z, Bx, Bz = compute_loop_field_on_grid(
        I, R_loop, x_range=(-2, 2), z_range=(-2, 2), Nx=50, Nz=50
    )
    plot_loop_field(X, Z, Bx, Bz, R_loop)

    # On-axis field check: B_z(0,0,z) = mu0*I*R^2 / (2*(R^2+z^2)^{3/2})
    field_func = biot_savart_loop(I, R_loop)
    _, _, Bz_center = field_func(0, 0, 0)
    Bz_analytic = MU0 * I / (2 * R_loop)
    print(f"  B_z at center: numerical = {Bz_center:.6e} T, "
          f"analytic = {Bz_analytic:.6e} T")

    # --- 2. Dipole approximation comparison ---
    print("\n=== Dipole Approximation ===")
    m = I * np.pi * R_loop**2  # magnetic moment
    Bx_dip, Bz_dip = magnetic_dipole_field(m, X, Z)
    plot_dipole_comparison(X, Z, Bx, Bz, Bx_dip, Bz_dip, R_loop)

    # --- 3. Solenoid field ---
    print("\n=== Finite Solenoid ===")
    R_sol = 0.3
    L_sol = 1.0
    N_turns = 50
    Xs, Zs, Bxs, Bzs, xs, zs = solenoid_field(I, R_sol, L_sol, N_turns,
                                                 Nx=40, Nz=60)
    plot_solenoid(Xs, Zs, Bxs, Bzs, R_sol, L_sol)

    # On-axis field profile
    mid_x = len(xs) // 2
    Bz_axis = Bzs[:, mid_x]
    B_ideal = MU0 * (N_turns / L_sol) * I
    print(f"  Ideal infinite solenoid B = {B_ideal:.6e} T")
    print(f"  Numerical B_z at center   = {Bz_axis[len(zs) // 2]:.6e} T")
    print(f"  Numerical B_z at end      = {Bz_axis[0]:.6e} T")
    print(f"  Ratio end/center          = {Bz_axis[0] / Bz_axis[len(zs) // 2]:.3f} "
          f"(should be ~0.5 for long solenoid)")
