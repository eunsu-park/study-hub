"""
Electrostatics: Coulomb's Law, Superposition, and Gauss's Law
=============================================================

Topics covered:
  1. Electric field of point charges (superposition principle)
  2. Field line visualization using streamplot
  3. Gauss's law verification (numerical surface flux integral)
  4. Electric field inside and outside a uniformly charged sphere

Why start with electrostatics?
  Electrostatics is the foundation of classical electrodynamics. Before
  tackling time-dependent phenomena (waves, radiation), we need to
  understand how *static* charge distributions create electric fields.
  Coulomb's law + superposition gives us the microscopic picture, while
  Gauss's law gives us the powerful integral tool for symmetric problems.

Physics background:
  - Coulomb's law:  E = (1/4*pi*eps0) * q / r^2  (radial, from charge)
  - Superposition:  E_total = sum of E from each charge
  - Gauss's law:    integral(E . dA) = Q_enc / eps0
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
# Why define eps0 explicitly?
#   We work in SI units throughout. Having eps0 as a named constant makes
#   the physics transparent in every formula, rather than hiding it inside
#   a combined constant like k = 1/(4*pi*eps0).
EPS0 = 8.854187817e-12  # F/m, vacuum permittivity
K_E = 1.0 / (4.0 * np.pi * EPS0)  # Coulomb constant ~8.99e9 N m^2/C^2


# ===========================
# 1. Electric Field of Point Charges (Superposition)
# ===========================

def electric_field_point_charges(charges, positions, X, Y):
    """
    Compute the electric field on a 2D grid from a set of point charges.

    Parameters
    ----------
    charges : array-like, shape (N,)
        Charge magnitudes in Coulombs.
    positions : array-like, shape (N, 2)
        (x, y) positions of each charge in meters.
    X, Y : 2D arrays (meshgrid)
        Grid coordinates where we evaluate the field.

    Returns
    -------
    Ex, Ey : 2D arrays
        x- and y-components of the electric field at each grid point.

    Why superposition works:
      Maxwell's equations (and therefore Coulomb's law) are *linear* in the
      sources. This means the field from many charges is simply the vector
      sum of the fields from each individual charge. This linearity is what
      makes the entire framework tractable.
    """
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)

    for q, (xq, yq) in zip(charges, positions):
        # Why compute displacement vectors?
        #   The electric field points radially from each charge.
        #   We need the vector from the charge to the field point.
        dx = X - xq
        dy = Y - yq

        # Why add a small epsilon to avoid division by zero?
        #   At the exact location of a point charge, the field diverges.
        #   Numerically, this would produce NaN/Inf. A tiny offset keeps
        #   the computation stable without visibly affecting the field
        #   at distances we actually plot.
        r_squared = dx**2 + dy**2 + 1e-20
        r_cubed = r_squared**1.5  # |r|^3 = (r^2)^{3/2}

        # Coulomb's law in component form:
        #   E_x = k * q * (x - xq) / |r|^3
        #   E_y = k * q * (y - yq) / |r|^3
        Ex += K_E * q * dx / r_cubed
        Ey += K_E * q * dy / r_cubed

    return Ex, Ey


def plot_field_lines(charges, positions, title="Electric Field Lines"):
    """
    Visualize electric field lines using matplotlib's streamplot.

    Why streamplot instead of quiver?
      - quiver draws arrows at grid points (shows magnitude + direction)
      - streamplot traces continuous field lines (shows topology)
      Field lines are more physically intuitive: they start on positive
      charges and end on negative charges, and their density indicates
      field strength.
    """
    # Why this grid resolution?
    #   200x200 gives smooth streamlines without excessive computation.
    #   For publication-quality plots you might use 400x400.
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)

    Ex, Ey = electric_field_point_charges(charges, positions, X, Y)

    # Why compute the magnitude separately?
    #   streamplot uses 'color' to encode field strength.
    #   We take log10 so the color map captures the large dynamic range
    #   near charges without washing out the far field.
    E_mag = np.sqrt(Ex**2 + Ey**2)
    E_log = np.log10(E_mag + 1e-10)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Why use density=2?
    #   This controls how many streamlines matplotlib draws.
    #   density=2 gives a good balance between clarity and coverage.
    strm = ax.streamplot(
        X, Y, Ex, Ey,
        color=E_log,
        cmap='inferno',
        density=2,
        linewidth=1,
        arrowsize=1.2,
    )
    fig.colorbar(strm.lines, ax=ax, label=r'$\log_{10}|E|$ (V/m)')

    # Mark charge positions
    for q, (xq, yq) in zip(charges, positions):
        color = 'red' if q > 0 else 'blue'
        marker = '+' if q > 0 else '_'
        ax.plot(xq, yq, marker=marker, color=color, markersize=15,
                markeredgewidth=3)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('01_field_lines.png', dpi=150)
    plt.close()
    print(f"[Saved] 01_field_lines.png  -- {title}")


# ===========================
# 2. Gauss's Law Verification (Numerical Flux Integral)
# ===========================

def verify_gauss_law(q, R, N_theta=200, N_phi=200):
    """
    Verify Gauss's law by computing the electric flux through a sphere
    of radius R centered on a point charge q.

    Gauss's law states:
        Phi_E = closed_integral(E . dA) = Q_enc / eps0

    Why verify numerically?
      Gauss's law is exact analytically, but verifying it numerically
      builds confidence in our field computation and demonstrates how
      to perform surface integrals -- a skill needed for more complex
      geometries where analytic solutions don't exist.

    Parameters
    ----------
    q : float
        Charge in Coulombs.
    R : float
        Radius of the Gaussian surface in meters.
    N_theta, N_phi : int
        Number of grid points in polar and azimuthal angles.
    """
    # Why use spherical coordinates?
    #   For a spherical Gaussian surface, the area element is
    #   dA = R^2 sin(theta) dtheta dphi * r_hat
    #   This is the natural coordinate system for the problem.
    theta = np.linspace(0, np.pi, N_theta)
    phi = np.linspace(0, 2 * np.pi, N_phi)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')

    # E field on the sphere surface (radially symmetric for a point charge)
    # |E| = k * q / R^2, and E is purely radial, so E . dA = |E| * dA
    E_r = K_E * q / R**2  # scalar, same everywhere on sphere

    # Why is the integrand E_r * R^2 * sin(theta)?
    #   The outward area element magnitude is |dA| = R^2 sin(theta) dtheta dphi.
    #   Since E is radial, E . dA = E_r * |dA|.
    integrand = E_r * R**2 * np.sin(THETA)
    flux_numerical = np.sum(integrand) * dtheta * dphi

    flux_analytic = q / EPS0

    print("\n=== Gauss's Law Verification ===")
    print(f"  Charge q = {q:.2e} C")
    print(f"  Gaussian sphere radius R = {R:.2f} m")
    print(f"  Numerical flux  = {flux_numerical:.6e} V*m")
    print(f"  Analytic (q/eps0) = {flux_analytic:.6e} V*m")
    print(f"  Relative error  = {abs(flux_numerical - flux_analytic) / abs(flux_analytic):.2e}")


# ===========================
# 3. Electric Field of a Uniformly Charged Sphere
# ===========================

def charged_sphere_field(Q, R_sphere, r_points):
    """
    Compute |E(r)| for a uniformly charged sphere of total charge Q
    and radius R_sphere, at radial distances r_points.

    Why is this a classic Gauss's law application?
      By spherical symmetry, E is purely radial and depends only on r.
      Gauss's law with a spherical surface of radius r gives:
        r < R:  E = (Q / 4*pi*eps0) * (r / R^3)   (linear in r)
        r > R:  E = (Q / 4*pi*eps0) * (1 / r^2)    (like a point charge)
      The field *inside* the sphere grows linearly because only the
      charge enclosed within radius r contributes.
    """
    E = np.zeros_like(r_points)

    # Inside the sphere: only the enclosed charge matters
    inside = r_points < R_sphere
    # Why r / R^3?
    #   Q_enc(r) = Q * (r/R)^3  (volume ratio for uniform density)
    #   E(r) = Q_enc / (4*pi*eps0 * r^2) = Q*r / (4*pi*eps0 * R^3)
    E[inside] = K_E * Q * r_points[inside] / R_sphere**3

    # Outside the sphere: all charge is enclosed, acts like a point charge
    outside = r_points >= R_sphere
    E[outside] = K_E * Q / r_points[outside]**2

    return E


def plot_charged_sphere():
    """Plot |E| vs r for a uniformly charged sphere."""
    Q = 1e-6       # 1 micro-Coulomb
    R_sphere = 1.0  # 1 meter radius

    r = np.linspace(0.01, 4.0, 500)
    E = charged_sphere_field(Q, R_sphere, r)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, E, 'b-', linewidth=2)
    ax.axvline(x=R_sphere, color='gray', linestyle='--', label='Sphere surface')

    # Why annotate the two regions?
    #   The inside/outside behavior is the key physics takeaway.
    ax.annotate(r'$E \propto r$ (inside)', xy=(0.5, charged_sphere_field(Q, R_sphere, np.array([0.5]))[0]),
                fontsize=11, color='green',
                xytext=(0.2, charged_sphere_field(Q, R_sphere, np.array([0.3]))[0] * 1.5),
                arrowprops=dict(arrowstyle='->', color='green'))
    ax.annotate(r'$E \propto 1/r^2$ (outside)', xy=(2.5, charged_sphere_field(Q, R_sphere, np.array([2.5]))[0]),
                fontsize=11, color='red',
                xytext=(2.8, charged_sphere_field(Q, R_sphere, np.array([1.5]))[0]),
                arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('r (m)')
    ax.set_ylabel('|E| (V/m)')
    ax.set_title('Electric Field of a Uniformly Charged Sphere')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('01_charged_sphere_E.png', dpi=150)
    plt.close()
    print("[Saved] 01_charged_sphere_E.png")


# ===========================
# Main
# ===========================

if __name__ == '__main__':
    # --- Demo 1: Dipole field lines ---
    # A positive and negative charge separated by a distance: the classic dipole.
    # Why a dipole?
    #   It's the simplest charge configuration that is overall neutral,
    #   and its field pattern (lines curving from + to -) is fundamental
    #   to understanding polarization, antennas, and molecular physics.
    charges_dipole = [1e-9, -1e-9]
    positions_dipole = [(-1.0, 0.0), (1.0, 0.0)]
    plot_field_lines(charges_dipole, positions_dipole,
                     title="Electric Dipole Field Lines")

    # --- Demo 2: Quadrupole field lines ---
    # Why a quadrupole?
    #   Quadrupole fields fall off faster (1/r^3) and show more complex topology.
    #   This demonstrates superposition with 4 charges.
    charges_quad = [1e-9, -1e-9, 1e-9, -1e-9]
    positions_quad = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    plot_field_lines(charges_quad, positions_quad,
                     title="Electric Quadrupole Field Lines")

    # --- Demo 3: Gauss's law verification ---
    verify_gauss_law(q=1e-6, R=2.0)

    # --- Demo 4: Charged sphere E(r) ---
    plot_charged_sphere()
