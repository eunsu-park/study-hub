"""
Exercises for Lesson 01: Electrostatics Review
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Constants
epsilon_0 = 8.854e-12  # permittivity of free space (C^2/(N*m^2))
k_e = 1.0 / (4.0 * np.pi * epsilon_0)  # Coulomb constant


def exercise_1():
    """
    Exercise 1: Superposition Practice
    Two charges q1 = +3 uC at (0,0,0) and q2 = -5 uC at (1,0,0) m.
    Find the electric field at (0.5, 0.5, 0) m analytically and numerically.
    """
    # Charge positions and values
    q1 = 3e-6   # +3 uC
    q2 = -5e-6  # -5 uC
    r1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([1.0, 0.0, 0.0])
    P = np.array([0.5, 0.5, 0.0])  # field point

    # Analytical: E = (1/4*pi*eps0) * q * r_hat / r^2 for each charge
    # Contribution from q1
    r_P1 = P - r1
    dist_P1 = np.linalg.norm(r_P1)
    E1 = k_e * q1 * r_P1 / dist_P1**3

    # Contribution from q2
    r_P2 = P - r2
    dist_P2 = np.linalg.norm(r_P2)
    E2 = k_e * q2 * r_P2 / dist_P2**3

    # Total field by superposition
    E_total = E1 + E2
    E_mag = np.linalg.norm(E_total)

    print("  Charge q1 = +3 uC at (0, 0, 0)")
    print("  Charge q2 = -5 uC at (1, 0, 0)")
    print("  Field point P = (0.5, 0.5, 0)")
    print()
    print(f"  E from q1: ({E1[0]:.2f}, {E1[1]:.2f}, {E1[2]:.2f}) V/m")
    print(f"  E from q2: ({E2[0]:.2f}, {E2[1]:.2f}, {E2[2]:.2f}) V/m")
    print(f"  E total:   ({E_total[0]:.2f}, {E_total[1]:.2f}, {E_total[2]:.2f}) V/m")
    print(f"  |E| = {E_mag:.2f} V/m")

    # Direction angle (in xy-plane)
    theta = np.degrees(np.arctan2(E_total[1], E_total[0]))
    print(f"  Direction: {theta:.2f} degrees from +x axis")

    # Numerical verification using a grid approach
    # Compute the field at a small neighborhood around P to verify consistency
    delta = 1e-6
    V_P = k_e * q1 / dist_P1 + k_e * q2 / dist_P2

    # Numerical gradient of V to get E = -grad(V)
    E_numerical = np.zeros(3)
    for i in range(3):
        P_plus = P.copy()
        P_plus[i] += delta
        r1_plus = np.linalg.norm(P_plus - r1)
        r2_plus = np.linalg.norm(P_plus - r2)
        V_plus = k_e * q1 / r1_plus + k_e * q2 / r2_plus
        E_numerical[i] = -(V_plus - V_P) / delta

    print(f"\n  Numerical E (from -grad V): ({E_numerical[0]:.2f}, "
          f"{E_numerical[1]:.2f}, {E_numerical[2]:.2f}) V/m")
    print(f"  Agreement: {np.linalg.norm(E_total - E_numerical)/E_mag:.2e} relative error")


def exercise_2():
    """
    Exercise 2: Gauss's Law -- Spherical Shell
    A thin spherical shell of radius R carries total charge Q.
    Prove E=0 inside, E=Q/(4*pi*eps0*r^2) outside.
    Numerically verify the discontinuity at r=R.
    """
    Q = 1e-9   # 1 nC
    R = 0.1    # shell radius 10 cm

    # Analytical field from Gauss's law
    r = np.linspace(0.001, 0.3, 1000)
    E_analytic = np.where(r < R, 0.0, k_e * Q / r**2)

    # Numerical verification: model shell as many point charges on the surface
    N_charges = 2000  # number of point charges on the shell
    dq = Q / N_charges

    # Distribute charges uniformly on sphere using golden spiral
    indices = np.arange(0, N_charges, dtype=float) + 0.5
    phi_angles = np.arccos(1 - 2 * indices / N_charges)
    theta_angles = np.pi * (1 + 5**0.5) * indices

    # Charge positions on the shell
    qx = R * np.sin(phi_angles) * np.cos(theta_angles)
    qy = R * np.sin(phi_angles) * np.sin(theta_angles)
    qz = R * np.cos(phi_angles)

    # Compute E along z-axis (x=0, y=0) for symmetry
    E_numerical = np.zeros(len(r))
    for i, ri in enumerate(r):
        # Field point on z-axis
        dx = 0.0 - qx
        dy = 0.0 - qy
        dz = ri - qz
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dist = np.maximum(dist, 1e-10)
        # Only z-component survives by symmetry
        E_numerical[i] = k_e * np.sum(dq * dz / dist**3)

    print(f"  Shell radius R = {R*100:.0f} cm, charge Q = {Q*1e9:.1f} nC")
    print()

    # Check values inside and outside
    idx_inside = np.argmin(np.abs(r - 0.05))
    idx_outside = np.argmin(np.abs(r - 0.2))
    print(f"  At r = 5 cm (inside):")
    print(f"    Analytic E = {E_analytic[idx_inside]:.4f} V/m")
    print(f"    Numerical E = {E_numerical[idx_inside]:.4f} V/m")
    print(f"  At r = 20 cm (outside):")
    print(f"    Analytic E = {E_analytic[idx_outside]:.4f} V/m")
    print(f"    Numerical E = {E_numerical[idx_outside]:.4f} V/m")

    # Discontinuity at R
    idx_just_below = np.argmin(np.abs(r - (R - 0.001)))
    idx_just_above = np.argmin(np.abs(r - (R + 0.001)))
    E_below = E_numerical[idx_just_below]
    E_above = E_numerical[idx_just_above]
    E_surface = k_e * Q / R**2

    print(f"\n  Discontinuity at r = R:")
    print(f"    E just below R: {E_below:.2f} V/m (should be ~0)")
    print(f"    E just above R: {E_above:.2f} V/m (should be ~{E_surface:.2f})")
    print(f"    Surface field (sigma/eps0): {Q/(4*np.pi*R**2)/epsilon_0:.2f} V/m")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r * 100, E_analytic, 'b-', linewidth=2, label='Analytic (Gauss)')
    ax.plot(r * 100, E_numerical, 'r--', linewidth=1.5, label=f'Numerical (N={N_charges})')
    ax.axvline(x=R * 100, color='gray', linestyle=':', label=f'R = {R*100:.0f} cm')
    ax.set_xlabel('r (cm)')
    ax.set_ylabel('E (V/m)')
    ax.set_title('Electric Field of a Spherical Shell')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex01_spherical_shell.png', dpi=150)
    plt.close()
    print("\n  Plot saved: ex01_spherical_shell.png")


def exercise_3():
    """
    Exercise 3: Non-Uniform Charge Distribution
    Solid sphere with rho(r) = rho_0*(1 - r/R) for r <= R.
    Find E(r) for r < R and r > R using Gauss's law. Plot the result.
    """
    R = 0.1      # sphere radius (10 cm)
    rho_0 = 1e-6  # charge density scale (C/m^3)

    # Total charge: Q = integral of rho * 4*pi*r^2 dr from 0 to R
    # Q = 4*pi*rho_0 * integral_0^R r^2(1 - r/R) dr
    # = 4*pi*rho_0 * [R^3/3 - R^3/4] = 4*pi*rho_0 * R^3/12 = pi*rho_0*R^3/3
    Q_total = np.pi * rho_0 * R**3 / 3.0

    r = np.linspace(0.001, 0.3, 1000)

    # Inside (r < R): enclosed charge Q_enc = 4*pi*rho_0 * integral_0^r s^2(1-s/R) ds
    # = 4*pi*rho_0 * [r^3/3 - r^4/(4R)]
    def Q_enc(ri):
        if ri <= R:
            return 4 * np.pi * rho_0 * (ri**3 / 3 - ri**4 / (4 * R))
        else:
            return Q_total

    E = np.zeros_like(r)
    for i, ri in enumerate(r):
        Q_enclosed = Q_enc(ri)
        E[i] = k_e * Q_enclosed / ri**2

    print(f"  Sphere radius R = {R*100:.0f} cm")
    print(f"  rho_0 = {rho_0:.1e} C/m^3")
    print(f"  Total charge Q = {Q_total:.4e} C")
    print()

    # Analytic expressions
    # Inside: E(r) = (rho_0/(3*eps_0)) * r * (1 - 3r/(4R))
    r_inside = r[r <= R]
    E_inside = (rho_0 / (3 * epsilon_0)) * r_inside * (1 - 3 * r_inside / (4 * R))

    # Outside: E(r) = Q_total / (4*pi*eps_0*r^2)
    r_outside = r[r > R]
    E_outside = k_e * Q_total / r_outside**2

    E_analytic = np.concatenate([E_inside, E_outside])

    print(f"  E at r=R/2: {E[np.argmin(np.abs(r - R/2))]:.4f} V/m")
    print(f"  E at r=R:   {E[np.argmin(np.abs(r - R))]:.4f} V/m")
    print(f"  E at r=2R:  {E[np.argmin(np.abs(r - 2*R))]:.4f} V/m")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r * 100, E, 'b-', linewidth=2, label='E(r)')
    ax.axvline(x=R * 100, color='gray', linestyle=':', alpha=0.7, label=f'R = {R*100:.0f} cm')
    ax.set_xlabel('r (cm)')
    ax.set_ylabel('E (V/m)')
    ax.set_title(r'E-field for $\rho(r) = \rho_0(1 - r/R)$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex01_non_uniform_sphere.png', dpi=150)
    plt.close()
    print("  Plot saved: ex01_non_uniform_sphere.png")


def exercise_4():
    """
    Exercise 4: Numerical Flux Calculation
    Three point charges inside a cube. Compute electric flux through each face
    and verify total flux = Q_enc / epsilon_0.
    """
    # Three charges at arbitrary positions inside unit cube [0,1]^3
    charges = [
        (np.array([0.3, 0.4, 0.5]), 2e-9),   # 2 nC
        (np.array([0.7, 0.2, 0.6]), -1e-9),   # -1 nC
        (np.array([0.5, 0.8, 0.3]), 3e-9),    # 3 nC
    ]
    Q_enc = sum(q for _, q in charges)

    # Compute flux through each face of the cube by numerical integration
    # Faces: x=0, x=1, y=0, y=1, z=0, z=1
    N = 200  # grid points per edge
    s = np.linspace(0, 1, N)
    ds = s[1] - s[0]
    S1, S2 = np.meshgrid(s, s)

    face_labels = ['x=0', 'x=1', 'y=0', 'y=1', 'z=0', 'z=1']
    total_flux = 0.0

    print(f"  Charges: {[(list(pos), f'{q*1e9:.1f} nC') for pos, q in charges]}")
    print(f"  Total enclosed charge: {Q_enc*1e9:.1f} nC")
    print(f"  Expected flux (Q/eps_0): {Q_enc/epsilon_0:.6f} V*m")
    print()

    fluxes = []
    for face_idx in range(6):
        # Build surface points and outward normal for each face
        if face_idx == 0:    # x = 0 face, normal = -x
            pts = np.stack([np.zeros_like(S1), S1, S2], axis=-1)
            normal = np.array([-1, 0, 0])
        elif face_idx == 1:  # x = 1 face, normal = +x
            pts = np.stack([np.ones_like(S1), S1, S2], axis=-1)
            normal = np.array([1, 0, 0])
        elif face_idx == 2:  # y = 0 face, normal = -y
            pts = np.stack([S1, np.zeros_like(S1), S2], axis=-1)
            normal = np.array([0, -1, 0])
        elif face_idx == 3:  # y = 1 face, normal = +y
            pts = np.stack([S1, np.ones_like(S1), S2], axis=-1)
            normal = np.array([0, 1, 0])
        elif face_idx == 4:  # z = 0 face, normal = -z
            pts = np.stack([S1, S2, np.zeros_like(S1)], axis=-1)
            normal = np.array([0, 0, -1])
        else:                # z = 1 face, normal = +z
            pts = np.stack([S1, S2, np.ones_like(S1)], axis=-1)
            normal = np.array([0, 0, 1])

        # Compute E dot n at each surface point
        E_dot_n = np.zeros_like(S1)
        for pos, q in charges:
            r_vec = pts - pos  # shape: (N, N, 3)
            r_mag = np.sqrt(np.sum(r_vec**2, axis=-1))
            r_mag = np.maximum(r_mag, 1e-10)
            E = k_e * q * r_vec / r_mag[:, :, np.newaxis]**3
            E_dot_n += np.sum(E * normal, axis=-1)

        flux = np.sum(E_dot_n) * ds**2
        fluxes.append(flux)
        total_flux += flux

    for label, flux in zip(face_labels, fluxes):
        print(f"  Flux through {label}: {flux:.6f} V*m")

    print(f"\n  Total numerical flux: {total_flux:.6f} V*m")
    print(f"  Exact flux (Q/eps_0): {Q_enc/epsilon_0:.6f} V*m")
    print(f"  Relative error: {abs(total_flux - Q_enc/epsilon_0)/(Q_enc/epsilon_0):.4e}")


def exercise_5():
    """
    Exercise 5: Dipole Field Visualization
    Compute and visualize the electric field of a quadrupole:
    +q, -q, -q, +q at the corners of a square.
    Compare far-field behavior with dipole (1/r^3 vs 1/r^4).
    """
    q = 1e-9     # charge magnitude (1 nC)
    d = 0.2      # half-side of square

    # Quadrupole: charges at corners of a square in xy-plane
    charges = [
        (-d, -d, +q),   # bottom-left: +q
        (+d, -d, -q),   # bottom-right: -q
        (-d, +d, -q),   # top-left: -q
        (+d, +d, +q),   # top-right: +q
    ]

    # Create grid
    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)

    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)

    for (cx, cy, qi) in charges:
        dx = X - cx
        dy = Y - cy
        r_sq = dx**2 + dy**2
        r_sq = np.maximum(r_sq, 0.01)
        r = np.sqrt(r_sq)
        Ex += k_e * qi * dx / r**3
        Ey += k_e * qi * dy / r**3

    E_mag = np.sqrt(Ex**2 + Ey**2)

    # Plot field lines
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].streamplot(X, Y, Ex, Ey, color=np.log10(E_mag + 1),
                       cmap='inferno', density=2, linewidth=0.8)
    for (cx, cy, qi) in charges:
        color = 'red' if qi > 0 else 'blue'
        axes[0].plot(cx, cy, 'o', color=color, markersize=10)
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    axes[0].set_title('Quadrupole Field Lines')
    axes[0].set_aspect('equal')

    # Far-field: E along x-axis (equatorial direction)
    r_vals = np.linspace(0.5, 5.0, 100)
    E_along_x = np.zeros(len(r_vals))
    for i, ri in enumerate(r_vals):
        E_total = 0.0
        for (cx, cy, qi) in charges:
            dx = ri - cx
            dy = 0 - cy
            r_mag = np.sqrt(dx**2 + dy**2)
            # x-component of field
            E_total += k_e * qi * dx / r_mag**3
        E_along_x[i] = abs(E_total)

    # Fit power law: E ~ A * r^n
    log_r = np.log(r_vals)
    log_E = np.log(E_along_x + 1e-30)
    # Linear fit in log-log space
    coeffs = np.polyfit(log_r, log_E, 1)
    power_law_exp = coeffs[0]

    axes[1].loglog(r_vals, E_along_x, 'b-', linewidth=2, label='Quadrupole |E|')
    axes[1].loglog(r_vals, np.exp(coeffs[1]) * r_vals**coeffs[0], 'r--',
                   linewidth=1.5, label=f'Fit: r^{power_law_exp:.2f}')
    axes[1].set_xlabel('r (m)')
    axes[1].set_ylabel('|E| (V/m)')
    axes[1].set_title('Far-Field Decay')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Electric Quadrupole', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex01_quadrupole.png', dpi=150)
    plt.close()

    print("  Quadrupole: +q, -q, -q, +q at corners of a square")
    print(f"  Far-field power law exponent: {power_law_exp:.2f}")
    print("  (Dipole falls as r^-3, quadrupole should fall as r^-4)")
    print("  Plot saved: ex01_quadrupole.png")


if __name__ == "__main__":
    print("=== Exercise 1: Superposition Practice ===")
    exercise_1()
    print("\n=== Exercise 2: Gauss's Law -- Spherical Shell ===")
    exercise_2()
    print("\n=== Exercise 3: Non-Uniform Charge Distribution ===")
    exercise_3()
    print("\n=== Exercise 4: Numerical Flux Calculation ===")
    exercise_4()
    print("\n=== Exercise 5: Dipole Field Visualization (Quadrupole) ===")
    exercise_5()
    print("\nAll exercises completed!")
