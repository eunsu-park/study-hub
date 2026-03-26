"""
Exercises for Lesson 02: Electric Potential and Energy
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Constants
epsilon_0 = 8.854e-12
k_e = 1.0 / (4.0 * np.pi * epsilon_0)


def exercise_1():
    """
    Exercise 1: Potential and Field Computation
    Thin ring of radius R=0.15 m, total charge Q=2 nC.
    Compute and plot V(z) and E_z(z) along the axis.
    Verify E_z = -dV/dz numerically.
    """
    R = 0.15    # ring radius (m)
    Q = 2e-9    # total charge (C)

    z = np.linspace(-0.6, 0.6, 1000)

    # Analytic potential on axis: V(z) = Q / (4*pi*eps_0 * sqrt(z^2 + R^2))
    V = k_e * Q / np.sqrt(z**2 + R**2)

    # Analytic E_z on axis: E_z = Q*z / (4*pi*eps_0*(z^2+R^2)^(3/2))
    E_z_analytic = k_e * Q * z / (z**2 + R**2)**1.5

    # Numerical E_z = -dV/dz
    E_z_numerical = -np.gradient(V, z)

    # Relative error (avoiding division by zero near z=0)
    mask = np.abs(E_z_analytic) > 1e-3
    rel_error = np.max(np.abs(E_z_numerical[mask] - E_z_analytic[mask]) /
                       np.abs(E_z_analytic[mask]))

    print(f"  Ring: R = {R*100:.0f} cm, Q = {Q*1e9:.1f} nC")
    print(f"  V at z=0: {V[len(z)//2]:.4f} V")
    print(f"  Max |E_z|: {np.max(np.abs(E_z_analytic)):.4f} V/m")
    print(f"  Max relative error (E_z numerical vs analytic): {rel_error:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(z * 100, V, 'b-', linewidth=2)
    axes[0].set_xlabel('z (cm)')
    axes[0].set_ylabel('V (V)')
    axes[0].set_title('Potential V(z) on Ring Axis')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(z * 100, E_z_analytic, 'b-', linewidth=2, label='Analytic')
    axes[1].plot(z * 100, E_z_numerical, 'r--', linewidth=1.5, label='Numerical (-dV/dz)')
    axes[1].set_xlabel('z (cm)')
    axes[1].set_ylabel('E_z (V/m)')
    axes[1].set_title('Electric Field E_z(z) on Ring Axis')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex02_ring_potential.png', dpi=150)
    plt.close()
    print("  Plot saved: ex02_ring_potential.png")


def exercise_2():
    """
    Exercise 2: Laplace Solver with Complex Boundaries
    Solve Laplace's equation in a square with V = 100*sin(pi*x/L) on top,
    V = 0 on other edges. Compare with analytic solution:
    V(x,y) = 100*sin(pi*x/L)*sinh(pi*y/L)/sinh(pi)
    """
    N = 100  # grid points per side
    L = 1.0  # domain size

    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    dx = x[1] - x[0]

    V = np.zeros((N, N))

    # Boundary conditions
    # Top edge (row 0 = y=L in our indexing where row 0 is top)
    # We use row index 0 as y=L (top) and row N-1 as y=0 (bottom)
    V[0, :] = 100.0 * np.sin(np.pi * x / L)  # top: V = 100*sin(pi*x/L)
    V[-1, :] = 0.0     # bottom
    V[:, 0] = 0.0      # left
    V[:, -1] = 0.0     # right

    # Jacobi relaxation
    n_iter = 5000
    for iteration in range(n_iter):
        V_old = V.copy()
        V[1:-1, 1:-1] = 0.25 * (
            V_old[0:-2, 1:-1] + V_old[2:, 1:-1] +
            V_old[1:-1, 0:-2] + V_old[1:-1, 2:]
        )
        # Re-enforce boundary conditions
        V[0, :] = 100.0 * np.sin(np.pi * x / L)
        V[-1, :] = 0.0
        V[:, 0] = 0.0
        V[:, -1] = 0.0

    # Analytic solution: V(x,y) = 100*sin(pi*x/L)*sinh(pi*y/L)/sinh(pi)
    X, Y = np.meshgrid(x, y)
    # Row 0 = top (y=L), row N-1 = bottom (y=0)
    Y_phys = L - Y  # map so that row 0 corresponds to y=L
    # Actually we need: row index i -> y = L - i*dx
    Y_grid = np.linspace(L, 0, N).reshape(-1, 1) * np.ones((1, N))
    V_analytic = 100.0 * np.sin(np.pi * X / L) * np.sinh(np.pi * Y_grid / L) / np.sinh(np.pi)

    # Error
    max_error = np.max(np.abs(V - V_analytic))
    mean_error = np.mean(np.abs(V - V_analytic))

    print(f"  Grid: {N}x{N}, iterations: {n_iter}")
    print(f"  Max error (numerical vs analytic): {max_error:.4f} V")
    print(f"  Mean error: {mean_error:.4f} V")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cs0 = axes[0].contourf(X, Y_grid, V, levels=30, cmap='hot')
    plt.colorbar(cs0, ax=axes[0], label='V (volts)')
    axes[0].set_title('Numerical Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    cs1 = axes[1].contourf(X, Y_grid, V_analytic, levels=30, cmap='hot')
    plt.colorbar(cs1, ax=axes[1], label='V (volts)')
    axes[1].set_title('Analytic Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    cs2 = axes[2].contourf(X, Y_grid, np.abs(V - V_analytic), levels=30, cmap='viridis')
    plt.colorbar(cs2, ax=axes[2], label='|Error| (V)')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')

    plt.suptitle("Laplace Equation: sin(pi*x) Boundary", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex02_laplace_sin_boundary.png', dpi=150)
    plt.close()
    print("  Plot saved: ex02_laplace_sin_boundary.png")


def exercise_3():
    """
    Exercise 3: Energy of Concentric Shells
    Two concentric spherical shells of radii a and b (a < b) carry charges Qa and Qb.
    Calculate total electrostatic energy using the field method.
    Verify for special case Qa = -Qb (capacitor).
    """
    a = 0.05    # inner shell radius (m)
    b = 0.10    # outer shell radius (m)
    Qa = 2e-9   # inner charge (2 nC)
    Qb = -2e-9  # outer charge (-2 nC) -- capacitor case

    # Electric field in three regions:
    # r < a: E = 0 (no enclosed charge)
    # a < r < b: E = Qa / (4*pi*eps_0*r^2)
    # r > b: E = (Qa + Qb) / (4*pi*eps_0*r^2)

    # Energy: W = (eps_0/2) * integral of E^2 dV
    # Region 1 (r < a): 0
    # Region 2 (a < r < b): W2 = (Qa^2)/(8*pi*eps_0) * integral_a^b dr/r^2
    #                       = (Qa^2)/(8*pi*eps_0) * (1/a - 1/b)
    # Region 3 (r > b): W3 = ((Qa+Qb)^2)/(8*pi*eps_0) * integral_b^inf dr/r^2
    #                   = ((Qa+Qb)^2)/(8*pi*eps_0*b)

    W2 = Qa**2 / (8 * np.pi * epsilon_0) * (1.0/a - 1.0/b)
    Q_total = Qa + Qb
    W3 = Q_total**2 / (8 * np.pi * epsilon_0 * b)
    W_total = W2 + W3

    print(f"  Inner shell: a = {a*100:.0f} cm, Qa = {Qa*1e9:.1f} nC")
    print(f"  Outer shell: b = {b*100:.0f} cm, Qb = {Qb*1e9:.1f} nC")
    print()
    print(f"  Energy in region a < r < b: {W2:.6e} J")
    print(f"  Energy in region r > b:     {W3:.6e} J")
    print(f"  Total energy:               {W_total:.6e} J")

    # Special case: Qa = -Qb (capacitor)
    # Then Q_total = 0, W3 = 0
    # W = Qa^2/(8*pi*eps_0) * (1/a - 1/b)
    if abs(Qa + Qb) < 1e-15:
        print("\n  Special case: Qa = -Qb (spherical capacitor)")
        C = 4 * np.pi * epsilon_0 * a * b / (b - a)
        V_cap = Qa / C
        W_cap = 0.5 * C * V_cap**2
        print(f"  Capacitance: C = {C:.4e} F = {C*1e12:.4f} pF")
        print(f"  Voltage: V = {V_cap:.4f} V")
        print(f"  Energy (1/2 CV^2): {W_cap:.6e} J")
        print(f"  Energy (field method): {W_total:.6e} J")
        print(f"  Agreement: {abs(W_cap - W_total)/W_total:.2e} relative error")

    # Numerical verification by integrating E^2 * 4*pi*r^2 dr
    N = 100000
    r = np.linspace(a * 0.001, 10 * b, N)
    dr = r[1] - r[0]

    E = np.zeros_like(r)
    mask_mid = (r >= a) & (r <= b)
    mask_out = r > b
    E[mask_mid] = k_e * Qa / r[mask_mid]**2
    E[mask_out] = k_e * Q_total / r[mask_out]**2

    W_numerical = 0.5 * epsilon_0 * np.sum(E**2 * 4 * np.pi * r**2 * dr)
    print(f"\n  Numerical integration: {W_numerical:.6e} J")
    print(f"  Relative error: {abs(W_numerical - W_total)/W_total:.4e}")


def exercise_4():
    """
    Exercise 4: Numerical Poisson Solver
    Extend relaxation method to solve Poisson's equation with
    rho(x,y) = rho_0 * exp(-(x^2+y^2)/w^2).
    """
    N = 100
    L = 1.0
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y)

    # Charge distribution: Gaussian blob
    rho_0 = 1e-6  # C/m^3
    w = 0.2        # width (m)
    rho = rho_0 * np.exp(-(X**2 + Y**2) / w**2)

    # Initialize potential
    V = np.zeros((N, N))

    # Poisson's equation: nabla^2 V = -rho/eps_0
    # Discretized: V[i,j] = 0.25*(V[i-1,j]+V[i+1,j]+V[i,j-1]+V[i,j+1] + dx^2*rho/eps_0)
    n_iter = 10000
    source_term = dx**2 * rho / epsilon_0

    for iteration in range(n_iter):
        V_old = V.copy()
        V[1:-1, 1:-1] = 0.25 * (
            V_old[0:-2, 1:-1] + V_old[2:, 1:-1] +
            V_old[1:-1, 0:-2] + V_old[1:-1, 2:] +
            source_term[1:-1, 1:-1]
        )
        # Dirichlet BC: V = 0 on boundaries (already initialized)

    # Compute E = -grad V
    Ey_grid, Ex_grid = np.gradient(-V, dx)

    print(f"  Grid: {N}x{N}, domain [-{L},{L}]^2")
    print(f"  Charge: Gaussian, rho_0 = {rho_0:.1e} C/m^3, width w = {w} m")
    print(f"  Iterations: {n_iter}")
    print(f"  V at center: {V[N//2, N//2]:.4f} V")
    print(f"  V at edge:   {V[N//2, -1]:.4f} V")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cs0 = axes[0].contourf(X, Y, rho * 1e6, levels=30, cmap='Reds')
    plt.colorbar(cs0, ax=axes[0], label=r'$\rho$ ($\mu$C/m$^3$)')
    axes[0].set_title('Charge Distribution')
    axes[0].set_aspect('equal')

    cs1 = axes[1].contourf(X, Y, V, levels=30, cmap='hot')
    plt.colorbar(cs1, ax=axes[1], label='V (V)')
    axes[1].set_title('Potential V(x,y)')
    axes[1].set_aspect('equal')

    E_mag = np.sqrt(Ex_grid**2 + Ey_grid**2)
    step = 5
    axes[2].quiver(X[::step, ::step], Y[::step, ::step],
                   Ex_grid[::step, ::step], Ey_grid[::step, ::step],
                   E_mag[::step, ::step], cmap='viridis')
    axes[2].set_title('Electric Field E = -grad V')
    axes[2].set_aspect('equal')

    plt.suptitle("Poisson's Equation: Gaussian Charge", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex02_poisson_gaussian.png', dpi=150)
    plt.close()
    print("  Plot saved: ex02_poisson_gaussian.png")


def exercise_5():
    """
    Exercise 5: Multipole Expansion
    Compute monopole, dipole, and quadrupole terms for a linear quadrupole:
    +q at (0,0,d), -2q at origin, +q at (0,0,-d).
    The leading far-field term is the quadrupole.
    """
    q = 1e-9     # 1 nC
    d = 0.1      # separation (m)

    charges = [
        (np.array([0.0, 0.0, d]), q),
        (np.array([0.0, 0.0, 0.0]), -2 * q),
        (np.array([0.0, 0.0, -d]), q),
    ]

    # Monopole: Q_total = q + (-2q) + q = 0
    Q_total = sum(qi for _, qi in charges)
    print(f"  Linear quadrupole: +q at (0,0,{d}), -2q at origin, +q at (0,0,-{d})")
    print(f"  q = {q*1e9:.1f} nC, d = {d*100:.0f} cm")
    print()
    print(f"  Monopole moment Q = {Q_total:.2e} C  (zero!)")

    # Dipole: p = sum(q_i * r_i)
    p = np.zeros(3)
    for pos, qi in charges:
        p += qi * pos
    print(f"  Dipole moment p = ({p[0]:.2e}, {p[1]:.2e}, {p[2]:.2e}) C*m  (zero!)")

    # Quadrupole: Q_ij = sum(q_k * (3*r_ki*r_kj - r_k^2 * delta_ij))
    Q_quad = np.zeros((3, 3))
    for pos, qi in charges:
        r2 = np.dot(pos, pos)
        for i in range(3):
            for j in range(3):
                Q_quad[i, j] += qi * (3 * pos[i] * pos[j] - r2 * (1 if i == j else 0))

    print(f"\n  Quadrupole moment tensor Q_ij (C*m^2):")
    for i in range(3):
        print(f"    [{Q_quad[i, 0]:.4e}, {Q_quad[i, 1]:.4e}, {Q_quad[i, 2]:.4e}]")

    Q_zz = Q_quad[2, 2]
    print(f"\n  Q_zz = {Q_zz:.4e} C*m^2 = 2*q*d^2 = {2*q*d**2:.4e} C*m^2")
    print(f"  Leading far-field term: Quadrupole (V ~ 1/r^3)")

    # Verify with exact potential along z-axis
    z_vals = np.linspace(0.5, 5.0, 100)
    V_exact = np.zeros_like(z_vals)
    for pos, qi in charges:
        dist = np.abs(z_vals - pos[2])
        V_exact += k_e * qi / dist

    # Quadrupole approximation on axis (theta=0):
    # V_quad = Q_zz / (4*pi*eps_0 * 4 * r^3) * (3*cos^2(theta) - 1)
    # On axis (theta=0): V_quad = Q_zz / (4*pi*eps_0 * 2 * r^3)
    V_quad = k_e * Q_zz / (2 * z_vals**3)

    print(f"\n  Comparison at r = 5d = {5*d} m:")
    idx = np.argmin(np.abs(z_vals - 5 * d))
    print(f"    Exact V:       {V_exact[idx]:.6e} V")
    print(f"    Quadrupole V:  {V_quad[idx]:.6e} V")
    print(f"    Relative error: {abs(V_exact[idx] - V_quad[idx])/abs(V_exact[idx]):.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_vals / d, np.abs(V_exact), 'b-', linewidth=2, label='Exact')
    ax.plot(z_vals / d, np.abs(V_quad), 'r--', linewidth=1.5, label='Quadrupole approx')
    ax.set_xlabel('r / d')
    ax.set_ylabel('|V| (V)')
    ax.set_title('Linear Quadrupole Potential on Axis')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex02_multipole_quadrupole.png', dpi=150)
    plt.close()
    print("  Plot saved: ex02_multipole_quadrupole.png")


if __name__ == "__main__":
    print("=== Exercise 1: Potential and Field Computation ===")
    exercise_1()
    print("\n=== Exercise 2: Laplace Solver with Complex Boundaries ===")
    exercise_2()
    print("\n=== Exercise 3: Energy of Concentric Shells ===")
    exercise_3()
    print("\n=== Exercise 4: Numerical Poisson Solver ===")
    exercise_4()
    print("\n=== Exercise 5: Multipole Expansion ===")
    exercise_5()
    print("\nAll exercises completed!")
