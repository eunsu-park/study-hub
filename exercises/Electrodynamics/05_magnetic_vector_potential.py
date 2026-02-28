"""
Exercises for Lesson 05: Magnetic Vector Potential
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Constants
mu_0 = 4.0 * np.pi * 1e-7
hbar = 1.055e-34
e_charge = 1.6e-19


def exercise_1():
    """
    Exercise 1: Vector Potential of a Finite Solenoid
    Compute A for a solenoid of finite length L and radius R.
    Compare with infinite solenoid formula.
    """
    R = 0.05     # solenoid radius
    L = 0.4      # solenoid length
    N = 200      # number of turns
    I = 1.0      # current
    n = N / L    # turns per unit length

    # Infinite solenoid: A_phi = mu_0*n*I*s/2 inside, mu_0*n*I*R^2/(2s) outside
    s_vals = np.linspace(0.001, 0.15, 200)

    A_infinite = np.where(
        s_vals < R,
        mu_0 * n * I * s_vals / 2,
        mu_0 * n * I * R**2 / (2 * s_vals)
    )

    # Finite solenoid: sum contributions from N loops
    # For a single loop at z0, A_phi at (s, z) involves elliptic integrals.
    # Simplified: compute on the midplane (z=0) using numerical integration.
    z_loops = np.linspace(-L / 2, L / 2, N)

    A_finite = np.zeros_like(s_vals)
    for z0 in z_loops:
        # Each loop contributes A_phi at distance s on the midplane (z=0)
        # Using the far-field approximation for A_phi from a magnetic dipole:
        # For a single loop: m = I*pi*R^2
        # Near-axis: A_phi ~ mu_0*I*R^2*s / (4*(R^2+z0^2)^(3/2))
        # This is approximate but captures the finite-length effects.
        for i, s in enumerate(s_vals):
            # Full numerical: compute line integral of I*dl'/|r-r'|
            N_seg = 200
            phi_seg = np.linspace(0, 2 * np.pi, N_seg, endpoint=False)
            dphi = 2 * np.pi / N_seg

            # Loop at z0, radius R
            x_loop = R * np.cos(phi_seg)
            y_loop = R * np.sin(phi_seg)
            z_loop = z0

            # Field point at (s, 0, 0) on midplane
            dx = s - x_loop
            dy = 0 - y_loop
            dz = 0 - z_loop
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            # dl' = R*dphi*(-sin(phi), cos(phi), 0)
            # A = (mu_0*I)/(4*pi) * integral of dl'/|r-r'|
            # A_phi at (s,0,0) = A_y component (since phi=0 -> phi_hat = y_hat)
            dl_y = R * np.cos(phi_seg) * dphi
            A_finite[i] += mu_0 * I / (4 * np.pi) * np.sum(dl_y / dist)

    print(f"  Finite solenoid: R = {R*100:.0f} cm, L = {L*100:.0f} cm, N = {N}")
    print(f"  Comparing A_phi at midplane (z=0)")
    idx_in = np.argmin(np.abs(s_vals - R / 2))
    idx_out = np.argmin(np.abs(s_vals - 2 * R))
    print(f"  At s = R/2 = {R/2*100:.1f} cm:")
    print(f"    Finite:   {A_finite[idx_in]*1e6:.4f} uT*m")
    print(f"    Infinite: {A_infinite[idx_in]*1e6:.4f} uT*m")
    print(f"  At s = 2R = {2*R*100:.0f} cm:")
    print(f"    Finite:   {A_finite[idx_out]*1e6:.4f} uT*m")
    print(f"    Infinite: {A_infinite[idx_out]*1e6:.4f} uT*m")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_vals * 100, A_finite * 1e6, 'b-', linewidth=2, label='Finite solenoid')
    ax.plot(s_vals * 100, A_infinite * 1e6, 'r--', linewidth=1.5, label='Infinite solenoid')
    ax.axvline(x=R * 100, color='gray', linestyle=':', alpha=0.5, label=f'R = {R*100:.0f} cm')
    ax.set_xlabel('s (cm)')
    ax.set_ylabel(r'$A_\phi$ ($\mu$T$\cdot$m)')
    ax.set_title('Vector Potential: Finite vs Infinite Solenoid')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex05_finite_solenoid_A.png', dpi=150)
    plt.close()
    print("  Plot saved: ex05_finite_solenoid_A.png")


def exercise_2():
    """
    Exercise 2: Verify Coulomb Gauge
    For a circular current loop, verify div(A) = 0 at several points.
    """
    R = 0.1   # loop radius
    I = 1.0   # current

    # Compute A numerically at several test points using Biot-Savart-like formula
    N_seg = 500
    phi_seg = np.linspace(0, 2 * np.pi, N_seg, endpoint=False)
    dphi = 2 * np.pi / N_seg

    def compute_A(point):
        """Compute vector potential A at a given 3D point from a loop at z=0."""
        Ax, Ay, Az = 0.0, 0.0, 0.0
        for p in phi_seg:
            xl = R * np.cos(p)
            yl = R * np.sin(p)
            dist = np.sqrt((point[0] - xl)**2 + (point[1] - yl)**2 + point[2]**2)
            dist = max(dist, 1e-10)
            dlx = -R * np.sin(p) * dphi
            dly = R * np.cos(p) * dphi
            Ax += mu_0 * I / (4 * np.pi) * dlx / dist
            Ay += mu_0 * I / (4 * np.pi) * dly / dist
        return np.array([Ax, Ay, Az])

    # Test points
    test_points = [
        np.array([0.05, 0.0, 0.1]),
        np.array([0.15, 0.1, 0.05]),
        np.array([0.0, 0.2, 0.0]),
        np.array([0.08, 0.08, 0.15]),
    ]

    print("  Verifying Coulomb gauge: div(A) = 0 at test points")
    print(f"  Current loop: R = {R*100:.0f} cm, I = {I} A")
    print()

    delta = 1e-6
    for pt in test_points:
        # Numerical divergence: div(A) = dAx/dx + dAy/dy + dAz/dz
        div_A = 0.0
        for axis in range(3):
            pt_plus = pt.copy()
            pt_plus[axis] += delta
            pt_minus = pt.copy()
            pt_minus[axis] -= delta
            A_plus = compute_A(pt_plus)
            A_minus = compute_A(pt_minus)
            div_A += (A_plus[axis] - A_minus[axis]) / (2 * delta)

        A_val = compute_A(pt)
        print(f"  Point ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}):")
        print(f"    |A| = {np.linalg.norm(A_val)*1e6:.4f} uT*m")
        print(f"    div(A) = {div_A:.4e} (should be ~0)")


def exercise_3():
    """
    Exercise 3: Gauge Transformation
    Find lambda that transforms symmetric gauge to Landau gauge.
    Symmetric: A1 = (B0/2)(-y, x, 0)
    Landau: A2 = (0, B0*x, 0)
    lambda = B0*x*y/2
    """
    B0 = 1e-3  # 1 mT

    print("  Symmetric gauge: A1 = (B0/2)(-y, x, 0)")
    print("  Landau gauge:    A2 = (0, B0*x, 0)")
    print()
    print("  A2 - A1 = (B0*y/2, B0*x/2, 0)")
    print("  This must equal grad(lambda)")
    print("  So: d(lambda)/dx = B0*y/2, d(lambda)/dy = B0*x/2")
    print("  Solution: lambda = B0*x*y/2")
    print()

    # Verify numerically
    N = 20
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]

    # lambda = B0*x*y/2
    lam = B0 * X * Y / 2

    # grad(lambda) = (d(lambda)/dx, d(lambda)/dy)
    grad_lam_x = np.gradient(lam, dx, axis=1)
    grad_lam_y = np.gradient(lam, dx, axis=0)

    # A1
    A1x = -B0 * Y / 2
    A1y = B0 * X / 2

    # A2 = A1 + grad(lambda)
    A2x_computed = A1x + grad_lam_x
    A2y_computed = A1y + grad_lam_y

    # Expected A2
    A2x_expected = np.zeros_like(X)
    A2y_expected = B0 * X

    error_x = np.max(np.abs(A2x_computed - A2x_expected))
    error_y = np.max(np.abs(A2y_computed - A2y_expected))

    print(f"  Verification (B0 = {B0*1e3:.1f} mT):")
    print(f"  Max error in A2_x: {error_x:.4e} (should be ~0)")
    print(f"  Max error in A2_y: {error_y:.4e} (should be ~0)")

    # Both gauges give same B
    Bz_1 = np.gradient(A1y, dx, axis=1) - np.gradient(A1x, dx, axis=0)
    Bz_2 = np.gradient(A2y_expected, dx, axis=1) - np.gradient(A2x_expected, dx, axis=0)
    print(f"  B_z from gauge 1: {np.mean(Bz_1)*1e3:.4f} mT")
    print(f"  B_z from gauge 2: {np.mean(Bz_2)*1e3:.4f} mT")


def exercise_4():
    """
    Exercise 4: Magnetic Quadrupole
    Two circular loops with opposite currents (magnetic quadrupole).
    Show A falls off faster than dipole.
    """
    R = 0.1    # loop radius
    I = 1.0    # current magnitude
    d = 0.05   # half-separation between loops

    # Loop 1 at z = +d with current +I (m1 = I*pi*R^2 z_hat)
    # Loop 2 at z = -d with current -I (m2 = -I*pi*R^2 z_hat)
    # Net dipole moment = 0 -> quadrupole dominates

    N_seg = 500
    phi_seg = np.linspace(0, 2 * np.pi, N_seg, endpoint=False)
    dphi = 2 * np.pi / N_seg

    def compute_A_y(s, z, z0, I_loop):
        """Compute A_phi (= A_y at phi=0) at (s, 0, z) from loop at z0."""
        Ay = 0.0
        for p in phi_seg:
            xl = R * np.cos(p)
            yl = R * np.sin(p)
            dist = np.sqrt((s - xl)**2 + yl**2 + (z - z0)**2)
            dist = max(dist, 1e-10)
            dly = R * np.cos(p) * dphi
            Ay += mu_0 * I_loop / (4 * np.pi) * dly / dist
        return Ay

    # Compute |A| along the equatorial plane (z=0) as function of distance s
    s_vals = np.linspace(0.3, 3.0, 50)
    A_quadrupole = np.zeros_like(s_vals)
    A_single_dipole = np.zeros_like(s_vals)

    for i, s in enumerate(s_vals):
        # Quadrupole: two loops with opposite currents
        A_quadrupole[i] = abs(compute_A_y(s, 0, d, I) + compute_A_y(s, 0, -d, -I))
        # Single dipole for comparison
        A_single_dipole[i] = abs(compute_A_y(s, 0, 0, I))

    # Fit power laws
    log_s = np.log(s_vals)
    log_Aq = np.log(A_quadrupole + 1e-30)
    log_Ad = np.log(A_single_dipole + 1e-30)

    mask = A_quadrupole > 0
    coeffs_q = np.polyfit(log_s[mask], log_Aq[mask], 1)
    coeffs_d = np.polyfit(log_s, log_Ad, 1)

    print(f"  Two loops at z = +/-{d*100:.0f} cm, R = {R*100:.0f} cm, opposite currents")
    print(f"  Net dipole moment = 0 (quadrupole)")
    print(f"\n  Power law exponents:")
    print(f"    Dipole A ~ s^{coeffs_d[0]:.2f} (expected: -2)")
    print(f"    Quadrupole A ~ s^{coeffs_q[0]:.2f} (expected: -3)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(s_vals, A_single_dipole, 'b-', linewidth=2, label=f'Dipole (~s^{coeffs_d[0]:.1f})')
    ax.loglog(s_vals, A_quadrupole, 'r-', linewidth=2, label=f'Quadrupole (~s^{coeffs_q[0]:.1f})')
    ax.set_xlabel('s (m)')
    ax.set_ylabel('|A_phi| (T*m)')
    ax.set_title('Vector Potential: Dipole vs Quadrupole')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex05_quadrupole_A.png', dpi=150)
    plt.close()
    print("  Plot saved: ex05_quadrupole_A.png")


def exercise_5():
    """
    Exercise 5: AB Effect Flux Quantization
    Calculate the flux quantum Phi_0 = h/(2e).
    For a superconducting ring of area 1 mm^2, find max B for one flux quantum.
    """
    h = 6.626e-34   # Planck constant
    e = 1.6e-19      # electron charge

    Phi_0 = h / (2 * e)   # flux quantum for Cooper pairs

    print(f"  Planck constant h = {h:.3e} J*s")
    print(f"  Electron charge e = {e:.3e} C")
    print()
    print(f"  Flux quantum Phi_0 = h/(2e) = {Phi_0:.6e} Wb")
    print(f"  (Literature value: 2.0678 x 10^-15 Wb)")
    print()

    # Superconducting ring
    A_ring = 1e-6    # 1 mm^2 = 1e-6 m^2
    B_max = Phi_0 / A_ring

    print(f"  Superconducting ring area: A = 1 mm^2 = {A_ring:.1e} m^2")
    print(f"  Maximum B for single flux quantum:")
    print(f"    B = Phi_0/A = {B_max:.6e} T = {B_max*1e9:.4f} nT")
    print()
    print("  This extremely small field (~2 nT) is what makes SQUIDs")
    print("  the most sensitive magnetometers, capable of detecting")
    print("  fields far weaker than Earth's magnetic field (~50 uT).")


def exercise_6():
    """
    Exercise 6: Vector Potential of a Current Sheet
    Infinite current sheet in xy-plane with K = K0 x_hat.
    A = -(mu_0*K0/2)|z| x_hat. Verify curl(A) gives correct B.
    """
    K0 = 1.0  # surface current density (A/m)

    z = np.linspace(-0.5, 0.5, 1000)
    dz = z[1] - z[0]

    # Vector potential: A_x = -(mu_0*K0/2)|z|, Ay = Az = 0
    A_x = -mu_0 * K0 / 2 * np.abs(z)

    # B = curl(A)
    # Since A = Ax(z) x_hat only, curl(A) = -dAx/dz y_hat
    B_y = -np.gradient(A_x, dz)

    # Expected: B = +(mu_0*K0/2) y_hat for z < 0, -(mu_0*K0/2) y_hat for z > 0
    B_expected = np.where(z > 0, -mu_0 * K0 / 2, mu_0 * K0 / 2)

    print(f"  Current sheet: K = {K0} A/m in x-direction")
    print(f"  A = -(mu_0*K0/2)|z| x_hat")
    print()
    print(f"  Expected B (z > 0): -(mu_0*K0/2) = {-mu_0*K0/2*1e6:.4f} uT in y-direction")
    print(f"  Expected B (z < 0): +(mu_0*K0/2) = {mu_0*K0/2*1e6:.4f} uT in y-direction")
    print()

    idx_above = np.argmin(np.abs(z - 0.2))
    idx_below = np.argmin(np.abs(z + 0.2))
    print(f"  Computed B_y at z = +0.2 m: {B_y[idx_above]*1e6:.4f} uT")
    print(f"  Computed B_y at z = -0.2 m: {B_y[idx_below]*1e6:.4f} uT")
    print(f"  Discontinuity at z=0: Delta B_y = {(B_y[idx_below]-B_y[idx_above])*1e6:.4f} uT")
    print(f"  Expected: mu_0*K0 = {mu_0*K0*1e6:.4f} uT")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(z * 100, A_x * 1e6, 'b-', linewidth=2)
    axes[0].set_xlabel('z (cm)')
    axes[0].set_ylabel(r'$A_x$ ($\mu$T$\cdot$m)')
    axes[0].set_title('Vector Potential of Current Sheet')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(z * 100, B_y * 1e6, 'r-', linewidth=2, label='Numerical curl(A)')
    axes[1].plot(z * 100, B_expected * 1e6, 'b--', linewidth=1.5, label='Expected')
    axes[1].set_xlabel('z (cm)')
    axes[1].set_ylabel(r'$B_y$ ($\mu$T)')
    axes[1].set_title('Magnetic Field B = curl(A)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Current Sheet: A and B', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex05_current_sheet.png', dpi=150)
    plt.close()
    print("  Plot saved: ex05_current_sheet.png")


if __name__ == "__main__":
    print("=== Exercise 1: Vector Potential of Finite Solenoid ===")
    exercise_1()
    print("\n=== Exercise 2: Verify Coulomb Gauge ===")
    exercise_2()
    print("\n=== Exercise 3: Gauge Transformation ===")
    exercise_3()
    print("\n=== Exercise 4: Magnetic Quadrupole ===")
    exercise_4()
    print("\n=== Exercise 5: AB Effect Flux Quantization ===")
    exercise_5()
    print("\n=== Exercise 6: Current Sheet Vector Potential ===")
    exercise_6()
    print("\nAll exercises completed!")
