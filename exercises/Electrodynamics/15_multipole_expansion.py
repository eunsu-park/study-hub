"""
Exercises for Lesson 15: Multipole Expansion
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import legendre


# Constants
k_e = 8.988e9          # Coulomb constant (N*m^2/C^2)
epsilon_0 = 8.854e-12
G = 6.674e-11          # gravitational constant
M_earth = 5.972e24     # kg
R_earth = 6.371e6      # m


def exercise_1():
    """
    Exercise 1: Quadrupole Potential Map
    Linear quadrupole: +q, -2q, +q along z-axis with spacing d.
    Compare exact vs quadrupole approximation at various distances.
    """
    q = 1e-9   # 1 nC
    d = 0.01   # 1 cm spacing

    # --- Exact potential from 3 point charges ---
    # Charges at z = -d, 0, +d with charges +q, -2q, +q
    def V_exact(x, z):
        """Exact potential from superposition of 3 point charges."""
        r1 = np.sqrt(x**2 + (z - d)**2)  # +q at (0, d)
        r2 = np.sqrt(x**2 + z**2)         # -2q at (0, 0)
        r3 = np.sqrt(x**2 + (z + d)**2)  # +q at (0, -d)
        # Avoid singularities
        r1 = np.maximum(r1, 1e-15)
        r2 = np.maximum(r2, 1e-15)
        r3 = np.maximum(r3, 1e-15)
        return k_e * (q / r1 - 2 * q / r2 + q / r3)

    def V_quadrupole(x, z):
        """Quadrupole approximation: V ~ (Q_zz / 4) * (3*cos^2(theta)-1) / r^3."""
        r = np.sqrt(x**2 + z**2)
        r = np.maximum(r, 1e-15)
        cos_theta = z / r
        # Quadrupole moment: Q_zz = 2*q*d^2 (for charges +q at +/-d, -2q at 0)
        Q_zz = 2 * q * d**2
        return k_e * Q_zz * (3 * cos_theta**2 - 1) / (4 * r**3)

    # 2D contour plot
    N = 300
    x_max = 15 * d
    x_vals = np.linspace(-x_max, x_max, N)
    z_vals = np.linspace(-x_max, x_max, N)
    X, Z = np.meshgrid(x_vals, z_vals)

    V_ex = V_exact(X, Z)
    V_qp = V_quadrupole(X, Z)

    # Clip for visualization
    V_max = 500
    V_ex_clip = np.clip(V_ex, -V_max, V_max)
    V_qp_clip = np.clip(V_qp, -V_max, V_max)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    levels = np.linspace(-V_max, V_max, 30)
    c1 = axes[0].contourf(X / d, Z / d, V_ex_clip, levels=levels, cmap='RdBu_r')
    axes[0].set_title('Exact Potential')
    axes[0].set_xlabel('x / d')
    axes[0].set_ylabel('z / d')
    plt.colorbar(c1, ax=axes[0], label='V (V)')

    c2 = axes[1].contourf(X / d, Z / d, V_qp_clip, levels=levels, cmap='RdBu_r')
    axes[1].set_title('Quadrupole Approximation')
    axes[1].set_xlabel('x / d')
    axes[1].set_ylabel('z / d')
    plt.colorbar(c2, ax=axes[1], label='V (V)')

    plt.suptitle('Linear Quadrupole: Exact vs Approximation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex15_quadrupole_potential.png', dpi=150)
    plt.close()

    # Compare accuracy at r = 2d, 5d, 10d along theta = 45 degrees
    print("  Linear quadrupole: +q, -2q, +q with spacing d")
    print(f"  q = {q*1e9:.1f} nC, d = {d*100:.0f} cm")
    print(f"  Quadrupole moment: Q_zz = 2*q*d^2 = {2*q*d**2:.4e} C*m^2")
    print()
    print("  Accuracy comparison along theta = 45 deg:")
    print(f"  {'r/d':>6s}  {'V_exact':>12s}  {'V_quad':>12s}  {'Error(%)':>10s}")

    theta_test = np.radians(45)
    r_over_d_vals = [2, 3, 5, 7, 10, 15, 20]
    r_5pct = None
    for r_d in r_over_d_vals:
        r = r_d * d
        x_t = r * np.sin(theta_test)
        z_t = r * np.cos(theta_test)
        V_e = V_exact(x_t, z_t)
        V_q = V_quadrupole(x_t, z_t)
        if V_e != 0:
            error = abs((V_q - V_e) / V_e) * 100
        else:
            error = 0
        print(f"  {r_d:6d}  {V_e:12.4e}  {V_q:12.4e}  {error:10.2f}")
        if error < 5 and r_5pct is None:
            r_5pct = r_d

    if r_5pct is not None:
        print(f"\n  Quadrupole approximation accurate to <5% for r >= {r_5pct}*d")
    else:
        print(f"\n  Quadrupole approximation not yet within 5% in tested range")

    print("  Plot saved: ex15_quadrupole_potential.png")


def exercise_2():
    """
    Exercise 2: Multipole Moments of a Ring
    Uniformly charged ring of radius R, total charge Q in the xy-plane.
    Compute q_{l0} up to l=6, show only even l contribute,
    compare exact on-axis potential with truncated multipole series.
    """
    Q = 1e-9   # 1 nC
    R = 0.05   # 5 cm ring radius

    # (a) Multipole moments q_{l0}
    # For a ring in the xy-plane (theta = pi/2), the charge density is
    # lambda = Q / (2*pi*R), and the multipole moments are:
    # q_{l0} = Q * R^l * P_l(0) (only m=0 by azimuthal symmetry)
    # P_l(0) = 0 for odd l, so only even l contribute.

    print(f"  Uniformly charged ring: Q = {Q*1e9:.1f} nC, R = {R*100:.0f} cm")
    print()
    print("  (a) Multipole moments q_{l,0} = Q * R^l * P_l(cos(pi/2)) = Q * R^l * P_l(0):")
    print(f"  {'l':>4s}  {'P_l(0)':>10s}  {'q_{l0}':>14s}")

    q_l0 = {}
    for l in range(7):
        P_l = legendre(l)
        P_l_0 = P_l(0.0)
        qlm = Q * R**l * P_l_0
        q_l0[l] = qlm
        print(f"  {l:4d}  {P_l_0:10.6f}  {qlm:14.6e}")

    # (b) Only even l contribute
    print()
    print("  (b) P_l(0) = 0 for all odd l. Therefore q_{l0} = 0 for odd l.")
    print("      Only even multipole moments (l = 0, 2, 4, 6, ...) are non-zero.")
    print("      This is because the ring is symmetric under z -> -z reflection.")

    # (c) Compare exact on-axis potential with multipole series truncated at l=4
    # Exact on axis (at point (0, 0, z)):
    # V_exact = k_e * Q / sqrt(R^2 + z^2)
    #
    # Multipole expansion on axis (theta = 0, P_l(1) = 1):
    # V_multi = k_e * sum_l q_{l0} / r^{l+1}  (for r > R)

    z_vals = np.linspace(1.5 * R, 10 * R, 200)
    V_exact = k_e * Q / np.sqrt(R**2 + z_vals**2)

    V_multi_l2 = np.zeros_like(z_vals)
    V_multi_l4 = np.zeros_like(z_vals)
    V_multi_l6 = np.zeros_like(z_vals)

    for l in range(7):
        if q_l0[l] == 0:
            continue
        term = k_e * q_l0[l] / z_vals**(l + 1)
        if l <= 2:
            V_multi_l2 += term
        if l <= 4:
            V_multi_l4 += term
        if l <= 6:
            V_multi_l6 += term

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(z_vals / R, V_exact, 'k-', linewidth=2, label='Exact')
    axes[0].plot(z_vals / R, V_multi_l2, 'b--', linewidth=1.5, label='l <= 2')
    axes[0].plot(z_vals / R, V_multi_l4, 'r-.', linewidth=1.5, label='l <= 4')
    axes[0].plot(z_vals / R, V_multi_l6, 'g:', linewidth=1.5, label='l <= 6')
    axes[0].set_xlabel('z / R')
    axes[0].set_ylabel('V (V)')
    axes[0].set_title('On-Axis Potential')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Relative error
    err_l4 = np.abs((V_multi_l4 - V_exact) / V_exact) * 100
    err_l6 = np.abs((V_multi_l6 - V_exact) / V_exact) * 100

    axes[1].semilogy(z_vals / R, err_l4, 'r-', linewidth=2, label='l <= 4')
    axes[1].semilogy(z_vals / R, err_l6, 'g-', linewidth=2, label='l <= 6')
    axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='1% error')
    axes[1].set_xlabel('z / R')
    axes[1].set_ylabel('Relative Error (%)')
    axes[1].set_title('Multipole Series Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Ring Charge: Multipole Expansion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex15_ring_multipole.png', dpi=150)
    plt.close()

    print()
    print("  (c) Comparison at z = 2R:")
    idx_2R = np.argmin(np.abs(z_vals - 2 * R))
    print(f"    V_exact  = {V_exact[idx_2R]:.6e} V")
    print(f"    V(l<=4)  = {V_multi_l4[idx_2R]:.6e} V (error: {err_l4[idx_2R]:.2f}%)")
    print(f"    V(l<=6)  = {V_multi_l6[idx_2R]:.6e} V (error: {err_l6[idx_2R]:.2f}%)")
    print("  Plot saved: ex15_ring_multipole.png")


def exercise_3():
    """
    Exercise 3: Magnetic Dipole Radiation
    Hydrogen 2s -> 1s transition: why E1 is forbidden, M1 rate estimate.
    """
    # E1 selection rules: delta_l = +/- 1, parity change
    # 2s: n=2, l=0;  1s: n=1, l=0
    # delta_l = 0, violates E1 selection rule

    # M1 transition rate / E1 rate
    # E1 (2p -> 1s): lifetime ~ 1.6 ns -> Gamma_E1 ~ 6.25e8 s^-1
    # M1 (2s -> 1s): suppressed by factor ~ (v/c)^2 ~ (alpha)^2 ~ (1/137)^2
    # But 2s -> 1s M1 is also forbidden for single-photon emission due to
    # the radial matrix element vanishing in non-relativistic limit.
    # Actual 2s -> 1s decay is via 2-photon emission: Gamma ~ 8.23 s^-1

    alpha = 1 / 137.036  # fine structure constant
    tau_2p_1s = 1.596e-9  # 2p -> 1s E1 lifetime (s)
    Gamma_E1 = 1 / tau_2p_1s

    # M1 rate estimate (order of magnitude)
    # M1 / E1 ~ (k*a_0)^2 ~ (omega/c * a_0)^2 ~ alpha^2
    Gamma_M1_estimate = Gamma_E1 * alpha**2

    # Actual 2-photon rate
    Gamma_2photon = 8.229  # s^-1

    print("  Hydrogen 2s -> 1s transition:")
    print()
    print("  (a) Why E1 radiation is forbidden:")
    print("      E1 selection rule requires delta_l = +/- 1 and parity change.")
    print("      2s state: n=2, l=0, parity = (-1)^0 = +1")
    print("      1s state: n=1, l=0, parity = (-1)^0 = +1")
    print("      delta_l = 0 (violates delta_l = +/-1)")
    print("      Same parity (violates parity change requirement)")
    print("      Therefore E1 (electric dipole) radiation is strictly forbidden.")
    print()
    print("  (b) M1 transition rate estimate:")
    print(f"      E1 rate (2p -> 1s): Gamma_E1 = {Gamma_E1:.3e} s^-1")
    print(f"      M1 suppression factor: (k*a0)^2 ~ alpha^2 = {alpha**2:.4e}")
    print(f"      Naive M1 rate estimate: Gamma_M1 ~ {Gamma_M1_estimate:.3e} s^-1")
    print(f"      Ratio M1/E1 ~ {alpha**2:.2e}")
    print()
    print("  (c) Actual 2s -> 1s decay mechanism:")
    print("      Single-photon M1 transition is also forbidden in non-relativistic")
    print("      hydrogen because the radial overlap integral vanishes due to")
    print("      orthogonality of the 1s and 2s wavefunctions.")
    print()
    print(f"      The dominant decay is TWO-PHOTON emission (2E1 process):")
    print(f"      Gamma_2gamma = {Gamma_2photon:.3f} s^-1")
    print(f"      Lifetime = {1/Gamma_2photon:.3f} s = {1/Gamma_2photon*1e3:.1f} ms")
    print(f"      This is ~ {Gamma_E1/Gamma_2photon:.2e} times slower than 2p -> 1s")
    print()
    print("      The 2s state is called 'metastable' because of this extremely")
    print("      long lifetime compared to typical atomic transitions (~ns).")


def exercise_4():
    """
    Exercise 4: E2 Radiation Pattern
    Electric quadrupole radiation pattern: dP/dOmega ~ sin^2(theta)*cos^2(theta).
    Compare with E1 pattern.
    """
    theta = np.linspace(0, np.pi, 500)

    # E1 (electric dipole) pattern: ~ sin^2(theta)
    P_E1 = np.sin(theta)**2

    # E2 (electric quadrupole, m=0) pattern: ~ sin^2(theta)*cos^2(theta)
    P_E2 = np.sin(theta)**2 * np.cos(theta)**2

    # Normalize both to max = 1
    P_E1 /= np.max(P_E1)
    P_E2 /= np.max(P_E2)

    # Identify null directions for E2
    # dP/dOmega = 0 when sin(theta) = 0 (theta = 0, pi) or cos(theta) = 0 (theta = pi/2)
    print("  E2 (electric quadrupole) radiation pattern: dP/dOmega ~ sin^2(theta)*cos^2(theta)")
    print()
    print("  Null directions (zero radiation):")
    print("    theta = 0 (along z-axis) -- same as E1")
    print("    theta = pi (along -z-axis) -- same as E1")
    print("    theta = pi/2 (equatorial plane) -- UNIQUE to E2, E1 has maximum here")
    print()
    print("  Maximum radiation at theta = 45 deg and 135 deg (four-lobed pattern)")
    print(f"  E2 peak angle: theta = {np.degrees(np.arctan(1)):.1f} degrees")

    # 2D polar plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': 'polar'})

    # E1 pattern
    axes[0].plot(theta, P_E1, 'b-', linewidth=2, label='E1')
    axes[0].plot(theta + np.pi, P_E1, 'b-', linewidth=2)
    axes[0].set_title('E1 (Electric Dipole)', pad=20, fontsize=12)
    axes[0].legend(loc='lower right')

    # E2 pattern
    axes[1].plot(theta, P_E2, 'r-', linewidth=2, label='E2')
    axes[1].plot(theta + np.pi, P_E2, 'r-', linewidth=2)
    axes[1].set_title('E2 (Electric Quadrupole)', pad=20, fontsize=12)
    axes[1].legend(loc='lower right')

    plt.suptitle('Multipole Radiation Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex15_e2_radiation_pattern.png', dpi=150)
    plt.close()

    # Also do a Cartesian plot for clarity
    fig, ax = plt.subplots(figsize=(8, 5))
    theta_deg = np.degrees(theta)
    ax.plot(theta_deg, P_E1, 'b-', linewidth=2, label='E1: sin^2(theta)')
    ax.plot(theta_deg, P_E2, 'r-', linewidth=2, label='E2: sin^2(theta)*cos^2(theta)')
    ax.axvline(x=45, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=90, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=135, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Polar angle theta (degrees)')
    ax.set_ylabel('Normalized power dP/dOmega')
    ax.set_title('E1 vs E2 Radiation Patterns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex15_e1_vs_e2.png', dpi=150)
    plt.close()

    # Total power comparison
    # Integrate sin^2*cos^2 * sin(theta) dtheta dphi over full sphere
    # int_0^pi sin^3(theta)*cos^2(theta) dtheta = 4/15
    # int_0^pi sin^3(theta) dtheta = 4/3 (for E1)
    ratio = (4 / 15) / (4 / 3)
    print(f"\n  Total power ratio: P_E2/P_E1 (same amplitude) = {ratio:.4f}")
    print(f"  E2 radiation is ~ {1/ratio:.1f}x less isotropically efficient than E1")
    print("  Plots saved: ex15_e2_radiation_pattern.png, ex15_e1_vs_e2.png")


def exercise_5():
    """
    Exercise 5: Earth's Gravitational Multipoles
    J2 = 1.0826e-3, J4 = -1.62e-6.
    Potential at equator/pole, geoid height difference, orbital precession.
    """
    J2 = 1.0826e-3
    J4 = -1.62e-6

    # Gravitational potential with zonal harmonics:
    # V(r, theta) = -(GM/r) * [1 - sum_l J_l * (R_e/r)^l * P_l(cos(theta))]
    # Note: J_l are defined with a minus sign convention:
    # V = -(GM/r)[1 - J2*(R/r)^2*P2(cos_theta) - J4*(R/r)^4*P4(cos_theta)]

    P2 = legendre(2)
    P4 = legendre(4)

    # (a) Potential at surface (r = R_earth)
    # At equator: theta = pi/2, cos(theta) = 0
    # At pole: theta = 0, cos(theta) = 1
    cos_equator = 0.0
    cos_pole = 1.0

    P2_eq = P2(cos_equator)   # P2(0) = -1/2
    P4_eq = P4(cos_equator)   # P4(0) = 3/8
    P2_pole = P2(cos_pole)    # P2(1) = 1
    P4_pole = P4(cos_pole)    # P4(1) = 1

    V0 = -G * M_earth / R_earth  # monopole term

    V_eq = V0 * (1 - J2 * P2_eq - J4 * P4_eq)
    V_pole = V0 * (1 - J2 * P2_pole - J4 * P4_pole)

    print(f"  Earth's gravitational multipoles:")
    print(f"  J2 = {J2:.4e}, J4 = {J4:.2e}")
    print(f"  GM/R_e = {abs(V0):.4e} J/kg")
    print()
    print(f"  (a) Gravitational potential at the surface:")
    print(f"      P2(0) = {P2_eq:.4f}, P4(0) = {P4_eq:.4f}")
    print(f"      P2(1) = {P2_pole:.4f}, P4(1) = {P4_pole:.4f}")
    print(f"      V(equator) = {V_eq:.6e} J/kg")
    print(f"      V(pole)    = {V_pole:.6e} J/kg")
    print(f"      Delta V    = {V_eq - V_pole:.6e} J/kg")

    # (b) Geoid height difference
    # The geoid is an equipotential surface. The height difference is:
    # Delta h = Delta V / g, where g ~ GM/R^2
    g_surface = G * M_earth / R_earth**2
    delta_h = (V_eq - V_pole) / g_surface

    print(f"\n  (b) Geoid height difference (equator - pole):")
    print(f"      g_surface = {g_surface:.4f} m/s^2")
    print(f"      Delta h = Delta V / g = {delta_h:.1f} m")
    print(f"      The equator is ~ {abs(delta_h)/1000:.1f} km further from Earth's center")
    print(f"      (mostly due to J2 oblateness from rotation)")

    # (c) Orbital precession from J2
    # Rate of nodal (RAAN) precession:
    # Omega_dot = -3/2 * n * J2 * (R_e/a)^2 * cos(i) / (1-e^2)^2
    # For circular LEO at 400 km altitude, i = 51.6 deg (ISS-like)
    h_orbit = 400e3  # 400 km altitude
    a = R_earth + h_orbit
    e = 0.0  # circular
    i_deg = 51.6
    i_rad = np.radians(i_deg)

    # Mean motion n = sqrt(GM/a^3)
    n = np.sqrt(G * M_earth / a**3)
    T_orbit = 2 * np.pi / n

    # RAAN precession rate
    Omega_dot = -1.5 * n * J2 * (R_earth / a)**2 * np.cos(i_rad)
    Omega_dot_deg_day = np.degrees(Omega_dot) * 86400

    # Argument of perigee precession:
    # omega_dot = 3/2 * n * J2 * (R/a)^2 * (2 - 5/2 * sin^2(i)) / (1-e^2)^2
    omega_dot = 1.5 * n * J2 * (R_earth / a)**2 * (2 - 2.5 * np.sin(i_rad)**2)
    omega_dot_deg_day = np.degrees(omega_dot) * 86400

    print(f"\n  (c) J2 orbital precession (LEO at {h_orbit/1e3:.0f} km, i = {i_deg} deg):")
    print(f"      Semi-major axis: a = {a/1e3:.1f} km")
    print(f"      Orbital period: T = {T_orbit/60:.1f} min")
    print(f"      Mean motion: n = {n*1e3:.4f} x10^-3 rad/s")
    print()
    print(f"      RAAN precession rate: {Omega_dot_deg_day:.3f} deg/day")
    print(f"        = {Omega_dot_deg_day*365.25:.1f} deg/year")
    print(f"      (Node regresses westward for prograde orbits)")
    print()
    print(f"      Perigee precession rate: {omega_dot_deg_day:.3f} deg/day")
    print(f"        = {omega_dot_deg_day*365.25:.1f} deg/year")
    print()

    # Sun-synchronous orbit: Omega_dot = 360/365.25 deg/day = 0.9856 deg/day
    # Requires cos(i) = -0.9856 / (1.5 * n * J2 * (R/a)^2 * deg2rad * 86400)
    target_rate = 0.9856  # deg/day for sun-synchronous
    # cos(i) = target / (-1.5 * n * J2 * (R/a)^2 * 86400 * 180/pi)
    cos_i_ss = -target_rate / (np.degrees(1.5 * n * J2 * (R_earth / a)**2) * 86400)
    if abs(cos_i_ss) <= 1:
        i_ss = np.degrees(np.arccos(cos_i_ss))
        print(f"      Sun-synchronous at {h_orbit/1e3:.0f} km requires i = {i_ss:.1f} deg")
    else:
        print(f"      Sun-synchronous orbit not achievable at {h_orbit/1e3:.0f} km altitude")


if __name__ == "__main__":
    print("=== Exercise 1: Quadrupole Potential Map ===")
    exercise_1()
    print("\n=== Exercise 2: Multipole Moments of a Ring ===")
    exercise_2()
    print("\n=== Exercise 3: Magnetic Dipole Radiation ===")
    exercise_3()
    print("\n=== Exercise 4: E2 Radiation Pattern ===")
    exercise_4()
    print("\n=== Exercise 5: Earth's Gravitational Multipoles ===")
    exercise_5()
    print("\nAll exercises completed!")
