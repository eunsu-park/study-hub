"""
Exercises for Lesson 01: MHD Equilibria
Topic: MHD
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad


def exercise_1():
    """
    Problem 1: Force Balance in a Cylindrical Plasma

    Given:
      - p(r) = p0 (1 - r^2/a^2) for r < a, p=0 for r >= a
      - B_z = B0 = const
      - Derive B_theta(r) from radial force balance

    The radial equilibrium equation in cylindrical coordinates:
      dp/dr = J_z * B_theta - J_theta * B_z
    Using Ampere's law and the simplified radial force balance:
      dp/dr + d(B_theta^2/(2*mu0))/dr + B_theta^2/(mu0*r) = 0
    (since B_z = const, d(B_z^2)/dr = 0)
    """
    # Parameters
    p0 = 1e5       # Central pressure [Pa]
    a = 0.1        # Plasma radius [m]
    B0 = 1.0       # Axial field [T]
    R0 = 5 * a     # Major radius [m]
    mu0 = 4 * np.pi * 1e-7

    Nr = 500
    r = np.linspace(1e-6, a, Nr)
    dr = r[1] - r[0]

    # (a) Derive B_theta from force balance
    # dp/dr = -2 p0 r / a^2
    # From the pressure balance (theta-pinch contribution):
    #   d/dr(B_theta^2/(2*mu0)) + B_theta^2/(mu0*r) = -dp/dr
    # For a Z-pinch component, integrate the equilibrium:
    #   (1/r) d(r*B_theta)/dr = mu0 * J_z
    # From dp/dr = J_z * B_theta (simplified), we use:
    #   B_theta^2(r) = mu0 * integral_0^r (2*p0*r'/a^2) * r'/r dr' (approximate)
    # More rigorously, integrate the radial equilibrium:
    #   B_theta(r) = sqrt(2*mu0*(p(0) - p(r))) * correction
    # For a pure Z-pinch with this pressure profile:
    #   B_theta^2/(2*mu0) = integral from r to a of (dp/dr' + B_theta^2/(mu0*r')) dr'
    # Simplification: use dp/dr = -mu0 J_z B_theta / mu0...
    #
    # Direct approach: from dp/dr + d(B_theta^2/(2mu0))/dr + B_theta^2/(mu0 r) = 0
    # Multiply by r: d(r * B_theta^2/(2mu0))/dr = -r * dp/dr
    # Integrate: r * B_theta^2/(2mu0) = integral_0^r r' * (-dp/dr') dr'
    # = integral_0^r r' * (2 p0 r'/a^2) dr' = 2 p0 / a^2 * r^3/3
    # So: B_theta^2 = (4 mu0 p0 r^2) / (3 a^2)
    # B_theta(r) = r * sqrt(4 mu0 p0 / (3 a^2))

    B_theta = r * np.sqrt(4 * mu0 * p0 / (3 * a**2))

    print("(a) B_theta(r) = r * sqrt(4 * mu0 * p0 / (3 * a^2))")
    print(f"    B_theta(a) = {B_theta[-1]:.6f} T")

    # (b) Total plasma current
    # From Ampere's law: I_p = 2*pi*a * B_theta(a) / mu0
    I_p = 2 * np.pi * a * B_theta[-1] / mu0
    print(f"\n(b) Total plasma current I_p = {I_p:.2f} A = {I_p/1e3:.2f} kA")

    # (c) Safety factor q(r) = r * B_z / (R0 * B_theta(r))
    q = r * B0 / (R0 * B_theta + 1e-30)
    print(f"\n(c) Safety factor profile computed.")
    print(f"    q(a) = {q[-1]:.4f}")

    # (d) q on axis (r -> 0): use L'Hopital's rule
    # q(0) = lim_{r->0} r * B0 / (R0 * B_theta(r))
    # B_theta(r) ~ r * C, so q(0) = B0 / (R0 * C)
    C = np.sqrt(4 * mu0 * p0 / (3 * a**2))
    q0 = B0 / (R0 * C)
    print(f"\n(d) q(0) = B0 / (R0 * C) = {q0:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(r / a, B_theta, 'b-', linewidth=2)
    axes[0].set_xlabel('r/a')
    axes[0].set_ylabel('B_theta [T]')
    axes[0].set_title('Azimuthal Field')
    axes[0].grid(True, alpha=0.3)

    p_profile = p0 * (1 - (r / a)**2)
    axes[1].plot(r / a, p_profile / 1e3, 'r-', linewidth=2)
    axes[1].set_xlabel('r/a')
    axes[1].set_ylabel('Pressure [kPa]')
    axes[1].set_title('Pressure Profile')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(r / a, q, 'g-', linewidth=2)
    axes[2].set_xlabel('r/a')
    axes[2].set_ylabel('q(r)')
    axes[2].set_title('Safety Factor')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/ex01_force_balance.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex01_force_balance.png")


def exercise_2():
    """
    Problem 2: Bennett Relation for Z-Pinch

    Given: n=1e20 m^-3, T=10 keV, L=1 m, a=1 cm
    Bennett relation: I^2 = (8*pi/mu0) * N * k_B * T
    """
    n = 1e20       # m^-3
    T_keV = 10
    kB = 1.38e-23  # J/K
    eV = 1.6e-19   # J
    T_joule = T_keV * 1e3 * eV  # Convert keV to Joules
    L = 1.0        # m
    a = 0.01       # m (1 cm)
    mu0 = 4 * np.pi * 1e-7

    # (a) Total number of particles (per unit length for Bennett)
    N_total = n * np.pi * a**2 * L
    N_per_length = n * np.pi * a**2
    print(f"(a) Total particles: N = {N_total:.2e}")
    print(f"    Particles per unit length: N/L = {N_per_length:.2e} m^-1")

    # (b) Bennett relation: I^2 = (8*pi/mu0) * N_per_length * kB * T
    # Note: for Bennett relation, kT includes both ions and electrons
    # so use 2*n*kT for total pressure
    I_squared = (8 * np.pi / mu0) * N_per_length * T_joule
    I_p = np.sqrt(I_squared)
    print(f"\n(b) Required current: I_p = {I_p:.2f} A = {I_p/1e3:.2f} kA")

    # (c) Magnetic field at surface
    B_theta_a = mu0 * I_p / (2 * np.pi * a)
    print(f"\n(c) B_theta(a) = {B_theta_a:.4f} T")

    # (d) Compare pressures
    p_mag = B_theta_a**2 / (2 * mu0)
    p_plasma = n * T_joule  # Single species; total = 2*n*kT for ion+electron
    p_plasma_total = 2 * n * T_joule
    print(f"\n(d) Magnetic pressure: p_mag = {p_mag:.2e} Pa")
    print(f"    Plasma pressure (ion+electron): p_plasma = {p_plasma_total:.2e} Pa")
    print(f"    Ratio p_mag/p_plasma = {p_mag/p_plasma_total:.4f}")

    # (e) Stability
    print("\n(e) Stability implications:")
    print("    A pure Z-pinch (no axial field B_z) is unstable to both")
    print("    sausage (m=0) and kink (m=1) instabilities. The safety")
    print("    factor q=0 everywhere (no B_z), violating the Kruskal-")
    print("    Shafranov criterion. An axial field B_z > B_theta(a)")
    print("    is needed for stabilization.")


def exercise_3():
    """
    Problem 3: Grad-Shafranov with Constant Pressure

    With p(psi) = p0 = const and F(psi) = F0 = const:
      Delta* psi = -mu0 R^2 dp/dpsi - F dF/dpsi
    Since p and F are constant in psi: dp/dpsi = 0, dF/dpsi = 0
    Actually, the GS equation is:
      Delta* psi = -mu0 R^2 p'(psi) - F F'(psi)
    With p' = dp/dpsi and F' = dF/dpsi.
    If p = p0 (const), p' = 0; if F = F0, F' = 0.
    Then Delta* psi = 0 (vacuum solution).

    But the problem states the equation reduces to Delta* psi = -mu0 p0 R^2.
    This implies p'(psi) = p0 (i.e., p(psi) = p0 * psi, linear in psi).
    Reinterpret: p' = p0, F' = 0.
    """
    mu0 = 4 * np.pi * 1e-7
    p0 = 1e4       # Pa (pressure gradient coefficient)
    R0 = 3.0       # Major radius [m]
    a = 1.0        # Minor radius [m]

    print("(a) With p'(psi) = p0 and F'(psi) = 0:")
    print("    Delta* psi = -mu0 * p0 * R^2")
    print("    For large aspect ratio (R ~ R0):")
    print("    (1/r) d/dr(r dpsi/dr) + d^2psi/dz^2 = -mu0 * p0 * R0^2")

    # (b) Already shown above
    source = mu0 * p0 * R0**2
    print(f"\n(b) Source term: -mu0 * p0 * R0^2 = {-source:.4f}")

    # (c) Separable solution psi(r,z) = R_r(r) * Z_z(z)
    # Not straightforward for non-zero source. Instead, try a particular solution:
    # psi = A * r^2 + B * z^2 (quadratic)
    # Substituting: 4A + 2B = -mu0 * p0 * R0^2
    print("\n(c) For psi = A*r^2 + B*z^2:")
    print("    d^2psi/dr^2 = 2A, (1/r)dpsi/dr = 2A")
    print("    d^2psi/dz^2 = 2B")
    print("    Sum: 4A + 2B = -mu0 * p0 * R0^2")

    # (d) Circular flux surfaces: psi = C(r^2 + kappa^2 * z^2)
    # Then: A = C, B = C * kappa^2
    # 4C + 2C*kappa^2 = -mu0 * p0 * R0^2
    # For circular cross-section, kappa = 1:
    # 6C = -mu0 * p0 * R0^2 => C = -mu0 * p0 * R0^2 / 6
    C_circ = -mu0 * p0 * R0**2 / 6
    print(f"\n(d) For circular flux surfaces (kappa=1):")
    print(f"    C = -mu0*p0*R0^2 / 6 = {C_circ:.6f}")
    print(f"    psi(r,z) = {C_circ:.6f} * (r^2 + z^2)")

    # For elongated cross-section, choose kappa != 1:
    # Any kappa satisfying 4C + 2C*kappa^2 = -mu0*p0*R0^2 works.
    # With A = C, B = C*kappa^2: 2C(2 + kappa^2) = -source
    # Different kappa give different elongations.
    kappa_vals = [0.5, 1.0, 1.5, 2.0]
    print("\n    For different elongations:")
    for kappa in kappa_vals:
        C_val = -source / (2 * (2 + kappa**2))
        print(f"    kappa = {kappa:.1f}: C = {C_val:.6f}")

    # Plot flux surfaces for different kappa
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    r_grid = np.linspace(-a, a, 200)
    z_grid = np.linspace(-a, a, 200)
    R, Z = np.meshgrid(r_grid, z_grid)

    for kappa in [1.0, 1.5]:
        C_val = -source / (2 * (2 + kappa**2))
        psi = C_val * (R**2 + kappa**2 * Z**2)
        ax = axes[0] if kappa == 1.0 else axes[1]
        cs = ax.contour(R, Z, psi, levels=15, cmap='RdBu_r')
        ax.set_xlabel('r [m]')
        ax.set_ylabel('z [m]')
        ax.set_title(f'Flux Surfaces (kappa={kappa})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/ex01_grad_shafranov.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex01_grad_shafranov.png")


def exercise_4():
    """
    Problem 4: Safety Factor and Current Profile

    Given: R0=3 m, a=1 m, B_t=5 T
    J_z(r) = J0 * (1 - r^2/a^2)
    """
    R0 = 3.0
    a = 1.0
    Bt = 5.0
    mu0 = 4 * np.pi * 1e-7

    Nr = 500
    r = np.linspace(1e-6, a, Nr)

    # (a) Enclosed current I(r)
    # I(r) = integral_0^r J0*(1 - r'^2/a^2) * 2*pi*r' dr'
    #       = J0 * 2*pi * [r^2/2 - r^4/(4*a^2)]
    # Total current: I(a) = J0 * 2*pi * [a^2/2 - a^2/4] = J0 * pi * a^2 / 2
    # Set J0 such that I(a) = 1 MA (typical tokamak)
    I_total = 1e6  # 1 MA
    J0 = I_total / (np.pi * a**2 / 2)

    I_enc = J0 * 2 * np.pi * (r**2 / 2 - r**4 / (4 * a**2))
    print(f"(a) J0 = {J0:.2e} A/m^2 (for I_total = 1 MA)")
    print(f"    I(a) = {I_enc[-1]:.2e} A")

    # (b) Poloidal field B_theta(r)
    B_theta = mu0 * I_enc / (2 * np.pi * r)
    print(f"\n(b) B_theta(a) = {B_theta[-1]:.4f} T")

    # (c) Safety factor q(r)
    q = r * Bt / (R0 * B_theta)
    print(f"\n(c) q(r) computed.")

    # (d) q(0) and q(a)
    # q(0): L'Hopital's rule. B_theta(r) ~ mu0*J0*pi*r / (2*pi) = mu0*J0*r/2 for small r
    # q(0) = lim r*Bt / (R0 * mu0*J0*r/2) = 2*Bt/(R0*mu0*J0)
    q0 = 2 * Bt / (R0 * mu0 * J0)
    qa = a * Bt / (R0 * B_theta[-1])
    print(f"\n(d) q(0) = {q0:.4f}")
    print(f"    q(a) = {qa:.4f}")

    # (e) Find r_s where q(r_s) = 2
    q_target = 2.0
    idx = np.argmin(np.abs(q - q_target))
    r_s = r[idx]
    print(f"\n(e) q=2 surface at r_s = {r_s:.4f} m (r_s/a = {r_s/a:.4f})")

    # (f) Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(r / a, J0 * (1 - (r / a)**2) / 1e6, 'b-', linewidth=2)
    axes[0].set_xlabel('r/a')
    axes[0].set_ylabel('J_z [MA/m^2]')
    axes[0].set_title('Current Density Profile')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(r / a, B_theta, 'r-', linewidth=2)
    axes[1].set_xlabel('r/a')
    axes[1].set_ylabel('B_theta [T]')
    axes[1].set_title('Poloidal Field')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(r / a, q, 'g-', linewidth=2)
    axes[2].axhline(1, color='r', linestyle='--', alpha=0.7, label='q=1')
    axes[2].axhline(2, color='orange', linestyle='--', alpha=0.7, label='q=2')
    axes[2].axhline(3, color='purple', linestyle='--', alpha=0.7, label='q=3')
    if r_s < a:
        axes[2].axvline(r_s / a, color='orange', linestyle=':', alpha=0.5)
    axes[2].set_xlabel('r/a')
    axes[2].set_ylabel('q(r)')
    axes[2].set_title('Safety Factor Profile')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, max(q) * 1.1])

    plt.tight_layout()
    plt.savefig('/tmp/ex01_safety_factor.png', dpi=100)
    plt.close()
    print("\n(f) Plot saved to /tmp/ex01_safety_factor.png")


def exercise_5():
    """
    Problem 5: Beta Limits

    Given: a=0.5 m, B_t=3 T, I_p=1 MA, p0=1e5 Pa
    p(r) = p0*(1 - r^2/a^2)^2
    """
    a = 0.5
    Bt = 3.0
    Ip = 1e6
    p0 = 1e5
    mu0 = 4 * np.pi * 1e-7

    # (a) Volume-averaged pressure
    # <p> = (1/V) integral p(r) dV = (2/(a^2)) integral_0^a p(r) r dr
    # = (2/a^2) * p0 * integral_0^a (1 - r^2/a^2)^2 r dr
    # Let u = r^2/a^2, du = 2r dr/a^2
    # = (2/a^2) * p0 * (a^2/2) * integral_0^1 (1-u)^2 du
    # = p0 * [u - u^2 + u^3/3]_0^1 = p0 * (1 - 1 + 1/3) = p0/3
    p_avg = p0 / 3.0
    print(f"(a) Volume-averaged pressure: <p> = p0/3 = {p_avg:.2e} Pa")

    # (b) Toroidal beta
    beta_t = 2 * mu0 * p_avg / Bt**2
    beta_t_percent = beta_t * 100
    print(f"\n(b) Toroidal beta: beta_t = {beta_t:.6f} = {beta_t_percent:.4f}%")

    # (c) Normalized beta
    # beta_N = beta_t(%) / (I_p[MA] / (a[m] * B_t[T]))
    Ip_MA = Ip / 1e6
    beta_N = beta_t_percent / (Ip_MA / (a * Bt))
    print(f"\n(c) Normalized beta: beta_N = {beta_N:.4f}")

    # (d) Compare with Troyon limit
    troyon_limit = 3.5
    print(f"\n(d) Troyon limit: beta_N < {troyon_limit}")
    print(f"    Current beta_N = {beta_N:.4f}")
    if beta_N < troyon_limit:
        print(f"    STABLE: beta_N ({beta_N:.4f}) < Troyon limit ({troyon_limit})")
        margin = (troyon_limit - beta_N) / troyon_limit * 100
        print(f"    Margin: {margin:.1f}%")
    else:
        print(f"    UNSTABLE: beta_N ({beta_N:.4f}) > Troyon limit ({troyon_limit})")

    # (e) Doubling pressure
    p0_new = 2 * p0
    p_avg_new = p0_new / 3.0
    beta_t_new = 2 * mu0 * p_avg_new / Bt**2
    beta_t_new_percent = beta_t_new * 100
    beta_N_new = beta_t_new_percent / (Ip_MA / (a * Bt))
    print(f"\n(e) With doubled pressure:")
    print(f"    New beta_N = {beta_N_new:.4f}")
    if beta_N_new > troyon_limit:
        # Need to increase I_p or B_t
        # Option 1: increase I_p to keep beta_N = Troyon limit
        Ip_needed = beta_t_new_percent * a * Bt / troyon_limit * 1e6
        print(f"    To maintain stability (beta_N = {troyon_limit}):")
        print(f"    Option 1: Increase I_p to {Ip_needed/1e6:.2f} MA")

        # Option 2: increase B_t to keep beta_N = Troyon limit
        # beta_t_new = 2*mu0*p_avg_new / B_new^2
        # beta_N_troyon = beta_t_new(%) / (I_p/(a*B_new))
        # Solving: B_new^3 = 2*mu0*p_avg_new*100*a / (troyon * I_p_MA)
        # Actually: beta_N = [2*mu0*p_avg_new*100/Bt^2] / [Ip_MA/(a*Bt)]
        # = 2*mu0*p_avg_new*100*a / (Bt * Ip_MA)
        # For new Bt: troyon = 2*mu0*p_avg_new*100*a / (Bt_new * Ip_MA)
        Bt_needed = 2 * mu0 * p_avg_new * 100 * a / (troyon_limit * Ip_MA)
        print(f"    Option 2: Increase B_t to {Bt_needed:.2f} T")
    else:
        print(f"    Still within Troyon limit.")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Force Balance in a Cylindrical Plasma ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Bennett Relation for Z-Pinch ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Grad-Shafranov with Constant Pressure ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Safety Factor and Current Profile ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Beta Limits ===")
    print("=" * 60)
    exercise_5()

    print("\nAll exercises completed!")
