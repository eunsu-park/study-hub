"""
Exercises for Lesson 07: Advanced Reconnection
Topic: MHD
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: Plasmoid Instability Onset

    Solar flare: L=1e9 m, v_A=1e6 m/s, eta=1e-4 Ohm.m
    """
    L = 1e9       # m
    v_A = 1e6     # m/s
    eta = 1e-4    # Ohm.m
    mu0 = 4 * np.pi * 1e-7

    # (a) Lundquist number
    S = L * v_A * mu0 / eta
    print(f"(a) Lundquist number:")
    print(f"    S = L*v_A*mu0/eta = {L:.1e}*{v_A:.1e}*{mu0:.4e}/{eta:.1e}")
    print(f"    S = {S:.4e}")

    # (b) Critical S
    S_c = 1e4
    print(f"\n(b) Critical Lundquist number: S_c ~ {S_c:.0e}")
    if S > S_c:
        print(f"    S = {S:.2e} >> S_c = {S_c:.0e}")
        print(f"    ABOVE critical: Plasmoid instability expected!")
    else:
        print(f"    BELOW critical: Sweet-Parker sheet stable")

    # (c) Number of plasmoids
    N = S**(0.25)
    print(f"\n(c) Estimated number of plasmoids: N ~ S^(1/4)")
    print(f"    N ~ ({S:.2e})^(1/4) = {N:.0f}")
    print(f"    The current sheet fragments into ~{N:.0f} plasmoids")
    print(f"    This creates a chain of X-points and O-points")


def exercise_2():
    """
    Problem 2: Plasmoid Growth Rate

    gamma*tau_A ~ S^(1/4), S = 1e12, tau_A = 1000 s
    """
    S = 1e12
    tau_A = 1000.0  # s (L/v_A for solar flare)
    eta = 1e-4      # Ohm.m (for diffusion time)
    mu0 = 4 * np.pi * 1e-7

    # (a) Growth rate
    gamma_tau_A = S**0.25
    gamma = gamma_tau_A / tau_A
    print(f"(a) Plasmoid growth rate:")
    print(f"    gamma * tau_A ~ S^(1/4) = ({S:.1e})^(1/4) = {gamma_tau_A:.2e}")
    print(f"    gamma = {gamma_tau_A:.2e} / {tau_A:.0f}")
    print(f"    gamma = {gamma:.4e} s^-1")
    print(f"    Growth time: tau_growth = 1/gamma = {1/gamma:.4e} s")
    print(f"    = {1/gamma:.2f} s")

    # (b) Compare to resistive diffusion time
    # tau_diff = L^2 * mu0 / eta, but we need L
    # S = L*v_A*mu0/eta, tau_A = L/v_A
    # L = v_A * tau_A
    L = 1e9  # from problem context
    tau_diff = L**2 * mu0 / eta
    print(f"\n(b) Resistive diffusion time:")
    print(f"    tau_diff = L^2 * mu0 / eta")
    print(f"    = ({L:.1e})^2 * {mu0:.4e} / {eta:.1e}")
    print(f"    = {tau_diff:.4e} s = {tau_diff/(3600*24*365):.2e} years")
    print(f"    Ratio gamma * tau_diff = {gamma * tau_diff:.2e}")
    print(f"    Plasmoid instability grows much faster than resistive diffusion")
    print(f"    (by a factor of ~S^(3/4))")


def exercise_3():
    """
    Problem 3: Plasmoid-Mediated Reconnection Rate

    Plot M_A vs S for SP (S^-1/2) and plasmoid-mediated (S^-1/8)
    Find where plasmoid rate is 10x faster than SP
    """
    S_vals = np.logspace(4, 16, 500)
    M_SP = S_vals**(-0.5)
    M_plasmoid = 0.01 * S_vals**(-1.0/8)  # normalized to match at S_c ~ 1e4

    # Normalize plasmoid rate: at S_c = 1e4, both should be comparable
    # M_SP(1e4) = 1e-2, so set M_plasmoid(1e4) = 1e-2 as well
    # 0.01 * (1e4)^(-1/8) = 0.01 * 0.178 = 0.00178 -> need different normalization
    # Actually the theoretical rate is M_A ~ S^0 ~ 0.01 (weakly dependent)
    # Use the standard form: M_plasmoid ~ S^(-1/8) * C
    # At S = 1e4 (onset), M_SP = 0.01, and plasmoid takes over
    # C * (1e4)^(-1/8) = 0.01 => C = 0.01 * (1e4)^(1/8) = 0.01 * 5.623 = 0.0562
    C_plasmoid = 0.01 * (1e4)**(1.0/8)
    M_plasmoid = C_plasmoid * S_vals**(-1.0/8)

    # (a) Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(S_vals, M_SP, 'b-', linewidth=2, label='Sweet-Parker: $S^{-1/2}$')
    ax.loglog(S_vals, M_plasmoid, 'r-', linewidth=2,
              label='Plasmoid-mediated: $\\sim S^{-1/8}$')
    ax.axhline(y=0.01, color='g', linestyle=':', alpha=0.5, label='$M_A = 0.01$')
    ax.axvline(x=1e4, color='gray', linestyle='--', alpha=0.5, label='$S_c \\sim 10^4$')
    ax.set_xlabel('Lundquist number S', fontsize=12)
    ax.set_ylabel('$M_A = v_{in}/v_A$', fontsize=12)
    ax.set_title('Sweet-Parker vs Plasmoid-Mediated Reconnection', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(1e-9, 1)
    plt.tight_layout()
    plt.savefig('/tmp/ex07_plasmoid_rate.png', dpi=100)
    plt.close()
    print("(a) Plot saved to /tmp/ex07_plasmoid_rate.png")

    # (b) At what S is plasmoid rate 10x faster?
    # M_plasmoid / M_SP = 10
    # C * S^(-1/8) / S^(-1/2) = 10
    # C * S^(3/8) = 10
    # S = (10/C)^(8/3)
    S_10x = (10 / C_plasmoid)**(8.0/3)
    print(f"\n(b) Plasmoid rate is 10x faster than SP when:")
    print(f"    C * S^(3/8) = 10")
    print(f"    S = (10/C)^(8/3) = {S_10x:.4e}")
    print(f"    At this S, M_SP = {S_10x**(-0.5):.4e},")
    print(f"    M_plasmoid = {C_plasmoid * S_10x**(-1.0/8):.4e}")


def exercise_4():
    """
    Problem 4: Turbulent Reconnection (LV99)

    l = 0.1*L, L = 1 pc, v_A = 1 km/s
    """
    L_pc = 1.0            # parsec
    L = L_pc * 3.086e16   # m (1 pc)
    l = 0.1 * L           # injection scale
    v_A = 1e3             # m/s
    year = 3.156e7        # s

    # (a) Reconnection rate
    M_A = np.sqrt(l / L)
    v_rec = M_A * v_A
    print(f"(a) LV99 turbulent reconnection rate:")
    print(f"    M_A ~ (l/L)^(1/2) = ({l/L:.1f})^(1/2) = {M_A:.4f}")
    print(f"    v_rec = M_A * v_A = {v_rec:.2f} m/s")

    # (b) Reconnection time
    tau_rec = L / v_rec
    tau_years = tau_rec / year
    tau_Myr = tau_years / 1e6
    print(f"\n(b) Reconnection time: tau = L/v_rec")
    print(f"    = {L:.2e} / {v_rec:.2f}")
    print(f"    = {tau_rec:.4e} s")
    print(f"    = {tau_years:.4e} years")
    print(f"    = {tau_Myr:.2f} Myr")

    # (c) Compare to star formation time
    tau_SF_Myr = 1.0  # ~1 Myr typical
    print(f"\n(c) Star formation timescale: ~1-10 Myr")
    print(f"    Turbulent reconnection time: {tau_Myr:.2f} Myr")
    if tau_Myr < 10:
        print(f"    Comparable to star formation timescale!")
        print(f"    Turbulent reconnection may play a role in removing")
        print(f"    magnetic support from molecular clouds, allowing collapse.")
    else:
        print(f"    Much longer than star formation timescale")


def exercise_5():
    """
    Problem 5: Guide Field Suppression

    M_A(Bg) = M_A(0) / (1 + Bg^2/B0^2)
    """
    M_A_0 = 0.1

    # (a) Bg = B0
    Bg_B0 = 1.0
    M_A_a = M_A_0 / (1 + Bg_B0**2)
    print(f"(a) With Bg = B0 (Bg/B0 = {Bg_B0}):")
    print(f"    M_A = {M_A_0} / (1 + {Bg_B0}^2) = {M_A_0} / {1 + Bg_B0**2}")
    print(f"    M_A = {M_A_a:.4f}")

    # (b) Bg = 3*B0
    Bg_B0 = 3.0
    M_A_b = M_A_0 / (1 + Bg_B0**2)
    print(f"\n(b) With Bg = 3*B0 (Bg/B0 = {Bg_B0}):")
    print(f"    M_A = {M_A_0} / (1 + {Bg_B0}^2) = {M_A_0} / {1 + Bg_B0**2}")
    print(f"    M_A = {M_A_b:.4f}")

    # (c) Plot
    Bg_B0_vals = np.linspace(0, 5, 200)
    M_A_vals = M_A_0 / (1 + Bg_B0_vals**2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Bg_B0_vals, M_A_vals, 'b-', linewidth=2)
    ax.axhline(y=M_A_a, color='r', linestyle='--', alpha=0.5, label=f'$B_g = B_0$: $M_A$ = {M_A_a:.4f}')
    ax.axhline(y=M_A_b, color='g', linestyle='--', alpha=0.5, label=f'$B_g = 3B_0$: $M_A$ = {M_A_b:.4f}')
    ax.axvline(x=1.0, color='r', linestyle=':', alpha=0.3)
    ax.axvline(x=3.0, color='g', linestyle=':', alpha=0.3)
    ax.set_xlabel('$B_g / B_0$', fontsize=12)
    ax.set_ylabel('$M_A$', fontsize=12)
    ax.set_title('Guide Field Suppression of Reconnection Rate', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/ex07_guide_field.png', dpi=100)
    plt.close()
    print("\n(c) Plot saved to /tmp/ex07_guide_field.png")


def exercise_6():
    """
    Problem 6: Relativistic Alfven Speed

    v_A = c * sqrt(sigma/(1+sigma))
    """
    c = 3e8  # m/s

    # (a) Show derivation
    print("(a) Relativistic Alfven speed derivation:")
    print("    In relativistic MHD, the enthalpy includes rest mass + internal + magnetic:")
    print("    w = rho*c^2 + gamma_ad/(gamma_ad-1)*p + B^2/mu0")
    print("    The Alfven speed is: v_A^2 = B^2/(mu0*w)")
    print("    Defining sigma = B^2/(mu0*rho*c^2):")
    print("    For cold plasma (p -> 0): w = rho*c^2 + B^2/mu0 = rho*c^2*(1 + sigma)")
    print("    v_A^2 = sigma*rho*c^2 / (rho*c^2*(1+sigma)) = sigma*c^2/(1+sigma)")
    print("    v_A = c * sqrt(sigma/(1+sigma))")

    # (b) Calculate for various sigma
    sigmas = [0.1, 1.0, 10.0, 100.0]
    print(f"\n(b) Relativistic Alfven speed for various sigma:")
    print(f"    {'sigma':>10} {'v_A/c':>10} {'v_A (m/s)':>15}")
    for sigma in sigmas:
        vA_c = np.sqrt(sigma / (1 + sigma))
        vA = vA_c * c
        print(f"    {sigma:10.1f} {vA_c:10.4f} {vA:15.4e}")

    # (c) At what sigma is v_A = 0.9c?
    # 0.9^2 = sigma/(1+sigma) => 0.81 + 0.81*sigma = sigma => 0.81 = 0.19*sigma
    target_vA = 0.9
    sigma_target = target_vA**2 / (1 - target_vA**2)
    print(f"\n(c) For v_A = {target_vA}c:")
    print(f"    0.81 = sigma/(1+sigma) => sigma = 0.81/0.19 = {sigma_target:.4f}")
    print(f"    sigma ~ {sigma_target:.1f}")
    print(f"    Verification: v_A/c = sqrt({sigma_target:.1f}/{sigma_target+1:.1f}) = {np.sqrt(sigma_target/(1+sigma_target)):.4f}")


def exercise_7():
    """
    Problem 7: Relativistic Outflow Lorentz Factor

    (a) v_out = 0.95c, (b) sigma = 1e4 for pulsar wind
    """
    c = 3e8

    # (a) Lorentz factor for v = 0.95c
    v = 0.95 * c
    beta = v / c
    Gamma = 1.0 / np.sqrt(1 - beta**2)
    print(f"(a) v_out = {beta}c:")
    print(f"    Gamma = 1/sqrt(1 - beta^2) = 1/sqrt(1 - {beta}^2)")
    print(f"    = 1/sqrt({1-beta**2:.6f})")
    print(f"    = {Gamma:.4f}")

    # (b) Pulsar wind with sigma = 1e4
    sigma = 1e4
    # Outflow speed ~ relativistic Alfven speed for high sigma
    vA_c = np.sqrt(sigma / (1 + sigma))
    Gamma_out = np.sqrt(1 + sigma)  # For sigma >> 1, Gamma ~ sqrt(sigma)
    print(f"\n(b) Pulsar wind with sigma = {sigma:.0e}:")
    print(f"    v_A/c = sqrt(sigma/(1+sigma)) = {vA_c:.6f}")
    print(f"    beta = {vA_c:.6f}")
    print(f"    For sigma >> 1: Gamma ~ sqrt(sigma) = sqrt({sigma:.0e}) = {np.sqrt(sigma):.0f}")
    print(f"    Exact: Gamma = 1/sqrt(1 - v_A^2/c^2) = sqrt(1+sigma) = {Gamma_out:.2f}")
    print(f"    The outflow is ultrarelativistic with Gamma ~ {Gamma_out:.0f}")


def exercise_8():
    """
    Problem 8: Sigma Problem in Pulsars

    sigma_init = 1e6, 99% conversion
    """
    sigma_init = 1e6
    f_convert = 0.99

    # (a) Final sigma
    # sigma = E_B / E_particle
    # E_B_final = (1-f)*E_B_init = 0.01 * sigma_init * E_rest
    # E_particle_final = E_rest + f*E_B_init = E_rest*(1 + f*sigma_init)
    sigma_final = (1 - f_convert) * sigma_init / (1 + f_convert * sigma_init)
    print(f"(a) Initial sigma = {sigma_init:.0e}, {f_convert*100:.0f}% conversion:")
    print(f"    sigma_final = (1-f)*sigma / (1 + f*sigma)")
    print(f"    = {1-f_convert}*{sigma_init:.0e} / (1 + {f_convert}*{sigma_init:.0e})")
    print(f"    = {(1-f_convert)*sigma_init:.0e} / {1+f_convert*sigma_init:.0e}")
    print(f"    = {sigma_final:.6f}")
    print(f"    ~ {sigma_final:.4f}")

    # (b) Compare to observed
    sigma_obs = 0.01
    print(f"\n(b) Observed sigma at termination shock: ~{sigma_obs}")
    print(f"    Computed sigma_final: {sigma_final:.4f}")
    if sigma_final < sigma_obs:
        print(f"    sigma_final < sigma_obs: SUFFICIENT to explain observations")
    else:
        print(f"    sigma_final > sigma_obs: NOT sufficient")
        print(f"    Need additional dissipation (see part c)")

    # (c) Additional mechanisms
    print(f"\n(c) Additional dissipation mechanisms needed:")
    print(f"    The sigma problem: even with 99% conversion via reconnection,")
    print(f"    sigma remains ~{sigma_final:.4f}, which is close to but may exceed")
    print(f"    the observed value of ~0.01.")
    print(f"    Additional mechanisms:")
    print(f"    1. Kink instability of the striped wind: destroys organized")
    print(f"       toroidal field, promoting turbulent dissipation")
    print(f"    2. Particle acceleration at the termination shock:")
    print(f"       shock converts remaining magnetic energy")
    print(f"    3. Turbulent cascade in the nebula: MHD turbulence")
    print(f"       provides continued dissipation downstream")
    print(f"    4. Multi-scale reconnection: cascade of plasmoids")
    print(f"       enhances the effective dissipation rate")


def exercise_9():
    """
    Problem 9: 3D Null Eigenvalues

    Spine-fan null with eigenvalues (a, a, -2a)
    B = (ax, ay, -2az)
    """
    a = 1.0  # arbitrary

    # (a) Verify div(B) = 0
    print(f"(a) For eigenvalues ({a}, {a}, {-2*a}):")
    print(f"    Jacobian matrix: diag({a}, {a}, {-2*a})")
    print(f"    div(B) = dBx/dx + dBy/dy + dBz/dz")
    print(f"    = {a} + {a} + ({-2*a}) = {a + a + (-2*a)}")
    print(f"    div(B) = 0 (solenoidal condition satisfied)")

    # (b) Field components and field line sketch
    print(f"\n(b) Field components: B = ({a}*x, {a}*y, {-2*a}*z)")
    print(f"    In the xy-plane (z=0): B_z = 0, field is radially outward")
    print(f"    Along z-axis (x=y=0): B = (0, 0, -2az), field points toward null")
    print(f"    The null point is at the origin where B = 0")

    # (c) Spine and fan
    print(f"\n(c) Spine and fan structures:")
    print(f"    Spine (1D structure): Along z-axis (x=y=0)")
    print(f"      - Field lines approach the null along z (eigenvalue -2a)")
    print(f"      - This is the eigenvector with the distinct eigenvalue")
    print(f"    Fan (2D structure): In the xy-plane (z=0)")
    print(f"      - Field lines radiate outward from null in xy-plane")
    print(f"      - This is the plane spanned by the two degenerate eigenvalues")
    print(f"    Classification: Positive fan (a > 0 in xy), spine inflow (along z)")

    # Plot 3D null structure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Fan plane field lines (in xy-plane, z=0)
    theta_vals = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for theta in theta_vals:
        r = np.linspace(0.1, 2, 50)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(r)
        ax.plot(x, y, z, 'b-', linewidth=1, alpha=0.7)

    # Spine field lines (along z-axis)
    z_spine = np.linspace(-2, -0.1, 30)
    ax.plot(np.zeros_like(z_spine), np.zeros_like(z_spine), z_spine, 'r-', linewidth=3, label='Spine')
    z_spine2 = np.linspace(0.1, 2, 30)
    ax.plot(np.zeros_like(z_spine2), np.zeros_like(z_spine2), z_spine2, 'r-', linewidth=3)

    # Add arrows for direction
    ax.quiver(0, 0, 1.5, 0, 0, -0.5, color='r', arrow_length_ratio=0.3)
    ax.quiver(0, 0, -1.5, 0, 0, 0.5, color='r', arrow_length_ratio=0.3)
    ax.quiver(1.5, 0, 0, 0.3, 0, 0, color='b', arrow_length_ratio=0.3)
    ax.quiver(0, 1.5, 0, 0, 0.3, 0, color='b', arrow_length_ratio=0.3)

    # Null point
    ax.scatter([0], [0], [0], color='k', s=100, zorder=5, label='Null point')

    # Fan plane (translucent)
    xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='blue')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Magnetic Null: Spine-Fan Structure', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig('/tmp/ex07_3d_null.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex07_3d_null.png")


def exercise_10():
    """
    Problem 10: QSL Squashing Factor

    Research and explain QSL squashing factor Q
    """
    print("(a) Squashing factor Q definition:")
    print("    Q measures how much the mapping of field lines between two")
    print("    boundaries (e.g., photosphere to photosphere) distorts")
    print("    infinitesimal flux tubes. Formally:")
    print()
    print("    Given a field line mapping F: (x1,y1) -> (x2,y2),")
    print("    Q = (|dF/d(x1,y1)|_F^2) / |det(dF/d(x1,y1))|")
    print("    where |...|_F is the Frobenius norm of the Jacobian matrix.")
    print()
    print("    High Q means the mapping strongly squashes flux tubes:")
    print("    nearby field lines at one footpoint map to widely separated")
    print("    points at the other footpoint. This creates intense current")
    print("    sheets because Ampere's law (J = curl(B)/mu0) requires strong")
    print("    currents where the field changes rapidly.")

    print("\n(b) Flare ribbons and QSL (Q > 2):")
    print("    Flare ribbons mark the chromospheric footpoints of field lines")
    print("    that are reconnecting or have recently reconnected. These field")
    print("    lines pass through the quasi-separatrix layer (QSL) where Q is")
    print("    large. The QSL is the 3D generalization of a separatrix.")
    print("    Ribbons trace Q > 2 regions because:")
    print("    1. Current sheets form preferentially at QSLs")
    print("    2. Reconnection occurs at these current sheets")
    print("    3. Accelerated particles follow reconnected field lines to")
    print("       their footpoints, heating the chromosphere (ribbons)")
    print("    In practice, Q >> 2 (often Q > 10^4) at actual reconnection sites.")

    print("\n(c) Numerical computation of Q:")
    print("    1. Start with a 3D magnetic field B(x,y,z) (e.g., from NLFFF")
    print("       extrapolation or MHD simulation)")
    print("    2. Select a grid of starting points on one boundary (e.g., z=0)")
    print("    3. For each starting point (x0,y0), trace the field line to")
    print("       the other boundary, recording endpoint (x1,y1)")
    print("    4. Compute the Jacobian matrix J = d(x1,y1)/d(x0,y0)")
    print("       by finite differencing (trace neighboring field lines)")
    print("    5. Calculate Q = (J11^2 + J12^2 + J21^2 + J22^2) / |det(J)|")
    print("    6. Plot Q on the boundary; high-Q regions are QSLs")
    print("    Software: QSL Squasher, TOPOTR, or custom Python codes")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Plasmoid Instability Onset ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Plasmoid Growth Rate ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Plasmoid-Mediated Reconnection Rate ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Turbulent Reconnection (LV99) ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Guide Field Suppression ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Relativistic Alfven Speed ===")
    print("=" * 60)
    exercise_6()

    print("\n" + "=" * 60)
    print("=== Exercise 7: Relativistic Outflow Lorentz Factor ===")
    print("=" * 60)
    exercise_7()

    print("\n" + "=" * 60)
    print("=== Exercise 8: Sigma Problem in Pulsars ===")
    print("=" * 60)
    exercise_8()

    print("\n" + "=" * 60)
    print("=== Exercise 9: 3D Null Eigenvalues ===")
    print("=" * 60)
    exercise_9()

    print("\n" + "=" * 60)
    print("=== Exercise 10: QSL Squashing Factor ===")
    print("=" * 60)
    exercise_10()

    print("\nAll exercises completed!")
