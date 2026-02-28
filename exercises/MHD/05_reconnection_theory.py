"""
Exercises for Lesson 05: Magnetic Reconnection Theory
Topic: MHD
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: Sweet-Parker Scaling

    Earth's magnetotail: L=1e7 m, B=20 nT, n=1e6 m^-3, eta=1e-2 Ohm.m
    """
    mu0 = 4 * np.pi * 1e-7
    mp = 1.67e-27

    # (a) Derive Sweet-Parker reconnection rate (analytical walkthrough)
    print("(a) Sweet-Parker derivation:")
    print("    Mass conservation: rho*v_in*L = rho*v_out*delta")
    print("    Momentum balance:  B^2/(2*mu0) = rho*v_out^2/2  =>  v_out = v_A")
    print("    Ohm's law:         v_in*B = eta*B/(mu0*delta)")
    print("    => delta = eta/(mu0*v_in)")
    print("    Combining: v_in = v_A / sqrt(S)")
    print("    where S = L*v_A*mu0/eta (Lundquist number)")
    print("    Reconnection rate: M_A = v_in/v_A = S^(-1/2)")

    # (b) Earth's magnetotail parameters
    L = 1e7      # m
    B = 20e-9    # T
    n = 1e6      # m^-3
    eta = 1e-2   # Ohm.m
    rho = n * mp

    v_A = B / np.sqrt(mu0 * rho)
    S = L * v_A * mu0 / eta
    M_A = 1.0 / np.sqrt(S)
    v_in = M_A * v_A

    print(f"\n(b) Earth's magnetotail:")
    print(f"    rho = n*mp = {rho:.4e} kg/m^3")
    print(f"    v_A = B/sqrt(mu0*rho) = {v_A:.4e} m/s")
    print(f"    S = L*v_A*mu0/eta = {S:.4e}")
    print(f"    M_A = S^(-1/2) = {M_A:.4e}")
    print(f"    v_in = M_A * v_A = {v_in:.4e} m/s")

    # (c) Reconnection timescale
    tau_SP = L / v_in
    print(f"\n(c) Reconnection timescale: tau = L/v_in = {tau_SP:.4e} s")
    print(f"    = {tau_SP/3600:.2f} hours")
    substorm_time = 3600  # ~1 hour
    print(f"    Observed substorm onset: ~1 hour = {substorm_time} s")
    if tau_SP < 10 * substorm_time:
        print("    Roughly consistent with observed substorm timescales")
    else:
        print("    Much longer than observed substorm timescales")
        print("    => Sweet-Parker is too slow; faster mechanisms needed")


def exercise_2():
    """
    Problem 2: Petschek vs Sweet-Parker

    Compare M_A scaling for both models as function of S
    """
    # (a) At what S does Petschek exceed Sweet-Parker?
    # M_SP = S^(-1/2), M_P = pi/(8*ln(S))
    # They cross when S^(-1/2) = pi/(8*ln(S))
    # ln(S) = pi * S^(1/2) / 8
    # For S >> 1, Petschek always exceeds SP. At S ~ 10^2, they are comparable.
    print("(a) Petschek always exceeds Sweet-Parker for S >> 1:")
    print("    M_SP = S^(-1/2), M_P = pi/(8*ln(S))")
    print("    For S = 10^2: M_SP = 0.1, M_P = pi/(8*ln(100)) = 0.085")
    print("    For S = 10^4: M_SP = 0.01, M_P = pi/(8*ln(10^4)) = 0.043")
    print("    Petschek exceeds SP for S > ~10^2 (crossover)")

    # Find crossover numerically
    S_test = np.logspace(1, 5, 10000)
    M_SP = S_test**(-0.5)
    M_P = np.pi / (8 * np.log(S_test))
    # Find where M_P first exceeds M_SP
    crossover_mask = M_P > M_SP
    if np.any(crossover_mask):
        S_cross = S_test[np.argmax(crossover_mask)]
        print(f"    Numerical crossover: S ~ {S_cross:.0f}")

    # (b) Plot ratio
    S_vals = np.logspace(2, 16, 500)
    M_SP_vals = S_vals**(-0.5)
    M_P_vals = np.pi / (8 * np.log(S_vals))
    ratio = M_P_vals / M_SP_vals

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: both rates
    ax1.loglog(S_vals, M_SP_vals, 'b-', linewidth=2, label='Sweet-Parker: $S^{-1/2}$')
    ax1.loglog(S_vals, M_P_vals, 'r-', linewidth=2, label=r'Petschek: $\pi/(8\ln S)$')
    ax1.axhline(y=0.1, color='g', linestyle=':', alpha=0.5, label='Observed: $M_A \\sim 0.1$')
    ax1.set_xlabel('Lundquist number S', fontsize=12)
    ax1.set_ylabel('$M_A = v_{in}/v_A$', fontsize=12)
    ax1.set_title('Reconnection Rates', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right panel: ratio
    ax2.semilogx(S_vals, ratio, 'k-', linewidth=2)
    ax2.set_xlabel('Lundquist number S', fontsize=12)
    ax2.set_ylabel('$M_P / M_{SP}$', fontsize=12)
    ax2.set_title('Petschek/Sweet-Parker Ratio', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/ex05_petschek_vs_sp.png', dpi=100)
    plt.close()
    print("\n(b) Plot saved to /tmp/ex05_petschek_vs_sp.png")
    print(f"    At S = 10^14: ratio = {np.pi / (8 * np.log(1e14)) * np.sqrt(1e14):.0f}")

    # (c) Physical explanation
    print("\n(c) Physical reason Petschek is faster:")
    print("    In Sweet-Parker, the entire current sheet (length L) acts as the")
    print("    diffusion region. Plasma must be ejected from this long, thin layer.")
    print("    In Petschek, only a tiny region near the X-point is diffusive.")
    print("    Extended slow-mode shocks handle the energy conversion, allowing")
    print("    a much shorter diffusion region and higher inflow speed.")


def exercise_3():
    """
    Problem 3: Hall Reconnection

    Solar corona: n = 1e16 m^-3, L = 1e9 m
    """
    n = 1e16      # m^-3
    L = 1e9       # m
    mu0 = 4 * np.pi * 1e-7
    e = 1.602e-19
    mp = 1.67e-27
    c = 3e8       # m/s

    # (a) Ion skin depth
    omega_pi = np.sqrt(n * e**2 / (mp * 8.854e-12))  # plasma frequency approach
    # More directly: d_i = c / omega_pi where omega_pi = sqrt(n*e^2/(eps0*mi))
    # Or: d_i = c / sqrt(mu0*n*e^2/mi) = sqrt(mi/(mu0*n*e^2)) * c... let's use standard
    # d_i = c/omega_pi = c * sqrt(mp*eps0/(n*e^2))
    # Simpler: d_i = c / sqrt(n * e^2 / (eps0 * mp)) but use SI consistently
    # omega_pi^2 = n*e^2 / (eps0 * mp)
    eps0 = 8.854e-12
    omega_pi = np.sqrt(n * e**2 / (eps0 * mp))
    d_i = c / omega_pi
    print(f"(a) Ion skin depth (ion inertial length):")
    print(f"    omega_pi = sqrt(n*e^2/(eps0*mp)) = {omega_pi:.4e} rad/s")
    print(f"    d_i = c/omega_pi = {d_i:.4e} m = {d_i/1e3:.2f} km")

    # (b) Scale separation
    ratio = L / d_i
    print(f"\n(b) Scale separation:")
    print(f"    L/d_i = {L:.1e} / {d_i:.4e} = {ratio:.2e}")
    print(f"    The global scale is ~{ratio:.0e} times larger than d_i")
    print(f"    This large separation is why MHD works on large scales")

    # (c) Two-scale structure sketch
    print("\n(c) Two-scale structure of Hall reconnection:")
    print("    Outer region (r >> d_i): Standard MHD, ions+electrons frozen to field")
    print("    Ion diffusion region (r ~ d_i):")
    print("        - Ions decouple from field lines")
    print("        - Electrons still frozen to field")
    print("        - Hall currents generate quadrupolar out-of-plane B")
    print("    Electron diffusion region (r ~ d_e << d_i):")
    print("        - Electrons also decouple")
    print("        - Reconnection electric field supported by electron pressure tensor")
    d_e = d_i * np.sqrt(9.109e-31 / mp)  # d_e = d_i * sqrt(me/mi)
    print(f"    d_e = d_i * sqrt(m_e/m_i) = {d_e:.4e} m = {d_e:.2f} m")


def exercise_4():
    """
    Problem 4: X-point Field

    B = B0*(x*xhat - y*yhat)/L
    """
    B0 = 1.0
    L = 1.0

    # (a) Field lines (contours of psi)
    print("(a) Field lines of B = B0*(x*xhat - y*yhat)/L:")
    print("    B_x = B0*x/L, B_y = -B0*y/L")
    print("    The vector potential A_z gives B = curl(A_z zhat):")
    print("    B_x = dA_z/dy, B_y = -dA_z/dx")
    print("    dA_z/dy = B0*x/L  =>  A_z = B0*x*y/L + f(x)")
    print("    -dA_z/dx = -B0*y/L - f'(x) = -B0*y/L => f'(x) = 0")
    print("    psi = A_z = B0*x*y/L")
    print("    Field lines: x*y = const (hyperbolas)")

    # (b) Field strength
    print("\n(b) |B| = B0*sqrt(x^2 + y^2)/L = B0*r/L")
    print("    The field strength increases linearly with distance from the X-point")

    # (c) Maximum and minimum
    print("\n(c) |B| minimum: at (0,0) where |B| = 0 (the X-point/null)")
    print("    |B| maximum: at the boundaries (increases with distance)")
    print("    The field has a null at the origin and grows outward")

    # Plot
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Bx = B0 * X / L
    By = -B0 * Y / L
    psi = B0 * X * Y / L
    B_mag = B0 * np.sqrt(X**2 + Y**2) / L

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Field lines
    levels = np.linspace(-3, 3, 21)
    ax1.contour(X, Y, psi, levels=levels, colors='blue', linewidths=0.8)
    ax1.streamplot(X, Y, Bx, By, color='gray', density=1.5, linewidth=0.5)
    ax1.plot(0, 0, 'rx', markersize=15, markeredgewidth=3, label='X-point')
    ax1.set_xlabel('x/L', fontsize=12)
    ax1.set_ylabel('y/L', fontsize=12)
    ax1.set_title('X-point Field Lines (psi = B0*xy/L)', fontsize=13)
    ax1.set_aspect('equal')
    ax1.legend()

    # Field magnitude
    im = ax2.pcolormesh(X, Y, B_mag, cmap='hot', shading='auto')
    plt.colorbar(im, ax=ax2, label='|B|/B0')
    ax2.plot(0, 0, 'cx', markersize=15, markeredgewidth=3, label='Null (|B|=0)')
    ax2.set_xlabel('x/L', fontsize=12)
    ax2.set_ylabel('y/L', fontsize=12)
    ax2.set_title('Field Magnitude |B|', fontsize=13)
    ax2.set_aspect('equal')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('/tmp/ex05_xpoint.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex05_xpoint.png")


def exercise_5():
    """
    Problem 5: Reconnection Electric Field

    v_in = 0.1*v_A, B_in = 0.01 T, v_A = 1e6 m/s
    """
    v_A = 1e6     # m/s
    B_in = 0.01   # T
    v_in = 0.1 * v_A
    L_z = 1000e3  # 1000 km

    # (a) Reconnection electric field
    E_rec = v_in * B_in
    print(f"(a) E_rec = v_in * B_in = {v_in:.1e} * {B_in} = {E_rec:.2f} V/m")

    # (b) Numerical value
    print(f"\n(b) With v_A = {v_A:.1e} m/s:")
    print(f"    v_in = 0.1 * v_A = {v_in:.1e} m/s")
    print(f"    E_rec = {E_rec:.2f} V/m = {E_rec:.2e} V/m")

    # (c) Flux reconnected per second
    # dPhi/dt = E_rec * L_z
    dPhi_dt = E_rec * L_z
    print(f"\n(c) Flux reconnected per second across L_z = {L_z/1e3:.0f} km:")
    print(f"    dPhi/dt = E_rec * L_z = {E_rec:.2f} * {L_z:.2e}")
    print(f"    = {dPhi_dt:.4e} Wb/s = {dPhi_dt:.4e} V")
    print(f"    = {dPhi_dt/1e6:.2f} MWb/s")


def exercise_6():
    """
    Problem 6: Quadrupolar Hall Field

    Physical explanation of Hall-generated out-of-plane field
    """
    print("(a) Physical origin of quadrupolar Hall field:")
    print("    The Hall term J x B / (ne) arises because ions and electrons")
    print("    decouple on scales < d_i (ion skin depth).")
    print("    Near the X-point:")
    print("    - Ions (heavy) are unmagnetized and flow straight in/out")
    print("    - Electrons (light) remain magnetized and follow field lines")
    print("    - Electrons spiral around field lines, creating an out-of-plane current")
    print("    - This current generates the out-of-plane (Hall) magnetic field")
    print("    - The pattern has quadrupolar symmetry due to the X-point geometry")

    print("\n(b) Quadrupolar B_z structure:")
    print("         +Bz  |  -Bz")
    print("              |")
    print("    ----------X----------")
    print("              |")
    print("         -Bz  |  +Bz")
    print("    (Where X is the reconnection point, horizontal is inflow,")
    print("     vertical is outflow direction)")

    print("\n(c) Spacecraft observation:")
    print("    A spacecraft crossing the diffusion region would observe:")
    print("    - Bipolar B_z signature (positive then negative, or vice versa)")
    print("    - The sign depends on the crossing trajectory")
    print("    - Example: crossing through the inflow region on one side,")
    print("      the spacecraft sees B_z change sign as it passes the")
    print("      current sheet center")
    print("    - MMS mission routinely observes these signatures at")
    print("      Earth's magnetopause and magnetotail")

    # Plot quadrupolar field model
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    # Simple model of quadrupolar field: Bz ~ x*y * exp(-(x^2+y^2))
    Bz = X * Y * np.exp(-(X**2 + Y**2) / 2)

    fig, ax = plt.subplots(figsize=(8, 7))
    levels = np.linspace(-0.4, 0.4, 17)
    cs = ax.contourf(X, Y, Bz, levels=levels, cmap='RdBu_r')
    ax.contour(X, Y, Bz, levels=[0], colors='k', linewidths=2)
    plt.colorbar(cs, ax=ax, label='$B_z$ (arbitrary units)')
    ax.plot(0, 0, 'kx', markersize=15, markeredgewidth=3)
    ax.set_xlabel('x (inflow direction)', fontsize=12)
    ax.set_ylabel('y (outflow direction)', fontsize=12)
    ax.set_title('Quadrupolar Hall Magnetic Field', fontsize=14)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('/tmp/ex05_hall_quadrupolar.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex05_hall_quadrupolar.png")


def exercise_7():
    """
    Problem 7: Simulation Analysis

    v_in = 0.05*v_A, eta = 1e-4 (code units), L = 10, v_A = 1
    """
    v_in_norm = 0.05  # in units of v_A
    eta_code = 1e-4
    L_code = 10.0
    vA_code = 1.0

    # (a) M_A
    M_A = v_in_norm
    print(f"(a) M_A = v_in / v_A = {M_A}")

    # (b) Lundquist number
    mu0_code = 1.0  # typically mu0 = 1 in code units
    S = L_code * vA_code / eta_code  # S = L*v_A/eta (with mu0=1 in code units)
    print(f"\n(b) S = L*v_A/eta = {L_code}*{vA_code}/{eta_code}")
    print(f"    S = {S:.2e}")

    # (c) Compare to models
    M_SP = 1.0 / np.sqrt(S)
    M_P = np.pi / (8 * np.log(S))
    M_Hall = 0.1

    print(f"\n(c) Comparison of measured rate to models:")
    print(f"    Measured:      M_A = {M_A}")
    print(f"    Sweet-Parker:  M_SP = S^(-1/2) = {M_SP:.4f}")
    print(f"    Petschek:      M_P = pi/(8*ln(S)) = {M_P:.4f}")
    print(f"    Hall:          M_Hall ~ 0.1")
    print()

    if abs(M_A - M_SP) / M_A < 0.3:
        print("    => Consistent with SWEET-PARKER")
    elif abs(M_A - M_P) / M_A < 0.3:
        print("    => Consistent with PETSCHEK")
    elif abs(M_A - M_Hall) / M_A < 0.5:
        print("    => Consistent with HALL reconnection")
    else:
        print(f"    => Between models. M_A={M_A} is faster than SP ({M_SP:.4f})")
        print(f"       but slower than Hall (0.1). May indicate Petschek-like")
        print(f"       or transitional regime.")


def exercise_8():
    """
    Problem 8: Energy Conversion

    M_A = 0.1, compare magnetic inflow to kinetic outflow
    """
    mu0 = 4 * np.pi * 1e-7
    M_A = 0.1

    # (a) Magnetic energy inflow rate
    print("(a) Magnetic energy inflow rate per unit area:")
    print("    S_mag = v_in * B^2 / (2*mu0)")
    print(f"    With M_A = {M_A}: v_in = {M_A} * v_A")
    print(f"    S_mag = {M_A} * v_A * B^2 / (2*mu0)")
    print(f"    Since v_A = B/sqrt(mu0*rho):")
    print(f"    S_mag = {M_A} * B^3 / (2*mu0^(3/2) * rho^(1/2))")

    # (b) Kinetic energy outflow rate
    print("\n(b) Kinetic energy outflow rate per unit area:")
    print("    S_kin = rho * v_out^3 / 2")
    print("    With v_out ~ v_A = B/sqrt(mu0*rho):")
    print("    S_kin = rho * (B/sqrt(mu0*rho))^3 / 2")
    print("    = B^3 / (2*mu0^(3/2) * rho^(1/2))")
    print()
    print("    Ratio S_kin / S_mag ~ 1/M_A (using mass conservation)")
    print("    But corrected for cross-sections: inflow area ~ 2*L*Lz,")
    print("    outflow area ~ 2*delta*Lz, where L/delta ~ 1/M_A")
    print()
    print("    Magnetic inflow power: P_mag ~ v_in * B^2/(2*mu0) * 2*L*Lz")
    print("    Kinetic outflow power: P_kin ~ rho*v_A^3/2 * 2*delta*Lz")
    print("    P_kin/P_mag = (rho*v_A^3 * delta) / (v_in * B^2/(2*mu0) * L)")
    print("    = (rho*v_A^3 * delta) / (M_A*v_A * rho*v_A^2 * L) = delta/(M_A*L) = 1")
    print("    => Energy roughly balances (magnetic in ~ kinetic out)")

    # (c) Missing energy
    print("\n(c) Energy partition beyond kinetic energy:")
    print("    Not all magnetic energy goes to bulk kinetic energy.")
    print("    The 'missing' energy goes to:")
    print("    1. Thermal heating (entropy generation at X-point and shocks)")
    print("    2. Particle acceleration (non-thermal tails, especially in")
    print("       collisionless reconnection)")
    print("    3. Waves (Alfven waves, whistler waves in Hall regime)")
    print("    Typical partition: ~50% thermal, ~30% kinetic, ~20% non-thermal")


def exercise_9():
    """
    Problem 9: GEM Reconnection Challenge

    Summary of GEM Challenge setup and results
    """
    print("(a) GEM Reconnection Challenge setup:")
    print("    - Harris current sheet equilibrium: B_x = B0 * tanh(y/lambda)")
    print("    - Uniform guide field B_z (optional)")
    print("    - Initial density: n(y) = n0 * sech^2(y/lambda) + n_bg")
    print("    - Small perturbation to seed reconnection")
    print("    - 2D periodic domain in x, bounded in y")
    print("    - Parameters: lambda = 0.5*d_i, L_x = 25.6*d_i, L_y = 12.8*d_i")

    print("\n(b) Key findings on reconnection rate:")
    print("    - Resistive MHD (with uniform eta): M_A ~ S^(-1/2) (Sweet-Parker)")
    print("      Reconnection rate depends on eta and is slow for large S")
    print("    - Hall MHD: M_A ~ 0.1, independent of eta for large S")
    print("    - Two-fluid: M_A ~ 0.1, similar to Hall MHD")
    print("    - Hybrid (kinetic ions, fluid electrons): M_A ~ 0.1")
    print("    - Full PIC: M_A ~ 0.1")
    print("    All non-resistive models converge to M_A ~ 0.1")

    print("\n(c) How resistive MHD differs:")
    print("    - Resistive MHD: single-fluid, no scale separation")
    print("      -> Long, thin SP-like current sheet, slow reconnection")
    print("    - Hall MHD/kinetic: two-scale structure (ion + electron layers)")
    print("      -> Compact diffusion region, fast reconnection")
    print("    - Key difference: Hall term enables ion-electron decoupling")
    print("      at scales < d_i, creating the fast reconnection mechanism")
    print("    - Quadrupolar Hall field is absent in resistive MHD")


def exercise_10():
    """
    Problem 10: Observational Signatures

    Reconnection signatures in magnetotail and MMS mission
    """
    print("(a) Three observational signatures of magnetotail reconnection:")
    print("    1. Bi-directional plasma jets: Fast earthward and tailward flows")
    print("       at ~v_A emanating from the X-line. Observed as flow reversals")
    print("       in the BPS (plasma sheet boundary layer).")
    print("    2. Quadrupolar Hall magnetic field: Out-of-plane B_y component")
    print("       with characteristic quadrupolar pattern around the X-line.")
    print("       Sign of B_y depends on position relative to reconnection site.")
    print("    3. Energetic particle acceleration: Beams of energetic electrons")
    print("       and ions streaming away from the X-line along separatrices.")
    print("       Power-law energy spectra with spectral index ~3-6.")

    print("\n(b) MMS measurement of Hall fields:")
    print("    - MMS (Magnetospheric Multiscale) uses 4 spacecraft in a tetrahedron")
    print("    - Spacecraft separation: ~10-100 km (electron scale)")
    print("    - High-cadence magnetometers: 128 samples/s (FGM)")
    print("    - Search coil: up to 8192 samples/s (SCM)")
    print("    - Electric field instruments (EDP, SDP): dual-probe measurements")
    print("    - 4-point measurements allow computation of curl(B) = mu0*J")
    print("    - Hall fields identified by their quadrupolar B_z structure")
    print("    - Electron distribution functions measured by FPI at 30 ms cadence")

    print("\n(c) Crossing ion diffusion region but not electron:")
    print("    The spacecraft would observe:")
    print("    - Ion flow reversal (jet): bulk ion velocity changes direction")
    print("    - Ion demagnetization: E + v_i x B != 0")
    print("    - But electrons still frozen in: E + v_e x B ~ 0")
    print("    - Hall electric field (bipolar E_N normal to current sheet)")
    print("    - Enhanced |J| but not the very intense electron-scale currents")
    print("    - Ion distribution function: counter-streaming beams")
    print("    - Would NOT see: electron demagnetization, crescent-shaped")
    print("      electron distributions, or sub-d_e scale structures")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Sweet-Parker Scaling ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Petschek vs Sweet-Parker ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Hall Reconnection ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: X-point Field ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Reconnection Electric Field ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Quadrupolar Hall Field ===")
    print("=" * 60)
    exercise_6()

    print("\n" + "=" * 60)
    print("=== Exercise 7: Simulation Analysis ===")
    print("=" * 60)
    exercise_7()

    print("\n" + "=" * 60)
    print("=== Exercise 8: Energy Conversion ===")
    print("=" * 60)
    exercise_8()

    print("\n" + "=" * 60)
    print("=== Exercise 9: GEM Reconnection Challenge ===")
    print("=" * 60)
    exercise_9()

    print("\n" + "=" * 60)
    print("=== Exercise 10: Observational Signatures ===")
    print("=" * 60)
    exercise_10()

    print("\nAll exercises completed!")
