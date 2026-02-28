"""
Exercises for Lesson 09: Dynamo Theory
Topic: MHD
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def exercise_1():
    """
    Problem 1: Free Decay Timescale

    Earth's core: L = 1e6 m, eta = 2 m^2/s
    Sun's convection zone: L = 2e8 m, eta = 1e4 m^2/s (turbulent)
    """
    year = 3.156e7  # s

    cases = [
        ("Earth's core", 1e6, 2.0, 4.5e9),
        ("Sun's convection zone", 2e8, 1e4, 4.6e9),
    ]

    print("Free decay timescale: tau_d = L^2 / (pi^2 * eta)")
    print("(fundamental mode of diffusion equation)\n")

    for name, L, eta, age_years in cases:
        tau_d = L**2 / (np.pi**2 * eta)
        tau_years = tau_d / year
        age_s = age_years * year

        print(f"  {name}:")
        print(f"    L = {L:.1e} m, eta = {eta} m^2/s")
        print(f"    tau_d = {tau_d:.4e} s")
        print(f"    = {tau_years:.4e} years")
        print(f"    Age: ~{age_years:.1e} years")
        ratio = age_s / tau_d
        print(f"    Age / tau_d = {ratio:.2e}")
        if ratio > 10:
            print(f"    Age >> tau_d => DYNAMO REQUIRED to sustain the field")
        else:
            print(f"    Age ~ tau_d => Primordial field could possibly survive")
        print()


def exercise_2():
    """
    Problem 2: Magnetic Reynolds Number

    Solar convection zone: v ~ 100 m/s, L ~ 1e8 m, eta ~ 1e4 m^2/s
    """
    v = 100.0     # m/s
    L = 1e8       # m
    eta = 1e4     # m^2/s (turbulent diffusivity)

    Rm = v * L / eta
    print(f"Magnetic Reynolds number:")
    print(f"    v = {v} m/s, L = {L:.1e} m, eta = {eta:.0e} m^2/s")
    print(f"    Rm = v*L/eta = {v}*{L:.1e}/{eta:.0e} = {Rm:.2e}")
    print()

    Rm_crit = 10  # typical critical Rm for dynamo action
    if Rm > Rm_crit:
        print(f"    Rm = {Rm:.0f} >> Rm_crit ~ {Rm_crit}")
        print(f"    YES: Dynamo action is possible (Rm well above critical)")
    else:
        print(f"    Rm = {Rm:.0f} < Rm_crit ~ {Rm_crit}")
        print(f"    NO: Rm is too low for dynamo action")

    print(f"\n    Note: The critical Rm depends on the flow geometry.")
    print(f"    For most astrophysical flows, Rm_crit ~ 10-100.")
    print(f"    With Rm = {Rm:.0f}, the solar convection zone is well")
    print(f"    above the threshold for dynamo action.")


def exercise_3():
    """
    Problem 3: Cowling's Theorem

    Show that a purely toroidal field B = B_phi(r,z,t) e_phi
    requires poloidal field sources
    """
    print("Cowling's theorem demonstration:")
    print()
    print("Consider B = B_phi(r,z,t) e_phi (purely toroidal).")
    print()
    print("The induction equation in cylindrical coordinates (r,phi,z):")
    print("  dB_phi/dt = [curl(v x B)]_phi + eta * [nabla^2 B - B_phi/r^2]")
    print()
    print("For v x B with B = B_phi e_phi:")
    print("  v x B = v_r*B_phi*e_z - v_z*B_phi*e_r")
    print("  (only r and z components)")
    print()
    print("  curl(v x B)_phi = d(v_r*B_phi)/dz - d(-v_z*B_phi)/dr")
    print()
    print("This only involves B_phi itself -- the phi component of the")
    print("induction equation is CLOSED in terms of B_phi alone.")
    print()
    print("However, for a steady state (dB_phi/dt = 0):")
    print("  0 = curl(v x B)_phi + eta * [nabla^2 B_phi - B_phi/r^2]")
    print()
    print("The diffusion term (last term) always causes B_phi to DECAY.")
    print("For the advection term to balance, v x B must have appropriate")
    print("structure. But at a magnetic null (B_phi = 0 on the axis),")
    print("the advection term also vanishes, so B_phi decays there.")
    print()
    print("CONCLUSION: A purely toroidal (or purely poloidal) axisymmetric")
    print("field cannot be maintained by dynamo action against ohmic decay.")
    print("A self-sustaining dynamo requires BOTH poloidal and toroidal")
    print("components that regenerate each other:")
    print("  - Differential rotation (Omega-effect): poloidal -> toroidal")
    print("  - Helical turbulence (alpha-effect): toroidal -> poloidal")


def exercise_4():
    """
    Problem 4: Ponomarenko Dynamo Growth Rate

    a = 1 m, U = 1 m/s, Omega = 1 rad/s, eta = 0.05 m^2/s, k = 1 m^-1
    """
    a = 1.0       # m
    U = 1.0       # m/s
    Omega = 1.0   # rad/s
    eta = 0.05    # m^2/s
    k = 1.0       # m^-1

    # Ponomarenko dynamo: helical flow inside a cylinder
    # Simplified growth rate estimate:
    # gamma ~ (Omega * U * a) / (eta) - eta * k^2 (very rough)
    # More precisely, the growth rate depends on the eigenvalue of the
    # boundary value problem

    # Magnetic Reynolds number
    Rm = Omega * a**2 / eta
    Rm_z = U * a / eta

    print(f"Ponomarenko dynamo parameters:")
    print(f"    a = {a} m, U = {U} m/s, Omega = {Omega} rad/s")
    print(f"    eta = {eta} m^2/s, k = {k} m^-1")
    print(f"    Rm (rotational) = Omega*a^2/eta = {Rm:.1f}")
    print(f"    Rm (axial) = U*a/eta = {Rm_z:.1f}")
    print(f"    Total Rm = sqrt(Rm_rot^2 + Rm_z^2) = {np.sqrt(Rm**2 + Rm_z**2):.1f}")

    # Simplified growth rate using the scaling:
    # For the Ponomarenko dynamo, onset at Rm_crit ~ 17.7
    # Growth rate near onset: gamma ~ eta/a^2 * (Rm - Rm_crit)/Rm_crit
    Rm_total = np.sqrt((Omega * a)**2 + U**2) * a / eta
    Rm_crit = 17.7  # approximate critical Rm for Ponomarenko

    if Rm_total > Rm_crit:
        # Rough estimate: gamma scales with (Rm - Rm_crit)
        gamma_est = (eta / a**2) * (Rm_total - Rm_crit) / Rm_crit
        print(f"\n    Total Rm = {Rm_total:.1f} > Rm_crit = {Rm_crit}")
        print(f"    DYNAMO ACTIVE")
        print(f"    Estimated growth rate:")
        print(f"    gamma ~ (eta/a^2) * (Rm - Rm_crit)/Rm_crit")
        print(f"    = ({eta}/{a}^2) * ({Rm_total:.1f} - {Rm_crit})/{Rm_crit}")
        print(f"    = {gamma_est:.4f} s^-1")
        print(f"    Growth time: tau = 1/gamma ~ {1/gamma_est:.2f} s")
    else:
        print(f"\n    Total Rm = {Rm_total:.1f} < Rm_crit = {Rm_crit}")
        print(f"    BELOW THRESHOLD: No dynamo action")
        print(f"    Field decays on diffusion timescale: tau_d = a^2/eta = {a**2/eta:.1f} s")


def exercise_5():
    """
    Problem 5: Alpha-Effect Estimate

    u_rms = 10 m/s, tau_c = 1e4 s, <h> = 1e-3 m/s^2
    """
    u_rms = 10.0      # m/s
    tau_c = 1e4        # s (correlation time)
    helicity = 1e-3    # m/s^2 (kinetic helicity)

    # alpha ~ -(1/3) * tau_c * <v . curl(v)> = -(1/3) * tau_c * helicity
    alpha = -(1.0 / 3) * tau_c * helicity
    print(f"Alpha-effect estimate:")
    print(f"    u_rms = {u_rms} m/s")
    print(f"    tau_c = {tau_c:.1e} s")
    print(f"    <v . curl(v)> = {helicity:.1e} m/s^2")
    print(f"")
    print(f"    alpha = -(1/3) * tau_c * <v.curl(v)>")
    print(f"    = -(1/3) * {tau_c:.1e} * {helicity:.1e}")
    print(f"    = {alpha:.4f} m/s")
    print(f"    |alpha| = {abs(alpha):.4f} m/s")
    print(f"")
    print(f"    Physical meaning: alpha measures the average EMF generated")
    print(f"    by helical turbulence parallel to the mean magnetic field.")
    print(f"    It converts toroidal field to poloidal field (or vice versa).")
    print(f"    The sign determines the direction of the alpha-effect.")


def exercise_6():
    """
    Problem 6: Alpha-Omega Dynamo Number

    R = 1e8 m, alpha = 1 m/s, Delta_Omega = 1e-6 rad/s, eta_eff = 1e4 m^2/s
    """
    R = 1e8            # m
    alpha = 1.0        # m/s
    Delta_Omega = 1e-6  # rad/s (differential rotation)
    eta_eff = 1e4       # m^2/s (effective turbulent diffusivity)

    # Dynamo number D = alpha * Delta_Omega * R^3 / eta_eff^2
    D = alpha * Delta_Omega * R**3 / eta_eff**2

    # Critical dynamo number (typical: |D_crit| ~ 1-100, depends on geometry)
    D_crit = 10.0  # typical for alpha-omega in spherical shell

    print(f"Alpha-Omega dynamo number:")
    print(f"    alpha = {alpha} m/s")
    print(f"    Delta_Omega = {Delta_Omega:.1e} rad/s")
    print(f"    R = {R:.1e} m")
    print(f"    eta_eff = {eta_eff:.0e} m^2/s")
    print(f"")
    print(f"    D = alpha * Delta_Omega * R^3 / eta_eff^2")
    print(f"    = {alpha} * {Delta_Omega:.1e} * ({R:.1e})^3 / ({eta_eff:.0e})^2")
    print(f"    = {D:.2e}")
    print(f"")
    print(f"    Critical dynamo number: |D_crit| ~ {D_crit}")
    if abs(D) > D_crit:
        print(f"    |D| = {abs(D):.1e} >> D_crit")
        print(f"    DYNAMO EXPECTED: The dynamo number is supercritical")
        print(f"    The magnetic field will grow exponentially until")
        print(f"    nonlinear saturation (e.g., alpha-quenching) occurs.")
    else:
        print(f"    |D| = {abs(D):.1e} < D_crit")
        print(f"    NO DYNAMO: Below critical threshold")

    if D < 0:
        print(f"    Sign of D is negative: oscillatory (propagating wave) solution")
        print(f"    This produces equatorward-propagating activity belts")
        print(f"    (like the solar butterfly diagram)")
    else:
        print(f"    Sign of D is positive: steady (non-oscillatory) dynamo")


def exercise_7():
    """
    Problem 7: Equipartition Field

    rho = 1e3 kg/m^3, v = 100 m/s
    """
    rho = 1e3      # kg/m^3
    v = 100.0      # m/s
    mu0 = 4 * np.pi * 1e-7

    # Equipartition: B_eq^2 / (2*mu0) = rho * v^2 / 2
    # B_eq = sqrt(mu0 * rho) * v
    B_eq = np.sqrt(mu0 * rho) * v
    print(f"Equipartition magnetic field:")
    print(f"    rho = {rho:.0e} kg/m^3, v = {v} m/s")
    print(f"")
    print(f"    At equipartition: B^2/(2*mu0) = rho*v^2/2")
    print(f"    B_eq = sqrt(mu0 * rho) * v")
    print(f"    = sqrt({mu0:.4e} * {rho:.0e}) * {v}")
    print(f"    = {B_eq:.4f} T")
    print(f"    = {B_eq*1e4:.2f} Gauss")
    print(f"")
    E_kin = 0.5 * rho * v**2
    E_mag = B_eq**2 / (2 * mu0)
    print(f"    Kinetic energy density: E_kin = {E_kin:.4e} J/m^3")
    print(f"    Magnetic energy density: E_mag = {E_mag:.4e} J/m^3")
    print(f"    Ratio E_mag/E_kin = {E_mag/E_kin:.4f} (should be 1)")
    print(f"")
    print(f"    Interpretation: The dynamo amplifies B until the magnetic")
    print(f"    energy reaches the kinetic energy of the driving flow.")
    print(f"    Beyond this, the Lorentz force suppresses the flow")
    print(f"    (back-reaction), limiting further amplification.")


def exercise_8():
    """
    Problem 8: Alpha-Quenching

    Modify alpha-omega 1D code to include alpha-quenching:
    alpha(B) = alpha0 / (1 + B^2/B_eq^2)
    """
    # Simple 1D alpha-omega dynamo model
    # dA/dt = alpha(B)*B + eta*d^2A/dr^2    (poloidal from alpha-effect)
    # dB/dt = G*dA/dr + eta*d^2B/dr^2        (toroidal from omega-effect)
    # where G = r*dOmega/dr is the shear

    Nr = 50
    r = np.linspace(0, 1, Nr)
    dr = r[1] - r[0]
    eta = 0.01       # diffusivity
    alpha0 = 1.0     # alpha parameter
    G = 10.0         # shear (omega-effect)
    B_eq = 1.0       # equipartition field

    # Initial conditions: small seed field
    A = 0.01 * np.sin(np.pi * r)
    B = 0.01 * np.cos(np.pi * r)

    dt = 0.3 * dr**2 / eta  # CFL condition
    Nt = 5000
    t_vals = []
    B_max_vals = []
    B_rms_vals = []

    for n in range(Nt):
        # Alpha-quenching
        alpha = alpha0 / (1 + B**2 / B_eq**2)

        # Diffusion (centered differences)
        d2A = np.zeros(Nr)
        d2B = np.zeros(Nr)
        d2A[1:-1] = (A[2:] - 2*A[1:-1] + A[:-2]) / dr**2
        d2B[1:-1] = (B[2:] - 2*B[1:-1] + B[:-2]) / dr**2

        # Shear term dA/dr
        dAdr = np.zeros(Nr)
        dAdr[1:-1] = (A[2:] - A[:-2]) / (2*dr)

        # Update
        A_new = A + dt * (alpha * B + eta * d2A)
        B_new = B + dt * (G * dAdr + eta * d2B)

        # Boundary conditions (zero at boundaries)
        A_new[0] = 0
        A_new[-1] = 0
        B_new[0] = 0
        B_new[-1] = 0

        A = A_new
        B = B_new

        if n % 50 == 0:
            t_vals.append(n * dt)
            B_max_vals.append(np.max(np.abs(B)))
            B_rms_vals.append(np.sqrt(np.mean(B**2)))

    t_vals = np.array(t_vals)
    B_max_vals = np.array(B_max_vals)
    B_rms_vals = np.array(B_rms_vals)

    print("Alpha-quenching dynamo simulation:")
    print(f"    Grid: Nr = {Nr}, Nt = {Nt}")
    print(f"    Parameters: alpha0 = {alpha0}, G = {G}, eta = {eta}")
    print(f"    B_eq = {B_eq}")
    print(f"")
    print(f"    Initial B_max = {0.01:.4f}")
    print(f"    Final B_max = {B_max_vals[-1]:.4f}")
    print(f"    Final B_rms = {B_rms_vals[-1]:.4f}")
    print(f"")

    # Determine if field saturated
    if len(B_max_vals) > 10:
        recent = B_max_vals[-10:]
        variation = (np.max(recent) - np.min(recent)) / np.mean(recent)
        if variation < 0.1:
            print(f"    Field appears SATURATED (variation {variation*100:.1f}%)")
            print(f"    Saturation level ~ {np.mean(recent):.4f} * B_eq")
        else:
            print(f"    Field still evolving (variation {variation*100:.1f}%)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.semilogy(t_vals, B_max_vals, 'b-', linewidth=2, label='$B_{max}$')
    ax1.semilogy(t_vals, B_rms_vals, 'r-', linewidth=2, label='$B_{rms}$')
    ax1.axhline(y=B_eq, color='k', linestyle='--', label='$B_{eq}$')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Field strength', fontsize=12)
    ax1.set_title('Dynamo Field Growth with Alpha-Quenching', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Final profiles
    ax2.plot(r, A, 'b-', linewidth=2, label='A (poloidal)')
    ax2.plot(r, B, 'r-', linewidth=2, label='B (toroidal)')
    ax2.set_xlabel('r', fontsize=12)
    ax2.set_ylabel('Field amplitude', fontsize=12)
    ax2.set_title('Saturated Field Profiles', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/ex09_alpha_quenching.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex09_alpha_quenching.png")


def exercise_9():
    """
    Problem 9: Butterfly Diagram Analysis

    From the alpha-omega simulation, measure period and propagation
    """
    # Run a longer simulation with oscillatory behavior
    Nr = 80
    r = np.linspace(0.1, 1.0, Nr)  # avoid r=0 singularity
    dr = r[1] - r[0]
    eta = 0.01
    alpha0 = 0.5
    G = -20.0  # negative shear for equatorward propagation
    B_eq = 1.0

    A = 0.01 * np.sin(np.pi * (r - 0.1) / 0.9)
    B = 0.01 * np.cos(np.pi * (r - 0.1) / 0.9)

    dt = 0.3 * dr**2 / eta
    Nt = 20000
    save_every = 100

    # Store time-radius diagram
    t_save = []
    B_save = []

    for n in range(Nt):
        alpha = alpha0 / (1 + B**2 / B_eq**2)

        d2A = np.zeros(Nr)
        d2B = np.zeros(Nr)
        d2A[1:-1] = (A[2:] - 2*A[1:-1] + A[:-2]) / dr**2
        d2B[1:-1] = (B[2:] - 2*B[1:-1] + B[:-2]) / dr**2

        dAdr = np.zeros(Nr)
        dAdr[1:-1] = (A[2:] - A[:-2]) / (2*dr)

        A_new = A + dt * (alpha * B + eta * d2A)
        B_new = B + dt * (G * dAdr + eta * d2B)

        A_new[0] = 0; A_new[-1] = 0
        B_new[0] = 0; B_new[-1] = 0

        A = A_new
        B = B_new

        if n % save_every == 0:
            t_save.append(n * dt)
            B_save.append(B.copy())

    t_save = np.array(t_save)
    B_save = np.array(B_save)

    # Measure period from B at midpoint
    mid_idx = Nr // 2
    B_mid = B_save[:, mid_idx]

    # Find zero crossings to estimate period
    crossings = []
    for i in range(1, len(B_mid)):
        if B_mid[i-1] * B_mid[i] < 0:
            crossings.append(t_save[i])

    if len(crossings) > 2:
        half_periods = np.diff(crossings)
        period = 2 * np.mean(half_periods)
        print(f"Butterfly diagram analysis:")
        print(f"    Parameters: alpha0={alpha0}, G={G}, eta={eta}")
        print(f"    Simulation time: {t_save[-1]:.2f}")
        print(f"    Number of zero crossings: {len(crossings)}")
        print(f"    Average half-period: {np.mean(half_periods):.4f}")
        print(f"    Estimated full period: {period:.4f}")
    else:
        period = None
        print(f"Butterfly diagram analysis:")
        print(f"    Could not determine period (insufficient crossings)")

    # Propagation direction
    print(f"\n    Shear G = {G}")
    if G < 0:
        print(f"    With negative G and positive alpha: equatorward propagation")
        print(f"    (Parker-Yoshimura sign rule: propagation ~ -alpha*G direction)")
    else:
        print(f"    With positive G and positive alpha: poleward propagation")

    # Dependence on parameters
    print(f"\n    Parameter dependence:")
    print(f"    - Period ~ eta / (alpha * |G|)^(1/2) (from alpha-omega dispersion)")
    print(f"    - Increasing |C_Omega| (= G*R^2/eta): shorter period, stronger shear")
    print(f"    - Increasing |C_alpha| (= alpha*R/eta): shorter period, stronger alpha")

    # Plot butterfly diagram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Time-radius (butterfly) diagram
    im = ax1.pcolormesh(t_save, r, B_save.T, cmap='RdBu_r', shading='auto',
                         vmin=-np.max(np.abs(B_save))*0.5,
                         vmax=np.max(np.abs(B_save))*0.5)
    plt.colorbar(im, ax=ax1, label='$B_\\phi$ (toroidal)')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Radius r', fontsize=12)
    ax1.set_title('Butterfly Diagram (Time-Latitude)', fontsize=14)

    # Time series at midpoint
    ax2.plot(t_save, B_mid, 'b-', linewidth=1)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('$B_\\phi$ at r=0.55', fontsize=12)
    ax2.set_title('Toroidal Field at Mid-Radius', fontsize=14)
    ax2.grid(True, alpha=0.3)
    if period is not None:
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('/tmp/ex09_butterfly.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex09_butterfly.png")


def exercise_10():
    """
    Problem 10: Simple 2D Kinematic Dynamo (Roberts Flow)

    Roberts flow: u_x = U*sin(k*y), u_y = U*sin(k*x), u_z = U*(cos(k*x) + cos(k*y))
    Solve for magnetic field growth rate
    """
    print("2D Kinematic Roberts Flow Dynamo:")
    print()
    print("Roberts flow (ABC flow variant):")
    print("  u_x = U*sin(k*y)")
    print("  u_y = U*sin(k*x)")
    print("  u_z = U*(cos(k*x) + cos(k*y))")
    print()
    print("This is a 2D flow with helicity that produces dynamo action.")
    print()

    # Simple mean-field estimate of growth rate
    U = 1.0
    k = 2 * np.pi
    eta_vals = np.logspace(-2, 0, 50)

    # For Roberts flow, the dynamo onset occurs at Rm_crit ~ 1
    # Growth rate scales as: gamma ~ U*k*(1 - eta/eta_crit)
    # More precisely: gamma ~ alpha_eff^2 / (eta + beta_eff) - eta*k^2
    # where alpha_eff ~ U/(k*eta) for large Rm

    Rm_vals = U / (k * eta_vals)
    Rm_crit = 1.0  # approximate

    # Growth rate model: gamma ~ (U^2*k)/(eta) - eta*k^2 for simple estimate
    # Better: gamma is the eigenvalue of the mean-field problem
    # Using asymptotic scaling: gamma ~ (alpha_eff * k_mean)^2 / eta_eff - eta * k^2
    # For large Rm: gamma ~ U^2/(eta) * C1 - eta*k^2 * C2
    gamma_vals = (U**2 * k**2 / eta_vals) * 0.01 - eta_vals * k**2
    gamma_vals = np.maximum(gamma_vals, -10)  # clip for plotting

    print(f"Parameters: U = {U}, k = {k:.2f}")
    print(f"Critical Rm ~ {Rm_crit}")
    print()

    # Print table
    print(f"  {'eta':>8} {'Rm':>8} {'gamma':>10}")
    for eta, Rm, gamma in zip(eta_vals[::10], Rm_vals[::10], gamma_vals[::10]):
        print(f"  {eta:8.4f} {Rm:8.2f} {gamma:10.4f}")

    print()
    print("Literature values for Roberts flow dynamo:")
    print("  Rm_crit ~ 1-2 (depends on normalization)")
    print("  Growth rate increases with Rm above threshold")
    print("  Saturates at gamma ~ O(U*k) for large Rm")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.semilogx(Rm_vals, gamma_vals, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axvline(x=Rm_crit, color='r', linestyle='--', label=f'$Rm_c$ ~ {Rm_crit}')
    ax1.set_xlabel('$Rm = U/(k\\eta)$', fontsize=12)
    ax1.set_ylabel('Growth rate $\\gamma$', fontsize=12)
    ax1.set_title('Roberts Flow Dynamo: Growth Rate', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Roberts flow pattern
    x = np.linspace(0, 2*np.pi/k, 100)
    y = np.linspace(0, 2*np.pi/k, 100)
    X, Y = np.meshgrid(x, y)
    Ux = U * np.sin(k * Y)
    Uy = U * np.sin(k * X)
    Uz = U * (np.cos(k * X) + np.cos(k * Y))
    speed = np.sqrt(Ux**2 + Uy**2)

    ax2.streamplot(X, Y, Ux, Uy, color=speed, cmap='viridis', density=2)
    im = ax2.pcolormesh(X, Y, Uz, cmap='RdBu_r', alpha=0.3, shading='auto')
    plt.colorbar(im, ax=ax2, label='$u_z$ (helical component)')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Roberts Flow Pattern', fontsize=14)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('/tmp/ex09_roberts_dynamo.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex09_roberts_dynamo.png")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Free Decay Timescale ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Magnetic Reynolds Number ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Cowling's Theorem ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Ponomarenko Growth Rate ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Alpha-Effect Estimate ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Alpha-Omega Dynamo Number ===")
    print("=" * 60)
    exercise_6()

    print("\n" + "=" * 60)
    print("=== Exercise 7: Equipartition Field ===")
    print("=" * 60)
    exercise_7()

    print("\n" + "=" * 60)
    print("=== Exercise 8: Alpha-Quenching Simulation ===")
    print("=" * 60)
    exercise_8()

    print("\n" + "=" * 60)
    print("=== Exercise 9: Butterfly Diagram Analysis ===")
    print("=" * 60)
    exercise_9()

    print("\n" + "=" * 60)
    print("=== Exercise 10: Roberts Flow Dynamo ===")
    print("=" * 60)
    exercise_10()

    print("\nAll exercises completed!")
