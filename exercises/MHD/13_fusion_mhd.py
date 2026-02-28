"""
Lesson 13: Fusion MHD
Topic: MHD
Description: Exercises on tokamak stability, Troyon beta limit, safety factor,
             sawtooth oscillations, Greenwald density, ELM heat flux,
             disruption forces, NTM island evolution, and RFP field reversal.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import j0, j1  # Bessel functions
from scipy.optimize import brentq


def exercise_1():
    """Troyon Limit.

    For a tokamak with I_p = 10 MA, a = 1.5 m, B_0 = 4 T,
    calculate the maximum achievable beta using the Troyon limit (beta_N = 3).
    What is the corresponding plasma pressure?
    """
    I_p = 10e6           # A (10 MA)
    a = 1.5              # m (minor radius)
    B_0 = 4.0            # T (toroidal field)
    beta_N = 3.0         # Troyon coefficient (% m T / MA)
    mu_0 = 4 * np.pi * 1e-7

    # Troyon limit: beta_max(%) = beta_N * I_p(MA) / (a(m) * B_0(T))
    I_MA = I_p / 1e6
    beta_max_percent = beta_N * I_MA / (a * B_0)
    beta_max = beta_max_percent / 100.0

    # Corresponding plasma pressure: beta = 2*mu_0*<p> / B_0^2
    # <p> = beta * B_0^2 / (2*mu_0)
    p_avg = beta_max * B_0**2 / (2 * mu_0)

    # Convert to atmospheres for reference
    p_atm = p_avg / 101325

    print(f"  Tokamak parameters:")
    print(f"    I_p = {I_MA:.0f} MA")
    print(f"    a = {a} m")
    print(f"    B_0 = {B_0} T")
    print(f"    beta_N = {beta_N}")
    print(f"  Troyon limit: beta_max = beta_N * I_p / (a * B_0)")
    print(f"  beta_max = {beta_N} * {I_MA:.0f} / ({a} * {B_0}) = {beta_max_percent:.1f}%")
    print(f"  beta_max = {beta_max:.4f}")
    print(f"  Average plasma pressure: <p> = beta * B^2 / (2*mu_0)")
    print(f"  <p> = {p_avg:.3e} Pa = {p_atm:.1f} atm")
    print(f"  For ITER: beta ~ 2-3% is the target operating point.")


def exercise_2():
    """Safety Factor.

    A tokamak has R = 3 m, a = 1 m, B_0 = 5 T, I_p = 5 MA.
    Calculate the edge safety factor q_a. Is this at risk of disruption (q_a < 2)?
    """
    R = 3.0              # m (major radius)
    a = 1.0              # m (minor radius)
    B_0 = 5.0            # T (toroidal field)
    I_p = 5e6            # A
    mu_0 = 4 * np.pi * 1e-7

    # Edge safety factor for circular cross-section with uniform current:
    # q_a = 2*pi*a^2*B_0 / (mu_0*R*I_p)
    # (from q = r*B_phi / (R*B_theta), B_theta(a) = mu_0*I_p/(2*pi*a))
    q_a = 2 * np.pi * a**2 * B_0 / (mu_0 * R * I_p)

    # Safety factor profile for flat current: q(r) = q_a * (r/a)^2
    # Central safety factor for uniform current: q_0 = q_a * 0 -> q_0 is finite
    # Actually q(r) = q_a * (r/a)^2 only for r/a <=1, so q_0 approaches q_a*(0)^2
    # For uniform j_phi: q(r) = 2*B_0*r^2 / (mu_0*R*j_phi*r^2) which is constant = q_a
    # Wait, for uniform current: q(r) = r*B_phi/(R*B_theta(r))
    # B_theta(r) = mu_0*j*r/2, j = I_p/(pi*a^2)
    # q(r) = r*B_0 / (R * mu_0*I_p*r/(2*pi*a^2)) = 2*pi*a^2*B_0/(mu_0*R*I_p) = q_a for all r
    # So flat current -> flat q profile = q_a everywhere

    print(f"  Tokamak parameters:")
    print(f"    R = {R} m, a = {a} m, B_0 = {B_0} T, I_p = {I_p / 1e6:.0f} MA")
    print(f"  Edge safety factor:")
    print(f"    q_a = 2*pi*a^2*B_0 / (mu_0*R*I_p)")
    print(f"    q_a = {q_a:.2f}")
    print(f"  For flat current profile: q(r) = q_a = {q_a:.2f} everywhere")
    print()

    if q_a < 2:
        print(f"  WARNING: q_a = {q_a:.2f} < 2 => HIGH RISK of disruption!")
        print(f"  The Kruskal-Shafranov limit requires q > 1 for stability.")
        print(f"  Practical operation requires q_a > 2 (typically q_a > 3).")
    elif q_a < 3:
        print(f"  q_a = {q_a:.2f}: borderline stability. q_a > 3 preferred.")
    else:
        print(f"  q_a = {q_a:.2f} > 3: good MHD stability margin.")

    # Plot q profile for peaked current
    r_norm = np.linspace(0, 1, 100)
    # For peaked current j(r) = j_0*(1-(r/a)^2): q(r) = q_a*(r/a)^2 / (1-(1-(r/a)^2)^2)
    # Simplified: q(r) = q_0 / (1 - (1-q_0/q_a)*(r/a)^2) with q_0 = q_a/2 (for parabolic j)
    q_0 = q_a / 2.0
    q_profile = q_0 / (1 - (1 - q_0 / q_a) * r_norm**2 + 1e-10)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_norm, np.full_like(r_norm, q_a), 'b-', linewidth=2, label='Flat current')
    ax.plot(r_norm, q_profile, 'r--', linewidth=2, label='Peaked current')
    ax.axhline(1.0, color='red', linestyle=':', alpha=0.7, label='q = 1 (kink limit)')
    ax.axhline(2.0, color='orange', linestyle=':', alpha=0.7, label='q = 2 (tearing)')
    ax.set_xlabel('r / a', fontsize=12)
    ax.set_ylabel('q(r)', fontsize=12)
    ax.set_title(f'Safety Factor Profile (q_a = {q_a:.2f})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(q_a * 1.5, 4))
    plt.tight_layout()
    plt.savefig('13_safety_factor.png', dpi=150)
    plt.close()
    print("  Plot saved to 13_safety_factor.png")


def exercise_3():
    """Sawtooth Period.

    Estimate the sawtooth period for a plasma with a = 1 m, T_e = 3 keV,
    n_e = 5e19 m^-3, B = 3 T.
    """
    a = 1.0              # m
    T_keV = 3.0          # keV
    T_eV = T_keV * 1e3   # eV
    T_J = T_eV * 1.602e-19  # Joules
    n_e = 5e19           # m^-3
    B = 3.0              # T
    mu_0 = 4 * np.pi * 1e-7
    e = 1.602e-19        # C
    m_e = 9.109e-31      # kg

    # Sawtooth period ~ resistive time / S^(1/3) (Kadomtsev reconnection)
    # Or empirically: tau_saw ~ tau_heat ~ a^2 * n_e / (n * chi)

    # Spitzer resistivity: eta = 1.65e-9 * Z_eff * ln(Lambda) / T_eV^(3/2) [Ohm*m]
    Z_eff = 1.5
    ln_Lambda = 17.0
    eta_spitzer = 1.65e-9 * Z_eff * ln_Lambda / T_eV**1.5  # Ohm*m

    # Resistive diffusion time: tau_R = mu_0 * a^2 / eta
    tau_R = mu_0 * a**2 / eta_spitzer

    # Alfven time: tau_A = a / v_A
    rho = n_e * 2 * 1.67e-27  # assume D-T, mass ~ 2*m_p per electron
    v_A = B / np.sqrt(mu_0 * rho)
    tau_A = a / v_A

    # Lundquist number
    S = tau_R / tau_A

    # Kadomtsev reconnection time: tau_K ~ tau_R * S^(-1/3) ~ (tau_R^2 * tau_A)^(1/3)
    # Or: tau_saw ~ S^(1/3) * tau_A
    tau_K = (tau_R**2 * tau_A)**(1.0 / 3.0)
    # Alternative: tau_saw ~ tau_R / S^(1/3) = S^(2/3) * tau_A
    tau_saw_alt = S**(2.0 / 3.0) * tau_A

    print(f"  Plasma parameters:")
    print(f"    a = {a} m, T_e = {T_keV} keV, n_e = {n_e:.1e} m^-3, B = {B} T")
    print(f"  Spitzer resistivity: eta = {eta_spitzer:.3e} Ohm*m")
    print(f"  Resistive time: tau_R = mu_0*a^2/eta = {tau_R:.3e} s = {tau_R:.1f} s")
    print(f"  Alfven speed: v_A = {v_A:.3e} m/s = {v_A / 1e3:.0f} km/s")
    print(f"  Alfven time: tau_A = a/v_A = {tau_A:.3e} s")
    print(f"  Lundquist number S = tau_R/tau_A = {S:.3e}")
    print(f"  Kadomtsev reconnection time: tau_K ~ (tau_R^2 * tau_A)^(1/3) = {tau_K:.3f} s")
    print(f"  Sawtooth period estimate: {tau_K * 1e3:.1f} ms")
    print(f"  Observed sawtooth periods: typically 1-100 ms in tokamaks.")

    # Effect of doubling temperature
    T_keV_2 = 2 * T_keV
    T_eV_2 = T_keV_2 * 1e3
    eta_2 = 1.65e-9 * Z_eff * ln_Lambda / T_eV_2**1.5
    tau_R_2 = mu_0 * a**2 / eta_2
    S_2 = tau_R_2 / tau_A
    tau_K_2 = (tau_R_2**2 * tau_A)**(1.0 / 3.0)
    print(f"\n  If T_e doubles to {T_keV_2} keV:")
    print(f"    eta decreases by factor {(T_eV / T_eV_2)**1.5:.2f}")
    print(f"    tau_saw ~ {tau_K_2 * 1e3:.1f} ms (longer, due to lower resistivity)")
    print(f"    Ratio: tau_saw(2T)/tau_saw(T) = {tau_K_2 / tau_K:.2f}")


def exercise_4():
    """Greenwald Density.

    For ITER (I_p = 15 MA, a = 2 m), compute the Greenwald fraction
    at n_e = 1.0e20 m^-3.
    """
    I_p = 15e6           # A
    a = 2.0              # m
    n_e = 1.0e20         # m^-3

    # Greenwald density limit: n_G = I_p / (pi * a^2) [in 10^20 m^-3 when I_p in MA, a in m]
    I_MA = I_p / 1e6
    n_G = I_MA / (np.pi * a**2) * 1e20  # m^-3

    f_G = n_e / n_G

    print(f"  ITER parameters: I_p = {I_MA:.0f} MA, a = {a} m")
    print(f"  Greenwald density limit:")
    print(f"    n_G = I_p(MA) / (pi * a^2) * 10^20 m^-3")
    print(f"    n_G = {I_MA:.0f} / (pi * {a}^2) * 10^20 = {n_G:.3e} m^-3")
    print(f"  Operating density: n_e = {n_e:.1e} m^-3")
    print(f"  Greenwald fraction: f_G = n_e / n_G = {f_G:.3f}")

    if f_G > 1.0:
        print(f"  WARNING: f_G > 1 => above Greenwald limit, disruption risk!")
    elif f_G > 0.8:
        print(f"  f_G = {f_G:.2f}: close to Greenwald limit, careful operation needed.")
    else:
        print(f"  f_G = {f_G:.2f}: safely below the Greenwald limit.")


def exercise_5():
    """ELM Heat Flux.

    An ELM expels W_ELM = 0.5 MJ over wetted area A_wet = 1 m^2
    in tau_ELM = 1 ms. Calculate peak heat flux and compare to material limits.
    """
    W_ELM = 0.5e6        # J (0.5 MJ)
    A_wet = 1.0           # m^2
    tau_ELM = 1e-3        # s (1 ms)

    # Peak heat flux
    q_ELM = W_ELM / (A_wet * tau_ELM)

    # Material limit for tungsten divertor
    q_limit = 10e6  # W/m^2 (10 MW/m^2)

    print(f"  ELM parameters:")
    print(f"    Energy W_ELM = {W_ELM / 1e6:.1f} MJ")
    print(f"    Wetted area A_wet = {A_wet} m^2")
    print(f"    Duration tau_ELM = {tau_ELM * 1e3:.1f} ms")
    print(f"  Peak heat flux: q = W_ELM / (A * tau)")
    print(f"    q = {q_ELM:.3e} W/m^2 = {q_ELM / 1e6:.0f} MW/m^2")
    print(f"  Material limit (tungsten): {q_limit / 1e6:.0f} MW/m^2")
    print(f"  Ratio q_ELM / q_limit = {q_ELM / q_limit:.0f}")

    if q_ELM > q_limit:
        print(f"  MITIGATION REQUIRED: ELM heat flux exceeds material limit by {q_ELM / q_limit:.0f}x!")
        print(f"  Options: RMP coils, pellet pacing, QH-mode, small ELM regimes")
    else:
        print(f"  Heat flux within material limits.")


def exercise_6():
    """Disruption Forces.

    During a disruption, I_p decays from 5 MA to zero in tau_CQ = 100 ms.
    Estimate current quench rate and induced forces.
    """
    I_p = 5e6            # A
    tau_CQ = 100e-3      # s (100 ms)
    R = 3.0              # m (major radius)
    a = 1.0              # m (minor radius)
    mu_0 = 4 * np.pi * 1e-7

    # Current quench rate
    dI_dt = I_p / tau_CQ

    # Plasma inductance: L ~ mu_0 * R * (ln(8*R/a) - 2)
    L = mu_0 * R * (np.log(8 * R / a) - 2.0)

    # Induced loop voltage: V = L * dI/dt
    V_loop = L * dI_dt

    # Magnetic energy: E = (1/2) * L * I^2
    E_mag = 0.5 * L * I_p**2

    # Vertical force on vessel (halo current interaction)
    # F_z ~ mu_0 * I_p * I_halo / (2*pi) ~ simplified estimate
    # More practically: F_z ~ B_external * I_p * 2*pi*R (j x B force)
    # For disruption: F ~ mu_0 * I_p^2 / (4*pi) * 2*pi*R / a (force per unit length * circumference)
    B_pol = mu_0 * I_p / (2 * np.pi * a)
    F_vertical = I_p * B_pol * 2 * np.pi * R  # order of magnitude

    print(f"  Disruption parameters:")
    print(f"    I_p = {I_p / 1e6:.0f} MA, tau_CQ = {tau_CQ * 1e3:.0f} ms")
    print(f"    R = {R} m, a = {a} m")
    print(f"  Current quench rate: dI/dt = {dI_dt:.3e} A/s = {dI_dt / 1e6:.0f} MA/s")
    print(f"  Plasma inductance: L = {L * 1e6:.2f} microH")
    print(f"  Induced loop voltage: V = L*dI/dt = {V_loop:.1f} V")
    print(f"  Magnetic energy: E = (1/2)*L*I^2 = {E_mag:.3e} J = {E_mag / 1e6:.1f} MJ")
    print(f"  Poloidal field: B_pol = {B_pol:.2f} T")
    print(f"  Vertical force (order of magnitude): F ~ {F_vertical:.3e} N = {F_vertical / 1e6:.0f} MN")
    print(f"  These forces can damage the vacuum vessel and in-vessel components!")
    print(f"  ITER disruption mitigation: massive gas injection (MGI) or shattered pellet injection (SPI)")


def exercise_7():
    """NTM Threshold.

    Estimate the critical island width for NTM onset.
    Delta'_bs = 0.001 m, Delta'_class = -1 m^-1, r_s = 0.5 m.
    """
    Delta_bs = 0.001     # m (bootstrap drive)
    Delta_class = -1.0   # m^-1 (classical tearing stability)
    r_s = 0.5            # m (rational surface radius)

    # The NTM island width evolution (modified Rutherford equation):
    # tau_R * dw/dt = r_s * [Delta'_class + Delta'_bs * r_s / w]
    # At marginal stability (dw/dt = 0):
    # 0 = Delta'_class + Delta'_bs * r_s / w_crit
    # w_crit = -Delta'_bs * r_s / Delta'_class

    # More precisely: w_crit^2 = Delta'_bs / |Delta'_class|
    # or considering the geometric factor:
    # The bootstrap term: Delta'_NTM ~ j_bs * L_q / (B_theta * w)
    # With L_q = q / (dq/dr)|_{r_s}

    w_crit = np.sqrt(Delta_bs * r_s / abs(Delta_class))

    # Alternative: from the Rutherford equation
    # The threshold seed island width for triggering NTM:
    w_seed = w_crit

    print(f"  NTM parameters:")
    print(f"    Delta'_bs = {Delta_bs} m (bootstrap drive)")
    print(f"    Delta'_class = {Delta_class} m^-1 (classical stability)")
    print(f"    r_s = {r_s} m (rational surface)")
    print(f"  Critical island width:")
    print(f"    From Rutherford equation: w_crit = sqrt(Delta'_bs * r_s / |Delta'_class|)")
    print(f"    w_crit = sqrt({Delta_bs} * {r_s} / {abs(Delta_class)})")
    print(f"    w_crit = {w_crit:.4f} m = {w_crit * 100:.2f} cm")
    print(f"  Required seed island size: w_seed > {w_crit:.4f} m")
    print(f"  If seed exceeds w_crit, island grows indefinitely (nonlinear instability).")
    print(f"  Mitigation: ECCD can provide localized current to reduce island width.")

    # Plot island width evolution
    tau_R = 1.0  # normalized resistive time
    dt = 0.001
    N_steps = 10000
    w_init_values = [0.5 * w_crit, 0.9 * w_crit, 1.1 * w_crit, 2.0 * w_crit]

    fig, ax = plt.subplots(figsize=(8, 5))
    for w_0 in w_init_values:
        w = np.zeros(N_steps)
        w[0] = w_0
        for i in range(N_steps - 1):
            if w[i] > 1e-6:
                dw_dt = r_s / tau_R * (Delta_class + Delta_bs * r_s / w[i])
                w[i + 1] = w[i] + dt * dw_dt
                w[i + 1] = max(w[i + 1], 1e-8)
            else:
                w[i + 1] = 1e-8
        t = np.arange(N_steps) * dt
        ax.plot(t, w * 100, linewidth=2, label=f'$w_0$ = {w_0 / w_crit:.1f} $w_c$')

    ax.axhline(w_crit * 100, color='red', linestyle=':', label=f'$w_c$ = {w_crit * 100:.2f} cm')
    ax.set_xlabel('Time (normalized)', fontsize=12)
    ax.set_ylabel('Island width w (cm)', fontsize=12)
    ax.set_title('NTM Island Width Evolution', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(w_init_values) * 300)
    plt.tight_layout()
    plt.savefig('13_ntm_island.png', dpi=150)
    plt.close()
    print("  Plot saved to 13_ntm_island.png")


def exercise_8():
    """RFP Field Reversal.

    For an RFP with mu*a = 4.0, find the radius where B_z = 0 (reversal surface).
    Use the Bessel function J_0(x) and find the first zero.
    """
    mu_a = 4.0  # mu * a (dimensionless)

    # In an RFP (Taylor state), the force-free equilibrium gives:
    # B_z(r) = B_0 * J_0(mu*r)
    # B_theta(r) = B_0 * J_1(mu*r)
    # Field reversal occurs where J_0(mu*r) = 0
    # First zero of J_0 is at x = 2.4048
    # So reversal at mu*r = 2.4048, i.e., r/a = 2.4048 / (mu*a) = 2.4048 / 4.0

    x_zero = 2.4048  # first zero of J_0
    r_reversal = x_zero / mu_a

    print(f"  RFP parameters: mu*a = {mu_a}")
    print(f"  Taylor state equilibrium:")
    print(f"    B_z(r) = B_0 * J_0(mu*r)")
    print(f"    B_theta(r) = B_0 * J_1(mu*r)")
    print(f"  Field reversal: B_z = 0 when J_0(mu*r) = 0")
    print(f"  First zero of J_0 at x = {x_zero}")
    print(f"  Reversal radius: r/a = {x_zero}/{mu_a} = {r_reversal:.3f}")

    if r_reversal < 1.0:
        print(f"  r/a = {r_reversal:.3f} < 1 => reversal occurs INSIDE the plasma!")
        print(f"  B_z reverses direction at r = {r_reversal:.3f} * a")
        print(f"  This is the defining feature of the RFP configuration.")
    else:
        print(f"  r/a = {r_reversal:.3f} > 1 => no reversal inside the plasma.")

    # Plot field profiles
    r_norm = np.linspace(0, 1, 200)
    x = mu_a * r_norm
    Bz = j0(x)
    Btheta = j1(x)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_norm, Bz, 'b-', linewidth=2, label=r'$B_z / B_0 = J_0(\mu r)$')
    ax.plot(r_norm, Btheta, 'r--', linewidth=2, label=r'$B_\theta / B_0 = J_1(\mu r)$')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(r_reversal, color='green', linestyle=':', alpha=0.7,
               label=f'Reversal at r/a = {r_reversal:.3f}')
    ax.set_xlabel('r / a', fontsize=12)
    ax.set_ylabel('B / B_0', fontsize=12)
    ax.set_title(f'RFP Field Profiles ($\\mu a$ = {mu_a})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('13_rfp_reversal.png', dpi=150)
    plt.close()
    print("  Plot saved to 13_rfp_reversal.png")


def exercise_9():
    """Stellarator vs Tokamak Comparison.

    List three advantages and three disadvantages of stellarators compared
    to tokamaks for fusion reactors.
    """
    print("  Stellarator Advantages over Tokamak:")
    print("  1. No plasma current needed => no disruptions")
    print("     - External coils produce entire rotational transform")
    print("     - Inherently steady-state operation")
    print("     - No current-driven instabilities (sawteeth, NTMs, RWMs)")
    print()
    print("  2. Inherently steady-state")
    print("     - No need for inductive current drive or external CD systems")
    print("     - No plasma current ramp-up/ramp-down phases")
    print("     - Higher duty cycle for power plant")
    print()
    print("  3. No density limit (Greenwald-type)")
    print("     - Density limited by radiation/transport, not current-related")
    print("     - Can potentially operate at higher density")
    print()
    print("  Stellarator Disadvantages:")
    print("  1. Complex 3D coil geometry")
    print("     - Extremely challenging engineering and manufacturing")
    print("     - Higher cost and longer construction time")
    print("     - Tight tolerances on coil positioning")
    print()
    print("  2. Higher neoclassical transport (historically)")
    print("     - 1/nu regime transport can be large")
    print("     - Solved by quasi-symmetric designs (quasi-helical, quasi-axisymmetric)")
    print("     - But optimization is computationally expensive")
    print()
    print("  3. Less mature physics and engineering database")
    print("     - Fewer devices worldwide (W7-X, LHD, TJ-II)")
    print("     - Less plasma time and operational experience")
    print("     - Tokamak has ITER => reactor-scale data coming")
    print()
    print("  When is stellarator preferred?")
    print("  - When disruption avoidance is paramount (e.g., power plants)")
    print("  - When steady-state operation is essential")
    print("  - If quasi-symmetric optimization achieves tokamak-level confinement")
    print("  - Long-term: potentially better suited for commercial reactor")


def exercise_10():
    """Beta Optimization.

    A tokamak operates at beta_N = 2.5 (below the Troyon limit of 3.0).
    Propose two methods to increase achievable beta.
    """
    beta_N_current = 2.5
    beta_N_limit = 3.0

    print(f"  Current: beta_N = {beta_N_current}, Troyon limit = {beta_N_limit}")
    print(f"  Available margin: {(beta_N_limit - beta_N_current) / beta_N_limit * 100:.1f}%")
    print()
    print("  Method 1: Equilibrium Shaping (Elongation + Triangularity)")
    print("    - Elongated cross-sections (kappa > 1.6) increase beta limit")
    print("    - Positive triangularity (delta > 0.3) improves stability")
    print("    - Modified Troyon: beta_N_max ~ 3.5 * (1 + 0.5*kappa) for shaped plasmas")
    print("    - ITER: kappa ~ 1.7, delta ~ 0.33")

    # Example: shaped plasma beta limit
    kappa = 1.8
    delta_tri = 0.4
    beta_N_shaped = 3.0 * (1 + 0.3 * (kappa - 1) + 0.2 * delta_tri)
    print(f"    - For kappa={kappa}, delta={delta_tri}: beta_N_max ~ {beta_N_shaped:.1f}")
    print()

    print("  Method 2: Current Profile Optimization")
    print("    - Broad current profiles (low internal inductance l_i) raise beta limit")
    print("    - beta_N_max ~ C / l_i where C ~ 4-5")
    print("    - Achieve with: off-axis ECCD, lower hybrid current drive")
    print("    - Reversed shear (negative central shear) creates internal transport barrier")
    print("    - But must avoid low-n external kinks (need conducting wall)")

    # Example: effect of l_i
    l_i_peaked = 1.2
    l_i_broad = 0.8
    beta_N_peaked = 4.0 / l_i_peaked
    beta_N_broad = 4.0 / l_i_broad
    print(f"    - Peaked current (l_i={l_i_peaked}): beta_N_max ~ {beta_N_peaked:.1f}")
    print(f"    - Broad current (l_i={l_i_broad}): beta_N_max ~ {beta_N_broad:.1f}")
    print()

    print("  Additional approach: Kinetic Stabilization")
    print("    - Fast ion population (from NBI/ICRH) provides stabilizing effect")
    print("    - Energetic particles can stabilize resistive wall modes (RWMs)")
    print("    - Plasma rotation (from NBI) also stabilizes RWMs")
    print("    - Combination of rotation + feedback control allows operation above no-wall limit")

    # Summary plot
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['Current\n(baseline)', 'Shaped\nplasma', 'Broad\ncurrent', 'Both +\nkinetic']
    beta_values = [beta_N_current, beta_N_shaped, beta_N_broad,
                   min(beta_N_shaped * 1.1, 5.5)]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(methods, beta_values, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(beta_N_limit, color='red', linestyle='--', label=f'Troyon limit ({beta_N_limit})')
    ax.set_ylabel(r'$\beta_N$', fontsize=14)
    ax.set_title('Beta Optimization Strategies', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, beta_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f'{val:.1f}',
                ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig('13_beta_optimization.png', dpi=150)
    plt.close()
    print("  Plot saved to 13_beta_optimization.png")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Troyon Limit", exercise_1),
        ("Exercise 2: Safety Factor", exercise_2),
        ("Exercise 3: Sawtooth Period", exercise_3),
        ("Exercise 4: Greenwald Density", exercise_4),
        ("Exercise 5: ELM Heat Flux", exercise_5),
        ("Exercise 6: Disruption Forces", exercise_6),
        ("Exercise 7: NTM Threshold", exercise_7),
        ("Exercise 8: RFP Field Reversal", exercise_8),
        ("Exercise 9: Stellarator vs Tokamak", exercise_9),
        ("Exercise 10: Beta Optimization", exercise_10),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
        func()
