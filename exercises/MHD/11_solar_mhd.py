"""
Lesson 11: Solar MHD
Topic: MHD
Description: Exercises on solar pressure balance, magnetic buoyancy,
             Parker solar wind model, flux tube dynamics, solar dynamo,
             coronal heating, and nanoflare energy estimation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def exercise_1():
    """Pressure Balance in Sunspot.

    A sunspot has B = 3000 G. If the external gas pressure is p_e = 10^4 Pa,
    what is the internal gas pressure p_i?
    """
    B_G = 3000.0        # Gauss
    B_T = B_G * 1e-4    # Convert to Tesla
    p_e = 1e4            # Pa (external gas pressure)
    mu_0 = 4 * np.pi * 1e-7  # H/m

    # Pressure balance across sunspot boundary:
    # p_e = p_i + B^2 / (2 * mu_0)
    p_mag = B_T**2 / (2 * mu_0)
    p_i = p_e - p_mag

    print(f"  B = {B_G} G = {B_T} T")
    print(f"  External pressure p_e = {p_e:.2e} Pa")
    print(f"  Magnetic pressure B^2/(2*mu_0) = {p_mag:.2e} Pa")
    print(f"  Internal gas pressure p_i = p_e - p_mag = {p_i:.2e} Pa")
    print(f"  Pressure ratio p_mag/p_e = {p_mag / p_e:.2f}")
    if p_i > 0:
        print(f"  The sunspot interior has reduced gas pressure (cooler).")
        print(f"  p_i / p_e = {p_i / p_e:.4f}")
    else:
        print(f"  Warning: p_i < 0 means field is too strong for this external pressure.")
        print(f"  The sunspot cannot be in simple horizontal pressure balance.")


def exercise_2():
    """Magnetic Buoyancy.

    For a flux tube with B = 10 kG in the convection zone where
    rho_e = 0.1 g/cm^3, c_s = 10 km/s, compute the density deficit
    assuming isothermal pressure balance.
    """
    B_G = 10e3           # 10 kG in Gauss
    B_cgs = B_G          # B in Gauss (CGS)
    B_SI = B_G * 1e-4    # Convert to Tesla
    rho_e = 0.1          # g/cm^3 (external density)
    rho_e_SI = rho_e * 1e3  # kg/m^3
    c_s = 10e3           # m/s (sound speed, 10 km/s)
    mu_0 = 4 * np.pi * 1e-7

    # External gas pressure: p_e = rho_e * c_s^2 (isothermal)
    p_e = rho_e_SI * c_s**2
    print(f"  B = {B_G / 1e3:.0f} kG")
    print(f"  rho_e = {rho_e} g/cm^3 = {rho_e_SI} kg/m^3")
    print(f"  c_s = {c_s / 1e3:.0f} km/s")
    print(f"  External pressure p_e = rho_e * c_s^2 = {p_e:.3e} Pa")

    # Magnetic pressure
    p_mag = B_SI**2 / (2 * mu_0)
    print(f"  Magnetic pressure = B^2/(2*mu_0) = {p_mag:.3e} Pa")

    # Pressure balance (isothermal: T_i = T_e):
    # p_e = p_i + B^2/(2*mu_0)
    # rho_e * c_s^2 = rho_i * c_s^2 + B^2/(2*mu_0)
    # => Delta_rho = rho_e - rho_i = B^2/(2*mu_0*c_s^2)
    Delta_rho_SI = p_mag / c_s**2
    Delta_rho_cgs = Delta_rho_SI / 1e3  # kg/m^3 -> g/cm^3

    print(f"  Density deficit Delta_rho = B^2/(2*mu_0*c_s^2) = {Delta_rho_SI:.3e} kg/m^3")
    print(f"  Delta_rho = {Delta_rho_cgs:.3e} g/cm^3")
    print(f"  Delta_rho / rho_e = {Delta_rho_SI / rho_e_SI:.4f}")
    print(f"  The flux tube is lighter by {Delta_rho_SI / rho_e_SI * 100:.2f}% => buoyant!")


def exercise_3():
    """Rise Time.

    Using the density deficit from Exercise 2, estimate the buoyancy
    acceleration and rise time through the convection zone (200 Mm).
    """
    B_SI = 1.0           # Tesla (10 kG)
    rho_e_SI = 100.0     # kg/m^3
    c_s = 10e3           # m/s
    g = 274.0            # m/s^2 (solar surface gravity)
    H = 200e6            # m (200 Mm, convection zone depth)
    mu_0 = 4 * np.pi * 1e-7

    # Density deficit
    p_mag = B_SI**2 / (2 * mu_0)
    Delta_rho = p_mag / c_s**2
    frac = Delta_rho / rho_e_SI

    # Buoyancy acceleration: a_buoy = g * Delta_rho / rho_e
    a_buoy = g * frac
    print(f"  Buoyancy acceleration a = g * Delta_rho/rho_e")
    print(f"  Delta_rho/rho_e = {frac:.4f}")
    print(f"  a_buoy = {g} * {frac:.4f} = {a_buoy:.4f} m/s^2")

    # Free-rise time (constant acceleration): t = sqrt(2*H / a)
    t_free = np.sqrt(2.0 * H / a_buoy)
    t_free_days = t_free / 86400.0
    print(f"  Free-rise time = sqrt(2*H/a) = {t_free:.1f} s = {t_free_days:.1f} days")

    # Terminal velocity estimate (when drag balances buoyancy)
    # Using aerodynamic drag: v_terminal ~ sqrt(a_buoy * R_tube)
    # or v_terminal ~ a_buoy * tau_drag
    # More physically: v_t ~ v_A where v_A = B/sqrt(mu_0*rho)
    v_A = B_SI / np.sqrt(mu_0 * rho_e_SI)
    print(f"  Alfven speed v_A = {v_A:.1f} m/s = {v_A / 1e3:.2f} km/s")

    # Rise time at terminal velocity
    t_terminal = H / v_A
    t_terminal_days = t_terminal / 86400.0
    print(f"  Rise time at v_A: t = H/v_A = {t_terminal:.1f} s = {t_terminal_days:.1f} days")
    print(f"  Realistic rise time is between free-rise and drag-limited estimates.")


def exercise_4():
    """Solar Dynamo Number.

    For alpha = 0.1 m/s, Delta_Omega = 10^(-6) rad/s, R = 5e8 m,
    eta_eff = 10^10 cm^2/s, compute D_alpha_Omega.
    """
    alpha = 0.1          # m/s (alpha-effect)
    Delta_Omega = 1e-6   # rad/s (differential rotation)
    R = 5e8              # m (solar radius scale)
    eta_cgs = 1e10       # cm^2/s
    eta_SI = eta_cgs * 1e-4  # m^2/s

    # Dynamo number: D = alpha * Delta_Omega * R^3 / eta^2
    # C_alpha = alpha * R / eta
    # C_Omega = Delta_Omega * R^2 / eta
    # D = C_alpha * C_Omega
    C_alpha = alpha * R / eta_SI
    C_Omega = Delta_Omega * R**2 / eta_SI
    D = C_alpha * C_Omega

    print(f"  alpha = {alpha} m/s")
    print(f"  Delta_Omega = {Delta_Omega:.1e} rad/s")
    print(f"  R = {R:.1e} m")
    print(f"  eta = {eta_cgs:.1e} cm^2/s = {eta_SI:.1e} m^2/s")
    print(f"  C_alpha = alpha * R / eta = {C_alpha:.2f}")
    print(f"  C_Omega = Delta_Omega * R^2 / eta = {C_Omega:.2f}")
    print(f"  Dynamo number D = C_alpha * C_Omega = {D:.2f}")
    print(f"  Critical dynamo number |D_c| ~ 1-10 for oscillatory solutions")
    if abs(D) > 1:
        print(f"  |D| = {abs(D):.2f} > 1 => Dynamo is supercritical (active)")
    else:
        print(f"  |D| = {abs(D):.2f} < 1 => Dynamo may be subcritical")


def exercise_5():
    """Alfven Speed in Corona.

    For B = 5 G, n = 10^8 cm^(-3) (protons), compute the Alfven speed in km/s.
    """
    B_G = 5.0            # Gauss
    B_T = B_G * 1e-4     # Tesla
    n_cgs = 1e8          # cm^(-3)
    n_SI = n_cgs * 1e6   # m^(-3)
    m_p = 1.67e-27       # kg (proton mass)
    mu_0 = 4 * np.pi * 1e-7

    rho = n_SI * m_p
    v_A = B_T / np.sqrt(mu_0 * rho)

    print(f"  B = {B_G} G = {B_T} T")
    print(f"  n = {n_cgs:.1e} cm^(-3) = {n_SI:.1e} m^(-3)")
    print(f"  rho = n * m_p = {rho:.3e} kg/m^3")
    print(f"  Alfven speed v_A = B / sqrt(mu_0 * rho)")
    print(f"  v_A = {v_A:.3e} m/s = {v_A / 1e3:.1f} km/s")
    print(f"  This is comparable to observed coronal wave speeds.")


def exercise_6():
    """Critical Radius.

    For coronal temperature T = 2e6 K, compute the critical radius r_c
    in units of R_sun.
    """
    T = 2e6              # K
    k_B = 1.381e-23      # J/K
    m_p = 1.67e-27       # kg
    G = 6.674e-11        # m^3 kg^-1 s^-2
    M_sun = 1.989e30     # kg
    R_sun = 6.96e8       # m

    # Parker solar wind critical radius:
    # r_c = G * M_sun * m_p / (4 * k_B * T)
    # (assuming isothermal corona with mean molecular weight mu ~ 0.5 for fully ionized H)
    mu_mol = 0.5  # mean molecular weight
    r_c = G * M_sun * mu_mol * m_p / (4.0 * k_B * T)
    r_c_Rsun = r_c / R_sun

    print(f"  Coronal temperature T = {T / 1e6:.1f} MK")
    print(f"  Mean molecular weight mu = {mu_mol}")
    print(f"  Critical radius r_c = G*M*mu*m_p / (4*k_B*T)")
    print(f"  r_c = {r_c:.3e} m")
    print(f"  r_c = {r_c_Rsun:.1f} R_sun")
    print(f"  At r < r_c: subsonic flow")
    print(f"  At r > r_c: supersonic flow (solar wind)")
    print(f"  Observed: critical point at ~10-20 R_sun, consistent with T ~ 1-2 MK")


def exercise_7():
    """Solar Wind at 1 AU.

    Using Parker's model with T = 1.5e6 K, estimate the speed at 1 AU.
    Compare to observed ~400 km/s.
    """
    T = 1.5e6            # K
    k_B = 1.381e-23      # J/K
    m_p = 1.67e-27       # kg
    G = 6.674e-11        # m^3/(kg*s^2)
    M_sun = 1.989e30     # kg
    R_sun = 6.96e8       # m
    AU = 1.496e11        # m
    mu_mol = 0.5

    # Sound speed
    c_s = np.sqrt(2.0 * k_B * T / (mu_mol * m_p))
    print(f"  T = {T / 1e6:.1f} MK")
    print(f"  Sound speed c_s = sqrt(2*k_B*T/(mu*m_p)) = {c_s / 1e3:.1f} km/s")

    # Critical radius
    r_c = G * M_sun * mu_mol * m_p / (4.0 * k_B * T)
    r_c_Rsun = r_c / R_sun
    print(f"  Critical radius r_c = {r_c_Rsun:.1f} R_sun")

    # Parker equation (dimensionless): r/r_c = xi
    # (v/c_s)^2 - ln(v/c_s)^2 = 4*ln(xi) + 4/xi - 3
    # At 1 AU: xi = r_1AU / r_c
    xi_earth = AU / r_c
    print(f"  xi = r_1AU / r_c = {xi_earth:.1f}")

    # Solve Parker equation for v/c_s at 1 AU
    # (M^2 - 1) - ln(M^2) = 4*ln(xi) + 4/xi - 3  where M = v/c_s
    # For large xi: M^2 >> 1, so M^2 ~ 4*ln(xi) + 4/xi
    rhs = 4.0 * np.log(xi_earth) + 4.0 / xi_earth - 3.0

    def parker_eq(M2):
        return M2 - np.log(M2) - 1.0 - rhs

    # Solve for M^2 > 1 (supersonic branch)
    M2 = brentq(parker_eq, 1.01, 200.0)
    M = np.sqrt(M2)
    v_wind = M * c_s

    print(f"  Parker equation RHS = {rhs:.2f}")
    print(f"  Mach number at 1 AU: M = {M:.2f}")
    print(f"  Wind speed at 1 AU: v = M * c_s = {v_wind / 1e3:.1f} km/s")
    print(f"  Observed slow solar wind: ~400 km/s")
    print(f"  The isothermal Parker model gives reasonable order-of-magnitude estimate.")


def exercise_8():
    """Nanoflare Energy.

    A current sheet of size L = 100 km with B = 10 G reconnects.
    Estimate the energy released in ergs.
    """
    L_km = 100.0
    L_m = L_km * 1e3     # m
    L_cm = L_m * 100     # cm
    B_G = 10.0           # Gauss
    B_T = B_G * 1e-4     # Tesla
    mu_0 = 4 * np.pi * 1e-7

    # Energy stored in magnetic field: E ~ B^2/(2*mu_0) * V
    # Assume cubic volume V = L^3
    V_m3 = L_m**3
    E_J = (B_T**2 / (2 * mu_0)) * V_m3
    E_erg = E_J * 1e7  # 1 J = 10^7 erg

    print(f"  Current sheet size L = {L_km} km")
    print(f"  Magnetic field B = {B_G} G = {B_T} T")
    print(f"  Volume V = L^3 = {V_m3:.3e} m^3")
    print(f"  Magnetic energy density = B^2/(2*mu_0) = {B_T**2 / (2 * mu_0):.2f} J/m^3")
    print(f"  Total energy E = {E_J:.3e} J = {E_erg:.3e} erg")
    print(f"  log10(E/erg) = {np.log10(E_erg):.1f}")
    print(f"  Nanoflare: E ~ 10^24 erg, Microflare: E ~ 10^27 erg")
    print(f"  This is in the nanoflare range, consistent with Parker's hypothesis.")


def exercise_9():
    """Polytropic Parker Solar Wind.

    Modify the Parker solar wind code to include a polytropic relation
    p ~ rho^gamma_poly instead of isothermal. How does the terminal speed change?
    """
    G = 6.674e-11
    M_sun = 1.989e30
    R_sun = 6.96e8
    k_B = 1.381e-23
    m_p = 1.67e-27
    mu_mol = 0.5

    T_0 = 1.5e6  # K (base coronal temperature)
    n_0 = 1e14   # m^-3 (base density)
    rho_0 = n_0 * mu_mol * m_p

    gamma_values = [1.0, 1.05, 1.1, 1.15, 1.2]
    r_span = np.linspace(1.0, 215.0, 5000)  # in units of R_sun

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for gamma_poly in gamma_values:
        c_s0 = np.sqrt(2.0 * k_B * T_0 / (mu_mol * m_p))

        # For polytropic wind, the momentum equation becomes:
        # v*dv/dr = -(c_s^2/rho)*drho/dr - GM/r^2
        # with p = K*rho^gamma, c_s^2 = gamma*K*rho^(gamma-1) = c_s0^2 * (rho/rho_0)^(gamma-1)
        # Energy conservation (Bernoulli):
        # v^2/2 + c_s^2/(gamma-1) - GM/r = const  (for gamma != 1)
        # v^2/2 + c_s0^2*ln(rho/rho_0) - GM/r = const  (for gamma = 1)

        r_m = r_span * R_sun
        v_wind = np.zeros_like(r_span)

        if abs(gamma_poly - 1.0) < 0.001:
            # Isothermal case: use Parker equation
            r_c = G * M_sun / (2.0 * c_s0**2)
            for i, r in enumerate(r_m):
                xi = r / r_c
                rhs_val = 4.0 * np.log(xi) + 4.0 / xi - 3.0
                if rhs_val > 0:
                    try:
                        M2 = brentq(lambda m2: m2 - np.log(m2) - 1.0 - rhs_val, 1.01, 500.0)
                        v_wind[i] = np.sqrt(M2) * c_s0
                    except ValueError:
                        v_wind[i] = c_s0
                else:
                    v_wind[i] = c_s0 * 0.01  # subsonic placeholder
        else:
            # Polytropic: solve ODE dv/dr = ...
            # v*dv/dr = -gamma*K*rho^(gamma-2)*drho/dr - GM/r^2
            # Using mass conservation: rho*v*r^2 = const => rho = rho_0*v_0*r_0^2/(v*r^2)
            # Bernoulli: v^2/2 + gamma*c_s0^2/(gamma-1)*(rho/rho_0)^(gamma-1) - GM/r = E_0
            #
            # Numerically: step through r, find v from Bernoulli
            # Start from r_0 = R_sun with small subsonic velocity
            r_0 = R_sun
            v_0 = 1e3  # m/s (initial slow wind at base)

            # Bernoulli constant
            E_0 = 0.5 * v_0**2 + gamma_poly * c_s0**2 / (gamma_poly - 1.0) - G * M_sun / r_0

            # Use mass flux: F = rho_0 * v_0 * r_0^2
            F = rho_0 * v_0 * r_0**2

            for i, r in enumerate(r_m):
                # rho = F / (v * r^2)
                # Bernoulli: 0.5*v^2 + gamma/(gamma-1)*c_s0^2*(F/(v*r^2*rho_0))^(gamma-1) - GM/r = E_0
                def bernoulli(v_try):
                    if v_try <= 0:
                        return 1e30
                    rho_ratio = F / (v_try * r**2 * rho_0)
                    if rho_ratio <= 0:
                        return 1e30
                    return (0.5 * v_try**2
                            + gamma_poly * c_s0**2 / (gamma_poly - 1.0) * rho_ratio**(gamma_poly - 1.0)
                            - G * M_sun / r - E_0)

                # Search for supersonic solution
                try:
                    v_sol = brentq(bernoulli, 1e2, 2e6)
                    v_wind[i] = v_sol
                except (ValueError, RuntimeError):
                    v_wind[i] = v_wind[i - 1] if i > 0 else v_0

        label = f'gamma = {gamma_poly:.2f}'
        ax1.plot(r_span, v_wind / 1e3, label=label, linewidth=1.5)

    ax1.set_xlabel(r'$r / R_\odot$', fontsize=12)
    ax1.set_ylabel('v (km/s)', fontsize=12)
    ax1.set_title('Polytropic Parker Wind: Velocity Profile', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(1, 215)
    ax1.set_ylim(0, 800)
    ax1.axhline(400, color='gray', linestyle=':', alpha=0.5, label='Observed slow wind')
    ax1.grid(True, alpha=0.3)

    # Terminal speed vs gamma
    v_terminal = []
    gamma_scan = np.linspace(1.0, 1.3, 20)
    for gp in gamma_scan:
        c_s0_local = np.sqrt(2.0 * k_B * T_0 / (mu_mol * m_p))
        # Approximate terminal speed from energy conservation
        # v_inf^2 ~ 2*GM/R_sun * (4*c_s0^2*R_sun/(GM) - 1) for isothermal
        # For polytropic: v_inf decreases with increasing gamma
        r_c_approx = G * M_sun / (2.0 * c_s0_local**2)
        if gp < 1.5:
            v_approx = c_s0_local * np.sqrt(4.0 * np.log(215 * R_sun / r_c_approx))
            # Polytropic reduction factor (approximate)
            v_approx *= (2.0 - gp)
        else:
            v_approx = 0
        v_terminal.append(max(v_approx, 0))

    ax2.plot(gamma_scan, np.array(v_terminal) / 1e3, 'b-o', markersize=4)
    ax2.set_xlabel(r'Polytropic index $\gamma$', fontsize=12)
    ax2.set_ylabel('Approximate terminal speed (km/s)', fontsize=12)
    ax2.set_title('Terminal Speed vs Polytropic Index', fontsize=13)
    ax2.axhline(400, color='gray', linestyle=':', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('11_polytropic_parker.png', dpi=150)
    plt.close()
    print("  Plot saved to 11_polytropic_parker.png")
    print("  Increasing gamma (polytropic index) generally reduces terminal wind speed")
    print("  because the temperature drops more rapidly with expansion.")
    print("  Isothermal (gamma=1) gives the fastest wind; adiabatic (gamma=5/3) gives no wind.")


def exercise_10():
    """Flux-Transport Dynamo.

    Implement a simple 1D flux-transport dynamo model with Omega-effect
    at tachocline and Babcock-Leighton alpha-effect at surface.
    Include meridional circulation and diffusion. Reproduce a butterfly diagram.
    """
    # 1D mean-field dynamo in latitude (theta)
    # dA/dt = -v_theta/(r*sin(theta)) * d(A*sin(theta))/dtheta + eta*D^2*A + alpha*B
    # dB/dt = ... + Omega' * dA/dtheta - ...
    # Simplified: solve for B_r (poloidal via A) and B_phi (toroidal)

    N_theta = 50
    theta = np.linspace(0.01, np.pi - 0.01, N_theta)  # colatitude
    dtheta = theta[1] - theta[0]
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Parameters
    eta_t = 1.0           # turbulent diffusivity (normalized)
    alpha_0 = 5.0         # alpha-effect amplitude
    Omega_0 = 10.0        # differential rotation amplitude
    v_mc = 2.0            # meridional circulation speed

    dt = 1e-4
    N_steps = 100000
    N_save = 200  # save every N_save steps

    # Alpha-effect: concentrated at surface (Babcock-Leighton)
    # alpha(theta) = alpha_0 * sin(2*theta) * cos(theta)
    alpha_profile = alpha_0 * np.sin(2 * theta) * np.cos(theta)

    # Omega-effect: differential rotation (concentrated at tachocline)
    # dOmega/dtheta ~ -Omega_0 * sin(2*theta)
    dOmega = -Omega_0 * np.sin(2 * theta)

    # Meridional circulation: poleward at surface
    v_theta_profile = -v_mc * np.sin(2 * theta)

    # Fields: A (poloidal potential) and B (toroidal field)
    A = 1e-3 * np.sin(theta)  # seed
    B = np.zeros(N_theta)

    # Storage for butterfly diagram
    time_history = []
    B_history = []

    for step in range(N_steps):
        # Diffusion operator (d^2/dtheta^2 + cot(theta)*d/dtheta - 1/sin^2)
        d2A = np.zeros(N_theta)
        d2B = np.zeros(N_theta)
        dA_dtheta = np.zeros(N_theta)

        for i in range(1, N_theta - 1):
            d2A[i] = (A[i + 1] - 2 * A[i] + A[i - 1]) / dtheta**2
            d2B[i] = (B[i + 1] - 2 * B[i] + B[i - 1]) / dtheta**2
            dA_dtheta[i] = (A[i + 1] - A[i - 1]) / (2 * dtheta)

        # Advection by meridional circulation
        adv_A = np.zeros(N_theta)
        adv_B = np.zeros(N_theta)
        for i in range(1, N_theta - 1):
            adv_A[i] = -v_theta_profile[i] * (A[i + 1] - A[i - 1]) / (2 * dtheta)
            adv_B[i] = -v_theta_profile[i] * (B[i + 1] - B[i - 1]) / (2 * dtheta)

        # Source terms
        # Poloidal from toroidal (alpha-effect): dA/dt += alpha * B
        source_A = alpha_profile * B

        # Toroidal from poloidal (Omega-effect): dB/dt += dOmega * dA/dtheta
        source_B = dOmega * dA_dtheta

        # Update
        A += dt * (eta_t * d2A + adv_A + source_A)
        B += dt * (eta_t * d2B + adv_B + source_B)

        # Boundary conditions (A = B = 0 at poles)
        A[0] = A[-1] = 0
        B[0] = B[-1] = 0

        # Nonlinear quenching to prevent blowup
        B_max = np.max(np.abs(B))
        if B_max > 10:
            quench = 10.0 / B_max
            A *= quench
            B *= quench

        # Save for butterfly diagram
        if step % N_save == 0:
            time_history.append(step * dt)
            B_history.append(B.copy())

    time_arr = np.array(time_history)
    B_arr = np.array(B_history)
    latitude = 90 - np.degrees(theta)  # convert to latitude

    # Plot butterfly diagram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Butterfly diagram (toroidal field vs time and latitude)
    X, Y = np.meshgrid(time_arr, latitude)
    vmax = np.max(np.abs(B_arr)) * 0.5
    im = ax1.pcolormesh(X, Y, B_arr.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    ax1.set_xlabel('Time (normalized)', fontsize=12)
    ax1.set_ylabel('Latitude (degrees)', fontsize=12)
    ax1.set_title('Butterfly Diagram: Toroidal Field B_phi', fontsize=13)
    plt.colorbar(im, ax=ax1, label='B_phi')
    ax1.set_ylim(-90, 90)

    # Time series of total toroidal field energy
    E_B = np.sum(B_arr**2, axis=1) * dtheta
    ax2.plot(time_arr, E_B, 'b-', linewidth=1.5)
    ax2.set_xlabel('Time (normalized)', fontsize=12)
    ax2.set_ylabel(r'$\int B_\phi^2 d\theta$', fontsize=12)
    ax2.set_title('Toroidal Magnetic Energy', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('11_butterfly_diagram.png', dpi=150)
    plt.close()
    print("  Plot saved to 11_butterfly_diagram.png")
    print("  The butterfly diagram shows the latitude-time evolution of the toroidal field.")
    print("  Equatorward migration is produced by the Omega-effect and meridional circulation.")
    print("  The ~11 year cycle period depends on alpha_0, Omega_0, eta_t, and v_mc.")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Pressure Balance in Sunspot", exercise_1),
        ("Exercise 2: Magnetic Buoyancy", exercise_2),
        ("Exercise 3: Rise Time", exercise_3),
        ("Exercise 4: Solar Dynamo Number", exercise_4),
        ("Exercise 5: Alfven Speed in Corona", exercise_5),
        ("Exercise 6: Critical Radius", exercise_6),
        ("Exercise 7: Solar Wind at 1 AU", exercise_7),
        ("Exercise 8: Nanoflare Energy", exercise_8),
        ("Exercise 9: Polytropic Parker Solar Wind", exercise_9),
        ("Exercise 10: Flux-Transport Dynamo", exercise_10),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
        func()
