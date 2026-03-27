"""
Plasma Physics - Lesson 13: Two-Fluid Model
Exercise Solutions

Topics covered:
- Moment derivation (energy equation from Vlasov)
- Hall MHD whistler wave dispersion
- Diamagnetic current in tokamak
- Generalized Ohm's law term analysis in current sheet
- Two-fluid drift wave instability
"""

import numpy as np

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
m_p = 1.673e-27
epsilon_0 = 8.854e-12
k_B = 1.381e-23
mu_0 = 4 * np.pi * 1e-7
c = 3e8
eV_to_J = e


def exercise_1():
    """
    Exercise 1: Energy Equation from Vlasov
    Derive the second moment (energy equation) from kinetic theory.
    Show the heat flux tensor q_s appears.
    """
    print("--- Exercise 1: Energy Equation (Second Moment) ---")

    print("Starting from the Vlasov equation:")
    print("  df_s/dt + v * grad(f_s) + (q_s/m_s)*(E + v x B) * grad_v(f_s) = 0")
    print()
    print("Taking the second velocity moment: multiply by (1/2)*m_s*w^2 and integrate")
    print("where w = v - u_s is the peculiar velocity.")
    print()
    print("Result: the energy equation")
    print("  d/dt(3/2 * n_s * T_s) + (5/2)*n_s*T_s * div(u_s) + div(q_s)")
    print("    + P_s : grad(u_s) = Q_s")
    print()
    print("where:")
    print("  n_s * T_s = P_s (scalar pressure)")
    print("  P_s = m_s * integral w*w * f_s d^3v (pressure tensor)")
    print("  q_s = (1/2) * m_s * integral w^2 * w * f_s d^3v (heat flux vector)")
    print("  Q_s = energy exchange with other species (collisional)")
    print()
    print("Physical meaning of each term:")
    print("  d/dt(3/2*nT):      time rate of change of thermal energy")
    print("  (5/2)*nT*div(u):   compressional heating (pdV work + enthalpy flux)")
    print("  div(q):            heat flux divergence (conduction)")
    print("  P:grad(u):         viscous heating")
    print("  Q_s:               collisional energy exchange")
    print()
    print("The heat flux q_s is the 3rd moment -> closure problem!")
    print("Common closures:")
    print("  - Adiabatic: q = 0 (no heat flux)")
    print("  - Braginskii: q_par = -kappa_par * grad_par(T)")
    print("  - Collisionless (CGL): separate p_perp, p_par evolution")

    # Numerical example: compression heating
    n = 1e19
    T_eV = 100.0
    T = T_eV * eV_to_J
    gamma = 5.0 / 3.0  # 3D adiabatic index
    compression = 2.0  # Factor 2 compression

    T_compressed = T * compression**(gamma - 1)
    T_comp_eV = T_compressed / eV_to_J

    print(f"\nNumerical example: adiabatic compression (gamma = {gamma:.3f})")
    print(f"  Initial: n = {n:.0e} m^-3, T = {T_eV} eV")
    print(f"  Compress to 2x density: T_final = T * 2^(gamma-1) = {T_comp_eV:.1f} eV")
    print(f"  Temperature increase: {T_comp_eV/T_eV:.2f}x")
    print()


def exercise_2():
    """
    Exercise 2: Hall MHD Whistler Dispersion
    Derive omega = k_par^2 * V_A^2 / omega_ci for whistler waves.
    """
    print("--- Exercise 2: Hall MHD Whistler Dispersion ---")

    print("Derivation from two-fluid equations with Hall term:")
    print()
    print("Starting equations (cold, omega << omega_ce, omega >> omega_ci):")
    print("  Momentum: rho * du/dt = J x B")
    print("  Ohm's law: E + v x B = (1/en) * J x B  (Hall term)")
    print("  Faraday:  dB/dt = -curl(E)")
    print("  Ampere:   J = (1/mu_0) * curl(B)")
    print()
    print("Linearize: B = B_0 z-hat + B_1, v = v_1")
    print("  rho * dv_1/dt = (1/mu_0)(curl B_1) x B_0")
    print("  dB_1/dt = curl(v_1 x B_0) - curl((1/en*mu_0)(curl B_1) x B_0)")
    print()
    print("For parallel propagation (k = k z-hat):")
    print("  The Hall term gives: omega = k^2 * V_A^2 / omega_ci")
    print("  This is the whistler wave in Hall MHD!")

    # Numerical verification
    B_0 = 5e-5     # Earth's field [T]
    n = 1e7         # Magnetosphere density [m^-3]
    m_i = m_p

    V_A = B_0 / np.sqrt(mu_0 * n * m_i)
    omega_ci = e * B_0 / m_i
    d_i = V_A / omega_ci  # Ion inertial length

    print(f"\nNumerical example (magnetosphere):")
    print(f"  B_0 = {B_0*1e6:.0f} uT, n = {n:.0e} m^-3")
    print(f"  V_A = {V_A/1e3:.0f} km/s")
    print(f"  omega_ci = {omega_ci:.1f} rad/s (f_ci = {omega_ci/(2*np.pi):.2f} Hz)")
    print(f"  d_i = V_A/omega_ci = {d_i/1e3:.1f} km")

    # Dispersion at different k
    print(f"\n  Whistler dispersion: omega = k^2 * V_A^2 / omega_ci")
    print(f"  {'k [m^-1]':>12} {'k*d_i':>10} {'omega [rad/s]':>14} {'f [Hz]':>12} {'v_phi [km/s]':>14}")
    print("  " + "-" * 66)

    k_values = np.array([1e-6, 5e-6, 1e-5, 5e-5, 1e-4])
    for k in k_values:
        omega_w = k**2 * V_A**2 / omega_ci
        f_w = omega_w / (2 * np.pi)
        v_phi = omega_w / k
        print(f"  {k:>12.0e} {k*d_i:>10.3f} {omega_w:>14.4e} {f_w:>12.2f} {v_phi/1e3:>14.1f}")

    print(f"\n  Note: v_phi = k*V_A^2/omega_ci increases with k")
    print(f"  -> Higher k (shorter wavelength) propagates faster")
    print(f"  -> Opposite of normal Alfven wave (v_phi = V_A = const)")
    print()


def exercise_3():
    """
    Exercise 3: Diamagnetic Current in Tokamak
    p_e(r) = p_0*(1-r^2/a^2)^2, p_0 = 5e5 Pa, R_0 = 3 m, a = 1 m, B_phi = 5 T.
    """
    print("--- Exercise 3: Diamagnetic Current in Tokamak ---")

    p_0 = 5e5
    R_0 = 3.0
    a = 1.0
    B_phi = 5.0

    # (a) Diamagnetic current density at r = a/2
    # J_dia = (1/B) * dp/dr x B-hat (perpendicular to B and grad(p))
    # In cylindrical tokamak: J_theta = -(1/B) * dp/dr
    r = a / 2
    p_at_r = p_0 * (1 - r**2 / a**2)**2
    dpdr = -4 * p_0 * r / a**2 * (1 - r**2 / a**2)

    # Diamagnetic current: J_dia = -(1/B) * dp/dr  (in theta direction)
    # More precisely: J_dia,theta = -(1/B_phi) * dp/dr
    J_dia = -dpdr / B_phi

    print(f"(a) Diamagnetic current at r = a/2 = {r} m:")
    print(f"    p(r) = {p_at_r:.4e} Pa")
    print(f"    dp/dr = {dpdr:.4e} Pa/m")
    print(f"    J_dia,theta = -(1/B_phi)*dp/dr = {J_dia:.4e} A/m^2")
    print(f"    J_dia = {J_dia/1e3:.2f} kA/m^2")

    # (b) Total poloidal current from diamagnetic effect
    from scipy.integrate import quad

    def J_dia_integrand(r_val):
        if r_val < 1e-6:
            return 0
        dp = -4 * p_0 * r_val / a**2 * max(0, 1 - r_val**2 / a**2)
        j_theta = -dp / B_phi
        return j_theta * 2 * np.pi * r_val  # Current through annular ring

    I_dia, _ = quad(J_dia_integrand, 0, a * 0.999)

    print(f"\n(b) Total poloidal diamagnetic current:")
    print(f"    I_dia = integral J_theta * 2*pi*r*dr")
    print(f"    I_dia = {I_dia:.4e} A = {I_dia/1e3:.1f} kA")

    # (c) Compare to bootstrap current
    # Bootstrap current has similar profile but different coefficient
    # j_bs ~ sqrt(epsilon) / B_p * dp/dr
    B_p = 0.5  # Approximate poloidal field [T]
    epsilon = r / R_0
    j_bs = np.sqrt(epsilon) / B_p * abs(dpdr)

    print(f"\n(c) Comparison with bootstrap current at r = a/2:")
    print(f"    J_bootstrap ~ sqrt(epsilon)/B_p * |dp/dr|")
    print(f"    epsilon = {epsilon:.3f}, B_p = {B_p} T")
    print(f"    J_bootstrap ~ {j_bs:.4e} A/m^2 = {j_bs/1e3:.2f} kA/m^2")
    print(f"    J_dia / J_bs ~ {J_dia/j_bs:.2f}")
    print(f"    Bootstrap current typically larger by factor B_phi/B_p * sqrt(epsilon)")
    print()


def exercise_4():
    """
    Exercise 4: Generalized Ohm's Law in Current Sheet
    L = 10*d_i, d_i = 100 km, n = 10^7 m^-3, T_e = 100 eV, B = 10 nT.
    """
    print("--- Exercise 4: Generalized Ohm's Law Terms ---")

    d_i = 100e3     # Ion inertial length [m]
    L = 10 * d_i    # Current sheet width [m]
    n = 1e7          # m^-3
    T_e_eV = 100.0
    T_e = T_e_eV * eV_to_J
    B = 10e-9        # T
    m_i = m_p

    # Derived quantities
    V_A = B / np.sqrt(mu_0 * n * m_i)
    omega_ci = e * B / m_i
    omega_pi = np.sqrt(n * e**2 / (epsilon_0 * m_i))
    d_i_calc = c / omega_pi  # Alternative: d_i = V_A / omega_ci

    # Current: J ~ B / (mu_0 * L)
    J = B / (mu_0 * L)

    print(f"Current sheet parameters:")
    print(f"  L = 10*d_i = {L/1e3:.0f} km")
    print(f"  d_i = {d_i/1e3:.0f} km (given), calculated: {d_i_calc/1e3:.0f} km")
    print(f"  n = {n:.0e} m^-3, T_e = {T_e_eV} eV, B = {B*1e9:.0f} nT")
    print(f"  V_A = {V_A/1e3:.1f} km/s")
    print(f"  J ~ B/(mu_0*L) = {J:.4e} A/m^2")
    v = V_A  # Typical flow velocity ~ V_A

    # (a) Ideal MHD term: v x B
    term_ideal = v * B
    print(f"\n(a) |v x B| ~ V_A * B = {term_ideal:.4e} V/m")

    # (b) Hall term: J x B / (en)
    term_hall = J * B / (e * n)
    print(f"(b) |J x B/(en)| = {term_hall:.4e} V/m")

    # (c) Electron pressure term: grad(p_e) / (en)
    # grad(p_e) ~ n*T_e / L
    grad_pe = n * T_e / L
    term_pressure = grad_pe / (e * n)
    print(f"(c) |grad(p_e)/(en)| ~ T_e/(e*L) = {term_pressure:.4e} V/m")

    # (d) Electron inertia: (m_e/(e^2*n)) * dJ/dt
    # dJ/dt ~ J * V_A / L (convective derivative)
    dJdt = J * V_A / L
    term_inertia = m_e * dJdt / (e**2 * n)
    # Alternative: m_e/(e*n) * d(ne*v_e)/dt ~ (m_e*V_A)/(e*L) * J/n
    # Simpler scaling: (d_e/L)^2 * v*B where d_e = c/omega_pe
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    d_e = c / omega_pe
    term_inertia_alt = (d_e / L)**2 * term_ideal

    print(f"(d) |electron inertia| ~ {term_inertia:.4e} V/m")
    print(f"    d_e = {d_e/1e3:.1f} km, (d_e/L)^2 = {(d_e/L)**2:.4e}")

    # Comparison
    print(f"\nTerm comparison (normalized to |v x B|):")
    print(f"  Ideal MHD:    |v x B|         = {term_ideal:.4e} V/m  (1.0)")
    print(f"  Hall:         |J x B/(en)|    = {term_hall:.4e} V/m  ({term_hall/term_ideal:.3f})")
    print(f"  Pressure:     |grad_pe/(en)|  = {term_pressure:.4e} V/m  ({term_pressure/term_ideal:.3f})")
    print(f"  Inertia:      |m_e dJ/dt|     = {term_inertia:.4e} V/m  ({term_inertia/term_ideal:.6f})")

    print(f"\n  At scale L = 10*d_i:")
    print(f"  Hall term ~ (d_i/L) * (v*B) ~ 0.1 * (v*B)")
    print(f"  -> Hall term is important! (governs reconnection dynamics)")
    print(f"  -> This is the Hall MHD regime of reconnection")
    print(f"  Electron inertia is negligible at this scale")
    print(f"  It becomes important at L ~ d_e (electron scale)")
    print()


def exercise_5():
    """
    Exercise 5: Two-Fluid Drift Wave Instability
    Density gradient: L_n = 1 cm, T_e = 1 eV, B = 0.1 T, k_y = 100 m^-1.
    """
    print("--- Exercise 5: Drift Wave ---")

    L_n = 0.01    # Density gradient scale length [m]
    T_e_eV = 1.0
    T_e = T_e_eV * eV_to_J
    B_0 = 0.1     # T
    k_y = 100.0   # m^-1

    # (a) Electron and ion diamagnetic drift velocities
    # v_de = -T_e / (e * B * L_n) (electron diamagnetic drift)
    # v_di = +T_i / (e * B * L_n) (ion diamagnetic drift)
    # For T_e = T_i: v_di = -v_de

    v_de = -T_e / (e * B_0 * L_n)  # In y-direction
    v_di = T_e / (e * B_0 * L_n)   # Opposite to v_de if T_i = T_e

    print(f"(a) Diamagnetic drift velocities:")
    print(f"    L_n = {L_n*100:.0f} cm, T_e = {T_e_eV} eV, B = {B_0} T")
    print(f"    v_de = -T_e/(e*B*L_n) = {v_de:.1f} m/s")
    print(f"    v_di = +T_i/(e*B*L_n) = {v_di:.1f} m/s (assuming T_i = T_e)")
    print(f"    Note: electrons and ions drift in opposite directions")

    # (b) Drift wave dispersion relation
    # omega = k_y * v_de = k_y * T_e / (e * B * L_n)
    # More precisely: omega* = k_y * T_e / (e * B * L_n)
    omega_star = k_y * T_e / (e * B_0 * L_n)
    f_star = omega_star / (2 * np.pi)

    print(f"\n(b) Drift wave dispersion:")
    print(f"    omega* = k_y * T_e / (e * B * L_n)")
    print(f"    omega* = {omega_star:.4e} rad/s")
    print(f"    f* = {f_star:.1f} Hz")

    # Derivation summary
    print(f"\n    Derivation outline:")
    print(f"    1. Electron continuity: dn_e/dt + n_0*div(v_e) = 0")
    print(f"    2. Electron perpendicular drift: v_e = v_ExB + v_de")
    print(f"       v_ExB = (E x B)/B^2 -> perturbed density response")
    print(f"    3. Quasi-neutrality: n_e = n_i")
    print(f"    4. Ion continuity with polarization drift:")
    print(f"       The ion polarization drift gives the E x B vorticity equation")
    print(f"    5. Combining with Boltzmann electron response:")
    print(f"       n_1/n_0 = e*phi_1 / T_e")
    print(f"    6. Result: omega = omega* = k_y*T_e/(e*B*L_n)")

    # (c) Numerical values
    print(f"\n(c) For L_n = {L_n*100:.0f} cm, T_e = {T_e_eV} eV, B = {B_0} T, k_y = {k_y} m^-1:")
    print(f"    Drift wave frequency: f = {f_star:.1f} Hz")
    print(f"    Wavelength: lambda_y = 2*pi/k_y = {2*np.pi/k_y*100:.1f} cm")

    # Compare to cyclotron frequency
    omega_ci = e * B_0 / m_p
    omega_ce = e * B_0 / m_e
    rho_i = np.sqrt(T_e / m_p) / omega_ci
    rho_s = np.sqrt(T_e / m_p) / omega_ci  # Sound Larmor radius

    print(f"\n    omega_ci = {omega_ci:.4e} rad/s")
    print(f"    omega*/omega_ci = {omega_star/omega_ci:.4e} << 1 (drift ordering)")
    print(f"    k_y * rho_s = {k_y * rho_s:.4f}")
    print(f"    rho_s = {rho_s*1e3:.2f} mm")

    # Growth mechanism
    print(f"\n    Drift wave instability mechanism:")
    print(f"    - Density perturbation n_1 -> potential phi_1 (Boltzmann)")
    print(f"    - phi_1 drives E x B drift that advects density gradient")
    print(f"    - Phase shift between n_1 and phi_1 (from finite resistivity,")
    print(f"      electron inertia, or kinetic effects) causes instability")
    print(f"    - Universal instability: always present with density gradient")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
