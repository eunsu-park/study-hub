"""
Plasma Physics - Lesson 09: Collisional Kinetics
Exercise Solutions

Topics covered:
- Collision frequency calculation (ee, ei, energy equilibration)
- Alpha particle slowing down in fusion plasma
- Classical vs neoclassical transport in tokamak
- Perpendicular vs parallel transport anisotropy
- Bootstrap current estimation
"""

import numpy as np

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
m_p = 1.673e-27
epsilon_0 = 8.854e-12
k_B = 1.381e-23
mu_0 = 4 * np.pi * 1e-7
eV_to_J = e


def exercise_1():
    """
    Exercise 1: Collision Frequencies
    Hydrogen plasma: n = 10^19 m^-3, T_e = T_i = 1 keV, ln(Lambda) = 15.
    """
    print("--- Exercise 1: Collision Frequencies ---")

    n = 1e19
    T_eV = 1e3  # 1 keV
    T_J = T_eV * eV_to_J
    ln_Lambda = 15.0
    m_i = m_p

    v_th_e = np.sqrt(T_J / m_e)
    v_th_i = np.sqrt(T_J / m_i)

    # (a) Electron-electron collision frequency
    # nu_ee = n * e^4 * ln(Lambda) / (6*pi^2 * epsilon_0^2 * m_e^2 * v_the^3)
    nu_ee = n * e**4 * ln_Lambda / (6 * np.pi**2 * epsilon_0**2 * m_e**2 * v_th_e**3)
    print(f"(a) Electron-electron collision frequency:")
    print(f"    v_th,e = {v_th_e:.4e} m/s")
    print(f"    nu_ee = {nu_ee:.4e} s^-1")
    print(f"    tau_ee = {1/nu_ee:.4e} s")

    # (b) Electron-ion collision frequency
    # nu_ei ~ sqrt(2) * nu_ee (for Z=1)
    nu_ei = np.sqrt(2) * nu_ee
    print(f"\n(b) Electron-ion collision frequency:")
    print(f"    nu_ei = sqrt(2) * nu_ee = {nu_ei:.4e} s^-1")
    print(f"    tau_ei = {1/nu_ei:.4e} s")

    # (c) Energy equilibration time
    # tau_eq = (m_i / (2*m_e)) * tau_ei
    tau_eq = (m_i / (2 * m_e)) * (1 / nu_ei)
    print(f"\n(c) Energy equilibration time:")
    print(f"    tau_eq = (m_i/(2*m_e)) * tau_ei = {tau_eq:.4e} s = {tau_eq*1e3:.2f} ms")
    print(f"    Mass ratio enhancement: m_i/(2*m_e) = {m_i/(2*m_e):.0f}")

    # (d) Compare to plasma period and transit time
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    T_pe = 2 * np.pi / omega_pe
    tau_transit = 1.0 / v_th_e  # Transit time across 1 m

    print(f"\n(d) Time scale comparison:")
    print(f"    Plasma period: T_pe = {T_pe:.4e} s")
    print(f"    Electron transit time (1m): tau_tr = {tau_transit:.4e} s")
    print(f"    tau_ee / T_pe = {(1/nu_ee)/T_pe:.0f}")
    print(f"    tau_ee / tau_tr = {(1/nu_ee)/tau_transit:.1f}")
    print(f"    Collisionality: omega_pe * tau_ee = {omega_pe/nu_ee:.0f}")
    if omega_pe / nu_ee > 10:
        print(f"    -> Weakly collisional (many oscillations between collisions)")
    else:
        print(f"    -> Collisional (few oscillations between collisions)")
    print()


def exercise_2():
    """
    Exercise 2: Alpha Particle Slowing Down
    3.5 MeV alpha in D-T fusion plasma:
    n = 5e19 m^-3, T_e = 15 keV, T_i = 12 keV.
    """
    print("--- Exercise 2: Alpha Particle Slowing Down ---")

    E_alpha = 3.5e6 * eV_to_J  # 3.5 MeV
    n = 5e19
    T_e_eV = 15e3   # 15 keV
    T_i_eV = 12e3   # 12 keV
    T_e = T_e_eV * eV_to_J
    T_i = T_i_eV * eV_to_J
    m_alpha = 4 * m_p
    Z_alpha = 2
    m_D = 2 * m_p
    m_T = 3 * m_p
    m_i_avg = 2.5 * m_p  # DT average
    ln_Lambda = 17.0

    v_alpha = np.sqrt(2 * E_alpha / m_alpha)
    v_th_e = np.sqrt(T_e / m_e)

    print(f"Alpha particle: E = 3.5 MeV, v_alpha = {v_alpha:.4e} m/s")
    print(f"Plasma: n = {n:.0e} m^-3, T_e = {T_e_eV/1e3:.0f} keV, T_i = {T_i_eV/1e3:.0f} keV")
    print(f"v_alpha / v_th,e = {v_alpha/v_th_e:.2f}")

    # (a) Critical energy separating electron and ion heating
    # E_c = (3*sqrt(pi)/4)^(2/3) * T_e * (m_alpha/m_e)^(1/3) * (m_alpha/m_i)^(1/3)
    # Simplified: E_c ~ 14.8 * T_e * (m_alpha/m_p)^(2/3) (for DT)
    E_c = 14.8 * T_e * (m_alpha / m_p)**(2.0 / 3)
    E_c_keV = E_c / (1e3 * eV_to_J)

    print(f"\n(a) Critical energy E_c:")
    print(f"    E_c = {E_c_keV:.1f} keV = {E_c/eV_to_J/1e6:.3f} MeV")
    print(f"    For E > E_c: alpha heats electrons (fast alpha, v >> v_th,e)")
    print(f"    For E < E_c: alpha heats ions (slow alpha, v ~ v_th,i)")

    # (b) Energy fraction to electrons vs ions
    # Fraction to electrons: f_e = integral from E_c to E_0 of (dE/E_c) / (1 + (E/E_c)^(3/2))
    # Approximate: f_e ~ 1 - (E_c/E_0)^(1/2) for E_0 >> E_c
    # More accurately:
    x = E_alpha / E_c
    # f_e = 1 - (2/3)*atan(1+2*x^(1/2)/sqrt(3))/sqrt(3) + (1/3)*ln((1+x^(1/2))^2/(1-x^(1/2)+x))/6...
    # Simpler approximation: f_e ~ 1 - (2/(3*x)) for x >> 1
    from scipy.integrate import quad

    def dE_electron_fraction(E_ratio):
        """Fraction of energy loss going to electrons at given E/E_c."""
        return 1.0 / (1.0 + E_ratio**(-1.5))

    f_e, _ = quad(dE_electron_fraction, E_c / E_alpha, 1.0)
    f_e = f_e  # Fraction of total energy to electrons
    f_i = 1 - f_e

    print(f"\n(b) Energy partition:")
    print(f"    Fraction to electrons: f_e ~ {f_e:.2f} ({f_e*100:.0f}%)")
    print(f"    Fraction to ions:      f_i ~ {f_i:.2f} ({f_i*100:.0f}%)")
    print(f"    Energy to electrons: {f_e * E_alpha/eV_to_J/1e6:.2f} MeV")
    print(f"    Energy to ions:      {f_i * E_alpha/eV_to_J/1e6:.2f} MeV")

    # (c) Slowing-down time
    # tau_s = tau_se / 3, where tau_se is electron drag time
    # tau_se = (4*pi*epsilon_0)^2 * m_alpha * m_e * v_th,e^3 / (2*n*Z_alpha^2*e^4*ln_Lambda)
    tau_se = (4 * np.pi * epsilon_0)**2 * m_alpha * m_e * v_th_e**3 / (
        2 * n * Z_alpha**2 * e**4 * ln_Lambda)
    tau_s = tau_se / 3

    print(f"\n(c) Slowing-down time:")
    print(f"    tau_se (electron drag) = {tau_se:.3f} s")
    print(f"    tau_s (total) = tau_se/3 = {tau_s:.3f} s")

    # (d) Compare to confinement time
    tau_E = 3.0  # Energy confinement time [s]
    print(f"\n(d) Comparison with confinement time:")
    print(f"    tau_s = {tau_s:.3f} s, tau_E = {tau_E:.1f} s")
    print(f"    tau_s / tau_E = {tau_s/tau_E:.2f}")
    if tau_s < tau_E:
        print(f"    tau_s < tau_E: Alphas thermalize before being lost -> good for self-heating!")
    else:
        print(f"    tau_s > tau_E: Many alphas lost before thermalizing -> poor self-heating")
    print()


def exercise_3():
    """
    Exercise 3: Classical vs Neoclassical Transport
    Tokamak: R = 3 m, a = 1 m, B = 2 T, n = 5e19 m^-3, T_i = 5 keV (D).
    """
    print("--- Exercise 3: Classical vs Neoclassical Transport ---")

    R = 3.0
    a = 1.0
    B = 2.0
    n = 5e19
    T_i_eV = 5e3
    T_i = T_i_eV * eV_to_J
    m_i = 2 * m_p  # Deuterium
    r = 0.5  # Evaluation radius [m]
    q = 1.5  # Safety factor
    ln_Lambda = 17.0

    epsilon = r / R
    v_th_i = np.sqrt(T_i / m_i)
    omega_ci = e * B / m_i
    rho_i = v_th_i / omega_ci

    # (a) Collisionality parameter
    nu_ii = n * e**4 * ln_Lambda / (6 * np.pi**2 * epsilon_0**2 * m_i**2 * v_th_i**3)
    nu_star = nu_ii * q * R / (epsilon**1.5 * v_th_i)

    print(f"(a) Collisionality at r = {r} m:")
    print(f"    epsilon = r/R = {epsilon:.3f}")
    print(f"    nu_ii = {nu_ii:.4e} s^-1")
    print(f"    nu* = nu_ii * q * R / (epsilon^(3/2) * v_th,i) = {nu_star:.4f}")

    # (b) Regime identification
    if nu_star < 1:
        regime = "Banana"
        print(f"(b) Regime: {regime} (nu* < 1)")
    elif nu_star < epsilon**(-1.5):
        regime = "Plateau"
        print(f"(b) Regime: {regime} (1 < nu* < epsilon^(-3/2) = {epsilon**(-1.5):.1f})")
    else:
        regime = "Pfirsch-Schluter"
        print(f"(b) Regime: {regime} (nu* > epsilon^(-3/2) = {epsilon**(-1.5):.1f})")

    # (c) Classical diffusion coefficient
    D_classical = rho_i**2 * nu_ii
    # Neoclassical enhancement
    if regime == "Banana":
        D_neo = q**2 * rho_i**2 * nu_ii / epsilon**1.5
        enhancement = q**2 / epsilon**1.5
    elif regime == "Plateau":
        D_neo = q * rho_i**2 * v_th_i / (R * epsilon**0.5)
        enhancement = D_neo / D_classical
    else:  # Pfirsch-Schluter
        D_neo = q**2 * rho_i**2 * nu_ii
        enhancement = q**2

    print(f"\n(c) Transport coefficients:")
    print(f"    rho_i = {rho_i*1e3:.2f} mm")
    print(f"    D_classical = rho_i^2 * nu_ii = {D_classical:.4e} m^2/s")
    print(f"    D_neoclassical = {D_neo:.4e} m^2/s")
    print(f"    D_neo / D_classical = {enhancement:.1f}")

    # (d) Compare to anomalous transport
    D_anomalous = 1.0  # m^2/s (typical experimental value)
    print(f"\n(d) Anomalous transport comparison:")
    print(f"    D_anomalous ~ {D_anomalous} m^2/s (typical)")
    print(f"    D_anomalous / D_classical = {D_anomalous/D_classical:.0f}")
    print(f"    D_anomalous / D_neo = {D_anomalous/D_neo:.1f}")
    print(f"    Anomalous >> neoclassical >> classical")
    print(f"    -> Turbulent transport dominates in most tokamak plasmas")
    print()


def exercise_4():
    """
    Exercise 4: Perpendicular vs Parallel Transport
    n = 10^20 m^-3, T_e = 10 keV, B = 5 T.
    """
    print("--- Exercise 4: Perpendicular vs Parallel Transport ---")

    n = 1e20
    T_e_eV = 10e3
    T_e = T_e_eV * eV_to_J
    B = 5.0
    ln_Lambda = 17.0

    v_th_e = np.sqrt(T_e / m_e)
    omega_ce = e * B / m_e
    rho_e = v_th_e / omega_ce
    nu_ei = n * e**4 * ln_Lambda / (6 * np.pi**2 * epsilon_0**2 * m_e**2 * v_th_e**3) * np.sqrt(2)

    # (a) Parallel thermal conductivity
    # kappa_par = 3.16 * n * T_e * tau_e / m_e  (Braginskii)
    tau_e = 1.0 / nu_ei
    kappa_par = 3.16 * n * T_e * tau_e / m_e

    print(f"(a) Parallel thermal conductivity:")
    print(f"    v_th,e = {v_th_e:.4e} m/s")
    print(f"    omega_ce = {omega_ce:.4e} rad/s")
    print(f"    nu_ei = {nu_ei:.4e} s^-1")
    print(f"    omega_ce * tau_e = {omega_ce/nu_ei:.0f}")
    print(f"    kappa_parallel = {kappa_par:.4e} W/(m*K)")

    # (b) Perpendicular thermal conductivity
    # kappa_perp = kappa_par / (omega_ce * tau_e)^2  (Braginskii)
    kappa_perp = kappa_par / (omega_ce / nu_ei)**2

    print(f"\n(b) Perpendicular thermal conductivity:")
    print(f"    kappa_perp = kappa_par / (omega_ce*tau_e)^2 = {kappa_perp:.4e} W/(m*K)")
    print(f"    kappa_par / kappa_perp = (omega_ce*tau_e)^2 = {kappa_par/kappa_perp:.2e}")

    # (c) Cross-field heat flux
    dTdx = 1e6  # K/m (temperature gradient)
    q_perp = kappa_perp * dTdx

    print(f"\n(c) Cross-field heat flux:")
    print(f"    dT/dx = {dTdx:.0e} K/m")
    print(f"    q_perp = kappa_perp * dT/dx = {q_perp:.4e} W/m^2")

    # (d) Equivalent parallel gradient
    dTdz_equiv = q_perp / kappa_par

    print(f"\n(d) Equivalent parallel gradient for same heat flux:")
    print(f"    dT/dz = q_perp / kappa_par = {dTdz_equiv:.4e} K/m")
    print(f"    This is {dTdx/dTdz_equiv:.0e} times smaller than the perp gradient")
    print(f"    -> Parallel transport is extremely efficient")
    print(f"    -> Temperature is nearly constant along field lines")
    print(f"    -> Magnetic field geometry determines temperature profile shape")
    print()


def exercise_5():
    """
    Exercise 5: Bootstrap Current
    Tokamak: p(r) = p_0*(1-r^2/a^2)^2, p_0 = 5e5 Pa, a = 2 m.
    R = 6 m, B_p = 0.5 T, I_p = 15 MA.
    """
    print("--- Exercise 5: Bootstrap Current ---")

    p_0 = 5e5    # Pa
    a = 2.0      # m
    R = 6.0      # m
    B_p = 0.5    # Poloidal field [T]
    I_p = 15e6   # Plasma current [A]
    B_t = 5.0    # Toroidal field [T]

    # (a) Pressure gradient at r = 1 m
    r = 1.0
    # p(r) = p_0 * (1 - r^2/a^2)^2
    # dp/dr = p_0 * 2*(1 - r^2/a^2) * (-2*r/a^2)
    #       = -4*p_0*r/a^2 * (1 - r^2/a^2)

    p_at_r = p_0 * (1 - r**2 / a**2)**2
    dpdr = -4 * p_0 * r / a**2 * (1 - r**2 / a**2)

    print(f"(a) Pressure profile: p(r) = p_0*(1 - r^2/a^2)^2")
    print(f"    p_0 = {p_0:.0e} Pa, a = {a} m")
    print(f"    At r = {r} m:")
    print(f"    p({r}) = {p_at_r:.4e} Pa")
    print(f"    dp/dr = {dpdr:.4e} Pa/m")

    # (b) Bootstrap current density
    # j_bs ~ epsilon^(1/2) / (1 + epsilon^2) * (1/(B_p)) * dp/dr
    # (simplified formula from neoclassical theory)
    epsilon = r / R

    # More standard: j_bs = -sqrt(epsilon) * (1/B_p) * dp/dr * (factor)
    # where factor accounts for geometry ~ 1/(1 + epsilon)
    j_bs = -np.sqrt(epsilon) / (1 + epsilon**2) * (1 / B_p) * dpdr

    print(f"\n(b) Bootstrap current density at r = {r} m:")
    print(f"    epsilon = r/R = {epsilon:.3f}")
    print(f"    j_bs = -epsilon^(1/2)/(1+epsilon^2) * dp/dr / B_p")
    print(f"    j_bs = {j_bs:.4e} A/m^2")
    print(f"    j_bs = {j_bs/1e6:.4f} MA/m^2")

    # (c) Bootstrap current fraction
    # Total bootstrap current: I_bs = integral j_bs * 2*pi*r*dr from 0 to a
    from scipy.integrate import quad

    def j_bs_integrand(r_val):
        if r_val < 1e-6:
            return 0
        eps = r_val / R
        dp = -4 * p_0 * r_val / a**2 * (1 - r_val**2 / a**2)
        if abs(1 - r_val**2 / a**2) < 1e-10:
            return 0
        j = -np.sqrt(eps) / (1 + eps**2) * dp / B_p
        return j * 2 * np.pi * r_val

    I_bs, _ = quad(j_bs_integrand, 0, a * 0.999)

    f_bs = I_bs / I_p

    print(f"\n(c) Bootstrap current fraction:")
    print(f"    I_bs = {I_bs/1e6:.2f} MA")
    print(f"    I_p = {I_p/1e6:.0f} MA")
    print(f"    f_bs = I_bs/I_p = {f_bs:.3f} = {f_bs*100:.1f}%")

    # (d) Why high bootstrap fraction is desirable
    print(f"\n(d) Importance of bootstrap current for fusion reactors:")
    print(f"    - Reduces the need for external current drive (expensive in power)")
    print(f"    - External CD methods (NBI, ECCD, LHCD) are ~30-50% efficient")
    print(f"    - ITER target: f_bs ~ 20-30%")
    print(f"    - DEMO/reactor target: f_bs ~ 50-70% for steady-state operation")
    print(f"    - High bootstrap fraction -> more efficient reactor")
    print(f"    - Requires high beta_p (strong pressure gradient)")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
