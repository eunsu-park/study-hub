"""
Plasma Physics - Lesson 02: Coulomb Collisions
Exercise Solutions

Topics covered:
- Collision frequency calculations for glow discharge
- Spitzer resistivity in tokamak plasmas
- Temperature equilibration between electrons and ions
- Impact parameter estimation
- Collisionality regime identification
"""

import numpy as np

# Physical constants
e = 1.602e-19          # Elementary charge [C]
m_e = 9.109e-31        # Electron mass [kg]
m_p = 1.673e-27        # Proton mass [kg]
epsilon_0 = 8.854e-12  # Vacuum permittivity [F/m]
k_B = 1.381e-23        # Boltzmann constant [J/K]
mu_0 = 4 * np.pi * 1e-7
eV_to_J = e
eV_to_K = e / k_B


def coulomb_logarithm(n_e, T_e_eV):
    """Calculate the Coulomb logarithm for electron-ion collisions."""
    T_e_J = T_e_eV * eV_to_J
    lambda_D = np.sqrt(epsilon_0 * T_e_J / (n_e * e**2))
    b_min = e**2 / (4 * np.pi * epsilon_0 * 3 * T_e_J)  # Classical distance of closest approach
    # Quantum minimum: de Broglie wavelength
    hbar = 1.055e-34
    v_th = np.sqrt(T_e_J / m_e)
    b_min_quantum = hbar / (2 * m_e * v_th)
    b_min_eff = max(b_min, b_min_quantum)
    ln_Lambda = np.log(lambda_D / b_min_eff)
    return ln_Lambda


def nu_ei(n_e, T_e_eV, Z=1, ln_Lambda=None):
    """Electron-ion collision frequency [s^-1]."""
    T_e_J = T_e_eV * eV_to_J
    if ln_Lambda is None:
        ln_Lambda = coulomb_logarithm(n_e, T_e_eV)
    v_th_e = np.sqrt(T_e_J / m_e)
    nu = (n_e * Z**2 * e**4 * ln_Lambda) / (
        6 * np.pi**2 * epsilon_0**2 * m_e**2 * v_th_e**3
    )
    return nu


def nu_ee(n_e, T_e_eV, ln_Lambda=None):
    """Electron-electron collision frequency [s^-1]."""
    # Roughly same order as nu_ei for Z=1
    return nu_ei(n_e, T_e_eV, Z=1, ln_Lambda=ln_Lambda) / np.sqrt(2)


def nu_ii(n_i, T_i_eV, Z=1, A=1, ln_Lambda=None):
    """Ion-ion collision frequency [s^-1]."""
    T_i_J = T_i_eV * eV_to_J
    m_i = A * m_p
    if ln_Lambda is None:
        ln_Lambda = 15.0  # Default
    v_th_i = np.sqrt(T_i_J / m_i)
    nu = (n_i * Z**4 * e**4 * ln_Lambda) / (
        6 * np.pi**2 * epsilon_0**2 * m_i**2 * v_th_i**3
    )
    return nu


def spitzer_resistivity(T_e_eV, Z=1, ln_Lambda=None):
    """Spitzer resistivity [Ohm*m]."""
    T_e_J = T_e_eV * eV_to_J
    if ln_Lambda is None:
        ln_Lambda = 15.0
    eta = (np.pi * Z * e**2 * m_e**0.5 * ln_Lambda) / (
        (4 * np.pi * epsilon_0)**2 * (2 * T_e_J)**1.5
    )
    # More standard formula: eta = 0.51 * m_e * nu_ei / (n_e * e^2)
    # Using the NRL formula: eta_perp = 1.03e-4 * Z * ln_Lambda / T_e_eV^1.5 [Ohm*m]
    eta_nrl = 1.03e-4 * Z * ln_Lambda / T_e_eV**1.5
    return eta_nrl


def exercise_1():
    """
    Exercise 1: Collision Frequencies in a Glow Discharge
    Glow discharge: n = 10^16 m^-3, T_e = 2 eV, T_i = 0.05 eV (room temp), argon.
    """
    print("--- Exercise 1: Collision Frequencies in Glow Discharge ---")

    n = 1e16
    T_e_eV = 2.0
    T_i_eV = 0.05
    Z = 1
    A = 40  # Argon
    m_i = A * m_p

    # (a) Coulomb logarithm
    ln_Lam = coulomb_logarithm(n, T_e_eV)
    print(f"(a) Coulomb logarithm: ln(Lambda) = {ln_Lam:.1f}")

    # (b) Electron-ion collision frequency
    nu_ei_val = nu_ei(n, T_e_eV, Z=Z, ln_Lambda=ln_Lam)
    print(f"(b) Electron-ion collision frequency: nu_ei = {nu_ei_val:.4e} s^-1")
    print(f"    Collision time: tau_ei = {1/nu_ei_val:.4e} s")

    # (c) Electron-electron collision frequency
    nu_ee_val = nu_ee(n, T_e_eV, ln_Lambda=ln_Lam)
    print(f"(c) Electron-electron collision frequency: nu_ee = {nu_ee_val:.4e} s^-1")

    # (d) Ion-ion collision frequency
    nu_ii_val = nu_ii(n, T_i_eV, Z=Z, A=A, ln_Lambda=ln_Lam)
    print(f"(d) Ion-ion collision frequency: nu_ii = {nu_ii_val:.4e} s^-1")

    # (e) Compare to plasma frequency
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    print(f"(e) Plasma frequency: omega_pe = {omega_pe:.4e} rad/s")
    print(f"    nu_ei / omega_pe = {nu_ei_val / omega_pe:.4e}")
    print(f"    Collisionless if nu_ei << omega_pe: {'Yes' if nu_ei_val < 0.01*omega_pe else 'No'}")

    # Mean free path
    v_th_e = np.sqrt(T_e_eV * eV_to_J / m_e)
    mfp_e = v_th_e / nu_ei_val
    print(f"    Electron mean free path: {mfp_e:.2f} m")
    print()


def exercise_2():
    """
    Exercise 2: Spitzer Resistivity in a Tokamak
    Tokamak: n = 10^20 m^-3, T_e = 10 keV, I_p = 15 MA, R = 6 m, a = 2 m.
    """
    print("--- Exercise 2: Spitzer Resistivity in Tokamak ---")

    n = 1e20
    T_e_eV = 10e3  # 10 keV
    I_p = 15e6      # 15 MA
    R = 6.0         # Major radius [m]
    a = 2.0         # Minor radius [m]

    # (a) Coulomb logarithm
    ln_Lam = coulomb_logarithm(n, T_e_eV)
    print(f"(a) Coulomb logarithm: ln(Lambda) = {ln_Lam:.1f}")

    # (b) Spitzer resistivity
    eta = spitzer_resistivity(T_e_eV, Z=1, ln_Lambda=ln_Lam)
    print(f"(b) Spitzer resistivity: eta = {eta:.4e} Ohm*m")

    # Compare to copper at room temperature
    eta_Cu = 1.68e-8  # Ohm*m
    print(f"    Ratio eta_plasma / eta_copper = {eta/eta_Cu:.4e}")
    print(f"    Hot plasma is MUCH better conductor than copper!")

    # (c) Loop voltage and ohmic power
    cross_section = np.pi * a**2
    J = I_p / cross_section  # Current density
    E = eta * J              # Electric field
    V_loop = E * 2 * np.pi * R  # Loop voltage
    print(f"(c) Current density: J = {J:.2e} A/m^2")
    print(f"    Electric field: E = {E:.4e} V/m")
    print(f"    Loop voltage: V_loop = {V_loop*1e3:.4f} mV")

    # (d) Ohmic heating power
    P_ohmic = eta * J**2 * (2 * np.pi * R * cross_section)
    print(f"(d) Ohmic heating power: P_ohmic = {P_ohmic/1e3:.2f} kW = {P_ohmic/1e6:.4f} MW")
    print(f"    At 10 keV, ohmic heating is very weak (T^-3/2 dependence).")
    print(f"    This is why auxiliary heating (NBI, ECRH, ICRH) is needed for hot plasmas.")
    print()


def exercise_3():
    """
    Exercise 3: Temperature Equilibration
    n = 10^20 m^-3, T_e = 15 keV, T_i = 5 keV, deuterium.
    How long until equilibration?
    """
    print("--- Exercise 3: Temperature Equilibration ---")

    n = 1e20
    T_e_eV = 15e3   # 15 keV
    T_i_eV = 5e3    # 5 keV
    A = 2            # Deuterium
    m_i = A * m_p
    ln_Lam = 17.0   # Typical for fusion plasmas

    # (a) Energy equilibration time: tau_eq ~ (m_i / m_e) * tau_ei / 2
    # More precisely: tau_eq = (3*m_e*m_i) / (8*sqrt(2*pi)*n*e^4*ln_Lambda) *
    #                         ((k_B*T_e/m_e + k_B*T_i/m_i)^(3/2))
    T_e_J = T_e_eV * eV_to_J
    T_i_J = T_i_eV * eV_to_J

    # NRL formula: tau_eq = (3 * sqrt(2*pi) * epsilon_0^2 * m_i) / (n * Z^2 * e^4 * ln_Lambda) *
    #              (m_e * T_e + m_i * T_i)^(3/2) / (m_e * m_i)^(1/2)
    # Simplified: tau_eq ~ 0.59 * m_i/(m_p) * T_e_eV^(3/2) / (n * ln_Lambda) * (m_p/m_e)
    # Using standard formula
    v_th_e = np.sqrt(T_e_J / m_e)
    nu_ei_val = nu_ei(n, T_e_eV, Z=1, ln_Lambda=ln_Lam)
    tau_ei = 1.0 / nu_ei_val

    # Energy exchange time: tau_eq ~ (m_i / (2*m_e)) * tau_ei
    tau_eq = (m_i / (2 * m_e)) * tau_ei
    print(f"(a) Electron-ion collision time: tau_ei = {tau_ei:.4e} s")
    print(f"    Energy equilibration time: tau_eq = {tau_eq:.4f} s = {tau_eq*1e3:.1f} ms")
    print(f"    (Mass ratio factor m_i/(2*m_e) = {m_i/(2*m_e):.0f})")

    # (b) Time evolution of temperatures
    print(f"\n(b) Temperature evolution (exponential approach):")
    print(f"    T_e(t) = T_avg + (T_e0 - T_avg) * exp(-t/tau_eq)")
    print(f"    T_i(t) = T_avg + (T_i0 - T_avg) * exp(-t/tau_eq)")
    T_avg = (T_e_eV + T_i_eV) / 2  # Simple average (equal density)
    print(f"    T_avg = {T_avg/1e3:.1f} keV")

    times = np.array([0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]) * tau_eq
    print(f"\n    {'t/tau_eq':>10} {'t [ms]':>10} {'T_e [keV]':>12} {'T_i [keV]':>12}")
    print("    " + "-" * 48)
    for t in times:
        T_e_t = T_avg + (T_e_eV - T_avg) * np.exp(-t / tau_eq)
        T_i_t = T_avg + (T_i_eV - T_avg) * np.exp(-t / tau_eq)
        print(f"    {t/tau_eq:>10.1f} {t*1e3:>10.1f} {T_e_t/1e3:>12.2f} {T_i_t/1e3:>12.2f}")

    # (c) Compare to energy confinement time
    tau_E = 3.0  # Typical ITER confinement time [s]
    print(f"\n(c) Energy confinement time: tau_E = {tau_E:.1f} s")
    print(f"    tau_eq / tau_E = {tau_eq/tau_E:.3f}")
    if tau_eq < tau_E:
        print(f"    tau_eq < tau_E: electrons and ions will equilibrate")
    else:
        print(f"    tau_eq > tau_E: electrons and ions may NOT fully equilibrate")
    print()


def exercise_4():
    """
    Exercise 4: Impact Parameter Estimates
    Calculate classical distance of closest approach and quantum minimum
    for various plasma conditions.
    """
    print("--- Exercise 4: Impact Parameter Estimates ---")

    conditions = [
        ("Room temp plasma", 1e16, 0.05),
        ("Warm plasma", 1e18, 10.0),
        ("Hot fusion plasma", 1e20, 10000.0),
    ]

    hbar = 1.055e-34

    print(f"{'Condition':<22} {'n [m^-3]':>10} {'T_e [eV]':>10} {'b_class [m]':>14} "
          f"{'b_quantum [m]':>14} {'b_min [m]':>14} {'lambda_D [m]':>14} {'ln(Lambda)':>12}")
    print("-" * 120)

    for name, n, T_eV in conditions:
        T_J = T_eV * eV_to_J
        v_th = np.sqrt(T_J / m_e)

        # Classical distance of closest approach: b_class = e^2 / (4*pi*epsilon_0 * m_e * v_th^2)
        b_class = e**2 / (4 * np.pi * epsilon_0 * m_e * v_th**2)

        # Quantum minimum (de Broglie wavelength): b_quantum = hbar / (2 * m_e * v_th)
        b_quantum = hbar / (2 * m_e * v_th)

        b_min = max(b_class, b_quantum)
        lambda_D = np.sqrt(epsilon_0 * T_J / (n * e**2))
        ln_Lambda = np.log(lambda_D / b_min)

        dominant = "classical" if b_class > b_quantum else "quantum"
        print(f"{name:<22} {n:>10.0e} {T_eV:>10.1f} {b_class:>14.4e} "
              f"{b_quantum:>14.4e} {b_min:>14.4e} {lambda_D:>14.4e} {ln_Lambda:>12.1f}")

    print()
    print("Notes:")
    print("  - At low temperatures, classical b_min dominates (thermal energy ~ potential energy)")
    print("  - At high temperatures, quantum b_min dominates (de Broglie wavelength > classical)")
    print("  - The crossover occurs at T ~ 20 eV for electron-ion collisions")
    print("  - ln(Lambda) typically ranges from 10-20 for most plasmas")
    print()


def exercise_5():
    """
    Exercise 5: Collisionality Regimes
    Classify plasmas into collisionality regimes based on nu* = nu_ei * q * R / v_th,i
    """
    print("--- Exercise 5: Collisionality Regimes ---")

    # For a tokamak, collisionality parameter: nu* = nu_ii * q * R / (epsilon^1.5 * v_th,i)
    # where epsilon = r/R (inverse aspect ratio), q = safety factor
    # Banana regime: nu* < 1
    # Plateau regime: 1 < nu* < epsilon^(-3/2)
    # Pfirsch-Schluter: nu* > epsilon^(-3/2)

    print("Collisionality parameter nu* = nu_ii * q * R / (epsilon^(3/2) * v_th,i)")
    print()

    plasmas = [
        ("ITER core", 1e20, 10000, 2, 6.2, 2.0, 1.5),       # n, T_i_eV, A, R, a, q
        ("ITER edge", 1e19, 100, 2, 6.2, 2.0, 4.0),
        ("Small tokamak core", 1e19, 1000, 2, 1.0, 0.25, 1.2),
        ("Small tokamak edge", 5e18, 50, 2, 1.0, 0.25, 3.5),
    ]

    print(f"{'Plasma':<25} {'n [m^-3]':>10} {'T_i [eV]':>10} {'nu_ii [s^-1]':>14} "
          f"{'nu*':>12} {'Regime':>20}")
    print("-" * 100)

    for name, n, T_i_eV, A, R, a, q in plasmas:
        m_i = A * m_p
        T_i_J = T_i_eV * eV_to_J
        v_th_i = np.sqrt(T_i_J / m_i)
        epsilon = a / (2 * R)  # Using r = a/2 as representative point
        ln_Lam = 17.0

        # Ion-ion collision frequency
        nu_ii_val = nu_ii(n, T_i_eV, Z=1, A=A, ln_Lambda=ln_Lam)

        # Collisionality parameter
        nu_star = nu_ii_val * q * R / (epsilon**1.5 * v_th_i)

        if nu_star < 1:
            regime = "Banana"
        elif nu_star < epsilon**(-1.5):
            regime = "Plateau"
        else:
            regime = "Pfirsch-Schluter"

        print(f"{name:<25} {n:>10.0e} {T_i_eV:>10.0f} {nu_ii_val:>14.4e} "
              f"{nu_star:>12.4e} {regime:>20}")

    print()
    print("Regime significance:")
    print("  - Banana: trapped particles complete bounce orbits between collisions")
    print("    -> Neoclassical transport dominated by banana orbits")
    print("  - Plateau: collisions interrupt bounce motion")
    print("    -> Transport independent of collision frequency")
    print("  - Pfirsch-Schluter: highly collisional, MHD-like behavior")
    print("    -> Classical transport enhanced by toroidal geometry")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
