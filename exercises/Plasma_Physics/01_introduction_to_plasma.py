"""
Plasma Physics - Lesson 01: Introduction to Plasma
Exercise Solutions

Topics covered:
- Plasma parameter calculations (density, temperature, Debye length, plasma frequency)
- Debye shielding in tokamak plasmas
- Scaling analysis of plasma parameters
- Solar corona plasma characterization
- Laboratory plasma regime identification
"""

import numpy as np

# Physical constants
e = 1.602e-19          # Elementary charge [C]
m_e = 9.109e-31        # Electron mass [kg]
m_p = 1.673e-27        # Proton mass [kg]
epsilon_0 = 8.854e-12  # Vacuum permittivity [F/m]
k_B = 1.381e-23        # Boltzmann constant [J/K]
mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability [H/m]
eV_to_K = e / k_B      # 1 eV in Kelvin (~11604.5 K)
eV_to_J = e             # 1 eV in Joules


def exercise_1():
    """
    Exercise 1: Plasma Parameter Calculation
    A hydrogen plasma has n = 10^18 m^-3, T_e = 100 eV, T_i = 50 eV.
    Calculate fundamental plasma parameters.
    """
    print("--- Exercise 1: Plasma Parameter Calculation ---")

    n = 1e18            # Electron density [m^-3]
    T_e_eV = 100.0      # Electron temperature [eV]
    T_i_eV = 50.0       # Ion temperature [eV]
    T_e = T_e_eV * eV_to_J  # Convert to Joules
    T_i = T_i_eV * eV_to_J
    m_i = m_p            # Hydrogen ions

    # (a) Debye length: lambda_D = sqrt(epsilon_0 * k_B * T_e / (n * e^2))
    lambda_D = np.sqrt(epsilon_0 * T_e / (n * e**2))
    print(f"(a) Debye length: lambda_D = {lambda_D*1e6:.2f} um = {lambda_D:.4e} m")

    # (b) Plasma frequency: omega_pe = sqrt(n * e^2 / (epsilon_0 * m_e))
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    f_pe = omega_pe / (2 * np.pi)
    print(f"(b) Electron plasma frequency: omega_pe = {omega_pe:.4e} rad/s")
    print(f"    f_pe = {f_pe:.4e} Hz = {f_pe/1e9:.2f} GHz")

    # Ion plasma frequency
    omega_pi = np.sqrt(n * e**2 / (epsilon_0 * m_i))
    f_pi = omega_pi / (2 * np.pi)
    print(f"    Ion plasma frequency: omega_pi = {omega_pi:.4e} rad/s")
    print(f"    f_pi = {f_pi:.4e} Hz = {f_pi/1e6:.2f} MHz")

    # (c) Number of particles in Debye sphere: N_D = (4/3) * pi * n * lambda_D^3
    N_D = (4.0 / 3.0) * np.pi * n * lambda_D**3
    print(f"(c) Particles in Debye sphere: N_D = {N_D:.2e}")
    print(f"    N_D >> 1? {'Yes' if N_D > 100 else 'No'} -> {'Collective behavior valid' if N_D > 100 else 'Marginal'}")

    # (d) Plasma coupling parameter: Gamma = (e^2 / (4*pi*epsilon_0)) / (k_B*T * a_ws)
    # where a_ws = (3/(4*pi*n))^(1/3) is the Wigner-Seitz radius
    a_ws = (3.0 / (4.0 * np.pi * n))**(1.0 / 3.0)
    Gamma = (e**2 / (4 * np.pi * epsilon_0 * a_ws)) / T_e
    print(f"(d) Coupling parameter: Gamma = {Gamma:.4e}")
    print(f"    Gamma << 1? {'Yes (weakly coupled)' if Gamma < 0.01 else 'No (strongly coupled)'}")

    # (e) Thermal velocities
    v_th_e = np.sqrt(T_e / m_e)
    v_th_i = np.sqrt(T_i / m_i)
    print(f"(e) Electron thermal velocity: v_th,e = {v_th_e:.4e} m/s")
    print(f"    Ion thermal velocity:      v_th,i = {v_th_i:.4e} m/s")
    print(f"    Ratio v_th,e/v_th,i = {v_th_e/v_th_i:.1f}")
    print()


def exercise_2():
    """
    Exercise 2: Debye Shielding in a Tokamak
    Tokamak plasma: n = 10^20 m^-3, T_e = 10 keV.
    Analyze Debye shielding for an impurity ion.
    """
    print("--- Exercise 2: Debye Shielding in a Tokamak ---")

    n = 1e20             # Density [m^-3]
    T_e_eV = 10e3        # T_e = 10 keV
    T_e = T_e_eV * eV_to_J
    Z_imp = 6            # Carbon impurity charge

    # (a) Debye length
    lambda_D = np.sqrt(epsilon_0 * T_e / (n * e**2))
    print(f"(a) Debye length: lambda_D = {lambda_D*1e6:.3f} um = {lambda_D:.4e} m")

    # (b) Number of particles in Debye sphere
    N_D = (4.0 / 3.0) * np.pi * n * lambda_D**3
    print(f"(b) Particles in Debye sphere: N_D = {N_D:.2e}")

    # (c) Shielded potential of impurity: phi(r) = (Z*e / 4*pi*epsilon_0*r) * exp(-r/lambda_D)
    # Compare with unshielded (Coulomb) potential at various distances
    distances = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0]) * lambda_D
    print(f"(c) Shielded potential of C{Z_imp}+ impurity:")
    print(f"    Distance [lambda_D]  |  phi_shielded/phi_Coulomb  |  Shielding fraction")
    for r in distances:
        ratio = np.exp(-r / lambda_D)
        print(f"    {r/lambda_D:8.1f}             |  {ratio:22.6f}         |  {(1-ratio)*100:.2f}%")

    # (d) Comparison with tokamak minor radius
    a = 1.0  # Typical minor radius [m]
    ratio_a = a / lambda_D
    print(f"(d) Tokamak minor radius a = {a} m")
    print(f"    a / lambda_D = {ratio_a:.2e}")
    print(f"    The Debye length is extremely small compared to the device size,")
    print(f"    confirming quasi-neutrality is well satisfied in the bulk plasma.")
    print()


def exercise_3():
    """
    Exercise 3: Scaling Analysis
    How plasma parameters scale with n and T.
    """
    print("--- Exercise 3: Scaling Analysis ---")

    # Base parameters
    n_base = 1e18
    T_base_eV = 10.0
    T_base = T_base_eV * eV_to_J

    # Calculate scaling for a range of densities and temperatures
    densities = np.logspace(10, 25, 16)  # m^-3
    temperatures_eV = np.array([0.1, 1, 10, 100, 1000, 10000])  # eV

    print("(a) Scaling relations:")
    print("    lambda_D ~ sqrt(T/n)    => lambda_D propto T^(1/2) * n^(-1/2)")
    print("    omega_pe ~ sqrt(n)      => omega_pe propto n^(1/2)")
    print("    N_D ~ n * lambda_D^3    => N_D propto T^(3/2) * n^(-1/2)")
    print()

    # (b) Calculate for specific plasma types
    print("(b) Plasma parameters across regimes:")
    print(f"{'Plasma Type':<25} {'n [m^-3]':>12} {'T_e [eV]':>10} {'lambda_D [m]':>14} {'omega_pe [rad/s]':>18} {'N_D':>12}")
    print("-" * 95)

    plasma_types = [
        ("Interstellar medium", 1e6, 1.0),
        ("Solar wind (1 AU)", 5e6, 10.0),
        ("Ionosphere", 1e12, 0.1),
        ("Glow discharge", 1e16, 2.0),
        ("Tokamak edge", 1e19, 100.0),
        ("Tokamak core", 1e20, 10000.0),
        ("Inertial fusion", 1e31, 10000.0),
        ("White dwarf", 1e36, 1000.0),
    ]

    for name, n, T_eV in plasma_types:
        T = T_eV * eV_to_J
        lam_D = np.sqrt(epsilon_0 * T / (n * e**2))
        w_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
        N_D = (4.0 / 3.0) * np.pi * n * lam_D**3
        print(f"{name:<25} {n:>12.2e} {T_eV:>10.1f} {lam_D:>14.4e} {w_pe:>18.4e} {N_D:>12.2e}")

    print()
    print("(c) Validity conditions:")
    print("    Plasma state requires: lambda_D << L (system size), N_D >> 1, omega_pe*tau >> 1")
    print("    Strongly coupled plasmas (white dwarf) may violate N_D >> 1 at low T")
    print()


def exercise_4():
    """
    Exercise 4: Solar Corona Parameters
    Solar corona: n ~ 10^14 m^-3, T ~ 10^6 K (~ 86 eV).
    Calculate key plasma parameters and assess plasma conditions.
    """
    print("--- Exercise 4: Solar Corona Parameters ---")

    n = 1e14                    # Coronal density [m^-3]
    T_K = 1e6                   # Temperature [K]
    T_eV = T_K / eV_to_K       # Convert to eV
    T_J = k_B * T_K             # Convert to Joules
    B = 1e-3                    # Typical coronal B field [T] (1 Gauss = 10^-4 T, ~10 Gauss)

    print(f"Input: n = {n:.0e} m^-3, T = {T_K:.0e} K = {T_eV:.1f} eV, B = {B*1e3:.0f} mT")

    # (a) Debye length
    lambda_D = np.sqrt(epsilon_0 * T_J / (n * e**2))
    print(f"\n(a) Debye length: lambda_D = {lambda_D:.2f} m")

    # (b) Plasma frequency
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    f_pe = omega_pe / (2 * np.pi)
    print(f"(b) Plasma frequency: f_pe = {f_pe/1e6:.2f} MHz")

    # (c) Particles in Debye sphere
    N_D = (4.0 / 3.0) * np.pi * n * lambda_D**3
    print(f"(c) Particles in Debye sphere: N_D = {N_D:.2e}")

    # (d) Electron gyrofrequency and Larmor radius
    omega_ce = e * B / m_e
    f_ce = omega_ce / (2 * np.pi)
    v_th_e = np.sqrt(T_J / m_e)
    rho_e = v_th_e / omega_ce
    print(f"(d) Electron cyclotron frequency: f_ce = {f_ce/1e6:.2f} MHz")
    print(f"    Electron thermal velocity: v_th,e = {v_th_e:.4e} m/s")
    print(f"    Electron Larmor radius: rho_e = {rho_e:.4f} m")

    # (e) Ion (proton) gyrofrequency and Larmor radius
    omega_ci = e * B / m_p
    f_ci = omega_ci / (2 * np.pi)
    v_th_i = np.sqrt(T_J / m_p)
    rho_i = v_th_i / omega_ci
    print(f"(e) Ion cyclotron frequency: f_ci = {f_ci:.2f} Hz")
    print(f"    Ion thermal velocity: v_th,i = {v_th_i:.4e} m/s")
    print(f"    Ion Larmor radius: rho_i = {rho_i:.2f} m")

    # (f) Plasma beta
    p = n * T_J * 2  # Total pressure (electrons + ions, T_e = T_i)
    B_pressure = B**2 / (2 * mu_0)
    beta = p / B_pressure
    print(f"(f) Plasma beta: beta = {beta:.4f}")
    print(f"    Magnetic pressure dominated? {'Yes (beta < 1)' if beta < 1 else 'No (beta > 1)'}")

    # (g) Alfven speed
    rho = n * m_p  # Mass density (protons)
    v_A = B / np.sqrt(mu_0 * rho)
    print(f"(g) Alfven speed: v_A = {v_A/1e3:.1f} km/s = {v_A:.4e} m/s")

    # (h) Coulomb logarithm
    ln_Lambda = np.log(12 * np.pi * N_D)
    print(f"(h) Coulomb logarithm: ln(Lambda) = {ln_Lambda:.1f}")
    print()


def exercise_5():
    """
    Exercise 5: Laboratory Plasma Regimes
    Compare parameters of different lab plasmas and identify their regimes.
    """
    print("--- Exercise 5: Laboratory Plasma Regimes ---")

    lab_plasmas = [
        ("DC glow discharge (Ar)", 1e16, 2.0, 0.0, 40),
        ("RF CCP (Ar)", 1e17, 3.0, 0.0, 40),
        ("Helicon source", 1e19, 5.0, 0.05, 40),
        ("Tokamak (ITER-like)", 1e20, 10000.0, 5.0, 2),
        ("Laser-produced", 1e26, 1000.0, 0.0, 1),
        ("Z-pinch", 1e24, 100.0, 100.0, 2),
    ]

    print(f"{'Plasma':<25} {'n [m^-3]':>10} {'T_e [eV]':>10} {'B [T]':>7} "
          f"{'lambda_D [m]':>14} {'omega_pe [s^-1]':>16} {'N_D':>12} {'beta':>10}")
    print("-" * 110)

    for name, n, T_eV, B, A in lab_plasmas:
        T_J = T_eV * eV_to_J
        m_i = A * m_p

        lambda_D = np.sqrt(epsilon_0 * T_J / (n * e**2))
        omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
        N_D = (4.0 / 3.0) * np.pi * n * lambda_D**3

        if B > 0:
            p = n * T_J  # Electron pressure only for simplicity
            B_pressure = B**2 / (2 * mu_0)
            beta = 2 * p / B_pressure  # Factor 2 for electrons + ions
        else:
            beta = float('inf')

        beta_str = f"{beta:.4f}" if beta != float('inf') else "N/A (B=0)"
        print(f"{name:<25} {n:>10.2e} {T_eV:>10.1f} {B:>7.2f} "
              f"{lambda_D:>14.4e} {omega_pe:>16.4e} {N_D:>12.2e} {beta_str:>10}")

    print()
    print("Regime classification:")
    print("  - Glow discharge: weakly ionized, collisional, unmagnetized")
    print("  - RF CCP: weakly ionized, moderate collisionality")
    print("  - Helicon: fully ionized, magnetized, moderate density")
    print("  - Tokamak: fully ionized, strongly magnetized, high N_D")
    print("  - Laser-produced: very dense, short-lived, may be strongly coupled")
    print("  - Z-pinch: very dense, strongly magnetized, transient")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
