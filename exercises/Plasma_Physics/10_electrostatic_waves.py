"""
Plasma Physics - Lesson 10: Electrostatic Waves
Exercise Solutions

Topics covered:
- Langmuir wave properties (Bohm-Gross, phase/group velocity)
- Ion acoustic wave damping in argon plasma
- Upper and lower hybrid resonances in tokamak
- Bernstein wave heating in overdense plasma
- Full electrostatic dispersion relation numerical solution
"""

import numpy as np
from scipy.optimize import brentq, fsolve

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
m_p = 1.673e-27
epsilon_0 = 8.854e-12
k_B = 1.381e-23
mu_0 = 4 * np.pi * 1e-7
eV_to_J = e
c = 3e8


def exercise_1():
    """
    Exercise 1: Langmuir Wave Properties
    n = 5e18 m^-3, T_e = 5 eV.
    """
    print("--- Exercise 1: Langmuir Wave Properties ---")

    n = 5e18
    T_eV = 5.0
    T_J = T_eV * eV_to_J

    # (a) Plasma frequency and Debye length
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    f_pe = omega_pe / (2 * np.pi)
    v_th = np.sqrt(T_J / m_e)
    lambda_D = np.sqrt(epsilon_0 * T_J / (n * e**2))

    print(f"(a) n = {n:.0e} m^-3, T_e = {T_eV} eV")
    print(f"    Plasma frequency: f_pe = {f_pe/1e9:.3f} GHz")
    print(f"    omega_pe = {omega_pe:.4e} rad/s")
    print(f"    Debye length: lambda_D = {lambda_D*1e6:.1f} um = {lambda_D:.4e} m")
    print(f"    v_th,e = {v_th:.4e} m/s")

    # (b) Bohm-Gross frequency at k = 0.1/lambda_D
    k = 0.1 / lambda_D
    k_lD = k * lambda_D
    omega_BG = np.sqrt(omega_pe**2 + 3 * k**2 * v_th**2)
    f_BG = omega_BG / (2 * np.pi)

    print(f"\n(b) Bohm-Gross at k = 0.1/lambda_D:")
    print(f"    k = {k:.4e} m^-1, k*lambda_D = {k_lD:.3f}")
    print(f"    omega_BG = {omega_BG:.4e} rad/s")
    print(f"    f_BG = {f_BG/1e9:.4f} GHz")
    print(f"    Thermal correction: {3*k_lD**2:.4f} ({3*k_lD**2*100:.2f}%)")

    # (c) Phase velocity and group velocity
    v_phi = omega_BG / k
    v_g = 3 * k * v_th**2 / omega_BG

    print(f"\n(c) Phase and group velocities:")
    print(f"    v_phi = omega/k = {v_phi:.4e} m/s = {v_phi/v_th:.2f} v_th")
    print(f"    v_g = d(omega)/dk = {v_g:.4e} m/s = {v_g/v_th:.4f} v_th")
    print(f"    v_phi / v_th = {v_phi/v_th:.2f}  (superluminal relative to v_th)")
    print(f"    v_g / v_th = {v_g/v_th:.4f}  (subluminal)")
    print(f"    v_phi * v_g = {v_phi*v_g:.4e} ~ 3*v_th^2 = {3*v_th**2:.4e}")

    # (d) k where thermal correction is 10%
    # 3*k^2*v_th^2 / omega_pe^2 = 0.1
    # k*lambda_D = sqrt(0.1/3) since v_th = omega_pe * lambda_D
    k_lD_10 = np.sqrt(0.1 / 3)
    k_10 = k_lD_10 / lambda_D
    wavelength_10 = 2 * np.pi / k_10

    print(f"\n(d) 10% thermal correction:")
    print(f"    k*lambda_D = sqrt(0.1/3) = {k_lD_10:.4f}")
    print(f"    k = {k_10:.4e} m^-1")
    print(f"    Wavelength = {wavelength_10*1e6:.1f} um")
    print(f"    = {wavelength_10/lambda_D:.1f} lambda_D")
    print()


def exercise_2():
    """
    Exercise 2: Ion Acoustic Wave Damping
    Argon plasma (A=40): n = 10^19 m^-3, T_e = 20 eV, T_i = 2 eV.
    """
    print("--- Exercise 2: Ion Acoustic Wave Damping ---")

    n = 1e19
    T_e_eV = 20.0
    T_i_eV = 2.0
    T_e = T_e_eV * eV_to_J
    T_i = T_i_eV * eV_to_J
    A = 40  # Argon
    m_i = A * m_p

    lambda_D = np.sqrt(epsilon_0 * T_e / (n * e**2))
    v_th_i = np.sqrt(T_i / m_i)
    v_th_e = np.sqrt(T_e / m_e)

    # (a) Ion acoustic speed
    c_s = np.sqrt((T_e + 3 * T_i) / m_i)
    c_s_simple = np.sqrt(T_e / m_i)

    print(f"(a) Ion acoustic speed:")
    print(f"    c_s = sqrt((T_e + 3*T_i)/m_i) = {c_s:.1f} m/s")
    print(f"    c_s (T_e only) = sqrt(T_e/m_i) = {c_s_simple:.1f} m/s")
    print(f"    c_s / v_th,i = {c_s/v_th_i:.2f}")
    print(f"    T_e/T_i = {T_e_eV/T_i_eV:.0f} >> 1 -> ions weakly damped")

    # (b) At k*lambda_D = 0.3
    k_lD = 0.3
    k = k_lD / lambda_D

    # Ion acoustic dispersion: omega = k*c_s / sqrt(1 + k^2*lambda_D^2)
    omega_r = k * c_s / np.sqrt(1 + k_lD**2)

    # Landau damping: dominated by ions
    # gamma_i ~ -sqrt(pi/8) * omega_r * sqrt(m_e/m_i) * (T_e/T_i)^1.5 * exp(-T_e/(2*T_i))
    # at this k, the ion damping comes from the ion contribution
    zeta_i = omega_r / (k * v_th_i * np.sqrt(2))
    gamma_i = -np.sqrt(np.pi / 8) * omega_r * (zeta_i**3) * np.exp(-zeta_i**2)

    # Alternatively use the full expression:
    gamma_approx = -np.sqrt(np.pi / 8) * omega_r * np.sqrt(m_e / m_i) * (T_e / T_i)**1.5 * np.exp(-T_e / (2 * T_i))

    print(f"\n(b) At k*lambda_D = {k_lD}:")
    print(f"    omega_r = {omega_r:.4e} rad/s")
    print(f"    f_r = {omega_r/(2*np.pi):.4e} Hz")
    print(f"    zeta_i = omega/(k*v_th,i*sqrt(2)) = {zeta_i:.3f}")
    print(f"    gamma (ion Landau) = {gamma_i:.4e} rad/s")
    print(f"    |gamma/omega_r| = {abs(gamma_i)/omega_r:.4e}")

    # (c) Damping length
    v_g = c_s * (1 + k_lD**2)**(-1.5)  # d(omega)/dk for IA waves
    L_d = abs(v_g / gamma_i) if gamma_i != 0 else float('inf')

    print(f"\n(c) Damping length:")
    print(f"    v_g = {v_g:.1f} m/s")
    print(f"    L_d = v_g / |gamma| = {L_d:.4e} m")
    print(f"    L_d / lambda = {L_d * k / (2*np.pi):.1f} wavelengths")

    # (d) Effect of increasing T_i to 10 eV
    T_i_new_eV = 10.0
    T_i_new = T_i_new_eV * eV_to_J
    v_th_i_new = np.sqrt(T_i_new / m_i)
    c_s_new = np.sqrt((T_e + 3 * T_i_new) / m_i)
    omega_r_new = k * c_s_new / np.sqrt(1 + k_lD**2)
    zeta_i_new = omega_r_new / (k * v_th_i_new * np.sqrt(2))
    gamma_i_new = -np.sqrt(np.pi / 8) * omega_r_new * (zeta_i_new**3) * np.exp(-zeta_i_new**2)

    print(f"\n(d) With T_i = {T_i_new_eV} eV (T_e/T_i = {T_e_eV/T_i_new_eV:.0f}):")
    print(f"    c_s = {c_s_new:.1f} m/s")
    print(f"    zeta_i = {zeta_i_new:.3f}")
    print(f"    |gamma/omega_r| = {abs(gamma_i_new)/omega_r_new:.4f}")
    observable = abs(gamma_i_new) / omega_r_new < 0.3
    print(f"    Wave still observable? {'Yes' if observable else 'No (heavily damped)'}")
    print(f"    As T_i -> T_e: c_s -> v_th,i, strong ion Landau damping")
    print()


def exercise_3():
    """
    Exercise 3: Hybrid Resonances in Tokamak
    B = 3 T, density varies from n = 10^20 (core) to 10^18 (edge).
    """
    print("--- Exercise 3: Hybrid Resonances ---")

    B = 3.0
    n_core = 1e20
    n_edge = 1e18
    A = 2  # Deuterium
    m_i = A * m_p

    omega_ce = e * B / m_e
    omega_ci = e * B / m_i
    f_ce = omega_ce / (2 * np.pi)
    f_ci = omega_ci / (2 * np.pi)

    print(f"Tokamak: B = {B} T, deuterium")
    print(f"  f_ce = {f_ce/1e9:.2f} GHz = {f_ce:.4e} Hz")
    print(f"  f_ci = {f_ci/1e6:.2f} MHz = {f_ci:.4e} Hz")

    # (a) Upper hybrid frequency: f_UH = sqrt(f_pe^2 + f_ce^2)
    for label, n_val in [("Core", n_core), ("Edge", n_edge)]:
        omega_pe = np.sqrt(n_val * e**2 / (epsilon_0 * m_e))
        f_pe = omega_pe / (2 * np.pi)
        f_UH = np.sqrt(f_pe**2 + f_ce**2)
        print(f"\n(a) {label} (n = {n_val:.0e} m^-3):")
        print(f"    f_pe = {f_pe/1e9:.2f} GHz")
        print(f"    f_UH = sqrt(f_pe^2 + f_ce^2) = {f_UH/1e9:.2f} GHz")

    # (b) 110 GHz heating: where is UH resonance?
    f_heat = 110e9  # Hz
    # f_UH^2 = f_pe^2 + f_ce^2 = f_heat^2
    # -> f_pe^2 = f_heat^2 - f_ce^2
    omega_pe_res = 2 * np.pi * np.sqrt(f_heat**2 - f_ce**2)
    n_res = omega_pe_res**2 * epsilon_0 * m_e / e**2

    print(f"\n(b) 110 GHz ECRH system:")
    print(f"    f_heat = {f_heat/1e9} GHz")
    print(f"    UH resonance at f_pe = {np.sqrt(f_heat**2 - f_ce**2)/1e9:.2f} GHz")
    print(f"    Corresponding density: n = {n_res:.2e} m^-3")
    if n_edge < n_res < n_core:
        print(f"    Location: within plasma (between edge and core)")
    elif n_res > n_core:
        print(f"    Location: density higher than core -> resonance not accessible")
    else:
        print(f"    Location: density lower than edge -> resonance outside plasma")

    # (c) Lower hybrid frequency
    print(f"\n(c) Lower hybrid frequency:")
    for label, n_val in [("Core", n_core), ("Edge", n_edge)]:
        omega_pe = np.sqrt(n_val * e**2 / (epsilon_0 * m_e))
        omega_pi = np.sqrt(n_val * e**2 / (epsilon_0 * m_i))
        f_pe = omega_pe / (2 * np.pi)
        f_pi = omega_pi / (2 * np.pi)

        # Lower hybrid: 1/omega_LH^2 = 1/(omega_pi^2 + omega_ci^2) + 1/(omega_ce*omega_ci + omega_pe^2)
        # Simplified for omega_pe >> omega_ce: f_LH ~ sqrt(f_ci * f_ce) * sqrt(1 + f_pi^2/f_ci^2) / sqrt(1 + f_pe^2/f_ce^2)
        # More practical: omega_LH^2 = omega_ci*omega_ce * (1 + omega_pe^2/omega_ce^2) / (1 + omega_pe^2/omega_ce^2)
        # Standard: 1/omega_LH^2 = 1/(omega_ci^2 + omega_pi^2) + 1/(omega_ci*omega_ce)
        # For dense plasma: omega_LH ~ sqrt(omega_ci * omega_ce) when omega_pe >> omega_ce

        omega_LH_sq = 1.0 / (1.0 / (omega_ci**2 + omega_pi**2) + 1.0 / (omega_ci * omega_ce))
        f_LH = np.sqrt(omega_LH_sq) / (2 * np.pi)

        f_LH_approx = np.sqrt(f_ci * f_ce)  # Dense plasma limit

        print(f"    {label} (n = {n_val:.0e}): f_LH = {f_LH/1e9:.3f} GHz")
        print(f"      Dense plasma approximation: f_LH ~ sqrt(f_ci*f_ce) = {f_LH_approx/1e9:.3f} GHz")

    # (d) Lower hybrid heating at 5 GHz
    f_LH_heat = 5e9
    print(f"\n(d) Lower hybrid system at {f_LH_heat/1e9} GHz:")
    print(f"    LH resonance is between {f_ci/1e6:.1f} MHz and {f_ce/1e9:.1f} GHz")
    print(f"    5 GHz is above f_LH for most densities -> propagates inward")
    print(f"    Resonance at the location where f_LH = 5 GHz")
    print(f"    LH heating used for current drive (LHCD) in tokamaks")
    print()


def exercise_4():
    """
    Exercise 4: Bernstein Wave Heating
    Dense plasma with omega_pe > omega_ce.
    """
    print("--- Exercise 4: Bernstein Wave Heating ---")

    B = 1.5    # T
    T_e_eV = 3e3  # 3 keV
    T_e = T_e_eV * eV_to_J

    omega_ce = e * B / m_e
    f_ce = omega_ce / (2 * np.pi)

    # (a) Maximum density for omega_pe < omega_ce
    # omega_pe = omega_ce -> n_max = epsilon_0 * m_e * omega_ce^2 / e^2
    n_max = epsilon_0 * m_e * omega_ce**2 / e**2

    print(f"(a) B = {B} T, T_e = {T_e_eV/1e3} keV")
    print(f"    omega_ce = {omega_ce:.4e} rad/s, f_ce = {f_ce/1e9:.2f} GHz")
    print(f"    Max density for omega_pe < omega_ce:")
    print(f"    n_max = {n_max:.4e} m^-3")

    # (b) Overdense case
    n_overdense = 5e20
    omega_pe = np.sqrt(n_overdense * e**2 / (epsilon_0 * m_e))
    f_pe = omega_pe / (2 * np.pi)
    f_UH = np.sqrt(f_pe**2 + f_ce**2)

    print(f"\n(b) Overdense plasma (n = {n_overdense:.0e} m^-3):")
    print(f"    f_pe = {f_pe/1e9:.2f} GHz")
    print(f"    omega_pe / omega_ce = {omega_pe/omega_ce:.2f}")
    print(f"    f_UH / f_ce = {f_UH/f_ce:.2f}")

    # (c) Mode conversion to Bernstein wave
    print(f"\n(c) Mode conversion strategy:")
    print(f"    O-mode at omega = omega_ce cannot penetrate (overdense)")
    print(f"    Strategy: Launch O-mode from low-field side")
    print(f"    -> O-mode reaches O-mode cutoff (omega = omega_pe)")
    print(f"    -> Mode converts to slow X-mode near the cutoff")
    print(f"    -> X-mode propagates to UH resonance layer")
    print(f"    -> Mode converts to electron Bernstein wave (EBW)")
    print(f"    -> EBW is electrostatic, not affected by density cutoff")
    print(f"    -> EBW propagates to cyclotron resonance and is absorbed")
    print(f"    This is called O-X-B mode conversion")

    # (d) Bernstein wave wavenumber
    v_th_e = np.sqrt(T_e / m_e)
    rho_e = v_th_e / omega_ce

    print(f"\n(d) Bernstein wave parameters:")
    print(f"    v_th,e = {v_th_e:.4e} m/s")
    print(f"    rho_e = {rho_e*1e3:.3f} mm")
    print(f"    For efficient 1st harmonic heating:")
    print(f"    k_perp * rho_e ~ 1")
    print(f"    k_perp ~ 1/rho_e = {1/rho_e:.4e} m^-1")
    print(f"    Wavelength ~ 2*pi*rho_e = {2*np.pi*rho_e*1e3:.2f} mm")
    print(f"    This is much shorter than EM waves -> electrostatic character")
    print()


def exercise_5():
    """
    Exercise 5: Full Electrostatic Dispersion Relation
    Solve epsilon(omega, k) = 0 numerically for electron-ion plasma.
    """
    print("--- Exercise 5: Dispersion Relation Analysis ---")

    n = 1e18
    T_e_eV = 1000.0  # 1 keV
    T_i_eV = 1000.0  # 1 keV
    T_e = T_e_eV * eV_to_J
    T_i = T_i_eV * eV_to_J
    m_i = m_p  # Hydrogen (m_i/m_e = 1836)

    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    omega_pi = np.sqrt(n * e**2 / (epsilon_0 * m_i))
    v_th_e = np.sqrt(T_e / m_e)
    v_th_i = np.sqrt(T_i / m_i)
    lambda_D = v_th_e / omega_pe

    print(f"Parameters: n = {n:.0e}, T_e = T_i = {T_e_eV} eV, hydrogen")
    print(f"  omega_pe = {omega_pe:.4e}, omega_pi = {omega_pi:.4e}")
    print(f"  v_th,e = {v_th_e:.4e}, v_th,i = {v_th_i:.4e}")
    print(f"  lambda_D = {lambda_D:.4e} m")
    print(f"  mass ratio: m_i/m_e = {m_i/m_e:.0f}")

    # (a) Langmuir branch: expand for omega >> k*v_th_e
    # 1 - omega_pe^2/omega^2 * (1 + 3*k^2*v_th_e^2/omega^2) = 0
    # -> omega^2 = omega_pe^2 + 3*k^2*v_th_e^2 (Bohm-Gross)

    k_lD_range = np.array([0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])

    print(f"\n(a) Langmuir branch (Bohm-Gross):")
    print(f"    {'k*lambda_D':>12} {'omega/omega_pe':>16}")
    print("    " + "-" * 30)
    for k_lD in k_lD_range:
        if k_lD < 1.0:
            omega_L = np.sqrt(omega_pe**2 + 3 * (k_lD / lambda_D)**2 * v_th_e**2)
            print(f"    {k_lD:>12.3f} {omega_L/omega_pe:>16.6f}")

    # (b) Ion acoustic branch: omega << k*v_th_e, omega ~ k*v_th_i
    # c_s = sqrt(T_e/m_i) (since T_e = T_i and electron Boltzmann)
    c_s = np.sqrt(T_e / m_i)
    print(f"\n(b) Ion acoustic branch:")
    print(f"    c_s = sqrt(T_e/m_i) = {c_s:.1f} m/s")
    print(f"    omega = k*c_s / sqrt(1 + k^2*lambda_D^2)")
    print(f"\n    {'k*lambda_D':>12} {'omega/omega_pi':>16}")
    print("    " + "-" * 30)
    for k_lD in k_lD_range:
        k = k_lD / lambda_D
        omega_IA = k * c_s / np.sqrt(1 + k_lD**2)
        print(f"    {k_lD:>12.3f} {omega_IA/omega_pi:>16.6f}")

    # (c) Crossover point
    print(f"\n(c) Branch crossover:")
    print(f"    Langmuir: omega ~ omega_pe for small k")
    print(f"    Ion acoustic: omega ~ k*c_s for small k")
    print(f"    They cross when omega_pe ~ k*c_s")
    k_cross = omega_pe / c_s
    k_lD_cross = k_cross * lambda_D
    print(f"    k_crossover * lambda_D = omega_pe/(c_s/lambda_D) = {k_lD_cross:.1f}")
    print(f"    At this scale, both branches have omega ~ omega_pe")
    print(f"    This is the regime where kinetic effects are essential")
    print(f"    (strong Landau damping on both branches)")

    # (d) Ratio at crossover
    print(f"\n(d) At k*lambda_D ~ {k_lD_cross:.0f}:")
    print(f"    Both branches strongly damped")
    print(f"    Langmuir waves: overdamped (v_phi ~ v_th,e)")
    print(f"    Ion acoustic: overdamped (v_phi ~ v_th,i for T_e = T_i)")
    print(f"    Pure electrostatic description breaks down at very high k")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
