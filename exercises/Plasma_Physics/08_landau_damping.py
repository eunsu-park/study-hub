"""
Plasma Physics - Lesson 08: Landau Damping
Exercise Solutions

Topics covered:
- Bohm-Gross dispersion relation analysis
- Landau damping rate at different k*lambda_D
- Bump-on-tail instability growth rate
- Ion acoustic wave damping with Te/Ti ratio effects
- Particle trapping and bounce frequency
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.special import erfc

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
m_p = 1.673e-27
epsilon_0 = 8.854e-12
k_B = 1.381e-23
eV_to_J = e


def exercise_1():
    """
    Exercise 1: Bohm-Gross Dispersion Relation
    omega^2 = omega_pe^2 + 3*k^2*v_the^2
    Analyze phase/group velocity and thermal corrections.
    """
    print("--- Exercise 1: Bohm-Gross Dispersion ---")

    n = 1e18
    T_eV = 10.0
    T_J = T_eV * eV_to_J
    v_th = np.sqrt(T_J / m_e)
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    lambda_D = v_th / omega_pe

    print(f"Parameters: n = {n:.0e} m^-3, T_e = {T_eV} eV")
    print(f"  omega_pe = {omega_pe:.4e} rad/s, f_pe = {omega_pe/(2*np.pi)/1e9:.3f} GHz")
    print(f"  v_th = {v_th:.4e} m/s")
    print(f"  lambda_D = {lambda_D:.4e} m")

    # (a) Dispersion for range of k
    k_values = np.logspace(-2, 0, 50) / lambda_D

    print(f"\n(a) Bohm-Gross dispersion: omega^2 = omega_pe^2 + 3*k^2*v_th^2")
    print(f"\n    {'k*lambda_D':>12} {'omega/omega_pe':>16} {'v_phi/v_th':>12} {'v_g/v_th':>12} {'thermal corr':>14}")
    print("    " + "-" * 70)

    for k_lD in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        k = k_lD / lambda_D
        omega_sq = omega_pe**2 + 3 * k**2 * v_th**2
        omega = np.sqrt(omega_sq)
        v_phi = omega / k
        v_g = 3 * k * v_th**2 / omega
        thermal_corr = 3 * k**2 * v_th**2 / omega_pe**2

        print(f"    {k_lD:>12.3f} {omega/omega_pe:>16.6f} {v_phi/v_th:>12.3f} "
              f"{v_g/v_th:>12.6f} {thermal_corr:>14.6f}")

    # (b) At what k does thermal correction become 10%?
    # 3*k^2*v_th^2 / omega_pe^2 = 0.1 -> k*lambda_D = sqrt(0.1/3)
    k_lD_10pct = np.sqrt(0.1 / 3)
    print(f"\n(b) Thermal correction = 10% at k*lambda_D = {k_lD_10pct:.4f}")
    print(f"    Wavelength: lambda = {2*np.pi*lambda_D/k_lD_10pct:.4e} m = {2*np.pi/k_lD_10pct:.1f} lambda_D")

    # (c) Group velocity * phase velocity
    print(f"\n(c) Product v_phi * v_g for k*lambda_D << 1:")
    print(f"    v_phi * v_g ~ 3*v_th^2 = {3*v_th**2:.4e} m^2/s^2")
    print(f"    This is analogous to de Broglie: v_phase * v_group = c^2 for relativistic particles")
    print()


def exercise_2():
    """
    Exercise 2: Landau Damping at Different k*lambda_D
    Calculate and compare damping rates for various wavenumbers.
    """
    print("--- Exercise 2: Landau Damping at Different k*lambda_D ---")

    n = 1e18
    T_eV = 10.0
    T_J = T_eV * eV_to_J
    v_th = np.sqrt(T_J / m_e)
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    lambda_D = v_th / omega_pe

    # Landau damping rate (weak damping approximation, k*lambda_D < 0.5):
    # gamma/omega_pe = -sqrt(pi/8) * (1/(k*lambda_D)^3) * exp(-1/(2*(k*lambda_D)^2) - 3/2)

    print(f"Landau damping rate for Langmuir waves:")
    print(f"gamma = -sqrt(pi/8) * omega_pe / (k*lambda_D)^3 * exp(-1/(2*(k*lambda_D)^2) - 3/2)")
    print()

    k_lD_values = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5])

    print(f"    {'k*lambda_D':>12} {'omega_r/omega_pe':>16} {'gamma/omega_pe':>16} "
          f"{'|gamma|/omega_r':>16} {'e-fold periods':>14}")
    print("    " + "-" * 78)

    for k_lD in k_lD_values:
        k = k_lD / lambda_D
        omega_r = np.sqrt(omega_pe**2 + 3 * k**2 * v_th**2)

        # Landau damping rate
        gamma = -np.sqrt(np.pi / 8) * omega_pe / k_lD**3 * np.exp(-1 / (2 * k_lD**2) - 1.5)

        ratio = abs(gamma) / omega_r
        # Number of oscillation periods for 1 e-folding
        if abs(gamma) > 0:
            n_periods = omega_r / (2 * np.pi * abs(gamma))
        else:
            n_periods = float('inf')

        print(f"    {k_lD:>12.3f} {omega_r/omega_pe:>16.6f} {gamma/omega_pe:>16.4e} "
              f"{ratio:>16.4e} {n_periods:>14.2f}")

    print()
    print("    Observations:")
    print("    - Damping is exponentially sensitive to k*lambda_D")
    print("    - For k*lambda_D < 0.1: effectively undamped (> 1000 periods)")
    print("    - For k*lambda_D ~ 0.3: damped in ~1 period (overdamped)")
    print("    - For k*lambda_D > 0.5: wave is strongly damped, no propagation")
    print()


def exercise_3():
    """
    Exercise 3: Bump-on-Tail Instability
    f0(v) = (1-n_b/n_0)*Maxwellian(v_th) + (n_b/n_0)*Maxwellian(v_th_b, u_b)
    Find the growth rate when df0/dv > 0 at the resonant velocity.
    """
    print("--- Exercise 3: Bump-on-Tail Instability ---")

    n_0 = 1e18
    T_bulk_eV = 10.0
    T_beam_eV = 1.0
    v_th_bulk = np.sqrt(T_bulk_eV * eV_to_J / m_e)
    v_th_beam = np.sqrt(T_beam_eV * eV_to_J / m_e)
    u_beam = 5 * v_th_bulk   # Beam drift velocity
    n_ratio = 0.01           # n_beam / n_total

    omega_pe = np.sqrt(n_0 * e**2 / (epsilon_0 * m_e))

    print(f"Bump-on-tail: bulk T = {T_bulk_eV} eV, beam T = {T_beam_eV} eV")
    print(f"  Beam velocity: u_b = {u_beam/v_th_bulk:.0f} * v_th = {u_beam:.4e} m/s")
    print(f"  Beam fraction: n_b/n_0 = {n_ratio}")

    # (a) Distribution function and its derivative
    def f0(v):
        f_bulk = (1 - n_ratio) * np.exp(-v**2 / (2 * v_th_bulk**2)) / np.sqrt(2 * np.pi * v_th_bulk**2)
        f_beam = n_ratio * np.exp(-(v - u_beam)**2 / (2 * v_th_beam**2)) / np.sqrt(2 * np.pi * v_th_beam**2)
        return n_0 * (f_bulk + f_beam)

    def df0dv(v):
        df_bulk = (1 - n_ratio) * (-v / v_th_bulk**2) * np.exp(-v**2 / (2 * v_th_bulk**2)) / np.sqrt(2 * np.pi * v_th_bulk**2)
        df_beam = n_ratio * (-(v - u_beam) / v_th_beam**2) * np.exp(-(v - u_beam)**2 / (2 * v_th_beam**2)) / np.sqrt(2 * np.pi * v_th_beam**2)
        return n_0 * (df_bulk + df_beam)

    # Find where df0/dv > 0 (between bulk and beam)
    v_test = np.linspace(0, 8 * v_th_bulk, 1000)
    df_values = np.array([df0dv(v) for v in v_test])

    # Region where df0/dv > 0
    positive_slope = v_test[df_values > 0]
    if len(positive_slope) > 0:
        v_min_pos = positive_slope[0]
        v_max_pos = positive_slope[-1]
        print(f"\n(a) df0/dv > 0 in range: v = [{v_min_pos/v_th_bulk:.2f}, {v_max_pos/v_th_bulk:.2f}] * v_th")
    else:
        print(f"\n(a) df0/dv <= 0 everywhere (no instability)")

    # (b) Maximum growth rate
    # Growth rate: gamma = (pi * omega_pe^3) / (2 * k^2) * (df0/dv)|_{v=omega/k}
    # At resonance v = omega_pe/k (for Langmuir waves where omega ~ omega_pe)
    # Maximum growth when df0/dv is maximum positive

    max_df_idx = np.argmax(df_values)
    v_max_growth = v_test[max_df_idx]
    max_df = df_values[max_df_idx]

    k_res = omega_pe / v_max_growth  # Resonant wavenumber
    lambda_D = v_th_bulk / omega_pe

    # gamma = (pi/2) * (omega_pe / (k*lambda_D)^2) * (v_th^2/n_0) * df0/dv|_res
    gamma = (np.pi / 2) * omega_pe / (k_res * lambda_D)**2 * (v_th_bulk**2 / n_0) * max_df

    print(f"\n(b) Maximum growth rate:")
    print(f"    Resonant velocity: v_res = {v_max_growth/v_th_bulk:.2f} * v_th")
    print(f"    Resonant k: k*lambda_D = {k_res*lambda_D:.4f}")
    print(f"    Growth rate: gamma = {gamma:.4e} rad/s")
    print(f"    gamma / omega_pe = {gamma/omega_pe:.4e}")

    # (c) Growth time and e-folding
    if gamma > 0:
        tau_growth = 1.0 / gamma
        n_osc = omega_pe / (2 * np.pi * gamma)
        print(f"\n(c) Growth time: tau = {tau_growth:.4e} s")
        print(f"    Oscillation periods per e-folding: {n_osc:.1f}")
    else:
        print(f"\n(c) No growth (stable)")

    # (d) Saturation mechanism
    print(f"\n(d) Saturation:")
    print(f"    The instability grows until the wave amplitude is large enough to")
    print(f"    trap resonant particles in the wave potential.")
    print(f"    Trapping width: delta_v ~ sqrt(e*phi/(m_e*k)) ~ gamma/k")
    print(f"    When delta_v ~ width of positive slope region, instability saturates.")
    print(f"    This leads to plateau formation in f(v) around the beam velocity.")
    print()


def exercise_4():
    """
    Exercise 4: Ion Acoustic Wave Damping
    Dispersion: omega = k * c_s / sqrt(1 + k^2*lambda_D^2)
    Damping depends on T_e/T_i ratio.
    """
    print("--- Exercise 4: Ion Acoustic Wave Damping ---")

    n = 1e18
    T_e_eV = 10.0
    T_e = T_e_eV * eV_to_J
    m_i = m_p  # Hydrogen

    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    omega_pi = np.sqrt(n * e**2 / (epsilon_0 * m_i))

    # Ion acoustic speed: c_s = sqrt(T_e / m_i)  (for T_e >> T_i)
    # More generally: c_s = sqrt((T_e + gamma_i * T_i) / m_i)

    T_ratios = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

    print(f"Ion acoustic waves: omega = k*c_s / sqrt(1 + k^2*lambda_De^2)")
    print(f"T_e = {T_e_eV} eV, n = {n:.0e} m^-3, hydrogen ions")
    print()

    lambda_De = np.sqrt(epsilon_0 * T_e / (n * e**2))
    k = 0.1 / lambda_De  # k*lambda_De = 0.1

    print(f"(a) Ion acoustic speed and damping vs T_e/T_i (k*lambda_De = 0.1):")
    print(f"\n    {'T_e/T_i':>8} {'c_s [m/s]':>12} {'omega/omega_pi':>16} "
          f"{'gamma/omega':>14} {'Propagates?':>12}")
    print("    " + "-" * 66)

    for T_ratio in T_ratios:
        T_i = T_e / T_ratio
        T_i_eV = T_i / eV_to_J

        # Ion acoustic speed (including ion temperature)
        c_s = np.sqrt((T_e + 3 * T_i) / m_i)

        # Real frequency
        k_lD = k * lambda_De
        omega_r = k * c_s / np.sqrt(1 + k_lD**2)

        # Ion Landau damping (approximate for k*lambda_Di < 1):
        # gamma/omega ~ -sqrt(pi/8) * (T_e/T_i)^(3/2) * exp(-T_e/(2*T_i))
        # (strong damping when T_e ~ T_i, weak when T_e >> T_i)
        v_th_i = np.sqrt(T_i / m_i)
        zeta_i = omega_r / (k * v_th_i * np.sqrt(2))

        # Ion contribution to damping
        gamma_i = -np.sqrt(np.pi / 8) * omega_r * np.sqrt(m_e / m_i) * \
                  (T_e / T_i)**1.5 * np.exp(-T_e / (2 * T_i))

        # Electron contribution (Landau damping from electrons, usually smaller)
        v_th_e = np.sqrt(T_e / m_e)
        zeta_e = omega_r / (k * v_th_e * np.sqrt(2))
        gamma_e = -np.sqrt(np.pi / 8) * omega_r * np.sqrt(m_e / m_i) * np.exp(-zeta_e**2)

        gamma_total = gamma_i  # Ion damping dominates

        ratio = abs(gamma_total) / omega_r if omega_r > 0 else float('inf')
        propagates = "Yes" if ratio < 0.5 else "No (overdamped)"

        print(f"    {T_ratio:>8.1f} {c_s:>12.0f} {omega_r/omega_pi:>16.6f} "
              f"{ratio:>14.4e} {propagates:>12}")

    # (b) Physical explanation
    print(f"\n(b) Why T_e >> T_i is needed:")
    print(f"    Ion acoustic waves have phase velocity v_phi ~ c_s ~ sqrt(T_e/m_i)")
    print(f"    Ion thermal speed: v_th,i = sqrt(T_i/m_i)")
    print(f"    v_phi/v_th,i ~ sqrt(T_e/T_i)")
    print(f"    When T_e ~ T_i: v_phi ~ v_th,i -> many resonant ions -> strong damping")
    print(f"    When T_e >> T_i: v_phi >> v_th,i -> few resonant ions -> weak damping")
    print()


def exercise_5():
    """
    Exercise 5: Particle Trapping and Bounce Frequency
    In a Langmuir wave with amplitude phi_0, particles with
    v ~ omega/k get trapped and oscillate at the bounce frequency.
    """
    print("--- Exercise 5: Particle Trapping and Bounce Frequency ---")

    n = 1e18
    T_eV = 10.0
    T_J = T_eV * eV_to_J
    v_th = np.sqrt(T_J / m_e)
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    lambda_D = v_th / omega_pe

    # Wave parameters
    k_lD = 0.2
    k = k_lD / lambda_D
    omega_r = np.sqrt(omega_pe**2 + 3 * k**2 * v_th**2)
    v_phi = omega_r / k

    print(f"Langmuir wave: k*lambda_D = {k_lD}, omega/omega_pe = {omega_r/omega_pe:.4f}")
    print(f"Phase velocity: v_phi = {v_phi:.4e} m/s = {v_phi/v_th:.2f}*v_th")
    print()

    # (a) Trapping bounce frequency: omega_B = sqrt(e*k*E_0/m_e) = k*sqrt(e*phi_0/m_e)
    # where E_0 = k*phi_0 is the wave electric field amplitude

    phi_0_values = np.array([0.01, 0.1, 1.0, 10.0])  # Potential amplitude in V

    print(f"(a) Bounce frequency vs wave amplitude:")
    print(f"    {'phi_0 [V]':>12} {'E_0 [V/m]':>12} {'omega_B [rad/s]':>16} "
          f"{'omega_B/omega_pe':>16} {'delta_v/v_th':>14}")
    print("    " + "-" * 74)

    for phi_0 in phi_0_values:
        E_0 = k * phi_0
        omega_B = np.sqrt(e * k * E_0 / m_e)
        # Alternatively: omega_B = k * sqrt(e * phi_0 / m_e)
        delta_v = omega_B / k  # Trapping width in velocity space

        print(f"    {phi_0:>12.2f} {E_0:>12.2f} {omega_B:>16.4e} "
              f"{omega_B/omega_pe:>16.4e} {delta_v/v_th:>14.4e}")

    # (b) Trapping condition: particle is trapped if its velocity deviation
    # from v_phi satisfies |v - v_phi| < delta_v = omega_B/k
    print(f"\n(b) Trapping criterion:")
    print(f"    Particle trapped if |v - v_phi| < delta_v = sqrt(2*e*phi_0/(m_e))/k")
    print(f"    Equivalently: kinetic energy in wave frame < potential well depth")

    # (c) Transition from linear to nonlinear regime
    # Linear (Landau) damping valid when omega_B << |gamma|
    # Nonlinear (trapping) regime when omega_B >> |gamma|

    gamma_landau = -np.sqrt(np.pi / 8) * omega_pe / k_lD**3 * np.exp(-1 / (2 * k_lD**2) - 1.5)

    print(f"\n(c) Linear vs nonlinear regime:")
    print(f"    Landau damping rate: gamma = {gamma_landau:.4e} rad/s")
    print(f"    |gamma|/omega_pe = {abs(gamma_landau)/omega_pe:.4e}")

    # Critical amplitude where omega_B = |gamma|
    # omega_B = sqrt(e*k^2*phi_0/m_e) = |gamma|
    # phi_0_crit = m_e * gamma^2 / (e * k^2)
    phi_crit = m_e * gamma_landau**2 / (e * k**2)
    E_crit = k * phi_crit

    print(f"    Critical amplitude (omega_B = |gamma|):")
    print(f"    phi_crit = {phi_crit:.4e} V")
    print(f"    E_crit = {E_crit:.4e} V/m")
    print(f"    For phi >> phi_crit: nonlinear trapping dominates")
    print(f"    For phi << phi_crit: linear Landau damping applies")

    # (d) O'Neil's nonlinear damping
    print(f"\n(d) Nonlinear evolution (O'Neil trapping):")
    print(f"    After t > 1/omega_B, trapped particles bounce in the wave potential")
    print(f"    This causes the damping rate to oscillate:")
    print(f"    gamma(t) ~ gamma_L * cos(omega_B * t) * exp(-some decay)")
    print(f"    The wave amplitude oscillates rather than monotonically damping")
    print(f"    Eventually: BGK (Bernstein-Greene-Kruskal) equilibrium forms")
    print(f"    -> Trapped particle distribution creates a self-consistent wave")

    # (e) Fraction of trapped particles
    print(f"\n(e) Trapped particle fraction for phi_0 = 1 V:")
    phi_0 = 1.0
    delta_v = np.sqrt(2 * e * phi_0 / m_e)
    # Fraction of Maxwellian within delta_v of v_phi
    from scipy.special import erf
    frac = 2 * delta_v / (np.sqrt(2 * np.pi) * v_th) * np.exp(-v_phi**2 / (2 * v_th**2))
    print(f"    Trapping width: delta_v = {delta_v:.4e} m/s = {delta_v/v_th:.4f}*v_th")
    print(f"    Fraction near resonance: ~ {frac:.4e}")
    print(f"    Despite small fraction, these particles control the wave dynamics!")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
