"""
Plasma Physics - Lesson 04: Single Particle Motion I
Exercise Solutions

Topics covered:
- Cyclotron frequency and Larmor radius calculation
- E x B drift velocity
- Magnetron configuration analysis
- Boris algorithm implementation and verification
- Combined electric and magnetic field trajectories
"""

import numpy as np
from scipy.integrate import solve_ivp

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
m_p = 1.673e-27
epsilon_0 = 8.854e-12
k_B = 1.381e-23
c = 3e8
eV_to_J = e


def exercise_1():
    """
    Exercise 1: Cyclotron Frequency and Larmor Radius
    Calculate for both electrons and protons in a 1 T magnetic field
    at various energies.
    """
    print("--- Exercise 1: Cyclotron Frequency and Larmor Radius ---")

    B = 1.0  # Magnetic field [T]

    # (a) Cyclotron frequencies
    omega_ce = e * B / m_e
    omega_ci = e * B / m_p
    f_ce = omega_ce / (2 * np.pi)
    f_ci = omega_ci / (2 * np.pi)

    print(f"(a) Magnetic field B = {B} T")
    print(f"    Electron cyclotron frequency: omega_ce = {omega_ce:.4e} rad/s")
    print(f"    f_ce = {f_ce/1e9:.2f} GHz")
    print(f"    Proton cyclotron frequency:   omega_ci = {omega_ci:.4e} rad/s")
    print(f"    f_ci = {f_ci/1e6:.2f} MHz")
    print(f"    Ratio omega_ce/omega_ci = m_p/m_e = {m_p/m_e:.1f}")

    # (b) Larmor radii at different energies
    energies_eV = [1, 10, 100, 1000, 10000]

    print(f"\n(b) Larmor radii (perpendicular energy = E_perp):")
    print(f"    {'E_perp [eV]':>12} {'rho_e [m]':>14} {'rho_p [m]':>14} {'rho_p/rho_e':>12}")
    print("    " + "-" * 55)

    for E_eV in energies_eV:
        E_J = E_eV * eV_to_J
        # v_perp = sqrt(2*E/m)
        v_perp_e = np.sqrt(2 * E_J / m_e)
        v_perp_p = np.sqrt(2 * E_J / m_p)
        rho_e = v_perp_e / omega_ce
        rho_p = v_perp_p / omega_ci

        print(f"    {E_eV:>12d} {rho_e:>14.4e} {rho_p:>14.4e} {rho_p/rho_e:>12.1f}")

    print(f"\n    rho_p/rho_e = sqrt(m_p/m_e) = {np.sqrt(m_p/m_e):.1f} (at same energy)")

    # (c) Relativistic correction for high-energy electrons
    print(f"\n(c) Relativistic correction at 1 MeV:")
    E_MeV = 1.0
    E_J = E_MeV * 1e6 * eV_to_J
    gamma = 1 + E_J / (m_e * c**2)
    v_perp = c * np.sqrt(1 - 1 / gamma**2)
    rho_rel = gamma * m_e * v_perp / (e * B)
    rho_nonrel = np.sqrt(2 * E_J / m_e) / omega_ce
    print(f"    gamma = {gamma:.2f}")
    print(f"    Non-relativistic rho = {rho_nonrel:.4e} m")
    print(f"    Relativistic rho = {rho_rel:.4e} m")
    print(f"    Ratio = {rho_rel/rho_nonrel:.2f}")
    print()


def exercise_2():
    """
    Exercise 2: E x B Drift
    Uniform B = 0.5 T z-hat, E = 1000 V/m x-hat.
    Calculate E x B drift and verify with particle orbit.
    """
    print("--- Exercise 2: E x B Drift ---")

    B_mag = 0.5  # T
    E_x = 1000.0  # V/m
    B = np.array([0, 0, B_mag])
    E = np.array([E_x, 0, 0])

    # (a) E x B drift velocity
    v_ExB = np.cross(E, B) / np.dot(B, B)
    print(f"(a) E = {E_x} V/m x-hat, B = {B_mag} T z-hat")
    print(f"    v_ExB = E x B / B^2 = {v_ExB} m/s")
    print(f"    |v_ExB| = {np.linalg.norm(v_ExB):.1f} m/s")
    print(f"    Direction: {'-y' if v_ExB[1] < 0 else '+y'}-hat")
    print(f"    Note: E x B drift is independent of charge, mass, and energy!")

    # (b) Verify with orbit integration for electron
    omega_ce = e * B_mag / m_e
    rho_e = np.sqrt(2 * 10 * eV_to_J / m_e) / omega_ce  # 10 eV electron

    def lorentz_force(t, state, q, m, E_field, B_field):
        x, y, z, vx, vy, vz = state
        v = np.array([vx, vy, vz])
        F = q * (E_field + np.cross(v, B_field))
        return [vx, vy, vz, F[0] / m, F[1] / m, F[2] / m]

    # Electron starting at rest
    v_perp = np.sqrt(2 * 10 * eV_to_J / m_e)  # 10 eV
    y0 = [0, 0, 0, v_perp, 0, 0]
    T_gyro = 2 * np.pi / omega_ce
    t_span = (0, 10 * T_gyro)
    t_eval = np.linspace(0, 10 * T_gyro, 2000)

    sol = solve_ivp(lorentz_force, t_span, y0, t_eval=t_eval,
                    args=(-e, m_e, E, B), method='RK45', rtol=1e-10)

    # Measure average y-velocity (should be v_ExB)
    avg_vy = np.mean(sol.y[4])

    print(f"\n(b) Orbit verification (10 eV electron, 10 gyro-periods):")
    print(f"    Average v_y from orbit = {avg_vy:.1f} m/s")
    print(f"    Theoretical v_ExB_y = {v_ExB[1]:.1f} m/s")
    print(f"    Agreement: {abs(avg_vy - v_ExB[1]) / abs(v_ExB[1]) * 100:.4f}%")

    # (c) Same for proton
    omega_ci = e * B_mag / m_p
    v_perp_p = np.sqrt(2 * 10 * eV_to_J / m_p)
    y0_p = [0, 0, 0, v_perp_p, 0, 0]
    T_gyro_p = 2 * np.pi / omega_ci
    t_span_p = (0, 10 * T_gyro_p)
    t_eval_p = np.linspace(0, 10 * T_gyro_p, 2000)

    sol_p = solve_ivp(lorentz_force, t_span_p, y0_p, t_eval=t_eval_p,
                      args=(e, m_p, E, B), method='RK45', rtol=1e-10)

    avg_vy_p = np.mean(sol_p.y[4])
    print(f"\n(c) Proton orbit verification:")
    print(f"    Average v_y from orbit = {avg_vy_p:.1f} m/s")
    print(f"    Same drift for both species: v_ExB is charge/mass independent")
    print()


def exercise_3():
    """
    Exercise 3: Magnetron Configuration
    Crossed E and B fields: B = 0.1 T z-hat, radial E field.
    Analyze electron motion in a cylindrical magnetron.
    """
    print("--- Exercise 3: Magnetron Configuration ---")

    B_mag = 0.1  # T
    V_anode = 1000  # V
    r_cathode = 0.005  # 5 mm cathode radius
    r_anode = 0.02    # 20 mm anode radius

    # (a) Radial electric field (cylindrical, ln profile)
    # E_r(r) = V_anode / (r * ln(r_anode/r_cathode))
    r_mid = (r_cathode + r_anode) / 2
    E_r_mid = V_anode / (r_mid * np.log(r_anode / r_cathode))

    print(f"(a) Magnetron geometry:")
    print(f"    Cathode radius: {r_cathode*1e3} mm")
    print(f"    Anode radius:   {r_anode*1e3} mm")
    print(f"    Anode voltage:  {V_anode} V")
    print(f"    B field: {B_mag} T (axial)")
    print(f"    E_r at midpoint (r={r_mid*1e3:.1f} mm) = {E_r_mid:.1f} V/m")

    # (b) E x B drift (azimuthal)
    v_ExB_mid = E_r_mid / B_mag
    print(f"\n(b) E x B drift at midpoint:")
    print(f"    v_ExB = E_r/B = {v_ExB_mid:.1f} m/s")
    print(f"    Direction: azimuthal (theta-hat)")

    # (c) Electron cyclotron parameters
    omega_ce = e * B_mag / m_e
    f_ce = omega_ce / (2 * np.pi)
    E_thermal = 5.0  # eV, typical secondary electron energy
    v_perp = np.sqrt(2 * E_thermal * eV_to_J / m_e)
    rho_e = v_perp / omega_ce

    print(f"\n(c) Electron parameters at 5 eV:")
    print(f"    Cyclotron frequency: f_ce = {f_ce/1e9:.2f} GHz")
    print(f"    Larmor radius: rho_e = {rho_e*1e3:.3f} mm")
    print(f"    Gap width: {(r_anode-r_cathode)*1e3:.1f} mm")
    print(f"    rho_e / gap = {rho_e/(r_anode-r_cathode):.3f}")
    if rho_e < (r_anode - r_cathode):
        print(f"    Electrons are confined (rho_e < gap) -> magnetron works!")
    else:
        print(f"    Electrons can reach anode -> magnetron fails")

    # (d) Hull cutoff condition: electrons reach anode when
    # V_anode > (e * B^2 * r_anode^2) / (8 * m_e) * (1 - (r_cathode/r_anode)^2)^2
    V_hull = (e * B_mag**2 * r_anode**2) / (8 * m_e) * (1 - (r_cathode / r_anode)**2)**2
    print(f"\n(d) Hull cutoff voltage:")
    print(f"    V_hull = {V_hull:.1f} V")
    print(f"    Applied voltage = {V_anode} V")
    if V_anode < V_hull:
        print(f"    V_anode < V_hull: electrons confined -> magnetron operates")
    else:
        print(f"    V_anode > V_hull: electrons reach anode -> need stronger B")
    print()


def exercise_4():
    """
    Exercise 4: Boris Algorithm Implementation and Analysis
    Implement the Boris pusher and verify energy conservation
    and orbit accuracy for cyclotron motion.
    """
    print("--- Exercise 4: Boris Algorithm ---")

    def boris_push(x, v, q, m, E, B, dt):
        """
        Single step of the Boris algorithm.
        Returns updated position and velocity.
        """
        # Half-step electric field acceleration
        v_minus = v + (q * dt / (2 * m)) * E

        # Magnetic rotation
        t_vec = (q * dt / (2 * m)) * B
        s_vec = 2 * t_vec / (1 + np.dot(t_vec, t_vec))

        v_prime = v_minus + np.cross(v_minus, t_vec)
        v_plus = v_minus + np.cross(v_prime, s_vec)

        # Second half-step electric field acceleration
        v_new = v_plus + (q * dt / (2 * m)) * E

        # Position update
        x_new = x + v_new * dt

        return x_new, v_new

    # Test case: electron gyration in uniform B
    B_field = np.array([0, 0, 1.0])  # 1 T
    E_field = np.array([0, 0, 0])     # No electric field

    q = -e
    m = m_e
    omega_c = e * 1.0 / m_e

    # Initial conditions: 100 eV perpendicular energy
    E_perp = 100.0 * eV_to_J
    v_perp = np.sqrt(2 * E_perp / m_e)
    rho_exact = v_perp / omega_c
    T_c = 2 * np.pi / omega_c

    x0 = np.array([rho_exact, 0.0, 0.0])
    v0 = np.array([0.0, v_perp, 0.0])  # Tangential velocity

    # (a) Run with different time steps and check energy conservation
    print(f"(a) Boris algorithm: electron in B = 1 T, E_perp = 100 eV")
    print(f"    Exact: omega_c = {omega_c:.4e} rad/s, rho = {rho_exact*1e3:.4f} mm")
    print(f"    Period T_c = {T_c:.4e} s")
    print()

    steps_per_period = [5, 10, 20, 50, 100, 200]
    N_periods = 100

    print(f"    {'Steps/period':>14} {'dt [s]':>14} {'dE/E_0':>14} {'Position err [m]':>16}")
    print("    " + "-" * 62)

    for Nstep in steps_per_period:
        dt = T_c / Nstep
        N_total = Nstep * N_periods

        x = x0.copy()
        v = v0.copy()

        E_kin_0 = 0.5 * m * np.dot(v, v)

        for _ in range(N_total):
            x, v = boris_push(x, v, q, m, E_field, B_field, dt)

        E_kin_final = 0.5 * m * np.dot(v, v)
        dE_rel = abs(E_kin_final - E_kin_0) / E_kin_0

        # Position should return to x0 after N_periods
        pos_err = np.linalg.norm(x - x0)

        print(f"    {Nstep:>14d} {dt:>14.4e} {dE_rel:>14.4e} {pos_err:>16.4e}")

    print()
    print("    Key properties of Boris algorithm:")
    print("    - Energy is conserved to machine precision (symplectic-like)")
    print("    - Position accuracy is 2nd order: error ~ (omega_c * dt)^2")
    print("    - Phase error grows linearly with time, but energy does not drift")
    print()


def exercise_5():
    """
    Exercise 5: Combined Electric and Magnetic Fields
    B = 0.5 T z-hat, E = 500 V/m x-hat + 200 V/m y-hat.
    Analyze resulting particle motion.
    """
    print("--- Exercise 5: Combined E and B Fields ---")

    B = np.array([0, 0, 0.5])
    E = np.array([500, 200, 0])

    # (a) E x B drift
    v_ExB = np.cross(E, B) / np.dot(B, B)
    print(f"(a) E = ({E[0]}, {E[1]}, {E[2]}) V/m")
    print(f"    B = ({B[0]}, {B[1]}, {B[2]}) T")
    print(f"    v_ExB = E x B / B^2 = ({v_ExB[0]:.1f}, {v_ExB[1]:.1f}, {v_ExB[2]:.1f}) m/s")
    print(f"    |v_ExB| = {np.linalg.norm(v_ExB):.1f} m/s")

    # (b) E_parallel (along B) causes acceleration
    B_hat = B / np.linalg.norm(B)
    E_par = np.dot(E, B_hat) * B_hat
    E_perp = E - E_par
    print(f"\n(b) E decomposition:")
    print(f"    E_parallel = ({E_par[0]:.1f}, {E_par[1]:.1f}, {E_par[2]:.1f}) V/m (accelerates along B)")
    print(f"    E_perp     = ({E_perp[0]:.1f}, {E_perp[1]:.1f}, {E_perp[2]:.1f}) V/m (causes ExB drift)")
    print(f"    |E_par| = {np.linalg.norm(E_par):.1f} V/m")
    print(f"    |E_perp| = {np.linalg.norm(E_perp):.1f} V/m")

    # (c) Orbit integration for electron and proton
    def lorentz_eq(t, state, q, m, E_f, B_f):
        x, y, z, vx, vy, vz = state
        v = np.array([vx, vy, vz])
        F = q * (E_f + np.cross(v, B_f))
        return [vx, vy, vz, F[0] / m, F[1] / m, F[2] / m]

    # Electron
    omega_ce = e * np.linalg.norm(B) / m_e
    T_ce = 2 * np.pi / omega_ce
    v_perp_e = np.sqrt(2 * 10 * eV_to_J / m_e)  # 10 eV thermal

    y0_e = [0, 0, 0, v_perp_e, 0, 0]
    N_gyro = 20
    sol_e = solve_ivp(lorentz_eq, (0, N_gyro * T_ce), y0_e,
                      args=(-e, m_e, E, B), method='RK45',
                      t_eval=np.linspace(0, N_gyro * T_ce, 5000), rtol=1e-10)

    # Measure average drift
    avg_vx_e = np.mean(sol_e.y[3])
    avg_vy_e = np.mean(sol_e.y[4])

    print(f"\n(c) Electron orbit (10 eV, {N_gyro} gyro-periods):")
    print(f"    Average drift velocity: ({avg_vx_e:.1f}, {avg_vy_e:.1f}) m/s")
    print(f"    Theoretical v_ExB:      ({v_ExB[0]:.1f}, {v_ExB[1]:.1f}) m/s")

    # Proton
    omega_ci = e * np.linalg.norm(B) / m_p
    T_ci = 2 * np.pi / omega_ci
    v_perp_p = np.sqrt(2 * 10 * eV_to_J / m_p)

    y0_p = [0, 0, 0, v_perp_p, 0, 0]
    sol_p = solve_ivp(lorentz_eq, (0, N_gyro * T_ci), y0_p,
                      args=(e, m_p, E, B), method='RK45',
                      t_eval=np.linspace(0, N_gyro * T_ci, 5000), rtol=1e-10)

    avg_vx_p = np.mean(sol_p.y[3])
    avg_vy_p = np.mean(sol_p.y[4])

    print(f"\n    Proton orbit (10 eV, {N_gyro} gyro-periods):")
    print(f"    Average drift velocity: ({avg_vx_p:.1f}, {avg_vy_p:.1f}) m/s")
    print(f"    Both species drift at the same v_ExB (charge/mass independent)")

    # (d) Polarization drift (for time-varying E)
    print(f"\n(d) Note on polarization drift:")
    print(f"    If E varies in time: v_pol = (m/(q*B^2)) * dE/dt")
    print(f"    Polarization drift IS charge/mass dependent -> creates current")
    print(f"    Important for low-frequency waves (omega << omega_ci)")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
