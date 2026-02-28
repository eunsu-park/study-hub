"""
Plasma Physics - Lesson 06: Magnetic Mirrors and Adiabatic Invariants
Exercise Solutions

Topics covered:
- Mirror confinement time estimation
- Conservation of magnetic moment mu
- Bounce frequency in a parabolic mirror
- Tokamak banana orbit width calculation
- Second adiabatic invariant J in a dipole field
"""

import numpy as np
from scipy.integrate import quad, solve_ivp

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
    Exercise 1: Mirror Confinement Time
    Simple mirror: B_min = 0.5 T, B_max = 3 T, L = 2 m.
    Hydrogen plasma, T = 1 keV.
    Estimate confinement time limited by Coulomb scattering into loss cone.
    """
    print("--- Exercise 1: Mirror Confinement Time ---")

    B_min = 0.5     # T
    B_max = 3.0     # T
    L = 2.0         # Mirror half-length [m]
    T_eV = 1e3      # 1 keV
    T_J = T_eV * eV_to_J
    n = 1e19         # Density [m^-3]
    m_i = m_p

    # (a) Mirror ratio and loss cone
    R_m = B_max / B_min
    sin2_alpha_lc = 1.0 / R_m
    alpha_lc = np.arcsin(np.sqrt(sin2_alpha_lc))
    alpha_lc_deg = np.degrees(alpha_lc)

    print(f"(a) Mirror ratio: R_m = B_max/B_min = {R_m:.1f}")
    print(f"    Loss cone angle: alpha_lc = {alpha_lc_deg:.1f} degrees")
    print(f"    Trapped fraction: {np.cos(alpha_lc):.3f}")

    # (b) Coulomb scattering time
    # Ion-ion collision frequency
    ln_Lambda = 17.0
    v_th = np.sqrt(T_J / m_i)
    nu_ii = (n * e**4 * ln_Lambda) / (
        6 * np.pi**2 * epsilon_0**2 * m_i**2 * v_th**3
    )
    tau_ii = 1.0 / nu_ii

    print(f"\n(b) Ion-ion collision time:")
    print(f"    v_th = {v_th:.4e} m/s")
    print(f"    nu_ii = {nu_ii:.4e} s^-1")
    print(f"    tau_ii = {tau_ii:.4e} s = {tau_ii*1e3:.2f} ms")

    # (c) Confinement time: tau_conf ~ tau_ii * ln(R_m)
    # Particles must scatter by angle alpha_lc to reach loss cone
    # Pastukhov formula (approximate): tau ~ tau_ii * R_m * ln(R_m)
    tau_conf_simple = tau_ii * np.log(R_m)
    tau_conf_pastukhov = tau_ii * R_m * np.log(R_m)

    print(f"\n(c) Confinement time estimates:")
    print(f"    Simple:     tau_conf ~ tau_ii * ln(R_m) = {tau_conf_simple:.4e} s")
    print(f"    Pastukhov: tau_conf ~ tau_ii * R_m * ln(R_m) = {tau_conf_pastukhov:.4e} s")

    # (d) Loss rate and power loss
    # Power carried by each escaping ion: ~ T
    # P_loss ~ n * T / tau_conf * Volume
    V = np.pi * 0.1**2 * 2 * L  # Cylindrical plasma, r=10cm
    P_loss = n * T_J / tau_conf_pastukhov * V
    print(f"\n(d) Estimated power loss (Pastukhov):")
    print(f"    Volume ~ {V:.4f} m^3")
    print(f"    P_loss ~ {P_loss:.4e} W = {P_loss/1e3:.2f} kW")
    print(f"    This end-loss is why simple mirrors are not ideal for fusion")
    print()


def exercise_2():
    """
    Exercise 2: Conservation of Magnetic Moment mu
    Verify mu = m*v_perp^2 / (2*B) is conserved along a particle orbit
    in a slowly varying magnetic field.
    """
    print("--- Exercise 2: Conservation of mu ---")

    # Mirror field: B_z(z) = B_0 * (1 + alpha * z^2)
    B_0 = 0.5
    alpha = 1.0  # m^-2
    L = 1.0

    def B_field(z):
        """Axial field magnitude."""
        return B_0 * (1 + alpha * z**2)

    def dBdz(z):
        """Axial field gradient."""
        return B_0 * 2 * alpha * z

    # Particle: proton, 1 keV, pitch angle 60 deg at midplane
    E_total = 1e3 * eV_to_J
    pitch_angle_deg = 60.0
    pitch_angle = np.radians(pitch_angle_deg)

    v_total = np.sqrt(2 * E_total / m_p)
    v_perp_0 = v_total * np.sin(pitch_angle)
    v_par_0 = v_total * np.cos(pitch_angle)

    mu = m_p * v_perp_0**2 / (2 * B_field(0))

    print(f"Initial conditions at z=0:")
    print(f"  B(0) = {B_field(0)} T")
    print(f"  v_total = {v_total:.4e} m/s")
    print(f"  v_perp = {v_perp_0:.4e} m/s, v_par = {v_par_0:.4e} m/s")
    print(f"  mu = m*v_perp^2/(2B) = {mu:.4e} J/T")
    print(f"  Pitch angle = {pitch_angle_deg} degrees")

    # (a) Mirror point: v_par = 0 -> all energy in v_perp
    # mu = m*v_total^2 / (2*B_mirror) -> B_mirror = m*v_total^2 / (2*mu)
    B_mirror = m_p * v_total**2 / (2 * mu)
    z_mirror = np.sqrt((B_mirror / B_0 - 1) / alpha)

    print(f"\n(a) Mirror point:")
    print(f"    B_mirror = {B_mirror:.3f} T")
    print(f"    z_mirror = {z_mirror:.4f} m")
    print(f"    Mirror ratio experienced: {B_mirror/B_0:.3f}")

    # (b) Track mu along orbit using guiding center equations
    # dz/dt = v_par, d(v_par)/dt = -(mu/m) * dB/dz
    def gc_equations(t, state):
        z, v_par = state
        dvpar_dt = -(mu / m_p) * dBdz(z)
        return [v_par, dvpar_dt]

    # Integrate for several bounce periods
    t_bounce_est = 4 * z_mirror / (v_par_0 * 0.7)  # Rough estimate
    t_final = 5 * t_bounce_est
    t_eval = np.linspace(0, t_final, 5000)

    sol = solve_ivp(gc_equations, (0, t_final), [0, v_par_0],
                    t_eval=t_eval, method='RK45', rtol=1e-12)

    # Calculate mu at each point
    z_traj = sol.y[0]
    vpar_traj = sol.y[1]
    B_traj = B_field(z_traj)
    # v_perp^2 = v_total^2 - v_par^2 (energy conservation)
    vperp2_traj = v_total**2 - vpar_traj**2
    mu_traj = m_p * vperp2_traj / (2 * B_traj)

    mu_max = np.max(mu_traj)
    mu_min = np.min(mu_traj)
    mu_avg = np.mean(mu_traj)

    print(f"\n(b) mu conservation over {5} bounce periods:")
    print(f"    mu_initial = {mu:.6e} J/T")
    print(f"    mu_mean    = {mu_avg:.6e} J/T")
    print(f"    max deviation: {abs(mu_max - mu)/mu:.2e}")
    print(f"    min deviation: {abs(mu_min - mu)/mu:.2e}")
    print(f"    mu is conserved to machine precision in guiding center theory")

    # (c) When does mu conservation break down?
    omega_ci = e * B_0 / m_p
    rho_i = v_perp_0 / omega_ci
    print(f"\n(c) Adiabaticity condition:")
    print(f"    rho_i = {rho_i*1e3:.3f} mm")
    print(f"    Scale length L_B = B/|grad B| at z_mirror = {B_mirror/dBdz(z_mirror):.3f} m")
    print(f"    rho_i / L_B = {rho_i / (B_mirror/dBdz(z_mirror)):.4e}")
    print(f"    mu conserved when rho/L_B << 1")
    print()


def exercise_3():
    """
    Exercise 3: Bounce Frequency in a Parabolic Mirror
    B(z) = B_0 * (1 + (z/L)^2), calculate bounce frequency
    as a function of pitch angle.
    """
    print("--- Exercise 3: Bounce Frequency in Parabolic Mirror ---")

    B_0 = 1.0
    L = 1.0
    E_keV = 1.0
    E_J = E_keV * 1e3 * eV_to_J
    v_total = np.sqrt(2 * E_J / m_p)

    # B(z) = B_0 * (1 + z^2/L^2)
    # Mirror point: B(z_m) = B_0/sin^2(alpha)
    # z_m = L * sqrt(1/sin^2(alpha) - 1) = L * cos(alpha)/sin(alpha) = L/tan(alpha)

    print(f"Parabolic mirror: B(z) = B_0*(1 + z^2/L^2)")
    print(f"B_0 = {B_0} T, L = {L} m, E = {E_keV} keV (proton)")
    print(f"v_total = {v_total:.4e} m/s")

    # (a) Bounce period: T_b = integral from -z_m to z_m of dz/v_par(z)
    # v_par(z) = v_total * sqrt(1 - sin^2(alpha) * B(z)/B_0)
    #          = v_total * sqrt(1 - sin^2(alpha) * (1 + z^2/L^2))
    #          = v_total * sqrt(cos^2(alpha) - sin^2(alpha)*z^2/L^2)

    pitch_angles_deg = [30, 45, 60, 70, 80, 85]

    print(f"\n(a) Bounce period vs pitch angle:")
    print(f"    {'alpha [deg]':>12} {'z_mirror [m]':>14} {'T_bounce [us]':>14} {'f_bounce [kHz]':>14}")
    print("    " + "-" * 58)

    for alpha_deg in pitch_angles_deg:
        alpha = np.radians(alpha_deg)
        sin_a = np.sin(alpha)
        cos_a = np.cos(alpha)

        # Mirror point
        z_m = L * cos_a / sin_a

        # Check if particle is confined (z_m must be within physical mirror)
        # For this exercise, the mirror extends indefinitely

        # Bounce period integral: T_b = 4 * int_0^z_m dz / (v_total * sqrt(cos^2(a) - sin^2(a)*z^2/L^2))
        # Substitution: u = z*sin(a)/(L*cos(a))
        # T_b = 4*L / (v_total * sin(a)) * int_0^1 du / sqrt(1 - u^2) = 4*L*pi/(2*v_total*sin(a))

        def integrand(z):
            arg = cos_a**2 - sin_a**2 * z**2 / L**2
            if arg <= 0:
                return 0
            return 1.0 / (v_total * np.sqrt(arg))

        T_bounce_half, _ = quad(integrand, 0, z_m * 0.9999)  # Avoid singularity at mirror point
        T_bounce = 4 * T_bounce_half

        # Analytical result for parabolic mirror
        T_bounce_analytical = 2 * np.pi * L / (v_total * sin_a)

        f_bounce = 1.0 / T_bounce if T_bounce > 0 else 0

        print(f"    {alpha_deg:>12.0f} {z_m:>14.4f} {T_bounce*1e6:>14.3f} {f_bounce/1e3:>14.3f}")

    # (b) Analytical formula
    print(f"\n(b) Analytical bounce period for parabolic mirror:")
    print(f"    T_b = 2*pi*L / (v * sin(alpha))")
    print(f"    For alpha = 90 deg (deeply trapped): T_b = 2*pi*L/v = {2*np.pi*L/v_total*1e6:.3f} us")
    print(f"    For alpha -> 0 (barely trapped): T_b -> infinity (approaches loss cone)")

    # (c) Bounce-averaged drift
    print(f"\n(c) Bounce-averaged grad-B drift:")
    print(f"    <v_gradB> = (1/T_b) * integral v_gradB(z) * dz/v_par")
    print(f"    Trapped particles average their drift over the bounce orbit")
    print(f"    Deeply trapped: sample mostly midplane (weak gradient)")
    print(f"    Barely trapped: sample near mirror points (strong gradient)")
    print()


def exercise_4():
    """
    Exercise 4: Tokamak Banana Orbit Width
    ITER-like: R = 6.2 m, a = 2 m, B_0 = 5.3 T, q = 1.5 (at r = a/2).
    Calculate banana width for thermal deuterium at 10 keV.
    """
    print("--- Exercise 4: Tokamak Banana Orbit Width ---")

    R = 6.2       # Major radius [m]
    a = 2.0       # Minor radius [m]
    B_0 = 5.3     # Toroidal field on axis [T]
    q = 1.5       # Safety factor at r = a/2
    T_keV = 10.0
    T_J = T_keV * 1e3 * eV_to_J
    m_i = 2 * m_p  # Deuterium

    r = a / 2     # Evaluation point
    epsilon = r / R  # Inverse aspect ratio

    # (a) Thermal velocity and Larmor radius
    v_th = np.sqrt(T_J / m_i)
    omega_ci = e * B_0 / m_i
    rho_i = v_th / omega_ci

    print(f"(a) Thermal deuterium at {T_keV} keV:")
    print(f"    v_th = {v_th:.4e} m/s")
    print(f"    omega_ci = {omega_ci:.4e} rad/s")
    print(f"    Larmor radius: rho_i = {rho_i*1e3:.2f} mm")

    # (b) Banana orbit width: w_b = q * rho_i / sqrt(epsilon)
    # This comes from the conservation of canonical angular momentum
    # p_phi = m*R*v_phi - e*psi = const
    # The radial excursion delta_r ~ q*rho_i/sqrt(epsilon) for barely trapped particles

    w_b = q * rho_i / np.sqrt(epsilon)

    print(f"\n(b) Banana orbit width:")
    print(f"    epsilon = r/R = {epsilon:.3f}")
    print(f"    w_b = q * rho_i / sqrt(epsilon) = {w_b*1e2:.2f} cm = {w_b*1e3:.1f} mm")
    print(f"    w_b / rho_i = q / sqrt(epsilon) = {w_b/rho_i:.1f}")
    print(f"    w_b / a = {w_b/a:.4f}")

    # (c) Trapped particle fraction
    f_trapped = np.sqrt(2 * epsilon)  # Approximate
    print(f"\n(c) Trapped particle fraction:")
    print(f"    f_trapped ~ sqrt(2*epsilon) = {f_trapped:.3f} = {f_trapped*100:.1f}%")

    # (d) Bounce frequency
    # omega_b = v_th * sqrt(epsilon) / (q * R)
    omega_b = v_th * np.sqrt(epsilon) / (q * R)
    f_b = omega_b / (2 * np.pi)
    T_b = 1.0 / f_b

    print(f"\n(d) Bounce frequency of trapped particles:")
    print(f"    omega_b = v_th * sqrt(epsilon) / (q*R) = {omega_b:.4e} rad/s")
    print(f"    f_b = {f_b:.1f} Hz, T_b = {T_b*1e3:.2f} ms")

    # (e) Compare to transit frequency of passing particles
    omega_t = v_th / (q * R)
    print(f"\n(e) Transit frequency (passing particles):")
    print(f"    omega_t = v_th / (q*R) = {omega_t:.4e} rad/s")
    print(f"    omega_b / omega_t = sqrt(epsilon) = {np.sqrt(epsilon):.3f}")

    # (f) Neoclassical enhancement: D_neo / D_classical ~ q^2 / epsilon^(3/2) (banana regime)
    neo_enhancement = q**2 / epsilon**1.5
    print(f"\n(f) Neoclassical transport enhancement (banana regime):")
    print(f"    D_neo / D_classical ~ q^2 / epsilon^(3/2) = {neo_enhancement:.1f}")
    print(f"    Banana orbits significantly enhance cross-field transport")
    print()


def exercise_5():
    """
    Exercise 5: Second Adiabatic Invariant in a Dipole Field
    Earth's dipole field: calculate J = integral v_par dl for
    trapped radiation belt particles.
    """
    print("--- Exercise 5: Second Adiabatic Invariant J in Dipole ---")

    # Earth's dipole field:
    # B(r, lambda) = (B_E / r^3) * sqrt(1 + 3*sin^2(lambda)) / cos^6(lambda)
    # where r is in Earth radii, lambda is magnetic latitude
    # Field line equation: r = L * cos^2(lambda)

    B_E = 3.12e-5  # Earth's equatorial surface field [T]
    R_E = 6.371e6   # Earth radius [m]

    L_shell = 4.0   # L = 4 (geosynchronous-like)
    E_keV = 100.0   # 100 keV proton
    E_J = E_keV * 1e3 * eV_to_J
    alpha_eq_deg = 45.0  # Equatorial pitch angle

    v_total = np.sqrt(2 * E_J / m_p)
    alpha_eq = np.radians(alpha_eq_deg)

    # (a) B at equator for L shell
    B_eq = B_E / L_shell**3
    print(f"(a) L = {L_shell} shell parameters:")
    print(f"    B_equator = B_E/L^3 = {B_eq*1e9:.1f} nT")
    print(f"    r_equator = L * R_E = {L_shell * R_E/1e6:.1f} Mm = {L_shell} R_E")

    # (b) Mirror point latitude
    # At mirror: sin^2(alpha_eq) = B_eq / B_mirror
    # B at latitude lambda on L shell:
    # B(lambda) = (B_E / L^3) * sqrt(1 + 3*sin^2(lambda)) / cos^6(lambda)
    # So B_mirror/B_eq = sqrt(1 + 3*sin^2(lambda_m)) / cos^6(lambda_m)

    sin2_alpha = np.sin(alpha_eq)**2

    def mirror_eq(lam):
        """B(lambda)/B_eq - 1/sin^2(alpha_eq) = 0"""
        return np.sqrt(1 + 3 * np.sin(lam)**2) / np.cos(lam)**6 - 1.0 / sin2_alpha

    # Solve by bisection
    from scipy.optimize import brentq
    lambda_m = brentq(mirror_eq, 0, np.pi / 2 - 0.01)
    lambda_m_deg = np.degrees(lambda_m)

    print(f"\n(b) Mirror point for alpha_eq = {alpha_eq_deg} degrees:")
    print(f"    Mirror latitude: lambda_m = {lambda_m_deg:.2f} degrees")
    print(f"    r at mirror: {L_shell * np.cos(lambda_m)**2:.2f} R_E")

    # (c) Second adiabatic invariant J = integral v_par * ds
    # Along field line, ds = r * sqrt(1 + 3*sin^2(lambda)) * d(lambda) / cos(lambda)
    # Wait, the arc length element along a dipole field line is:
    # ds = L*R_E * cos(lambda) * sqrt(1 + 3*sin^2(lambda)) * d(lambda)
    # v_par = v * sqrt(1 - sin^2(alpha_eq) * B(lambda)/B_eq)

    mu = m_p * (v_total * np.sin(alpha_eq))**2 / (2 * B_eq)

    def J_integrand(lam):
        B_ratio = np.sqrt(1 + 3 * np.sin(lam)**2) / np.cos(lam)**6
        v_par = v_total * np.sqrt(max(0, 1 - sin2_alpha * B_ratio))
        ds_dlam = L_shell * R_E * np.cos(lam) * np.sqrt(1 + 3 * np.sin(lam)**2)
        return v_par * ds_dlam

    J_half, _ = quad(J_integrand, 0, lambda_m * 0.999)
    J = 2 * m_p * J_half  # Factor 2 for both hemispheres, multiply by m for proper units

    print(f"\n(c) Second adiabatic invariant:")
    print(f"    J = 2 * m * integral v_par ds = {J:.4e} kg*m^2/s")
    print(f"    (integrated from -lambda_m to +lambda_m)")

    # (d) Bounce period
    def bounce_integrand(lam):
        B_ratio = np.sqrt(1 + 3 * np.sin(lam)**2) / np.cos(lam)**6
        v_par = v_total * np.sqrt(max(1e-30, 1 - sin2_alpha * B_ratio))
        ds_dlam = L_shell * R_E * np.cos(lam) * np.sqrt(1 + 3 * np.sin(lam)**2)
        return ds_dlam / v_par

    T_bounce_half, _ = quad(bounce_integrand, 0, lambda_m * 0.999)
    T_bounce = 4 * T_bounce_half  # Factor 4: both hemispheres, there and back
    f_bounce = 1.0 / T_bounce

    print(f"\n(d) Bounce period:")
    print(f"    T_bounce = {T_bounce:.3f} s")
    print(f"    f_bounce = {f_bounce:.3f} Hz")

    # (e) Third invariant (drift invariant)
    # Phi = integral B . dA over the drift shell
    # Conservation of Phi means particles stay on their L shell
    # (valid when drift period << external field change timescale)
    omega_ci = e * B_eq / m_p
    rho_i = v_total / omega_ci

    # Drift period (rough estimate):
    # T_drift ~ 2*pi*R / v_drift
    # v_drift ~ m*v^2 / (2*e*B*R) (combined grad-B + curvature)
    v_drift = E_J / (e * B_eq * L_shell * R_E)
    T_drift = 2 * np.pi * L_shell * R_E / v_drift

    print(f"\n(e) Drift period (third invariant):")
    print(f"    v_drift ~ {v_drift:.1f} m/s")
    print(f"    T_drift ~ {T_drift:.0f} s = {T_drift/60:.1f} min")
    print(f"    Hierarchy: T_gyro << T_bounce << T_drift")
    print(f"    T_gyro = {2*np.pi/omega_ci:.4e} s")
    print(f"    Ratios: T_b/T_gyro = {T_bounce/(2*np.pi/omega_ci):.0f}, T_drift/T_b = {T_drift/T_bounce:.0f}")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
