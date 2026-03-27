"""
Plasma Physics - Lesson 14: From Kinetic to MHD
Exercise Solutions

Topics covered:
- Ideal MHD ordering from generalized Ohm's law
- CGL double adiabatic pressure evolution
- Frozen-in flux theorem and flux conservation
- Gyrokinetic ordering in tokamak
- Hall MHD vs resistive MHD reconnection rates
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


def spitzer_resistivity(T_e_eV, Z=1, ln_Lambda=17.0):
    """Spitzer perpendicular resistivity [Ohm*m]."""
    return 1.03e-4 * Z * ln_Lambda / T_e_eV**1.5


def exercise_1():
    """
    Exercise 1: Ideal MHD from Generalized Ohm's Law
    Tokamak: n = 10^20, T_e = 10 keV, B = 5 T, L = 1 m, V = 100 m/s.
    Estimate each term and identify which can be neglected.
    """
    print("--- Exercise 1: Ideal MHD from Generalized Ohm's Law ---")

    n = 1e20
    T_e_eV = 10e3
    T_e = T_e_eV * eV_to_J
    B = 5.0
    L = 1.0
    V = 100.0
    ln_Lambda = 17.0

    # (a) Characteristic time scale
    tau = L / V
    print(f"(a) Parameters: n = {n:.0e}, T_e = {T_e_eV/1e3:.0f} keV, B = {B} T")
    print(f"    L = {L} m, V = {V} m/s")
    print(f"    Characteristic time: tau = L/V = {tau:.1f} s")

    # Derived quantities
    v_th_e = np.sqrt(T_e / m_e)
    omega_ce = e * B / m_e
    omega_ci = e * B / m_p
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    d_i = c / np.sqrt(n * e**2 / (epsilon_0 * m_p))
    d_e = c / omega_pe
    rho_i = np.sqrt(T_e / m_p) / omega_ci  # Using T_e for estimate

    # Current estimate: J ~ B / (mu_0 * L)
    J = B / (mu_0 * L)

    # (b) Estimate each term
    # LHS: |v x B| ~ V * B
    term_vxB = V * B

    # Resistive term: eta * J
    eta = spitzer_resistivity(T_e_eV, ln_Lambda=ln_Lambda)
    term_resistive = eta * J

    # Hall term: J x B / (en)
    term_hall = J * B / (e * n)

    # Electron pressure: grad(pe) / (en) ~ n*T_e / (L*e*n) = T_e / (e*L)
    term_pressure = T_e / (e * L)

    # Electron inertia: (m_e/(e^2*n^2)) * dJ/dt ~ (m_e*J*V)/(e^2*n^2*L)
    # Scaling: (d_e/L)^2 * V * B
    term_inertia = (d_e / L)**2 * V * B

    print(f"\n(b) Term magnitudes:")
    print(f"    |v x B|          = {term_vxB:.4e} V/m  (LHS)")
    print(f"    |eta*J|          = {term_resistive:.4e} V/m  (ratio: {term_resistive/term_vxB:.2e})")
    print(f"    |J x B/(en)|     = {term_hall:.4e} V/m  (ratio: {term_hall/term_vxB:.2e})")
    print(f"    |grad(pe)/(en)|  = {term_pressure:.4e} V/m  (ratio: {term_pressure/term_vxB:.2e})")
    print(f"    |me inertia|     = {term_inertia:.4e} V/m  (ratio: {term_inertia/term_vxB:.2e})")

    # (c) Which terms negligible?
    print(f"\n(c) Scale parameters:")
    print(f"    eta/(mu_0*V*L) = {eta/(mu_0*V*L):.2e}  (Magnetic Reynolds number Rm = {mu_0*V*L/eta:.0e})")
    print(f"    d_i/L = {d_i/L:.2e}  (Hall parameter)")
    print(f"    d_e/L = {d_e/L:.2e}  (Electron inertia)")
    print(f"    rho_i/L = {rho_i/L:.2e}  (FLR effects)")
    print()
    print(f"    All RHS terms << v x B")
    print(f"    -> Ideal MHD (E + v x B = 0) is an excellent approximation")
    print(f"    -> Corrections are O(d_i/L) ~ O(10^-4)")
    print()


def exercise_2():
    """
    Exercise 2: CGL Double Adiabatic Pressure Evolution
    Compression: B_0 -> 2*B_0, n = n_0 (constant density).
    """
    print("--- Exercise 2: CGL Pressure Evolution ---")

    # CGL double adiabatic laws:
    # d/dt(p_perp / (n*B)) = 0  ->  p_perp * B / n = const
    # d/dt(p_par * B^2 / n^3) = 0  ->  p_par * B^2 / n^3 = const

    # Wait: the standard CGL equations are:
    # p_perp / (n * B) = const  (first CGL invariant: related to mu conservation)
    # p_par * B^2 / n^3 = const  (second CGL invariant)

    B_0 = 1.0  # Normalized
    B_f = 2.0  # Final B
    n_0 = 1.0  # Normalized (constant)

    # (a) Final pressures
    print(f"(a) CGL double adiabatic laws:")
    print(f"    p_perp / (n*B) = const  ->  p_perp propto n*B")
    print(f"    p_par * B^2 / n^3 = const  ->  p_par propto n^3/B^2")
    print()
    print(f"    Compression: B: {B_0} -> {B_f}, n: {n_0} -> {n_0} (constant)")

    # p_perp_f / (n_0 * B_f) = p_perp_0 / (n_0 * B_0)
    # p_perp_f = p_perp_0 * B_f / B_0
    ratio_perp = B_f / B_0
    print(f"\n    p_perp_f / p_perp_0 = B_f/B_0 = {ratio_perp:.1f}")

    # p_par_f * B_f^2 / n_0^3 = p_par_0 * B_0^2 / n_0^3
    # p_par_f = p_par_0 * (B_0/B_f)^2
    ratio_par = (B_0 / B_f)**2
    print(f"    p_par_f / p_par_0 = (B_0/B_f)^2 = {ratio_par:.4f}")

    # (b) Anisotropy ratio
    # Initially p_perp_0 = p_par_0 = p_0
    # After: p_perp/p_par = (B_f/B_0) / (B_0/B_f)^2 = (B_f/B_0)^3
    anisotropy = (B_f / B_0)**3
    print(f"\n(b) If initially isotropic (p_perp_0 = p_par_0 = p_0):")
    print(f"    p_perp / p_par = (B_f/B_0)^3 = {anisotropy:.1f}")
    print(f"    Strong anisotropy develops from isotropic initial state!")
    print(f"    Perpendicular pressure increases (mu conservation)")
    print(f"    Parallel pressure decreases (particles squeezed along B)")

    # (c) Mirror instability check
    beta_perp_0 = 0.5
    p_perp_f = ratio_perp  # Normalized to p_0
    p_par_f = ratio_par

    # beta values: beta = 2*mu_0*p / B^2
    # beta_perp_f = beta_perp_0 * (p_perp_f/p_0) * (B_0/B_f)^2
    beta_perp_f = beta_perp_0 * ratio_perp * (B_0 / B_f)**2
    beta_par_f = beta_perp_0 * ratio_par * (B_0 / B_f)**2

    mirror_param = beta_perp_f * (p_perp_f / p_par_f - 1)

    print(f"\n(c) Mirror instability check:")
    print(f"    Initial beta_perp = {beta_perp_0}")
    print(f"    Final beta_perp = {beta_perp_f:.4f}")
    print(f"    Final beta_par = {beta_par_f:.6f}")
    print(f"    Mirror parameter: beta_perp*(p_perp/p_par - 1) = {mirror_param:.4f}")
    print(f"    Mirror threshold: > 1")
    if mirror_param > 1:
        print(f"    -> UNSTABLE to mirror instability!")
    else:
        print(f"    -> Stable (mirror parameter < 1)")
    print(f"    Note: in practice, mirror modes would limit the anisotropy")
    print()


def exercise_3():
    """
    Exercise 3: Frozen-In Flux Theorem
    Prove and apply the theorem to flux tube compression.
    """
    print("--- Exercise 3: Frozen-In Flux ---")

    print("(a) Proof of frozen-in theorem:")
    print("    Start: Ideal Ohm's law: E + v x B = 0")
    print("    -> E = -v x B")
    print("    Faraday's law: dB/dt = -curl(E) = curl(v x B)")
    print()
    print("    The magnetic flux through a surface S moving with the fluid:")
    print("    Phi = integral_S B . dA")
    print()
    print("    d(Phi)/dt = integral_S dB/dt . dA + integral_C (v x dl) . B")
    print("    = integral_S curl(v x B) . dA - integral_C B x v . dl")
    print("    = integral_C (v x B) . dl - integral_C (v x B) . dl = 0")
    print("    (using Stokes' theorem)")
    print()
    print("    Therefore d(Phi)/dt = 0. QED.")

    # (b) Numerical example: flux tube compression
    r_0 = 0.1   # Initial radius [m]
    B_0 = 0.1   # Initial field [T]
    r_f = 0.05  # Final radius [m]

    Phi = B_0 * np.pi * r_0**2
    B_f = Phi / (np.pi * r_f**2)

    print(f"\n(b) Flux tube compression:")
    print(f"    Initial: r_0 = {r_0*100:.0f} cm, B_0 = {B_0} T")
    print(f"    Final:   r_f = {r_f*100:.0f} cm (incompressible radial compression)")
    print(f"    Conserved flux: Phi = B_0*pi*r_0^2 = {Phi:.4e} Wb")
    print(f"    Final field: B_f = Phi/(pi*r_f^2) = {B_f:.2f} T")
    print(f"    B_f/B_0 = (r_0/r_f)^2 = {B_f/B_0:.1f}")

    # (c) Physical meaning
    print(f"\n(c) Physical meaning of 'frozen-in':")
    print(f"    - Field lines move with the fluid (in ideal MHD)")
    print(f"    - Topology of field lines is preserved")
    print(f"    - Field lines cannot break or reconnect")
    print(f"    - Consequence: magnetic flux is a material invariant")
    print()
    print(f"    Can field lines reconnect in ideal MHD? NO!")
    print(f"    Reconnection requires non-ideal effects:")
    print(f"    - Resistivity (resistive MHD)")
    print(f"    - Hall term (Hall MHD)")
    print(f"    - Electron inertia (electron MHD)")
    print(f"    - Kinetic effects (collisionless reconnection)")
    print()


def exercise_4():
    """
    Exercise 4: Gyrokinetic Ordering
    Tokamak: L = 1 m, rho_i = 5 mm, omega_ci = 10^8 rad/s.
    """
    print("--- Exercise 4: Gyrokinetic Ordering ---")

    L = 1.0           # Equilibrium scale length [m]
    rho_i = 5e-3      # Ion Larmor radius [m]
    omega_ci = 1e8     # Ion cyclotron frequency [rad/s]

    # (a) Ordering parameter
    delta = rho_i / L

    print(f"(a) Gyrokinetic ordering parameter:")
    print(f"    delta = rho_i/L = {rho_i*1e3:.0f} mm / {L:.0f} m = {delta:.0e}")
    print()
    print(f"    Ordering: rho_i/L ~ omega/omega_ci ~ delta_f/f_0 ~ delta << 1")

    # (b) Maximum resolved frequency
    omega_max = delta * omega_ci
    f_max = omega_max / (2 * np.pi)

    print(f"\n(b) Maximum frequency resolved by gyrokinetics:")
    print(f"    omega_max = delta * omega_ci = {omega_max:.4e} rad/s")
    print(f"    f_max = {f_max:.0f} Hz = {f_max/1e3:.2f} kHz")
    print(f"    Period: T_min = {1/f_max*1e3:.2f} ms")

    # Compare to relevant frequencies
    print(f"\n    Time scale hierarchy:")
    print(f"    omega_ci = {omega_ci:.0e} rad/s (gyration, averaged out)")
    print(f"    omega_max = {omega_max:.0e} rad/s (drift waves, ITG, etc.)")
    T_transit = L / (rho_i * omega_ci)  # Transit time ~ L/v_thi
    print(f"    Transit frequency: omega_t ~ v_thi/L = {rho_i*omega_ci/L:.0e} rad/s")

    # (c) Fast magnetosonic wave
    V_A_est = 1e6  # Typical Alfven speed [m/s]
    k_max = 1.0 / rho_i  # Maximum resolved wavenumber
    omega_fast = k_max * V_A_est  # Fast wave frequency at k_max

    print(f"\n(c) Fast magnetosonic wave:")
    print(f"    omega_fast = k * V_A (no upper limit in k)")
    print(f"    At k ~ 1/rho_i: omega_fast ~ V_A/rho_i = {V_A_est/rho_i:.0e} rad/s")
    print(f"    omega_fast / omega_ci = V_A / (rho_i * omega_ci) = {V_A_est/(rho_i*omega_ci):.0f}")
    print(f"    omega_fast >> omega_ci -> violates gyrokinetic ordering!")
    print()
    print(f"    Why gyrokinetics can't capture fast waves:")
    print(f"    - Gyrokinetics averages over the gyro-period (1/omega_ci)")
    print(f"    - Fast waves oscillate faster than gyration at high k")
    print(f"    - The ordering omega << omega_ci is violated")
    print(f"    - This is actually useful: fast waves are eliminated,")
    print(f"      reducing the CFL condition for numerical stability")
    print()


def exercise_5():
    """
    Exercise 5: Hall MHD vs Sweet-Parker Reconnection
    Solar flare: B = 0.01 T, n = 10^16 m^-3, L = 10^4 km, T_e = 10^6 K.
    """
    print("--- Exercise 5: Hall MHD Reconnection ---")

    B = 0.01         # T
    n = 1e16          # m^-3
    L = 1e7           # 10^4 km = 10^7 m
    T_e_K = 1e6       # K
    T_e_eV = T_e_K * k_B / eV_to_J
    m_i = m_p

    # (a) Alfven speed and ion skin depth
    V_A = B / np.sqrt(mu_0 * n * m_i)
    d_i = c / np.sqrt(n * e**2 / (epsilon_0 * m_i))

    print(f"(a) Solar flare parameters:")
    print(f"    B = {B*1e3:.0f} mT, n = {n:.0e} m^-3")
    print(f"    L = {L/1e3:.0f} km, T_e = {T_e_K:.0e} K = {T_e_eV:.1f} eV")
    print(f"    Alfven speed: V_A = {V_A/1e3:.0f} km/s = {V_A:.4e} m/s")
    print(f"    Ion skin depth: d_i = {d_i:.1f} m")
    print(f"    L/d_i = {L/d_i:.0e}")

    # (b) Sweet-Parker reconnection time
    eta = spitzer_resistivity(T_e_eV, ln_Lambda=17.0)
    S = mu_0 * L * V_A / eta  # Lundquist number
    V_in_SP = V_A / np.sqrt(S)
    tau_SP = L / V_in_SP

    print(f"\n(b) Sweet-Parker reconnection:")
    print(f"    Spitzer resistivity: eta = {eta:.4e} Ohm*m")
    print(f"    Lundquist number: S = mu_0*L*V_A/eta = {S:.2e}")
    print(f"    V_in (Sweet-Parker) = V_A/sqrt(S) = {V_in_SP:.2f} m/s")
    print(f"    tau_SP = L/V_in = {tau_SP:.2e} s = {tau_SP/(3600*24):.0f} days")

    # (c) Hall MHD (Petschek-like) reconnection time
    V_in_Hall = 0.1 * V_A
    tau_Hall = L / V_in_Hall

    print(f"\n(c) Hall MHD reconnection:")
    print(f"    V_in (Hall) ~ 0.1 * V_A = {V_in_Hall/1e3:.0f} km/s")
    print(f"    tau_Hall = L/V_in = {tau_Hall:.0f} s = {tau_Hall/60:.1f} min")

    # (d) Comparison with observations
    tau_obs_min = 60.0   # minutes (typical flare)
    tau_obs_s = tau_obs_min * 60

    print(f"\n(d) Comparison with observations:")
    print(f"    Observed flare timescale: ~{tau_obs_min:.0f} minutes = {tau_obs_s:.0f} s")
    print(f"    Sweet-Parker: tau_SP = {tau_SP:.0e} s ({tau_SP/(3600*24):.0f} days)")
    print(f"    Hall MHD:     tau_Hall = {tau_Hall:.0f} s ({tau_Hall/60:.0f} min)")
    print()
    print(f"    Sweet-Parker is FAR too slow (by factor {tau_SP/tau_obs_s:.0e})")
    print(f"    Hall MHD is consistent with observations!")
    print()
    print(f"    Why Hall MHD is faster:")
    print(f"    - At scale L < d_i, ions decouple from field lines")
    print(f"    - Electrons remain magnetized (d_e << d_i)")
    print(f"    - The reconnection rate becomes independent of resistivity")
    print(f"    - Open X-point geometry (Petschek) rather than Sweet-Parker sheet")
    print(f"    - This 'fast reconnection' explains solar flares, magnetotail, etc.")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
