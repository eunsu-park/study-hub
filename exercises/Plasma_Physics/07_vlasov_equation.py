"""
Plasma Physics - Lesson 07: Vlasov Equation
Exercise Solutions

Topics covered:
- Moments of a drifting Maxwellian distribution
- Bi-Maxwellian anisotropy and pressure tensor
- Kappa distribution vs Maxwellian comparison
- Conservation of entropy in collisionless plasmas
- Linearized Vlasov-Poisson for Langmuir waves
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as gamma_func

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
m_p = 1.673e-27
epsilon_0 = 8.854e-12
k_B = 1.381e-23
eV_to_J = e


def exercise_1():
    """
    Exercise 1: Moments of a Drifting Maxwellian
    f(v) = n/(2*pi*v_th^2)^(3/2) * exp(-(v - u)^2 / (2*v_th^2))
    Compute density, mean velocity, pressure, and heat flux.
    """
    print("--- Exercise 1: Moments of a Drifting Maxwellian ---")

    n_0 = 1e19      # m^-3
    T_eV = 100.0    # eV
    T_J = T_eV * eV_to_J
    v_th = np.sqrt(T_J / m_e)
    u_drift = 1e5    # Drift velocity [m/s]

    print(f"Parameters: n = {n_0:.0e} m^-3, T = {T_eV} eV, u_drift = {u_drift:.0e} m/s")
    print(f"v_th = {v_th:.4e} m/s")
    print(f"u_drift / v_th = {u_drift/v_th:.4f}")

    # 1D drifting Maxwellian: f(v) = (n/sqrt(2*pi*v_th^2)) * exp(-(v-u)^2/(2*v_th^2))

    # (a) Zeroth moment (density): integral f dv = n
    def f_maxwellian(v, n, u, vth):
        return n / np.sqrt(2 * np.pi * vth**2) * np.exp(-(v - u)**2 / (2 * vth**2))

    # Numerical verification
    density, _ = quad(f_maxwellian, -10 * v_th + u_drift, 10 * v_th + u_drift,
                      args=(n_0, u_drift, v_th))
    print(f"\n(a) 0th moment (density):")
    print(f"    Analytical: n = {n_0:.4e} m^-3")
    print(f"    Numerical:  n = {density:.4e} m^-3")

    # (b) First moment (mean velocity): (1/n) * integral v*f dv = u
    def v_f(v, n, u, vth):
        return v * f_maxwellian(v, n, u, vth)

    momentum, _ = quad(v_f, -10 * v_th + u_drift, 10 * v_th + u_drift,
                       args=(n_0, u_drift, v_th))
    mean_v = momentum / density
    print(f"\n(b) 1st moment (mean velocity):")
    print(f"    Analytical: u = {u_drift:.4e} m/s")
    print(f"    Numerical:  u = {mean_v:.4e} m/s")

    # (c) Second moment (pressure): P = m * integral (v-u)^2 * f dv = n*m*v_th^2 = n*T
    def pressure_integrand(v, n, u, vth):
        return m_e * (v - u_drift)**2 * f_maxwellian(v, n, u, vth)

    P_numerical, _ = quad(pressure_integrand, -10 * v_th + u_drift, 10 * v_th + u_drift,
                          args=(n_0, u_drift, v_th))
    P_analytical = n_0 * T_J

    print(f"\n(c) 2nd moment (pressure):")
    print(f"    Analytical: P = n*T = {P_analytical:.4e} Pa")
    print(f"    Numerical:  P = {P_numerical:.4e} Pa")

    # (d) Third moment (heat flux): q = m * integral (v-u)^3 * f dv = 0 for Maxwellian
    def heat_flux_integrand(v, n, u, vth):
        return m_e * (v - u_drift)**3 * f_maxwellian(v, n, u, vth)

    q_numerical, _ = quad(heat_flux_integrand, -10 * v_th + u_drift, 10 * v_th + u_drift,
                          args=(n_0, u_drift, v_th))

    print(f"\n(d) 3rd moment (heat flux):")
    print(f"    Analytical: q = 0 (by symmetry of Maxwellian)")
    print(f"    Numerical:  q = {q_numerical:.4e} W/m^2")
    print(f"    (Maxwellian has zero heat flux -> closure for fluid equations)")
    print()


def exercise_2():
    """
    Exercise 2: Bi-Maxwellian Anisotropy
    f(v_perp, v_par) = n * (m/(2*pi))^(3/2) / (T_perp * sqrt(T_par))
                       * exp(-m*v_perp^2/(2*T_perp) - m*v_par^2/(2*T_par))
    """
    print("--- Exercise 2: Bi-Maxwellian Anisotropy ---")

    n = 1e19
    T_perp_eV = 500.0
    T_par_eV = 100.0
    T_perp = T_perp_eV * eV_to_J
    T_par = T_par_eV * eV_to_J

    # (a) Pressure tensor components
    P_perp = n * T_perp
    P_par = n * T_par
    anisotropy = T_perp / T_par

    print(f"(a) Bi-Maxwellian pressure tensor:")
    print(f"    T_perp = {T_perp_eV} eV, T_par = {T_par_eV} eV")
    print(f"    P_perp = n*T_perp = {P_perp:.4e} Pa")
    print(f"    P_par  = n*T_par  = {P_par:.4e} Pa")
    print(f"    Anisotropy ratio: T_perp/T_par = {anisotropy:.1f}")

    # (b) Scalar pressure and effective temperature
    # P_scalar = (2*P_perp + P_par) / 3 (trace of pressure tensor / 3)
    P_scalar = (2 * P_perp + P_par) / 3
    T_eff = P_scalar / n
    T_eff_eV = T_eff / eV_to_J

    print(f"\n(b) Effective (scalar) quantities:")
    print(f"    P_scalar = (2*P_perp + P_par)/3 = {P_scalar:.4e} Pa")
    print(f"    T_eff = P_scalar/n = {T_eff_eV:.1f} eV")

    # (c) Firehose instability threshold: T_par > T_perp + B^2/(mu_0*n)
    # Mirror instability threshold: T_perp/T_par > 1 + 1/beta_perp
    B = 0.01  # T (e.g., solar wind)
    from scipy.constants import mu_0 as mu0

    beta_perp = 2 * mu0 * P_perp / B**2
    beta_par = 2 * mu0 * P_par / B**2

    print(f"\n(c) Instability thresholds (B = {B*1e3} mT):")
    print(f"    beta_perp = {beta_perp:.2f}")
    print(f"    beta_par  = {beta_par:.2f}")

    # Mirror instability: T_perp/T_par - 1 > 1/beta_perp
    mirror_LHS = T_perp / T_par - 1
    mirror_RHS = 1.0 / beta_perp
    mirror_unstable = mirror_LHS > mirror_RHS

    print(f"    Mirror threshold: T_perp/T_par - 1 = {mirror_LHS:.2f} > 1/beta_perp = {mirror_RHS:.4f}?")
    print(f"    Mirror unstable: {mirror_unstable}")

    # Firehose instability: 1 - T_perp/T_par > 2/beta_par
    firehose_LHS = 1 - T_perp / T_par
    firehose_RHS = 2.0 / beta_par
    firehose_unstable = firehose_LHS > firehose_RHS

    print(f"    Firehose threshold: 1 - T_perp/T_par = {firehose_LHS:.2f} > 2/beta_par = {firehose_RHS:.4f}?")
    print(f"    Firehose unstable: {firehose_unstable}")

    # (d) Which instability dominates?
    if T_perp > T_par:
        print(f"\n(d) T_perp > T_par: Mirror instability is relevant")
        print(f"    Physical mechanism: pressure anisotropy creates magnetic bottles")
    else:
        print(f"\n(d) T_par > T_perp: Firehose instability is relevant")
        print(f"    Physical mechanism: field line tension insufficient to balance pressure")
    print()


def exercise_3():
    """
    Exercise 3: Kappa Distribution vs Maxwellian
    f_kappa(v) = n * A_kappa / (pi*kappa*theta^2)^(3/2)
                 * (1 + v^2/(kappa*theta^2))^(-(kappa+1))
    Compare tails, moments, and entropy.
    """
    print("--- Exercise 3: Kappa Distribution vs Maxwellian ---")

    n = 1e6          # Solar wind density [m^-3]
    T_eV = 10.0      # Temperature [eV]
    T_J = T_eV * eV_to_J
    v_th = np.sqrt(T_J / m_p)  # Proton thermal speed

    # Kappa distribution (1D for simplicity):
    # f_kappa(v) = n * Gamma(kappa+1) / (sqrt(pi*kappa) * theta * Gamma(kappa+1/2))
    #              * (1 + v^2/(kappa*theta^2))^(-(kappa+1))
    # where theta^2 = (2*kappa-3)/(kappa) * v_th^2 (for kappa > 3/2)

    kappa_values = [2, 3, 5, 10, 50]

    # (a) Compare distribution tails
    v_values = np.linspace(0, 6 * v_th, 200)

    # Maxwellian (1D)
    f_max = n / np.sqrt(2 * np.pi * v_th**2) * np.exp(-v_values**2 / (2 * v_th**2))

    print(f"(a) Distribution function values at v = 4*v_th:")
    print(f"    v_th = {v_th:.4e} m/s")
    print(f"    {'Distribution':>15} {'f(4*v_th)':>14} {'f(4*v_th)/f_Max':>16}")
    print("    " + "-" * 50)

    v_test = 4 * v_th
    f_max_test = n / np.sqrt(2 * np.pi * v_th**2) * np.exp(-v_test**2 / (2 * v_th**2))
    print(f"    {'Maxwellian':>15} {f_max_test:.4e} {'1.0':>16}")

    for kappa in kappa_values:
        if kappa <= 1.5:
            continue
        theta2 = (2 * kappa - 3) / kappa * v_th**2
        theta = np.sqrt(theta2)
        A = gamma_func(kappa + 1) / (np.sqrt(np.pi * kappa) * theta * gamma_func(kappa + 0.5))
        f_kap = n * A * (1 + v_test**2 / (kappa * theta2))**(-(kappa + 1))
        print(f"    {'kappa='+str(kappa):>15} {f_kap:.4e} {f_kap/f_max_test:>16.2f}")

    # (b) Second moment: temperature
    print(f"\n(b) Effective temperature (2nd moment):")
    print(f"    For kappa distribution, T_eff = T * kappa/(kappa - 3/2)")
    print(f"    {'kappa':>8} {'T_eff/T':>10} {'T_eff [eV]':>12}")
    print("    " + "-" * 34)
    for kappa in kappa_values:
        if kappa <= 1.5:
            T_ratio = float('inf')
        else:
            T_ratio = kappa / (kappa - 1.5)
        T_eff = T_eV * T_ratio
        print(f"    {kappa:>8d} {T_ratio:>10.3f} {T_eff:>12.2f}")
    print(f"    {'inf (Max)':>8} {'1.000':>10} {T_eV:>12.2f}")

    # (c) High-energy tail fraction (v > 3*v_th)
    print(f"\n(c) Fraction of particles with |v| > 3*v_th:")
    # Maxwellian: erfc(3/sqrt(2))
    from scipy.special import erfc
    frac_max = erfc(3.0 / np.sqrt(2))
    print(f"    Maxwellian: {frac_max:.6f} = {frac_max*100:.4f}%")

    for kappa in kappa_values:
        if kappa <= 1.5:
            continue
        theta2 = (2 * kappa - 3) / kappa * v_th**2
        theta = np.sqrt(theta2)
        A = gamma_func(kappa + 1) / (np.sqrt(np.pi * kappa) * theta * gamma_func(kappa + 0.5))

        def f_kap(v):
            return n * A * (1 + v**2 / (kappa * theta2))**(-(kappa + 1))

        # Fraction above 3*v_th (both tails)
        tail, _ = quad(f_kap, 3 * v_th, 20 * v_th)
        frac = 2 * tail / n
        print(f"    kappa={kappa}: {frac:.6f} = {frac*100:.4f}%  (ratio to Max: {frac/frac_max:.1f}x)")

    print(f"\n    Key: Kappa distributions have power-law tails -> more energetic particles")
    print(f"    Common in space plasmas (solar wind, magnetosphere)")
    print(f"    kappa -> infinity recovers the Maxwellian")
    print()


def exercise_4():
    """
    Exercise 4: Conservation of Entropy
    Boltzmann entropy: S = -integral f*ln(f) dv
    Show S is conserved for collisionless (Vlasov) evolution.
    """
    print("--- Exercise 4: Conservation of Entropy ---")

    # S[f] = -integral f(v) * ln(f(v)) dv
    # dS/dt = -integral (df/dt)*ln(f) dv - integral (df/dt) dv
    # The second term vanishes (particle conservation).
    # Using the Vlasov equation: df/dt = -v*df/dx - (q*E/m)*df/dv
    # After integration by parts, dS/dt = 0

    print("Proof that S = -int f*ln(f) dv is conserved by Vlasov equation:")
    print()
    print("dS/dt = -integral [df/dt * (1 + ln(f))] dv")
    print("      = -integral [-v*(df/dx) - (qE/m)*(df/dv)] * (1 + ln(f)) dv")
    print()
    print("The v*(df/dx) term: integrate by parts in x (periodic or vanishing BC)")
    print("  -> gives zero (divergence form)")
    print()
    print("The (qE/m)*(df/dv) term: integrate by parts in v")
    print("  integral (df/dv)*(1 + ln(f)) dv = integral d/dv[f*ln(f)] dv = 0")
    print("  (vanishes at boundaries f -> 0 as v -> +/- infinity)")
    print()
    print("Therefore dS/dt = 0. QED.")
    print()

    # Numerical verification: evolve a simple distribution and check entropy
    print("Numerical verification with free-streaming:")
    Nv = 1000
    v_max = 5.0
    v = np.linspace(-v_max, v_max, Nv)
    dv = v[1] - v[0]

    # Maxwellian initial condition
    v_th = 1.0
    f0 = np.exp(-v**2 / (2 * v_th**2)) / np.sqrt(2 * np.pi * v_th**2)

    # Add a small perturbation (non-equilibrium)
    f0_perturbed = f0 * (1 + 0.3 * np.cos(2 * v))
    f0_perturbed = np.maximum(f0_perturbed, 1e-30)

    # Entropy of initial state
    S0 = -np.sum(f0_perturbed * np.log(f0_perturbed)) * dv

    # "Free streaming" in velocity space doesn't change entropy
    # (just a rearrangement). But let's show the Casimir invariant:
    # C_2 = integral f^2 dv is also conserved
    C2_0 = np.sum(f0_perturbed**2) * dv

    print(f"  Initial entropy: S = {S0:.6f}")
    print(f"  Initial C_2:     C_2 = {C2_0:.6e}")

    # For a Maxwellian (equilibrium):
    S_max = -np.sum(f0 * np.log(np.maximum(f0, 1e-30))) * dv
    C2_max = np.sum(f0**2) * dv

    print(f"  Maxwellian entropy: S_max = {S_max:.6f}")
    print(f"  S_perturbed < S_max? {S0 < S_max}")
    print(f"  -> Maxwellian has the maximum entropy for given n, T")
    print()
    print("  All Casimir invariants C_n = int f^n dv are conserved by Vlasov:")
    print("  This is because Vlasov preserves the phase space density exactly.")
    print("  Coarse-grained entropy can increase (phase mixing), but fine-grained cannot.")
    print()


def exercise_5():
    """
    Exercise 5: Linearized Vlasov-Poisson for Langmuir Waves
    Derive the dielectric function and find the Langmuir wave dispersion.
    """
    print("--- Exercise 5: Linearized Vlasov-Poisson ---")

    n0 = 1e18
    T_eV = 10.0
    T_J = T_eV * eV_to_J
    v_th = np.sqrt(T_J / m_e)
    omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
    lambda_D = v_th / omega_pe

    print("Linearized Vlasov-Poisson system:")
    print("  df1/dt + v*df1/dx - (e/m_e)*E1*df0/dv = 0  (perturbed Vlasov)")
    print("  dE1/dx = -(e/epsilon_0)*n1                   (Poisson)")
    print()
    print("Assuming f1, E1 ~ exp(ikx - iwt):")
    print("  f1 = (e*E1/(m_e)) * (df0/dv) / (omega - k*v)")
    print("  -> epsilon(omega, k) = 1 + (omega_pe^2/k^2) * integral (df0/dv)/(v - omega/k) dv")
    print()

    # (a) For Maxwellian f0: epsilon involves the plasma dispersion function Z(zeta)
    # epsilon(omega, k) = 1 - (1/(k*lambda_D)^2) * [1 + zeta * Z(zeta)]
    # where zeta = omega / (k * v_th * sqrt(2))

    print(f"(a) Parameters: n0 = {n0:.0e} m^-3, T_e = {T_eV} eV")
    print(f"    omega_pe = {omega_pe:.4e} rad/s")
    print(f"    lambda_D = {lambda_D:.4e} m")
    print(f"    v_th = {v_th:.4e} m/s")

    # (b) Bohm-Gross dispersion: omega^2 = omega_pe^2 + 3*k^2*v_th^2
    k_values = np.array([0.01, 0.05, 0.1, 0.2, 0.3]) / lambda_D

    print(f"\n(b) Bohm-Gross dispersion relation:")
    print(f"    omega^2 = omega_pe^2 + 3*k^2*v_th^2")
    print(f"\n    {'k*lambda_D':>12} {'omega/omega_pe':>16} {'v_phi/v_th':>12} {'v_g/v_th':>12}")
    print("    " + "-" * 56)

    for k in k_values:
        k_lD = k * lambda_D
        omega = np.sqrt(omega_pe**2 + 3 * k**2 * v_th**2)
        v_phi = omega / k
        v_g = 3 * k * v_th**2 / omega  # d(omega)/dk

        print(f"    {k_lD:>12.3f} {omega/omega_pe:>16.6f} {v_phi/v_th:>12.3f} {v_g/v_th:>12.6f}")

    # (c) Physical interpretation
    print(f"\n(c) Physical interpretation:")
    print(f"    - Phase velocity v_phi >> v_th for k*lambda_D << 1")
    print(f"      -> Wave is faster than most particles -> weak damping")
    print(f"    - As k*lambda_D -> 1, v_phi -> v_th")
    print(f"      -> More resonant particles -> strong Landau damping")
    print(f"    - Group velocity v_g << v_th for small k")
    print(f"      -> Wave energy propagates slowly")
    print(f"    - Product: v_phi * v_g ~ 3*v_th^2 (for k*lambda_D << 1)")

    # (d) Threshold for wave-particle interaction
    print(f"\n(d) Resonant particle fraction:")
    print(f"    Resonant particles satisfy v ~ omega/k = v_phi")
    print(f"    Fraction of particles near resonance:")
    for k in k_values:
        k_lD = k * lambda_D
        omega = np.sqrt(omega_pe**2 + 3 * k**2 * v_th**2)
        v_phi = omega / k
        # f(v_phi) / f(0) = exp(-v_phi^2/(2*v_th^2))
        frac = np.exp(-v_phi**2 / (2 * v_th**2))
        print(f"    k*lambda_D = {k_lD:.3f}: v_phi/v_th = {v_phi/v_th:.2f}, "
              f"f(v_phi)/f(0) = {frac:.2e}")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
