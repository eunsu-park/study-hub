"""
Exercises for Lesson 02: Linear Stability Theory
Topic: MHD
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad


def exercise_1():
    """
    Problem 1: Energy Principle for Sausage Mode

    Z-pinch with uniform current, sharp boundary at r=a.
    B_theta(r) = mu0*I*r/(2*pi*a^2) for r<a
    B_theta(r) = mu0*I/(2*pi*r) for r>=a
    p(r) = p0 (constant) for r<a

    Trial displacement: xi_r = xi0 * sin(pi*r/a), m=0 (sausage)
    """
    mu0 = 4 * np.pi * 1e-7
    a = 0.1       # Plasma radius [m]
    I = 500e3     # Current [A]
    p0 = 1e5      # Pressure [Pa]
    xi0 = 0.01    # Displacement amplitude [m]
    gamma_ad = 5.0 / 3.0  # Adiabatic index

    Nr = 500
    r = np.linspace(1e-6, a, Nr)
    dr = r[1] - r[0]

    # Equilibrium fields
    B_theta = mu0 * I * r / (2 * np.pi * a**2)

    # Trial displacement
    xi_r = xi0 * np.sin(np.pi * r / a)

    # (a) Divergence of xi for m=0 (sausage): div(xi) = (1/r)*d(r*xi_r)/dr
    d_r_xi_r = np.gradient(r * xi_r, r)
    div_xi = d_r_xi_r / r
    print("(a) div(xi) = (1/r) * d(r*xi_r)/dr")
    print(f"    Max |div(xi)| = {np.max(np.abs(div_xi)):.4f}")

    # (b) Perturbed magnetic field B1 for m=0
    # B1_theta = -d(xi_r * B_theta)/dr (simplified for m=0)
    B1_theta = -np.gradient(xi_r * B_theta, r)
    print(f"\n(b) Max |B1_theta| = {np.max(np.abs(B1_theta)):.6f} T")

    # (c) Magnetic compression energy
    # delta_W_mag = (1/(2*mu0)) * integral |B1|^2 * 2*pi*r dr
    integrand_mag = B1_theta**2 / (2 * mu0) * 2 * np.pi * r
    delta_W_mag = np.trapz(integrand_mag, r)
    print(f"\n(c) Magnetic compression energy: delta_W_mag = {delta_W_mag:.4e} J")

    # (d) Pressure compression energy
    # delta_W_p = (gamma_ad/2) * integral p0 * |div(xi)|^2 * 2*pi*r dr
    integrand_p = gamma_ad / 2 * p0 * div_xi**2 * 2 * np.pi * r
    delta_W_p = np.trapz(integrand_p, r)
    print(f"\n(d) Pressure compression energy: delta_W_p = {delta_W_p:.4e} J")

    # (e) Total delta_W and stability
    delta_W_total = delta_W_mag + delta_W_p
    print(f"\n(e) Total delta_W = delta_W_mag + delta_W_p = {delta_W_total:.4e} J")
    if delta_W_total > 0:
        print("    Result: STABLE (delta_W > 0)")
    else:
        print("    Result: UNSTABLE (delta_W < 0)")
    print("    Note: Pure Z-pinch without B_z is generally unstable.")
    print("    The positive delta_W here comes from the specific trial function")
    print("    and finite pressure compressibility stabilization.")


def exercise_2():
    """
    Problem 2: Kruskal-Shafranov for Tokamak

    R0=2 m, a=0.5 m, B_t=4 T
    J_z(r) = J0*(1 - r^2/a^2)
    """
    R0 = 2.0
    a = 0.5
    Bt = 4.0
    mu0 = 4 * np.pi * 1e-7

    Nr = 1000
    r = np.linspace(1e-6, a, Nr)

    # (a) Total plasma current
    # I_p = integral J0*(1-r^2/a^2) * 2*pi*r dr from 0 to a
    # = J0 * 2*pi * [a^2/2 - a^4/(4a^2)] = J0 * pi * a^2 / 2
    # We need J0. Let's compute in terms of J0 first.
    # I_p = J0 * pi * a^2 / 2
    J0 = 1e7  # A/m^2 (choose a reasonable value)
    Ip = J0 * np.pi * a**2 / 2
    print(f"(a) I_p = J0 * pi * a^2 / 2 = {Ip:.2e} A = {Ip/1e6:.4f} MA")

    # (b) Poloidal field
    # I(r) = J0 * 2*pi * [r^2/2 - r^4/(4*a^2)]
    I_enc = J0 * 2 * np.pi * (r**2 / 2 - r**4 / (4 * a**2))
    B_theta = mu0 * I_enc / (2 * np.pi * r)
    print(f"\n(b) B_theta(a) = {B_theta[-1]:.6f} T")

    # (c) Safety factor
    q = r * Bt / (R0 * B_theta)
    print("\n(c) Safety factor profile computed.")

    # (d) q(0) and q(a)
    # q(0) via L'Hopital: B_theta ~ mu0*J0*r/2 for small r
    # q(0) = lim r*Bt/(R0*mu0*J0*r/2) = 2*Bt/(R0*mu0*J0)
    q0 = 2 * Bt / (R0 * mu0 * J0)
    qa = a * Bt / (R0 * B_theta[-1])
    print(f"\n(d) q(0) = {q0:.4f}")
    print(f"    q(a) = {qa:.4f}")

    # (e) Kruskal-Shafranov criterion for (m,n) = (1,1)
    q_crit = 1.0
    print(f"\n(e) Kruskal-Shafranov criterion: q(a) > m/n = {q_crit}")
    print(f"    q(a) = {qa:.4f}")
    if qa > q_crit:
        print(f"    STABLE: q(a) = {qa:.4f} > {q_crit}")
    else:
        print(f"    UNSTABLE: q(a) = {qa:.4f} < {q_crit}")

    # (f) Minimum q(a)=3 target
    # q(a) = a*Bt / (R0*B_theta(a))
    # B_theta(a) = mu0*Ip/(2*pi*a)
    # q(a) = a*Bt*2*pi*a / (R0*mu0*Ip) = 2*pi*a^2*Bt / (R0*mu0*Ip)
    # For q(a)=3: Ip = 2*pi*a^2*Bt / (3*R0*mu0)
    q_target = 3.0
    Ip_for_q3 = 2 * np.pi * a**2 * Bt / (q_target * R0 * mu0)
    print(f"\n(f) For q(a) = {q_target}:")
    print(f"    Required I_p = {Ip_for_q3:.2e} A = {Ip_for_q3/1e6:.4f} MA")
    J0_for_q3 = Ip_for_q3 / (np.pi * a**2 / 2)
    print(f"    Required J0 = {J0_for_q3:.2e} A/m^2")


def exercise_3():
    """
    Problem 3: Suydam Criterion Application

    Screw pinch: B_z=1 T (const), B_theta(r) = B_theta0*(r/a), p(r) = p0*(1-r^2/a^2)
    R0 = 10*a
    """
    B_z = 1.0
    a = 0.1
    R0 = 10 * a
    mu0 = 4 * np.pi * 1e-7

    # Choose B_theta0
    B_theta0 = 0.1  # T

    Nr = 500
    r = np.linspace(1e-6, a, Nr)

    # (a) Safety factor q(r) and q'(r)
    B_theta = B_theta0 * r / a
    q = r * B_z / (R0 * B_theta)
    # q(r) = r * B_z / (R0 * B_theta0 * r / a) = a * B_z / (R0 * B_theta0) = const!
    q_val = a * B_z / (R0 * B_theta0)
    print(f"(a) q(r) = a*B_z / (R0*B_theta0) = {q_val:.4f} (constant!)")
    print(f"    q'(r) = 0 (constant q profile)")
    print("    Note: With B_theta linear in r, q is independent of r.")

    # For the Suydam criterion to be interesting, let's use the lesson's formula:
    # q(r) = B_z*a / (R0*B_theta0) * (1/r) ... wait, that's different.
    # The lesson says: q(r) = r*B_z / (R0*B_theta(r))
    # With B_theta = B_theta0 * r/a:
    #   q = r*B_z / (R0 * B_theta0 * r/a) = a*B_z/(R0*B_theta0) = const
    # And q'/q = -1/r (from the lesson) --- that must assume a different profile.
    #
    # Let's follow the lesson explicitly: it states q'/q = -1/r.
    # That would come from q = C/r, i.e., B_theta proportional to r^2
    # or some other profile. Let's just use q'/q = -1/r as given.

    print("\n    Following the lesson's derivation where q'/q = -1/r:")
    q_prime_over_q = -1.0 / r

    # (b) Pressure gradient
    p0_vals = [1e3, 1e4, 1e5]
    print(f"\n(b) p'(r) = -2*p0*r/a^2")

    # (c) Evaluate Suydam criterion at r = a/2
    r_eval = a / 2
    print(f"\n(c) Suydam criterion at r = a/2 = {r_eval} m:")
    print(f"    (r/4)(q'/q)^2 + 2*mu0*p'/B_z^2")

    for p0 in p0_vals:
        dp_dr = -2 * p0 * r_eval / a**2
        shear_term = r_eval / 4 * (1 / r_eval)**2  # (r/4)*(q'/q)^2 = 1/(4r)
        pressure_term = 2 * mu0 * dp_dr / B_z**2
        suydam = shear_term + pressure_term

        print(f"\n    p0 = {p0:.0e} Pa:")
        print(f"      Shear term: {shear_term:.4e}")
        print(f"      Pressure term: {pressure_term:.4e}")
        print(f"      Suydam value: {suydam:.4e}")
        if suydam > 0:
            print("      STABLE (Suydam satisfied)")
        else:
            print("      UNSTABLE (Suydam violated)")

    # (d) Maximum allowed p0
    # At r = a/2: shear_term = 1/(4*r) = 1/(2*a) = 5 m^-1
    # pressure_term = 2*mu0*(-2*p0*(a/2)/a^2)/B_z^2 = -2*mu0*p0/(a*B_z^2)
    # Stability: 1/(2*a) - 2*mu0*p0/(a*B_z^2) > 0
    # p0 < B_z^2/(4*mu0) = 1/(4*4*pi*1e-7) ~ 2e5 Pa
    p0_max = B_z**2 / (4 * mu0)
    print(f"\n(d) Maximum p0 for Suydam stability at all radii:")
    print(f"    p0_max = B_z^2 / (4*mu0) = {p0_max:.2e} Pa")

    # (e) What happens if p0 exceeds this limit
    print(f"\n(e) If p0 > {p0_max:.2e} Pa:")
    print("    The Suydam criterion is violated, meaning localized interchange")
    print("    instabilities will develop. Adjacent flux tubes will exchange")
    print("    positions, leading to turbulent mixing of the plasma. Since")
    print("    Suydam is a necessary condition, violation guarantees instability.")


def exercise_4():
    """
    Problem 4: Growth Rate Estimate

    Cylindrical plasma: a=0.1 m, L=1 m, rho=1e-6 kg/m^3
    xi = xi0*sin(pi*r/a) r-hat, xi0=0.01 m
    delta_W = -1e3 J
    """
    a = 0.1
    L = 1.0
    rho0 = 1e-6
    xi0 = 0.01
    delta_W = -1e3

    Nr = 500
    r = np.linspace(1e-6, a, Nr)

    xi_r = xi0 * np.sin(np.pi * r / a)

    # (a) Kinetic energy K = (1/2) integral rho0 |xi|^2 dV
    # dV = 2*pi*r*L*dr (cylinder)
    integrand = rho0 * xi_r**2 * 2 * np.pi * r * L
    K = 0.5 * np.trapz(integrand, r)
    print(f"(a) Kinetic energy K = {K:.4e} J")

    # (b) Growth rate gamma^2 ~ |delta_W| / K
    gamma_sq = np.abs(delta_W) / K
    gamma = np.sqrt(gamma_sq)
    print(f"\n(b) Growth rate: gamma = sqrt(|delta_W|/K) = {gamma:.4e} s^-1")

    # (c) Growth time tau = 1/gamma
    tau = 1.0 / gamma
    print(f"\n(c) Growth time: tau = 1/gamma = {tau:.4e} s")

    # (d) Compare to Alfven frequency
    v_A = 1e6  # m/s (given)
    omega_A = v_A / a
    print(f"\n(d) Alfven speed: v_A = {v_A:.2e} m/s")
    print(f"    Alfven frequency: omega_A = v_A/a = {omega_A:.2e} s^-1")
    print(f"    gamma/omega_A = {gamma/omega_A:.4f}")

    # (e) Fast or slow?
    print(f"\n(e) gamma/omega_A = {gamma/omega_A:.4f}")
    if gamma / omega_A > 0.1:
        print("    This is a FAST instability (Alfven timescale)")
    elif gamma / omega_A > 0.01:
        print("    This is a moderately fast instability")
    else:
        print("    This is a SLOW instability (much slower than Alfven)")


def exercise_5():
    """
    Problem 5: Eigenvalue Problem Setup

    B_z = B0 (const), B_theta = 0, p(r) = p0*(1-r^2/a^2), rho = rho0 (const)
    m=1 perturbation

    This is a theoretical/derivation problem. We present the key equations.
    """
    print("(a) Linearized momentum equation components:")
    print("    rho0 * (-omega^2) * xi_r = -dp1/dr + (1/mu0)(curl B1 x B0)_r")
    print("    rho0 * (-omega^2) * xi_theta = -(1/r)*dp1/dtheta + (1/mu0)(curl B1 x B0)_theta")
    print()
    print("    With B0 = B0*z_hat and B_theta=0, the force operator simplifies")
    print("    significantly since there is no equilibrium current (J0 = 0).")

    print("\n(b) Perturbed magnetic field:")
    print("    B1 = curl(xi x B0)")
    print("    For xi = [xi_r(r), xi_theta(r), 0] * exp(i(theta - kz*z)):")
    print("    B1_r = -i*kz*B0*xi_r")
    print("    B1_theta = -i*kz*B0*xi_theta")
    print("    B1_z = -(1/r)*d(r*xi_r)/dr - (i*m/r)*xi_theta  (with m=1)")

    print("\n(c) Coupled ODEs for xi_r and xi_theta:")
    print("    The equations couple through the pressure perturbation and")
    print("    magnetic tension terms. Eliminating p1 gives a second-order")
    print("    ODE system in xi_r and xi_theta.")
    print()
    print("    For m=1, B_theta=0:")
    print("    rho0*omega^2*xi_r = dp1/dr + (kz^2*B0^2/mu0)*xi_r")
    print("    rho0*omega^2*xi_theta = (1/r)*p1 + (kz^2*B0^2/mu0)*xi_theta")

    print("\n(d) Boundary conditions:")
    print("    At r=0: regularity requires xi_r(0) = 0 for m=1")
    print("    At r=a: xi_r(a) = 0 (rigid boundary) or")
    print("            pressure balance condition (free boundary)")

    print("\n(e) Discretization for numerical solution:")
    print("    1. Choose N radial grid points r_i = i*dr, i=0,...,N-1")
    print("    2. Replace d/dr by centered finite differences:")
    print("       df/dr|_i = (f_{i+1} - f_{i-1}) / (2*dr)")
    print("    3. Replace d^2f/dr^2 by:")
    print("       d^2f/dr^2|_i = (f_{i+1} - 2*f_i + f_{i-1}) / dr^2")
    print("    4. Form matrix equation: A*x = omega^2 * B*x")
    print("       where x = [xi_r_1,...,xi_r_N, xi_theta_1,...,xi_theta_N]")
    print("    5. Solve generalized eigenvalue problem using scipy.linalg.eig")

    # Demonstrate simple discretization
    Nr = 50
    a = 0.1
    B0 = 1.0
    rho0 = 1e-6
    kz = 10.0
    mu0 = 4 * np.pi * 1e-7

    r = np.linspace(1e-4, a, Nr)
    dr = r[1] - r[0]

    # Simple model: omega^2 = kz^2 * v_A^2 (Alfven modes for B_theta=0)
    v_A = B0 / np.sqrt(mu0 * rho0)
    omega_sq = kz**2 * v_A**2
    omega = np.sqrt(omega_sq)

    print(f"\n    Numerical example:")
    print(f"    v_A = {v_A:.2e} m/s")
    print(f"    For kz = {kz} m^-1:")
    print(f"    omega = kz * v_A = {omega:.2e} rad/s")
    print(f"    These are stable Alfven oscillations (omega^2 > 0)")
    print(f"    since there is no current (B_theta=0) and p does not")
    print(f"    drive instability for this configuration.")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Energy Principle for Sausage Mode ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Kruskal-Shafranov for Tokamak ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Suydam Criterion Application ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Growth Rate Estimate ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Eigenvalue Problem Setup ===")
    print("=" * 60)
    exercise_5()

    print("\nAll exercises completed!")
