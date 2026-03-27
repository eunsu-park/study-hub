"""
Lesson 16: Relativistic MHD
Topic: MHD
Description: Exercises on Lorentz transformations of electromagnetic fields,
             primitive variable recovery via Newton-Raphson, relativistic
             shock tubes, magnetization parameter, light cylinder radius,
             ISCO velocity, relativistic Alfven speed, and GRMHD timestep.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def exercise_1():
    """Lorentz Transformations.

    Given E = (1, 0, 0) and B = (0, 1, 0) in the lab frame, compute
    E' and B' in a frame moving with v = (0.5c, 0, 0).
    Verify E' . B' = E . B (Lorentz invariant).
    """
    c = 1.0  # normalized (c = 1)

    E = np.array([1.0, 0.0, 0.0])
    B = np.array([0.0, 1.0, 0.0])
    v = np.array([0.5, 0.0, 0.0])  # v/c

    beta = np.linalg.norm(v)
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    v_hat = v / beta if beta > 0 else np.array([1, 0, 0])

    # Lorentz transformation of E and B for boost along arbitrary direction:
    # E_parallel' = E_parallel
    # E_perp' = gamma * (E_perp + v x B)
    # B_parallel' = B_parallel
    # B_perp' = gamma * (B_perp - v x E / c^2)

    # Decompose into parallel and perpendicular to v
    E_par = np.dot(E, v_hat) * v_hat
    E_perp = E - E_par
    B_par = np.dot(B, v_hat) * v_hat
    B_perp = B - B_par

    # Transform
    E_par_prime = E_par
    E_perp_prime = gamma * (E_perp + np.cross(v, B))
    B_par_prime = B_par
    B_perp_prime = gamma * (B_perp - np.cross(v, E) / c**2)

    E_prime = E_par_prime + E_perp_prime
    B_prime = B_par_prime + B_perp_prime

    # Lorentz invariants
    EdotB = np.dot(E, B)
    EdotB_prime = np.dot(E_prime, B_prime)

    # Second invariant: E^2 - B^2
    E2_B2 = np.dot(E, E) - np.dot(B, B)
    E2_B2_prime = np.dot(E_prime, E_prime) - np.dot(B_prime, B_prime)

    print(f"  Lab frame:")
    print(f"    E = ({E[0]:.3f}, {E[1]:.3f}, {E[2]:.3f})")
    print(f"    B = ({B[0]:.3f}, {B[1]:.3f}, {B[2]:.3f})")
    print(f"    v = ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}) c")
    print(f"    gamma = {gamma:.4f}")
    print()
    print(f"  Boosted frame:")
    print(f"    E' = ({E_prime[0]:.4f}, {E_prime[1]:.4f}, {E_prime[2]:.4f})")
    print(f"    B' = ({B_prime[0]:.4f}, {B_prime[1]:.4f}, {B_prime[2]:.4f})")
    print()
    print(f"  Lorentz invariants:")
    print(f"    E . B  = {EdotB:.6f}")
    print(f"    E'. B' = {EdotB_prime:.6f}")
    print(f"    |E.B - E'.B'| = {abs(EdotB - EdotB_prime):.2e}  (should be ~0)")
    print()
    print(f"    E^2 - B^2   = {E2_B2:.6f}")
    print(f"    E'^2 - B'^2 = {E2_B2_prime:.6f}")
    print(f"    |difference| = {abs(E2_B2 - E2_B2_prime):.2e}  (should be ~0)")


def exercise_2():
    """Primitive Recovery.

    Implement a 1D Newton-Raphson primitive recovery routine.
    Test on: D = 2.0, S_x = 1.0, tau = 3.0, B_x = 0.5, B_y = 1.0.
    """
    # Conserved variables
    D = 2.0      # rest-mass density * Lorentz factor: D = rho * W
    Sx = 1.0     # momentum density: S_i = (rho*h*W^2 + B^2)*v_i - (B.v)*B_i
    tau = 3.0    # energy density - D: tau = rho*h*W^2 - p + B^2/2 + (B.v)^2/2 - D
    Bx = 0.5     # magnetic field
    By = 1.0
    Bz = 0.0

    # Ideal gas EOS: p = (Gamma - 1) * rho * epsilon, h = 1 + epsilon + p/rho
    Gamma = 5.0 / 3.0

    # Initial guess for primitives
    rho_guess = 1.0
    vx_guess = 0.3
    p_guess = 1.0

    def compute_conserved(rho, vx, p):
        """Compute conserved from primitives (1D, vy=vz=0)."""
        v2 = vx**2
        W = 1.0 / np.sqrt(1.0 - v2)
        eps = p / ((Gamma - 1.0) * rho)
        h = 1.0 + eps + p / rho

        B2 = Bx**2 + By**2 + Bz**2
        Bv = Bx * vx  # B.v (only x component of v is nonzero)

        D_calc = rho * W
        S_calc = (rho * h * W**2 + B2) * vx - Bv * Bx
        tau_calc = rho * h * W**2 - p + 0.5 * B2 + 0.5 * Bv**2 - D_calc

        return D_calc, S_calc, tau_calc

    def residual(x):
        """Residual function for Newton-Raphson."""
        rho_try, vx_try, p_try = x

        if rho_try <= 0 or p_try <= 0 or abs(vx_try) >= 0.999:
            return [1e10, 1e10, 1e10]

        D_calc, S_calc, tau_calc = compute_conserved(rho_try, vx_try, p_try)
        return [D_calc - D, S_calc - Sx, tau_calc - tau]

    # Newton-Raphson via scipy
    x0 = [rho_guess, vx_guess, p_guess]

    print(f"  Conserved variables: D={D}, S_x={Sx}, tau={tau}")
    print(f"  Magnetic field: B_x={Bx}, B_y={By}")
    print(f"  EOS: Gamma = {Gamma:.3f}")
    print(f"  Initial guess: rho={rho_guess}, v_x={vx_guess}, p={p_guess}")
    print()

    solution, info, ier, msg = fsolve(residual, x0, full_output=True)
    rho_sol, vx_sol, p_sol = solution

    if ier == 1:
        # Verify
        D_check, S_check, tau_check = compute_conserved(rho_sol, vx_sol, p_sol)
        W_sol = 1.0 / np.sqrt(1.0 - vx_sol**2)
        eps_sol = p_sol / ((Gamma - 1.0) * rho_sol)
        h_sol = 1.0 + eps_sol + p_sol / rho_sol

        print(f"  Converged! Solution:")
        print(f"    rho = {rho_sol:.6f}")
        print(f"    v_x = {vx_sol:.6f}")
        print(f"    p   = {p_sol:.6f}")
        print(f"    W (Lorentz factor) = {W_sol:.6f}")
        print(f"    h (enthalpy) = {h_sol:.6f}")
        print(f"    epsilon (specific internal energy) = {eps_sol:.6f}")
        print()
        print(f"  Verification (recomputed conserved):")
        print(f"    D:   {D_check:.6f} (target: {D})")
        print(f"    S_x: {S_check:.6f} (target: {Sx})")
        print(f"    tau: {tau_check:.6f} (target: {tau})")
        print(f"    Max error: {max(abs(D_check-D), abs(S_check-Sx), abs(tau_check-tau)):.2e}")
        print(f"  Number of function evaluations: {info['nfev']}")
    else:
        print(f"  Newton-Raphson did NOT converge: {msg}")


def exercise_3():
    """Relativistic Brio-Wu with B_x = 0.

    Compare wave structure with B_x = 0 (purely transverse field)
    vs the standard B_x = 0.5 case.
    """
    # Standard Brio-Wu (relativistic)
    # Left state:  rho=1, p=1, vx=0, By=1,   Bx=0.5
    # Right state: rho=0.125, p=0.1, vx=0, By=-1, Bx=0.5

    # Modified: B_x = 0 (purely transverse field)
    # This removes fast/slow magnetosonic waves along x that depend on B_x
    # Only transverse Alfven-like waves and HD shocks remain

    print("  Relativistic Brio-Wu Shock Tube Comparison:")
    print("  ============================================")
    print()
    print("  Standard case (B_x = 0.5):")
    print("    Left:  rho=1.0, p=1.0, v_x=0, B_y=1.0")
    print("    Right: rho=0.125, p=0.1, v_x=0, B_y=-1.0")
    print("    Wave structure: 7 waves (fast shock, Alfven, slow, contact,")
    print("                   slow, Alfven, fast)")
    print()
    print("  Modified case (B_x = 0):")
    print("    Same states but B_x = 0 (purely transverse field)")
    print("    Wave structure simplified:")
    print("    - No coupling between longitudinal and transverse modes")
    print("    - Fast magnetosonic speed depends only on B_y (perpendicular)")
    print("    - Compound waves may form where fast and Alfven coincide")
    print()

    # Simple numerical solution using HLL-type scheme
    N = 400
    x = np.linspace(-0.5, 0.5, N)
    dx = x[1] - x[0]

    Gamma = 2.0  # relativistic Brio-Wu uses gamma = 2

    for case_name, Bx_val in [("Standard (Bx=0.5)", 0.5), ("Modified (Bx=0)", 0.0)]:
        # Initialize
        rho = np.where(x < 0, 1.0, 0.125)
        p = np.where(x < 0, 1.0, 0.1)
        vx = np.zeros(N)
        By = np.where(x < 0, 1.0, -1.0)
        Bx = np.full(N, Bx_val)

        # Approximate wave speeds
        # Fast magnetosonic: c_f ~ sqrt(c_s^2 + v_A^2)
        c_s = np.sqrt(Gamma * p / (rho + Gamma * p / (Gamma - 1)))
        B2 = Bx**2 + By**2
        rho_h = rho + Gamma * p / (Gamma - 1)
        v_A = np.sqrt(B2 / (rho_h + B2))
        c_f = np.sqrt(c_s**2 + v_A**2 * (1 - c_s**2))  # relativistic combination

        print(f"  {case_name}:")
        print(f"    Left:  c_s={c_s[0]:.3f}, v_A={v_A[0]:.3f}, c_f={c_f[0]:.3f}")
        print(f"    Right: c_s={c_s[-1]:.3f}, v_A={v_A[-1]:.3f}, c_f={c_f[-1]:.3f}")

    print()
    print("  Key differences with B_x = 0:")
    print("  1. The Alfven speed along x vanishes (no B_x component)")
    print("  2. The 7-wave structure degenerates (some waves coincide)")
    print("  3. The fast shock speed differs because c_f depends on B_x")
    print("  4. Rotational discontinuities disappear (no B_x to rotate around)")
    print("  5. The solution is closer to a 1D HD shock tube with B_y pressure")

    # Illustrative plot of wave speeds
    fig, ax = plt.subplots(figsize=(8, 5))
    Bx_range = np.linspace(0, 2, 100)
    rho_test = 1.0
    p_test = 1.0
    By_test = 1.0
    rho_h_test = rho_test + Gamma * p_test / (Gamma - 1)
    c_s_test = np.sqrt(Gamma * p_test / rho_h_test)

    for By_val in [0.5, 1.0, 2.0]:
        B2_test = Bx_range**2 + By_val**2
        vA_test = np.sqrt(B2_test / (rho_h_test + B2_test))
        cf_test = np.sqrt(c_s_test**2 + vA_test**2 * (1 - c_s_test**2))
        # Alfven speed along x: vAx = |Bx|/sqrt(rho_h + B^2)
        vAx_test = np.abs(Bx_range) / np.sqrt(rho_h_test + B2_test)
        ax.plot(Bx_range, cf_test, linewidth=2, label=f'$c_f$ ($B_y$={By_val})')
        ax.plot(Bx_range, vAx_test, '--', linewidth=1.5, label=f'$v_{{Ax}}$ ($B_y$={By_val})')

    ax.set_xlabel(r'$B_x$', fontsize=12)
    ax.set_ylabel('Wave speed / c', fontsize=12)
    ax.set_title('Relativistic MHD Wave Speeds vs $B_x$', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('16_wave_speeds.png', dpi=150)
    plt.close()
    print("  Plot saved to 16_wave_speeds.png")


def exercise_4():
    """Magnetization Parameter.

    For a pulsar with B = 10^12 G, rho = 10^7 g/cm^3, Gamma = 10,
    compute sigma = B^2/(4*pi*rho*h*W^2*c^2). Assume h ~ 1.
    """
    B_G = 1e12           # Gauss
    rho_cgs = 1e7        # g/cm^3
    Gamma_L = 10.0       # Lorentz factor
    h = 1.0              # specific enthalpy (cold plasma)
    c = 3e10             # cm/s

    # Magnetization parameter (in Gaussian CGS):
    # sigma = B^2 / (4*pi * rho * h * W^2 * c^2)
    # where W = Gamma_L (Lorentz factor)

    sigma = B_G**2 / (4 * np.pi * rho_cgs * h * Gamma_L**2 * c**2)

    print(f"  Pulsar parameters:")
    print(f"    B = {B_G:.1e} G")
    print(f"    rho = {rho_cgs:.1e} g/cm^3")
    print(f"    Lorentz factor W = {Gamma_L}")
    print(f"    Specific enthalpy h ~ {h}")
    print(f"  Magnetization parameter:")
    print(f"    sigma = B^2 / (4*pi*rho*h*W^2*c^2)")
    print(f"    sigma = {sigma:.4e}")
    print()

    if sigma >> 1:
        print(f"  sigma = {sigma:.2e} >> 1: FORCE-FREE regime")
        print(f"  Magnetic energy dominates over particle energy.")
        print(f"  MHD description may break down; need force-free electrodynamics.")
    elif sigma > 1:
        print(f"  sigma = {sigma:.2e} > 1: MAGNETICALLY DOMINATED")
        print(f"  Relativistic MHD with strong field effects.")
    else:
        print(f"  sigma = {sigma:.2e} < 1: FLUID-DOMINATED")
        print(f"  Standard relativistic MHD is appropriate.")

    # In the pulsar magnetosphere, sigma varies dramatically with radius
    print()
    print("  Note: In pulsar wind, sigma decreases with radius.")
    print("  At the light cylinder: sigma ~ 10^4-10^6 (force-free)")
    print("  At the termination shock (Crab Nebula): sigma ~ 0.003 (sigma problem!)")


def exercise_5():
    """Light Cylinder.

    For the Crab pulsar (P = 33 ms), compute R_L. Compare to R_NS ~ 10 km.
    """
    P = 33e-3            # s (period)
    c = 3e8              # m/s
    R_NS = 10e3          # m (neutron star radius)

    # Light cylinder radius: R_L = c * P / (2*pi)
    # At R_L, the corotation velocity equals c
    Omega = 2 * np.pi / P
    R_L = c / Omega

    ratio = R_L / R_NS

    print(f"  Crab pulsar period: P = {P * 1e3:.0f} ms")
    print(f"  Angular velocity: Omega = 2*pi/P = {Omega:.1f} rad/s")
    print(f"  Light cylinder radius: R_L = c/Omega = c*P/(2*pi)")
    print(f"  R_L = {R_L:.3e} m = {R_L / 1e3:.0f} km")
    print(f"  Neutron star radius: R_NS = {R_NS / 1e3:.0f} km")
    print(f"  R_L / R_NS = {ratio:.0f}")
    print()
    print(f"  At the light cylinder, corotation velocity = c:")
    print(f"    v = Omega * R_L = {Omega * R_L / c:.3f} c")
    print()
    print(f"  Inside R_L: plasma corotates with the pulsar (force-free)")
    print(f"  Outside R_L: magnetic field lines open, forming the pulsar wind")
    print(f"  The spin-down luminosity is ~{(Omega * R_L)**4:.1e} times higher than")
    print(f"  the magnetic dipole radiation formula predicts (relativistic corrections)")


def exercise_6():
    """ISCO Velocity.

    Compute orbital velocity at the ISCO for Schwarzschild (r = 6GM/c^2)
    and Kerr (a = 0.998) black holes.
    """
    # Schwarzschild ISCO: r_ISCO = 6 GM/c^2 = 6 r_g
    # (using G = M = c = 1 natural units)

    # Orbital velocity at ISCO (Schwarzschild):
    # v_orbit = sqrt(M / (2*r)) at r = 6M (in geometric units)
    # More precisely: v/c = 1/sqrt(2*r/M - 3) for circular orbits
    # At ISCO (r=6M): v/c = 1/sqrt(12-3) = 1/3

    r_ISCO_Sch = 6.0  # in units of r_g = GM/c^2

    # Circular orbit velocity in Schwarzschild:
    # v = sqrt(M / (r - 2M)) * (r-2M)/r (in coordinate velocity)
    # But the orbital velocity measured by a local ZAMO observer:
    # v_local = sqrt(M/r) / sqrt(1 - 2M/r)  (Schwarzschild)
    # Wait - let me use the standard formula:
    # For circular orbits: v_phi = r * d_phi/d_tau_proper_of_observer
    # The proper velocity of circular orbit:
    # u^phi = sqrt(M / (r^3 - 3Mr^2)) (Schwarzschild, r in geometric units)
    # Lorentz factor: W = 1/sqrt(1 - 3M/r) at circular orbit

    # At ISCO r = 6M:
    r_isco = 6.0  # in units of M (geometric: G=c=1)
    W_isco_sch = 1.0 / np.sqrt(1.0 - 3.0 / r_isco)
    v_isco_sch = np.sqrt(1.0 - 1.0 / W_isco_sch**2)

    print(f"  Schwarzschild Black Hole:")
    print(f"    ISCO radius: r_ISCO = 6 GM/c^2")
    print(f"    Orbital velocity at ISCO: v/c = sqrt(1 - 1/W^2)")
    print(f"    Lorentz factor W = 1/sqrt(1 - 3M/r) = 1/sqrt(1-0.5) = {W_isco_sch:.4f}")
    print(f"    v/c = {v_isco_sch:.4f}")
    print(f"    This is mildly relativistic.")
    print()

    # Kerr ISCO for a = 0.998 (prograde orbit):
    # For prograde ISCO of nearly extremal Kerr: r_ISCO -> M (in Boyer-Lindquist)
    # Formula: r_ISCO = M * (3 + Z2 - sqrt((3-Z1)(3+Z1+2*Z2)))
    # where Z1 = 1 + (1-a^2)^(1/3) * ((1+a)^(1/3) + (1-a)^(1/3))
    #       Z2 = sqrt(3*a^2 + Z1^2)
    a = 0.998
    Z1 = 1 + (1 - a**2)**(1.0 / 3.0) * ((1 + a)**(1.0 / 3.0) + (1 - a)**(1.0 / 3.0))
    Z2 = np.sqrt(3 * a**2 + Z1**2)
    r_isco_kerr = 3 + Z2 - np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))  # prograde

    # Orbital velocity at Kerr ISCO:
    # v = (r^2 - 2*M*r + a*sqrt(M*r)) / (r * sqrt(r^2 - 3*M*r + 2*a*sqrt(M*r)))
    # Simplified: for near-extremal a, v -> c
    r_k = r_isco_kerr
    v_kerr_num = (r_k**2 - 2 * r_k + a * np.sqrt(r_k))
    v_kerr_den = r_k * np.sqrt(r_k**2 - 3 * r_k + 2 * a * np.sqrt(r_k))
    if v_kerr_den > 0:
        v_isco_kerr = abs(v_kerr_num / v_kerr_den)
    else:
        v_isco_kerr = 0.999  # near-extremal

    # Lorentz factor for Kerr ISCO
    W_isco_kerr = 1.0 / np.sqrt(max(1.0 - v_isco_kerr**2, 1e-10))

    print(f"  Kerr Black Hole (a = {a}):")
    print(f"    ISCO radius: r_ISCO = {r_isco_kerr:.4f} GM/c^2")
    print(f"    (Compare to Schwarzschild: 6.0 GM/c^2)")
    print(f"    Orbital velocity at ISCO: v/c = {min(v_isco_kerr, 0.999):.4f}")
    print(f"    Lorentz factor W = {min(W_isco_kerr, 100):.2f}")
    print(f"    Highly relativistic! The ISCO is very close to the horizon.")
    print()
    print(f"  Summary:")
    print(f"    Schwarzschild: r_ISCO = 6M, v = {v_isco_sch:.3f}c, W = {W_isco_sch:.3f}")
    print(f"    Kerr a=0.998: r_ISCO = {r_isco_kerr:.3f}M, v = {min(v_isco_kerr, 0.999):.3f}c, W = {min(W_isco_kerr, 100):.1f}")
    print(f"    Frame dragging in Kerr allows orbits much closer to the horizon.")


def exercise_7():
    """Alfven Speed Saturation.

    Plot the relativistic Alfven speed vs B for fixed rho and p.
    Show that v_A -> c as B -> infinity but never exceeds c.
    """
    rho = 1.0
    p = 0.1
    Gamma_eos = 4.0 / 3.0  # ultra-relativistic EOS

    # Specific enthalpy
    eps = p / ((Gamma_eos - 1) * rho)
    h = 1.0 + eps + p / rho

    B_range = np.logspace(-2, 3, 500)

    # Non-relativistic Alfven speed: v_A = B / sqrt(4*pi*rho) (in Gaussian units)
    # Relativistic: v_A = B / sqrt(4*pi*(rho*h + B^2/(4*pi)))
    # In normalized units (4*pi = 1): v_A = B / sqrt(rho*h + B^2)
    # This is bounded by c = 1

    v_A_rel = B_range / np.sqrt(rho * h + B_range**2)
    v_A_nonrel = B_range / np.sqrt(rho * h)  # non-relativistic (can exceed c!)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(B_range, v_A_rel, 'b-', linewidth=2.5, label=r'Relativistic $v_A$')
    ax.loglog(B_range, np.minimum(v_A_nonrel, 10), 'r--', linewidth=1.5,
              label=r'Non-relativistic $v_A = B/\sqrt{\rho h}$')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=2, label='c = 1 (speed of light)')

    ax.set_xlabel('B (normalized)', fontsize=12)
    ax.set_ylabel(r'$v_A / c$', fontsize=12)
    ax.set_title(r'Relativistic Alfv\'en Speed Saturation', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-3, 5)
    ax.set_xlim(B_range[0], B_range[-1])

    plt.tight_layout()
    plt.savefig('16_alfven_saturation.png', dpi=150)
    plt.close()

    print(f"  Parameters: rho = {rho}, p = {p}, Gamma = {Gamma_eos}")
    print(f"  h = 1 + eps + p/rho = {h:.4f}")
    print()
    print(f"  Relativistic Alfven speed: v_A = B / sqrt(rho*h + B^2)")
    print(f"  Non-relativistic: v_A = B / sqrt(rho*h)")
    print()
    print(f"  Limiting behavior:")
    print(f"    B << sqrt(rho*h): v_A ~ B/sqrt(rho*h) (non-relativistic)")
    print(f"    B >> sqrt(rho*h): v_A ~ 1 - rho*h/(2*B^2) -> 1 (approaches c)")
    print(f"    v_A NEVER exceeds c (causality preserved!)")
    print()

    # Print table
    print(f"  {'B':>10s}  {'v_A (rel)':>10s}  {'v_A (non-rel)':>14s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*14}")
    for B_test in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        va_r = B_test / np.sqrt(rho * h + B_test**2)
        va_nr = B_test / np.sqrt(rho * h)
        print(f"  {B_test:10.2f}  {va_r:10.6f}  {va_nr:14.4f}")

    print("  Plot saved to 16_alfven_saturation.png")


def exercise_8():
    """HARM Timestep.

    In GRMHD near a black hole, the lapse function alpha -> 0.
    Why does this require smaller timesteps? Estimate CFL timestep
    near r = 2.01 GM/c^2.
    """
    # In 3+1 GRMHD, the equations are written with the lapse alpha and shift beta^i
    # The coordinate speed of light is c_coord = alpha (in the radial direction, simplified)
    # As alpha -> 0 near the horizon, the coordinate speed of signals -> 0
    # BUT: the grid spacing in r is finite, so dt ~ alpha * dr (from CFL)
    # This means dt -> 0 near the horizon!

    # Schwarzschild metric: ds^2 = -(1-2M/r)*dt^2 + (1-2M/r)^(-1)*dr^2 + r^2*dOmega^2
    # Lapse: alpha = sqrt(1 - 2M/r) (in Schwarzschild coordinates)

    # At r = 2.01 GM/c^2 (just outside horizon at r=2M):
    r = 2.01  # in units of GM/c^2 = M
    dr = 0.01  # grid spacing (in units of M)

    # Schwarzschild lapse
    alpha = np.sqrt(1.0 - 2.0 / r) if r > 2.0 else 0.0

    # CFL timestep: dt < CFL * dr * alpha / c_max
    # where c_max = 1 (speed of light in natural units)
    CFL = 0.5
    dt = CFL * dr * alpha  # normalized (c=1, M=1)

    print(f"  GRMHD near black hole horizon:")
    print(f"    r = {r} M (horizon at r = 2M for Schwarzschild)")
    print(f"    dr = {dr} M")
    print(f"    Lapse: alpha = sqrt(1 - 2M/r) = sqrt(1 - {2.0 / r:.4f}) = {alpha:.6f}")
    print()
    print(f"  Why alpha -> 0 requires smaller timesteps:")
    print(f"  - The lapse alpha measures the ratio of proper time to coordinate time")
    print(f"  - Near the horizon, proper time passes very slowly relative to coordinates")
    print(f"  - The coordinate speed of signals ~ alpha * c -> 0")
    print(f"  - CFL condition: dt < CFL * dr / (signal_speed) = CFL * dr * alpha")
    print(f"  - As alpha -> 0, dt -> 0 => extremely small timesteps!")
    print()
    print(f"  CFL timestep estimate:")
    print(f"    dt = CFL * dr * alpha = {CFL} * {dr} * {alpha:.6f} = {dt:.8f} M")
    print(f"    In physical units (M = M_sun): dt = {dt * 5e-6:.3e} s")
    print()
    print(f"  Solutions to this problem:")
    print(f"  1. Use horizon-penetrating coordinates (Kerr-Schild)")
    print(f"     - alpha remains finite at the horizon")
    print(f"     - dt is set by the bulk flow, not horizon geometry")
    print(f"  2. Use logarithmic radial grids (dr increases with r)")
    print(f"  3. Excise the region inside the horizon (causality!)")
    print(f"  4. HARM code uses modified Kerr-Schild coordinates")

    # Plot lapse vs radius
    r_range = np.linspace(2.001, 20, 500)
    alpha_range = np.sqrt(1.0 - 2.0 / r_range)
    dt_range = CFL * dr * alpha_range

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(r_range, alpha_range, 'b-', linewidth=2)
    ax1.axvline(2.0, color='red', linestyle='--', label='Horizon (r = 2M)')
    ax1.set_xlabel('r / M', fontsize=12)
    ax1.set_ylabel(r'Lapse $\alpha$', fontsize=12)
    ax1.set_title('Schwarzschild Lapse Function', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(r_range, dt_range, 'b-', linewidth=2)
    ax2.axvline(2.0, color='red', linestyle='--', label='Horizon')
    ax2.set_xlabel('r / M', fontsize=12)
    ax2.set_ylabel(r'$\Delta t$ (CFL timestep, units of M)', fontsize=12)
    ax2.set_title('CFL Timestep Near Horizon', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('16_harm_timestep.png', dpi=150)
    plt.close()
    print("  Plot saved to 16_harm_timestep.png")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Lorentz Transformations", exercise_1),
        ("Exercise 2: Primitive Recovery", exercise_2),
        ("Exercise 3: Relativistic Brio-Wu (B_x = 0)", exercise_3),
        ("Exercise 4: Magnetization Parameter", exercise_4),
        ("Exercise 5: Light Cylinder", exercise_5),
        ("Exercise 6: ISCO Velocity", exercise_6),
        ("Exercise 7: Alfven Speed Saturation", exercise_7),
        ("Exercise 8: HARM Timestep", exercise_8),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
        func()
