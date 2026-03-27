"""
Exercises for Lesson 04: Current-Driven Instabilities
Topic: MHD
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: External Kink Stability

    Cylindrical Z-pinch: a=0.1 m, I=500 kA, rho=1e-6 kg/m^3, no Bz
    """
    a = 0.1       # m
    I = 500e3     # A
    rho = 1e-6    # kg/m^3
    mu0 = 4 * np.pi * 1e-7

    # (a) Azimuthal field at the edge
    B_theta = mu0 * I / (2 * np.pi * a)
    print(f"(a) B_theta(a) = mu0*I/(2*pi*a) = {B_theta:.4f} T")

    # (b) Alfven speed
    v_A = B_theta / np.sqrt(mu0 * rho)
    print(f"\n(b) v_A = B_theta/sqrt(mu0*rho) = {v_A:.4e} m/s")

    # (c) Safety factor with no Bz
    # q(a) = r*Bz/(R0*B_theta) -> 0 if Bz=0
    print("\n(c) With Bz=0: q(a) = a*Bz/(R0*B_theta) = 0")
    print("    (q is zero since there is no axial field)")
    q_a = 0.0

    # (d) Stability check
    print("\n(d) Kruskal-Shafranov criterion: stable if q(a) > 1")
    print(f"    q(a) = {q_a} < 1 => UNSTABLE to external kink")

    # (e) Growth rate
    # gamma = (B_theta / sqrt(mu0*rho)) * sqrt(1 - q^2)
    gamma = (B_theta / np.sqrt(mu0 * rho)) * np.sqrt(1 - q_a**2)
    print(f"\n(e) gamma = v_A * sqrt(1 - q^2) = {gamma:.4e} s^-1")
    tau = 1.0 / gamma
    print(f"    Growth time tau = 1/gamma = {tau:.4e} s")

    # (f) Minimum Bz for stabilization
    # q(a) = a*Bz/(R0*B_theta) > 1 (cylindrical approx: use q = 2*pi*a*Bz/(mu0*I/a))
    # For a cylinder of length 2*pi*R0, q = 2*pi*a*Bz/(L_z * B_theta)
    # With toroidal identification L_z = 2*pi*R0:
    # q = a*Bz / (R0*B_theta) > 1 => Bz > R0*B_theta/a
    # For a pure cylinder, the simpler condition is Bz > B_theta for sausage stability
    # For kink (m=1): Bz > B_theta(a) is sufficient in a simple pinch
    Bz_min = B_theta
    print(f"\n(f) For kink stabilization (q > 1), need Bz > B_theta(a)")
    print(f"    Minimum Bz = B_theta(a) = {Bz_min:.4f} T")
    print("    (In a toroidal geometry q = a*Bz/(R0*B_theta),")
    print("     so Bz_min depends on aspect ratio R0/a)")


def exercise_2():
    """
    Problem 2: Sawtooth Period

    T0(0) = 5 keV, P = 10 MW, V_c ~ 1 m^3, n = 1e20 m^-3
    Crash at T0 = 6 keV, drops to 4 keV
    """
    T0_init = 5.0    # keV
    T_crash = 6.0     # keV
    T_after = 4.0     # keV
    P = 10e6           # W
    V_c = 1.0          # m^3
    n = 1e20           # m^-3
    kB_eV = 1.602e-19  # J per eV
    kB_keV = kB_eV * 1e3  # J per keV

    # (a) Heating rate dT0/dt = P / (3 * n * V_c * kB)
    # Factor 3 = 2 species * 3/2 degrees of freedom
    dTdt = P / (3 * n * V_c * kB_keV)  # keV/s
    print(f"(a) Heating rate: dT0/dt = P/(3*n*V_c*kB)")
    print(f"    = {P:.2e} / (3 * {n:.2e} * {V_c} * {kB_keV:.4e})")
    print(f"    = {dTdt:.4f} keV/s")

    # (b) Time to crash (from T_after to T_crash)
    # Sawtooth ramp: from T_after (4 keV) to T_crash (6 keV)
    delta_T_ramp = T_crash - T_after
    tau_ramp = delta_T_ramp / dTdt
    print(f"\n(b) Temperature rise: {T_after} -> {T_crash} keV, Delta T = {delta_T_ramp} keV")
    print(f"    Time to crash: tau_ramp = Delta T / (dT/dt) = {tau_ramp:.4f} s")
    print(f"    = {tau_ramp*1e3:.2f} ms")

    # (c) Sawtooth period (same as ramp time since crash is nearly instantaneous)
    print(f"\n(c) Sawtooth period ~ tau_ramp = {tau_ramp:.4f} s = {tau_ramp*1e3:.2f} ms")
    print("    (Crash is essentially instantaneous compared to ramp)")

    # (d) Energy redistributed per crash
    delta_T_crash = T_crash - T_after
    delta_E = 3 * n * V_c * kB_keV * delta_T_crash
    print(f"\n(d) Energy per crash: Delta E = 3*n*V_c*kB*Delta T")
    print(f"    = 3 * {n:.2e} * {V_c} * {kB_keV:.4e} * {delta_T_crash}")
    print(f"    = {delta_E:.4e} J = {delta_E/1e3:.2f} kJ")

    # (e) Compare crash timescale to ramp
    tau_crash = 1e-6  # ~1 microsecond (Alfven time)
    ratio = tau_ramp / tau_crash
    print(f"\n(e) Crash timescale (Alfven time): tau_A ~ {tau_crash:.1e} s")
    print(f"    Ramp timescale: {tau_ramp:.4f} s")
    print(f"    Ratio tau_ramp/tau_crash = {ratio:.2e}")
    print("    The crash is ~10^6 times faster than the ramp,")
    print("    justifying treating it as instantaneous.")


def exercise_3():
    """
    Problem 3: Tearing Mode Delta'

    J_z(r) = J0*(1 - r^2/a^2), q(r) = q0*(1 + (r/a)^2)/(1 - r^2/a^2)
    Find r_s where q=2, compute Delta'
    """
    q0 = 1.0
    a = 0.5   # m
    eta = 1e-7  # Ohm.m
    mu0 = 4 * np.pi * 1e-7

    # (a) Find r_s where q(r_s) = 2
    # q(r) = q0*(1 + (r/a)^2) / (1 - r^2/a^2)
    # Note: this q profile diverges at r=a (edge), so solution is in (0, a)
    # Set q = 2: 2 = q0*(1 + x^2)/(1 - x^2) where x = r/a
    # 2*(1 - x^2) = 1 + x^2
    # 2 - 2x^2 = 1 + x^2
    # 1 = 3x^2
    # x^2 = 1/3 => x = 1/sqrt(3)
    x_s = 1.0 / np.sqrt(3)
    r_s = x_s * a
    print(f"(a) Solving q(r_s) = 2 with q0 = {q0}:")
    print(f"    2*(1 - x^2) = 1 + x^2 => x^2 = 1/3 => x = 1/sqrt(3)")
    print(f"    r_s = a/sqrt(3) = {r_s:.6f} m")

    # Verify
    q_check = q0 * (1 + x_s**2) / (1 - x_s**2)
    print(f"    Verification: q(r_s) = {q_check:.4f}")

    # (b) Compute q'(r_s) and q''(r_s) numerically
    def q_profile(r):
        x = r / a
        return q0 * (1 + x**2) / (1 - x**2)

    dr = 1e-6
    q_prime = (q_profile(r_s + dr) - q_profile(r_s - dr)) / (2 * dr)
    q_double_prime = (q_profile(r_s + dr) - 2 * q_profile(r_s) + q_profile(r_s - dr)) / dr**2
    print(f"\n(b) Numerical derivatives at r_s:")
    print(f"    q'(r_s) = {q_prime:.6f} m^-1")
    print(f"    q''(r_s) = {q_double_prime:.6f} m^-2")

    # Analytical: q(x) = q0*(1+x^2)/(1-x^2), x = r/a
    # dq/dr = (1/a)*dq/dx
    # dq/dx = q0 * [2x*(1-x^2) + 2x*(1+x^2)] / (1-x^2)^2 = q0 * 4x / (1-x^2)^2
    dqdx_analytical = q0 * 4 * x_s / (1 - x_s**2)**2
    q_prime_analytical = dqdx_analytical / a
    print(f"    Analytical: q'(r_s) = {q_prime_analytical:.6f} m^-1 (check)")

    # (c) Estimate Delta'
    Delta_prime = 2.0 / r_s + q_double_prime / q_prime
    print(f"\n(c) Delta' ~ 2/r_s + q''/q'")
    print(f"    = 2/{r_s:.6f} + {q_double_prime:.6f}/{q_prime:.6f}")
    print(f"    = {2.0/r_s:.4f} + {q_double_prime/q_prime:.4f}")
    print(f"    = {Delta_prime:.4f} m^-1")

    # (d) Stability
    print(f"\n(d) m=2, n=1 tearing mode:")
    if Delta_prime > 0:
        print(f"    Delta' = {Delta_prime:.4f} > 0 => UNSTABLE")
    else:
        print(f"    Delta' = {Delta_prime:.4f} < 0 => STABLE")

    # (e) Estimate growth rate
    # gamma ~ eta^(3/5) * (Delta')^(4/5) * (v_A/a) * S^(-3/5) (approximate)
    # More precisely: gamma = (eta/(mu0*a^2))^(3/5) * (v_A/a)^(2/5) * (Delta'*a)^(4/5) / a
    # Simplified: gamma ~ (eta * Delta'^4 / (mu0^3 * a^2))^(1/5) * v_A^(2/5)
    # Using typical scaling: gamma ~ (eta/mu0)^(3/5) * (Delta')^(4/5) / (a * tau_A^(2/5))
    # Let's use the standard constant-psi result
    tau_R = mu0 * a**2 / eta  # resistive time
    # Assume typical Alfven speed for tokamak
    B_typical = 1.0  # T
    rho_typical = 1e-6  # kg/m^3 (for illustration)
    v_A = B_typical / np.sqrt(mu0 * rho_typical)
    tau_A = a / v_A

    # Constant-psi growth rate: gamma * tau_A = (tau_A/tau_R)^(3/5) * (Delta'*a)^(4/5)
    gamma_est = (tau_A / tau_R)**(3.0/5.0) * (abs(Delta_prime) * a)**(4.0/5.0) / tau_A
    print(f"\n(e) Using eta = {eta} Ohm.m, a = {a} m:")
    print(f"    tau_R = mu0*a^2/eta = {tau_R:.4e} s")
    print(f"    tau_A = a/v_A = {tau_A:.4e} s (assuming B={B_typical} T, rho={rho_typical} kg/m^3)")
    print(f"    gamma ~ (tau_A/tau_R)^(3/5) * (Delta'*a)^(4/5) / tau_A")
    print(f"    = {gamma_est:.4e} s^-1")

    # Plot q profile
    r_vals = np.linspace(0.01 * a, 0.95 * a, 200)
    q_vals = [q_profile(r) for r in r_vals]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r_vals / a, q_vals, 'b-', linewidth=2, label='q(r)')
    ax.axhline(y=2.0, color='r', linestyle='--', label='q = 2 (resonant)')
    ax.axvline(x=r_s / a, color='g', linestyle=':', label=f'r_s/a = {r_s/a:.4f}')
    ax.set_xlabel('r/a', fontsize=12)
    ax.set_ylabel('q(r)', fontsize=12)
    ax.set_title('Safety Factor Profile and Resonant Surface', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 8)
    plt.tight_layout()
    plt.savefig('/tmp/ex04_q_profile.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex04_q_profile.png")


def exercise_4():
    """
    Problem 4: Magnetic Island Width

    delta_psi = 1e-3 Wb, r_s = 0.3 m, B_theta(r_s) = 0.4 T, m=2
    """
    delta_psi = 1e-3  # Wb
    r_s = 0.3   # m
    B_theta = 0.4  # T
    m = 2
    dpdr = -1e6  # Pa/m

    # (a) Island width
    w = 4 * np.sqrt(delta_psi * r_s / (m * B_theta))
    print(f"(a) Island width: w = 4*sqrt(delta_psi * r_s / (m * B_theta))")
    print(f"    = 4*sqrt({delta_psi:.1e} * {r_s} / ({m} * {B_theta}))")
    print(f"    = {w:.6f} m = {w*100:.3f} cm")

    # (b) If w = 5 cm, find new delta_psi
    w_new = 0.05  # 5 cm
    # w = 4*sqrt(delta_psi * r_s / (m * B_theta))
    # delta_psi = (w/4)^2 * m * B_theta / r_s
    delta_psi_new = (w_new / 4)**2 * m * B_theta / r_s
    print(f"\n(b) For w = {w_new*100:.0f} cm:")
    print(f"    delta_psi = (w/4)^2 * m * B_theta / r_s")
    print(f"    = ({w_new}/4)^2 * {m} * {B_theta} / {r_s}")
    print(f"    = {delta_psi_new:.6e} Wb")

    # (c) Flattened region
    print(f"\n(c) Flattened region: +/- w/2 around r_s")
    print(f"    With w = {w_new*100:.0f} cm: from r = {r_s - w_new/2:.4f} m to r = {r_s + w_new/2:.4f} m")
    print(f"    Width of flattened region: {w_new*100:.0f} cm")

    # (d) Pressure gradient lost
    # Over the island width, the pressure profile is flattened
    # The gradient that would have existed is dp/dr * w
    delta_p = abs(dpdr) * w_new
    print(f"\n(d) Pressure gradient lost in island region:")
    print(f"    dp/dr = {dpdr:.1e} Pa/m")
    print(f"    Pressure drop that would occur over w: |dp/dr| * w = {delta_p:.2e} Pa")
    print(f"    = {delta_p/1e3:.2f} kPa")
    print(f"    This gradient is flattened to zero inside the island.")

    # (e) Impact on bootstrap current and NTM
    print(f"\n(e) Impact on bootstrap current and NTM drive:")
    print(f"    - Bootstrap current J_bs ~ dp/dr. Flattening the pressure gradient")
    print(f"      inside the island eliminates the local bootstrap current.")
    print(f"    - Missing bootstrap current creates a helical perturbation that")
    print(f"      reinforces the original island (positive feedback).")
    print(f"    - This is the drive mechanism for NTM: once an island exceeds a")
    print(f"      threshold width w_d, the bootstrap deficit destabilizes it further.")
    print(f"    - Larger islands flatten more pressure -> more bootstrap deficit")
    print(f"      -> stronger drive (until saturation).")


def exercise_5():
    """
    Problem 5: Resistive Wall Mode

    a=1 m, r_w=1.2 m, d=5 cm, sigma=5e7 S/m (copper)
    """
    a = 1.0        # m (plasma minor radius)
    r_w = 1.2      # m (wall radius)
    d = 0.05       # m (wall thickness)
    sigma = 5e7    # S/m (copper)
    mu0 = 4 * np.pi * 1e-7

    # (a) Wall time constant
    tau_w = mu0 * sigma * d * r_w
    print(f"(a) Wall time constant: tau_w = mu0 * sigma * d * r_w")
    print(f"    = {mu0:.4e} * {sigma:.1e} * {d} * {r_w}")
    print(f"    = {tau_w:.4f} s")
    print(f"    = {tau_w*1e3:.2f} ms")

    # (b) RWM growth rate
    gamma_RWM = (1.0 / tau_w) * (r_w - a) / r_w
    print(f"\n(b) RWM growth rate: gamma = (1/tau_w) * (r_w - a)/r_w")
    print(f"    = (1/{tau_w:.4f}) * ({r_w} - {a})/{r_w}")
    print(f"    = {gamma_RWM:.4f} s^-1")
    tau_growth = 1.0 / gamma_RWM
    print(f"    Growth time: 1/gamma = {tau_growth:.4f} s = {tau_growth*1e3:.2f} ms")

    # (c) Feedback stabilization
    tau_fb = 10e-3  # 10 ms
    print(f"\n(c) Feedback response time: tau_fb = {tau_fb*1e3:.0f} ms")
    print(f"    RWM growth time: {tau_growth*1e3:.2f} ms")
    if tau_fb < tau_growth:
        print(f"    tau_fb ({tau_fb*1e3:.0f} ms) < tau_growth ({tau_growth*1e3:.2f} ms)")
        print(f"    YES: Feedback CAN stabilize the RWM (feedback is faster)")
    else:
        print(f"    tau_fb ({tau_fb*1e3:.0f} ms) > tau_growth ({tau_growth*1e3:.2f} ms)")
        print(f"    NO: Feedback CANNOT stabilize the RWM (too slow)")

    # (d) Required bandwidth
    f_required = gamma_RWM / (2 * np.pi)
    print(f"\n(d) Required feedback bandwidth:")
    print(f"    f > gamma/(2*pi) = {f_required:.4f} Hz")
    print(f"    Bandwidth must exceed {f_required:.2f} Hz")
    print(f"    In practice, a safety margin of 5-10x is needed:")
    print(f"    Recommended bandwidth: > {5*f_required:.1f} Hz")

    # (e) Ideal wall
    print(f"\n(e) With ideal (perfectly conducting) wall:")
    print(f"    Image currents never decay (tau_w -> infinity)")
    print(f"    The external kink is completely stabilized")
    print(f"    gamma_RWM -> 0 (no growth)")
    print(f"    This allows operation above the no-wall beta limit")
    print(f"    but below the ideal-wall beta limit.")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: External Kink Stability ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Sawtooth Period ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Tearing Mode Delta' ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Magnetic Island Width ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Resistive Wall Mode ===")
    print("=" * 60)
    exercise_5()

    print("\nAll exercises completed!")
