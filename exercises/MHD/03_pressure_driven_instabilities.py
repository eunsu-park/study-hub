"""
Exercises for Lesson 03: Pressure-Driven Instabilities
Topic: MHD
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: Interchange Stability in a Mirror

    B(z) = B0*(1 + z^2/L^2), uniform pressure p = p0
    """
    B0 = 1.0
    L = 1.0
    p0 = 1e4

    print("(a) Curvature kappa = b . grad(b), where b = B/|B|:")
    print("    B = B0*(1 + z^2/L^2) z_hat (approximately, for paraxial field)")
    print("    The curvature is related to d^2B/dz^2 evaluated on axis.")
    print("    For a mirror field B(z) = B0(1 + z^2/L^2):")
    print("    The field has a minimum at z=0 and increases outward.")
    print("    Curvature kappa ~ (1/B)(d^2B/dz^2) = 2/(L^2*B0) at z=0")
    kappa_0 = 2.0 / (L**2 * B0)
    print(f"    kappa(z=0) ~ {kappa_0:.4f} m^-1")
    print("    The curvature points outward (away from minimum B).")

    print("\n(b) kappa . grad(p):")
    print("    Since p = p0 = const, grad(p) = 0")
    print("    Therefore kappa . grad(p) = 0")

    print("\n(c) With uniform pressure, kappa . grad(p) = 0,")
    print("    so the interchange criterion is marginally satisfied.")
    print("    The configuration is STABLE to interchange (no pressure drive).")

    print("\n(d) With p(z) = p0 * exp(-z^2/L_p^2):")
    print("    dp/dz = -2*p0*z/L_p^2 * exp(-z^2/L_p^2)")
    print("    At z != 0: grad(p) points toward z=0 (decreasing |z|)")
    print("    Curvature kappa points outward (increasing |z|)")
    print("    If kappa . grad(p) > 0 somewhere, the system is UNSTABLE")
    print("    to interchange at those locations.")
    print()
    print("    At z > 0: kappa ~ +z direction, grad(p) ~ -z direction")
    print("    => kappa . grad(p) < 0 (favorable curvature, STABLE)")
    print()
    print("    At the midplane z=0: grad(p) = 0, marginally stable.")
    print()
    print("    For a minimum-B mirror, the pressure gradient is generally")
    print("    favorable for confinement. This makes minimum-B mirrors")
    print("    inherently MHD-stable (unlike simple mirrors).")


def exercise_2():
    """
    Problem 2: RT Critical Wavelength

    rho2 = 1e-6 kg/m^3 (plasma on top), rho1 ~ 0 (vacuum below)
    B0 = 1 T, g_eff = 1e4 m/s^2
    """
    rho2 = 1e-6
    rho1 = 0.0
    B0 = 1.0
    g_eff = 1e4
    mu0 = 4 * np.pi * 1e-7

    # (a) Critical wavenumber
    # k_c = g_eff * (rho2 - rho1) * mu0 / B0^2
    k_c = g_eff * (rho2 - rho1) * mu0 / B0**2
    print(f"(a) Critical wavenumber: k_c = {k_c:.4e} m^-1")

    # (b) Critical wavelength
    lambda_c = 2 * np.pi / k_c
    print(f"\n(b) Critical wavelength: lambda_c = 2*pi/k_c = {lambda_c:.4f} m")

    # (c) Growth rate at k = 0.5 * k_c (with k perpendicular to B, kx=0)
    k = 0.5 * k_c
    # For k perp to B (kx = 0), magnetic field provides no stabilization
    A = (rho2 - rho1) / (rho2 + rho1)  # Atwood number (=1 for vacuum below)
    gamma = np.sqrt(g_eff * k * A)
    print(f"\n(c) At k = 0.5*k_c = {k:.4e} m^-1 (k perp B):")
    print(f"    Atwood number A = {A:.4f}")
    print(f"    Growth rate: gamma = {gamma:.4e} s^-1")

    # (d) Growth time
    tau = 1.0 / gamma
    print(f"\n(d) Growth time: tau = 1/gamma = {tau:.4e} s")

    # (e) Compare to Alfven time
    v_A = B0 / np.sqrt(mu0 * rho2)
    tau_A = lambda_c / v_A
    print(f"\n(e) Alfven speed: v_A = {v_A:.4e} m/s")
    print(f"    Alfven time: tau_A = lambda_c/v_A = {tau_A:.4e} s")
    print(f"    Ratio tau/tau_A = {tau/tau_A:.4f}")
    if tau < tau_A:
        print("    Growth is FASTER than Alfven time (strong instability)")
    else:
        print("    Growth is SLOWER than Alfven time")


def exercise_3():
    """
    Problem 3: Ballooning Stability Boundary

    s = 2.5 at mid-radius
    """
    s = 2.5

    # (a) Critical pressure parameter
    alpha_c = 0.6 * s
    print(f"(a) Critical alpha_c = 0.6 * s = 0.6 * {s} = {alpha_c:.2f}")

    # (b) With alpha = 2.0
    alpha = 2.0
    print(f"\n(b) Actual alpha = {alpha}")
    if alpha < alpha_c:
        print(f"    STABLE: alpha ({alpha}) < alpha_c ({alpha_c})")
    else:
        print(f"    UNSTABLE: alpha ({alpha}) > alpha_c ({alpha_c})")

    # (c) Factor to reduce pressure gradient
    if alpha > alpha_c:
        reduction_factor = alpha_c / alpha
        print(f"\n(c) Reduction factor: alpha_c/alpha = {reduction_factor:.4f}")
        print(f"    Pressure gradient must be reduced to {reduction_factor*100:.1f}% of current value")
    else:
        print(f"\n(c) No reduction needed (already stable)")

    # (d) Increase s to 4.0
    s_new = 4.0
    alpha_c_new = 0.6 * s_new
    print(f"\n(d) With s = {s_new}: alpha_c = {alpha_c_new:.2f}")
    if alpha < alpha_c_new:
        print(f"    STABLE: alpha ({alpha}) < alpha_c ({alpha_c_new})")
    else:
        print(f"    UNSTABLE: alpha ({alpha}) > alpha_c ({alpha_c_new})")

    # (e) Methods to increase s
    print("\n(e) Experimental methods to increase magnetic shear s:")
    print("    1. Current profile control: Use LHCD (Lower Hybrid Current Drive)")
    print("       or ECCD (Electron Cyclotron Current Drive) to modify the")
    print("       radial current profile, increasing dq/dr (hence s = (r/q)(dq/dr)).")
    print("    2. Plasma shaping: Elongation and triangularity of the plasma")
    print("       cross-section modify the magnetic geometry, effectively")
    print("       increasing the shear at the edge. Strong elongation")
    print("       increases edge shear through geometric effects.")

    # Plot s-alpha diagram
    s_vals = np.linspace(0, 6, 100)
    alpha_c_vals = 0.6 * s_vals

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(s_vals, 0, alpha_c_vals, alpha=0.3, color='green', label='STABLE')
    ax.fill_between(s_vals, alpha_c_vals, 5, alpha=0.3, color='red', label='UNSTABLE')
    ax.plot(s_vals, alpha_c_vals, 'b-', linewidth=2, label='Marginal stability')
    ax.plot(s, alpha, 'ko', markersize=10, label=f's={s}, alpha={alpha}')
    ax.plot(s_new, alpha, 's', color='orange', markersize=10,
            label=f's={s_new}, alpha={alpha}')
    ax.set_xlabel('Magnetic shear s', fontsize=12)
    ax.set_ylabel('Pressure parameter alpha', fontsize=12)
    ax.set_title('Ballooning Stability (s-alpha diagram)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    plt.tight_layout()
    plt.savefig('/tmp/ex03_ballooning.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex03_ballooning.png")


def exercise_4():
    """
    Problem 4: Parker Instability in Galactic Disk

    rho0 = 1e-24 kg/m^3, T = 1e4 K, B = 5e-10 T
    H = 100 pc = 3e18 m
    """
    rho0 = 1e-24
    T = 1e4
    B = 5e-10
    H = 3e18
    mu0 = 4 * np.pi * 1e-7
    kB = 1.38e-23
    mp = 1.67e-27
    gamma_ad = 5.0 / 3.0
    year = 3.156e7  # seconds in a year

    # (a) Gas pressure
    n = rho0 / mp
    p = n * kB * T
    print(f"(a) Number density: n = rho0/mp = {n:.2e} m^-3")
    print(f"    Gas pressure: p = nkT = {p:.2e} Pa")

    # (b) Plasma beta
    beta = 2 * mu0 * p / B**2
    print(f"\n(b) Plasma beta: beta = 2*mu0*p/B^2 = {beta:.4f}")

    # (c) Parker instability condition
    beta_crit = 2.0 / gamma_ad
    print(f"\n(c) Parker instability condition: beta > 2/gamma = {beta_crit:.4f}")
    if beta > beta_crit:
        print(f"    UNSTABLE: beta ({beta:.4f}) > {beta_crit:.4f}")
    else:
        print(f"    STABLE: beta ({beta:.4f}) < {beta_crit:.4f}")

    # (d) Growth rate
    g_eff = kB * T / (mp * H)  # effective gravity
    gamma_parker = np.sqrt(g_eff / H)
    print(f"\n(d) Effective gravity: g_eff ~ kT/(mp*H) = {g_eff:.4e} m/s^2")
    print(f"    Growth rate: gamma ~ sqrt(g/H) = {gamma_parker:.4e} s^-1")

    # (e) Growth time in years
    tau = 1.0 / gamma_parker
    tau_years = tau / year
    tau_Myr = tau_years / 1e6
    print(f"\n(e) Growth time: tau = {tau:.4e} s = {tau_years:.4e} years")
    print(f"    = {tau_Myr:.2f} Myr")
    print("    Observed molecular cloud formation timescale: ~1-10 Myr")
    if 1 < tau_Myr < 100:
        print("    This is roughly consistent with observations.")
    else:
        print("    This is significantly different from observed timescales.")


def exercise_5():
    """
    Problem 5: Mercier Criterion Components

    At r = 0.5a: q=1.5, (r/q)(dq/dr)=1.0
    p = 2e5 Pa, dp/dr = -1e6 Pa/m, B_p=0.4 T, R0=3 m
    """
    q = 1.5
    rq_dqdr = 1.0  # (r/q)(dq/dr)
    p = 2e5
    dpdr = -1e6
    Bp = 0.4
    R0 = 3.0
    a = 1.0
    r = 0.5 * a
    mu0 = 4 * np.pi * 1e-7

    # (a) Shear contribution D_S
    D_S = 0.25 * rq_dqdr**2
    print(f"(a) D_S = (1/4)(r/q * dq/dr)^2 = (1/4)*{rq_dqdr}^2 = {D_S:.4f}")

    # (b) Magnetic well contribution D_W
    D_W = (mu0 * r / Bp**2) * dpdr * (1 + 2 * q**2)
    print(f"\n(b) D_W = (mu0*r/Bp^2) * dp/dr * (1 + 2q^2)")
    print(f"    = ({mu0:.4e} * {r} / {Bp}^2) * ({dpdr:.0e}) * (1 + 2*{q}^2)")
    print(f"    = {D_W:.4f}")

    # (c) Geodesic curvature D_G
    D_G = r**2 / (R0**2 * q**2)
    print(f"\n(c) D_G = r^2 / (R0^2 * q^2) = {r}^2 / ({R0}^2 * {q}^2) = {D_G:.4f}")

    # (d) Total D_I
    D_I = D_S + D_W + D_G
    print(f"\n(d) D_I = D_S + D_W + D_G = {D_S:.4f} + {D_W:.4f} + {D_G:.4f} = {D_I:.4f}")

    # (e) Mercier stability check
    D_crit = 0.25
    print(f"\n(e) Mercier criterion: D_I > 0.25")
    print(f"    D_I = {D_I:.4f}, threshold = {D_crit}")
    if D_I > D_crit:
        print(f"    STABLE: D_I ({D_I:.4f}) > {D_crit}")
    else:
        print(f"    UNSTABLE: D_I ({D_I:.4f}) < {D_crit}")

    # (f) Dominant contribution
    print(f"\n(f) Contributions:")
    print(f"    D_S (shear, stabilizing) = {D_S:+.4f}")
    print(f"    D_W (well/pressure)      = {D_W:+.4f}")
    print(f"    D_G (geodesic)           = {D_G:+.4f}")

    contributions = {'D_S (shear)': D_S, 'D_W (pressure)': D_W, 'D_G (geodesic)': D_G}
    max_key = max(contributions, key=lambda k: abs(contributions[k]))
    print(f"\n    Dominant contribution: {max_key} = {contributions[max_key]:+.4f}")
    if D_W < 0:
        print("    D_W is destabilizing (negative) due to pressure gradient.")
        print("    Magnetic shear (D_S) provides the main stabilization.")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Interchange Stability in a Mirror ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: RT Critical Wavelength ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Ballooning Stability Boundary ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Parker Instability in Galactic Disk ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Mercier Criterion Components ===")
    print("=" * 60)
    exercise_5()

    print("\nAll exercises completed!")
