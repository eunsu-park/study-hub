"""
Exercises for Lesson 12: Waveguides and Cavities
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, jn


# Constants
c = 2.998e8
mu_0 = 4.0 * np.pi * 1e-7
epsilon_0 = 8.854e-12


def exercise_1():
    """
    Exercise 1: WR-284 Waveguide
    a = 72.14 mm, b = 34.04 mm. Cutoff frequencies, single-mode range,
    phase/group velocity at 3 GHz.
    """
    a = 72.14e-3   # m
    b = 34.04e-3   # m

    # Cutoff frequencies: f_mn = (c/2)*sqrt((m/a)^2 + (n/b)^2)
    modes = []
    for m in range(4):
        for n in range(4):
            if m == 0 and n == 0:
                continue
            f_c = (c / 2) * np.sqrt((m / a)**2 + (n / b)**2)
            mode_type = 'TE' if (m >= 0 and n >= 0) else ''
            # TM modes require m >= 1 AND n >= 1
            if m >= 1 and n >= 1:
                modes.append((f_c, f'TM{m}{n}'))
            if not (m == 0 and n == 0):
                modes.append((f_c, f'TE{m}{n}'))

    # Remove duplicates and sort
    modes = sorted(set(modes), key=lambda x: x[0])

    print(f"  WR-284 waveguide: a = {a*1e3:.2f} mm, b = {b*1e3:.2f} mm")
    print(f"\n  (a) First 8 modes by cutoff frequency:")
    for i, (f_c, name) in enumerate(modes[:8]):
        print(f"    {i+1}. {name}: f_c = {f_c/1e9:.4f} GHz")

    # (b) Single-mode range: between TE10 and the next mode
    f_TE10 = c / (2 * a)
    f_next = modes[1][0]  # second mode
    print(f"\n  (b) Single-mode range: {f_TE10/1e9:.4f} GHz to {f_next/1e9:.4f} GHz")

    # (c) At 3 GHz for TE10
    f = 3e9
    f_c10 = f_TE10
    if f > f_c10:
        k = 2 * np.pi * f / c
        kc = np.pi / a
        kz = np.sqrt(k**2 - kc**2)
        v_phase = 2 * np.pi * f / kz
        v_group = c**2 / v_phase
        lambda_g = 2 * np.pi / kz
        lambda_0 = c / f

        print(f"\n  (c) TE10 mode at f = {f/1e9:.0f} GHz:")
        print(f"    Phase velocity: v_p = {v_phase/c:.4f} c = {v_phase:.4e} m/s")
        print(f"    Group velocity: v_g = {v_group/c:.4f} c = {v_group:.4e} m/s")
        print(f"    Guide wavelength: lambda_g = {lambda_g*100:.2f} cm")
        print(f"    Free-space wavelength: lambda_0 = {lambda_0*100:.2f} cm")
        print(f"    v_p * v_g = {v_phase*v_group:.4e} (should be c^2 = {c**2:.4e})")


def exercise_2():
    """
    Exercise 2: Microwave Cavity Design
    TE101 mode at 2.45 GHz. Compute Q-factor for copper walls.
    """
    f_target = 2.45e9  # target frequency
    sigma_Cu = 5.96e7  # copper conductivity (S/m)

    # TE101 resonance: f = (c/2)*sqrt((1/a)^2 + (1/d)^2) with b < d < a
    # Choose a = 2*b, d = 1.5*b for aspect ratio constraints
    # f = (c/2)*sqrt(1/a^2 + 1/d^2)
    # Try: let a be the largest dimension
    # f_101 = (c/2)*sqrt((1/a)^2 + (1/d)^2)

    # For a square cavity (a = d): f = c*sqrt(2)/(2*a)
    # a = c*sqrt(2)/(2*f) = ...
    a = c * np.sqrt(2) / (2 * f_target)
    d = a   # square cavity
    b = a / 2   # height

    # Verify
    f_check = (c / 2) * np.sqrt((1 / a)**2 + (1 / d)**2)

    print(f"  Target: TE101 at {f_target/1e9:.2f} GHz")
    print(f"  Cavity dimensions: a = {a*100:.2f} cm, b = {b*100:.2f} cm, d = {d*100:.2f} cm")
    print(f"  Verification: f_101 = {f_check/1e9:.4f} GHz")

    # Q-factor for copper walls
    # Skin depth: delta = sqrt(2/(mu_0*sigma*omega))
    omega = 2 * np.pi * f_target
    delta_s = np.sqrt(2 / (mu_0 * sigma_Cu * omega))

    # Q ~ volume / (delta * surface_area) * factor
    # For TE101 rectangular cavity (approximate):
    # Q ~ (k*a*b*d) / (2*delta_s*(a*b + b*d + a*d))
    # More precisely: Q = omega * U / P_loss
    # For TE101: Q ≈ (k*a*b*d*pi) / (delta_s * ...)
    # Simplified formula: Q ~ (volume * k) / (surface_area * delta)
    V = a * b * d
    S = 2 * (a * b + b * d + a * d)
    k = omega / c
    Q_approx = k * V / (delta_s * S) * 2  # approximate factor

    # Decay time
    tau_decay = Q_approx / omega

    print(f"\n  Copper conductivity: sigma = {sigma_Cu:.2e} S/m")
    print(f"  Skin depth at {f_target/1e9:.2f} GHz: delta = {delta_s*1e6:.2f} um")
    print(f"  Q-factor (approximate): Q = {Q_approx:.0f}")
    print(f"  Decay time: tau = Q/omega = {tau_decay*1e9:.2f} ns")
    print(f"  Energy decays to 1/e in {tau_decay*1e9:.2f} ns")


def exercise_3():
    """
    Exercise 3: Single-Mode Fiber Design
    Step-index fiber at 1310 nm. n1=1.468, n2=1.463.
    """
    lambda_op = 1310e-9  # operating wavelength
    n1 = 1.468           # core
    n2 = 1.463           # cladding

    # (a) Maximum core radius for single-mode: V < 2.405
    # V = (2*pi*a/lambda) * sqrt(n1^2 - n2^2)
    NA = np.sqrt(n1**2 - n2**2)
    a_max = 2.405 * lambda_op / (2 * np.pi * NA)

    # (b) Numerical aperture and acceptance angle
    theta_accept = np.degrees(np.arcsin(NA))

    print(f"  Step-index fiber: n1 = {n1}, n2 = {n2}")
    print(f"  Operating wavelength: {lambda_op*1e9:.0f} nm")
    print()
    print(f"  (a) NA = sqrt(n1^2 - n2^2) = {NA:.6f}")
    print(f"      Max core radius for single-mode (V < 2.405):")
    print(f"      a_max = {a_max*1e6:.2f} um")
    print()
    print(f"  (b) Acceptance angle: theta = arcsin(NA) = {theta_accept:.2f} degrees")

    # (c) Number of modes at 850 nm
    lambda_850 = 850e-9
    a_use = a_max  # use the max radius for 1310 nm
    V_850 = (2 * np.pi * a_use / lambda_850) * NA
    N_modes = int(V_850**2 / 2)

    print(f"\n  (c) At 850 nm with a = {a_use*1e6:.2f} um:")
    print(f"      V number = {V_850:.4f}")
    print(f"      Approximate number of modes: N ~ V^2/2 = {N_modes}")
    print(f"      (No longer single-mode at 850 nm)")


def exercise_4():
    """
    Exercise 4: Mode Visualization
    TE21 and TM21 modes in rectangular waveguide with a = 2b.
    """
    a = 0.04   # width (m)
    b = 0.02   # height (a = 2b)

    N = 100
    x = np.linspace(0, a, N)
    y = np.linspace(0, b, N)
    X, Y = np.meshgrid(x, y)

    m, n = 2, 1

    # TE21 mode: Hz = H0 * cos(m*pi*x/a) * cos(n*pi*y/b)
    # Ex proportional to sin(m*pi*x/a) * cos(n*pi*y/b) * (n*pi/b)
    # Ey proportional to cos(m*pi*x/a) * sin(n*pi*y/b) * (m*pi/a)
    # (Proportionality constants involve kc^2)

    Hz_TE = np.cos(m * np.pi * X / a) * np.cos(n * np.pi * Y / b)

    kc2 = (m * np.pi / a)**2 + (n * np.pi / b)**2
    Ex_TE = (n * np.pi / b) * np.cos(m * np.pi * X / a) * np.sin(n * np.pi * Y / b) / kc2
    Ey_TE = -(m * np.pi / a) * np.sin(m * np.pi * X / a) * np.cos(n * np.pi * Y / b) / kc2

    # TM21 mode: Ez = E0 * sin(m*pi*x/a) * sin(n*pi*y/b)
    Ez_TM = np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)

    Ex_TM = (m * np.pi / a) * np.cos(m * np.pi * X / a) * np.sin(n * np.pi * Y / b) / kc2
    Ey_TM = (n * np.pi / b) * np.sin(m * np.pi * X / a) * np.cos(n * np.pi * Y / b) / kc2

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # TE21 field pattern
    step = 5
    axes[0, 0].quiver(X[::step, ::step] * 100, Y[::step, ::step] * 100,
                       Ex_TE[::step, ::step], Ey_TE[::step, ::step], color='blue')
    axes[0, 0].set_xlabel('x (cm)')
    axes[0, 0].set_ylabel('y (cm)')
    axes[0, 0].set_title('TE21: Transverse E field')
    axes[0, 0].set_aspect('equal')

    cs = axes[0, 1].contourf(X * 100, Y * 100, Hz_TE, levels=20, cmap='RdBu_r')
    plt.colorbar(cs, ax=axes[0, 1])
    axes[0, 1].set_xlabel('x (cm)')
    axes[0, 1].set_ylabel('y (cm)')
    axes[0, 1].set_title('TE21: Hz pattern')
    axes[0, 1].set_aspect('equal')

    # TM21 field pattern
    axes[1, 0].quiver(X[::step, ::step] * 100, Y[::step, ::step] * 100,
                       Ex_TM[::step, ::step], Ey_TM[::step, ::step], color='red')
    axes[1, 0].set_xlabel('x (cm)')
    axes[1, 0].set_ylabel('y (cm)')
    axes[1, 0].set_title('TM21: Transverse E field')
    axes[1, 0].set_aspect('equal')

    cs2 = axes[1, 1].contourf(X * 100, Y * 100, Ez_TM, levels=20, cmap='RdBu_r')
    plt.colorbar(cs2, ax=axes[1, 1])
    axes[1, 1].set_xlabel('x (cm)')
    axes[1, 1].set_ylabel('y (cm)')
    axes[1, 1].set_title('TM21: Ez pattern')
    axes[1, 1].set_aspect('equal')

    plt.suptitle(f'Waveguide Modes (a={a*100:.0f} cm, b={b*100:.0f} cm)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex12_mode_patterns.png', dpi=150)
    plt.close()

    f_c = c / 2 * np.sqrt((m / a)**2 + (n / b)**2)
    print(f"  Rectangular waveguide: a = {a*100:.0f} cm, b = {b*100:.0f} cm")
    print(f"  TE21/TM21 cutoff: {f_c/1e9:.4f} GHz")
    print("  Plot saved: ex12_mode_patterns.png")


def exercise_5():
    """
    Exercise 5: Circular Waveguide Modes
    a = 15 mm. First 6 modes, plot TE11 radial pattern.
    """
    a = 0.015  # radius (15 mm)

    # TE modes: cutoff at j'_mn (derivatives of Bessel zeros)
    # TM modes: cutoff at j_mn (Bessel zeros)
    # f_c = c * x_mn / (2*pi*a) where x_mn is the appropriate zero

    # First zeros of Bessel functions and their derivatives:
    # J0 zeros: 2.4048, 5.5201, 8.6537
    # J1 zeros: 3.8317, 7.0156
    # J2 zeros: 5.1356
    # J0' zeros: 3.8317, 7.0156 (same as J1 zeros by identity)
    # J1' zeros: 1.8412, 5.3314
    # J2' zeros: 3.0542, 6.7061

    modes = []
    # TE modes use j'_mn (zeros of J_m')
    te_zeros = {
        (1, 1): 1.8412,   # TE11 (fundamental, lowest cutoff)
        (2, 1): 3.0542,   # TE21
        (0, 1): 3.8317,   # TE01
        (3, 1): 4.2012,   # TE31
        (1, 2): 5.3314,   # TE12
        (2, 2): 6.7061,   # TE22
    }

    # TM modes use j_mn (zeros of J_m)
    tm_zeros = {
        (0, 1): 2.4048,   # TM01
        (1, 1): 3.8317,   # TM11
        (2, 1): 5.1356,   # TM21
        (0, 2): 5.5201,   # TM02
    }

    for (m_mode, n_mode), x_mn in te_zeros.items():
        f_c = c * x_mn / (2 * np.pi * a)
        modes.append((f_c, f'TE{m_mode}{n_mode}', x_mn))

    for (m_mode, n_mode), x_mn in tm_zeros.items():
        f_c = c * x_mn / (2 * np.pi * a)
        modes.append((f_c, f'TM{m_mode}{n_mode}', x_mn))

    modes.sort(key=lambda x: x[0])

    print(f"  Circular waveguide: a = {a*1e3:.0f} mm")
    print(f"\n  First 6 modes:")
    for i, (f_c, name, x) in enumerate(modes[:6]):
        print(f"    {i+1}. {name}: f_c = {f_c/1e9:.4f} GHz (x = {x:.4f})")

    # Single-mode bandwidth
    bw = modes[1][0] - modes[0][0]
    print(f"\n  Single-mode bandwidth: {bw/1e9:.4f} GHz")

    # Plot TE11 radial field pattern
    r = np.linspace(0, a, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    R_grid, PHI = np.meshgrid(r, phi)

    x_11 = 1.8412  # first zero of J1'
    # TE11: Hz ~ J1(x_11*r/a) * cos(phi)
    Hz = jn(1, x_11 * R_grid / a) * np.cos(PHI)

    X_cart = R_grid * np.cos(PHI)
    Y_cart = R_grid * np.sin(PHI)

    fig, ax = plt.subplots(figsize=(7, 7))
    cs = ax.contourf(X_cart * 1e3, Y_cart * 1e3, Hz, levels=20, cmap='RdBu_r')
    plt.colorbar(cs, ax=ax, label='Hz (arb)')
    circle = plt.Circle((0, 0), a * 1e3, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'TE11 Mode: Hz Pattern (a = {a*1e3:.0f} mm)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('ex12_circular_te11.png', dpi=150)
    plt.close()
    print("  Plot saved: ex12_circular_te11.png")


if __name__ == "__main__":
    print("=== Exercise 1: WR-284 Waveguide ===")
    exercise_1()
    print("\n=== Exercise 2: Microwave Cavity Design ===")
    exercise_2()
    print("\n=== Exercise 3: Single-Mode Fiber Design ===")
    exercise_3()
    print("\n=== Exercise 4: Mode Visualization ===")
    exercise_4()
    print("\n=== Exercise 5: Circular Waveguide Modes ===")
    exercise_5()
    print("\nAll exercises completed!")
