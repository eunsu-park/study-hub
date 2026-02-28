"""
Exercises for Lesson 09: EM Waves in Vacuum
Topic: Electrodynamics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Constants
epsilon_0 = 8.854e-12
mu_0 = 4.0 * np.pi * 1e-7
c = 1.0 / np.sqrt(mu_0 * epsilon_0)


def exercise_1():
    """
    Exercise 1: Wave Verification
    Verify E = E0*cos(kz - wt) x_hat and B = (E0/c)*cos(kz - wt) y_hat
    satisfy all four Maxwell equations in vacuum.
    """
    print("  E = E0*cos(kz - wt) x_hat")
    print("  B = (E0/c)*cos(kz - wt) y_hat")
    print()

    # Check each Maxwell equation:
    print("  1. div(E) = 0:")
    print("     dEx/dx = 0 (Ex depends only on z and t)")
    print("     SATISFIED")
    print()

    print("  2. div(B) = 0:")
    print("     dBy/dy = 0 (By depends only on z and t)")
    print("     SATISFIED")
    print()

    print("  3. curl(E) = -dB/dt:")
    print("     curl(E) = dEx/dz y_hat = -E0*k*sin(kz-wt) y_hat")
    print("     -dB/dt = -(E0/c)*w*sin(kz-wt) (-1) y_hat = (E0*w/c)*sin(kz-wt) y_hat")
    print("     Need: E0*k = E0*w/c => k = w/c => w = ck")
    print("     SATISFIED (dispersion relation)")
    print()

    print("  4. curl(B) = mu_0*eps_0*dE/dt:")
    print("     curl(B) = -dBy/dz x_hat = (E0/c)*k*sin(kz-wt) x_hat")
    print("     mu_0*eps_0*dE/dt = (1/c^2)*E0*w*sin(kz-wt) x_hat")
    print("     Need: k/c = w/c^2 => k = w/c")
    print("     SATISFIED (same dispersion relation)")

    # Numerical verification at random (z, t)
    E0 = 100.0   # V/m
    omega = 2 * np.pi * 1e9  # 1 GHz
    k = omega / c

    z_test = 0.3
    t_test = 1e-9

    Ex = E0 * np.cos(k * z_test - omega * t_test)
    By = (E0 / c) * np.cos(k * z_test - omega * t_test)

    # Numerical derivatives
    dz = 1e-8
    dt = 1e-18
    dEx_dz = E0 * (-np.sin(k * z_test - omega * t_test)) * k
    dBy_dt = (E0 / c) * np.sin(k * z_test - omega * t_test) * omega

    print(f"\n  Numerical check at z = {z_test} m, t = {t_test*1e9:.1f} ns:")
    print(f"    curl(E)_y = dEx/dz = {dEx_dz:.4f}")
    print(f"    -dB/dt_y = {dBy_dt:.4f}")
    print(f"    Ratio: {dEx_dz/(-dBy_dt):.6f} (should be -1)")


def exercise_2():
    """
    Exercise 2: Circular Polarization Decomposition
    Show linear polarization = superposition of two circular polarizations.
    """
    omega = 2 * np.pi * 1e9
    k = omega / c
    E0 = 1.0

    t = np.linspace(0, 2 * np.pi / omega, 200)
    z = 0.0  # observe at z=0

    # Linearly polarized: E = E0*cos(wt) x_hat
    Ex_linear = E0 * np.cos(omega * t)
    Ey_linear = np.zeros_like(t)

    # Right circular: E_R = (E0/2)(cos(wt) x_hat + sin(wt) y_hat)
    Ex_R = (E0 / 2) * np.cos(omega * t)
    Ey_R = (E0 / 2) * np.sin(omega * t)

    # Left circular: E_L = (E0/2)(cos(wt) x_hat - sin(wt) y_hat)
    Ex_L = (E0 / 2) * np.cos(omega * t)
    Ey_L = -(E0 / 2) * np.sin(omega * t)

    # Sum should equal linear
    Ex_sum = Ex_R + Ex_L
    Ey_sum = Ey_R + Ey_L

    error_x = np.max(np.abs(Ex_sum - Ex_linear))
    error_y = np.max(np.abs(Ey_sum - Ey_linear))

    print("  Linear polarization: E = E0*cos(wt) x_hat")
    print("  = (E0/2)[cos(wt) x_hat + sin(wt) y_hat]  (right circular)")
    print("  + (E0/2)[cos(wt) x_hat - sin(wt) y_hat]  (left circular)")
    print()
    print(f"  Verification: max error in Ex = {error_x:.2e}")
    print(f"  Verification: max error in Ey = {error_y:.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(Ex_R, Ey_R, 'r-', linewidth=1.5)
    axes[0].set_xlabel('Ex')
    axes[0].set_ylabel('Ey')
    axes[0].set_title('Right Circular (E_R)')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(Ex_L, Ey_L, 'b-', linewidth=1.5)
    axes[1].set_xlabel('Ex')
    axes[1].set_ylabel('Ey')
    axes[1].set_title('Left Circular (E_L)')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(Ex_sum, Ey_sum, 'g-', linewidth=2, label='Sum (R+L)')
    axes[2].plot(Ex_linear, Ey_linear, 'k--', linewidth=1.5, label='Linear')
    axes[2].set_xlabel('Ex')
    axes[2].set_ylabel('Ey')
    axes[2].set_title('Sum = Linear Polarization')
    axes[2].set_aspect('equal')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Circular Polarization Decomposition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex09_circular_decomposition.png', dpi=150)
    plt.close()
    print("  Plot saved: ex09_circular_decomposition.png")


def exercise_3():
    """
    Exercise 3: Solar Sail
    A=100 m^2, R=0.95 reflectivity, at 1 AU.
    """
    I_sun = 1361.0     # W/m^2
    A_sail = 100.0     # m^2
    reflectivity = 0.95
    m_sail = 1.0       # kg

    # (a) Radiation force: F = (1+R)*I*A/c
    F_rad = (1 + reflectivity) * I_sun * A_sail / c

    # (b) Acceleration
    a_sail = F_rad / m_sail

    # (c) Time to Mars (1.5 AU) -- ignoring gravity, constant force
    r_AU = 1.496e11  # 1 AU in meters
    delta_r = 0.5 * r_AU  # distance to Mars orbit from Earth orbit
    # s = 0.5*a*t^2 => t = sqrt(2*s/a)
    t_mars = np.sqrt(2 * delta_r / a_sail)

    # Compare with Sun's gravity
    G = 6.674e-11
    M_sun = 1.989e30
    F_grav = G * M_sun * m_sail / r_AU**2

    print(f"  Solar sail: A = {A_sail} m^2, R = {reflectivity}, mass = {m_sail} kg")
    print(f"  Solar irradiance: {I_sun} W/m^2 at 1 AU")
    print()
    print(f"  (a) Radiation force: F = (1+R)*I*A/c = {F_rad*1e3:.4f} mN")
    print(f"  (b) Acceleration: a = {a_sail*1e6:.4f} um/s^2 = {a_sail*1e3:.4f} mm/s^2")
    print(f"  (c) Time to Mars orbit (ignoring gravity):")
    print(f"      Distance: {delta_r:.3e} m")
    print(f"      Time: {t_mars:.2e} s = {t_mars/(3600*24):.1f} days")
    print(f"  Gravitational force from Sun: {F_grav:.4e} N")
    print(f"  Ratio F_rad/F_grav: {F_rad/F_grav:.4e}")
    print(f"  (Radiation force is much smaller than gravity for 1 kg)")


def exercise_4():
    """
    Exercise 4: Polarizer Chain
    Three polarizers at 0, 45, 90 degrees. Compute transmission with Malus's law.
    """
    I0 = 1.0  # incident unpolarized intensity

    # After first polarizer (0 deg): I1 = I0/2 (unpolarized -> linear)
    I1 = I0 / 2

    # After second polarizer (45 deg): I2 = I1*cos^2(45)
    I2 = I1 * np.cos(np.radians(45))**2

    # After third polarizer (90 deg): I3 = I2*cos^2(90-45) = I2*cos^2(45)
    I3 = I2 * np.cos(np.radians(45))**2

    print("  Three polarizers at 0, 45, and 90 degrees:")
    print(f"  I0 (unpolarized) = {I0}")
    print(f"  After P1 (0 deg):   I1 = I0/2 = {I1:.4f}")
    print(f"  After P2 (45 deg):  I2 = I1*cos^2(45) = {I2:.4f}")
    print(f"  After P3 (90 deg):  I3 = I2*cos^2(45) = {I3:.4f}")
    print(f"  Transmission: {I3/I0*100:.2f}%")

    # Two crossed polarizers (0 and 90 only):
    I_crossed = I0 / 2 * np.cos(np.radians(90))**2
    print(f"\n  Two crossed polarizers (0 and 90 only):")
    print(f"  I = I0/2 * cos^2(90) = {I_crossed:.4f}")
    print("  No light passes through (zero transmission)!")
    print()
    print("  The intermediate polarizer at 45 deg 'rotates' the polarization,")
    print("  allowing some light through that would otherwise be blocked.")

    # Stokes vector simulation
    print("\n  Stokes vector simulation:")
    # Unpolarized: S = [1, 0, 0, 0]
    S = np.array([1.0, 0.0, 0.0, 0.0])
    print(f"  Initial Stokes: {S}")

    # Mueller matrix for ideal linear polarizer at angle theta
    def mueller_polarizer(theta_deg):
        theta = np.radians(theta_deg)
        c2 = np.cos(2 * theta)
        s2 = np.sin(2 * theta)
        M = 0.5 * np.array([
            [1, c2, s2, 0],
            [c2, c2**2, c2 * s2, 0],
            [s2, c2 * s2, s2**2, 0],
            [0, 0, 0, 0]
        ])
        return M

    for angle in [0, 45, 90]:
        M = mueller_polarizer(angle)
        S = M @ S
        print(f"  After P({angle} deg): S = [{S[0]:.4f}, {S[1]:.4f}, {S[2]:.4f}, {S[3]:.4f}]")
        print(f"    Intensity = {S[0]:.4f}")


def exercise_5():
    """
    Exercise 5: Standing Waves
    Two counter-propagating plane waves create a standing wave.
    Show time-averaged Poynting vector is zero.
    """
    omega = 2 * np.pi * 1e9
    k = omega / c
    E0 = 1.0

    z = np.linspace(0, 4 * np.pi / k, 500)
    lam = 2 * np.pi / k

    # Standing wave: sum of right-moving and left-moving
    # E_total = E0*cos(kz - wt) + E0*cos(kz + wt) = 2*E0*cos(kz)*cos(wt)
    # B_total = (E0/c)*cos(kz - wt) - (E0/c)*cos(kz + wt) = 2*(E0/c)*sin(kz)*sin(wt)

    # Time-averaged Poynting vector:
    # <S> = <E x B>/mu_0 = <2E0*cos(kz)*cos(wt) * 2(E0/c)*sin(kz)*sin(wt)>/mu_0
    # = (4*E0^2/(c*mu_0)) * sin(kz)*cos(kz) * <cos(wt)*sin(wt)>
    # = 0 (since <cos*sin> = 0)

    # Numerical verification at several time points
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    times = [0, 0.25 / (omega / (2 * np.pi)), 0.5 / (omega / (2 * np.pi)),
             0.75 / (omega / (2 * np.pi))]

    S_time_avg = np.zeros_like(z)
    N_avg = 1000
    t_avg = np.linspace(0, 2 * np.pi / omega, N_avg)

    for t_i in t_avg:
        Ex_total = 2 * E0 * np.cos(k * z) * np.cos(omega * t_i)
        By_total = 2 * (E0 / c) * np.sin(k * z) * np.sin(omega * t_i)
        S_inst = Ex_total * By_total / mu_0
        S_time_avg += S_inst / N_avg

    for idx, t_i in enumerate(times):
        ax = axes[idx // 2, idx % 2]
        Ex = 2 * E0 * np.cos(k * z) * np.cos(omega * t_i)
        By = 2 * (E0 / c) * np.sin(k * z) * np.sin(omega * t_i)
        S = Ex * By / mu_0

        ax.plot(z / lam, Ex / E0, 'b-', linewidth=1.5, label='E/E0')
        ax.plot(z / lam, By * c / E0, 'r-', linewidth=1.5, label='cB/E0')
        ax.plot(z / lam, S / (E0**2 / (c * mu_0)), 'g--', linewidth=1.5, label='S (norm)')
        ax.set_xlabel(r'z / $\lambda$')
        ax.set_ylabel('Amplitude')
        ax.set_title(f't = {t_i * omega / (2*np.pi):.2f} T')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2.5, 2.5)

    plt.suptitle('Standing Wave: E, B, and S', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ex09_standing_wave.png', dpi=150)
    plt.close()

    print(f"  Standing wave from two counter-propagating waves")
    print(f"  E = 2*E0*cos(kz)*cos(wt)")
    print(f"  B = 2*(E0/c)*sin(kz)*sin(wt)")
    print(f"  <S> time-averaged: max|<S>| = {np.max(np.abs(S_time_avg)):.4e}")
    print(f"  (Should be ~0; numerical residual from finite averaging)")
    print("  Plot saved: ex09_standing_wave.png")


if __name__ == "__main__":
    print("=== Exercise 1: Wave Verification ===")
    exercise_1()
    print("\n=== Exercise 2: Circular Polarization Decomposition ===")
    exercise_2()
    print("\n=== Exercise 3: Solar Sail ===")
    exercise_3()
    print("\n=== Exercise 4: Polarizer Chain ===")
    exercise_4()
    print("\n=== Exercise 5: Standing Waves ===")
    exercise_5()
    print("\nAll exercises completed!")
