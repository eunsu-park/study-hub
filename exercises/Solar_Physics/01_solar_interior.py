"""
Exercises for Lesson 01: Solar Interior
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np


# --- Physical constants ---
G = 6.674e-11          # gravitational constant [m^3 kg^-1 s^-2]
M_sun = 1.989e30       # solar mass [kg]
R_sun = 6.957e8        # solar radius [m]
L_sun = 3.828e26       # solar luminosity [W]
k_B = 1.381e-23        # Boltzmann constant [J/K]
m_H = 1.673e-27        # hydrogen mass [kg]
c = 3.0e8              # speed of light [m/s]
sigma_SB = 5.670e-8    # Stefan-Boltzmann constant [W m^-2 K^-4]


def exercise_1():
    """
    Problem 1: Hydrostatic Equilibrium Estimate

    Integrate dP/dr = -G m(r) rho / r^2 from surface (P=0) to center
    assuming uniform density rho = 3 M_sun / (4 pi R_sun^3).
    Compare with SSM value of 2.5e16 Pa.
    """
    # Mean density of a uniform-density Sun
    rho_bar = 3.0 * M_sun / (4.0 * np.pi * R_sun**3)
    print(f"  Mean density rho_bar = {rho_bar:.1f} kg/m^3")

    # For uniform density, m(r) = (4/3) pi r^3 rho
    # dP/dr = -G * (4/3 pi r^3 rho) * rho / r^2 = -(4/3) pi G rho^2 r
    # Integrating from R to 0: P(0) - P(R) = integral from R to 0 of dP/dr dr
    # P(R) = 0, so P_c = integral from 0 to R of (4/3) pi G rho^2 r dr
    # P_c = (4/3) pi G rho^2 * R^2 / 2 = (2/3) pi G rho^2 R^2

    # Analytical result
    P_c_analytical = (2.0 / 3.0) * np.pi * G * rho_bar**2 * R_sun**2
    print(f"  Central pressure (analytical): P_c = {P_c_analytical:.3e} Pa")

    # Numerical integration for verification
    N = 10000
    r = np.linspace(R_sun, 0, N + 1)
    dr = r[1] - r[0]  # negative (integrating inward)
    P = np.zeros(N + 1)
    P[0] = 0.0  # surface pressure = 0
    for i in range(N):
        m_r = (4.0 / 3.0) * np.pi * r[i]**3 * rho_bar
        if r[i] > 0:
            dPdr = -G * m_r * rho_bar / r[i]**2
        else:
            dPdr = 0.0
        P[i + 1] = P[i] - dPdr * dr  # minus because dr is negative
    P_c_numerical = P[-1]
    print(f"  Central pressure (numerical):  P_c = {P_c_numerical:.3e} Pa")

    # Compare with SSM value
    P_c_SSM = 2.5e16  # Pa
    ratio = P_c_SSM / P_c_analytical
    print(f"  SSM central pressure: {P_c_SSM:.1e} Pa")
    print(f"  SSM / estimate ratio: {ratio:.0f}")
    print(f"  The estimate is too low because the real Sun is centrally concentrated.")
    print(f"  The core density (~1.5e5 kg/m^3) >> mean density (~{rho_bar:.0f} kg/m^3).")


def exercise_2():
    """
    Problem 2: Photon Random Walk

    (a) Mean free path at the solar center.
    (b) Number of scatterings to traverse R_sun.
    (c) Photon escape time.
    (d) Time for luminosity change if nuclear reactions stop.
    """
    kappa = 1.0        # opacity [m^2/kg]
    rho_c = 1.5e5      # central density [kg/m^3]

    # (a) Mean free path
    l_mfp = 1.0 / (kappa * rho_c)
    print(f"  (a) Mean free path: l_mfp = {l_mfp:.2e} m = {l_mfp*100:.4f} cm")

    # (b) Number of scatterings for random walk across R_sun
    # d = l * sqrt(N)  =>  N = (R_sun / l_mfp)^2
    N_scatter = (R_sun / l_mfp)**2
    print(f"  (b) Number of scatterings: N = {N_scatter:.2e}")

    # (c) Photon escape time
    # Total path length = N * l_mfp, travel time = total path / c
    total_path = N_scatter * l_mfp
    t_escape_s = total_path / c
    t_escape_yr = t_escape_s / (3.156e7)
    print(f"  (c) Total path length: {total_path:.2e} m")
    print(f"      Photon escape time: {t_escape_s:.2e} s = {t_escape_yr:.0f} years")

    # (d) Kelvin-Helmholtz timescale
    # If nuclear reactions stop, the Sun continues to radiate on the thermal
    # (Kelvin-Helmholtz) timescale: t_KH ~ G M^2 / (R L)
    t_KH = G * M_sun**2 / (R_sun * L_sun)
    t_KH_yr = t_KH / 3.156e7
    print(f"  (d) Kelvin-Helmholtz timescale: {t_KH:.2e} s = {t_KH_yr:.2e} years")
    print(f"      The thermal diffusion time (~{t_escape_yr:.0f} yr) sets the timescale")
    print(f"      for luminosity changes to become noticeable at the surface.")


def exercise_3():
    """
    Problem 3: Schwarzschild Criterion

    At the base of the convection zone (r ~ 0.71 R_sun):
    T ~ 2.3e6 K, H_P ~ 6e7 m, g ~ 500 m/s^2, mu = 0.6
    (a) Calculate the adiabatic temperature gradient |dT/dr|_ad = g / c_p.
    (b) Compare with radiative gradient of 7e-2 K/m.
    """
    T = 2.3e6           # temperature [K]
    H_P = 6.0e7         # pressure scale height [m]
    g = 500.0            # gravitational acceleration [m/s^2]
    mu = 0.6             # mean molecular weight
    gamma = 5.0 / 3.0    # adiabatic index (monatomic ideal gas)

    # (a) c_p = (gamma / (gamma-1)) * k_B / (mu * m_H)  for ideal gas
    # Then |dT/dr|_ad = g / c_p
    c_p = (gamma / (gamma - 1.0)) * k_B / (mu * m_H)
    print(f"  (a) c_p = {c_p:.3e} J/(kg K)")

    dTdr_ad = g / c_p
    print(f"      |dT/dr|_ad = g / c_p = {dTdr_ad:.4e} K/m")

    # Alternative: |dT/dr|_ad = (1 - 1/gamma) * T/P * |dP/dr|
    # With |dP/dr| = g * rho  and P = rho * k_B * T / (mu * m_H):
    # |dT/dr|_ad = (1 - 1/gamma) * (mu * m_H * g) / k_B
    dTdr_ad_alt = (1.0 - 1.0 / gamma) * mu * m_H * g / k_B
    print(f"      Alternative calculation: {dTdr_ad_alt:.4e} K/m")

    # (b) Convective stability check
    dTdr_rad = 7.0e-2  # K/m (actual radiative gradient)
    print(f"\n  (b) Radiative gradient: |dT/dr|_rad = {dTdr_rad:.4e} K/m")
    print(f"      Adiabatic gradient:  |dT/dr|_ad  = {dTdr_ad:.4e} K/m")

    if dTdr_rad > dTdr_ad:
        print(f"      |dT/dr|_rad > |dT/dr|_ad => CONVECTIVELY UNSTABLE")
        print(f"      Schwarzschild criterion is violated; convection occurs.")
    else:
        print(f"      |dT/dr|_rad < |dT/dr|_ad => convectively stable")


def exercise_4():
    """
    Problem 4: Convective Velocity

    Estimate the convective velocity near the base of the convection zone.
    dT/T ~ 1e-6, l = 1.5 * H_P, g = 500 m/s^2.
    v ~ (g * dT * l / T)^{1/2}
    Compare with local sound speed c_s ~ 2e5 m/s.
    """
    dT_over_T = 1.0e-6     # superadiabatic excess delta T / T
    alpha_MLT = 1.5         # mixing length parameter
    H_P = 6.0e7             # pressure scale height [m]
    g = 500.0               # gravitational acceleration [m/s^2]
    c_s = 2.0e5             # local sound speed [m/s]

    # Mixing length
    l_mix = alpha_MLT * H_P
    print(f"  Mixing length: l = {alpha_MLT} * H_P = {l_mix:.2e} m")

    # Convective velocity
    v_conv = np.sqrt(g * dT_over_T * l_mix)
    print(f"  Convective velocity: v_conv = {v_conv:.1f} m/s")

    # Compare with sound speed
    ratio = v_conv / c_s
    print(f"  Sound speed: c_s = {c_s:.1e} m/s")
    print(f"  v_conv / c_s = {ratio:.2e}")
    print(f"\n  The convective velocity is much smaller than the sound speed because")
    print(f"  the superadiabatic excess (dT/T ~ 1e-6) is tiny in the deep convection zone.")
    print(f"  Nearly adiabatic stratification means convection is very efficient:")
    print(f"  a tiny temperature perturbation carries the full solar luminosity.")
    print(f"  The convective Mach number ~ {ratio:.1e} => subsonic, as expected.")


def exercise_5():
    """
    Problem 5: Tachocline Shear

    (a) Radial shear dOmega/dr in the tachocline.
    (b) Toroidal field generated after one rotation period.
    (c) Number of rotation periods to amplify to 10^5 G.
    """
    # Given rotation rates at two radii
    Omega_070 = 430.0e-9 * 2.0 * np.pi   # rad/s at r = 0.70 R_sun
    Omega_072 = 460.0e-9 * 2.0 * np.pi   # rad/s at r = 0.72 R_sun
    r_070 = 0.70 * R_sun
    r_072 = 0.72 * R_sun
    dr = r_072 - r_070

    # (a) Radial shear
    dOmega_dr = (Omega_072 - Omega_070) / dr
    print(f"  (a) Omega at 0.70 R_sun = {Omega_070/(2*np.pi)*1e9:.0f} nHz = {Omega_070:.4e} rad/s")
    print(f"      Omega at 0.72 R_sun = {Omega_072/(2*np.pi)*1e9:.0f} nHz = {Omega_072:.4e} rad/s")
    print(f"      dr = {dr:.3e} m")
    print(f"      dOmega/dr = {dOmega_dr:.3e} rad/(s m)")

    # (b) Toroidal field after one rotation
    # B_phi ~ B_r * r * (dOmega/dr) * P
    B_r = 1.0e-4       # 1 G = 1e-4 T
    r_tach = 0.71 * R_sun
    P_rot = 27.0 * 86400.0  # rotation period in seconds

    B_phi = B_r * r_tach * dOmega_dr * P_rot
    B_phi_G = B_phi * 1.0e4  # convert T to Gauss
    print(f"\n  (b) B_r = 1 G")
    print(f"      r = 0.71 R_sun = {r_tach:.3e} m")
    print(f"      P_rot = 27 days = {P_rot:.3e} s")
    print(f"      B_phi = B_r * r * (dOmega/dr) * P = {B_phi:.3e} T = {B_phi_G:.1f} G")

    # (c) Number of rotations to reach 10^5 G
    B_target = 1.0e5      # target field in Gauss
    # B_phi grows linearly: B_phi(N) = N * B_phi_per_rotation
    B_phi_per_rotation_G = B_phi_G
    N_rotations = B_target / B_phi_per_rotation_G
    time_needed_yr = N_rotations * 27.0 / 365.25
    print(f"\n  (c) Target field: {B_target:.0e} G")
    print(f"      B_phi per rotation: {B_phi_per_rotation_G:.1f} G")
    print(f"      Number of rotations needed: {N_rotations:.0f}")
    print(f"      Time needed: {time_needed_yr:.1f} years")
    print(f"      This is comparable to the ~11-year solar cycle period.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Hydrostatic Equilibrium Estimate ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Photon Random Walk ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Schwarzschild Criterion ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Convective Velocity ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Tachocline Shear ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
