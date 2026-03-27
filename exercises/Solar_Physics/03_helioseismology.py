"""
Exercises for Lesson 03: Helioseismology
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import integrate


# --- Physical constants ---
R_sun = 6.957e8        # solar radius [m]
G = 6.674e-11          # gravitational constant [m^3 kg^-1 s^-2]
M_sun = 1.989e30       # solar mass [kg]


def exercise_1():
    """
    Problem 1: Mode Identification

    A solar oscillation: nu = 3.2 mHz, l = 100, n = 8.
    (a) p-mode, g-mode, or f-mode?
    (b) Horizontal wavelength on the solar surface.
    (c) Turning point depth using c(r)/r = 2*pi*nu / sqrt(l(l+1))
        with c(r) = c0 * (r/R_sun)^(-0.3), c0 = 7 km/s.
    (d) What is the mode primarily sensitive to?
    """
    nu = 3.2e-3         # Hz
    l = 100
    n = 8

    # (a) Mode type
    print(f"  (a) Mode with nu = {nu*1e3:.1f} mHz, l = {l}, n = {n}")
    print(f"      n > 0 and nu in the ~mHz range => this is a p-mode.")
    print(f"      (f-modes have n = 0; g-modes have lower frequencies and are")
    print(f"       confined to the radiative interior.)")

    # (b) Horizontal wavelength
    # lambda_h = 2*pi*R_sun / sqrt(l(l+1))
    lambda_h = 2.0 * np.pi * R_sun / np.sqrt(l * (l + 1))
    lambda_h_km = lambda_h / 1e3
    lambda_h_Mm = lambda_h / 1e6
    print(f"\n  (b) Horizontal wavelength:")
    print(f"      lambda_h = 2 pi R_sun / sqrt(l(l+1))")
    print(f"              = {lambda_h_km:.0f} km = {lambda_h_Mm:.1f} Mm")
    # Angular size
    theta_deg = 360.0 / np.sqrt(l * (l + 1))
    print(f"      Angular wavelength: ~{theta_deg:.1f} degrees")

    # (c) Turning point depth
    # At turning point: c(r_t) / r_t = 2*pi*nu / sqrt(l(l+1))
    # c(r) = c0 * (r / R_sun)^(-0.3)
    # => c0 * (r_t / R_sun)^(-0.3) / r_t = 2*pi*nu / sqrt(l(l+1))
    # => c0 / (R_sun^(-0.3) * r_t^(0.7)) = ... wait, let me redo
    # c(r_t) / r_t = c0 * (r_t / R_sun)^(-0.3) / r_t
    #              = c0 / (r_t * (r_t/R_sun)^0.3)
    #              = c0 / (R_sun * (r_t/R_sun)^1.3)
    # Setting equal to omega/L where L = sqrt(l(l+1)):
    # c0 / (R_sun * x^1.3) = 2*pi*nu / L  where x = r_t / R_sun
    # x^1.3 = c0 * L / (R_sun * 2*pi*nu)

    c0 = 7.0e3  # m/s
    L_ell = np.sqrt(l * (l + 1))
    omega_over_L = 2.0 * np.pi * nu / L_ell

    x_1p3 = c0 / (R_sun * omega_over_L)
    x = x_1p3**(1.0 / 1.3)  # r_t / R_sun

    r_t = x * R_sun
    depth = R_sun - r_t
    depth_Mm = depth / 1e6

    print(f"\n  (c) Turning point calculation:")
    print(f"      c0 = {c0/1e3:.0f} km/s")
    print(f"      L = sqrt(l(l+1)) = {L_ell:.1f}")
    print(f"      2 pi nu / L = {omega_over_L:.4e} rad/s/m")
    print(f"      (r_t / R_sun)^1.3 = {x_1p3:.4f}")
    print(f"      r_t / R_sun = {x:.4f}")
    print(f"      r_t = {r_t/1e6:.0f} km")
    print(f"      Depth below surface = {depth_Mm:.0f} Mm = {depth/(R_sun)*100:.1f}% of R_sun")

    # (d) Sensitivity
    print(f"\n  (d) This high-l p-mode is primarily sensitive to the sound speed")
    print(f"      profile in the outer ~{depth_Mm:.0f} Mm of the Sun.")
    print(f"      It probes the upper convection zone and photosphere region.")


def exercise_2():
    """
    Problem 2: Large Separation

    c(r) = c_c * [1 - 0.9*(r/R_sun)^2]^(1/2), c_c = 500 km/s.
    (a) Compute acoustic travel time tau = integral_0^R dr/c(r).
    (b) Large separation Delta_nu = 1/(2*tau).
    (c) Compare with observed ~135 muHz.
    (d) Star with 2x solar radius, same mean sound speed.
    """
    c_c = 500.0e3  # central sound speed [m/s]

    # (a) Numerical integration of tau = integral from 0 to R_sun of dr/c(r)
    def sound_speed(r):
        x = r / R_sun
        return c_c * np.sqrt(1.0 - 0.9 * x**2)

    def integrand(r):
        return 1.0 / sound_speed(r)

    # Use scipy for accurate numerical integration
    tau, error = integrate.quad(integrand, 0, R_sun)
    print(f"  (a) Sound speed model: c(r) = {c_c/1e3:.0f} km/s * sqrt(1 - 0.9*(r/R)^2)")
    print(f"      c(0) = {c_c/1e3:.0f} km/s (center)")
    print(f"      c(R) = {sound_speed(R_sun)/1e3:.1f} km/s (surface)")
    print(f"      Acoustic travel time: tau = {tau:.1f} s = {tau/60:.1f} min")

    # (b) Large separation
    Delta_nu = 1.0 / (2.0 * tau)
    Delta_nu_muHz = Delta_nu * 1e6
    print(f"\n  (b) Large separation: Delta_nu = 1/(2*tau) = {Delta_nu_muHz:.1f} muHz")

    # (c) Compare with observation
    Delta_nu_obs = 135.0  # muHz
    print(f"\n  (c) Observed large separation: ~{Delta_nu_obs:.0f} muHz")
    print(f"      Our estimate: {Delta_nu_muHz:.1f} muHz")
    print(f"      Ratio: {Delta_nu_muHz/Delta_nu_obs:.2f}")
    print(f"      The simple model gives a reasonable order-of-magnitude estimate.")

    # (d) Star with 2x R_sun, same mean sound speed
    # tau scales as R/c_mean, so tau_star = 2*tau_sun
    # Delta_nu_star = 1/(2*tau_star) = Delta_nu_sun / 2
    Delta_nu_star = Delta_nu_muHz / 2.0
    print(f"\n  (d) Star with 2 R_sun and same mean sound speed:")
    print(f"      tau_star = 2 * tau_sun")
    print(f"      Delta_nu_star = Delta_nu_sun / 2 = {Delta_nu_star:.1f} muHz")
    print(f"      The large separation scales inversely with stellar radius,")
    print(f"      making it a powerful asteroseismic probe of stellar size.")


def exercise_3():
    """
    Problem 3: Probing Different Depths

    Modes at l = 1, 20, 200, 1000, all with nu ~ 3 mHz.
    (a) Rank by penetration depth.
    (b-c) Which are useful for core/surface?
    (d) Why observe a wide range of l?
    """
    nu = 3.0e-3  # Hz
    l_values = [1, 20, 200, 1000]

    print(f"  Mode frequency: nu = {nu*1e3:.0f} mHz")
    print(f"  Turning point: r_t determined by c(r_t)/r_t = 2*pi*nu / sqrt(l(l+1))")
    print(f"  Lower l => deeper penetration (smaller l(l+1) => lower c/r threshold)")
    print()

    # Estimate turning points using a simple sound speed model
    # c(r) ~ c_c * sqrt(1 - 0.8*(r/R)^2), c_c = 500 km/s
    c_c = 500.0e3  # m/s

    print(f"  {'l':>6} {'sqrt(l(l+1))':>14} {'2pi*nu/L [s^-1/m]':>20} {'r_t/R_sun':>12} {'Depth [Mm]':>12}")
    print(f"  {'-'*6} {'-'*14} {'-'*20} {'-'*12} {'-'*12}")

    for l in l_values:
        L = np.sqrt(l * (l + 1))
        omega_over_L = 2.0 * np.pi * nu / L

        # Solve c(r_t)/r_t = omega_over_L numerically
        # c(r)/r = c_c * sqrt(1 - 0.8*(r/R)^2) / r
        r_test = np.linspace(0.01 * R_sun, 0.999 * R_sun, 10000)
        cr_over_r = c_c * np.sqrt(1.0 - 0.8 * (r_test / R_sun)**2) / r_test

        # Find where cr_over_r crosses omega_over_L
        idx = np.argmin(np.abs(cr_over_r - omega_over_L))
        r_t = r_test[idx]
        depth = (R_sun - r_t) / 1e6  # Mm

        print(f"  {l:6d} {L:14.1f} {omega_over_L:20.4e} {r_t/R_sun:12.3f} {depth:12.0f}")

    # (a-d) Answers
    print(f"\n  (a) Ranked by penetration depth (deepest first): l=1, l=20, l=200, l=1000")
    print(f"      Low-l modes penetrate to the core; high-l modes stay near the surface.")
    print(f"\n  (b) l=1 modes are most useful for probing the core sound speed.")
    print(f"\n  (c) l=1000 modes are most useful for studying near-surface structure.")
    print(f"\n  (d) A wide range of l provides tomographic coverage of the entire interior:")
    print(f"      - Low l: core (nuclear burning, composition)")
    print(f"      - Medium l: radiative zone, tachocline")
    print(f"      - High l: convection zone, near-surface layers")


def exercise_4():
    """
    Problem 4: Rotation Splitting

    (a) Rigid rotation splitting for Omega/(2pi) = 430 nHz.
    (b) Qualitative difference between low-l and high-l splitting patterns.
    """
    Omega_rigid = 430.0e-9  # Hz (= 430 nHz)

    # (a) For rigid rotation, the splitting between adjacent m values is:
    # delta_nu = integral K(r) Omega(r) dr ≈ Omega/(2pi) for rigid rotation
    # Actually, for rigid rotation: nu_{nlm} = nu_{nl0} + m * Omega/(2pi)
    # (with simplifications -- the exact rotational kernel integral)
    # The splitting between adjacent m values is Omega/(2pi)
    delta_nu = Omega_rigid  # Hz
    delta_nu_nHz = delta_nu * 1e9

    print(f"  (a) Rigid rotation rate: Omega/(2pi) = {Omega_rigid*1e9:.0f} nHz")
    print(f"      Splitting between adjacent m values: {delta_nu_nHz:.0f} nHz")
    print(f"      This is {delta_nu*1e6:.2f} muHz")
    print(f"      For a mode with l = 2: m = -2,-1,0,1,2 => 5 components")
    print(f"      Total spread = 2l * delta_nu = {2*2*delta_nu_nHz:.0f} nHz = {2*2*delta_nu*1e6:.2f} muHz")

    # (b) Low-l vs high-l splitting
    print(f"\n  (b) Low-l modes (l=1,2,3) penetrate to the core, which rotates")
    print(f"      nearly rigidly at ~430 nHz. Their rotational kernels K(r)")
    print(f"      sample the entire interior, so the splitting reflects the")
    print(f"      volume-averaged rotation rate.")
    print(f"      ")
    print(f"      High-l modes (l=100+) are trapped in the outer convection zone,")
    print(f"      where differential rotation is strong. The equatorial splitting")
    print(f"      (~460 nHz) is larger than the polar splitting (~340 nHz).")
    print(f"      By observing modes of different l, we can invert for Omega(r,theta).")
    print(f"      This is how the internal rotation profile was mapped, revealing")
    print(f"      the tachocline where rigid rotation transitions to differential.")


def exercise_5():
    """
    Problem 5: Far-Side Imaging

    (a) Round-trip travel time for sound through the Sun.
    (b) Travel time change for 1% sound speed increase.
    (c) Is this detectable (precision ~1 s)?
    (d) Why useful for space weather?
    """
    c_mean = 100.0e3   # mean sound speed [m/s]
    path_length = 2.0 * R_sun  # round trip through interior

    # (a) Round-trip travel time
    t_roundtrip = path_length / c_mean
    t_hours = t_roundtrip / 3600.0
    print(f"  (a) Mean sound speed: c_mean = {c_mean/1e3:.0f} km/s")
    print(f"      Path length: ~2 R_sun = {path_length:.3e} m")
    print(f"      Round-trip travel time: {t_roundtrip:.0f} s = {t_hours:.2f} hours")
    print(f"                            = {t_roundtrip/60:.0f} minutes")

    # (b) Travel time change for 1% sound speed increase
    # If c -> c(1 + 0.01) locally, the travel time changes by:
    # delta_t / t ≈ -delta_c / c for the affected segment
    # Assume the active region affects ~10% of the path
    frac_affected = 0.1  # fraction of path through active region
    dc_over_c = 0.01     # 1% speed increase

    delta_t = -t_roundtrip * frac_affected * dc_over_c
    print(f"\n  (b) Sound speed increase: {dc_over_c*100:.0f}%")
    print(f"      Fraction of path affected: ~{frac_affected*100:.0f}%")
    print(f"      Travel time change: delta_t = -{t_roundtrip:.0f} x {frac_affected} x {dc_over_c}")
    print(f"                        = {delta_t:.1f} s")

    # Even if the whole path is affected:
    delta_t_full = -t_roundtrip * dc_over_c
    print(f"      If entire path affected: delta_t = {delta_t_full:.1f} s")

    # (c) Detectability
    precision = 1.0  # s
    print(f"\n  (c) Measurement precision: ~{precision:.0f} s")
    print(f"      |delta_t| = {abs(delta_t):.1f} s (partial path)")
    if abs(delta_t) > precision:
        print(f"      Yes, this is detectable! |delta_t| > precision")
    else:
        print(f"      Marginal detection for 10% path fraction.")
    print(f"      |delta_t| = {abs(delta_t_full):.1f} s (full path) -- clearly detectable")

    # (d) Space weather relevance
    print(f"\n  (d) Far-side imaging is crucial for space weather because:")
    print(f"      - Active regions rotating onto the visible disk can produce")
    print(f"        flares and CMEs within days of appearing at the east limb.")
    print(f"      - Far-side imaging gives ~1-2 weeks advance warning of large")
    print(f"        active regions that will rotate to face Earth.")
    print(f"      - This is used operationally by NOAA/SWPC for forecasting.")
    print(f"      - Currently implemented using SDO/HMI helioseismic data.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Mode Identification ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Large Separation ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Probing Different Depths ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Rotation Splitting ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Far-Side Imaging ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
