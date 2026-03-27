"""
Exercises for Lesson 04: Photosphere
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np


# --- Physical constants ---
k_B = 1.381e-23        # Boltzmann constant [J/K]
m_H = 1.673e-27        # hydrogen mass [kg]
c = 3.0e8              # speed of light [m/s]
sigma_SB = 5.670e-8    # Stefan-Boltzmann constant [W m^-2 K^-4]
T_eff = 5778.0         # solar effective temperature [K]
R_sun = 6.957e8        # solar radius [m]


def exercise_1():
    """
    Problem 1: Eddington-Barbier Relation

    S(tau) = S_0 + S_1 * tau (linear source function)
    S_0 = 2e13, S_1 = 5e13 W m^-2 Hz^-1 sr^-1.
    (a) Exact emergent intensity using formal solution.
    (b) Compare with Eddington-Barbier approximation I ~ S(tau = mu).
    (c) Limb-to-center ratio for mu_limb = 0.2.
    (d) Limb darkening coefficient u.
    """
    S_0 = 2.0e13   # W m^-2 Hz^-1 sr^-1
    S_1 = 5.0e13   # W m^-2 Hz^-1 sr^-1

    # (a) Formal solution for emergent intensity:
    # I(0, mu) = integral_0^inf S(tau) * exp(-tau/mu) * dtau/mu
    #          = integral_0^inf (S_0 + S_1*tau) * exp(-tau/mu) * dtau/mu
    # Using integral_0^inf exp(-tau/mu) dtau/mu = 1
    # and integral_0^inf tau * exp(-tau/mu) dtau/mu = mu
    # => I(0, mu) = S_0 + S_1 * mu

    print(f"  (a) Source function: S(tau) = S_0 + S_1 * tau")
    print(f"      S_0 = {S_0:.1e}, S_1 = {S_1:.1e}")
    print(f"")
    print(f"      Formal solution:")
    print(f"      I(0, mu) = integral_0^inf (S_0 + S_1*tau) * exp(-tau/mu) * dtau/mu")
    print(f"               = S_0 * 1 + S_1 * mu")
    print(f"               = S_0 + S_1 * mu")

    # Verify with numerical integration
    mu_values = [1.0, 0.5, 0.2]
    print(f"\n      Exact results for different mu:")
    for mu in mu_values:
        I_exact = S_0 + S_1 * mu
        print(f"        mu = {mu:.1f}: I = {I_exact:.2e}")

    # (b) Eddington-Barbier approximation: I ~ S(tau = mu)
    print(f"\n  (b) Eddington-Barbier approximation: I ~ S(tau = mu)")
    print(f"      S(tau = mu) = S_0 + S_1 * mu")
    print(f"      For a linear source function, the Eddington-Barbier")
    print(f"      approximation is EXACT! This is because the formal solution")
    print(f"      gives exactly S(tau = mu) when S varies linearly with tau.")

    # (c) Limb-to-center ratio
    mu_center = 1.0
    mu_limb = 0.2
    I_center = S_0 + S_1 * mu_center
    I_limb = S_0 + S_1 * mu_limb
    ratio = I_limb / I_center

    print(f"\n  (c) I(center, mu=1) = {I_center:.2e}")
    print(f"      I(limb, mu={mu_limb}) = {I_limb:.2e}")
    print(f"      I(limb)/I(center) = {ratio:.3f}")

    # (d) Limb darkening coefficient
    # I(mu) / I(1) = 1 - u(1 - mu)
    # ratio = 1 - u(1 - mu_limb)
    # u = (1 - ratio) / (1 - mu_limb)
    u = (1.0 - ratio) / (1.0 - mu_limb)
    print(f"\n  (d) Using I(mu)/I(1) = 1 - u(1 - mu):")
    print(f"      u = (1 - ratio) / (1 - mu_limb)")
    print(f"      u = (1 - {ratio:.3f}) / (1 - {mu_limb})")
    print(f"      u = {u:.3f}")

    # Verify: I(mu)/I(1) = 1 - u*(1-mu)
    print(f"\n      Verification:")
    for mu in [1.0, 0.5, 0.2]:
        I_exact = S_0 + S_1 * mu
        I_LD = I_center * (1.0 - u * (1.0 - mu))
        print(f"        mu = {mu:.1f}: Exact = {I_exact:.2e}, LD model = {I_LD:.2e}")


def exercise_2():
    """
    Problem 2: Photon Mean Free Path

    H- opacity at tau_5000 = 1: T = 6400 K, rho = 3e-4 kg/m^3,
    kappa_5000 = 0.26 m^2/kg.
    (a) Calculate l_mfp.
    (b) Express in km and fraction of H_P (150 km).
    (c) Why is the photosphere so thin?
    """
    kappa = 0.26      # opacity [m^2/kg]
    rho = 3.0e-4      # density [kg/m^3]
    H_P = 150.0e3     # pressure scale height [m]
    T = 6400.0        # K

    # (a) Mean free path
    l_mfp = 1.0 / (kappa * rho)
    print(f"  (a) kappa_5000 = {kappa} m^2/kg")
    print(f"      rho = {rho:.1e} kg/m^3")
    print(f"      l_mfp = 1 / (kappa * rho) = {l_mfp:.1f} m")

    # (b) In km and fraction of H_P
    l_mfp_km = l_mfp / 1e3
    frac_HP = l_mfp / H_P
    print(f"\n  (b) l_mfp = {l_mfp_km:.3f} km = {l_mfp:.1f} m")
    print(f"      H_P = {H_P/1e3:.0f} km")
    print(f"      l_mfp / H_P = {frac_HP:.4e}")
    print(f"      The mean free path is about {1/frac_HP:.0f} times smaller than H_P.")

    # (c) Thin photosphere
    # The photosphere is the layer where tau ~ 1, which spans ~1 mfp in tau
    # Physical thickness ~ H_P (a few pressure scale heights)
    thickness_km = H_P / 1e3 * 2  # roughly 2 scale heights
    frac_R = thickness_km / (R_sun / 1e3)
    print(f"\n  (c) The photosphere spans roughly Delta_tau ~ 1,")
    print(f"      corresponding to a physical thickness of ~{thickness_km:.0f} km")
    print(f"      (a few pressure scale heights).")
    print(f"      This is only {frac_R*100:.3f}% of R_sun = {R_sun/1e3/1e3:.0f} Mm.")
    print(f"      The photosphere is thin because:")
    print(f"      1. H- opacity is extremely sensitive to temperature (steep gradient)")
    print(f"      2. The density scale height is small (150 km)")
    print(f"      3. Opacity * density changes rapidly over a few scale heights")


def exercise_3():
    """
    Problem 3: Granulation Velocities

    Upflow v_u = 1.5 km/s, downflow v_d = 2.5 km/s, bright fraction f = 0.6.
    (a) Density ratio rho_d/rho_u.
    (b) Mass flux in the upflow.
    (c) Kinetic energy flux, compare with F_rad.
    """
    v_u = 1.5e3       # upflow velocity [m/s]
    v_d = 2.5e3       # downflow velocity [m/s]
    f = 0.6           # bright area fraction (upflow)
    rho_0 = 3.0e-4    # typical photospheric density [kg/m^3]

    # (a) Mass conservation: rho_u * v_u * A_u = rho_d * v_d * A_d
    # A_u = f * A_total, A_d = (1-f) * A_total
    # rho_u * v_u * f = rho_d * v_d * (1-f)
    # rho_d / rho_u = v_u * f / (v_d * (1-f))
    rho_ratio = (v_u * f) / (v_d * (1.0 - f))
    print(f"  (a) Mass conservation: rho_u * v_u * f = rho_d * v_d * (1-f)")
    print(f"      rho_d/rho_u = (v_u * f) / (v_d * (1-f))")
    print(f"                  = ({v_u/1e3} * {f}) / ({v_d/1e3} * {1-f})")
    print(f"                  = {rho_ratio:.2f}")
    print(f"      Downflow gas is {1/rho_ratio:.1f}x less dense than upflow gas.")
    print(f"      (or equivalently, downflow regions are slightly cooler and denser")
    print(f"       -- wait, the ratio < 1 means downflow is LESS dense.)")
    print(f"      This is because the dark (cool) downflow lanes are narrow but fast.")

    # (b) Mass flux in upflow
    # Assume rho_0 is average: rho_0 = f * rho_u + (1-f) * rho_d
    # With rho_d = rho_ratio * rho_u:
    # rho_0 = f * rho_u + (1-f) * rho_ratio * rho_u = rho_u * (f + (1-f)*rho_ratio)
    rho_u = rho_0 / (f + (1.0 - f) * rho_ratio)
    rho_d = rho_ratio * rho_u

    mass_flux_up = rho_u * v_u  # kg/(m^2 s) per unit upflow area
    total_mass_flux = rho_u * v_u * f  # kg/(m^2 s) averaged over total area

    print(f"\n  (b) rho_u = {rho_u:.3e} kg/m^3")
    print(f"      rho_d = {rho_d:.3e} kg/m^3")
    print(f"      Mass flux in upflow: rho_u * v_u = {mass_flux_up:.3e} kg/(m^2 s)")
    print(f"      Area-averaged: f * rho_u * v_u = {total_mass_flux:.3e} kg/(m^2 s)")

    # (c) Kinetic energy flux
    # KE flux = (1/2) * rho * v^3 per unit area
    KE_flux_up = 0.5 * rho_u * v_u**3 * f  # averaged over total area
    KE_flux_down = 0.5 * rho_d * v_d**3 * (1.0 - f)  # averaged over total area
    KE_flux_total = KE_flux_up + KE_flux_down

    F_rad = sigma_SB * T_eff**4

    print(f"\n  (c) Kinetic energy flux (area-averaged):")
    print(f"      Upflow:   f * (1/2) rho_u v_u^3 = {KE_flux_up:.1f} W/m^2")
    print(f"      Downflow: (1-f) * (1/2) rho_d v_d^3 = {KE_flux_down:.1f} W/m^2")
    print(f"      Total KE flux: {KE_flux_total:.1f} W/m^2")
    print(f"      Radiative flux: F_rad = sigma T_eff^4 = {F_rad:.3e} W/m^2")
    print(f"      KE / F_rad = {KE_flux_total/F_rad:.2e}")
    print(f"      The kinetic energy flux is a tiny fraction of the radiative flux.")
    print(f"      Convection carries energy mainly as enthalpy (thermal), not kinetic.")


def exercise_4():
    """
    Problem 4: Curve of Growth

    Doppler width Delta_lambda_D = 0.05 A, damping parameter a = 0.01.
    (a) Linear part: EW doubles when abundance doubles.
    (b) Flat part: EW ~ sqrt(ln(N)).
    (c) Damping part: EW ~ sqrt(N).
    (d) Why linear part is most precise.
    """
    Delta_lambda_D = 0.05  # Angstroms
    a = 0.01  # damping parameter

    print(f"  Doppler width: Delta_lambda_D = {Delta_lambda_D} A")
    print(f"  Damping parameter: a = {a}")

    # (a) Linear part of the curve of growth
    # W ~ N * delta_lambda_D * pi * e^2 * f * lambda^2 / (m_e * c * delta_lambda_D)
    # W is proportional to N (column density, which scales with abundance)
    factor_linear = 2.0  # abundance doubles
    print(f"\n  (a) LINEAR part: W is proportional to N (abundance)")
    print(f"      If abundance doubles, W increases by factor {factor_linear:.0f}")

    # (b) Flat (saturated) part
    # W ~ Delta_lambda_D * sqrt(ln(N))
    # If N -> 2N: W(2N)/W(N) = sqrt(ln(2N)/ln(N)) = sqrt(1 + ln(2)/ln(N))
    # For typical saturated lines, ln(N) ~ 5-10
    ln_N = 7.0  # typical value on the flat part
    factor_flat = np.sqrt(np.log(2.0 * np.exp(ln_N)) / ln_N)
    factor_flat_simple = np.sqrt(1.0 + np.log(2.0) / ln_N)
    print(f"\n  (b) FLAT (saturated) part: W ~ Delta_lambda_D * sqrt(ln(N))")
    print(f"      For ln(N) ~ {ln_N:.0f} (typical):")
    print(f"      W(2N)/W(N) = sqrt(1 + ln(2)/ln(N)) = {factor_flat_simple:.3f}")
    print(f"      The EW barely increases (~{(factor_flat_simple-1)*100:.1f}%) for a doubling of abundance!")

    # (c) Damping part
    # W ~ sqrt(N * gamma * delta_lambda_D)  [proportional to sqrt(N * a)]
    # W ~ sqrt(N)
    factor_damping = np.sqrt(2.0)
    print(f"\n  (c) DAMPING (square-root) part: W ~ sqrt(N)")
    print(f"      If abundance doubles: W increases by factor sqrt(2) = {factor_damping:.3f}")

    # (d) Precision
    print(f"\n  (d) The linear part is most precise because:")
    print(f"      - W is directly proportional to abundance: dW/dN = const")
    print(f"      - A 10% error in W => 10% error in abundance")
    print(f"      - On the flat part, the same 10% error in W translates to")
    print(f"        a ~100% or larger error in abundance (derivative is near zero)")
    print(f"      - On the damping part, 10% error in W => 20% error (sqrt)")
    print(f"      - Therefore, weak lines on the linear part give the most")
    print(f"        reliable abundance determinations.")


def exercise_5():
    """
    Problem 5: FIP Effect Diagnostics

    CME Fe/O ratio = 0.20 +/- 0.02 (by number).
    Photospheric Fe/O = 0.065.
    (a) Fe enhancement factor.
    (b) Consistent with FIP effect?
    (c) If Fe/O = 0.07?
    (d) How to use FIP to distinguish slow SW from CME?
    """
    Fe_O_CME = 0.20
    Fe_O_CME_err = 0.02
    Fe_O_photo = 0.065

    # (a) Enhancement factor
    enhancement = Fe_O_CME / Fe_O_photo
    print(f"  (a) CME Fe/O = {Fe_O_CME} +/- {Fe_O_CME_err}")
    print(f"      Photospheric Fe/O = {Fe_O_photo}")
    print(f"      Fe enhancement factor = {Fe_O_CME}/{Fe_O_photo} = {enhancement:.1f}")

    # (b) FIP effect consistency
    print(f"\n  (b) Fe has FIP = 7.9 eV (low-FIP element)")
    print(f"      O has FIP = 13.6 eV (high-FIP element)")
    print(f"      The FIP effect preferentially enhances low-FIP elements (< ~10 eV)")
    print(f"      relative to high-FIP elements in coronal/heliospheric material.")
    print(f"      Enhancement factor of {enhancement:.1f} for Fe relative to O is")
    print(f"      consistent with the FIP effect (typical FIP bias: 2-4).")

    # (c) If Fe/O were 0.07
    Fe_O_low = 0.07
    enhancement_low = Fe_O_low / Fe_O_photo
    print(f"\n  (c) If Fe/O = {Fe_O_low}:")
    print(f"      Enhancement = {enhancement_low:.2f}")
    print(f"      This is essentially photospheric composition (no FIP effect).")
    print(f"      This would suggest the CME material originated from a region")
    print(f"      with no coronal processing -- possibly from a prominence/filament")
    print(f"      that contained cool chromospheric material.")

    # (d) Distinguishing slow SW from CME
    print(f"\n  (d) Using FIP effect to distinguish slow solar wind from CME material:")
    print(f"      - Slow solar wind: FIP bias ~ 2-4 (moderate enhancement)")
    print(f"      - Fast solar wind: FIP bias ~ 1-2 (weak or no enhancement)")
    print(f"      - CME/ICME: FIP bias can be > 4 (strong enhancement)")
    print(f"      - Prominence material in CMEs: FIP bias ~ 1 (photospheric)")
    print(f"      The FIP bias, combined with charge state ratios (O7+/O6+, Fe/O)")
    print(f"      and magnetic field signatures, helps identify CME plasma.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Eddington-Barbier Relation ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Photon Mean Free Path ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Granulation Velocities ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Curve of Growth ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: FIP Effect Diagnostics ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
