"""
Exercises for Lesson 11: Coronal Mass Ejections
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import integrate


# --- Physical constants ---
mu_0 = 4.0e-7 * np.pi  # vacuum permeability [T m/A]
R_sun = 6.957e8        # solar radius [m]
AU = 1.496e11          # astronomical unit [m]


def exercise_1():
    """
    Problem 1: CME Mass and Energy

    CME: v = 1200 km/s, mass = 5e12 kg.
    Internal B = 50 mT (50 nT at 1 AU), sphere of radius 2e10 cm.
    (a) Kinetic energy.
    (b) Magnetic energy.
    (c) Compare KE and magnetic energy at different distances.
    """
    v = 1200.0e3         # m/s
    m = 5.0e12           # kg
    B_corona = 50.0e-3   # T (50 mT) -- this seems high; likely 50 mG = 50e-4 T = 50 G
    # Re-reading: "B = 50 mT" in the problem. Let me check context...
    # "50 nT at 1 AU scaled back to the corona" -- scaling B ~ 1/r^2,
    # from 1 AU to ~10 R_sun: factor ~ (215/10)^2 ~ 460
    # 50 nT * 460 ~ 23000 nT ~ 23 uT ~ 0.23 G
    # The problem says 50 mT which is 500 G -- that seems like it should be 50 mG or 5 mT
    # Let's go with what's written but note the physics
    # Actually re-reading more carefully: "B = 50 mT (50 nT at 1 AU scaled back to the corona)"
    # 50 nT -> back to corona. If B ~ 1/r^2: B_corona = 50 nT * (1 AU / 10 R_sun)^2
    # But that gives too much. Let's use the problem as stated.
    B = 50.0e-3          # T (50 mT as stated)
    r_sphere = 2.0e10 * 0.01  # cm to m = 2e8 m

    # (a) Kinetic energy
    KE = 0.5 * m * v**2
    KE_erg = KE / 1.0e-7  # J to erg

    print(f"  (a) CME speed: v = {v/1e3:.0f} km/s")
    print(f"      CME mass: m = {m:.1e} kg")
    print(f"      Kinetic energy: KE = (1/2) m v^2")
    print(f"                        = {KE:.2e} J = {KE_erg:.2e} erg")

    # (b) Magnetic energy
    V_sphere = (4.0 / 3.0) * np.pi * r_sphere**3
    E_mag = (B**2 / (2.0 * mu_0)) * V_sphere
    E_mag_erg = E_mag / 1.0e-7

    print(f"\n  (b) Internal magnetic field: B = {B*1e3:.0f} mT")
    print(f"      Sphere radius: r = {r_sphere:.1e} m = {r_sphere/1e6:.0f} Mm")
    print(f"      Volume: V = {V_sphere:.2e} m^3")
    print(f"      Magnetic energy: E_mag = B^2 V / (2 mu_0)")
    print(f"                            = {E_mag:.2e} J = {E_mag_erg:.2e} erg")

    # (c) Comparison
    ratio = KE / E_mag
    print(f"\n  (c) KE / E_mag = {ratio:.1f}")
    print(f"      Near the Sun (low corona):")
    print(f"      - Magnetic energy dominates: the CME is magnetically driven")
    print(f"      - The magnetic pressure drives acceleration")
    print(f"      At 1 AU:")
    print(f"      - B has dropped by factor (r_corona/r_AU)^2 ~ {(r_sphere/AU)**2:.2e}")
    print(f"      - Magnetic energy drops much faster than KE")
    print(f"      - Kinetic energy dominates at large distances")
    print(f"      The transition from magnetically to kinetically dominated")
    print(f"      typically occurs within ~10-20 R_sun.")


def exercise_2():
    """
    Problem 2: Torus Instability

    B_ext(h) = B_0 * (h/h_0)^(-n), B_0 = 200 G, h_0 = 5e9 cm.
    (a) Critical decay index n_crit = 1.5.
    (b) Height of onset if n varies linearly with ln(h).
    """
    B_0 = 200.0       # G
    h_0 = 5.0e9       # cm
    n_crit = 1.5

    print(f"  (a) Torus instability occurs when the decay index n exceeds")
    print(f"      the critical value n_crit = {n_crit}")
    print(f"      where n = -d(ln B_ext)/d(ln h) = -(h/B_ext)(dB_ext/dh)")
    print(f"      For B_ext ~ h^(-n): the decay index IS n itself.")
    print(f"      Instability onset: n >= n_crit = {n_crit}")

    # (b) n varies linearly with ln(h)
    # n(h) = n_low + (n_high - n_low) * (ln(h) - ln(h_low)) / (ln(h_high) - ln(h_low))
    # Given: n = 1.0 at low heights, n = 2.0 at h = 1.5e10 cm
    n_low = 1.0
    n_high = 2.0
    h_low = h_0  # assume n_low at h_0
    h_high = 1.5e10  # cm

    # Linear in ln(h): n(h) = n_low + (n_high - n_low) * (ln(h) - ln(h_low)) / (ln(h_high) - ln(h_low))
    # Onset: n(h_onset) = n_crit
    # n_crit = n_low + (n_high - n_low) * (ln(h_onset) - ln(h_low)) / (ln(h_high) - ln(h_low))
    # (n_crit - n_low) / (n_high - n_low) = (ln(h_onset) - ln(h_low)) / (ln(h_high) - ln(h_low))
    # ln(h_onset) = ln(h_low) + (n_crit - n_low)/(n_high - n_low) * (ln(h_high) - ln(h_low))

    frac = (n_crit - n_low) / (n_high - n_low)
    ln_h_onset = np.log(h_low) + frac * (np.log(h_high) - np.log(h_low))
    h_onset = np.exp(ln_h_onset)
    h_onset_Mm = h_onset * 0.01 / 1.0e6  # cm -> m -> Mm

    print(f"\n  (b) n varies linearly with ln(h):")
    print(f"      n = {n_low} at h = {h_low:.1e} cm")
    print(f"      n = {n_high} at h = {h_high:.1e} cm")
    print(f"      At n_crit = {n_crit}:")
    print(f"      Fraction = (n_crit - n_low)/(n_high - n_low) = {frac:.2f}")
    print(f"      ln(h_onset) = ln({h_low:.1e}) + {frac} * [ln({h_high:.1e}) - ln({h_low:.1e})]")
    print(f"      h_onset = {h_onset:.2e} cm = {h_onset_Mm:.0f} Mm")
    print(f"      The instability onset occurs at approximately {h_onset_Mm:.0f} Mm above")
    print(f"      the photosphere, consistent with typical eruption initiation heights.")


def exercise_3():
    """
    Problem 3: Drag-Based Model

    v(r) = v_sw + (v0 - v_sw) * exp(-gamma*(r - r0))
    v0 = 2000 km/s, v_sw = 400 km/s, gamma = 0.5e-7 km^-1.
    r0 = 20 R_sun, 1 AU = 215 R_sun.
    (a) Speed at 1 AU.
    (b) Transit time.
    """
    v0 = 2000.0        # km/s
    v_sw = 400.0        # km/s
    gamma = 0.5e-7      # km^-1
    r0_Rsun = 20.0
    r0 = r0_Rsun * R_sun / 1e3  # km
    r_AU = 215.0 * R_sun / 1e3  # km (1 AU in km)

    # (a) Speed at 1 AU
    dr = r_AU - r0
    v_1AU = v_sw + (v0 - v_sw) * np.exp(-gamma * dr)

    print(f"  Parameters:")
    print(f"    v0 = {v0:.0f} km/s (initial)")
    print(f"    v_sw = {v_sw:.0f} km/s (ambient wind)")
    print(f"    gamma = {gamma:.1e} km^-1 (drag coefficient)")
    print(f"    r0 = {r0_Rsun:.0f} R_sun = {r0:.0f} km")
    print(f"    r(1 AU) = 215 R_sun = {r_AU:.0f} km")
    print(f"    dr = {dr:.3e} km")

    print(f"\n  (a) v(1 AU) = v_sw + (v0 - v_sw) * exp(-gamma * dr)")
    print(f"             = {v_sw} + ({v0} - {v_sw}) * exp(-{gamma:.1e} * {dr:.3e})")
    print(f"             = {v_sw} + {v0 - v_sw} * exp(-{gamma * dr:.2f})")
    print(f"             = {v_sw} + {v0 - v_sw} * {np.exp(-gamma * dr):.4f}")
    print(f"             = {v_1AU:.0f} km/s")

    # (b) Transit time
    # v(r) = v_sw + (v0 - v_sw) * exp(-gamma*(r - r0))
    # dt = dr / v(r)
    # t = integral from r0 to r_AU of dr / v(r)

    # Numerical integration
    def integrand(r):
        v = v_sw + (v0 - v_sw) * np.exp(-gamma * (r - r0))
        return 1.0 / v

    t_transit_s, _ = integrate.quad(integrand, r0, r_AU)
    # t is in s/km * km = s? No: dr is in km, v is in km/s, so dt = dr/v is in s
    t_transit_hr = t_transit_s / 3600.0

    print(f"\n  (b) Transit time: t = integral dr / v(r)")
    print(f"      Numerical integration: t = {t_transit_s:.0f} s")
    print(f"                            = {t_transit_hr:.1f} hours")
    print(f"                            = {t_transit_hr/24:.1f} days")

    # Simple estimate: use average speed
    v_avg = (v0 + v_1AU) / 2.0  # rough average
    t_simple = dr / v_avg / 3600.0  # hours
    print(f"\n      Simple estimate (average speed): {t_simple:.1f} hours")
    print(f"      The fast CME (initially 2000 km/s) is decelerated by drag")
    print(f"      to ~{v_1AU:.0f} km/s by 1 AU, with a transit time of ~{t_transit_hr:.0f} hours.")


def exercise_4():
    """
    Problem 4: Magnetic Cloud Fitting

    B_z rotates from +15 nT to -20 nT to +5 nT over 24 hours.
    (a) Sketch expected B_z for Lundquist rope near axis.
    (b) Estimate impact parameter from asymmetry.
    (c) Chirality from B_y rotation.
    """
    print(f"  (a) For a Lundquist flux rope (force-free, cylindrically symmetric),")
    print(f"      passing near the axis:")
    print(f"      B_z shows smooth rotation from positive to negative (or vice versa)")
    print(f"      The profile is approximately sinusoidal: B_z ~ B_0 * J_0(alpha*r)")
    print(f"      For a near-axis crossing: peak positive -> zero crossing -> peak negative")
    print(f"      with a symmetric profile about the center.")
    print(f"      ")
    print(f"      Expected ideal profile (axial crossing):")
    print(f"          +B_max   ---____")
    print(f"                          ----___")
    print(f"               0  _______________----> time")
    print(f"                                     ___---")
    print(f"          -B_max              ____---")

    # (b) Impact parameter from asymmetry
    Bz_lead = 15.0      # nT (leading edge)
    Bz_min = -20.0       # nT (central minimum)
    Bz_trail = 5.0       # nT (trailing edge)

    print(f"\n  (b) Observed B_z: +{Bz_lead} nT -> {Bz_min} nT -> +{Bz_trail} nT")
    print(f"      Asymmetry: leading ({Bz_lead} nT) != trailing ({Bz_trail} nT)")
    print(f"      A symmetric crossing (through the axis) would give equal magnitude")
    print(f"      at entry and exit. The asymmetry indicates the spacecraft passed")
    print(f"      OFF-AXIS (non-zero impact parameter).")
    print(f"      ")
    # Rough estimate: the ratio of leading/trailing gives an idea of impact parameter
    # For a Lundquist rope, the asymmetry increases with impact parameter p/R
    ratio = abs(Bz_trail) / abs(Bz_lead)
    print(f"      |B_trailing| / |B_leading| = {abs(Bz_trail)}/{abs(Bz_lead)} = {ratio:.2f}")
    print(f"      For a Lundquist rope, this ratio decreases from 1 (axial) to 0")
    print(f"      as the impact parameter approaches the rope radius.")
    print(f"      An asymmetry ratio of {ratio:.2f} suggests a moderate impact parameter,")
    print(f"      roughly p/R ~ 0.3-0.5 (the spacecraft passed about 30-50% of the")
    print(f"      rope radius from the axis).")

    # (c) Chirality
    print(f"\n  (c) B_y rotates from negative to positive:")
    print(f"      In a flux rope, the handedness is determined by the relationship")
    print(f"      between the axial field (B along the rope axis) and the twist.")
    print(f"      If B_y goes from negative to positive (south to north) while")
    print(f"      B_z goes from positive to negative (east to west),")
    print(f"      this corresponds to a LEFT-HANDED (negative) helicity rope.")
    print(f"      Chirality convention: negative helicity = sinistral.")
    print(f"      This is consistent with the northern hemisphere helicity rule")
    print(f"      (negative helicity in the northern hemisphere).")


def exercise_5():
    """
    Problem 5: Geo-effectiveness

    ICME-A: v=700 km/s, Bz=-15 nT, 6 hours.
    ICME-B: v=500 km/s, Bz=-25 nT, 4 hours.
    (a) v*Bs for each.
    (b) Dst_min ~ -v * Bs * dt^0.5.
    (c) Discussion.
    """
    # ICME parameters
    v_A, Bz_A, dt_A = 700.0, -15.0, 6.0   # km/s, nT, hours
    v_B, Bz_B, dt_B = 500.0, -25.0, 4.0

    Bs_A = abs(Bz_A)  # southward component [nT]
    Bs_B = abs(Bz_B)

    # (a) v * Bs
    vBs_A = v_A * Bs_A  # (km/s)(nT)
    vBs_B = v_B * Bs_B

    print(f"  ICME-A: v = {v_A:.0f} km/s, Bz = {Bz_A:.0f} nT, dt = {dt_A:.0f} hr")
    print(f"  ICME-B: v = {v_B:.0f} km/s, Bz = {Bz_B:.0f} nT, dt = {dt_B:.0f} hr")

    print(f"\n  (a) v * Bs:")
    print(f"      ICME-A: {v_A:.0f} * {Bs_A:.0f} = {vBs_A:.0f} (km/s)(nT)")
    print(f"      ICME-B: {v_B:.0f} * {Bs_B:.0f} = {vBs_B:.0f} (km/s)(nT)")

    # (b) Dst_min ~ -v * Bs * dt^0.5  (proportional to)
    Dst_proxy_A = vBs_A * np.sqrt(dt_A)
    Dst_proxy_B = vBs_B * np.sqrt(dt_B)

    print(f"\n  (b) Dst_min proxy ~ v * Bs * dt^0.5:")
    print(f"      ICME-A: {vBs_A:.0f} * sqrt({dt_A:.0f}) = {vBs_A:.0f} * {np.sqrt(dt_A):.2f}")
    print(f"             = {Dst_proxy_A:.0f}")
    print(f"      ICME-B: {vBs_B:.0f} * sqrt({dt_B:.0f}) = {vBs_B:.0f} * {np.sqrt(dt_B):.2f}")
    print(f"             = {Dst_proxy_B:.0f}")

    if Dst_proxy_A > Dst_proxy_B:
        print(f"\n      ICME-A produces a LARGER Dst minimum (more intense storm).")
    else:
        print(f"\n      ICME-B produces a LARGER Dst minimum (more intense storm).")

    ratio = Dst_proxy_A / Dst_proxy_B
    print(f"      Ratio (A/B): {ratio:.2f}")

    # (c) Discussion
    print(f"\n  (c) The simple v*Bs scaling does NOT capture all physics:")
    print(f"      - Preconditioning: if ICME-A arrives into an already disturbed")
    print(f"        magnetosphere, the effect can be amplified.")
    print(f"      - Sheath fields: the shock sheath ahead of the ICME often has")
    print(f"        strong, turbulent Bz that can be geo-effective.")
    print(f"      - Duration matters: longer southward Bz (ICME-A, 6 hr) allows")
    print(f"        more energy injection (ring current buildup).")
    print(f"      - Dynamic pressure: higher v means stronger magnetopause compression.")
    print(f"      - Reconnection efficiency depends on B, v, and density together.")
    print(f"      - Recovery time (Dst decay) is not included in this proxy.")
    print(f"      More sophisticated empirical models (e.g., Burton, O'Brien-McPherron)")
    print(f"      integrate the driving function over time with a decay term.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: CME Mass and Energy ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Torus Instability ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Drag-Based Model ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Magnetic Cloud Fitting ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Geo-effectiveness ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
