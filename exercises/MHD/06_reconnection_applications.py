"""
Exercises for Lesson 06: Reconnection Applications
Topic: MHD
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: Solar Flare Energetics

    B = 0.02 T, V = (1e8 m)^3, 20% released in 1000 s
    """
    B = 0.02       # T
    L_side = 1e8   # m
    V = L_side**3  # m^3
    f_release = 0.2
    tau = 1000.0   # s
    mu0 = 4 * np.pi * 1e-7
    L_sun = 3.8e26  # W

    # (a) Magnetic energy
    E_mag = B**2 * V / (2 * mu0)
    print(f"(a) Magnetic energy in active region:")
    print(f"    E_mag = B^2 * V / (2*mu0)")
    print(f"    = {B}^2 * {V:.2e} / (2 * {mu0:.4e})")
    print(f"    = {E_mag:.4e} J")
    E_mag_erg = E_mag * 1e7  # J to erg
    print(f"    = {E_mag_erg:.4e} erg")

    # (b) Average power
    E_released = f_release * E_mag
    P = E_released / tau
    print(f"\n(b) Released energy (20%): {E_released:.4e} J = {E_released*1e7:.4e} erg")
    print(f"    Average power: P = {P:.4e} W")
    print(f"    = {P*1e7:.4e} erg/s")

    # (c) Compare to solar luminosity
    ratio = P / L_sun
    print(f"\n(c) Solar luminosity: L_sun = {L_sun:.2e} W")
    print(f"    P / L_sun = {ratio:.4e}")
    print(f"    Flare power is ~{ratio:.1e} of total solar luminosity")
    print(f"    (X-class flares can briefly reach ~{ratio*10:.0e} of L_sun)")


def exercise_2():
    """
    Problem 2: Flare Ribbon Motion

    v_sep = 50 km/s, h = 1e7 m, d = 1e8 m, v_A = 1000 km/s
    """
    v_sep = 50e3    # m/s (ribbon separation speed)
    h = 1e7         # m (coronal height)
    d = 1e8         # m (footpoint separation)
    v_A = 1000e3    # m/s

    # (a) Reconnection inflow speed
    # The ribbon separation speed relates to the reconnection rate:
    # v_in ~ v_sep * (h/d) (geometry of field lines)
    # More directly: the rate of flux reconnection is:
    # dPhi/dt = v_sep * B_footpoint * L_ribbon
    # The inflow speed at the coronal X-point:
    # v_in ~ v_sep (the ribbon motion maps to the inflow at the X-point through
    # the magnetic field geometry)
    # For a simple 2D model: v_in ~ v_sep * (Bf / Bc) where Bf/Bc is the
    # field compression ratio. In practice, v_in ~ v_sep for order of magnitude.
    v_in = v_sep
    print(f"(a) Reconnection inflow speed:")
    print(f"    Ribbon separation speed: v_sep = {v_sep/1e3:.0f} km/s")
    print(f"    In a 2D reconnection geometry, the ribbon motion maps to")
    print(f"    the coronal inflow speed: v_in ~ v_sep = {v_in/1e3:.0f} km/s")
    print(f"    (Geometric correction factor depends on field topology)")

    # (b) M_A
    M_A = v_in / v_A
    print(f"\n(b) M_A = v_in / v_A = {v_in/1e3:.0f} / {v_A/1e3:.0f} = {M_A:.3f}")

    # (c) Comparison to models
    print(f"\n(c) M_A = {M_A:.3f}")
    print(f"    Sweet-Parker: M_A ~ S^(-1/2) << 0.01 for solar S ~ 10^14")
    print(f"    Petschek: M_A ~ pi/(8*ln(S)) ~ 0.01 for S ~ 10^14")
    print(f"    Hall: M_A ~ 0.1")
    print(f"    Measured value {M_A:.3f} is consistent with Petschek/Hall range")
    print(f"    This fast rate rules out Sweet-Parker as the sole mechanism")


def exercise_3():
    """
    Problem 3: CME Kinetic Energy

    M = 1e15 g, v = 1000 km/s, decelerated to 500 km/s
    """
    M = 1e15 * 1e-3  # g to kg = 1e12 kg
    v1 = 1000e3       # m/s
    v2 = 500e3        # m/s

    # (a) Kinetic energy
    KE1 = 0.5 * M * v1**2
    KE1_erg = KE1 * 1e7
    print(f"(a) CME kinetic energy:")
    print(f"    M = {M:.2e} kg")
    print(f"    v = {v1/1e3:.0f} km/s")
    print(f"    KE = 0.5 * M * v^2 = {KE1:.4e} J")
    print(f"    = {KE1_erg:.4e} erg")

    # (b) Energy dissipated
    KE2 = 0.5 * M * v2**2
    delta_KE = KE1 - KE2
    delta_KE_erg = delta_KE * 1e7
    print(f"\n(b) After deceleration to {v2/1e3:.0f} km/s:")
    print(f"    KE_final = {KE2:.4e} J")
    print(f"    Energy dissipated: {delta_KE:.4e} J = {delta_KE_erg:.4e} erg")
    print(f"    Fraction dissipated: {delta_KE/KE1:.1%}")

    # (c) Where does the energy go?
    print(f"\n(c) Dissipated energy goes to:")
    print(f"    1. Solar wind heating: Compression and shock heating of")
    print(f"       ambient solar wind plasma ahead of the CME")
    print(f"    2. Wave generation: MHD waves (shocks, Alfven waves)")
    print(f"       propagating away from the CME front")
    print(f"    3. Particle acceleration: Energetic particles at the")
    print(f"       CME-driven shock (SEP events)")
    print(f"    4. Magnetic field compression: Piling up of interplanetary")
    print(f"       magnetic field ahead of the CME (sheath region)")


def exercise_4():
    """
    Problem 4: Torus Instability

    Criterion: d(ln B_ext)/d(ln h) < -3/2
    Dipole field B ~ r^-3
    """
    print("(a) Torus instability criterion:")
    print("    A current-carrying flux rope in an external field B_ext(h)")
    print("    experiences a hoop force (outward) balanced by the strapping")
    print("    field (inward). If the field decays too fast with height,")
    print("    the balance is lost and the rope erupts.")
    print("    Critical condition: n = -d(ln B_ext)/d(ln h) > n_crit = 3/2")
    print("    where n is the decay index.")
    print("    When n > 3/2, the flux rope is UNSTABLE to eruption.")

    # (b) Dipole field
    # B ~ r^-3
    # d(ln B)/d(ln r) = d(ln r^-3)/d(ln r) = -3
    n_dipole = 3.0
    print(f"\n(b) For dipole field B ~ r^(-3):")
    print(f"    d(ln B)/d(ln r) = -3")
    print(f"    Decay index n = -d(ln B)/d(ln h) = {n_dipole}")

    # (c) Stability
    n_crit = 1.5
    print(f"\n(c) n = {n_dipole} > n_crit = {n_crit}")
    print(f"    The dipole field is UNSTABLE to torus instability")
    print(f"    This means a flux rope at any height in a pure dipole")
    print(f"    field would erupt. In practice, non-dipole components")
    print(f"    (quadrupole, etc.) create regions where n < 3/2 near")
    print(f"    the surface, confining the flux rope until it rises")
    print(f"    to a critical height where n exceeds 3/2.")

    # Plot decay index for different field geometries
    h = np.linspace(0.5, 5, 100)  # normalized height

    n_dipole_arr = 3.0 * np.ones_like(h)
    n_quadrupole = 4.0 * np.ones_like(h)
    # Realistic: transition from ~1 to ~3 with height
    n_realistic = 1.0 + 2.0 / (1 + np.exp(-(h - 2.0) / 0.5))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(h, n_dipole_arr, 'b-', linewidth=2, label='Dipole ($r^{-3}$)')
    ax.plot(h, n_quadrupole, 'g-', linewidth=2, label='Quadrupole ($r^{-4}$)')
    ax.plot(h, n_realistic, 'r-', linewidth=2, label='Realistic (mixed)')
    ax.axhline(y=1.5, color='k', linestyle='--', linewidth=2, label='$n_{crit} = 3/2$')
    ax.fill_between(h, 1.5, 5, alpha=0.1, color='red', label='Unstable region')
    ax.set_xlabel('Height h (normalized)', fontsize=12)
    ax.set_ylabel('Decay index n', fontsize=12)
    ax.set_title('Torus Instability: Decay Index vs Height', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)
    plt.tight_layout()
    plt.savefig('/tmp/ex06_torus_instability.png', dpi=100)
    plt.close()
    print("    Plot saved to /tmp/ex06_torus_instability.png")


def exercise_5():
    """
    Problem 5: Dungey Cycle Timescale

    v_in = 100 km/s, B_sw = 5 nT, L_y = 20 R_E, Phi_tail ~ 1 GWb
    """
    v_in = 100e3     # m/s
    B_sw = 5e-9      # T
    R_E = 6.4e6      # m
    L_y = 20 * R_E   # m
    Phi_tail = 1e9   # Wb (1 GWb)

    # (a) Flux transfer rate
    E_rec = v_in * B_sw
    dPhi_dt = E_rec * L_y
    print(f"(a) Reconnection electric field: E_rec = v_in * B_sw")
    print(f"    = {v_in/1e3:.0f} km/s * {B_sw*1e9:.0f} nT = {E_rec:.4e} V/m")
    print(f"    = {E_rec*1e3:.2f} mV/m")
    print(f"    Flux transfer rate: dPhi/dt = E_rec * L_y")
    print(f"    = {E_rec:.4e} * {L_y:.4e}")
    print(f"    = {dPhi_dt:.4e} Wb/s = {dPhi_dt/1e3:.2f} kV")

    # (b) Time to load tail flux
    tau_load = Phi_tail / dPhi_dt
    print(f"\n(b) Total tail lobe flux: Phi_tail = {Phi_tail/1e9:.0f} GWb")
    print(f"    Loading time: tau = Phi_tail / (dPhi/dt)")
    print(f"    = {Phi_tail:.2e} / {dPhi_dt:.4e}")
    print(f"    = {tau_load:.0f} s = {tau_load/60:.1f} min")

    # (c) Compare to substorm growth phase
    print(f"\n(c) Observed substorm growth phase: ~30-60 min")
    print(f"    Computed loading time: {tau_load/60:.1f} min")
    if 20 < tau_load / 60 < 120:
        print(f"    CONSISTENT with observed growth phase duration")
    else:
        print(f"    Not directly matching (depends on actual conditions)")
    print(f"    The Dungey cycle time is roughly the time to open and")
    print(f"    close all the magnetic flux in the tail lobes.")


def exercise_6():
    """
    Problem 6: Substorm Energy Release

    B = 20 nT, V ~ (10 R_E)^3, released over 1 hour
    """
    B = 20e-9       # T
    R_E = 6.4e6     # m
    L_tail = 10 * R_E
    V = L_tail**3   # m^3
    tau = 3600       # s (1 hour)
    mu0 = 4 * np.pi * 1e-7

    # (a) Stored energy
    E_stored = B**2 * V / (2 * mu0)
    print(f"(a) Magnetotail stored energy:")
    print(f"    E = B^2 * V / (2*mu0)")
    print(f"    B = {B*1e9:.0f} nT = {B:.1e} T")
    print(f"    V = (10*R_E)^3 = ({L_tail:.2e})^3 = {V:.4e} m^3")
    print(f"    E = {E_stored:.4e} J")

    # (b) Average power
    P = E_stored / tau
    print(f"\n(b) Average power over {tau/3600:.0f} hour:")
    print(f"    P = E/tau = {P:.4e} W")
    print(f"    = {P/1e9:.2f} GW")

    # (c) Compare to solar wind input
    P_sw_low = 1e11  # W
    P_sw_high = 1e12  # W
    print(f"\n(c) Solar wind input power: ~{P_sw_low:.0e} - {P_sw_high:.0e} W")
    print(f"    Substorm power: {P:.2e} W")
    print(f"    Ratio (low): P_sub / P_sw = {P/P_sw_low:.2f}")
    print(f"    Ratio (high): P_sub / P_sw = {P/P_sw_high:.4f}")
    print(f"    Substorm energy release is comparable to or exceeds")
    print(f"    instantaneous solar wind input, consistent with stored")
    print(f"    energy being released explosively.")


def exercise_7():
    """
    Problem 7: Sawtooth Crash Time

    tau_crash ~ 50 us, a = 0.5 m, B = 3 T, n = 1e20 m^-3
    """
    tau_crash = 50e-6   # s
    a = 0.5             # m
    B = 3.0             # T
    n = 1e20            # m^-3
    mu0 = 4 * np.pi * 1e-7
    mp = 1.67e-27

    # (a) Given parameters
    print(f"(a) Sawtooth crash parameters:")
    print(f"    tau_crash = {tau_crash*1e6:.0f} us")
    print(f"    a = {a} m, B = {B} T, n = {n:.0e} m^-3")

    # (b) Alfven time
    rho = n * mp
    v_A = B / np.sqrt(mu0 * rho)
    tau_A = a / v_A
    print(f"\n(b) rho = n*mp = {rho:.4e} kg/m^3")
    print(f"    v_A = B/sqrt(mu0*rho) = {v_A:.4e} m/s")
    print(f"    tau_A = a/v_A = {tau_A:.4e} s = {tau_A*1e6:.2f} us")

    # (c) Reconnection rate
    M_A = tau_A / tau_crash
    print(f"\n(c) Reconnection rate estimate:")
    print(f"    M_A ~ tau_A / tau_crash = {tau_A:.4e} / {tau_crash:.1e}")
    print(f"    = {M_A:.4f}")
    print(f"    This is consistent with fast reconnection (M_A ~ 0.01-0.1)")
    print(f"    The sawtooth crash timescale is ~{1/M_A:.0f} Alfven times")


def exercise_8():
    """
    Problem 8: Island Coalescence

    w = 5 cm, d = 10 cm, v_A = 1e6 m/s, approach at 0.1*v_A
    """
    w = 0.05         # m
    d = 0.10         # m
    v_A = 1e6        # m/s
    v_approach = 0.1 * v_A

    # (a) Parameters
    print(f"(a) Two magnetic islands:")
    print(f"    Width: w = {w*100:.0f} cm")
    print(f"    Separation: d = {d*100:.0f} cm")
    print(f"    Alfven speed: v_A = {v_A:.1e} m/s")

    # (b) Merging time
    # Gap between islands: d - w = 5 cm (distance between edges)
    gap = d - w
    tau_merge = gap / v_approach
    print(f"\n(b) Approach speed: v = 0.1*v_A = {v_approach:.1e} m/s")
    print(f"    Gap between island edges: d - w = {gap*100:.0f} cm")
    print(f"    Time to merge: tau = gap / v = {gap} / {v_approach:.1e}")
    print(f"    = {tau_merge:.4e} s = {tau_merge*1e6:.2f} us")

    # (c) Energy fraction released
    # Before: two islands, each with energy ~ w^2 * B^2
    # After: one island with w_final = sqrt(2) * w (flux conservation)
    w_final = np.sqrt(2) * w
    # Energy ~ w^2 (for fixed B), so E_before ~ 2*w^2, E_after ~ w_final^2 = 2*w^2
    # Actually energy goes as w^2 * B^2, and B inside island is roughly preserved
    # The released energy comes from the current sheet between the islands
    print(f"\n(c) Energy released during coalescence:")
    print(f"    Before: 2 islands, each ~E_island ~ w^2*B^2 (proportional)")
    print(f"    After: 1 island, w_final = sqrt(2)*w = {w_final*100:.2f} cm")
    print(f"    E_before ~ 2*w^2, E_after ~ w_final^2 = 2*w^2")
    print(f"    In this simple scaling, total magnetic energy is conserved.")
    print(f"    However, the current sheet between islands releases energy:")
    E_current_sheet_fraction = 0.5  # rough estimate
    print(f"    The current sheet energy (fraction of island energy) is")
    print(f"    released as kinetic energy and heating.")
    print(f"    Typical fraction released: ~10-50% of inter-island magnetic energy")
    print(f"    Key point: coalescence is a fast process (Alfvenic timescale)")
    print(f"    and can trigger further reconnection/instability cascades.")


def exercise_9():
    """
    Problem 9: AGN Jet Power

    R = 1e16 m, v = 0.5c, B = 1 G = 1e-4 T
    M_BH = 1e9 M_sun
    """
    R = 1e16        # m
    c = 3e8         # m/s
    v = 0.5 * c     # m/s
    B = 1e-4        # T (1 Gauss)
    mu0 = 4 * np.pi * 1e-7
    M_sun = 2e30    # kg
    M_BH = 1e9 * M_sun
    G = 6.674e-11   # m^3/(kg s^2)
    sigma_T = 6.65e-29  # m^2 (Thomson cross-section)
    mp = 1.67e-27

    # (a) Poynting flux
    S = B**2 * v / mu0
    print(f"(a) AGN jet parameters:")
    print(f"    R = {R:.1e} m, v = 0.5c = {v:.1e} m/s, B = {B*1e4:.0f} G = {B} T")
    print(f"    Poynting flux: S = B^2 * v / mu0")
    print(f"    = {B}^2 * {v:.1e} / {mu0:.4e}")
    print(f"    = {S:.4e} W/m^2")

    # (b) Jet power
    A = np.pi * R**2
    P_jet = S * A
    P_jet_erg = P_jet * 1e7  # to erg/s
    print(f"\n(b) Jet cross-section: A = pi*R^2 = {A:.4e} m^2")
    print(f"    Jet power: P = S * A = {P_jet:.4e} W")
    print(f"    = {P_jet_erg:.4e} erg/s")

    # (c) Eddington luminosity
    L_Edd = 4 * np.pi * G * M_BH * mp * c / sigma_T
    L_Edd_erg = L_Edd * 1e7
    print(f"\n(c) Eddington luminosity for {M_BH/M_sun:.0e} M_sun:")
    print(f"    L_Edd = 4*pi*G*M*mp*c/sigma_T")
    print(f"    = {L_Edd:.4e} W = {L_Edd_erg:.4e} erg/s")
    ratio = P_jet / L_Edd
    print(f"    P_jet / L_Edd = {ratio:.4e}")
    if ratio > 1:
        print(f"    Jet power EXCEEDS Eddington luminosity")
        print(f"    (Super-Eddington jets possible via magnetic launching)")
    else:
        print(f"    Jet power is {ratio*100:.1f}% of Eddington luminosity")


def exercise_10():
    """
    Problem 10: Relativistic Reconnection

    Pulsar wind sigma = 10^3 near light cylinder
    50% magnetic energy converted to kinetic
    """
    sigma_init = 1e3
    f_convert = 0.5

    # (a) Parameters
    print(f"(a) Pulsar wind magnetization:")
    print(f"    sigma = B^2 / (mu0 * rho * c^2) = {sigma_init:.0e}")
    print(f"    This means magnetic energy >> particle rest mass energy")
    print(f"    The plasma is magnetically dominated")

    # (b) Final sigma after reconnection
    # Initially: E_mag = sigma * E_rest, E_kin ~ 0 (approximately)
    # Total energy: E_tot = E_rest + E_mag = E_rest * (1 + sigma)
    # After 50% conversion: E_mag_final = 0.5 * E_mag = 0.5 * sigma * E_rest
    # E_kin_final = 0.5 * E_mag = 0.5 * sigma * E_rest
    # sigma_final = E_mag_final / E_rest = 0.5 * sigma
    # But actually sigma is defined relative to *total* particle energy including kinetic:
    # sigma_final = B'^2 / (mu0 * rho * c^2 * Gamma)
    # Simple estimate: if half the magnetic energy goes to particles,
    # the particle energy increases, so sigma_final ~ sigma / (1 + 0.5*sigma) for large sigma
    # ~ 2 for sigma >> 1

    # More careful: sigma = E_B / E_particle
    # E_B_final = (1 - f) * E_B_init
    # E_particle_final = E_particle_init + f * E_B_init = E_rest + f * sigma_init * E_rest
    # sigma_final = (1-f)*sigma_init*E_rest / (E_rest + f*sigma_init*E_rest)
    # = (1-f)*sigma_init / (1 + f*sigma_init)
    sigma_final = (1 - f_convert) * sigma_init / (1 + f_convert * sigma_init)
    print(f"\n(b) After {f_convert*100:.0f}% conversion:")
    print(f"    sigma_final = (1-f)*sigma / (1 + f*sigma)")
    print(f"    = {1-f_convert}*{sigma_init:.0e} / (1 + {f_convert}*{sigma_init:.0e})")
    print(f"    = {sigma_final:.4f}")
    print(f"    ~ 1 (magnetic and particle energies roughly equal)")

    # (c) Comparison to observations
    sigma_obs_low = 0.01
    sigma_obs_high = 0.1
    print(f"\n(c) Observed sigma at termination shock: {sigma_obs_low} - {sigma_obs_high}")
    print(f"    Reconnection gives sigma_final ~ {sigma_final:.2f}")
    if sigma_final > sigma_obs_high:
        print(f"    NOT sufficient: sigma_final ({sigma_final:.2f}) >> sigma_obs ({sigma_obs_high})")
        print(f"    Additional dissipation mechanisms needed:")
        print(f"    1. Turbulent cascade: MHD turbulence further converts")
        print(f"       magnetic energy to thermal")
        print(f"    2. Kink instability: Destroys the ordered field structure")
        print(f"    3. Multiple reconnection events during propagation")
        print(f"    4. Interaction with the supernova remnant (termination shock)")
        print(f"    This is the 'sigma problem' of pulsar wind physics.")
    else:
        print(f"    Sufficient to explain observations")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Solar Flare Energetics ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Flare Ribbon Motion ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: CME Kinetic Energy ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Torus Instability ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Dungey Cycle Timescale ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Substorm Energy Release ===")
    print("=" * 60)
    exercise_6()

    print("\n" + "=" * 60)
    print("=== Exercise 7: Sawtooth Crash Time ===")
    print("=" * 60)
    exercise_7()

    print("\n" + "=" * 60)
    print("=== Exercise 8: Island Coalescence ===")
    print("=" * 60)
    exercise_8()

    print("\n" + "=" * 60)
    print("=== Exercise 9: AGN Jet Power ===")
    print("=" * 60)
    exercise_9()

    print("\n" + "=" * 60)
    print("=== Exercise 10: Relativistic Reconnection ===")
    print("=" * 60)
    exercise_10()

    print("\nAll exercises completed!")
