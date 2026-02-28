"""
Exercise Solutions for Lesson 11: Geomagnetically Induced Currents (GIC)

Topics covered:
  - Geoelectric field in a uniform half-space (skin depth, impedance)
  - GIC in a simple two-node power grid network
  - Transformer saturation from GIC
  - Pipeline pipe-to-soil potential (PSP)
  - Carrington-class risk assessment
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Geoelectric Field in Uniform Half-Space

    B0 = 500 nT, T = 300 s, sigma = 1e-3 S/m.
    (a) Angular frequency omega.
    (b) Skin depth delta.
    (c) Geoelectric field amplitude.
    (d) Factor change if sigma decreases to 1e-4 S/m.
    (e) Why resistive ground is more vulnerable.
    """
    print("=" * 70)
    print("Exercise 1: Geoelectric Field in Uniform Half-Space")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7
    B0 = 500e-9    # T
    T = 300        # s
    sigma = 1e-3   # S/m

    # (a) Angular frequency
    omega = 2 * np.pi / T

    print(f"\n    B0 = {B0*1e9:.0f} nT, T = {T} s, sigma = {sigma:.0e} S/m")

    print(f"\n(a) Angular frequency:")
    print(f"    omega = 2*pi/T = 2*pi/{T} = {omega:.4f} rad/s")

    # (b) Skin depth
    delta = np.sqrt(2 / (omega * mu0 * sigma))

    print(f"\n(b) Skin depth:")
    print(f"    delta = sqrt(2 / (omega * mu0 * sigma))")
    print(f"    = sqrt(2 / ({omega:.4f} * {mu0:.3e} * {sigma:.0e}))")
    print(f"    = {delta:.0f} m = {delta/1e3:.1f} km")

    # (c) Geoelectric field amplitude
    # |E| = sqrt(omega / (2*mu0*sigma)) * B0
    # Or equivalently: |E| = (omega * delta / 2) * B0 / (mu0 * sigma * delta)
    # Simpler: |E| = B0 / (mu0 * sigma * delta)
    # = B0 * sqrt(omega / (2*mu0*sigma))
    E0 = np.sqrt(omega / (2 * mu0 * sigma)) * B0

    # Also: E0 = omega * B0 * delta / 2
    E0_alt = B0 / (mu0 * sigma * delta)

    print(f"\n(c) Geoelectric field amplitude:")
    print(f"    |E| = sqrt(omega/(2*mu0*sigma)) * B0")
    print(f"    = sqrt({omega:.4f} / (2*{mu0:.3e}*{sigma:.0e})) * {B0:.1e}")
    print(f"    = {E0:.4e} V/m = {E0*1e3:.2f} mV/m = {E0*1e3:.2f} V/km")

    # (d) Resistive ground: sigma = 1e-4 S/m
    sigma2 = 1e-4
    delta2 = np.sqrt(2 / (omega * mu0 * sigma2))
    E0_2 = np.sqrt(omega / (2 * mu0 * sigma2)) * B0
    factor = E0_2 / E0

    print(f"\n(d) Resistive ground (sigma = {sigma2:.0e} S/m):")
    print(f"    delta = {delta2:.0f} m = {delta2/1e3:.1f} km")
    print(f"    |E| = {E0_2:.4e} V/m = {E0_2*1e3:.2f} V/km")
    print(f"    Factor increase: {factor:.2f}")
    print(f"    E scales as 1/sqrt(sigma), so 10x lower sigma -> {np.sqrt(10):.1f}x higher E")

    # (e) Discussion
    print(f"\n(e) Why resistive ground is more vulnerable:")
    print(f"    - Lower conductivity -> larger skin depth -> deeper current penetration")
    print(f"    - The geoelectric field is inversely proportional to sqrt(sigma)")
    print(f"    - Resistive regions (Canadian Shield, Scandinavian bedrock)")
    print(f"      experience 3-10x larger geoelectric fields than conductive regions")
    print(f"    - This explains why Quebec (resistive shield rock) and Finland/Sweden")
    print(f"      are among the most GIC-vulnerable regions")


def exercise_2():
    """
    Exercise 2: GIC in Two-Node Network

    Two substations A-B, 200 km east-west. R_line = 4 ohm, R_g = 0.5 ohm each.
    Uniform E_x = 2 V/km eastward.
    (a) Voltage source V_AB.
    (b) Equivalent circuit.
    (c) GIC through each transformer ground.
    (d) Extreme storm: E = 10 V/km.
    """
    print("\n" + "=" * 70)
    print("Exercise 2: GIC in Two-Node Network")
    print("=" * 70)

    L = 200      # km
    R_line = 4   # ohm
    R_gA = 0.5   # ohm
    R_gB = 0.5   # ohm
    E_x = 2      # V/km

    # (a) Voltage source
    V_AB = E_x * L

    print(f"\n    Grid: 2 substations, 200 km apart (east-west)")
    print(f"    R_line = {R_line} ohm, R_gA = R_gB = {R_gA} ohm")
    print(f"    E_x = {E_x} V/km (eastward)")

    print(f"\n(a) Voltage source along transmission line:")
    print(f"    V_AB = E_x * L = {E_x} * {L} = {V_AB} V")

    # (b) Circuit description
    print(f"\n(b) Equivalent circuit:")
    print(f"    V_AB in series with R_line = {R_line} ohm")
    print(f"    R_gA = {R_gA} ohm at node A (to ground)")
    print(f"    R_gB = {R_gB} ohm at node B (to ground)")
    print(f"    Total loop resistance = R_line + R_gA + R_gB")

    # (c) GIC
    R_total = R_line + R_gA + R_gB
    I_GIC = V_AB / R_total

    print(f"\n(c) GIC through each transformer ground:")
    print(f"    R_total = {R_line} + {R_gA} + {R_gB} = {R_total} ohm")
    print(f"    I_GIC = V_AB / R_total = {V_AB} / {R_total} = {I_GIC:.0f} A")
    print(f"    Same current flows through both transformer grounds (series circuit)")

    # (d) Extreme storm
    E_extreme = 10  # V/km
    V_extreme = E_extreme * L
    I_extreme = V_extreme / R_total

    print(f"\n(d) Extreme storm (E = {E_extreme} V/km):")
    print(f"    V_AB = {E_extreme} * {L} = {V_extreme} V")
    print(f"    I_GIC = {V_extreme} / {R_total} = {I_extreme:.0f} A")
    print(f"    Is this dangerous? Typical thresholds:")
    print(f"    - 10-20 A: measurable reactive power increase")
    print(f"    - 50-100 A: significant transformer heating, harmonic generation")
    print(f"    - >100 A: risk of transformer damage")
    print(f"    At {I_extreme:.0f} A, this is VERY dangerous for the transformers")


def exercise_3():
    """
    Exercise 3: Transformer Saturation

    500 kV transformer: 1000 turns, Phi_sat = 1.5 Wb, Phi_max = 1.4 Wb.
    Rated magnetizing current = 0.5 A. L_mag/N = 0.01 Wb/A.
    (a) Flux margin.
    (b) GIC needed for saturation.
    (c) Number of saturated transformers to exhaust 200 Mvar reserve.
    (d) Why the 1989 cascade was so rapid.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Transformer Saturation")
    print("=" * 70)

    N = 1000          # turns
    Phi_sat = 1.5     # Wb
    Phi_max = 1.4     # Wb
    I_mag = 0.5       # A (rated magnetizing current)
    Lmag_N = 0.01     # Wb/A  (L_mag / N)
    Q_sat = 50        # Mvar (reactive power when saturated)
    Q_reserve = 200   # Mvar

    # (a) Flux margin
    Delta_Phi = Phi_sat - Phi_max

    print(f"\n    Transformer: {N} turns, Phi_sat = {Phi_sat} Wb, "
          f"Phi_max = {Phi_max} Wb")

    print(f"\n(a) Flux margin:")
    print(f"    Delta_Phi = Phi_sat - Phi_max = {Phi_sat} - {Phi_max} = {Delta_Phi} Wb")
    print(f"    Only {Delta_Phi/Phi_max*100:.0f}% margin before saturation")

    # (b) GIC for saturation
    # Phi_DC = (L_mag / N) * I_GIC = Lmag_N * I_GIC
    # Need Phi_DC >= Delta_Phi
    I_GIC_sat = Delta_Phi / Lmag_N

    print(f"\n(b) GIC needed for saturation:")
    print(f"    Phi_DC = (L_mag/N) * I_GIC")
    print(f"    I_GIC = Delta_Phi / (L_mag/N) = {Delta_Phi} / {Lmag_N}")
    print(f"    I_GIC = {I_GIC_sat:.0f} A")
    print(f"    Only {I_GIC_sat:.0f} A of GIC saturates this transformer!")
    print(f"    This is a surprisingly small current for a large power transformer")

    # (c) Number to exhaust reactive power reserve
    n_transformers = Q_reserve / Q_sat

    print(f"\n(c) Transformers to exhaust reactive power reserve:")
    print(f"    Each saturated transformer consumes {Q_sat} Mvar")
    print(f"    Reserve = {Q_reserve} Mvar")
    print(f"    N = {Q_reserve}/{Q_sat} = {n_transformers:.0f} transformers")
    print(f"    Only {n_transformers:.0f} simultaneously saturated transformers")
    print(f"    exhaust the entire reactive power reserve!")

    # (d) Discussion
    print(f"\n(d) Why the 1989 Quebec cascade was so rapid (~90 seconds):")
    print(f"    1. GICs saturated multiple transformers simultaneously")
    print(f"    2. Saturated transformers consumed massive reactive power ({Q_sat} Mvar each)")
    print(f"    3. Reactive power depletion caused voltage to drop")
    print(f"    4. SVCs (static VAR compensators) tripped on harmonic overcurrent")
    print(f"       generated by saturated transformers")
    print(f"    5. Loss of SVCs removed reactive power support -> voltage collapse")
    print(f"    6. Each SVC trip worsened the reactive power deficit -> cascading")
    print(f"    7. 7 SVCs tripped in 26 seconds, then total system collapse")
    print(f"    The cascade was inherently rapid because each protection action")
    print(f"    (SVC trip) worsened the underlying problem (reactive power deficit)")


def exercise_4():
    """
    Exercise 4: Pipeline Pipe-to-Soil Potential

    Pipeline: 500 km N-S, sigma = 5e-4 S/m, E_y = 3 V/km.
    r_pipe = 5e-5 ohm/m, g = 2e-5 S/m (leakage conductance).
    (a) Total EMF.
    (b) Characteristic length.
    (c) Maximum PSP shift.
    (d) Corrosion threshold check.
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Pipeline Pipe-to-Soil Potential")
    print("=" * 70)

    L = 500e3      # m (500 km)
    sigma = 5e-4   # S/m
    E_y = 3        # V/km = 3e-3 V/m
    r_pipe = 5e-5  # ohm/m
    g = 2e-5       # S/m (leakage conductance per unit length)

    # (a) Total EMF
    V_total = E_y * L / 1e3  # E_y in V/km * L in km
    L_km = L / 1e3

    print(f"\n    Pipeline: {L_km:.0f} km N-S, E_y = {E_y} V/km")
    print(f"    r_pipe = {r_pipe:.0e} ohm/m, g = {g:.0e} S/m")

    print(f"\n(a) Total EMF along pipeline:")
    print(f"    V = E_y * L = {E_y} V/km * {L_km:.0f} km = {V_total:.0f} V")

    # (b) Characteristic length
    ell = 1 / np.sqrt(r_pipe * g)

    print(f"\n(b) Characteristic length:")
    print(f"    ell = 1 / sqrt(r_pipe * g)")
    print(f"    = 1 / sqrt({r_pipe:.0e} * {g:.0e})")
    print(f"    = 1 / sqrt({r_pipe*g:.0e})")
    print(f"    = {ell:.0f} m = {ell/1e3:.0f} km")

    # (c) Maximum PSP shift
    # Delta_V_PSP ~ E_y / sqrt(r_pipe * g) = E_y * ell
    # But E_y needs to be in V/m: E_y = 3 V/km = 3e-3 V/m
    E_y_Vm = E_y * 1e-3  # V/m
    Delta_V = E_y_Vm / np.sqrt(r_pipe * g)
    # Or equivalently: Delta_V = E_y_Vm * ell

    print(f"\n(c) Maximum PSP shift at pipeline endpoints:")
    print(f"    Delta_V_PSP ~ E_y / sqrt(r_pipe * g)")
    print(f"    = {E_y_Vm:.0e} V/m / sqrt({r_pipe*g:.0e})")
    print(f"    = {E_y_Vm:.0e} / {np.sqrt(r_pipe*g):.3e}")
    print(f"    = {Delta_V:.1f} V")

    # (d) Corrosion check
    PSP_normal = -0.95  # V
    PSP_threshold = -0.85  # V (corrosion accelerates above this)

    PSP_storm_pos = PSP_normal + Delta_V  # positive shift at one end
    PSP_storm_neg = PSP_normal - Delta_V  # negative shift at other end

    print(f"\n(d) Corrosion assessment:")
    print(f"    Normal PSP = {PSP_normal} V")
    print(f"    Corrosion threshold = {PSP_threshold} V")
    print(f"    Storm PSP shift = +/- {Delta_V:.1f} V")
    print(f"    PSP at positive end: {PSP_normal} + {Delta_V:.1f} = {PSP_storm_pos:.1f} V")
    print(f"    PSP at negative end: {PSP_normal} - {Delta_V:.1f} = {PSP_storm_neg:.1f} V")

    if PSP_storm_pos > PSP_threshold:
        print(f"\n    YES - the positive end ({PSP_storm_pos:.1f} V) exceeds the")
        print(f"    corrosion threshold ({PSP_threshold} V)")
        print(f"    The cathodic protection is overwhelmed at one end of the pipeline")
        print(f"    This can accelerate corrosion if the storm persists for hours")
    else:
        print(f"    No - PSP remains below corrosion threshold")


def exercise_5():
    """
    Exercise 5: Carrington-Class Risk Assessment

    dB/dt ~ 100 nT/s for ~5 min, dominant period T ~ 120 s.
    (a) Geoelectric field (sigma = 1e-3 S/m). Compare with 1989.
    (b) GIC scaling from 1989 (~100 A).
    (c) Economic impact: 50 EHV transformers, 10% damaged.
    (d) Why recovery could take years.
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Carrington-Class Risk Assessment")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7
    sigma = 1e-3    # S/m
    T = 120         # s
    dBdt = 100e-9   # T/s (100 nT/s)

    # (a) Geoelectric field
    omega = 2 * np.pi / T
    delta = np.sqrt(2 / (omega * mu0 * sigma))

    # For a sinusoidal signal with dB/dt = omega * B0:
    B0 = dBdt / omega

    E0 = np.sqrt(omega / (2 * mu0 * sigma)) * B0
    E0_Vkm = E0 * 1e3  # V/km

    # 1989 Quebec: E ~ 2 V/km
    E_1989 = 2  # V/km

    print(f"\n    Carrington-class: dB/dt = {dBdt*1e9:.0f} nT/s, T = {T} s")
    print(f"    Ground conductivity: sigma = {sigma:.0e} S/m")

    print(f"\n(a) Geoelectric field:")
    print(f"    omega = 2*pi/{T} = {omega:.4f} rad/s")
    print(f"    B0 = dB/dt / omega = {dBdt:.1e} / {omega:.4f} = {B0:.2e} T "
          f"= {B0*1e9:.0f} nT")
    print(f"    delta = {delta:.0f} m = {delta/1e3:.1f} km")
    print(f"    |E| = sqrt(omega/(2*mu0*sigma)) * B0")
    print(f"    = {E0:.4e} V/m = {E0_Vkm:.1f} V/km")
    print(f"    1989 Quebec event: ~{E_1989} V/km")
    print(f"    Carrington-class: ~{E0_Vkm:.1f} V/km ({E0_Vkm/E_1989:.0f}x larger)")

    # (b) GIC scaling
    I_1989 = 100  # A (approximate peak GIC in 1989)
    I_carrington = I_1989 * (E0_Vkm / E_1989)

    print(f"\n(b) GIC estimate:")
    print(f"    1989 GIC: ~{I_1989} A")
    print(f"    Scaling: GIC ~ E_field (linear)")
    print(f"    Carrington GIC: {I_1989} * {E0_Vkm/E_1989:.0f} = {I_carrington:.0f} A")
    print(f"    This exceeds all known transformer tolerance levels")

    # (c) Economic impact
    n_transformers = 50
    frac_damaged = 0.10
    cost_per = 8e6  # $8M each
    lead_time = 18  # months
    n_damaged = int(n_transformers * frac_damaged)
    total_cost = n_damaged * cost_per

    print(f"\n(c) Economic impact:")
    print(f"    EHV transformers in region: {n_transformers}")
    print(f"    Damaged (10%): {n_damaged}")
    print(f"    Cost per transformer: ${cost_per/1e6:.0f} million")
    print(f"    Total replacement cost: ${total_cost/1e6:.0f} million")
    print(f"    Lead time: {lead_time} months per transformer")
    print(f"    If replacements sequential: {n_damaged * lead_time / 12:.0f} years!")
    print(f"    Even with parallel ordering: {lead_time} months minimum")
    print(f"    (assuming manufacturing capacity is available)")

    # (d) Discussion
    print(f"\n(d) Why recovery could take years:")
    print(f"    1. EHV transformers are custom-built, not off-the-shelf")
    print(f"    2. Only a handful of factories worldwide can build them")
    print(f"    3. A Carrington event would damage transformers globally,")
    print(f"       creating unprecedented demand with limited supply")
    print(f"    4. Without transformers, power cannot flow from generators")
    print(f"       to consumers -> cascading failure of dependent systems")
    print(f"    5. Water treatment, fuel pumping, communications, banking")
    print(f"       all depend on electricity")
    print(f"    6. Without power, the factories that MAKE transformers")
    print(f"       cannot operate -> chicken-and-egg problem")
    print(f"    7. The 1989 Quebec recovery (9 hours) was possible because")
    print(f"       only 1 transformer was damaged; Carrington-class damage")
    print(f"       would be 100-1000x more extensive")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
