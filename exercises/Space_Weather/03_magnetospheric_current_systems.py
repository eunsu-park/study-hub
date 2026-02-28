"""
Exercise Solutions for Lesson 03: Magnetospheric Current Systems

Topics covered:
  - Ring current energy via Dessler-Parker-Sckopke (DPS) relation
  - Magnetopause current density from pressure balance
  - Joule heating estimation from cross-polar cap potential
  - Birkeland current mapping from ionosphere to magnetopause
  - Electrojet magnetic perturbation and AE index
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Ring Current Energy (DPS Relation)

    Dst = -200 nT. Using the DPS relation, calculate total ring current energy.
    If 40% is carried by O+ ions with average energy 50 keV, find the total
    number of O+ ions.

    DPS relation: E_rc = (3/2) * (4*pi / mu0) * B0 * R_E^3 * |Dst| / B_0
    Simplified: E_rc ~ (3 * |Dst| * B0 * R_E^3) / mu0
    Or more precisely: Delta_B / B0 = -(2/3) * E_rc / E_B
    where E_B = B0^2 * R_E^3 / (3*mu0) is the magnetic energy of the dipole
    above Earth's surface.
    """
    print("=" * 70)
    print("Exercise 1: Ring Current Energy (DPS Relation)")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7
    B0 = 3.1e-5     # T (31,000 nT, equatorial surface field)
    R_E = 6.371e6    # m
    Dst = -200e-9    # T (-200 nT)
    eV = 1.602e-19   # J per eV

    # DPS relation: |Delta_B| / B_0 = (2/3) * E_rc / E_B
    # where E_B = B_0^2 * R_E^3 / (3 * mu0)  [dipole magnetic energy]
    # => E_rc = (3/2) * |Delta_B| / B_0 * E_B
    # => E_rc = (3/2) * |Dst| / B_0 * B_0^2 * R_E^3 / (3 * mu0)
    # => E_rc = |Dst| * B_0 * R_E^3 / (2 * mu0)

    # Or equivalently, a commonly used form:
    # E_rc [J] ≈ 2.8e13 * |Dst| [nT]
    # Let's derive it properly:
    Delta_B = abs(Dst)  # |Dst| in Tesla
    E_B = B0**2 * R_E**3 / (3 * mu0)
    E_rc = (3 / 2) * (Delta_B / B0) * E_B

    print(f"\n    Dst = {Dst*1e9:.0f} nT = {Dst:.1e} T")
    print(f"    B_0 = {B0:.1e} T, R_E = {R_E:.3e} m")
    print(f"\n    DPS relation: |Delta_B|/B_0 = (2/3) * E_rc / E_B")
    print(f"    E_B (dipole energy) = B_0^2 * R_E^3 / (3*mu0) = {E_B:.3e} J")
    print(f"    E_rc = (3/2) * (|Dst|/B_0) * E_B")
    print(f"    E_rc = {E_rc:.3e} J")

    # Alternative: E_rc = |Dst| * B0 * R_E^3 / (2 * mu0)
    E_rc_alt = Delta_B * B0 * R_E**3 / (2 * mu0)
    print(f"    Verification: |Dst| * B_0 * R_E^3 / (2*mu0) = {E_rc_alt:.3e} J")

    # O+ contribution
    f_O = 0.40         # 40% of energy
    E_avg = 50e3 * eV  # 50 keV in Joules

    E_O = f_O * E_rc
    N_O = E_O / E_avg

    print(f"\n    O+ contribution (40% of ring current energy):")
    print(f"    E_O+ = 0.40 * {E_rc:.3e} = {E_O:.3e} J")
    print(f"    Average O+ energy = 50 keV = {E_avg:.3e} J")
    print(f"    Number of O+ ions = E_O+ / E_avg = {N_O:.3e}")
    print(f"    That is ~{N_O:.1e} oxygen ions in the ring current")


def exercise_2():
    """
    Exercise 2: Magnetopause Current Density

    B_in = 60 nT (inside), B_out = 20 nT (outside, different direction).
    (a) Calculate surface current density K.
    (b) With thickness 500 km, estimate volume current density j.
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Magnetopause Current Density")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7

    B_in = 60e-9    # T
    B_out = 20e-9   # T
    d = 500e3       # m (500 km thickness)

    # Surface current from boundary condition: K = (1/mu0) * (B_in - B_out) x n_hat
    # For the tangential discontinuity at the magnetopause,
    # the jump in B gives the surface current:
    # K = Delta_B / mu0 (magnitude, for fields in different directions)
    # If fields are antiparallel on inside/outside, Delta_B = B_in + B_out
    Delta_B = B_in + B_out  # fields in opposite directions across boundary
    K = Delta_B / mu0

    print(f"\n    B_in = {B_in*1e9:.0f} nT, B_out = {B_out*1e9:.0f} nT")
    print(f"    (Fields are in different directions across the magnetopause)")
    print(f"\n(a) Surface current density:")
    print(f"    K = Delta_B / mu0 = (B_in + B_out) / mu0")
    print(f"    K = ({B_in*1e9:.0f} + {B_out*1e9:.0f}) nT / {mu0:.3e}")
    print(f"    K = {Delta_B:.1e} T / {mu0:.3e} T*m/A")
    print(f"    K = {K:.3e} A/m = {K*1e-3:.2f} mA/m")

    # Volume current density
    j = K / d
    print(f"\n(b) Volume current density (thickness = {d/1e3:.0f} km):")
    print(f"    j = K / d = {K:.3e} / {d:.1e}")
    print(f"    j = {j:.3e} A/m^2 = {j*1e9:.2f} nA/m^2")

    # Context
    print(f"\n    For comparison:")
    print(f"    - Ionospheric Pedersen currents: ~0.1-1 uA/m^2")
    print(f"    - Ring current density: ~1-10 nA/m^2")
    print(f"    - Magnetopause current is relatively thin and intense")


def exercise_3():
    """
    Exercise 3: Joule Heating Estimate

    CPCP = 150 kV, polar cap radius = 15 deg (~1670 km).
    (a) Average electric field in polar cap.
    (b) Total Joule heating rate (one hemisphere) with Sigma_P = 8 S.
    (c) Compare with Akasofu epsilon.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Joule Heating Estimate")
    print("=" * 70)

    Phi_PC = 150e3   # V (150 kV)
    R_pc = 1670e3    # m (polar cap radius, ~15 deg latitude)
    Sigma_P = 8      # S (height-integrated Pedersen conductance)

    # (a) Average electric field
    # E ~ Phi_PC / (2 * R_pc)  (voltage drop across the polar cap diameter)
    E_avg = Phi_PC / (2 * R_pc)

    print(f"\n    CPCP = {Phi_PC/1e3:.0f} kV, Polar cap radius = {R_pc/1e3:.0f} km")
    print(f"    Sigma_P = {Sigma_P} S")

    print(f"\n(a) Average electric field in polar cap:")
    print(f"    E ~ Phi_PC / (2 * R_pc) = {Phi_PC/1e3:.0f} kV / "
          f"(2 * {R_pc/1e3:.0f} km)")
    print(f"    E = {E_avg*1e3:.2f} mV/m")

    # (b) Total Joule heating
    # Q_J = Sigma_P * E^2 * A  where A = pi * R_pc^2
    A_pc = np.pi * R_pc**2
    Q_J = Sigma_P * E_avg**2 * A_pc
    Q_J_GW = Q_J * 1e-9

    print(f"\n(b) Total Joule heating (one hemisphere):")
    print(f"    Q_J = Sigma_P * E^2 * A")
    print(f"    A = pi * R_pc^2 = {A_pc:.3e} m^2")
    print(f"    Q_J = {Sigma_P} * ({E_avg*1e3:.2f}e-3)^2 * {A_pc:.3e}")
    print(f"    Q_J = {Q_J:.3e} W = {Q_J_GW:.1f} GW")
    print(f"    Both hemispheres: ~{2*Q_J_GW:.0f} GW")

    # (c) Compare with epsilon
    # A typical epsilon for these conditions might be ~5e11 W
    # Let's estimate: for the driving needed to produce 150 kV CPCP,
    # epsilon ~ 4e7 * Phi_PC (rough relation from problem statement)
    epsilon_est = 4e7 * Phi_PC  # W (using the relation from the problem)
    epsilon_GW = epsilon_est * 1e-9

    print(f"\n(c) Comparison with Akasofu epsilon:")
    print(f"    Using Phi_PC ~ epsilon / (4e7 W/V):")
    print(f"    epsilon ~ {epsilon_est:.1e} W = {epsilon_GW:.0f} GW")
    print(f"    Joule heating fraction: Q_J(both) / epsilon = "
          f"{2*Q_J/epsilon_est:.2f}")
    print(f"    => Joule heating accounts for ~{2*Q_J/epsilon_est*100:.0f}% "
          f"of total energy input")
    print(f"    (Consistent with ~35-45% expected from energy partitioning)")


def exercise_4():
    """
    Exercise 4: Birkeland Current Mapping

    R1 FAC: 3 MA downward on dawn side at 70 deg invariant latitude,
    distributed over 2 deg latitude band.
    (a) Average current density at 110 km altitude.
    (b) Mapped current density at equatorial magnetopause (B_iono/B_eq ~ 50).
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Birkeland Current Mapping")
    print("=" * 70)

    R_E = 6.371e6   # m
    I_total = 3e6    # A (3 MA)
    Lambda = 70      # degrees invariant latitude
    dLambda = 2      # degrees latitude band width
    h = 110e3        # m altitude
    B_ratio = 50     # B_iono / B_eq

    # (a) Current density at ionosphere
    # Annulus at 110 km altitude centered at 70 deg colatitude = 20 deg
    # from geographic pole
    r_iono = R_E + h
    colatitude = 90 - Lambda  # = 20 degrees
    colat_rad = np.radians(colatitude)
    dcolat_rad = np.radians(dLambda)

    # Area of annulus: A = 2*pi*r^2 * sin(colat) * d(colat)
    A_annulus = 2 * np.pi * r_iono**2 * np.sin(colat_rad) * dcolat_rad

    j_iono = I_total / A_annulus

    print(f"\n    Total current: {I_total/1e6:.0f} MA at {Lambda} deg "
          f"invariant latitude")
    print(f"    Latitude band: {dLambda} deg at {h/1e3:.0f} km altitude")

    print(f"\n(a) Ionospheric current density:")
    print(f"    Colatitude = {colatitude} deg, r_iono = R_E + h = {r_iono:.3e} m")
    print(f"    Annulus area = 2*pi*r^2*sin(colat)*d(colat)")
    print(f"    = 2*pi*({r_iono:.3e})^2 * sin({colatitude} deg) * "
          f"{dLambda} deg * (pi/180)")
    print(f"    = {A_annulus:.3e} m^2")
    print(f"    j_iono = I / A = {I_total:.1e} / {A_annulus:.3e}")
    print(f"    j_iono = {j_iono:.3e} A/m^2 = {j_iono*1e6:.2f} uA/m^2")

    # (b) Mapped to equatorial magnetopause
    # Current density maps as j * B = const along flux tube (current continuity)
    # j_eq = j_iono * (B_eq / B_iono) = j_iono / B_ratio
    j_eq = j_iono / B_ratio

    print(f"\n(b) Current density at equatorial magnetopause:")
    print(f"    B_iono / B_eq = {B_ratio}")
    print(f"    Current continuity: j * A = const, and A * B = const (flux)")
    print(f"    => j_eq / j_iono = B_eq / B_iono = 1/{B_ratio}")
    print(f"    j_eq = j_iono / {B_ratio} = {j_iono:.3e} / {B_ratio}")
    print(f"    j_eq = {j_eq:.3e} A/m^2 = {j_eq*1e9:.2f} nA/m^2")

    print(f"\n    Note: The current density decreases dramatically from")
    print(f"    ionosphere to equatorial plane because the magnetic flux tube")
    print(f"    area expands by a factor of ~{B_ratio} (the field weakens).")


def exercise_5():
    """
    Exercise 5: Electrojet and AE Index

    Westward auroral electrojet: 500 kA at 110 km altitude,
    centered at 67 deg magnetic latitude. Model as infinite line current.
    (a) Magnetic perturbation (H) directly below at ground.
    (b) If substorm doubles the current, change in AL index.
    (c) Why is infinite line current a poor approximation?
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Electrojet and AE Index")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7
    R_E = 6.371e6
    I = 500e3     # A (500 kA)
    h = 110e3     # m altitude

    # (a) Magnetic field from infinite line current at distance d
    # B = mu0 * I / (2 * pi * d)
    # Distance from electrojet (at 110 km altitude) to ground below = h
    d = h
    B_line = mu0 * I / (2 * np.pi * d)
    B_line_nT = B_line * 1e9

    print(f"\n    Electrojet: I = {I/1e3:.0f} kA at {h/1e3:.0f} km altitude")
    print(f"    Centered at 67 deg magnetic latitude")

    print(f"\n(a) H-component perturbation directly below:")
    print(f"    Model: infinite line current at height h = {h/1e3:.0f} km")
    print(f"    B = mu0 * I / (2*pi*d) where d = h = {d/1e3:.0f} km")
    print(f"    B = {mu0:.3e} * {I:.1e} / (2*pi*{d:.1e})")
    print(f"    B = {B_line:.3e} T = {B_line_nT:.0f} nT")
    print(f"    The H-component perturbation is ~{B_line_nT:.0f} nT (negative,")
    print(f"    since the westward current produces a southward H perturbation)")

    # (b) Substorm doubles the current
    I_substorm = 2 * I
    B_substorm = mu0 * I_substorm / (2 * np.pi * d)
    Delta_B = (B_substorm - B_line) * 1e9
    print(f"\n(b) Substorm doubles current to {I_substorm/1e3:.0f} kA:")
    print(f"    New H perturbation = {B_substorm*1e9:.0f} nT")
    print(f"    Change in perturbation: Delta_H = {Delta_B:.0f} nT")
    print(f"    Since AL tracks the most negative H perturbation,")
    print(f"    Delta_AL ~ -{Delta_B:.0f} nT (becomes more negative)")
    print(f"    So AL decreases by ~{Delta_B:.0f} nT during the substorm")

    # (c) Why infinite line current is a poor approximation
    print(f"\n(c) Limitations of infinite line current model:")
    print(f"    - The electrojet has finite length (~few thousand km), not infinite")
    print(f"    - The electrojet has finite width (~200-500 km latitude extent)")
    print(f"    - Current varies in intensity along its length")
    print(f"    - The image current in the conducting Earth is not accounted for")
    print(f"    - The return currents (through the magnetosphere via FACs) also")
    print(f"      produce ground magnetic perturbations")
    print(f"    - The infinite line current OVERESTIMATES the field because a")
    print(f"      finite-length current produces a weaker field than an infinite one")
    print(f"    - A more realistic model: use Biot-Savart for a current ribbon")
    print(f"      with finite extent plus image currents in the conducting Earth")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
