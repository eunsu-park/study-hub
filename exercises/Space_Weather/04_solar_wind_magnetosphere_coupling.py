"""
Exercise Solutions for Lesson 04: Solar Wind-Magnetosphere Coupling

Topics covered:
  - Coupling function calculation (Akasofu epsilon, Newell dPhi/dt)
  - Cross-polar cap potential (CPCP) saturation
  - Burton equation analytical solution for Dst prediction
  - Energy budget during a geomagnetic storm
  - Kelvin-Helmholtz instability threshold at magnetopause
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Exercise 1: Coupling Function Calculation

    Solar wind: v_sw = 600 km/s, By = 8 nT, Bz = -12 nT.
    (a) Clock angle and transverse field B_T.
    (b) Akasofu epsilon with l0 = 7 R_E.
    (c) Newell coupling function dPhi/dt.
    (d) Compare predicted CPCP with saturation value of 200 kV.
    """
    print("=" * 70)
    print("Exercise 1: Coupling Function Calculation")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7
    R_E = 6.371e6

    v_sw = 600e3   # m/s
    By = 8e-9      # T
    Bz = -12e-9    # T
    l0 = 7 * R_E   # m (effective coupling length)

    # (a) Clock angle and B_T
    B_T = np.sqrt(By**2 + Bz**2)
    theta_c = np.arctan2(abs(By), Bz)  # angle from +z in y-z plane
    # More precisely, clock angle measured from north (positive Bz):
    theta_c = np.arctan2(By, Bz)  # This gives angle from +Bz direction
    # For southward Bz, we want the angle in the range [0, 2*pi]
    # Standard definition: theta_c = arctan(By / Bz) but measured from north
    # With By > 0 and Bz < 0, theta is in second quadrant
    theta_c = np.arctan2(By, Bz)  # gives angle from +z axis
    if theta_c < 0:
        theta_c += 2 * np.pi
    theta_c_deg = np.degrees(theta_c)

    print(f"\n    Solar wind: v = {v_sw/1e3:.0f} km/s, "
          f"By = {By*1e9:.0f} nT, Bz = {Bz*1e9:.0f} nT")

    print(f"\n(a) Clock angle and transverse field:")
    print(f"    B_T = sqrt(By^2 + Bz^2) = sqrt({By*1e9:.0f}^2 + {Bz*1e9:.0f}^2)")
    print(f"    B_T = {B_T*1e9:.2f} nT")
    print(f"    theta_c = arctan2(By, Bz) = {theta_c_deg:.1f} deg")
    print(f"    (Measured from northward Bz; {theta_c_deg:.0f} deg indicates "
          f"strongly southward with dawnward By)")

    # (b) Akasofu epsilon
    # epsilon = (1/mu0) * v * B_T^2 * l0^2 * sin^4(theta_c/2)
    sin4_half = np.sin(theta_c / 2)**4
    epsilon = (1 / mu0) * v_sw * B_T**2 * l0**2 * sin4_half

    print(f"\n(b) Akasofu epsilon parameter:")
    print(f"    epsilon = (1/mu0) * v * B_T^2 * l0^2 * sin^4(theta_c/2)")
    print(f"    sin^4(theta_c/2) = sin^4({theta_c_deg/2:.1f} deg) = {sin4_half:.4f}")
    print(f"    epsilon = {epsilon:.3e} W")
    print(f"    = {epsilon*1e-9:.1f} GW")

    # (c) Newell coupling function
    # dPhi/dt = v^(4/3) * B_T^(2/3) * sin^(8/3)(theta_c/2)
    sin_83 = np.sin(theta_c / 2)**(8 / 3)
    # Note: Newell uses v in km/s and B in nT for the standard units
    v_kms = v_sw / 1e3
    B_T_nT = B_T * 1e9
    dPhi_dt = v_kms**(4 / 3) * B_T_nT**(2 / 3) * sin_83

    print(f"\n(c) Newell coupling function dPhi/dt:")
    print(f"    dPhi/dt = v^(4/3) * B_T^(2/3) * sin^(8/3)(theta_c/2)")
    print(f"    v^(4/3) = {v_kms:.0f}^(4/3) = {v_kms**(4/3):.1f} (km/s)^(4/3)")
    print(f"    B_T^(2/3) = {B_T_nT:.2f}^(2/3) = {B_T_nT**(2/3):.2f} nT^(2/3)")
    print(f"    sin^(8/3)(theta_c/2) = {sin_83:.4f}")
    print(f"    dPhi/dt = {dPhi_dt:.1f} (Wb/s in Newell's units)")

    # (d) CPCP comparison
    # Phi_PC ~ epsilon / (4e7)  (rough conversion from problem)
    Phi_PC_pred = epsilon / 4e7  # Volts
    Phi_sat = 200e3  # V (200 kV)

    print(f"\n(d) CPCP comparison:")
    print(f"    Predicted CPCP = epsilon / (4e7 W/V) = {Phi_PC_pred/1e3:.1f} kV")
    print(f"    Saturation value = {Phi_sat/1e3:.0f} kV")
    if Phi_PC_pred > Phi_sat:
        print(f"    => Predicted CPCP ({Phi_PC_pred/1e3:.0f} kV) EXCEEDS "
              f"saturation ({Phi_sat/1e3:.0f} kV)")
        print(f"    => The actual CPCP would be limited to ~{Phi_sat/1e3:.0f} kV")
    else:
        print(f"    => Below saturation; linear regime applies")


def exercise_2():
    """
    Exercise 2: CPCP Saturation

    Combined formula: Phi_PC = Phi_lin * Phi_sat / (Phi_lin + Phi_sat)
    (a) Calculate Phi_PC for Phi_lin = 0 to 500 kV (Phi_sat = 200 kV).
    (b) At what Phi_lin is actual CPCP = half unsaturated value?
    (c) If Sigma_P doubles and Phi_sat ~ Sigma_P^(-2/3), how does Phi_sat change?
    """
    print("\n" + "=" * 70)
    print("Exercise 2: CPCP Saturation")
    print("=" * 70)

    Phi_sat = 200  # kV

    # (a) Plot Phi_PC vs Phi_lin
    Phi_lin = np.linspace(0, 500, 200)  # kV
    Phi_PC = Phi_lin * Phi_sat / (Phi_lin + Phi_sat)

    print(f"\n(a) CPCP vs unsaturated potential (Phi_sat = {Phi_sat} kV):")
    print(f"    Phi_PC = Phi_lin * Phi_sat / (Phi_lin + Phi_sat)")
    print(f"\n    {'Phi_lin (kV)':>14} {'Phi_PC (kV)':>14} {'Ratio':>10}")
    print(f"    {'-'*40}")
    for val in [0, 50, 100, 150, 200, 300, 400, 500]:
        pc = val * Phi_sat / (val + Phi_sat) if val > 0 else 0
        ratio = pc / val if val > 0 else 1.0
        print(f"    {val:>14.0f} {pc:>14.1f} {ratio:>10.3f}")

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(Phi_lin, Phi_PC, 'b-', linewidth=2, label='Saturated CPCP')
    ax.plot(Phi_lin, Phi_lin, 'r--', linewidth=1, label='Unsaturated (linear)')
    ax.axhline(y=Phi_sat, color='gray', linestyle=':', label=f'Phi_sat = {Phi_sat} kV')
    ax.set_xlabel('Phi_lin (kV)')
    ax.set_ylabel('Phi_PC (kV)')
    ax.set_title('Cross-Polar Cap Potential Saturation')
    ax.legend()
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 300)
    ax.grid(True, alpha=0.3)
    fig.savefig('/opt/projects/01_Personal/03_Study/exercises/Space_Weather/'
                'ex04_cpcp_saturation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n    Plot saved: ex04_cpcp_saturation.png")

    # (b) When is Phi_PC = Phi_lin / 2 ?
    # Phi_lin * Phi_sat / (Phi_lin + Phi_sat) = Phi_lin / 2
    # Phi_sat / (Phi_lin + Phi_sat) = 1/2
    # 2 * Phi_sat = Phi_lin + Phi_sat
    # Phi_lin = Phi_sat
    print(f"\n(b) When does Phi_PC = Phi_lin / 2?")
    print(f"    Phi_sat / (Phi_lin + Phi_sat) = 1/2")
    print(f"    => Phi_lin = Phi_sat = {Phi_sat} kV")
    print(f"    At Phi_lin = {Phi_sat} kV, actual CPCP = {Phi_sat/2:.0f} kV "
          f"(half of linear prediction)")

    # (c) Conductance doubling
    # Phi_sat ~ Sigma_P^(-2/3)
    # If Sigma_P -> 2*Sigma_P:
    factor = 2**(-2 / 3)
    Phi_sat_new = Phi_sat * factor

    print(f"\n(c) Conductance doubling effect on saturation:")
    print(f"    Phi_sat ~ Sigma_P^(-2/3)")
    print(f"    If Sigma_P doubles: Phi_sat_new = Phi_sat * 2^(-2/3)")
    print(f"    Factor = 2^(-2/3) = {factor:.4f}")
    print(f"    Phi_sat_new = {Phi_sat} * {factor:.4f} = {Phi_sat_new:.1f} kV")
    print(f"\n    Implications:")
    print(f"    - Enhanced precipitation during storms increases Sigma_P")
    print(f"    - This REDUCES the saturation potential by ~{(1-factor)*100:.0f}%")
    print(f"    - Lower CPCP means weaker convection (negative feedback)")
    print(f"    - BUT total energy input can still increase through other channels")
    print(f"      (e.g., Joule heating rate = Sigma_P * E^2 increases with Sigma_P)")


def exercise_3():
    """
    Exercise 3: Burton Equation

    CME: v_sw = 500 km/s, Bz = -15 nT for 6 hours, then stops.
    Initial Dst* = 0.
    Parameters: Q = a*(v_sw*Bs - Ec), a = -4.4 nT/hr/(mV/m), Ec = 0.5 mV/m,
    tau = 8 hours.
    (a) E_sw in mV/m.
    (b) Q in nT/hr.
    (c) Analytical solution for Dst*(t=6h).
    (d) Minimum Dst* if driving continues indefinitely.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Burton Equation Analytical Solution")
    print("=" * 70)

    v_sw = 500e3    # m/s
    Bs = 15e-9      # T (magnitude of southward component)
    a = -4.4        # nT/hr per (mV/m)
    Ec = 0.5        # mV/m
    tau = 8.0       # hours
    t_drive = 6.0   # hours

    # (a) E_sw = v_sw * Bs
    E_sw = v_sw * Bs  # V/m
    E_sw_mVm = E_sw * 1e3  # mV/m

    print(f"\n    Parameters:")
    print(f"    v_sw = {v_sw/1e3:.0f} km/s, Bs = {Bs*1e9:.0f} nT")
    print(f"    a = {a} nT/hr/(mV/m), Ec = {Ec} mV/m, tau = {tau} hr")

    print(f"\n(a) Solar wind electric field:")
    print(f"    E_sw = v_sw * Bs = {v_sw/1e3:.0f} km/s * {Bs*1e9:.0f} nT")
    print(f"    = {v_sw:.0f} m/s * {Bs:.1e} T")
    print(f"    = {E_sw:.4e} V/m = {E_sw_mVm:.2f} mV/m")

    # (b) Q = a * (E_sw - Ec)
    Q = a * (E_sw_mVm - Ec)

    print(f"\n(b) Injection rate Q:")
    print(f"    Q = a * (E_sw - Ec) = {a} * ({E_sw_mVm:.2f} - {Ec})")
    print(f"    Q = {a} * {E_sw_mVm - Ec:.2f} = {Q:.2f} nT/hr")

    # (c) Analytical solution
    # dDst*/dt = Q - Dst*/tau
    # Solution: Dst*(t) = Q*tau*(1 - exp(-t/tau))  for Dst*(0) = 0
    t = t_drive
    Dst_star_6h = Q * tau * (1 - np.exp(-t / tau))

    print(f"\n(c) Analytical solution of Burton equation:")
    print(f"    dDst*/dt = Q - Dst*/tau")
    print(f"    Solution: Dst*(t) = Q*tau*(1 - exp(-t/tau)) for Dst*(0) = 0")
    print(f"    Dst*(t=6h) = {Q:.2f} * {tau} * (1 - exp(-{t}/{tau}))")
    print(f"    = {Q * tau:.2f} * (1 - {np.exp(-t/tau):.4f})")
    print(f"    = {Q * tau:.2f} * {1 - np.exp(-t/tau):.4f}")
    print(f"    = {Dst_star_6h:.1f} nT")

    # Time evolution
    print(f"\n    Time evolution during driving phase:")
    times = np.arange(0, t_drive + 0.5, 0.5)
    print(f"    {'t (hr)':>8} {'Dst* (nT)':>12}")
    print(f"    {'-'*22}")
    for ti in times:
        dst_i = Q * tau * (1 - np.exp(-ti / tau))
        print(f"    {ti:>8.1f} {dst_i:>12.1f}")

    # (d) Equilibrium (t -> infinity)
    Dst_star_eq = Q * tau

    print(f"\n(d) Equilibrium Dst* (driving continues indefinitely):")
    print(f"    Dst*_eq = Q * tau = {Q:.2f} * {tau} = {Dst_star_eq:.1f} nT")
    print(f"    This is the balance between injection ({Q:.1f} nT/hr) and")
    print(f"    decay (Dst*/tau) — reached when dDst*/dt = 0")
    print(f"    Time to reach 95%: t_95 = 3*tau = {3*tau:.0f} hours")


def exercise_4():
    """
    Exercise 4: Energy Budget

    Average epsilon = 8e11 W for 12 hours.
    (a) Total energy input.
    (b) 40% to Joule heating over auroral zone (65-75 deg, both hemispheres).
    (c) Compare with solar EUV flux (~1 mW/m^2).
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Energy Budget")
    print("=" * 70)

    epsilon = 8e11       # W
    dt = 12 * 3600       # seconds (12 hours)
    R_E = 6.371e6        # m
    f_joule = 0.40       # 40% to Joule heating

    # (a) Total energy
    E_total = epsilon * dt
    print(f"\n(a) Total energy input:")
    print(f"    epsilon = {epsilon:.1e} W for {dt/3600:.0f} hours")
    print(f"    E_total = {epsilon:.1e} * {dt:.1e} = {E_total:.3e} J")

    # (b) Joule heating over auroral zone
    E_joule = f_joule * E_total
    P_joule = f_joule * epsilon

    # Auroral zone: annulus at 65-75 deg latitude, both hemispheres
    # Area of latitude annulus on sphere: A = 2*pi*R^2 * |cos(lat1) - cos(lat2)|
    lat1 = np.radians(65)
    lat2 = np.radians(75)
    A_one = 2 * np.pi * R_E**2 * abs(np.cos(lat1) - np.cos(lat2))
    A_both = 2 * A_one  # both hemispheres

    # Energy flux
    q_joule = P_joule / A_both  # W/m^2

    print(f"\n(b) Joule heating in auroral zone:")
    print(f"    Joule power = {f_joule} * epsilon = {P_joule:.3e} W "
          f"= {P_joule*1e-9:.0f} GW")
    print(f"    Auroral annulus (65-75 deg latitude):")
    print(f"    A_one_hemi = 2*pi*R_E^2 * |cos(65) - cos(75)|")
    print(f"    = 2*pi*({R_E:.3e})^2 * |{np.cos(lat1):.4f} - {np.cos(lat2):.4f}|")
    print(f"    = {A_one:.3e} m^2 per hemisphere")
    print(f"    A_both = {A_both:.3e} m^2")
    print(f"    Energy flux = P_joule / A = {q_joule:.4e} W/m^2")
    print(f"    = {q_joule*1e3:.2f} mW/m^2")

    # (c) Compare with solar EUV
    q_EUV = 1e-3  # W/m^2 (1 mW/m^2)
    ratio = q_joule / q_EUV

    print(f"\n(c) Comparison with solar EUV flux:")
    print(f"    Solar EUV at top of atmosphere: ~{q_EUV*1e3:.0f} mW/m^2")
    print(f"    Storm Joule heating flux: ~{q_joule*1e3:.2f} mW/m^2")
    print(f"    Ratio: {ratio:.1f}x")
    print(f"\n    Interpretation:")
    print(f"    Storm-time Joule heating flux is ~{ratio:.0f}x the solar EUV flux.")
    print(f"    This explains why storms cause dramatic thermospheric heating,")
    print(f"    neutral density enhancements, and increased satellite drag.")
    print(f"    The ionosphere-thermosphere system is fundamentally perturbed.")


def exercise_5():
    """
    Exercise 5: Kelvin-Helmholtz Instability Threshold

    Dawn flank: v_sheath = 300 km/s, n_sheath = 20 cm^-3, n_sphere = 1 cm^-3.
    B = 20 nT both sides.
    (a) KHI threshold for waves perpendicular to B (B.k = 0).
    (b) Is dawn flank unstable?
    (c) Dusk flank with B parallel to flow (B.k = B). Threshold? Stable?
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Kelvin-Helmholtz Instability Threshold")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7
    m_p = 1.67e-27

    v_sheath = 300e3   # m/s
    n_sheath = 20e6    # m^-3
    n_sphere = 1e6     # m^-3
    B = 20e-9          # T (both sides)

    rho1 = n_sheath * m_p  # magnetosheath density
    rho2 = n_sphere * m_p  # magnetospheric density

    print(f"\n    Magnetosheath: v = {v_sheath/1e3:.0f} km/s, "
          f"n = {n_sheath/1e6:.0f} cm^-3")
    print(f"    Magnetosphere: stationary, n = {n_sphere/1e6:.0f} cm^-3")
    print(f"    B = {B*1e9:.0f} nT on both sides")
    print(f"    rho_sheath = {rho1:.3e} kg/m^3")
    print(f"    rho_sphere = {rho2:.3e} kg/m^3")

    # KHI threshold for incompressible flow:
    # (rho1 * rho2 * (Delta_v)^2) / (rho1 + rho2)^2 > (B1.k)^2/(mu0*(rho1+rho2))
    #                                                   + (B2.k)^2/(mu0*(rho1+rho2))
    # For B.k = 0 (waves perpendicular to B):
    # The magnetic tension term vanishes, threshold velocity = 0!

    print(f"\n(a) KHI threshold for waves perpendicular to B (B.k = 0):")
    print(f"    The KHI criterion:")
    print(f"    rho1*rho2*(Delta_v)^2/(rho1+rho2)^2 > "
          f"(B1.k)^2 + (B2.k)^2) / (mu0*(rho1+rho2))")
    print(f"    For B.k = 0, the RHS = 0")
    print(f"    => ANY velocity shear is unstable!")
    print(f"    Threshold velocity = 0 km/s")

    # (b) Dawn flank stability
    print(f"\n(b) Dawn flank stability:")
    print(f"    v_sheath = {v_sheath/1e3:.0f} km/s > 0 km/s (threshold)")
    print(f"    => Dawn flank is KH-UNSTABLE for waves perpendicular to B")
    print(f"    This is why KH waves are commonly observed on the flanks")

    # (c) Dusk flank with B parallel to k (B.k = B)
    # Threshold: Delta_v > v_threshold
    # v_threshold^2 = (rho1 + rho2) / (rho1 * rho2) * (B1^2 + B2^2) / mu0
    # Since B1 = B2 = B and B.k = B:
    v_thresh_sq = (rho1 + rho2) / (rho1 * rho2) * (2 * B**2) / mu0
    v_thresh = np.sqrt(v_thresh_sq)

    print(f"\n(c) Dusk flank with B parallel to k (B.k = B):")
    print(f"    v_threshold^2 = (rho1+rho2)/(rho1*rho2) * 2*B^2/mu0")
    print(f"    = ({rho1+rho2:.3e}) / ({rho1*rho2:.3e}) * "
          f"2*({B:.1e})^2 / {mu0:.3e}")
    print(f"    = {(rho1+rho2)/(rho1*rho2):.3e} * {2*B**2/mu0:.3e}")
    print(f"    v_threshold = {v_thresh:.1f} m/s = {v_thresh/1e3:.1f} km/s")

    is_stable = v_sheath < v_thresh
    print(f"\n    v_sheath = {v_sheath/1e3:.0f} km/s vs "
          f"v_threshold = {v_thresh/1e3:.1f} km/s")
    if is_stable:
        print(f"    => Dusk flank is STABLE when B is parallel to k")
        print(f"    The magnetic tension stabilizes the KH instability")
    else:
        print(f"    => Dusk flank is UNSTABLE even with B parallel to k")

    print(f"\n    Physical explanation:")
    print(f"    - When B is perpendicular to k, magnetic field lines are not")
    print(f"      bent by the wave, so there is no restoring tension force")
    print(f"    - When B is parallel to k, bending field lines costs energy")
    print(f"      (magnetic tension), creating a stabilizing restoring force")
    print(f"    - The Parker spiral geometry makes B nearly perpendicular to")
    print(f"      the flow on the dawn flank (favorable for KHI) and nearly")
    print(f"      parallel on the dusk flank (stabilizing)")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
