"""
Lesson 14: Space Weather MHD
Topic: MHD
Description: Exercises on magnetopause standoff distance, bow shock
             compression, Dst index prediction, CME transit time,
             reconnection rates, GIC hazard estimation, substorm energy,
             and magnetic cloud analysis.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def exercise_1():
    """Magnetopause Compression.

    During a CME, v_sw = 900 km/s and n_sw = 30 cm^-3.
    Calculate the magnetopause standoff distance.
    Does it compress inside geosynchronous orbit (6.6 R_E)?
    """
    v_sw = 900e3         # m/s
    n_sw = 30e6          # m^-3 (30 cm^-3)
    m_p = 1.67e-27       # kg
    mu_0 = 4 * np.pi * 1e-7
    R_E = 6.371e6        # m
    B_E = 3.11e-5        # T (Earth's equatorial dipole field)

    # Dynamic pressure: P_dyn = (1/2) * rho * v^2
    rho = n_sw * m_p
    P_dyn = 0.5 * rho * v_sw**2

    # Magnetopause standoff: pressure balance
    # P_dyn = B_mp^2 / (2*mu_0) where B_mp = B_E * (R_E/r_mp)^3
    # P_dyn = B_E^2 / (2*mu_0) * (R_E/r_mp)^6
    # r_mp = R_E * (B_E^2 / (2*mu_0*P_dyn))^(1/6)

    r_mp = R_E * (B_E**2 / (2 * mu_0 * P_dyn))**(1.0 / 6.0)
    r_mp_RE = r_mp / R_E

    print(f"  CME conditions:")
    print(f"    v_sw = {v_sw / 1e3:.0f} km/s")
    print(f"    n_sw = {n_sw / 1e6:.0f} cm^-3")
    print(f"  Dynamic pressure: P_dyn = (1/2)*rho*v^2 = {P_dyn:.3e} Pa")
    print(f"  Normal conditions: P_dyn ~ 2 nPa, r_mp ~ 10 R_E")
    print(f"  Magnetopause standoff: r_mp = {r_mp_RE:.2f} R_E")
    print(f"  Geosynchronous orbit: 6.6 R_E")

    if r_mp_RE < 6.6:
        print(f"  WARNING: Magnetopause at {r_mp_RE:.1f} R_E is INSIDE geosynchronous orbit!")
        print(f"  Satellites at GEO are directly exposed to the solar wind!")
    else:
        print(f"  Magnetopause at {r_mp_RE:.1f} R_E is outside GEO. Satellites are shielded.")

    # Comparison with quiet conditions
    n_quiet = 5e6
    v_quiet = 400e3
    P_quiet = 0.5 * n_quiet * m_p * v_quiet**2
    r_mp_quiet = R_E * (B_E**2 / (2 * mu_0 * P_quiet))**(1.0 / 6.0)
    print(f"\n  Quiet conditions (n=5 cm^-3, v=400 km/s):")
    print(f"    P_dyn = {P_quiet:.3e} Pa, r_mp = {r_mp_quiet / R_E:.1f} R_E")
    print(f"  Compression ratio: {r_mp_quiet / r_mp:.1f}x")


def exercise_2():
    """Bow Shock.

    For v_sw = 600 km/s, T_sw = 10^5 K, calculate the sonic Mach number
    and density compression ratio.
    """
    v_sw = 600e3         # m/s
    T_sw = 1e5           # K
    k_B = 1.381e-23      # J/K
    m_p = 1.67e-27       # kg
    gamma = 5.0 / 3.0    # adiabatic index

    # Sound speed: c_s = sqrt(gamma * k_B * T / m_p)
    c_s = np.sqrt(gamma * k_B * T_sw / m_p)

    # Sonic Mach number
    M_s = v_sw / c_s

    # Rankine-Hugoniot density compression for strong shock:
    # rho_2/rho_1 = (gamma+1)*M^2 / ((gamma-1)*M^2 + 2)
    compression = (gamma + 1) * M_s**2 / ((gamma - 1) * M_s**2 + 2)

    # Maximum compression (M -> infinity):
    comp_max = (gamma + 1) / (gamma - 1)

    print(f"  Solar wind: v_sw = {v_sw / 1e3:.0f} km/s, T_sw = {T_sw:.1e} K")
    print(f"  Sound speed: c_s = sqrt(gamma*k_B*T/m_p) = {c_s / 1e3:.1f} km/s")
    print(f"  Sonic Mach number: M = v_sw/c_s = {M_s:.1f}")
    print(f"  Density compression ratio (Rankine-Hugoniot):")
    print(f"    rho_2/rho_1 = (gamma+1)*M^2 / ((gamma-1)*M^2 + 2) = {compression:.2f}")
    print(f"  Maximum compression (M -> inf): (gamma+1)/(gamma-1) = {comp_max:.1f}")
    print(f"  The bow shock compresses the solar wind by factor ~{compression:.1f}")

    # Note: actual bow shock is also magnetosonic, so should include B
    # Fast magnetosonic Mach number would be even higher
    print(f"  Note: Full magnetosonic Mach number is typically M_ms ~ {M_s * 0.8:.1f}")
    print(f"  (somewhat reduced due to magnetic pressure contribution)")


def exercise_3():
    """Dst Prediction.

    Using the Burton model, estimate minimum Dst for a storm with
    E_sw = 5 mV/m sustained for 6 hours.
    """
    E_sw = 5.0           # mV/m
    duration = 6.0       # hours
    Dst_0 = 0.0          # initial Dst
    a_coeff = 1e-3       # nT/(mV/m) (injection rate coefficient)
    b_coeff = 0.5        # mV/m (threshold)
    tau = 8.0            # hours (decay time)

    # Burton model: dDst*/dt = Q(E) - Dst*/tau
    # Q(E) = a * (E_sw - b) for E_sw > b, else 0
    # Dst* = Dst - b*sqrt(P_dyn) + c (pressure-corrected Dst)

    # Simplified: ignore pressure correction (Dst* ~ Dst)
    if E_sw > b_coeff:
        Q = a_coeff * (E_sw - b_coeff)  # nT/hour... but let's use consistent units
    else:
        Q = 0

    # Actually, let's use the more standard formulation:
    # Q = -4.4 * (E_sw - 0.5) nT/hr for E_sw > 0.5 mV/m
    Q = -4.4 * (E_sw - 0.5)  # nT/hr (negative = ring current injection)

    # Time integration
    dt = 0.01  # hours
    t = np.arange(0, 24, dt)
    Dst = np.zeros_like(t)
    Dst[0] = Dst_0

    for i in range(len(t) - 1):
        # Injection during first 'duration' hours
        if t[i] < duration:
            Q_now = Q
        else:
            Q_now = 0

        # Recovery
        dDst = Q_now - Dst[i] / tau
        Dst[i + 1] = Dst[i] + dt * dDst

    Dst_min = np.min(Dst)
    t_min = t[np.argmin(Dst)]

    print(f"  Burton model parameters:")
    print(f"    E_sw = {E_sw} mV/m, duration = {duration} hours")
    print(f"    Decay time tau = {tau} hours")
    print(f"  Injection rate: Q = -4.4*(E-0.5) = {Q:.1f} nT/hr")
    print(f"  Minimum Dst = {Dst_min:.1f} nT at t = {t_min:.1f} hours")

    if Dst_min > -50:
        storm_class = "Weak"
    elif Dst_min > -100:
        storm_class = "Moderate"
    elif Dst_min > -200:
        storm_class = "Intense"
    elif Dst_min > -350:
        storm_class = "Super-intense"
    else:
        storm_class = "Extreme"

    print(f"  Storm classification: {storm_class} (Dst = {Dst_min:.0f} nT)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, Dst, 'b-', linewidth=2)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(-50, color='orange', linestyle=':', alpha=0.7, label='Moderate (-50 nT)')
    ax.axhline(-100, color='red', linestyle=':', alpha=0.7, label='Intense (-100 nT)')
    ax.axvspan(0, duration, alpha=0.1, color='yellow', label='CME driving')
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Dst (nT)', fontsize=12)
    ax.set_title('Dst Index: Burton Model Prediction', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('14_dst_prediction.png', dpi=150)
    plt.close()
    print("  Plot saved to 14_dst_prediction.png")


def exercise_4():
    """CME Transit Time.

    A CME is launched at v_0 = 1500 km/s. Using the drag model with
    v_sw = 400 km/s and gamma^-1 = 1 day, estimate arrival time at Earth.
    """
    v_0 = 1500e3         # m/s (initial CME speed)
    v_sw = 400e3         # m/s (ambient solar wind)
    gamma_drag = 1.0 / 86400.0  # s^-1 (drag coefficient, 1/1 day)
    AU = 1.496e11        # m
    R_sun = 6.96e8       # m
    r_start = 20 * R_sun  # starting radius (after CME liftoff)

    # Drag model: dv/dt = -gamma * (v - v_sw) * |v - v_sw|
    # For v > v_sw: v(t) = v_sw + (v_0 - v_sw) * exp(-gamma * (v_0 - v_sw) * t)
    # This is the "linear drag" approximation
    # More common: aerodynamic drag dv/dt = -gamma*(v-v_sw)|v-v_sw|

    # Solve ODE: dv/dt = -gamma*(v-v_sw)*|v-v_sw|
    # dr/dt = v
    def cme_drag(t, y):
        r, v = y
        dv_dt = -gamma_drag * (v - v_sw) * abs(v - v_sw)
        return [v, dv_dt]

    # Integrate until reaching 1 AU
    def event_earth(t, y):
        return y[0] - AU

    event_earth.terminal = True
    event_earth.direction = 1

    sol = solve_ivp(cme_drag, [0, 10 * 86400], [r_start, v_0],
                    events=event_earth, max_step=100, dense_output=True)

    if sol.t_events[0].size > 0:
        t_arrival = sol.t_events[0][0]
        v_arrival = sol.y_events[0][0][1]
    else:
        t_arrival = sol.t[-1]
        v_arrival = sol.y[1, -1]

    t_arrival_hr = t_arrival / 3600
    t_arrival_days = t_arrival / 86400

    print(f"  CME parameters:")
    print(f"    Initial speed v_0 = {v_0 / 1e3:.0f} km/s")
    print(f"    Ambient solar wind v_sw = {v_sw / 1e3:.0f} km/s")
    print(f"    Drag parameter gamma = 1/{1.0 / gamma_drag / 86400:.0f} day")
    print(f"  Arrival at 1 AU:")
    print(f"    Transit time = {t_arrival_hr:.1f} hours = {t_arrival_days:.2f} days")
    print(f"    Arrival speed = {v_arrival / 1e3:.0f} km/s")

    # Ballistic estimate (no drag)
    t_ballistic = (AU - r_start) / v_0
    print(f"  Ballistic (no drag): {t_ballistic / 3600:.1f} hours")
    print(f"  Drag slows CME by {(t_arrival - t_ballistic) / 3600:.1f} hours")

    # Plot velocity and distance profiles
    t_plot = np.linspace(0, t_arrival * 1.2, 500)
    sol_dense = sol.sol(t_plot)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(t_plot / 3600, sol_dense[1] / 1e3, 'b-', linewidth=2)
    ax1.axhline(v_sw / 1e3, color='red', linestyle='--', label=f'Solar wind ({v_sw / 1e3:.0f} km/s)')
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('CME Speed (km/s)', fontsize=12)
    ax1.set_title('CME Velocity Profile (Drag Model)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_plot / 3600, sol_dense[0] / AU, 'b-', linewidth=2)
    ax2.axhline(1.0, color='red', linestyle='--', label='1 AU (Earth)')
    ax2.axvline(t_arrival / 3600, color='green', linestyle=':', label=f'Arrival: {t_arrival_hr:.1f} hr')
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Distance (AU)', fontsize=12)
    ax2.set_title('CME Distance Profile', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('14_cme_transit.png', dpi=150)
    plt.close()
    print("  Plot saved to 14_cme_transit.png")


def exercise_5():
    """Reconnection Rate.

    Calculate the reconnection electric field for northward and southward IMF.
    v_sw = 400 km/s, B_sw = 5 nT.
    """
    v_sw = 400e3         # m/s
    B_sw = 5e-9          # T (5 nT)

    # (a) Northward IMF: theta = 0
    theta_N = 0.0        # degrees
    theta_N_rad = np.radians(theta_N)

    # Reconnection electric field: E = v_sw * B_sw * sin^2(theta/2)
    # (coupling function based on clock angle)
    E_N = v_sw * B_sw * np.sin(theta_N_rad / 2)**2

    # (b) Southward IMF: theta = 180
    theta_S = 180.0      # degrees
    theta_S_rad = np.radians(theta_S)
    E_S = v_sw * B_sw * np.sin(theta_S_rad / 2)**2

    print(f"  Solar wind: v_sw = {v_sw / 1e3:.0f} km/s, B_sw = {B_sw * 1e9:.0f} nT")
    print()
    print(f"  (a) Northward IMF (theta = {theta_N}°):")
    print(f"    E_rec = v_sw * B_sw * sin^2(theta/2) = {E_N * 1e3:.4f} mV/m")
    print(f"    Minimal reconnection => quiet magnetosphere")
    print()
    print(f"  (b) Southward IMF (theta = {theta_S}°):")
    print(f"    E_rec = v_sw * B_sw * sin^2(theta/2) = {E_S * 1e3:.2f} mV/m")
    print(f"    Maximum reconnection => geomagnetic activity")
    print()
    print(f"  Ratio (southward/northward): ")
    if E_N > 0:
        print(f"    E_S / E_N = {E_S / E_N:.1f}")
    else:
        print(f"    E_S / E_N = infinity (no reconnection for pure northward)")
    print(f"  Southward IMF drives reconnection at the dayside magnetopause,")
    print(f"  opening magnetic flux and enabling solar wind energy entry.")

    # Plot reconnection rate vs clock angle
    theta_range = np.linspace(0, 360, 361)
    E_rec = v_sw * B_sw * np.sin(np.radians(theta_range) / 2)**2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta_range, E_rec * 1e3, 'b-', linewidth=2)
    ax.set_xlabel('IMF clock angle (degrees)', fontsize=12)
    ax.set_ylabel('Reconnection E-field (mV/m)', fontsize=12)
    ax.set_title('Reconnection Rate vs IMF Clock Angle', fontsize=13)
    ax.axvline(180, color='red', linestyle=':', alpha=0.7, label='Pure southward')
    ax.axvline(0, color='blue', linestyle=':', alpha=0.7, label='Pure northward')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('14_reconnection_rate.png', dpi=150)
    plt.close()
    print("  Plot saved to 14_reconnection_rate.png")


def exercise_6():
    """GIC Hazard.

    During a storm, dB/dt = 1000 nT/min. Estimate induced E-field
    and GIC in a 200 km transmission line.
    """
    dB_dt = 1000e-9 / 60.0  # T/s (1000 nT/min)
    rho_earth = 1000.0   # Ohm*m (ground resistivity)
    L_line = 200e3       # m (transmission line length)
    R_line = 0.2         # Ohm (line resistance)
    mu_0 = 4 * np.pi * 1e-7

    # Induced electric field (plane-wave approximation):
    # E = sqrt(omega * mu_0 * rho / 2) * B
    # For a step function (simplification): E ~ sqrt(mu_0 * rho) * dB/dt / sqrt(period)
    # Simplified: E ~ dB/dt * sqrt(rho * mu_0 / omega)
    # At ~1 mHz: omega ~ 6e-3 rad/s

    # More direct estimate: E ~ dB/dt * skin_depth
    # skin_depth = sqrt(2*rho/(omega*mu_0))
    omega = 2 * np.pi / (60 * 10)  # ~10 minute period
    skin_depth = np.sqrt(2 * rho_earth / (omega * mu_0))

    # Simple plane-wave: E = sqrt(omega*mu_0*rho/2) * dB/dt / omega
    # Better: for the quasi-static limit: E ~ dB/dt * sqrt(rho/(mu_0*omega))
    E_field = np.sqrt(omega * mu_0 * rho_earth / 2) * (dB_dt / omega * 60)  # approximate

    # Direct dimensional estimate: E ~ dB/dt * L_scale
    # where L_scale ~ sqrt(rho / (mu_0 * frequency))
    freq = 1.0 / 600.0  # Hz
    E_direct = dB_dt * np.sqrt(rho_earth / (mu_0 * 2 * np.pi * freq))

    # GIC = V / R = E * L / R
    V_induced = E_direct * L_line
    GIC = V_induced / R_line

    print(f"  Storm conditions:")
    print(f"    dB/dt = 1000 nT/min = {dB_dt:.3e} T/s")
    print(f"    Ground resistivity = {rho_earth} Ohm*m")
    print(f"    Transmission line: L = {L_line / 1e3:.0f} km, R = {R_line} Ohm")
    print(f"  Skin depth (period ~ 10 min): {skin_depth / 1e3:.1f} km")
    print(f"  Induced electric field: E ~ {E_direct * 1e3:.2f} mV/m = {E_direct:.3e} V/m")
    print(f"  Induced voltage: V = E * L = {V_induced:.1f} V")
    print(f"  GIC = V / R = {GIC:.1f} A")
    print()

    if GIC > 10:
        print(f"  WARNING: GIC = {GIC:.0f} A exceeds transformer damage threshold (~10-100 A)!")
        print(f"  This can cause:")
        print(f"    - Transformer saturation and overheating")
        print(f"    - Harmonic distortion in the power grid")
        print(f"    - Voltage instability and potential blackout")
        print(f"  Example: 1989 Quebec blackout (GIC ~ 100 A, dB/dt ~ 500 nT/min)")
    else:
        print(f"  GIC = {GIC:.1f} A: moderate concern for the power grid.")


def exercise_7():
    """Substorm Energy.

    A substorm releases 10^15 J over 30 minutes into the ionosphere
    at 100 km altitude over 10^12 m^2 area. Estimate energy flux.
    """
    E_total = 1e15       # J
    duration = 30 * 60   # s (30 minutes)
    area = 1e12          # m^2
    solar_constant = 1360  # W/m^2

    # Power
    P = E_total / duration

    # Energy flux
    flux = P / area

    print(f"  Substorm parameters:")
    print(f"    Total energy: E = {E_total:.1e} J")
    print(f"    Duration: {duration / 60:.0f} minutes")
    print(f"    Deposition area: {area:.1e} m^2 = {area / 1e6:.0e} km^2")
    print(f"  Power: P = E/t = {P:.3e} W")
    print(f"  Energy flux: F = P/A = {flux:.3e} W/m^2 = {flux:.1f} mW/m^2")
    print(f"  Solar constant: {solar_constant} W/m^2")
    print(f"  Ratio F/F_sun = {flux / solar_constant:.4e}")
    print(f"  The substorm energy flux is much less than solar irradiance,")
    print(f"  but concentrated in the auroral zone it drives significant")
    print(f"  ionospheric heating, currents, and aurora.")


def exercise_8():
    """Ring Current Energy.

    Dst ~ -E_ring / (4e14 J/nT). For Dst = -150 nT, estimate ring current energy.
    """
    Dst = -150.0         # nT
    conversion = 4e14    # J/nT (Dessler-Parker-Sckopke relation)

    E_ring = -Dst * conversion

    print(f"  Dessler-Parker-Sckopke relation:")
    print(f"    Dst ~ -E_ring / ({conversion:.0e} J/nT)")
    print(f"  For Dst = {Dst:.0f} nT:")
    print(f"    E_ring = -Dst * {conversion:.0e} = {E_ring:.3e} J")
    print(f"    E_ring = {E_ring / 1e15:.1f} PJ (petajoules)")
    print(f"  For comparison:")
    print(f"    Large nuclear weapon: ~10^15 J")
    print(f"    US daily electricity: ~1.2 x 10^16 J")
    print(f"    Ring current during intense storm: {E_ring:.1e} J")
    print(f"  This energy is stored in trapped ions (mostly keV protons and O+)")
    print(f"  orbiting Earth at 3-8 R_E in the ring current.")


def exercise_9():
    """CME Magnetic Cloud.

    A magnetic cloud has B ~ 30 nT, n ~ 10 cm^-3, T ~ 10^4 K.
    Calculate plasma beta. Is this consistent with a flux rope structure?
    """
    B = 30e-9            # T
    n = 10e6             # m^-3 (10 cm^-3)
    T = 1e4              # K
    k_B = 1.381e-23      # J/K
    mu_0 = 4 * np.pi * 1e-7
    m_p = 1.67e-27       # kg

    # Gas pressure (proton + electron): p = 2 * n * k_B * T
    p = 2 * n * k_B * T

    # Magnetic pressure: p_mag = B^2 / (2*mu_0)
    p_mag = B**2 / (2 * mu_0)

    # Plasma beta
    beta = p / p_mag

    print(f"  Magnetic cloud parameters:")
    print(f"    B = {B * 1e9:.0f} nT")
    print(f"    n = {n / 1e6:.0f} cm^-3")
    print(f"    T = {T:.1e} K")
    print(f"  Gas pressure (p = 2*n*k_B*T): p = {p:.3e} Pa = {p * 1e9:.2f} nPa")
    print(f"  Magnetic pressure (B^2/(2*mu_0)): p_mag = {p_mag:.3e} Pa = {p_mag * 1e9:.2f} nPa")
    print(f"  Plasma beta = p / p_mag = {beta:.4f}")
    print()

    if beta < 1:
        print(f"  beta = {beta:.3f} << 1 => MAGNETICALLY DOMINATED")
        print(f"  Consistent with a flux rope structure where:")
        print(f"    - Magnetic forces dominate over pressure forces")
        print(f"    - The field maintains coherent helical structure")
        print(f"    - Low temperature (expansion cooling in transit)")
        print(f"    - High field strength (compressed/amplified)")
    else:
        print(f"  beta > 1: not magnetically dominated")


def exercise_10():
    """Space Weather Forecasting.

    Explain why CME arrival time can be predicted ~12 hours in advance,
    but storm intensity (Dst minimum) is uncertain until L1.
    """
    print("  Space Weather Forecasting Challenge:")
    print()
    print("  CME Arrival Time (predictable ~12-24 hours ahead):")
    print("  - CME speed can be estimated from coronagraph images (SOHO/LASCO)")
    print("  - Speed at launch correlates with transit time (1-5 days)")
    print("  - Drag-based models use initial speed + solar wind conditions")
    print("  - STEREO provides side views for better 3D speed estimates")
    print("  - Accuracy: ~6-12 hours uncertainty (reasonable)")
    print()
    print("  Storm Intensity (uncertain until L1 data):")
    print("  - Dst depends critically on the IMF Bz (north-south component)")
    print("  - Bz cannot be measured remotely from the Sun:")
    print("    * Coronagraph images show white light (no B field info)")
    print("    * EUV/X-ray images show plasma, not field polarity")
    print("    * Heliospheric models predict |B| but not B direction")
    print("  - The flux rope orientation changes during transit (rotation, deflection)")
    print("  - Only L1 monitors (ACE, DSCOVR) measure actual Bz at 1 AU")
    print("  - L1 is ~1.5 million km upstream => only ~30-60 min warning")
    print()
    print("  Critical missing information: IMF Bz component at Earth")
    print("  - This determines reconnection rate and energy coupling")
    print("  - Southward Bz => intense storm; northward Bz => weak storm")
    print("  - Same CME can produce Dst = -50 nT or Dst = -300 nT")
    print("    depending on Bz orientation!")
    print()
    print("  Current research:")
    print("  - Faraday rotation measurements (radio propagation through CME)")
    print("  - AI/ML models to predict Bz from solar source region")
    print("  - Ensemble modeling with uncertainty quantification")
    print("  - In-situ fleet at sub-L1 distances for longer warning time")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Magnetopause Compression", exercise_1),
        ("Exercise 2: Bow Shock", exercise_2),
        ("Exercise 3: Dst Prediction", exercise_3),
        ("Exercise 4: CME Transit Time", exercise_4),
        ("Exercise 5: Reconnection Rate", exercise_5),
        ("Exercise 6: GIC Hazard", exercise_6),
        ("Exercise 7: Substorm Energy", exercise_7),
        ("Exercise 8: Ring Current Energy", exercise_8),
        ("Exercise 9: CME Magnetic Cloud", exercise_9),
        ("Exercise 10: Space Weather Forecasting", exercise_10),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
        func()
