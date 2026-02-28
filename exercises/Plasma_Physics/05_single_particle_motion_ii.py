"""
Plasma Physics - Lesson 05: Single Particle Motion II
Exercise Solutions

Topics covered:
- Grad-B drift in a magnetic mirror
- Tokamak vertical drift and its consequences
- Polarization current in an oscillating wave field
- Loss cone and grad-B drift combined analysis
- Gravitational sedimentation in solar prominences
"""

import numpy as np
from scipy.integrate import solve_ivp

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
m_p = 1.673e-27
epsilon_0 = 8.854e-12
k_B = 1.381e-23
mu_0 = 4 * np.pi * 1e-7
eV_to_J = e
g_sun = 274.0  # Solar surface gravity [m/s^2]


def exercise_1():
    """
    Exercise 1: Grad-B Drift in a Magnetic Mirror
    Mirror field: B_z(z) = B_0 * (1 + (z/L)^2), B_0 = 0.5 T, L = 1 m.
    Calculate grad-B drift for 1 keV proton.
    """
    print("--- Exercise 1: Grad-B Drift in Magnetic Mirror ---")

    B_0 = 0.5    # T
    L = 1.0       # m (mirror scale length)
    E_perp = 1e3 * eV_to_J  # 1 keV

    # (a) Magnetic field gradient at z = 0
    # B_z(z) = B_0 * (1 + z^2/L^2)
    # dB/dz = B_0 * 2z/L^2 -> zero at z=0
    # d^2B/dz^2 = 2*B_0/L^2

    # At z = L/2:
    z_eval = L / 2
    B_at_z = B_0 * (1 + (z_eval / L)**2)
    dBdz = B_0 * 2 * z_eval / L**2

    print(f"(a) Mirror field: B_z(z) = B_0*(1 + z^2/L^2)")
    print(f"    B_0 = {B_0} T, L = {L} m")
    print(f"    At z = L/2 = {z_eval} m:")
    print(f"    B = {B_at_z:.3f} T")
    print(f"    dB/dz = {dBdz:.3f} T/m")

    # (b) Grad-B drift: v_gradB = (v_perp^2 / (2 * omega_c)) * (B x grad_B) / B^2
    # For 1D mirror, grad_B is along z, so drift is perpendicular to both B and grad_B
    # v_gradB = (m * v_perp^2) / (2 * q * B) * (1/B) * dB/dz
    # (direction perpendicular to B and grad B)

    v_perp = np.sqrt(2 * E_perp / m_p)
    omega_ci = e * B_at_z / m_p
    rho_i = v_perp / omega_ci

    # Magnitude of grad-B drift
    v_gradB = (m_p * v_perp**2) / (2 * e * B_at_z**2) * dBdz

    print(f"\n(b) Grad-B drift for 1 keV proton at z = L/2:")
    print(f"    v_perp = {v_perp:.4e} m/s")
    print(f"    omega_ci = {omega_ci:.4e} rad/s")
    print(f"    Larmor radius: rho_i = {rho_i*1e3:.3f} mm")
    print(f"    v_gradB = {v_gradB:.4f} m/s")
    print(f"    Direction: perpendicular to B and grad(B)")

    # (c) Compare to thermal speed
    print(f"\n(c) v_gradB / v_perp = {v_gradB/v_perp:.6f}")
    print(f"    Drift is much slower than thermal motion")
    print(f"    Drift parameter rho_i/L = {rho_i/L:.4e} (must be << 1 for guiding center)")

    # (d) Time to drift across 1 cm
    d_perp = 0.01  # 1 cm
    t_drift = d_perp / v_gradB
    print(f"\n(d) Time to drift 1 cm perpendicular to B:")
    print(f"    t = {t_drift:.4f} s")
    print(f"    Number of gyro-orbits: {t_drift * omega_ci / (2*np.pi):.0f}")
    print()


def exercise_2():
    """
    Exercise 2: Tokamak Vertical Drift
    In a tokamak, grad-B and curvature drifts cause vertical particle drift.
    R = 6 m, B_0 = 5 T, T = 10 keV (deuterium).
    """
    print("--- Exercise 2: Tokamak Vertical Drift ---")

    R = 6.0       # Major radius [m]
    a = 2.0       # Minor radius [m]
    B_0 = 5.0     # Toroidal field on axis [T]
    T_eV = 10e3   # 10 keV
    T_J = T_eV * eV_to_J
    m_i = 2 * m_p  # Deuterium

    # (a) Toroidal field: B_phi ~ B_0 * R / (R + r*cos(theta))
    # On axis: B = B_0 at R
    # grad B is in -R direction: |grad B| / B = 1/R (for large aspect ratio)
    # Curvature: kappa = 1/R (for circular field lines)

    print(f"(a) Tokamak parameters: R = {R} m, a = {a} m, B_0 = {B_0} T")
    print(f"    grad(B)/B = 1/R = {1/R:.3f} m^-1")
    print(f"    Field line curvature: kappa = 1/R = {1/R:.3f} m^-1")

    # (b) Combined grad-B + curvature drift (vertical, for passing particles)
    # v_d = (m/(q*B)) * (v_perp^2/2 + v_par^2) * (R_c x B) / (R_c^2 * B)
    # For tokamak: v_d_vertical = (2*T) / (q*B*R) (average over pitch angles)
    # where the factor 2 accounts for both grad-B and curvature

    v_th_i = np.sqrt(T_J / m_i)
    omega_ci = e * B_0 / m_i

    # Grad-B drift: v_gradB ~ (m * v_perp^2) / (2*q*B*R)
    # Curvature drift: v_curv ~ (m * v_par^2) / (q*B*R)
    # Total (thermal average): v_d ~ (m * v_th^2 * (1/2 + 1)) / (q*B*R) = (3/2)*T/(q*B*R)
    # More precisely: v_d = (2*T) / (q*B*R) for the combined drift

    v_d = 2 * T_J / (e * B_0 * R)

    print(f"\n(b) Vertical drift velocity (thermal deuterium, combined grad-B + curvature):")
    print(f"    v_d = 2*T / (q*B*R) = {v_d:.2f} m/s")
    print(f"    Direction: vertically up for ions, down for electrons")

    # (c) How far does an ion drift in one toroidal transit?
    v_par = v_th_i  # Typical parallel velocity
    t_transit = 2 * np.pi * R / v_par
    delta_z = v_d * t_transit

    print(f"\n(c) In one toroidal transit:")
    print(f"    v_parallel ~ v_th = {v_th_i:.4e} m/s")
    print(f"    Transit time: {t_transit*1e6:.1f} us")
    print(f"    Vertical drift: delta_z = {delta_z*1e3:.2f} mm")

    # (d) Why rotational transform (q) solves this
    print(f"\n(d) Role of rotational transform:")
    print(f"    Without twist: particles drift vertically out of the plasma")
    print(f"    With twist (q ~ 1-3): particles sample both top and bottom")
    print(f"    Average vertical drift cancels over a flux surface")
    print(f"    Banana width: w_b ~ q * rho_i * sqrt(R/r)")
    rho_i = v_th_i / omega_ci
    q = 1.5
    r = a / 2
    epsilon = r / R
    w_b = q * rho_i / np.sqrt(epsilon)
    print(f"    rho_i = {rho_i*1e3:.2f} mm")
    print(f"    Banana width (r=a/2, q={q}): w_b = {w_b*1e3:.1f} mm")
    print()


def exercise_3():
    """
    Exercise 3: Polarization Current in a Wave Field
    Time-varying E field: E(t) = E_0 * sin(omega*t), B = B_0 z-hat.
    Calculate polarization drift and current.
    """
    print("--- Exercise 3: Polarization Current in Wave ---")

    B_0 = 0.5      # T
    E_0 = 1000.0   # V/m
    omega = 1e6     # Wave frequency [rad/s]
    n = 1e19        # Density [m^-3]
    m_i = m_p       # Hydrogen

    omega_ci = e * B_0 / m_i

    # (a) Polarization drift: v_pol = (m / (q*B^2)) * dE/dt
    # For E(t) = E_0 * sin(omega*t): dE/dt = E_0 * omega * cos(omega*t)
    # v_pol = (m * omega * E_0) / (q * B^2) * cos(omega*t)

    v_pol_max = (m_i * omega * E_0) / (e * B_0**2)

    print(f"(a) Polarization drift:")
    print(f"    B = {B_0} T, E_0 = {E_0} V/m, omega = {omega:.0e} rad/s")
    print(f"    omega_ci = {omega_ci:.4e} rad/s")
    print(f"    omega / omega_ci = {omega/omega_ci:.4e}")
    print(f"    v_pol,max = m_i*omega*E_0 / (q*B^2) = {v_pol_max:.4f} m/s")

    # (b) Polarization current: j_pol = n * q * v_pol (ions dominate due to mass)
    # Since electrons also have polarization drift (opposite sign, much smaller magnitude):
    # j_pol ~ n * e * v_pol_ion (ion contribution dominates by m_i/m_e)

    j_pol_max = n * e * v_pol_max
    print(f"\n(b) Polarization current density:")
    print(f"    j_pol,max = n*e*v_pol = {j_pol_max:.4e} A/m^2")

    # Electron polarization drift (much smaller by m_e/m_i)
    v_pol_e_max = (m_e * omega * E_0) / (e * B_0**2)
    print(f"    Electron contribution: v_pol,e = {v_pol_e_max:.4e} m/s")
    print(f"    Ratio ion/electron = m_i/m_e = {m_i/m_e:.0f}")

    # (c) Validity condition: omega << omega_ci
    print(f"\n(c) Validity of guiding center theory:")
    print(f"    Requires omega << omega_ci")
    print(f"    omega / omega_ci = {omega/omega_ci:.4e}")
    print(f"    Condition satisfied? {'Yes' if omega < 0.1*omega_ci else 'No (marginal)' if omega < omega_ci else 'No'}")

    # (d) Energy stored in polarization
    # The polarization current is equivalent to a dielectric response
    # epsilon_perp = 1 + sum_s(omega_ps^2 / omega_cs^2) ~ 1 + c^2/v_A^2
    omega_pi = np.sqrt(n * e**2 / (epsilon_0 * m_i))
    epsilon_perp = 1 + omega_pi**2 / omega_ci**2
    v_A = B_0 / np.sqrt(mu_0 * n * m_i)

    print(f"\n(d) Dielectric response:")
    print(f"    omega_pi = {omega_pi:.4e} rad/s")
    print(f"    epsilon_perp = 1 + omega_pi^2/omega_ci^2 = {epsilon_perp:.2f}")
    print(f"    Alfven speed: v_A = {v_A/1e3:.1f} km/s")
    print(f"    Equivalent: epsilon_perp ~ 1 + c^2/v_A^2 = {1 + (3e8/v_A)**2:.2f}")
    print()


def exercise_4():
    """
    Exercise 4: Loss Cone and Grad-B Drift
    Magnetic mirror with R_m = 5 (mirror ratio).
    Analyze loss cone and combined effects.
    """
    print("--- Exercise 4: Loss Cone and Grad-B Drift ---")

    R_m = 5.0        # Mirror ratio B_max/B_min
    B_min = 0.1       # T (midplane)
    B_max = R_m * B_min  # T (mirror point)
    L_mirror = 2.0    # Half-length of mirror [m]
    E_keV = 1.0       # Particle energy
    E_J = E_keV * 1e3 * eV_to_J

    # (a) Loss cone angle
    sin2_alpha_lc = 1.0 / R_m
    alpha_lc = np.arcsin(np.sqrt(sin2_alpha_lc))
    alpha_lc_deg = np.degrees(alpha_lc)

    print(f"(a) Loss cone analysis:")
    print(f"    Mirror ratio: R_m = B_max/B_min = {R_m}")
    print(f"    B_min = {B_min} T (midplane), B_max = {B_max} T (mirror)")
    print(f"    Loss cone half-angle: alpha_lc = {alpha_lc_deg:.1f} degrees")
    print(f"    sin^2(alpha_lc) = 1/R_m = {sin2_alpha_lc:.3f}")

    # (b) Fraction of isotropic distribution that is lost
    # For isotropic f(v), fraction in loss cone = 1 - cos(alpha_lc)
    # (both ends: 2 * (1 - cos(alpha_lc)))
    f_trapped = np.cos(alpha_lc)
    f_lost = 1 - f_trapped  # One end
    f_lost_both = 2 * f_lost  # Both ends

    print(f"\n(b) Fraction confined (isotropic distribution):")
    print(f"    Trapped fraction (one mirror): {f_trapped:.3f}")
    print(f"    Lost fraction (one end): {f_lost:.3f}")
    print(f"    Lost fraction (both ends): {f_lost_both:.3f}")
    print(f"    Confined fraction: {1 - f_lost_both:.3f} = {(1-f_lost_both)*100:.1f}%")

    # (c) Grad-B drift at midplane for 1 keV proton
    # At midplane, the field is B_min and has a gradient from the mirror geometry
    # For a parabolic mirror: B(z) = B_min * (1 + (z/L)^2*(R_m-1))
    # dB/dr at midplane comes from the radial gradient needed for div(B) = 0

    # In a mirror, radial gradient near axis: B_r ~ -(r/2)*dB_z/dz
    # The azimuthal grad-B drift is:
    v_perp = np.sqrt(2 * E_J / m_p)
    omega_ci = e * B_min / m_p
    rho_i = v_perp / omega_ci

    # Axial gradient at z=0 is zero, but at z=L/2:
    z = L_mirror / 2
    dBdz = B_min * 2 * z * (R_m - 1) / L_mirror**2
    B_at_z = B_min * (1 + (z / L_mirror)**2 * (R_m - 1))

    v_gradB = v_perp**2 / (2 * omega_ci * B_at_z) * dBdz

    print(f"\n(c) Grad-B drift for 1 keV proton at z = L/2:")
    print(f"    B(z=L/2) = {B_at_z:.3f} T")
    print(f"    dB/dz = {dBdz:.3f} T/m")
    print(f"    v_gradB = {v_gradB:.2f} m/s")
    print(f"    rho_i = {rho_i*1e3:.2f} mm")

    # (d) Confinement time estimate limited by grad-B drift
    r_plasma = 0.1  # Plasma radius [m]
    t_loss = r_plasma / abs(v_gradB) if v_gradB != 0 else float('inf')
    n_bounces = t_loss / (4 * L_mirror / v_perp) if v_gradB != 0 else float('inf')

    print(f"\n(d) Confinement time estimate:")
    print(f"    Plasma radius: {r_plasma*100:.0f} cm")
    print(f"    Time to drift r_plasma: {t_loss:.2f} s")
    print(f"    Number of bounces: ~{n_bounces:.0f}")
    print(f"    This is why simple mirrors have poor confinement!")
    print()


def exercise_5():
    """
    Exercise 5: Gravitational Sedimentation in Solar Prominences
    g x B drift causes charge separation and resulting E x B drift.
    Solar prominence: B = 10 G, T = 8000 K, n = 10^16 m^-3, H plasma.
    """
    print("--- Exercise 5: Gravitational Sedimentation in Solar Prominences ---")

    B = 10e-4       # 10 Gauss = 10^-3 T
    T = 8000.0      # K
    T_J = k_B * T
    n = 1e16        # m^-3
    g = g_sun       # m/s^2

    print(f"Solar prominence parameters:")
    print(f"  B = 10 G = {B*1e3:.1f} mT")
    print(f"  T = {T:.0f} K = {T_J/eV_to_J:.2f} eV")
    print(f"  n = {n:.0e} m^-3")
    print(f"  g_sun = {g:.0f} m/s^2")

    # (a) Gravitational drift: v_g = (m * g x B) / (q * B^2)
    # For B in z-direction and g in -z-direction (downward):
    # g x B is in the horizontal direction
    # Ions and electrons drift in OPPOSITE directions (different charge)

    v_g_proton = m_p * g / (e * B)
    v_g_electron = m_e * g / (e * B)

    print(f"\n(a) Gravitational drift velocities:")
    print(f"    Proton: v_g,p = m_p*g/(e*B) = {v_g_proton:.4e} m/s")
    print(f"    Electron: v_g,e = m_e*g/(e*B) = {v_g_electron:.4e} m/s")
    print(f"    Ions and electrons drift in OPPOSITE directions")
    print(f"    -> Charge separation -> Electric field builds up")

    # (b) Resulting current and electric field
    # The gravitational drift produces a current: j_g = n*e*(v_g,i - v_g,e)
    # Since v_g_e << v_g_i: j_g ~ n*e*v_g,i
    j_g = n * e * (v_g_proton + v_g_electron)  # Opposite drifts = additive current
    print(f"\n(b) Gravitational drift current:")
    print(f"    j_g = n*e*(v_g,p + v_g,e) = {j_g:.4e} A/m^2")

    # (c) In equilibrium, E x B drift balances gravity
    # The charge separation creates E such that E x B drift = g x B / B^2 drift
    # is modified. In steady state, net downward flow = 0 when:
    # E x B drift compensates for the differential gravitational drift

    # The scale height is: H = kT / (m_p * g)
    H_p = k_B * T / (m_p * g)
    H_e = k_B * T / (m_e * g)

    print(f"\n(c) Scale heights:")
    print(f"    Proton scale height: H_p = kT/(m_p*g) = {H_p/1e3:.1f} km")
    print(f"    Electron scale height: H_e = kT/(m_e*g) = {H_e/1e3:.0f} km")
    print(f"    Ambipolar: H_amb ~ 2*H_p = {2*H_p/1e3:.1f} km")
    print(f"    (Electrons pull ions up, ions pull electrons down)")

    # (d) Compare to thermal speed and Alfven speed
    v_th_i = np.sqrt(k_B * T / m_p)
    v_A = B / np.sqrt(mu_0 * n * m_p)

    print(f"\n(d) Velocity comparisons:")
    print(f"    Ion thermal speed: v_th,i = {v_th_i:.1f} m/s")
    print(f"    Alfven speed: v_A = {v_A/1e3:.1f} km/s")
    print(f"    v_g / v_th = {v_g_proton/v_th_i:.6f}")
    print(f"    Gravitational drift is tiny compared to thermal motion")
    print(f"    But over many hours, it can cause measurable sedimentation")

    # (e) Sedimentation timescale
    # Time for prominence to drain: t ~ H / v_g
    H_prom = 10e6  # Typical prominence height [m] = 10,000 km
    t_sed = H_prom / v_g_proton

    print(f"\n(e) Sedimentation timescale:")
    print(f"    Prominence height ~ {H_prom/1e6:.0f} Mm")
    print(f"    t_sedimentation ~ H / v_g = {t_sed:.0e} s = {t_sed/3600:.1f} hours")
    print(f"    Real prominences last days-weeks -> magnetic support essential")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
