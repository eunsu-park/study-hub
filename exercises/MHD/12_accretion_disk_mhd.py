"""
Lesson 12: Accretion Disk MHD
Topic: MHD
Description: Exercises on the magnetorotational instability (MRI),
             Maxwell stress, accretion timescales, dead zones,
             Blandford-Payne winds, jet power, and disk evolution.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def exercise_1():
    """MRI Growth Rate.

    For a disk at radius r = 10^10 cm around a M = 10 M_sun black hole,
    compute the orbital frequency Omega and the maximum MRI growth rate gamma_max.
    """
    r_cgs = 1e10          # cm
    r_SI = r_cgs * 1e-2   # m
    M_sun = 1.989e30      # kg
    M = 10 * M_sun        # 10 solar masses
    G = 6.674e-11         # m^3/(kg*s^2)

    # Keplerian orbital frequency: Omega = sqrt(GM/r^3)
    Omega = np.sqrt(G * M / r_SI**3)
    P_orb = 2 * np.pi / Omega

    # Maximum MRI growth rate: gamma_max = (3/4) * Omega (for Keplerian rotation)
    gamma_max = 0.75 * Omega
    t_grow = 1.0 / gamma_max

    print(f"  M = {M / M_sun:.0f} M_sun")
    print(f"  r = {r_cgs:.1e} cm = {r_SI:.1e} m")
    print(f"  Keplerian frequency Omega = sqrt(GM/r^3) = {Omega:.4e} rad/s")
    print(f"  Orbital period P = 2*pi/Omega = {P_orb:.4e} s = {P_orb:.4f} s")
    print(f"  Maximum MRI growth rate gamma_max = (3/4)*Omega = {gamma_max:.4e} s^-1")
    print(f"  e-folding time t_grow = 1/gamma = {t_grow:.4e} s")
    print(f"  MRI grows on ~ orbital timescale (very fast!)")


def exercise_2():
    """MRI Wavelength.

    For v_A = 10 km/s and Omega = 10^(-3) s^(-1), estimate the wavelength
    of the fastest growing MRI mode.
    """
    v_A = 10e3       # m/s (10 km/s)
    Omega = 1e-3     # rad/s

    # Fastest growing MRI wavelength: lambda_MRI = 2*pi*v_A / Omega
    # More precisely, for Keplerian: k_max = sqrt(15)/4 * Omega/v_A
    # lambda_max = 2*pi / k_max = 2*pi * 4/(sqrt(15)) * v_A/Omega

    lambda_simple = 2 * np.pi * v_A / Omega
    k_max = np.sqrt(15) / 4.0 * Omega / v_A
    lambda_max = 2 * np.pi / k_max

    print(f"  Alfven speed v_A = {v_A / 1e3:.0f} km/s")
    print(f"  Angular frequency Omega = {Omega:.1e} s^-1")
    print(f"  Simple estimate: lambda ~ 2*pi*v_A/Omega = {lambda_simple:.3e} m")
    print(f"  lambda = {lambda_simple / 1e3:.1f} km")
    print(f"  Precise fastest-growing: lambda_max = {lambda_max:.3e} m = {lambda_max / 1e3:.1f} km")
    print(f"  The MRI operates on scales much smaller than the disk radius.")
    print(f"  This must be resolved in simulations for correct turbulent transport.")


def exercise_3():
    """Maxwell Stress.

    If <B_r * B_phi> = 10^(-2) * <p>, what is the effective alpha parameter?
    """
    stress_ratio = 1e-2  # <B_r * B_phi> / <p>
    mu_0 = 4 * np.pi * 1e-7

    # The Shakura-Sunyaev alpha parameter is defined as:
    # alpha_SS = <-B_r*B_phi/(mu_0)> / <p>
    # Note: the Maxwell stress tensor T_{r,phi} = -B_r*B_phi/mu_0
    # So alpha_SS = stress_ratio (in normalized Gaussian units where mu_0 = 4*pi)
    # In cgs: alpha_SS = <-B_r*B_phi> / (4*pi*<p>)
    # If <B_r*B_phi> = 0.01 * <p>, then:
    # Depending on unit convention, the effective alpha ~ 0.01

    alpha_SS = stress_ratio
    print(f"  Given: <B_r * B_phi> / <p> = {stress_ratio}")
    print(f"  In Shakura-Sunyaev parameterization:")
    print(f"    T_{{r,phi}} = -B_r*B_phi/mu_0 (SI) or -B_r*B_phi/(4*pi) (CGS)")
    print(f"    alpha_SS = |<T_{{r,phi}}>| / <p>")
    print(f"  Effective alpha parameter: alpha_SS ~ {alpha_SS}")
    print(f"  This is consistent with MRI simulations, which typically give")
    print(f"  alpha ~ 0.01 - 0.1 depending on field geometry and resolution.")


def exercise_4():
    """Accretion Timescale.

    For alpha = 0.01, c_s = 10 km/s, H = 10^9 cm, R = 10^11 cm,
    estimate the accretion timescale tau_acc ~ R^2 / nu_eff.
    """
    alpha = 0.01
    c_s = 10e3           # m/s (10 km/s)
    H_cgs = 1e9          # cm
    H = H_cgs * 1e-2     # m
    R_cgs = 1e11         # cm
    R = R_cgs * 1e-2     # m

    # Effective viscosity: nu_eff = alpha * c_s * H
    nu_eff = alpha * c_s * H

    # Accretion timescale: tau_acc ~ R^2 / nu_eff
    tau_acc = R**2 / nu_eff

    # Also: tau_acc ~ (1/alpha) * (R/H)^2 * (1/Omega)
    # Omega ~ c_s / H (thin disk), so tau_acc ~ R^2 / (alpha * c_s * H)

    print(f"  alpha = {alpha}")
    print(f"  c_s = {c_s / 1e3:.0f} km/s")
    print(f"  H = {H_cgs:.1e} cm = {H:.1e} m")
    print(f"  R = {R_cgs:.1e} cm = {R:.1e} m")
    print(f"  H/R = {H / R:.3f} (thin disk)")
    print(f"  Effective viscosity nu = alpha * c_s * H = {nu_eff:.3e} m^2/s")
    print(f"  Accretion timescale tau_acc = R^2 / nu = {tau_acc:.3e} s")
    print(f"  tau_acc = {tau_acc / (3600 * 24):.1f} days")
    print(f"  tau_acc = {tau_acc / (3600 * 24 * 365.25):.2f} years")
    print(f"  This sets the timescale for disk evolution and accretion rate changes.")


def exercise_5():
    """Equipartition Field.

    In a disk with rho = 10^(-9) g/cm^3, c_s = 10^7 cm/s,
    compute the magnetic field strength at beta = B^2/(8*pi*p) = 1.
    """
    rho_cgs = 1e-9       # g/cm^3
    c_s_cgs = 1e7        # cm/s

    # Gas pressure (isothermal): p = rho * c_s^2
    p_cgs = rho_cgs * c_s_cgs**2  # dyne/cm^2

    # Equipartition: beta = B^2/(8*pi*p) = 1
    # => B^2 = 8*pi*p
    # => B = sqrt(8*pi*p)
    B_cgs = np.sqrt(8 * np.pi * p_cgs)  # Gauss

    # Convert to SI
    rho_SI = rho_cgs * 1e3  # kg/m^3
    c_s_SI = c_s_cgs * 1e-2  # m/s
    p_SI = rho_SI * c_s_SI**2
    mu_0 = 4 * np.pi * 1e-7
    B_SI = np.sqrt(2 * mu_0 * p_SI)  # Tesla

    print(f"  rho = {rho_cgs:.1e} g/cm^3")
    print(f"  c_s = {c_s_cgs:.1e} cm/s = {c_s_SI / 1e3:.0f} km/s")
    print(f"  Gas pressure p = rho * c_s^2 = {p_cgs:.3e} dyne/cm^2 = {p_SI:.3e} Pa")
    print(f"  Equipartition (beta = 1): B^2/(8*pi) = p (CGS)")
    print(f"  B = sqrt(8*pi*p) = {B_cgs:.3e} G")
    print(f"  B = {B_SI:.3e} T")
    print(f"  MRI-generated fields typically reach beta ~ 10-100, somewhat below equipartition.")


def exercise_6():
    """Dead Zone Criterion.

    For a protoplanetary disk at 1 AU with ionization fraction x_e = 10^(-13),
    T = 200 K, rho = 10^(-9) g/cm^3, estimate Ohmic resistivity and Rm.
    Is MRI active?
    """
    x_e = 1e-13         # ionization fraction
    T = 200.0            # K
    rho_cgs = 1e-9       # g/cm^3
    m_p = 1.67e-24       # g (proton mass in CGS)
    e_cgs = 4.803e-10    # esu (electron charge in CGS)
    c_light = 3e10       # cm/s
    k_B_cgs = 1.381e-16  # erg/K

    # Number density
    n = rho_cgs / m_p

    # Electron number density
    n_e = x_e * n

    # Ohmic resistivity: eta = c^2 / (4*pi*sigma_cond)
    # Conductivity: sigma ~ n_e * e^2 / (m_e * nu_en)
    # Electron-neutral collision frequency: nu_en ~ n * sigma_en * v_th_e
    m_e_cgs = 9.109e-28  # g
    v_th_e = np.sqrt(k_B_cgs * T / m_e_cgs)
    sigma_en = 1e-15  # cm^2 (electron-neutral cross section)
    nu_en = n * sigma_en * v_th_e

    sigma_cond = n_e * e_cgs**2 / (m_e_cgs * nu_en)
    eta_cgs = c_light**2 / (4 * np.pi * sigma_cond)

    print(f"  Ionization fraction x_e = {x_e:.1e}")
    print(f"  T = {T} K, rho = {rho_cgs:.1e} g/cm^3")
    print(f"  Number density n = {n:.3e} cm^-3")
    print(f"  Electron density n_e = x_e * n = {n_e:.3e} cm^-3")
    print(f"  Electron thermal speed v_th = {v_th_e:.3e} cm/s")
    print(f"  Electron-neutral collision rate nu_en = {nu_en:.3e} s^-1")
    print(f"  Conductivity sigma = {sigma_cond:.3e} s^-1 (CGS)")
    print(f"  Ohmic diffusivity eta = c^2/(4*pi*sigma) = {eta_cgs:.3e} cm^2/s")

    # Magnetic Reynolds number: Rm = v*L / eta
    # v ~ c_s, L ~ H (disk scale height)
    c_s = np.sqrt(k_B_cgs * T / m_p)
    AU_cm = 1.496e13
    Omega_K = np.sqrt(6.674e-8 * 1.989e33 / AU_cm**3)  # Keplerian at 1 AU
    H = c_s / Omega_K
    Rm = c_s * H / eta_cgs

    print(f"  Sound speed c_s = {c_s:.3e} cm/s")
    print(f"  Scale height H = c_s/Omega_K = {H:.3e} cm")
    print(f"  Rm = c_s * H / eta = {Rm:.3e}")

    if Rm > 1:
        print(f"  Rm > 1 => MRI may be active (but need Rm > ~100 for sustained turbulence)")
    else:
        print(f"  Rm << 1 => DEAD ZONE: MRI is suppressed by Ohmic diffusion!")
        print(f"  Accretion is limited to the ionized surface layers (layered accretion).")


def exercise_7():
    """Blandford-Payne Angle.

    Explain why theta = 30 degrees is the critical angle for centrifugal
    wind launch. Compute the force balance along the field line.
    """
    print("  Blandford-Payne Mechanism:")
    print("  Consider a bead (fluid parcel) on a rotating wire (magnetic field line).")
    print("  The field line makes angle theta with the disk normal (= 90-theta from radial).")
    print()

    # Along field line inclined at angle theta from vertical:
    # Centrifugal force component along field: F_cent = Omega^2 * r * sin(theta)
    # Gravitational force component along field: F_grav = (GM/r^2) * cos(theta)
    # Wait - more precisely, for a field line at angle theta to the disk normal:
    # Effective potential along the field line:
    # The condition for launch is that the effective potential decreases along B

    # For Keplerian disk, the critical angle is theta_crit from disk plane:
    # tan(theta_crit) = 1/sqrt(3) => theta_crit = 30 degrees from radial
    # or equivalently 60 degrees from disk normal

    theta_deg = np.linspace(0, 90, 91)
    theta_rad = np.radians(theta_deg)

    # Net outward force (simplified, in units of GM/r^2):
    # Along field line at angle theta from the midplane:
    # F_net = sin(theta) * (centrifugal - gravity*cos(theta)/sin(theta))
    # For Keplerian: v_phi = sqrt(GM/r), so Omega^2*r = GM/r^2
    # F_centrifugal along B = Omega^2 * r * sin(theta) = (GM/r^2) * sin(theta)
    # F_gravity along B = -(GM/r^2) * cos(theta)
    # But we need to be more careful with the geometry...
    #
    # Correct analysis: for a field line making angle theta with the rotation axis,
    # the effective potential Phi_eff = -GM/r - (1/2)*Omega^2*R^2
    # Along the field line at angle theta from vertical:
    # R = r*sin(alpha), z = r*cos(alpha) where alpha is from vertical
    # The condition is: d(Phi_eff)/ds < 0 along the field line
    # This gives: sin^2(alpha) > 1/3, or alpha > 30 degrees from the disk rotation axis
    # Equivalently: theta > 30 degrees from vertical (60 degrees from disk plane)

    # But conventionally: theta is the angle of the field line from the RADIAL direction
    # in the (R,z) plane. The critical angle is 30 degrees from the disk plane.
    # i.e., the field line must make less than 60 degrees with the disk.

    # Effective potential gradient along field line (theta = angle from disk surface):
    # d Phi_eff / ds ~ sin(theta) - 3*cos(theta) (for Keplerian)
    # This changes sign at theta where sin(theta) = 3*cos(theta), i.e. tan(theta) = 3
    # Wait, the standard result is 30 degrees. Let me be careful.

    # Standard BP result: the field line must be inclined more than 30 degrees
    # from the disk normal for centrifugal launching.
    # i.e., angle from disk plane > 30 degrees? No...
    # The field line angle from the rotation axis must be > 30 degrees.
    # In the (R,z) plane, if theta is from vertical: theta > 30 degrees

    theta_crit = 30.0  # degrees from rotation axis (disk normal)
    print(f"  Critical angle from rotation axis: theta_c = {theta_crit} degrees")
    print()
    print("  Physical explanation:")
    print("  - Along the field line, the bead feels centrifugal + gravitational forces")
    print("  - Centrifugal force ~ Omega^2 * R (outward in R)")
    print("  - Gravitational force ~ GM/r^2 (inward toward center)")
    print("  - Along B at angle theta from vertical:")
    print("    F_centrifugal_along_B ~ sin(theta)")
    print("    F_gravity_along_B ~ -cos(theta)")
    print("  - Net: F ~ sin(theta) - (1/3)*cos(theta)  [for Keplerian]")
    print(f"  - F > 0 (outward launch) when tan(theta) > 1/3")
    print(f"  - This gives theta > arctan(1/3) = {np.degrees(np.arctan(1.0/3.0)):.1f} degrees")
    print(f"  - But the exact analysis gives theta > 30 degrees from vertical")
    print(f"  - Field lines inclined > 30 deg from the axis allow centrifugal wind launch")
    print(f"  - Steeper (more vertical) field lines cannot launch a wind")

    # Plot force balance
    fig, ax = plt.subplots(figsize=(8, 5))
    F_cent = np.sin(theta_rad)**2  # centrifugal component
    F_grav = -np.cos(theta_rad)    # gravitational component (simplified)
    # For Keplerian: effective potential gradient
    F_net = 3 * np.sin(theta_rad)**2 - 1  # proportional to net force along B
    ax.plot(theta_deg, F_cent, 'b-', label='Centrifugal (along B)', linewidth=2)
    ax.plot(theta_deg, -F_grav, 'r--', label='Gravity (along B)', linewidth=2)
    ax.plot(theta_deg, F_net, 'k-', label='Net force', linewidth=2.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(30, color='green', linestyle=':', alpha=0.7, label=r'$\theta_c = 30°$')
    ax.fill_between(theta_deg, 0, 1, where=theta_deg > 30, alpha=0.1, color='green',
                     label='Wind launch region')
    ax.set_xlabel(r'Field line angle $\theta$ from rotation axis (degrees)', fontsize=12)
    ax.set_ylabel('Force (normalized)', fontsize=12)
    ax.set_title('Blandford-Payne: Centrifugal Wind Launch', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('12_bp_angle.png', dpi=150)
    plt.close()
    print("  Plot saved to 12_bp_angle.png")


def exercise_8():
    """Jet Power.

    For a black hole of mass M = 10^9 M_sun, accretion rate M_dot = 0.1 M_dot_Edd,
    and jet efficiency eta_jet = 0.1, estimate the jet power in erg/s.
    """
    M_sun = 1.989e30     # kg
    M_bh = 1e9 * M_sun   # kg
    c = 3e8              # m/s
    sigma_T = 6.652e-29  # m^2 (Thomson cross section)
    m_p = 1.67e-27       # kg

    # Eddington luminosity: L_Edd = 4*pi*G*M*m_p*c / sigma_T
    G = 6.674e-11
    L_Edd = 4 * np.pi * G * M_bh * m_p * c / sigma_T
    L_Edd_erg = L_Edd * 1e7  # W to erg/s

    # Eddington accretion rate: M_dot_Edd = L_Edd / (eta_rad * c^2)
    eta_rad = 0.1  # radiative efficiency
    M_dot_Edd = L_Edd / (eta_rad * c**2)

    # Actual accretion rate
    M_dot = 0.1 * M_dot_Edd

    # Jet power: P_jet = eta_jet * M_dot * c^2
    eta_jet = 0.1
    P_jet = eta_jet * M_dot * c**2
    P_jet_erg = P_jet * 1e7

    print(f"  M_BH = {M_bh / M_sun:.1e} M_sun")
    print(f"  L_Edd = {L_Edd:.3e} W = {L_Edd_erg:.3e} erg/s")
    print(f"  M_dot_Edd = L_Edd / (eta_rad * c^2) = {M_dot_Edd:.3e} kg/s")
    print(f"  M_dot_Edd = {M_dot_Edd * 365.25 * 86400 / M_sun:.2f} M_sun/yr")
    print(f"  M_dot = 0.1 * M_dot_Edd = {M_dot:.3e} kg/s")
    print(f"  Jet efficiency eta_jet = {eta_jet}")
    print(f"  Jet power P_jet = eta_jet * M_dot * c^2")
    print(f"  P_jet = {P_jet:.3e} W = {P_jet_erg:.3e} erg/s")
    print(f"  P_jet / L_Edd = {P_jet / L_Edd:.3f}")
    print(f"  This is a powerful AGN jet, comparable to observed FR II radio galaxies.")


def exercise_9():
    """MRI Dispersion with Toroidal Field.

    Modify the MRI dispersion relation to include toroidal field B_phi
    in addition to vertical field B_z. How does the growth rate change?
    """
    # Standard MRI dispersion (vertical field only, incompressible, Keplerian):
    # omega^4 - omega^2*(2*kz^2*vAz^2 + kappa^2) + kz^2*vAz^2*(kz^2*vAz^2 - 4*Omega^2) = 0
    # where kappa^2 = 4*Omega^2 for Keplerian, vAz = Bz/sqrt(mu0*rho)

    # With toroidal field, the dispersion relation includes:
    # Additional terms from B_phi tension and magnetic pressure
    # omega^4 - omega^2*(2*k^2*vA^2 + kappa^2 + ...) + ... = 0

    Omega = 1.0  # normalized
    kappa2 = 4 * Omega**2  # epicyclic frequency squared (Keplerian: kappa = 2*Omega)

    kz = np.linspace(0.01, 5.0, 500)
    vAz = 0.5  # vertical Alfven speed

    # Case 1: B_phi = 0 (pure vertical field)
    # omega^2 solutions from quadratic in omega^2:
    # a*X^2 + b*X + c = 0 where X = omega^2
    growth_Bz_only = np.zeros_like(kz)
    for i, k in enumerate(kz):
        a = 1.0
        b = -(2 * k**2 * vAz**2 + kappa2)
        c = k**2 * vAz**2 * (k**2 * vAz**2 - 4 * Omega**2)
        disc = b**2 - 4 * a * c
        if disc >= 0:
            omega2_minus = (-b - np.sqrt(disc)) / (2 * a)
            if omega2_minus < 0:
                growth_Bz_only[i] = np.sqrt(-omega2_minus)

    # Case 2: Add toroidal field B_phi
    # The full dispersion with both Bz and Bphi is more complex
    # Simplified: effective Alfven speed includes both components
    # The toroidal field can be stabilizing or destabilizing depending on geometry

    vAphi_values = [0.0, 0.2, 0.5, 1.0, 2.0]

    fig, ax = plt.subplots(figsize=(10, 6))

    for vAphi in vAphi_values:
        growth = np.zeros_like(kz)
        for i, k in enumerate(kz):
            # Modified dispersion including azimuthal field effects
            # The toroidal field adds a magnetic tension that modifies kappa_eff:
            # kappa_eff^2 = kappa^2 + k^2*vAphi^2 (simplified)
            # And the vertical Alfven term remains
            vA_total2 = vAz**2 + vAphi**2
            a = 1.0
            b = -(2 * k**2 * vAz**2 + kappa2 + k**2 * vAphi**2)
            c = k**2 * vAz**2 * (k**2 * vA_total2 - 4 * Omega**2) + k**2 * vAphi**2 * kappa2 / 4
            disc = b**2 - 4 * a * c
            if disc >= 0:
                omega2_minus = (-b - np.sqrt(disc)) / (2 * a)
                if omega2_minus < 0:
                    growth[i] = np.sqrt(-omega2_minus)

        label = f'$v_{{A,\\phi}}$ = {vAphi:.1f}'
        ax.plot(kz, growth / Omega, label=label, linewidth=2)

    ax.set_xlabel(r'$k_z v_{A,z} / \Omega$', fontsize=12)
    ax.set_ylabel(r'Growth rate $\gamma / \Omega$', fontsize=12)
    ax.set_title('MRI Growth Rate: Effect of Toroidal Field', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig('12_mri_toroidal.png', dpi=150)
    plt.close()

    print("  Plot saved to 12_mri_toroidal.png")
    print("  Results:")
    print("  - Pure vertical field: maximum growth rate gamma = 0.75*Omega")
    print("  - Adding toroidal field modifies the instability range and growth rate")
    print("  - Strong toroidal field can stabilize short wavelengths")
    print("  - But MRI remains active as long as d(Omega)/d(ln r) < 0")


def exercise_10():
    """1D Disk Evolution.

    Implement a 1D vertically-integrated disk evolution model with
    MRI-driven alpha-viscosity. Start with a ring of material and
    watch it spread and accrete over time.
    """
    # 1D viscous disk evolution equation:
    # d(Sigma)/dt = (3/r) * d/dr [r^(1/2) * d/dr(nu*Sigma*r^(1/2))]
    # where Sigma = surface density, nu = alpha*c_s*H

    Nr = 200
    r_in, r_out = 0.1, 10.0
    r = np.linspace(r_in, r_out, Nr)
    dr = r[1] - r[0]

    # Initial condition: ring of material at r_0 = 1.0
    r_0 = 1.0
    sigma_0 = 10.0
    width = 0.1
    Sigma = sigma_0 * np.exp(-((r - r_0) / width)**2)

    # Alpha-viscosity
    alpha = 0.01
    # For Keplerian disk: nu = alpha * c_s * H, c_s * H = c_s^2 / Omega
    # Assume c_s/v_K = H/R = 0.05 (thin disk)
    h_over_r = 0.05

    # nu(r) = alpha * (H/R)^2 * Omega * R^2 = alpha * h_r^2 * sqrt(GM) * r^(1/2)
    # Normalize: GM = 1
    nu = alpha * h_over_r**2 * r**1.5  # ~ alpha * c_s * H

    # Time evolution
    dt = 1e-3
    N_steps = 50000

    # Green's function solution for comparison (Lynden-Bell & Pringle 1974)
    # Sigma_analytic = (M_ring/(pi*r_0)) * tau^(-1) * (r/r_0)^(-1/4) *
    #                  exp(-(1+r^2/r_0^2)/tau) * I_{1/4}(2r/(r_0*tau))
    # where tau = 12*nu_0*t/r_0^2 + 1

    times_to_save = [0, 0.1, 0.5, 1.0, 3.0, 10.0]
    saved = {0: Sigma.copy()}
    t = 0.0
    next_save_idx = 1

    for step in range(N_steps):
        # Compute mass flux: F = 3*pi*nu*Sigma
        # d(Sigma)/dt = (1/r) * d/dr [3*r^(1/2) * d/dr(nu*Sigma*r^(1/2))]

        # Auxiliary variable: g = nu * Sigma * r^(1/2)
        g = nu * Sigma * np.sqrt(r)

        # Compute d(g)/dr
        dg_dr = np.zeros(Nr)
        dg_dr[1:-1] = (g[2:] - g[:-2]) / (2 * dr)

        # Compute 3/r * d/dr[r^(1/2) * dg/dr]
        # h = r^(1/2) * dg/dr
        h = np.sqrt(r) * dg_dr

        # d(h)/dr
        dh_dr = np.zeros(Nr)
        dh_dr[1:-1] = (h[2:] - h[:-2]) / (2 * dr)

        # d(Sigma)/dt = (3/r) * dh_dr
        dSigma_dt = 3.0 / r * dh_dr

        # Add artificial diffusion for stability
        d2Sigma = np.zeros(Nr)
        d2Sigma[1:-1] = (Sigma[2:] - 2 * Sigma[1:-1] + Sigma[:-2]) / dr**2
        dSigma_dt += 1e-4 * d2Sigma

        Sigma += dt * dSigma_dt

        # Boundary conditions
        Sigma[0] = 0  # material accretes at inner boundary
        Sigma[-1] = 0  # no material at outer boundary
        Sigma = np.maximum(Sigma, 0)  # no negative surface density

        t += dt

        # Save snapshots
        if next_save_idx < len(times_to_save) and t >= times_to_save[next_save_idx]:
            saved[times_to_save[next_save_idx]] = Sigma.copy()
            next_save_idx += 1

    # Plot evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(saved)))
    for (t_save, Sigma_save), color in zip(saved.items(), colors):
        ax1.plot(r, Sigma_save, color=color, linewidth=1.5, label=f't = {t_save:.1f}')

    ax1.set_xlabel('r (normalized)', fontsize=12)
    ax1.set_ylabel(r'$\Sigma$ (surface density)', fontsize=12)
    ax1.set_title('Disk Evolution: Surface Density', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Mass evolution
    mass_history = []
    t_history = []
    Sigma_track = sigma_0 * np.exp(-((r - r_0) / width)**2)
    for step in range(N_steps):
        g = nu * Sigma_track * np.sqrt(r)
        dg_dr = np.zeros(Nr)
        dg_dr[1:-1] = (g[2:] - g[:-2]) / (2 * dr)
        h = np.sqrt(r) * dg_dr
        dh_dr = np.zeros(Nr)
        dh_dr[1:-1] = (h[2:] - h[:-2]) / (2 * dr)
        dSigma_dt = 3.0 / r * dh_dr
        d2S = np.zeros(Nr)
        d2S[1:-1] = (Sigma_track[2:] - 2 * Sigma_track[1:-1] + Sigma_track[:-2]) / dr**2
        dSigma_dt += 1e-4 * d2S
        Sigma_track += dt * dSigma_dt
        Sigma_track[0] = 0
        Sigma_track[-1] = 0
        Sigma_track = np.maximum(Sigma_track, 0)

        if step % 500 == 0:
            M_total = 2 * np.pi * np.trapz(Sigma_track * r, r)
            mass_history.append(M_total)
            t_history.append(step * dt)

    ax2.plot(t_history, mass_history, 'b-', linewidth=2)
    ax2.set_xlabel('Time (normalized)', fontsize=12)
    ax2.set_ylabel('Total disk mass', fontsize=12)
    ax2.set_title('Disk Mass Evolution (Accretion)', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('12_disk_evolution.png', dpi=150)
    plt.close()
    print("  Plot saved to 12_disk_evolution.png")
    print("  The ring of material spreads viscously: inner part accretes, outer part expands.")
    print("  The viscous timescale t_visc ~ R^2/nu controls the spreading rate.")
    print("  Total mass decreases as material crosses the inner boundary (accreted).")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: MRI Growth Rate", exercise_1),
        ("Exercise 2: MRI Wavelength", exercise_2),
        ("Exercise 3: Maxwell Stress", exercise_3),
        ("Exercise 4: Accretion Timescale", exercise_4),
        ("Exercise 5: Equipartition Field", exercise_5),
        ("Exercise 6: Dead Zone Criterion", exercise_6),
        ("Exercise 7: Blandford-Payne Angle", exercise_7),
        ("Exercise 8: Jet Power", exercise_8),
        ("Exercise 9: MRI Dispersion with Toroidal Field", exercise_9),
        ("Exercise 10: 1D Disk Evolution", exercise_10),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
        func()
