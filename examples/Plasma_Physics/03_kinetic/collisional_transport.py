#!/usr/bin/env python3
"""
Collisional Kinetics and Transport

Demonstrates key results from collisional kinetic theory:
  1. Fokker-Planck slowing-down of a fast (test) particle in a background plasma
  2. Classical vs neoclassical transport comparison across collisionality regimes
  3. Spitzer-Harm heat flux calculation
  4. Bootstrap current estimation for a tokamak

All expressions follow the standard Braginskii / Helander-Sigmar conventions.

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, m_p, epsilon_0, k as k_B
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

m_i = m_p           # proton mass (deuterium would be 2*m_p; use m_p for simplicity)


# ---------------------------------------------------------------------------
# Helper: Coulomb logarithm and collision frequencies
# ---------------------------------------------------------------------------

def coulomb_log(n_e, T_eV):
    """
    Coulomb logarithm (Spitzer approximation).

    Parameters
    ----------
    n_e : float
        Electron density [m^-3]
    T_eV : float
        Electron temperature [eV]

    Returns
    -------
    ln_Lambda : float
    """
    if T_eV > 10:
        ln_L = 24.0 - np.log(np.sqrt(n_e * 1e-6) / T_eV)
    else:
        ln_L = 23.0 - np.log(np.sqrt(n_e * 1e-6) / T_eV ** 1.5)
    return max(ln_L, 2.0)


def nu_ei_spitzer(n_e, T_eV):
    """
    Spitzer electron–ion collision frequency [Hz].

    ν_ei = n_e e⁴ lnΛ / (12 π^{3/2} ε₀² m_e^{1/2} T_e^{3/2})

    Parameters
    ----------
    n_e : float
        Electron density [m^-3]
    T_eV : float
        Electron temperature [eV]

    Returns
    -------
    nu : float  [Hz]
    """
    T_J = T_eV * e
    ln_L = coulomb_log(n_e, T_eV)
    return (n_e * e ** 4 * ln_L) / (
        12.0 * np.pi ** 1.5 * epsilon_0 ** 2 * m_e ** 0.5 * T_J ** 1.5
    )


# ---------------------------------------------------------------------------
# 1. Fokker-Planck slowing-down
# ---------------------------------------------------------------------------

def fokker_planck_slowing_down(v0, n_e, T_eV, m_test=m_e, Z=1,
                               t_max_factor=5.0, n_points=500):
    """
    Simulate the Fokker-Planck slowing-down (drag) of a fast test particle.

    The speed equation in the Lorentz limit (test faster than field particles):
        dv/dt = -ν_s v
    where ν_s is the slowing-down frequency:
        ν_s = (1 + m_test/m_field) * G(v/v_th) * ν_0 / v³  (approximate)

    For v >> v_th_e the slowing-down is dominated by:
        dv/dt ≈ -ν_ee * v_th_e³ / v²   (for test electron on background electrons)

    Here we use the standard pitch-angle + slowing-down:
        dv/dt = -A_drag / v²
    with A_drag = ν₀ * v_th_e³

    Parameters
    ----------
    v0 : float
        Initial speed [m/s], should be >> v_th
    n_e : float
        Background electron density [m^-3]
    T_eV : float
        Background electron temperature [eV]
    m_test : float
        Test particle mass [kg]
    Z : int
        Test particle charge number
    t_max_factor : float
        Run until t_max_factor * tau_s
    n_points : int
        Number of time points

    Returns
    -------
    t : ndarray  [s]
    v : ndarray  [m/s]
    tau_s : float  slowing-down time [s]
    """
    T_J = T_eV * e
    v_th_e = np.sqrt(2.0 * T_J / m_e)
    ln_L = coulomb_log(n_e, T_eV)

    # Slowing-down coefficient  A = dv/dt * v² = -const
    # For a fast electron on background electrons:
    #   A = (e⁴ n_e lnΛ) / (4π ε₀² m_e²)  ×  (m_e / m_test)
    A_drag = (e ** 4 * n_e * ln_L * Z ** 2) / (
        4.0 * np.pi * epsilon_0 ** 2 * m_e * m_test
    )

    # Slowing-down time  τ_s = v₀³ / (3 A_drag)
    tau_s = v0 ** 3 / (3.0 * A_drag)
    t_end = t_max_factor * tau_s

    def ode(t, y):
        v = y[0]
        if v <= v_th_e:         # stop drag when thermalised
            return [0.0]
        return [-A_drag / v ** 2]

    sol = solve_ivp(ode, [0, t_end], [v0],
                    t_eval=np.linspace(0, t_end, n_points),
                    method='RK45', rtol=1e-8, atol=1e-10,
                    events=lambda t, y: y[0] - v_th_e)

    return sol.t, sol.y[0], tau_s


# ---------------------------------------------------------------------------
# 2. Classical vs neoclassical transport
# ---------------------------------------------------------------------------

def classical_diffusivity(n_e, T_eV, B):
    """
    Classical cross-field electron diffusion coefficient.

    D_cl = ν_ei * r_Le²

    Parameters
    ----------
    n_e : float  [m^-3]
    T_eV : float  [eV]
    B : float  [T]

    Returns
    -------
    D_cl : float  [m²/s]
    """
    T_J = T_eV * e
    nu = nu_ei_spitzer(n_e, T_eV)
    omega_ce = e * B / m_e
    r_Le = np.sqrt(2.0 * T_J / m_e) / omega_ce
    return nu * r_Le ** 2


def bohm_diffusivity(T_eV, B):
    """
    Bohm diffusion coefficient (empirical upper bound for anomalous transport).

    D_Bohm = (1/16) * T_e / (e B)

    Parameters
    ----------
    T_eV : float  [eV]
    B : float  [T]

    Returns
    -------
    D_Bohm : float  [m²/s]
    """
    return T_eV / (16.0 * B)


def neoclassical_diffusivity(n_e, T_eV, B, R, q_safety=2.0):
    """
    Neoclassical electron diffusion in a tokamak (plateau / banana regimes).

    Regime selection via collisionality ν*:
        ν* = ν_ei * q R / (ε^{3/2} v_th)   with ε = r/R ~ 0.3 (mid-radius)

    Banana regime (ν* < ε^{3/2}):
        D_neo = q² / ε^{3/2} * D_cl   (banana orbit enhancement)

    Plateau regime (ε^{3/2} < ν* < 1):
        D_neo = q * v_th / (R * Ω_ce)   (plateau diffusion)

    Pfirsch-Schlüter regime (ν* > 1):
        D_neo = 2 q² * D_cl

    Parameters
    ----------
    n_e : float  [m^-3]
    T_eV : float  [eV]
    B : float  [T]
    R : float  major radius [m]
    q_safety : float  safety factor

    Returns
    -------
    D_neo : float  [m²/s]
    regime : str
    """
    eps = 0.3   # inverse aspect ratio r/R at mid-radius

    T_J = T_eV * e
    v_th = np.sqrt(2.0 * T_J / m_e)
    nu = nu_ei_spitzer(n_e, T_eV)

    # Dimensionless collisionality
    nu_star = nu * q_safety * R / (eps ** 1.5 * v_th)

    D_cl = classical_diffusivity(n_e, T_eV, B)

    if nu_star < eps ** 1.5:
        # Banana regime
        D_neo = (q_safety ** 2 / eps ** 1.5) * D_cl
        regime = 'Banana'
    elif nu_star < 1.0:
        # Plateau regime
        omega_ce = e * B / m_e
        D_neo = q_safety * v_th / (R * omega_ce)
        regime = 'Plateau'
    else:
        # Pfirsch-Schlüter regime
        D_neo = 2.0 * q_safety ** 2 * D_cl
        regime = 'Pfirsch-Schlüter'

    return D_neo, regime


# ---------------------------------------------------------------------------
# 3. Spitzer-Harm heat flux
# ---------------------------------------------------------------------------

def spitzer_harm_conductivity(n_e, T_eV):
    """
    Spitzer-Harm electron thermal conductivity [W m^-1 K^-1].

    κ_e = 3.16 * n_e k_B T_e / (m_e ν_ei)

    Parameters
    ----------
    n_e : float  [m^-3]
    T_eV : float  [eV]

    Returns
    -------
    kappa : float  [W m^-1 K^-1]
    """
    T_J = T_eV * e
    nu = nu_ei_spitzer(n_e, T_eV)
    return 3.16 * n_e * k_B * T_J / (m_e * nu)


def spitzer_harm_heat_flux(n_e, T_eV, grad_T_eV_per_m):
    """
    Spitzer-Harm heat flux  q_SH = -κ_e ∇T.

    Parameters
    ----------
    n_e : float  [m^-3]
    T_eV : float  [eV]
    grad_T_eV_per_m : float
        Temperature gradient [eV/m]

    Returns
    -------
    q_SH : float  [W/m²]
    """
    kappa = spitzer_harm_conductivity(n_e, T_eV)
    grad_T_J = grad_T_eV_per_m * e    # convert to J/m = K/m × k_B factor handled in kappa
    return -kappa * grad_T_J / k_B    # kappa is in W/(m K), grad in J/m → need /k_B


# ---------------------------------------------------------------------------
# 4. Bootstrap current
# ---------------------------------------------------------------------------

def bootstrap_current_density(n_e, T_eV, B, R, grad_n_per_m, grad_T_eV_per_m,
                               q_safety=2.0, Z_eff=1.5):
    """
    Estimate the bootstrap current density (Wesson formula).

    j_bs ≈ -R * B_p / B * (2.44 * ε^{1/2}) * (p' / B_p²) * (1 + T_i/T_e)

    Simplified form (electron contribution only, ε = 0.3):

        j_bs = -f_t * p' / (ω_ce * τ_ei)

    where f_t ≈ 1.46 ε^{1/2} is the trapped particle fraction.

    Parameters
    ----------
    n_e : float  [m^-3]
    T_eV : float  [eV]
    B : float  [T]
    R : float  major radius [m]
    grad_n_per_m : float  density gradient [m^-4]
    grad_T_eV_per_m : float  temperature gradient [eV/m]
    q_safety : float
    Z_eff : float  effective charge

    Returns
    -------
    j_bs : float  [A/m²]  (negative = counter-current in standard convention)
    f_trapped : float  trapped-particle fraction
    """
    eps = 0.3

    T_J = T_eV * e
    nu = nu_ei_spitzer(n_e, T_eV)
    omega_ce = e * B / m_e

    # Pressure gradient (electron contribution)
    grad_p = k_B * (T_J * grad_n_per_m + n_e * grad_T_eV_per_m * e)

    # Trapped particle fraction
    f_trapped = 1.46 * np.sqrt(eps)

    # Bootstrap current density (approximate, e-contribution)
    j_bs = -f_trapped * grad_p / (m_e * omega_ce * nu)

    return j_bs, f_trapped


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_slowing_down():
    """Fokker-Planck slowing-down of a suprathermal electron."""

    n_e = 1e20     # m^-3
    T_eV = 5e3     # 5 keV (tokamak-like)
    T_J = T_eV * e
    v_th = np.sqrt(2.0 * T_J / m_e)
    v0 = 10.0 * v_th     # fast test electron at 10 v_th

    print(f"  Background: n_e = {n_e:.1e} m^-3,  T_e = {T_eV:.0f} eV")
    print(f"  Thermal velocity: v_th = {v_th:.2e} m/s")
    print(f"  Initial speed:    v₀   = {v0:.2e} m/s  ({v0/v_th:.1f} v_th)")

    t, v, tau_s = fokker_planck_slowing_down(v0, n_e, T_eV)
    analytic_v = (v0 ** 3 - 3.0 * (v0 ** 3 / (3.0 * tau_s)) * t) ** (1.0 / 3.0)
    analytic_v = np.where(analytic_v ** 3 > v_th ** 3, analytic_v, v_th)

    print(f"  Slowing-down time: τ_s = {tau_s:.2e} s")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(t / tau_s, v / v_th, 'b-', linewidth=2, label='Fokker-Planck ODE')
    ax.plot(t / tau_s, analytic_v / v_th, 'r--', linewidth=1.5,
            label='Analytic: v(t) = (v₀³ − 3A t)^{1/3}')
    ax.axhline(1.0, color='grey', linestyle=':', linewidth=1.2,
               label='Thermal speed v_th')
    ax.set_xlabel('t / τ_s', fontsize=12)
    ax.set_ylabel('v / v_th', fontsize=12)
    ax.set_title('Fokker-Planck Slowing-Down\n(fast electron in Maxwellian background)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Energy decay
    ax2 = axes[1]
    E_kin = 0.5 * m_e * v ** 2
    E0 = 0.5 * m_e * v0 ** 2
    ax2.plot(t / tau_s, E_kin / E0, 'b-', linewidth=2, label='Kinetic energy')
    ax2.axhline(T_eV / (v0 / v_th) ** 2 / T_eV, color='grey', linestyle=':',
                linewidth=1.2)
    ax2.set_xlabel('t / τ_s', fontsize=12)
    ax2.set_ylabel('E_k / E₀', fontsize=12)
    ax2.set_title('Kinetic Energy Thermalisation', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('collisional_transport_slowing_down.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_transport_comparison():
    """
    Compare classical, neoclassical, and Bohm diffusivities
    as a function of electron temperature for tokamak parameters.
    """
    T_range = np.logspace(1, 4, 80)   # 10 eV to 10 keV
    n_e = 1e19                         # m^-3
    B = 3.0                            # T
    R = 3.0                            # m

    D_cl  = np.array([classical_diffusivity(n_e, T, B) for T in T_range])
    D_bohm = np.array([bohm_diffusivity(T, B) for T in T_range])
    D_neo  = np.array([neoclassical_diffusivity(n_e, T, B, R)[0] for T in T_range])
    regimes = [neoclassical_diffusivity(n_e, T, B, R)[1] for T in T_range]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(T_range, D_cl,   'b-',  linewidth=2.0, label='Classical D_cl = ν_ei r_Le²')
    ax.loglog(T_range, D_neo,  'g-',  linewidth=2.0, label='Neoclassical D_neo')
    ax.loglog(T_range, D_bohm, 'r--', linewidth=2.0, label='Bohm D_Bohm = T/(16eB)  [anomalous]')

    # Shade regime boundaries
    banana_mask   = np.array(regimes) == 'Banana'
    plateau_mask  = np.array(regimes) == 'Plateau'
    ps_mask       = np.array(regimes) == 'Pfirsch-Schlüter'

    for mask, color, label in [
        (banana_mask,  '#d0ffd8', 'Banana regime'),
        (plateau_mask, '#fff3cd', 'Plateau regime'),
        (ps_mask,      '#ffd8d8', 'Pfirsch-Schlüter'),
    ]:
        if np.any(mask):
            T_lo = T_range[mask].min()
            T_hi = T_range[mask].max()
            ax.axvspan(T_lo, T_hi, alpha=0.25, color=color, label=label)

    ax.set_xlabel('Electron Temperature [eV]', fontsize=12)
    ax.set_ylabel('Diffusion Coefficient D [m²/s]', fontsize=12)
    ax.set_title(f'Transport Coefficients vs Temperature\n'
                 f'(n_e = {n_e:.0e} m⁻³, B = {B} T, R = {R} m)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('collisional_transport_diffusivity.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_spitzer_harm():
    """
    Spitzer-Harm thermal conductivity and heat flux profile along a flux tube.
    """
    n_e = 1e20    # m^-3
    L = 50.0      # connection length [m]
    x = np.linspace(0, L, 200)

    # Temperature profile: hot core, cooler edge
    T_core = 5e3   # eV
    T_edge = 200   # eV
    T_profile = T_edge + (T_core - T_edge) * np.cos(np.pi * x / (2 * L)) ** 2

    dT_dx = np.gradient(T_profile, x)    # eV/m

    kappa = np.array([spitzer_harm_conductivity(n_e, T) for T in T_profile])
    q_SH  = np.array([spitzer_harm_heat_flux(n_e, T, dT)
                      for T, dT in zip(T_profile, dT_dx)])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].plot(x, T_profile / 1e3, 'r-', linewidth=2)
    axes[0].set_xlabel('Position along flux tube [m]', fontsize=11)
    axes[0].set_ylabel('T_e [keV]', fontsize=11)
    axes[0].set_title('Temperature Profile', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(x, kappa, 'b-', linewidth=2)
    axes[1].set_xlabel('Position along flux tube [m]', fontsize=11)
    axes[1].set_ylabel('κ_e [W m⁻¹ K⁻¹]', fontsize=11)
    axes[1].set_title('Spitzer-Harm Conductivity\nκ_e ∝ T_e^{5/2}', fontsize=11,
                      fontweight='bold')
    axes[1].grid(True, alpha=0.3, which='both')

    axes[2].plot(x, q_SH / 1e6, 'g-', linewidth=2, label='q_SH')
    axes[2].axhline(0, color='grey', linestyle=':', linewidth=1)
    axes[2].set_xlabel('Position along flux tube [m]', fontsize=11)
    axes[2].set_ylabel('Heat flux [MW/m²]', fontsize=11)
    axes[2].set_title('Spitzer-Harm Heat Flux\nq_SH = −κ_e ∇T', fontsize=11,
                      fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('collisional_transport_heat_flux.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_bootstrap_current():
    """
    Bootstrap current density as a function of temperature gradient
    and comparison with ohmic current for a tokamak scenario.
    """
    n_e  = 1e20       # m^-3
    T_eV = 5e3        # eV
    B    = 5.0        # T
    R    = 6.0        # m (ITER-like)
    L_n  = 0.5        # density gradient scale length [m]: dn/dx = -n/L_n
    q_safety = 1.8

    grad_n = -n_e / L_n    # m^-4

    # Vary temperature gradient scale length L_T
    L_T_range = np.linspace(0.1, 5.0, 100)    # m
    j_bs_values = []

    for L_T in L_T_range:
        grad_T = -T_eV / L_T    # eV/m
        j_bs, _ = bootstrap_current_density(
            n_e, T_eV, B, R, grad_n, grad_T, q_safety=q_safety
        )
        j_bs_values.append(j_bs)

    j_bs_arr = np.array(j_bs_values)

    # Ohmic current estimate: j_ohm = σ E_parallel ≈ j_bs for comparison
    # Use Spitzer resistivity: η = m_e ν_ei / (n_e e²)
    eta_spitzer = m_e * nu_ei_spitzer(n_e, T_eV) / (n_e * e ** 2)
    E_loop = 0.1   # V/m (typical loop voltage / 2πR)
    j_ohm  = E_loop / eta_spitzer

    print(f"\n  Bootstrap parameters (n_e={n_e:.0e} m^-3, T_e={T_eV:.0f} eV, B={B} T):")
    print(f"    Spitzer resistivity: η = {eta_spitzer:.2e} Ω·m")
    print(f"    Ohmic current density (E={E_loop} V/m): j_ohm = {j_ohm:.2e} A/m²")
    _, f_t = bootstrap_current_density(n_e, T_eV, B, R, grad_n, -T_eV / 1.0,
                                       q_safety=q_safety)
    print(f"    Trapped particle fraction: f_t = {f_t:.3f}")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(L_T_range, np.abs(j_bs_arr) / 1e6, 'b-', linewidth=2,
            label='|j_bootstrap|')
    ax.axhline(j_ohm / 1e6, color='red', linestyle='--', linewidth=2,
               label=f'j_ohm (E = {E_loop} V/m)')

    # Fraction of bootstrap vs ohmic
    ax2 = ax.twinx()
    bs_fraction = np.abs(j_bs_arr) / j_ohm * 100
    ax2.plot(L_T_range, bs_fraction, 'g:', linewidth=1.8,
             label='Bootstrap fraction [%]')
    ax2.set_ylabel('Bootstrap / Ohmic [%]', fontsize=11, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, max(bs_fraction) * 1.2)

    ax.set_xlabel('Temperature gradient scale length L_T [m]', fontsize=12)
    ax.set_ylabel('Current density [MA/m²]', fontsize=12)
    ax.set_title('Bootstrap Current vs Temperature Gradient\n'
                 f'(n_e = {n_e:.0e} m⁻³, T_e = {T_eV:.0f} eV, B = {B} T, q = {q_safety})',
                 fontsize=11, fontweight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('collisional_transport_bootstrap.png', dpi=300, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("\n" + "=" * 75)
    print("COLLISIONAL KINETICS AND TRANSPORT")
    print("=" * 75)

    print("\nPart 1: Fokker-Planck slowing-down of a fast electron")
    print("-" * 75)
    plot_slowing_down()

    print("\nPart 2: Classical vs neoclassical vs Bohm transport")
    print("-" * 75)
    plot_transport_comparison()

    print("\nPart 3: Spitzer-Harm thermal conductivity and heat flux")
    print("-" * 75)
    plot_spitzer_harm()

    print("\nPart 4: Bootstrap current estimation")
    print("-" * 75)
    plot_bootstrap_current()

    print("\nDone! Generated 4 figures:")
    print("  - collisional_transport_slowing_down.png")
    print("  - collisional_transport_diffusivity.png")
    print("  - collisional_transport_heat_flux.png")
    print("  - collisional_transport_bootstrap.png")
