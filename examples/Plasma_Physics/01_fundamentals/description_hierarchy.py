#!/usr/bin/env python3
"""
Plasma Description Hierarchy

Illustrates the Klimontovich → Vlasov → fluid hierarchy of plasma descriptions.
Demonstrates moment closure by computing density, bulk velocity, and pressure
from a distribution function, and compares model validity across collisionality
regimes for representative plasmas.

Author: Plasma Physics Examples
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, m_p, epsilon_0, k as k_B


# ---------------------------------------------------------------------------
# Distribution function utilities
# ---------------------------------------------------------------------------

def maxwellian_1d(v, n, T_eV, m=m_e):
    """
    1-D Maxwellian distribution function.

    Parameters
    ----------
    v : ndarray
        Velocity grid [m/s]
    n : float
        Number density [m^-3]
    T_eV : float
        Temperature [eV]
    m : float
        Particle mass [kg]

    Returns
    -------
    f : ndarray
        Distribution function [m^-3 (m/s)^-1]
    """
    T_J = T_eV * e
    v_th = np.sqrt(2.0 * T_J / m)
    return (n / (np.sqrt(np.pi) * v_th)) * np.exp(-(v / v_th) ** 2)


def shifted_maxwellian_1d(v, n, T_eV, u, m=m_e):
    """
    Drifting Maxwellian (shifted by bulk velocity u).

    Parameters
    ----------
    v : ndarray
        Velocity grid [m/s]
    n : float
        Number density [m^-3]
    T_eV : float
        Temperature [eV]
    u : float
        Bulk drift velocity [m/s]
    m : float
        Particle mass [kg]

    Returns
    -------
    f : ndarray
        Distribution function [m^-3 (m/s)^-1]
    """
    T_J = T_eV * e
    v_th = np.sqrt(2.0 * T_J / m)
    return (n / (np.sqrt(np.pi) * v_th)) * np.exp(-((v - u) / v_th) ** 2)


# ---------------------------------------------------------------------------
# Moment extraction (Vlasov → fluid closure)
# ---------------------------------------------------------------------------

def compute_moments(f, v, m=m_e):
    """
    Compute the first three fluid moments from a 1-D distribution function.

    Moments are:
        Zeroth: n       = ∫ f dv              (number density)
        First : u       = (1/n) ∫ v f dv       (bulk velocity)
        Second: p       = m ∫ (v-u)² f dv      (scalar pressure)
        Scalar: T       = p / (n k_B)           (temperature)

    Parameters
    ----------
    f : ndarray
        Distribution function values on velocity grid
    v : ndarray
        Velocity grid [m/s]
    m : float
        Particle mass [kg]

    Returns
    -------
    dict with keys: n, u, p, T_eV
    """
    dv = v[1] - v[0]

    n = np.trapz(f, v)
    if n <= 0:
        return {'n': 0.0, 'u': 0.0, 'p': 0.0, 'T_eV': 0.0}

    u = np.trapz(v * f, v) / n
    p = m * np.trapz((v - u) ** 2 * f, v)
    T_eV = p / (n * k_B) / (e / k_B)   # convert Joules → eV via p = n k_B T

    return {'n': n, 'u': u, 'p': p, 'T_eV': T_eV}


# ---------------------------------------------------------------------------
# Collisionality regime diagnostics
# ---------------------------------------------------------------------------

def collisionality_parameters(n_e, T_eV, L, B=0.0, Z=1):
    """
    Compute dimensionless parameters that indicate which description is appropriate.

    Parameters
    ----------
    n_e : float
        Electron density [m^-3]
    T_eV : float
        Electron temperature [eV]
    L : float
        Characteristic length scale [m]
    B : float
        Magnetic field [T] (0 = unmagnetised)
    Z : int
        Ion charge number

    Returns
    -------
    dict with diagnostic parameters and recommended model
    """
    T_J = T_eV * e
    v_th = np.sqrt(2.0 * T_J / m_e)

    # Debye length
    lambda_D = np.sqrt(epsilon_0 * T_J / (n_e * e ** 2))

    # Plasma parameter (number of particles in Debye sphere)
    N_D = n_e * (4.0 / 3.0) * np.pi * lambda_D ** 3

    # Coulomb logarithm
    if T_eV > 10:
        ln_Lambda = 24.0 - np.log(np.sqrt(n_e * 1e-6) / T_eV)
    else:
        ln_Lambda = 23.0 - np.log(np.sqrt(n_e * 1e-6) * Z / T_eV ** 1.5)
    ln_Lambda = max(ln_Lambda, 2.0)

    # Electron–ion collision frequency [Hz]
    nu_ei = (n_e * e ** 4 * ln_Lambda) / (
        12.0 * np.pi ** 1.5 * epsilon_0 ** 2 * m_e ** 0.5 * T_J ** 1.5
    )

    # Mean free path
    mfp = v_th / nu_ei

    # Collisionality: ratio of collision frequency to transit frequency
    nu_star = (nu_ei * L) / v_th   # dimensionless collisionality ν*

    # Magnetisation (if B > 0)
    if B > 0:
        omega_ce = e * B / m_e
        r_Le = v_th / omega_ce
        rho_star = r_Le / L     # normalised Larmor radius
    else:
        omega_ce = 0.0
        r_Le = np.inf
        rho_star = np.inf

    # Model selection heuristic
    if N_D < 10:
        model = 'N/A — not a plasma (N_D < 10)'
    elif nu_star > 1e3:
        model = 'Fluid (MHD) — highly collisional'
    elif nu_star > 1.0:
        model = 'Extended MHD / Braginskii fluid'
    elif B > 0 and rho_star < 0.05:
        model = 'Gyrokinetic'
    else:
        model = 'Kinetic (Vlasov–Poisson/Maxwell)'

    return {
        'lambda_D': lambda_D,
        'N_D': N_D,
        'ln_Lambda': ln_Lambda,
        'nu_ei': nu_ei,
        'mfp': mfp,
        'nu_star': nu_star,
        'r_Le': r_Le,
        'rho_star': rho_star,
        'model': model,
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_moment_closure():
    """
    Show how fluid moments are extracted from distribution functions.

    Illustrates: Maxwellian, drifting Maxwellian, and a non-Maxwellian
    (two-beam) distribution, verifying moment recovery.
    """
    v = np.linspace(-8e6, 8e6, 2000)   # velocity grid [m/s]

    n0 = 1e18     # m^-3
    T0 = 100.0    # eV
    u0 = 1.5e6    # m/s drift

    # Three representative distributions
    f_max   = maxwellian_1d(v, n0, T0)
    f_drift = shifted_maxwellian_1d(v, n0, T0, u0)
    # Two symmetric counter-propagating beams (non-Maxwellian)
    v_beam = 2.5e6
    f_beam = 0.5 * maxwellian_1d(v, n0, T0 * 0.3, m_e) * 0   # zero first
    f_beam = (shifted_maxwellian_1d(v, n0 / 2, T0 * 0.3, +v_beam) +
              shifted_maxwellian_1d(v, n0 / 2, T0 * 0.3, -v_beam))

    distributions = [
        ('Maxwellian',        f_max,   'steelblue'),
        ('Drifting Maxwellian', f_drift, 'darkorange'),
        ('Two-beam (non-Maxwellian)', f_beam, 'mediumseagreen'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for ax, (label, f, color) in zip(axes, distributions):
        moments = compute_moments(f, v)
        ax.plot(v / 1e6, f, color=color, linewidth=2.0)
        ax.axvline(moments['u'] / 1e6, color='red', linestyle='--',
                   linewidth=1.5, label=f"u = {moments['u']/1e6:.2f} Mm/s")
        ax.fill_between(v / 1e6, f, alpha=0.15, color=color)
        ax.set_xlabel('v [Mm/s]', fontsize=11)
        ax.set_ylabel('f(v) [m⁻³ (m/s)⁻¹]', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        info = (f"n = {moments['n']:.2e} m⁻³\n"
                f"u = {moments['u']/1e6:.2f} Mm/s\n"
                f"T = {moments['T_eV']:.1f} eV")
        ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=9,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Moment Closure: Fluid Variables from Distribution Functions',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('description_hierarchy_moments.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_hierarchy_diagram():
    """
    Visual summary of the Klimontovich → Vlasov → fluid hierarchy.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # --- boxes ---
    boxes = [
        (5.0, 7.0, 'Klimontovich Equation\n(exact, 6N + time dims)\nf̂_s(x, v, t) = Σᵢ δ³(x−xᵢ) δ³(v−vᵢ)',
         '#d0e8ff'),
        (5.0, 4.8, 'Vlasov–Boltzmann Equation\n(ensemble average, N_D ≫ 1)\n∂f/∂t + v·∇f + (q/m)E·∂f/∂v = C[f]',
         '#d0ffd8'),
        (2.5, 2.4, 'Kinetic / Gyrokinetic\n(ν* ≪ 1  or  ρ*/a ≪ 1)',
         '#fff3cd'),
        (7.5, 2.4, 'Fluid / MHD\n(ν* ≫ 1,  many collisions)\nclosure: p = nkT',
         '#ffd8d8'),
    ]

    for (x, y, text, color) in boxes:
        bbox = dict(boxstyle='round,pad=0.5', facecolor=color,
                    edgecolor='grey', linewidth=1.5)
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
                bbox=bbox, linespacing=1.5)

    # --- arrows ---
    arrow_kwargs = dict(arrowstyle='->', color='#444444',
                        lw=1.8, mutation_scale=18)
    for (x0, y0, x1, y1, label) in [
        (5.0, 6.35, 5.0, 5.55, 'ensemble avg.\nCoulomb correlations → lnΛ'),
        (4.0, 4.20, 2.8, 3.10, 'take moments\nclosure needed'),
        (6.0, 4.20, 7.2, 3.10, 'take moments\nclosure: fluid eqs.'),
    ]:
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=arrow_kwargs)
        mid_x = (x0 + x1) / 2 + 0.15
        mid_y = (y0 + y1) / 2
        ax.text(mid_x, mid_y, label, ha='left', va='center',
                fontsize=8, color='#333333')

    ax.set_title('Plasma Description Hierarchy',
                 fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('description_hierarchy_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_selection():
    """
    Plot model selection map as a function of temperature and density for
    representative plasma regimes, colour-coded by appropriate description.
    """
    plasmas = [
        # (name, n_e [m^-3], T_eV, L [m], B [T])
        ('Tokamak core',    1e20, 10e3, 1.0,    5.0),
        ('Tokamak edge',    1e19, 100,  0.05,   2.0),
        ('Solar wind',      5e6,  10,   1e10,   5e-9),
        ('Solar corona',    1e12, 200,  1e8,    1e-3),
        ('Ionosphere',      1e11, 0.1,  1e5,    5e-5),
        ('Lightning',       1e24, 2,    0.05,   0.0),
        ('Neon sign',       1e16, 2,    0.1,    0.0),
        ('ICF capsule',     1e28, 5e3,  5e-4,   0.0),
    ]

    # Colour map per model class
    model_colors = {
        'Fluid (MHD) — highly collisional':        '#e74c3c',
        'Extended MHD / Braginskii fluid':         '#e67e22',
        'Gyrokinetic':                             '#27ae60',
        'Kinetic (Vlasov–Poisson/Maxwell)':        '#2980b9',
        'N/A — not a plasma (N_D < 10)':           '#7f8c8d',
    }

    print("\nModel selection for representative plasmas:")
    print("=" * 80)
    header = f"{'Plasma':<20} {'n_e [m⁻³]':<12} {'T [eV]':<10} {'ν*':<12} {'Model'}"
    print(header)
    print("-" * 80)

    fig, ax = plt.subplots(figsize=(10, 7))

    for name, n_e, T_eV, L, B in plasmas:
        p = collisionality_parameters(n_e, T_eV, L, B)
        nu_s = p['nu_star']
        model = p['model']
        color = model_colors.get(model, 'black')

        print(f"{name:<20} {n_e:<12.2e} {T_eV:<10.2e} {nu_s:<12.2e} {model}")

        marker = 'o' if B > 0 else 's'
        ax.scatter(np.log10(n_e), np.log10(T_eV), c=color, s=120,
                   marker=marker, edgecolors='black', linewidths=0.8, zorder=5)
        ax.text(np.log10(n_e) + 0.15, np.log10(T_eV), name,
                fontsize=8, va='center', color='#333333')

    print("=" * 80)
    print("  ● = magnetised  ■ = unmagnetised\n")

    # Legend for model colours
    for label, color in model_colors.items():
        ax.scatter([], [], c=color, s=80, label=label, edgecolors='black',
                   linewidths=0.6)

    ax.set_xlabel('log₁₀(n_e  [m⁻³])', fontsize=12)
    ax.set_ylabel('log₁₀(T_e  [eV])', fontsize=12)
    ax.set_title('Appropriate Plasma Description by Regime', fontsize=13,
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('description_hierarchy_model_map.png', dpi=300, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("PLASMA DESCRIPTION HIERARCHY")
    print("=" * 70)

    print("\nPart 1: Moment closure from distribution functions")
    print("-" * 70)
    plot_moment_closure()

    print("\nPart 2: Klimontovich → Vlasov → fluid hierarchy diagram")
    print("-" * 70)
    plot_hierarchy_diagram()

    print("\nPart 3: Model selection across plasma regimes")
    print("-" * 70)
    plot_model_selection()

    print("\nDone! Generated 3 figures:")
    print("  - description_hierarchy_moments.png")
    print("  - description_hierarchy_diagram.png")
    print("  - description_hierarchy_model_map.png")
