"""
Synthetic Solar Spectrum: Blackbody, Absorption Lines, and Zeeman Splitting.

Demonstrates:
- Planck function B(lambda, T) for the solar effective temperature T_eff = 5778 K
- Fraunhofer absorption lines modeled as Gaussian dips
- Zeeman effect on Fe I 617.3 nm: splitting into sigma+, sigma-, pi components
- Stokes I and V profiles for magnetized photospheric plasma
- Limb darkening effect on line profiles

Physics:
    The solar spectrum closely follows a blackbody at T_eff = 5778 K but is
    crossed by thousands of absorption lines (Fraunhofer lines) formed in the
    cooler photosphere/chromosphere. In magnetized regions (sunspots), spectral
    lines split via the Zeeman effect:
        Delta_lambda = g_eff * e * B * lambda^2 / (4*pi*m_e*c)
    where g_eff is the effective Lande g-factor. The Stokes V profile
    (circular polarization) is proportional to dI/dlambda for weak fields,
    enabling magnetometry.
"""

import numpy as np
import matplotlib.pyplot as plt

# === Physical Constants ===
h = 6.626e-34       # Planck constant (J*s)
c = 2.998e8          # speed of light (m/s)
k_B = 1.381e-23      # Boltzmann constant (J/K)
e_charge = 1.602e-19  # electron charge (C)
m_e = 9.109e-31       # electron mass (kg)

T_eff = 5778  # solar effective temperature (K)


def planck(lam, T):
    """
    Planck spectral radiance B(lambda, T) in W / (m^2 sr m).
    lam: wavelength in meters
    T: temperature in Kelvin
    """
    return (2 * h * c ** 2 / lam ** 5) / (np.exp(h * c / (lam * k_B * T)) - 1)


def gaussian_line(lam, lam_0, depth, sigma):
    """Gaussian absorption line profile (returns fractional absorption 0-1)."""
    return depth * np.exp(-0.5 * ((lam - lam_0) / sigma) ** 2)


# === 1. Continuum Spectrum ===
lam = np.linspace(300e-9, 1000e-9, 5000)  # 300 to 1000 nm
B_continuum = planck(lam, T_eff)

# Wien's displacement law: lambda_max = b / T
b_wien = 2.898e-3  # Wien displacement constant (m*K)
lam_peak = b_wien / T_eff

# Total flux (Stefan-Boltzmann): F = sigma * T^4
sigma_SB = 5.670e-8  # W/(m^2 K^4)
F_total = sigma_SB * T_eff ** 4

print("=" * 60)
print("SOLAR SPECTRUM MODEL")
print("=" * 60)
print(f"Effective temperature: {T_eff} K")
print(f"Wien peak wavelength: {lam_peak * 1e9:.1f} nm")
print(f"Total surface flux: {F_total:.3e} W/m^2")

# === 2. Fraunhofer Absorption Lines ===
fraunhofer_lines = {
    'Ca II K': {'lam0': 393.4e-9, 'depth': 0.85, 'sigma': 0.3e-9},
    'Ca II H': {'lam0': 396.8e-9, 'depth': 0.80, 'sigma': 0.3e-9},
    'H-beta':  {'lam0': 486.1e-9, 'depth': 0.50, 'sigma': 0.2e-9},
    'Na D1':   {'lam0': 589.0e-9, 'depth': 0.65, 'sigma': 0.15e-9},
    'Na D2':   {'lam0': 589.6e-9, 'depth': 0.60, 'sigma': 0.15e-9},
    'H-alpha': {'lam0': 656.3e-9, 'depth': 0.70, 'sigma': 0.25e-9},
    'Fe I':    {'lam0': 617.3e-9, 'depth': 0.55, 'sigma': 0.12e-9},
    'O2 A':    {'lam0': 762.0e-9, 'depth': 0.40, 'sigma': 0.5e-9},
}

# Apply absorption lines to continuum
absorption = np.zeros_like(lam)
for name, params in fraunhofer_lines.items():
    absorption += gaussian_line(lam, params['lam0'], params['depth'], params['sigma'])
absorption = np.clip(absorption, 0, 0.99)
B_with_lines = B_continuum * (1 - absorption)

# === 3. Zeeman Splitting of Fe I 617.3 nm ===
# Zeeman splitting: Delta_lambda = g_eff * e * B * lambda^2 / (4*pi*m_e*c)
lam_FeI = 617.3e-9  # Fe I line wavelength
g_eff = 2.5         # effective Lande g-factor (large for this line)
thermal_width = 0.012e-9  # Doppler width at T ~ 5000 K (nm -> m)

B_fields = [0, 0.1, 0.3]  # magnetic field strengths in Tesla (0, 1000, 3000 G)

print("\n--- Zeeman Splitting of Fe I 617.3 nm (g_eff = 2.5) ---")
for B_mag in B_fields:
    if B_mag > 0:
        delta_lam = g_eff * e_charge * B_mag * lam_FeI ** 2 / (4 * np.pi * m_e * c)
        print(f"  B = {B_mag * 1e4:.0f} G: Delta_lambda = {delta_lam * 1e12:.2f} pm "
              f"({delta_lam * 1e9:.4f} nm)")
    else:
        print(f"  B = 0 G: no splitting")

# High-resolution wavelength grid around Fe I line
lam_hr = np.linspace(lam_FeI - 0.1e-9, lam_FeI + 0.1e-9, 1000)

# Compute Stokes I and V profiles for each B
stokes_profiles = {}
for B_mag in B_fields:
    if B_mag == 0:
        # Unsplit line: single Gaussian
        I_profile = 1 - 0.55 * np.exp(-0.5 * ((lam_hr - lam_FeI) / thermal_width) ** 2)
        V_profile = np.zeros_like(lam_hr)
    else:
        delta_lam = g_eff * e_charge * B_mag * lam_FeI ** 2 / (4 * np.pi * m_e * c)

        # sigma- component (lam_0 - delta_lam): right circular
        sigma_minus = 0.275 * np.exp(
            -0.5 * ((lam_hr - (lam_FeI - delta_lam)) / thermal_width) ** 2)
        # pi component (lam_0): linear (does not contribute to Stokes V)
        pi_comp = 0.275 * np.exp(
            -0.5 * ((lam_hr - lam_FeI) / thermal_width) ** 2)
        # sigma+ component (lam_0 + delta_lam): left circular
        sigma_plus = 0.275 * np.exp(
            -0.5 * ((lam_hr - (lam_FeI + delta_lam)) / thermal_width) ** 2)

        # Stokes I: sum of all components
        I_profile = 1 - (sigma_minus + pi_comp + sigma_plus)
        # Stokes V: difference of sigma components (circular polarization)
        V_profile = sigma_plus - sigma_minus

    stokes_profiles[B_mag] = {'I': I_profile, 'V': V_profile}

# === 4. Limb Darkening Effect ===
# Limb darkening: I(mu) = I_0 * (1 - u*(1-mu))
# where mu = cos(theta) and u ~ 0.6 (linear coefficient)
u_ld = 0.6
mu_values = [1.0, 0.5, 0.2]  # center, mid, near-limb

print("\n--- Limb Darkening ---")
for mu in mu_values:
    I_ratio = 1 - u_ld * (1 - mu)
    print(f"  mu = {mu:.1f} (theta = {np.degrees(np.arccos(mu)):.0f} deg): "
          f"I/I_center = {I_ratio:.3f}")

# === Plotting ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Full solar spectrum with Fraunhofer lines
ax = axes[0, 0]
lam_nm = lam * 1e9
ax.plot(lam_nm, B_continuum / 1e12, 'gray', linewidth=1, alpha=0.5,
        label='Blackbody continuum')
ax.plot(lam_nm, B_with_lines / 1e12, 'k-', linewidth=0.5,
        label='With absorption lines')
# Mark prominent lines
for name, params in fraunhofer_lines.items():
    lam0_nm = params['lam0'] * 1e9
    if params['depth'] > 0.45:
        ax.annotate(name, xy=(lam0_nm, planck(params['lam0'], T_eff) / 1e12 * 0.4),
                    fontsize=6, ha='center', color='red', rotation=90)
ax.axvline(lam_peak * 1e9, color='orange', linestyle=':', alpha=0.5,
           label=f'Wien peak ({lam_peak * 1e9:.0f} nm)')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Spectral Radiance ($10^{12}$ W/m$^2$/sr/m)')
ax.set_title('Solar Spectrum (T = 5778 K)')
ax.legend(fontsize=8)

# 2. Zeeman-split Stokes I profiles
ax = axes[0, 1]
lam_hr_pm = (lam_hr - lam_FeI) * 1e12  # offset in pm
colors_B = {0: 'black', 0.1: 'blue', 0.3: 'red'}
for B_mag in B_fields:
    label_B = f'B = {B_mag * 1e4:.0f} G'
    ax.plot(lam_hr_pm, stokes_profiles[B_mag]['I'],
            color=colors_B[B_mag], linewidth=1.5, label=label_B)
ax.set_xlabel('$\\Delta\\lambda$ from 617.3 nm (pm)')
ax.set_ylabel('Stokes I (normalized)')
ax.set_title(f'Zeeman Effect: Fe I 617.3 nm (g$_{{eff}}$ = {g_eff})')
ax.legend(fontsize=9)

# 3. Stokes V profiles
ax = axes[1, 0]
for B_mag in B_fields:
    if B_mag == 0:
        continue
    label_B = f'B = {B_mag * 1e4:.0f} G'
    ax.plot(lam_hr_pm, stokes_profiles[B_mag]['V'],
            color=colors_B[B_mag], linewidth=2, label=label_B)
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('$\\Delta\\lambda$ from 617.3 nm (pm)')
ax.set_ylabel('Stokes V')
ax.set_title('Stokes V (Circular Polarization) - Magnetogram Signal')
ax.legend(fontsize=9)

# 4. Limb darkening effect on continuum spectrum
ax = axes[1, 1]
for mu in mu_values:
    I_factor = 1 - u_ld * (1 - mu)
    B_limb = B_with_lines * I_factor
    theta_deg = np.degrees(np.arccos(mu))
    ax.plot(lam_nm, B_limb / 1e12, linewidth=1,
            label=f'$\\mu$ = {mu:.1f} ($\\theta$ = {theta_deg:.0f}$^\\circ$)')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Spectral Radiance ($10^{12}$ W/m$^2$/sr/m)')
ax.set_title(f'Limb Darkening (u = {u_ld})')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('/opt/projects/01_Personal/03_Study/examples/Solar_Physics/11_solar_spectrum.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to 11_solar_spectrum.png")
