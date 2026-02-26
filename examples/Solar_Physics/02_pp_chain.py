"""
Nuclear Energy Generation: pp Chain Reaction Rates and Gamow Peak.

Demonstrates thermonuclear fusion in the solar core via the proton-proton
chain. The key concept is the Gamow peak: the energy window where the
product of the Maxwell-Boltzmann tail and quantum tunneling probability
is maximized.

Key physics:
  - Coulomb barrier tunneling: P ~ exp(-sqrt(E_G/E))
  - Maxwell-Boltzmann distribution: f(E) ~ exp(-E/kT)
  - Gamow peak energy: E_0 = (E_G * (kT)^2 / 4)^(1/3)
  - pp chain: 4p -> He-4 + 2e+ + 2nu + 26.73 MeV
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Physical constants ---
k_B = 1.381e-23       # Boltzmann constant [J/K]
k_B_keV = 8.617e-8    # Boltzmann constant [keV/K]
m_p = 1.673e-27       # proton mass [kg]
e_charge = 1.602e-19  # elementary charge [C]
hbar = 1.055e-34      # reduced Planck constant [J s]
pi = np.pi

# --- Gamow energy for pp reaction ---
# E_G = (pi * alpha_em * Z1 * Z2)^2 * 2 * m_reduced * c^2
# For pp: Z1=Z2=1, m_reduced = m_p/2
alpha_em = 1 / 137.036  # fine structure constant
c = 3e8                  # speed of light [m/s]
m_reduced = m_p / 2      # reduced mass for pp

E_G_J = (pi * alpha_em) ** 2 * 2 * m_reduced * c**2
E_G_keV = E_G_J / (e_charge * 1e3)

print("=" * 60)
print("Gamow Energy for pp Reaction")
print(f"  E_G = {E_G_keV:.2f} keV  (literature: ~493 keV)")
print("=" * 60)

# --- Gamow peak for solar core temperature ---
T_core = 1.57e7  # K
kT = k_B_keV * T_core  # keV

# Gamow peak energy: E_0 = (E_G * (kT)^2 / 4)^(1/3)
E_0 = (E_G_keV * kT**2 / 4) ** (1.0 / 3.0)

# Width of Gamow peak: Delta = 4 * sqrt(E_0 * kT / 3)
Delta_E = 4 * np.sqrt(E_0 * kT / 3)

print(f"\nAt T_core = {T_core:.2e} K  (kT = {kT:.3f} keV):")
print(f"  Gamow peak energy:  E_0     = {E_0:.2f} keV")
print(f"  Gamow peak width:   Delta_E = {Delta_E:.2f} keV")
print(f"  E_0 / kT = {E_0 / kT:.1f}  (peak is far above thermal energy)")

# --- Plot the Gamow peak integrand ---
# f(E) ~ exp(-E/kT) * exp(-sqrt(E_G/E))  (unnormalized)
E = np.linspace(0.1, 30, 1000)  # keV

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Components of the Gamow peak
mb_tail = np.exp(-E / kT)
tunnel = np.exp(-np.sqrt(E_G_keV / E))
gamow_integrand = np.exp(-E / kT - np.sqrt(E_G_keV / E))

# Normalize for display
gamow_norm = gamow_integrand / gamow_integrand.max()
mb_norm = mb_tail / mb_tail[0]
tunnel_norm = tunnel / tunnel.max()

axes[0].plot(E, mb_norm, 'b--', lw=2, label=r'Maxwell-Boltzmann: $e^{-E/kT}$')
axes[0].plot(E, tunnel_norm, 'r--', lw=2, label=r'Tunneling: $e^{-\sqrt{E_G/E}}$')
axes[0].plot(E, gamow_norm, 'k-', lw=3, label='Gamow peak (product)')
axes[0].axvline(E_0, color='green', ls=':', lw=2, label=f'$E_0$ = {E_0:.1f} keV')
axes[0].fill_between(E, 0, gamow_norm, where=(np.abs(E - E_0) < Delta_E / 2),
                     alpha=0.3, color='yellow', label=f'$\\Delta E$ = {Delta_E:.1f} keV')
axes[0].set_xlabel("Energy [keV]")
axes[0].set_ylabel("Relative probability")
axes[0].set_title("Gamow Peak: Where Fusion Happens")
axes[0].legend(fontsize=8)
axes[0].set_xlim(0, 30)
axes[0].set_ylim(0, 1.1)
axes[0].grid(True, alpha=0.3)

# --- Energy generation rates ---
# Simplified pp rate: epsilon_pp = 1.08e-12 * rho * X^2 * T6^4 * psi * f_pp * g_pp [W/kg]
# For simplicity: psi ~ 1, f_pp ~ 1, g_pp ~ 1 + 0.0123*T6^(1/3) + 0.0109*T6^(2/3) + 0.0009*T6
# Simplified CNO: epsilon_CNO ~ 8.24e-31 * rho * X * X_CNO * T6^16 [W/kg]

rho_core = 1.5e5   # central density [kg/m^3]
X = 0.73            # hydrogen mass fraction
X_CNO = 0.01        # CNO mass fraction

T6_range = np.linspace(5, 30, 500)  # T in units of 10^6 K
T_range = T6_range * 1e6

# pp chain rate
g_pp = 1 + 0.0123 * T6_range**(1.0/3) + 0.0109 * T6_range**(2.0/3) + 0.0009 * T6_range
eps_pp = 1.08e-12 * rho_core * X**2 * T6_range**4 * g_pp  # W/kg

# CNO cycle rate (approximate)
eps_cno = 8.24e-31 * rho_core * X * X_CNO * T6_range**16  # W/kg

# Panel 2: Energy generation vs temperature
axes[1].semilogy(T6_range, eps_pp, 'b-', lw=2, label='pp chain')
axes[1].semilogy(T6_range, eps_cno, 'r-', lw=2, label='CNO cycle')
axes[1].axvline(15.7, color='green', ls=':', lw=2, label='Solar core (15.7 MK)')
axes[1].set_xlabel(r"Temperature [$10^6$ K]")
axes[1].set_ylabel(r"$\epsilon$ [W/kg]")
axes[1].set_title("Energy Generation Rate vs Temperature")
axes[1].legend()
axes[1].set_ylim(1e-8, 1e8)
axes[1].grid(True, alpha=0.3)

# Find crossover temperature
cross_idx = np.argmin(np.abs(eps_pp - eps_cno))
T_cross = T6_range[cross_idx]
print(f"\npp-CNO crossover temperature: ~{T_cross:.1f} MK (literature: ~17 MK)")

# Panel 3: Power-law index (d ln epsilon / d ln T)
# Shows T^4 for pp and T^16-20 for CNO
d_ln_eps_pp = np.gradient(np.log(eps_pp + 1e-30), np.log(T6_range))
d_ln_eps_cno = np.gradient(np.log(np.maximum(eps_cno, 1e-50)), np.log(T6_range))

axes[2].plot(T6_range, d_ln_eps_pp, 'b-', lw=2, label='pp chain')
axes[2].plot(T6_range, d_ln_eps_cno, 'r-', lw=2, label='CNO cycle')
axes[2].axhline(4, color='b', ls=':', alpha=0.5, label=r'$T^4$ reference')
axes[2].axhline(16, color='r', ls=':', alpha=0.5, label=r'$T^{16}$ reference')
axes[2].set_xlabel(r"Temperature [$10^6$ K]")
axes[2].set_ylabel(r"$d \ln \epsilon / d \ln T$")
axes[2].set_title("Temperature Sensitivity Index")
axes[2].legend()
axes[2].set_ylim(0, 25)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/opt/projects/01_Personal/03_Study/examples/Solar_Physics/02_pp_chain.png",
            dpi=150, bbox_inches='tight')
plt.show()

# --- Luminosity estimate ---
# Simple integral: L ~ 4*pi*int(r^2 * rho * epsilon dr) over core
# Using core average: rho_avg ~ rho_c/3, T_avg ~ T_c * 0.8, r_core ~ 0.25 R_sun
R_core = 0.25 * 6.957e8
rho_avg = rho_core / 3
T6_avg = 15.7 * 0.8
g = 1 + 0.0123 * T6_avg**(1.0/3) + 0.0109 * T6_avg**(2.0/3) + 0.0009 * T6_avg
eps_avg = 1.08e-12 * rho_avg * X**2 * T6_avg**4 * g
V_core = (4.0 / 3.0) * pi * R_core**3
L_est = rho_avg * eps_avg * V_core
print(f"\nLuminosity estimate (crude core integral):")
print(f"  eps_avg = {eps_avg:.3e} W/kg")
print(f"  L_est   = {L_est:.3e} W")
print(f"  L_sun   = 3.828e+26 W")
print(f"  Ratio   = {L_est / 3.828e26:.2f}")

print("\nPlot saved: 02_pp_chain.png")
