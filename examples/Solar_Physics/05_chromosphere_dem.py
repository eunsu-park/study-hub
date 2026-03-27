"""
Differential Emission Measure (DEM) Analysis for Solar Atmosphere.

Demonstrates how we deduce the temperature structure of the solar corona
from multi-wavelength EUV observations. The DEM quantifies the amount of
plasma at each temperature along the line of sight.

Key physics:
  - DEM(T) = n_e^2 * ds/dT  [cm^-5 K^-1]
  - Observed intensity: I_i = integral of R_i(T) * DEM(T) dT
  - Each SDO/AIA channel has a temperature response function R_i(T)
  - DEM inversion: ill-posed problem requiring regularization
  - Reveals multi-thermal nature of solar atmosphere
"""

import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt

# --- Temperature grid ---
log_T = np.linspace(4.0, 7.5, 500)  # log10(T/K)
T = 10**log_T

# --- Synthetic DEM model for quiet Sun ---
# The quiet Sun DEM has contributions from chromosphere, transition region, and corona
def gaussian_dem(log_T, log_T0, sigma, amplitude):
    """Gaussian component in log-T space."""
    return amplitude * np.exp(-0.5 * ((log_T - log_T0) / sigma)**2)

# Three main components of quiet Sun DEM
# Chromospheric component: peak at T ~ 10^4 K
dem_chrom = gaussian_dem(log_T, 4.2, 0.3, 1e22)

# Transition region: peak at T ~ 2e5 K (log T ~ 5.3)
dem_tr = gaussian_dem(log_T, 5.3, 0.25, 5e21)

# Coronal component: peak at T ~ 1.5e6 K (log T ~ 6.2)
dem_corona = gaussian_dem(log_T, 6.18, 0.15, 3e21)

# Total DEM
dem_total = dem_chrom + dem_tr + dem_corona

print("=" * 60)
print("Differential Emission Measure (DEM) Analysis")
print("  Quiet Sun model with 3 thermal components:")
print(f"  Chromosphere: peak at log T = 4.2 (T ~ 1.6e4 K)")
print(f"  Trans. Region: peak at log T = 5.3 (T ~ 2.0e5 K)")
print(f"  Corona:       peak at log T = 6.18 (T ~ 1.5e6 K)")
print("=" * 60)

# --- SDO/AIA temperature response functions ---
# Simplified as Gaussian-like functions peaked at characteristic temperatures
# Real response functions come from CHIANTI atomic database
aia_channels = {
    '94 A':  {'log_T_peak': 6.85, 'sigma': 0.2, 'R_peak': 1.5e-26, 'color': 'green'},
    '131 A': {'log_T_peak': 5.75, 'sigma': 0.25, 'R_peak': 2.0e-26, 'color': 'teal'},
    '171 A': {'log_T_peak': 5.85, 'sigma': 0.15, 'R_peak': 8.0e-26, 'color': 'gold'},
    '193 A': {'log_T_peak': 6.20, 'sigma': 0.20, 'R_peak': 5.0e-26, 'color': 'brown'},
    '211 A': {'log_T_peak': 6.30, 'sigma': 0.18, 'R_peak': 3.0e-26, 'color': 'purple'},
    '335 A': {'log_T_peak': 6.45, 'sigma': 0.22, 'R_peak': 1.0e-26, 'color': 'blue'},
}

# Compute response functions
response = {}
for ch_name, ch in aia_channels.items():
    R = ch['R_peak'] * np.exp(-0.5 * ((log_T - ch['log_T_peak']) / ch['sigma'])**2)
    # Add secondary peaks for multi-thermal channels (e.g., 94A has cool and hot peaks)
    if ch_name == '94 A':
        R += 0.3 * ch['R_peak'] * np.exp(-0.5 * ((log_T - 6.2) / 0.15)**2)
    if ch_name == '131 A':
        R += 0.4 * ch['R_peak'] * np.exp(-0.5 * ((log_T - 7.0) / 0.2)**2)
    response[ch_name] = R

# --- Forward model: compute predicted intensities ---
d_logT = log_T[1] - log_T[0]
dT = T * np.log(10) * d_logT  # dT = T * ln(10) * d(logT)

print("\nForward-modeled AIA intensities (DN/s/pixel):")
print(f"{'Channel':>10} {'Intensity':>15} {'Peak log T':>12}")
intensities = {}
for ch_name, R in response.items():
    # I = integral R(T) * DEM(T) dT
    I = np.sum(R * dem_total * dT)
    intensities[ch_name] = I
    print(f"{ch_name:>10} {I:15.3e} {aia_channels[ch_name]['log_T_peak']:12.2f}")

# --- DEM inversion using regularized least squares ---
# Discretize: I_i = sum_j R_ij * DEM_j * dT_j
# This is a linear system: I = A * dem_vec

# Use coarser temperature grid for inversion (ill-posed problem)
n_bins = 30
log_T_inv = np.linspace(4.5, 7.0, n_bins)
T_inv = 10**log_T_inv
d_logT_inv = log_T_inv[1] - log_T_inv[0]
dT_inv = T_inv * np.log(10) * d_logT_inv

# Build response matrix A
n_channels = len(aia_channels)
A = np.zeros((n_channels, n_bins))
for i, (ch_name, ch) in enumerate(aia_channels.items()):
    R_inv = ch['R_peak'] * np.exp(-0.5 * ((log_T_inv - ch['log_T_peak']) / ch['sigma'])**2)
    if ch_name == '94 A':
        R_inv += 0.3 * ch['R_peak'] * np.exp(-0.5 * ((log_T_inv - 6.2) / 0.15)**2)
    if ch_name == '131 A':
        R_inv += 0.4 * ch['R_peak'] * np.exp(-0.5 * ((log_T_inv - 7.0) / 0.2)**2)
    A[i, :] = R_inv * dT_inv

# Add Tikhonov regularization (smoothness constraint)
# Minimize ||A*x - I||^2 + lambda * ||L*x||^2
# where L is second-order difference operator
lambda_reg = 1e-3 * np.max(A)**2
L = np.zeros((n_bins - 2, n_bins))
for j in range(n_bins - 2):
    L[j, j] = 1
    L[j, j + 1] = -2
    L[j, j + 2] = 1

# Stack: [A; sqrt(lambda)*L] * x = [I; 0]
A_reg = np.vstack([A, np.sqrt(lambda_reg) * L])
I_vec = np.array(list(intensities.values()))
b_reg = np.concatenate([I_vec, np.zeros(n_bins - 2)])

# Non-negative least squares (DEM must be positive)
dem_recovered, residual = nnls(A_reg, b_reg)

print(f"\nDEM inversion residual: {residual:.3e}")

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Differential Emission Measure (DEM) Analysis", fontsize=14, y=0.98)

# Panel 1: True DEM
ax = axes[0, 0]
ax.semilogy(log_T, dem_total, 'k-', lw=2.5, label='Total DEM')
ax.semilogy(log_T, dem_chrom, 'r--', lw=1.5, label='Chromosphere', alpha=0.7)
ax.semilogy(log_T, dem_tr, 'g--', lw=1.5, label='Transition Region', alpha=0.7)
ax.semilogy(log_T, dem_corona, 'b--', lw=1.5, label='Corona', alpha=0.7)
ax.set_xlabel(r"$\log_{10}(T / \mathrm{K})$")
ax.set_ylabel(r"DEM [$\mathrm{cm}^{-5}\,\mathrm{K}^{-1}$]")
ax.set_title("Synthetic Quiet Sun DEM")
ax.legend()
ax.set_xlim(4, 7.5)
ax.set_ylim(1e18, 1e23)
ax.grid(True, alpha=0.3)

# Panel 2: AIA response functions
ax = axes[0, 1]
for ch_name, R in response.items():
    ax.semilogy(log_T, R, color=aia_channels[ch_name]['color'], lw=2, label=ch_name)
ax.set_xlabel(r"$\log_{10}(T / \mathrm{K})$")
ax.set_ylabel(r"Response $R(T)$ [DN cm$^5$ s$^{-1}$ pixel$^{-1}$]")
ax.set_title("SDO/AIA Temperature Response Functions")
ax.legend(fontsize=8)
ax.set_xlim(5.0, 7.5)
ax.set_ylim(1e-28, 1e-24)
ax.grid(True, alpha=0.3)

# Panel 3: Predicted channel intensities
ax = axes[1, 0]
ch_names = list(intensities.keys())
ch_vals = list(intensities.values())
colors = [aia_channels[ch]['color'] for ch in ch_names]
bars = ax.bar(ch_names, ch_vals, color=colors, edgecolor='black', alpha=0.8)
ax.set_ylabel(r"Intensity [DN/s/pixel]")
ax.set_title("Predicted AIA Channel Intensities")
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar, val in zip(bars, ch_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.5,
            f'{val:.1e}', ha='center', va='bottom', fontsize=7, rotation=45)

# Panel 4: DEM inversion comparison
ax = axes[1, 1]
ax.semilogy(log_T, dem_total, 'k-', lw=2, label='True DEM', alpha=0.8)
ax.semilogy(log_T_inv, dem_recovered, 'ro-', lw=2, markersize=4,
            label='Recovered DEM (NNLS + regularization)')
ax.set_xlabel(r"$\log_{10}(T / \mathrm{K})$")
ax.set_ylabel(r"DEM [$\mathrm{cm}^{-5}\,\mathrm{K}^{-1}$]")
ax.set_title("DEM Inversion: True vs Recovered")
ax.legend()
ax.set_xlim(4.5, 7.0)
ax.set_ylim(1e18, 1e23)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/opt/projects/01_Personal/03_Study/examples/Solar_Physics/05_chromosphere_dem.png",
            dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved: 05_chromosphere_dem.png")
