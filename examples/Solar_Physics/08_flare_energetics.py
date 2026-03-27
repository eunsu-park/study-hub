"""
Solar Flare Energy Partition Model.

Demonstrates:
- Magnetic energy estimation from coronal field parameters
- Energy partition among CME, thermal, non-thermal, and radiated components
  (based on Emslie et al. 2012 statistical study)
- Thick-target bremsstrahlung model: electron spectrum -> photon spectrum
- Neupert effect: SXR ~ integral of HXR, linking thermal and non-thermal emission

Physics:
    Solar flares release magnetic energy stored in the corona through reconnection.
    The total energy E_mag = B^2/(2*mu_0) * V is partitioned into bulk kinetic
    energy (CME), thermal plasma heating, non-thermal particle acceleration,
    and direct radiation. The thick-target model (Brown 1971) predicts the
    hard X-ray photon spectrum from the injected electron power-law spectrum.
    The Neupert effect (1968) states that the time derivative of SXR flux
    approximates the HXR flux, because thermal emission accumulates as
    non-thermal electrons heat the chromosphere.
"""

import numpy as np
import matplotlib.pyplot as plt

# === Physical Constants ===
mu_0 = 4 * np.pi * 1e-7  # vacuum permeability (H/m)
keV_to_J = 1.602e-16      # 1 keV in Joules
erg_to_J = 1e-7            # 1 erg in Joules

# === 1. Magnetic Energy and Partition ===
print("=" * 60)
print("SOLAR FLARE ENERGY PARTITION MODEL")
print("=" * 60)

# Define flare classes by typical parameters
# (B in Tesla, Volume in m^3, reconnection fraction)
flare_classes = {
    'C1.0': {'B': 0.005, 'L': 1e7, 'f_reconn': 0.1},   # B=50 G, L=10 Mm
    'M1.0': {'B': 0.010, 'L': 2e7, 'f_reconn': 0.2},   # B=100 G, L=20 Mm
    'X1.0': {'B': 0.020, 'L': 3e7, 'f_reconn': 0.3},   # B=200 G, L=30 Mm
    'X10':  {'B': 0.030, 'L': 5e7, 'f_reconn': 0.4},   # B=300 G, L=50 Mm
}

# Energy partition fractions (Emslie et al. 2012 median values)
partition = {
    'CME kinetic':   0.40,
    'Thermal':       0.30,
    'Non-thermal e': 0.20,
    'Radiated':      0.10,
}

print("\nFlare Class | B (G) | Volume (cm^3) | E_mag (erg) | Partitioned Energies")
print("-" * 95)

flare_energies = {}
for cls, params in flare_classes.items():
    B = params['B']           # Tesla
    L = params['L']           # meters (characteristic length)
    V = L ** 3                # volume (m^3)
    f = params['f_reconn']    # fraction of field that reconnects

    # Total available magnetic energy (Joules)
    E_mag = (B ** 2 / (2 * mu_0)) * V * f
    E_mag_erg = E_mag / erg_to_J  # convert to erg (standard solar physics unit)

    flare_energies[cls] = E_mag_erg

    B_gauss = B * 1e4  # Tesla to Gauss
    V_cm3 = V * 1e6    # m^3 to cm^3

    print(f"  {cls:5s}     | {B_gauss:5.0f} | {V_cm3:12.2e} | {E_mag_erg:10.2e} |", end="")
    for name, frac in partition.items():
        print(f" {name}: {frac * E_mag_erg:.1e}", end=",")
    print()

# === 2. Thick-Target Bremsstrahlung Model ===
print("\n" + "=" * 60)
print("THICK-TARGET BREMSSTRAHLUNG MODEL")
print("=" * 60)

# Injected electron spectrum: F(E) = F_0 * E^(-delta) [electrons/s/keV]
# For thick target: photon spectrum I(eps) ~ eps^(-(delta+1)/2) at low energies
#   and I(eps) ~ eps^(-(delta-1)) at high energies (Brown 1971)
# Simplified: I(eps) ~ eps^(-gamma) where gamma ~ delta - 1

delta_values = [3.0, 4.0, 5.0, 7.0]  # electron spectral indices
E_electron = np.logspace(1, 3, 200)   # 10 keV to 1000 keV
eps_photon = np.logspace(0.5, 2.7, 200)  # 3 keV to 500 keV

E_cutoff = 20.0  # low-energy cutoff (keV)

for delta in delta_values:
    gamma = delta - 1  # photon spectral index (thick-target approximation)
    print(f"  delta = {delta:.1f} -> photon index gamma = {gamma:.1f}")

# === 3. Neupert Effect Demonstration ===
# Generate synthetic HXR (impulsive) and SXR (gradual) light curves
t_lc = np.linspace(0, 600, 2000)  # 0 to 600 seconds

# HXR: sum of impulsive peaks (representing individual energy release episodes)
hxr = np.zeros_like(t_lc)
# Multiple impulsive bursts
bursts = [
    {'t0': 60, 'sigma': 8, 'amp': 0.3},
    {'t0': 100, 'sigma': 12, 'amp': 1.0},   # main peak
    {'t0': 150, 'sigma': 6, 'amp': 0.5},
    {'t0': 180, 'sigma': 10, 'amp': 0.4},
    {'t0': 230, 'sigma': 5, 'amp': 0.15},
]
for burst in bursts:
    hxr += burst['amp'] * np.exp(-0.5 * ((t_lc - burst['t0']) / burst['sigma']) ** 2)

# SXR = time integral of HXR (Neupert effect) + slow decay (radiative cooling)
dt_lc = t_lc[1] - t_lc[0]
sxr_raw = np.cumsum(hxr) * dt_lc
# Add exponential cooling after peak
cooling_time = 300.0  # seconds
peak_idx = np.argmax(sxr_raw)
sxr = sxr_raw.copy()
for i in range(peak_idx, len(t_lc)):
    decay = np.exp(-(t_lc[i] - t_lc[peak_idx]) / cooling_time)
    sxr[i] = sxr_raw[peak_idx] * decay + sxr_raw[i] * (1 - decay) * 0.3

# Normalize for display
hxr_norm = hxr / np.max(hxr)
sxr_norm = sxr / np.max(sxr)

# === Plotting ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Energy partition bar chart
ax = axes[0, 0]
classes = list(flare_energies.keys())
x_pos = np.arange(len(classes))
bottom = np.zeros(len(classes))
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
for (name, frac), color in zip(partition.items(), colors):
    vals = [frac * flare_energies[c] for c in classes]
    ax.bar(x_pos, vals, bottom=bottom, label=name, color=color, width=0.6)
    bottom += vals
ax.set_xticks(x_pos)
ax.set_xticklabels(classes)
ax.set_yscale('log')
ax.set_ylabel('Energy (erg)')
ax.set_title('Flare Energy Partition by Class')
ax.legend(fontsize=8, loc='upper left')

# 2. Electron and photon spectra
ax = axes[0, 1]
colors_spec = plt.cm.viridis(np.linspace(0.2, 0.9, len(delta_values)))
for delta, c in zip(delta_values, colors_spec):
    # Electron spectrum with low-energy cutoff
    F_e = np.where(E_electron > E_cutoff,
                   E_electron ** (-delta), 0)
    F_e /= np.max(F_e[F_e > 0]) if np.any(F_e > 0) else 1
    ax.loglog(E_electron, F_e, '--', color=c, alpha=0.6,
              label=f'e$^-$ $\\delta$={delta:.0f}')

    # Photon spectrum (thick-target: gamma = delta - 1)
    gamma = delta - 1
    I_ph = eps_photon ** (-gamma)
    I_ph /= np.max(I_ph)
    ax.loglog(eps_photon, I_ph * 0.1, '-', color=c, linewidth=2,
              label=f'$\\gamma$ $\\Gamma$={gamma:.0f}')

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Flux (normalized)')
ax.set_title('Thick-Target: Electron & Photon Spectra')
ax.legend(fontsize=7, ncol=2)
ax.set_ylim(1e-8, 2)

# 3. Neupert Effect
ax = axes[1, 0]
ax.plot(t_lc, hxr_norm, 'b-', linewidth=1.5, label='HXR (non-thermal)')
ax.plot(t_lc, sxr_norm, 'r-', linewidth=2, label='SXR (thermal)')
ax.fill_between(t_lc, 0, hxr_norm, alpha=0.15, color='blue')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Normalized Flux')
ax.set_title('Neupert Effect: SXR $\\approx \\int$ HXR dt')
ax.legend()
ax.set_xlim(0, 500)

# 4. dSXR/dt vs HXR comparison (Neupert verification)
ax = axes[1, 1]
dsxr_dt = np.gradient(sxr_norm, dt_lc)
dsxr_dt = np.maximum(dsxr_dt, 0)  # only positive derivative
dsxr_dt_norm = dsxr_dt / np.max(dsxr_dt) if np.max(dsxr_dt) > 0 else dsxr_dt
ax.plot(t_lc, hxr_norm, 'b-', linewidth=1.5, label='HXR', alpha=0.8)
ax.plot(t_lc, dsxr_dt_norm, 'r--', linewidth=1.5, label='dSXR/dt (normalized)')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Normalized Flux')
ax.set_title('Neupert Verification: HXR vs dSXR/dt')
ax.legend()
ax.set_xlim(0, 400)

# Compute correlation between HXR and dSXR/dt in impulsive phase
imp_mask = (t_lc > 30) & (t_lc < 280)
corr_neupert = np.corrcoef(hxr_norm[imp_mask], dsxr_dt_norm[imp_mask])[0, 1]

print(f"\nNeupert effect correlation (HXR vs dSXR/dt): {corr_neupert:.3f}")
print(f"HXR peak time: {t_lc[np.argmax(hxr_norm)]:.0f} s")
print(f"SXR peak time: {t_lc[np.argmax(sxr_norm)]:.0f} s")
print(f"SXR peak delay: {t_lc[np.argmax(sxr_norm)] - t_lc[np.argmax(hxr_norm)]:.0f} s")

plt.tight_layout()
plt.savefig('/opt/projects/01_Personal/03_Study/examples/Solar_Physics/08_flare_energetics.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to 08_flare_energetics.png")
