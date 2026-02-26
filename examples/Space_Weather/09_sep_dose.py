"""
Solar Energetic Particle (SEP) Events: Spectra, Dose, and Shielding.

Demonstrates:
- Power-law proton energy spectrum with spectral break (Band-like function)
- Bethe-Bloch stopping power for protons in water (tissue equivalent)
- Dose rate computation by integrating flux * LET over energy
- Dose behind various shielding thicknesses (aluminum equivalent)
- NOAA S-scale classification based on >10 MeV proton flux
- Comparison of extreme (Halloween 2003-type) vs moderate SEP events

Physics:
    Solar Energetic Particles (SEPs) are accelerated in solar flares and
    CME-driven shocks. The proton differential energy spectrum is often
    described by a power law with a high-energy rollover:

        dJ/dE = J_0 * (E/E_0)^(-gamma) * exp(-E/E_c)

    where gamma ~ 2-4 is the spectral index, E_c is the cutoff energy,
    and J_0 is the normalization [protons / (cm^2 s sr MeV)].

    The dose to biological tissue depends on the Linear Energy Transfer (LET),
    which for protons in water follows the Bethe-Bloch formula:

        S(E) = (K * Z^2 / beta^2) * [ln(2*m_e*c^2*beta^2*gamma^2/I) - beta^2]

    where beta = v/c, gamma = Lorentz factor, I ~ 75 eV for water, and
    K = 0.307 MeV cm^2/g.

    Shielding reduces dose by attenuating low-energy particles. The effective
    dose behind shielding of thickness x [g/cm^2] is computed by finding
    the minimum proton energy that penetrates the shield (range-energy relation).

    NOAA S-scale (S1-S5) classifies SEP events by >10 MeV integral flux:
        S1: >10 pfu, S2: >100 pfu, S3: >1000 pfu, S4: >10000 pfu, S5: >100000 pfu
    where 1 pfu = 1 proton / (cm^2 s sr).

References:
    - Band, D. et al. (1993). "BATSE observations of gamma-ray burst spectra."
    - Mewaldt, R.A. et al. (2005). "Proton, helium, and electron spectra
      during the large solar particle events of October-November 2003."
    - NCRP Report 153 (2006). "Information Needed to Make Radiation Protection
      Recommendations for Space Missions Beyond Low-Earth Orbit."
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# === Physical Constants ===
m_p = 938.272       # proton rest mass [MeV/c^2]
m_e = 0.511         # electron rest mass [MeV/c^2]
c = 3e8             # speed of light [m/s]
I_water = 75e-6     # mean excitation energy of water [MeV]
K_BB = 0.307        # Bethe-Bloch constant [MeV cm^2/g]
Z_p = 1             # proton charge number

print("=" * 65)
print("SOLAR ENERGETIC PARTICLE DOSE AND SHIELDING")
print("=" * 65)


# =========================================================================
# 1. SEP PROTON ENERGY SPECTRUM
# =========================================================================
def sep_spectrum(E, J0, gamma, E_c, E_break=5.0):
    """
    Differential proton flux spectrum (power law with cutoffs at both ends).

    dJ/dE = J0 * (E/E0)^(-gamma) * exp(-E/E_c) * [1 - exp(-E/E_break)]

    The high-energy exponential cutoff (E_c) models spectral rollover.
    The low-energy factor [1 - exp(-E/E_break)] models the spectral
    turnover below a few MeV, where acceleration efficiency drops and
    adiabatic deceleration reduces flux.

    Parameters:
        E       : kinetic energy [MeV]
        J0      : flux normalization at E0 [protons / (cm^2 s sr MeV)]
        gamma   : spectral index (harder = smaller gamma)
        E_c     : high-energy cutoff/rollover energy [MeV]
        E_break : low-energy turnover energy [MeV]

    Returns:
        dJ/dE : differential flux [protons / (cm^2 s sr MeV)]
    """
    E0 = 10.0  # reference energy [MeV]
    low_cutoff = 1.0 - np.exp(-E / E_break)  # suppresses flux below E_break
    return J0 * (E / E0)**(-gamma) * np.exp(-E / E_c) * low_cutoff


def integral_flux_above(E_threshold, E_array, spectrum):
    """
    Compute integral flux above E_threshold.

    J(>E_threshold) = integral from E_threshold to E_max of dJ/dE dE
    Returns flux in [protons / (cm^2 s sr)]
    """
    mask = E_array >= E_threshold
    if mask.sum() < 2:
        return 0.0
    return np.trapezoid(spectrum[mask], E_array[mask])


# === Event Parameters ===
# J0 is the differential flux at E0=10 MeV [protons/(cm^2 s sr MeV)]
# Calibrated so that integral >10 MeV flux matches typical SEP event magnitudes.
# Halloween 2003: >10 MeV ~ 30000 pfu peak, >100 MeV ~ 300 pfu
# Typical S3: >10 MeV ~ 1000-5000 pfu
# Typical S2: >10 MeV ~ 100-500 pfu
extreme = {'J0': 3e3, 'gamma': 2.0, 'E_c': 500.0, 'label': 'Extreme (Halloween 2003-type)'}
large = {'J0': 5e2, 'gamma': 2.5, 'E_c': 200.0, 'label': 'Large (S3-level)'}
moderate = {'J0': 80, 'gamma': 3.0, 'E_c': 100.0, 'label': 'Moderate (S2-level)'}

events = [extreme, large, moderate]
colors_ev = ['red', 'orange', 'blue']

# Energy grid (logarithmic, 1 MeV to 1 GeV)
E = np.logspace(0, 3, 500)  # [MeV]

print("\n--- SEP Event Spectra ---")
print(f"{'Event':<35} {'gamma':<8} {'E_c [MeV]':<12} {'>10 MeV [pfu]':<15} {'>100 MeV [pfu]':<15}")
for ev in events:
    spec = sep_spectrum(E, ev['J0'], ev['gamma'], ev['E_c'])
    f10 = integral_flux_above(10, E, spec)
    f100 = integral_flux_above(100, E, spec)
    ev['f10'] = f10
    ev['f100'] = f100
    print(f"  {ev['label']:<33} {ev['gamma']:<8.1f} {ev['E_c']:<12.0f} "
          f"{f10:<15.1f} {f100:<15.1f}")


# =========================================================================
# 2. NOAA S-SCALE CLASSIFICATION
# =========================================================================
def noaa_s_scale(flux_10mev):
    """
    NOAA Space Weather S-scale from >10 MeV proton flux [pfu].

    S1: Minor (10), S2: Moderate (100), S3: Strong (1000),
    S4: Severe (10000), S5: Extreme (100000)
    """
    thresholds = [(1e5, 'S5 (Extreme)'), (1e4, 'S4 (Severe)'),
                  (1e3, 'S3 (Strong)'), (1e2, 'S2 (Moderate)'),
                  (1e1, 'S1 (Minor)')]
    for thresh, label in thresholds:
        if flux_10mev >= thresh:
            return label
    return 'Below S1'


print("\n--- NOAA S-Scale Classification ---")
for ev in events:
    scale = noaa_s_scale(ev['f10'])
    print(f"  {ev['label']:<35} -> {scale}")


# =========================================================================
# 3. BETHE-BLOCH STOPPING POWER (PROTONS IN WATER)
# =========================================================================
def bethe_bloch_water(E_kin):
    """
    Stopping power for protons in water (simplified Bethe-Bloch).

    S(E) = (K * Z^2 / beta^2) * [ln(2*m_e*c^2*beta^2*gamma_L^2 / I) - beta^2]

    Parameters:
        E_kin : proton kinetic energy [MeV]

    Returns:
        S : mass stopping power [MeV cm^2/g]
    """
    E_total = E_kin + m_p  # total energy [MeV]
    gamma_L = E_total / m_p  # Lorentz factor
    beta2 = 1.0 - 1.0 / gamma_L**2
    beta2 = np.maximum(beta2, 1e-10)  # avoid log(0)

    # Bethe-Bloch formula
    ln_arg = 2 * m_e * beta2 * gamma_L**2 / I_water
    ln_arg = np.maximum(ln_arg, 1.0)
    S = K_BB * Z_p**2 / beta2 * (np.log(ln_arg) - beta2)
    return np.maximum(S, 0.0)


# === Proton Range in Water ===
def proton_range_water(E_kin):
    """
    CSDA (continuous slowing-down) range of proton in water [g/cm^2].

    R(E) = integral from 0 to E of dE'/S(E')

    Computed by numerical integration of the inverse stopping power.
    """
    E_arr = np.linspace(0.1, E_kin, 500)
    S_arr = bethe_bloch_water(E_arr)
    S_arr = np.maximum(S_arr, 0.1)  # prevent division by zero
    return np.trapezoid(1.0 / S_arr, E_arr)


print("\n--- Proton Stopping Power and Range in Water ---")
print(f"{'Energy [MeV]':<15} {'S [MeV cm²/g]':<18} {'Range [g/cm²]':<18} {'Range [cm H₂O]':<18}")
for E_check in [10, 50, 100, 200, 500, 1000]:
    S = bethe_bloch_water(E_check)
    R = proton_range_water(E_check)
    print(f"  {E_check:<13} {S:<18.2f} {R:<18.2f} {R:<18.2f}")


# =========================================================================
# 4. DOSE COMPUTATION
# =========================================================================
def compute_dose_rate(E_array, spectrum):
    """
    Dose rate from SEP proton spectrum.

    dose_rate = integral of (dJ/dE * S(E) / rho) dE * geometric_factor

    For omnidirectional flux, geometric factor is 4*pi sr.
    Convert from [MeV / g] to [Gy/hr] using 1 MeV/g = 1.602e-10 Gy.

    Parameters:
        E_array  : energy grid [MeV]
        spectrum : dJ/dE [protons / (cm^2 s sr MeV)]

    Returns:
        dose_rate : [mGy/hr]
    """
    S = bethe_bloch_water(E_array)
    # Integrand: flux * stopping power, units: protons/(cm^2 s sr MeV) * MeV*cm^2/g
    #   = protons * MeV / (g s sr)
    integrand = spectrum * S

    # Integrate over energy
    dose_mev_per_g_s_sr = np.trapezoid(integrand, E_array)

    # Multiply by 4*pi sr (omnidirectional) and convert units
    # 1 MeV/g = 1.602e-10 J/g = 1.602e-7 J/kg = 1.602e-7 Gy
    dose_gy_per_s = dose_mev_per_g_s_sr * 4 * np.pi * 1.602e-7
    dose_mgy_per_hr = dose_gy_per_s * 3600 * 1e3  # mGy/hr

    return dose_mgy_per_hr


print("\n--- Unshielded Dose Rates ---")
for i, ev in enumerate(events):
    spec = sep_spectrum(E, ev['J0'], ev['gamma'], ev['E_c'])
    dr = compute_dose_rate(E, spec)
    print(f"  {ev['label']:<35}: {dr:.2f} mGy/hr")
    ev['dose_rate_unshielded'] = dr


# =========================================================================
# 5. SHIELDING EFFECT
# =========================================================================
def _build_range_energy_table():
    """
    Build a lookup table for proton range vs energy in water.

    Returns (E_table, R_table) arrays for interpolation.
    """
    E_table = np.logspace(-0.5, 3.5, 400)  # 0.3 MeV to 3 GeV
    R_table = np.array([proton_range_water(e) for e in E_table])
    return E_table, R_table


# Pre-compute range table once
_E_table, _R_table = _build_range_energy_table()


def residual_energy(E_in, shield_gcm2):
    """
    Compute residual kinetic energy of a proton after traversing shield.

    A proton with incident energy E_in has range R(E_in) in water.
    After passing through shield_gcm2 of material, its residual range is
    R_out = R(E_in) - shield_gcm2. The residual energy E_out is found
    by inverting the range-energy relation.

    Parameters:
        E_in        : incident kinetic energy [MeV]
        shield_gcm2 : shield thickness [g/cm^2]

    Returns:
        E_out : residual energy [MeV], or 0 if stopped
    """
    R_in = np.interp(E_in, _E_table, _R_table)
    R_out = R_in - shield_gcm2
    if R_out <= 0:
        return 0.0
    # Invert: find E where R(E) = R_out
    E_out = np.interp(R_out, _R_table, _E_table)
    return E_out


def shielded_dose_rate(E_array, spectrum, shield_gcm2):
    """
    Dose rate behind shielding of given thickness.

    For each incident proton energy E_in:
    1. If range(E_in) < shield, the proton is stopped -> no dose
    2. Otherwise, compute residual energy E_out after shield
    3. Dose contribution uses S(E_out), the stopping power at the
       degraded energy behind the shield

    The transmitted spectrum has the same particle flux (particles are
    not removed, just slowed down) but with degraded energies, so
    dose = integral(J(E_in) * S(E_out(E_in)) * dE_in) * geometric_factor

    Parameters:
        E_array     : energy grid [MeV]
        spectrum    : dJ/dE incident spectrum
        shield_gcm2 : shielding thickness [g/cm^2]

    Returns:
        dose_rate : shielded dose rate [mGy/hr]
    """
    if shield_gcm2 <= 0:
        return compute_dose_rate(E_array, spectrum)

    # Find minimum energy to penetrate shield
    E_min_idx = np.searchsorted(_R_table, shield_gcm2)
    if E_min_idx >= len(_E_table):
        return 0.0
    E_min = _E_table[E_min_idx]

    # For each incident energy above E_min, compute residual energy
    # and use S(E_out) for dose calculation
    integrand = np.zeros_like(E_array)
    for i, E_in in enumerate(E_array):
        if E_in < E_min:
            continue
        E_out = residual_energy(E_in, shield_gcm2)
        if E_out > 0.1:  # minimum meaningful energy
            integrand[i] = spectrum[i] * bethe_bloch_water(E_out)

    # Integrate and convert
    dose_mev = np.trapezoid(integrand, E_array)
    dose_gy_per_s = dose_mev * 4 * np.pi * 1.602e-7
    dose_mgy_per_hr = dose_gy_per_s * 3600 * 1e3
    return dose_mgy_per_hr


# === Shielding Comparison ===
shield_thicknesses = [0, 1, 5, 10, 20, 50]  # [g/cm^2]
# For reference: 1 g/cm^2 Al ~ 3.7 mm Al, ISS ~20 g/cm^2
# NOTE: This simplified model overestimates absolute dose values because it
# ignores nuclear interactions (which attenuate high-energy protons by ~30-50%),
# secondary particle production, and 3D geometry effects. Real radiation
# transport codes (HZETRN, GEANT4, FLUKA) give ~10-100x lower values.
# The RELATIVE trends (spectrum shape, shielding effectiveness) are correct.

print("\n--- Dose Rate Behind Shielding [mGy/hr] ---")
print(f"{'Shield [g/cm²]':<18}", end="")
for ev in events:
    print(f"{ev['label'][:20]:<22}", end="")
print()

dose_vs_shield = {ev['label']: [] for ev in events}
for sh in shield_thicknesses:
    print(f"  {sh:<16}", end="")
    for ev in events:
        spec = sep_spectrum(E, ev['J0'], ev['gamma'], ev['E_c'])
        dr = shielded_dose_rate(E, spec, sh)
        dose_vs_shield[ev['label']].append(dr)
        print(f"{dr:<22.4f}", end="")
    print()


# =========================================================================
# 6. CUMULATIVE DOSE DURING EVENT
# =========================================================================
# Simulate a 3-day SEP event with time-varying flux
t_hours = np.linspace(0, 72, 200)

def event_time_profile(t_hr, t_onset=2, t_peak=6, t_decay=24):
    """
    SEP event time profile (Reid-Axford diffusion-like shape).

    Rapid rise to peak, followed by exponential decay.
    """
    profile = np.zeros_like(t_hr)
    mask = t_hr >= t_onset
    t_rel = t_hr[mask] - t_onset
    # Rise phase
    rise = 1 - np.exp(-t_rel / (t_peak - t_onset))
    # Decay phase
    decay = np.exp(-t_rel / t_decay)
    profile[mask] = rise * decay
    # Normalize peak to 1
    if profile.max() > 0:
        profile /= profile.max()
    return profile

time_profile = event_time_profile(t_hours)

print("\n--- Cumulative Dose During 72-hour Event (5 g/cm² shielding) ---")
for ev in events:
    spec_peak = sep_spectrum(E, ev['J0'], ev['gamma'], ev['E_c'])
    dr_peak = shielded_dose_rate(E, spec_peak, 5.0)
    # Integrate time profile * peak dose rate
    cumulative = np.trapezoid(time_profile * dr_peak, t_hours)
    # Annual limit for radiation workers: 50 mSv ~ 50 mGy (for protons, Q~1-2)
    print(f"  {ev['label']:<35}: {cumulative:.2f} mGy "
          f"(~{cumulative:.0f} mSv with Q~1)")


# =========================================================================
# 7. PLOTTING
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# --- Panel 1: Proton energy spectra ---
ax = axes[0, 0]
for i, ev in enumerate(events):
    spec = sep_spectrum(E, ev['J0'], ev['gamma'], ev['E_c'])
    ax.loglog(E, spec, color=colors_ev[i], linewidth=2, label=ev['label'])

ax.set_xlabel('Proton Kinetic Energy [MeV]')
ax.set_ylabel('dJ/dE [protons/(cm² s sr MeV)]')
ax.set_title('SEP Proton Energy Spectra')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(1, 1000)
ax.set_ylim(1e-3, 1e7)

# --- Panel 2: Stopping power and dose contribution ---
ax = axes[0, 1]
S_water = bethe_bloch_water(E)
ax.loglog(E, S_water, 'k-', linewidth=2.5)
ax.set_xlabel('Proton Kinetic Energy [MeV]')
ax.set_ylabel('Stopping Power [MeV cm²/g]')
ax.set_title('Proton Stopping Power in Water (Bethe-Bloch)')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(1, 1000)

# Mark Bragg peak region
ax.annotate('Bragg peak\n(low energy)', xy=(3, S_water[E < 4][-1]),
            xytext=(20, 300), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='red'),
            color='red')

# Inset: dose integrand (flux * S)
ax_in = ax.inset_axes([0.4, 0.5, 0.55, 0.45])
for i, ev in enumerate(events):
    spec = sep_spectrum(E, ev['J0'], ev['gamma'], ev['E_c'])
    integrand = spec * S_water
    ax_in.loglog(E, integrand, color=colors_ev[i], linewidth=1.5)
ax_in.set_xlabel('E [MeV]', fontsize=7)
ax_in.set_ylabel('dJ/dE × S', fontsize=7)
ax_in.set_title('Dose integrand', fontsize=8)
ax_in.tick_params(labelsize=6)
ax_in.set_xlim(1, 1000)
ax_in.grid(True, alpha=0.2)

# --- Panel 3: Dose vs shielding depth ---
ax = axes[0, 2]
for i, ev in enumerate(events):
    ax.semilogy(shield_thicknesses, dose_vs_shield[ev['label']],
                color=colors_ev[i], linewidth=2, marker='o', markersize=6,
                label=ev['label'])

ax.axhline(0.1, color='gray', linestyle=':', alpha=0.5)
ax.text(25, 0.12, 'Background (~0.1 mGy/hr)', fontsize=8, color='gray')
ax.set_xlabel('Shielding Thickness [g/cm²]')
ax.set_ylabel('Dose Rate [mGy/hr]')
ax.set_title('Dose Rate vs Shielding Depth')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

# Reference shielding thicknesses
for sh_ref, sh_label in [(2, 'Spacesuit'), (10, 'Spacecraft'), (20, 'ISS')]:
    ax.axvline(sh_ref, color='green', linestyle='--', alpha=0.3)
    ax.text(sh_ref + 0.5, ax.get_ylim()[1] * 0.5, sh_label,
            fontsize=7, rotation=90, color='green')

# --- Panel 4: Cumulative dose during event ---
ax = axes[1, 0]
for i, ev in enumerate(events):
    spec_peak = sep_spectrum(E, ev['J0'], ev['gamma'], ev['E_c'])
    dr_peak_0 = shielded_dose_rate(E, spec_peak, 0.0)
    dr_peak_5 = shielded_dose_rate(E, spec_peak, 5.0)
    dr_peak_20 = shielded_dose_rate(E, spec_peak, 20.0)

    cum_0 = np.cumsum(time_profile * dr_peak_0) * (t_hours[1] - t_hours[0])
    cum_5 = np.cumsum(time_profile * dr_peak_5) * (t_hours[1] - t_hours[0])
    cum_20 = np.cumsum(time_profile * dr_peak_20) * (t_hours[1] - t_hours[0])

    ax.plot(t_hours, cum_5, color=colors_ev[i], linewidth=2,
            label=f'{ev["label"][:15]} (5 g/cm²)')
    ax.plot(t_hours, cum_20, color=colors_ev[i], linewidth=1.5,
            linestyle='--', alpha=0.6)

ax.set_xlabel('Time [hours]')
ax.set_ylabel('Cumulative Dose [mGy]')
ax.set_title('Cumulative Dose During 72-hr Event')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)

# --- Panel 5: NOAA S-scale visualization ---
ax = axes[1, 1]
s_boundaries = [10, 100, 1000, 10000, 100000]
s_labels = ['S1\nMinor', 'S2\nModerate', 'S3\nStrong', 'S4\nSevere', 'S5\nExtreme']
s_colors_bg = ['#90EE90', '#FFFF00', '#FFA500', '#FF4500', '#FF0000']

for j in range(len(s_boundaries)):
    lower = s_boundaries[j]
    upper = s_boundaries[j + 1] if j < len(s_boundaries) - 1 else 1e6
    ax.axhspan(lower, upper, alpha=0.2, color=s_colors_bg[j])
    ax.text(0.02, np.sqrt(lower * upper), s_labels[j],
            transform=ax.get_yaxis_transform(), fontsize=9, fontweight='bold',
            va='center')

# Plot event fluxes
for i, ev in enumerate(events):
    flux_time = time_profile * ev['f10']
    ax.semilogy(t_hours, flux_time + 1, color=colors_ev[i], linewidth=2,
                label=ev['label'])

ax.set_xlabel('Time [hours]')
ax.set_ylabel('>10 MeV Proton Flux [pfu]')
ax.set_title('SEP Event Profiles & NOAA S-Scale')
ax.set_ylim(1, 2e5)
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

# --- Panel 6: Event time profile ---
ax = axes[1, 2]
ax.plot(t_hours, time_profile, 'k-', linewidth=2.5, label='Normalized flux profile')
ax.fill_between(t_hours, time_profile, alpha=0.2, color='orange')

# Mark phases
ax.axvline(2, color='red', linestyle='--', alpha=0.5, label='Onset')
t_peak_idx = np.argmax(time_profile)
ax.axvline(t_hours[t_peak_idx], color='green', linestyle='--', alpha=0.5,
           label=f'Peak (t={t_hours[t_peak_idx]:.0f} hr)')
ax.annotate('Rapid rise\n(shock acceleration)',
            xy=(4, 0.5), fontsize=9, color='red',
            arrowprops=dict(arrowstyle='->', color='red'),
            xytext=(15, 0.7))
ax.annotate('Exponential decay\n(diffusive transport)',
            xy=(30, 0.3), fontsize=9, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue'),
            xytext=(40, 0.6))

ax.set_xlabel('Time [hours]')
ax.set_ylabel('Normalized Flux')
ax.set_title('SEP Event Time Profile')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 72)

plt.tight_layout()
plt.savefig('/opt/projects/01_Personal/03_Study/examples/Space_Weather/09_sep_dose.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nKey insights:")
print("  - SEP proton spectra span 5+ orders of magnitude in flux")
print("  - The Bragg peak means low-energy protons deposit MORE energy per unit path")
print("  - 10-20 g/cm² shielding reduces dose by ~10-100x (blocks <100 MeV protons)")
print("  - Extreme events (S5) can deliver lethal doses to unshielded astronauts in hours")
print("  - EVA during SEP events is extremely hazardous; storm shelters are essential")
print("\nPlot saved to 09_sep_dose.png")
