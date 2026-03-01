"""
Space Weather Geomagnetic Indices: Dst, Kp, AE, and F10.7.

Demonstrates:
- Synthetic magnetometer H-component data for multiple stations
- Dst index computation: pressure-corrected ΔH averaged over low-latitude stations
- Kp-like index: quasi-logarithmic 3-hour range index (0-9 scale)
- AE index: AU (upper) and AL (lower) envelopes from auroral stations
- F10.7 solar radio flux proxy with 27-day solar rotation periodicity
- NOAA G-scale classification from Kp values

Physics:
    Geomagnetic indices distill global magnetic field variations into single
    numbers that characterize space weather conditions:

    Dst (Disturbance Storm Time):
        Measures symmetric ring current intensity. Computed from horizontal
        component perturbations (ΔH) at 4-6 low-latitude stations:
            Dst = <ΔH / cos(λ)>_stations
        where λ is magnetic latitude. Dst ~ -100 nT = moderate storm,
        Dst ~ -300 nT = superstorm.

    Kp (Planetary K-index):
        3-hour quasi-logarithmic index (0 to 9) derived from the range
        (max - min) of horizontal magnetic field at mid-latitude stations.
        Uses a quasi-logarithmic scale to account for latitude-dependent
        sensitivity. Kp ≥ 5 = geomagnetic storm.

    AE (Auroral Electrojet):
        Derived from ~12 auroral zone stations. AU = maximum positive
        perturbation (eastward electrojet), AL = minimum negative perturbation
        (westward electrojet). AE = AU - AL. Substorms: AL < -500 nT.

    F10.7 (Solar Radio Flux):
        The 10.7 cm (2800 MHz) solar radio flux is a proxy for solar EUV
        output and correlates with solar activity. Shows 27-day periodicity
        from solar rotation. Range: ~65 (minimum) to ~300 (maximum) SFU.

    NOAA G-scale (G1-G5) classifies geomagnetic storms by Kp:
        G1: Kp=5, G2: Kp=6, G3: Kp=7, G4: Kp=8, G5: Kp=9

References:
    - Sugiura, M. (1964). "Hourly values of equatorial Dst for the IGY."
    - Bartels, J. et al. (1939). "The three-hour-range index measuring
      geomagnetic activity." (Kp index)
    - Davis, T.N. & Sugiura, M. (1966). "Auroral electrojet activity index AE."
    - Tapping, K.F. (2013). "The 10.7 cm solar radio flux (F10.7)."
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 65)
print("SPACE WEATHER GEOMAGNETIC INDICES")
print("=" * 65)


# =========================================================================
# 1. GENERATE SYNTHETIC MAGNETOMETER DATA
# =========================================================================
np.random.seed(42)
dt_min = 1.0  # time resolution [minutes]
T_hours = 72  # total duration [hours]
t_min = np.arange(0, T_hours * 60, dt_min)
t_hr = t_min / 60.0
N_t = len(t_min)

# --- Low-latitude stations for Dst (4 stations, different longitudes) ---
dst_stations = {
    'Honolulu':   {'lat': 21.3,  'mlon': 266},
    'San Juan':   {'lat': 18.1,  'mlon': 3},
    'Hermanus':   {'lat': -34.4, 'mlon': 27},
    'Kakioka':    {'lat': 36.2,  'mlon': 208},
}

# --- Auroral stations for AE (5 stations, ~65-70 deg magnetic latitude) ---
ae_stations = {
    'Abisko':     {'mlat': 65.3},
    'Leirvogur':  {'mlat': 65.0},
    'Narssarssuaq': {'mlat': 66.5},
    'Fort Churchill': {'mlat': 68.7},
    'College':    {'mlat': 64.9},
}


def generate_storm_perturbation(t_hr, storm_onset=10, main_phase_dur=12,
                                 dst_min=-150, recovery_tau=15):
    """
    Generate synthetic ring current perturbation (Dst-like).

    Models a geomagnetic storm with:
    - Quiet period, sudden commencement, main phase, recovery
    """
    dH = np.zeros_like(t_hr)

    # Sudden commencement: positive spike from magnetopause compression
    sc_mask = (t_hr >= storm_onset) & (t_hr < storm_onset + 0.5)
    dH[sc_mask] = 30 * np.sin(np.pi * (t_hr[sc_mask] - storm_onset) / 0.5)

    # Main phase: exponential decrease to dst_min
    main_mask = t_hr >= storm_onset + 0.5
    t_rel = t_hr[main_mask] - (storm_onset + 0.5)

    # Injection (exponential approach to minimum)
    injection = dst_min * (1 - np.exp(-t_rel / (main_phase_dur / 3)))

    # Recovery (exponential decay back to 0)
    recovery = np.exp(-np.maximum(t_rel - main_phase_dur, 0) / recovery_tau)

    # Combine: main phase injection then recovery
    main_recovery = np.where(t_rel < main_phase_dur,
                             injection,
                             dst_min * recovery)
    dH[main_mask] = main_recovery

    return dH


def generate_substorm_perturbation(t_hr, onset_times, amplitudes):
    """
    Generate substorm perturbation for auroral stations.

    Each substorm: rapid negative bay (~30 min onset, ~2 hr recovery)
    """
    dH = np.zeros_like(t_hr)
    for t_onset, amp in zip(onset_times, amplitudes):
        mask = t_hr >= t_onset
        t_rel = t_hr[mask] - t_onset
        # Rapid onset (30 min)
        onset = -amp * (1 - np.exp(-t_rel / 0.5))
        # Recovery (~2 hours)
        decay = np.exp(-np.maximum(t_rel - 0.5, 0) / 2.0)
        dH[mask] += onset * decay
    return dH


# --- Generate station data ---
# Ring current perturbation (same for all Dst stations, with cos(lat) modulation)
ring_current = generate_storm_perturbation(t_hr, storm_onset=10,
                                            main_phase_dur=12, dst_min=-150)

print("\n--- Generating Synthetic Magnetometer Data ---")
print(f"  Duration: {T_hours} hours, resolution: {dt_min} min")
print(f"  Storm onset: t = 10 hr, Dst minimum: ~-150 nT")

# Dst station H-component data
dst_data = {}
for name, stn in dst_stations.items():
    lat_rad = np.radians(stn['lat'])
    # H perturbation: ring current * cos(lat) + noise
    noise = np.random.normal(0, 3, N_t)
    dH = ring_current * np.cos(lat_rad) + noise
    # Add diurnal Sq variation (~20 nT amplitude)
    sq = 20 * np.sin(2 * np.pi * (t_hr - 6 + stn['mlon'] / 15) / 24)
    dst_data[name] = dH + sq
    print(f"  Dst station {name}: lat={stn['lat']:.1f}°, "
          f"peak ΔH = {np.min(dH):.0f} nT")

# AE station H-component data
ae_data = {}
# Substorms during main phase and recovery
substorm_onsets = [12, 15, 18, 22, 30]
substorm_amps = [800, 1200, 600, 400, 300]

for name, stn in ae_stations.items():
    # Substorm perturbation + noise
    noise = np.random.normal(0, 15, N_t)
    # Each station sees substorms slightly differently
    phase_shift = np.random.uniform(-0.3, 0.3)
    amp_scale = np.random.uniform(0.7, 1.3)
    shifted_onsets = [t + phase_shift for t in substorm_onsets]
    scaled_amps = [a * amp_scale for a in substorm_amps]
    dH = generate_substorm_perturbation(t_hr, shifted_onsets, scaled_amps) + noise
    ae_data[name] = dH

print(f"  AE stations: {len(ae_stations)} auroral zone stations")
print(f"  Substorm onsets at t = {substorm_onsets} hr")


# =========================================================================
# 2. COMPUTE Dst INDEX
# =========================================================================
def compute_dst(dst_data, dst_stations, t_hr):
    """
    Compute Dst-like index from low-latitude station data.

    Dst(t) = <ΔH(t) / cos(λ)>_stations

    In practice, quiet-day baseline is subtracted first and pressure
    correction applied. Here we use simplified version.
    """
    N_stations = len(dst_stations)
    dst_sum = np.zeros_like(t_hr)

    for name, stn in dst_stations.items():
        lat_rad = np.radians(stn['lat'])
        cos_lat = np.cos(lat_rad)
        # Normalize by cos(latitude) to correct for latitude effect
        dst_sum += dst_data[name] / cos_lat

    dst = dst_sum / N_stations
    return dst


dst_index = compute_dst(dst_data, dst_stations, t_hr)

print("\n--- Dst Index ---")
print(f"  Minimum Dst = {np.min(dst_index):.0f} nT "
      f"at t = {t_hr[np.argmin(dst_index)]:.1f} hr")
print(f"  Storm classification:")
print(f"    Moderate: -50 > Dst > -100 nT")
print(f"    Intense:  -100 > Dst > -250 nT")
print(f"    Superstorm: Dst < -250 nT")

if np.min(dst_index) > -100:
    print(f"  This event: Moderate storm")
elif np.min(dst_index) > -250:
    print(f"  This event: Intense storm")
else:
    print(f"  This event: Superstorm")


# =========================================================================
# 3. COMPUTE Kp INDEX
# =========================================================================
def compute_kp(dst_data, t_hr):
    """
    Compute Kp-like index from 3-hour range in H-component.

    For each 3-hour interval:
    1. Compute range (max - min) of ΔH at each station
    2. Convert range to K value using quasi-logarithmic scale
    3. Average K values -> Kp

    The K scale is quasi-logarithmic:
        K=0: range < 5 nT, K=1: 5-10, K=2: 10-20, K=3: 20-40,
        K=4: 40-70, K=5: 70-120, K=6: 120-200, K=7: 200-330,
        K=8: 330-500, K=9: > 500 nT
    """
    k_thresholds = [5, 10, 20, 40, 70, 120, 200, 330, 500]

    def range_to_k(r):
        for k, thresh in enumerate(k_thresholds):
            if r < thresh:
                return k
        return 9

    # 3-hour intervals
    interval_hours = 3
    n_intervals = int(np.ceil(t_hr[-1] / interval_hours))
    kp_values = []
    kp_times = []

    for i_int in range(n_intervals):
        t_start = i_int * interval_hours
        t_end = (i_int + 1) * interval_hours
        mask = (t_hr >= t_start) & (t_hr < t_end)
        if mask.sum() < 2:
            continue

        k_stations = []
        for name in dst_data:
            h_seg = dst_data[name][mask]
            h_range = np.max(h_seg) - np.min(h_seg)
            k_stations.append(range_to_k(h_range))

        kp = np.mean(k_stations)
        kp_values.append(kp)
        kp_times.append(t_start + interval_hours / 2)

    return np.array(kp_times), np.array(kp_values)


kp_times, kp_values = compute_kp(dst_data, t_hr)

print("\n--- Kp Index (3-hour intervals) ---")
print(f"  Maximum Kp = {kp_values.max():.0f} "
      f"at t = {kp_times[np.argmax(kp_values)]:.1f} hr")


# === NOAA G-Scale ===
def noaa_g_scale(kp):
    """NOAA G-scale from Kp value."""
    if kp >= 9:
        return 'G5 (Extreme)'
    elif kp >= 8:
        return 'G4 (Severe)'
    elif kp >= 7:
        return 'G3 (Strong)'
    elif kp >= 6:
        return 'G2 (Moderate)'
    elif kp >= 5:
        return 'G1 (Minor)'
    else:
        return 'Below G1'


max_kp = kp_values.max()
print(f"  NOAA G-scale: {noaa_g_scale(max_kp)}")


# =========================================================================
# 4. COMPUTE AE INDEX
# =========================================================================
def compute_ae(ae_data, t_hr):
    """
    Compute AE-like indices from auroral station data.

    AU = max(ΔH) across all auroral stations (eastward electrojet)
    AL = min(ΔH) across all auroral stations (westward electrojet)
    AE = AU - AL (total electrojet activity)
    AO = (AU + AL) / 2 (asymmetry)
    """
    all_data = np.array([ae_data[name] for name in ae_data])
    AU = np.max(all_data, axis=0)
    AL = np.min(all_data, axis=0)
    AE = AU - AL
    AO = (AU + AL) / 2
    return AU, AL, AE, AO


AU, AL, AE, AO = compute_ae(ae_data, t_hr)

print("\n--- AE Index ---")
print(f"  Peak AU = {np.max(AU):.0f} nT (eastward electrojet)")
print(f"  Peak |AL| = {np.min(AL):.0f} nT (westward electrojet)")
print(f"  Peak AE = {np.max(AE):.0f} nT (total electrojet)")
print(f"  Substorm threshold: |AL| > 500 nT or AE > 1000 nT")


# =========================================================================
# 5. F10.7 SOLAR RADIO FLUX
# =========================================================================
def generate_f107(t_days, base=120, amplitude=30, rotation_period=27):
    """
    Generate synthetic F10.7 time series.

    Shows 27-day periodicity from solar rotation, plus long-term
    solar cycle modulation and short-term variability.
    """
    # 27-day rotation
    f107 = base + amplitude * np.sin(2 * np.pi * t_days / rotation_period)
    # Add shorter period variations (active regions appearing/disappearing)
    f107 += 15 * np.sin(2 * np.pi * t_days / 13.5 + 1.2)
    # Random daily variations
    np.random.seed(123)
    f107 += np.random.normal(0, 5, len(t_days))
    return np.maximum(f107, 65)  # F10.7 never below ~65 SFU


# Generate 90 days of F10.7 for context
t_days = np.arange(0, 90, 1)
f107 = generate_f107(t_days)

print("\n--- F10.7 Solar Radio Flux ---")
print(f"  Mean F10.7 = {np.mean(f107):.0f} SFU")
print(f"  Range: {np.min(f107):.0f} - {np.max(f107):.0f} SFU")
print(f"  27-day periodicity from solar rotation")
print(f"  Classification: <70 = deep minimum, 70-120 = low, "
      f"120-180 = moderate, >180 = high")


# =========================================================================
# 6. PLOTTING
# =========================================================================
fig = plt.figure(figsize=(18, 14))

# --- Panel 1: Raw magnetometer traces (Dst stations) ---
ax1 = fig.add_subplot(3, 2, 1)
colors_dst = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, (name, dH) in enumerate(dst_data.items()):
    ax1.plot(t_hr, dH, color=colors_dst[i], linewidth=0.8, alpha=0.8,
             label=name)

ax1.set_xlabel('Time [hours]')
ax1.set_ylabel('ΔH [nT]')
ax1.set_title('Magnetometer H-Component (Dst Stations)')
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.axvline(10, color='red', linestyle=':', alpha=0.5, label='Storm onset')

# --- Panel 2: Dst index ---
ax2 = fig.add_subplot(3, 2, 2)
ax2.plot(t_hr, dst_index, 'k-', linewidth=2)
ax2.fill_between(t_hr, dst_index, 0, where=dst_index < 0,
                 alpha=0.3, color='red')
ax2.fill_between(t_hr, dst_index, 0, where=dst_index > 0,
                 alpha=0.3, color='blue')

# Storm intensity bands
ax2.axhspan(-100, -50, alpha=0.1, color='yellow')
ax2.axhspan(-250, -100, alpha=0.1, color='orange')
ax2.axhspan(-500, -250, alpha=0.1, color='red')
ax2.text(0.5, -75, 'Moderate', fontsize=8, color='goldenrod')
ax2.text(0.5, -175, 'Intense', fontsize=8, color='darkorange')

ax2.set_xlabel('Time [hours]')
ax2.set_ylabel('Dst [nT]')
ax2.set_title('Dst Index')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, T_hours)

# --- Panel 3: Kp bar chart ---
ax3 = fig.add_subplot(3, 2, 3)
# Color bars by Kp value
kp_colors = []
for kp in kp_values:
    if kp >= 7:
        kp_colors.append('red')
    elif kp >= 5:
        kp_colors.append('orange')
    elif kp >= 4:
        kp_colors.append('yellow')
    else:
        kp_colors.append('green')

ax3.bar(kp_times, kp_values, width=2.5, color=kp_colors, edgecolor='black',
        linewidth=0.5)
ax3.axhline(5, color='orange', linestyle='--', linewidth=1.5, label='G1 threshold')
ax3.axhline(7, color='red', linestyle='--', linewidth=1.5, label='G3 threshold')

ax3.set_xlabel('Time [hours]')
ax3.set_ylabel('Kp')
ax3.set_title('Kp Index (3-hour intervals)')
ax3.set_ylim(0, 9.5)
ax3.set_yticks(range(10))
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xlim(0, T_hours)

# Add G-scale labels on right
ax3_right = ax3.twinx()
ax3_right.set_ylim(0, 9.5)
ax3_right.set_yticks([5, 6, 7, 8, 9])
ax3_right.set_yticklabels(['G1', 'G2', 'G3', 'G4', 'G5'], fontsize=8)

# --- Panel 4: AE index (AU/AL envelope) ---
ax4 = fig.add_subplot(3, 2, 4)
ax4.plot(t_hr, AU, 'r-', linewidth=1.5, label='AU (eastward EJ)')
ax4.plot(t_hr, AL, 'b-', linewidth=1.5, label='AL (westward EJ)')
ax4.fill_between(t_hr, AU, AL, alpha=0.15, color='purple')
ax4.plot(t_hr, AE, 'k--', linewidth=1, alpha=0.5, label='AE = AU - AL')

# Mark substorm onsets
for t_sub in substorm_onsets:
    ax4.axvline(t_sub, color='green', linestyle=':', alpha=0.4)

ax4.set_xlabel('Time [hours]')
ax4.set_ylabel('Index [nT]')
ax4.set_title('Auroral Electrojet (AE) Indices')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, T_hours)

# --- Panel 5: F10.7 ---
ax5 = fig.add_subplot(3, 2, 5)
ax5.plot(t_days, f107, 'b-', linewidth=1.5)
ax5.axhline(np.mean(f107), color='red', linestyle='--', alpha=0.5,
            label=f'Mean = {np.mean(f107):.0f} SFU')

# Mark 27-day period
for i in range(3):
    ax5.axvline(i * 27, color='green', linestyle=':', alpha=0.3)
ax5.text(13, np.max(f107) + 5, '27-day rotation', fontsize=8, color='green',
         ha='center')

ax5.set_xlabel('Time [days]')
ax5.set_ylabel('F10.7 [SFU]')
ax5.set_title('F10.7 Solar Radio Flux (90-day window)')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# --- Panel 6: Summary dashboard ---
ax6 = fig.add_subplot(3, 2, 6)
ax6.axis('off')

summary_text = (
    "SPACE WEATHER SUMMARY\n"
    "=" * 40 + "\n\n"
    f"Storm Period: 0-{T_hours} hours\n\n"
    f"Dst:  min = {np.min(dst_index):.0f} nT\n"
    f"      Classification: {'Intense' if np.min(dst_index) < -100 else 'Moderate'} storm\n\n"
    f"Kp:   max = {kp_values.max():.0f}\n"
    f"      NOAA: {noaa_g_scale(kp_values.max())}\n\n"
    f"AE:   max = {np.max(AE):.0f} nT\n"
    f"      Substorms: {len(substorm_onsets)} events\n\n"
    f"F10.7: mean = {np.mean(f107):.0f} SFU\n"
    f"       Solar activity: {'Moderate' if np.mean(f107) < 180 else 'High'}\n\n"
    "NOAA G-Scale Reference:\n"
    "  G1 (Kp=5): Minor\n"
    "  G2 (Kp=6): Moderate\n"
    "  G3 (Kp=7): Strong\n"
    "  G4 (Kp=8): Severe\n"
    "  G5 (Kp=9): Extreme"
)

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=10, fontfamily='monospace', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('/opt/projects/01_Personal/03_Study/examples/Space_Weather/11_space_weather_indices.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nKey insights:")
print("  - Dst captures the ring current (global storm intensity)")
print("  - Kp measures mid-latitude disturbance on a quasi-log scale")
print("  - AE tracks auroral electrojet activity (substorms)")
print("  - F10.7 is a proxy for solar EUV driving the ionosphere/thermosphere")
print("  - These indices serve different purposes: Dst for storms, Kp for overall")
print("    activity, AE for substorms, F10.7 for solar forcing")
print("  - Real indices use carefully selected station networks and baselines")
print("\nPlot saved to 11_space_weather_indices.png")
