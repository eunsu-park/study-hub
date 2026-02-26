"""
Magnetospheric Substorms: AL Index Model, Current Wedge, and Onset Detection.

Simulates the substorm cycle (growth -> onset -> expansion -> recovery) using
a synthetic AL index and models the substorm current wedge (SCW). Includes
an onset detection algorithm based on the sharp decrease in AL.

Key physics:
  - Substorm cycle:
    * Growth phase (~30-60 min): energy stored in tail lobe
    * Onset: abrupt current disruption in near-Earth tail
    * Expansion phase (~15-30 min): AL drops sharply, aurora expands poleward
    * Recovery phase (~1-2 h): currents return to quiet configuration
  - AL index: lower envelope of H-component at auroral magnetometers
    Reflects westward electrojet intensity
  - Substorm Current Wedge (SCW):
    * Tail current disrupts -> current diverts through ionosphere
    * Downward FAC on dawn side, upward FAC on dusk side
    * Westward electrojet connects them in the ionosphere
  - Onset detection: identify dAL/dt exceeding a threshold
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

print("=" * 65)
print("Magnetospheric Substorm Model")
print("=" * 65)


# =========================================================================
# 1. SUBSTORM WAVEFORM TEMPLATE
# =========================================================================
# Each substorm has characteristic phases with specific AL signatures:
#   Growth: gradual decrease (energy loading, ~1 nT/min decrease)
#   Onset: sudden drop (current disruption)
#   Expansion: rapid deepening of AL (peak electrojet)
#   Recovery: exponential return to baseline

def substorm_waveform(t_minutes, onset_time, peak_AL, growth_duration=40,
                      expansion_duration=20, recovery_tau=40):
    """Generate a single substorm AL waveform.

    Parameters
    ----------
    t_minutes : array
        Time array [minutes].
    onset_time : float
        Time of substorm onset [minutes].
    peak_AL : float
        Peak (most negative) AL value [nT].
    growth_duration : float
        Duration of growth phase before onset [minutes].
    expansion_duration : float
        Duration of expansion phase [minutes].
    recovery_tau : float
        e-folding time of recovery [minutes].

    Returns
    -------
    AL : array
        AL index contribution from this substorm [nT].
    """
    AL = np.zeros_like(t_minutes, dtype=float)
    growth_start = onset_time - growth_duration

    for i, ti in enumerate(t_minutes):
        if ti < growth_start:
            # Before substorm: quiet
            AL[i] = 0
        elif ti < onset_time:
            # Growth phase: gradual loading (linear ramp of ~20% of peak)
            phase = (ti - growth_start) / growth_duration
            AL[i] = 0.2 * peak_AL * phase
        elif ti < onset_time + expansion_duration:
            # Expansion phase: rapid deepening
            phase = (ti - onset_time) / expansion_duration
            # Use a fast sigmoid-like drop
            AL[i] = 0.2 * peak_AL + 0.8 * peak_AL * (1 - np.exp(-3 * phase))
        else:
            # Recovery phase: exponential decay back to zero
            expansion_end_val = 0.2 * peak_AL + 0.8 * peak_AL * (1 - np.exp(-3))
            AL[i] = expansion_end_val * np.exp(-(ti - onset_time - expansion_duration) / recovery_tau)

    return AL


# =========================================================================
# 2. GENERATE SYNTHETIC MAGNETOMETER DATA (6 hours, multiple substorms)
# =========================================================================
np.random.seed(77)
dt_min = 1.0  # 1-minute cadence (typical for geomagnetic indices)
t_min = np.arange(0, 360, dt_min)  # 6 hours
N = len(t_min)

# Define 3 substorms during the interval
substorms = [
    {'onset': 60, 'peak_AL': -500, 'growth': 35, 'expansion': 15, 'recovery_tau': 45},
    {'onset': 170, 'peak_AL': -800, 'growth': 50, 'expansion': 25, 'recovery_tau': 50},
    {'onset': 280, 'peak_AL': -350, 'growth': 30, 'expansion': 12, 'recovery_tau': 35},
]

# Build composite AL
AL_total = np.zeros(N)
for ss in substorms:
    AL_single = substorm_waveform(
        t_min, ss['onset'], ss['peak_AL'],
        ss['growth'], ss['expansion'], ss['recovery_tau']
    )
    AL_total += AL_single

# Add quiet-time baseline and noise
AL_baseline = -20  # quiet-time AL [nT]
noise = 15 * np.random.randn(N)
AL_noisy = AL_total + AL_baseline + noise

# Smooth slightly to simulate real 1-min indices
AL_smooth = uniform_filter1d(AL_noisy, size=3)

# Also create AU index (simplified: roughly mirrors AL but positive and weaker)
AU_total = -0.4 * AL_total + 10 + 8 * np.random.randn(N)
AU_smooth = uniform_filter1d(AU_total, size=3)

print(f"\nSynthetic substorm data (6 hours, {N} points):")
for i, ss in enumerate(substorms):
    print(f"  Substorm {i+1}: onset={ss['onset']} min, peak AL={ss['peak_AL']} nT")
print(f"  Overall min AL = {AL_smooth.min():.0f} nT")


# =========================================================================
# 3. ONSET DETECTION ALGORITHM
# =========================================================================
# A substorm onset is identified by a sharp negative change in AL.
# Algorithm:
#   1. Compute dAL/dt (first derivative)
#   2. Smooth the derivative to avoid noise triggers
#   3. Flag times where dAL/dt < threshold (e.g., -15 nT/min)
#   4. Cluster flagged points and take the first in each cluster

def detect_substorm_onsets(AL, dt, threshold=-15, min_separation=30,
                           smooth_window=5):
    """Detect substorm onsets from AL index time series.

    Parameters
    ----------
    AL : array
        AL index [nT].
    dt : float
        Time step [minutes].
    threshold : float
        dAL/dt threshold for onset detection [nT/min].
    min_separation : float
        Minimum separation between detected onsets [minutes].
    smooth_window : int
        Window for smoothing the derivative.

    Returns
    -------
    onset_indices : list of int
        Indices of detected onsets.
    dAL_dt : array
        Smoothed time derivative of AL.
    """
    # Compute derivative
    dAL_dt = np.gradient(AL, dt)

    # Smooth to reduce noise
    dAL_dt_smooth = uniform_filter1d(dAL_dt, size=smooth_window)

    # Find points below threshold
    candidates = np.where(dAL_dt_smooth < threshold)[0]

    if len(candidates) == 0:
        return [], dAL_dt_smooth

    # Cluster and take first point in each cluster
    onsets = [candidates[0]]
    for idx in candidates[1:]:
        if (idx - onsets[-1]) * dt > min_separation:
            onsets.append(idx)

    return onsets, dAL_dt_smooth


onset_indices, dAL_dt = detect_substorm_onsets(
    AL_smooth, dt_min, threshold=-12, min_separation=40, smooth_window=5
)

print(f"\nOnset Detection (threshold = -12 nT/min):")
print(f"  Number of onsets detected: {len(onset_indices)}")
for i, idx in enumerate(onset_indices):
    print(f"  Onset {i+1}: t = {t_min[idx]:.0f} min, AL = {AL_smooth[idx]:.0f} nT, "
          f"dAL/dt = {dAL_dt[idx]:.1f} nT/min")
print(f"  True onsets: {[ss['onset'] for ss in substorms]}")


# =========================================================================
# 4. SUBSTORM CURRENT WEDGE (SCW) MODEL
# =========================================================================
# The SCW consists of:
#   - Downward FAC on the dawn side (eastward of onset meridian)
#   - Upward FAC on the dusk side (westward of onset meridian)
#   - Westward electrojet connecting them at ionospheric altitude
#
# Model: J_FAC(phi, mlat) at ionospheric altitude
# phi = magnetic local time angle, mlat = magnetic latitude
# Simplified as two Gaussian current spots in the auroral zone

phi_mlt = np.linspace(0, 2 * np.pi, 360)  # 0 = midnight, pi/2 = dawn, etc.
mlat = np.linspace(60, 80, 100)  # magnetic latitude [deg]

PHI, MLAT = np.meshgrid(phi_mlt, mlat)

# SCW parameters (centered around midnight sector)
phi_center = 0       # midnight meridian [rad]
phi_width = 0.7      # half-width in MLT [rad] (~2.7 hours of MLT)
mlat_center = 67.0   # auroral zone latitude [deg]
mlat_width = 3.0     # latitude width [deg]
J_max = 2.0          # peak FAC density [uA/m^2]

# Upward FAC (dusk side of midnight, phi < 0 or phi > 2*pi - offset)
# Downward FAC (dawn side of midnight, phi > 0)
# Use cos(phi) pattern centered on midnight
J_FAC = J_max * np.sin(PHI - phi_center) * \
        np.exp(-0.5 * ((MLAT - mlat_center) / mlat_width)**2)

# Westward electrojet: flows along latitude circle in between the FAC spots
J_WEJ = -J_max * 0.5 * np.exp(-0.5 * ((MLAT - mlat_center) / mlat_width)**2) * \
         (np.abs(PHI - np.pi) < np.pi/2).astype(float)  # nightside only

print(f"\nSubstorm Current Wedge Model:")
print(f"  Peak FAC density = {J_max} uA/m^2")
print(f"  Auroral zone center = {mlat_center} deg MLAT")
print(f"  MLT half-width = {np.degrees(phi_width):.1f} deg ({phi_width*12/np.pi:.1f} h MLT)")


# =========================================================================
# 5. PLOTTING
# =========================================================================
fig = plt.figure(figsize=(16, 14))
fig.suptitle("Magnetospheric Substorm Analysis", fontsize=15, y=0.98)

# --- Panel 1: AL and AU indices with onset markers ---
ax1 = fig.add_subplot(3, 2, (1, 2))

ax1.plot(t_min / 60, AL_smooth, 'b-', lw=1.5, label='AL [nT]')
ax1.plot(t_min / 60, AU_smooth, 'r-', lw=1, alpha=0.7, label='AU [nT]')
ax1.fill_between(t_min / 60, AL_smooth, 0, alpha=0.1, color='blue')

# Mark detected onsets
for i, idx in enumerate(onset_indices):
    ax1.axvline(t_min[idx] / 60, color='red', ls='--', lw=1.5, alpha=0.7)
    ax1.annotate(f'Onset {i+1}\n({t_min[idx]:.0f} min)',
                 xy=(t_min[idx] / 60, AL_smooth[idx]),
                 xytext=(t_min[idx] / 60 + 0.15, AL_smooth[idx] + 80),
                 fontsize=8, ha='left',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1))

# Mark true onsets
for ss in substorms:
    ax1.axvline(ss['onset'] / 60, color='green', ls=':', lw=1, alpha=0.5)

# Shade substorm phases for first substorm as example
ss0 = substorms[0]
ax1.axvspan((ss0['onset'] - ss0['growth']) / 60, ss0['onset'] / 60,
            color='yellow', alpha=0.15, label='Growth phase')
ax1.axvspan(ss0['onset'] / 60, (ss0['onset'] + ss0['expansion']) / 60,
            color='red', alpha=0.15, label='Expansion phase')
ax1.axvspan((ss0['onset'] + ss0['expansion']) / 60,
            (ss0['onset'] + ss0['expansion'] + 2 * ss0['recovery_tau']) / 60,
            color='green', alpha=0.1, label='Recovery phase')

ax1.set_xlabel("Time [hours]", fontsize=11)
ax1.set_ylabel("Index [nT]", fontsize=11)
ax1.set_title("AL/AU Indices with Substorm Onset Detection", fontsize=12)
ax1.legend(fontsize=8, loc='lower left', ncol=2)
ax1.grid(True, alpha=0.3)

# --- Panel 2: dAL/dt with threshold ---
ax2 = fig.add_subplot(3, 2, (3, 4))
ax2.plot(t_min / 60, dAL_dt, 'b-', lw=1, label='dAL/dt [nT/min]')
ax2.axhline(-12, color='red', ls='--', lw=1.5, label='Onset threshold (-12 nT/min)')
ax2.axhline(0, color='gray', ls=':', lw=0.5)

for idx in onset_indices:
    ax2.plot(t_min[idx] / 60, dAL_dt[idx], 'rv', markersize=10, zorder=5)

ax2.set_xlabel("Time [hours]", fontsize=11)
ax2.set_ylabel("dAL/dt [nT/min]", fontsize=11)
ax2.set_title("AL Time Derivative (Onset Detection)", fontsize=12)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-40, 20)

# --- Panel 3: SCW field-aligned currents (polar view) ---
ax3 = fig.add_subplot(3, 2, 5, projection='polar')

# Plot FAC pattern
c = ax3.pcolormesh(PHI, 90 - MLAT, J_FAC, cmap='bwr', vmin=-J_max, vmax=J_max,
                   shading='auto')
plt.colorbar(c, ax=ax3, label=r'$J_\parallel$ [$\mu$A/m$^2$]', shrink=0.8, pad=0.1)

ax3.set_theta_zero_location('S')
ax3.set_theta_direction(-1)
ax3.set_ylim(10, 30)
ax3.set_yticks([15, 20, 25])
ax3.set_yticklabels(['75', '70', '65'])
ax3.set_xticks(np.array([0, 6, 12, 18]) * np.pi / 12)
ax3.set_xticklabels(['00\n(mid)', '06\n(dawn)', '12\n(noon)', '18\n(dusk)'], fontsize=8)
ax3.set_title("Substorm Current Wedge\nFAC Pattern (Red=up, Blue=down)", fontsize=11, pad=15)

# --- Panel 4: Individual substorm waveforms ---
ax4 = fig.add_subplot(3, 2, 6)

# Plot each substorm component separately
colors = ['royalblue', 'darkred', 'forestgreen']
for i, ss in enumerate(substorms):
    AL_single = substorm_waveform(
        t_min, ss['onset'], ss['peak_AL'],
        ss['growth'], ss['expansion'], ss['recovery_tau']
    )
    ax4.plot(t_min / 60, AL_single, color=colors[i], lw=1.5,
             label=f"SS{i+1} (AL={ss['peak_AL']} nT)", alpha=0.8)

ax4.plot(t_min / 60, AL_total, 'k-', lw=2, label='Superposition')

ax4.set_xlabel("Time [hours]", fontsize=11)
ax4.set_ylabel("AL contribution [nT]", fontsize=11)
ax4.set_title("Individual Substorm Waveforms", fontsize=12)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/opt/projects/01_Personal/03_Study/examples/Space_Weather/05_substorm.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: 05_substorm.png")
