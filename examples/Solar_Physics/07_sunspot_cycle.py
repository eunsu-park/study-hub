"""
Solar Cycle Analysis: Sunspot Number, FFT Periodicity, and Waldmeier Effect.

Demonstrates:
- Synthetic sunspot number time series spanning ~300 years
- Waldmeier effect: shorter rise time for stronger cycles
- Maunder Minimum: period of anomalously low solar activity (~1645-1715)
- FFT power spectrum analysis to extract the ~11-year periodicity
- Simplified butterfly diagram showing latitude migration of sunspots
- Correlation between cycle rise time and amplitude

Physics:
    The solar magnetic activity cycle has a mean period of ~11 years (Schwabe cycle).
    Sunspot number is the primary observational proxy for this cycle. The Waldmeier
    effect states that stronger cycles rise faster to maximum, i.e., rise time is
    anti-correlated with peak amplitude. Sporer's law describes the equatorward
    migration of sunspot emergence latitudes over each cycle.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Parameters ---
dt = 1.0 / 12  # monthly resolution (years)
T_total = 300.0  # total span in years
t = np.arange(0, T_total, dt)
N = len(t)

# --- Generate synthetic sunspot cycles ---
# Each cycle has a period, amplitude, and asymmetric profile (fast rise, slow decay)
cycle_amplitudes = []
cycle_rise_times = []
cycle_starts = []

ssn = np.zeros(N)  # sunspot number array
current_t = 0.0
cycle_num = 0

while current_t < T_total:
    # Random cycle period: 9-14 years, centered on 11
    period = np.random.normal(11.0, 1.2)
    period = np.clip(period, 9.0, 14.0)

    # Random amplitude: 50-200
    amplitude = np.random.uniform(50, 200)

    # Waldmeier effect: rise time inversely related to amplitude
    # Stronger cycles rise faster (rise fraction 0.25-0.45 of period)
    rise_fraction = 0.45 - 0.15 * (amplitude - 50) / 150
    rise_time = rise_fraction * period
    decay_time = period - rise_time

    # Maunder Minimum: suppress amplitude for years ~120-190 (mimicking 1645-1715)
    if 120 < current_t < 190:
        amplitude *= 0.05  # nearly zero sunspots

    cycle_amplitudes.append(amplitude)
    cycle_rise_times.append(rise_time)
    cycle_starts.append(current_t)

    # Build asymmetric profile using Hathaway function approximation
    # f(t') = A * (t'/rise)^3 / (exp((t'/rise)^2) - 0.71)  (simplified)
    for i in range(N):
        t_rel = t[i] - current_t
        if 0 <= t_rel < period:
            if t_rel < rise_time:
                # Rising phase: cubic rise
                phase = t_rel / rise_time
                ssn[i] += amplitude * (phase ** 3) * np.exp(-3 * (phase - 1))
            else:
                # Decay phase: exponential decay
                phase = (t_rel - rise_time) / decay_time
                ssn[i] += amplitude * np.exp(-3 * phase)

    current_t += period
    cycle_num += 1

# Add Gaussian noise (proportional to sqrt of signal, Poisson-like)
noise = np.random.normal(0, 1, N) * np.sqrt(np.maximum(ssn, 1.0)) * 0.5
ssn = np.maximum(ssn + noise, 0)

# --- FFT Analysis ---
# Remove mean and apply Hann window to reduce spectral leakage
ssn_detrended = ssn - np.mean(ssn)
window = np.hanning(N)
ssn_windowed = ssn_detrended * window

fft_vals = np.fft.rfft(ssn_windowed)
freqs = np.fft.rfftfreq(N, d=dt)  # in cycles/year
power = np.abs(fft_vals) ** 2

# Find dominant period
valid = freqs > 0.01  # exclude very low frequencies
peak_idx = np.argmax(power[valid])
peak_freq = freqs[valid][peak_idx]
peak_period = 1.0 / peak_freq

print("=" * 60)
print("SOLAR CYCLE ANALYSIS")
print("=" * 60)
print(f"Total simulated span: {T_total:.0f} years ({cycle_num} cycles)")
print(f"Mean cycle period (from generation): "
      f"{np.mean(np.diff(cycle_starts)):.2f} years")
print(f"FFT dominant period: {peak_period:.2f} years")
print(f"Amplitude range: {min(cycle_amplitudes):.1f} - {max(cycle_amplitudes):.1f}")

# Waldmeier effect correlation
amps = np.array(cycle_amplitudes)
rises = np.array(cycle_rise_times)
# Exclude Maunder Minimum cycles (very low amplitude) for correlation
mask = amps > 10
corr = np.corrcoef(amps[mask], rises[mask])[0, 1]
print(f"Waldmeier correlation (amplitude vs rise time): {corr:.3f}")
print(f"  (Negative = stronger cycles rise faster)")

# --- Butterfly Diagram Data ---
# For each cycle, sunspot latitude starts at ~30 deg and migrates to ~5 deg
butterfly_t = []
butterfly_lat = []
for i, t_start in enumerate(cycle_starts):
    period = (cycle_starts[i + 1] - t_start) if i + 1 < len(cycle_starts) else 11.0
    amp = cycle_amplitudes[i]
    if amp < 5:
        continue  # skip Maunder Minimum cycles
    n_spots = int(amp * 2)  # number of synthetic sunspots proportional to amplitude
    for _ in range(n_spots):
        t_spot = t_start + np.random.uniform(0, period)
        phase = (t_spot - t_start) / period
        # Latitude: starts at ±(25-35) deg, ends at ±(5-10) deg (Sporer's law)
        lat_center = 30 - 25 * phase
        lat = np.random.normal(lat_center, 3)
        sign = np.random.choice([-1, 1])
        butterfly_t.append(t_spot)
        butterfly_lat.append(sign * lat)

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Sunspot number time series
ax = axes[0, 0]
ax.plot(t, ssn, 'k-', linewidth=0.3, alpha=0.7)
ax.fill_between(t, 0, ssn, alpha=0.3, color='orange')
ax.axvspan(120, 190, alpha=0.15, color='blue', label='Maunder Minimum')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Sunspot Number')
ax.set_title('Synthetic Sunspot Number Time Series')
ax.legend(loc='upper right')
ax.set_xlim(0, T_total)

# 2. FFT Power Spectrum
ax = axes[0, 1]
periods = 1.0 / freqs[1:]  # skip DC component
ax.semilogy(periods, power[1:], 'b-', linewidth=0.8)
ax.axvline(peak_period, color='r', linestyle='--',
           label=f'Peak = {peak_period:.1f} yr')
ax.axvline(11.0, color='g', linestyle=':', alpha=0.7, label='11.0 yr reference')
ax.set_xlabel('Period (years)')
ax.set_ylabel('Power')
ax.set_title('FFT Power Spectrum')
ax.set_xlim(2, 50)
ax.legend()

# 3. Butterfly Diagram
ax = axes[1, 0]
ax.scatter(butterfly_t, butterfly_lat, s=0.3, c='black', alpha=0.4)
ax.set_xlabel('Time (years)')
ax.set_ylabel('Latitude (degrees)')
ax.set_title('Butterfly Diagram (Sporer\'s Law)')
ax.set_ylim(-45, 45)
ax.set_xlim(0, T_total)
ax.axhline(0, color='gray', linewidth=0.5)

# 4. Waldmeier Effect
ax = axes[1, 1]
sc = ax.scatter(rises[mask], amps[mask], c=np.array(cycle_starts)[mask],
                cmap='viridis', s=40, edgecolors='k', linewidths=0.5)
ax.set_xlabel('Rise Time (years)')
ax.set_ylabel('Cycle Amplitude')
ax.set_title(f'Waldmeier Effect (r = {corr:.3f})')
plt.colorbar(sc, ax=ax, label='Cycle Start (years)')
# Fit line
z = np.polyfit(rises[mask], amps[mask], 1)
x_fit = np.linspace(rises[mask].min(), rises[mask].max(), 50)
ax.plot(x_fit, np.polyval(z, x_fit), 'r--', linewidth=2, label='Linear fit')
ax.legend()

plt.tight_layout()
plt.savefig('/opt/projects/01_Personal/03_Study/examples/Solar_Physics/07_sunspot_cycle.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to 07_sunspot_cycle.png")
