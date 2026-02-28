"""
Exercises for Lesson 05: Sampling and Reconstruction
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import fft, ifft, fftfreq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Sampling Rate Determination ===
# Problem: Determine sampling rates for a signal with 100, 250, 500 Hz components.

def exercise_1():
    """Sampling rate determination and aliasing analysis."""
    freqs = [100, 250, 500]
    f_max = max(freqs)

    # (a) Minimum sampling rate (Nyquist rate)
    f_nyquist = 2 * f_max
    print(f"Signal components: {freqs} Hz")
    print(f"(a) Minimum sampling rate (Nyquist): {f_nyquist} Hz")
    print()

    # (b) Sampling at 800 Hz
    fs = 800
    f_nyquist_freq = fs / 2  # 400 Hz
    print(f"(b) Sampling at {fs} Hz (Nyquist freq = {f_nyquist_freq} Hz):")
    for f in freqs:
        if f > f_nyquist_freq:
            f_alias = fs - f  # folding
            print(f"    {f} Hz -> ALIASES to {f_alias} Hz")
        else:
            print(f"    {f} Hz -> passes unaliased")
    print()

    # (c) Recommended practical rate
    fs_rec = int(2.5 * f_max)
    print(f"(c) Recommended rate: ~{fs_rec} Hz (2.5x max frequency)")
    print("    Provides margin for anti-aliasing filter rolloff")


# === Exercise 2: Aliasing Analysis ===
# Problem: 1 kHz signal sampled at 1.5 kHz.

def exercise_2():
    """Aliasing demonstration."""
    f_signal = 1000  # Hz
    fs = 1500  # Hz

    # (a) Alias frequency
    f_alias = fs - f_signal
    print(f"(a) Original: {f_signal} Hz, fs: {fs} Hz")
    print(f"    Alias frequency: {f_alias} Hz")
    print()

    # (b) Visual demonstration
    t_cont = np.linspace(0, 0.01, 10000)
    x_cont = np.sin(2 * np.pi * f_signal * t_cont)

    n = np.arange(0, int(0.01 * fs))
    ts = n / fs
    x_sampled = np.sin(2 * np.pi * f_signal * ts)

    # The aliased signal
    x_alias = np.sin(2 * np.pi * f_alias * t_cont)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(t_cont * 1000, x_cont, 'b-', alpha=0.3, label=f'{f_signal} Hz (original)')
    ax.plot(t_cont * 1000, x_alias, 'r--', alpha=0.5, label=f'{f_alias} Hz (alias)')
    ax.stem(ts * 1000, x_sampled, linefmt='g-', markerfmt='go', basefmt='k-',
            label=f'Samples at {fs} Hz')
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'Aliasing: {f_signal} Hz appears as {f_alias} Hz at fs={fs} Hz')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex05_aliasing.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("(b) Plot saved to ex05_aliasing.png")
    print()

    # (c) Anti-aliasing filter
    f_cutoff = 600  # Hz
    order = 8
    print(f"(c) Anti-aliasing filter:")
    print(f"    Type: Butterworth lowpass")
    print(f"    Order: {order}")
    print(f"    Cutoff: {f_cutoff} Hz")
    print(f"    Nyquist freq: {fs / 2} Hz")

    sos = sig.butter(order, f_cutoff, fs=fs, btype='low', output='sos')
    w, h = sig.sosfreqz(sos, worN=2048, fs=fs)
    atten_at_signal = 20 * np.log10(np.abs(np.interp(f_signal, w, np.abs(h))) + 1e-15)
    print(f"    Attenuation at {f_signal} Hz: {atten_at_signal:.1f} dB")


# === Exercise 3: Sinc Interpolation Implementation ===
# Problem: Implement sinc interpolation for signal reconstruction.

def exercise_3():
    """Sinc interpolation for bandlimited signal reconstruction."""
    def sinc_reconstruct(samples, Ts, t_recon):
        """Reconstruct a CT signal from samples using sinc interpolation."""
        x_recon = np.zeros_like(t_recon)
        for n, s in enumerate(samples):
            x_recon += s * np.sinc((t_recon - n * Ts) / Ts)
        return x_recon

    # Test with known bandlimited signal
    f_signal = 100  # Hz
    fs = 500  # Hz (well above Nyquist)
    Ts = 1 / fs
    duration = 0.05

    # Generate samples
    n_samples = np.arange(0, int(duration * fs))
    samples = np.sin(2 * np.pi * f_signal * n_samples * Ts)

    # Dense reconstruction
    t_recon = np.linspace(0, duration - Ts, 5000)
    x_true = np.sin(2 * np.pi * f_signal * t_recon)
    x_recon = sinc_reconstruct(samples, Ts, t_recon)

    error = np.max(np.abs(x_true - x_recon))
    print(f"(a) Sinc reconstruction of {f_signal} Hz sinusoid at fs={fs} Hz")
    print(f"    Max reconstruction error: {error:.6f}")
    print()

    # (b) Truncation analysis
    print("(b) Reconstruction error vs truncation length:")
    for L in [5, 10, 20, 50, len(samples)]:
        L_actual = min(L, len(samples))
        x_trunc = np.zeros_like(t_recon)
        for i, t_val in enumerate(t_recon):
            center = int(t_val / Ts)
            start = max(0, center - L_actual // 2)
            end = min(len(samples), center + L_actual // 2 + 1)
            for n in range(start, end):
                x_trunc[i] += samples[n] * np.sinc((t_val - n * Ts) / Ts)
        err = np.max(np.abs(x_true - x_trunc))
        print(f"    L={L_actual:3d} sinc terms: max error = {err:.6f}")


# === Exercise 4: Quantization Noise Analysis ===
# Problem: Verify SQNR = 6.02N + 1.76 dB formula.

def exercise_4():
    """Quantization noise analysis and SQNR verification."""
    fs = 8000
    t = np.arange(0, 1.0, 1 / fs)

    print("SQNR verification (full-scale sinusoidal input):")
    print(f"{'Bits':<8} {'Analytical (dB)':<18} {'Measured (dB)':<18} {'Error (dB)':<12}")

    for N_bits in [4, 8, 12, 16]:
        # Full-scale sinusoid
        levels = 2 ** N_bits
        x = np.sin(2 * np.pi * 100 * t)  # amplitude 1.0

        # Quantize
        x_scaled = (x + 1) / 2 * (levels - 1)
        x_quantized = np.round(x_scaled) / (levels - 1) * 2 - 1

        # Compute SQNR
        noise = x - x_quantized
        signal_power = np.mean(x ** 2)
        noise_power = np.mean(noise ** 2)
        sqnr_measured = 10 * np.log10(signal_power / noise_power)
        sqnr_analytical = 6.02 * N_bits + 1.76

        print(f"{N_bits:<8} {sqnr_analytical:<18.2f} {sqnr_measured:<18.2f} "
              f"{abs(sqnr_measured - sqnr_analytical):<12.2f}")

    # (c) Half-range signal
    print(f"\n(c) Half-range signal (amplitude = 0.5 instead of 1.0):")
    for N_bits in [8, 16]:
        levels = 2 ** N_bits
        x_half = 0.5 * np.sin(2 * np.pi * 100 * t)
        x_scaled = (x_half + 1) / 2 * (levels - 1)
        x_quantized = np.round(x_scaled) / (levels - 1) * 2 - 1
        noise = x_half - x_quantized
        sqnr = 10 * np.log10(np.mean(x_half ** 2) / np.mean(noise ** 2))
        sqnr_full = 6.02 * N_bits + 1.76
        print(f"  {N_bits} bits: SQNR = {sqnr:.2f} dB (full-scale: {sqnr_full:.2f} dB, "
              f"loss: {sqnr_full - sqnr:.2f} dB)")
    print("  Using half the range costs ~6 dB (equivalent to losing 1 bit)")


# === Exercise 5: ZOH Frequency Response ===
# Problem: Plot ZOH magnitude/phase and design compensation filter.

def exercise_5():
    """Zero-order hold reconstruction analysis."""
    fs = 8000
    f = np.linspace(0, fs / 2, 1000)

    # ZOH frequency response: H_zoh(f) = sinc(f/fs) * e^{-j*pi*f/fs}
    # |H_zoh(f)| = |sinc(f/fs)|
    Ts = 1 / fs
    H_zoh_mag = np.abs(np.sinc(f / fs))
    H_zoh_phase = -np.pi * f / fs  # linear phase from delay of Ts/2

    # (b) Attenuation at 3000 Hz
    atten_3k = 20 * np.log10(np.abs(np.sinc(3000 / fs)))
    print(f"(a) ZOH frequency response at fs = {fs} Hz")
    print(f"(b) Attenuation at 3000 Hz: {atten_3k:.2f} dB")
    print()

    # (c) Sinc compensation filter (inverse sinc in passband)
    N_taps = 31
    # Design FIR filter with inverse sinc response
    n = np.arange(N_taps)
    # Target: H_comp(f) = 1/sinc(f/fs) for f in passband
    # Use frequency sampling method
    N_fft = 512
    f_fft = np.linspace(0, fs / 2, N_fft // 2 + 1)
    H_target = np.ones(N_fft // 2 + 1)
    for i, fi in enumerate(f_fft):
        sinc_val = np.sinc(fi / fs)
        if np.abs(sinc_val) > 0.1:
            H_target[i] = 1 / sinc_val
        else:
            H_target[i] = 1.0  # limit in stopband

    # Create symmetric frequency response for IFFT
    H_full = np.concatenate([H_target, H_target[-2:0:-1]])
    h_comp = np.real(ifft(H_full))

    # Window and shift
    h_comp = np.roll(h_comp, N_taps // 2)[:N_taps]
    h_comp *= np.hamming(N_taps)
    h_comp /= np.sum(h_comp)  # normalize

    # Verify compensation
    w_comp, H_comp_freq = sig.freqz(h_comp, 1, worN=1024, fs=fs)
    H_compensated = H_zoh_mag * np.interp(f, w_comp, np.abs(H_comp_freq))

    droop_before = 20 * np.log10(H_zoh_mag[int(3000 / (fs / 2) * 999)])
    # Find closest frequency index in compensation response
    droop_after = 20 * np.log10(H_compensated[int(3000 / (fs / 2) * 999)] + 1e-10)
    print(f"(c) Sinc compensation filter ({N_taps} taps):")
    print(f"    Passband droop at 3 kHz before: {droop_before:.2f} dB")
    print(f"    Passband droop at 3 kHz after:  {droop_after:.2f} dB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(f, 20 * np.log10(H_zoh_mag + 1e-10), 'b-', linewidth=2)
    axes[0].set_title('ZOH Magnitude Response')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f, H_zoh_phase * 180 / np.pi, 'r-', linewidth=2)
    axes[1].set_title('ZOH Phase Response')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex05_zoh.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex05_zoh.png")


# === Exercise 6: Oversampling vs. Bit Depth Trade-off ===
# Problem: Compute oversampling ratios for 16-bit effective resolution.

def exercise_6():
    """Oversampling and noise shaping analysis."""
    target_sqnr = 98  # dB (16-bit)
    N_adc = 12  # bits
    sqnr_adc = 6.02 * N_adc + 1.76  # ~74 dB

    # (a) Without noise shaping: SQNR improves by 3 dB per doubling
    # SQNR_eff = SQNR_adc + 10*log10(M)
    # 98 = 74 + 10*log10(M) -> M = 10^(24/10) = 10^2.4
    gap = target_sqnr - sqnr_adc
    M_no_shaping = 10 ** (gap / 10)
    print(f"(a) No noise shaping:")
    print(f"    ADC SQNR: {sqnr_adc:.2f} dB, target: {target_sqnr} dB")
    print(f"    Gap: {gap:.2f} dB")
    print(f"    Required oversampling ratio M = {M_no_shaping:.0f}x")
    print()

    # (b) With first-order noise shaping: improvement = 30*log10(M) - 10*log10(pi^2/3)
    # SQNR_eff ≈ SQNR_adc + 30*log10(M) - 5.17
    # 98 = 74 + 30*log10(M) - 5.17
    # 30*log10(M) = 29.17
    # M = 10^(29.17/30)
    correction = 10 * np.log10(np.pi ** 2 / 3)
    M_1st_order = 10 ** ((gap + correction) / 30)
    print(f"(b) First-order noise shaping:")
    print(f"    Required oversampling ratio M = {M_1st_order:.0f}x")
    print()

    # (c) Simulate first-order sigma-delta modulator
    fs_base = 8000
    M_sim = 64
    fs_os = fs_base * M_sim
    duration = 0.01
    t = np.arange(0, duration, 1 / fs_os)

    # Input signal
    f_in = 100  # Hz
    x = 0.5 * np.sin(2 * np.pi * f_in * t)

    # First-order sigma-delta modulator
    y = np.zeros(len(x))
    integrator = 0.0
    for i in range(len(x)):
        integrator += x[i] - y[i - 1] if i > 0 else x[i]
        y[i] = 1.0 if integrator >= 0 else -1.0

    # Spectrum of sigma-delta output
    N_fft = len(y)
    Y = fft(y)
    freqs = fftfreq(N_fft, 1 / fs_os)
    psd = np.abs(Y[:N_fft // 2]) ** 2 / N_fft

    print(f"(c) First-order Sigma-Delta modulator (M={M_sim}):")
    print(f"    Input freq: {f_in} Hz, OSR: {M_sim}")
    print(f"    Output is 1-bit (+/-1)")

    # Measure in-band noise
    f_band = fs_base / 2
    in_band = freqs[:N_fft // 2] < f_band
    noise_in_band = np.sum(psd[in_band]) / np.sum(psd)
    print(f"    In-band noise fraction: {noise_in_band:.4f}")
    print("    Noise is shaped toward high frequencies (noise shaping)")


# === Exercise 7: Complete ADC/DAC Pipeline ===
# Problem: Build complete sampling-reconstruction pipeline.

def exercise_7():
    """Complete ADC/DAC pipeline simulation."""
    # 1. Test signal
    fs_orig = 100000  # high rate for "continuous" reference
    duration = 0.01
    t_orig = np.arange(0, duration, 1 / fs_orig)

    f1, f2, f3 = 500, 1200, 1800
    x_orig = (np.sin(2 * np.pi * f1 * t_orig) +
              0.5 * np.sin(2 * np.pi * f2 * t_orig) +
              0.3 * np.sin(2 * np.pi * f3 * t_orig))

    # 2. Anti-aliasing filter
    fs = 8000
    sos_aa = sig.butter(8, 0.9 * fs / 2, fs=fs_orig, btype='low', output='sos')
    x_filtered = sig.sosfilt(sos_aa, x_orig)

    # 3. Sample
    downsample_factor = fs_orig // fs
    n_samples = np.arange(0, len(x_orig), downsample_factor)
    x_sampled = x_filtered[n_samples]
    t_sampled = t_orig[n_samples]

    # 4. Quantize
    N_bits = 12
    levels = 2 ** N_bits
    x_max = np.max(np.abs(x_sampled))
    x_norm = x_sampled / x_max
    x_quant = np.round((x_norm + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1
    x_quant *= x_max

    # 5. Reconstruct with three methods
    t_recon = t_orig

    # (a) ZOH
    x_zoh = np.zeros_like(t_recon)
    for i, ts in enumerate(t_sampled):
        mask = (t_recon >= ts) & (t_recon < ts + 1 / fs)
        x_zoh[mask] = x_quant[i]

    # (b) Linear interpolation
    x_linear = np.interp(t_recon, t_sampled, x_quant)

    # (c) Sinc interpolation (truncated)
    x_sinc = np.zeros_like(t_recon)
    Ts = 1 / fs
    for i, s in enumerate(x_quant):
        x_sinc += s * np.sinc((t_recon - i * Ts) / Ts)

    # 6. Reconstruction lowpass filter
    sos_recon = sig.butter(8, 0.9 * fs / 2, fs=fs_orig, btype='low', output='sos')
    x_zoh_filt = sig.sosfilt(sos_recon, x_zoh)
    x_linear_filt = sig.sosfilt(sos_recon, x_linear)
    x_sinc_filt = sig.sosfilt(sos_recon, x_sinc)

    # 7. Compare
    methods = {
        'ZOH': x_zoh_filt,
        'Linear': x_linear_filt,
        'Sinc': x_sinc_filt
    }

    print(f"ADC/DAC Pipeline: fs={fs} Hz, {N_bits}-bit quantization")
    print(f"Signal: {f1}+{f2}+{f3} Hz")
    print()
    print(f"{'Method':<12} {'RMSE':<12} {'SNR (dB)':<12}")
    print("-" * 36)
    for name, x_recon in methods.items():
        # Align signals (account for filter delay)
        delay = 200  # approximate group delay in samples
        x_ref = x_orig[delay:delay + len(x_recon) - delay]
        x_rec = x_recon[delay:len(x_recon)][:len(x_ref)]
        rmse = np.sqrt(np.mean((x_ref - x_rec) ** 2))
        snr = 10 * np.log10(np.mean(x_ref ** 2) / np.mean((x_ref - x_rec) ** 2))
        print(f"{name:<12} {rmse:<12.6f} {snr:<12.2f}")


if __name__ == "__main__":
    print("=== Exercise 1: Sampling Rate Determination ===")
    exercise_1()
    print("\n=== Exercise 2: Aliasing Analysis ===")
    exercise_2()
    print("\n=== Exercise 3: Sinc Interpolation ===")
    exercise_3()
    print("\n=== Exercise 4: Quantization Noise Analysis ===")
    exercise_4()
    print("\n=== Exercise 5: ZOH Frequency Response ===")
    exercise_5()
    print("\n=== Exercise 6: Oversampling vs. Bit Depth ===")
    exercise_6()
    print("\n=== Exercise 7: Complete ADC/DAC Pipeline ===")
    exercise_7()
    print("\nAll exercises completed!")
