#!/usr/bin/env python3
"""
Signal Processing Applications: Audio, Communications, Radar, Biomedical
=========================================================================

This example demonstrates four major application domains where signal
processing is essential, bringing together techniques from the entire course.

Domains covered:
    1. Audio — pitch detection via autocorrelation
    2. Communications — BPSK modulation and demodulation
    3. Radar — chirp pulse compression with matched filtering
    4. Biomedical — ECG R-peak detection with bandpass filtering

Author: Educational example for Signal Processing
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig


# ============================================================================
# 1. AUDIO: PITCH DETECTION VIA AUTOCORRELATION
# ============================================================================

def demo_pitch_detection():
    """
    Detect the fundamental frequency of a synthetic vowel sound using
    the autocorrelation method.

    Autocorrelation pitch detection:
        R(tau) = sum_n x(n) * x(n + tau)
    The first significant peak in R(tau) after the origin corresponds to
    the fundamental period T0, giving f0 = fs / T0.
    """
    print("=" * 60)
    print("SECTION 1: Audio — Pitch Detection (Autocorrelation)")
    print("=" * 60)

    fs = 16000           # 16 kHz sample rate (speech quality)
    duration = 0.05      # 50 ms analysis window
    f0_true = 220.0      # A3 note (220 Hz)
    t = np.arange(int(fs * duration)) / fs

    # Synthesize a vowel-like signal: fundamental + harmonics
    vowel = (1.0 * np.sin(2 * np.pi * f0_true * t) +
             0.6 * np.sin(2 * np.pi * 2 * f0_true * t) +
             0.3 * np.sin(2 * np.pi * 3 * f0_true * t) +
             0.1 * np.sin(2 * np.pi * 4 * f0_true * t))
    vowel += 0.05 * np.random.randn(len(t))

    # Autocorrelation
    corr = np.correlate(vowel, vowel, mode='full')
    corr = corr[len(corr) // 2:]   # keep positive lags only
    corr /= corr[0]                 # normalise

    # Find the first peak after a minimum (skip lag 0 region)
    min_lag = int(fs / 500)   # 500 Hz upper bound
    max_lag = int(fs / 80)    # 80 Hz lower bound
    search = corr[min_lag:max_lag]
    peak_idx = np.argmax(search) + min_lag
    f0_detected = fs / peak_idx

    print(f"  True f0      : {f0_true:.1f} Hz (A3)")
    print(f"  Detected f0  : {f0_detected:.1f} Hz")
    print(f"  Error        : {abs(f0_detected - f0_true):.2f} Hz "
          f"({abs(f0_detected - f0_true) / f0_true * 100:.2f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Audio Pitch Detection via Autocorrelation",
                 fontsize=12, fontweight='bold')

    axes[0].plot(t * 1000, vowel, 'C0', linewidth=0.8)
    axes[0].set_title("(a) Synthetic Vowel Waveform")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    lags_ms = np.arange(len(corr)) / fs * 1000
    axes[1].plot(lags_ms, corr, 'C1', linewidth=0.8)
    axes[1].axvline(peak_idx / fs * 1000, color='r', linestyle='--',
                     label=f'T0 = {peak_idx / fs * 1000:.2f} ms')
    axes[1].set_title("(b) Autocorrelation")
    axes[1].set_xlabel("Lag (ms)")
    axes[1].set_xlim(0, max_lag / fs * 1000 * 1.2)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    freqs = np.fft.rfftfreq(len(vowel), d=1 / fs)
    spectrum = np.abs(np.fft.rfft(vowel))
    axes[2].plot(freqs, spectrum, 'C2', linewidth=0.8)
    axes[2].set_title("(c) Magnitude Spectrum")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_xlim(0, 1500)
    axes[2].grid(True, alpha=0.3)

    for k in range(1, 5):
        axes[2].axvline(k * f0_true, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig("16_pitch_detection.png", dpi=120)
    print(f"  Saved: 16_pitch_detection.png")
    plt.show()


# ============================================================================
# 2. COMMUNICATIONS: BPSK MODULATION / DEMODULATION
# ============================================================================

def demo_bpsk():
    """
    Binary Phase Shift Keying (BPSK):
        s(t) = A * d(t) * cos(2*pi*fc*t)

    where d(t) in {-1, +1} encodes bits.  Demodulation multiplies by the
    carrier, lowpass-filters, and decides the sign.
    """
    print("\n" + "=" * 60)
    print("SECTION 2: Communications — BPSK Modulation / Demodulation")
    print("=" * 60)

    # Parameters
    fc = 1000            # carrier frequency (Hz)
    bit_rate = 100       # bits per second
    fs = 10000           # sample rate
    snr_db = 8           # signal-to-noise ratio

    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    symbols = 2 * bits - 1   # map 0 -> -1, 1 -> +1

    samples_per_bit = int(fs / bit_rate)
    t_total = len(bits) / bit_rate
    t = np.arange(int(fs * t_total)) / fs

    # Modulate
    baseband = np.repeat(symbols, samples_per_bit)
    carrier = np.cos(2 * np.pi * fc * t)
    tx_signal = baseband * carrier

    # Add AWGN noise
    signal_power = np.mean(tx_signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(tx_signal))
    rx_signal = tx_signal + noise

    # Demodulate: multiply by carrier, then integrate-and-dump
    demod = rx_signal * carrier * 2   # coherent detection

    # Integrate-and-dump: average over each bit period
    decoded_bits = []
    for i in range(len(bits)):
        start = i * samples_per_bit
        end = start + samples_per_bit
        bit_value = 1 if np.mean(demod[start:end]) > 0 else 0
        decoded_bits.append(bit_value)

    decoded_bits = np.array(decoded_bits)
    ber = np.sum(bits != decoded_bits) / len(bits)

    print(f"  TX bits : {bits}")
    print(f"  RX bits : {decoded_bits}")
    print(f"  BER     : {ber:.2f} ({int(ber * len(bits))}/{len(bits)} errors)")
    print(f"  SNR     : {snr_db} dB")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"BPSK Modulation / Demodulation (SNR = {snr_db} dB)",
                 fontsize=12, fontweight='bold')

    # Show first 3 bit periods for clarity
    n_show = 3 * samples_per_bit
    t_show = t[:n_show] * 1000

    axes[0, 0].step(range(len(bits)), bits, 'C0', where='mid', linewidth=2)
    axes[0, 0].set_title("(a) Transmitted Bits")
    axes[0, 0].set_xlabel("Bit index")
    axes[0, 0].set_ylim(-0.2, 1.2)
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t_show, tx_signal[:n_show], 'C1', linewidth=0.6)
    axes[0, 1].set_title("(b) BPSK Modulated Signal (first 3 bits)")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_show, rx_signal[:n_show], 'C2', linewidth=0.6)
    axes[1, 0].set_title("(c) Received Signal (with noise)")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].step(range(len(decoded_bits)), decoded_bits, 'C3',
                     where='mid', linewidth=2)
    axes[1, 1].set_title(f"(d) Decoded Bits (BER = {ber:.2f})")
    axes[1, 1].set_xlabel("Bit index")
    axes[1, 1].set_ylim(-0.2, 1.2)
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("16_bpsk.png", dpi=120)
    print(f"  Saved: 16_bpsk.png")
    plt.show()


# ============================================================================
# 3. RADAR: CHIRP PULSE COMPRESSION
# ============================================================================

def demo_radar_chirp():
    """
    Linear FM chirp pulse compression via matched filtering.

    A chirp sweeps frequency linearly:
        s(t) = cos(2*pi*(f0*t + 0.5*mu*t^2))
    where mu = bandwidth / pulse_width is the chirp rate.

    Matched filter output:  y(t) = s(t) * s*(-t)
    This compresses the long pulse into a narrow peak whose width is
    approximately 1/B, giving range resolution independent of pulse duration.
    """
    print("\n" + "=" * 60)
    print("SECTION 3: Radar — Chirp Pulse Compression")
    print("=" * 60)

    fs = 100e3           # sample rate (Hz)
    pulse_width = 10e-3  # 10 ms pulse
    f0 = 5e3             # start frequency
    bandwidth = 20e3     # chirp bandwidth
    mu = bandwidth / pulse_width  # chirp rate

    t_pulse = np.arange(int(fs * pulse_width)) / fs
    chirp = np.cos(2 * np.pi * (f0 * t_pulse + 0.5 * mu * t_pulse ** 2))

    # Simulate received signal: two targets at different delays
    total_samples = int(fs * 30e-3)  # 30 ms observation window
    rx = np.zeros(total_samples)

    target_delays = [8e-3, 15e-3]   # seconds
    target_amps = [1.0, 0.6]

    for delay, amp in zip(target_delays, target_amps):
        idx = int(delay * fs)
        end = min(idx + len(chirp), total_samples)
        rx[idx:end] += amp * chirp[:end - idx]

    # Add noise
    rx += 0.3 * np.random.randn(total_samples)

    # Matched filter: cross-correlate with the chirp
    matched = np.correlate(rx, chirp, mode='full')
    matched = matched[len(chirp) - 1:]  # causal part
    matched_db = 20 * np.log10(np.abs(matched) / np.max(np.abs(matched)) + 1e-12)

    t_rx = np.arange(total_samples) / fs * 1000

    # Theoretical compressed pulse width
    compressed_width = 1.0 / bandwidth
    compression_ratio = pulse_width * bandwidth

    print(f"  Chirp bandwidth     : {bandwidth / 1e3:.0f} kHz")
    print(f"  Pulse width         : {pulse_width * 1e3:.1f} ms")
    print(f"  Compression ratio   : {compression_ratio:.0f}")
    print(f"  Compressed width    : {compressed_width * 1e6:.1f} us")
    print(f"  Target delays       : {[d * 1e3 for d in target_delays]} ms")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Radar Chirp Pulse Compression",
                 fontsize=12, fontweight='bold')

    axes[0, 0].plot(t_pulse * 1000, chirp, 'C0', linewidth=0.5)
    axes[0, 0].set_title(f"(a) Transmitted Chirp ({f0 / 1e3:.0f}–"
                          f"{(f0 + bandwidth) / 1e3:.0f} kHz)")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].grid(True, alpha=0.3)

    # Spectrogram of chirp
    axes[0, 1].specgram(chirp, Fs=fs, NFFT=128, noverlap=120, cmap='hot')
    axes[0, 1].set_title("(b) Chirp Spectrogram")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Frequency (Hz)")

    axes[1, 0].plot(t_rx, rx, 'C1', linewidth=0.5)
    axes[1, 0].set_title("(c) Received Signal (2 targets + noise)")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t_rx, matched_db[:len(t_rx)], 'C2', linewidth=0.8)
    axes[1, 1].set_title("(d) Matched Filter Output (dB)")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[1, 1].set_ylabel("Magnitude (dB)")
    axes[1, 1].set_ylim(-40, 5)
    axes[1, 1].grid(True, alpha=0.3)
    for delay in target_delays:
        axes[1, 1].axvline(delay * 1000, color='r', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("16_radar_chirp.png", dpi=120)
    print(f"  Saved: 16_radar_chirp.png")
    plt.show()


# ============================================================================
# 4. BIOMEDICAL: ECG R-PEAK DETECTION
# ============================================================================

def demo_ecg_rpeak():
    """
    Detect R-peaks in a synthetic ECG signal using bandpass filtering
    and thresholding (simplified Pan-Tompkins approach).

    Steps:
        1. Bandpass filter (5-15 Hz) to isolate QRS complex energy
        2. Square the signal to emphasise large deflections
        3. Moving-average integration
        4. Adaptive thresholding to detect R-peaks
    """
    print("\n" + "=" * 60)
    print("SECTION 4: Biomedical — ECG R-Peak Detection")
    print("=" * 60)

    fs = 360             # typical ECG sample rate (Hz)
    duration = 5.0       # seconds
    heart_rate = 72      # beats per minute
    t = np.arange(int(fs * duration)) / fs

    # Synthesize ECG-like signal with PQRST morphology
    ecg = np.zeros_like(t)
    beat_interval = 60.0 / heart_rate
    beat_times = np.arange(0, duration - 0.2, beat_interval)

    for bt in beat_times:
        # P wave
        ecg += 0.15 * np.exp(-((t - bt - 0.0) ** 2) / (2 * 0.01 ** 2))
        # Q wave (small negative)
        ecg -= 0.08 * np.exp(-((t - bt - 0.06) ** 2) / (2 * 0.005 ** 2))
        # R wave (tall positive)
        ecg += 1.0 * np.exp(-((t - bt - 0.08) ** 2) / (2 * 0.005 ** 2))
        # S wave (negative)
        ecg -= 0.15 * np.exp(-((t - bt - 0.10) ** 2) / (2 * 0.005 ** 2))
        # T wave
        ecg += 0.25 * np.exp(-((t - bt - 0.22) ** 2) / (2 * 0.02 ** 2))

    # Add baseline wander and noise
    ecg += 0.1 * np.sin(2 * np.pi * 0.3 * t)  # baseline wander
    ecg += 0.05 * np.random.randn(len(t))       # measurement noise

    # Step 1: Bandpass filter (5-15 Hz)
    sos = sig.butter(4, [5, 15], btype='bandpass', fs=fs, output='sos')
    filtered = sig.sosfilt(sos, ecg)

    # Step 2: Square
    squared = filtered ** 2

    # Step 3: Moving average integration (150 ms window)
    window_size = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(window_size) / window_size,
                              mode='same')

    # Step 4: Adaptive threshold
    threshold = 0.4 * np.max(integrated)
    above = integrated > threshold

    # Find peaks (local maxima above threshold)
    peaks, _ = sig.find_peaks(integrated, height=threshold,
                               distance=int(0.4 * fs))

    # Compute heart rate from R-R intervals
    rr_intervals = np.diff(peaks) / fs  # seconds
    detected_hr = 60.0 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0

    print(f"  True heart rate     : {heart_rate} bpm")
    print(f"  Detected heart rate : {detected_hr:.1f} bpm")
    print(f"  R-peaks found       : {len(peaks)}")
    print(f"  Mean R-R interval   : {np.mean(rr_intervals) * 1000:.1f} ms"
          if len(rr_intervals) > 0 else "  No R-R intervals")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("ECG R-Peak Detection (Simplified Pan-Tompkins)",
                 fontsize=12, fontweight='bold')

    axes[0].plot(t, ecg, 'C0', linewidth=0.8)
    axes[0].plot(t[peaks], ecg[peaks], 'rv', markersize=8, label='R-peaks')
    axes[0].set_title("(a) Raw ECG with Detected R-peaks")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, filtered, 'C1', linewidth=0.8)
    axes[1].set_title("(b) Bandpass Filtered (5–15 Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, integrated, 'C2', linewidth=0.8)
    axes[2].axhline(threshold, color='r', linestyle='--', alpha=0.7,
                     label=f'Threshold = {threshold:.4f}')
    axes[2].plot(t[peaks], integrated[peaks], 'rv', markersize=8)
    axes[2].set_title("(c) Squared + Integrated + Threshold")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Energy")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("16_ecg_rpeak.png", dpi=120)
    print(f"  Saved: 16_ecg_rpeak.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Signal Processing Applications")
    print("=" * 60)
    print("Four domains: Audio, Communications, Radar, Biomedical")
    print()

    demo_pitch_detection()
    demo_bpsk()
    demo_radar_chirp()
    demo_ecg_rpeak()

    print("\nDone.  Four PNG files saved.")
    print("\nKey takeaways:")
    print("  - Audio: autocorrelation reliably estimates pitch (fundamental freq)")
    print("  - Communications: BPSK is the simplest digital modulation scheme;")
    print("    coherent demodulation recovers bits even in noisy channels")
    print("  - Radar: chirp pulse compression via matched filtering achieves")
    print("    fine range resolution without sacrificing transmit energy")
    print("  - Biomedical: bandpass filtering + squaring + integration is a")
    print("    robust approach to QRS/R-peak detection in ECG signals")
