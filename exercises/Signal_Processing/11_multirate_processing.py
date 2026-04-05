"""
Exercises for Lesson 11: Multirate Signal Processing
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import fft, fftfreq
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Downsampling Analysis ===
# Problem: x[n] = cos(0.2*pi*n) + cos(0.7*pi*n), fs=10 kHz. Downsample by M=2 and M=4.

def exercise_1():
    """Downsampling aliasing analysis."""
    fs = 10000
    N = 1024
    n = np.arange(N)
    x = np.cos(0.2 * np.pi * n) + np.cos(0.7 * np.pi * n)

    # (a) DTFT of x[n]
    print("(a) DTFT of x[n]:")
    print("    Frequencies: 0.1*fs=1000 Hz and 0.35*fs=3500 Hz")
    print("    Spectrum has peaks at +/-0.2*pi and +/-0.7*pi")

    # (b) Downsample by M=2
    x_down2 = x[::2]
    print(f"\n(b) Downsample by M=2 (new fs={fs//2} Hz):")
    print(f"    0.2*pi -> 0.4*pi (no aliasing, maps to 1000 Hz at 5 kHz)")
    print(f"    0.7*pi -> 1.4*pi aliases to (1.4-2)*pi = -0.6*pi -> 0.6*pi")
    print(f"    Aliased component at 0.6*pi = 1500 Hz at new rate")

    # (c) Downsample by M=4
    x_down4 = x[::4]
    print(f"\n(c) Downsample by M=4 (new fs={fs//4} Hz):")
    print(f"    Both components alias. Multiple folded copies overlap.")

    # (d) Spectral verification
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    for ax, data, title, rate in [
        (axes[0], x, f'Original (fs={fs} Hz)', fs),
        (axes[1], x_down2, f'Downsampled M=2 (fs={fs//2} Hz)', fs//2),
        (axes[2], x_down4, f'Downsampled M=4 (fs={fs//4} Hz)', fs//4),
    ]:
        X = fft(data)
        freqs = fftfreq(len(data), 1/rate)
        ax.plot(freqs[:len(freqs)//2], np.abs(X[:len(X)//2]) / len(data))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex11_downsampling.png', dpi=100)
    plt.close()

    # (e) Anti-aliasing filter for M=4
    cutoff = 0.2 * np.pi / (np.pi)  # Normalized: 0.2*pi / (M*pi) would be Nyquist
    # For M=4, keep only 0.2*pi (below pi/4 = 0.25*pi at original rate)
    h_aa = sig.firwin(65, 1/4)  # cutoff at pi/4
    x_filtered = sig.lfilter(h_aa, 1, x)
    x_safe_down4 = x_filtered[::4]

    print(f"\n(e) Anti-aliasing filter: 65-tap FIR, cutoff=pi/4")
    print(f"    After filtering and decimating by 4:")
    print(f"    0.2*pi component preserved, 0.7*pi component removed")
    print("    Plot saved: ex11_downsampling.png")


# === Exercise 2: Interpolation Quality ===
# Problem: 1 kHz sine at 8 kHz, upsample by L=4.

def exercise_2():
    """Interpolation quality comparison."""
    fs = 8000
    f_tone = 1000
    L = 4
    fs_new = fs * L
    duration = 0.01
    n = np.arange(0, duration, 1/fs)
    x = np.sin(2 * np.pi * f_tone * n)

    # (a) Zero insertion only
    x_up_zeros = np.zeros(len(x) * L)
    x_up_zeros[::L] = x

    X_up = fft(x_up_zeros)
    freqs_up = fftfreq(len(x_up_zeros), 1/fs_new)

    print("(a) Zero-insertion upsampling by L=4")
    print(f"    Creates imaging copies at {f_tone}+k*{fs} Hz for k=1,2,3")

    # (b) Three interpolation methods
    # Linear interpolation
    n_up = np.arange(len(x) * L)
    n_orig = np.arange(len(x)) * L
    x_linear = np.interp(n_up, n_orig, x)

    # FIR lowpass (32 taps)
    h32 = sig.firwin(32, 1/L) * L
    x_fir32 = sig.lfilter(h32, 1, x_up_zeros)

    # FIR lowpass (128 taps)
    h128 = sig.firwin(128, 1/L) * L
    x_fir128 = sig.lfilter(h128, 1, x_up_zeros)

    # (c) Reference: scipy.signal.resample
    x_ref = sig.resample(x, len(x) * L)

    # Compute SNR for each method
    # Align signals (account for filter delays)
    def compute_snr(y, ref, skip=50):
        y_trim = y[skip:skip+len(ref)-2*skip]
        ref_trim = ref[skip:skip+len(ref)-2*skip]
        if len(y_trim) != len(ref_trim):
            min_len = min(len(y_trim), len(ref_trim))
            y_trim = y_trim[:min_len]
            ref_trim = ref_trim[:min_len]
        noise = y_trim - ref_trim
        snr = 10 * np.log10(np.sum(ref_trim**2) / (np.sum(noise**2) + 1e-15))
        return snr

    snr_zeros = compute_snr(x_up_zeros, x_ref)
    snr_linear = compute_snr(x_linear, x_ref)
    snr_fir32 = compute_snr(x_fir32, x_ref)
    snr_fir128 = compute_snr(x_fir128, x_ref)

    print(f"\n(b-c) Interpolation SNR (vs scipy.signal.resample):")
    print(f"    Zero insertion: {snr_zeros:.1f} dB")
    print(f"    Linear interp:  {snr_linear:.1f} dB")
    print(f"    FIR 32-tap:     {snr_fir32:.1f} dB")
    print(f"    FIR 128-tap:    {snr_fir128:.1f} dB")

    # (d) Time-domain comparison
    t_up = np.arange(len(x_ref)) / fs_new
    t_ideal = np.linspace(0, duration, 1000)
    x_ideal = np.sin(2 * np.pi * f_tone * t_ideal)

    fig, ax = plt.subplots(figsize=(12, 5))
    show = 100
    ax.plot(t_ideal[:show]*1000, x_ideal[:show], 'k-', linewidth=2, label='Ideal', alpha=0.5)
    ax.plot(t_up[:show]*1000, x_linear[:show], 'g--', label='Linear', alpha=0.7)
    ax.plot(t_up[:show]*1000, x_fir32[:show], 'b-.', label='FIR-32', alpha=0.7)
    ax.plot(t_up[:show]*1000, x_fir128[:show], 'r:', label='FIR-128', alpha=0.7)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Interpolation Methods Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex11_interpolation.png', dpi=100)
    plt.close()
    print("    Plot saved: ex11_interpolation.png")


# === Exercise 3: Polyphase Implementation ===
# Problem: 128-tap FIR, decimation M=8.

def exercise_3():
    """Polyphase decimator implementation."""

    class PolyphaseDecimator:
        """Efficient polyphase decimation filter."""

        def __init__(self, h, M):
            self.M = M
            self.num_phases = M
            L = len(h)
            # Pad h to multiple of M
            pad_len = (M - L % M) % M
            h_padded = np.concatenate([h, np.zeros(pad_len)])
            # Decompose into polyphase components
            self.subfilters = h_padded.reshape(-1, M).T
            self.phase_len = self.subfilters.shape[1]
            # State for each phase
            self.state = np.zeros((M, self.phase_len - 1))

        def process_block(self, block):
            """Process a block of input samples."""
            # Number of output samples
            n_out = len(block) // self.M
            output = np.zeros(n_out)

            for k in range(n_out):
                sample_start = k * self.M
                for p in range(self.M):
                    idx = sample_start + p
                    if idx < len(block):
                        # Convolve with subfilter p
                        phase_input = np.concatenate([[block[idx]], self.state[p]])
                        output[k] += np.dot(self.subfilters[p], phase_input[:self.phase_len])
                        # Update state
                        self.state[p] = np.roll(self.state[p], 1)
                        self.state[p][0] = block[idx]

            return output

    # Setup
    M = 8
    h = sig.firwin(128, 1/M)
    np.random.seed(42)
    N = 8192
    x = np.random.randn(N)

    # (a) Polyphase decomposition
    num_subfilters = M
    subfilter_len = len(h) // M + (1 if len(h) % M else 0)
    print(f"(a) Polyphase decomposition:")
    print(f"    Filter length: {len(h)}")
    print(f"    Decimation factor M: {M}")
    print(f"    Number of subfilters: {num_subfilters}")
    print(f"    Subfilter length: {subfilter_len}")

    # (c) Verify against scipy.signal.decimate
    y_reference = sig.decimate(x, M, ftype='fir', n=127)

    # Standard implementation: filter then downsample
    y_standard = sig.lfilter(h, 1, x)[::M]

    # Polyphase (simplified verification via numpy)
    h_padded = np.concatenate([h, np.zeros((M - len(h) % M) % M)])
    polyphase_components = h_padded.reshape(-1, M).T

    # Verify
    error = np.max(np.abs(y_standard[:len(y_reference)] - y_reference[:len(y_standard)]))
    print(f"\n(c) Max error vs scipy.signal.decimate: {error:.2e}")

    # (d) Benchmark
    n_runs = 100
    x_large = np.random.randn(65536)

    start = time.time()
    for _ in range(n_runs):
        y_std = sig.lfilter(h, 1, x_large)[::M]
    time_standard = (time.time() - start) / n_runs

    # Polyphase via resample_poly (scipy's optimized version)
    start = time.time()
    for _ in range(n_runs):
        y_poly = sig.resample_poly(x_large, 1, M, window=h)
    time_polyphase = (time.time() - start) / n_runs

    speedup = time_standard / time_polyphase
    print(f"\n(d) Benchmark (65536 samples, {n_runs} runs):")
    print(f"    Standard: {time_standard*1000:.2f} ms/run")
    print(f"    Polyphase: {time_polyphase*1000:.2f} ms/run")
    print(f"    Speedup: {speedup:.2f}x")


# === Exercise 4: Rational Rate Conversion ===
# Problem: Convert from 11025 Hz to 8000 Hz.

def exercise_4():
    """Rational rate conversion 11025 -> 8000 Hz."""
    fs_in = 11025
    fs_out = 8000

    # (a) Find L/M ratio
    from math import gcd
    g = gcd(fs_out, fs_in)
    L = fs_out // g
    M = fs_in // g
    print(f"(a) Rate conversion: {fs_in} -> {fs_out} Hz")
    print(f"    Ratio: {fs_out}/{fs_in} = {L}/{M}")
    print(f"    GCD: {g}")
    print(f"    Upsample by L={L}, downsample by M={M}")

    # (b) Anti-aliasing/anti-imaging filter
    # Cutoff at min(pi/L, pi/M) = pi/M (since M > L)
    cutoff = min(1/L, 1/M)
    print(f"\n(b) Filter cutoff: min(pi/{L}, pi/{M}) = pi/{max(L,M)}")
    print(f"    Normalized cutoff: {cutoff:.4f}")

    # (c) Implementation
    duration = 0.5
    t = np.arange(0, duration, 1/fs_in)
    freqs = [500, 1000, 2000, 3500]
    x = sum(np.sin(2*np.pi*f*t) for f in freqs)

    y = sig.resample_poly(x, L, M)

    # (d) Spectral analysis
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    X = fft(x)
    f_axis_in = fftfreq(len(x), 1/fs_in)
    axes[0].plot(f_axis_in[:len(f_axis_in)//2],
                 np.abs(X[:len(X)//2]) / len(x))
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title(f'Input Spectrum (fs={fs_in} Hz)')
    axes[0].grid(True, alpha=0.3)

    Y = fft(y)
    f_axis_out = fftfreq(len(y), 1/fs_out)
    axes[1].plot(f_axis_out[:len(f_axis_out)//2],
                 np.abs(Y[:len(Y)//2]) / len(y))
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title(f'Output Spectrum (fs={fs_out} Hz)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex11_rational_conversion.png', dpi=100)
    plt.close()

    nyquist_out = fs_out / 2
    print(f"\n(d) Nyquist frequency at output: {nyquist_out} Hz")
    print(f"    Tones that should survive (< {nyquist_out} Hz):")
    for f in freqs:
        survives = f < nyquist_out
        print(f"      {f} Hz: {'SURVIVES' if survives else 'ATTENUATED'}")
    print("    Plot saved: ex11_rational_conversion.png")


# === Exercise 5: Multistage Decimation Optimization ===
# Problem: Decimate by M=48. Find optimal factorization.

def exercise_5():
    """Multistage decimation optimization for M=48."""
    M = 48

    # (a) List factorizations
    def factorizations(n, min_factor=2):
        """Generate all factorizations of n into factors >= min_factor."""
        if n <= 1:
            return [[]]
        result = []
        for f in range(min_factor, n + 1):
            if n % f == 0:
                for rest in factorizations(n // f, f):
                    result.append([f] + rest)
        return result

    facts = factorizations(M)
    # Filter to 2-4 stages
    facts_2_4 = [f for f in facts if 2 <= len(f) <= 4]

    print(f"(a) Factorizations of {M} with 2-4 stages:")
    for f in facts_2_4:
        print(f"    {' x '.join(map(str, f))}")

    # (b) Estimate MACs per output sample
    # Assume filter order = 10 * M_stage for each stage
    print(f"\n(b) Estimated MACs per output sample:")
    results = []
    for f in facts_2_4:
        total_macs = 0
        decimation_so_far = 1
        for stage, m in enumerate(f):
            filter_order = 10 * m
            # At this stage, input rate is original / decimation_so_far
            # MACs per output sample = filter_order / m (polyphase)
            # But relative to final output, multiply by product of remaining decimations
            remaining = 1
            for k in range(stage + 1, len(f)):
                remaining *= f[k]
            macs_this_stage = filter_order  # per input sample at this stage
            # Per final output sample
            macs_per_output = macs_this_stage / decimation_so_far
            total_macs += macs_per_output
            decimation_so_far *= m

        results.append((f, total_macs))
        print(f"    {' x '.join(map(str, f)):>15}: {total_macs:.1f} MACs/output")

    # (c) Optimal factorization
    optimal = min(results, key=lambda x: x[1])
    print(f"\n(c) Optimal factorization: {' x '.join(map(str, optimal[0]))} ({optimal[1]:.1f} MACs)")

    # Implement and verify
    np.random.seed(42)
    N = 48000
    x = np.random.randn(N)

    # Multistage
    y_multi = x.copy()
    for m in optimal[0]:
        y_multi = sig.decimate(y_multi, m, ftype='fir')

    # Single stage
    y_single = sig.decimate(x, M, ftype='fir')

    min_len = min(len(y_multi), len(y_single))
    error = np.max(np.abs(y_multi[:min_len] - y_single[:min_len]))
    print(f"    Max error vs single-stage: {error:.4f}")

    # (d) Timing comparison
    x_test = np.random.randn(480000)
    n_runs = 10

    start = time.time()
    for _ in range(n_runs):
        sig.decimate(x_test, M, ftype='fir')
    t_single = (time.time() - start) / n_runs

    start = time.time()
    for _ in range(n_runs):
        y = x_test.copy()
        for m in optimal[0]:
            y = sig.decimate(y, m, ftype='fir')
    t_multi = (time.time() - start) / n_runs

    print(f"\n(d) Timing ({n_runs} runs, {len(x_test)} samples):")
    print(f"    Single stage: {t_single*1000:.1f} ms")
    print(f"    Multistage:   {t_multi*1000:.1f} ms")
    print(f"    Speedup:      {t_single/t_multi:.2f}x")


# === Exercise 6: QMF Filter Bank ===
# Problem: Design a two-channel QMF filter bank.

def exercise_6():
    """Two-channel QMF filter bank for audio processing."""
    # (a) Design lowpass prototype
    N_taps = 32
    h_lp = sig.firwin(N_taps, 0.5, window='hamming')

    # (b) Construct highpass: h_hp[n] = (-1)^n * h_lp[n]
    n = np.arange(N_taps)
    h_hp = h_lp * ((-1) ** n)

    # Synthesis filters (time-reversed for perfect reconstruction approximation)
    g_lp = h_lp[::-1]
    g_hp = h_hp[::-1]

    print(f"(a-b) QMF filter bank:")
    print(f"    Lowpass prototype: {N_taps} taps")
    print(f"    Highpass: modulated version")

    # (c) Test near-perfect reconstruction
    np.random.seed(42)
    N = 2048
    # Music-like signal: harmonics
    fs = 16000
    t = np.arange(N) / fs
    x = sum(0.5**k * np.sin(2*np.pi*(220*k)*t) for k in range(1, 10))

    # Analysis
    x_lp = sig.lfilter(h_lp, 1, x)
    x_hp = sig.lfilter(h_hp, 1, x)

    # Downsample
    x_lp_down = x_lp[::2]
    x_hp_down = x_hp[::2]

    # Upsample
    x_lp_up = np.zeros(N)
    x_lp_up[::2] = x_lp_down
    x_hp_up = np.zeros(N)
    x_hp_up[::2] = x_hp_down

    # Synthesis
    y_lp = sig.lfilter(g_lp, 1, x_lp_up) * 2
    y_hp = sig.lfilter(g_hp, 1, x_hp_up) * 2
    y = y_lp + y_hp

    # Align (account for delay)
    delay = N_taps - 1
    recon_error = np.max(np.abs(x[delay:N-delay] - y[2*delay:N]))
    print(f"\n(c) Reconstruction error: {recon_error:.6f}")

    # (d) Reconstruction error vs filter order
    orders = [8, 16, 32, 64, 128]
    errors = []

    for order in orders:
        h = sig.firwin(order, 0.5, window='hamming')
        h_h = h * ((-1) ** np.arange(order))
        g_l = h[::-1]
        g_h = h_h[::-1]

        xl = sig.lfilter(h, 1, x)[::2]
        xh = sig.lfilter(h_h, 1, x)[::2]

        xlu = np.zeros(N)
        xlu[::2] = xl
        xhu = np.zeros(N)
        xhu[::2] = xh

        yl = sig.lfilter(g_l, 1, xlu) * 2
        yh = sig.lfilter(g_h, 1, xhu) * 2
        yr = yl + yh

        d = order - 1
        safe = min(N - 2*d, len(yr) - 2*d)
        if safe > 0:
            err = np.max(np.abs(x[d:d+safe] - yr[2*d:2*d+safe]))
        else:
            err = float('inf')
        errors.append(err)

    print(f"\n(d) Reconstruction error vs filter order:")
    for order, err in zip(orders, errors):
        print(f"    {order:>4} taps: error = {err:.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(orders, errors, 'bo-')
    ax.set_xlabel('Filter Order (taps)')
    ax.set_ylabel('Max Reconstruction Error')
    ax.set_title('QMF Reconstruction Error vs Filter Order')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex11_qmf.png', dpi=100)
    plt.close()

    # (e) Subband coding: quantize LP to 16 bits, HP to 8 bits
    x_lp_16 = np.round(x_lp_down * 32767) / 32767
    x_hp_8 = np.round(x_hp_down * 127) / 127

    xlu_q = np.zeros(N)
    xlu_q[::2] = x_lp_16
    xhu_q = np.zeros(N)
    xhu_q[::2] = x_hp_8

    y_coded = sig.lfilter(g_lp, 1, xlu_q) * 2 + sig.lfilter(g_hp, 1, xhu_q) * 2
    coding_snr = 10 * np.log10(np.sum(x[delay:]**2) / (np.sum((x[delay:N-delay] - y_coded[2*delay:N])**2) + 1e-15))

    print(f"\n(e) Subband coding SNR: {coding_snr:.1f} dB")
    print(f"    (LP: 16-bit, HP: 8-bit quantization)")
    print("    Plot saved: ex11_qmf.png")


# === Exercise 7: CIC Filter ===
# Problem: Implement CIC decimation filter.

def exercise_7():
    """Cascaded Integrator-Comb (CIC) filter implementation."""

    def cic_decimate(x, M, K=1):
        """
        CIC decimation filter.

        Parameters
        ----------
        x : ndarray
            Input signal
        M : int
            Decimation factor
        K : int
            Number of cascaded stages

        Returns
        -------
        y : ndarray
            Decimated output
        """
        signal = x.copy().astype(float)

        # K stages of integrator-comb
        for _ in range(K):
            # Comb filter: y[n] = x[n] - x[n-M]
            comb_out = np.zeros_like(signal)
            comb_out[:M] = signal[:M]
            comb_out[M:] = signal[M:] - signal[:-M]

            # Integrator: y[n] = x[n] + y[n-1]
            integrator_out = np.cumsum(comb_out)

            signal = integrator_out

        # Downsample
        y = signal[::M] / (M ** K)
        return y

    # (a) Single-stage CIC
    np.random.seed(42)
    N = 4096
    x = np.random.randn(N)
    M = 8
    y_cic = cic_decimate(x, M, K=1)
    print(f"(a) Single-stage CIC filter (M={M}):")
    print(f"    Input length: {N}, Output length: {len(y_cic)}")

    # (b) Frequency response: H(z) = (1/M)(1 - z^{-M})/(1 - z^{-1})
    print(f"\n(b) CIC frequency response: H(z) = (1/M)(1-z^{{-M}})/(1-z^{{-1}})")
    print(f"    Zeros at z = e^{{j*2*pi*k/M}} for k=1,...,M-1")

    # (c) Plot magnitude response for different M
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    M_values = [8, 16, 32]

    for M_val in M_values:
        # CIC frequency response
        w = np.linspace(0.001, np.pi, 4096)
        H_cic = np.sin(M_val * w / 2) / (M_val * np.sin(w / 2))
        mag_db = 20 * np.log10(np.abs(H_cic) + 1e-15)

        axes[0].plot(w / np.pi, mag_db, label=f'M={M_val}')

    # Ideal lowpass for reference
    axes[0].axhline(-3, color='r', linestyle=':', alpha=0.3)
    axes[0].set_xlabel('Normalized Frequency (x pi)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('CIC Filter Frequency Response (K=1)')
    axes[0].set_ylim([-60, 5])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (d) Multi-stage CIC
    M_fixed = 16
    K_values = [1, 2, 3, 4]
    w = np.linspace(0.001, np.pi, 4096)

    for K in K_values:
        H_single = np.sin(M_fixed * w / 2) / (M_fixed * np.sin(w / 2))
        H_multi = np.abs(H_single) ** K
        mag_db = 20 * np.log10(H_multi + 1e-15)
        axes[1].plot(w / np.pi, mag_db, label=f'K={K}')

    axes[1].set_xlabel('Normalized Frequency (x pi)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_title(f'Multi-stage CIC Filter (M={M_fixed})')
    axes[1].set_ylim([-100, 5])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex11_cic.png', dpi=100)
    plt.close()

    # (e) Sigma-delta ADC decimation chain (64x oversampling)
    M_total = 64
    K_cic = 4

    # Simulate oversampled sigma-delta output (1-bit quantized noise-shaped)
    N_oversamp = 64000
    x_sd = np.sign(np.random.randn(N_oversamp))  # Simplified 1-bit stream

    y_decimated = cic_decimate(x_sd, M_total, K=K_cic)

    print(f"\n(e) Sigma-delta ADC decimation:")
    print(f"    Oversample rate: {M_total}x")
    print(f"    CIC stages: K={K_cic}")
    print(f"    Input: {N_oversamp} samples -> Output: {len(y_decimated)} samples")
    print(f"    Stopband attenuation (K={K_cic}, M={M_total}): "
          f"~{K_cic * 13:.0f} dB at first null")
    print("    Plot saved: ex11_cic.png")


if __name__ == "__main__":
    print("=== Exercise 1: Downsampling Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: Interpolation Quality ===")
    exercise_2()
    print("\n=== Exercise 3: Polyphase Implementation ===")
    exercise_3()
    print("\n=== Exercise 4: Rational Rate Conversion ===")
    exercise_4()
    print("\n=== Exercise 5: Multistage Decimation Optimization ===")
    exercise_5()
    print("\n=== Exercise 6: QMF Filter Bank ===")
    exercise_6()
    print("\n=== Exercise 7: CIC Filter ===")
    exercise_7()
    print("\nAll exercises completed!")
