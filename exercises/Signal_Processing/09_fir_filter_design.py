"""
Exercises for Lesson 09: FIR Filter Design
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import fft, ifft
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Window Selection ===
# Problem: Design a lowpass FIR filter with fs=10000 Hz, fp=1500 Hz,
#          fstop=2000 Hz, min stopband attenuation 60 dB.

def exercise_1():
    """Window selection for lowpass FIR filter design."""
    fs = 10000
    fp = 1500
    fstop = 2000
    atten_db = 60

    # Transition width
    delta_f = fstop - fp
    delta_omega = 2 * np.pi * delta_f / fs
    print(f"Transition width: {delta_f} Hz ({delta_omega:.4f} rad/sample)")

    # (a) Window types that can meet 60 dB stopband attenuation
    # Hamming: ~53 dB -> NO
    # Blackman: ~74 dB -> YES
    # Kaiser (beta chosen): YES
    windows_info = {
        'hamming': {'atten': 53, 'mainlobe': 8 * np.pi},
        'blackman': {'atten': 74, 'mainlobe': 12 * np.pi},
        'kaiser': {'atten': 60, 'mainlobe': None},  # adjustable
    }

    print("\n(a) Window suitability for 60 dB stopband attenuation:")
    viable = []
    for name, info in windows_info.items():
        suitable = info['atten'] >= atten_db
        print(f"    {name}: ~{info['atten']} dB -> {'YES' if suitable else 'NO'}")
        if suitable:
            viable.append(name)

    # (b) Compute required filter order for viable windows
    print(f"\n(b) Required filter orders:")
    for name in viable:
        if name == 'blackman':
            # Blackman: mainlobe width ~12*pi/M -> transition width
            M = int(np.ceil(12 * np.pi / delta_omega))
            if M % 2 == 0:
                M += 1  # odd length for Type I
        elif name == 'kaiser':
            M, beta = sig.kaiserord(atten_db, delta_omega / np.pi)
            if M % 2 == 0:
                M += 1
        print(f"    {name}: M = {M}")

    # (c) Design with scipy and verify
    print(f"\n(c) Filter design and verification:")
    cutoff = (fp + fstop) / 2
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for name in viable:
        if name == 'blackman':
            M = int(np.ceil(12 * np.pi / delta_omega))
            if M % 2 == 0:
                M += 1
            h = sig.firwin(M, cutoff, fs=fs, window='blackman')
        elif name == 'kaiser':
            M, beta = sig.kaiserord(atten_db, delta_omega / np.pi)
            if M % 2 == 0:
                M += 1
            h = sig.firwin(M, cutoff, fs=fs, window=('kaiser', beta))

        w, H = sig.freqz(h, 1, worN=4096, fs=fs)
        mag_db = 20 * np.log10(np.abs(H) + 1e-15)

        # Measure actual specs
        passband_idx = w <= fp
        stopband_idx = w >= fstop
        passband_ripple = np.max(mag_db[passband_idx]) - np.min(mag_db[passband_idx])
        actual_atten = -np.max(mag_db[stopband_idx])

        print(f"    {name}: order={M}, passband ripple={passband_ripple:.3f} dB, "
              f"stopband atten={actual_atten:.1f} dB, "
              f"meets spec: {actual_atten >= atten_db}")

        axes[0].plot(w, mag_db, label=f'{name} (N={M})')
        axes[1].plot(w[passband_idx], mag_db[passband_idx], label=f'{name}')

    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('FIR Filter Comparison - Full Response')
    axes[0].axhline(-60, color='r', linestyle='--', alpha=0.5, label='-60 dB')
    axes[0].axvline(fp, color='g', linestyle=':', alpha=0.5)
    axes[0].axvline(fstop, color='g', linestyle=':', alpha=0.5)
    axes[0].set_ylim([-100, 5])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_title('Passband Detail')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex09_window_selection.png', dpi=100)
    plt.close()
    print("    Plot saved: ex09_window_selection.png")


# === Exercise 2: Kaiser Window Design ===
# Problem: Passband 0-3 kHz (ripple <= 0.1 dB), stopband >= 4 kHz (atten >= 50 dB),
#          fs = 20 kHz.

def exercise_2():
    """Kaiser window filter design with specification verification."""
    fs = 20000
    fp = 3000
    fstop = 4000
    passband_ripple_db = 0.1
    stopband_atten_db = 50

    # (a) Kaiser window parameters
    # delta_p from passband ripple
    delta_p = 1 - 10 ** (-passband_ripple_db / 20)
    delta_s = 10 ** (-stopband_atten_db / 20)
    delta = min(delta_p, delta_s)
    A_s = -20 * np.log10(delta)
    print(f"(a) delta_p={delta_p:.6f}, delta_s={delta_s:.6f}")
    print(f"    A_s = {A_s:.2f} dB")

    # Kaiser beta
    if A_s > 50:
        beta = 0.1102 * (A_s - 8.7)
    elif A_s >= 21:
        beta = 0.5842 * (A_s - 21) ** 0.4 + 0.07886 * (A_s - 21)
    else:
        beta = 0.0
    print(f"    Kaiser beta = {beta:.4f}")

    # (b) Minimum filter order
    delta_f = fstop - fp
    M = int(np.ceil((A_s - 7.95) / (2.285 * 2 * np.pi * delta_f / fs)))
    if M % 2 == 0:
        M += 1
    print(f"\n(b) Minimum filter order M = {M}")

    # (c) Design and plot
    cutoff = (fp + fstop) / 2
    h = sig.firwin(M, cutoff, fs=fs, window=('kaiser', beta))

    w, H = sig.freqz(h, 1, worN=8192, fs=fs)
    mag_db = 20 * np.log10(np.abs(H) + 1e-15)
    phase = np.unwrap(np.angle(H))
    _, gd = sig.group_delay((h, 1), w=8192, fs=fs)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(w, mag_db)
    axes[0].axhline(-stopband_atten_db, color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title(f'Kaiser Window FIR Filter (N={M}, beta={beta:.2f})')
    axes[0].set_ylim([-80, 5])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(w, phase)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Phase (rad)')
    axes[1].set_title('Phase Response')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(w, gd)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Group Delay (samples)')
    axes[2].set_title('Group Delay')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex09_kaiser_design.png', dpi=100)
    plt.close()

    # (d) Verify specifications
    passband_idx = w <= fp
    stopband_idx = w >= fstop

    actual_ripple = np.max(mag_db[passband_idx]) - np.min(mag_db[passband_idx])
    actual_atten = -np.max(mag_db[stopband_idx])

    print(f"\n(d) Verification:")
    print(f"    Passband ripple: {actual_ripple:.4f} dB (spec: <= {passband_ripple_db} dB) "
          f"{'PASS' if actual_ripple <= passband_ripple_db else 'FAIL'}")
    print(f"    Stopband atten:  {actual_atten:.1f} dB (spec: >= {stopband_atten_db} dB) "
          f"{'PASS' if actual_atten >= stopband_atten_db else 'FAIL'}")
    print("    Plot saved: ex09_kaiser_design.png")


# === Exercise 3: Parks-McClellan Bandpass Filter ===
# Problem: fs=44100, stopband1=0-800, passband=1000-3000, stopband2=3500-22050 Hz

def exercise_3():
    """Parks-McClellan bandpass filter design."""
    fs = 44100

    # Frequency bands (normalized to [0, 0.5])
    bands = [0, 800, 1000, 3000, 3500, fs/2]
    desired = [0, 0, 1, 1, 0, 0]

    # (a) Weights for equal ripple
    # Passband ripple 0.5 dB -> delta_p ~ 0.057
    # Stopband attenuation 40 dB -> delta_s = 0.01
    delta_p = 1 - 10**(-0.5/20)
    delta_s = 10**(-40/20)
    weight_ratio = delta_s / delta_p
    weights = [1.0, weight_ratio, 1.0]  # stopband, passband, stopband
    print(f"(a) delta_p={delta_p:.4f}, delta_s={delta_s:.4f}")
    print(f"    Weight ratio (stopband/passband): {1/weight_ratio:.2f}")
    print(f"    Weights: {weights}")

    # (b) Determine minimum filter order
    # Start from estimate and iterate
    for N in range(30, 200, 2):
        try:
            h = sig.remez(N + 1, bands, [0, 1, 0], weight=weights, fs=fs)
            w, H = sig.freqz(h, 1, worN=4096, fs=fs)
            mag_db = 20 * np.log10(np.abs(H) + 1e-15)

            sb1 = w <= 800
            pb = (w >= 1000) & (w <= 3000)
            sb2 = w >= 3500

            atten1 = -np.max(mag_db[sb1]) if np.any(sb1) else 0
            atten2 = -np.max(mag_db[sb2]) if np.any(sb2) else 0
            ripple = np.max(mag_db[pb]) - np.min(mag_db[pb])

            if atten1 >= 40 and atten2 >= 40 and ripple <= 0.5:
                print(f"\n(b) Minimum filter order: N = {N}")
                break
        except Exception:
            continue
    else:
        N = 80
        h = sig.remez(N + 1, bands, [0, 1, 0], weight=weights, fs=fs)
        print(f"\n(b) Using filter order N = {N}")

    # (c) Plot magnitude response
    w, H = sig.freqz(h, 1, worN=8192, fs=fs)
    mag_db = 20 * np.log10(np.abs(H) + 1e-15)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(w, mag_db)
    ax.axhline(-40, color='r', linestyle='--', alpha=0.5, label='-40 dB')
    ax.axvline(800, color='g', linestyle=':', alpha=0.3)
    ax.axvline(1000, color='g', linestyle=':', alpha=0.3)
    ax.axvline(3000, color='g', linestyle=':', alpha=0.3)
    ax.axvline(3500, color='g', linestyle=':', alpha=0.3)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'Parks-McClellan Bandpass Filter (N={N})')
    ax.set_ylim([-80, 5])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex09_parks_mcclellan.png', dpi=100)
    plt.close()

    # (d) Apply to chirp signal
    duration = 0.5
    t = np.arange(0, duration, 1/fs)
    chirp_sig = sig.chirp(t, f0=100, t1=duration, f1=10000, method='linear')

    filtered = sig.lfilter(h, 1, chirp_sig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for i, (data, title) in enumerate([(chirp_sig, 'Before filtering'),
                                        (filtered, 'After filtering')]):
        axes[i].specgram(data, NFFT=256, Fs=fs, noverlap=200, cmap='inferno')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Frequency (Hz)')
        axes[i].set_title(title)
        axes[i].set_ylim([0, 10000])

    plt.tight_layout()
    plt.savefig('ex09_chirp_filtered.png', dpi=100)
    plt.close()
    print("(c-d) Plots saved: ex09_parks_mcclellan.png, ex09_chirp_filtered.png")


# === Exercise 4: Overlap-Add Implementation ===
# Problem: Implement a streaming FIR filter using overlap-add.

def exercise_4():
    """Overlap-add streaming FIR filter implementation."""

    class StreamingFIRFilter:
        """Block-based FIR filter using overlap-add method."""

        def __init__(self, h, block_size):
            self.h = h
            self.M = len(h)
            self.block_size = block_size
            self.fft_size = 1
            while self.fft_size < block_size + self.M - 1:
                self.fft_size *= 2
            self.H_fft = fft(h, n=self.fft_size)
            self.overlap = np.zeros(self.M - 1)

        def process_block(self, block):
            """Process one block of samples."""
            X_fft = fft(block, n=self.fft_size)
            Y_fft = X_fft * self.H_fft
            y_full = np.real(ifft(Y_fft))

            # Add overlap from previous block
            y_out = y_full[:self.block_size].copy()
            y_out[:len(self.overlap)] += self.overlap

            # Save overlap for next block
            self.overlap = y_full[self.block_size:self.block_size + self.M - 1]

            return y_out

    # Create a test filter (200-tap lowpass)
    M_filter = 200
    h = sig.firwin(M_filter, 0.3)

    # Generate test signal
    np.random.seed(42)
    fs = 44100
    duration = 10
    N = fs * duration
    x = np.random.randn(N)

    # (a-c) Test with various block sizes and verify output
    print("Testing StreamingFIRFilter (overlap-add):")
    y_reference = np.convolve(x, h, mode='full')[:N]

    block_sizes = [64, 256, 1024, 4096]
    times = []

    for bs in block_sizes:
        filt = StreamingFIRFilter(h, bs)
        y_blocks = []

        start = time.time()
        for i in range(0, N, bs):
            block = x[i:i+bs]
            if len(block) < bs:
                block = np.pad(block, (0, bs - len(block)))
            y_block = filt.process_block(block)
            y_blocks.append(y_block)
        elapsed = time.time() - start

        y_ola = np.concatenate(y_blocks)[:N]
        error = np.max(np.abs(y_ola - y_reference))

        times.append(elapsed)
        print(f"    Block size {bs:5d}: time={elapsed:.4f}s, max error={error:.2e}")

    # (d) Plot processing time vs block size
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(bs) for bs in block_sizes], times, color='steelblue')
    ax.set_xlabel('Block Size (samples)')
    ax.set_ylabel('Processing Time (s)')
    ax.set_title(f'Overlap-Add: Processing Time vs Block Size\n({M_filter}-tap filter, {duration}s signal at {fs} Hz)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('ex09_overlap_add.png', dpi=100)
    plt.close()

    # (e) Optimal block size
    optimal_idx = np.argmin(times)
    print(f"\n    Optimal block size: {block_sizes[optimal_idx]} samples")
    print("    Plot saved: ex09_overlap_add.png")


# === Exercise 5: Linear Phase Constraints ===
# Problem: Prove Type II filter has H(pi)=0; compare Type I and Type IV for highpass.

def exercise_5():
    """Linear phase FIR filter type analysis."""
    # (a) Type II: odd order, symmetric -> H(e^j*pi) = 0
    print("(a) Type II FIR filter: even number of taps (odd order), symmetric.")
    print("    For symmetric h[n] with even M:")
    print("    H(e^j*pi) = sum h[n](-1)^n = 0 due to symmetry pairing.")
    print("    This makes Type II unsuitable for highpass design.\n")

    # Numerical verification
    M_type2 = 20  # even number of taps
    h_type2 = sig.firwin(M_type2, 0.5, window='hamming')  # lowpass
    w, H = sig.freqz(h_type2, 1, worN=4096)
    print(f"    Type II (M={M_type2}): |H(pi)| = {np.abs(H[-1]):.6e}")

    # (b) Highpass filter comparison: Type I vs Type IV
    cutoff = 0.6  # normalized frequency (0 to 1 maps to 0 to pi)

    # Type I: odd number of taps, symmetric
    M1 = 41
    h_type1 = sig.firwin(M1, cutoff, pass_zero=False, window='hamming')
    w1, H1 = sig.freqz(h_type1, 1, worN=4096)

    # Type IV: even number of taps, antisymmetric
    M4 = 40
    h_type4 = sig.firwin(M4, cutoff, pass_zero=False, window='hamming')
    w4, H4 = sig.freqz(h_type4, 1, worN=4096)

    print(f"\n(b) Highpass filter comparison (cutoff = {cutoff}*pi):")
    print(f"    Type I  (M={M1}): |H(pi)| = {np.abs(H1[-1]):.4f}")
    print(f"    Type IV (M={M4}): |H(pi)| = {np.abs(H4[-1]):.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(w1/np.pi, 20*np.log10(np.abs(H1)+1e-15), label=f'Type I (N={M1})')
    ax.plot(w4/np.pi, 20*np.log10(np.abs(H4)+1e-15), label=f'Type IV (N={M4})')
    ax.set_xlabel('Normalized Frequency (x pi rad/sample)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Highpass FIR Filter: Type I vs Type IV')
    ax.set_ylim([-80, 5])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex09_linear_phase.png', dpi=100)
    plt.close()

    # (c) Hilbert transform filter (Type III)
    M_hilbert = 31
    n = np.arange(M_hilbert)
    center = (M_hilbert - 1) / 2
    h_hilbert = np.zeros(M_hilbert)
    for i in range(M_hilbert):
        if i != center:
            k = i - center
            if k % 2 != 0:
                h_hilbert[i] = 2 / (np.pi * k)
    # Apply Hamming window
    h_hilbert *= np.hamming(M_hilbert)

    w_h, H_h = sig.freqz(h_hilbert, 1, worN=4096)
    print(f"\n(c) Hilbert transform (Type III, M={M_hilbert}):")
    print(f"    |H(0)|  = {np.abs(H_h[0]):.6f} (should be ~0)")
    print(f"    |H(pi)| = {np.abs(H_h[-1]):.6f} (should be ~0)")
    print("    Type III: zero at both omega=0 and omega=pi")
    print("    Plot saved: ex09_linear_phase.png")


# === Exercise 6: Comparison Study ===
# Problem: Compare Hamming, Kaiser, Parks-McClellan for same spec.

def exercise_6():
    """Comparison of three FIR design methods for same specification."""
    fs = 16000
    fp = 2000
    fstop = 2500
    A_s = 50  # dB

    cutoff = (fp + fstop) / 2
    delta_f = fstop - fp
    delta_omega = 2 * np.pi * delta_f / fs

    results = {}

    # (a) Hamming window
    M_ham = int(np.ceil(8 * np.pi / delta_omega))
    if M_ham % 2 == 0:
        M_ham += 1
    h_ham = sig.firwin(M_ham, cutoff, fs=fs, window='hamming')
    results['Hamming'] = (h_ham, M_ham)

    # Kaiser window
    M_kai, beta = sig.kaiserord(A_s, delta_f / (fs/2))
    if M_kai % 2 == 0:
        M_kai += 1
    h_kai = sig.firwin(M_kai, cutoff, fs=fs, window=('kaiser', beta))
    results['Kaiser'] = (h_kai, M_kai)

    # Parks-McClellan
    delta_s = 10 ** (-A_s / 20)
    for N_pm in range(20, 200, 2):
        try:
            h_pm = sig.remez(N_pm + 1, [0, fp, fstop, fs/2], [1, 0], fs=fs)
            w_t, H_t = sig.freqz(h_pm, 1, worN=4096, fs=fs)
            mag_t = 20 * np.log10(np.abs(H_t) + 1e-15)
            sb_idx = w_t >= fstop
            if -np.max(mag_t[sb_idx]) >= A_s:
                results['Parks-McClellan'] = (h_pm, N_pm)
                break
        except Exception:
            continue

    # (b) Compare metrics
    print(f"Specification: fs={fs}, fp={fp}, fstop={fstop}, As={A_s} dB\n")
    print(f"{'Method':<18} {'Order':>6} {'PB Ripple (dB)':>15} {'SB Atten (dB)':>15} {'Trans BW (Hz)':>14}")
    print("-" * 72)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for name, (h, M) in results.items():
        w, H = sig.freqz(h, 1, worN=8192, fs=fs)
        mag_db = 20 * np.log10(np.abs(H) + 1e-15)

        pb_idx = w <= fp
        sb_idx = w >= fstop
        ripple = np.max(mag_db[pb_idx]) - np.min(mag_db[pb_idx])
        atten = -np.max(mag_db[sb_idx])

        # Transition width (actual -3 dB to -As dB)
        idx_3db = np.argmin(np.abs(mag_db + 3))
        tw = w[np.argmin(np.abs(mag_db + A_s))] - w[idx_3db]

        print(f"{name:<18} {M:>6} {ripple:>15.4f} {atten:>15.1f} {tw:>14.1f}")

        # (c) Overlay plots
        axes[0, 0].plot(w, mag_db, label=name)
        axes[0, 1].plot(w[pb_idx], mag_db[pb_idx], label=name)
        axes[1, 0].plot(h, label=name, alpha=0.7)

        # Pole-zero
        zeros_h = np.roots(h)
        axes[1, 1].scatter(np.real(zeros_h), np.imag(zeros_h), s=10, alpha=0.3, label=name)

    axes[0, 0].set_title('Magnitude Responses')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].set_ylim([-80, 5])
    axes[0, 0].axhline(-A_s, color='r', linestyle='--', alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Passband Detail')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Impulse Responses')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    theta = np.linspace(0, 2 * np.pi, 100)
    axes[1, 1].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    axes[1, 1].set_title('Zero Locations')
    axes[1, 1].set_xlabel('Real')
    axes[1, 1].set_ylabel('Imaginary')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex09_comparison.png', dpi=100)
    plt.close()

    # (d) Best method
    print(f"\n(d) Parks-McClellan generally provides the best result for lowest order")
    print(f"    because equiripple design distributes error optimally.")
    print("    Plot saved: ex09_comparison.png")


# === Exercise 7: Minimum-Phase FIR Filter ===
# Problem: Convert linear-phase FIR to minimum-phase via cepstral method.

def exercise_7():
    """Minimum-phase FIR filter from linear-phase using cepstrum."""
    # (a) Linear-phase lowpass
    M = 41
    h_lp = sig.firwin(M, 0.4, window='hamming')
    print(f"(a) Linear-phase FIR: order={M-1}, {M} taps")

    # (b) Convert to minimum-phase using cepstral method
    N_fft = 2048
    H_mag = np.abs(fft(h_lp, n=N_fft))
    H_mag = np.maximum(H_mag, 1e-10)  # avoid log(0)

    # Cepstrum
    log_H = np.log(H_mag)
    cepstrum = np.real(ifft(log_H))

    # Minimum-phase cepstrum: causal part only, double it (except DC and Nyquist)
    cep_min = np.zeros(N_fft)
    cep_min[0] = cepstrum[0]
    cep_min[1:N_fft//2] = 2 * cepstrum[1:N_fft//2]
    cep_min[N_fft//2] = cepstrum[N_fft//2]

    # Minimum-phase spectrum
    H_min = np.exp(fft(cep_min))
    h_min = np.real(ifft(H_min))[:M]

    print(f"(b) Minimum-phase FIR: {M} taps (cepstral method)")

    # (c) Compare
    w, H_linear = sig.freqz(h_lp, 1, worN=4096)
    _, H_minimum = sig.freqz(h_min, 1, worN=4096)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Magnitude
    axes[0].plot(w/np.pi, 20*np.log10(np.abs(H_linear)+1e-15), label='Linear phase')
    axes[0].plot(w/np.pi, 20*np.log10(np.abs(H_minimum)+1e-15), '--', label='Minimum phase')
    axes[0].set_xlabel('Normalized Frequency (x pi)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('Magnitude Response Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Phase
    axes[1].plot(w/np.pi, np.unwrap(np.angle(H_linear)), label='Linear phase')
    axes[1].plot(w/np.pi, np.unwrap(np.angle(H_minimum)), '--', label='Minimum phase')
    axes[1].set_xlabel('Normalized Frequency (x pi)')
    axes[1].set_ylabel('Phase (rad)')
    axes[1].set_title('Phase Response Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Group delay
    _, gd_lin = sig.group_delay((h_lp, 1), w=4096)
    _, gd_min = sig.group_delay((h_min, 1), w=4096)
    axes[2].plot(w/np.pi, gd_lin, label='Linear phase')
    axes[2].plot(w/np.pi, gd_min, '--', label='Minimum phase')
    axes[2].set_xlabel('Normalized Frequency (x pi)')
    axes[2].set_ylabel('Group Delay (samples)')
    axes[2].set_title('Group Delay Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex09_min_phase.png', dpi=100)
    plt.close()

    # (d) Discussion
    print(f"\n(c) Magnitude responses are nearly identical.")
    print(f"    Linear phase: constant group delay = {(M-1)/2} samples")
    print(f"    Minimum phase: lower group delay, energy concentrated at start")
    print(f"\n(d) Prefer minimum-phase when:")
    print(f"    - Low latency is critical (real-time audio)")
    print(f"    - Phase distortion is acceptable (human hearing less sensitive)")
    print(f"    Prefer linear-phase when:")
    print(f"    - Waveform fidelity matters (medical signals, seismology)")
    print(f"    - Multi-channel coherent processing")
    print("    Plot saved: ex09_min_phase.png")


if __name__ == "__main__":
    print("=== Exercise 1: Window Selection ===")
    exercise_1()
    print("\n=== Exercise 2: Kaiser Window Design ===")
    exercise_2()
    print("\n=== Exercise 3: Parks-McClellan Bandpass Filter ===")
    exercise_3()
    print("\n=== Exercise 4: Overlap-Add Implementation ===")
    exercise_4()
    print("\n=== Exercise 5: Linear Phase Constraints ===")
    exercise_5()
    print("\n=== Exercise 6: Comparison Study ===")
    exercise_6()
    print("\n=== Exercise 7: Minimum-Phase FIR Filter ===")
    exercise_7()
    print("\nAll exercises completed!")
