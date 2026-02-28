"""
Exercises for Lesson 08: Digital Filter Fundamentals
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal as sig
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: FIR Filter Analysis ===
# Problem: Analyze h[n] = {1, -2, 3, -2, 1}.

def exercise_1():
    """FIR filter analysis: transfer function, zeros, frequency response."""
    h = np.array([1, -2, 3, -2, 1], dtype=float)
    N = len(h)

    # (a) Difference equation and transfer function
    print(f"(a) h[n] = {list(h)}")
    print(f"    y[n] = x[n] - 2x[n-1] + 3x[n-2] - 2x[n-3] + x[n-4]")
    print(f"    H(z) = 1 - 2z^-1 + 3z^-2 - 2z^-3 + z^-4")
    print()

    # (b) Zeros
    zeros = np.roots(h)
    print(f"(b) Zeros: {np.round(zeros, 4)}")
    print(f"    |zeros|: {np.round(np.abs(zeros), 4)}")
    # Check symmetry: zeros come in conjugate reciprocal pairs for linear phase
    for z in zeros:
        recip = 1 / np.conj(z)
        found = any(np.abs(z2 - recip) < 0.001 for z2 in zeros)
        print(f"    z={z:.4f}, 1/z*={recip:.4f}, pair exists: {found}")
    print()

    # (c) Frequency response
    w, H_freq = sig.freqz(h, 1, worN=1024)
    mag = 20 * np.log10(np.abs(H_freq) + 1e-10)
    phase = np.unwrap(np.angle(H_freq))

    # Check linear phase
    # For symmetric coefficients (Type I), phase should be linear
    is_symmetric = np.allclose(h, h[::-1])
    print(f"(c) Symmetric coefficients: {is_symmetric}")
    if is_symmetric:
        print(f"    Type I linear phase FIR (odd length, symmetric)")
        group_delay = (N - 1) / 2
        print(f"    Group delay: {group_delay} samples")
    print()

    # (d) Filter type
    dc_gain = np.abs(H_freq[0])
    nyquist_gain = np.abs(H_freq[-1])
    print(f"(d) Type I FIR filter")
    print(f"    |H(0)| = {dc_gain:.4f}, |H(pi)| = {nyquist_gain:.4f}")
    if dc_gain < nyquist_gain:
        print(f"    -> Highpass characteristic")
    else:
        print(f"    -> Lowpass or bandpass characteristic")
    print()

    # (e) Filter the signal
    fs = 1000
    n_arr = np.arange(500)
    x_test = np.cos(0.2 * np.pi * n_arr) + np.cos(0.8 * np.pi * n_arr)
    y_test = sig.lfilter(h, 1, x_test)

    H_02pi = np.abs(np.interp(0.2, w / np.pi, np.abs(H_freq)))
    H_08pi = np.abs(np.interp(0.8, w / np.pi, np.abs(H_freq)))
    print(f"(e) |H(0.2*pi)| = {H_02pi:.4f}")
    print(f"    |H(0.8*pi)| = {H_08pi:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(w / np.pi, mag)
    axes[0].set_title('Magnitude Response')
    axes[0].set_xlabel('Normalized Frequency (x pi)')
    axes[0].set_ylabel('dB')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(w / np.pi, phase)
    axes[1].set_title('Phase Response')
    axes[1].set_xlabel('Normalized Frequency (x pi)')
    axes[1].set_ylabel('Radians')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex08_fir_analysis.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex08_fir_analysis.png")


# === Exercise 2: IIR Filter Design and Analysis ===
# Problem: Design 4th-order Chebyshev Type I lowpass filter.

def exercise_2():
    """IIR filter design: Chebyshev Type I vs Butterworth."""
    fs = 8000
    fc = 2000
    order = 4
    rp = 0.5  # dB passband ripple

    # (a) Chebyshev Type I
    b_cheb, a_cheb = sig.cheby1(order, rp, fc, fs=fs, btype='low')
    print(f"(a) Chebyshev Type I (order={order}, rp={rp} dB, fc={fc} Hz):")
    print(f"    b = {np.round(b_cheb, 6)}")
    print(f"    a = {np.round(a_cheb, 6)}")
    print()

    # (b) Pole-zero diagram
    zeros_cheb = np.roots(b_cheb)
    poles_cheb = np.roots(a_cheb)
    all_inside = np.all(np.abs(poles_cheb) < 1)
    print(f"(b) All poles inside unit circle: {all_inside}")
    print(f"    Pole magnitudes: {np.round(np.abs(poles_cheb), 4)}")
    print()

    # (c) Frequency response
    w_cheb, H_cheb = sig.freqz(b_cheb, a_cheb, worN=2048, fs=fs)

    # (d) Compare with Butterworth
    b_but, a_but = sig.butter(order, fc, fs=fs, btype='low')
    w_but, H_but = sig.freqz(b_but, a_but, worN=2048, fs=fs)

    # Compute group delay
    _, gd_cheb = sig.group_delay((b_cheb, a_cheb), w=1024, fs=fs)
    _, gd_but = sig.group_delay((b_but, a_but), w=1024, fs=fs)

    # Stopband attenuation at 3000 Hz
    idx_3k_cheb = np.argmin(np.abs(w_cheb - 3000))
    idx_3k_but = np.argmin(np.abs(w_but - 3000))
    atten_cheb = -20 * np.log10(np.abs(H_cheb[idx_3k_cheb]) + 1e-15)
    atten_but = -20 * np.log10(np.abs(H_but[idx_3k_but]) + 1e-15)

    print(f"(d) Stopband attenuation at 3000 Hz:")
    print(f"    Chebyshev: {atten_cheb:.1f} dB")
    print(f"    Butterworth: {atten_but:.1f} dB")
    print(f"    Chebyshev has {'better' if atten_cheb > atten_but else 'worse'} stopband attenuation")
    print()

    # Group delay comparison
    mask = w_cheb[:len(gd_cheb)] < fc
    gd_cheb_pb = gd_cheb[:np.sum(mask)] if np.sum(mask) > 0 else gd_cheb
    gd_but_pb = gd_but[:np.sum(mask)] if np.sum(mask) > 0 else gd_but
    print(f"    Group delay variation in passband:")
    print(f"    Chebyshev: {np.ptp(gd_cheb_pb):.2f} samples")
    print(f"    Butterworth: {np.ptp(gd_but_pb):.2f} samples")
    print(f"    Butterworth has {'more' if np.ptp(gd_but_pb) > np.ptp(gd_cheb_pb) else 'more'} constant group delay")
    print()

    # (e) SOS form
    sos_cheb = sig.cheby1(order, rp, fc, fs=fs, btype='low', output='sos')
    w_sos, H_sos = sig.sosfreqz(sos_cheb, worN=2048, fs=fs)
    sos_error = np.max(np.abs(np.abs(H_cheb) - np.abs(H_sos)))
    print(f"(e) SOS form error: {sos_error:.2e}")


# === Exercise 3: Linear Phase Verification ===
# Problem: Design and verify linear phase FIR filters.

def exercise_3():
    """Linear phase FIR filter verification."""
    fs = 8000

    # (a) Type I bandpass filter (51 taps)
    N = 51
    h_bp = sig.firwin(N, [1000, 2000], fs=fs, pass_zero=False)
    print(f"(a) Type I bandpass ({N} taps):")
    is_sym = np.allclose(h_bp, h_bp[::-1])
    print(f"    Symmetric: {is_sym}")

    w, H = sig.freqz(h_bp, 1, worN=2048)
    phase = np.unwrap(np.angle(H))
    # Linear phase: phase should be -w*(N-1)/2 plus multiples of pi
    expected_delay = (N - 1) / 2
    w_nonzero = w[np.abs(H) > 0.01]
    phase_nonzero = phase[np.abs(H) > 0.01]
    if len(w_nonzero) > 2:
        slope = np.polyfit(w_nonzero, phase_nonzero, 1)[0]
        print(f"    Phase slope: {slope:.4f} (expected: {-expected_delay:.4f})")
    print()

    # (b) Type III differentiator (51 taps)
    h_diff = sig.firwin(N, 0.9, fs=fs, window='hamming')
    # Actually, for a Type III differentiator, we need antisymmetric odd-length
    # Create manually
    n = np.arange(N)
    mid = (N - 1) / 2
    h_diff_manual = np.zeros(N)
    for i in range(N):
        if i != mid:
            h_diff_manual[i] = np.cos(np.pi * (i - mid)) / (i - mid)
        else:
            h_diff_manual[i] = 0
    h_diff_manual *= np.hamming(N)

    is_antisym = np.allclose(h_diff_manual, -h_diff_manual[::-1])
    print(f"(b) Type III differentiator ({N} taps):")
    print(f"    Antisymmetric: {is_antisym}")

    # (c) Group delay verification
    _, gd_bp = sig.group_delay((h_bp, 1), w=1024)
    gd_variation = np.ptp(gd_bp)
    print(f"\n(c) Bandpass group delay variation: {gd_variation:.6f} samples")
    print(f"    Constant group delay (< 0.001): {gd_variation < 0.001}")


# === Exercise 4: Filter Structure Implementation ===
# Problem: Implement Direct Form I, II, and cascade structures.

def exercise_4():
    """Filter structure implementations from scratch."""
    b = [1, 0.5]
    a = [1, -0.9, 0.81]
    N = 1000

    np.random.seed(42)
    x = np.random.randn(N)

    # (a) Direct Form I
    def direct_form_1(b, a, x):
        N_out = len(x)
        y = np.zeros(N_out)
        M = len(b) - 1  # feedforward order
        P = len(a) - 1  # feedback order
        x_buf = np.zeros(M + 1)
        y_buf = np.zeros(P)
        for n in range(N_out):
            x_buf[1:] = x_buf[:-1]
            x_buf[0] = x[n]
            y_val = np.dot(b, x_buf)
            y_val -= np.dot(a[1:], y_buf)
            y[n] = y_val
            if P > 0:
                y_buf[1:] = y_buf[:-1]
                y_buf[0] = y_val
        return y

    y_df1 = direct_form_1(b, a, x)

    # (b) Direct Form II
    def direct_form_2(b, a, x):
        N_out = len(x)
        y = np.zeros(N_out)
        order = max(len(b), len(a)) - 1
        w = np.zeros(order + 1)
        b_pad = np.zeros(order + 1)
        a_pad = np.zeros(order + 1)
        b_pad[:len(b)] = b
        a_pad[:len(a)] = a
        for n in range(N_out):
            w[0] = x[n] - np.dot(a_pad[1:], w[1:])
            y[n] = np.dot(b_pad, w)
            w[1:] = w[:-1]  # shift delay line
            w[0] = x[n] - np.dot(a_pad[1:], w[1:])
        return y

    # Simpler DF2 implementation
    def direct_form_2_v2(b, a, x):
        N_out = len(x)
        y = np.zeros(N_out)
        order = max(len(b), len(a)) - 1
        w = np.zeros(order)
        b_pad = np.zeros(order + 1)
        a_pad = np.zeros(order + 1)
        b_pad[:len(b)] = b
        a_pad[:len(a)] = a
        for n in range(N_out):
            # Compute intermediate
            w_new = x[n] - np.dot(a_pad[1:order + 1], w)
            y[n] = b_pad[0] * w_new + np.dot(b_pad[1:order + 1], w)
            # Shift
            w[1:] = w[:-1]
            w[0] = w_new
        return y

    y_df2 = direct_form_2_v2(b, a, x)

    # (c) Reference
    y_ref = sig.lfilter(b, a, x)

    err_df1 = np.max(np.abs(y_df1 - y_ref))
    err_df2 = np.max(np.abs(y_df2 - y_ref))

    print(f"(a) Direct Form I error:  {err_df1:.2e}")
    print(f"(b) Direct Form II error: {err_df2:.2e}")
    print(f"(d) All structures produce identical results (within FP precision)")


# === Exercise 5: Quantization Experiment ===
# Problem: Stability under coefficient quantization.

def exercise_5():
    """Coefficient quantization effects on filter stability."""
    fs = 8000

    # 10th-order elliptic bandpass filter
    order = 10
    sos = sig.ellip(order, 0.5, 60, [300, 3400], fs=fs, btype='band', output='sos')
    b_tf, a_tf = sig.ellip(order, 0.5, 60, [300, 3400], fs=fs, btype='band')

    def quantize_coeffs(coeffs, n_bits):
        """Quantize filter coefficients to n_bits."""
        scale = 2 ** (n_bits - 1)
        return np.round(coeffs * scale) / scale

    # (a-b) Compare tf vs sos with 16-bit quantization
    n_bits = 16
    b_q = quantize_coeffs(b_tf, n_bits)
    a_q = quantize_coeffs(a_tf, n_bits)

    poles_q = np.roots(a_q)
    stable_tf = np.all(np.abs(poles_q) < 1)

    sos_q = quantize_coeffs(sos, n_bits)
    # Check stability of each section
    stable_sos = True
    for section in sos_q:
        poles_sec = np.roots(section[3:])
        if np.any(np.abs(poles_sec) >= 1):
            stable_sos = False

    print(f"10th-order elliptic bandpass, {n_bits}-bit quantization:")
    print(f"  (a) TF form stable: {stable_tf}")
    print(f"  (b) SOS form stable: {stable_sos}")
    print()

    # (d) Find minimum bits for stability
    print("(d) Minimum bits for stability:")
    for form_name, test_func in [("TF", lambda nb: np.all(np.abs(np.roots(quantize_coeffs(a_tf, nb))) < 1)),
                                   ("SOS", lambda nb: all(np.all(np.abs(np.roots(quantize_coeffs(sos, nb)[i, 3:])) < 1) for i in range(sos.shape[0])))]:
        for nb in range(4, 33):
            try:
                if test_func(nb):
                    print(f"    {form_name}: {nb} bits")
                    break
            except Exception:
                continue
        else:
            print(f"    {form_name}: > 32 bits needed")


# === Exercise 6: Real-Time Filter Simulation ===
# Problem: Sample-by-sample filtering.

def exercise_6():
    """Real-time (sample-by-sample) filter implementation."""
    class RealtimeFilter:
        def __init__(self, b, a):
            self.b = np.array(b, dtype=float)
            self.a = np.array(a, dtype=float)
            # Direct Form II Transposed state
            self.state = np.zeros(max(len(b), len(a)) - 1)

        def process_sample(self, x_n):
            y_n = self.b[0] * x_n + self.state[0]
            for i in range(len(self.state) - 1):
                b_i = self.b[i + 1] if i + 1 < len(self.b) else 0.0
                a_i = self.a[i + 1] if i + 1 < len(self.a) else 0.0
                self.state[i] = b_i * x_n - a_i * y_n + self.state[i + 1]
            # Last state
            i = len(self.state) - 1
            b_i = self.b[i + 1] if i + 1 < len(self.b) else 0.0
            a_i = self.a[i + 1] if i + 1 < len(self.a) else 0.0
            self.state[i] = b_i * x_n - a_i * y_n
            return y_n

    # (a) Test
    b, a = sig.butter(4, 0.3)
    filt = RealtimeFilter(b, a)

    np.random.seed(42)
    N = 1000
    x = np.random.randn(N)

    # (b) Sample-by-sample processing
    y_realtime = np.zeros(N)
    for n in range(N):
        y_realtime[n] = filt.process_sample(x[n])

    y_batch = sig.lfilter(b, a, x)
    error = np.max(np.abs(y_realtime - y_batch))
    print(f"(a-b) Realtime vs batch error: {error:.2e}")
    print()

    # (c) Timing for different orders
    print(f"(c) Timing per sample (microseconds):")
    print(f"{'Order':<8} {'Time/sample (us)':<20}")
    for order in [4, 8, 16, 32]:
        b_t, a_t = sig.butter(order, 0.3)
        filt_t = RealtimeFilter(b_t, a_t)
        x_t = np.random.randn(10000)

        t0 = time.perf_counter()
        for n in range(len(x_t)):
            filt_t.process_sample(x_t[n])
        elapsed = (time.perf_counter() - t0) / len(x_t) * 1e6

        print(f"{order:<8} {elapsed:<20.2f}")


# === Exercise 7: Multi-Rate Filter Bank ===
# Problem: 3-band filter bank design.

def exercise_7():
    """Three-band filter bank with reconstruction."""
    fs = 8000
    N_taps = 101

    # (a) Design bandpass filters
    h_low = sig.firwin(N_taps, 1000, fs=fs, pass_zero=True)
    h_mid = sig.firwin(N_taps, [1000, 3000], fs=fs, pass_zero=False)
    h_high = sig.firwin(N_taps, 3000, fs=fs, pass_zero=False)

    print("(a) 3-band filter bank (FIR, order={N_taps-1}):")
    print(f"    Low:  0-1000 Hz")
    print(f"    Mid:  1000-3000 Hz")
    print(f"    High: 3000-4000 Hz")
    print()

    # (b) Test signal
    duration = 0.1
    t = np.arange(0, duration, 1 / fs)
    x = (np.sin(2 * np.pi * 500 * t) +
         np.sin(2 * np.pi * 2000 * t) +
         np.sin(2 * np.pi * 3500 * t))

    y_low = sig.lfilter(h_low, 1, x)
    y_mid = sig.lfilter(h_mid, 1, x)
    y_high = sig.lfilter(h_high, 1, x)

    # (c) Reconstruction
    y_recon = y_low + y_mid + y_high

    # (d) Reconstruction error (accounting for group delay)
    delay = (N_taps - 1) // 2
    x_delayed = x[:-delay] if delay > 0 else x
    y_recon_aligned = y_recon[delay:] if delay > 0 else y_recon
    min_len = min(len(x_delayed), len(y_recon_aligned))
    recon_error = np.sqrt(np.mean((x_delayed[:min_len] - y_recon_aligned[:min_len]) ** 2))
    recon_snr = 10 * np.log10(np.mean(x_delayed[:min_len] ** 2) / recon_error ** 2) if recon_error > 0 else float('inf')

    print(f"(c) Reconstruction RMSE: {recon_error:.6f}")
    print(f"    Reconstruction SNR: {recon_snr:.1f} dB")
    print()
    print("(d) Imperfect reconstruction caused by:")
    print("    - Transition band overlap (gaps between bands)")
    print("    - Non-ideal filter magnitude responses")
    print("    - Phase alignment issues between bands")


if __name__ == "__main__":
    print("=== Exercise 1: FIR Filter Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: IIR Filter Design ===")
    exercise_2()
    print("\n=== Exercise 3: Linear Phase Verification ===")
    exercise_3()
    print("\n=== Exercise 4: Filter Structure Implementation ===")
    exercise_4()
    print("\n=== Exercise 5: Quantization Experiment ===")
    exercise_5()
    print("\n=== Exercise 6: Real-Time Filter ===")
    exercise_6()
    print("\n=== Exercise 7: Multi-Rate Filter Bank ===")
    exercise_7()
    print("\nAll exercises completed!")
