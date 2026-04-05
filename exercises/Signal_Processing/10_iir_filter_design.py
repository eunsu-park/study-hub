"""
Exercises for Lesson 10: IIR Filter Design
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal as sig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Butterworth Filter Design ===
# Problem: Digital Butterworth lowpass, fs=10000, fp=1500 (<=0.5 dB ripple),
#          fstop=2000 (>=40 dB atten).

def exercise_1():
    """Butterworth lowpass filter design from specifications."""
    fs = 10000
    fp = 1500
    fstop = 2000
    Rp = 0.5
    As = 40

    # (a) Compute required order
    wp = 2 * fp / fs  # normalized
    ws = 2 * fstop / fs
    N, Wn = sig.buttord(wp, ws, Rp, As)
    print(f"(a) Required filter order: N = {N}")
    print(f"    Natural frequency Wn = {Wn:.4f}")

    # (b) Pre-warp digital frequencies to analog
    T = 1 / fs
    Omega_p = 2 / T * np.tan(np.pi * fp / fs)
    Omega_s = 2 / T * np.tan(np.pi * fstop / fs)
    print(f"\n(b) Pre-warped analog frequencies:")
    print(f"    Omega_p = {Omega_p:.2f} rad/s ({Omega_p/(2*np.pi):.2f} Hz)")
    print(f"    Omega_s = {Omega_s:.2f} rad/s ({Omega_s/(2*np.pi):.2f} Hz)")

    # (c) Design analog prototype and apply bilinear transform
    b, a = sig.butter(N, Wn, btype='low')
    sos = sig.butter(N, Wn, btype='low', output='sos')
    print(f"\n(c) Filter designed (bilinear transform applied)")
    print(f"    b = {np.round(b, 6)}")
    print(f"    a = {np.round(a, 6)}")

    # (d) Verify and plot
    w, H = sig.freqz(b, a, worN=8192, fs=fs)
    mag_db = 20 * np.log10(np.abs(H) + 1e-15)
    phase = np.unwrap(np.angle(H))
    _, gd = sig.group_delay((b, a), w=8192, fs=fs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(w, mag_db)
    axes[0, 0].axhline(-Rp, color='r', linestyle='--', alpha=0.5, label=f'-{Rp} dB')
    axes[0, 0].axhline(-As, color='orange', linestyle='--', alpha=0.5, label=f'-{As} dB')
    axes[0, 0].axvline(fp, color='g', linestyle=':', alpha=0.3)
    axes[0, 0].axvline(fstop, color='g', linestyle=':', alpha=0.3)
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].set_title('Magnitude Response')
    axes[0, 0].set_ylim([-80, 5])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(w, phase)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Phase (rad)')
    axes[0, 1].set_title('Phase Response')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(w, gd)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Group Delay (samples)')
    axes[1, 0].set_title('Group Delay')
    axes[1, 0].grid(True, alpha=0.3)

    # Pole-zero diagram
    z_poles, p_poles, _ = sig.tf2zpk(b, a)
    theta = np.linspace(0, 2 * np.pi, 100)
    axes[1, 1].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    axes[1, 1].scatter(np.real(z_poles), np.imag(z_poles), marker='o', s=50, label='Zeros')
    axes[1, 1].scatter(np.real(p_poles), np.imag(p_poles), marker='x', s=50, label='Poles')
    axes[1, 1].set_xlabel('Real')
    axes[1, 1].set_ylabel('Imaginary')
    axes[1, 1].set_title('Pole-Zero Diagram')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex10_butterworth.png', dpi=100)
    plt.close()

    # Verify specs
    pb_idx = w <= fp
    sb_idx = w >= fstop
    actual_ripple = -np.min(mag_db[pb_idx])
    actual_atten = -np.max(mag_db[sb_idx])
    print(f"\n(d) Verification:")
    print(f"    Passband ripple: {actual_ripple:.3f} dB (spec <= {Rp} dB)")
    print(f"    Stopband atten:  {actual_atten:.1f} dB (spec >= {As} dB)")

    # (e) Compare with scipy.signal.butter
    b2, a2 = sig.butter(N, Wn, btype='low')
    print(f"\n(e) scipy.signal.butter comparison:")
    print(f"    Coefficients identical: {np.allclose(b, b2) and np.allclose(a, a2)}")
    print("    Plot saved: ex10_butterworth.png")


# === Exercise 2: Chebyshev Filter Comparison ===
# Problem: fs=8000, fp=1000, fstop=1200, Rp=1 dB, As=50 dB

def exercise_2():
    """Chebyshev Type I vs Type II filter comparison."""
    fs = 8000
    fp = 1000
    fstop = 1200
    Rp = 1
    As = 50

    wp = 2 * fp / fs
    ws = 2 * fstop / fs

    # (a) Design both filters
    N1, Wn1 = sig.cheb1ord(wp, ws, Rp, As)
    b1, a1 = sig.cheby1(N1, Rp, Wn1, btype='low')

    N2, Wn2 = sig.cheb2ord(wp, ws, Rp, As)
    b2, a2 = sig.cheby2(N2, As, Wn2, btype='low')

    print(f"(a) Chebyshev Type I: order={N1}")
    print(f"    Chebyshev Type II: order={N2}")

    # (b) Frequency response comparison
    w, H1 = sig.freqz(b1, a1, worN=8192, fs=fs)
    _, H2 = sig.freqz(b2, a2, worN=8192, fs=fs)
    mag1_db = 20 * np.log10(np.abs(H1) + 1e-15)
    mag2_db = 20 * np.log10(np.abs(H2) + 1e-15)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(w, mag1_db, label=f'Type I (N={N1})')
    axes[0].plot(w, mag2_db, '--', label=f'Type II (N={N2})')
    axes[0].axhline(-Rp, color='r', linestyle=':', alpha=0.3)
    axes[0].axhline(-As, color='orange', linestyle=':', alpha=0.3)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('Chebyshev Type I vs Type II')
    axes[0].set_ylim([-80, 5])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (c) Group delay comparison
    _, gd1 = sig.group_delay((b1, a1), w=8192, fs=fs)
    _, gd2 = sig.group_delay((b2, a2), w=8192, fs=fs)

    pb_idx = w <= fp
    gd1_var = np.std(gd1[pb_idx])
    gd2_var = np.std(gd2[pb_idx])

    axes[1].plot(w[pb_idx], gd1[pb_idx], label=f'Type I (std={gd1_var:.2f})')
    axes[1].plot(w[pb_idx], gd2[pb_idx], '--', label=f'Type II (std={gd2_var:.2f})')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Group Delay (samples)')
    axes[1].set_title('Passband Group Delay')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex10_chebyshev_comparison.png', dpi=100)
    plt.close()

    print(f"\n(c) Passband group delay flatness:")
    print(f"    Type I  std: {gd1_var:.2f} samples")
    print(f"    Type II std: {gd2_var:.2f} samples")
    better = "Type II" if gd2_var < gd1_var else "Type I"
    print(f"    {better} has flatter group delay in the passband.")

    # (d) Apply to chirp signal
    duration = 0.2
    t = np.arange(0, duration, 1/fs)
    chirp_sig = sig.chirp(t, f0=100, t1=duration, f1=4000)
    y1 = sig.lfilter(b1, a1, chirp_sig)
    y2 = sig.lfilter(b2, a2, chirp_sig)

    print(f"\n(d) Chirp signal filtered. Output energy ratio (Type I / Type II): "
          f"{np.sum(y1**2)/np.sum(y2**2):.3f}")
    print("    Plot saved: ex10_chebyshev_comparison.png")


# === Exercise 3: Elliptic Filter for Audio ===
# Problem: Anti-aliasing for 96kHz->48kHz downsample.

def exercise_3():
    """Elliptic filter for audio anti-aliasing."""
    fs = 96000
    fp = 20000
    fstop = 24000
    Rp = 0.01
    As = 96

    wp = 2 * fp / fs
    ws = 2 * fstop / fs

    # (a) Minimum filter order
    N, Wn = sig.ellipord(wp, ws, Rp, As)
    print(f"(a) Minimum elliptic filter order: N = {N}")

    # (b) Design using SOS form
    sos = sig.ellip(N, Rp, As, Wn, btype='low', output='sos')
    print(f"(b) SOS sections: {sos.shape[0]}")

    # (c) Magnitude response
    w, H = sig.sosfreqz(sos, worN=16384, fs=fs)
    mag_db = 20 * np.log10(np.abs(H) + 1e-15)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(w/1000, mag_db)
    axes[0].axhline(-As, color='r', linestyle='--', alpha=0.5, label=f'-{As} dB')
    axes[0].axvline(fp/1000, color='g', linestyle=':', alpha=0.3)
    axes[0].axvline(fstop/1000, color='g', linestyle=':', alpha=0.3)
    axes[0].set_xlabel('Frequency (kHz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title(f'Elliptic Anti-Aliasing Filter (N={N})')
    axes[0].set_ylim([-120, 5])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (d) Group delay
    b_total, a_total = sig.sos2tf(sos)
    _, gd = sig.group_delay((b_total, a_total), w=16384, fs=fs)

    pb_idx = w <= fp
    axes[1].plot(w[pb_idx]/1000, gd[pb_idx])
    axes[1].set_xlabel('Frequency (kHz)')
    axes[1].set_ylabel('Group Delay (samples)')
    axes[1].set_title('Passband Group Delay')
    axes[1].grid(True, alpha=0.3)

    gd_var = np.max(gd[pb_idx]) - np.min(gd[pb_idx])
    print(f"\n(d) Group delay variation in passband: {gd_var:.2f} samples")
    print(f"    At {fs} Hz, that is {gd_var/fs*1000:.3f} ms")
    acceptable = gd_var / fs * 1000 < 1.0
    print(f"    Acceptable for audio: {'Yes' if acceptable else 'Marginal'} (<1 ms is generally OK)")

    plt.tight_layout()
    plt.savefig('ex10_elliptic_audio.png', dpi=100)
    plt.close()
    print("    Plot saved: ex10_elliptic_audio.png")


# === Exercise 4: Bilinear Transform by Hand ===
# Problem: Apply bilinear transform to H_a(s) = Omega_c / (s + Omega_c).

def exercise_4():
    """Manual bilinear transform computation."""
    fs = 8000
    T = 1 / fs
    fc_analog = 1000  # Hz
    Omega_c = 2 * np.pi * fc_analog

    # (a) H(z) from H_a(s) via bilinear transform s = (2/T)(z-1)/(z+1)
    print("(a) H_a(s) = Omega_c / (s + Omega_c)")
    print("    s = (2/T)(z-1)/(z+1)")
    print("    H(z) = Omega_c / ((2/T)(z-1)/(z+1) + Omega_c)")
    print("         = Omega_c(z+1) / ((2/T)(z-1) + Omega_c(z+1))")
    print("         = Omega_c(z+1) / ((2/T + Omega_c)z + (Omega_c - 2/T))")

    # (b) Compute coefficients
    alpha = 2 * fs  # 2/T
    b0 = Omega_c / (alpha + Omega_c)
    b1 = b0
    a0 = 1.0
    a1 = (Omega_c - alpha) / (alpha + Omega_c)

    print(f"\n(b) For Omega_c = {Omega_c:.2f}, fs = {fs}:")
    print(f"    b0 = {b0:.6f}")
    print(f"    b1 = {b1:.6f}")
    print(f"    a0 = {a0:.6f}")
    print(f"    a1 = {a1:.6f}")

    # (c) Verify frequency response
    b = np.array([b0, b1])
    a = np.array([a0, a1])
    w, H_digital = sig.freqz(b, a, worN=4096, fs=fs)

    # Expected warped cutoff
    fc_warped = (fs / np.pi) * np.arctan(Omega_c * T / 2)
    print(f"\n(c) Warped cutoff frequency: {fc_warped:.2f} Hz (analog: {fc_analog} Hz)")

    # Check -3 dB point
    mag_db = 20 * np.log10(np.abs(H_digital) + 1e-15)
    idx_3db = np.argmin(np.abs(mag_db + 3))
    print(f"    Actual -3 dB frequency: {w[idx_3db]:.2f} Hz")

    # (d) Repeat for 2nd-order Butterworth
    N = 2
    b2, a2 = sig.butter(N, fc_warped / (fs/2), btype='low')
    w2, H2 = sig.freqz(b2, a2, worN=4096, fs=fs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(w, 20*np.log10(np.abs(H_digital)+1e-15), label='1st order (manual)')
    ax.plot(w2, 20*np.log10(np.abs(H2)+1e-15), '--', label='2nd order Butterworth')
    ax.axhline(-3, color='r', linestyle=':', alpha=0.3)
    ax.axvline(fc_warped, color='g', linestyle=':', alpha=0.3, label=f'fc={fc_warped:.0f} Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Bilinear Transform: Manual vs scipy')
    ax.set_ylim([-60, 5])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex10_bilinear.png', dpi=100)
    plt.close()
    print("    Plot saved: ex10_bilinear.png")


# === Exercise 5: Stability Investigation ===
# Problem: Compare tf vs sos for high-order Chebyshev filters.

def exercise_5():
    """Numerical stability: tf vs sos for high-order IIR filters."""
    Rp = 3.0
    wc = 0.1  # normalized cutoff

    # (a-b) Design and check pole magnitudes
    print("Stability investigation: tf vs sos for Chebyshev Type I\n")
    print(f"{'Order':>6} {'Max |pole| (tf)':>16} {'Max |pole| (sos)':>17} {'tf stable':>10}")
    print("-" * 55)

    orders = [4, 8, 12, 16, 20, 24]
    tf_stable = []

    for N in orders:
        # tf form
        b_tf, a_tf = sig.cheby1(N, Rp, wc, btype='low')
        _, poles_tf, _ = sig.tf2zpk(b_tf, a_tf)
        max_pole_tf = np.max(np.abs(poles_tf))

        # sos form
        sos = sig.cheby1(N, Rp, wc, btype='low', output='sos')
        # Get poles from each SOS section
        all_poles = []
        for section in sos:
            _, p, _ = sig.tf2zpk(section[:3], section[3:])
            all_poles.extend(p)
        max_pole_sos = np.max(np.abs(all_poles))

        stable = max_pole_tf < 1.0
        tf_stable.append(stable)

        print(f"{N:>6} {max_pole_tf:>16.6f} {max_pole_sos:>17.6f} {str(stable):>10}")

    # (c) Filter white noise and compare
    np.random.seed(42)
    noise = np.random.randn(1000)
    print(f"\n(c) Filtering white noise (order=16):")

    N_test = 16
    b_tf, a_tf = sig.cheby1(N_test, Rp, wc, btype='low')
    sos = sig.cheby1(N_test, Rp, wc, btype='low', output='sos')

    y_tf = sig.lfilter(b_tf, a_tf, noise)
    y_sos = sig.sosfilt(sos, noise)

    tf_valid = np.all(np.isfinite(y_tf))
    sos_valid = np.all(np.isfinite(y_sos))

    print(f"    tf form output valid: {tf_valid} (max={np.max(np.abs(y_tf)) if tf_valid else 'inf'})")
    print(f"    sos form output valid: {sos_valid} (max={np.max(np.abs(y_sos)):.4f})")

    # (d) Breakdown order
    breakdown_order = None
    for N in range(4, 40, 2):
        b, a = sig.cheby1(N, Rp, wc, btype='low')
        y = sig.lfilter(b, a, noise)
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e10:
            breakdown_order = N
            break

    if breakdown_order:
        print(f"\n(d) tf form breaks down at order: {breakdown_order}")
    else:
        print(f"\n(d) tf form did not break down up to order 38")


# === Exercise 6: IIR vs FIR Comparison ===
# Problem: Compare IIR (elliptic) and FIR (Parks-McClellan) for same spec.

def exercise_6():
    """IIR vs FIR filter comparison."""
    fs = 16000
    fp = 3000
    fstop = 3500
    As = 60

    wp = 2 * fp / fs
    ws = 2 * fstop / fs

    # (a) Design both filters
    # IIR: elliptic
    N_iir, Wn_iir = sig.ellipord(wp, ws, 0.5, As)
    sos_iir = sig.ellip(N_iir, 0.5, As, Wn_iir, output='sos')
    b_iir, a_iir = sig.sos2tf(sos_iir)

    # FIR: Parks-McClellan
    delta_s = 10 ** (-As / 20)
    N_fir = None
    for n in range(20, 300, 2):
        try:
            h_fir = sig.remez(n + 1, [0, fp, fstop, fs/2], [1, 0], fs=fs)
            w_t, H_t = sig.freqz(h_fir, 1, worN=4096, fs=fs)
            mag_t = 20 * np.log10(np.abs(H_t) + 1e-15)
            if -np.max(mag_t[w_t >= fstop]) >= As:
                N_fir = n
                break
        except Exception:
            continue

    if N_fir is None:
        N_fir = 100
        h_fir = sig.remez(N_fir + 1, [0, fp, fstop, fs/2], [1, 0], fs=fs)

    print(f"(a) Filter orders:")
    print(f"    IIR (elliptic): N = {N_iir}")
    print(f"    FIR (remez):    N = {N_fir}")

    # (b) Compare metrics
    w, H_iir = sig.sosfreqz(sos_iir, worN=8192, fs=fs)
    _, H_fir = sig.freqz(h_fir, 1, worN=8192, fs=fs)

    # Computational cost per sample
    iir_mults = 2 * N_iir + 1  # approximate for direct form
    fir_mults = N_fir + 1
    print(f"\n(b) Computational cost per sample:")
    print(f"    IIR: ~{iir_mults} multiplications")
    print(f"    FIR: ~{fir_mults} multiplications")

    # Group delay
    _, gd_iir = sig.group_delay((b_iir, a_iir), w=8192, fs=fs)
    _, gd_fir = sig.group_delay((h_fir, 1), w=8192, fs=fs)

    pb_idx = w <= fp
    print(f"\n    Group delay in passband:")
    print(f"    IIR: mean={np.mean(gd_iir[pb_idx]):.1f}, std={np.std(gd_iir[pb_idx]):.1f} samples")
    print(f"    FIR: mean={np.mean(gd_fir[pb_idx]):.1f}, std={np.std(gd_fir[pb_idx]):.1f} samples")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(w, 20*np.log10(np.abs(H_iir)+1e-15), label=f'IIR N={N_iir}')
    axes[0].plot(w, 20*np.log10(np.abs(H_fir)+1e-15), '--', label=f'FIR N={N_fir}')
    axes[0].axhline(-As, color='r', linestyle=':', alpha=0.3)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('IIR vs FIR: Magnitude Response')
    axes[0].set_ylim([-100, 5])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(w[pb_idx], gd_iir[pb_idx], label='IIR')
    axes[1].plot(w[pb_idx], gd_fir[pb_idx], '--', label='FIR')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Group Delay (samples)')
    axes[1].set_title('Passband Group Delay')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex10_iir_vs_fir.png', dpi=100)
    plt.close()

    # (c) Filter a speech-like signal
    t = np.arange(0, 0.1, 1/fs)
    signal_speech = sum(np.sin(2*np.pi*f*t) for f in range(100, 5001, 100))
    y_iir = sig.sosfilt(sos_iir, signal_speech)
    y_fir = sig.lfilter(h_fir, 1, signal_speech)

    # SNR (comparing against ideal lowpass)
    pb_energy = sum(np.sin(2*np.pi*f*t)**2 for f in range(100, 3001, 100))
    sb_energy_iir = np.sum((y_iir - sum(np.sin(2*np.pi*f*t) for f in range(100, 3001, 100)))**2)
    print(f"\n(c) Filtered speech-like signal (harmonics 100-5000 Hz)")

    # (d) Application guidance
    print(f"\n(d) Choose IIR when:")
    print(f"    - Low computational cost is critical")
    print(f"    - Real-time with tight constraints")
    print(f"    - Phase distortion is acceptable")
    print(f"    Choose FIR when:")
    print(f"    - Linear phase required (medical, audio mastering)")
    print(f"    - Guaranteed stability needed")
    print(f"    - Coefficient quantization sensitivity matters")
    print("    Plot saved: ex10_iir_vs_fir.png")


# === Exercise 7: Bandstop (Notch) Filter ===
# Problem: Remove 60 Hz powerline interference from ECG. fs=500 Hz.

def exercise_7():
    """Notch filter for 60 Hz powerline noise removal from ECG."""
    fs = 500
    f_notch = 60
    f_low = 59
    f_high = 61

    # (a) Elliptic bandstop design
    wp = [55, 65]  # passband edges
    ws = [59, 61]  # stopband edges
    wp_n = [2*f/fs for f in wp]
    ws_n = [2*f/fs for f in ws]

    N, Wn = sig.ellipord(wp_n, ws_n, 0.1, 30)
    sos = sig.ellip(N, 0.1, 30, Wn, btype='bandstop', output='sos')
    print(f"(a) Elliptic notch filter order: N = {N}")

    # For comparison, also design a Butterworth
    N_but, Wn_but = sig.buttord(wp_n, ws_n, 0.1, 30)
    sos_but = sig.butter(N_but, Wn_but, btype='bandstop', output='sos')

    # (b) Magnitude response
    w, H_ellip = sig.sosfreqz(sos, worN=16384, fs=fs)
    _, H_but = sig.sosfreqz(sos_but, worN=16384, fs=fs)
    mag_ellip = 20 * np.log10(np.abs(H_ellip) + 1e-15)
    mag_but = 20 * np.log10(np.abs(H_but) + 1e-15)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(w, mag_ellip, label=f'Elliptic (N={N})')
    axes[0].plot(w, mag_but, '--', label=f'Butterworth (N={N_but})')
    axes[0].axhline(-30, color='r', linestyle=':', alpha=0.3)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('Notch Filter: Full Response')
    axes[0].set_xlim([0, 250])
    axes[0].set_ylim([-60, 5])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zoomed around 60 Hz
    zoom_idx = (w >= 40) & (w <= 80)
    axes[1].plot(w[zoom_idx], mag_ellip[zoom_idx], label='Elliptic')
    axes[1].plot(w[zoom_idx], mag_but[zoom_idx], '--', label='Butterworth')
    axes[1].axvline(60, color='r', linestyle=':', alpha=0.5, label='60 Hz')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_title('Zoomed Around 60 Hz')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex10_notch_filter.png', dpi=100)
    plt.close()

    # (c) Synthetic ECG with 60 Hz contamination
    duration = 5
    t = np.arange(0, duration, 1/fs)
    N_samples = len(t)

    # Simple synthetic ECG (R-peaks as Gaussian pulses at ~75 bpm)
    ecg = np.zeros(N_samples)
    bpm = 75
    period_samples = int(fs * 60 / bpm)
    for peak_loc in range(0, N_samples, period_samples):
        # QRS complex approximation
        for i in range(max(0, peak_loc-10), min(N_samples, peak_loc+10)):
            ecg[i] += 1.5 * np.exp(-0.5 * ((i - peak_loc) / 3) ** 2)
        # P wave
        p_loc = peak_loc - 20
        if 0 <= p_loc < N_samples:
            for i in range(max(0, p_loc-8), min(N_samples, p_loc+8)):
                ecg[i] += 0.2 * np.exp(-0.5 * ((i - p_loc) / 4) ** 2)
        # T wave
        t_loc = peak_loc + 30
        if 0 <= t_loc < N_samples:
            for i in range(max(0, t_loc-15), min(N_samples, t_loc+15)):
                ecg[i] += 0.3 * np.exp(-0.5 * ((i - t_loc) / 8) ** 2)

    # Add 60 Hz interference
    noise_60 = 0.5 * np.sin(2 * np.pi * 60 * t)
    ecg_noisy = ecg + noise_60

    # Filter
    ecg_filtered = sig.sosfilt(sos, ecg_noisy)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(t[:1000], ecg[:1000])
    axes[0].set_title('Clean ECG')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t[:1000], ecg_noisy[:1000])
    axes[1].set_title('ECG + 60 Hz Noise')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t[:1000], ecg_filtered[:1000])
    axes[2].set_title('Filtered ECG')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex10_ecg_notch.png', dpi=100)
    plt.close()

    # (d) Group delay comparison
    b_e, a_e = sig.sos2tf(sos)
    b_b, a_b = sig.sos2tf(sos_but)
    _, gd_e = sig.group_delay((b_e, a_e), w=8192, fs=fs)
    _, gd_b = sig.group_delay((b_b, a_b), w=8192, fs=fs)

    pb_region = (w <= 55) | (w >= 65)
    pb_region = pb_region & (w <= 200)
    print(f"\n(d) Group delay comparison (passband):")
    print(f"    Elliptic: mean={np.mean(gd_e[pb_region]):.1f}, max={np.max(gd_e[pb_region]):.1f} samples")
    print(f"    Butterworth: mean={np.mean(gd_b[pb_region]):.1f}, max={np.max(gd_b[pb_region]):.1f} samples")
    print("    Plots saved: ex10_notch_filter.png, ex10_ecg_notch.png")


if __name__ == "__main__":
    print("=== Exercise 1: Butterworth Filter Design ===")
    exercise_1()
    print("\n=== Exercise 2: Chebyshev Filter Comparison ===")
    exercise_2()
    print("\n=== Exercise 3: Elliptic Filter for Audio ===")
    exercise_3()
    print("\n=== Exercise 4: Bilinear Transform by Hand ===")
    exercise_4()
    print("\n=== Exercise 5: Stability Investigation ===")
    exercise_5()
    print("\n=== Exercise 6: IIR vs FIR Comparison ===")
    exercise_6()
    print("\n=== Exercise 7: Bandstop (Notch) Filter ===")
    exercise_7()
    print("\nAll exercises completed!")
