"""
Exercises for Lesson 12: Spectral Analysis
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import fft, fftfreq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Periodogram vs Welch ===
# Problem: Three sinusoids (100, 150, 200 Hz) in noise, fs=1000, N=4096.

def exercise_1():
    """Periodogram vs Welch's method comparison."""
    fs = 1000
    N = 4096
    t = np.arange(N) / fs
    np.random.seed(42)

    # Signal: three tones + noise at SNR=10 dB
    signal_clean = 1.0*np.sin(2*np.pi*100*t) + 0.5*np.sin(2*np.pi*150*t) + 0.1*np.sin(2*np.pi*200*t)
    signal_power = np.mean(signal_clean**2)
    noise_power = signal_power / (10**(10/10))
    noise = np.sqrt(noise_power) * np.random.randn(N)
    x = signal_clean + noise

    # (a) Periodogram
    f_per, Pxx_per = sig.periodogram(x, fs)
    print("(a) Periodogram: can identify 100 Hz and 150 Hz clearly.")
    print(f"    200 Hz (amplitude 0.1) may be hard to see in noise.")

    # (b) Welch with different nperseg
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].semilogy(f_per, Pxx_per)
    axes[0, 0].set_title('Periodogram')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('PSD')
    axes[0, 0].set_xlim([0, 300])
    axes[0, 0].grid(True, alpha=0.3)

    for i, nperseg in enumerate([256, 512]):
        f_w, Pxx_w = sig.welch(x, fs, nperseg=nperseg)
        ax = axes[0, 1] if i == 0 else axes[1, 0]
        ax.semilogy(f_w, Pxx_w)
        ax.set_title(f'Welch (nperseg={nperseg})')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.set_xlim([0, 300])
        ax.grid(True, alpha=0.3)

    print(f"(b) nperseg=512 resolves all three tones better than nperseg=256")

    # (c) Window function effect
    windows = ['boxcar', 'hann', 'blackmanharris']
    for win in windows:
        f_w, Pxx_w = sig.welch(x, fs, nperseg=512, window=win)
        axes[1, 1].semilogy(f_w, Pxx_w, label=win, alpha=0.7)

    axes[1, 1].set_title('Window Function Comparison (nperseg=512)')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('PSD')
    axes[1, 1].set_xlim([0, 300])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex12_periodogram_welch.png', dpi=100)
    plt.close()

    # (d) Variance via Monte Carlo
    n_trials = 100
    psd_periodogram = np.zeros((n_trials, N // 2 + 1))
    psd_welch = np.zeros((n_trials, len(sig.welch(x, fs, nperseg=256)[0])))

    for trial in range(n_trials):
        noise_trial = np.sqrt(noise_power) * np.random.randn(N)
        x_trial = signal_clean + noise_trial
        _, psd_periodogram[trial] = sig.periodogram(x_trial, fs)
        _, psd_welch[trial] = sig.welch(x_trial, fs, nperseg=256)

    var_per = np.var(psd_periodogram, axis=0)
    var_welch = np.var(psd_welch, axis=0)

    print(f"\n(d) Mean PSD variance (100 Monte Carlo trials):")
    print(f"    Periodogram: {np.mean(var_per):.6f}")
    print(f"    Welch (256): {np.mean(var_welch):.6f}")
    print(f"    Welch has ~{np.mean(var_per)/np.mean(var_welch):.1f}x lower variance")
    print("    Plot saved: ex12_periodogram_welch.png")


# === Exercise 2: AR Model Identification ===
# Problem: AR(4) process, estimate parameters using Yule-Walker.

def exercise_2():
    """AR model identification and order selection."""
    np.random.seed(42)
    N = 512

    # True AR(4) coefficients: x[n] + 1.5x[n-1] - 0.75x[n-2] + 0.2x[n-3] - 0.05x[n-4] = w[n]
    a_true = np.array([1.0, 1.5, -0.75, 0.2, -0.05])

    # Generate AR process
    w = np.random.randn(N)
    x = sig.lfilter([1], a_true, w)

    # (a) Yule-Walker estimation for various orders
    def yule_walker(x, order):
        """Estimate AR parameters using Yule-Walker method."""
        r = np.correlate(x, x, mode='full')[len(x)-1:]
        r = r[:order+1] / len(x)

        # Toeplitz system
        R = np.zeros((order, order))
        for i in range(order):
            for j in range(order):
                R[i, j] = r[abs(i-j)]

        a = np.linalg.solve(R, r[1:order+1])
        sigma2 = r[0] - np.dot(a, r[1:order+1])
        return a, sigma2

    orders = [2, 4, 6, 8, 12]
    fig, ax = plt.subplots(figsize=(10, 6))

    # True PSD
    w_freq = np.linspace(0, np.pi, 1024)
    H_true = 1.0 / np.polyval(a_true, np.exp(-1j * w_freq))
    Pxx_true = np.abs(H_true)**2
    ax.semilogy(w_freq/np.pi, Pxx_true, 'k-', linewidth=2, label='True AR(4)')

    print("(a-b) AR parameter estimation:")
    for p in orders:
        a_est, sigma2 = yule_walker(x, p)
        a_full = np.concatenate([[1], -a_est])
        H_est = 1.0 / np.polyval(a_full[::-1], np.exp(1j * w_freq))
        Pxx_est = sigma2 * np.abs(H_est)**2
        ax.semilogy(w_freq/np.pi, Pxx_est, '--', alpha=0.7, label=f'AR({p})')

        if p == 4:
            print(f"    AR({p}) estimated: a = {np.round(-a_est, 4)}")
            print(f"    True:             a = {a_true[1:]}")

    ax.set_xlabel('Normalized Frequency (x pi)')
    ax.set_ylabel('PSD')
    ax.set_title('AR Spectral Estimation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex12_ar_estimation.png', dpi=100)
    plt.close()

    # (c) AIC and BIC
    print(f"\n(c) Model order selection:")
    aic_vals = []
    bic_vals = []
    for p in range(1, 16):
        a_est, sigma2 = yule_walker(x, p)
        if sigma2 > 0:
            aic = N * np.log(sigma2) + 2 * p
            bic = N * np.log(sigma2) + p * np.log(N)
            aic_vals.append((p, aic))
            bic_vals.append((p, bic))

    best_aic = min(aic_vals, key=lambda v: v[1])
    best_bic = min(bic_vals, key=lambda v: v[1])
    print(f"    AIC selects order: {best_aic[0]}")
    print(f"    BIC selects order: {best_bic[0]}")
    print(f"    Agreement: {'Yes' if best_aic[0] == best_bic[0] else 'No'}")

    # (d) Reduced data (N=64)
    x_short = x[:64]
    a_est_short, sigma2_short = yule_walker(x_short, 4)
    print(f"\n(d) With N=64: estimated a = {np.round(-a_est_short, 4)}")
    print(f"    Estimation quality degrades with fewer samples.")
    print("    Plot saved: ex12_ar_estimation.png")


# === Exercise 3: Cross-Spectral Analysis ===
# Problem: Two sensors with bandpass transfer function.

def exercise_3():
    """Cross-spectral analysis of two-sensor system."""
    np.random.seed(42)
    fs = 1000
    N = 8192
    t = np.arange(N) / fs

    # (a) Input: broadband random excitation
    x_input = np.random.randn(N)

    # Transfer function: 4th-order bandpass at 150 Hz
    b_sys, a_sys = sig.butter(4, [120, 180], btype='bandpass', fs=fs)
    y_output = sig.lfilter(b_sys, a_sys, x_input) + 0.1 * np.random.randn(N)

    # (b) Coherence
    f_coh, Cxy = sig.coherence(x_input, y_output, fs=fs, nperseg=512)

    print("(b) Coherence analysis:")
    # Find frequency of max coherence
    max_coh_idx = np.argmax(Cxy)
    print(f"    Max coherence: {Cxy[max_coh_idx]:.4f} at {f_coh[max_coh_idx]:.1f} Hz")
    print(f"    High coherence in 120-180 Hz band (as expected)")

    # (c) Transfer function estimation from CSD
    f_csd, Pxy = sig.csd(x_input, y_output, fs=fs, nperseg=512)
    _, Pxx = sig.welch(x_input, fs=fs, nperseg=512)
    H_est = Pxy / (Pxx + 1e-15)

    # True transfer function
    w_true, H_true = sig.freqz(b_sys, a_sys, worN=len(f_csd), fs=fs)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(f_coh, Cxy)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Coherence')
    axes[0].set_title('Coherence Function')
    axes[0].set_xlim([0, 500])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f_csd, 20*np.log10(np.abs(H_est)+1e-15), label='Estimated')
    axes[1].plot(w_true, 20*np.log10(np.abs(H_true)+1e-15), '--', label='True')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_title('Transfer Function Magnitude')
    axes[1].set_xlim([0, 500])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(f_csd, np.angle(H_est), '.', markersize=2, label='Estimated')
    axes[2].plot(w_true, np.angle(H_true), '--', label='True')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Phase (rad)')
    axes[2].set_title('Transfer Function Phase')
    axes[2].set_xlim([0, 500])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex12_cross_spectral.png', dpi=100)
    plt.close()

    # (d) Group delay from phase
    print(f"\n(c-d) Transfer function estimated from cross-spectral density.")
    print("    Plot saved: ex12_cross_spectral.png")


# === Exercise 4: Spectrogram of Bat Echolocation ===
# Problem: Synthetic chirps sweeping 100-50 kHz in 5ms.

def exercise_4():
    """Bat echolocation signal spectrogram analysis."""
    fs = 250000  # 250 kHz sampling rate
    chirp_duration = 0.005  # 5 ms
    ipi = 0.050  # 50 ms inter-pulse interval
    n_chirps = 5
    total_duration = n_chirps * ipi + chirp_duration

    t_total = np.arange(0, total_duration, 1/fs)
    N = len(t_total)
    signal = np.zeros(N)

    # Create chirps
    for i in range(n_chirps):
        start = int(i * ipi * fs)
        chirp_samples = int(chirp_duration * fs)
        if start + chirp_samples <= N:
            t_chirp = np.arange(chirp_samples) / fs
            chirp = sig.chirp(t_chirp, f0=100000, t1=chirp_duration, f1=50000)
            signal[start:start+chirp_samples] = chirp

    # Add noise at -20 dB
    sig_power = np.mean(signal[signal != 0]**2)
    noise_power = sig_power * 10**(-20/10)
    noise = np.sqrt(noise_power) * np.random.randn(N)
    signal_noisy = signal + noise

    # (a) Spectrogram with different nperseg
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    npersegs = [64, 256, 1024]

    for ax, nperseg in zip(axes, npersegs):
        f, t_spec, Sxx = sig.spectrogram(signal_noisy, fs, nperseg=nperseg,
                                          noverlap=nperseg//2, window='hann')
        ax.pcolormesh(t_spec*1000, f/1000, 10*np.log10(Sxx+1e-15), shading='auto', cmap='inferno')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (kHz)')
        ax.set_title(f'nperseg={nperseg}')
        ax.set_ylim([0, 125])

    plt.tight_layout()
    plt.savefig('ex12_bat_spectrogram.png', dpi=100)
    plt.close()

    print(f"(a) Best compromise: nperseg=256 balances time and frequency resolution")

    # (b) Instantaneous frequency
    f, t_spec, Sxx = sig.spectrogram(signal_noisy, fs, nperseg=256, noverlap=200, window='hann')
    for chirp_idx in range(n_chirps):
        t_start = chirp_idx * ipi
        t_end = t_start + chirp_duration
        mask = (t_spec >= t_start) & (t_spec <= t_end)
        if np.any(mask):
            peak_freqs = f[np.argmax(Sxx[:, mask], axis=0)]
            if len(peak_freqs) > 0:
                chirp_rate = (peak_freqs[-1] - peak_freqs[0]) / chirp_duration if len(peak_freqs) > 1 else 0
                if chirp_idx == 0:
                    print(f"\n(b) Chirp {chirp_idx}: freq range {peak_freqs[0]/1e3:.0f}-{peak_freqs[-1]/1e3:.0f} kHz")
                    print(f"    Known chirp rate: {(50000-100000)/chirp_duration/1e6:.1f} MHz/s")

    # (c) Add echo
    echo_delay_samples = int(0.001 * fs)  # 1 ms delay
    echo = 0.3 * np.roll(signal, echo_delay_samples)
    signal_echo = signal + echo + noise
    print(f"\n(c) Echo added with {echo_delay_samples/fs*1000:.1f} ms delay, 0.3 attenuation")
    print(f"    Echo detectable if delay > ~{256/fs*1000:.2f} ms (window length)")
    print("    Plot saved: ex12_bat_spectrogram.png")


# === Exercise 5: Resolution Limits ===
# Problem: Two sinusoids separated by delta_f.

def exercise_5():
    """Spectral resolution limits: periodogram, Welch, and Burg."""
    fs = 1000
    N = 256
    f1 = 100

    # (a) Minimum resolvable delta_f for periodogram
    # Resolution = fs/N = 1000/256 ≈ 3.9 Hz
    print(f"(a) Periodogram frequency resolution: fs/N = {fs/N:.1f} Hz")
    print(f"    Two tones must be separated by at least ~{fs/N:.1f} Hz")

    delta_fs = [2, 3, 4, 5, 8, 10]
    for df in delta_fs:
        t = np.arange(N) / fs
        x = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*(f1+df)*t)
        f_per, Pxx = sig.periodogram(x, fs)
        # Check if two peaks visible between f1-5 and f1+df+5
        mask = (f_per >= f1-5) & (f_per <= f1+df+5)
        peaks_idx = sig.argrelmax(Pxx[mask])[0]
        resolved = len(peaks_idx) >= 2
        if df == 4:
            print(f"    delta_f={df} Hz: resolved={resolved} (borderline)")

    # (b) Compare methods at delta_f=5 Hz
    np.random.seed(42)
    delta_f = 5
    t = np.arange(N) / fs
    x = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*(f1+delta_f)*t) + 0.1*np.random.randn(N)

    # Periodogram
    f_per, Pxx_per = sig.periodogram(x, fs)

    # Welch
    f_welch, Pxx_welch = sig.welch(x, fs, nperseg=128)

    # Burg (AR model-based)
    def burg_psd(x, order, nfft=1024, fs=1):
        """Burg's method for AR spectral estimation."""
        N = len(x)
        ef = x.copy()
        eb = x.copy()
        a = np.array([1.0])

        for p in range(order):
            efp = ef[1:]
            ebp = eb[:-1]
            num = -2 * np.sum(efp * ebp)
            den = np.sum(efp**2) + np.sum(ebp**2)
            k = num / (den + 1e-15)

            a = np.concatenate([a, [0]]) + k * np.concatenate([[0], a[::-1]])

            ef_new = efp + k * ebp
            eb_new = ebp + k * efp
            ef = ef_new
            eb = eb_new

        # PSD from AR coefficients
        sigma2 = np.sum(ef**2) / len(ef)
        w = np.linspace(0, np.pi, nfft)
        H = 1.0 / np.polyval(a[::-1], np.exp(1j * w))
        Pxx = sigma2 * np.abs(H)**2
        freqs = w * fs / (2 * np.pi)
        return freqs, Pxx

    f_burg, Pxx_burg = burg_psd(x, order=20, nfft=1024, fs=fs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(f_per, Pxx_per / np.max(Pxx_per), label='Periodogram')
    ax.semilogy(f_welch, Pxx_welch / np.max(Pxx_welch), label='Welch (128)')
    ax.semilogy(f_burg, Pxx_burg / np.max(Pxx_burg), label='Burg (p=20)')
    ax.set_xlim([80, 120])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized PSD')
    ax.set_title(f'Resolution comparison: f1={f1}, f2={f1+delta_f} Hz, N={N}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex12_resolution.png', dpi=100)
    plt.close()

    print(f"\n(b) At delta_f={delta_f} Hz with N={N}:")
    print(f"    Burg's method (AR) typically has better resolution for short data.")
    print("    Plot saved: ex12_resolution.png")


# === Exercise 6: EEG Analysis ===
# Problem: Simulated EEG with alpha/beta band transition.

def exercise_6():
    """Simulated EEG spectral analysis with state transition."""
    np.random.seed(42)
    fs = 256
    duration = 10
    t = np.arange(0, duration, 1/fs)
    N = len(t)

    # (a) Create EEG with transition at t=5s
    alpha = np.zeros(N)
    beta = np.zeros(N)
    transition = int(5 * fs)

    # Eyes closed (0-5s): strong alpha
    alpha[:transition] = 1.0 * np.sin(2*np.pi*10*t[:transition])
    beta[:transition] = 0.2 * np.sin(2*np.pi*22*t[:transition])

    # Eyes open (5-10s): strong beta
    alpha[transition:] = 0.2 * np.sin(2*np.pi*10*t[transition:])
    beta[transition:] = 1.0 * np.sin(2*np.pi*22*t[transition:])

    noise = 0.5 * np.random.randn(N)
    eeg = alpha + beta + noise

    # (b) Spectrogram
    f, t_spec, Sxx = sig.spectrogram(eeg, fs, nperseg=512, noverlap=448, window='hann')

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].pcolormesh(t_spec, f, 10*np.log10(Sxx+1e-15), shading='auto', cmap='jet')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('EEG Spectrogram')
    axes[0].set_ylim([0, 40])
    axes[0].axvline(5, color='w', linestyle='--', alpha=0.7)

    # (c) Band power time series
    alpha_band = (f >= 8) & (f <= 13)
    beta_band = (f >= 13) & (f <= 30)

    alpha_power = np.mean(Sxx[alpha_band, :], axis=0)
    beta_power = np.mean(Sxx[beta_band, :], axis=0)

    axes[1].plot(t_spec, alpha_power, label='Alpha (8-13 Hz)')
    axes[1].plot(t_spec, beta_power, label='Beta (13-30 Hz)')
    axes[1].axvline(5, color='r', linestyle='--', alpha=0.5, label='Transition')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Band Power')
    axes[1].set_title('Alpha and Beta Band Power')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # (d) Alpha/beta power ratio
    ratio = alpha_power / (beta_power + 1e-15)
    axes[2].plot(t_spec, ratio)
    axes[2].axvline(5, color='r', linestyle='--', alpha=0.5, label='Transition')
    axes[2].axhline(1, color='k', linestyle=':', alpha=0.3)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Alpha/Beta Ratio')
    axes[2].set_title('Alpha/Beta Power Ratio')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex12_eeg.png', dpi=100)
    plt.close()

    # Detection clarity
    ratio_before = np.mean(ratio[t_spec < 5])
    ratio_after = np.mean(ratio[t_spec > 5])
    print(f"(d) Alpha/Beta ratio:")
    print(f"    Eyes closed (t<5s): {ratio_before:.2f}")
    print(f"    Eyes open (t>5s):   {ratio_after:.2f}")
    print(f"    Transition clearly detectable: ratio changes by {ratio_before/ratio_after:.1f}x")
    print("    Plot saved: ex12_eeg.png")


# === Exercise 7: Burg vs Welch for Short Data ===
# Problem: Compare for two tones at 100 and 108 Hz as N varies.

def exercise_7():
    """Burg vs Welch resolution for short data records."""
    fs = 1000
    f1, f2 = 100, 108
    n_trials = 100

    def burg_psd_simple(x, order, nfft=1024, fs=1):
        """Simplified Burg's method."""
        N = len(x)
        ef = x.astype(float).copy()
        eb = x.astype(float).copy()
        a = np.array([1.0])
        for p in range(order):
            efp = ef[1:]
            ebp = eb[:-1]
            num = -2 * np.sum(efp * ebp)
            den = np.sum(efp**2) + np.sum(ebp**2) + 1e-15
            k = num / den
            a = np.concatenate([a, [0]]) + k * np.concatenate([[0], a[::-1]])
            ef_new = efp + k * ebp
            eb_new = ebp + k * efp
            ef = ef_new
            eb = eb_new
        sigma2 = np.sum(ef**2) / (len(ef) + 1e-15)
        w = np.linspace(0, np.pi, nfft)
        H = 1.0 / np.polyval(a[::-1], np.exp(1j * w))
        Pxx = sigma2 * np.abs(H)**2
        freqs = w * fs / (2 * np.pi)
        return freqs, Pxx

    def check_resolved(freqs, psd, f1, f2, notch_depth_db=3):
        """Check if two peaks are resolved with at least notch_depth_db dip."""
        mask = (freqs >= f1 - 5) & (freqs <= f2 + 5)
        psd_region = psd[mask]
        freqs_region = freqs[mask]
        if len(psd_region) < 5:
            return False
        # Find two local maxima
        peaks = sig.argrelmax(psd_region, order=2)[0]
        if len(peaks) < 2:
            return False
        # Check notch between peaks
        valley = np.min(psd_region[peaks[0]:peaks[-1]+1])
        peak_val = min(psd_region[peaks[0]], psd_region[peaks[-1]])
        notch = 10 * np.log10(peak_val / (valley + 1e-15))
        return notch >= notch_depth_db

    N_values = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    welch_success = []
    burg_success = []

    print("(a-d) Resolution success rate (100 trials, 3 dB criterion):\n")
    print(f"{'N':>6} {'Welch':>10} {'Burg (p=20)':>12}")
    print("-" * 30)

    for N in N_values:
        w_count = 0
        b_count = 0

        for trial in range(n_trials):
            t = np.arange(N) / fs
            x = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + 0.3*np.random.randn(N)

            # Welch
            nperseg_w = max(N // 2, 16)
            try:
                f_w, Pxx_w = sig.welch(x, fs, nperseg=nperseg_w)
                if check_resolved(f_w, Pxx_w, f1, f2):
                    w_count += 1
            except Exception:
                pass

            # Burg
            try:
                order = min(20, N // 3)
                f_b, Pxx_b = burg_psd_simple(x, order=order, nfft=1024, fs=fs)
                if check_resolved(f_b, Pxx_b, f1, f2):
                    b_count += 1
            except Exception:
                pass

        welch_success.append(w_count / n_trials)
        burg_success.append(b_count / n_trials)
        print(f"{N:>6} {w_count/n_trials:>10.0%} {b_count/n_trials:>12.0%}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(N_values, welch_success, 'bo-', label='Welch')
    ax.plot(N_values, burg_success, 'rs-', label='Burg (p=20)')
    ax.axhline(0.9, color='g', linestyle=':', alpha=0.3, label='90% threshold')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Number of Samples (N)')
    ax.set_ylabel('Resolution Success Rate')
    ax.set_title(f'Resolution of {f1} Hz and {f2} Hz Tones')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex12_burg_vs_welch.png', dpi=100)
    plt.close()

    # Find minimum N for 90% success
    for method, rates, name in [(welch_success, N_values, 'Welch'),
                                 (burg_success, N_values, 'Burg')]:
        min_n = None
        for n, rate in zip(rates, method):
            if rate >= 0.9:
                min_n = n
                break
        if min_n:
            print(f"\n    {name}: minimum N for 90% success = {min_n}")

    print("    Plot saved: ex12_burg_vs_welch.png")


if __name__ == "__main__":
    print("=== Exercise 1: Periodogram vs Welch ===")
    exercise_1()
    print("\n=== Exercise 2: AR Model Identification ===")
    exercise_2()
    print("\n=== Exercise 3: Cross-Spectral Analysis ===")
    exercise_3()
    print("\n=== Exercise 4: Spectrogram of Bat Echolocation ===")
    exercise_4()
    print("\n=== Exercise 5: Resolution Limits ===")
    exercise_5()
    print("\n=== Exercise 6: EEG Analysis ===")
    exercise_6()
    print("\n=== Exercise 7: Burg vs Welch for Short Data ===")
    exercise_7()
    print("\nAll exercises completed!")
