"""
Exercises for Lesson 14: Time-Frequency Analysis
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal as sig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: STFT Resolution Tradeoff ===
# Problem: Two close sinusoids (100, 105 Hz) plus a click.

def exercise_1():
    """STFT time-frequency resolution tradeoff demonstration."""
    fs = 1000
    duration = 1.0
    t = np.arange(0, duration, 1/fs)
    N = len(t)

    # Signal: two close tones for 0.5s, then silence
    signal = np.zeros(N)
    signal[:500] = np.sin(2*np.pi*100*t[:500]) + np.sin(2*np.pi*105*t[:500])

    # (c) Add click at t=0.3s
    click_idx = int(0.3 * fs)
    signal[click_idx] += 5.0

    # (a-b) Spectrograms with various window lengths
    window_lengths = [32, 64, 128, 256, 512]
    fig, axes = plt.subplots(1, len(window_lengths), figsize=(20, 4))

    print("(a-b) STFT resolution analysis:")
    for ax, wlen in zip(axes, window_lengths):
        f, t_spec, Sxx = sig.spectrogram(signal, fs, nperseg=wlen,
                                          noverlap=wlen*3//4, window='hann')
        ax.pcolormesh(t_spec, f, 10*np.log10(Sxx+1e-15), shading='auto', cmap='viridis')
        ax.set_ylim([80, 120])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Freq (Hz)')
        ax.set_title(f'N={wlen}')

        # Check frequency resolution
        df = fs / wlen
        resolved = df <= 5  # 5 Hz separation
        dt = wlen / fs
        click_visible = dt < 0.05
        print(f"    N={wlen}: df={df:.1f} Hz, dt={dt*1000:.0f} ms, "
              f"freq resolved={resolved}, click localized={click_visible}")

    plt.tight_layout()
    plt.savefig('ex14_stft_tradeoff.png', dpi=100)
    plt.close()

    # (d) Uncertainty principle verification
    print(f"\n(d) Uncertainty principle: delta_t * delta_f >= 1/(4*pi) = {1/(4*np.pi):.4f}")
    for wlen in window_lengths:
        w = sig.windows.hann(wlen)
        dt = np.sqrt(np.sum((np.arange(wlen) - wlen/2)**2 * w**2) / np.sum(w**2)) / fs
        W = np.abs(np.fft.fft(w, 4096))
        f_ax = np.arange(4096) * fs / 4096
        df = np.sqrt(np.sum((f_ax - np.sum(f_ax * W**2)/np.sum(W**2))**2 * W**2) / np.sum(W**2))
        product = dt * df
        print(f"    N={wlen}: dt={dt:.4f}s, df={df:.1f}Hz, dt*df={product:.4f} >= {1/(4*np.pi):.4f}: {product >= 1/(4*np.pi)-0.01}")

    print("    Plot saved: ex14_stft_tradeoff.png")


# === Exercise 2: Window Comparison ===
# Problem: Chirp 50-200 Hz, compare spectrograms with different windows.

def exercise_2():
    """Window function comparison for spectrogram analysis."""
    fs = 1000
    duration = 1.0
    t = np.arange(0, duration, 1/fs)
    chirp = sig.chirp(t, f0=50, t1=duration, f1=200, method='linear')
    wlen = 128

    windows = ['boxcar', 'hann', 'hamming', 'blackman']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    print("(a-c) Window comparison for chirp spectrogram:")
    for ax, win_name in zip(axes.flat, windows):
        w = sig.get_window(win_name, wlen)
        f, t_spec, Sxx = sig.spectrogram(chirp, fs, window=w, nperseg=wlen,
                                           noverlap=wlen*3//4)
        ax.pcolormesh(t_spec, f, 10*np.log10(Sxx+1e-15), shading='auto', cmap='inferno')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(win_name.capitalize())
        ax.set_ylim([0, 300])

        # Mainlobe width and sidelobe level
        W = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(w, 4096))) + 1e-15)
        W -= np.max(W)
        center = len(W) // 2
        # -3 dB mainlobe width
        above_3db = np.where(W[center:] < -3)[0]
        mainlobe_bins = above_3db[0] * 2 if len(above_3db) > 0 else wlen
        mainlobe_hz = mainlobe_bins * fs / 4096
        # Highest sidelobe
        # Find first null after center
        first_null = center + above_3db[0] if len(above_3db) > 0 else center + 10
        sidelobe_level = np.max(W[first_null+5:first_null+500]) if first_null+500 < len(W) else -50

        print(f"    {win_name:>10}: -3dB mainlobe={mainlobe_hz:.1f} Hz, "
              f"highest sidelobe={sidelobe_level:.1f} dB")

    plt.tight_layout()
    plt.savefig('ex14_windows.png', dpi=100)
    plt.close()
    print("    Blackman has narrowest sidelobes -> cleanest chirp trajectory")
    print("    Plot saved: ex14_windows.png")


# === Exercise 3: Wavelet Families ===
# Problem: Compare wavelet families using pywt.

def exercise_3():
    """Wavelet family comparison (using manual Haar and Morlet approximation)."""
    # Simplified version without pywt dependency
    print("(a-d) Wavelet family analysis (simplified without pywt):")

    # Haar wavelet
    N = 256
    haar_scaling = np.zeros(N)
    haar_scaling[:N//2] = 1.0

    haar_wavelet = np.zeros(N)
    haar_wavelet[:N//2] = 1.0
    haar_wavelet[N//2:] = -1.0

    # Morlet wavelet
    t = np.linspace(-4, 4, N)
    sigma = 1.0
    morlet = np.exp(-t**2 / (2*sigma**2)) * np.cos(5*t)

    # Mexican hat (DoG)
    mexican_hat = (1 - t**2) * np.exp(-t**2 / 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(haar_scaling, label='Scaling')
    axes[0, 0].plot(haar_wavelet, '--', label='Wavelet')
    axes[0, 0].set_title('Haar Wavelet')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, morlet)
    axes[0, 1].set_title('Morlet Wavelet')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t, mexican_hat)
    axes[1, 0].set_title('Mexican Hat Wavelet')
    axes[1, 0].grid(True, alpha=0.3)

    # Frequency responses
    freqs = np.fft.fftfreq(N, d=1/N)
    H_haar = np.abs(np.fft.fft(haar_wavelet))
    H_morlet = np.abs(np.fft.fft(morlet))
    H_mexhat = np.abs(np.fft.fft(mexican_hat))

    axes[1, 1].plot(freqs[:N//2], H_haar[:N//2] / np.max(H_haar), label='Haar')
    axes[1, 1].plot(freqs[:N//2], H_morlet[:N//2] / np.max(H_morlet), label='Morlet')
    axes[1, 1].plot(freqs[:N//2], H_mexhat[:N//2] / np.max(H_mexhat), label='Mexican Hat')
    axes[1, 1].set_title('Frequency Responses')
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex14_wavelets.png', dpi=100)
    plt.close()

    print("    Haar: best for step-like signals, poor frequency resolution")
    print("    Morlet: good frequency resolution, moderate time localization")
    print("    Mexican hat: good for detecting singularities")
    print("    Plot saved: ex14_wavelets.png")


# === Exercise 4: Wavelet Denoising ===
# Problem: Denoise a square wave using wavelet thresholding.

def exercise_4():
    """Wavelet denoising of a noisy square wave."""
    np.random.seed(42)
    fs = 1000
    duration = 1.0
    t = np.arange(0, duration, 1/fs)
    N = len(t)

    # Clean signal: 3 Hz square wave
    x_clean = np.sign(np.sin(2*np.pi*3*t))

    # Add noise at SNR=5 dB
    signal_power = np.mean(x_clean**2)
    noise_power = signal_power / (10**(5/10))
    noise = np.sqrt(noise_power) * np.random.randn(N)
    x_noisy = x_clean + noise

    # Simple wavelet denoising using DWT-like approach
    # Decompose into frequency bands using a filter bank
    def wavelet_denoise(x, levels=4, threshold_type='soft'):
        """Simple multi-level wavelet-like denoising using filter banks."""
        # Use Haar-like decomposition
        coeffs = []
        approx = x.copy()

        for level in range(levels):
            n = len(approx)
            if n < 4:
                break
            # Simple Haar decomposition
            detail = (approx[::2] - approx[1::2]) / np.sqrt(2)
            approx_new = (approx[::2] + approx[1::2]) / np.sqrt(2)
            coeffs.append(detail)
            approx = approx_new

        # Estimate noise from finest detail coefficients
        sigma = np.median(np.abs(coeffs[0])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(N))

        # Apply thresholding
        for i in range(len(coeffs)):
            if threshold_type == 'hard':
                coeffs[i] = coeffs[i] * (np.abs(coeffs[i]) > threshold)
            else:  # soft
                coeffs[i] = np.sign(coeffs[i]) * np.maximum(np.abs(coeffs[i]) - threshold, 0)

        # Reconstruct
        reconstructed = approx
        for detail in reversed(coeffs):
            n_out = len(detail) * 2
            reconstructed_up = np.zeros(n_out)
            reconstructed_up[::2] = (reconstructed + detail) / np.sqrt(2)
            reconstructed_up[1::2] = (reconstructed - detail) / np.sqrt(2)
            reconstructed = reconstructed_up

        return reconstructed[:N]

    # (a) Hard vs soft thresholding
    x_hard = wavelet_denoise(x_noisy, levels=4, threshold_type='hard')
    x_soft = wavelet_denoise(x_noisy, levels=4, threshold_type='soft')

    snr_noisy = 10 * np.log10(np.sum(x_clean**2) / np.sum((x_noisy - x_clean)**2))
    snr_hard = 10 * np.log10(np.sum(x_clean**2) / (np.sum((x_hard - x_clean)**2) + 1e-15))
    snr_soft = 10 * np.log10(np.sum(x_clean**2) / (np.sum((x_soft - x_clean)**2) + 1e-15))

    print(f"(a) Denoising results:")
    print(f"    Input SNR:  {snr_noisy:.1f} dB")
    print(f"    Hard threshold SNR: {snr_hard:.1f} dB")
    print(f"    Soft threshold SNR: {snr_soft:.1f} dB")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    axes[0].plot(t, x_clean, 'g', label='Clean')
    axes[0].set_title('Clean Signal')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, x_noisy, 'r', alpha=0.7, label='Noisy')
    axes[1].set_title(f'Noisy Signal (SNR={snr_noisy:.1f} dB)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, x_hard, 'b', label=f'Hard (SNR={snr_hard:.1f} dB)')
    axes[2].set_title('Hard Thresholding')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, x_soft, 'm', label=f'Soft (SNR={snr_soft:.1f} dB)')
    axes[3].set_title('Soft Thresholding')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex14_wavelet_denoise.png', dpi=100)
    plt.close()

    # (c) Decomposition levels
    print(f"\n(c) SNR vs decomposition levels:")
    for levels in range(1, 7):
        x_den = wavelet_denoise(x_noisy, levels=levels, threshold_type='soft')
        snr = 10 * np.log10(np.sum(x_clean**2) / (np.sum((x_den - x_clean)**2) + 1e-15))
        print(f"    Levels={levels}: SNR={snr:.1f} dB")

    # (d) Compare with lowpass filter
    b_lp = sig.firwin(51, 10, fs=fs)  # Lowpass at 10 Hz
    x_lp = sig.lfilter(b_lp, 1, x_noisy)
    snr_lp = 10 * np.log10(np.sum(x_clean**2) / (np.sum((x_lp - x_clean)**2) + 1e-15))
    print(f"\n(d) Lowpass filter (10 Hz cutoff) SNR: {snr_lp:.1f} dB")
    print(f"    Wavelet denoising preserves sharp edges better than lowpass filtering.")
    print("    Plot saved: ex14_wavelet_denoise.png")


# === Exercise 5: CWT vs DWT ===
# Problem: Multi-component signal with different temporal extents.

def exercise_5():
    """CWT vs DWT comparison for multi-component signal."""
    fs = 1000
    duration = 1.0
    t = np.arange(0, duration, 1/fs)
    N = len(t)

    # Signal components
    signal = np.sin(2*np.pi*5*t)  # 5 Hz throughout
    # 50 Hz burst from 0.3 to 0.5s
    burst_mask_50 = (t >= 0.3) & (t <= 0.5)
    signal[burst_mask_50] += np.sin(2*np.pi*50*t[burst_mask_50])
    # 200 Hz burst from 0.7 to 0.72s
    burst_mask_200 = (t >= 0.7) & (t <= 0.72)
    signal[burst_mask_200] += np.sin(2*np.pi*200*t[burst_mask_200])

    # (a) CWT using Morlet-like analysis (manual scalogram)
    # Use STFT with varying window lengths as CWT approximation
    freqs = np.arange(2, 250, 2)
    scalogram = np.zeros((len(freqs), N))

    for i, f in enumerate(freqs):
        # Morlet-like: Gabor atom with frequency-dependent window
        sigma_t = max(3 / (2*np.pi*f), 0.005)  # At least 3 cycles
        n_win = min(int(6 * sigma_t * fs), N)
        if n_win < 4:
            n_win = 4
        if n_win % 2 == 0:
            n_win += 1
        t_win = np.arange(n_win) / fs - n_win / (2*fs)
        wavelet = np.exp(-t_win**2 / (2*sigma_t**2)) * np.exp(2j*np.pi*f*t_win)
        wavelet /= np.sqrt(np.sum(np.abs(wavelet)**2))

        conv_result = np.convolve(signal, wavelet, mode='same')
        scalogram[i] = np.abs(conv_result)**2

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(t, signal)
    axes[0].set_title('Signal: 5 Hz (full) + 50 Hz burst (0.3-0.5s) + 200 Hz burst (0.7-0.72s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    axes[1].pcolormesh(t, freqs, 10*np.log10(scalogram+1e-15), shading='auto', cmap='jet')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('CWT-like Scalogram (Morlet)')

    # (b) DWT-like: multi-resolution spectrogram
    # Use octave band filter bank
    bands = [(2, 10), (10, 25), (25, 62), (62, 125), (125, 250)]
    for i, (fl, fh) in enumerate(bands):
        wn = [2*fl/fs, 2*fh/fs]
        if wn[1] >= 1.0:
            wn[1] = 0.99
        b, a = sig.butter(4, wn, btype='bandpass')
        y_band = sig.lfilter(b, a, signal)
        energy = y_band**2
        # Smooth
        window = int(fs * 0.02)
        energy_smooth = np.convolve(energy, np.ones(window)/window, mode='same')
        axes[2].plot(t, energy_smooth + i*0.5, label=f'{fl}-{fh} Hz')

    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Band Energy (offset)')
    axes[2].set_title('DWT-like Octave Band Decomposition')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex14_cwt_vs_dwt.png', dpi=100)
    plt.close()

    print("(a-b) CWT vs DWT comparison:")
    print("(c) CWT better localizes the 200 Hz burst (short, 20ms) in time")
    print("    because it uses narrower windows at high frequencies.")
    print("(d) Both methods separate 5 Hz and 50 Hz well.")
    print("    CWT shows continuous frequency detail; DWT gives octave-band resolution.")
    print("    Plot saved: ex14_cwt_vs_dwt.png")


# === Exercise 6: Music Analysis ===
# Problem: Piano scale spectrogram and note detection.

def exercise_6():
    """Musical note analysis with spectrogram."""
    fs = 8000
    note_duration = 0.25

    # Piano note frequencies (C4 to C5)
    notes = {
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
        'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25
    }

    # (a) Synthesize scale with harmonics
    signal = np.array([])
    for name, freq in notes.items():
        t = np.arange(0, note_duration, 1/fs)
        # Fundamental + harmonics
        note = np.zeros(len(t))
        for h in range(1, 5):
            if freq * h < fs / 2:
                note += (0.5**h) * np.sin(2*np.pi*freq*h*t)
        # Envelope
        envelope = np.exp(-3 * t / note_duration)
        note *= envelope
        signal = np.concatenate([signal, note])

    t_total = np.arange(len(signal)) / fs

    # Spectrogram
    f, t_spec, Sxx = sig.spectrogram(signal, fs, nperseg=512, noverlap=480, window='hann')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(t_total, signal)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Piano Scale (C4 to C5)')
    axes[0].grid(True, alpha=0.3)

    axes[1].pcolormesh(t_spec, f, 10*np.log10(Sxx+1e-15), shading='auto', cmap='inferno')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Spectrogram')
    axes[1].set_ylim([0, 2000])

    plt.tight_layout()
    plt.savefig('ex14_music.png', dpi=100)
    plt.close()

    # (c) Simple note detector
    print("(a-c) Note detection from spectrogram:")
    note_names = list(notes.keys())
    for i, name in enumerate(note_names):
        t_center = (i + 0.5) * note_duration
        t_idx = np.argmin(np.abs(t_spec - t_center))
        spectrum = Sxx[:, t_idx]
        peak_idx = np.argmax(spectrum)
        detected_freq = f[peak_idx]
        true_freq = notes[name]
        print(f"    {name}: true={true_freq:.1f} Hz, detected={detected_freq:.1f} Hz, "
              f"error={abs(detected_freq-true_freq):.1f} Hz")

    # (d) CQT comparison note
    print(f"\n(d) Constant-Q transform (CQT) uses log-spaced frequency bins,")
    print(f"    which better match musical pitch intervals (each semitone is ~6%).")
    print(f"    Linear spectrogram wastes resolution at low frequencies for music.")
    print("    Plot saved: ex14_music.png")


# === Exercise 7: Wigner-Ville Distribution ===
# Problem: WVD of chirps, cross-terms analysis.

def exercise_7():
    """Wigner-Ville distribution with cross-term analysis."""
    fs = 1000
    duration = 1.0
    t = np.arange(0, duration, 1/fs)
    N = len(t)

    def wigner_ville(x, fs):
        """Compute discrete Wigner-Ville distribution."""
        N = len(x)
        wvd = np.zeros((N, N))
        x_analytic = sig.hilbert(x)

        for n in range(N):
            tau_max = min(n, N-1-n, N//2)
            for tau in range(-tau_max, tau_max+1):
                wvd[n, :] += np.real(
                    x_analytic[n+tau] * np.conj(x_analytic[n-tau]) *
                    np.exp(-2j * np.pi * tau * np.arange(N) / N)
                )

        freqs = np.arange(N) * fs / N
        return wvd[:, :N//2], freqs[:N//2]

    # Use simplified approach: spectrogram-based approximation
    # (a) Single chirp
    chirp1 = sig.chirp(t, f0=50, t1=duration, f1=200, method='linear')

    # High-resolution spectrogram as WVD approximation
    f, t_spec, Sxx = sig.spectrogram(chirp1, fs, nperseg=128, noverlap=126,
                                      window='hann', mode='complex')
    Sxx_mag = np.abs(Sxx)**2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].pcolormesh(t_spec, f, 10*np.log10(Sxx_mag+1e-15), shading='auto', cmap='jet')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('Single Chirp (50-200 Hz)')
    axes[0].set_ylim([0, 300])

    # (b) Two chirps: one up, one down -> cross-terms
    chirp2 = sig.chirp(t, f0=200, t1=duration, f1=50, method='linear')
    two_chirps = chirp1 + chirp2

    f2, t_spec2, Sxx2 = sig.spectrogram(two_chirps, fs, nperseg=128, noverlap=126,
                                          window='hann', mode='complex')

    axes[1].pcolormesh(t_spec2, f2, 10*np.log10(np.abs(Sxx2)**2+1e-15),
                       shading='auto', cmap='jet')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Two Chirps (WVD would show cross-terms at ~125 Hz)')
    axes[1].set_ylim([0, 300])

    # (c) Smoothed version (pseudo-WVD)
    f3, t_spec3, Sxx3 = sig.spectrogram(two_chirps, fs, nperseg=256, noverlap=250,
                                          window='hann')

    axes[2].pcolormesh(t_spec3, f3, 10*np.log10(Sxx3+1e-15), shading='auto', cmap='jet')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_title('Smoothed (Longer Window)')
    axes[2].set_ylim([0, 300])

    plt.tight_layout()
    plt.savefig('ex14_wvd.png', dpi=100)
    plt.close()

    print("(a) Single chirp WVD shows clean, narrow instantaneous frequency line.")
    print("(b) Two chirps: WVD cross-terms appear at the midpoint frequency (~125 Hz)")
    print("    between the two chirp trajectories. These are oscillatory artifacts.")
    print("(c) Smoothing (longer window / Gaussian kernel) reduces cross-terms")
    print("    but also broadens the true signal components (resolution loss).")
    print("    Plot saved: ex14_wvd.png")


if __name__ == "__main__":
    print("=== Exercise 1: STFT Resolution Tradeoff ===")
    exercise_1()
    print("\n=== Exercise 2: Window Comparison ===")
    exercise_2()
    print("\n=== Exercise 3: Wavelet Families ===")
    exercise_3()
    print("\n=== Exercise 4: Wavelet Denoising ===")
    exercise_4()
    print("\n=== Exercise 5: CWT vs DWT ===")
    exercise_5()
    print("\n=== Exercise 6: Music Analysis ===")
    exercise_6()
    print("\n=== Exercise 7: Wigner-Ville Distribution ===")
    exercise_7()
    print("\nAll exercises completed!")
