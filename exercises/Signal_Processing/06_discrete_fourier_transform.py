"""
Exercises for Lesson 06: Discrete Fourier Transform
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal as sig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: DFT by Hand ===
# Problem: Compute 4-point DFT of x = {1, 0, -1, 0}.

def exercise_1():
    """Manual 4-point DFT computation and verification."""
    x = np.array([1, 0, -1, 0])
    N = len(x)

    # Manual computation
    W = np.exp(-1j * 2 * np.pi / N)
    X_manual = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X_manual[k] += x[n] * W ** (k * n)

    print("4-point DFT of x = {1, 0, -1, 0}:")
    print(f"  X[0] = {x[0]}*1 + {x[1]}*1 + {x[2]}*1 + {x[3]}*1 = {X_manual[0]:.1f}")
    print(f"  X[1] = {x[0]}*1 + {x[1]}*W + {x[2]}*W^2 + {x[3]}*W^3 = {X_manual[1]:.1f}")
    print(f"  X[2] = {x[0]}*1 + {x[1]}*W^2 + {x[2]}*W^4 + {x[3]}*W^6 = {X_manual[2]:.1f}")
    print(f"  X[3] = {x[0]}*1 + {x[1]}*W^3 + {x[2]}*W^6 + {x[3]}*W^9 = {X_manual[3]:.1f}")
    print()

    # (b) Verify with FFT
    X_fft = np.fft.fft(x)
    print(f"  FFT verification: {X_fft}")
    print(f"  Match: {np.allclose(X_manual, X_fft)}")
    print()

    # (c) Inverse DFT
    x_recovered = np.fft.ifft(X_fft)
    print(f"  IDFT recovery: {np.real(x_recovered)}")
    print(f"  Match: {np.allclose(x, np.real(x_recovered))}")


# === Exercise 2: Circular Convolution ===
# Problem: 4-point circular convolution.

def exercise_2():
    """Circular convolution computation and comparison with linear."""
    x1 = np.array([1, 2, 3, 4])
    x2 = np.array([1, 0, 1, 0])
    N = 4

    # (a) Circular convolution by hand
    y_circ = np.zeros(N)
    for n in range(N):
        for k in range(N):
            y_circ[n] += x1[k] * x2[(n - k) % N]

    print(f"(a) 4-point circular convolution:")
    print(f"    x1 = {x1}, x2 = {x2}")
    print(f"    y_circ = {y_circ}")
    print()

    # (b) DFT multiplication property
    X1 = np.fft.fft(x1)
    X2 = np.fft.fft(x2)
    y_dft = np.real(np.fft.ifft(X1 * X2))
    print(f"(b) Via DFT: y = IDFT{{X1 * X2}} = {y_dft}")
    print(f"    Match: {np.allclose(y_circ, y_dft)}")
    print()

    # (c) Linear convolution comparison
    y_linear = np.convolve(x1, x2)
    print(f"(c) Linear convolution: {y_linear}")
    print(f"    Linear conv length: {len(y_linear)}")

    # Zero-pad for linear = circular
    N_pad = len(x1) + len(x2) - 1
    X1_pad = np.fft.fft(x1, N_pad)
    X2_pad = np.fft.fft(x2, N_pad)
    y_circ_pad = np.real(np.fft.ifft(X1_pad * X2_pad))
    print(f"    Zero-padded circular (N={N_pad}): {y_circ_pad}")
    print(f"    Matches linear: {np.allclose(y_linear, y_circ_pad)}")


# === Exercise 3: Frequency Resolution Challenge ===
# Problem: Resolve two close sinusoids (100 Hz and 103 Hz).

def exercise_3():
    """Frequency resolution of closely spaced tones."""
    fs = 1000
    f1, f2 = 100, 103
    delta_f = f2 - f1

    # (a) Minimum N for resolution
    N_min = int(np.ceil(fs / delta_f))
    print(f"(a) To resolve {delta_f} Hz: N >= fs/delta_f = {fs}/{delta_f} = {N_min:.0f} samples")
    print()

    # (b) Try N=256 and N=512
    for N in [256, 512]:
        t = np.arange(N) / fs
        x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

        X = np.fft.fft(x)
        freqs = np.fft.fftfreq(N, 1 / fs)
        mag = np.abs(X[:N // 2])
        f_pos = freqs[:N // 2]

        # Find peaks near 100 Hz
        mask = (f_pos > 90) & (f_pos < 115)
        peaks_idx = np.where(mask)[0]
        peak_vals = mag[peaks_idx]
        local_peaks = []
        for i in range(1, len(peak_vals) - 1):
            if peak_vals[i] > peak_vals[i - 1] and peak_vals[i] > peak_vals[i + 1]:
                local_peaks.append(f_pos[peaks_idx[i]])

        df = fs / N
        resolved = len(local_peaks) >= 2
        print(f"(b) N={N}: df={df:.2f} Hz, peaks found: {local_peaks}, "
              f"resolved: {resolved}")

    # (c) With Hann window
    print("\n(c) With Hann window:")
    for N in [256, 512]:
        t = np.arange(N) / fs
        x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
        window = np.hanning(N)
        X = np.fft.fft(x * window)
        freqs = np.fft.fftfreq(N, 1 / fs)
        mag = np.abs(X[:N // 2])
        f_pos = freqs[:N // 2]

        mask = (f_pos > 90) & (f_pos < 115)
        peaks_idx = np.where(mask)[0]
        peak_vals = mag[peaks_idx]
        local_peaks = []
        for i in range(1, len(peak_vals) - 1):
            if peak_vals[i] > peak_vals[i - 1] and peak_vals[i] > peak_vals[i + 1]:
                local_peaks.append(f_pos[peaks_idx[i]])

        resolved = len(local_peaks) >= 2
        print(f"    N={N}: peaks: {local_peaks}, resolved: {resolved}")

    print("\nWindowing widens the main lobe (hurts resolution) but reduces sidelobes")
    print("(helps detect weak signals near strong ones)")


# === Exercise 4: Cooley-Tukey FFT Implementation ===
# Problem: Implement radix-2 DIF FFT.

def exercise_4():
    """Radix-2 DIF (Decimation-In-Frequency) FFT implementation."""
    def fft_dif(x):
        """Radix-2 DIF FFT (recursive)."""
        N = len(x)
        if N == 1:
            return x.copy()

        # Split into first half and second half
        x = x.astype(complex)
        half = N // 2
        W = np.exp(-2j * np.pi / N)

        # Butterfly: combine pairs
        x_top = x[:half] + x[half:]
        x_bot = (x[:half] - x[half:]) * np.array([W ** k for k in range(half)])

        # Recurse
        X_even = fft_dif(x_top)
        X_odd = fft_dif(x_bot)

        # Interleave
        X = np.zeros(N, dtype=complex)
        X[0::2] = X_even
        X[1::2] = X_odd
        return X

    # Test
    N = 1024
    np.random.seed(42)
    x = np.random.randn(N)

    X_custom = fft_dif(x)
    X_numpy = np.fft.fft(x)

    error = np.max(np.abs(X_custom - X_numpy))
    print(f"DIF FFT implementation (N={N}):")
    print(f"  Max error vs numpy: {error:.2e}")
    print(f"  Match: {error < 1e-10}")
    print()

    # Benchmark
    import time
    sizes = [2 ** k for k in range(10, 16)]
    print(f"{'N':<10} {'Custom (ms)':<15} {'NumPy (ms)':<15}")
    for N in sizes:
        x = np.random.randn(N)

        t0 = time.perf_counter()
        for _ in range(3):
            fft_dif(x)
        t_custom = (time.perf_counter() - t0) / 3 * 1000

        t0 = time.perf_counter()
        for _ in range(100):
            np.fft.fft(x)
        t_numpy = (time.perf_counter() - t0) / 100 * 1000

        print(f"{N:<10} {t_custom:<15.3f} {t_numpy:<15.3f}")


# === Exercise 5: Spectral Analysis of Synthetic Audio ===
# Problem: Analyze a synthetic audio signal.

def exercise_5():
    """Spectral analysis of a multi-component signal."""
    fs = 8000
    duration = 1.0
    t = np.arange(0, duration, 1 / fs)
    N = len(t)

    # Synthesize signal with harmonics (like a musical note)
    f0 = 440  # A4
    np.random.seed(42)
    x = np.zeros(N)
    for k in range(1, 6):
        x += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t)
    x += 0.05 * np.random.randn(N)

    # (b) Magnitude spectrum with Hann window
    window = np.hanning(N)
    X = np.fft.fft(x * window)
    freqs = np.fft.fftfreq(N, 1 / fs)
    mag = 20 * np.log10(np.abs(X[:N // 2]) / N + 1e-10)

    print("Dominant frequencies:")
    f_pos = freqs[:N // 2]
    mag_lin = np.abs(X[:N // 2])
    for k in range(1, 6):
        idx = np.argmin(np.abs(f_pos - k * f0))
        print(f"  Harmonic {k}: {f_pos[idx]:.1f} Hz, mag = {mag[idx]:.1f} dB")

    # (c) Spectrogram
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(f_pos, mag)
    axes[0].set_title('Magnitude Spectrum')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_xlim([0, 3000])
    axes[0].grid(True, alpha=0.3)

    axes[1].specgram(x, NFFT=512, Fs=fs, noverlap=256, cmap='viridis')
    axes[1].set_title('Spectrogram')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.savefig('ex06_spectral_analysis.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex06_spectral_analysis.png")


# === Exercise 6: Windowing Experiment ===
# Problem: Detect weak signal near strong signal using windows.

def exercise_6():
    """Window comparison for weak signal detection."""
    fs = 1000
    N = 256
    n = np.arange(N)
    x = np.sin(2 * np.pi * 100 * n / fs) + 0.01 * np.sin(2 * np.pi * 150 * n / fs)

    windows = {
        'Rectangular': np.ones(N),
        'Hann': np.hanning(N),
        'Hamming': np.hamming(N),
        'Blackman': np.blackman(N),
        'Kaiser (b=12)': np.kaiser(N, 12)
    }

    fig, axes = plt.subplots(len(windows), 1, figsize=(12, 3 * len(windows)))

    print(f"{'Window':<18} {'Peak sidelobe (dB)':<22} {'150Hz visible?':<16}")
    print("-" * 56)

    for idx, (name, w) in enumerate(windows.items()):
        X = np.fft.fft(x * w, 4 * N)  # zero-pad for interpolation
        freqs = np.fft.fftfreq(4 * N, 1 / fs)
        mag = 20 * np.log10(np.abs(X[:2 * N]) / np.max(np.abs(X)) + 1e-15)
        f_pos = freqs[:2 * N]

        # Find 150 Hz peak relative to 100 Hz
        idx_100 = np.argmin(np.abs(f_pos - 100))
        idx_150 = np.argmin(np.abs(f_pos - 150))
        mag_100 = mag[idx_100]
        mag_150 = mag[idx_150]

        # Check sidelobe at 150 Hz region
        mask_150 = (f_pos > 140) & (f_pos < 160)
        local_peak = np.max(mag[mask_150])
        visible = local_peak > -50

        # Peak sidelobe of window
        W = np.fft.fft(w, 4 * N)
        W_mag = 20 * np.log10(np.abs(W[:2 * N]) / np.max(np.abs(W)) + 1e-15)
        # Skip main lobe
        mainlobe_end = np.where(W_mag[1:] < W_mag[:-1])[0]
        if len(mainlobe_end) > 0:
            sidelobe_peak = np.max(W_mag[mainlobe_end[0]:])
        else:
            sidelobe_peak = -100

        print(f"{name:<18} {sidelobe_peak:<22.1f} {'YES' if visible else 'NO':<16}")

        axes[idx].plot(f_pos, mag)
        axes[idx].set_title(f'{name}: 150Hz peak at {local_peak:.1f} dB')
        axes[idx].set_xlim([50, 200])
        axes[idx].set_ylim([-80, 5])
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axvline(x=150, color='r', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex06_windowing.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to ex06_windowing.png")


# === Exercise 7: 2D FFT Application ===
# Problem: Image filtering in frequency domain.

def exercise_7():
    """2D FFT for image filtering."""
    # Create synthetic grayscale image
    N = 256
    x, y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))

    # Simple image: circle + background gradient
    image = np.zeros((N, N))
    circle = (x ** 2 + y ** 2) < 0.3 ** 2
    image[circle] = 1.0
    image += 0.3 * x  # gradient
    np.random.seed(42)
    image += 0.1 * np.random.randn(N, N)  # noise

    # (a-b) 2D FFT
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)
    magnitude = np.log10(np.abs(F_shifted) + 1)

    # (c) Lowpass filter (circular mask)
    center = N // 2
    Y, X = np.ogrid[:N, :N]
    dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 1].imshow(magnitude, cmap='viridis')
    axes[0, 1].set_title('2D FFT Magnitude (log)')

    for idx, cutoff in enumerate([20, 40, 80]):
        mask = dist <= cutoff
        F_filtered = F_shifted * mask
        img_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))

        ax_idx = idx if idx < 2 else idx
        if idx == 0:
            axes[0, 2].imshow(img_filtered, cmap='gray')
            axes[0, 2].set_title(f'Lowpass (r={cutoff})')
        elif idx == 1:
            axes[1, 0].imshow(img_filtered, cmap='gray')
            axes[1, 0].set_title(f'Lowpass (r={cutoff})')
        else:
            axes[1, 1].imshow(img_filtered, cmap='gray')
            axes[1, 1].set_title(f'Lowpass (r={cutoff})')

    # (d) Highpass filter (edge detection)
    cutoff_hp = 10
    mask_hp = dist > cutoff_hp
    F_highpass = F_shifted * mask_hp
    img_highpass = np.real(np.fft.ifft2(np.fft.ifftshift(F_highpass)))
    axes[1, 2].imshow(np.abs(img_highpass), cmap='gray')
    axes[1, 2].set_title(f'Highpass (r>{cutoff_hp}) - Edges')

    plt.tight_layout()
    plt.savefig('ex06_2d_fft.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("2D FFT filtering results saved to ex06_2d_fft.png")


if __name__ == "__main__":
    print("=== Exercise 1: DFT by Hand ===")
    exercise_1()
    print("\n=== Exercise 2: Circular Convolution ===")
    exercise_2()
    print("\n=== Exercise 3: Frequency Resolution Challenge ===")
    exercise_3()
    print("\n=== Exercise 4: Cooley-Tukey FFT (DIF) ===")
    exercise_4()
    print("\n=== Exercise 5: Spectral Analysis ===")
    exercise_5()
    print("\n=== Exercise 6: Windowing Experiment ===")
    exercise_6()
    print("\n=== Exercise 7: 2D FFT Application ===")
    exercise_7()
    print("\nAll exercises completed!")
