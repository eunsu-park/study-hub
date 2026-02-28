"""
Exercises for Lesson 02: LTI Systems and Convolution
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import fft, ifft

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Analytical Convolution ===
# Problem: Compute convolutions by hand and verify with np.convolve().

def exercise_1():
    """Discrete-time convolution examples."""
    # 1. x[n] = {2, 1, -1}, h[n] = {1, 3, 2}
    x1 = np.array([2, 1, -1])
    h1 = np.array([1, 3, 2])
    y1 = np.convolve(x1, h1)
    # By hand: y[0]=2*1=2, y[1]=1*1+2*3=7, y[2]=-1*1+1*3+2*2=6,
    #           y[3]=-1*3+1*2=-1, y[4]=-1*2=-2
    print("1a) x = {2,1,-1} * h = {1,3,2}")
    print(f"   y = {y1}")
    print(f"   Expected: [2, 7, 6, -1, -2]")
    print()

    # 2. x[n] = u[n]-u[n-4], h[n] = (0.5)^n * u[n]
    N = 100
    n = np.arange(N)
    x2 = np.where((n >= 0) & (n < 4), 1.0, 0.0)
    h2 = 0.5 ** n
    y2 = np.convolve(x2, h2)[:N]

    # Analytical: y[n] = sum_{k=0}^{3} 0.5^{n-k} for n >= 0
    # = 0.5^n * sum_{k=0}^{min(n,3)} 0.5^{-k} = 0.5^n * sum_{k=0}^{min(n,3)} 2^k
    y2_analytical = np.zeros(N)
    for nn in range(N):
        upper = min(nn, 3)
        y2_analytical[nn] = (0.5 ** nn) * sum(2.0 ** k for k in range(upper + 1))

    print("1b) x = u[n]-u[n-4], h = 0.5^n*u[n]")
    print(f"   First 8 values (numerical): {y2[:8]}")
    print(f"   First 8 values (analytical): {y2_analytical[:8]}")
    print(f"   Max error: {np.max(np.abs(y2 - y2_analytical)):.2e}")
    print()

    # 3. x[n] = 0.8^n * u[n], h[n] = 0.6^n * u[n]
    x3 = 0.8 ** n
    h3 = 0.6 ** n
    y3 = np.convolve(x3, h3)[:N]

    # Analytical: y[n] = sum_{k=0}^{n} 0.8^k * 0.6^{n-k}
    # = 0.6^n * sum_{k=0}^{n} (0.8/0.6)^k = 0.6^n * (1 - (4/3)^{n+1}) / (1 - 4/3)
    # = 0.6^n * 3 * ((4/3)^{n+1} - 1) = 3 * (0.8^{n+1} - 0.6^{n+1}) / (0.8 - 0.6)
    # = (0.8^{n+1} - 0.6^{n+1}) / (0.8 - 0.6) * ... let me use the closed form:
    # y[n] = (0.8^{n+1} - 0.6^{n+1}) / (0.8 - 0.6)
    y3_analytical = (0.8 ** (n + 1) - 0.6 ** (n + 1)) / (0.8 - 0.6)

    print("1c) x = 0.8^n*u[n], h = 0.6^n*u[n]")
    print(f"   First 8 values (numerical):  {y3[:8]}")
    print(f"   First 8 values (analytical): {y3_analytical[:8]}")
    print(f"   Max error: {np.max(np.abs(y3[:50] - y3_analytical[:50])):.2e}")


# === Exercise 2: Continuous-Time Convolution ===
# Problem: Compute continuous-time convolutions.

def exercise_2():
    """Continuous-time convolution (numerical verification)."""
    dt = 0.001
    t = np.arange(-2, 20, dt)

    # 1. y(t) = e^{-t}u(t) * e^{-2t}u(t)
    # Analytical: y(t) = (e^{-t} - e^{-2t}) * u(t)
    x1 = np.exp(-t) * (t >= 0)
    h1 = np.exp(-2 * t) * (t >= 0)
    y1_num = np.convolve(x1, h1, mode='full')[:len(t)] * dt
    y1_analytical = (np.exp(-t) - np.exp(-2 * t)) * (t >= 0)

    print("2a) e^{-t}u(t) * e^{-2t}u(t) = (e^{-t} - e^{-2t})u(t)")
    print(f"   Max error: {np.max(np.abs(y1_num[:5000] - y1_analytical[:5000])):.4f}")
    print()

    # 2. y(t) = u(t) * u(t) = t*u(t) (ramp function)
    # This reveals instability: output grows without bound
    x2 = (t >= 0).astype(float)
    y2_num = np.convolve(x2, x2, mode='full')[:len(t)] * dt
    y2_analytical = t * (t >= 0)

    print("2b) u(t) * u(t) = t*u(t) (ramp)")
    print("   Output grows without bound -> accumulator/integrator is unstable")
    print(f"   y(5.0) numerical = {y2_num[int(5.0 / dt)]:.4f}, analytical = 5.0")
    print(f"   y(10.0) numerical = {y2_num[int(10.0 / dt)]:.4f}, analytical = 10.0")
    print()

    # 3. y(t) = rect(t/2) * e^{-t}u(t)
    # rect(t/2): 1 for |t| < 1, zero otherwise
    # y(t) = integral_{max(0, t-1)}^{min(t+1, inf)} e^{-tau} dtau (for appropriate ranges)
    x3 = np.where(np.abs(t) <= 1, 1.0, 0.0)
    h3 = np.exp(-t) * (t >= 0)
    y3_num = np.convolve(x3, h3, mode='full')[:len(t)] * dt

    print("2c) rect(t/2) * e^{-t}u(t)")
    print(f"   y(0.0) = {y3_num[int(2.0 / dt)]:.4f}")  # offset by 2 since t starts at -2
    print(f"   y(2.0) = {y3_num[int(4.0 / dt)]:.4f}")
    print(f"   y(5.0) = {y3_num[int(7.0 / dt)]:.4f}")


# === Exercise 3: System Analysis ===
# Problem: Analyze the system y[n] = 0.5*y[n-1] + x[n].

def exercise_3():
    """Analysis of first-order recursive system."""
    # 1. Impulse response: h[n] = (0.5)^n * u[n]
    N = 50
    n = np.arange(N)
    h = 0.5 ** n
    print("3a) Impulse response: h[n] = (0.5)^n * u[n]")
    print(f"   h[0..5] = {h[:6]}")
    print()

    # 2. BIBO stability: sum |h[n]| = 1/(1-0.5) = 2 < inf -> STABLE
    print("3b) BIBO stability:")
    print(f"   sum |h[n]| = {np.sum(np.abs(h)):.4f} (analytical: 2.0)")
    print("   BIBO STABLE (sum is finite)")
    print()

    # 3. Step response: s[n] = sum_{k=0}^{n} h[k] = (1 - 0.5^{n+1})/(1-0.5) = 2(1-0.5^{n+1})
    s = np.cumsum(h)
    s_analytical = 2 * (1 - 0.5 ** (n + 1))
    print("3c) Step response: s[n] = 2*(1 - 0.5^{n+1})")
    print(f"   s[0..5] (numerical):  {s[:6]}")
    print(f"   s[0..5] (analytical): {s_analytical[:6]}")
    print()

    # 4. Frequency response: H(e^{jw}) = 1 / (1 - 0.5*e^{-jw})
    w = np.linspace(0, np.pi, 512)
    H = 1.0 / (1 - 0.5 * np.exp(-1j * w))

    print("3d) Frequency response: H(e^{jw}) = 1/(1 - 0.5*e^{-jw})")
    print(f"   |H(0)| = {np.abs(H[0]):.4f} (analytical: 2.0)")
    print(f"   |H(pi)| = {np.abs(H[-1]):.4f} (analytical: {1 / 1.5:.4f})")

    # 5. Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(w / np.pi, 20 * np.log10(np.abs(H)), 'b-', linewidth=2)
    axes[0].set_title('Magnitude Response')
    axes[0].set_xlabel('Normalized Frequency (x pi rad/sample)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(w / np.pi, np.angle(H) * 180 / np.pi, 'r-', linewidth=2)
    axes[1].set_title('Phase Response')
    axes[1].set_xlabel('Normalized Frequency (x pi rad/sample)')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex02_system_analysis.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("\n   Plot saved to ex02_system_analysis.png")


# === Exercise 4: Cascade vs. Parallel ===
# Problem: Analyze cascade and parallel connections.

def exercise_4():
    """Cascade and parallel system connections."""
    N = 100
    n = np.arange(N)
    h1 = 0.7 ** n
    h2 = 0.5 ** n

    # 1. Cascade: h_cascade = h1 * h2 (convolution)
    h_cascade = np.convolve(h1, h2)[:N]
    # Analytical: h_cascade[n] = (0.7^{n+1} - 0.5^{n+1}) / (0.7 - 0.5)
    h_cascade_analytical = (0.7 ** (n + 1) - 0.5 ** (n + 1)) / 0.2
    print("4a) Cascade: h1 * h2")
    print(f"   First 5 values: {h_cascade[:5]}")
    print(f"   Analytical:     {h_cascade_analytical[:5]}")
    print()

    # 2. Parallel: h_parallel = h1 + h2
    h_parallel = h1 + h2
    print("4b) Parallel: h1 + h2")
    print(f"   First 5 values: {h_parallel[:5]}")
    print()

    # 3. Commutativity verification
    h_cascade_21 = np.convolve(h2, h1)[:N]
    comm_error = np.max(np.abs(h_cascade - h_cascade_21))
    print(f"4c) Commutativity error: {comm_error:.2e}")
    print()

    # 4. Output for x[n] = delta[n] - 0.3*delta[n-1]
    x = np.zeros(N)
    x[0] = 1.0
    x[1] = -0.3

    y_cascade = np.convolve(x, h_cascade)[:N]
    y_parallel = np.convolve(x, h_parallel)[:N]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].stem(n[:30], y_cascade[:30], linefmt='b-', markerfmt='bo', basefmt='k-')
    axes[0].set_title('Cascade Output')
    axes[0].set_xlabel('n')
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(n[:30], y_parallel[:30], linefmt='r-', markerfmt='ro', basefmt='k-')
    axes[1].set_title('Parallel Output')
    axes[1].set_xlabel('n')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex02_cascade_parallel.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("4d) Plot saved to ex02_cascade_parallel.png")


# === Exercise 5: Moving Average Filter Analysis ===
# Problem: Analyze M-point moving average filter.

def exercise_5():
    """Moving average filter frequency response analysis."""
    M_values = [3, 7, 15, 31]
    w = np.linspace(0, np.pi, 1024)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for M in M_values:
        # H(e^{jw}) = (1/M) * (1 - e^{-jMw}) / (1 - e^{-jw})
        H = np.zeros(len(w), dtype=complex)
        for i, wi in enumerate(w):
            if np.abs(wi) < 1e-10:
                H[i] = 1.0
            else:
                H[i] = (1 / M) * (1 - np.exp(-1j * M * wi)) / (1 - np.exp(-1j * wi))

        H_dB = 20 * np.log10(np.abs(H) + 1e-15)
        ax.plot(w / np.pi, H_dB, linewidth=1.5, label=f'M={M}')

        # 3-dB bandwidth
        H_mag = np.abs(H)
        idx_3dB = np.where(H_mag < 1 / np.sqrt(2))[0]
        if len(idx_3dB) > 0:
            bw = w[idx_3dB[0]] / np.pi
            print(f"M={M}: 3-dB bandwidth = {bw:.4f} * pi rad/sample")

        # Nulls
        null_indices = np.where(np.abs(H) < 0.01)[0]
        null_freqs = w[null_indices] / np.pi
        # Remove duplicates by rounding
        if len(null_freqs) > 0:
            unique_nulls = [null_freqs[0]]
            for nf in null_freqs[1:]:
                if nf - unique_nulls[-1] > 0.05:
                    unique_nulls.append(nf)
            print(f"  Nulls at: {[f'{x:.3f}*pi' for x in unique_nulls[:5]]}")

    ax.set_xlabel('Normalized Frequency (x pi rad/sample)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Moving Average Filter Frequency Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-60, 5])
    plt.tight_layout()
    plt.savefig('ex02_moving_average.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to ex02_moving_average.png")

    # Test with signal containing 50 Hz and 400 Hz at fs=1000 Hz
    fs = 1000
    t = np.arange(0, 0.5, 1 / fs)
    x_test = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 400 * t)
    print(f"\nFiltering signal with 50 Hz + 400 Hz components (fs={fs} Hz):")
    for M in M_values:
        h = np.ones(M) / M
        y = np.convolve(x_test, h, mode='same')
        # Measure residual of each component
        energy_50 = np.sum((np.sin(2 * np.pi * 50 * t)) ** 2)
        # Estimate attenuation at each frequency
        w_50 = 2 * np.pi * 50 / fs
        w_400 = 2 * np.pi * 400 / fs
        H_50 = np.abs((1 / M) * np.sin(M * w_50 / 2) / np.sin(w_50 / 2)) if np.abs(np.sin(w_50 / 2)) > 1e-10 else 1.0
        H_400 = np.abs((1 / M) * np.sin(M * w_400 / 2) / np.sin(w_400 / 2)) if np.abs(np.sin(w_400 / 2)) > 1e-10 else 1.0
        print(f"  M={M:2d}: |H(50Hz)|={H_50:.4f}, |H(400Hz)|={H_400:.4f}, "
              f"ratio={H_50 / (H_400 + 1e-10):.2f}")


# === Exercise 6: Convolution in Practice ===
# Problem: Chirp with echo system, FFT vs direct convolution.

def exercise_6():
    """Practical convolution with chirp and echo system."""
    import time

    fs = 8000
    duration = 1.0
    t = np.arange(0, duration, 1 / fs)

    # 1. Generate chirp
    x = sig.chirp(t, f0=100, f1=2000, t1=duration)

    # 2. Echo system: 3 reflections
    delays_ms = [50, 120, 200]
    amplitudes = [0.7, 0.4, 0.2]
    h_len = int(max(delays_ms) * fs / 1000) + 1
    h = np.zeros(h_len)
    h[0] = 1.0  # direct path
    for d, a in zip(delays_ms, amplitudes):
        h[int(d * fs / 1000)] = a

    # 3. Convolve
    y_direct_start = time.perf_counter()
    y_direct = np.convolve(x, h)
    t_direct = time.perf_counter() - y_direct_start

    # 4. FFT convolution
    y_fft_start = time.perf_counter()
    n_fft = len(x) + len(h) - 1
    X = fft(x, n_fft)
    H = fft(h, n_fft)
    y_fft = np.real(ifft(X * H))
    t_fft = time.perf_counter() - y_fft_start

    print(f"Chirp length: {len(x)} samples")
    print(f"Echo IR length: {len(h)} samples")
    print(f"Direct convolution time: {t_direct * 1000:.2f} ms")
    print(f"FFT convolution time:    {t_fft * 1000:.2f} ms")
    print(f"Speedup: {t_direct / t_fft:.1f}x")
    print(f"Max difference: {np.max(np.abs(y_direct - y_fft[:len(y_direct)])):.2e}")

    # 5. Plot spectrograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].specgram(x, NFFT=256, Fs=fs, noverlap=128, cmap='viridis')
    axes[0].set_title('Input (Chirp)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')

    axes[1].specgram(y_direct[:len(x) + 2000], NFFT=256, Fs=fs, noverlap=128, cmap='viridis')
    axes[1].set_title('Output (Chirp + Echoes)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.savefig('ex02_chirp_echo.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex02_chirp_echo.png")


# === Exercise 7: System Identification ===
# Problem: Identify an unknown LTI system using different methods.

def exercise_7():
    """System identification via impulse, step, and cross-correlation."""
    def black_box_system(x):
        """Unknown LTI system."""
        h = np.array([0.2, 0.5, 1.0, 0.5, 0.2, -0.1, -0.05])
        return np.convolve(x, h, mode='full')[:len(x)]

    h_true = np.array([0.2, 0.5, 1.0, 0.5, 0.2, -0.1, -0.05])
    N = 10000
    L = len(h_true)

    # Method 1: Unit impulse
    x_impulse = np.zeros(N)
    x_impulse[0] = 1.0
    h_impulse = black_box_system(x_impulse)[:L]
    err1 = np.max(np.abs(h_impulse - h_true))
    energy1 = np.sum(x_impulse ** 2)
    print(f"Method 1 (Impulse): error = {err1:.2e}, input energy = {energy1:.1f}")

    # Method 2: Unit step -> differentiate
    x_step = np.ones(N)
    s = black_box_system(x_step)
    h_step = np.zeros(L)
    h_step[0] = s[0]
    for i in range(1, L):
        h_step[i] = s[i] - s[i - 1]
    err2 = np.max(np.abs(h_step - h_true))
    energy2 = np.sum(x_step ** 2)
    print(f"Method 2 (Step):    error = {err2:.2e}, input energy = {energy2:.1f}")

    # Method 3: White noise + cross-correlation
    np.random.seed(42)
    x_noise = np.random.randn(N)
    y_noise = black_box_system(x_noise)
    # Cross-correlation estimate: R_xy[k] / R_xx[0] approximates h[k]
    Rxy = np.correlate(y_noise, x_noise, mode='full')
    mid = len(Rxy) // 2
    h_xcorr = Rxy[mid:mid + L] / np.sum(x_noise ** 2)
    err3 = np.max(np.abs(h_xcorr - h_true))
    energy3 = np.sum(x_noise ** 2)
    print(f"Method 3 (XCorr):   error = {err3:.2e}, input energy = {energy3:.1f}")
    print()
    print(f"True h:     {h_true}")
    print(f"Impulse h:  {h_impulse}")
    print(f"Step h:     {h_step}")
    print(f"XCorr h:    {np.round(h_xcorr, 4)}")
    print()
    print("Method 1 (impulse) gives exact result with minimum energy (1.0).")
    print("Method 3 (cross-correlation) uses most energy but works with broadband excitation.")


# === Exercise 8: Deconvolution ===
# Problem: Deconvolution using FFT and Wiener deconvolution.

def exercise_8():
    """Deconvolution: direct FFT and Wiener methods."""
    np.random.seed(42)

    # 1. Sparse signal
    N = 100
    x = np.zeros(N)
    positions = [10, 30, 50, 70, 90]
    values = [1.0, -0.5, 2.0, 0.8, -1.5]
    for p, v in zip(positions, values):
        x[p] = v

    # 2. Convolve with known h
    h = np.array([1.0, 0.5, 0.25])
    y_clean = np.convolve(x, h)[:N]

    # 3. Add noise
    noise = 0.01 * np.random.randn(N)
    y_noisy = y_clean + noise

    # 4. Direct FFT deconvolution: X = Y / H
    N_fft = 256
    Y = fft(y_noisy, N_fft)
    H = fft(h, N_fft)
    X_direct = np.real(ifft(Y / (H + 1e-15)))[:N]

    # 5. Wiener deconvolution: X_hat = H* Y / (|H|^2 + lambda)
    lam = 0.01
    X_wiener = np.real(ifft(np.conj(H) * Y / (np.abs(H) ** 2 + lam)))[:N]

    err_direct = np.sqrt(np.mean((X_direct - x) ** 2))
    err_wiener = np.sqrt(np.mean((X_wiener - x) ** 2))

    print(f"Direct deconvolution RMSE: {err_direct:.6f}")
    print(f"Wiener deconvolution RMSE: {err_wiener:.6f}")
    print()
    print("Direct deconvolution amplifies noise at frequencies where |H| is small.")
    print("Wiener deconvolution adds regularization (lambda) to suppress noise amplification.")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    axes[0].stem(np.arange(N), x, linefmt='b-', markerfmt='bo', basefmt='k-')
    axes[0].set_title('Original sparse signal x[n]')

    axes[1].plot(np.arange(N), y_noisy, 'r-')
    axes[1].set_title('Noisy convolved signal y[n]')

    axes[2].stem(np.arange(N), X_direct, linefmt='g-', markerfmt='go', basefmt='k-')
    axes[2].set_title(f'Direct FFT deconvolution (RMSE={err_direct:.4f})')
    axes[2].set_ylim([-5, 5])

    axes[3].stem(np.arange(N), X_wiener, linefmt='m-', markerfmt='mo', basefmt='k-')
    axes[3].set_title(f'Wiener deconvolution (RMSE={err_wiener:.4f})')

    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex02_deconvolution.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex02_deconvolution.png")


if __name__ == "__main__":
    print("=== Exercise 1: Analytical Convolution ===")
    exercise_1()
    print("\n=== Exercise 2: Continuous-Time Convolution ===")
    exercise_2()
    print("\n=== Exercise 3: System Analysis ===")
    exercise_3()
    print("\n=== Exercise 4: Cascade vs. Parallel ===")
    exercise_4()
    print("\n=== Exercise 5: Moving Average Filter Analysis ===")
    exercise_5()
    print("\n=== Exercise 6: Convolution in Practice ===")
    exercise_6()
    print("\n=== Exercise 7: System Identification ===")
    exercise_7()
    print("\n=== Exercise 8: Deconvolution ===")
    exercise_8()
    print("\nAll exercises completed!")
