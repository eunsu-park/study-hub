"""
Exercises for Lesson 04: Continuous Fourier Transform
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Transform Computation ===
# Problem: Compute Fourier transforms analytically and verify with Python.

def exercise_1():
    """Fourier transform computation and verification."""
    N = 8192
    dt = 0.001
    t = np.arange(-10, 10, dt)
    freqs = fftfreq(len(t), dt)
    omega = 2 * np.pi * freqs

    # 1a) x(t) = e^{-3t}u(t) - e^{-5t}u(t)
    # X(omega) = 1/(3+jw) - 1/(5+jw) = 2/((3+jw)(5+jw))
    x1 = np.exp(-3 * t) * (t >= 0) - np.exp(-5 * t) * (t >= 0)
    X1_num = fft(x1) * dt
    X1_analytical = 1 / (3 + 1j * omega) - 1 / (5 + 1j * omega)

    # Compare at a few frequencies
    idx = np.where((freqs > 0) & (freqs < 5))[0][::100]
    err1 = np.max(np.abs(X1_num[idx] - X1_analytical[idx]))
    print(f"1a) e^{{-3t}}u(t) - e^{{-5t}}u(t)")
    print(f"    X(w) = 1/(3+jw) - 1/(5+jw)")
    print(f"    Max error at low freq: {err1:.4f}")
    print()

    # 1b) x(t) = t*e^{-2t}u(t)
    # X(omega) = 1/(2+jw)^2  (from freq differentiation property)
    x2 = t * np.exp(-2 * t) * (t >= 0)
    X2_num = fft(x2) * dt
    X2_analytical = 1 / (2 + 1j * omega) ** 2

    idx = np.where((freqs > 0) & (freqs < 5))[0][::100]
    err2 = np.max(np.abs(X2_num[idx] - X2_analytical[idx]))
    print(f"1b) t*e^{{-2t}}*u(t)")
    print(f"    X(w) = 1/(2+jw)^2")
    print(f"    Max error at low freq: {err2:.4f}")
    print()

    # 1c) x(t) = e^{-|t|}*cos(10t)
    # e^{-|t|} <-> 2/(1+w^2), modulation shifts: X(w) = 1/(1+(w-10)^2) + 1/(1+(w+10)^2)
    x3 = np.exp(-np.abs(t)) * np.cos(10 * t)
    X3_num = fft(x3) * dt
    X3_analytical = 1 / (1 + (omega - 10) ** 2) + 1 / (1 + (omega + 10) ** 2)

    idx = np.where((freqs > 0) & (freqs < 20))[0][::100]
    err3 = np.max(np.abs(X3_num[idx] - X3_analytical[idx]))
    print(f"1c) e^{{-|t|}}*cos(10t)")
    print(f"    X(w) = 1/(1+(w-10)^2) + 1/(1+(w+10)^2)")
    print(f"    Max error at low freq: {err3:.4f}")
    print()

    # 1d) x(t) = rect(t)*cos(20*pi*t)
    # rect(t) <-> sinc(f), modulation shifts by +/-10 Hz
    x4 = np.where(np.abs(t) <= 0.5, 1.0, 0.0) * np.cos(20 * np.pi * t)
    X4_num = fft(x4) * dt
    # X(f) = 0.5*sinc(f-10) + 0.5*sinc(f+10)
    X4_analytical = 0.5 * np.sinc(freqs - 10) + 0.5 * np.sinc(freqs + 10)
    # Convert to angular frequency representation
    X4_analytical_omega = X4_analytical / (2 * np.pi) if False else X4_analytical

    print(f"1d) rect(t)*cos(20*pi*t)")
    print(f"    X(f) = 0.5*sinc(f-10) + 0.5*sinc(f+10)")
    print(f"    Peak at f=10 Hz: |X_num|={np.max(np.abs(X4_num)):.4f}")


# === Exercise 2: Property Application ===
# Problem: Use transform properties (no re-derivation from integral).

def exercise_2():
    """Fourier transform property applications."""
    print("Base pair: e^{-at}u(t) <-> 1/(a+jw)")
    print()

    print("2a) F{e^{-a(t-3)}u(t-3)} (time shift)")
    print("    = e^{-j3w} * 1/(a+jw)")
    print()

    print("2b) F{e^{-at}u(t) * e^{j5t}} (frequency shift)")
    print("    = 1/(a+j(w-5))")
    print()

    print("2c) F{e^{-2at}u(2t)} (scaling)")
    print("    u(2t) = u(t) for t>0, so x(t) = e^{-2at}u(t)")
    print("    Using scaling: x(t) = e^{-bt}u(t) with b=2a")
    print("    X(w) = 1/(2a+jw)")
    print()

    print("2d) F{t*e^{-at}u(t)} (frequency differentiation)")
    print("    = j * d/dw [1/(a+jw)] = j * (-j)/(a+jw)^2 = 1/(a+jw)^2")
    print()

    print("2e) F{d/dt[e^{-at}u(t)]} (time differentiation)")
    print("    d/dt[e^{-at}u(t)] = -a*e^{-at}u(t) + delta(t)")
    print("    F = -a/(a+jw) + 1 = jw/(a+jw)")

    # Numerical verification of 2a
    a = 2.0
    dt = 0.001
    t = np.arange(-2, 20, dt)
    freqs = fftfreq(len(t), dt)
    omega = 2 * np.pi * freqs

    x = np.exp(-a * (t - 3)) * (t >= 3)
    X_num = fft(x) * dt
    X_analytical = np.exp(-1j * omega * 3) / (a + 1j * omega)

    idx = np.where((np.abs(freqs) > 0.1) & (np.abs(freqs) < 5))[0][::200]
    err = np.max(np.abs(X_num[idx] - X_analytical[idx]))
    print(f"\nVerification (2a): max error = {err:.4f}")


# === Exercise 3: Convolution Theorem Applications ===
# Problem: Compute convolutions via frequency domain.

def exercise_3():
    """Convolution theorem applications."""
    dt = 0.001
    t = np.arange(-5, 30, dt)

    # 3a) e^{-t}u(t) * e^{-2t}u(t)
    # In frequency domain: 1/(1+jw) * 1/(2+jw)
    # Partial fractions: 1/(1+jw) - 1/(2+jw) -> (e^{-t} - e^{-2t})*u(t)
    x = np.exp(-t) * (t >= 0)
    h = np.exp(-2 * t) * (t >= 0)
    y_direct = np.convolve(x, h, mode='full')[:len(t)] * dt
    y_analytical = (np.exp(-t) - np.exp(-2 * t)) * (t >= 0)

    # FFT method
    N_fft = len(t)
    X = fft(x * dt, N_fft)
    H = fft(h * dt, N_fft)
    y_fft = np.real(ifft(X * H)) / dt

    err_direct = np.max(np.abs(y_direct[:10000] - y_analytical[:10000]))
    err_fft = np.max(np.abs(y_fft[:10000] - y_analytical[:10000]))
    print(f"3a) e^{{-t}}u(t) * e^{{-2t}}u(t) = (e^{{-t}} - e^{{-2t}})u(t)")
    print(f"    Direct convolution error: {err_direct:.4f}")
    print(f"    FFT convolution error:    {err_fft:.4f}")
    print()

    # 3b) First-order lowpass with e^{-t}u(t) input
    omega_c = 5.0  # rad/s
    # H(w) = 1/(1+jw/omega_c) = omega_c/(omega_c+jw)
    # X(w) = 1/(1+jw)
    # Y(w) = omega_c / ((omega_c+jw)(1+jw))
    # Partial fractions: A/(1+jw) + B/(omega_c+jw)
    # A = omega_c/(omega_c-1), B = -omega_c/(omega_c-1) (for omega_c != 1)
    A = omega_c / (omega_c - 1)
    B = -1 / (omega_c - 1)  # coefficient for omega_c*e^{-omega_c*t}
    y_analytical_b = (A * np.exp(-t) + B * omega_c * np.exp(-omega_c * t)) * (t >= 0)

    print(f"3b) Lowpass H(w) = 1/(1+jw/{omega_c}) with input e^{{-t}}u(t)")
    print(f"    y(t) = {A:.4f}*e^{{-t}} + {B * omega_c:.4f}*e^{{-{omega_c}t}} for t >= 0")
    print()

    # 3c) sinc(Bt) through ideal lowpass
    B = 5.0
    print(f"3c) x(t) = sinc({B}t) through ideal lowpass:")
    print(f"    X(f) is rect function with bandwidth pi*B = {np.pi * B:.2f} rad/s")
    print(f"    (a) If omega_c > pi*B: output = x(t) (signal passes unchanged)")
    print(f"    (b) If omega_c < pi*B: output = sinc(omega_c*t/pi) * omega_c/pi")
    print(f"        (bandwidth is clipped to omega_c)")


# === Exercise 4: Parseval's Theorem ===
# Problem: Compute energies using Parseval's theorem.

def exercise_4():
    """Parseval's theorem energy computations."""
    # 4a) x(t) = 1/(1+t^2)
    # X(omega) = pi * e^{-|omega|}
    # E = (1/2pi) * integral |X(w)|^2 dw = (1/2pi) * pi^2 * integral e^{-2|w|} dw
    # = (pi/2) * (2/(2)) = pi/2
    dt = 0.001
    t = np.arange(-50, 50, dt)
    x = 1 / (1 + t ** 2)
    E_time = np.trapz(x ** 2, t)
    E_analytical = np.pi / 2

    print(f"4a) x(t) = 1/(1+t^2)")
    print(f"    E_time = {E_time:.6f}")
    print(f"    E_analytical = pi/2 = {E_analytical:.6f}")
    print()

    # 4b) Gaussian pulse energy fraction within |w| <= 1/sigma
    sigma = 1.0
    # x(t) = e^{-t^2/(2*sigma^2)}, X(w) = sigma*sqrt(2*pi)*e^{-w^2*sigma^2/2}
    # Total energy = sigma*sqrt(pi)
    # Energy in |w| <= 1/sigma:
    # E_band = (1/2pi) * integral_{-1/sigma}^{1/sigma} |X(w)|^2 dw
    # = (sigma^2/pi) * integral_0^{1/sigma} e^{-w^2*sigma^2} dw

    from scipy.special import erf
    fraction = erf(1 / np.sqrt(2))
    print(f"4b) Gaussian pulse, fraction within |w| <= 1/sigma:")
    print(f"    Fraction = erf(1/sqrt(2)) = {fraction:.6f} = {fraction * 100:.2f}%")
    print()

    # 4c) Rectangular pulse 99% energy bandwidth
    T = 1.0
    # E_total = T
    # X(f) = T*sinc(fT)
    # Need: integral_{-B}^{B} |X(f)|^2 df >= 0.99 * T^2
    # Numerical search for B
    E_total = T
    from scipy.integrate import quad
    for B in np.arange(0.5, 50, 0.1):
        E_band, _ = quad(lambda f: (T * np.sinc(f * T)) ** 2, -B, B)
        frac = E_band / E_total
        if frac >= 0.99:
            print(f"4c) Rectangular pulse (T={T}): 99% energy bandwidth = {B:.1f} Hz")
            print(f"    BW*T product = {B * T:.1f}")
            break


# === Exercise 5: Filter Design ===
# Problem: Extract 1kHz tone from noisy signal.

def exercise_5():
    """Frequency-domain filter design and application."""
    fs = 8000
    duration = 1.0
    t = np.arange(0, duration, 1 / fs)
    N = len(t)

    # Signal: 1 kHz tone + 60 Hz hum + broadband noise
    np.random.seed(42)
    x_signal = np.sin(2 * np.pi * 1000 * t)
    x_hum = 0.5 * np.sin(2 * np.pi * 60 * t)
    x_noise = 0.3 * np.random.randn(N)
    x = x_signal + x_hum + x_noise

    # FFT
    X = fft(x)
    freqs = fftfreq(N, 1 / fs)

    # Bandpass filter around 1 kHz (+/- 100 Hz)
    H = np.zeros(N)
    f_center = 1000
    bw = 100
    mask = (np.abs(np.abs(freqs) - f_center) <= bw)
    # Smooth transition using Gaussian
    H = np.exp(-0.5 * ((np.abs(freqs) - f_center) / (bw / 3)) ** 2)

    Y = X * H
    y = np.real(ifft(Y))

    snr_before = 10 * np.log10(np.mean(x_signal ** 2) / np.mean((x_hum + x_noise) ** 2))
    residual = y - x_signal
    snr_after = 10 * np.log10(np.mean(x_signal ** 2) / np.mean(residual ** 2))
    print(f"SNR before filtering: {snr_before:.2f} dB")
    print(f"SNR after filtering:  {snr_after:.2f} dB")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(t[:500] * 1000, x[:500])
    axes[0, 0].set_title('Input Signal (time domain)')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(freqs[:N // 2], 20 * np.log10(np.abs(X[:N // 2]) + 1e-10))
    axes[0, 1].set_title('Input Spectrum')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].set_xlim([0, 2000])
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t[:500] * 1000, y[:500])
    axes[1, 0].set_title('Filtered Signal (time domain)')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(freqs[:N // 2], 20 * np.log10(np.abs(Y[:N // 2]) + 1e-10))
    axes[1, 1].set_title('Filtered Spectrum')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude (dB)')
    axes[1, 1].set_xlim([0, 2000])
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex04_filter_design.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex04_filter_design.png")


# === Exercise 6: Time-Bandwidth Product ===
# Problem: Compute TBP for various pulse shapes.

def exercise_6():
    """Time-bandwidth product computation."""
    dt = 0.0001
    t = np.arange(-50, 50, dt)
    N = len(t)
    freqs = fftfreq(N, dt)

    pulses = {}

    # 1. Rectangular pulse rect(t/T), T=1
    T = 1.0
    x_rect = np.where(np.abs(t) <= T / 2, 1.0, 0.0)
    pulses['Rectangular'] = x_rect

    # 2. Gaussian pulse e^{-pi*t^2}
    x_gauss = np.exp(-np.pi * t ** 2)
    pulses['Gaussian'] = x_gauss

    # 3. Exponential e^{-t}u(t)
    x_exp = np.exp(-t) * (t >= 0)
    pulses['Exponential'] = x_exp

    # 4. Raised cosine
    x_rc = np.where(np.abs(t) <= T, 0.5 * (1 + np.cos(np.pi * t / T)), 0.0)
    pulses['Raised Cosine'] = x_rc

    print("Time-Bandwidth Products:")
    print(f"{'Pulse':<20} {'Delta_t':<12} {'Delta_f':<12} {'TBP':<12}")
    print("-" * 56)

    tbp_values = {}
    for name, x in pulses.items():
        # RMS time duration
        E = np.trapz(np.abs(x) ** 2, t)
        t_mean = np.trapz(t * np.abs(x) ** 2, t) / E
        delta_t = np.sqrt(np.trapz((t - t_mean) ** 2 * np.abs(x) ** 2, t) / E)

        # RMS bandwidth
        X = fft(x) * dt
        S = np.abs(X) ** 2
        E_f = np.sum(S) * (freqs[1] - freqs[0]) if len(freqs) > 1 else 0
        f_mean = np.sum(freqs * S) / np.sum(S) if np.sum(S) > 0 else 0
        delta_f = np.sqrt(np.sum((freqs - f_mean) ** 2 * S) / np.sum(S)) if np.sum(S) > 0 else 0

        tbp = delta_t * delta_f
        tbp_values[name] = tbp
        print(f"{name:<20} {delta_t:<12.4f} {delta_f:<12.4f} {tbp:<12.4f}")

    print()
    ranked = sorted(tbp_values.items(), key=lambda item: item[1])
    print("Ranking (best to worst TBP):")
    for i, (name, tbp) in enumerate(ranked, 1):
        print(f"  {i}. {name}: TBP = {tbp:.4f}")

    print()
    print("Radar: Gaussian pulse (minimum TBP = optimal resolution)")
    print("Communications: Raised cosine (controlled spectral rolloff, low ISI)")


# === Exercise 7: Duality ===
# Problem: Use duality to find transforms.

def exercise_7():
    """Duality property applications."""
    # If x(t) <-> X(w), then X(t) <-> 2*pi*x(-w)

    # 7a) Find F{1/(a^2+t^2)} from e^{-a|t|} <-> 2a/(a^2+w^2)
    # Duality: 2a/(a^2+t^2) <-> 2*pi*e^{-a|w|}
    # So: 1/(a^2+t^2) <-> (pi/a)*e^{-a|w|}
    a = 2.0
    dt = 0.001
    t = np.arange(-20, 20, dt)
    freqs = fftfreq(len(t), dt)
    omega = 2 * np.pi * freqs

    x = 1 / (a ** 2 + t ** 2)
    X_num = fft(x) * dt
    X_analytical = (np.pi / a) * np.exp(-a * np.abs(omega))

    idx = np.where(np.abs(freqs) < 5)[0][::100]
    err = np.max(np.abs(X_num[idx] - X_analytical[idx]))
    print(f"7a) F{{1/(a^2+t^2)}} = (pi/a)*e^{{-a|w|}}, a={a}")
    print(f"    Max error: {err:.4f}")
    print()

    # 7b) Find F{sinc(Wt)} from rect(t/tau) <-> tau*sinc(w*tau/(2*pi))
    # rect(t/tau) <-> tau*sinc(f*tau) (using f, not omega)
    # Duality: tau*sinc(tau*t) <-> rect(f/tau) (rect in frequency is 1 for |f|<tau/2)
    # So: sinc(Wt) <-> (1/W)*rect(f/W)
    W = 3.0
    x2 = np.sinc(W * t)
    X2_num = fft(x2) * dt
    X2_analytical = np.where(np.abs(freqs) <= W / 2, 1 / W, 0.0)

    print(f"7b) F{{sinc({W}t)}} = (1/{W})*rect(f/{W})")
    print(f"    At f=0: X_num={np.abs(X2_num[0]):.4f}, analytical={1 / W:.4f}")
    peak_val = np.max(np.abs(X2_num))
    print(f"    Peak: {peak_val:.4f}")


# === Exercise 8: Comprehensive Analysis ===
# Problem: Complete spectral analysis of a multi-component signal.

def exercise_8():
    """Comprehensive spectral analysis with windowing."""
    fs = 8000
    duration = 2.0
    t = np.arange(0, duration, 1 / fs)
    N = len(t)

    # Generate signal with multiple components
    np.random.seed(42)
    x = (1.0 * np.sin(2 * np.pi * 440 * t) +
         0.7 * np.sin(2 * np.pi * 880 * t) +
         0.3 * np.sin(2 * np.pi * 1320 * t) +
         0.1 * np.random.randn(N))

    # 1. Magnitude and phase spectra
    X = fft(x)
    freqs = fftfreq(N, 1 / fs)
    mag = np.abs(X[:N // 2]) / N
    phase = np.angle(X[:N // 2])

    # 2. Dominant frequencies
    mag_thresh = np.max(mag) * 0.05
    peaks = []
    for i in range(1, len(mag) - 1):
        if mag[i] > mag[i - 1] and mag[i] > mag[i + 1] and mag[i] > mag_thresh:
            peaks.append((freqs[i], mag[i]))
    peaks.sort(key=lambda p: p[1], reverse=True)
    print("Dominant frequency components:")
    for f, m in peaks[:5]:
        print(f"  {f:.1f} Hz, magnitude = {m:.4f}")

    # 3. Bandpass filter to isolate 440 Hz
    H = np.exp(-0.5 * ((np.abs(freqs) - 440) / 30) ** 2)
    Y = X * H
    y = np.real(ifft(Y))

    E_before = np.sum(x ** 2) / fs
    E_after = np.sum(y ** 2) / fs
    frac = E_after / E_before
    print(f"\nEnergy before: {E_before:.4f}")
    print(f"Energy in 440 Hz band: {E_after:.4f} ({frac * 100:.1f}%)")

    # 5. Hamming windowed spectrum
    window = np.hamming(N)
    X_windowed = fft(x * window)
    mag_windowed = np.abs(X_windowed[:N // 2]) / np.sum(window)

    print(f"\nPeak sidelobe (no window): {20 * np.log10(np.sort(mag)[::-1][3] / mag.max() + 1e-15):.1f} dB")
    print(f"Peak sidelobe (Hamming):   {20 * np.log10(np.sort(mag_windowed)[::-1][3] / mag_windowed.max() + 1e-15):.1f} dB")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(freqs[:N // 2], 20 * np.log10(mag + 1e-10))
    axes[0, 0].set_title('Magnitude Spectrum (no window)')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_xlim([0, 2000])
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(freqs[:N // 2], 20 * np.log10(mag_windowed + 1e-10))
    axes[0, 1].set_title('Magnitude Spectrum (Hamming window)')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_xlim([0, 2000])
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t[:1000] * 1000, x[:1000])
    axes[1, 0].set_title('Original Signal')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t[:1000] * 1000, y[:1000])
    axes[1, 1].set_title('Filtered (440 Hz isolated)')
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex04_comprehensive.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex04_comprehensive.png")


if __name__ == "__main__":
    print("=== Exercise 1: Transform Computation ===")
    exercise_1()
    print("\n=== Exercise 2: Property Application ===")
    exercise_2()
    print("\n=== Exercise 3: Convolution Theorem Applications ===")
    exercise_3()
    print("\n=== Exercise 4: Parseval's Theorem ===")
    exercise_4()
    print("\n=== Exercise 5: Filter Design ===")
    exercise_5()
    print("\n=== Exercise 6: Time-Bandwidth Product ===")
    exercise_6()
    print("\n=== Exercise 7: Duality ===")
    exercise_7()
    print("\n=== Exercise 8: Comprehensive Analysis ===")
    exercise_8()
    print("\nAll exercises completed!")
