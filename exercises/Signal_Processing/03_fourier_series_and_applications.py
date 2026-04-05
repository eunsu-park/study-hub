"""
Exercises for Lesson 03: Fourier Series and Applications
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import integrate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Fourier Coefficient Computation ===
# Problem: Compute Fourier series coefficients for various waveforms.

def exercise_1():
    """Fourier coefficient computation for standard waveforms."""
    T0 = 1.0
    omega0 = 2 * np.pi / T0
    N_harmonics = 20
    t = np.linspace(0, T0, 1000, endpoint=False)

    # 1. Full-wave rectified sine: |sin(omega0 * t)|
    # c_0 = 2/pi, c_{2k} = (-1)^{k+1} * 2 / (pi * (4k^2 - 1)), c_{odd} = 0
    # Period is T0/2 for |sin|, so fundamental = 2*omega0
    print("1a) x(t) = |sin(omega0 * t)|")
    print("   Period = T0/2, fundamental freq = 2*f0")
    cn_rectified = {}
    for n in range(-N_harmonics, N_harmonics + 1):
        if n % 2 != 0:
            cn_rectified[n] = 0.0
        else:
            k = n // 2
            if k == 0:
                cn_rectified[n] = 2 / np.pi
            else:
                cn_rectified[n] = 2 / (np.pi * (1 - 4 * k ** 2))

    print(f"   c_0 = {cn_rectified[0]:.6f} (analytical: {2 / np.pi:.6f})")
    print(f"   c_2 = {cn_rectified[2]:.6f}")
    print(f"   c_4 = {cn_rectified[4]:.6f}")
    print()

    # 2. cos^2(omega0 * t) = 0.5 + 0.5*cos(2*omega0*t)
    # c_0 = 0.5, c_{+/-1} using 2*omega0 base: c_0=0.5, c_{+/-2}=0.25
    print("1b) x(t) = cos^2(omega0 * t) = 0.5 + 0.5*cos(2*omega0*t)")
    print("   c_0 = 0.5, c_{+/-2} = 0.25, all others = 0")
    print()

    # 3. Pulse train: numerical computation
    tau = 0.3  # duty cycle = tau/T0 = 0.3
    print(f"1c) Pulse train with tau/T0 = {tau}")
    cn_pulse = {}
    for n in range(-N_harmonics, N_harmonics + 1):
        if n == 0:
            cn_pulse[n] = tau / T0
        else:
            cn_pulse[n] = (tau / T0) * np.sinc(n * tau / T0)
    print(f"   c_0 = {cn_pulse[0]:.6f} (= tau/T0 = {tau / T0:.6f})")
    print(f"   c_1 = {cn_pulse[1]:.6f}")
    print(f"   c_2 = {cn_pulse[2]:.6f}")


# === Exercise 2: Symmetry Exploitation ===
# Problem: Identify symmetry and determine which coefficients are zero.

def exercise_2():
    """Symmetry-based Fourier coefficient analysis."""
    print("2a) x(t) = t^2 on [-pi, pi], periodic")
    print("   Even function -> b_n = 0 (only cosine terms)")
    print("   No half-wave symmetry -> both even and odd harmonics present")
    print()

    print("2b) x(t) = t on [-pi, pi], periodic")
    print("   Odd function -> a_n = 0, a_0 = 0 (only sine terms)")
    print()

    print("2c) x(t) = |t| on [-pi, pi], periodic")
    print("   Even function -> b_n = 0 (only cosine terms)")
    print()

    print("2d) Staircase: +1 for 0<t<pi, -1 for -pi<t<0")
    print("   Odd function -> a_n = 0, a_0 = 0 (only sine terms)")
    print("   Half-wave symmetry -> only odd harmonics")

    # Numerical verification
    T0 = 2 * np.pi
    omega0 = 2 * np.pi / T0
    N_max = 10

    # Signal 2a: t^2
    print("\n--- Numerical verification for x(t) = t^2 ---")
    for n in range(6):
        if n == 0:
            a_n, _ = integrate.quad(lambda t: t ** 2, -np.pi, np.pi)
            a_n /= (2 * np.pi)
            print(f"   a_0 = {a_n:.6f}")
        else:
            a_n, _ = integrate.quad(lambda t: t ** 2 * np.cos(n * t), -np.pi, np.pi)
            a_n /= np.pi
            b_n, _ = integrate.quad(lambda t: t ** 2 * np.sin(n * t), -np.pi, np.pi)
            b_n /= np.pi
            print(f"   a_{n} = {a_n:.6f}, b_{n} = {b_n:.6f}")


# === Exercise 3: Gibbs Phenomenon Investigation ===
# Problem: Compute Fourier partial sums of square wave and analyze overshoot.

def exercise_3():
    """Gibbs phenomenon analysis."""
    T0 = 2 * np.pi
    t = np.linspace(-np.pi, np.pi, 10000)

    # Square wave: +1 for 0<t<pi, -1 for -pi<t<0
    x_true = np.sign(np.sin(t + 1e-10))

    N_values = [5, 21, 101]

    fig, axes = plt.subplots(len(N_values), 1, figsize=(12, 9))

    for idx, N in enumerate(N_values):
        # Fourier partial sum: b_n = 4/(n*pi) for odd n
        S_N = np.zeros_like(t)
        for n in range(1, N + 1):
            if n % 2 == 1:
                b_n = 4 / (n * np.pi)
                S_N += b_n * np.sin(n * t)

        # Peak overshoot
        max_val = np.max(S_N)
        overshoot_pct = (max_val - 1.0) * 100
        print(f"N = {N:3d}: peak value = {max_val:.6f}, overshoot = {overshoot_pct:.2f}%")

        axes[idx].plot(t, x_true, 'b--', alpha=0.3, linewidth=1)
        axes[idx].plot(t, S_N, 'r-', linewidth=1.5)
        axes[idx].set_title(f'N = {N}, Overshoot = {overshoot_pct:.2f}%')
        axes[idx].set_ylim([-1.4, 1.4])
        axes[idx].grid(True, alpha=0.3)

    print(f"\nGibbs overshoot converges to ~8.95% (Si(pi)/pi*2 - 1)")
    print(f"Analytical limit: {(integrate.quad(lambda t: np.sin(t) / t, 0, np.pi)[0] * 2 / np.pi - 1) * 100:.2f}%")

    # Lanczos sigma factors
    print("\n--- Lanczos sigma factors ---")
    S_lanczos = np.zeros_like(t)
    N_lanczos = 101
    for n in range(1, N_lanczos + 1):
        if n % 2 == 1:
            b_n = 4 / (n * np.pi)
            sigma = np.sinc(n / N_lanczos)  # Lanczos factor
            S_lanczos += sigma * b_n * np.sin(n * t)

    max_lanczos = np.max(S_lanczos)
    overshoot_lanczos = (max_lanczos - 1.0) * 100
    print(f"Lanczos (N=101): overshoot = {overshoot_lanczos:.2f}%")

    mse_standard = np.mean((x_true - S_N) ** 2)  # S_N from N=101
    mse_lanczos = np.mean((x_true - S_lanczos) ** 2)
    print(f"MSE standard: {mse_standard:.6f}")
    print(f"MSE Lanczos:  {mse_lanczos:.6f}")

    plt.tight_layout()
    plt.savefig('ex03_gibbs.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex03_gibbs.png")


# === Exercise 4: Parseval's Theorem Applications ===
# Problem: Use Parseval's theorem to compute infinite sums.

def exercise_4():
    """Parseval's theorem for computing infinite sums."""
    # 1. Sum 1/n^2 from sawtooth wave
    # Sawtooth: x(t) = t/pi on (-pi, pi), b_n = (-1)^{n+1} * 2/(n*pi) * pi = 2*(-1)^{n+1}/n
    # Actually, for x(t) = t on (-pi, pi): b_n = 2*(-1)^{n+1}/n
    # Power = (1/2pi) * int_{-pi}^{pi} t^2 dt = pi^2/3
    # Parseval: sum |b_n|^2 / 2 = sum (2/n)^2 / 2 = 2 * sum 1/n^2
    # So: pi^2/3 = 2 * sum 1/n^2 -> sum 1/n^2 = pi^2/6

    # More carefully: for x(t) = t on (-pi, pi) periodic:
    # Power P = (1/(2*pi)) * int_{-pi}^{pi} t^2 dt = pi^2/3
    # b_n = 2*(-1)^{n+1}/n, a_0 = 0
    # Parseval: P = sum_{n=1}^{inf} (b_n^2)/2 = sum 4/(2*n^2) = 2 * sum 1/n^2
    # pi^2/3 = 2 * sum 1/n^2 -> sum 1/n^2 = pi^2/6

    partial_sum = sum(1.0 / n ** 2 for n in range(1, 10001))
    print(f"4a) sum(1/n^2) = pi^2/6 = {np.pi ** 2 / 6:.10f}")
    print(f"    Partial sum (10000 terms): {partial_sum:.10f}")
    print()

    # 2. Sum 1/(2k+1)^4 from triangle wave
    # Triangle wave: a_n = -4/(pi^2 * n^2) for odd n, 0 for even n
    # Power = 1/3 (for triangle wave with amplitude 1, period 2*pi)
    # Parseval: 1/3 = sum_{k=0}^{inf} (4/(pi^2*(2k+1)^2))^2 / 2
    # = 8/pi^4 * sum 1/(2k+1)^4
    # -> sum 1/(2k+1)^4 = pi^4/96

    partial_sum_4 = sum(1.0 / (2 * k + 1) ** 4 for k in range(10000))
    print(f"4b) sum(1/(2k+1)^4) = pi^4/96 = {np.pi ** 4 / 96:.10f}")
    print(f"    Partial sum (10000 terms): {partial_sum_4:.10f}")
    print()

    # 3. Pulse train power vs duty cycle
    duty_cycles = [0.1, 0.25, 0.5]
    print("4c) Pulse train power vs duty cycle:")
    for d in duty_cycles:
        # Power = d (fraction of time signal is 1)
        P_analytical = d
        # Verify with Parseval: c_0 = d, c_n = d*sinc(n*d)
        P_parseval = d ** 2  # |c_0|^2
        for n in range(1, 1000):
            cn = d * np.sinc(n * d)
            P_parseval += 2 * cn ** 2  # factor 2 for c_n and c_{-n}
        print(f"   d={d}: P_analytical={P_analytical:.6f}, P_parseval={P_parseval:.6f}")


# === Exercise 5: Signal Reconstruction Challenge ===
# Problem: Reconstruct signal from given Fourier coefficients.

def exercise_5():
    """Signal reconstruction from amplitude and phase spectra."""
    # Given coefficients
    coeffs = {
        0: (0.5, 0),
        1: (0.8, -np.pi / 4),
        2: (0.3, -np.pi / 2),
        3: (0.6, np.pi / 3),
        5: (0.2, -np.pi / 6),
        7: (0.1, np.pi / 4)
    }

    T0 = 1.0
    omega0 = 2 * np.pi / T0
    t = np.linspace(0, 2 * T0, 2000)

    # 1. Reconstruct x(t)
    x = np.zeros_like(t)
    for n, (amp, phase) in coeffs.items():
        c_n = amp * np.exp(1j * phase)
        if n == 0:
            x += amp  # c_0 is real
        else:
            # c_n * e^{jn*omega0*t} + c_{-n} * e^{-jn*omega0*t}
            # = 2*|c_n|*cos(n*omega0*t + angle(c_n))
            x += 2 * amp * np.cos(n * omega0 * t + phase)

    # 2. Average power using Parseval
    P = coeffs[0][0] ** 2  # |c_0|^2
    for n, (amp, phase) in coeffs.items():
        if n > 0:
            P += 2 * amp ** 2  # |c_n|^2 + |c_{-n}|^2
    print(f"Average power (Parseval): {P:.6f}")

    # 3. Zero phase version
    x_zero_phase = np.zeros_like(t)
    for n, (amp, _) in coeffs.items():
        if n == 0:
            x_zero_phase += amp
        else:
            x_zero_phase += 2 * amp * np.cos(n * omega0 * t)

    # 4. Random phase version
    np.random.seed(42)
    x_random_phase = np.zeros_like(t)
    for n, (amp, _) in coeffs.items():
        rand_phase = np.random.uniform(-np.pi, np.pi)
        if n == 0:
            x_random_phase += amp
        else:
            x_random_phase += 2 * amp * np.cos(n * omega0 * t + rand_phase)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].plot(t, x, 'b-', linewidth=1.5)
    axes[0].set_title('Reconstructed Signal (original phases)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, x_zero_phase, 'r-', linewidth=1.5)
    axes[1].set_title('Zero Phase (all phases = 0)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, x_random_phase, 'g-', linewidth=1.5)
    axes[2].set_title('Random Phase')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex03_reconstruction.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex03_reconstruction.png")
    print("Zero phase: signal becomes symmetric, peaky")
    print("Random phase: signal shape changes completely, power stays the same")


# === Exercise 6: Fourier Series of a Real-World Signal ===
# Problem: Sawtooth wave analysis.

def exercise_6():
    """Sawtooth wave Fourier series analysis."""
    from scipy.signal import sawtooth as make_sawtooth

    f0 = 100
    fs = 44100
    duration = 0.1
    t = np.arange(0, duration, 1 / fs)
    x = make_sawtooth(2 * np.pi * f0 * t)

    T0 = 1 / f0
    N_period = int(T0 * fs)

    # Analytical Fourier coefficients for sawtooth
    # b_n = 2*(-1)^{n+1} / (n*pi)  (for normalized sawtooth)
    # More precisely for scipy sawtooth: b_n = -2/(n*pi)
    N_harmonics_list = [5, 10, 20]

    print(f"Sawtooth wave: f0={f0} Hz, fs={fs} Hz")

    # Numerical computation of Fourier coefficients
    one_period = x[:N_period]
    t_period = t[:N_period]
    cn_numerical = {}
    for n in range(-25, 26):
        integrand = one_period * np.exp(-1j * 2 * np.pi * n * f0 * t_period)
        cn_numerical[n] = np.trapz(integrand, t_period) * f0

    # Compare numerical vs analytical
    print("\nCoefficient comparison (|c_n|):")
    for n in [1, 2, 3, 4, 5]:
        analytical = 1 / (n * np.pi)  # magnitude
        numerical = np.abs(cn_numerical[n])
        print(f"  n={n}: analytical={analytical:.6f}, numerical={numerical:.6f}")

    # Reconstruction with different numbers of harmonics
    print("\nReconstruction SNR:")
    for N_h in N_harmonics_list:
        x_recon = np.zeros_like(t)
        for n in range(-N_h, N_h + 1):
            if n in cn_numerical:
                x_recon += np.real(cn_numerical[n] * np.exp(1j * 2 * np.pi * n * f0 * t))
            elif n != 0:
                # Use analytical
                pass

        # Using analytical coefficients for cleaner reconstruction
        x_recon_a = np.zeros_like(t)
        for n in range(1, N_h + 1):
            b_n = -2 / (n * np.pi)
            x_recon_a += b_n * np.sin(2 * np.pi * n * f0 * t)

        noise = x - x_recon_a
        signal_power = np.mean(x ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        print(f"  N_harmonics={N_h:2d}: SNR = {snr:.2f} dB")


# === Exercise 7: Fourier Series and LTI Systems ===
# Problem: Square wave through RC lowpass filter.

def exercise_7():
    """Square wave through RC lowpass filter."""
    f0 = 100  # Hz
    omega0 = 2 * np.pi * f0
    fc = 500  # cutoff frequency
    omega_c = 2 * np.pi * fc

    # Square wave coefficients: c_n = 2/(j*n*pi) for odd n, 0 for even n
    # RC lowpass: H(jw) = 1/(1 + jw/omega_c)
    N_harmonics = 50
    t = np.linspace(0, 3 / f0, 3000)

    # Input reconstruction
    x = np.zeros_like(t)
    y = np.zeros_like(t)

    print("Square wave (f0=100 Hz) through RC lowpass (fc=500 Hz)")
    print("\nOutput Fourier coefficients d_n = c_n * H(jn*omega0):")

    for n in range(-N_harmonics, N_harmonics + 1):
        if n == 0:
            c_n = 0.5  # DC component
            H_n = 1.0
        elif n % 2 != 0:
            c_n = 2 / (1j * n * np.pi)  # but actually for square wave 0 to 1
            # Standard: c_n = 1/(j*n*pi) * (1 - (-1)^n)
            c_n = (1 - (-1) ** n) / (2j * n * np.pi)
            H_n = 1 / (1 + 1j * n * omega0 / omega_c)
        else:
            continue

        d_n = c_n * H_n
        x += np.real(c_n * np.exp(1j * n * omega0 * t))
        y += np.real(d_n * np.exp(1j * n * omega0 * t))

        if 0 < abs(n) <= 5:
            print(f"  n={n:+3d}: |c_n|={np.abs(c_n):.4f}, |H|={np.abs(H_n):.4f}, "
                  f"|d_n|={np.abs(d_n):.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(t * 1000, x, 'b-', linewidth=1.5)
    axes[0].set_title('Input: Square Wave')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t * 1000, y, 'r-', linewidth=1.5)
    axes[1].set_title('Output: After RC Lowpass Filter (fc=500 Hz)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex03_lti_filtering.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("\nThe filter smooths the sharp transitions (reduces Gibbs phenomenon)")
    print("Plot saved to ex03_lti_filtering.png")


# === Exercise 8: Discrete-Time Fourier Series ===
# Problem: DT Fourier series of a periodic square wave.

def exercise_8():
    """Discrete-time Fourier series with exact reconstruction."""
    N = 32  # period

    # Discrete square wave: 1 for n=0..15, -1 for n=16..31
    x = np.ones(N)
    x[N // 2:] = -1

    # DT Fourier series coefficients (= DFT / N)
    X = np.fft.fft(x)
    c = X / N

    print(f"DT periodic square wave, period N = {N}")
    print(f"Number of nonzero coefficients: {np.sum(np.abs(c) > 1e-10)}")
    print(f"First 8 coefficients |c_k|: {np.abs(c[:8])}")

    # Reconstruction
    x_recon = np.zeros(N, dtype=complex)
    for k in range(N):
        x_recon += c[k] * np.exp(1j * 2 * np.pi * k * np.arange(N) / N)

    recon_error = np.max(np.abs(x - np.real(x_recon)))
    print(f"Reconstruction error: {recon_error:.2e}")
    print("Reconstruction is EXACT (no Gibbs phenomenon in DT)")

    # Compare with np.fft.fft
    X_fft = np.fft.fft(x)
    x_ifft = np.real(np.fft.ifft(X_fft))
    fft_error = np.max(np.abs(x - x_ifft))
    print(f"FFT/IFFT error: {fft_error:.2e}")
    print()
    print("DT Fourier series has NO Gibbs phenomenon because:")
    print("  - The representation uses exactly N coefficients for N samples")
    print("  - There is no truncation of harmonics")
    print("  - The DFT is an exact, invertible transform")


if __name__ == "__main__":
    print("=== Exercise 1: Fourier Coefficient Computation ===")
    exercise_1()
    print("\n=== Exercise 2: Symmetry Exploitation ===")
    exercise_2()
    print("\n=== Exercise 3: Gibbs Phenomenon Investigation ===")
    exercise_3()
    print("\n=== Exercise 4: Parseval's Theorem Applications ===")
    exercise_4()
    print("\n=== Exercise 5: Signal Reconstruction Challenge ===")
    exercise_5()
    print("\n=== Exercise 6: Fourier Series of a Real-World Signal ===")
    exercise_6()
    print("\n=== Exercise 7: Fourier Series and LTI Systems ===")
    exercise_7()
    print("\n=== Exercise 8: Discrete-Time Fourier Series ===")
    exercise_8()
    print("\nAll exercises completed!")
