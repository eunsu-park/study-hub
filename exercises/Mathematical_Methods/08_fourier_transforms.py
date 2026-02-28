"""
Exercise Solutions: Lesson 08 - Fourier Transforms
Mathematical Methods for Physical Sciences

Covers: basic transforms, shifting/scaling properties, convolution,
        FFT practice, physics applications, Parseval's theorem
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import integrate as sci_integrate


def exercise_1_basic_transforms():
    """
    Problem 1: Find the Fourier transform of:
    (a) f(x) = e^{-3|x|}
    (b) f(x) = x*e^{-x^2}
    (c) f(x) = 1/(1+x^2)
    """
    print("=" * 60)
    print("Problem 1: Basic Fourier Transforms")
    print("=" * 60)

    # Convention: F(k) = integral f(x) e^{-ikx} dx

    # (a) f = e^{-3|x|}
    # F(k) = int_{-inf}^{inf} e^{-3|x|} e^{-ikx} dx
    #       = int_0^inf e^{-3x} e^{-ikx} dx + int_{-inf}^0 e^{3x} e^{-ikx} dx
    #       = 1/(3+ik) + 1/(3-ik) = 6/(9+k^2)
    print(f"\n(a) f(x) = exp(-3|x|)")
    print(f"  F(k) = 6/(9 + k^2)")

    k = np.linspace(-10, 10, 500)
    F_a = 6 / (9 + k**2)

    # Numerical verification
    k_test = 2.0
    F_num, _ = sci_integrate.quad(
        lambda x: np.exp(-3*abs(x)) * np.exp(-1j*k_test*x).real,
        -np.inf, np.inf)
    print(f"  Numerical at k={k_test}: {F_num:.8f}")
    print(f"  Analytical at k={k_test}: {6/(9+k_test**2):.8f}")

    # (b) f = x*e^{-x^2}
    # This is an odd function => F(k) is purely imaginary
    # F(k) = -ik * sqrt(pi)/2 * e^{-k^2/4}   (using transform of Gaussian derivative)
    print(f"\n(b) f(x) = x*exp(-x^2)")
    print(f"  f is odd => F(k) is purely imaginary")
    print(f"  F(k) = -i*(k/2)*sqrt(pi)*exp(-k^2/4)")

    F_b_imag = -(k/2) * np.sqrt(np.pi) * np.exp(-k**2/4)

    # Numerical verification
    F_num_b, _ = sci_integrate.quad(
        lambda x: x * np.exp(-x**2) * np.sin(-k_test*x),  # imaginary part
        -np.inf, np.inf)
    print(f"  Im[F({k_test})] numerical:   {F_num_b:.8f}")
    print(f"  Im[F({k_test})] analytical: {-(k_test/2)*np.sqrt(np.pi)*np.exp(-k_test**2/4):.8f}")

    # (c) f = 1/(1+x^2)
    # F(k) = pi * e^{-|k|}
    print(f"\n(c) f(x) = 1/(1+x^2)")
    print(f"  F(k) = pi * exp(-|k|)")

    F_c = np.pi * np.exp(-np.abs(k))

    F_num_c, _ = sci_integrate.quad(
        lambda x: 1/(1+x**2) * np.cos(k_test*x),  # real part (f is even)
        -np.inf, np.inf)
    print(f"  F({k_test}) numerical:   {F_num_c:.8f}")
    print(f"  F({k_test}) analytical: {np.pi*np.exp(-abs(k_test)):.8f}")


def exercise_2_properties():
    """
    Problem 2: Use shifting, scaling, and differentiation properties.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Fourier Transform Properties")
    print("=" * 60)

    # Given: F{e^{-|x|}} = 2/(1+k^2)
    print(f"\nGiven: F{{exp(-|x|)}} = 2/(1+k^2)")

    # (a) Shifting: F{e^{-|x-3|}} = e^{-3ik} * 2/(1+k^2)
    print(f"\n(a) Shifting property: F{{f(x-a)}} = e^{{-iak}} F(k)")
    print(f"  F{{exp(-|x-3|)}} = e^{{-3ik}} * 2/(1+k^2)")
    print(f"  |F| is unchanged; phase shifts by -3k")

    # (b) Scaling: F{e^{-2|x|}} using F{f(ax)} = (1/|a|) F(k/a)
    print(f"\n(b) Scaling property: F{{f(ax)}} = (1/|a|) F(k/a)")
    print(f"  F{{exp(-2|x|)}} = (1/2) * 2/(1+(k/2)^2) = 4/(4+k^2)")
    print(f"  Compare with direct: 2*2/(4+k^2) = 4/(4+k^2) [check]")

    # (c) Differentiation: F{f'(x)} = ik * F(k)
    # f(x) = e^{-|x|}: f'(x) = -sgn(x)*e^{-|x|}
    print(f"\n(c) Differentiation property: F{{f'(x)}} = ik * F(k)")
    print(f"  f(x) = e^{{-|x|}}, f'(x) = -sgn(x)*e^{{-|x|}}")
    print(f"  F{{f'}} = ik * 2/(1+k^2) = 2ik/(1+k^2)")

    # Numerical verification
    k_test = 1.5
    x = np.linspace(-20, 20, 100000)
    dx = x[1] - x[0]

    # Direct transform of derivative
    f_prime = -np.sign(x) * np.exp(-np.abs(x))
    F_prime_num = np.trapz(f_prime * np.exp(-1j*k_test*x), x)

    # Using property
    F_prop = 1j * k_test * 2 / (1 + k_test**2)

    print(f"\n  Verification at k={k_test}:")
    print(f"    Direct: {F_prime_num:.6f}")
    print(f"    Property: {F_prop:.6f}")


def exercise_3_convolution():
    """
    Problem 3: Compute the convolution of rect(x) * rect(x) = triangle(x).
    """
    print("\n" + "=" * 60)
    print("Problem 3: Convolution (rect * rect)")
    print("=" * 60)

    print(f"\nrect(x) = 1 for |x| < 1/2, 0 otherwise")
    print(f"\nConvolution: (f*g)(x) = integral f(t)*g(x-t) dt")
    print(f"\n  rect * rect = triangle function:")
    print(f"  tri(x) = 1 - |x| for |x| < 1, 0 otherwise")

    # Convolution theorem: F{f*g} = F{f} * F{g}
    # F{rect(x)} = sinc(k/(2*pi)) * 1 = sin(k/2)/(k/2)
    print(f"\nConvolution theorem verification:")
    print(f"  F{{rect}} = sin(k/2)/(k/2)")
    print(f"  F{{rect*rect}} = [sin(k/2)/(k/2)]^2 = sinc^2(k/2)")
    print(f"  F{{tri}} = sinc^2(k/2)  [same -> confirmed]")

    # Numerical demonstration
    x = np.linspace(-2, 2, 1000)
    dx = x[1] - x[0]

    rect = np.where(np.abs(x) < 0.5, 1.0, 0.0)
    conv = np.convolve(rect, rect, mode='same') * dx
    tri_exact = np.maximum(1 - np.abs(x), 0)

    print(f"\n  Max difference |conv - tri|: {np.max(np.abs(conv - tri_exact)):.6f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(x, rect, 'b-', linewidth=2)
    axes[0].set_title('rect(x)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, conv, 'r-', linewidth=2, label='Numerical convolution')
    axes[1].plot(x, tri_exact, 'g--', linewidth=2, label='tri(x) exact')
    axes[1].set_title('rect * rect = tri')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    k = np.linspace(-15, 15, 500)
    sinc_sq = (np.sinc(k / (2*np.pi)))**2  # np.sinc(x) = sin(pi*x)/(pi*x)
    # Actually F{rect} with our convention: sin(k/2)/(k/2)
    F_rect = np.where(np.abs(k) > 1e-10, np.sin(k/2)/(k/2), 1.0)
    F_conv = F_rect**2

    axes[2].plot(k, F_conv, 'm-', linewidth=2)
    axes[2].set_title('F{tri} = sinc$^2$(k/2)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex08_convolution.png', dpi=150)
    plt.close()
    print("Plot saved to ex08_convolution.png")


def exercise_4_fft_practice():
    """
    Problem 4: FFT practice - analyze a signal with known frequencies.
    """
    print("\n" + "=" * 60)
    print("Problem 4: FFT Practice")
    print("=" * 60)

    # Generate signal: f(t) = 2*sin(5*2*pi*t) + 0.5*cos(12*2*pi*t) + noise
    fs = 100  # sampling rate
    T = 2.0   # duration
    N = int(fs * T)
    t = np.linspace(0, T, N, endpoint=False)

    np.random.seed(42)
    signal = 2 * np.sin(5 * 2 * np.pi * t) + \
             0.5 * np.cos(12 * 2 * np.pi * t) + \
             0.3 * np.random.randn(N)

    print(f"\nSignal: 2*sin(10*pi*t) + 0.5*cos(24*pi*t) + noise")
    print(f"Sampling rate: {fs} Hz, Duration: {T} s, N = {N}")

    # Compute FFT
    freqs = np.fft.fftfreq(N, d=1/fs)
    fft_vals = np.fft.fft(signal) / N

    # Only positive frequencies
    mask = freqs > 0
    freqs_pos = freqs[mask]
    amplitudes = 2 * np.abs(fft_vals[mask])  # factor 2 for one-sided

    # Find peaks
    peak_indices = np.argsort(amplitudes)[-5:][::-1]
    print(f"\nTop frequency components:")
    for idx in peak_indices:
        print(f"  f = {freqs_pos[idx]:.1f} Hz, amplitude = {amplitudes[idx]:.4f}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(t, signal, 'b-', linewidth=0.8)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Domain Signal')
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(freqs_pos[:50], amplitudes[:50], linefmt='r-', markerfmt='ro',
                 basefmt='k-', use_line_collection=True)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('FFT Spectrum')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex08_fft.png', dpi=150)
    plt.close()
    print("Plot saved to ex08_fft.png")


def exercise_5_physics_applications():
    """
    Problem 5: Physics applications -
    (a) Fraunhofer diffraction (single slit)
    (b) Heisenberg uncertainty with Gaussian wave packet
    """
    print("\n" + "=" * 60)
    print("Problem 5: Physics Applications")
    print("=" * 60)

    # (a) Single slit diffraction
    # Aperture function: rect(x/a) => F(k) ~ sinc(ka/2)
    # Intensity: I(theta) ~ sinc^2(pi*a*sin(theta)/lambda)
    print(f"\n(a) Fraunhofer diffraction (single slit width a):")
    print(f"  Aperture: rect(x/a)")
    print(f"  Far-field: F(k) = a * sin(ka/2)/(ka/2)")
    print(f"  Intensity: I(theta) ~ sinc^2(pi*a*sin(theta)/lambda)")

    a = 5.0  # slit width in units of lambda
    theta = np.linspace(-0.5, 0.5, 1000)  # small angle
    beta = np.pi * a * np.sin(theta)
    I = np.where(np.abs(beta) > 1e-10, (np.sin(beta) / beta)**2, 1.0)

    # First minimum at sin(theta) = lambda/a
    theta_min1 = np.arcsin(1/a)
    print(f"  For a = {a}*lambda:")
    print(f"  First minimum at theta = arcsin(lambda/a) = {np.degrees(theta_min1):.2f} deg")
    print(f"  Angular width of central max ~ 2*lambda/a = {2*np.degrees(theta_min1):.2f} deg")

    # (b) Gaussian wave packet uncertainty
    print(f"\n(b) Gaussian wave packet uncertainty:")
    print(f"  psi(x) = (2*pi*sigma^2)^(-1/4) * exp(-x^2/(4*sigma^2))")
    print(f"  phi(k) = (2*sigma^2/pi)^(1/4) * exp(-sigma^2*k^2)")
    print(f"\n  Delta_x = sigma (position uncertainty)")
    print(f"  Delta_k = 1/(2*sigma) (momentum uncertainty)")
    print(f"  Delta_x * Delta_k = 1/2 (minimum uncertainty)")

    # Numerical demonstration
    sigmas = [0.5, 1.0, 2.0]

    fig, axes = plt.subplots(len(sigmas), 2, figsize=(14, 4*len(sigmas)))

    x = np.linspace(-8, 8, 1000)
    k = np.linspace(-8, 8, 1000)

    for i, sigma in enumerate(sigmas):
        psi = (2*np.pi*sigma**2)**(-0.25) * np.exp(-x**2/(4*sigma**2))
        phi = (2*sigma**2/np.pi)**0.25 * np.exp(-sigma**2*k**2)

        dx_rms = sigma
        dk_rms = 1/(2*sigma)

        axes[i, 0].plot(x, np.abs(psi)**2, 'b-', linewidth=2)
        axes[i, 0].axvline(-dx_rms, color='r', linestyle='--', alpha=0.5)
        axes[i, 0].axvline(dx_rms, color='r', linestyle='--', alpha=0.5)
        axes[i, 0].set_title(f'|psi(x)|^2 (sigma={sigma})')
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(k, np.abs(phi)**2, 'r-', linewidth=2)
        axes[i, 1].axvline(-dk_rms, color='b', linestyle='--', alpha=0.5)
        axes[i, 1].axvline(dk_rms, color='b', linestyle='--', alpha=0.5)
        axes[i, 1].set_title(f'|phi(k)|^2 (Dk={dk_rms:.2f})')
        axes[i, 1].grid(True, alpha=0.3)

        print(f"  sigma={sigma}: Dx={dx_rms:.3f}, Dk={dk_rms:.3f}, Dx*Dk={dx_rms*dk_rms:.3f}")

    plt.tight_layout()
    plt.savefig('ex08_physics.png', dpi=150)
    plt.close()
    print("Plot saved to ex08_physics.png")


def exercise_6_parseval():
    """
    Problem 6: Verify Parseval's theorem numerically.
    integral |f(x)|^2 dx = (1/2pi) integral |F(k)|^2 dk
    """
    print("\n" + "=" * 60)
    print("Problem 6: Parseval's Theorem Verification")
    print("=" * 60)

    print(f"\nParseval's theorem: integral |f(x)|^2 dx = (1/(2*pi)) integral |F(k)|^2 dk")

    # Test with f(x) = e^{-x^2/2}
    # F(k) = sqrt(2*pi) * e^{-k^2/2}
    print(f"\nTest function: f(x) = exp(-x^2/2)")
    print(f"Transform: F(k) = sqrt(2*pi) * exp(-k^2/2)")

    x = np.linspace(-10, 10, 100000)
    dx = x[1] - x[0]

    f = np.exp(-x**2 / 2)
    lhs = np.trapz(np.abs(f)**2, x)

    k = np.linspace(-10, 10, 100000)
    dk = k[1] - k[0]
    F_k = np.sqrt(2 * np.pi) * np.exp(-k**2 / 2)
    rhs = np.trapz(np.abs(F_k)**2, k) / (2 * np.pi)

    print(f"\n  LHS: int |f|^2 dx = {lhs:.10f}")
    print(f"  RHS: (1/2pi) int |F|^2 dk = {rhs:.10f}")
    print(f"  Exact: sqrt(pi) = {np.sqrt(np.pi):.10f}")
    print(f"  Relative error: {abs(lhs - rhs)/lhs:.2e}")

    # Second test: f(x) = e^{-|x|}
    # F(k) = 2/(1+k^2)
    print(f"\nTest function 2: f(x) = exp(-|x|)")
    f2 = np.exp(-np.abs(x))
    lhs2 = np.trapz(np.abs(f2)**2, x)

    F2_k = 2 / (1 + k**2)
    rhs2 = np.trapz(np.abs(F2_k)**2, k) / (2 * np.pi)

    # Exact: int e^{-2|x|} dx = 1, (1/2pi) int 4/(1+k^2)^2 dk = 1
    print(f"  LHS: int |f|^2 dx = {lhs2:.10f}")
    print(f"  RHS: (1/2pi) int |F|^2 dk = {rhs2:.10f}")
    print(f"  Exact: 1.0")
    print(f"  Relative error: {abs(lhs2 - rhs2)/lhs2:.2e}")


if __name__ == "__main__":
    exercise_1_basic_transforms()
    exercise_2_properties()
    exercise_3_convolution()
    exercise_4_fft_practice()
    exercise_5_physics_applications()
    exercise_6_parseval()
