"""
Exercises for Lesson 01: Signals and Systems
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from fractions import Fraction

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Signal Classification ===
# Problem: Classify each signal as (a) CT/DT, (b) periodic/aperiodic,
# (c) energy/power, (d) causal/noncausal.

def exercise_1():
    """Classify signals by their properties."""
    print("Signal 1: x(t) = 3*cos(100*pi*t + pi/3)")
    print("  (a) Continuous-time")
    print("  (b) Periodic with T0 = 2*pi/(100*pi) = 0.02 s, f0 = 50 Hz")
    print("  (c) Power signal: P = A^2/2 = 9/2 = 4.5")
    print("  (d) Noncausal (defined for all t)")
    print()

    print("Signal 2: x[n] = (-0.5)^n * u[n]")
    print("  (a) Discrete-time")
    print("  (b) Aperiodic (decaying, not repeating)")
    E2 = 1 / (1 - 0.25)  # sum of |(-0.5)|^{2n} = sum of 0.25^n
    print(f"  (c) Energy signal: E = sum(0.25^n) = {E2:.4f}")
    print("  (d) Causal (zero for n < 0)")
    print()

    print("Signal 3: x(t) = e^{-2|t|}")
    print("  (a) Continuous-time")
    print("  (b) Aperiodic")
    E3 = 2 * (1 / (2 * 2))  # 2 * integral_0^inf e^{-4t} dt = 2/(4) = 0.5
    print(f"  (c) Energy signal: E = integral e^{{-4|t|}} dt = {E3:.4f}")
    print("  (d) Noncausal (nonzero for t < 0)")
    print()

    print("Signal 4: x[n] = cos(0.3*pi*n)")
    ratio = Fraction(3, 10)  # omega0/(2*pi) = 0.3*pi/(2*pi) = 0.15 = 3/20
    N = ratio.denominator
    print(f"  (a) Discrete-time")
    print(f"  (b) Periodic: omega0/(2*pi) = 0.15 = {Fraction(15, 100)} -> N = {Fraction(3,20).denominator}")
    # 0.3*pi / (2*pi) = 0.15 = 3/20, so N = 20
    print(f"      Fundamental period N = 20")
    print("  (c) Power signal: P = A^2/2 = 0.5")
    print("  (d) Noncausal (defined for all n)")
    print()

    print("Signal 5: x(t) = u(t) - u(t-5)")
    print("  (a) Continuous-time")
    print("  (b) Aperiodic (rectangular pulse)")
    print("  (c) Energy signal: E = integral_0^5 1 dt = 5.0")
    print("  (d) Causal (zero for t < 0)")
    print()

    print("Signal 6: x[n] = 2^n * u[-n]")
    print("  (a) Discrete-time")
    print("  (b) Aperiodic")
    E6 = 1 / (1 - 4)  # sum_{n=-inf}^{0} 4^n = sum_{k=0}^{inf} (1/4)^k = 4/3
    E6_correct = 1 / (1 - 1 / 4)
    print(f"  (c) Energy signal: E = sum_{{n=-inf}}^{{0}} 4^n = sum_{{k=0}}^{{inf}} (1/4)^k = {E6_correct:.4f}")
    print("  (d) Anticausal (nonzero for n <= 0)")

    # Numerical verification
    print("\n--- Numerical Verification ---")
    # Signal 2: energy
    n = np.arange(0, 100)
    x2 = (-0.5) ** n
    E2_num = np.sum(np.abs(x2) ** 2)
    print(f"Signal 2 energy (numerical): {E2_num:.4f}, analytical: {1 / (1 - 0.25):.4f}")

    # Signal 5: energy
    t = np.linspace(-1, 7, 100000)
    dt = t[1] - t[0]
    x5 = np.where((t >= 0) & (t <= 5), 1.0, 0.0)
    E5_num = np.trapz(x5 ** 2, t)
    print(f"Signal 5 energy (numerical): {E5_num:.4f}, analytical: 5.0000")

    # Signal 6: energy
    n_neg = np.arange(-100, 1)
    x6 = 2.0 ** n_neg
    E6_num = np.sum(np.abs(x6) ** 2)
    print(f"Signal 6 energy (numerical): {E6_num:.4f}, analytical: {4 / 3:.4f}")


# === Exercise 2: Even-Odd Decomposition ===
# Problem: Decompose x(t) = {t+1 for 0<=t<=1, 2 for 1<t<=2, 0 otherwise}
# into even and odd components.

def exercise_2():
    """Even-odd decomposition of a piecewise signal."""
    t = np.linspace(-3, 3, 2000)

    def x(t_val):
        """Original signal."""
        result = np.zeros_like(t_val)
        mask1 = (t_val >= 0) & (t_val <= 1)
        mask2 = (t_val > 1) & (t_val <= 2)
        result[mask1] = t_val[mask1] + 1
        result[mask2] = 2.0
        return result

    x_t = x(t)
    x_neg_t = x(-t)  # x(-t)

    x_even = 0.5 * (x_t + x_neg_t)
    x_odd = 0.5 * (x_t - x_neg_t)

    # Verify reconstruction
    reconstruction_error = np.max(np.abs(x_t - (x_even + x_odd)))
    print(f"Reconstruction error: {reconstruction_error:.2e}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(t, x_t, 'b-', linewidth=2)
    axes[0].set_title('Original x(t)')
    axes[0].set_xlabel('t')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-1.5, 2.5])

    axes[1].plot(t, x_even, 'r-', linewidth=2)
    axes[1].set_title('Even part x_e(t)')
    axes[1].set_xlabel('t')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-1.5, 2.5])

    axes[2].plot(t, x_odd, 'g-', linewidth=2)
    axes[2].set_title('Odd part x_o(t)')
    axes[2].set_xlabel('t')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-1.5, 2.5])

    plt.tight_layout()
    plt.savefig('ex01_even_odd_decomp.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex01_even_odd_decomp.png")


# === Exercise 3: Energy Computation ===
# Problem: Compute energy analytically and verify numerically.

def exercise_3():
    """Energy computation for various signals."""
    # Signal 1: x(t) = 2*e^{-3t}*u(t)
    # E = integral_0^inf |2*e^{-3t}|^2 dt = 4 * integral_0^inf e^{-6t} dt = 4/6 = 2/3
    E1_analytical = 4 / 6
    t = np.linspace(0, 20, 200000)
    dt = t[1] - t[0]
    x1 = 2 * np.exp(-3 * t)
    E1_numerical = np.trapz(x1 ** 2, t)
    print(f"Signal 1: x(t) = 2*e^{{-3t}}*u(t)")
    print(f"  Analytical energy: {E1_analytical:.6f}")
    print(f"  Numerical energy:  {E1_numerical:.6f}")
    print()

    # Signal 2: x[n] = (0.8)^n * u[n]
    # E = sum_{n=0}^{inf} 0.64^n = 1/(1-0.64) = 1/0.36
    E2_analytical = 1 / (1 - 0.64)
    n = np.arange(0, 500)
    x2 = 0.8 ** n
    E2_numerical = np.sum(x2 ** 2)
    print(f"Signal 2: x[n] = (0.8)^n * u[n]")
    print(f"  Analytical energy: {E2_analytical:.6f}")
    print(f"  Numerical energy:  {E2_numerical:.6f}")
    print()

    # Signal 3: x(t) = rect(t/4)  (width 4, centered at 0)
    # E = integral_{-2}^{2} 1 dt = 4
    E3_analytical = 4.0
    t3 = np.linspace(-5, 5, 200000)
    x3 = np.where(np.abs(t3) <= 2, 1.0, 0.0)
    E3_numerical = np.trapz(x3 ** 2, t3)
    print(f"Signal 3: x(t) = rect(t/4)")
    print(f"  Analytical energy: {E3_analytical:.6f}")
    print(f"  Numerical energy:  {E3_numerical:.6f}")


# === Exercise 4: System Properties ===
# Problem: For each system, determine linearity, time-invariance, causality,
# stability, and memorylessness.

def exercise_4():
    """Determine system properties analytically and verify numerically."""
    print("System 1: y(t) = x(t-2)")
    print("  Linear: YES (superposition holds)")
    print("  Time-invariant: YES (pure delay)")
    print("  Causal: YES (output depends only on past input)")
    print("  Stable: YES (bounded input -> bounded output)")
    print("  Memoryless: NO (depends on x at t-2)")
    print()

    print("System 2: y[n] = n*x[n]")
    print("  Linear: YES (superposition: n*(a*x1+b*x2) = a*n*x1 + b*n*x2)")
    print("  Time-invariant: NO (coefficient n changes with time)")
    print("  Causal: YES (depends only on x[n])")
    print("  Stable: NO (for bounded x, |y[n]| = |n|*|x[n]| grows unbounded)")
    print("  Memoryless: YES (depends only on current x[n])")
    print()

    print("System 3: y(t) = cos(x(t))")
    print("  Linear: NO (cos is nonlinear, e.g., cos(0) = 1 != 0)")
    print("  Time-invariant: YES (no explicit time dependence)")
    print("  Causal: YES (depends only on current input)")
    print("  Stable: YES (|cos(x)| <= 1 always)")
    print("  Memoryless: YES")
    print()

    print("System 4: y[n] = x[-n]")
    print("  Linear: YES (superposition holds)")
    print("  Time-invariant: NO (time reversal is not shift-invariant)")
    print("  Causal: NO (y[n] depends on x[-n], i.e., future for n<0)")
    print("  Stable: YES (bounded input -> bounded output)")
    print("  Memoryless: NO (depends on x at different times)")
    print()

    print("System 5: y(t) = x(t)*cos(2*pi*f0*t)")
    print("  Linear: YES (scaling and addition preserved)")
    print("  Time-invariant: NO (time-varying coefficient cos(2*pi*f0*t))")
    print("  Causal: YES (depends only on current input)")
    print("  Stable: YES (|y| <= |x| since |cos| <= 1)")
    print("  Memoryless: YES")
    print()

    print("System 6: y[n] = sum_{k=n-2}^{n+2} x[k] (5-point centered average)")
    print("  Linear: YES (sum is linear)")
    print("  Time-invariant: YES (fixed window relative to n)")
    print("  Causal: NO (depends on x[n+1] and x[n+2])")
    print("  Stable: YES (bounded input -> bounded output)")
    print("  Memoryless: NO (depends on past and future)")
    print()

    print("System 7: y(t) = x(t) + 3")
    print("  Linear: NO (T{0} = 3 != 0)")
    print("  Time-invariant: YES")
    print("  Causal: YES")
    print("  Stable: YES")
    print("  Memoryless: YES")

    # Numerical verification for system 1 (linearity + time-invariance)
    print("\n--- Numerical Verification (System 1: y[n] = x[n-2]) ---")
    np.random.seed(42)
    N = 50

    def sys_delay2(x):
        y = np.zeros_like(x)
        y[2:] = x[:-2]
        return y

    x1 = np.random.randn(N)
    x2 = np.random.randn(N)
    a, b = 2.5, -1.3

    # Linearity test
    lhs = sys_delay2(a * x1 + b * x2)
    rhs = a * sys_delay2(x1) + b * sys_delay2(x2)
    lin_err = np.max(np.abs(lhs - rhs))
    print(f"  Linearity error: {lin_err:.2e} -> {'LINEAR' if lin_err < 1e-10 else 'NONLINEAR'}")

    # Time-invariance test
    delay = 3
    x_del = np.zeros(N)
    x_del[delay:] = x1[:N - delay]
    y_shifted_input = sys_delay2(x_del)
    y = sys_delay2(x1)
    y_then_shift = np.zeros(N)
    y_then_shift[delay:] = y[:N - delay]
    ti_err = np.max(np.abs(y_shifted_input - y_then_shift))
    print(f"  TI error: {ti_err:.2e} -> {'TIME-INVARIANT' if ti_err < 1e-10 else 'TIME-VARYING'}")


# === Exercise 5: BIBO Stability ===
# Problem: Determine BIBO stability for given impulse responses.

def exercise_5():
    """BIBO stability determination."""
    print("System 1: h(t) = e^{-2t}*u(t)")
    print("  integral |h(t)| dt = integral_0^inf e^{-2t} dt = 1/2 < inf")
    print("  BIBO STABLE")
    print()

    print("System 2: h[n] = u[n]")
    print("  sum |h[n]| = sum_{n=0}^{inf} 1 = inf")
    print("  BIBO UNSTABLE")
    print()

    print("System 3: h[n] = (0.9)^|n|")
    s = 1 / (1 - 0.9) + 1 / (1 - 0.9) - 1  # sum from -inf to inf of 0.9^|n|
    print(f"  sum |h[n]| = sum 0.9^|n| = 2/(1-0.9) - 1 = {s:.2f} < inf")
    print("  BIBO STABLE")
    print()

    print("System 4: h(t) = sin(t)/t")
    print("  integral |sin(t)/t| dt = inf (known result: Si integral diverges)")
    print("  BIBO UNSTABLE")
    print()

    print("System 5: h[n] = delta[n] - 0.5*delta[n-1]")
    h5 = np.array([1.0, -0.5])
    s5 = np.sum(np.abs(h5))
    print(f"  sum |h[n]| = |1| + |-0.5| = {s5:.2f} < inf")
    print("  BIBO STABLE")

    # Numerical verification
    print("\n--- Numerical Verification ---")
    n = np.arange(0, 10000)

    # System 1 (discrete approximation)
    t = np.linspace(0, 50, 100000)
    h1 = np.exp(-2 * t)
    dt = t[1] - t[0]
    print(f"  System 1: integral |h| = {np.trapz(np.abs(h1), t):.4f} (analytical: 0.5)")

    # System 3
    n_full = np.arange(-500, 501)
    h3 = 0.9 ** np.abs(n_full)
    print(f"  System 3: sum |h| = {np.sum(np.abs(h3)):.4f} (analytical: {s:.4f})")


# === Exercise 6: Signal Analyzer ===
# Problem: Write a signal_analyzer function that returns signal properties.

def exercise_6():
    """Signal analyzer implementation."""
    def signal_analyzer(x, fs):
        """Analyze a discrete-time signal and return its properties."""
        N = len(x)
        duration = N / fs

        # Basic properties
        max_abs = np.max(np.abs(x))
        energy = np.sum(np.abs(x) ** 2) / fs  # approximate continuous energy
        avg_power = energy / duration

        # Periodicity detection using autocorrelation
        x_centered = x - np.mean(x)
        acf = np.correlate(x_centered, x_centered, mode='full')
        acf = acf[N - 1:]  # keep positive lags only
        acf = acf / acf[0] if acf[0] > 0 else acf

        # Find peaks in autocorrelation (skip lag 0)
        min_lag = int(fs / 1000)  # minimum period: 1 ms
        max_lag = N // 2
        if max_lag > min_lag + 2:
            acf_search = acf[min_lag:max_lag]
            # Find local maxima
            peaks = []
            for i in range(1, len(acf_search) - 1):
                if acf_search[i] > acf_search[i - 1] and acf_search[i] > acf_search[i + 1]:
                    if acf_search[i] > 0.5:  # threshold
                        peaks.append(i + min_lag)

            is_periodic = len(peaks) > 0
            if is_periodic:
                fund_period_samples = peaks[0]
                fund_freq = fs / fund_period_samples
            else:
                fund_freq = None
        else:
            is_periodic = False
            fund_freq = None

        return {
            'duration': duration,
            'max_abs': max_abs,
            'energy': energy,
            'avg_power': avg_power,
            'is_periodic': is_periodic,
            'fundamental_frequency': fund_freq
        }

    fs = 8000
    duration = 1.0
    t = np.arange(0, duration, 1 / fs)

    # Test 1: Sinusoid
    x_sin = np.sin(2 * np.pi * 440 * t)
    result = signal_analyzer(x_sin, fs)
    print("Sinusoid (440 Hz):")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print()

    # Test 2: Chirp
    from scipy.signal import chirp as make_chirp
    x_chirp = make_chirp(t, f0=100, f1=2000, t1=duration)
    result = signal_analyzer(x_chirp, fs)
    print("Chirp (100-2000 Hz):")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print()

    # Test 3: White noise
    np.random.seed(42)
    x_noise = np.random.randn(len(t))
    result = signal_analyzer(x_noise, fs)
    print("White noise:")
    for k, v in result.items():
        print(f"  {k}: {v}")


# === Exercise 7: Signal Operations Challenge ===
# Problem: Apply various operations to a triangular pulse and compute energies.

def exercise_7():
    """Signal operations on a triangular pulse."""
    t = np.linspace(-5, 10, 10000)
    dt = t[1] - t[0]

    def tri(t_val):
        """Triangular pulse: peak 1 at t=0, support [-1, 1]."""
        return np.maximum(0, 1 - np.abs(t_val))

    # y1(t) = x(2t - 3): compress by 2, shift right by 3/2
    # Peak at t = 3/2, support [1/2, 5/2]
    x_t = tri(t)
    y1 = tri(2 * t - 3)

    # y2(t) = x(-t + 1) + x(t - 1)
    # x(-t+1): reflect, shift right by 1 -> peak at t=1, support [0, 2]
    # x(t-1): shift right by 1 -> peak at t=1, support [0, 2]
    y2 = tri(-t + 1) + tri(t - 1)

    # Energy computations
    E_x = np.trapz(x_t ** 2, t)
    E_y1 = np.trapz(y1 ** 2, t)
    E_y2 = np.trapz(y2 ** 2, t)

    # Analytical: E_x = integral_{-1}^{1} (1-|t|)^2 dt = 2 * integral_0^1 (1-t)^2 dt = 2/3
    E_x_analytical = 2 / 3
    # E_y1: x(2t-3) -> time scaling by a=2: E_y1 = (1/|a|) * E_x = (1/2) * 2/3 = 1/3
    E_y1_analytical = E_x_analytical / 2

    print(f"Energy of x(t):  numerical={E_x:.6f}, analytical={E_x_analytical:.6f}")
    print(f"Energy of y1(t): numerical={E_y1:.6f}, analytical={E_y1_analytical:.6f}")
    print(f"Energy of y2(t): numerical={E_y2:.6f}")
    print()
    print("Relationship: E_y1 = E_x / 2 (time compression by 2 halves energy)")
    print("y2 is a sum of two overlapping pulses, so E_y2 != 2*E_x due to cross-term")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(t, x_t, 'b-', linewidth=2)
    axes[0].set_title(f'x(t), E = {E_x:.4f}')
    axes[0].set_xlim([-3, 5])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, y1, 'r-', linewidth=2)
    axes[1].set_title(f'y1(t) = x(2t-3), E = {E_y1:.4f}')
    axes[1].set_xlim([-3, 5])
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, y2, 'g-', linewidth=2)
    axes[2].set_title(f'y2(t) = x(-t+1) + x(t-1), E = {E_y2:.4f}')
    axes[2].set_xlim([-3, 5])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex01_signal_operations.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex01_signal_operations.png")


# === Exercise 8: Complex Exponentials and Phasors ===
# Problem: Express x(t) = 3*cos(10t+pi/4) + 4*sin(10t-pi/6) as A*cos(10t+phi).

def exercise_8():
    """Phasor addition of sinusoids."""
    # x(t) = 3*cos(10t + pi/4) + 4*sin(10t - pi/6)
    # Convert sin to cos: sin(theta) = cos(theta - pi/2)
    # 4*sin(10t - pi/6) = 4*cos(10t - pi/6 - pi/2) = 4*cos(10t - 2pi/3)

    # Phasor representation: X = A1*e^{j*phi1} + A2*e^{j*phi2}
    A1, phi1 = 3.0, np.pi / 4
    A2, phi2 = 4.0, -2 * np.pi / 3  # after converting sin to cos

    # Complex phasors
    P1 = A1 * np.exp(1j * phi1)
    P2 = A2 * np.exp(1j * phi2)
    P_total = P1 + P2

    A = np.abs(P_total)
    phi = np.angle(P_total)

    print(f"Phasor 1: {A1} * exp(j * {phi1:.4f}) = {P1.real:.4f} + j{P1.imag:.4f}")
    print(f"Phasor 2: {A2} * exp(j * {phi2:.4f}) = {P2.real:.4f} + j{P2.imag:.4f}")
    print(f"Total:    {A:.4f} * exp(j * {phi:.4f})")
    print(f"\nResult: x(t) = {A:.4f} * cos(10t + ({phi:.4f}))")
    print(f"        A = {A:.4f}")
    print(f"        phi = {phi:.4f} rad = {np.degrees(phi):.2f} degrees")

    # Complex exponential form: x(t) = Re{C * e^{j*10t}} where C = A*e^{j*phi}
    C = P_total
    print(f"\nComplex exponential: C = {C.real:.4f} + j{C.imag:.4f}")
    print(f"  x(t) = Re{{ ({C.real:.4f} + j{C.imag:.4f}) * e^{{j*10t}} }}")

    # Verification
    t = np.linspace(0, 2, 1000)
    x_orig = 3 * np.cos(10 * t + np.pi / 4) + 4 * np.sin(10 * t - np.pi / 6)
    x_combined = A * np.cos(10 * t + phi)
    error = np.max(np.abs(x_orig - x_combined))
    print(f"\nVerification error: {error:.2e}")

    # Phasor diagram
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.arrow(0, 0, P1.real, P1.imag, head_width=0.1, head_length=0.05,
             fc='blue', ec='blue', linewidth=2)
    ax.arrow(0, 0, P2.real, P2.imag, head_width=0.1, head_length=0.05,
             fc='red', ec='red', linewidth=2)
    ax.arrow(0, 0, P_total.real, P_total.imag, head_width=0.1, head_length=0.05,
             fc='green', ec='green', linewidth=2.5)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Phasor Diagram')
    ax.legend(['P1 (3, pi/4)', 'P2 (4, -2pi/3)', 'P_total'])
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('ex01_phasor_diagram.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Phasor diagram saved to ex01_phasor_diagram.png")


if __name__ == "__main__":
    print("=== Exercise 1: Signal Classification ===")
    exercise_1()
    print("\n=== Exercise 2: Even-Odd Decomposition ===")
    exercise_2()
    print("\n=== Exercise 3: Energy Computation ===")
    exercise_3()
    print("\n=== Exercise 4: System Properties ===")
    exercise_4()
    print("\n=== Exercise 5: BIBO Stability ===")
    exercise_5()
    print("\n=== Exercise 6: Signal Analyzer ===")
    exercise_6()
    print("\n=== Exercise 7: Signal Operations Challenge ===")
    exercise_7()
    print("\n=== Exercise 8: Complex Exponentials and Phasors ===")
    exercise_8()
    print("\nAll exercises completed!")
