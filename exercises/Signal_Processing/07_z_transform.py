"""
Exercises for Lesson 07: Z-Transform
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal as sig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Z-Transform Computation ===
# Problem: Compute Z-transforms and ROCs.

def exercise_1():
    """Z-transform computation with ROC specification."""
    print("1a) x[n] = (0.5)^n u[n] + (0.8)^n u[n]")
    print("    X(z) = z/(z-0.5) + z/(z-0.8)")
    print("         = z(2z-1.3) / ((z-0.5)(z-0.8))")
    print("    ROC: |z| > 0.8")
    print()

    print("1b) x[n] = (0.6)^n u[n] - 2*(0.6)^n u[n-3]")
    print("    X(z) = z/(z-0.6) - 2*z^{-3} * z/(z-0.6) * (0.6)^3 ... ")
    print("    = z/(z-0.6) * (1 - 2*0.216*z^{-3})")
    print("    ROC: |z| > 0.6")
    print()

    print("1c) x[n] = n*(0.9)^n u[n]")
    print("    Using z-domain differentiation: n*a^n u[n] <-> az/(z-a)^2")
    print("    X(z) = 0.9z/(z-0.9)^2")
    print("    ROC: |z| > 0.9")
    print()

    print("1d) x[n] = (0.7)^|n| (two-sided)")
    print("    x[n] = (0.7)^n u[n] + (0.7)^{-n} u[-n-1]")
    print("    X(z) = z/(z-0.7) - z/(z-1/0.7)")
    print("    ROC: 0.7 < |z| < 1/0.7 ≈ 1.429")

    # Numerical verification via inverse Z (1a)
    N = 50
    n = np.arange(N)
    x_a = 0.5 ** n + 0.8 ** n
    print(f"\nVerification (1a): x[0..4] = {x_a[:5]}")


# === Exercise 2: Inverse Z-Transform ===
# Problem: Find x[n] using partial fractions.

def exercise_2():
    """Inverse Z-transform via partial fractions."""
    # 2a) X(z) = z/((z-0.5)(z-0.8)), |z| > 0.8 (causal)
    # Partial fractions: X(z)/z = 1/((z-0.5)(z-0.8))
    # = A/(z-0.5) + B/(z-0.8)
    # A = 1/(0.5-0.8) = -1/0.3 ≈ -3.333
    # B = 1/(0.8-0.5) = 1/0.3 ≈ 3.333
    A = 1 / (0.5 - 0.8)
    B = 1 / (0.8 - 0.5)
    print(f"2a) |z| > 0.8 (causal):")
    print(f"    A = {A:.4f}, B = {B:.4f}")
    print(f"    x[n] = ({A:.4f} * 0.5^n + {B:.4f} * 0.8^n) * u[n]")
    print()

    # 2b) Same X(z), |z| < 0.5 (anti-causal)
    print(f"2b) |z| < 0.5 (anti-causal):")
    print(f"    x[n] = -({A:.4f} * 0.5^n + {B:.4f} * 0.8^n) * u[-n-1]")
    print()

    # 2c) X(z) = z^2/((z-0.5)(z-0.8)), |z| > 0.8
    # X(z)/z = z/((z-0.5)(z-0.8))
    # Long division or partial fractions
    # X(z) = z^2/((z-0.5)(z-0.8)) = 1 + 1.3z/((z-0.5)(z-0.8)) + ...
    # Actually: z^2 = (z-0.5)(z-0.8) + 1.3z - 0.4
    # So X(z) = 1 + (1.3z - 0.4)/((z-0.5)(z-0.8))
    print(f"2c) |z| > 0.8:")
    print(f"    x[n] = delta[n] + (partial fraction expansion of remainder)*u[n]")

    # Verify with scipy
    b = [1, 0, 0]  # z^2
    a = np.convolve([1, -0.5], [1, -0.8])  # (z-0.5)(z-0.8)
    # z^2 / (z^2 - 1.3z + 0.4)
    # Impulse response
    N = 20
    t_imp = np.zeros(N)
    t_imp[0] = 1.0
    b_tf = [1, 0, 0]
    a_tf = [1, -1.3, 0.4]
    h = sig.lfilter(b_tf, a_tf, t_imp)
    print(f"    h[0..5] = {h[:6]}")
    print()

    # 2d) X(z) = (1+2z^{-1})/(1-z^{-1}+0.5z^{-2}), |z| > 0.707
    b_d = [1, 2]
    a_d = [1, -1, 0.5]
    h_d = sig.lfilter(b_d, a_d, t_imp)
    print(f"2d) x[n] via lfilter: {h_d[:8]}")

    # Poles
    poles = np.roots(a_d)
    print(f"    Poles: {poles}")
    print(f"    |poles|: {np.abs(poles)}")
    print(f"    (|poles| = sqrt(0.5) ≈ 0.707, confirms ROC condition)")


# === Exercise 3: System Analysis ===
# Problem: Analyze y[n] = 0.8y[n-1] - 0.64y[n-2] + x[n] + x[n-1].

def exercise_3():
    """Complete system analysis via Z-transform."""
    b = [1, 1]
    a = [1, -0.8, 0.64]

    # (a) Transfer function and pole-zero plot
    zeros = np.roots(b)
    poles = np.roots(a)
    print(f"(a) H(z) = (1 + z^{{-1}}) / (1 - 0.8z^{{-1}} + 0.64z^{{-2}})")
    print(f"    Zeros: {zeros}")
    print(f"    Poles: {poles}")
    print(f"    |poles|: {np.abs(poles)}")
    print()

    # (b) Stability
    stable = np.all(np.abs(poles) < 1)
    print(f"(b) Stable: {stable} (all poles inside unit circle)")
    print()

    # (c) Impulse response
    N = 50
    imp = np.zeros(N)
    imp[0] = 1.0
    h = sig.lfilter(b, a, imp)
    print(f"(c) First 8 impulse response values: {np.round(h[:8], 4)}")
    print()

    # (d-e) Frequency response
    w, H = sig.freqz(b, a, worN=1024)
    mag = 20 * np.log10(np.abs(H))
    phase = np.angle(H)

    # Determine filter type
    dc_gain = np.abs(H[0])
    nyquist_gain = np.abs(H[-1])
    mid_gain = np.abs(H[len(H) // 4])
    print(f"(e) |H(0)| = {dc_gain:.4f}")
    print(f"    |H(pi/2)| = {mid_gain:.4f}")
    print(f"    |H(pi)| = {nyquist_gain:.4f}")

    if dc_gain > nyquist_gain and dc_gain > mid_gain:
        ftype = "lowpass"
    elif nyquist_gain > dc_gain and nyquist_gain > mid_gain:
        ftype = "highpass"
    elif mid_gain > dc_gain and mid_gain > nyquist_gain:
        ftype = "bandpass"
    else:
        ftype = "other"
    print(f"    Filter type: {ftype}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Pole-zero plot
    theta = np.linspace(0, 2 * np.pi, 100)
    axes[0].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    axes[0].plot(np.real(zeros), np.imag(zeros), 'bo', markersize=10, label='Zeros')
    axes[0].plot(np.real(poles), np.imag(poles), 'rx', markersize=10, mew=2, label='Poles')
    axes[0].set_title('Pole-Zero Plot')
    axes[0].set_aspect('equal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(w / np.pi, mag)
    axes[1].set_title('Magnitude Response')
    axes[1].set_xlabel('Normalized Freq (x pi)')
    axes[1].set_ylabel('dB')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(w / np.pi, phase * 180 / np.pi)
    axes[2].set_title('Phase Response')
    axes[2].set_xlabel('Normalized Freq (x pi)')
    axes[2].set_ylabel('Degrees')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex07_system_analysis.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex07_system_analysis.png")


# === Exercise 4: Stability Determination ===
# Problem: Determine stability using Jury test and root computation.

def exercise_4():
    """Stability determination via Jury test and root analysis."""
    systems = {
        '4a': [1, -0.5, 0.06],
        '4b': [1, -1.4, 0.85],
        '4c': [1, -1.8 * np.cos(0.4 * np.pi), 0.81]
    }

    for name, a in systems.items():
        poles = np.roots(a)
        pole_mags = np.abs(poles)
        stable = np.all(pole_mags < 1)

        print(f"{name}) a = {[f'{c:.4f}' for c in a]}")
        print(f"    Poles: {poles}")
        print(f"    |poles|: {pole_mags}")
        print(f"    Stable: {stable}")

        # Jury stability test (simplified for 2nd order)
        # For a(z) = z^2 + a1*z + a2:
        # Conditions: |a2| < 1, a(1) > 0, a(-1) > 0
        a2 = a[2]
        a_of_1 = sum(a)
        a_of_neg1 = a[0] - a[1] + a[2]
        print(f"    Jury test: |a2|={abs(a2):.4f}<1? {abs(a2) < 1}, "
              f"a(1)={a_of_1:.4f}>0? {a_of_1 > 0}, "
              f"a(-1)={a_of_neg1:.4f}>0? {a_of_neg1 > 0}")
        jury_stable = abs(a2) < 1 and a_of_1 > 0 and a_of_neg1 > 0
        print(f"    Jury result: {'STABLE' if jury_stable else 'UNSTABLE'}")
        print()


# === Exercise 5: ROC and Signal Determination ===
# Problem: Find all ROCs and corresponding signals.

def exercise_5():
    """ROC analysis and signal determination."""
    # X(z) = (2z^2 - 1.5z) / (z^2 - 0.9z + 0.2)
    b = [2, -1.5, 0]
    a = [1, -0.9, 0.2]
    poles = np.roots(a)

    print(f"X(z) = (2z^2 - 1.5z) / (z^2 - 0.9z + 0.2)")
    print(f"Poles: {poles}")
    print()

    p1, p2 = sorted(np.abs(poles))

    print(f"(a) Three possible ROCs:")
    print(f"    ROC 1: |z| > {np.max(np.abs(poles)):.2f} (causal)")
    print(f"    ROC 2: |z| < {np.min(np.abs(poles)):.2f} (anti-causal)")
    print(f"    ROC 3: {np.min(np.abs(poles)):.2f} < |z| < {np.max(np.abs(poles)):.2f} (two-sided)")
    print()

    # Partial fractions: X(z)/z = (2z-1.5)/((z-p1)(z-p2))
    r, p, k = sig.residuez(b, a)
    print(f"Partial fraction residues: {r}")
    print(f"Partial fraction poles: {p}")
    print(f"Direct term: {k}")
    print()

    # (b) Signals for each ROC
    print("(b) Corresponding signals:")
    print(f"    ROC 1 (causal): x[n] = ({r[0]:.4f}*{p[0]:.4f}^n + {r[1]:.4f}*{p[1]:.4f}^n)*u[n]")
    print(f"    ROC 2 (anti-causal): x[n] = -(...)*u[-n-1]")
    print(f"    ROC 3 (two-sided): mixed causal/anti-causal terms")
    print()

    # (c) DTFT exists when ROC includes unit circle
    print("(c) DTFT exists for ROC that includes |z|=1:")
    for i, (roc_name, includes_uc) in enumerate([
        ("ROC 1: |z|>0.5", True),
        ("ROC 2: |z|<0.4", False),
        ("ROC 3: 0.4<|z|<0.5", False)
    ]):
        print(f"    {roc_name}: DTFT {'EXISTS' if includes_uc else 'does NOT exist'}")


# === Exercise 6: Transfer Function Design ===
# Problem: Design a resonant second-order system.

def exercise_6():
    """Second-order resonant filter design."""
    omega0 = np.pi / 3  # rad/sample
    delta_omega = 0.1  # desired bandwidth

    # (a) Pole placement: r determines bandwidth, approx bw = 2(1-r)
    r = 1 - delta_omega / 2  # r = 0.95
    print(f"(a) Resonant frequency: omega0 = pi/3 = {omega0:.4f} rad/sample")
    print(f"    Desired bandwidth: {delta_omega} rad/sample")
    print(f"    Pole radius: r = {r:.4f}")

    # Poles at z = r*e^{+/-j*omega0}
    p1 = r * np.exp(1j * omega0)
    p2 = r * np.exp(-1j * omega0)
    print(f"    Poles: {p1:.4f}, {p2:.4f}")

    # (b) Zeros at origin for simplicity, normalize for unity gain at resonance
    # H(z) = b0 / ((z - p1)(z - p2))
    # Denominator: z^2 - 2r*cos(omega0)*z + r^2
    a = [1, -2 * r * np.cos(omega0), r ** 2]

    # Unity gain at resonance: |H(e^{j*omega0})| = 1
    z_res = np.exp(1j * omega0)
    H_at_res = 1 / (z_res ** 2 + a[1] * z_res + a[2])
    b0 = 1 / np.abs(H_at_res)
    b = [b0, 0, 0]

    print(f"(b) b0 for unity gain at resonance: {b0:.6f}")
    print(f"    H(z) = {b0:.4f} / (1 - {-a[1]:.4f}z^-1 + {a[2]:.4f}z^-2)")
    print()

    # (c) Frequency response
    w, H = sig.freqz(b, a, worN=1024)
    gain_at_res = np.abs(np.interp(omega0, w, np.abs(H)))
    bw_3dB = np.abs(np.interp(1 / np.sqrt(2), np.abs(H), w) - omega0) * 2
    print(f"(c) Gain at resonance: {gain_at_res:.4f}")
    print(f"    Measured bandwidth: ~{2 * (1 - r):.4f} rad/sample")

    # (d) Test with signal
    fs = 1000  # for convenience
    N = 500
    n = np.arange(N)
    f_res = omega0 / (2 * np.pi) * fs
    x = (np.sin(2 * np.pi * f_res / fs * n) +
         np.sin(2 * np.pi * 50 / fs * n) +
         np.sin(2 * np.pi * 400 / fs * n))
    y = sig.lfilter(b, a, x)

    print(f"\n(d) Input: {f_res:.0f} Hz + 50 Hz + 400 Hz")
    print(f"    Output dominated by {f_res:.0f} Hz component")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(w / np.pi, 20 * np.log10(np.abs(H) + 1e-10))
    axes[0].set_title(f'Resonant Filter (f0={omega0 / np.pi:.2f}*pi)')
    axes[0].set_xlabel('Normalized Frequency (x pi)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(n[:200], x[:200], alpha=0.5, label='Input')
    axes[1].plot(n[:200], y[:200], label='Output')
    axes[1].set_title('Filter Test')
    axes[1].set_xlabel('n')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex07_resonant_filter.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex07_resonant_filter.png")


# === Exercise 7: Digital Oscillator ===
# Problem: Implement a digital oscillator using difference equation.

def exercise_7():
    """Digital oscillator implementation and analysis."""
    # y[n] = 2*cos(omega0)*y[n-1] - y[n-2]
    fs = 8000
    f0 = 440
    omega0 = 2 * np.pi * f0 / fs
    N = 10000

    # (a) Transfer function: H(z) = z/(z^2 - 2*cos(omega0)*z + 1)
    a = [1, -2 * np.cos(omega0), 1]
    poles = np.roots(a)
    print(f"(a) Digital oscillator for {f0} Hz at fs={fs} Hz")
    print(f"    omega0 = {omega0:.6f} rad/sample")
    print(f"    Poles: {poles}")
    print(f"    |poles|: {np.abs(poles)}")
    print(f"    Poles ON the unit circle -> marginally stable")
    print()

    # (b) Marginal stability
    print("(b) Marginally stable: poles exactly on unit circle")
    print("    In finite precision, |poles| may drift slightly from 1")
    print("    This causes amplitude to grow or decay over time")
    print()

    # (c) Implementation
    y = np.zeros(N)
    # Initial conditions for cos(omega0*n): y[-1] = cos(-omega0), y[-2] = cos(-2*omega0)
    y_prev1 = np.cos(-omega0)
    y_prev2 = np.cos(-2 * omega0)

    coeff = 2 * np.cos(omega0)
    for n in range(N):
        y[n] = coeff * y_prev1 - y_prev2
        y_prev2 = y_prev1
        y_prev1 = y[n]

    # Direct computation
    n_arr = np.arange(N)
    y_direct = np.cos(omega0 * n_arr)

    error = np.abs(y - y_direct)
    print(f"(c) After {N} samples:")
    print(f"    Max error: {np.max(error):.2e}")
    print(f"    Mean error: {np.mean(error):.2e}")
    print(f"    Final amplitude (oscillator): {y[-1]:.6f}")
    print(f"    Final amplitude (direct):     {y_direct[-1]:.6f}")
    print()
    print("    Error grows over time due to finite-precision arithmetic")
    print("    but for audio applications, this drift is usually acceptable")


if __name__ == "__main__":
    print("=== Exercise 1: Z-Transform Computation ===")
    exercise_1()
    print("\n=== Exercise 2: Inverse Z-Transform ===")
    exercise_2()
    print("\n=== Exercise 3: System Analysis ===")
    exercise_3()
    print("\n=== Exercise 4: Stability Determination ===")
    exercise_4()
    print("\n=== Exercise 5: ROC and Signal Determination ===")
    exercise_5()
    print("\n=== Exercise 6: Transfer Function Design ===")
    exercise_6()
    print("\n=== Exercise 7: Digital Oscillator ===")
    exercise_7()
    print("\nAll exercises completed!")
