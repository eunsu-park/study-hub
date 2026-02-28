"""
Exercises for Lesson 13: Adaptive Filters
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def lms_filter(x, d, M, mu):
    """LMS adaptive filter."""
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    w_history = np.zeros((N, M))
    for n in range(M, N):
        x_vec = x[n-M+1:n+1][::-1]
        y[n] = np.dot(w, x_vec)
        e[n] = d[n] - y[n]
        w = w + mu * e[n] * x_vec
        w_history[n] = w
    return y, e, w_history


def nlms_filter(x, d, M, mu_tilde, delta=1e-6):
    """Normalized LMS adaptive filter."""
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    w_history = np.zeros((N, M))
    for n in range(M, N):
        x_vec = x[n-M+1:n+1][::-1]
        y[n] = np.dot(w, x_vec)
        e[n] = d[n] - y[n]
        norm_sq = np.dot(x_vec, x_vec) + delta
        w = w + (mu_tilde / norm_sq) * e[n] * x_vec
        w_history[n] = w
    return y, e, w_history


def rls_filter(x, d, M, lam=0.99, delta=0.01):
    """RLS adaptive filter."""
    N = len(x)
    w = np.zeros(M)
    P = (1.0 / delta) * np.eye(M)
    y = np.zeros(N)
    e = np.zeros(N)
    w_history = np.zeros((N, M))
    for n in range(M, N):
        x_vec = x[n-M+1:n+1][::-1]
        Px = P @ x_vec
        denom = lam + x_vec @ Px
        k = Px / denom
        y[n] = np.dot(w, x_vec)
        e[n] = d[n] - y[n]
        w = w + k * e[n]
        P = (1.0 / lam) * (P - np.outer(k, x_vec @ P))
        w_history[n] = w
    return y, e, w_history


# === Exercise 1: Wiener Filter ===
# Problem: Compute optimal Wiener filter for a given system.

def exercise_1():
    """Wiener filter computation."""
    M = 3
    sigma_x2 = 1.0
    sigma_v2 = 0.1
    h = np.array([0.8, 0.5, -0.3])

    # (a) Autocorrelation matrix R (white noise input)
    R = sigma_x2 * np.eye(M)
    print("(a) R = sigma_x^2 * I =")
    print(R)

    # (b) Cross-correlation vector p
    # p = E[d(n) * x(n-k)] = h (for white input)
    p = h * sigma_x2
    print(f"\n(b) p = {p}")

    # (c) Optimal Wiener filter
    w_opt = np.linalg.solve(R, p)
    print(f"\n(c) w_opt = R^-1 * p = {w_opt}")
    print(f"    True system h = {h}")
    print(f"    Match: {np.allclose(w_opt, h)}")

    # (d) Minimum MSE
    sigma_d2 = np.sum(h**2) * sigma_x2 + sigma_v2
    J_min = sigma_d2 - p @ np.linalg.solve(R, p)
    print(f"\n(d) J_min = sigma_d^2 - p^T R^-1 p = {J_min:.4f}")
    print(f"    This equals the noise variance sigma_v^2 = {sigma_v2}")

    # Verify numerically
    np.random.seed(42)
    N = 50000
    x = np.random.randn(N)
    d = np.convolve(x, h, mode='full')[:N] + np.sqrt(sigma_v2) * np.random.randn(N)
    _, e, _ = lms_filter(x, d, M, mu=0.01)
    mse_empirical = np.mean(e[5000:]**2)
    print(f"    Empirical MSE (LMS, after convergence): {mse_empirical:.4f}")


# === Exercise 2: LMS Convergence ===
# Problem: Analyze LMS convergence for given eigenvalue spread.

def exercise_2():
    """LMS convergence analysis."""
    M = 10
    lambda_max = 5.0
    lambda_min = 0.1
    tr_R = 10.0
    mu = 0.01

    # (a) Maximum step size
    mu_max = 1.0 / lambda_max
    print(f"(a) Max step size for mean convergence: mu < 1/lambda_max = {mu_max:.4f}")

    # (b) Condition number
    chi = lambda_max / lambda_min
    print(f"\n(b) Condition number chi(R) = lambda_max/lambda_min = {chi:.1f}")

    # (c) Misadjustment
    M_misadj = mu * tr_R
    print(f"\n(c) Misadjustment M = mu * tr(R) = {M_misadj:.4f} = {M_misadj*100:.2f}%")

    # (d) Convergence time constant
    tau_mse = 1 / (4 * mu * lambda_min)
    print(f"\n(d) Time constant (slowest mode): tau_mse = 1/(4*mu*lambda_min) = {tau_mse:.1f} iterations")

    # (e) Effect of whitening
    print(f"\n(e) If input is whitened:")
    print(f"    All eigenvalues become equal (chi=1)")
    print(f"    Convergence becomes uniform across all modes")
    print(f"    Time constant reduces to tau = 1/(4*mu*sigma_x^2)")
    print(f"    No more zig-zagging on the performance surface")


# === Exercise 3: NLMS vs LMS ===
# Problem: Non-stationary input power, compare LMS and NLMS.

def exercise_3():
    """NLMS vs LMS with varying input power."""
    np.random.seed(42)
    M = 16
    N = 5000
    h_true = np.random.randn(M) * 0.5

    # Input with alternating power
    x = np.zeros(N)
    for i in range(N):
        segment = i // 500
        power = 0.1 if segment % 2 == 0 else 10.0
        x[i] = np.sqrt(power) * np.random.randn()

    d = np.convolve(x, h_true, mode='full')[:N] + 0.01 * np.random.randn(N)

    # (a) LMS with fixed step size
    mu_safe = 0.001  # Safe for low-power segments
    _, e_lms_safe, _ = lms_filter(x, d, M, mu=mu_safe)

    mu_fast = 0.01  # Fast for low-power, potentially unstable for high
    _, e_lms_fast, _ = lms_filter(x, d, M, mu=mu_fast)

    # (b) NLMS
    _, e_nlms, _ = nlms_filter(x, d, M, mu_tilde=0.5)

    # (c) Plot MSE learning curves
    window = 50
    mse_safe = np.convolve(e_lms_safe**2, np.ones(window)/window, mode='valid')
    mse_fast = np.convolve(e_lms_fast**2, np.ones(window)/window, mode='valid')
    mse_nlms = np.convolve(e_nlms**2, np.ones(window)/window, mode='valid')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(x[:2000], alpha=0.5)
    axes[0].set_ylabel('Input amplitude')
    axes[0].set_title('Input Signal (alternating power)')
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(mse_safe, label=f'LMS (mu={mu_safe})', alpha=0.7)
    axes[1].semilogy(mse_fast, label=f'LMS (mu={mu_fast})', alpha=0.7)
    axes[1].semilogy(mse_nlms, label='NLMS (mu_tilde=0.5)', alpha=0.7)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Learning Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex13_nlms_vs_lms.png', dpi=100)
    plt.close()

    lms_fast_stable = np.all(np.isfinite(e_lms_fast))
    print(f"(a) LMS mu={mu_safe}: slow but stable everywhere")
    print(f"    LMS mu={mu_fast}: stable={lms_fast_stable}, may diverge in high-power segments")
    print(f"\n(b) NLMS handles power variation gracefully via normalization")
    print(f"\n(c) Final MSE (last 500 samples):")
    print(f"    LMS (safe):  {np.mean(e_lms_safe[-500:]**2):.6f}")
    if lms_fast_stable:
        print(f"    LMS (fast):  {np.mean(e_lms_fast[-500:]**2):.6f}")
    print(f"    NLMS:        {np.mean(e_nlms[-500:]**2):.6f}")
    print("    Plot saved: ex13_nlms_vs_lms.png")


# === Exercise 4: RLS Implementation ===
# Problem: System identification with RLS, compare with LMS/NLMS.

def exercise_4():
    """RLS system identification with tracking."""
    np.random.seed(42)
    h_true = np.array([1.0, -0.5, 0.25, -0.125])
    M = len(h_true)
    N = 2000

    x = np.random.randn(N)
    d = np.convolve(x, h_true, mode='full')[:N] + 0.01 * np.random.randn(N)

    # (a) Compare convergence
    _, e_lms, w_lms = lms_filter(x, d, M, mu=0.01)
    _, e_nlms, w_nlms = nlms_filter(x, d, M, mu_tilde=0.5)
    _, e_rls, w_rls = rls_filter(x, d, M, lam=0.99)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for i in range(M):
        axes[0].plot(w_rls[:, i], alpha=0.7, label=f'w[{i}]')
        axes[0].axhline(h_true[i], color='k', linestyle=':', alpha=0.3)
    axes[0].set_title('RLS Weight Convergence')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Weight Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning curves
    window = 30
    axes[1].semilogy(np.convolve(e_lms**2, np.ones(window)/window, mode='valid'), label='LMS', alpha=0.7)
    axes[1].semilogy(np.convolve(e_nlms**2, np.ones(window)/window, mode='valid'), label='NLMS', alpha=0.7)
    axes[1].semilogy(np.convolve(e_rls**2, np.ones(window)/window, mode='valid'), label='RLS', alpha=0.7)
    axes[1].set_title('Learning Curves')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('MSE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex13_rls_sysid.png', dpi=100)
    plt.close()

    print(f"(a) Weight convergence after {N} iterations:")
    print(f"    True:  {h_true}")
    print(f"    LMS:   {np.round(w_lms[-1], 4)}")
    print(f"    NLMS:  {np.round(w_nlms[-1], 4)}")
    print(f"    RLS:   {np.round(w_rls[-1], 4)}")

    # (b) Lambda tradeoff
    lambdas = [0.9, 0.95, 0.98, 0.99, 0.995, 1.0]
    print(f"\n(b) Lambda vs steady-state MSE:")
    for lam in lambdas:
        _, e_rls_l, _ = rls_filter(x, d, M, lam=lam)
        ss_mse = np.mean(e_rls_l[500:]**2)
        print(f"    lambda={lam:.3f}: MSE={ss_mse:.6f}")

    # (c) System change at n=1000
    h_new = np.array([0.5, 0.3, -0.2, 0.1])
    d_change = np.zeros(N)
    for n in range(M, N):
        x_vec = x[n-M+1:n+1][::-1]
        h = h_true if n < 1000 else h_new
        d_change[n] = np.dot(h, x_vec) + 0.01 * np.random.randn()

    _, e_lms_c, _ = lms_filter(x, d_change, M, mu=0.02)
    _, e_nlms_c, _ = nlms_filter(x, d_change, M, mu_tilde=0.8)
    _, e_rls_c, _ = rls_filter(x, d_change, M, lam=0.98)

    print(f"\n(c) Tracking after system change at n=1000:")
    print(f"    MSE (n=1200-1500):")
    print(f"    LMS:  {np.mean(e_lms_c[1200:1500]**2):.6f}")
    print(f"    NLMS: {np.mean(e_nlms_c[1200:1500]**2):.6f}")
    print(f"    RLS:  {np.mean(e_rls_c[1200:1500]**2):.6f}")
    print("    RLS tracks fastest due to exponential forgetting.")
    print("    Plot saved: ex13_rls_sysid.png")


# === Exercise 5: Echo Cancellation ===
# Problem: Acoustic echo cancellation simulation.

def exercise_5():
    """Acoustic echo cancellation simulation."""
    np.random.seed(42)
    fs = 8000
    duration = 5
    N = fs * duration
    t = np.arange(N) / fs

    # (a) Far-end speech (sum of sinusoids with varying frequencies)
    far_end = np.zeros(N)
    for i in range(10):
        start = int(i * 0.5 * fs)
        end = min(start + int(0.4 * fs), N)
        freq = 200 + i * 50
        far_end[start:end] = 0.5 * np.sin(2*np.pi*freq*t[start:end])

    # (b) Room impulse response (exponentially decaying random)
    rir_length = 100
    rir = np.random.randn(rir_length)
    rir *= np.exp(-np.arange(rir_length) / 20)
    rir /= np.max(np.abs(rir))

    # Echo
    echo = np.convolve(far_end, rir, mode='full')[:N]

    # (c) Near-end noise
    near_noise = 0.01 * np.random.randn(N)
    d = echo + near_noise  # Primary microphone signal

    # (d) NLMS echo canceller
    M_filter = 128
    _, e_nlms, _ = nlms_filter(far_end, d, M_filter, mu_tilde=0.5)

    # ERLE computation
    window = 400
    d_power = np.convolve(d**2, np.ones(window)/window, mode='same')
    e_power = np.convolve(e_nlms**2, np.ones(window)/window, mode='same')
    erle = 10 * np.log10(d_power / (e_power + 1e-15))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    axes[0].plot(t, d, alpha=0.7, label='Microphone (echo + noise)')
    axes[0].set_title('Primary Input')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, e_nlms, alpha=0.7, label='Error (residual)')
    axes[1].set_title('After Echo Cancellation')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, erle, alpha=0.7)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('ERLE (dB)')
    axes[2].set_title('Echo Return Loss Enhancement')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex13_echo_cancellation.png', dpi=100)
    plt.close()

    avg_erle = np.mean(erle[2*fs:])  # After convergence
    print(f"(d) Average ERLE after convergence: {avg_erle:.1f} dB")

    # (e) Double-talk effect
    near_speech = np.zeros(N)
    near_speech[int(2.5*fs):int(3.5*fs)] = 0.3 * np.sin(2*np.pi*300*t[int(2.5*fs):int(3.5*fs)])
    d_dt = echo + near_noise + near_speech

    _, e_dt, _ = nlms_filter(far_end, d_dt, M_filter, mu_tilde=0.5)
    e_power_dt = np.convolve(e_dt**2, np.ones(window)/window, mode='same')
    d_power_dt = np.convolve(d_dt**2, np.ones(window)/window, mode='same')
    erle_dt = 10 * np.log10(d_power_dt / (e_power_dt + 1e-15))

    print(f"\n(e) Double-talk effect:")
    print(f"    ERLE during double-talk (2.5-3.5s): {np.mean(erle_dt[int(2.5*fs):int(3.5*fs)]):.1f} dB")
    print(f"    ERLE after double-talk (4-5s): {np.mean(erle_dt[4*fs:]):.1f} dB")
    print(f"    Double-talk degrades ERLE and may cause filter divergence.")
    print("    Plot saved: ex13_echo_cancellation.png")


# === Exercise 6: Adaptive Equalization ===
# Problem: BPSK over ISI channel, adaptive equalizer.

def exercise_6():
    """Adaptive channel equalization for BPSK."""
    np.random.seed(42)
    N_symbols = 5000
    M_eq = 11
    delay = 5
    SNR_dB = 20

    # Channel
    c = np.array([0.5, 1.0, 0.5])

    # (a) BPSK signal
    a = 2 * np.random.randint(0, 2, N_symbols) - 1  # +/-1

    # Channel output + noise
    x = np.convolve(a, c, mode='full')[:N_symbols]
    noise_power = np.mean(x**2) / (10**(SNR_dB/10))
    x += np.sqrt(noise_power) * np.random.randn(N_symbols)

    # (b) LMS equalizer with training
    d = np.zeros(N_symbols)
    d[delay:] = a[:N_symbols-delay]  # Desired = delayed transmitted symbols

    _, e_eq, w_eq = lms_filter(x, d, M_eq, mu=0.01)

    # (c) BER vs training length
    training_lengths = [50, 100, 200, 500, 1000, 2000]
    bers = []

    for train_len in training_lengths:
        # Retrain
        _, _, w_train = lms_filter(x[:train_len], d[:train_len], M_eq, mu=0.01)
        w_final = w_train[-1]

        # Apply fixed filter to test data
        y_test = np.convolve(x[train_len:], w_final, mode='full')[:N_symbols-train_len]
        a_hat = np.sign(y_test)
        a_true = a[delay+train_len:delay+train_len+len(a_hat)]
        min_len = min(len(a_hat), len(a_true))
        ber = np.mean(a_hat[:min_len] != a_true[:min_len])
        bers.append(ber)

    print("(c) BER vs training length:")
    for tl, ber in zip(training_lengths, bers):
        print(f"    Training={tl:>5}: BER={ber:.4f}")

    # (d) Decision-directed mode after 500 training symbols
    w_dd = np.zeros(M_eq)
    mu = 0.01
    errors = np.zeros(N_symbols)

    for n in range(M_eq, N_symbols):
        x_vec = x[n-M_eq+1:n+1][::-1]
        y = np.dot(w_dd, x_vec)

        if n < 500:
            # Training mode
            errors[n] = d[n] - y
        else:
            # Decision-directed mode
            a_dec = np.sign(y)
            errors[n] = a_dec - y

        w_dd = w_dd + mu * errors[n] * x_vec

    # BER in DD mode
    y_dd = np.convolve(x[1000:], w_dd, mode='full')[:N_symbols-1000]
    a_hat_dd = np.sign(y_dd)
    a_true_dd = a[delay+1000:delay+1000+len(a_hat_dd)]
    min_l = min(len(a_hat_dd), len(a_true_dd))
    ber_dd = np.mean(a_hat_dd[:min_l] != a_true_dd[:min_l])
    print(f"\n(d) Decision-directed BER (after training): {ber_dd:.4f}")

    # (e) Eye diagram
    y_full = np.convolve(x, w_eq[-1], mode='full')[:N_symbols]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before equalization
    for i in range(0, min(200, N_symbols-2)):
        axes[0].plot([0, 1], x[i:i+2], 'b-', alpha=0.1)
    axes[0].set_title('Eye Diagram: Before Equalization')
    axes[0].set_xlabel('Symbol period')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # After equalization
    for i in range(0, min(200, len(y_full)-2)):
        axes[1].plot([0, 1], y_full[i:i+2], 'r-', alpha=0.1)
    axes[1].set_title('Eye Diagram: After Equalization')
    axes[1].set_xlabel('Symbol period')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex13_equalization.png', dpi=100)
    plt.close()
    print("(e) Eye diagram saved: ex13_equalization.png")


# === Exercise 7: Effect of Filter Order ===
# Problem: System identification with various filter orders.

def exercise_7():
    """Effect of filter order on system identification."""
    np.random.seed(42)
    h_true = np.array([0.5, 1.2, -0.8, 0.3, -0.1])
    M_true = len(h_true)
    N = 3000

    x = np.random.randn(N)
    d = np.convolve(x, h_true, mode='full')[:N] + 0.01 * np.random.randn(N)

    orders = [3, 5, 7, 10, 20]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    print("(a) Steady-state MSE vs filter order:")
    for M in orders:
        _, e, w_hist = lms_filter(x, d, M, mu=0.01)
        ss_mse = np.mean(e[2000:]**2)
        print(f"    M={M:>3}: MSE = {ss_mse:.6f}")

        # (c) Plot identified impulse response
        w_final = w_hist[-1]
        axes[0].stem(np.arange(M), w_final, label=f'M={M}', linefmt='-', markerfmt='o', basefmt=' ')

    axes[0].stem(np.arange(M_true), h_true, label='True', linefmt='k-', markerfmt='k*', basefmt=' ')
    axes[0].set_xlabel('Tap index')
    axes[0].set_ylabel('Weight value')
    axes[0].set_title('Identified Impulse Responses')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # MSE vs order
    mses = []
    for M in range(1, 25):
        _, e, _ = lms_filter(x, d, M, mu=0.01)
        mses.append(np.mean(e[2000:]**2))

    axes[1].semilogy(range(1, 25), mses, 'bo-')
    axes[1].axvline(M_true, color='r', linestyle='--', label=f'True order={M_true}')
    axes[1].set_xlabel('Filter Order M')
    axes[1].set_ylabel('Steady-State MSE')
    axes[1].set_title('MSE vs Filter Order')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex13_filter_order.png', dpi=100)
    plt.close()

    # (b) Explanation
    print(f"\n(b) Under-modeling (M < {M_true}):")
    print(f"    Cannot capture full impulse response -> higher MSE")
    print(f"    Over-modeling (M > {M_true}):")
    print(f"    Extra coefficients converge to ~0 but add noise (excess MSE)")
    print(f"    Optimal at M = {M_true} (matches true system order)")
    print("    Plot saved: ex13_filter_order.png")


if __name__ == "__main__":
    print("=== Exercise 1: Wiener Filter ===")
    exercise_1()
    print("\n=== Exercise 2: LMS Convergence ===")
    exercise_2()
    print("\n=== Exercise 3: NLMS vs LMS ===")
    exercise_3()
    print("\n=== Exercise 4: RLS Implementation ===")
    exercise_4()
    print("\n=== Exercise 5: Echo Cancellation ===")
    exercise_5()
    print("\n=== Exercise 6: Adaptive Equalization ===")
    exercise_6()
    print("\n=== Exercise 7: Effect of Filter Order ===")
    exercise_7()
    print("\nAll exercises completed!")
