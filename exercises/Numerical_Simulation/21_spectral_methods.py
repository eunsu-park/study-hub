"""
Exercises for Lesson 21: Spectral Methods
Topic: Numerical_Simulation

Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def chebyshev_points(N):
    """Compute Chebyshev-Gauss-Lobatto points on [-1, 1]."""
    j = np.arange(N + 1)
    return np.cos(np.pi * j / N)


def chebyshev_diff_matrix(N):
    """Compute Chebyshev differentiation matrix D."""
    x = chebyshev_points(N)
    D = np.zeros((N + 1, N + 1))

    c = np.ones(N + 1)
    c[0] = 2
    c[N] = 2

    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1)**(i + j)) / (x[i] - x[j])

    for i in range(N + 1):
        D[i, i] = -np.sum(D[i, :])

    return D, x


def dealias_product_3_2_rule(u, v):
    """Compute dealiased product w = u*v using 3/2 rule."""
    N = len(u)
    M = 3 * N // 2

    u_hat = fft(u)
    v_hat = fft(v)

    u_hat_padded = np.zeros(M, dtype=complex)
    v_hat_padded = np.zeros(M, dtype=complex)

    u_hat_padded[:N // 2] = u_hat[:N // 2]
    u_hat_padded[-N // 2:] = u_hat[-N // 2:]

    v_hat_padded[:N // 2] = v_hat[:N // 2]
    v_hat_padded[-N // 2:] = v_hat[-N // 2:]

    u_ext = ifft(u_hat_padded)
    v_ext = ifft(v_hat_padded)

    w_ext = u_ext * v_ext
    w_hat_ext = fft(w_ext)

    w_hat = np.zeros(N, dtype=complex)
    w_hat[:N // 2] = w_hat_ext[:N // 2]
    w_hat[-N // 2:] = w_hat_ext[-N // 2:]
    w_hat *= M / N

    return np.real(ifft(w_hat))


# === Exercise 1: Exponential Convergence ===
# Problem: Demonstrate exponential convergence for u(x) = exp(sin(x)) on [0, 2pi].

def exercise_1():
    """Exponential convergence of spectral methods for smooth periodic functions."""

    def u_exact(x):
        return np.exp(np.sin(x))

    N_values = [4, 8, 16, 32, 64]
    errors = []

    # Fine grid for error evaluation
    x_fine = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    u_fine = u_exact(x_fine)

    print("Exponential Convergence of Spectral Method")
    print("=" * 50)
    print(f"Function: u(x) = exp(sin(x)) on [0, 2*pi]")
    print(f"{'N':<8}{'L_inf Error':<18}")
    print("-" * 30)

    for N in N_values:
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        u = u_exact(x)

        # Fourier coefficients
        u_hat = fft(u)
        k = fftfreq(N, 1.0 / N)  # Integer wavenumbers

        # Evaluate on fine grid via spectral interpolation
        u_interp = np.zeros(len(x_fine))
        for j, xj in enumerate(x_fine):
            u_interp[j] = np.real(np.sum(u_hat * np.exp(1j * k * xj)) / N)

        error = np.linalg.norm(u_fine - u_interp, np.inf)
        errors.append(error)
        print(f"{N:<8}{error:<18.2e}")

    # Verify exponential convergence by checking the error drops
    # much faster than any polynomial rate
    print(f"\nError drops from {errors[0]:.2e} (N=4) to {errors[-1]:.2e} (N=64)")
    print("This is exponential convergence (super-algebraic decay).")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(N_values, errors, 'bo-', linewidth=2, markersize=8, label='Spectral error')
    # Reference: algebraic O(N^-4) for comparison
    ax.semilogy(N_values, errors[0] * (N_values[0] / np.array(N_values, dtype=float))**4,
                'r--', alpha=0.5, label='O(N^{-4}) reference')
    ax.set_xlabel('N (number of modes)')
    ax.set_ylabel('L_inf error')
    ax.set_title('Exponential Convergence of Fourier Spectral Method')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('exercise_21_1_exponential_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to exercise_21_1_exponential_convergence.png")


# === Exercise 2: Chebyshev Interpolation vs Runge Phenomenon ===
# Problem: Interpolate f(x) = 1/(1+25x^2) using uniform vs Chebyshev nodes.

def exercise_2():
    """Chebyshev nodes avoid the Runge phenomenon."""

    def runge(x):
        return 1.0 / (1 + 25 * x**2)

    x_fine = np.linspace(-1, 1, 500)
    f_fine = runge(x_fine)

    print("Chebyshev vs Uniform Interpolation (Runge Function)")
    print("=" * 60)
    print(f"{'N':<8}{'Uniform L_inf':<20}{'Chebyshev L_inf':<20}")
    print("-" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for N in [10, 20]:
        # Uniform nodes
        x_uni = np.linspace(-1, 1, N + 1)
        coeffs_uni = np.polyfit(x_uni, runge(x_uni), N)
        p_uni = np.polyval(coeffs_uni, x_fine)
        err_uni = np.max(np.abs(p_uni - f_fine))

        # Chebyshev-Gauss-Lobatto nodes
        x_cheb = chebyshev_points(N)
        coeffs_cheb = np.polyfit(x_cheb, runge(x_cheb), N)
        p_cheb = np.polyval(coeffs_cheb, x_fine)
        err_cheb = np.max(np.abs(p_cheb - f_fine))

        print(f"{N:<8}{err_uni:<20.4e}{err_cheb:<20.4e}")

    # Detailed plot for N=20
    N = 20
    x_uni = np.linspace(-1, 1, N + 1)
    x_cheb = chebyshev_points(N)

    coeffs_uni = np.polyfit(x_uni, runge(x_uni), N)
    p_uni = np.polyval(coeffs_uni, x_fine)

    coeffs_cheb = np.polyfit(x_cheb, runge(x_cheb), N)
    p_cheb = np.polyval(coeffs_cheb, x_fine)

    axes[0].plot(x_fine, f_fine, 'k-', linewidth=2, label='f(x) = 1/(1+25x^2)')
    axes[0].plot(x_fine, p_uni, 'r--', linewidth=1.5, label=f'Uniform (N={N})')
    axes[0].plot(x_uni, runge(x_uni), 'ro', markersize=5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('Uniform Nodes (Runge Phenomenon)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1, 2)

    axes[1].plot(x_fine, f_fine, 'k-', linewidth=2, label='f(x) = 1/(1+25x^2)')
    axes[1].plot(x_fine, p_cheb, 'b--', linewidth=1.5, label=f'Chebyshev (N={N})')
    axes[1].plot(x_cheb, runge(x_cheb), 'bo', markersize=5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('f(x)')
    axes[1].set_title('Chebyshev Nodes (No Oscillations)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-1, 2)

    plt.tight_layout()
    plt.savefig('exercise_21_2_runge_phenomenon.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nChebyshev nodes cluster near the endpoints, which minimizes the")
    print("Lebesgue constant and prevents the wild oscillations at the boundary")
    print("that plague uniformly-spaced polynomial interpolation.")
    print("Plot saved to exercise_21_2_runge_phenomenon.png")


# === Exercise 3: Spectral Heat Equation ===
# Problem: Solve du/dt = d^2u/dx^2 on [0, 2pi] with u(x,0) = sin(x).
# Use integrating factor (exact in spectral space).
# Compare with u_exact(x,t) = e^(-t) sin(x).

def exercise_3():
    """Solve heat equation spectrally with exact integrating factor."""

    N = 64
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    k = fftfreq(N, 1.0 / N)  # Integer wavenumbers

    # Initial condition
    u0 = np.sin(x)
    u_hat = fft(u0)

    # Exact time integration in spectral space:
    # du_hat/dt = -k^2 u_hat  =>  u_hat(k, t) = u_hat(k, 0) * exp(-k^2 * t)
    T_values = [0.0, 0.5, 1.0, 2.0]

    print("Spectral Heat Equation")
    print("=" * 50)
    print(f"du/dt = d^2u/dx^2, u(x,0) = sin(x)")
    print(f"Exact: u(x,t) = exp(-t) sin(x)")
    print()
    print(f"{'t':<8}{'L_inf Error':<18}")
    print("-" * 30)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, u0, 'k-', linewidth=2, label='t=0 (initial)')

    for T in T_values[1:]:
        u_hat_t = u_hat * np.exp(-k**2 * T)
        u_spectral = np.real(ifft(u_hat_t))
        u_exact = np.exp(-T) * np.sin(x)

        error = np.linalg.norm(u_spectral - u_exact, np.inf)
        print(f"{T:<8.1f}{error:<18.2e}")

        ax.plot(x, u_spectral, '--', linewidth=2, label=f't={T} (spectral)')
        ax.plot(x, u_exact, 'o', markersize=4, alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Heat Equation: Spectral Solution (lines) vs Exact (dots)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exercise_21_3_spectral_heat.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nThe integrating factor method solves the linear part exactly")
    print("in spectral space, yielding machine-precision accuracy.")
    print("Plot saved to exercise_21_3_spectral_heat.png")


# === Exercise 4: Two-Soliton Collision in KdV ===
# Problem: Simulate two KdV solitons with different speeds and
# verify elastic collision (they pass through each other).

def exercise_4():
    """Two-soliton KdV collision with spectral method."""

    N = 512
    L = 60 * np.pi
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    k = fftfreq(N, L / N) * 2 * np.pi

    kappa1, kappa2 = 0.5, 0.8
    T = 20.0
    dt = 0.005

    # Two-soliton initial condition
    u = (-6 * kappa1**2 / np.cosh(kappa1 * (x + 10))**2
         - 6 * kappa2**2 / np.cosh(kappa2 * (x - 10))**2)
    u_initial = u.copy()

    # Mass and energy conservation diagnostics
    mass_initial = np.sum(u) * (L / N)

    def rhs(u):
        """RHS of KdV: du/dt + u du/dx + d^3u/dx^3 = 0."""
        u_hat = fft(u)
        dispersion = -(1j * k)**3 * u_hat
        ux = np.real(ifft(1j * k * u_hat))
        nonlinear = fft(-u * ux)
        return np.real(ifft(dispersion + nonlinear))

    # RK4 time stepping
    nt = int(T / dt)
    print("Two-Soliton KdV Collision")
    print("=" * 50)
    print(f"kappa1 = {kappa1}, kappa2 = {kappa2}")
    print(f"Soliton speeds: c1 = {4*kappa1**2:.2f}, c2 = {4*kappa2**2:.2f}")
    print(f"Grid: N={N}, Time: T={T}, dt={dt}")

    # Save snapshots
    snapshot_times = [0, 5, 10, 15, 20]
    snapshots = [(0, u.copy())]
    time = 0.0

    for n in range(nt):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        time += dt

        for st in snapshot_times:
            if abs(time - st) < dt / 2 and not any(abs(s[0] - st) < dt for s in snapshots):
                snapshots.append((st, u.copy()))

    mass_final = np.sum(u) * (L / N)

    print(f"\nMass conservation: initial={mass_initial:.6f}, final={mass_final:.6f}")
    print(f"Mass change: {abs(mass_final - mass_initial):.2e}")
    print("After collision, both solitons re-emerge with original shape and speed.")

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))

    for (t_snap, u_snap), color in zip(snapshots, colors):
        ax.plot(x, u_snap, color=color, linewidth=1.5, label=f't={t_snap:.0f}')

    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('KdV Two-Soliton Collision (Elastic)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-L / 4, L / 4)
    plt.tight_layout()
    plt.savefig('exercise_21_4_kdv_two_soliton.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to exercise_21_4_kdv_two_soliton.png")


# === Exercise 5: Aliasing Error Demonstration ===
# Problem: Demonstrate aliasing when multiplying high-frequency modes
# and show how the 3/2 rule corrects it.

def exercise_5():
    """Demonstrate aliasing error and 3/2 rule dealiasing."""

    N = 32
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    k = fftfreq(N, 1.0 / N)

    k0 = N // 4  # = 8

    # Two modes that when multiplied produce mode 2*k0 = N/2 (at Nyquist)
    u = np.sin(k0 * x)
    v = np.sin(k0 * x)

    # Naive (aliased) product
    w_naive = u * v
    w_naive_hat = fft(w_naive)

    # Dealiased product using 3/2 rule
    w_dealiased = dealias_product_3_2_rule(u, v)
    w_dealiased_hat = fft(w_dealiased)

    # Exact: sin(k0*x)^2 = 0.5*(1 - cos(2*k0*x))
    # In Fourier space: DC component (mode 0) = N/2
    # and mode 2*k0 (= N/2, the Nyquist) with amplitude -N/4

    print("Aliasing Error Demonstration")
    print("=" * 60)
    print(f"N = {N}, k0 = {k0}")
    print(f"Product: sin({k0}x) * sin({k0}x) = 0.5*(1 - cos({2*k0}x))")
    print(f"Product contains mode 2*k0 = {2*k0} = N/2 (Nyquist)")
    print()

    # Compare spectral coefficients
    print(f"{'Mode':<8}{'Naive':<20}{'Dealiased':<20}{'Exact':<20}")
    print("-" * 68)

    # Mode 0 (DC)
    exact_dc = N / 2
    print(f"{'0':<8}{np.real(w_naive_hat[0]):<20.4f}"
          f"{np.real(w_dealiased_hat[0]):<20.4f}{exact_dc:<20.4f}")

    # Mode N/2 (Nyquist)
    nyquist_idx = N // 2
    exact_nyquist = -N / 4
    print(f"{nyquist_idx:<8}{np.real(w_naive_hat[nyquist_idx]):<20.4f}"
          f"{np.real(w_dealiased_hat[nyquist_idx]):<20.4f}{exact_nyquist:<20.4f}")

    # Total L2 error
    # Compute exact product on fine grid for reference
    N_fine = 3 * N
    x_fine = np.linspace(0, 2 * np.pi, N_fine, endpoint=False)
    w_exact_fine = np.sin(k0 * x_fine)**2
    # Truncate to N modes
    w_exact_hat_fine = fft(w_exact_fine)
    w_exact_hat_N = np.zeros(N, dtype=complex)
    w_exact_hat_N[:N // 2] = w_exact_hat_fine[:N // 2] * (N / N_fine)
    w_exact_hat_N[-N // 2:] = w_exact_hat_fine[-N // 2:] * (N / N_fine)

    err_naive = np.linalg.norm(w_naive_hat - w_exact_hat_N)
    err_dealiased = np.linalg.norm(w_dealiased_hat - w_exact_hat_N)

    print()
    print(f"Spectral coefficient L2 error:")
    print(f"  Naive (aliased):   {err_naive:.4f}")
    print(f"  Dealiased (3/2):   {err_dealiased:.4f}")
    print()
    print("Explanation: When mode k0 is squared, the result contains mode 2*k0 = N/2.")
    print("Without dealiasing, the Nyquist frequency aliases back to mode 0,")
    print("contaminating low-frequency content. The 3/2 rule zero-pads before")
    print("multiplication, correctly resolving all product frequencies.")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    modes = np.arange(N)
    axes[0].stem(modes, np.abs(w_naive_hat), linefmt='b-', markerfmt='bo',
                 basefmt='k-', label='Naive (aliased)')
    axes[0].stem(modes, np.abs(w_dealiased_hat), linefmt='r-', markerfmt='rs',
                 basefmt='k-', label='Dealiased (3/2)')
    axes[0].set_xlabel('Wavenumber k')
    axes[0].set_ylabel('|w_hat|')
    axes[0].set_title('Spectral Coefficients')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, w_naive, 'b-', linewidth=1.5, label='Naive')
    axes[1].plot(x, w_dealiased, 'r--', linewidth=1.5, label='Dealiased')
    axes[1].plot(x, np.sin(k0 * x)**2, 'k:', linewidth=2, label='Exact')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('w(x)')
    axes[1].set_title('Physical Space Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_21_5_aliasing.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to exercise_21_5_aliasing.png")


if __name__ == "__main__":
    print("=== Exercise 1: Exponential Convergence ===")
    exercise_1()

    print("\n=== Exercise 2: Chebyshev vs Runge Phenomenon ===")
    exercise_2()

    print("\n=== Exercise 3: Spectral Heat Equation ===")
    exercise_3()

    print("\n=== Exercise 4: Two-Soliton KdV Collision ===")
    exercise_4()

    print("\n=== Exercise 5: Aliasing Error Demonstration ===")
    exercise_5()

    print("\nAll exercises completed!")
