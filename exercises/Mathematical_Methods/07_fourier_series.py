"""
Exercise Solutions: Lesson 07 - Fourier Series
Mathematical Methods for Physical Sciences

Covers: Fourier coefficients, Parseval's theorem, half-range expansion,
        Gibbs phenomenon, vibrating string, heat conduction
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1_fourier_coefficients():
    """
    Problem 1: Find the Fourier series of f(x) = x^2 on [-pi, pi].
    """
    print("=" * 60)
    print("Problem 1: Fourier Series of x^2")
    print("=" * 60)

    # f(x) = x^2 is an even function => b_n = 0
    # a_0 = (1/pi) * integral_{-pi}^{pi} x^2 dx = (2/pi) * pi^3/3 = 2*pi^2/3
    # a_n = (2/pi) * integral_0^{pi} x^2 * cos(nx) dx
    #     = (2/pi) * [x^2*sin(nx)/n + 2x*cos(nx)/n^2 - 2*sin(nx)/n^3]_0^{pi}
    #     = (2/pi) * [2*pi*cos(n*pi)/n^2]
    #     = 4*(-1)^n / n^2

    a_0 = 2 * np.pi**2 / 3
    print(f"\nf(x) = x^2 on [-pi, pi], even function => b_n = 0")
    print(f"\na_0/2 = pi^2/3 = {np.pi**2/3:.8f}")

    N_terms = 10
    a_n = np.array([4 * (-1)**n / n**2 for n in range(1, N_terms + 1)])

    print(f"\na_n = 4*(-1)^n / n^2:")
    for n in range(1, 6):
        print(f"  a_{n} = {4*(-1)**n / n**2:+.8f}")

    print(f"\nFourier series: x^2 = pi^2/3 + sum_{{n=1}}^inf 4*(-1)^n/n^2 * cos(nx)")

    # Plot
    x = np.linspace(-np.pi, np.pi, 500)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, x**2, 'k-', linewidth=2, label='$x^2$')

    for N in [1, 3, 5, 10]:
        y = np.full_like(x, np.pi**2 / 3)
        for n in range(1, N + 1):
            y += 4 * (-1)**n / n**2 * np.cos(n * x)
        ax.plot(x, y, '--', linewidth=1.5, label=f'N = {N}')

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Fourier series of $x^2$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex07_fourier_x2.png', dpi=150)
    plt.close()
    print("Plot saved to ex07_fourier_x2.png")


def exercise_2_parseval():
    """
    Problem 2: Use Parseval's theorem with f(x) = x^2 to find sum 1/n^4.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Parseval's Theorem -> sum 1/n^4")
    print("=" * 60)

    # Parseval: (1/pi) * integral_{-pi}^{pi} |f(x)|^2 dx = a_0^2/2 + sum (a_n^2 + b_n^2)
    # Note: using the convention a_0/2 for the constant term

    # LHS: (1/pi) * int x^4 dx = (2/pi) * pi^5/5 = 2*pi^4/5
    lhs = 2 * np.pi**4 / 5

    # RHS: (a_0/2)^2/? Let's use standard Parseval:
    # (1/L) int |f|^2 dx = (a_0/2)^2 + (1/2)*sum(a_n^2 + b_n^2)
    # Here L = pi (half-period)
    # (1/pi) * int_{-pi}^{pi} x^4 dx = 2*pi^4/5

    # a_0/2 = pi^2/3, (a_0/2)^2 = pi^4/9
    # a_n = 4*(-1)^n/n^2, a_n^2 = 16/n^4

    # Parseval: 2*pi^4/5 = pi^4/9 + (1/2)*sum 16/n^4
    # 2*pi^4/5 - pi^4/9 = 8 * sum 1/n^4
    # pi^4*(18-5)/45 = 8 * sum 1/n^4
    # 13*pi^4/45 = 8 * sum 1/n^4

    # Wait, let me re-derive more carefully.
    # Standard Parseval's theorem (period 2L = 2*pi):
    # (1/2L) * int_{-L}^{L} |f|^2 dx = (a_0/2)^2 + (1/2)*sum(a_n^2 + b_n^2)

    # Actually, the standard form is:
    # (1/pi) int_{-pi}^{pi} [f(x)]^2 dx = (a_0)^2/2 + sum(a_n^2 + b_n^2)
    # where f = a_0/2 + sum(a_n cos(nx) + b_n sin(nx))
    # and a_0 = (1/pi) int f dx = 2*pi^2/3

    # Let's use the correct form:
    # (2/pi) int_0^{pi} x^4 dx = a_0^2/2 + sum a_n^2
    # 2*pi^4/5 = (2*pi^2/3)^2/2 + sum (4*(-1)^n/n^2)^2
    # 2*pi^4/5 = 4*pi^4/18 + 16*sum(1/n^4)
    # 2*pi^4/5 = 2*pi^4/9 + 16*sum(1/n^4)
    # 16*sum(1/n^4) = 2*pi^4/5 - 2*pi^4/9 = 2*pi^4*(9-5)/45 = 8*pi^4/45
    # sum(1/n^4) = pi^4/90

    print(f"\nUsing Fourier series of x^2:")
    print(f"  a_0 = 2*pi^2/3")
    print(f"  a_n = 4*(-1)^n/n^2")
    print(f"\nParseval's theorem:")
    print(f"  (2/pi)*int_0^pi x^4 dx = a_0^2/2 + sum a_n^2")
    print(f"  2*pi^4/5 = (2*pi^2/3)^2/2 + 16*sum(1/n^4)")
    print(f"  2*pi^4/5 = 2*pi^4/9 + 16*sum(1/n^4)")
    print(f"  16*sum(1/n^4) = 2*pi^4*(1/5 - 1/9) = 8*pi^4/45")
    print(f"  sum(1/n^4) = pi^4/90")

    exact = np.pi**4 / 90
    numerical = sum(1.0 / n**4 for n in range(1, 100001))
    print(f"\n  pi^4/90 = {exact:.10f}")
    print(f"  Numerical (100000 terms) = {numerical:.10f}")
    print(f"  Relative error = {abs(numerical - exact)/exact:.2e}")


def exercise_3_half_range():
    """
    Problem 3: Find the half-range sine expansion of cos(x) on [0, pi].
    """
    print("\n" + "=" * 60)
    print("Problem 3: Half-Range Sine Expansion of cos(x)")
    print("=" * 60)

    # b_n = (2/pi) * int_0^pi cos(x) * sin(nx) dx
    # Using product-to-sum: cos(x)*sin(nx) = [sin((n+1)x) + sin((n-1)x)] / 2
    # For n != 1:
    #   b_n = (1/pi) * [-cos((n+1)x)/(n+1) - cos((n-1)x)/(n-1)]_0^pi
    #       = (1/pi) * {[-cos((n+1)pi)/(n+1) - cos((n-1)pi)/(n-1)] - [-1/(n+1) - 1/(n-1)]}
    #       = (1/pi) * {[-(-1)^{n+1}/(n+1) - (-1)^{n-1}/(n-1)] + [1/(n+1) + 1/(n-1)]}
    # For n even: (-1)^{n+1} = -1, (-1)^{n-1} = -1
    #   b_n = (1/pi) * {[1/(n+1) + 1/(n-1)] + [1/(n+1) + 1/(n-1)]}
    #       = (2/pi) * [1/(n+1) + 1/(n-1)]
    #       = (2/pi) * 2n/(n^2-1)
    #       = 4n / (pi*(n^2-1))
    # For n odd (n != 1): b_n = 0
    # For n = 1: b_1 = (2/pi) * int_0^pi cos(x)*sin(x) dx = (1/pi)*int_0^pi sin(2x) dx = 0

    print(f"\nb_n = (2/pi) * int_0^pi cos(x)*sin(nx) dx")

    N = 20
    b_n = np.zeros(N + 1)
    x_int = np.linspace(0, np.pi, 10000)

    for n in range(1, N + 1):
        b_n[n] = (2 / np.pi) * np.trapz(np.cos(x_int) * np.sin(n * x_int), x_int)

    print(f"\nCoefficients:")
    for n in range(1, 11):
        exact_val = 4 * n / (np.pi * (n**2 - 1)) if n % 2 == 0 else 0
        print(f"  b_{n:2d} = {b_n[n]:+.8f} (formula: {exact_val:+.8f})")

    print(f"\n  Pattern: b_n = 0 for odd n, b_n = 4n/(pi*(n^2-1)) for even n")

    # Plot
    x = np.linspace(0, np.pi, 500)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, np.cos(x), 'k-', linewidth=2, label='cos(x)')

    for N_sum in [2, 6, 10, 20]:
        y = np.zeros_like(x)
        for n in range(1, N_sum + 1):
            y += b_n[n] * np.sin(n * x)
        ax.plot(x, y, '--', linewidth=1.5, label=f'N = {N_sum}')

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Half-range sine expansion of cos(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex07_half_range.png', dpi=150)
    plt.close()
    print("Plot saved to ex07_half_range.png")


def exercise_4_gibbs():
    """
    Problem 4: Demonstrate the Gibbs phenomenon for the square wave
    Fourier series.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Gibbs Phenomenon Analysis")
    print("=" * 60)

    # Square wave: f(x) = 1 for 0 < x < pi, -1 for -pi < x < 0
    # Fourier: f(x) = (4/pi) * sum_{n=1,3,5,...} sin(nx)/n

    print(f"\nSquare wave Fourier series:")
    print(f"  f(x) = (4/pi) * sum_{{n odd}} sin(nx)/n")

    x = np.linspace(-np.pi, np.pi, 10000)

    # Compute overshoot for different N
    N_values = [5, 11, 21, 51, 101, 501]
    print(f"\n{'N terms':>8} | {'Max value':>10} | {'Overshoot %':>12}")
    print("-" * 38)

    for N in N_values:
        y = np.zeros_like(x)
        for n in range(1, N + 1, 2):  # odd n only
            y += (4 / np.pi) * np.sin(n * x) / n
        max_val = np.max(y)
        overshoot = (max_val - 1.0) * 100
        print(f"  {N:6d} | {max_val:10.6f} | {overshoot:10.4f}%")

    # Gibbs constant: the overshoot converges to (2/pi)*Si(pi) - 1 ~ 8.9%
    from scipy.special import sici
    Si_pi, _ = sici(np.pi)
    gibbs_overshoot = (2 / np.pi) * Si_pi - 1
    print(f"\n  Gibbs constant: overshoot -> {gibbs_overshoot*100:.4f}%")
    print(f"  (2/pi)*Si(pi) - 1 = {gibbs_overshoot:.8f}")
    print(f"  The overshoot does NOT vanish as N -> infinity!")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for N in [5, 21, 101]:
        y = np.zeros_like(x)
        for n in range(1, N + 1, 2):
            y += (4 / np.pi) * np.sin(n * x) / n
        axes[0].plot(x, y, linewidth=1.5, label=f'N={N}')

    axes[0].axhline(1, color='k', linewidth=0.5, linestyle='--')
    axes[0].axhline(-1, color='k', linewidth=0.5, linestyle='--')
    axes[0].set_title('Square Wave Fourier Series')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zoom near discontinuity
    x_zoom = np.linspace(-0.5, 0.5, 5000)
    for N in [11, 51, 201]:
        y = np.zeros_like(x_zoom)
        for n in range(1, N + 1, 2):
            y += (4 / np.pi) * np.sin(n * x_zoom) / n
        axes[1].plot(x_zoom, y, linewidth=1.5, label=f'N={N}')

    axes[1].axhline(1, color='k', linewidth=0.5, linestyle='--')
    axes[1].axhline(-1, color='k', linewidth=0.5, linestyle='--')
    axes[1].axhline(1 + gibbs_overshoot, color='r', linewidth=0.5,
                    linestyle=':', label=f'Gibbs ({gibbs_overshoot*100:.1f}%)')
    axes[1].set_title('Gibbs Phenomenon (zoom near x=0)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex07_gibbs.png', dpi=150)
    plt.close()
    print("Plot saved to ex07_gibbs.png")


def exercise_5_vibrating_string():
    """
    Problem 5: Vibrating string problem.
    u_tt = c^2 * u_xx, u(0,t) = u(L,t) = 0
    u(x,0) = sin(pi*x/L) + 0.5*sin(3*pi*x/L), u_t(x,0) = 0
    """
    print("\n" + "=" * 60)
    print("Problem 5: Vibrating String")
    print("=" * 60)

    print(f"\nWave equation: u_tt = c^2 * u_xx")
    print(f"BC: u(0,t) = u(L,t) = 0")
    print(f"IC: u(x,0) = sin(pi*x/L) + 0.5*sin(3*pi*x/L), u_t(x,0) = 0")

    print(f"\nSolution by separation of variables:")
    print(f"  u(x,t) = sum B_n * sin(n*pi*x/L) * cos(n*pi*c*t/L)")
    print(f"\n  From IC: B_1 = 1, B_3 = 0.5, all other B_n = 0")
    print(f"\n  u(x,t) = sin(pi*x/L)*cos(pi*c*t/L)")
    print(f"         + 0.5*sin(3*pi*x/L)*cos(3*pi*c*t/L)")

    print(f"\n  Frequencies:")
    print(f"    omega_1 = pi*c/L (fundamental)")
    print(f"    omega_3 = 3*pi*c/L (3rd harmonic)")

    # Numerical visualization
    L = 1.0
    c = 1.0
    x = np.linspace(0, L, 200)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    times = [0, 0.1, 0.25, 0.5, 0.75, 1.0]

    for idx, t in enumerate(times):
        ax = axes[idx // 3, idx % 3]
        u = (np.sin(np.pi * x / L) * np.cos(np.pi * c * t / L) +
             0.5 * np.sin(3 * np.pi * x / L) * np.cos(3 * np.pi * c * t / L))
        ax.plot(x, u, 'b-', linewidth=2)
        ax.fill_between(x, u, alpha=0.2)
        ax.set_title(f't = {t:.2f}')
        ax.set_ylim(-1.6, 1.6)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')

    plt.suptitle('Vibrating String: Two-Mode Superposition', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex07_vibrating_string.png', dpi=150)
    plt.close()
    print("Plot saved to ex07_vibrating_string.png")


def exercise_6_heat_conduction():
    """
    Problem 6: Heat conduction in a rod.
    u_t = alpha^2 * u_xx, u(0,t) = u(L,t) = 0
    u(x,0) = 100*x/L * (1 - x/L) (parabolic initial profile)
    """
    print("\n" + "=" * 60)
    print("Problem 6: Heat Conduction Rod")
    print("=" * 60)

    print(f"\nHeat equation: u_t = alpha^2 * u_xx")
    print(f"BC: u(0,t) = u(L,t) = 0")
    print(f"IC: u(x,0) = 100*x/L*(1-x/L)")

    # Fourier sine series of initial condition
    # B_n = (2/L) * int_0^L 100*x/L*(1-x/L) * sin(n*pi*x/L) dx
    # With substitution xi = x/L:
    # B_n = 200 * int_0^1 xi*(1-xi)*sin(n*pi*xi) dxi

    print(f"\nSolution: u(x,t) = sum B_n * sin(n*pi*x/L) * exp(-n^2*pi^2*alpha^2*t/L^2)")
    print(f"\nFourier coefficients:")

    L = 1.0
    alpha = 0.01
    N = 10
    xi = np.linspace(0, 1, 10000)

    B_n = []
    for n in range(1, N + 1):
        integrand = 200 * xi * (1 - xi) * np.sin(n * np.pi * xi)
        bn = 2 * np.trapz(integrand, xi)  # factor of 2 for the (2/L) in sine series with L=1
        # Analytical: B_n = 800/(n^3*pi^3) for odd n, 0 for even n
        if n % 2 == 1:
            exact = 800 / (n**3 * np.pi**3)
        else:
            exact = 0
        B_n.append(bn)
        print(f"  B_{n:2d} = {bn:10.6f} (exact: {exact:10.6f})")

    # Time evolution plot
    x = np.linspace(0, L, 200)
    fig, ax = plt.subplots(figsize=(10, 6))

    times = [0, 0.5, 1, 2, 5, 10]
    for t in times:
        u = np.zeros_like(x)
        for n in range(1, N + 1):
            u += B_n[n - 1] * np.sin(n * np.pi * x / L) * \
                 np.exp(-n**2 * np.pi**2 * alpha**2 * t / L**2)
        ax.plot(x, u, linewidth=2, label=f't = {t}')

    ax.set_xlabel('x')
    ax.set_ylabel('Temperature u(x,t)')
    ax.set_title('Heat conduction: parabolic initial temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex07_heat_conduction.png', dpi=150)
    plt.close()
    print("\nPlot saved to ex07_heat_conduction.png")

    # Decay rates
    print(f"\nDecay time constants (tau_n = L^2/(n^2*pi^2*alpha^2)):")
    for n in range(1, 5):
        tau = L**2 / (n**2 * np.pi**2 * alpha**2)
        print(f"  n={n}: tau = {tau:.4f}")
    print(f"\n  Higher modes decay much faster (n^2 dependence)")


if __name__ == "__main__":
    exercise_1_fourier_coefficients()
    exercise_2_parseval()
    exercise_3_half_range()
    exercise_4_gibbs()
    exercise_5_vibrating_string()
    exercise_6_heat_conduction()
