"""
Exercise Solutions: Lesson 18 - Fourier Series and PDE
Calculus and Differential Equations

Topics covered:
- Fourier sine coefficients of x(1-x)
- Heat equation on [0, pi]
- Guitar string wave equation
- Laplace equation on unit square
- Gibbs phenomenon
"""

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Fourier Sine Coefficients
# ============================================================
def exercise_1():
    """
    Fourier sine series of f(x) = x(1-x) on [0,1].
    Show b_n = 0 for even n, closed form for odd n.
    """
    print("=" * 60)
    print("Problem 1: Fourier Sine Coefficients")
    print("=" * 60)

    x, n = sp.symbols('x n', positive=True)

    # b_n = 2/L * integral_0^L f(x)*sin(n*pi*x/L) dx, L=1
    # b_n = 2 * integral_0^1 x(1-x)*sin(n*pi*x) dx
    f = x*(1 - x)
    b_n = 2 * sp.integrate(f * sp.sin(n*sp.pi*x), (x, 0, 1))
    b_n_simplified = sp.simplify(b_n)

    print(f"\n  f(x) = x(1-x) on [0, 1]")
    print(f"  b_n = 2 * integral_0^1 x(1-x)*sin(n*pi*x) dx")
    print(f"      = {b_n_simplified}")

    # Compute for specific n
    print(f"\n  Specific values:")
    for n_val in range(1, 9):
        b_val = b_n_simplified.subs(n, n_val)
        b_float = float(b_val)
        print(f"    b_{n_val} = {b_val} = {b_float:.8f}")

    print(f"\n  Pattern: b_n = 0 for even n")
    print(f"  For odd n: b_n = 8/(n*pi)^3")
    print(f"  The coefficients decay as 1/n^3 (much faster than 1/n for square wave)")
    print(f"  This is because f(x) = x(1-x) is smooth and satisfies the BCs.")

    # Verification: compute 8/(n*pi)^3 for odd n
    print(f"\n  Verification: 8/(n*pi)^3 for odd n:")
    for n_val in [1, 3, 5, 7]:
        exact = 8 / (n_val*np.pi)**3
        computed = float(b_n_simplified.subs(n, n_val))
        print(f"    n={n_val}: formula = {exact:.8f}, integral = {computed:.8f}")


# ============================================================
# Problem 2: Heat Equation
# ============================================================
def exercise_2():
    """
    u_t = u_xx on [0, pi], u(0,t) = u(pi,t) = 0, u(x,0) = 100.
    Fourier series solution. Time for max temp to drop to 50.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Heat Equation on [0, pi]")
    print("=" * 60)

    x, t, n = sp.symbols('x t n', positive=True)

    # Solution: u(x,t) = sum b_n * sin(n*x) * exp(-n^2*t)
    # where b_n = (2/pi) * integral_0^pi 100*sin(nx) dx
    # = (200/pi) * [-cos(nx)/n]_0^pi = (200/(n*pi)) * (1 - cos(n*pi))
    # = (200/(n*pi)) * (1 - (-1)^n)
    # For odd n: b_n = 400/(n*pi)
    # For even n: b_n = 0

    print(f"\n  u_t = u_xx on [0, pi], u(0,t) = u(pi,t) = 0, u(x,0) = 100")
    print(f"\n  Solution: u(x,t) = sum b_n * sin(nx) * e^(-n^2*t)")
    print(f"  b_n = (2/pi) integral_0^pi 100*sin(nx) dx")
    print(f"      = 400/(n*pi) for odd n, 0 for even n")
    print(f"\n  u(x,t) = (400/pi) * sum_{{n=1,3,5,...}} (1/n)*sin(nx)*e^(-n^2*t)")

    # Time for max temp to drop to 50 (first term approximation)
    # Max at x = pi/2. First term: (400/pi)*sin(pi/2)*e^(-t) = (400/pi)*e^(-t)
    # Set equal to 50: (400/pi)*e^(-t) = 50 => e^(-t) = pi/8 => t = ln(8/pi)
    t_half = np.log(8/np.pi)
    print(f"\n  Maximum temperature at x = pi/2:")
    print(f"  First term: u(pi/2, t) ~ (400/pi)*e^(-t)")
    print(f"  Set = 50: (400/pi)*e^(-t) = 50")
    print(f"  e^(-t) = pi/8")
    print(f"  t = ln(8/pi) = {t_half:.6f}")
    print(f"  (Higher terms decay much faster, so first-term estimate is good)")

    # Plot
    x_vals = np.linspace(0, np.pi, 200)
    t_values = [0, 0.1, 0.5, 1.0, t_half, 2.0]

    fig, ax = plt.subplots(figsize=(10, 6))
    for tv in t_values:
        u_vals = np.zeros_like(x_vals)
        for k in range(1, 50, 2):  # odd n only
            u_vals += (400/(k*np.pi)) * np.sin(k*x_vals) * np.exp(-k**2*tv)
        label = f't = {tv:.2f}' if tv != t_half else f't = ln(8/pi) = {tv:.3f}'
        ax.plot(x_vals, u_vals, linewidth=2, label=label)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='T = 50')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x,t)', fontsize=12)
    ax.set_title('Heat Equation: u(x,0) = 100', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex18_heat_equation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: ex18_heat_equation.png]")


# ============================================================
# Problem 3: Guitar String Wave Equation
# ============================================================
def exercise_3():
    """
    String L=0.65 m, plucked at L/4, c=300 m/s.
    Find first four nonzero Fourier terms and frequencies.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Guitar String Wave Equation")
    print("=" * 60)

    L = 0.65  # m
    c = 300   # m/s

    # Initial shape: triangular pluck at x = L/4
    # f(x) = 4x/L for 0 <= x <= L/4
    # f(x) = 4(L-x)/(3L) for L/4 < x <= L
    # Released from rest: g(x) = 0

    # Solution: u(x,t) = sum b_n * sin(n*pi*x/L) * cos(n*pi*c*t/L)
    # b_n = (2/L) integral_0^L f(x)*sin(n*pi*x/L) dx

    print(f"\n  L = {L} m, c = {c} m/s")
    print(f"  Plucked at x = L/4 with triangular shape")
    print(f"  f(x) = 4x/L for 0 <= x <= L/4")
    print(f"  f(x) = 4(L-x)/(3L) for L/4 < x <= L")
    print(f"\n  u(x,t) = sum b_n * sin(n*pi*x/L) * cos(n*pi*c*t/L)")

    x = sp.Symbol('x', positive=True)
    n = sp.Symbol('n', positive=True, integer=True)

    # Compute b_n symbolically
    L_sym = sp.Rational(65, 100)
    b_n_part1 = sp.integrate(4*x/L_sym * sp.sin(n*sp.pi*x/L_sym), (x, 0, L_sym/4))
    b_n_part2 = sp.integrate(4*(L_sym-x)/(3*L_sym) * sp.sin(n*sp.pi*x/L_sym), (x, L_sym/4, L_sym))
    b_n_total = (2/L_sym) * (b_n_part1 + b_n_part2)
    b_n_simplified = sp.simplify(b_n_total)

    print(f"\n  Fourier coefficients:")
    coeffs = []
    for n_val in range(1, 20):
        b_val = float(b_n_simplified.subs(n, n_val))
        if abs(b_val) > 1e-10:
            freq = n_val * c / (2*L)
            coeffs.append((n_val, b_val, freq))
            if len(coeffs) <= 4:
                print(f"    n={n_val}: b_{n_val} = {b_val:>10.6f}, f_{n_val} = {freq:.2f} Hz")

    print(f"\n  First four nonzero terms:")
    for i, (nv, bv, fv) in enumerate(coeffs[:4]):
        omega = 2*np.pi*fv
        print(f"    b_{nv}*sin({nv}*pi*x/L)*cos({omega:.2f}*t)")
        print(f"      frequency = {fv:.2f} Hz" +
              (f" (fundamental)" if i == 0 else f" ({nv}th harmonic)"))

    # Musical note
    f1 = c / (2*L)
    print(f"\n  Fundamental frequency f_1 = c/(2L) = {f1:.2f} Hz")
    print(f"  (close to E4 = 329.6 Hz)")


# ============================================================
# Problem 4: Laplace Equation
# ============================================================
def exercise_4():
    """
    u_xx + u_yy = 0 on [0,1]x[0,1].
    u(x,0)=0, u(x,1)=sin(pi*x), u(0,y)=u(1,y)=0.
    Verify mean value property.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Laplace Equation on Unit Square")
    print("=" * 60)

    # Separation of variables: u(x,y) = X(x)*Y(y)
    # X'' + lambda*X = 0, X(0)=X(1)=0 => X_n = sin(n*pi*x), lambda_n = (n*pi)^2
    # Y'' - (n*pi)^2*Y = 0 => Y_n = A*sinh(n*pi*y) + B*cosh(n*pi*y)
    # BC: u(x,0)=0 => Y_n(0)=0 => B=0, so Y_n = sinh(n*pi*y)
    # BC: u(x,1)=sin(pi*x) => only n=1 term contributes
    # A_1*sinh(pi) = 1 => A_1 = 1/sinh(pi)
    # u(x,y) = sin(pi*x) * sinh(pi*y) / sinh(pi)

    print(f"\n  Separation of variables:")
    print(f"  u(x,y) = sin(pi*x) * sinh(pi*y) / sinh(pi)")

    # Value at center
    u_center = np.sin(np.pi*0.5) * np.sinh(np.pi*0.5) / np.sinh(np.pi)
    print(f"\n  u(1/2, 1/2) = sin(pi/2)*sinh(pi/2)/sinh(pi)")
    print(f"              = 1 * {np.sinh(np.pi/2):.6f} / {np.sinh(np.pi):.6f}")
    print(f"              = {u_center:.10f}")

    # Mean value property: average of u over a circle centered at (1/2, 1/2)
    # Use a small radius (say r = 0.1)
    r = 0.1
    N_theta = 1000
    theta_vals = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
    x_circle = 0.5 + r*np.cos(theta_vals)
    y_circle = 0.5 + r*np.sin(theta_vals)

    u_circle = np.sin(np.pi*x_circle) * np.sinh(np.pi*y_circle) / np.sinh(np.pi)
    u_avg = np.mean(u_circle)

    print(f"\n  Mean value property verification:")
    print(f"  Circle radius r = {r}, centered at (1/2, 1/2)")
    print(f"  Average of u on circle = {u_avg:.10f}")
    print(f"  u at center            = {u_center:.10f}")
    print(f"  Difference             = {abs(u_avg - u_center):.6e}")
    print(f"  Mean value property SATISFIED (harmonic functions = their circle average)")

    # Plot
    Nx, Ny = 100, 100
    x_grid = np.linspace(0, 1, Nx)
    y_grid = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x_grid, y_grid)
    U = np.sin(np.pi*X) * np.sinh(np.pi*Y) / np.sinh(np.pi)

    fig, ax = plt.subplots(figsize=(8, 7))
    c = ax.contourf(X, Y, U, levels=30, cmap='hot')
    plt.colorbar(c, ax=ax, label='u(x,y)')
    ax.plot(0.5, 0.5, 'w*', markersize=15, label=f'Center: u = {u_center:.4f}')
    theta_plot = np.linspace(0, 2*np.pi, 100)
    ax.plot(0.5 + r*np.cos(theta_plot), 0.5 + r*np.sin(theta_plot), 'w--', linewidth=1.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Laplace Equation: $u(x,y) = \\sin(\\pi x)\\sinh(\\pi y)/\\sinh(\\pi)$', fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('ex18_laplace_equation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex18_laplace_equation.png]")


# ============================================================
# Problem 5: Gibbs Phenomenon
# ============================================================
def exercise_5():
    """
    Fourier sine series of f(x)=1 on [0,pi] for N=10, 50, 200.
    Measure overshoot near boundaries. Verify ~9% convergence.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Gibbs Phenomenon")
    print("=" * 60)

    # f(x) = 1 on [0, pi]
    # Fourier sine series: b_n = (2/pi) integral_0^pi sin(nx) dx
    # = (2/pi) * [-cos(nx)/n]_0^pi = (2/(n*pi))*(1 - cos(n*pi))
    # = (2/(n*pi))*(1 - (-1)^n)
    # Odd n: b_n = 4/(n*pi); Even n: b_n = 0

    print(f"\n  f(x) = 1 on [0, pi]")
    print(f"  Fourier sine series: b_n = 4/(n*pi) for odd n, 0 for even n")
    print(f"  S_N(x) = (4/pi) * sum_{{n=1,3,5,...}}^N (1/n)*sin(nx)")

    x = np.linspace(0, np.pi, 10000)
    N_values = [10, 50, 200]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    print(f"\n  Overshoot analysis:")
    print(f"  {'N':>6s}  {'Max value':>12s}  {'Overshoot %':>14s}")
    print(f"  {'------':>6s}  {'------------':>12s}  {'--------------':>14s}")

    for idx, N in enumerate(N_values):
        S_N = np.zeros_like(x)
        for n in range(1, N + 1, 2):  # odd n
            S_N += (4/(n*np.pi)) * np.sin(n*x)

        max_val = np.max(S_N)
        overshoot = (max_val - 1.0) / 1.0 * 100
        print(f"  {N:>6d}  {max_val:>12.8f}  {overshoot:>14.6f}%")

        axes[idx].plot(x, S_N, 'b-', linewidth=1)
        axes[idx].axhline(y=1, color='r', linestyle='--', alpha=0.7)
        axes[idx].axhline(y=max_val, color='g', linestyle=':', alpha=0.7,
                          label=f'Max = {max_val:.4f}')
        axes[idx].set_title(f'N = {N}', fontsize=12)
        axes[idx].set_xlabel('x', fontsize=10)
        axes[idx].set_ylabel('$S_N(x)$', fontsize=10)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(-0.2, 1.3)

    plt.suptitle('Gibbs Phenomenon: Fourier Sine Series of f(x)=1', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex18_gibbs_phenomenon.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Theoretical Gibbs overshoot
    gibbs_factor = 2/np.pi * (np.pi/2 + sum(np.sin(k)/k for k in range(1, 10000)))
    # The exact Gibbs constant: integral_0^pi sin(t)/t dt / pi ~ 1.0895
    # Overshoot ~ 8.95%
    from scipy.integrate import quad
    Si_pi, _ = quad(lambda t: np.sin(t)/t, 0, np.pi)
    gibbs_exact = 2*Si_pi/np.pi
    gibbs_overshoot = (gibbs_exact - 1) * 100

    print(f"\n  Theoretical Gibbs overshoot:")
    print(f"  (2/pi)*Si(pi) = {gibbs_exact:.8f}")
    print(f"  Overshoot = {gibbs_overshoot:.4f}%")
    print(f"  The overshoot converges to ~{gibbs_overshoot:.1f}% regardless of N")
    print(f"  (it does not vanish -- the peak moves closer to the boundary)")
    print("\n  [Plot saved: ex18_gibbs_phenomenon.png]")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("\n" + "=" * 60)
    print("All exercises for Lesson 18 completed.")
    print("=" * 60)
