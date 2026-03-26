"""
Exercises for Lesson 16: Green's Functions
Topic: Mathematical_Methods
Solutions to practice problems from the lesson.
"""

import numpy as np
import sympy as sp
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: Calculate Dirac delta function integrals.
    (a) int (x^3 + 2x + 1) delta(x-2) dx
    (b) int_0^5 e^{-x} delta(x-3) dx
    (c) int cos(x) delta'(x) dx
    """
    print("=" * 60)
    print("Problem 1: Dirac Delta Function Integrals")
    print("=" * 60)

    # (a) Sifting property: int f(x) delta(x-a) dx = f(a)
    print("\n(a) int (x^3 + 2x + 1) delta(x - 2) dx")
    f_at_2 = 2**3 + 2 * 2 + 1  # = 8 + 4 + 1 = 13
    print(f"  By sifting property: f(2) = 2^3 + 2*2 + 1 = {f_at_2}")

    # Numerical verification using Gaussian approximation
    eps = 1e-4
    x = np.linspace(-10, 10, 1000000)
    f = x**3 + 2 * x + 1
    delta_approx = np.exp(-(x - 2)**2 / eps**2) / (eps * np.sqrt(np.pi))
    numerical = np.trapz(f * delta_approx, x)
    print(f"  Numerical (Gaussian approx): {numerical:.6f}")

    # (b) int_0^5 e^{-x} delta(x-3) dx
    print("\n(b) int_0^5 e^{-x} delta(x - 3) dx")
    f_at_3 = np.exp(-3)
    print(f"  Since 3 is in [0,5]: f(3) = e^{{-3}} = {f_at_3:.6f}")

    # (c) int cos(x) delta'(x) dx = -f'(0) = -(-sin(0)) = sin(0) = 0
    print("\n(c) int cos(x) delta'(x) dx")
    print("  Using integration by parts: int f(x) delta'(x) dx = -f'(0)")
    print("  f(x) = cos(x), f'(x) = -sin(x)")
    result_c = -(-np.sin(0))  # = sin(0) = 0
    print(f"  Result: -f'(0) = sin(0) = {result_c:.6f}")


def exercise_2():
    """
    Problem 2: Show delta(x^2-a^2) = (1/2|a|)[delta(x-a) + delta(x+a)]
    and compute int e^x delta(x^2-4) dx.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Delta Function Composition Identity")
    print("=" * 60)

    print("\nShow: delta(x^2 - a^2) = (1/(2|a|))[delta(x-a) + delta(x+a)]")
    print("\nProof: g(x) = x^2 - a^2 has zeros at x = +/- a")
    print("  g'(x) = 2x, so |g'(a)| = 2|a|, |g'(-a)| = 2|a|")
    print("  delta(g(x)) = sum delta(x-x_i)/|g'(x_i)|")
    print("             = delta(x-a)/(2|a|) + delta(x+a)/(2|a|)")
    print("             = [delta(x-a) + delta(x+a)]/(2|a|)  QED")

    # Compute the integral with a=2
    print("\nint e^x delta(x^2 - 4) dx  (a = 2)")
    a = 2
    result = (np.exp(a) + np.exp(-a)) / (2 * a)
    print(f"  = (1/(2*2))[e^2 + e^{{-2}}] = cosh(2)/2 = {result:.6f}")
    print(f"  Verification: cosh(2)/2 = {np.cosh(2) / 2:.6f}")


def exercise_3():
    """
    Problem 3: Construct Green's function for y''=f(x), y(0)=y(L)=0.
    Verify G(x,x') = G(x',x).
    """
    print("\n" + "=" * 60)
    print("Problem 3: Green's Function for y'' = f(x)")
    print("=" * 60)

    print("\ny'' = f(x), y(0) = 0, y(L) = 0")
    print("\nHomogeneous solutions: y1 = x (satisfies y(0)=0)")
    print("                       y2 = L - x (satisfies y(L)=0)")
    print("\nWronskian: W = y1*y2' - y1'*y2 = x*(-1) - 1*(L-x) = -L")
    print("\nGreen's function (for L = pi):")
    print("  G(x,x') = -y1(x<)*y2(x>)/W")
    print("          = x<(L-x>)/L")
    print("  where x< = min(x,x'), x> = max(x,x')")

    # Verify symmetry numerically
    L = np.pi
    x_test, xp_test = 1.0, 2.0

    def G(x, xp, L_val):
        x_less = np.minimum(x, xp)
        x_greater = np.maximum(x, xp)
        return x_less * (L_val - x_greater) / L_val

    G_12 = G(x_test, xp_test, L)
    G_21 = G(xp_test, x_test, L)
    print(f"\nSymmetry verification (L = pi):")
    print(f"  G({x_test:.1f}, {xp_test:.1f}) = {G_12:.6f}")
    print(f"  G({xp_test:.1f}, {x_test:.1f}) = {G_21:.6f}")
    print(f"  |G(x,x') - G(x',x)| = {abs(G_12 - G_21):.2e}")

    # Plot Green's function
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, L, 200)
    for xp in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ax.plot(x, G(x, xp, L), label=f"x' = {xp:.1f}")
    ax.set_xlabel('x')
    ax.set_ylabel("G(x, x')")
    ax.set_title("Green's Function for y'' = f(x), y(0)=y(L)=0")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex16_greens_function_bvp.png', dpi=150)
    plt.close()
    print("Plot saved to ex16_greens_function_bvp.png")


def exercise_4():
    """
    Problem 4: Green's function for y'' + y = f(x), y(0) = y(pi/2) = 0.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Green's Function for y'' + y = f(x)")
    print("=" * 60)

    print("\ny'' + y = f(x), y(0) = 0, y(pi/2) = 0")
    print("\nHomogeneous solutions of y'' + y = 0: sin(x) and cos(x)")
    print("y1 = sin(x) satisfies y(0) = 0")
    print("y2 = cos(x) satisfies y(pi/2) = 0")
    print("\nWronskian: W = sin(x)*(-sin(x)) - cos(x)*cos(x) = -1")

    print("\nGreen's function:")
    print("  G(x,x') = { sin(x)*cos(x') / (-1)  for x < x'")
    print("            { sin(x')*cos(x) / (-1)   for x > x'")
    print("  = { -sin(x)*cos(x')  for x < x'")
    print("    { -sin(x')*cos(x)  for x > x'")

    # Verify: solve y'' + y = delta(x - x0) with BCs
    L = np.pi / 2
    x0 = 1.0  # source point

    def G_func(x, xp):
        return np.where(x < xp,
                        -np.sin(x) * np.cos(xp),
                        -np.sin(xp) * np.cos(x))

    # Verify by solving y'' + y = 1 using Green's function
    x = np.linspace(0, L, 500)

    # Solution: y(x) = int_0^{pi/2} G(x,x') * 1 dx'
    y_greens = np.array([np.trapz(G_func(xi, x), x) for xi in x])

    # Exact solution of y'' + y = 1: y = 1 - cos(x) - (1-cos(pi/2))/sin(pi/2) * sin(x)
    # y_p = 1, y_h = A sin(x) + B cos(x)
    # y(0) = 0: B + 1 = 0 => B = -1
    # y(pi/2) = 0: A + 1 - 0 = 0 => A = -1
    y_exact = 1 - np.sin(x) - np.cos(x)

    print(f"\nVerification with f(x) = 1:")
    print(f"  y(0.5) Green's: {np.interp(0.5, x, y_greens):.6f}")
    print(f"  y(0.5) exact:   {1 - np.sin(0.5) - np.cos(0.5):.6f}")
    print(f"  Max error: {np.max(np.abs(y_greens - y_exact)):.2e}")


def exercise_5():
    """
    Problem 5: Eigenfunction expansion of Green's function for y''=f(x),
    y(0)=y(pi)=0. Compare with closed form.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Eigenfunction Expansion of Green's Function")
    print("=" * 60)

    L = np.pi
    print(f"\ny'' = f(x), y(0) = 0, y({L:.4f}) = 0")
    print("\nEigenvalues: lambda_n = n^2, eigenfunctions: phi_n = sin(nx)")
    print("\nEigenfunction expansion:")
    print("  G(x,x') = sum_{n=1}^inf phi_n(x)*phi_n(x') / (lambda_n * ||phi_n||^2)")
    print("          = (2/pi) * sum_{n=1}^inf sin(nx)*sin(nx') / n^2")

    # Compare series with closed form at specific points
    x_val, xp_val = 1.0, 2.0

    # Closed form: G(x,x') = x<(pi-x>)/pi
    x_less = min(x_val, xp_val)
    x_greater = max(x_val, xp_val)
    G_closed = x_less * (L - x_greater) / L
    print(f"\nAt x = {x_val}, x' = {xp_val}:")
    print(f"  Closed form: G = {G_closed:.8f}")

    # Series expansion with increasing terms
    for N in [5, 10, 50, 200]:
        G_series = (2.0 / L) * sum(
            np.sin(n * x_val) * np.sin(n * xp_val) / n**2
            for n in range(1, N + 1)
        )
        print(f"  Series (N={N:3d}): G = {G_series:.8f}  error = {abs(G_series - G_closed):.2e}")

    # Verify the identity
    print("\nVerified identity:")
    print("  sum sin(nx)sin(nx')/n^2 = (pi/2)*x<*(1 - x>/pi)")


def exercise_6():
    """
    Problem 6: Method of images for half-plane Dirichlet Green's function.
    nabla^2 G = delta(r - r'), G(x,0)=0.
    """
    print("\n" + "=" * 60)
    print("Problem 6: Method of Images (Half-Plane)")
    print("=" * 60)

    print("\nnabla^2 G = delta^2(r - r'), G(x,y=0) = 0, y > 0")
    print("\nFree-space 2D Green's function:")
    print("  G_0(r,r') = (1/(2*pi)) * ln|r - r'|")
    print("\nMethod of images: place image charge at r'' = (x', -y')")
    print("  G(r,r') = G_0(r,r') - G_0(r,r'')")
    print("          = (1/(2*pi)) * ln(|r-r'|/|r-r''|)")
    print("          = (1/(4*pi)) * ln[((x-x')^2+(y-y')^2)/((x-x')^2+(y+y')^2)]")

    # Verify G=0 at y=0
    xp, yp = 1.0, 2.0
    x_test = np.linspace(-5, 5, 100)
    y_test = 0.0
    r1 = np.sqrt((x_test - xp)**2 + (y_test - yp)**2)
    r2 = np.sqrt((x_test - xp)**2 + (y_test + yp)**2)
    G_boundary = (1.0 / (4 * np.pi)) * np.log((r1**2) / (r2**2))
    print(f"\nVerification: max|G(x, y=0)| = {np.max(np.abs(G_boundary)):.2e}")

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 7))
    x_grid = np.linspace(-4, 6, 300)
    y_grid = np.linspace(0.01, 5, 200)
    X, Y = np.meshgrid(x_grid, y_grid)

    R1_sq = (X - xp)**2 + (Y - yp)**2
    R2_sq = (X - xp)**2 + (Y + yp)**2
    G_vals = (1.0 / (4 * np.pi)) * np.log(R1_sq / R2_sq)

    levels = np.linspace(-0.5, 0.5, 21)
    cs = ax.contourf(X, Y, G_vals, levels=levels, cmap='RdBu_r')
    ax.contour(X, Y, G_vals, levels=levels, colors='k', linewidths=0.3)
    plt.colorbar(cs, ax=ax, label="G(x, y)")
    ax.plot(xp, yp, 'ko', markersize=8, label=f"Source ({xp}, {yp})")
    ax.plot(xp, -yp, 'kx', markersize=8, label=f"Image ({xp}, {-yp})")
    ax.axhline(y=0, color='k', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Method of Images: Half-Plane Green's Function")
    ax.legend()
    ax.set_ylim(-0.5, 5)
    plt.tight_layout()
    plt.savefig('ex16_method_of_images.png', dpi=150)
    plt.close()
    print("Plot saved to ex16_method_of_images.png")


def exercise_7():
    """
    Problem 7: Verify 1D heat equation Green's function properties.
    G = (1/sqrt(4*pi*alpha^2*t)) * exp(-x^2/(4*alpha^2*t))
    """
    print("\n" + "=" * 60)
    print("Problem 7: Heat Equation Green's Function Verification")
    print("=" * 60)

    alpha2 = 1.0  # alpha^2

    def G_heat(x, t):
        return np.exp(-x**2 / (4 * alpha2 * t)) / np.sqrt(4 * np.pi * alpha2 * t)

    # (a) Verify dG/dt = alpha^2 * d^2G/dx^2
    print("\n(a) Verify: dG/dt = alpha^2 * d^2G/dx^2")
    t0 = 0.5
    x = np.linspace(-5, 5, 10000)
    dx = x[1] - x[0]

    G_vals = G_heat(x, t0)
    dt = 1e-6
    dGdt = (G_heat(x, t0 + dt) - G_heat(x, t0 - dt)) / (2 * dt)
    d2Gdx2 = np.gradient(np.gradient(G_vals, dx), dx)

    residual = np.max(np.abs(dGdt[100:-100] - alpha2 * d2Gdx2[100:-100]))
    print(f"  max|dG/dt - alpha^2 * d^2G/dx^2| = {residual:.2e}  (at t={t0})")

    # (b) Verify int G dx = 1 for all t > 0
    print("\n(b) Verify: int G dx = 1 for all t > 0")
    for t_val in [0.01, 0.1, 1.0, 10.0]:
        x_wide = np.linspace(-50, 50, 100000)
        integral = np.trapz(G_heat(x_wide, t_val), x_wide)
        print(f"  t = {t_val:5.2f}: int G dx = {integral:.8f}")

    # (c) Verify lim_{t->0+} G = delta(x)
    print("\n(c) Verify: lim_{t->0+} G(x,t) -> delta(x)")
    print("  G becomes narrower and taller as t -> 0:")
    for t_val in [1.0, 0.1, 0.01, 0.001]:
        peak = G_heat(0, t_val)
        width = 2 * np.sqrt(4 * alpha2 * t_val * np.log(2))  # FWHM
        print(f"  t = {t_val:.3f}: peak = {peak:.4f}, FWHM = {width:.6f}")


def exercise_8():
    """
    Problem 8: Damped harmonic oscillator Green's functions for three cases.
    x'' + 2*gamma*x' + omega0^2*x = delta(t)
    """
    print("\n" + "=" * 60)
    print("Problem 8: Damped Oscillator Green's Functions")
    print("=" * 60)

    omega0 = 1.0

    print("\nx'' + 2*gamma*x' + omega0^2*x = delta(t)")
    print("Characteristic equation: s^2 + 2*gamma*s + omega0^2 = 0")
    print("s = -gamma +/- sqrt(gamma^2 - omega0^2)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    t = np.linspace(0, 15, 1000)

    # Case 1: Overdamped (gamma > omega0)
    gamma1 = 2.0
    beta1 = np.sqrt(gamma1**2 - omega0**2)
    G1 = np.where(t > 0,
                  np.exp(-gamma1 * t) * np.sinh(beta1 * t) / beta1,
                  0)
    print(f"\n(1) Overdamped: gamma = {gamma1} > omega0 = {omega0}")
    print(f"  beta = sqrt(gamma^2 - omega0^2) = {beta1:.4f}")
    print(f"  G(t) = exp(-gamma*t)*sinh(beta*t)/beta  (t > 0)")

    axes[0].plot(t, G1, 'b-', linewidth=2)
    axes[0].set_title(f'Overdamped (gamma={gamma1})')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('G(t)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linewidth=0.5)

    # Case 2: Critically damped (gamma = omega0)
    gamma2 = omega0
    G2 = np.where(t > 0,
                  t * np.exp(-gamma2 * t),
                  0)
    print(f"\n(2) Critically damped: gamma = omega0 = {gamma2}")
    print(f"  G(t) = t*exp(-gamma*t)  (t > 0)")

    axes[1].plot(t, G2, 'r-', linewidth=2)
    axes[1].set_title(f'Critical (gamma={gamma2})')
    axes[1].set_xlabel('t')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linewidth=0.5)

    # Case 3: Underdamped (gamma < omega0)
    gamma3 = 0.2
    omega_d = np.sqrt(omega0**2 - gamma3**2)
    G3 = np.where(t > 0,
                  np.exp(-gamma3 * t) * np.sin(omega_d * t) / omega_d,
                  0)
    print(f"\n(3) Underdamped: gamma = {gamma3} < omega0 = {omega0}")
    print(f"  omega_d = sqrt(omega0^2 - gamma^2) = {omega_d:.4f}")
    print(f"  G(t) = exp(-gamma*t)*sin(omega_d*t)/omega_d  (t > 0)")

    axes[2].plot(t, G3, 'g-', linewidth=2)
    axes[2].set_title(f'Underdamped (gamma={gamma3})')
    axes[2].set_xlabel('t')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linewidth=0.5)

    plt.suptitle("Damped Oscillator Green's Functions", fontsize=14)
    plt.tight_layout()
    plt.savefig('ex16_damped_oscillator_greens.png', dpi=150)
    plt.close()
    print("\nPlot saved to ex16_damped_oscillator_greens.png")


def exercise_9():
    """
    Problem 9: 2D Poisson equation with dipole source.
    nabla^2 phi = -delta(r-r1) + delta(r-r2), r1=(1,0), r2=(-1,0).
    """
    print("\n" + "=" * 60)
    print("Problem 9: 2D Poisson Equation - Dipole")
    print("=" * 60)

    r1 = np.array([1.0, 0.0])
    r2 = np.array([-1.0, 0.0])

    print(f"\nnabla^2 phi = -delta(r-r1) + delta(r-r2)")
    print(f"r1 = {r1}, r2 = {r2}")
    print("\nSolution by superposition:")
    print("  phi(r) = -(1/(2*pi))*ln|r-r1| + (1/(2*pi))*ln|r-r2|")
    print("         = (1/(2*pi))*ln(|r-r2|/|r-r1|)")

    # Visualize potential and electric field
    x = np.linspace(-4, 4, 400)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)

    # Distances from each source
    R1 = np.sqrt((X - r1[0])**2 + (Y - r1[1])**2)
    R2 = np.sqrt((X - r2[0])**2 + (Y - r2[1])**2)

    # Avoid singularities
    R1 = np.clip(R1, 0.05, None)
    R2 = np.clip(R2, 0.05, None)

    # Potential (positive charge at r1, negative at r2)
    phi = -(1 / (2 * np.pi)) * np.log(R1) + (1 / (2 * np.pi)) * np.log(R2)

    # Electric field E = -grad(phi)
    Ey, Ex = np.gradient(-phi, y, x)

    # Normalize for streamlines
    E_mag = np.sqrt(Ex**2 + Ey**2)
    E_mag = np.clip(E_mag, 1e-6, None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Potential contour
    levels = np.linspace(-0.5, 0.5, 21)
    cs = axes[0].contourf(X, Y, phi, levels=levels, cmap='RdBu_r')
    axes[0].contour(X, Y, phi, levels=levels, colors='k', linewidths=0.3)
    plt.colorbar(cs, ax=axes[0], label='phi')
    axes[0].plot(*r1, 'r+', markersize=12, markeredgewidth=3, label='+')
    axes[0].plot(*r2, 'b_', markersize=12, markeredgewidth=3, label='-')
    axes[0].set_title('Potential phi(x,y)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    axes[0].legend()

    # Electric field streamlines
    axes[1].streamplot(X, Y, Ex, Ey, color=np.log10(E_mag),
                       cmap='inferno', density=2, linewidth=0.8)
    axes[1].plot(*r1, 'r+', markersize=12, markeredgewidth=3, label='+')
    axes[1].plot(*r2, 'b_', markersize=12, markeredgewidth=3, label='-')
    axes[1].set_title('Electric Field E = -grad(phi)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    axes[1].legend()

    plt.suptitle("2D Dipole: Poisson Equation Solution", fontsize=14)
    plt.tight_layout()
    plt.savefig('ex16_poisson_dipole.png', dpi=150)
    plt.close()
    print("Plot saved to ex16_poisson_dipole.png")


def exercise_10():
    """
    Problem 10: Prove G(x,x') = G(x',x) using Green's second identity
    for self-adjoint operators.
    """
    print("\n" + "=" * 60)
    print("Problem 10: Symmetry of Green's Functions")
    print("=" * 60)

    print("\nProof using Green's second identity:")
    print("\nFor a self-adjoint operator L, Green's second identity gives:")
    print("  int_V [u*L[v] - v*L[u]] dV = boundary terms")
    print("\nLet u(x) = G(x, x1) and v(x) = G(x, x2), so:")
    print("  L[u] = delta(x - x1)  and  L[v] = delta(x - x2)")
    print("\nSubstituting:")
    print("  int [G(x,x1)*delta(x-x2) - G(x,x2)*delta(x-x1)] dx = [boundary]")
    print("  G(x2, x1) - G(x1, x2) = [boundary terms]")
    print("\nIf G satisfies homogeneous boundary conditions (Dirichlet or Neumann),")
    print("the boundary terms vanish, giving:")
    print("  G(x2, x1) = G(x1, x2)  QED")

    # Numerical verification with the Green's function from Problem 3
    print("\nNumerical verification with G for y''=f(x), y(0)=y(pi)=0:")
    L = np.pi
    test_pairs = [(0.5, 2.0), (1.0, 2.5), (0.3, 2.8)]
    for x1, x2 in test_pairs:
        G12 = min(x1, x2) * (L - max(x1, x2)) / L
        G21 = min(x2, x1) * (L - max(x2, x1)) / L
        print(f"  G({x1}, {x2}) = {G12:.8f}, G({x2}, {x1}) = {G21:.8f}, "
              f"diff = {abs(G12 - G21):.2e}")


if __name__ == "__main__":
    print("=== Exercise 1 ===")
    exercise_1()
    print("\n=== Exercise 2 ===")
    exercise_2()
    print("\n=== Exercise 3 ===")
    exercise_3()
    print("\n=== Exercise 4 ===")
    exercise_4()
    print("\n=== Exercise 5 ===")
    exercise_5()
    print("\n=== Exercise 6 ===")
    exercise_6()
    print("\n=== Exercise 7 ===")
    exercise_7()
    print("\n=== Exercise 8 ===")
    exercise_8()
    print("\n=== Exercise 9 ===")
    exercise_9()
    print("\n=== Exercise 10 ===")
    exercise_10()
    print("\nAll exercises completed!")
