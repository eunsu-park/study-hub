"""
Exercises for Lesson 07: PDE Overview

Topics: PDE classification (elliptic/parabolic/hyperbolic), boundary conditions,
        analytical solutions, well-posedness.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Exercise 1: PDE Classification
# ---------------------------------------------------------------------------
# Classify the following PDEs (elliptic/parabolic/hyperbolic):
#   1. d2u/dx2 + 2*d2u/dy2 = 0
#   2. du/dt = 4*d2u/dx2
#   3. d2u/dt2 = 9*d2u/dx2
#   4. d2u/dx2 - d2u/dy2 = 0
#
# For Au_xx + 2Bu_xy + Cu_yy + ... = 0, the discriminant is D = B^2 - AC:
#   D < 0 => elliptic
#   D = 0 => parabolic
#   D > 0 => hyperbolic
# ---------------------------------------------------------------------------

def exercise_1():
    """PDE classification using the discriminant."""
    cases = [
        ("d2u/dx2 + 2*d2u/dy2 = 0", 1, 0, 2),
        ("du/dt = 4*d2u/dx2",         4, 0, 0),
        ("d2u/dt2 = 9*d2u/dx2",       -9, 0, 1),
        ("d2u/dx2 - d2u/dy2 = 0",     1, 0, -1),
    ]

    print(f"{'PDE':>35}  {'A':>3} {'B':>3} {'C':>3}  {'D=B^2-AC':>9}  {'Type':>12}")
    print("-" * 78)

    for desc, A, B, C in cases:
        D = B**2 - A * C
        if D < 0:
            pde_type = "Elliptic"
        elif D == 0:
            pde_type = "Parabolic"
        else:
            pde_type = "Hyperbolic"
        print(f"{desc:>35}  {A:>3} {B:>3} {C:>3}  {D:>9}  {pde_type:>12}")

    print()
    print("Notes:")
    print("  1. A=1, B=0, C=2: D = 0 - 2 = -2 < 0 => Elliptic (Laplace-type)")
    print("  2. Heat equation: parabolic (du/dt = alpha * d2u/dx2)")
    print("  3. Wave equation: hyperbolic (d2u/dt2 = c^2 * d2u/dx2)")
    print("  4. A=1, B=0, C=-1: D = 0 - (-1) = 1 > 0 => Hyperbolic (wave-type)")


# ---------------------------------------------------------------------------
# Exercise 2: Setting Boundary Conditions
# ---------------------------------------------------------------------------
# Choose appropriate boundary condition type for rod heat conduction:
#   1. Left end immersed in ice water (0 C)
#   2. Right end perfectly insulated
#   3. Heat exchange with air at left end
# ---------------------------------------------------------------------------

def exercise_2():
    """Choosing boundary condition types for physical situations."""
    print("Rod heat conduction problem: du/dt = alpha * d2u/dx2")
    print()
    print("1. Left end immersed in ice water (0 C):")
    print("   => Dirichlet BC: u(0, t) = 0")
    print("   Reason: Temperature is fixed at the boundary.")
    print()
    print("2. Right end perfectly insulated:")
    print("   => Neumann BC: du/dx(L, t) = 0")
    print("   Reason: No heat flux through an insulated boundary.")
    print()
    print("3. Heat exchange with air at left end:")
    print("   => Robin (mixed) BC: -k * du/dx(0,t) = h * (u(0,t) - T_air)")
    print("   or equivalently: alpha * u + beta * du/dn = gamma")
    print("   Reason: Convective heat transfer involves both temperature")
    print("   and its gradient (Newton's law of cooling).")


# ---------------------------------------------------------------------------
# Exercise 3: Deriving Analytical Solution
# ---------------------------------------------------------------------------
# For the 1D heat equation u_t = alpha*u_xx with u(0,t) = u(L,t) = 0
# and u(x,0) = sin(2*pi*x/L), derive the analytical solution.
# ---------------------------------------------------------------------------

def exercise_3():
    """Derive and verify the analytical solution for 1D heat equation."""
    print("Problem: u_t = alpha * u_xx on [0, L]")
    print("  BC: u(0,t) = u(L,t) = 0")
    print("  IC: u(x,0) = sin(2*pi*x/L)")
    print()
    print("Solution by separation of variables:")
    print("  Assume u(x,t) = X(x) * T(t)")
    print("  => X''/X = T'/(alpha*T) = -lambda (separation constant)")
    print()
    print("  Spatial part: X'' + lambda*X = 0, X(0) = X(L) = 0")
    print("  => lambda_n = (n*pi/L)^2, X_n(x) = sin(n*pi*x/L)")
    print()
    print("  Temporal part: T' = -alpha*lambda*T")
    print("  => T_n(t) = exp(-alpha*(n*pi/L)^2 * t)")
    print()
    print("  The IC u(x,0) = sin(2*pi*x/L) matches mode n=2.")
    print()
    print("  ANSWER: u(x,t) = sin(2*pi*x/L) * exp(-alpha*(2*pi/L)^2 * t)")
    print()

    # Numerical verification
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    L = 1.0
    alpha = 0.01
    nx = 101
    x = np.linspace(0, L, nx)

    def exact_solution(x, t):
        return np.sin(2 * np.pi * x / L) * np.exp(-alpha * (2 * np.pi / L)**2 * t)

    plt.figure(figsize=(9, 5))
    for t in [0, 0.5, 1.0, 2.0, 5.0]:
        u = exact_solution(x, t)
        plt.plot(x, u, label=f't = {t}')

    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Analytical Solution: u = sin(2*pi*x/L) * exp(-alpha*(2*pi/L)^2*t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ex07_analytical_solution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex07_analytical_solution.png")


# ---------------------------------------------------------------------------
# Exercise 4: Checking Well-Posedness
# ---------------------------------------------------------------------------
# Determine whether the following problems are well-posed:
#   1. Laplace equation + Dirichlet BC
#   2. Heat equation + initial condition + Dirichlet BC
#   3. Heat equation (time reversed) + final condition
# ---------------------------------------------------------------------------

def exercise_4():
    """Determine well-posedness of various PDE problems."""
    print("Hadamard's well-posedness criteria:")
    print("  1. Existence: a solution exists")
    print("  2. Uniqueness: the solution is unique")
    print("  3. Stability: continuous dependence on data")
    print()
    print("=" * 60)
    print("Problem 1: Laplace equation + Dirichlet BC")
    print("  nabla^2 u = 0 in Omega, u = g on dOmega")
    print("  Existence:  YES (via maximum principle, harmonic function theory)")
    print("  Uniqueness: YES (maximum principle: if two solutions exist,")
    print("              their difference is harmonic with zero boundary => zero)")
    print("  Stability:  YES (maximum principle bounds interior values)")
    print("  ==> WELL-POSED")
    print()
    print("=" * 60)
    print("Problem 2: Heat equation + IC + Dirichlet BC")
    print("  u_t = alpha*u_xx, u(x,0) = f(x), u(0,t) = u(L,t) = 0")
    print("  Existence:  YES (Fourier series solution)")
    print("  Uniqueness: YES (energy method / maximum principle)")
    print("  Stability:  YES (solution decays exponentially)")
    print("  ==> WELL-POSED")
    print()
    print("=" * 60)
    print("Problem 3: Backward heat equation + final condition")
    print("  u_t = -alpha*u_xx (time reversed), u(x,T) = g(x)")
    print("  Existence:  Questionable (not all final data lead to solutions)")
    print("  Uniqueness: YES (if solution exists, it is unique)")
    print("  Stability:  NO! Small perturbations in final data grow")
    print("              exponentially backward in time.")
    print("              High-frequency Fourier modes are amplified")
    print("              by e^{alpha*n^2*pi^2*T/L^2} as we go backward.")
    print("  ==> ILL-POSED (violates stability)")
    print()
    print("This is why the backward heat equation requires regularization")
    print("techniques (e.g., Tikhonov, quasi-reversibility) in practice.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 1: PDE Classification")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("Exercise 2: Setting Boundary Conditions")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("Exercise 3: Deriving Analytical Solution")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("Exercise 4: Checking Well-Posedness")
    print("=" * 60)
    exercise_4()
