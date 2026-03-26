"""
Exercise Solutions: Lesson 10 - Higher Order ODE and Systems
Mathematical Methods for Physical Sciences

Covers: higher-order ODEs, eigenvalue systems, coupled oscillators,
        nonlinear equilibrium analysis, Hopf bifurcation
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1_higher_order():
    """
    Problem 1: Find general solutions:
    (a) y^(4) + 4y'' = 0
    (b) y''' - y = 0
    (c) y'' + 4y' + 13y = 0
    """
    print("=" * 60)
    print("Problem 1: Higher-Order ODEs")
    print("=" * 60)

    r = sp.Symbol('r')

    # (a) y^(4) + 4y'' = 0
    char_a = r**4 + 4*r**2
    roots_a = sp.solve(char_a, r)
    print(f"\n(a) y'''' + 4y'' = 0")
    print(f"  Characteristic: r^4 + 4r^2 = r^2(r^2 + 4) = 0")
    print(f"  Roots: {roots_a}")
    print(f"  r = 0 (double), r = +/- 2i")
    print(f"  y = C1 + C2*x + C3*cos(2x) + C4*sin(2x)")

    # (b) y''' - y = 0
    char_b = r**3 - 1
    roots_b = sp.solve(char_b, r)
    print(f"\n(b) y''' - y = 0")
    print(f"  Characteristic: r^3 - 1 = (r-1)(r^2+r+1) = 0")
    print(f"  Roots: {roots_b}")
    print(f"  r = 1, r = (-1 +/- i*sqrt(3))/2")
    print(f"  y = C1*e^x + e^(-x/2)*[C2*cos(sqrt(3)*x/2) + C3*sin(sqrt(3)*x/2)]")

    # (c) y'' + 4y' + 13y = 0
    char_c = r**2 + 4*r + 13
    roots_c = sp.solve(char_c, r)
    print(f"\n(c) y'' + 4y' + 13y = 0")
    print(f"  Characteristic: r^2 + 4r + 13 = 0")
    print(f"  Roots: {roots_c}")
    print(f"  r = -2 +/- 3i")
    print(f"  y = e^(-2x)*[C1*cos(3x) + C2*sin(3x)]")


def exercise_2_eigenvalue_system():
    """
    Problem 2: Solve the system x' = Ax where A = [[3, -2], [2, -2]].
    """
    print("\n" + "=" * 60)
    print("Problem 2: Eigenvalue System")
    print("=" * 60)

    A = np.array([[3, -2], [2, -2]], dtype=float)
    print(f"\nx' = Ax, A = [[3, -2], [2, -2]]")

    # Eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print(f"\nCharacteristic equation:")
    print(f"  det(A - lambda*I) = (3-lambda)(-2-lambda) + 4 = 0")
    print(f"  lambda^2 - lambda - 2 = 0")
    print(f"  (lambda - 2)(lambda + 1) = 0")

    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        vec = eigenvectors[:, i]
        print(f"\n  lambda_{i+1} = {lam:.1f}")
        print(f"  eigenvector v_{i+1} = [{vec[0]:.4f}, {vec[1]:.4f}]")

    print(f"\nGeneral solution:")
    print(f"  x(t) = C1 * [2, 1]^T * e^(2t) + C2 * [1, 2]^T * e^(-t)")

    # Phase portrait
    fig, ax = plt.subplots(figsize=(8, 8))
    t_span = np.linspace(-2, 2, 20)

    for x0 in np.linspace(-3, 3, 8):
        for y0 in np.linspace(-3, 3, 8):
            sol = solve_ivp(lambda t, s: A @ s, [0, 3], [x0, y0],
                          t_eval=np.linspace(0, 3, 100), method='RK45')
            ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.5, alpha=0.5)

    # Eigenvectors
    for i in range(2):
        vec = eigenvectors[:, i]
        ax.arrow(0, 0, vec[0]*2, vec[1]*2, head_width=0.15,
                color='red', linewidth=2)
        ax.annotate(f'v{i+1} (l={eigenvalues[i]:.0f})',
                   xy=(vec[0]*2.2, vec[1]*2.2), fontsize=12, color='red')

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Phase Portrait (Saddle Point)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex10_phase_portrait.png', dpi=150)
    plt.close()
    print("Phase portrait saved to ex10_phase_portrait.png")


def exercise_3_coupled_oscillators():
    """
    Problem 3: Three masses connected by springs.
    m*x1'' = -k*x1 + k*(x2-x1) = -2k*x1 + k*x2
    m*x2'' = -k*(x2-x1) + k*(x3-x2) = k*x1 - 2k*x2 + k*x3
    m*x3'' = -k*(x3-x2) - k*x3 = k*x2 - 2k*x3
    """
    print("\n" + "=" * 60)
    print("Problem 3: Three Coupled Oscillators")
    print("=" * 60)

    # K/m matrix (all k and m normalized to 1)
    K_over_m = np.array([[-2, 1, 0],
                         [1, -2, 1],
                         [0, 1, -2]], dtype=float)

    print(f"\nEquation: m*x'' = K*x (normalized: m=k=1)")
    print(f"K/m matrix:\n{-K_over_m}")

    # omega^2 are eigenvalues of -K/m
    eigenvalues, eigenvectors = np.linalg.eigh(-K_over_m)

    print(f"\nNormal modes (omega^2 values and eigenvectors):")
    for i in range(3):
        omega_sq = eigenvalues[i]
        omega = np.sqrt(omega_sq)
        vec = eigenvectors[:, i]
        # Normalize to max = 1
        vec = vec / np.abs(vec).max()
        print(f"\n  Mode {i+1}:")
        print(f"    omega^2 = {omega_sq:.6f}")
        print(f"    omega   = {omega:.6f}")
        print(f"    Pattern: [{vec[0]:+.4f}, {vec[1]:+.4f}, {vec[2]:+.4f}]")

    print(f"\n  Mode 1: all masses move in phase (lowest frequency)")
    print(f"  Mode 2: outer masses opposite, middle stationary")
    print(f"  Mode 3: alternating motion (highest frequency)")

    # Simulate initial condition x1=1, x2=x3=0
    def coupled_ode(t, state):
        x = state[:3]
        v = state[3:]
        a = K_over_m @ x
        return np.concatenate([v, a])

    sol = solve_ivp(coupled_ode, [0, 30], [1, 0, 0, 0, 0, 0],
                    t_eval=np.linspace(0, 30, 1000), method='RK45')

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(3):
        ax.plot(sol.t, sol.y[i], linewidth=1.5, label=f'x_{i+1}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Displacement')
    ax.set_title('Three Coupled Oscillators (IC: x1=1, x2=x3=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex10_coupled_oscillators.png', dpi=150)
    plt.close()
    print("Plot saved to ex10_coupled_oscillators.png")


def exercise_4_nonlinear_equilibrium():
    """
    Problem 4: Lotka-Volterra system
    dx/dt = x(3 - x - 2y)
    dy/dt = y(2 - x - y)
    Find equilibria and classify.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Nonlinear Equilibrium Analysis")
    print("=" * 60)

    x, y = sp.symbols('x y')
    f = x * (3 - x - 2*y)
    g = y * (2 - x - y)

    print(f"\ndx/dt = x(3 - x - 2y)")
    print(f"dy/dt = y(2 - x - y)")

    # Equilibria: f = g = 0
    equilibria = sp.solve([f, g], [x, y])
    print(f"\nEquilibrium points:")

    # Jacobian
    J = sp.Matrix([[sp.diff(f, x), sp.diff(f, y)],
                   [sp.diff(g, x), sp.diff(g, y)]])

    print(f"\nJacobian:")
    print(f"  J = [[3-2x-2y, -2x], [-y, 2-x-2y]]")

    for eq in equilibria:
        px, py = eq
        J_at = J.subs([(x, px), (y, py)])
        eigenvals = J_at.eigenvals()

        print(f"\n  ({px}, {py}):")
        print(f"    J = {J_at.tolist()}")
        print(f"    Eigenvalues: {list(eigenvals.keys())}")

        # Classify
        eigs = [complex(e) for e in eigenvals.keys()]
        real_parts = [e.real for e in eigs]

        if all(r < 0 for r in real_parts):
            stability = "STABLE NODE/SPIRAL"
        elif all(r > 0 for r in real_parts):
            stability = "UNSTABLE NODE/SPIRAL"
        elif any(r > 0 for r in real_parts) and any(r < 0 for r in real_parts):
            stability = "SADDLE POINT"
        else:
            stability = "CENTER/MARGINAL"
        print(f"    Classification: {stability}")


def exercise_5_hopf_bifurcation():
    """
    Problem 5: System exhibiting Hopf bifurcation:
    dx/dt = mu*x - y - x*(x^2 + y^2)
    dy/dt = x + mu*y - y*(x^2 + y^2)
    """
    print("\n" + "=" * 60)
    print("Problem 5: Hopf Bifurcation")
    print("=" * 60)

    print(f"\ndx/dt = mu*x - y - x*(x^2 + y^2)")
    print(f"dy/dt = x + mu*y - y*(x^2 + y^2)")

    print(f"\nEquilibrium: (0, 0) for all mu")
    print(f"\nJacobian at origin:")
    print(f"  J = [[mu, -1], [1, mu]]")
    print(f"  Eigenvalues: mu +/- i")
    print(f"\n  mu < 0: stable spiral (Re(lambda) < 0)")
    print(f"  mu = 0: center (bifurcation point)")
    print(f"  mu > 0: unstable spiral (Re(lambda) > 0)")
    print(f"         limit cycle of radius sqrt(mu)")

    # Polar form: r' = mu*r - r^3 = r*(mu - r^2)
    # theta' = 1
    print(f"\nIn polar coordinates:")
    print(f"  r' = r*(mu - r^2)")
    print(f"  theta' = 1")
    print(f"\n  For mu > 0: stable limit cycle at r = sqrt(mu)")
    print(f"  For mu <= 0: origin is globally stable")

    # Phase portraits for different mu
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    mus = [-0.5, 0.0, 0.5]
    titles = ['mu = -0.5\n(Stable spiral)', 'mu = 0\n(Center)',
              'mu = 0.5\n(Limit cycle)']

    for ax, mu, title in zip(axes, mus, titles):
        def system(t, state, mu=mu):
            x, y = state
            r_sq = x**2 + y**2
            return [mu*x - y - x*r_sq, x + mu*y - y*r_sq]

        # Multiple initial conditions
        for r0 in [0.3, 0.6, 1.0, 1.5]:
            for theta0 in np.linspace(0, 2*np.pi, 4, endpoint=False):
                x0 = r0 * np.cos(theta0)
                y0 = r0 * np.sin(theta0)
                sol = solve_ivp(system, [0, 20], [x0, y0],
                              t_eval=np.linspace(0, 20, 2000),
                              method='RK45', rtol=1e-8)
                ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.5, alpha=0.5)

        if mu > 0:
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(np.sqrt(mu)*np.cos(theta), np.sqrt(mu)*np.sin(theta),
                   'r-', linewidth=2, label=f'r = sqrt({mu})')
            ax.legend(fontsize=10)

        ax.plot(0, 0, 'ko', markersize=8)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.suptitle('Hopf Bifurcation', fontsize=15)
    plt.tight_layout()
    plt.savefig('ex10_hopf_bifurcation.png', dpi=150)
    plt.close()
    print("Plot saved to ex10_hopf_bifurcation.png")


if __name__ == "__main__":
    exercise_1_higher_order()
    exercise_2_eigenvalue_system()
    exercise_3_coupled_oscillators()
    exercise_4_nonlinear_equilibrium()
    exercise_5_hopf_bifurcation()
