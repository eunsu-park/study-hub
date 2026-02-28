"""
Exercises for Lesson 05: ODE Advanced Topics

Topics: Stiff systems, implicit solvers (BDF, Radau), boundary value problems.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Exercise 1: Stiff System -- BDF vs. Radau Comparison
# ---------------------------------------------------------------------------
# Solve the stiff system:
#   dy1/dt = -1000*y1 + y2
#   dy2/dt = 999*y1 - 2*y2
# with y1(0) = 1, y2(0) = 0 over [0, 1].
# Compare BDF and Radau in terms of solution quality and function evals.
# ---------------------------------------------------------------------------

def exercise_1():
    """Stiff system: BDF vs. Radau comparison."""

    def system(t, y):
        return [-1000 * y[0] + y[1], 999 * y[0] - 2 * y[1]]

    y0 = [1.0, 0.0]
    t_span = (0, 1)
    t_eval = np.linspace(0, 1, 200)

    # Eigenvalues of the Jacobian
    J = np.array([[-1000, 1], [999, -2]])
    eigvals = np.linalg.eigvals(J)
    stiffness_ratio = max(abs(eigvals)) / min(abs(eigvals))
    print(f"Jacobian eigenvalues: {eigvals}")
    print(f"Stiffness ratio: {stiffness_ratio:.1f}")
    print()

    results = {}
    for method in ['BDF', 'Radau', 'RK45']:
        try:
            sol = solve_ivp(system, t_span, y0, method=method,
                            t_eval=t_eval, rtol=1e-8, atol=1e-10)
            results[method] = sol
            print(f"{method:>6}: nfev={sol.nfev:>6}  success={sol.success}")
        except Exception as e:
            print(f"{method:>6}: failed -- {e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for method in ['BDF', 'Radau']:
        if method in results:
            sol = results[method]
            axes[0].semilogy(sol.t, np.abs(sol.y[0]) + 1e-20, label=f'{method} y1')
            axes[1].semilogy(sol.t, np.abs(sol.y[1]) + 1e-20, label=f'{method} y2')

    axes[0].set_xlabel('t')
    axes[0].set_ylabel('|y1|')
    axes[0].set_title('Component y1 (fast decay)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel('t')
    axes[1].set_ylabel('|y2|')
    axes[1].set_title('Component y2 (slow decay)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('ex05_stiff_system.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to ex05_stiff_system.png")
    print("\nBoth BDF and Radau handle stiffness well. RK45 (explicit) requires")
    print("many more function evaluations due to stability constraints.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 1: Stiff System -- BDF vs. Radau Comparison")
    print("=" * 60)
    exercise_1()
