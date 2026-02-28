"""
Exercises for Lesson 06: ODE Systems

Topics: Lotka-Volterra linearization, Jacobian analysis, eigenvalue stability.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Exercise 1: Linearize Lotka-Volterra and Analyze Eigenvalues
# ---------------------------------------------------------------------------
# For the Lotka-Volterra system:
#   dx/dt = alpha*x - beta*x*y
#   dy/dt = delta*x*y - gamma*y
# with alpha=1.0, beta=0.1, gamma=1.5, delta=0.075:
# 1. Find the nontrivial equilibrium point.
# 2. Compute the Jacobian at the equilibrium.
# 3. Find eigenvalues and classify the equilibrium.
# 4. Simulate and plot trajectories around the equilibrium.
# ---------------------------------------------------------------------------

def exercise_1():
    """Linearize Lotka-Volterra around equilibrium and analyze eigenvalues."""
    alpha, beta, gamma, delta = 1.0, 0.1, 1.5, 0.075

    # Equilibrium point (nontrivial)
    x_eq = gamma / delta
    y_eq = alpha / beta
    print(f"Equilibrium: (x*, y*) = ({x_eq:.2f}, {y_eq:.2f})")

    # Jacobian of the system:
    # J = [[alpha - beta*y, -beta*x],
    #      [delta*y,        delta*x - gamma]]
    # At equilibrium:
    J = np.array([[0, -beta * x_eq],
                  [delta * y_eq, 0]])
    print(f"\nJacobian at equilibrium:")
    print(J)

    eigenvalues = np.linalg.eigvals(J)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Real parts:  {eigenvalues.real}")
    print(f"Imag parts:  {eigenvalues.imag}")
    print()

    if np.allclose(eigenvalues.real, 0):
        print("Pure imaginary eigenvalues => center (periodic orbits)")
        print("The Lotka-Volterra system has a conserved quantity,")
        print("so trajectories form closed loops around the equilibrium.")
    elif np.all(eigenvalues.real < 0):
        print("Negative real parts => stable node/spiral")
    else:
        print("Positive real parts present => unstable")

    # Simulate trajectories around equilibrium
    def lotka_volterra(t, state):
        x, y = state
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return [dxdt, dydt]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Phase portrait with multiple initial conditions
    t_span = (0, 50)
    t_eval = np.linspace(0, 50, 2000)
    ics = [
        (x_eq * 0.5, y_eq * 0.5),
        (x_eq * 0.8, y_eq * 1.2),
        (x_eq * 1.5, y_eq * 0.8),
        (x_eq * 1.2, y_eq * 1.5),
    ]

    for x0, y0 in ics:
        sol = solve_ivp(lotka_volterra, t_span, [x0, y0], t_eval=t_eval,
                        rtol=1e-10, atol=1e-12)
        axes[0].plot(sol.y[0], sol.y[1], linewidth=0.8)

    axes[0].scatter([x_eq], [y_eq], s=100, color='red', zorder=5,
                    label=f'Equilibrium ({x_eq:.0f}, {y_eq:.0f})')
    axes[0].set_xlabel('Prey (x)')
    axes[0].set_ylabel('Predator (y)')
    axes[0].set_title('Phase Portrait')
    axes[0].legend()
    axes[0].grid(True)

    # Time series for one trajectory
    sol = solve_ivp(lotka_volterra, t_span, [x_eq * 0.8, y_eq * 1.2],
                    t_eval=t_eval, rtol=1e-10, atol=1e-12)
    axes[1].plot(sol.t, sol.y[0], 'b-', label='Prey (x)')
    axes[1].plot(sol.t, sol.y[1], 'r-', label='Predator (y)')
    axes[1].axhline(x_eq, color='b', linestyle=':', alpha=0.5)
    axes[1].axhline(y_eq, color='r', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Population')
    axes[1].set_title('Lotka-Volterra Dynamics')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('ex06_lotka_volterra.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to ex06_lotka_volterra.png")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 1: Lotka-Volterra Linearization and Eigenvalue Analysis")
    print("=" * 60)
    exercise_1()
