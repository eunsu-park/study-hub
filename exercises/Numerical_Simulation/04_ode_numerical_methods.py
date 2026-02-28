"""
Exercises for Lesson 04: ODE Numerical Methods

Topics: Euler method convergence, RK2/RK4 order verification,
        stiff ODE solver comparison, energy conservation in Hamiltonian systems.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Exercise 1: Euler Method Step Size and Global Error
# ---------------------------------------------------------------------------
# Solve dy/dt = -2y + 2, y(0) = 0 (exact: y(t) = 1 - e^(-2t)) using
# forward Euler with h = 0.5, 0.25, 0.1, 0.05, 0.01 over [0, 3].
# Verify O(h) convergence.
# ---------------------------------------------------------------------------

def exercise_1():
    """Euler method step size and global error."""

    def forward_euler(f, y0, t_span, n_steps):
        t = np.linspace(t_span[0], t_span[1], n_steps + 1)
        h = t[1] - t[0]
        y = np.zeros(n_steps + 1)
        y[0] = y0
        for i in range(n_steps):
            y[i + 1] = y[i] + h * f(t[i], y[i])
        return t, y

    f = lambda t, y: -2 * y + 2
    y_exact = lambda t: 1 - np.exp(-2 * t)
    t_end = 3.0

    print(f"{'h':>8}  {'Global error at t=3':>22}  {'Ratio':>8}")
    print("-" * 46)
    prev_err = None
    for h in [0.5, 0.25, 0.1, 0.05, 0.01]:
        n = int(t_end / h)
        t, y = forward_euler(f, 0.0, (0, t_end), n)
        err = abs(y[-1] - y_exact(t_end))
        ratio = prev_err / err if prev_err else float('nan')
        print(f"{h:>8.3f}  {err:>22.6e}  {ratio:>8.2f}")
        prev_err = err

    print("\nRatio ~ 2.0 confirms O(h) convergence (halving h halves error).")


# ---------------------------------------------------------------------------
# Exercise 2: RK4 vs. RK2 Convergence Order Verification
# ---------------------------------------------------------------------------
# For dy/dt = cos(t)*y, y(0)=1 (exact: y(t) = e^sin(t)), verify
# convergence orders at t=2*pi for n = 10, 20, 40, 80, 160.
# ---------------------------------------------------------------------------

def exercise_2():
    """RK4 vs. RK2 convergence order verification."""

    def rk2_heun(f, y0, t_span, n):
        t = np.linspace(t_span[0], t_span[1], n + 1)
        h = t[1] - t[0]
        y = np.zeros(n + 1)
        y[0] = y0
        for i in range(n):
            k1 = f(t[i], y[i])
            k2 = f(t[i] + h, y[i] + h * k1)
            y[i + 1] = y[i] + h / 2 * (k1 + k2)
        return t, y

    def rk4(f, y0, t_span, n):
        t = np.linspace(t_span[0], t_span[1], n + 1)
        h = t[1] - t[0]
        y = np.zeros(n + 1)
        y[0] = y0
        for i in range(n):
            k1 = f(t[i], y[i])
            k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
            k3 = f(t[i] + h / 2, y[i] + h / 2 * k2)
            k4 = f(t[i] + h, y[i] + h * k3)
            y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return t, y

    f = lambda t, y: np.cos(t) * y
    y_exact_end = np.exp(np.sin(2 * np.pi))  # y(2*pi) = e^0 = 1
    t_span = (0, 2 * np.pi)
    n_values = [10, 20, 40, 80, 160]

    print(f"{'n':>5}  {'RK2 error':>12}  {'RK2 order':>10}  {'RK4 error':>12}  {'RK4 order':>10}")
    print("-" * 60)
    prev2 = prev4 = None
    for n in n_values:
        _, y2 = rk2_heun(f, 1.0, t_span, n)
        _, y4 = rk4(f, 1.0, t_span, n)
        e2 = abs(y2[-1] - y_exact_end)
        e4 = abs(y4[-1] - y_exact_end)
        ord2 = np.log2(prev2 / e2) if prev2 else float('nan')
        ord4 = np.log2(prev4 / e4) if prev4 else float('nan')
        print(f"{n:>5}  {e2:>12.3e}  {ord2:>10.2f}  {e4:>12.3e}  {ord4:>10.2f}")
        prev2, prev4 = e2, e4

    print("\nRK2 order -> ~2.0, RK4 order -> ~4.0 as expected.")


# ---------------------------------------------------------------------------
# Exercise 3: Stiffness and Solver Choice
# ---------------------------------------------------------------------------
# Solve dy/dt = -100(y - cos(t)) - sin(t), y(0) = 0 (exact: y(t) = cos(t))
# using RK45 and Radau. Compare nfev and explain why RK45 needs more.
# ---------------------------------------------------------------------------

def exercise_3():
    """Stiffness and solver choice comparison."""

    def stiff_ode(t, y):
        return [-100 * (y[0] - np.cos(t)) - np.sin(t)]

    t_span = (0, 2 * np.pi)
    y0 = [0.0]
    t_eval = np.linspace(0, 2 * np.pi, 200)
    y_exact = np.cos(t_eval)

    results = {}
    for method in ['RK45', 'Radau', 'BDF']:
        sol = solve_ivp(stiff_ode, t_span, y0, method=method,
                        t_eval=t_eval, rtol=1e-6, atol=1e-8)
        results[method] = sol
        err = np.max(np.abs(sol.y[0] - y_exact))
        print(f"{method:>6}: nfev={sol.nfev:>6}  nsteps={len(sol.t):>5}  "
              f"max_error={err:.2e}  success={sol.success}")

    # Plot
    plt.figure(figsize=(9, 4))
    for method, sol in results.items():
        plt.plot(sol.t, sol.y[0], label=f'{method} (nfev={sol.nfev})', alpha=0.8)
    plt.plot(t_eval, y_exact, 'k--', linewidth=2, label='Exact cos(t)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title("Stiff ODE: y' = -100(y - cos t) - sin t")
    plt.legend()
    plt.grid(True)
    plt.savefig('ex04_stiff_solvers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to ex04_stiff_solvers.png")
    print("\nExplanation: RK45 is explicit with limited stability region.")
    print("For lambda=-100, stability requires h < 0.035 regardless of accuracy.")
    print("Radau is implicit and L-stable, allowing large steps dictated by accuracy.")


# ---------------------------------------------------------------------------
# Exercise 4: Energy Conservation in Hamiltonian Systems
# ---------------------------------------------------------------------------
# Solve x'' + omega^2*x = 0 (omega=1) with x(0)=1, x'(0)=0 using
# forward Euler and RK4 with h=0.1 over 50 periods. Compare energy
# conservation: E(t) = 0.5*v^2 + 0.5*omega^2*x^2.
# ---------------------------------------------------------------------------

def exercise_4():
    """Energy conservation in Hamiltonian systems."""
    omega = 1.0
    h = 0.1
    T = 2 * np.pi / omega
    n_steps = int(50 * T / h)

    def harmonic(t, y):
        return [y[1], -omega**2 * y[0]]

    y0 = [1.0, 0.0]

    # Forward Euler
    t_eu = [0.0]
    y_eu = [np.array(y0)]
    for _ in range(n_steps):
        yi = y_eu[-1]
        ti = t_eu[-1]
        dy = np.array(harmonic(ti, yi))
        y_eu.append(yi + h * dy)
        t_eu.append(ti + h)
    t_eu = np.array(t_eu)
    y_eu = np.array(y_eu)

    # RK4
    def rk4_step(f, t, y, h):
        k1 = np.array(f(t, y))
        k2 = np.array(f(t + h / 2, y + h / 2 * k1))
        k3 = np.array(f(t + h / 2, y + h / 2 * k2))
        k4 = np.array(f(t + h, y + h * k3))
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    t_rk = [0.0]
    y_rk = [np.array(y0)]
    for _ in range(n_steps):
        y_rk.append(rk4_step(harmonic, t_rk[-1], y_rk[-1], h))
        t_rk.append(t_rk[-1] + h)
    t_rk = np.array(t_rk)
    y_rk = np.array(y_rk)

    # Energy
    E_exact = 0.5
    E_eu = 0.5 * y_eu[:, 1]**2 + 0.5 * omega**2 * y_eu[:, 0]**2
    E_rk = 0.5 * y_rk[:, 1]**2 + 0.5 * omega**2 * y_rk[:, 0]**2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t_eu, E_eu, 'b-', linewidth=0.8, label='Forward Euler')
    axes[0].plot(t_rk, E_rk, 'r-', linewidth=0.8, label='RK4')
    axes[0].axhline(E_exact, color='k', linestyle='--', label=f'Exact E={E_exact}')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('Energy')
    axes[0].set_title('Energy vs Time (50 periods)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].semilogy(t_eu, np.abs(E_eu - E_exact), 'b-', linewidth=0.8, label='Euler |dE|')
    axes[1].semilogy(t_rk, np.abs(E_rk - E_exact) + 1e-17, 'r-', linewidth=0.8, label='RK4 |dE|')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('|Energy error|')
    axes[1].set_title('Energy Error (log scale)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('ex04_energy_conservation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex04_energy_conservation.png")

    print(f"Euler: energy grows ~ linearly (not symplectic)")
    print(f"RK4:   energy error ~ {np.max(np.abs(E_rk - E_exact)):.2e} (small but non-zero)")
    print("For long-time Hamiltonian simulations, symplectic integrators are preferred.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 1: Euler Method Step Size and Global Error")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("Exercise 2: RK4 vs. RK2 Convergence Order Verification")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("Exercise 3: Stiffness and Solver Choice")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("Exercise 4: Energy Conservation in Hamiltonian Systems")
    print("=" * 60)
    exercise_4()
