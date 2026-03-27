"""
Runge-Kutta Methods
Runge-Kutta Methods for ODEs

Higher-order accurate numerical methods for solving ODEs.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Union


# =============================================================================
# 1. RK2 (2nd-order Runge-Kutta)
# =============================================================================
def rk2_midpoint(
    f: Callable[[float, float], float],
    y0: float,
    t_span: Tuple[float, float],
    h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RK2 Midpoint Method

    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h*k1/2)
    y_{n+1} = y_n + h*k2

    Error: O(h^2)
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / h)

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h*k1/2)
        y[i + 1] = y[i] + h * k2

    return t, y


def rk2_heun(
    f: Callable[[float, float], float],
    y0: float,
    t_span: Tuple[float, float],
    h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RK2 Heun's Method (same as modified Euler)

    k1 = f(t_n, y_n)
    k2 = f(t_n + h, y_n + h*k1)
    y_{n+1} = y_n + h*(k1 + k2)/2

    Error: O(h^2)
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / h)

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h, y[i] + h*k1)
        y[i + 1] = y[i] + h * (k1 + k2) / 2

    return t, y


# =============================================================================
# 2. RK4 (4th-order Runge-Kutta) - Most widely used
# =============================================================================
def rk4(
    f: Callable[[float, float], float],
    y0: float,
    t_span: Tuple[float, float],
    h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classical RK4 (Classical 4th-order Runge-Kutta)

    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h*k1/2)
    k3 = f(t_n + h/2, y_n + h*k2/2)
    k4 = f(t_n + h, y_n + h*k3)
    y_{n+1} = y_n + h*(k1 + 2*k2 + 2*k3 + k4)/6

    Error: O(h^4)
    The most widely used method
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / h)

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h*k1/2)
        k3 = f(t[i] + h/2, y[i] + h*k2/2)
        k4 = f(t[i] + h, y[i] + h*k3)
        y[i + 1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6

    return t, y


# =============================================================================
# 3. Vector RK4 (System of ODEs)
# =============================================================================
def rk4_system(
    f: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RK4 for systems of ODEs

    dy/dt = f(t, y), y = [y1, y2, ..., yn]
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / h)
    n_vars = len(y0)

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, n_vars))
    y[0] = y0

    for i in range(n_steps):
        k1 = np.array(f(t[i], y[i]))
        k2 = np.array(f(t[i] + h/2, y[i] + h*k1/2))
        k3 = np.array(f(t[i] + h/2, y[i] + h*k2/2))
        k4 = np.array(f(t[i] + h, y[i] + h*k3))
        y[i + 1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6

    return t, y


# =============================================================================
# 4. Adaptive RK45 (Runge-Kutta-Fehlberg)
# =============================================================================
def rkf45(
    f: Callable[[float, float], float],
    y0: float,
    t_span: Tuple[float, float],
    tol: float = 1e-6,
    h_init: float = 0.1,
    h_min: float = 1e-10,
    h_max: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive RK4(5) (Runge-Kutta-Fehlberg)

    Simultaneously computes 4th and 5th order approximations for error estimation
    Automatically adjusts step size based on error
    """
    # RK45 coefficients (Fehlberg)
    c = [0, 1/4, 3/8, 12/13, 1, 1/2]
    a = [
        [],
        [1/4],
        [3/32, 9/32],
        [1932/2197, -7200/2197, 7296/2197],
        [439/216, -8, 3680/513, -845/4104],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ]
    b4 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
    b5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]

    t0, tf = t_span
    t_list = [t0]
    y_list = [y0]

    t = t0
    y = y0
    h = h_init

    while t < tf:
        if t + h > tf:
            h = tf - t

        # Compute 6 k values
        k = [0] * 6
        k[0] = f(t, y)
        for j in range(1, 6):
            y_temp = y + h * sum(a[j][m] * k[m] for m in range(j))
            k[j] = f(t + c[j] * h, y_temp)

        # 4th and 5th order approximations
        y4 = y + h * sum(b4[j] * k[j] for j in range(6))
        y5 = y + h * sum(b5[j] * k[j] for j in range(6))

        # Error estimate
        error = abs(y5 - y4)

        if error < 1e-15:
            error = 1e-15

        # Step size adjustment
        h_new = 0.9 * h * (tol / error) ** 0.2

        if error <= tol:
            # Step accepted
            t = t + h
            y = y5
            t_list.append(t)
            y_list.append(y)
            h = min(h_max, max(h_min, h_new))
        else:
            # Step rejected, retry
            h = max(h_min, h_new)

    return np.array(t_list), np.array(y_list)


# =============================================================================
# 5. RK4 vs Euler Comparison
# =============================================================================
def compare_methods():
    """Compare various methods"""
    # dy/dt = y, y(0) = 1  ->  y = e^t
    f = lambda t, y: y
    y0 = 1
    t_span = (0, 2)
    exact = lambda t: np.exp(t)

    print("\nRK Method Comparison (dy/dt = y, t in [0, 2])")
    print("-" * 70)
    print(f"{'h':>10} | {'Euler':>12} | {'RK2':>12} | {'RK4':>12}")
    print("-" * 70)

    from os.path import dirname, abspath
    import sys
    sys.path.insert(0, dirname(abspath(__file__)))

    try:
        # 03_ode_euler.py starts with a digit so it cannot be imported directly
        from importlib import import_module
        euler_module = import_module("03_ode_euler")
        euler_forward = euler_module.euler_forward
    except (ImportError, ModuleNotFoundError):
        def euler_forward(f, y0, t_span, h):
            t0, tf = t_span
            n_steps = int((tf - t0) / h)
            t = np.linspace(t0, tf, n_steps + 1)
            y = np.zeros(n_steps + 1)
            y[0] = y0
            for i in range(n_steps):
                y[i + 1] = y[i] + h * f(t[i], y[i])
            return t, y

    hs = [0.2, 0.1, 0.05, 0.025, 0.0125]

    for h in hs:
        _, y_euler = euler_forward(f, y0, t_span, h)
        _, y_rk2 = rk2_heun(f, y0, t_span, h)
        _, y_rk4 = rk4(f, y0, t_span, h)

        e_euler = abs(y_euler[-1] - exact(2))
        e_rk2 = abs(y_rk2[-1] - exact(2))
        e_rk4 = abs(y_rk4[-1] - exact(2))

        print(f"{h:>10.4f} | {e_euler:>12.2e} | {e_rk2:>12.2e} | {e_rk4:>12.2e}")


# =============================================================================
# Visualization
# =============================================================================
def plot_rk_comparison():
    """RK method comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Example 1: Exponential growth
    f = lambda t, y: y
    exact = lambda t: np.exp(t)
    t_exact = np.linspace(0, 2, 200)

    ax = axes[0, 0]
    ax.plot(t_exact, exact(t_exact), 'k-', linewidth=2, label='Exact')
    for h in [0.5, 0.25, 0.1]:
        t, y = rk4(f, 1, (0, 2), h)
        ax.plot(t, y, 'o-', markersize=4, label=f'RK4 h={h}')
    ax.set_title("RK4: dy/dt = y")
    ax.legend()
    ax.grid(True)

    # Example 2: Harmonic oscillator (energy conservation)
    def harmonic(t, state):
        y, v = state
        return np.array([v, -y])

    ax = axes[0, 1]
    t_exact = np.linspace(0, 20, 500)
    ax.plot(t_exact, np.cos(t_exact), 'k-', linewidth=2, label='Exact')

    t, sol = rk4_system(harmonic, np.array([1, 0]), (0, 20), 0.1)
    ax.plot(t, sol[:, 0], 'r-', label='RK4 h=0.1')

    t, sol = rk4_system(harmonic, np.array([1, 0]), (0, 20), 0.01)
    ax.plot(t, sol[:, 0], 'b--', label='RK4 h=0.01')

    ax.set_title("Harmonic Oscillator: y'' = -y")
    ax.legend()
    ax.grid(True)

    # Example 3: Adaptive vs fixed step
    f_stiff = lambda t, y: -50 * (y - np.cos(t))

    ax = axes[1, 0]
    t_ex = np.linspace(0, 1, 500)
    y_ex = np.cos(t_ex) + (0 - 1) * np.exp(-50 * t_ex)  # Approximate analytical solution

    t1, y1 = rk4(f_stiff, 0, (0, 1), 0.01)
    t2, y2 = rkf45(f_stiff, 0, (0, 1), tol=1e-6)

    ax.plot(t_ex, y_ex, 'k-', linewidth=2, label='Reference')
    ax.plot(t1, y1, 'r-', label=f'RK4 fixed ({len(t1)} pts)')
    ax.plot(t2, y2, 'b.', label=f'RKF45 adaptive ({len(t2)} pts)')
    ax.set_title("Adaptive Step Size")
    ax.legend()
    ax.grid(True)

    # Example 4: Lotka-Volterra
    alpha, beta, gamma, delta = 1.5, 1.0, 3.0, 1.0

    def lotka_volterra(t, state):
        x, y = state
        dx = alpha * x - beta * x * y
        dy = delta * x * y - gamma * y
        return np.array([dx, dy])

    ax = axes[1, 1]
    t, sol = rk4_system(lotka_volterra, np.array([10, 5]), (0, 15), 0.01)

    ax.plot(t, sol[:, 0], 'b-', label='Prey')
    ax.plot(t, sol[:, 1], 'r-', label='Predator')
    ax.set_title("Lotka-Volterra Model")
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/runge_kutta.png', dpi=150)
    plt.close()
    print("Graph saved: runge_kutta.png")


# =============================================================================
# Test
# =============================================================================
def main():
    print("=" * 60)
    print("Runge-Kutta Methods")
    print("=" * 60)

    # Example 1: Basic test
    print("\n[Example 1] dy/dt = y, y(0) = 1, t in [0, 1]")
    print("-" * 40)

    f = lambda t, y: y
    exact = lambda t: np.exp(t)

    t, y = rk4(f, 1, (0, 1), 0.1)
    print(f"RK4 (h=0.1): y(1) = {y[-1]:.10f}")
    print(f"Exact value: e   = {exact(1):.10f}")
    print(f"Error:           = {abs(y[-1] - exact(1)):.2e}")

    # Example 2: Van der Pol oscillator
    print("\n[Example 2] Van der Pol Oscillator")
    print("-" * 40)

    mu = 1.0

    def van_der_pol(t, state):
        x, v = state
        return np.array([v, mu * (1 - x**2) * v - x])

    t, sol = rk4_system(van_der_pol, np.array([2, 0]), (0, 20), 0.01)
    print(f"Initial: x=2, v=0")
    print(f"t=20: x={sol[-1, 0]:.4f}, v={sol[-1, 1]:.4f}")

    # Example 3: Lorenz system (chaos)
    print("\n[Example 3] Lorenz System (Chaos)")
    print("-" * 40)

    sigma, rho, beta = 10, 28, 8/3

    def lorenz(t, state):
        x, y, z = state
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])

    t, sol = rk4_system(lorenz, np.array([1, 1, 1]), (0, 50), 0.01)
    print(f"Initial: (1, 1, 1)")
    print(f"t=50: ({sol[-1, 0]:.4f}, {sol[-1, 1]:.4f}, {sol[-1, 2]:.4f})")
    print("(Chaotic system - sensitive to initial conditions)")

    # Example 4: Adaptive RK45
    print("\n[Example 4] Adaptive RK45")
    print("-" * 40)

    f_test = lambda t, y: -2 * t * y
    exact_test = lambda t: np.exp(-t**2)

    t, y = rkf45(f_test, 1, (0, 3), tol=1e-8)
    print(f"dy/dt = -2ty, y(0) = 1")
    print(f"Number of steps: {len(t)}")
    print(f"y(3) = {y[-1]:.10f}")
    print(f"Exact: {exact_test(3):.10f}")
    print(f"Error: {abs(y[-1] - exact_test(3)):.2e}")

    # Method comparison
    compare_methods()

    # Visualization
    try:
        plot_rk_comparison()
    except Exception as e:
        print(f"Graph generation failed: {e}")

    print("\n" + "=" * 60)
    print("Runge-Kutta Methods Summary")
    print("=" * 60)
    print("""
    | Method   | Order | Fn calls/step | Characteristics                |
    |----------|-------|---------------|--------------------------------|
    | RK2      | 2     | 2             | Same as modified Euler         |
    | RK4      | 4     | 4             | Most widely used, accurate/efficient|
    | RK45     | 4(5)  | 6             | Adaptive, error estimation     |
    | RK8      | 8     | 13            | Very high accuracy             |

    RK4 Butcher Tableau:
    +-----+-----+-----+-----+-----+
    |  0  |     |     |     |     |
    | 1/2 | 1/2 |     |     |     |
    | 1/2 |  0  | 1/2 |     |     |
    |  1  |  0  |  0  |  1  |     |
    +-----+-----+-----+-----+-----+
    |     | 1/6 | 1/3 | 1/3 | 1/6 |
    +-----+-----+-----+-----+-----+

    Production use:
    - scipy.integrate.solve_ivp: Various RK methods supported
    - scipy.integrate.odeint: LSODA (adaptive multistep)
    """)


if __name__ == "__main__":
    main()
