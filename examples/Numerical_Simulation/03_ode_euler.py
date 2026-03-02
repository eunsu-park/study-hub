"""
Ordinary Differential Equations - Euler Method
ODE - Euler Method

The most basic method for numerically solving initial value problems dy/dx = f(x, y), y(x_0) = y_0.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List


# =============================================================================
# 1. Forward Euler Method
# =============================================================================
def euler_forward(
    f: Callable[[float, float], float],
    y0: float,
    t_span: Tuple[float, float],
    h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward Euler method (explicit Euler)

    y_{n+1} = y_n + h * f(t_n, y_n)

    Error: O(h) - first-order accuracy
    Stability: Conditionally stable

    Args:
        f: dy/dt = f(t, y)
        y0: Initial value y(t0)
        t_span: (t0, tf) time interval
        h: Time step size

    Returns:
        (t array, y array)
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / h)

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        y[i + 1] = y[i] + h * f(t[i], y[i])

    return t, y


# =============================================================================
# 2. Backward Euler Method
# =============================================================================
def euler_backward(
    f: Callable[[float, float], float],
    df_dy: Callable[[float, float], float],
    y0: float,
    t_span: Tuple[float, float],
    h: float,
    newton_tol: float = 1e-10,
    max_iter: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward Euler method (implicit Euler)

    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})

    Solves implicit equation using Newton's method
    Error: O(h) - first-order accuracy
    Stability: Unconditionally stable (suitable for stiff problems)

    Args:
        f: dy/dt = f(t, y)
        df_dy: partial f / partial y (Jacobian)
        y0: Initial value
        t_span: Time interval
        h: Time step size
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / h)

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        # Solve for y_{n+1} using Newton's method
        # g(y) = y - y_n - h*f(t_{n+1}, y) = 0
        y_new = y[i]  # Initial guess (forward Euler)
        t_new = t[i + 1]

        for _ in range(max_iter):
            g = y_new - y[i] - h * f(t_new, y_new)
            dg = 1 - h * df_dy(t_new, y_new)

            if abs(dg) < 1e-15:
                break

            delta = g / dg
            y_new = y_new - delta

            if abs(delta) < newton_tol:
                break

        y[i + 1] = y_new

    return t, y


# =============================================================================
# 3. Modified Euler Method (Heun's Method)
# =============================================================================
def euler_modified(
    f: Callable[[float, float], float],
    y0: float,
    t_span: Tuple[float, float],
    h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modified Euler method (Heun's Method)

    Predict: y*_{n+1} = y_n + h * f(t_n, y_n)
    Correct: y_{n+1} = y_n + h/2 * [f(t_n, y_n) + f(t_{n+1}, y*_{n+1})]

    Error: O(h^2) - second-order accuracy
    A variant of RK2
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / h)

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        k1 = f(t[i], y[i])
        y_pred = y[i] + h * k1
        k2 = f(t[i + 1], y_pred)
        y[i + 1] = y[i] + h * (k1 + k2) / 2

    return t, y


# =============================================================================
# 4. Solving Systems of ODEs
# =============================================================================
def euler_system(
    f: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve a system of ODEs using forward Euler

    dy/dt = f(t, y), y = [y1, y2, ..., yn]

    Args:
        f: Vector function f(t, y) -> dy/dt
        y0: Initial value vector
        t_span: Time interval
        h: Time step size

    Returns:
        (t array, y array) - y.shape = (n_steps+1, n_vars)
    """
    t0, tf = t_span
    n_steps = int((tf - t0) / h)
    n_vars = len(y0)

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, n_vars))
    y[0] = y0

    for i in range(n_steps):
        y[i + 1] = y[i] + h * np.array(f(t[i], y[i]))

    return t, y


# =============================================================================
# 5. Converting 2nd-Order ODE to 1st-Order System
# =============================================================================
def solve_second_order(
    f: Callable[[float, float, float], float],
    y0: float,
    v0: float,
    t_span: Tuple[float, float],
    h: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2nd-order ODE: y'' = f(t, y, y')
    Transform: y1 = y, y2 = y'
               y1' = y2
               y2' = f(t, y1, y2)

    Args:
        f: y'' = f(t, y, y')
        y0: Initial position
        v0: Initial velocity
        t_span, h: Time interval and step size

    Returns:
        (t, y, v) - position and velocity
    """
    def system(t, state):
        y, v = state
        return np.array([v, f(t, y, v)])

    t, solution = euler_system(system, np.array([y0, v0]), t_span, h)
    return t, solution[:, 0], solution[:, 1]


# =============================================================================
# Error Analysis
# =============================================================================
def analyze_euler_error():
    """Error analysis of Euler methods"""
    # dy/dt = y, y(0) = 1  ->  y = e^t
    f = lambda t, y: y
    df_dy = lambda t, y: 1
    y0 = 1
    t_span = (0, 1)
    exact = lambda t: np.exp(t)

    print("\nEuler method error analysis (dy/dt = y, y(0) = 1)")
    print("-" * 70)
    print(f"{'h':>10} | {'Forward Euler':>15} | {'Modified Euler':>15} | {'Backward Euler':>15}")
    print("-" * 70)

    hs = [0.1, 0.05, 0.025, 0.0125, 0.00625]

    for h in hs:
        t1, y1 = euler_forward(f, y0, t_span, h)
        t2, y2 = euler_modified(f, y0, t_span, h)
        t3, y3 = euler_backward(f, df_dy, y0, t_span, h)

        error1 = abs(y1[-1] - exact(1))
        error2 = abs(y2[-1] - exact(1))
        error3 = abs(y3[-1] - exact(1))

        print(f"{h:>10.5f} | {error1:>15.2e} | {error2:>15.2e} | {error3:>15.2e}")


# =============================================================================
# Visualization
# =============================================================================
def plot_euler_comparison():
    """Euler method comparison visualization"""
    # dy/dt = -2y + sin(t), y(0) = 1
    f = lambda t, y: -2*y + np.sin(t)
    df_dy = lambda t, y: -2
    y0 = 1
    t_span = (0, 5)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Exact solution (using scipy)
    from scipy.integrate import odeint
    t_exact = np.linspace(0, 5, 500)
    y_exact = odeint(lambda y, t: f(t, y), y0, t_exact).flatten()

    # Compare various h values
    hs = [0.5, 0.25, 0.1]
    colors = ['r', 'g', 'b']

    ax = axes[0, 0]
    ax.plot(t_exact, y_exact, 'k-', linewidth=2, label='Exact')
    for h, c in zip(hs, colors):
        t, y = euler_forward(f, y0, t_span, h)
        ax.plot(t, y, f'{c}o-', markersize=4, label=f'h={h}')
    ax.set_title('Forward Euler')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t_exact, y_exact, 'k-', linewidth=2, label='Exact')
    for h, c in zip(hs, colors):
        t, y = euler_modified(f, y0, t_span, h)
        ax.plot(t, y, f'{c}o-', markersize=4, label=f'h={h}')
    ax.set_title('Modified Euler')
    ax.legend()
    ax.grid(True)

    # Stability comparison (stiff problem)
    # dy/dt = -50(y - cos(t)), y(0) = 0
    f_stiff = lambda t, y: -50*(y - np.cos(t))
    df_stiff = lambda t, y: -50
    h = 0.05
    t_span_stiff = (0, 1)

    ax = axes[1, 0]
    t_ex = np.linspace(0, 1, 500)
    y_ex = odeint(lambda y, t: f_stiff(t, y), 0, t_ex).flatten()
    ax.plot(t_ex, y_ex, 'k-', linewidth=2, label='Exact')

    t1, y1 = euler_forward(f_stiff, 0, t_span_stiff, h)
    t2, y2 = euler_backward(f_stiff, df_stiff, 0, t_span_stiff, h)

    ax.plot(t1, y1, 'r.-', label=f'Forward Euler (h={h})')
    ax.plot(t2, y2, 'b.-', label=f'Backward Euler (h={h})')
    ax.set_title('Stiff Problem Stability')
    ax.legend()
    ax.grid(True)

    # Harmonic oscillator
    # y'' = -y  ->  y' = v, v' = -y
    def harmonic(t, state):
        y, v = state
        return np.array([v, -y])

    ax = axes[1, 1]
    t_ho = np.linspace(0, 20, 1000)
    y_ho_exact = np.cos(t_ho)  # y(0)=1, y'(0)=0

    t, sol = euler_system(harmonic, np.array([1, 0]), (0, 20), 0.1)
    ax.plot(t_ho, y_ho_exact, 'k-', linewidth=2, label='Exact')
    ax.plot(t, sol[:, 0], 'r-', label='Euler (h=0.1)')
    ax.set_title('Harmonic Oscillator y\'\' = -y')
    ax.set_xlabel('t')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/Numerical_Simulation/examples/ode_euler.png', dpi=150)
    plt.close()
    print("Graph saved: ode_euler.png")


# =============================================================================
# Test
# =============================================================================
def main():
    print("=" * 60)
    print("Ordinary Differential Equations - Euler Method")
    print("=" * 60)

    # Example 1: Exponential decay dy/dt = -y
    print("\n[Example 1] Exponential decay: dy/dt = -y, y(0) = 1")
    print("-" * 40)

    f = lambda t, y: -y
    exact = lambda t: np.exp(-t)

    t, y_forward = euler_forward(f, 1, (0, 2), 0.2)
    t, y_modified = euler_modified(f, 1, (0, 2), 0.2)

    print(f"At t=2:")
    print(f"  Exact value:      {exact(2):.6f}")
    print(f"  Forward Euler:    {y_forward[-1]:.6f}")
    print(f"  Modified Euler:   {y_modified[-1]:.6f}")

    # Example 2: Logistic equation
    print("\n[Example 2] Logistic equation: dy/dt = y(1-y), y(0) = 0.1")
    print("-" * 40)

    f_logistic = lambda t, y: y * (1 - y)
    exact_logistic = lambda t: 0.1 * np.exp(t) / (1 + 0.1 * (np.exp(t) - 1))

    t, y = euler_modified(f_logistic, 0.1, (0, 5), 0.1)
    print(f"At t=5:")
    print(f"  Exact value:      {exact_logistic(5):.6f}")
    print(f"  Modified Euler:   {y[-1]:.6f}")

    # Example 3: Pendulum motion (2nd-order ODE)
    print("\n[Example 3] Simple pendulum: theta'' = -sin(theta), theta(0) = pi/4, theta'(0) = 0")
    print("-" * 40)

    f_pendulum = lambda t, theta, omega: -np.sin(theta)
    t, theta, omega = solve_second_order(f_pendulum, np.pi/4, 0, (0, 10), 0.01)

    print(f"Periodic motion simulation complete")
    print(f"  Initial angle: {np.degrees(np.pi/4):.1f} degrees")
    print(f"  Angle at t=10: {np.degrees(theta[-1]):.2f} degrees")

    # Example 4: Lotka-Volterra (predator-prey model)
    print("\n[Example 4] Lotka-Volterra: Predator-Prey Model")
    print("-" * 40)

    alpha, beta, gamma, delta = 1.5, 1.0, 3.0, 1.0

    def lotka_volterra(t, state):
        x, y = state  # x=prey, y=predator
        dx = alpha * x - beta * x * y
        dy = delta * x * y - gamma * y
        return np.array([dx, dy])

    t, sol = euler_system(lotka_volterra, np.array([10, 5]), (0, 15), 0.001)
    print(f"  Initial: prey={10}, predator={5}")
    print(f"  t=15: prey={sol[-1, 0]:.2f}, predator={sol[-1, 1]:.2f}")

    # Error analysis
    analyze_euler_error()

    # Visualization
    try:
        plot_euler_comparison()
    except Exception as e:
        print(f"Graph generation failed: {e}")

    print("\n" + "=" * 60)
    print("Euler Method Summary")
    print("=" * 60)
    print("""
    | Method          | Accuracy | Stability      | Characteristics              |
    |-----------------|----------|----------------|------------------------------|
    | Forward Euler   | O(h)     | Conditional    | Simplest, explicit           |
    | Backward Euler  | O(h)     | Unconditional  | Implicit, suitable for stiff |
    | Modified Euler  | O(h^2)   | Conditional    | 2nd-order, a variant of RK2  |

    Limitations:
    - Low accuracy (use RK4 for higher order)
    - Cannot conserve energy (symplectic integrators needed)

    Production use:
    - scipy.integrate.odeint: Adaptive multistep method
    - scipy.integrate.solve_ivp: Various method selection available
    """)


if __name__ == "__main__":
    main()
