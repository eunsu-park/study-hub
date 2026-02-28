"""
Exercise Solutions: Lesson 19 - Numerical Methods for Differential Equations
Calculus and Differential Equations

Topics covered:
- Euler vs RK4 comparison
- Convergence rate verification
- Lorenz system sensitivity and Lyapunov exponent
- Van der Pol stiffness detection
- Adaptive step size Euler method
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Euler vs RK4
# ============================================================
def exercise_1():
    """
    y' = y*sin(t), y(0) = 1 on [0, 10].
    Exact: y = e^{1 - cos(t)}.
    Compare Euler and RK4 with h = 0.1.
    """
    print("=" * 60)
    print("Problem 1: Euler vs RK4")
    print("=" * 60)

    def f(t, y):
        return y * np.sin(t)

    def exact(t):
        return np.exp(1 - np.cos(t))

    def euler(f, t0, y0, h, t_end):
        t_vals = [t0]
        y_vals = [y0]
        t, y = t0, y0
        while t < t_end - 1e-12:
            y = y + h * f(t, y)
            t = t + h
            t_vals.append(t)
            y_vals.append(y)
        return np.array(t_vals), np.array(y_vals)

    def rk4(f, t0, y0, h, t_end):
        t_vals = [t0]
        y_vals = [y0]
        t, y = t0, y0
        while t < t_end - 1e-12:
            k1 = f(t, y)
            k2 = f(t + h/2, y + h*k1/2)
            k3 = f(t + h/2, y + h*k2/2)
            k4 = f(t + h, y + h*k3)
            y = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            t = t + h
            t_vals.append(t)
            y_vals.append(y)
        return np.array(t_vals), np.array(y_vals)

    h = 0.1
    t_e, y_e = euler(lambda t, y: y*np.sin(t), 0, 1, h, 10)
    t_r, y_r = rk4(lambda t, y: y*np.sin(t), 0, 1, h, 10)

    err_euler = abs(y_e[-1] - exact(10))
    err_rk4 = abs(y_r[-1] - exact(10))

    print(f"\n  y' = y*sin(t), y(0) = 1, exact: y = e^(1-cos(t))")
    print(f"  h = {h}")
    print(f"\n  At t = 10:")
    print(f"    Exact:     {exact(10):.10f}")
    print(f"    Euler:     {y_e[-1]:.10f}, error = {err_euler:.6e}")
    print(f"    RK4:       {y_r[-1]:.10f}, error = {err_rk4:.6e}")
    print(f"    RK4 is {err_euler/err_rk4:.0f}x more accurate")

    # Find h for Euler to match RK4 accuracy at t=10
    print(f"\n  Finding h for Euler to match RK4 accuracy...")
    target = err_rk4
    for h_test in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]:
        _, y_test = euler(lambda t, y: y*np.sin(t), 0, 1, h_test, 10)
        err_test = abs(y_test[-1] - exact(10))
        if err_test <= target:
            print(f"    h = {h_test}: error = {err_test:.6e} <= {target:.6e}")
            print(f"    Euler needs h ~ {h_test} to match RK4 with h = 0.1")
            break
        else:
            print(f"    h = {h_test}: error = {err_test:.6e}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    t_fine = np.linspace(0, 10, 1000)
    ax1.plot(t_fine, exact(t_fine), 'k-', linewidth=2, label='Exact')
    ax1.plot(t_e, y_e, 'r--', linewidth=1.5, label=f'Euler h={h}')
    ax1.plot(t_r, y_r, 'b-.', linewidth=1.5, label=f'RK4 h={h}')
    ax1.set_xlabel('t')
    ax1.set_ylabel('y')
    ax1.set_title('Solutions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(t_e, np.abs(y_e - exact(t_e)), 'r-', linewidth=1.5, label='Euler')
    ax2.semilogy(t_r, np.abs(y_r - exact(t_r)) + 1e-16, 'b-', linewidth=1.5, label='RK4')
    ax2.set_xlabel('t')
    ax2.set_ylabel('|Error|')
    ax2.set_title('Error Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex19_euler_vs_rk4.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex19_euler_vs_rk4.png]")


# ============================================================
# Problem 2: Convergence Rates
# ============================================================
def exercise_2():
    """
    y' = -y + sin(t), y(0) = 0. Compute errors at t=5 for
    various h using Euler, RK2, RK4. Verify convergence rates.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Convergence Rates")
    print("=" * 60)

    def f(t, y):
        return -y + np.sin(t)

    # Exact: y = (sin(t) - cos(t) + e^{-t})/2
    def exact(t):
        return (np.sin(t) - np.cos(t) + np.exp(-t)) / 2

    def euler_step(f, t, y, h):
        return y + h * f(t, y)

    def rk2_step(f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + h, y + h*k1)
        return y + h * (k1 + k2) / 2

    def rk4_step(f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h/2, y + h*k2/2)
        k4 = f(t + h, y + h*k3)
        return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    def solve(stepper, f, t0, y0, h, t_end):
        t, y = t0, y0
        while t < t_end - 1e-12:
            h_actual = min(h, t_end - t)
            y = stepper(f, t, y, h_actual)
            t += h_actual
        return y

    h_values = [0.5, 0.25, 0.1, 0.05, 0.01]
    t_end = 5.0
    y_exact = exact(t_end)

    methods = [
        ("Euler", euler_step, 1),
        ("RK2", rk2_step, 2),
        ("RK4", rk4_step, 4),
    ]

    print(f"\n  y' = -y + sin(t), y(0) = 0, exact y(5) = {y_exact:.10f}")
    print(f"\n  {'h':>8s}", end="")
    for name, _, _ in methods:
        print(f"  {name + ' err':>14s}", end="")
    print()
    print(f"  {'--------':>8s}", end="")
    for _ in methods:
        print(f"  {'-'*14:>14s}", end="")
    print()

    errors = {name: [] for name, _, _ in methods}

    for h in h_values:
        print(f"  {h:>8.4f}", end="")
        for name, stepper, order in methods:
            y_num = solve(stepper, f, 0, 0, h, t_end)
            err = abs(y_num - y_exact)
            errors[name].append(err)
            print(f"  {err:>14.6e}", end="")
        print()

    # Compute slopes
    print(f"\n  Convergence rates (log-log slopes):")
    for name, _, expected in methods:
        errs = errors[name]
        slopes = [np.log(errs[i]/errs[i+1])/np.log(h_values[i]/h_values[i+1])
                  for i in range(len(h_values)-1)]
        avg_slope = np.mean(slopes)
        print(f"    {name}: slopes = {[f'{s:.2f}' for s in slopes]}, avg = {avg_slope:.2f} (expected {expected})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, _, order in methods:
        ax.loglog(h_values, errors[name], 'o-', linewidth=2, markersize=8, label=f'{name} (O(h^{order}))')
    # Reference lines
    h_ref = np.array(h_values)
    ax.loglog(h_ref, 0.5*h_ref, 'k--', alpha=0.3, label='O(h)')
    ax.loglog(h_ref, 0.1*h_ref**2, 'k:', alpha=0.3, label='O(h^2)')
    ax.loglog(h_ref, 0.01*h_ref**4, 'k-.', alpha=0.3, label='O(h^4)')
    ax.set_xlabel('Step size h', fontsize=12)
    ax.set_ylabel('Global error at t=5', fontsize=12)
    ax.set_title('Convergence Rate Verification', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('ex19_convergence_rates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex19_convergence_rates.png]")


# ============================================================
# Problem 3: Lorenz Sensitivity
# ============================================================
def exercise_3():
    """
    Lorenz system with two ICs differing by 1e-10.
    Estimate Lyapunov exponent.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Lorenz Sensitivity")
    print("=" * 60)

    sigma, rho, beta = 10.0, 28.0, 8.0/3.0

    def lorenz(t, state):
        x, y, z = state
        return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]

    ic1 = [1.0, 1.0, 1.0]
    ic2 = [1.0 + 1e-10, 1.0, 1.0]

    t_span = [0, 40]
    t_eval = np.linspace(0, 40, 10000)

    sol1 = solve_ivp(lorenz, t_span, ic1, t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-14)
    sol2 = solve_ivp(lorenz, t_span, ic2, t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-14)

    # Distance between trajectories
    dist = np.sqrt((sol1.y[0]-sol2.y[0])**2 + (sol1.y[1]-sol2.y[1])**2 + (sol1.y[2]-sol2.y[2])**2)

    print(f"\n  Lorenz: sigma={sigma}, rho={rho}, beta={beta:.4f}")
    print(f"  IC1 = {ic1}")
    print(f"  IC2 = {ic2}")
    print(f"  Initial distance: {1e-10:.1e}")

    # Estimate Lyapunov exponent from exponential growth phase
    # d(t) ~ d(0) * e^{lambda*t}
    # Find region where distance grows exponentially (before saturation)
    valid = (dist > 1e-8) & (dist < 10)  # exponential growth region
    if np.any(valid):
        t_valid = t_eval[valid]
        d_valid = dist[valid]
        # Linear fit of log(d) vs t
        coeffs = np.polyfit(t_valid, np.log(d_valid), 1)
        lyapunov = coeffs[0]
        print(f"\n  Lyapunov exponent estimate: lambda ~ {lyapunov:.4f}")
        print(f"  (Literature value for standard Lorenz: ~0.905)")
    else:
        lyapunov = 0.9
        print(f"\n  Could not estimate Lyapunov exponent from data")

    # Time of divergence
    diverge_idx = np.argmax(dist > 1)
    if diverge_idx > 0:
        print(f"  Trajectories diverge to O(1) at t ~ {t_eval[diverge_idx]:.2f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(sol1.t, sol1.y[0], 'b-', linewidth=0.8, label='IC1')
    axes[0, 0].plot(sol2.t, sol2.y[0], 'r-', linewidth=0.8, label='IC2')
    axes[0, 0].set_xlabel('t')
    axes[0, 0].set_ylabel('x')
    axes[0, 0].set_title('x(t) for both trajectories')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(t_eval, dist + 1e-16, 'b-', linewidth=1.5)
    if np.any(valid):
        t_fit = np.linspace(t_valid[0], t_valid[-1], 100)
        axes[0, 1].semilogy(t_fit, np.exp(coeffs[1])*np.exp(lyapunov*t_fit), 'r--',
                            linewidth=2, label=f'$\\lambda$ = {lyapunov:.3f}')
    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('Distance')
    axes[0, 1].set_title('Distance Between Trajectories')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3D trajectories
    ax3d = fig.add_subplot(2, 2, 3, projection='3d')
    ax3d.plot(sol1.y[0], sol1.y[1], sol1.y[2], 'b-', linewidth=0.3, alpha=0.7)
    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    ax3d.set_title('Lorenz Attractor (IC1)')

    ax3d2 = fig.add_subplot(2, 2, 4, projection='3d')
    ax3d2.plot(sol2.y[0], sol2.y[1], sol2.y[2], 'r-', linewidth=0.3, alpha=0.7)
    ax3d2.set_xlabel('x')
    ax3d2.set_ylabel('y')
    ax3d2.set_zlabel('z')
    ax3d2.set_title('Lorenz Attractor (IC2)')

    plt.tight_layout()
    plt.savefig('ex19_lorenz_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex19_lorenz_sensitivity.png]")


# ============================================================
# Problem 4: Van der Pol Stiffness
# ============================================================
def exercise_4():
    """
    Van der Pol: y'' - mu*(1-y^2)*y' + y = 0.
    Compare RK45 and Radau for mu = 1, 10, 100, 1000.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Van der Pol Stiffness")
    print("=" * 60)

    def vdp(t, state, mu):
        y, v = state
        return [v, mu*(1 - y**2)*v - y]

    mus = [1, 10, 100, 1000]
    t_end = 200

    print(f"\n  Van der Pol: y'' - mu*(1-y^2)*y' + y = 0, y(0)=2, y'(0)=0")
    print(f"\n  {'mu':>6s}  {'RK45 evals':>12s}  {'Radau evals':>13s}  {'RK45 time?':>12s}")
    print(f"  {'------':>6s}  {'------------':>12s}  {'-------------':>13s}  {'----------':>12s}")

    for mu in mus:
        try:
            sol_rk45 = solve_ivp(lambda t, s: vdp(t, s, mu), [0, t_end], [2.0, 0.0],
                                 method='RK45', rtol=1e-6, atol=1e-9, max_step=0.5)
            rk45_evals = sol_rk45.nfev
            rk45_ok = "OK" if sol_rk45.success else "FAIL"
        except Exception:
            rk45_evals = -1
            rk45_ok = "ERROR"

        try:
            sol_radau = solve_ivp(lambda t, s: vdp(t, s, mu), [0, t_end], [2.0, 0.0],
                                  method='Radau', rtol=1e-6, atol=1e-9)
            radau_evals = sol_radau.nfev
        except Exception:
            radau_evals = -1

        slow = "SLOW" if rk45_evals > 50000 else rk45_ok
        print(f"  {mu:>6d}  {rk45_evals:>12d}  {radau_evals:>13d}  {slow:>12s}")

    print(f"\n  For large mu, the Van der Pol oscillator becomes stiff:")
    print(f"  rapid transitions near |y| = 1 (relaxation oscillations).")
    print(f"  RK45 requires extremely small steps during transitions,")
    print(f"  while Radau (implicit) handles stiffness efficiently.")
    print(f"  RK45 becomes impractically slow around mu ~ 100-1000.")


# ============================================================
# Problem 5: Adaptive Step Size Euler
# ============================================================
def exercise_5():
    """
    Adaptive Euler for y' = -50(y - cos(t)), y(0)=0, tol=1e-4.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Adaptive Step Size Euler")
    print("=" * 60)

    def f(t, y):
        return -50 * (y - np.cos(t))

    # Exact solution (particular + homogeneous):
    # y = (50*cos(t) + sin(t))/2501 + C*e^(-50t)
    # y(0) = 0 => C = -50/2501
    def exact(t):
        return (50*np.cos(t) + np.sin(t))/2501 - (50/2501)*np.exp(-50*t)

    tol = 1e-4
    t0, y0, t_end = 0.0, 0.0, 5.0
    h = 0.001  # initial step
    h_min, h_max = 1e-8, 0.1

    t_vals = [t0]
    y_vals = [y0]
    h_vals = []

    t, y = t0, y0
    steps = 0

    while t < t_end - 1e-12:
        h = min(h, t_end - t)

        # Two Euler steps of h/2
        y_half1 = y + (h/2) * f(t, y)
        y_half2 = y_half1 + (h/2) * f(t + h/2, y_half1)

        # One Euler step of h
        y_full = y + h * f(t, y)

        # Error estimate
        err = abs(y_half2 - y_full)

        if err <= tol or h <= h_min:
            # Accept step (use the more accurate value)
            t = t + h
            y = y_half2  # Richardson extrapolation: 2*y_half2 - y_full
            t_vals.append(t)
            y_vals.append(y)
            h_vals.append(h)
            steps += 1

        # Adjust step size
        if err < tol / 10 and h < h_max:
            h = min(h * 2, h_max)
        elif err > tol:
            h = max(h / 2, h_min)

    t_vals = np.array(t_vals)
    y_vals = np.array(y_vals)
    h_vals = np.array(h_vals)

    y_exact_vals = exact(t_vals)
    max_err = np.max(np.abs(y_vals - y_exact_vals))

    print(f"\n  y' = -50(y - cos(t)), y(0) = 0, tol = {tol}")
    print(f"  This is a STIFF equation (eigenvalue = -50)")
    print(f"\n  Adaptive Euler results:")
    print(f"    Total steps:  {steps}")
    print(f"    Max error:    {max_err:.6e}")
    print(f"    h range:      [{np.min(h_vals):.6e}, {np.max(h_vals):.6e}]")

    print(f"\n  Step size pattern:")
    print(f"    Initially small (stiff transient e^(-50t) dies off quickly)")
    print(f"    Then larger (smooth sinusoidal steady state)")
    print(f"    The adaptive method detects the transient and uses tiny steps,")
    print(f"    then increases h once the solution becomes smooth.")

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    t_fine = np.linspace(0, 5, 2000)
    ax1.plot(t_fine, exact(t_fine), 'r-', linewidth=2, label='Exact')
    ax1.plot(t_vals, y_vals, 'b.', markersize=3, label='Adaptive Euler')
    ax1.set_xlabel('t')
    ax1.set_ylabel('y')
    ax1.set_title('Solution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    t_h_plot = 0.5*(t_vals[:-1] + t_vals[1:])  # midpoints
    ax2.semilogy(t_h_plot, h_vals, 'b-', linewidth=1.5)
    ax2.set_xlabel('t')
    ax2.set_ylabel('Step size h')
    ax2.set_title('Adaptive Step Size')
    ax2.grid(True, alpha=0.3)

    ax3.semilogy(t_vals, np.abs(y_vals - y_exact_vals) + 1e-16, 'b-', linewidth=1.5)
    ax3.axhline(y=tol, color='r', linestyle='--', label=f'Tolerance = {tol}')
    ax3.set_xlabel('t')
    ax3.set_ylabel('|Error|')
    ax3.set_title('Error vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex19_adaptive_euler.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex19_adaptive_euler.png]")


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
    print("All exercises for Lesson 19 completed.")
    print("=" * 60)
