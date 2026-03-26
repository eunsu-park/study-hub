"""
Exercise Solutions: Lesson 12 - First-Order Ordinary Differential Equations
Calculus and Differential Equations

Topics covered:
- Separable ODE
- Mixing tank problem (linear first-order)
- Exact equation
- Euler's method comparison with exact solution
- Newton's cooling law with parameter fitting
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Separable ODE
# ============================================================
def exercise_1():
    """
    Solve dy/dx = x(1+y^2) / (y(1+x^2)), y(0) = 1.
    """
    print("=" * 60)
    print("Problem 1: Separable ODE")
    print("=" * 60)

    x, y, C = sp.symbols('x y C')

    print(f"\n  dy/dx = x(1+y^2) / (y(1+x^2))")
    print(f"\n  Separate variables:")
    print(f"  y/(1+y^2) dy = x/(1+x^2) dx")
    print(f"\n  Integrate both sides:")
    print(f"  (1/2)*ln(1+y^2) = (1/2)*ln(1+x^2) + C_1")
    print(f"  ln(1+y^2) = ln(1+x^2) + C_2")
    print(f"  1+y^2 = K*(1+x^2)  where K = e^C_2")

    # Apply IC: y(0) = 1 => 1 + 1 = K*(1+0) => K = 2
    print(f"\n  Apply y(0) = 1:")
    print(f"  1 + 1 = K*(1+0) => K = 2")
    print(f"  1 + y^2 = 2*(1+x^2)")
    print(f"  y^2 = 1 + 2*x^2")
    print(f"  y = sqrt(1 + 2*x^2)  (taking positive root since y(0) = 1 > 0)")

    # SymPy verification
    y_func = sp.Function('y')
    ode = sp.Eq(y_func(x).diff(x), x*(1 + y_func(x)**2) / (y_func(x)*(1 + x**2)))
    sol = sp.dsolve(ode, y_func(x), ics={y_func(0): 1})
    print(f"\n  SymPy solution: {sol}")

    # Numerical verification
    x_vals = np.linspace(0, 3, 100)
    y_exact = np.sqrt(1 + 2*x_vals**2)

    sol_num = solve_ivp(
        lambda t, y: t*(1 + y[0]**2) / (y[0]*(1 + t**2)),
        [0, 3], [1.0], t_eval=x_vals, method='RK45'
    )
    max_err = np.max(np.abs(sol_num.y[0] - y_exact))
    print(f"  Numerical check: max error = {max_err:.2e}")


# ============================================================
# Problem 2: Mixing Tank
# ============================================================
def exercise_2():
    """
    Tank: 500 L, initial 10 kg salt, fresh water in at 5 L/min,
    mixed out at 5 L/min. Find y(t) and time for concentration < 0.01 kg/L.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Mixing Tank")
    print("=" * 60)

    t = sp.Symbol('t', positive=True)

    # dy/dt = rate_in - rate_out
    # rate_in = 0 (fresh water)
    # rate_out = y(t)/500 * 5 = y/100
    # dy/dt = -y/100, y(0) = 10
    # Solution: y(t) = 10*e^(-t/100)

    print(f"\n  Volume constant at 500 L (in = out = 5 L/min)")
    print(f"  dy/dt = rate_in - rate_out = 0 - y(t)/500 * 5 = -y/100")
    print(f"  y(0) = 10 kg")
    print(f"\n  This is a separable equation:")
    print(f"  dy/y = -dt/100")
    print(f"  ln|y| = -t/100 + C")
    print(f"  y(t) = 10*e^(-t/100)")

    # SymPy verification
    y_func = sp.Function('y')
    ode = sp.Eq(y_func(t).diff(t), -y_func(t)/100)
    sol = sp.dsolve(ode, y_func(t), ics={y_func(0): 10})
    print(f"\n  SymPy: {sol}")

    # Concentration < 0.01 kg/L means y(t)/500 < 0.01 => y(t) < 5
    # 10*e^(-t/100) < 5
    # e^(-t/100) < 0.5
    # -t/100 < ln(0.5)
    # t > -100*ln(0.5) = 100*ln(2)
    t_threshold = -100 * np.log(0.5)
    print(f"\n  Time for concentration < 0.01 kg/L:")
    print(f"  y(t)/500 < 0.01 => y(t) < 5 kg")
    print(f"  10*e^(-t/100) < 5 => t > 100*ln(2) = {t_threshold:.4f} min")
    print(f"  = {t_threshold/60:.2f} hours")

    # Plot
    t_vals = np.linspace(0, 500, 500)
    y_vals = 10 * np.exp(-t_vals / 100)
    conc_vals = y_vals / 500

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(t_vals, y_vals, 'b-', linewidth=2)
    ax1.axhline(y=5, color='r', linestyle='--', label='y = 5 kg threshold')
    ax1.axvline(x=t_threshold, color='g', linestyle='--', label=f't = {t_threshold:.1f} min')
    ax1.set_xlabel('Time (min)', fontsize=12)
    ax1.set_ylabel('Salt (kg)', fontsize=12)
    ax1.set_title('Salt in Tank', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_vals, conc_vals, 'b-', linewidth=2)
    ax2.axhline(y=0.01, color='r', linestyle='--', label='0.01 kg/L threshold')
    ax2.set_xlabel('Time (min)', fontsize=12)
    ax2.set_ylabel('Concentration (kg/L)', fontsize=12)
    ax2.set_title('Concentration vs Time', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex12_mixing_tank.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex12_mixing_tank.png]")


# ============================================================
# Problem 3: Exact Equation
# ============================================================
def exercise_3():
    """
    (3x^2 + y*cos(x)) dx + (sin(x) - 4y^3) dy = 0.
    Check exactness, find solution.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Exact Equation")
    print("=" * 60)

    x, y = sp.symbols('x y')

    M = 3*x**2 + y*sp.cos(x)
    N = sp.sin(x) - 4*y**3

    # Check exactness: dM/dy = dN/dx
    dM_dy = sp.diff(M, y)
    dN_dx = sp.diff(N, x)

    print(f"\n  M(x,y) = {M}")
    print(f"  N(x,y) = {N}")
    print(f"\n  dM/dy = {dM_dy}")
    print(f"  dN/dx = {dN_dx}")
    print(f"  Exact: dM/dy = dN/dx? {sp.simplify(dM_dy - dN_dx) == 0}")

    # Find F(x,y) such that F_x = M, F_y = N
    # F = integral M dx = x^3 + y*sin(x) + g(y)
    # F_y = sin(x) + g'(y) = sin(x) - 4y^3 => g'(y) = -4y^3 => g(y) = -y^4
    # F(x,y) = x^3 + y*sin(x) - y^4 = C
    F = x**3 + y*sp.sin(x) - y**4

    print(f"\n  Finding F(x,y):")
    print(f"    F = integral M dx = x^3 + y*sin(x) + g(y)")
    print(f"    F_y = sin(x) + g'(y) = N = sin(x) - 4y^3")
    print(f"    g'(y) = -4y^3 => g(y) = -y^4")
    print(f"    F(x,y) = {F}")
    print(f"\n  Solution: {F} = C (implicit)")

    # Verify
    assert sp.simplify(sp.diff(F, x) - M) == 0
    assert sp.simplify(sp.diff(F, y) - N) == 0
    print(f"  Verification: F_x = M, F_y = N [PASSED]")


# ============================================================
# Problem 4: Euler's Method
# ============================================================
def exercise_4():
    """
    y' = y - x^2 + 1, y(0) = 0.5 on [0, 2].
    Euler's method with h=0.2 and h=0.1.
    Exact: y = (x+1)^2 - 0.5*e^x
    """
    print("\n" + "=" * 60)
    print("Problem 4: Euler's Method")
    print("=" * 60)

    def f(x, y):
        return y - x**2 + 1

    def exact(x):
        return (x + 1)**2 - 0.5 * np.exp(x)

    def euler_method(f, x0, y0, h, x_end):
        """Euler's method implementation."""
        n_steps = int((x_end - x0) / h)
        xs = [x0]
        ys = [y0]
        for i in range(n_steps):
            y_new = ys[-1] + h * f(xs[-1], ys[-1])
            x_new = xs[-1] + h
            xs.append(x_new)
            ys.append(y_new)
        return np.array(xs), np.array(ys)

    print(f"\n  y' = y - x^2 + 1, y(0) = 0.5")
    print(f"  Exact solution: y = (x+1)^2 - 0.5*e^x")

    # h = 0.2
    x_02, y_02 = euler_method(f, 0, 0.5, 0.2, 2)
    err_02 = abs(y_02[-1] - exact(2))

    # h = 0.1
    x_01, y_01 = euler_method(f, 0, 0.5, 0.1, 2)
    err_01 = abs(y_01[-1] - exact(2))

    print(f"\n  Exact y(2) = {exact(2):.10f}")
    print(f"\n  h = 0.2: y(2) = {y_02[-1]:.10f}, error = {err_02:.6e}")
    print(f"  h = 0.1: y(2) = {y_01[-1]:.10f}, error = {err_01:.6e}")
    print(f"  Error ratio: {err_02/err_01:.4f} (expected ~2.0 for first-order method)")

    # Table for h = 0.2
    print(f"\n  Euler's method with h = 0.2:")
    print(f"  {'x':>6s}  {'y_euler':>12s}  {'y_exact':>12s}  {'error':>12s}")
    print(f"  {'------':>6s}  {'------------':>12s}  {'------------':>12s}  {'------------':>12s}")
    for xi, yi in zip(x_02, y_02):
        ye = exact(xi)
        print(f"  {xi:>6.1f}  {yi:>12.6f}  {ye:>12.6f}  {abs(yi-ye):>12.6e}")

    # Plot
    x_fine = np.linspace(0, 2, 200)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_fine, exact(x_fine), 'k-', linewidth=2, label='Exact')
    ax.plot(x_02, y_02, 'ro-', linewidth=1.5, markersize=6, label='Euler h=0.2')
    ax.plot(x_01, y_01, 'bs-', linewidth=1.5, markersize=4, label='Euler h=0.1')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title("Euler's Method vs Exact Solution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex12_euler_method.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: ex12_euler_method.png]")


# ============================================================
# Problem 5: Newton's Cooling
# ============================================================
def exercise_5():
    """
    Coffee at 95C in room at 22C. After 5 min: 70C.
    (a) Find cooling constant k
    (b) When T = 40C?
    (c) Plot with solve_ivp
    """
    print("\n" + "=" * 60)
    print("Problem 5: Newton's Cooling")
    print("=" * 60)

    t = sp.Symbol('t', positive=True)
    k = sp.Symbol('k', positive=True)

    T_room = 22
    T_0 = 95

    # Newton's law: dT/dt = -k*(T - T_room)
    # Solution: T(t) = T_room + (T_0 - T_room)*e^(-k*t)
    # T(t) = 22 + 73*e^(-k*t)
    print(f"\n  Newton's cooling: dT/dt = -k*(T - {T_room})")
    print(f"  T(0) = {T_0}")
    print(f"  Solution: T(t) = {T_room} + {T_0 - T_room}*e^(-k*t)")

    # (a) Find k: T(5) = 70
    # 70 = 22 + 73*e^(-5k) => e^(-5k) = 48/73 => k = -ln(48/73)/5
    k_val = -np.log(48/73) / 5
    print(f"\n(a) T(5) = 70:")
    print(f"    70 = 22 + 73*e^(-5k)")
    print(f"    e^(-5k) = 48/73")
    print(f"    k = -ln(48/73)/5 = {k_val:.6f} per minute")

    # (b) When T = 40?
    # 40 = 22 + 73*e^(-k*t) => e^(-k*t) = 18/73 => t = -ln(18/73)/k
    t_40 = -np.log(18/73) / k_val
    print(f"\n(b) When T = 40C:")
    print(f"    40 = 22 + 73*e^(-k*t)")
    print(f"    e^(-k*t) = 18/73")
    print(f"    t = -ln(18/73)/k = {t_40:.2f} minutes")
    print(f"    = {t_40/60:.2f} hours")

    # (c) Plot with solve_ivp
    def cooling_ode(t, T):
        return [-k_val * (T[0] - T_room)]

    t_span = [0, 60]
    t_eval = np.linspace(0, 60, 500)
    sol = solve_ivp(cooling_ode, t_span, [T_0], t_eval=t_eval, method='RK45')

    # Analytical
    T_analytical = T_room + (T_0 - T_room) * np.exp(-k_val * t_eval)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sol.t, sol.y[0], 'b-', linewidth=2, label='solve_ivp (numerical)')
    ax.plot(t_eval, T_analytical, 'r--', linewidth=2, label='Analytical')
    ax.axhline(y=40, color='g', linestyle=':', label=f'T = 40C at t = {t_40:.1f} min')
    ax.axhline(y=T_room, color='gray', linestyle='--', alpha=0.5, label=f'Room temp = {T_room}C')
    ax.plot(5, 70, 'ko', markersize=8, label='Data point: T(5) = 70C')
    ax.plot(t_40, 40, 'gs', markersize=10)
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Temperature (C)', fontsize=12)
    ax.set_title("Newton's Law of Cooling: Coffee Cooling", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex12_newton_cooling.png', dpi=150, bbox_inches='tight')
    plt.close()

    max_err = np.max(np.abs(sol.y[0] - T_analytical))
    print(f"\n(c) [Plot saved: ex12_newton_cooling.png]")
    print(f"    Max difference (numerical vs analytical): {max_err:.2e} C")


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
    print("All exercises for Lesson 12 completed.")
    print("=" * 60)
