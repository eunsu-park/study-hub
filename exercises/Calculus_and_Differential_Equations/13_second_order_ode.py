"""
Exercise Solutions: Lesson 13 - Second-Order Ordinary Differential Equations
Calculus and Differential Equations

Topics covered:
- Homogeneous IVP with constant coefficients
- Nonhomogeneous with repeated roots (undetermined coefficients)
- Variation of parameters (y'' + y = tan(x))
- Spring-mass system (underdamped)
- Forced oscillator resonance analysis
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Homogeneous IVP
# ============================================================
def exercise_1():
    """
    y'' + 6y' + 8y = 0, y(0) = 2, y'(0) = -1.
    """
    print("=" * 60)
    print("Problem 1: Homogeneous IVP")
    print("=" * 60)

    x = sp.Symbol('x')
    y = sp.Function('y')

    # Characteristic equation: r^2 + 6r + 8 = 0 => (r+2)(r+4) = 0
    r = sp.Symbol('r')
    char_eq = r**2 + 6*r + 8
    roots = sp.solve(char_eq, r)

    print(f"\n  y'' + 6y' + 8y = 0")
    print(f"  Characteristic equation: r^2 + 6r + 8 = 0")
    print(f"  Roots: {roots} (real distinct)")
    print(f"  General solution: y = C1*e^(-2x) + C2*e^(-4x)")

    # Apply ICs
    ode = sp.Eq(y(x).diff(x, 2) + 6*y(x).diff(x) + 8*y(x), 0)
    sol = sp.dsolve(ode, y(x), ics={y(0): 2, y(x).diff(x).subs(x, 0): -1})
    print(f"\n  SymPy solution with ICs: {sol}")

    # Manual: C1 + C2 = 2, -2*C1 - 4*C2 = -1
    # From first: C1 = 2 - C2; substituting: -2(2-C2) - 4C2 = -1
    # -4 + 2C2 - 4C2 = -1 => -2C2 = 3 => C2 = -3/2, C1 = 7/2
    print(f"\n  Manual calculation:")
    print(f"    C1 + C2 = 2")
    print(f"    -2*C1 - 4*C2 = -1")
    print(f"    C1 = 7/2, C2 = -3/2")
    print(f"    y(x) = (7/2)*e^(-2x) - (3/2)*e^(-4x)")

    # Classification: both roots negative => overdamped decay
    print(f"\n  Both roots negative and real => overdamped decay to 0")

    # Plot
    x_vals = np.linspace(0, 3, 200)
    y_vals = 3.5 * np.exp(-2*x_vals) - 1.5 * np.exp(-4*x_vals)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label=r'$y = \frac{7}{2}e^{-2x} - \frac{3}{2}e^{-4x}$')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Solution: Overdamped Second-Order ODE', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex13_homogeneous_ivp.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex13_homogeneous_ivp.png]")


# ============================================================
# Problem 2: Nonhomogeneous with Repeated Roots
# ============================================================
def exercise_2():
    """
    y'' + 4y' + 4y = 3*e^(-2x).
    Homogeneous has repeated root r = -2; need y_p = A*x^2*e^(-2x).
    """
    print("\n" + "=" * 60)
    print("Problem 2: Nonhomogeneous with Repeated Roots")
    print("=" * 60)

    x = sp.Symbol('x')
    y = sp.Function('y')

    # Characteristic: r^2 + 4r + 4 = (r+2)^2 = 0 => r = -2 (repeated)
    print(f"\n  y'' + 4y' + 4y = 3*e^(-2x)")
    print(f"  Characteristic: (r+2)^2 = 0 => r = -2 (repeated)")
    print(f"  Homogeneous: y_h = (C1 + C2*x)*e^(-2x)")
    print(f"\n  For particular: e^(-2x) duplicates both y_1=e^(-2x) and y_2=x*e^(-2x)")
    print(f"  Must try y_p = A*x^2*e^(-2x)")

    # Substitution to find A:
    # y_p = A*x^2*e^(-2x)
    # y_p' = A*(2x - 2x^2)*e^(-2x)
    # y_p'' = A*(2 - 8x + 4x^2)*e^(-2x)
    # y_p'' + 4y_p' + 4y_p = A*(2-8x+4x^2+8x-8x^2+4x^2)*e^(-2x) = 2A*e^(-2x)
    # 2A = 3 => A = 3/2
    print(f"\n  Substituting y_p = A*x^2*e^(-2x):")
    print(f"  y_p'' + 4y_p' + 4y_p = 2A*e^(-2x)")
    print(f"  2A = 3 => A = 3/2")
    print(f"\n  General solution: y = (C1 + C2*x)*e^(-2x) + (3/2)*x^2*e^(-2x)")
    print(f"                    = (C1 + C2*x + (3/2)*x^2)*e^(-2x)")

    # SymPy verification
    ode = sp.Eq(y(x).diff(x, 2) + 4*y(x).diff(x) + 4*y(x), 3*sp.exp(-2*x))
    sol = sp.dsolve(ode, y(x))
    print(f"\n  SymPy: {sol}")


# ============================================================
# Problem 3: Variation of Parameters
# ============================================================
def exercise_3():
    """
    y'' + y = tan(x). Use variation of parameters.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Variation of Parameters")
    print("=" * 60)

    x = sp.Symbol('x')
    y = sp.Function('y')

    print(f"\n  y'' + y = tan(x)")
    print(f"  Homogeneous: y'' + y = 0 => y_h = C1*cos(x) + C2*sin(x)")
    print(f"  y1 = cos(x), y2 = sin(x)")
    print(f"\n  Wronskian: W = y1*y2' - y2*y1' = cos^2(x) + sin^2(x) = 1")
    print(f"\n  Variation of parameters:")
    print(f"  u1' = -y2*g(x)/W = -sin(x)*tan(x) = -sin^2(x)/cos(x)")
    print(f"  u2' = y1*g(x)/W = cos(x)*tan(x) = sin(x)")

    # u1 = integral -sin^2(x)/cos(x) dx = integral (cos(x) - sec(x)) dx
    #    = sin(x) - ln|sec(x) + tan(x)|
    # u2 = integral sin(x) dx = -cos(x)
    u1 = sp.integrate(-sp.sin(x)**2/sp.cos(x), x)
    u2 = sp.integrate(sp.sin(x), x)

    print(f"\n  u1 = integral -sin^2(x)/cos(x) dx")
    print(f"     = integral (cos(x) - sec(x)) dx  [using sin^2 = 1-cos^2]")
    print(f"     = {u1}")
    print(f"  u2 = integral sin(x) dx = {u2}")

    # Particular solution
    y_p = u1 * sp.cos(x) + u2 * sp.sin(x)
    y_p_simplified = sp.simplify(y_p)
    print(f"\n  y_p = u1*cos(x) + u2*sin(x)")
    print(f"      = {sp.simplify(y_p)}")

    # SymPy full solution
    ode = sp.Eq(y(x).diff(x, 2) + y(x), sp.tan(x))
    sol = sp.dsolve(ode, y(x), hint='variation_of_parameters')
    print(f"\n  SymPy (variation of parameters): {sol}")


# ============================================================
# Problem 4: Spring-Mass System
# ============================================================
def exercise_4():
    """
    m=2 kg, k=50 N/m, gamma=4 kg/s.
    (a) Classification (underdamped/critical/overdamped)
    (b) Solution with x(0)=0.1, x'(0)=0
    (c) Plot with solve_ivp overlay
    (d) Time for amplitude to drop to 1%
    """
    print("\n" + "=" * 60)
    print("Problem 4: Spring-Mass System")
    print("=" * 60)

    m, k, gamma = 2, 50, 4

    # Equation: m*x'' + gamma*x' + k*x = 0 => 2x'' + 4x' + 50x = 0
    # Characteristic: 2r^2 + 4r + 50 = 0 => r = (-4 +/- sqrt(16-400))/4
    discriminant = gamma**2 - 4*m*k
    omega_0 = np.sqrt(k/m)  # natural frequency
    zeta = gamma / (2*np.sqrt(m*k))  # damping ratio

    print(f"\n  2x'' + 4x' + 50x = 0")
    print(f"  omega_0 = sqrt(k/m) = sqrt({k}/{m}) = {omega_0:.4f} rad/s")
    print(f"  Damping ratio zeta = gamma/(2*sqrt(m*k)) = {gamma}/(2*sqrt({m}*{k})) = {zeta:.4f}")

    # (a) Classification
    if discriminant < 0:
        classification = "UNDERDAMPED"
    elif discriminant == 0:
        classification = "CRITICALLY DAMPED"
    else:
        classification = "OVERDAMPED"
    print(f"\n(a) Discriminant = gamma^2 - 4mk = {discriminant}")
    print(f"    System is {classification}")

    # (b) Solution: x(t) = e^(-t) * (A*cos(omega_d*t) + B*sin(omega_d*t))
    alpha = gamma / (2*m)  # decay rate = 1
    omega_d = np.sqrt(k/m - (gamma/(2*m))**2)  # damped frequency

    print(f"\n(b) alpha = gamma/(2m) = {alpha} s^-1")
    print(f"    omega_d = sqrt(k/m - alpha^2) = sqrt({k/m} - {alpha**2}) = {omega_d:.4f} rad/s")
    print(f"    General: x(t) = e^(-{alpha}t)*(A*cos({omega_d:.4f}t) + B*sin({omega_d:.4f}t))")

    # ICs: x(0) = 0.1 => A = 0.1
    # x'(0) = 0 => -alpha*A + omega_d*B = 0 => B = alpha*A/omega_d
    A = 0.1
    B = alpha * A / omega_d
    print(f"    x(0) = 0.1 => A = {A}")
    print(f"    x'(0) = 0 => B = alpha*A/omega_d = {B:.6f}")

    # (c) Plot
    t_vals = np.linspace(0, 6, 1000)
    x_analytical = np.exp(-alpha*t_vals) * (A*np.cos(omega_d*t_vals) + B*np.sin(omega_d*t_vals))

    # Numerical solution
    def spring_ode(t, state):
        x, v = state
        return [v, -(gamma*v + k*x)/m]

    sol = solve_ivp(spring_ode, [0, 6], [0.1, 0], t_eval=t_vals, method='RK45')
    envelope = A * np.sqrt(1 + (alpha/omega_d)**2) * np.exp(-alpha*t_vals)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t_vals, x_analytical, 'b-', linewidth=2, label='Analytical')
    ax.plot(sol.t, sol.y[0], 'r--', linewidth=1.5, label='solve_ivp')
    ax.plot(t_vals, envelope, 'g:', linewidth=1.5, label='Envelope')
    ax.plot(t_vals, -envelope, 'g:', linewidth=1.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Displacement (m)', fontsize=12)
    ax.set_title('Underdamped Spring-Mass System', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex13_spring_mass.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n(c) [Plot saved: ex13_spring_mass.png]")

    # (d) 1% of initial: amplitude envelope = A_0 * e^(-alpha*t) = 0.01 * A_0
    # e^(-alpha*t) = 0.01 => t = -ln(0.01)/alpha
    t_1percent = -np.log(0.01) / alpha
    print(f"\n(d) Time for amplitude to drop to 1%:")
    print(f"    e^(-{alpha}*t) = 0.01 => t = ln(100)/{alpha} = {t_1percent:.4f} s")


# ============================================================
# Problem 5: Forced Oscillator Resonance
# ============================================================
def exercise_5():
    """
    x'' + 0.4*x' + 25*x = 5*cos(omega*t).
    (a) Natural frequency and Q-factor
    (b) Resonant driving frequency
    (c) Amplitude and phase plots
    (d) Simulate at resonance vs far from resonance
    """
    print("\n" + "=" * 60)
    print("Problem 5: Forced Oscillator Resonance")
    print("=" * 60)

    gamma_coeff = 0.4
    omega_0_sq = 25
    omega_0 = np.sqrt(omega_0_sq)
    F0 = 5

    # (a) Natural frequency and Q-factor
    Q = omega_0 / gamma_coeff  # Q = omega_0 / (2*zeta*omega_0) = 1/(2*zeta)
    # Actually Q = omega_0*m/gamma for x'' + (gamma/m)x' + omega_0^2*x = ...
    # Here m=1: Q = omega_0/gamma = 5/0.4 = 12.5
    print(f"\n  x'' + 0.4*x' + 25*x = 5*cos(omega*t)")
    print(f"\n(a) omega_0 = sqrt(25) = {omega_0}")
    print(f"    Q = omega_0 / gamma = {omega_0} / {gamma_coeff} = {Q}")

    # (b) Resonant frequency (for amplitude)
    # omega_max = omega_0 * sqrt(1 - 1/(2*Q^2)) for displacement
    # = omega_0 * sqrt(1 - gamma^2/(2*omega_0^2))
    omega_res = omega_0 * np.sqrt(1 - gamma_coeff**2 / (2*omega_0_sq))
    print(f"\n(b) Resonant driving frequency (max amplitude):")
    print(f"    omega_res = omega_0 * sqrt(1 - gamma^2/(2*omega_0^2))")
    print(f"              = {omega_0} * sqrt(1 - {gamma_coeff**2}/(2*{omega_0_sq}))")
    print(f"              = {omega_res:.6f} rad/s")
    print(f"    (Very close to omega_0 = {omega_0} since damping is light)")

    # (c) Amplitude and phase plots
    omega_range = np.linspace(0.1, 10, 1000)
    # Amplitude: A(omega) = F0 / sqrt((omega_0^2 - omega^2)^2 + (gamma*omega)^2)
    A_omega = F0 / np.sqrt((omega_0_sq - omega_range**2)**2 + (gamma_coeff*omega_range)**2)
    # Phase: delta = arctan(gamma*omega / (omega_0^2 - omega^2))
    phase_omega = np.arctan2(gamma_coeff*omega_range, omega_0_sq - omega_range**2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(omega_range, A_omega, 'b-', linewidth=2)
    ax1.axvline(x=omega_res, color='r', linestyle='--', alpha=0.7, label=f'Resonance: $\\omega$ = {omega_res:.2f}')
    ax1.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
    ax1.set_ylabel('Amplitude A', fontsize=12)
    ax1.set_title('Amplitude Response', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(omega_range, np.degrees(phase_omega), 'b-', linewidth=2)
    ax2.axvline(x=omega_0, color='r', linestyle='--', alpha=0.7, label=f'$\\omega_0$ = {omega_0}')
    ax2.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
    ax2.set_ylabel(r'Phase $\delta$ (degrees)', fontsize=12)
    ax2.set_title('Phase Response', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex13_resonance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n(c) [Plot saved: ex13_resonance.png]")

    # (d) Simulate at resonance and far from resonance
    def forced_oscillator(t, state, omega):
        x, v = state
        return [v, F0*np.cos(omega*t) - gamma_coeff*v - omega_0_sq*x]

    t_sim = np.linspace(0, 50, 5000)

    # At resonance
    sol_res = solve_ivp(lambda t, s: forced_oscillator(t, s, omega_res),
                        [0, 50], [0, 0], t_eval=t_sim, method='RK45')
    # Far from resonance (omega = 1)
    sol_far = solve_ivp(lambda t, s: forced_oscillator(t, s, 1.0),
                        [0, 50], [0, 0], t_eval=t_sim, method='RK45')

    A_res_theory = F0 / np.sqrt((omega_0_sq - omega_res**2)**2 + (gamma_coeff*omega_res)**2)
    A_far_theory = F0 / np.sqrt((omega_0_sq - 1)**2 + (gamma_coeff*1)**2)

    # Steady-state amplitudes (from last portion)
    A_res_num = (np.max(sol_res.y[0, -2000:]) - np.min(sol_res.y[0, -2000:])) / 2
    A_far_num = (np.max(sol_far.y[0, -2000:]) - np.min(sol_far.y[0, -2000:])) / 2

    print(f"\n(d) Steady-state amplitudes:")
    print(f"    At resonance (omega={omega_res:.2f}):")
    print(f"      Theory: A = {A_res_theory:.6f}")
    print(f"      Numerical: A ~ {A_res_num:.6f}")
    print(f"    Far from resonance (omega=1):")
    print(f"      Theory: A = {A_far_theory:.6f}")
    print(f"      Numerical: A ~ {A_far_num:.6f}")
    print(f"    Amplitude ratio: {A_res_theory/A_far_theory:.2f}x")


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
    print("All exercises for Lesson 13 completed.")
    print("=" * 60)
