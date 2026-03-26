"""
Exercise Solutions: Lesson 15 - Laplace Transform for ODE
Calculus and Differential Equations

Topics covered:
- Basic Laplace transforms using properties
- Inverse transform via partial fractions
- IVP solving with repeated roots
- Step function spring response
- Impulse response and convolution
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Basic Transforms
# ============================================================
def exercise_1():
    """
    Compute L{t^3 * e^(-2t)} and L{e^(3t) * cos(4t)} using properties.
    """
    print("=" * 60)
    print("Problem 1: Basic Laplace Transforms")
    print("=" * 60)

    t, s = sp.symbols('t s', positive=True)

    # (a) L{t^3 * e^(-2t)}
    # Use first shifting: L{e^{at}f(t)} = F(s-a)
    # L{t^3} = 3!/s^4 = 6/s^4
    # L{t^3 * e^{-2t}} = 6/(s+2)^4
    expr_a = t**3 * sp.exp(-2*t)
    result_a = sp.laplace_transform(expr_a, t, s, noconds=True)

    print(f"\n(a) L{{t^3 * e^(-2t)}}")
    print(f"    L{{t^n}} = n!/s^(n+1), so L{{t^3}} = 6/s^4")
    print(f"    First shifting theorem: L{{e^(at)*f(t)}} = F(s-a)")
    print(f"    L{{t^3 * e^(-2t)}} = 6/(s+2)^4")
    print(f"    SymPy: {result_a}")

    # (b) L{e^(3t) * cos(4t)}
    # L{cos(4t)} = s/(s^2 + 16)
    # L{e^(3t) * cos(4t)} = (s-3)/((s-3)^2 + 16)
    expr_b = sp.exp(3*t) * sp.cos(4*t)
    result_b = sp.laplace_transform(expr_b, t, s, noconds=True)

    print(f"\n(b) L{{e^(3t) * cos(4t)}}")
    print(f"    L{{cos(bt)}} = s/(s^2 + b^2), so L{{cos(4t)}} = s/(s^2 + 16)")
    print(f"    First shifting: L{{e^(3t)*cos(4t)}} = (s-3)/((s-3)^2 + 16)")
    print(f"    SymPy: {result_b}")


# ============================================================
# Problem 2: Inverse Transform
# ============================================================
def exercise_2():
    """
    L^{-1}{(3s + 7) / ((s+1)(s^2 + 4))}.
    Partial fractions, identify transient and oscillatory parts.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Inverse Laplace Transform")
    print("=" * 60)

    s, t = sp.symbols('s t')

    F = (3*s + 7) / ((s + 1) * (s**2 + 4))

    # Partial fractions: A/(s+1) + (Bs + C)/(s^2 + 4)
    pf = sp.apart(F, s)
    print(f"\n  F(s) = (3s + 7) / ((s+1)(s^2 + 4))")
    print(f"  Partial fractions: {pf}")

    # Inverse transform
    f_t = sp.inverse_laplace_transform(F, s, t)
    f_t_simplified = sp.simplify(f_t)
    print(f"\n  f(t) = L^-1{{F(s)}} = {f_t_simplified}")

    # Manual decomposition:
    # (3s+7) = A(s^2+4) + (Bs+C)(s+1)
    # s=-1: 4 = 5A => A = 4/5
    # s=0: 7 = 4A + C => C = 7 - 16/5 = 19/5
    # s=1: 10 = 5A + 2(B+C) => 10 = 4 + 2B + 38/5 => 2B = 10 - 4 - 38/5 = -8/5 => B = -4/5 ... wait
    # Let me let SymPy do it right
    A, B, C_coeff = sp.symbols('A B C')
    eq = sp.Eq(3*s + 7, A*(s**2 + 4) + (B*s + C_coeff)*(s + 1))
    coeffs = sp.solve(eq.subs(s, -1), A)
    print(f"\n  Manual partial fractions:")
    print(f"  Decomposition: {pf}")

    # Identify parts
    print(f"\n  Transient part: terms with e^(-t) (decay to 0)")
    print(f"  Oscillatory part: terms with sin(2t), cos(2t) (persist forever)")


# ============================================================
# Problem 3: IVP with Repeated Root
# ============================================================
def exercise_3():
    """
    y'' + 4y' + 4y = e^(-2t), y(0) = 0, y'(0) = 1.
    """
    print("\n" + "=" * 60)
    print("Problem 3: IVP Solving with Laplace Transform")
    print("=" * 60)

    s, t = sp.symbols('s t')

    print(f"\n  y'' + 4y' + 4y = e^(-2t), y(0) = 0, y'(0) = 1")

    # L{y''} = s^2*Y - s*y(0) - y'(0) = s^2*Y - 1
    # L{y'} = s*Y - y(0) = s*Y
    # L{e^{-2t}} = 1/(s+2)
    # (s^2*Y - 1) + 4*(s*Y) + 4*Y = 1/(s+2)
    # Y*(s^2 + 4s + 4) = 1 + 1/(s+2)
    # Y*(s+2)^2 = 1 + 1/(s+2) = (s+3)/(s+2)
    # Y = (s+3)/((s+2)^3)

    print(f"\n  Taking Laplace transform:")
    print(f"  (s^2*Y - 1) + 4*(s*Y) + 4*Y = 1/(s+2)")
    print(f"  Y*(s+2)^2 = 1 + 1/(s+2) = (s+3)/(s+2)")
    print(f"  Y(s) = (s+3)/(s+2)^3")

    # Partial fractions
    Y = (s + 3) / (s + 2)**3
    pf = sp.apart(Y, s)
    print(f"\n  Y(s) = {Y}")
    print(f"  Partial fractions: {pf}")

    # Inverse transform
    y_t = sp.inverse_laplace_transform(Y, s, t)
    print(f"\n  y(t) = L^-1{{Y(s)}} = {sp.simplify(y_t)}")

    # SymPy full solution for verification
    y_func = sp.Function('y')
    ode = sp.Eq(y_func(t).diff(t, 2) + 4*y_func(t).diff(t) + 4*y_func(t), sp.exp(-2*t))
    sol = sp.dsolve(ode, y_func(t), ics={y_func(0): 0, y_func(t).diff(t).subs(t, 0): 1})
    print(f"\n  SymPy dsolve verification: {sol}")

    # Note about repeated root
    print(f"\n  Note: (s+2)^3 in the denominator means the partial fractions")
    print(f"  include terms A/(s+2), B/(s+2)^2, C/(s+2)^3, which correspond")
    print(f"  to e^(-2t), t*e^(-2t), t^2*e^(-2t)/2 in the time domain.")


# ============================================================
# Problem 4: Step Function Spring
# ============================================================
def exercise_4():
    """
    y'' + 9y = 5*u(t - pi), y(0) = 0, y'(0) = 0.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Step Function Spring Response")
    print("=" * 60)

    s, t = sp.symbols('s t')

    print(f"\n  y'' + 9y = 5*u(t - pi), y(0) = 0, y'(0) = 0")
    print(f"\n  L{{u(t-pi)}} = e^(-pi*s)/s")
    print(f"  L{{y''}} + 9*L{{y}} = 5*e^(-pi*s)/s")
    print(f"  s^2*Y + 9*Y = 5*e^(-pi*s)/s")
    print(f"  Y(s) = 5*e^(-pi*s) / (s*(s^2+9))")

    # Partial fractions of 5/(s*(s^2+9)):
    F = 5 / (s * (s**2 + 9))
    pf = sp.apart(F, s)
    print(f"\n  5/(s*(s^2+9)) = {pf}")
    print(f"  = 5/9 * (1/s - s/(s^2+9))")

    # Inverse of 5/(s*(s^2+9)):
    f_t = sp.inverse_laplace_transform(F, s, t)
    print(f"\n  L^-1{{5/(s(s^2+9))}} = {sp.simplify(f_t)}")
    print(f"  = (5/9)*(1 - cos(3t))")

    # With the time shift:
    # y(t) = (5/9)*(1 - cos(3(t-pi)))*u(t-pi)
    print(f"\n  Final solution:")
    print(f"  y(t) = (5/9)*(1 - cos(3(t-pi)))*u(t-pi)")
    print(f"       = 0                            for t < pi")
    print(f"       = (5/9)*(1 - cos(3(t-pi)))     for t >= pi")
    print(f"\n  Physical interpretation:")
    print(f"  No motion until t=pi, then the constant force 5 is applied,")
    print(f"  causing oscillation about the new equilibrium y = 5/9.")

    # Numerical verification and plot
    def spring_ode(t_val, state):
        y_v, v = state
        force = 5.0 if t_val >= np.pi else 0.0
        return [v, force - 9*y_v]

    t_eval = np.linspace(0, 10, 2000)
    sol = solve_ivp(spring_ode, [0, 10], [0, 0], t_eval=t_eval, method='RK45',
                    max_step=0.01)

    # Analytical
    y_analytical = np.where(sol.t >= np.pi,
                            (5.0/9)*(1 - np.cos(3*(sol.t - np.pi))),
                            0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sol.t, sol.y[0], 'b-', linewidth=2, label='Numerical (solve_ivp)')
    ax.plot(sol.t, y_analytical, 'r--', linewidth=1.5, label='Analytical')
    ax.axvline(x=np.pi, color='g', linestyle=':', alpha=0.7, label=f't = pi (step applied)')
    ax.axhline(y=5/9, color='gray', linestyle='--', alpha=0.5, label='New equilibrium y=5/9')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('y(t)', fontsize=12)
    ax.set_title('Step Function Spring Response', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex15_step_spring.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: ex15_step_spring.png]")


# ============================================================
# Problem 5: Impulse Response and Convolution
# ============================================================
def exercise_5():
    """
    y'' + 2y' + 5y = delta(t), y(0^-) = 0, y'(0^-) = 0.
    Find h(t), then convolve with f(t) = e^{-t}.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Impulse Response and Convolution")
    print("=" * 60)

    s, t, tau = sp.symbols('s t tau', positive=True)

    # Impulse response: L{delta(t)} = 1
    # s^2*H + 2s*H + 5*H = 1
    # H(s) = 1/(s^2 + 2s + 5) = 1/((s+1)^2 + 4)
    print(f"\n  y'' + 2y' + 5y = delta(t)")
    print(f"  L{{delta(t)}} = 1")
    print(f"  H(s) = 1/(s^2 + 2s + 5) = 1/((s+1)^2 + 4)")

    H = 1 / (s**2 + 2*s + 5)
    h_t = sp.inverse_laplace_transform(H, s, t)
    print(f"\n  h(t) = L^-1{{H(s)}} = {h_t}")
    print(f"       = (1/2)*e^(-t)*sin(2t)  [damped sinusoid]")

    # Response to f(t) = e^(-t) via convolution:
    # y(t) = h * f = integral_0^t h(tau)*f(t-tau) d(tau)
    # = integral_0^t (1/2)*e^(-tau)*sin(2*tau) * e^(-(t-tau)) d(tau)
    # = (1/2)*e^(-t) * integral_0^t sin(2*tau) d(tau)
    # = (1/2)*e^(-t) * [-cos(2*tau)/2]_0^t
    # = (1/4)*e^(-t)*(1 - cos(2t))

    print(f"\n  Response to f(t) = e^(-t) via convolution:")
    print(f"  y(t) = integral_0^t h(tau)*f(t-tau) d(tau)")
    print(f"       = integral_0^t (1/2)*e^(-tau)*sin(2tau)*e^(-(t-tau)) d(tau)")
    print(f"       = (1/2)*e^(-t) * integral_0^t sin(2tau) d(tau)")
    print(f"       = (1/2)*e^(-t) * [-cos(2tau)/2]_0^t")
    print(f"       = (1/4)*e^(-t)*(1 - cos(2t))")

    # SymPy convolution
    h_expr = sp.Rational(1, 2) * sp.exp(-tau) * sp.sin(2*tau)
    f_expr = sp.exp(-(t - tau))
    conv = sp.integrate(h_expr * f_expr, (tau, 0, t))
    conv_simplified = sp.simplify(conv)
    print(f"\n  SymPy convolution: {conv_simplified}")

    # Verify by solving the ODE directly with f(t) = e^(-t)
    print(f"\n  Verification via direct Laplace transform:")
    print(f"  L{{e^(-t)}} = 1/(s+1)")
    print(f"  Y(s) = H(s)*F(s) = 1/((s+1)^2+4) * 1/(s+1) = 1/((s+1)^3+4(s+1))")
    Y_direct = H * 1/(s+1)
    y_direct = sp.inverse_laplace_transform(Y_direct, s, t)
    y_direct_simplified = sp.simplify(y_direct)
    print(f"  y(t) = {y_direct_simplified}")

    # Plot
    t_vals = np.linspace(0, 10, 500)
    h_vals = 0.5 * np.exp(-t_vals) * np.sin(2*t_vals)
    y_conv = 0.25 * np.exp(-t_vals) * (1 - np.cos(2*t_vals))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(t_vals, h_vals, 'b-', linewidth=2)
    ax1.set_xlabel('t', fontsize=12)
    ax1.set_ylabel('h(t)', fontsize=12)
    ax1.set_title('Impulse Response $h(t) = \\frac{1}{2}e^{-t}\\sin(2t)$', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_vals, y_conv, 'r-', linewidth=2, label='$h * e^{-t}$')
    ax2.plot(t_vals, np.exp(-t_vals), 'g--', linewidth=1.5, label='$f(t) = e^{-t}$')
    ax2.set_xlabel('t', fontsize=12)
    ax2.set_ylabel('y(t)', fontsize=12)
    ax2.set_title('Convolution Response', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex15_impulse_convolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: ex15_impulse_convolution.png]")


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
    print("All exercises for Lesson 15 completed.")
    print("=" * 60)
