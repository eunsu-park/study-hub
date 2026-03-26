"""
Exercises for Lesson 15: Laplace Transform
Topic: Mathematical_Methods
Solutions to practice problems from the lesson.
"""

import numpy as np
import sympy as sp
from scipy import signal
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Problem 1: Find Laplace transforms.
    (a) f(t) = 3t^2 - 2e^{-t} + 5cos(4t)
    (b) f(t) = t^3 e^{2t}
    (c) f(t) = e^{-3t} sin(5t)
    """
    print("=" * 60)
    print("Problem 1: Laplace Transforms")
    print("=" * 60)

    t, s = sp.symbols('t s', positive=True)

    print("\n(a) f(t) = 3t^2 - 2e^{-t} + 5cos(4t)")
    f_a = 3 * t**2 - 2 * sp.exp(-t) + 5 * sp.cos(4 * t)
    F_a = sp.laplace_transform(f_a, t, s, noconds=True)
    print(f"  L{{3t^2}} = 3*2!/s^3 = 6/s^3")
    print(f"  L{{-2e^(-t)}} = -2/(s+1)")
    print(f"  L{{5cos(4t)}} = 5s/(s^2+16)")
    print(f"  F(s) = {F_a}")

    print("\n(b) f(t) = t^3 e^{2t}")
    f_b = t**3 * sp.exp(2 * t)
    F_b = sp.laplace_transform(f_b, t, s, noconds=True)
    print(f"  Using s-shift: L{{t^3}} = 6/s^4, replace s -> s-2")
    print(f"  F(s) = 6/(s-2)^4")
    print(f"  SymPy: {F_b}")

    print("\n(c) f(t) = e^{-3t} sin(5t)")
    f_c = sp.exp(-3 * t) * sp.sin(5 * t)
    F_c = sp.laplace_transform(f_c, t, s, noconds=True)
    print(f"  Using s-shift: L{{sin(5t)}} = 5/(s^2+25), replace s -> s+3")
    print(f"  F(s) = 5/((s+3)^2+25)")
    print(f"  SymPy: {F_c}")


def exercise_2():
    """
    Problem 2: Inverse Laplace transforms.
    (a) F(s) = 5/s^3
    (b) F(s) = (2s+1)/(s^2+4s+13)
    (c) F(s) = 3/((s-1)(s+2)(s-3))
    """
    print("\n" + "=" * 60)
    print("Problem 2: Inverse Laplace Transforms")
    print("=" * 60)

    s, t = sp.symbols('s t')

    print("\n(a) F(s) = 5/s^3")
    F_a = 5 / s**3
    f_a = sp.inverse_laplace_transform(F_a, s, t)
    print(f"  L^-1{{n!/s^(n+1)}} = t^n, so L^-1{{5/s^3}} = 5/2 * t^2")
    print(f"  f(t) = {f_a}")

    print("\n(b) F(s) = (2s+1)/(s^2+4s+13)")
    F_b = (2 * s + 1) / (s**2 + 4 * s + 13)
    f_b = sp.inverse_laplace_transform(F_b, s, t)
    print(f"  Complete the square: (s+2)^2 + 9")
    print(f"  = 2(s+2)/((s+2)^2+9) - 3/(((s+2)^2+9)")
    print(f"  = 2*e^(-2t)*cos(3t) - e^(-2t)*sin(3t)")
    print(f"  f(t) = {f_b}")

    print("\n(c) F(s) = 3/((s-1)(s+2)(s-3))")
    F_c = 3 / ((s - 1) * (s + 2) * (s - 3))
    pf_c = sp.apart(F_c, s)
    f_c = sp.inverse_laplace_transform(F_c, s, t)
    print(f"  Partial fractions: {pf_c}")
    print(f"  f(t) = {f_c}")


def exercise_3():
    """
    Problem 3: Convolution theorem for L^{-1}{1/(s(s+1))}.
    """
    print("\n" + "=" * 60)
    print("Problem 3: Convolution Theorem")
    print("=" * 60)

    s, t, tau = sp.symbols('s t tau')

    print("\nF(s) = 1/(s(s+1)) = F1(s)*F2(s)")
    print("where F1 = 1/s -> f1 = 1, F2 = 1/(s+1) -> f2 = e^{-t}")
    print("\nBy convolution theorem:")
    print("f(t) = (f1 * f2)(t) = int_0^t 1 * e^{-(t-tau)} dtau")

    conv = sp.integrate(sp.exp(-(t - tau)), (tau, 0, t))
    print(f"     = {sp.simplify(conv)}")
    print(f"     = 1 - e^{-t}")

    # Verify via partial fractions
    F = 1 / (s * (s + 1))
    f_pf = sp.inverse_laplace_transform(F, s, t)
    print(f"\nVerification (partial fractions): {f_pf}")


def exercise_4():
    """
    Problem 4: Solve y'' - 4y' + 4y = e^{2t}, y(0)=0, y'(0)=1.
    """
    print("\n" + "=" * 60)
    print("Problem 4: ODE via Laplace Transform")
    print("=" * 60)

    s, t = sp.symbols('s t')

    print("\ny'' - 4y' + 4y = e^{2t}, y(0)=0, y'(0)=1")
    print("\nLaplace transform:")
    print("  [s^2 Y - s*0 - 1] - 4[sY - 0] + 4Y = 1/(s-2)")
    print("  (s^2 - 4s + 4)Y = 1 + 1/(s-2)")
    print("  (s-2)^2 Y = 1 + 1/(s-2)")
    print("  Y(s) = 1/(s-2)^2 + 1/(s-2)^3")

    Y = 1 / (s - 2)**2 + 1 / (s - 2)**3
    y_sol = sp.inverse_laplace_transform(Y, s, t)
    print(f"\n  Y(s) = {Y}")
    print(f"  y(t) = {y_sol}")

    # Verify
    y_func = sp.Function('y')
    ode = sp.Eq(y_func(t).diff(t, 2) - 4 * y_func(t).diff(t) + 4 * y_func(t),
                sp.exp(2 * t))
    sol = sp.dsolve(ode, y_func(t),
                    ics={y_func(0): 0, y_func(t).diff(t).subs(t, 0): 1})
    print(f"\n  Verification (dsolve): {sol}")


def exercise_5():
    """
    Problem 5: Second shifting theorem for piecewise function.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Second Shifting Theorem")
    print("=" * 60)

    s, t = sp.symbols('s t', positive=True)

    print("\nf(t) = 0 for 0<=t<2, (t-2) for t>=2")
    print("     = (t-2)*u(t-2)")
    print("\nBy second shifting theorem:")
    print("  L{f(t-a)*u(t-a)} = e^{-as} F(s)")
    print("  Here a=2, f(t) = t, F(s) = 1/s^2")
    print("  L{(t-2)*u(t-2)} = e^{-2s}/s^2")

    f = (t - 2) * sp.Heaviside(t - 2)
    F = sp.laplace_transform(f, t, s, noconds=True)
    print(f"\n  SymPy verification: L{{(t-2)*u(t-2)}} = {F}")


def exercise_6():
    """
    Problem 6: Coupled system x'+y=e^t, x+y'=0, x(0)=1, y(0)=0.
    """
    print("\n" + "=" * 60)
    print("Problem 6: Coupled ODE System")
    print("=" * 60)

    s, t = sp.symbols('s t')

    print("\nx' + y = e^t,  x + y' = 0")
    print("x(0)=1, y(0)=0")
    print("\nLaplace transform:")
    print("  sX - 1 + Y = 1/(s-1)")
    print("  X + sY = 0")
    print("\nFrom equation 2: X = -sY")
    print("Substitute into equation 1:")
    print("  s(-sY) - 1 + Y = 1/(s-1)")
    print("  (-s^2 + 1)Y = 1 + 1/(s-1)")
    print("  -(s^2-1)Y = (s-1+1)/(s-1) = s/(s-1)")
    print("  -(s+1)(s-1)Y = s/(s-1)")
    print("  Y = -s/((s-1)^2(s+1))")

    Y = -s / ((s - 1)**2 * (s + 1))
    X = -s * Y

    y_sol = sp.inverse_laplace_transform(Y, s, t)
    x_sol = sp.inverse_laplace_transform(X, s, t)

    print(f"\n  Y(s) = {Y}")
    print(f"  X(s) = {sp.simplify(X)}")
    print(f"\n  y(t) = {sp.simplify(y_sol)}")
    print(f"  x(t) = {sp.simplify(x_sol)}")

    # Verify ICs
    print(f"\nVerification:")
    print(f"  x(0) = {sp.simplify(x_sol).subs(t, 0)}")
    print(f"  y(0) = {sp.simplify(y_sol).subs(t, 0)}")


def exercise_7():
    """
    Problem 7: Transfer function H(s) = (s+3)/(s^2+4s+8).
    """
    print("\n" + "=" * 60)
    print("Problem 7: Transfer Function Analysis")
    print("=" * 60)

    s, t = sp.symbols('s t')

    H_expr = (s + 3) / (s**2 + 4 * s + 8)
    print(f"\nH(s) = (s+3)/(s^2+4s+8)")

    # Poles and zeros
    zeros = sp.solve(s + 3, s)
    poles = sp.solve(s**2 + 4 * s + 8, s)
    print(f"\nZeros: {zeros}")
    print(f"Poles: {poles}")
    print(f"Poles = -2 +/- 2i  (in left half-plane)")
    print(f"System is STABLE (all poles have negative real part)")

    # Impulse response
    h_t = sp.inverse_laplace_transform(H_expr, s, t)
    print(f"\nImpulse response h(t) = L^-1{{H(s)}}:")
    print(f"  h(t) = {h_t}")

    # Numerical step response
    num = [1, 3]
    den = [1, 4, 8]
    sys = signal.TransferFunction(num, den)
    t_sim = np.linspace(0, 5, 500)

    _, y_step = signal.step(sys, T=t_sim)
    _, y_imp = signal.impulse(sys, T=t_sim)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(t_sim, y_imp, 'b-', linewidth=2)
    ax1.set_xlabel('t')
    ax1.set_ylabel('h(t)')
    ax1.set_title('Impulse Response')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_sim, y_step, 'r-', linewidth=2)
    ax2.set_xlabel('t')
    ax2.set_ylabel('y(t)')
    ax2.set_title('Step Response')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('H(s) = (s+3)/(s^2+4s+8)', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex15_transfer_function.png', dpi=150)
    plt.close()
    print("Plot saved to ex15_transfer_function.png")


def exercise_8():
    """
    Problem 8: Solve integral equation y(t) = 1 + int_0^t y(tau)sin(t-tau) dtau.
    """
    print("\n" + "=" * 60)
    print("Problem 8: Integral Equation")
    print("=" * 60)

    s, t = sp.symbols('s t')

    print("\ny(t) = 1 + int_0^t y(tau)*sin(t-tau) dtau")
    print("\nLaplace transform (convolution theorem):")
    print("  Y(s) = 1/s + Y(s) * 1/(s^2+1)")
    print("  Y(s) [1 - 1/(s^2+1)] = 1/s")
    print("  Y(s) * s^2/(s^2+1) = 1/s")
    print("  Y(s) = (s^2+1)/s^3 = 1/s + 1/s^3")

    Y = 1 / s + 1 / s**3
    y_sol = sp.inverse_laplace_transform(Y, s, t)
    print(f"\n  Y(s) = {Y}")
    print(f"  y(t) = {y_sol}")
    print(f"       = 1 + t^2/2")


def exercise_9():
    """
    Problem 9: RLC circuit R=4, L=1, C=1/5, V=10u(t).
    """
    print("\n" + "=" * 60)
    print("Problem 9: RLC Circuit")
    print("=" * 60)

    R, L_val, C = 4.0, 1.0, 0.2
    V0 = 10.0
    s, t = sp.symbols('s t')

    omega0 = 1 / np.sqrt(L_val * C)
    zeta = R / (2 * np.sqrt(L_val / C))
    print(f"\nR={R} Ohm, L={L_val} H, C={C} F, V(t)={V0}*u(t)")
    print(f"omega_0 = {omega0:.4f} rad/s")
    print(f"zeta = {zeta:.4f} ({'underdamped' if zeta < 1 else 'overdamped'})")

    # Transfer function for current: I(s) = V(s) / Z(s)
    # Z(s) = Ls + R + 1/(Cs) = (Ls^2 + Rs + 1/C)/s
    # I(s) = V0/s * s / (Ls^2 + Rs + 1/C) = V0 / (Ls^2 + Rs + 1/C)
    print(f"\nI(s) = V0 / (Ls^2 + Rs + 1/C)")
    print(f"     = {V0} / (s^2 + {R / L_val}s + {1 / (L_val * C)})")

    I_s = V0 / (L_val * s**2 + R * s + 1 / C)
    i_t = sp.inverse_laplace_transform(I_s, s, t)
    print(f"\ni(t) = {i_t}")

    # Numerical
    num = [V0 / L_val]
    den = [1, R / L_val, 1 / (L_val * C)]
    sys = signal.TransferFunction(num, den)
    t_sim = np.linspace(0, 5, 500)
    _, i_sim = signal.impulse(sys, T=t_sim)  # impulse of I(s)*s = step of I(s)...
    # Actually for step input: I(s) = V0/(s*(Ls^2+Rs+1/C)) = need step response
    _, i_step = signal.step(signal.TransferFunction([V0], [L_val, R, 1 / C]), T=t_sim)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_sim, i_step, 'b-', linewidth=2)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('i(t) (A)')
    ax.set_title(f'RLC Circuit Current (R={R}, L={L_val}, C={C})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex15_rlc_circuit.png', dpi=150)
    plt.close()
    print("Plot saved to ex15_rlc_circuit.png")


def exercise_10():
    """
    Problem 10: Final value theorem for H(s) = 10(s+2)/(s^2+5s+6).
    """
    print("\n" + "=" * 60)
    print("Problem 10: Final Value Theorem")
    print("=" * 60)

    s = sp.Symbol('s')

    H = 10 * (s + 2) / (s**2 + 5 * s + 6)
    print(f"\nH(s) = 10(s+2)/(s^2+5s+6)")

    # Check stability: poles
    poles = sp.solve(s**2 + 5 * s + 6, s)
    print(f"Poles: {poles}  (both in left half-plane -> stable)")

    # Step response: Y(s) = H(s)/s
    Y = H / s
    steady_state = sp.limit(s * Y, s, 0)

    print(f"\nFinal value theorem: lim_{{t->inf}} y(t) = lim_{{s->0}} s*Y(s)")
    print(f"  s*Y(s) = s * H(s)/s = H(s)")
    print(f"  lim_{{s->0}} H(s) = {sp.simplify(H.subs(s, 0))}")
    print(f"  = 10*2/6 = {float(10 * 2 / 6):.6f}")
    print(f"\nSteady-state value of step response: {steady_state}")


if __name__ == "__main__":
    print("=== Exercise 1 ===")
    exercise_1()
    print("\n=== Exercise 2 ===")
    exercise_2()
    print("\n=== Exercise 3 ===")
    exercise_3()
    print("\n=== Exercise 4 ===")
    exercise_4()
    print("\n=== Exercise 5 ===")
    exercise_5()
    print("\n=== Exercise 6 ===")
    exercise_6()
    print("\n=== Exercise 7 ===")
    exercise_7()
    print("\n=== Exercise 8 ===")
    exercise_8()
    print("\n=== Exercise 9 ===")
    exercise_9()
    print("\n=== Exercise 10 ===")
    exercise_10()
    print("\nAll exercises completed!")
