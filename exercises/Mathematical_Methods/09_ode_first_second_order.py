"""
Exercise Solutions: Lesson 09 - ODE First and Second Order
Mathematical Methods for Physical Sciences

Covers: separable, integrating factor, exact equations, characteristic equation,
        undetermined coefficients, damped oscillator, RLC, variation of parameters,
        Wronskian, Picard iteration
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


def exercise_1_separable():
    """
    Problem 1: Solve dy/dx = x^2/(1+y^2) by separation of variables.
    """
    print("=" * 60)
    print("Problem 1: Separable ODE")
    print("=" * 60)

    x = sp.Symbol('x')
    y = sp.Function('y')

    # (1+y^2) dy = x^2 dx
    # Integrate: y + y^3/3 = x^3/3 + C

    print(f"\ndy/dx = x^2/(1+y^2)")
    print(f"\nSeparation: (1+y^2) dy = x^2 dx")
    print(f"Integrate both sides:")
    print(f"  y + y^3/3 = x^3/3 + C")

    # Verify with sympy
    ode = sp.Eq(y(x).diff(x), x**2 / (1 + y(x)**2))
    sol = sp.dsolve(ode, y(x))
    print(f"\nSympy solution: {sol}")

    # With IC y(0) = 0: C = 0
    print(f"\nWith IC y(0) = 0: C = 0")
    print(f"Implicit solution: y + y^3/3 = x^3/3")


def exercise_2_integrating_factor():
    """
    Problem 2: Solve dy/dx + 2xy = x using integrating factor.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Integrating Factor")
    print("=" * 60)

    x = sp.Symbol('x')
    y = sp.Function('y')

    # Standard form: y' + P(x)y = Q(x)
    # P(x) = 2x, Q(x) = x
    # Integrating factor: mu = e^{integral P dx} = e^{x^2}

    print(f"\ndy/dx + 2xy = x")
    print(f"\nP(x) = 2x, Q(x) = x")
    print(f"Integrating factor: mu(x) = exp(int 2x dx) = exp(x^2)")

    print(f"\nMultiply: d/dx[e^{{x^2}} y] = x*e^{{x^2}}")
    print(f"Integrate: e^{{x^2}} y = (1/2)e^{{x^2}} + C")
    print(f"Solution: y = 1/2 + C*e^{{-x^2}}")

    # Verify with sympy
    ode = sp.Eq(y(x).diff(x) + 2*x*y(x), x)
    sol = sp.dsolve(ode, y(x))
    print(f"\nSympy: {sol}")


def exercise_3_exact_equation():
    """
    Problem 3: Test for exactness and solve:
    (2xy + 3) dx + (x^2 + 4y) dy = 0
    """
    print("\n" + "=" * 60)
    print("Problem 3: Exact Equation")
    print("=" * 60)

    x, y = sp.symbols('x y')

    M = 2*x*y + 3
    N = x**2 + 4*y

    dM_dy = sp.diff(M, y)
    dN_dx = sp.diff(N, x)

    print(f"\n(2xy + 3) dx + (x^2 + 4y) dy = 0")
    print(f"\nM = 2xy + 3, N = x^2 + 4y")
    print(f"dM/dy = {dM_dy}")
    print(f"dN/dx = {dN_dx}")
    print(f"Exact: dM/dy == dN/dx: {dM_dy == dN_dx}")

    # Find F such that F_x = M, F_y = N
    # F = integral M dx = x^2*y + 3x + g(y)
    # F_y = x^2 + g'(y) = x^2 + 4y => g'(y) = 4y => g(y) = 2y^2
    # F = x^2*y + 3x + 2y^2 = C

    F = x**2*y + 3*x + 2*y**2
    print(f"\nF_x = M => F = x^2*y + 3x + g(y)")
    print(f"F_y = x^2 + g'(y) = N = x^2 + 4y => g'(y) = 4y => g = 2y^2")
    print(f"\nSolution: x^2*y + 3x + 2y^2 = C")

    # Verify
    print(f"\nVerification:")
    print(f"  F_x = {sp.diff(F, x)} = M [check]")
    print(f"  F_y = {sp.diff(F, y)} = N [check]")


def exercise_4_characteristic():
    """
    Problem 4: Solve y'' - 6y' + 9y = 0 (repeated roots).
    """
    print("\n" + "=" * 60)
    print("Problem 4: Characteristic Equation (Repeated Roots)")
    print("=" * 60)

    r = sp.Symbol('r')
    char_eq = r**2 - 6*r + 9

    roots = sp.solve(char_eq, r)
    print(f"\ny'' - 6y' + 9y = 0")
    print(f"\nCharacteristic equation: r^2 - 6r + 9 = 0")
    print(f"  (r - 3)^2 = 0")
    print(f"  Roots: r = {roots} (repeated)")

    print(f"\nGeneral solution: y = (C1 + C2*x)*e^{{3x}}")
    print(f"  y1 = e^{{3x}}, y2 = x*e^{{3x}}")

    # Verify with sympy
    x = sp.Symbol('x')
    y = sp.Function('y')
    ode = sp.Eq(y(x).diff(x, 2) - 6*y(x).diff(x) + 9*y(x), 0)
    sol = sp.dsolve(ode, y(x))
    print(f"\nSympy: {sol}")


def exercise_5_undetermined_coefficients():
    """
    Problem 5: Solve y'' + 3y' + 2y = 4*e^{-x}.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Undetermined Coefficients")
    print("=" * 60)

    x = sp.Symbol('x')
    y = sp.Function('y')

    print(f"\ny'' + 3y' + 2y = 4*e^(-x)")

    # Homogeneous: r^2 + 3r + 2 = (r+1)(r+2) = 0 => r = -1, -2
    print(f"\nHomogeneous solution:")
    print(f"  r^2 + 3r + 2 = (r+1)(r+2) = 0")
    print(f"  y_h = C1*e^(-x) + C2*e^(-2x)")

    # Particular: try y_p = A*x*e^(-x) (since e^(-x) is a homogeneous solution)
    print(f"\nParticular solution:")
    print(f"  Since e^(-x) is a homogeneous solution, try y_p = A*x*e^(-x)")

    A = sp.Symbol('A')
    y_p = A * x * sp.exp(-x)
    y_p_prime = sp.diff(y_p, x)
    y_p_double = sp.diff(y_p, x, 2)

    lhs = sp.expand(y_p_double + 3*y_p_prime + 2*y_p)
    print(f"  y_p'' + 3y_p' + 2y_p = {lhs}")

    # Solve for A
    A_val = sp.solve(sp.Eq(lhs, 4*sp.exp(-x)), A)
    print(f"  A = {A_val}")

    print(f"\n  y_p = {A_val[0]}*x*e^(-x)")
    print(f"\nGeneral solution: y = C1*e^(-x) + C2*e^(-2x) + {A_val[0]}*x*e^(-x)")

    # Verify with sympy
    ode = sp.Eq(y(x).diff(x, 2) + 3*y(x).diff(x) + 2*y(x), 4*sp.exp(-x))
    sol = sp.dsolve(ode, y(x))
    print(f"\nSympy: {sol}")


def exercise_6_damped_oscillator():
    """
    Problem 6: Damped harmonic oscillator m*x'' + gamma*x' + k*x = 0.
    m = 0.5 kg, k = 8 N/m, gamma = 2 Ns/m.
    Find x(t) with x(0) = 1, x'(0) = 0.
    """
    print("\n" + "=" * 60)
    print("Problem 6: Damped Harmonic Oscillator")
    print("=" * 60)

    m, gamma_coeff, k = 0.5, 2.0, 8.0

    # Divide by m: x'' + (gamma/m)*x' + (k/m)*x = 0
    b = gamma_coeff / m  # = 4
    omega_0_sq = k / m    # = 16, omega_0 = 4

    print(f"\nm = {m}, gamma = {gamma_coeff}, k = {k}")
    print(f"x'' + {b}x' + {omega_0_sq}x = 0")

    # Characteristic: r^2 + 4r + 16 = 0
    # r = (-4 +/- sqrt(16-64))/2 = -2 +/- i*sqrt(12) = -2 +/- 2i*sqrt(3)
    disc = b**2 - 4*omega_0_sq
    print(f"\nDiscriminant: {b}^2 - 4*{omega_0_sq} = {disc} < 0 => UNDERDAMPED")

    alpha = b / 2  # = 2 (damping rate)
    omega_d = np.sqrt(-disc) / 2  # = sqrt(12) = 2*sqrt(3)

    print(f"alpha = gamma/(2m) = {alpha}")
    print(f"omega_d = sqrt(omega_0^2 - alpha^2) = {omega_d:.6f} = 2*sqrt(3)")
    print(f"\nx(t) = e^(-{alpha}t) * [A*cos({omega_d:.4f}*t) + B*sin({omega_d:.4f}*t)]")

    # IC: x(0) = 1, x'(0) = 0
    # A = 1
    # x'(0) = -alpha*A + omega_d*B = 0 => B = alpha/omega_d = 2/(2*sqrt(3)) = 1/sqrt(3)
    A = 1.0
    B = alpha / omega_d
    print(f"\nIC: x(0) = 1 => A = 1")
    print(f"    x'(0) = 0 => B = alpha/omega_d = {B:.6f} = 1/sqrt(3)")

    # Numerical solution
    t = np.linspace(0, 5, 500)
    x_t = np.exp(-alpha * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

    # Quality factor
    Q = np.sqrt(omega_0_sq) / b
    print(f"\nQuality factor Q = omega_0/(gamma/m) = {Q:.4f}")
    print(f"Number of oscillations to decay: ~Q/pi = {Q/np.pi:.2f}")


def exercise_7_rlc_circuit():
    """
    Problem 7: Series RLC circuit
    L*q'' + R*q' + q/C = V_0, q(0) = 0, q'(0) = 0.
    L=1H, R=10 Ohm, C=0.01F, V_0 = 10V.
    """
    print("\n" + "=" * 60)
    print("Problem 7: RLC Circuit")
    print("=" * 60)

    L, R, C_val, V_0 = 1.0, 10.0, 0.01, 10.0

    print(f"\nL*q'' + R*q' + q/C = V_0")
    print(f"L={L}H, R={R}Ohm, C={C_val}F, V_0={V_0}V")

    omega_0 = 1 / np.sqrt(L * C_val)  # = 10
    alpha = R / (2 * L)  # = 5

    print(f"\nomega_0 = 1/sqrt(LC) = {omega_0:.2f} rad/s")
    print(f"alpha = R/(2L) = {alpha:.2f}")

    disc = alpha**2 - omega_0**2
    print(f"alpha^2 - omega_0^2 = {disc}")

    if disc < 0:
        omega_d = np.sqrt(-disc)
        print(f"Underdamped: omega_d = {omega_d:.4f}")
    elif disc == 0:
        print("Critically damped")
    else:
        print("Overdamped")

    # Solve numerically
    def rlc_ode(t, state):
        q, q_dot = state
        q_ddot = (V_0 - R * q_dot - q / C_val) / L
        return [q_dot, q_ddot]

    sol = solve_ivp(rlc_ode, [0, 2], [0, 0], t_eval=np.linspace(0, 2, 500),
                    method='RK45', rtol=1e-10)

    q_steady = C_val * V_0
    print(f"\nSteady state: q_ss = C*V_0 = {q_steady:.4f} C")
    print(f"Numerical q at t=2: {sol.y[0, -1]:.6f}")
    print(f"Current i = q' at t=2: {sol.y[1, -1]:.6f} (should -> 0)")


def exercise_8_variation_of_parameters():
    """
    Problem 8: Solve y'' + 4y = 1/sin(2x) by variation of parameters.
    """
    print("\n" + "=" * 60)
    print("Problem 8: Variation of Parameters")
    print("=" * 60)

    x = sp.Symbol('x')

    print(f"\ny'' + 4y = 1/sin(2x)")
    print(f"\nHomogeneous solutions: y1 = cos(2x), y2 = sin(2x)")

    y1 = sp.cos(2*x)
    y2 = sp.sin(2*x)

    # Wronskian
    W = y1 * sp.diff(y2, x) - y2 * sp.diff(y1, x)
    W_simplified = sp.simplify(W)
    print(f"Wronskian W = y1*y2' - y2*y1' = {W_simplified}")

    # Particular solution:
    # u1' = -y2*g/W = -sin(2x)*(1/sin(2x))/2 = -1/2
    # u2' = y1*g/W = cos(2x)*(1/sin(2x))/2 = cos(2x)/(2*sin(2x))

    g = 1/sp.sin(2*x)
    u1_prime = sp.simplify(-y2 * g / W_simplified)
    u2_prime = sp.simplify(y1 * g / W_simplified)

    print(f"\nu1' = -y2*g/W = {u1_prime}")
    print(f"u2' = y1*g/W = {u2_prime}")

    u1 = sp.integrate(u1_prime, x)
    u2 = sp.integrate(u2_prime, x)

    print(f"\nu1 = {u1}")
    print(f"u2 = {u2}")

    y_p = sp.simplify(u1 * y1 + u2 * y2)
    print(f"\ny_p = u1*y1 + u2*y2 = {y_p}")

    # General solution
    print(f"\ny = C1*cos(2x) + C2*sin(2x) + {y_p}")


def exercise_9_wronskian():
    """
    Problem 9: Show {1, x, x^2} are linearly independent using Wronskian.
    """
    print("\n" + "=" * 60)
    print("Problem 9: Wronskian of {1, x, x^2}")
    print("=" * 60)

    x = sp.Symbol('x')
    y1, y2, y3 = sp.Integer(1), x, x**2

    W = sp.Matrix([
        [y1, y2, y3],
        [sp.diff(y1, x), sp.diff(y2, x), sp.diff(y3, x)],
        [sp.diff(y1, x, 2), sp.diff(y2, x, 2), sp.diff(y3, x, 2)]
    ])

    print(f"\nFunctions: y1 = 1, y2 = x, y3 = x^2")
    print(f"\nWronskian matrix:")
    print(f"  | 1   x   x^2 |")
    print(f"  | 0   1   2x  |")
    print(f"  | 0   0    2  |")

    det_W = W.det()
    print(f"\nW = det = {det_W}")
    print(f"\nSince W = {det_W} != 0, the functions are LINEARLY INDEPENDENT")


def exercise_10_picard():
    """
    Problem 10: Picard iteration for y' = y, y(0) = 1.
    Show iterates converge to e^x.
    """
    print("\n" + "=" * 60)
    print("Problem 10: Picard Iteration for y' = y")
    print("=" * 60)

    x = sp.Symbol('x')

    # Picard: y_{n+1}(x) = y_0 + int_0^x f(t, y_n(t)) dt
    # f(t, y) = y, y_0 = 1

    print(f"\ny' = y, y(0) = 1")
    print(f"Picard iteration: y_{{n+1}}(x) = 1 + int_0^x y_n(t) dt")

    y_n = sp.Integer(1)
    print(f"\ny_0 = 1")

    for n in range(6):
        t = sp.Symbol('t')
        y_next = 1 + sp.integrate(y_n.subs(x, t), (t, 0, x))
        y_next = sp.expand(y_next)
        y_n = y_next
        print(f"y_{n+1} = {y_n}")

    # Compare with Taylor series of e^x
    taylor_ex = sum(x**k / sp.factorial(k) for k in range(7))
    print(f"\nTaylor e^x (6 terms) = {taylor_ex}")
    print(f"Match: {sp.expand(y_n - taylor_ex) == 0}")

    # Numerical comparison
    x_vals = np.linspace(0, 2, 100)
    y_exact = np.exp(x_vals)

    print(f"\nNumerical comparison at x = 2:")
    y_approx = sum(2.0**k / np.math.factorial(k) for k in range(7))
    print(f"  6th iterate at x=2: {y_approx:.8f}")
    print(f"  e^2 = {np.exp(2):.8f}")
    print(f"  Error: {abs(y_approx - np.exp(2)):.6f}")


if __name__ == "__main__":
    exercise_1_separable()
    exercise_2_integrating_factor()
    exercise_3_exact_equation()
    exercise_4_characteristic()
    exercise_5_undetermined_coefficients()
    exercise_6_damped_oscillator()
    exercise_7_rlc_circuit()
    exercise_8_variation_of_parameters()
    exercise_9_wronskian()
    exercise_10_picard()
