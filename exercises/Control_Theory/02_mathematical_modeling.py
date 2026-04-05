"""
Exercises for Lesson 02: Mathematical Modeling of Physical Systems
Topic: Control_Theory
Solutions to practice problems from the lesson.
"""

import numpy as np
from numpy.polynomial import polynomial as P
import sympy as sp


def exercise_1():
    """
    Exercise 1: Mechanical Modeling
    Two-mass system: m1 connected to wall by k1, b1.
    m1 and m2 connected by k12, b12. Force F(t) applied to m2.
    """
    print("Part 1: Free-body diagrams")
    print("  Mass m1: Forces acting on m1:")
    print("    - Spring k1: -k1*x1 (restoring toward wall)")
    print("    - Damper b1: -b1*dx1/dt")
    print("    - Spring k12: k12*(x2 - x1) (coupling)")
    print("    - Damper b12: b12*(dx2/dt - dx1/dt) (coupling)")

    print("\n  Mass m2: Forces acting on m2:")
    print("    - Spring k12: -k12*(x2 - x1) (coupling, reaction)")
    print("    - Damper b12: -b12*(dx2/dt - dx1/dt) (coupling, reaction)")
    print("    - External force: F(t)")

    print("\nPart 2: Coupled differential equations (Newton's 2nd law)")
    print("  m1*x1'' = -k1*x1 - b1*x1' + k12*(x2 - x1) + b12*(x2' - x1')")
    print("  m1*x1'' + (b1 + b12)*x1' + (k1 + k12)*x1 - b12*x2' - k12*x2 = 0")
    print()
    print("  m2*x2'' = -k12*(x2 - x1) - b12*(x2' - x1') + F(t)")
    print("  m2*x2'' + b12*x2' + k12*x2 - b12*x1' - k12*x1 = F(t)")

    print("\nPart 3: Transfer function X2(s)/F(s)")
    print("  Taking Laplace transforms (zero ICs):")
    print("  [m1*s^2 + (b1+b12)*s + (k1+k12)] X1 = [b12*s + k12] X2")
    print("  [m2*s^2 + b12*s + k12] X2 = [b12*s + k12] X1 + F(s)")
    print()
    print("  From equation 1: X1 = [b12*s + k12] / [m1*s^2 + (b1+b12)*s + (k1+k12)] * X2")
    print("  Let D1(s) = m1*s^2 + (b1+b12)*s + (k1+k12)")
    print("  Let D2(s) = m2*s^2 + b12*s + k12")
    print("  Let C(s) = b12*s + k12")
    print()
    print("  Substituting into equation 2:")
    print("  D2(s)*X2 = C(s)*X1 + F(s) = C(s)^2/D1(s) * X2 + F(s)")
    print("  [D2(s) - C(s)^2/D1(s)] X2 = F(s)")
    print()
    print("  X2(s)/F(s) = D1(s) / [D1(s)*D2(s) - C(s)^2]")

    # Numerical example
    print("\n  --- Numerical example ---")
    m1, m2 = 1.0, 1.0
    k1, k12 = 2.0, 1.0
    b1, b12 = 0.5, 0.3

    s = sp.Symbol('s')
    D1 = m1*s**2 + (b1+b12)*s + (k1+k12)
    D2 = m2*s**2 + b12*s + k12
    C_s = b12*s + k12

    tf_num = D1
    tf_den = sp.expand(D1*D2 - C_s**2)

    print(f"  m1={m1}, m2={m2}, k1={k1}, k12={k12}, b1={b1}, b12={b12}")
    print(f"  Numerator: {sp.expand(tf_num)}")
    print(f"  Denominator: {tf_den}")
    print(f"  X2(s)/F(s) = ({sp.expand(tf_num)}) / ({tf_den})")


def exercise_2():
    """
    Exercise 2: Electrical-Mechanical Analogy
    Series RLC circuit, input v_in, output v_C.
    """
    print("Part 1: Differential equation for RLC circuit")
    print("  KVL: L*di/dt + R*i + v_C = v_in")
    print("  Since i = C*dv_C/dt:")
    print("  LC*d²v_C/dt² + RC*dv_C/dt + v_C = v_in")

    print("\nPart 2: Mechanical analogy")
    print("  Comparing LC*v_C'' + RC*v_C' + v_C = v_in")
    print("  with     m*x''    + b*x'    + k*x = F")
    print()
    print("  | Electrical     | Mechanical        |")
    print("  |----------------|-------------------|")
    print("  | Inductance L   | Mass m            |")
    print("  | Resistance R   | Damping coeff. b  |")
    print("  | 1/Capacitance  | Spring const. k   |")
    print("  | Voltage v_in   | Force F           |")
    print("  | Cap. voltage   | Displacement x    |")
    print("  | Current i      | Velocity dx/dt    |")

    print("\nPart 3: Transfer function V_C(s)/V_in(s)")

    # Symbolic
    s = sp.Symbol('s')
    L, R, C = sp.symbols('L R C', positive=True)
    G = 1 / (L*C*s**2 + R*C*s + 1)
    print(f"  V_C(s)/V_in(s) = 1 / (LCs² + RCs + 1)")
    print(f"  In standard form: omega_n² / (s² + 2*zeta*omega_n*s + omega_n²)")
    print(f"  where omega_n = 1/sqrt(LC) and zeta = R/(2) * sqrt(C/L)")

    # Numerical example
    L_val, R_val, C_val = 1.0, 2.0, 0.5
    omega_n = 1.0 / np.sqrt(L_val * C_val)
    zeta = R_val / 2.0 * np.sqrt(C_val / L_val)
    print(f"\n  Numerical example: L={L_val} H, R={R_val} Ohm, C={C_val} F")
    print(f"  omega_n = 1/sqrt({L_val}*{C_val}) = {omega_n:.4f} rad/s")
    print(f"  zeta = {R_val}/2 * sqrt({C_val}/{L_val}) = {zeta:.4f}")
    if zeta < 1:
        print("  System is underdamped")
    elif zeta == 1:
        print("  System is critically damped")
    else:
        print("  System is overdamped")


def exercise_3():
    """
    Exercise 3: Linearization of a tank system
    A*dh/dt = q_in - c*sqrt(h)
    """
    print("Part 1: Equilibrium")
    print("  At equilibrium: dh/dt = 0")
    print("  0 = q_bar_in - c*sqrt(h_bar)")
    print("  h_bar = (q_bar_in / c)^2")

    print("\nPart 2: Linearization around equilibrium")
    print("  Let delta_h = h - h_bar, delta_q = q_in - q_bar_in")
    print("  f(h, q_in) = (q_in - c*sqrt(h)) / A")
    print("  df/dh = -c/(2*A*sqrt(h)) evaluated at h_bar = -c/(2*A*sqrt(h_bar))")
    print("  df/dq_in = 1/A")
    print()
    print("  Since h_bar = (q_bar_in/c)^2, sqrt(h_bar) = q_bar_in/c")
    print("  df/dh|_eq = -c/(2*A*(q_bar_in/c)) = -c^2/(2*A*q_bar_in)")
    print()
    print("  Linearized equation:")
    print("  A * d(delta_h)/dt = delta_q_in - c/(2*sqrt(h_bar)) * delta_h")
    print("  A * d(delta_h)/dt = delta_q_in - c^2/(2*q_bar_in) * delta_h")

    print("\nPart 3: Transfer function delta_H(s) / delta_Q_in(s)")
    print("  Taking Laplace transform:")
    print("  A*s*H(s) = Q(s) - c^2/(2*q_bar_in) * H(s)")
    print("  H(s) * [A*s + c^2/(2*q_bar_in)] = Q(s)")
    print()
    print("  H(s)/Q(s) = 1 / (A*s + c^2/(2*q_bar_in))")
    print("            = (2*q_bar_in) / (2*A*q_bar_in*s + c^2)")
    print()
    print("  This is a first-order system with:")
    print("  DC gain K = 2*q_bar_in / c^2")
    print("  Time constant tau = 2*A*q_bar_in / c^2")
    print("  In standard form: K / (tau*s + 1)")

    # Numerical example
    A_val = 1.0   # m^2
    c_val = 0.5   # valve coefficient
    q_bar = 1.0   # m^3/s

    h_bar = (q_bar / c_val)**2
    K = 2 * q_bar / c_val**2
    tau = 2 * A_val * q_bar / c_val**2

    print(f"\n  Numerical: A={A_val}, c={c_val}, q_bar_in={q_bar}")
    print(f"  h_bar = ({q_bar}/{c_val})^2 = {h_bar:.2f} m")
    print(f"  DC gain K = {K:.2f}")
    print(f"  Time constant tau = {tau:.2f} s")
    print(f"  G(s) = {K:.2f} / ({tau:.2f}*s + 1)")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Mechanical Modeling ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Electrical-Mechanical Analogy ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Linearization ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
