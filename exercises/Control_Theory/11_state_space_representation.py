"""
Exercises for Lesson 11: State-Space Representation
Topic: Control_Theory
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import linalg, signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Exercise 1: State-Space Modeling of DC Motor
    Electrical: La*dia/dt + Ra*ia + Kb*dtheta/dt = va
    Mechanical: J*d2theta/dt2 + B*dtheta/dt = Kt*ia
    States: x1 = theta, x2 = dtheta/dt, x3 = ia
    Input: u = va, Output: y = theta
    """
    print("DC Motor state-space model")
    print("  Electrical: La*di/dt + Ra*i + Kb*w = va")
    print("  Mechanical: J*dw/dt + B*w = Kt*i")
    print("  States: x1 = theta, x2 = w = dtheta/dt, x3 = ia")

    # Part 1: State-space matrices
    print("\nPart 1: State-space form")
    print("  dx1/dt = x2")
    print("  dx2/dt = -(B/J)*x2 + (Kt/J)*x3")
    print("  dx3/dt = -(Kb/La)*x2 - (Ra/La)*x3 + (1/La)*va")
    print()
    print("  A = [ 0       1        0      ]")
    print("      [ 0      -B/J      Kt/J   ]")
    print("      [ 0      -Kb/La   -Ra/La   ]")
    print()
    print("  B = [ 0      ]")
    print("      [ 0      ]")
    print("      [ 1/La   ]")
    print()
    print("  C = [ 1  0  0 ]")
    print("  D = [ 0 ]")

    # Numerical example
    Ra = 1.0    # Ohm
    La = 0.5    # H
    Kt = 0.01   # N*m/A
    Kb = 0.01   # V*s/rad (= Kt in SI)
    J = 0.01    # kg*m^2
    B_motor = 0.1  # N*m*s/rad

    A = np.array([
        [0, 1, 0],
        [0, -B_motor/J, Kt/J],
        [0, -Kb/La, -Ra/La]
    ])
    B_mat = np.array([[0], [0], [1/La]])
    C = np.array([[1, 0, 0]])
    D = np.array([[0]])

    print(f"\n  Numerical values: Ra={Ra}, La={La}, Kt={Kt}, Kb={Kb}, J={J}, B={B_motor}")
    print(f"\n  A = \n{A}")
    print(f"\n  B = {B_mat.flatten()}")
    print(f"  C = {C.flatten()}")

    # Part 2: Transfer function via G(s) = C(sI-A)^{-1}B
    print("\nPart 2: Transfer function G(s) = C(sI-A)^{-1}B")

    sys_ss = signal.StateSpace(A, B_mat, C, D)
    sys_tf = signal.ss2tf(A, B_mat, C, D)
    num, den = sys_tf

    print(f"  Numerator coefficients: {np.round(num[0], 6)}")
    print(f"  Denominator coefficients: {np.round(den, 6)}")

    # Analytical form
    print("\n  Analytical derivation:")
    print("  Neglecting La (La -> 0):")
    print("    Ra*ia + Kb*w = va  =>  ia = (va - Kb*w)/Ra")
    print("    J*dw/dt + B*w = Kt*(va - Kb*w)/Ra")
    print("    J*dw/dt + (B + Kt*Kb/Ra)*w = Kt*va/Ra")
    print()
    print("  G(s) = Theta(s)/Va(s) = Kt / [s(J*Ra*s + B*Ra + Kt*Kb)]")

    Kt_Kb = Kt * Kb
    denom_coeff = [J * Ra, B_motor * Ra + Kt_Kb, 0]
    print(f"  = {Kt} / [s({J*Ra}s + {B_motor*Ra + Kt_Kb})]")
    print(f"  = {Kt} / ({J*Ra}s^2 + {B_motor*Ra + Kt_Kb}s)")

    # Eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    print(f"\n  Eigenvalues of A: {np.round(eigenvalues, 4)}")
    print(f"  These are the poles of the transfer function")


def exercise_2():
    """
    Exercise 2: Canonical Forms
    G(s) = (2s+3) / (s^3 + 4s^2 + 5s + 6)
    """
    print("G(s) = (2s + 3) / (s^3 + 4s^2 + 5s + 6)")

    # Coefficients
    a0, a1, a2 = 6, 5, 4  # from s^3 + a2*s^2 + a1*s + a0
    b0, b1 = 3, 2  # from b1*s + b0

    # Part 1: Controllable Canonical Form (CCF)
    print("\nPart 1: Controllable Canonical Form")
    A_ccf = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [-a0, -a1, -a2]
    ])
    B_ccf = np.array([[0], [0], [1]])
    C_ccf = np.array([[b0, b1, 0]])
    D_ccf = np.array([[0]])

    print(f"  A_ccf = \n{A_ccf}")
    print(f"  B_ccf = {B_ccf.flatten()}")
    print(f"  C_ccf = {C_ccf.flatten()}")
    print(f"  D_ccf = {D_ccf.flatten()}")

    # Part 2: Observable Canonical Form (OCF)
    print("\nPart 2: Observable Canonical Form")
    A_ocf = np.array([
        [0, 0, -a0],
        [1, 0, -a1],
        [0, 1, -a2]
    ])
    B_ocf = np.array([[b0], [b1], [0]])
    C_ocf = np.array([[0, 0, 1]])
    D_ocf = np.array([[0]])

    print(f"  A_ocf = \n{A_ocf}")
    print(f"  B_ocf = {B_ocf.flatten()}")
    print(f"  C_ocf = {C_ocf.flatten()}")
    print(f"  D_ocf = {D_ocf.flatten()}")

    # Verify: OCF is transpose of CCF
    print(f"\n  Verification: A_ocf = A_ccf^T? {np.allclose(A_ocf, A_ccf.T)}")

    # Part 3: Eigenvalues
    print("\nPart 3: Eigenvalues")
    eigenvalues = np.linalg.eigvals(A_ccf)
    print(f"  Eigenvalues of A: {np.round(eigenvalues, 4)}")

    # Poles of G(s)
    poles = np.roots([1, a2, a1, a0])
    print(f"  Poles of G(s): {np.round(poles, 4)}")
    print(f"  Match: {np.allclose(sorted(eigenvalues.real), sorted(poles.real))}")

    # Verify transfer function recovery
    num_check, den_check = signal.ss2tf(A_ccf, B_ccf, C_ccf, D_ccf)
    print(f"\n  TF from CCF: num = {np.round(num_check[0], 4)}")
    print(f"               den = {np.round(den_check, 4)}")


def exercise_3():
    """
    Exercise 3: State Transition Matrix
    A = [[0, 1], [-2, -3]]
    """
    A = np.array([[0, 1], [-2, -3]])
    x0 = np.array([1, 0])

    print(f"A = \n{A}")

    # Part 1: Eigenvalues
    print("\nPart 1: Eigenvalues")
    eigenvalues = np.linalg.eigvals(A)
    # s^2 + 3s + 2 = (s+1)(s+2) = 0
    print(f"  Characteristic polynomial: s^2 + 3s + 2 = (s+1)(s+2)")
    print(f"  Eigenvalues: {eigenvalues}")

    # Part 2: Matrix exponential e^{At} using Laplace method
    print("\nPart 2: e^{At} via Laplace transform")
    print("  (sI - A)^{-1} = (1/det) * adj(sI - A)")
    print()
    print("  sI - A = [[s, -1], [2, s+3]]")
    print("  det(sI-A) = s(s+3) + 2 = s^2 + 3s + 2 = (s+1)(s+2)")
    print()
    print("  (sI-A)^{-1} = [[s+3, 1], [-2, s]] / [(s+1)(s+2)]")
    print()
    print("  Partial fraction decomposition of each element:")

    # (s+3)/[(s+1)(s+2)]:
    # A/(s+1) + B/(s+2); s=-1: 2/1 = A => A=2; s=-2: 1/(-1) = B => B=-1
    print("  (1,1): (s+3)/[(s+1)(s+2)] = 2/(s+1) - 1/(s+2)")
    print("         -> 2*exp(-t) - exp(-2t)")

    # 1/[(s+1)(s+2)]:
    # A/(s+1) + B/(s+2); s=-1: 1/1 = A => A=1; s=-2: 1/(-1) = B => B=-1
    print("  (1,2): 1/[(s+1)(s+2)] = 1/(s+1) - 1/(s+2)")
    print("         -> exp(-t) - exp(-2t)")

    # -2/[(s+1)(s+2)]:
    print("  (2,1): -2/[(s+1)(s+2)] = -2/(s+1) + 2/(s+2)")
    print("         -> -2*exp(-t) + 2*exp(-2t)")

    # s/[(s+1)(s+2)]:
    # s=-1: -1/1 = A => A=-1; s=-2: -2/(-1) = B => B=2
    print("  (2,2): s/[(s+1)(s+2)] = -1/(s+1) + 2/(s+2)")
    print("         -> -exp(-t) + 2*exp(-2t)")

    print("\n  e^{At} = [[2e^{-t} - e^{-2t},    e^{-t} - e^{-2t}  ],")
    print("            [-2e^{-t} + 2e^{-2t}, -e^{-t} + 2e^{-2t}]]")

    # Verify at t=0: should be I
    def eAt(t):
        return np.array([
            [2*np.exp(-t) - np.exp(-2*t), np.exp(-t) - np.exp(-2*t)],
            [-2*np.exp(-t) + 2*np.exp(-2*t), -np.exp(-t) + 2*np.exp(-2*t)]
        ])

    print(f"\n  Verification at t=0:")
    print(f"  e^{{A*0}} = \n{eAt(0)}")
    print(f"  Should be I: {np.allclose(eAt(0), np.eye(2))}")

    # Verify with scipy
    eA1 = linalg.expm(A * 1.0)
    eA1_analytical = eAt(1.0)
    print(f"\n  Verification at t=1:")
    print(f"  scipy expm:    {eA1}")
    print(f"  analytical:    {eA1_analytical}")
    print(f"  Match: {np.allclose(eA1, eA1_analytical)}")

    # Part 3: x(t) for x(0) = [1, 0]^T with no input
    print(f"\nPart 3: x(t) for x(0) = {x0}")
    print("  x(t) = e^{At} * x(0)")
    print("  x1(t) = (2e^{-t} - e^{-2t}) * 1 + (e^{-t} - e^{-2t}) * 0")
    print("        = 2e^{-t} - e^{-2t}")
    print("  x2(t) = (-2e^{-t} + 2e^{-2t}) * 1 + (-e^{-t} + 2e^{-2t}) * 0")
    print("        = -2e^{-t} + 2e^{-2t}")

    # Plot
    t = np.linspace(0, 5, 500)
    x1 = 2*np.exp(-t) - np.exp(-2*t)
    x2 = -2*np.exp(-t) + 2*np.exp(-2*t)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, x1, 'b-', linewidth=2, label='x1(t) = theta')
    ax.plot(t, x2, 'r-', linewidth=2, label='x2(t) = dtheta/dt')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('State value')
    ax.set_title('Free Response: x(0) = [1, 0]^T')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex11_state_response.png',
                dpi=100)
    plt.close()
    print("  State response plot saved to 'ex11_state_response.png'")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: State-Space Modeling ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Canonical Forms ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: State Transition Matrix ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
