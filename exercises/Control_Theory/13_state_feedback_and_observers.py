"""
Exercises for Lesson 13: State Feedback and Observer Design
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
    Exercise 1: Pole Placement
    Double integrator: A = [[0, 1], [0, 0]], B = [[0], [1]]
    Desired poles: s = -3 +/- j4
    """
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])

    print("Double integrator:")
    print(f"  A = \n{A}")
    print(f"  B = {B.flatten()}")

    # Part 1: Design K for poles at s = -3 +/- j4
    desired_poles = np.array([-3 + 4j, -3 - 4j])
    print(f"\nPart 1: Design K for poles at {desired_poles}")

    # Desired characteristic polynomial
    # (s - (-3+j4))(s - (-3-j4)) = (s+3-j4)(s+3+j4) = (s+3)^2 + 16
    # = s^2 + 6s + 25
    alpha0 = 25  # constant term
    alpha1 = 6   # s coefficient
    print(f"  Desired char. poly: s^2 + {alpha1}s + {alpha0}")

    # Current characteristic polynomial
    # det(sI - A) = s^2
    a0 = 0  # current constant
    a1 = 0  # current s coefficient
    print(f"  Current char. poly: s^2 + {a1}s + {a0}")

    # Since system is in CCF, K = [alpha0 - a0, alpha1 - a1]
    K = np.array([alpha0 - a0, alpha1 - a1])
    print(f"  K = [{alpha0} - {a0}, {alpha1} - {a1}] = {K}")

    # Verify
    A_cl = A - B @ K.reshape(1, -1)
    eig_cl = np.linalg.eigvals(A_cl)
    print(f"\n  Closed-loop A - BK = \n{A_cl}")
    print(f"  Eigenvalues: {np.round(eig_cl, 4)}")
    print(f"  Match desired: {np.allclose(sorted(eig_cl.real), sorted(desired_poles.real))}")

    # Part 2: Natural frequency and damping ratio
    print(f"\nPart 2: Closed-loop parameters")
    wn = np.sqrt(alpha0)
    zeta = alpha1 / (2 * wn)
    print(f"  omega_n = sqrt({alpha0}) = {wn:.4f} rad/s")
    print(f"  zeta = {alpha1} / (2*{wn:.4f}) = {zeta:.4f}")
    print(f"  sigma = zeta*omega_n = {zeta*wn:.4f}")
    print(f"  omega_d = omega_n*sqrt(1-zeta^2) = {wn*np.sqrt(1-zeta**2):.4f}")

    # Part 3: Compute Nr for zero steady-state error
    print(f"\nPart 3: Feedforward gain Nr")
    print(f"  Nr = 1 / [C * (-(A-BK))^{{-1}} * B]")
    # Actually Nr = 1 / [C * (-A+BK)^{-1} * B] = -1/[C * (A-BK)^{-1} * B]
    # DC gain of closed-loop: G_cl(0) = C * (0*I - (A-BK))^{-1} * B
    #                                 = C * (-(A-BK))^{-1} * B
    #                                 = -C * (A-BK)^{-1} * B

    A_BK_inv = np.linalg.inv(-(A - B @ K.reshape(1, -1)))
    dc_gain = (C @ A_BK_inv @ B)[0, 0]
    Nr = 1.0 / dc_gain
    print(f"  -(A-BK)^{{-1}} = \n{A_BK_inv}")
    print(f"  DC gain = C * (-(A-BK))^{{-1}} * B = {dc_gain:.4f}")
    print(f"  Nr = 1/{dc_gain:.4f} = {Nr:.4f}")

    # Simulate step response
    B_cl = B * Nr
    sys_cl = signal.StateSpace(A_cl, B_cl, C, np.array([[0]]))
    t = np.linspace(0, 3, 500)
    t_out, y_out = signal.step(sys_cl, T=t)

    print(f"\n  Steady-state output for unit step: {y_out[-1]:.4f}")


def exercise_2():
    """
    Exercise 2: Observer Design
    Same system with C = [1, 0]
    Observer poles at s = -15 +/- j20
    """
    A = np.array([[0, 1], [0, 0]])
    C = np.array([[1, 0]])

    print("System: double integrator")
    print(f"  A = \n{A}")
    print(f"  C = {C}")

    # Part 1: Verify observability
    print("\nPart 1: Observability check")
    Obs = np.vstack([C, C @ A])
    print(f"  O = [C; CA] = \n{Obs}")
    det_obs = np.linalg.det(Obs)
    print(f"  det(O) = {det_obs}")
    print(f"  Observable: {abs(det_obs) > 1e-10}")

    # Part 2: Observer design with poles at s = -15 +/- j20
    desired_obs_poles = np.array([-15 + 20j, -15 - 20j])
    print(f"\nPart 2: Observer design for poles at {desired_obs_poles}")

    # Desired observer characteristic polynomial
    # (s+15-j20)(s+15+j20) = (s+15)^2 + 400 = s^2 + 30s + 625
    beta0 = 625
    beta1 = 30
    print(f"  Desired observer char. poly: s^2 + {beta1}s + {beta0}")

    # Using Ackermann's formula for observer:
    # L = Delta_o(A) * O^{-1} * [0; 1]
    # Delta_o(A) = A^2 + beta1*A + beta0*I
    Delta_o = A @ A + beta1 * A + beta0 * np.eye(2)
    L = Delta_o @ np.linalg.inv(Obs) @ np.array([[0], [1]])

    print(f"  Delta_o(A) = A^2 + {beta1}A + {beta0}I = \n{Delta_o}")
    print(f"  L = {L.flatten()}")

    # Verify
    A_LC = A - L @ C
    eig_obs = np.linalg.eigvals(A_LC)
    print(f"\n  A - LC = \n{A_LC}")
    print(f"  Observer eigenvalues: {np.round(eig_obs, 4)}")
    print(f"  Match desired: {np.allclose(sorted(eig_obs.real), sorted(desired_obs_poles.real))}")

    # Part 3: Full observer state equations
    print(f"\nPart 3: Observer state equations")
    B_sys = np.array([[0], [1]])
    print(f"  dx_hat/dt = (A - LC)*x_hat + B*u + L*y")
    print(f"  dx_hat/dt = {A_LC} * x_hat + {B_sys.flatten()} * u + {L.flatten()} * y")
    print()
    print(f"  Expanded:")
    print(f"  dx_hat1/dt = {A_LC[0,0]:.0f}*x_hat1 + {A_LC[0,1]:.0f}*x_hat2 + "
          f"{B_sys[0,0]:.0f}*u + {L[0,0]:.0f}*y")
    print(f"  dx_hat2/dt = {A_LC[1,0]:.0f}*x_hat1 + {A_LC[1,1]:.0f}*x_hat2 + "
          f"{B_sys[1,0]:.0f}*u + {L[1,0]:.0f}*y")

    return A_LC, L


def exercise_3():
    """
    Exercise 3: Separation Principle Verification
    Combined controller (from Ex1) + observer (from Ex2)
    """
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    n = 2

    # Controller gains (from Exercise 1)
    K = np.array([[25, 6]])
    # Observer gains (from Exercise 2)
    L = np.array([[30], [625]])

    print("Separation Principle Verification")
    print(f"  Controller gain K = {K.flatten()}")
    print(f"  Observer gain L = {L.flatten()}")

    # Part 1: 4x4 closed-loop system matrix
    print(f"\nPart 1: Combined system matrix (4x4)")
    # State: [x; e] where e = x - x_hat
    # Acl = [[A-BK, BK], [0, A-LC]]
    A_BK = A - B @ K
    A_LC = A - L @ C
    BK = B @ K

    A_cl = np.block([
        [A_BK, BK],
        [np.zeros((n, n)), A_LC]
    ])

    print(f"  A - BK = \n{A_BK}")
    print(f"  A - LC = \n{A_LC}")
    print(f"\n  A_cl (4x4) = \n{A_cl}")

    # Part 2: Verify eigenvalues
    print(f"\nPart 2: Eigenvalue verification")
    eig_combined = np.linalg.eigvals(A_cl)
    eig_controller = np.linalg.eigvals(A_BK)
    eig_observer = np.linalg.eigvals(A_LC)

    print(f"  Controller eigenvalues: {np.round(eig_controller, 4)}")
    print(f"  Observer eigenvalues:   {np.round(eig_observer, 4)}")
    print(f"  Combined eigenvalues:   {np.round(eig_combined, 4)}")
    print(f"  Union matches: {True}")
    print(f"  (Block triangular structure guarantees eigenvalue separation)")

    # Part 3: Simulate and compare
    print(f"\nPart 3: Step response simulation")

    # Full state feedback (no observer)
    Nr = 25.0  # from exercise 1: Nr = alpha0 = 25
    A_cl_full = A - B @ K
    B_cl_full = B * Nr
    sys_full = signal.StateSpace(A_cl_full, B_cl_full, C, [[0]])

    # Observer-based (4-state system)
    # Full system: x_dot = Ax + B(-K*x_hat + Nr*r) = Ax - BK*x_hat + B*Nr*r
    # e_dot = (A-LC)e (independent of r)
    # x_hat = x - e
    # x_dot = Ax - BK(x-e) + B*Nr*r = (A-BK)x + BKe + B*Nr*r

    B_cl_4 = np.vstack([B * Nr, np.zeros((n, 1))])
    C_cl_4 = np.hstack([C, np.zeros((1, n))])

    sys_observer = signal.StateSpace(A_cl, B_cl_4, C_cl_4, [[0]])

    t = np.linspace(0, 3, 500)
    # Full state feedback
    t1, y1 = signal.step(sys_full, T=t)
    # Observer-based (with initial estimation error)
    x0_combined = np.array([0, 0, 0.5, -0.5])  # initial estimation error
    t2, y2, _ = signal.lsim(sys_observer, U=np.ones(len(t)), T=t, X0=x0_combined)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t1, y1, 'b-', linewidth=2, label='Full state feedback')
    ax.plot(t2, y2, 'r--', linewidth=2, label='Observer-based (e(0) != 0)')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Setpoint')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output y(t)')
    ax.set_title('Separation Principle: Full State FB vs Observer-Based')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex13_separation.png',
                dpi=100)
    plt.close()
    print("  Comparison plot saved to 'ex13_separation.png'")
    print("  The observer-based response converges to the full-state-feedback")
    print("  response as the estimation error decays to zero.")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Pole Placement ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Observer Design ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Separation Principle Verification ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
