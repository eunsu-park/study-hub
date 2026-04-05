"""
Exercises for Lesson 14: Optimal Control - LQR and Kalman Filter
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
    Exercise 1: LQR Design
    A = [[0, 1], [-1, -1]], B = [[0], [1]]
    Q = I, R = 1
    """
    A = np.array([[0, 1], [-1, -1]])
    B = np.array([[0], [1]])

    print(f"A = \n{A}")
    print(f"B = {B.flatten()}")

    # Part 1: Solve ARE with Q = I, R = 1
    Q = np.eye(2)
    R = np.array([[1.0]])

    print(f"\nPart 1: Solve ARE with Q = I, R = 1")
    print(f"  A^T P + PA - PBR^(-1)B^T P + Q = 0")
    print(f"  P = [[p11, p12], [p12, p22]] (symmetric)")
    print()
    print("  Setting up the 3 equations (symmetric 2x2 => 3 unknowns):")
    print("  Row 1,1: 0 + 0 - p12^2 + 1 = 0  =>  p12^2 = 1... ")
    print("  (Full derivation would involve solving the coupled nonlinear system)")

    # Solve numerically using scipy
    P = linalg.solve_continuous_are(A, B, Q, R)
    print(f"\n  Numerical solution (scipy):")
    print(f"  P = \n{np.round(P, 6)}")

    # Verify: A^T P + PA - PBR^{-1}B^T P + Q should be ~0
    residual = A.T @ P + P @ A - P @ B @ np.linalg.inv(R) @ B.T @ P + Q
    print(f"  Residual (should be ~0): \n{np.round(residual, 10)}")

    # Part 2: Optimal gain K
    K = np.linalg.inv(R) @ B.T @ P
    print(f"\nPart 2: Optimal gain K = R^(-1) B^T P")
    print(f"  K = {np.round(K, 6)}")

    # Part 3: Closed-loop poles
    A_cl = A - B @ K
    eig_cl = np.linalg.eigvals(A_cl)
    print(f"\nPart 3: Closed-loop poles")
    print(f"  A - BK = \n{np.round(A_cl, 6)}")
    print(f"  Eigenvalues: {np.round(eig_cl, 6)}")
    print(f"  All stable: {all(e.real < 0 for e in eig_cl)}")

    # Part 4: Effect of changing R
    print(f"\nPart 4: Effect of R on closed-loop poles")

    R_values = [0.1, 1.0, 10.0]
    for R_val in R_values:
        R_test = np.array([[R_val]])
        P_test = linalg.solve_continuous_are(A, B, Q, R_test)
        K_test = np.linalg.inv(R_test) @ B.T @ P_test
        eig_test = np.linalg.eigvals(A - B @ K_test)
        print(f"  R = {R_val:5.1f}: K = {np.round(K_test.flatten(), 4)}, "
              f"poles = {np.round(eig_test, 4)}")

    print()
    print("  Observations:")
    print("  - Smaller R (cheap control) => larger K, faster poles (more aggressive)")
    print("  - Larger R (expensive control) => smaller K, slower poles (conservative)")
    print("  - The Q/R ratio determines the performance-effort tradeoff")

    # Plot step responses for different R
    fig, ax = plt.subplots(figsize=(10, 6))
    t = np.linspace(0, 5, 500)

    for R_val in R_values:
        R_test = np.array([[R_val]])
        P_test = linalg.solve_continuous_are(A, B, Q, R_test)
        K_test = np.linalg.inv(R_test) @ B.T @ P_test
        A_cl_test = A - B @ K_test
        C_sys = np.array([[1, 0]])
        # Compute Nr
        Nr = 1.0 / (C_sys @ np.linalg.inv(-A_cl_test) @ B)[0, 0]
        sys = signal.StateSpace(A_cl_test, B * Nr, C_sys, [[0]])
        t_out, y_out = signal.step(sys, T=t)
        ax.plot(t_out, y_out, linewidth=2, label=f'R = {R_val}')

    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.set_title('LQR Step Response for Different R (Q = I)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/opt/projects/01_Personal/03_Study/exercises/Control_Theory/ex14_lqr_tuning.png',
                dpi=100)
    plt.close()
    print("  LQR tuning comparison saved to 'ex14_lqr_tuning.png'")


def exercise_2():
    """
    Exercise 2: Kalman Filter
    Same system with C = [1, 0], W = 0.1, V = 1
    """
    A = np.array([[0, 1], [-1, -1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    G = np.eye(2)  # process noise input matrix

    W = 0.1 * np.eye(2)  # process noise covariance (2x2 since G is 2x2)
    V = np.array([[1.0]])  # measurement noise covariance

    print("Kalman Filter Design")
    print(f"  A = \n{A}")
    print(f"  C = {C}")
    print(f"  W = {W[0,0]} * I (process noise intensity)")
    print(f"  V = {V[0,0]} (measurement noise variance)")

    # Part 1: Set up filter ARE
    print(f"\nPart 1: Filter ARE")
    print(f"  AP + PA^T - PC^T V^(-1) CP + GWG^T = 0")
    print(f"  This is the dual of the control ARE with substitutions:")
    print(f"  A -> A^T, B -> C^T, Q -> GWG^T, R -> V")

    # Solve: use the dual formulation
    # Filter ARE: A*Pf + Pf*A^T - Pf*C^T*V^{-1}*C*Pf + G*W*G^T = 0
    GWGt = G @ W @ G.T
    Pf = linalg.solve_continuous_are(A.T, C.T, GWGt, V)

    print(f"\n  Pf = \n{np.round(Pf, 6)}")

    # Part 2: Kalman gain
    L = Pf @ C.T @ np.linalg.inv(V)
    print(f"\nPart 2: Kalman gain L = Pf * C^T * V^(-1)")
    print(f"  L = {np.round(L.flatten(), 6)}")

    # Part 3: Observer poles
    A_LC = A - L @ C
    eig_obs = np.linalg.eigvals(A_LC)
    print(f"\nPart 3: Observer (Kalman filter) poles")
    print(f"  A - LC = \n{np.round(A_LC, 6)}")
    print(f"  Eigenvalues: {np.round(eig_obs, 6)}")
    print(f"  All stable: {all(e.real < 0 for e in eig_obs)}")

    # Compare with LQR controller poles
    Q = np.eye(2)
    R = np.array([[1.0]])
    P_lqr = linalg.solve_continuous_are(A, B, Q, R)
    K_lqr = np.linalg.inv(R) @ B.T @ P_lqr
    eig_ctrl = np.linalg.eigvals(A - B @ K_lqr)

    print(f"\n  Comparison with LQR controller poles:")
    print(f"  Controller poles: {np.round(eig_ctrl, 6)}")
    print(f"  Kalman poles:     {np.round(eig_obs, 6)}")

    ratio = min(abs(eig_obs[i].real) for i in range(len(eig_obs))) / \
            min(abs(eig_ctrl[i].real) for i in range(len(eig_ctrl)))
    print(f"  Kalman/Controller speed ratio: {ratio:.2f}")
    print(f"  (Rule of thumb suggests observer 3-5x faster than controller)")


def exercise_3():
    """
    Exercise 3: LQR Return Difference Property
    Show |1 + K(jwI - A)^{-1}B| >= 1 for SISO LQR
    """
    print("LQR Return Difference Property (SISO)")
    print("  Claim: |1 + K(jwI - A)^{-1}B| >= 1 for all w")
    print()
    print("Proof sketch:")
    print("  1. The return difference is: F(jw) = 1 + K(jwI - A)^{-1}B")
    print("     where K = R^{-1}B^T P and P satisfies the ARE:")
    print("     A^T P + PA - PBR^{-1}B^T P + Q = 0")
    print()
    print("  2. Substituting K = R^{-1}B^T P:")
    print("     F(jw) = 1 + R^{-1}B^T P(jwI - A)^{-1}B")
    print()
    print("  3. From the ARE, adding and subtracting (jwI)^T P + P(jwI):")
    print("     (-jwI - A)^T P + P(jwI - A) = -Q - PBR^{-1}B^T P")
    print()
    print("  4. Multiplying: P(jwI-A)^{-1}B = [(-jwI-A)^T]^{-1}(Q + PBR^{-1}B^T P)(jwI-A)^{-1}B")
    print()
    print("  5. After algebraic manipulation:")
    print("     |F(jw)|^2 = 1 + B^T(-jwI-A)^{-T} Q (jwI-A)^{-1} B / R")
    print("     Since Q >= 0 and R > 0, the added term is non-negative")
    print("     Therefore |F(jw)|^2 >= 1, hence |F(jw)| >= 1")

    print("\nImplications for gain and phase margins:")
    print("  |F(jw)| >= 1 means the Nyquist plot of L(s) = K(sI-A)^{-1}B")
    print("  never enters the unit circle centered at (-1, 0)")
    print()
    print("  Gain margin: The Nyquist plot can be scaled by a factor of 2")
    print("  (or reduced to 1/2) before touching (-1, 0)")
    print("  => GM = [1/2, infinity) in linear scale = [-6dB, infinity)")
    print()
    print("  Phase margin: The closest approach to (-1,0) determines PM")
    print("  Since |F| >= 1, the distance from L(jw) to -1 >= 1 when |L|=1")
    print("  This guarantees PM >= 60 degrees")

    # Numerical verification
    print("\nNumerical verification:")
    A = np.array([[0, 1], [-1, -1]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.array([[1.0]])

    P = linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    w = np.logspace(-2, 2, 10000)
    F_mag = np.zeros(len(w))

    for i, wi in enumerate(w):
        jwI_A_inv = np.linalg.inv(1j * wi * np.eye(2) - A)
        L_val = (K @ jwI_A_inv @ B)[0, 0]
        F_val = 1 + L_val
        F_mag[i] = abs(F_val)

    print(f"  min |F(jw)| over all w: {np.min(F_mag):.6f}")
    print(f"  |F(jw)| >= 1 verified: {np.min(F_mag) >= 1.0 - 1e-10}")

    # Compute actual margins
    L_values = np.array([(K @ np.linalg.inv(1j*wi*np.eye(2)-A) @ B)[0,0] for wi in w])
    L_mag_dB = 20 * np.log10(np.abs(L_values))
    L_phase = np.degrees(np.angle(L_values))

    # Phase margin: phase at |L| = 0 dB
    gc_idx = np.argmin(np.abs(L_mag_dB))
    PM = 180 + L_phase[gc_idx]
    print(f"  Phase margin: {PM:.1f} degrees (>= 60 guaranteed)")

    # Gain margin: magnitude at phase = -180
    pc_idx = np.argmin(np.abs(L_phase + 180))
    GM = -L_mag_dB[pc_idx]
    print(f"  Gain margin: {GM:.1f} dB (infinite for this system)")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: LQR Design ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Kalman Filter ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: LQR Properties ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
