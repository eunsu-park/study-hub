"""
Exercises for Lesson 12: Controllability and Observability
Topic: Control_Theory
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal


def exercise_1():
    """
    Exercise 1: Controllability and Observability Check
    A = [[0, 1, 0], [0, 0, 1], [-6, -11, -6]]
    B = [[0], [0], [1]]
    C = [[1, 0, 0]]
    """
    A = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [-6, -11, -6]])
    B = np.array([[0], [0], [1]])
    C = np.array([[1, 0, 0]])

    print(f"A = \n{A}")
    print(f"B = {B.flatten()}")
    print(f"C = {C.flatten()}")

    # Part 1: Controllability
    print("\nPart 1: Controllability")
    # Controllability matrix: [B, AB, A^2*B]
    AB = A @ B
    A2B = A @ AB
    Ctrl = np.hstack([B, AB, A2B])
    rank_ctrl = np.linalg.matrix_rank(Ctrl)

    print(f"  B   = {B.flatten()}")
    print(f"  AB  = {AB.flatten()}")
    print(f"  A^2B = {A2B.flatten()}")
    print(f"  Controllability matrix:")
    print(f"  {Ctrl}")
    print(f"  rank(C) = {rank_ctrl}")
    print(f"  n = {A.shape[0]}")
    print(f"  Controllable: {rank_ctrl == A.shape[0]}")
    det_ctrl = np.linalg.det(Ctrl)
    print(f"  det(Controllability matrix) = {det_ctrl:.4f}")

    # Part 2: Observability
    print("\nPart 2: Observability")
    CA = C @ A
    CA2 = C @ A @ A
    Obs = np.vstack([C, CA, CA2])
    rank_obs = np.linalg.matrix_rank(Obs)

    print(f"  C    = {C}")
    print(f"  CA   = {CA}")
    print(f"  CA^2 = {CA2}")
    print(f"  Observability matrix:")
    print(f"  {Obs}")
    print(f"  rank(O) = {rank_obs}")
    print(f"  Observable: {rank_obs == A.shape[0]}")
    det_obs = np.linalg.det(Obs)
    print(f"  det(Observability matrix) = {det_obs:.4f}")

    # Part 3: Transfer function
    print("\nPart 3: Transfer function")
    D = np.array([[0]])
    num, den = signal.ss2tf(A, B, C, D)
    print(f"  Numerator:   {np.round(num[0], 4)}")
    print(f"  Denominator: {np.round(den, 4)}")

    poles = np.roots(den)
    zeros = np.roots(num[0][num[0] != 0]) if np.any(num[0] != 0) else []
    print(f"  Poles: {np.round(poles, 4)}")
    print(f"  Zeros: {np.round(zeros, 4) if len(zeros) > 0 else 'None'}")

    # Check the characteristic polynomial
    char_poly = np.poly(A)
    print(f"  Characteristic polynomial: {np.round(char_poly, 4)}")
    print(f"  = s^3 + 6s^2 + 11s + 6 = (s+1)(s+2)(s+3)")

    # If controllable and observable, no pole-zero cancellations
    if rank_ctrl == A.shape[0] and rank_obs == A.shape[0]:
        print("  System is controllable AND observable => minimal realization")
        print("  => No pole-zero cancellations in transfer function")


def exercise_2():
    """
    Exercise 2: PBH Test
    A = [[-2, 1], [0, -2]], B = [[0], [1]], C = [[1, 0]]
    """
    A = np.array([[-2, 1],
                  [0, -2]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])

    print(f"A = \n{A}")
    print(f"B = {B.flatten()}")
    print(f"C = {C.flatten()}")

    eigenvalues = np.linalg.eigvals(A)
    print(f"\nEigenvalues: {eigenvalues}")
    print("  Repeated eigenvalue at s = -2")

    # Part 1: PBH controllability test at s = -2
    print("\nPart 1: PBH controllability test at s = -2")
    s = -2
    sI_minus_A = s * np.eye(2) - A
    PBH_ctrl = np.hstack([sI_minus_A, B])
    rank_pbh_ctrl = np.linalg.matrix_rank(PBH_ctrl)

    print(f"  sI - A = \n{sI_minus_A}")
    print(f"  [sI-A | B] = \n{PBH_ctrl}")
    print(f"  rank([sI-A | B]) = {rank_pbh_ctrl}")
    print(f"  n = {A.shape[0]}")
    print(f"  Controllable at s=-2: {rank_pbh_ctrl == A.shape[0]}")

    # Also check via controllability matrix
    Ctrl = np.hstack([B, A @ B])
    print(f"\n  Controllability matrix check:")
    print(f"  [B, AB] = \n{Ctrl}")
    print(f"  det = {np.linalg.det(Ctrl):.4f}")
    print(f"  rank = {np.linalg.matrix_rank(Ctrl)}")

    # Part 2: PBH observability test at s = -2
    print("\nPart 2: PBH observability test at s = -2")
    PBH_obs = np.vstack([sI_minus_A, C])
    rank_pbh_obs = np.linalg.matrix_rank(PBH_obs)

    print(f"  [sI-A; C] = \n{PBH_obs}")
    print(f"  rank = {rank_pbh_obs}")
    print(f"  Observable at s=-2: {rank_pbh_obs == A.shape[0]}")

    # Observability matrix check
    Obs = np.vstack([C, C @ A])
    print(f"\n  Observability matrix check:")
    print(f"  [C; CA] = \n{Obs}")
    print(f"  det = {np.linalg.det(Obs):.4f}")
    print(f"  rank = {np.linalg.matrix_rank(Obs)}")

    # Part 3: Transfer function
    print("\nPart 3: Transfer function")
    D = np.array([[0]])
    num, den = signal.ss2tf(A, B, C, D)
    print(f"  G(s) numerator: {np.round(num[0], 4)}")
    print(f"  G(s) denominator: {np.round(den, 4)}")

    # Expected: G(s) = C(sI-A)^{-1}B
    # (sI-A)^{-1} = [[1/(s+2), 1/(s+2)^2], [0, 1/(s+2)]]
    # ... times B = [0, 1]^T:
    # (sI-A)^{-1}B = [1/(s+2)^2, 1/(s+2)]^T
    # C * ... = 1/(s+2)^2
    print(f"  G(s) = 1/(s+2)^2")
    print(f"  Order of TF = 2, order of state-space = 2")
    print(f"  No cancellation => minimal realization: "
          f"{'Yes' if rank_pbh_ctrl == 2 and rank_pbh_obs == 2 else 'No'}")


def exercise_3():
    """
    Exercise 3: Hidden Modes
    A = [[-1, 0], [0, 2]], B = [[1], [0]], C = [[1, 0]]
    """
    A = np.array([[-1, 0],
                  [0, 2]])
    B = np.array([[1], [0]])
    C = np.array([[1, 0]])
    D = np.array([[0]])

    print(f"A = \n{A}")
    print(f"B = {B.flatten()}")
    print(f"C = {C.flatten()}")

    eigenvalues = np.linalg.eigvals(A)
    print(f"\nEigenvalues of A: {eigenvalues}")

    # Part 1: BIBO stability
    print("\nPart 1: BIBO stability (from transfer function)")
    num, den = signal.ss2tf(A, B, C, D)
    print(f"  G(s) numerator: {np.round(num[0], 4)}")
    print(f"  G(s) denominator: {np.round(den, 4)}")

    # G(s) = C(sI-A)^{-1}B
    # (sI-A) = [[s+1, 0], [0, s-2]]
    # (sI-A)^{-1} = [[1/(s+1), 0], [0, 1/(s-2)]]
    # (sI-A)^{-1}B = [1/(s+1), 0]^T
    # G(s) = C * [1/(s+1), 0]^T = 1/(s+1)
    print(f"  G(s) = 1/(s+1)")
    print(f"  Transfer function pole: s = -1 (in LHP)")
    print(f"  BIBO stable: YES")

    # Part 2: Internal stability
    print("\nPart 2: Internal stability")
    print(f"  Eigenvalues of A: {eigenvalues}")
    print(f"  Eigenvalue at s = 2 is in the RHP")
    print(f"  Internally stable: NO")

    # Check controllability and observability
    Ctrl = np.hstack([B, A @ B])
    Obs = np.vstack([C, C @ A])
    print(f"\n  Controllability matrix [B, AB] = \n{Ctrl}")
    print(f"  rank = {np.linalg.matrix_rank(Ctrl)} (not full rank => not controllable)")
    print(f"  Observability matrix [C; CA] = \n{Obs}")
    print(f"  rank = {np.linalg.matrix_rank(Obs)} (not full rank => not observable)")

    print(f"\n  Mode at s = 2 (eigenvalue of A):")
    print(f"  - B = [1, 0]^T: input does NOT excite x2 => mode uncontrollable")
    print(f"  - C = [1, 0]: output does NOT observe x2 => mode unobservable")
    print(f"  The unstable mode is both uncontrollable and unobservable")

    # Part 3: Danger
    print("\nPart 3: Danger of this system")
    print("  Although the system appears BIBO stable from input-output tests,")
    print("  the internal state x2 grows exponentially: x2(t) = x2(0)*exp(2t)")
    print()
    print("  Dangers:")
    print("  1. If x2(0) != 0 (any nonzero initial condition), x2 diverges")
    print("  2. Even tiny perturbations or noise can excite x2")
    print("  3. In practice, the mode is never perfectly decoupled --")
    print("     model uncertainty or nonlinearities can couple the unstable")
    print("     mode to the input/output, causing the system to blow up")
    print("  4. The growing x2 can cause physical damage (e.g., overheating,")
    print("     mechanical failure) even though the measured output looks fine")
    print()
    print("  This is why INTERNAL stability (all eigenvalues in LHP) is")
    print("  the correct requirement for real systems, not just BIBO stability.")

    # Demonstrate
    t = np.linspace(0, 3, 300)
    x2_vals = 0.01 * np.exp(2 * t)  # tiny initial condition on x2
    print(f"\n  Example: x2(0) = 0.01")
    print(f"  x2(1) = {0.01 * np.exp(2):.4f}")
    print(f"  x2(2) = {0.01 * np.exp(4):.4f}")
    print(f"  x2(3) = {0.01 * np.exp(6):.4f}")
    print(f"  Hidden unstable mode grows exponentially!")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Controllability and Observability Check ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: PBH Test ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Hidden Modes ===")
    print("=" * 60)
    exercise_3()

    print("\n\nAll exercises completed!")
