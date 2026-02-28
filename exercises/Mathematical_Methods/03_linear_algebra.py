"""
Exercise Solutions: Lesson 03 - Linear Algebra
Mathematical Methods for Physical Sciences

Covers: determinants, Gaussian elimination, eigenvalues/eigenvectors,
        quadratic forms, coupled oscillations, spin operators
"""

import numpy as np
from numpy.linalg import det, inv, eig, solve


def exercise_1_determinant_inverse():
    """
    Problem 1: For A = [[2, 1, 3], [1, -1, 2], [3, 2, 1]],
    find det(A) and A^{-1}.
    """
    print("=" * 60)
    print("Problem 1: Determinant and Inverse of 3x3 Matrix")
    print("=" * 60)

    A = np.array([[2, 1, 3],
                  [1, -1, 2],
                  [3, 2, 1]])

    print(f"\nA = \n{A}")

    # Determinant by cofactor expansion along first row
    det_A = det(A)
    print(f"\ndet(A) = 2*(-1*1 - 2*2) - 1*(1*1 - 2*3) + 3*(1*2 - (-1)*3)")
    print(f"       = 2*(-5) - 1*(-5) + 3*(5)")
    print(f"       = -10 + 5 + 15 = 10")
    print(f"  numpy det(A) = {det_A:.6f}")

    # Inverse
    A_inv = inv(A)
    print(f"\nA^(-1) =\n{A_inv}")

    # Verify
    product = A @ A_inv
    print(f"\nVerification A * A^(-1) =\n{np.round(product, 10)}")


def exercise_2_gaussian_elimination():
    """
    Problem 2: Solve the system using Gaussian elimination:
    2x + y - z = 3
    x - y + 2z = 1
    3x + 2y + z = 4
    """
    print("\n" + "=" * 60)
    print("Problem 2: Gaussian Elimination")
    print("=" * 60)

    A = np.array([[2, 1, -1],
                  [1, -1, 2],
                  [3, 2, 1]], dtype=float)
    b = np.array([3, 1, 4], dtype=float)

    print(f"\nSystem:")
    print(f"  2x +  y -  z = 3")
    print(f"   x -  y + 2z = 1")
    print(f"  3x + 2y +  z = 4")

    # Manual Gaussian elimination (showing steps)
    # Augmented matrix
    M = np.column_stack([A.copy(), b.copy()])
    print(f"\nAugmented matrix:\n{M}")

    # Step 1: Swap R1 and R2 for pivot
    M[[0, 1]] = M[[1, 0]]
    print(f"\nAfter R1 <-> R2:\n{M}")

    # Step 2: R2 = R2 - 2*R1, R3 = R3 - 3*R1
    M[1] = M[1] - 2 * M[0]
    M[2] = M[2] - 3 * M[0]
    print(f"\nAfter R2 = R2 - 2*R1, R3 = R3 - 3*R1:\n{M}")

    # Step 3: R3 = R3 - 5/3 * R2
    M[2] = M[2] - (M[2, 1] / M[1, 1]) * M[1]
    print(f"\nAfter eliminating in column 2:\n{M}")

    # Back substitution
    z = M[2, 3] / M[2, 2]
    y = (M[1, 3] - M[1, 2] * z) / M[1, 1]
    x = (M[0, 3] - M[0, 1] * y - M[0, 2] * z) / M[0, 0]

    print(f"\nBack substitution:")
    print(f"  z = {z:.6f}")
    print(f"  y = {y:.6f}")
    print(f"  x = {x:.6f}")

    # Verify with numpy
    solution = solve(A, b)
    print(f"\nnumpy.linalg.solve: x={solution[0]:.6f}, y={solution[1]:.6f}, z={solution[2]:.6f}")

    # Verify
    residual = A @ solution - b
    print(f"Residual |Ax - b| = {np.linalg.norm(residual):.2e}")


def exercise_3_eigenvalues():
    """
    Problem 3: Find eigenvalues and eigenvectors of A = [[2, 1], [1, 2]].
    """
    print("\n" + "=" * 60)
    print("Problem 3: Eigenvalues and Eigenvectors")
    print("=" * 60)

    A = np.array([[2, 1],
                  [1, 2]], dtype=float)

    print(f"\nA = [[2, 1], [1, 2]]")

    # Characteristic equation: det(A - lambda*I) = 0
    # (2-lambda)^2 - 1 = 0
    # lambda^2 - 4*lambda + 3 = 0
    # (lambda - 1)(lambda - 3) = 0
    print("\nCharacteristic equation:")
    print("  det(A - lambda*I) = (2-lambda)^2 - 1 = 0")
    print("  lambda^2 - 4*lambda + 3 = 0")
    print("  (lambda - 1)(lambda - 3) = 0")

    eigenvalues, eigenvectors = eig(A)

    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        vec = eigenvectors[:, i]
        print(f"\n  lambda_{i+1} = {lam:.1f}")
        print(f"  eigenvector: [{vec[0]:+.6f}, {vec[1]:+.6f}]")
        # Verify
        Av = A @ vec
        lv = lam * vec
        print(f"  Verify: A*v = [{Av[0]:+.6f}, {Av[1]:+.6f}]")
        print(f"          l*v = [{lv[0]:+.6f}, {lv[1]:+.6f}]")

    print("\n  lambda_1 = 1: eigenvector (1, -1)/sqrt(2) [antisymmetric]")
    print("  lambda_2 = 3: eigenvector (1, 1)/sqrt(2)  [symmetric]")


def exercise_4_quadratic_form():
    """
    Problem 4: Determine the definiteness of the quadratic form
    Q = 3x^2 + 2xy + 3y^2.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Quadratic Form Definiteness")
    print("=" * 60)

    # Q = 3x^2 + 2xy + 3y^2
    # Matrix form: Q = [x, y] * [[3, 1], [1, 3]] * [x, y]^T
    A = np.array([[3, 1],
                  [1, 3]], dtype=float)

    print(f"\nQ = 3x^2 + 2xy + 3y^2")
    print(f"Matrix: A = [[3, 1], [1, 3]]")

    eigenvalues = np.sort(np.linalg.eigvalsh(A))
    print(f"\nEigenvalues: {eigenvalues}")

    # Check definiteness
    if np.all(eigenvalues > 0):
        print("All eigenvalues positive => Q is POSITIVE DEFINITE")
    elif np.all(eigenvalues >= 0):
        print("All eigenvalues non-negative => Q is POSITIVE SEMI-DEFINITE")
    elif np.all(eigenvalues < 0):
        print("All eigenvalues negative => Q is NEGATIVE DEFINITE")
    else:
        print("Mixed signs => Q is INDEFINITE")

    # Sylvester's criterion: leading principal minors
    d1 = A[0, 0]
    d2 = det(A)
    print(f"\nSylvester's criterion (leading principal minors):")
    print(f"  d1 = {d1} > 0  [check]")
    print(f"  d2 = det(A) = {d2:.1f} > 0  [check]")
    print(f"  All positive => POSITIVE DEFINITE")

    # Minimum value of Q
    print(f"\n  Since lambda_min = {eigenvalues[0]:.1f}:")
    print(f"  Q >= {eigenvalues[0]:.1f} * (x^2 + y^2) for all (x, y)")
    print(f"  Q = 0 only at (x, y) = (0, 0)")


def exercise_5_coupled_oscillations():
    """
    Problem 5: Coupled spring-mass system.
    Two masses m1, m2 connected by springs k1, k2, k3.
    k1=k3=k, k2=k' (coupling spring), m1=m2=m.
    Find normal modes.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Coupled Oscillations")
    print("=" * 60)

    # Equations of motion: M*x_ddot = -K*x
    # For m1 = m2 = m = 1, k1 = k3 = k = 1, k2 = k' = 0.5:
    m = 1.0
    k = 1.0
    k_coupling = 0.5

    # Stiffness matrix
    K = np.array([[k + k_coupling, -k_coupling],
                  [-k_coupling, k + k_coupling]])

    M = np.array([[m, 0],
                  [0, m]])

    print(f"\nParameters: m1 = m2 = {m}, k1 = k3 = {k}, k2 (coupling) = {k_coupling}")
    print(f"\nStiffness matrix K =\n{K}")
    print(f"Mass matrix M =\n{M}")

    # Generalized eigenvalue problem: K*v = omega^2 * M*v
    # Since M = m*I, this simplifies to K*v = m*omega^2 * v
    eigenvalues, eigenvectors = eig(np.linalg.inv(M) @ K)

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"\nNormal mode frequencies:")
    for i in range(len(eigenvalues)):
        omega_sq = eigenvalues[i]
        omega = np.sqrt(omega_sq)
        vec = eigenvectors[:, i]
        # Normalize
        vec = vec / np.abs(vec).max()
        print(f"\n  Mode {i+1}:")
        print(f"    omega^2 = {omega_sq:.4f}")
        print(f"    omega   = {omega:.4f}")
        print(f"    Eigenvector: [{vec[0]:+.4f}, {vec[1]:+.4f}]")

    print("\n  Mode 1 (in-phase): masses move together, coupling spring undeformed")
    print("  Mode 2 (out-of-phase): masses move oppositely, coupling spring active")


def exercise_6_spin_operator():
    """
    Problem 6: Spin-1/2 measurement operator.
    Pauli matrices: sigma_x, sigma_y, sigma_z.
    Find eigenvalues and eigenvectors of sigma_x.
    Probability of measuring spin-up along x for state |+z>.
    """
    print("\n" + "=" * 60)
    print("Problem 6: Quantum Spin-1/2 Measurement")
    print("=" * 60)

    # Pauli matrices (hbar/2 factor omitted)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    print("\nPauli matrices:")
    print(f"  sigma_x = [[0, 1], [1, 0]]")
    print(f"  sigma_y = [[0, -i], [i, 0]]")
    print(f"  sigma_z = [[1, 0], [0, -1]]")

    # Eigenvalues of sigma_x
    eigenvalues, eigenvectors = eig(sigma_x)

    print(f"\nEigenvalues of sigma_x:")
    for i in range(2):
        lam = eigenvalues[i].real
        vec = eigenvectors[:, i]
        print(f"  lambda_{i+1} = {lam:+.1f}")
        print(f"  eigenvector: |{'+' if lam > 0 else '-'}x> = "
              f"[{vec[0].real:+.4f}, {vec[1].real:+.4f}] / sqrt(2)")

    # |+z> = [1, 0] state
    psi_z_up = np.array([1, 0], dtype=complex)

    print(f"\n  State: |+z> = [1, 0]")

    # Probability of measuring +1 (spin-up along x)
    # |+x> = [1, 1]/sqrt(2)
    plus_x = np.array([1, 1], dtype=complex) / np.sqrt(2)
    prob_plus = abs(np.dot(plus_x.conj(), psi_z_up))**2

    minus_x = np.array([1, -1], dtype=complex) / np.sqrt(2)
    prob_minus = abs(np.dot(minus_x.conj(), psi_z_up))**2

    print(f"  P(+x) = |<+x|+z>|^2 = |1/sqrt(2)|^2 = {prob_plus:.4f}")
    print(f"  P(-x) = |<-x|+z>|^2 = |1/sqrt(2)|^2 = {prob_minus:.4f}")
    print(f"  P(+x) + P(-x) = {prob_plus + prob_minus:.4f}  [normalization check]")

    # Expectation value
    exp_val = psi_z_up.conj() @ sigma_x @ psi_z_up
    print(f"\n  <sigma_x> = <+z|sigma_x|+z> = {exp_val.real:.4f}")
    print(f"  (Expected: 0, since |+z> is symmetric in x)")

    # Verify commutation relations
    comm_xy = sigma_x @ sigma_y - sigma_y @ sigma_x
    print(f"\n  Commutation: [sigma_x, sigma_y] = 2i*sigma_z")
    print(f"  Computed: {comm_xy}")
    print(f"  2i*sigma_z: {2j * sigma_z}")
    print(f"  Match: {np.allclose(comm_xy, 2j * sigma_z)}")


if __name__ == "__main__":
    exercise_1_determinant_inverse()
    exercise_2_gaussian_elimination()
    exercise_3_eigenvalues()
    exercise_4_quadratic_form()
    exercise_5_coupled_oscillations()
    exercise_6_spin_operator()
