"""
Matrix Operations with NumPy

Demonstrates fundamental matrix operations:
- Matrix multiplication and element-wise operations
- Transpose, inverse, and pseudoinverse
- Determinant and its geometric interpretation
- Rank, trace, and matrix norms
- Special matrices: symmetric, orthogonal, idempotent

Dependencies: numpy
"""

import numpy as np


def matrix_creation():
    """Demonstrate matrix creation and basic properties."""
    print("=" * 60)
    print("MATRIX CREATION AND BASIC PROPERTIES")
    print("=" * 60)

    # Creating matrices
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print(f"\nMatrix A (2x3):\n{A}")
    print(f"Shape: {A.shape}")
    print(f"Size (total elements): {A.size}")
    print(f"Number of dimensions: {A.ndim}")

    # Special matrices
    print("\n--- Special Matrices ---")
    I = np.eye(3)
    print(f"Identity I_3:\n{I}")

    Z = np.zeros((2, 3))
    print(f"\nZero matrix (2x3):\n{Z}")

    D = np.diag([1, 2, 3])
    print(f"\nDiagonal matrix:\n{D}")

    # Extract diagonal from matrix
    print(f"Diagonal of A: {np.diag(A[:, :2])}")


def matrix_multiplication():
    """Demonstrate matrix multiplication and its properties."""
    print("\n" + "=" * 60)
    print("MATRIX MULTIPLICATION")
    print("=" * 60)

    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])

    # Matrix multiplication (3 equivalent ways)
    C1 = A @ B
    C2 = np.matmul(A, B)
    C3 = np.dot(A, B)
    print(f"\nA:\n{A}")
    print(f"B:\n{B}")
    print(f"A @ B:\n{C1}")
    print(f"All methods agree: {np.allclose(C1, C2) and np.allclose(C2, C3)}")

    # Non-commutativity: AB != BA in general
    print(f"\nB @ A:\n{B @ A}")
    print(f"AB == BA? {np.allclose(A @ B, B @ A)}")

    # Element-wise multiplication (Hadamard product)
    print(f"\nA * B (element-wise):\n{A * B}")

    # Associativity: (AB)C = A(BC)
    C = np.array([[1, 0], [0, 1]])
    print(f"\nAssociativity: (AB)C == A(BC)? "
          f"{np.allclose((A @ B) @ C, A @ (B @ C))}")

    # Distributivity: A(B+C) = AB + AC
    print(f"Distributivity: A(B+C) == AB+AC? "
          f"{np.allclose(A @ (B + C), A @ B + A @ C)}")

    # Matrix-vector multiplication
    x = np.array([1, 2])
    print(f"\nMatrix-vector product:")
    print(f"A @ x = {A} @ {x} = {A @ x}")

    # Non-square multiplication
    M = np.array([[1, 2, 3],
                  [4, 5, 6]])  # 2x3
    N = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])     # 3x2
    print(f"\nM (2x3) @ N (3x2) = (2x2):\n{M @ N}")
    print(f"N (3x2) @ M (2x3) = (3x3):\n{N @ M}")


def transpose_and_inverse():
    """Demonstrate transpose, inverse, and related operations."""
    print("\n" + "=" * 60)
    print("TRANSPOSE AND INVERSE")
    print("=" * 60)

    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print(f"\nA:\n{A}")
    print(f"A^T:\n{A.T}")
    print(f"(A^T)^T == A: {np.allclose(A.T.T, A)}")

    # (AB)^T = B^T A^T
    B = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    AB = A @ B
    print(f"\n(AB)^T:\n{AB.T}")
    print(f"B^T A^T:\n{B.T @ A.T}")
    print(f"(AB)^T == B^T A^T: {np.allclose(AB.T, B.T @ A.T)}")

    # Matrix inverse
    print("\n--- Matrix Inverse ---")
    M = np.array([[2, 1],
                  [5, 3]])
    M_inv = np.linalg.inv(M)
    print(f"M:\n{M}")
    print(f"M^(-1):\n{M_inv}")
    print(f"M @ M^(-1):\n{np.round(M @ M_inv, 10)}")
    print(f"M^(-1) @ M:\n{np.round(M_inv @ M, 10)}")

    # (AB)^{-1} = B^{-1} A^{-1}
    A_sq = np.array([[1, 2], [3, 4]])
    B_sq = np.array([[5, 6], [7, 8]])
    print(f"\n(AB)^(-1) == B^(-1) A^(-1): "
          f"{np.allclose(np.linalg.inv(A_sq @ B_sq), np.linalg.inv(B_sq) @ np.linalg.inv(A_sq))}")

    # Pseudoinverse for non-square matrices
    print("\n--- Pseudoinverse ---")
    C = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    C_pinv = np.linalg.pinv(C)
    print(f"C (3x2):\n{C}")
    print(f"C^+ (2x3):\n{np.round(C_pinv, 4)}")
    print(f"C^+ @ C:\n{np.round(C_pinv @ C, 4)}")


def determinant_and_trace():
    """Demonstrate determinant and trace computations."""
    print("\n" + "=" * 60)
    print("DETERMINANT AND TRACE")
    print("=" * 60)

    A = np.array([[3, 1],
                  [2, 4]])
    print(f"\nA:\n{A}")

    # Determinant
    det_A = np.linalg.det(A)
    print(f"det(A) = {det_A:.4f}")
    print(f"Manual (ad - bc): {A[0,0]*A[1,1] - A[0,1]*A[1,0]}")

    # Determinant properties
    B = np.array([[1, 2],
                  [3, 4]])
    print(f"\ndet(A) * det(B) = {np.linalg.det(A) * np.linalg.det(B):.4f}")
    print(f"det(AB) = {np.linalg.det(A @ B):.4f}")
    print(f"det(AB) == det(A)*det(B): {np.isclose(np.linalg.det(A @ B), np.linalg.det(A) * np.linalg.det(B))}")

    print(f"\ndet(A^T) = {np.linalg.det(A.T):.4f}")
    print(f"det(A^T) == det(A): {np.isclose(np.linalg.det(A), np.linalg.det(A.T))}")

    # Singular matrix (det = 0)
    S = np.array([[1, 2],
                  [2, 4]])
    print(f"\nSingular matrix S:\n{S}")
    print(f"det(S) = {np.linalg.det(S):.4f} (rows are linearly dependent)")

    # 3x3 determinant
    C = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 0]])
    print(f"\n3x3 matrix C:\n{C}")
    print(f"det(C) = {np.linalg.det(C):.4f}")

    # Trace
    print("\n--- Trace ---")
    print(f"tr(A) = {np.trace(A)}")
    print(f"Manual (sum of diagonal): {A[0,0] + A[1,1]}")

    # Trace properties
    print(f"\ntr(A + B) = {np.trace(A + B)}")
    print(f"tr(A) + tr(B) = {np.trace(A) + np.trace(B)}")
    print(f"tr(AB) = {np.trace(A @ B):.4f}")
    print(f"tr(BA) = {np.trace(B @ A):.4f}")
    print(f"tr(AB) == tr(BA): {np.isclose(np.trace(A @ B), np.trace(B @ A))}")


def rank_computation():
    """Demonstrate rank computation and its implications."""
    print("\n" + "=" * 60)
    print("MATRIX RANK")
    print("=" * 60)

    # Full rank matrix
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 0]])
    print(f"\nA:\n{A}")
    print(f"rank(A) = {np.linalg.matrix_rank(A)}")

    # Rank-deficient matrix
    B = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [5, 7, 9]])  # row3 = row1 + row2
    print(f"\nB (row3 = row1 + row2):\n{B}")
    print(f"rank(B) = {np.linalg.matrix_rank(B)}")

    # Rank of non-square matrix
    C = np.array([[1, 0, 1],
                  [0, 1, 1]])
    print(f"\nC (2x3):\n{C}")
    print(f"rank(C) = {np.linalg.matrix_rank(C)} (at most min(m,n) = {min(C.shape)})")

    # Rank and invertibility
    print("\n--- Rank and Invertibility ---")
    M = np.array([[1, 2], [3, 4]])
    print(f"M:\n{M}")
    print(f"rank(M) = {np.linalg.matrix_rank(M)}, n = {M.shape[0]}")
    print(f"Invertible: {np.linalg.matrix_rank(M) == M.shape[0]}")


def matrix_norms():
    """Demonstrate matrix norms."""
    print("\n" + "=" * 60)
    print("MATRIX NORMS")
    print("=" * 60)

    A = np.array([[1, 2],
                  [3, 4]])
    print(f"\nA:\n{A}")

    # Frobenius norm (element-wise L2)
    frob = np.linalg.norm(A, 'fro')
    print(f"\nFrobenius norm: {frob:.4f}")
    print(f"Manual: sqrt(sum(a_ij^2)) = {np.sqrt(np.sum(A**2)):.4f}")

    # Operator norms (induced)
    norm_1 = np.linalg.norm(A, 1)     # max column sum
    norm_inf = np.linalg.norm(A, np.inf)  # max row sum
    norm_2 = np.linalg.norm(A, 2)     # largest singular value

    print(f"\n1-norm (max col sum): {norm_1}")
    print(f"inf-norm (max row sum): {norm_inf}")
    print(f"2-norm (spectral): {norm_2:.4f}")

    # Nuclear norm (sum of singular values)
    svs = np.linalg.svd(A, compute_uv=False)
    nuclear = np.sum(svs)
    print(f"\nSingular values: {svs}")
    print(f"Nuclear norm (sum of SVs): {nuclear:.4f}")


def special_matrices():
    """Demonstrate properties of special matrices."""
    print("\n" + "=" * 60)
    print("SPECIAL MATRICES")
    print("=" * 60)

    # Symmetric matrix: A = A^T
    print("--- Symmetric Matrix ---")
    A = np.array([[2, 1, 3],
                  [1, 4, 5],
                  [3, 5, 6]])
    print(f"A:\n{A}")
    print(f"A == A^T: {np.allclose(A, A.T)}")

    # Symmetric matrices have real eigenvalues
    eigenvalues = np.linalg.eigvalsh(A)
    print(f"Eigenvalues (all real): {np.round(eigenvalues, 4)}")

    # Orthogonal matrix: Q^T Q = I, det(Q) = +/-1
    print("\n--- Orthogonal Matrix ---")
    theta = np.pi / 4
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    print(f"Rotation by 45 degrees:\n{np.round(Q, 4)}")
    print(f"Q^T @ Q:\n{np.round(Q.T @ Q, 10)}")
    print(f"det(Q) = {np.linalg.det(Q):.4f}")

    # Orthogonal matrices preserve norms
    v = np.array([3, 4])
    print(f"\n||v|| = {np.linalg.norm(v):.4f}")
    print(f"||Qv|| = {np.linalg.norm(Q @ v):.4f}")

    # Idempotent matrix: P^2 = P
    print("\n--- Idempotent (Projection) Matrix ---")
    v_proj = np.array([1, 1]) / np.sqrt(2)
    P = np.outer(v_proj, v_proj)
    print(f"P = v v^T (projection onto [1,1]):\n{np.round(P, 4)}")
    print(f"P^2 == P: {np.allclose(P @ P, P)}")

    # Positive definite matrix
    print("\n--- Positive Definite Matrix ---")
    M = np.array([[2, -1],
                  [-1, 2]])
    eigenvalues_m = np.linalg.eigvalsh(M)
    print(f"M:\n{M}")
    print(f"Eigenvalues: {eigenvalues_m}")
    print(f"All positive: {np.all(eigenvalues_m > 0)} (positive definite)")

    # x^T M x > 0 for all nonzero x
    x = np.array([1, 2])
    print(f"x^T M x = {x @ M @ x} > 0")


if __name__ == "__main__":
    matrix_creation()
    matrix_multiplication()
    transpose_and_inverse()
    determinant_and_trace()
    rank_computation()
    matrix_norms()
    special_matrices()
    print("\nAll examples completed!")
