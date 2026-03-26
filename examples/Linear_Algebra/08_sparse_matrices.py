"""
Sparse Matrices with SciPy

Demonstrates sparse matrix concepts:
- Creating sparse matrices: CSR, CSC, COO, LIL formats
- Sparse matrix arithmetic and slicing
- Format conversion and efficiency comparison
- Sparse direct solvers (spsolve)
- Sparse eigenvalue computation (eigsh)

Dependencies: numpy, scipy
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import time


def sparse_creation():
    """Create sparse matrices in various formats."""
    print("=" * 60)
    print("SPARSE MATRIX CREATION")
    print("=" * 60)

    # COO format: coordinate list (row, col, data)
    print("--- COO Format (Coordinate) ---")
    row = np.array([0, 0, 1, 2, 2, 3])
    col = np.array([0, 2, 1, 0, 2, 3])
    data = np.array([4, 1, 3, 2, 5, 7], dtype=float)
    A_coo = sparse.coo_matrix((data, (row, col)), shape=(4, 4))
    print(f"COO matrix:\n{A_coo.toarray()}")
    print(f"nnz (non-zeros): {A_coo.nnz}")
    print(f"Density: {A_coo.nnz / np.prod(A_coo.shape):.2%}")

    # CSR format: Compressed Sparse Row
    print("\n--- CSR Format (Compressed Sparse Row) ---")
    A_csr = A_coo.tocsr()
    print(f"indptr (row pointers): {A_csr.indptr}")
    print(f"indices (column indices): {A_csr.indices}")
    print(f"data: {A_csr.data}")
    print(f"Row 0: columns {A_csr.indices[A_csr.indptr[0]:A_csr.indptr[1]]}")

    # CSC format: Compressed Sparse Column
    print("\n--- CSC Format (Compressed Sparse Column) ---")
    A_csc = A_coo.tocsc()
    print(f"indptr (col pointers): {A_csc.indptr}")
    print(f"indices (row indices): {A_csc.indices}")

    # LIL format: List of Lists (good for construction)
    print("\n--- LIL Format (List of Lists) ---")
    A_lil = sparse.lil_matrix((4, 4))
    A_lil[0, 0] = 4
    A_lil[0, 2] = 1
    A_lil[1, 1] = 3
    A_lil[2, 0] = 2
    A_lil[2, 2] = 5
    A_lil[3, 3] = 7
    print(f"LIL matrix:\n{A_lil.toarray()}")

    # Identity and diagonal
    print("\n--- Special Sparse Matrices ---")
    I = sparse.eye(5)
    print(f"Sparse identity (5x5): nnz = {I.nnz}")

    D = sparse.diags([1, 2, 3, 4, 5])
    print(f"Sparse diagonal: nnz = {D.nnz}")

    # Tridiagonal matrix
    T = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(5, 5))
    print(f"Tridiagonal:\n{T.toarray()}")


def sparse_operations():
    """Demonstrate sparse matrix arithmetic and operations."""
    print("\n" + "=" * 60)
    print("SPARSE MATRIX OPERATIONS")
    print("=" * 60)

    A = sparse.csr_matrix(np.array([[1, 0, 2],
                                     [0, 3, 0],
                                     [4, 0, 5]], dtype=float))
    B = sparse.csr_matrix(np.array([[0, 1, 0],
                                     [2, 0, 3],
                                     [0, 4, 0]], dtype=float))

    print(f"A:\n{A.toarray()}")
    print(f"B:\n{B.toarray()}")

    # Addition
    C = A + B
    print(f"\nA + B:\n{C.toarray()}")
    print(f"nnz: A={A.nnz}, B={B.nnz}, A+B={C.nnz}")

    # Multiplication
    D = A @ B
    print(f"\nA @ B:\n{D.toarray()}")

    # Scalar multiplication
    print(f"\n3 * A:\n{(3 * A).toarray()}")

    # Matrix-vector product
    x = np.array([1, 2, 3])
    y = A @ x
    print(f"\nA @ x = {y}")

    # Transpose
    print(f"\nA^T:\n{A.T.toarray()}")

    # Element-wise multiply
    E = A.multiply(B)
    print(f"\nA .* B (element-wise):\n{E.toarray()}")


def format_comparison():
    """Compare efficiency of different sparse formats."""
    print("\n" + "=" * 60)
    print("FORMAT EFFICIENCY COMPARISON")
    print("=" * 60)

    n = 1000
    density = 0.01

    # Create random sparse matrix
    np.random.seed(42)
    A_coo = sparse.random(n, n, density=density, format='coo')
    x = np.random.randn(n)

    print(f"Matrix size: {n}x{n}")
    print(f"Density: {density}")
    print(f"Non-zeros: {A_coo.nnz}")
    print(f"Dense storage: {n*n*8 / 1024:.1f} KB")
    print(f"Sparse storage (COO): ~{A_coo.nnz * (8 + 4 + 4) / 1024:.1f} KB")

    formats = {
        'COO': A_coo.tocoo(),
        'CSR': A_coo.tocsr(),
        'CSC': A_coo.tocsc(),
    }

    # Matrix-vector product timing
    print(f"\n{'Format':>6}  {'matvec time (ms)':>16}  {'row_slice time (ms)':>20}")
    print("-" * 50)

    for name, A_fmt in formats.items():
        # Matvec
        start = time.perf_counter()
        for _ in range(100):
            _ = A_fmt @ x
        t_matvec = (time.perf_counter() - start) / 100 * 1000

        # Row slicing (CSR is fast, others may be slow)
        start = time.perf_counter()
        A_csr_tmp = A_fmt.tocsr() if name != 'CSR' else A_fmt
        for _ in range(100):
            _ = A_csr_tmp[42, :]
        t_row = (time.perf_counter() - start) / 100 * 1000

        print(f"{name:>6}  {t_matvec:16.4f}  {t_row:20.4f}")

    # Dense comparison
    A_dense = A_coo.toarray()
    start = time.perf_counter()
    for _ in range(100):
        _ = A_dense @ x
    t_dense = (time.perf_counter() - start) / 100 * 1000
    print(f"{'Dense':>6}  {t_dense:16.4f}")


def sparse_solvers():
    """Demonstrate sparse direct solvers."""
    print("\n" + "=" * 60)
    print("SPARSE DIRECT SOLVERS")
    print("=" * 60)

    # Create SPD banded system (common in FEM/FDM)
    n = 500
    diagonals = [
        -np.ones(n - 1),
        2 * np.ones(n),
        -np.ones(n - 1)
    ]
    A = sparse.diags(diagonals, [-1, 0, 1], format='csc')
    b = np.ones(n)

    print(f"System size: {n}x{n}")
    print(f"Non-zeros: {A.nnz}")

    # Sparse direct solve
    start = time.perf_counter()
    x_sparse = splinalg.spsolve(A, b)
    t_sparse = (time.perf_counter() - start) * 1000

    # Dense solve for comparison
    A_dense = A.toarray()
    start = time.perf_counter()
    x_dense = np.linalg.solve(A_dense, b)
    t_dense = (time.perf_counter() - start) * 1000

    print(f"\nSparse solve: {t_sparse:.2f} ms")
    print(f"Dense solve:  {t_dense:.2f} ms")
    print(f"Speedup: {t_dense / t_sparse:.1f}x")
    print(f"Solutions match: {np.allclose(x_sparse, x_dense)}")
    print(f"||Ax - b|| (sparse): {np.linalg.norm(A @ x_sparse - b):.2e}")


def sparse_eigenvalues():
    """Compute a few eigenvalues of large sparse matrix."""
    print("\n" + "=" * 60)
    print("SPARSE EIGENVALUE COMPUTATION")
    print("=" * 60)

    n = 1000
    # Symmetric tridiagonal (1D Laplacian)
    A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csr')

    print(f"Matrix size: {n}x{n}")
    print(f"Non-zeros: {A.nnz}")

    # Compute 6 smallest eigenvalues
    k = 6
    start = time.perf_counter()
    eigenvalues, eigenvectors = splinalg.eigsh(A, k=k, which='SM')
    t_sparse = (time.perf_counter() - start) * 1000

    print(f"\n{k} smallest eigenvalues (sparse eigsh): {np.round(eigenvalues, 6)}")
    print(f"Time: {t_sparse:.2f} ms")

    # Analytical eigenvalues for comparison
    # lambda_j = 2 - 2*cos(j*pi/(n+1)) for j = 1, ..., n
    analytical = np.array([2 - 2 * np.cos(j * np.pi / (n + 1)) for j in range(1, k + 1)])
    print(f"Analytical eigenvalues: {np.round(analytical, 6)}")
    print(f"Max error: {np.max(np.abs(np.sort(eigenvalues) - analytical)):.2e}")

    # Compute 6 largest eigenvalues
    eigenvalues_large = splinalg.eigsh(A, k=k, which='LM')[0]
    print(f"\n{k} largest eigenvalues: {np.round(eigenvalues_large, 6)}")


if __name__ == "__main__":
    sparse_creation()
    sparse_operations()
    format_comparison()
    sparse_solvers()
    sparse_eigenvalues()
    print("\nAll examples completed!")
