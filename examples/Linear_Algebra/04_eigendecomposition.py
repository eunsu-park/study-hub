"""
Eigenvalues and Eigenvectors

Demonstrates eigendecomposition concepts:
- Eigenvalue computation with np.linalg.eig and eigh
- Matrix diagonalization: A = P D P^{-1}
- Power iteration for dominant eigenvalue
- Spectral analysis of symmetric matrices
- Eigenvalue applications: stability, Markov chains

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def eigenvalue_basics():
    """Compute eigenvalues and eigenvectors."""
    print("=" * 60)
    print("EIGENVALUE BASICS")
    print("=" * 60)

    A = np.array([[4, 1],
                  [2, 3]])

    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\nA:\n{A}")
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors (columns):\n{eigenvectors}")

    # Verify Av = lambda * v for each pair
    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        v = eigenvectors[:, i]
        Av = A @ v
        lam_v = lam * v
        print(f"\nlambda_{i} = {lam:.4f}")
        print(f"v_{i} = {v}")
        print(f"A @ v = {Av}")
        print(f"lambda * v = {lam_v}")
        print(f"Av == lambda*v: {np.allclose(Av, lam_v)}")

    # Properties of eigenvalues
    print(f"\n--- Eigenvalue Properties ---")
    print(f"Sum of eigenvalues: {np.sum(eigenvalues):.4f}")
    print(f"Trace of A: {np.trace(A)}")
    print(f"tr(A) == sum(lambda): {np.isclose(np.trace(A), np.sum(eigenvalues))}")

    print(f"\nProduct of eigenvalues: {np.prod(eigenvalues):.4f}")
    print(f"det(A): {np.linalg.det(A):.4f}")
    print(f"det(A) == prod(lambda): {np.isclose(np.linalg.det(A), np.prod(eigenvalues))}")


def symmetric_eigendecomposition():
    """Eigendecomposition of symmetric matrices (guaranteed real eigenvalues)."""
    print("\n" + "=" * 60)
    print("SYMMETRIC EIGENDECOMPOSITION")
    print("=" * 60)

    # Symmetric matrix: use eigh for numerically stable computation
    A = np.array([[4, 2, 1],
                  [2, 5, 3],
                  [1, 3, 6]])

    eigenvalues, Q = np.linalg.eigh(A)
    print(f"\nSymmetric A:\n{A}")
    print(f"Eigenvalues (ascending): {eigenvalues}")
    print(f"Eigenvectors Q:\n{np.round(Q, 4)}")

    # Spectral theorem: A = Q Lambda Q^T
    Lambda = np.diag(eigenvalues)
    A_reconstructed = Q @ Lambda @ Q.T
    print(f"\nQ @ Lambda @ Q^T:\n{np.round(A_reconstructed, 10)}")
    print(f"A == Q Lambda Q^T: {np.allclose(A, A_reconstructed)}")

    # Eigenvectors are orthonormal
    print(f"\nQ^T @ Q (should be I):\n{np.round(Q.T @ Q, 10)}")
    print(f"Orthonormal eigenvectors: {np.allclose(Q.T @ Q, np.eye(3))}")


def diagonalization():
    """Demonstrate matrix diagonalization A = P D P^{-1}."""
    print("\n" + "=" * 60)
    print("DIAGONALIZATION")
    print("=" * 60)

    A = np.array([[3, 1],
                  [0, 2]])

    eigenvalues, P = np.linalg.eig(A)
    D = np.diag(eigenvalues)
    P_inv = np.linalg.inv(P)

    print(f"\nA:\n{A}")
    print(f"P (eigenvectors):\n{np.round(P, 4)}")
    print(f"D (eigenvalues):\n{np.round(D, 4)}")
    print(f"P^(-1):\n{np.round(P_inv, 4)}")

    # Verify A = P D P^{-1}
    A_from_diag = P @ D @ P_inv
    print(f"\nP @ D @ P^(-1):\n{np.round(A_from_diag, 10)}")
    print(f"A == P D P^(-1): {np.allclose(A, A_from_diag)}")

    # Matrix power via diagonalization: A^n = P D^n P^{-1}
    print("\n--- Matrix Power via Diagonalization ---")
    n = 10
    D_n = np.diag(eigenvalues ** n)
    A_n_diag = P @ D_n @ P_inv
    A_n_direct = np.linalg.matrix_power(A, n)
    print(f"A^{n} via diagonalization:\n{np.round(A_n_diag, 4)}")
    print(f"A^{n} via matrix_power:\n{np.round(A_n_direct, 4)}")
    print(f"Match: {np.allclose(A_n_diag, A_n_direct)}")


def power_iteration():
    """Implement power iteration to find dominant eigenvalue."""
    print("\n" + "=" * 60)
    print("POWER ITERATION")
    print("=" * 60)

    A = np.array([[4, 1],
                  [2, 3]], dtype=float)

    # Power iteration algorithm
    np.random.seed(42)
    v = np.random.randn(2)
    v = v / np.linalg.norm(v)

    print(f"\nA:\n{A}")
    print(f"Initial random vector: {np.round(v, 4)}")

    eigenvalues_exact = np.linalg.eigvalsh(np.array([[4, 1.5], [1.5, 3]]))  # Not needed
    eigenvalues_exact, _ = np.linalg.eig(A)
    dominant = np.max(np.abs(eigenvalues_exact))
    print(f"True dominant eigenvalue: {dominant:.6f}")

    print(f"\n{'Iter':>4}  {'Eigenvalue estimate':>20}  {'Error':>12}")
    print("-" * 40)

    for i in range(15):
        w = A @ v
        lambda_est = np.dot(w, v) / np.dot(v, v)  # Rayleigh quotient
        v = w / np.linalg.norm(w)
        error = abs(lambda_est - dominant)
        print(f"{i+1:4d}  {lambda_est:20.10f}  {error:12.2e}")

    print(f"\nConverged eigenvalue: {lambda_est:.10f}")
    print(f"Corresponding eigenvector: {v}")


def spectral_analysis():
    """Demonstrate spectral analysis and matrix functions."""
    print("\n" + "=" * 60)
    print("SPECTRAL ANALYSIS")
    print("=" * 60)

    # Positive definite matrix
    A = np.array([[5, 2],
                  [2, 3]])

    eigenvalues, Q = np.linalg.eigh(A)
    print(f"\nA:\n{A}")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"All positive (PD): {np.all(eigenvalues > 0)}")

    # Condition number from eigenvalues
    cond = np.max(eigenvalues) / np.min(eigenvalues)
    print(f"Condition number (lambda_max/lambda_min): {cond:.4f}")
    print(f"np.linalg.cond: {np.linalg.cond(A):.4f}")

    # Matrix square root via spectral decomposition
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    A_sqrt = Q @ np.diag(sqrt_eigenvalues) @ Q.T
    print(f"\nMatrix square root A^(1/2):\n{np.round(A_sqrt, 4)}")
    print(f"A^(1/2) @ A^(1/2):\n{np.round(A_sqrt @ A_sqrt, 10)}")
    print(f"(A^(1/2))^2 == A: {np.allclose(A_sqrt @ A_sqrt, A)}")

    # Matrix exponential via eigendecomposition
    exp_eigenvalues = np.exp(eigenvalues)
    A_exp = Q @ np.diag(exp_eigenvalues) @ Q.T
    from scipy.linalg import expm
    A_exp_scipy = expm(A)
    print(f"\nMatrix exponential e^A (spectral):\n{np.round(A_exp, 4)}")
    print(f"Matrix exponential e^A (scipy):\n{np.round(A_exp_scipy, 4)}")
    print(f"Match: {np.allclose(A_exp, A_exp_scipy)}")


def markov_chain():
    """Eigenvalue application: steady-state of Markov chain."""
    print("\n" + "=" * 60)
    print("APPLICATION: MARKOV CHAIN STEADY STATE")
    print("=" * 60)

    # Transition matrix (columns sum to 1)
    # States: Sunny, Cloudy, Rainy
    P = np.array([[0.7, 0.3, 0.2],
                  [0.2, 0.4, 0.3],
                  [0.1, 0.3, 0.5]])
    print(f"\nTransition matrix P:\n{P}")
    print(f"Column sums: {np.sum(P, axis=0)}")

    # Find steady state: P pi = pi (eigenvalue = 1)
    eigenvalues, eigenvectors = np.linalg.eig(P)
    print(f"\nEigenvalues: {np.round(eigenvalues, 6)}")

    # Find eigenvector for eigenvalue = 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.abs(eigenvectors[:, idx])
    pi = pi / np.sum(pi)  # Normalize to probability distribution

    print(f"\nSteady-state distribution:")
    print(f"  Sunny:  {pi[0]:.4f}")
    print(f"  Cloudy: {pi[1]:.4f}")
    print(f"  Rainy:  {pi[2]:.4f}")
    print(f"  Sum:    {np.sum(pi):.4f}")

    # Verify P @ pi = pi
    print(f"\nP @ pi = {np.round(P @ pi, 6)}")
    print(f"pi     = {np.round(pi, 6)}")
    print(f"P @ pi == pi: {np.allclose(P @ pi, pi)}")

    return P, pi


def visualize_eigendecomposition():
    """Visualize eigenvectors and their effect."""
    A = np.array([[2, 1],
                  [1, 3]])

    eigenvalues, eigenvectors = np.linalg.eigh(A)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Eigenvectors and transformation
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle

    ax.plot(circle[0], circle[1], 'b-', alpha=0.3, label='Unit circle')
    ax.plot(ellipse[0], ellipse[1], 'r-', label='A @ circle')

    for i in range(2):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                  color='green', width=0.02, label=f'v{i+1} (lam={lam:.2f})')
        ax.quiver(0, 0, lam*v[0], lam*v[1], angles='xy', scale_units='xy', scale=1,
                  color='orange', width=0.02, alpha=0.7)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('Eigenvectors stretch along principal axes\nA maps circle to ellipse')

    # Plot 2: Power iteration convergence
    ax = axes[1]
    A2 = np.array([[4, 1], [2, 3]], dtype=float)
    np.random.seed(42)
    v = np.random.randn(2)
    v = v / np.linalg.norm(v)
    true_eigenvalue = np.max(np.abs(np.linalg.eigvals(A2)))

    estimates = []
    for _ in range(20):
        w = A2 @ v
        lam_est = np.dot(w, v) / np.dot(v, v)
        estimates.append(lam_est)
        v = w / np.linalg.norm(w)

    errors = np.abs(np.array(estimates) - true_eigenvalue)
    ax.semilogy(range(1, 21), errors, 'bo-', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('|lambda_est - lambda_true|')
    ax.set_title(f'Power Iteration Convergence\nTrue eigenvalue = {true_eigenvalue:.4f}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eigendecomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: eigendecomposition.png")


if __name__ == "__main__":
    eigenvalue_basics()
    symmetric_eigendecomposition()
    diagonalization()
    power_iteration()
    spectral_analysis()
    markov_chain()
    visualize_eigendecomposition()
    print("\nAll examples completed!")
