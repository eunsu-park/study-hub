"""
Jacobian and Hessian

Demonstrates second-order derivative concepts:
- Numerical Jacobian computation
- Softmax Jacobian (analytical vs numerical)
- Hessian computation and eigenvalue analysis
- Hessian-vector products
- Newton's method vs gradient descent

Dependencies: numpy, matplotlib
"""

import numpy as np


def numerical_jacobian(f, x, eps=1e-5):
    """Compute Jacobian of f: R^n -> R^m by central differences."""
    f0 = np.asarray(f(x))
    n, m = len(x), len(f0)
    J = np.zeros((m, n))
    for j in range(n):
        e = np.zeros(n); e[j] = eps
        J[:, j] = (f(x + e) - f(x - e)) / (2 * eps)
    return J


def softmax_jacobian_demo():
    """Compare analytical and numerical softmax Jacobian."""
    print("=" * 60)
    print("SOFTMAX JACOBIAN")
    print("=" * 60)

    def softmax(z):
        e = np.exp(z - np.max(z))
        return e / e.sum()

    z = np.array([1.0, 2.0, 3.0])
    s = softmax(z)
    J_ana = np.diag(s) - np.outer(s, s)
    J_num = numerical_jacobian(softmax, z)

    print(f"Softmax output: {s.round(4)}")
    print(f"Row sums of Jacobian: {J_ana.sum(axis=1).round(6)}")
    print(f"Jacobian symmetric: {np.allclose(J_ana, J_ana.T)}")
    print(f"Max error: {np.max(np.abs(J_ana - J_num)):.2e}")
    print(f"Rank: {np.linalg.matrix_rank(J_ana)} (expected {len(z)-1})")


def hessian_analysis():
    """Compute Hessian and classify critical points."""
    print("\n" + "=" * 60)
    print("HESSIAN ANALYSIS")
    print("=" * 60)

    def f_saddle(x):
        return x[0]**2 - x[1]**2

    def f_minimum(x):
        return x[0]**2 + 2*x[1]**2

    for name, f in [('Saddle x^2-y^2', f_saddle), ('Minimum x^2+2y^2', f_minimum)]:
        x0 = np.array([0.0, 0.0])
        eps = 1e-5
        H = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                ei, ej = np.zeros(2), np.zeros(2)
                ei[i] = eps; ej[j] = eps
                H[i,j] = (f(x0+ei+ej) - f(x0+ei-ej) - f(x0-ei+ej) + f(x0-ei-ej)) / (4*eps**2)
        eigvals = np.linalg.eigvalsh(0.5*(H+H.T))
        ctype = 'minimum' if all(eigvals > 0) else ('maximum' if all(eigvals < 0) else 'saddle')
        print(f"  {name}: eigenvalues = {eigvals.round(2)}, type = {ctype}")


def hessian_vector_product():
    """Compute H@v via finite differences of gradient."""
    print("\n" + "=" * 60)
    print("HESSIAN-VECTOR PRODUCT")
    print("=" * 60)

    def f(x):
        return x[0]**3 + x[1]**3 + x[0]*x[1]

    def grad_f(x):
        return np.array([3*x[0]**2 + x[1], 3*x[1]**2 + x[0]])

    x0 = np.array([1.0, 2.0])
    v = np.array([1.0, 0.5])

    # Analytical Hessian
    H = np.array([[6*x0[0], 1.0], [1.0, 6*x0[1]]])
    Hv_exact = H @ v

    # HVP via finite differences
    eps = 1e-4
    Hv_fd = (grad_f(x0 + eps*v) - grad_f(x0 - eps*v)) / (2*eps)

    print(f"H@v exact:    {Hv_exact}")
    print(f"H@v finite-d: {Hv_fd}")
    print(f"Error: {np.linalg.norm(Hv_exact - Hv_fd):.2e}")


def newton_vs_gd():
    """Compare Newton's method and gradient descent."""
    print("\n" + "=" * 60)
    print("NEWTON'S METHOD VS GRADIENT DESCENT")
    print("=" * 60)

    A = np.array([[10.0, 3.0], [3.0, 2.0]])
    b = np.array([1.0, 2.0])
    x_star = np.linalg.solve(A, b)

    # GD
    x = np.array([5.0, -3.0])
    for t in range(100):
        x = x - 0.05 * (A @ x - b)
    err_gd = np.linalg.norm(x - x_star)

    # Newton
    x = np.array([5.0, -3.0])
    for t in range(3):
        x = x - np.linalg.solve(A, A @ x - b)
    err_nt = np.linalg.norm(x - x_star)

    print(f"GD (100 steps):    error = {err_gd:.2e}")
    print(f"Newton (3 steps):  error = {err_nt:.2e}")
    print(f"Condition number:  {np.linalg.cond(A):.2f}")


if __name__ == "__main__":
    softmax_jacobian_demo()
    hessian_analysis()
    hessian_vector_product()
    newton_vs_gd()
