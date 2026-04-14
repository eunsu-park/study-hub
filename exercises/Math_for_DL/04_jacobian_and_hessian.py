"""
Exercises for Lesson 04: Jacobian and Hessian
Topic: Math_for_DL

Complete the TODO sections.
"""

import numpy as np


def exercise_1_analytical_jacobian():
    """Compute Jacobian of f(x,y) = (x^2*y, sin(xy), e^x + y) at (1, 2)."""
    x, y = 1.0, 2.0

    # TODO: Compute the 3x2 Jacobian analytically
    # Row 0: d(x^2*y)/dx, d(x^2*y)/dy
    # Row 1: d(sin(xy))/dx, d(sin(xy))/dy
    # Row 2: d(e^x+y)/dx, d(e^x+y)/dy
    J = None  # Replace with np.array([[...], [...], [...]])

    return J


def exercise_2_hessian_critical_points():
    """Compute Hessian of f(x,y) = x^4 + y^4 - 2x^2*y^2 and classify critical points.

    Critical points are at (0,0), and along x=y and x=-y.
    Classify (0,0) using the Hessian eigenvalues.
    """
    # TODO: Compute the Hessian at (0, 0)
    # d^2f/dx^2 = 12x^2 - 4y^2
    # d^2f/dxdy = -4xy (wait, check: d/dy(4x^3 - 4xy^2) = -4*2xy... recompute)
    # Actually: df/dx = 4x^3 - 4xy^2, df/dy = 4y^3 - 4x^2y
    # d^2f/dx^2 = 12x^2 - 4y^2
    # d^2f/dy^2 = 12y^2 - 4x^2
    # d^2f/dxdy = -8xy
    H_at_origin = None  # Replace with 2x2 array

    if H_at_origin is not None:
        eigvals = np.linalg.eigvalsh(H_at_origin)
        return H_at_origin, eigvals
    return None, None


def exercise_3_hessian_vector_product():
    """Implement HVP via finite differences and find largest eigenvalue via power iteration."""
    def f(x):
        return 0.5 * x @ np.array([[4, 1], [1, 3]]) @ x

    def grad_f(x):
        return np.array([[4, 1], [1, 3]]) @ x

    x0 = np.array([1.0, 1.0])

    def hvp(v, eps=1e-4):
        # TODO: Compute H @ v using finite differences of grad_f
        return None  # Replace

    # TODO: Power iteration to find largest eigenvalue
    # Start with random v, iterate: v = hvp(v) / ||hvp(v)||, lambda = v^T H v
    largest_eigenvalue = None

    return largest_eigenvalue


if __name__ == "__main__":
    print("Exercise 1: Analytical Jacobian")
    J = exercise_1_analytical_jacobian()
    if J is not None:
        x, y = 1.0, 2.0
        expected = np.array([[2*x*y, x**2],
                             [y*np.cos(x*y), x*np.cos(x*y)],
                             [np.exp(x), 1.0]])
        print(f"  Jacobian:\n{J.round(4)}")
        print(f"  Pass: {np.allclose(J, expected, atol=1e-4)}")
    else:
        print("  Not implemented yet")

    print("\nExercise 2: Hessian critical points")
    H, eigvals = exercise_2_hessian_critical_points()
    if H is not None:
        print(f"  H at (0,0):\n{H}")
        print(f"  Eigenvalues: {eigvals}")
        if np.all(eigvals == 0):
            print("  Type: degenerate (all eigenvalues zero)")
        elif np.all(eigvals > 0):
            print("  Type: minimum")
        else:
            print("  Type: saddle or other")
    else:
        print("  Not implemented yet")

    print("\nExercise 3: HVP and power iteration")
    lam = exercise_3_hessian_vector_product()
    if lam is not None:
        expected_max = np.max(np.linalg.eigvalsh(np.array([[4, 1], [1, 3]])))
        print(f"  Largest eigenvalue: {lam:.4f}, Expected: {expected_max:.4f}")
    else:
        print("  Not implemented yet")
