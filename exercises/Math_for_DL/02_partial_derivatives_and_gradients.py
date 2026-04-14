"""
Exercises for Lesson 02: Partial Derivatives and Gradients
Topic: Math_for_DL

Complete the TODO sections. Run with: python 02_partial_derivatives_and_gradients.py
"""

import numpy as np


def exercise_1_multivariable_gradient():
    """Compute gradient of f(x1,x2,x3) = x1*x2 + x2*x3^2 - ln(x1).

    Return the analytical gradient at point (1, 2, 3).
    """
    x = np.array([1.0, 2.0, 3.0])

    # TODO: Compute the gradient analytically
    # df/dx1 = x2 - 1/x1
    # df/dx2 = x1 + x3^2
    # df/dx3 = 2*x2*x3
    grad = None  # Replace with np.array([...])

    return grad


def exercise_2_rosenbrock_gd():
    """Minimize the Rosenbrock function using gradient descent.

    f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
    Start from (-1, 1), use lr=0.001, run 5000 steps.
    Return the final (x, y) and loss.
    """
    # TODO: Implement gradient descent on Rosenbrock
    x = np.array([-1.0, 1.0])
    lr = 0.001
    n_steps = 5000

    for _ in range(n_steps):
        # TODO: Compute gradient of Rosenbrock
        # df/dx = -2*(1-x) - 400*x*(y - x^2)
        # df/dy = 200*(y - x^2)
        grad = None  # Replace
        x = x - lr * grad

    loss = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    return x, loss


def exercise_3_gradient_checker():
    """Implement a gradient checker using relative error.

    Compare analytical and numerical gradients using:
    relative_error = |analytical - numerical| / max(|analytical|, |numerical|, eps)
    """
    def f(x):
        return np.sum(x**3 + np.sin(x))

    def grad_analytical(x):
        # TODO: Compute analytical gradient of f
        return None  # Replace

    def grad_numerical(f, x, h=1e-5):
        # TODO: Compute central-difference numerical gradient
        grad = np.zeros_like(x)
        return grad  # Replace

    x = np.array([1.0, -2.0, 0.5])
    ga = grad_analytical(x)
    gn = grad_numerical(f, x)

    if ga is not None and gn is not None:
        eps = 1e-7
        rel_err = np.abs(ga - gn) / np.maximum(np.maximum(np.abs(ga), np.abs(gn)), eps)
        return np.max(rel_err)
    return None


if __name__ == "__main__":
    print("Exercise 1: Multivariable gradient")
    grad = exercise_1_multivariable_gradient()
    if grad is not None:
        expected = np.array([2.0 - 1.0, 1.0 + 9.0, 12.0])
        print(f"  Gradient: {grad}, Expected: {expected}, Pass: {np.allclose(grad, expected)}")
    else:
        print("  Not implemented yet")

    print("\nExercise 2: Rosenbrock GD")
    result = exercise_2_rosenbrock_gd()
    if result[1] is not None:
        x_final, loss = result
        print(f"  Final x: {x_final.round(4)}, loss: {loss:.6f}")
        print(f"  Near optimum (1,1): {np.allclose(x_final, [1, 1], atol=0.1)}")
    else:
        print("  Not implemented yet")

    print("\nExercise 3: Gradient checker")
    err = exercise_3_gradient_checker()
    if err is not None:
        print(f"  Max relative error: {err:.2e}, Pass: {err < 1e-5}")
    else:
        print("  Not implemented yet")
