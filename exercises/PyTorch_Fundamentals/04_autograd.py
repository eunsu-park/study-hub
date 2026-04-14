"""
Autograd - Exercises
====================
Lesson 04: Autograd

Exercises:
  1. Compute gradients of a polynomial
  2. Implement gradient descent for a quadratic
"""

import torch


def exercise_1_polynomial_grad(x_val):
    """Compute the gradient of f(x) = 3x^3 - 2x^2 + x - 5 at x=x_val.

    The analytical derivative is: f'(x) = 9x^2 - 4x + 1

    Args:
        x_val: float, the point at which to evaluate the gradient

    Returns:
        float: the gradient value

    TODO:
      - Create a tensor with requires_grad=True
      - Compute f(x)
      - Call backward()
      - Return the gradient as a float
    """
    # TODO: implement
    raise NotImplementedError


def exercise_2_gradient_descent(f, x_init, lr=0.1, steps=100):
    """Find the minimum of function f using gradient descent.

    Args:
        f: callable that takes a tensor and returns a scalar tensor
        x_init: float, initial value for x
        lr: learning rate
        steps: number of gradient descent steps

    Returns:
        float: the optimized x value

    TODO:
      - Create x as a tensor with requires_grad=True
      - For each step: compute f(x), backward(), update x, zero grad
      - Return the final x value as a float
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    print("Exercise 1: Polynomial Gradient")
    print("-" * 40)
    try:
        grad = exercise_1_polynomial_grad(2.0)
        expected = 9 * 4 - 4 * 2 + 1  # 29
        print(f"f'(2) = {grad:.4f} (expected {expected})")
        assert abs(grad - expected) < 1e-4
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: Gradient Descent")
    print("-" * 40)
    try:
        # Minimize f(x) = (x - 3)^2 + 1. Minimum at x=3.
        def f(x):
            return (x - 3) ** 2 + 1

        x_opt = exercise_2_gradient_descent(f, x_init=0.0, lr=0.1, steps=100)
        print(f"Optimized x = {x_opt:.4f} (expected ~3.0)")
        assert abs(x_opt - 3.0) < 0.1
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
