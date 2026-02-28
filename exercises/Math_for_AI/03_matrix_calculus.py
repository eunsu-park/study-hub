"""
Exercises for Lesson 03: Matrix Calculus
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Derive Matrix Derivative Identity ===
# Problem: Prove d/dx (x^T A x) = (A + A^T)x using index notation.
# Verify with numerical gradients.

def exercise_1():
    """Verify the quadratic form derivative identity."""
    np.random.seed(42)
    n = 5
    A = np.random.randn(n, n)
    x = np.random.randn(n)

    # Analytical gradient: (A + A^T) @ x
    grad_analytical = (A + A.T) @ x

    # Numerical gradient via finite differences
    eps = 1e-5
    grad_numerical = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        f_plus = x_plus @ A @ x_plus
        f_minus = x_minus @ A @ x_minus
        grad_numerical[i] = (f_plus - f_minus) / (2 * eps)

    print("Derivation using index notation:")
    print("  f(x) = x^T A x = sum_{i,j} x_i A_{ij} x_j")
    print("  df/dx_k = sum_j A_{kj} x_j + sum_i x_i A_{ik}")
    print("          = (Ax)_k + (A^T x)_k")
    print("          = ((A + A^T)x)_k")
    print("  Therefore: nabla f = (A + A^T)x")
    print()
    print(f"Analytical gradient:  {np.round(grad_analytical, 6)}")
    print(f"Numerical gradient:   {np.round(grad_numerical, 6)}")
    print(f"Max difference: {np.max(np.abs(grad_analytical - grad_numerical)):.2e}")


# === Exercise 2: Logistic Regression Gradient ===
# Problem: Derive and verify gradient of logistic regression loss.
# nabla_w L = (1/n) X^T (sigma - y)

def exercise_2():
    """Logistic regression gradient derivation and verification."""
    np.random.seed(42)
    n, d = 100, 5
    X = np.random.randn(n, d)
    w = np.random.randn(d)
    y = (np.random.rand(n) > 0.5).astype(float)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def log_loss(w):
        z = X @ w
        p = sigmoid(z)
        # Clip for numerical stability
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    # Analytical gradient: (1/n) X^T (sigma - y)
    z = X @ w
    sigma = sigmoid(z)
    grad_analytical = X.T @ (sigma - y) / n

    # Numerical gradient
    eps = 1e-5
    grad_numerical = np.zeros(d)
    for i in range(d):
        w_plus = w.copy()
        w_plus[i] += eps
        w_minus = w.copy()
        w_minus[i] -= eps
        grad_numerical[i] = (log_loss(w_plus) - log_loss(w_minus)) / (2 * eps)

    print("Logistic regression gradient derivation:")
    print("  L(w) = -(1/n) sum [y_i log sigma(w^T x_i) + (1-y_i) log(1-sigma(w^T x_i))]")
    print("  Since d/dz sigma(z) = sigma(z)(1-sigma(z)):")
    print("  dL/dw = (1/n) sum (sigma(w^T x_i) - y_i) x_i")
    print("        = (1/n) X^T (sigma - y)")
    print()
    print(f"Analytical gradient:  {np.round(grad_analytical, 6)}")
    print(f"Numerical gradient:   {np.round(grad_numerical, 6)}")
    print(f"Max difference: {np.max(np.abs(grad_analytical - grad_numerical)):.2e}")


# === Exercise 3: Batch Normalization Gradient ===
# Problem: Derive gradient of batch normalization and verify numerically.

def exercise_3():
    """Batch normalization forward and backward pass."""
    np.random.seed(42)
    n = 10
    x = np.random.randn(n) * 3 + 5  # input
    eps = 1e-5

    # Forward pass: batch norm
    mu = np.mean(x)
    var = np.var(x)
    x_hat = (x - mu) / np.sqrt(var + eps)

    print(f"Input x: {np.round(x, 4)}")
    print(f"Mean: {mu:.4f}, Variance: {var:.4f}")
    print(f"Normalized x_hat: {np.round(x_hat, 4)}")

    # Backward pass: dL/dx_i given dL/dx_hat = ones (simplified)
    # Using upstream gradient as all ones
    dL_dxhat = np.ones(n)

    # Analytical gradient of batch norm (from derivation)
    inv_std = 1.0 / np.sqrt(var + eps)

    # dL/dx_i = (1/sqrt(var+eps)) * (dL/dxhat_i
    #            - (1/n)*sum(dL/dxhat)
    #            - (1/n)*xhat_i * sum(dL/dxhat * xhat))
    sum_dxhat = np.sum(dL_dxhat)
    sum_dxhat_xhat = np.sum(dL_dxhat * x_hat)
    dL_dx_analytical = inv_std * (dL_dxhat - sum_dxhat / n - x_hat * sum_dxhat_xhat / n)

    # Numerical gradient
    def bn_forward_loss(x_input):
        mu_loc = np.mean(x_input)
        var_loc = np.var(x_input)
        x_hat_loc = (x_input - mu_loc) / np.sqrt(var_loc + eps)
        return np.sum(x_hat_loc)  # sum is our "loss" since upstream grad is ones

    eps_num = 1e-5
    dL_dx_numerical = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps_num
        x_minus = x.copy()
        x_minus[i] -= eps_num
        dL_dx_numerical[i] = (bn_forward_loss(x_plus) - bn_forward_loss(x_minus)) / (2 * eps_num)

    print(f"\nAnalytical dL/dx: {np.round(dL_dx_analytical, 6)}")
    print(f"Numerical  dL/dx: {np.round(dL_dx_numerical, 6)}")
    print(f"Max difference: {np.max(np.abs(dL_dx_analytical - dL_dx_numerical)):.2e}")


# === Exercise 4: Softmax Jacobian ===
# Problem: Compute the Jacobian of softmax: d sigma_i / d z_j = sigma_i (delta_ij - sigma_j)

def exercise_4():
    """Compute and verify the softmax Jacobian."""
    z = np.array([2.0, 1.0, 0.1, -1.0, 3.0])
    n = len(z)

    # Softmax
    exp_z = np.exp(z - np.max(z))  # numerical stability
    sigma = exp_z / np.sum(exp_z)

    print(f"Input z: {z}")
    print(f"Softmax sigma: {np.round(sigma, 6)}")
    print(f"Sum of sigma: {np.sum(sigma):.10f}")

    # Analytical Jacobian: J_ij = sigma_i * (delta_ij - sigma_j)
    J_analytical = np.diag(sigma) - np.outer(sigma, sigma)
    print(f"\nAnalytical Jacobian (5x5):\n{np.round(J_analytical, 6)}")

    # Verify row sums are zero (since sum of softmax = 1, derivative of constant = 0)
    row_sums = J_analytical.sum(axis=1)
    print(f"\nRow sums (should be ~0): {np.round(row_sums, 10)}")

    # Numerical Jacobian
    eps = 1e-5
    J_numerical = np.zeros((n, n))
    for j in range(n):
        z_plus = z.copy()
        z_plus[j] += eps
        z_minus = z.copy()
        z_minus[j] -= eps

        exp_plus = np.exp(z_plus - np.max(z_plus))
        sigma_plus = exp_plus / np.sum(exp_plus)
        exp_minus = np.exp(z_minus - np.max(z_minus))
        sigma_minus = exp_minus / np.sum(exp_minus)

        J_numerical[:, j] = (sigma_plus - sigma_minus) / (2 * eps)

    print(f"\nNumerical Jacobian:\n{np.round(J_numerical, 6)}")
    print(f"Max difference: {np.max(np.abs(J_analytical - J_numerical)):.2e}")


# === Exercise 5: L2 Regularization Gradient ===
# Problem: Derive gradient of ridge regression loss:
# L(w) = (1/2n)||y - Xw||^2 + (lambda/2)||w||^2
# Compare gradient descent and analytical solution.

def exercise_5():
    """Ridge regression: gradient descent vs analytical solution."""
    np.random.seed(42)
    n, d = 100, 5
    X = np.random.randn(n, d)
    true_w = np.array([2, -1, 0.5, 3, -0.5])
    y = X @ true_w + np.random.randn(n) * 0.5
    lam = 1.0  # regularization strength

    # Analytical solution: w = (X^T X + n*lambda*I)^{-1} X^T y
    w_analytical = np.linalg.solve(X.T @ X + n * lam * np.eye(d), X.T @ y)
    print(f"Analytical solution: {np.round(w_analytical, 4)}")

    # Gradient: nabla L = -(1/n) X^T (y - Xw) + lambda * w
    def ridge_loss(w):
        return 0.5 * np.mean((y - X @ w) ** 2) + 0.5 * lam * np.dot(w, w)

    def ridge_gradient(w):
        return -X.T @ (y - X @ w) / n + lam * w

    # Gradient descent
    w_gd = np.zeros(d)
    lr = 0.01
    n_steps = 1000
    losses = []

    for step in range(n_steps):
        grad = ridge_gradient(w_gd)
        w_gd = w_gd - lr * grad
        if step % 200 == 0:
            loss = ridge_loss(w_gd)
            losses.append(loss)
            print(f"  Step {step:4d}: loss = {loss:.6f}")

    print(f"\nGradient descent solution: {np.round(w_gd, 4)}")
    print(f"Analytical solution:       {np.round(w_analytical, 4)}")
    print(f"Max difference: {np.max(np.abs(w_gd - w_analytical)):.6f}")

    # Verify gradient at analytical solution is near zero
    grad_at_optimal = ridge_gradient(w_analytical)
    print(f"\nGradient at analytical optimum (should be ~0): {np.round(grad_at_optimal, 8)}")
    print(f"Gradient norm: {np.linalg.norm(grad_at_optimal):.2e}")


if __name__ == "__main__":
    print("=== Exercise 1: Matrix Derivative Identity ===")
    exercise_1()
    print("\n=== Exercise 2: Logistic Regression Gradient ===")
    exercise_2()
    print("\n=== Exercise 3: Batch Normalization Gradient ===")
    exercise_3()
    print("\n=== Exercise 4: Softmax Jacobian ===")
    exercise_4()
    print("\n=== Exercise 5: L2 Regularization Gradient ===")
    exercise_5()
    print("\nAll exercises completed!")
