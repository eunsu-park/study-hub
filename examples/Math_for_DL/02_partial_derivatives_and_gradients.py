"""
Partial Derivatives and Gradients

Demonstrates gradient computation and gradient descent:
- Analytical vs numerical gradient comparison
- Gradient visualization on loss surfaces
- Directional derivatives
- Gradient descent on quadratics with varying condition numbers
- Activation function gradients

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def gradient_comparison():
    """Compare analytical and numerical gradients."""
    print("=" * 60)
    print("ANALYTICAL VS NUMERICAL GRADIENT")
    print("=" * 60)

    def f(x):
        return x[0]**2 + 3*x[1]**2 - 2*x[0]*x[1] + x[0] - 4*x[1] + 5

    def grad_analytical(x):
        return np.array([2*x[0] - 2*x[1] + 1, 6*x[1] - 2*x[0] - 4])

    def grad_numerical(f, x, h=1e-5):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            e = np.zeros_like(x); e[i] = h
            grad[i] = (f(x + e) - f(x - e)) / (2 * h)
        return grad

    test_points = [np.array([0.0, 0.0]), np.array([1.0, 2.0]), np.array([-3.0, 1.5])]
    for pt in test_points:
        ga = grad_analytical(pt)
        gn = grad_numerical(f, pt)
        err = np.max(np.abs(ga - gn))
        print(f"  x = {pt}, analytical = {ga}, numerical = {gn}, error = {err:.2e}")


def gradient_descent_quadratic():
    """Gradient descent on quadratics with different condition numbers."""
    print("\n" + "=" * 60)
    print("GRADIENT DESCENT VS CONDITION NUMBER")
    print("=" * 60)

    np.random.seed(42)
    n = 10

    for kappa in [2, 10, 100]:
        eigvals = np.linspace(1, kappa, n)
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        A = Q @ np.diag(eigvals) @ Q.T
        b = np.random.randn(n)
        x_star = np.linalg.solve(A, b)

        lr = 2.0 / (eigvals[0] + eigvals[-1])
        x = np.zeros(n)
        errors = []

        for t in range(200):
            errors.append(np.linalg.norm(x - x_star))
            grad = A @ x - b
            x = x - lr * grad

        print(f"  kappa={kappa:4d}: lr={lr:.4f}, final_error={errors[-1]:.2e}, "
              f"converged_in~{next((i for i, e in enumerate(errors) if e < 1e-6), 200)} steps")


def activation_gradients():
    """Visualize activation functions and their gradients."""
    print("\n" + "=" * 60)
    print("ACTIVATION FUNCTION GRADIENTS")
    print("=" * 60)

    x = np.linspace(-5, 5, 500)

    activations = {
        'Sigmoid': (1 / (1 + np.exp(-x)), lambda s: s * (1 - s)),
        'ReLU': (np.maximum(0, x), lambda _: (x > 0).astype(float)),
        'Tanh': (np.tanh(x), lambda t: 1 - t**2),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, (name, (val, grad_fn)) in enumerate(activations.items()):
        grad = grad_fn(val)
        axes[0, i].plot(x, val, 'b-', linewidth=2)
        axes[0, i].set_title(f'{name}')
        axes[0, i].grid(True, alpha=0.3)
        axes[1, i].plot(x, grad, 'r-', linewidth=2)
        axes[1, i].set_title(f"{name}' (gradient)")
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle('Activation Functions and Their Gradients')
    plt.tight_layout()
    plt.savefig('activation_gradients.png', dpi=100)
    plt.close()
    print("  Saved activation_gradients.png")


def gradient_norm_monitoring():
    """Simulate gradient norm tracking during training."""
    print("\n" + "=" * 60)
    print("GRADIENT NORM MONITORING")
    print("=" * 60)

    np.random.seed(42)
    A = np.eye(10) * 2
    b = np.random.randn(10)
    x = np.random.randn(10) * 5
    eta = 0.1

    for t in [0, 10, 50, 100, 199]:
        x_t = np.random.randn(10) * 5
        for _ in range(t):
            grad = A @ x_t - b
            x_t = x_t - eta * grad
        grad = A @ x_t - b
        loss = 0.5 * x_t @ A @ x_t - b @ x_t
        print(f"  Step {t:3d}: loss = {loss:10.4f}, ||grad|| = {np.linalg.norm(grad):.4f}")


if __name__ == "__main__":
    gradient_comparison()
    gradient_descent_quadratic()
    activation_gradients()
    gradient_norm_monitoring()
