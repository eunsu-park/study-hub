"""
Exercises for Lesson 03: Backpropagation
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn


# === Exercise 1: Chain Rule by Hand ===
# Problem: Given x -> z = 3x + 1 -> a = relu(z) -> L = a^2,
# compute dL/dx analytically and verify numerically.

def exercise_1():
    """Chain rule computation and numerical verification."""
    x_val = 2.0

    # Analytical derivatives:
    # dL/da = 2a
    # da/dz = 1 if z > 0 else 0  (ReLU derivative)
    # dz/dx = 3
    # dL/dx = dL/da * da/dz * dz/dx

    z = 3 * x_val + 1  # z = 7
    a = max(0, z)       # a = 7 (ReLU)
    L = a ** 2          # L = 49

    dL_da = 2 * a       # 14
    da_dz = 1.0 if z > 0 else 0.0  # 1.0
    dz_dx = 3.0

    dL_dx_analytical = dL_da * da_dz * dz_dx  # 14 * 1 * 3 = 42
    print(f"  Forward: z={z}, a={a}, L={L}")
    print(f"  dL/da={dL_da}, da/dz={da_dz}, dz/dx={dz_dx}")
    print(f"  Analytical dL/dx = {dL_dx_analytical}")

    # Numerical verification with central differences
    h = 1e-5

    def f(x):
        z = 3 * x + 1
        a = max(0, z)
        return a ** 2

    dL_dx_numerical = (f(x_val + h) - f(x_val - h)) / (2 * h)
    print(f"  Numerical dL/dx  = {dL_dx_numerical:.6f}")
    print(f"  Match: {abs(dL_dx_analytical - dL_dx_numerical) < 1e-4}")


# === Exercise 2: Single Neuron Backprop ===
# Problem: Implement forward and backward passes for a single neuron
# with sigmoid activation and MSE loss.

def exercise_2():
    """Single neuron forward/backward pass with sigmoid and MSE."""
    w, b, x, y_true = 0.5, 0.1, 1.0, 0.8

    # Forward pass
    z = w * x + b            # 0.5 * 1.0 + 0.1 = 0.6
    a = 1 / (1 + np.exp(-z))  # sigmoid(0.6)
    L = (a - y_true) ** 2     # MSE loss

    print(f"  Forward: z={z:.4f}, a={a:.4f}, L={L:.6f}")

    # Backward pass (hand-computed)
    dL_da = 2 * (a - y_true)
    da_dz = a * (1 - a)       # sigmoid derivative
    dz_dw = x
    dz_db = 1.0

    dL_dw = dL_da * da_dz * dz_dw
    dL_db = dL_da * da_dz * dz_db

    print(f"  dL/da={dL_da:.6f}, da/dz={da_dz:.6f}")
    print(f"  Analytical dL/dw={dL_dw:.6f}, dL/db={dL_db:.6f}")

    # Verify with PyTorch autograd
    w_t = torch.tensor(w, requires_grad=True)
    b_t = torch.tensor(b, requires_grad=True)
    x_t = torch.tensor(x)
    y_t = torch.tensor(y_true)

    z_t = w_t * x_t + b_t
    a_t = torch.sigmoid(z_t)
    L_t = (a_t - y_t) ** 2
    L_t.backward()

    print(f"  PyTorch dL/dw={w_t.grad.item():.6f}, dL/db={b_t.grad.item():.6f}")
    print(f"  w match: {abs(dL_dw - w_t.grad.item()) < 1e-6}")
    print(f"  b match: {abs(dL_db - b_t.grad.item()) < 1e-6}")


# === Exercise 3: Gradient Check for a 2-Layer MLP ===
# Problem: Verify backprop correctness using numerical gradient checking.

def exercise_3():
    """Gradient check for a 2-layer MLP in NumPy."""
    np.random.seed(42)
    input_dim, hidden_dim, output_dim = 4, 8, 2

    # Initialize weights
    W1 = np.random.randn(input_dim, hidden_dim) * 0.1
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    b2 = np.zeros(output_dim)

    x = np.random.randn(1, input_dim)
    y = np.array([[1.0, 0.0]])

    def forward(W1, b1, W2, b2, x, y):
        z1 = x @ W1 + b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ W2 + b2
        loss = 0.5 * np.sum((z2 - y) ** 2)
        return loss, z1, a1, z2

    def backward(W1, b1, W2, b2, x, y):
        loss, z1, a1, z2 = forward(W1, b1, W2, b2, x, y)
        dz2 = z2 - y
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0).astype(float)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        return dW1, db1, dW2, db2

    # Analytical gradients
    dW1_a, db1_a, dW2_a, db2_a = backward(W1, b1, W2, b2, x, y)

    # Numerical gradients for W1
    h = 1e-5

    def numerical_gradient(param, param_name):
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]
            param[idx] = old_val + h
            loss_plus = forward(W1, b1, W2, b2, x, y)[0]
            param[idx] = old_val - h
            loss_minus = forward(W1, b1, W2, b2, x, y)[0]
            param[idx] = old_val
            grad[idx] = (loss_plus - loss_minus) / (2 * h)
            it.iternext()
        return grad

    dW1_n = numerical_gradient(W1, "W1")
    dW2_n = numerical_gradient(W2, "W2")

    # Relative error
    def rel_error(a, n):
        return np.linalg.norm(a - n) / (np.linalg.norm(a) + np.linalg.norm(n) + 1e-8)

    err_W1 = rel_error(dW1_a, dW1_n)
    err_W2 = rel_error(dW2_a, dW2_n)

    print(f"  Relative error W1: {err_W1:.2e} (should be < 1e-5)")
    print(f"  Relative error W2: {err_W2:.2e} (should be < 1e-5)")
    print(f"  Gradient check passed: {err_W1 < 1e-5 and err_W2 < 1e-5}")


# === Exercise 4: Vanishing Gradients with Sigmoid vs ReLU ===
# Problem: Build deep MLPs and compare gradient norms across layers.

def exercise_4():
    """Compare gradient norms: sigmoid vs ReLU in a 10-layer MLP."""
    torch.manual_seed(42)

    def build_deep_mlp(activation):
        layers = []
        for i in range(10):
            linear = nn.Linear(32, 32)
            nn.init.normal_(linear.weight, std=1.0)
            layers.append(linear)
            layers.append(activation())
        layers.append(nn.Linear(32, 1))
        return nn.Sequential(*layers)

    for act_name, act_cls in [("Sigmoid", nn.Sigmoid), ("ReLU", nn.ReLU)]:
        model = build_deep_mlp(act_cls)
        x = torch.randn(4, 32)
        y = torch.randn(4, 1)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()

        print(f"  {act_name} gradient norms per layer:")
        for i, layer in enumerate(model):
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                norm = layer.weight.grad.norm().item()
                print(f"    Layer {i}: {norm:.6f}")
        print()


# === Exercise 5: Computational Graph and retain_graph ===
# Problem: Explore PyTorch's dynamic computational graph behavior.

def exercise_5():
    """Demonstrate retain_graph and gradient accumulation."""
    # Part 1: Basic backward
    x = torch.tensor(3.0, requires_grad=True)
    y = torch.tensor(4.0, requires_grad=True)
    z = x ** 2 + y ** 2
    z.backward()
    print(f"  After first backward: x.grad={x.grad.item()}, y.grad={y.grad.item()}")
    # x.grad = 2*x = 6.0, y.grad = 2*y = 8.0

    # Part 2: Calling backward again raises an error
    print("  Calling z.backward() again would raise RuntimeError (graph freed)")

    # Part 3: With retain_graph=True, gradients accumulate
    x2 = torch.tensor(3.0, requires_grad=True)
    y2 = torch.tensor(4.0, requires_grad=True)
    z2 = x2 ** 2 + y2 ** 2

    z2.backward(retain_graph=True)
    print(f"  After 1st backward (retain): x.grad={x2.grad.item()}, y.grad={y2.grad.item()}")

    z2.backward(retain_graph=True)
    print(f"  After 2nd backward (retain): x.grad={x2.grad.item()}, y.grad={y2.grad.item()}")
    # Gradients accumulate: x.grad = 12.0, y.grad = 16.0

    print("  Gradients accumulate! This is why optimizer.zero_grad() is needed.")


if __name__ == "__main__":
    print("=== Exercise 1: Chain Rule by Hand ===")
    exercise_1()
    print("\n=== Exercise 2: Single Neuron Backprop ===")
    exercise_2()
    print("\n=== Exercise 3: Gradient Check for 2-Layer MLP ===")
    exercise_3()
    print("\n=== Exercise 4: Vanishing Gradients (Sigmoid vs ReLU) ===")
    exercise_4()
    print("\n=== Exercise 5: Computational Graph and retain_graph ===")
    exercise_5()
    print("\nAll exercises completed!")
