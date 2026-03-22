"""
Exercises for Lesson 02: Neural Network Basics
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# === Exercise 1: Perceptron from Scratch ===
# Problem: Implement a single perceptron in NumPy with sigmoid activation.
# Compute the output for x = [1.0, 2.0, 3.0], w = [0.5, -0.3, 0.8], b = 0.1.

def exercise_1():
    """Single perceptron forward pass in NumPy."""
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([0.5, -0.3, 0.8])
    b = 0.1

    z = np.dot(x, w) + b   # Linear combination: 0.5 - 0.6 + 2.4 + 0.1 = 2.4
    a = 1 / (1 + np.exp(-z))  # Sigmoid activation

    print("  z (pre-activation) = {:.4f}".format(z))
    print("  a (sigmoid output) = {:.6f}".format(a))
    print("  Expected z ≈ 2.4, a ≈ 0.9168")


# === Exercise 2: Activation Functions ===
# Problem: Implement Sigmoid, Tanh, ReLU, Leaky ReLU, and GELU in NumPy.
# Evaluate each at x = [-2.0, -1.0, 0.0, 1.0, 2.0] and compare outputs.

def exercise_2():
    """Implement and compare activation functions."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    leaky_relu = np.where(x >= 0, x, 0.01 * x)
    # GELU approximation: x * Phi(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    gelu = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    print("  x:          {}".format(x))
    print("  sigmoid:    {}".format(np.round(sigmoid, 4)))
    print("  tanh:       {}".format(np.round(tanh, 4)))
    print("  relu:       {}".format(np.round(relu, 4)))
    print("  leaky_relu: {}".format(np.round(leaky_relu, 4)))
    print("  gelu:       {}".format(np.round(gelu, 4)))

    # Verify with PyTorch
    x_t = torch.tensor(x, dtype=torch.float32)
    pt_gelu = F.gelu(x_t).numpy()
    print("  PyTorch GELU match: {}".format(np.allclose(gelu, pt_gelu, atol=1e-4)))


# === Exercise 3: 2-Layer MLP Forward Pass (NumPy) ===
# Problem: Implement a 2-layer MLP forward pass manually in NumPy.
# Architecture: Input(4) -> Hidden(8, ReLU) -> Output(3, Softmax).

def exercise_3():
    """2-layer MLP forward pass in NumPy."""
    np.random.seed(42)
    input_dim, hidden_dim, output_dim = 4, 8, 3

    # Random weight initialization (small values)
    W1 = np.random.randn(input_dim, hidden_dim) * 0.1
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    b2 = np.zeros(output_dim)

    # Single sample input
    x = np.array([0.5, -1.2, 0.8, 0.3])

    # Forward pass
    z1 = x @ W1 + b1              # (hidden_dim,)
    a1 = np.maximum(0, z1)        # ReLU
    z2 = a1 @ W2 + b2             # (output_dim,)
    exp_z2 = np.exp(z2 - z2.max())  # numerically stable softmax
    a2 = exp_z2 / exp_z2.sum()   # Softmax

    print("  z1 (first 3): {}".format(np.round(z1[:3], 4)))
    print("  a1 (first 3, ReLU): {}".format(np.round(a1[:3], 4)))
    print("  z2: {}".format(np.round(z2, 4)))
    print("  a2 (softmax): {}".format(np.round(a2, 4)))
    print("  Softmax sums to 1: {}".format(np.allclose(a2.sum(), 1.0)))


# === Exercise 4: nn.Module — XOR Problem ===
# Problem: Build and train an MLP with PyTorch nn.Module to solve the XOR
# classification problem. A single linear layer cannot solve XOR; a hidden
# layer is required.

def exercise_4():
    """Solve XOR problem with a 2-layer MLP using PyTorch."""
    torch.manual_seed(42)

    # XOR truth table
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    # MLP: Input(2) -> Hidden(4, ReLU) -> Output(1, Sigmoid)
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(2000):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    with torch.no_grad():
        pred = model(X)
        pred_labels = (pred > 0.5).float()

    print("  XOR predictions:")
    for i in range(4):
        print("    x={}, y_true={}, y_pred={:.4f}, label={}".format(
            X[i].tolist(), int(y[i].item()), pred[i].item(), int(pred_labels[i].item())))
    accuracy = (pred_labels == y).float().mean().item()
    print("  Accuracy: {:.1%}".format(accuracy))


# === Exercise 5: Weight Initialization Comparison ===
# Problem: Build the same 3-layer MLP with different weight initializations
# (zeros, Xavier, He/Kaiming) and compare gradient norms after one backward pass.

def exercise_5():
    """Compare weight initialization strategies via gradient norms."""
    torch.manual_seed(0)

    def build_mlp_with_init(init_fn):
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        for m in model.modules():
            if isinstance(m, nn.Linear):
                init_fn(m)
        return model

    def zero_init(m):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)

    def xavier_init(m):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

    def he_init(m):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

    x = torch.randn(16, 32)
    y = torch.randint(0, 10, (16,))
    criterion = nn.CrossEntropyLoss()

    for name, init_fn in [("Zeros", zero_init), ("Xavier", xavier_init), ("He/Kaiming", he_init)]:
        model = build_mlp_with_init(init_fn)
        loss = criterion(model(x), y)
        loss.backward()
        total_grad_norm = sum(
            p.grad.norm().item() for p in model.parameters() if p.grad is not None
        )
        print("  {}: total gradient norm = {:.6f}".format(name, total_grad_norm))


if __name__ == "__main__":
    print("=== Exercise 1: Single Perceptron Forward Pass ===")
    exercise_1()
    print("\n=== Exercise 2: Activation Functions Comparison ===")
    exercise_2()
    print("\n=== Exercise 3: 2-Layer MLP Forward Pass (NumPy) ===")
    exercise_3()
    print("\n=== Exercise 4: XOR Problem with nn.Module ===")
    exercise_4()
    print("\n=== Exercise 5: Weight Initialization Comparison ===")
    exercise_5()
    print("\nAll exercises completed!")
