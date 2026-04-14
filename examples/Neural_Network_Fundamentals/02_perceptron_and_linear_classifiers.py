"""
02. Perceptron and Linear Classifiers
======================================
Implements the perceptron learning algorithm, demonstrates convergence
on linearly separable data, and shows failure on XOR.

Key Concepts:
  - Perceptron learning rule
  - Decision boundary visualization
  - XOR problem and multi-layer solution
  - Adaline (adaptive linear neuron)
"""

import numpy as np


# ============================================================
# 1. Perceptron Implementation
# ============================================================
print("=" * 60)
print("1. Perceptron Learning Algorithm")
print("=" * 60)


class Perceptron:
    """Single-layer perceptron classifier."""

    def __init__(self, n_features, learning_rate=1.0):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        return 1 if z >= 0 else 0

    def train(self, X, y, n_epochs=100):
        """Train using the perceptron learning rule."""
        errors_per_epoch = []
        for epoch in range(n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                error = yi - pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1
            errors_per_epoch.append(errors)
            if errors == 0:
                print(f"  Converged at epoch {epoch + 1}")
                break
        return errors_per_epoch


# AND gate
print("\n--- AND Gate ---")
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_and = np.array([0, 0, 0, 1])

p_and = Perceptron(n_features=2)
p_and.train(X_and, y_and)
print(f"  Weights: {p_and.weights}, Bias: {p_and.bias:.1f}")
for xi, yi in zip(X_and, y_and):
    print(f"  {xi} -> {p_and.predict(xi)} (expected {yi})")

# OR gate
print("\n--- OR Gate ---")
y_or = np.array([0, 1, 1, 1])
p_or = Perceptron(n_features=2)
p_or.train(X_and, y_or)
print(f"  Weights: {p_or.weights}, Bias: {p_or.bias:.1f}")
for xi, yi in zip(X_and, y_or):
    print(f"  {xi} -> {p_or.predict(xi)} (expected {yi})")


# ============================================================
# 2. XOR Failure
# ============================================================
print("\n" + "=" * 60)
print("2. XOR — Perceptron Fails")
print("=" * 60)

y_xor = np.array([0, 1, 1, 0])
p_xor = Perceptron(n_features=2)
errors = p_xor.train(X_and, y_xor, n_epochs=1000)
print(f"  After 1000 epochs, errors in last epoch: {errors[-1]}")
print("  Perceptron CANNOT learn XOR (not linearly separable)")


# ============================================================
# 3. Multi-Layer XOR Solution
# ============================================================
print("\n" + "=" * 60)
print("3. Multi-Layer Network Solves XOR")
print("=" * 60)


def xor_network(x1, x2):
    """XOR = AND(OR(x1,x2), NAND(x1,x2))."""
    h1 = 1 if (x1 + x2 - 0.5) >= 0 else 0      # OR
    h2 = 1 if (-x1 - x2 + 1.5) >= 0 else 0      # NAND
    y = 1 if (h1 + h2 - 1.5) >= 0 else 0         # AND
    return y


print("\nXOR via 2-layer network:")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"  XOR({x1}, {x2}) = {xor_network(x1, x2)}")


# ============================================================
# 4. Adaline (Adaptive Linear Neuron)
# ============================================================
print("\n" + "=" * 60)
print("4. Adaline — Gradient-Based Learning")
print("=" * 60)


class Adaline:
    """Adaptive Linear Neuron (Widrow-Hoff)."""

    def __init__(self, n_features, learning_rate=0.01):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.lr = learning_rate

    def net_input(self, x):
        return np.dot(self.weights, x) + self.bias

    def predict(self, x):
        return 1 if self.net_input(x) >= 0.5 else 0

    def train(self, X, y, n_epochs=100):
        losses = []
        for epoch in range(n_epochs):
            total_loss = 0.0
            for xi, yi in zip(X, y):
                z = self.net_input(xi)
                error = yi - z
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                total_loss += 0.5 * error ** 2
            losses.append(total_loss / len(y))
        return losses


np.random.seed(42)
adaline = Adaline(n_features=2, learning_rate=0.1)
losses = adaline.train(X_and, y_and, n_epochs=50)

print(f"\nAdaline on AND gate (50 epochs):")
print(f"  Final loss: {losses[-1]:.6f}")
print(f"  Weights: {adaline.weights}, Bias: {adaline.bias:.4f}")
for xi, yi in zip(X_and, y_and):
    print(f"  {xi} -> {adaline.predict(xi)} (expected {yi})")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
