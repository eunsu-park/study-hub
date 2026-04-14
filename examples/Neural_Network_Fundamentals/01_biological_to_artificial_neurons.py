"""
01. Biological to Artificial Neurons
=====================================
Demonstrates the McCulloch-Pitts neuron model and the evolution
to artificial neurons with learnable weights and activation functions.

Key Concepts:
  - McCulloch-Pitts threshold neuron
  - Boolean logic gates (AND, OR, NAND)
  - Artificial neuron with sigmoid activation
"""

import numpy as np

# ============================================================
# 1. McCulloch-Pitts Neuron
# ============================================================
print("=" * 60)
print("1. McCulloch-Pitts Neuron")
print("=" * 60)


def mcculloch_pitts(inputs, threshold):
    """McCulloch-Pitts neuron: fires if sum of inputs >= threshold."""
    return int(np.sum(inputs) >= threshold)


# AND gate (threshold = 2)
print("\nAND gate (threshold=2):")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    result = mcculloch_pitts([x1, x2], threshold=2)
    print(f"  AND({x1}, {x2}) = {result}")

# OR gate (threshold = 1)
print("\nOR gate (threshold=1):")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    result = mcculloch_pitts([x1, x2], threshold=1)
    print(f"  OR({x1}, {x2}) = {result}")

# NAND gate (threshold = 2, inverted logic)
print("\nNAND gate:")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    # NAND = NOT(AND) — use negative weights and adjusted threshold
    result = int((-x1 + -x2 + 2) >= 1)  # equivalent to sum <= 1
    print(f"  NAND({x1}, {x2}) = {result}")


# ============================================================
# 2. Artificial Neuron with Sigmoid
# ============================================================
print("\n" + "=" * 60)
print("2. Artificial Neuron with Sigmoid Activation")
print("=" * 60)


class ArtificialNeuron:
    """Single artificial neuron with sigmoid activation."""

    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, x):
        """Compute y = sigmoid(w^T x + b)."""
        z = np.dot(self.weights, x) + self.bias
        return self.sigmoid(z)


np.random.seed(42)
neuron = ArtificialNeuron(n_inputs=3)
x = np.array([1.0, 0.5, -1.5])
output = neuron.forward(x)
print(f"\nInput:   {x}")
print(f"Weights: {neuron.weights}")
print(f"Bias:    {neuron.bias}")
print(f"Output:  {output:.6f}")


# ============================================================
# 3. Manual AND Gate with Artificial Neuron
# ============================================================
print("\n" + "=" * 60)
print("3. Manual AND Gate with Artificial Neuron")
print("=" * 60)

# Manually set weights to implement AND
neuron_and = ArtificialNeuron(n_inputs=2)
neuron_and.weights = np.array([20.0, 20.0])  # large positive weights
neuron_and.bias = -30.0  # threshold

print("\nAND gate via artificial neuron (sigmoid):")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    x = np.array([x1, x2], dtype=float)
    out = neuron_and.forward(x)
    print(f"  ({x1}, {x2}) -> {out:.4f} (rounded: {round(out)})")


# ============================================================
# 4. Comparison: Biological vs Artificial
# ============================================================
print("\n" + "=" * 60)
print("4. Biological vs Artificial Neuron Comparison")
print("=" * 60)

comparison = [
    ("Processing", "Electrochemical", "Numerical computation"),
    ("Speed", "~100 Hz", "Billions of ops/sec"),
    ("Learning", "Synaptic plasticity", "Gradient descent"),
    ("Connections", "~10,000 synapses", "Arbitrary (dense layers)"),
    ("Signal", "Spike trains", "Single scalar value"),
    ("Energy", "~20 W (brain)", "~300 W (GPU)"),
]

print(f"\n{'Aspect':<15} {'Biological':<25} {'Artificial':<25}")
print("-" * 65)
for aspect, bio, art in comparison:
    print(f"{aspect:<15} {bio:<25} {art:<25}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
