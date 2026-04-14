"""
02. Perceptron and Linear Classifiers - Exercises
===================================================
Lesson 02: Perceptron and Linear Classifiers

Exercises cover:
  1. Perceptron on OR gate with convergence tracking
  2. Verify perceptron fails on XOR
  3. Two-layer XOR solution
"""

import numpy as np


# ============================================================
# Exercise 1: Perceptron on OR Gate
# Train a perceptron on OR gate data and report convergence.
# ============================================================
def exercise_1_or_perceptron():
    """Train perceptron on OR gate."""
    print("=" * 60)
    print("Exercise 1: Perceptron on OR Gate")
    print("=" * 60)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 1])

    # TODO: Implement perceptron training
    # Initialize weights to zeros, learning_rate = 1.0
    # Train until convergence or max 100 epochs
    # Return the epoch at which it converges
    weights = np.zeros(2)
    bias = 0.0
    lr = 1.0

    # TODO: Training loop
    raise NotImplementedError("Implement perceptron training on OR gate")


# ============================================================
# Exercise 2: XOR Failure
# Show that the perceptron fails on XOR after 1000 epochs.
# ============================================================
def exercise_2_xor_failure():
    """Demonstrate perceptron fails on XOR."""
    print("\n" + "=" * 60)
    print("Exercise 2: XOR Failure")
    print("=" * 60)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])

    # TODO: Train perceptron for 1000 epochs on XOR
    # Track errors per epoch and show it never reaches 0
    raise NotImplementedError("Demonstrate XOR failure")


# ============================================================
# Exercise 3: Two-Layer XOR
# Build a 2-layer network to solve XOR with manually chosen weights.
# ============================================================
def exercise_3_two_layer_xor():
    """Solve XOR with a two-layer network."""
    print("\n" + "=" * 60)
    print("Exercise 3: Two-Layer XOR Solution")
    print("=" * 60)

    def step(z):
        return 1 if z >= 0 else 0

    def xor_network(x1, x2):
        # TODO: Implement XOR using two layers
        # Hidden layer: 2 neurons (e.g., OR and NAND)
        # Output layer: 1 neuron (AND of hidden outputs)
        # Choose weights and biases manually
        raise NotImplementedError("Implement 2-layer XOR")

    expected = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    for (x1, x2), y in expected.items():
        result = xor_network(x1, x2)
        status = "OK" if result == y else "FAIL"
        print(f"  XOR({x1}, {x2}) = {result} (expected {y}) [{status}]")


if __name__ == "__main__":
    exercise_1_or_perceptron()
    exercise_2_xor_failure()
    exercise_3_two_layer_xor()
