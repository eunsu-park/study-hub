"""
01. Biological to Artificial Neurons - Exercises
==================================================
Lesson 01: Biological to Artificial Neurons

Exercises cover:
  1. NAND gate with McCulloch-Pitts neuron
  2. Artificial neuron implementing AND gate
"""

import numpy as np


# ============================================================
# Exercise 1: NAND Gate with McCulloch-Pitts
# Implement a McCulloch-Pitts neuron that computes NAND.
# NAND(x1, x2) = NOT(AND(x1, x2))
# ============================================================
def exercise_1_nand_gate():
    """Implement NAND using McCulloch-Pitts neuron with inhibitory inputs."""
    print("=" * 60)
    print("Exercise 1: NAND Gate")
    print("=" * 60)

    def mcculloch_pitts_nand(x1, x2):
        # TODO: Implement NAND gate using threshold logic
        # Hint: NAND outputs 1 for all inputs EXCEPT (1,1)
        # Consider using negative weights or adjusted threshold
        raise NotImplementedError("Implement NAND gate")

    expected = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    for (x1, x2), y in expected.items():
        result = mcculloch_pitts_nand(x1, x2)
        status = "OK" if result == y else "FAIL"
        print(f"  NAND({x1}, {x2}) = {result} (expected {y}) [{status}]")


# ============================================================
# Exercise 2: Artificial Neuron AND Gate
# Manually set weights and bias of an artificial neuron
# to implement the AND gate.
# ============================================================
def exercise_2_and_neuron():
    """Create an artificial neuron that implements AND."""
    print("\n" + "=" * 60)
    print("Exercise 2: AND Gate with Artificial Neuron")
    print("=" * 60)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    # TODO: Set weights and bias so that the neuron computes AND
    # The output should be > 0.5 only when both inputs are 1
    weights = np.array([0.0, 0.0])  # TODO: Set appropriate values
    bias = 0.0                       # TODO: Set appropriate value

    print(f"  Weights: {weights}, Bias: {bias}")
    expected = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}
    for (x1, x2), y in expected.items():
        x = np.array([x1, x2], dtype=float)
        z = np.dot(weights, x) + bias
        out = sigmoid(z)
        pred = 1 if out > 0.5 else 0
        status = "OK" if pred == y else "FAIL"
        print(f"  AND({x1}, {x2}) = {out:.4f} -> {pred} (expected {y}) [{status}]")


if __name__ == "__main__":
    exercise_1_nand_gate()
    exercise_2_and_neuron()
