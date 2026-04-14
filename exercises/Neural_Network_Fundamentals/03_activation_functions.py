"""
03. Activation Functions - Exercises
======================================
Lesson 03: Activation Functions

Exercises cover:
  1. Implement all activation functions and verify derivatives
  2. Prove sigmoid max derivative is 0.25
  3. Demonstrate the dying ReLU problem
"""

import numpy as np


# ============================================================
# Exercise 1: Implement Activation Functions
# Implement ELU and verify its derivative numerically.
# ============================================================
def exercise_1_elu():
    """Implement ELU and verify its derivative."""
    print("=" * 60)
    print("Exercise 1: ELU Implementation and Verification")
    print("=" * 60)

    def elu(z, alpha=1.0):
        # TODO: Implement ELU
        # ELU(z) = z if z > 0, else alpha * (exp(z) - 1)
        raise NotImplementedError("Implement ELU")

    def elu_derivative(z, alpha=1.0):
        # TODO: Implement ELU derivative
        # ELU'(z) = 1 if z > 0, else alpha * exp(z)
        raise NotImplementedError("Implement ELU derivative")

    # Verify numerically
    z_test = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    eps = 1e-7
    for z_val in z_test:
        z_arr = np.array([z_val])
        numerical = (elu(z_arr + eps) - elu(z_arr - eps)) / (2 * eps)
        analytical = elu_derivative(z_arr)
        error = abs(numerical[0] - analytical[0])
        status = "OK" if error < 1e-5 else "FAIL"
        print(f"  z={z_val:+.1f}: analytical={analytical[0]:.6f}, "
              f"numerical={numerical[0]:.6f}, error={error:.2e} [{status}]")


# ============================================================
# Exercise 2: Sigmoid Max Derivative
# Show that sigma'(z) achieves its maximum of 0.25 at z=0.
# ============================================================
def exercise_2_sigmoid_max():
    """Show sigmoid derivative maximum is 0.25 at z=0."""
    print("\n" + "=" * 60)
    print("Exercise 2: Sigmoid Maximum Derivative")
    print("=" * 60)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    # TODO: Compute sigmoid'(z) for z in [-5, 5]
    # Find the maximum value and the z where it occurs
    # Show that max value = 0.25 and it occurs at z = 0
    raise NotImplementedError("Find sigmoid max derivative")


# ============================================================
# Exercise 3: Dying ReLU
# Show that a neuron with negative bias + ReLU never recovers.
# ============================================================
def exercise_3_dying_relu():
    """Demonstrate the dying ReLU problem."""
    print("\n" + "=" * 60)
    print("Exercise 3: Dying ReLU Problem")
    print("=" * 60)

    # TODO: Create a single ReLU neuron with negative bias (e.g., -5)
    # Feed random inputs and show that:
    # 1. The pre-activation z is always negative
    # 2. The ReLU output is always 0
    # 3. The gradient is always 0
    # 4. The weights never update → the neuron is "dead"
    raise NotImplementedError("Demonstrate dying ReLU")


if __name__ == "__main__":
    exercise_1_elu()
    exercise_2_sigmoid_max()
    exercise_3_dying_relu()
