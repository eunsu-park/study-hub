"""
08. Weight Initialization - Exercises
=======================================
Lesson 08: Weight Initialization

Exercises cover:
  1. Visualize activation distributions with different inits
  2. Derive init variance for Leaky ReLU
"""

import numpy as np


# ============================================================
# Exercise 1: 20-Layer Activation Distributions
# Build a 20-layer network and track activation std with
# random, Xavier, and He initialization.
# ============================================================
def exercise_1_deep_activations():
    """Track activations through 20 layers with different inits."""
    print("=" * 60)
    print("Exercise 1: Activation Statistics (20 Layers)")
    print("=" * 60)

    # TODO: For each initialization (random N(0,1), Xavier, He):
    # 1. Start with random input x of shape (256, 1)
    # 2. For 20 layers, apply W @ a (256x256) + ReLU
    # 3. Record std of activation at each layer
    # 4. Print results and show which init keeps std stable
    raise NotImplementedError("Track activations through 20 layers")


# ============================================================
# Exercise 2: Leaky ReLU Initialization
# Derive the initialization variance for Leaky ReLU with alpha=0.2.
# ============================================================
def exercise_2_leaky_relu_init():
    """Derive and verify init variance for Leaky ReLU."""
    print("\n" + "=" * 60)
    print("Exercise 2: Leaky ReLU Initialization (alpha=0.2)")
    print("=" * 60)

    alpha = 0.2

    # TODO: Derive the correct variance
    # For Leaky ReLU: E[LReLU(z)^2] = (1/2 + alpha^2/2) * E[z^2]
    # So: Var(w) = 2 / ((1 + alpha^2) * fan_in)
    # Implement this initialization and verify by tracking activation std
    raise NotImplementedError("Derive and verify Leaky ReLU init")


if __name__ == "__main__":
    exercise_1_deep_activations()
    exercise_2_leaky_relu_init()
