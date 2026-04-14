# Universal Approximation Theorem

**Previous**: [Batch Normalization](./10_Batch_Normalization.md) | **Next**: [Training Pipeline](./12_Training_Pipeline.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. State the Universal Approximation Theorem precisely
2. Explain the constructive proof using step functions
3. Visualize how neurons approximate arbitrary functions
4. Distinguish between existence theorems and practical feasibility
5. Explain the width-depth tradeoff in approximation
6. Describe the limitations of the theorem in practice
7. Connect the theorem to network architecture choices
8. Implement a visual demonstration of universal approximation

---

The Universal Approximation Theorem (UAT) is one of the most important theoretical results about neural networks. It tells us that a neural network with a single hidden layer can approximate any continuous function to arbitrary accuracy -- given enough neurons. This result explains why neural networks are so powerful, but also has crucial limitations that every practitioner should understand.

---

## 1. The Theorem

### 1.1 Informal Statement

> A feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of R^n, to any desired degree of accuracy.

### 1.2 Formal Statement (Cybenko, 1989; Hornik, 1991)

Let σ be a non-constant, bounded, continuous activation function (e.g., sigmoid). For any continuous function f: [0,1]^n → R, any ε > 0, there exist N ∈ N, weights w_i ∈ R^n, biases b_i ∈ R, and output weights v_i ∈ R such that:

```
F(x) = Σ_{i=1}^{N} v_i · σ(w_i · x + b_i)

satisfies:    |F(x) - f(x)| < ε    for all x ∈ [0,1]^n
```

### 1.3 What This Means

```
Any continuous function:
y │     ╱╲
  │    ╱  ╲    ╱╲
  │   ╱    ╲  ╱  ╲
  │  ╱      ╲╱    ╲
  │ ╱                ╲
  └────────────────────► x

Can be approximated to ANY precision by:

  F(x) = v1·σ(w1·x + b1) + v2·σ(w2·x + b2) + ... + vN·σ(wN·x + bN)

  (A single hidden layer with N neurons)
```

---

## 2. Intuitive Proof: Building Functions from Bumps

### 2.1 Step 1: Sigmoid as an Approximate Step Function

When the weight w is very large, σ(w·x + b) approximates a step function:

```
w = 1:                    w = 100:
σ(x)                      σ(100·x)
  1 ┤       ────           1 ┤         │──────
    │     ──                 │         │
    │   ──                   │         │
0.5 ┤  ─                  0.5 ┤         │
    │──                      │         │
    │                        │─────────│
  0 ┤                      0 ┤
    └──────────► x           └──────────► x
```

### 2.2 Step 2: Two Sigmoids Make a Bump

Subtract two shifted step functions to create a "bump":

```
bump(x) = σ(w(x - a)) - σ(w(x - b))      where a < b

  1 ┤
    │    ┌────────┐
    │    │        │
    │    │  bump  │
  0 ┤────┘        └────
    └────┬────────┬────► x
         a        b
```

### 2.3 Step 3: Scale Bumps to Match the Target

Create one bump for each region of the target function, scaled to match the function value:

```
Target:  f(x)
y │   ╱╲
  │  ╱  ╲
  │ ╱    ╲
  └────────► x

Approximation using 4 bumps:
y │  ┌┐
  │  ││┌┐
  │ ┌┘│││
  │ │  └┘│
  └─┘────┘─► x

More bumps (neurons) → better approximation
```

### 2.4 Code Demonstration

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bump(x, center, width, height, sharpness=50):
    """Create a bump function using two sigmoids."""
    left = sigmoid(sharpness * (x - (center - width/2)))
    right = sigmoid(sharpness * (x - (center + width/2)))
    return height * (left - right)

# Target function: sin(x)
x = np.linspace(0, 2 * np.pi, 1000)
target = np.sin(x)

# Approximate with N bumps
N = 20
approx = np.zeros_like(x)
for i in range(N):
    center = (i + 0.5) * 2 * np.pi / N
    width = 2 * np.pi / N
    height = np.sin(center)  # sample target at center
    approx += bump(x, center, width, height)

error = np.max(np.abs(target - approx))
print(f"Max error with {N} bumps: {error:.4f}")
# With 20 bumps: error ≈ 0.08
# With 100 bumps: error ≈ 0.003
```

---

## 3. Width vs. Depth

### 3.1 Width Theorem (1 Layer)

The UAT says: one hidden layer is sufficient. But the required width (number of neurons) can be exponentially large:

```
Function complexity    Required width (1 hidden layer)
─────────────────────────────────────────────────────
Simple (linear)        1 neuron
Moderate (polynomial)  O(d) neurons (d = degree)
Complex (fractal-like) Exponentially many neurons
```

### 3.2 Depth Theorem

Deep networks can approximate certain functions with exponentially fewer parameters than shallow networks:

```
Example: Computing parity of n bits
  - 1 hidden layer: needs 2^n neurons
  - O(log n) layers: needs O(n) neurons total

Example: Hierarchical composition f(g(h(x)))
  - Shallow: must learn the entire mapping at once
  - Deep: each layer learns one level of composition
```

### 3.3 The Depth-Width Tradeoff

```
Approximation Quality
  │
  │        Deep (4 layers × 16 neurons = 64 total)
  │       ╱
  │      ╱
  │     ╱    Shallow (1 layer × 64 neurons)
  │    ╱   ╱
  │   ╱  ╱
  │  ╱ ╱
  │ ╱╱
  │╱
  └──────────────────────► Number of Parameters

Deep networks often achieve the same approximation quality with fewer parameters.
But they are harder to train (vanishing gradients, more hyperparameters).
```

---

## 4. Limitations of the Theorem

### 4.1 Existence ≠ Learnability

The UAT guarantees that an approximating network **exists**, but says nothing about:

```
1. How to FIND the right weights (optimization is NP-hard in general)
2. How MANY neurons are needed (could be astronomically large)
3. How much TRAINING DATA is required
4. Whether gradient descent will CONVERGE to the solution
```

### 4.2 Curse of Dimensionality

For functions in high dimensions, the required number of neurons grows exponentially with dimension:

```
Approximating f: R^d → R to accuracy ε

Required neurons: O(ε^(-d))   ← exponential in d!

d = 2:   ε = 0.01 → ~10,000 neurons (feasible)
d = 100: ε = 0.01 → 10^200 neurons (impossible)
```

### 4.3 What the Theorem Does NOT Cover

- Discontinuous functions (only continuous functions on compact sets)
- Generalization to unseen data (only approximation on training domain)
- Computational efficiency (may need impractically many neurons)
- Optimal architecture (does not tell us the best layer sizes)

---

## 5. Practical Implications

### 5.1 What the UAT Tells Us

```
✓ Neural networks are universal function approximators
✓ The architecture has sufficient expressive power
✓ The limitation is NOT representational but computational
✓ Deeper is often better than wider (efficiency)
```

### 5.2 What Practitioners Should Do

```
✗ Don't rely on 1 hidden layer (despite the theorem)
✓ Use multiple hidden layers for better parameter efficiency
✓ Use ReLU (piecewise linear → efficient approximation)
✓ Focus on optimization and generalization, not just approximation
```

---

## 6. Extensions and Modern Results

### 6.1 ReLU Networks

The UAT has been extended to ReLU activation (Lu et al., 2017):

```
ReLU networks with width ≥ d+1 (d = input dimension)
and arbitrary depth can approximate any Lebesgue-integrable function.

ReLU is piecewise linear: the network creates a piecewise linear 
approximation with more pieces = better accuracy.
```

### 6.2 Depth Separation Results

```
Theorem (Telgarsky, 2016): There exist functions that can be 
represented by a network with O(k^3) layers and O(1) width per layer,
but require width 2^Ω(k) for any network with O(k) layers.

→ Deep networks are EXPONENTIALLY more efficient for some functions.
```

### 6.3 The Lottery Ticket Hypothesis

Frankle & Carlin (2019): Dense networks contain sparse sub-networks (winning tickets) that can achieve the same accuracy when trained in isolation. This suggests the UAT's existence guarantee is realized through a tiny fraction of the network's capacity.

---

## 7. Visualization: Approximating sin(x)

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(0, z)

# Train a simple network to approximate sin(x)
np.random.seed(42)
x_train = np.linspace(-np.pi, np.pi, 200).reshape(1, -1)
y_train = np.sin(x_train)

# Network: 1 → N → 1 with ReLU
N = 50
W1 = np.random.randn(N, 1) * 0.5
b1 = np.random.randn(N, 1) * 0.5
W2 = np.random.randn(1, N) * 0.1
b2 = np.zeros((1, 1))

lr = 0.001
for epoch in range(5000):
    # Forward
    z1 = W1 @ x_train + b1
    a1 = relu(z1)
    y_pred = W2 @ a1 + b2

    # Loss
    loss = np.mean((y_pred - y_train) ** 2)

    # Backward
    dy = 2 * (y_pred - y_train) / x_train.shape[1]
    dW2 = dy @ a1.T
    db2 = np.sum(dy, axis=1, keepdims=True)
    da1 = W2.T @ dy
    dz1 = da1 * (z1 > 0)
    dW1 = dz1 @ x_train.T
    db1 = np.sum(dz1, axis=1, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

print(f"Final loss: {loss:.6f}")
```

---

## 8. Summary

```
Key Takeaways
═══════════════════════════════════════════════════════
1. UAT: one hidden layer can approximate any continuous function
2. Proof idea: sum of scaled "bumps" (sigmoid pairs)
3. Width requirement can be exponentially large
4. Deep networks are exponentially more efficient for some functions
5. UAT guarantees existence, NOT learnability
6. Curse of dimensionality: high-D functions need exponentially many neurons
7. ReLU networks: piecewise linear approximation
8. Practical takeaway: use deep networks, focus on optimization
═══════════════════════════════════════════════════════
```

---

## Exercises

1. Implement the bump-function construction to approximate f(x) = x^2 on [0, 1]
2. Train a 1-hidden-layer network to approximate sin(x) and measure the error vs. width
3. Compare approximation quality: shallow (1 layer, 256 neurons) vs. deep (4 layers, 16 neurons each)
4. Verify that increasing the number of bumps monotonically decreases approximation error

---

**Previous**: [Batch Normalization](./10_Batch_Normalization.md) | **Next**: [Training Pipeline](./12_Training_Pipeline.md)
