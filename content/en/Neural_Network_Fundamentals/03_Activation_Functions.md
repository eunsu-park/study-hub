# Activation Functions

**Previous**: [Perceptron and Linear Classifiers](./02_Perceptron_and_Linear_Classifiers.md) | **Next**: [Feedforward Networks](./04_Feedforward_Networks.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why nonlinear activation functions are essential in neural networks
2. Implement Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, GELU, and Softmax from scratch
3. Compute the derivative of each activation function
4. Identify the vanishing gradient problem and which activations suffer from it
5. Explain the "dying ReLU" problem and its solutions
6. Choose the right activation function for hidden layers vs. output layers
7. Plot and visually compare activation functions and their gradients

---

Without activation functions, a neural network is just a stack of linear transformations -- which collapses to a single linear transformation regardless of depth. Activation functions introduce the nonlinearity that gives neural networks their expressive power. Choosing the right activation function can be the difference between a model that trains in minutes and one that never converges.

---

## 1. Why Do We Need Activation Functions?

### 1.1 The Linearity Problem

Consider a 2-layer network without activation functions:

```
Layer 1:  h = W1 · x + b1
Layer 2:  y = W2 · h + b2
         = W2 · (W1 · x + b1) + b2
         = (W2 · W1) · x + (W2 · b1 + b2)
         = W' · x + b'           ← Just a single linear layer!
```

No matter how many layers you stack, **without nonlinearity the entire network collapses to a single linear transformation**. A 100-layer linear network has the same expressiveness as a 1-layer linear model.

### 1.2 What Activation Functions Do

```
Input ──► Linear Transform ──► Activation ──► Output
          z = Wx + b           a = σ(z)

The activation function σ bends, squashes, or clips the output,
introducing curves into the function the network can represent.
```

---

## 2. Activation Function Catalog

### 2.1 Sigmoid (Logistic)

```
σ(z) = 1 / (1 + e^(-z))

Output range: (0, 1)
Derivative:   σ'(z) = σ(z) · (1 - σ(z))

         1.0 ┤                    ─────────
             │                ───
             │             ──
         0.5 ┤           ─
             │         ──
             │      ───
         0.0 ┤─────
             └────┬────┬────┬────┬────┬───► z
                 -6   -3    0    3    6
```

**Properties**:
- Smooth, differentiable everywhere
- Output bounded in (0, 1) -- interpretable as probability
- Maximum gradient = 0.25 at z = 0

**Problems**:
- **Vanishing gradients**: For |z| > 5, σ'(z) ≈ 0 → gradients die
- **Not zero-centered**: Outputs always positive → zig-zag gradient updates
- **Expensive**: exp() is computationally costly

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```

### 2.2 Tanh (Hyperbolic Tangent)

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z)) = 2·σ(2z) - 1

Output range: (-1, 1)
Derivative:   tanh'(z) = 1 - tanh^2(z)

         1.0 ┤                    ─────────
             │                ───
             │             ──
         0.0 ┤───────────┼
             │         ──
             │      ───
        -1.0 ┤─────
             └────┬────┬────┬────┬────┬───► z
                 -6   -3    0    3    6
```

**Properties**:
- **Zero-centered** (outputs range from -1 to 1)
- Stronger gradients than sigmoid (max gradient = 1.0 at z = 0)
- Relationship: tanh(z) = 2·sigmoid(2z) - 1

**Problems**:
- Still suffers from vanishing gradients for large |z|
- Still uses exp(), somewhat expensive

```python
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2
```

### 2.3 ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)

Output range: [0, ∞)
Derivative:   ReLU'(z) = { 1  if z > 0
                          { 0  if z < 0
                          { undefined at z = 0 (use 0 or 1)

             │           /
         4.0 ┤          /
             │         /
         2.0 ┤        /
             │       /
         0.0 ┤──────/
             └────┬────┬────┬────┬───► z
                 -4   -2    0    2    4
```

**Properties**:
- **Fast**: No exp(), just a comparison
- **No vanishing gradient** for z > 0 (gradient = 1)
- **Sparse activation**: Negative inputs produce zero → natural sparsity
- The most widely used activation in hidden layers since 2012

**Problems**:
- **Dying ReLU**: If z < 0, gradient = 0 → neuron stops learning permanently
- Not zero-centered (outputs are always ≥ 0)
- Unbounded output → can cause exploding activations

```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

### 2.4 Leaky ReLU

```
LeakyReLU(z) = { z           if z > 0
               { α · z       if z ≤ 0      (typically α = 0.01)

Output range: (-∞, ∞)
Derivative:   LeakyReLU'(z) = { 1   if z > 0
                               { α   if z ≤ 0

             │           /
         4.0 ┤          /
             │         /
         2.0 ┤        /
             │       /
         0.0 ┤──── /       (slight negative slope α)
             │  ──
        -0.1 ┤──
             └────┬────┬────┬────┬───► z
                 -4   -2    0    2    4
```

**Properties**:
- Fixes the dying ReLU problem -- small gradient for negative inputs
- Still fast (just a comparison + multiplication)
- Parametric ReLU (PReLU): learns α as a trainable parameter

```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)
```

### 2.5 ELU (Exponential Linear Unit)

```
ELU(z) = { z                if z > 0
          { α · (e^z - 1)   if z ≤ 0      (typically α = 1.0)

Output range: [-α, ∞)
Derivative:   ELU'(z) = { 1               if z > 0
                         { α · e^z = ELU(z) + α   if z ≤ 0
```

**Properties**:
- Smooth everywhere (unlike ReLU's sharp corner at z=0)
- Negative saturation provides noise robustness
- Pushes mean activations closer to zero

```python
def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

def elu_derivative(z, alpha=1.0):
    return np.where(z > 0, 1.0, alpha * np.exp(z))
```

### 2.6 GELU (Gaussian Error Linear Unit)

```
GELU(z) = z · Φ(z)    where Φ(z) is the standard normal CDF

Approximation:
GELU(z) ≈ 0.5 · z · (1 + tanh(√(2/π) · (z + 0.044715 · z^3)))

Output range: [≈-0.17, ∞)
```

**Properties**:
- Used in modern Transformers (BERT, GPT)
- Smooth approximation of ReLU with probabilistic interpretation
- Stochastic regularization effect: "gate" the input by its percentile

```python
def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))

def gelu_derivative(z):
    # Numerical approximation
    eps = 1e-7
    return (gelu(z + eps) - gelu(z - eps)) / (2 * eps)
```

### 2.7 Softmax (for Output Layer)

```
Softmax(zi) = e^zi / Σ(e^zj)    for j = 1, ..., K

Output range: (0, 1) for each class, sums to 1
```

Softmax converts a vector of raw scores (logits) into a probability distribution:

```
logits:        [2.0, 1.0, 0.1]
exp(logits):   [7.39, 2.72, 1.11]
sum:           11.22
softmax:       [0.659, 0.242, 0.099]   ← sums to 1.0
```

**Numerical stability trick**: Subtract max(z) before exponentiating to prevent overflow.

```python
def softmax(z):
    """Numerically stable softmax."""
    z_shifted = z - np.max(z)  # prevent overflow
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

# Example
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Logits: {logits}")
print(f"Softmax: {probs}")
print(f"Sum: {probs.sum():.4f}")  # 1.0000
```

---

## 3. The Vanishing Gradient Problem

### 3.1 The Problem

During backpropagation, gradients are multiplied through layers via the chain rule:

```
∂L/∂w1 = ∂L/∂aL · ∂aL/∂zL · ∂zL/∂aL-1 · ... · ∂a2/∂z2 · ∂z2/∂a1 · ∂a1/∂z1 · ∂z1/∂w1
                    └──┬──┘             └──┬──┘
                  σ'(zL) ≤ 0.25      σ'(z2) ≤ 0.25

For sigmoid: each σ'(z) ≤ 0.25
After L layers: gradient ≤ 0.25^L

L=4:   gradient ≤ 0.0039  (vanishing!)
L=10:  gradient ≤ 9.5e-7  (essentially zero)
```

### 3.2 Impact

- **Early layers barely learn**: Gradients shrink exponentially as they flow backward
- **Deep networks stall**: With sigmoid/tanh, networks deeper than ~5 layers are nearly untrainable
- **This was the core problem** during the second AI winter (1990s)

### 3.3 Solutions

| Solution | How It Helps |
|----------|-------------|
| ReLU activation | Gradient = 1 for z > 0 (no multiplication decay) |
| Residual connections | Gradient flows through skip connections |
| Batch normalization | Keeps activations in the non-saturating regime |
| Proper initialization | Prevents activations from saturating at initialization |
| LSTM/GRU gates | Explicit gradient highways in recurrent networks |

---

## 4. Activation Function Selection Guide

### 4.1 Hidden Layers

```
Decision Tree for Hidden Layer Activation:
──────────────────────────────────────────
Start with ReLU
  │
  ├── Training works? → Keep ReLU ✓
  │
  ├── Dying neurons? (many zero activations)
  │     ├── Try Leaky ReLU or ELU
  │     └── Check initialization (He init recommended)
  │
  └── Using Transformer architecture?
        └── Use GELU
```

**Rule of thumb**: Start with ReLU. Switch to Leaky ReLU or GELU only if needed.

### 4.2 Output Layer

The output activation depends on the task:

| Task | Output Activation | Loss Function |
|------|------------------|---------------|
| Binary classification | Sigmoid | Binary Cross-Entropy |
| Multi-class classification | Softmax | Categorical Cross-Entropy |
| Regression | None (identity) | MSE |
| Regression (bounded) | Sigmoid or Tanh | MSE |
| Multi-label classification | Sigmoid (per class) | Binary Cross-Entropy |

### 4.3 Comparison Table

| Activation | Range | Zero-Centered | Gradient Issues | Computation |
|-----------|-------|---------------|-----------------|-------------|
| Sigmoid | (0, 1) | No | Vanishing | Slow (exp) |
| Tanh | (-1, 1) | Yes | Vanishing | Slow (exp) |
| ReLU | [0, ∞) | No | Dying neurons | Fast |
| Leaky ReLU | (-∞, ∞) | No* | Minimal | Fast |
| ELU | [-α, ∞) | Near-zero | Minimal | Medium (exp) |
| GELU | [≈-0.17, ∞) | No | Minimal | Medium |
| Softmax | (0, 1)^K | N/A | N/A | Medium |

---

## 5. Derivatives Summary

For backpropagation, we need the derivative of each activation function:

```
Sigmoid:     σ'(z) = σ(z) · (1 - σ(z))
Tanh:        tanh'(z) = 1 - tanh²(z)
ReLU:        ReLU'(z) = 1 if z > 0, else 0
Leaky ReLU:  LReLU'(z) = 1 if z > 0, else α
ELU:         ELU'(z) = 1 if z > 0, else α·e^z
Softmax:     ∂Si/∂zj = Si·(δij - Sj)   where δij is Kronecker delta
```

The Softmax Jacobian is a matrix (not a simple element-wise derivative):

```
∂S/∂z = diag(S) - S · S^T

For S = [s1, s2, s3]:
┌                              ┐
│ s1(1-s1)   -s1·s2    -s1·s3 │
│ -s2·s1    s2(1-s2)   -s2·s3 │
│ -s3·s1    -s3·s2    s3(1-s3) │
└                              ┘
```

---

## 6. Visualization Code

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 200)

activations = {
    'Sigmoid': (sigmoid(z), sigmoid_derivative(z)),
    'Tanh': (np.tanh(z), 1 - np.tanh(z)**2),
    'ReLU': (relu(z), relu_derivative(z)),
    'Leaky ReLU': (leaky_relu(z), leaky_relu_derivative(z)),
    'GELU': (gelu(z), gelu_derivative(z)),
}

fig, axes = plt.subplots(2, len(activations), figsize=(20, 8))
for i, (name, (act, deriv)) in enumerate(activations.items()):
    axes[0, i].plot(z, act, 'b-', linewidth=2)
    axes[0, i].set_title(name)
    axes[0, i].axhline(y=0, color='k', linewidth=0.5)
    axes[0, i].axvline(x=0, color='k', linewidth=0.5)
    axes[0, i].grid(True, alpha=0.3)

    axes[1, i].plot(z, deriv, 'r-', linewidth=2)
    axes[1, i].set_title(f"{name} derivative")
    axes[1, i].axhline(y=0, color='k', linewidth=0.5)
    axes[1, i].axvline(x=0, color='k', linewidth=0.5)
    axes[1, i].grid(True, alpha=0.3)

axes[0, 0].set_ylabel('Activation')
axes[1, 0].set_ylabel('Derivative')
plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150)
plt.show()
```

---

## 7. Summary

```
Key Takeaways
═══════════════════════════════════════════════════════
1. Without activation functions, deep networks = one linear layer
2. Sigmoid: bounded (0,1), good for output, vanishing gradients
3. Tanh: zero-centered (-1,1), better than sigmoid, still vanishes
4. ReLU: fast, default choice, but "dying neurons" possible
5. Leaky ReLU / ELU: fix dying ReLU, slight negative slope
6. GELU: smooth ReLU variant, standard in Transformers
7. Softmax: converts logits to probabilities for classification
8. Choose ReLU for hidden layers; output activation depends on task
═══════════════════════════════════════════════════════
```

---

## Exercises

1. Implement all activation functions and verify their derivatives numerically
2. Show that σ'(z) achieves its maximum of 0.25 at z = 0
3. Create a plot comparing all activation functions on the same axes
4. Demonstrate the dying ReLU problem: initialize a neuron with negative bias and show it never recovers

---

**Previous**: [Perceptron and Linear Classifiers](./02_Perceptron_and_Linear_Classifiers.md) | **Next**: [Feedforward Networks](./04_Feedforward_Networks.md)
