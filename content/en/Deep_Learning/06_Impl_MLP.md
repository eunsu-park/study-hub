# 06. Multi-Layer Perceptron (MLP)

[Previous: Linear & Logistic Regression](./05_Impl_Linear_Logistic.md) | [Next: CNN Basics](./07_CNN_Basics.md)

---

## Overview

MLP is the fundamental building block of deep learning. Understanding how to train multiple layers through **Backpropagation** is key.

## Learning Objectives

1. **Forward Pass**: Understanding forward propagation in multi-layer structures
2. **Backward Pass**: Backpropagation using the Chain Rule
3. **Activation Functions**: Characteristics and derivatives of ReLU, Sigmoid, Tanh
4. **Weight Initialization**: Importance of proper initialization

---

## Theory & Principles

The MLP is the simplest neural architecture where the full backprop machinery actually has something to do. Tracing the gradient through one hidden layer by hand — and seeing how it generalizes to L layers — is the most useful exercise in this entire course. Once you can derive these formulas you will recognize them inside every PyTorch backward you ever read.

This section covers:

- **A.** Forward pass of an L-layer MLP and the shapes of every quantity
- **B.** Backward pass derived layer by layer
- **C.** Initialization: why Xavier and He are not interchangeable
- **D.** Why width-based scaling matters (variance preservation)

### A. Forward Pass of an L-Layer MLP

For an MLP with L layers, batch size N, and layer widths `d_0 -> d_1 -> ... -> d_L`:

```
z_l = h_{l-1} W_l + b_l                  (pre-activation)   [N x d_l]
h_l = \sigma_l(z_l)                      (activation)       [N x d_l]
h_0 = X                                  (input)            [N x d_0]
\hat{y} = h_L                            (output)           [N x d_L]
```

Notation:
- `W_l \in R^{d_{l-1} x d_l}` (weight matrix), `b_l \in R^{d_l}` (bias broadcast across batch)
- `\sigma_l` is the activation at layer l (typically ReLU for hidden, softmax for last classification layer)

The total parameter count is `sum_l (d_{l-1} d_l + d_l)`, dominated by the matrix terms.

### B. Backward Pass: Derive Once, Apply L Times

Define the upstream gradient at each layer:

```
\delta_l = dL / dz_l                     [N x d_l]
```

The chain rule gives a clean recurrence. Starting from the output:

```
\delta_L = (dL / d\hat{y}) \odot \sigma_L'(z_L)
```

For softmax + cross-entropy, this simplifies dramatically: `\delta_L = (\hat{y} - y_onehot) / N`.

Then for `l = L-1, L-2, ..., 1`:

```
\delta_l = (\delta_{l+1} W_{l+1}^T) \odot \sigma_l'(z_l)         [N x d_l]
dL/dW_l = h_{l-1}^T \delta_l                                    [d_{l-1} x d_l]
dL/db_l = sum over batch of \delta_l                            [d_l]
```

Two patterns to memorize:

1. **The transpose flips direction.** Forward multiplies by `W_l`; backward multiplies by `W_l^T`. The "weight tying" between forward and backward Jacobians is intrinsic to linear layers.
2. **Each layer's gradient depends only on `\delta_{l+1}` and locally cached `(h_{l-1}, z_l)`.** This is the property that makes backprop linear-time and modular.

### C. Initialization: Xavier vs He

If you initialize weights as `N(0, \sigma^2)`, the variance of `z_l` after one layer is `Var(z_l) = d_{l-1} \sigma^2 Var(h_{l-1})` (treating activations as independent). To keep variance constant across layers (so the signal neither explodes nor vanishes), pick `\sigma^2 = 1 / d_{l-1}`.

But that calculation assumes the activation is linear (or ReLU half-active in expectation). For the two main activation regimes:

- **Xavier / Glorot init** (Glorot & Bengio 2010): `\sigma^2 = 2 / (d_{in} + d_{out})`. Designed for tanh / sigmoid, where activations are roughly linear near zero.
- **He init** (He et al. 2015): `\sigma^2 = 2 / d_{in}`. The factor 2 compensates for ReLU killing half the activations on average. Use this for ReLU/Leaky ReLU/GELU.

Picking the wrong one for your activation is one of the most common reasons "deep MLPs do not train" — the variance either decays to zero (vanishing gradients in backward) or explodes (NaN losses).

### D. Variance Preservation Across Depth

The reason these initialization rules matter so much in deep networks: variance compounds multiplicatively. If each layer changes the variance by a factor `c`, then after L layers the variance of activations is `c^L Var(h_0)`. With `c = 1.5` and `L = 50`, this is `1.5^50 \approx 6 * 10^8` — gradient explosion. With `c = 0.5`, this is `0.5^50 \approx 10^{-15}` — gradient vanishing. Only `c \approx 1` permits stable training, and that is exactly what He / Xavier achieve in expectation. BatchNorm and LayerNorm enforce this same property by *measurement* rather than by careful init.

### From Theory to the Code Below

| Theory concept | Code construct in this lesson |
|----------------|-------------------------------|
| Forward pass `h_l = \sigma(h_{l-1} W_l + b_l)` | The hand-written forward loop |
| Backward recurrence `\delta_l = (\delta_{l+1} W_{l+1}^T) \odot \sigma'(z_l)` | The hand-written backward loop |
| He init for ReLU | `nn.init.kaiming_normal_` or manual `* sqrt(2/d_in)` |
| Variance preservation | The empirical observation that loss converges, not diverges |

---


## Mathematical Background

### 1. Forward Pass

```
Input: x ∈ ℝ^d₀

Layer 1: z₁ = W₁x + b₁,  a₁ = σ(z₁)
Layer 2: z₂ = W₂a₁ + b₂,  a₂ = σ(z₂)
...
Output:  ŷ = aₙ

Where:
- Wᵢ ∈ ℝ^(dᵢ × dᵢ₋₁): weight matrix
- bᵢ ∈ ℝ^dᵢ: bias
- σ: activation function
```

### 2. Backward Pass (Backpropagation)

```
Loss: L = Loss(y, ŷ)

Chain Rule:
∂L/∂Wᵢ = ∂L/∂aᵢ × ∂aᵢ/∂zᵢ × ∂zᵢ/∂Wᵢ

Backpropagation order:
1. ∂L/∂ŷ (derivative of loss w.r.t. output)
2. ∂L/∂zₙ = ∂L/∂ŷ × σ'(zₙ)
3. ∂L/∂Wₙ = aₙ₋₁ᵀ × ∂L/∂zₙ
4. ∂L/∂aₙ₋₁ = ∂L/∂zₙ × Wₙᵀ
5. Repeat...
```

### 3. Activation Functions

```
ReLU:     σ(z) = max(0, z)
          σ'(z) = 1 if z > 0 else 0

Sigmoid:  σ(z) = 1/(1 + e⁻ᶻ)
          σ'(z) = σ(z)(1 - σ(z))

Tanh:     σ(z) = (eᶻ - e⁻ᶻ)/(eᶻ + e⁻ᶻ)
          σ'(z) = 1 - σ(z)²
```

---

## File Structure

```
02_MLP/
├── README.md
├── numpy/
│   ├── mlp_numpy.py          # Complete MLP implementation
│   ├── activations_numpy.py   # Activation functions
│   └── test_mlp.py           # Tests
├── pytorch_lowlevel/
│   └── mlp_lowlevel.py       # Implementation without nn.Linear
├── paper/
│   └── mlp_paper.py          # Clean nn.Module
└── exercises/
    ├── 01_add_dropout.md
    ├── 02_batch_norm.md
    └── 03_xor_problem.md
```

---

## Core Concepts

### 1. Vanishing/Exploding Gradients

```
Problem: Gradients vanish or explode as layers get deeper
- Sigmoid: σ'(z) ≤ 0.25 → product converges to 0
- Solution: ReLU, proper initialization, BatchNorm, ResNet

Example:
10 layers, Sigmoid → gradient ≈ 0.25^10 ≈ 10^-6
```

### 2. Xavier/He Initialization

```python
# Xavier (Glorot): for tanh, sigmoid
W = np.random.randn(in_dim, out_dim) * np.sqrt(1 / in_dim)
# Or
W = np.random.randn(in_dim, out_dim) * np.sqrt(2 / (in_dim + out_dim))

# He (Kaiming): for ReLU
W = np.random.randn(in_dim, out_dim) * np.sqrt(2 / in_dim)
```

### 3. Universal Approximation Theorem

> A feedforward network with a single hidden layer can approximate any continuous function, given enough neurons.

---

## Practice Problems

### Basic
1. Solve XOR problem (2-layer MLP)
2. Compare different activation functions
3. Compare learning curves with different initialization methods

### Intermediate
1. Implement Dropout
2. Implement Batch Normalization
3. Implement Learning Rate Scheduler

### Advanced
1. MNIST classification (>98% accuracy)
2. Implement Gradient Clipping
3. Implement Weight Decay (L2 regularization)

---

## References

- Rumelhart et al. (1986). "Learning representations by back-propagating errors"
- Glorot & Bengio (2010). "Understanding the difficulty of training deep feedforward neural networks"
- He et al. (2015). "Delving Deep into Rectifiers" (He initialization)
