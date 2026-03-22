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
