# 02. Neural Network Basics

[Previous: Tensors and Autograd](./01_Tensors_and_Autograd.md) | [Next: Backpropagation](./03_Backpropagation.md)

---

## Learning Objectives

- Understand perceptrons and Multi-Layer Perceptrons (MLP)
- Learn the role and types of activation functions
- Build neural networks using PyTorch's `nn.Module`

## 1. Perceptron

The most basic unit of a neural network. The perceptron is loosely inspired by biological neurons: dendrites receive signals (inputs), the soma performs a weighted sum, the axon hillock applies a threshold (activation function), and the axon transmits the output. While real neurons are far more complex, this analogy captures the core idea — gather information, aggregate it, and fire (or not) based on the result.

```
Input(x₁) ──w₁──┐
                │
Input(x₂) ──w₂──┼──▶ Σ(wᵢxᵢ + b) ──▶ Activation ──▶ Output(y)
                │
Input(x₃) ──w₃──┘
```

### Theory: Linear Layers and the Need for Nonlinearity

A single linear layer computes `y = W x + b` with `W \in R^{m x n}` and `b \in R^m`. This is an **affine map**, and stacking two affine maps yields another affine map:

```
W_2 (W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)
```

The compositions collapse. No matter how many linear layers you stack, the result is still a single affine function — which can only model linear separating hyperplanes. The **activation function** σ breaks this collapse:

```
y = σ(W_2 σ(W_1 x + b_1) + b_2)
```

Now the composition is genuinely nonlinear and can represent decision boundaries with curvature. This is the entire reason activations exist: not for biological plausibility but to prevent the network from collapsing into a single linear map.


### Formula

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = Σwᵢxᵢ + b
y = activation(z)
```

### NumPy Implementation

```python
import numpy as np

def perceptron(x, w, b, activation):
    z = np.dot(x, w) + b
    return activation(z)

# Example: Simple linear output
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, -0.3, 0.8])
b = 0.1

z = np.dot(x, w) + b  # 1*0.5 + 2*(-0.3) + 3*0.8 + 0.1 = 2.4
```

---

## 2. Activation Functions

Without non-linear activation functions, stacking N linear layers collapses into a single matrix multiplication: `W_N ... W_2 W_1 x = W x`. No matter how deep the network, it can only learn linear mappings. Non-linear activations break this collapse, letting the network approximate *any* continuous function — a result known as the Universal Approximation Theorem.

Add non-linearity to enable learning of complex patterns.

### Theory: Activation Functions Compared

| Activation | Formula | Range | Saturation | Gradient at saturation |
|-----------|---------|-------|------------|------------------------|
| Sigmoid | `1 / (1 + e^{-x})` | (0, 1) | both ends | ≤ 0.25 |
| Tanh | `tanh(x)` | (-1, 1) | both ends | → 0 |
| ReLU | `max(0, x)` | [0, ∞) | left only | exactly 0 |
| Leaky ReLU | `max(αx, x)` | R | none (small α) | α |
| GELU | `x · Φ(x)` | R | smooth left | small but nonzero |

Sigmoid and tanh suffer from **vanishing gradients**: when `|x|` is large their derivative is near zero, and stacking many such layers makes backprop signals decay multiplicatively. ReLU (Glorot et al. 2011) avoided this by being non-saturating on the positive side, which is the single biggest reason deep networks became trainable beyond ~5 layers. The trade-off is **dying ReLU** — neurons stuck in the negative region produce zero gradient forever — which Leaky ReLU and GELU address with smoother left-tails.


### Main Activation Functions

| Function | Formula | Characteristics |
|----------|---------|----------------|
| Sigmoid | σ(x) = 1/(1+e⁻ˣ) | Output 0~1, vanishing gradient problem |
| Tanh | tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | Output -1~1 |
| ReLU | max(0, x) | Most widely used, simple and effective |
| Leaky ReLU | max(αx, x) | Small gradient in negative region |
| GELU | x·Φ(x) | Used in Transformers |

**Why does sigmoid cause vanishing gradients?** The sigmoid derivative is `σ'(x) = σ(x)(1 - σ(x))`, which reaches its maximum at `x = 0` where `σ'(0) = 0.25`. In a deep network, gradients are multiplied across layers. After just 10 layers, the gradient shrinks to at most `0.25^10 ≈ 0.0000009` — essentially zero. Early layers receive almost no learning signal, so training stalls.

**Why ReLU fixes this:** For positive inputs, ReLU's derivative is exactly 1, so the gradient passes through unchanged regardless of depth. This is why ReLU (and its variants) enabled training of much deeper networks.

### NumPy Implementation

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)
```

### PyTorch

```python
import torch.nn.functional as F

y = F.sigmoid(x)
y = F.relu(x)
y = F.tanh(x)
```

---

## 3. Multi-Layer Perceptron (MLP)

Approximates complex functions by stacking multiple layers.

```
Input Layer ──▶ Hidden Layer 1 ──▶ Hidden Layer 2 ──▶ Output Layer
(n units)        (h1 units)          (h2 units)          (m units)
```

### Theory: Width vs Depth

Both wide-shallow and narrow-deep networks are universal approximators in principle, but they trade off differently in practice. Several theoretical results (Telgarsky 2016, Eldan & Shamir 2016) show that there exist functions a deep network can represent with `O(n)` parameters but a shallow network needs `O(2^n)` — an *exponential* depth/width gap. Intuitively, depth lets the network reuse intermediate features hierarchically (edges → textures → parts → objects), while width forces every feature to be expressed independently. The empirical success of deep networks is consistent with this picture: depth is a strong inductive bias for compositional structure.


### Theory: Universal Approximation Theorem

**Cybenko (1989)** proved that a feedforward network with a single hidden layer of sigmoid units can approximate any continuous function on a compact subset of `R^n` to arbitrary precision, given enough hidden units. **Hornik (1991)** generalized this: any non-polynomial activation works.

Formally: for any continuous `f: K -> R` on compact `K` and any `\epsilon > 0`, there exist `N`, weights `w_i, v_i`, biases `b_i`, and a non-polynomial activation `\sigma` such that:

```
| f(x) - sum_{i=1}^{N} v_i \sigma(w_i^T x + b_i) | < \epsilon  for all x in K
```

Two important caveats:

1. **Existence, not construction.** The theorem says an approximating network exists; it does not tell you how to find its weights. Training (gradient descent + backprop) is the construction algorithm.
2. **Width can blow up.** `N` may need to grow exponentially in input dimension. This is the *curse of dimensionality* — it does not forbid a solution but makes shallow solutions impractical.


### Forward Pass

```python
# 2-layer MLP forward pass
z1 = x @ W1 + b1       # First linear transformation
a1 = relu(z1)          # Activation
z2 = a1 @ W2 + b2      # Second linear transformation
y = softmax(z2)        # Output (for classification)
```

---

## 4. PyTorch nn.Module

The standard way to define neural networks in PyTorch.

### Basic Structure

```python
import torch
import torch.nn as nn

# Why inherit nn.Module?  It automatically tracks all parameters (for optimizer),
# handles device transfer (model.to('cuda')), enables save/load (state_dict),
# and provides training/eval mode switching.
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    # Why define forward()?  This method is called automatically when you do
    # model(x).  It defines the computation graph — the path data takes through
    # layers.  PyTorch records each operation here for autograd.
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

### Using nn.Sequential

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

---

## 5. Weight Initialization

Proper initialization significantly impacts training performance.

| Method | Characteristics | Usage |
|--------|----------------|-------|
| Xavier/Glorot | Suitable for Sigmoid, Tanh | `nn.init.xavier_uniform_` |
| He/Kaiming | Suitable for ReLU | `nn.init.kaiming_uniform_` |
| Zero Initialization | Not recommended (symmetry problem) | - |

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

model.apply(init_weights)
```

---

## 6. Exercise: Solving the XOR Problem

Solve the XOR problem with an MLP, which cannot be solved by a single-layer perceptron.

### Data

```
Input      Output
(0, 0) → 0
(0, 1) → 1
(1, 0) → 1
(1, 1) → 0
```

### MLP Structure

```
Input(2) ──▶ Hidden(4) ──▶ Output(1)
```

### PyTorch Implementation

```python
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(1000):
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 7. NumPy vs PyTorch Comparison

### MLP Forward Pass

```python
# NumPy (manual)
def forward_numpy(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = a1 @ W2 + b2
    return z2

# PyTorch (automatic)
class MLP(nn.Module):
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Key Differences

| Item | NumPy | PyTorch |
|------|-------|---------|
| Forward Pass | Direct implementation | `forward()` method |
| Backpropagation | Manual derivative computation | Automatic `loss.backward()` |
| Parameter Management | Direct array management | `model.parameters()` |

---

## Summary

### Core Concepts

1. **Perceptron**: Linear transformation + activation function
2. **Activation Functions**: Add non-linearity (ReLU recommended)
3. **MLP**: Stack multiple layers to learn complex functions
4. **nn.Module**: PyTorch's base class for neural networks

### What You Learn from NumPy Implementation

- Meaning of matrix operations
- Mathematical definition of activation functions
- Data flow in forward pass

---

## Next Steps

In [03_Backpropagation.md](./03_Backpropagation.md), we'll directly implement the backpropagation algorithm with NumPy.
