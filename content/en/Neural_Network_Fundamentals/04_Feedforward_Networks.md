# Feedforward Networks

**Previous**: [Activation Functions](./03_Activation_Functions.md) | **Next**: [Loss Functions](./05_Loss_Functions.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define the architecture of a Multi-Layer Perceptron (MLP)
2. Express the forward pass as a sequence of matrix operations
3. Implement a feedforward pass for an arbitrary-depth MLP using NumPy
4. Explain the role of each layer type: input, hidden, output
5. Calculate the total number of parameters in an MLP
6. Draw the computational graph for a 2-layer MLP
7. Distinguish between width and depth and their effects on capacity
8. Use proper notation (superscripts for layers, subscripts for neurons)

---

A feedforward neural network -- also called a Multi-Layer Perceptron (MLP) -- is the simplest deep architecture. Data flows in one direction: from input to output, passing through one or more hidden layers. Despite its simplicity, the MLP is the foundation upon which every modern architecture (CNN, RNN, Transformer) is built.

---

## 1. MLP Architecture

### 1.1 Terminology

```
Input Layer        Hidden Layer 1      Hidden Layer 2      Output Layer
(n0 neurons)       (n1 neurons)        (n2 neurons)        (n3 neurons)

  x1 ─────────┬──► h1^(1) ────────┬──► h1^(2) ────────┬──► y1
              │                   │                   │
  x2 ─────────┼──► h2^(1) ────────┼──► h2^(2) ────────┼──► y2
              │                   │                   │
  x3 ─────────┼──► h3^(1) ────────┼──►   ...  ────────┤
              │                   │                   │
  x4 ─────────┘    ...            └───────────────────┘

  Layer 0           Layer 1             Layer 2             Layer 3
  (input)           (hidden)            (hidden)            (output)
```

**Key terms**:
- **Input layer** (Layer 0): Receives raw features. Not a "real" layer -- no computation.
- **Hidden layer** (Layer l): Performs linear transform + activation. "Hidden" because we don't directly observe its outputs.
- **Output layer** (Layer L): Produces the final prediction. Activation depends on the task.
- **Depth**: Number of layers with learnable parameters (hidden + output).
- **Width**: Number of neurons in a layer.
- **Fully connected (Dense)**: Every neuron in layer l connects to every neuron in layer l+1.

### 1.2 Notation Convention

We use superscripts for layers and subscripts for neurons within a layer:

```
W^(l)     = Weight matrix for layer l    (shape: n_l × n_{l-1})
b^(l)     = Bias vector for layer l      (shape: n_l × 1)
z^(l)     = Pre-activation at layer l    (shape: n_l × 1)
a^(l)     = Activation at layer l        (shape: n_l × 1)

W^(l)_ij  = Weight from neuron j in layer (l-1) to neuron i in layer l
```

---

## 2. The Forward Pass

### 2.1 Single Layer Computation

For layer l with n_l neurons, receiving input a^(l-1) of size n_{l-1}:

```
z^(l) = W^(l) · a^(l-1) + b^(l)     (linear transform)
a^(l) = σ(z^(l))                      (activation function)
```

Expanded for a layer with 3 inputs and 2 neurons:

```
┌ z1^(l) ┐   ┌ w11  w12  w13 ┐   ┌ a1^(l-1) ┐   ┌ b1^(l) ┐
│         │ = │               │ · │           │ + │        │
└ z2^(l) ┘   └ w21  w22  w23 ┘   │ a2^(l-1) │   └ b2^(l) ┘
                                  └ a3^(l-1) ┘

    (2×1)         (2×3)              (3×1)          (2×1)
```

### 2.2 Full Forward Pass

For an L-layer network:

```
a^(0) = x                                         (input)
z^(1) = W^(1) · a^(0) + b^(1)                     (layer 1 pre-activation)
a^(1) = σ(z^(1))                                   (layer 1 activation)
z^(2) = W^(2) · a^(1) + b^(2)                     (layer 2 pre-activation)
a^(2) = σ(z^(2))                                   (layer 2 activation)
...
z^(L) = W^(L) · a^(L-1) + b^(L)                   (output pre-activation)
a^(L) = σ_out(z^(L))                               (output activation)

ŷ = a^(L)
```

### 2.3 Computational Graph

```
x ──► [W^(1), b^(1)] ──► z^(1) ──► σ ──► a^(1) ──► [W^(2), b^(2)] ──► z^(2) ──► σ_out ──► ŷ
       │                                              │
       │    layer 1                                   │    layer 2
       │    (hidden)                                  │    (output)
       └──────────────────────────────────────────────┘
```

---

## 3. Matrix Dimensions

### 3.1 Shape Rules

For a network with layer sizes [n0, n1, n2, ..., nL]:

```
Layer l:
  W^(l):  shape (n_l, n_{l-1})     ← n_l rows, n_{l-1} columns
  b^(l):  shape (n_l, 1)           ← one bias per neuron
  z^(l):  shape (n_l, 1)           ← one pre-activation per neuron
  a^(l):  shape (n_l, 1)           ← one activation per neuron
```

### 3.2 Example: Network [4, 8, 6, 3]

```
Layer sizes: input=4, hidden1=8, hidden2=6, output=3

Layer 1: W^(1) is (8, 4),  b^(1) is (8, 1)  →  8×4 + 8 = 40 params
Layer 2: W^(2) is (6, 8),  b^(2) is (6, 1)  →  6×8 + 6 = 54 params
Layer 3: W^(3) is (3, 6),  b^(3) is (3, 1)  →  3×6 + 3 = 21 params

Total parameters: 40 + 54 + 21 = 115
```

### 3.3 Parameter Count Formula

```
Total parameters = Σ_{l=1}^{L} (n_l × n_{l-1} + n_l)
                 = Σ_{l=1}^{L} n_l × (n_{l-1} + 1)
```

---

## 4. Batch Processing

### 4.1 Single Sample vs. Batch

In practice, we process multiple samples simultaneously using matrix operations:

```
Single sample:
  a^(0) = x          shape: (n0, 1)
  z^(l) = W·a + b    shape: (n_l, 1)

Batch of m samples:
  A^(0) = X           shape: (n0, m)     ← each column is one sample
  Z^(l) = W·A + b     shape: (n_l, m)    ← b is broadcast across columns
```

### 4.2 Why Batches Are Faster

```
100 samples, individually:     100 × matrix-vector multiply
100 samples, as batch:         1 × matrix-matrix multiply

Matrix-matrix multiply exploits CPU/GPU parallelism (BLAS, CUDA).
Speedup: often 10-100× compared to looping over samples.
```

---

## 5. Implementation

### 5.1 NumPy Forward Pass

```python
import numpy as np

def initialize_network(layer_sizes):
    """Initialize weights and biases for an MLP.
    
    Args:
        layer_sizes: List of layer sizes, e.g., [4, 8, 6, 3]
    
    Returns:
        List of (W, b) tuples for each layer.
    """
    params = []
    for i in range(1, len(layer_sizes)):
        W = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * 0.01
        b = np.zeros((layer_sizes[i], 1))
        params.append((W, b))
    return params

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward(X, params):
    """Forward pass through the network.
    
    Args:
        X: Input matrix of shape (n_features, n_samples)
        params: List of (W, b) tuples
    
    Returns:
        output: Network output
        cache: List of (z, a) for each layer (needed for backprop)
    """
    cache = []
    a = X
    for i, (W, b) in enumerate(params):
        z = W @ a + b
        if i < len(params) - 1:
            a = relu(z)             # hidden layers: ReLU
        else:
            a = softmax(z)          # output layer: Softmax
        cache.append((z, a))
    return a, cache

# Example: 3-class classification with 4 features
layer_sizes = [4, 8, 6, 3]
params = initialize_network(layer_sizes)

# Single sample
x = np.random.randn(4, 1)
output, cache = forward(x, params)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output (probabilities): {output.ravel()}")
print(f"Sum of probabilities: {output.sum():.4f}")

# Batch of 32 samples
X_batch = np.random.randn(4, 32)
output_batch, _ = forward(X_batch, params)
print(f"\nBatch input shape: {X_batch.shape}")
print(f"Batch output shape: {output_batch.shape}")
```

### 5.2 Parameter Count Verification

```python
def count_parameters(params):
    """Count total trainable parameters."""
    total = 0
    for i, (W, b) in enumerate(params):
        n_params = W.size + b.size
        print(f"Layer {i+1}: W{W.shape} + b{b.shape} = {n_params} params")
        total += n_params
    print(f"Total: {total}")
    return total

count_parameters(params)
# Layer 1: W(8, 4) + b(8, 1) = 40 params
# Layer 2: W(6, 8) + b(6, 1) = 54 params
# Layer 3: W(3, 6) + b(3, 1) = 21 params
# Total: 115
```

---

## 6. Width vs. Depth

### 6.1 Width: More Neurons Per Layer

```
Narrow:  [4, 4, 4, 3]     →  4×4+4 + 4×4+4 + 3×4+3 = 51 params
Wide:    [4, 64, 3]        →  64×4+64 + 3×64+3 = 515 params
```

- More width → can learn more features per layer
- Too much width → overfitting, more compute

### 6.2 Depth: More Layers

```
Shallow: [4, 128, 3]       →  128×4+128 + 3×128+3 = 1027 params
Deep:    [4, 16, 16, 16, 3] → 16×4+16 + 16×16+16 + 16×16+16 + 3×16+3 = 643 params
```

- More depth → can learn **hierarchical** features
- Too much depth → vanishing gradients, harder to train
- Deep networks are more **parameter-efficient** than wide shallow networks

### 6.3 Practical Guidelines

```
┌──────────────────────────────────────────────────────────┐
│ Start with 1-2 hidden layers.                            │
│ Use 64-256 neurons per hidden layer.                     │
│ Increase depth before width for structured data.         │
│ Use skip connections (ResNet) for very deep networks.    │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Common MLP Architectures

### 7.1 Binary Classification

```
[n_features] → [64] → [32] → [1]
  input        ReLU    ReLU   Sigmoid
                               ↓
                          P(y=1|x) ∈ (0,1)
```

### 7.2 Multi-class Classification (K classes)

```
[n_features] → [128] → [64] → [K]
  input         ReLU    ReLU   Softmax
                                ↓
                         [P(y=1), P(y=2), ..., P(y=K)]
```

### 7.3 Regression

```
[n_features] → [64] → [32] → [1]
  input        ReLU    ReLU   Identity (no activation)
                               ↓
                          ŷ ∈ (-∞, ∞)
```

---

## 8. Summary

```
Key Takeaways
═══════════════════════════════════════════════════════
1. MLP = Input → Hidden layers → Output (fully connected)
2. Forward pass: z = W·a + b, then a = σ(z) per layer
3. Matrix dimensions: W^(l) is (n_l × n_{l-1})
4. Batch processing: process m samples simultaneously
5. Total params = Σ n_l × (n_{l-1} + 1)
6. Depth enables hierarchical features; width adds capacity
7. Output activation depends on task (Sigmoid/Softmax/Identity)
═══════════════════════════════════════════════════════
```

---

## Exercises

1. Compute the parameter count for a network with layer sizes [784, 256, 128, 10]
2. Implement forward pass for a 3-layer MLP and verify output shapes
3. Compare a wide network [4, 256, 3] vs. deep network [4, 32, 32, 32, 3] on the same data
4. Draw the computational graph for a network with layer sizes [2, 3, 1]

---

**Previous**: [Activation Functions](./03_Activation_Functions.md) | **Next**: [Loss Functions](./05_Loss_Functions.md)
