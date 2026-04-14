# Weight Initialization

**Previous**: [Gradient Descent Variants](./07_Gradient_Descent_Variants.md) | **Next**: [Regularization](./09_Regularization.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why proper weight initialization is critical for training
2. Demonstrate the symmetry breaking problem with zero initialization
3. Derive Xavier/Glorot initialization from variance analysis
4. Derive He/Kaiming initialization for ReLU networks
5. Implement both initialization schemes from scratch
6. Diagnose initialization problems from activation statistics
7. Choose the appropriate initialization for a given activation function
8. Analyze variance propagation through layers

---

Weight initialization might seem like a minor detail, but it can make or break neural network training. Initialize too large, and activations explode; too small, and they vanish. Initialize symmetrically, and neurons never differentiate. This lesson derives the theory behind Xavier and He initialization and shows why these principled approaches work so well.

---

## 1. Why Initialization Matters

### 1.1 The Experiment

Consider a 10-layer MLP with 256 neurons per layer, all using tanh activation. Let us track how activations behave with different initializations:

```
Too Large (w ~ N(0, 1)):
  Layer 1:  mean activation ≈ 0,  std ≈ 0.98 (saturated)
  Layer 5:  mean ≈ 0,  std ≈ 1.0 (fully saturated at ±1)
  Layer 10: mean ≈ 0,  std ≈ 1.0 (all neurons at ±1, gradients ≈ 0)

Too Small (w ~ N(0, 0.001)):
  Layer 1:  mean ≈ 0,  std ≈ 0.015
  Layer 5:  mean ≈ 0,  std ≈ 0.00001
  Layer 10: mean ≈ 0,  std ≈ 0.0000000001 (effectively zero)

Just Right (Xavier: w ~ N(0, 1/n)):
  Layer 1:  mean ≈ 0,  std ≈ 0.58
  Layer 5:  mean ≈ 0,  std ≈ 0.55
  Layer 10: mean ≈ 0,  std ≈ 0.51 (stable!)
```

### 1.2 The Goal

We want activations to maintain roughly the same variance across layers:

```
Var(a^(l)) ≈ Var(a^(l-1))    for all layers l

This ensures:
- No vanishing activations (information flows forward)
- No exploding activations (numerical stability)
- No vanishing gradients (learning happens in all layers)
```

---

## 2. The Symmetry Breaking Problem

### 2.1 What Happens with Zero Initialization?

If all weights are initialized to zero (or any constant):

```
All neurons in the same layer compute the same output:
  z_1 = w·x + b = 0·x + 0 = 0
  z_2 = w·x + b = 0·x + 0 = 0
  z_3 = w·x + b = 0·x + 0 = 0

During backprop, all neurons receive the same gradient:
  ∂L/∂w_1 = ∂L/∂w_2 = ∂L/∂w_3

After update, all neurons still have the same weights!
```

**Consequence**: The neurons are permanently locked in sync. The network has the effective capacity of a single neuron per layer. No amount of training will fix this.

### 2.2 Why Random Initialization Works

Random initialization breaks symmetry: each neuron starts with different weights, computes different outputs, receives different gradients, and specializes differently.

```
Random init:
  w1 = [0.3, -0.1]    →  z1 = 0.3x1 - 0.1x2
  w2 = [-0.2, 0.4]    →  z2 = -0.2x1 + 0.4x2
  w3 = [0.1, 0.1]     →  z3 = 0.1x1 + 0.1x2

All different → different gradients → different specialization
```

---

## 3. Variance Analysis

### 3.1 Setup

For a single neuron with n inputs:

```
z = Σ_{i=1}^{n} w_i · x_i     (ignoring bias for analysis)
```

Assuming:
- w_i and x_i are independent
- E[w_i] = 0, E[x_i] = 0
- Var(w_i) = σ_w^2 (same for all i)
- Var(x_i) = σ_x^2 (same for all i)

### 3.2 Variance of z

```
Var(z) = Var(Σ w_i · x_i)
       = Σ Var(w_i · x_i)                 (independence)
       = Σ [E[w_i^2] · E[x_i^2]]          (independence, zero mean)
       = Σ [Var(w_i) · Var(x_i)]
       = n · σ_w^2 · σ_x^2
```

### 3.3 The Stability Condition

For activations to maintain variance across layers:

```
Var(a^(l)) = Var(a^(l-1))

This requires: n_l · σ_w^2 = 1
            →  σ_w^2 = 1/n_l

Where n_l = fan_in (number of inputs to each neuron in layer l)
```

---

## 4. Xavier/Glorot Initialization (2010)

### 4.1 Derivation

Xavier Glorot and Yoshua Bengio analyzed variance propagation in both forward and backward directions:

**Forward**: Var(z^(l)) = n_{l-1} · Var(w) · Var(a^(l-1))
→ Var(w) = 1/n_{l-1} (fan_in)

**Backward**: Var(δ^(l)) = n_l · Var(w) · Var(δ^(l+1))
→ Var(w) = 1/n_l (fan_out)

**Compromise** (satisfy both approximately):

```
Var(w) = 2 / (fan_in + fan_out)

Xavier Normal:   w ~ N(0, 2/(fan_in + fan_out))
Xavier Uniform:  w ~ U(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
```

### 4.2 Assumptions

Xavier initialization assumes:
- **Linear activation** (or sigmoid/tanh near zero where they are approximately linear)
- Zero-mean inputs
- Independent weights and activations

### 4.3 Implementation

```python
import numpy as np

def xavier_normal(fan_in, fan_out):
    """Xavier/Glorot normal initialization."""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_out, fan_in) * std

def xavier_uniform(fan_in, fan_out):
    """Xavier/Glorot uniform initialization."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_out, fan_in))

# Example: layer with 256 inputs, 128 outputs
W = xavier_normal(256, 128)
print(f"Shape: {W.shape}")
print(f"Mean: {W.mean():.6f}")
print(f"Std: {W.std():.6f}")
print(f"Expected std: {np.sqrt(2.0 / (256 + 128)):.6f}")
```

### 4.4 When to Use Xavier

- **Sigmoid activation**: Xavier works well
- **Tanh activation**: Xavier works well
- **ReLU activation**: Xavier is suboptimal (use He instead)

---

## 5. He/Kaiming Initialization (2015)

### 5.1 Why Xavier Fails for ReLU

ReLU zeros out half the activations (z < 0 → 0). This halves the variance:

```
For ReLU: E[ReLU(z)^2] = (1/2) · E[z^2]    (half the outputs are zero)

So: Var(a^(l)) = (1/2) · n_{l-1} · Var(w) · Var(a^(l-1))

For stability: (1/2) · n · Var(w) = 1
            →  Var(w) = 2/n
```

### 5.2 The Formula

```
He Normal:   w ~ N(0, 2/fan_in)
He Uniform:  w ~ U(-√(6/fan_in), √(6/fan_in))
```

Note the factor of 2 compared to Xavier -- compensating for ReLU's half-zeroing.

### 5.3 Implementation

```python
def he_normal(fan_in, fan_out):
    """He/Kaiming normal initialization (for ReLU)."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_out, fan_in) * std

def he_uniform(fan_in, fan_out):
    """He/Kaiming uniform initialization (for ReLU)."""
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, (fan_out, fan_in))

# Example
W = he_normal(256, 128)
print(f"He std: {W.std():.6f}")
print(f"Expected std: {np.sqrt(2.0 / 256):.6f}")
```

### 5.4 Variants for Different Activations

| Activation | Initialization | Variance |
|-----------|---------------|---------|
| Sigmoid, Tanh | Xavier | 2/(fan_in + fan_out) |
| ReLU | He (fan_in) | 2/fan_in |
| Leaky ReLU (α) | Modified He | 2/((1+α^2) · fan_in) |
| SELU | LeCun | 1/fan_in |

---

## 6. Empirical Verification

### 6.1 Tracking Activation Statistics

```python
def test_initialization(init_fn, n_layers=10, n_neurons=256, activation='relu'):
    """Track activation statistics through layers."""
    np.random.seed(42)
    x = np.random.randn(n_neurons, 1)  # single sample

    stats = []
    a = x
    for i in range(n_layers):
        W = init_fn(n_neurons, n_neurons)
        z = W @ a
        if activation == 'relu':
            a = np.maximum(0, z)
        elif activation == 'tanh':
            a = np.tanh(z)
        stats.append({'layer': i+1, 'mean': a.mean(), 'std': a.std()})
        
    for s in stats:
        bar = '█' * int(s['std'] * 50)
        print(f"Layer {s['layer']:2d}: mean={s['mean']:+.4f}, std={s['std']:.4f} |{bar}")

print("=== Random N(0,1) with ReLU ===")
test_initialization(lambda fi, fo: np.random.randn(fo, fi))

print("\n=== He Init with ReLU ===")
test_initialization(he_normal)

print("\n=== Xavier Init with Tanh ===")
test_initialization(xavier_normal, activation='tanh')
```

---

## 7. Special Initialization Strategies

### 7.1 Orthogonal Initialization

Initialize weight matrices as orthogonal matrices. Preserves gradient norms exactly in linear networks.

```python
def orthogonal_init(fan_in, fan_out, gain=1.0):
    """Orthogonal initialization."""
    shape = (fan_out, fan_in)
    flat = np.random.randn(max(shape), min(shape))
    q, r = np.linalg.qr(flat)
    q = q[:shape[0], :shape[1]]
    return gain * q
```

### 7.2 LSUV (Layer-Sequential Unit-Variance)

A data-driven approach: initialize each layer so that the output variance is exactly 1 using a mini-batch.

```
For each layer l:
  1. Initialize W^(l) with orthogonal init
  2. Forward pass a mini-batch
  3. Measure Var(a^(l))
  4. Scale: W^(l) ← W^(l) / √(Var(a^(l)))
  5. Repeat until Var(a^(l)) ≈ 1
```

### 7.3 Bias Initialization

- **Default**: Initialize biases to zero (b = 0)
- **ReLU hidden layers**: Some recommend small positive bias (b = 0.01) to prevent dead neurons at initialization
- **Output layer**: Initialize bias to the log-odds of the prior (for classification)

```python
# For imbalanced binary classification (e.g., 5% positive)
# Initialize output bias so that the initial prediction ≈ prior probability
prior = 0.05
bias_init = np.log(prior / (1 - prior))  # ≈ -2.94
```

---

## 8. Summary

```
Key Takeaways
═══════════════════════════════════════════════════════
1. Never initialize all weights to zero (symmetry problem)
2. Goal: Var(activation) ≈ constant across layers
3. Xavier: Var(w) = 2/(fan_in + fan_out) for sigmoid/tanh
4. He: Var(w) = 2/fan_in for ReLU
5. He accounts for ReLU zeroing half the activations
6. Verify initialization by tracking activation stats per layer
7. Orthogonal init preserves gradient norms exactly
8. Bias: usually zero, except for special cases
═══════════════════════════════════════════════════════
```

---

## Exercises

1. Build a 20-layer network and visualize activation distributions with random, Xavier, and He init
2. Derive the initialization variance for Leaky ReLU with α = 0.2
3. Implement LSUV initialization and compare with He init
4. Show experimentally that zero initialization leads to identical neurons

---

**Previous**: [Gradient Descent Variants](./07_Gradient_Descent_Variants.md) | **Next**: [Regularization](./09_Regularization.md)
