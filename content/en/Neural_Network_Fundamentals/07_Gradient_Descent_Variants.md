# Gradient Descent Variants

**Previous**: [Backpropagation](./06_Backpropagation.md) | **Next**: [Weight Initialization](./08_Weight_Initialization.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the differences between batch, mini-batch, and stochastic gradient descent
2. Implement SGD with momentum from scratch
3. Derive and implement the Adam optimizer
4. Compare RMSProp, AdaGrad, and Adam mathematically
5. Implement learning rate scheduling strategies
6. Explain why adaptive learning rates improve training
7. Diagnose common training failures from loss curves
8. Select an appropriate optimizer for a given problem

---

Backpropagation gives us the gradient -- the direction to adjust weights to reduce the loss. But *how* we use that gradient matters enormously. Vanilla gradient descent is slow and fragile. Modern optimizers like Adam use momentum and adaptive learning rates to converge faster and more reliably. This lesson covers the family of gradient descent algorithms, from the simplest to the most widely used.

---

## 1. Vanilla Gradient Descent

### 1.1 The Update Rule

```
θ ← θ - η · ∂J/∂θ

Where:
  θ = parameters (weights and biases)
  η = learning rate (step size)
  ∂J/∂θ = gradient of cost over entire dataset
```

### 1.2 Three Variants

```
┌─────────────────────────────────────────────────────────────┐
│ Variant           │ Batch Size    │ Gradient Estimate       │
├───────────────────┼───────────────┼─────────────────────────┤
│ Batch GD          │ N (all data)  │ Exact (low variance)    │
│ Stochastic GD     │ 1 sample      │ Noisy (high variance)   │
│ Mini-batch GD     │ B (e.g., 32)  │ Balanced                │
└─────────────────────────────────────────────────────────────┘
```

**Batch Gradient Descent**: Uses the entire dataset for each update.
```
θ ← θ - η · (1/N) Σ_{i=1}^{N} ∇L(x_i, y_i; θ)
```
- Exact gradient, stable convergence
- Very slow for large datasets
- Gets stuck in sharp local minima

**Stochastic Gradient Descent (SGD)**: Uses one sample per update.
```
θ ← θ - η · ∇L(x_i, y_i; θ)
```
- Very noisy gradient → can escape local minima
- Cannot exploit GPU parallelism
- Convergence is erratic

**Mini-batch GD**: Uses a batch of B samples per update (standard practice).
```
θ ← θ - η · (1/B) Σ_{j=1}^{B} ∇L(x_j, y_j; θ)
```
- Best of both worlds: stable enough, fast enough
- B = 32, 64, 128, or 256 are common choices
- Fits GPU memory blocks efficiently

### 1.3 Implementation

```python
import numpy as np

def sgd_update(params, grads, learning_rate):
    """Vanilla SGD update."""
    updated = []
    for (W, b), (dW, db) in zip(params, grads):
        W = W - learning_rate * dW
        b = b - learning_rate * db
        updated.append((W, b))
    return updated
```

---

## 2. SGD with Momentum

### 2.1 The Problem with Vanilla SGD

```
Loss landscape with elongated contours:

    │  ╲         ╱
    │   ╲  ↗  ╱    ← SGD oscillates across the narrow dimension
    │    ╲  ╱       while making slow progress along the long axis
    │     ◆
    │    ╱  ╲
    │   ╱  ↙  ╲
    └──────────────

Momentum smooths out oscillations and accelerates along consistent directions.
```

### 2.2 The Algorithm

Momentum introduces a velocity term v that accumulates past gradients:

```
v ← β · v + (1 - β) · ∇J(θ)     (or v ← β · v + ∇J(θ) in some formulations)
θ ← θ - η · v

Where β ∈ [0, 1) is the momentum coefficient (typically β = 0.9)
```

**Intuition**: A ball rolling downhill. The velocity accumulates in consistent directions and cancels out in oscillating directions.

```
Without momentum:     ↗ ↙ ↗ ↙ ↗  (zigzag)
With momentum:        → → → → →  (smooth acceleration)
```

### 2.3 Exponential Moving Average Perspective

The velocity is an exponential moving average of past gradients:

```
v_t = β · v_{t-1} + (1-β) · g_t
    = (1-β) · g_t + β(1-β) · g_{t-1} + β^2(1-β) · g_{t-2} + ...

Effective window: ~1/(1-β) steps
β = 0.9 → averages over ~10 recent gradients
β = 0.99 → averages over ~100 recent gradients
```

### 2.4 Implementation

```python
def sgd_momentum_update(params, grads, velocities, learning_rate, beta=0.9):
    """SGD with momentum."""
    updated_params = []
    updated_velocities = []
    for (W, b), (dW, db), (vW, vb) in zip(params, grads, velocities):
        vW = beta * vW + (1 - beta) * dW
        vb = beta * vb + (1 - beta) * db
        W = W - learning_rate * vW
        b = b - learning_rate * vb
        updated_params.append((W, b))
        updated_velocities.append((vW, vb))
    return updated_params, updated_velocities
```

---

## 3. Nesterov Accelerated Gradient (NAG)

### 3.1 Look-Ahead Idea

Nesterov momentum computes the gradient at the "look-ahead" position:

```
Standard momentum:
  v ← β · v + ∇J(θ)
  θ ← θ - η · v

Nesterov momentum:
  v ← β · v + ∇J(θ - η · β · v)    ← gradient at the look-ahead position
  θ ← θ - η · v
```

**Intuition**: Before rolling further, peek ahead to see if you're about to overshoot. If the gradient at the look-ahead position points backward, slow down.

### 3.2 Why It Helps

```
Standard momentum overshoots:     ──────→ ──→ ←── → (oscillates past minimum)
Nesterov corrects early:          ──────→ ──→ → (smooth deceleration)
```

---

## 4. AdaGrad (Adaptive Gradient)

### 4.1 Per-Parameter Learning Rate

AdaGrad adapts the learning rate for each parameter based on historical gradient magnitudes:

```
s ← s + (∇J)^2                    (accumulate squared gradients)
θ ← θ - η / (√s + ε) · ∇J        (scale learning rate by inverse sqrt)

Where ε ≈ 1e-8 prevents division by zero.
```

### 4.2 Effect

- Parameters with **large past gradients** → small learning rate (careful steps)
- Parameters with **small past gradients** → large learning rate (bigger exploration)

### 4.3 Problem

```
s grows monotonically → learning rate shrinks to near zero over time
→ Training effectively stops prematurely
```

---

## 5. RMSProp (Root Mean Square Propagation)

### 5.1 Fixing AdaGrad's Decay

RMSProp uses an exponential moving average instead of cumulative sum:

```
s ← β · s + (1 - β) · (∇J)^2       (exponential moving average of squared gradients)
θ ← θ - η / (√s + ε) · ∇J

Typically β = 0.999, ε = 1e-8
```

### 5.2 Why It Works

```
AdaGrad:    s_t = Σ_{i=1}^{t} g_i^2        (grows forever)
RMSProp:    s_t = β·s_{t-1} + (1-β)·g_t^2   (forgets old gradients)
```

The denominator √s adapts to the recent gradient scale, not the entire history.

### 5.3 Implementation

```python
def rmsprop_update(params, grads, sq_grads, learning_rate, beta=0.999, eps=1e-8):
    """RMSProp optimizer."""
    updated_params = []
    updated_sq_grads = []
    for (W, b), (dW, db), (sW, sb) in zip(params, grads, sq_grads):
        sW = beta * sW + (1 - beta) * dW**2
        sb = beta * sb + (1 - beta) * db**2
        W = W - learning_rate * dW / (np.sqrt(sW) + eps)
        b = b - learning_rate * db / (np.sqrt(sb) + eps)
        updated_params.append((W, b))
        updated_sq_grads.append((sW, sb))
    return updated_params, updated_sq_grads
```

---

## 6. Adam (Adaptive Moment Estimation)

### 6.1 Combining Momentum + RMSProp

Adam combines the best of momentum (first moment) and RMSProp (second moment):

```
First moment (mean):     m ← β1 · m + (1 - β1) · g        (momentum)
Second moment (variance): v ← β2 · v + (1 - β2) · g^2     (RMSProp)

Bias correction:
  m̂ = m / (1 - β1^t)
  v̂ = v / (1 - β2^t)

Update:
  θ ← θ - η · m̂ / (√v̂ + ε)

Defaults: β1 = 0.9, β2 = 0.999, ε = 1e-8, η = 0.001
```

### 6.2 Why Bias Correction?

At the start (t=1), m and v are initialized to 0. Without correction, the estimates are biased toward zero:

```
t=1:  m = (1 - β1) · g_1 = 0.1 · g_1    (biased low by 10×!)
      m̂ = m / (1 - 0.9^1) = g_1          (unbiased)

As t → ∞: (1 - β^t) → 1, correction vanishes
```

### 6.3 Implementation

```python
class Adam:
    """Adam optimizer."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None  # first moment
        self.v = None  # second moment

    def initialize(self, params):
        self.m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
        self.v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]

    def update(self, params, grads):
        if self.m is None:
            self.initialize(params)

        self.t += 1
        updated = []

        for i, ((W, b), (dW, db)) in enumerate(zip(params, grads)):
            # Update first moment
            mW, mb = self.m[i]
            mW = self.beta1 * mW + (1 - self.beta1) * dW
            mb = self.beta1 * mb + (1 - self.beta1) * db

            # Update second moment
            vW, vb = self.v[i]
            vW = self.beta2 * vW + (1 - self.beta2) * dW**2
            vb = self.beta2 * vb + (1 - self.beta2) * db**2

            self.m[i] = (mW, mb)
            self.v[i] = (vW, vb)

            # Bias correction
            mW_hat = mW / (1 - self.beta1**self.t)
            mb_hat = mb / (1 - self.beta1**self.t)
            vW_hat = vW / (1 - self.beta2**self.t)
            vb_hat = vb / (1 - self.beta2**self.t)

            # Update parameters
            W = W - self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            b = b - self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)
            updated.append((W, b))

        return updated
```

---

## 7. Optimizer Comparison

```
                Momentum              Adam
Loss ┤\         Loss ┤\
     │ \             │ \
     │  ╲            │  ╲──────────
     │   ╲           │
     │    ╲╲         │
     │      ╲╲       │
     │        ╲──    └──────────── epochs
     └──────────── epochs

Momentum: smooth but can be slow    Adam: fast and adaptive
```

| Optimizer | Learning Rate | Momentum | Adaptive | Bias Correction |
|-----------|:-------------|:---------|:---------|:---------------|
| SGD | Global η | No | No | No |
| SGD+Momentum | Global η | Yes (β) | No | No |
| NAG | Global η | Yes (look-ahead) | No | No |
| AdaGrad | Per-parameter | No | Yes (cumulative) | No |
| RMSProp | Per-parameter | No | Yes (EMA) | No |
| Adam | Per-parameter | Yes (β1) | Yes (β2) | Yes |
| AdamW | Per-parameter | Yes | Yes | Yes + weight decay |

---

## 8. Learning Rate Scheduling

### 8.1 Why Schedule?

A fixed learning rate is suboptimal:
- **Too large early**: Loss oscillates or diverges
- **Too large late**: Cannot settle into a precise minimum
- **Too small early**: Converges very slowly

Solution: Start with a large learning rate, then decrease it.

### 8.2 Common Schedules

**Step Decay**:
```
η(t) = η_0 · γ^(floor(t / step_size))

Example: η_0 = 0.1, γ = 0.1, step = 30
  Epoch 0-29:  η = 0.1
  Epoch 30-59: η = 0.01
  Epoch 60-89: η = 0.001
```

**Exponential Decay**:
```
η(t) = η_0 · e^(-λt)
```

**Cosine Annealing**:
```
η(t) = η_min + (η_max - η_min) / 2 · (1 + cos(πt / T))

η_max ┤╲
      │  ╲
      │    ╲
      │      ╲
η_min ┤        ╲___
      └────┬────┬──► t
           0    T
```

**Warmup + Cosine**:
```
η(t) = { η_max · t / T_warmup              if t < T_warmup
       { η_min + (η_max - η_min)/2 · (1 + cos(π(t-T_warmup)/(T-T_warmup)))  otherwise

        η_max ┤    ╱╲
              │   ╱   ╲
              │  ╱      ╲
              │ ╱         ╲
        η_min ┤╱            ╲___
              └──┬────────┬──► t
              T_warmup    T
```

### 8.3 Implementation

```python
def step_decay(epoch, lr_init=0.1, gamma=0.1, step_size=30):
    return lr_init * gamma ** (epoch // step_size)

def cosine_annealing(epoch, lr_max=0.1, lr_min=1e-6, T=100):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T))

def warmup_cosine(epoch, lr_max=0.001, lr_min=1e-6, warmup=10, total=100):
    if epoch < warmup:
        return lr_max * epoch / warmup
    progress = (epoch - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))
```

---

## 9. Diagnosing Training from Loss Curves

```
Good training:          Overfitting:           Too high LR:
Loss ┤\                 Loss ┤\   train        Loss ┤ /\/\/\
     │ \                     │ \    ↘               │/      \
     │  \                    │  ╲───              │        \/
     │   ╲───               │    ↗ val           │
     │       ──             │   ↗                │
     └──────── epoch        └──────── epoch      └──────── epoch

Too low LR:             Learning rate decay needed:
Loss ┤\                 Loss ┤\
     │ \                     │ ╲
     │  \                    │  ╲──── plateau
     │   \                   │
     │    \                  │
     │     \                 │
     └──────── epoch        └──────── epoch
```

---

## 10. Summary

```
Key Takeaways
═══════════════════════════════════════════════════════
1. Mini-batch SGD is the standard (batch size 32-256)
2. Momentum smooths oscillations (β = 0.9)
3. RMSProp adapts per-parameter learning rates
4. Adam = Momentum + RMSProp + bias correction
5. Adam defaults: η=0.001, β1=0.9, β2=0.999
6. Learning rate scheduling: start high, decay over time
7. Warmup + cosine annealing is state-of-the-art
8. Diagnose issues from loss curve shapes
═══════════════════════════════════════════════════════
```

---

## Exercises

1. Implement SGD, SGD+Momentum, and Adam, and compare convergence on a quadratic function
2. Visualize the optimization trajectories of different optimizers on the Rosenbrock function
3. Implement cosine annealing with warm restarts
4. Train an MLP with different learning rates and plot the loss curves

---

**Previous**: [Backpropagation](./06_Backpropagation.md) | **Next**: [Weight Initialization](./08_Weight_Initialization.md)
