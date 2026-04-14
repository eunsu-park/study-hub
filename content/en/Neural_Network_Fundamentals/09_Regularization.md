# Regularization

**Previous**: [Weight Initialization](./08_Weight_Initialization.md) | **Next**: [Batch Normalization](./10_Batch_Normalization.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why neural networks are prone to overfitting
2. Implement L1 and L2 regularization and explain their geometric interpretation
3. Implement dropout from scratch and explain its theoretical basis
4. Apply early stopping using validation loss monitoring
5. Distinguish between regularization at the parameter, activation, and data levels
6. Combine multiple regularization techniques effectively
7. Tune regularization hyperparameters using validation performance
8. Explain the connection between regularization and Bayesian priors

---

Neural networks have millions (or billions) of parameters -- far more than the number of training examples. Left unchecked, they will memorize the training data rather than learning generalizable patterns. Regularization is the set of techniques that prevent this overfitting, constraining the model to be simpler and more robust. This lesson covers the most important regularization methods, from mathematical penalties to practical training tricks.

---

## 1. The Overfitting Problem

### 1.1 Underfitting vs. Overfitting

```
Underfitting                Good Fit                  Overfitting
(High Bias)                 (Balanced)                (High Variance)

y │                        y │                        y │
  │    ──────                │    ╱╲                    │ ╱╲  ╱╲
  │  ●  ●  ●                │  ●╱  ╲●                  │╱  ╲╱  ╲●
  │ ●                       │ ●     ╲                  │●        ╲
  │●                        │●       ●                 │          ●
  └──────────► x             └──────────► x             └──────────► x

Too simple to               Captures the               Memorizes noise,
capture pattern              true pattern               fails on new data
```

### 1.2 Why Neural Networks Overfit

- **Parameter count >> training samples**: A network with 1M parameters can memorize 1M distinct patterns
- **Universal approximation**: NNs can fit any function, including noise
- **Training too long**: The network keeps reducing training loss, eventually fitting noise

### 1.3 The Regularization Toolkit

```
┌──────────────────────────────────────────────────────────────┐
│ Level              │ Technique                                │
├────────────────────┼──────────────────────────────────────────┤
│ Parameter          │ L1 regularization, L2 regularization     │
│ Activation         │ Dropout, Activity regularization         │
│ Training process   │ Early stopping, Learning rate decay      │
│ Data               │ Data augmentation, Noise injection       │
│ Architecture       │ Smaller network, Weight sharing          │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. L2 Regularization (Weight Decay)

### 2.1 Formulation

Add a penalty proportional to the squared magnitude of weights:

```
J_reg = J_data + (λ/2) · Σ ||W^(l)||_F^2

Where:
  J_data = original loss (MSE or cross-entropy)
  ||W||_F = Frobenius norm = √(Σ w_ij^2)
  λ = regularization strength (hyperparameter)
```

### 2.2 Effect on Gradient

```
∂J_reg/∂W = ∂J_data/∂W + λ · W

Update rule:
  W ← W - η · (∂J_data/∂W + λ · W)
  W ← (1 - η·λ) · W - η · ∂J_data/∂W
       └────────┘
       "weight decay" — shrinks weights toward zero each step
```

### 2.3 Geometric Interpretation

```
               L2 constraint
               (circle)
            ╱──────────╲
          ╱    ●optimal   ╲
        │     ↙              │
        │   ◆ constrained    │
        │                    │
          ╲                ╱
            ╲──────────╱

L2 constrains weights to lie within a ball.
The solution is the point on the constraint boundary 
closest to the unconstrained optimum.
→ Weights shrink proportionally, large weights more penalized.
```

### 2.4 Implementation

```python
import numpy as np

def l2_regularization_loss(params, lambd):
    """Compute L2 regularization penalty."""
    reg = 0.0
    for W, b in params:
        reg += np.sum(W ** 2)
    return (lambd / 2) * reg

def l2_gradient(W, lambd):
    """L2 gradient: add λ·W to the weight gradient."""
    return lambd * W

# In training loop:
# dW += l2_gradient(W, lambd=0.01)
```

---

## 3. L1 Regularization (Lasso)

### 3.1 Formulation

```
J_reg = J_data + λ · Σ |W^(l)|

Gradient (subgradient):
  ∂J_reg/∂w = ∂J_data/∂w + λ · sign(w)
```

### 3.2 L1 vs. L2

```
              L1 constraint        L2 constraint
              (diamond)            (circle)
                 ╱╲
               ╱    ╲              ╱──────╲
             ╱   ◆    ╲         ╱    ◆      ╲
           ╱──────────────╲   │              │
             ╲        ╱         ╲          ╱
               ╲    ╱              ╲──────╱
                 ╲╱

L1: Solution often lies on an axis → sparse weights (some = exactly 0)
L2: Solution lies on the circle → small but non-zero weights
```

| Property | L1 | L2 |
|----------|----|----|
| Sparsity | Yes (feature selection) | No (all weights stay) |
| Gradient | Constant magnitude | Proportional to weight |
| Solution | Corner of diamond | Smooth on circle |
| Use case | Feature selection | General regularization |

### 3.3 Implementation

```python
def l1_regularization_loss(params, lambd):
    """Compute L1 regularization penalty."""
    reg = 0.0
    for W, b in params:
        reg += np.sum(np.abs(W))
    return lambd * reg

def l1_gradient(W, lambd):
    """L1 subgradient: λ · sign(W)."""
    return lambd * np.sign(W)
```

---

## 4. Dropout

### 4.1 The Idea

During training, randomly set a fraction p of neuron activations to zero at each forward pass:

```
Training (dropout rate p=0.5):

   a1 ──────► ×1 ──► a1 (kept)
   a2 ──────► ×0 ──► 0  (dropped)
   a3 ──────► ×1 ──► a3 (kept)
   a4 ──────► ×0 ──► 0  (dropped)
   a5 ──────► ×1 ──► a5 (kept)

Each forward pass uses a different random subset of neurons.
```

### 4.2 Why Dropout Works

1. **Ensemble effect**: Each training step trains a different "sub-network." The final model is an average of 2^n possible sub-networks.
2. **Redundancy**: Neurons cannot co-adapt (rely on specific other neurons), so each neuron must be independently useful.
3. **Noise injection**: Acts as a form of data augmentation in activation space.

### 4.3 Inverted Dropout (Standard Implementation)

During training, scale surviving activations by 1/(1-p) so that expected values match inference:

```
Training:
  mask = (random > p)              # binary mask
  a_dropped = a * mask / (1 - p)   # scale up to compensate

Inference:
  a_output = a                     # no dropout, no scaling needed
```

### 4.4 Implementation

```python
class DropoutLayer:
    """Inverted dropout for training."""

    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, a, training=True):
        if not training:
            return a
        self.mask = (np.random.rand(*a.shape) < self.keep_prob).astype(float)
        return a * self.mask / self.keep_prob

    def backward(self, da):
        """Gradient flows through kept neurons only."""
        return da * self.mask / self.keep_prob

# Usage
dropout = DropoutLayer(keep_prob=0.8)  # drop 20%

# Training
a = np.random.randn(128, 32)  # 128 neurons, batch of 32
a_dropped = dropout.forward(a, training=True)
print(f"Fraction zeroed: {(a_dropped == 0).mean():.2f}")  # ≈ 0.20

# Inference
a_infer = dropout.forward(a, training=False)
# No dropout applied; a_infer == a
```

### 4.5 Dropout Rates

| Layer Type | Typical Rate (p = drop probability) |
|-----------|--------------------------------------|
| Input layer | 0.0 - 0.2 (keep most inputs) |
| Hidden layers | 0.2 - 0.5 |
| Output layer | 0.0 (never dropout) |
| After BatchNorm | Often 0.0 (BN already regularizes) |

---

## 5. Early Stopping

### 5.1 The Concept

Monitor validation loss during training. Stop when it starts increasing:

```
Loss
  │
  │  ╲ train
  │   ╲
  │    ╲╲────────────────── train keeps decreasing
  │     ╲
  │      ╲  val
  │       ╲───── ← STOP HERE (best validation)
  │            ╲
  │              ╲  val starts increasing → overfitting
  └──────────────────────────────────► epoch
         best epoch
```

### 5.2 Implementation with Patience

```python
class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_params = None

    def check(self, val_loss, params):
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Deep copy best parameters
            self.best_params = [(W.copy(), b.copy()) for W, b in params]
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at epoch, best val_loss: {self.best_loss:.4f}")
                return True
            return False
```

---

## 6. Data Augmentation

### 6.1 The Idea

Artificially increase the size and diversity of the training set by applying transformations that preserve the label:

```
Original image:        Augmented versions:
┌──────┐              ┌──────┐  ┌──────┐  ┌──────┐
│  🐱  │   ──────►    │ 🐱   │  │  🐱  │  │🐱    │
│      │              │ flip  │  │rotate│  │ crop  │
└──────┘              └──────┘  └──────┘  └──────┘
```

### 6.2 Common Augmentations

| Domain | Techniques |
|--------|-----------|
| Images | Flip, rotate, crop, color jitter, cutout |
| Text | Synonym replacement, back-translation |
| Tabular | Feature noise, SMOTE |
| Audio | Time stretch, pitch shift, noise injection |

### 6.3 For Tabular Data / MLPs

```python
def add_gaussian_noise(X, std=0.01):
    """Add Gaussian noise for regularization."""
    noise = np.random.randn(*X.shape) * std
    return X + noise

# During training:
# X_batch_noisy = add_gaussian_noise(X_batch, std=0.05)
```

---

## 7. Combining Regularization Techniques

### 7.1 Common Combinations

```
Modern Practice:
  ✓ He initialization
  ✓ Batch normalization
  ✓ L2 regularization (weight decay, λ = 1e-4 to 1e-2)
  ✓ Dropout (p = 0.1 to 0.3, after BN)
  ✓ Data augmentation
  ✓ Early stopping with patience

Not Recommended Together:
  ✗ Heavy dropout + heavy L2 (over-regularization)
  ✗ Dropout + Batch Normalization in some architectures (interaction effects)
```

### 7.2 Tuning Regularization

```
Start with no regularization → establish baseline
  │
  ├── Overfitting? (val loss >> train loss)
  │     ├── Add L2 (λ = 1e-4), increase gradually
  │     ├── Add dropout (p = 0.1), increase gradually
  │     ├── Add data augmentation
  │     └── Reduce model size (fewer layers/neurons)
  │
  └── Underfitting? (train loss still high)
        ├── Reduce regularization
        ├── Increase model size
        └── Train longer
```

---

## 8. Bayesian Interpretation

### 8.1 L2 = Gaussian Prior

L2 regularization is equivalent to placing a Gaussian prior on the weights:

```
P(w) = N(0, 1/λ)

MAP estimation: maximize P(w|data) = P(data|w) · P(w)
             = minimize -log P(data|w) - log P(w)
             = minimize [loss + (λ/2)||w||^2]
```

### 8.2 L1 = Laplace Prior

L1 regularization corresponds to a Laplace (double exponential) prior:

```
P(w) = Laplace(0, 1/λ)

The sharp peak at zero encourages sparsity (weights become exactly zero).
```

---

## 9. Summary

```
Key Takeaways
═══════════════════════════════════════════════════════
1. Overfitting: network memorizes noise instead of patterns
2. L2 (weight decay): shrinks weights, prevents large values
3. L1 (lasso): promotes sparsity, drives some weights to zero
4. Dropout: randomly zeros neurons, creates ensemble effect
5. Early stopping: halt when validation loss stops improving
6. Data augmentation: increase training data diversity
7. Combine techniques carefully; tune via validation performance
8. Regularization = implicit Bayesian prior on weights
═══════════════════════════════════════════════════════
```

---

## Exercises

1. Train an MLP with and without L2 regularization and compare overfitting
2. Implement dropout and verify that expected activations match at train/test time
3. Implement early stopping with patience and plot train/val loss curves
4. Compare the sparsity of L1 vs. L2 regularized weights

---

**Previous**: [Weight Initialization](./08_Weight_Initialization.md) | **Next**: [Batch Normalization](./10_Batch_Normalization.md)
