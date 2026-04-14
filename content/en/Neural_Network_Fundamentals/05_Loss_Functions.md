# Loss Functions

**Previous**: [Feedforward Networks](./04_Feedforward_Networks.md) | **Next**: [Backpropagation](./06_Backpropagation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the role of loss functions in neural network training
2. Derive and implement Mean Squared Error (MSE) and its gradient
3. Derive and implement Binary and Categorical Cross-Entropy losses
4. Explain why cross-entropy is preferred over MSE for classification
5. Implement Hinge loss for SVM-style neural networks
6. Apply the numerically stable log-sum-exp trick
7. Choose the appropriate loss function for a given task
8. Compute the gradient of each loss with respect to the network output

---

A loss function (also called a cost function or objective function) measures how far the network's predictions are from the true values. Training a neural network is fundamentally an optimization problem: minimize the loss by adjusting the weights. The choice of loss function determines what the network learns to optimize and has a profound impact on training dynamics.

---

## 1. Loss Function Basics

### 1.1 Loss vs. Cost

```
Loss:   L(ŷ, y)          ← for a single sample
Cost:   J = (1/N) Σ L    ← average over all N samples

We minimize the cost J with respect to weights W and biases b.
```

### 1.2 What Makes a Good Loss Function?

1. **Differentiable**: We need gradients for backpropagation
2. **Convex (ideally)**: Easier optimization, single global minimum
3. **Consistent with the task**: Classification needs different loss than regression
4. **Informative gradients**: Large error → large gradient → fast learning

---

## 2. Mean Squared Error (MSE)

### 2.1 Definition

```
L_MSE(ŷ, y) = (1/2)(ŷ - y)^2              (single sample)

J_MSE = (1/2N) Σ_{i=1}^{N} (ŷ_i - y_i)^2  (cost over N samples)
```

The 1/2 is a convention that simplifies the derivative.

### 2.2 Gradient

```
∂L/∂ŷ = ŷ - y

When ŷ is far from y → large gradient → fast update
When ŷ is close to y → small gradient → fine-tuning
```

### 2.3 For Multi-dimensional Output

For output vector ŷ ∈ R^K:

```
L_MSE = (1/2) Σ_{k=1}^{K} (ŷ_k - y_k)^2 = (1/2)||ŷ - y||^2

∂L/∂ŷ_k = ŷ_k - y_k    (gradient per output dimension)
```

### 2.4 Implementation

```python
import numpy as np

def mse_loss(y_pred, y_true):
    """Mean Squared Error loss."""
    return 0.5 * np.mean((y_pred - y_true) ** 2)

def mse_gradient(y_pred, y_true):
    """Gradient of MSE with respect to y_pred."""
    return (y_pred - y_true) / y_true.shape[0]

# Example
y_true = np.array([1.0, 0.0, 1.0])
y_pred = np.array([0.8, 0.2, 0.9])
print(f"MSE Loss: {mse_loss(y_pred, y_true):.4f}")
print(f"Gradient: {mse_gradient(y_pred, y_true)}")
```

### 2.5 When to Use MSE

- **Regression tasks**: Predicting continuous values (price, temperature, etc.)
- **Autoencoders**: Reconstruction loss
- **NOT recommended for classification** (see Section 4)

---

## 3. Cross-Entropy Loss

### 3.1 Binary Cross-Entropy (BCE)

For binary classification with output ŷ ∈ (0, 1) from a sigmoid:

```
L_BCE = -[y · log(ŷ) + (1 - y) · log(1 - ŷ)]

When y = 1:  L = -log(ŷ)       ← penalizes ŷ close to 0
When y = 0:  L = -log(1 - ŷ)   ← penalizes ŷ close to 1

         Loss
         │
    5.0  ┤\
         │ \
    3.0  ┤  \
         │   \
    1.0  ┤    \___
         │        \_______
    0.0  ┤                ─────
         └──┬──┬──┬──┬──┬──► ŷ
            0  0.2 0.4 0.6 0.8 1.0
         
         L = -log(ŷ) when y=1
```

### 3.2 BCE Gradient

```
∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ) = (ŷ - y) / (ŷ(1 - ŷ))
```

When combined with sigmoid output z → ŷ = σ(z):

```
∂L/∂z = ŷ - y    ← Beautifully simple! No σ' term!
```

This is why sigmoid + BCE is such a natural pairing.

### 3.3 Categorical Cross-Entropy (CCE)

For multi-class classification with K classes, using one-hot encoded y and softmax output ŷ:

```
L_CCE = -Σ_{k=1}^{K} y_k · log(ŷ_k)

Since y is one-hot (only one y_k = 1, rest are 0):
L_CCE = -log(ŷ_c)    where c is the true class

∂L/∂ŷ_k = -y_k / ŷ_k
```

When combined with softmax:

```
∂L/∂z_k = ŷ_k - y_k    ← Same elegant simplification!
```

### 3.4 Implementation

```python
def binary_cross_entropy(y_pred, y_true, eps=1e-15):
    """Binary cross-entropy loss with numerical stability."""
    y_pred = np.clip(y_pred, eps, 1 - eps)  # prevent log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_pred, y_true, eps=1e-15):
    """Categorical cross-entropy loss.
    
    Args:
        y_pred: Softmax probabilities, shape (K, N) or (K,)
        y_true: One-hot encoded labels, same shape as y_pred
    """
    y_pred = np.clip(y_pred, eps, 1.0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

def cross_entropy_gradient_with_softmax(y_pred, y_true):
    """Gradient of CCE+Softmax combined: simply (ŷ - y)."""
    return y_pred - y_true

# Binary example
y_true_b = np.array([1, 0, 1, 1])
y_pred_b = np.array([0.9, 0.1, 0.8, 0.7])
print(f"BCE: {binary_cross_entropy(y_pred_b, y_true_b):.4f}")

# Multi-class example (3 classes, 1 sample)
y_true_m = np.array([0, 1, 0])  # class 1
y_pred_m = np.array([0.1, 0.7, 0.2])  # softmax output
print(f"CCE: {categorical_cross_entropy(y_pred_m, y_true_m):.4f}")
```

---

## 4. Why Cross-Entropy Beats MSE for Classification

### 4.1 The Gradient Problem

Consider a sigmoid output neuron with MSE loss:

```
L_MSE = (1/2)(σ(z) - y)^2
∂L/∂z = (σ(z) - y) · σ'(z)
                       ↑
                σ'(z) ≤ 0.25 → gradient is suppressed!
```

With cross-entropy:

```
L_BCE = -[y·log(σ(z)) + (1-y)·log(1-σ(z))]
∂L/∂z = σ(z) - y
                 ↑
          No σ'(z) term → gradient scales with error!
```

### 4.2 Learning Speed Comparison

```
Suppose y = 1, z = -5 (very wrong prediction):

MSE gradient:   (σ(-5) - 1) · σ'(-5) = (-0.993) × 0.0066 = -0.0066
                                         ↑ large error   ↑ tiny σ'

BCE gradient:   σ(-5) - 1 = -0.993
                              ↑ large gradient proportional to error

BCE gradient is 150× larger → learns 150× faster from large errors!
```

### 4.3 Information-Theoretic Justification

Cross-entropy has a deep connection to information theory:

```
H(p, q) = -Σ p(x) · log(q(x))

- p = true distribution (one-hot label)
- q = predicted distribution (softmax output)

Minimizing cross-entropy = minimizing KL divergence = making q close to p
```

---

## 5. Hinge Loss

### 5.1 Definition

Used in SVM-style classifiers with labels y ∈ {-1, +1}:

```
L_hinge = max(0, 1 - y · ŷ)

ŷ > 1 and y = +1  → L = 0  (correct and confident)
ŷ = 0.5 and y = +1 → L = 0.5  (correct but not confident enough)
ŷ = -1 and y = +1  → L = 2  (wrong)
```

### 5.2 Gradient

```
∂L/∂ŷ = { -y   if 1 - y·ŷ > 0
         { 0    otherwise
```

### 5.3 Implementation

```python
def hinge_loss(y_pred, y_true):
    """Hinge loss for binary classification (y ∈ {-1, +1})."""
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

def hinge_gradient(y_pred, y_true):
    """Gradient of hinge loss."""
    mask = (1 - y_true * y_pred) > 0
    return -mask * y_true / len(y_true)

# Example
y_true_h = np.array([1, -1, 1, -1])
y_pred_h = np.array([0.5, -0.8, 1.5, 0.3])
print(f"Hinge loss: {hinge_loss(y_pred_h, y_true_h):.4f}")
```

---

## 6. Numerical Stability

### 6.1 The Log-Sum-Exp Trick

Computing log(Σ e^z_i) directly can overflow when z values are large:

```
Naive:      log(e^1000 + e^1001) → overflow!

Stable:     M = max(z)
            log(Σ e^z_i) = M + log(Σ e^(z_i - M))
            = 1001 + log(e^(-1) + e^0)
            = 1001 + log(1.368) = 1001.31
```

```python
def log_sum_exp(z):
    """Numerically stable log-sum-exp."""
    z_max = np.max(z)
    return z_max + np.log(np.sum(np.exp(z - z_max)))

# Compare
z = np.array([1000, 1001, 999])
# np.log(np.sum(np.exp(z)))  # overflow!
print(f"Stable log-sum-exp: {log_sum_exp(z):.4f}")
```

### 6.2 Stable Cross-Entropy

Always clip predictions before taking log:

```python
def stable_bce(y_pred, y_true, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

---

## 7. Loss Function Selection Guide

```
What is your task?
    │
    ├── Regression (continuous output)
    │     ├── Normal errors → MSE
    │     ├── Outlier-robust → MAE (L1) or Huber
    │     └── Probabilistic → Negative log-likelihood
    │
    ├── Binary Classification
    │     ├── Probabilistic output → Binary Cross-Entropy + Sigmoid
    │     └── Margin-based → Hinge Loss
    │
    └── Multi-class Classification (K classes)
          ├── Single label → Categorical Cross-Entropy + Softmax
          └── Multi-label → Binary Cross-Entropy per class + Sigmoid
```

### Quick Reference

| Task | Loss | Output Activation | Output Shape |
|------|------|--------------------|-------------|
| Regression | MSE | Identity | (1,) |
| Binary classification | BCE | Sigmoid | (1,) |
| Multi-class (K) | CCE | Softmax | (K,) |
| Multi-label | BCE (per class) | Sigmoid | (K,) |
| Ranking/Margin | Hinge | Identity | (1,) |

---

## 8. Putting It All Together

```python
# Complete example: loss computation for a 3-class classifier
np.random.seed(42)

# Simulate network output (logits) and true labels
logits = np.array([2.1, 0.5, -1.2])   # raw network output
y_true = np.array([1, 0, 0])           # one-hot: class 0

# Apply softmax
def softmax(z):
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

y_pred = softmax(logits)
print(f"Logits:     {logits}")
print(f"Softmax:    {y_pred}")
print(f"True label: {y_true}")

# Cross-entropy loss
loss = -np.sum(y_true * np.log(y_pred + 1e-15))
print(f"CCE Loss:   {loss:.4f}")

# Gradient (softmax + CCE combined)
grad = y_pred - y_true
print(f"Gradient:   {grad}")
# grad[0] is negative (push logit[0] up, it's the true class)
# grad[1], grad[2] are positive (push logits down)
```

---

## 9. Summary

```
Key Takeaways
═══════════════════════════════════════════════════════
1. Loss function measures prediction error; training minimizes it
2. MSE: good for regression, bad for classification (gradient issue)
3. BCE: pairs with sigmoid for binary classification
4. CCE: pairs with softmax for multi-class classification
5. Sigmoid+BCE and Softmax+CCE both give gradient = ŷ - y
6. Hinge loss: margin-based, for SVM-style classifiers
7. Always use numerical stability tricks (clip, log-sum-exp)
8. Loss choice depends on task type and output activation
═══════════════════════════════════════════════════════
```

---

## Exercises

1. Prove that ∂L_BCE/∂z = σ(z) - y when ŷ = σ(z)
2. Implement Huber loss and compare its behavior to MSE with outliers
3. Plot the loss surface of BCE as a function of ŷ for y=0 and y=1
4. Implement CCE for a batch of 32 samples with 5 classes

---

**Previous**: [Feedforward Networks](./04_Feedforward_Networks.md) | **Next**: [Backpropagation](./06_Backpropagation.md)
