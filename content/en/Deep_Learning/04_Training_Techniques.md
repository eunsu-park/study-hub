# 04. Training Techniques

[Previous: Backpropagation](./03_Backpropagation.md) | [Next: Linear & Logistic Regression](./05_Impl_Linear_Logistic.md)

---

## Learning Objectives

- Understand gradient descent variants (SGD, Momentum, Adam)
- Learn learning rate scheduling
- Learn regularization techniques (Dropout, Weight Decay, Batch Norm)
- Learn overfitting prevention and early stopping

---

## 1. Gradient Descent

### Basic Principle

```
W(t+1) = W(t) - η × ∇L
```
- η: learning rate
- ∇L: gradient of loss function

### Variants

| Method | Formula | Characteristics |
|--------|---------|-----------------|
| SGD | W -= lr × g | Simple, slow |
| Momentum | v = βv + g; W -= lr × v | Adds inertia |
| AdaGrad | Adaptive learning rate | Good for sparse data |
| RMSprop | Exponential moving average | Improved AdaGrad |
| Adam | Momentum + RMSprop | Most commonly used |

---

## 2. Momentum

Think of a ball rolling down a hilly landscape. Without momentum, the ball moves only based on the local slope — it can get stuck in small dips or oscillate in narrow valleys. With momentum, the ball accumulates velocity from past gradients. This lets it roll through small local minima and dampens oscillations in directions where the gradient keeps changing sign.

Adds inertia to reduce oscillations.

```
v(t) = β × v(t-1) + ∇L
W(t+1) = W(t) - η × v(t)
```

### NumPy Implementation

```python
def sgd_momentum(W, grad, v, lr=0.01, beta=0.9):
    # Why exponential moving average?  v accumulates past gradients with
    # exponential decay (beta=0.9 means ~10-step memory).  This smooths out
    # noisy per-sample gradients and builds up speed in consistent directions.
    v = beta * v + grad          # Update velocity
    W = W - lr * v               # Update weights using smoothed direction
    return W, v
```

### PyTorch

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

## 3. Adam Optimizer

Combines advantages of Momentum and RMSprop.

```
m(t) = β₁ × m(t-1) + (1-β₁) × g      # 1st moment (mean of gradients)
v(t) = β₂ × v(t-1) + (1-β₂) × g²     # 2nd moment (mean of squared gradients)
m̂ = m / (1 - β₁ᵗ)                    # Bias correction
v̂ = v / (1 - β₂ᵗ)
W = W - η × m̂ / (√v̂ + ε)
```

**Why bias correction?** Both `m` and `v` are initialized to 0. At step `t=1` with default `β₁=0.9`: `m₁ = 0.9 × 0 + 0.1 × g₁ = 0.1 × g₁`. This underestimates the true gradient mean by a factor of 10. Dividing by `(1 - β₁^t) = (1 - 0.9^1) = 0.1` corrects this: `m̂₁ = 0.1 × g₁ / 0.1 = g₁`. Without this correction, the first few updates would be far too small. The bias vanishes as `t` grows (since `β₁^t → 0`).

### NumPy Implementation

```python
def adam(W, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    # Why first moment?  m tracks the exponential moving average of gradients —
    # this is the "momentum" component, smoothing noisy gradient estimates.
    m = beta1 * m + (1 - beta1) * grad
    # Why second moment?  v tracks the EMA of squared gradients — this estimates
    # per-parameter gradient variance, giving each weight its own adaptive lr.
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # Why bias correction?  m and v are initialized to 0, so early estimates
    # are biased toward zero.  Dividing by (1 - beta^t) exactly cancels this.
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Why divide by sqrt(v_hat)?  Parameters with large gradient variance get
    # smaller effective lr (cautious updates), while stable gradients get larger
    # lr (confident updates).  eps prevents division by zero.
    W = W - lr * m_hat / (np.sqrt(v_hat) + eps)
    return W, m, v
```

### PyTorch

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 4. Learning Rate Scheduling

Why not use a fixed learning rate? A high learning rate enables fast initial progress — the model quickly moves out of bad regions of the loss landscape — but oscillates and overshoots near the optimum. A low learning rate converges smoothly but can be painfully slow in early epochs. Learning rate scheduling gives the best of both worlds: start high for speed, decay over time for precision.

Adjust learning rate during training.

### Main Methods

| Method | Characteristics |
|--------|----------------|
| Step Decay | Reduce by γ every N epochs |
| Exponential | lr = lr₀ × γᵉᵖᵒᶜʰ |
| Cosine Annealing | Reduce following cosine function |
| ReduceLROnPlateau | Reduce when validation loss plateaus |
| Warmup | Gradual increase at beginning |

### PyTorch Examples

```python
# Step Decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine Annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5
)

# In training loop
for epoch in range(epochs):
    train(...)
    scheduler.step()  # Call at end of epoch
```

---

## 5. Dropout

Randomly deactivates neurons during training.

### Principle

```
Training: y = x × mask / (1 - p)   # mask is Bernoulli(1-p)
Inference: y = x                   # No mask
```

### NumPy Implementation

```python
def dropout(x, p=0.5, training=True):
    if not training:
        return x
    mask = (np.random.rand(*x.shape) > p).astype(float)
    return x * mask / (1 - p)
```

### PyTorch

```python
class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Active only during training
        x = self.fc2(x)
        return x

# During inference
model.eval()  # Disable dropout
```

---

## 6. Batch Normalization

Normalizes inputs at each layer.

### Formula

```
μ = mean(x)
σ² = var(x)
x̂ = (x - μ) / √(σ² + ε)
y = γ × x̂ + β   # Learnable parameters
```

### NumPy Implementation

```python
def batch_norm(x, gamma, beta, eps=1e-5, training=True,
               running_mean=None, running_var=None, momentum=0.1):
    if training:
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        # Update running averages
        if running_mean is not None:
            running_mean = momentum * mean + (1 - momentum) * running_mean
            running_var = momentum * var + (1 - momentum) * running_var
    else:
        mean = running_mean
        var = running_var

    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

### PyTorch

```python
class CNNWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 10)
        self.bn_fc = nn.BatchNorm1d(10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.flatten(1)
        x = self.bn_fc(self.fc1(x))
        return x
```

---

## 7. Weight Decay (L2 Regularization)

Why penalize large weights? Large weights amplify small input variations, making the model overly sensitive to training noise (overfitting). L2 regularization adds `lambda * ||W||^2` to the loss, which pushes all weights proportionally toward zero — a smooth, gentle shrinkage. In contrast, L1 regularization drives some weights to *exactly* zero (useful for feature selection but less common in deep learning). L2 is the default choice because it produces smoother optimization landscapes.

Penalizes weight magnitudes.

### Formula

```
L_total = L_data + λ × ||W||²
∇L_total = ∇L_data + 2λW
```

### PyTorch

```python
# Method 1: Set in optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Method 2: Add directly to loss
l2_lambda = 1e-4
l2_reg = sum(p.pow(2).sum() for p in model.parameters())
loss = criterion(output, target) + l2_lambda * l2_reg
```

---

## 8. Early Stopping

Stop training when validation loss stops improving.

### PyTorch Implementation

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=10)
for epoch in range(epochs):
    train_loss = train(model, train_loader)
    val_loss = validate(model, val_loader)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 9. Data Augmentation

Transform training data to increase diversity.

### Image Data

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

---

## 10. NumPy vs PyTorch Comparison

### Optimizer Implementation

```python
# NumPy (manual implementation)
m = np.zeros_like(W)
v = np.zeros_like(W)
for t in range(1, epochs + 1):
    grad = compute_gradient(W, X, y)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    W -= lr * m_hat / (np.sqrt(v_hat) + eps)

# PyTorch (automatic)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    loss = criterion(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Summary

### Core Concepts

1. **Optimizer**: Adam is the default choice, SGD+Momentum still valid
2. **Learning Rate**: Improve convergence with proper scheduling
3. **Regularization**: Combine Dropout, BatchNorm, Weight Decay
4. **Early Stopping**: Basic overfitting prevention

### Recommended Starting Settings

```python
# Basic configuration
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
```

---

## Exercises

### Exercise 1: Adam vs SGD Convergence

Compare Adam and SGD optimizers on a simple regression task.

1. Generate synthetic data: `y = 2x + 1 + noise` with 200 samples.
2. Train a single linear layer for 200 epochs using both `torch.optim.SGD(lr=0.01)` and `torch.optim.Adam(lr=0.001)`.
3. Record and plot the training loss curve for each optimizer.
4. Note how quickly each converges and how stable the loss curve is.

### Exercise 2: Implement Adam from Scratch

Implement the Adam update rule in NumPy without using any optimizer library.

1. Initialize `m=0`, `v=0`, `t=0` for a single parameter `W`.
2. Simulate 100 gradient steps using random gradients drawn from `N(0, 1)`.
3. Apply the bias-corrected Adam update at each step.
4. Compare the final `W` value against `torch.optim.Adam` to verify correctness.

### Exercise 3: Learning Rate Scheduling Comparison

Observe how different schedules affect training dynamics.

1. Train a small MLP on MNIST for 20 epochs using Adam with `lr=0.01`.
2. Compare three schedulers: no scheduling, `StepLR(step_size=5, gamma=0.5)`, and `CosineAnnealingLR(T_max=20)`.
3. Plot validation accuracy vs. epoch for all three cases.
4. Explain why a high initial learning rate combined with decay often outperforms a fixed low rate.

### Exercise 4: Dropout Regularization Effect

Empirically verify that dropout reduces overfitting.

1. Create a small dataset by sampling 500 points from MNIST training data.
2. Train two identical 3-layer MLPs for 50 epochs — one with `Dropout(0.5)`, one without.
3. Record both training accuracy and validation accuracy.
4. Plot the accuracy gap (train - val) for each model. Explain the observed regularization effect.

### Exercise 5: Early Stopping with Model Checkpointing

Extend the `EarlyStopping` class to save the best model checkpoint.

1. Add a `save_path` parameter to `EarlyStopping.__init__`.
2. When validation loss improves, save the model with `torch.save(model.state_dict(), save_path)`.
3. After training ends (either by early stop or epoch limit), reload the best checkpoint and evaluate on the test set.
4. Compare test accuracy using the final checkpoint vs. the best checkpoint. Explain when saving the best model matters most.

---

## Next Steps

In [07_CNN_Basics.md](./07_CNN_Basics.md), we'll learn convolutional neural networks.
