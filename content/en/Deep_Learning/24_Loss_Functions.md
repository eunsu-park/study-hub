[Previous: Training Optimization](./23_Training_Optimization.md) | [Next: Optimizers](./25_Optimizers.md)

---

# 24. Loss Functions

## Learning Objectives

- Understand the role of loss functions in training neural networks and their relationship with optimization
- Master regression losses (MSE, MAE, Huber) and their applications in different scenarios
- Learn classification losses (BCE, Cross-Entropy, Focal Loss) and their use cases for balanced/imbalanced datasets
- Explore metric learning losses (Contrastive, Triplet, InfoNCE) for representation learning
- Implement custom loss functions in PyTorch for segmentation, detection, and generative models

**Difficulty**: ⭐⭐⭐

---

## 1. Introduction to Loss Functions

### 1.1 Role in Neural Network Training

A loss function (also called objective function or cost function) measures how well a neural network's predictions match the ground truth. During training, we minimize this loss using optimization algorithms like SGD or Adam.

```
Training Loop:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Input (x) ──▶ Model(θ) ──▶ Prediction (ŷ)            │
│                                  │                      │
│                                  ▼                      │
│                         Loss = L(ŷ, y)                 │
│                                  │                      │
│                                  ▼                      │
│                         ∂L/∂θ (Backprop)               │
│                                  │                      │
│                                  ▼                      │
│                    θ ← θ - η·∂L/∂θ (Update)            │
│                                  │                      │
│                                  └──────────────────────┘
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Key Properties of Good Loss Functions:**
- **Differentiable**: Must have gradients for backpropagation
- **Convex (ideally)**: Single global minimum makes optimization easier
- **Task-aligned**: Reflects the actual evaluation metric when possible
- **Numerically stable**: Avoids overflow/underflow

### 1.2 Loss Landscape Visualization

The loss landscape shows how loss varies with model parameters:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_loss_landscape():
    """Visualize a simple 2D loss landscape"""
    # Create a grid of parameter values
    w1 = np.linspace(-5, 5, 100)
    w2 = np.linspace(-5, 5, 100)
    W1, W2 = np.meshgrid(w1, w2)

    # Example loss: Rosenbrock function (non-convex)
    a, b = 1, 100
    Z = (a - W1)**2 + b * (W2 - W1**2)**2

    # Plot 3D surface
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('w1')
    ax1.set_ylabel('w2')
    ax1.set_zlabel('Loss')
    ax1.set_title('3D Loss Landscape')

    # Plot contour
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(W1, W2, Z, levels=30, cmap='viridis')
    ax2.set_xlabel('w1')
    ax2.set_ylabel('w2')
    ax2.set_title('Contour Plot')
    plt.colorbar(contour, ax=ax2)

    plt.tight_layout()
    plt.savefig('loss_landscape.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_loss_landscape()
```

### 1.3 Relationship with Optimization

Different loss functions create different optimization challenges:

| Loss Type | Landscape | Optimization Challenge |
|-----------|-----------|------------------------|
| MSE | Smooth, convex | Easy, stable gradients |
| Cross-Entropy | Smooth, convex | Can have vanishing gradients |
| Triplet Loss | Non-convex, many local minima | Requires careful mining |
| GAN Loss | Non-convex, saddle points | Unstable, mode collapse |

---

## 2. Regression Losses

Regression losses are used when predicting continuous values (e.g., house prices, temperatures, coordinates).

### 2.1 Mean Squared Error (L2 Loss)

**Formula:**
```
MSE = (1/n) Σ(ŷᵢ - yᵢ)²
```

**Properties:**
- Penalizes large errors heavily (quadratic)
- Sensitive to outliers
- Smooth gradients everywhere

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn

# Built-in version
mse_loss = nn.MSELoss()

# Example usage
predictions = torch.tensor([2.5, 0.0, 2.1, 7.8])
targets = torch.tensor([3.0, -0.5, 2.0, 8.0])

loss = mse_loss(predictions, targets)
print(f"MSE Loss: {loss.item():.4f}")  # 0.0825

# Manual implementation
def mse_manual(pred, target):
    return torch.mean((pred - target) ** 2)

loss_manual = mse_manual(predictions, targets)
print(f"Manual MSE: {loss_manual.item():.4f}")  # 0.0825
```

**When to Use:**
- Regression tasks where large errors should be heavily penalized
- Data without significant outliers
- When you want smooth gradients for optimization

### 2.2 Mean Absolute Error (L1 Loss)

**Formula:**
```
MAE = (1/n) Σ|ŷᵢ - yᵢ|
```

**Properties:**
- Linear penalty (robust to outliers)
- Gradient has constant magnitude
- Can be unstable at zero (non-differentiable)

**PyTorch Implementation:**

```python
# Built-in version
mae_loss = nn.L1Loss()

predictions = torch.tensor([2.5, 0.0, 2.1, 7.8])
targets = torch.tensor([3.0, -0.5, 2.0, 8.0])

loss = mae_loss(predictions, targets)
print(f"MAE Loss: {loss.item():.4f}")  # 0.2250

# Manual implementation
def mae_manual(pred, target):
    return torch.mean(torch.abs(pred - target))

loss_manual = mae_manual(predictions, targets)
print(f"Manual MAE: {loss_manual.item():.4f}")  # 0.2250
```

**When to Use:**
- Data with outliers
- When all errors should be weighted equally
- Robust regression tasks

### 2.3 Huber Loss (Smooth L1)

**Formula:**
```
         ⎧  0.5(ŷ - y)²          if |ŷ - y| ≤ δ
L_δ(ŷ,y) = ⎨
         ⎩  δ|ŷ - y| - 0.5δ²    otherwise
```

**Properties:**
- Combines benefits of L1 and L2
- Quadratic for small errors, linear for large errors
- Controlled by δ (transition point)

**PyTorch Implementation:**

```python
# Built-in version (SmoothL1Loss uses δ=1.0)
huber_loss = nn.SmoothL1Loss(beta=1.0)  # beta is δ

predictions = torch.tensor([2.5, 0.0, 2.1, 10.0])  # Last value is outlier
targets = torch.tensor([3.0, -0.5, 2.0, 8.0])

loss = huber_loss(predictions, targets)
print(f"Huber Loss: {loss.item():.4f}")  # 0.4125

# Manual implementation
def huber_manual(pred, target, delta=1.0):
    error = pred - target
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic**2 + delta * linear)

loss_manual = huber_manual(predictions, targets, delta=1.0)
print(f"Manual Huber: {loss_manual.item():.4f}")  # 0.4125
```

**When to Use:**
- Data with some outliers, but you still want to penalize large errors
- Object detection (bounding box regression)
- Robotics (sensor fusion)

### 2.4 Log-Cosh Loss

**Formula:**
```
L(ŷ, y) = Σ log(cosh(ŷᵢ - yᵢ))
```

**Properties:**
- Twice differentiable (smoother than Huber)
- Approximately equal to (x²/2) for small x, |x| for large x
- Less sensitive to outliers than MSE

**PyTorch Implementation:**

```python
def log_cosh_loss(pred, target):
    """Log-Cosh Loss"""
    error = pred - target
    return torch.mean(torch.log(torch.cosh(error)))

predictions = torch.tensor([2.5, 0.0, 2.1, 10.0])
targets = torch.tensor([3.0, -0.5, 2.0, 8.0])

loss = log_cosh_loss(predictions, targets)
print(f"Log-Cosh Loss: {loss.item():.4f}")
```

**When to Use:**
- When you need twice-differentiable loss (e.g., for Hessian-based optimizers)
- XGBoost and other gradient boosting methods

### 2.5 Regression Loss Comparison

```python
import torch
import matplotlib.pyplot as plt

def compare_regression_losses():
    """Compare different regression losses"""
    errors = torch.linspace(-5, 5, 200)

    # Calculate losses
    mse = errors ** 2
    mae = torch.abs(errors)
    huber = torch.where(
        torch.abs(errors) <= 1.0,
        0.5 * errors ** 2,
        torch.abs(errors) - 0.5
    )
    log_cosh = torch.log(torch.cosh(errors))

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(errors.numpy(), mse.numpy(), label='MSE (L2)', linewidth=2)
    plt.plot(errors.numpy(), mae.numpy(), label='MAE (L1)', linewidth=2)
    plt.plot(errors.numpy(), huber.numpy(), label='Huber (δ=1)', linewidth=2)
    plt.plot(errors.numpy(), log_cosh.numpy(), label='Log-Cosh', linewidth=2)
    plt.xlabel('Prediction Error (ŷ - y)')
    plt.ylabel('Loss')
    plt.title('Regression Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 10)

    # Gradient plot
    plt.subplot(1, 2, 2)
    grad_mse = 2 * errors
    grad_mae = torch.sign(errors)
    grad_huber = torch.where(
        torch.abs(errors) <= 1.0,
        errors,
        torch.sign(errors)
    )
    grad_log_cosh = torch.tanh(errors)

    plt.plot(errors.numpy(), grad_mse.numpy(), label='MSE grad', linewidth=2)
    plt.plot(errors.numpy(), grad_mae.numpy(), label='MAE grad', linewidth=2)
    plt.plot(errors.numpy(), grad_huber.numpy(), label='Huber grad', linewidth=2)
    plt.plot(errors.numpy(), grad_log_cosh.numpy(), label='Log-Cosh grad', linewidth=2)
    plt.xlabel('Prediction Error (ŷ - y)')
    plt.ylabel('Gradient')
    plt.title('Loss Gradients')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 5)

    plt.tight_layout()
    plt.savefig('regression_losses.png', dpi=150, bbox_inches='tight')
    plt.show()

compare_regression_losses()
```

**Comparison Table:**

| Loss | Outlier Robustness | Gradient Smoothness | Use Case |
|------|-------------------|---------------------|----------|
| **MSE** | Low | High | Clean data, stable training |
| **MAE** | High | Low (discontinuous at 0) | Outlier-prone data |
| **Huber** | Medium | Medium | Balance between MSE/MAE |
| **Log-Cosh** | Medium-High | High (twice differentiable) | Advanced optimizers |

---

## 3. Classification Losses

Classification losses are used for discrete label prediction tasks.

### 3.1 Binary Cross-Entropy (BCE)

**Formula:**
```
BCE = -(1/n) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

where:
- yᵢ ∈ {0, 1} (true label)
- ŷᵢ ∈ (0, 1) (predicted probability)
```

**Properties:**
- For binary classification (2 classes)
- Requires sigmoid activation to output probabilities
- Convex loss function

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn

# Method 1: BCELoss (requires sigmoid applied first)
sigmoid = nn.Sigmoid()
bce_loss = nn.BCELoss()

logits = torch.tensor([0.5, 2.0, -1.0, 0.0])  # Raw outputs
targets = torch.tensor([1.0, 1.0, 0.0, 0.0])  # Binary labels

probabilities = sigmoid(logits)
loss = bce_loss(probabilities, targets)
print(f"BCE Loss: {loss.item():.4f}")

# Method 2: BCEWithLogitsLoss (numerically stable, combines sigmoid + BCE)
bce_with_logits = nn.BCEWithLogitsLoss()
loss_stable = bce_with_logits(logits, targets)
print(f"BCE with Logits: {loss_stable.item():.4f}")

# Manual implementation
def bce_manual(pred_probs, target):
    """Manual BCE (expects probabilities)"""
    epsilon = 1e-7  # For numerical stability
    pred_probs = torch.clamp(pred_probs, epsilon, 1 - epsilon)
    return -torch.mean(
        target * torch.log(pred_probs) +
        (1 - target) * torch.log(1 - pred_probs)
    )

loss_manual = bce_manual(probabilities, targets)
print(f"Manual BCE: {loss_manual.item():.4f}")
```

**Binary Classification Example:**

```python
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Single output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Raw logit

# Training setup
model = BinaryClassifier(input_dim=10)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy data
inputs = torch.randn(32, 10)  # Batch of 32
targets = torch.randint(0, 2, (32, 1)).float()

# Training step
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()

print(f"Training Loss: {loss.item():.4f}")
```

**When to Use:**
- Binary classification (spam/not spam, cat/dog)
- Multi-label classification (each label is independent)
- Sigmoid output activation

### 3.2 Cross-Entropy Loss (Multi-Class)

**Formula:**
```
CE = -(1/n) Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)

where:
- yᵢⱼ is 1 if sample i is class j, else 0
- ŷᵢⱼ is predicted probability for class j
```

**Properties:**
- For multi-class classification (K > 2 classes)
- Requires softmax activation
- PyTorch's `CrossEntropyLoss` combines softmax + NLLLoss

**PyTorch Implementation:**

```python
# Method 1: CrossEntropyLoss (combines log_softmax + NLLLoss)
ce_loss = nn.CrossEntropyLoss()

logits = torch.tensor([
    [2.0, 1.0, 0.1],  # Sample 1
    [0.5, 2.5, 0.3],  # Sample 2
    [0.1, 0.2, 3.0],  # Sample 3
])
targets = torch.tensor([0, 1, 2])  # Class indices

loss = ce_loss(logits, targets)
print(f"CrossEntropy Loss: {loss.item():.4f}")

# Method 2: Manual with softmax + NLLLoss
log_softmax = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()

log_probs = log_softmax(logits)
loss_manual = nll_loss(log_probs, targets)
print(f"Manual CE (LogSoftmax + NLL): {loss_manual.item():.4f}")

# Method 3: From scratch
def cross_entropy_manual(logits, targets):
    """Manual cross-entropy implementation"""
    # Compute log softmax
    max_logits = torch.max(logits, dim=1, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)  # Numerical stability
    log_probs = (logits - max_logits) - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))

    # Gather log probabilities for correct classes
    batch_size = logits.size(0)
    loss = -log_probs[range(batch_size), targets].mean()
    return loss

loss_scratch = cross_entropy_manual(logits, targets)
print(f"From Scratch CE: {loss_scratch.item():.4f}")
```

**Multi-Class Classification Example:**

```python
class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Raw logits (no softmax)

# Training setup
num_classes = 10
model = MultiClassClassifier(input_dim=20, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy data
inputs = torch.randn(64, 20)  # Batch of 64
targets = torch.randint(0, num_classes, (64,))  # Class indices

# Training step
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()

print(f"Training Loss: {loss.item():.4f}")

# Inference
with torch.no_grad():
    test_input = torch.randn(1, 20)
    logits = model(test_input)
    probs = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1)
    print(f"Predicted Class: {predicted_class.item()}")
    print(f"Class Probabilities: {probs[0].tolist()}")
```

### 3.3 Label Smoothing

**Concept:**
Instead of hard labels (0 or 1), use soft labels:
```
y_smooth = y(1 - ε) + ε/K

where:
- ε is smoothing parameter (e.g., 0.1)
- K is number of classes
```

**Benefits:**
- Prevents overconfidence
- Better generalization
- Regularization effect

**PyTorch Implementation:**

```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,) class indices
        """
        num_classes = logits.size(1)
        log_probs = torch.log_softmax(logits, dim=1)

        # Create smooth targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

# Example usage
criterion = LabelSmoothingCrossEntropy(epsilon=0.1)

logits = torch.randn(4, 5)  # 4 samples, 5 classes
targets = torch.tensor([0, 2, 1, 4])

loss = criterion(logits, targets)
print(f"Label Smoothing CE: {loss.item():.4f}")

# Compare with standard CE
standard_ce = nn.CrossEntropyLoss()
loss_standard = standard_ce(logits, targets)
print(f"Standard CE: {loss_standard.item():.4f}")
```

**When to Use:**
- Image classification (ImageNet, CIFAR)
- Preventing overconfidence
- Model calibration

### 3.4 Focal Loss

**Formula:**
```
FL(pₜ) = -αₜ(1 - pₜ)^γ log(pₜ)

where:
- pₜ = p if y=1, else 1-p
- α balances class frequencies
- γ focuses on hard examples (typically 2)
```

**Motivation:**
- Addresses class imbalance (e.g., 1:1000 ratio in object detection)
- Down-weights easy examples, focuses on hard negatives
- Introduced in RetinaNet paper

**PyTorch Implementation:**

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for class imbalance (default 0.25)
            gamma: Focusing parameter (default 2.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C) raw predictions
            targets: (N,) class indices
        """
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        p = torch.exp(-ce_loss)  # Probability of correct class

        # Focal loss formula
        focal_weight = (1 - p) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Binary Focal Loss variant
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: (N,) or (N, 1) raw predictions
            targets: (N,) or (N, 1) binary labels {0, 1}
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()

# Example: Imbalanced dataset
num_classes = 3
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

# Simulate imbalanced batch (mostly class 0)
logits = torch.randn(100, num_classes)
targets = torch.cat([
    torch.zeros(80, dtype=torch.long),   # 80% class 0
    torch.ones(15, dtype=torch.long),    # 15% class 1
    torch.full((5,), 2, dtype=torch.long)  # 5% class 2
])

loss_focal = focal_loss(logits, targets)
loss_ce = nn.CrossEntropyLoss()(logits, targets)

print(f"Focal Loss: {loss_focal.item():.4f}")
print(f"CE Loss: {loss_ce.item():.4f}")
```

**Focal Loss Visualization:**

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_focal_loss():
    """Visualize how focal loss down-weights easy examples"""
    p = np.linspace(0.01, 1, 100)  # Probability of correct class

    # Standard CE
    ce = -np.log(p)

    # Focal loss with different γ
    fl_gamma_0 = ce  # γ=0 is same as CE
    fl_gamma_1 = (1 - p) * ce
    fl_gamma_2 = (1 - p)**2 * ce
    fl_gamma_5 = (1 - p)**5 * ce

    plt.figure(figsize=(10, 6))
    plt.plot(p, ce, label='CE (γ=0)', linewidth=2)
    plt.plot(p, fl_gamma_1, label='FL (γ=1)', linewidth=2)
    plt.plot(p, fl_gamma_2, label='FL (γ=2)', linewidth=2)
    plt.plot(p, fl_gamma_5, label='FL (γ=5)', linewidth=2)

    plt.xlabel('Probability of Correct Class (p)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Focal Loss: Down-weighting Easy Examples', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 5)

    # Annotate easy vs hard examples
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    plt.text(0.25, 4.5, 'Hard Examples\n(low confidence)', fontsize=10, ha='center')
    plt.text(0.75, 4.5, 'Easy Examples\n(high confidence)', fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig('focal_loss.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_focal_loss()
```

**When to Use:**
- Object detection (RetinaNet, FCOS)
- Imbalanced classification (fraud detection, medical diagnosis)
- When you have many easy negatives

### 3.5 BCE vs Cross-Entropy

**Decision Guide:**

| Task | Loss | Activation | Notes |
|------|------|------------|-------|
| **Binary classification** | `BCEWithLogitsLoss` | None (included) | 2 classes, mutually exclusive |
| **Multi-class classification** | `CrossEntropyLoss` | None (included) | K classes, mutually exclusive |
| **Multi-label classification** | `BCEWithLogitsLoss` | None (included) | Multiple independent labels |
| **Imbalanced classification** | `FocalLoss` | Softmax | Class imbalance |

**Example: Multi-Label Classification:**

```python
# Multi-label: Each sample can belong to multiple classes
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_labels)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Raw logits

model = MultiLabelClassifier(input_dim=50, num_labels=5)
criterion = nn.BCEWithLogitsLoss()  # Use BCE, not CE!

# Multi-label targets (sample can have multiple 1s)
inputs = torch.randn(32, 50)
targets = torch.tensor([
    [1, 0, 1, 0, 1],  # Sample 1: classes 0, 2, 4
    [0, 1, 1, 0, 0],  # Sample 2: classes 1, 2
    # ... more samples
]).float()

outputs = model(inputs[:2])
loss = criterion(outputs, targets)
print(f"Multi-Label Loss: {loss.item():.4f}")
```

---

## 4. Ranking and Metric Learning Losses

These losses learn embeddings where similar items are close and dissimilar items are far apart.

### 4.1 Contrastive Loss

**Formula:**
```
L = (1/2) * [y * d² + (1-y) * max(0, m - d)²]

where:
- d = ||f(x₁) - f(x₂)||₂ (Euclidean distance)
- y = 1 if similar, 0 if dissimilar
- m is margin (e.g., 1.0)
```

**Use Case:**
- Siamese networks
- Face verification
- Signature verification

**PyTorch Implementation:**

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Args:
            output1: (N, embedding_dim) embeddings from first input
            output2: (N, embedding_dim) embeddings from second input
            label: (N,) 1 if similar, 0 if dissimilar
        """
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)

        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(
                torch.clamp(self.margin - euclidean_distance, min=0.0), 2
            )
        )
        return loss_contrastive

# Siamese Network Example
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward_one(self, x):
        return self.fc(x)

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2

# Training
model = SiameseNetwork(input_dim=784, embedding_dim=128)
criterion = ContrastiveLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy data
x1 = torch.randn(32, 784)
x2 = torch.randn(32, 784)
labels = torch.randint(0, 2, (32,)).float()  # 1=similar, 0=dissimilar

# Training step
optimizer.zero_grad()
output1, output2 = model(x1, x2)
loss = criterion(output1, output2, labels)
loss.backward()
optimizer.step()

print(f"Contrastive Loss: {loss.item():.4f}")
```

### 4.2 Triplet Loss

**Formula:**
```
L = max(0, d(a, p) - d(a, n) + margin)

where:
- a: anchor
- p: positive (same class as anchor)
- n: negative (different class)
- d(x, y) = ||f(x) - f(y)||₂
```

**Mining Strategies:**
- **Hard negatives**: max d(a, p) - d(a, n)
- **Semi-hard negatives**: d(a, p) < d(a, n) < d(a, p) + margin
- **Batch-all**: All valid triplets in batch

**PyTorch Implementation:**

```python
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: (N, embedding_dim)
            positive: (N, embedding_dim)
            negative: (N, embedding_dim)
        """
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)

        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# Online Triplet Mining
class OnlineTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (N, embedding_dim)
            labels: (N,) class labels
        """
        # Compute pairwise distances
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        # For each anchor, get hardest positive and negative
        batch_size = embeddings.size(0)
        triplet_loss = 0
        num_triplets = 0

        for i in range(batch_size):
            # Positive mask: same class as anchor
            pos_mask = labels == labels[i]
            pos_mask[i] = False  # Exclude anchor itself

            # Negative mask: different class
            neg_mask = labels != labels[i]

            if pos_mask.any() and neg_mask.any():
                # Hardest positive
                hardest_positive_dist = pairwise_dist[i][pos_mask].max()

                # Hardest negative (closest negative)
                hardest_negative_dist = pairwise_dist[i][neg_mask].min()

                loss = torch.relu(
                    hardest_positive_dist - hardest_negative_dist + self.margin
                )
                triplet_loss += loss
                num_triplets += 1

        return triplet_loss / max(num_triplets, 1)

# Example usage
embedding_dim = 128
criterion = TripletLoss(margin=1.0)
online_criterion = OnlineTripletLoss(margin=1.0)

# Method 1: Pre-mined triplets
anchor = torch.randn(32, embedding_dim)
positive = torch.randn(32, embedding_dim)
negative = torch.randn(32, embedding_dim)

loss = criterion(anchor, positive, negative)
print(f"Triplet Loss: {loss.item():.4f}")

# Method 2: Online mining
embeddings = torch.randn(64, embedding_dim)
labels = torch.randint(0, 10, (64,))  # 10 classes

loss_online = online_criterion(embeddings, labels)
print(f"Online Triplet Loss: {loss_online.item():.4f}")
```

**Training Example:**

```python
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Training loop
model = EmbeddingNet(input_dim=784, embedding_dim=128)
criterion = OnlineTripletLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    # Simulate batch
    inputs = torch.randn(64, 784)
    labels = torch.randint(0, 10, (64,))

    optimizer.zero_grad()
    embeddings = model(inputs)
    loss = criterion(embeddings, labels)
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

**When to Use:**
- Face recognition (FaceNet)
- Person re-identification
- Image retrieval

### 4.3 InfoNCE / NT-Xent Loss

**Formula:**
```
L = -log [exp(sim(z_i, z_j)/τ) / Σₖ exp(sim(z_i, z_k)/τ)]

where:
- z_i, z_j are positive pair embeddings
- τ is temperature parameter (e.g., 0.07)
- sim(u, v) = u·v / (||u|| ||v||) (cosine similarity)
```

**Use Case:**
- Self-supervised learning (SimCLR, MoCo)
- Contrastive language-image pre-training (CLIP)

**PyTorch Implementation:**

```python
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: (N, embedding_dim) embeddings of view 1
            z_j: (N, embedding_dim) embeddings of view 2
        """
        batch_size = z_i.size(0)

        # Normalize embeddings
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # Compute similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)  # (2N, dim)
        similarity_matrix = torch.mm(representations, representations.T)  # (2N, 2N)

        # Create labels: positive pairs are at (i, N+i) and (N+i, i)
        labels = torch.cat([
            torch.arange(batch_size) + batch_size,
            torch.arange(batch_size)
        ], dim=0).to(z_i.device)

        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

        # Compute loss
        similarity_matrix = similarity_matrix / self.temperature
        loss = nn.functional.cross_entropy(similarity_matrix, labels)

        return loss

# Simplified NT-Xent (for SimCLR)
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """Simplified version"""
        batch_size = z_i.size(0)

        # L2 normalize
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # Positive similarity
        pos_sim = (z_i * z_j).sum(dim=1) / self.temperature  # (N,)

        # All similarities
        z = torch.cat([z_i, z_j], dim=0)  # (2N, dim)
        sim_matrix = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # Remove diagonal
        sim_matrix.fill_diagonal_(-float('inf'))

        # Compute loss for i -> j
        pos_sim_expanded = pos_sim.unsqueeze(1)  # (N, 1)
        negatives_i = sim_matrix[:batch_size]  # (N, 2N)
        logits_i = torch.cat([pos_sim_expanded, negatives_i], dim=1)  # (N, 2N+1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)

        loss = nn.functional.cross_entropy(logits_i, labels)
        return loss

# Example usage
criterion = InfoNCELoss(temperature=0.07)

# Simulate augmented views
z_i = torch.randn(128, 256)  # View 1 embeddings
z_j = torch.randn(128, 256)  # View 2 embeddings

loss = criterion(z_i, z_j)
print(f"InfoNCE Loss: {loss.item():.4f}")
```

**When to Use:**
- Self-supervised pre-training (SimCLR)
- Vision-language models (CLIP)
- Contrastive learning

---

## 5. Segmentation and Detection Losses

### 5.1 Dice Loss

**Formula:**
```
Dice Loss = 1 - (2 * |X ∩ Y|) / (|X| + |Y|)

where:
- X is predicted segmentation
- Y is ground truth
```

**Properties:**
- Handles class imbalance (background vs foreground)
- Differentiable approximation of IoU
- Range: [0, 1]

**PyTorch Implementation:**

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        """
        Args:
            smooth: Smoothing constant to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C, H, W) predicted probabilities
            target: (N, C, H, W) one-hot encoded ground truth
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1 - dice_score

# Multi-class Dice Loss
class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C, H, W) logits
            target: (N, H, W) class indices
        """
        num_classes = pred.size(1)
        pred_softmax = torch.softmax(pred, dim=1)

        # Convert target to one-hot
        target_one_hot = nn.functional.one_hot(target, num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        total_loss = 0
        for c in range(num_classes):
            pred_c = pred_softmax[:, c]
            target_c = target_one_hot[:, c]

            intersection = (pred_c * target_c).sum()
            dice = (2. * intersection + self.smooth) / (
                pred_c.sum() + target_c.sum() + self.smooth
            )
            total_loss += 1 - dice

        return total_loss / num_classes

# Example usage
batch_size, num_classes, H, W = 4, 3, 256, 256

# Binary segmentation
pred_binary = torch.sigmoid(torch.randn(batch_size, 1, H, W))
target_binary = torch.randint(0, 2, (batch_size, 1, H, W)).float()

dice_loss = DiceLoss()
loss_binary = dice_loss(pred_binary, target_binary)
print(f"Binary Dice Loss: {loss_binary.item():.4f}")

# Multi-class segmentation
pred_multi = torch.randn(batch_size, num_classes, H, W)
target_multi = torch.randint(0, num_classes, (batch_size, H, W))

dice_multi = MultiClassDiceLoss()
loss_multi = dice_multi(pred_multi, target_multi)
print(f"Multi-class Dice Loss: {loss_multi.item():.4f}")
```

### 5.2 IoU Loss / GIoU Loss

**IoU (Intersection over Union):**
```
IoU = Area(Intersection) / Area(Union)
```

**GIoU (Generalized IoU):**
```
GIoU = IoU - |C \ (A ∪ B)| / |C|

where C is smallest box enclosing A and B
```

**PyTorch Implementation:**

```python
def iou_loss(pred_boxes, target_boxes):
    """
    Args:
        pred_boxes: (N, 4) [x1, y1, x2, y2]
        target_boxes: (N, 4) [x1, y1, x2, y2]
    """
    # Intersection coordinates
    x1_inter = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1_inter = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2_inter = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2_inter = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    # Intersection area
    inter_area = torch.clamp(x2_inter - x1_inter, min=0) * \
                 torch.clamp(y2_inter - y1_inter, min=0)

    # Union area
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                  (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-6)

    return 1 - iou.mean()

def giou_loss(pred_boxes, target_boxes):
    """Generalized IoU Loss"""
    # Intersection
    x1_inter = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1_inter = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2_inter = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2_inter = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = torch.clamp(x2_inter - x1_inter, min=0) * \
                 torch.clamp(y2_inter - y1_inter, min=0)

    # Union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                  (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-6)

    # Enclosing box
    x1_c = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1_c = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2_c = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2_c = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    enclosing_area = (x2_c - x1_c) * (y2_c - y1_c)

    # GIoU
    giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-6)

    return 1 - giou.mean()

# Example usage
pred_boxes = torch.tensor([
    [10, 10, 50, 50],
    [20, 20, 60, 60]
]).float()

target_boxes = torch.tensor([
    [15, 15, 55, 55],
    [25, 25, 65, 65]
]).float()

loss_iou = iou_loss(pred_boxes, target_boxes)
loss_giou = giou_loss(pred_boxes, target_boxes)

print(f"IoU Loss: {loss_iou.item():.4f}")
print(f"GIoU Loss: {loss_giou.item():.4f}")
```

### 5.3 Combined Losses

**Example: CE + Dice for Segmentation:**

```python
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss()

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C, H, W) logits
            target: (N, H, W) class indices
        """
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

# Example
criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
pred = torch.randn(2, 3, 128, 128)
target = torch.randint(0, 3, (2, 128, 128))

loss = criterion(pred, target)
print(f"Combined Loss (CE + Dice): {loss.item():.4f}")
```

---

## 6. Generative Model Losses

### 6.1 Adversarial Loss (GAN)

**Minimax GAN:**
```
min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

**PyTorch Implementation:**

```python
# Standard GAN loss
def gan_loss_discriminator(real_output, fake_output):
    """Discriminator loss"""
    real_loss = nn.functional.binary_cross_entropy_with_logits(
        real_output, torch.ones_like(real_output)
    )
    fake_loss = nn.functional.binary_cross_entropy_with_logits(
        fake_output, torch.zeros_like(fake_output)
    )
    return real_loss + fake_loss

def gan_loss_generator(fake_output):
    """Generator loss (non-saturating)"""
    return nn.functional.binary_cross_entropy_with_logits(
        fake_output, torch.ones_like(fake_output)
    )

# Wasserstein GAN loss
def wgan_loss_discriminator(real_output, fake_output):
    """WGAN discriminator loss"""
    return -(torch.mean(real_output) - torch.mean(fake_output))

def wgan_loss_generator(fake_output):
    """WGAN generator loss"""
    return -torch.mean(fake_output)

# Example usage
real_output = torch.randn(32, 1)  # Discriminator output for real images
fake_output = torch.randn(32, 1)  # Discriminator output for fake images

# Standard GAN
d_loss = gan_loss_discriminator(real_output, fake_output)
g_loss = gan_loss_generator(fake_output)
print(f"GAN D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# WGAN
d_loss_wgan = wgan_loss_discriminator(real_output, fake_output)
g_loss_wgan = wgan_loss_generator(fake_output)
print(f"WGAN D Loss: {d_loss_wgan.item():.4f}, G Loss: {g_loss_wgan.item():.4f}")
```

### 6.2 VAE Loss

**Formula:**
```
L = Reconstruction Loss + KL Divergence
  = E[log p(x|z)] + KL(q(z|x) || p(z))
```

**PyTorch Implementation:**

```python
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL term (β-VAE)
    """
    # Reconstruction loss (BCE for binary images, MSE for continuous)
    recon_loss = nn.functional.binary_cross_entropy(
        recon_x, x, reduction='sum'
    )

    # KL divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_div

# Example VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# Training
model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x = torch.randn(64, 784).sigmoid()  # Dummy data
recon_x, mu, logvar = model(x)

loss = vae_loss(recon_x, x, mu, logvar, beta=1.0)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"VAE Loss: {loss.item():.4f}")
```

### 6.3 Perceptual Loss

**Concept:**
Instead of pixel-wise comparison, use feature-wise comparison from a pretrained network (e.g., VGG).

**PyTorch Implementation:**

```python
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3']):
        super().__init__()
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.feature_extractor = nn.ModuleDict()

        # Map layer names to indices
        layer_map = {
            'relu1_2': 4, 'relu2_2': 9, 'relu3_3': 16,
            'relu4_3': 23, 'relu5_3': 30
        }

        for name in layers:
            idx = layer_map[name]
            self.feature_extractor[name] = nn.Sequential(*list(vgg[:idx+1]))

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        """
        Args:
            pred: (N, 3, H, W) predicted image
            target: (N, 3, H, W) target image
        """
        loss = 0
        for name, extractor in self.feature_extractor.items():
            pred_features = extractor(pred)
            target_features = extractor(target)
            loss += nn.functional.mse_loss(pred_features, target_features)

        return loss / len(self.feature_extractor)

# Example usage (requires torchvision)
# perceptual_loss = PerceptualLoss()
# pred_img = torch.randn(4, 3, 224, 224)
# target_img = torch.randn(4, 3, 224, 224)
# loss = perceptual_loss(pred_img, target_img)
# print(f"Perceptual Loss: {loss.item():.4f}")
```

**When to Use:**
- Style transfer
- Super-resolution
- Image-to-image translation

---

## 7. Advanced Topics

### 7.1 Multi-Task Loss Weighting

**Problem:**
When training on multiple tasks simultaneously, how to balance the losses?

**Method 1: Manual Weighting**

```python
total_loss = w1 * task1_loss + w2 * task2_loss + w3 * task3_loss
```

**Method 2: Uncertainty Weighting**

Based on "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018).

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        # Learnable log variance for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        Args:
            losses: List of losses for each task
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]

        return total_loss

# Example
mtl = MultiTaskLoss(num_tasks=3)
optimizer = torch.optim.Adam(mtl.parameters(), lr=0.01)

# Simulate task losses
task_losses = [
    torch.tensor(2.5),  # Task 1
    torch.tensor(0.8),  # Task 2
    torch.tensor(1.2),  # Task 3
]

total_loss = mtl(task_losses)
print(f"Multi-task Loss: {total_loss.item():.4f}")
print(f"Learned weights: {torch.exp(-mtl.log_vars).detach()}")
```

**Method 3: GradNorm**

Balances task losses by normalizing gradients.

```python
class GradNorm:
    def __init__(self, model, num_tasks, alpha=1.5):
        """
        Args:
            model: Shared network
            num_tasks: Number of tasks
            alpha: Restoring force (typically 1.5)
        """
        self.model = model
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(num_tasks))
        self.initial_losses = None

    def compute_weights(self, losses, shared_params):
        """Update task weights based on gradient norms"""
        if self.initial_losses is None:
            self.initial_losses = losses.detach()

        # Compute weighted loss
        weighted_losses = losses * self.weights
        total_loss = weighted_losses.sum()

        # Compute gradients
        total_loss.backward(retain_graph=True)

        # Get gradient norms for shared layers
        grad_norms = []
        for i in range(self.num_tasks):
            # ... compute grad norm for task i
            pass

        # Update weights (simplified version)
        return self.weights

# Note: Full GradNorm implementation requires careful gradient manipulation
```

### 7.2 Curriculum Loss

**Concept:**
Start training on easy examples, gradually increase difficulty.

```python
class CurriculumLoss(nn.Module):
    def __init__(self, base_criterion, total_epochs):
        super().__init__()
        self.base_criterion = base_criterion
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def forward(self, pred, target, difficulty):
        """
        Args:
            pred: Predictions
            target: Ground truth
            difficulty: (N,) difficulty score for each sample [0, 1]
        """
        # Compute base loss
        losses = self.base_criterion(pred, target)

        # Curriculum weight: easier samples first
        progress = self.current_epoch / self.total_epochs
        threshold = progress  # Gradually increase difficulty threshold

        weights = (difficulty <= threshold).float()
        weights = weights / (weights.sum() + 1e-6) * weights.size(0)

        return (losses * weights).sum()

    def step_epoch(self):
        self.current_epoch += 1

# Example usage
criterion = CurriculumLoss(nn.CrossEntropyLoss(reduction='none'), total_epochs=100)

for epoch in range(100):
    pred = torch.randn(32, 10)
    target = torch.randint(0, 10, (32,))
    difficulty = torch.rand(32)  # Random difficulty scores

    loss = criterion(pred, target, difficulty)
    # ... backward, optimize ...

    criterion.step_epoch()
```

### 7.3 Custom Loss Functions

**Template for Custom Losses:**

```python
class CustomLoss(nn.Module):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions
            target: Ground truth

        Returns:
            loss: Scalar tensor
        """
        # Implement your loss computation
        loss = torch.mean((pred - target) ** 2)  # Example
        return loss

# Example: Asymmetric Loss (penalize overestimation more than underestimation)
class AsymmetricMSELoss(nn.Module):
    def __init__(self, over_penalty=2.0):
        super().__init__()
        self.over_penalty = over_penalty

    def forward(self, pred, target):
        error = pred - target

        # Penalize overestimation more
        loss = torch.where(
            error > 0,
            self.over_penalty * error ** 2,
            error ** 2
        )

        return loss.mean()

# Usage
asymmetric_loss = AsymmetricMSELoss(over_penalty=2.0)
pred = torch.tensor([2.5, 1.0, 3.0])
target = torch.tensor([2.0, 2.0, 2.0])

loss = asymmetric_loss(pred, target)
print(f"Asymmetric Loss: {loss.item():.4f}")
```

### 7.4 Numerical Stability Tips

**Problem 1: Log-Sum-Exp Trick**

```python
# Numerically unstable (can overflow)
def unstable_softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x))

# Stable version
def stable_softmax(x):
    x_max = torch.max(x)
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x)

# Example
x = torch.tensor([1000.0, 1001.0, 1002.0])
# unstable_softmax(x)  # Would cause overflow
stable = stable_softmax(x)
print(f"Stable Softmax: {stable}")
```

**Problem 2: Avoiding log(0)**

```python
# Bad: Can produce NaN
loss = -torch.log(pred)

# Good: Add small epsilon
epsilon = 1e-7
loss = -torch.log(pred + epsilon)

# Better: Use clamp
loss = -torch.log(torch.clamp(pred, min=epsilon))

# Best: Use built-in stable versions
loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
```

**Problem 3: Gradient Clipping**

```python
# Prevent exploding gradients
for param in model.parameters():
    if param.grad is not None:
        torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)

# Or clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

---

## 8. Practical Guide

### 8.1 Decision Tree for Loss Selection

```
Start
  │
  ├─ Task: Regression?
  │    │
  │    ├─ Clean data, no outliers → MSE
  │    ├─ Data with outliers → MAE or Huber
  │    └─ Need smooth gradients → Huber or Log-Cosh
  │
  ├─ Task: Classification?
  │    │
  │    ├─ Binary classification → BCEWithLogitsLoss
  │    ├─ Multi-class (mutually exclusive) → CrossEntropyLoss
  │    ├─ Multi-label (independent labels) → BCEWithLogitsLoss
  │    └─ Class imbalance → FocalLoss or weighted CE
  │
  ├─ Task: Segmentation?
  │    │
  │    ├─ Small objects → DiceLoss
  │    ├─ Class imbalance → DiceLoss or FocalLoss
  │    └─ Balanced data → CrossEntropyLoss or CE + Dice
  │
  ├─ Task: Object Detection?
  │    │
  │    ├─ Classification head → CrossEntropyLoss or FocalLoss
  │    └─ Bounding box regression → IoU Loss, GIoU Loss, or Smooth L1
  │
  ├─ Task: Metric Learning?
  │    │
  │    ├─ Pair verification → ContrastiveLoss
  │    ├─ Triplet comparison → TripletLoss
  │    └─ Self-supervised → InfoNCE (NT-Xent)
  │
  └─ Task: Generative Model?
       │
       ├─ GAN → Adversarial Loss (BCE or Wasserstein)
       ├─ VAE → Reconstruction Loss + KL Divergence
       └─ Image translation → Perceptual Loss + L1/L2
```

### 8.2 Comparison Table

| Task | Loss Function | Activation | Notes |
|------|---------------|------------|-------|
| **Regression** | MSE | None | Clean data |
| | MAE | None | Outlier-robust |
| | Huber | None | Balance L1/L2 |
| **Binary Classification** | BCEWithLogitsLoss | None (internal sigmoid) | Numerically stable |
| **Multi-Class** | CrossEntropyLoss | None (internal softmax) | Mutually exclusive |
| **Multi-Label** | BCEWithLogitsLoss | None (internal sigmoid) | Independent labels |
| **Imbalanced Classification** | FocalLoss | Softmax | Addresses imbalance |
| **Segmentation** | DiceLoss | Softmax | Handles class imbalance |
| | CE + Dice | Softmax | Combined approach |
| **Detection (bbox)** | IoU / GIoU Loss | None | Bounding box regression |
| **Detection (class)** | FocalLoss | Softmax | Handles easy negatives |
| **Face Verification** | ContrastiveLoss | None | Siamese networks |
| **Face Recognition** | TripletLoss | None | Triplet mining |
| **Self-Supervised** | InfoNCE | None | Contrastive learning |
| **GAN** | BCELoss (adversarial) | Sigmoid | Minimax game |
| **VAE** | BCE/MSE + KL | Sigmoid/None | Reconstruction + regularization |

### 8.3 Common Pitfalls

**1. Using wrong reduction**

```python
# Bad: Default reduction='mean' might not be what you want
criterion = nn.CrossEntropyLoss()

# Good: Explicit reduction
criterion = nn.CrossEntropyLoss(reduction='mean')  # or 'sum', 'none'
```

**2. Forgetting to apply activation**

```python
# Bad: Using BCELoss with raw logits
pred = model(x)  # Raw logits
loss = nn.BCELoss()(pred, target)  # Wrong! BCELoss expects probabilities

# Good: Use BCEWithLogitsLoss
loss = nn.BCEWithLogitsLoss()(pred, target)

# Or apply sigmoid first
loss = nn.BCELoss()(torch.sigmoid(pred), target)
```

**3. Wrong target format**

```python
# CrossEntropyLoss expects class indices, not one-hot
# Bad
target = torch.tensor([[1, 0, 0], [0, 1, 0]])  # One-hot
loss = nn.CrossEntropyLoss()(pred, target)  # Error!

# Good
target = torch.tensor([0, 1])  # Class indices
loss = nn.CrossEntropyLoss()(pred, target)
```

**4. Not handling class imbalance**

```python
# Bad: Ignoring class imbalance (e.g., 95% class 0, 5% class 1)
criterion = nn.CrossEntropyLoss()

# Good: Use class weights
class_weights = torch.tensor([1.0, 19.0])  # Weight minority class higher
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Or use FocalLoss
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

**5. Incorrect loss scaling**

```python
# Bad: Losses of different magnitudes
total_loss = loss1 + loss2  # loss1 ~ 0.01, loss2 ~ 100.0

# Good: Normalize or weight appropriately
total_loss = 100 * loss1 + loss2
# Or use learnable weights
total_loss = w1 * loss1 + w2 * loss2
```

### 8.4 Debugging Tips

**1. Check loss values**

```python
# Monitor loss statistics
def check_loss(loss, name="Loss"):
    print(f"{name}: {loss.item():.4f}")
    assert not torch.isnan(loss), f"{name} is NaN!"
    assert not torch.isinf(loss), f"{name} is Inf!"
    assert loss >= 0, f"{name} is negative!"

# Use in training
loss = criterion(pred, target)
check_loss(loss, name="Training Loss")
```

**2. Visualize loss landscape**

```python
import matplotlib.pyplot as plt

# Track loss over training
losses = []
for epoch in range(num_epochs):
    # ... training ...
    losses.append(loss.item())

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.yscale('log')  # Use log scale if loss varies widely
plt.show()
```

**3. Compare with baseline**

```python
# Sanity check: random predictions should give expected loss
# For CrossEntropyLoss with C classes: expected loss ≈ log(C)

num_classes = 10
random_pred = torch.randn(100, num_classes)
random_target = torch.randint(0, num_classes, (100,))

loss = nn.CrossEntropyLoss()(random_pred, random_target)
expected = torch.log(torch.tensor(num_classes, dtype=torch.float))

print(f"Random Loss: {loss.item():.4f}")
print(f"Expected: {expected.item():.4f}")  # Should be ≈ 2.3026 for 10 classes
```

**4. Gradient flow check**

```python
# Check if gradients are flowing
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad norm = {grad_norm:.6f}")
            if grad_norm == 0:
                print(f"  WARNING: Zero gradient!")
        else:
            print(f"{name}: No gradient!")

# After backward
loss.backward()
check_gradients(model)
```

### 8.5 Loss Function Impact on Convergence

**Example: Comparing Convergence Speed**

```python
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def train_with_loss(criterion, num_epochs=100):
    """Train and return loss history"""
    model = SimpleNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Dummy data
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)

    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

# Compare different losses
mse_losses = train_with_loss(nn.MSELoss())
mae_losses = train_with_loss(nn.L1Loss())
huber_losses = train_with_loss(nn.SmoothL1Loss())

plt.figure(figsize=(10, 6))
plt.plot(mse_losses, label='MSE', linewidth=2)
plt.plot(mae_losses, label='MAE', linewidth=2)
plt.plot(huber_losses, label='Huber', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Convergence Speed Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.savefig('convergence_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Exercises

### Exercise 1: Implement Tversky Loss

The Tversky loss is a generalization of Dice loss for segmentation, with controllable false positive/negative trade-off:

```
Tversky = (TP) / (TP + α*FP + β*FN)

where:
- TP = true positives
- FP = false positives
- FN = false negatives
- α, β control the trade-off (typically α + β = 1)
```

**Task:**
1. Implement `TverskyLoss` as a PyTorch `nn.Module`
2. Test with α=0.3, β=0.7 (focus on reducing false negatives)
3. Compare with DiceLoss on a binary segmentation task

**Starter Code:**

```python
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        # TODO: Initialize parameters
        pass

    def forward(self, pred, target):
        # TODO: Implement Tversky loss
        # Hint: Calculate TP, FP, FN
        pass
```

### Exercise 2: Multi-Scale Perceptual Loss

Implement a perceptual loss that compares features at multiple scales (different layers of a pretrained network).

**Task:**
1. Extract features from layers `['relu2_2', 'relu3_3', 'relu4_3']` of VGG16
2. Compute MSE between predicted and target features at each layer
3. Combine losses with learnable weights
4. Test on image reconstruction task

**Questions:**
- How do different layers affect the loss?
- What happens if you only use early layers vs. only deep layers?

### Exercise 3: Adaptive Loss Balancing

Implement an adaptive loss balancing scheme for a multi-task learning scenario.

**Task:**
You're training a model for autonomous driving with three tasks:
1. Semantic segmentation (CrossEntropyLoss)
2. Depth estimation (L1Loss)
3. Object detection (FocalLoss)

Implement:
1. Uncertainty-based weighting (Section 7.1)
2. Track the learned weights over training
3. Visualize how weights change as training progresses

**Starter Code:**

```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define three task heads
        pass

    def forward(self, x):
        # TODO: Return predictions for all three tasks
        pass

# Training loop
for epoch in range(num_epochs):
    # TODO:
    # 1. Forward pass
    # 2. Compute three losses
    # 3. Apply uncertainty weighting
    # 4. Backward and optimize
    pass
```

**Questions:**
- Which task gets the highest weight? Why?
- How quickly do weights stabilize?
- What happens if you initialize weights differently?

---

## References

1. **Loss Functions Survey:**
   - Janocha, K., & Czarnecki, W. M. (2017). "On Loss Functions for Deep Neural Networks in Classification"

2. **Regression Losses:**
   - Huber, P. J. (1964). "Robust Estimation of a Location Parameter"

3. **Classification Losses:**
   - Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection" (RetinaNet)
   - Müller, R., et al. (2019). "When Does Label Smoothing Help?"

4. **Metric Learning:**
   - Schroff, F., et al. (2015). "FaceNet: A Unified Embedding for Face Recognition and Clustering" (Triplet Loss)
   - Chopra, S., et al. (2005). "Learning a Similarity Metric Discriminatively" (Contrastive Loss)
   - Chen, T., et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR, NT-Xent)

5. **Segmentation Losses:**
   - Milletari, F., et al. (2016). "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" (Dice Loss)
   - Rezatofighi, H., et al. (2019). "Generalized Intersection over Union" (GIoU)

6. **Generative Models:**
   - Goodfellow, I., et al. (2014). "Generative Adversarial Networks"
   - Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes"
   - Johnson, J., et al. (2016). "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"

7. **Multi-Task Learning:**
   - Kendall, A., et al. (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
   - Chen, Z., et al. (2018). "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"

8. **PyTorch Documentation:**
   - https://pytorch.org/docs/stable/nn.html#loss-functions
   - https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

9. **Additional Resources:**
   - Murphy, K. P. (2022). "Probabilistic Machine Learning: An Introduction" (Chapter on Loss Functions)
   - Goodfellow, I., et al. (2016). "Deep Learning" (Chapter 8: Optimization for Training Deep Models)
