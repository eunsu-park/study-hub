# Loss Functions and Optimizers

**Previous**: [nn.Module](./05_nn_Module.md) | **Next**: [Dataset and DataLoader](./07_Dataset_and_DataLoader.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Choose the appropriate loss function for classification, regression, and ranking tasks
2. Implement and use `nn.CrossEntropyLoss`, `nn.MSELoss`, `nn.BCEWithLogitsLoss`, and others
3. Understand the mathematical formulation behind each loss function
4. Configure optimizers (`SGD`, `Adam`, `AdamW`) with proper hyperparameters
5. Apply learning rate schedulers for improved training convergence
6. Use per-parameter optimization (different learning rates for different layers)
7. Explain the relationship between loss, gradients, and parameter updates

---

Loss functions measure how wrong a model's predictions are. Optimizers use the gradients of the loss to update model parameters. Together, they form the core of the training process.

---

## 1. Loss Functions for Classification

### 1.1 CrossEntropyLoss

The workhorse loss for multi-class classification. Combines `LogSoftmax` and `NLLLoss`:

```python
import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()

# Raw logits (NOT softmax outputs)
logits = torch.tensor([[2.0, 1.0, 0.1],    # sample 1: predicts class 0
                        [0.1, 2.0, 0.3]])   # sample 2: predicts class 1
targets = torch.tensor([0, 1])               # true classes

loss = loss_fn(logits, targets)
print(loss)  # tensor(0.4170)

# Math: -log(softmax(logit)[target_class])
# For sample 1: -log(softmax([2.0, 1.0, 0.1])[0]) = -log(0.6590) = 0.4170
```

**Key details**:
- Input: raw logits of shape `[batch, num_classes]` -- do NOT apply softmax before
- Target: class indices of shape `[batch]` (integers, not one-hot)
- Supports class weights: `nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.0]))`
- Supports label smoothing: `nn.CrossEntropyLoss(label_smoothing=0.1)`

### 1.2 BCEWithLogitsLoss

For binary or multi-label classification:

```python
loss_fn = nn.BCEWithLogitsLoss()

# Raw logits (single value per sample for binary)
logits = torch.tensor([0.5, -1.0, 2.0])
targets = torch.tensor([1.0, 0.0, 1.0])

loss = loss_fn(logits, targets)
print(loss)  # tensor(0.3780)

# Multi-label: each sample can belong to multiple classes
logits = torch.randn(4, 5)          # 4 samples, 5 labels
targets = torch.zeros(4, 5)         # binary targets
targets[0, [0, 2]] = 1.0            # sample 0 has labels 0 and 2
loss = loss_fn(logits, targets)
```

### 1.3 NLLLoss (Negative Log-Likelihood)

Used with log-probabilities (log-softmax output):

```python
loss_fn = nn.NLLLoss()

# Must apply log_softmax yourself
logits = torch.randn(3, 5)
log_probs = torch.nn.functional.log_softmax(logits, dim=1)
targets = torch.tensor([0, 3, 2])

loss = loss_fn(log_probs, targets)
# Equivalent to: nn.CrossEntropyLoss()(logits, targets)
```

---

## 2. Loss Functions for Regression

### 2.1 MSELoss (L2 Loss)

```python
loss_fn = nn.MSELoss()

predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
targets = torch.tensor([3.0, -0.5, 2.0, 7.0])

loss = loss_fn(predictions, targets)
print(loss)  # tensor(0.3750)
# Math: mean((2.5-3)^2 + (0-(-0.5))^2 + (2-2)^2 + (8-7)^2) / 4

# Reduction options
nn.MSELoss(reduction='mean')   # default: mean over all elements
nn.MSELoss(reduction='sum')    # sum of squared errors
nn.MSELoss(reduction='none')   # per-element loss (no reduction)
```

### 2.2 L1Loss (MAE)

```python
loss_fn = nn.L1Loss()

predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
targets = torch.tensor([3.0, -0.5, 2.0, 7.0])

loss = loss_fn(predictions, targets)
print(loss)  # tensor(0.5000)
# Math: mean(|2.5-3| + |0-(-0.5)| + |2-2| + |8-7|) / 4
```

### 2.3 SmoothL1Loss (Huber Loss)

Combines L1 and L2 -- quadratic for small errors, linear for large:

```python
loss_fn = nn.SmoothL1Loss(beta=1.0)

predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
targets = torch.tensor([3.0, -0.5, 2.0, 7.0])

loss = loss_fn(predictions, targets)
```

---

## 3. Other Useful Loss Functions

### 3.1 Loss Function Selection Guide

| Task | Loss Function | Input | Target |
|------|-------------|-------|--------|
| Multi-class classification | `CrossEntropyLoss` | Logits `[B, C]` | Class indices `[B]` |
| Binary classification | `BCEWithLogitsLoss` | Logits `[B]` | Float 0/1 `[B]` |
| Multi-label classification | `BCEWithLogitsLoss` | Logits `[B, C]` | Float 0/1 `[B, C]` |
| Regression | `MSELoss` | Predictions `[B]` | Targets `[B]` |
| Robust regression | `SmoothL1Loss` | Predictions `[B]` | Targets `[B]` |
| Ranking / similarity | `CosineEmbeddingLoss` | Pairs of embeddings | +1/-1 labels |
| Contrastive learning | `TripletMarginLoss` | Anchor, positive, negative | N/A |

### 3.2 Custom Loss Functions

```python
# Option 1: plain function
def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    ce = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()

# Option 2: nn.Module (for stateful losses or when using in Sequential)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()
```

---

## 4. Optimizers

### 4.1 SGD (Stochastic Gradient Descent)

```python
import torch.optim as optim

model = nn.Linear(784, 10)

# Basic SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD with momentum (recommended in practice)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# SGD with momentum and weight decay (L2 regularization)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                      weight_decay=1e-4)

# Nesterov momentum (usually better than standard momentum)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                      nesterov=True, weight_decay=1e-4)
```

### 4.2 Adam

```python
# Adam: adaptive learning rates per parameter
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Adam with weight decay fix (AdamW -- recommended over Adam)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Custom betas (momentum parameters)
optimizer = optim.Adam(model.parameters(), lr=1e-3,
                       betas=(0.9, 0.999), eps=1e-8)
```

### 4.3 Other Optimizers

```python
# RMSprop (good for RNNs)
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

# Adagrad (good for sparse data)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# LBFGS (second-order, for small models)
optimizer = optim.LBFGS(model.parameters(), lr=1)
```

### 4.4 Optimizer Selection Guide

| Optimizer | Best For | Default LR |
|-----------|----------|------------|
| `SGD + momentum` | CNN training, when tuning LR carefully | 0.01 - 0.1 |
| `Adam` | Quick prototyping, most tasks | 1e-3 |
| `AdamW` | Transformers, when using weight decay | 1e-3 to 5e-5 |
| `RMSprop` | RNNs, reinforcement learning | 1e-3 |

---

## 5. The Optimizer Step

### 5.1 Basic Training Step

```python
model = nn.Linear(784, 10)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))

# THE training step:
optimizer.zero_grad()       # 1. Zero gradients from previous step
output = model(x)           # 2. Forward pass
loss = loss_fn(output, y)   # 3. Compute loss
loss.backward()             # 4. Backward pass (compute gradients)
optimizer.step()            # 5. Update parameters

print(f"Loss: {loss.item():.4f}")
```

### 5.2 Per-Parameter Options

```python
class TwoPartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(784, 256)  # pretrained, slow LR
        self.head = nn.Linear(256, 10)       # new, fast LR

model = TwoPartModel()

# Different learning rates for different parts
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.head.parameters(), 'lr': 1e-3},
], weight_decay=0.01)
```

### 5.3 Gradient Accumulation

For simulating larger batch sizes:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (x, y) in enumerate(dataloader):
    output = model(x)
    loss = loss_fn(output, y) / accumulation_steps  # normalize
    loss.backward()  # gradients accumulate

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 6. Learning Rate Schedulers

### 6.1 StepLR

```python
from torch.optim.lr_scheduler import StepLR

optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# LR: 0.01 for epochs 0-9, 0.001 for 10-19, 0.0001 for 20-29, ...

for epoch in range(30):
    train_one_epoch(model, optimizer)
    scheduler.step()  # call AFTER optimizer.step()
    print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()}")
```

### 6.2 CosineAnnealingLR

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
# Cosine decay from initial LR to eta_min over T_max epochs
```

### 6.3 OneCycleLR

```python
from torch.optim.lr_scheduler import OneCycleLR

# One-cycle policy: warmup then cosine decay (per-step scheduler)
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    total_steps=len(dataloader) * num_epochs
)

for epoch in range(num_epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()  # call per batch, not per epoch
```

### 6.4 ReduceLROnPlateau

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=5, verbose=True)

for epoch in range(100):
    train_loss = train_one_epoch(model, optimizer)
    val_loss = validate(model)
    scheduler.step(val_loss)  # pass the monitored metric
```

### 6.5 Warmup + Decay (Manual)

```python
from torch.optim.lr_scheduler import LambdaLR

warmup_steps = 1000
total_steps = 10000

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps  # linear warmup
    else:
        # cosine decay after warmup
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

scheduler = LambdaLR(optimizer, lr_lambda)
```

---

## 7. Putting It All Together

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=20)

# Simulated training
for epoch in range(20):
    model.train()
    x = torch.randn(64, 784)
    y = torch.randint(0, 10, (64,))

    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Accuracy
    with torch.no_grad():
        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean()

    print(f"Epoch {epoch+1:2d} | Loss: {loss.item():.4f} | "
          f"Acc: {acc.item():.2%} | LR: {scheduler.get_last_lr()[0]:.6f}")
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| CrossEntropyLoss | Multi-class classification; input is raw logits, NOT softmax |
| BCEWithLogitsLoss | Binary/multi-label; input is raw logits |
| MSELoss | Regression; L2 loss |
| Adam/AdamW | Default optimizer choice; AdamW for weight decay |
| SGD+momentum | Better generalization with careful LR tuning |
| zero_grad -> forward -> loss -> backward -> step | The sacred training sequence |
| LR schedulers | Adjust LR during training for better convergence |
| Per-parameter groups | Different LR/weight_decay for backbone vs head |

---

**Next**: [Dataset and DataLoader](./07_Dataset_and_DataLoader.md) -- Building efficient data pipelines.
