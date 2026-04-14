# Training Loop

**Previous**: [Dataset and DataLoader](./07_Dataset_and_DataLoader.md) | **Next**: [Model Saving and Loading](./09_Model_Saving_and_Loading.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write a complete training loop with forward pass, loss, backward, and optimizer step
2. Implement a validation loop with proper `eval()` mode and `no_grad()` context
3. Track and log training metrics (loss, accuracy) across epochs
4. Implement early stopping to prevent overfitting
5. Use progress bars and logging for training monitoring
6. Structure training code for readability and reuse
7. Handle common training issues (NaN loss, overfitting, underfitting)

---

The training loop is where everything comes together. This lesson teaches you to write clean, correct, and production-quality training code.

---

## 1. The Minimal Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setup
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Dummy data
X = torch.randn(200, 10)
y = torch.randint(0, 2, (200,))
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# Training loop
for epoch in range(10):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()           # 1. zero gradients
        output = model(batch_X)         # 2. forward pass
        loss = loss_fn(output, batch_y) # 3. compute loss
        loss.backward()                 # 4. backward pass
        optimizer.step()                # 5. update parameters

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## 2. Training + Validation Loop

### 2.1 Complete Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == batch_y).sum().item()
        total += batch_X.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        output = model(batch_X)
        loss = loss_fn(output, batch_y)

        total_loss += loss.item() * batch_X.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == batch_y).sum().item()
        total += batch_X.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy
```

### 2.2 Main Training Script

```python
# Hyperparameters
num_epochs = 50
batch_size = 64
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
X = torch.randn(1000, 20)
y = torch.randint(0, 5, (1000,))
dataset = TensorDataset(X, y)
train_set, val_set = random_split(dataset, [800, 200])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Model
model = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 5)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Training
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, loss_fn, device
    )
    val_loss, val_acc = validate(model, val_loader, loss_fn, device)

    print(f"Epoch {epoch+1:3d}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"  -> Saved best model (val_loss={val_loss:.4f})")
```

---

## 3. Early Stopping

```python
class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader,
                                             optimizer, loss_fn, device)
    val_loss, val_acc = validate(model, val_loader, loss_fn, device)

    early_stopping(val_loss)
    if early_stopping.should_stop:
        print(f"Early stopping at epoch {epoch+1}")
        break
```

---

## 4. Progress Tracking

### 4.1 Using tqdm

```python
from tqdm import tqdm

def train_one_epoch_tqdm(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_X, batch_y in pbar:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == batch_y).sum().item()
        total += batch_X.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss/total:.4f}",
            'acc': f"{correct/total:.4f}"
        })

    return total_loss / total, correct / total
```

### 4.2 Logging History

```python
history = {'train_loss': [], 'val_loss': [],
           'train_acc': [], 'val_acc': []}

for epoch in range(num_epochs):
    t_loss, t_acc = train_one_epoch(model, train_loader,
                                     optimizer, loss_fn, device)
    v_loss, v_acc = validate(model, val_loader, loss_fn, device)

    history['train_loss'].append(t_loss)
    history['val_loss'].append(v_loss)
    history['train_acc'].append(t_acc)
    history['val_acc'].append(v_acc)

# Plot learning curves
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['val_loss'], label='Validation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('Loss Curve')

ax2.plot(history['train_acc'], label='Train')
ax2.plot(history['val_acc'], label='Validation')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.set_title('Accuracy Curve')

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=100)
plt.close()
```

---

## 5. Training with Learning Rate Scheduler

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, loss_fn, device
    )
    val_loss, val_acc = validate(model, val_loader, loss_fn, device)

    scheduler.step()  # update LR after each epoch

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} | LR: {current_lr:.6f} | "
          f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
```

---

## 6. Regression Training Loop

```python
def train_regression(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    loss_fn = nn.MSELoss()

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device).float()

        optimizer.zero_grad()
        output = model(batch_X).squeeze()  # [B, 1] -> [B]
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)
        total += batch_X.size(0)

    return total_loss / total

@torch.no_grad()
def evaluate_regression(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    all_preds = []
    all_targets = []

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        output = model(batch_X).squeeze().cpu()
        all_preds.append(output)
        all_targets.append(batch_y)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets).float()
    mse = loss_fn(preds, targets).item()
    mae = (preds - targets).abs().mean().item()
    return mse, mae
```

---

## 7. Common Training Issues

### 7.1 NaN Loss

```python
# Check for NaN after each step
loss = loss_fn(output, batch_y)
if torch.isnan(loss):
    print("NaN loss detected!")
    print(f"  output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  any NaN in output: {torch.isnan(output).any()}")
    break

# Common causes and fixes:
# 1. Learning rate too high -> reduce LR
# 2. Numerical overflow -> use gradient clipping
# 3. Log of zero -> add epsilon (log(x + 1e-8))
# 4. Division by zero -> add epsilon to denominator
```

### 7.2 Gradient Clipping

```python
# Clip by global norm (most common)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# In training loop:
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()  # clip BEFORE step
```

### 7.3 Overfitting Symptoms and Fixes

```
Symptoms:
- Train loss decreases, val loss increases
- Train accuracy >> val accuracy

Fixes:
- Add dropout: nn.Dropout(p=0.5)
- Add weight decay: optimizer = Adam(params, weight_decay=1e-4)
- Reduce model size: fewer layers/neurons
- Data augmentation: more transforms
- Early stopping: stop when val loss stops improving
- More data: if possible
```

### 7.4 Underfitting Symptoms and Fixes

```
Symptoms:
- Train loss doesn't decrease (stays high)
- Both train and val accuracy are low

Fixes:
- Increase model capacity: more layers/neurons
- Increase learning rate
- Train for more epochs
- Remove excessive regularization
- Check data pipeline: are labels correct? Are transforms too aggressive?
```

---

## 8. Structured Training Template

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train(config):
    """Complete training function template."""

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader = DataLoader(config['train_dataset'],
                               batch_size=config['batch_size'],
                               shuffle=True, num_workers=4)
    val_loader = DataLoader(config['val_dataset'],
                             batch_size=config['batch_size'],
                             shuffle=False, num_workers=4)

    # Model, loss, optimizer
    model = config['model'].to(device)
    loss_fn = config['loss_fn']
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'],
                             weight_decay=config.get('weight_decay', 0.01))
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    early_stopping = EarlyStopping(patience=config.get('patience', 10))

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['epochs']):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        # Schedule
        scheduler.step()

        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch+1:3d} | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_checkpoint.pt')

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return model, history
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Training step | zero_grad -> forward -> loss -> backward -> step |
| Validation | `model.eval()` + `torch.no_grad()`, no optimizer step |
| Metrics | Accumulate loss * batch_size, divide by total samples |
| Early stopping | Stop when val loss hasn't improved for `patience` epochs |
| Gradient clipping | `clip_grad_norm_` before `optimizer.step()` |
| Learning curves | Plot train vs val loss/accuracy to diagnose problems |
| NaN debugging | Check output ranges, use gradient clipping, add epsilon |

---

**Next**: [Model Saving and Loading](./09_Model_Saving_and_Loading.md) -- Checkpointing, state dicts, and model export.
