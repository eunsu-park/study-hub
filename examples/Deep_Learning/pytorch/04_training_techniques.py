"""
04. Training Techniques - PyTorch Version

Implements various optimization techniques and regularization in PyTorch.
Compare with the NumPy version (examples/numpy/04_training_techniques.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("PyTorch Training Techniques")
print("=" * 60)


# ============================================
# 1. Optimizer Comparison
# ============================================
print("\n[1] Optimizer Comparison")
print("-" * 40)

# Simple model definition
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# XOR data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

def train_with_optimizer(optimizer_class, **kwargs):
    """Train with a given optimizer"""
    torch.manual_seed(42)
    model = SimpleNet()
    optimizer = optimizer_class(model.parameters(), **kwargs)
    criterion = nn.BCELoss()

    losses = []
    for epoch in range(500):
        pred = model(X)
        loss = criterion(pred, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses

# Test various optimizers
optimizers = {
    'SGD (lr=0.5)': (torch.optim.SGD, {'lr': 0.5}),
    'SGD+Momentum': (torch.optim.SGD, {'lr': 0.5, 'momentum': 0.9}),
    'Adam': (torch.optim.Adam, {'lr': 0.01}),
    'RMSprop': (torch.optim.RMSprop, {'lr': 0.01}),
}

results = {}
for name, (opt_class, params) in optimizers.items():
    losses = train_with_optimizer(opt_class, **params)
    results[name] = losses
    print(f"{name}: Final loss = {losses[-1]:.6f}")

# Visualization
plt.figure(figsize=(10, 5))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Optimizer Comparison')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('optimizer_comparison.png', dpi=100)
plt.close()
print("Plot saved: optimizer_comparison.png")


# ============================================
# 2. Learning Rate Schedulers
# ============================================
print("\n[2] Learning Rate Schedulers")
print("-" * 40)

# Scheduler test
def test_scheduler(scheduler_class, **kwargs):
    torch.manual_seed(42)
    model = SimpleNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = scheduler_class(optimizer, **kwargs)

    lrs = []
    for epoch in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    return lrs

schedulers = {
    'StepLR': (torch.optim.lr_scheduler.StepLR, {'step_size': 20, 'gamma': 0.5}),
    'ExponentialLR': (torch.optim.lr_scheduler.ExponentialLR, {'gamma': 0.95}),
    'CosineAnnealingLR': (torch.optim.lr_scheduler.CosineAnnealingLR, {'T_max': 50}),
}

plt.figure(figsize=(10, 5))
for name, (sched_class, params) in schedulers.items():
    lrs = test_scheduler(sched_class, **params)
    plt.plot(lrs, label=name)
    print(f"{name}: Start {lrs[0]:.4f} -> End {lrs[-1]:.4f}")

plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedulers')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('lr_schedulers.png', dpi=100)
plt.close()
print("Plot saved: lr_schedulers.png")


# ============================================
# 3. Dropout
# ============================================
print("\n[3] Dropout")
print("-" * 40)

class NetWithDropout(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Check Dropout effect
model = NetWithDropout(dropout_p=0.5)
x_test = torch.randn(1, 2)

model.train()
print("Train mode (Dropout active):")
for i in range(3):
    out = model.fc1(x_test)
    out = F.relu(out)
    out = model.dropout(out)
    print(f"  Attempt {i+1}: Active neurons = {(out != 0).sum().item()}/32")

model.eval()
print("\nEval mode (Dropout inactive):")
out = model.fc1(x_test)
out = F.relu(out)
out = model.dropout(out)  # Passes through in eval mode
print(f"  Active neurons = {(out != 0).sum().item()}/32")


# ============================================
# 4. Batch Normalization
# ============================================
print("\n[4] Batch Normalization")
print("-" * 40)

class NetWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = torch.sigmoid(self.fc2(x))
        return x

bn_model = NetWithBatchNorm()
print(f"BatchNorm1d parameters:")
print(f"  weight (gamma): {bn_model.bn1.weight.shape}")
print(f"  bias (beta): {bn_model.bn1.bias.shape}")
print(f"  running_mean: {bn_model.bn1.running_mean.shape}")
print(f"  running_var: {bn_model.bn1.running_var.shape}")

# Train vs eval mode
x_batch = torch.randn(32, 2)

bn_model.train()
out_train = bn_model.fc1(x_batch)
out_train = bn_model.bn1(out_train)
print(f"\nTrain mode - output statistics:")
print(f"  mean: {out_train.mean(dim=0)[:3].tolist()}")
print(f"  std: {out_train.std(dim=0)[:3].tolist()}")

bn_model.eval()
out_eval = bn_model.fc1(x_batch)
out_eval = bn_model.bn1(out_eval)
print(f"Eval mode - output statistics:")
print(f"  mean: {out_eval.mean(dim=0)[:3].tolist()}")


# ============================================
# 5. Weight Decay (L2 Regularization)
# ============================================
print("\n[5] Weight Decay")
print("-" * 40)

def train_with_weight_decay(weight_decay):
    torch.manual_seed(42)
    model = SimpleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    for epoch in range(500):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check weight magnitude
    weight_norm = sum(p.norm().item() for p in model.parameters())
    return loss.item(), weight_norm

for wd in [0, 0.01, 0.1]:
    loss, w_norm = train_with_weight_decay(wd)
    print(f"Weight Decay={wd}: Loss={loss:.4f}, Weight norm={w_norm:.4f}")


# ============================================
# 6. Early Stopping
# ============================================
print("\n[6] Early Stopping")
print("-" * 40)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0

# Demo (simulated validation loss)
early_stopping = EarlyStopping(patience=5)
val_losses = [1.0, 0.9, 0.85, 0.8, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87]

model = SimpleNet()
for epoch, val_loss in enumerate(val_losses):
    early_stopping(val_loss, model)
    status = "STOP" if early_stopping.early_stop else f"patience={early_stopping.counter}"
    print(f"Epoch {epoch+1}: val_loss={val_loss:.2f}, {status}")
    if early_stopping.early_stop:
        break


# ============================================
# 7. Full Training Example
# ============================================
print("\n[7] Full Training Example")
print("-" * 40)

# Generate a larger dataset
np.random.seed(42)
n_samples = 200

# Circular data (nonlinear problem)
theta = np.random.uniform(0, 2*np.pi, n_samples)
r = np.random.uniform(0, 1, n_samples)
X_train = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
y_train = (r > 0.5).astype(np.float32)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# Validation data
X_val = X_train[:40]
y_val = y_train[:40]
X_train = X_train[40:]
y_train = y_train[40:]

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# Model initialization
torch.manual_seed(42)
model = FullModel()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
early_stopping = EarlyStopping(patience=20)

# Training
train_losses = []
val_losses = []

for epoch in range(200):
    # Train
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        pred = model(X_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # Validate
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()
        val_losses.append(val_loss)

    # Update scheduler
    scheduler.step(val_loss)

    # Check early stopping
    early_stopping(val_loss, model)

    if (epoch + 1) % 40 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: train={train_losses[-1]:.4f}, val={val_loss:.4f}, lr={lr:.6f}")

    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Restore best model
if early_stopping.best_model:
    model.load_state_dict(early_stopping.best_model)

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training with Regularization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('full_training.png', dpi=100)
plt.close()
print("Plot saved: full_training.png")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Training Techniques Summary")
print("=" * 60)

summary = """
Recommended defaults:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    EarlyStopping(patience=10)

Regularization combinations:
    - Dropout (0.2~0.5): Prevents overfitting
    - BatchNorm: Stabilizes training
    - Weight Decay (1e-4~1e-2): Limits weight magnitude

Training loop checklist:
    1. model.train() / model.eval() mode switching
    2. optimizer.zero_grad() call
    3. loss.backward()
    4. optimizer.step()
    5. scheduler.step() (at end of epoch)
    6. EarlyStopping check
"""
print(summary)
print("=" * 60)
