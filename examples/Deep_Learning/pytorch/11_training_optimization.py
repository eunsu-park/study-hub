"""
11. Training Optimization

Implements hyperparameter tuning, Mixed Precision, Gradient Accumulation, and more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import time

print("=" * 60)
print("PyTorch Training Optimization")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ============================================
# 1. Reproducibility Settings
# ============================================
print("\n[1] Reproducibility Settings")
print("-" * 40)

def set_seed(seed=42):
    """Set seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
print("Seed set: 42")


# ============================================
# 2. Sample Model and Data
# ============================================
print("\n[2] Sample Model and Data")
print("-" * 40)

class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Dummy data
X_train = torch.randn(1000, 1, 28, 28)
y_train = torch.randint(0, 10, (1000,))
X_val = torch.randn(200, 1, 28, 28)
y_val = torch.randint(0, 10, (200,))

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

print(f"Training data: {len(train_dataset)}")
print(f"Validation data: {len(val_dataset)}")


# ============================================
# 3. Learning Rate Scheduler
# ============================================
print("\n[3] Learning Rate Scheduler")
print("-" * 40)

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Warmup + Cosine Decay scheduler"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Test
model = SimpleNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=100, total_steps=1000)

lrs = []
for step in range(1000):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

print(f"Warmup phase (0-100): {lrs[0]:.6f} -> {lrs[99]:.6f}")
print(f"Decay phase (100-1000): {lrs[100]:.6f} -> {lrs[-1]:.6f}")


# ============================================
# 4. Early Stopping
# ============================================
print("\n[4] Early Stopping")
print("-" * 40)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_loss = val_loss
            self._save_checkpoint(model)
            self.counter = 0

    def _save_checkpoint(self, model):
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# Test
early_stopping = EarlyStopping(patience=3)
losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88]

print("Early stopping simulation:")
for epoch, loss in enumerate(losses):
    early_stopping(loss, model)
    status = "STOP" if early_stopping.early_stop else f"counter={early_stopping.counter}"
    print(f"  Epoch {epoch}: loss={loss:.2f}, {status}")
    if early_stopping.early_stop:
        break


# ============================================
# 5. Gradient Accumulation
# ============================================
print("\n[5] Gradient Accumulation")
print("-" * 40)

def train_with_accumulation(model, train_loader, optimizer, accumulation_steps=4):
    """Train with Gradient Accumulation"""
    model.train()
    optimizer.zero_grad()
    total_loss = 0

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = F.cross_entropy(output, target)
        loss = loss / accumulation_steps  # Scaling
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(train_loader)

# Test
model = SimpleNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

loss = train_with_accumulation(model, train_loader, optimizer, accumulation_steps=4)
print(f"Accumulation training loss: {loss:.4f}")
print(f"Effective batch size: 32 x 4 = 128")


# ============================================
# 6. Mixed Precision Training
# ============================================
print("\n[6] Mixed Precision Training")
print("-" * 40)

if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler

    def train_with_amp(model, train_loader, optimizer, scaler):
        """Mixed Precision training"""
        model.train()
        total_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with autocast():
                output = model(data)
                loss = F.cross_entropy(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    model = SimpleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()

    loss = train_with_amp(model, train_loader, optimizer, scaler)
    print(f"AMP training loss: {loss:.4f}")
else:
    print("CUDA not available - skipping AMP")


# ============================================
# 7. Gradient Clipping
# ============================================
print("\n[7] Gradient Clipping")
print("-" * 40)

def train_with_clipping(model, train_loader, optimizer, max_norm=1.0):
    """Train with Gradient Clipping"""
    model.train()
    total_loss = 0
    grad_norms = []

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Record gradient norm (before clipping)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms.append(total_norm ** 0.5)

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader), grad_norms

model = SimpleNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss, norms = train_with_clipping(model, train_loader, optimizer, max_norm=1.0)
print(f"Clipping training loss: {loss:.4f}")
print(f"Mean gradient norm: {np.mean(norms):.4f}")
print(f"Max gradient norm: {np.max(norms):.4f}")


# ============================================
# 8. Hyperparameter Search (Random Search)
# ============================================
print("\n[8] Hyperparameter Search")
print("-" * 40)

def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

def train_with_config(lr, batch_size, dropout, epochs=5):
    """Train with given configuration"""
    set_seed(42)

    model = SimpleNet(dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()

    return evaluate(model, val_loader)

# Random Search
import random
print("Running Random Search...")

best_acc = 0
best_config = None
results = []

for trial in range(5):
    lr = 10 ** random.uniform(-4, -2)
    batch_size = random.choice([32, 64, 128])
    dropout = random.uniform(0.2, 0.5)

    acc = train_with_config(lr, batch_size, dropout, epochs=3)
    results.append((lr, batch_size, dropout, acc))

    if acc > best_acc:
        best_acc = acc
        best_config = (lr, batch_size, dropout)

    print(f"  Trial {trial+1}: lr={lr:.6f}, bs={batch_size}, dropout={dropout:.2f} -> acc={acc:.4f}")

print(f"\nBest config: lr={best_config[0]:.6f}, bs={best_config[1]}, dropout={best_config[2]:.2f}")
print(f"Best accuracy: {best_acc:.4f}")


# ============================================
# 9. Full Training Pipeline
# ============================================
print("\n[9] Full Training Pipeline")
print("-" * 40)

def full_training_pipeline(config):
    """Full training pipeline with optimization techniques"""
    set_seed(config['seed'])

    # Model
    model = SimpleNet(dropout=config['dropout']).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Scheduler
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'])

    # AMP (if CUDA)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Training
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.cross_entropy(output, target).item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    return model, history

# Configuration
config = {
    'seed': 42,
    'lr': 1e-3,
    'batch_size': 64,
    'epochs': 20,
    'dropout': 0.3,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'patience': 5,
    'max_grad_norm': 1.0
}

print("Running full pipeline...")
model, history = full_training_pipeline(config)
print(f"\nFinal validation accuracy: {history['val_acc'][-1]:.4f}")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Training Optimization Summary")
print("=" * 60)

summary = """
Key Techniques:

1. Learning Rate Scheduling
   - Warmup: Initial stabilization
   - Cosine Decay: Gradual reduction
   - OneCycleLR: Per-batch adjustment

2. Mixed Precision (AMP)
   - Saves memory, improves speed
   - autocast() + GradScaler()

3. Gradient Accumulation
   - Small batch -> large batch effect
   - loss /= accumulation_steps

4. Gradient Clipping
   - Prevents gradient explosion
   - clip_grad_norm_(params, max_norm)

5. Early Stopping
   - Prevents overfitting
   - Restores best weights

Recommended settings:
    optimizer = AdamW(lr=1e-4, weight_decay=0.01)
    scheduler = OneCycleLR(max_lr=1e-3)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=10)
"""
print(summary)
print("=" * 60)
