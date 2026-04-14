"""
Training Loop - Examples
========================
Lesson 08: Training Loop

Demonstrates:
  1. Complete training + validation loop
  2. Early stopping
  3. Learning curve tracking and visualization
  4. Gradient clipping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


def create_data(n=1000, d=20, c=5):
    """Create synthetic classification data."""
    torch.manual_seed(42)
    X = torch.randn(n, d)
    y = torch.randint(0, c, (n,))
    return X, y


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def example_full_training():
    """Complete training with validation and early stopping."""
    print("=" * 60)
    print("Full Training Loop Example")
    print("=" * 60)

    device = torch.device('cpu')

    X, y = create_data()
    dataset = TensorDataset(X, y)
    train_set, val_set = random_split(dataset, [800, 200])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    model = nn.Sequential(
        nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 5),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=10)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(50):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer,
                                         loss_fn, device)
        v_loss, v_acc = validate(model, val_loader, loss_fn, device)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {t_loss:.4f}/{t_acc:.2%} | "
                  f"Val: {v_loss:.4f}/{v_acc:.2%}")

        early_stop(v_loss)
        if early_stop.should_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nFinal train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Best val loss: {early_stop.best_loss:.4f}")


if __name__ == "__main__":
    example_full_training()
