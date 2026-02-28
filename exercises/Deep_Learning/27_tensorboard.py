"""
Exercises for Lesson 27: TensorBoard
Topic: Deep_Learning

Solutions to practice problems from the lesson.
Note: TensorBoard logging is demonstrated in code; actual visualization
requires running `tensorboard --logdir=runs` separately.
"""

import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# TensorBoard import with fallback
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


# === Exercise 1: Implement Basic Logging ===
# Problem: Log training/validation metrics, LR, and sample images.

def exercise_1():
    """Basic TensorBoard logging: loss, accuracy, LR, and images."""
    torch.manual_seed(42)

    class MNISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(32 * 7 * 7, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    # Synthetic MNIST-like data
    X_train = torch.randn(512, 1, 28, 28)
    y_train = torch.randint(0, 10, (512,))
    X_val = torch.randn(128, 1, 28, 28)
    y_val = torch.randint(0, 10, (128,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    model = MNISTNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    log_dir = tempfile.mkdtemp(prefix='tb_ex1_')
    writer = SummaryWriter(log_dir) if HAS_TB else None

    # Log sample images
    if writer:
        writer.add_images('train/sample_images', X_train[:8], 0)

    for epoch in range(5):
        # Train
        model.train()
        train_loss = 0.0
        correct = total = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = nn.CrossEntropyLoss()(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (pred.argmax(1) == yb).sum().item()
            total += len(yb)

        train_loss /= len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = nn.CrossEntropyLoss()(val_pred, y_val).item()
            val_acc = (val_pred.argmax(1) == y_val).float().mean().item()

        # Log metrics
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step()
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, "
              f"val_acc={val_acc:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

    if writer:
        writer.close()
        print(f"\n  TensorBoard logs saved to: {log_dir}")
    print("  Run: tensorboard --logdir=<log_dir>")


# === Exercise 2: Model Analysis ===
# Problem: Visualize weight histograms, feature maps, and Grad-CAM.

def exercise_2():
    """Model analysis: weight histograms and feature maps."""
    torch.manual_seed(42)

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(16 * 7 * 7, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = SimpleCNN()

    # Train briefly
    X = torch.randn(256, 1, 28, 28)
    y = torch.randint(0, 10, (256,))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(5):
        loss = nn.CrossEntropyLoss()(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Weight histogram analysis
    print("  Weight statistics per layer:")
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"    {name}: mean={param.data.mean():.4f}, "
                  f"std={param.data.std():.4f}, "
                  f"shape={list(param.shape)}")

    # Feature map extraction
    feature_maps = {}

    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()
        return hook

    model.conv1.register_forward_hook(hook_fn('conv1'))
    model.conv2.register_forward_hook(hook_fn('conv2'))

    model.eval()
    with torch.no_grad():
        model(X[:1])

    for name, fmap in feature_maps.items():
        print(f"    {name} feature map: shape={list(fmap.shape)}, "
              f"mean={fmap.mean():.4f}, std={fmap.std():.4f}")

    # Simple Grad-CAM
    model.eval()
    x_test = X[:1].requires_grad_(True)
    output = model(x_test)
    target_class = output.argmax(1).item()
    output[0, target_class].backward()

    # Get gradients at conv2 (would need hooks in practice)
    print(f"\n  Grad-CAM target class: {target_class}")
    print("  In practice: hook conv2 gradients, GAP them, weight feature maps.")
    print("  Highlights regions the model considers important for its prediction.")


# === Exercise 3: Hyperparameter Tuning ===
# Problem: Grid search over LR, batch size, dropout.

def exercise_3():
    """Hyperparameter grid search with logging."""
    torch.manual_seed(42)

    X = torch.randn(800, 20)
    y = (X[:, :5].sum(dim=1) > 0).long()
    X_train, y_train = X[:600], y[:600]
    X_val, y_val = X[600:], y[600:]

    class TunableNet(nn.Module):
        def __init__(self, dropout=0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(20, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 2),
            )

        def forward(self, x):
            return self.net(x)

    # Grid search
    lrs = [0.01, 0.001]
    batch_sizes = [32, 64]
    dropouts = [0.0, 0.3]

    best_acc = 0
    best_config = None

    print(f"  {'LR':>6} {'BS':>4} {'Drop':>5} {'Val Acc':>8}")
    print(f"  {'-'*6} {'-'*4} {'-'*5} {'-'*8}")

    for lr in lrs:
        for bs in batch_sizes:
            for dropout in dropouts:
                torch.manual_seed(42)
                model = TunableNet(dropout=dropout)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                loader = DataLoader(TensorDataset(X_train, y_train),
                                    batch_size=bs, shuffle=True)

                for epoch in range(20):
                    model.train()
                    for xb, yb in loader:
                        loss = nn.CrossEntropyLoss()(model(xb), yb)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()

                print(f"  {lr:6.3f} {bs:4d} {dropout:5.1f} {val_acc:8.4f}")

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_config = (lr, bs, dropout)

    print(f"\n  Best config: lr={best_config[0]}, bs={best_config[1]}, "
          f"dropout={best_config[2]}, val_acc={best_acc:.4f}")
    print("  In TensorBoard HParams: compare runs interactively.")


if __name__ == "__main__":
    print("=== Exercise 1: Basic Logging ===")
    exercise_1()
    print("\n=== Exercise 2: Model Analysis ===")
    exercise_2()
    print("\n=== Exercise 3: Hyperparameter Tuning ===")
    exercise_3()
    print("\nAll exercises completed!")
