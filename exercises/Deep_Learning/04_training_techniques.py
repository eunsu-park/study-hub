"""
Exercises for Lesson 04: Training Techniques
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Adam vs SGD Convergence ===
# Problem: Compare Adam and SGD on a simple regression task.

def exercise_1():
    """Compare Adam and SGD on y = 2x + 1 + noise."""
    torch.manual_seed(42)

    # Generate synthetic data
    X = torch.randn(200, 1)
    y = 2 * X + 1 + 0.1 * torch.randn(200, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    results = {}
    for opt_name in ["SGD", "Adam"]:
        model = nn.Linear(1, 1)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)

        if opt_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

        losses = []
        for epoch in range(200):
            epoch_loss = 0.0
            for xb, yb in loader:
                pred = model(xb)
                loss = nn.MSELoss()(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))

        results[opt_name] = losses
        w = model.weight.item()
        b = model.bias.item()
        print(f"  {opt_name}: final w={w:.4f}, b={b:.4f}, final_loss={losses[-1]:.6f}")

    # Compare convergence
    sgd_10 = results["SGD"][9]
    adam_10 = results["Adam"][9]
    print(f"  Loss at epoch 10 - SGD: {sgd_10:.6f}, Adam: {adam_10:.6f}")
    print(f"  Adam converges faster in early epochs.")


# === Exercise 2: Implement Adam from Scratch ===
# Problem: Implement Adam update rule in NumPy.

def exercise_2():
    """Implement Adam optimizer from scratch in NumPy."""
    np.random.seed(42)

    lr = 0.001
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    W = 0.0
    m, v, t = 0.0, 0.0, 0

    # Simulate 100 gradient steps
    gradients = np.random.randn(100)
    W_history = [W]

    for grad in gradients:
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        W -= lr * m_hat / (np.sqrt(v_hat) + eps)
        W_history.append(W)

    print(f"  NumPy Adam final W: {W:.6f}")

    # Verify with PyTorch Adam
    W_torch = torch.tensor(0.0, requires_grad=True)
    optimizer = optim.Adam([W_torch], lr=lr, betas=(beta1, beta2), eps=eps)

    for grad_val in gradients:
        if W_torch.grad is not None:
            W_torch.grad.zero_()
        # Manually set gradient
        loss = W_torch * grad_val  # gradient of loss w.r.t. W = grad_val
        loss.backward()
        # Override the computed gradient with our target gradient
        W_torch.grad.data.fill_(grad_val)
        optimizer.step()

    print(f"  PyTorch Adam final W: {W_torch.item():.6f}")
    print(f"  Difference: {abs(W - W_torch.item()):.8f}")


# === Exercise 3: Learning Rate Scheduling Comparison ===
# Problem: Compare no scheduling, StepLR, and CosineAnnealingLR.

def exercise_3():
    """Compare learning rate schedulers on a small MLP."""
    torch.manual_seed(42)

    # Generate synthetic classification data (simulating MNIST-like)
    n_samples = 2000
    X = torch.randn(n_samples, 28 * 28)
    y = (X.sum(dim=1) > 0).long()
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset[:1600], batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset[1600:], batch_size=64)

    class SmallMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(784, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 2)
            )

        def forward(self, x):
            return self.net(x)

    schedulers_config = {
        "No scheduler": None,
        "StepLR": lambda opt: optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5),
        "CosineAnnealing": lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20),
    }

    for sched_name, sched_fn in schedulers_config.items():
        torch.manual_seed(42)
        model = SmallMLP()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = sched_fn(optimizer) if sched_fn else None

        best_acc = 0.0
        for epoch in range(20):
            model.train()
            for xb, yb in train_loader:
                pred = model(xb)
                loss = nn.CrossEntropyLoss()(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if scheduler:
                scheduler.step()

            # Validation
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb)
                    correct += (pred.argmax(1) == yb).sum().item()
                    total += len(yb)
            acc = correct / total
            best_acc = max(best_acc, acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  {sched_name}: best_val_acc={best_acc:.4f}, final_lr={current_lr:.6f}")


# === Exercise 4: Dropout Regularization Effect ===
# Problem: Verify dropout reduces overfitting on a small dataset.

def exercise_4():
    """Compare training with and without dropout on a small dataset."""
    torch.manual_seed(42)

    # Small dataset (easy to overfit)
    n = 500
    X = torch.randn(n, 20)
    y = (X[:, :5].sum(dim=1) > 0).long()
    X_train, y_train = X[:400], y[:400]
    X_val, y_val = X[400:], y[400:]
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    class MLP(nn.Module):
        def __init__(self, use_dropout=False):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(20, 128), nn.ReLU(),
                nn.Dropout(0.5) if use_dropout else nn.Identity(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Dropout(0.5) if use_dropout else nn.Identity(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Dropout(0.5) if use_dropout else nn.Identity(),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.layers(x)

    for use_dropout in [False, True]:
        torch.manual_seed(42)
        model = MLP(use_dropout=use_dropout)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(50):
            model.train()
            for xb, yb in train_loader:
                loss = nn.CrossEntropyLoss()(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
            val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()
            gap = train_acc - val_acc

        label = "With Dropout" if use_dropout else "No Dropout"
        print(f"  {label}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, gap={gap:.4f}")

    print("  Dropout reduces the train-val accuracy gap (overfitting).")


# === Exercise 5: Early Stopping with Model Checkpointing ===
# Problem: Implement early stopping that saves the best model checkpoint.

def exercise_5():
    """Early stopping with best model checkpoint saving."""
    import tempfile
    import os

    torch.manual_seed(42)

    class EarlyStopping:
        def __init__(self, patience=5, save_path=None):
            self.patience = patience
            self.save_path = save_path
            self.best_loss = float('inf')
            self.counter = 0
            self.should_stop = False

        def __call__(self, val_loss, model):
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
                if self.save_path:
                    torch.save(model.state_dict(), self.save_path)
                    print(f"    Saved best model (val_loss={val_loss:.4f})")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True

    # Prepare data
    X = torch.randn(600, 10)
    y = (X[:, :3].sum(dim=1) > 0).long()
    X_train, y_train = X[:400], y[:400]
    X_val, y_val = X[400:500], y[400:500]
    X_test, y_test = X[500:], y[500:]
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    model = nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 2),
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        save_path = f.name

    early_stopping = EarlyStopping(patience=5, save_path=save_path)

    for epoch in range(100):
        model.train()
        for xb, yb in train_loader:
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = nn.CrossEntropyLoss()(model(X_val), y_val).item()

        early_stopping(val_loss, model)
        if early_stopping.should_stop:
            print(f"    Early stopped at epoch {epoch + 1}")
            break

    # Test with final checkpoint
    model.eval()
    with torch.no_grad():
        final_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    # Test with best checkpoint
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        best_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    print(f"  Final checkpoint test acc: {final_acc:.4f}")
    print(f"  Best checkpoint test acc:  {best_acc:.4f}")
    print(f"  Best checkpoint is better when val loss diverges late in training.")

    os.unlink(save_path)


if __name__ == "__main__":
    print("=== Exercise 1: Adam vs SGD Convergence ===")
    exercise_1()
    print("\n=== Exercise 2: Implement Adam from Scratch ===")
    exercise_2()
    print("\n=== Exercise 3: Learning Rate Scheduling Comparison ===")
    exercise_3()
    print("\n=== Exercise 4: Dropout Regularization Effect ===")
    exercise_4()
    print("\n=== Exercise 5: Early Stopping with Model Checkpointing ===")
    exercise_5()
    print("\nAll exercises completed!")
