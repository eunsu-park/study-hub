"""
Exercises for Lesson 25: Optimizers
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Implement and Compare Optimizers ===
# Problem: Implement RMSprop from scratch, compare with SGD and Adam on Rosenbrock.

def exercise_1():
    """Optimizer comparison on the Rosenbrock function."""

    def rosenbrock(x, y):
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def rosenbrock_grad(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
        dy = 200 * (y - x ** 2)
        return dx, dy

    # SGD with momentum
    def sgd_momentum(lr=0.0001, momentum=0.9, n_steps=5000):
        x, y = -1.0, -1.0
        vx, vy = 0.0, 0.0
        path = [(x, y)]
        for _ in range(n_steps):
            dx, dy = rosenbrock_grad(x, y)
            vx = momentum * vx - lr * dx
            vy = momentum * vy - lr * dy
            x += vx
            y += vy
            path.append((x, y))
            if rosenbrock(x, y) < 0.01:
                break
        return path

    # RMSprop from scratch
    def rmsprop(lr=0.001, decay=0.99, eps=1e-8, n_steps=5000):
        x, y = -1.0, -1.0
        sx, sy = 0.0, 0.0
        path = [(x, y)]
        for _ in range(n_steps):
            dx, dy = rosenbrock_grad(x, y)
            sx = decay * sx + (1 - decay) * dx ** 2
            sy = decay * sy + (1 - decay) * dy ** 2
            x -= lr * dx / (np.sqrt(sx) + eps)
            y -= lr * dy / (np.sqrt(sy) + eps)
            path.append((x, y))
            if rosenbrock(x, y) < 0.01:
                break
        return path

    # Adam
    def adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=5000):
        x, y = -1.0, -1.0
        mx, my = 0.0, 0.0
        vx, vy = 0.0, 0.0
        path = [(x, y)]
        for t in range(1, n_steps + 1):
            dx, dy = rosenbrock_grad(x, y)
            mx = beta1 * mx + (1 - beta1) * dx
            my = beta1 * my + (1 - beta1) * dy
            vx = beta2 * vx + (1 - beta2) * dx ** 2
            vy = beta2 * vy + (1 - beta2) * dy ** 2
            mx_hat = mx / (1 - beta1 ** t)
            my_hat = my / (1 - beta1 ** t)
            vx_hat = vx / (1 - beta2 ** t)
            vy_hat = vy / (1 - beta2 ** t)
            x -= lr * mx_hat / (np.sqrt(vx_hat) + eps)
            y -= lr * my_hat / (np.sqrt(vy_hat) + eps)
            path.append((x, y))
            if rosenbrock(x, y) < 0.01:
                break
        return path

    for name, opt_fn in [("SGD+Momentum", sgd_momentum),
                          ("RMSprop", rmsprop),
                          ("Adam", adam)]:
        path = opt_fn()
        final_loss = rosenbrock(path[-1][0], path[-1][1])
        converged = final_loss < 0.01
        steps = len(path) - 1
        print(f"  {name:<13}: steps={steps:5d}, "
              f"final_loss={final_loss:.6f}, converged={converged}")

    print("\n  Adam typically converges fastest on non-convex surfaces.")
    print("  RMSprop adapts learning rates per-parameter like Adam but lacks momentum correction.")


# === Exercise 2: Scheduler Ablation Study ===
# Problem: Compare 5 different schedulers on a classification task.

def exercise_2():
    """Scheduler comparison on synthetic classification."""
    torch.manual_seed(42)

    X = torch.randn(2000, 784)
    y = (X[:, :10].sum(dim=1) > 0).long()
    X_test, y_test = X[1600:], y[1600:]
    loader = DataLoader(TensorDataset(X[:1600], y[:1600]), batch_size=128, shuffle=True)

    class SmallNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(784, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.net(x)

    scheduler_configs = {
        "No scheduler": lambda opt: None,
        "StepLR": lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5),
        "CosineAnnealing": lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20),
        "OneCycleLR": lambda opt: torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=0.01, epochs=20, steps_per_epoch=len(loader)),
        "Warmup+Cosine": None,  # Custom implementation below
    }

    print(f"  {'Scheduler':<18} {'Final Test Acc':>14}")
    print(f"  {'-'*18} {'-'*14}")

    for sched_name in ["No scheduler", "StepLR", "CosineAnnealing", "OneCycleLR"]:
        torch.manual_seed(42)
        model = SmallNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = scheduler_configs[sched_name](optimizer)

        for epoch in range(20):
            model.train()
            for xb, yb in loader:
                loss = nn.CrossEntropyLoss()(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if sched_name == "OneCycleLR" and scheduler:
                    scheduler.step()

            if sched_name != "OneCycleLR" and scheduler:
                scheduler.step()

        model.eval()
        with torch.no_grad():
            acc = (model(X_test).argmax(1) == y_test).float().mean().item()
        print(f"  {sched_name:<18} {acc:14.4f}")

    # Warmup + Cosine (custom)
    torch.manual_seed(42)
    model = SmallNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    warmup_epochs = 3

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (20 - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    import numpy as np
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(20):
        model.train()
        for xb, yb in loader:
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item()
    print(f"  {'Warmup+Cosine':<18} {acc:14.4f}")


# === Exercise 3: Large-Batch Training Simulation ===
# Problem: Simulate large-batch training with gradient accumulation.

def exercise_3():
    """Gradient accumulation to simulate large-batch training."""
    torch.manual_seed(42)

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(100, 64), nn.ReLU(),
                nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.net(x)

    X = torch.randn(1024, 100)
    y = torch.randint(0, 10, (1024,))

    # Method 1: Direct large batch
    torch.manual_seed(42)
    model1 = SimpleNet()
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    loader1 = DataLoader(TensorDataset(X, y), batch_size=1024, shuffle=False)

    model1.train()
    for xb, yb in loader1:
        loss = nn.CrossEntropyLoss()(model1(xb), yb)
        opt1.zero_grad()
        loss.backward()
        opt1.step()

    # Method 2: Gradient accumulation (micro_batch=64, accum=16 -> effective 1024)
    torch.manual_seed(42)
    model2 = SimpleNet()
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    loader2 = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=False)
    accum_steps = 16

    model2.train()
    opt2.zero_grad()
    for i, (xb, yb) in enumerate(loader2):
        loss = nn.CrossEntropyLoss()(model2(xb), yb) / accum_steps
        loss.backward()
        if (i + 1) % accum_steps == 0:
            opt2.step()
            opt2.zero_grad()

    # Compare parameters
    param_diff = 0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        param_diff += (p1 - p2).abs().max().item()

    print(f"  Direct batch=1024: loss={nn.CrossEntropyLoss()(model1(X), y).item():.4f}")
    print(f"  Accum (64x16=1024): loss={nn.CrossEntropyLoss()(model2(X), y).item():.4f}")
    print(f"  Max param difference: {param_diff:.6f}")
    print("  Gradient accumulation achieves same effective batch size with less memory.")

    # LARS-style layer-wise LR scaling
    print("\n  LARS scales LR per layer: lr_layer = lr * ||W|| / ||grad(W)||")
    print("  This prevents large-batch training from destabilizing some layers.")


if __name__ == "__main__":
    print("=== Exercise 1: Optimizer Comparison (Rosenbrock) ===")
    exercise_1()
    print("\n=== Exercise 2: Scheduler Ablation ===")
    exercise_2()
    print("\n=== Exercise 3: Large-Batch Training ===")
    exercise_3()
    print("\nAll exercises completed!")
