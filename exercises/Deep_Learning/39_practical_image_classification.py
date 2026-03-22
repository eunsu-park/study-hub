"""
Exercises for Lesson 39: Practical Image Classification
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Train Baseline CNN and Analyze Errors ===
# Problem: Train CNN, generate confusion matrix, find hardest classes.

def exercise_1():
    """Train CNN on synthetic data, analyze per-class accuracy."""
    torch.manual_seed(42)

    class CIFAR10CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 10))

        def forward(self, x):
            return self.classifier(self.features(x))

    # Synthetic CIFAR-10-like data
    n_classes = 10
    class_names = ["plane", "car", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
    X_train = torch.randn(2000, 3, 32, 32)
    y_train = torch.randint(0, n_classes, (2000,))
    X_test = torch.randn(500, 3, 32, 32)
    y_test = torch.randint(0, n_classes, (500,))

    # Add class-specific patterns to make some classes similar
    for c in range(n_classes):
        mask_train = y_train == c
        mask_test = y_test == c
        X_train[mask_train, c % 3, :, :] += 0.5  # Class signature
        X_test[mask_test, c % 3, :, :] += 0.5

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    model = CIFAR10CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        for xb, yb in loader:
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Confusion matrix
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(1)

    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for true, pred in zip(y_test, preds):
        confusion[true, pred] += 1

    print("  Confusion matrix (rows=true, cols=predicted):")
    header = "      " + " ".join(f"{c[:3]:>5}" for c in class_names)
    print(f"  {header}")
    for i in range(n_classes):
        row = " ".join(f"{confusion[i, j].item():5d}" for j in range(n_classes))
        print(f"  {class_names[i][:5]:>5} {row}")

    # Per-class accuracy
    per_class = confusion.diag().float() / confusion.sum(dim=1).float()
    worst_classes = per_class.argsort()[:2]

    print(f"\n  Per-class accuracy:")
    for i in range(n_classes):
        print(f"    {class_names[i]}: {per_class[i].item():.4f}")
    print(f"\n  Hardest classes: {class_names[worst_classes[0]]}, {class_names[worst_classes[1]]}")


# === Exercise 2: Compare Data Augmentation Strategies ===
# Problem: No augmentation vs basic vs full augmentation.

def exercise_2():
    """Compare augmentation strategies on synthetic data."""
    torch.manual_seed(42)

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.net(x)

    X_train = torch.randn(1000, 3, 32, 32)
    y_train = torch.randint(0, 10, (1000,))
    X_test = torch.randn(200, 3, 32, 32)
    y_test = torch.randint(0, 10, (200,))

    def augment_none(x):
        return x

    def augment_basic(x):
        """Random horizontal flip and small noise (simulating crop)."""
        if torch.rand(1) > 0.5:
            x = x.flip(-1)
        return x + torch.randn_like(x) * 0.05

    def augment_full(x):
        """Flip + noise + color jitter simulation."""
        if torch.rand(1) > 0.5:
            x = x.flip(-1)
        x = x + torch.randn_like(x) * 0.1
        # Color jitter: random brightness/contrast
        x = x * (0.8 + 0.4 * torch.rand(1))
        return x

    strategies = {"No augmentation": augment_none,
                  "Basic (flip+crop)": augment_basic,
                  "Full (+ color)": augment_full}

    for name, aug_fn in strategies.items():
        torch.manual_seed(42)
        model = SimpleCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(30):
            model.train()
            perm = torch.randperm(1000)
            for i in range(0, 1000, 64):
                idx = perm[i:i + 64]
                xb = torch.stack([aug_fn(X_train[j]) for j in idx])
                yb = y_train[idx]
                loss = nn.CrossEntropyLoss()(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            acc = (model(X_test).argmax(1) == y_test).float().mean().item()

        print(f"  {name:<20}: test_acc={acc:.4f}")

    print("  Color jitter typically has the biggest impact as it prevents")
    print("  the model from relying on color statistics as shortcuts.")


# === Exercise 3: Mixup Training ===
# Problem: Implement and test Mixup with different alpha values.

def exercise_3():
    """Mixup training with different alpha values."""
    torch.manual_seed(42)

    def mixup_data(x, y, alpha=0.2):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.net(x)

    X_train = torch.randn(1000, 3, 32, 32)
    y_train = torch.randint(0, 10, (1000,))
    X_test = torch.randn(200, 3, 32, 32)
    y_test = torch.randint(0, 10, (200,))
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    for alpha in [0.0, 0.2, 0.5, 1.0]:
        torch.manual_seed(42)
        np.random.seed(42)
        model = SimpleCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(50):
            model.train()
            for xb, yb in loader:
                if alpha > 0:
                    mixed_x, y_a, y_b, lam = mixup_data(xb, yb, alpha)
                    loss = mixup_criterion(criterion, model(mixed_x), y_a, y_b, lam)
                else:
                    loss = criterion(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            acc = (model(X_test).argmax(1) == y_test).float().mean().item()

        label = "No Mixup" if alpha == 0 else f"alpha={alpha}"
        print(f"  {label:<12}: test_acc={acc:.4f}")

    print("\n  Mixup regularizes by training on convex combinations of examples,")
    print("  preventing the model from memorizing individual samples.")


# === Exercise 4: Transfer Learning vs From Scratch ===
# Problem: Compare pretrained vs random initialization.

def exercise_4():
    """Transfer learning comparison with synthetic data."""
    torch.manual_seed(42)

    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(ch)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            return F.relu(self.bn2(self.conv2(out)) + x)

    class MiniResNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
            )
            self.blocks = nn.Sequential(
                ResBlock(32), ResBlock(32), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                ResBlock(64), ResBlock(64),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.head(self.blocks(self.stem(x)))

    X_train = torch.randn(500, 3, 32, 32)
    y_train = torch.randint(0, 10, (500,))
    X_test = torch.randn(100, 3, 32, 32)
    y_test = torch.randint(0, 10, (100,))
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    # "Pretrained" model (train on more data first)
    pretrained = MiniResNet()
    opt_pt = torch.optim.Adam(pretrained.parameters(), lr=0.001)
    X_pretrain = torch.randn(2000, 3, 32, 32)
    y_pretrain = torch.randint(0, 10, (2000,))
    pt_loader = DataLoader(TensorDataset(X_pretrain, y_pretrain), batch_size=64, shuffle=True)

    for epoch in range(10):
        pretrained.train()
        for xb, yb in pt_loader:
            loss = F.cross_entropy(pretrained(xb), yb)
            opt_pt.zero_grad()
            loss.backward()
            opt_pt.step()

    # Fine-tune pretrained
    ft_model = MiniResNet()
    ft_model.load_state_dict(pretrained.state_dict())
    optimizer_ft = torch.optim.Adam(ft_model.parameters(), lr=0.0001)

    for epoch in range(30):
        ft_model.train()
        for xb, yb in loader:
            loss = F.cross_entropy(ft_model(xb), yb)
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()

    ft_model.eval()
    with torch.no_grad():
        ft_acc = (ft_model(X_test).argmax(1) == y_test).float().mean().item()

    # From scratch
    torch.manual_seed(42)
    scratch_model = MiniResNet()
    optimizer_s = torch.optim.Adam(scratch_model.parameters(), lr=0.001)

    for epoch in range(30):
        scratch_model.train()
        for xb, yb in loader:
            loss = F.cross_entropy(scratch_model(xb), yb)
            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()

    scratch_model.eval()
    with torch.no_grad():
        scratch_acc = (scratch_model(X_test).argmax(1) == y_test).float().mean().item()

    print(f"  Fine-tuned (pretrained): test_acc={ft_acc:.4f}")
    print(f"  From scratch:            test_acc={scratch_acc:.4f}")
    print("  Pretrained weights provide better initialization, converging faster")
    print("  and often achieving better accuracy, especially with limited data.")


# === Exercise 5: Combine All Techniques ===
# Problem: Build the best classifier combining all techniques.

def exercise_5():
    """Combine ResBlocks, augmentation, Mixup, label smoothing, cosine LR."""
    torch.manual_seed(42)

    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(ch)

        def forward(self, x):
            return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)

    class BestNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
            )
            self.blocks = nn.Sequential(
                ResBlock(64), ResBlock(64), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                ResBlock(128), nn.MaxPool2d(2),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Dropout(0.3), nn.Linear(128, 10)
            )

        def forward(self, x):
            return self.head(self.blocks(self.stem(x)))

    X = torch.randn(2000, 3, 32, 32)
    y = torch.randint(0, 10, (2000,))
    X_test = torch.randn(500, 3, 32, 32)
    y_test = torch.randint(0, 10, (500,))

    model = BestNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    for epoch in range(50):
        model.train()
        for xb, yb in loader:
            # Mixup
            lam = np.random.beta(0.2, 0.2)
            idx = torch.randperm(xb.size(0))
            mixed = lam * xb + (1 - lam) * xb[idx]
            # Augmentation: random flip
            if torch.rand(1) > 0.5:
                mixed = mixed.flip(-1)

            loss = lam * criterion(model(mixed), yb) + \
                   (1 - lam) * criterion(model(mixed), yb[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    techniques = [
        ("ResBlocks", "Gradient flow + depth"),
        ("Data augmentation", "Regularization via input diversity"),
        ("Mixup (alpha=0.2)", "Linear interpolation regularization"),
        ("Label smoothing (0.1)", "Soft targets prevent overconfidence"),
        ("Cosine annealing", "Smooth LR decay for convergence"),
    ]

    print(f"  Combined model test accuracy: {acc:.4f}")
    print(f"\n  Technique contribution summary:")
    for tech, desc in techniques:
        print(f"    {tech}: {desc}")


if __name__ == "__main__":
    print("=== Exercise 1: Baseline CNN + Error Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: Augmentation Comparison ===")
    exercise_2()
    print("\n=== Exercise 3: Mixup Training ===")
    exercise_3()
    print("\n=== Exercise 4: Transfer Learning vs Scratch ===")
    exercise_4()
    print("\n=== Exercise 5: Combine All Techniques ===")
    exercise_5()
    print("\nAll exercises completed!")
