"""
Exercises for Lesson 09: Transfer Learning
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Feature Extraction vs Fine-tuning Comparison ===
# Problem: Compare feature extraction (frozen backbone) vs full fine-tuning.

def exercise_1():
    """Feature extraction vs fine-tuning on synthetic data."""
    torch.manual_seed(42)

    # Simulated pretrained backbone (small CNN)
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.fc = nn.Linear(32, 5)

        def forward(self, x):
            feat = self.features(x)
            return self.fc(feat)

    # Synthetic data (500 training, 100 val)
    X_train = torch.randn(500, 3, 32, 32)
    y_train = torch.randint(0, 5, (500,))
    X_val = torch.randn(100, 3, 32, 32)
    y_val = torch.randint(0, 5, (100,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    for strategy in ["Feature Extraction", "Fine-tuning"]:
        torch.manual_seed(42)
        model = Backbone()

        if strategy == "Feature Extraction":
            # Freeze all except the final FC layer
            for param in model.features.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
        else:
            # Unfreeze all layers with small LR
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        for epoch in range(10):
            model.train()
            for xb, yb in train_loader:
                loss = nn.CrossEntropyLoss()(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {strategy}: val_acc={val_acc:.4f}, trainable_params={trainable:,}")

    print("  Feature extraction is fast but limited; fine-tuning adapts the whole model.")


# === Exercise 2: Gradual Unfreezing Schedule ===
# Problem: Implement three-stage gradual unfreezing.

def exercise_2():
    """Gradual unfreezing: head -> last block -> all layers."""
    torch.manual_seed(42)

    class SimpleResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU()
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 5)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = SimpleResNet()

    X = torch.randn(200, 3, 32, 32)
    y = torch.randint(0, 5, (200,))
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    val_accs = []

    # Stage 1 (epochs 1-5): Only train FC
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    for epoch in range(5):
        model.train()
        for xb, yb in loader:
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X).argmax(1) == y).float().mean().item()
        val_accs.append(acc)

    print(f"  Stage 1 (FC only, epochs 1-5): final_acc={val_accs[-1]:.4f}")

    # Stage 2 (epochs 6-10): Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    for epoch in range(5):
        model.train()
        for xb, yb in loader:
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X).argmax(1) == y).float().mean().item()
        val_accs.append(acc)

    print(f"  Stage 2 (+ layer4, epochs 6-10): final_acc={val_accs[-1]:.4f}")

    # Stage 3 (epochs 11-20): Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(10):
        model.train()
        for xb, yb in loader:
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X).argmax(1) == y).float().mean().item()
        val_accs.append(acc)

    print(f"  Stage 3 (all layers, epochs 11-20): final_acc={val_accs[-1]:.4f}")
    print(f"  Gradual unfreezing prevents catastrophic forgetting of pretrained features.")


# === Exercise 3: Discriminative Learning Rates ===
# Problem: Apply different LR per layer group.

def exercise_3():
    """Discriminative learning rates: lower LR for earlier layers."""
    torch.manual_seed(42)

    class SmallNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
            self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
            self.layer3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    X = torch.randn(200, 3, 32, 32)
    y = torch.randint(0, 10, (200,))
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    results = {}
    for mode in ["Uniform LR", "Discriminative LR"]:
        torch.manual_seed(42)
        model = SmallNet()

        if mode == "Uniform LR":
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        else:
            optimizer = torch.optim.Adam([
                {"params": model.layer1.parameters(), "lr": 1e-5},
                {"params": model.layer2.parameters(), "lr": 3e-5},
                {"params": model.layer3.parameters(), "lr": 1e-4},
                {"params": model.fc.parameters(), "lr": 1e-3},
            ])

        for epoch in range(15):
            model.train()
            for xb, yb in loader:
                loss = nn.CrossEntropyLoss()(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            acc = (model(X).argmax(1) == y).float().mean().item()
        results[mode] = acc
        print(f"  {mode}: train_acc={acc:.4f}")

    print("  Earlier layers learn general features (small updates needed);")
    print("  later layers are more task-specific (larger updates needed).")


# === Exercise 4: Domain Gap Investigation ===
# Problem: Compare from-scratch vs feature extraction vs fine-tuning.

def exercise_4():
    """Domain gap: training strategies on similar vs different domains."""
    torch.manual_seed(42)

    class SmallCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.fc = nn.Linear(32, num_classes)

        def forward(self, x):
            return self.fc(self.features(x))

    # Two synthetic datasets: "similar" (natural-like) and "different" (noise patterns)
    datasets = {
        "Similar domain": (torch.randn(300, 3, 32, 32), torch.randint(0, 5, (300,))),
        "Different domain": (torch.rand(300, 3, 32, 32) * 2 - 1, torch.randint(0, 5, (300,))),
    }

    for domain, (X, y) in datasets.items():
        X_train, y_train = X[:200], y[:200]
        X_test, y_test = X[200:], y[200:]
        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

        strategies = {
            "From scratch": {"freeze": False, "lr": 1e-3},
            "Feature extract": {"freeze": True, "lr": 1e-3},
            "Fine-tune": {"freeze": False, "lr": 1e-4},
        }

        print(f"  {domain}:")
        for strat_name, config in strategies.items():
            torch.manual_seed(42)
            model = SmallCNN(num_classes=5)

            if config["freeze"]:
                for param in model.features.parameters():
                    param.requires_grad = False
                optimizer = torch.optim.Adam(model.fc.parameters(), lr=config["lr"])
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

            for epoch in range(20):
                model.train()
                for xb, yb in loader:
                    loss = nn.CrossEntropyLoss()(model(xb), yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                acc = (model(X_test).argmax(1) == y_test).float().mean().item()
            print(f"    {strat_name}: test_acc={acc:.4f}")


# === Exercise 5: ImageNet Normalization Importance ===
# Problem: Show that wrong normalization breaks pretrained features.

def exercise_5():
    """Demonstrate importance of correct normalization for transfer learning."""
    torch.manual_seed(42)

    # Simulate pretrained features that expect specific input distribution
    class PretrainedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            return self.fc(self.features(x))

    # Train model on normalized data
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    raw_data = torch.rand(200, 3, 32, 32)
    labels = torch.randint(0, 10, (200,))

    # Train with correct normalization
    model = PretrainedModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X_correct = (raw_data - imagenet_mean) / imagenet_std

    for epoch in range(20):
        loss = nn.CrossEntropyLoss()(model(X_correct), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # Test with correct normalization
        acc_correct = (model(X_correct).argmax(1) == labels).float().mean().item()

        # Test with wrong normalization (mean=0.5, std=0.5)
        X_wrong = (raw_data - 0.5) / 0.5
        acc_wrong = (model(X_wrong).argmax(1) == labels).float().mean().item()

        # Test with no normalization
        acc_none = (model(raw_data).argmax(1) == labels).float().mean().item()

    print(f"  Correct normalization (ImageNet stats): acc={acc_correct:.4f}")
    print(f"  Wrong normalization (mean=0.5, std=0.5): acc={acc_wrong:.4f}")
    print(f"  No normalization (raw pixels):           acc={acc_none:.4f}")
    print("  Wrong normalization shifts input distribution, breaking learned features.")


if __name__ == "__main__":
    print("=== Exercise 1: Feature Extraction vs Fine-tuning ===")
    exercise_1()
    print("\n=== Exercise 2: Gradual Unfreezing ===")
    exercise_2()
    print("\n=== Exercise 3: Discriminative Learning Rates ===")
    exercise_3()
    print("\n=== Exercise 4: Domain Gap Investigation ===")
    exercise_4()
    print("\n=== Exercise 5: ImageNet Normalization Importance ===")
    exercise_5()
    print("\nAll exercises completed!")
