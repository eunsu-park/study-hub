"""
Exercises for Lesson 08: CNN Advanced
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: ResNet Skip Connection -- Why It Works ===
# Problem: Build a plain CNN vs ResNet and compare gradient flow.

def exercise_1():
    """Compare plain CNN vs ResNet: gradient norms at first layer."""
    torch.manual_seed(42)

    class PlainBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out)

    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            identity = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += identity  # Skip connection
            return F.relu(out)

    class CNN6(nn.Module):
        def __init__(self, block_cls, num_blocks=6):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
            )
            self.blocks = nn.Sequential(*[block_cls(32) for _ in range(num_blocks)])
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10)
            )

        def forward(self, x):
            return self.classifier(self.blocks(self.stem(x)))

    X = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))

    for name, block_cls in [("Plain CNN", PlainBlock), ("ResNet", ResBlock)]:
        torch.manual_seed(42)
        model = CNN6(block_cls)
        loss = nn.CrossEntropyLoss()(model(X), y)
        loss.backward()

        first_grad = model.stem[0].weight.grad.norm().item()
        last_grad = model.blocks[-1].conv2.weight.grad.norm().item()
        print(f"  {name}: first_layer_grad_norm={first_grad:.6f}, "
              f"last_block_grad_norm={last_grad:.6f}")

    print("  Skip connections allow gradients to flow directly, preventing vanishing.")


# === Exercise 2: Implement a Custom BasicBlock ===
# Problem: Implement BasicBlock from scratch and build a small ResNet.

def exercise_2():
    """Custom BasicBlock with downsample shortcut."""
    torch.manual_seed(42)

    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.downsample = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x):
            identity = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return F.relu(out)

    class SmallResNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(),
            )
            self.block1 = BasicBlock(16, 16)
            self.block2 = BasicBlock(16, 32, stride=2)
            self.block3 = BasicBlock(32, 32)
            self.block4 = BasicBlock(32, 64, stride=2)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes)
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            return self.classifier(x)

    model = SmallResNet(num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())

    print(f"  SmallResNet output shape: {out.shape}")
    print(f"  Total parameters: {params:,}")
    print(f"  Output shape confirms (batch=4, num_classes=10)")


# === Exercise 3: Squeeze-and-Excitation Channel Attention ===
# Problem: Implement SEBlock and measure accuracy improvement.

def exercise_3():
    """Implement Squeeze-and-Excitation block and add to a CNN."""
    torch.manual_seed(42)

    class SEBlock(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.excitation = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x):
            b, c, _, _ = x.shape
            s = self.squeeze(x).view(b, c)
            e = self.excitation(s).view(b, c, 1, 1)
            return x * e

    class SimpleCNN(nn.Module):
        def __init__(self, use_se=False):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.se = SEBlock(64, reduction=16) if use_se else nn.Identity()
            self.pool = nn.MaxPool2d(2)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)
            )

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.se(x)
            x = self.pool(x)
            return self.classifier(x)

    # Synthetic data
    X = torch.randn(256, 3, 32, 32)
    y = torch.randint(0, 10, (256,))
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    for use_se in [False, True]:
        torch.manual_seed(42)
        model = SimpleCNN(use_se=use_se)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

        params = sum(p.numel() for p in model.parameters())
        label = "With SE" if use_se else "No SE  "
        print(f"  {label}: train_acc={acc:.4f}, params={params:,}")

    print("  SE block learns to amplify informative channels and suppress less useful ones.")


# === Exercise 4: Model Efficiency Trade-offs ===
# Problem: Count parameters for different architectures.

def exercise_4():
    """Compare parameter counts across architectures."""

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    # Build simple versions to compare parameter counts
    class LeNetStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
                nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    class VGGStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    class ResNetStyle(nn.Module):
        def __init__(self):
            super().__init__()

            class Block(nn.Module):
                def __init__(self, ch):
                    super().__init__()
                    self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
                    self.bn1 = nn.BatchNorm2d(ch)
                    self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
                    self.bn2 = nn.BatchNorm2d(ch)

                def forward(self, x):
                    out = F.relu(self.bn1(self.conv1(x)))
                    out = self.bn2(self.conv2(out))
                    return F.relu(out + x)

            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
            )
            self.blocks = nn.Sequential(*[Block(64) for _ in range(4)])
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)
            )

        def forward(self, x):
            return self.classifier(self.blocks(self.stem(x)))

    models = {
        "LeNet-style": LeNetStyle(),
        "VGG-style": VGGStyle(),
        "ResNet-style": ResNetStyle(),
    }

    print(f"  {'Model':<15} {'Parameters':>12}")
    print(f"  {'-'*15} {'-'*12}")
    for name, model in models.items():
        params = count_parameters(model)
        print(f"  {name:<15} {params:>12,}")

    print("\n  VGG-style has most params due to stacked 3x3 convolutions.")
    print("  ResNet achieves similar depth with skip connections.")


# === Exercise 5: Architecture Evolution Experiment ===
# Problem: Train LeNet, VGG-style, ResNet-style on synthetic data and compare.

def exercise_5():
    """Compare learning curves of three architecture generations."""
    torch.manual_seed(42)

    X = torch.randn(512, 3, 32, 32)
    y = torch.randint(0, 10, (512,))
    X_test = torch.randn(128, 3, 32, 32)
    y_test = torch.randint(0, 10, (128,))
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
                nn.Linear(120, 10),
            )

        def forward(self, x):
            return self.net(x)

    class VGGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.net(x)

    class ResNet4(nn.Module):
        def __init__(self):
            super().__init__()

            class Block(nn.Module):
                def __init__(self, ch):
                    super().__init__()
                    self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
                    self.bn1 = nn.BatchNorm2d(ch)
                    self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
                    self.bn2 = nn.BatchNorm2d(ch)

                def forward(self, x):
                    out = F.relu(self.bn1(self.conv1(x)))
                    return F.relu(self.bn2(self.conv2(out)) + x)

            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
            )
            self.blocks = nn.Sequential(*[Block(32) for _ in range(4)])
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10)
            )

        def forward(self, x):
            return self.head(self.blocks(self.stem(x)))

    architectures = {"LeNet": LeNet, "VGG-4": VGGNet, "ResNet-4": ResNet4}

    for name, model_cls in architectures.items():
        torch.manual_seed(42)
        model = model_cls()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        final_loss = 0.0
        for epoch in range(25):
            model.train()
            ep_loss = 0.0
            for xb, yb in loader:
                loss = nn.CrossEntropyLoss()(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            final_loss = ep_loss / len(loader)

        model.eval()
        with torch.no_grad():
            test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

        print(f"  {name:<10}: final_loss={final_loss:.4f}, test_acc={test_acc:.4f}")


if __name__ == "__main__":
    print("=== Exercise 1: ResNet Skip Connection ===")
    exercise_1()
    print("\n=== Exercise 2: Custom BasicBlock ===")
    exercise_2()
    print("\n=== Exercise 3: Squeeze-and-Excitation ===")
    exercise_3()
    print("\n=== Exercise 4: Model Efficiency Trade-offs ===")
    exercise_4()
    print("\n=== Exercise 5: Architecture Evolution ===")
    exercise_5()
    print("\nAll exercises completed!")
