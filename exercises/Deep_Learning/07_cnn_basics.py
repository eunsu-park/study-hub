"""
Exercises for Lesson 07: CNN Basics
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# === Exercise 1: Output Dimension Calculation ===
# Problem: Compute output dimensions for various conv/pool configurations.

def exercise_1():
    """Verify output spatial dimensions for different conv configurations."""
    # Formula: out = floor((in + 2*padding - kernel) / stride) + 1

    # 1. Input: 28x28, Conv2d(kernel=5, stride=1, padding=0)
    out1 = (28 + 2 * 0 - 5) // 1 + 1  # 24
    print(f"  Case 1: 28x28 -> Conv(k=5,s=1,p=0) -> {out1}x{out1}")

    x1 = torch.randn(1, 1, 28, 28)
    conv1 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0)
    print(f"    Verified: {conv1(x1).shape[2]}x{conv1(x1).shape[3]}")

    # 2. Input: 64x64, Conv2d(kernel=3, stride=2, padding=1)
    out2 = (64 + 2 * 1 - 3) // 2 + 1  # 32
    print(f"  Case 2: 64x64 -> Conv(k=3,s=2,p=1) -> {out2}x{out2}")

    x2 = torch.randn(1, 1, 64, 64)
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
    print(f"    Verified: {conv2(x2).shape[2]}x{conv2(x2).shape[3]}")

    # 3. Input: 32x32, three Conv2d(k=3,s=1,p=1) + MaxPool2d(2,2)
    # Each conv with padding=1 preserves spatial dims, pool halves them
    out3_after_convs = 32  # Conv preserves: (32+2-3)/1+1 = 32
    out3 = out3_after_convs // 2  # MaxPool(2) -> 16
    print(f"  Case 3: 32x32 -> 3xConv(k=3,s=1,p=1) + MaxPool(2) -> {out3}x{out3}")

    x3 = torch.randn(1, 1, 32, 32)
    layers3 = nn.Sequential(
        nn.Conv2d(1, 8, 3, 1, 1),
        nn.Conv2d(8, 16, 3, 1, 1),
        nn.Conv2d(16, 16, 3, 1, 1),
        nn.MaxPool2d(2, 2),
    )
    print(f"    Verified: {layers3(x3).shape[2]}x{layers3(x3).shape[3]}")


# === Exercise 2: Manual 2D Convolution ===
# Problem: Implement convolution manually in NumPy, apply Sobel filters.

def exercise_2():
    """Manual 2D convolution with Sobel filters."""

    def conv2d_numpy(image, kernel):
        """Naive 2D convolution (no padding, stride=1)."""
        h, w = image.shape
        kh, kw = kernel.shape
        out_h = h - kh + 1
        out_w = w - kw + 1
        output = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                output[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
        return output

    # Create a 5x5 test image with a simple pattern
    image = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=float)

    # Sobel filters
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    edge_x = conv2d_numpy(image, sobel_x)
    edge_y = conv2d_numpy(image, sobel_y)

    print(f"  Original image (5x5):\n{image}")
    print(f"\n  Sobel-X (vertical edges):\n{edge_x}")
    print(f"\n  Sobel-Y (horizontal edges):\n{edge_y}")
    print(f"\n  Sobel-X detects vertical edges, Sobel-Y detects horizontal edges.")


# === Exercise 3: Train MNISTNet and Visualize Feature Maps ===
# Problem: Build and train MNISTNet, inspect first conv layer feature maps.

def exercise_3():
    """Build MNISTNet, train briefly on synthetic data, extract feature maps."""
    torch.manual_seed(42)

    class MNISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 28->14
            x = self.pool(F.relu(self.conv2(x)))   # 14->7
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    model = MNISTNet()

    # Synthetic MNIST-like data (28x28 grayscale, 10 classes)
    X_train = torch.randn(256, 1, 28, 28)
    y_train = torch.randint(0, 10, (256,))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(3):
        for i in range(0, 256, 64):
            xb = X_train[i:i + 64]
            yb = y_train[i:i + 64]
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Extract feature maps using a forward hook
    feature_maps = {}

    def hook_fn(module, input, output):
        feature_maps['conv1'] = output.detach()

    hook = model.conv1.register_forward_hook(hook_fn)

    # Run one test image
    model.eval()
    test_img = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        model(test_img)

    hook.remove()
    fmaps = feature_maps['conv1']
    print(f"  MNISTNet trained for 3 epochs on synthetic data.")
    print(f"  Feature map shape from conv1: {fmaps.shape}")
    print(f"  (batch=1, channels=32, height=28, width=28)")
    print(f"  Feature map stats - mean: {fmaps.mean():.4f}, std: {fmaps.std():.4f}")
    print(f"  Each of the 32 feature maps shows a different pattern detector.")


# === Exercise 4: CIFAR-10 with BatchNorm vs Without ===
# Problem: Compare effect of BatchNorm on training stability.

def exercise_4():
    """Compare CNN with and without BatchNorm on synthetic CIFAR-10-like data."""
    torch.manual_seed(42)

    class CIFAR10Net(nn.Module):
        def __init__(self, use_batchnorm=False):
            super().__init__()
            layers1 = [nn.Conv2d(3, 32, 3, padding=1)]
            if use_batchnorm:
                layers1.append(nn.BatchNorm2d(32))
            layers1.extend([nn.ReLU(), nn.MaxPool2d(2)])

            layers2 = [nn.Conv2d(32, 64, 3, padding=1)]
            if use_batchnorm:
                layers2.append(nn.BatchNorm2d(64))
            layers2.extend([nn.ReLU(), nn.MaxPool2d(2)])

            self.features = nn.Sequential(*layers1, *layers2)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # Synthetic data
    X_train = torch.randn(512, 3, 32, 32)
    y_train = torch.randint(0, 10, (512,))
    X_test = torch.randn(128, 3, 32, 32)
    y_test = torch.randint(0, 10, (128,))

    for use_bn in [False, True]:
        torch.manual_seed(42)
        model = CIFAR10Net(use_batchnorm=use_bn)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train for 15 epochs
        losses = []
        for epoch in range(15):
            model.train()
            epoch_loss = 0.0
            for i in range(0, 512, 64):
                xb = X_train[i:i + 64]
                yb = y_train[i:i + 64]
                loss = nn.CrossEntropyLoss()(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / 8)

        model.eval()
        with torch.no_grad():
            test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

        label = "With BN" if use_bn else "No BN  "
        print(f"  {label}: final_loss={losses[-1]:.4f}, test_acc={test_acc:.4f}")

    print("  BatchNorm helps converge faster and more stably by normalizing")
    print("  internal activations, reducing internal covariate shift.")


# === Exercise 5: Parameter Count Analysis ===
# Problem: Compare parameter counts of CNN vs MLP.

def exercise_5():
    """Compare parameter counts: CNN (MNISTNet) vs fully connected MLP."""

    class MNISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)    # 1*32*3*3 + 32 = 320
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # 32*64*3*3 + 64 = 18496
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)           # 64*7*7*128 + 128 = 401536
            self.fc2 = nn.Linear(128, 10)                    # 128*10 + 10 = 1290

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)  # 784*128 + 128 = 100480
            self.fc2 = nn.Linear(128, 10)   # 128*10 + 10 = 1290

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc2(F.relu(self.fc1(x)))

    cnn = MNISTNet()
    mlp = SimpleMLP()

    cnn_params = sum(p.numel() for p in cnn.parameters())
    mlp_params = sum(p.numel() for p in mlp.parameters())

    print(f"  CNN (MNISTNet) parameters: {cnn_params:,}")
    print(f"    conv1: {sum(p.numel() for p in cnn.conv1.parameters()):,}")
    print(f"    conv2: {sum(p.numel() for p in cnn.conv2.parameters()):,}")
    print(f"    fc1:   {sum(p.numel() for p in cnn.fc1.parameters()):,}")
    print(f"    fc2:   {sum(p.numel() for p in cnn.fc2.parameters()):,}")
    print(f"\n  MLP parameters: {mlp_params:,}")
    print(f"    fc1:   {sum(p.numel() for p in mlp.fc1.parameters()):,}")
    print(f"    fc2:   {sum(p.numel() for p in mlp.fc2.parameters()):,}")
    print(f"\n  CNNs use weight sharing (same kernel across spatial positions),")
    print(f"  drastically reducing parameters vs fully connected layers.")


if __name__ == "__main__":
    print("=== Exercise 1: Output Dimension Calculation ===")
    exercise_1()
    print("\n=== Exercise 2: Manual 2D Convolution ===")
    exercise_2()
    print("\n=== Exercise 3: Train MNISTNet and Feature Maps ===")
    exercise_3()
    print("\n=== Exercise 4: CIFAR-10 with BatchNorm vs Without ===")
    exercise_4()
    print("\n=== Exercise 5: Parameter Count Analysis ===")
    exercise_5()
    print("\nAll exercises completed!")
