"""
Exercises for Lesson 24: Loss Functions
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Implement Tversky Loss ===
# Problem: Implement Tversky loss for segmentation with controllable FP/FN trade-off.

def exercise_1():
    """Tversky loss implementation and comparison with Dice loss."""

    class DiceLoss(nn.Module):
        def __init__(self, smooth=1.0):
            super().__init__()
            self.smooth = smooth

        def forward(self, pred, target):
            pred = torch.sigmoid(pred)
            intersection = (pred * target).sum()
            return 1 - (2 * intersection + self.smooth) / \
                   (pred.sum() + target.sum() + self.smooth)

    class TverskyLoss(nn.Module):
        def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
            super().__init__()
            self.alpha = alpha  # Weight for false positives
            self.beta = beta    # Weight for false negatives
            self.smooth = smooth

        def forward(self, pred, target):
            pred = torch.sigmoid(pred)
            # True positives, false positives, false negatives
            TP = (pred * target).sum()
            FP = (pred * (1 - target)).sum()
            FN = ((1 - pred) * target).sum()
            tversky = (TP + self.smooth) / \
                      (TP + self.alpha * FP + self.beta * FN + self.smooth)
            return 1 - tversky

    # Synthetic segmentation data
    torch.manual_seed(42)
    # Pred logits and binary target masks
    pred = torch.randn(4, 1, 32, 32)
    target = (torch.randn(4, 1, 32, 32) > 0.5).float()

    dice = DiceLoss()
    tversky_balanced = TverskyLoss(alpha=0.5, beta=0.5)  # Equivalent to Dice
    tversky_fn_focus = TverskyLoss(alpha=0.3, beta=0.7)  # Focus on reducing FN

    print(f"  Dice Loss:              {dice(pred, target).item():.4f}")
    print(f"  Tversky (a=0.5, b=0.5): {tversky_balanced(pred, target).item():.4f}")
    print(f"  Tversky (a=0.3, b=0.7): {tversky_fn_focus(pred, target).item():.4f}")

    # Show training effect
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 1, 3, padding=1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = TverskyLoss(alpha=0.3, beta=0.7)

    # Simple segmentation task
    X = torch.randn(32, 1, 16, 16)
    Y = (X > 0).float()

    for epoch in range(20):
        out = model(X)
        loss = loss_fn(out, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"  After 20 epochs training loss: {loss.item():.4f}")
    print("  alpha=0.3, beta=0.7 penalizes false negatives more heavily,")
    print("  useful when missing a positive pixel is worse than a false alarm.")


# === Exercise 2: Multi-Scale Perceptual Loss ===
# Problem: Extract features at multiple VGG layers and compute perceptual loss.

def exercise_2():
    """Multi-scale perceptual loss using VGG-like feature extractor."""

    # Simplified VGG-like feature extractor (no pretrained weights needed)
    class SimpleVGGFeatures(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            )
            self.block2 = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            )
            self.block3 = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            )
            # Freeze all weights
            for param in self.parameters():
                param.requires_grad = False

        def forward(self, x):
            f1 = self.block1(x)
            f2 = self.block2(f1)
            f3 = self.block3(f2)
            return [f1, f2, f3]

    class MultiScalePerceptualLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = SimpleVGGFeatures()
            # Learnable weights for each scale
            self.weights = nn.Parameter(torch.ones(3) / 3)

        def forward(self, pred, target):
            pred_features = self.feature_extractor(pred)
            target_features = self.feature_extractor(target)

            total_loss = 0.0
            w = F.softmax(self.weights, dim=0)

            for i, (pf, tf) in enumerate(zip(pred_features, target_features)):
                scale_loss = F.mse_loss(pf, tf)
                total_loss += w[i] * scale_loss

            return total_loss

    torch.manual_seed(42)
    pred_img = torch.randn(2, 3, 32, 32)
    target_img = torch.randn(2, 3, 32, 32)

    loss_fn = MultiScalePerceptualLoss()
    loss = loss_fn(pred_img, target_img)

    print(f"  Multi-scale perceptual loss: {loss.item():.4f}")
    print(f"  Scale weights (softmax): {F.softmax(loss_fn.weights, dim=0).detach().tolist()}")

    # Test with identical images (loss should be ~0)
    loss_same = loss_fn(pred_img, pred_img)
    print(f"  Loss on identical images: {loss_same.item():.6f}")
    print("\n  Early layers capture textures/edges; deep layers capture structure.")
    print("  Combining scales gives a richer similarity measure than pixel-level MSE.")


# === Exercise 3: Adaptive Loss Balancing ===
# Problem: Multi-task learning with uncertainty-based loss weighting.

def exercise_3():
    """Adaptive loss balancing for multi-task learning."""
    torch.manual_seed(42)

    class MultiTaskModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Shared backbone
            self.backbone = nn.Sequential(
                nn.Linear(20, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
            )
            # Task heads
            self.seg_head = nn.Linear(64, 5)       # Segmentation (5 classes)
            self.depth_head = nn.Linear(64, 1)      # Depth estimation
            self.det_head = nn.Linear(64, 3)        # Detection (3 classes)

            # Log variance parameters for uncertainty weighting
            self.log_vars = nn.Parameter(torch.zeros(3))

        def forward(self, x):
            feat = self.backbone(x)
            return self.seg_head(feat), self.depth_head(feat), self.det_head(feat)

    model = MultiTaskModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Synthetic multi-task data
    n_samples = 500
    X = torch.randn(n_samples, 20)
    y_seg = torch.randint(0, 5, (n_samples,))
    y_depth = torch.randn(n_samples, 1)
    y_det = torch.randint(0, 3, (n_samples,))

    loader = DataLoader(TensorDataset(X, y_seg, y_depth, y_det),
                        batch_size=64, shuffle=True)

    weight_history = []
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        for xb, ys, yd, ydet in loader:
            seg_out, depth_out, det_out = model(xb)

            # Individual losses
            loss_seg = F.cross_entropy(seg_out, ys)
            loss_depth = F.l1_loss(depth_out, yd)
            loss_det = F.cross_entropy(det_out, ydet)

            # Uncertainty-based weighting: L = sum( 1/(2*sigma_i^2) * L_i + log(sigma_i) )
            # Using log_var = log(sigma^2), so precision = exp(-log_var)
            losses = [loss_seg, loss_depth, loss_det]
            total_loss = 0
            for i, loss_i in enumerate(losses):
                precision = torch.exp(-model.log_vars[i])
                total_loss += 0.5 * precision * loss_i + 0.5 * model.log_vars[i]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Record weights
        with torch.no_grad():
            weights = torch.exp(-model.log_vars).detach().tolist()
            weight_history.append(weights)

    print("  Task weights over training (precision = 1/sigma^2):")
    print(f"  {'Epoch':>6} {'Seg':>8} {'Depth':>8} {'Det':>8}")
    for i in [0, 9, 19, 29]:
        w = weight_history[i]
        print(f"  {i+1:6d} {w[0]:8.4f} {w[1]:8.4f} {w[2]:8.4f}")

    print("\n  Tasks with higher loss get lower weight (higher uncertainty).")
    print("  This prevents any single task from dominating the training signal.")


if __name__ == "__main__":
    print("=== Exercise 1: Tversky Loss ===")
    exercise_1()
    print("\n=== Exercise 2: Multi-Scale Perceptual Loss ===")
    exercise_2()
    print("\n=== Exercise 3: Adaptive Loss Balancing ===")
    exercise_3()
    print("\nAll exercises completed!")
