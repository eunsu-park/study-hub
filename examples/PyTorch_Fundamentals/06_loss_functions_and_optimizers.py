"""
Loss Functions and Optimizers - Examples
========================================
Lesson 06: Loss Functions and Optimizers

Demonstrates:
  1. CrossEntropyLoss for classification
  2. MSELoss for regression
  3. Optimizer setup and training step
  4. Per-parameter learning rates
  5. Learning rate schedulers
"""

import torch
import torch.nn as nn
import torch.optim as optim


def example_1_classification_loss():
    """CrossEntropyLoss and BCEWithLogitsLoss."""
    print("=" * 60)
    print("Example 1: Classification Loss Functions")
    print("=" * 60)

    # CrossEntropyLoss: input is raw logits, target is class index
    ce_loss = nn.CrossEntropyLoss()
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.1, 2.0, 0.3]])
    targets = torch.tensor([0, 1])
    loss = ce_loss(logits, targets)
    print(f"CrossEntropyLoss: {loss.item():.4f}")

    # BCEWithLogitsLoss: binary/multi-label
    bce_loss = nn.BCEWithLogitsLoss()
    logits = torch.tensor([0.5, -1.0, 2.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    loss = bce_loss(logits, targets)
    print(f"BCEWithLogitsLoss: {loss.item():.4f}")


def example_2_regression_loss():
    """MSELoss, L1Loss, SmoothL1Loss."""
    print("\n" + "=" * 60)
    print("Example 2: Regression Loss Functions")
    print("=" * 60)

    pred = torch.tensor([2.5, 0.0, 2.0, 8.0])
    target = torch.tensor([3.0, -0.5, 2.0, 7.0])

    for loss_fn_cls in [nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss]:
        loss = loss_fn_cls()(pred, target)
        print(f"{loss_fn_cls.__name__}: {loss.item():.4f}")


def example_3_training_step():
    """Complete optimizer training step."""
    print("\n" + "=" * 60)
    print("Example 3: Training Step")
    print("=" * 60)

    model = nn.Linear(10, 3)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(8, 10)
    y = torch.randint(0, 3, (8,))

    for step in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        print(f"Step {step+1}: loss={loss.item():.4f}")


def example_4_per_param_lr():
    """Different learning rates for different parameter groups."""
    print("\n" + "=" * 60)
    print("Example 4: Per-Parameter Learning Rates")
    print("=" * 60)

    class TwoPartModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(10, 20)
            self.head = nn.Linear(20, 3)

        def forward(self, x):
            return self.head(torch.relu(self.backbone(x)))

    model = TwoPartModel()
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-4},
        {'params': model.head.parameters(), 'lr': 1e-2},
    ])

    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i}: lr={group['lr']}, "
              f"params={sum(p.numel() for p in group['params'])}")


def example_5_scheduler():
    """Learning rate scheduler demonstration."""
    print("\n" + "=" * 60)
    print("Example 5: Learning Rate Scheduler")
    print("=" * 60)

    model = nn.Linear(10, 3)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(20):
        lr = optimizer.param_groups[0]['lr']
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: lr={lr:.6f}")
        scheduler.step()


if __name__ == "__main__":
    example_1_classification_loss()
    example_2_regression_loss()
    example_3_training_step()
    example_4_per_param_lr()
    example_5_scheduler()
