"""
Exercises for Lesson 04: Pruning
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import numpy as np


# === Exercise 1: Magnitude-Based Pruning Analysis ===
# Problem: Implement magnitude pruning from scratch (without torch.prune)
# and compare weight distributions before and after.

def exercise_1():
    """Implement magnitude pruning from scratch."""
    torch.manual_seed(42)

    # Create a layer with weights
    layer = nn.Linear(256, 128)
    weights = layer.weight.data.clone()

    print(f"  Original weights: shape={weights.shape}, "
          f"total={weights.numel():,}")
    print(f"  Weight stats: mean={weights.mean():.4f}, "
          f"std={weights.std():.4f}")
    print(f"  Range: [{weights.min():.4f}, {weights.max():.4f}]")

    # Implement magnitude pruning manually
    sparsity_target = 0.5  # Remove 50% of weights

    # Step 1: Compute magnitude of all weights
    magnitudes = weights.abs()

    # Step 2: Find the threshold (50th percentile of magnitudes)
    k = int(weights.numel() * sparsity_target)
    threshold = torch.topk(magnitudes.flatten(), k, largest=False).values[-1]

    # Step 3: Create binary mask
    mask = (magnitudes > threshold).float()

    # Step 4: Apply mask
    pruned_weights = weights * mask

    actual_sparsity = (pruned_weights == 0).float().mean().item()
    print(f"\n  After {sparsity_target:.0%} magnitude pruning:")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Actual sparsity: {actual_sparsity:.1%}")
    print(f"  Non-zero weights: {(pruned_weights != 0).sum().item():,} "
          f"/ {weights.numel():,}")

    # Weight distribution comparison
    remaining = pruned_weights[pruned_weights != 0]
    print(f"\n  Remaining weight stats:")
    print(f"    mean={remaining.mean():.4f}, std={remaining.std():.4f}")
    print(f"    Range: [{remaining.min():.4f}, {remaining.max():.4f}]")
    print("  Small-magnitude weights removed; distribution becomes bimodal")


# === Exercise 2: Structured vs Unstructured Pruning ===
# Problem: Compare structured (channel) and unstructured (weight) pruning
# on the same model, measuring actual speedup.

def exercise_2():
    """Compare structured and unstructured pruning effects."""
    torch.manual_seed(42)

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, 10)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = SmallCNN()
    total_params = sum(p.numel() for p in model.parameters())

    # Unstructured pruning (50% of individual weights)
    model_unstruct = copy.deepcopy(model)
    for module in [model_unstruct.conv1, model_unstruct.conv2, model_unstruct.fc]:
        prune.l1_unstructured(module, name="weight", amount=0.5)

    # Structured pruning (50% of conv channels)
    model_struct = copy.deepcopy(model)
    prune.ln_structured(model_struct.conv1, "weight", amount=0.5, n=2, dim=0)
    prune.ln_structured(model_struct.conv2, "weight", amount=0.5, n=2, dim=0)
    prune.l1_unstructured(model_struct.fc, "weight", amount=0.5)

    # Compare
    print("  Unstructured Pruning (50% per-weight):")
    for name, module in model_unstruct.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            sparsity = (w == 0).float().mean().item()
            print(f"    {name}: sparsity={sparsity:.1%}, "
                  f"shape={list(w.shape)}")

    print("\n  Structured Pruning (50% channels for convs):")
    for name, module in model_struct.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            sparsity = (w == 0).float().mean().item()
            if isinstance(module, nn.Conv2d):
                active_channels = (w.sum(dim=(1, 2, 3)) != 0).sum().item()
                total_channels = w.shape[0]
                print(f"    {name}: {active_channels}/{total_channels} active channels, "
                      f"sparsity={sparsity:.1%}")
            else:
                print(f"    {name}: sparsity={sparsity:.1%}")

    print("\n  Key difference:")
    print("    Unstructured: sparse tensors, same shape -> needs sparse HW")
    print("    Structured: smaller dense tensors -> works on standard HW")
    print("    Structured gives real speedup without special hardware support")


# === Exercise 3: Lottery Ticket Hypothesis ===
# Problem: Implement a simplified lottery ticket experiment:
# train, prune, reset to original weights, retrain.

def exercise_3():
    """Simplified lottery ticket experiment."""
    torch.manual_seed(42)

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(20, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    # Synthetic dataset
    X = torch.randn(200, 20)
    y = (X[:, 0] + X[:, 1] > 0).long()

    def train_model(model, epochs=50):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(epochs):
            out = model(X)
            loss = nn.CrossEntropyLoss()(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X).argmax(1) == y).float().mean().item()
        return acc

    # Step 1: Initialize and save initial weights
    model = TinyMLP()
    initial_state = copy.deepcopy(model.state_dict())

    # Step 2: Train to convergence
    full_acc = train_model(model)
    print(f"  Step 1: Full model accuracy: {full_acc:.1%}")

    # Step 3: Prune 60% of weights
    for module in [model.fc1, model.fc2, model.fc3]:
        prune.l1_unstructured(module, "weight", amount=0.6)

    # Get the pruning masks
    masks = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            masks[name] = module.weight_mask.clone()

    # Check pruned accuracy
    model.eval()
    with torch.no_grad():
        pruned_acc = (model(X).argmax(1) == y).float().mean().item()
    print(f"  Step 2: Pruned accuracy (before retrain): {pruned_acc:.1%}")

    # Step 4a: Random reinitialization + mask (control)
    model_random = TinyMLP()
    for name, module in model_random.named_modules():
        if name in masks:
            prune.custom_from_mask(module, "weight", masks[name])
    random_acc = train_model(model_random)
    print(f"  Step 3a: Random init + mask accuracy: {random_acc:.1%}")

    # Step 4b: Lottery ticket: original init + mask
    model_lottery = TinyMLP()
    model_lottery.load_state_dict(initial_state)
    for name, module in model_lottery.named_modules():
        if name in masks:
            prune.custom_from_mask(module, "weight", masks[name])
    lottery_acc = train_model(model_lottery)
    print(f"  Step 3b: Lottery ticket (original init + mask): {lottery_acc:.1%}")

    print(f"\n  Summary:")
    print(f"    Full model:     {full_acc:.1%}")
    print(f"    Lottery ticket: {lottery_acc:.1%}")
    print(f"    Random reinit:  {random_acc:.1%}")
    print("  Lottery Ticket Hypothesis: the winning ticket (initial weights +")
    print("  discovered mask) can match full model accuracy at 60% sparsity.")


# === Exercise 4: Pruning Sensitivity Analysis ===
# Problem: Measure accuracy at different sparsity levels to find the
# maximum sparsity before significant accuracy degradation.

def exercise_4():
    """Pruning sensitivity: accuracy vs sparsity level."""
    torch.manual_seed(42)

    class SmallNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(50, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 5)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    # Generate data
    X = torch.randn(300, 50)
    y = torch.randint(0, 5, (300,))

    # Train baseline
    model = SmallNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(100):
        loss = nn.CrossEntropyLoss()(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        baseline_acc = (model(X).argmax(1) == y).float().mean().item()

    print(f"  Baseline accuracy: {baseline_acc:.1%}\n")
    print(f"  {'Sparsity':>10} {'Accuracy':>10} {'Acc Drop':>10} {'Status'}")
    print("  " + "-" * 45)

    for sparsity_pct in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
        sparsity = sparsity_pct / 100.0

        # One-shot pruning + fine-tuning
        pruned_model = copy.deepcopy(model)
        for module in [pruned_model.fc1, pruned_model.fc2, pruned_model.fc3]:
            prune.l1_unstructured(module, "weight", amount=sparsity)

        # Fine-tune for 30 epochs
        pruned_model.train()
        ft_opt = torch.optim.Adam(pruned_model.parameters(), lr=1e-4)
        for _ in range(30):
            loss = nn.CrossEntropyLoss()(pruned_model(X), y)
            ft_opt.zero_grad()
            loss.backward()
            ft_opt.step()

        pruned_model.eval()
        with torch.no_grad():
            acc = (pruned_model(X).argmax(1) == y).float().mean().item()

        drop = baseline_acc - acc
        status = "OK" if drop < 0.02 else ("WARN" if drop < 0.05 else "FAIL")
        print(f"  {sparsity_pct:>9}% {acc:>10.1%} {drop:>10.1%} {status:>6}")

    print("\n  Rule of thumb: most models tolerate 50-70% pruning")
    print("  with fine-tuning before significant accuracy degradation.")


if __name__ == "__main__":
    print("=== Exercise 1: Magnitude-Based Pruning Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: Structured vs Unstructured Pruning ===")
    exercise_2()
    print("\n=== Exercise 3: Lottery Ticket Hypothesis ===")
    exercise_3()
    print("\n=== Exercise 4: Pruning Sensitivity Analysis ===")
    exercise_4()
    print("\nAll exercises completed!")
