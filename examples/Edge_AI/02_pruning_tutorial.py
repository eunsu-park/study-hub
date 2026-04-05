"""
02. Pruning Tutorial

Demonstrates PyTorch pruning techniques for reducing model size
and computational cost on edge devices.

Covers:
- Unstructured pruning (magnitude-based, L1 norm)
- Structured pruning (entire channels/filters)
- Global pruning (across all layers)
- Iterative pruning with fine-tuning loop

Requirements:
    pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy

print("=" * 60)
print("Edge AI — Pruning Tutorial")
print("=" * 60)


# ============================================
# 1. Model Definition
# ============================================
print("\n[1] Define Model for Pruning")
print("-" * 40)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


model = SmallCNN()
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")


def sparsity(module, param_name="weight"):
    """Calculate the sparsity (fraction of zeros) of a parameter."""
    param = getattr(module, param_name)
    total = param.numel()
    zeros = (param == 0).sum().item()
    return zeros / total


# ============================================
# 2. Unstructured Pruning (L1 Magnitude)
# ============================================
print("\n[2] Unstructured Pruning — L1 Magnitude")
print("-" * 40)
print("Removes individual weights with the smallest absolute values.")
print("Creates sparse weight matrices (scattered zeros).\n")

model_unstruct = copy.deepcopy(model)

# Prune 30% of weights in conv1 by L1 magnitude
prune.l1_unstructured(model_unstruct.conv1, name="weight", amount=0.3)

print(f"conv1 sparsity after 30% L1 pruning: {sparsity(model_unstruct.conv1):.1%}")

# Inspect the pruning mechanism
print(f"\nPruning creates a mask stored as 'weight_mask':")
print(f"  weight_mask shape: {model_unstruct.conv1.weight_mask.shape}")
print(f"  weight_mask unique values: {model_unstruct.conv1.weight_mask.unique().tolist()}")

# The original weight is stored as weight_orig
print(f"  weight_orig shape: {model_unstruct.conv1.weight_orig.shape}")
print(f"  Effective weight = weight_orig * weight_mask")

# Apply multiple pruning passes (they compound)
prune.l1_unstructured(model_unstruct.conv1, name="weight", amount=0.3)
print(f"\nAfter second 30% pruning pass: {sparsity(model_unstruct.conv1):.1%}")
print("  (Compounds: 1 - 0.7 * 0.7 = 0.51 expected)")


# ============================================
# 3. Structured Pruning (Channel Pruning)
# ============================================
print("\n[3] Structured Pruning — Channel/Filter Level")
print("-" * 40)
print("Removes entire output channels (filters). Unlike unstructured,")
print("this directly reduces tensor dimensions for real speedup.\n")

model_struct = copy.deepcopy(model)

# Prune 50% of output channels in conv1 using L2 norm
prune.ln_structured(
    model_struct.conv1,
    name="weight",
    amount=0.5,     # Remove 50% of channels
    n=2,            # L2 norm
    dim=0           # Prune along output channel dimension
)

# Count non-zero channels
mask = model_struct.conv1.weight_mask
channel_norms = mask.sum(dim=(1, 2, 3))
active_channels = (channel_norms > 0).sum().item()
total_channels = mask.shape[0]

print(f"conv1 channels: {active_channels}/{total_channels} active")
print(f"conv1 weight sparsity: {sparsity(model_struct.conv1):.1%}")

# Show which channels were pruned
pruned_channels = (channel_norms == 0).nonzero(as_tuple=True)[0].tolist()
print(f"Pruned channel indices: {pruned_channels[:10]}...")


# ============================================
# 4. Random Pruning
# ============================================
print("\n[4] Random Pruning (Baseline Comparison)")
print("-" * 40)

model_random = copy.deepcopy(model)
prune.random_unstructured(model_random.conv1, name="weight", amount=0.3)
print(f"conv1 sparsity after 30% random pruning: {sparsity(model_random.conv1):.1%}")
print("Random pruning serves as a baseline — magnitude pruning is usually better.")


# ============================================
# 5. Global Pruning
# ============================================
print("\n[5] Global Pruning — Across All Layers")
print("-" * 40)
print("Prunes the globally smallest weights, regardless of which layer")
print("they belong to. Layers with larger weights keep more parameters.\n")

model_global = copy.deepcopy(model)

# Collect all (module, param_name) pairs to prune
parameters_to_prune = [
    (model_global.conv1, "weight"),
    (model_global.conv2, "weight"),
    (model_global.fc, "weight"),
]

# Global L1 unstructured pruning at 40%
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.4,
)

print(f"{'Layer':<12} {'Sparsity':>10}")
print("-" * 24)
for name, module in [("conv1", model_global.conv1),
                     ("conv2", model_global.conv2),
                     ("fc", model_global.fc)]:
    print(f"{name:<12} {sparsity(module):>10.1%}")

# Overall sparsity
total_zeros = sum(
    (getattr(m, "weight") == 0).sum().item()
    for m in [model_global.conv1, model_global.conv2, model_global.fc]
)
total_params = sum(
    getattr(m, "weight").numel()
    for m in [model_global.conv1, model_global.conv2, model_global.fc]
)
print(f"{'Overall':<12} {total_zeros / total_params:>10.1%}")


# ============================================
# 6. Making Pruning Permanent
# ============================================
print("\n[6] Making Pruning Permanent")
print("-" * 40)

model_perm = copy.deepcopy(model)
prune.l1_unstructured(model_perm.conv1, name="weight", amount=0.5)

print("Before remove():")
print(f"  Named buffers: {[n for n, _ in model_perm.conv1.named_buffers()]}")
print(f"  Named params:  {[n for n, _ in model_perm.conv1.named_parameters()]}")

# Make pruning permanent (remove mask, apply to weight)
prune.remove(model_perm.conv1, "weight")

print("\nAfter remove():")
print(f"  Named buffers: {[n for n, _ in model_perm.conv1.named_buffers()]}")
print(f"  Named params:  {[n for n, _ in model_perm.conv1.named_parameters()]}")
print(f"  Sparsity preserved: {sparsity(model_perm.conv1):.1%}")


# ============================================
# 7. Iterative Pruning with Fine-tuning
# ============================================
print("\n[7] Iterative Pruning with Fine-tuning Loop")
print("-" * 40)
print("Best practice: prune a little, fine-tune, repeat.")
print("This preserves accuracy much better than one-shot pruning.\n")

model_iter = copy.deepcopy(model)
optimizer = torch.optim.Adam(model_iter.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Simulate iterative pruning schedule
pruning_schedule = [0.2, 0.2, 0.2, 0.2]  # 20% each round
# Cumulative: 20% -> 36% -> 49% -> 59%

dummy_data = torch.randn(64, 1, 28, 28)
dummy_labels = torch.randint(0, 10, (64,))

for round_idx, prune_amount in enumerate(pruning_schedule):
    # Prune step
    for name, module in model_iter.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=prune_amount)

    # Calculate overall sparsity
    total_zeros = 0
    total_elements = 0
    for name, module in model_iter.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            total_zeros += (w == 0).sum().item()
            total_elements += w.numel()

    current_sparsity = total_zeros / total_elements

    # Simulate fine-tuning (3 epochs)
    for epoch in range(3):
        optimizer.zero_grad()
        output = model_iter(dummy_data)
        loss = criterion(output, dummy_labels)
        loss.backward()
        optimizer.step()

    print(f"Round {round_idx + 1}: pruned {prune_amount:.0%} -> "
          f"cumulative sparsity {current_sparsity:.1%}, "
          f"fine-tune loss {loss.item():.4f}")

print()
print("Summary:")
print("- Iterative pruning + fine-tuning retains more accuracy")
print("- Typical targets: 50-90% sparsity depending on model/task")
print("- Structured pruning gives real speedup; unstructured needs sparse HW")
