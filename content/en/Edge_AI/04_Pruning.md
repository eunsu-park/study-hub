# Lesson 4: Pruning

[Previous: Quantization](./03_Quantization.md) | [Next: Knowledge Distillation](./05_Knowledge_Distillation.md)

---

## Learning Objectives

- Distinguish between structured and unstructured pruning and their hardware implications
- Implement magnitude-based pruning using PyTorch's pruning API
- Apply iterative pruning with fine-tuning to maintain accuracy at high sparsity levels
- Understand the Lottery Ticket Hypothesis and its significance for efficient model design
- Design pruning schedules that balance compression rate and accuracy recovery
- Select hardware-aware pruning strategies for specific deployment targets

---

## 1. What is Pruning?

Pruning removes redundant parameters from a neural network. The key insight is that trained networks contain many weights that contribute negligibly to the output. Removing these weights reduces model size, computation, and potentially latency.

```
Dense Network:                    Pruned Network (50% sparsity):
┌─────────────────┐              ┌─────────────────┐
│ Input Layer     │              │ Input Layer     │
│ ●─●─●─●─●      │              │ ●─●─○─●─○      │
│ ●─●─●─●─●      │              │ ○─●─●─○─●      │
│ ●─●─●─●─●      │              │ ●─○─●─●─○      │
├─────────────────┤              ├─────────────────┤
│ Hidden Layer    │              │ Hidden Layer    │
│ ●─●─●─●─●      │              │ ●─○─●─○─●      │
│ ●─●─●─●─●      │              │ ○─●─○─●─●      │
├─────────────────┤              ├─────────────────┤
│ Output Layer    │              │ Output Layer    │
│ ●─●─●          │              │ ●─●─●          │
└─────────────────┘              └─────────────────┘
  All connections active           Half removed (○ = pruned)
  100% parameters                  50% parameters
```

### 1.1 The Pruning Premise

Empirically, a large portion of weights in trained networks are near-zero and can be removed with minimal impact:

```python
import torch
import torch.nn as nn
import torchvision.models as models


def weight_distribution_analysis(model):
    """
    Analyze the weight distribution of a pretrained model.
    Shows what percentage of weights fall below various thresholds.
    """
    all_weights = []
    layer_stats = []

    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            weights = param.data.cpu().flatten()
            all_weights.append(weights)
            layer_stats.append({
                "name": name,
                "shape": tuple(param.shape),
                "numel": param.numel(),
                "mean_abs": weights.abs().mean().item(),
                "std": weights.std().item(),
                "pct_below_0.01": (weights.abs() < 0.01).float().mean().item() * 100,
            })

    all_weights = torch.cat(all_weights)

    print("=== Global Weight Statistics ===")
    print(f"Total weights:     {all_weights.numel():,}")
    print(f"Mean |w|:          {all_weights.abs().mean():.6f}")
    print(f"Std:               {all_weights.std():.6f}")

    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    print(f"\n{'Threshold':<12} {'% Below':>8} {'Cumulative Params':>18}")
    print("-" * 40)
    for t in thresholds:
        pct = (all_weights.abs() < t).float().mean().item() * 100
        count = int(pct / 100 * all_weights.numel())
        print(f"|w| < {t:<6} {pct:>7.1f}% {count:>17,}")

    print(f"\n=== Per-Layer Statistics ===")
    print(f"{'Layer':<40} {'Params':>10} {'Mean|w|':>10} {'%<0.01':>8}")
    print("-" * 70)
    for s in layer_stats:
        print(f"{s['name']:<40} {s['numel']:>10,} {s['mean_abs']:>10.4f} "
              f"{s['pct_below_0.01']:>7.1f}%")


# Analyze a pretrained ResNet-18
model = models.resnet18(weights="IMAGENET1K_V1")
weight_distribution_analysis(model)
```

---

## 2. Unstructured Pruning

Unstructured pruning removes individual weights regardless of their position in the tensor. It produces a **sparse** weight matrix.

### 2.1 Magnitude-Based Pruning

The simplest and most common approach: remove weights with the smallest absolute values.

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def magnitude_prune_demo():
    """Demonstrate basic magnitude-based unstructured pruning."""
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )

    # Check original weight
    print("=== Before Pruning ===")
    layer = model[0]  # First linear layer
    print(f"Weight shape: {layer.weight.shape}")
    print(f"Non-zero weights: {(layer.weight != 0).sum().item()} / {layer.weight.numel()}")

    # Apply L1 unstructured pruning (remove 50% of smallest weights)
    prune.l1_unstructured(layer, name="weight", amount=0.5)

    print("\n=== After Pruning (50%) ===")
    print(f"Non-zero weights: {(layer.weight != 0).sum().item()} / {layer.weight.numel()}")
    print(f"Sparsity: {100 * (layer.weight == 0).sum().item() / layer.weight.numel():.1f}%")

    # PyTorch pruning works via a mask
    # The original weight is stored as weight_orig, and a binary mask as weight_mask
    print(f"\nweight_orig exists: {hasattr(layer, 'weight_orig')}")
    print(f"weight_mask exists: {hasattr(layer, 'weight_mask')}")
    print(f"Mask sample:\n{layer.weight_mask[:3, :8]}")

    # Make pruning permanent (remove the reparameterization)
    prune.remove(layer, "weight")
    print(f"\nAfter prune.remove:")
    print(f"weight_orig exists: {hasattr(layer, 'weight_orig')}")
    print(f"Sparsity preserved: {100 * (layer.weight == 0).sum().item() / layer.weight.numel():.1f}%")


magnitude_prune_demo()
```

### 2.2 Global Unstructured Pruning

Instead of pruning each layer independently, **global pruning** considers all weights together and removes the globally smallest ones. This is usually better because it automatically allocates more capacity to important layers.

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def global_pruning_demo(sparsity=0.5):
    """
    Global pruning: remove the globally smallest weights across all layers.
    """
    model = SimpleCNN()

    # Collect all prunable layers and their parameter names
    parameters_to_prune = [
        (model.conv1, "weight"),
        (model.conv2, "weight"),
        (model.fc1, "weight"),
        (model.fc2, "weight"),
    ]

    # Apply global L1 pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )

    # Check per-layer sparsity (global pruning distributes unevenly)
    print(f"=== Global Pruning at {sparsity:.0%} ===")
    total_zeros = 0
    total_params = 0
    for name, module in [("conv1", model.conv1), ("conv2", model.conv2),
                         ("fc1", model.fc1), ("fc2", model.fc2)]:
        zeros = (module.weight == 0).sum().item()
        numel = module.weight.numel()
        total_zeros += zeros
        total_params += numel
        print(f"  {name:<8} Sparsity: {100 * zeros / numel:>5.1f}%  "
              f"({zeros:>6,} / {numel:>6,})")

    print(f"  {'Global':<8} Sparsity: {100 * total_zeros / total_params:>5.1f}%  "
          f"({total_zeros:>6,} / {total_params:>6,})")

    # Make pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)

    return model


model_pruned = global_pruning_demo(sparsity=0.5)
```

### 2.3 Limitations of Unstructured Pruning

```
Problem: Sparse matrices are hard to accelerate on standard hardware.

Dense matrix (GPU-friendly):     Sparse matrix (irregular access):
┌─────────────┐                 ┌─────────────┐
│ 0.3 0.1 0.5 │                 │ 0.3  0  0.5 │
│ 0.2 0.4 0.7 │                 │  0  0.4  0  │
│ 0.6 0.8 0.1 │                 │ 0.6  0  0.1 │
└─────────────┘                 └─────────────┘
 Contiguous memory                Scattered non-zeros
 SIMD/Tensor Core friendly        Cache-unfriendly
 Predictable access pattern        Irregular access pattern

Result: 50% sparse matrix often runs at same speed as dense on GPU/CPU!
```

Unstructured pruning reduces model **size** (storage) but may not reduce **latency** unless hardware supports sparse computation (e.g., NVIDIA Ampere N:M sparsity).

---

## 3. Structured Pruning

Structured pruning removes entire structures — channels, filters, attention heads, or layers — producing a smaller but still **dense** model that runs faster on standard hardware.

### 3.1 Channel Pruning

```
Channel Pruning for Conv2d:

Before (Conv: 64 output channels):
  ┌───┬───┬───┬───┬───┬───┬─── ... ───┬───┐
  │Ch0│Ch1│Ch2│Ch3│Ch4│Ch5│           │C63│
  └───┴───┴───┴───┴───┴───┴─── ... ───┴───┘
  64 channels × C_in × 3 × 3

After pruning 50% (32 channels remain):
  ┌───┬───┬───┬───┬─── ... ───┬───┐
  │Ch0│Ch2│Ch4│Ch5│           │C60│
  └───┴───┴───┴───┴─── ... ───┴───┘
  32 channels × C_in × 3 × 3

The pruned model is a genuinely smaller Conv2d layer.
→ Direct speedup on any hardware!
```

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def structured_pruning_demo():
    """
    Structured (channel) pruning: remove entire output channels
    from Conv2d layers based on their L1 norm.
    """
    model = SimpleCNN()

    # Compute importance of each output channel (L1 norm of filter)
    conv = model.conv1  # Shape: (32, 1, 3, 3)
    filter_norms = conv.weight.data.abs().sum(dim=(1, 2, 3))  # Sum over (C_in, H, W)
    print("Filter norms (importance) for conv1:")
    print(f"  {filter_norms}")

    # Prune 50% of channels with smallest norm
    prune.ln_structured(conv, name="weight", amount=0.5, n=1, dim=0)

    print(f"\nAfter structured pruning:")
    print(f"  Non-zero filters: {(conv.weight.data.abs().sum(dim=(1,2,3)) > 0).sum().item()} / 32")

    # In practice, you'd physically remove the pruned channels
    # to get a smaller Conv2d. Here's how:
    mask = conv.weight_mask.sum(dim=(1, 2, 3)) > 0  # Which channels survived
    surviving_indices = mask.nonzero(as_tuple=True)[0]

    # Create a physically smaller Conv2d
    new_out_channels = surviving_indices.shape[0]
    new_conv = nn.Conv2d(1, new_out_channels, 3, padding=1, bias=conv.bias is not None)
    prune.remove(conv, "weight")

    with torch.no_grad():
        new_conv.weight.copy_(conv.weight[surviving_indices])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias[surviving_indices])

    print(f"\nOriginal conv1: Conv2d(1, 32, 3) → {conv.weight.numel()} params")
    print(f"Pruned conv1:   Conv2d(1, {new_out_channels}, 3) → {new_conv.weight.numel()} params")

    return new_conv


new_conv = structured_pruning_demo()
```

### 3.2 Importance Criteria for Structured Pruning

Different metrics for deciding which channels to remove:

```python
import torch
import torch.nn as nn


def compute_channel_importance(conv_layer, method="l1_norm"):
    """
    Compute importance scores for each output channel.

    Args:
        conv_layer: nn.Conv2d module
        method: "l1_norm", "l2_norm", "geometric_median", "taylor"

    Returns:
        importance scores, shape (C_out,)
    """
    weight = conv_layer.weight.data  # (C_out, C_in, H, W)

    if method == "l1_norm":
        # Sum of absolute values — simple and effective
        return weight.abs().sum(dim=(1, 2, 3))

    elif method == "l2_norm":
        # L2 norm of each filter
        return weight.pow(2).sum(dim=(1, 2, 3)).sqrt()

    elif method == "geometric_median":
        # Distance from the geometric median of all filters
        # Filters closest to the median are redundant → prune them
        flat = weight.view(weight.shape[0], -1)  # (C_out, C_in*H*W)
        median = flat.median(dim=0)[0]  # Approximate geometric median
        distances = (flat - median).pow(2).sum(dim=1).sqrt()
        return distances  # Low distance = redundant = prune

    elif method == "taylor":
        # Taylor expansion: |weight * gradient|
        # Requires gradient information from a forward-backward pass
        if conv_layer.weight.grad is not None:
            return (conv_layer.weight * conv_layer.weight.grad).abs().sum(dim=(1, 2, 3))
        else:
            raise ValueError("Taylor method requires gradients. Run a backward pass first.")

    else:
        raise ValueError(f"Unknown method: {method}")


# Compare methods
conv = nn.Conv2d(3, 64, 3, padding=1)
for method in ["l1_norm", "l2_norm", "geometric_median"]:
    scores = compute_channel_importance(conv, method=method)
    print(f"{method:<20} Range: [{scores.min():.4f}, {scores.max():.4f}], "
          f"Std: {scores.std():.4f}")
```

### 3.3 N:M Structured Sparsity (NVIDIA Ampere)

NVIDIA's Ampere and later GPUs support **2:4 structured sparsity**: in every group of 4 weights, exactly 2 must be zero. This gives 50% sparsity with near-zero overhead.

```
2:4 Sparsity Pattern:
  Every group of 4 consecutive weights has exactly 2 non-zeros.

  [0.3, 0.0, 0.5, 0.0]  ✓  (2 non-zeros in 4)
  [0.0, 0.4, 0.0, 0.7]  ✓
  [0.3, 0.5, 0.0, 0.0]  ✓
  [0.3, 0.5, 0.7, 0.0]  ✗  (3 non-zeros — invalid)

  Hardware stores a compact index + non-zero values → 2x speedup
```

```python
import torch


def apply_nm_sparsity(tensor, n=2, m=4):
    """
    Apply N:M structured sparsity to a weight tensor.
    In every group of M consecutive elements, keep only the N largest.
    """
    original_shape = tensor.shape
    flat = tensor.view(-1)

    # Pad to multiple of M
    pad_size = (m - flat.numel() % m) % m
    if pad_size > 0:
        flat = torch.cat([flat, torch.zeros(pad_size)])

    # Reshape into groups of M
    groups = flat.view(-1, m)

    # In each group, keep top-N by magnitude, zero out the rest
    mask = torch.zeros_like(groups, dtype=torch.bool)
    _, topk_indices = groups.abs().topk(n, dim=1)
    mask.scatter_(1, topk_indices, True)

    pruned = groups * mask.float()

    # Remove padding and reshape
    result = pruned.view(-1)[:tensor.numel()].view(original_shape)

    # Verify sparsity
    actual_sparsity = (result == 0).float().mean().item()
    expected_sparsity = 1.0 - n / m

    return result, actual_sparsity


# Example
weight = torch.randn(64, 64)
pruned_weight, sparsity = apply_nm_sparsity(weight, n=2, m=4)
print(f"Original non-zeros: {(weight != 0).sum().item()}")
print(f"Pruned non-zeros:   {(pruned_weight != 0).sum().item()}")
print(f"Sparsity: {sparsity:.1%}")  # Should be ~50%

# Verify N:M pattern
flat = pruned_weight.view(-1)
groups = flat.view(-1, 4)
nonzero_per_group = (groups != 0).sum(dim=1)
print(f"All groups have exactly 2 non-zeros: {(nonzero_per_group == 2).all()}")
```

---

## 4. Iterative Pruning with Fine-Tuning

Pruning a model aggressively in one shot often causes accuracy collapse. **Iterative pruning** alternates between pruning and fine-tuning to recover accuracy at each step.

### 4.1 The Iterative Pruning Algorithm

```
Iteration 1:    Prune 20%  → Fine-tune 5 epochs → Accuracy: 95.2% → 94.8%
Iteration 2:    Prune 20%  → Fine-tune 5 epochs → Accuracy: 94.8% → 94.3%
Iteration 3:    Prune 20%  → Fine-tune 5 epochs → Accuracy: 94.3% → 93.7%
Iteration 4:    Prune 20%  → Fine-tune 5 epochs → Accuracy: 93.7% → 92.8%

Final sparsity: 1 - (0.8)^4 = 59% sparsity
Final accuracy: 92.8% (vs 91.0% with one-shot 59% pruning)
```

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy


class IterativePruner:
    """
    Iterative Magnitude Pruning (IMP).

    Repeatedly:
    1. Prune a fraction of remaining weights
    2. Fine-tune to recover accuracy
    3. Evaluate and record metrics
    """

    def __init__(self, model, train_loader, val_loader, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.history = []

    def get_sparsity(self):
        """Calculate global sparsity of the model."""
        total = 0
        zeros = 0
        for param in self.model.parameters():
            total += param.numel()
            zeros += (param == 0).sum().item()
        return zeros / total if total > 0 else 0

    def prune_step(self, amount):
        """Prune `amount` fraction of remaining weights globally."""
        parameters_to_prune = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, "weight"))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

    def fine_tune(self, epochs, lr=1e-4):
        """Fine-tune to recover accuracy after pruning."""
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def evaluate(self):
        """Evaluate model accuracy on validation set."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return 100.0 * correct / total

    def run(self, num_iterations, prune_amount_per_step, finetune_epochs, lr=1e-4):
        """
        Run the full iterative pruning pipeline.

        Args:
            num_iterations: Number of prune-finetune cycles
            prune_amount_per_step: Fraction of remaining weights to prune each step
            finetune_epochs: Epochs of fine-tuning after each prune step
            lr: Learning rate for fine-tuning
        """
        # Record baseline
        baseline_acc = self.evaluate()
        self.history.append({
            "iteration": 0,
            "sparsity": self.get_sparsity(),
            "accuracy": baseline_acc,
        })
        print(f"Baseline — Sparsity: {self.get_sparsity():.1%}, "
              f"Accuracy: {baseline_acc:.2f}%")

        for i in range(1, num_iterations + 1):
            # Prune
            self.prune_step(prune_amount_per_step)

            # Fine-tune
            self.fine_tune(finetune_epochs, lr=lr)

            # Evaluate
            accuracy = self.evaluate()
            sparsity = self.get_sparsity()

            self.history.append({
                "iteration": i,
                "sparsity": sparsity,
                "accuracy": accuracy,
            })
            print(f"Iteration {i}/{num_iterations} — "
                  f"Sparsity: {sparsity:.1%}, Accuracy: {accuracy:.2f}%")

        # Make pruning permanent
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, "weight_orig"):
                    prune.remove(module, "weight")

        return self.history


# Usage:
# pruner = IterativePruner(model, train_loader, val_loader)
# history = pruner.run(
#     num_iterations=5,
#     prune_amount_per_step=0.2,  # Prune 20% each iteration
#     finetune_epochs=5,
#     lr=1e-4,
# )
```

---

## 5. The Lottery Ticket Hypothesis

The **Lottery Ticket Hypothesis** (Frankle & Carlin, 2019) states that dense networks contain sparse subnetworks ("winning tickets") that, when trained in isolation from the same initialization, can match the full network's accuracy.

### 5.1 Core Idea

```
Dense Network (100% params)        Lottery Ticket (10% params)
┌─────────────────────────┐       ┌─────────────────────────┐
│ Random Init → Train     │       │ Same Init → Train       │
│ → 95.0% accuracy        │       │ → 95.0% accuracy        │
│ (all weights active)    │       │ (only 10% of weights)   │
└─────────────────────────┘       └─────────────────────────┘

Key Insight: The specific initial values of the surviving weights matter.
If you reinitialize the surviving weights randomly, accuracy drops.
```

### 5.2 Finding Lottery Tickets

```python
import torch
import torch.nn as nn
import copy


class LotteryTicketFinder:
    """
    Implements the Iterative Magnitude Pruning (IMP) algorithm
    to find lottery tickets.

    Algorithm:
    1. Initialize network with random weights W_0
    2. Train to convergence → W_T
    3. Prune smallest p% of weights → get mask M
    4. Reset surviving weights to W_0 (original initialization)
    5. Retrain masked network M ⊙ W_0
    6. Repeat from step 2 until target sparsity
    """

    def __init__(self, model_class, model_kwargs, train_fn, eval_fn):
        """
        Args:
            model_class: Class to instantiate the model
            model_kwargs: Arguments for model_class()
            train_fn: Function that trains a model: train_fn(model) → trained_model
            eval_fn: Function that evaluates: eval_fn(model) → accuracy
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.train_fn = train_fn
        self.eval_fn = eval_fn

    def find_ticket(self, target_sparsity=0.9, prune_rate=0.2, max_rounds=20):
        """
        Find a lottery ticket through iterative magnitude pruning.

        Args:
            target_sparsity: Desired final sparsity (e.g., 0.9 = 90%)
            prune_rate: Fraction of remaining weights to prune each round
            max_rounds: Maximum pruning rounds
        """
        # Step 1: Create initial model and save initialization
        model = self.model_class(**self.model_kwargs)
        initial_state = copy.deepcopy(model.state_dict())  # W_0

        # Global mask (1 = keep, 0 = pruned)
        masks = {}
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                masks[name] = torch.ones_like(param, dtype=torch.bool)

        results = []
        current_sparsity = 0.0

        for round_num in range(max_rounds):
            if current_sparsity >= target_sparsity:
                break

            # Step 2: Train to convergence
            model = self.train_fn(model)
            accuracy = self.eval_fn(model)

            # Step 3: Prune smallest weights (of surviving weights)
            all_surviving = []
            for name, param in model.named_parameters():
                if name in masks:
                    surviving = param.data[masks[name]].abs()
                    all_surviving.append(surviving)

            all_surviving = torch.cat(all_surviving)
            threshold = torch.quantile(all_surviving, prune_rate)

            # Update masks
            for name, param in model.named_parameters():
                if name in masks:
                    masks[name] &= (param.data.abs() > threshold)

            # Calculate sparsity
            total = sum(m.numel() for m in masks.values())
            pruned = sum((~m).sum().item() for m in masks.values())
            current_sparsity = pruned / total

            results.append({
                "round": round_num + 1,
                "sparsity": current_sparsity,
                "accuracy_before_reset": accuracy,
            })

            print(f"Round {round_num + 1}: Sparsity {current_sparsity:.1%}, "
                  f"Accuracy: {accuracy:.2f}%")

            # Step 4: Reset to initial weights, apply mask
            model = self.model_class(**self.model_kwargs)
            model.load_state_dict(initial_state)  # Reset to W_0

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        param.mul_(masks[name].float())  # Apply mask

        # Step 5: Final training of the ticket
        print("\nTraining final lottery ticket...")
        model = self.train_fn(model)
        final_acc = self.eval_fn(model)
        print(f"Lottery Ticket — Sparsity: {current_sparsity:.1%}, "
              f"Accuracy: {final_acc:.2f}%")

        return model, masks, results
```

### 5.3 Late Rewinding

The original lottery ticket hypothesis requires resetting to the **initial** weights. In practice, **late rewinding** (resetting to weights from an early training epoch, e.g., epoch 5) works better for larger networks.

```python
def late_rewind_pruning(model, train_fn, eval_fn, rewind_epoch=5, total_epochs=100):
    """
    Late Rewinding variant of lottery ticket pruning.

    Instead of rewinding to epoch 0 (initialization), rewind to
    a checkpoint from early training (e.g., epoch 5).
    """
    # Phase 1: Train for a few epochs and save checkpoint
    print(f"Phase 1: Training for {rewind_epoch} epochs (saving rewind point)...")
    for epoch in range(rewind_epoch):
        train_fn(model, epochs=1)

    rewind_state = copy.deepcopy(model.state_dict())  # Save W_k
    print(f"Rewind point saved at epoch {rewind_epoch}")

    # Phase 2: Continue training to completion
    print(f"Phase 2: Training for {total_epochs - rewind_epoch} more epochs...")
    for epoch in range(total_epochs - rewind_epoch):
        train_fn(model, epochs=1)

    # Phase 3: Prune based on final magnitudes
    # (same as standard IMP)

    # Phase 4: Rewind to W_k (not W_0!) and apply mask
    model.load_state_dict(rewind_state)
    # Apply pruning mask...

    # Phase 5: Retrain from rewind point
    # train_fn(model, epochs=total_epochs - rewind_epoch)

    return model
```

---

## 6. Pruning Schedules

### 6.1 Common Schedules

```python
import math


def one_shot_schedule(target_sparsity, total_steps):
    """Prune all at once at step 0."""
    return [target_sparsity] + [0.0] * (total_steps - 1)


def linear_schedule(target_sparsity, total_steps):
    """
    Linearly increase sparsity at each step.
    Step k prunes to (k / total_steps) * target_sparsity.
    """
    return [target_sparsity / total_steps] * total_steps


def cubic_schedule(target_sparsity, total_steps, t_start=0, t_end=None):
    """
    Gradual pruning schedule from Zhu & Gupta (2017).
    Prunes more aggressively at the start, then tapers off.

    s(t) = s_f * (1 - (1 - t/t_end)^3)
    """
    if t_end is None:
        t_end = total_steps

    sparsities = []
    prev = 0.0
    for t in range(total_steps):
        if t < t_start:
            sparsities.append(0.0)
        elif t >= t_end:
            sparsities.append(0.0)
        else:
            frac = (t - t_start) / (t_end - t_start)
            current = target_sparsity * (1 - (1 - frac) ** 3)
            delta = current - prev
            sparsities.append(max(0, delta))
            prev = current

    return sparsities


def exponential_schedule(target_sparsity, total_steps):
    """
    Exponential pruning: remove a fixed fraction of REMAINING weights each step.
    Final sparsity = 1 - (1 - rate)^total_steps = target_sparsity
    """
    rate = 1 - (1 - target_sparsity) ** (1 / total_steps)
    return [rate] * total_steps


# Compare schedules
print(f"{'Step':<6}", end="")
for name in ["One-Shot", "Linear", "Cubic", "Exponential"]:
    print(f"{name:<14}", end="")
print()

schedules = {
    "One-Shot": one_shot_schedule(0.9, 10),
    "Linear": linear_schedule(0.9, 10),
    "Cubic": cubic_schedule(0.9, 10),
    "Exponential": exponential_schedule(0.9, 10),
}

for step in range(10):
    print(f"{step:<6}", end="")
    for name, sched in schedules.items():
        print(f"{sched[step]:<14.3f}", end="")
    print()
```

### 6.2 Gradual Pruning During Training

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class GradualPruningCallback:
    """
    Callback that gradually increases sparsity during training.
    Uses the cubic schedule from Zhu & Gupta (2017).
    """

    def __init__(self, model, initial_sparsity=0.0, final_sparsity=0.9,
                 start_epoch=0, end_epoch=30, frequency=1):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.frequency = frequency  # Prune every N epochs

        # Collect prunable parameters
        self.prunable = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.prunable.append((module, "weight"))

    def _compute_target_sparsity(self, epoch):
        """Cubic schedule for target sparsity at given epoch."""
        if epoch < self.start_epoch:
            return self.initial_sparsity
        if epoch >= self.end_epoch:
            return self.final_sparsity

        frac = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        s = self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * (1 - frac) ** 3
        return s

    def on_epoch_end(self, epoch):
        """Called at the end of each training epoch."""
        if epoch % self.frequency != 0:
            return

        target = self._compute_target_sparsity(epoch)
        current = self._get_current_sparsity()

        if target > current:
            # Need to prune more
            # Amount is relative to remaining non-zero weights
            amount = (target - current) / (1 - current) if current < 1 else 0
            amount = min(max(amount, 0), 1)

            prune.global_unstructured(
                self.prunable,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )

            actual = self._get_current_sparsity()
            print(f"  Epoch {epoch}: Target sparsity {target:.1%}, "
                  f"Actual {actual:.1%}")

    def _get_current_sparsity(self):
        total = 0
        zeros = 0
        for module, name in self.prunable:
            w = getattr(module, name)
            total += w.numel()
            zeros += (w == 0).sum().item()
        return zeros / total if total > 0 else 0
```

---

## 7. Hardware-Aware Pruning

### 7.1 Matching Pruning to Hardware

| Hardware | Best Pruning Strategy | Why |
|----------|---------------------|-----|
| **CPU (x86/ARM)** | Structured (channel) | Dense GEMM is optimized; sparse ops are slow |
| **GPU (pre-Ampere)** | Structured (channel) | Same reason as CPU; no sparse tensor core support |
| **GPU (Ampere+)** | 2:4 structured sparsity | Hardware-native 2:4 sparse tensor cores |
| **Mobile NPU** | Structured (channel) | Fixed compute units expect dense tensors |
| **Edge TPU** | Structured (channel) | Only supports dense operations |
| **FPGA** | Unstructured OK | Can implement custom sparse dataflow |

### 7.2 Latency-Aware Pruning

```python
import torch
import torch.nn as nn
import time


def latency_aware_pruning(model, input_shape, target_latency_ms,
                          max_iterations=20, device="cpu"):
    """
    Prune channels iteratively, measuring actual latency at each step,
    until the target latency is reached.

    This approach is more reliable than FLOPs-based pruning because
    latency depends on memory bandwidth, cache effects, and operator
    implementation — not just FLOPs.
    """
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape, device=device)

    def measure_latency(model, num_runs=50):
        """Measure average inference latency."""
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                model(dummy)
            # Measure
            start = time.perf_counter()
            for _ in range(num_runs):
                model(dummy)
            elapsed = (time.perf_counter() - start) / num_runs * 1000
        return elapsed

    baseline_latency = measure_latency(model)
    print(f"Baseline latency: {baseline_latency:.1f} ms")
    print(f"Target latency:   {target_latency_ms:.1f} ms")

    if baseline_latency <= target_latency_ms:
        print("Already meets target!")
        return model

    # Identify Conv2d layers and their contribution to latency
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    print(f"\nFound {len(conv_layers)} Conv2d layers")

    # Iteratively prune the "cheapest" channel (biggest latency gain per accuracy cost)
    for iteration in range(max_iterations):
        current_latency = measure_latency(model)
        if current_latency <= target_latency_ms:
            print(f"\nTarget reached at iteration {iteration}!")
            print(f"Final latency: {current_latency:.1f} ms")
            break

        # Prune 10% of channels from each Conv2d layer
        for name, module in conv_layers:
            if module.out_channels > 4:  # Keep minimum channels
                n_prune = max(1, module.out_channels // 10)
                # Use L1 norm to identify least important channels
                importance = module.weight.data.abs().sum(dim=(1, 2, 3))
                _, indices = importance.sort()
                prune_indices = indices[:n_prune]

                # Zero out pruned channels (for demonstration)
                with torch.no_grad():
                    module.weight.data[prune_indices] = 0
                    if module.bias is not None:
                        module.bias.data[prune_indices] = 0

        print(f"Iteration {iteration + 1}: Latency = {current_latency:.1f} ms")

    return model
```

---

## 8. Practical Pruning Recipe

### 8.1 End-to-End Pruning Workflow

```python
def pruning_recipe(model, train_loader, val_loader, target_sparsity=0.5):
    """
    A practical pruning recipe that combines best practices.

    Steps:
    1. Analyze weight distribution (identify easy wins)
    2. Choose pruning granularity (structured vs unstructured)
    3. Apply gradual pruning during fine-tuning
    4. Export and benchmark
    """
    import torch.nn.utils.prune as prune

    # Step 1: Analyze which layers to prune
    print("=== Step 1: Weight Analysis ===")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            near_zero = (w.abs() < 0.01).float().mean().item()
            print(f"  {name}: {near_zero:.1%} of weights near zero")

    # Step 2: Choose strategy
    # For CPU/mobile: structured pruning
    # For GPU with Ampere: 2:4 sparsity
    # For storage only: unstructured is fine

    # Step 3: Apply iterative pruning with fine-tuning
    print("\n=== Step 3: Iterative Pruning ===")
    num_steps = 5
    sparsity_per_step = 1 - (1 - target_sparsity) ** (1 / num_steps)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for step in range(num_steps):
        # Prune
        params = [(m, "weight") for m in model.modules()
                  if isinstance(m, (nn.Conv2d, nn.Linear))]
        prune.global_unstructured(
            params, pruning_method=prune.L1Unstructured,
            amount=sparsity_per_step,
        )

        # Fine-tune for a few epochs
        model.train()
        for epoch in range(3):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()

        # Report
        total = sum(p.numel() for p in model.parameters())
        zeros = sum((p == 0).sum().item() for p in model.parameters())
        print(f"  Step {step + 1}/{num_steps}: Sparsity = {zeros/total:.1%}")

    # Make permanent
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if hasattr(m, "weight_orig"):
                prune.remove(m, "weight")

    print(f"\n=== Final Model ===")
    print(f"Target sparsity: {target_sparsity:.1%}")
    total = sum(p.numel() for p in model.parameters())
    zeros = sum((p == 0).sum().item() for p in model.parameters())
    print(f"Actual sparsity: {zeros/total:.1%}")

    return model
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Unstructured pruning** | Removes individual weights; high sparsity but no speedup on standard HW |
| **Structured pruning** | Removes channels/filters; guaranteed speedup on any hardware |
| **Magnitude-based** | Simple and effective: remove smallest |w| values |
| **Global pruning** | Prune across all layers simultaneously (better than per-layer) |
| **Iterative pruning** | Prune gradually with fine-tuning between steps for best accuracy |
| **Lottery Ticket Hypothesis** | Sparse subnetworks exist that match dense accuracy from initialization |
| **N:M sparsity** | Hardware-friendly (2:4 on Ampere): exactly N non-zeros per M elements |
| **Pruning schedules** | Cubic schedule (gradual) works better than one-shot |
| **Hardware-aware** | Match pruning strategy to target hardware for actual speedup |

---

## Exercises

### Exercise 1: Unstructured vs Structured Pruning

Using a pretrained ResNet-18 on CIFAR-10:
1. Apply 50% global unstructured pruning and measure accuracy + inference latency
2. Apply 50% structured (channel) pruning and measure the same
3. Compare: which gives better accuracy? Which gives better speedup?
4. At what sparsity level does unstructured pruning start to hurt accuracy significantly?

### Exercise 2: Iterative Pruning

1. Train a CNN on MNIST to >99% accuracy
2. Apply one-shot pruning at 90% sparsity and measure accuracy
3. Apply iterative pruning (5 rounds of 37% each, with fine-tuning) to reach 90% sparsity
4. Compare the accuracy of one-shot vs iterative at the same sparsity
5. Plot accuracy vs sparsity for both approaches

### Exercise 3: Lottery Ticket Experiment

1. Train a small MLP (784-300-100-10) on MNIST
2. Implement the lottery ticket algorithm: prune 20% per round, rewind to initialization
3. Find the sparsity at which the lottery ticket matches the dense network's accuracy
4. Verify: re-initialize the surviving weights randomly and retrain. Does accuracy drop?

### Exercise 4: Pruning Schedule Comparison

1. Implement one-shot, linear, cubic, and exponential pruning schedules
2. Apply each schedule to reach 80% sparsity over 10 pruning steps (with fine-tuning)
3. Plot accuracy vs training step for each schedule
4. Which schedule preserves the most accuracy? Why?

### Exercise 5: Hardware-Aware Channel Pruning

1. Implement channel pruning that physically removes pruned channels (creating smaller Conv2d layers)
2. Apply to MobileNetV2 at 30% channel pruning
3. Measure latency before and after on CPU
4. Compare the actual speedup with the theoretical speedup (based on FLOPs reduction)

---

[Previous: Quantization](./03_Quantization.md) | [Overview](./00_Overview.md) | [Next: Knowledge Distillation](./05_Knowledge_Distillation.md)

**License**: CC BY-NC 4.0
