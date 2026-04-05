"""
Exercises for Lesson 07: Neural Architecture Search
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


# === Exercise 1: Search Space Design ===
# Problem: Define a NAS search space for a mobile-friendly CNN and
# enumerate the total number of possible architectures.

def exercise_1():
    """Design and analyze a NAS search space."""
    # Define search space
    search_space = {
        "num_blocks": [3, 4, 5],
        "block_type": ["MBConv1", "MBConv3", "MBConv6", "Skip"],
        "kernel_size": [3, 5, 7],
        "channels": [16, 24, 32, 40, 48, 64],
        "stride": [1, 2],
        "se_ratio": [0, 0.25],  # Squeeze-and-Excitation
    }

    print("  NAS Search Space Definition:\n")
    for param, values in search_space.items():
        print(f"    {param:<15} choices: {values}")

    # Calculate total architectures
    n_blocks = len(search_space["num_blocks"])
    choices_per_block = (
        len(search_space["block_type"]) *
        len(search_space["kernel_size"]) *
        len(search_space["channels"]) *
        len(search_space["stride"]) *
        len(search_space["se_ratio"])
    )

    max_blocks = max(search_space["num_blocks"])
    # For each num_blocks value, the number of architectures is choices^num_blocks
    total = sum(choices_per_block ** n for n in search_space["num_blocks"])

    print(f"\n    Choices per block: {choices_per_block}")
    print(f"    Total architectures: {total:,}")
    print(f"    This is {'infeasible' if total > 1e6 else 'feasible'} for exhaustive search")

    # Compare search strategies
    print(f"\n  Search strategy comparison:")
    strategies = [
        ("Random Search", "Sample random architectures", total, 1000),
        ("Bayesian Opt", "GP surrogate model", total, 300),
        ("Evolutionary", "Mutation + crossover", total, 500),
        ("RL-based", "Controller generates archs", total, 1000),
        ("One-shot/SuperNet", "Train supernet, sample subnets", total, 1),
        ("Differentiable", "Relax discrete to continuous", total, 1),
    ]

    print(f"    {'Strategy':<20} {'Evaluations':>13} {'GPU-hours (est)':>16}")
    print("    " + "-" * 52)
    for name, desc, space, evals in strategies:
        gpu_hours = evals * 0.5  # ~30min per evaluation
        print(f"    {name:<20} {evals:>13,} {gpu_hours:>15.0f}h")


# === Exercise 2: Simple Random NAS ===
# Problem: Implement a simple random NAS that searches for the best
# architecture configuration within a budget.

def exercise_2():
    """Random NAS implementation with budget constraint."""
    torch.manual_seed(42)
    random.seed(42)

    # Search space (simplified)
    space = {
        "hidden_dims": [16, 32, 64, 128],
        "num_layers": [1, 2, 3, 4],
        "activation": ["relu", "gelu"],
        "dropout": [0.0, 0.1, 0.2, 0.3],
    }

    # Synthetic dataset
    X = torch.randn(200, 20)
    y = (X[:, :5].sum(1) > 0).long()
    X_val = torch.randn(100, 20)
    y_val = (X_val[:, :5].sum(1) > 0).long()

    def build_model(config):
        """Build model from config dict."""
        act_fn = nn.ReLU if config["activation"] == "relu" else nn.GELU
        layers = []
        in_dim = 20
        for _ in range(config["num_layers"]):
            layers.extend([
                nn.Linear(in_dim, config["hidden_dims"]),
                act_fn(),
                nn.Dropout(config["dropout"]),
            ])
            in_dim = config["hidden_dims"]
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def evaluate(config):
        """Train and evaluate a configuration."""
        model = build_model(config)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(30):
            loss = nn.CrossEntropyLoss()(model(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X_val).argmax(1) == y_val).float().mean().item()
        params = sum(p.numel() for p in model.parameters())
        return acc, params

    # Random search with 20 trials
    n_trials = 20
    results = []

    for i in range(n_trials):
        config = {k: random.choice(v) for k, v in space.items()}
        acc, params = evaluate(config)
        results.append((config, acc, params))

    # Sort by accuracy
    results.sort(key=lambda x: -x[1])

    print(f"  Random NAS: {n_trials} trials\n")
    print(f"  {'Rank':>4} {'Accuracy':>10} {'Params':>8} {'Hidden':>8} "
          f"{'Layers':>7} {'Act':>6} {'Drop':>6}")
    print("  " + "-" * 58)

    for i, (config, acc, params) in enumerate(results[:10]):
        print(f"  {i+1:>4} {acc:>10.1%} {params:>8,} "
              f"{config['hidden_dims']:>8} {config['num_layers']:>7} "
              f"{config['activation']:>6} {config['dropout']:>6.1f}")

    best_config, best_acc, best_params = results[0]
    print(f"\n  Best: accuracy={best_acc:.1%}, params={best_params:,}")
    print(f"  Config: {best_config}")


# === Exercise 3: Hardware-Aware NAS Objective ===
# Problem: Define a multi-objective fitness function that balances
# accuracy, latency, and model size for a specific target device.

def exercise_3():
    """Multi-objective NAS fitness function for hardware constraints."""

    def hardware_aware_fitness(accuracy, latency_ms, model_size_kb,
                                target_latency_ms=30, target_size_kb=500,
                                w_acc=1.0, w_lat=0.5, w_size=0.3):
        """
        Multi-objective fitness for hardware-aware NAS.

        Combines accuracy with penalty terms for exceeding hardware constraints.
        Returns a single scalar score (higher is better).
        """
        # Accuracy reward (0 to 1)
        acc_score = accuracy

        # Latency penalty (soft constraint with exponential penalty)
        if latency_ms <= target_latency_ms:
            lat_score = 1.0
        else:
            overshoot = (latency_ms - target_latency_ms) / target_latency_ms
            lat_score = max(0, 1.0 - overshoot)

        # Size penalty
        if model_size_kb <= target_size_kb:
            size_score = 1.0
        else:
            overshoot = (model_size_kb - target_size_kb) / target_size_kb
            size_score = max(0, 1.0 - overshoot)

        fitness = w_acc * acc_score + w_lat * lat_score + w_size * size_score
        return fitness / (w_acc + w_lat + w_size)  # Normalize to [0, 1]

    # Evaluate several candidate architectures
    candidates = [
        ("Large ResNet", 0.96, 120, 2000),    # Accurate but too big/slow
        ("MobileNet V2", 0.92, 25, 400),      # Good balance
        ("Tiny CNN", 0.85, 5, 50),            # Fast but low accuracy
        ("EfficientNet-B0", 0.94, 45, 800),   # Slightly over latency
        ("NAS-found", 0.93, 22, 350),         # NAS-optimized
    ]

    target_lat = 30   # ms
    target_size = 500  # KB

    print(f"  Hardware constraints: latency <= {target_lat}ms, "
          f"size <= {target_size}KB\n")
    print(f"  {'Model':<18} {'Acc':>6} {'Lat(ms)':>8} {'Size(KB)':>9} "
          f"{'Fitness':>8} {'Status'}")
    print("  " + "-" * 60)

    for name, acc, lat, size in candidates:
        fitness = hardware_aware_fitness(acc, lat, size,
                                         target_lat, target_size)
        lat_ok = "ok" if lat <= target_lat else "OVER"
        size_ok = "ok" if size <= target_size else "OVER"
        status = f"lat:{lat_ok} size:{size_ok}"
        print(f"  {name:<18} {acc:>5.1%} {lat:>8} {size:>9} "
              f"{fitness:>8.3f} {status}")

    print("\n  The NAS-found model scores highest because it jointly")
    print("  optimizes for accuracy AND hardware constraints.")
    print("  This is why hardware-aware NAS outperforms accuracy-only NAS")
    print("  for edge deployment.")


# === Exercise 4: Supernet Weight Sharing ===
# Problem: Implement a simplified one-shot NAS supernet where
# different sub-networks share weights.

def exercise_4():
    """Simplified one-shot supernet with weight sharing."""
    torch.manual_seed(42)

    class SuperNetBlock(nn.Module):
        """A block with multiple operation choices (weight sharing)."""

        def __init__(self, in_dim, out_dim):
            super().__init__()
            # Each choice shares the supernet weights
            self.ops = nn.ModuleDict({
                "linear": nn.Linear(in_dim, out_dim),
                "linear_relu": nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.ReLU()
                ),
                "linear_bn_relu": nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU()
                ),
            })
            self.op_names = list(self.ops.keys())

        def forward(self, x, choice=None):
            if choice is None:
                choice = random.choice(self.op_names)
            return self.ops[choice](x)

    class SuperNet(nn.Module):
        """Supernet with configurable blocks."""

        def __init__(self, in_dim=20, hidden=32, n_blocks=3, out_dim=2):
            super().__init__()
            self.input_proj = nn.Linear(in_dim, hidden)
            self.blocks = nn.ModuleList([
                SuperNetBlock(hidden, hidden) for _ in range(n_blocks)
            ])
            self.head = nn.Linear(hidden, out_dim)

        def forward(self, x, arch=None):
            x = F.relu(self.input_proj(x))
            for i, block in enumerate(self.blocks):
                choice = arch[i] if arch else None
                x = block(x, choice)
            return self.head(x)

    # Train the supernet
    X = torch.randn(200, 20)
    y = (X[:, :5].sum(1) > 0).long()

    supernet = SuperNet(n_blocks=3)
    opt = torch.optim.Adam(supernet.parameters(), lr=1e-3)

    print("  Training supernet with random path sampling...\n")
    supernet.train()
    for epoch in range(100):
        # Random path sampling: each forward uses random operations
        out = supernet(X)  # Random choices
        loss = nn.CrossEntropyLoss()(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate all possible sub-networks
    supernet.eval()
    op_names = ["linear", "linear_relu", "linear_bn_relu"]
    sub_results = []

    for o1 in op_names:
        for o2 in op_names:
            for o3 in op_names:
                arch = [o1, o2, o3]
                with torch.no_grad():
                    out = supernet(X, arch=arch)
                    acc = (out.argmax(1) == y).float().mean().item()
                sub_results.append((arch, acc))

    sub_results.sort(key=lambda x: -x[1])

    print(f"  Total sub-networks: {len(sub_results)}")
    print(f"\n  Top 5 sub-networks:")
    print(f"  {'Rank':>4} {'Block 1':<15} {'Block 2':<15} {'Block 3':<15} {'Acc':>8}")
    print("  " + "-" * 60)
    for i, (arch, acc) in enumerate(sub_results[:5]):
        print(f"  {i+1:>4} {arch[0]:<15} {arch[1]:<15} {arch[2]:<15} {acc:>8.1%}")

    print(f"\n  Worst sub-network: {sub_results[-1][0]} ({sub_results[-1][1]:.1%})")
    print(f"\n  One-shot NAS advantage: all {len(sub_results)} sub-networks")
    print("  evaluated with a SINGLE supernet training run.")
    print("  Traditional NAS would need to train each independently.")


if __name__ == "__main__":
    print("=== Exercise 1: Search Space Design ===")
    exercise_1()
    print("\n=== Exercise 2: Simple Random NAS ===")
    exercise_2()
    print("\n=== Exercise 3: Hardware-Aware NAS Objective ===")
    exercise_3()
    print("\n=== Exercise 4: Supernet Weight Sharing ===")
    exercise_4()
    print("\nAll exercises completed!")
