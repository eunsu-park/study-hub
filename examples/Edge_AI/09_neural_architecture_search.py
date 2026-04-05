"""
09. Neural Architecture Search for Edge AI

Demonstrates lightweight Neural Architecture Search (NAS) techniques
for discovering efficient model architectures under edge constraints.

Covers:
- Search space definition (kernel sizes, expansion ratios, channels)
- Random search baseline
- Evolutionary architecture search
- Latency-aware multi-objective search
- Pareto-optimal architecture selection

Requirements:
    pip install torch numpy
"""

import torch
import torch.nn as nn
import random
import copy
import time
import numpy as np

print("=" * 60)
print("Edge AI — Neural Architecture Search")
print("=" * 60)


# ============================================
# 1. Search Space Definition
# ============================================
print("\n[1] Define Architecture Search Space")
print("-" * 40)

SEARCH_SPACE = {
    "num_blocks": [3, 4, 5, 6],
    "kernel_sizes": [3, 5, 7],
    "expansion_ratios": [1, 2, 4, 6],
    "channels": [16, 24, 32, 48, 64],
    "use_se": [True, False],  # Squeeze-and-Excitation
}

print("Search space:")
for key, values in SEARCH_SPACE.items():
    print(f"  {key}: {values}")

total_configs = 1
for v in SEARCH_SPACE.values():
    total_configs *= len(v)
print(f"\nTotal possible configurations: {total_configs:,}")


# ============================================
# 2. Candidate Architecture Representation
# ============================================
print("\n[2] Architecture Encoding")
print("-" * 40)


class ArchConfig:
    """Encodes a candidate architecture as a gene."""

    def __init__(self, num_blocks=4, kernel_size=3, expansion=4,
                 channels=32, use_se=False):
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.expansion = expansion
        self.channels = channels
        self.use_se = use_se

    @classmethod
    def random(cls):
        """Sample a random architecture from the search space."""
        return cls(
            num_blocks=random.choice(SEARCH_SPACE["num_blocks"]),
            kernel_size=random.choice(SEARCH_SPACE["kernel_sizes"]),
            expansion=random.choice(SEARCH_SPACE["expansion_ratios"]),
            channels=random.choice(SEARCH_SPACE["channels"]),
            use_se=random.choice(SEARCH_SPACE["use_se"]),
        )

    def mutate(self):
        """Return a mutated copy of this architecture."""
        child = copy.deepcopy(self)
        gene = random.choice(["num_blocks", "kernel_size", "expansion",
                              "channels", "use_se"])
        if gene == "num_blocks":
            child.num_blocks = random.choice(SEARCH_SPACE["num_blocks"])
        elif gene == "kernel_size":
            child.kernel_size = random.choice(SEARCH_SPACE["kernel_sizes"])
        elif gene == "expansion":
            child.expansion = random.choice(SEARCH_SPACE["expansion_ratios"])
        elif gene == "channels":
            child.channels = random.choice(SEARCH_SPACE["channels"])
        else:
            child.use_se = not child.use_se
        return child

    def to_dict(self):
        return {
            "blocks": self.num_blocks,
            "kernel": self.kernel_size,
            "expand": self.expansion,
            "ch": self.channels,
            "se": self.use_se,
        }

    def __repr__(self):
        return (f"Arch(blocks={self.num_blocks}, k={self.kernel_size}, "
                f"exp={self.expansion}, ch={self.channels}, "
                f"se={self.use_se})")


sample = ArchConfig.random()
print(f"Random architecture: {sample}")
print(f"Gene dict: {sample.to_dict()}")


# ============================================
# 3. Build Model from Architecture Config
# ============================================
print("\n[3] Build Model from Config")
print("-" * 40)


class InvertedResidual(nn.Module):
    """MobileNet-style inverted residual block."""

    def __init__(self, in_ch, out_ch, kernel_size=3, expansion=4, use_se=False):
        super().__init__()
        mid_ch = in_ch * expansion
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=padding,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
        ]
        if use_se:
            layers.append(SEBlock(mid_ch))
        layers += [
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)
        self.use_residual = (in_ch == out_ch)

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


def build_model(config: ArchConfig, num_classes=10):
    """Build a model from an architecture config."""
    layers = [
        nn.Conv2d(3, config.channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(config.channels),
        nn.ReLU6(inplace=True),
    ]
    for _ in range(config.num_blocks):
        layers.append(InvertedResidual(
            config.channels, config.channels,
            kernel_size=config.kernel_size,
            expansion=config.expansion,
            use_se=config.use_se,
        ))
    layers += [
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(config.channels, num_classes),
    ]
    return nn.Sequential(*layers)


model = build_model(sample)
params = sum(p.numel() for p in model.parameters())
print(f"Config: {sample}")
print(f"Parameters: {params:,}")


# ============================================
# 4. Proxy Metrics (Accuracy + Latency)
# ============================================
print("\n[4] Proxy Evaluation Metrics")
print("-" * 40)


def estimate_accuracy(config: ArchConfig):
    """Proxy accuracy score (in real NAS, train and validate on proxy task)."""
    score = 0.60
    score += config.num_blocks * 0.02
    score += (config.channels / 64) * 0.10
    score += (config.expansion / 6) * 0.05
    if config.use_se:
        score += 0.03
    score += random.gauss(0, 0.01)
    return min(max(score, 0.50), 0.99)


def estimate_latency_ms(config: ArchConfig):
    """Proxy latency in ms (simulates on-device measurement)."""
    latency = 2.0
    latency += config.num_blocks * 1.5
    latency += (config.kernel_size / 3) * 1.0
    latency += (config.expansion / 2) * 0.8
    latency += (config.channels / 16) * 0.5
    if config.use_se:
        latency += 1.5
    latency += random.gauss(0, 0.3)
    return max(latency, 1.0)


acc = estimate_accuracy(sample)
lat = estimate_latency_ms(sample)
print(f"Sample config: {sample}")
print(f"  Estimated accuracy: {acc:.3f}")
print(f"  Estimated latency:  {lat:.1f} ms")


# ============================================
# 5. Random Search Baseline
# ============================================
print("\n[5] Random Search (Baseline)")
print("-" * 40)

N_RANDOM = 50
random_results = []

for _ in range(N_RANDOM):
    cfg = ArchConfig.random()
    acc = estimate_accuracy(cfg)
    lat = estimate_latency_ms(cfg)
    random_results.append((cfg, acc, lat))

random_results.sort(key=lambda x: x[1], reverse=True)
best_random = random_results[0]
print(f"Searched {N_RANDOM} random architectures")
print(f"Best accuracy: {best_random[1]:.3f} (latency: {best_random[2]:.1f} ms)")
print(f"  Config: {best_random[0]}")


# ============================================
# 6. Evolutionary Search
# ============================================
print("\n[6] Evolutionary Architecture Search")
print("-" * 40)

POPULATION = 20
GENERATIONS = 10
TOURNAMENT_K = 5
LATENCY_BUDGET = 12.0  # ms

population = [(ArchConfig.random(), 0.0, 0.0) for _ in range(POPULATION)]
population = [(cfg, estimate_accuracy(cfg), estimate_latency_ms(cfg))
              for cfg, _, _ in population]

for gen in range(GENERATIONS):
    # Tournament selection
    parents = []
    for _ in range(POPULATION):
        candidates = random.sample(population, TOURNAMENT_K)
        feasible = [c for c in candidates if c[2] <= LATENCY_BUDGET]
        if feasible:
            winner = max(feasible, key=lambda x: x[1])
        else:
            winner = min(candidates, key=lambda x: x[2])
        parents.append(winner[0])

    # Mutation
    children = []
    for parent in parents:
        child_cfg = parent.mutate()
        acc = estimate_accuracy(child_cfg)
        lat = estimate_latency_ms(child_cfg)
        children.append((child_cfg, acc, lat))

    # Elitism: keep top 2 from old population
    population.sort(key=lambda x: x[1], reverse=True)
    population = population[:2] + children[:POPULATION - 2]

best_evo = max(
    [p for p in population if p[2] <= LATENCY_BUDGET],
    key=lambda x: x[1],
    default=max(population, key=lambda x: x[1]),
)

print(f"Generations: {GENERATIONS}, Population: {POPULATION}")
print(f"Latency budget: {LATENCY_BUDGET} ms")
print(f"Best architecture: {best_evo[0]}")
print(f"  Accuracy: {best_evo[1]:.3f}, Latency: {best_evo[2]:.1f} ms")


# ============================================
# 7. Pareto Front Analysis
# ============================================
print("\n[7] Pareto Front (Accuracy vs Latency)")
print("-" * 40)

all_results = random_results + list(population)


def pareto_front(results):
    """Extract Pareto-optimal architectures (maximize accuracy, minimize latency)."""
    sorted_res = sorted(results, key=lambda x: x[2])  # sort by latency
    front = []
    best_acc = -1.0
    for cfg, acc, lat in sorted_res:
        if acc > best_acc:
            front.append((cfg, acc, lat))
            best_acc = acc
    return front


pareto = pareto_front(all_results)
print(f"Total evaluated: {len(all_results)}")
print(f"Pareto-optimal architectures: {len(pareto)}")
print()
print(f"{'Accuracy':<12} {'Latency (ms)':<14} {'Config'}")
print("-" * 70)
for cfg, acc, lat in pareto:
    print(f"{acc:<12.3f} {lat:<14.1f} {cfg}")


# ============================================
# 8. Summary
# ============================================
print("\n[8] NAS Summary")
print("-" * 40)
print("Key takeaways:")
print("- Define a structured search space covering edge-relevant choices")
print("- Random search is a strong baseline for small search spaces")
print("- Evolutionary search efficiently navigates large search spaces")
print("- Latency-aware NAS uses hardware constraints in the search loop")
print("- Pareto analysis reveals the accuracy-latency trade-off frontier")
