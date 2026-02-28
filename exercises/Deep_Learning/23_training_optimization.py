"""
Exercises for Lesson 23: Training Optimization
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: torch.compile() Speedup Measurement ===
# Problem: Compare training time with and without torch.compile().

def exercise_1():
    """Measure torch.compile() speedup on a simple CNN."""

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(64 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, 10)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # Synthetic CIFAR-10-like data
    X = torch.randn(512, 3, 32, 32)
    y = torch.randint(0, 10, (512,))
    loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

    def train_one_epoch(model, loader, optimizer):
        model.train()
        for data, target in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()

    for use_compile in [False, True]:
        torch.manual_seed(42)
        model = SimpleCNN()

        if use_compile:
            try:
                model = torch.compile(model)
                label = "compiled"
            except Exception as e:
                print(f"  torch.compile not available: {e}")
                continue
        else:
            label = "eager"

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Warmup
        train_one_epoch(model, loader, optimizer)

        # Measure
        start = time.time()
        for _ in range(3):
            train_one_epoch(model, loader, optimizer)
        elapsed = (time.time() - start) / 3

        print(f"  {label}: {elapsed:.3f}s per epoch (avg of 3)")

    print("  torch.compile() fuses operations and reduces Python overhead.")
    print("  Speedup is more significant on GPU with larger models.")


# === Exercise 2: AMP + Gradient Accumulation ===
# Problem: Implement training loop combining AMP with gradient accumulation.

def exercise_2():
    """AMP with gradient accumulation (simulated, CPU only)."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(784, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    accumulation_steps = 4

    # Synthetic data
    X = torch.randn(256, 784)
    y = torch.randint(0, 10, (256,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    step_count = 0

    for i, (data, target) in enumerate(loader):
        # On CPU, autocast uses bfloat16 if available
        try:
            with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                output = model(data)
                loss = F.cross_entropy(output, target) / accumulation_steps
        except Exception:
            # Fallback without AMP for older PyTorch
            output = model(data)
            loss = F.cross_entropy(output, target) / accumulation_steps

        loss.backward()
        total_loss += loss.item() * accumulation_steps

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

    avg_loss = total_loss / len(loader)
    print(f"  Effective batch size: {16 * accumulation_steps} "
          f"(micro_batch=16, accum_steps={accumulation_steps})")
    print(f"  Optimizer steps: {step_count}")
    print(f"  Average loss: {avg_loss:.4f}")
    print("  Gradient accumulation simulates large batches with limited memory.")


# === Exercise 3: DDP vs Single GPU Comparison ===
# Problem: Explain key differences (conceptual exercise, no multi-GPU needed).

def exercise_3():
    """Compare DataParallel vs DistributedDataParallel (conceptual)."""

    comparison = {
        "Processes": ("Single process, multi-thread", "One process per GPU"),
        "GIL": ("Affected by Python GIL", "No GIL bottleneck"),
        "Gradient sync": ("Gather to GPU 0, then broadcast", "All-reduce (balanced)"),
        "Memory": ("GPU 0 uses more memory", "Equal across GPUs"),
        "Scalability": ("Poor beyond 2-4 GPUs", "Scales to hundreds"),
        "Multi-node": ("Not supported", "Supported"),
    }

    print(f"  {'Aspect':<16} {'DataParallel':<35} {'DDP':<35}")
    print(f"  {'-'*16} {'-'*35} {'-'*35}")
    for aspect, (dp, ddp) in comparison.items():
        print(f"  {aspect:<16} {dp:<35} {ddp:<35}")

    print("\n  DDP is preferred because:")
    print("  1. No GIL bottleneck (separate processes)")
    print("  2. All-reduce distributes communication evenly")
    print("  3. Equal memory usage across GPUs")
    print("  4. Linear scaling with GPU count")
    print("  5. Supports multi-node training")

    # Demonstrate basic model parameter count (as proxy)
    model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Example model: {params:,} params")
    print(f"  In DP: GPU 0 holds full model + all gradients -> memory bottleneck")
    print(f"  In DDP: each GPU holds a copy, gradients are all-reduced -> balanced")


if __name__ == "__main__":
    print("=== Exercise 1: torch.compile() Speedup ===")
    exercise_1()
    print("\n=== Exercise 2: AMP + Gradient Accumulation ===")
    exercise_2()
    print("\n=== Exercise 3: DDP vs DataParallel ===")
    exercise_3()
    print("\nAll exercises completed!")
