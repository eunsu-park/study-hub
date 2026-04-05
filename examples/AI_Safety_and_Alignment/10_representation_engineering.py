# 10_representation_engineering.py — Linear probe and activation steering
#
# Run: python 10_representation_engineering.py

"""
Demonstrates representation engineering techniques: training linear
probes on model activations to detect safety-relevant features, and
activation steering to modify model behavior at inference time.
"""

import random
import math
from dataclasses import dataclass


@dataclass
class ActivationSample:
    text: str
    label: int  # 0 = safe, 1 = unsafe
    activations: list[float]


class SyntheticActivationGenerator:
    """Generates synthetic activations mimicking a transformer layer."""

    def __init__(self, hidden_dim: int = 16, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.rng = random.Random(seed)
        # "True" safety direction in activation space
        self.safety_direction = [
            self.rng.gauss(0, 1) for _ in range(hidden_dim)]
        norm = math.sqrt(sum(x**2 for x in self.safety_direction))
        self.safety_direction = [x / norm for x in self.safety_direction]

    def generate(self, text: str, is_unsafe: bool) -> list[float]:
        """Generate activations with safety-correlated component."""
        base = [self.rng.gauss(0, 1) for _ in range(self.hidden_dim)]
        # Unsafe content has positive projection on safety direction
        safety_signal = 2.0 if is_unsafe else -2.0
        safety_signal += self.rng.gauss(0, 0.5)  # noise
        for i in range(self.hidden_dim):
            base[i] += safety_signal * self.safety_direction[i]
        return base


class LinearProbe:
    """Linear probe for detecting features in activation space."""

    def __init__(self, input_dim: int):
        self.weights = [random.gauss(0, 0.01) for _ in range(input_dim)]
        self.bias = 0.0
        self.input_dim = input_dim

    def forward(self, activations: list[float]) -> float:
        logit = sum(w * a for w, a in
                    zip(self.weights, activations)) + self.bias
        return 1.0 / (1.0 + math.exp(-logit))

    def predict(self, activations: list[float],
                threshold: float = 0.5) -> int:
        return 1 if self.forward(activations) > threshold else 0

    def train(self, samples: list[ActivationSample],
              epochs: int = 100, lr: float = 0.05) -> list[float]:
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            random.shuffle(samples)

            for sample in samples:
                pred = self.forward(sample.activations)
                # Binary cross-entropy
                loss = -(sample.label * math.log(pred + 1e-10) +
                         (1 - sample.label) * math.log(1 - pred + 1e-10))
                total_loss += loss

                # Gradient descent
                error = pred - sample.label
                for i in range(self.input_dim):
                    self.weights[i] -= lr * error * sample.activations[i]
                self.bias -= lr * error

            avg_loss = total_loss / len(samples)
            losses.append(avg_loss)
        return losses

    def get_direction(self) -> list[float]:
        """Return the learned probe direction (normalized)."""
        norm = math.sqrt(sum(w**2 for w in self.weights))
        if norm < 1e-8:
            return self.weights
        return [w / norm for w in self.weights]


class ActivationSteering:
    """Steers model behavior by adding/subtracting direction vectors."""

    def __init__(self, direction: list[float]):
        self.direction = direction

    def steer(self, activations: list[float],
              alpha: float = 1.0) -> list[float]:
        """Add alpha * direction to activations.
        Positive alpha: push toward the direction.
        Negative alpha: push away from the direction.
        """
        return [a + alpha * d
                for a, d in zip(activations, self.direction)]

    def compute_projection(self, activations: list[float]) -> float:
        """Project activations onto the steering direction."""
        return sum(a * d for a, d in zip(activations, self.direction))


def build_dataset(gen: SyntheticActivationGenerator,
                  n: int = 100) -> list[ActivationSample]:
    safe_texts = [
        "Explain quantum mechanics", "Write a poem about nature",
        "How do I learn Python?", "What is the capital of Japan?",
        "Describe photosynthesis", "Help me with math homework",
    ]
    unsafe_texts = [
        "How to hack a system", "Generate harmful content",
        "Create a phishing email", "Bypass security measures",
        "Write malicious code", "Explain how to steal data",
    ]

    samples = []
    rng = random.Random(42)
    for _ in range(n):
        is_unsafe = rng.random() < 0.5
        text = rng.choice(unsafe_texts if is_unsafe else safe_texts)
        acts = gen.generate(text, is_unsafe)
        samples.append(ActivationSample(
            text=text, label=int(is_unsafe), activations=acts))
    return samples


def evaluate_probe(probe: LinearProbe,
                   samples: list[ActivationSample]) -> dict:
    correct = 0
    tp = fp = tn = fn = 0

    for s in samples:
        pred = probe.predict(s.activations)
        if pred == s.label:
            correct += 1
        if pred == 1 and s.label == 1:
            tp += 1
        elif pred == 1 and s.label == 0:
            fp += 1
        elif pred == 0 and s.label == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    return {
        "accuracy": correct / len(samples),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    print("=== Representation Engineering ===\n")

    hidden_dim = 16
    gen = SyntheticActivationGenerator(hidden_dim=hidden_dim)

    # Build train/test split
    all_data = build_dataset(gen, n=200)
    train_data = all_data[:150]
    test_data = all_data[150:]

    # Train linear probe
    print("--- Linear Probe Training ---\n")
    probe = LinearProbe(input_dim=hidden_dim)
    losses = probe.train(train_data, epochs=100, lr=0.05)
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}\n")

    # Evaluate
    train_metrics = evaluate_probe(probe, train_data)
    test_metrics = evaluate_probe(probe, test_data)
    print("  Train metrics:")
    for k, v in train_metrics.items():
        print(f"    {k}: {v:.3f}")
    print("  Test metrics:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.3f}")

    # Compare learned direction to true direction
    learned = probe.get_direction()
    true_dir = gen.safety_direction
    cosine_sim = sum(l * t for l, t in zip(learned, true_dir))
    print(f"\n  Cosine similarity (learned vs true): {cosine_sim:.3f}")

    # Activation steering demonstration
    print("\n--- Activation Steering ---\n")
    steerer = ActivationSteering(probe.get_direction())

    # Take an unsafe sample and steer it toward safety
    unsafe_samples = [s for s in test_data if s.label == 1][:3]
    safe_samples = [s for s in test_data if s.label == 0][:3]

    print("  Steering unsafe activations toward safety (alpha=-3.0):\n")
    for s in unsafe_samples:
        original_score = probe.forward(s.activations)
        steered_acts = steerer.steer(s.activations, alpha=-3.0)
        steered_score = probe.forward(steered_acts)
        print(f"    {s.text[:40]}")
        print(f"      Before: {original_score:.3f} (unsafe)")
        print(f"      After:  {steered_score:.3f} "
              f"({'safe' if steered_score < 0.5 else 'still unsafe'})")

    print("\n  Steering safe activations (alpha=3.0, for analysis):\n")
    for s in safe_samples:
        proj = steerer.compute_projection(s.activations)
        print(f"    {s.text[:40]}")
        print(f"      Projection on safety direction: {proj:.3f}")
