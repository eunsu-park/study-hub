# 03_rlhf_alignment.py — Reward model training and PPO loop skeleton
#
# Run: python 03_rlhf_alignment.py

"""
Demonstrates the RLHF (Reinforcement Learning from Human Feedback)
pipeline: reward model training on preference pairs and a simplified
PPO optimization loop for language model alignment.
"""

import random
import math
from dataclasses import dataclass


@dataclass
class PreferencePair:
    """A pair of responses where 'chosen' is preferred over 'rejected'."""
    prompt: str
    chosen: str
    rejected: str


class SimpleRewardModel:
    """Reward model trained on human preference data via Bradley-Terry."""

    def __init__(self, feature_dim: int = 8):
        self.weights = [random.gauss(0, 0.1) for _ in range(feature_dim)]
        self.feature_dim = feature_dim

    def _extract_features(self, text: str) -> list[float]:
        """Extract simple text features for demonstration."""
        return [
            len(text) / 500.0,
            text.count(".") / 10.0,
            1.0 if "help" in text.lower() else 0.0,
            1.0 if "sorry" in text.lower() else 0.0,
            text.count(",") / 10.0,
            len(set(text.split())) / max(len(text.split()), 1),
            1.0 if "?" in text else 0.0,
            min(len(text.split()) / 50.0, 1.0),
        ]

    def score(self, text: str) -> float:
        features = self._extract_features(text)
        return sum(w * f for w, f in zip(self.weights, features))

    def bradley_terry_loss(self, chosen: str, rejected: str) -> float:
        """Compute Bradley-Terry pairwise loss: -log(sigma(r_c - r_r))."""
        r_chosen = self.score(chosen)
        r_rejected = self.score(rejected)
        diff = r_chosen - r_rejected
        return -math.log(1.0 / (1.0 + math.exp(-diff)) + 1e-8)

    def train_step(self, pair: PreferencePair, lr: float = 0.01) -> float:
        """One gradient step on a preference pair."""
        loss = self.bradley_terry_loss(pair.chosen, pair.rejected)

        feat_c = self._extract_features(pair.chosen)
        feat_r = self._extract_features(pair.rejected)
        r_diff = self.score(pair.chosen) - self.score(pair.rejected)
        sigmoid = 1.0 / (1.0 + math.exp(-r_diff))
        grad_scale = sigmoid - 1.0  # derivative of -log(sigma(x))

        for i in range(self.feature_dim):
            grad = grad_scale * (feat_c[i] - feat_r[i])
            self.weights[i] -= lr * grad

        return loss

    def train(self, pairs: list[PreferencePair],
              epochs: int = 50, lr: float = 0.05) -> list[float]:
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            random.shuffle(pairs)
            for pair in pairs:
                epoch_loss += self.train_step(pair, lr)
            avg_loss = epoch_loss / len(pairs)
            losses.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")
        return losses


class PPOTrainer:
    """Simplified PPO loop for LM alignment (skeleton implementation)."""

    def __init__(self, reward_model: SimpleRewardModel,
                 kl_coeff: float = 0.1, clip_range: float = 0.2):
        self.reward_model = reward_model
        self.kl_coeff = kl_coeff
        self.clip_range = clip_range

    def compute_kl_penalty(self, old_logprob: float,
                           new_logprob: float) -> float:
        """KL divergence penalty to prevent policy from drifting."""
        return math.exp(new_logprob) * (new_logprob - old_logprob)

    def ppo_objective(self, reward: float, old_logprob: float,
                      new_logprob: float) -> float:
        """Clipped PPO objective with KL penalty."""
        ratio = math.exp(new_logprob - old_logprob)
        clipped = max(1 - self.clip_range,
                      min(1 + self.clip_range, ratio))
        kl = self.compute_kl_penalty(old_logprob, new_logprob)
        surrogate = min(ratio * reward, clipped * reward)
        return surrogate - self.kl_coeff * kl

    def train_step(self, prompt: str, responses: list[str]) -> dict:
        """Simulate one PPO step: score responses, compute objectives."""
        scores = [(r, self.reward_model.score(r)) for r in responses]
        scores.sort(key=lambda x: x[1], reverse=True)

        best_response, best_reward = scores[0]
        old_lp = random.gauss(-2.0, 0.5)
        new_lp = old_lp + random.gauss(0.1, 0.05)
        objective = self.ppo_objective(best_reward, old_lp, new_lp)

        return {
            "prompt": prompt,
            "best_response": best_response[:60],
            "reward": round(best_reward, 4),
            "objective": round(objective, 4),
            "kl_penalty": round(
                self.compute_kl_penalty(old_lp, new_lp), 4),
        }


def build_preference_data() -> list[PreferencePair]:
    """Create synthetic preference pairs for demonstration."""
    return [
        PreferencePair(
            "How do I fix this error?",
            "I'd be happy to help. First, check the error message carefully.",
            "Just Google it."
        ),
        PreferencePair(
            "Write a poem about AI",
            "In circuits deep, where numbers flow, a mind begins to learn.",
            "AI is cool. The end."
        ),
        PreferencePair(
            "Explain quantum computing",
            "Quantum computing uses qubits, which can exist in superposition.",
            "It's complicated, you wouldn't understand."
        ),
        PreferencePair(
            "What's the weather like?",
            "Sorry, I don't have real-time data, but I can help you find it.",
            "It's sunny. Trust me."
        ),
        PreferencePair(
            "Help me with my code",
            "Of course! Could you share the code and the error? I'll help.",
            "Your code is bad."
        ),
    ]


if __name__ == "__main__":
    print("=== RLHF Pipeline Demonstration ===\n")

    # Phase 1: Train reward model
    print("--- Phase 1: Reward Model Training ---\n")
    pairs = build_preference_data()
    rm = SimpleRewardModel()
    losses = rm.train(pairs, epochs=50, lr=0.05)

    print(f"\n  Final loss: {losses[-1]:.4f}")
    print(f"  Loss reduction: {losses[0] - losses[-1]:.4f}\n")

    # Verify reward model preferences
    print("--- Reward Model Evaluation ---\n")
    correct = 0
    for pair in pairs:
        sc = rm.score(pair.chosen)
        sr = rm.score(pair.rejected)
        match = sc > sr
        correct += int(match)
        print(f"  Prompt: {pair.prompt[:40]}")
        print(f"    Chosen: {sc:.3f} | Rejected: {sr:.3f} | "
              f"{'OK' if match else 'WRONG'}")
    print(f"\n  Accuracy: {correct}/{len(pairs)}\n")

    # Phase 2: PPO optimization
    print("--- Phase 2: PPO Optimization Loop ---\n")
    ppo = PPOTrainer(rm, kl_coeff=0.1, clip_range=0.2)

    prompts_and_responses = [
        ("Explain AI safety", [
            "AI safety ensures systems behave as intended and help users.",
            "Safety is overrated.",
            "I'd be happy to explain. AI safety is a critical field.",
        ]),
        ("What is alignment?", [
            "Alignment means making AI goals match human values.",
            "Don't worry about it.",
            "Sorry, but alignment is an important topic to discuss.",
        ]),
    ]

    for prompt, responses in prompts_and_responses:
        result = ppo.train_step(prompt, responses)
        print(f"  Prompt: {result['prompt']}")
        print(f"  Best: {result['best_response']}...")
        print(f"  Reward: {result['reward']:.4f} | "
              f"Objective: {result['objective']:.4f} | "
              f"KL: {result['kl_penalty']:.4f}\n")
