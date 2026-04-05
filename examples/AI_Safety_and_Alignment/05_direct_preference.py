# 05_direct_preference.py — DPO loss implementation and preference data
#
# Run: python 05_direct_preference.py

"""
Implements Direct Preference Optimization (DPO) loss computation.
DPO simplifies RLHF by directly optimizing a policy from preference
data without a separate reward model or RL loop.
"""

import math
import random
from dataclasses import dataclass


@dataclass
class PreferenceSample:
    prompt: str
    chosen: str
    rejected: str


class SimpleTokenizer:
    """Minimal tokenizer mapping words to integer IDs."""

    def __init__(self):
        self.vocab = {}
        self._next_id = 1

    def encode(self, text: str) -> list[int]:
        tokens = text.lower().split()
        ids = []
        for t in tokens:
            if t not in self.vocab:
                self.vocab[t] = self._next_id
                self._next_id += 1
            ids.append(self.vocab[t])
        return ids


class SimplePolicy:
    """Simple log-probability model using learnable word scores."""

    def __init__(self, tokenizer: SimpleTokenizer, seed: int = 42):
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.word_scores: dict[int, float] = {}

    def _get_score(self, token_id: int) -> float:
        if token_id not in self.word_scores:
            self.word_scores[token_id] = self.rng.gauss(0, 0.1)
        return self.word_scores[token_id]

    def log_prob(self, prompt: str, response: str) -> float:
        """Compute log probability of response given prompt."""
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)
        total = 0.0
        for tid in response_ids:
            score = self._get_score(tid)
            # Simplified log-softmax approximation
            total += score - math.log(1 + math.exp(score))
        return total

    def copy(self) -> "SimplePolicy":
        """Create a frozen copy for reference policy."""
        new = SimplePolicy(self.tokenizer, seed=42)
        new.word_scores = dict(self.word_scores)
        return new


def dpo_loss(policy: SimplePolicy, ref_policy: SimplePolicy,
             sample: PreferenceSample, beta: float = 0.1) -> float:
    """
    Compute DPO loss for a single preference pair.

    L_DPO = -log(sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x)
                              - log pi(y_l|x)/pi_ref(y_l|x))))

    Args:
        policy: Current policy being optimized
        ref_policy: Frozen reference policy (SFT model)
        sample: Preference pair (chosen, rejected)
        beta: Temperature parameter controlling deviation from reference
    """
    # Log ratios: log(pi/pi_ref) for chosen and rejected
    log_ratio_chosen = (policy.log_prob(sample.prompt, sample.chosen) -
                        ref_policy.log_prob(sample.prompt, sample.chosen))
    log_ratio_rejected = (policy.log_prob(sample.prompt, sample.rejected) -
                          ref_policy.log_prob(sample.prompt, sample.rejected))

    # DPO loss: -log(sigmoid(beta * (log_ratio_w - log_ratio_l)))
    logit = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -math.log(1.0 / (1.0 + math.exp(-logit)) + 1e-10)
    return loss


def dpo_gradient_step(policy: SimplePolicy, ref_policy: SimplePolicy,
                      sample: PreferenceSample,
                      beta: float = 0.1, lr: float = 0.01):
    """Approximate gradient step for DPO using finite differences."""
    epsilon = 1e-4
    current_loss = dpo_loss(policy, ref_policy, sample, beta)

    for tid in list(policy.word_scores.keys()):
        original = policy.word_scores[tid]
        policy.word_scores[tid] = original + epsilon
        loss_plus = dpo_loss(policy, ref_policy, sample, beta)
        grad = (loss_plus - current_loss) / epsilon
        policy.word_scores[tid] = original - lr * grad

    return current_loss


def build_preference_dataset() -> list[PreferenceSample]:
    """Build synthetic preference dataset."""
    return [
        PreferenceSample(
            "Explain machine learning",
            "Machine learning is a subset of AI where models learn "
            "patterns from data to make predictions and decisions.",
            "ML is when computers do stuff automatically."
        ),
        PreferenceSample(
            "What is climate change?",
            "Climate change refers to long-term shifts in temperatures "
            "and weather patterns, primarily driven by human activities.",
            "The weather changes sometimes, that's climate change."
        ),
        PreferenceSample(
            "How does encryption work?",
            "Encryption transforms readable data into ciphertext using "
            "mathematical algorithms and keys for secure communication.",
            "It scrambles your data so nobody can read it."
        ),
        PreferenceSample(
            "Describe neural networks",
            "Neural networks are computing systems inspired by biological "
            "neurons, organized in layers that process information.",
            "They're like brains but for computers, I guess."
        ),
        PreferenceSample(
            "What is quantum computing?",
            "Quantum computing leverages quantum mechanical phenomena like "
            "superposition and entanglement for parallel computation.",
            "Super fast computers using quantum physics or something."
        ),
    ]


def compare_rlhf_vs_dpo():
    """Print comparison of RLHF vs DPO approaches."""
    print("=== RLHF vs DPO Comparison ===\n")
    comparison = [
        ("Reward Model", "Required (separate training)", "Not needed"),
        ("RL Algorithm", "PPO (complex, unstable)", "Not needed"),
        ("Training Steps", "RM + RL (two phases)", "Single phase"),
        ("Hyperparameters", "Many (PPO clip, KL coeff)", "Few (beta)"),
        ("Stability", "Can be unstable", "Generally stable"),
        ("Memory", "High (4 models in memory)", "Low (2 models)"),
        ("Theoretical", "Approximate optimization", "Closed-form solution"),
    ]
    print(f"  {'Aspect':<20} {'RLHF':<35} {'DPO':<30}")
    print(f"  {'─' * 20} {'─' * 35} {'─' * 30}")
    for aspect, rlhf, dpo_val in comparison:
        print(f"  {aspect:<20} {rlhf:<35} {dpo_val:<30}")
    print()


if __name__ == "__main__":
    print("=== Direct Preference Optimization (DPO) ===\n")

    tokenizer = SimpleTokenizer()
    policy = SimplePolicy(tokenizer, seed=42)
    ref_policy = policy.copy()

    dataset = build_preference_dataset()

    # Training loop
    print("--- DPO Training ---\n")
    betas = [0.05, 0.1, 0.5]

    for beta in betas:
        policy_b = SimplePolicy(tokenizer, seed=42)
        ref_b = policy_b.copy()

        print(f"  Beta = {beta}")
        for epoch in range(30):
            total_loss = 0.0
            for sample in dataset:
                loss = dpo_gradient_step(policy_b, ref_b, sample,
                                        beta=beta, lr=0.01)
                total_loss += loss
            avg = total_loss / len(dataset)
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d} | Loss: {avg:.4f}")
        print()

    # Evaluate preference accuracy
    print("--- Preference Accuracy After Training ---\n")
    policy_trained = SimplePolicy(tokenizer, seed=42)
    ref_trained = policy_trained.copy()

    for epoch in range(50):
        for sample in dataset:
            dpo_gradient_step(policy_trained, ref_trained, sample,
                              beta=0.1, lr=0.01)

    correct = 0
    for sample in dataset:
        sc = policy_trained.log_prob(sample.prompt, sample.chosen)
        sr = policy_trained.log_prob(sample.prompt, sample.rejected)
        if sc > sr:
            correct += 1
        print(f"  Prompt: {sample.prompt[:35]}")
        print(f"    Chosen logp: {sc:.3f} | Rejected logp: {sr:.3f} | "
              f"{'OK' if sc > sr else 'WRONG'}")

    print(f"\n  Accuracy: {correct}/{len(dataset)}\n")

    compare_rlhf_vs_dpo()
