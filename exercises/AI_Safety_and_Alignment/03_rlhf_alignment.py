# Exercise: Lesson 03 — RLHF Alignment
# Complete the TODO items below.
#
# Run: python 03_rlhf_alignment.py

import math


def compute_preference_loss(preferred_score: float,
                            rejected_score: float) -> float:
    """Compute the Bradley-Terry preference loss for a single pair.

    The loss is: -log(sigmoid(preferred_score - rejected_score))

    Args:
        preferred_score: Reward model score for the preferred response.
        rejected_score: Reward model score for the rejected response.

    Returns:
        The scalar loss value.
    """
    # TODO: Compute the difference between preferred and rejected scores.

    # TODO: Apply the sigmoid function: sigmoid(x) = 1 / (1 + exp(-x)).

    # TODO: Return the negative log of the sigmoid value.
    pass


def train_reward_model(comparisons: list[dict],
                       learning_rate: float = 0.01,
                       epochs: int = 10) -> dict:
    """Train a simple linear reward model from pairwise comparisons.

    Args:
        comparisons: List of dicts with "preferred_features" (list of float),
                     "rejected_features" (list of float).
        learning_rate: Step size for gradient updates.
        epochs: Number of training passes.

    Returns:
        dict with "weights" (list of float) and "final_loss" (float).
    """
    # TODO: Initialize weights to zeros (same length as feature vectors).

    # TODO: For each epoch, iterate over comparisons:
    #   - Compute scores as dot product of weights and features.
    #   - Compute gradient of the preference loss w.r.t. weights.
    #   - Update weights using gradient descent.

    # TODO: Return final weights and the loss on the last epoch.
    pass


def simulate_rlhf_loop(initial_policy: dict, reward_fn: callable,
                       num_rounds: int = 5) -> list[dict]:
    """Simulate a simplified RLHF training loop.

    Args:
        initial_policy: dict mapping "prompt" -> list of possible "responses".
        reward_fn: Function(response_str) -> float returning a reward score.
        num_rounds: Number of RLHF rounds to simulate.

    Returns:
        List of dicts per round: "round", "best_response", "reward",
        "policy_entropy" (measure of response diversity).
    """
    # TODO: Start with the initial policy (uniform distribution over responses).

    # TODO: Each round:
    #   - Score all responses with reward_fn.
    #   - Shift probability mass toward higher-reward responses.
    #   - Track the best response and policy entropy.

    # TODO: Return the per-round history.
    pass


def detect_reward_model_overoptimization(
    rewards: list[float], true_quality: list[float]
) -> dict:
    """Detect overoptimization of the reward model (reward hacking).

    Args:
        rewards: Proxy reward model scores over training steps.
        true_quality: True quality scores (from gold evaluator) over steps.

    Returns:
        dict with "overoptimized" (bool), "peak_quality_step" (int),
        "quality_drop" (float), "reward_at_peak" (float).
    """
    # TODO: Find the step where true_quality is maximized.

    # TODO: Check if reward continues to increase after true_quality peaks.

    # TODO: Compute the drop in true quality from peak to final step.

    # TODO: Return the analysis.
    pass


if __name__ == "__main__":
    # Test preference loss
    loss = compute_preference_loss(2.5, 1.0)
    print(f"Preference loss: {loss}")

    # Test reward model training
    comparisons = [
        {"preferred_features": [1.0, 0.5, 0.2],
         "rejected_features": [0.3, 0.8, 0.1]},
        {"preferred_features": [0.9, 0.3, 0.7],
         "rejected_features": [0.1, 0.9, 0.4]},
    ]
    model = train_reward_model(comparisons)
    print(f"Trained model: {model}")

    # Test RLHF loop
    policy = {"What is AI?": ["AI is machines.", "AI is intelligence.",
                               "AI is artificial intelligence by machines."]}
    reward = lambda r: len(r) * 0.1  # simple length-based reward
    history = simulate_rlhf_loop(policy, reward)
    print(f"RLHF history: {history}")

    # Test overoptimization detection
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    quality = [1.0, 1.8, 2.5, 2.3, 1.9, 1.2]
    result = detect_reward_model_overoptimization(rewards, quality)
    print(f"Overoptimization: {result}")
