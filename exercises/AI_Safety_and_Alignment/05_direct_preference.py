# Exercise: Lesson 05 — Direct Preference Optimization
# Complete the TODO items below.
#
# Run: python 05_direct_preference.py

import math


def dpo_loss(pi_logprob_preferred: float, pi_logprob_rejected: float,
             ref_logprob_preferred: float, ref_logprob_rejected: float,
             beta: float = 0.1) -> float:
    """Compute the DPO loss for a single preference pair.

    DPO loss = -log(sigmoid(beta * (
        (pi_logprob_preferred - ref_logprob_preferred) -
        (pi_logprob_rejected - ref_logprob_rejected)
    )))

    Args:
        pi_logprob_preferred: Policy log-prob of preferred response.
        pi_logprob_rejected: Policy log-prob of rejected response.
        ref_logprob_preferred: Reference model log-prob of preferred.
        ref_logprob_rejected: Reference model log-prob of rejected.
        beta: Temperature parameter controlling deviation from reference.

    Returns:
        Scalar DPO loss value.
    """
    # TODO: Compute the log-ratio differences for preferred and rejected.

    # TODO: Scale by beta and apply sigmoid.

    # TODO: Return -log(sigmoid(...)).
    pass


def compute_implicit_reward(pi_logprob: float, ref_logprob: float,
                            beta: float = 0.1) -> float:
    """Compute the implicit reward from DPO's closed-form solution.

    Implicit reward = beta * (pi_logprob - ref_logprob)

    Args:
        pi_logprob: Policy log-probability.
        ref_logprob: Reference model log-probability.
        beta: Temperature parameter.

    Returns:
        The implicit reward value.
    """
    # TODO: Implement the implicit reward formula.
    pass


def compare_rlhf_vs_dpo(dataset: list[dict]) -> dict:
    """Compare RLHF and DPO on a preference dataset (conceptual analysis).

    Args:
        dataset: List of dicts with "prompt", "preferred", "rejected",
                 "preference_strength" (float 0-1).

    Returns:
        dict with:
            - "rlhf_advantages": list of str
            - "dpo_advantages": list of str
            - "dataset_suitability": "rlhf" or "dpo" with reasoning
            - "estimated_dpo_beta": recommended beta based on data
    """
    # TODO: Analyze the dataset characteristics (size, preference strength
    # distribution, diversity of prompts).

    # TODO: List advantages of each method for this dataset.

    # TODO: Recommend which method suits this dataset better.

    # TODO: Estimate an appropriate beta value based on preference
    # strength distribution (higher beta for clearer preferences).
    pass


def simulate_dpo_training(pairs: list[dict], beta: float = 0.1,
                          learning_rate: float = 0.01,
                          epochs: int = 20) -> dict:
    """Simulate DPO training with simple linear policy and reference models.

    Args:
        pairs: List of dicts with "preferred_features" (list of float)
               and "rejected_features" (list of float).
        beta: DPO temperature parameter.
        learning_rate: Gradient descent step size.
        epochs: Number of training passes.

    Returns:
        dict with "final_weights", "loss_history" (list of float),
        "implicit_rewards_preferred" (list), "implicit_rewards_rejected" (list).
    """
    # TODO: Initialize policy weights and fix reference weights at zero.

    # TODO: For each epoch, compute DPO loss and gradients over all pairs.

    # TODO: Update policy weights via gradient descent.

    # TODO: After training, compute implicit rewards for all examples.

    # TODO: Return results.
    pass


if __name__ == "__main__":
    # Test DPO loss
    loss = dpo_loss(-1.0, -2.0, -1.5, -1.8, beta=0.1)
    print(f"DPO loss: {loss}")

    # Test implicit reward
    reward = compute_implicit_reward(-1.0, -1.5, beta=0.1)
    print(f"Implicit reward: {reward}")

    # Test RLHF vs DPO comparison
    dataset = [
        {"prompt": "Explain gravity", "preferred": "clear explanation",
         "rejected": "vague answer", "preference_strength": 0.9},
        {"prompt": "Write a poem", "preferred": "creative poem",
         "rejected": "generic poem", "preference_strength": 0.6},
    ]
    comparison = compare_rlhf_vs_dpo(dataset)
    print(f"Comparison: {comparison}")

    # Test DPO training simulation
    pairs = [
        {"preferred_features": [1.0, 0.5], "rejected_features": [0.2, 0.8]},
        {"preferred_features": [0.8, 0.3], "rejected_features": [0.1, 0.9]},
    ]
    result = simulate_dpo_training(pairs)
    print(f"DPO training: {result}")
