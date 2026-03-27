"""
RLHF Deep Dive — Example Code

Demonstrates:
  1. Reward model training from pairwise human preferences
  2. KL penalty to prevent reward hacking (KL-constrained objective)
  3. Direct Preference Optimization (DPO) loss computation
  4. Reward model overfitting / Goodhart's Law illustration

No external dependencies required.
"""

import numpy as np


# ============================================================
# 1. Reward Model from Pairwise Preferences
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def train_reward_model(preferences, n_features, n_epochs=200, lr=0.05,
                       l2_reg=0.01, seed=0):
    """
    Learn a linear reward model from pairwise preferences.

    Dataset: list of (phi_chosen, phi_rejected) feature pairs
    Loss: -E[log sigma(r(chosen) - r(rejected))]  (Bradley-Terry model)
    """
    rng = np.random.RandomState(seed)
    weights = rng.randn(n_features) * 0.1
    losses = []

    for epoch in range(n_epochs):
        grad = np.zeros(n_features)
        total_loss = 0.0

        for phi_c, phi_r in preferences:
            r_c = np.dot(weights, phi_c)
            r_r = np.dot(weights, phi_r)
            diff = r_c - r_r
            prob = sigmoid(diff)
            loss = -np.log(prob + 1e-10)
            total_loss += loss
            # Gradient of -log sigma(r_c - r_r) wrt weights
            grad_diff = prob - 1.0  # d(-log sigma(x))/dx = sigma(x) - 1
            grad += grad_diff * (phi_c - phi_r)

        # L2 regularization
        grad += l2_reg * weights

        weights -= lr * grad / len(preferences)
        losses.append(total_loss / len(preferences))

    return weights, losses


def generate_preference_data(n_pairs=300, n_features=5, seed=42):
    """
    Simulate human preferences.
    True reward: weighted sum of features (known to us for evaluation).
    """
    rng = np.random.RandomState(seed)
    true_weights = rng.randn(n_features)

    data = []
    for _ in range(n_pairs):
        phi_a = rng.randn(n_features)
        phi_b = rng.randn(n_features)
        r_a = np.dot(true_weights, phi_a)
        r_b = np.dot(true_weights, phi_b)
        # Human comparison with noise
        noise = rng.randn() * 0.3
        if (r_a + noise) > r_b:
            data.append((phi_a, phi_b))  # preferred phi_a
        else:
            data.append((phi_b, phi_a))  # preferred phi_b

    return data, true_weights


def demonstrate_reward_model():
    """Train and evaluate a reward model from preference data."""
    print("=" * 60)
    print("1. Reward Model Training from Pairwise Preferences")
    print("=" * 60)

    data, true_w = generate_preference_data(n_pairs=500, n_features=5)
    learned_w, losses = train_reward_model(data, n_features=5, n_epochs=300)

    # Normalize for cosine comparison
    corr = np.corrcoef(true_w, learned_w)[0, 1]
    cos_sim = np.dot(true_w, learned_w) / (
        np.linalg.norm(true_w) * np.linalg.norm(learned_w) + 1e-10)

    print(f"\n  True weights:    {true_w.round(3)}")
    print(f"  Learned weights: {learned_w.round(3)}")
    print(f"  Pearson correlation: {corr:.3f}")
    print(f"  Cosine similarity:   {cos_sim:.3f}")
    print(f"  Training loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Preference accuracy
    correct = 0
    for phi_c, phi_r in data:
        pred_preferred = np.dot(learned_w, phi_c) > np.dot(learned_w, phi_r)
        correct += int(pred_preferred)
    print(f"  Preference prediction accuracy: {correct/len(data):.1%}")


# ============================================================
# 2. KL Penalty — Preventing Reward Hacking
# ============================================================

def demonstrate_kl_penalty():
    """
    Show how KL divergence penalty prevents the policy from
    deviating too far from the reference (base) policy.
    """
    print("\n" + "=" * 60)
    print("2. KL Penalty in RLHF (Preventing Reward Hacking)")
    print("=" * 60)

    np.random.seed(0)
    n_actions = 10

    # Reference policy (uniform)
    pi_ref = np.ones(n_actions) / n_actions

    # "Hacked" reward: very high reward for action 0, others near zero
    def hacked_reward(action):
        return 10.0 if action == 0 else 0.1

    def kl_divergence(p, q):
        """KL(p || q)"""
        mask = p > 1e-10
        return np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-10)))

    def rlhf_objective(pi, beta):
        """
        RLHF objective: E_pi[r(a)] - beta * KL(pi || pi_ref)
        """
        expected_reward = sum(pi[a] * hacked_reward(a) for a in range(n_actions))
        kl = kl_divergence(pi, pi_ref)
        return expected_reward - beta * kl

    def optimize_policy(beta, n_iters=200, lr=0.05):
        """Gradient ascent on RLHF objective with softmax policy."""
        logits = np.zeros(n_actions)
        for _ in range(n_iters):
            # Softmax
            logits_s = logits - np.max(logits)
            pi = np.exp(logits_s) / np.sum(np.exp(logits_s))

            # Gradient of RLHF objective wrt logits
            # dL/d_logit_i = pi_i * [r_i - E[r] - beta * (log pi_i - log pi_ref_i)
            #                        + beta * KL]
            e_r = sum(pi[a] * hacked_reward(a) for a in range(n_actions))
            for i in range(n_actions):
                ri = hacked_reward(i)
                advantage = ri - e_r - beta * (np.log(pi[i] + 1e-10) -
                                               np.log(pi_ref[i] + 1e-10))
                grad_i = pi[i] * advantage
                logits[i] += lr * grad_i

        logits_s = logits - np.max(logits)
        return np.exp(logits_s) / np.sum(np.exp(logits_s))

    print(f"\n  Reward function: action 0 -> 10.0, all others -> 0.1")
    print(f"  Reference policy: uniform over {n_actions} actions")
    print()
    print(f"  {'Beta':>6} | {'P(a=0)':>8} | {'E[r]':>8} | {'KL':>8} | {'Objective':>10}")
    print("  " + "-" * 50)

    for beta in [0.0, 0.1, 0.5, 2.0, 10.0]:
        pi_opt = optimize_policy(beta)
        e_r = sum(pi_opt[a] * hacked_reward(a) for a in range(n_actions))
        kl = kl_divergence(pi_opt, pi_ref)
        obj = rlhf_objective(pi_opt, beta)
        print(f"  {beta:>6.1f} | {pi_opt[0]:>8.3f} | {e_r:>8.3f} | "
              f"{kl:>8.3f} | {obj:>10.3f}")

    print("\n  => Large beta keeps policy close to reference (safe).")
    print("     Small beta allows reward hacking (exploits reward model).")


# ============================================================
# 3. Direct Preference Optimization (DPO) Loss
# ============================================================

def dpo_loss(log_prob_chosen_policy, log_prob_rejected_policy,
             log_prob_chosen_ref, log_prob_rejected_ref, beta=0.1):
    """
    DPO loss (Rafailov et al., 2023).

    L_DPO = -log sigma(beta * (log pi(y_w|x) - log pi_ref(y_w|x))
                       - beta * (log pi(y_l|x) - log pi_ref(y_l|x)))

    where y_w = chosen (preferred) response, y_l = rejected response.
    """
    log_ratio_chosen = log_prob_chosen_policy - log_prob_chosen_ref
    log_ratio_rejected = log_prob_rejected_policy - log_prob_rejected_ref
    reward_diff = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -np.log(sigmoid(reward_diff) + 1e-10)
    return loss, reward_diff


def demonstrate_dpo():
    """Show DPO loss computation and comparison with PPO approach."""
    print("\n" + "=" * 60)
    print("3. Direct Preference Optimization (DPO)")
    print("=" * 60)

    print("\n  DPO computes implicit rewards from policy ratios:")
    print("  r(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x))")
    print()

    scenarios = [
        {
            'name': 'Policy strongly prefers chosen',
            'log_pi_c': -1.0, 'log_pi_r': -4.0,
            'log_ref_c': -2.0, 'log_ref_r': -2.0,
        },
        {
            'name': 'Policy does not discriminate',
            'log_pi_c': -2.0, 'log_pi_r': -2.0,
            'log_ref_c': -2.0, 'log_ref_r': -2.0,
        },
        {
            'name': 'Policy incorrectly prefers rejected',
            'log_pi_c': -4.0, 'log_pi_r': -1.0,
            'log_ref_c': -2.0, 'log_ref_r': -2.0,
        },
    ]

    for sc in scenarios:
        loss, reward_diff = dpo_loss(
            sc['log_pi_c'], sc['log_pi_r'],
            sc['log_ref_c'], sc['log_ref_r'], beta=0.1)
        print(f"  Scenario: {sc['name']}")
        print(f"    Implicit reward diff: {reward_diff:.4f}  DPO loss: {loss:.4f}")
        print()

    print("  DPO advantages over PPO for LLM fine-tuning:")
    print("  - No separate reward model (reward implicit in policy ratio)")
    print("  - No RL sampling loop — just supervised fine-tuning")
    print("  - Simpler to implement and more stable")
    print("  - But: assumes preferences are well-calibrated and IID")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demonstrate_reward_model()
    demonstrate_kl_penalty()
    demonstrate_dpo()

    print("\n" + "=" * 60)
    print("RLHF Deep Dive examples complete!")
    print("=" * 60)
