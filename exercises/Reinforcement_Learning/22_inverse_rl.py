"""
Exercises for Lesson 22: Inverse Reinforcement Learning
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


def exercise_1():
    """
    Exercise 1: Forward vs Inverse RL

    Articulate the difference between forward and inverse RL
    and when each is applicable.
    """
    print("Forward vs Inverse Reinforcement Learning:")
    print("=" * 60)

    directions = [
        (
            "Forward RL",
            "Given: Environment dynamics + Reward function R\n"
            "   Find: Optimal policy π*\n"
            "   Direction: R -> π*\n"
            "   Use when: You can specify the reward, want to find the policy.",
            "   Examples: Game playing, robotics with known objectives"
        ),
        (
            "Inverse RL",
            "Given: Expert demonstrations D = {τ₁, τ₂, ...}\n"
            "   Find: Reward function R that explains demonstrations\n"
            "   Direction: π* -> R (then use R for forward RL)\n"
            "   Use when: You cannot specify reward, but have expert data.",
            "   Examples: Imitate driving, surgery, animal behavior"
        ),
        (
            "Imitation Learning (BC)",
            "Given: Expert demonstrations D\n"
            "   Find: Policy π that mimics expert\n"
            "   Direction: π* -> π_clone (skip reward entirely)\n"
            "   Use when: You only need to copy behavior, not understand goals.",
            "   Examples: Robot manipulation with fixed task"
        ),
    ]

    for name, desc, examples in directions:
        print(f"\n  {name}:")
        print(f"   {desc}")
        print(f"   {examples}")

    # Numerical example: recover a reward from demonstrations
    print("\n  Quick numerical check:")
    print("  Expert always goes right in a 5-state chain (0->1->2->3->4).")
    print("  IRL should recover: reward increases toward state 4.")

    # Expert demonstrations
    demos_state_freqs = {0: 0.1, 1: 0.2, 2: 0.25, 3: 0.25, 4: 0.2}
    # IRL gradient: increase reward for states expert visits more
    theta = np.zeros(5)
    lr = 0.1
    for _ in range(50):
        for s, freq in demos_state_freqs.items():
            theta[s] += lr * freq
    theta -= theta.min()
    print(f"\n  Recovered reward (IRL): {theta.round(3)}")
    print(f"  Expected: increasing from state 0 to state 4.")


def exercise_2():
    """
    Exercise 2: Maximum Entropy IRL

    Implement MaxEnt IRL and verify it recovers the correct reward.
    """
    print("\nMaximum Entropy IRL — Reward Recovery:")
    print("=" * 60)

    np.random.seed(42)

    # 1-D chain: states 0-4, actions: left(0) / right(1)
    n_states = 5
    n_actions = 2
    gamma = 0.99

    # True reward: +10 at state 4, else 0
    true_reward = np.array([0.0, 0.0, 0.0, 0.0, 10.0])

    def transition(s, a):
        if a == 1: return min(n_states-1, s+1)
        return max(0, s-1)

    def get_optimal_policy(reward):
        """Value iteration to get optimal policy given reward."""
        V = np.zeros(n_states)
        for _ in range(100):
            for s in range(n_states):
                V[s] = max(reward[s] + gamma * V[transition(s, a)] for a in range(n_actions))
        policy = np.zeros((n_states, n_actions))
        for s in range(n_states):
            q_vals = [reward[s] + gamma * V[transition(s, a)] for a in range(n_actions)]
            best = np.argmax(q_vals)
            policy[s, best] = 1.0
        return policy

    def state_visitation(policy, n_steps=20):
        """Compute state visitation frequency under policy."""
        visit = np.zeros(n_states)
        for start in range(n_states):
            s = start
            for t in range(n_steps):
                visit[s] += (gamma**t) / n_states
                a = np.argmax(policy[s])
                s = transition(s, a)
        return visit / visit.sum()

    # Expert state visitation
    expert_policy = get_optimal_policy(true_reward)
    expert_visit = state_visitation(expert_policy)

    # MaxEnt IRL
    theta = np.zeros(n_states)
    losses = []

    for it in range(100):
        current_policy = get_optimal_policy(theta)
        agent_visit = state_visitation(current_policy)
        grad = expert_visit - agent_visit
        theta += 0.1 * grad
        losses.append(np.sum(np.abs(grad)))

    print(f"\n  True reward:     {true_reward}")
    print(f"  Recovered theta: {theta.round(3)}")

    # Normalize for comparison
    theta_norm = theta - theta.min()
    theta_norm = theta_norm / (theta_norm.max() + 1e-10)
    true_norm = true_reward / (true_reward.max() + 1e-10)

    corr = np.corrcoef(theta_norm, true_norm)[0, 1]
    print(f"\n  Correlation (recovered vs true): {corr:.4f}")
    print(f"  IRL gradient convergence: {losses[0]:.4f} -> {losses[-1]:.5f}")


def exercise_3():
    """
    Exercise 3: Preference-Based Reward Learning (RLHF Foundation)

    Implement Bradley-Terry preference model to learn reward from comparisons.
    """
    print("\nPreference-Based Reward Learning:")
    print("=" * 60)

    np.random.seed(7)

    # True reward weights: [speed_bonus, safety_penalty, comfort_bonus]
    true_w = np.array([2.0, -3.0, 1.0])
    n_features = len(true_w)

    def generate_trajectory_features():
        return np.random.uniform(0, 1, n_features)

    def human_prefers_A(phi_A, phi_B, noise=0.3):
        """Human comparison: A preferred over B if r(A) > r(B) + noise."""
        r_A = np.dot(true_w, phi_A) + np.random.randn() * noise
        r_B = np.dot(true_w, phi_B)
        return r_A > r_B

    # Collect pairwise preferences
    n_pairs = 300
    preferences = []
    for _ in range(n_pairs):
        phi_A = generate_trajectory_features()
        phi_B = generate_trajectory_features()
        pref_A = human_prefers_A(phi_A, phi_B)
        preferences.append((phi_A, phi_B, pref_A))

    # Learn reward weights
    w = np.zeros(n_features)
    lr = 0.1

    for epoch in range(200):
        grad = np.zeros(n_features)
        for phi_A, phi_B, pref_A in preferences:
            diff = phi_A - phi_B
            logit = np.dot(w, diff)
            p_A = 1 / (1 + np.exp(-logit))
            if pref_A:
                grad += (1 - p_A) * diff
            else:
                grad -= p_A * diff
        w += lr * grad / n_pairs

    print(f"\n  True reward weights:    {true_w}")
    print(f"  Learned weights:        {w.round(3)}")
    print(f"\n  Sign check (should match true):")
    for i, (tw, lw) in enumerate(zip(true_w, w)):
        match = "✓" if np.sign(tw) == np.sign(lw) else "✗"
        print(f"    w[{i}]: true={tw:+.1f}, learned={lw:+.3f}  {match}")

    # Prediction accuracy
    correct = sum(
        1 for phi_A, phi_B, pref_A in preferences
        if (np.dot(w, phi_A) > np.dot(w, phi_B)) == pref_A
    )
    print(f"\n  Preference prediction accuracy: {correct/n_pairs:.1%}")


def exercise_4():
    """
    Exercise 4: GAIL Concept — Generator vs Discriminator

    Show the adversarial structure of GAIL conceptually
    with a simple simulation.
    """
    print("\nGAIL — Adversarial Imitation Learning Concept:")
    print("=" * 60)

    print()
    print("GAIL structure:")
    print("  Generator G (policy π):  produces (state, action) pairs")
    print("  Discriminator D:         distinguishes expert from generator")
    print()
    print("  D objective: maximize log D(s,a) + log(1 - D(π(s), a))")
    print("  G objective: maximize log D(π(s), a)  [fool discriminator]")
    print("  => Policy is trained to produce behaviors indistinguishable")
    print("     from expert. Reward signal = D's confidence.")
    print()

    np.random.seed(3)

    # Expert: 2D Gaussian at (5, 5), std=0.5
    # Policy starts at (0, 0), std=2.0
    def expert_sample(n=1):
        return np.random.normal([5, 5], 0.5, (n, 2))

    def policy_sample(mu, sigma, n=1):
        return np.random.normal(mu, sigma, (n, 2))

    def train_discriminator(expert_data, policy_data, w, n_steps=20, lr=0.1):
        """Linear discriminator: D(x) = sigmoid(w^T x + b)"""
        b = 0.0
        for _ in range(n_steps):
            for x, label in [(expert_data, 1), (policy_data, 0)]:
                for xi in x:
                    logit = np.dot(w, xi) + b
                    p = 1 / (1 + np.exp(-logit))
                    grad_w = (label - p) * xi
                    grad_b = label - p
                    w += lr * grad_w
                    b += lr * grad_b
        return w, b

    # Simulate GAIL training for a few iterations
    policy_mu = np.array([0.0, 0.0])
    policy_sigma = 2.0
    w = np.random.randn(2) * 0.1

    print(f"  Initial policy: mu={policy_mu}")
    print(f"  Expert:         mu=[5, 5]")
    print()
    print(f"  GAIL training (5 iterations):")
    print(f"  {'Iter':>5} | {'Policy mu':>20} | {'D accuracy':>12}")
    print("  " + "-" * 44)

    for iteration in range(5):
        expert_data = expert_sample(50)
        policy_data = policy_sample(policy_mu, policy_sigma, 50)

        # Train discriminator
        w, b = train_discriminator(expert_data, policy_data, w.copy())

        # Update policy: move toward expert using discriminator gradient
        p_logit = np.array([np.dot(w, x) + b for x in policy_data])
        p_scores = 1 / (1 + np.exp(-p_logit))
        # Policy gradient: maximize E[log D(π(s), a)]
        policy_grad = np.mean((1 - p_scores)[:, None] * (expert_data - policy_data), axis=0)
        policy_mu += 0.5 * policy_grad
        policy_sigma = max(0.3, policy_sigma * 0.9)

        # Discriminator accuracy
        expert_scores = [1 / (1 + np.exp(-(np.dot(w, x) + b))) for x in expert_sample(100)]
        policy_scores = [1 / (1 + np.exp(-(np.dot(w, x) + b))) for x in policy_data]
        acc = (np.mean(np.array(expert_scores) > 0.5) + np.mean(np.array(policy_scores) < 0.5)) / 2

        print(f"  {iteration+1:>5} | mu=[{policy_mu[0]:>6.2f}, {policy_mu[1]:>6.2f}]    | {acc:>10.1%}")

    print(f"\n  Policy moved from (0,0) toward expert at (5,5).")
    print(f"  GAIL recovers the behavior without explicit reward specification.")


if __name__ == "__main__":
    print("=== Exercise 1: Forward vs Inverse RL ===")
    exercise_1()

    print("\n=== Exercise 2: MaxEnt IRL ===")
    exercise_2()

    print("\n=== Exercise 3: Preference-Based Reward Learning ===")
    exercise_3()

    print("\n=== Exercise 4: GAIL Concept ===")
    exercise_4()

    print("\nAll exercises completed!")
