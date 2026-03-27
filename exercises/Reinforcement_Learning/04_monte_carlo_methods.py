"""
Exercises for Lesson 04: Monte Carlo Methods
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


def exercise_1():
    """
    First-visit vs Every-visit: Calculate the difference between the two
    methods for an episode visiting the same state 3 times.

    Episode: s0 -> s1 -> s2 -> s1 -> s3 -> s1 -> terminal
    Rewards:     1      2      3      4      5      6
    gamma = 1.0 (undiscounted for clarity)
    """
    print("Episode: s0 -> s1 -> s2 -> s1 -> s3 -> s1 -> terminal")
    print("Rewards:     1      2      3      4      5      6")
    print("gamma = 1.0")
    print()

    # Episode data: (state, action, reward)
    episode = [
        ('s0', 'a', 1),
        ('s1', 'a', 2),
        ('s2', 'a', 3),
        ('s1', 'a', 4),
        ('s3', 'a', 5),
        ('s1', 'a', 6),
    ]
    gamma = 1.0

    # Calculate returns for each timestep
    returns_at_t = []
    G = 0
    for t in range(len(episode) - 1, -1, -1):
        _, _, r = episode[t]
        G = r + gamma * G
        returns_at_t.insert(0, G)

    print("Returns at each timestep:")
    for t, (state, _, reward) in enumerate(episode):
        print(f"  t={t}: state={state}, reward={reward}, G_t={returns_at_t[t]}")
    print()

    # First-visit MC for s1: only count the FIRST occurrence (t=1)
    first_visit_s1 = None
    for t, (state, _, _) in enumerate(episode):
        if state == 's1':
            first_visit_s1 = returns_at_t[t]
            print(f"First-visit MC for s1:")
            print(f"  First occurrence at t={t}, G = {first_visit_s1}")
            break

    # Every-visit MC for s1: count ALL occurrences
    every_visit_returns = []
    for t, (state, _, _) in enumerate(episode):
        if state == 's1':
            every_visit_returns.append(returns_at_t[t])

    every_visit_s1 = np.mean(every_visit_returns)
    print(f"\nEvery-visit MC for s1:")
    print(f"  Occurrences at t=1 (G={every_visit_returns[0]}), "
          f"t=3 (G={every_visit_returns[1]}), t=5 (G={every_visit_returns[2]})")
    print(f"  Average: ({' + '.join(str(g) for g in every_visit_returns)}) / 3 "
          f"= {every_visit_s1:.2f}")

    print(f"\nDifference: |First-visit - Every-visit| = "
          f"|{first_visit_s1} - {every_visit_s1:.2f}| = "
          f"{abs(first_visit_s1 - every_visit_s1):.2f}")
    print()
    print("Note: First-visit gives an unbiased estimate. Every-visit uses")
    print("more data points but with correlated samples from the same episode.")
    print("In practice, both converge to the true value as episodes increase.")


def exercise_2():
    """
    Exploration Problem: What problem occurs when epsilon=0 in epsilon-greedy?

    Solution: Explain and demonstrate the problem with a simple multi-armed bandit.
    """
    print("Problem with epsilon=0 in epsilon-greedy:")
    print("=" * 50)
    print()
    print("When epsilon=0, the agent NEVER explores. It always exploits")
    print("the action with the highest estimated Q-value (pure greedy).")
    print()
    print("This causes several problems:")
    print("1. The agent may lock onto a suboptimal action if the initial")
    print("   Q-estimates are wrong (which they almost always are).")
    print("2. Many state-action pairs may never be visited, so their")
    print("   Q-values remain at their initial (arbitrary) values.")
    print("3. The agent cannot discover better strategies.")
    print()

    # Demonstration with a simple bandit problem
    print("Demonstration: 5-armed bandit")
    print("-" * 40)

    np.random.seed(42)
    n_arms = 5
    true_means = [1.0, 2.0, 3.0, 1.5, 0.5]  # Arm 2 (index 2) is best
    n_steps = 1000

    def run_bandit(epsilon, n_steps, true_means):
        q_estimates = np.zeros(len(true_means))
        action_counts = np.zeros(len(true_means))
        total_reward = 0

        for step in range(n_steps):
            if np.random.random() < epsilon:
                action = np.random.randint(len(true_means))
            else:
                action = np.argmax(q_estimates)

            reward = np.random.normal(true_means[action], 1.0)
            total_reward += reward
            action_counts[action] += 1
            n = action_counts[action]
            q_estimates[action] += (reward - q_estimates[action]) / n

        return total_reward / n_steps, q_estimates, action_counts

    # Run with epsilon=0 (pure greedy)
    avg_reward_0, q_est_0, counts_0 = run_bandit(0.0, n_steps, true_means)
    print(f"\nepsilon=0 (pure greedy):")
    print(f"  Average reward: {avg_reward_0:.3f}")
    print(f"  Action counts: {counts_0.astype(int)}")
    print(f"  Q-estimates: {np.round(q_est_0, 2)}")
    print(f"  Best arm found: {np.argmax(q_est_0)} (true best: 2)")

    # Run with epsilon=0.1
    avg_reward_1, q_est_1, counts_1 = run_bandit(0.1, n_steps, true_means)
    print(f"\nepsilon=0.1:")
    print(f"  Average reward: {avg_reward_1:.3f}")
    print(f"  Action counts: {counts_1.astype(int)}")
    print(f"  Q-estimates: {np.round(q_est_1, 2)}")
    print(f"  Best arm found: {np.argmax(q_est_1)} (true best: 2)")

    print(f"\nConclusion: epsilon=0 got stuck on arm {np.argmax(counts_0)} "
          f"and missed the optimal arm 2.")
    print(f"epsilon=0.1 explored enough to find the optimal arm.")


def exercise_3():
    """
    Importance Sampling: When can importance ratios diverge if the target
    policy is deterministic and behavior policy is epsilon-greedy?

    Solution: Explain and demonstrate with a numerical example.
    """
    print("Importance Sampling Ratio Divergence:")
    print("=" * 50)
    print()
    print("The importance sampling ratio is:")
    print("  rho = product_{k=t}^{T-1} pi(a_k|s_k) / b(a_k|s_k)")
    print()
    print("For a deterministic target policy pi and epsilon-greedy behavior b:")
    print("  pi(a|s) = 1 if a = pi*(s), else 0")
    print("  b(a|s) = 1 - epsilon + epsilon/|A| if a = argmax Q,")
    print("           epsilon/|A| otherwise")
    print()
    print("The ratio for a single step is:")
    print("  1 / b(a|s) when a matches the target policy")
    print("  0 when a does NOT match the target policy")
    print()
    print("DIVERGENCE occurs when:")
    print("  The target action has LOW probability under the behavior policy.")
    print("  Specifically, if pi*(s) != argmax_Q(s) at that state,")
    print("  then b(a|s) = epsilon/|A|, and the ratio = 1/(epsilon/|A|) = |A|/epsilon")
    print()
    print("  For a long trajectory where target and greedy actions differ at many steps:")
    print("  rho = product of (|A|/epsilon)^k, which grows exponentially!")
    print()

    # Numerical example
    n_actions = 4
    epsilon = 0.1
    print(f"Example: |A|={n_actions}, epsilon={epsilon}")
    print(f"  Ratio when target matches greedy: 1 / (1 - {epsilon} + {epsilon}/{n_actions})")
    match_ratio = 1.0 / (1 - epsilon + epsilon / n_actions)
    print(f"    = {match_ratio:.4f}")
    print()

    mismatch_ratio = 1.0 / (epsilon / n_actions)
    print(f"  Ratio when target does NOT match greedy: 1 / ({epsilon}/{n_actions})")
    print(f"    = {mismatch_ratio:.1f}")
    print()

    print("For a trajectory of length T with k mismatches:")
    for T in [1, 5, 10, 20]:
        for k in [0, 1, 2, 5]:
            if k > T:
                continue
            rho = match_ratio ** (T - k) * mismatch_ratio ** k
            print(f"  T={T:2d}, mismatches={k}: rho = {rho:.2e}")
        print()

    print("Conclusion: Even a few mismatches cause the ratio to explode,")
    print("leading to extremely high variance in the value estimate.")
    print("This is why weighted importance sampling and per-decision IS")
    print("are preferred in practice.")


def exercise_4():
    """
    Convergence Speed: Explain why MC has high variance and solutions.

    Solution: Demonstrate variance of MC estimates and show how to reduce it.
    """
    print("Why Monte Carlo Has High Variance:")
    print("=" * 50)
    print()
    print("MC estimates V(s) by averaging returns G from complete episodes.")
    print("The return G = r_1 + gamma*r_2 + gamma^2*r_3 + ... is a sum of")
    print("many random variables, so its variance can be very large:")
    print()
    print("  Var(G) = Var(sum of T discounted rewards)")
    print("         >= sum of individual reward variances")
    print()
    print("Sources of variance:")
    print("  1. Stochastic policy: different actions at same state")
    print("  2. Stochastic environment: different transitions")
    print("  3. Long episodes: more random variables in the sum")
    print()

    # Demonstrate with a simple random walk
    np.random.seed(42)
    n_states = 7  # 0=left terminal, 1-5=states, 6=right terminal
    true_values = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0])

    def run_mc_episode(start_state=3, gamma=1.0):
        """Run one MC episode on the random walk"""
        state = start_state
        rewards = []
        while 0 < state < 6:
            if np.random.random() < 0.5:
                state -= 1
            else:
                state += 1
            reward = 1.0 if state == 6 else 0.0
            rewards.append(reward)
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
        return G

    # Show variance of MC returns
    n_episodes_list = [10, 100, 1000, 10000]
    start_state = 3
    true_v = true_values[start_state]

    print(f"\nMC estimates for state {start_state} (true value = {true_v:.4f}):")
    print("-" * 60)

    for n_episodes in n_episodes_list:
        returns = [run_mc_episode(start_state) for _ in range(n_episodes)]
        mc_estimate = np.mean(returns)
        mc_variance = np.var(returns)
        mc_std = np.std(returns)
        mc_error = abs(mc_estimate - true_v)
        print(f"  N={n_episodes:5d}: V_hat={mc_estimate:.4f}, "
              f"Var={mc_variance:.4f}, Std={mc_std:.4f}, "
              f"Error={mc_error:.4f}")

    # Solutions to reduce variance
    print(f"\nVariance Reduction Solutions:")
    print("-" * 60)

    # Solution 1: Use a baseline (subtract mean return)
    print("\n1. Baseline Subtraction:")
    returns = [run_mc_episode(start_state) for _ in range(1000)]
    raw_variance = np.var(returns)
    baseline = np.mean(returns)
    centered_returns = [r - baseline for r in returns]
    centered_variance = np.var(centered_returns)
    print(f"   Raw variance: {raw_variance:.4f}")
    print(f"   Centered variance: {centered_variance:.4f}")
    print(f"   (Baseline shifts mean to 0 but variance stays the same)")
    print(f"   (In policy gradient, baseline reduces gradient variance)")

    # Solution 2: Use more episodes
    print("\n2. More Episodes (Law of Large Numbers):")
    print(f"   Std of mean estimate = Std / sqrt(N)")
    for N in [10, 100, 1000, 10000]:
        sem = np.std(returns[:min(N, len(returns))]) / np.sqrt(N)
        print(f"   N={N:5d}: Standard Error of Mean = {sem:.4f}")

    # Solution 3: TD learning (lower variance, some bias)
    print("\n3. Use TD Learning Instead (trades bias for lower variance):")
    print("   TD uses 1-step bootstrap: V(s) <- V(s) + alpha[r + gamma*V(s') - V(s)]")
    print("   Only one reward is random (lower variance) but V(s') may be")
    print("   inaccurate (introduces bias).")

    # Solution 4: Discount factor
    print("\n4. Lower Discount Factor (gamma < 1):")
    for gamma_val in [1.0, 0.99, 0.9, 0.5]:
        returns_g = [run_mc_episode(start_state, gamma=gamma_val) for _ in range(1000)]
        print(f"   gamma={gamma_val}: Var={np.var(returns_g):.4f}, "
              f"Mean={np.mean(returns_g):.4f}")


if __name__ == "__main__":
    print("=== Exercise 1: First-visit vs Every-visit ===")
    exercise_1()

    print("\n=== Exercise 2: Exploration Problem ===")
    exercise_2()

    print("\n=== Exercise 3: Importance Sampling ===")
    exercise_3()

    print("\n=== Exercise 4: Convergence Speed ===")
    exercise_4()

    print("\nAll exercises completed!")
