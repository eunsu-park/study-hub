"""
Exercises for Lesson 07: Deep Q-Network (DQN)
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import deque


def exercise_1():
    """
    Exercise 1: Why Experience Replay Is Necessary

    Explain and demonstrate the effect of correlated samples
    on Q-network training without replay.
    """
    print("Why Experience Replay Is Necessary:")
    print("=" * 60)

    print()
    print("Problem with sequential samples:")
    print("  Consecutive transitions (s_t, a_t, s_{t+1}), (s_{t+1}, a_{t+1}, s_{t+2}), ...")
    print("  are highly correlated because each state leads to the next.")
    print()
    print("  In supervised learning: i.i.d. assumption is violated.")
    print("  Result: gradient updates overfit to recent trajectory,")
    print("          causing oscillations and divergence.")
    print()

    # Simulate: train a linear regressor with correlated vs iid data
    np.random.seed(42)
    n_steps = 500

    # Correlated data: random walk
    correlated_data = np.cumsum(np.random.randn(n_steps))
    correlated_data = correlated_data / (np.abs(correlated_data).max() + 1e-8)

    # IID data: uniform random
    iid_data = np.random.uniform(-1, 1, n_steps)

    # Gradient variance (proxy for training instability)
    def gradient_variance(data, window=50):
        """Rolling variance of data as proxy for gradient noise."""
        variances = []
        for i in range(window, len(data)):
            variances.append(np.var(data[i-window:i]))
        return np.mean(variances)

    var_corr = gradient_variance(correlated_data)
    var_iid = gradient_variance(iid_data)

    print(f"  Rolling variance (proxy for gradient instability):")
    print(f"  Sequential (correlated) data: {var_corr:.5f}")
    print(f"  Replay buffer (i.i.d.) data:  {var_iid:.5f}")
    print(f"  Variance ratio: {var_corr/var_iid:.2f}x")
    print()
    print("  Experience replay breaks correlation by randomly sampling")
    print("  from a large buffer, making training more stable.")

    # Replay buffer
    class ReplayBuffer:
        def __init__(self, capacity=10000):
            self.buffer = deque(maxlen=capacity)

        def push(self, *transition):
            self.buffer.append(transition)

        def sample(self, batch_size):
            idx = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in idx]

        def __len__(self):
            return len(self.buffer)

    buf = ReplayBuffer(capacity=1000)
    for i in range(500):
        buf.push(i, np.random.randint(2), float(i % 5), i + 1, False)

    batch = buf.sample(32)
    states_in_batch = [t[0] for t in batch]
    print(f"\n  Replay buffer sample (32 from 500): first 5 state indices:")
    print(f"  {sorted(states_in_batch)[:5]}")
    print("  => Indices are spread across the full buffer (not sequential).")


def exercise_2():
    """
    Exercise 2: Target Network — Why It Stabilizes Training

    Analyze the instability caused by using a live network as target.
    """
    print("\nTarget Network Stabilization:")
    print("=" * 60)

    print()
    print("Without target network:")
    print("  Update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a*) - Q(s,a)]")
    print("  Problem: Q(s',a*) is computed by the SAME network being updated.")
    print("  Each weight update changes the target, creating a moving target.")
    print()
    print("  Analogy: trying to hit a target while the target is moving.")
    print("           Each shot (gradient step) moves the bullseye.")
    print()

    # Simulate convergence with and without target network
    np.random.seed(0)
    true_q = 5.0  # true Q-value we want to learn
    n_steps = 200

    def train_with_target(use_target, alpha=0.1, gamma=0.9, sync_freq=10):
        Q = 0.0       # online Q estimate
        Q_target = 0.0  # target Q estimate
        q_history = []

        for step in range(n_steps):
            # Noisy target (simulates stochastic transitions)
            r = np.random.normal(true_q * (1 - gamma), 0.5)
            next_q = Q_target if use_target else Q

            td_target = r + gamma * next_q
            Q = Q + alpha * (td_target - Q)

            if use_target and step % sync_freq == 0:
                Q_target = Q  # periodic hard update

            q_history.append(Q)

        return q_history

    hist_live = train_with_target(use_target=False)
    hist_target = train_with_target(use_target=True, sync_freq=10)

    print(f"  True Q-value: {true_q}")
    print(f"\n  {'Steps':>8} | {'Live network':>14} | {'Target network':>16}")
    print("  " + "-" * 46)
    for step in [0, 49, 99, 149, 199]:
        print(f"  {step+1:>8} | {hist_live[step]:>14.4f} | {hist_target[step]:>16.4f}")

    # Oscillation measure (variance in last 50 steps)
    var_live = np.var(hist_live[-50:])
    var_target = np.var(hist_target[-50:])
    print(f"\n  Variance (last 50 steps):")
    print(f"  Live network:   {var_live:.6f}")
    print(f"  Target network: {var_target:.6f}")
    print(f"  => Target network is {var_live/var_target:.1f}x more stable.")


def exercise_3():
    """
    Exercise 3: Double DQN vs DQN — Overestimation Analysis

    DQN overestimates Q-values due to the max operator.
    Double DQN: use online network to SELECT action, target network to EVALUATE.
    """
    print("\nDouble DQN vs Standard DQN — Overestimation:")
    print("=" * 60)

    print()
    print("Standard DQN update:")
    print("  y = r + gamma * max_a' Q_target(s', a')")
    print("  Issue: argmax on noisy Q-values always picks the max noise component.")
    print()
    print("Double DQN update:")
    print("  a* = argmax_a' Q_online(s', a')    [SELECT with online network]")
    print("  y  = r + gamma * Q_target(s', a*)  [EVALUATE with target network]")
    print("  Since online and target networks have independent noise,")
    print("  the max-noise bias is greatly reduced.")
    print()

    np.random.seed(7)
    n_actions = 8
    true_q_next = np.zeros(n_actions)  # True Q-values are all 0
    n_experiments = 10000

    dqn_estimates = []
    ddqn_estimates = []

    for _ in range(n_experiments):
        # Simulate noisy Q-networks
        q_online = true_q_next + np.random.randn(n_actions)   # online net
        q_target = true_q_next + np.random.randn(n_actions)   # target net

        # Standard DQN: max over target network
        dqn_value = np.max(q_target)

        # Double DQN: select with online, evaluate with target
        best_action_online = np.argmax(q_online)
        ddqn_value = q_target[best_action_online]

        dqn_estimates.append(dqn_value)
        ddqn_estimates.append(ddqn_value)

    print(f"  True max Q-value (all 0.0, {n_actions} actions):")
    print(f"  DQN estimate (mean):    {np.mean(dqn_estimates):.4f} "
          f"(expected bias > 0)")
    print(f"  Double DQN estimate:    {np.mean(ddqn_estimates):.4f} "
          f"(expected ~0)")
    print(f"\n  DQN overestimation bias: {np.mean(dqn_estimates):.4f}")
    print(f"  Overestimation is E[max noise] > 0 — unavoidable with standard DQN.")


def exercise_4():
    """
    Exercise 4: Prioritized Experience Replay

    High-TD-error transitions are more informative.
    PER samples them more frequently, accelerating learning.
    """
    print("\nPrioritized Experience Replay (PER):")
    print("=" * 60)

    class PrioritizedReplayBuffer:
        """
        Simple proportional PER.
        Priority p_i = |delta_i| + epsilon
        P(i) = p_i^alpha / sum(p_j^alpha)
        """

        def __init__(self, capacity=1000, alpha=0.6, epsilon=1e-5):
            self.buffer = []
            self.priorities = []
            self.capacity = capacity
            self.alpha = alpha
            self.epsilon = epsilon

        def push(self, transition, td_error=1.0):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
                self.priorities.pop(0)
            self.buffer.append(transition)
            self.priorities.append(priority)

        def sample(self, batch_size):
            probs = np.array(self.priorities)
            probs = probs / probs.sum()
            indices = np.random.choice(len(self.buffer), batch_size,
                                       replace=False, p=probs)
            return [self.buffer[i] for i in indices], indices, probs[indices]

        def update_priorities(self, indices, td_errors):
            for i, td_err in zip(indices, td_errors):
                self.priorities[i] = (abs(td_err) + self.epsilon) ** self.alpha

    # Demonstrate: high-TD-error transitions get sampled more
    np.random.seed(42)
    per = PrioritizedReplayBuffer(capacity=100, alpha=0.6)

    # Add transitions with varying TD errors
    for i in range(100):
        td_error = np.random.exponential(scale=1.0)  # varies widely
        per.push(('transition', i), td_error=td_error)

    # Sample and check if high-priority transitions appear more often
    sample_counts = np.zeros(100)
    for _ in range(10000):
        batch, indices, _ = per.sample(10)
        for idx in indices:
            sample_counts[idx] += 1

    # Correlation between priority and sample count
    priorities = np.array(per.priorities)
    corr = np.corrcoef(priorities, sample_counts)[0, 1]

    print(f"\n  Buffer: 100 transitions with random TD errors")
    print(f"  Sampling correlation (priority vs count): {corr:.4f}")
    print(f"  (1.0 = perfect, high priority always sampled more)")
    print()

    # Show top-5 most sampled vs lowest 5
    top5 = np.argsort(sample_counts)[-5:]
    bot5 = np.argsort(sample_counts)[:5]
    print(f"  Top-5 most sampled transitions (index, priority, count):")
    for idx in top5:
        print(f"    idx={idx:3d}, priority={priorities[idx]:.4f}, "
              f"count={int(sample_counts[idx])}")
    print(f"\n  Bottom-5 least sampled transitions:")
    for idx in bot5:
        print(f"    idx={idx:3d}, priority={priorities[idx]:.4f}, "
              f"count={int(sample_counts[idx])}")


if __name__ == "__main__":
    print("=== Exercise 1: Experience Replay ===")
    exercise_1()

    print("\n=== Exercise 2: Target Network ===")
    exercise_2()

    print("\n=== Exercise 3: Double DQN ===")
    exercise_3()

    print("\n=== Exercise 4: Prioritized Experience Replay ===")
    exercise_4()

    print("\nAll exercises completed!")
