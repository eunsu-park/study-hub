"""
Offline Reinforcement Learning — Example Code

Demonstrates:
  1. Dataset collection and distribution shift visualization
  2. Naive Q-learning on offline data (overestimation problem)
  3. Conservative Q-Learning (CQL) penalty concept
  4. Behavior Cloning baseline

No external gym/PyTorch dependency — NumPy only.
"""

import numpy as np
from collections import deque


# ============================================================
# 1. Offline Dataset and Distribution Shift
# ============================================================

class SimpleEnv:
    """
    1-D corridor: states 0..9, actions {left, right}.
    Goal: reach state 9 (reward +10), step cost -0.1.
    """

    N_STATES = 10
    N_ACTIONS = 2  # 0=left, 1=right

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 1:  # right
            self.state = min(self.N_STATES - 1, self.state + 1)
        else:            # left
            self.state = max(0, self.state - 1)
        done = self.state == self.N_STATES - 1
        reward = 10.0 if done else -0.1
        return self.state, reward, done


def collect_offline_dataset(env, behavior_epsilon=0.5, n_episodes=200, seed=0):
    """
    Collect a dataset using a sub-optimal behavior policy
    (epsilon-greedy with high epsilon = lots of exploration but poor coverage
    of the goal region).
    """
    rng = np.random.RandomState(seed)
    dataset = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        for _ in range(50):
            # Behavior policy: biased towards right but noisy
            if rng.random() < behavior_epsilon:
                action = rng.randint(env.N_ACTIONS)
            else:
                action = 1  # go right
            next_state, reward, done = env.step(action)
            dataset.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                break

    return dataset


def demonstrate_distribution_shift():
    """Show that offline data does not cover the full state-action space."""
    print("=" * 60)
    print("1. Offline Dataset and Distribution Shift")
    print("=" * 60)

    env = SimpleEnv()
    dataset = collect_offline_dataset(env, behavior_epsilon=0.7)

    # State-action visitation counts
    visit_counts = np.zeros((env.N_STATES, env.N_ACTIONS))
    for s, a, r, ns, d in dataset:
        visit_counts[s, a] += 1

    total = len(dataset)
    print(f"\n  Dataset size: {total} transitions")
    print(f"\n  State-action visitation (rows=states, cols=[left,right]):")
    print(f"  {'State':>6} | {'Left':>8} | {'Right':>8} | {'Left%':>7} | {'Right%':>7}")
    print("  " + "-" * 46)
    for s in range(env.N_STATES):
        l, r = int(visit_counts[s, 0]), int(visit_counts[s, 1])
        print(f"  {s:>6} | {l:>8} | {r:>8} | "
              f"{100*l/total:>6.1f}% | {100*r/total:>6.1f}%")

    unvisited = np.sum(visit_counts == 0)
    print(f"\n  Unvisited (s,a) pairs: {unvisited}/{env.N_STATES * env.N_ACTIONS}")
    print("  => Policy may encounter states not in dataset (distribution shift).")


# ============================================================
# 2. Naive Offline Q-Learning vs CQL
# ============================================================

def train_q_table(dataset, n_states, n_actions, gamma=0.99, lr=0.1, n_epochs=50):
    """Offline Q-learning: iterate over dataset multiple times."""
    Q = np.zeros((n_states, n_actions))

    for epoch in range(n_epochs):
        np.random.shuffle(dataset)
        for s, a, r, ns, done in dataset:
            if done:
                target = r
            else:
                target = r + gamma * np.max(Q[ns])
            Q[s, a] += lr * (target - Q[s, a])

    return Q


def train_q_table_cql(dataset, n_states, n_actions, gamma=0.99, lr=0.1,
                      alpha=1.0, n_epochs=50):
    """
    Offline Q-learning with CQL penalty.
    CQL adds: alpha * (logsumexp(Q(s,*)) - Q(s,a_behavior))
    This conservative penalty pulls down Q-values for unobserved actions.
    """
    Q = np.zeros((n_states, n_actions))

    for epoch in range(n_epochs):
        np.random.shuffle(dataset)
        for s, a, r, ns, done in dataset:
            if done:
                target = r
            else:
                target = r + gamma * np.max(Q[ns])

            # Standard TD update
            td_error = target - Q[s, a]
            Q[s, a] += lr * td_error

            # CQL conservative penalty on all actions in state s
            # Penalizes Q-values for actions not in dataset
            log_sum = np.log(np.sum(np.exp(Q[s] - np.max(Q[s]))) + 1e-10) + np.max(Q[s])
            for a2 in range(n_actions):
                Q[s, a2] -= lr * alpha * (np.exp(Q[s, a2] - log_sum))
                Q[s, a] += lr * alpha * (1.0 / n_actions)  # add back average

    return Q


def evaluate_policy(Q, env, n_episodes=100):
    """Evaluate the greedy policy derived from Q."""
    total_rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        for _ in range(50):
            action = int(np.argmax(Q[state]))
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        total_rewards.append(total_reward)
    return np.mean(total_rewards), np.std(total_rewards)


def demonstrate_cql_vs_naive():
    """Compare naive offline QL with CQL conservative approach."""
    print("\n" + "=" * 60)
    print("2. Naive Offline Q-Learning vs CQL")
    print("=" * 60)

    env = SimpleEnv()
    dataset = collect_offline_dataset(env, behavior_epsilon=0.7, n_episodes=200)

    Q_naive = train_q_table(dataset, env.N_STATES, env.N_ACTIONS)
    Q_cql = train_q_table_cql(dataset, env.N_STATES, env.N_ACTIONS, alpha=2.0)

    print("\n  Q-values for unvisited state (s=9, near goal):")
    print(f"  Naive: Q(9, left)={Q_naive[9,0]:.3f}, Q(9, right)={Q_naive[9,1]:.3f}")
    print(f"  CQL:   Q(9, left)={Q_cql[9,0]:.3f},  Q(9, right)={Q_cql[9,1]:.3f}")

    mean_naive, std_naive = evaluate_policy(Q_naive, env)
    mean_cql, std_cql = evaluate_policy(Q_cql, env)

    print(f"\n  Policy evaluation (100 episodes):")
    print(f"  Naive:  mean_reward={mean_naive:.2f} ± {std_naive:.2f}")
    print(f"  CQL:    mean_reward={mean_cql:.2f} ± {std_cql:.2f}")
    print("  => CQL's conservative penalty reduces overestimation of unseen actions.")


# ============================================================
# 3. Behavior Cloning Baseline
# ============================================================

def train_behavior_cloning(dataset, n_states, n_actions, n_epochs=50):
    """
    Behavior cloning: supervised learning on (state, action) pairs.
    Learns pi_BC(a|s) = argmax count(s, a) in dataset.
    """
    # Count action frequencies per state
    action_counts = np.zeros((n_states, n_actions))
    for s, a, r, ns, done in dataset:
        action_counts[s, a] += 1

    # Policy: pick most common action; uniform if no data
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        if action_counts[s].sum() > 0:
            policy[s] = int(np.argmax(action_counts[s]))
        else:
            policy[s] = 1  # default: go right

    return policy


def evaluate_bc_policy(policy, env, n_episodes=100):
    """Evaluate a deterministic BC policy."""
    total_rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        for _ in range(50):
            action = policy[state]
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        total_rewards.append(total_reward)
    return np.mean(total_rewards), np.std(total_rewards)


def demonstrate_behavior_cloning():
    """Show that behavior cloning is limited by behavior policy quality."""
    print("\n" + "=" * 60)
    print("3. Behavior Cloning Baseline")
    print("=" * 60)

    env = SimpleEnv()

    for eps in [0.1, 0.5, 0.9]:
        dataset = collect_offline_dataset(env, behavior_epsilon=eps)
        bc_policy = train_behavior_cloning(dataset, env.N_STATES, env.N_ACTIONS)
        mean_r, std_r = evaluate_bc_policy(bc_policy, env)
        print(f"\n  Behavior epsilon={eps}: BC policy reward = {mean_r:.2f} ± {std_r:.2f}")
        print(f"  Learned policy: {bc_policy}")

    print("\n  => BC upper bound = behavior policy quality.")
    print("     With noisy behavior (eps=0.9), BC also becomes noisy.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demonstrate_distribution_shift()
    demonstrate_cql_vs_naive()
    demonstrate_behavior_cloning()

    print("\n" + "=" * 60)
    print("Offline RL examples complete!")
    print("=" * 60)
