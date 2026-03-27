"""
Introduction to Reinforcement Learning — Example Code

Demonstrates core RL concepts:
  1. Agent-environment interaction loop
  2. Discounted return calculation
  3. Epsilon-greedy exploration with decay
  4. Multi-armed bandit as the simplest RL problem

No external dependencies required.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# 1. Agent-Environment Loop
# ============================================================

class SimpleGridEnv:
    """
    Minimal 1-D grid environment.
    States: 0..width-1
    Actions: 0=left, 1=right
    Goal: reach state (width-1)
    """

    def __init__(self, width=5):
        self.width = width
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.width - 1, self.state + 1)

        done = self.state == self.width - 1
        reward = 1.0 if done else -0.01
        return self.state, reward, done

    def render(self):
        row = ['.' for _ in range(self.width)]
        row[self.state] = 'A'
        row[-1] = 'G'
        print('|' + '|'.join(row) + '|')


def demonstrate_interaction_loop():
    """Show the fundamental agent-environment interaction loop."""
    print("=" * 60)
    print("1. Agent-Environment Interaction Loop")
    print("=" * 60)

    env = SimpleGridEnv(width=5)
    state = env.reset()
    total_reward = 0
    step = 0

    print(f"Initial state: {state}")
    env.render()

    done = False
    while not done and step < 20:
        # Random policy
        action = np.random.randint(2)
        next_state, reward, done = env.step(action)
        total_reward += reward
        step += 1

        action_name = 'left' if action == 0 else 'right'
        print(f"Step {step}: action={action_name}, state={next_state}, "
              f"reward={reward:.2f}, done={done}")
        env.render()

    print(f"\nEpisode finished in {step} steps, total reward: {total_reward:.2f}")


# ============================================================
# 2. Discounted Return
# ============================================================

def compute_returns(rewards, gamma=0.99):
    """
    Compute discounted returns G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    Uses backward recursion: O(n) time.
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def demonstrate_discounted_return():
    """Show how discounting affects the value of future rewards."""
    print("\n" + "=" * 60)
    print("2. Discounted Return")
    print("=" * 60)

    rewards = [0, 0, 0, 0, 10.0]  # Reward only at the end

    for gamma in [1.0, 0.99, 0.9, 0.5]:
        returns = compute_returns(rewards, gamma)
        print(f"gamma={gamma:.2f}: G_0={returns[0]:.4f}  "
              f"(rewards={rewards})")

    print("\nKey insight: smaller gamma makes the agent 'short-sighted'.")
    print("With gamma=0.5 a reward 5 steps away is worth only 0.5^5=0.03125"
          " of its face value.")


# ============================================================
# 3. Epsilon-Greedy Exploration
# ============================================================

class EpsilonGreedy:
    """Epsilon-greedy action selector with optional decay."""

    def __init__(self, n_actions, epsilon=1.0, epsilon_min=0.01, decay=0.995):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.q_values = np.zeros(n_actions)

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # explore
        return int(np.argmax(self.q_values))           # exploit

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)


def demonstrate_epsilon_greedy():
    """Show exploration-exploitation trade-off with epsilon decay."""
    print("\n" + "=" * 60)
    print("3. Epsilon-Greedy Exploration")
    print("=" * 60)

    agent = EpsilonGreedy(n_actions=4, epsilon=1.0, decay=0.99)

    # Simulate 500 steps
    n_steps = 500
    epsilon_log = []
    explore_count = 0

    for step in range(n_steps):
        action = agent.select_action()
        if np.random.random() < agent.epsilon:
            explore_count += 1
        agent.decay_epsilon()
        epsilon_log.append(agent.epsilon)

    print(f"After {n_steps} steps:")
    print(f"  Initial epsilon: 1.00")
    print(f"  Final epsilon:   {epsilon_log[-1]:.4f}")

    # Print epsilon at key milestones
    for milestone in [0, 99, 199, 299, 399, 499]:
        print(f"  Step {milestone+1:4d}: epsilon = {epsilon_log[milestone]:.4f}")


# ============================================================
# 4. Multi-Armed Bandit
# ============================================================

class BanditEnv:
    """k-armed bandit environment."""

    def __init__(self, k=10, seed=42):
        rng = np.random.RandomState(seed)
        self.means = rng.normal(0, 1, k)
        self.k = k

    def pull(self, arm):
        return np.random.normal(self.means[arm], 1.0)

    def optimal_arm(self):
        return int(np.argmax(self.means))


def run_bandit(env, epsilon=0.1, n_steps=1000):
    """Run epsilon-greedy bandit and return cumulative reward history."""
    q = np.zeros(env.k)
    n = np.zeros(env.k)
    cumulative_rewards = []
    total = 0.0

    for _ in range(n_steps):
        if np.random.random() < epsilon:
            action = np.random.randint(env.k)
        else:
            action = int(np.argmax(q))

        reward = env.pull(action)
        n[action] += 1
        q[action] += (reward - q[action]) / n[action]  # incremental mean
        total += reward
        cumulative_rewards.append(total)

    return cumulative_rewards, q


def demonstrate_bandit():
    """Compare epsilon values in a 10-armed bandit."""
    print("\n" + "=" * 60)
    print("4. Multi-Armed Bandit")
    print("=" * 60)

    env = BanditEnv(k=10)
    print(f"Optimal arm: {env.optimal_arm()} (mean={env.means[env.optimal_arm()]:.3f})")
    print()

    for eps in [0.0, 0.01, 0.1, 0.5]:
        np.random.seed(0)
        rewards, q_final = run_bandit(env, epsilon=eps, n_steps=1000)
        best_found = int(np.argmax(q_final))
        print(f"  epsilon={eps:.2f}: total_reward={rewards[-1]:.1f}, "
              f"learned_best_arm={best_found}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    demonstrate_interaction_loop()
    demonstrate_discounted_return()
    demonstrate_epsilon_greedy()
    demonstrate_bandit()

    print("\n" + "=" * 60)
    print("RL Introduction examples complete!")
    print("=" * 60)
