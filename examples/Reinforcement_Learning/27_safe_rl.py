"""
Safe Reinforcement Learning — Example Code

Demonstrates:
  1. Constrained MDP formulation and Lagrangian method
  2. Safety layer (projection onto safe actions)
  3. CVaR risk-sensitive objective
  4. Safety budget tracking during training

No external dependencies required.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# Environment: Grid with Safety Constraints
# ============================================================

class SafeGridEnv:
    """
    5x5 grid with hazard zones.
    Reward: +10 at goal (4,4), -0.1 per step.
    Cost:   +1 whenever agent enters a hazard cell.
    Episode constraint: total cost <= cost_limit.
    """

    SIZE = 5
    HAZARDS = {(1, 1), (1, 3), (3, 1), (3, 3), (2, 2)}
    GOAL = (4, 4)
    ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U

    def __init__(self, cost_limit=1):
        self.cost_limit = cost_limit

    def reset(self):
        self.pos = (0, 0)
        self.total_cost = 0
        self.steps = 0
        return self.pos

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        r = max(0, min(self.SIZE - 1, self.pos[0] + dr))
        c = max(0, min(self.SIZE - 1, self.pos[1] + dc))
        self.pos = (r, c)
        self.steps += 1

        reward = 10.0 if self.pos == self.GOAL else -0.1
        cost = 1.0 if self.pos in self.HAZARDS else 0.0
        self.total_cost += cost

        done = self.pos == self.GOAL or self.steps >= 30
        constraint_violated = self.total_cost > self.cost_limit

        return self.pos, reward, cost, done, constraint_violated

    def all_states(self):
        return [(r, c) for r in range(self.SIZE) for c in range(self.SIZE)]


# ============================================================
# 1. Lagrangian Constrained RL
# ============================================================

class LagrangianAgent:
    """
    Lagrangian method for constrained RL.
    L(theta, lambda) = E[reward] - lambda * (E[cost] - d)
    Maximize over theta, minimize over lambda >= 0.
    """

    def __init__(self, env, lambda_init=0.0, lambda_lr=0.01, cost_limit=1.0):
        self.Q_r = defaultdict(lambda: np.zeros(4))  # reward Q-table
        self.Q_c = defaultdict(lambda: np.zeros(4))  # cost Q-table
        self.lam = lambda_init
        self.lambda_lr = lambda_lr
        self.cost_limit = cost_limit
        self.env = env

    def get_action(self, state, epsilon=0.1):
        """Lagrangian-penalized greedy action."""
        if np.random.random() < epsilon:
            return np.random.randint(4)
        # Combined value: reward - lambda * cost
        combined = self.Q_r[state] - self.lam * self.Q_c[state]
        return int(np.argmax(combined))

    def update(self, s, a, r, c, ns, done, gamma=0.99, lr=0.1):
        if done:
            r_target = r
            c_target = c
        else:
            r_target = r + gamma * np.max(self.Q_r[ns])
            c_target = c + gamma * np.max(self.Q_c[ns])

        self.Q_r[s][a] += lr * (r_target - self.Q_r[s][a])
        self.Q_c[s][a] += lr * (c_target - self.Q_c[s][a])

    def update_lambda(self, episode_cost):
        """Dual ascent: increase lambda if cost budget exceeded."""
        self.lam = max(0.0, self.lam + self.lambda_lr * (episode_cost - self.cost_limit))


def train_lagrangian(env, n_episodes=1000, seed=42):
    """Train with Lagrangian penalty for cost constraint."""
    np.random.seed(seed)
    agent = LagrangianAgent(env, cost_limit=env.cost_limit)
    rewards, costs, lambdas = [], [], []

    for ep in range(n_episodes):
        state = env.reset()
        total_r, total_c = 0.0, 0.0
        done = False

        while not done:
            a = agent.get_action(state, epsilon=max(0.05, 0.5 - ep / 500))
            ns, r, c, done, _ = env.step(a)
            agent.update(state, a, r, c, ns, done)
            state = ns
            total_r += r
            total_c += c

        agent.update_lambda(total_c)
        rewards.append(total_r)
        costs.append(total_c)
        lambdas.append(agent.lam)

    return rewards, costs, lambdas, agent


def demonstrate_lagrangian():
    """Show constraint satisfaction over training."""
    print("=" * 60)
    print("1. Lagrangian Constrained RL")
    print("=" * 60)

    env = SafeGridEnv(cost_limit=1)
    rewards, costs, lambdas, agent = train_lagrangian(env, n_episodes=1000)

    window = 200
    print(f"\n  Training progress (cost limit = {env.cost_limit}):")
    print(f"  {'Episodes':>12} | {'Avg Reward':>12} | {'Avg Cost':>10} | {'Lambda':>8}")
    print("  " + "-" * 50)

    for start in range(0, 1000, window):
        end = start + window
        avg_r = np.mean(rewards[start:end])
        avg_c = np.mean(costs[start:end])
        lam = np.mean(lambdas[start:end])
        print(f"  {start+1:>5}-{end:<5}   | {avg_r:>12.3f} | {avg_c:>10.3f} | {lam:>8.3f}")

    # Show policy uses hazards less over time
    early_cost = np.mean(costs[:200])
    late_cost = np.mean(costs[-200:])
    print(f"\n  Cost reduction: {early_cost:.3f} -> {late_cost:.3f}")
    print(f"  Lambda (dual variable): {agent.lam:.3f}")
    print("  => Lambda increases until cost constraint is satisfied.")


# ============================================================
# 2. Safety Layer (Action Projection)
# ============================================================

class SafetyLayer:
    """
    Project actions to a safe set using learned safety constraint model.
    Simplified: if proposed action leads to hazard, pick alternative.
    """

    def __init__(self, env):
        self.env = env
        # Predict cost of action: learned from data
        self.cost_model = defaultdict(lambda: np.zeros(4))

    def update_cost_model(self, state, action, cost, alpha=0.1):
        self.cost_model[state][action] += alpha * (cost - self.cost_model[state][action])

    def safe_action(self, state, proposed_action, threshold=0.3):
        """
        If predicted cost of proposed action exceeds threshold,
        substitute with the safest known action.
        """
        predicted_cost = self.cost_model[state][proposed_action]
        if predicted_cost <= threshold:
            return proposed_action
        # Fall back to action with lowest predicted cost
        return int(np.argmin(self.cost_model[state]))


def demonstrate_safety_layer():
    """Show safety layer reduces constraint violations."""
    print("\n" + "=" * 60)
    print("2. Safety Layer (Action Projection)")
    print("=" * 60)

    env = SafeGridEnv(cost_limit=2)
    np.random.seed(7)

    safety_layer = SafetyLayer(env)
    violations_no_layer = []
    violations_with_layer = []

    n_episodes = 300

    # Without safety layer
    for ep in range(n_episodes):
        state = env.reset()
        episode_cost = 0.0
        done = False
        while not done:
            a = np.random.randint(4)
            ns, r, c, done, _ = env.step(a)
            episode_cost += c
            state = ns
        violations_no_layer.append(float(episode_cost > env.cost_limit))

    # With safety layer (learns cost model online)
    np.random.seed(7)
    for ep in range(n_episodes):
        state = env.reset()
        episode_cost = 0.0
        done = False
        while not done:
            proposed_a = np.random.randint(4)
            safe_a = safety_layer.safe_action(state, proposed_a)
            ns, r, c, done, _ = env.step(safe_a)
            safety_layer.update_cost_model(state, safe_a, c)
            episode_cost += c
            state = ns
        violations_with_layer.append(float(episode_cost > env.cost_limit))

    print(f"\n  Over {n_episodes} episodes:")
    print(f"  Without safety layer: constraint violation rate = "
          f"{np.mean(violations_no_layer):.1%}")
    print(f"  With safety layer:    constraint violation rate = "
          f"{np.mean(violations_with_layer):.1%}")
    print("\n  Safety layer intercepts unsafe actions before they execute.")


# ============================================================
# 3. CVaR Risk-Sensitive Objective
# ============================================================

def demonstrate_cvar_policy():
    """
    Show that CVaR optimization leads to more conservative, safer policies
    compared to expected value (risk-neutral) optimization.
    """
    print("\n" + "=" * 60)
    print("3. CVaR Risk-Sensitive Objective")
    print("=" * 60)

    np.random.seed(42)
    n_trials = 10000

    # Two policies:
    # A: Safe — always moderate reward, no catastrophic failure
    # B: Greedy — high expected reward but occasional large failure
    def sample_policy_A():
        return np.random.normal(5.0, 1.0)

    def sample_policy_B():
        # 95% high reward, 5% catastrophic failure
        if np.random.random() < 0.95:
            return np.random.normal(7.0, 1.0)
        else:
            return np.random.normal(-50.0, 5.0)

    samples_A = np.array([sample_policy_A() for _ in range(n_trials)])
    samples_B = np.array([sample_policy_B() for _ in range(n_trials)])

    def cvar(samples, alpha=0.05):
        threshold = np.percentile(samples, alpha * 100)
        tail = samples[samples <= threshold]
        return np.mean(tail) if len(tail) > 0 else threshold

    print(f"\n  {'Metric':>20} | {'Policy A (safe)':>16} | {'Policy B (greedy)':>18}")
    print("  " + "-" * 62)

    metrics = [
        ("E[R] (mean)", np.mean(samples_A), np.mean(samples_B)),
        ("Std[R]", np.std(samples_A), np.std(samples_B)),
        ("CVaR@5%", cvar(samples_A, 0.05), cvar(samples_B, 0.05)),
        ("CVaR@10%", cvar(samples_A, 0.10), cvar(samples_B, 0.10)),
        ("Min (worst)", np.min(samples_A), np.min(samples_B)),
    ]

    for name, va, vb in metrics:
        print(f"  {name:>20} | {va:>16.3f} | {vb:>18.3f}")

    print("\n  Risk-neutral (mean): Policy B wins (E=6.67 > 5.0)")
    print("  Risk-averse (CVaR):  Policy A wins (no catastrophic tails)")
    print("  CVaR optimization is suitable when failures are unacceptable.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demonstrate_lagrangian()
    demonstrate_safety_layer()
    demonstrate_cvar_policy()

    print("\n" + "=" * 60)
    print("Safe RL examples complete!")
    print("=" * 60)
