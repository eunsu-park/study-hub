"""
Imitation Learning — Example Code

Demonstrates:
  1. Behavioral Cloning and compounding error problem
  2. DAgger: iterative dataset aggregation
  3. Comparison: BC vs DAgger on distribution shift

No external dependencies required.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# Environment: 1-D Continuous Driving
# ============================================================

class RoadFollowEnv:
    """
    1-D road following task.
    State: lateral offset from center line (continuous).
    Action: steering correction (continuous, clipped to [-1, 1]).
    Goal: keep offset near 0 for as long as possible.
    """

    def __init__(self, noise_std=0.05, wind=0.0):
        self.noise_std = noise_std   # sensor noise
        self.wind = wind             # external disturbance
        self.max_steps = 50
        self.max_offset = 2.0

    def reset(self):
        self.offset = np.random.uniform(-0.5, 0.5)
        self.step_count = 0
        return self.observe()

    def observe(self):
        return self.offset + np.random.randn() * self.noise_std

    def step(self, steering):
        steering = np.clip(steering, -1.0, 1.0)
        self.offset = self.offset - 0.5 * steering + self.wind
        self.offset += np.random.randn() * 0.02  # process noise
        self.step_count += 1
        obs = self.observe()
        done = abs(self.offset) > self.max_offset or self.step_count >= self.max_steps
        reward = 1.0 - abs(self.offset) / self.max_offset
        return obs, reward, done

    def expert_action(self, obs):
        """Expert: proportional controller (optimal in this linear system)."""
        return np.clip(obs * 2.0, -1.0, 1.0)


# ============================================================
# 1. Behavior Cloning
# ============================================================

class LinearPolicy:
    """Simple linear policy: steering = w * obs."""

    def __init__(self):
        self.w = 0.0
        self.bias = 0.0

    def predict(self, obs):
        return np.clip(self.w * obs + self.bias, -1.0, 1.0)

    def fit(self, states, actions, n_epochs=100, lr=0.01):
        """Least-squares regression on (state, action) pairs."""
        X = np.array(states).reshape(-1, 1)
        y = np.array(actions)
        # Normal equations: w = (X^T X)^{-1} X^T y
        Xt = X.T
        XtX = Xt @ X + 1e-6 * np.eye(1)  # regularization
        Xty = Xt @ y
        self.w = (np.linalg.solve(XtX, Xty))[0]
        self.bias = np.mean(y) - self.w * np.mean(states)
        return self


def train_bc(env, n_expert_episodes=20, seed=0):
    """Collect expert demonstrations and train BC policy."""
    np.random.seed(seed)
    states, actions = [], []

    for _ in range(n_expert_episodes):
        obs = env.reset()
        done = False
        while not done:
            a = env.expert_action(obs)
            states.append(obs)
            actions.append(a)
            obs, _, done = env.step(a)

    policy = LinearPolicy()
    policy.fit(states, actions)
    return policy, len(states)


# ============================================================
# 2. DAgger
# ============================================================

def train_dagger(env, n_iterations=10, n_episodes_per_iter=5, seed=0):
    """
    DAgger: at each iteration
      1. Run current policy to collect states
      2. Query expert for labels on those states
      3. Aggregate all data and retrain
    """
    np.random.seed(seed)
    all_states, all_actions = [], []
    policy = LinearPolicy()
    policy.w = 0.1  # small initial guess

    for iteration in range(n_iterations):
        # Collect states using current policy
        new_states = []
        for _ in range(n_episodes_per_iter):
            obs = env.reset()
            done = False
            while not done:
                a = policy.predict(obs)
                new_states.append(obs)
                obs, _, done = env.step(a)

        # Query expert for labels (DAgger key step)
        new_actions = [env.expert_action(s) for s in new_states]

        # Aggregate
        all_states.extend(new_states)
        all_actions.extend(new_actions)

        # Retrain on aggregated dataset
        policy.fit(all_states, all_actions)

    return policy, len(all_states)


# ============================================================
# 3. Evaluation and Comparison
# ============================================================

def evaluate(policy_fn, env, n_episodes=100, use_expert=False, seed=99):
    """
    Evaluate a policy. Returns (mean_reward, mean_survival_steps).
    """
    np.random.seed(seed)
    rewards, steps = [], []

    for _ in range(n_episodes):
        obs = env.reset()
        total_r = 0.0
        step = 0
        done = False
        while not done:
            if use_expert:
                a = env.expert_action(obs)
            else:
                a = policy_fn(obs)
            obs, r, done = env.step(a)
            total_r += r
            step += 1
        rewards.append(total_r)
        steps.append(step)

    return np.mean(rewards), np.mean(steps)


def demonstrate_bc_vs_dagger():
    """Compare BC and DAgger on road following with varying wind."""
    print("=" * 60)
    print("Behavioral Cloning vs DAgger")
    print("=" * 60)

    # Train on clean env
    train_env = RoadFollowEnv(noise_std=0.02, wind=0.0)
    bc_policy, n_bc = train_bc(train_env, n_expert_episodes=30)
    dagger_policy, n_da = train_dagger(train_env, n_iterations=10, n_episodes_per_iter=5)

    print(f"\n  BC training data:     {n_bc} transitions")
    print(f"  DAgger training data: {n_da} transitions")
    print(f"  BC learned w:      {bc_policy.w:.3f}")
    print(f"  DAgger learned w:  {dagger_policy.w:.3f}")
    print(f"  Expert true gain:  ~2.0")

    # Evaluate on environments with increasing distribution shift (wind)
    print(f"\n  Evaluation (100 episodes, mean reward):")
    print(f"  {'Wind':>6} | {'Expert':>8} | {'BC':>8} | {'DAgger':>8}")
    print("  " + "-" * 40)

    for wind in [0.0, 0.05, 0.1, 0.2]:
        eval_env = RoadFollowEnv(noise_std=0.05, wind=wind)
        r_expert, _ = evaluate(None, eval_env, use_expert=True)
        r_bc, _ = evaluate(bc_policy.predict, eval_env)
        r_da, _ = evaluate(dagger_policy.predict, eval_env)
        print(f"  {wind:>6.2f} | {r_expert:>8.3f} | {r_bc:>8.3f} | {r_da:>8.3f}")

    print("\n  BC suffers from compounding errors when wind causes states")
    print("  outside the training distribution.")
    print("  DAgger explicitly trains on visited states, handling drift better.")


# ============================================================
# 4. Compounding Error Illustration
# ============================================================

def demonstrate_compounding_error():
    """
    Show that small per-step errors accumulate over long horizons.
    """
    print("\n" + "=" * 60)
    print("Compounding Error in Behavioral Cloning")
    print("=" * 60)

    env = RoadFollowEnv(noise_std=0.01, wind=0.0)

    # BC policy with a slight miscalibration (w=1.5 instead of 2.0)
    bc = LinearPolicy()
    bc.w = 1.5

    horizons = [10, 20, 30, 50]
    n_trials = 500

    print(f"\n  Mean absolute offset at each horizon (BC w=1.5 vs Expert w=2.0):")
    print(f"  {'Horizon':>8} | {'BC offset':>12} | {'Expert offset':>14}")
    print("  " + "-" * 42)

    for H in horizons:
        bc_offsets, expert_offsets = [], []
        np.random.seed(0)

        for _ in range(n_trials):
            env.offset = np.random.uniform(-0.2, 0.2)
            env.step_count = 0
            obs = env.observe()

            for _ in range(H):
                a = bc.predict(obs)
                obs, _, done = env.step(a)
                if done:
                    break
            bc_offsets.append(abs(env.offset))

        np.random.seed(0)
        for _ in range(n_trials):
            env.offset = np.random.uniform(-0.2, 0.2)
            env.step_count = 0
            obs = env.observe()

            for _ in range(H):
                a = env.expert_action(obs)
                obs, _, done = env.step(a)
                if done:
                    break
            expert_offsets.append(abs(env.offset))

        print(f"  {H:>8} | {np.mean(bc_offsets):>12.4f} | {np.mean(expert_offsets):>14.4f}")

    print("\n  BC error grows with horizon (T^2 bound in theory).")
    print("  DAgger reduces this by training on states the learner visits.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demonstrate_bc_vs_dagger()
    demonstrate_compounding_error()

    print("\n" + "=" * 60)
    print("Imitation Learning examples complete!")
    print("=" * 60)
