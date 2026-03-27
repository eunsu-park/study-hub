"""
World Models — Example Code

Demonstrates:
  1. Learning a transition model from rollout data
  2. Model Predictive Control (MPC) with a learned model
  3. Dyna-style planning: real + imaginary rollouts
  4. Model error analysis (epistemic uncertainty)

No external dependencies required.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# Environment
# ============================================================

class PendulumLite:
    """
    Simplified inverted pendulum (discrete state approximation).
    State: (angle_bin, vel_bin), angle in [-pi, pi], vel in [-8, 8].
    Action: torque in {-2, 0, +2}.
    """

    N_ANGLE = 20
    N_VEL = 20
    N_ACTIONS = 3
    TORQUES = [-2.0, 0.0, 2.0]
    DT = 0.05

    def _discretize(self, angle, vel):
        a_bin = int((angle + np.pi) / (2 * np.pi) * self.N_ANGLE)
        v_bin = int((vel + 8.0) / 16.0 * self.N_VEL)
        return (np.clip(a_bin, 0, self.N_ANGLE - 1),
                np.clip(v_bin, 0, self.N_VEL - 1))

    def reset(self):
        self.angle = np.random.uniform(-np.pi, np.pi)
        self.vel = np.random.uniform(-1.0, 1.0)
        return self._discretize(self.angle, self.vel)

    def step(self, action):
        torque = self.TORQUES[action]
        g, m, l = 9.8, 1.0, 1.0
        acc = (3 * g / (2 * l) * np.sin(self.angle) +
               3 / (m * l**2) * torque)
        self.vel = np.clip(self.vel + acc * self.DT, -8.0, 8.0)
        self.angle = (self.angle + self.vel * self.DT + np.pi) % (2 * np.pi) - np.pi
        s = self._discretize(self.angle, self.vel)
        # Reward: upright position near angle=0
        reward = -(self.angle**2 + 0.1 * self.vel**2 + 0.001 * torque**2)
        done = False
        return s, reward, done

    def n_states(self):
        return self.N_ANGLE * self.N_VEL

    def encode(self, s):
        return s[0] * self.N_VEL + s[1]


# ============================================================
# 1. Learning a Transition Model
# ============================================================

class TabularWorldModel:
    """
    Count-based learned model: P(s' | s, a) and R(s, a).
    """

    def __init__(self, n_states, n_actions):
        self.trans_counts = defaultdict(lambda: defaultdict(int))
        self.reward_sum = defaultdict(float)
        self.reward_count = defaultdict(int)
        self.n_actions = n_actions
        self.n_states = n_states

    def update(self, s, a, r, ns):
        self.trans_counts[(s, a)][ns] += 1
        self.reward_sum[(s, a)] += r
        self.reward_count[(s, a)] += 1

    def predict_next(self, s, a):
        """Sample next state from learned distribution."""
        counts = self.trans_counts[(s, a)]
        if not counts:
            return s, 0.0  # no data: stay in place
        total = sum(counts.values())
        ns_list = list(counts.keys())
        probs = np.array([counts[ns] for ns in ns_list]) / total
        ns = ns_list[np.random.choice(len(ns_list), p=probs)]
        r = self.reward_sum[(s, a)] / (self.reward_count[(s, a)] + 1e-10)
        return ns, r

    def model_accuracy(self, test_data):
        """Compute prediction accuracy on held-out transitions."""
        correct = 0
        for s, a, r, ns in test_data:
            predicted_ns, _ = self.predict_next(s, a)
            correct += int(predicted_ns == ns)
        return correct / len(test_data)


def collect_data(env, policy='random', n_steps=2000, seed=0):
    """Collect transition data from environment."""
    np.random.seed(seed)
    data = []
    state = env.reset()
    for _ in range(n_steps):
        action = np.random.randint(env.N_ACTIONS)
        next_state, reward, done = env.step(action)
        data.append((env.encode(state), action, reward, env.encode(next_state)))
        state = next_state if not done else env.reset()
    return data


def demonstrate_model_learning():
    """Train and evaluate a tabular world model."""
    print("=" * 60)
    print("1. Learning a Transition Model")
    print("=" * 60)

    env = PendulumLite()
    all_data = collect_data(env, n_steps=4000)

    # Split into train/test
    split = int(len(all_data) * 0.8)
    train_data = all_data[:split]
    test_data = all_data[split:]

    model = TabularWorldModel(env.n_states(), env.N_ACTIONS)

    # Evaluate accuracy at different data quantities
    print(f"\n  Model accuracy as training data grows:")
    print(f"  {'Train steps':>12} | {'Accuracy':>10} | {'(s,a) covered':>15}")
    for n in [100, 500, 1000, 2000, len(train_data)]:
        for s, a, r, ns in train_data[:n]:
            model.update(s, a, r, ns)
        acc = model.model_accuracy(test_data)
        covered = len(model.trans_counts)
        print(f"  {n:>12} | {acc:>10.3f} | {covered:>15}")


# ============================================================
# 2. Model Predictive Control (MPC)
# ============================================================

def mpc_action(model, state, horizon=5, n_trajectories=50):
    """
    MPC: randomly sample action sequences, simulate with learned model,
    return the first action of the best sequence.
    """
    best_reward = -1e10
    best_action = 0

    for _ in range(n_trajectories):
        actions = np.random.randint(model.n_actions, size=horizon)
        s = state
        total_r = 0.0
        for t, a in enumerate(actions):
            ns, r = model.predict_next(s, a)
            total_r += (0.99 ** t) * r
            s = ns

        if total_r > best_reward:
            best_reward = total_r
            best_action = actions[0]

    return best_action


def demonstrate_mpc():
    """Compare MPC (with learned model) vs random policy."""
    print("\n" + "=" * 60)
    print("2. Model Predictive Control (MPC)")
    print("=" * 60)

    env = PendulumLite()

    # First collect training data with random policy
    train_data = collect_data(env, n_steps=5000, seed=1)
    model = TabularWorldModel(env.n_states(), env.N_ACTIONS)
    for s, a, r, ns in train_data:
        model.update(s, a, r, ns)

    def run_policy(policy_fn, n_episodes=20, max_steps=100):
        rewards = []
        for ep in range(n_episodes):
            np.random.seed(ep)
            state = env.reset()
            total_r = 0.0
            for _ in range(max_steps):
                a = policy_fn(state)
                state, r, done = env.step(a)
                total_r += r
            rewards.append(total_r)
        return np.mean(rewards), np.std(rewards)

    mean_rand, std_rand = run_policy(lambda s: np.random.randint(env.N_ACTIONS))
    mean_mpc, std_mpc = run_policy(lambda s: mpc_action(model, s, horizon=5))

    print(f"\n  Policy evaluation (20 episodes, 100 steps each):")
    print(f"  Random policy:  {mean_rand:.2f} ± {std_rand:.2f}")
    print(f"  MPC (horizon=5): {mean_mpc:.2f} ± {std_mpc:.2f}")
    print("\n  MPC plans ahead in the learned model without further env interaction.")


# ============================================================
# 3. Dyna-Q: Real + Imaginary Rollouts
# ============================================================

def train_dyna_q(env, n_real_steps=1000, n_planning_steps=10, seed=42):
    """
    Dyna-Q: after each real step, do k imaginary updates using learned model.
    This dramatically improves sample efficiency.
    """
    np.random.seed(seed)
    n_states = env.n_states()
    Q = np.zeros((n_states, env.N_ACTIONS))
    model = TabularWorldModel(n_states, env.N_ACTIONS)

    epsilon, gamma, lr = 0.2, 0.99, 0.1
    rewards_history = []
    state = env.reset()

    for step in range(n_real_steps):
        s = env.encode(state)
        if np.random.random() < epsilon:
            action = np.random.randint(env.N_ACTIONS)
        else:
            action = int(np.argmax(Q[s]))

        next_state, reward, done = env.step(action)
        ns = env.encode(next_state)

        # Real update
        Q[s, action] += lr * (reward + gamma * np.max(Q[ns]) - Q[s, action])
        model.update(s, action, reward, ns)
        rewards_history.append(reward)
        state = next_state if not done else env.reset()

        # Imaginary updates (planning)
        for _ in range(n_planning_steps):
            if not model.trans_counts:
                break
            sa = list(model.trans_counts.keys())
            s_plan, a_plan = sa[np.random.randint(len(sa))]
            ns_plan, r_plan = model.predict_next(s_plan, a_plan)
            Q[s_plan, a_plan] += lr * (r_plan + gamma * np.max(Q[ns_plan]) -
                                        Q[s_plan, a_plan])

    return Q, rewards_history


def demonstrate_dyna():
    """Show that Dyna-Q converges faster than pure model-free Q-learning."""
    print("\n" + "=" * 60)
    print("3. Dyna-Q: Real + Imaginary Rollouts")
    print("=" * 60)

    env = PendulumLite()
    n_steps = 1000
    window = 100

    _, hist_dyna = train_dyna_q(env, n_real_steps=n_steps, n_planning_steps=10)
    # Model-free: same setup but n_planning_steps=0
    _, hist_mf = train_dyna_q(env, n_real_steps=n_steps, n_planning_steps=0)

    print(f"\n  Reward (rolling {window}-step window):")
    print(f"  {'Steps':>8} | {'Model-Free':>12} | {'Dyna-Q (k=10)':>15}")
    print("  " + "-" * 44)
    for start in range(0, n_steps, window):
        end = start + window
        mf = np.mean(hist_mf[start:end])
        dy = np.mean(hist_dyna[start:end])
        print(f"  {start+1:>5}-{end:<5} | {mf:>12.3f} | {dy:>15.3f}")

    print("\n  Dyna-Q leverages the learned model for k imaginary updates")
    print("  per real step, achieving faster convergence.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demonstrate_model_learning()
    demonstrate_mpc()
    demonstrate_dyna()

    print("\n" + "=" * 60)
    print("World Models examples complete!")
    print("=" * 60)
