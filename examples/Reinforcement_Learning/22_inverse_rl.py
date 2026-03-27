"""
Inverse Reinforcement Learning — Example Code

Demonstrates:
  1. Behavior Cloning from expert demonstrations
  2. Maximum Entropy IRL (tabular, small grid)
  3. Reward learning from pairwise preferences

No external dependencies required.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# Environment and Expert
# ============================================================

class GridEnvIRL:
    """4x4 grid. Terminal: (3,3). Step reward: unknown (to be recovered)."""

    SIZE = 4
    ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U
    NAMES = ['right', 'left', 'down', 'up']

    def reset(self, start=None):
        if start is None:
            start = (np.random.randint(self.SIZE), np.random.randint(self.SIZE))
        self.pos = list(start)
        return tuple(self.pos)

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        self.pos[0] = max(0, min(self.SIZE - 1, self.pos[0] + dr))
        self.pos[1] = max(0, min(self.SIZE - 1, self.pos[1] + dc))
        s = tuple(self.pos)
        done = s == (self.SIZE - 1, self.SIZE - 1)
        return s, done

    def all_states(self):
        return [(r, c) for r in range(self.SIZE) for c in range(self.SIZE)]

    def encode(self, s):
        return s[0] * self.SIZE + s[1]


# True reward (known only to the expert)
def true_reward(state):
    """Reward = negative Manhattan distance to (3,3) + terminal bonus."""
    if state == (3, 3):
        return 10.0
    return -(abs(state[0] - 3) + abs(state[1] - 3)) * 0.5


def collect_expert_demos(env, n_demos=30, max_steps=20, seed=42):
    """
    Expert policy: always move toward (3,3) — greedy on true reward.
    Returns list of trajectories: [(s0, a0, s1, a1, ...)]
    """
    np.random.seed(seed)
    demos = []
    for _ in range(n_demos):
        start = (np.random.randint(env.SIZE), np.random.randint(env.SIZE))
        state = env.reset(start)
        traj = [state]
        for _ in range(max_steps):
            # Greedy toward goal
            best_a, best_r = 0, -1e9
            for a, (dr, dc) in enumerate(env.ACTIONS):
                ns = (max(0, min(env.SIZE - 1, state[0] + dr)),
                      max(0, min(env.SIZE - 1, state[1] + dc)))
                r = true_reward(ns)
                if r > best_r:
                    best_r, best_a = r, a
            next_state, done = env.step(best_a)
            traj.append(best_a)
            traj.append(next_state)
            state = next_state
            if done:
                break
        demos.append(traj)
    return demos


# ============================================================
# 1. Behavior Cloning
# ============================================================

def train_behavior_cloning(demos, env):
    """BC: learn π_BC(a|s) = most frequent action in that state."""
    counts = defaultdict(lambda: np.zeros(4))
    for traj in demos:
        # traj = [s0, a0, s1, a1, ...]
        for i in range(0, len(traj) - 2, 2):
            s, a = traj[i], traj[i + 1]
            counts[s][a] += 1

    policy = {}
    for s in env.all_states():
        if counts[s].sum() > 0:
            policy[s] = int(np.argmax(counts[s]))
        else:
            policy[s] = 1  # default: go right

    return policy


def evaluate_policy(policy, env, n_eval=200, max_steps=20):
    """Evaluate success rate of a deterministic policy."""
    successes = 0
    for _ in range(n_eval):
        start = (np.random.randint(env.SIZE), np.random.randint(env.SIZE))
        state = env.reset(start)
        for _ in range(max_steps):
            a = policy[state]
            state, done = env.step(a)
            if done:
                successes += 1
                break
    return successes / n_eval


def demonstrate_bc():
    """Show behavior cloning success rate on expert demos."""
    print("=" * 60)
    print("1. Behavior Cloning from Expert Demonstrations")
    print("=" * 60)

    env = GridEnvIRL()
    np.random.seed(42)
    demos = collect_expert_demos(env, n_demos=50)
    bc_policy = train_behavior_cloning(demos, env)

    print(f"\n  Learned BC Policy (best action per state):")
    arrows = {0: '→', 1: '←', 2: '↓', 3: '↑'}
    for r in range(env.SIZE):
        row = "  "
        for c in range(env.SIZE):
            s = (r, c)
            if s == (env.SIZE - 1, env.SIZE - 1):
                row += " G "
            else:
                row += f" {arrows[bc_policy[s]]} "
        print(row)

    success = evaluate_policy(bc_policy, env)
    print(f"\n  BC success rate: {success:.1%}")
    print("  (Expert demos alone give a reasonable policy.)")


# ============================================================
# 2. Maximum Entropy IRL (tabular)
# ============================================================

def compute_state_visitation(policy, env, gamma=0.99, n_steps=20):
    """
    Compute (un-discounted) state visitation frequency under policy.
    Simulates from all start states.
    """
    visit = defaultdict(float)
    n_starts = env.SIZE * env.SIZE
    for r in range(env.SIZE):
        for c in range(env.SIZE):
            state = env.reset((r, c))
            for t in range(n_steps):
                visit[state] += (gamma ** t) / n_starts
                a = policy[state]
                state, done = env.step(a)
                if done:
                    break
    return visit


def maxent_irl(demos, env, n_iters=100, lr=0.05, gamma=0.99):
    """
    Maximum Entropy IRL (Ziebart et al. 2008) — simplified tabular version.

    Gradient: d L / d theta_s = mu_expert(s) - mu_agent(s)
    where mu is the expected state visitation frequency.
    """
    # Compute expert state visitation from demos
    expert_visit = defaultdict(float)
    total_transitions = 0
    for traj in demos:
        for i in range(0, len(traj), 2):
            if i < len(traj):
                expert_visit[traj[i]] += 1.0
                total_transitions += 1
    for s in expert_visit:
        expert_visit[s] /= total_transitions

    # Initialize reward weights (one per state)
    theta = np.zeros(env.SIZE * env.SIZE)  # reward parameter per state

    def reward_from_theta(state):
        return theta[env.encode(state)]

    losses = []
    for it in range(n_iters):
        # Derive greedy policy from current reward estimate
        policy = {}
        for s in env.all_states():
            best_a, best_r = 0, -1e9
            for a, (dr, dc) in enumerate(env.ACTIONS):
                ns = (max(0, min(env.SIZE - 1, s[0] + dr)),
                      max(0, min(env.SIZE - 1, s[1] + dc)))
                r = reward_from_theta(ns)
                if r > best_r:
                    best_r, best_a = r, a
            policy[s] = best_a

        # Compute agent state visitation
        agent_visit = compute_state_visitation(policy, env, gamma)
        total_agent = sum(agent_visit.values()) + 1e-10
        for s in agent_visit:
            agent_visit[s] /= total_agent

        # Gradient: mu_expert - mu_agent
        for s in env.all_states():
            idx = env.encode(s)
            grad = expert_visit.get(s, 0.0) - agent_visit.get(s, 0.0)
            theta[idx] += lr * grad

        # Loss (log-likelihood proxy)
        ll = sum(expert_visit.get(s, 0.0) * reward_from_theta(s)
                 for s in env.all_states())
        losses.append(ll)

    return theta, policy, losses


def demonstrate_maxent_irl():
    """Recover reward function using MaxEnt IRL."""
    print("\n" + "=" * 60)
    print("2. Maximum Entropy IRL — Reward Recovery")
    print("=" * 60)

    env = GridEnvIRL()
    np.random.seed(42)
    demos = collect_expert_demos(env, n_demos=100)
    theta, recovered_policy, losses = maxent_irl(demos, env, n_iters=80)

    print("\n  Recovered reward (per state):")
    for r in range(env.SIZE):
        row = "  "
        for c in range(env.SIZE):
            row += f"{theta[env.encode((r,c))]:6.2f} "
        print(row)

    print("\n  True reward (per state):")
    for r in range(env.SIZE):
        row = "  "
        for c in range(env.SIZE):
            row += f"{true_reward((r,c)):6.2f} "
        print(row)

    # Correlation between recovered and true reward
    recovered = np.array([theta[env.encode(s)] for s in env.all_states()])
    true_r = np.array([true_reward(s) for s in env.all_states()])
    corr = np.corrcoef(recovered, true_r)[0, 1]
    print(f"\n  Correlation (recovered vs true): {corr:.3f}")
    print(f"  Training loss trend: {losses[0]:.4f} -> {losses[-1]:.4f}")

    success = evaluate_policy(recovered_policy, env)
    print(f"  Recovered policy success rate: {success:.1%}")


# ============================================================
# 3. Reward Learning from Pairwise Preferences
# ============================================================

def demonstrate_preference_reward_learning():
    """
    Learn a reward function from binary pairwise comparisons.
    Simplified version of RLHF preference modeling.
    """
    print("\n" + "=" * 60)
    print("3. Reward Learning from Pairwise Preferences")
    print("=" * 60)

    # Generate trajectory pairs with preference labels
    # Each trajectory is represented by feature counts
    # Features: [steps_taken, reached_goal, collisions]
    np.random.seed(5)
    n_pairs = 200

    # True reward: -0.1 * steps + 5.0 * goal - 2.0 * collisions
    true_weights = np.array([-0.1, 5.0, -2.0])

    def trajectory_features():
        steps = np.random.randint(5, 30)
        goal = np.random.choice([0, 1], p=[0.4, 0.6])
        collisions = np.random.randint(0, 5)
        return np.array([steps, goal, collisions], dtype=float)

    # Collect pairs with preference from true reward
    pairs = []
    for _ in range(n_pairs):
        phi_a = trajectory_features()
        phi_b = trajectory_features()
        r_a = np.dot(true_weights, phi_a)
        r_b = np.dot(true_weights, phi_b)
        # Human prefers the higher-reward trajectory (noisy)
        noise = np.random.randn() * 0.5
        preferred = 0 if (r_a + noise) > r_b else 1  # 0=prefer a, 1=prefer b
        pairs.append((phi_a, phi_b, preferred))

    # Learn reward weights via logistic regression (preference model)
    weights = np.zeros(3)
    lr = 0.01

    for epoch in range(200):
        grad = np.zeros(3)
        for phi_a, phi_b, pref in pairs:
            diff = phi_a - phi_b  # phi(a) - phi(b)
            r_diff = np.dot(weights, diff)
            # P(prefer a) = sigmoid(r_a - r_b)
            prob_a = 1 / (1 + np.exp(-r_diff))
            if pref == 0:   # human preferred a
                grad += (1 - prob_a) * diff
            else:            # human preferred b
                grad += -prob_a * diff
        weights += lr * grad / n_pairs

    print(f"\n  True reward weights:      {true_weights}")
    print(f"  Learned reward weights:   {weights.round(3)}")

    # Normalize for comparison
    scale = np.linalg.norm(true_weights) / (np.linalg.norm(weights) + 1e-10)
    scaled = weights * scale
    print(f"  Learned (scaled):         {scaled.round(3)}")
    print(f"\n  Features: [steps_taken, reached_goal, collisions]")
    print(f"  Sign check — steps:{np.sign(weights[0]):.0f}(exp:-1), "
          f"goal:{np.sign(weights[1]):.0f}(exp:+1), "
          f"collisions:{np.sign(weights[2]):.0f}(exp:-1)")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demonstrate_bc()
    demonstrate_maxent_irl()
    demonstrate_preference_reward_learning()

    print("\n" + "=" * 60)
    print("Inverse RL examples complete!")
    print("=" * 60)
