"""
Reward Shaping and Intrinsic Motivation — Example Code

Demonstrates:
  1. Potential-based reward shaping and policy invariance
  2. Curiosity-driven exploration (prediction error bonus)
  3. Count-based exploration bonus
  4. Comparison: shaped vs unshaped reward on sparse-reward grid

No external dependencies required.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# 1. Potential-Based Reward Shaping
# ============================================================

class MazeSparse:
    """
    5x5 maze — sparse reward (+10 only at goal, -0.01 step penalty).
    Used to show the benefit of potential-based shaping.
    """

    SIZE = 5
    GOAL = (4, 4)
    ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def reset(self):
        self.pos = np.array([0, 0])
        return tuple(self.pos)

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        np_pos = np.clip(self.pos + np.array([dr, dc]), 0, self.SIZE - 1)
        self.pos = np_pos
        s = tuple(self.pos)
        done = s == self.GOAL
        reward = 10.0 if done else -0.01
        return s, reward, done


def manhattan_potential(state, goal=(4, 4)):
    """
    Potential function: negative Manhattan distance to goal.
    phi(s) = -(|r - gr| + |c - gc|)
    Closer to goal => higher potential.
    """
    return -(abs(state[0] - goal[0]) + abs(state[1] - goal[1]))


def shaped_reward(s, ns, reward, gamma=0.99):
    """
    F(s, s') = gamma * phi(s') - phi(s)
    Shaping is policy-invariant: optimal policy under shaped R
    is also optimal under original R.
    """
    return reward + gamma * manhattan_potential(ns) - manhattan_potential(s)


def train_q(env, shaping=False, n_episodes=500, gamma=0.99, lr=0.1, epsilon=0.3):
    """Train tabular Q-learning with or without reward shaping."""
    Q = defaultdict(lambda: np.zeros(4))
    success_history = []

    for ep in range(n_episodes):
        state = env.reset()
        ep_success = False

        for _ in range(50):
            if np.random.random() < epsilon:
                action = np.random.randint(4)
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, done = env.step(action)

            if shaping:
                r = shaped_reward(state, next_state, reward, gamma)
            else:
                r = reward

            target = r if done else r + gamma * np.max(Q[next_state])
            Q[state][action] += lr * (target - Q[state][action])

            state = next_state
            if done:
                ep_success = True
                break

        success_history.append(float(ep_success))

    return success_history


def demonstrate_potential_shaping():
    """Show that potential-based shaping speeds up learning."""
    print("=" * 60)
    print("1. Potential-Based Reward Shaping")
    print("=" * 60)

    env = MazeSparse()
    np.random.seed(42)

    hist_no_shape = train_q(env, shaping=False, n_episodes=500)
    np.random.seed(42)
    hist_shaped = train_q(env, shaping=True, n_episodes=500)

    window = 100
    print(f"\n  Success rate (rolling {window}-ep window):")
    print(f"  {'Episodes':>12} | {'No Shaping':>12} | {'Shaped':>12}")
    print("  " + "-" * 44)
    for start in range(0, 500, window):
        ns = np.mean(hist_no_shape[start:start + window])
        sh = np.mean(hist_shaped[start:start + window])
        print(f"  {start+1:>5}-{start+window:<5}   | {ns:>10.1%} | {sh:>10.1%}")

    # Verify policy invariance: final optimal policy should be the same
    print("\n  Potential phi(s) values for each grid cell:")
    for r in range(5):
        row = "  "
        for c in range(5):
            row += f"{manhattan_potential((r,c)):4d} "
        print(row)
    print("  (Shapes gradient: agent pulled toward goal)")


# ============================================================
# 2. Curiosity-Driven Exploration (Prediction Error Bonus)
# ============================================================

class WorldModel:
    """
    Simple learned forward model: predicts next state from (state, action).
    Prediction error -> intrinsic reward (curiosity bonus).
    """

    def __init__(self, n_states, n_actions):
        # Prediction: next_state ~ categorical, learned via counts
        self.counts = np.ones((n_states, n_actions, n_states))  # Laplace smoothing

    def predict(self, state, action):
        probs = self.counts[state, action] / self.counts[state, action].sum()
        return int(np.argmax(probs))

    def update(self, state, action, next_state):
        self.counts[state, action, next_state] += 1

    def prediction_error(self, state, action, actual_next):
        """
        Uncertainty as prediction error.
        High error = model hasn't seen this transition much = intrinsic reward.
        """
        predicted = self.predict(state, action)
        return 0.0 if predicted == actual_next else 1.0


def demonstrate_curiosity():
    """Show curiosity bonus leads to better state coverage."""
    print("\n" + "=" * 60)
    print("2. Curiosity-Driven Exploration")
    print("=" * 60)

    n_states = 20
    n_actions = 4
    np.random.seed(7)

    model = WorldModel(n_states, n_actions)
    Q = np.zeros((n_states, n_actions))

    # Simple chain: action 1 moves right, 0 moves left, 2/3 stay
    def transition(s, a):
        if a == 1:
            return min(n_states - 1, s + 1)
        elif a == 0:
            return max(0, s - 1)
        return s

    # Track visits
    visits_no_curiosity = np.zeros(n_states)
    visits_curiosity = np.zeros(n_states)

    gamma = 0.99
    lr = 0.1
    epsilon = 0.1
    eta = 0.5   # curiosity bonus scale
    n_steps = 2000

    # Without curiosity
    state = 0
    for _ in range(n_steps):
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = int(np.argmax(Q[state]))
        ns = transition(state, action)
        visits_no_curiosity[state] += 1
        Q[state, action] += lr * (0 + gamma * np.max(Q[ns]) - Q[state, action])
        state = ns

    # With curiosity
    Q2 = np.zeros((n_states, n_actions))
    state = 0
    for _ in range(n_steps):
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = int(np.argmax(Q2[state]))
        ns = transition(state, action)
        visits_curiosity[state] += 1
        intrinsic = eta * model.prediction_error(state, action, ns)
        model.update(state, action, ns)
        total_r = 0 + intrinsic
        Q2[state, action] += lr * (total_r + gamma * np.max(Q2[ns]) - Q2[state, action])
        state = ns

    print("\n  State visit counts (2000 steps from state=0):")
    print(f"  {'State':>6} | {'No Curiosity':>14} | {'With Curiosity':>16}")
    print("  " + "-" * 44)
    for s in range(0, n_states, 2):
        print(f"  {s:>6} | {visits_no_curiosity[s]:>14.0f} | {visits_curiosity[s]:>16.0f}")

    n_visited_no = np.sum(visits_no_curiosity > 0)
    n_visited_cu = np.sum(visits_curiosity > 0)
    print(f"\n  States visited: no_curiosity={n_visited_no}, curiosity={n_visited_cu}")
    print("  => Curiosity bonus encourages exploration of novel states.")


# ============================================================
# 3. Count-Based Exploration
# ============================================================

def demonstrate_count_based():
    """Show exploration bonus = 1/sqrt(N(s,a)) pushes into unvisited states."""
    print("\n" + "=" * 60)
    print("3. Count-Based Exploration Bonus")
    print("=" * 60)

    n_states = 10
    n_actions = 2
    np.random.seed(3)

    def transition(s, a):
        return min(n_states - 1, s + 1) if a == 1 else max(0, s - 1)

    # Run with and without count bonus
    for use_bonus in [False, True]:
        Q = np.zeros((n_states, n_actions))
        N = np.zeros((n_states, n_actions))
        visits = np.zeros(n_states)
        state = 0

        for _ in range(1000):
            action = int(np.argmax(Q[state]))
            ns = transition(state, action)
            N[state, action] += 1
            visits[state] += 1

            bonus = 1.0 / np.sqrt(N[state, action]) if use_bonus else 0.0
            Q[state, action] += 0.1 * (bonus + 0.99 * np.max(Q[ns]) - Q[state, action])
            state = ns

        label = "With count bonus" if use_bonus else "No count bonus  "
        farthest = np.max(np.where(visits > 0)[0])
        print(f"  {label}: farthest state reached = {farthest}, "
              f"states visited = {int(np.sum(visits > 0))}")

    print("  => Count-based bonus actively explores unvisited (s,a) pairs.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demonstrate_potential_shaping()
    demonstrate_curiosity()
    demonstrate_count_based()

    print("\n" + "=" * 60)
    print("Reward Shaping examples complete!")
    print("=" * 60)
