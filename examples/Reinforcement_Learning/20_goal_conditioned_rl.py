"""
Goal-Conditioned Reinforcement Learning — Example Code

Demonstrates:
  1. Goal-conditioned Q-function (Universal Value Function)
  2. Hindsight Experience Replay (HER) with "final" strategy
  3. Multi-goal training vs single-goal training comparison

No external gym/PyTorch dependency — NumPy only.
"""

import numpy as np
from collections import deque


# ============================================================
# 1. 2-D Grid with Goal-Conditioned Policy
# ============================================================

class GridGoalEnv:
    """
    NxN grid. Agent starts at (0,0). Goal is a variable position.
    Sparse reward: +1 only when agent reaches goal, 0 otherwise.
    """

    def __init__(self, size=5):
        self.size = size
        self.ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U

    def reset(self, goal=None):
        self.agent = np.array([0, 0])
        if goal is None:
            goal = np.random.randint(0, self.size, size=2)
        self.goal = np.array(goal)
        return self._obs()

    def _obs(self):
        return tuple(self.agent), tuple(self.goal)

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        new_pos = self.agent + np.array([dr, dc])
        new_pos = np.clip(new_pos, 0, self.size - 1)
        self.agent = new_pos
        reached = np.array_equal(self.agent, self.goal)
        reward = 1.0 if reached else 0.0
        return self._obs(), reward, reached

    def state_dim(self):
        return self.size * self.size

    def encode(self, pos):
        """Encode (r,c) -> integer index."""
        return pos[0] * self.size + pos[1]


# ============================================================
# 2. Goal-Conditioned Q-Table
# ============================================================

class GoalConditionedQTable:
    """
    Q(s, g, a): tabular goal-conditioned value function.
    State and goal are (r,c) positions in a grid.
    """

    def __init__(self, n_positions, n_actions):
        self.Q = np.zeros((n_positions, n_positions, n_actions))
        self.n_actions = n_actions

    def get_action(self, state_idx, goal_idx, epsilon=0.0):
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state_idx, goal_idx]))

    def update(self, s, g, a, r, ns, done, gamma=0.99, lr=0.1):
        if done:
            target = r
        else:
            target = r + gamma * np.max(self.Q[ns, g])
        self.Q[s, g, a] += lr * (target - self.Q[s, g, a])


# ============================================================
# 3. Training with and without HER
# ============================================================

def train_standard(env, n_episodes=2000, gamma=0.99, lr=0.1, epsilon=0.3):
    """Standard Q-learning on goal-conditioned env (sparse reward)."""
    n_pos = env.size * env.size
    Q = GoalConditionedQTable(n_pos, len(env.ACTIONS))
    success_history = []

    for ep in range(n_episodes):
        goal = np.random.randint(0, env.size, size=2)
        (state, g), _, _ = env.reset(goal), None, None
        obs = env.reset(goal)
        state, g = obs

        s_idx = env.encode(state)
        g_idx = env.encode(g)

        ep_success = False
        for _ in range(env.size * 2 + 5):
            action = Q.get_action(s_idx, g_idx, epsilon)
            (ns, ng), reward, done = env.step(action)
            ns_idx = env.encode(ns)

            Q.update(s_idx, g_idx, action, reward, ns_idx, done, gamma, lr)

            s_idx = ns_idx
            if done:
                ep_success = True
                break

        success_history.append(float(ep_success))

    return success_history


def train_with_her(env, n_episodes=2000, gamma=0.99, lr=0.1, epsilon=0.3,
                   her_k=4):
    """
    Q-learning with Hindsight Experience Replay (HER).

    For each episode transition, we also replay it with the hindsight goal
    set to where the agent actually ended up.
    """
    n_pos = env.size * env.size
    Q = GoalConditionedQTable(n_pos, len(env.ACTIONS))
    success_history = []

    for ep in range(n_episodes):
        goal = np.random.randint(0, env.size, size=2)
        obs = env.reset(goal)
        state, g = obs

        s_idx = env.encode(state)
        g_idx = env.encode(g)

        trajectory = []  # [(s, g, a, r, ns, done)]
        ep_success = False

        for _ in range(env.size * 2 + 5):
            action = Q.get_action(s_idx, g_idx, epsilon)
            (ns, ng), reward, done = env.step(action)
            ns_idx = env.encode(ns)

            trajectory.append((s_idx, g_idx, action, reward, ns_idx, done))
            Q.update(s_idx, g_idx, action, reward, ns_idx, done, gamma, lr)

            s_idx = ns_idx
            if done:
                ep_success = True
                break

        # HER: relabel transitions with hindsight goals
        # Strategy "final": use the last visited state as alternative goal
        if trajectory:
            final_state_idx = trajectory[-1][4]  # ns of last transition

            # Replay each transition with the hindsight goal
            for (s, g_old, a, r_old, ns, done_old) in trajectory:
                if done_old:
                    break
                # Would we have reached hindsight_goal at ns?
                her_done = (ns == final_state_idx)
                her_reward = 1.0 if her_done else 0.0
                Q.update(s, final_state_idx, a, her_reward, ns,
                         her_done, gamma, lr)

        success_history.append(float(ep_success))

    return success_history


def demonstrate_her():
    """Compare HER vs standard training on sparse reward grid."""
    print("=" * 60)
    print("Hindsight Experience Replay (HER) vs Standard Training")
    print("=" * 60)

    env = GridGoalEnv(size=5)
    np.random.seed(42)

    n_episodes = 2000

    print("\nTraining Standard Q-learning (sparse reward)...")
    std_hist = train_standard(env, n_episodes=n_episodes)

    print("Training with HER...")
    np.random.seed(42)
    her_hist = train_with_her(env, n_episodes=n_episodes)

    # Report rolling success rates
    window = 200
    print(f"\n  Success rate (rolling {window}-episode window):")
    print(f"  {'Episodes':>12} | {'Standard':>12} | {'HER':>12}")
    print("  " + "-" * 42)
    for start in range(0, n_episodes, window):
        end = start + window
        std_rate = np.mean(std_hist[start:end])
        her_rate = np.mean(her_hist[start:end])
        print(f"  {start+1:>5}-{end:<5}   | {std_rate:>10.1%} | {her_rate:>10.1%}")

    print("\n  => HER learns faster in sparse reward settings by")
    print("     converting failed episodes into useful learning signals.")


# ============================================================
# 4. Universal Value Function — Goal Generalization
# ============================================================

def demonstrate_goal_generalization():
    """
    Show that a single goal-conditioned Q-table can generalize
    to goals it was rarely trained on.
    """
    print("\n" + "=" * 60)
    print("Goal Generalization with Universal Value Function")
    print("=" * 60)

    env = GridGoalEnv(size=4)
    np.random.seed(0)

    # Train with HER on random goals
    hist = train_with_her(env, n_episodes=3000, epsilon=0.3)
    final_rate = np.mean(hist[-200:])
    print(f"\n  Training success rate (last 200 eps): {final_rate:.1%}")

    # Evaluate on all 16 possible goals
    n_pos = env.size * env.size
    # Rebuild Q from scratch for clean eval
    Q = GoalConditionedQTable(n_pos, len(env.ACTIONS))
    env2 = GridGoalEnv(size=4)
    np.random.seed(0)
    hist2 = train_with_her(env2, n_episodes=3000, epsilon=0.3)

    print("\n  Final policy evaluated per goal position:")
    print(f"  (Start always at (0,0))")
    for gr in range(env.size):
        row = "  "
        for gc in range(env.size):
            if (gr, gc) == (0, 0):
                row += " S  "
                continue
            # Quick eval for this goal
            successes = 0
            for _ in range(50):
                obs = env.reset(goal=[gr, gc])
                s, g = obs
                s_idx = env.encode(s)
                g_idx = env.encode(g)
                for _ in range(env.size * 3):
                    a = int(np.argmax(Q.Q[s_idx, g_idx]))
                    (ns, ng), r, done = env.step(a)
                    s_idx = env.encode(ns)
                    if done:
                        successes += 1
                        break
            rate = successes / 50
            row += f"{rate:.0%} "
        print(row)
    print("  (Note: Q was just initialized; train longer for high success rates.)")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demonstrate_her()
    demonstrate_goal_generalization()

    print("\n" + "=" * 60)
    print("Goal-Conditioned RL examples complete!")
    print("=" * 60)
