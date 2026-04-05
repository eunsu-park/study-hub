"""
Nano RL — REINFORCE Policy Gradient on a GridWorld

A self-contained reinforcement learning agent that learns to navigate
a 5x5 gridworld using the REINFORCE algorithm with a value baseline.
No external dependencies beyond NumPy.

Learning Objectives:
1. Understand Markov Decision Processes (states, actions, rewards)
2. Implement a tabular softmax policy: pi(a|s) = softmax(theta[s,:])
3. Apply the REINFORCE gradient estimator: nabla J = E[sum G_t * nabla log pi]
4. Use a value function baseline to reduce variance
5. Visualize learned policy and value function
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RLConfig:
    """Hyperparameters for the REINFORCE experiment."""
    grid_size: int = 5
    max_steps: int = 50
    gamma: float = 0.99
    lr_policy: float = 0.01
    lr_value: float = 0.05
    n_episodes: int = 3000
    walls: List[Tuple[int, int]] = field(
        default_factory=lambda: [(1, 1), (2, 3), (3, 1), (1, 3)]
    )
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (4, 4)


# Action constants
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GridWorld:
    """5x5 deterministic gridworld with walls."""

    def __init__(self, cfg: RLConfig) -> None:
        self.size = cfg.grid_size
        self.start = cfg.start
        self.goal = cfg.goal
        self.walls = set(cfg.walls)
        self.max_steps = cfg.max_steps
        self.state: Tuple[int, int] = self.start
        self.steps: int = 0

    def _to_index(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    def reset(self) -> int:
        self.state = self.start
        self.steps = 0
        return self._to_index(self.state)

    def step(self, action: int) -> Tuple[int, float, bool]:
        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.state[0] + dr, self.state[1] + dc
        # Boundary check
        if 0 <= nr < self.size and 0 <= nc < self.size:
            if (nr, nc) not in self.walls:
                self.state = (nr, nc)
        self.steps += 1
        if self.state == self.goal:
            return self._to_index(self.state), 1.0, True
        if self.steps >= self.max_steps:
            return self._to_index(self.state), -0.01, True
        return self._to_index(self.state), -0.01, False

# ---------------------------------------------------------------------------
# Policy (tabular softmax)
# ---------------------------------------------------------------------------

class PolicyNetwork:
    """Tabular softmax policy: pi(a|s) = softmax(theta[s, :])."""

    def __init__(self, n_states: int, n_actions: int = 4) -> None:
        self.theta: np.ndarray = np.zeros((n_states, n_actions))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        x = logits - logits.max()
        e = np.exp(x)
        return e / e.sum()

    def get_probs(self, state: int) -> np.ndarray:
        return self._softmax(self.theta[state])

    def sample_action(self, state: int) -> int:
        probs = self.get_probs(state)
        return int(np.random.choice(len(probs), p=probs))

    def log_prob(self, state: int, action: int) -> float:
        probs = self.get_probs(state)
        return float(np.log(probs[action] + 1e-10))

# ---------------------------------------------------------------------------
# Value function (tabular)
# ---------------------------------------------------------------------------

class ValueNetwork:
    """Tabular state-value function."""

    def __init__(self, n_states: int) -> None:
        self.V: np.ndarray = np.zeros(n_states)

    def predict(self, state: int) -> float:
        return float(self.V[state])

    def update(self, state: int, target: float, lr: float) -> None:
        self.V[state] += lr * (target - self.V[state])

# ---------------------------------------------------------------------------
# REINFORCE Agent
# ---------------------------------------------------------------------------

class REINFORCEAgent:
    """REINFORCE with baseline for the tabular gridworld."""

    @staticmethod
    def collect_episode(
        env: GridWorld, policy: PolicyNetwork
    ) -> Tuple[List[int], List[int], List[float]]:
        states: List[int] = []
        actions: List[int] = []
        rewards: List[float] = []
        s = env.reset()
        done = False
        while not done:
            a = policy.sample_action(s)
            states.append(s)
            actions.append(a)
            s_next, r, done = env.step(a)
            rewards.append(r)
            s = s_next
        return states, actions, rewards

    @staticmethod
    def compute_returns(rewards: List[float], gamma: float) -> np.ndarray:
        T = len(rewards)
        G = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + gamma * running
            G[t] = running
        return G

    @staticmethod
    def update(
        states: List[int],
        actions: List[int],
        returns: np.ndarray,
        policy: PolicyNetwork,
        value: ValueNetwork,
        cfg: RLConfig,
    ) -> None:
        for t in range(len(states)):
            s, a, G_t = states[t], actions[t], returns[t]
            advantage = G_t - value.predict(s)
            # Policy gradient (tabular softmax)
            probs = policy.get_probs(s)
            # d/d_theta[s,a'] log pi(a|s) = 1(a'=a) - pi(a'|s)
            grad_log = -probs.copy()
            grad_log[a] += 1.0  # now: (1{a'=a} - pi(a'|s))
            policy.theta[s] += cfg.lr_policy * advantage * grad_log
            # Value update
            value.update(s, G_t, cfg.lr_value)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: RLConfig) -> Tuple[PolicyNetwork, ValueNetwork, List[float], List[int]]:
    n_states = cfg.grid_size * cfg.grid_size
    env = GridWorld(cfg)
    policy = PolicyNetwork(n_states)
    value = ValueNetwork(n_states)
    agent = REINFORCEAgent()

    ep_rewards: List[float] = []
    ep_lengths: List[int] = []

    for ep in range(1, cfg.n_episodes + 1):
        states, actions, rewards = agent.collect_episode(env, policy)
        returns = agent.compute_returns(rewards, cfg.gamma)
        agent.update(states, actions, returns, policy, value, cfg)

        ep_rewards.append(sum(rewards))
        ep_lengths.append(len(rewards))

        if ep % 500 == 0:
            recent_r = np.mean(ep_rewards[-100:])
            recent_l = np.mean(ep_lengths[-100:])
            print(f"Episode {ep:>5d} | mean reward (last 100): {recent_r:+.3f} | mean length: {recent_l:.1f}")

    return policy, value, ep_rewards, ep_lengths

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(
    policy: PolicyNetwork,
    value: ValueNetwork,
    ep_rewards: List[float],
    cfg: RLConfig,
    save_path: str,
) -> None:
    import matplotlib.pyplot as plt

    arrow_map = {UP: (0, 0.3), DOWN: (0, -0.3), LEFT: (-0.3, 0), RIGHT: (0.3, 0)}
    wall_set = set(cfg.walls)
    gs = cfg.grid_size

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Reward curve (smoothed) ---
    ax = axes[0]
    window = 100
    smoothed = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
    ax.plot(smoothed, color="steelblue", linewidth=1.0)
    ax.set_title("Episode Reward (rolling avg 100)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Policy arrows ---
    ax = axes[1]
    ax.set_xlim(-0.5, gs - 0.5)
    ax.set_ylim(-0.5, gs - 0.5)
    ax.set_aspect("equal")
    ax.set_title("Learned Policy")
    ax.set_xticks(range(gs))
    ax.set_yticks(range(gs))
    ax.invert_yaxis()

    for r in range(gs):
        for c in range(gs):
            if (r, c) in wall_set:
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="0.25")
                ax.add_patch(rect)
            elif (r, c) == cfg.start:
                ax.text(c, r, "S", ha="center", va="center", fontsize=14, fontweight="bold", color="green")
            elif (r, c) == cfg.goal:
                ax.text(c, r, "G", ha="center", va="center", fontsize=14, fontweight="bold", color="red")
            else:
                s = r * gs + c
                best = int(np.argmax(policy.theta[s]))
                dx, dy = arrow_map[best]
                ax.annotate("", xy=(c + dx, r - dy), xytext=(c, r),
                            arrowprops=dict(arrowstyle="->", lw=1.5, color="navy"))
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Value heatmap ---
    ax = axes[2]
    V_grid = np.zeros((gs, gs))
    for r in range(gs):
        for c in range(gs):
            V_grid[r, c] = value.V[r * gs + c]
    for wr, wc in wall_set:
        V_grid[wr, wc] = np.nan

    im = ax.imshow(V_grid, cmap="YlOrRd", origin="upper")
    ax.set_title("State-Value Function V(s)")
    ax.set_xticks(range(gs))
    ax.set_yticks(range(gs))
    for r in range(gs):
        for c in range(gs):
            if (r, c) in wall_set:
                ax.text(c, r, "W", ha="center", va="center", fontsize=10, color="white")
            else:
                ax.text(c, r, f"{V_grid[r, c]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Nano RL — REINFORCE on 5×5 GridWorld", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    save_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = RLConfig()
    policy, value, ep_rewards, ep_lengths = train(cfg)

    # Expected: mean reward > 0.7 (last 100 episodes) after 3000 episodes
    final_mean = np.mean(ep_rewards[-100:])
    print(f"\nFinal mean reward (last 100): {final_mean:+.3f}")

    png_path = os.path.join(save_dir, "nano_rl.png")
    visualize(policy, value, ep_rewards, cfg, png_path)
