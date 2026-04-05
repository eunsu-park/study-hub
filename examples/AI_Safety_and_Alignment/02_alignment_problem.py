# 02_alignment_problem.py — Reward hacking demonstration and Goodhart's law
#
# Run: python 02_alignment_problem.py

"""
Demonstrates reward hacking and Goodhart's law in a simple gridworld
environment. An agent optimizes a proxy reward that diverges from
the true objective, illustrating core alignment failures.
"""

import random
from dataclasses import dataclass


@dataclass
class GridState:
    x: int
    y: int


class GridWorld:
    """Simple gridworld with a true objective and a proxy reward."""

    def __init__(self, size: int = 5):
        self.size = size
        self.goal = GridState(size - 1, size - 1)
        self.hazards = {(2, 2), (3, 1), (1, 3)}
        self.reward_hack_tile = GridState(0, size - 1)

    def true_reward(self, state: GridState) -> float:
        """True objective: reach goal while avoiding hazards."""
        if (state.x, state.y) in self.hazards:
            return -10.0
        if state.x == self.goal.x and state.y == self.goal.y:
            return 100.0
        dist = abs(state.x - self.goal.x) + abs(state.y - self.goal.y)
        return -dist * 0.5

    def proxy_reward(self, state: GridState) -> float:
        """Proxy: maximize score metric (hackable)."""
        if state.x == self.reward_hack_tile.x and \
           state.y == self.reward_hack_tile.y:
            return 80.0  # High proxy reward but not the true goal
        dist = abs(state.x - self.goal.x) + abs(state.y - self.goal.y)
        return -dist * 0.3  # Weaker penalty makes proxy easier to game

    def step(self, state: GridState, action: str) -> GridState:
        dx, dy = {"up": (0, 1), "down": (0, -1),
                  "left": (-1, 0), "right": (1, 0)}[action]
        nx = max(0, min(self.size - 1, state.x + dx))
        ny = max(0, min(self.size - 1, state.y + dy))
        return GridState(nx, ny)

    def actions(self) -> list[str]:
        return ["up", "down", "left", "right"]


class GreedyAgent:
    """Agent that greedily maximizes a given reward function."""

    def __init__(self, env: GridWorld, reward_fn):
        self.env = env
        self.reward_fn = reward_fn

    def choose_action(self, state: GridState) -> str:
        best_action = None
        best_reward = float("-inf")
        for action in self.env.actions():
            next_state = self.env.step(state, action)
            r = self.reward_fn(next_state)
            if r > best_reward:
                best_reward = r
                best_action = action
        return best_action

    def run_episode(self, max_steps: int = 20) -> dict:
        state = GridState(0, 0)
        trajectory = [state]
        total_true = 0.0
        total_proxy = 0.0

        for _ in range(max_steps):
            action = self.choose_action(state)
            state = self.env.step(state, action)
            trajectory.append(state)
            total_true += self.env.true_reward(state)
            total_proxy += self.env.proxy_reward(state)

            if state.x == self.env.goal.x and state.y == self.env.goal.y:
                break

        return {
            "trajectory": [(s.x, s.y) for s in trajectory],
            "total_true_reward": round(total_true, 2),
            "total_proxy_reward": round(total_proxy, 2),
            "reached_goal": trajectory[-1].x == self.env.goal.x and
                            trajectory[-1].y == self.env.goal.y,
        }


def demonstrate_goodharts_law():
    """Show how optimizing a proxy metric leads to poor true outcomes."""
    print("=== Goodhart's Law Demonstration ===\n")
    print("'When a measure becomes a target,")
    print(" it ceases to be a good measure.'\n")

    metrics = {
        "code_lines": {"proxy": 500, "true_quality": 30},
        "test_count": {"proxy": 200, "true_quality": 40},
        "response_time_ms": {"proxy": 5, "true_quality": 45},
    }

    print("Scenario: Optimizing proxy metrics for software quality\n")
    for metric, values in metrics.items():
        gaming_penalty = values["proxy"] * 0.1
        effective_quality = values["true_quality"] - gaming_penalty
        print(f"  Metric: {metric}")
        print(f"    Proxy score: {values['proxy']}")
        print(f"    True quality: {values['true_quality']}")
        print(f"    Quality after gaming: {effective_quality:.1f}")
        print(f"    Degradation: {gaming_penalty:.1f}\n")


def specification_gaming_examples():
    """Catalog of known specification gaming patterns."""
    print("=== Specification Gaming Catalog ===\n")
    examples = [
        ("Boat Racing AI", "Learned to spin in circles collecting "
         "boost pads instead of finishing the race"),
        ("Block Stacking", "Robot learned to flip the table so the "
         "block appears 'higher' in sensor readings"),
        ("List Sorting", "Deleted list items instead of sorting, "
         "achieving 'sorted' on empty list"),
        ("Content Filter", "Learned to rephrase harmful content to "
         "bypass keyword filters while preserving meaning"),
        ("Engagement Metric", "Optimized for clicks by generating "
         "outrage content, harming user wellbeing"),
    ]
    for name, description in examples:
        print(f"  [{name}]")
        print(f"    {description}\n")


if __name__ == "__main__":
    env = GridWorld(size=5)

    # Agent using true reward
    print("=== Reward Hacking in Gridworld ===\n")
    print("--- Agent with TRUE reward function ---")
    true_agent = GreedyAgent(env, env.true_reward)
    result_true = true_agent.run_episode()
    print(f"  Path: {result_true['trajectory']}")
    print(f"  True reward: {result_true['total_true_reward']}")
    print(f"  Reached goal: {result_true['reached_goal']}\n")

    # Agent using proxy reward (susceptible to hacking)
    print("--- Agent with PROXY reward function ---")
    proxy_agent = GreedyAgent(env, env.proxy_reward)
    result_proxy = proxy_agent.run_episode()
    print(f"  Path: {result_proxy['trajectory']}")
    print(f"  True reward: {result_proxy['total_true_reward']}")
    print(f"  Proxy reward: {result_proxy['total_proxy_reward']}")
    print(f"  Reached goal: {result_proxy['reached_goal']}\n")

    # Alignment gap analysis
    gap = result_true["total_true_reward"] - result_proxy["total_true_reward"]
    print(f"Alignment gap (true reward difference): {gap:.2f}")
    if gap > 0:
        print("-> Proxy-optimizing agent performs WORSE on true objective!\n")

    demonstrate_goodharts_law()
    specification_gaming_examples()
