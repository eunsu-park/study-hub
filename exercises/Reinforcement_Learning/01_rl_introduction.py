"""
Exercises for Lesson 01: Introduction to Reinforcement Learning
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Concept Check: Explain 3 main differences between supervised learning
    and reinforcement learning.

    Solution: We print the three key differences.
    """
    differences = [
        (
            "1. Feedback Type",
            "Supervised Learning: Receives immediate correct answer (label) for each input.\n"
            "   Reinforcement Learning: Receives delayed reward signal that may come many "
            "steps after the action was taken."
        ),
        (
            "2. Data Source",
            "Supervised Learning: Learns from a fixed, pre-collected labeled dataset.\n"
            "   Reinforcement Learning: Generates its own data through interaction with "
            "the environment; the data distribution depends on the agent's policy."
        ),
        (
            "3. Goal",
            "Supervised Learning: Minimize prediction error (e.g., classification or "
            "regression loss).\n"
            "   Reinforcement Learning: Maximize cumulative (discounted) reward over time "
            "through sequential decision making."
        ),
    ]

    for title, explanation in differences:
        print(f"{title}")
        print(f"   {explanation}")
        print()


def exercise_2():
    """
    Discount Factor Calculation: Calculate the discounted cumulative reward
    for reward sequence [1, 2, 3, 4, 5] with gamma=0.9.

    G = r0 + gamma*r1 + gamma^2*r2 + gamma^3*r3 + gamma^4*r4
    """
    rewards = [1, 2, 3, 4, 5]
    gamma = 0.9

    # Method 1: Direct summation
    G_direct = sum(gamma**k * r for k, r in enumerate(rewards))
    print(f"Rewards: {rewards}, gamma={gamma}")
    print(f"\nMethod 1 (direct summation):")
    terms = [f"{gamma}^{k} * {r} = {gamma**k * r:.4f}" for k, r in enumerate(rewards)]
    for term in terms:
        print(f"  {term}")
    print(f"  G = {G_direct:.4f}")

    # Method 2: Backward recursion (more efficient)
    G_recursive = 0
    for r in reversed(rewards):
        G_recursive = r + gamma * G_recursive
    print(f"\nMethod 2 (backward recursion):")
    print(f"  G = {G_recursive:.4f}")

    # Verify both methods agree
    assert abs(G_direct - G_recursive) < 1e-10
    print(f"\nBoth methods agree: G = {G_direct:.4f}")

    # Breakdown:
    # G = 1 + 0.9*2 + 0.81*3 + 0.729*4 + 0.6561*5
    # G = 1 + 1.8 + 2.43 + 2.916 + 3.2805 = 11.4265
    print(f"\nExpected: 1 + 1.8 + 2.43 + 2.916 + 3.2805 = 11.4265")


def exercise_3():
    """
    Exploration-Exploitation: With epsilon=0.2 in epsilon-greedy strategy,
    how many exploration actions are expected in 100 actions?

    Solution: Each action has probability epsilon=0.2 of being an exploration
    action, so expected number = n * epsilon = 100 * 0.2 = 20.
    """
    epsilon = 0.2
    n_actions = 100

    # Analytical answer
    expected_exploration = n_actions * epsilon
    print(f"epsilon = {epsilon}, total actions = {n_actions}")
    print(f"Expected exploration actions = {n_actions} * {epsilon} = {expected_exploration:.0f}")

    # Empirical verification through simulation
    np.random.seed(42)
    n_simulations = 10000
    exploration_counts = []

    for _ in range(n_simulations):
        # Simulate 100 actions with epsilon-greedy
        random_draws = np.random.random(n_actions)
        n_explore = np.sum(random_draws < epsilon)
        exploration_counts.append(n_explore)

    mean_explore = np.mean(exploration_counts)
    std_explore = np.std(exploration_counts)

    print(f"\nEmpirical verification ({n_simulations} simulations):")
    print(f"  Mean exploration actions: {mean_explore:.2f}")
    print(f"  Std deviation: {std_explore:.2f}")
    print(f"  Expected std (binomial): {np.sqrt(n_actions * epsilon * (1 - epsilon)):.2f}")


def exercise_4():
    """
    Reward Design: Design an appropriate reward function for a maze escape problem.

    Solution: We implement a simple maze environment with a well-designed
    reward function and demonstrate the agent interacting with it.
    """
    class MazeRewardFunction:
        """
        Reward function for a maze escape problem.

        Design principles:
        1. Goal reward: Large positive reward for reaching the exit
        2. Step penalty: Small negative reward per step to encourage efficiency
        3. Wall collision penalty: Negative reward for hitting walls
        4. Progress reward (optional): Small reward for moving closer to exit
        """

        def __init__(self, maze_size=5):
            self.maze_size = maze_size
            self.exit_pos = (maze_size - 1, maze_size - 1)

        def compute_reward(self, state, action, next_state, hit_wall):
            """
            Args:
                state: Current position (row, col)
                action: Action taken
                next_state: Resulting position (row, col)
                hit_wall: Whether agent hit a wall

            Returns:
                reward: Scalar reward value
            """
            # 1. Goal reward (+100): large enough to dominate the cumulative
            #    penalty from step costs, ensuring the agent always prefers
            #    reaching the goal over wandering
            if next_state == self.exit_pos:
                return 100.0

            # 2. Wall collision (-5): penalizes futile moves that waste steps
            #    and don't change the agent's position
            if hit_wall:
                return -5.0

            # 3. Step penalty (-1): discourages long paths; the agent must
            #    find the exit quickly to minimize accumulated penalty
            step_penalty = -1.0

            # 4. Progress reward: reward proportional to distance reduction
            #    toward the goal, providing a gradient signal in sparse-reward mazes
            old_dist = abs(state[0] - self.exit_pos[0]) + abs(state[1] - self.exit_pos[1])
            new_dist = abs(next_state[0] - self.exit_pos[0]) + abs(next_state[1] - self.exit_pos[1])
            progress = (old_dist - new_dist) * 0.5  # Small bonus for getting closer

            return step_penalty + progress

    # Demonstrate the reward function
    reward_fn = MazeRewardFunction(maze_size=5)

    print("Maze Reward Function Design:")
    print("=" * 50)
    print("  Reaching exit:      +100.0")
    print("  Hitting wall:        -5.0")
    print("  Each step:           -1.0")
    print("  Moving closer:       +0.5")
    print("  Moving farther:      -0.5")
    print()

    # Example scenarios
    scenarios = [
        {"state": (0, 0), "next_state": (0, 1), "hit_wall": False,
         "desc": "Moving right from start (closer to goal)"},
        {"state": (0, 0), "next_state": (0, 0), "hit_wall": True,
         "desc": "Hitting left wall at start"},
        {"state": (2, 2), "next_state": (2, 1), "hit_wall": False,
         "desc": "Moving left (farther from goal)"},
        {"state": (4, 3), "next_state": (4, 4), "hit_wall": False,
         "desc": "Reaching the exit!"},
    ]

    for scenario in scenarios:
        r = reward_fn.compute_reward(
            scenario["state"], None, scenario["next_state"], scenario["hit_wall"]
        )
        print(f"  {scenario['desc']}")
        print(f"    {scenario['state']} -> {scenario['next_state']}: reward = {r:.1f}")
        print()


if __name__ == "__main__":
    print("=== Exercise 1: SL vs RL Differences ===")
    exercise_1()

    print("\n=== Exercise 2: Discount Factor Calculation ===")
    exercise_2()

    print("\n=== Exercise 3: Exploration-Exploitation ===")
    exercise_3()

    print("\n=== Exercise 4: Reward Design ===")
    exercise_4()

    print("\nAll exercises completed!")
