"""
Dynamic Programming Implementation
- Policy Evaluation
- Policy Improvement
- Policy Iteration
- Value Iteration
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict


class GridWorld:
    """Simple grid world environment"""

    def __init__(self, size=4):
        self.size = size
        self.actions = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.actions)

    def get_states(self):
        """Return all states"""
        return [(i, j) for i in range(self.size) for j in range(self.size)]

    def is_terminal(self, state):
        """Check if the state is terminal"""
        return state == (0, 0) or state == (self.size-1, self.size-1)

    def get_transitions(self, state, action):
        """Return transition probabilities: [(prob, next_state, reward, done)]"""
        if self.is_terminal(state):
            return [(1.0, state, 0, True)]

        deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        delta = deltas[action]

        # Handle grid boundaries
        new_row = max(0, min(self.size-1, state[0] + delta[0]))
        new_col = max(0, min(self.size-1, state[1] + delta[1]))
        next_state = (new_row, new_col)

        # Reward: -1 for each move
        reward = -1
        done = self.is_terminal(next_state)

        return [(1.0, next_state, reward, done)]


def create_uniform_policy(grid):
    """Create a uniform random policy"""
    policy = {}
    for s in grid.get_states():
        policy[s] = {a: 1.0/len(grid.actions) for a in grid.actions}
    return policy


def policy_evaluation(grid, policy: Dict, gamma: float = 0.9, theta: float = 1e-6):
    """
    Policy Evaluation: Compute the value function for a given policy

    Args:
        grid: GridWorld environment
        policy: Policy {state: {action: probability}}
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        V: State value function {state: value}
    """
    # Initialize value function
    V = {s: 0.0 for s in grid.get_states()}

    iteration = 0
    while True:
        delta = 0  # Track maximum change
        iteration += 1

        # Update for all states
        for s in grid.get_states():
            if grid.is_terminal(s):
                continue

            v = V[s]  # Store previous value
            new_v = 0

            # Apply Bellman expectation equation
            for a in grid.actions:
                action_prob = policy[s].get(a, 0)

                for prob, next_s, reward, done in grid.get_transitions(s, a):
                    if done:
                        new_v += action_prob * prob * reward
                    else:
                        new_v += action_prob * prob * (reward + gamma * V[next_s])

            V[s] = new_v
            delta = max(delta, abs(v - new_v))

        # Convergence check
        if delta < theta:
            print(f"Policy evaluation converged: {iteration} iterations, delta={delta:.8f}")
            break

    return V


def policy_improvement(grid, V: Dict, gamma: float = 0.9):
    """
    Policy Improvement: Generate a greedy policy based on V

    Args:
        grid: GridWorld environment
        V: Current value function
        gamma: Discount factor

    Returns:
        new_policy: Improved policy
        policy_stable: True if the policy did not change
    """
    new_policy = {}
    policy_stable = True

    for s in grid.get_states():
        if grid.is_terminal(s):
            new_policy[s] = {a: 1.0/len(grid.actions) for a in grid.actions}
            continue

        # Compute Q-value for each action
        q_values = {}
        for a in grid.actions:
            q = 0
            for prob, next_s, reward, done in grid.get_transitions(s, a):
                if done:
                    q += prob * reward
                else:
                    q += prob * (reward + gamma * V[next_s])
            q_values[a] = q

        # Find the best action
        best_action = max(q_values, key=q_values.get)
        best_q = q_values[best_action]

        # Find tied actions (considering numerical error)
        best_actions = [a for a, q in q_values.items()
                        if abs(q - best_q) < 1e-8]

        # Create deterministic policy
        new_policy[s] = {a: 0.0 for a in grid.actions}
        for a in best_actions:
            new_policy[s][a] = 1.0 / len(best_actions)

    return new_policy, policy_stable


def policy_iteration(grid, gamma: float = 0.9, theta: float = 1e-6):
    """
    Policy Iteration Algorithm

    Returns:
        V: Optimal value function
        policy: Optimal policy
    """
    # Initialize with a uniform random policy
    policy = create_uniform_policy(grid)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n=== Policy Iteration {iteration} ===")

        # 1. Policy Evaluation
        V = policy_evaluation(grid, policy, gamma, theta)

        # 2. Policy Improvement
        old_policy = policy.copy()
        policy, _ = policy_improvement(grid, V, gamma)

        # 3. Policy stability check
        policy_stable = True
        for s in grid.get_states():
            if grid.is_terminal(s):
                continue

            old_best = max(old_policy[s], key=old_policy[s].get)
            new_best = max(policy[s], key=policy[s].get)

            if old_best != new_best:
                policy_stable = False
                break

        if policy_stable:
            print(f"\nPolicy iteration converged! (total {iteration} iterations)")
            break

    return V, policy


def value_iteration(grid, gamma: float = 0.9, theta: float = 1e-6):
    """
    Value Iteration Algorithm

    Returns:
        V: Optimal value function
        policy: Optimal policy
    """
    # Initialize value function
    V = {s: 0.0 for s in grid.get_states()}

    iteration = 0
    while True:
        delta = 0
        iteration += 1

        for s in grid.get_states():
            if grid.is_terminal(s):
                continue

            v = V[s]

            # Bellman optimality equation: max over actions
            q_values = []
            for a in grid.actions:
                q = 0
                for prob, next_s, reward, done in grid.get_transitions(s, a):
                    if done:
                        q += prob * reward
                    else:
                        q += prob * (reward + gamma * V[next_s])
                q_values.append(q)

            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: delta = {delta:.8f}")

        if delta < theta:
            print(f"\nValue iteration converged: {iteration} iterations")
            break

    # Extract optimal policy
    policy = {}
    for s in grid.get_states():
        if grid.is_terminal(s):
            policy[s] = {a: 1.0/len(grid.actions) for a in grid.actions}
            continue

        q_values = {}
        for a in grid.actions:
            q = 0
            for prob, next_s, reward, done in grid.get_transitions(s, a):
                if done:
                    q += prob * reward
                else:
                    q += prob * (reward + gamma * V[next_s])
            q_values[a] = q

        best_action = max(q_values, key=q_values.get)
        policy[s] = {a: 0.0 for a in grid.actions}
        policy[s][best_action] = 1.0

    return V, policy


def print_value_function(grid, V):
    """Print the value function"""
    print("\nValue Function:")
    for i in range(grid.size):
        row = [f"{V[(i,j)]:7.2f}" for j in range(grid.size)]
        print(" ".join(row))


def print_policy(grid, policy):
    """Print the policy (as arrows)"""
    print("\nOptimal Policy:")
    arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

    for i in range(grid.size):
        row = []
        for j in range(grid.size):
            s = (i, j)
            if grid.is_terminal(s):
                row.append('  *  ')
            else:
                best_a = max(policy[s], key=policy[s].get)
                row.append(f'  {arrows[best_a]}  ')
        print(" ".join(row))


def visualize_value_function(grid, V, title="Value Function"):
    """Visualize the value function"""
    value_grid = np.zeros((grid.size, grid.size))
    for i in range(grid.size):
        for j in range(grid.size):
            value_grid[i, j] = V[(i, j)]

    plt.figure(figsize=(8, 6))
    plt.imshow(value_grid, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(title)

    # Display numbers
    for i in range(grid.size):
        for j in range(grid.size):
            plt.text(j, i, f'{value_grid[i, j]:.1f}',
                    ha='center', va='center', color='black', fontsize=12)

    plt.xticks(range(grid.size))
    plt.yticks(range(grid.size))
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('value_function.png', dpi=150)
    print(f"Value function visualization saved: value_function.png")


def compare_algorithms():
    """Compare DP algorithms"""
    print("=" * 60)
    print("Dynamic Programming Algorithm Comparison")
    print("=" * 60)

    grid = GridWorld(size=4)
    gamma = 0.9

    # 1. Policy Evaluation (uniform random policy)
    print("\n[1] Policy Evaluation - Uniform Random Policy")
    print("-" * 60)
    uniform_policy = create_uniform_policy(grid)
    V_uniform = policy_evaluation(grid, uniform_policy, gamma)
    print_value_function(grid, V_uniform)

    # 2. Policy Iteration
    print("\n[2] Policy Iteration")
    print("-" * 60)
    V_pi, policy_pi = policy_iteration(grid, gamma)
    print_value_function(grid, V_pi)
    print_policy(grid, policy_pi)

    # 3. Value Iteration
    print("\n[3] Value Iteration")
    print("-" * 60)
    V_vi, policy_vi = value_iteration(grid, gamma)
    print_value_function(grid, V_vi)
    print_policy(grid, policy_vi)

    # 4. Result Comparison
    print("\n[4] Result Comparison")
    print("-" * 60)
    print("Value function difference between Policy Iteration and Value Iteration:")
    max_diff = 0
    for s in grid.get_states():
        diff = abs(V_pi[s] - V_vi[s])
        max_diff = max(max_diff, diff)
    print(f"Maximum difference: {max_diff:.10f}")

    # Visualization
    visualize_value_function(grid, V_pi, "Policy Iteration - Value Function")

    return V_pi, policy_pi, V_vi, policy_vi


def frozen_lake_example():
    """Apply DP to the Frozen Lake environment"""
    import gymnasium as gym

    print("\n" + "=" * 60)
    print("Frozen Lake Example")
    print("=" * 60)

    # Create environment (non-slippery version)
    env = gym.make('FrozenLake-v1', is_slippery=False)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    gamma = 0.99
    theta = 1e-8

    # P[s][a] = [(prob, next_state, reward, done), ...]
    P = env.unwrapped.P

    # Value Iteration
    V = np.zeros(n_states)
    iteration = 0

    print("\nStarting value iteration...")
    while True:
        delta = 0
        iteration += 1

        for s in range(n_states):
            v = V[s]

            # Compute value of each action
            q_values = []
            for a in range(n_actions):
                q = sum(prob * (reward + gamma * V[next_s] * (not done))
                       for prob, next_s, reward, done in P[s][a])
                q_values.append(q)

            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            print(f"Converged: {iteration} iterations")
            break

    # Extract optimal policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = []
        for a in range(n_actions):
            q = sum(prob * (reward + gamma * V[next_s] * (not done))
                   for prob, next_s, reward, done in P[s][a])
            q_values.append(q)
        policy[s] = np.argmax(q_values)

    # Visualize results
    action_names = ['←', '↓', '→', '↑']
    print("\nOptimal Policy (4x4 grid):")
    print("S: Start, H: Hole, F: Frozen, G: Goal")
    for i in range(4):
        row = ""
        for j in range(4):
            s = i * 4 + j
            if s == 0:
                row += "  S  "
            elif s in [5, 7, 11, 12]:  # Holes
                row += "  H  "
            elif s == 15:  # Goal
                row += "  G  "
            else:
                row += f"  {action_names[policy[s]]}  "
        print(row)

    print("\nValue Function:")
    print(V.reshape(4, 4).round(3))

    # Test the policy
    print("\nTesting the policy...")
    success = 0
    n_tests = 100

    for _ in range(n_tests):
        state, _ = env.reset()
        done = False

        while not done:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if reward > 0:
            success += 1

    print(f"Success rate: {success}/{n_tests} = {success/n_tests*100:.1f}%")

    env.close()
    return V, policy


if __name__ == "__main__":
    # Grid world algorithm comparison
    V_pi, policy_pi, V_vi, policy_vi = compare_algorithms()

    # Frozen Lake example
    try:
        V_fl, policy_fl = frozen_lake_example()
    except Exception as e:
        print(f"\nFailed to run Frozen Lake example: {e}")
        print("Make sure the gymnasium package is installed: pip install gymnasium")

    print("\n" + "=" * 60)
    print("Dynamic programming examples complete!")
    print("=" * 60)
