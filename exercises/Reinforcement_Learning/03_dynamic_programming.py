"""
Exercises for Lesson 03: Dynamic Programming
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Policy Evaluation: Manually calculate the value function for a uniform
    random policy on a 2x2 grid.

    Grid:
    [T][ ]     T = Terminal (top-left), G = Terminal (bottom-right)
    [ ][G]

    Actions: up, down, left, right (equal probability 0.25 each)
    Reward: -1 per step until terminal
    gamma = 1.0 (undiscounted, episodic)
    """
    print("2x2 Grid World with uniform random policy:")
    print("  [T][ ]   T = terminal (0,0)")
    print("  [ ][G]   G = terminal (1,1)")
    print("  Actions: up, down, left, right (each p=0.25)")
    print("  Reward: -1 per step, gamma = 1.0")
    print()

    # States: (0,0)=Terminal, (0,1), (1,0), (1,1)=Terminal
    # Non-terminal states: (0,1) and (1,0)
    # V(T) = V(G) = 0

    # For state (0,1):
    # up: stay (0,1), r=-1 -> 0.25 * (-1 + V(0,1))
    # down: go to (1,1)=G, r=-1 -> 0.25 * (-1 + 0) = -0.25
    # left: go to (0,0)=T, r=-1 -> 0.25 * (-1 + 0) = -0.25
    # right: stay (0,1), r=-1 -> 0.25 * (-1 + V(0,1))
    # V(0,1) = 0.25*(-1 + V(0,1)) + 0.25*(-1) + 0.25*(-1) + 0.25*(-1 + V(0,1))
    # V(0,1) = 0.5*(-1 + V(0,1)) + 0.5*(-1)
    # V(0,1) = -0.5 + 0.5*V(0,1) - 0.5
    # V(0,1) = -1 + 0.5*V(0,1)
    # 0.5*V(0,1) = -1
    # V(0,1) = -2

    # For state (1,0):
    # up: go to (0,0)=T, r=-1 -> 0.25 * (-1 + 0) = -0.25
    # down: stay (1,0), r=-1 -> 0.25 * (-1 + V(1,0))
    # left: stay (1,0), r=-1 -> 0.25 * (-1 + V(1,0))
    # right: go to (1,1)=G, r=-1 -> 0.25 * (-1 + 0) = -0.25
    # V(1,0) = 0.25*(-1) + 0.5*(-1 + V(1,0)) + 0.25*(-1)
    # V(1,0) = -1 + 0.5*V(1,0)
    # V(1,0) = -2

    print("Analytical solution (Bellman equations):")
    print()
    print("For state (0,1):")
    print("  V(0,1) = 0.25*(-1+V(0,1)) + 0.25*(-1+0) + 0.25*(-1+0) + 0.25*(-1+V(0,1))")
    print("  V(0,1) = -1 + 0.5*V(0,1)")
    print("  0.5*V(0,1) = -1")
    print("  V(0,1) = -2")
    print()
    print("For state (1,0):")
    print("  V(1,0) = 0.25*(-1+0) + 0.25*(-1+V(1,0)) + 0.25*(-1+V(1,0)) + 0.25*(-1+0)")
    print("  V(1,0) = -1 + 0.5*V(1,0)")
    print("  V(1,0) = -2")
    print()

    # Verify with iterative policy evaluation
    print("Verification with iterative policy evaluation:")
    V = {'(0,0)': 0.0, '(0,1)': 0.0, '(1,0)': 0.0, '(1,1)': 0.0}
    gamma = 1.0
    theta = 1e-10

    # Transitions for non-terminal states with wall bouncing
    transitions = {
        '(0,1)': {
            'up': ('(0,1)', -1),    # wall bounce
            'down': ('(1,1)', -1),   # to terminal G
            'left': ('(0,0)', -1),   # to terminal T
            'right': ('(0,1)', -1),  # wall bounce
        },
        '(1,0)': {
            'up': ('(0,0)', -1),     # to terminal T
            'down': ('(1,0)', -1),   # wall bounce
            'left': ('(1,0)', -1),   # wall bounce
            'right': ('(1,1)', -1),  # to terminal G
        },
    }

    iteration = 0
    while True:
        delta = 0
        iteration += 1
        for s in ['(0,1)', '(1,0)']:
            v = V[s]
            new_v = 0
            for action in ['up', 'down', 'left', 'right']:
                next_s, reward = transitions[s][action]
                is_terminal = next_s in ['(0,0)', '(1,1)']
                if is_terminal:
                    new_v += 0.25 * reward
                else:
                    new_v += 0.25 * (reward + gamma * V[next_s])
            V[s] = new_v
            delta = max(delta, abs(v - new_v))

        if delta < theta:
            break

    print(f"  Converged in {iteration} iterations")
    print(f"  V(0,0) = {V['(0,0)']:6.2f} (terminal)")
    print(f"  V(0,1) = {V['(0,1)']:6.2f}")
    print(f"  V(1,0) = {V['(1,0)']:6.2f}")
    print(f"  V(1,1) = {V['(1,1)']:6.2f} (terminal)")


def exercise_2():
    """
    Policy Iteration: Explain why policy iteration always terminates in finite steps.

    Solution: Theoretical explanation with a small numerical demonstration.
    """
    print("Why Policy Iteration Always Terminates in Finite Steps:")
    print("=" * 60)
    print()
    print("1. FINITE POLICY SPACE:")
    print("   For a finite MDP with |S| states and |A| actions,")
    print("   there are at most |A|^|S| deterministic policies.")
    print("   This is a finite number.")
    print()
    print("2. MONOTONIC IMPROVEMENT:")
    print("   The Policy Improvement Theorem guarantees that each")
    print("   new greedy policy pi' is strictly better than or")
    print("   equal to the current policy pi:")
    print("   V^{pi'}(s) >= V^{pi}(s) for all states s.")
    print()
    print("3. NO CYCLING:")
    print("   Since each improvement step either improves the value")
    print("   function or leaves it unchanged, and values are bounded,")
    print("   the algorithm cannot revisit a previous policy.")
    print()
    print("4. TERMINATION:")
    print("   The algorithm terminates when pi' = pi (no improvement),")
    print("   which means the policy is already optimal.")
    print("   Combined with monotonic improvement and finite policies,")
    print("   this must happen within at most |A|^|S| iterations.")
    print()

    # Demonstration with a tiny MDP
    print("Demonstration: 3-state MDP")
    print("-" * 40)

    # Simple 3-state MDP
    # States: 0, 1, 2 (state 2 is terminal)
    # Actions: left (0), right (1)
    n_states = 3
    n_actions = 2
    gamma = 0.9

    # Transition probabilities P[s][a] = [(prob, next_state, reward)]
    P = {
        0: {0: [(1.0, 0, -1)], 1: [(1.0, 1, 0)]},
        1: {0: [(1.0, 0, 0)], 1: [(1.0, 2, 10)]},
        2: {0: [(1.0, 2, 0)], 1: [(1.0, 2, 0)]},
    }

    max_policies = n_actions ** (n_states - 1)  # Exclude terminal
    print(f"  Max possible policies: {n_actions}^{n_states-1} = {max_policies}")

    # Policy iteration
    policy = np.zeros(n_states, dtype=int)  # Start with all-left
    iteration = 0

    while True:
        iteration += 1
        # Policy Evaluation
        V = np.zeros(n_states)
        for _ in range(1000):
            delta = 0
            for s in range(n_states - 1):  # Skip terminal
                v = V[s]
                a = policy[s]
                new_v = sum(p * (r + gamma * V[ns]) for p, ns, r in P[s][a])
                V[s] = new_v
                delta = max(delta, abs(v - new_v))
            if delta < 1e-10:
                break

        # Policy Improvement
        old_policy = policy.copy()
        for s in range(n_states - 1):
            q_values = []
            for a in range(n_actions):
                q = sum(p * (r + gamma * V[ns]) for p, ns, r in P[s][a])
                q_values.append(q)
            policy[s] = np.argmax(q_values)

        actions = ['left', 'right']
        print(f"  Iteration {iteration}: policy=[{actions[policy[0]]}, {actions[policy[1]]}], "
              f"V=[{V[0]:.2f}, {V[1]:.2f}, {V[2]:.2f}]")

        if np.array_equal(policy, old_policy):
            print(f"\n  Converged in {iteration} iterations (out of max {max_policies})!")
            break


def exercise_3():
    """
    Value Iteration: Why does value iteration converge in 1 iteration when gamma=0?

    Solution: Mathematical explanation and numerical verification.
    """
    print("Why Value Iteration Converges in 1 Iteration When gamma = 0:")
    print("=" * 60)
    print()
    print("The value iteration update is:")
    print("  V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) [R + gamma * V_k(s')]")
    print()
    print("When gamma = 0:")
    print("  V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) [R + 0 * V_k(s')]")
    print("             = max_a sum_{s'} P(s'|s,a) * R")
    print("             = max_a E[R | s, a]")
    print()
    print("Key insight: When gamma=0, V_{k+1}(s) depends ONLY on immediate")
    print("rewards, NOT on V_k(s'). So the update is independent of the")
    print("previous estimate, and the correct answer is obtained in a")
    print("single sweep regardless of initialization.")
    print()

    # Numerical verification
    print("Numerical verification on a 4-state MDP:")
    n_states = 4
    n_actions = 2
    gamma = 0.0

    # Random MDP
    np.random.seed(42)
    # P[s][a] = (next_state, reward) deterministic for simplicity
    rewards = np.random.randn(n_states, n_actions)

    print(f"\n  Immediate rewards R(s,a):")
    for s in range(n_states):
        print(f"    State {s}: left={rewards[s,0]:.3f}, right={rewards[s,1]:.3f}")

    # Value iteration with gamma=0
    V = np.zeros(n_states)  # Initialize to 0
    print(f"\n  Initial V: {V}")

    # Single iteration
    V_new = np.max(rewards, axis=1)  # max_a R(s,a) since gamma=0
    print(f"  After 1 iteration: V = {V_new}")

    # Second iteration (should be identical)
    V_check = np.max(rewards, axis=1)
    print(f"  After 2 iterations: V = {V_check}")
    print(f"  Change: {np.max(np.abs(V_new - V_check)):.1e}")
    print(f"\n  Confirmed: converged in exactly 1 iteration!")

    # Compare with gamma=0.9
    print(f"\nComparison with gamma=0.9:")
    gamma_compare = 0.9
    V_cmp = np.zeros(n_states)
    # Simple MDP: each action leads to a random next state
    transitions = np.random.randint(0, n_states, (n_states, n_actions))

    for iteration in range(100):
        V_old = V_cmp.copy()
        for s in range(n_states):
            q_values = []
            for a in range(n_actions):
                q = rewards[s, a] + gamma_compare * V_cmp[transitions[s, a]]
                q_values.append(q)
            V_cmp[s] = max(q_values)
        if np.max(np.abs(V_cmp - V_old)) < 1e-10:
            print(f"  gamma=0.9 converged in {iteration + 1} iterations")
            break


def exercise_4():
    """
    Asynchronous DP: Explain situations where asynchronous DP can be faster
    than synchronous DP.

    Solution: Theoretical explanation with numerical comparison.
    """
    print("When Asynchronous DP is Faster Than Synchronous DP:")
    print("=" * 60)
    print()
    print("1. STATES WITH UNEQUAL IMPORTANCE:")
    print("   Some states are visited more frequently or are closer to")
    print("   important regions. Async DP can prioritize these states,")
    print("   updating them more often while ignoring irrelevant states.")
    print()
    print("2. PRIORITIZED SWEEPING:")
    print("   Update states whose values changed the most (largest")
    print("   Bellman error). This propagates value changes quickly")
    print("   through the most affected parts of the state space.")
    print()
    print("3. REAL-TIME DP:")
    print("   When an agent is acting in the environment, async DP")
    print("   can focus updates on states the agent actually visits,")
    print("   avoiding computation on unreachable states.")
    print()
    print("4. LARGE STATE SPACES WITH STRUCTURE:")
    print("   In a large maze where the goal is in one corner,")
    print("   synchronous DP wastes computation updating states far")
    print("   from the goal. Async DP with backward-focused updates")
    print("   can converge much faster.")
    print()

    # Demonstration: Compare sync vs async on a corridor MDP
    print("Demonstration: 1D Corridor (20 states)")
    print("-" * 50)

    n_states = 20
    gamma = 0.99
    theta = 1e-6

    # Corridor: state 0 is start, state 19 is goal (+1 reward)
    # Actions: left (0), right (1)

    def get_next_state(s, a):
        if s == n_states - 1:
            return s, 0.0, True
        if a == 0:  # left
            return max(0, s - 1), -0.01, False
        else:  # right
            return min(n_states - 1, s + 1), -0.01, False

    # Synchronous value iteration
    V_sync = np.zeros(n_states)
    sync_updates = 0
    while True:
        delta = 0
        for s in range(n_states - 1):
            v = V_sync[s]
            q_values = []
            for a in range(2):
                ns, r, done = get_next_state(s, a)
                q = r + (0 if done else gamma * V_sync[ns])
                q_values.append(q)
            V_sync[s] = max(q_values)
            delta = max(delta, abs(v - V_sync[s]))
            sync_updates += 1
        if delta < theta:
            break
    print(f"Synchronous: {sync_updates} state updates to converge")

    # Asynchronous with prioritized sweeping (backward from goal)
    V_async = np.zeros(n_states)
    async_updates = 0

    # Priority queue: start from states near the goal
    priority = np.zeros(n_states)
    # Initialize: states near goal have highest priority
    for s in range(n_states - 1):
        for a in range(2):
            ns, r, done = get_next_state(s, a)
            if done or ns == n_states - 1:
                priority[s] = 1.0  # High priority for states near goal

    converged = False
    while not converged:
        # Pick state with highest priority
        s = np.argmax(priority)
        if priority[s] < theta:
            converged = True
            break

        priority[s] = 0  # Reset priority

        v = V_async[s]
        q_values = []
        for a in range(2):
            ns, r, done = get_next_state(s, a)
            q = r + (0 if done else gamma * V_async[ns])
            q_values.append(q)
        V_async[s] = max(q_values)
        async_updates += 1

        change = abs(v - V_async[s])
        if change > theta:
            # Update priorities of predecessors
            for pred_s in range(n_states - 1):
                for a in range(2):
                    ns, _, _ = get_next_state(pred_s, a)
                    if ns == s:
                        priority[pred_s] = max(priority[pred_s], change)

    print(f"Asynchronous (prioritized): {async_updates} state updates to converge")
    print(f"Speedup: {sync_updates / async_updates:.1f}x fewer updates")
    print()

    # Verify both converge to similar values
    max_diff = np.max(np.abs(V_sync - V_async))
    print(f"Max difference between solutions: {max_diff:.8f}")


if __name__ == "__main__":
    print("=== Exercise 1: Policy Evaluation ===")
    exercise_1()

    print("\n=== Exercise 2: Policy Iteration ===")
    exercise_2()

    print("\n=== Exercise 3: Value Iteration ===")
    exercise_3()

    print("\n=== Exercise 4: Asynchronous DP ===")
    exercise_4()

    print("\nAll exercises completed!")
