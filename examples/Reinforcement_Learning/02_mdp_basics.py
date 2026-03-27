"""
Markov Decision Process (MDP) — Example Code

Demonstrates core MDP concepts:
  1. Markov property and state representation
  2. Bellman expectation equation evaluation
  3. Bellman optimality equation
  4. State and action value function computation

No external dependencies required.
"""

import numpy as np
from typing import Dict, Tuple


# ============================================================
# 1. MDP Definition
# ============================================================

class StudentMDP:
    """
    Classic student MDP from Sutton & Barto.

    States:  Facebook, Class1, Class2, Class3, Pass, Pub, Sleep
    Actions: (state-dependent)
    Terminal states: Pass, Sleep
    """

    STATES = ['Facebook', 'Class1', 'Class2', 'Class3', 'Pass', 'Pub', 'Sleep']
    TERMINAL = {'Pass', 'Sleep'}

    # transitions: {state: [(prob, next_state, reward), ...]}
    TRANSITIONS = {
        'Facebook': [(0.5, 'Facebook', -1), (0.5, 'Class1', -1)],
        'Class1':   [(0.5, 'Facebook', -1), (0.5, 'Class2', -1)],
        'Class2':   [(0.4, 'Class3', -1),   (0.6, 'Pub', 1)],
        'Class3':   [(0.6, 'Pass', 10),     (0.4, 'Pub', 1)],
        'Pub':      [(0.2, 'Class1', 1),    (0.4, 'Class2', 1), (0.4, 'Class3', 1)],
        'Pass':     [(1.0, 'Sleep', 0)],
        'Sleep':    [(1.0, 'Sleep', 0)],
    }

    def is_terminal(self, state):
        return state in self.TERMINAL

    def get_transitions(self, state):
        return self.TRANSITIONS[state]


def bellman_evaluation(mdp: StudentMDP, gamma: float = 1.0, theta: float = 1e-8):
    """
    Compute the value function V(s) for a fixed (random) policy
    using iterative Bellman expectation evaluation.
    """
    V = {s: 0.0 for s in mdp.STATES}

    iteration = 0
    while True:
        delta = 0.0
        iteration += 1
        for s in mdp.STATES:
            if mdp.is_terminal(s):
                continue
            v_old = V[s]
            # Bellman expectation equation
            V[s] = sum(prob * (r + gamma * V[ns])
                       for prob, ns, r in mdp.get_transitions(s))
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break

    print(f"Converged in {iteration} iterations (gamma={gamma})")
    return V


def demonstrate_bellman_evaluation():
    """Show Bellman evaluation on the Student MDP."""
    print("=" * 60)
    print("1. Bellman Expectation Equation (Student MDP)")
    print("=" * 60)

    mdp = StudentMDP()

    for gamma in [0.9, 1.0]:
        print(f"\ngamma = {gamma}")
        V = bellman_evaluation(mdp, gamma=gamma)
        for s in mdp.STATES:
            print(f"  V({s}) = {V[s]:7.3f}")


# ============================================================
# 2. MDP with Explicit Actions
# ============================================================

class GridMDP:
    """
    4x4 grid MDP.
    Actions: up, down, left, right
    Terminal: top-left (0,0) and bottom-right (3,3)
    Reward: -1 per step
    """

    SIZE = 4
    ACTIONS = ['up', 'down', 'left', 'right']
    DELTAS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

    def states(self):
        return [(r, c) for r in range(self.SIZE) for c in range(self.SIZE)]

    def is_terminal(self, s):
        return s == (0, 0) or s == (self.SIZE - 1, self.SIZE - 1)

    def transition(self, s, a):
        """Returns (next_state, reward, done)."""
        dr, dc = self.DELTAS[a]
        ns = (max(0, min(self.SIZE - 1, s[0] + dr)),
              max(0, min(self.SIZE - 1, s[1] + dc)))
        done = self.is_terminal(ns)
        return ns, -1, done


def compute_q_from_v(mdp: GridMDP, V: Dict, gamma: float = 0.9):
    """Compute Q(s,a) from V(s) using the Bellman equation."""
    Q = {}
    for s in mdp.states():
        Q[s] = {}
        for a in mdp.ACTIONS:
            ns, r, done = mdp.transition(s, a)
            if done:
                Q[s][a] = r
            else:
                Q[s][a] = r + gamma * V[ns]
    return Q


def demonstrate_q_from_v():
    """Show how Q(s,a) is derived from V(s) via the Bellman equation."""
    print("\n" + "=" * 60)
    print("2. Q(s,a) from V(s) — Bellman Relationship")
    print("=" * 60)

    mdp = GridMDP()
    gamma = 0.9

    # Uniform random policy value function
    V = {s: 0.0 for s in mdp.states()}
    for _ in range(200):  # rough iterative evaluation
        for s in mdp.states():
            if mdp.is_terminal(s):
                continue
            V[s] = sum(
                (-1 + gamma * V[mdp.transition(s, a)[0]])
                if not mdp.transition(s, a)[2]
                else -1
                for a in mdp.ACTIONS
            ) / len(mdp.ACTIONS)

    # Compute Q from V
    Q = compute_q_from_v(mdp, V, gamma)

    # Print Q values for state (1,1) in the center
    s = (1, 1)
    print(f"\nQ-values for state {s}:")
    for a in mdp.ACTIONS:
        print(f"  Q({s}, {a}) = {Q[s][a]:.3f}")

    best_a = max(Q[s], key=Q[s].get)
    print(f"  => Best action: {best_a}")

    # Verify: V(s) = max_a Q(s,a) should hold for optimal V
    print(f"\n  V({s}) from evaluation = {V[s]:.3f}")
    print(f"  max_a Q({s},a) = {max(Q[s].values()):.3f}")
    print("  (Difference is expected because V was evaluated under random policy,")
    print("   not optimal policy.)")


# ============================================================
# 3. Bellman Optimality
# ============================================================

def value_iteration_mdp(mdp: GridMDP, gamma: float = 0.9, theta: float = 1e-8):
    """
    Value iteration: repeatedly apply Bellman optimality operator.
    V*(s) = max_a sum_{s'} P(s'|s,a) [r + gamma * V*(s')]
    """
    V = {s: 0.0 for s in mdp.states()}

    iteration = 0
    while True:
        delta = 0.0
        iteration += 1
        for s in mdp.states():
            if mdp.is_terminal(s):
                continue
            v_old = V[s]
            q_values = []
            for a in mdp.ACTIONS:
                ns, r, done = mdp.transition(s, a)
                q_values.append(r if done else r + gamma * V[ns])
            V[s] = max(q_values)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break

    return V, iteration


def demonstrate_bellman_optimality():
    """Solve the GridMDP optimally via value iteration."""
    print("\n" + "=" * 60)
    print("3. Bellman Optimality — Value Iteration on GridMDP")
    print("=" * 60)

    mdp = GridMDP()
    V_opt, iters = value_iteration_mdp(mdp)

    print(f"\nOptimal V* (converged in {iters} iterations):")
    for r in range(mdp.SIZE):
        row = "  "
        for c in range(mdp.SIZE):
            row += f"{V_opt[(r,c)]:7.2f}"
        print(row)

    # Extract optimal policy
    arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    print("\nOptimal Policy π*:")
    for r in range(mdp.SIZE):
        row = "  "
        for c in range(mdp.SIZE):
            s = (r, c)
            if mdp.is_terminal(s):
                row += "   *   "
            else:
                q_vals = {a: (-1 if mdp.transition(s, a)[2]
                              else -1 + 0.9 * V_opt[mdp.transition(s, a)[0]])
                          for a in mdp.ACTIONS}
                best = max(q_vals, key=q_vals.get)
                row += f"   {arrows[best]}   "
        print(row)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demonstrate_bellman_evaluation()
    demonstrate_q_from_v()
    demonstrate_bellman_optimality()

    print("\n" + "=" * 60)
    print("MDP examples complete!")
    print("=" * 60)
