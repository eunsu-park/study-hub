# 03. Dynamic Programming

**Difficulty: ⭐⭐ (Basic)**

## Learning Objectives
- Understand the basic concepts of Dynamic Programming (DP)
- Implement the Policy Evaluation algorithm
- Understand and implement the Policy Iteration algorithm
- Understand and implement the Value Iteration algorithm
- Identify the limitations of DP methods

---

## 1. What is Dynamic Programming?

### 1.1 Overview

**Dynamic Programming (DP)** is a method of solving complex problems by breaking them down into smaller subproblems. In RL, DP computes the optimal policy when the **complete environment model (MDP)** is known.

### 1.2 Key Ideas of DP

1. **Optimal Substructure**: The optimal solution is composed of optimal solutions to subproblems
2. **Overlapping Subproblems**: The same subproblems are computed multiple times
3. **Memoization**: Store and reuse computation results

```
Use the recursive structure of Bellman equations to iteratively improve value functions

V_{k+1}(s) = f(V_k(s'))

Current estimate     Computed from
  updated           next state estimates
```

### 1.3 What DP Requires

| Requirement | Description |
|-------------|-------------|
| Complete MDP | Must know transition probabilities P(s'|s,a) |
| Finite state/action | Must be storable in tables |
| Computational resources | Must traverse all states |

---

## 2. Policy Evaluation

### 2.1 Objective

Compute the value function $V^\pi$ of a given policy π.

### 2.2 Iterative Policy Evaluation Algorithm

Apply the Bellman expectation equation iteratively until convergence.

$$V_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V_k(s') \right]$$

```python
import numpy as np
from typing import Dict, List, Tuple

def policy_evaluation(mdp, policy: Dict, gamma: float = 0.9,
                      theta: float = 1e-6) -> Dict:
    """
    Policy Evaluation: Calculate value function for given policy

    Args:
        mdp: MDP environment
        policy: Policy {state: {action: probability}}
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        V: State value function {state: value}
    """
    # Start with zero values — any initial estimate works because the Bellman
    # operator is a contraction mapping that converges regardless of initialization
    V = {s: 0.0 for s in mdp.get_states()}

    iteration = 0
    while True:
        # delta tracks the largest single-state change this sweep — using max (not sum)
        # gives a worst-case bound, so convergence means every state has stabilized
        delta = 0
        iteration += 1

        # Update for all states
        for s in mdp.get_states():
            if mdp.is_terminal(s):
                continue

            v = V[s]  # Store previous value
            new_v = 0

            # Apply Bellman expectation equation
            for a in mdp.actions:
                action_prob = policy[s].get(a, 0)

                for prob, next_s, reward, done in mdp.get_transitions(s, a):
                    if done:
                        new_v += action_prob * prob * reward
                    else:
                        # Using V[next_s] from the current sweep (not a separate copy)
                        # is the "in-place" variant — it converges faster because later
                        # states in the loop already benefit from earlier updates this iteration
                        new_v += action_prob * prob * (reward + gamma * V[next_s])

            V[s] = new_v
            delta = max(delta, abs(v - new_v))

        # theta is the convergence threshold: stopping when delta < theta guarantees
        # the value function is within theta/(1-gamma) of the true V^π
        if delta < theta:
            print(f"Policy evaluation converged: {iteration} iterations")
            break

    return V


# Usage example
class SimpleGridWorld:
    """Simple grid world"""
    def __init__(self, size=4):
        self.size = size
        self.actions = ['up', 'down', 'left', 'right']

    def get_states(self):
        return [(i, j) for i in range(self.size) for j in range(self.size)]

    def is_terminal(self, state):
        return state == (0, 0) or state == (self.size-1, self.size-1)

    def get_transitions(self, state, action):
        if self.is_terminal(state):
            return [(1.0, state, 0, True)]

        deltas = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        delta = deltas[action]

        new_row = max(0, min(self.size-1, state[0] + delta[0]))
        new_col = max(0, min(self.size-1, state[1] + delta[1]))
        next_state = (new_row, new_col)

        return [(1.0, next_state, -1, self.is_terminal(next_state))]


# Uniform random policy
def create_uniform_policy(mdp):
    policy = {}
    for s in mdp.get_states():
        policy[s] = {a: 1/len(mdp.actions) for a in mdp.actions}
    return policy


# Test
grid = SimpleGridWorld(4)
uniform_policy = create_uniform_policy(grid)
V = policy_evaluation(grid, uniform_policy)

# Print results
print("\nValue function for uniform random policy:")
for i in range(grid.size):
    row = [f"{V[(i,j)]:6.1f}" for j in range(grid.size)]
    print(" ".join(row))
```

### 2.3 Convergence

Policy evaluation always converges to $V^\pi$ (when gamma < 1 or episodic tasks).

```
Iteration 0: V = [0, 0, 0, 0, ...]
Iteration 1: V = [-1, -1, -1, ...]
Iteration 2: V = [-1.9, -2.0, ...]
  ...
Convergence: V = V^π
```

---

## 3. Policy Improvement

### 3.1 Policy Improvement Theorem

Given the value function $V^\pi$ of the current policy π, the **greedy policy** π' is equal to or better than π.

$$\pi'(s) = \arg\max_{a} Q^\pi(s, a) = \arg\max_{a} \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V^\pi(s') \right]$$

```python
def policy_improvement(mdp, V: Dict, gamma: float = 0.9) -> Tuple[Dict, bool]:
    """
    Policy Improvement: Generate greedy policy based on V

    Args:
        mdp: MDP environment
        V: Current value function
        gamma: Discount factor

    Returns:
        new_policy: Improved policy
        stable: True if policy unchanged
    """
    new_policy = {}
    policy_stable = True

    for s in mdp.get_states():
        if mdp.is_terminal(s):
            new_policy[s] = {a: 1/len(mdp.actions) for a in mdp.actions}
            continue

        # Recompute Q values from V rather than storing a separate Q table —
        # this saves memory and ensures Q is always consistent with the current V
        q_values = {}
        for a in mdp.actions:
            q = 0
            for prob, next_s, reward, done in mdp.get_transitions(s, a):
                if done:
                    q += prob * reward
                else:
                    q += prob * (reward + gamma * V[next_s])
            q_values[a] = q

        # Find best action
        best_action = max(q_values, key=q_values.get)
        best_q = q_values[best_action]

        # Tie-breaking with 1e-8 tolerance handles floating-point noise — without this,
        # two mathematically equal actions might be treated as different, causing
        # unnecessary policy oscillation between iterations
        best_actions = [a for a, q in q_values.items()
                        if abs(q - best_q) < 1e-8]

        # Distribute probability equally among tied best actions — this produces a
        # deterministic policy in the common case (one winner) but degrades gracefully
        # when multiple actions are genuinely equivalent
        new_policy[s] = {a: 0.0 for a in mdp.actions}
        for a in best_actions:
            new_policy[s][a] = 1.0 / len(best_actions)

    return new_policy, policy_stable


def get_greedy_action(mdp, V, state, gamma):
    """Select greedy action from value function"""
    q_values = {}
    for a in mdp.actions:
        q = 0
        for prob, next_s, reward, done in mdp.get_transitions(state, a):
            if done:
                q += prob * reward
            else:
                q += prob * (reward + gamma * V[next_s])
        q_values[a] = q

    return max(q_values, key=q_values.get)
```

---

## 4. Policy Iteration

### 4.1 Algorithm

Find the optimal policy by alternating between policy evaluation and policy improvement.

```
1. Initialize: arbitrary policy π₀

2. Iterate:
   a. Policy Evaluation: compute V^πₖ
   b. Policy Improvement: πₖ₊₁ = greedy(V^πₖ)

3. Terminate if πₖ₊₁ = πₖ (optimal policy)
```

### 4.2 Implementation

```python
def policy_iteration(mdp, gamma: float = 0.9, theta: float = 1e-6):
    """
    Policy Iteration Algorithm

    Returns:
        V: Optimal value function
        policy: Optimal policy
    """
    # A uniform random policy is a safe starting point — it's unlikely to be optimal
    # but guarantees every state gets explored during the first evaluation pass
    policy = create_uniform_policy(mdp)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n=== Policy Iteration {iteration} ===")

        # 1. Policy Evaluation — run to full convergence before improving;
        # this is the key difference from value iteration (which evaluates only once)
        V = policy_evaluation(mdp, policy, gamma, theta)

        # 2. Policy Improvement
        old_policy = policy.copy()
        new_policy = {}
        policy_stable = True

        for s in mdp.get_states():
            if mdp.is_terminal(s):
                new_policy[s] = {a: 1/len(mdp.actions) for a in mdp.actions}
                continue

            # Calculate Q values
            q_values = {}
            for a in mdp.actions:
                q = 0
                for prob, next_s, reward, done in mdp.get_transitions(s, a):
                    if done:
                        q += prob * reward
                    else:
                        q += prob * (reward + gamma * V[next_s])
                q_values[a] = q

            # Greedy policy: commit fully (probability 1.0) to the best action —
            # a deterministic greedy policy is always at least as good as the mixed policy
            best_action = max(q_values, key=q_values.get)
            new_policy[s] = {a: 0.0 for a in mdp.actions}
            new_policy[s][best_action] = 1.0

            # policy_stable tracks whether any state changed its best action —
            # if no state changed, the policy is already greedy w.r.t. its own V, which
            # means it is optimal (Policy Improvement Theorem)
            old_best = max(old_policy[s], key=old_policy[s].get)
            if old_best != best_action:
                policy_stable = False

        policy = new_policy

        # 3. Check convergence
        if policy_stable:
            print(f"\nPolicy iteration converged! (total {iteration} iterations)")
            break

    return V, policy


# Test
print("=" * 50)
print("Policy Iteration")
print("=" * 50)

grid = SimpleGridWorld(4)
V_star, optimal_policy = policy_iteration(grid)

# Print results
print("\nOptimal value function:")
for i in range(grid.size):
    row = [f"{V_star[(i,j)]:6.1f}" for j in range(grid.size)]
    print(" ".join(row))

print("\nOptimal policy:")
arrows = {'up': '^', 'down': 'v', 'left': '<', 'right': '>'}
for i in range(grid.size):
    row = []
    for j in range(grid.size):
        s = (i, j)
        if grid.is_terminal(s):
            row.append('  *  ')
        else:
            best_a = max(optimal_policy[s], key=optimal_policy[s].get)
            row.append(f'  {arrows[best_a]}  ')
    print(" ".join(row))
```

---

## 5. Value Iteration

### 5.1 Idea

Instead of fully converging policy evaluation, perform policy improvement immediately after **a single update**.

$$V_{k+1}(s) = \max_{a} \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V_k(s') \right]$$

### 5.2 Algorithm

```
1. Initialize: V₀(s) = 0 for all s

2. Iterate:
   Vₖ₊₁(s) = max_a Σ P(s'|s,a)[R + γVₖ(s')]

3. Terminate if |Vₖ₊₁ - Vₖ| < θ

4. Optimal policy: π*(s) = argmax_a Q*(s,a)
```

### 5.3 Implementation

```python
def value_iteration(mdp, gamma: float = 0.9, theta: float = 1e-6):
    """
    Value Iteration Algorithm

    Returns:
        V: Optimal value function
        policy: Optimal policy
    """
    # Initialize value function
    V = {s: 0.0 for s in mdp.get_states()}

    iteration = 0
    while True:
        delta = 0
        iteration += 1

        for s in mdp.get_states():
            if mdp.is_terminal(s):
                continue

            v = V[s]

            # Bellman optimality equation: taking max instead of a policy-weighted average
            # implicitly performs policy improvement every sweep — this collapses the
            # evaluation + improvement loop into a single update per state
            q_values = []
            for a in mdp.actions:
                q = 0
                for prob, next_s, reward, done in mdp.get_transitions(s, a):
                    if done:
                        q += prob * reward
                    else:
                        q += prob * (reward + gamma * V[next_s])
                q_values.append(q)

            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: delta = {delta:.6f}")

        if delta < theta:
            print(f"\nValue iteration converged: {iteration} iterations")
            break

    # Extract optimal policy after V has converged — policy extraction is a separate
    # pass rather than storing π during iteration, because we only care about the
    # final greedy policy, not the implicit intermediate policies
    policy = {}
    for s in mdp.get_states():
        if mdp.is_terminal(s):
            policy[s] = {a: 1/len(mdp.actions) for a in mdp.actions}
            continue

        q_values = {}
        for a in mdp.actions:
            q = 0
            for prob, next_s, reward, done in mdp.get_transitions(s, a):
                if done:
                    q += prob * reward
                else:
                    q += prob * (reward + gamma * V[next_s])
            q_values[a] = q

        best_action = max(q_values, key=q_values.get)
        policy[s] = {a: 0.0 for a in mdp.actions}
        policy[s][best_action] = 1.0

    return V, policy


# Test
print("=" * 50)
print("Value Iteration")
print("=" * 50)

grid = SimpleGridWorld(4)
V_star, optimal_policy = value_iteration(grid)

# Print results
print("\nOptimal value function:")
for i in range(grid.size):
    row = [f"{V_star[(i,j)]:6.1f}" for j in range(grid.size)]
    print(" ".join(row))
```

---

## 6. Policy Iteration vs Value Iteration

### 6.1 Comparison

| Property | Policy Iteration | Value Iteration |
|----------|------------------|-----------------|
| Inner loop | Full convergence | Single update |
| Outer iterations | Few | Many |
| Computation per iteration | High | Low |
| Convergence speed | Fast (in iterations) | Slow (in iterations) |
| Memory | Store both V and π | Store only V |

### 6.2 Generalized Policy Iteration (GPI)

```
        Evaluation
    ┌─────────────┐
    │  V → V^π   │
    └─────────────┘
         ↑    ↓
         │    │
         │    ↓
    ┌─────────────┐
    │  π' = greedy(V) │
    └─────────────┘
      Improvement

Policy Iteration: Full convergence then improvement
Value Iteration: Single update then improvement
k-step Iteration: k updates then improvement
```

### 6.3 Asynchronous DP

Instead of updating all states in order, update **selected states only**.

```python
def async_value_iteration(mdp, gamma=0.9, n_updates=10000):
    """Asynchronous value iteration"""
    V = {s: 0.0 for s in mdp.get_states()}
    states = [s for s in mdp.get_states() if not mdp.is_terminal(s)]

    for i in range(n_updates):
        # Randomly select state
        s = np.random.choice(len(states))
        s = states[s]

        # Update only that state
        q_values = []
        for a in mdp.actions:
            q = 0
            for prob, next_s, reward, done in mdp.get_transitions(s, a):
                if done:
                    q += prob * reward
                else:
                    q += prob * (reward + gamma * V[next_s])
            q_values.append(q)

        V[s] = max(q_values)

    return V
```

---

## 7. Limitations of DP

### 7.1 Main Limitations

| Limitation | Description |
|------------|-------------|
| Requires complete model | Must know P(s'\|s,a) and R exactly |
| Curse of dimensionality | Impossible to compute with large state/action spaces |
| Table storage | Impossible for continuous state spaces |
| Computation | Must traverse all states each iteration |

### 7.2 Solution Directions

```
Limitation              Solution
──────────────────────────────────────────────
Model required    →     Model-free methods (MC, TD)
Curse of dim.     →     Function approximation (neural nets)
Table impossible  →     Continuous space algorithms
Computation       →     Sample-based learning
```

---

## 8. Complete Example: Frozen Lake

```python
import gymnasium as gym
import numpy as np

def dp_frozen_lake():
    """Apply DP to the Frozen Lake environment"""

    # is_slippery=False removes stochastic transitions so DP converges in very few
    # iterations — use is_slippery=True to test robustness against wind/slip noise
    env = gym.make('FrozenLake-v1', is_slippery=False)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # gamma=0.99 instead of 0.9: with a 4x4 grid the goal is up to 6 steps away,
    # so a higher gamma is needed to propagate the reward signal back to the start
    gamma = 0.99
    theta = 1e-8

    # P[s][a] = [(prob, next_state, reward, done), ...]
    # Accessing env.unwrapped.P is the standard way to read the model from Gymnasium —
    # without .unwrapped, wrappers may intercept attribute access
    P = env.unwrapped.P

    # Value iteration
    V = np.zeros(n_states)

    for iteration in range(1000):
        delta = 0

        for s in range(n_states):
            v = V[s]

            # Compute value for each action
            q_values = []
            for a in range(n_actions):
                # (not done) is 1 for non-terminal and 0 for terminal, so V[next_s]
                # is automatically zeroed out at terminal states without a separate if-branch
                q = sum(prob * (reward + gamma * V[next_s] * (not done))
                        for prob, next_s, reward, done in P[s][a])
                q_values.append(q)

            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            print(f"Converged: {iteration + 1} iterations")
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
    print("\nOptimal policy (4x4 grid):")
    for i in range(4):
        row = ""
        for j in range(4):
            s = i * 4 + j
            if s in [5, 7, 11, 12]:  # holes
                row += "  H  "
            elif s == 15:  # goal
                row += "  G  "
            else:
                row += f"  {action_names[policy[s]]}  "
        print(row)

    print("\nValue function:")
    print(V.reshape(4, 4).round(3))

    # Test policy on a fresh environment to measure real-world performance —
    # re-instantiating env avoids state leakage from the training phase
    env = gym.make('FrozenLake-v1', is_slippery=False)
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

    print(f"\nSuccess rate: {success}/{n_tests} = {success/n_tests*100:.1f}%")

    env.close()
    return V, policy


# Run
if __name__ == "__main__":
    V, policy = dp_frozen_lake()
```

---

## 9. Summary

### Core Algorithms

| Algorithm | Purpose | Key Equation |
|-----------|---------|--------------|
| Policy Evaluation | Compute V^π | Bellman expectation equation |
| Policy Improvement | π → π' | argmax Q |
| Policy Iteration | Optimal policy | Evaluation + Improvement loop |
| Value Iteration | Optimal policy | Bellman optimality equation |

### Time Complexity

| Algorithm | Complexity per Iteration |
|-----------|--------------------------|
| Policy Evaluation | O(\|S\|^2 \|A\|) |
| Policy Improvement | O(\|S\| \|A\|) |
| Value Iteration | O(\|S\|^2 \|A\|) |

---

## 10. Practice Problems

1. **Policy Evaluation**: Manually calculate the value function for a uniform random policy on a 2x2 grid.

2. **Policy Iteration**: Explain why policy iteration always terminates in finite steps.

3. **Value Iteration**: Why does value iteration converge in 1 iteration when γ=0?

4. **Asynchronous DP**: Explain situations where asynchronous DP can be faster than synchronous DP.

---

## Next Steps

In the next lesson **04_Monte_Carlo_Methods.md**, we will learn Monte Carlo methods that learn from experience **without an environment model**.

---

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 4
- David Silver's RL Course, Lecture 3: Planning by Dynamic Programming
- [Gymnasium FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
