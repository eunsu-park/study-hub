# 02. Markov Decision Process (MDP)

**Difficulty: ⭐⭐ (Basics)**

## Learning Objectives
- Understand the Markov property and definition of MDP
- Learn state value function V(s) and action value function Q(s,a)
- Derive and interpret Bellman expectation and optimality equations
- Grasp the existence and properties of optimal policies

---

## 1. Markov Property

### 1.1 Definition

**Markov Property**: The future depends only on the current state and is independent of past history.

$$P(S_{t+1} | S_t) = P(S_{t+1} | S_1, S_2, \ldots, S_t)$$

```python
"""
Markov Property Examples:

✓ Chess game: Knowing only the current board state is sufficient for optimal moves
  - Past moves don't matter

✗ Poker game: Opponent's past betting patterns are important
  - Cannot make optimal decisions with just current cards
  → Can be made Markovian by including past information in the state
"""
```

### 1.2 Importance of State Representation

To satisfy the Markov property, **states must contain sufficient information**.

```python
# Example: Non-Markov → Markov transformation
class StateRepresentation:
    """
    Original state: Ball's current position (x)
    Problem: Cannot predict where ball will move (no velocity info)

    Markov state: (position, velocity) = (x, v)
    Solution: Next position predictable (x' = x + v)
    """

    def non_markov_state(self, ball):
        return ball.position  # Insufficient information

    def markov_state(self, ball):
        return (ball.position, ball.velocity)  # Sufficient information
```

---

## 2. Markov Decision Process (MDP)

### 2.1 Five Elements of MDP

An MDP is defined by the tuple $(S, A, P, R, \gamma)$.

| Element | Symbol | Description |
|---------|--------|-------------|
| State Space | $S$ | Set of all possible states |
| Action Space | $A$ | Set of all possible actions |
| Transition Probability | $P$ | $P(s'|s, a)$ - State transition probability |
| Reward Function | $R$ | $R(s, a, s')$ - Immediate reward |
| Discount Factor | $\gamma$ | Discount for future rewards (0 ≤ γ ≤ 1) |

### 2.2 MDP Diagram

```
           a₁                    a₂
    ┌──────────────┐      ┌──────────────┐
    │              │      │              │
    ▼     r=+1     │      ▼     r=-1     │
   s₁ ─────────────┼─────s₂──────────────┼────► s₃
    │    p=0.7     │      │    p=0.3     │      │
    │              │      │              │      │
    └──────────────┘      └──────────────┘      ▼
                                              Terminal
```

### 2.3 Defining MDP in Python

```python
import numpy as np
from typing import Dict, Tuple, List

class MDP:
    """Markov Decision Process class"""

    def __init__(self):
        # State space
        self.states = ['s0', 's1', 's2', 'terminal']

        # Action space
        self.actions = ['left', 'right']

        # Discount factor
        # γ=0.9 means a reward 10 steps away is worth only 0.9^10 ≈ 35% of an immediate reward —
        # this prevents the agent from equally weighting the distant future and the present
        self.gamma = 0.9

        # Transition probabilities: P[s][a] = [(prob, next_state, reward, done), ...]
        self.P = self._build_transitions()

    def _build_transitions(self) -> Dict:
        """Define transition probabilities"""
        P = {}

        # Transitions from state s0
        # Wall collision gives -1: the negative reward teaches the agent that
        # bumping walls wastes time and should be avoided
        P['s0'] = {
            'left': [(1.0, 's0', -1, False)],      # Hit wall
            'right': [(1.0, 's1', 0, False)]       # Move to s1
        }

        # Transitions from state s1
        # 80/20 stochastic split models real-world uncertainty — the agent must learn
        # a robust policy that accounts for uncontrolled slippage, not just the happy path
        P['s1'] = {
            'left': [(1.0, 's0', 0, False)],       # Move to s0
            'right': [(0.8, 's2', 0, False),       # 80%: move to s2
                      (0.2, 's1', 0, False)]       # 20%: stay
        }

        # Transitions from state s2
        P['s2'] = {
            'left': [(1.0, 's1', 0, False)],       # Move to s1
            'right': [(1.0, 'terminal', +10, True)] # Reach goal!
        }

        # Terminal state absorbs all actions with zero reward — this prevents the
        # agent from trying to "escape" the goal and artificially inflating returns
        P['terminal'] = {
            'left': [(1.0, 'terminal', 0, True)],
            'right': [(1.0, 'terminal', 0, True)]
        }

        return P

    def get_transitions(self, state: str, action: str) -> List[Tuple]:
        """Return transition information for given state and action"""
        return self.P[state][action]

    def step(self, state: str, action: str) -> Tuple[str, float, bool]:
        """Execute one step in environment (stochastic transition)"""
        transitions = self.P[state][action]
        probs = [t[0] for t in transitions]
        # np.random.choice with p= samples from the full probability distribution in one call —
        # this correctly handles multi-outcome stochastic transitions without manual loops
        idx = np.random.choice(len(transitions), p=probs)
        prob, next_state, reward, done = transitions[idx]
        return next_state, reward, done


# MDP usage example
mdp = MDP()

# Run episode
state = 's0'
total_reward = 0

print("=== MDP Simulation ===")
while True:
    action = np.random.choice(mdp.actions)  # Random policy
    next_state, reward, done = mdp.step(state, action)

    print(f"{state} --[{action}]--> {next_state}, reward={reward}")

    total_reward += reward
    state = next_state

    if done:
        break

print(f"\nTotal reward: {total_reward}")
```

---

## 3. Policy

### 3.1 Definition of Policy

**Policy π** is a rule for selecting actions in states.

- **Deterministic policy**: $\pi(s) = a$
- **Stochastic policy**: $\pi(a|s) = P(A_t = a | S_t = s)$

```python
class Policy:
    """Policy class"""

    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        # Stochastic policy: action probability distribution for each state
        self.policy = {s: {a: 1/len(actions) for a in actions}
                       for s in states}

    def get_action_prob(self, state: str, action: str) -> float:
        """Return π(a|s)"""
        return self.policy[state][action]

    def sample_action(self, state: str) -> str:
        """Sample action according to policy"""
        actions = list(self.policy[state].keys())
        probs = list(self.policy[state].values())
        return np.random.choice(actions, p=probs)

    def set_deterministic(self, state: str, action: str):
        """Set as deterministic policy"""
        for a in self.actions:
            self.policy[state][a] = 1.0 if a == action else 0.0


# Example: Policy that always moves right
policy = Policy(['s0', 's1', 's2'], ['left', 'right'])
policy.set_deterministic('s0', 'right')
policy.set_deterministic('s1', 'right')
policy.set_deterministic('s2', 'right')
```

---

## 4. Value Functions

### 4.1 State Value Function V(s)

The expected **cumulative reward** when following policy π from state s.

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t = s\right]$$

```python
def compute_state_value(mdp, policy, state, gamma, depth=100):
    """
    Estimate state value using Monte Carlo approach
    (average return over multiple episodes)
    """
    returns = []
    # 1000 episodes gives a statistically stable average — fewer episodes produce
    # high variance estimates; more would be accurate but computationally wasteful
    n_episodes = 1000

    for _ in range(n_episodes):
        s = state
        episode_return = 0
        # Start discount at 1.0 so the first reward is undiscounted (r * γ^0 = r),
        # then multiply by gamma each step to implement γ^k weighting automatically
        discount = 1.0

        for _ in range(depth):
            action = policy.sample_action(s)
            next_s, reward, done = mdp.step(s, action)

            episode_return += discount * reward
            discount *= gamma
            s = next_s

            if done:
                break

        returns.append(episode_return)

    # Average over episodes to approximate E_π[G_t | S_t=state] —
    # by the law of large numbers this converges to the true V^π(state)
    return np.mean(returns)
```

### 4.2 Action Value Function Q(s, a)

Expected **cumulative reward** when taking action a in state s, then following policy π.

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

```python
def compute_action_value(mdp, policy, state, action, gamma, depth=100):
    """Estimate action value"""
    returns = []
    n_episodes = 1000

    for _ in range(n_episodes):
        # First action is given — this is what distinguishes Q(s,a) from V(s):
        # we fix the first action, then let the policy take over for all subsequent steps
        s = state
        next_s, reward, done = mdp.step(s, action)

        episode_return = reward
        # Discount starts at gamma (not 1.0) because the first reward is already
        # collected above; all subsequent rewards belong to step k≥1, so they scale by γ^k
        discount = gamma
        s = next_s

        # Then follow policy
        for _ in range(depth - 1):
            if done:
                break

            a = policy.sample_action(s)
            next_s, reward, done = mdp.step(s, a)

            episode_return += discount * reward
            discount *= gamma
            s = next_s

        returns.append(episode_return)

    return np.mean(returns)
```

### 4.3 Relationship between V and Q

$$V^\pi(s) = \sum_{a \in A} \pi(a|s) \cdot Q^\pi(s, a)$$

```python
def v_from_q(policy, q_values, state, actions):
    """Calculate V value from Q values"""
    v = 0
    for a in actions:
        v += policy.get_action_prob(state, a) * q_values[(state, a)]
    return v
```

---

## 5. Bellman Equations

### 5.1 Bellman Expectation Equation

Expresses the value of the current state **recursively** in terms of the value of the next state.

**State value function:**

$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V^\pi(s') \right]$$

**Action value function:**

$$Q^\pi(s, a) = \sum_{s', r} P(s', r | s, a) \left[ r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]$$

### 5.2 Intuitive Understanding of Bellman Equation

```
V(s) = Immediate reward + Discounted future value

     ┌─────────────────────────────────────────┐
     │   V(s) = r + γ * V(s')                  │
     │                                          │
     │   Current value = Immediate reward + γ × Next state value│
     └─────────────────────────────────────────┘
```

### 5.3 Bellman Equation Implementation

```python
def bellman_expectation_v(mdp, policy, V, state, gamma):
    """
    Calculate V(s) using Bellman expectation equation

    V(s) = Σ_a π(a|s) * Σ_{s',r} P(s',r|s,a) * [r + γV(s')]
    """
    # Terminal states always have value 0 — there are no future rewards to collect,
    # so adding a base case here prevents the recursion from looping infinitely
    if state == 'terminal':
        return 0

    value = 0

    for action in mdp.actions:
        action_prob = policy.get_action_prob(state, action)

        for prob, next_state, reward, done in mdp.get_transitions(state, action):
            if done:
                # When transitioning to terminal, future value is zero — we only
                # credit the immediate reward, not γ * V(terminal)
                value += action_prob * prob * reward
            else:
                # The two-level sum (over actions, then over outcomes) computes the
                # full expectation: E_π[r + γV(s')] = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[r + γV(s')]
                value += action_prob * prob * (reward + gamma * V.get(next_state, 0))

    return value


def bellman_expectation_q(mdp, policy, Q, state, action, gamma):
    """
    Calculate Q(s,a) using Bellman expectation equation

    Q(s,a) = Σ_{s',r} P(s',r|s,a) * [r + γ * Σ_a' π(a'|s') * Q(s',a')]
    """
    value = 0

    for prob, next_state, reward, done in mdp.get_transitions(state, action):
        if done:
            value += prob * reward
        else:
            # V(s') = Σ_a' π(a'|s') Q(s',a') — converting Q back to V at the next
            # state lets us express Q entirely in terms of existing Q estimates
            next_value = sum(
                policy.get_action_prob(next_state, a) * Q.get((next_state, a), 0)
                for a in mdp.actions
            )
            value += prob * (reward + gamma * next_value)

    return value
```

---

## 6. Optimal Value Functions and Optimal Policy

### 6.1 Optimal Value Functions

**Optimal state value function** $V^*(s)$: Maximum value across all policies

$$V^*(s) = \max_\pi V^\pi(s)$$

**Optimal action value function** $Q^*(s, a)$:

$$Q^*(s, a) = \max_\pi Q^\pi(s, a)$$

### 6.2 Bellman Optimality Equation

$$V^*(s) = \max_{a} \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V^*(s') \right]$$

$$Q^*(s, a) = \sum_{s', r} P(s', r | s, a) \left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$

```python
def bellman_optimality_v(mdp, V, state, gamma):
    """
    Calculate V*(s) using Bellman optimality equation

    V*(s) = max_a Σ_{s',r} P(s',r|s,a) * [r + γV*(s')]
    """
    if state == 'terminal':
        return 0

    # Initialize to -inf so any real action value will replace it —
    # using 0 would be wrong when all actions yield negative returns
    max_value = float('-inf')

    for action in mdp.actions:
        action_value = 0

        for prob, next_state, reward, done in mdp.get_transitions(state, action):
            if done:
                action_value += prob * reward
            else:
                action_value += prob * (reward + gamma * V.get(next_state, 0))

        # The max replaces the policy's weighted average — optimality means we
        # always commit to the single best action rather than averaging over all actions
        max_value = max(max_value, action_value)

    return max_value


def bellman_optimality_q(mdp, Q, state, action, gamma):
    """
    Calculate Q*(s,a) using Bellman optimality equation

    Q*(s,a) = Σ_{s',r} P(s',r|s,a) * [r + γ * max_a' Q*(s',a')]
    """
    value = 0

    for prob, next_state, reward, done in mdp.get_transitions(state, action):
        if done:
            value += prob * reward
        else:
            # max over next actions encodes the optimality principle: at s', we
            # assume the agent will always take the best available action onward
            max_next_q = max(Q.get((next_state, a), 0) for a in mdp.actions)
            value += prob * (reward + gamma * max_next_q)

    return value
```

### 6.3 Optimal Policy

Optimal policy $\pi^*$ can be easily derived if $Q^*$ is known:

$$\pi^*(s) = \arg\max_{a} Q^*(s, a)$$

```python
def get_optimal_policy(mdp, Q_star):
    """Derive optimal policy from Q*"""
    optimal_policy = {}

    for state in mdp.states:
        if state == 'terminal':
            continue

        # Calculate Q* value for each action
        q_values = {a: Q_star.get((state, a), 0) for a in mdp.actions}

        # Select action with maximum Q* value
        optimal_action = max(q_values, key=q_values.get)
        optimal_policy[state] = optimal_action

    return optimal_policy
```

---

## 7. Types of MDPs

### 7.1 Finite MDP vs Infinite MDP

| Characteristic | Finite MDP | Infinite MDP |
|----------------|------------|--------------|
| State space | Finite | Infinite (continuous) |
| Action space | Finite | Infinite (continuous) |
| Representation | Table | Function approximation required |
| Example | Grid world | Robot control |

### 7.2 Deterministic vs Stochastic MDP

```python
# Deterministic MDP: P(s'|s,a) = 1 for one s'
deterministic_transitions = {
    's0': {'right': [(1.0, 's1', 0, False)]}  # 100% move to s1
}

# Stochastic MDP: P(s'|s,a) can be < 1
stochastic_transitions = {
    's0': {'right': [(0.8, 's1', 0, False),   # 80% move to s1
                     (0.2, 's0', -1, False)]}  # 20% stay
}
```

### 7.3 Partially Observable MDP (POMDP)

When the state cannot be directly observed, only **observations** are available

```python
"""
POMDP example: Poker game
- True state: All players' cards + deck cards
- Observation: Only own cards + public cards
- Belief state: Maintain probability distribution over true states

Solutions:
1. Use belief state as state (belief MDP)
2. Use history of past observations
3. Learn implicit state using RNN/LSTM
"""
```

---

## 8. GridWorld Example

### 8.1 Environment Definition

```python
import numpy as np

class GridWorld:
    """
    4x4 Gridworld

    [S][ ][ ][ ]
    [ ][X][ ][ ]
    [ ][ ][ ][ ]
    [ ][ ][ ][G]

    S: Start, G: Goal(+1), X: Obstacle(-1)
    """

    def __init__(self):
        self.size = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.obstacle = (1, 1)

        self.actions = ['up', 'down', 'left', 'right']
        self.action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        self.gamma = 0.9

    def get_states(self):
        """Return all states"""
        states = []
        for i in range(self.size):
            for j in range(self.size):
                states.append((i, j))
        return states

    def is_terminal(self, state):
        """Check if terminal state"""
        return state == self.goal

    def get_reward(self, state, action, next_state):
        """Reward function"""
        if next_state == self.goal:
            return 1.0
        elif next_state == self.obstacle:
            return -1.0
        else:
            return -0.01  # Step penalty

    def get_transitions(self, state, action):
        """
        Return transition probabilities
        80% intended direction, 10% each for slipping left/right
        """
        # Early return for terminal states avoids indexing into action_deltas for a
        # state that has no meaningful transitions — keeps the absorbing-state contract clean
        if self.is_terminal(state):
            return [(1.0, state, 0, True)]

        transitions = []

        # Intended direction
        intended_delta = self.action_deltas[action]

        # Slip directions are perpendicular to the intended action — this models
        # real environments where uncertainty is orthogonal, not backward (which would
        # be even harder to recover from and rarely matches physical slip dynamics)
        if action in ['up', 'down']:
            slip_actions = ['left', 'right']
        else:
            slip_actions = ['up', 'down']

        # Bundling all three outcomes into one list lets us loop uniformly over the
        # probability-weighted directions rather than duplicating the transition logic
        directions = [(0.8, action), (0.1, slip_actions[0]), (0.1, slip_actions[1])]

        for prob, dir_action in directions:
            delta = self.action_deltas[dir_action]
            next_state = self._move(state, delta)
            # Reward is evaluated at the outcome state, not the intended state —
            # the agent gets penalized for landing on an obstacle even when it slipped there
            reward = self.get_reward(state, action, next_state)
            done = self.is_terminal(next_state)
            transitions.append((prob, next_state, reward, done))

        return transitions

    def _move(self, state, delta):
        """Handle movement (stay in place on wall collision)"""
        new_row = state[0] + delta[0]
        new_col = state[1] + delta[1]

        # Check grid boundaries
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            return (new_row, new_col)
        else:
            return state  # Hit wall
```

### 8.2 Value Function Visualization

```python
def visualize_values(grid, V):
    """Visualize value function"""
    print("\n=== State Values ===")
    print("-" * 40)

    for i in range(grid.size):
        row_str = "|"
        for j in range(grid.size):
            state = (i, j)
            value = V.get(state, 0)

            if state == grid.goal:
                row_str += f"  G   |"
            elif state == grid.obstacle:
                row_str += f"  X   |"
            else:
                row_str += f"{value:6.2f}|"
        print(row_str)
        print("-" * 40)


def visualize_policy(grid, policy):
    """Visualize policy"""
    arrows = {'up': '^', 'down': 'v', 'left': '<', 'right': '>'}

    print("\n=== Optimal Policy ===")
    print("-" * 25)

    for i in range(grid.size):
        row_str = "|"
        for j in range(grid.size):
            state = (i, j)

            if state == grid.goal:
                row_str += "  G  |"
            elif state == grid.obstacle:
                row_str += "  X  |"
            elif state in policy:
                row_str += f"  {arrows[policy[state]]}  |"
            else:
                row_str += "  ?  |"
        print(row_str)
        print("-" * 25)
```

---

## 9. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| Markov Property | Future depends only on current state |
| MDP | Decision problem defined by (S, A, P, R, γ) |
| Policy π | Strategy for selecting actions in states |
| V(s) | Long-term expected value of state |
| Q(s,a) | Expected value of state-action pair |

### Bellman Equation Summary

| Equation | Formula |
|----------|---------|
| Expectation (V) | $V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R + \gamma V^\pi(s')]$ |
| Expectation (Q) | $Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$ |
| Optimality (V) | $V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R + \gamma V^*(s')]$ |
| Optimality (Q) | $Q^*(s,a) = \sum_{s'} P(s'|s,a)[R + \gamma \max_{a'} Q^*(s',a')]$ |

---

## 10. Practice Problems

1. **Markov Property**: Which of the following satisfies the Markov property?
   - (a) Today's stock price determines tomorrow's price
   - (b) Chess board state
   - (c) Determining appropriate response using only current sentence in a conversation

2. **MDP Definition**: Define the MDP elements for a 3x3 grid world.

3. **Bellman Equation**: Given discount factor γ=0.9, immediate reward r=1, V(s')=5, what is V(s)?

4. **Optimal Policy**: Given Q*(s, left)=3, Q*(s, right)=5, what is the optimal action?

---

## Next Steps

In the next lesson **03_Dynamic_Programming.md**, we will learn **dynamic programming** methods (policy iteration, value iteration) for solving MDPs.

---

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 3
- David Silver's RL Course, Lecture 2: Markov Decision Processes
- [MDP Wikipedia](https://en.wikipedia.org/wiki/Markov_decision_process)
