[Previous: Curriculum Learning](./15_Curriculum_Learning.md)

---

# 16. Hierarchical Reinforcement Learning

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the motivation for hierarchical decomposition in RL and temporal abstraction
2. Implement the Options framework for semi-Markov decision processes
3. Describe feudal networks and their manager-worker architecture
4. Build goal-conditioned policies with Hindsight Action Relabeling (HAR)
5. Compare HIRO, Option-Critic, and feudal approaches for continuous control

---

## Table of Contents

1. [Why Hierarchical RL?](#1-why-hierarchical-rl)
2. [The Options Framework](#2-the-options-framework)
3. [Option-Critic Architecture](#3-option-critic-architecture)
4. [Feudal Networks (FeUdal)](#4-feudal-networks-feudal)
5. [HIRO: Goal-Conditioned Hierarchies](#5-hiro-goal-conditioned-hierarchies)
6. [HAM and Other Approaches](#6-ham-and-other-approaches)
7. [Practical Considerations](#7-practical-considerations)
8. [Exercises](#8-exercises)

---

## 1. Why Hierarchical RL?

### 1.1 The Curse of Long Horizons

Standard RL struggles with tasks requiring hundreds or thousands of sequential decisions. The credit assignment problem becomes severe — which of 1000 actions led to the final reward?

```
Flat RL: "Make a sandwich"

  a₁  a₂  a₃  a₄  ...  a₅₀₀  a₅₀₁  ...  a₁₀₀₀  → reward
  ↑    ↑    ↑    ↑        ↑      ↑          ↑
  Which action caused the reward? Credit assignment over 1000 steps.

Hierarchical RL: "Make a sandwich"

  High level:  "Get bread" → "Get filling" → "Assemble" → reward
  Low level:   5-20 actions each (walk, open, grab, ...)

  Credit assignment: 3 high-level decisions + 5-20 low-level per sub-task
  Much easier to learn!
```

### 1.2 Temporal Abstraction

Hierarchical RL introduces **temporal abstraction**: actions that extend over multiple time steps. A "go to kitchen" action might take 50 primitive steps, but the high-level policy sees it as a single decision.

```
                    Time →
Primitive actions:  ←→←→←→←→←→←→←→←→←→←→←→←→←→←→
                    ↕  Individual motor commands (Δt)

Options/Skills:     ←────────→←──────→←─────────→
                    "Go to     "Pick    "Return
                     kitchen"   up cup"  to desk"

High-level goals:   ←──────────────────→←────────→
                    "Get coffee"         "Drink"

Each level operates at a different temporal scale.
```

### 1.3 Benefits of Hierarchy

| Benefit | Explanation |
|---------|-------------|
| **Faster credit assignment** | High-level decisions are few → easier to identify good strategies |
| **Transfer learning** | Skills (options) learned in one task can be reused in others |
| **Exploration** | Temporally extended actions explore more efficiently than random primitives |
| **Interpretability** | "Go to door → open door → exit" is more understandable than 500 motor commands |
| **State abstraction** | High-level policies can operate on compressed state representations |

---

## 2. The Options Framework

### 2.1 Formal Definition

An **option** o = (I, π, β) consists of:
- **I ⊆ S**: Initiation set (states where the option can start)
- **π: S → A**: Intra-option policy (what primitive actions to take)
- **β: S → [0,1]**: Termination function (probability of ending at each state)

```
Standard MDP:          Semi-MDP with options:

  s₀ --a₁--> s₁       s₀ --option₁--> s₃     (option took 3 steps)
  s₁ --a₂--> s₂            accumulated reward = r₁ + γr₂ + γ²r₃
  s₂ --a₃--> s₃
                       s₃ --option₂--> s₅     (option took 2 steps)
  3 decisions               accumulated reward = r₄ + γr₅

                       2 decisions, same outcome
```

### 2.2 Implementation

```python
import numpy as np
from collections import defaultdict


class Option:
    """A temporally extended action (option) in the Options framework.

    An option encapsulates a sub-policy that runs for multiple
    time steps until its termination condition is met.
    """

    def __init__(self, name, initiation_set, policy_fn, termination_fn):
        self.name = name
        self.initiation_set = initiation_set  # set of valid start states
        self.policy_fn = policy_fn            # state → action
        self.termination_fn = termination_fn  # state → P(terminate)

    def can_initiate(self, state):
        """Check if this option can start in the given state."""
        return state in self.initiation_set

    def get_action(self, state):
        """Get the primitive action prescribed by this option."""
        return self.policy_fn(state)

    def should_terminate(self, state):
        """Check if this option should terminate."""
        return np.random.random() < self.termination_fn(state)


class OptionAgent:
    """Agent that learns over options using SMDP Q-learning.

    The policy-over-options selects which option to execute,
    then the option runs until termination.
    """

    def __init__(self, options, states, gamma=0.99, lr=0.1, epsilon=0.1):
        self.options = options
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        # Q-values for state-option pairs
        self.Q = defaultdict(lambda: {o.name: 0.0 for o in options})

    def select_option(self, state):
        """Epsilon-greedy option selection."""
        available = [o for o in self.options if o.can_initiate(state)]
        if not available:
            return None

        if np.random.random() < self.epsilon:
            return np.random.choice(available)

        # Greedy: pick option with highest Q-value
        return max(available, key=lambda o: self.Q[state][o.name])

    def execute_option(self, env, option, state):
        """Execute an option until termination, collecting rewards.

        Returns (final_state, cumulative_discounted_reward, steps).
        """
        total_reward = 0.0
        discount = 1.0
        steps = 0

        while True:
            action = option.get_action(state)
            next_state, reward, done = env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            steps += 1

            if done or option.should_terminate(next_state):
                return next_state, total_reward, steps, done

            state = next_state

    def update(self, state, option_name, reward, next_state, steps, done):
        """SMDP Q-learning update for options.

        Uses multi-step discounting since options span multiple steps.
        """
        if done:
            target = reward
        else:
            # Best option value at next state
            best_q = max(self.Q[next_state].values())
            # Discount by γ^steps (option duration)
            target = reward + (self.gamma ** steps) * best_q

        old_q = self.Q[state][option_name]
        self.Q[state][option_name] = old_q + self.lr * (target - old_q)
```

### 2.3 Designing Options

Good options typically correspond to natural sub-goals:

```
Navigation task:                Robot manipulation:
  • "Go to room A"               • "Reach object"
  • "Go to room B"               • "Grasp object"
  • "Open door"                   • "Lift object"
  • "Pick up key"                 • "Place object"

Each option has a clear termination condition and
can be reused across different high-level tasks.
```

---

## 3. Option-Critic Architecture

### 3.1 Learning Options End-to-End

The Options framework requires manually designed options. The **Option-Critic** (Bacon et al., 2017) learns options, their policies, and termination conditions simultaneously through gradient descent.

```
┌──────────────────────────────────────────────┐
│              Option-Critic                    │
│                                              │
│  State → Feature Extractor → shared features │
│                     │                        │
│         ┌───────────┼───────────┐            │
│         ▼           ▼           ▼            │
│    ┌─────────┐ ┌─────────┐ ┌──────────┐     │
│    │Intra-opt│ │Terminat.│ │Policy    │     │
│    │policies │ │functions│ │over opts │     │
│    │π(a|s,o) │ │β(s,o)   │ │Ω(o|s)   │     │
│    └─────────┘ └─────────┘ └──────────┘     │
│                                              │
│  All three are learned via policy gradient   │
└──────────────────────────────────────────────┘
```

### 3.2 Key Gradients

```python
class OptionCritic:
    """Simplified Option-Critic architecture.

    Learns intra-option policies, termination functions,
    and the policy over options simultaneously.
    """

    def __init__(self, state_dim, action_dim, num_options,
                 hidden_dim=64, lr=0.001):
        self.num_options = num_options
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Intra-option policies: π_o(a|s) for each option o
        # Each option has its own action distribution
        self.option_policies = self._build_policies(
            state_dim, action_dim, num_options, hidden_dim
        )

        # Termination functions: β_o(s) for each option o
        # Sigmoid output ∈ [0,1] = probability of terminating
        self.terminations = self._build_terminations(
            state_dim, num_options, hidden_dim
        )

        # Q(s, o): value of starting option o in state s
        self.q_options = self._build_q(
            state_dim, num_options, hidden_dim
        )

    def _build_policies(self, s_dim, a_dim, n_opts, h_dim):
        """Build intra-option policy networks."""
        # In practice: n_opts separate small networks
        # Each maps state → action probabilities
        policies = []
        for _ in range(n_opts):
            policies.append({
                'W1': np.random.randn(s_dim, h_dim) * 0.1,
                'W2': np.random.randn(h_dim, a_dim) * 0.1,
            })
        return policies

    def _build_terminations(self, s_dim, n_opts, h_dim):
        """Build termination function networks."""
        terms = []
        for _ in range(n_opts):
            terms.append({
                'W1': np.random.randn(s_dim, h_dim) * 0.1,
                'W2': np.random.randn(h_dim, 1) * 0.1,
            })
        return terms

    def _build_q(self, s_dim, n_opts, h_dim):
        """Build Q-value network for options."""
        return {
            'W1': np.random.randn(s_dim, h_dim) * 0.1,
            'W2': np.random.randn(h_dim, n_opts) * 0.1,
        }

    def get_termination_prob(self, state, option_idx):
        """Compute termination probability for an option."""
        t = self.terminations[option_idx]
        h = np.tanh(state @ t['W1'])
        logit = (h @ t['W2']).item()
        return 1.0 / (1.0 + np.exp(-logit))  # sigmoid

    def get_action_probs(self, state, option_idx):
        """Compute action probabilities for an option's policy."""
        p = self.option_policies[option_idx]
        h = np.tanh(state @ p['W1'])
        logits = h @ p['W2']
        # Softmax
        exp_logits = np.exp(logits - logits.max())
        return exp_logits / exp_logits.sum()
```

### 3.3 The Deliberation Cost

A key challenge: options tend to terminate immediately (degenerate to primitive actions) unless there is a reason to persist. The **deliberation cost** adds a small penalty each time an option terminates, encouraging longer options:

```
Without deliberation cost:
  Options terminate every step → equivalent to flat RL
  (No temporal abstraction learned)

With deliberation cost ξ = 0.01:
  Termination penalty discourages unnecessary switching
  Options learn to persist for meaningful durations
  Result: interpretable sub-skills emerge
```

---

## 4. Feudal Networks (FeUdal)

### 4.1 Manager-Worker Architecture

Feudal Networks (Vezhnevets et al., 2017) implement a two-level hierarchy inspired by feudal governance:

```
┌──────────────────────────────────────┐
│             Manager                   │
│  Operates at lower temporal resolution│
│  Sets goals (directions in latent     │
│  space) every c steps                 │
│                                       │
│  g_t = f_manager(s_t)                │
│  (goal = direction vector)            │
└───────────────┬──────────────────────┘
                │ goal g_t
                ▼
┌──────────────────────────────────────┐
│             Worker                    │
│  Operates at every time step          │
│  Conditioned on manager's goal        │
│                                       │
│  a_t = f_worker(s_t, g_t)           │
│  (primitive action)                   │
└──────────────────────────────────────┘
```

### 4.2 Key Design Choices

**Directional goals**: The manager outputs a direction vector g in a learned latent space, not an absolute target. The worker is rewarded for moving in that direction:

```python
class FeudalManager:
    """Manager module that sets sub-goals for the worker.

    Operates at a slower time scale (every c steps) and
    outputs directional goals in a learned latent space.
    """

    def __init__(self, state_dim, goal_dim, c=10):
        self.goal_dim = goal_dim
        self.c = c  # manager acts every c steps
        self.step_count = 0
        self.current_goal = None

        # Manager network weights (simplified)
        self.W_percept = np.random.randn(state_dim, goal_dim) * 0.1
        self.W_goal = np.random.randn(goal_dim, goal_dim) * 0.1

    def get_goal(self, state):
        """Produce a goal vector every c steps."""
        self.step_count += 1
        if self.step_count % self.c == 1 or self.current_goal is None:
            # Compute latent state
            z = np.tanh(state @ self.W_percept)
            # Goal is a direction in latent space
            goal_raw = z @ self.W_goal
            # Normalize to unit vector (direction only)
            norm = np.linalg.norm(goal_raw) + 1e-8
            self.current_goal = goal_raw / norm
        return self.current_goal


class FeudalWorker:
    """Worker module that executes primitive actions toward goals.

    Receives directional goals from the manager and produces
    actions that move the agent in the goal direction.
    """

    def __init__(self, state_dim, goal_dim, action_dim):
        self.W_state = np.random.randn(state_dim, 64) * 0.1
        self.W_goal = np.random.randn(goal_dim, 64) * 0.1
        self.W_action = np.random.randn(64, action_dim) * 0.1

    def get_action(self, state, goal):
        """Produce primitive action conditioned on state and goal."""
        h_state = np.tanh(state @ self.W_state)
        h_goal = np.tanh(goal @ self.W_goal)
        # Element-wise combination of state and goal representations
        combined = h_state * h_goal
        logits = combined @ self.W_action
        # Softmax for discrete actions
        exp_l = np.exp(logits - logits.max())
        probs = exp_l / exp_l.sum()
        return np.random.choice(len(probs), p=probs)

    @staticmethod
    def intrinsic_reward(state_t, state_t_plus_c, goal):
        """Reward worker for moving in the goal direction.

        Cosine similarity between the direction actually traveled
        and the goal direction set by the manager.
        """
        direction = state_t_plus_c - state_t
        d_norm = np.linalg.norm(direction) + 1e-8
        g_norm = np.linalg.norm(goal) + 1e-8
        cosine_sim = np.dot(direction, goal) / (d_norm * g_norm)
        return cosine_sim
```

### 4.3 Training Procedure

- **Manager** is trained to maximize extrinsic (environment) reward
- **Worker** is trained to maximize intrinsic reward (cosine similarity with manager's goal direction)
- The manager learns **what** direction leads to reward; the worker learns **how** to move in that direction

---

## 5. HIRO: Goal-Conditioned Hierarchies

### 5.1 Data-Efficient Hierarchical RL

HIRO (Nachum et al., 2018) combines goal-conditioned policies with off-policy learning:

```
High-level policy μ_hi:
  Every c steps, outputs a sub-goal g ∈ state space
  Goal: state the low-level should reach in c steps

Low-level policy μ_lo:
  At every step, outputs action a conditioned on (state, goal)
  Reward: -||s_t - g||  (distance to sub-goal)
  Trained with standard off-policy RL (e.g., TD3)
```

### 5.2 Goal Relabeling (Off-Policy Correction)

A key innovation: when replaying experience from the buffer, the high-level goal may no longer be valid (the low-level policy has changed). HIRO relabels goals to match what actually happened:

```python
class HIRO:
    """Hierarchical RL with Off-Policy Correction (HIRO).

    Two-level hierarchy: high-level sets goals in state space,
    low-level executes primitive actions to reach goals.
    """

    def __init__(self, state_dim, action_dim, goal_dim, c=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.c = c  # sub-goal horizon

        # High-level: state → goal (every c steps)
        self.high_policy = self._build_policy(state_dim, goal_dim)
        # Low-level: (state, goal) → action (every step)
        self.low_policy = self._build_policy(
            state_dim + goal_dim, action_dim
        )

    def _build_policy(self, input_dim, output_dim):
        """Simple MLP policy (placeholder)."""
        return {
            'W1': np.random.randn(input_dim, 64) * 0.1,
            'W2': np.random.randn(64, output_dim) * 0.1,
        }

    def high_level_goal(self, state):
        """High-level policy outputs a sub-goal."""
        h = np.tanh(state @ self.high_policy['W1'])
        goal = np.tanh(h @ self.high_policy['W2'])
        return goal * 5.0  # scale to reasonable range

    def low_level_action(self, state, goal):
        """Low-level policy outputs primitive action."""
        combined = np.concatenate([state, goal])
        h = np.tanh(combined @ self.low_policy['W1'])
        action = np.tanh(h @ self.low_policy['W2'])
        return action

    @staticmethod
    def low_level_reward(state, goal):
        """Intrinsic reward: negative distance to goal."""
        return -np.linalg.norm(state - goal)

    @staticmethod
    def relabel_goal(states_sequence, original_goal):
        """Off-policy goal relabeling for HIRO.

        Find the goal that best explains the low-level's
        actual behavior, enabling off-policy learning.

        Candidate goals: 10 random goals + the state actually reached.
        Select the one that maximizes low-level action log-probability.
        """
        # In practice: sample candidate goals, evaluate which
        # best explains the observed trajectory
        actual_endpoint = states_sequence[-1]
        # The actually reached state is often the best relabeled goal
        return actual_endpoint
```

### 5.3 HIRO Results

HIRO demonstrated state-of-the-art results on challenging continuous control tasks:

| Task | Flat TD3 | HIRO |
|------|----------|------|
| Ant Navigate | 0.0 | 0.97 |
| Ant Maze (U-shape) | 0.0 | 0.89 |
| Ant Push | 0.0 | 0.73 |
| Ant Fall | 0.0 | 0.58 |

Flat RL completely fails on these long-horizon tasks, while HIRO's hierarchical decomposition enables consistent success.

---

## 6. HAM and Other Approaches

### 6.1 Hierarchy of Abstract Machines (HAM)

HAMs constrain the policy space using finite-state machines:

```
HAM for "Get Coffee":

  ┌──────────┐    door     ┌──────────┐   coffee   ┌──────────┐
  │  Go to   │───reached──►│  Go to   │───reached─►│  Pick    │
  │  door    │             │  machine │             │  up cup  │
  └──────────┘             └──────────┘             └────┬─────┘
       │                                                  │
       │ stuck (timeout)                                  ▼
       └──────────────── retry ──────────────────── Return to desk
```

Each HAM state runs a sub-policy, and transitions are triggered by conditions. The RL agent only needs to learn within each HAM state, dramatically reducing the search space.

### 6.2 MAXQ Decomposition

MAXQ decomposes the value function hierarchically:

```
Q(root, s, a) = V(Navigate, s) + C(root, s, Navigate)
                ↑                  ↑
                Value of           Completion function:
                sub-task           expected reward AFTER
                Navigate           Navigate finishes
```

### 6.3 Comparison of HRL Methods

| Method | Options learned? | Sub-goal space | Off-policy? | Key innovation |
|--------|-----------------|----------------|-------------|----------------|
| Options | Manual | Discrete sub-tasks | Semi-MDP Q | Temporal abstraction |
| Option-Critic | Yes (end-to-end) | Discrete | Yes | Gradient-based option learning |
| FeUdal | Implicit | Latent direction | No (on-policy) | Directional goals |
| HIRO | Via goals | State space | Yes | Off-policy goal relabeling |
| HAM | Manual structure | FSM states | Partial | Constrained policy space |

---

## 7. Practical Considerations

### 7.1 When to Use HRL

| Scenario | Use HRL? | Reason |
|----------|----------|--------|
| Short episodes (< 100 steps) | No | Flat RL is sufficient |
| Long horizons (> 500 steps) | Yes | Credit assignment benefit |
| Clear sub-task structure | Yes | Natural decomposition |
| Dense reward, simple task | No | Over-engineering |
| Sparse reward, navigation | Yes | Sub-goals help exploration |
| Transfer across tasks | Yes | Skills (options) reusable |

### 7.2 Choosing Sub-Goal Frequency (c)

The sub-goal horizon c is a critical hyperparameter:

```
c too small (c=1):
  → Equivalent to flat RL (no temporal abstraction)
  → No benefit from hierarchy

c too large (c=100):
  → Sub-goals too far apart
  → Low-level policy struggles to reach distant goals
  → High-level gets very sparse feedback

Sweet spot (c=10-25):
  → Meaningful temporal abstraction
  → Low-level can reliably reach goals
  → High-level gets regular feedback
```

### 7.3 Debugging HRL

```
Common HRL failure modes:

1. Low-level ignores goals
   Symptom: Worker takes same actions regardless of goal
   Fix: Check intrinsic reward magnitude, ensure goal enters network

2. Options degenerate (all terminate immediately)
   Symptom: Option duration ≈ 1 step
   Fix: Add deliberation cost, initialize β low

3. Mode collapse (only 1-2 options used)
   Symptom: Most options never activated
   Fix: Add diversity bonus, entropy regularization on option selection

4. High-level sets unreachable goals
   Symptom: Low-level always fails, high-level reward always low
   Fix: Constrain goal space, use relative goals, goal relabeling
```

---

## 8. Exercises

### Exercise 1: Options in GridWorld

Implement the Options framework for a 4-room GridWorld:
1. Define 4 options: "go to hallway N/S/E/W" (one per room exit)
2. Train intra-option policies using Q-learning within each room
3. Train a policy-over-options using SMDP Q-learning
4. Compare with flat Q-learning: measure episodes to convergence

### Exercise 2: Option-Critic

Implement a simple Option-Critic:
1. Create a 1D chain environment (20 states, goal at state 19)
2. Implement 4 options with learned policies and termination functions
3. Add deliberation cost and observe its effect on option duration
4. Visualize which states each option covers (option "specialization")

### Exercise 3: Feudal Goal-Setting

Build a manager-worker system:
1. Create a 2D navigation environment with 3 sub-goals
2. Manager outputs direction vectors every 5 steps
3. Worker receives intrinsic reward (cosine similarity with direction)
4. Plot the manager's goal directions at different training stages to show how goals evolve

### Exercise 4: HIRO Goal Relabeling

Implement HIRO-style goal relabeling:
1. Create a 2D continuous environment with sparse terminal reward
2. High-level policy sets position goals every 10 steps
3. Low-level policy receives dense distance-based reward
4. Implement off-policy goal relabeling and compare with on-policy training

### Exercise 5: HRL Architecture Comparison

Compare three HRL approaches on the same task:
1. Implement a navigation task with 2 rooms connected by a door
2. Apply: (a) flat Q-learning, (b) Options with hand-designed skills, (c) simple feudal (manager + worker)
3. Plot learning curves for all three
4. Discuss: when does hierarchy help vs hurt?

---

*End of Lesson 16*
