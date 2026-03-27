[Previous: Goal-Conditioned RL](./20_Goal_Conditioned_RL.md)

---

# 21. Reward Shaping and Intrinsic Motivation

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain potential-based reward shaping and its policy invariance guarantee
2. Implement intrinsic motivation via curiosity-driven exploration
3. Build Random Network Distillation (RND) for exploration bonuses
4. Design reward functions that avoid common pitfalls (reward hacking, sparse rewards)
5. Compare different exploration strategies in hard-exploration environments

---

## Table of Contents

1. [The Reward Design Problem](#1-the-reward-design-problem)
2. [Potential-Based Reward Shaping](#2-potential-based-reward-shaping)
3. [Curiosity-Driven Exploration](#3-curiosity-driven-exploration)
4. [Random Network Distillation (RND)](#4-random-network-distillation-rnd)
5. [Count-Based Exploration](#5-count-based-exploration)
6. [Reward Hacking and Misalignment](#6-reward-hacking-and-misalignment)
7. [Practical Reward Engineering](#7-practical-reward-engineering)
8. [Exercises](#8-exercises)

---

## 1. The Reward Design Problem

### 1.1 Why Reward Design Matters

```
The reward function is the MOST important part of any RL system.

"If the reward function is wrong, you'll get the wrong behavior."

Examples of reward misspecification:
  ❌ Robot vacuum: reward = dirt collected
     Result: Dumps dirt then re-collects it forever

  ❌ Game agent: reward = score
     Result: Exploits glitches instead of playing properly

  ❌ Trading agent: reward = profit
     Result: Takes extreme risk for short-term gain

  ✓ Good reward design requires understanding WHAT you actually want
```

### 1.2 Sparse vs Dense Rewards

```
Sparse reward:
  R = 1 if goal reached, 0 otherwise
  Pro: Easy to specify correctly (less reward hacking)
  Con: Agent may never see positive reward (exploration nightmare)

Dense reward:
  R = -||state - goal|| at every step
  Pro: Constant learning signal
  Con: Can lead to unintended behaviors (local optima)

Shaped reward (best of both):
  R = sparse_reward + shaping_bonus
  Pro: Learning signal + correct optimal policy
  Con: Must design shaping carefully
```

---

## 2. Potential-Based Reward Shaping

### 2.1 The Key Theorem

Potential-based reward shaping (PBRS) adds a bonus F(s, s') to the reward that provably does NOT change the optimal policy:

```
Shaping function:
  F(s, s') = γ · Φ(s') - Φ(s)

Where Φ: S -> R is a "potential function" (any function of state).

Theorem (Ng, Harada, & Russell 1999):
  If F(s,s') = γΦ(s') - Φ(s), then the optimal policy under
  R + F is the same as the optimal policy under R.

  This is the ONLY form of additive shaping that preserves optimality!
```

### 2.2 Implementation

```python
import numpy as np


class PotentialBasedShaping:
    """Potential-based reward shaping that preserves optimal policy."""

    def __init__(self, potential_fn, gamma=0.99):
        self.potential = potential_fn
        self.gamma = gamma

    def shape(self, state, next_state, base_reward):
        """Add shaping bonus to base reward."""
        phi_s = self.potential(state)
        phi_s_next = self.potential(next_state)

        shaping = self.gamma * phi_s_next - phi_s
        return base_reward + shaping


# Example: Grid world with goal at (9, 9)
def manhattan_potential(state, goal=np.array([9, 9])):
    """Potential = negative Manhattan distance to goal."""
    return -np.abs(state - goal).sum()


def euclidean_potential(state, goal=np.array([9, 9])):
    """Potential = negative Euclidean distance to goal."""
    return -np.linalg.norm(state - goal)


# Usage
shaper = PotentialBasedShaping(manhattan_potential, gamma=0.99)

state = np.array([3, 3])
next_state = np.array([4, 3])  # moved closer to goal
base_reward = 0  # sparse reward (not at goal)

shaped_reward = shaper.shape(state, next_state, base_reward)
print(f"Base reward: {base_reward}")
print(f"Shaped reward: {shaped_reward:.4f}")
# Shaped reward > 0 because we moved closer to goal
```

### 2.3 Why Only Potential-Based Shaping Is Safe

```python
def demonstrate_bad_shaping():
    """Show how non-potential-based shaping changes optimal policy."""

    # Simple MDP: 3 states, 2 actions
    # State 0 -> State 1 (action 0): reward = 1
    # State 0 -> State 2 (action 1): reward = 2
    # Optimal: action 1 (go to state 2 for reward 2)

    # Bad shaping: give +5 bonus for going to state 1
    # Result: action 0 now looks better (1 + 5 = 6 > 2)
    # Optimal policy CHANGED!

    print("Without shaping:")
    print("  Action 0: R = 1  |  Action 1: R = 2  |  Optimal: Action 1")
    print()
    print("With bad shaping (+5 for state 1):")
    print("  Action 0: R = 1 + 5 = 6  |  Action 1: R = 2  |  'Optimal': Action 0")
    print("  WRONG! Optimal policy changed!")
    print()
    print("With potential-based shaping (Φ(s1)=5, Φ(s2)=0, Φ(s0)=0):")
    print("  Action 0: R = 1 + γ·5 - 0 = 5.95")
    print("  Action 1: R = 2 + γ·0 - 0 = 2.0")
    print("  Looks wrong at first, but VALUE FUNCTION adjusts.")
    print("  The optimal POLICY is still action 1 for the long run!")

demonstrate_bad_shaping()
```

---

## 3. Curiosity-Driven Exploration

### 3.1 Intrinsic Motivation

```
Extrinsic reward:  From the environment (task reward)
Intrinsic reward:  Self-generated by the agent (curiosity bonus)

Total reward = R_extrinsic + β · R_intrinsic

Intrinsic motivation sources:
├── Prediction error (curiosity)
│   "I'm surprised by this state -> explore more!"
├── Information gain
│   "This reduces my uncertainty -> explore more!"
├── Novelty (count-based)
│   "I haven't been here often -> explore more!"
└── Empowerment
    "I have more control here -> explore more!"
```

### 3.2 ICM: Intrinsic Curiosity Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ICM(nn.Module):
    """
    Intrinsic Curiosity Module (Pathak et al., 2017).

    Key idea: Curiosity = prediction error in a learned feature space.
    Uses forward model (predict next features) and inverse model
    (predict action from features) jointly.
    """

    def __init__(self, state_dim, action_dim, feature_dim=64, beta=0.2):
        super().__init__()
        self.beta = beta

        # Feature encoder: state -> features
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

        # Forward model: (φ(s), a) -> predicted φ(s')
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

        # Inverse model: (φ(s), φ(s')) -> predicted action
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state, next_state, action):
        """Compute intrinsic reward and auxiliary losses."""
        # Encode states
        phi_s = self.encoder(state)
        phi_s_next = self.encoder(next_state)

        # Forward model: predict next features
        forward_input = torch.cat([phi_s, action], dim=-1)
        phi_s_next_pred = self.forward_model(forward_input)

        # Intrinsic reward = forward prediction error
        intrinsic_reward = 0.5 * ((phi_s_next_pred - phi_s_next.detach()) ** 2).sum(dim=-1)

        # Inverse model: predict action
        inverse_input = torch.cat([phi_s, phi_s_next], dim=-1)
        action_pred = self.inverse_model(inverse_input)

        # Losses
        forward_loss = F.mse_loss(phi_s_next_pred, phi_s_next.detach())
        inverse_loss = F.mse_loss(action_pred, action)

        # Combined ICM loss
        icm_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

        return intrinsic_reward, icm_loss

    def get_intrinsic_reward(self, state, next_state, action):
        """Get intrinsic reward for a transition."""
        with torch.no_grad():
            phi_s = self.encoder(state)
            phi_s_next = self.encoder(next_state)

            forward_input = torch.cat([phi_s, action], dim=-1)
            phi_s_next_pred = self.forward_model(forward_input)

            reward = 0.5 * ((phi_s_next_pred - phi_s_next) ** 2).sum(dim=-1)
        return reward
```

### 3.3 Why Prediction Error in Feature Space?

```
Why not predict raw pixels/states?

Problem: "Noisy TV" effect
  If next state has irreducible randomness (e.g., TV static),
  forward model can NEVER predict it accurately.
  -> Perpetual "curiosity" about noise
  -> Agent just stares at noisy TV forever!

Solution: Learn features that are:
  1. Relevant to agent's actions (inverse model ensures this)
  2. Predictable from (state, action)
  3. Ignore irrelevant noise

The inverse model acts as a filter:
  If φ(s) and φ(s') don't help predict the action taken,
  those features are useless -> encoder learns to ignore them.
```

---

## 4. Random Network Distillation (RND)

### 4.1 RND Concept

RND uses a simpler curiosity signal: the prediction error of a fixed random network.

```
RND Architecture:

  Target network f: s -> R^d    (FIXED random weights, never updated)
  Predictor network f̂: s -> R^d  (trained to match target)

  Intrinsic reward = ||f(s) - f̂(s)||²

  Intuition:
  - For frequently visited states: predictor has seen many examples
    -> f̂(s) ≈ f(s) -> low reward
  - For novel states: predictor hasn't been trained on these
    -> f̂(s) ≠ f(s) -> high reward (explore!)
```

### 4.2 RND Implementation

```python
class RNDModule(nn.Module):
    """Random Network Distillation for exploration."""

    def __init__(self, state_dim, feature_dim=128):
        super().__init__()

        # Target: fixed random network (never updated)
        self.target = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

        # Predictor: trained to match target
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)

        # Running statistics for reward normalization
        self.reward_running_mean = 0
        self.reward_running_var = 1
        self.reward_count = 0

    def compute_intrinsic_reward(self, state):
        """Compute exploration bonus."""
        with torch.no_grad():
            target_features = self.target(state)
            predicted_features = self.predictor(state)

            reward = ((target_features - predicted_features) ** 2).sum(dim=-1)
        return reward

    def update(self, states):
        """Train predictor to match target."""
        target_features = self.target(states).detach()
        predicted_features = self.predictor(states)

        loss = F.mse_loss(predicted_features, target_features)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def normalize_reward(self, reward):
        """Normalize intrinsic reward using running statistics."""
        self.reward_count += 1
        delta = reward - self.reward_running_mean
        self.reward_running_mean += delta / self.reward_count
        self.reward_running_var += delta * (reward - self.reward_running_mean)

        std = np.sqrt(self.reward_running_var / max(self.reward_count, 1)) + 1e-8
        return (reward - self.reward_running_mean) / std
```

### 4.3 RND Agent Training

```python
class RNDAgent:
    """PPO agent with RND exploration bonus."""

    def __init__(self, state_dim, action_dim, intrinsic_coef=1.0,
                 extrinsic_coef=2.0, gamma_i=0.99, gamma_e=0.999):
        self.intrinsic_coef = intrinsic_coef
        self.extrinsic_coef = extrinsic_coef
        self.gamma_i = gamma_i  # discount for intrinsic reward
        self.gamma_e = gamma_e  # discount for extrinsic reward

        self.rnd = RNDModule(state_dim)

        # Separate value heads for intrinsic and extrinsic rewards
        self.value_ext = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.value_int = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Shared policy
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def compute_combined_reward(self, state, extrinsic_reward):
        """Combine extrinsic and intrinsic rewards."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        intrinsic = self.rnd.compute_intrinsic_reward(state_t).item()
        intrinsic = self.rnd.normalize_reward(intrinsic)

        combined = (self.extrinsic_coef * extrinsic_reward +
                    self.intrinsic_coef * intrinsic)
        return combined, intrinsic
```

### 4.4 RND on Montezuma's Revenge

```
Montezuma's Revenge: The Exploration Benchmark

Without exploration bonus:
  DQN score: ~0 (never finds first key)
  PPO score: ~0

With RND:
  Score: ~10,000+ (explores multiple rooms, finds keys)
  First algorithm to achieve significant progress without demonstrations

Why is this game hard?
  - Very sparse rewards (hundreds of actions between rewards)
  - Requires specific sequence of actions (get key -> open door)
  - Random exploration almost never reaches rewards

RND solves it because:
  - Novel states (new rooms) give high intrinsic reward
  - Agent is "curious" about unexplored areas
  - Gradually explores the entire map
```

---

## 5. Count-Based Exploration

### 5.1 Classic Count-Based Methods

```python
class CountBasedExploration:
    """Exploration bonus based on state visitation counts."""

    def __init__(self, bonus_coef=1.0):
        self.counts = {}
        self.bonus_coef = bonus_coef

    def discretize_state(self, state, bins=20):
        """Discretize continuous state for counting."""
        return tuple(np.digitize(state, np.linspace(-5, 5, bins)))

    def get_bonus(self, state):
        """Exploration bonus = β / sqrt(N(s))."""
        key = self.discretize_state(state)
        count = self.counts.get(key, 0)
        self.counts[key] = count + 1
        return self.bonus_coef / np.sqrt(count + 1)
```

### 5.2 Pseudo-Count Methods for Large State Spaces

```python
class HashCountExploration:
    """SimHash-based pseudo-counts for high-dimensional states."""

    def __init__(self, state_dim, n_hash_bits=32, bonus_coef=0.5):
        self.bonus_coef = bonus_coef
        self.n_hash_bits = n_hash_bits

        # Random projection for SimHash
        self.projection = np.random.randn(n_hash_bits, state_dim)
        self.counts = {}

    def hash_state(self, state):
        """Locality-sensitive hash of state."""
        projection = self.projection @ state
        return tuple((projection > 0).astype(int))

    def get_bonus(self, state):
        h = self.hash_state(state)
        count = self.counts.get(h, 0)
        self.counts[h] = count + 1
        return self.bonus_coef / np.sqrt(count + 1)
```

---

## 6. Reward Hacking and Misalignment

### 6.1 Examples of Reward Hacking

```
Famous reward hacking examples:

1. Boat Racing Game:
   Reward: points for hitting checkpoints
   Hack: Agent drives in circles hitting same checkpoint repeatedly
   Never finishes the race, but gets infinite points

2. Block Stacking:
   Reward: height of tallest tower
   Hack: Agent flips the table, making the floor the "tower"

3. Cleaning Robot:
   Reward: -1 for each piece of visible dirt
   Hack: Covers its camera sensor (no visible dirt = max reward)

4. CoastRunners Game:
   Reward: high score
   Expected: finish the race
   Hack: Found a loop of turbo pickups, catches fire repeatedly

Root cause: Reward function doesn't fully capture intended behavior
```

### 6.2 Reward Design Principles

```
Principles for good reward design:

1. SPECIFY WHAT, NOT HOW
   Bad:  R = sum of rewards for each subtask step
   Good: R = 1 if task complete, 0 otherwise + PBRS for speed

2. USE POTENTIAL-BASED SHAPING
   Preserve optimal policy while adding learning signal

3. AVOID REWARD MAGNITUDE ISSUES
   Normalize rewards to reasonable range [-1, 1] or [0, 1]
   Large magnitude differences confuse learning

4. TEST FOR DEGENERATE SOLUTIONS
   Ask: "What's the laziest way to maximize this reward?"
   If the answer is not your intended behavior, redesign

5. COMBINE SPARSE + SHAPED
   Sparse for correctness, shaped for speed
   R_total = R_sparse + clip(R_shaped, -max_shape, max_shape)
```

---

## 7. Practical Reward Engineering

### 7.1 Reward Design Template

```python
class RewardDesigner:
    """Template for composing reward functions safely."""

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.components = []

    def add_sparse_goal(self, goal_fn, reward=1.0):
        """Add sparse reward for goal achievement."""
        self.components.append(('sparse', goal_fn, reward))

    def add_potential_shaping(self, potential_fn, weight=1.0):
        """Add potential-based shaping (policy-invariant)."""
        self.components.append(('potential', potential_fn, weight))

    def add_penalty(self, penalty_fn, weight=-0.01):
        """Add small penalty (e.g., for time or energy)."""
        self.components.append(('penalty', penalty_fn, weight))

    def compute_reward(self, state, next_state, action, info=None):
        """Compute total reward from all components."""
        total = 0

        for comp_type, fn, weight in self.components:
            if comp_type == 'sparse':
                total += weight * fn(next_state)
            elif comp_type == 'potential':
                shaping = self.gamma * fn(next_state) - fn(state)
                total += weight * shaping
            elif comp_type == 'penalty':
                total += weight * fn(state, action)

        return total


# Example usage
designer = RewardDesigner(gamma=0.99)

# Sparse goal
designer.add_sparse_goal(
    lambda s: 1.0 if np.linalg.norm(s[:2] - np.array([5, 5])) < 0.1 else 0.0,
    reward=10.0
)

# Distance-based shaping (potential-based, preserves optimal policy)
designer.add_potential_shaping(
    lambda s: -np.linalg.norm(s[:2] - np.array([5, 5])),
    weight=1.0
)

# Small time penalty
designer.add_penalty(
    lambda s, a: 1.0,  # constant penalty per step
    weight=-0.01
)
```

### 7.2 Debugging Reward Functions

```python
def debug_reward_function(env, reward_fn, n_episodes=10, max_steps=200):
    """Visualize reward statistics to catch design issues."""
    all_rewards = []
    episode_returns = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        rewards = []

        for step in range(max_steps):
            action = env.action_space.sample()
            next_state, _, terminated, truncated, info = env.step(action)

            r = reward_fn(state, next_state, action, info)
            rewards.append(r)
            state = next_state

            if terminated or truncated:
                break

        all_rewards.extend(rewards)
        episode_returns.append(sum(rewards))

    rewards_arr = np.array(all_rewards)
    print("Reward diagnostics:")
    print(f"  Mean:   {rewards_arr.mean():.4f}")
    print(f"  Std:    {rewards_arr.std():.4f}")
    print(f"  Min:    {rewards_arr.min():.4f}")
    print(f"  Max:    {rewards_arr.max():.4f}")
    print(f"  % zero: {(rewards_arr == 0).mean()*100:.1f}%")
    print(f"  Episode returns: {np.mean(episode_returns):.2f} "
          f"+/- {np.std(episode_returns):.2f}")

    # Warning flags
    if rewards_arr.std() > 100 * abs(rewards_arr.mean()):
        print("  WARNING: Very high variance relative to mean!")
    if (rewards_arr == 0).mean() > 0.99:
        print("  WARNING: >99% zero rewards - too sparse!")
    if rewards_arr.min() < -100:
        print("  WARNING: Very large negative rewards - check penalties")
```

---

## 8. Exercises

### Exercise 1: Potential-Based Reward Shaping

Implement and validate PBRS:
1. Create a 10x10 grid world with sparse reward at (9,9)
2. Train Q-learning without shaping: measure episodes to convergence
3. Add potential-based shaping using Manhattan distance
4. Show convergence speedup while verifying optimal policy is unchanged
5. Try NON-potential-based shaping and show it changes the optimal policy

### Exercise 2: ICM Curiosity Module

Build and train ICM for exploration:
1. Implement the full ICM (encoder + forward + inverse models)
2. Create a 2D maze with dead ends and sparse reward
3. Compare: epsilon-greedy, ICM curiosity, and RND
4. Visualize exploration heatmaps for each method
5. Show that curiosity-driven exploration finds the goal faster

### Exercise 3: RND Implementation

Implement RND and test on hard exploration:
1. Build the RND module with target and predictor networks
2. Implement reward normalization with running statistics
3. Test on MountainCar (sparse reward only at the top)
4. Plot intrinsic reward magnitude over training
5. Show that intrinsic reward decreases for familiar states

### Exercise 4: Reward Hacking Detection

Create a reward hacking scenario and detect it:
1. Design a simple environment with an exploitable reward function
2. Train an RL agent and observe it finding the exploit
3. Add monitoring to detect the hack (reward vs intended metric)
4. Redesign the reward function to be hack-resistant
5. Document the original hack and your fix

### Exercise 5: Exploration Method Comparison

Comprehensive comparison of exploration methods:
1. Create a challenging maze environment (multiple rooms, sparse reward)
2. Implement: epsilon-greedy, Boltzmann, count-based, ICM, RND
3. Run each for 1M steps, measuring coverage and final performance
4. Create a wall-clock-time comparison (some methods have higher overhead)
5. Recommend which method to use in different scenarios

---

*End of Lesson 21*
