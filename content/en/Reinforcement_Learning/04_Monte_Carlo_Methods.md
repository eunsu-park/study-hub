# 04. Monte Carlo Methods

**Difficulty: ⭐⭐ (Beginner)**

## Learning Objectives
- Understand the basic concepts of Monte Carlo (MC) methods
- Grasp the meaning of Model-Free learning
- Understand the difference between First-visit MC and Every-visit MC
- Implement MC policy evaluation and control algorithms
- Learn the importance of exploration and solutions

---

## 1. What are Monte Carlo Methods?

### 1.1 Overview

**Monte Carlo (MC) methods** estimate value functions from **actual experience (episodes)**. They are **model-free** methods that learn without an environment model.

### 1.2 DP vs MC Comparison

| Feature | Dynamic Programming (DP) | Monte Carlo (MC) |
|---------|-------------------------|------------------|
| Environment Model | Required (must know P, R) | Not required |
| Learning Method | Computation (bootstrapping) | Sampling (experience) |
| Update Timing | Every step possible | After episode termination |
| Applicable | Episodic/Continuous | Episodic only |

### 1.3 Core Idea

$$V(s) \approx \text{Average}(\text{Returns from episodes starting at state } s)$$

```
Episode 1: S₀ → S₁ → S₂ → ... → Terminal, G₁ = 10
Episode 2: S₀ → S₃ → S₁ → ... → Terminal, G₂ = 8
Episode 3: S₀ → S₂ → S₁ → ... → Terminal, G₃ = 12

V(S₀) = (10 + 8 + 12) / 3 = 10
```

---

## 2. Return Calculation

### 2.1 Calculating Returns from Episodes

```python
def calculate_returns(episode, gamma=0.99):
    """
    Calculate return for each state in episode

    Args:
        episode: List of (state, action, reward) tuples
        gamma: Discount factor

    Returns:
        returns: Dictionary {state: return}
    """
    G = 0  # Initialize return
    returns = {}

    # Why reverse iteration: exploits the recurrence G_t = r_t + gamma * G_{t+1},
    # computing all returns in a single O(n) pass instead of O(n^2) forward re-summation
    for t in range(len(episode) - 1, -1, -1):
        state, action, reward = episode[t]
        G = reward + gamma * G  # Discounted return — each step reuses the already-computed future return
        returns[t] = (state, G)

    return returns


# Example
episode = [
    ('s0', 'right', -1),
    ('s1', 'right', -1),
    ('s2', 'right', 10),  # Reached goal
]

returns = calculate_returns(episode, gamma=0.9)
for t, (state, G) in returns.items():
    print(f"t={t}: {state}, G={G:.2f}")

# Output:
# t=2: s2, G=10.00
# t=1: s1, G=8.00  (= -1 + 0.9 * 10)
# t=0: s0, G=6.20  (= -1 + 0.9 * 8)
```

---

## 3. MC Policy Evaluation

### 3.1 First-visit MC vs Every-visit MC

| Method | Description |
|--------|-------------|
| First-visit MC | Record return **only on first visit** to state s in episode |
| Every-visit MC | Record return **on every visit** to state s in episode |

```
Episode: S₀ → S₁ → S₂ → S₁ → S₃ (terminal)
                     ↑      ↑
              First visit   Second visit

First-visit: Count only first visit to S₁
Every-visit: Count all visits to S₁
```

### 3.2 First-visit MC Implementation

```python
import numpy as np
from collections import defaultdict

def first_visit_mc_prediction(env, policy, n_episodes=10000, gamma=0.99):
    """
    First-visit MC Policy Evaluation

    Args:
        env: Gymnasium environment
        policy: Policy function policy(state) -> action
        n_episodes: Number of episodes
        gamma: Discount factor

    Returns:
        V: State value function
    """
    # Why incremental mean: storing sum and count avoids keeping every return in memory,
    # giving O(1) memory per state while converging to the true expected value
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)

    for episode_num in range(n_episodes):
        # Generate episode
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        # Why first-visit: only counting the first occurrence per state produces an
        # unbiased estimator of V(s). Every-visit MC is also consistent but has different
        # bias/variance trade-offs — first-visit is simpler to analyze theoretically.
        visited = set()
        G = 0

        # Why reverse iteration: G_t = r_t + gamma * G_{t+1} — one backward pass
        # computes all returns in O(n) instead of re-summing from each timestep
        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]

            G = gamma * G + reward_t

            # Why first-visit check: skipping later visits avoids double-counting returns
            # for the same state within one episode, preserving the unbiased property
            if state_t not in visited:
                visited.add(state_t)
                returns_sum[state_t] += G
                returns_count[state_t] += 1
                V[state_t] = returns_sum[state_t] / returns_count[state_t]

        if (episode_num + 1) % 1000 == 0:
            print(f"Episode {episode_num + 1}/{n_episodes}")

    return dict(V)


# Usage example
import gymnasium as gym

env = gym.make('Blackjack-v1')

def random_policy(state):
    """Random policy"""
    return env.action_space.sample()

V = first_visit_mc_prediction(env, random_policy, n_episodes=50000)
print(f"\nNumber of estimated states: {len(V)}")
```

### 3.3 Every-visit MC Implementation

```python
def every_visit_mc_prediction(env, policy, n_episodes=10000, gamma=0.99):
    """
    Every-visit MC Policy Evaluation
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)

    for episode_num in range(n_episodes):
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        G = 0

        # Why every-visit differs: updating on all visits trades unbiasedness for
        # lower variance — more data points per episode, but the samples from the
        # same episode are correlated. Converges to the same value asymptotically.
        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]

            G = gamma * G + reward_t

            # No first-visit filter: every occurrence contributes a return sample
            returns_sum[state_t] += G
            returns_count[state_t] += 1
            V[state_t] = returns_sum[state_t] / returns_count[state_t]

    return dict(V)
```

---

## 4. MC Policy Control

### 4.1 Exploring Starts (ES)

Assumes episodes can start from all state-action pairs.

```python
def mc_exploring_starts(env, n_episodes=100000, gamma=0.99):
    """
    MC with Exploring Starts

    Assumes all (s, a) pairs can be starting points
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize Q function and visit counts
    Q = defaultdict(lambda: np.zeros(n_actions))
    returns_sum = defaultdict(lambda: np.zeros(n_actions))
    returns_count = defaultdict(lambda: np.zeros(n_actions))

    # Policy (greedy)
    policy = defaultdict(lambda: 0)

    for episode_num in range(n_episodes):
        # Why exploring starts: a greedy policy can only improve if Q values are
        # accurate for all (s, a) pairs, which requires visiting them all.
        # Random initialization guarantees every pair has nonzero visit probability.
        start_state = env.observation_space.sample()
        start_action = env.action_space.sample()

        # Generate episode
        episode = []
        state = start_state
        action = start_action

        # First step
        env.reset()
        env.unwrapped.s = state  # Force set state (depends on environment)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated

        # Rest of episode
        while not done:
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        # Calculate returns and update Q
        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            if (state_t, action_t) not in visited:
                visited.add((state_t, action_t))
                returns_sum[state_t][action_t] += G
                returns_count[state_t][action_t] += 1
                Q[state_t][action_t] = (returns_sum[state_t][action_t] /
                                        returns_count[state_t][action_t])

                # Why greedy improvement here: the policy improvement theorem guarantees
                # that acting greedily w.r.t. Q is at least as good as the old policy
                policy[state_t] = np.argmax(Q[state_t])

    return Q, policy
```

### 4.2 ε-greedy Policy

Exploring Starts is unrealistic, so use **ε-greedy** policy.

$$\pi(a|s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s, a') \\
\frac{\epsilon}{|A|} & \text{otherwise}
\end{cases}$$

```python
def epsilon_greedy_policy(Q, state, n_actions, epsilon=0.1):
    """
    ε-greedy action selection

    Args:
        Q: Action value function
        state: Current state
        n_actions: Number of actions
        epsilon: Exploration probability

    Returns:
        action: Selected action
    """
    # Why epsilon-greedy: balances exploration vs exploitation — with probability
    # epsilon we try a random action to discover potentially better strategies,
    # otherwise we exploit the best known action. Pure greedy (epsilon=0) risks
    # getting stuck in a suboptimal policy by never discovering better alternatives.
    if np.random.random() < epsilon:
        # Explore: random action to gather information about unvisited (s, a) pairs
        return np.random.randint(n_actions)
    else:
        # Exploit: choose the action with highest estimated Q value
        return np.argmax(Q[state])
```

### 4.3 On-policy MC Control

```python
def mc_on_policy_control(env, n_episodes=100000, gamma=0.99,
                         epsilon=0.1, epsilon_decay=0.9999):
    """
    On-policy MC Control (ε-greedy)

    Args:
        env: Gymnasium environment
        n_episodes: Number of episodes
        gamma: Discount factor
        epsilon: Exploration rate
        epsilon_decay: Epsilon decay rate

    Returns:
        Q: Action value function
        policy: Learned policy
    """
    n_actions = env.action_space.n

    Q = defaultdict(lambda: np.zeros(n_actions))
    returns_sum = defaultdict(lambda: np.zeros(n_actions))
    returns_count = defaultdict(lambda: np.zeros(n_actions))

    episode_rewards = []

    for episode_num in range(n_episodes):
        episode = []
        state, _ = env.reset()
        done = False
        total_reward = 0

        # Generate episode with ε-greedy policy
        while not done:
            action = epsilon_greedy_policy(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode.append((state, action, reward))
            total_reward += reward

            state = next_state
            done = terminated or truncated

        episode_rewards.append(total_reward)

        # Update Q (First-visit)
        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            if (state_t, action_t) not in visited:
                visited.add((state_t, action_t))
                returns_sum[state_t][action_t] += G
                returns_count[state_t][action_t] += 1
                Q[state_t][action_t] = (returns_sum[state_t][action_t] /
                                        returns_count[state_t][action_t])

        # Why decay epsilon: early training needs heavy exploration (high epsilon) to
        # build accurate Q estimates; later we shift toward exploitation. The floor of
        # 0.01 prevents complete lock-in to a potentially suboptimal policy — without
        # this minimum, the agent could never escape a local optimum.
        epsilon = max(0.01, epsilon * epsilon_decay)

        if (episode_num + 1) % 10000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            print(f"Episode {episode_num + 1}: avg_reward = {avg_reward:.3f}, "
                  f"epsilon = {epsilon:.4f}")

    # Why greedy at inference: during deployment there is no need to explore —
    # we simply pick the best learned action to maximize expected return
    policy = {}
    for state in Q:
        policy[state] = np.argmax(Q[state])

    return dict(Q), policy, episode_rewards
```

---

## 5. Off-policy MC (Importance Sampling)

### 5.1 What is Off-policy Learning?

- **On-policy**: Behavior policy = Target policy (explore and learn with same policy)
- **Off-policy**: Behavior policy ≠ Target policy (explore and learn with different policies)

```
Behavior Policy b: For data collection
Target Policy π: Optimal policy to learn
```

### 5.2 Importance Sampling

$$\mathbb{E}_b[X] = \sum_x x \cdot b(x) = \sum_x x \cdot \frac{\pi(x)}{b(x)} \cdot b(x) = \mathbb{E}_b\left[\frac{\pi(X)}{b(X)} X\right]$$

Importance sampling ratio:
$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k | S_k)}{b(A_k | S_k)}$$

```python
def importance_sampling_ratio(episode, target_policy, behavior_policy, t):
    """
    Calculate importance sampling ratio

    ρ = π(a₀|s₀)/b(a₀|s₀) × π(a₁|s₁)/b(a₁|s₁) × ...
    """
    # Why importance sampling ratio: data was collected under behavior policy b,
    # but we want to evaluate target policy pi. The ratio pi/b re-weights each
    # transition to correct for the distribution mismatch between the two policies.
    rho = 1.0

    for k in range(t, len(episode)):
        state, action, _ = episode[k]
        pi_prob = target_policy(state, action)  # π(a|s)
        b_prob = behavior_policy(state, action)  # b(a|s)

        if b_prob == 0:
            return 0  # Safety check: avoid division by zero when b never takes this action

        rho *= pi_prob / b_prob  # Product of ratios across all remaining steps

    return rho
```

### 5.3 Off-policy MC Implementation

```python
def mc_off_policy_prediction(env, target_policy, behavior_policy,
                              n_episodes=100000, gamma=0.99):
    """
    Off-policy MC Policy Evaluation (Weighted Importance Sampling)

    Args:
        target_policy: Target policy (deterministic) - π(s) -> a
        behavior_policy: Behavior policy - b(s) -> a (ε-greedy, etc.)
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))  # Weight sum

    for episode_num in range(n_episodes):
        # Generate episode with behavior policy
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        G = 0
        W = 1.0  # Importance sampling weight

        # Why reverse processing: same O(n) recurrence trick as on-policy MC, plus
        # we can early-terminate when the behavior and target policies diverge
        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            # Why weighted importance sampling: ordinary IS has high variance because
            # the ratio product can explode. Weighted IS (self-normalizing) divides by
            # the cumulative weight C, which dramatically reduces variance at the cost
            # of a small bias that vanishes asymptotically.
            C[state_t][action_t] += W
            Q[state_t][action_t] += W / C[state_t][action_t] * (G - Q[state_t][action_t])

            # Action from target policy
            target_action = target_policy(state_t)

            # Why early break: for a deterministic target policy, if the behavior
            # policy took a different action, pi(a|s) = 0 so the IS ratio becomes 0
            # and all earlier timesteps contribute nothing — no need to continue
            if action_t != target_action:
                break

            # Update importance ratio
            # π(a|s) = 1 (deterministic), b(a|s) = behavior policy probability
            b_prob = behavior_policy_prob(state_t, action_t)  # Need to implement
            W = W * 1.0 / b_prob

    return dict(Q)
```

---

## 6. Blackjack Example

### 6.1 Blackjack Environment

```python
import gymnasium as gym
import numpy as np
from collections import defaultdict

# Blackjack environment
# State: (player sum, dealer open card, usable ace)
# Action: 0 = stick (keep cards), 1 = hit (draw card)

env = gym.make('Blackjack-v1', sab=True)  # sab: Sutton and Barto version

print("State space:", env.observation_space)
print("Action space:", env.action_space)
```

### 6.2 MC Learning in Blackjack

```python
def learn_blackjack(n_episodes=500000, gamma=1.0, epsilon=0.1):
    """MC control in Blackjack"""

    env = gym.make('Blackjack-v1', sab=True)

    Q = defaultdict(lambda: np.zeros(2))
    returns_sum = defaultdict(lambda: np.zeros(2))
    returns_count = defaultdict(lambda: np.zeros(2))

    def get_action(state, Q, epsilon):
        # Why epsilon-greedy here: Blackjack has a small discrete state space, but
        # we still need exploration to discover the correct hit/stick boundary
        if np.random.random() < epsilon:
            return env.action_space.sample()
        return np.argmax(Q[state])

    wins = 0
    for ep in range(n_episodes):
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = get_action(state, Q, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        if episode[-1][2] == 1:  # Win
            wins += 1

        # Update Q
        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            if (state_t, action_t) not in visited:
                visited.add((state_t, action_t))
                returns_sum[state_t][action_t] += G
                returns_count[state_t][action_t] += 1
                Q[state_t][action_t] = (returns_sum[state_t][action_t] /
                                        returns_count[state_t][action_t])

        if (ep + 1) % 50000 == 0:
            win_rate = wins / (ep + 1)
            print(f"Episode {ep + 1}: Win rate = {win_rate:.3f}")

    env.close()
    return Q


def visualize_blackjack_policy(Q):
    """Visualize Blackjack policy"""
    print("\n=== When no usable ace ===")
    print("Dealer card:  A  2  3  4  5  6  7  8  9  10")
    print("-" * 50)

    for player_sum in range(21, 10, -1):
        row = f"Sum {player_sum:2d}:  "
        for dealer in range(1, 11):
            state = (player_sum, dealer, False)
            if state in Q:
                action = np.argmax(Q[state])
                row += "H  " if action == 1 else "S  "
            else:
                row += "?  "
        print(row)

    print("\n=== When usable ace ===")
    print("Dealer card:  A  2  3  4  5  6  7  8  9  10")
    print("-" * 50)

    for player_sum in range(21, 11, -1):
        row = f"Sum {player_sum:2d}:  "
        for dealer in range(1, 11):
            state = (player_sum, dealer, True)
            if state in Q:
                action = np.argmax(Q[state])
                row += "H  " if action == 1 else "S  "
            else:
                row += "?  "
        print(row)


# Run learning
Q = learn_blackjack(n_episodes=500000)
visualize_blackjack_policy(Q)
```

---

## 7. Advantages and Disadvantages of MC Methods

### 7.1 Advantages

| Advantage | Description |
|-----------|-------------|
| Model-free | No environment model needed |
| Unbiased | Use actual returns without bootstrapping |
| Intuitive | Simple average return calculation |
| Episode independence | Parallelizable across episodes |

### 7.2 Disadvantages

| Disadvantage | Description |
|--------------|-------------|
| High variance | Variance of full returns is large |
| Episode termination required | Unsuitable for continuous tasks |
| Slow learning | Must wait until episode end |
| Exploration problem | Cannot learn unvisited states |

### 7.3 DP vs MC Summary

```
                    DP                    MC
              ─────────────          ─────────────
Required Info Environment model      Experience (episodes)

Update        V(s) ← Σ P(s'|s,a)    V(s) ← Average(G)
Method        [R + γV(s')]

Bias          Biased                 Unbiased
              (bootstrapping)        (actual returns)

Variance      Low                    High
              (expectation calc)     (sample variance)

Applicable    Finite MDP             Episodic tasks
```

---

## 8. Summary

### Core Concepts

| Concept | Description |
|---------|-------------|
| Monte Carlo | Learn from experience |
| First-visit | Count only first visit |
| Every-visit | Count all visits |
| On-policy | Behavior policy = Target policy |
| Off-policy | Behavior policy ≠ Target policy |
| Importance Sampling | Weights for distribution correction |

### MC Policy Evaluation Formula

$$V(s) = \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i(s)$$

### MC Q-function Update

$$Q(s, a) \leftarrow Q(s, a) + \frac{1}{N(s,a)} (G - Q(s, a))$$

---

## 9. Practice Problems

1. **First-visit vs Every-visit**: Calculate the difference between the two methods for an episode visiting the same state 3 times.

2. **Exploration Problem**: What problem occurs when ε=0 in ε-greedy?

3. **Importance Sampling**: When can importance ratios diverge if the target policy is deterministic and behavior policy is ε-greedy?

4. **Convergence Speed**: Explain why MC has high variance and solutions.

---

## Next Steps

In the next lesson **05_TD_Learning.md**, we'll learn **TD learning** that combines DP's bootstrapping with MC's sampling.

---

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 5
- David Silver's RL Course, Lecture 4: Model-Free Prediction
- [Gymnasium Blackjack](https://gymnasium.farama.org/environments/toy_text/blackjack/)
