# 05. Temporal Difference Learning

**Difficulty: ⭐⭐⭐ (Intermediate)**

## Learning Objectives
- Understand basic concepts of TD learning and the TD(0) algorithm
- Grasp TD Target and bootstrapping concepts
- Compare differences between TD, MC, and DP
- Understand TD's bias-variance tradeoff
- Learn n-step TD and TD(λ)

---

## 1. What is TD Learning?

### 1.1 Overview

**Temporal Difference (TD) learning** combines DP's **bootstrapping** with MC's **sampling**.

```
     DP: V(s) ← E[R + γV(s')]           (model required, bootstrap)
     MC: V(s) ← Average(G)               (model-free, full return)
     TD: V(s) ← V(s) + α[R + γV(s') - V(s)]  (model-free, bootstrap)
```

### 1.2 TD vs MC vs DP Comparison

| Feature | DP | MC | TD |
|---------|-----|-----|-----|
| Environment Model | Required | Not required | Not required |
| Bootstrapping | O | X | O |
| Sampling | X | O | O |
| Update Timing | Every step | Episode termination | Every step |
| Continuous Tasks | O | X | O |
| Bias | O | X | O |
| Variance | Low | High | Medium |

---

## 2. TD(0) Algorithm

### 2.1 TD Target

TD(0) updates the current state's value using the next state's estimate.

**TD Target**: $R_{t+1} + \gamma V(S_{t+1})$

**TD Error (δ)**: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

**Update Rule**:
$$V(S_t) \leftarrow V(S_t) + \alpha \cdot \delta_t$$
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

### 2.2 TD(0) Implementation

```python
import numpy as np
import gymnasium as gym
from collections import defaultdict

def td0_prediction(env, policy, n_episodes=10000, alpha=0.1, gamma=0.99):
    """
    TD(0) Policy Evaluation

    Args:
        env: Gymnasium environment
        policy: Policy function policy(state) -> action
        n_episodes: Number of episodes
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        V: State value function
    """
    # defaultdict(float) starts every unseen state at 0 — a safe neutral prior
    # that doesn't bias the agent toward or away from unvisited states
    V = defaultdict(float)

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD(0) update
            if done:
                td_target = reward  # Terminal state: V(s') = 0
            else:
                # Bootstrap: blend actual reward with discounted estimate of V(s') —
                # this lets us update immediately without waiting for the episode to end
                td_target = reward + gamma * V[next_state]

            # TD error = how wrong our current estimate was; alpha controls
            # how aggressively we correct it (high alpha = fast but noisy)
            td_error = td_target - V[state]
            V[state] = V[state] + alpha * td_error

            state = next_state

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{n_episodes}")

    return dict(V)


# Usage example
env = gym.make('CliffWalking-v0')

def random_policy(state):
    return env.action_space.sample()

V = td0_prediction(env, random_policy, n_episodes=5000)
print(f"Number of learned states: {len(V)}")
```

### 2.3 TD(0) vs MC Comparison Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def compare_td_mc(env, policy, n_episodes=500, alpha=0.1, gamma=0.99):
    """Compare learning curves of TD(0) and MC"""

    # TD(0)
    V_td = defaultdict(float)
    td_errors = []

    # MC
    V_mc = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    mc_errors = []

    for episode in range(n_episodes):
        # Generate episode
        episode_data = []
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_data.append((state, action, reward, next_state, done))
            state = next_state

        # TD(0) update (online)
        for state, action, reward, next_state, done in episode_data:
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * V_td[next_state]
            V_td[state] += alpha * (td_target - V_td[state])

        # MC update (offline)
        G = 0
        for state, action, reward, _, _ in reversed(episode_data):
            G = reward + gamma * G
            returns_sum[state] += G
            returns_count[state] += 1
            V_mc[state] = returns_sum[state] / returns_count[state]

        # Record estimates for test state (for comparison)
        test_state = episode_data[0][0]
        td_errors.append(V_td[test_state])
        mc_errors.append(V_mc[test_state])

    return td_errors, mc_errors


# Visualize learning curves
# env = gym.make('CliffWalking-v0')
# td_curve, mc_curve = compare_td_mc(env, random_policy)
#
# plt.figure(figsize=(10, 5))
# plt.plot(td_curve, label='TD(0)', alpha=0.7)
# plt.plot(mc_curve, label='MC', alpha=0.7)
# plt.xlabel('Episode')
# plt.ylabel('Value Estimate')
# plt.legend()
# plt.title('TD(0) vs MC Learning Curves')
# plt.show()
```

---

## 3. Bootstrapping

### 3.1 Concept

**Bootstrapping** is updating an estimate using other estimates.

```
MC:  V(s) ← V(s) + α[G - V(s)]
     Use actual return G (no bootstrapping)

TD:  V(s) ← V(s) + α[R + γV(s') - V(s)]
     Use estimate V(s') (bootstrapping)
```

### 3.2 Impact of Bootstrapping

```python
"""
Advantages of bootstrapping:
1. Can learn before episode termination
2. Applicable to continuous tasks
3. Lower variance (use only one-step reward)

Disadvantages of bootstrapping:
1. Introduces bias (propagates if V(s') is inaccurate)
2. Sensitive to initial estimates
3. Convergence guarantee more complex than MC
"""

# Bootstrapping visualization
def visualize_bootstrapping():
    """
    MC: S₀ → S₁ → S₂ → S₃ → Terminal (G = r₁ + γr₂ + γ²r₃ + γ³r₄)
                                       └── Use full return

    TD: S₀ → S₁ → S₂ → ...
        V(S₀) ← R₁ + γV(S₁)
                     └── Use estimate (bootstrap)
    """
    pass
```

---

## 4. Meaning of TD Error

### 4.1 TD Error Analysis

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

- **δ > 0**: Better than expected → Increase V(s)
- **δ < 0**: Worse than expected → Decrease V(s)
- **δ = 0**: Matches expectations → No change

### 4.2 TD Error and Neuroscience

```
Dopamine neuron responses are similar to TD Error!

Unexpected reward → Dopamine increase (δ > 0)
Expected reward received → No dopamine change (δ ≈ 0)
Expected reward not received → Dopamine decrease (δ < 0)

→ TD learning may be similar to brain learning mechanisms
```

---

## 5. Advantages of TD Learning

### 5.1 Random Walk Example

```python
def random_walk_comparison():
    """
    Compare TD and MC on Random Walk

    Environment: A - B - C - D - E - [Terminal]
                  ←              →
    Left terminal: Reward 0
    Right terminal: Reward 1
    """
    import numpy as np

    # States: 0=left terminal, 1-5=A-E, 6=right terminal
    n_states = 7
    true_values = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1])  # True values

    def run_episode():
        """Generate episode"""
        state = 3  # Start at C
        episode = [(state, 0, state)]

        while 0 < state < 6:
            if np.random.random() < 0.5:
                state -= 1  # Left
            else:
                state += 1  # Right

            reward = 1 if state == 6 else 0
            episode.append((state, reward, state))

        return episode

    # TD(0) learning
    V_td = np.full(n_states, 0.5)
    V_td[0] = V_td[6] = 0

    alpha = 0.1
    n_episodes = 100

    for _ in range(n_episodes):
        state = 3
        while 0 < state < 6:
            if np.random.random() < 0.5:
                next_state = state - 1
            else:
                next_state = state + 1

            reward = 1 if next_state == 6 else 0
            V_td[state] += alpha * (reward + V_td[next_state] - V_td[state])
            state = next_state

    print("True Values:", true_values[1:6])
    print("TD Estimates:", V_td[1:6].round(3))
    print("TD RMSE:", np.sqrt(np.mean((V_td[1:6] - true_values[1:6])**2)))

    return V_td


V_td = random_walk_comparison()
```

### 5.2 Advantages of TD Summary

| Advantage | Description |
|-----------|-------------|
| Online learning | Can learn before episode termination |
| Continuous tasks | Applicable to tasks without termination |
| Lower variance | Use only one-step reward |
| Incremental improvement | Can improve policy in real-time |

---

## 6. Batch TD vs Batch MC

### 6.1 Batch Learning

When learning repeatedly from the same dataset, TD and MC converge to different values.

```python
def batch_td_mc_comparison():
    """
    Compare TD and MC in batch learning

    Example data:
    Episode 1: A → B → 0 (reward 0)
    Episode 2: B → 1 (reward 1)

    MC: V(A) = 0, V(B) = 1/2
    TD: V(A) = 3/4 * V(B) = 3/4, V(B) = 1 (A→B so V(A) ≈ V(B))
    """
    # Simple example
    episodes = [
        [('A', 'B', 0), ('B', 'terminal', 0)],  # A → B → Terminal(0)
        [('B', 'terminal', 1)]                   # B → Terminal(1)
    ]

    # Batch MC
    V_mc = {'A': 0, 'B': 0}
    returns = {'A': [], 'B': []}

    for ep in episodes:
        G = 0
        for state, next_state, reward in reversed(ep):
            G = reward + G
            if state != 'terminal':
                returns[state].append(G)

    for state in V_mc:
        if returns[state]:
            V_mc[state] = np.mean(returns[state])

    # Batch TD (iterate)
    V_td = {'A': 0, 'B': 0, 'terminal': 0}
    alpha = 0.1

    for _ in range(100):  # Batch iterations
        for ep in episodes:
            for state, next_state, reward in ep:
                if state != 'terminal':
                    V_td[state] += alpha * (reward + V_td[next_state] - V_td[state])

    print("Batch MC:", V_mc)
    print("Batch TD:", {k: round(v, 3) for k, v in V_td.items() if k != 'terminal'})


batch_td_mc_comparison()
```

### 6.2 Convergence Properties

- **MC**: Minimizes mean squared error (fits observed returns)
- **TD**: Maximum likelihood MDP estimate (implicitly learns transition probabilities)

---

## 7. n-step TD

### 7.1 Concept

While TD(0) uses 1-step returns, n-step TD uses n actual rewards.

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

```
1-step: G_t^(1) = R_{t+1} + γV(S_{t+1})                    ← TD(0)
2-step: G_t^(2) = R_{t+1} + γR_{t+2} + γ²V(S_{t+2})
3-step: G_t^(3) = R_{t+1} + γR_{t+2} + γ²R_{t+3} + γ³V(S_{t+3})
...
∞-step: G_t^(∞) = R_{t+1} + γR_{t+2} + ...                ← MC
```

### 7.2 n-step TD Implementation

```python
def n_step_td(env, policy, n=3, n_episodes=1000, alpha=0.1, gamma=0.99):
    """
    n-step TD Policy Evaluation

    Args:
        n: Number of steps (n=1 is TD(0), n=∞ is MC)
    """
    V = defaultdict(float)

    for episode in range(n_episodes):
        states = []
        rewards = [0]  # R_0 = 0 (not used)
        # T starts at infinity so the action loop runs until a terminal is found
        T = float('inf')  # Terminal time
        t = 0

        state, _ = env.reset()
        states.append(state)

        while True:
            if t < T:
                action = policy(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t + 1

                state = next_state

            # tau is n steps behind the current time step — we can only update
            # state tau once we have n actual rewards ahead of it
            tau = t - n + 1  # Time to update

            if tau >= 0:
                # Sum n discounted real rewards starting from tau+1;
                # using actual rewards reduces bias compared to pure bootstrapping
                G = sum(gamma ** (i - tau - 1) * rewards[i]
                        for i in range(tau + 1, min(tau + n, T) + 1))

                if tau + n < T:
                    # Append bootstrap estimate only when the episode hasn't ended —
                    # at the boundary we use pure MC return (no V needed)
                    G += gamma ** n * V[states[tau + n]]

                # Update
                V[states[tau]] += alpha * (G - V[states[tau]])

            t += 1

            if tau == T - 1:
                break

    return dict(V)
```

### 7.3 Choice of n

| n value | Characteristics |
|---------|----------------|
| n=1 (TD(0)) | High bias, low variance |
| n=∞ (MC) | No bias, high variance |
| Middle n | Balance point (optimal n depends on environment) |

---

## 8. TD(λ) - Eligibility Traces

### 8.1 Concept

Use **weighted average** of all n-step returns.

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

- **λ=0**: TD(0)
- **λ=1**: MC

### 8.2 Eligibility Trace

Efficiently compute by tracking the "eligibility" of each state.

```python
def td_lambda(env, policy, lambd=0.8, n_episodes=1000, alpha=0.1, gamma=0.99):
    """
    TD(λ) with Eligibility Traces

    Args:
        lambd: λ value (0 ≤ λ ≤ 1)
    """
    V = defaultdict(float)

    for episode in range(n_episodes):
        # Reset traces at episode start — eligibility is episode-scoped so that
        # credit from one episode never bleeds into the next
        E = defaultdict(float)

        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD error
            if done:
                delta = reward - V[state]
            else:
                delta = reward + gamma * V[next_state] - V[state]

            # Accumulate trace for the visited state — states visited more recently
            # or more frequently receive a larger eligibility and thus a bigger update
            E[state] += 1  # accumulating traces

            # Propagate TD error to ALL states weighted by their eligibility —
            # this is the key efficiency gain over forward-view n-step TD
            for s in E:
                V[s] += alpha * delta * E[s]
                # gamma*lambda decay: gamma discounts future rewards, lambda
                # controls how far back credit propagates (lambda=0 → TD(0), lambda=1 → MC)
                E[s] *= gamma * lambd  # Decay trace

            state = next_state

    return dict(V)
```

### 8.3 Types of Eligibility Traces

```python
"""
1. Accumulating Traces (accumulating)
   E(s) ← E(s) + 1    (accumulate on each visit)

2. Replacing Traces (replace)
   E(s) ← 1           (reset to 1 on visit)

3. Dutch Traces
   E(s) ← (1-α)E(s) + 1
"""

def accumulating_trace(E, state):
    E[state] += 1
    return E

def replacing_trace(E, state):
    E[state] = 1
    return E

def dutch_trace(E, state, alpha):
    E[state] = (1 - alpha) * E[state] + 1
    return E
```

---

## 9. Example: Cliff Walking

```python
import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def cliff_walking_td():
    """TD learning in Cliff Walking"""

    env = gym.make('CliffWalking-v0')

    # State: 0-47 (4x12 grid)
    # Actions: 0=up, 1=right, 2=down, 3=left

    # lambda initializer ensures each new state gets its own zero array —
    # sharing one array across states would corrupt all Q-values
    Q = defaultdict(lambda: np.zeros(4))
    epsilon = 0.1
    # alpha=0.5 is higher than the 0.1 used in prediction — faster learning
    # is viable here because the cliff environment has clear reward structure
    alpha = 0.5
    gamma = 0.99
    n_episodes = 500

    episode_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Q-learning update (detailed in next lesson)
            # best_next=0 at terminal avoids adding phantom future value after the episode ends
            best_next = np.max(Q[next_state]) if not done else 0
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

            state = next_state

        episode_rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {ep + 1}: avg_reward = {avg:.1f}")

    # Visualize learned policy
    print("\n=== Learned Policy (4x12 grid) ===")
    arrows = {0: '^', 1: '>', 2: 'v', 3: '<'}

    for row in range(4):
        line = ""
        for col in range(12):
            state = row * 12 + col
            if state == 36:  # Start
                line += " S "
            elif state == 47:  # Goal
                line += " G "
            elif 37 <= state <= 46:  # Cliff
                line += " C "
            else:
                action = np.argmax(Q[state])
                line += f" {arrows[action]} "
        print(line)

    env.close()
    return Q, episode_rewards


Q, rewards = cliff_walking_td()
```

---

## 10. Summary

### Key Formulas

| Method | Update Rule |
|--------|-------------|
| TD(0) | $V(s) \leftarrow V(s) + \alpha[R + \gamma V(s') - V(s)]$ |
| n-step TD | $V(s) \leftarrow V(s) + \alpha[G^{(n)} - V(s)]$ |
| TD(λ) | $V(s) \leftarrow V(s) + \alpha \delta E(s)$ |

### TD Error

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

### Method Comparison

```
                    TD(0)        n-step        MC
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bias                High         Medium        None
Variance            Low          Medium        High
Update timing       Every step   After n steps Episode end
Continuous tasks    Possible     Possible      Impossible
```

---

## 11. Practice Problems

1. **TD Error**: What is the TD error when V(s)=5, R=1, γ=0.9, V(s')=6?

2. **n-step**: Write the return formula for n=2.

3. **TD(λ)**: What is the weight ratio of 1-step and 2-step returns when λ=0.5?

4. **Eligibility Trace**: What is the value of accumulating trace if state s is visited 2 times consecutively?

---

## Next Steps

In the next lesson **06_Q_Learning_SARSA.md**, we'll learn **Q-Learning** and **SARSA** algorithms that apply TD to control.

---

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapters 6, 7
- David Silver's RL Course, Lectures 4 & 5
- [Gymnasium CliffWalking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)
