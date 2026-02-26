# 08. Policy Gradient

**Difficulty: ⭐⭐⭐⭐ (Advanced)**

## Learning Objectives
- Understand advantages and disadvantages of policy-based methods
- Derive the Policy Gradient Theorem
- Implement the REINFORCE algorithm
- Variance reduction techniques with baselines
- Connection to Actor-Critic

---

## 1. Value-Based vs Policy-Based

### 1.1 Comparison

| Feature | Value-Based (DQN) | Policy-Based |
|---------|-------------------|--------------|
| Learning Target | Q(s, a) | π(a\|s) |
| Policy Derivation | Indirect from Q | Direct learning |
| Action Space | Discrete (mainly) | Discrete + Continuous |
| Stochastic Policy | Difficult | Natural |
| Convergence | Can be unstable | Local optima |

### 1.2 Advantages of Policy-Based

```
1. Handles continuous action spaces (robot control)
2. Learns stochastic policies (rock-paper-scissors)
3. Policy space can be simpler
4. Better convergence guarantees (in some cases)
```

---

## 2. Policy Parameterization

### 2.1 Softmax Policy (Discrete Actions)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        logits = self.network(state)
        # Softmax converts raw scores to a valid probability distribution —
        # working in probability space lets us sample stochastic actions naturally
        return F.softmax(logits, dim=-1)

    def get_action(self, state):
        probs = self.forward(state)
        # Categorical wraps the probabilities so we can sample AND compute
        # the exact log-probability of the chosen action in one step
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        # We store log_prob rather than prob because the policy gradient theorem
        # uses ∇ log π — log turns products into sums and keeps gradients stable
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
```

### 2.2 Gaussian Policy (Continuous Actions)

```python
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        features = self.shared(state)
        mean = self.mean_layer(features)
        std = self.log_std.exp()
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
```

---

## 3. Policy Gradient Theorem

### 3.1 Objective Function

Maximize performance of policy π_θ:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

where τ = (s₀, a₀, r₀, s₁, a₁, r₁, ...) is a trajectory

### 3.2 Policy Gradient Theorem

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

**Intuitive Interpretation:**
- Increase probability of actions that led to good outcomes (high G_t)
- Decrease probability of actions that led to bad outcomes

### 3.3 Derivation (Log-derivative trick)

```
∇_θ π(a|s;θ) = π(a|s;θ) · ∇_θ log π(a|s;θ)

Therefore:
∇_θ J(θ) = E[R · ∇_θ log π(a|s;θ)]
         = E[∇_θ log π(a|s;θ) · R]
```

---

## 4. REINFORCE Algorithm

### 4.1 Basic REINFORCE

A Monte Carlo policy gradient method.

```python
class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = DiscretePolicy(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        # Episode storage
        self.log_probs = []
        self.rewards = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.policy.get_action(state_tensor)
        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def compute_returns(self):
        """Compute discounted returns"""
        returns = []
        G = 0
        # Traverse rewards in reverse so each G already includes all future rewards —
        # a single backward pass avoids recomputing the full sum from scratch each step
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        # Normalize returns to zero-mean unit-variance — this acts as a
        # dynamic baseline that stabilizes gradients across episodes of different lengths
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self):
        returns = self.compute_returns()

        # Compute policy loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            # Negative sign converts gradient ascent (maximize J) into gradient
            # descent (minimize loss) so standard optimizers like Adam work unchanged
            policy_loss.append(-log_prob * G)  # Negative (gradient ascent)

        loss = torch.stack(policy_loss).sum()

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset episode data
        self.log_probs = []
        self.rewards = []

        return loss.item()
```

### 4.2 Training Loop

```python
import gymnasium as gym
import numpy as np

def train_reinforce(env_name='CartPole-v1', n_episodes=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCE(state_dim, action_dim, lr=1e-3)

    scores = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)
            state = next_state
            total_reward += reward

        # Update after episode ends
        loss = agent.update()
        scores.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Avg Score: {np.mean(scores[-100:]):.2f}")

    return agent, scores
```

---

## 5. Variance Reduction with Baseline

### 5.1 Variance Problem

REINFORCE gradients have high variance.

```
Var(∇_θ J) ∝ E[(G - b)²]
```

### 5.2 Introducing Baseline

Subtracting a constant b doesn't change the expected value but reduces variance.

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b) \right]$$

Best baseline: b = V(s)

```python
class REINFORCEWithBaseline:
    def __init__(self, state_dim, action_dim, lr_policy=1e-3, lr_value=1e-3, gamma=0.99):
        self.policy = DiscretePolicy(state_dim, action_dim)
        self.value = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_value)
        self.gamma = gamma

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.states = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Sample action from policy
        action, log_prob = self.policy.get_action(state_tensor)

        # Value prediction
        value = self.value(state_tensor)

        self.log_probs.append(log_prob)
        self.values.append(value)
        self.states.append(state_tensor)

        return action

    def update(self):
        returns = self.compute_returns()

        values = torch.cat(self.values).squeeze()
        log_probs = torch.stack(self.log_probs)

        # detach() stops gradients from flowing into the value network through
        # the advantage — policy and value are trained with separate objectives
        # and should not interfere with each other's gradients
        advantages = returns - values.detach()

        # Policy loss: scale log-prob by advantage so good-outcome actions get
        # stronger gradient updates; mean (not sum) keeps the loss scale
        # independent of episode length
        policy_loss = -(log_probs * advantages).mean()

        # MSE against the actual returns teaches the value network to predict
        # how much total reward to expect from each state — this is supervised learning
        value_loss = F.mse_loss(values, returns)

        # Two separate optimizers prevent the value gradient from corrupting
        # the policy parameters and vice versa
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Value update
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Reset
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.states = []

        return policy_loss.item(), value_loss.item()
```

---

## 6. Continuous Action Space Example

### 6.1 Continuous Action REINFORCE

```python
class ContinuousREINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = GaussianPolicy(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        self.log_probs = []
        self.rewards = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.policy.get_action(state_tensor)

        self.log_probs.append(log_prob)
        return action.detach().numpy().squeeze()

    def update(self):
        returns = self.compute_returns()

        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
```

### 6.2 MountainCarContinuous Example

```python
def train_continuous():
    env = gym.make('MountainCarContinuous-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = ContinuousREINFORCE(state_dim, action_dim, lr=1e-3)

    for episode in range(500):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, done, truncated, _ = env.step(action)
            agent.rewards.append(reward)

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        agent.update()
        print(f"Episode {episode + 1}, Reward: {total_reward:.2f}")
```

---

## 7. Advanced Techniques

### 7.1 Entropy Regularization

Add policy entropy to the loss to encourage exploration.

```python
def compute_entropy(probs):
    """Compute policy entropy"""
    return -(probs * probs.log()).sum(dim=-1).mean()

# Loss function
total_loss = policy_loss - entropy_coef * entropy
```

### 7.2 Reward Shaping

Reward transformation to solve sparse reward problems:

```python
def shape_reward(reward, state, next_state, done):
    """Reward shaping example"""
    # Additional signal to original reward
    position_reward = abs(next_state[0] - state[0])  # Encourage movement

    if done and reward > 0:
        bonus = 100  # Goal achievement bonus
    else:
        bonus = 0

    return reward + 0.1 * position_reward + bonus
```

---

## 8. Limitations of REINFORCE

### 8.1 Problems

1. **High Variance**: Uses entire episodes, resulting in high variance
2. **Sample Inefficiency**: Must wait until episode ends
3. **Credit Assignment**: Difficult to determine which actions led to good outcomes

### 8.2 Solution → Actor-Critic

- Combines TD learning with policy gradients
- Reduces variance through bootstrapping
- Allows per-step updates

---

## Summary

| Algorithm | Update Timing | Baseline | Characteristics |
|-----------|---------------|----------|-----------------|
| REINFORCE | Episode end | None | Simple, high variance |
| REINFORCE + Baseline | Episode end | V(s) | Lower variance |
| Actor-Critic | Every step | V(s) or Q(s,a) | Efficient |

**Key Formula:**
```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) · (G - b)]
```

---

## Exercises

### Exercise 1: Log-Derivative Trick Derivation

Derive the policy gradient theorem from first principles.

1. Start from the objective J(θ) = E_τ[R(τ)] and expand the expectation as an integral over trajectories.
2. Show that ∇_θ p(τ;θ) = p(τ;θ) · ∇_θ log p(τ;θ) (the log-derivative identity).
3. Factor log p(τ;θ) into contributions from the policy and the environment dynamics.
4. Explain why the environment transition probabilities P(s'|s, a) disappear from the final gradient expression.
5. Write the final form ∇_θ J(θ) = E[∑_t ∇_θ log π_θ(a_t|s_t) · G_t] in your own words.

### Exercise 2: Implement Vanilla REINFORCE

Implement and train a REINFORCE agent on CartPole-v1 without using the `REINFORCEWithBaseline` class.

1. Use the `DiscretePolicy` class from Section 2.1.
2. Implement the `compute_returns()` method with discounted returns (γ = 0.99).
3. Do NOT normalize the returns (keep raw G_t values).
4. Train for 500 episodes and plot the episode reward over time.
5. Observe the high variance in the learning curve — describe what you see and why it occurs.

```python
# Starter code
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
# ... your implementation here
```

### Exercise 3: Baseline Effect on Variance

Empirically compare the effect of different baselines on training stability.

1. Train three separate agents for 500 episodes each on CartPole-v1:
   - Agent A: REINFORCE with no baseline (raw returns G_t)
   - Agent B: REINFORCE with return normalization (zero-mean, unit-variance G_t)
   - Agent C: REINFORCE with learned value baseline V(s)
2. Record the episode rewards for all three agents.
3. Plot the rolling mean (window = 50 episodes) and standard deviation for each agent.
4. Compute the variance of the 100-episode reward at the end of training.
5. Explain the results: why does subtracting a baseline not change the expected gradient but reduce its variance?

### Exercise 4: Gaussian Policy for Continuous Control

Extend the REINFORCE algorithm to a continuous action environment.

1. Use `ContinuousREINFORCE` from Section 6.1 on the `Pendulum-v1` environment (1D continuous action in [-2, 2]).
2. Inspect the `GaussianPolicy`: verify that `log_std` is a learnable parameter and explain why it is initialized to zero (std = 1).
3. Add a call to `np.clip(action, env.action_space.low, env.action_space.high)` before `env.step()` — explain why this is necessary.
4. Train for 300 episodes. Plot the evolution of `policy.log_std.item()` (the learned standard deviation) alongside the episode reward.
5. What does the change in standard deviation tell you about the agent's exploration behavior over time?

### Exercise 5: Entropy Regularization Ablation

Investigate how the entropy coefficient affects exploration and convergence.

1. Modify the `REINFORCE` update method to add an entropy bonus:
   ```python
   total_loss = policy_loss - entropy_coef * entropy_bonus
   ```
   where `entropy_bonus = compute_entropy(probs)` from Section 7.1.
2. Train four agents on CartPole-v1 with `entropy_coef` ∈ {0.0, 0.01, 0.1, 0.5}.
3. For each run, record: (a) mean reward of last 50 episodes, (b) episodes to reach mean reward ≥ 195.
4. Plot training curves for all four settings on the same figure.
5. Explain the trade-off: what goes wrong when entropy_coef is too large? Too small?

---

## Next Steps

- [09_Actor_Critic.md](./09_Actor_Critic.md) - Actor-Critic Methods
