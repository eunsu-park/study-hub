# 09. Actor-Critic Methods

**Difficulty: ⭐⭐⭐⭐ (Advanced)**

## Learning Objectives
- Understand Actor-Critic architecture
- Learn Advantage function and GAE
- Compare A2C and A3C algorithms
- Implement Actor-Critic with PyTorch

---

## 1. Actor-Critic Overview

### 1.1 Core Idea

**Actor**: Learns policy π(a|s;θ)
**Critic**: Learns value function V(s;w)

```
Actor-Critic = Policy Gradient + TD Learning
```

### 1.2 REINFORCE vs Actor-Critic

| REINFORCE | Actor-Critic |
|-----------|--------------|
| Update after episode ends | Update every step |
| Uses actual return G | Uses TD target |
| High variance | Lower variance, some bias |

---

## 2. Advantage Function

### 2.1 Definition

$$A(s, a) = Q(s, a) - V(s)$$

**Meaning:** How much better is this action compared to average

### 2.2 TD Error as Advantage

```
δ_t = r_t + γV(s_{t+1}) - V(s_t)

E[δ_t | s_t, a_t] = Q(s_t, a_t) - V(s_t) = A(s_t, a_t)
```

TD Error is an unbiased estimator of Advantage.

```python
def compute_advantage(rewards, values, next_values, dones, gamma=0.99):
    """Compute 1-step Advantage"""
    advantages = []
    for r, v, nv, d in zip(rewards, values, next_values, dones):
        if d:
            advantage = r - v
        else:
            advantage = r + gamma * nv - v
        advantages.append(advantage)
    return advantages
```

---

## 3. A2C (Advantage Actor-Critic)

### 3.1 Network Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Shared feature extraction: actor and critic share early layers because
        # low-level state representations (e.g., position, velocity) are useful
        # for both policy decisions and value estimation — sharing reduces computation
        # and forces the network to learn representations beneficial to both heads
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
            # No activation here — softmax is applied in forward() so we can
            # work with raw logits, which is numerically more stable
        )

        # Critic outputs a scalar value V(s), not per-action values — it estimates
        # how good the *current state* is on average, independent of any specific action
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.shared(state)
        policy = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return policy, value

    def get_action(self, state):
        policy, value = self.forward(state)
        # Categorical distribution enables stochastic action selection — essential
        # for exploration; log_prob is stored for the policy gradient update
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
```

### 3.2 A2C Agent

```python
class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 value_coef=0.5, entropy_coef=0.01):
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Episode buffer
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.entropies = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        policy, value = self.network(state_tensor)

        dist = torch.distributions.Categorical(policy)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)
        self.entropies.append(dist.entropy())

        return action.item()

    def store(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns(self, next_value):
        """Compute n-step returns"""
        returns = []
        R = next_value

        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)

        return torch.tensor(returns)

    def update(self, next_state):
        # Bootstrapping: instead of waiting for the episode to end, we estimate
        # the remaining return using the critic — this is what makes Actor-Critic
        # an online algorithm capable of updating mid-episode
        with torch.no_grad():
            _, next_value = self.network(
                torch.FloatTensor(next_state).unsqueeze(0)
            )
            next_value = next_value.item()

        returns = self.compute_returns(next_value)
        values = torch.cat(self.values).squeeze()
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        # Detach values from the computational graph for advantage calculation —
        # we treat the critic's estimate as a fixed baseline, not a target to
        # differentiate through; only the actor_loss should update via log_probs
        advantages = returns - values.detach()

        # Policy gradient loss: REINFORCE with baseline — multiply log probability
        # by advantage so actions better than average are reinforced, worse ones suppressed
        actor_loss = -(log_probs * advantages).mean()
        # MSE loss trains the critic to accurately predict discounted returns —
        # a more accurate baseline reduces actor gradient variance over time
        critic_loss = F.mse_loss(values, returns)
        # Entropy bonus encourages exploration by penalizing overly confident policies;
        # negative sign because we maximize entropy but minimize the total loss
        entropy_loss = -entropies.mean()

        total_loss = (actor_loss +
                     self.value_coef * critic_loss +
                     self.entropy_coef * entropy_loss)

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping prevents exploding gradients — common in RNNs and
        # environments with sparse or high-variance rewards; 0.5 is a conservative bound
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        # Reset buffer
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.entropies = []

        return actor_loss.item(), critic_loss.item()
```

### 3.3 A2C Training

```python
import gymnasium as gym
import numpy as np

def train_a2c(env_name='CartPole-v1', n_episodes=1000, n_steps=5):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(state_dim, action_dim)
    scores = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(reward, done)
            state = next_state
            total_reward += reward
            step_count += 1

            # n-step update
            if step_count % n_steps == 0 or done:
                agent.update(next_state)

        scores.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Avg: {np.mean(scores[-100:]):.2f}")

    return agent, scores
```

---

## 4. GAE (Generalized Advantage Estimation)

### 4.1 n-step Returns Tradeoff

| n | Bias | Variance |
|---|------|----------|
| 1 (TD) | High | Low |
| ∞ (MC) | Low | High |

### 4.2 GAE Formula

Exponentially-weighted average of all n-step advantages:

$$A^{GAE}\_t = \sum\_{l=0}^{\infty} (\gamma \lambda)^l \delta\_{t+l}$$

where δ_t = r_t + γV(s_{t+1}) - V(s_t)

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation"""
    advantages = []
    # Traverse backwards because GAE is a recursive sum: A_t depends on A_{t+1}.
    # Computing forward would require storing all future deltas first; backward
    # traversal lets us accumulate the running sum gae in a single pass
    gae = 0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            # Episode boundary: future returns beyond this point don't belong to
            # this trajectory, so reset the accumulator to prevent value leakage
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * next_values[t] - values[t]
            # GAE combines TD-1 (low variance, high bias) with Monte Carlo (high variance,
            # low bias) via lambda: lam=0 gives pure TD, lam=1 gives Monte Carlo returns
            gae = delta + gamma * lam * gae

        advantages.insert(0, gae)

    return torch.tensor(advantages)
```

### 4.3 A2C with GAE

```python
class A2CWithGAE(A2CAgent):
    def __init__(self, *args, gae_lambda=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.gae_lambda = gae_lambda
        self.next_values = []

    def compute_gae_returns(self):
        """GAE-based advantage and returns"""
        values = torch.cat(self.values).squeeze().tolist()
        next_vals = self.next_values

        advantages = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values[t]
                gae = delta
            else:
                delta = self.rewards[t] + self.gamma * next_vals[t] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        advantages = torch.tensor(advantages)
        returns = advantages + torch.tensor(values)

        return advantages, returns
```

---

## 5. A3C (Asynchronous Advantage Actor-Critic)

### 5.1 Core Idea

Multiple workers interact with environments in parallel and asynchronously update gradients.

```
┌─────────────────────────────────────┐
│          Global Network             │
│         (Shared Parameters)          │
└──────────┬──────────┬───────────────┘
           │          │
     ┌─────┴────┐  ┌──┴─────┐
     │ Worker 1 │  │ Worker 2│  ...
     │   Env 1  │  │  Env 2  │
     └──────────┘  └─────────┘
```

### 5.2 Pseudocode

```python
# Worker behavior
def worker(global_network, optimizer, env):
    local_network = copy(global_network)

    while True:
        # Collect experience with local network
        trajectory = collect_trajectory(local_network, env)

        # Compute gradients
        loss = compute_loss(trajectory)
        gradients = compute_gradients(loss, local_network)

        # Asynchronous update
        apply_gradients(optimizer, global_network, gradients)

        # Sync local network
        local_network.load_state_dict(global_network.state_dict())
```

### 5.3 A2C vs A3C

| A2C | A3C |
|-----|-----|
| Synchronous update | Asynchronous update |
| Batch processing | Stream processing |
| More stable | Faster (parallel) |
| GPU efficient | CPU efficient |

**Current Recommendation:** A2C is more commonly used as it's more efficient on GPUs

---

## 6. Continuous Action Space Actor-Critic

```python
class ContinuousActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Shared network
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor: mean and standard deviation
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # log_std as a learnable parameter (not state-dependent) keeps the policy
        # simpler and more stable early in training; exp() ensures std > 0 always
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.shared(state)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp()
        value = self.critic(features)
        return mean, std, value

    def get_action(self, state, deterministic=False):
        mean, std, value = self.forward(state)

        if deterministic:
            # At test time, use the mean directly — no exploration noise needed
            # since we want the best known action, not a random sample
            action = mean
        else:
            # Normal distribution models continuous action spaces; sampling adds
            # stochastic exploration proportional to the learned std
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()

        return action, value

    def evaluate(self, state, action):
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)

        # Sum log_probs across action dimensions: for multivariate Normal with
        # independent dimensions, the joint log-probability is the sum of marginals
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)

        return value, log_prob, entropy
```

---

## 7. Learning Stabilization Techniques

### 7.1 Gradient Clipping

```python
# Gradient norm clipping
torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)

# Gradient value clipping
torch.nn.utils.clip_grad_value_(network.parameters(), clip_value=1.0)
```

### 7.2 Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import LinearLR

scheduler = LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.1,
    total_iters=total_timesteps
)
```

### 7.3 Reward Normalization

```python
class RunningMeanStd:
    def __init__(self):
        self.mean = 0
        self.var = 1
        # Small initial count prevents division-by-zero before any data is seen
        # while having negligible effect once real samples arrive
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Welford's online algorithm: updates mean and variance incrementally
        # without storing all past data — O(1) memory regardless of episode count
        self.mean += delta * batch_count / total_count
        self.var = (self.var * self.count + batch_var * batch_count +
                   delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, x):
        # 1e-8 epsilon prevents division by zero when variance is near zero
        # (e.g., if all rewards in a batch are identical)
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
```

---

## 8. Practice: LunarLander

```python
def train_lunarlander():
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(
        state_dim, action_dim,
        lr=7e-4, gamma=0.99,
        value_coef=0.5, entropy_coef=0.01
    )

    scores = []
    n_steps = 5

    for episode in range(2000):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.store(reward, done or truncated)
            state = next_state
            total_reward += reward
            steps += 1

            if steps % n_steps == 0 or done or truncated:
                agent.update(next_state)

            if done or truncated:
                break

        scores.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(scores[-100:])
            print(f"Episode {episode + 1}, Avg: {avg:.2f}")

            if avg >= 200:
                print("Solved!")
                break

    return agent, scores
```

---

## Summary

| Component | Role | Learning Target |
|-----------|------|-----------------|
| Actor | Policy | θ (policy parameters) |
| Critic | Value evaluation | w (value parameters) |
| Advantage | Action quality measure | A = Q - V ≈ δ |

**Loss Function:**
```
L = L_actor + c1 * L_critic + c2 * L_entropy
L_actor = -log π(a|s) * A
L_critic = (V - target)²
L_entropy = -Σ π log π
```

---

## Next Steps

- [10_PPO_TRPO.md](./10_PPO_TRPO.md) - Trust Region Policy Optimization
