# 10. PPO and TRPO

**Difficulty: ⭐⭐⭐⭐ (Advanced)**

## Learning Objectives
- Understand stability issues in policy updates
- Learn TRPO's trust region concept
- Understand PPO's clipping mechanism
- Implement PPO with PyTorch

---

## 1. Problems in Policy Optimization

### 1.1 Dangers of Large Updates

In policy gradients, overly large updates can drastically degrade performance.

```
θ_new = θ_old + α∇J(θ)

Problem: Large α causes drastic policy changes, making learning unstable
Solution: Constrain policy changes
```

### 1.2 Solution Approaches

- **TRPO**: Constrain with KL divergence trust region (complex)
- **PPO**: Simple constraint using clipping

---

## 2. TRPO (Trust Region Policy Optimization)

### 2.1 Objective Function

Use ratio of new and old policies:

$$L^{CPI}(\theta) = \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A^{\pi_{old}}(s, a)\right]$$

### 2.2 KL Divergence Constraint

$$\text{maximize}_\theta \quad L^{CPI}(\theta)$$
$$\text{subject to} \quad \mathbb{E}[D_{KL}(\pi_{\theta_{old}} || \pi_\theta)] \leq \delta$$

### 2.3 Problems with TRPO

- Requires second-order derivatives (Hessian)

  The Hessian is the matrix of second-order partial derivatives. Computing it requires O(n²) space for n parameters — prohibitive for neural networks with millions of parameters. PPO avoids this by using first-order gradient descent with a simple clipping constraint.

- Needs conjugate gradient algorithm
- Complex implementation and high computational cost

---

## 3. PPO (Proximal Policy Optimization)

### 3.1 Core Idea

Use clipping to constrain policy ratio.

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

### 3.2 Clipped Objective Function

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)\right]$$

```python
def compute_ppo_loss(ratio, advantage, clip_epsilon=0.2):
    """PPO Clipped Loss"""
    # Clip the probability ratio to [1-ε, 1+ε] — prevents catastrophically large
    # policy updates that could destabilize training; ε=0.2 means at most a 20% change
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    # Take the minimum of clipped and unclipped objectives: this is a pessimistic bound —
    # when advantage > 0, min prevents the ratio from growing beyond 1+ε (caps benefit);
    # when advantage < 0, min prevents the ratio from falling below 1-ε (caps punishment)
    loss1 = ratio * advantage
    loss2 = clipped_ratio * advantage

    return -torch.min(loss1, loss2).mean()
```

### 3.3 Clipping Intuition

```
Advantage > 0 (good action):
- ratio increases → probability increases
- But ignore if ratio > 1+ε (prevent drastic increase)

Advantage < 0 (bad action):
- ratio decreases → probability decreases
- But ignore if ratio < 1-ε (prevent drastic decrease)
```

---

## 4. Complete PPO Implementation

### 4.1 PPO Agent

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def get_action(self, state, action=None):
        probs, value = self.forward(state)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=10,
        batch_size=64
    ):
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    def collect_rollouts(self, env, n_steps):
        """Collect experience"""
        states, actions, rewards, dones = [], [], [], []
        values, log_probs = [], []

        state, _ = env.reset()

        for _ in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            log_probs.append(log_prob.item())

            state = next_state if not done else env.reset()[0]

        # Last state value
        with torch.no_grad():
            _, _, _, last_value = self.network.get_action(
                torch.FloatTensor(state).unsqueeze(0)
            )

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'values': np.array(values),
            'log_probs': np.array(log_probs),
            'last_value': last_value.item()
        }

    def compute_gae(self, rollout):
        """Compute GAE"""
        rewards = rollout['rewards']
        values = rollout['values']
        dones = rollout['dones']
        last_value = rollout['last_value']

        advantages = np.zeros_like(rewards)
        last_gae = 0

        # Traverse backwards to accumulate the GAE sum in one pass without storing
        # all future deltas; last_value provides the bootstrap for the final step
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            # Mask out the next state's contribution when the episode ended —
            # multiplying by (1 - done) cleanly zeros out future bootstrap value
            # at terminal transitions without needing an if-branch inside the loop
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        # Returns = advantages + values: the critic learns to predict these returns,
        # while advantages (centered around zero by subtraction) drive the actor update
        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """PPO Update"""
        advantages, returns = self.compute_gae(rollout)

        # Normalize advantages to zero mean and unit variance — reduces sensitivity
        # to reward scale differences across environments and stabilizes gradient magnitude
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(rollout['states'])
        actions = torch.LongTensor(rollout['actions'])
        old_log_probs = torch.FloatTensor(rollout['log_probs'])
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # Multiple epoch updates: reusing the same rollout data multiple times
        # improves sample efficiency — this is PPO's key advantage over on-policy methods
        # like A2C that discard data after a single update
        for _ in range(self.update_epochs):
            # Random permutation for minibatching breaks temporal correlations in the
            # rollout and reduces gradient variance across the update epochs
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Evaluate with current policy
                _, new_log_probs, entropy, values = self.network.get_action(
                    batch_states, batch_actions
                )

                # Ratio in log space for numerical stability: exp(log π_new - log π_old)
                # is equivalent to π_new / π_old but avoids overflow with small probabilities
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective: surr1 is the unclipped update, surr2 limits
                # the policy change; taking the min makes this a pessimistic (conservative) bound
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Entropy bonus: negative because we minimize total_loss but want to
                # maximize entropy — encourages exploration throughout training
                entropy_loss = -entropy.mean()

                # Total loss
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients before stepping to prevent a single bad minibatch
                # from causing a runaway update that collapses the policy
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return actor_loss.item(), critic_loss.item()
```

### 4.2 PPO Training Loop

```python
import gymnasium as gym

def train_ppo(env_name='CartPole-v1', total_timesteps=100000, n_steps=2048):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)

    timesteps = 0
    episode_rewards = []
    current_episode_reward = 0

    while timesteps < total_timesteps:
        # Collect rollouts
        rollout = agent.collect_rollouts(env, n_steps)
        timesteps += n_steps

        # Track episode rewards
        for r, d in zip(rollout['rewards'], rollout['dones']):
            current_episode_reward += r
            if d:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0

        # PPO update
        actor_loss, critic_loss = agent.update(rollout)

        # Logging
        if len(episode_rewards) > 0 and timesteps % 10000 < n_steps:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            print(f"Timesteps: {timesteps}, Avg Reward: {avg_reward:.2f}")

    return agent, episode_rewards
```

---

## 5. PPO Variants

### 5.1 PPO-Clip (Basic)

The method implemented above.

### 5.2 PPO-Penalty

Add KL divergence as a penalty:

```python
def ppo_penalty_loss(ratio, advantage, old_probs, new_probs, beta=0.01):
    policy_loss = (ratio * advantage).mean()

    kl_div = F.kl_div(new_probs.log(), old_probs, reduction='batchmean')

    return -policy_loss + beta * kl_div
```

### 5.3 Clipped Value Loss

Apply clipping to value function as well:

```python
def clipped_value_loss(values, old_values, returns, clip_epsilon=0.2):
    # Clipped values
    clipped_values = old_values + torch.clamp(
        values - old_values, -clip_epsilon, clip_epsilon
    )

    # Maximum of two losses
    loss1 = (values - returns) ** 2
    loss2 = (clipped_values - returns) ** 2

    return 0.5 * torch.max(loss1, loss2).mean()
```

---

## 6. Continuous Action Space PPO

```python
class ContinuousPPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        # Actor: Tanh activations are preferred over ReLU in PPO for continuous actions
        # because they bound outputs to (-1, 1), preventing extreme pre-activations
        # that can cause the Normal distribution's mean to drift to unreachable values
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        # State-independent log_std initialized to zeros (std=1): starts with maximum
        # exploration and lets the network learn to reduce uncertainty as training progresses
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        mean = self.actor_mean(state)
        std = self.actor_log_std.exp()
        value = self.critic(state)
        return mean, std, value

    def get_action(self, state, action=None):
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        # Sum log_probs across action dimensions: assumes independent Gaussians per
        # dimension, so joint log-prob = sum of marginals (factored distribution)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value
```

---

## 7. Hyperparameter Guide

### 7.1 General Settings

```python
config = {
    # Learning
    'lr': 3e-4,                  # Learning rate
    'gamma': 0.99,               # Discount factor
    'gae_lambda': 0.95,          # GAE lambda

    # PPO specific
    'clip_epsilon': 0.2,         # Clipping range
    'update_epochs': 10,         # Update iterations
    'batch_size': 64,            # Minibatch size

    # Loss coefficients
    'value_coef': 0.5,           # Value loss coefficient
    'entropy_coef': 0.01,        # Entropy coefficient

    # Rollout
    'n_steps': 2048,             # Rollout length
    'n_envs': 8,                 # Parallel environments

    # Stabilization
    'max_grad_norm': 0.5,        # Gradient clipping
}
```

### 7.2 Environment-Specific Tuning

| Environment | lr | n_steps | clip_epsilon |
|-------------|-----|---------|--------------|
| CartPole | 3e-4 | 128 | 0.2 |
| LunarLander | 3e-4 | 2048 | 0.2 |
| Atari | 2.5e-4 | 128 | 0.1 |
| MuJoCo | 3e-4 | 2048 | 0.2 |

---

## 8. PPO vs Other Algorithms

| Algorithm | Complexity | Sample Efficiency | Stability |
|-----------|-----------|-------------------|-----------|
| REINFORCE | Low | Low | Low |
| A2C | Medium | Medium | Medium |
| TRPO | High | High | High |
| **PPO** | **Medium** | **High** | **High** |
| SAC | Medium | High | High |

**PPO Advantages:**
- TRPO-level performance, simple implementation
- Stable across various environments
- Low sensitivity to hyperparameters

---

## Summary

**PPO Core:**
```
L^{CLIP} = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

r(θ) = π_θ(a|s) / π_θ_old(a|s)  # Policy ratio
```

**Clipping Effect:**
- Constrains policy changes to [1-ε, 1+ε] range
- Prevents drastic updates
- Ensures learning stability

---

## Exercises

### Exercise 1: TRPO vs PPO Trade-off Analysis

Analyze the theoretical and practical differences between TRPO and PPO.

1. Explain why TRPO requires the Hessian of the KL divergence. What is the computational complexity of computing and inverting an n×n Hessian for a network with n parameters?
2. Describe how PPO's clipped objective approximates the TRPO constraint without second-order information.
3. Fill in the following comparison table based on the lesson content:

| Property | TRPO | PPO-Clip |
|----------|------|----------|
| Constraint type | | |
| Gradient order | | |
| Implementation complexity | | |
| Memory cost | | |
| Typical wall-clock speed | | |

4. In what scenario might TRPO's hard constraint be strictly necessary over PPO's soft clipping?

### Exercise 2: Clipping Behavior Visualization

Visualize how the PPO clipped objective responds to different ratio values.

1. For advantage values A ∈ {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0} and ε = 0.2, compute L^CLIP(r) for r ∈ [0.5, 1.5] using:
   ```python
   import numpy as np
   import matplotlib.pyplot as plt

   def l_clip(r, A, eps=0.2):
       clipped = np.clip(r, 1 - eps, 1 + eps)
       return np.minimum(r * A, clipped * A)
   ```
2. Plot L^CLIP as a function of r for each advantage value on the same axes.
3. Mark the clipping boundaries (r = 0.8 and r = 1.2) with vertical dashed lines.
4. Explain why the function is flat (zero gradient) outside the clipping region when A > 0 but not when A < 0 (and vice versa for A < 0).

### Exercise 3: GAE Lambda Sensitivity

Investigate how the GAE lambda parameter affects the bias-variance trade-off.

1. Train five PPO agents on LunarLander-v2 for 200,000 timesteps, varying only `gae_lambda` ∈ {0.0, 0.5, 0.9, 0.95, 1.0}.
2. For each run, record:
   - Mean episode reward (last 20 episodes)
   - Standard deviation of episode rewards
3. Plot learning curves for all five settings.
4. Explain the results in terms of the bias-variance trade-off:
   - λ = 0: 1-step TD advantage (low variance, high bias)
   - λ = 1: Monte Carlo advantage (high variance, no bias)
5. Which value of λ works best? Does this match the default λ = 0.95 used in the lesson?

### Exercise 4: Implement PPO-Penalty

Implement the KL-penalty variant of PPO from Section 5.2 and compare it with PPO-Clip.

1. Implement `ppo_penalty_loss(ratio, advantage, old_probs, new_probs, beta)` from Section 5.2.
2. Add adaptive beta adjustment: after each update epoch, measure the mean KL divergence:
   - If KL > 1.5 × target_kl: multiply beta by 2
   - If KL < target_kl / 1.5: divide beta by 2
   - Use target_kl = 0.01
3. Train PPO-Penalty and PPO-Clip on CartPole-v1 for 100,000 timesteps each.
4. Plot training curves and record final performance.
5. Does adaptive beta effectively control the KL divergence to stay near target_kl?

### Exercise 5: Vectorized Environment PPO

Extend PPO to use multiple parallel environments for faster data collection.

1. Create 4 parallel CartPole-v1 environments using `gymnasium.vector.SyncVectorEnv`.
2. Modify `collect_rollouts()` to collect n_steps transitions from all 4 environments simultaneously, producing a buffer of shape `(n_steps × 4,)`.
3. Verify that the total rollout size is 4× larger for the same n_steps.
4. Train for 100,000 total timesteps and compare wall-clock time against the single-environment version from Section 4.2.
5. Analyze: does using 4 environments change the quality of learning, or only the speed? Explain why in terms of sample diversity.

---

## Next Steps

- [11_Multi_Agent_RL.md](./11_Multi_Agent_RL.md) - Multi-Agent Reinforcement Learning
