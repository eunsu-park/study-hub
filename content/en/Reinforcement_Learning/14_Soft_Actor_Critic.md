# 14. Soft Actor-Critic (SAC)

**Difficulty: ⭐⭐⭐⭐ (Advanced)**

## Learning Objectives
- Understand maximum entropy reinforcement learning
- Implement the SAC algorithm for continuous action spaces
- Learn automatic temperature (alpha) tuning
- Compare SAC with PPO and TD3
- Apply SAC to practical continuous control tasks

---

## Table of Contents

1. [Maximum Entropy RL](#1-maximum-entropy-rl)
2. [SAC Algorithm](#2-sac-algorithm)
3. [SAC Implementation](#3-sac-implementation)
4. [Automatic Temperature Tuning](#4-automatic-temperature-tuning)
5. [SAC vs Other Algorithms](#5-sac-vs-other-algorithms)
6. [Practical Tips](#6-practical-tips)
7. [Practice Problems](#7-practice-problems)

---

## 1. Maximum Entropy RL

### 1.1 Standard RL vs Maximum Entropy RL

```
┌─────────────────────────────────────────────────────────────────┐
│              Maximum Entropy Framework                            │
│                                                                 │
│  Standard RL objective:                                         │
│  π* = argmax_π  E [ Σ γ^t r_t ]                               │
│                  → maximize expected return only                 │
│                                                                 │
│  Maximum Entropy RL objective:                                  │
│  π* = argmax_π  E [ Σ γ^t (r_t + α H(π(·|s_t))) ]            │
│                  → maximize return + policy entropy             │
│                                                                 │
│  Where:                                                         │
│  • H(π(·|s)) = -E[log π(a|s)] is the policy entropy           │
│  • α (temperature) controls exploration-exploitation balance    │
│                                                                 │
│  Benefits of maximum entropy:                                   │
│  1. Encourages exploration (higher entropy = more random)       │
│  2. Captures multiple modes (doesn't collapse to one solution)  │
│  3. More robust to perturbations                                │
│  4. Better transfer and fine-tuning                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Soft Bellman Equation

```
┌─────────────────────────────────────────────────────────────────┐
│              Soft Value Functions                                 │
│                                                                 │
│  Soft state value:                                              │
│  V(s) = E_a~π [ Q(s,a) - α log π(a|s) ]                      │
│                                                                 │
│  Soft Q-value (Bellman equation):                               │
│  Q(s,a) = r(s,a) + γ E_s' [ V(s') ]                          │
│         = r(s,a) + γ E_s' [ E_a'~π [ Q(s',a') - α log π(a'|s') ] ]
│                                                                 │
│  Soft policy improvement:                                       │
│  π_new = argmin_π  D_KL( π(·|s) || exp(Q(s,·)/α) / Z(s) )   │
│                                                                 │
│  In practice: π outputs mean and std of Gaussian               │
│  a ~ tanh(μ + σ · ε),  ε ~ N(0, I)                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. SAC Algorithm

### 2.1 SAC Components

```
┌─────────────────────────────────────────────────────────────────┐
│              SAC Architecture                                    │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │                  Actor (Policy)                    │           │
│  │  π_φ(a|s): Squashed Gaussian                     │           │
│  │  Input: state s                                   │           │
│  │  Output: μ(s), σ(s) → a = tanh(μ + σ·ε)         │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │              Twin Critics (Q1, Q2)                 │           │
│  │  Q_θ1(s, a), Q_θ2(s, a)                          │           │
│  │  Input: state s, action a                         │           │
│  │  Output: Q-value                                  │           │
│  │  → Use min(Q1, Q2) to prevent overestimation     │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │              Target Networks (Q1', Q2')            │           │
│  │  Soft update: θ' ← τθ + (1-τ)θ'                  │           │
│  │  Provides stable targets for critic training      │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │              Temperature (α)                       │           │
│  │  Controls entropy bonus                           │           │
│  │  Can be fixed or automatically tuned              │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 SAC Update Rules

```
┌─────────────────────────────────────────────────────────────────┐
│              SAC Training Steps                                  │
│                                                                 │
│  For each gradient step:                                        │
│                                                                 │
│  1. Sample batch (s, a, r, s', done) from replay buffer         │
│                                                                 │
│  2. Compute target:                                             │
│     a' ~ π_φ(·|s')                                              │
│     y = r + γ(1-done) × [min(Q'₁(s',a'), Q'₂(s',a'))          │
│                           - α log π_φ(a'|s')]                   │
│                                                                 │
│  3. Update Critics (minimize MSE):                              │
│     L_Q = E[(Q_θi(s,a) - y)²]  for i = 1, 2                  │
│                                                                 │
│  4. Update Actor (maximize):                                    │
│     ã ~ π_φ(·|s)  (reparameterization trick)                   │
│     L_π = E[α log π_φ(ã|s) - min(Q_θ1(s,ã), Q_θ2(s,ã))]    │
│                                                                 │
│  5. Update Temperature (if auto-tuning):                        │
│     L_α = E[-α (log π_φ(ã|s) + H_target)]                    │
│                                                                 │
│  6. Soft update target networks:                                │
│     θ'i ← τ θi + (1-τ) θ'i                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. SAC Implementation

### 3.1 Actor Network (Squashed Gaussian Policy)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Why reparameterization trick (rsample instead of sample): allows
        # gradients to flow through the stochastic sampling step — z = μ + σ·ε
        # moves randomness to a fixed ε ~ N(0,I) so the actor loss can be
        # differentiated with respect to μ and σ
        z = normal.rsample()

        # Why tanh squashing: maps unbounded Gaussian samples to (-1, 1),
        # matching the bounded action spaces common in continuous control
        # (e.g., joint torques); without squashing the policy could produce
        # arbitrarily large actions that saturate actuators
        action = torch.tanh(z)

        # Why the correction term -log(1 - action^2 + 1e-6): the change-of-
        # variables formula for log-probabilities requires subtracting the
        # log absolute Jacobian of tanh; omitting this makes log_prob incorrect,
        # causing the entropy estimate and actor gradient to be wrong
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic_action(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)
```

### 3.2 Critic Networks

```python
class TwinQCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)
```

### 3.3 SAC Agent

```python
import copy

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 auto_alpha=True, target_entropy=None):

        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha

        # Networks
        self.actor = GaussianActor(state_dim, action_dim, hidden_dim)
        # Why twin critics: using min(Q1, Q2) as the target prevents the
        # overestimation bias that occurs when a single Q-network is used —
        # overestimation causes the actor to exploit incorrect high Q-values,
        # leading to policy collapse; twin critics were first shown effective in TD3
        self.critic = TwinQCritic(state_dim, action_dim, hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)

        # Why freeze target parameters: target networks provide stable regression
        # targets during critic updates; allowing gradients to flow into the target
        # would create a moving-target problem that destabilizes training
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Temperature (alpha)
        if auto_alpha:
            # Why default target entropy = -action_dim: a heuristic that sets
            # the entropy target to "one bit per action dimension", giving the
            # policy enough freedom to explore without becoming nearly uniform
            self.target_entropy = target_entropy or -action_dim
            # Why optimize log_alpha (not alpha directly): log_alpha is
            # unconstrained and its gradient is well-behaved; exponentiating
            # guarantees alpha > 0 without any clipping or projection
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic_action(state)
            else:
                action, _ = self.actor.sample(state)
        return action.squeeze(0).numpy()

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch

        # --- Update Critics ---
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            # Why min of twin targets: taking the minimum of Q1 and Q2 in the
            # Bellman target propagates pessimistic Q-estimates, counteracting
            # the upward bias that accumulates when a critic is bootstrapped
            # from its own (often optimistic) predictions
            q_target = torch.min(q1_target, q2_target)
            # Why subtract alpha * log_prob in the target: this is the soft
            # Bellman backup — the entropy bonus augments every reward with
            # the policy's log-probability, encouraging the policy to remain
            # stochastic and explore all high-value actions
            target = rewards + self.gamma * (1 - dones) * \
                     (q_target - self.alpha * next_log_probs)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Temperature ---
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- Soft Update Target Networks ---
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            # Why soft update with small τ (0.005): slowly blending the online
            # weights into the target (θ' ← τθ + (1-τ)θ') keeps targets stable
            # across consecutive gradient steps — this is critical for continuous-
            # action algorithms like SAC and TD3 where the actor is updated every
            # step; DQN typically uses hard updates (τ=1) every N steps instead
            # because its discrete action space makes the policy less sensitive
            # to small target fluctuations
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'entropy': -log_probs.mean().item()
        }
```

### 3.4 Replay Buffer

```python
import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=1_000_000):
        self.capacity = capacity
        self.idx = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idxs]),
            torch.FloatTensor(self.actions[idxs]),
            torch.FloatTensor(self.rewards[idxs]),
            torch.FloatTensor(self.next_states[idxs]),
            torch.FloatTensor(self.dones[idxs])
        )

    def __len__(self):
        return self.size
```

### 3.5 Training Loop

```python
import gymnasium as gym

def train_sac(env_name='Pendulum-v1', total_steps=100_000,
              batch_size=256, start_steps=5000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = env.action_space.high[0]

    agent = SACAgent(state_dim, action_dim)
    buffer = ReplayBuffer(state_dim, action_dim)

    state, _ = env.reset()
    episode_reward = 0
    episode_rewards = []

    for step in range(total_steps):
        # Random actions for initial exploration
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state) * action_scale

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(state, action / action_scale, reward, next_state, float(terminated))

        state = next_state
        episode_reward += reward

        if done:
            episode_rewards.append(episode_reward)
            state, _ = env.reset()
            episode_reward = 0

        # Update after collecting enough data
        if step >= start_steps and len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            metrics = agent.update(batch)

            if step % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                print(f"Step {step}: avg_reward={avg_reward:.1f}, "
                      f"alpha={metrics['alpha']:.3f}, "
                      f"entropy={metrics['entropy']:.3f}")

    return agent, episode_rewards
```

---

## 4. Automatic Temperature Tuning

### 4.1 Why Auto-Tuning Matters

```
┌─────────────────────────────────────────────────────────────────┐
│              Temperature (α) Effect                              │
│                                                                 │
│  α too high:                     α too low:                     │
│  ┌─────────────────────┐         ┌─────────────────────┐       │
│  │ Entropy dominates    │         │ Return dominates     │       │
│  │ → nearly random     │         │ → premature          │       │
│  │ → slow learning     │         │   convergence        │       │
│  │ → poor performance  │         │ → poor exploration   │       │
│  └─────────────────────┘         └─────────────────────┘       │
│                                                                 │
│  Auto-tuning:                                                   │
│  ┌──────────────────────────────────────────────────┐           │
│  │ Constraint: H(π(·|s)) ≥ H_target               │           │
│  │ If entropy < target: increase α (explore more)   │           │
│  │ If entropy > target: decrease α (exploit more)   │           │
│  │ H_target = -dim(A) (heuristic for continuous)    │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  Typical α trajectory during training:                          │
│  α                                                              │
│  │                                                              │
│  │  ╲                                                           │
│  │   ╲                                                          │
│  │    ╲___                                                      │
│  │        ╲____                                                 │
│  │             ╲_________                                       │
│  │                       ─────                                  │
│  └──────────────────────────────▶ steps                        │
│  (starts high for exploration, decreases as policy converges)   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Temperature Loss

```python
# Automatic temperature tuning objective
# L(α) = E_a~π [-α log π(a|s) - α H_target]
# = E_a~π [-α (log π(a|s) + H_target)]

# In the update step:
alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

# Intuition:
# When entropy (-log_probs) < H_target: log_probs + H_target > 0
# → gradient pushes log_alpha up → α increases → more entropy encouraged
# When entropy > H_target: log_probs + H_target < 0
# → gradient pushes log_alpha down → α decreases
```

---

## 5. SAC vs Other Algorithms

### 5.1 Comparison Table

```
┌───────────────┬──────────┬──────────┬──────────┬───────────────┐
│               │ SAC      │ PPO      │ TD3      │ DDPG          │
├───────────────┼──────────┼──────────┼──────────┼───────────────┤
│ Policy type   │ Stochas. │ Stochas. │ Determin.│ Deterministic │
│ On/Off policy │ Off      │ On       │ Off      │ Off           │
│ Action space  │ Contin.  │ Both     │ Contin.  │ Continuous    │
│ Entropy reg.  │ Yes      │ Yes      │ No       │ No            │
│ Twin critics  │ Yes      │ No       │ Yes      │ No            │
│ Sample eff.   │ High     │ Low      │ High     │ Medium        │
│ Stability     │ High     │ High     │ Medium   │ Low           │
│ Hyperparams   │ Few      │ Many     │ Medium   │ Many          │
│ Auto-tuning   │ α tuning │ No       │ No       │ No            │
└───────────────┴──────────┴──────────┴──────────┴───────────────┘
```

### 5.2 When to Use SAC

```
Use SAC when:
✓ Continuous action spaces (robotics, control)
✓ Sample efficiency matters (real-world, expensive simulation)
✓ You want stable training with minimal tuning
✓ Multi-modal optimal policies exist

Use PPO instead when:
✓ Discrete action spaces
✓ On-policy learning is preferred
✓ Simulation is cheap (can generate many samples)
✓ Distributed training (PPO scales better)

Use TD3 instead when:
✓ Deterministic policy is preferred
✓ Simpler implementation needed
✓ No entropy regularization wanted
```

---

## 6. Practical Tips

### 6.1 Hyperparameters

```
┌────────────────────────┬──────────────┬──────────────────────────┐
│ Hyperparameter         │ Default      │ Notes                    │
├────────────────────────┼──────────────┼──────────────────────────┤
│ Learning rate          │ 3e-4         │ Same for actor & critic  │
│ Discount (γ)           │ 0.99         │ Standard                 │
│ Soft update (τ)        │ 0.005        │ Slow target updates      │
│ Batch size             │ 256          │ Larger is more stable    │
│ Buffer size            │ 1M           │ Large replay buffer      │
│ Hidden layers          │ (256, 256)   │ 2 layers is standard     │
│ Start steps            │ 5000-10000   │ Random exploration first │
│ Target entropy         │ -dim(A)      │ Heuristic, works well    │
│ Gradient steps/env step│ 1            │ 1:1 ratio is standard    │
└────────────────────────┴──────────────┴──────────────────────────┘
```

### 6.2 Common Issues and Solutions

```
Issue: Training instability / Q-values diverge
→ Check reward scale (normalize if needed)
→ Reduce learning rate
→ Increase batch size

Issue: Low entropy (premature convergence)
→ Enable auto alpha tuning
→ Increase initial alpha
→ Check action bounds

Issue: Slow learning
→ Increase start_steps for better initial exploration
→ Try larger networks
→ Check reward shaping

Issue: Action values saturating at bounds
→ Ensure proper action scaling
→ Check tanh squashing implementation
→ Verify log_prob correction term
```

---

## 7. Practice Problems

### Exercise 1: SAC on Pendulum
Train SAC on `Pendulum-v1` and plot the learning curve.

```python
# Train SAC
agent, rewards = train_sac('Pendulum-v1', total_steps=50_000)

# Plot learning curve
import matplotlib.pyplot as plt
window = 10
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(smoothed)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('SAC on Pendulum-v1')
plt.show()

# Expected: converges to ~-200 within 20K steps
```

### Exercise 2: Compare SAC vs PPO
Train both SAC and PPO on a continuous control task and compare sample efficiency.

```python
# Use HalfCheetah-v4 or Hopper-v4
# Plot reward vs environment steps for both algorithms
# Expected: SAC reaches same performance in ~5x fewer environment steps
# But PPO may have lower wall-clock time per step
```

### Exercise 3: Ablation Study
Run SAC with these variations and compare:
1. Fixed alpha = 0.2 (no auto-tuning)
2. Auto alpha (default)
3. No entropy term (alpha = 0, equivalent to TD3-like)
4. Single Q-network (no twin critics)

```python
# Expected findings:
# - Auto alpha > fixed alpha (adapts to task)
# - With entropy > without (better exploration)
# - Twin critics > single (prevents overestimation)
```

### Exercise 4: Custom Environment
Apply SAC to a custom continuous control task.

```python
# Example: reaching task
import gymnasium as gym
from gymnasium import spaces

class ReachingEnv(gym.Env):
    """2D reaching task: move arm tip to target."""

    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,))
        self.target = np.array([0.5, 0.5])

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.pos = np.random.uniform(-1, 1, size=2)
        return np.concatenate([self.pos, self.target]), {}

    def step(self, action):
        self.pos = np.clip(self.pos + action * 0.1, -1, 1)
        dist = np.linalg.norm(self.pos - self.target)
        reward = -dist
        done = dist < 0.05
        return np.concatenate([self.pos, self.target]), reward, done, False, {}
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│              SAC Key Components                                  │
│                                                                 │
│  1. Maximum entropy objective: reward + α × entropy             │
│  2. Squashed Gaussian policy: a = tanh(μ + σε)                 │
│  3. Twin Q-critics: min(Q1, Q2) prevents overestimation        │
│  4. Automatic temperature: α adapts to maintain target entropy  │
│  5. Off-policy: high sample efficiency via replay buffer        │
│                                                                 │
│  SAC is the go-to algorithm for continuous control tasks         │
│  due to its stability, sample efficiency, and minimal tuning.   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## References

- [SAC Paper (v1)](https://arxiv.org/abs/1801.01290) — Haarnoja et al. 2018
- [SAC Paper (v2, auto-alpha)](https://arxiv.org/abs/1812.05905) — Haarnoja et al. 2018
- [Spinning Up: SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [CleanRL SAC Implementation](https://docs.cleanrl.dev/rl-algorithms/sac/)
