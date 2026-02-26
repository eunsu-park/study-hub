# 13. Model-Based Reinforcement Learning

**Difficulty: ⭐⭐⭐⭐ (Advanced)**

## Learning Objectives
- Understand the distinction between model-free and model-based RL
- Implement the Dyna architecture for planning with learned models
- Learn world model approaches (Dreamer, MuZero)
- Apply Model-Based Policy Optimization (MBPO)
- Understand when model-based methods outperform model-free ones

---

## Table of Contents

1. [Model-Free vs Model-Based RL](#1-model-free-vs-model-based-rl)
2. [Dyna Architecture](#2-dyna-architecture)
3. [Learning World Models](#3-learning-world-models)
4. [Model-Based Policy Optimization (MBPO)](#4-model-based-policy-optimization-mbpo)
5. [MuZero: Planning without a Known Model](#5-muzero-planning-without-a-known-model)
6. [Dreamer: World Models for Continuous Control](#6-dreamer-world-models-for-continuous-control)
7. [Practice Problems](#7-practice-problems)

---

## 1. Model-Free vs Model-Based RL

### 1.1 Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│              Model-Free vs Model-Based RL                        │
│                                                                 │
│  Model-Free:                                                    │
│  ┌────────────────────────────────────────────────┐             │
│  │ Agent ──(action)──▶ Environment ──(s',r)──▶ Agent           │
│  │                                                │             │
│  │ • Learn value/policy directly from experience  │             │
│  │ • No explicit model of dynamics                │             │
│  │ • Examples: DQN, PPO, SAC                      │             │
│  │ • Simple but sample-inefficient                │             │
│  └────────────────────────────────────────────────┘             │
│                                                                 │
│  Model-Based:                                                   │
│  ┌────────────────────────────────────────────────┐             │
│  │ Agent ──(action)──▶ Environment ──(s',r)──▶ Agent           │
│  │   │                                      │     │             │
│  │   │         ┌──────────────────┐         │     │             │
│  │   └────────▶│  Learned Model   │◀────────┘     │             │
│  │             │  ŝ', r̂ = f(s,a) │               │             │
│  │             └────────┬─────────┘               │             │
│  │                      │                         │             │
│  │                 Planning                        │             │
│  │            (simulated rollouts)                 │             │
│  │                                                │             │
│  │ • Learn a model of environment dynamics         │             │
│  │ • Plan using the learned model                  │             │
│  │ • Examples: Dyna, MBPO, MuZero, Dreamer        │             │
│  │ • Sample-efficient but model errors accumulate  │             │
│  └────────────────────────────────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Trade-offs

```
┌──────────────────────┬──────────────────┬───────────────────────┐
│                      │ Model-Free       │ Model-Based           │
├──────────────────────┼──────────────────┼───────────────────────┤
│ Sample efficiency    │ Low              │ High (10-100x fewer)  │
│ Asymptotic perf.     │ High             │ Limited by model err  │
│ Computation          │ Low per step     │ High (planning)       │
│ Implementation       │ Simpler          │ More complex          │
│ Robustness           │ More robust      │ Sensitive to model    │
│ Best for             │ Simulation-heavy │ Real-world, expensive │
│                      │ environments     │ interactions          │
└──────────────────────┴──────────────────┴───────────────────────┘
```

---

## 2. Dyna Architecture

### 2.1 Dyna-Q Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│              Dyna-Q Architecture                                 │
│                                                                 │
│  ┌────────────────────────────────────────────┐                 │
│  │          Real Experience                    │                 │
│  │  s ──(a)──▶ Environment ──▶ s', r          │                 │
│  └─────────────────┬──────────────────────────┘                 │
│                    │                                            │
│           ┌────────┼────────┐                                   │
│           ▼        ▼        ▼                                   │
│     ┌──────────┐ ┌──────┐ ┌──────────┐                         │
│     │ Direct   │ │Model │ │ Planning │                         │
│     │ RL       │ │Learn │ │ (n steps)│                         │
│     │ Q-update │ │      │ │          │                         │
│     └──────────┘ └──────┘ └──────────┘                         │
│           │                     │                               │
│           └──────────┬──────────┘                               │
│                      ▼                                          │
│               Q-value Table / Network                           │
│                                                                 │
│  Loop:                                                          │
│  1. Act in real environment, observe (s, a, r, s')              │
│  2. Direct RL: Update Q(s,a)                                    │
│  3. Model learning: Update model(s,a) → (r̂, ŝ')               │
│  4. Planning: Repeat n times:                                   │
│     - Sample random (s, a) from experience                      │
│     - Simulate: r̂, ŝ' = model(s, a)                            │
│     - Update Q(s, a) using simulated experience                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Dyna-Q Implementation

```python
import numpy as np
from collections import defaultdict

class DynaQ:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=0.1, n_planning=5):
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}  # (s, a) → (r, s')
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning
        self.visited = []  # track visited (s, a) pairs

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, s, a, r, s_next, done):
        # Step 1: Direct RL update — always apply one real TD update first
        # so the Q-table reflects actual transitions before the model is used
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

        # Step 2: Model learning — store the deterministic observed transition;
        # this simple tabular model is exact for the visited (s,a) pairs but
        # cannot generalize to unvisited states (unlike neural network models)
        self.model[(s, a)] = (r, s_next, done)
        if (s, a) not in self.visited:
            self.visited.append((s, a))

        # Step 3: Planning (n simulated updates) — reusing the stored model
        # for n Q-updates costs far less than n real environment steps;
        # larger n_planning gives more sample efficiency at the cost of
        # more computation per real step
        for _ in range(self.n_planning):
            # Why sample uniformly from visited pairs: ensures all parts of
            # the known state space are updated equally, not just recent states
            idx = np.random.randint(len(self.visited))
            sim_s, sim_a = self.visited[idx]
            sim_r, sim_s_next, sim_done = self.model[(sim_s, sim_a)]

            # Q-learning update with simulated experience
            sim_target = sim_r if sim_done else sim_r + self.gamma * np.max(self.Q[sim_s_next])
            self.Q[sim_s, sim_a] += self.alpha * (sim_target - self.Q[sim_s, sim_a])
```

### 2.3 Dyna-Q+ (Exploration Bonus)

```python
class DynaQPlus(DynaQ):
    """Dyna-Q+ adds exploration bonus for states not visited recently."""

    def __init__(self, *args, kappa=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.kappa = kappa
        self.last_visit = defaultdict(int)  # (s, a) → last time step
        self.time_step = 0

    def update(self, s, a, r, s_next, done):
        self.time_step += 1
        self.last_visit[(s, a)] = self.time_step

        # Direct RL update (same as Dyna-Q)
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

        # Model learning
        self.model[(s, a)] = (r, s_next, done)
        if (s, a) not in self.visited:
            self.visited.append((s, a))

        # Planning with exploration bonus
        for _ in range(self.n_planning):
            idx = np.random.randint(len(self.visited))
            sim_s, sim_a = self.visited[idx]
            sim_r, sim_s_next, sim_done = self.model[(sim_s, sim_a)]

            # Add bonus for unvisited time
            tau = self.time_step - self.last_visit.get((sim_s, sim_a), 0)
            bonus = self.kappa * np.sqrt(tau)

            sim_target = (sim_r + bonus) if sim_done else \
                         (sim_r + bonus) + self.gamma * np.max(self.Q[sim_s_next])
            self.Q[sim_s, sim_a] += self.alpha * (sim_target - self.Q[sim_s, sim_a])
```

---

## 3. Learning World Models

### 3.1 Neural Network Dynamics Model

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DynamicsModel(nn.Module):
    """Predicts next state and reward given current state and action."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        features = self.network(x)
        next_state = state + self.state_head(features)  # predict residual
        reward = self.reward_head(features)
        return next_state, reward.squeeze(-1)


class EnsembleDynamicsModel(nn.Module):
    """Ensemble of dynamics models for uncertainty estimation."""

    def __init__(self, state_dim, action_dim, n_models=5, hidden_dim=256):
        super().__init__()
        # Why an ensemble of 5 models: a single model cannot distinguish
        # "I've seen this region and it's predictable" from "I haven't seen
        # this region at all" — disagreement between models is a cheap proxy
        # for epistemic uncertainty without requiring Bayesian inference
        self.models = nn.ModuleList([
            DynamicsModel(state_dim, action_dim, hidden_dim)
            for _ in range(n_models)
        ])

    def forward(self, state, action):
        predictions = [model(state, action) for model in self.models]
        next_states = torch.stack([p[0] for p in predictions])
        rewards = torch.stack([p[1] for p in predictions])

        # Mean prediction — averaging reduces variance compared to any single model
        mean_next_state = next_states.mean(dim=0)
        mean_reward = rewards.mean(dim=0)

        # Why std as uncertainty: high disagreement among models signals that
        # we are in a region of the state space with little training data;
        # MBPO uses this to terminate rollouts before errors compound
        uncertainty = next_states.std(dim=0).mean(dim=-1)

        return mean_next_state, mean_reward, uncertainty
```

### 3.2 Training the Model

```python
class ModelTrainer:
    def __init__(self, ensemble, lr=1e-3):
        self.ensemble = ensemble
        self.optimizers = [
            optim.Adam(model.parameters(), lr=lr)
            for model in ensemble.models
        ]

    def train(self, replay_buffer, batch_size=256, epochs=5):
        for epoch in range(epochs):
            for i, (model, optimizer) in enumerate(
                zip(self.ensemble.models, self.optimizers)
            ):
                # Why each model is trained on a different bootstrap sample:
                # if all models saw the same data they would converge to
                # similar predictions, defeating the purpose of the ensemble;
                # random sampling introduces diversity in model weights
                states, actions, rewards, next_states = \
                    replay_buffer.sample(batch_size)

                pred_next_states, pred_rewards = model(states, actions)

                # Why separate state and reward losses: they have different
                # scales and semantics; training jointly with equal weight
                # is a simplification — in practice, reward prediction is
                # often easier (scalar vs high-dim state) so its gradient
                # shouldn't dominate
                state_loss = nn.MSELoss()(pred_next_states, next_states)
                reward_loss = nn.MSELoss()(pred_rewards, rewards)
                loss = state_loss + reward_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

### 3.3 Model Error and Compounding

```
┌─────────────────────────────────────────────────────────────────┐
│              Model Error Compounding                             │
│                                                                 │
│  Real trajectory:     s₀ → s₁ → s₂ → s₃ → ...                │
│                                                                 │
│  Model rollout:       s₀ → ŝ₁ → ŝ₂ → ŝ₃ → ...               │
│                            ↑     ↑     ↑                       │
│                          small  medium  LARGE error             │
│                                                                 │
│  Error grows exponentially with rollout length!                 │
│                                                                 │
│  Mitigation strategies:                                         │
│  1. Short rollouts (H = 1-5 steps)                              │
│  2. Ensemble disagreement as uncertainty                        │
│  3. Truncate when uncertainty is high                           │
│  4. Mix real and simulated data                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Model-Based Policy Optimization (MBPO)

### 4.1 MBPO Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│              MBPO: Model-Based Policy Optimization               │
│                                                                 │
│  Key idea: Use learned model for SHORT rollouts only            │
│            (branched from real states)                           │
│                                                                 │
│  Algorithm:                                                     │
│  1. Collect real data D_env from environment                    │
│  2. Train ensemble dynamics model on D_env                      │
│  3. For each real state s in D_env:                             │
│     - Generate k-step model rollout (k = 1~5)                  │
│     - Add simulated transitions to D_model                      │
│  4. Train SAC policy on D_env ∪ D_model                        │
│  5. Repeat                                                      │
│                                                                 │
│  Benefits:                                                      │
│  • 10-100x more sample efficient than SAC alone                 │
│  • Model only used for short rollouts → less error              │
│  • Guaranteed monotonic improvement (under assumptions)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Simplified MBPO Implementation

```python
class MBPO:
    def __init__(self, env, state_dim, action_dim,
                 model_rollout_length=1, rollouts_per_step=400):
        self.env = env
        self.ensemble = EnsembleDynamicsModel(state_dim, action_dim)
        self.model_trainer = ModelTrainer(self.ensemble)

        # SAC as the model-free backbone
        self.policy = SACAgent(state_dim, action_dim)

        self.env_buffer = ReplayBuffer(capacity=1_000_000)
        self.model_buffer = ReplayBuffer(capacity=1_000_000)

        self.rollout_length = model_rollout_length
        self.rollouts_per_step = rollouts_per_step

    def train(self, total_steps=100_000):
        state, _ = self.env.reset()

        for step in range(total_steps):
            # 1. Real environment interaction
            action = self.policy.select_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            self.env_buffer.add(state, action, reward, next_state, done)
            state = next_state if not (done or truncated) else self.env.reset()[0]

            # 2. Train dynamics model periodically
            if step % 250 == 0 and len(self.env_buffer) > 1000:
                self.model_trainer.train(self.env_buffer, epochs=5)

            # 3. Generate model rollouts
            if len(self.env_buffer) > 1000:
                self._generate_model_rollouts()

            # 4. Train policy on mixed data
            if len(self.env_buffer) > 1000:
                # Sample from both buffers
                real_batch = self.env_buffer.sample(128)
                model_batch = self.model_buffer.sample(128) \
                    if len(self.model_buffer) > 128 else real_batch
                self.policy.update(real_batch, model_batch)

    def _generate_model_rollouts(self):
        """Branch short rollouts from real states."""
        # Why branch from real states: starting from states that were actually
        # observed keeps the rollout anchored to the real distribution;
        # errors compound quickly in pure imagination — real-state branching
        # limits the "distance" the model can drift from the true dynamics
        states = self.env_buffer.sample_states(self.rollouts_per_step)

        for state in states:
            s = state.clone()
            for h in range(self.rollout_length):
                a = self.policy.select_action(s)
                s_next, r, uncertainty = self.ensemble(
                    s.unsqueeze(0), a.unsqueeze(0)
                )

                # Why truncate on high uncertainty: ensemble disagreement
                # signals that the model is extrapolating beyond its training
                # data; continuing the rollout would add unreliable transitions
                # that corrupt the policy's value estimates
                if uncertainty.item() > 0.5:
                    break

                # Why mark done=False for model transitions: the model does
                # not simulate terminal conditions reliably; treating all
                # simulated transitions as non-terminal avoids spurious
                # value truncation at artificial episode ends
                self.model_buffer.add(
                    s.numpy(), a.numpy(), r.item(),
                    s_next.squeeze(0).detach().numpy(), False
                )
                s = s_next.squeeze(0).detach()
```

---

## 5. MuZero: Planning without a Known Model

### 5.1 MuZero Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              MuZero: Three Learned Functions                     │
│                                                                 │
│  1. Representation: h(observation) → hidden state               │
│     ┌───────────┐                                               │
│     │ obs (o_t) │──▶ h_θ ──▶ s_0 (hidden state)               │
│     └───────────┘                                               │
│                                                                 │
│  2. Dynamics: g(s_k, a_k) → s_{k+1}, r_k                      │
│     ┌──────────────────┐                                        │
│     │ s_k + action a_k │──▶ g_θ ──▶ s_{k+1}, r̂_k             │
│     └──────────────────┘                                        │
│                                                                 │
│  3. Prediction: f(s_k) → policy, value                          │
│     ┌───────────┐                                               │
│     │   s_k     │──▶ f_θ ──▶ π_k, v_k                         │
│     └───────────┘                                               │
│                                                                 │
│  Key insight: Model operates in LEARNED hidden state space,     │
│  not observation space → no need to predict pixels              │
│                                                                 │
│  Planning: MCTS (Monte Carlo Tree Search) using learned model   │
│                                                                 │
│  Results:                                                       │
│  • Atari: superhuman in 57 games                                │
│  • Go/Chess/Shogi: matches AlphaZero without rules              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 MuZero Planning (MCTS)

```
┌─────────────────────────────────────────────────────────────────┐
│              MCTS in MuZero                                      │
│                                                                 │
│  For each action decision:                                      │
│  1. Run N simulations through the learned model                 │
│  2. Each simulation:                                            │
│     a. SELECT: traverse tree using UCB                          │
│     b. EXPAND: use dynamics model g(s,a) → s', r̂              │
│     c. EVALUATE: use prediction model f(s') → π, v             │
│     d. BACKUP: update visit counts and values                   │
│                                                                 │
│  Tree after 50 simulations:                                     │
│                                                                 │
│             s₀ (root = current state)                           │
│            / | \                                                │
│          a₀  a₁  a₂                                            │
│         /    |     \                                            │
│       s₁    s₂    s₃        N(a₀)=20, N(a₁)=25, N(a₂)=5     │
│      / \    / \    |                                            │
│    a₀  a₁ a₀  a₂  a₁       Q(a₁) highest → select a₁        │
│    ...                                                          │
│                                                                 │
│  Final action: proportional to visit count N(a) at root         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Dreamer: World Models for Continuous Control

### 6.1 Dreamer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              DreamerV3 Architecture                               │
│                                                                 │
│  World Model (RSSM — Recurrent State-Space Model):              │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │  Deterministic path:                              │           │
│  │  h_t = f_θ(h_{t-1}, z_{t-1}, a_{t-1})           │           │
│  │                                                   │           │
│  │  Stochastic path:                                 │           │
│  │  Prior:     ẑ_t ~ p_θ(ẑ_t | h_t)                │           │
│  │  Posterior: z_t ~ q_θ(z_t | h_t, o_t)            │           │
│  │                                                   │           │
│  │  Decoder:   ô_t = dec_θ(h_t, z_t)                │           │
│  │  Reward:    r̂_t = rew_θ(h_t, z_t)                │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  Actor-Critic (trained entirely in imagination):                │
│  ┌──────────────────────────────────────────────────┐           │
│  │  Imagine trajectories using world model only      │           │
│  │  Actor:  a_t ~ π_θ(a_t | h_t, z_t)              │           │
│  │  Critic: v_θ(h_t, z_t)                           │           │
│  │  No real environment interaction during policy    │           │
│  │  training!                                        │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  Results:                                                       │
│  • First single algorithm to master 150+ diverse tasks          │
│  • Atari, DMControl, Minecraft diamond without task-specific    │
│    tuning                                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Imagination-Based Policy Learning

```
┌─────────────────────────────────────────────────────────────────┐
│              Dreamer Training Loop                               │
│                                                                 │
│  Outer loop (real environment):                                 │
│  1. Collect experience with current policy → replay buffer      │
│  2. Train world model on replay buffer                          │
│                                                                 │
│  Inner loop (imagination):                                      │
│  3. Sample starting states from replay buffer                   │
│  4. "Dream" H-step trajectories using world model:              │
│     s₀ → s₁ → s₂ → ... → s_H  (all in latent space)          │
│  5. Compute imagined rewards and values                         │
│  6. Update actor and critic on imagined trajectories            │
│                                                                 │
│  Key advantage: Can train policy on 10000s of imagined          │
│  trajectories per real environment step                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Practice Problems

### Exercise 1: Dyna-Q on GridWorld
Implement Dyna-Q and compare performance with different planning steps (n=0, 5, 50).

```python
# Starter code
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Simple GridWorld (or use FrozenLake)
env = gym.make("FrozenLake-v1", is_slippery=False)

results = {}
for n_planning in [0, 5, 50]:
    agent = DynaQ(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        n_planning=n_planning
    )

    episode_rewards = []
    for episode in range(500):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
    results[n_planning] = episode_rewards

# Plot: more planning steps → faster learning
for n, rewards in results.items():
    # Smooth with moving average
    smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
    plt.plot(smoothed, label=f'n_planning={n}')
plt.xlabel('Episode')
plt.ylabel('Reward (smoothed)')
plt.legend()
plt.title('Dyna-Q: Effect of Planning Steps')
plt.show()
```

### Exercise 2: Neural Dynamics Model
Train a neural network dynamics model on CartPole and evaluate prediction accuracy.

```python
# Collect data from random policy
# Train DynamicsModel to predict next state
# Evaluate: 1-step prediction error vs multi-step rollout error
# Show that error grows with rollout length

# Key metrics to plot:
# - 1-step prediction MSE
# - k-step rollout MSE for k = 1, 5, 10, 20
# - Ensemble disagreement correlation with actual error
```

### Exercise 3: Compare Sample Efficiency
Compare model-free SAC vs MBPO on a continuous control task.

```python
# Use HalfCheetah-v4 or Pendulum-v1
# Plot: reward vs environment steps
# Expected: MBPO reaches good performance in ~10x fewer steps
# But: MBPO has higher wall-clock time per step (model training + planning)
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│              Model-Based RL Landscape                            │
│                                                                 │
│  Simple                                                 Complex │
│  ←──────────────────────────────────────────────────────────→   │
│                                                                 │
│  Dyna-Q       MBPO          MuZero         DreamerV3            │
│  (tabular)    (ensemble +   (MCTS +        (RSSM +              │
│               short         learned        imagination)         │
│               rollouts)     hidden space)                       │
│                                                                 │
│  ┌──────┐    ┌──────┐     ┌──────┐       ┌──────┐              │
│  │Sample│    │Sample│     │Sample│       │Sample│              │
│  │eff:  │    │eff:  │     │eff:  │       │eff:  │              │
│  │ Med  │    │ High │     │V.High│       │V.High│              │
│  └──────┘    └──────┘     └──────┘       └──────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Takeaways:**
- Model-based RL trades computation for sample efficiency
- Ensemble models provide uncertainty estimates
- Short rollouts mitigate compounding model errors
- MuZero: planning in learned latent space (no observation prediction)
- Dreamer: entire policy training in imagination

---

## References

- Sutton & Barto Ch. 8: "Planning and Learning with Tabular Methods"
- [MBPO Paper](https://arxiv.org/abs/1906.08253) — Janner et al. 2019
- [MuZero Paper](https://arxiv.org/abs/1911.08265) — Schrittwieser et al. 2020
- [DreamerV3 Paper](https://arxiv.org/abs/2301.04104) — Hafner et al. 2023
- [Spinning Up: Model-Based Methods](https://spinningup.openai.com/)
