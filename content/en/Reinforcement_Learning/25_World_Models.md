[Previous: RLHF Deep Dive](./24_RLHF_Deep_Dive.md)

---

# 25. World Models

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain world models as learned environment simulators for imagination-based planning
2. Implement the Dreamer architecture with RSSM latent dynamics
3. Build model predictive control (MPC) using learned dynamics models
4. Understand IRIS and other transformer-based world models
5. Compare model-based imagination with model-free methods on sample efficiency

---

## Table of Contents

1. [What Are World Models?](#1-what-are-world-models)
2. [Recurrent State-Space Models (RSSM)](#2-recurrent-state-space-models-rssm)
3. [Dreamer Architecture](#3-dreamer-architecture)
4. [Model Predictive Control](#4-model-predictive-control)
5. [Transformer World Models (IRIS)](#5-transformer-world-models-iris)
6. [Learning in Imagination](#6-learning-in-imagination)
7. [Practical World Model Training](#7-practical-world-model-training)
8. [Exercises](#8-exercises)

---

## 1. What Are World Models?

### 1.1 The Core Idea

```
Model-Free RL:
  Agent interacts with REAL environment to learn
  Needs millions of interactions (samples)
  Each interaction may be expensive or dangerous

World Models:
  Agent learns a MODEL of the environment
  Then "imagines" trajectories in the model
  Plans and learns inside its own imagination
  Needs far fewer real interactions!

Analogy:
  Chess player doesn't need to play 10M games.
  They think ahead: "If I move here, opponent does this, then I..."
  This mental simulation IS a world model.
```

### 1.2 Components of a World Model

```
A world model has three components:

1. Representation Model (Encoder):
   o_t -> z_t
   Maps observations to compact latent states

2. Transition Model (Dynamics):
   (z_t, a_t) -> z_{t+1}
   Predicts next latent state given current state and action

3. Observation Model (Decoder):
   z_t -> ô_t
   Reconstructs observations from latent states (optional)

Additionally:
4. Reward Predictor:
   z_t -> r̂_t
   Predicts reward from latent state

5. Continuation Predictor:
   z_t -> ĉ_t ∈ [0,1]
   Predicts whether episode continues
```

```text
┌─────────────────────────────────────────────────────────────────┐
│                   World Model Data Flow                          │
│                                                                 │
│  REAL EXPERIENCE (training the world model)                     │
│                                                                 │
│  o_t ──▶ [Encoder] ──▶ z_t ──┐                                 │
│                               │                                 │
│  a_t ─────────────────────────┤                                 │
│                               ▼                                 │
│                          [Dynamics] ──▶ z_{t+1}                │
│                               │                                 │
│                               ├──▶ [Decoder]  ──▶ ô_t          │
│                               ├──▶ [Reward]   ──▶ r̂_t          │
│                               └──▶ [Continue] ──▶ ĉ_t          │
│                                                                 │
│  IMAGINATION (training the actor-critic, no real env!)          │
│                                                                 │
│  z_t ──▶ [Actor π] ──▶ a_t                                     │
│    │                      │                                     │
│    └──────────────────────┤                                     │
│                           ▼                                     │
│                      [Dynamics] ──▶ z_{t+1} ──▶ (repeat H steps)
│                           │                                     │
│                           ├──▶ r̂_t                             │
│                           └──▶ [Critic V] ──▶ v_t              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 World Model History

```
Timeline:
1990: Schmidhuber - "Making the World Differentiable"
2015: Oh et al.   - Action-conditional video prediction
2018: Ha & Schmidhuber - "World Models" (VAE + RNN + controller)
2020: Hafner et al. - DreamerV1 (RSSM + actor-critic in imagination)
2021: Hafner et al. - DreamerV2 (discrete latents, Atari benchmark)
2023: Hafner et al. - DreamerV3 (single config, many domains)
2023: Micheli et al. - IRIS (transformer world model)
2024: Various      - Video prediction as world models (Genie, DIAMOND)
```

---

## 2. Recurrent State-Space Models (RSSM)

### 2.1 RSSM Architecture

```
RSSM combines deterministic and stochastic components:

Deterministic path (RNN):
  h_t = f(h_{t-1}, z_{t-1}, a_{t-1})    GRU/LSTM recurrence

Stochastic path:
  Prior:     p(z_t | h_t)                Predict from dynamics only
  Posterior: q(z_t | h_t, o_t)           Incorporate observation

Full state: s_t = (h_t, z_t)
  h_t captures long-term memory
  z_t captures stochastic variation

Why both?
  - Deterministic: stable long-term predictions
  - Stochastic: captures uncertainty and multimodality
```

```text
┌─────────────────────────────────────────────────────────────────┐
│            Recurrent State-Space Model (RSSM)                   │
│                                                                 │
│  t-1                          t                                 │
│   │                           │                                 │
│   │    a_{t-1}                │    a_t                          │
│   │       │                   │       │                         │
│   ▼       ▼                   ▼       ▼                         │
│  z_{t-1} ─┬──▶ [GRU] ──▶ h_t ─┬──▶ [GRU] ──▶ h_{t+1}         │
│            │         │         │         │                      │
│            │    ┌────┘         │    ┌────┘                      │
│            │    ▼              │    ▼                           │
│            │  [Prior]          │  [Prior]                       │
│            │  p(z_t|h_t)       │  p(z_{t+1}|h_{t+1})           │
│            │    │              │    │  (imagination: no obs)    │
│            │    │  o_t ─┐      │    │                           │
│            │    ▼       ▼      │    │                           │
│            │  [Posterior]      │    │                           │
│            │  q(z_t|h_t, o_t) │    │                           │
│            │    │              │    │                           │
│            └────▼──────────────┘    │                           │
│                z_t ─────────────────┘                           │
│                │                                                │
│           [h_t, z_t] = full state ──▶ decoder, reward head     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 RSSM Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np


class RSSM(nn.Module):
    """Recurrent State-Space Model for world model dynamics."""

    def __init__(self, state_dim=32, hidden_dim=200, action_dim=4,
                 obs_embed_dim=256, stoch_dim=32, n_categories=32):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        self.n_categories = n_categories

        # Deterministic state transition (GRU)
        self.gru = nn.GRUCell(stoch_dim * n_categories + action_dim, hidden_dim)

        # Prior: p(z_t | h_t) - predict stochastic state from deterministic
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ELU(),
            nn.Linear(256, stoch_dim * n_categories),
        )

        # Posterior: q(z_t | h_t, o_t) - incorporate observation
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + obs_embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, stoch_dim * n_categories),
        )

    def initial_state(self, batch_size, device='cpu'):
        """Return initial hidden and stochastic state."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim * self.n_categories,
                        device=device)
        return h, z

    def observe_step(self, h_prev, z_prev, action, obs_embed):
        """One step with observation (training)."""
        # Deterministic transition
        x = torch.cat([z_prev, action], dim=-1)
        h = self.gru(x, h_prev)

        # Prior
        prior_logits = self.prior_net(h)
        prior_logits = prior_logits.view(-1, self.stoch_dim, self.n_categories)

        # Posterior (uses observation)
        post_input = torch.cat([h, obs_embed], dim=-1)
        post_logits = self.posterior_net(post_input)
        post_logits = post_logits.view(-1, self.stoch_dim, self.n_categories)

        # Sample from posterior (straight-through Gumbel-Softmax)
        z_post = self._sample_categorical(post_logits)
        z_flat = z_post.view(-1, self.stoch_dim * self.n_categories)

        return h, z_flat, prior_logits, post_logits

    def imagine_step(self, h_prev, z_prev, action):
        """One step without observation (imagination)."""
        x = torch.cat([z_prev, action], dim=-1)
        h = self.gru(x, h_prev)

        prior_logits = self.prior_net(h)
        prior_logits = prior_logits.view(-1, self.stoch_dim, self.n_categories)

        z = self._sample_categorical(prior_logits)
        z_flat = z.view(-1, self.stoch_dim * self.n_categories)

        return h, z_flat

    def _sample_categorical(self, logits, temperature=1.0):
        """Sample from categorical with straight-through gradients."""
        dist = td.OneHotCategorical(logits=logits / temperature)
        sample = dist.sample()
        # Straight-through: use sample in forward, logits in backward
        return sample + dist.probs - dist.probs.detach()

    def kl_loss(self, prior_logits, post_logits):
        """KL divergence between posterior and prior."""
        prior_dist = td.OneHotCategorical(logits=prior_logits)
        post_dist = td.OneHotCategorical(logits=post_logits)
        kl = td.kl_divergence(post_dist, prior_dist).sum(dim=-1)
        return kl.mean()
```

---

## 3. Dreamer Architecture

### 3.1 DreamerV3 Overview

```
DreamerV3 Architecture:

Real Experience:
  o_t → [Encoder] → e_t → [RSSM posterior] → (h_t, z_t)
                                                   │
  Train: reconstruction loss + reward loss + KL loss

Imagination:
  (h_t, z_t) → [RSSM prior] → (h_{t+1}, z_{t+1}) → ... → (h_{t+H}, z_{t+H})
       ↑             │
    action       [Reward pred]
  from actor      [Continue pred]
                       │
  Train: actor-critic in imagined trajectories (no real env needed!)
```

### 3.2 World Model Training

```python
class WorldModel(nn.Module):
    """Complete world model: encoder + RSSM + decoder + reward/continue."""

    def __init__(self, obs_dim, action_dim, embed_dim=256,
                 hidden_dim=200, stoch_dim=32, n_categories=32):
        super().__init__()
        self.state_dim = stoch_dim * n_categories

        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, embed_dim),
            nn.ELU(),
        )

        # RSSM dynamics
        self.rssm = RSSM(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            obs_embed_dim=embed_dim,
            stoch_dim=stoch_dim,
            n_categories=n_categories,
        )

        # Observation decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + self.state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, obs_dim),
        )

        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim + self.state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

        # Continue predictor (probability episode continues)
        self.continue_head = nn.Sequential(
            nn.Linear(hidden_dim + self.state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def train_step(self, observations, actions, rewards, dones):
        """
        Train world model on a sequence of real experience.

        observations: (batch, T, obs_dim)
        actions: (batch, T, action_dim)
        rewards: (batch, T)
        dones: (batch, T)
        """
        batch_size, T = observations.shape[:2]
        device = observations.device

        # Initialize RSSM state
        h, z = self.rssm.initial_state(batch_size, device)

        # Process sequence
        all_prior_logits = []
        all_post_logits = []
        all_features = []

        for t in range(T):
            obs_embed = self.encoder(observations[:, t])
            action = actions[:, t] if t > 0 else torch.zeros_like(actions[:, 0])

            h, z, prior_logits, post_logits = self.rssm.observe_step(
                h, z, action, obs_embed
            )

            features = torch.cat([h, z], dim=-1)
            all_features.append(features)
            all_prior_logits.append(prior_logits)
            all_post_logits.append(post_logits)

        features = torch.stack(all_features, dim=1)  # (batch, T, feat_dim)

        # Reconstruction loss
        obs_pred = self.decoder(features)
        recon_loss = F.mse_loss(obs_pred, observations)

        # Reward prediction loss
        reward_pred = self.reward_head(features).squeeze(-1)
        reward_loss = F.mse_loss(reward_pred, rewards)

        # Continue prediction loss
        continue_pred = self.continue_head(features).squeeze(-1)
        continue_loss = F.binary_cross_entropy(continue_pred, 1 - dones.float())

        # KL loss (posterior vs prior)
        prior_logits = torch.stack(all_prior_logits, dim=1)
        post_logits = torch.stack(all_post_logits, dim=1)
        kl_loss = self.rssm.kl_loss(prior_logits, post_logits)

        # Total loss
        total_loss = recon_loss + reward_loss + continue_loss + 0.1 * kl_loss

        return {
            'total': total_loss,
            'recon': recon_loss.item(),
            'reward': reward_loss.item(),
            'continue': continue_loss.item(),
            'kl': kl_loss.item(),
        }

    def imagine(self, initial_h, initial_z, actor, horizon=15):
        """Generate imagined trajectory using the actor."""
        h, z = initial_h, initial_z
        imagined_features = []
        imagined_actions = []

        for t in range(horizon):
            features = torch.cat([h, z], dim=-1)
            imagined_features.append(features)

            # Actor selects action based on imagined state
            action = actor(features.detach())
            imagined_actions.append(action)

            # Step dynamics (imagination - no observation)
            h, z = self.rssm.imagine_step(h, z, action)

        # Final state features
        imagined_features.append(torch.cat([h, z], dim=-1))

        features = torch.stack(imagined_features, dim=1)
        actions = torch.stack(imagined_actions, dim=1)

        # Predict rewards and continues
        rewards = self.reward_head(features[:, :-1]).squeeze(-1)
        continues = self.continue_head(features[:, :-1]).squeeze(-1)

        return features, actions, rewards, continues
```

### 3.3 Actor-Critic in Imagination

```python
class DreamerActorCritic:
    """Actor-Critic trained entirely in imagination."""

    def __init__(self, feature_dim, action_dim, hidden_dim=256,
                 gamma=0.997, gae_lambda=0.95, actor_lr=3e-5,
                 critic_lr=3e-5, imagination_horizon=15):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.horizon = imagination_horizon

        # Actor: features -> action distribution
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Critic: features -> value
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def update(self, world_model, initial_states):
        """Update actor and critic using imagined trajectories."""
        h, z = initial_states  # From real experience replay

        # Imagine trajectories
        features, actions, rewards, continues = world_model.imagine(
            h, z, self.actor, self.horizon
        )

        # Compute values
        values = self.critic(features).squeeze(-1)

        # Compute lambda-returns (GAE-style)
        returns = self._compute_returns(
            rewards, values[:, :-1], values[:, 1:], continues
        )

        # Critic loss
        critic_loss = F.mse_loss(values[:, :-1], returns.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # Actor loss (maximize returns)
        # Use straight-through estimator for discrete actions
        actor_loss = -returns.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_return': returns.mean().item(),
        }

    def _compute_returns(self, rewards, values, next_values, continues):
        """Compute lambda-returns."""
        T = rewards.shape[1]
        returns = torch.zeros_like(rewards)
        last_return = next_values[:, -1]

        for t in reversed(range(T)):
            returns[:, t] = rewards[:, t] + \
                continues[:, t] * self.gamma * (
                    (1 - self.gae_lambda) * next_values[:, t] +
                    self.gae_lambda * last_return
                )
            last_return = returns[:, t]

        return returns
```

---

## 4. Model Predictive Control

### 4.1 MPC with Learned Models

```
MPC (Model Predictive Control):
  At each step:
  1. Generate many candidate action sequences
  2. Simulate each sequence in the world model
  3. Pick the sequence with highest predicted return
  4. Execute only the FIRST action
  5. Re-plan at next step

  This is "planning" - no need for a learned policy!
```

### 4.2 Cross-Entropy Method (CEM)

```python
class CEMPlanner:
    """Planning with Cross-Entropy Method."""

    def __init__(self, world_model, action_dim, horizon=12,
                 n_candidates=1000, n_elite=100, n_iterations=5):
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_elite = n_elite
        self.n_iterations = n_iterations

    def plan(self, h, z):
        """Plan best action sequence using CEM."""
        # Initialize action distribution
        mean = torch.zeros(self.horizon, self.action_dim)
        std = torch.ones(self.horizon, self.action_dim)

        for iteration in range(self.n_iterations):
            # Sample candidate action sequences
            actions = mean + std * torch.randn(
                self.n_candidates, self.horizon, self.action_dim
            )
            actions = actions.clamp(-1, 1)

            # Evaluate each candidate
            returns = self._evaluate_sequences(h, z, actions)

            # Select elite (top-k)
            elite_idx = returns.topk(self.n_elite).indices
            elite_actions = actions[elite_idx]

            # Update distribution
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp(min=0.01)

        return mean[0]  # Return first action

    @torch.no_grad()
    def _evaluate_sequences(self, h, z, action_sequences):
        """Evaluate action sequences in world model."""
        n = action_sequences.shape[0]
        h_exp = h.expand(n, -1)
        z_exp = z.expand(n, -1)

        total_reward = torch.zeros(n)

        for t in range(self.horizon):
            h_exp, z_exp = self.world_model.rssm.imagine_step(
                h_exp, z_exp, action_sequences[:, t]
            )
            features = torch.cat([h_exp, z_exp], dim=-1)
            reward = self.world_model.reward_head(features).squeeze(-1)
            cont = self.world_model.continue_head(features).squeeze(-1)

            total_reward += reward * (0.99 ** t)

        return total_reward
```

---

## 5. Transformer World Models (IRIS)

### 5.1 IRIS Architecture

```
IRIS (Imagination with auto-Regression over an Inner Speech):

Instead of RSSM, uses a transformer to model dynamics:

1. Tokenize observations (VQ-VAE)
   Image -> discrete tokens [t₁, t₂, ..., t_K]

2. Sequence: [obs_tokens₁, action₁, obs_tokens₂, action₂, ...]

3. GPT-style autoregressive prediction
   Predict next observation tokens given history

Advantages over RSSM:
  - Better at long-range dependencies
  - Scales well with compute
  - Leverages transformer infrastructure
```

### 5.2 Simplified Transformer World Model

```python
class TransformerWorldModel(nn.Module):
    """Simplified transformer-based world model."""

    def __init__(self, obs_vocab_size, action_dim, d_model=256,
                 n_heads=4, n_layers=4, max_seq_len=1000):
        super().__init__()

        # Observation tokenizer (simplified - could use VQ-VAE)
        self.obs_embedding = nn.Embedding(obs_vocab_size, d_model)
        self.action_embedding = nn.Linear(action_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4*d_model, dropout=0.1,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Prediction heads
        self.obs_head = nn.Linear(d_model, obs_vocab_size)
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, obs_tokens, actions, positions):
        """Predict next observation tokens and rewards."""
        # Interleave observation and action embeddings
        obs_emb = self.obs_embedding(obs_tokens)
        act_emb = self.action_embedding(actions)
        pos_emb = self.pos_embedding(positions)

        # Simple interleaving: [o₁, a₁, o₂, a₂, ...]
        seq = torch.cat([obs_emb, act_emb], dim=1) + pos_emb

        # Causal mask
        seq_len = seq.shape[1]
        mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1
        ).to(seq.device)

        output = self.transformer(seq, mask=mask)

        # Predict next obs tokens and rewards from action positions
        next_obs_logits = self.obs_head(output)
        reward_pred = self.reward_head(output)

        return next_obs_logits, reward_pred
```

---

## 6. Learning in Imagination

### 6.1 Imagination vs Real Experience

```
Dreamer training loop:

while not done:
    # Phase 1: Interact with real environment (small amount)
    for _ in range(collect_interval):
        action = actor(encode(observation))
        observation, reward, done, _ = env.step(action)
        replay_buffer.add(observation, action, reward, done)

    # Phase 2: Train world model on real data
    batch = replay_buffer.sample()
    world_model.train_step(batch)

    # Phase 3: Improve actor-critic in imagination (large amount!)
    for _ in range(imagination_steps):
        initial_states = replay_buffer.sample_states()
        actor_critic.update(world_model, initial_states)

Key insight: Phase 3 is FREE (no real environment needed).
We can do thousands of imagination steps per real step.
This is why world models are so sample efficient!
```

### 6.2 Sample Efficiency Comparison

```
Steps to solve common benchmarks:

Environment     | Model-Free (PPO) | World Model (Dreamer)
----------------|-------------------|----------------------
HalfCheetah     | 1,000,000        | 100,000
Walker          | 2,000,000        | 200,000
Hopper          | 500,000          | 50,000
Humanoid        | 10,000,000       | 1,000,000
Atari (median)  | 200,000,000      | 20,000,000

World models typically 10x more sample efficient!
But wall-clock time can be similar (world model training is compute-heavy).
```

---

## 7. Practical World Model Training

### 7.1 Training Tips

```
World model training best practices:

1. Balanced losses
   Total = recon + reward + continue + β * KL
   Typical: β starts small (0.1) and increases

2. Sequence length
   Train on sequences of 50-100 steps
   Imagination horizon: 15 steps (longer can compound errors)

3. Replay buffer
   Large buffer (1M+ transitions)
   Uniform sampling (prioritized helps less for world models)

4. Exploration
   Initial random exploration: 5000-10000 steps
   Then: policy + small noise

5. Model ensemble (optional)
   Train 5 models, use disagreement for exploration
   Penalize predictions with high variance (pessimism)
```

### 7.2 Debugging World Models

```python
def diagnose_world_model(world_model, test_data, horizon=50):
    """Diagnostic checks for world model quality."""
    observations, actions, rewards, dones = test_data

    # 1. One-step prediction accuracy
    one_step_errors = []
    for t in range(len(observations) - 1):
        pred = world_model.predict_one_step(observations[t], actions[t])
        error = np.linalg.norm(pred - observations[t+1])
        one_step_errors.append(error)

    print(f"One-step prediction error: {np.mean(one_step_errors):.4f}")

    # 2. Multi-step rollout error (compounds!)
    rollout_errors = []
    state = observations[0]
    for t in range(min(horizon, len(observations) - 1)):
        state = world_model.predict_one_step(state, actions[t])
        error = np.linalg.norm(state - observations[t+1])
        rollout_errors.append(error)

    print(f"Rollout errors at horizon 1/10/50:")
    for h in [1, 10, min(50, len(rollout_errors))]:
        if h <= len(rollout_errors):
            print(f"  h={h}: {rollout_errors[h-1]:.4f}")

    # 3. Reward prediction accuracy
    reward_preds = []
    for t in range(len(rewards)):
        r_pred = world_model.predict_reward(observations[t])
        reward_preds.append(r_pred)

    reward_correlation = np.corrcoef(rewards, reward_preds)[0, 1]
    print(f"Reward prediction correlation: {reward_correlation:.4f}")

    # Warning thresholds
    if np.mean(one_step_errors) > 1.0:
        print("WARNING: High one-step error - check encoder/decoder")
    if rollout_errors[-1] > 10 * rollout_errors[0]:
        print("WARNING: Error compounds quickly - shorten imagination horizon")
```

---

## 8. Exercises

### Exercise 1: Simple World Model

Build a world model for CartPole:
1. Implement encoder, transition model, and decoder
2. Collect 10,000 transitions with random policy
3. Train the world model on collected data
4. Evaluate: 1-step, 5-step, 20-step prediction accuracy
5. Visualize predicted vs actual trajectories

### Exercise 2: RSSM Dynamics Model

Implement the full RSSM:
1. Build RSSM with GRU, prior, and posterior networks
2. Implement the KL loss between prior and posterior
3. Train on CartPole/Pendulum sequences
4. Compare: deterministic-only vs RSSM (stochastic) predictions
5. Show that RSSM captures uncertainty in stochastic environments

### Exercise 3: Dreamer-Lite

Build a simplified Dreamer agent:
1. Implement world model (encoder + RSSM + decoders)
2. Implement actor-critic trained in imagination
3. Create the full training loop (collect, train WM, imagine, update AC)
4. Train on Pendulum-v1 or CartPole and compare with model-free PPO
5. Measure sample efficiency: episodes to solve

### Exercise 4: MPC with CEM

Implement model predictive control:
1. Train a world model on collected data
2. Implement CEM planner with the world model
3. Compare: CEM planning vs learned policy vs random
4. Vary planning horizon (5, 10, 20 steps) and measure effect
5. Show that re-planning at each step handles model errors

### Exercise 5: World Model Imagination Quality

Study how imagination quality affects learning:
1. Train world models of varying quality (different sizes/data)
2. Use each to train policies in imagination
3. Evaluate policies in the REAL environment
4. Plot: world model accuracy vs final policy performance
5. Identify the minimum world model quality needed for good policies

---

*End of Lesson 25*
