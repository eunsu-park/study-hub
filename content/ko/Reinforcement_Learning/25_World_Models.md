[이전: RLHF Deep Dive](./24_RLHF_Deep_Dive.md)

---

# 25. 월드 모델

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 월드 모델을 상상 기반 계획을 위한 학습된 환경 시뮬레이터로 설명
2. RSSM 잠재 역학을 사용한 Dreamer 아키텍처 구현
3. 학습된 역학 모델을 사용한 모델 예측 제어 (MPC) 구축
4. IRIS 및 기타 트랜스포머 기반 월드 모델 이해
5. 모델 기반 상상과 모델 프리 방법의 샘플 효율성 비교

---

## 목차

1. [월드 모델이란?](#1-월드-모델이란)
2. [Recurrent State-Space Models (RSSM)](#2-recurrent-state-space-models-rssm)
3. [Dreamer 아키텍처](#3-dreamer-아키텍처)
4. [모델 예측 제어](#4-모델-예측-제어)
5. [트랜스포머 월드 모델 (IRIS)](#5-트랜스포머-월드-모델-iris)
6. [상상 속에서의 학습](#6-상상-속에서의-학습)
7. [실전 월드 모델 학습](#7-실전-월드-모델-학습)
8. [연습문제](#8-연습문제)

---

## 1. 월드 모델이란?

### 1.1 핵심 아이디어

```
Model-Free RL:
  에이전트가 실제 환경과 상호작용하여 학습
  수백만 번의 상호작용이 필요 (샘플)
  각 상호작용이 비싸거나 위험할 수 있음

월드 모델:
  에이전트가 환경의 모델을 학습
  그 다음 모델 안에서 궤적을 "상상"
  자신의 상상 속에서 계획하고 학습
  실제 상호작용이 훨씬 적게 필요!

비유:
  체스 플레이어는 천만 게임을 할 필요가 없습니다.
  미리 생각합니다: "여기로 움직이면, 상대가 이렇게 하고, 그러면 내가..."
  이 정신적 시뮬레이션이 바로 월드 모델입니다.
```

### 1.2 월드 모델의 구성 요소

```
월드 모델은 세 가지 구성 요소를 가집니다:

1. 표현 모델 (인코더):
   o_t -> z_t
   관찰을 압축된 잠재 상태로 매핑

2. 전이 모델 (역학):
   (z_t, a_t) -> z_{t+1}
   현재 상태와 행동이 주어지면 다음 잠재 상태를 예측

3. 관찰 모델 (디코더):
   z_t -> ô_t
   잠재 상태에서 관찰을 복원 (선택적)

추가로:
4. 보상 예측기:
   z_t -> r̂_t
   잠재 상태에서 보상을 예측

5. 지속 예측기:
   z_t -> ĉ_t ∈ [0,1]
   에피소드가 계속되는지 예측
```

```text
┌─────────────────────────────────────────────────────────────────┐
│                   월드 모델 데이터 흐름                           │
│                                                                 │
│  실제 경험 (월드 모델 학습)                                       │
│                                                                 │
│  o_t ──▶ [인코더] ──▶ z_t ──┐                                   │
│                              │                                  │
│  a_t ──────────────────────▶─┤                                  │
│                              ▼                                  │
│                         [역학 모델] ──▶ z_{t+1}                 │
│                              │                                  │
│                              ├──▶ [디코더]   ──▶ ô_t            │
│                              ├──▶ [보상 예측] ──▶ r̂_t           │
│                              └──▶ [지속 예측] ──▶ ĉ_t           │
│                                                                 │
│  상상 (액터-크리틱 학습, 실제 환경 불필요!)                       │
│                                                                 │
│  z_t ──▶ [액터 π] ──▶ a_t                                      │
│    │                      │                                     │
│    └──────────────────────┤                                     │
│                           ▼                                     │
│                      [역학 모델] ──▶ z_{t+1} ──▶ (H 스텝 반복)  │
│                           │                                     │
│                           ├──▶ r̂_t                             │
│                           └──▶ [크리틱 V] ──▶ v_t               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 월드 모델의 역사

```
연혁:
1990: Schmidhuber - "세계를 미분 가능하게 만들기"
2015: Oh et al.   - 행동 조건부 비디오 예측
2018: Ha & Schmidhuber - "World Models" (VAE + RNN + 컨트롤러)
2020: Hafner et al. - DreamerV1 (RSSM + 상상 속 액터-크리틱)
2021: Hafner et al. - DreamerV2 (이산 잠재, Atari 벤치마크)
2023: Hafner et al. - DreamerV3 (단일 설정, 다양한 도메인)
2023: Micheli et al. - IRIS (트랜스포머 월드 모델)
2024: Various      - 비디오 예측 기반 월드 모델 (Genie, DIAMOND)
```

---

## 2. Recurrent State-Space Models (RSSM)

### 2.1 RSSM 아키텍처

```
RSSM은 결정적 요소와 확률적 요소를 결합합니다:

결정적 경로 (RNN):
  h_t = f(h_{t-1}, z_{t-1}, a_{t-1})    GRU/LSTM 순환

확률적 경로:
  사전:   p(z_t | h_t)                역학만으로 예측
  사후:   q(z_t | h_t, o_t)           관찰을 반영

전체 상태: s_t = (h_t, z_t)
  h_t는 장기 기억을 포착
  z_t는 확률적 변동을 포착

왜 둘 다?
  - 결정적: 안정적인 장기 예측
  - 확률적: 불확실성과 다중 모드를 포착
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
│            │  [사전 분포]       │  [사전 분포]                    │
│            │  p(z_t|h_t)       │  p(z_{t+1}|h_{t+1})           │
│            │    │              │    │  (상상: 관찰 없음)         │
│            │    │  o_t ─┐      │    │                           │
│            │    ▼       ▼      │    │                           │
│            │  [사후 분포]       │    │                           │
│            │  q(z_t|h_t, o_t) │    │                           │
│            │    │              │    │                           │
│            └────▼──────────────┘    │                           │
│                z_t ─────────────────┘                           │
│                │                                                │
│           [h_t, z_t] = 전체 상태 ──▶ 디코더, 보상 예측          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 RSSM 구현

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

## 3. Dreamer 아키텍처

### 3.1 DreamerV3 개요

```
DreamerV3 아키텍처:

실제 경험:
  o_t → [인코더] → e_t → [RSSM 사후] → (h_t, z_t)
                                              │
  학습: 복원 손실 + 보상 손실 + KL 손실

상상:
  (h_t, z_t) → [RSSM 사전] → (h_{t+1}, z_{t+1}) → ... → (h_{t+H}, z_{t+H})
       ↑             │
    행동          [보상 예측]
  (액터에서)     [지속 예측]
                       │
  학습: 상상된 궤적에서 액터-크리틱 (실제 환경 불필요!)
```

### 3.2 월드 모델 학습

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

### 3.3 상상 속의 액터-크리틱

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

## 4. 모델 예측 제어

### 4.1 학습된 모델을 사용한 MPC

```
MPC (Model Predictive Control):
  각 스텝에서:
  1. 많은 후보 행동 시퀀스 생성
  2. 월드 모델에서 각 시퀀스를 시뮬레이션
  3. 가장 높은 예측 리턴의 시퀀스 선택
  4. 첫 번째 행동만 실행
  5. 다음 스텝에서 재계획

  이것은 "계획" - 학습된 정책이 필요 없습니다!
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

## 5. 트랜스포머 월드 모델 (IRIS)

### 5.1 IRIS 아키텍처

```
IRIS (Imagination with auto-Regression over an Inner Speech):

RSSM 대신 트랜스포머를 사용하여 역학을 모델링:

1. 관찰 토큰화 (VQ-VAE)
   이미지 -> 이산 토큰 [t₁, t₂, ..., t_K]

2. 시퀀스: [관찰_토큰₁, 행동₁, 관찰_토큰₂, 행동₂, ...]

3. GPT 스타일 자기회귀 예측
   히스토리가 주어지면 다음 관찰 토큰을 예측

RSSM 대비 장점:
  - 장거리 의존성 처리 능력 향상
  - 계산량에 따라 잘 확장
  - 트랜스포머 인프라 활용
```

### 5.2 간소화된 트랜스포머 월드 모델

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

## 6. 상상 속에서의 학습

### 6.1 상상 vs 실제 경험

```
Dreamer 학습 루프:

while not done:
    # 1단계: 실제 환경과 상호작용 (소량)
    for _ in range(collect_interval):
        action = actor(encode(observation))
        observation, reward, done, _ = env.step(action)
        replay_buffer.add(observation, action, reward, done)

    # 2단계: 실제 데이터로 월드 모델 학습
    batch = replay_buffer.sample()
    world_model.train_step(batch)

    # 3단계: 상상 속에서 액터-크리틱 개선 (대량!)
    for _ in range(imagination_steps):
        initial_states = replay_buffer.sample_states()
        actor_critic.update(world_model, initial_states)

핵심 통찰: 3단계는 무료입니다 (실제 환경 불필요).
실제 스텝당 수천 번의 상상 스텝을 할 수 있습니다.
이것이 월드 모델의 샘플 효율성이 높은 이유입니다!
```

### 6.2 샘플 효율성 비교

```
일반적인 벤치마크를 풀기 위한 스텝:

환경            | Model-Free (PPO) | 월드 모델 (Dreamer)
----------------|-------------------|----------------------
HalfCheetah     | 1,000,000        | 100,000
Walker          | 2,000,000        | 200,000
Hopper          | 500,000          | 50,000
Humanoid        | 10,000,000       | 1,000,000
Atari (중간값)  | 200,000,000      | 20,000,000

월드 모델은 일반적으로 10배 더 샘플 효율적!
하지만 실측 시간은 비슷할 수 있음 (월드 모델 학습에 계산이 많이 듦).
```

---

## 7. 실전 월드 모델 학습

### 7.1 학습 팁

```
월드 모델 학습 모범 사례:

1. 균형 잡힌 손실
   Total = 복원 + 보상 + 지속 + β * KL
   일반적: β는 작게 시작하여 (0.1) 점차 증가

2. 시퀀스 길이
   50-100 스텝의 시퀀스로 학습
   상상 지평선: 15 스텝 (더 길면 오차 누적)

3. 리플레이 버퍼
   큰 버퍼 (1M+ 전이)
   균일 샘플링 (우선순위가 월드 모델에는 덜 도움)

4. 탐색
   초기 랜덤 탐색: 5000-10000 스텝
   그 다음: 정책 + 작은 잡음

5. 모델 앙상블 (선택적)
   5개 모델 학습, 불일치를 탐색에 사용
   높은 분산의 예측에 페널티 (비관주의)
```

### 7.2 월드 모델 디버깅

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

## 8. 연습문제

### 연습문제 1: 간단한 월드 모델

CartPole을 위한 월드 모델을 구축하세요:
1. 인코더, 전이 모델, 디코더 구현
2. 랜덤 정책으로 10,000개의 전이 수집
3. 수집된 데이터로 월드 모델 학습
4. 평가: 1단계, 5단계, 20단계 예측 정확도
5. 예측된 궤적 vs 실제 궤적 시각화

### 연습문제 2: RSSM 역학 모델

완전한 RSSM을 구현하세요:
1. GRU, 사전, 사후 네트워크로 RSSM 구축
2. 사전과 사후 사이의 KL 손실 구현
3. CartPole/Pendulum 시퀀스에서 학습
4. 비교: 결정적만 vs RSSM (확률적) 예측
5. RSSM이 확률적 환경에서 불확실성을 포착함을 시연

### 연습문제 3: Dreamer-Lite

간소화된 Dreamer 에이전트를 구축하세요:
1. 월드 모델 구현 (인코더 + RSSM + 디코더)
2. 상상 속에서 학습되는 액터-크리틱 구현
3. 전체 학습 루프 생성 (수집, WM 학습, 상상, AC 업데이트)
4. Pendulum-v1 또는 CartPole에서 학습하고 모델 프리 PPO와 비교
5. 샘플 효율성 측정: 해결까지의 에피소드 수

### 연습문제 4: CEM을 사용한 MPC

모델 예측 제어를 구현하세요:
1. 수집된 데이터로 월드 모델 학습
2. 월드 모델로 CEM 플래너 구현
3. 비교: CEM 계획 vs 학습된 정책 vs 랜덤
4. 계획 지평선 변경 (5, 10, 20 스텝)과 효과 측정
5. 각 스텝에서의 재계획이 모델 오차를 처리함을 시연

### 연습문제 5: 월드 모델 상상 품질

상상 품질이 학습에 미치는 영향을 연구하세요:
1. 다양한 품질의 월드 모델 학습 (다른 크기/데이터)
2. 각각으로 상상 속에서 정책 학습
3. 실제 환경에서 정책 평가
4. 그래프: 월드 모델 정확도 vs 최종 정책 성능
5. 좋은 정책에 필요한 최소 월드 모델 품질 식별

---

*25강 끝*
