# 13. 모델 기반 강화학습(Model-Based Reinforcement Learning)

**난이도: ⭐⭐⭐⭐ (고급)**

## 학습 목표
- 모델 프리(model-free)와 모델 기반(model-based) 강화학습의 차이 이해
- 학습된 모델을 사용한 계획(planning)을 위한 Dyna 아키텍처 구현
- 세계 모델(world model) 접근법 학습 (Dreamer, MuZero)
- 모델 기반 정책 최적화(Model-Based Policy Optimization, MBPO) 적용
- 모델 기반 방법이 모델 프리 방법을 능가하는 경우 이해

---

## 목차

1. [모델 프리 vs 모델 기반 강화학습](#1-모델-프리-vs-모델-기반-강화학습)
2. [Dyna 아키텍처](#2-dyna-아키텍처)
3. [세계 모델 학습](#3-세계-모델-학습)
4. [모델 기반 정책 최적화(MBPO)](#4-모델-기반-정책-최적화mbpo)
5. [MuZero: 알려진 모델 없이 계획하기](#5-muzero-알려진-모델-없이-계획하기)
6. [Dreamer: 연속 제어를 위한 세계 모델](#6-dreamer-연속-제어를-위한-세계-모델)
7. [연습 문제](#7-연습-문제)

---

## 1. 모델 프리 vs 모델 기반 강화학습

### 1.1 비교

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

### 1.2 트레이드오프

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

## 2. Dyna 아키텍처

### 2.1 Dyna-Q 알고리즘

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

### 2.2 Dyna-Q 구현

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
        # 1단계: 직접 강화학습 업데이트 — 항상 하나의 실제 TD 업데이트를 먼저 적용하여
        # 모델을 사용하기 전에 Q-테이블이 실제 전이(transition)를 반영하도록 한다
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

        # 2단계: 모델 학습 — 관찰된 결정론적 전이를 저장한다;
        # 이 단순한 테이블형 모델은 방문한 (s,a) 쌍에 대해서는 정확하지만
        # 방문하지 않은 상태로는 일반화할 수 없다 (신경망 모델과 달리)
        self.model[(s, a)] = (r, s_next, done)
        if (s, a) not in self.visited:
            self.visited.append((s, a))

        # 3단계: 계획(n 시뮬레이션 업데이트) — 저장된 모델로 n번의 Q-업데이트를 수행하는 것은
        # n번의 실제 환경 스텝보다 훨씬 비용이 적게 든다;
        # n_planning이 클수록 실제 스텝당 더 많은 계산 비용으로 더 높은 샘플 효율성을 얻는다
        for _ in range(self.n_planning):
            # 방문한 쌍에서 균일하게 샘플링하는 이유: 최근 상태만이 아니라 알려진 상태 공간의
            # 모든 부분이 동등하게 업데이트되도록 보장한다
            idx = np.random.randint(len(self.visited))
            sim_s, sim_a = self.visited[idx]
            sim_r, sim_s_next, sim_done = self.model[(sim_s, sim_a)]

            # Q-learning update with simulated experience
            sim_target = sim_r if sim_done else sim_r + self.gamma * np.max(self.Q[sim_s_next])
            self.Q[sim_s, sim_a] += self.alpha * (sim_target - self.Q[sim_s, sim_a])
```

### 2.3 Dyna-Q+ (탐색 보너스)

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

## 3. 세계 모델 학습

### 3.1 신경망 동역학 모델(Neural Network Dynamics Model)

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
        # 5개 모델로 앙상블을 구성하는 이유: 단일 모델은 "이 영역을 봤고 예측 가능하다"와
        # "이 영역을 전혀 보지 못했다"를 구분할 수 없다 — 모델 간 불일치(disagreement)는
        # 베이지안 추론 없이 인식론적 불확실성(epistemic uncertainty)에 대한 저렴한 대리 지표가 된다
        self.models = nn.ModuleList([
            DynamicsModel(state_dim, action_dim, hidden_dim)
            for _ in range(n_models)
        ])

    def forward(self, state, action):
        predictions = [model(state, action) for model in self.models]
        next_states = torch.stack([p[0] for p in predictions])
        rewards = torch.stack([p[1] for p in predictions])

        # 평균 예측 — 평균화는 단일 모델에 비해 분산을 줄인다
        mean_next_state = next_states.mean(dim=0)
        mean_reward = rewards.mean(dim=0)

        # 표준편차(std)를 불확실성으로 사용하는 이유: 모델 간 높은 불일치는 우리가
        # 학습 데이터가 거의 없는 상태 공간 영역에 있음을 나타낸다;
        # MBPO는 이를 활용하여 오류가 누적되기 전에 롤아웃을 종료한다
        uncertainty = next_states.std(dim=0).mean(dim=-1)

        return mean_next_state, mean_reward, uncertainty
```

### 3.2 모델 학습

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
                # 각 모델이 다른 부트스트랩 샘플(bootstrap sample)로 학습되는 이유:
                # 모든 모델이 동일한 데이터를 보면 유사한 예측으로 수렴하여
                # 앙상블의 목적을 상실한다; 랜덤 샘플링이 모델 가중치의 다양성을 도입한다
                states, actions, rewards, next_states = \
                    replay_buffer.sample(batch_size)

                pred_next_states, pred_rewards = model(states, actions)

                # 상태 손실과 보상 손실을 분리하는 이유: 서로 다른 스케일과 의미를 가진다;
                # 동등한 가중치로 함께 학습하는 것은 단순화로, 실제로 보상 예측이 더 쉬운 경우
                # (스칼라 vs 고차원 상태), 그 그래디언트가 지배적이 되어서는 안 된다
                state_loss = nn.MSELoss()(pred_next_states, next_states)
                reward_loss = nn.MSELoss()(pred_rewards, rewards)
                loss = state_loss + reward_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

### 3.3 모델 오류와 누적

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

## 4. 모델 기반 정책 최적화(MBPO)

### 4.1 MBPO 알고리즘

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

### 4.2 간소화된 MBPO 구현

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
        # 실제 상태에서 분기하는 이유: 실제로 관찰된 상태에서 시작하면 롤아웃이
        # 실제 분포에 고정된다; 오류는 순수 상상(imagination)에서 빠르게 누적되므로
        # 실제 상태 분기는 모델이 진짜 역학(dynamics)에서 벗어나는 "거리"를 제한한다
        states = self.env_buffer.sample_states(self.rollouts_per_step)

        for state in states:
            s = state.clone()
            for h in range(self.rollout_length):
                a = self.policy.select_action(s)
                s_next, r, uncertainty = self.ensemble(
                    s.unsqueeze(0), a.unsqueeze(0)
                )

                # 높은 불확실성에서 롤아웃을 중단하는 이유: 앙상블 불일치는 모델이
                # 학습 데이터 너머로 외삽(extrapolation)하고 있음을 나타낸다;
                # 롤아웃을 계속하면 정책의 가치 추정을 오염시키는 신뢰할 수 없는
                # 전이(transition)가 추가된다
                if uncertainty.item() > 0.5:
                    break

                # 모델 전이에 done=False로 표시하는 이유: 모델은 종료 조건을
                # 신뢰할 수 있게 시뮬레이션하지 않는다; 모든 시뮬레이션 전이를
                # 비종료로 처리하면 인위적인 에피소드 끝에서 잘못된 가치 절단을 방지한다
                self.model_buffer.add(
                    s.numpy(), a.numpy(), r.item(),
                    s_next.squeeze(0).detach().numpy(), False
                )
                s = s_next.squeeze(0).detach()
```

---

## 5. MuZero: 알려진 모델 없이 계획하기

### 5.1 MuZero 아키텍처

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

### 5.2 MuZero 계획(MCTS)

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

## 6. Dreamer: 연속 제어를 위한 세계 모델

### 6.1 Dreamer 아키텍처

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

### 6.2 상상 기반 정책 학습(Imagination-Based Policy Learning)

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

## 7. 연습 문제

### 연습 1: GridWorld에서 Dyna-Q
Dyna-Q를 구현하고 서로 다른 계획 단계(n=0, 5, 50)에서 성능을 비교하세요.

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

### 연습 2: 신경망 동역학 모델
CartPole에서 신경망 동역학 모델을 학습하고 예측 정확도를 평가하세요.

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

### 연습 3: 샘플 효율성 비교
연속 제어 태스크에서 모델 프리 SAC와 MBPO를 비교하세요.

```python
# Use HalfCheetah-v4 or Pendulum-v1
# Plot: reward vs environment steps
# Expected: MBPO reaches good performance in ~10x fewer steps
# But: MBPO has higher wall-clock time per step (model training + planning)
```

---

## 요약

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

**핵심 요점:**
- 모델 기반 강화학습은 계산을 샘플 효율성과 교환
- 앙상블 모델은 불확실성 추정을 제공
- 짧은 롤아웃은 누적 모델 오류를 완화
- MuZero: 학습된 잠재 공간에서 계획 (관측 예측 불필요)
- Dreamer: 전체 정책 학습을 상상 속에서 수행

---

## 참고 문헌

- Sutton & Barto Ch. 8: "Planning and Learning with Tabular Methods"
- [MBPO Paper](https://arxiv.org/abs/1906.08253) — Janner et al. 2019
- [MuZero Paper](https://arxiv.org/abs/1911.08265) — Schrittwieser et al. 2020
- [DreamerV3 Paper](https://arxiv.org/abs/2301.04104) — Hafner et al. 2023
- [Spinning Up: Model-Based Methods](https://spinningup.openai.com/)
