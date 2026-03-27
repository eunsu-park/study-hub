[이전: World Models](./25_World_Models.md)

---

# 26. 모방 학습

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 행동 복제에서 대화형 방법까지의 모방 학습 스펙트럼 설명
2. 전문가 쿼리를 통한 반복적 데이터셋 집계를 위한 DAgger 구현
3. GAIL을 넘어서는 적대적 모방 학습 방법 구축
4. 관찰만의 모방 학습 (비디오로부터 학습) 이해
5. 샘플 효율성과 일반화에 대해 모방 접근법 비교

---

## 목차

1. [모방 학습 기초](#1-모방-학습-기초)
2. [행동 복제 심층 분석](#2-행동-복제-심층-분석)
3. [DAgger와 대화형 방법](#3-dagger와-대화형-방법)
4. [적대적 모방 학습](#4-적대적-모방-학습)
5. [관찰만의 모방](#5-관찰만의-모방)
6. [소수 샷 및 원샷 모방](#6-소수-샷-및-원샷-모방)
7. [실전 모방 파이프라인](#7-실전-모방-파이프라인)
8. [연습문제](#8-연습문제)

---

## 1. 모방 학습 기초

### 1.1 모방 스펙트럼

```
모방 학습 방법 (필요한 전문가 접근 순서):

1. 행동 복제 (BC)
   전문가 접근: 오프라인 시연 데이터셋
   방법: 지도 학습 π(a|s) = argmax P(a|s)
   장점: 간단, 환경 불필요
   단점: 누적 오차, 분포 이동

2. DAgger (Dataset Aggregation)
   전문가 접근: 학습 중 전문가에게 쿼리 가능
   방법: 학습자가 방문하는 곳에서 반복적으로 데이터 수집
   장점: 분포 이동 처리
   단점: 온라인 전문가 접근 필요

3. IRL / GAIL
   전문가 접근: 시연만
   방법: 보상 학습 / 점유 측도 매칭
   장점: 시연 너머로 일반화 가능
   단점: 환경 접근 필요

4. 관찰만의 IL
   전문가 접근: 비디오만 (행동 없음!)
   방법: 상태 매핑 또는 역역학 학습
   장점: 많은 "시연"이 무료로 이용 가능
   단점: 행동 추론이 어려움
```

### 1.2 문제 정식화

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard notation:
# Expert demonstrations: D_E = {(s_i, a_i)}_{i=1}^N
# Expert policy: π_E(a|s)
# Learner policy: π_θ(a|s)
# Environment dynamics: T(s'|s,a)

# Objective varies by method:
# BC:     min_θ E_{(s,a)~D_E} [L(π_θ(s), a)]
# DAgger: min_θ E_{s~d^{π_θ}} [L(π_θ(s), π_E(s))]
# GAIL:   min_π max_D E_E[log D(s,a)] + E_π[log(1-D(s,a))]
```

---

## 2. 행동 복제 심층 분석

### 2.1 고급 BC 아키텍처

```python
class TransformerBCPolicy(nn.Module):
    """Transformer-based behavioral cloning with history."""

    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 n_heads=4, n_layers=2, context_length=10):
        super().__init__()
        self.context_length = context_length

        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.pos_embed = nn.Embedding(context_length, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=4*hidden_dim, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state_history):
        """
        Args:
            state_history: (batch, T, state_dim) last T states
        Returns:
            action: (batch, action_dim) predicted action
        """
        T = state_history.shape[1]
        positions = torch.arange(T, device=state_history.device)

        x = self.state_embed(state_history) + self.pos_embed(positions)

        mask = torch.triu(
            torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1
        )
        x = self.transformer(x, mask=mask)

        return self.action_head(x[:, -1])  # Predict from last position


class GaussianBCPolicy(nn.Module):
    """BC with Gaussian action prediction (for continuous actions)."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.net(state)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-5, 2)
        return mean, log_std

    def loss(self, states, expert_actions):
        """Negative log-likelihood loss."""
        mean, log_std = self.forward(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return -dist.log_prob(expert_actions).sum(dim=-1).mean()
```

### 2.2 BC를 위한 데이터 증강

```python
class BCDataAugmentation:
    """Data augmentation to improve BC robustness."""

    def __init__(self, noise_std=0.01, action_noise_std=0.0):
        self.noise_std = noise_std
        self.action_noise_std = action_noise_std

    def augment_state(self, states):
        """Add Gaussian noise to states."""
        noise = torch.randn_like(states) * self.noise_std
        return states + noise

    def augment_continuous_actions(self, actions):
        """Small noise on continuous actions."""
        noise = torch.randn_like(actions) * self.action_noise_std
        return actions + noise

    def temporal_augment(self, trajectory, p_drop=0.1):
        """Randomly drop/duplicate timesteps."""
        augmented = []
        for t in range(len(trajectory)):
            if np.random.random() > p_drop:
                augmented.append(trajectory[t])
                if np.random.random() < p_drop:
                    augmented.append(trajectory[t])  # duplicate
        return augmented
```

---

## 3. DAgger와 대화형 방법

### 3.1 DAgger 알고리즘

```
DAgger (Dataset Aggregation, Ross et al., 2011):

1. D ← 전문가 시연으로 초기화
2. D에서 π₁ 학습 (행동 복제)
3. i = 1, 2, ..., N에 대해:
   a. 환경에서 π_i 실행, 상태 S_i 수집
   b. S_i에서 전문가 π_E에 쿼리하여 레이블 획득
   c. D ← D ∪ {(s, π_E(s)) for s in S_i}
   d. D에서 π_{i+1} 학습

핵심 통찰: 학습자가 방문한 상태에서 학습하므로
(전문가만의 상태가 아닌), DAgger는 분포 이동을 처리합니다!
```

### 3.2 DAgger 구현

```python
class DAggerTrainer:
    """DAgger: Dataset Aggregation for imitation learning."""

    def __init__(self, policy, expert_fn, env, lr=1e-3,
                 mixing_decay=0.99):
        self.policy = policy
        self.expert = expert_fn  # Can query expert for any state
        self.env = env
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.dataset = []  # Aggregated dataset
        self.beta = 1.0  # Mixing coefficient (start with expert)
        self.mixing_decay = mixing_decay

    def collect_episode(self, max_steps=200):
        """Collect episode, mixing learner and expert actions."""
        state, _ = self.env.reset()
        episode_data = []

        for step in range(max_steps):
            # Mix learner and expert actions
            if np.random.random() < self.beta:
                action = self.expert(state)  # Expert action
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action = self.policy(state_t).squeeze(0).numpy()

            # ALWAYS label with expert action (regardless of who acted)
            expert_action = self.expert(state)
            episode_data.append((state.copy(), expert_action.copy()))

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            state = next_state

            if terminated or truncated:
                break

        return episode_data

    def train(self, n_iterations=50, episodes_per_iter=10,
              train_epochs=10, batch_size=64):
        """Full DAgger training loop."""
        performance_history = []

        for iteration in range(n_iterations):
            # Collect data with current policy
            new_data = []
            for _ in range(episodes_per_iter):
                episode = self.collect_episode()
                new_data.extend(episode)

            self.dataset.extend(new_data)

            # Train on aggregated dataset
            states = torch.FloatTensor([d[0] for d in self.dataset])
            actions = torch.FloatTensor([d[1] for d in self.dataset])

            for epoch in range(train_epochs):
                indices = torch.randperm(len(self.dataset))
                for i in range(0, len(self.dataset), batch_size):
                    batch_idx = indices[i:i+batch_size]
                    s_batch = states[batch_idx]
                    a_batch = actions[batch_idx]

                    pred_actions = self.policy(s_batch)
                    loss = F.mse_loss(pred_actions, a_batch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Decay mixing coefficient
            self.beta *= self.mixing_decay

            # Evaluate
            avg_return = self.evaluate()
            performance_history.append(avg_return)
            print(f"Iter {iteration+1}: Return={avg_return:.1f}, "
                  f"Dataset={len(self.dataset)}, Beta={self.beta:.3f}")

        return performance_history

    def evaluate(self, n_episodes=10):
        """Evaluate current policy without expert."""
        returns = []
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            ep_return = 0
            done = False

            while not done:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action = self.policy(state_t).squeeze(0).numpy()

                state, reward, terminated, truncated, _ = self.env.step(action)
                ep_return += reward
                done = terminated or truncated

            returns.append(ep_return)

        return np.mean(returns)
```

### 3.3 DAgger 변형

```
DAgger 변형:

1. SafeDAgger: 학습자가 불확실할 때만 전문가에게 쿼리
   -> 전문가 쿼리를 크게 줄임

2. EnsembleDAgger: 앙상블 불일치를 불확실성으로 사용
   -> 앙상블이 불일치할 때 전문가에게 쿼리

3. ThriftyDAgger: 예산 제한 전문가 쿼리
   -> 총 전문가 쿼리 수가 고정

4. HG-DAgger: 인간이 제어하는 DAgger
   -> 인간이 개입 시기를 결정

5. LazyDAgger: 안전에 기반하여 전문가와 학습자 간 전환
```

---

## 4. 적대적 모방 학습

### 4.1 GAIL을 넘어서

```python
class AIRL(nn.Module):
    """
    Adversarial Inverse RL (AIRL).
    Unlike GAIL, AIRL recovers a transferable reward function.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, gamma=0.99):
        super().__init__()
        self.gamma = gamma

        # Reward function: r(s,a)
        self.reward = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Shaping function: Φ(s) for PBRS
        self.shaping = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, next_state, done):
        """
        AIRL discriminator: f(s,a,s') = r(s) + γΦ(s') - Φ(s)
        """
        r = self.reward(state)
        phi_s = self.shaping(state)
        phi_s_next = self.shaping(next_state)

        f = r + self.gamma * (1 - done.float().unsqueeze(1)) * phi_s_next - phi_s
        return f

    def get_reward(self, state):
        """Get the disentangled reward (transferable!)."""
        return self.reward(state)
```

### 4.2 ValueDICE

```python
class ValueDICE:
    """
    ValueDICE: Offline imitation learning without environment interaction.
    Estimates the divergence between expert and offline policy.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4):
        # Discriminator (nu function)
        self.nu = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.optimizer = torch.optim.Adam(self.nu.parameters(), lr=lr)

    def compute_loss(self, expert_states, expert_actions,
                     offline_states, offline_actions, offline_next_states,
                     gamma=0.99):
        """ValueDICE loss for offline imitation."""
        # Expert nu values
        expert_input = torch.cat([expert_states, expert_actions], dim=-1)
        expert_nu = self.nu(expert_input).mean()

        # Offline data terms
        offline_input = torch.cat([offline_states, offline_actions], dim=-1)
        current_nu = self.nu(offline_input)

        # This implements the Fenchel dual of f-divergence
        loss = expert_nu - torch.logsumexp(current_nu.squeeze(), dim=0)

        return loss
```

---

## 5. 관찰만의 모방

### 5.1 비디오로부터 학습

```
비디오를 보고 학습할 수 있을까? (행동 레이블 없음!)

도전: 비디오는 무엇이 일어났는지 보여주지만, 어떻게 했는지는 보여주지 않음 (행동 없음).

접근법:
1. 역역학: 자신의 경험에서 f(s_t, s_{t+1}) -> a_t 학습
   그 다음 전문가 비디오의 행동을 추론

2. 상태 매칭: 상태 방문 분포를 매칭
   행동이 전혀 필요 없음!

3. 시간적 정렬: 자신의 경험과
   전문가 비디오 사이의 대응 학습
```

### 5.2 행동 추론을 위한 역역학

```python
class InverseDynamicsModel(nn.Module):
    """Predict action from state transition."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=-1)
        return self.net(x)


class ObservationOnlyIL:
    """Imitation learning from observation-only demonstrations."""

    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.inverse_model = InverseDynamicsModel(state_dim, action_dim)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.inv_optimizer = torch.optim.Adam(
            self.inverse_model.parameters(), lr=lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr
        )

    def train_inverse_model(self, own_transitions, epochs=50):
        """Train inverse dynamics on agent's own experience."""
        states = torch.FloatTensor(own_transitions['states'])
        next_states = torch.FloatTensor(own_transitions['next_states'])
        actions = torch.FloatTensor(own_transitions['actions'])

        for epoch in range(epochs):
            pred_actions = self.inverse_model(states, next_states)
            loss = F.mse_loss(pred_actions, actions)

            self.inv_optimizer.zero_grad()
            loss.backward()
            self.inv_optimizer.step()

    def infer_and_train(self, expert_states):
        """Infer actions from expert states, then BC."""
        states = torch.FloatTensor(expert_states[:-1])
        next_states = torch.FloatTensor(expert_states[1:])

        # Infer expert actions
        with torch.no_grad():
            inferred_actions = self.inverse_model(states, next_states)

        # Behavioral cloning on inferred actions
        pred_actions = self.policy(states)
        loss = F.mse_loss(pred_actions, inferred_actions)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()
```

---

## 6. 소수 샷 및 원샷 모방

### 6.1 메타 모방 학습

```
목표: 단 하나의 시연만으로 모방 학습!

접근법:
  많은 과제에서 학습, 각각 소수의 시연으로
  테스트 시: 새로운 과제의 시연 하나가 주어지면, 모방

모방을 위한 메타 학습:
  메타 학습 중:
    과제 i: K개 시연이 주어지면 모방 학습
    손실: 정책이 과제 i에서 얼마나 잘 수행하는가

  메타 테스트 중:
    새로운 과제: 시연 1개가 주어지면 적응하고 수행
```

### 6.2 과제 조건부 정책

```python
class OneShotiImitationPolicy(nn.Module):
    """Policy that takes a demonstration as input."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Demo encoder: process the demonstration sequence
        self.demo_encoder = nn.GRU(
            state_dim + action_dim, hidden_dim, batch_first=True
        )

        # Policy: conditioned on demo embedding + current state
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def encode_demo(self, demo_states, demo_actions):
        """Encode demonstration into a task embedding."""
        demo_input = torch.cat([demo_states, demo_actions], dim=-1)
        _, h_n = self.demo_encoder(demo_input.unsqueeze(0))
        return h_n.squeeze(0)  # (hidden_dim,)

    def forward(self, state, demo_embedding):
        """Predict action given current state and task embedding."""
        x = torch.cat([state, demo_embedding.expand(len(state), -1)], dim=-1)
        return self.policy(x)
```

---

## 7. 실전 모방 파이프라인

### 7.1 언제 무엇을 사용할 것인가

```
결정 가이드:

전문가에게 자유롭게 쿼리 가능?
├── 예 → DAgger (최고의 이론적 보장)
│         또는: 전문가 시간이 제한적이면 HG-DAgger
└── 아니오
    ├── 행동 레이블이 있는가?
    │   ├── 작은 데이터셋 (< 1000 시연)
    │   │   └── 데이터 증강 + 앙상블로 BC
    │   └── 큰 데이터셋 (> 10000 시연)
    │       └── BC가 잘 작동; 환경 이용 가능하면 GAIL
    └── 행동 레이블 없음 (비디오만)
        ├── 자체 로봇 데이터가 있는가?
        │   └── 역역학 + BC
        └── 자체 데이터 없음
            └── 매우 어려움. 상태 매칭 또는 수동 보상 설계
```

### 7.2 평가 프로토콜

```python
def evaluate_imitation(agent, env, expert_fn, n_episodes=100):
    """Comprehensive evaluation of imitation learning agent."""
    agent_returns = []
    expert_returns = []
    action_errors = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        agent_return = 0
        expert_return = 0
        ep_action_errors = []
        done = False

        while not done:
            agent_action = agent.get_action(state)
            expert_action = expert_fn(state)

            action_error = np.linalg.norm(agent_action - expert_action)
            ep_action_errors.append(action_error)

            state, reward, terminated, truncated, _ = env.step(agent_action)
            agent_return += reward
            done = terminated or truncated

        agent_returns.append(agent_return)
        action_errors.append(np.mean(ep_action_errors))

    print(f"Agent return:  {np.mean(agent_returns):.1f} +/- {np.std(agent_returns):.1f}")
    print(f"Action error:  {np.mean(action_errors):.4f}")
    print(f"Expert return: {np.mean(expert_returns):.1f} (reference)")
    print(f"Normalized:    {np.mean(agent_returns)/np.mean(expert_returns)*100:.1f}%")
```

---

## 8. 연습문제

### 연습문제 1: BC vs DAgger 비교

행동 복제와 DAgger를 비교하세요:
1. CartPole을 위한 전문가 정책 생성 (PPO로 ~500 보상까지 학습)
2. BC를 위해 50개 전문가 시연 수집
3. 같은 총 전문가 쿼리 (50 에피소드)로 DAgger 구현
4. 비교: 모든 데이터를 사전에 사용하는 BC vs 반복적 수집의 DAgger
5. 두 방법의 전문가 쿼리 대비 성능 그래프

### 연습문제 2: 불확실성이 있는 Gaussian BC

불확실성 추정이 있는 BC를 구축하세요:
1. 평균과 표준편차를 예측하는 GaussianBCPolicy 구현
2. 전문가 시연으로 학습
3. 불확실성 시각화: 익숙하지 않은 상태에서 높은 불확실성
4. 능동적 쿼리를 위해 불확실성 사용 (불확실한 상태에서 전문가에게 쿼리)
5. 분포 외 상태에서 결정적 BC와 비교

### 연습문제 3: 관찰만의 모방

전문가 비디오에서 학습 (행동 레이블 없음):
1. 에이전트의 랜덤 탐색에서 역역학 모델 학습
2. 전문가 상태 궤적 기록 (행동 없이)
3. 역역학으로 행동 추론 후 BC
4. 행동 레이블이 있는 표준 BC와 비교
5. 측정: 역모델 정확도가 최종 정책에 미치는 영향

### 연습문제 4: 예산 제약이 있는 DAgger

예산 인식 DAgger를 구현하세요:
1. 총 100개 전문가 쿼리 고정 예산의 DAgger
2. 전략 1: 반복 간 균등 쿼리
3. 전략 2: 초기 반복에서 더 많이 쿼리
4. 전략 3: 앙상블 불확실성이 높을 때만 쿼리
5. 모든 전략의 최종 정책 성능 비교

### 연습문제 5: 다중 과제 모방

과제 조건부 모방 학습자를 구축하세요:
1. 5개의 간단한 내비게이션 과제 생성 (다른 목표 위치)
2. 과제당 10개 시연 수집
3. 시연 조건부 정책 학습
4. 보류된 과제에서 테스트 (새로운 목표 위치)
5. 비교: (a) 과제별 별도 BC, (b) 모든 데이터에서 단일 BC

---

*26강 끝*
