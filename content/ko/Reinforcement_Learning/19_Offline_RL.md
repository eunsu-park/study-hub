[이전: 분포적 RL](./18_Distributional_RL.md)

---

# 19. 오프라인 강화학습 (Offline RL)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 오프라인 RL 문제 설정과 분포 이동(Distribution Shift) 문제를 설명할 수 있습니다
2. 하한 Q-값 추정을 사용하는 Conservative Q-Learning (CQL)을 구현할 수 있습니다
3. RL을 시퀀스 모델링으로 변환하는 Decision Transformer를 구축할 수 있습니다
4. 행동 복제(Behavior Cloning)의 한계와 전체 오프라인 RL이 필요한 경우를 이해할 수 있습니다
5. D4RL 벤치마크에서 BCQ, BEAR, CQL, Decision Transformer를 비교할 수 있습니다

---

## 목차

1. [오프라인 RL 문제](#1-오프라인-rl-문제)
2. [분포 이동과 외삽 오류](#2-분포-이동과-외삽-오류)
3. [행동 복제 재검토](#3-행동-복제-재검토)
4. [Conservative Q-Learning (CQL)](#4-conservative-q-learning-cql)
5. [Decision Transformer](#5-decision-transformer)
6. [기타 오프라인 방법: BCQ와 BEAR](#6-기타-오프라인-방법-bcq와-bear)
7. [실용 가이드와 D4RL 벤치마크](#7-실용-가이드와-d4rl-벤치마크)
8. [연습 문제](#8-연습-문제)

---

## 1. 오프라인 RL 문제

### 1.1 온라인 vs 오프라인 RL

```
온라인 RL:
  에이전트 ──▶ 환경 ──▶ 에이전트 ──▶ 환경 ──▶ ...
  (행동)     (관찰)     (학습)     (더 탐색)
  + 자유롭게 탐색 가능
  - 실제 세계에서 비용/위험 (로봇공학, 의료, 자율주행)

오프라인 RL (배치 RL):
  고정 데이터셋 D = {(s, a, r, s')₁, (s, a, r, s')₂, ...}
        │
        ▼
  추가 상호작용 없이 D로부터 정책 π를 학습
  + 안전: 위험한 탐색 없음
  + 기존 로그 데이터 활용 (병원 기록, 운전 로그)
  - 분포 이동: 데이터셋 밖의 행동을 질의할 수 없음
```

### 1.2 오프라인 RL이 어려운 이유

근본적인 도전은 **분포 이동**입니다: 학습된 정책이 데이터셋에 포함되지 않은 상태-행동 쌍을 방문할 수 있습니다.

```python
import numpy as np

def demonstrate_distribution_shift():
    """순진한 오프폴리시 학습이 오프라인에서 실패하는 이유를 보여줍니다."""
    # 행동 정책이 action=0 근처에서 데이터를 수집한다고 가정
    n_samples = 1000
    states = np.random.uniform(-1, 1, n_samples)
    actions = np.random.normal(0, 0.3, n_samples)  # 0 근처에 집중
    rewards = -(states ** 2) - (actions - states) ** 2  # 최적: a = s

    print("데이터셋 통계:")
    print(f"  행동: 평균={actions.mean():.2f}, 표준편차={actions.std():.2f}")
    print(f"  행동 범위: [{actions.min():.2f}, {actions.max():.2f}]")
    print()
    print("문제: Q(s, a=5)는 데이터셋에서 관찰된 적이 없습니다.")
    print("Q-러닝이 임의로 높은 값을 외삽할 수 있습니다!")
    print("학습된 정책은 이 분포 밖 행동을 선택합니다.")

demonstrate_distribution_shift()
```

### 1.3 외삽 오류

```
Q-값 공간:

  Q(s,a)
    ▲        / 외삽됨 (잘못됨!)
    │      //
    │    //
    │  //      분포 내              분포 밖
    │ /   ●●●●●●●●●●●              ???
    │/    (데이터가 이 범위를 커버)   (데이터 없음)
    └─────────────────────────────▶ 행동
         a_min        a_max

  표준 DQN은 argmax Q 선택 → 분포 밖 행동 선택 → 재앙
```

---

## 2. 분포 이동과 외삽 오류

### 2.1 형식적 분석

π_β를 행동 정책(데이터를 수집한 정책), π를 학습된 정책이라 합니다.

```
오프라인 RL 목표:
  max_π E_{s~d^π} [Σ γᵗ r(sₜ, π(sₜ))]

하지만 d^{π_β} (행동 정책의 상태 분포) 데이터만 가지고 있습니다.

π ≠ π_β일 때:
  - π가 d^{π_β}(s) ≈ 0인 상태를 방문 → 학습 데이터 없음
  - 관찰되지 않은 (s, a)에 대한 Q(s, a)는 신뢰할 수 없음
  - 다단계 롤아웃에서 오류가 복합됨
```

### 2.2 OOD 정도 측정

```python
from sklearn.neighbors import KernelDensity

def measure_ood_degree(dataset_actions, policy_actions, bandwidth=0.1):
    """
    행동 데이터셋에 대한 정책 행동의 분포 밖 정도를 추정합니다.
    """
    # 데이터셋 행동에 KDE 적합
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(dataset_actions.reshape(-1, 1))

    # 정책 행동 점수 매김
    log_density = kde.score_samples(policy_actions.reshape(-1, 1))
    density = np.exp(log_density)

    print(f"정책 행동에서의 데이터셋 행동 밀도:")
    print(f"  평균 밀도: {density.mean():.4f}")
    print(f"  최소 밀도: {density.min():.6f}")
    print(f"  임계값 이하 % (OOD): "
          f"{(density < 0.01).mean()*100:.1f}%")

    return density
```

### 2.3 해결법 분류

```
오프라인 RL 방법:

1. 정책 제약 방법
   ├── BCQ (Batch-Constrained Q-learning)
   │     └── 데이터셋에 "가까운" 행동만 고려
   ├── BEAR (Bootstrapping Error Accumulation Reduction)
   │     └── 행동 분포에 대한 MMD 제약
   └── TD3+BC
         └── 간단한 행동 복제 정규화

2. 가치 비관주의 방법
   ├── CQL (Conservative Q-Learning)
   │     └── OOD 행동의 Q-값을 낮춤
   └── PBRL (Pessimistic Bellman Reinforcement Learning)
         └── Q의 하한 신뢰 구간

3. 모델 기반 오프라인 RL
   ├── MOPO (Model-based Offline Policy Optimization)
   │     └── 학습된 동역학의 불확실성에 페널티
   └── MOReL
         └── 비관적 MDP 구축

4. 시퀀스 모델링
   └── Decision Transformer
         └── RL을 조건부 시퀀스 생성으로 변환
```

---

## 3. 행동 복제 재검토

### 3.1 BC가 충분한 경우

행동 복제(전문가 행동에 대한 지도 학습)는 가장 간단한 접근법입니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviorCloningPolicy(nn.Module):
    """지도 학습을 통한 간단한 행동 복제."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)

    def get_action(self, state):
        with torch.no_grad():
            return self.forward(torch.FloatTensor(state)).numpy()


def train_bc(dataset, state_dim, action_dim, epochs=100, batch_size=256):
    """행동 복제 정책을 학습합니다."""
    policy = BehaviorCloningPolicy(state_dim, action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    states = torch.FloatTensor(dataset['states'])
    actions = torch.FloatTensor(dataset['actions'])
    n = len(states)

    for epoch in range(epochs):
        indices = torch.randperm(n)
        total_loss = 0

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            s_batch = states[batch_idx]
            a_batch = actions[batch_idx]

            predicted = policy(s_batch)
            loss = F.mse_loss(predicted, a_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / (n // batch_size)
            print(f"에포크 {epoch+1}, BC 손실: {avg_loss:.4f}")

    return policy
```

### 3.2 BC의 한계: 오류 복합

```
순차적 의사결정에서 BC의 문제:

단계 1: 작은 오류 ε → 약간 벗어남
단계 2: 이제 낯선 상태 → 오류가 2ε로 성장
단계 3: 더 벗어남 → 오류 3ε
...
단계 T: 오류 ~ T·ε  (선형 복합!)

전문가 궤적:   ● → ● → ● → ● → ● → ● → 목표
BC 궤적:       ● → ●↗ → ●↗↗ → ???  → ???  → 충돌

DAgger (Dataset Aggregation)는 학습된 정책이 방문하는 상태에서
반복적으로 전문가에게 질의하여 이를 해결합니다.
하지만 오프라인 RL에서는 전문가에게 질의할 수 없습니다!
```

### 3.3 BC가 충분한 경우 vs 오프라인 RL이 필요한 경우

| 시나리오 | BC 충분? | 이유 |
|----------|---------|------|
| 전문가 전용 데이터, 짧은 시간 지평 | 예 | 낮은 복합 오류 |
| 전문가 전용 데이터, 긴 시간 지평 | 아마도 | 가능하면 DAgger 사용 |
| 혼합 품질 데이터 | 아니오 | BC가 나쁜+좋은 시연을 평균냄 |
| 차선책 데이터 | 아니오 | BC는 데이터 품질만 맞출 수 있음 |
| 데이터 품질을 초과해야 함 | 아니오 | 가치 기반 오프라인 RL 필요 |

---

## 4. Conservative Q-Learning (CQL)

### 4.1 CQL 아이디어

CQL은 분포 밖 행동의 Q-값을 낮추고 분포 내 행동의 Q-값을 유지하는 정규화를 추가합니다.

```
표준 Q-러닝:
  최소화: E_{(s,a,r,s')~D} [(Q(s,a) - (r + γ max_a' Q(s',a')))²]

CQL 추가:
  + α · E_{s~D} [log Σ_a exp(Q(s,a))]     ← 모든 Q(s,a) 낮춤
  - α · E_{(s,a)~D} [Q(s,a)]               ← 데이터 내 행동의 Q 올림

순효과: OOD 행동의 Q-값이 보수적으로 낮아짐.
```

### 4.2 CQL 구현

```python
class QNetwork(nn.Module):
    """연속 상태-행동 공간을 위한 간단한 Q-네트워크."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    """연속 행동을 위한 가우시안 정책."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state, n_samples=1):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        if n_samples > 1:
            actions = dist.rsample((n_samples,)).permute(1, 0, 2)
            log_probs = dist.log_prob(actions.permute(1, 0, 2)).sum(-1).permute(1, 0)
            return actions, log_probs
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class CQLAgent:
    """오프라인 RL을 위한 Conservative Q-Learning."""

    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 lr=3e-4, gamma=0.99, cql_alpha=1.0, tau=0.005,
                 n_random_actions=10):
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.action_dim = action_dim
        self.n_random_actions = n_random_actions

        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim)

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr
        )

    def compute_cql_loss(self, states, actions):
        """CQL 정규화 항을 계산합니다."""
        batch_size = states.shape[0]

        # 균일하게 무작위 행동 샘플링
        random_actions = torch.FloatTensor(
            batch_size, self.n_random_actions, self.action_dim
        ).uniform_(-1, 1)

        # 현재 정책에서 행동 샘플링
        with torch.no_grad():
            policy_actions, policy_log_probs = self.policy.sample(
                states, n_samples=self.n_random_actions
            )

        # 무작위 및 정책 행동에 대한 Q-값
        random_q1 = self._get_q_batch(self.q1, states, random_actions)
        random_q2 = self._get_q_batch(self.q2, states, random_actions)
        policy_q1 = self._get_q_batch(self.q1, states, policy_actions)
        policy_q2 = self._get_q_batch(self.q2, states, policy_actions)

        # 샘플링된 행동에 대한 LogSumExp
        cat_q1 = torch.cat([random_q1, policy_q1], dim=1)
        cat_q2 = torch.cat([random_q2, policy_q2], dim=1)

        logsumexp_q1 = torch.logsumexp(cat_q1, dim=1).mean()
        logsumexp_q2 = torch.logsumexp(cat_q2, dim=1).mean()

        # 데이터 Q-값 차감
        data_q1 = self.q1(states, actions).mean()
        data_q2 = self.q2(states, actions).mean()

        cql_loss = (logsumexp_q1 - data_q1) + (logsumexp_q2 - data_q2)
        return cql_loss

    def _get_q_batch(self, q_net, states, action_samples):
        """상태당 여러 행동 샘플에 대한 Q-값을 얻습니다."""
        batch_size = states.shape[0]
        n_samples = action_samples.shape[1]

        states_expanded = states.unsqueeze(1).expand(-1, n_samples, -1)
        states_flat = states_expanded.reshape(-1, states.shape[-1])
        actions_flat = action_samples.reshape(-1, self.action_dim)

        q_values = q_net(states_flat, actions_flat)
        return q_values.reshape(batch_size, n_samples)

    def train_step(self, batch):
        """오프라인 배치에 대한 하나의 학습 단계."""
        states, actions, rewards, next_states, dones = [
            torch.FloatTensor(x) for x in batch
        ]

        # 표준 TD 손실
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            target_q = torch.min(
                self.target_q1(next_states, next_actions),
                self.target_q2(next_states, next_actions)
            )
            target_value = rewards + self.gamma * (1 - dones) * target_q.squeeze()

        current_q1 = self.q1(states, actions).squeeze()
        current_q2 = self.q2(states, actions).squeeze()

        td_loss = F.mse_loss(current_q1, target_value) + \
                  F.mse_loss(current_q2, target_value)

        # CQL 정규화
        cql_loss = self.compute_cql_loss(states, actions)

        # 총 비평가 손실
        critic_loss = td_loss + self.cql_alpha * cql_loss

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # 정책 업데이트
        new_actions, log_probs = self.policy.sample(states)
        q_new = torch.min(
            self.q1(states, new_actions),
            self.q2(states, new_actions)
        )
        policy_loss = -q_new.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 소프트 목표 업데이트
        for param, target_param in zip(self.q1.parameters(),
                                        self.target_q1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(self.q2.parameters(),
                                        self.target_q2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'policy_loss': policy_loss.item(),
        }
```

### 4.3 CQL 변형

```
CQL 변형과 정규화:

CQL(H):  α · (E_s[log Σ_a exp(Q(s,a))] - E_{(s,a)~D}[Q(s,a)])
          라그랑지안을 통한 자동 α: 목표 간격 유지를 위해 조정

CQL(ρ):  α · (E_{s,a~ρ}[Q(s,a)] - E_{(s,a)~D}[Q(s,a)])
          여기서 ρ는 균일, 정책, 또는 혼합일 수 있음

핵심 통찰: CQL은 Q^π의 하한을 증명적으로 학습합니다
  Q_CQL(s,a) ≤ Q^π(s,a) 모든 (s,a)에 대해 높은 확률로
  → 안전한 정책 개선 보장!
```

---

## 5. Decision Transformer

### 5.1 시퀀스 모델링으로서의 RL

Decision Transformer는 오프라인 RL을 시퀀스 예측 문제로 재구성합니다:

```
전통적 RL: Q/π 학습 → 가치 최대화를 통해 계획
Decision Transformer: 원하는 보상이 주어지면 행동을 예측하도록 학습

입력 시퀀스:
  (R̂₁, s₁, a₁, R̂₂, s₂, a₂, ..., R̂ₜ, sₜ, ???)
                                              ↑
                                    목표 보상 R̂ₜ가 주어지면 aₜ를 예측

R̂ₜ = "보상-투-고" = 시간 단계 t 이후의 원하는 누적 보상

테스트 시: R̂₁ = 높은 목표 설정 → 모델이 전문가 수준의 행동을 생성
```

### 5.2 아키텍처

```
                    보상-투-고    상태       행동
                    임베딩       임베딩     임베딩
                         │           │        │
                         ▼           ▼        ▼
타임스텝 ──▶ [R̂₁ s₁ a₁ | R̂₂ s₂ a₂ | R̂₃ s₃ ???]
임베딩                                       ↑
                         │                     │
                    ┌────▼─────────────────────┐
                    │    GPT-2 Transformer     │
                    │    (인과적 어텐션)         │
                    └──────────────────────────┘
                                    │
                                    ▼
                              예측된 a₃
```

### 5.3 구현

```python
class DecisionTransformer(nn.Module):
    """오프라인 RL을 위한 Decision Transformer."""

    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 n_heads=4, n_layers=3, max_length=20,
                 max_ep_length=1000, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # 각 모달리티의 임베딩
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.return_embed = nn.Linear(1, hidden_dim)

        # 위치 (타임스텝) 임베딩
        self.timestep_embed = nn.Embedding(max_ep_length, hidden_dim)

        # 레이어 정규화
        self.embed_ln = nn.LayerNorm(hidden_dim)

        # GPT-2 스타일 트랜스포머
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # 예측 헤드
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, returns_to_go, states, actions, timesteps):
        """
        Args:
            returns_to_go: (배치, T, 1)
            states: (배치, T, state_dim)
            actions: (배치, T, action_dim)
            timesteps: (배치, T)
        Returns:
            predicted_actions: (배치, T, action_dim)
        """
        batch_size, T = states.shape[0], states.shape[1]

        # 각 모달리티 임베딩
        state_embeddings = self.state_embed(states)
        action_embeddings = self.action_embed(actions)
        return_embeddings = self.return_embed(returns_to_go)

        # 타임스텝 임베딩 추가
        time_embeddings = self.timestep_embed(timesteps)
        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        return_embeddings += time_embeddings

        # 교차 배치: [R1, s1, a1, R2, s2, a2, ...]
        stacked = torch.stack(
            [return_embeddings, state_embeddings, action_embeddings],
            dim=2
        ).reshape(batch_size, 3 * T, self.hidden_dim)

        stacked = self.embed_ln(stacked)

        # 인과적 마스크
        causal_mask = torch.triu(
            torch.ones(3 * T, 3 * T, device=states.device) * float('-inf'),
            diagonal=1
        )

        # 트랜스포머 순전파
        output = self.transformer(stacked, mask=causal_mask)

        # 상태 위치 출력 추출 (각 타임스텝에서 행동 예측)
        state_outputs = output[:, 1::3, :]

        predicted_actions = self.predict_action(state_outputs)
        return predicted_actions
```

### 5.4 Decision Transformer 추론

```python
def evaluate_decision_transformer(model, env, target_return,
                                  max_ep_length=1000, context_length=20):
    """목표 보상을 가지고 Decision Transformer를 평가합니다."""
    model.eval()

    state, _ = env.reset()
    states = [state]
    actions = []
    returns_to_go = [target_return]
    timesteps = [0]

    episode_return = 0

    for t in range(max_ep_length):
        K = min(t + 1, context_length)

        rtg_input = torch.FloatTensor(returns_to_go[-K:]).reshape(1, K, 1)
        state_input = torch.FloatTensor(np.array(states[-K:])).unsqueeze(0)

        if len(actions) > 0:
            action_input = torch.FloatTensor(
                np.array(actions[-(K-1):] + [[0]*model.action_dim])
            ).unsqueeze(0)
        else:
            action_input = torch.zeros(1, K, model.action_dim)

        timestep_input = torch.LongTensor(timesteps[-K:]).unsqueeze(0)

        with torch.no_grad():
            predicted = model(rtg_input, state_input, action_input,
                            timestep_input)
            action = predicted[0, -1].numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += reward

        states.append(next_state)
        actions.append(action)
        returns_to_go.append(returns_to_go[-1] - reward)
        timesteps.append(t + 1)

        if done:
            break

    return episode_return
```

### 5.5 Decision Transformer의 강점과 한계

| 측면 | 강점 | 한계 |
|------|------|------|
| **단순성** | 벨만 업데이트 없음, 가치 추정 없음 | 궤적 연결 불가 |
| **조건화** | 원하는 보상 쉽게 지정 | 좋은 목표 보상을 알아야 함 |
| **안정성** | 부트스트래핑 없음 → 발산 없음 | 연결 작업에서 성능 저하 가능 |
| **확장성** | 트랜스포머 확장 법칙 활용 | 계산 비용 높음 |
| **멀티태스크** | 하나의 모델, 다양한 보상 목표 | 보상 스케일에 민감 |

---

## 6. 기타 오프라인 방법: BCQ와 BEAR

### 6.1 Batch-Constrained Q-Learning (BCQ)

BCQ는 학습된 정책을 행동 정책의 지지(support) 내에서만 행동을 선택하도록 제한합니다:

```python
class BCQAgent:
    """Batch-Constrained Q-learning 에이전트."""

    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 lr=3e-4, gamma=0.99, tau=0.005, phi=0.05):
        self.gamma = gamma
        self.tau = tau
        self.phi = phi  # 행동 교란 범위

        # 쌍둥이 Q-네트워크
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)

    def select_action(self, state, vae, perturbation, n_candidates=100):
        """BCQ 절차를 사용하여 행동을 선택합니다."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)

            # 1. VAE(행동 모델)에서 후보 행동 샘플링
            state_repeated = state_t.repeat(n_candidates, 1)
            candidates = vae.decode(state_repeated)

            # 2. 각 후보를 약간 교란
            perturbed = perturbation(state_repeated, candidates)

            # 3. Q-값이 가장 높은 것을 선택
            q1 = self.q1(state_repeated, perturbed)
            q2 = self.q2(state_repeated, perturbed)
            q = torch.min(q1, q2)

            best_idx = q.argmax(dim=0)
            return perturbed[best_idx].cpu().numpy()
```

### 6.2 BEAR: Bootstrapping Error Accumulation Reduction

BEAR는 Maximum Mean Discrepancy (MMD)를 사용하여 학습된 정책을 제약합니다:

```python
def mmd_loss(policy_actions, dataset_actions, kernel='laplacian', sigma=20.0):
    """
    정책 행동 분포와 데이터셋 행동 간의 MMD를 계산합니다.
    MMD^2(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    """
    if kernel == 'laplacian':
        def k(x, y):
            diff = (x.unsqueeze(1) - y.unsqueeze(0)).abs().sum(-1)
            return torch.exp(-diff / sigma)
    elif kernel == 'gaussian':
        def k(x, y):
            diff = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1)
            return torch.exp(-diff / (2 * sigma ** 2))

    kpp = k(policy_actions, policy_actions).mean()
    kdd = k(dataset_actions, dataset_actions).mean()
    kpd = k(policy_actions, dataset_actions).mean()

    mmd_squared = kpp + kdd - 2 * kpd
    return mmd_squared
```

### 6.3 D4RL에서의 방법 비교

```
D4RL 벤치마크 결과 (정규화 점수, 높을수록 좋음):

환경                  | BC   | BCQ  | BEAR | CQL  | DT
------------------------------------------------------
halfcheetah-medium    | 42.6 | 40.7 | 41.7 | 44.0 | 42.6
hopper-medium         | 52.9 | 54.5 | 52.1 | 58.5 | 67.6
walker2d-medium       | 75.3 | 53.1 | 59.1 | 72.5 | 74.0
halfcheetah-med-expert| 55.2 | 64.7 | 53.4 | 91.6 | 86.8
hopper-med-expert     | 52.5 | 110.9| 96.3 | 105.4| 107.6
walker2d-med-expert   | 107.5| 57.5 | 40.1 | 108.8| 108.1

주요 발견:
- CQL은 전반적으로 강력, 특히 혼합 품질 데이터에서
- DT는 데이터에 높은 보상 궤적이 있을 때 뛰어남
- BCQ/BEAR는 좋지만 혼합 데이터에서 어려움을 겪을 수 있음
- BC는 전문가 전용 데이터에서 놀랍도록 경쟁력 있음
```

---

## 7. 실용 가이드와 D4RL 벤치마크

### 7.1 알고리즘 선택 가이드

```
어떤 오프라인 RL 방법을 사용해야 하나?

데이터 품질?
├── 전문가 전용 → 행동 복제 (가장 간단, 종종 충분)
├── 혼합 품질 → CQL 또는 IQL (다봉 데이터를 잘 처리)
├── 차선책만 → CQL (데이터 이상으로 개선 가능)
└── 무작위 → 모든 방법에 어려움; 모델 기반 고려

주요 고려사항:
├── 궤적 연결 필요? → CQL/IQL (DT는 연결 불가)
├── 간단한 구현? → TD3+BC (TD3에 BC 정규화 추가)
├── 시퀀스 모델링 선호? → Decision Transformer
└── 연속 제어? → CQL-SAC 또는 IQL
```

### 7.2 일반적인 오프라인 RL 함정

```python
# 함정 1: 평균 성능으로만 평가
# 오프라인 RL은 분산이 클 수 있음; 신뢰 구간 보고
def evaluate_properly(agent, env, n_episodes=100):
    returns = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_return = 0
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)

    print(f"평균: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"중앙값: {np.median(returns):.1f}")
    print(f"최소/최대: {np.min(returns):.1f} / {np.max(returns):.1f}")
    return returns

# 함정 2: 하이퍼파라미터 민감성
# CQL의 alpha가 중요: 너무 높으면 → 과도하게 보수적, 너무 낮으면 → OOD 문제
# 해결: 라그랑지안을 통한 자동 alpha 조정

# 함정 3: 보상/상태 정규화 미실시
def normalize_dataset(dataset):
    state_mean = dataset['observations'].mean(axis=0)
    state_std = dataset['observations'].std(axis=0) + 1e-6
    dataset['observations'] = (dataset['observations'] - state_mean) / state_std
    dataset['next_observations'] = (
        dataset['next_observations'] - state_mean
    ) / state_std
    return dataset, state_mean, state_std
```

### 7.3 오프라인에서 온라인으로의 미세 조정

```python
def offline_to_online(agent, env, offline_steps=50000,
                      online_steps=50000, batch_size=256):
    """
    2단계 학습:
    1단계: 정적 데이터셋에서 오프라인 사전학습
    2단계: 환경 상호작용으로 온라인 미세 조정
    """
    # 1단계: 오프라인
    print("1단계: 오프라인 사전학습...")
    dataset = env.get_dataset()
    for step in range(offline_steps):
        batch = sample_batch(dataset, batch_size)
        agent.train_step(batch)

        if (step + 1) % 10000 == 0:
            scores = evaluate_properly(agent, env, n_episodes=10)
            print(f"  오프라인 스텝 {step+1}: {np.mean(scores):.1f}")

    # 2단계: 온라인 미세 조정
    print("2단계: 온라인 미세 조정...")
    from collections import deque
    replay_buffer = deque(maxlen=online_steps)

    state, _ = env.reset()
    for step in range(online_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.append((state, action, reward, next_state, float(done)))
        state = next_state if not done else env.reset()[0]

        if len(replay_buffer) >= batch_size:
            import random
            batch = random.sample(replay_buffer, batch_size)
            batch = [np.array(x) for x in zip(*batch)]
            agent.train_step(batch)

        if (step + 1) % 10000 == 0:
            scores = evaluate_properly(agent, env, n_episodes=10)
            print(f"  온라인 스텝 {step+1}: {np.mean(scores):.1f}")
```

---

## 8. 연습 문제

### 연습 1: 행동 복제 기준선 구현

사용자 정의 오프라인 데이터셋에서 BC 기준선을 구축하고 평가하세요:
1. 다양한 기술 수준(무작위, 중간, 전문가)에서 학습된 CartPole 에이전트로부터 데이터를 수집하세요
2. 각 데이터셋에서 BC를 학습하고 성능을 비교하세요
3. 복합 오류를 측정하세요: 다양한 시간 지평 길이에서 BC 성능 비교
4. 데이터셋 크기의 함수로 학습 곡선을 그리세요

### 연습 2: Conservative Q-Learning 처음부터 구현

이산 행동을 위한 CQL을 구현하세요:
1. 중간 품질 CartPole 정책으로부터 오프라인 데이터셋을 만드세요
2. 보수적 정규화를 가진 CQL을 구현하세요
3. CQL을 순진한 오프라인 DQN(제약 없음)과 비교하세요
4. CQL alpha 소거: alpha = {0.1, 0.5, 1.0, 5.0, 10.0}에 대한 성능 그래프
5. CQL이 보수적 Q-값을 학습함을 보이세요 (실제 Q와 비교)

### 연습 3: Decision Transformer

최소한의 Decision Transformer를 구축하세요:
1. 여러 CartPole 에피소드에서 궤적 데이터셋을 수집하세요
2. 보상-투-고 조건화를 가진 인과적 트랜스포머를 구현하세요
3. 다양한 목표 보상으로 학습하고 평가하세요
4. 더 높은 목표 보상이 더 나은 정책을 생산함을 보이세요 (데이터 품질 한계까지)
5. 어텐션 패턴을 시각화하세요: 모델이 무엇에 주의를 기울이는지?

### 연습 4: 오프라인 RL 데이터 품질 연구

데이터 품질이 오프라인 RL에 미치는 영향을 체계적으로 연구하세요:
1. 다양한 품질의 데이터셋 생성: 무작위, 25%, 50%, 75%, 전문가
2. 데이터셋 크기도 변경: 1K, 10K, 100K, 1M 전이
3. 각 조합에서 BC, CQL, DT를 학습하세요
4. 히트맵 생성: 행=품질, 열=데이터셋 크기, 셀=성능
5. 오프라인 RL이 BC를 능가하는 교차점을 식별하세요

### 연습 5: 궤적 연결 시연

가치 기반 오프라인 RL을 BC/DT와 구분하는 "연결" 능력을 시연하세요:
1. 2개의 방이 있는 간단한 2D 탐색 환경을 만드세요
2. 데이터 수집: 일부 궤적은 A→B, 다른 것은 B→C, 어떤 것도 A→C가 아님
3. BC/DT가 A→C를 학습할 수 없음을 보이세요 (데이터에서 본 적 없음)
4. CQL이 하위 궤적을 연결하여 A→C 경로를 발견할 수 있음을 보이세요
5. CQL이 학습한 가치 함수를 시각화하여 연결된 경로를 보이세요

---

*레슨 19 끝*
