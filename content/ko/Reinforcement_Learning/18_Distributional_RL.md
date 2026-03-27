[이전: 계층적 RL](./16_Hierarchical_RL.md)

---

# 18. 분포적 강화학습 (Distributional RL)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 강화학습에서의 분포적 관점과 전체 보상 분포를 모델링하는 것이 왜 중요한지 설명할 수 있습니다
2. 범주형 분포 투사를 사용한 C51 알고리즘을 구현할 수 있습니다
3. Quantile Regression DQN (QR-DQN)을 구축하고 분위수 Huber 손실을 이해할 수 있습니다
4. 위험 민감 제어를 위한 Implicit Quantile Networks (IQN)를 구현할 수 있습니다
5. 표준 벤치마크에서 분포적 방법들을 비교하고 그 장점을 분석할 수 있습니다

---

## 목차

1. [왜 분포적 RL인가?](#1-왜-분포적-rl인가)
2. [C51 알고리즘](#2-c51-알고리즘)
3. [Quantile Regression DQN (QR-DQN)](#3-quantile-regression-dqn-qr-dqn)
4. [Implicit Quantile Networks (IQN)](#4-implicit-quantile-networks-iqn)
5. [분포적 정책 경사법](#5-분포적-정책-경사법)
6. [위험 민감 제어](#6-위험-민감-제어)
7. [실용 구현 가이드](#7-실용-구현-가이드)
8. [연습 문제](#8-연습-문제)

---

## 1. 왜 분포적 RL인가?

### 1.1 기대값을 넘어서

전통적인 RL 알고리즘은 각 상태-행동 쌍에서 *기대* 보상을 학습합니다. 하지만 기대값은 불확실성에 대한 귀중한 정보를 버립니다.

```
전통적 Q-러닝:
  Q(s, a) = E[R₁ + γR₂ + γ²R₃ + ...]  ← 단일 스칼라

분포적 RL:
  Z(s, a) = R₁ + γR₂ + γ²R₃ + ...      ← 전체 확률 변수

  예시: 두 슬롯 머신
  머신 A: 항상 $5 지급           → E[R] = $5
  머신 B: 50/50으로 $0 또는 $10  → E[R] = $5

  같은 기대값이지만 매우 다른 분포!
  위험 회피 에이전트는 머신 A를 선호해야 합니다.
  위험 추구 에이전트는 머신 B를 선호할 수 있습니다.
```

### 1.2 보상 분포

Q(s,a) = E[Z(s,a)]를 학습하는 대신, Z(s,a)의 전체 분포를 학습합니다.

```
                    전통적 RL              분포적 RL
                    ┌─────────────┐         ┌─────────────────┐
상태-행동  ──────▶│  Q(s,a) = 5 │         │  Z(s,a):        │
  (s, a)           └─────────────┘         │  ▓▓░░▓▓░░▓▓    │
                    단일 숫자               │  확률           │
                                            │  분포           │
                                            └─────────────────┘
```

### 1.3 분포적 벨만 방정식

표준 벨만 방정식은 기대값에 대해 동작합니다:

```
Q(s, a) = E[R + γ max_a' Q(s', a')]
```

분포적 벨만 방정식은 분포에 대해 동작합니다:

```
Z(s, a) =ᵈ R + γ Z(s', a*)    여기서 a* = argmax_a' E[Z(s', a')]
                                 =ᵈ는 "분포적으로 동일"을 의미
```

이것은 Wasserstein 거리(p-Wasserstein 거리)에서의 축소입니다:

```python
import numpy as np

def wasserstein_distance(p, q, support_p, support_q):
    """두 이산 분포 간의 1-Wasserstein 거리를 계산합니다."""
    # CDF 기반 계산
    all_points = np.sort(np.unique(np.concatenate([support_p, support_q])))
    cdf_p = np.zeros_like(all_points, dtype=float)
    cdf_q = np.zeros_like(all_points, dtype=float)

    for i, x in enumerate(all_points):
        cdf_p[i] = np.sum(p[support_p <= x])
        cdf_q[i] = np.sum(q[support_q <= x])

    # W₁ = integral |CDF_p - CDF_q|
    dx = np.diff(all_points, prepend=all_points[0])
    return np.sum(np.abs(cdf_p - cdf_q) * dx)
```

### 1.4 분포가 도움이 되는 이유

| 이점 | 설명 |
|------|------|
| **더 풍부한 신호** | 분포는 스칼라보다 더 많은 그래디언트 정보를 제공 |
| **보조 학습** | 분포 예측은 더 어렵고 더 유익한 작업 |
| **위험 민감성** | CVaR, 분산 또는 기타 위험 측도를 최적화 가능 |
| **더 나은 탐색** | 인식론적 불확실성이 탐색을 유도 가능 |
| **다봉성** | 여러 가능한 결과를 포착 (예: 승/패) |

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_return_distributions():
    """분포적 RL이 더 유익한 이유를 보여줍니다."""
    np.random.seed(42)

    # 같은 기대 보상이지만 다른 분포를 가진 두 행동
    returns_safe = np.random.normal(5.0, 0.5, 10000)
    returns_risky = np.concatenate([
        np.random.normal(2.0, 0.5, 5000),
        np.random.normal(8.0, 0.5, 5000)
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(returns_safe, bins=50, alpha=0.7, color='blue', density=True)
    axes[0].axvline(np.mean(returns_safe), color='red', linestyle='--',
                    label=f'E[Z] = {np.mean(returns_safe):.2f}')
    axes[0].set_title('안전한 행동: Z(s, a_safe)')
    axes[0].legend()

    axes[1].hist(returns_risky, bins=50, alpha=0.7, color='orange', density=True)
    axes[1].axvline(np.mean(returns_risky), color='red', linestyle='--',
                    label=f'E[Z] = {np.mean(returns_risky):.2f}')
    axes[1].set_title('위험한 행동: Z(s, a_risky)')
    axes[1].legend()

    plt.suptitle('같은 E[Q] ~ 5.0, 매우 다른 분포')
    plt.tight_layout()
    plt.savefig('distributional_comparison.png', dpi=150)
    plt.show()

# visualize_return_distributions()
```

---

## 2. C51 알고리즘

### 2.1 범주형 분포 표현

C51 (51개 원자를 가진 범주형)은 보상 분포를 균등 간격의 "원자" 고정 집합에 대한 범주형 분포로 표현합니다:

```
원자:  z₁=V_MIN, z₂, z₃, ..., z_N=V_MAX    (기본값 N=51)

             p(zᵢ | s, a) = 보상이 zᵢ일 확률
                  │
                  ▼
  확률
    ▓
    ▓ ▓
    ▓ ▓ ▓
    ▓ ▓ ▓ ▓
    ▓ ▓ ▓ ▓ ▓ ▓
  ──────────────── 보상 값
  V_MIN         V_MAX

  Q(s,a) = Σᵢ zᵢ · p(zᵢ | s, a)   ← 기대값 복원
```

### 2.2 네트워크 아키텍처

```
            ┌──────────────┐
  상태 s ──▶  공유 CNN     │
            │  /FC 레이어  │──────┐
            └──────────────┘      │
                                  ▼
                    ┌─────────────────────────┐
                    │  행동별 출력 헤드         │
                    │                         │
                    │  행동 0: [p₁...p₅₁]     │  (softmax)
                    │  행동 1: [p₁...p₅₁]     │  (softmax)
                    │  ...                    │
                    │  행동 K: [p₁...p₅₁]     │  (softmax)
                    └─────────────────────────┘
```

### 2.3 투사 단계

목표 분포를 계산할 때 벨만 업데이트는 원자를 이동하고 스케일링합니다. 결과 원자가 고정 지지(support)와 정렬되지 않을 수 있으므로 다시 *투사*해야 합니다:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class C51Network(nn.Module):
    """C51 분포적 DQN 네트워크."""

    def __init__(self, state_dim, action_dim, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # 지지: 고정된 원자 집합
        self.register_buffer(
            'support', torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # 네트워크 레이어
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim * n_atoms)

    def forward(self, state):
        """각 행동에 대한 확률 분포를 반환합니다."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # (배치, 행동, 원자)로 변환하고 원자에 대해 softmax 적용
        x = x.view(-1, self.action_dim, self.n_atoms)
        probs = F.softmax(x, dim=-1)
        return probs

    def get_q_values(self, state):
        """분포의 기대값으로 Q-값을 계산합니다."""
        probs = self.forward(state)
        q_values = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return q_values


def c51_projection(next_probs, rewards, dones, support, gamma, v_min, v_max, n_atoms):
    """
    벨만 업데이트된 분포를 고정 지지 위에 투사합니다.

    Args:
        next_probs: (배치, n_atoms) - 목표 분포
        rewards: (배치,) - 즉각적 보상
        dones: (배치,) - 종료 플래그
        support: (n_atoms,) - 원자 위치
        gamma: 할인 계수
    Returns:
        projected: (배치, n_atoms) - 투사된 분포
    """
    batch_size = rewards.shape[0]
    delta_z = (v_max - v_min) / (n_atoms - 1)

    # 각 원자에 대해 Tz = r + γz 계산
    Tz = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * support.unsqueeze(0)
    Tz = Tz.clamp(v_min, v_max)

    # 투사 인덱스 계산
    b = (Tz - v_min) / delta_z  # 분수 인덱스
    l = b.floor().long()         # 하한 인덱스
    u = (l + 1).clamp(max=n_atoms - 1)  # 상한 인덱스
    l = l.clamp(min=0)

    # 확률 배분
    projected = torch.zeros(batch_size, n_atoms, device=rewards.device)

    # 하한 기여
    projected.scatter_add_(1, l, next_probs * (u.float() - b))
    # 상한 기여
    projected.scatter_add_(1, u, next_probs * (b - l.float()))

    return projected
```

### 2.4 C51 학습 루프

```python
class C51Agent:
    """경험 재생을 사용하는 완전한 C51 에이전트."""

    def __init__(self, state_dim, action_dim, n_atoms=51,
                 v_min=-10, v_max=10, lr=2.5e-4, gamma=0.99):
        self.gamma = gamma
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.network = C51Network(state_dim, action_dim, n_atoms, v_min, v_max)
        self.target_network = C51Network(state_dim, action_dim, n_atoms, v_min, v_max)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.support = self.network.support

    def select_action(self, state, epsilon=0.0):
        """엡실론-탐욕 행동 선택."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.network.get_q_values(state_t)
            return q_values.argmax(dim=-1).item()

    def train_step(self, batch):
        """재생 버퍼에서 배치에 대한 하나의 학습 단계."""
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 현재 분포: p(s, a)
        current_probs = self.network(states)
        current_probs = current_probs[
            torch.arange(len(states)), actions
        ]

        with torch.no_grad():
            # 다음 상태: 온라인 네트워크로 행동 선택 (Double DQN 스타일)
            next_q = self.network.get_q_values(next_states)
            next_actions = next_q.argmax(dim=-1)

            # 선택된 행동의 목표 분포 획득
            next_probs = self.target_network(next_states)
            next_probs = next_probs[
                torch.arange(len(next_states)), next_actions
            ]

            # 목표 분포 투사
            target_probs = c51_projection(
                next_probs, rewards, dones,
                self.support, self.gamma,
                self.v_min, self.v_max, self.n_atoms
            )

        # 투사된 목표와 현재 사이의 교차 엔트로피 손실
        loss = -(target_probs * torch.log(current_probs + 1e-8)).sum(dim=-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()
```

### 2.5 C51 결과 및 분석

```
Atari 게임에서의 성능 (Bellemare et al., 2017):

게임           | DQN     | C51     | 개선
─────────────────────────────────────────────────
Asterix        | 8,503   | 406,211 | 47.8배
Breakout       | 401     | 748     | 1.9배
Pong           | 20.9    | 20.9    | 1.0배
Seaquest       | 5,286   | 266,434 | 50.4배
Space Invaders | 1,976   | 5,747   | 2.9배

핵심 통찰: 확률적/다봉 환경에서 가장 큰 이득.
```

---

## 3. Quantile Regression DQN (QR-DQN)

### 3.1 고정 원자에서 고정 확률로

C51은 원자 위치를 고정하고 확률을 학습합니다. QR-DQN은 반대로: 확률(균일 분위수)을 고정하고 원자 위치를 학습합니다.

```
C51:
  고정:   z₁, z₂, ..., z₅₁  (원자 위치)
  학습:   p₁, p₂, ..., p₅₁  (확률)

QR-DQN:
  고정:   τ₁=1/2N, τ₂=3/2N, ..., τ_N=(2N-1)/2N  (분위수 중간점)
  학습:   θ₁, θ₂, ..., θ_N  (분위수 값)

  장점: V_MIN/V_MAX 하이퍼파라미터 불필요!
```

### 3.2 분위수 Huber 손실

QR-DQN은 안정성을 위해 분위수 회귀와 Huber 손실을 결합한 분위수 Huber 손실을 사용합니다:

```python
def quantile_huber_loss(predictions, targets, taus, kappa=1.0):
    """
    분위수 Huber 손실을 계산합니다.

    Args:
        predictions: (배치, N) - 예측 분위수 값
        targets: (배치, N) - 목표 분위수 값
        taus: (N,) - 분위수 중간점
        kappa: Huber 손실 임계값
    Returns:
        loss: 스칼라
    """
    # 쌍별 TD 오류: (배치, N_pred, N_target)
    td_errors = targets.unsqueeze(1) - predictions.unsqueeze(2)

    # Huber 손실 요소
    huber = torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors ** 2,
        kappa * (td_errors.abs() - 0.5 * kappa)
    )

    # 분위수 가중치: 비대칭 가중
    taus_expanded = taus.unsqueeze(0).unsqueeze(2)
    quantile_weight = torch.abs(
        taus_expanded - (td_errors < 0).float()
    )

    loss = (quantile_weight * huber).sum(dim=-1).mean(dim=-1)
    return loss.mean()
```

### 3.3 QR-DQN 네트워크

```python
class QRDQNNetwork(nn.Module):
    """Quantile Regression DQN 네트워크."""

    def __init__(self, state_dim, action_dim, n_quantiles=200):
        super().__init__()
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles

        # 분위수 중간점
        taus = torch.arange(1, n_quantiles + 1, dtype=torch.float32)
        self.register_buffer('taus', (2 * taus - 1) / (2 * n_quantiles))

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim * n_quantiles)

    def forward(self, state):
        """각 행동에 대한 분위수 값을 반환합니다."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        quantiles = x.view(-1, self.action_dim, self.n_quantiles)
        return quantiles

    def get_q_values(self, state):
        """분위수 값의 평균으로 Q-값을 계산합니다."""
        quantiles = self.forward(state)
        return quantiles.mean(dim=-1)
```

### 3.4 QR-DQN vs C51 비교

| 특성 | C51 | QR-DQN |
|------|-----|--------|
| **표현** | 고정 지지, 학습된 확률 | 고정 확률, 학습된 지지 |
| **하이퍼파라미터** | V_MIN, V_MAX, N_atoms | N_quantiles만 |
| **손실 함수** | 교차 엔트로피 | 분위수 Huber |
| **수렴 거리** | KL 발산 | Wasserstein 거리 |
| **유연성** | 지지 범위에 제한 | 무한 지지 |
| **일반적인 N** | 51 | 200 |

---

## 4. Implicit Quantile Networks (IQN)

### 4.1 고정에서 샘플링된 분위수로

IQN은 QR-DQN보다 더 나아가 고정 집합 대신 학습 중에 무작위로 분위수 수준을 샘플링합니다:

```
QR-DQN:  고정 τ = {0.025, 0.075, ..., 0.975}  (N=20 분위수)
IQN:     τ ~ Uniform(0, 1) 샘플링              (필요에 따라 어떤 분위수든)

이는 IQN이 전체 분위수 함수 F⁻¹(τ)를 근사할 수 있음을 의미합니다.
```

### 4.2 분위수 임베딩

IQN은 코사인 기저를 사용하여 분위수 수준 τ를 임베딩합니다:

```python
class QuantileEmbedding(nn.Module):
    """코사인 기저 함수를 사용한 분위수 수준 임베딩."""

    def __init__(self, embedding_dim=64, n_cos=64):
        super().__init__()
        self.n_cos = n_cos
        self.embedding = nn.Linear(n_cos, embedding_dim)

        self.register_buffer(
            'i_pi',
            torch.arange(1, n_cos + 1, dtype=torch.float32) * np.pi
        )

    def forward(self, taus):
        """
        분위수 수준을 임베딩합니다.
        Args:
            taus: (배치, N) [0, 1] 범위의 분위수 수준
        Returns:
            embedding: (배치, N, embedding_dim)
        """
        cos_features = torch.cos(taus.unsqueeze(-1) * self.i_pi)
        embedding = F.relu(self.embedding(cos_features))
        return embedding


class IQNNetwork(nn.Module):
    """Implicit Quantile Network."""

    def __init__(self, state_dim, action_dim, embedding_dim=64, n_cos=64):
        super().__init__()
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        # 상태 인코더
        self.state_fc1 = nn.Linear(state_dim, embedding_dim)
        self.state_fc2 = nn.Linear(embedding_dim, embedding_dim)

        # 분위수 임베딩
        self.quantile_embed = QuantileEmbedding(embedding_dim, n_cos)

        # 결합 레이어
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, state, taus):
        """
        주어진 상태와 분위수 수준에 대한 분위수 값을 계산합니다.

        Args:
            state: (배치, state_dim)
            taus: (배치, N) 샘플링된 분위수 수준
        Returns:
            quantile_values: (배치, N, action_dim)
        """
        # 상태 인코딩
        state_feat = F.relu(self.state_fc1(state))
        state_feat = F.relu(self.state_fc2(state_feat))

        # 분위수 임베딩
        tau_feat = self.quantile_embed(taus)

        # 원소별 곱
        combined = state_feat.unsqueeze(1) * tau_feat

        # 각 행동에 대한 분위수 값 출력
        x = F.relu(self.fc1(combined))
        quantile_values = self.fc2(x)

        return quantile_values

    def get_q_values(self, state, n_quantiles=32):
        """분위수를 샘플링하고 평균하여 Q를 추정합니다."""
        batch_size = state.shape[0]
        taus = torch.rand(batch_size, n_quantiles, device=state.device)
        quantile_values = self.forward(state, taus)
        return quantile_values.mean(dim=1)
```

### 4.3 IQN을 이용한 위험 민감 정책

IQN의 주요 장점: 어떤 분위수를 평가할지 선택함으로써 다양한 위험 태도를 구현할 수 있습니다:

```python
class RiskSensitiveIQN:
    """조절 가능한 위험 태도를 가진 IQN 에이전트."""

    def __init__(self, network, risk_level='neutral'):
        self.network = network
        self.risk_level = risk_level

    def select_action(self, state, n_quantiles=32):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)

            if self.risk_level == 'neutral':
                # [0, 1]에서 균일 → 기대값
                taus = torch.rand(1, n_quantiles)

            elif self.risk_level == 'averse':
                # 하위 분위수만: [0, 0.25] → 비관적 (CVaR)
                taus = torch.rand(1, n_quantiles) * 0.25

            elif self.risk_level == 'seeking':
                # 상위 분위수만: [0.75, 1] → 낙관적
                taus = 0.75 + torch.rand(1, n_quantiles) * 0.25

            elif self.risk_level == 'cvar_50':
                # 50%에서의 CVaR: 하위 50% 분위수의 평균
                taus = torch.rand(1, n_quantiles) * 0.5

            quantile_values = self.network(state_t, taus)
            q_values = quantile_values.mean(dim=1)
            return q_values.argmax(dim=-1).item()
```

### 4.4 비교: C51 vs QR-DQN vs IQN

```
                C51              QR-DQN           IQN
              ──────────       ──────────       ──────────
지지:         고정 원자        학습된 원자       암묵적 (임의 τ)
확률:         학습됨           고정 (균일)       샘플링
유연성:        ★★              ★★★              ★★★★★
위험 제어:     제한적           제한적            전체 CVaR/CPT
계산량:        낮음             중간              중간-높음
성능:          좋음             더 좋음           최고
```

---

## 5. 분포적 정책 경사법

### 5.1 연속 행동으로의 확장

분포적 방법은 원래 DQN(이산 행동)을 위해 개발되었습니다. 연속 행동으로 확장하려면 분포적 정책 경사법이 필요합니다.

```python
class DistributionalCritic(nn.Module):
    """연속 행동을 위한 QR-DQN 스타일 비평가."""

    def __init__(self, state_dim, action_dim, n_quantiles=25, hidden_dim=256):
        super().__init__()
        self.n_quantiles = n_quantiles

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_quantiles),
        )

    def forward(self, state, action):
        """(상태, 행동) 쌍에 대한 분위수 값을 반환합니다."""
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
```

### 5.2 D4PG: Distributed Distributional DDPG

D4PG는 다음을 결합합니다:
- 분포적 비평가 (C51 스타일)
- 분산 학습 (다중 액터)
- N-스텝 보상
- 우선 경험 재생

```
D4PG 아키텍처:

  액터 1 ─────┐
  액터 2 ─────┤
  액터 3 ─────┼──▶ 우선 순위    ──▶ 학습기
  ...          │    재생 버퍼       (C51 비평가 + DDPG 액터)
  액터 K ─────┘

  각 액터는 병렬로 실행되며 다양한 경험을 수집합니다.
  학습기는 교차 엔트로피 손실로 분포적 비평가를 업데이트합니다.
```

### 5.3 분포적 SAC

```python
class DistributionalSAC:
    """
    분포적 Soft Actor-Critic: SAC와 분포적 비평가를 결합합니다.
    연속 행동 공간에 QR-DQN 스타일 비평가를 사용합니다.
    """

    def __init__(self, state_dim, action_dim, n_quantiles=25,
                 hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau_soft = tau
        self.alpha = alpha
        self.n_quantiles = n_quantiles

        # 분위수 중간점
        taus = torch.arange(1, n_quantiles + 1, dtype=torch.float32)
        self.taus = (2 * taus - 1) / (2 * n_quantiles)

        # 쌍둥이 분포적 비평가
        self.critic1 = DistributionalCritic(state_dim, action_dim, n_quantiles, hidden_dim)
        self.critic2 = DistributionalCritic(state_dim, action_dim, n_quantiles, hidden_dim)

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )

    def update_critics(self, states, actions, rewards, next_states, dones,
                       next_actions, next_log_probs):
        """분위수 회귀로 분포적 비평가를 업데이트합니다."""
        with torch.no_grad():
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_quantiles = torch.min(target_q1, target_q2)

            target_quantiles = rewards.unsqueeze(1) + \
                self.gamma * (1 - dones.unsqueeze(1)) * \
                (target_quantiles - self.alpha * next_log_probs.unsqueeze(1))

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        loss1 = quantile_huber_loss(current_q1, target_quantiles, self.taus)
        loss2 = quantile_huber_loss(current_q2, target_quantiles, self.taus)

        critic_loss = loss1 + loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()
```

---

## 6. 위험 민감 제어

### 6.1 위험 측도

전체 보상 분포에 접근하면 다양한 위험 측도를 최적화할 수 있습니다:

```
금융에서 RL에 적용된 위험 측도:

1. 기대값 (위험 중립):
   ρ(Z) = E[Z]

2. 분산:
   ρ(Z) = E[Z] - λ·Var(Z)     (평균-분산)

3. CVaR (조건부 위험 가치):
   CVaR_α(Z) = E[Z | Z ≤ F⁻¹(α)]
   "최악의 α% 경우의 평균 보상"

4. Wang 위험 측도:
   분위수 함수를 왜곡: g(τ) = Φ(Φ⁻¹(τ) + η)
   η > 0: 위험 회피, η < 0: 위험 추구

5. 누적 전망 이론 (CPT):
   이득과 손실에 대한 다른 가중
   희귀한 극단적 사건을 과대 가중
```

### 6.2 IQN을 이용한 CVaR 최적화

```python
def cvar_action_selection(iqn_network, state, alpha=0.25, n_samples=64):
    """
    CVaR_α를 최대화하는 행동을 선택합니다.

    CVaR_α = 보상 분포의 하위 α 분위수의 평균.
    낮은 α → 더 보수적 (최악의 경우에 집중).
    """
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)

        # CVaR을 위해 [0, α]에서만 분위수 샘플링
        taus = torch.rand(1, n_samples) * alpha

        quantile_values = iqn_network(state_t, taus)
        cvar_values = quantile_values.mean(dim=1)

        return cvar_values.argmax(dim=-1).item()


def evaluate_risk_policies(env, iqn_network, n_episodes=100):
    """같은 환경에서 다른 위험 태도를 비교합니다."""
    risk_levels = {
        'CVaR 10% (매우 보수적)': 0.10,
        'CVaR 25% (보수적)': 0.25,
        'CVaR 50% (중간)': 0.50,
        '위험 중립 (CVaR 100%)': 1.00,
    }

    results = {}
    for name, alpha in risk_levels.items():
        returns = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_return = 0
            done = False

            while not done:
                action = cvar_action_selection(iqn_network, state, alpha)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                done = terminated or truncated

            returns.append(episode_return)

        results[name] = {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'cvar_10': np.percentile(returns, 10),
        }

    return results
```

### 6.3 위험-보상 트레이드오프

```
           평균 보상
              ▲
              │         ○ 위험 중립
              │       ○
              │     ○ CVaR-50%
              │   ○
              │ ○ CVaR-25%
              │○ CVaR-10%
              └──────────────────────▶ 안전성 (높은 CVaR₁₀)

  더 보수적으로 갈수록:
  - 평균 보상 감소 (안전에 대한 비용 지불)
  - 최악의 경우 성능 크게 향상
  - 표준 편차 일반적으로 감소
```

---

## 7. 실용 구현 가이드

### 7.1 올바른 알고리즘 선택

```
분포적 RL을 위한 의사 결정 트리:

  위험 민감성 필요?
  ├── 예 → IQN (전체 분위수 함수)
  └── 아니오
       ├── 이산 행동?
       │   ├── 간단한 설정 → C51 (잘 이해됨, 신뢰성 높음)
       │   └── V_MIN/V_MAX 불필요 → QR-DQN (적은 하이퍼파라미터)
       └── 연속 행동?
           └── D4PG 또는 분포적 SAC
```

### 7.2 하이퍼파라미터 가이드라인

| 파라미터 | C51 | QR-DQN | IQN |
|----------|-----|--------|-----|
| **N (원자/분위수)** | 51 | 200 | 64 (샘플링) |
| **V_MIN, V_MAX** | 작업에 따라 다름 | 해당 없음 | 해당 없음 |
| **학습률** | 2.5e-4 | 5e-5 | 5e-5 |
| **Huber kappa** | 해당 없음 | 1.0 | 1.0 |
| **코사인 임베딩 차원** | 해당 없음 | 해당 없음 | 64 |
| **배치 크기** | 32 | 32 | 32 |
| **목표 업데이트** | 8000 스텝 | 8000 스텝 | 8000 스텝 |

### 7.3 일반적인 함정

```python
# 함정 1: C51에서 잘못된 V_MIN/V_MAX
# 보상이 [V_MIN, V_MAX]를 초과하면 분포가 잘립니다
# 해결: 예상 보상보다 넓은 범위 설정

# 함정 2: 로그 확률에서의 수치 불안정성
# 나쁜 예:
loss = -(target * torch.log(predicted)).sum()
# 좋은 예:
loss = -(target * torch.log(predicted + 1e-8)).sum()

# 함정 3: QR-DQN에서 분위수 정렬 잊음
# 분위수 값은 대략 정렬되어야 합니다
# 손실이 자연스럽게 이를 장려하지만 초기화가 중요합니다

# 함정 4: IQN 평가 시 너무 적은 분위수 샘플
# 학습: N=8 샘플이면 충분 (확률적이어도 괜찮음)
# 평가: 안정적인 Q-값 추정을 위해 N=32+ 사용
```

### 7.4 전체 학습 예제

```python
import gymnasium as gym
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones, dtype=float))

    def __len__(self):
        return len(self.buffer)


def train_c51(env_name='CartPole-v1', n_episodes=500, n_atoms=51):
    """Gymnasium 환경에서 C51 에이전트를 학습합니다."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = C51Agent(state_dim, action_dim, n_atoms=n_atoms)
    buffer = ReplayBuffer()
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 64
    target_update_freq = 500
    step_count = 0

    episode_returns = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_return = 0
        done = False

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_return += reward
            step_count += 1

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                agent.train_step(batch)

            if step_count % target_update_freq == 0:
                agent.target_network.load_state_dict(
                    agent.network.state_dict()
                )

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_returns.append(episode_return)

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_returns[-50:])
            print(f"에피소드 {episode+1}, 평균 보상: {avg:.1f}, "
                  f"엡실론: {epsilon:.3f}")

    env.close()
    return agent, episode_returns
```

---

## 8. 연습 문제

### 연습 1: C51 처음부터 구현

CartPole-v1을 위한 완전한 C51 에이전트를 구축하세요:
1. 조절 가능한 N_atoms를 가진 C51Network를 구현하세요
2. 범주형 투사 단계를 구현하세요
3. 500 에피소드 동안 학습하고 학습 곡선을 그리세요
4. 다양한 상태에서 학습된 보상 분포를 시각화하세요
5. N_atoms = {11, 21, 51}을 비교하고 효과를 분석하세요

### 연습 2: QR-DQN 구현

QR-DQN을 구현하고 C51과 비교하세요:
1. 분위수 Huber 손실을 가진 QR-DQN 네트워크를 구축하세요
2. CartPole-v1과 LunarLander-v2에서 학습하세요
3. 주요 상태에서 학습된 분위수 함수를 그리세요
4. C51과 수렴 속도 및 최종 성능을 비교하세요
5. N_quantiles = {10, 25, 50, 200}으로 실험하세요

### 연습 3: 위험 민감 IQN

위험 민감 정책을 가진 IQN 에이전트를 구축하세요:
1. 분위수 수준을 위한 코사인 임베딩을 구현하세요
2. 확률적 환경에서 학습하세요 (예: 랜덤 바람이 있는 수정된 CartPole)
3. 다양한 알파 수준에서 CVaR 행동 선택을 구현하세요
4. 위험 중립 vs CVaR-25% 정책을 비교하세요: 보상 분포를 그리세요
5. 보수적 정책이 낮은 분산을 위해 평균 보상을 희생함을 보이세요

### 연습 4: 분포 시각화 대시보드

분포적 RL을 위한 시각화 도구를 만드세요:
1. 학습 중 100 에피소드마다 주요 상태에서 보상 분포를 저장하세요
2. 분포 변화를 보여주는 애니메이션 그래프를 만드세요
3. 투사 단계를 시각적으로 보여주세요: 투사 전후
4. 연속적인 분포 추정치 간의 Wasserstein 거리를 그리세요
5. 분포가 이봉인 상태를 식별하고 그 이유를 설명하세요

### 연습 5: 분포적 Rainbow

분포적 RL을 다른 DQN 개선사항과 결합하세요:
1. C51을 기반으로 시작하세요
2. 우선 경험 재생 추가 (분포적 TD 오류로 우선순위 지정)
3. 분포 투사를 사용한 n-스텝 보상 추가
4. 잡음 네트워크 추가 (엡실론-탐욕 대체)
5. 비교: DQN, C51, QR-DQN, 그리고 분포적 Rainbow을 Atari에서

---

*레슨 18 끝*
