[이전: Goal-Conditioned RL](./20_Goal_Conditioned_RL.md)

---

# 21. 보상 설계와 내적 동기 부여

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. Potential-based reward shaping과 정책 불변성 보장에 대해 설명
2. 호기심 기반 탐색을 통한 내적 동기 부여 구현
3. 탐색 보너스를 위한 Random Network Distillation (RND) 구축
4. 일반적인 함정(보상 해킹, 희소 보상)을 피하는 보상 함수 설계
5. 탐색이 어려운 환경에서 다양한 탐색 전략 비교

---

## 목차

1. [보상 설계 문제](#1-보상-설계-문제)
2. [Potential-Based Reward Shaping](#2-potential-based-reward-shaping)
3. [호기심 기반 탐색](#3-호기심-기반-탐색)
4. [Random Network Distillation (RND)](#4-random-network-distillation-rnd)
5. [카운트 기반 탐색](#5-카운트-기반-탐색)
6. [보상 해킹과 정렬 오류](#6-보상-해킹과-정렬-오류)
7. [실전 보상 엔지니어링](#7-실전-보상-엔지니어링)
8. [연습문제](#8-연습문제)

---

## 1. 보상 설계 문제

### 1.1 보상 설계가 중요한 이유

```
보상 함수는 모든 RL 시스템에서 가장 중요한 부분입니다.

"보상 함수가 잘못되면, 잘못된 행동을 얻게 됩니다."

보상 오명세 사례:
  ❌ 로봇 청소기: 보상 = 수집한 먼지
     결과: 먼지를 버리고 영원히 다시 수집함

  ❌ 게임 에이전트: 보상 = 점수
     결과: 제대로 플레이하는 대신 글리치를 악용

  ❌ 트레이딩 에이전트: 보상 = 수익
     결과: 단기 이익을 위해 극단적 위험을 감수

  ✓ 좋은 보상 설계는 실제로 원하는 것이 무엇인지 이해해야 함
```

### 1.2 희소 보상 vs 밀집 보상

```
희소 보상:
  R = 1 (목표 도달 시), 그 외 0
  장점: 정확하게 지정하기 쉬움 (보상 해킹 감소)
  단점: 에이전트가 양의 보상을 보지 못할 수 있음 (탐색의 악몽)

밀집 보상:
  R = -||상태 - 목표|| (매 스텝)
  장점: 지속적인 학습 신호
  단점: 의도하지 않은 행동 유발 가능 (지역 최적)

형상화된 보상 (두 가지의 장점):
  R = 희소_보상 + 형상화_보너스
  장점: 학습 신호 + 올바른 최적 정책
  단점: 형상화를 신중하게 설계해야 함
```

---

## 2. Potential-Based Reward Shaping

### 2.1 핵심 정리

Potential-based reward shaping (PBRS)은 최적 정책을 변경하지 않는다는 것이 증명된 보너스 F(s, s')를 보상에 추가합니다:

```
형상화 함수:
  F(s, s') = γ · Φ(s') - Φ(s)

여기서 Φ: S -> R은 "포텐셜 함수" (상태의 임의 함수)입니다.

정리 (Ng, Harada, & Russell 1999):
  F(s,s') = γΦ(s') - Φ(s)이면, R + F 하에서의 최적 정책은
  R 하에서의 최적 정책과 동일합니다.

  이것이 최적성을 보존하는 유일한 형태의 가산적 형상화입니다!
```

### 2.2 구현

```python
import numpy as np


class PotentialBasedShaping:
    """Potential-based reward shaping that preserves optimal policy."""

    def __init__(self, potential_fn, gamma=0.99):
        self.potential = potential_fn
        self.gamma = gamma

    def shape(self, state, next_state, base_reward):
        """Add shaping bonus to base reward."""
        phi_s = self.potential(state)
        phi_s_next = self.potential(next_state)

        shaping = self.gamma * phi_s_next - phi_s
        return base_reward + shaping


# Example: Grid world with goal at (9, 9)
def manhattan_potential(state, goal=np.array([9, 9])):
    """Potential = negative Manhattan distance to goal."""
    return -np.abs(state - goal).sum()


def euclidean_potential(state, goal=np.array([9, 9])):
    """Potential = negative Euclidean distance to goal."""
    return -np.linalg.norm(state - goal)


# Usage
shaper = PotentialBasedShaping(manhattan_potential, gamma=0.99)

state = np.array([3, 3])
next_state = np.array([4, 3])  # moved closer to goal
base_reward = 0  # sparse reward (not at goal)

shaped_reward = shaper.shape(state, next_state, base_reward)
print(f"Base reward: {base_reward}")
print(f"Shaped reward: {shaped_reward:.4f}")
# Shaped reward > 0 because we moved closer to goal
```

### 2.3 Potential 기반 형상화만 안전한 이유

```python
def demonstrate_bad_shaping():
    """Show how non-potential-based shaping changes optimal policy."""

    # Simple MDP: 3 states, 2 actions
    # State 0 -> State 1 (action 0): reward = 1
    # State 0 -> State 2 (action 1): reward = 2
    # Optimal: action 1 (go to state 2 for reward 2)

    # Bad shaping: give +5 bonus for going to state 1
    # Result: action 0 now looks better (1 + 5 = 6 > 2)
    # Optimal policy CHANGED!

    print("Without shaping:")
    print("  Action 0: R = 1  |  Action 1: R = 2  |  Optimal: Action 1")
    print()
    print("With bad shaping (+5 for state 1):")
    print("  Action 0: R = 1 + 5 = 6  |  Action 1: R = 2  |  'Optimal': Action 0")
    print("  WRONG! Optimal policy changed!")
    print()
    print("With potential-based shaping (Φ(s1)=5, Φ(s2)=0, Φ(s0)=0):")
    print("  Action 0: R = 1 + γ·5 - 0 = 5.95")
    print("  Action 1: R = 2 + γ·0 - 0 = 2.0")
    print("  Looks wrong at first, but VALUE FUNCTION adjusts.")
    print("  The optimal POLICY is still action 1 for the long run!")

demonstrate_bad_shaping()
```

---

## 3. 호기심 기반 탐색

### 3.1 내적 동기 부여

```
외적 보상:  환경으로부터 (과제 보상)
내적 보상:  에이전트가 자체 생성 (호기심 보너스)

총 보상 = R_외적 + β · R_내적

내적 동기 부여 원천:
├── 예측 오차 (호기심)
│   "이 상태에 놀랐다 -> 더 탐색하자!"
├── 정보 이득
│   "이것이 불확실성을 줄인다 -> 더 탐색하자!"
├── 새로움 (카운트 기반)
│   "여기 자주 오지 않았다 -> 더 탐색하자!"
└── 역능감
    "여기서 더 많은 제어력이 있다 -> 더 탐색하자!"
```

### 3.2 ICM: Intrinsic Curiosity Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ICM(nn.Module):
    """
    Intrinsic Curiosity Module (Pathak et al., 2017).

    Key idea: Curiosity = prediction error in a learned feature space.
    Uses forward model (predict next features) and inverse model
    (predict action from features) jointly.
    """

    def __init__(self, state_dim, action_dim, feature_dim=64, beta=0.2):
        super().__init__()
        self.beta = beta

        # Feature encoder: state -> features
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

        # Forward model: (φ(s), a) -> predicted φ(s')
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

        # Inverse model: (φ(s), φ(s')) -> predicted action
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state, next_state, action):
        """Compute intrinsic reward and auxiliary losses."""
        # Encode states
        phi_s = self.encoder(state)
        phi_s_next = self.encoder(next_state)

        # Forward model: predict next features
        forward_input = torch.cat([phi_s, action], dim=-1)
        phi_s_next_pred = self.forward_model(forward_input)

        # Intrinsic reward = forward prediction error
        intrinsic_reward = 0.5 * ((phi_s_next_pred - phi_s_next.detach()) ** 2).sum(dim=-1)

        # Inverse model: predict action
        inverse_input = torch.cat([phi_s, phi_s_next], dim=-1)
        action_pred = self.inverse_model(inverse_input)

        # Losses
        forward_loss = F.mse_loss(phi_s_next_pred, phi_s_next.detach())
        inverse_loss = F.mse_loss(action_pred, action)

        # Combined ICM loss
        icm_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

        return intrinsic_reward, icm_loss

    def get_intrinsic_reward(self, state, next_state, action):
        """Get intrinsic reward for a transition."""
        with torch.no_grad():
            phi_s = self.encoder(state)
            phi_s_next = self.encoder(next_state)

            forward_input = torch.cat([phi_s, action], dim=-1)
            phi_s_next_pred = self.forward_model(forward_input)

            reward = 0.5 * ((phi_s_next_pred - phi_s_next) ** 2).sum(dim=-1)
        return reward
```

### 3.3 왜 특징 공간에서 예측 오차를 사용하는가?

```
왜 원시 픽셀/상태를 예측하지 않는가?

문제: "시끄러운 TV" 효과
  다음 상태에 줄일 수 없는 무작위성이 있으면 (예: TV 정적 잡음),
  순방향 모델은 이를 정확하게 예측할 수 없습니다.
  -> 잡음에 대한 영구적인 "호기심"
  -> 에이전트가 시끄러운 TV를 영원히 바라봄!

해결책: 다음과 같은 특징을 학습:
  1. 에이전트의 행동과 관련이 있는 (역방향 모델이 이를 보장)
  2. (상태, 행동)으로부터 예측 가능한
  3. 관련 없는 잡음을 무시하는

역방향 모델은 필터 역할을 합니다:
  φ(s)와 φ(s')가 취한 행동을 예측하는 데 도움이 안 되면,
  그 특징은 쓸모없음 -> 인코더가 이를 무시하도록 학습합니다.
```

---

## 4. Random Network Distillation (RND)

### 4.1 RND 개념

RND는 더 간단한 호기심 신호를 사용합니다: 고정된 랜덤 네트워크의 예측 오차입니다.

```
RND 아키텍처:

  타겟 네트워크 f: s -> R^d    (고정된 랜덤 가중치, 업데이트 안 함)
  예측 네트워크 f̂: s -> R^d  (타겟에 맞추도록 학습)

  내적 보상 = ||f(s) - f̂(s)||²

  직관:
  - 자주 방문한 상태: 예측기가 많은 예제를 봄
    -> f̂(s) ≈ f(s) -> 낮은 보상
  - 새로운 상태: 예측기가 이런 것을 학습하지 않음
    -> f̂(s) ≠ f(s) -> 높은 보상 (탐색!)
```

### 4.2 RND 구현

```python
class RNDModule(nn.Module):
    """Random Network Distillation for exploration."""

    def __init__(self, state_dim, feature_dim=128):
        super().__init__()

        # Target: fixed random network (never updated)
        self.target = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

        # Predictor: trained to match target
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)

        # Running statistics for reward normalization
        self.reward_running_mean = 0
        self.reward_running_var = 1
        self.reward_count = 0

    def compute_intrinsic_reward(self, state):
        """Compute exploration bonus."""
        with torch.no_grad():
            target_features = self.target(state)
            predicted_features = self.predictor(state)

            reward = ((target_features - predicted_features) ** 2).sum(dim=-1)
        return reward

    def update(self, states):
        """Train predictor to match target."""
        target_features = self.target(states).detach()
        predicted_features = self.predictor(states)

        loss = F.mse_loss(predicted_features, target_features)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def normalize_reward(self, reward):
        """Normalize intrinsic reward using running statistics."""
        self.reward_count += 1
        delta = reward - self.reward_running_mean
        self.reward_running_mean += delta / self.reward_count
        self.reward_running_var += delta * (reward - self.reward_running_mean)

        std = np.sqrt(self.reward_running_var / max(self.reward_count, 1)) + 1e-8
        return (reward - self.reward_running_mean) / std
```

### 4.3 RND 에이전트 학습

```python
class RNDAgent:
    """PPO agent with RND exploration bonus."""

    def __init__(self, state_dim, action_dim, intrinsic_coef=1.0,
                 extrinsic_coef=2.0, gamma_i=0.99, gamma_e=0.999):
        self.intrinsic_coef = intrinsic_coef
        self.extrinsic_coef = extrinsic_coef
        self.gamma_i = gamma_i  # discount for intrinsic reward
        self.gamma_e = gamma_e  # discount for extrinsic reward

        self.rnd = RNDModule(state_dim)

        # Separate value heads for intrinsic and extrinsic rewards
        self.value_ext = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.value_int = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Shared policy
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def compute_combined_reward(self, state, extrinsic_reward):
        """Combine extrinsic and intrinsic rewards."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        intrinsic = self.rnd.compute_intrinsic_reward(state_t).item()
        intrinsic = self.rnd.normalize_reward(intrinsic)

        combined = (self.extrinsic_coef * extrinsic_reward +
                    self.intrinsic_coef * intrinsic)
        return combined, intrinsic
```

### 4.4 Montezuma's Revenge에서의 RND

```
Montezuma's Revenge: 탐색 벤치마크

탐색 보너스 없이:
  DQN 점수: ~0 (첫 번째 열쇠를 찾지 못함)
  PPO 점수: ~0

RND 사용 시:
  점수: ~10,000+ (여러 방을 탐색하고 열쇠를 찾음)
  시연 없이 유의미한 진전을 달성한 최초의 알고리즘

이 게임이 어려운 이유:
  - 매우 희소한 보상 (보상 사이에 수백 개의 행동)
  - 특정 행동 순서 필요 (열쇠 획득 -> 문 열기)
  - 무작위 탐색으로는 보상에 거의 도달 불가

RND가 해결하는 이유:
  - 새로운 상태 (새로운 방)가 높은 내적 보상을 줌
  - 에이전트가 미탐색 영역에 대해 "호기심"을 가짐
  - 점진적으로 전체 맵을 탐색
```

---

## 5. 카운트 기반 탐색

### 5.1 고전적 카운트 기반 방법

```python
class CountBasedExploration:
    """Exploration bonus based on state visitation counts."""

    def __init__(self, bonus_coef=1.0):
        self.counts = {}
        self.bonus_coef = bonus_coef

    def discretize_state(self, state, bins=20):
        """Discretize continuous state for counting."""
        return tuple(np.digitize(state, np.linspace(-5, 5, bins)))

    def get_bonus(self, state):
        """Exploration bonus = β / sqrt(N(s))."""
        key = self.discretize_state(state)
        count = self.counts.get(key, 0)
        self.counts[key] = count + 1
        return self.bonus_coef / np.sqrt(count + 1)
```

### 5.2 대규모 상태 공간을 위한 의사 카운트 방법

```python
class HashCountExploration:
    """SimHash-based pseudo-counts for high-dimensional states."""

    def __init__(self, state_dim, n_hash_bits=32, bonus_coef=0.5):
        self.bonus_coef = bonus_coef
        self.n_hash_bits = n_hash_bits

        # Random projection for SimHash
        self.projection = np.random.randn(n_hash_bits, state_dim)
        self.counts = {}

    def hash_state(self, state):
        """Locality-sensitive hash of state."""
        projection = self.projection @ state
        return tuple((projection > 0).astype(int))

    def get_bonus(self, state):
        h = self.hash_state(state)
        count = self.counts.get(h, 0)
        self.counts[h] = count + 1
        return self.bonus_coef / np.sqrt(count + 1)
```

---

## 6. 보상 해킹과 정렬 오류

### 6.1 보상 해킹 사례

```
유명한 보상 해킹 사례:

1. 보트 레이싱 게임:
   보상: 체크포인트 통과 시 점수
   해킹: 에이전트가 같은 체크포인트를 반복적으로 돌며 통과
   레이스를 끝내지 않지만 무한 점수 획득

2. 블록 쌓기:
   보상: 가장 높은 타워의 높이
   해킹: 에이전트가 테이블을 뒤집어 바닥을 "타워"로 만듦

3. 청소 로봇:
   보상: 보이는 먼지 하나당 -1
   해킹: 카메라 센서를 가림 (보이는 먼지 없음 = 최대 보상)

4. CoastRunners 게임:
   보상: 높은 점수
   기대: 레이스 완주
   해킹: 터보 아이템 루프를 찾아 반복적으로 불에 탐

근본 원인: 보상 함수가 의도한 행동을 완전히 포착하지 못함
```

### 6.2 보상 설계 원칙

```
좋은 보상 설계를 위한 원칙:

1. 무엇을 지정하고, 어떻게는 지정하지 마라
   나쁨: R = 각 하위 과제 단계별 보상의 합
   좋음: R = 1 (과제 완료 시), 그 외 0 + 속도를 위한 PBRS

2. POTENTIAL 기반 형상화 사용
   학습 신호를 추가하면서 최적 정책 보존

3. 보상 크기 문제 회피
   보상을 합리적 범위 [-1, 1] 또는 [0, 1]로 정규화
   큰 크기 차이는 학습을 혼란스럽게 함

4. 퇴화 해를 테스트
   질문: "이 보상을 최대화하는 가장 게으른 방법은?"
   답이 의도한 행동이 아니면 재설계

5. 희소 + 형상화 결합
   정확성을 위한 희소, 속도를 위한 형상화
   R_total = R_희소 + clip(R_형상화, -max_shape, max_shape)
```

---

## 7. 실전 보상 엔지니어링

### 7.1 보상 설계 템플릿

```python
class RewardDesigner:
    """Template for composing reward functions safely."""

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.components = []

    def add_sparse_goal(self, goal_fn, reward=1.0):
        """Add sparse reward for goal achievement."""
        self.components.append(('sparse', goal_fn, reward))

    def add_potential_shaping(self, potential_fn, weight=1.0):
        """Add potential-based shaping (policy-invariant)."""
        self.components.append(('potential', potential_fn, weight))

    def add_penalty(self, penalty_fn, weight=-0.01):
        """Add small penalty (e.g., for time or energy)."""
        self.components.append(('penalty', penalty_fn, weight))

    def compute_reward(self, state, next_state, action, info=None):
        """Compute total reward from all components."""
        total = 0

        for comp_type, fn, weight in self.components:
            if comp_type == 'sparse':
                total += weight * fn(next_state)
            elif comp_type == 'potential':
                shaping = self.gamma * fn(next_state) - fn(state)
                total += weight * shaping
            elif comp_type == 'penalty':
                total += weight * fn(state, action)

        return total


# Example usage
designer = RewardDesigner(gamma=0.99)

# Sparse goal
designer.add_sparse_goal(
    lambda s: 1.0 if np.linalg.norm(s[:2] - np.array([5, 5])) < 0.1 else 0.0,
    reward=10.0
)

# Distance-based shaping (potential-based, preserves optimal policy)
designer.add_potential_shaping(
    lambda s: -np.linalg.norm(s[:2] - np.array([5, 5])),
    weight=1.0
)

# Small time penalty
designer.add_penalty(
    lambda s, a: 1.0,  # constant penalty per step
    weight=-0.01
)
```

### 7.2 보상 함수 디버깅

```python
def debug_reward_function(env, reward_fn, n_episodes=10, max_steps=200):
    """Visualize reward statistics to catch design issues."""
    all_rewards = []
    episode_returns = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        rewards = []

        for step in range(max_steps):
            action = env.action_space.sample()
            next_state, _, terminated, truncated, info = env.step(action)

            r = reward_fn(state, next_state, action, info)
            rewards.append(r)
            state = next_state

            if terminated or truncated:
                break

        all_rewards.extend(rewards)
        episode_returns.append(sum(rewards))

    rewards_arr = np.array(all_rewards)
    print("Reward diagnostics:")
    print(f"  Mean:   {rewards_arr.mean():.4f}")
    print(f"  Std:    {rewards_arr.std():.4f}")
    print(f"  Min:    {rewards_arr.min():.4f}")
    print(f"  Max:    {rewards_arr.max():.4f}")
    print(f"  % zero: {(rewards_arr == 0).mean()*100:.1f}%")
    print(f"  Episode returns: {np.mean(episode_returns):.2f} "
          f"+/- {np.std(episode_returns):.2f}")

    # Warning flags
    if rewards_arr.std() > 100 * abs(rewards_arr.mean()):
        print("  WARNING: Very high variance relative to mean!")
    if (rewards_arr == 0).mean() > 0.99:
        print("  WARNING: >99% zero rewards - too sparse!")
    if rewards_arr.min() < -100:
        print("  WARNING: Very large negative rewards - check penalties")
```

---

## 8. 연습문제

### 연습문제 1: Potential-Based Reward Shaping

PBRS를 구현하고 검증하세요:
1. (9,9)에 희소 보상이 있는 10x10 그리드 월드 생성
2. 형상화 없이 Q-learning 학습: 수렴까지 에피소드 수 측정
3. 맨해튼 거리를 사용한 potential-based shaping 추가
4. 최적 정책이 변경되지 않았음을 확인하면서 수렴 속도 향상 시연
5. NON-potential 기반 형상화를 시도하고 최적 정책이 변경됨을 시연

### 연습문제 2: ICM 호기심 모듈

탐색을 위한 ICM을 구축하고 학습하세요:
1. 완전한 ICM (인코더 + 순방향 + 역방향 모델) 구현
2. 막다른 길과 희소 보상이 있는 2D 미로 생성
3. 비교: 엡실론-그리디, ICM 호기심, RND
4. 각 방법의 탐색 히트맵 시각화
5. 호기심 기반 탐색이 목표를 더 빨리 찾는 것을 시연

### 연습문제 3: RND 구현

RND를 구현하고 어려운 탐색 문제에서 테스트하세요:
1. 타겟 및 예측 네트워크로 RND 모듈 구축
2. 이동 통계를 사용한 보상 정규화 구현
3. MountainCar에서 테스트 (정상에서만 희소 보상)
4. 학습 과정에서 내적 보상 크기 그래프
5. 익숙한 상태에서 내적 보상이 감소하는 것을 시연

### 연습문제 4: 보상 해킹 감지

보상 해킹 시나리오를 만들고 감지하세요:
1. 악용 가능한 보상 함수가 있는 간단한 환경 설계
2. RL 에이전트를 학습시키고 악용을 찾는 것을 관찰
3. 해킹 감지를 위한 모니터링 추가 (보상 vs 의도된 지표)
4. 해킹에 강한 보상 함수 재설계
5. 원래 해킹과 수정 사항을 문서화

### 연습문제 5: 탐색 방법 비교

탐색 방법의 종합 비교:
1. 도전적인 미로 환경 생성 (여러 방, 희소 보상)
2. 구현: 엡실론-그리디, Boltzmann, 카운트 기반, ICM, RND
3. 각각 1M 스텝 실행, 커버리지 및 최종 성능 측정
4. 실측 시간 비교 생성 (일부 방법은 오버헤드가 높음)
5. 다양한 시나리오에서 어떤 방법을 사용할지 추천

---

*21강 끝*
