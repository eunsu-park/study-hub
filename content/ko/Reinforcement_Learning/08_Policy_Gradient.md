# 08. 정책 경사 (Policy Gradient)

**난이도: ⭐⭐⭐⭐ (고급)**

## 학습 목표
- 정책 기반 방법의 장단점 이해
- 정책 경사 정리 (Policy Gradient Theorem) 유도
- REINFORCE 알고리즘 구현
- Baseline을 통한 분산 감소 기법
- Actor-Critic으로의 연결

---

## 1. 가치 기반 vs 정책 기반

### 1.1 비교

| 특성 | 가치 기반 (DQN) | 정책 기반 |
|------|----------------|----------|
| 학습 대상 | Q(s, a) | π(a\|s) |
| 정책 도출 | Q에서 간접 유도 | 직접 학습 |
| 행동 공간 | 이산 (주로) | 이산 + 연속 |
| 확률적 정책 | 어려움 | 자연스러움 |
| 수렴 | 불안정 가능 | 지역 최적 |

### 1.2 정책 기반의 장점

```
1. 연속 행동 공간 처리 가능 (로봇 제어)
2. 확률적 정책 학습 가능 (가위바위보)
3. 정책 공간이 더 단순할 수 있음
4. 더 나은 수렴 보장 (일부 경우)
```

---

## 2. 정책의 파라미터화

### 2.1 소프트맥스 정책 (이산 행동)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        logits = self.network(state)
        # Softmax는 원시 점수를 유효한 확률 분포로 변환 —
        # 확률 공간에서 작동하면 확률적 행동(stochastic action)을 자연스럽게 샘플링 가능
        return F.softmax(logits, dim=-1)

    def get_action(self, state):
        probs = self.forward(state)
        # Categorical은 확률을 감싸서 한 번에 샘플링과 선택된 행동의 로그 확률을 계산
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        # 확률(prob) 대신 로그 확률(log_prob)을 저장하는 이유: 정책 경사 정리(Policy Gradient Theorem)가
        # ∇ log π를 사용하며, 로그는 곱셈을 덧셈으로 바꿔 수치적으로 안정적인 그래디언트를 제공
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
```

### 2.2 가우시안 정책 (연속 행동)

```python
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        features = self.shared(state)
        mean = self.mean_layer(features)
        std = self.log_std.exp()
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
```

---

## 3. 정책 경사 정리

### 3.1 목표 함수

정책 π_θ의 성능을 최대화:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

여기서 τ = (s₀, a₀, r₀, s₁, a₁, r₁, ...) 는 궤적(trajectory)

### 3.2 정책 경사 정리

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

**직관적 해석:**
- 좋은 결과(높은 G_t)를 가져온 행동의 확률을 높임
- 나쁜 결과를 가져온 행동의 확률을 낮춤

### 3.3 유도 (Log-derivative trick)

```
∇_θ π(a|s;θ) = π(a|s;θ) · ∇_θ log π(a|s;θ)

따라서:
∇_θ J(θ) = E[R · ∇_θ log π(a|s;θ)]
         = E[∇_θ log π(a|s;θ) · R]
```

---

## 4. REINFORCE 알고리즘

### 4.1 기본 REINFORCE

몬테카를로 정책 경사 방법입니다.

```python
class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = DiscretePolicy(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        # 에피소드 저장
        self.log_probs = []
        self.rewards = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.policy.get_action(state_tensor)
        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def compute_returns(self):
        """할인된 리턴 계산"""
        returns = []
        G = 0
        # 역방향 순회를 통해 G에 미래 보상을 누적 —
        # 매 스텝마다 전체 합산을 새로 계산하는 것을 피하는 단일 역방향 패스
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        # 리턴을 평균 0, 분산 1로 정규화 — 에피소드 길이에 무관하게
        # 그래디언트 크기를 안정시키는 동적 기준선(dynamic baseline) 역할
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self):
        returns = self.compute_returns()

        # 정책 손실 계산
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            # 음수 부호는 경사 상승(gradient ascent, J 최대화)을 경사 하강으로 전환 —
            # Adam과 같은 표준 옵티마이저를 수정 없이 그대로 사용 가능
            policy_loss.append(-log_prob * G)  # 음수 (경사 상승)

        loss = torch.stack(policy_loss).sum()

        # 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 에피소드 데이터 초기화
        self.log_probs = []
        self.rewards = []

        return loss.item()
```

### 4.2 학습 루프

```python
import gymnasium as gym
import numpy as np

def train_reinforce(env_name='CartPole-v1', n_episodes=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCE(state_dim, action_dim, lr=1e-3)

    scores = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)
            state = next_state
            total_reward += reward

        # 에피소드 종료 후 업데이트
        loss = agent.update()
        scores.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Avg Score: {np.mean(scores[-100:]):.2f}")

    return agent, scores
```

---

## 5. Baseline을 통한 분산 감소

### 5.1 분산 문제

REINFORCE의 그래디언트는 높은 분산을 가집니다.

```
Var(∇_θ J) ∝ E[(G - b)²]
```

### 5.2 Baseline 도입

상수 b를 빼도 기대값은 변하지 않지만 분산은 감소합니다.

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b) \right]$$

가장 좋은 baseline: b = V(s)

```python
class REINFORCEWithBaseline:
    def __init__(self, state_dim, action_dim, lr_policy=1e-3, lr_value=1e-3, gamma=0.99):
        self.policy = DiscretePolicy(state_dim, action_dim)
        self.value = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_value)
        self.gamma = gamma

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.states = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # 정책에서 행동 샘플링
        action, log_prob = self.policy.get_action(state_tensor)

        # 가치 예측
        value = self.value(state_tensor)

        self.log_probs.append(log_prob)
        self.values.append(value)
        self.states.append(state_tensor)

        return action

    def update(self):
        returns = self.compute_returns()

        values = torch.cat(self.values).squeeze()
        log_probs = torch.stack(self.log_probs)

        # detach()는 어드밴티지(advantage)를 통해 가치 네트워크로 그래디언트가
        # 역류하는 것을 차단 — 정책과 가치는 별도의 목적함수로 학습해야 서로 간섭하지 않음
        advantages = returns - values.detach()

        # 정책 손실: 어드밴티지로 log-prob을 스케일링하면 좋은 결과의 행동이
        # 더 강한 그래디언트 업데이트를 받음; 합(sum) 대신 평균(mean)을 사용해
        # 에피소드 길이와 무관하게 손실 크기를 일정하게 유지
        policy_loss = -(log_probs * advantages).mean()

        # 실제 리턴에 대한 MSE는 가치 네트워크가 각 상태에서 기대 누적 보상을
        # 예측하도록 학습 — 이것은 지도 학습(supervised learning)에 해당
        value_loss = F.mse_loss(values, returns)

        # 두 개의 독립된 옵티마이저를 사용해 가치 그래디언트가 정책 파라미터를
        # 오염시키는 것을 방지하고, 그 역방향도 차단
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 가치 업데이트
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 초기화
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.states = []

        return policy_loss.item(), value_loss.item()
```

---

## 6. 연속 행동 공간 예제

### 6.1 연속 행동 REINFORCE

```python
class ContinuousREINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = GaussianPolicy(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        self.log_probs = []
        self.rewards = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.policy.get_action(state_tensor)

        self.log_probs.append(log_prob)
        return action.detach().numpy().squeeze()

    def update(self):
        returns = self.compute_returns()

        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
```

### 6.2 MountainCarContinuous 예제

```python
def train_continuous():
    env = gym.make('MountainCarContinuous-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = ContinuousREINFORCE(state_dim, action_dim, lr=1e-3)

    for episode in range(500):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, done, truncated, _ = env.step(action)
            agent.rewards.append(reward)

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        agent.update()
        print(f"Episode {episode + 1}, Reward: {total_reward:.2f}")
```

---

## 7. 고급 기법

### 7.1 엔트로피 정규화

탐색을 장려하기 위해 정책의 엔트로피를 손실에 추가합니다.

```python
def compute_entropy(probs):
    """정책의 엔트로피 계산"""
    return -(probs * probs.log()).sum(dim=-1).mean()

# 손실 함수
total_loss = policy_loss - entropy_coef * entropy
```

### 7.2 Reward Shaping

희소 보상 문제를 해결하기 위한 보상 변환:

```python
def shape_reward(reward, state, next_state, done):
    """보상 형성 예시"""
    # 원래 보상에 추가적인 시그널
    position_reward = abs(next_state[0] - state[0])  # 움직임 장려

    if done and reward > 0:
        bonus = 100  # 목표 달성 보너스
    else:
        bonus = 0

    return reward + 0.1 * position_reward + bonus
```

---

## 8. REINFORCE의 한계

### 8.1 문제점

1. **높은 분산**: 에피소드 전체를 사용하므로 분산이 큼
2. **샘플 비효율**: 에피소드 종료까지 기다려야 함
3. **크레딧 할당**: 어떤 행동이 좋은 결과를 가져왔는지 파악 어려움

### 8.2 해결책 → Actor-Critic

- TD 학습과 정책 경사의 결합
- 부트스트래핑으로 분산 감소
- 스텝마다 업데이트 가능

---

## 요약

| 알고리즘 | 업데이트 시점 | Baseline | 특징 |
|---------|-------------|----------|------|
| REINFORCE | 에피소드 종료 | 없음 | 단순, 높은 분산 |
| REINFORCE + Baseline | 에피소드 종료 | V(s) | 낮은 분산 |
| Actor-Critic | 매 스텝 | V(s) 또는 Q(s,a) | 효율적 |

**핵심 공식:**
```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) · (G - b)]
```

---

## 연습 문제

### 연습 1: 로그 미분 트릭(Log-Derivative Trick) 유도

정책 경사 정리(Policy Gradient Theorem)를 기초부터 유도하세요.

1. 목표 함수 J(θ) = E_τ[R(τ)]에서 시작하여 궤적에 대한 적분으로 기대값을 전개하세요.
2. ∇_θ p(τ;θ) = p(τ;θ) · ∇_θ log p(τ;θ) (로그 미분 항등식)을 증명하세요.
3. log p(τ;θ)를 정책의 기여분과 환경 동역학의 기여분으로 인수분해하세요.
4. 최종 경사 표현식에서 환경 전이 확률 P(s'|s, a)가 사라지는 이유를 설명하세요.
5. 최종 형태 ∇_θ J(θ) = E[∑_t ∇_θ log π_θ(a_t|s_t) · G_t]를 자신의 언어로 설명하세요.

### 연습 2: 기본 REINFORCE 구현

`REINFORCEWithBaseline` 클래스를 사용하지 않고 CartPole-v1에서 REINFORCE 에이전트를 구현하고 학습하세요.

1. Section 2.1의 `DiscretePolicy` 클래스를 활용하세요.
2. 할인 리턴(discounted return, γ = 0.99)으로 `compute_returns()` 메서드를 구현하세요.
3. 리턴을 정규화하지 마세요 (원시 G_t 값 유지).
4. 500 에피소드 동안 학습하고 에피소드별 보상을 그래프로 그리세요.
5. 학습 곡선의 높은 분산을 관찰하고 그 원인을 설명하세요.

```python
# 시작 코드
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
# ... 여기에 구현
```

### 연습 3: Baseline이 분산에 미치는 효과

서로 다른 기준선(baseline)이 학습 안정성에 미치는 효과를 실험적으로 비교하세요.

1. CartPole-v1에서 각각 500 에피소드씩 세 에이전트를 학습시키세요:
   - 에이전트 A: baseline 없는 REINFORCE (원시 리턴 G_t)
   - 에이전트 B: 리턴 정규화를 사용한 REINFORCE (평균 0, 단위 분산 G_t)
   - 에이전트 C: 학습된 가치 함수 V(s)를 baseline으로 사용하는 REINFORCE
2. 세 에이전트의 에피소드 보상을 모두 기록하세요.
3. 각 에이전트의 이동 평균(윈도우 50 에피소드)과 표준 편차를 그래프로 그리세요.
4. 학습 마지막 100 에피소드의 보상 분산을 계산하세요.
5. 결과를 설명하세요: baseline을 빼도 기대 경사(expected gradient)는 변하지 않으면서 분산이 감소하는 이유는 무엇인가요?

### 연습 4: 연속 제어를 위한 가우시안 정책

REINFORCE 알고리즘을 연속 행동 환경으로 확장하세요.

1. Section 6.1의 `ContinuousREINFORCE`를 `Pendulum-v1` 환경([-2, 2] 범위의 1차원 연속 행동)에 적용하세요.
2. `GaussianPolicy`를 살펴보세요: `log_std`가 학습 가능한 파라미터인지 확인하고, 0으로 초기화(std = 1)하는 이유를 설명하세요.
3. `env.step()` 호출 전에 `np.clip(action, env.action_space.low, env.action_space.high)`를 추가하세요 — 이것이 필요한 이유를 설명하세요.
4. 300 에피소드 동안 학습하고, 에피소드 보상과 함께 `policy.log_std.item()`(학습된 표준 편차)의 변화를 그래프로 그리세요.
5. 표준 편차의 변화가 학습 과정에서 에이전트의 탐험(exploration) 행동에 대해 무엇을 알려주나요?

### 연습 5: 엔트로피 정규화 절제 실험 (Entropy Regularization Ablation)

엔트로피 계수가 탐험과 수렴에 미치는 영향을 조사하세요.

1. `REINFORCE`의 update 메서드를 수정하여 엔트로피 보너스를 추가하세요:
   ```python
   total_loss = policy_loss - entropy_coef * entropy_bonus
   ```
   여기서 `entropy_bonus = compute_entropy(probs)`는 Section 7.1에서 정의된 함수입니다.
2. CartPole-v1에서 `entropy_coef` ∈ {0.0, 0.01, 0.1, 0.5}로 네 에이전트를 학습시키세요.
3. 각 실행에 대해 기록하세요: (a) 마지막 50 에피소드의 평균 보상, (b) 평균 보상 ≥ 195에 도달하는 데 걸린 에피소드 수.
4. 네 가지 설정의 학습 곡선을 같은 그래프에 그리세요.
5. 트레이드오프를 설명하세요: `entropy_coef`가 너무 크면 무슨 문제가 생기나요? 너무 작으면요?

---

## 다음 단계

- [09_Actor_Critic.md](./09_Actor_Critic.md) - Actor-Critic 방법론
