# 09. Actor-Critic 방법론

**난이도: ⭐⭐⭐⭐ (고급)**

## 학습 목표
- Actor-Critic 아키텍처 이해
- Advantage 함수와 GAE 학습
- A2C와 A3C 알고리즘 비교
- PyTorch로 Actor-Critic 구현

---

## 1. Actor-Critic 개요

### 1.1 핵심 아이디어

**Actor**: 정책 π(a|s;θ)를 학습
**Critic**: 가치 함수 V(s;w)를 학습

```
Actor-Critic = Policy Gradient + TD Learning
```

### 1.2 REINFORCE vs Actor-Critic

| REINFORCE | Actor-Critic |
|-----------|--------------|
| 에피소드 종료 후 업데이트 | 매 스텝 업데이트 |
| 실제 리턴 G 사용 | TD Target 사용 |
| 높은 분산 | 낮은 분산, 약간의 편향 |

---

## 2. Advantage 함수

### 2.1 정의

$$A(s, a) = Q(s, a) - V(s)$$

**의미:** 평균보다 얼마나 좋은 행동인가

### 2.2 TD Error를 Advantage로 사용

```
δ_t = r_t + γV(s_{t+1}) - V(s_t)

E[δ_t | s_t, a_t] = Q(s_t, a_t) - V(s_t) = A(s_t, a_t)
```

TD Error는 Advantage의 불편 추정량입니다.

```python
def compute_advantage(rewards, values, next_values, dones, gamma=0.99):
    """1-step Advantage 계산"""
    advantages = []
    for r, v, nv, d in zip(rewards, values, next_values, dones):
        if d:
            advantage = r - v
        else:
            advantage = r + gamma * nv - v
        advantages.append(advantage)
    return advantages
```

---

## 3. A2C (Advantage Actor-Critic)

### 3.1 네트워크 구조

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # 공유 특징 추출(shared feature extraction): Actor와 Critic이 초기 레이어를 공유하는 이유는
        # 위치·속도 같은 저수준 상태 표현이 정책 결정과 가치 추정 모두에 유용하기 때문이다 —
        # 공유를 통해 연산량을 줄이고, 두 헤드 모두에 유익한 표현을 학습하도록 강제한다
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (정책)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
            # 여기서는 활성화 없음 — softmax는 forward()에서 적용하여
            # 수치적으로 더 안정적인 raw logits 상태로 작업할 수 있다
        )

        # Critic은 스칼라 가치 V(s)를 출력하며, 특정 행동과 무관하게
        # *현재 상태*가 평균적으로 얼마나 좋은지를 추정한다
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.shared(state)
        policy = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return policy, value

    def get_action(self, state):
        policy, value = self.forward(state)
        # Categorical 분포(Categorical distribution)는 확률적 행동 선택을 가능하게 해 탐험에 필수적이다;
        # log_prob는 정책 경사(policy gradient) 업데이트에 사용하기 위해 저장된다
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
```

### 3.2 A2C 에이전트

```python
class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 value_coef=0.5, entropy_coef=0.01):
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # 에피소드 버퍼
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.entropies = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        policy, value = self.network(state_tensor)

        dist = torch.distributions.Categorical(policy)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)
        self.entropies.append(dist.entropy())

        return action.item()

    def store(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns(self, next_value):
        """n-step returns 계산"""
        returns = []
        R = next_value

        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)

        return torch.tensor(returns)

    def update(self, next_state):
        # 부트스트래핑(bootstrapping): 에피소드 종료를 기다리는 대신 Critic을 사용해
        # 남은 리턴을 추정한다 — 이것이 Actor-Critic을 에피소드 중간에도 업데이트할 수 있는
        # 온라인 알고리즘으로 만드는 핵심이다
        with torch.no_grad():
            _, next_value = self.network(
                torch.FloatTensor(next_state).unsqueeze(0)
            )
            next_value = next_value.item()

        returns = self.compute_returns(next_value)
        values = torch.cat(self.values).squeeze()
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        # Advantage 계산 시 values를 계산 그래프에서 분리(detach)한다 —
        # Critic 추정값을 고정 기준선(baseline)으로 취급하여 log_probs를 통해서만
        # Actor 손실이 역전파되도록 한다
        advantages = returns - values.detach()

        # 정책 경사(policy gradient) 손실: 기준선이 있는 REINFORCE —
        # 평균보다 좋은 행동의 확률은 높이고, 나쁜 행동의 확률은 낮춘다
        actor_loss = -(log_probs * advantages).mean()
        # MSE 손실로 Critic이 할인된 리턴을 정확히 예측하도록 훈련한다 —
        # 더 정확한 기준선은 시간이 지날수록 Actor 경사의 분산을 줄인다
        critic_loss = F.mse_loss(values, returns)
        # 엔트로피 보너스(entropy bonus)는 지나치게 확신에 찬 정책에 패널티를 주어 탐험을 장려한다;
        # 음수 부호는 엔트로피를 최대화하면서 전체 손실을 최소화하기 때문이다
        entropy_loss = -entropies.mean()

        total_loss = (actor_loss +
                     self.value_coef * critic_loss +
                     self.entropy_coef * entropy_loss)

        # 업데이트
        self.optimizer.zero_grad()
        total_loss.backward()
        # 그래디언트 클리핑(gradient clipping)은 폭발적 그래디언트를 방지한다 —
        # 희소 보상이나 분산이 높은 환경에서 자주 발생하며, 0.5는 보수적인 상한값이다
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        # 버퍼 초기화
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.entropies = []

        return actor_loss.item(), critic_loss.item()
```

### 3.3 A2C 학습

```python
import gymnasium as gym
import numpy as np

def train_a2c(env_name='CartPole-v1', n_episodes=1000, n_steps=5):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(state_dim, action_dim)
    scores = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(reward, done)
            state = next_state
            total_reward += reward
            step_count += 1

            # n-step 업데이트
            if step_count % n_steps == 0 or done:
                agent.update(next_state)

        scores.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Avg: {np.mean(scores[-100:]):.2f}")

    return agent, scores
```

---

## 4. GAE (Generalized Advantage Estimation)

### 4.1 n-step Returns 트레이드오프

| n | 편향 | 분산 |
|---|------|------|
| 1 (TD) | 높음 | 낮음 |
| ∞ (MC) | 낮음 | 높음 |

### 4.2 GAE 공식

모든 n-step advantage를 기하급수적으로 가중 평균:

$$A^{GAE}\_t = \sum\_{l=0}^{\infty} (\gamma \lambda)^l \delta\_{t+l}$$

여기서 δ_t = r_t + γV(s_{t+1}) - V(s_t)

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation"""
    advantages = []
    # 역방향 순회(backward traversal): GAE는 재귀적 합이라 A_t가 A_{t+1}에 의존한다.
    # 앞에서부터 계산하면 미래 델타를 모두 저장해야 하지만, 역방향으로 순회하면
    # gae 누적합을 단일 패스로 계산할 수 있다
    gae = 0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            # 에피소드 경계: 이 시점 이후의 미래 리턴은 현재 궤적에 속하지 않으므로
            # 누적기를 초기화하여 가치 누출(value leakage)을 방지한다
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * next_values[t] - values[t]
            # GAE는 TD-1(낮은 분산, 높은 편향)과 몬테카를로(높은 분산, 낮은 편향)를
            # lambda로 결합한다: lam=0이면 순수 TD, lam=1이면 몬테카를로 리턴
            gae = delta + gamma * lam * gae

        advantages.insert(0, gae)

    return torch.tensor(advantages)
```

### 4.3 GAE 적용 A2C

```python
class A2CWithGAE(A2CAgent):
    def __init__(self, *args, gae_lambda=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.gae_lambda = gae_lambda
        self.next_values = []

    def compute_gae_returns(self):
        """GAE 기반 advantage와 returns"""
        values = torch.cat(self.values).squeeze().tolist()
        next_vals = self.next_values

        advantages = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values[t]
                gae = delta
            else:
                delta = self.rewards[t] + self.gamma * next_vals[t] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        advantages = torch.tensor(advantages)
        returns = advantages + torch.tensor(values)

        return advantages, returns
```

---

## 5. A3C (Asynchronous Advantage Actor-Critic)

### 5.1 핵심 아이디어

여러 워커가 병렬로 환경과 상호작용하며 비동기적으로 그래디언트를 업데이트합니다.

```
┌─────────────────────────────────────┐
│          Global Network             │
│         (Shared Parameters)          │
└──────────┬──────────┬───────────────┘
           │          │
     ┌─────┴────┐  ┌──┴─────┐
     │ Worker 1 │  │ Worker 2│  ...
     │   Env 1  │  │  Env 2  │
     └──────────┘  └─────────┘
```

### 5.2 의사 코드

```python
# 각 워커의 동작
def worker(global_network, optimizer, env):
    local_network = copy(global_network)

    while True:
        # 로컬 네트워크로 경험 수집
        trajectory = collect_trajectory(local_network, env)

        # 그래디언트 계산
        loss = compute_loss(trajectory)
        gradients = compute_gradients(loss, local_network)

        # 비동기 업데이트
        apply_gradients(optimizer, global_network, gradients)

        # 로컬 네트워크 동기화
        local_network.load_state_dict(global_network.state_dict())
```

### 5.3 A2C vs A3C

| A2C | A3C |
|-----|-----|
| 동기 업데이트 | 비동기 업데이트 |
| 배치 처리 | 스트림 처리 |
| 더 안정적 | 더 빠름 (병렬) |
| GPU 효율적 | CPU 효율적 |

**현재 권장:** A2C가 GPU에서 더 효율적이므로 많이 사용됨

---

## 6. 연속 행동 공간 Actor-Critic

```python
class ContinuousActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # 공유 네트워크
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor: 평균과 표준편차
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # log_std를 상태 독립적인 학습 파라미터로 두면 훈련 초기에 정책을 더 단순하고
        # 안정적으로 유지할 수 있다; exp()를 통해 항상 std > 0을 보장한다
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.shared(state)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp()
        value = self.critic(features)
        return mean, std, value

    def get_action(self, state, deterministic=False):
        mean, std, value = self.forward(state)

        if deterministic:
            # 테스트 시에는 평균값을 직접 사용한다 — 탐험 노이즈 없이
            # 알려진 최선의 행동을 원하기 때문에 무작위 샘플링이 필요 없다
            action = mean
        else:
            # Normal 분포(Normal distribution)는 연속 행동 공간을 모델링한다;
            # 샘플링은 학습된 std에 비례하는 확률적 탐험을 추가한다
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()

        return action, value

    def evaluate(self, state, action):
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)

        # 행동 차원에 걸쳐 log_prob를 합산: 독립적인 차원을 가진 다변량 Normal의 경우
        # 결합 로그 확률(joint log-probability)은 주변 확률의 합이다
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)

        return value, log_prob, entropy
```

---

## 7. 학습 안정화 기법

### 7.1 그래디언트 클리핑

```python
# 그래디언트 노름 클리핑
torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)

# 그래디언트 값 클리핑
torch.nn.utils.clip_grad_value_(network.parameters(), clip_value=1.0)
```

### 7.2 학습률 스케줄링

```python
from torch.optim.lr_scheduler import LinearLR

scheduler = LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.1,
    total_iters=total_timesteps
)
```

### 7.3 보상 정규화

```python
class RunningMeanStd:
    def __init__(self):
        self.mean = 0
        self.var = 1
        # 작은 초기 count는 데이터가 들어오기 전 0으로 나누는 것을 방지하면서도
        # 실제 샘플이 충분히 쌓이면 영향이 무시할 수준이 된다
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Welford의 온라인 알고리즘(Welford's online algorithm): 과거 데이터를 저장하지 않고
        # 평균과 분산을 증분적으로 업데이트한다 — 에피소드 수와 무관하게 O(1) 메모리 사용
        self.mean += delta * batch_count / total_count
        self.var = (self.var * self.count + batch_var * batch_count +
                   delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, x):
        # 1e-8 엡실론(epsilon)은 분산이 0에 가까울 때(예: 배치 내 모든 보상이 동일할 때)
        # 0으로 나누는 것을 방지한다
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
```

---

## 8. 실습: LunarLander

```python
def train_lunarlander():
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(
        state_dim, action_dim,
        lr=7e-4, gamma=0.99,
        value_coef=0.5, entropy_coef=0.01
    )

    scores = []
    n_steps = 5

    for episode in range(2000):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.store(reward, done or truncated)
            state = next_state
            total_reward += reward
            steps += 1

            if steps % n_steps == 0 or done or truncated:
                agent.update(next_state)

            if done or truncated:
                break

        scores.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(scores[-100:])
            print(f"Episode {episode + 1}, Avg: {avg:.2f}")

            if avg >= 200:
                print("Solved!")
                break

    return agent, scores
```

---

## 요약

| 구성요소 | 역할 | 학습 대상 |
|---------|------|----------|
| Actor | 정책 | θ (정책 파라미터) |
| Critic | 가치 평가 | w (가치 파라미터) |
| Advantage | 행동 품질 측정 | A = Q - V ≈ δ |

**손실 함수:**
```
L = L_actor + c1 * L_critic + c2 * L_entropy
L_actor = -log π(a|s) * A
L_critic = (V - target)²
L_entropy = -Σ π log π
```

---

## 다음 단계

- [10_PPO_TRPO.md](./10_PPO_TRPO.md) - 신뢰 영역 정책 최적화
