# 10. PPO와 TRPO

**난이도: ⭐⭐⭐⭐ (고급)**

## 학습 목표
- 정책 업데이트의 안정성 문제 이해
- TRPO의 신뢰 영역 개념 학습
- PPO의 클리핑 메커니즘 이해
- PyTorch로 PPO 구현

---

## 1. 정책 최적화의 문제

### 1.1 큰 업데이트의 위험성

정책 경사에서 너무 큰 업데이트는 성능을 급격히 저하시킬 수 있습니다.

```
θ_new = θ_old + α∇J(θ)

문제: α가 크면 정책이 급격히 변해 학습 불안정
해결: 정책 변화를 제한
```

### 1.2 해결 방향

- **TRPO**: KL divergence로 신뢰 영역 제한 (복잡)
- **PPO**: Clipping으로 간단하게 제한

---

## 2. TRPO (Trust Region Policy Optimization)

### 2.1 목표 함수

새 정책과 이전 정책의 비율을 사용:

$$L^{CPI}(\theta) = \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A^{\pi_{old}}(s, a)\right]$$

### 2.2 KL Divergence 제약

$$\text{maximize}_\theta \quad L^{CPI}(\theta)$$
$$\text{subject to} \quad \mathbb{E}[D_{KL}(\pi_{\theta_{old}} || \pi_\theta)] \leq \delta$$

### 2.3 TRPO의 문제점

- 2차 미분(Hessian) 계산 필요

  Hessian은 2차 편미분으로 이루어진 행렬입니다. n개의 파라미터에 대해 O(n²)의 공간이 필요하므로, 수백만 개의 파라미터를 가진 신경망에서는 사실상 사용이 불가능합니다. PPO는 단순한 클리핑(clipping) 제약과 1차 경사 하강법(first-order gradient descent)을 사용하여 이 문제를 회피합니다.

- Conjugate gradient 알고리즘 필요
- 구현이 복잡하고 계산 비용이 높음

---

## 3. PPO (Proximal Policy Optimization)

### 3.1 핵심 아이디어

Clipping을 사용하여 정책 비율을 제한합니다.

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

### 3.2 Clipped 목표 함수

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)\right]$$

```python
def compute_ppo_loss(ratio, advantage, clip_epsilon=0.2):
    """PPO Clipped 손실"""
    # 확률 비율(probability ratio)을 [1-ε, 1+ε]로 클리핑 — 훈련을 불안정하게 만들 수 있는
    # 파국적인 대규모 정책 업데이트를 방지한다; ε=0.2는 최대 20% 변화를 의미한다
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    # 클리핑된 목표와 클리핑되지 않은 목표 중 최솟값을 선택: 이는 비관적 하한(pessimistic bound)이다 —
    # advantage > 0이면 min이 비율을 1+ε 이상으로 증가하지 못하게 막고(이익 상한);
    # advantage < 0이면 비율이 1-ε 아래로 내려가지 못하게 막는다(처벌 상한)
    loss1 = ratio * advantage
    loss2 = clipped_ratio * advantage

    return -torch.min(loss1, loss2).mean()
```

### 3.3 Clipping 직관

```
Advantage > 0 (좋은 행동):
- ratio 증가 → 확률 증가
- 단, ratio > 1+ε 이상은 무시 (급격한 증가 방지)

Advantage < 0 (나쁜 행동):
- ratio 감소 → 확률 감소
- 단, ratio < 1-ε 이하는 무시 (급격한 감소 방지)
```

---

## 4. PPO 전체 구현

### 4.1 PPO 에이전트

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def get_action(self, state, action=None):
        probs, value = self.forward(state)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=10,
        batch_size=64
    ):
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    def collect_rollouts(self, env, n_steps):
        """경험 수집"""
        states, actions, rewards, dones = [], [], [], []
        values, log_probs = [], []

        state, _ = env.reset()

        for _ in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            log_probs.append(log_prob.item())

            state = next_state if not done else env.reset()[0]

        # 마지막 상태의 가치
        with torch.no_grad():
            _, _, _, last_value = self.network.get_action(
                torch.FloatTensor(state).unsqueeze(0)
            )

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'values': np.array(values),
            'log_probs': np.array(log_probs),
            'last_value': last_value.item()
        }

    def compute_gae(self, rollout):
        """GAE 계산"""
        rewards = rollout['rewards']
        values = rollout['values']
        dones = rollout['dones']
        last_value = rollout['last_value']

        advantages = np.zeros_like(rewards)
        # 역방향 순회로 GAE 합을 단일 패스에서 누적한다;
        # last_value는 마지막 스텝의 부트스트랩(bootstrap) 값을 제공한다
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            # 에피소드 종료 시 다음 상태의 기여를 마스킹(masking)한다 —
            # (1 - done)을 곱하면 루프 내 if-분기 없이 종료 전이에서
            # 미래 부트스트랩 값을 깔끔하게 0으로 만든다
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        # Returns = advantages + values: Critic은 이 returns를 예측하도록 학습하고,
        # 0을 중심으로 정규화된 advantages가 Actor 업데이트를 유도한다
        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """PPO 업데이트"""
        advantages, returns = self.compute_gae(rollout)

        # Advantages를 평균 0, 단위 분산으로 정규화 — 환경 간 보상 스케일 차이에 대한
        # 민감도를 줄이고 경사(gradient) 크기를 안정시킨다
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 텐서 변환
        states = torch.FloatTensor(rollout['states'])
        actions = torch.LongTensor(rollout['actions'])
        old_log_probs = torch.FloatTensor(rollout['log_probs'])
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # 여러 에폭 업데이트: 동일한 롤아웃 데이터를 여러 번 재사용하여 샘플 효율성을 높인다 —
        # 이것이 업데이트 후 데이터를 버리는 A2C 같은 온-폴리시(on-policy) 방법에 비한
        # PPO의 핵심 장점이다
        for _ in range(self.update_epochs):
            # 무작위 순열로 미니배치를 구성하면 롤아웃의 시간적 상관관계를 끊어
            # 업데이트 에폭 전반에 걸쳐 경사 분산을 줄인다
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 현재 정책으로 평가
                _, new_log_probs, entropy, values = self.network.get_action(
                    batch_states, batch_actions
                )

                # 로그 공간에서의 비율 계산: exp(log π_new - log π_old)는
                # π_new / π_old와 동일하지만 작은 확률값에서도 오버플로(overflow)를 방지한다
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # 클리핑된 대리 목표(clipped surrogate objective): surr1은 클리핑 없는 업데이트,
                # surr2는 정책 변화를 제한한다; min을 취함으로써 비관적(보수적) 하한이 된다
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 가치 손실
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)

                # 엔트로피 보너스: total_loss를 최소화하면서 엔트로피를 최대화하려 하므로
                # 음수 부호를 붙인다 — 훈련 내내 탐험을 장려한다
                entropy_loss = -entropy.mean()

                # 총 손실
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                # 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                # 경사 클리핑(gradient clipping): 하나의 나쁜 미니배치가
                # 정책을 붕괴시키는 폭주 업데이트를 방지한다
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return actor_loss.item(), critic_loss.item()
```

### 4.2 PPO 학습 루프

```python
import gymnasium as gym

def train_ppo(env_name='CartPole-v1', total_timesteps=100000, n_steps=2048):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)

    timesteps = 0
    episode_rewards = []
    current_episode_reward = 0

    while timesteps < total_timesteps:
        # 롤아웃 수집
        rollout = agent.collect_rollouts(env, n_steps)
        timesteps += n_steps

        # 에피소드 보상 추적
        for r, d in zip(rollout['rewards'], rollout['dones']):
            current_episode_reward += r
            if d:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0

        # PPO 업데이트
        actor_loss, critic_loss = agent.update(rollout)

        # 로깅
        if len(episode_rewards) > 0 and timesteps % 10000 < n_steps:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            print(f"Timesteps: {timesteps}, Avg Reward: {avg_reward:.2f}")

    return agent, episode_rewards
```

---

## 5. PPO 변형들

### 5.1 PPO-Clip (기본)

위에서 구현한 방식입니다.

### 5.2 PPO-Penalty

KL divergence를 페널티로 추가:

```python
def ppo_penalty_loss(ratio, advantage, old_probs, new_probs, beta=0.01):
    policy_loss = (ratio * advantage).mean()

    kl_div = F.kl_div(new_probs.log(), old_probs, reduction='batchmean')

    return -policy_loss + beta * kl_div
```

### 5.3 Clipped Value Loss

가치 함수에도 클리핑 적용:

```python
def clipped_value_loss(values, old_values, returns, clip_epsilon=0.2):
    # 클리핑된 가치
    clipped_values = old_values + torch.clamp(
        values - old_values, -clip_epsilon, clip_epsilon
    )

    # 두 손실 중 큰 값
    loss1 = (values - returns) ** 2
    loss2 = (clipped_values - returns) ** 2

    return 0.5 * torch.max(loss1, loss2).mean()
```

---

## 6. 연속 행동 공간 PPO

```python
class ContinuousPPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        # Actor: PPO에서 연속 행동에 ReLU 대신 Tanh 활성화를 선호하는 이유는
        # 출력을 (-1, 1)로 제한하여 Normal 분포의 평균이 도달 불가능한 값으로 발산하는
        # 극단적인 pre-activation을 방지하기 때문이다
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        # log_std를 0으로 초기화(std=1): 최대 탐험 상태에서 시작하여
        # 훈련이 진행될수록 불확실성을 줄이는 방향으로 학습한다
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        mean = self.actor_mean(state)
        std = self.actor_log_std.exp()
        value = self.critic(state)
        return mean, std, value

    def get_action(self, state, action=None):
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        # 행동 차원에 걸쳐 log_prob를 합산: 차원별로 독립적인 가우시안(Gaussian)을 가정하므로
        # 결합 로그 확률(joint log-prob) = 주변 확률의 합 (인수분해된 분포)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value
```

---

## 7. 하이퍼파라미터 가이드

### 7.1 일반적인 설정

```python
config = {
    # 학습
    'lr': 3e-4,                  # 학습률
    'gamma': 0.99,               # 할인율
    'gae_lambda': 0.95,          # GAE lambda

    # PPO 특정
    'clip_epsilon': 0.2,         # 클리핑 범위
    'update_epochs': 10,         # 업데이트 반복
    'batch_size': 64,            # 미니배치 크기

    # 손실 계수
    'value_coef': 0.5,           # 가치 손실 계수
    'entropy_coef': 0.01,        # 엔트로피 계수

    # 롤아웃
    'n_steps': 2048,             # 롤아웃 길이
    'n_envs': 8,                 # 병렬 환경 수

    # 안정화
    'max_grad_norm': 0.5,        # 그래디언트 클리핑
}
```

### 7.2 환경별 튜닝

| 환경 | lr | n_steps | clip_epsilon |
|------|-----|---------|--------------|
| CartPole | 3e-4 | 128 | 0.2 |
| LunarLander | 3e-4 | 2048 | 0.2 |
| Atari | 2.5e-4 | 128 | 0.1 |
| MuJoCo | 3e-4 | 2048 | 0.2 |

---

## 8. PPO vs 다른 알고리즘

| 알고리즘 | 복잡도 | 샘플 효율 | 안정성 |
|---------|-------|----------|--------|
| REINFORCE | 낮음 | 낮음 | 낮음 |
| A2C | 중간 | 중간 | 중간 |
| TRPO | 높음 | 높음 | 높음 |
| **PPO** | **중간** | **높음** | **높음** |
| SAC | 중간 | 높음 | 높음 |

**PPO의 장점:**
- TRPO 수준의 성능, 구현은 간단
- 다양한 환경에서 안정적
- 하이퍼파라미터 민감도 낮음

---

## 요약

**PPO 핵심:**
```
L^{CLIP} = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

r(θ) = π_θ(a|s) / π_θ_old(a|s)  # 정책 비율
```

**클리핑 효과:**
- 정책 변화를 [1-ε, 1+ε] 범위로 제한
- 급격한 업데이트 방지
- 학습 안정성 확보

---

## 연습 문제

### 연습 1: TRPO vs PPO 트레이드오프 분석

TRPO와 PPO의 이론적·실용적 차이를 분석하세요.

1. TRPO가 KL 발산의 헤시안(Hessian)을 필요로 하는 이유를 설명하세요. n개의 파라미터를 가진 네트워크에서 n×n 헤시안을 계산하고 역행렬을 구하는 계산 복잡도는 얼마인가요?
2. PPO의 클리핑 목적함수가 2차 정보 없이 TRPO의 제약을 어떻게 근사하는지 설명하세요.
3. 레슨 내용을 바탕으로 다음 비교표를 완성하세요:

| 속성 | TRPO | PPO-Clip |
|-----|------|----------|
| 제약 방식 | | |
| 경사 차수 | | |
| 구현 복잡도 | | |
| 메모리 비용 | | |
| 전형적인 실행 속도 | | |

4. PPO의 소프트 클리핑보다 TRPO의 하드 제약이 반드시 필요한 시나리오는 어떤 경우인가요?

### 연습 2: 클리핑 동작 시각화

PPO 클리핑 목적함수가 다양한 비율(ratio) 값에 어떻게 반응하는지 시각화하세요.

1. 어드밴티지 값 A ∈ {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0}와 ε = 0.2에 대해, r ∈ [0.5, 1.5] 범위에서 L^CLIP(r)을 계산하세요:
   ```python
   import numpy as np
   import matplotlib.pyplot as plt

   def l_clip(r, A, eps=0.2):
       clipped = np.clip(r, 1 - eps, 1 + eps)
       return np.minimum(r * A, clipped * A)
   ```
2. 각 어드밴티지 값에 대한 L^CLIP을 r의 함수로 같은 축에 그리세요.
3. 클리핑 경계선(r = 0.8, r = 1.2)을 수직 점선으로 표시하세요.
4. A > 0일 때 클리핑 영역 밖에서 함수가 평탄(기울기 0)한 이유와, A < 0일 때 그렇지 않은 이유를 설명하세요.

### 연습 3: GAE 람다(Lambda) 민감도

GAE 람다 파라미터가 편향-분산 트레이드오프에 미치는 영향을 조사하세요.

1. LunarLander-v2에서 `gae_lambda` ∈ {0.0, 0.5, 0.9, 0.95, 1.0}만 변경하면서 다섯 개의 PPO 에이전트를 200,000 타임스텝 동안 학습시키세요.
2. 각 실행에 대해 기록하세요:
   - 평균 에피소드 보상 (마지막 20 에피소드)
   - 에피소드 보상의 표준 편차
3. 다섯 가지 설정의 학습 곡선을 그리세요.
4. 편향-분산 트레이드오프 측면에서 결과를 설명하세요:
   - λ = 0: 1-스텝 TD 어드밴티지 (낮은 분산, 높은 편향)
   - λ = 1: 몬테카를로 어드밴티지 (높은 분산, 편향 없음)
5. 어떤 λ 값이 가장 잘 작동하나요? 레슨의 기본값 λ = 0.95와 일치하나요?

### 연습 4: PPO-Penalty 구현

Section 5.2의 KL 페널티 방식의 PPO를 구현하고 PPO-Clip과 비교하세요.

1. Section 5.2의 `ppo_penalty_loss(ratio, advantage, old_probs, new_probs, beta)`를 구현하세요.
2. 적응형 베타(adaptive beta) 조정을 추가하세요: 각 업데이트 에폭 후 평균 KL 발산을 측정하여:
   - KL > 1.5 × target_kl이면: beta를 2배로 증가
   - KL < target_kl / 1.5이면: beta를 2로 나눔
   - target_kl = 0.01 사용
3. PPO-Penalty와 PPO-Clip을 CartPole-v1에서 각각 100,000 타임스텝 동안 학습시키세요.
4. 학습 곡선을 그리고 최종 성능을 기록하세요.
5. 적응형 베타(adaptive beta)가 KL 발산을 target_kl 근처로 효과적으로 제어하나요?

### 연습 5: 벡터화 환경(Vectorized Environment) PPO

더 빠른 데이터 수집을 위해 여러 병렬 환경을 활용하도록 PPO를 확장하세요.

1. `gymnasium.vector.SyncVectorEnv`를 사용하여 CartPole-v1 환경 4개를 병렬로 생성하세요.
2. 모든 4개의 환경에서 동시에 n_steps 전이를 수집하도록 `collect_rollouts()`를 수정하여, `(n_steps × 4,)` 형태의 버퍼를 만드세요.
3. 동일한 n_steps에 대해 전체 롤아웃 크기가 4배 커졌는지 확인하세요.
4. 100,000 총 타임스텝 동안 학습하고, Section 4.2의 단일 환경 버전과 실제 실행 시간을 비교하세요.
5. 분석: 4개의 환경을 사용하면 학습 품질이 변하나요, 아니면 속도만 달라지나요? 샘플 다양성 측면에서 이유를 설명하세요.

---

## 다음 단계

- [11_Multi_Agent_RL.md](./11_Multi_Agent_RL.md) - 다중 에이전트 강화학습
