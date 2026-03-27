[이전: Reward Shaping](./21_Reward_Shaping.md)

---

# 22. 역강화학습

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 역강화학습을 전문가 시연으로부터의 보상 복원으로 설명
2. 보상 함수 추론을 위한 Maximum Entropy IRL 구현
3. Generative Adversarial Imitation Learning (GAIL) 구축
4. 인간 선호도로부터의 보상 학습 이해 (RLHF 기초)
5. IRL 접근법 비교 및 로보틱스와 정렬에서의 응용

---

## 목차

1. [역강화학습이란?](#1-역강화학습이란)
2. [Maximum Entropy IRL](#2-maximum-entropy-irl)
3. [Deep IRL과 보상 네트워크](#3-deep-irl과-보상-네트워크)
4. [Generative Adversarial Imitation Learning (GAIL)](#4-generative-adversarial-imitation-learning-gail)
5. [선호도로부터의 보상 학습](#5-선호도로부터의-보상-학습)
6. [IRL 응용](#6-irl-응용)
7. [도전과 한계](#7-도전과-한계)
8. [연습문제](#8-연습문제)

---

## 1. 역강화학습이란?

### 1.1 순방향 RL vs 역강화학습

```
순방향 RL:
  주어짐: 환경 + 보상 함수 R
  찾기:   최적 정책 π*
  방향:   R -> π*

역강화학습:
  주어짐: 전문가 시연 D = {τ₁, τ₂, ...}
  찾기:   시연을 설명하는 보상 함수 R
  방향:   π* -> R

  그 다음: 복원된 R을 사용하여 순방향 RL로 새 정책 학습

왜 행동 복제를 사용하지 않는가?
  BC: π(a|s) = argmax P(a|s, D)  [행동을 복사]
  IRL: R = argmax P(D|R), 그 다음 π* = argmax E[Σ R(s,a)]  [의도를 이해]

  IRL은 "왜"를 포착하므로 새로운 상황에 일반화할 수 있음
  BC는 "무엇"만 포착 (특정 행동)
```

### 1.2 보상 모호성 문제

```
여러 보상 함수가 같은 행동을 설명할 수 있습니다!

전문가가 조심스럽게 운전:
  R₁ = -충돌_페널티              (충돌 회피)
  R₂ = +편안함_보상              (부드러운 운전 선호)
  R₃ = -충돌 - 속도_페널티       (전반적으로 신중)
  R₄ = 0 (상수)                  (어떤 정책이든 "최적")

  이 모든 R이 관찰된 조심스러운 운전을 만들어낼 수 있습니다!

해결책:
  - Maximum Entropy IRL: 가장 단순한 설명 선호
  - 특징 매칭: 특징 기대값 매칭
  - Bayesian IRL: R에 대한 분포 유지
```

### 1.3 IRL 문제 정식화

```python
import numpy as np

def feature_expectations(trajectories, feature_fn, gamma=0.99):
    """
    Compute expected feature counts from demonstrations.

    μ_E = E_τ~expert [Σ_t γ^t φ(s_t, a_t)]
    """
    mu = None
    for trajectory in trajectories:
        traj_features = np.zeros_like(feature_fn(trajectory[0][0], trajectory[0][1]))
        for t, (state, action) in enumerate(trajectory):
            traj_features += (gamma ** t) * feature_fn(state, action)

        if mu is None:
            mu = traj_features
        else:
            mu += traj_features

    return mu / len(trajectories)
```

---

## 2. Maximum Entropy IRL

### 2.1 MaxEnt IRL 정식화

```
Maximum Entropy IRL (Ziebart et al., 2008):

가정: 전문가는 Boltzmann-합리적
  P(τ) ∝ exp(R(τ))  여기서 R(τ) = Σ_t r(s_t, a_t)

  높은 보상의 궤적이 기하급수적으로 더 가능성이 높습니다.
  이것은 전문가가 대부분 최적이지만 약간의 잡음을 허용합니다.

목적: 다음을 최대화하는 보상 r(s,a) = θᵀφ(s,a) 찾기
  log P(D|θ) = Σ_{τ∈D} [θᵀμ(τ)] - |D| · log Z(θ)

  여기서 Z(θ) = ∫ exp(θᵀμ(τ)) dτ  (분배 함수)

그래디언트:
  ∇_θ log P(D|θ) = μ_expert - E_π[μ]
  = (전문가 특징 기대값) - (정책 특징 기대값)

수렴 시: E_expert[φ(s,a)] = E_π[φ(s,a)]
  정책의 특징 기대값이 전문가의 것과 일치합니다!
```

### 2.2 MaxEnt IRL 구현

```python
import torch
import torch.nn as nn


class MaxEntIRL:
    """Maximum Entropy Inverse Reinforcement Learning."""

    def __init__(self, feature_dim, lr=0.01, gamma=0.99):
        self.theta = np.zeros(feature_dim)
        self.lr = lr
        self.gamma = gamma

    def reward(self, state, action, feature_fn):
        """Compute reward r(s,a) = θᵀφ(s,a)."""
        features = feature_fn(state, action)
        return self.theta @ features

    def update(self, expert_features, policy_features):
        """
        Gradient step on reward parameters.

        expert_features: average feature expectations from expert demos
        policy_features: average feature expectations from current policy
        """
        gradient = expert_features - policy_features
        self.theta += self.lr * gradient
        return np.linalg.norm(gradient)

    def train(self, expert_demos, feature_fn, env, rl_agent,
              n_iterations=100, n_policy_episodes=50):
        """
        Full MaxEnt IRL training loop.

        1. Compute expert feature expectations (once)
        2. Loop:
           a. Train policy with current reward
           b. Compute policy feature expectations
           c. Update reward parameters
        """
        # Expert feature expectations (computed once)
        expert_mu = feature_expectations(expert_demos, feature_fn, self.gamma)

        for iteration in range(n_iterations):
            # Train forward RL with current reward
            def reward_fn(s, a):
                return self.reward(s, a, feature_fn)

            rl_agent.train(env, reward_fn, n_episodes=n_policy_episodes)

            # Collect policy trajectories
            policy_demos = rl_agent.collect_trajectories(env, n_policy_episodes)
            policy_mu = feature_expectations(policy_demos, feature_fn, self.gamma)

            # Update reward parameters
            grad_norm = self.update(expert_mu, policy_mu)

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}, Gradient norm: {grad_norm:.4f}")

            if grad_norm < 1e-4:
                print("Converged!")
                break

        return self.theta
```

---

## 3. Deep IRL과 보상 네트워크

### 3.1 신경망 보상 함수

```python
class DeepRewardNetwork(nn.Module):
    """Neural network reward function for deep IRL."""

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


class DeepMaxEntIRL:
    """Deep Maximum Entropy IRL with neural reward."""

    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.reward_net = DeepRewardNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=lr)

    def compute_reward(self, states, actions):
        return self.reward_net(states, actions)

    def update(self, expert_states, expert_actions,
               policy_states, policy_actions):
        """Update reward network: increase for expert, decrease for policy."""
        expert_s = torch.FloatTensor(expert_states)
        expert_a = torch.FloatTensor(expert_actions)
        policy_s = torch.FloatTensor(policy_states)
        policy_a = torch.FloatTensor(policy_actions)

        expert_reward = self.reward_net(expert_s, expert_a).mean()
        policy_reward = self.reward_net(policy_s, policy_a).mean()

        # MaxEnt IRL loss: maximize expert reward, minimize policy reward
        loss = -(expert_reward - torch.logsumexp(
            self.reward_net(
                torch.cat([expert_s, policy_s]),
                torch.cat([expert_a, policy_a])
            ).squeeze(), dim=0
        ))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

---

## 4. Generative Adversarial Imitation Learning (GAIL)

### 4.1 GAIL 프레임워크

```
GAIL (Ho & Ermon, 2016)은 IRL을 GAN 문제로 재정식화합니다:

  판별자 D(s,a): 전문가와 에이전트를 구별하려 함
  정책/생성자 π(a|s): 판별자를 속이려 함

  min_π max_D E_expert[log D(s,a)] + E_π[log(1 - D(s,a))]

  수렴 시:
  - D가 전문가와 에이전트를 구별할 수 없음
  - 에이전트의 점유 측도가 전문가와 일치

  MaxEnt IRL 대비 장점:
  - 보상 함수를 명시적으로 복원할 필요 없음
  - 정책을 직접 학습
  - 복잡한 상태/행동 공간으로 확장 가능
```

### 4.2 GAIL 구현

```python
class GAILDiscriminator(nn.Module):
    """Discriminator for GAIL."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

    def reward(self, state, action):
        """GAIL reward: -log(1 - D(s,a))."""
        with torch.no_grad():
            d = self.forward(state, action)
            return -torch.log(1 - d + 1e-8)


class GAIL:
    """Generative Adversarial Imitation Learning."""

    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 d_lr=3e-4, p_lr=3e-4, gamma=0.99):
        self.discriminator = GAILDiscriminator(state_dim, action_dim, hidden_dim)
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=d_lr
        )
        self.gamma = gamma

        # Policy can be any RL algorithm (PPO works well)
        # Omitted here for brevity - use standard PPO implementation

    def update_discriminator(self, expert_states, expert_actions,
                              policy_states, policy_actions):
        """Update discriminator to distinguish expert from policy."""
        expert_s = torch.FloatTensor(expert_states)
        expert_a = torch.FloatTensor(expert_actions)
        policy_s = torch.FloatTensor(policy_states)
        policy_a = torch.FloatTensor(policy_actions)

        # Expert should be classified as 1
        expert_pred = self.discriminator(expert_s, expert_a)
        expert_loss = -torch.log(expert_pred + 1e-8).mean()

        # Policy should be classified as 0
        policy_pred = self.discriminator(policy_s, policy_a)
        policy_loss = -torch.log(1 - policy_pred + 1e-8).mean()

        # Gradient penalty for stability
        gp = self._gradient_penalty(expert_s, expert_a, policy_s, policy_a)

        d_loss = expert_loss + policy_loss + 10.0 * gp

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.item(), expert_pred.mean().item(), policy_pred.mean().item()

    def _gradient_penalty(self, expert_s, expert_a, policy_s, policy_a, lambda_gp=10.0):
        """WGAN-GP style gradient penalty."""
        batch_size = min(len(expert_s), len(policy_s))
        alpha = torch.rand(batch_size, 1)

        interp_s = alpha * expert_s[:batch_size] + (1 - alpha) * policy_s[:batch_size]
        interp_a = alpha * expert_a[:batch_size] + (1 - alpha) * policy_a[:batch_size]
        interp_s.requires_grad_(True)
        interp_a.requires_grad_(True)

        d_interp = self.discriminator(interp_s, interp_a)

        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=[interp_s, interp_a],
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True
        )

        grad_norm = sum(g.reshape(batch_size, -1).norm(2, dim=1) for g in gradients)
        return ((grad_norm - 1) ** 2).mean()

    def get_reward(self, states, actions):
        """Get GAIL reward for policy training."""
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        return self.discriminator.reward(states_t, actions_t).numpy()
```

### 4.3 GAIL 학습 루프

```python
def train_gail(env, expert_demos, n_iterations=1000,
               n_policy_steps=2048, n_d_updates=5, batch_size=64):
    """Full GAIL training loop."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    gail = GAIL(state_dim, action_dim)

    # Flatten expert demonstrations
    expert_states = np.concatenate([t['states'] for t in expert_demos])
    expert_actions = np.concatenate([t['actions'] for t in expert_demos])

    for iteration in range(n_iterations):
        # 1. Collect policy trajectories with GAIL reward
        policy_data = collect_rollouts(env, gail, n_steps=n_policy_steps)

        # 2. Update discriminator
        for _ in range(n_d_updates):
            # Sample expert batch
            idx = np.random.randint(0, len(expert_states), batch_size)
            e_s, e_a = expert_states[idx], expert_actions[idx]

            # Sample policy batch
            idx = np.random.randint(0, len(policy_data['states']), batch_size)
            p_s = policy_data['states'][idx]
            p_a = policy_data['actions'][idx]

            d_loss, e_score, p_score = gail.update_discriminator(
                e_s, e_a, p_s, p_a
            )

        # 3. Update policy with GAIL reward using PPO
        gail_rewards = gail.get_reward(
            policy_data['states'], policy_data['actions']
        )
        # policy.update(policy_data, gail_rewards)  # PPO update

        if (iteration + 1) % 50 == 0:
            print(f"Iter {iteration+1}: D_loss={d_loss:.3f}, "
                  f"Expert={e_score:.3f}, Policy={p_score:.3f}")
```

---

## 5. 선호도로부터의 보상 학습

### 5.1 선호도 기반 보상 학습

```
완전한 시연 대신, 선호도로부터 보상을 학습:

"어떤 궤적이 더 좋은가? A 아니면 B?"

인간이 제공: τ_A > τ_B  (궤적 A가 선호됨)

Bradley-Terry 모델:
  P(τ_A > τ_B) = exp(R(τ_A)) / (exp(R(τ_A)) + exp(R(τ_B)))

  여기서 R(τ) = Σ_t r(s_t, a_t)

이것이 바로 RLHF (Reinforcement Learning from Human Feedback)의 기초입니다!
```

### 5.2 선호도 기반 보상 학습 구현

```python
class PreferenceRewardModel(nn.Module):
    """Learn reward from pairwise preferences."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states, actions):
        """Predict reward for each (s, a) pair."""
        x = torch.cat([states, actions], dim=-1)
        return self.reward_net(x)

    def trajectory_reward(self, trajectory_states, trajectory_actions):
        """Sum of rewards along a trajectory."""
        rewards = self.forward(trajectory_states, trajectory_actions)
        return rewards.sum()

    def preference_loss(self, traj_a_states, traj_a_actions,
                        traj_b_states, traj_b_actions, preference):
        """
        Bradley-Terry preference loss.

        preference: 0 if traj_a preferred, 1 if traj_b preferred
        """
        r_a = self.trajectory_reward(traj_a_states, traj_a_actions)
        r_b = self.trajectory_reward(traj_b_states, traj_b_actions)

        logits = torch.stack([r_a, r_b])
        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.LongTensor([preference])
        )
        return loss
```

### 5.3 RLHF와의 연결

```
RLHF 파이프라인 (Christiano et al., 2017 -> Ouyang et al., 2022):

1. 비교 데이터 수집:
   인간이 출력 쌍에 순위를 매김: (y_A, y_B) -> 선호도

2. 보상 모델 학습:
   r_θ(x, y)를 Bradley-Terry 선호도 손실로 학습

3. 정책 최적화:
   max_π E[r_θ(x, π(x))] - β · KL(π || π_ref)

이것은 언어 모델에 적용된 IRL과 정확히 같습니다!
(24강: RLHF 심층 분석에서 상세히 다룸)
```

---

## 6. IRL 응용

### 6.1 자율주행

```
자율주행을 위한 IRL:
1. 인간 운전 시연 수집
2. 특징 설계: 속도, 차선 위치, 차량 간 거리, 가속도
3. 보상 복원: R = θ₁·속도 + θ₂·차선_중심 + θ₃·안전_거리 + ...
4. 학습된 보상이 인간의 운전 스타일을 포착

수동 설계 보상 대비 장점:
- 미묘한 선호도를 자동으로 포착
- 다른 인간 운전자 -> 다른 보상 함수
- "방어적" vs "공격적" 운전 스타일 모델링 가능
```

### 6.2 로봇 조작

```python
def irl_for_manipulation():
    """Example: Learn manipulation reward from demonstrations."""
    # Features for pick-and-place task
    features = {
        'gripper_to_object': lambda s: -np.linalg.norm(s['grip'] - s['obj']),
        'object_to_goal': lambda s: -np.linalg.norm(s['obj'] - s['goal']),
        'gripper_open': lambda s: s['gripper_width'],
        'object_height': lambda s: s['obj'][2],
        'smoothness': lambda s, a: -np.linalg.norm(a),
    }

    # IRL recovers weights:
    # Likely: high weight on object_to_goal and gripper_to_object
    # Moderate: smoothness (gentle movements)
    # Low: gripper_open (only matters during grasp)
    pass
```

---

## 7. 도전과 한계

### 7.1 주요 도전

```
IRL 도전:

1. 계산 비용:
   IRL은 내부 루프에서 순방향 RL을 풀어야 함
   각 보상 업데이트 -> 정책 재학습 -> 비용이 큼!
   GAIL은 공동 학습으로 이를 줄임

2. 보상 모호성:
   많은 보상이 같은 행동을 설명
   상수 보상 R=0은 항상 "작동"
   정규화 필요 (MaxEnt, 희소성 등)

3. 시연 품질:
   IRL은 시연이 (거의) 최적임을 가정
   잡음이 있거나 차선의 시연 -> 열악한 보상 복원
   신뢰도 기반 IRL이 혼합 품질을 처리할 수 있음

4. 특징 설계:
   선형 IRL은 좋은 특징이 필요
   Deep IRL (GAIL)은 이를 피하지만 더 많은 데이터 필요
   상태만의 보상은 행동 선호도를 놓침

5. 평가:
   복원된 보상의 "정확성" 평가가 어려움
   정책 성능은 간접 측정
   참 보상은 드물게 이용 가능
```

### 7.2 IRL vs 모방 학습 비교

| 방법 | 학습 대상 | 일반화 | 데이터 요구 | 계산량 |
|--------|--------|-------------|------------|---------|
| 행동 복제 | 정책 | 잘 안됨 | 적음 | 적음 |
| DAgger | 정책 | 더 나음 | 중간 | 중간 |
| MaxEnt IRL | 보상 | 잘됨 | 많음 | 매우 많음 |
| GAIL | 정책 | 잘됨 | 중간 | 많음 |
| 선호도 RL | 보상 | 잘됨 | 적음 (쌍) | 많음 |

---

## 8. 연습문제

### 연습문제 1: 선형 MaxEnt IRL

선형 보상으로 MaxEnt IRL을 구현하세요:
1. 수동 설계된 보상 특징이 있는 그리드 월드 생성
2. 참 보상을 사용하여 전문가 시연 생성
3. 보상 가중치를 복원하기 위한 MaxEnt IRL 구현
4. 복원된 가중치와 참 가중치 비교
5. 복원된 보상으로 새 에이전트를 학습시키고 전문가와 비교

### 연습문제 2: GAIL 구현

GAIL을 처음부터 구축하세요:
1. 판별자와 GAIL 학습 루프 구현
2. CartPole 또는 MountainCar에서 전문가 시연 생성
3. PPO를 생성자로 사용하여 GAIL 에이전트 학습
4. 학습 과정에서 판별자 정확도 그래프 (0.5에 접근해야 함)
5. 같은 시연에서 GAIL 성능 vs 행동 복제 비교

### 연습문제 3: Deep IRL 보상 시각화

심층 보상 네트워크를 학습시키고 학습 내용을 시각화하세요:
1. 장애물이 있는 2D 내비게이션 환경 생성
2. 전문가 시연 생성 (장애물을 피하는 최단 경로)
3. MaxEnt IRL을 사용하여 신경 보상 네트워크 학습
4. 학습된 보상을 상태 공간에 대한 히트맵으로 시각화
5. 높은 보상 영역이 전문가가 선호하는 경로에 해당함을 시연

### 연습문제 4: 선호도 기반 보상 학습

선호도 기반 보상 학습을 구현하세요:
1. CartPole에서 다양한 품질의 궤적 생성
2. 합성 선호도 생성 (더 긴 에피소드 선호)
3. Bradley-Terry 선호도 손실을 사용하여 보상 모델 학습
4. 학습된 보상으로 새 정책 학습
5. 같은 양의 인간 피드백에서 BC 및 GAIL과 비교

### 연습문제 5: 운전 스타일 전이를 위한 IRL

IRL을 통한 운전 스타일 전이:
1. 간단한 2D 운전 시뮬레이터 생성 (차선이 있는 고속도로)
2. "신중한"과 "공격적인" 전문가 시연 생성
3. 각 세트에 대해 독립적으로 IRL을 실행하여 두 보상 함수 복원
4. 복원된 보상 가중치 비교 (속도, 거리, 차선 선호도)
5. 각 보상으로 새 에이전트를 학습시키고 스타일 전이 확인

---

*22강 끝*
