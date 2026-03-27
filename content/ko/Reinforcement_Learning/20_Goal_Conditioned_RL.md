[이전: 오프라인 RL](./19_Offline_RL.md)

---

# 20. 목표 조건부 강화학습 (Goal-Conditioned RL)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 목표 조건부 RL 프레임워크와 범용 가치 함수를 설명할 수 있습니다
2. 희소 보상 환경을 위한 Hindsight Experience Replay (HER)를 구현할 수 있습니다
3. 재라벨링 전략을 사용한 목표 조건부 정책을 구축할 수 있습니다
4. 다목표 학습과 목표 표현 설계를 이해할 수 있습니다
5. 로봇 조작 작업에 목표 조건부 방법을 적용할 수 있습니다

---

## 목차

1. [목표 조건부 프레임워크](#1-목표-조건부-프레임워크)
2. [범용 가치 함수 근사기](#2-범용-가치-함수-근사기)
3. [Hindsight Experience Replay (HER)](#3-hindsight-experience-replay-her)
4. [목표 재라벨링 전략](#4-목표-재라벨링-전략)
5. [목표 표현 학습](#5-목표-표현-학습)
6. [로봇 조작 응용](#6-로봇-조작-응용)
7. [고급 목표 조건부 방법](#7-고급-목표-조건부-방법)
8. [연습 문제](#8-연습-문제)

---

## 1. 목표 조건부 프레임워크

### 1.1 고정 목표에서 가변 목표로

표준 RL은 단일 목표를 위한 단일 정책을 학습합니다. 목표 조건부 RL은 *어떤* 지정된 목표도 달성할 수 있는 정책을 학습합니다.

```
표준 RL:
  π(a | s)          정책은 상태에만 의존
  목표는 암묵적 (누적 보상 최대화)

목표 조건부 RL:
  π(a | s, g)       정책은 상태와 목표에 의존
  하나의 정책으로 다양한 목표를 달성 가능!

예시: 로봇 팔
  표준: 위치 (5, 3)에 도달하도록 학습
  목표 조건부: 어떤 위치 (x, y)에든 도달하도록 학습

  π(a | s, g=(5,3)) → (5,3)에 도달하기 위한 행동
  π(a | s, g=(1,7)) → (1,7)에 도달하기 위한 행동
  같은 정책, 다른 목표!
```

### 1.2 목표 조건부 MDP

```
표준 MDP:    (S, A, T, R, γ)
목표 조건부: (S, A, G, T, R_g, γ)

여기서:
  G = 목표 공간 (상태 공간과 같거나 다를 수 있음)
  R_g(s, a, g) = 목표에 의존하는 보상 함수

일반적인 보상 함수:
  희소:   R(s, a, g) = 1 (||s' - g|| < ε인 경우),  그 외 0
  밀집:   R(s, a, g) = -||s' - g||₂           (음의 거리)
  이진:   R(s, a, g) = -1 (목표에 아닌 경우)    (단계당 페널티)
```

### 1.3 목표 조건화가 중요한 이유

```python
import numpy as np

def demonstrate_gc_advantage():
    """목표 조건부 학습의 샘플 효율성을 보여줍니다."""
    # GC 없이: 각 목표에 대해 별도 정책 학습 필요
    n_goals = 100
    episodes_per_goal = 1000
    total_standard = n_goals * episodes_per_goal  # 100,000 에피소드

    # GC 사용: 단일 정책이 모든 목표를 동시에 학습
    # + HER이 각 에피소드의 학습 신호를 증폭
    total_gc = 10000  # 10,000 에피소드, HER이 다목표 학습 제공
    her_multiplier = 4  # 실제 에피소드당 4개의 재라벨링 목표
    effective_gc = total_gc * (1 + her_multiplier)  # 50,000 유효 에피소드

    print(f"표준 접근법: {total_standard:,} 에피소드")
    print(f"GC + HER 접근법: {total_gc:,} 에피소드 "
          f"({effective_gc:,} 유효)")
    print(f"효율성 향상: {total_standard / total_gc:.0f}배")

demonstrate_gc_advantage()
```

---

## 2. 범용 가치 함수 근사기

### 2.1 UVFA 아키텍처

범용 가치 함수 근사기(UVFA)는 Q-함수를 목표에 대해 조건화하도록 확장합니다:

```
표준 Q:  Q(s, a)     → 스칼라 값
UVFA Q:  Q(s, a, g)  → 스칼라 값 (목표에 조건화)

아키텍처 옵션:

옵션 A: 연결
  [s, a, g] → MLP → Q(s,a,g)

옵션 B: 별도 인코더 + 결합
  s → encoder_s → φ(s)
  g → encoder_g → ψ(g)     → 결합 → Q
  a → encoder_a → α(a)

옵션 C: 관계형 (어텐션 기반)
  s, g → 교차 어텐션 → Q
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalConditionedQNetwork(nn.Module):
    """목표에 조건화된 Q-네트워크."""

    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.net(x)


class GoalConditionedPolicy(nn.Module):
    """목표에 조건화된 결정론적 정책."""

    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim=256,
                 max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        return self.net(x) * self.max_action
```

### 2.2 UVFA 학습

```python
class GoalConditionedDDPG:
    """목표 조건화를 가진 DDPG 에이전트."""

    def __init__(self, state_dim, action_dim, goal_dim,
                 hidden_dim=256, lr=1e-3, gamma=0.98, tau=0.005):
        self.gamma = gamma
        self.tau = tau

        self.actor = GoalConditionedPolicy(state_dim, goal_dim, action_dim, hidden_dim)
        self.critic = GoalConditionedQNetwork(state_dim, action_dim, goal_dim, hidden_dim)
        self.target_actor = GoalConditionedPolicy(state_dim, goal_dim, action_dim, hidden_dim)
        self.target_critic = GoalConditionedQNetwork(state_dim, action_dim, goal_dim, hidden_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, goal, noise=0.1):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        goal_t = torch.FloatTensor(goal).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_t, goal_t).squeeze(0).numpy()

        action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def train_step(self, states, actions, rewards, next_states, goals, dones):
        """목표 정보를 가진 배치에서 학습합니다."""
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        goals = torch.FloatTensor(goals)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 비평가 업데이트
        with torch.no_grad():
            next_actions = self.target_actor(next_states, goals)
            target_q = self.target_critic(next_states, next_actions, goals)
            target = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(states, actions, goals)
        critic_loss = F.mse_loss(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 액터 업데이트
        pred_actions = self.actor(states, goals)
        actor_loss = -self.critic(states, pred_actions, goals).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 소프트 목표 업데이트
        for p, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return critic_loss.item(), actor_loss.item()
```

---

## 3. Hindsight Experience Replay (HER)

### 3.1 희소 보상 문제

```
핵심 문제:
  목표: 위치 (5, 3)에 도달
  보상: ||위치 - (5,3)|| < 0.1일 때만 +1
  그 외: 0

  무작위 탐색으로 우연히 목표에 도달할 확률 ≈ 0
  보상 신호 없음 → 학습 없음!

  100만 에피소드를 해도 에이전트가 (5,3)에 도달하지 못할 수 있습니다.
```

### 3.2 HER의 핵심 통찰

**핵심 통찰**: 실패한 에피소드도 무언가를 가르쳐줍니다. (5,3)에 도달하려 했지만 (2,7)에 도착했다면, 목표를 (2,7)로 재라벨링하고 "(2,7)에 성공적으로 도달했다!"고 말할 수 있습니다.

```
원래 경험:
  목표: (5,3)
  궤적: s₀ → s₁ → s₂ → s₃ = (2,7)
  보상: 0, 0, 0, 0 ((5,3)에 도달하지 못함)
  학습 신호: 없음

HER 재라벨링:
  재라벨링 목표: (2,7)  [실제로 도착한 곳]
  궤적: s₀ → s₁ → s₂ → s₃ = (2,7)
  보상: 0, 0, 0, 1 (재라벨링 목표에 도달!)
  학습 신호: (2,7)에 도달하는 방법을 학습

  많은 재라벨링 에피소드에 걸쳐, 에이전트는 많은 목표에 도달하는 법을 배웁니다.
  결국, 실제 원하는 목표에도 도달하는 법을 배우게 됩니다!
```

### 3.3 HER 구현

```python
class HindsightExperienceReplay:
    """사후 목표로 에피소드를 보강하는 HER 버퍼."""

    def __init__(self, capacity=1_000_000, goal_strategy='future',
                 n_sampled_goals=4, reward_fn=None):
        self.capacity = capacity
        self.goal_strategy = goal_strategy
        self.n_sampled_goals = n_sampled_goals
        self.reward_fn = reward_fn or self._default_reward

        self.episodes = []
        self.transitions = []  # 샘플링을 위한 평탄한 리스트

    @staticmethod
    def _default_reward(achieved_goal, desired_goal, threshold=0.05):
        """희소 보상: 달성하면 0, 아니면 -1."""
        dist = np.linalg.norm(achieved_goal - desired_goal)
        return 0.0 if dist < threshold else -1.0

    def store_episode(self, episode):
        """
        에피소드를 저장하고 HER 재라벨링 전이를 생성합니다.

        episode: 키를 가진 딕셔너리 리스트:
            'state', 'action', 'next_state', 'achieved_goal',
            'desired_goal', 'done'
        """
        T = len(episode)

        # 원래 전이 저장
        for t, transition in enumerate(episode):
            self.transitions.append({
                'state': transition['state'],
                'action': transition['action'],
                'reward': self.reward_fn(
                    transition['achieved_goal'],
                    transition['desired_goal']
                ),
                'next_state': transition['next_state'],
                'goal': transition['desired_goal'],
                'done': transition['done'],
            })

            # HER 목표 생성
            her_goals = self._sample_her_goals(episode, t)

            for goal in her_goals:
                her_reward = self.reward_fn(
                    transition['achieved_goal'], goal
                )
                her_done = (her_reward == 0.0)

                self.transitions.append({
                    'state': transition['state'],
                    'action': transition['action'],
                    'reward': her_reward,
                    'next_state': transition['next_state'],
                    'goal': goal,
                    'done': her_done,
                })

        # 용량 초과 시 자르기
        if len(self.transitions) > self.capacity:
            self.transitions = self.transitions[-self.capacity:]

    def _sample_her_goals(self, episode, current_idx):
        """지정된 전략을 사용하여 목표를 샘플링합니다."""
        T = len(episode)
        goals = []

        if self.goal_strategy == 'future':
            # 이 에피소드의 미래 달성 목표에서 샘플링
            future_indices = list(range(current_idx + 1, T))
            if not future_indices:
                return goals

            n = min(self.n_sampled_goals, len(future_indices))
            selected = np.random.choice(future_indices, n, replace=False)

            for idx in selected:
                goals.append(episode[idx]['achieved_goal'].copy())

        elif self.goal_strategy == 'final':
            # 최종 달성 목표 사용
            goals.append(episode[-1]['achieved_goal'].copy())

        elif self.goal_strategy == 'episode':
            # 에피소드의 임의의 달성 목표에서 샘플링
            indices = np.random.randint(0, T, self.n_sampled_goals)
            for idx in indices:
                goals.append(episode[idx]['achieved_goal'].copy())

        return goals

    def sample(self, batch_size):
        """전이 배치를 샘플링합니다."""
        indices = np.random.randint(0, len(self.transitions), batch_size)
        batch = [self.transitions[i] for i in indices]

        return {
            'states': np.array([t['state'] for t in batch]),
            'actions': np.array([t['action'] for t in batch]),
            'rewards': np.array([t['reward'] for t in batch]),
            'next_states': np.array([t['next_state'] for t in batch]),
            'goals': np.array([t['goal'] for t in batch]),
            'dones': np.array([t['done'] for t in batch], dtype=float),
        }
```

---

## 4. 목표 재라벨링 전략

### 4.1 전략 비교

```
HER 목표 재라벨링 전략:

1. 'future' (기본값, 최고 성능):
   같은 에피소드의 미래 상태에서 목표 선택
   장점: 가장 유익 (전진 진행 학습)
   단점: 에피소드 끝에 치우침

2. 'final':
   항상 최종 상태를 재라벨링 목표로 사용
   장점: 간단, 항상 양의 신호 제공
   단점: 목표 다양성 낮음

3. 'episode':
   에피소드의 임의 상태에서 샘플링
   장점: 최대 다양성
   단점: 이미 달성된 목표로 재라벨링 가능 (덜 유익)

4. 'random':
   이전에 관찰된 임의의 달성 목표에서 샘플링
   장점: 에피소드 간 다양한 목표
   단점: 현재 궤적에서 너무 멀 수 있음

성능 순위 (일반적):
  future > episode > final > random
```

---

## 5. 목표 표현 학습

### 5.1 상태 기반 vs 학습된 목표

```
목표 표현:

1. 상태 기반 (간단):
   g = 원하는 상태 (또는 상태의 부분집합)
   목표 공간 = 상태 공간일 때 작동
   예시: g = (x, y) 목표 위치

2. 이미지 기반:
   g = 원하는 설정을 보여주는 목표 이미지
   목표 임베딩 학습 필요

3. 언어 기반:
   g = "빨간 블록을 파란 블록 위에 놓아라"
   언어 그라운딩 필요

4. 학습된 잠재 목표:
   g = z ∈ R^d, 학습된 표현
   추상적 목표를 포착할 수 있음
```

### 5.2 대조적 목표 표현

```python
class ContrastiveGoalEncoder(nn.Module):
    """대조 학습을 사용한 목표 표현 학습."""

    def __init__(self, state_dim, embedding_dim=64, temperature=0.1):
        super().__init__()
        self.temperature = temperature

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, state):
        return F.normalize(self.encoder(state), dim=-1)

    def contrastive_loss(self, states, goals, negative_goals):
        """
        목표 표현 학습을 위한 InfoNCE 손실.
        양의 쌍: 같은 궤적의 (상태, 달성_목표)
        음의 쌍: 다른 궤적의 (상태, 무작위_목표)
        """
        state_embed = self.forward(states)
        goal_embed = self.forward(goals)
        neg_embed = self.forward(negative_goals)

        # 양의 유사도
        pos_sim = (state_embed * goal_embed).sum(dim=-1) / self.temperature

        # 음의 유사도
        neg_sim = torch.bmm(
            neg_embed, state_embed.unsqueeze(-1)
        ).squeeze(-1) / self.temperature

        # InfoNCE
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(len(states), dtype=torch.long, device=states.device)
        loss = F.cross_entropy(logits, labels)

        return loss
```

---

## 6. 로봇 조작 응용

### 6.1 Gymnasium Robotics 환경

```python
# Gymnasium Robotics의 목표 조건부 환경
# pip install gymnasium-robotics

import gymnasium as gym

# FetchReach: 그리퍼를 목표 위치로 이동
env = gym.make('FetchReach-v3')

# FetchPush: 블록을 목표 위치로 밀기
env = gym.make('FetchPush-v3')

# FetchSlide: 퍽을 목표로 슬라이드 (도달 범위 밖)
env = gym.make('FetchSlide-v3')

# FetchPickAndPlace: 물체를 집어서 놓기
env = gym.make('FetchPickAndPlace-v3')

# 관찰 구조:
obs, info = env.reset()
print(f"관찰: {obs['observation'].shape}")      # 로봇 상태
print(f"달성 목표: {obs['achieved_goal'].shape}")  # 현재 물체 위치
print(f"원하는 목표: {obs['desired_goal'].shape}")  # 목표 위치
```

### 6.2 FetchReach 예제

```python
def train_fetch_reach():
    """FetchReach에서 목표 조건부 에이전트를 학습합니다."""
    env = gym.make('FetchReach-v3')

    obs, _ = env.reset()
    state_dim = obs['observation'].shape[0]
    goal_dim = obs['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]

    agent = GoalConditionedDDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        lr=1e-3,
        gamma=0.98,
    )

    success_rates = train_with_her(
        env, agent,
        n_epochs=50,
        n_cycles=50,
        n_episodes=16,
    )

    print(f"최종 성공률: {success_rates[-1]:.2%}")
    return agent, success_rates
```

### 6.3 다목표 평가

```python
def evaluate_multi_goal(agent, env, n_goals=100):
    """여러 무작위 목표에 대해 에이전트를 평가합니다."""
    successes = 0

    for _ in range(n_goals):
        obs, _ = env.reset()
        state = obs['observation']
        goal = obs['desired_goal']

        for step in range(50):
            action = agent.select_action(state, goal, noise=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            state = obs['observation']

            if info.get('is_success', False):
                successes += 1
                break

            if terminated or truncated:
                break

    success_rate = successes / n_goals
    print(f"다목표 성공률: {success_rate:.2%} ({successes}/{n_goals})")
    return success_rate
```

---

## 7. 고급 목표 조건부 방법

### 7.1 자동 목표 생성

```python
class GoalGAN:
    """에이전트 능력의 경계에서 목표를 생성합니다."""

    def __init__(self, goal_dim, hidden_dim=128, noise_dim=4):
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.noise_dim = noise_dim

    def generate_goals(self, n_goals):
        """적절한 난이도의 목표를 생성합니다."""
        noise = torch.randn(n_goals, self.noise_dim)
        with torch.no_grad():
            goals = self.generator(noise)
        return goals.numpy()
```

### 7.2 사후 목표 순위 (HGR)

```python
def hindsight_goal_ranking(episode, n_goals=4, method='energy'):
    """
    가장 유익한 사후 목표를 선택합니다.
    무작위 미래 목표 대신, 학습 유용성으로 순위를 매깁니다.
    """
    T = len(episode)
    achieved_goals = [ep['achieved_goal'] for ep in episode]

    if method == 'energy':
        # 다양하고 중간 거리에 있는 목표를 선호
        scores = []
        for i in range(T):
            diversity = np.mean([
                np.linalg.norm(achieved_goals[i] - achieved_goals[j])
                for j in range(T) if j != i
            ])
            scores.append(diversity)

        # 상위-k 다양한 목표 선택
        top_indices = np.argsort(scores)[-n_goals:]
        return [achieved_goals[i] for i in top_indices]

    elif method == 'td_error':
        # 에이전트의 TD 오차가 높은 목표를 선호
        # (학습 잠재력이 가장 높은 곳)
        pass

    return [achieved_goals[i] for i in
            np.random.choice(T, min(n_goals, T), replace=False)]
```

### 7.3 RIG: 상상 목표를 이용한 강화학습

```
RIG 프레임워크:
1. 관찰된 상태로 VAE를 학습
2. 사전 분포에서 잠재 목표 z 샘플링 (z ~ prior)
3. z를 디코딩하여 목표 상태 시각화
4. 잠재 공간에서 목표 조건부 정책 학습

장점:
- 이전에 보지 못한 목표를 상상 가능
- 압축된 목표 표현
- 이미지 관찰에서 동작 가능

파이프라인:
  이미지 관찰 → VAE 인코더 → z_현재
  사전 분포에서 z_목표 샘플링
  정책: π(a | z_현재, z_목표)
```

---

## 8. 연습 문제

### 연습 1: HER 처음부터 구현

간단한 2D 도달 작업을 위한 HER을 구축하세요:
1. 에이전트가 목표 위치로 이동하는 2D 환경을 만드세요
2. 희소 이진 보상 사용 (성공 임계값 = 0.05)
3. 'future' 전략과 n_sampled_goals=4로 HER을 구현하세요
4. 학습 곡선 비교: HER 있음 vs HER 없음
5. HER 없이는 에이전트가 학습하지 못함을 보이세요 (희소 보상이 너무 어려움)

### 연습 2: 목표 재라벨링 전략 비교

HER 전략을 체계적으로 비교하세요:
1. 네 가지 전략 모두 구현: future, final, episode, random
2. 같은 도달 환경에서 각각 100 에포크 학습
3. 모든 전략의 성공률 곡선 그리기
4. 각 전략의 목표 다양성 측정 (재라벨링 목표의 분포)
5. 왜 'future'가 일반적으로 가장 잘 작동하는지 설명

### 연습 3: FetchReach에서 목표 조건부 DDPG

완전한 목표 조건부 DDPG 파이프라인을 구축하세요:
1. Gymnasium Robotics에서 FetchReach-v3 환경 설정
2. 쌍둥이 비평가를 가진 GoalConditionedDDPG 구현
3. 'future' 전략으로 HER 버퍼 통합
4. 50 에포크에 걸쳐 성공률 학습 및 보고
5. 학습된 정책 시각화: 다양한 목표로의 궤적 그리기

### 연습 4: 커리큘럼 목표 생성

목표 난이도를 위한 자동 커리큘럼을 구현하세요:
1. 쉬운 목표(초기 상태에 가까운)부터 시작
2. 에이전트가 개선됨에 따라 점진적으로 목표 거리 증가
3. 각 난이도 수준에서 성공률 추적
4. 균일 목표 샘플링과 비교
5. 커리큘럼이 어려운 목표에 대한 더 빠른 학습을 이끌어냄을 보이세요

### 연습 5: 다목표 전이

목표 조건부 작업 간 전이를 시연하세요:
1. FetchReach (그리퍼 위치 지정)에서 에이전트 학습
2. 목표 조건부 정책을 FetchPush로 전이
3. 측정: 사전학습이 얼마나 도움이 되는지?
4. HER로 FetchPush에서 미세 조정
5. 학습 곡선 비교: 처음부터 vs 전이 포함

---

*레슨 20 끝*
