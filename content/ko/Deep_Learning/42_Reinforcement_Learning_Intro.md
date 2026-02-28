[이전: 모델 저장 및 배포](./41_Model_Saving_Deployment.md) | [다음: 퓨샷 학습](./43_Few_Shot_Learning.md)

---

# 42. 강화학습 입문 (Reinforcement Learning Introduction)

## 학습 목표

- 강화학습의 기본 개념과 용어 이해
- MDP (Markov Decision Process) 프레임워크
- Q-Learning과 Value-based 방법
- Policy Gradient 개요
- Deep RL 기초 (DQN)
- PyTorch 구현 및 실습

---

## 1. 강화학습 개요

### 정의와 특징

```
강화학습: 에이전트가 환경과 상호작용하며 보상을 최대화하는 행동을 학습

특징:
1. 시행착오 학습 (Trial and Error)
2. 지연된 보상 (Delayed Reward)
3. 탐색-활용 균형 (Exploration-Exploitation)
4. 순차적 의사결정 (Sequential Decision Making)
```

### 지도학습 vs 강화학습

```
┌─────────────────────────────────────────────────────────────┐
│           Supervised Learning vs Reinforcement Learning      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Supervised Learning                                         │
│  ┌─────────┐    정답     ┌─────────┐                        │
│  │ 입력 x  │ ─────────→ │ 레이블 y │                        │
│  └─────────┘             └─────────┘                        │
│  즉각적인 피드백, 정답 제공                                  │
│                                                              │
│  Reinforcement Learning                                      │
│  ┌─────────┐  행동   ┌─────────┐  보상   ┌─────────┐       │
│  │ 상태 s  │ ──────→ │ 행동 a  │ ──────→ │ 보상 r  │       │
│  └─────────┘         └─────────┘         └─────────┘       │
│       ↑                    │                   │             │
│       └────────────────────┴───────────────────┘             │
│  지연된 피드백, 탐색 필요                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 강화학습 응용 분야

```
게임: AlphaGo, Atari, StarCraft II
로봇: 로봇 제어, 자율 주행
금융: 포트폴리오 최적화, 알고리즘 트레이딩
추천: 개인화 추천, 대화 시스템
자원 관리: 데이터센터 쿨링, 네트워크 최적화
```

---

## 2. MDP (Markov Decision Process)

### 구성 요소

```
MDP = (S, A, P, R, γ)

S: State (상태 집합)
   - 에이전트가 관측하는 환경의 상태
   - 예: 게임 화면, 로봇의 위치/속도

A: Action (행동 집합)
   - 에이전트가 취할 수 있는 행동
   - 예: 상하좌우 이동, 모터 토크

P: Transition Probability (전이 확률)
   - P(s'|s, a): 상태 s에서 행동 a를 취할 때 s'로 전이할 확률

R: Reward (보상 함수)
   - R(s, a, s'): 상태 전이 시 받는 보상

γ: Discount Factor (할인율)
   - 미래 보상의 현재 가치 (0 < γ ≤ 1)
```

### Markov Property

```
미래는 현재 상태에만 의존 (과거 무관):

P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, ..., s_t, a_t)

의미: 현재 상태가 충분한 정보를 담고 있음
```

### 상호작용 루프

```python
# RL 기본 루프
def rl_loop(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 1. 에이전트가 행동 선택
            action = agent.select_action(state)

            # 2. 환경이 다음 상태와 보상 반환
            next_state, reward, done, info = env.step(action)

            # 3. 에이전트가 경험에서 학습
            agent.learn(state, action, reward, next_state, done)

            # 4. 상태 업데이트
            state = next_state
            total_reward += reward

        print(f"Episode {episode}: Total Reward = {total_reward}")
```

---

## 3. Value Functions

### State Value Function (V)

```
V^π(s) = E[G_t | S_t = s, π]

G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
    = Σ_{k=0}^∞ γ^k R_{t+k+1}

의미: 상태 s에서 정책 π를 따를 때 기대되는 누적 보상
```

### Action Value Function (Q)

```
Q^π(s, a) = E[G_t | S_t = s, A_t = a, π]

의미: 상태 s에서 행동 a를 취하고, 이후 π를 따를 때의 기대 보상
```

### Bellman Equation

```python
# Bellman 방정식 (핵심!)

# Value Function
V(s) = max_a [ R(s, a) + γ * Σ P(s'|s,a) * V(s') ]

# Q Function
Q(s, a) = R(s, a) + γ * Σ P(s'|s,a) * max_a' Q(s', a')

# 의미: 현재 가치 = 즉각 보상 + 할인된 미래 가치
```

---

## 4. Q-Learning

### 알고리즘 개요

```
Q-Learning: Model-free, Off-policy 알고리즘

특징:
1. 환경 모델(P) 필요 없음
2. 다른 정책(ε-greedy)으로 수집한 데이터로 최적 정책 학습
3. 테이블 형태로 Q 값 저장
```

### Q-Learning 업데이트

```python
# Q-Learning 업데이트 규칙

Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

# 분해:
# TD Target: r + γ * max_a' Q(s', a')  (목표)
# TD Error: TD Target - Q(s, a)        (오차)
# α: Learning Rate                     (학습률)
```

### PyTorch 구현

```python
import numpy as np

class QLearningAgent:
    """Q-Learning Agent (Tabular) (⭐⭐⭐)"""
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Q-Table 초기화
        self.q_table = np.zeros((state_size, action_size))

    def select_action(self, state):
        """ε-greedy 행동 선택"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)  # 탐색
        else:
            return np.argmax(self.q_table[state])       # 활용

    def learn(self, state, action, reward, next_state, done):
        """Q-Table 업데이트"""
        # TD Target
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])

        # TD Error
        td_error = td_target - self.q_table[state, action]

        # Update
        self.q_table[state, action] += self.lr * td_error

        # Epsilon Decay
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """학습된 정책 반환"""
        return np.argmax(self.q_table, axis=1)
```

---

## 5. Deep Q-Network (DQN)

### 핵심 아이디어

```
문제: 큰 상태 공간에서 Q-Table 불가능
해결: 신경망으로 Q(s, a) 근사

Q(s, a; θ) ≈ Q*(s, a)

핵심 기법:
1. Experience Replay: 경험 재사용으로 효율성 향상
2. Target Network: 학습 안정화
```

### DQN 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     DQN Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  State s                                                     │
│     │                                                        │
│     ▼                                                        │
│  ┌─────────────────────────────────────┐                    │
│  │  Neural Network (CNN/MLP)           │                    │
│  │  Input: State s                     │                    │
│  │  Output: Q(s, a) for all actions    │                    │
│  └─────────────────────────────────────┘                    │
│     │                                                        │
│     ▼                                                        │
│  [Q(s, a_1), Q(s, a_2), ..., Q(s, a_n)]                     │
│     │                                                        │
│     ▼                                                        │
│  Action = argmax_a Q(s, a)                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Experience Replay

```python
from collections import deque
import random

class ReplayBuffer:
    """Experience Replay Buffer (⭐⭐⭐)"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """랜덤 샘플링 — 왜 랜덤인가?
        연속적인 경험은 높은 상관관계를 가집니다(유사한 상태, 유사한 행동).
        상관된 배치로 학습하면 네트워크가 최근 경험에 과적합(Overfitting)되고
        이전 교훈을 잊게 됩니다. 랜덤 샘플링은 이러한 시간적 상관관계를 깨뜨려
        i.i.d.에 가까운 미니배치를 생성하여 SGD를 안정화합니다."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)
```

### DQN PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class QNetwork(nn.Module):
    """Q-Network (MLP) (⭐⭐⭐)"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN Agent (⭐⭐⭐⭐)"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.learn_step = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Q-Networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = copy.deepcopy(self.q_network)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        """ε-greedy 행동 선택"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """DQN 학습"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (with target network)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss and update
        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()
```

### DQN 학습 루프

```python
def train_dqn(env, agent, episodes=500):
    """DQN Training Loop (⭐⭐⭐)"""
    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):  # gym 0.26+
            state = state[0]

        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            result = env.step(action)
            if len(result) == 5:  # gym 0.26+
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_state, reward, done, info = result

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn()

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

    return rewards_history
```

---

## 6. Policy Gradient

### 아이디어

```
Value-based (DQN): Q 함수 학습 → 간접적으로 정책 도출
Policy-based: 정책을 직접 학습

정책 = π_θ(a|s) = P(a|s; θ)

장점:
1. 연속 행동 공간 처리 가능
2. 확률적 정책 학습 가능
3. 수렴 보장 (지역 최적)
```

### Policy Gradient Theorem

```python
# 목표: J(θ) = E[Σ R_t] 최대화

# Gradient:
∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t|s_t) * G_t ]

# G_t: t 시점부터의 누적 보상 (Return)
# log π_θ: 정책의 로그 확률
```

### REINFORCE 알고리즘

```python
class PolicyNetwork(nn.Module):
    """Policy Network (⭐⭐⭐)"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class REINFORCEAgent:
    """REINFORCE (Monte Carlo Policy Gradient) (⭐⭐⭐⭐)"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Episode buffer
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """확률적 행동 선택"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state)

        # Categorical distribution에서 샘플링
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))

        return action.item()

    def store_reward(self, reward):
        """보상 저장"""
        self.rewards.append(reward)

    def learn(self):
        """에피소드 끝에 학습"""
        # Returns 계산 (뒤에서부터)
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize (baseline 효과)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy Gradient 손실
        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss -= log_prob * G  # 음수: gradient ascent

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 버퍼 초기화
        self.log_probs = []
        self.rewards = []

        return loss.item()
```

### REINFORCE 학습

```python
def train_reinforce(env, agent, episodes=1000):
    """REINFORCE Training (⭐⭐⭐)"""
    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            agent.store_reward(reward)

            state = next_state
            total_reward += reward

        # Episode 끝에 학습
        agent.learn()
        rewards_history.append(total_reward)

        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")

    return rewards_history
```

---

## 7. Actor-Critic

### 아이디어

```
REINFORCE 문제: 높은 분산 (Monte Carlo 추정)
해결: Critic으로 Value 추정 → 분산 감소

Actor: 정책 π_θ (행동 결정)
Critic: Value V_φ (상태 평가)
```

### Advantage Function

```python
# Advantage = Q(s,a) - V(s)
# 의미: 평균 대비 해당 행동이 얼마나 좋은가

# TD Error로 추정:
A(s, a) ≈ r + γV(s') - V(s)
```

### Actor-Critic 구현

```python
class ActorCritic(nn.Module):
    """Actor-Critic Network (⭐⭐⭐⭐)"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (Policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic (Value)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        policy = self.actor(features)
        value = self.critic(features)
        return policy, value


class A2CAgent:
    """Advantage Actor-Critic (⭐⭐⭐⭐)"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        policy, _ = self.network(state)
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def learn(self, state, action, reward, next_state, done, log_prob):
        """One-step Actor-Critic Update"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        _, value = self.network(state)
        _, next_value = self.network(next_state)

        # TD Target and Advantage
        td_target = reward + self.gamma * next_value * (1 - done)
        advantage = td_target - value

        # Actor Loss (policy gradient with advantage)
        actor_loss = -log_prob * advantage.detach()

        # Critic Loss (value function)
        critic_loss = advantage.pow(2)

        # Total Loss
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

---

## 8. 환경과 실험

### OpenAI Gym 사용

```python
import gymnasium as gym

# 환경 생성
env = gym.make('CartPole-v1')

# 환경 정보
print(f"State space: {env.observation_space}")      # Box(4,)
print(f"Action space: {env.action_space}")          # Discrete(2)
print(f"State dim: {env.observation_space.shape}")  # (4,)
print(f"Action dim: {env.action_space.n}")          # 2

# 에피소드 실행
state, _ = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # 랜덤 행동
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state

env.close()
```

### 실험 예제: CartPole

```python
def run_experiment():
    """CartPole 실험 (⭐⭐⭐)"""
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n              # 2

    # DQN 에이전트
    agent = DQNAgent(state_dim, action_dim)

    # 학습
    rewards = train_dqn(env, agent, episodes=500)

    # 결과 시각화
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN on CartPole-v1')

    # Moving average
    window = 50
    ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards)), ma, 'r-', linewidth=2)

    plt.savefig('cartpole_dqn.png')
    plt.close()

    env.close()
    return rewards
```

---

## 9. 알고리즘 비교

### 주요 알고리즘 특성

| 알고리즘 | 유형 | On/Off-Policy | 특징 |
|----------|------|---------------|------|
| Q-Learning | Value-based | Off-policy | 테이블, 간단 |
| DQN | Value-based | Off-policy | 신경망, 경험 재현 |
| REINFORCE | Policy-based | On-policy | Monte Carlo, 높은 분산 |
| A2C/A3C | Actor-Critic | On-policy | Advantage, 병렬화 |
| PPO | Actor-Critic | On-policy | 안정적, 실용적 |
| SAC | Actor-Critic | Off-policy | 연속 행동, entropy |

### 선택 가이드

```
이산 행동 공간:
- 간단한 문제: DQN
- 복잡한 문제: PPO

연속 행동 공간:
- 안정적: SAC
- 빠른 학습: PPO

자원 제한:
- A2C (단일 머신)

대규모 병렬:
- A3C, PPO
```

---

## 10. 심화 주제 개요

### Double DQN

```python
# DQN 문제: Q 값 과대평가
# 해결: 행동 선택과 평가를 다른 네트워크로

# 기존 DQN:
target_q = reward + gamma * target_net(next_state).max()

# Double DQN:
best_action = q_net(next_state).argmax()
target_q = reward + gamma * target_net(next_state)[best_action]
```

### Dueling DQN

```python
# Q = V + A (Value + Advantage)
# 상태의 가치와 행동의 이점을 분리

class DuelingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, action_dim)

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q
```

### Prioritized Experience Replay

```python
# 중요한 경험(TD Error 큰)을 더 자주 샘플링
# P(i) ∝ |TD_error_i|^α

# 구현 시 Sum Tree 자료구조 사용
```

---

## 정리

### 핵심 개념

1. **MDP**: 상태, 행동, 보상, 전이로 문제 정의
2. **Bellman Equation**: 현재 가치 = 즉각 보상 + 미래 가치
3. **Q-Learning**: TD로 Q 함수 학습
4. **DQN**: 신경망 + 경험 재현 + 타겟 네트워크
5. **Policy Gradient**: 정책 직접 최적화
6. **Actor-Critic**: Actor + Critic으로 분산 감소

### 실전 팁

```python
# 1. 보상 설계가 핵심
# - Sparse reward → 학습 어려움
# - Shaped reward → 학습 도움 (but 편향 가능)

# 2. 하이퍼파라미터 튜닝
# - Learning rate: 1e-4 ~ 1e-3
# - Gamma: 0.99
# - Epsilon decay: 천천히

# 3. 디버깅
# - Reward 곡선 확인
# - Q 값 분포 모니터링
# - 학습된 정책 시각화
```

---

## 연습 문제

### 연습 1: FrozenLake에서 Q-Learning 구현

`QLearningAgent`와 Gymnasium의 `FrozenLake-v1` 환경을 사용하여:
1. FrozenLake 그리드는 16개의 상태와 4개의 행동을 가집니다. Q-테이블을 모두 0으로 초기화하세요.
2. `lr=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995`로 10,000 에피소드 학습하세요.
3. 학습 후 학습된 Q-테이블 (4×16 행렬)을 출력하고 4×4 그리드에 화살표로 탐욕(greedy) 정책을 시각화하세요.
4. 마지막 1,000 에피소드에서 성공률 (목표에 도달하는 에피소드 비율)을 계산하세요. 성공률이 0.7 근처라면 무엇을 의미하나요? Q-Learning이 이 환경에서 최적 정책을 찾는 것을 보장하나요?

### 연습 2: CartPole-v1에서 DQN 학습 및 수렴 측정

제공된 `DQNAgent`와 `train_dqn`을 사용하여:
1. 기본 하이퍼파라미터로 `CartPole-v1`에서 500 에피소드 학습하세요.
2. 50 에피소드 이동 평균 오버레이를 포함하여 에피소드 보상 곡선을 플롯하세요.
3. CartPole은 이동 평균이 475에 도달하면 "해결"됩니다. DQN이 몇 에피소드가 필요한가요?
4. 실험: 타겟 네트워크를 제거하세요 (`target_update=1`로 설정하면 매 스텝마다 가중치를 복사합니다). 학습 안정성이 어떻게 변하나요? 지도학습의 "움직이는 타겟" 개념을 사용하여 타겟 네트워크가 왜 필요한지 설명하세요.

### 연습 3: 경험 재현(Experience Replay)에 대한 절제 연구

리플레이 버퍼의 중요성을 조사하세요:
1. 전체 리플레이 버퍼(capacity=10,000, 랜덤 샘플링)로 CartPole에서 DQN을 학습하세요.
2. 경험 재현 없이 DQN을 학습하세요: 각 전환을 즉시 처리하고 버립니다 (batch_size=1, 크기 1의 deque).
3. 두 보상 곡선을 플롯하고 최종 성능과 수렴 속도를 비교하세요.
4. 상관된 샘플의 개념을 사용하여 경우 (2)의 불안정성을 설명하세요: 연속적인 전환으로 학습하는 것이 왜 SGD에서 요구하는 i.i.d. 가정을 위반하나요?

### 연습 4: Double DQN 구현

표준 DQN은 동일한 네트워크가 최적의 다음 행동을 선택하고 평가하기 때문에 Q-값을 과대추정합니다. Double DQN은 이 두 단계를 분리합니다:
1. `DQNAgent.learn` 메서드에서 타겟 계산을 수정하세요:
   - `q_network`를 사용하여 최적 다음 행동을 선택하세요: `best_action = q_network(next_states).argmax(dim=1)`.
   - `target_network`를 사용하여 평가하세요: `next_q = target_network(next_states).gather(1, best_action.unsqueeze(1)).squeeze(1)`.
2. 표준 DQN과 Double DQN을 CartPole에서 500 에피소드 학습하세요.
3. 학습 중 평균 Q-값 추정치를 비교하세요 (에피소드당 `current_q.mean().item()`을 로깅). Double DQN이 더 낮고 정확한 Q 추정치를 생성하나요?

### 연습 5: CartPole에서 REINFORCE 학습 및 DQN과 비교

제공된 `REINFORCEAgent`를 사용하여:
1. `CartPole-v1`에서 1,000 에피소드 학습하세요.
2. 연습 2의 DQN 곡선과 함께 에피소드 보상 곡선을 플롯하세요.
3. REINFORCE는 일반적으로 DQN보다 더 불안정합니다. 관찰하고 설명하세요: REINFORCE의 보상 곡선에서 높은 분산(variance)을 유발하는 것은 무엇인가요? 리턴 정규화 단계 (`(returns - returns.mean()) / returns.std()`)가 분산을 줄이는 기준선(baseline) 역할을 어떻게 하나요?
4. 간단한 Actor-Critic을 구현하여 기준선을 추가하세요: 전체 리턴 `G_t`를 사용하는 대신, 작은 가치 네트워크로 예측된 가치 추정치 `V(s_t)`를 빼세요. 이것이 기울기 추정치의 분산을 줄이는지 확인하세요.

---

## 참고 자료

- Sutton & Barto: http://incompleteideas.net/book/the-book.html
- DQN: https://arxiv.org/abs/1312.5602
- Policy Gradient: https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
- OpenAI Spinning Up: https://spinningup.openai.com/
- Gymnasium: https://gymnasium.farama.org/
