# 07. Deep Q-Network (DQN)

**난이도: ⭐⭐⭐ (중급)**

## 학습 목표
- DQN의 핵심 아이디어와 구조 이해
- Experience Replay의 원리와 구현
- Target Network의 필요성과 동작 방식
- Double DQN, Dueling DQN 등 개선 기법
- PyTorch로 DQN 구현

---

## 1. Q-Learning의 한계와 DQN

### 1.1 테이블 기반 Q-Learning의 한계

```
문제점:
1. 상태 공간이 크면 테이블 저장 불가 (Atari: 256^(84*84*4) 상태)
2. 연속 상태 공간 처리 불가
3. 비슷한 상태 간 일반화 불가
```

### 1.2 함수 근사 (Function Approximation)

```python
# 테이블 대신 신경망으로 Q 함수 근사
# Q(s, a) ≈ Q(s, a; θ)

import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """왜 신경망으로 Q를 근사하는가: 테이블 Q-learning은 모든 (상태, 행동) 쌍의 값을 저장해야
    하므로 상태 공간이 크거나 연속적이면 불가능. 신경망은 유사한 상태 간 일반화(generalization)가
    가능하여 Atari(84x84x4 픽셀 입력) 같은 고차원 환경에서도 학습 가능."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # 왜 모든 행동값을 한번에 출력하는가: 단일 순전파(forward pass)로 모든 행동의
        # Q(s,a)를 얻으면 argmax 선택이 O(1) — 행동별로 순전파하면 O(|A|)
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)  # 모든 행동의 Q값을 동시에 출력
```

---

## 2. DQN의 핵심 기법

### 2.1 Experience Replay

경험을 버퍼에 저장하고 무작위로 샘플링하여 학습합니다.

**장점:**
- 데이터 효율성 향상 (경험 재사용)
- 연속 샘플의 상관관계 제거
- 학습 안정화

```python
from collections import deque
import random

class ReplayBuffer:
    # 왜 경험 리플레이(experience replay)인가: 연속 전이(transition)는 매우 상관되어 있음
    # (s_t와 s_{t+1}이 거의 동일). 상관된 배치로 학습하면 최근 경험에 과적합(overfit)하고
    # 이전 지식을 잊음. 큰 버퍼에서 무작위 샘플링하면 이 상관을 깨고 i.i.d.에 가까운
    # 학습 분포를 제공.
    def __init__(self, capacity=100000):
        # 왜 maxlen이 있는 deque인가: 버퍼가 가득 차면 자동으로 가장 오래된 경험을
        # 제거하여 메모리 한도 내에서 최신 데이터를 유지
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)
```

### 2.2 Target Network

별도의 타겟 네트워크를 사용하여 학습 안정화합니다.

**문제:** Q(s,a;θ) 업데이트 시 타겟 y = r + γ max Q(s',a';θ)도 변함
**해결:** 타겟 네트워크 θ⁻를 고정하고 주기적으로 업데이트

**DQN 손실 함수(Loss Function):**

$$L(\theta) = \mathbb{E}\left[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2\right]$$

여기서:
- $\theta$ = 온라인 네트워크(online network) 매개변수 (매 스텝 업데이트)
- $\theta^-$ = 타겟 네트워크(target network) 매개변수 ($\theta$의 동결된 복사본, N 스텝마다 업데이트)
- $r + \gamma \max_{a'} Q(s', a'; \theta^-)$ = TD 타겟 (안정성을 위해 동결된 $\theta^-$로 계산)

**왜 타겟 네트워크(target network)가 필요한가?** 타겟 네트워크 없이는 예측값 $Q(s,a;\theta)$와 타겟값 $r + \gamma \max_{a'} Q(s',a';\theta)$가 각 경사 하강(gradient step)마다 동시에 변합니다. 이것이 "이동 타겟(moving target)" 문제를 일으켜 — 네트워크가 계속 변하는 타겟을 쫓으며 진동하거나 발산합니다. $\theta^-$를 동결하면 타겟이 현재 매개변수로부터 분리되어, N 스텝 동안 지도 회귀(supervised regression) 문제로 변환됩니다.

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)

        # 왜 동일 초기화인가: 타겟 네트워크가 같은 지점에서 시작하여
        # 초기 TD 타겟이 온라인 네트워크와 일관되도록 보장
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = 0.99

    def update_target_network(self):
        """하드 업데이트(Hard update): N 스텝마다 전체 가중치를 한번에 복사.
        왜 하드 업데이트인가: 간단하며 정확히 N 스텝 동안 타겟이 안정적 —
        온라인 네트워크에 고정된 회귀 타겟(regression target)을 제공."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update_target(self, tau=0.005):
        """소프트 업데이트(Soft update): 매 스텝 가중치를 점진적으로 혼합.
        왜 소프트 업데이트(Polyak averaging)인가: 하드 업데이트의 급격한 변화를 피하여
        일부 환경에서 더 부드러운 타겟으로 안정성을 향상시킬 수 있음."""
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
```

---

## 3. DQN 전체 구현

### 3.1 에이전트 클래스

```python
import numpy as np

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step = 0

        # 네트워크
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def choose_action(self, state, training=True):
        # 왜 감소하는 epsilon-greedy인가: 초기에는 Q 추정이 무작위이므로 많은 탐험으로
        # 다양한 전이를 발견. Q가 개선되면 점차 활용으로 전환.
        # 테스트 시(training=False)에는 순수 탐욕적(pure greedy) 사용.
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        # 왜 최소 버퍼 확인인가: 버퍼가 차기 전의 매우 작은 배치는 상관된 최근 전이에서
        # 높은 분산의 경사(gradient)를 생성
        if len(self.buffer) < self.batch_size:
            return None

        # 왜 무작위 배치 샘플링인가: 연속 전이 간의 시간적 상관을 깨서
        # i.i.d. 학습 세트에 근사
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # 왜 gather인가: 네트워크는 모든 행동의 Q를 출력하므로 gather로 실제
        # 수행된 행동의 Q값만 추출
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 왜 여기서 타겟 네트워크를 사용하는가: 동결된 theta^-로 TD 타겟을 계산하여
        # "이동 타겟(moving target)" 문제를 방지 — target_update_freq 스텝 동안
        # 타겟이 고정되어 표준 지도 회귀(supervised regression) 문제로 변환
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            # 왜 (1 - dones)인가: 종료 상태(terminal state)에는 미래 보상이 없으므로
            # 부트스트랩 값을 0으로 만들어 존재하지 않는 미래 리턴을 환상하는 것을 방지
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # MSE 손실: L(theta) = E[(target_q - current_q)^2]
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # 왜 경사 클리핑(gradient clipping)인가: 심층 네트워크는 특히 Q 추정이 부정확한
        # 학습 초기에 경사 폭발(exploding gradient)이 발생 가능. 클리핑으로 업데이트를
        # 제한하여 치명적인 매개변수 점프를 방지.
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        # 왜 주기적 하드 업데이트인가: N 스텝마다 온라인 가중치를 타겟 네트워크에 복사.
        # 더 잦은 업데이트는 온라인 네트워크를 빠르게 추적하지만 안정성 감소;
        # 덜 잦은 업데이트는 더 안정적이지만 새 지식 반영이 느림.
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # 왜 epsilon_min 하한값인가: 항상 약간의 탐험을 유지하여
        # 차선 정책(suboptimal policy)에 영구 고정되는 것을 방지
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()
```

### 3.2 학습 루프

```python
import gymnasium as gym

def train_dqn(env_name='CartPole-v1', n_episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    rewards_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn()

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")

    return agent, rewards_history
```

---

## 4. DQN 개선 기법

### 4.1 Double DQN

일반 DQN의 Q값 과대추정 문제를 해결합니다.

```python
# 일반 DQN: y = r + γ max_a' Q(s', a'; θ⁻)
# Double DQN: y = r + γ Q(s', argmax_a' Q(s', a'; θ); θ⁻)

def compute_double_dqn_target(self, rewards, next_states, dones):
    # 왜 Double DQN인가: 일반 DQN은 행동 선택과 평가 모두에 max_a' Q(s', a'; theta^-)를
    # 사용. max 연산자는 양의 편향(positive bias)을 가짐 — Q 추정에 잡음이 있으면 max가
    # 가장 과대추정된 값을 선택. 선택(온라인 네트워크)과 평가(타겟 네트워크)를 분리하면
    # 편향이 크게 감소하여 더 정확한 Q값과 더 나은 정책으로 이어짐.
    with torch.no_grad():
        # 왜 온라인 네트워크로 선택하는가: 온라인 네트워크가 가장 최신 추정을 가지므로
        # 최적 행동 식별에 더 적합
        next_actions = self.q_network(next_states).argmax(1, keepdim=True)

        # 왜 타겟 네트워크로 평가하는가: 선택된 행동이 약간 틀리더라도
        # 타겟 네트워크가 더 편향이 적은(less biased) Q 추정을 제공
        next_q = self.target_network(next_states).gather(1, next_actions).squeeze()

        target_q = rewards + self.gamma * next_q * (1 - dones)

    return target_q
```

### 4.2 Dueling DQN

Q 함수를 V(상태 가치)와 A(어드밴티지)로 분해합니다.

```
Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))
```

```python
class DuelingQNetwork(nn.Module):
    """왜 Dueling 아키텍처인가: 많은 상태에서 어떤 행동을 취하든 가치가 비슷함
    (예: 장애물이 없으면 모든 행동이 거의 동등). V(s)와 A(s,a)를 분리하면
    행동 선택이 중요하지 않은 상태에서 상태 가치를 독립적으로 학습하여
    샘플 효율성(sample efficiency) 향상."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # 왜 공유 특징인가: V와 A 모두 상태를 이해해야 하므로 초기 레이어를
        # 공유하면 중복 계산을 피하고 특징 재사용(feature reuse) 향상
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # 가치 스트림 (V)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 어드밴티지 스트림 (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # 왜 mean(A)를 빼는가: 분해를 식별 가능(identifiable)하게 만듦 — 중심화
        # 없이는 V와 A가 유일하게 결정되지 않음(어떤 상수도 둘 사이를 이동 가능).
        # 평균을 빼면 A의 평균이 0이 되어 V가 진정한 상태 가치를 나타냄.
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
```

### 4.3 Prioritized Experience Replay (PER)

TD 오류가 큰 경험을 더 자주 샘플링합니다.

```python
class PrioritizedReplayBuffer:
    """왜 우선순위 리플레이(prioritized replay)인가: 균일 샘플링은 네트워크가 이미 잘
    예측하는 전이를 재생하며 시간을 낭비. TD 오류에 비례하여 샘플링하면 놀라운/예측이
    부정확한 전이에 학습을 집중하여 수렴을 가속화."""

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        # 왜 alpha인가: 우선순위 사용 정도를 제어 (alpha=0 → 균일, alpha=1 → TD 오류에
        # 완전 비례). 0.6이 일반적인 균형점.
        self.alpha = alpha
        # 왜 beta인가: 중요도 샘플링(importance sampling) 보정 — 우선순위 샘플링은
        # 높은 오류 전이를 과다 샘플링하여 편향 도입. Beta가 학습 중 0.4에서 1.0으로
        # 점진적으로 증가하여 학습 종료 시 완전히 편향을 보정.
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0

    def push(self, *experience, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        total_priority = self.priorities[:len(self.buffer)].sum()
        probs = self.priorities[:len(self.buffer)] / total_priority

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # 중요도 샘플링 가중치
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        return batch, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha
```

---

## 5. CNN 기반 DQN (Atari)

### 5.1 이미지 입력 네트워크

```python
class AtariDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        # 왜 4-프레임 스택인가: 단일 프레임에는 속도 정보가 없음 — 4개의 연속
        # 프레임을 쌓으면 네트워크가 움직임(방향과 속도)을 추론할 수 있어
        # Pong, Breakout 같은 게임에서 핵심적
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # x shape: (batch, 4, 84, 84)
        # 왜 [0,1]로 정규화하는가: [0, 255] 픽셀 값은 매우 큰 활성화를 생성;
        # 255로 나누면 입력이 경사 하강(gradient descent)에 안정적인 범위로 유지
        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

### 5.2 프레임 전처리

```python
import cv2

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess_frame(self, frame):
        """84x84 그레이스케일로 변환"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, frame):
        processed = self.preprocess_frame(frame)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        return np.array(self.frames)

    def step(self, frame):
        processed = self.preprocess_frame(frame)
        self.frames.append(processed)
        return np.array(self.frames)
```

---

## 6. 실습: CartPole-v1

```python
def main():
    # 환경 설정
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # DQN 에이전트
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=100
    )

    # 학습
    n_episodes = 300
    scores = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        score = 0

        for t in range(500):
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done or truncated)
            agent.learn()

            state = next_state
            score += reward

            if done or truncated:
                break

        scores.append(score)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Score: {np.mean(scores[-10:]):.2f}")

        # 해결 조건
        if np.mean(scores[-100:]) >= 475:
            print(f"Solved in {episode + 1} episodes!")
            break

    env.close()
    return agent, scores

if __name__ == "__main__":
    agent, scores = main()
```

---

## 요약

| 기법 | 목적 | 핵심 아이디어 |
|------|------|--------------|
| Experience Replay | 데이터 효율성, 상관관계 제거 | 버퍼에서 무작위 샘플링 |
| Target Network | 학습 안정화 | 타겟 고정, 주기적 업데이트 |
| Double DQN | 과대추정 방지 | 행동 선택/평가 분리 |
| Dueling DQN | 효율적 학습 | V와 A 분리 |
| PER | 효율적 샘플링 | TD 오류 기반 우선순위 |

---

## 다음 단계

- [08_Policy_Gradient.md](./08_Policy_Gradient.md) - 정책 기반 방법론
