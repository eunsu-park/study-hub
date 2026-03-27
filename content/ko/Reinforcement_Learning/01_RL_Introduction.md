# 01. 강화학습 개요 (Introduction to Reinforcement Learning)

**난이도: ⭐ (입문)**

## 학습 목표
- 강화학습의 정의와 특징 이해
- 에이전트-환경 상호작용 패러다임 파악
- 보상, 에피소드, 연속 태스크의 개념 학습
- 지도학습/비지도학습과의 차이 이해

---

## 1. 강화학습이란?

### 1.1 정의

**강화학습(Reinforcement Learning, RL)**은 에이전트(Agent)가 **환경(Environment)**과 상호작용하면서 **보상(Reward)**을 최대화하는 행동 방식을 학습하는 기계학습의 한 분야입니다.

```
       ┌─────────────────────────────────────────────────────┐
       │                    강화학습 루프                      │
       └─────────────────────────────────────────────────────┘

                        행동 (Action)
                    ┌─────────────────┐
                    │                 ▼
              ┌─────────┐        ┌─────────────┐
              │         │        │             │
              │ 에이전트 │        │    환경     │
              │ (Agent) │        │(Environment)│
              │         │        │             │
              └─────────┘        └─────────────┘
                    ▲                 │
                    │                 │
                    └─────────────────┘
                     상태 (State) +
                     보상 (Reward)
```

### 1.2 핵심 특징

1. **시행착오 학습 (Trial and Error)**: 직접 행동해보고 결과로부터 학습
2. **지연된 보상 (Delayed Reward)**: 즉각적 보상뿐 아니라 미래 보상도 고려
3. **탐험과 활용 (Exploration vs Exploitation)**: 새로운 시도와 기존 지식 활용 사이의 균형
4. **순차적 의사결정 (Sequential Decision Making)**: 연속된 결정들의 누적 효과 고려

### 1.3 머신러닝 패러다임 비교

| 특성 | 지도학습 | 비지도학습 | 강화학습 |
|------|----------|------------|----------|
| 데이터 | 레이블 있음 | 레이블 없음 | 보상 신호 |
| 피드백 | 즉각적 정답 | 없음 | 지연된 보상 |
| 목표 | 예측/분류 | 패턴/구조 발견 | 누적 보상 최대화 |
| 예시 | 이미지 분류 | 클러스터링 | 게임 플레이 |

---

## 2. 에이전트-환경 상호작용

### 2.1 기본 구성 요소

```python
# 강화학습의 기본 요소
class RLComponents:
    """
    1. State (s): 환경의 현재 상황
       - 게임: 화면 픽셀, 점수, 위치 등
       - 로봇: 관절 각도, 속도, 센서 값 등

    2. Action (a): 에이전트가 취할 수 있는 행동
       - 이산적: 상/하/좌/우 이동
       - 연속적: 모터 토크 값

    3. Reward (r): 행동에 대한 수치적 피드백
       - 양수: 좋은 행동 장려
       - 음수: 나쁜 행동 억제

    4. Policy (π): 상태에서 행동을 선택하는 전략
       - 결정적: π(s) = a
       - 확률적: π(a|s) = P(A=a|S=s)

    5. Value Function (V, Q): 장기적 가치 추정
       - V(s): 상태의 가치
       - Q(s,a): 상태-행동 쌍의 가치
    """
    pass
```

### 2.2 상호작용 과정

```python
import gymnasium as gym

# 환경 생성
env = gym.make("CartPole-v1")

# 초기화
state, info = env.reset()

total_reward = 0
done = False

while not done:
    # 랜덤 행동을 사용하는 이유: 학습된 정책이 없는 상태에서 랜덤 샘플링은 에이전트-환경 루프를
    # 시연하기 위한 기준선 역할을 한다. 실제로는 학습된 정책(예: epsilon-greedy)이 이를 대체한다.
    action = env.action_space.sample()

    # env.step이 5개 값을 반환하는 이유: Gymnasium은 "terminated"(자연 종료, 예: 막대 쓰러짐)와
    # "truncated"(시간 제한 도달)를 분리하는데, 이는 리턴(return)을 올바르게 계산하는 데 중요하다 —
    # truncation은 미래 가치를 0으로 만들어서는 안 된다.
    next_state, reward, terminated, truncated, info = env.step(action)

    # 에피소드 전반의 정책 성능을 측정하기 위해 보상을 누적한다
    total_reward += reward

    # 상태를 덮어쓰는 이유: 마르코프 속성(Markov property)에 따라 현재 상태만이 중요하다
    state = next_state

    # terminated와 truncated를 합치는 이유: 둘 다 에피소드를 종료하지만, 부트스트래핑(bootstrapping)에
    # 대해 서로 다른 의미를 가진다 (TD 학습 레슨 참고)
    done = terminated or truncated

print(f"Total reward: {total_reward}")
env.close()
```

### 2.3 Gymnasium 환경 구조

```python
import gymnasium as gym

# 환경 정보 확인
env = gym.make("CartPole-v1")

print("=== 환경 정보 ===")
print(f"관찰 공간: {env.observation_space}")
# Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf], (4,), float32)

print(f"행동 공간: {env.action_space}")
# Discrete(2) - 0: 왼쪽, 1: 오른쪽

print(f"보상 범위: {env.reward_range}")
# (-inf, inf)

# 관찰 공간의 의미 (CartPole)
# [카트 위치, 카트 속도, 막대 각도, 막대 각속도]
```

---

## 3. 보상 (Reward)

### 3.1 보상의 역할

보상은 에이전트에게 **무엇이 좋고 나쁜지** 알려주는 신호입니다.

```python
# 보상 설계 예시
class RewardExamples:
    """
    게임 예시:
    - 점수 획득: +10
    - 적 처치: +100
    - 목표 도달: +1000
    - 피격: -50
    - 게임 오버: -100

    로봇 예시:
    - 목표 방향 이동: +1
    - 장애물 충돌: -10
    - 에너지 소비: -0.1
    - 목표 도달: +100
    """
    pass
```

### 3.2 보상 가설 (Reward Hypothesis)

> "모든 목표는 기대되는 누적 보상의 최대화로 표현될 수 있다."
> - Richard Sutton

$$G_t = R_{t+1} + R_{t+2} + R_{t+3} + \cdots$$

### 3.3 할인된 보상 (Discounted Return)

미래 보상에 **할인율(γ)**을 적용하여 현재 가치로 환산합니다.

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

```python
def calculate_return(rewards, gamma=0.99):
    """
    할인된 누적 보상 계산

    Args:
        rewards: 보상 시퀀스 [r1, r2, r3, ...]
        gamma: 할인율 (0 ~ 1)

    Returns:
        G: 할인된 누적 보상
    """
    # 역방향 반복을 사용하는 이유: G_t = r_t + gamma * G_{t+1} 점화식을 활용하여
    # 단일 역방향 패스로 O(n) 시간에 할인된 리턴을 계산한다
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
    return G

# 예시
rewards = [1, 1, 1, 1, 10]  # 마지막에 큰 보상
gamma = 0.9

G = calculate_return(rewards, gamma)
print(f"할인된 누적 보상 (γ={gamma}): {G:.2f}")
# 1 + 0.9 + 0.81 + 0.729 + 6.561 = 10.0
```

### 3.4 할인율의 의미

| γ 값 | 특성 | 적용 상황 |
|------|------|----------|
| γ = 0 | 근시안적 (즉각 보상만 고려) | 단기 최적화 필요시 |
| γ = 0.9 | 미래 보상 적당히 고려 | 일반적인 경우 |
| γ = 0.99 | 장기적 관점 | 에피소드가 긴 경우 |
| γ = 1 | 미래 보상 동등 평가 | 에피소딕 태스크만 가능 |

---

## 4. 에피소드와 연속 태스크

### 4.1 에피소딕 태스크 (Episodic Tasks)

명확한 **시작**과 **종료**가 있는 태스크입니다.

```python
# 에피소딕 태스크 예시: 게임 한 판
def episodic_task_example():
    env = gym.make("CartPole-v1")

    episodes = 10
    for episode in range(episodes):
        state, _ = env.reset()  # 에피소드 시작
        episode_reward = 0
        step = 0

        while True:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step += 1

            # 에피소드 종료 조건
            if terminated or truncated:
                print(f"Episode {episode + 1}: "
                      f"Steps = {step}, Reward = {episode_reward}")
                break

    env.close()

# 에피소딕 태스크 예시:
# - 게임 (시작 → 게임 오버 또는 클리어)
# - 미로 탈출 (시작점 → 출구)
# - 바둑/체스 (게임 시작 → 승/패/무)
```

### 4.2 연속 태스크 (Continuing Tasks)

종료 없이 **무한히 계속**되는 태스크입니다.

```python
# 연속 태스크 예시: 서버 부하 관리
def continuing_task_example():
    """
    연속 태스크는 자연스러운 종료점이 없음
    - 서버 부하 분산
    - 온도 제어 시스템
    - 주식 트레이딩
    - 로봇 보행 (무한 보행)
    """
    # 시뮬레이션을 위해 인위적으로 스텝 제한
    max_steps = 10000

    state = initialize_system()

    for step in range(max_steps):
        action = select_action(state)
        next_state, reward = environment_step(action)

        # 에이전트 업데이트
        update_agent(state, action, reward, next_state)

        state = next_state

        # 연속 태스크에서는 할인율 γ < 1 필수
        # (γ = 1이면 무한 보상 → 발산)

def initialize_system():
    return None  # placeholder

def select_action(state):
    return None  # placeholder

def environment_step(action):
    return None, 0  # placeholder

def update_agent(*args):
    pass  # placeholder
```

### 4.3 비교

| 특성 | 에피소딕 | 연속 |
|------|----------|------|
| 종료 | 자연스러운 종료점 있음 | 없음 (인위적 truncation 가능) |
| 리턴 | 유한 (γ=1 가능) | 무한대 가능 (γ<1 필수) |
| 예시 | 게임, 미로, 대화 | 서버 관리, 트레이딩 |
| 학습 | 에피소드 단위 업데이트 | 지속적 업데이트 |

---

## 5. 탐험과 활용 (Exploration vs Exploitation)

### 5.1 딜레마

- **탐험 (Exploration)**: 새로운 행동을 시도하여 더 좋은 전략 발견
- **활용 (Exploitation)**: 현재까지 알려진 최선의 행동 수행

```python
import numpy as np

class EpsilonGreedy:
    """
    ε-탐욕 전략: 가장 기본적인 탐험-활용 균형 방법
    """
    def __init__(self, n_actions, epsilon=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.q_values = np.zeros(n_actions)  # 각 행동의 추정 가치
        self.action_counts = np.zeros(n_actions)

    def select_action(self):
        """
        확률 ε로 랜덤 행동 (탐험)
        확률 1-ε로 최선 행동 (활용)
        """
        # epsilon-greedy를 사용하는 이유: 탐험(Exploration)과 활용(Exploitation)을 균형 있게 조합하는
        # 가장 단순한 전략이다. epsilon 확률로 랜덤 행동을 시도하여 더 나은 선택지를 발견하고,
        # 그 외에는 현재 알려진 최선의 행동을 사용한다.
        if np.random.random() < self.epsilon:
            # 탐험: 랜덤 행동
            return np.random.randint(self.n_actions)
        else:
            # 활용: 최고 가치 행동
            return np.argmax(self.q_values)

    def update(self, action, reward):
        """행동 가치 업데이트 (점진적 평균)"""
        self.action_counts[action] += 1
        n = self.action_counts[action]
        # 점진적 평균 업데이트를 사용하는 이유: 전체 평균을 계산하는 것과 동일하지만
        # 업데이트당 O(1) 메모리와 계산만 필요하다. 스텝 크기 1/n은 모든 과거 보상에
        # 동등한 가중치를 부여하여 실제 평균으로 수렴한다.
        self.q_values[action] += (reward - self.q_values[action]) / n
```

### 5.2 탐험 전략들

```python
class ExplorationStrategies:
    """다양한 탐험 전략"""

    @staticmethod
    def epsilon_greedy(q_values, epsilon):
        """ε-탐욕"""
        if np.random.random() < epsilon:
            return np.random.randint(len(q_values))
        return np.argmax(q_values)

    @staticmethod
    def softmax(q_values, temperature=1.0):
        """
        Softmax (Boltzmann) 탐험
        - temperature 높음: 더 많은 탐험
        - temperature 낮음: 더 많은 활용
        """
        exp_q = np.exp(q_values / temperature)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probs)

    @staticmethod
    def ucb(q_values, action_counts, t, c=2.0):
        """
        Upper Confidence Bound (UCB)
        - 불확실성이 높은 행동 우선 탐험
        """
        ucb_values = q_values + c * np.sqrt(np.log(t + 1) / (action_counts + 1e-5))
        return np.argmax(ucb_values)
```

### 5.3 탐험율 감소 (Epsilon Decay)

```python
class EpsilonDecay:
    """시간에 따라 탐험율 감소"""

    def __init__(self, start=1.0, end=0.01, decay=0.995):
        self.epsilon = start
        self.end = end
        self.decay = decay

    def get_epsilon(self):
        return self.epsilon

    def update(self):
        """에피소드 끝날 때마다 호출"""
        self.epsilon = max(self.end, self.epsilon * self.decay)

# 사용 예시
epsilon_scheduler = EpsilonDecay(start=1.0, end=0.01, decay=0.995)

for episode in range(1000):
    epsilon = epsilon_scheduler.get_epsilon()
    # ... 에피소드 실행 ...
    epsilon_scheduler.update()

    if episode % 100 == 0:
        print(f"Episode {episode}: ε = {epsilon:.4f}")
```

---

## 6. 실제 응용 사례

### 6.1 게임 AI

```python
"""
AlphaGo (DeepMind, 2016)
- 바둑에서 인간 세계 챔피언을 이김
- Monte Carlo Tree Search + Deep Learning + RL
- Self-play로 학습

OpenAI Five (2019)
- Dota 2에서 프로 팀을 이김
- 분산 PPO 알고리즘
- 약 45,000년치의 게임 플레이로 학습

AlphaStar (DeepMind, 2019)
- StarCraft II 그랜드마스터 달성
- Multi-agent RL + League Training
"""
```

### 6.2 로봇공학

```python
"""
로봇 보행 (Locomotion)
- 물리 시뮬레이션에서 학습 후 실제 로봇에 전이
- Sim-to-Real Transfer

로봇 팔 제어 (Manipulation)
- 물체 집기, 조립 등
- 연속 행동 공간 (관절 토크)

자율주행 (Autonomous Driving)
- 차선 유지, 장애물 회피
- 안전성이 중요하여 시뮬레이션 기반 학습
"""
```

### 6.3 LLM과 RLHF

```python
"""
RLHF (Reinforcement Learning from Human Feedback)
- ChatGPT, Claude 등의 핵심 학습 방법
- 인간 피드백으로 보상 모델 학습
- PPO로 언어 모델 fine-tuning

프로세스:
1. 사전 학습된 LLM
2. 인간이 응답 품질 순위 매김
3. 순위 데이터로 보상 모델 학습
4. 보상 모델을 사용해 LLM을 RL로 fine-tuning
"""
```

---

## 7. 강화학습의 도전 과제

### 7.1 주요 문제점

| 문제 | 설명 | 해결 방향 |
|------|------|----------|
| 샘플 효율성 | 학습에 많은 경험 필요 | 모델 기반 RL, 전이 학습 |
| 신용 할당 | 어떤 행동이 보상에 기여했는지 | TD 학습, GAE |
| 안정성 | 학습이 불안정할 수 있음 | PPO, TRPO 등 안정화 기법 |
| 보상 설계 | 올바른 보상 정의가 어려움 | 역강화학습, 보상 성형 |
| 안전성 | 위험한 행동 방지 | Constrained RL, Safe RL |

### 7.2 보상 해킹 (Reward Hacking)

```python
"""
에이전트가 의도치 않은 방법으로 보상을 최대화하는 현상

예시:
- 청소 로봇: 쓰레기를 숨기기 (청소 완료로 오인)
- 보트 경주: 원을 그리며 파워업 아이템만 수집
- 테트리스: 게임을 일시정지하여 게임오버 회피

교훈: 보상 설계는 신중하게!
"""
```

---

## 8. 요약

### 핵심 개념 정리

1. **강화학습**: 에이전트가 환경과 상호작용하며 보상 최대화 학습
2. **MDP 구성요소**: 상태, 행동, 보상, 전이 확률, 할인율
3. **보상**: 행동의 좋고 나쁨을 알려주는 신호
4. **할인율 (γ)**: 미래 보상의 현재 가치 결정
5. **탐험-활용**: 새로운 시도와 알려진 최선 사이의 균형

### 수식 정리

| 개념 | 수식 |
|------|------|
| 누적 보상 | $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ |
| 상태 가치 | $V^\pi(s) = \mathbb{E}_\pi[G_t \| S_t = s]$ |
| 행동 가치 | $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \| S_t = s, A_t = a]$ |

---

## 9. 연습 문제

1. **개념 확인**: 지도학습과 강화학습의 주요 차이점 3가지를 설명하세요.

2. **할인율 계산**: 보상 시퀀스 [1, 2, 3, 4, 5]에 대해 γ=0.9일 때 할인된 누적 보상을 계산하세요.

3. **탐험-활용**: ε-greedy 전략에서 ε=0.2일 때, 100번의 행동 중 예상되는 탐험 횟수는?

4. **보상 설계**: 미로 탈출 문제에서 적절한 보상 함수를 설계해보세요.

---

## 다음 단계

다음 레슨 **02_MDP_Basics.md**에서는 강화학습의 수학적 기반인 **마르코프 결정 과정(MDP)**과 **벨만 방정식**을 학습합니다.

---

## 참고 자료

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 1
- David Silver's RL Course, Lecture 1: Introduction to RL
- [Gymnasium Documentation](https://gymnasium.farama.org/)
