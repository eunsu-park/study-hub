# 06. Q-Learning과 SARSA

**난이도: ⭐⭐⭐ (중급)**

## 학습 목표
- Q-Learning의 원리와 off-policy 특성 이해
- SARSA의 원리와 on-policy 특성 이해
- Q-Learning vs SARSA 차이점 비교
- Epsilon-greedy 탐색 전략 구현
- 수렴 조건과 실전 적용 팁

---

## 1. 행동 가치 함수 (Q 함수)

### 1.1 Q 함수의 정의

상태-행동 쌍의 가치를 평가하는 함수입니다.

$$Q(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a]$$

```python
import numpy as np

class QTable:
    def __init__(self, n_states, n_actions):
        # Q 테이블 초기화 (0 또는 작은 랜덤값)
        self.q_table = np.zeros((n_states, n_actions))

    def get_q(self, state, action):
        return self.q_table[state, action]

    def get_best_action(self, state):
        return np.argmax(self.q_table[state])

    def update(self, state, action, value):
        self.q_table[state, action] = value
```

### 1.2 V와 Q의 관계

```
V(s) = max_a Q(s, a)           # 최적 정책에서
V(s) = Σ_a π(a|s) Q(s, a)      # 일반 정책에서
```

---

## 2. Q-Learning (Off-Policy TD)

### 2.1 Q-Learning 알고리즘

**Off-Policy**: 행동 정책(behavior policy)과 타겟 정책(target policy)이 다름

```
Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
```

- 행동 정책: ε-greedy (탐색용)
- 타겟 정책: greedy (학습용)

```python
class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha      # 학습률
        self.gamma = gamma      # 할인율
        self.epsilon = epsilon  # 탐색률
        self.n_actions = n_actions

    def choose_action(self, state):
        """ε-greedy 정책으로 행동 선택"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # 탐색
        return np.argmax(self.q_table[state])          # 활용

    def update(self, state, action, reward, next_state, done):
        """Q-Learning 업데이트"""
        if done:
            target = reward
        else:
            # Off-policy(오프-정책)의 핵심: 실제로 선택한 행동이 아닌 다음 상태에서의
            # 최대(MAX) Q값을 사용 — 이 덕분에 노이즈가 있는 ε-greedy 행동 정책으로
            # 탐색하면서도 최적 정책을 학습할 수 있음
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # 벨만 잔차(Bellman residual): 양수이면 기대보다 좋은 결과이므로 Q를 높이고,
        # alpha는 과거 추정을 얼마나 빠르게 수정할지를 조절
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

        return td_error
```

### 2.2 Q-Learning 학습 루프

```python
def train_qlearning(env, agent, n_episodes=1000):
    rewards_history = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # ε-greedy로 행동 선택
            action = agent.choose_action(state)

            # 환경에서 한 스텝 진행
            next_state, reward, done, _ = env.step(action)

            # Q-Learning 업데이트는 다음 행동이 무엇인지 알 필요가 없음 —
            # 이 독립성 덕분에 오프-정책(off-policy)이 되어 경험 재현(experience replay)
            # 버퍼의 오래된 전이(transition)도 재활용할 수 있음
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

        # epsilon을 줄여 탐색에서 활용(exploitation)으로 점진적으로 전환;
        # 0.01 하한선은 드물지만 중요한 상태-행동 쌍을 영구히 약간 탐색하게 유지
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

    return rewards_history
```

---

## 3. SARSA (On-Policy TD)

### 3.1 SARSA 알고리즘

**On-Policy**: 행동 정책과 타겟 정책이 동일

이름의 유래: **S**tate, **A**ction, **R**eward, **S**tate, **A**ction

```
Q(s, a) ← Q(s, a) + α[r + γ Q(s', a') - Q(s, a)]
```

여기서 a'는 실제로 선택될 다음 행동입니다.

```python
class SARSA:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        """ε-greedy 정책"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA 업데이트"""
        if done:
            target = reward
        else:
            # On-policy: 실제 다음 행동의 Q값 사용
            target = reward + self.gamma * self.q_table[next_state, next_action]

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

        return td_error
```

### 3.2 SARSA 학습 루프

```python
def train_sarsa(env, agent, n_episodes=1000):
    rewards_history = []

    for episode in range(n_episodes):
        state = env.reset()
        # 루프 진입 전에 첫 번째 행동을 선택해 (s, a) 쌍을 미리 준비 —
        # SARSA가 요구하는 (S, A, R, S, A) 튜플 구조를 매 반복마다 갖출 수 있게 함
        action = agent.choose_action(state)  # 초기 행동 선택
        total_reward = 0
        done = False

        while not done:
            # 환경에서 한 스텝 진행
            next_state, reward, done, _ = env.step(action)

            # 업데이트 이전에 next_action을 선택해야 Q 타겟이 실제로 따르는
            # 정책을 반영할 수 있음 — 가상의 행동이 아닌 실제 행동이 타겟에 포함
            next_action = agent.choose_action(next_state)

            # SARSA 업데이트 (다음 행동 필요)
            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            # next_action을 재활용 — 두 번 샘플링하면 업데이트에 쓴 행동과
            # 실제로 실행하는 행동이 달라져 온-정책(on-policy) 보장이 깨짐
            action = next_action
            total_reward += reward

        rewards_history.append(total_reward)

    return rewards_history
```

---

## 4. Q-Learning vs SARSA 비교

### 4.1 핵심 차이점

| 특성 | Q-Learning | SARSA |
|------|------------|-------|
| 정책 유형 | Off-policy | On-policy |
| 타겟 계산 | max Q(s', a') | Q(s', a') |
| 학습 대상 | 최적 정책 | 현재 정책 |
| 탐색 영향 | 학습에 영향 없음 | 학습에 직접 영향 |
| 안전성 | 더 공격적 | 더 보수적 |

### 4.2 Cliff Walking 예제

```
[S][.][.][.][.][.][.][.][.][.][.][G]
[C][C][C][C][C][C][C][C][C][C][C][C]

S: 시작, G: 목표, C: 절벽 (큰 음수 보상)
```

```python
def cliff_walking_comparison():
    """
    Q-Learning: 절벽 가장자리의 최단 경로 선호 (위험하지만 빠름)
    SARSA: 절벽에서 떨어진 안전한 경로 선호 (느리지만 안전)
    """
    # Q-Learning은 최적 경로를 학습하지만
    # ε-greedy 탐색 중 절벽으로 떨어질 수 있음

    # SARSA는 탐색을 고려하여
    # 절벽에서 떨어진 경로를 학습
    pass
```

### 4.3 시각화 비교

```python
import matplotlib.pyplot as plt

def compare_algorithms(env, n_episodes=500, n_runs=10):
    q_rewards = np.zeros((n_runs, n_episodes))
    sarsa_rewards = np.zeros((n_runs, n_episodes))

    for run in range(n_runs):
        q_agent = QLearning(env.n_states, env.n_actions)
        sarsa_agent = SARSA(env.n_states, env.n_actions)

        q_rewards[run] = train_qlearning(env, q_agent, n_episodes)
        sarsa_rewards[run] = train_sarsa(env, sarsa_agent, n_episodes)

    # 평균 및 표준편차
    plt.figure(figsize=(10, 6))

    q_mean = q_rewards.mean(axis=0)
    sarsa_mean = sarsa_rewards.mean(axis=0)

    plt.plot(q_mean, label='Q-Learning', alpha=0.8)
    plt.plot(sarsa_mean, label='SARSA', alpha=0.8)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning vs SARSA')
    plt.legend()
    plt.show()
```

---

## 5. 탐색 전략 (Exploration Strategies)

### 5.1 Epsilon-Greedy

```python
def epsilon_greedy(q_values, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))
    return np.argmax(q_values)

# Epsilon decay 스케줄
def get_epsilon(episode, min_eps=0.01, max_eps=1.0, decay=0.995):
    return max(min_eps, max_eps * (decay ** episode))
```

### 5.2 Softmax (Boltzmann) 탐색

```python
def softmax_action(q_values, temperature=1.0):
    """온도에 따른 확률적 행동 선택"""
    exp_q = np.exp(q_values / temperature)
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(len(q_values), p=probs)
```

### 5.3 UCB (Upper Confidence Bound)

```python
class UCBAgent:
    def __init__(self, n_states, n_actions, c=2.0):
        self.q_table = np.zeros((n_states, n_actions))
        self.n_visits = np.zeros((n_states, n_actions))
        self.total_visits = np.zeros(n_states)
        self.c = c

    def choose_action(self, state):
        self.total_visits[state] += 1

        # 방문하지 않은 행동이 있으면 선택
        if 0 in self.n_visits[state]:
            return np.argmin(self.n_visits[state])

        # UCB 값 계산
        ucb_values = self.q_table[state] + self.c * np.sqrt(
            np.log(self.total_visits[state]) / self.n_visits[state]
        )
        return np.argmax(ucb_values)
```

---

## 6. Expected SARSA

### 6.1 개념

SARSA와 Q-Learning의 중간 형태로, 다음 상태에서 기대값을 사용합니다.

```
Q(s, a) ← Q(s, a) + α[r + γ Σ_a' π(a'|s') Q(s', a') - Q(s, a)]
```

```python
class ExpectedSARSA:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def get_policy_probs(self, state):
        """ε-greedy 정책의 확률 분포"""
        probs = np.ones(self.n_actions) * self.epsilon / self.n_actions
        best_action = np.argmax(self.q_table[state])
        probs[best_action] += 1 - self.epsilon
        return probs

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            # 기대값 계산
            probs = self.get_policy_probs(next_state)
            expected_q = np.sum(probs * self.q_table[next_state])
            target = reward + self.gamma * expected_q

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
```

---

## 7. 수렴성과 하이퍼파라미터

### 7.1 수렴 조건

Q-Learning이 최적 Q*로 수렴하기 위한 조건:

1. **모든 상태-행동 쌍을 무한히 방문**
2. **학습률 조건**: Σ α = ∞, Σ α² < ∞ (예: α = 1/n)
3. **보상이 유계**

### 7.2 하이퍼파라미터 튜닝

```python
# 일반적인 시작점
config = {
    'alpha': 0.1,        # 학습률: 0.01 ~ 0.5
    'gamma': 0.99,       # 할인율: 0.9 ~ 0.999
    'epsilon': 1.0,      # 초기 탐색률
    'epsilon_min': 0.01, # 최소 탐색률
    'epsilon_decay': 0.995  # 감소율
}

# 학습률 스케줄링
def learning_rate_schedule(episode, initial_lr=0.5, decay=0.001):
    return initial_lr / (1 + decay * episode)
```

---

## 8. 실습: FrozenLake

```python
import gymnasium as gym

def train_frozen_lake():
    env = gym.make('FrozenLake-v1', is_slippery=True)

    agent = QLearning(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        # 순수 탐색(epsilon=1.0)에서 시작 — 정책을 확정하기 전에 모든 셀을 방문하기 위함;
        # FrozenLake는 희소 보상(sparse reward)이라 광범위한 탐색 없이는 목표를 찾기 어려움
        epsilon=1.0
    )

    n_episodes = 10000
    rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        # 느린 감쇠(0.9995 vs 0.995)로 탐색 시간을 더 확보 —
        # FrozenLake의 확률적 전이(stochastic transition)는 신뢰할 만한 Q값 추정을
        # 위해 각 상태를 여러 번 방문해야 함
        agent.epsilon = max(0.01, agent.epsilon * 0.9995)

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.3f}")

    return agent, rewards

if __name__ == "__main__":
    agent, rewards = train_frozen_lake()
```

---

## 요약

| 알고리즘 | 타겟 | 정책 | 특징 |
|---------|------|------|------|
| Q-Learning | max Q(s',a') | Off-policy | 최적 정책 학습 |
| SARSA | Q(s',a') | On-policy | 현재 정책 평가 |
| Expected SARSA | E[Q(s',a')] | Off-policy | 낮은 분산 |

**핵심 포인트:**
- Q-Learning은 최적 정책을 직접 학습
- SARSA는 탐색을 고려한 안전한 학습
- 적절한 탐색-활용 균형이 중요

---

## 다음 단계

- [07_Deep_Q_Network.md](./07_Deep_Q_Network.md) - 신경망과 Q-Learning의 결합
