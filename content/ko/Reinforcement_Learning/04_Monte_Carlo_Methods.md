# 04. 몬테카를로 방법 (Monte Carlo Methods)

**난이도: ⭐⭐ (기초)**

## 학습 목표
- 몬테카를로(MC) 방법의 기본 개념 이해
- 모델 프리(Model-Free) 학습의 의미 파악
- First-visit MC와 Every-visit MC의 차이 이해
- MC 정책 평가 및 제어 알고리즘 구현
- 탐험의 중요성과 해결 방법 학습

---

## 1. 몬테카를로 방법이란?

### 1.1 개요

**몬테카를로(Monte Carlo, MC) 방법**은 **실제 경험(에피소드)**으로부터 가치 함수를 추정하는 방법입니다. 환경 모델 없이 학습하는 **모델 프리(Model-Free)** 방법입니다.

### 1.2 DP vs MC 비교

| 특성 | 동적 프로그래밍 (DP) | 몬테카를로 (MC) |
|------|---------------------|-----------------|
| 환경 모델 | 필요 (P, R 알아야 함) | 불필요 |
| 학습 방식 | 계산 (부트스트래핑) | 샘플링 (경험) |
| 업데이트 시점 | 매 스텝 가능 | 에피소드 종료 후 |
| 적용 환경 | 에피소딕/연속 | 에피소딕만 |

### 1.3 핵심 아이디어

$$V(s) \approx \text{평균}(\text{상태 } s \text{에서 시작한 에피소드들의 리턴})$$

```
에피소드 1: S₀ → S₁ → S₂ → ... → 종료, G₁ = 10
에피소드 2: S₀ → S₃ → S₁ → ... → 종료, G₂ = 8
에피소드 3: S₀ → S₂ → S₁ → ... → 종료, G₃ = 12

V(S₀) = (10 + 8 + 12) / 3 = 10
```

---

## 2. 리턴 (Return) 계산

### 2.1 에피소드에서 리턴 계산

```python
def calculate_returns(episode, gamma=0.99):
    """
    에피소드에서 각 상태의 리턴 계산

    Args:
        episode: [(state, action, reward), ...] 형태의 리스트
        gamma: 할인율

    Returns:
        returns: {state: return} 딕셔너리
    """
    G = 0  # 리턴 초기화
    returns = {}

    # 왜 역순 반복인가: G_t = r_t + gamma * G_{t+1} 점화식(recurrence)을 활용하여
    # 한 번의 O(n) 패스로 모든 리턴 계산 — 정방향이면 각 t마다 재합산하여 O(n^2)
    for t in range(len(episode) - 1, -1, -1):
        state, action, reward = episode[t]
        G = reward + gamma * G  # 할인된 리턴 — 이미 계산된 미래 리턴을 재활용
        returns[t] = (state, G)

    return returns


# 예시
episode = [
    ('s0', 'right', -1),
    ('s1', 'right', -1),
    ('s2', 'right', 10),  # 목표 도달
]

returns = calculate_returns(episode, gamma=0.9)
for t, (state, G) in returns.items():
    print(f"t={t}: {state}, G={G:.2f}")

# 출력:
# t=2: s2, G=10.00
# t=1: s1, G=8.00  (= -1 + 0.9 * 10)
# t=0: s0, G=6.20  (= -1 + 0.9 * 8)
```

---

## 3. MC 정책 평가

### 3.1 First-visit MC vs Every-visit MC

| 방법 | 설명 |
|------|------|
| First-visit MC | 에피소드에서 상태 s를 **처음 방문했을 때만** 리턴 기록 |
| Every-visit MC | 에피소드에서 상태 s를 **방문할 때마다** 리턴 기록 |

```
에피소드: S₀ → S₁ → S₂ → S₁ → S₃ (종료)
                     ↑      ↑
              첫 방문   두 번째 방문

First-visit: S₁의 첫 방문만 카운트
Every-visit: S₁의 모든 방문 카운트
```

### 3.2 First-visit MC 구현

```python
import numpy as np
from collections import defaultdict

def first_visit_mc_prediction(env, policy, n_episodes=10000, gamma=0.99):
    """
    First-visit MC 정책 평가

    Args:
        env: Gymnasium 환경
        policy: 정책 함수 policy(state) -> action
        n_episodes: 에피소드 수
        gamma: 할인율

    Returns:
        V: 상태 가치 함수
    """
    # 왜 점진적 평균(incremental mean)인가: 합계와 횟수만 저장하면 상태당 O(1) 메모리로
    # 모든 리턴을 보관하지 않아도 참 기대값으로 수렴
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)

    for episode_num in range(n_episodes):
        # 에피소드 생성
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        # 왜 First-visit인가: 상태의 첫 방문만 카운트하면 V(s)의 비편향(unbiased) 추정량을
        # 얻음. Every-visit MC도 일치(consistent)하지만 편향/분산 트레이드오프가 다름
        # — First-visit이 이론적 분석이 더 간단.
        visited = set()
        G = 0

        # 왜 역순 반복인가: G_t = r_t + gamma * G_{t+1} — 한 번의 역방향 패스로
        # O(n)에 모든 리턴 계산 (각 시점에서 재합산하면 O(n^2))
        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]

            G = gamma * G + reward_t

            # 왜 First-visit 체크인가: 같은 에피소드 내 중복 방문을 제외하여
            # 비편향(unbiased) 성질을 유지
            if state_t not in visited:
                visited.add(state_t)
                returns_sum[state_t] += G
                returns_count[state_t] += 1
                V[state_t] = returns_sum[state_t] / returns_count[state_t]

        if (episode_num + 1) % 1000 == 0:
            print(f"Episode {episode_num + 1}/{n_episodes}")

    return dict(V)


# 사용 예시
import gymnasium as gym

env = gym.make('Blackjack-v1')

def random_policy(state):
    """랜덤 정책"""
    return env.action_space.sample()

V = first_visit_mc_prediction(env, random_policy, n_episodes=50000)
print(f"\n추정된 상태 수: {len(V)}")
```

### 3.3 Every-visit MC 구현

```python
def every_visit_mc_prediction(env, policy, n_episodes=10000, gamma=0.99):
    """
    Every-visit MC 정책 평가
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)

    for episode_num in range(n_episodes):
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        G = 0

        # 왜 Every-visit가 다른가: 모든 방문에서 업데이트하면 비편향성(unbiasedness)을 포기하는
        # 대신 분산(variance)이 낮아짐 — 에피소드당 더 많은 데이터 포인트를 얻지만 같은
        # 에피소드의 샘플은 상관(correlated)됨. 점근적으로는 동일 값에 수렴.
        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]

            G = gamma * G + reward_t

            # First-visit 필터 없음: 모든 발생이 리턴 샘플에 기여
            returns_sum[state_t] += G
            returns_count[state_t] += 1
            V[state_t] = returns_sum[state_t] / returns_count[state_t]

    return dict(V)
```

---

## 4. MC 정책 제어

### 4.1 Exploring Starts (ES)

모든 상태-행동 쌍에서 에피소드가 시작할 수 있다고 가정합니다.

```python
def mc_exploring_starts(env, n_episodes=100000, gamma=0.99):
    """
    MC with Exploring Starts

    모든 (s, a) 쌍이 시작점이 될 수 있음을 가정
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Q 함수와 방문 횟수 초기화
    Q = defaultdict(lambda: np.zeros(n_actions))
    returns_sum = defaultdict(lambda: np.zeros(n_actions))
    returns_count = defaultdict(lambda: np.zeros(n_actions))

    # 정책 (탐욕적)
    policy = defaultdict(lambda: 0)

    for episode_num in range(n_episodes):
        # 왜 Exploring Starts인가: 탐욕적(greedy) 정책은 모든 (s, a) 쌍의 Q값이
        # 정확해야만 개선 가능 → 모든 쌍을 방문해야 함. 랜덤 초기화로
        # 모든 쌍의 방문 확률이 0이 아님을 보장.
        start_state = env.observation_space.sample()
        start_action = env.action_space.sample()

        # 에피소드 생성
        episode = []
        state = start_state
        action = start_action

        # 첫 스텝
        env.reset()
        env.unwrapped.s = state  # 상태 강제 설정 (환경에 따라 다름)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated

        # 나머지 에피소드
        while not done:
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        # 리턴 계산 및 Q 업데이트
        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            if (state_t, action_t) not in visited:
                visited.add((state_t, action_t))
                returns_sum[state_t][action_t] += G
                returns_count[state_t][action_t] += 1
                Q[state_t][action_t] = (returns_sum[state_t][action_t] /
                                        returns_count[state_t][action_t])

                # 왜 탐욕적(greedy) 개선인가: 정책 개선 정리(policy improvement theorem)에 의해
                # Q에 대해 탐욕적으로 행동하면 기존 정책 이상의 성능이 보장됨
                policy[state_t] = np.argmax(Q[state_t])

    return Q, policy
```

### 4.2 ε-greedy 정책

Exploring Starts는 현실적이지 않으므로, **ε-greedy** 정책을 사용합니다.

$$\pi(a|s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s, a') \\
\frac{\epsilon}{|A|} & \text{otherwise}
\end{cases}$$

```python
def epsilon_greedy_policy(Q, state, n_actions, epsilon=0.1):
    """
    ε-탐욕 행동 선택

    Args:
        Q: 행동 가치 함수
        state: 현재 상태
        n_actions: 행동 수
        epsilon: 탐험 확률

    Returns:
        action: 선택된 행동
    """
    # 왜 epsilon-greedy인가: 탐험(exploration)과 활용(exploitation)의 균형 —
    # epsilon 확률로 랜덤 행동을 시도하여 더 나은 전략을 발견하고,
    # 나머지 시간에는 최선의 알려진 행동을 활용. 순수 탐욕(epsilon=0)은
    # 더 나은 대안을 발견하지 못해 차선 정책에 갇힐 위험.
    if np.random.random() < epsilon:
        # 탐험: 미방문 (s, a) 쌍에 대한 정보를 수집하기 위한 랜덤 행동
        return np.random.randint(n_actions)
    else:
        # 활용: 추정된 Q값이 가장 높은 행동 선택
        return np.argmax(Q[state])
```

### 4.3 On-policy MC 제어

```python
def mc_on_policy_control(env, n_episodes=100000, gamma=0.99,
                         epsilon=0.1, epsilon_decay=0.9999):
    """
    On-policy MC 제어 (ε-greedy)

    Args:
        env: Gymnasium 환경
        n_episodes: 에피소드 수
        gamma: 할인율
        epsilon: 탐험율
        epsilon_decay: epsilon 감소율

    Returns:
        Q: 행동 가치 함수
        policy: 학습된 정책
    """
    n_actions = env.action_space.n

    Q = defaultdict(lambda: np.zeros(n_actions))
    returns_sum = defaultdict(lambda: np.zeros(n_actions))
    returns_count = defaultdict(lambda: np.zeros(n_actions))

    episode_rewards = []

    for episode_num in range(n_episodes):
        episode = []
        state, _ = env.reset()
        done = False
        total_reward = 0

        # ε-greedy 정책으로 에피소드 생성
        while not done:
            action = epsilon_greedy_policy(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode.append((state, action, reward))
            total_reward += reward

            state = next_state
            done = terminated or truncated

        episode_rewards.append(total_reward)

        # Q 업데이트 (First-visit)
        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            if (state_t, action_t) not in visited:
                visited.add((state_t, action_t))
                returns_sum[state_t][action_t] += G
                returns_count[state_t][action_t] += 1
                Q[state_t][action_t] = (returns_sum[state_t][action_t] /
                                        returns_count[state_t][action_t])

        # 왜 epsilon 감소인가: 초기 학습에는 높은 탐험(high epsilon)으로 정확한 Q 추정을
        # 구축하고, 이후 점차 활용(exploitation)으로 전환. 최솟값 0.01은 차선 정책에
        # 완전히 갇히는 것을 방지 — 이 최솟값 없이는 지역 최적에서 벗어날 수 없음.
        epsilon = max(0.01, epsilon * epsilon_decay)

        if (episode_num + 1) % 10000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            print(f"Episode {episode_num + 1}: avg_reward = {avg_reward:.3f}, "
                  f"epsilon = {epsilon:.4f}")

    # 왜 추론 시 탐욕적(greedy)인가: 배포(deployment) 시에는 탐험이 필요 없음 —
    # 기대 보상(expected return)을 최대화하기 위해 최선의 학습된 행동만 선택
    policy = {}
    for state in Q:
        policy[state] = np.argmax(Q[state])

    return dict(Q), policy, episode_rewards
```

---

## 5. Off-policy MC (Importance Sampling)

### 5.1 Off-policy 학습이란?

- **On-policy**: 행동 정책 = 목표 정책 (같은 정책으로 탐험하고 학습)
- **Off-policy**: 행동 정책 ≠ 목표 정책 (다른 정책으로 탐험하고 학습)

```
행동 정책 (Behavior Policy) b: 데이터 수집용
목표 정책 (Target Policy) π: 학습하고자 하는 최적 정책
```

### 5.2 중요도 샘플링 (Importance Sampling)

$$\mathbb{E}_b[X] = \sum_x x \cdot b(x) = \sum_x x \cdot \frac{\pi(x)}{b(x)} \cdot b(x) = \mathbb{E}_b\left[\frac{\pi(X)}{b(X)} X\right]$$

중요도 샘플링 비율:
$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k | S_k)}{b(A_k | S_k)}$$

```python
def importance_sampling_ratio(episode, target_policy, behavior_policy, t):
    """
    중요도 샘플링 비율 계산

    ρ = π(a₀|s₀)/b(a₀|s₀) × π(a₁|s₁)/b(a₁|s₁) × ...
    """
    # 왜 중요도 샘플링 비율(importance sampling ratio)인가: 데이터는 행동 정책(behavior policy) b
    # 하에서 수집되었지만 목표 정책(target policy) pi를 평가하고자 함. pi/b 비율이 각 전이를
    # 재가중하여 두 정책 간 분포 불일치(distribution mismatch)를 보정.
    rho = 1.0

    for k in range(t, len(episode)):
        state, action, _ = episode[k]
        pi_prob = target_policy(state, action)  # π(a|s)
        b_prob = behavior_policy(state, action)  # b(a|s)

        if b_prob == 0:
            return 0  # 안전 검사: b가 이 행동을 취하지 않을 때 0으로 나누기 방지

        rho *= pi_prob / b_prob  # 나머지 모든 스텝에 대한 비율의 곱

    return rho
```

### 5.3 Off-policy MC 구현

```python
def mc_off_policy_prediction(env, target_policy, behavior_policy,
                              n_episodes=100000, gamma=0.99):
    """
    Off-policy MC 정책 평가 (Weighted Importance Sampling)

    Args:
        target_policy: 목표 정책 (결정적) - π(s) -> a
        behavior_policy: 행동 정책 - b(s) -> a (ε-greedy 등)
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))  # 가중치 합

    for episode_num in range(n_episodes):
        # 행동 정책으로 에피소드 생성
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        G = 0
        W = 1.0  # 중요도 샘플링 가중치

        # 왜 역순 처리인가: on-policy MC와 동일한 O(n) 점화식 활용 + 행동 정책과
        # 목표 정책이 분기할 때 조기 종료(early-terminate) 가능
        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            # 왜 가중 중요도 샘플링(weighted IS)인가: 일반 IS는 비율 곱이 폭발하여
            # 분산이 매우 높음. 가중 IS(자기 정규화)는 누적 가중치 C로 나누어 분산을
            # 크게 줄임 — 약간의 편향이 생기지만 점근적으로 소멸.
            C[state_t][action_t] += W
            Q[state_t][action_t] += W / C[state_t][action_t] * (G - Q[state_t][action_t])

            # 목표 정책에서의 행동
            target_action = target_policy(state_t)

            # 왜 조기 종료인가: 결정적(deterministic) 목표 정책에서 행동 정책이 다른 행동을
            # 취했으면 pi(a|s) = 0이므로 IS 비율이 0이 됨 — 이전 시점들은 기여 없으므로
            # 계속 진행할 필요 없음
            if action_t != target_action:
                break

            # 중요도 비율 업데이트
            # π(a|s) = 1 (결정적), b(a|s) = 행동 정책 확률
            b_prob = behavior_policy_prob(state_t, action_t)  # 구현 필요
            W = W * 1.0 / b_prob

    return dict(Q)
```

---

## 6. 블랙잭 예제

### 6.1 블랙잭 환경

```python
import gymnasium as gym
import numpy as np
from collections import defaultdict

# 블랙잭 환경
# 상태: (플레이어 합, 딜러 오픈카드, 사용 가능한 에이스)
# 행동: 0 = stick (패 유지), 1 = hit (카드 추가)

env = gym.make('Blackjack-v1', sab=True)  # sab: Sutton and Barto 버전

print("상태 공간:", env.observation_space)
print("행동 공간:", env.action_space)
```

### 6.2 블랙잭에서 MC 학습

```python
def learn_blackjack(n_episodes=500000, gamma=1.0, epsilon=0.1):
    """블랙잭에서 MC 제어"""

    env = gym.make('Blackjack-v1', sab=True)

    Q = defaultdict(lambda: np.zeros(2))
    returns_sum = defaultdict(lambda: np.zeros(2))
    returns_count = defaultdict(lambda: np.zeros(2))

    def get_action(state, Q, epsilon):
        # 왜 여기서 epsilon-greedy인가: 블랙잭은 이산 상태 공간이 작지만
        # 올바른 hit/stick 경계를 발견하기 위해 여전히 탐험이 필요
        if np.random.random() < epsilon:
            return env.action_space.sample()
        return np.argmax(Q[state])

    wins = 0
    for ep in range(n_episodes):
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = get_action(state, Q, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        if episode[-1][2] == 1:  # 승리
            wins += 1

        # Q 업데이트
        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            if (state_t, action_t) not in visited:
                visited.add((state_t, action_t))
                returns_sum[state_t][action_t] += G
                returns_count[state_t][action_t] += 1
                Q[state_t][action_t] = (returns_sum[state_t][action_t] /
                                        returns_count[state_t][action_t])

        if (ep + 1) % 50000 == 0:
            win_rate = wins / (ep + 1)
            print(f"Episode {ep + 1}: 승률 = {win_rate:.3f}")

    env.close()
    return Q


def visualize_blackjack_policy(Q):
    """블랙잭 정책 시각화"""
    print("\n=== 사용 가능한 에이스가 없을 때 ===")
    print("딜러 카드:  A  2  3  4  5  6  7  8  9  10")
    print("-" * 50)

    for player_sum in range(21, 10, -1):
        row = f"합계 {player_sum:2d}:  "
        for dealer in range(1, 11):
            state = (player_sum, dealer, False)
            if state in Q:
                action = np.argmax(Q[state])
                row += "H  " if action == 1 else "S  "
            else:
                row += "?  "
        print(row)

    print("\n=== 사용 가능한 에이스가 있을 때 ===")
    print("딜러 카드:  A  2  3  4  5  6  7  8  9  10")
    print("-" * 50)

    for player_sum in range(21, 11, -1):
        row = f"합계 {player_sum:2d}:  "
        for dealer in range(1, 11):
            state = (player_sum, dealer, True)
            if state in Q:
                action = np.argmax(Q[state])
                row += "H  " if action == 1 else "S  "
            else:
                row += "?  "
        print(row)


# 학습 실행
Q = learn_blackjack(n_episodes=500000)
visualize_blackjack_policy(Q)
```

---

## 7. MC 방법의 장단점

### 7.1 장점

| 장점 | 설명 |
|------|------|
| 모델 프리 | 환경 모델 불필요 |
| 편향 없음 | 부트스트래핑 없이 실제 리턴 사용 |
| 직관적 | 평균 리턴 계산으로 간단함 |
| 에피소드 독립 | 에피소드별 병렬 처리 가능 |

### 7.2 단점

| 단점 | 설명 |
|------|------|
| 높은 분산 | 전체 리턴의 분산이 큼 |
| 에피소드 종료 필요 | 연속 태스크에 부적합 |
| 느린 학습 | 에피소드 끝까지 기다려야 함 |
| 탐험 문제 | 방문하지 않은 상태 학습 불가 |

### 7.3 DP vs MC 요약

```
                    DP                    MC
              ─────────────          ─────────────
필요 정보     환경 모델              경험 (에피소드)

업데이트      V(s) ← Σ P(s'|s,a)    V(s) ← 평균(G)
방식          [R + γV(s')]

편향          편향 있음              편향 없음
              (부트스트랩)           (실제 리턴)

분산          낮음                   높음
              (기대값 계산)          (샘플 분산)

적용 환경     유한 MDP               에피소딕 태스크
```

---

## 8. 요약

### 핵심 개념

| 개념 | 설명 |
|------|------|
| 몬테카를로 | 경험으로부터 학습 |
| First-visit | 첫 방문만 카운트 |
| Every-visit | 모든 방문 카운트 |
| On-policy | 행동 정책 = 목표 정책 |
| Off-policy | 행동 정책 ≠ 목표 정책 |
| Importance Sampling | 분포 보정을 위한 가중치 |

### MC 정책 평가 공식

$$V(s) = \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i(s)$$

### MC Q-함수 업데이트

$$Q(s, a) \leftarrow Q(s, a) + \frac{1}{N(s,a)} (G - Q(s, a))$$

---

## 9. 연습 문제

1. **First-visit vs Every-visit**: 동일한 상태를 3번 방문하는 에피소드에서 두 방법의 차이를 계산하세요.

2. **탐험 문제**: ε-greedy에서 ε=0이면 어떤 문제가 발생하나요?

3. **Importance Sampling**: 목표 정책이 결정적이고 행동 정책이 ε-greedy일 때, 중요도 비율이 발산할 수 있는 경우는?

4. **수렴 속도**: MC의 분산이 높은 이유와 해결 방법을 설명하세요.

---

## 다음 단계

다음 레슨 **05_TD_Learning.md**에서는 DP의 부트스트래핑과 MC의 샘플링을 결합한 **TD 학습**을 배웁니다.

---

## 참고 자료

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 5
- David Silver's RL Course, Lecture 4: Model-Free Prediction
- [Gymnasium Blackjack](https://gymnasium.farama.org/environments/toy_text/blackjack/)
