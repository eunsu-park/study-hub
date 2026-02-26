# 05. 시간차 학습 (Temporal Difference Learning)

**난이도: ⭐⭐⭐ (중급)**

## 학습 목표
- TD 학습의 기본 개념과 TD(0) 알고리즘 이해
- TD Target과 부트스트래핑 개념 파악
- TD와 MC, DP의 차이점 비교
- TD의 편향-분산 트레이드오프 이해
- n-step TD와 TD(λ) 학습

---

## 1. TD 학습이란?

### 1.1 개요

**시간차 학습(Temporal Difference, TD)**은 DP의 **부트스트래핑**과 MC의 **샘플링**을 결합한 방법입니다.

```
     DP: V(s) ← E[R + γV(s')]           (모델 필요, 부트스트랩)
     MC: V(s) ← 평균(G)                  (모델 불필요, 전체 리턴)
     TD: V(s) ← V(s) + α[R + γV(s') - V(s)]  (모델 불필요, 부트스트랩)
```

### 1.2 TD vs MC vs DP 비교

| 특성 | DP | MC | TD |
|------|-----|-----|-----|
| 환경 모델 | 필요 | 불필요 | 불필요 |
| 부트스트래핑 | O | X | O |
| 샘플링 | X | O | O |
| 업데이트 시점 | 매 스텝 | 에피소드 종료 | 매 스텝 |
| 연속 태스크 | O | X | O |
| 편향 | O | X | O |
| 분산 | 낮음 | 높음 | 중간 |

---

## 2. TD(0) 알고리즘

### 2.1 TD Target

TD(0)는 다음 상태의 추정값을 사용하여 현재 상태의 가치를 업데이트합니다.

**TD Target**: $R_{t+1} + \gamma V(S_{t+1})$

**TD Error (δ)**: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

**업데이트 규칙**:
$$V(S_t) \leftarrow V(S_t) + \alpha \cdot \delta_t$$
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

### 2.2 TD(0) 구현

```python
import numpy as np
import gymnasium as gym
from collections import defaultdict

def td0_prediction(env, policy, n_episodes=10000, alpha=0.1, gamma=0.99):
    """
    TD(0) 정책 평가

    Args:
        env: Gymnasium 환경
        policy: 정책 함수 policy(state) -> action
        n_episodes: 에피소드 수
        alpha: 학습률
        gamma: 할인율

    Returns:
        V: 상태 가치 함수
    """
    # defaultdict(float)은 방문하지 않은 상태를 0으로 초기화 — 에이전트가
    # 미방문 상태에 대해 편향 없이 중립적으로 시작할 수 있는 안전한 초기값
    V = defaultdict(float)

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD(0) 업데이트
            if done:
                td_target = reward  # 종료 상태: V(s') = 0
            else:
                # 부트스트랩(Bootstrap): 실제 보상과 V(s')의 할인 추정값을 혼합 —
                # 에피소드 종료를 기다리지 않고 즉시 업데이트할 수 있는 핵심 원리
                td_target = reward + gamma * V[next_state]

            # TD 오차(error)는 현재 추정이 얼마나 틀렸는지를 나타내며, alpha는
            # 수정 강도를 조절 (alpha가 크면 빠르지만 불안정, 작으면 안정적이지만 느림)
            td_error = td_target - V[state]
            V[state] = V[state] + alpha * td_error

            state = next_state

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{n_episodes}")

    return dict(V)


# 사용 예시
env = gym.make('CliffWalking-v0')

def random_policy(state):
    return env.action_space.sample()

V = td0_prediction(env, random_policy, n_episodes=5000)
print(f"학습된 상태 수: {len(V)}")
```

### 2.3 TD(0) vs MC 비교 시각화

```python
import matplotlib.pyplot as plt
import numpy as np

def compare_td_mc(env, policy, n_episodes=500, alpha=0.1, gamma=0.99):
    """TD(0)와 MC의 학습 곡선 비교"""

    # TD(0)
    V_td = defaultdict(float)
    td_errors = []

    # MC
    V_mc = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    mc_errors = []

    for episode in range(n_episodes):
        # 에피소드 생성
        episode_data = []
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_data.append((state, action, reward, next_state, done))
            state = next_state

        # TD(0) 업데이트 (온라인)
        for state, action, reward, next_state, done in episode_data:
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * V_td[next_state]
            V_td[state] += alpha * (td_target - V_td[state])

        # MC 업데이트 (오프라인)
        G = 0
        for state, action, reward, _, _ in reversed(episode_data):
            G = reward + gamma * G
            returns_sum[state] += G
            returns_count[state] += 1
            V_mc[state] = returns_sum[state] / returns_count[state]

        # 특정 상태의 추정값 기록 (비교용)
        test_state = episode_data[0][0]
        td_errors.append(V_td[test_state])
        mc_errors.append(V_mc[test_state])

    return td_errors, mc_errors


# 학습 곡선 시각화
# env = gym.make('CliffWalking-v0')
# td_curve, mc_curve = compare_td_mc(env, random_policy)
#
# plt.figure(figsize=(10, 5))
# plt.plot(td_curve, label='TD(0)', alpha=0.7)
# plt.plot(mc_curve, label='MC', alpha=0.7)
# plt.xlabel('Episode')
# plt.ylabel('Value Estimate')
# plt.legend()
# plt.title('TD(0) vs MC Learning Curves')
# plt.show()
```

---

## 3. 부트스트래핑 (Bootstrapping)

### 3.1 개념

**부트스트래핑**은 다른 추정값을 사용하여 추정값을 업데이트하는 것입니다.

```
MC:  V(s) ← V(s) + α[G - V(s)]
     실제 리턴 G 사용 (부트스트래핑 X)

TD:  V(s) ← V(s) + α[R + γV(s') - V(s)]
     추정값 V(s') 사용 (부트스트래핑 O)
```

### 3.2 부트스트래핑의 영향

```python
"""
부트스트래핑의 장점:
1. 에피소드 종료 전에 학습 가능
2. 연속 태스크에 적용 가능
3. 분산이 낮음 (한 스텝 보상만 사용)

부트스트래핑의 단점:
1. 편향 발생 (V(s')이 부정확하면 전파)
2. 초기 추정값에 민감
3. 수렴 보장이 MC보다 복잡
"""

# 부트스트래핑 시각화
def visualize_bootstrapping():
    """
    MC: S₀ → S₁ → S₂ → S₃ → 종료 (G = r₁ + γr₂ + γ²r₃ + γ³r₄)
                                   └── 전체 리턴 사용

    TD: S₀ → S₁ → S₂ → ...
        V(S₀) ← R₁ + γV(S₁)
                     └── 추정값 사용 (부트스트랩)
    """
    pass
```

---

## 4. TD Error의 의미

### 4.1 TD Error 분석

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

- **δ > 0**: 예상보다 좋은 결과 → V(s) 증가
- **δ < 0**: 예상보다 나쁜 결과 → V(s) 감소
- **δ = 0**: 예상과 일치 → 변화 없음

### 4.2 TD Error와 신경과학

```
도파민 신경세포의 반응이 TD Error와 유사!

예상치 못한 보상 → 도파민 증가 (δ > 0)
예상한 보상 획득 → 도파민 변화 없음 (δ ≈ 0)
예상 보상 미획득 → 도파민 감소 (δ < 0)

→ TD 학습이 뇌의 학습 메커니즘과 유사할 수 있음
```

---

## 5. TD 학습의 장점

### 5.1 Random Walk 예제

```python
def random_walk_comparison():
    """
    Random Walk에서 TD와 MC 비교

    환경: A - B - C - D - E - [종료]
          ←              →
    왼쪽 종료: 보상 0
    오른쪽 종료: 보상 1
    """
    import numpy as np

    # 상태: 0=왼쪽종료, 1-5=A-E, 6=오른쪽종료
    n_states = 7
    true_values = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1])  # 실제 가치

    def run_episode():
        """에피소드 생성"""
        state = 3  # C에서 시작
        episode = [(state, 0, state)]

        while 0 < state < 6:
            if np.random.random() < 0.5:
                state -= 1  # 왼쪽
            else:
                state += 1  # 오른쪽

            reward = 1 if state == 6 else 0
            episode.append((state, reward, state))

        return episode

    # TD(0) 학습
    V_td = np.full(n_states, 0.5)
    V_td[0] = V_td[6] = 0

    alpha = 0.1
    n_episodes = 100

    for _ in range(n_episodes):
        state = 3
        while 0 < state < 6:
            if np.random.random() < 0.5:
                next_state = state - 1
            else:
                next_state = state + 1

            reward = 1 if next_state == 6 else 0
            V_td[state] += alpha * (reward + V_td[next_state] - V_td[state])
            state = next_state

    print("True Values:", true_values[1:6])
    print("TD Estimates:", V_td[1:6].round(3))
    print("TD RMSE:", np.sqrt(np.mean((V_td[1:6] - true_values[1:6])**2)))

    return V_td


V_td = random_walk_comparison()
```

### 5.2 TD의 장점 요약

| 장점 | 설명 |
|------|------|
| 온라인 학습 | 에피소드 종료 전에 학습 가능 |
| 연속 태스크 | 종료 없는 태스크에 적용 가능 |
| 낮은 분산 | 한 스텝 보상만 사용 |
| 점진적 개선 | 실시간으로 정책 개선 가능 |

---

## 6. Batch TD vs Batch MC

### 6.1 배치 학습

동일한 데이터셋에서 반복 학습할 때, TD와 MC는 다른 값으로 수렴합니다.

```python
def batch_td_mc_comparison():
    """
    배치 학습에서 TD와 MC 비교

    예시 데이터:
    Episode 1: A → B → 0 (보상 0)
    Episode 2: B → 1 (보상 1)

    MC: V(A) = 0, V(B) = 1/2
    TD: V(A) = 3/4 * V(B) = 3/4, V(B) = 1 (A→B이므로 V(A) ≈ V(B))
    """
    # 간단한 예시
    episodes = [
        [('A', 'B', 0), ('B', 'terminal', 0)],  # A → B → 종료(0)
        [('B', 'terminal', 1)]                   # B → 종료(1)
    ]

    # Batch MC
    V_mc = {'A': 0, 'B': 0}
    returns = {'A': [], 'B': []}

    for ep in episodes:
        G = 0
        for state, next_state, reward in reversed(ep):
            G = reward + G
            if state != 'terminal':
                returns[state].append(G)

    for state in V_mc:
        if returns[state]:
            V_mc[state] = np.mean(returns[state])

    # Batch TD (반복)
    V_td = {'A': 0, 'B': 0, 'terminal': 0}
    alpha = 0.1

    for _ in range(100):  # 배치 반복
        for ep in episodes:
            for state, next_state, reward in ep:
                if state != 'terminal':
                    V_td[state] += alpha * (reward + V_td[next_state] - V_td[state])

    print("Batch MC:", V_mc)
    print("Batch TD:", {k: round(v, 3) for k, v in V_td.items() if k != 'terminal'})


batch_td_mc_comparison()
```

### 6.2 수렴 특성

- **MC**: 최소 제곱 오차 최소화 (관찰된 리턴에 적합)
- **TD**: 최대 가능도 MDP 추정 (전이 확률을 암묵적으로 학습)

---

## 7. n-step TD

### 7.1 개념

TD(0)는 1-step 리턴을 사용하지만, n-step TD는 n개의 실제 보상을 사용합니다.

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

```
1-step: G_t^(1) = R_{t+1} + γV(S_{t+1})                    ← TD(0)
2-step: G_t^(2) = R_{t+1} + γR_{t+2} + γ²V(S_{t+2})
3-step: G_t^(3) = R_{t+1} + γR_{t+2} + γ²R_{t+3} + γ³V(S_{t+3})
...
∞-step: G_t^(∞) = R_{t+1} + γR_{t+2} + ...                ← MC
```

### 7.2 n-step TD 구현

```python
def n_step_td(env, policy, n=3, n_episodes=1000, alpha=0.1, gamma=0.99):
    """
    n-step TD 정책 평가

    Args:
        n: 스텝 수 (n=1이면 TD(0), n=∞이면 MC)
    """
    V = defaultdict(float)

    for episode in range(n_episodes):
        states = []
        rewards = [0]  # R_0 = 0 (사용하지 않음)
        # T를 무한대로 시작하면 종료 상태(terminal)를 발견할 때까지
        # 행동 루프가 자연스럽게 계속 실행됨
        T = float('inf')  # 종료 시점
        t = 0

        state, _ = env.reset()
        states.append(state)

        while True:
            if t < T:
                action = policy(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t + 1

                state = next_state

            # tau는 현재 시점에서 n 스텝 뒤처진 위치 — tau 이후 n개의
            # 실제 보상이 확보되어야 비로소 상태 tau를 업데이트할 수 있음
            tau = t - n + 1  # 업데이트할 시점

            if tau >= 0:
                # tau+1부터 n개의 실제 보상을 할인 합산;
                # 실제 보상을 더 많이 사용할수록 순수 부트스트랩보다 편향이 줄어듦
                G = sum(gamma ** (i - tau - 1) * rewards[i]
                        for i in range(tau + 1, min(tau + n, T) + 1))

                if tau + n < T:
                    # 에피소드가 아직 끝나지 않았을 때만 부트스트랩 추정값을 더함 —
                    # 경계에서는 순수 MC 리턴이 되어 V가 필요 없음
                    G += gamma ** n * V[states[tau + n]]

                # 업데이트
                V[states[tau]] += alpha * (G - V[states[tau]])

            t += 1

            if tau == T - 1:
                break

    return dict(V)
```

### 7.3 n의 선택

| n 값 | 특성 |
|------|------|
| n=1 (TD(0)) | 편향 높음, 분산 낮음 |
| n=∞ (MC) | 편향 없음, 분산 높음 |
| 중간 n | 균형점 (환경에 따라 최적 n 다름) |

---

## 8. TD(λ) - Eligibility Traces

### 8.1 개념

모든 n-step 리턴의 **가중 평균**을 사용합니다.

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

- **λ=0**: TD(0)
- **λ=1**: MC

### 8.2 Eligibility Trace

각 상태의 "자격"을 추적하여 효율적으로 계산합니다.

```python
def td_lambda(env, policy, lambd=0.8, n_episodes=1000, alpha=0.1, gamma=0.99):
    """
    TD(λ) with Eligibility Traces

    Args:
        lambd: λ 값 (0 ≤ λ ≤ 1)
    """
    V = defaultdict(float)

    for episode in range(n_episodes):
        # 에피소드 시작 시 trace를 초기화 — eligibility(자격)는 에피소드 단위로
        # 관리되므로 한 에피소드의 크레딧이 다음 에피소드로 누출되지 않음
        E = defaultdict(float)

        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD error
            if done:
                delta = reward - V[state]
            else:
                delta = reward + gamma * V[next_state] - V[state]

            # 방문한 상태의 trace를 누적 — 최근에 또는 자주 방문한 상태일수록
            # eligibility가 커져 더 큰 업데이트를 받음
            E[state] += 1  # accumulating traces

            # TD 오차를 eligibility 가중치로 모든 상태에 전파 —
            # 이것이 순방향(forward-view) n-step TD보다 효율적인 핵심 이유
            for s in E:
                V[s] += alpha * delta * E[s]
                # gamma*lambda 감쇠: gamma는 미래 보상을 할인하고, lambda는
                # 크레딧이 얼마나 과거로 전파될지를 결정 (lambda=0 → TD(0), lambda=1 → MC)
                E[s] *= gamma * lambd  # trace 감소

            state = next_state

    return dict(V)
```

### 8.3 Eligibility Trace 종류

```python
"""
1. Accumulating Traces (누적)
   E(s) ← E(s) + 1    (방문할 때마다 누적)

2. Replacing Traces (교체)
   E(s) ← 1           (방문 시 1로 리셋)

3. Dutch Traces (네덜란드)
   E(s) ← (1-α)E(s) + 1
"""

def accumulating_trace(E, state):
    E[state] += 1
    return E

def replacing_trace(E, state):
    E[state] = 1
    return E

def dutch_trace(E, state, alpha):
    E[state] = (1 - alpha) * E[state] + 1
    return E
```

---

## 9. 예제: Cliff Walking

```python
import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def cliff_walking_td():
    """Cliff Walking에서 TD 학습"""

    env = gym.make('CliffWalking-v0')

    # 상태: 0-47 (4x12 그리드)
    # 행동: 0=up, 1=right, 2=down, 3=left

    # lambda 초기화를 사용하면 새 상태마다 독립적인 0 배열이 생성됨 —
    # 하나의 배열을 여러 상태가 공유하면 모든 Q값이 오염될 수 있음
    Q = defaultdict(lambda: np.zeros(4))
    epsilon = 0.1
    # alpha=0.5는 예측(prediction)에서 쓰는 0.1보다 높음 — 절벽 환경은
    # 보상 구조가 명확하여 공격적인 학습률도 안정적으로 동작함
    alpha = 0.5
    gamma = 0.99
    n_episodes = 500

    episode_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # ε-greedy 행동 선택
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Q-learning 업데이트 (다음 레슨에서 상세히)
            # 종료 상태에서 best_next=0으로 설정해 에피소드 종료 후 가상의
            # 미래 가치가 추가되는 것을 방지
            best_next = np.max(Q[next_state]) if not done else 0
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

            state = next_state

        episode_rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {ep + 1}: avg_reward = {avg:.1f}")

    # 최적 경로 시각화
    print("\n=== 학습된 정책 (4x12 그리드) ===")
    arrows = {0: '^', 1: '>', 2: 'v', 3: '<'}

    for row in range(4):
        line = ""
        for col in range(12):
            state = row * 12 + col
            if state == 36:  # 시작
                line += " S "
            elif state == 47:  # 목표
                line += " G "
            elif 37 <= state <= 46:  # 절벽
                line += " C "
            else:
                action = np.argmax(Q[state])
                line += f" {arrows[action]} "
        print(line)

    env.close()
    return Q, episode_rewards


Q, rewards = cliff_walking_td()
```

---

## 10. 요약

### 핵심 공식

| 방법 | 업데이트 규칙 |
|------|---------------|
| TD(0) | $V(s) \leftarrow V(s) + \alpha[R + \gamma V(s') - V(s)]$ |
| n-step TD | $V(s) \leftarrow V(s) + \alpha[G^{(n)} - V(s)]$ |
| TD(λ) | $V(s) \leftarrow V(s) + \alpha \delta E(s)$ |

### TD Error

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

### 방법 비교

```
                    TD(0)        n-step        MC
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
편향               높음         중간          없음
분산               낮음         중간          높음
업데이트 시점      매 스텝      n 스텝 후     에피소드 종료
연속 태스크        가능         가능          불가능
```

---

## 11. 연습 문제

1. **TD Error**: V(s)=5, R=1, γ=0.9, V(s')=6일 때 TD error는?

2. **n-step**: n=2일 때의 리턴 공식을 작성하세요.

3. **TD(λ)**: λ=0.5일 때 1-step과 2-step 리턴의 가중치 비율은?

4. **Eligibility Trace**: 상태 s가 연속으로 2번 방문되면 accumulating trace의 값은?

---

## 다음 단계

다음 레슨 **06_Q_Learning_SARSA.md**에서는 TD를 제어에 적용하는 **Q-Learning**과 **SARSA** 알고리즘을 학습합니다.

---

## 참고 자료

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 6, 7
- David Silver's RL Course, Lecture 4 & 5
- [Gymnasium CliffWalking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)
