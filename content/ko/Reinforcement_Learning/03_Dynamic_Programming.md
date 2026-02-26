# 03. 동적 프로그래밍 (Dynamic Programming)

**난이도: ⭐⭐ (기초)**

## 학습 목표
- 동적 프로그래밍(DP)의 기본 개념 이해
- 정책 평가 (Policy Evaluation) 알고리즘 구현
- 정책 반복 (Policy Iteration) 알고리즘 이해 및 구현
- 가치 반복 (Value Iteration) 알고리즘 이해 및 구현
- DP 방법의 한계점 파악

---

## 1. 동적 프로그래밍이란?

### 1.1 개요

**동적 프로그래밍(Dynamic Programming, DP)**은 복잡한 문제를 더 작은 부분 문제로 나누어 해결하는 방법입니다. RL에서 DP는 **완전한 환경 모델(MDP)**을 알고 있을 때 최적 정책을 계산합니다.

### 1.2 DP의 핵심 아이디어

1. **최적 부분 구조**: 최적 해가 부분 문제의 최적 해로 구성됨
2. **중복 부분 문제**: 같은 부분 문제가 여러 번 계산됨
3. **메모이제이션**: 계산 결과를 저장하여 재사용

```
벨만 방정식의 재귀적 구조를 활용하여 가치 함수를 반복적으로 개선

V_{k+1}(s) = f(V_k(s'))

현재 추정값     다음 상태 추정값으로
 업데이트           계산
```

### 1.3 DP가 필요로 하는 것

| 필요 조건 | 설명 |
|-----------|------|
| 완전한 MDP | 전이 확률 P(s'|s,a)를 알아야 함 |
| 유한 상태/행동 | 테이블로 저장 가능해야 함 |
| 계산 자원 | 모든 상태를 순회해야 함 |

---

## 2. 정책 평가 (Policy Evaluation)

### 2.1 목표

주어진 정책 π의 가치 함수 $V^\pi$를 계산합니다.

### 2.2 반복적 정책 평가 알고리즘

벨만 기대 방정식을 반복 적용하여 수렴할 때까지 업데이트합니다.

$$V_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V_k(s') \right]$$

```python
import numpy as np
from typing import Dict, List, Tuple

def policy_evaluation(mdp, policy: Dict, gamma: float = 0.9,
                      theta: float = 1e-6) -> Dict:
    """
    정책 평가: 주어진 정책의 가치 함수 계산

    Args:
        mdp: MDP 환경
        policy: 정책 {state: {action: probability}}
        gamma: 할인율
        theta: 수렴 임계값

    Returns:
        V: 상태 가치 함수 {state: value}
    """
    # 0으로 초기화 — 벨만 연산자(Bellman operator)는 수축 사상(contraction mapping)이므로
    # 초기값과 관계없이 수렴이 보장됨
    V = {s: 0.0 for s in mdp.get_states()}

    iteration = 0
    while True:
        # delta는 이번 스윕에서 가장 큰 단일 상태 변화를 추적 — 합이 아닌 max를 사용하면
        # 최악의 경우 경계를 제공하므로, 수렴 시 모든 상태가 안정됨을 보장
        delta = 0
        iteration += 1

        # 모든 상태에 대해 업데이트
        for s in mdp.get_states():
            if mdp.is_terminal(s):
                continue

            v = V[s]  # 이전 값 저장
            new_v = 0

            # 벨만 기대 방정식 적용
            for a in mdp.actions:
                action_prob = policy[s].get(a, 0)

                for prob, next_s, reward, done in mdp.get_transitions(s, a):
                    if done:
                        new_v += action_prob * prob * reward
                    else:
                        # 현재 스윕의 V[next_s]를 사용하는 것이 "인플레이스(in-place)" 방식 —
                        # 루프 앞쪽 상태의 업데이트가 뒤쪽 상태에 바로 반영되어 더 빠르게 수렴
                        new_v += action_prob * prob * (reward + gamma * V[next_s])

            V[s] = new_v
            delta = max(delta, abs(v - new_v))

        # theta는 수렴 임계값: delta < theta이면 가치 함수가 진짜 V^π의
        # theta/(1-gamma) 이내에 있음을 보장
        if delta < theta:
            print(f"정책 평가 수렴: {iteration} iterations")
            break

    return V


# 사용 예시
class SimpleGridWorld:
    """간단한 그리드 월드"""
    def __init__(self, size=4):
        self.size = size
        self.actions = ['up', 'down', 'left', 'right']

    def get_states(self):
        return [(i, j) for i in range(self.size) for j in range(self.size)]

    def is_terminal(self, state):
        return state == (0, 0) or state == (self.size-1, self.size-1)

    def get_transitions(self, state, action):
        if self.is_terminal(state):
            return [(1.0, state, 0, True)]

        deltas = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        delta = deltas[action]

        new_row = max(0, min(self.size-1, state[0] + delta[0]))
        new_col = max(0, min(self.size-1, state[1] + delta[1]))
        next_state = (new_row, new_col)

        return [(1.0, next_state, -1, self.is_terminal(next_state))]


# 균등 랜덤 정책
def create_uniform_policy(mdp):
    policy = {}
    for s in mdp.get_states():
        policy[s] = {a: 1/len(mdp.actions) for a in mdp.actions}
    return policy


# 테스트
grid = SimpleGridWorld(4)
uniform_policy = create_uniform_policy(grid)
V = policy_evaluation(grid, uniform_policy)

# 결과 출력
print("\n균등 랜덤 정책의 가치 함수:")
for i in range(grid.size):
    row = [f"{V[(i,j)]:6.1f}" for j in range(grid.size)]
    print(" ".join(row))
```

### 2.3 수렴성

정책 평가는 항상 $V^\pi$로 수렴합니다 (감마 < 1 또는 에피소딕 태스크).

```
반복 0: V = [0, 0, 0, 0, ...]
반복 1: V = [-1, -1, -1, ...]
반복 2: V = [-1.9, -2.0, ...]
  ...
수렴: V = V^π
```

---

## 3. 정책 개선 (Policy Improvement)

### 3.1 정책 개선 정리

현재 정책 π의 가치 함수 $V^\pi$가 주어졌을 때, **탐욕적 정책** π'은 π보다 같거나 더 좋습니다.

$$\pi'(s) = \arg\max_{a} Q^\pi(s, a) = \arg\max_{a} \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V^\pi(s') \right]$$

```python
def policy_improvement(mdp, V: Dict, gamma: float = 0.9) -> Tuple[Dict, bool]:
    """
    정책 개선: V를 기반으로 탐욕적 정책 생성

    Args:
        mdp: MDP 환경
        V: 현재 가치 함수
        gamma: 할인율

    Returns:
        new_policy: 개선된 정책
        stable: 정책이 변하지 않았으면 True
    """
    new_policy = {}
    policy_stable = True

    for s in mdp.get_states():
        if mdp.is_terminal(s):
            new_policy[s] = {a: 1/len(mdp.actions) for a in mdp.actions}
            continue

        # 별도의 Q 테이블 저장 없이 V에서 Q 값을 재계산 —
        # 메모리를 절약하고 Q가 항상 현재 V와 일관되게 유지됨
        q_values = {}
        for a in mdp.actions:
            q = 0
            for prob, next_s, reward, done in mdp.get_transitions(s, a):
                if done:
                    q += prob * reward
                else:
                    q += prob * (reward + gamma * V[next_s])
            q_values[a] = q

        # 최적 행동 찾기
        best_action = max(q_values, key=q_values.get)
        best_q = q_values[best_action]

        # 1e-8 허용 오차로 동률 처리: 부동소수점 노이즈 처리 — 없으면 수학적으로 동일한
        # 두 행동이 다르게 취급되어 반복 간 불필요한 정책 진동이 발생할 수 있음
        best_actions = [a for a, q in q_values.items()
                        if abs(q - best_q) < 1e-8]

        # 동률인 최선 행동들에 균등 확률 분배 — 일반적으로 단일 승자인 결정적 정책을 만들되
        # 여러 행동이 진정으로 동등할 때는 균등 분배로 자연스럽게 처리
        new_policy[s] = {a: 0.0 for a in mdp.actions}
        for a in best_actions:
            new_policy[s][a] = 1.0 / len(best_actions)

    return new_policy, policy_stable


def get_greedy_action(mdp, V, state, gamma):
    """가치 함수에서 탐욕적 행동 선택"""
    q_values = {}
    for a in mdp.actions:
        q = 0
        for prob, next_s, reward, done in mdp.get_transitions(state, a):
            if done:
                q += prob * reward
            else:
                q += prob * (reward + gamma * V[next_s])
        q_values[a] = q

    return max(q_values, key=q_values.get)
```

---

## 4. 정책 반복 (Policy Iteration)

### 4.1 알고리즘

정책 평가와 정책 개선을 번갈아 수행하여 최적 정책을 찾습니다.

```
1. 초기화: 임의의 정책 π₀

2. 반복:
   a. 정책 평가: V^πₖ 계산
   b. 정책 개선: πₖ₊₁ = greedy(V^πₖ)

3. πₖ₊₁ = πₖ이면 종료 (최적 정책)
```

### 4.2 구현

```python
def policy_iteration(mdp, gamma: float = 0.9, theta: float = 1e-6):
    """
    정책 반복 알고리즘

    Returns:
        V: 최적 가치 함수
        policy: 최적 정책
    """
    # 균등 랜덤 정책으로 초기화 — 최적이 아닐 수 있지만, 첫 번째 평가 과정에서
    # 모든 상태가 탐색되도록 보장하는 안전한 출발점
    policy = create_uniform_policy(mdp)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n=== 정책 반복 {iteration} ===")

        # 1. 정책 평가 — 개선 전에 완전히 수렴할 때까지 실행;
        # 이것이 가치 반복(1회만 평가)과의 핵심 차이점
        V = policy_evaluation(mdp, policy, gamma, theta)

        # 2. 정책 개선
        old_policy = policy.copy()
        new_policy = {}
        policy_stable = True

        for s in mdp.get_states():
            if mdp.is_terminal(s):
                new_policy[s] = {a: 1/len(mdp.actions) for a in mdp.actions}
                continue

            # Q 값 계산
            q_values = {}
            for a in mdp.actions:
                q = 0
                for prob, next_s, reward, done in mdp.get_transitions(s, a):
                    if done:
                        q += prob * reward
                    else:
                        q += prob * (reward + gamma * V[next_s])
                q_values[a] = q

            # 탐욕적 정책: 최선의 행동에 확률 1.0을 완전히 부여 —
            # 결정적 탐욕 정책은 항상 혼합 정책 이상으로 좋음
            best_action = max(q_values, key=q_values.get)
            new_policy[s] = {a: 0.0 for a in mdp.actions}
            new_policy[s][best_action] = 1.0

            # policy_stable은 어느 상태든 최선 행동이 바뀌었는지 추적 —
            # 아무 상태도 변하지 않았다면 정책은 이미 자신의 V에 대해 탐욕적,
            # 즉 최적임을 의미 (정책 개선 정리, Policy Improvement Theorem)
            old_best = max(old_policy[s], key=old_policy[s].get)
            if old_best != best_action:
                policy_stable = False

        policy = new_policy

        # 3. 수렴 체크
        if policy_stable:
            print(f"\n정책 반복 수렴! (총 {iteration} iterations)")
            break

    return V, policy


# 테스트
print("=" * 50)
print("정책 반복 (Policy Iteration)")
print("=" * 50)

grid = SimpleGridWorld(4)
V_star, optimal_policy = policy_iteration(grid)

# 결과 출력
print("\n최적 가치 함수:")
for i in range(grid.size):
    row = [f"{V_star[(i,j)]:6.1f}" for j in range(grid.size)]
    print(" ".join(row))

print("\n최적 정책:")
arrows = {'up': '^', 'down': 'v', 'left': '<', 'right': '>'}
for i in range(grid.size):
    row = []
    for j in range(grid.size):
        s = (i, j)
        if grid.is_terminal(s):
            row.append('  *  ')
        else:
            best_a = max(optimal_policy[s], key=optimal_policy[s].get)
            row.append(f'  {arrows[best_a]}  ')
    print(" ".join(row))
```

---

## 5. 가치 반복 (Value Iteration)

### 5.1 아이디어

정책 평가를 완전히 수렴시키지 않고, **한 번의 업데이트** 후 바로 정책 개선을 수행합니다.

$$V_{k+1}(s) = \max_{a} \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V_k(s') \right]$$

### 5.2 알고리즘

```
1. 초기화: V₀(s) = 0 for all s

2. 반복:
   Vₖ₊₁(s) = max_a Σ P(s'|s,a)[R + γVₖ(s')]

3. |Vₖ₊₁ - Vₖ| < θ이면 종료

4. 최적 정책: π*(s) = argmax_a Q*(s,a)
```

### 5.3 구현

```python
def value_iteration(mdp, gamma: float = 0.9, theta: float = 1e-6):
    """
    가치 반복 알고리즘

    Returns:
        V: 최적 가치 함수
        policy: 최적 정책
    """
    # 가치 함수 초기화
    V = {s: 0.0 for s in mdp.get_states()}

    iteration = 0
    while True:
        delta = 0
        iteration += 1

        for s in mdp.get_states():
            if mdp.is_terminal(s):
                continue

            v = V[s]

            # 벨만 최적성 방정식: 정책 가중 평균 대신 max를 취하면
            # 매 스윕마다 암묵적으로 정책 개선이 수행됨 — 상태별 평가+개선 루프가
            # 단일 업데이트로 압축되는 것
            q_values = []
            for a in mdp.actions:
                q = 0
                for prob, next_s, reward, done in mdp.get_transitions(s, a):
                    if done:
                        q += prob * reward
                    else:
                        q += prob * (reward + gamma * V[next_s])
                q_values.append(q)

            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))

        if iteration % 10 == 0:
            print(f"반복 {iteration}: delta = {delta:.6f}")

        if delta < theta:
            print(f"\n가치 반복 수렴: {iteration} iterations")
            break

    # V가 수렴한 후 최적 정책 추출 — 반복 중 π를 저장하지 않고 별도 패스로 추출하는 이유는
    # 중간의 암묵적 정책이 아닌 최종 탐욕 정책만 필요하기 때문
    policy = {}
    for s in mdp.get_states():
        if mdp.is_terminal(s):
            policy[s] = {a: 1/len(mdp.actions) for a in mdp.actions}
            continue

        q_values = {}
        for a in mdp.actions:
            q = 0
            for prob, next_s, reward, done in mdp.get_transitions(s, a):
                if done:
                    q += prob * reward
                else:
                    q += prob * (reward + gamma * V[next_s])
            q_values[a] = q

        best_action = max(q_values, key=q_values.get)
        policy[s] = {a: 0.0 for a in mdp.actions}
        policy[s][best_action] = 1.0

    return V, policy


# 테스트
print("=" * 50)
print("가치 반복 (Value Iteration)")
print("=" * 50)

grid = SimpleGridWorld(4)
V_star, optimal_policy = value_iteration(grid)

# 결과 출력
print("\n최적 가치 함수:")
for i in range(grid.size):
    row = [f"{V_star[(i,j)]:6.1f}" for j in range(grid.size)]
    print(" ".join(row))
```

---

## 6. 정책 반복 vs 가치 반복

### 6.1 비교

| 특성 | 정책 반복 | 가치 반복 |
|------|----------|----------|
| 내부 루프 | 완전 수렴까지 평가 | 1회 업데이트 |
| 외부 반복 수 | 적음 | 많음 |
| 반복당 계산량 | 많음 | 적음 |
| 수렴 속도 | 빠름 (반복 수) | 느림 (반복 수) |
| 메모리 | V와 π 모두 저장 | V만 저장 |

### 6.2 일반화된 정책 반복 (GPI)

```
        평가
    ┌─────────────┐
    │  V → V^π   │
    └─────────────┘
         ↑    ↓
         │    │
         │    ↓
    ┌─────────────┐
    │  π' = greedy(V) │
    └─────────────┘
        개선

정책 반복: 완전 수렴 후 개선
가치 반복: 1회 업데이트 후 개선
k-step 반복: k회 업데이트 후 개선
```

### 6.3 비동기 DP (Asynchronous DP)

모든 상태를 순서대로 업데이트하지 않고, **일부 상태만 선택적**으로 업데이트합니다.

```python
def async_value_iteration(mdp, gamma=0.9, n_updates=10000):
    """비동기 가치 반복"""
    V = {s: 0.0 for s in mdp.get_states()}
    states = [s for s in mdp.get_states() if not mdp.is_terminal(s)]

    for i in range(n_updates):
        # 무작위로 상태 선택
        s = np.random.choice(len(states))
        s = states[s]

        # 해당 상태만 업데이트
        q_values = []
        for a in mdp.actions:
            q = 0
            for prob, next_s, reward, done in mdp.get_transitions(s, a):
                if done:
                    q += prob * reward
                else:
                    q += prob * (reward + gamma * V[next_s])
            q_values.append(q)

        V[s] = max(q_values)

    return V
```

---

## 7. DP의 한계

### 7.1 주요 한계점

| 한계 | 설명 |
|------|------|
| 완전한 모델 필요 | P(s'\|s,a)와 R을 정확히 알아야 함 |
| 차원의 저주 | 상태/행동 공간이 크면 계산 불가능 |
| 테이블 저장 | 연속 상태 공간에서 불가능 |
| 계산량 | 매 반복마다 모든 상태 순회 |

### 7.2 해결 방향

```
한계                    해결책
──────────────────────────────────────────────
모델 필요         →     모델 프리 방법 (MC, TD)
차원의 저주       →     함수 근사 (신경망)
테이블 불가       →     연속 공간 알고리즘
계산량           →     샘플 기반 학습
```

---

## 8. 완전한 예제: 얼음 호수 (Frozen Lake)

```python
import gymnasium as gym
import numpy as np

def dp_frozen_lake():
    """Frozen Lake 환경에서 DP 적용"""

    # is_slippery=False는 확률적 전이를 제거하여 DP가 매우 적은 반복으로 수렴 —
    # is_slippery=True를 사용하면 바람/미끄러짐에 대한 강건성을 테스트할 수 있음
    env = gym.make('FrozenLake-v1', is_slippery=False)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # gamma=0.99: 4x4 그리드에서 목표까지 최대 6단계이므로 보상 신호를
    # 시작 상태까지 전파하려면 높은 gamma가 필요 (0.9이면 너무 많이 할인됨)
    gamma = 0.99
    theta = 1e-8

    # P[s][a] = [(prob, next_state, reward, done), ...]
    # env.unwrapped.P는 Gymnasium에서 모델을 읽는 표준 방법 —
    # .unwrapped 없이는 래퍼(wrapper)가 속성 접근을 가로챌 수 있음
    P = env.unwrapped.P

    # 가치 반복
    V = np.zeros(n_states)

    for iteration in range(1000):
        delta = 0

        for s in range(n_states):
            v = V[s]

            # 각 행동의 가치 계산
            q_values = []
            for a in range(n_actions):
                # (not done)은 비종료 상태에서 1, 종료 상태에서 0이므로
                # 별도의 if 분기 없이 종료 상태의 V[next_s]를 자동으로 0으로 만듦
                q = sum(prob * (reward + gamma * V[next_s] * (not done))
                        for prob, next_s, reward, done in P[s][a])
                q_values.append(q)

            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            print(f"수렴: {iteration + 1} iterations")
            break

    # 최적 정책 추출
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = []
        for a in range(n_actions):
            q = sum(prob * (reward + gamma * V[next_s] * (not done))
                    for prob, next_s, reward, done in P[s][a])
            q_values.append(q)
        policy[s] = np.argmax(q_values)

    # 결과 시각화
    action_names = ['←', '↓', '→', '↑']
    print("\n최적 정책 (4x4 그리드):")
    for i in range(4):
        row = ""
        for j in range(4):
            s = i * 4 + j
            if s in [5, 7, 11, 12]:  # 구멍
                row += "  H  "
            elif s == 15:  # 목표
                row += "  G  "
            else:
                row += f"  {action_names[policy[s]]}  "
        print(row)

    print("\n가치 함수:")
    print(V.reshape(4, 4).round(3))

    # 새로운 환경에서 정책 테스트: 실제 성능 측정 —
    # env를 재생성하여 학습 단계의 상태 누수를 방지
    env = gym.make('FrozenLake-v1', is_slippery=False)
    success = 0
    n_tests = 100

    for _ in range(n_tests):
        state, _ = env.reset()
        done = False

        while not done:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if reward > 0:
            success += 1

    print(f"\n성공률: {success}/{n_tests} = {success/n_tests*100:.1f}%")

    env.close()
    return V, policy


# 실행
if __name__ == "__main__":
    V, policy = dp_frozen_lake()
```

---

## 9. 요약

### 핵심 알고리즘

| 알고리즘 | 용도 | 핵심 방정식 |
|----------|------|-------------|
| 정책 평가 | V^π 계산 | 벨만 기대 방정식 |
| 정책 개선 | π → π' | argmax Q |
| 정책 반복 | 최적 정책 | 평가 + 개선 반복 |
| 가치 반복 | 최적 정책 | 벨만 최적성 방정식 |

### 시간 복잡도

| 알고리즘 | 반복당 복잡도 |
|----------|---------------|
| 정책 평가 | O(\|S\|^2 \|A\|) |
| 정책 개선 | O(\|S\| \|A\|) |
| 가치 반복 | O(\|S\|^2 \|A\|) |

---

## 10. 연습 문제

1. **정책 평가**: 2x2 그리드에서 균등 랜덤 정책의 가치 함수를 손으로 계산하세요.

2. **정책 반복**: 정책 반복이 항상 유한 단계에서 종료함을 설명하세요.

3. **가치 반복**: γ=0이면 가치 반복이 1회 만에 수렴하는 이유는?

4. **비동기 DP**: 비동기 DP가 동기 DP보다 빠를 수 있는 상황을 설명하세요.

---

## 다음 단계

다음 레슨 **04_Monte_Carlo_Methods.md**에서는 **환경 모델 없이** 경험으로부터 학습하는 몬테카를로 방법을 배웁니다.

---

## 참고 자료

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 4
- David Silver's RL Course, Lecture 3: Planning by Dynamic Programming
- [Gymnasium FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
