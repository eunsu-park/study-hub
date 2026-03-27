# 02. 마르코프 결정 과정 (Markov Decision Process)

**난이도: ⭐⭐ (기초)**

## 학습 목표
- 마르코프 성질과 MDP의 정의 이해
- 상태 가치 함수 V(s)와 행동 가치 함수 Q(s,a) 이해
- 벨만 기대 방정식과 최적성 방정식 유도 및 해석
- 최적 정책의 존재성과 특성 파악

---

## 1. 마르코프 성질 (Markov Property)

### 1.1 정의

**마르코프 성질**: 미래는 오직 현재 상태에만 의존하고, 과거 이력에는 독립적입니다.

$$P(S_{t+1} | S_t) = P(S_{t+1} | S_1, S_2, \ldots, S_t)$$

```python
"""
마르코프 성질 예시:

✓ 체스 게임: 현재 보드 상태만 알면 최적 수를 둘 수 있음
  - 과거에 어떤 수를 두었는지는 중요하지 않음

✗ 포커 게임: 상대방의 과거 베팅 패턴이 중요
  - 단순히 현재 카드만으로는 최적 결정 불가
  → 과거 정보를 상태에 포함시켜 마르코프화 가능
"""
```

### 1.2 상태 표현의 중요성

마르코프 성질을 만족하려면 **상태가 충분한 정보**를 담아야 합니다.

```python
# 예: 비마르코프 → 마르코프 변환
class StateRepresentation:
    """
    원래 상태: 공의 현재 위치 (x)
    문제: 공이 어디로 움직일지 알 수 없음 (속도 정보 없음)

    마르코프 상태: (위치, 속도) = (x, v)
    해결: 다음 위치 예측 가능 (x' = x + v)
    """

    def non_markov_state(self, ball):
        return ball.position  # 불충분한 정보

    def markov_state(self, ball):
        return (ball.position, ball.velocity)  # 충분한 정보
```

---

## 2. 마르코프 결정 과정 (MDP)

### 2.1 MDP의 5요소

MDP는 튜플 $(S, A, P, R, \gamma)$로 정의됩니다.

| 요소 | 기호 | 설명 |
|------|------|------|
| 상태 공간 | $S$ | 가능한 모든 상태의 집합 |
| 행동 공간 | $A$ | 가능한 모든 행동의 집합 |
| 전이 확률 | $P$ | $P(s'|s, a)$ - 상태 전이 확률 |
| 보상 함수 | $R$ | $R(s, a, s')$ - 즉각 보상 |
| 할인율 | $\gamma$ | 미래 보상의 할인 (0 ≤ γ ≤ 1) |

### 2.2 MDP 다이어그램

```
           a₁                    a₂
    ┌──────────────┐      ┌──────────────┐
    │              │      │              │
    ▼     r=+1     │      ▼     r=-1     │
   s₁ ─────────────┼─────s₂──────────────┼────► s₃
    │    p=0.7     │      │    p=0.3     │      │
    │              │      │              │      │
    └──────────────┘      └──────────────┘      ▼
                                              종료
```

### 2.3 Python으로 MDP 정의

```python
import numpy as np
from typing import Dict, Tuple, List

class MDP:
    """마르코프 결정 과정 클래스"""

    def __init__(self):
        # 상태 공간
        self.states = ['s0', 's1', 's2', 'terminal']

        # 행동 공간
        self.actions = ['left', 'right']

        # 할인율 (discount factor)
        # γ=0.9이면 10단계 후의 보상은 즉각 보상의 0.9^10 ≈ 35% 가치만 가짐 —
        # 이로써 에이전트가 먼 미래와 현재를 동등하게 취급하지 않도록 방지
        self.gamma = 0.9

        # 전이 확률: P[s][a] = [(확률, 다음상태, 보상, 종료여부), ...]
        self.P = self._build_transitions()

    def _build_transitions(self) -> Dict:
        """전이 확률 정의"""
        P = {}

        # 상태 s0에서의 전이
        # 벽 충돌에 -1 보상: 음수 보상을 통해 에이전트가 벽 충돌이
        # 시간 낭비임을 학습하고 회피하도록 유도
        P['s0'] = {
            'left': [(1.0, 's0', -1, False)],      # 벽에 부딪힘
            'right': [(1.0, 's1', 0, False)]       # s1으로 이동
        }

        # 상태 s1에서의 전이
        # 80/20 확률적 분할은 현실의 불확실성을 모델링 — 에이전트는 통제 불가능한
        # 미끄러짐을 고려한 강건한 정책(robust policy)을 학습해야 함
        P['s1'] = {
            'left': [(1.0, 's0', 0, False)],       # s0으로 이동
            'right': [(0.8, 's2', 0, False),       # 80%: s2로 이동
                      (0.2, 's1', 0, False)]       # 20%: 제자리
        }

        # 상태 s2에서의 전이
        P['s2'] = {
            'left': [(1.0, 's1', 0, False)],       # s1으로 이동
            'right': [(1.0, 'terminal', +10, True)] # 목표 도달!
        }

        # 종료 상태는 모든 행동을 흡수하고 보상 0을 반환 —
        # 에이전트가 목표를 "탈출"해 리턴을 인위적으로 부풀리는 것을 방지
        P['terminal'] = {
            'left': [(1.0, 'terminal', 0, True)],
            'right': [(1.0, 'terminal', 0, True)]
        }

        return P

    def get_transitions(self, state: str, action: str) -> List[Tuple]:
        """주어진 상태와 행동에서의 전이 정보 반환"""
        return self.P[state][action]

    def step(self, state: str, action: str) -> Tuple[str, float, bool]:
        """환경에서 한 스텝 실행 (확률적 전이)"""
        transitions = self.P[state][action]
        probs = [t[0] for t in transitions]
        # p= 인수를 사용하는 np.random.choice는 한 번의 호출로 전체 확률 분포를 샘플링 —
        # 수동 루프 없이 다중 결과 확률적 전이를 올바르게 처리
        idx = np.random.choice(len(transitions), p=probs)
        prob, next_state, reward, done = transitions[idx]
        return next_state, reward, done


# MDP 사용 예시
mdp = MDP()

# 에피소드 실행
state = 's0'
total_reward = 0

print("=== MDP 시뮬레이션 ===")
while True:
    action = np.random.choice(mdp.actions)  # 랜덤 정책
    next_state, reward, done = mdp.step(state, action)

    print(f"{state} --[{action}]--> {next_state}, reward={reward}")

    total_reward += reward
    state = next_state

    if done:
        break

print(f"\n총 보상: {total_reward}")
```

---

## 3. 정책 (Policy)

### 3.1 정책의 정의

**정책 π**는 상태에서 행동을 선택하는 규칙입니다.

- **결정적 정책**: $\pi(s) = a$
- **확률적 정책**: $\pi(a|s) = P(A_t = a | S_t = s)$

```python
class Policy:
    """정책 클래스"""

    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        # 확률적 정책: 각 상태에서 행동 확률 분포
        self.policy = {s: {a: 1/len(actions) for a in actions}
                       for s in states}

    def get_action_prob(self, state: str, action: str) -> float:
        """π(a|s) 반환"""
        return self.policy[state][action]

    def sample_action(self, state: str) -> str:
        """정책에 따라 행동 샘플링"""
        actions = list(self.policy[state].keys())
        probs = list(self.policy[state].values())
        return np.random.choice(actions, p=probs)

    def set_deterministic(self, state: str, action: str):
        """결정적 정책으로 설정"""
        for a in self.actions:
            self.policy[state][a] = 1.0 if a == action else 0.0


# 예시: 항상 오른쪽으로 이동하는 정책
policy = Policy(['s0', 's1', 's2'], ['left', 'right'])
policy.set_deterministic('s0', 'right')
policy.set_deterministic('s1', 'right')
policy.set_deterministic('s2', 'right')
```

---

## 4. 가치 함수 (Value Functions)

### 4.1 상태 가치 함수 V(s)

상태 s에서 정책 π를 따를 때 기대되는 **누적 보상**입니다.

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t = s\right]$$

```python
def compute_state_value(mdp, policy, state, gamma, depth=100):
    """
    몬테카를로 방식으로 상태 가치 추정
    (여러 에피소드의 평균 리턴)
    """
    returns = []
    # 1000 에피소드는 통계적으로 안정된 평균을 제공 — 더 적으면 분산이 크고,
    # 더 많으면 정확하지만 계산 낭비
    n_episodes = 1000

    for _ in range(n_episodes):
        s = state
        episode_return = 0
        # 할인을 1.0에서 시작하면 첫 보상이 할인 없이 반영됨 (r * γ^0 = r) —
        # 이후 매 스텝마다 gamma를 곱해 γ^k 가중치를 자동으로 구현
        discount = 1.0

        for _ in range(depth):
            action = policy.sample_action(s)
            next_s, reward, done = mdp.step(s, action)

            episode_return += discount * reward
            discount *= gamma
            s = next_s

            if done:
                break

        returns.append(episode_return)

    # 에피소드 평균으로 E_π[G_t | S_t=state]를 근사 —
    # 큰 수의 법칙(law of large numbers)에 의해 진짜 V^π(state)로 수렴
    return np.mean(returns)
```

### 4.2 행동 가치 함수 Q(s, a)

상태 s에서 행동 a를 취한 후 정책 π를 따를 때의 기대 **누적 보상**입니다.

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

```python
def compute_action_value(mdp, policy, state, action, gamma, depth=100):
    """행동 가치 추정"""
    returns = []
    n_episodes = 1000

    for _ in range(n_episodes):
        # 첫 행동은 주어진 action으로 고정 — 이것이 Q(s,a)와 V(s)의 핵심 차이:
        # 첫 행동을 고정하고, 이후 모든 단계는 정책이 결정
        s = state
        next_s, reward, done = mdp.step(s, action)

        episode_return = reward
        # 할인을 gamma에서 시작 (1.0이 아님): 첫 보상은 이미 위에서 수집했으므로
        # 이후 보상은 k≥1에 해당하여 γ^k로 스케일링
        discount = gamma
        s = next_s

        # 이후는 정책을 따름
        for _ in range(depth - 1):
            if done:
                break

            a = policy.sample_action(s)
            next_s, reward, done = mdp.step(s, a)

            episode_return += discount * reward
            discount *= gamma
            s = next_s

        returns.append(episode_return)

    return np.mean(returns)
```

### 4.3 V와 Q의 관계

$$V^\pi(s) = \sum_{a \in A} \pi(a|s) \cdot Q^\pi(s, a)$$

```python
def v_from_q(policy, q_values, state, actions):
    """Q 값으로부터 V 값 계산"""
    v = 0
    for a in actions:
        v += policy.get_action_prob(state, a) * q_values[(state, a)]
    return v
```

---

## 5. 벨만 방정식 (Bellman Equations)

### 5.1 벨만 기대 방정식 (Bellman Expectation Equation)

현재 상태의 가치를 다음 상태의 가치로 **재귀적**으로 표현합니다.

**상태 가치 함수:**

$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V^\pi(s') \right]$$

**행동 가치 함수:**

$$Q^\pi(s, a) = \sum_{s', r} P(s', r | s, a) \left[ r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]$$

### 5.2 벨만 방정식의 직관적 이해

```
V(s) = 즉각 보상 + 할인된 미래 가치

     ┌─────────────────────────────────────────┐
     │   V(s) = r + γ * V(s')                  │
     │                                          │
     │   현재 가치 = 즉각 보상 + γ × 다음 상태 가치│
     └─────────────────────────────────────────┘
```

### 5.3 벨만 방정식 구현

```python
def bellman_expectation_v(mdp, policy, V, state, gamma):
    """
    벨만 기대 방정식으로 V(s) 계산

    V(s) = Σ_a π(a|s) * Σ_{s',r} P(s',r|s,a) * [r + γV(s')]
    """
    # 종료 상태의 가치는 항상 0 — 앞으로 수집할 보상이 없으므로,
    # 이 베이스 케이스로 재귀가 무한 루프에 빠지는 것을 방지
    if state == 'terminal':
        return 0

    value = 0

    for action in mdp.actions:
        action_prob = policy.get_action_prob(state, action)

        for prob, next_state, reward, done in mdp.get_transitions(state, action):
            if done:
                # 종료 상태로 전이 시 미래 가치는 0 — 즉각 보상만 반영하고
                # γ * V(terminal)은 더하지 않음
                value += action_prob * prob * reward
            else:
                # 이중 합산(행동 → 결과)이 완전한 기댓값을 계산:
                # E_π[r + γV(s')] = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[r + γV(s')]
                value += action_prob * prob * (reward + gamma * V.get(next_state, 0))

    return value


def bellman_expectation_q(mdp, policy, Q, state, action, gamma):
    """
    벨만 기대 방정식으로 Q(s,a) 계산

    Q(s,a) = Σ_{s',r} P(s',r|s,a) * [r + γ * Σ_a' π(a'|s') * Q(s',a')]
    """
    value = 0

    for prob, next_state, reward, done in mdp.get_transitions(state, action):
        if done:
            value += prob * reward
        else:
            # V(s') = Σ_a' π(a'|s') Q(s',a') — 다음 상태에서 Q를 다시 V로 변환하면
            # Q를 기존 Q 추정값만으로 완전히 표현할 수 있음
            next_value = sum(
                policy.get_action_prob(next_state, a) * Q.get((next_state, a), 0)
                for a in mdp.actions
            )
            value += prob * (reward + gamma * next_value)

    return value
```

---

## 6. 최적 가치 함수와 최적 정책

### 6.1 최적 가치 함수

**최적 상태 가치 함수** $V^*(s)$: 모든 정책 중 최대 가치

$$V^*(s) = \max_\pi V^\pi(s)$$

**최적 행동 가치 함수** $Q^*(s, a)$:

$$Q^*(s, a) = \max_\pi Q^\pi(s, a)$$

### 6.2 벨만 최적성 방정식 (Bellman Optimality Equation)

$$V^*(s) = \max_{a} \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V^*(s') \right]$$

$$Q^*(s, a) = \sum_{s', r} P(s', r | s, a) \left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$

```python
def bellman_optimality_v(mdp, V, state, gamma):
    """
    벨만 최적성 방정식으로 V*(s) 계산

    V*(s) = max_a Σ_{s',r} P(s',r|s,a) * [r + γV*(s')]
    """
    if state == 'terminal':
        return 0

    # -inf로 초기화하면 실제 행동 가치가 무조건 이를 대체 —
    # 0으로 초기화하면 모든 행동이 음의 보상을 줄 때 잘못된 결과를 낳음
    max_value = float('-inf')

    for action in mdp.actions:
        action_value = 0

        for prob, next_state, reward, done in mdp.get_transitions(state, action):
            if done:
                action_value += prob * reward
            else:
                action_value += prob * (reward + gamma * V.get(next_state, 0))

        # max는 정책의 가중 평균을 대체 — 최적성이란 모든 행동을 평균하는 것이 아니라
        # 항상 최선의 단일 행동을 선택한다는 의미
        max_value = max(max_value, action_value)

    return max_value


def bellman_optimality_q(mdp, Q, state, action, gamma):
    """
    벨만 최적성 방정식으로 Q*(s,a) 계산

    Q*(s,a) = Σ_{s',r} P(s',r|s,a) * [r + γ * max_a' Q*(s',a')]
    """
    value = 0

    for prob, next_state, reward, done in mdp.get_transitions(state, action):
        if done:
            value += prob * reward
        else:
            # 다음 행동의 max는 최적성 원리를 구현: s'에서는 에이전트가
            # 이후 항상 최선의 행동을 취한다고 가정
            max_next_q = max(Q.get((next_state, a), 0) for a in mdp.actions)
            value += prob * (reward + gamma * max_next_q)

    return value
```

### 6.3 최적 정책

최적 정책 $\pi^*$은 $Q^*$를 알면 쉽게 도출됩니다:

$$\pi^*(s) = \arg\max_{a} Q^*(s, a)$$

```python
def get_optimal_policy(mdp, Q_star):
    """Q*로부터 최적 정책 도출"""
    optimal_policy = {}

    for state in mdp.states:
        if state == 'terminal':
            continue

        # 각 행동의 Q* 값 계산
        q_values = {a: Q_star.get((state, a), 0) for a in mdp.actions}

        # 최대 Q* 값을 가진 행동 선택
        optimal_action = max(q_values, key=q_values.get)
        optimal_policy[state] = optimal_action

    return optimal_policy
```

---

## 7. MDP의 종류

### 7.1 유한 MDP vs 무한 MDP

| 특성 | 유한 MDP | 무한 MDP |
|------|----------|----------|
| 상태 공간 | 유한 | 무한 (연속) |
| 행동 공간 | 유한 | 무한 (연속) |
| 표현 | 테이블 | 함수 근사 필요 |
| 예시 | 그리드 월드 | 로봇 제어 |

### 7.2 결정적 vs 확률적 MDP

```python
# 결정적 MDP: P(s'|s,a) = 1 for one s'
deterministic_transitions = {
    's0': {'right': [(1.0, 's1', 0, False)]}  # 100% s1으로 이동
}

# 확률적 MDP: P(s'|s,a) can be < 1
stochastic_transitions = {
    's0': {'right': [(0.8, 's1', 0, False),   # 80% s1으로 이동
                     (0.2, 's0', -1, False)]}  # 20% 제자리
}
```

### 7.3 Partially Observable MDP (POMDP)

상태를 직접 관찰할 수 없고 **관측(observation)**만 가능한 경우

```python
"""
POMDP 예시: 포커 게임
- 실제 상태: 모든 플레이어의 카드 + 덱의 카드
- 관측: 자신의 카드 + 공개 카드만
- 신념 상태(belief state): 실제 상태에 대한 확률 분포 유지

해결 방법:
1. 신념 상태를 상태로 사용 (belief MDP)
2. 과거 관측 히스토리 사용
3. RNN/LSTM으로 암묵적 상태 학습
"""
```

---

## 8. 그리드 월드 예제

### 8.1 환경 정의

```python
import numpy as np

class GridWorld:
    """
    4x4 그리드 월드

    [S][ ][ ][ ]
    [ ][X][ ][ ]
    [ ][ ][ ][ ]
    [ ][ ][ ][G]

    S: 시작, G: 목표(+1), X: 장애물(-1)
    """

    def __init__(self):
        self.size = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.obstacle = (1, 1)

        self.actions = ['up', 'down', 'left', 'right']
        self.action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        self.gamma = 0.9

    def get_states(self):
        """모든 상태 반환"""
        states = []
        for i in range(self.size):
            for j in range(self.size):
                states.append((i, j))
        return states

    def is_terminal(self, state):
        """종료 상태 확인"""
        return state == self.goal

    def get_reward(self, state, action, next_state):
        """보상 함수"""
        if next_state == self.goal:
            return 1.0
        elif next_state == self.obstacle:
            return -1.0
        else:
            return -0.01  # 스텝 페널티

    def get_transitions(self, state, action):
        """
        전이 확률 반환
        80% 의도한 방향, 10%씩 좌우로 미끄러짐
        """
        # 종료 상태에 대한 조기 반환으로 action_deltas 인덱싱을 회피 —
        # 흡수 상태(absorbing state) 계약을 깔끔하게 유지
        if self.is_terminal(state):
            return [(1.0, state, 0, True)]

        transitions = []

        # 의도한 방향
        intended_delta = self.action_deltas[action]

        # 미끄러짐 방향은 의도한 행동에 수직 — 실제 환경에서 불확실성은 직교 방향으로
        # 발생하는 것이 일반적이며(뒤로 밀리는 것은 드묾), 물리적 미끄러짐 역학을 반영
        if action in ['up', 'down']:
            slip_actions = ['left', 'right']
        else:
            slip_actions = ['up', 'down']

        # 세 결과를 하나의 리스트로 묶어 전이 로직 중복 없이
        # 확률 가중 방향에 대해 균일하게 반복 처리
        directions = [(0.8, action), (0.1, slip_actions[0]), (0.1, slip_actions[1])]

        for prob, dir_action in directions:
            delta = self.action_deltas[dir_action]
            next_state = self._move(state, delta)
            # 보상은 의도한 상태가 아닌 결과 상태에서 평가 —
            # 미끄러져서 장애물에 도달해도 페널티를 받음
            reward = self.get_reward(state, action, next_state)
            done = self.is_terminal(next_state)
            transitions.append((prob, next_state, reward, done))

        return transitions

    def _move(self, state, delta):
        """이동 처리 (벽 충돌 시 제자리)"""
        new_row = state[0] + delta[0]
        new_col = state[1] + delta[1]

        # 격자 범위 체크
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            return (new_row, new_col)
        else:
            return state  # 벽에 부딪힘
```

### 8.2 가치 함수 시각화

```python
def visualize_values(grid, V):
    """가치 함수 시각화"""
    print("\n=== State Values ===")
    print("-" * 40)

    for i in range(grid.size):
        row_str = "|"
        for j in range(grid.size):
            state = (i, j)
            value = V.get(state, 0)

            if state == grid.goal:
                row_str += f"  G   |"
            elif state == grid.obstacle:
                row_str += f"  X   |"
            else:
                row_str += f"{value:6.2f}|"
        print(row_str)
        print("-" * 40)


def visualize_policy(grid, policy):
    """정책 시각화"""
    arrows = {'up': '^', 'down': 'v', 'left': '<', 'right': '>'}

    print("\n=== Optimal Policy ===")
    print("-" * 25)

    for i in range(grid.size):
        row_str = "|"
        for j in range(grid.size):
            state = (i, j)

            if state == grid.goal:
                row_str += "  G  |"
            elif state == grid.obstacle:
                row_str += "  X  |"
            elif state in policy:
                row_str += f"  {arrows[policy[state]]}  |"
            else:
                row_str += "  ?  |"
        print(row_str)
        print("-" * 25)
```

---

## 9. 요약

### 핵심 개념

| 개념 | 설명 |
|------|------|
| 마르코프 성질 | 미래는 현재 상태에만 의존 |
| MDP | (S, A, P, R, γ)로 정의되는 의사결정 문제 |
| 정책 π | 상태에서 행동을 선택하는 전략 |
| V(s) | 상태의 장기적 기대 가치 |
| Q(s,a) | 상태-행동 쌍의 기대 가치 |

### 벨만 방정식 요약

| 방정식 | 수식 |
|--------|------|
| 기대 (V) | $V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R + \gamma V^\pi(s')]$ |
| 기대 (Q) | $Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$ |
| 최적 (V) | $V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R + \gamma V^*(s')]$ |
| 최적 (Q) | $Q^*(s,a) = \sum_{s'} P(s'|s,a)[R + \gamma \max_{a'} Q^*(s',a')]$ |

---

## 10. 연습 문제

1. **마르코프 성질**: 다음 중 마르코프 성질을 만족하는 것은?
   - (a) 오늘의 주가가 내일 주가 결정
   - (b) 체스 보드 상태
   - (c) 대화의 현재 문장만으로 적절한 응답 결정

2. **MDP 정의**: 3x3 그리드 월드의 MDP 요소를 정의하세요.

3. **벨만 방정식**: 할인율 γ=0.9, 즉각 보상 r=1, V(s')=5일 때 V(s)는?

4. **최적 정책**: Q*(s, left)=3, Q*(s, right)=5일 때 최적 행동은?

---

## 다음 단계

다음 레슨 **03_Dynamic_Programming.md**에서는 MDP의 해를 구하는 **동적 프로그래밍** 방법(정책 반복, 가치 반복)을 학습합니다.

---

## 참고 자료

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 3
- David Silver's RL Course, Lecture 2: Markov Decision Processes
- [MDP Wikipedia](https://en.wikipedia.org/wiki/Markov_decision_process)
