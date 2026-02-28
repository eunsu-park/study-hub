[이전: 커리큘럼 학습](./15_Curriculum_Learning.md)

---

# 16. 계층적 강화학습(Hierarchical Reinforcement Learning)

## 학습 목표

이 레슨을 완료한 후 다음을 할 수 있습니다:

1. 강화학습에서 계층적 분해(hierarchical decomposition)와 시간적 추상화(temporal abstraction)의 동기를 설명한다
2. 반-마르코프 결정 프로세스(semi-Markov decision processes)를 위한 옵션(Options) 프레임워크를 구현한다
3. 봉건 네트워크(feudal networks)와 그 관리자-작업자(manager-worker) 아키텍처를 설명한다
4. 사후 행동 재레이블링(Hindsight Action Relabeling, HAR)이 있는 목표 조건부 정책(goal-conditioned policies)을 구축한다
5. 연속 제어(continuous control)를 위한 HIRO, Option-Critic, 봉건 방식을 비교한다

---

## 목차

1. [계층적 강화학습이 필요한 이유?](#1-계층적-강화학습이-필요한-이유)
2. [옵션 프레임워크](#2-옵션-프레임워크)
3. [Option-Critic 아키텍처](#3-option-critic-아키텍처)
4. [봉건 네트워크(FeUdal)](#4-봉건-네트워크-feudal)
5. [HIRO: 목표 조건부 계층](#5-hiro-목표-조건부-계층)
6. [HAM 및 다른 접근법](#6-ham-및-다른-접근법)
7. [실용적 고려사항](#7-실용적-고려사항)
8. [연습문제](#8-연습문제)

---

## 1. 계층적 강화학습이 필요한 이유?

### 1.1 긴 지평선(Long Horizon)의 저주

표준 강화학습은 수백 또는 수천 개의 순차적 결정을 필요로 하는 과제에서 어려움을 겪습니다. 신용 할당(credit assignment) 문제가 심각해집니다 — 1000개의 행동 중 어떤 것이 최종 보상으로 이어졌을까요?

```
평면 강화학습: "샌드위치 만들기"

  a₁  a₂  a₃  a₄  ...  a₅₀₀  a₅₀₁  ...  a₁₀₀₀  → 보상
  ↑    ↑    ↑    ↑        ↑      ↑          ↑
  어떤 행동이 보상을 유발했나? 1000단계에 걸친 신용 할당.

계층적 강화학습: "샌드위치 만들기"

  상위 수준:  "빵 가져오기" → "속재료 가져오기" → "조립하기" → 보상
  하위 수준:  각각 5-20개 행동 (걷기, 열기, 잡기, ...)

  신용 할당: 3개 상위 결정 + 하위 과제당 5-20개 하위 수준
  훨씬 배우기 쉬움!
```

### 1.2 시간적 추상화(Temporal Abstraction)

계층적 강화학습은 **시간적 추상화(temporal abstraction)**를 도입합니다: 여러 시간 단계에 걸쳐 확장되는 행동. "주방으로 가기" 행동은 50개의 기본 단계가 걸릴 수 있지만, 상위 수준 정책은 이를 단일 결정으로 봅니다.

```
                    시간 →
기본 행동:  ←→←→←→←→←→←→←→←→←→←→←→←→←→←→
                    ↕  개별 모터 명령 (Δt)

옵션/스킬:  ←────────→←──────→←─────────→
                    "주방으로    "컵        "책상으로
                     가기"       집기"       돌아오기"

상위 목표:  ←──────────────────→←────────→
                    "커피 가져오기"  "마시기"

각 수준은 서로 다른 시간적 규모에서 동작함.
```

### 1.3 계층의 이점

| 이점 | 설명 |
|------|------|
| **빠른 신용 할당** | 상위 수준 결정은 소수 → 좋은 전략 파악이 더 쉬움 |
| **전이 학습(Transfer learning)** | 한 과제에서 배운 스킬(옵션)을 다른 과제에서 재사용 가능 |
| **탐색** | 시간적으로 확장된 행동이 무작위 기본 행동보다 더 효율적으로 탐색 |
| **해석 가능성** | "문으로 가기 → 문 열기 → 나가기"는 500개 모터 명령보다 이해하기 쉬움 |
| **상태 추상화** | 상위 수준 정책이 압축된 상태 표현에서 동작 가능 |

---

## 2. 옵션 프레임워크(The Options Framework)

### 2.1 형식적 정의

**옵션(option)** o = (I, π, β)는 다음으로 구성됩니다:
- **I ⊆ S**: 시작 집합(Initiation set) (옵션이 시작될 수 있는 상태)
- **π: S → A**: 옵션 내 정책(Intra-option policy) (어떤 기본 행동을 취할지)
- **β: S → [0,1]**: 종료 함수(Termination function) (각 상태에서 종료될 확률)

```
표준 MDP:              옵션이 있는 반-MDP:

  s₀ --a₁--> s₁       s₀ --option₁--> s₃     (옵션이 3단계 소요)
  s₁ --a₂--> s₂            누적 보상 = r₁ + γr₂ + γ²r₃
  s₂ --a₃--> s₃
                       s₃ --option₂--> s₅     (옵션이 2단계 소요)
  3번의 결정                누적 보상 = r₄ + γr₅

                       2번의 결정, 동일한 결과
```

### 2.2 구현

```python
import numpy as np
from collections import defaultdict


class Option:
    """A temporally extended action (option) in the Options framework.

    An option encapsulates a sub-policy that runs for multiple
    time steps until its termination condition is met.
    """

    def __init__(self, name, initiation_set, policy_fn, termination_fn):
        self.name = name
        self.initiation_set = initiation_set  # set of valid start states
        self.policy_fn = policy_fn            # state → action
        self.termination_fn = termination_fn  # state → P(terminate)

    def can_initiate(self, state):
        """Check if this option can start in the given state."""
        return state in self.initiation_set

    def get_action(self, state):
        """Get the primitive action prescribed by this option."""
        return self.policy_fn(state)

    def should_terminate(self, state):
        """Check if this option should terminate."""
        return np.random.random() < self.termination_fn(state)


class OptionAgent:
    """Agent that learns over options using SMDP Q-learning.

    The policy-over-options selects which option to execute,
    then the option runs until termination.
    """

    def __init__(self, options, states, gamma=0.99, lr=0.1, epsilon=0.1):
        self.options = options
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        # Q-values for state-option pairs
        self.Q = defaultdict(lambda: {o.name: 0.0 for o in options})

    def select_option(self, state):
        """Epsilon-greedy option selection."""
        available = [o for o in self.options if o.can_initiate(state)]
        if not available:
            return None

        if np.random.random() < self.epsilon:
            return np.random.choice(available)

        # Greedy: pick option with highest Q-value
        return max(available, key=lambda o: self.Q[state][o.name])

    def execute_option(self, env, option, state):
        """Execute an option until termination, collecting rewards.

        Returns (final_state, cumulative_discounted_reward, steps).
        """
        total_reward = 0.0
        discount = 1.0
        steps = 0

        while True:
            action = option.get_action(state)
            next_state, reward, done = env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            steps += 1

            if done or option.should_terminate(next_state):
                return next_state, total_reward, steps, done

            state = next_state

    def update(self, state, option_name, reward, next_state, steps, done):
        """SMDP Q-learning update for options.

        Uses multi-step discounting since options span multiple steps.
        """
        if done:
            target = reward
        else:
            # Best option value at next state
            best_q = max(self.Q[next_state].values())
            # Discount by γ^steps (option duration)
            target = reward + (self.gamma ** steps) * best_q

        old_q = self.Q[state][option_name]
        self.Q[state][option_name] = old_q + self.lr * (target - old_q)
```

### 2.3 옵션 설계하기

좋은 옵션은 일반적으로 자연스러운 하위 목표(sub-goals)에 해당합니다:

```
내비게이션 과제:                로봇 조작:
  • "방 A로 가기"                • "물체에 닿기"
  • "방 B로 가기"                • "물체 잡기"
  • "문 열기"                    • "물체 들기"
  • "열쇠 집기"                  • "물체 놓기"

각 옵션은 명확한 종료 조건을 가지며
서로 다른 상위 수준 과제에서 재사용 가능.
```

---

## 3. Option-Critic 아키텍처

### 3.1 종단간 옵션 학습(Learning Options End-to-End)

옵션 프레임워크는 수동으로 설계된 옵션이 필요합니다. **Option-Critic**(Bacon et al., 2017)은 경사 하강법(gradient descent)을 통해 옵션, 그 정책, 종료 조건을 동시에 학습합니다.

```
┌──────────────────────────────────────────────┐
│              Option-Critic                    │
│                                              │
│  상태 → 특징 추출기 → 공유 특징              │
│                     │                        │
│         ┌───────────┼───────────┐            │
│         ▼           ▼           ▼            │
│    ┌─────────┐ ┌─────────┐ ┌──────────┐     │
│    │옵션 내  │ │종료     │ │옵션에    │     │
│    │정책들   │ │함수들   │ │대한 정책 │     │
│    │π(a|s,o) │ │β(s,o)   │ │Ω(o|s)   │     │
│    └─────────┘ └─────────┘ └──────────┘     │
│                                              │
│  세 가지 모두 정책 경사(policy gradient)로 학습│
└──────────────────────────────────────────────┘
```

### 3.2 핵심 경사

```python
class OptionCritic:
    """Simplified Option-Critic architecture.

    Learns intra-option policies, termination functions,
    and the policy over options simultaneously.
    """

    def __init__(self, state_dim, action_dim, num_options,
                 hidden_dim=64, lr=0.001):
        self.num_options = num_options
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Intra-option policies: π_o(a|s) for each option o
        # Each option has its own action distribution
        self.option_policies = self._build_policies(
            state_dim, action_dim, num_options, hidden_dim
        )

        # Termination functions: β_o(s) for each option o
        # Sigmoid output ∈ [0,1] = probability of terminating
        self.terminations = self._build_terminations(
            state_dim, num_options, hidden_dim
        )

        # Q(s, o): value of starting option o in state s
        self.q_options = self._build_q(
            state_dim, num_options, hidden_dim
        )

    def _build_policies(self, s_dim, a_dim, n_opts, h_dim):
        """Build intra-option policy networks."""
        # In practice: n_opts separate small networks
        # Each maps state → action probabilities
        policies = []
        for _ in range(n_opts):
            policies.append({
                'W1': np.random.randn(s_dim, h_dim) * 0.1,
                'W2': np.random.randn(h_dim, a_dim) * 0.1,
            })
        return policies

    def _build_terminations(self, s_dim, n_opts, h_dim):
        """Build termination function networks."""
        terms = []
        for _ in range(n_opts):
            terms.append({
                'W1': np.random.randn(s_dim, h_dim) * 0.1,
                'W2': np.random.randn(h_dim, 1) * 0.1,
            })
        return terms

    def _build_q(self, s_dim, n_opts, h_dim):
        """Build Q-value network for options."""
        return {
            'W1': np.random.randn(s_dim, h_dim) * 0.1,
            'W2': np.random.randn(h_dim, n_opts) * 0.1,
        }

    def get_termination_prob(self, state, option_idx):
        """Compute termination probability for an option."""
        t = self.terminations[option_idx]
        h = np.tanh(state @ t['W1'])
        logit = (h @ t['W2']).item()
        return 1.0 / (1.0 + np.exp(-logit))  # sigmoid

    def get_action_probs(self, state, option_idx):
        """Compute action probabilities for an option's policy."""
        p = self.option_policies[option_idx]
        h = np.tanh(state @ p['W1'])
        logits = h @ p['W2']
        # Softmax
        exp_logits = np.exp(logits - logits.max())
        return exp_logits / exp_logits.sum()
```

### 3.3 숙고 비용(The Deliberation Cost)

핵심 과제: 옵션이 지속할 이유가 없으면 즉시 종료하는 경향이 있습니다 (기본 행동으로 퇴화). **숙고 비용(deliberation cost)**은 옵션이 종료될 때마다 작은 패널티를 추가하여 더 긴 옵션을 장려합니다:

```
숙고 비용 없이:
  옵션이 매 단계 종료 → 평면 강화학습과 동일
  (시간적 추상화가 학습되지 않음)

숙고 비용 ξ = 0.01:
  종료 패널티가 불필요한 전환을 억제
  옵션이 의미 있는 기간 동안 지속하도록 학습
  결과: 해석 가능한 하위 스킬이 나타남
```

---

## 4. 봉건 네트워크(Feudal Networks, FeUdal)

### 4.1 관리자-작업자 아키텍처(Manager-Worker Architecture)

봉건 네트워크(Vezhnevets et al., 2017)는 봉건 통치에서 영감을 받은 2수준 계층을 구현합니다:

```
┌──────────────────────────────────────┐
│             관리자(Manager)           │
│  낮은 시간적 해상도에서 동작           │
│  잠재 공간에서 목표(방향)를            │
│  c 단계마다 설정                       │
│                                       │
│  g_t = f_manager(s_t)                │
│  (목표 = 방향 벡터)                   │
└───────────────┬──────────────────────┘
                │ 목표 g_t
                ▼
┌──────────────────────────────────────┐
│             작업자(Worker)            │
│  매 시간 단계에서 동작                │
│  관리자의 목표에 조건화됨              │
│                                       │
│  a_t = f_worker(s_t, g_t)           │
│  (기본 행동)                          │
└──────────────────────────────────────┘
```

### 4.2 핵심 설계 선택

**방향 목표(Directional goals)**: 관리자는 절대 목표가 아닌 학습된 잠재 공간에서의 방향 벡터 g를 출력합니다. 작업자는 그 방향으로 이동하는 데 대한 보상을 받습니다:

```python
class FeudalManager:
    """Manager module that sets sub-goals for the worker.

    Operates at a slower time scale (every c steps) and
    outputs directional goals in a learned latent space.
    """

    def __init__(self, state_dim, goal_dim, c=10):
        self.goal_dim = goal_dim
        self.c = c  # manager acts every c steps
        self.step_count = 0
        self.current_goal = None

        # Manager network weights (simplified)
        self.W_percept = np.random.randn(state_dim, goal_dim) * 0.1
        self.W_goal = np.random.randn(goal_dim, goal_dim) * 0.1

    def get_goal(self, state):
        """Produce a goal vector every c steps."""
        self.step_count += 1
        if self.step_count % self.c == 1 or self.current_goal is None:
            # Compute latent state
            z = np.tanh(state @ self.W_percept)
            # Goal is a direction in latent space
            goal_raw = z @ self.W_goal
            # Normalize to unit vector (direction only)
            norm = np.linalg.norm(goal_raw) + 1e-8
            self.current_goal = goal_raw / norm
        return self.current_goal


class FeudalWorker:
    """Worker module that executes primitive actions toward goals.

    Receives directional goals from the manager and produces
    actions that move the agent in the goal direction.
    """

    def __init__(self, state_dim, goal_dim, action_dim):
        self.W_state = np.random.randn(state_dim, 64) * 0.1
        self.W_goal = np.random.randn(goal_dim, 64) * 0.1
        self.W_action = np.random.randn(64, action_dim) * 0.1

    def get_action(self, state, goal):
        """Produce primitive action conditioned on state and goal."""
        h_state = np.tanh(state @ self.W_state)
        h_goal = np.tanh(goal @ self.W_goal)
        # Element-wise combination of state and goal representations
        combined = h_state * h_goal
        logits = combined @ self.W_action
        # Softmax for discrete actions
        exp_l = np.exp(logits - logits.max())
        probs = exp_l / exp_l.sum()
        return np.random.choice(len(probs), p=probs)

    @staticmethod
    def intrinsic_reward(state_t, state_t_plus_c, goal):
        """Reward worker for moving in the goal direction.

        Cosine similarity between the direction actually traveled
        and the goal direction set by the manager.
        """
        direction = state_t_plus_c - state_t
        d_norm = np.linalg.norm(direction) + 1e-8
        g_norm = np.linalg.norm(goal) + 1e-8
        cosine_sim = np.dot(direction, goal) / (d_norm * g_norm)
        return cosine_sim
```

### 4.3 훈련 절차

- **관리자**는 외재적(환경) 보상을 최대화하도록 훈련됨
- **작업자**는 내재적 보상(관리자의 목표 방향과의 코사인 유사도)을 최대화하도록 훈련됨
- 관리자는 보상으로 이어지는 **방향**을 학습하고; 작업자는 그 방향으로 **어떻게** 이동할지를 학습함

---

## 5. HIRO: 목표 조건부 계층(Goal-Conditioned Hierarchies)

### 5.1 데이터 효율적 계층적 강화학습(Data-Efficient Hierarchical RL)

HIRO(Nachum et al., 2018)는 목표 조건부 정책(goal-conditioned policies)을 오프-정책(off-policy) 학습과 결합합니다:

```
상위 수준 정책 μ_hi:
  c 단계마다 상태 공간에서 하위 목표 g 출력
  목표: 하위 수준이 c 단계 안에 도달해야 할 상태

하위 수준 정책 μ_lo:
  매 단계, (상태, 목표)에 조건화된 행동 a 출력
  보상: -||s_t - g||  (하위 목표까지의 거리)
  표준 오프-정책 강화학습으로 훈련 (예: TD3)
```

### 5.2 목표 재레이블링(Goal Relabeling, Off-Policy Correction)

핵심 혁신: 버퍼에서 경험을 재현할 때, 상위 수준 목표가 더 이상 유효하지 않을 수 있습니다 (하위 수준 정책이 변경됨). HIRO는 실제로 일어난 것에 맞게 목표를 재레이블링합니다:

```python
class HIRO:
    """Hierarchical RL with Off-Policy Correction (HIRO).

    Two-level hierarchy: high-level sets goals in state space,
    low-level executes primitive actions to reach goals.
    """

    def __init__(self, state_dim, action_dim, goal_dim, c=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.c = c  # sub-goal horizon

        # High-level: state → goal (every c steps)
        self.high_policy = self._build_policy(state_dim, goal_dim)
        # Low-level: (state, goal) → action (every step)
        self.low_policy = self._build_policy(
            state_dim + goal_dim, action_dim
        )

    def _build_policy(self, input_dim, output_dim):
        """Simple MLP policy (placeholder)."""
        return {
            'W1': np.random.randn(input_dim, 64) * 0.1,
            'W2': np.random.randn(64, output_dim) * 0.1,
        }

    def high_level_goal(self, state):
        """High-level policy outputs a sub-goal."""
        h = np.tanh(state @ self.high_policy['W1'])
        goal = np.tanh(h @ self.high_policy['W2'])
        return goal * 5.0  # scale to reasonable range

    def low_level_action(self, state, goal):
        """Low-level policy outputs primitive action."""
        combined = np.concatenate([state, goal])
        h = np.tanh(combined @ self.low_policy['W1'])
        action = np.tanh(h @ self.low_policy['W2'])
        return action

    @staticmethod
    def low_level_reward(state, goal):
        """Intrinsic reward: negative distance to goal."""
        return -np.linalg.norm(state - goal)

    @staticmethod
    def relabel_goal(states_sequence, original_goal):
        """Off-policy goal relabeling for HIRO.

        Find the goal that best explains the low-level's
        actual behavior, enabling off-policy learning.

        Candidate goals: 10 random goals + the state actually reached.
        Select the one that maximizes low-level action log-probability.
        """
        # In practice: sample candidate goals, evaluate which
        # best explains the observed trajectory
        actual_endpoint = states_sequence[-1]
        # The actually reached state is often the best relabeled goal
        return actual_endpoint
```

### 5.3 HIRO 결과

HIRO는 어려운 연속 제어 과제에서 최첨단 결과를 보여주었습니다:

| 과제 | 평면 TD3 | HIRO |
|------|----------|------|
| Ant Navigate | 0.0 | 0.97 |
| Ant Maze (U형태) | 0.0 | 0.89 |
| Ant Push | 0.0 | 0.73 |
| Ant Fall | 0.0 | 0.58 |

평면 강화학습은 이 긴 지평선 과제들에서 완전히 실패하는 반면, HIRO의 계층적 분해는 일관된 성공을 가능하게 합니다.

---

## 6. HAM 및 다른 접근법

### 6.1 추상 기계 계층(Hierarchy of Abstract Machines, HAM)

HAM은 유한 상태 기계(finite-state machines)를 사용하여 정책 공간을 제한합니다:

```
"커피 가져오기"를 위한 HAM:

  ┌──────────┐    문       ┌──────────┐   커피     ┌──────────┐
  │  문으로  │───도달──────►│  기계로  │───도달────►│   컵     │
  │  가기    │             │  가기    │             │  집기    │
  └──────────┘             └──────────┘             └────┬─────┘
       │                                                  │
       │ 막힘 (타임아웃)                                   ▼
       └──────────────── 재시도 ──────────────────── 책상으로 돌아오기
```

각 HAM 상태는 하위 정책을 실행하고, 전환은 조건에 의해 트리거됩니다. 강화학습 에이전트는 각 HAM 상태 내에서만 학습하면 되므로 탐색 공간이 대폭 줄어듭니다.

### 6.2 MAXQ 분해(MAXQ Decomposition)

MAXQ는 가치 함수를 계층적으로 분해합니다:

```
Q(루트, s, a) = V(Navigate, s) + C(루트, s, Navigate)
                ↑                  ↑
                하위 과제           완료 함수:
                Navigate의          Navigate 완료 후
                가치                기대 보상
```

### 6.3 계층적 강화학습(HRL) 방법 비교

| 방법 | 옵션 학습 여부 | 하위 목표 공간 | 오프-정책? | 핵심 혁신 |
|------|--------------|----------------|------------|-----------|
| Options | 수동 | 이산 하위 과제 | 반-MDP Q | 시간적 추상화 |
| Option-Critic | 예 (종단간) | 이산 | 예 | 경사 기반 옵션 학습 |
| FeUdal | 암묵적 | 잠재 방향 | 아니오 (온-정책) | 방향 목표 |
| HIRO | 목표를 통해 | 상태 공간 | 예 | 오프-정책 목표 재레이블링 |
| HAM | 수동 구조 | 유한 상태 기계 상태 | 부분적 | 제한된 정책 공간 |

---

## 7. 실용적 고려사항

### 7.1 계층적 강화학습 사용 시점

| 시나리오 | HRL 사용? | 이유 |
|----------|-----------|------|
| 짧은 에피소드 (< 100단계) | 아니오 | 평면 강화학습으로 충분 |
| 긴 지평선 (> 500단계) | 예 | 신용 할당 이점 |
| 명확한 하위 과제 구조 | 예 | 자연스러운 분해 |
| 밀집 보상, 단순 과제 | 아니오 | 과도한 설계 |
| 희소 보상, 내비게이션 | 예 | 하위 목표가 탐색을 도움 |
| 과제 간 전이 | 예 | 스킬(옵션)이 재사용 가능 |

### 7.2 하위 목표 빈도 선택하기 (c)

하위 목표 지평선 c는 중요한 하이퍼파라미터입니다:

```
c가 너무 작음 (c=1):
  → 평면 강화학습과 동일 (시간적 추상화 없음)
  → 계층의 이점 없음

c가 너무 큼 (c=100):
  → 하위 목표 간격이 너무 넓음
  → 하위 수준 정책이 먼 목표에 도달하기 어려움
  → 상위 수준이 매우 희소한 피드백 받음

최적 범위 (c=10-25):
  → 의미 있는 시간적 추상화
  → 하위 수준이 안정적으로 목표에 도달 가능
  → 상위 수준이 규칙적인 피드백 받음
```

### 7.3 계층적 강화학습 디버깅

```
계층적 강화학습의 일반적인 실패 모드:

1. 하위 수준이 목표를 무시함
   증상: 작업자가 목표에 관계없이 동일한 행동 취함
   해결: 내재적 보상 크기 확인, 목표가 네트워크에 입력되는지 확인

2. 옵션 퇴화 (모두 즉시 종료)
   증상: 옵션 지속 시간 ≈ 1단계
   해결: 숙고 비용 추가, β를 낮게 초기화

3. 모드 붕괴 (1-2개 옵션만 사용)
   증상: 대부분의 옵션이 활성화되지 않음
   해결: 다양성 보너스 추가, 옵션 선택에 엔트로피 정규화

4. 상위 수준이 도달 불가능한 목표 설정
   증상: 하위 수준이 항상 실패, 상위 수준 보상이 항상 낮음
   해결: 목표 공간 제한, 상대 목표 사용, 목표 재레이블링
```

---

## 8. 연습문제

### 연습문제 1: GridWorld에서의 옵션

4개의 방이 있는 GridWorld에서 옵션 프레임워크를 구현하세요:
1. 4개의 옵션 정의: "복도 N/S/E/W로 가기" (방 출구당 하나)
2. 각 방 내에서 Q-학습을 사용하여 옵션 내 정책(intra-option policies) 훈련
3. 반-MDP Q-학습을 사용하여 옵션에 대한 정책(policy-over-options) 훈련
4. 평면 Q-학습과 비교: 수렴까지의 에피소드 수 측정

### 연습문제 2: Option-Critic

간단한 Option-Critic을 구현하세요:
1. 1D 연쇄 환경 생성 (20개 상태, 상태 19에 목표)
2. 학습된 정책과 종료 함수를 가진 4개의 옵션 구현
3. 숙고 비용 추가하고 옵션 지속 시간에 미치는 영향 관찰
4. 각 옵션이 담당하는 상태 시각화 (옵션 "전문화")

### 연습문제 3: 봉건 목표 설정

관리자-작업자 시스템을 구축하세요:
1. 3개의 하위 목표가 있는 2D 내비게이션 환경 생성
2. 관리자가 5 단계마다 방향 벡터 출력
3. 작업자가 내재적 보상 받음 (방향과의 코사인 유사도)
4. 서로 다른 훈련 단계에서 관리자의 목표 방향을 플롯하여 목표가 어떻게 진화하는지 보임

### 연습문제 4: HIRO 목표 재레이블링

HIRO 스타일의 목표 재레이블링을 구현하세요:
1. 희소 단말 보상이 있는 2D 연속 환경 생성
2. 상위 수준 정책이 10 단계마다 위치 목표 설정
3. 하위 수준 정책이 밀집 거리 기반 보상 받음
4. 오프-정책 목표 재레이블링 구현하고 온-정책 훈련과 비교

### 연습문제 5: 계층적 강화학습 아키텍처 비교

동일한 과제에서 세 가지 계층적 강화학습 방법을 비교하세요:
1. 문으로 연결된 2개의 방이 있는 내비게이션 과제 구현
2. 적용: (a) 평면 Q-학습, (b) 수동 설계 스킬이 있는 옵션, (c) 단순 봉건 (관리자 + 작업자)
3. 세 방법 모두의 학습 곡선 플롯
4. 토론: 계층이 도움이 되는 경우 vs 해가 되는 경우

---

*16번 레슨 끝*
