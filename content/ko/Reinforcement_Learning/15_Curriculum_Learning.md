[이전: Soft Actor-Critic](./14_Soft_Actor_Critic.md) | [다음: 계층적 강화학습](./16_Hierarchical_RL.md)

---

# 15. 강화학습을 위한 커리큘럼 학습(Curriculum Learning for RL)

## 학습 목표

이 레슨을 완료한 후 다음을 할 수 있습니다:

1. 커리큘럼 학습(Curriculum Learning)이 어려운 과제에서 강화학습 훈련을 가속화하는 이유를 설명한다
2. 보상 기반 난이도 점수(reward-based difficulty scoring)를 사용하여 자동 커리큘럼 생성(automatic curriculum generation)을 구현한다
3. 에이전트의 숙련도(competence)에 적응하는 자기 주도 학습(self-paced learning) 일정을 설계한다
4. 커리큘럼 설계를 위한 교사-학생(teacher-student) 프레임워크를 적용한다
5. 시뮬레이션에서 실제 환경으로의 전이(sim-to-real transfer)를 위한 암묵적 커리큘럼(implicit curriculum)으로서 도메인 무작위화(domain randomization)를 활용한다

---

## 목차

1. [커리큘럼 학습이 필요한 이유?](#1-커리큘럼-학습이-필요한-이유)
2. [수동 커리큘럼](#2-수동-커리큘럼)
3. [자동 커리큘럼 생성](#3-자동-커리큘럼-생성)
4. [자기 주도 학습](#4-자기-주도-학습)
5. [교사-학생 프레임워크](#5-교사-학생-프레임워크)
6. [도메인 무작위화](#6-도메인-무작위화)
7. [실용적 가이드라인](#7-실용적-가이드라인)
8. [연습문제](#8-연습문제)

---

## 1. 커리큘럼 학습이 필요한 이유?

### 1.1 희소 보상(Sparse Reward) 문제

많은 강화학습 과제는 희소 보상(sparse reward)을 가집니다 — 에이전트는 과제를 완료했을 때만 신호를 받습니다. 복잡한 환경에서 무작위 탐색은 거의 목표에 도달하지 못합니다:

```
과제: 로봇 팔이 블록 5개를 쌓아야 함

무작위 탐색 성공 확률:
  블록 1개: ~10%  에피소드당    ✓  학습 가능
  블록 2개: ~1%                ✓  느리지만 가능
  블록 3개: ~0.01%             ✗  비현실적
  블록 5개: ~0.000001%         ✗  사실상 0

커리큘럼 방식:
  1단계: 블록 1개 놓기 학습    → ~1000 에피소드에 마스터
  2단계: 블록 2개 쌓기         → ~2000 에피소드에 마스터
  3단계: 블록 3개 쌓기         → ~3000 에피소드에 마스터
  ...
  합계: ~10,000 에피소드 vs 커리큘럼 없이는 수렴하지 못함
```

### 1.2 커리큘럼 학습 정의

커리큘럼 학습(Curriculum Learning)은 훈련 과제를 의미 있는 순서 — 쉬운 것에서 어려운 것으로 — 로 제시하여 에이전트가 기술을 점진적으로 쌓도록 합니다. 이는 인간이 학습하는 방식(기기 → 걷기 → 달리기)을 반영합니다.

```
일반 강화학습:                        커리큘럼 강화학습:

┌─────────────────────┐              ┌──────┐  ┌──────┐  ┌──────┐
│     어려운 과제      │              │ 쉬운 │→ │ 중간 │→ │어려운│
│  (희소 보상)         │              │ 과제 │  │ 과제 │  │ 과제 │
│                     │              └──────┘  └──────┘  └──────┘
│  무작위 탐색이       │                 ↑         ↑         ↑
│  보상을 거의 못 찾음 │              기술      기술      완전한
└─────────────────────┘              전이      전이      마스터
```

### 1.3 핵심 설계 질문

| 질문 | 선택지 |
|------|--------|
| "쉬운" vs "어려운"을 무엇으로 정의하는가? | 보상 밀도, 에피소드 길이, 환경 복잡도 |
| 언제 다음 단계로 넘어갈까? | 고정 일정, 성능 임계값, 자동 |
| 과제를 어떻게 순서 지을까? | 선형 진행, 적응형 샘플링, 다중 과제 |
| 커리큘럼을 누가 설계하는가? | 사람(수동), 알고리즘(자동), 학습(메타) |

---

## 2. 수동 커리큘럼

### 2.1 단계 기반 커리큘럼(Stage-Based Curriculum)

가장 단순한 접근: 명시적 진행 기준이 있는 이산적 단계(discrete stages)를 정의합니다.

```python
class StageCurriculum:
    """Manual stage-based curriculum with fixed progression."""

    def __init__(self, stages):
        """
        stages: list of dicts with keys:
            'name': stage identifier
            'config': environment configuration
            'threshold': performance to advance (e.g., success rate)
            'min_episodes': minimum episodes before advancement
        """
        self.stages = stages
        self.current_stage = 0
        self.episode_count = 0
        self.recent_returns = []

    def get_env_config(self):
        """Return current stage's environment configuration."""
        return self.stages[self.current_stage]['config']

    def update(self, episode_return, success):
        """Update curriculum state after each episode."""
        self.episode_count += 1
        self.recent_returns.append(float(success))

        # Keep sliding window of last 100 episodes
        if len(self.recent_returns) > 100:
            self.recent_returns.pop(0)

        # Check advancement criteria
        stage = self.stages[self.current_stage]
        if (self.episode_count >= stage['min_episodes'] and
                len(self.recent_returns) >= 50):
            success_rate = sum(self.recent_returns[-50:]) / 50
            if success_rate >= stage['threshold']:
                self._advance()

    def _advance(self):
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.episode_count = 0
            self.recent_returns.clear()
            print(f"Advanced to stage: {self.stages[self.current_stage]['name']}")
```

### 2.2 예시: 내비게이션 커리큘럼

```python
navigation_stages = [
    {
        'name': 'short_corridors',
        'config': {'maze_size': 5, 'num_obstacles': 0},
        'threshold': 0.9,   # 90% success rate to advance
        'min_episodes': 200
    },
    {
        'name': 'medium_obstacles',
        'config': {'maze_size': 8, 'num_obstacles': 3},
        'threshold': 0.8,
        'min_episodes': 500
    },
    {
        'name': 'large_complex',
        'config': {'maze_size': 15, 'num_obstacles': 10},
        'threshold': 0.7,
        'min_episodes': 1000
    },
]

curriculum = StageCurriculum(navigation_stages)
```

### 2.3 수동 커리큘럼의 한계

- 좋은 단계를 설계하려면 도메인 전문 지식이 필요
- 고정 임계값이 다른 랜덤 시드에서 너무 쉽거나 너무 어려울 수 있음
- 단계 사이에서 세밀한 난이도 조절 불가
- 이산적 도약(discrete jumps)이 단계 전환 시 성능 저하를 초래할 수 있음

---

## 3. 자동 커리큘럼 생성

### 3.1 핵심 아이디어: 학습 경계(Learning Frontier)에서의 과제

이상적인 난이도 수준은 너무 쉽지(이미 마스터됨)도 너무 어렵지(불가능)도 않습니다. 자동 방법들은 이 "학습 경계(learning frontier)"를 추정하고 거기서 과제를 샘플링합니다.

```
                                    ┌─── 학습 경계(Learning Frontier)
                                    │    (학습이 일어나는 곳)
                                    ▼
마스터 ────────────┐         ┌──────────────┐
                    │         │              │
     이미 해결됨    │         │   근접 발달  │   너무 어려움
     (지루함,      │         │   영역(Zone  │   (무작위 보상,
      학습 없음)   │         │    of Prox.  │    학습 없음)
                    │         │    Dev.)     │
                    └─────────┴──────────────┴──────────────
                              ↑
                     여기서 과제를 샘플링!
```

### 3.2 보상 기반 과제 점수(Reward-Based Task Scoring)

각 과제를 에이전트의 최근 성능으로 점수를 매기고, 성능이 중간 수준인 과제를 샘플링합니다:

```python
import numpy as np
from collections import defaultdict


class AutomaticCurriculum:
    """Automatic curriculum that samples tasks at the learning frontier.

    Tasks where the agent achieves intermediate success rates
    (~50%) are most informative for learning.
    """

    def __init__(self, task_space, target_success=0.5, window=20):
        self.task_space = task_space  # list of possible tasks
        self.target_success = target_success
        self.window = window
        self.task_history = defaultdict(list)  # task_id → [success, ...]

    def sample_task(self):
        """Sample a task from the learning frontier."""
        scores = []
        for task in self.task_space:
            history = self.task_history[task['id']]
            if len(history) < 3:
                # Explore: not enough data → high priority
                score = 1.0
            else:
                recent = history[-self.window:]
                success_rate = np.mean(recent)
                # Tasks near target_success are most useful
                # Score peaks at target_success, drops at 0 and 1
                score = 1.0 - abs(success_rate - self.target_success) * 2
                score = max(score, 0.05)  # minimum exploration probability
            scores.append(score)

        # Sample proportional to scores
        probs = np.array(scores) / sum(scores)
        idx = np.random.choice(len(self.task_space), p=probs)
        return self.task_space[idx]

    def update(self, task_id, success):
        """Record outcome for a task."""
        self.task_history[task_id].append(float(success))
```

### 3.3 PAIRED: 적대적 커리큘럼(Adversarial Curriculum)

PAIRED(Dennis et al., 2020)는 세 에이전트를 사용합니다:
1. **환경 설계자(Environment designer)**: 환경을 만드는 적대자(adversary)
2. **주인공(Protagonist)**: 설계된 환경을 해결하도록 학습
3. **대립자(Antagonist)**: 같은 환경을 시도

설계자는 주인공이 해결하지만 대립자는 해결하지 못하는 환경을 만들 때 보상을 받습니다 — 이로써 적절한 난이도 수준의 환경(불가능하지 않지만 도전적인)이 만들어집니다.

```
┌──────────────┐    설계     ┌──────────────┐
│  환경 설계자  │───────────►│    환경       │
│  (적대자)     │            │ (레벨/미로)   │
└──────────────┘            └──────┬───────┘
       ↑                           │
       │                ┌──────────┼──────────┐
       │                ▼                     ▼
   보상 =          ┌────────────┐        ┌────────────┐
   R_주인공 -      │  주인공    │        │  대립자    │
   R_대립자        │ (학습자)   │        │ (기준선)   │
                   └────────────┘        └────────────┘

주인공이 해결하고 대립자가 실패하면 → 적절한 난이도
둘 다 해결하면 → 너무 쉬움 (설계자 보상 낮음)
둘 다 실패하면 → 너무 어려움 (설계자 보상 낮음)
```

---

## 4. 자기 주도 학습(Self-Paced Learning)

### 4.1 숙련도 기반 진행(Competence-Based Progression)

이산적 단계 대신, 자기 주도 학습은 에이전트의 현재 숙련도에 적응하는 연속 난이도 파라미터(continuous difficulty parameter)를 사용합니다:

```python
class SelfPacedLearning:
    """Self-paced curriculum with continuous difficulty adjustment.

    Difficulty increases smoothly as the agent demonstrates competence,
    with rollback if performance drops.
    """

    def __init__(self, min_difficulty=0.0, max_difficulty=1.0,
                 step_up=0.05, step_down=0.1, target_return=0.7):
        self.difficulty = min_difficulty
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.step_up = step_up
        self.step_down = step_down
        self.target_return = target_return
        self.recent_returns = []

    def get_difficulty(self):
        return self.difficulty

    def update(self, normalized_return):
        """Adjust difficulty based on recent performance.

        normalized_return: float in [0, 1], where 1 = perfect
        """
        self.recent_returns.append(normalized_return)
        if len(self.recent_returns) > 50:
            self.recent_returns.pop(0)

        if len(self.recent_returns) >= 10:
            avg = np.mean(self.recent_returns[-10:])
            if avg >= self.target_return:
                # Agent is competent → increase difficulty
                self.difficulty = min(
                    self.difficulty + self.step_up,
                    self.max_difficulty
                )
            elif avg < self.target_return * 0.5:
                # Agent is struggling → decrease difficulty
                self.difficulty = max(
                    self.difficulty - self.step_down,
                    self.min_difficulty
                )

    def get_env_params(self, base_params):
        """Scale environment parameters by current difficulty."""
        d = self.difficulty
        return {
            'obstacle_density': base_params['max_obstacles'] * d,
            'goal_distance': base_params['min_dist'] + (
                base_params['max_dist'] - base_params['min_dist']) * d,
            'noise_level': base_params['max_noise'] * d,
            'time_limit': int(base_params['max_steps'] * (1 - 0.5 * d)),
        }
```

### 4.2 난이도 차원(Difficulty Dimensions)

하나의 난이도 파라미터가 환경의 여러 측면을 제어할 수 있습니다:

| 난이도 0.0 (쉬움) | 난이도 1.0 (어려움) |
|-------------------|---------------------|
| 짧은 거리 | 긴 거리 |
| 장애물 없음 | 장애물 밀집 |
| 노이즈 없음 | 높은 센서 노이즈 |
| 긴 시간 제한 | 짧은 시간 제한 |
| 평탄한 지형 | 거친 지형 |
| 느린 적 | 빠른 적 |

---

## 5. 교사-학생 프레임워크(Teacher-Student Frameworks)

### 5.1 비대칭 자기 대전(Asymmetric Self-Play)

OpenAI의 비대칭 자기 대전(Sukhbaatar et al., 2018)은 두 에이전트를 사용합니다:
- **앨리스(Alice, 교사)**: 일련의 행동을 수행한 후 리셋
- **밥(Bob, 학생)**: 앨리스의 행동을 되돌려야 함 (상태를 역전)

```
앨리스의 과제: "흥미로운 무언가를 해라"
  앨리스가 물체 A, B, C를 이리저리 옮김

밥의 과제: "앨리스가 한 것을 되돌려라"
  밥은 A, B, C를 원래 위치로 되돌려야 함

핵심 통찰: 앨리스는 밥에게 어렵지만 여전히
해결 가능한 과제를 만들도록 동기 부여됨
(앨리스 스스로 할 수 있었던 것이므로!)
```

### 5.2 커리큘럼으로서의 보상 형성(Reward Shaping)

과제를 변경하는 대신, 초기 학습을 안내하기 위해 보상을 재설계합니다:

```python
class CurriculumRewardShaper:
    """Gradually transition from shaped (dense) to sparse reward.

    Early training: dense reward provides learning signal
    Late training: sparse reward matches true objective
    """

    def __init__(self, total_steps, warmup_fraction=0.3):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.current_step = 0

    def shape_reward(self, sparse_reward, dense_reward):
        """Blend sparse and dense rewards based on training progress."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linearly decrease shaping weight
            alpha = 1.0 - (self.current_step / self.warmup_steps)
        else:
            alpha = 0.0  # pure sparse reward

        return sparse_reward + alpha * dense_reward

    @staticmethod
    def compute_dense_reward(state, goal):
        """Example: distance-based shaping for navigation."""
        distance = np.linalg.norm(np.array(state) - np.array(goal))
        return -distance * 0.1  # small negative proportional to distance
```

### 5.3 사후 경험 재현(Hindsight Experience Replay, HER)

HER(Andrychowicz et al., 2017)은 암묵적 커리큘럼 전략입니다. 에이전트가 목표에 도달하지 못했을 때, HER은 실제로 도달한 상태를 "목표"로 소급하여 궤적에 레이블을 다시 붙입니다:

```
원래 궤적 (목표 = [5, 5], 실패):
  s0 → s1 → s2 → s3=[2,3]
  모든 보상 = -1 ([5,5]에 결코 도달하지 못함)

HER 재레이블링 (목표 = [2, 3], 성공):
  s0 → s1 → s2 → s3=[2,3]
  최종 보상 = 0 ("목표" [2,3]에 도달)

에이전트는 실패로부터 학습함:
"원하는 곳에 도달하지 못했지만, 어딘가에는 도달했다.
 저 장소에 안정적으로 도달하는 법을 배우자."
```

이는 자연스러운 커리큘럼을 제공합니다: 초기 궤적은 쉬운 상태에 도달하고, 에이전트가 개선될수록 점점 더 어려운 상태에 도달하게 됩니다.

---

## 6. 도메인 무작위화(Domain Randomization)

### 6.1 암묵적 커리큘럼으로서의 무작위화(Randomization as Implicit Curriculum)

도메인 무작위화(Domain Randomization, DR)는 훈련 중에 환경 파라미터를 무작위로 변화시킵니다. 전통적인 커리큘럼은 아니지만, 비슷한 목표를 달성합니다: 에이전트가 점점 더 도전적인 변형을 만나게 됩니다.

```python
class DomainRandomization:
    """Domain randomization for sim-to-real transfer.

    Randomizes environment parameters during training so the
    policy generalizes across parameter variations.
    """

    def __init__(self, param_ranges):
        """
        param_ranges: dict of parameter name → (low, high)
        Example: {'gravity': (8.0, 12.0), 'friction': (0.5, 1.5)}
        """
        self.param_ranges = param_ranges

    def sample_params(self):
        """Sample a random parameter configuration."""
        params = {}
        for name, (low, high) in self.param_ranges.items():
            params[name] = np.random.uniform(low, high)
        return params

    def get_curriculum_params(self, difficulty=None):
        """Optionally narrow randomization range for early training.

        At difficulty=0: narrow range (near defaults)
        At difficulty=1: full randomization range
        """
        if difficulty is None:
            return self.sample_params()

        params = {}
        for name, (low, high) in self.param_ranges.items():
            default = (low + high) / 2
            spread = (high - low) / 2 * difficulty
            params[name] = np.random.uniform(
                default - spread, default + spread
            )
        return params
```

### 6.2 자동 도메인 무작위화(Automatic Domain Randomization, ADR)

ADR(OpenAI, 2019)은 좁은 파라미터 범위에서 시작해 에이전트가 성공함에 따라 점차 범위를 넓혀갑니다:

```
ADR 루프:
  1. 좁은 범위로 시작: gravity ∈ [9.5, 10.5]
  2. 성능 임계값을 만족할 때까지 에이전트 훈련
  3. 범위 확장: gravity ∈ [9.0, 11.0]
  4. 범위가 실제 세계의 변동을 포괄할 때까지 반복

이로써 쉬운(좁은) 것에서 어려운(넓은) 무작위화로의
자동 커리큘럼이 만들어짐.
```

---

## 7. 실용적 가이드라인

### 7.1 커리큘럼 전략 선택하기

| 상황 | 권장 방법 |
|------|-----------|
| 명확한 난이도 순서 존재 | 수동 단계 커리큘럼 |
| 파라미터화된 환경 | 자기 주도 연속형 |
| 많은 변형을 가진 다중 과제 | 자동 경계 샘플링 |
| 목표 조건부 과제(goal-conditioned tasks) | HER (암묵적 커리큘럼) |
| 시뮬레이션→실제 전이(sim-to-real transfer) | 도메인 무작위화 + ADR |
| 매우 희소한 보상 | 보상 형성 + 커리큘럼 |

### 7.2 일반적인 함정

1. **파국적 망각(Catastrophic forgetting)**: 어려운 과제를 훈련할 때 쉬운 과제를 잊어버림
   - 해결책: 이후 단계에도 일부 쉬운 과제를 혼합 (재현 버퍼)

2. **조기 진급(Premature advancement)**: 기초를 마스터하기 전에 어려운 과제로 이동
   - 해결책: 보수적인 임계값, 최소 에피소드 요건

3. **단조롭지 않은 난이도(Non-monotonic difficulty)**: 과제 B가 더 어려워 보이지만 A와 다른 기술을 가르침
   - 해결책: 단일 선형 축이 아닌 다차원 난이도

4. **커리큘럼 과적합(Curriculum overfitting)**: 에이전트가 일반적인 기술을 배우는 대신 커리큘럼 구조를 활용
   - 해결책: 난이도 수준 내에서 무작위화, 보지 못한 구성에서 테스트

### 7.3 평가

항상 커리큘럼 단계가 아닌 **최종 목표 과제**에서 평가합니다:

```
훈련: 단계 1 → 2 → 3 → 4 → 5 (커리큘럼)

평가 (올바름):
  단계 5 (목표 과제)에서 100 에피소드 실행
  보고: 성공률, 평균 리턴, 수렴 속도

평가 (잘못됨):
  모든 단계에서 평균 성능
  (이것은 결과를 부풀림 — 쉬운 단계가 점수를 높임)
```

---

## 8. 연습문제

### 연습문제 1: GridWorld를 위한 단계 커리큘럼

장애물이 있는 20×20 GridWorld를 위한 4단계 커리큘럼을 설계하세요:
1. 단계 구성(격자 크기, 장애물 수, 목표 거리)을 정의한다
2. Q-학습으로 커리큘럼을 구현한다
3. 학습 곡선 비교: 커리큘럼 vs 가장 어려운 단계에서 직접 훈련
4. 두 방법에 대해 에피소드 대비 성공률을 플롯한다

### 연습문제 2: 자동 난이도 점수

자동 커리큘럼 생성을 구현하세요:
1. 파라미터화된 환경 생성 (예: 변하는 막대 길이와 질량을 가진 CartPole)
2. 쉬운 것에서 어려운 것까지 범위의 20개 과제 구성 정의
3. 성공률 기반 경계 샘플링 구현
4. 커리큘럼이 자연스러운 난이도 순서를 자동으로 발견함을 보임

### 연습문제 3: 보상 형성 커리큘럼

밀집→희소 보상 전환을 구현하세요:
1. 거리 기반 밀집 보상이 있는 목표 도달 환경 생성
2. 훈련 중 밀집에서 희소로의 선형 블렌딩 구현
3. 세 가지 방법 비교: 희소만, 밀집만, 커리큘럼 블렌드
4. 전환이 언제 일어나는지와 최종 성능에 미치는 영향을 분석

### 연습문제 4: 도메인 무작위화 연구

전이에 대한 무작위화 범위의 영향을 연구하세요:
1. 3개의 무작위화 가능한 파라미터가 있는 환경 생성
2. 좁은, 중간, 넓은 무작위화로 에이전트 훈련
3. 10개의 보지 못한 파라미터 구성에서 테스트
4. 트레이드오프 보임: 좁음 = 좋은 소스 성능, 넓음 = 더 좋은 전이

### 연습문제 5: HER 구현

사후 경험 재현(Hindsight Experience Replay)을 구현하세요:
1. 2D 목표 도달 환경 생성
2. 재현 버퍼(replay buffer)가 있는 DQN 에이전트 구현
3. HER 재레이블링 추가 (미래 전략: 에피소드 후반에서 달성된 목표 샘플링)
4. 다양한 난이도의 목표에서 HER 유무 학습 속도 비교

---

*15번 레슨 끝*
