# 11. 다중 에이전트 강화학습 (Multi-Agent RL)

**난이도: ⭐⭐⭐⭐ (고급)**

## 학습 목표
- 다중 에이전트 환경의 특성 이해
- 협력, 경쟁, 혼합 시나리오 구분
- 중앙집중/분산 학습 패러다임
- MARL 알고리즘: IQL, QMIX, MAPPO

---

## 1. 다중 에이전트 환경 개요

### 1.1 단일 vs 다중 에이전트

| 특성 | 단일 에이전트 | 다중 에이전트 |
|------|-------------|--------------|
| 환경 | 정적 (에이전트 관점) | 동적 (다른 에이전트) |
| 보상 | 개인 보상 | 개인/팀/글로벌 |
| 최적성 | 최적 정책 존재 | 내쉬 균형 추구 |
| 학습 | 정상성 가정 | 비정상성 (이동 타겟) |

### 1.2 환경 유형

```
┌─────────────────────────────────────────────────────┐
│                    MARL 환경 유형                    │
├──────────────┬──────────────┬──────────────────────┤
│    협력       │     경쟁      │        혼합          │
│ (Cooperative) │(Competitive) │      (Mixed)         │
├──────────────┼──────────────┼──────────────────────┤
│ 팀 스포츠    │ 제로섬 게임   │ 일반 섬 게임         │
│ 로봇 협동    │ 1v1 대전     │ 협력적 경쟁          │
│ 스웜 로봇    │ 가위바위보    │ 사회적 딜레마         │
└──────────────┴──────────────┴──────────────────────┘
```

---

## 2. MARL의 도전 과제

### 2.1 비정상성 (Non-stationarity)

다른 에이전트도 학습하므로 환경이 계속 변합니다.

```python
# 에이전트 i의 관점에서
# 환경 전이: P(s'|s, a_i, a_{-i})
# 다른 에이전트의 정책이 변하면 전이 확률도 변함

class NonStationaryEnv:
    def step(self, actions):
        # actions: 모든 에이전트의 행동
        joint_action = tuple(actions)
        next_state = self.transition(self.state, joint_action)
        rewards = self.reward_function(self.state, joint_action, next_state)
        return next_state, rewards
```

### 2.2 신용 할당 (Credit Assignment)

팀 보상에서 개인 기여도를 파악하기 어렵습니다.

### 2.3 확장성 (Scalability)

에이전트 수가 늘면 상태-행동 공간이 기하급수적으로 증가합니다.

---

## 3. 학습 패러다임

### 3.1 중앙집중 학습, 분산 실행 (CTDE)

**Centralized Training, Decentralized Execution**

```
훈련 시: 글로벌 정보 접근 가능
실행 시: 로컬 관측만 사용

┌─────────────────────────────────┐
│       Central Critic            │  (훈련 시)
│   (글로벌 상태, 모든 행동 접근) │
└─────────────┬───────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│Actor 1│ │Actor 2│ │Actor 3│  (실행 시)
│(로컬) │ │(로컬) │ │(로컬) │
└───────┘ └───────┘ └───────┘
```

### 3.2 완전 분산 (Independent Learning)

각 에이전트가 독립적으로 학습합니다.

```python
class IndependentQLearning:
    """각 에이전트가 독립적으로 Q-learning"""
    def __init__(self, n_agents, state_dim, action_dim):
        self.agents = [
            QLearningAgent(state_dim, action_dim)
            for _ in range(n_agents)
        ]

    def choose_actions(self, observations):
        return [
            agent.choose_action(obs)
            for agent, obs in zip(self.agents, observations)
        ]

    def update(self, observations, actions, rewards, next_observations, dones):
        for i, agent in enumerate(self.agents):
            agent.update(
                observations[i], actions[i],
                rewards[i], next_observations[i], dones[i]
            )
```

---

## 4. IQL (Independent Q-Learning)

### 4.1 개념

각 에이전트가 다른 에이전트를 환경의 일부로 취급합니다.

```python
import torch
import torch.nn as nn
import numpy as np

class IQLAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.1):
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

    def choose_action(self, obs):
        # 엡실론-그리디(epsilon-greedy): epsilon 확률로 무작위 탐험하여 아직 시도하지 않은
        # 행동을 발견하고, 그 외에는 현재 Q 추정값을 이용해 탐욕적으로 행동한다
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(obs))
            return q_values.argmax().item()

    def update(self, obs, action, reward, next_obs, done):
        obs_tensor = torch.FloatTensor(obs)
        next_obs_tensor = torch.FloatTensor(next_obs)

        current_q = self.q_network(obs_tensor)[action]

        with torch.no_grad():
            if done:
                # 종료 후에는 미래 보상이 없으므로 목표값은 즉각 보상만이다;
                # 종료 상태 너머로 부트스트래핑하면 가치 추정이 오염된다
                target_q = reward
            else:
                # 벨만 최적성(Bellman optimality): Q(s,a) = r + γ max_a' Q(s',a') —
                # 목표값이 탐욕적인 다음 행동을 사용하므로 이는 오프-폴리시(off-policy) Q-학습이다
                target_q = reward + self.gamma * self.q_network(next_obs_tensor).max()

        # TD 오차의 제곱 손실(squared TD error): 이를 최소화하면 current_q가
        # 고정 레이블로 취급되는 벨만 목표값에 수렴한다 (위에서 no_grad 적용)
        loss = (current_q - target_q) ** 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class IQLSystem:
    def __init__(self, n_agents, obs_dims, action_dims):
        self.agents = [
            IQLAgent(obs_dims[i], action_dims[i])
            for i in range(n_agents)
        ]

    def step(self, env):
        observations = env.get_observations()
        actions = [
            agent.choose_action(obs)
            for agent, obs in zip(self.agents, observations)
        ]

        next_obs, rewards, dones, _ = env.step(actions)

        for i, agent in enumerate(self.agents):
            agent.update(
                observations[i], actions[i],
                rewards[i], next_obs[i], dones[i]
            )

        return rewards, dones
```

### 4.2 IQL의 한계

- 다른 에이전트 정책 변화로 환경이 비정상적
- 협력 학습에서 수렴이 어려울 수 있음

---

## 5. VDN과 QMIX (가치 분해)

### 5.1 VDN (Value Decomposition Networks)

팀 Q값을 개인 Q값의 합으로 분해:

$$Q_{tot}(s, \mathbf{a}) = \sum_{i=1}^{n} Q_i(o_i, a_i)$$

```python
class VDN:
    def __init__(self, n_agents, obs_dim, action_dim):
        self.agents = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )
            for _ in range(n_agents)
        ])

    def get_q_values(self, observations):
        """각 에이전트의 Q값"""
        return [
            agent(obs)
            for agent, obs in zip(self.agents, observations)
        ]

    def get_total_q(self, observations, actions):
        """팀 Q값 = 개인 Q값의 합"""
        q_values = self.get_q_values(observations)
        individual_q = [
            q[a] for q, a in zip(q_values, actions)
        ]
        return sum(individual_q)
```

### 5.2 QMIX

더 일반적인 분해를 허용합니다. 단조성 조건만 만족:

$$\frac{\partial Q_{tot}}{\partial Q_i} \geq 0$$

```python
class QMIXMixer(nn.Module):
    """QMIX 믹싱 네트워크"""
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents

        # 하이퍼네트워크(hypernetworks)는 글로벌 상태에 조건화된 믹싱 가중치를 생성한다 —
        # 이를 통해 믹싱 함수가 현재 상황에 적응하면서도 전체 아키텍처가
        # 엔드-투-엔드(end-to-end)로 미분 가능하게 유지된다
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        self.embed_dim = embed_dim

    def forward(self, agent_qs, state):
        """
        agent_qs: [batch, n_agents] - 각 에이전트의 Q값
        state: [batch, state_dim] - 글로벌 상태
        """
        batch_size = agent_qs.size(0)
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)

        # abs()로 비음수 가중치를 강제하는 것이 단조성(monotonicity) 조건의 핵심이다:
        # 모든 i에 대해 ∂Q_tot/∂Q_i ≥ 0 — 이를 통해 Q_tot에서의 탐욕적 행동 선택이
        # 각 에이전트의 로컬 탐욕적 선택과 일치함을 보장한다 (IGM 속성)
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)

        # 편향(bias)은 단조성 조건에 영향을 주지 않으므로 제약 없이 사용한다 —
        # 비음수 조건이 필요한 것은 개별 Q값에 곱해지는 가중치뿐이다
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # 믹싱: bmm은 배치 행렬 곱셈(batch matrix multiplication)을 수행하여
        # 개별 Q값들을 상태 조건화된 단조 함수를 통해 Q_tot으로 결합한다
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.squeeze(-1).squeeze(-1)
```

---

## 6. MADDPG (Multi-Agent DDPG)

### 6.1 개념

CTDE 패러다임 + Actor-Critic

- **Actor**: 로컬 관측만 사용
- **Critic**: 모든 에이전트의 관측과 행동 사용

```python
class MADDPGAgent:
    def __init__(self, agent_id, obs_dims, action_dims, n_agents):
        self.agent_id = agent_id
        self.n_agents = n_agents

        # Actor는 로컬 관측만 사용한다 — 이는 실제 배치 시 다른 에이전트의 상태를
        # 관측할 수 없는 분산 실행(decentralized execution) 요건을 반영한다
        self.actor = nn.Sequential(
            nn.Linear(obs_dims[agent_id], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dims[agent_id]),
            # Tanh는 연속 행동을 [-1, 1]로 제한한다 — 환경이 실제 물리적 범위로
            # 스케일링하는 일반적인 관례이다
            nn.Tanh()
        )

        # 중앙집중 Critic은 모든 에이전트의 관측과 행동을 연결(concatenate)한다 —
        # 훈련 중 개별 에이전트의 정책이 변해도 결합 상태-행동 공간은 정상적(stationary)이므로
        # 비정상성(non-stationarity) 문제가 해결된다
        total_obs_dim = sum(obs_dims)
        total_action_dim = sum(action_dims)
        self.critic = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def act(self, obs, noise_scale=0.1):
        """로컬 관측으로 행동 결정"""
        action = self.actor(torch.FloatTensor(obs))
        # 훈련 시 가우시안 탐험 노이즈(Gaussian exploration noise)를 추가한다;
        # clamp로 노이즈 추가 후에도 행동이 유효 범위 [-1, 1] 내에 유지된다
        noise = torch.randn_like(action) * noise_scale
        return (action + noise).clamp(-1, 1)

    def get_q_value(self, all_obs, all_actions):
        """글로벌 정보로 Q값 계산"""
        # 모든 관측과 행동을 단일 벡터로 연결 — 중앙집중 훈련 시에만 가능하며,
        # 실행 시에는 불가능하다
        x = torch.cat([*all_obs, *all_actions], dim=-1)
        return self.critic(x)
```

---

## 7. MAPPO (Multi-Agent PPO)

### 7.1 구조

PPO를 다중 에이전트로 확장:

```python
class MAPPOAgent:
    def __init__(self, obs_dim, action_dim, state_dim):
        # Actor (로컬 관측)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic (글로벌 상태)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def get_action(self, obs):
        probs = self.actor(torch.FloatTensor(obs))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def get_value(self, state):
        return self.critic(torch.FloatTensor(state))


class MAPPO:
    def __init__(self, n_agents, obs_dims, action_dims, state_dim):
        self.agents = [
            MAPPOAgent(obs_dims[i], action_dims[i], state_dim)
            for i in range(n_agents)
        ]
        self.n_agents = n_agents

    def collect_rollout(self, env, n_steps):
        """모든 에이전트의 경험 수집"""
        rollouts = [{
            'obs': [], 'actions': [], 'rewards': [],
            'values': [], 'log_probs': [], 'dones': []
        } for _ in range(self.n_agents)]

        obs = env.reset()
        state = env.get_state()

        for _ in range(n_steps):
            actions = []
            for i, agent in enumerate(self.agents):
                action, log_prob = agent.get_action(obs[i])
                value = agent.get_value(state)

                actions.append(action)
                rollouts[i]['obs'].append(obs[i])
                rollouts[i]['actions'].append(action)
                rollouts[i]['values'].append(value.item())
                rollouts[i]['log_probs'].append(log_prob)

            next_obs, rewards, dones, _ = env.step(actions)
            next_state = env.get_state()

            for i in range(self.n_agents):
                rollouts[i]['rewards'].append(rewards[i])
                rollouts[i]['dones'].append(dones[i])

            obs = next_obs
            state = next_state

        return rollouts
```

---

## 8. Self-Play

### 8.1 개념

에이전트가 자기 자신의 복사본과 대전하며 학습합니다.

```python
class SelfPlayTrainer:
    def __init__(self, agent_class, env):
        self.current_agent = agent_class()
        self.opponent_pool = []
        self.env = env

    def train_episode(self):
        # 80% 확률로 현재 자신이 아닌 과거 버전과 대전한다 — 에이전트가 현재 전략에
        # 과적합(overfitting)되는 것을 방지하고, 훈련 전반에 걸쳐 다양한
        # 상대 스타일에 대한 견고성을 유지한다
        if len(self.opponent_pool) > 0 and np.random.random() < 0.8:
            opponent = np.random.choice(self.opponent_pool)
        else:
            opponent = self.current_agent  # 자기 자신

        # 대전
        state = self.env.reset()
        done = False

        while not done:
            # 현재 에이전트 행동
            action1 = self.current_agent.choose_action(state[0])
            # 상대 행동
            action2 = opponent.choose_action(state[1])

            next_state, rewards, done, _ = self.env.step([action1, action2])

            # 현재 에이전트만 업데이트 — 상대는 동결된 스냅샷(frozen snapshot)이다;
            # 양쪽을 동시에 훈련하면 학습 신호가 불안정해진다
            self.current_agent.update(
                state[0], action1, rewards[0], next_state[0], done
            )

            state = next_state

    def save_snapshot(self):
        """현재 에이전트를 상대 풀에 추가"""
        # deepcopy로 현재 파라미터를 동결시킨다 — 이후 훈련이 과거 스냅샷에
        # 영향을 주지 않아 상대 풀의 역사적 다양성이 보존된다
        snapshot = copy.deepcopy(self.current_agent)
        self.opponent_pool.append(snapshot)

        # 풀 크기를 제한하여 메모리를 절약하고 상대가 적절히 경쟁력을 유지하게 한다
        # (너무 오래된 에이전트는 유용한 훈련 신호를 제공하기 어렵다)
        if len(self.opponent_pool) > 10:
            self.opponent_pool.pop(0)
```

---

## 9. MARL 환경 예시

### 9.1 PettingZoo

```python
from pettingzoo.mpe import simple_spread_v2

def run_pettingzoo():
    env = simple_spread_v2.parallel_env()
    observations = env.reset()

    while env.agents:
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)

    env.close()
```

---

## 요약

| 알고리즘 | 패러다임 | 협력/경쟁 | 특징 |
|---------|---------|----------|------|
| IQL | 분산 | 둘 다 | 간단, 비정상성 문제 |
| VDN | CTDE | 협력 | 합 분해 |
| QMIX | CTDE | 협력 | 단조적 분해 |
| MADDPG | CTDE | 둘 다 | 연속 행동 |
| MAPPO | CTDE | 둘 다 | PPO 확장 |

---

## 연습 문제

### 연습 1: 비정상성(Non-Stationarity) 분석

다중 에이전트 환경에서 단일 에이전트 RL 기법이 실패하는 이유를 분석하세요.

1. MARL에서 비정상성 문제를 자신의 언어로 설명하세요: 다른 에이전트들이 학습할 때 마르코프 속성(Markov property)이 왜 깨지나요?
2. 에이전트 i가 다음 벨만 업데이트로 Q-학습(Q-learning)을 사용한다고 가정하세요:
   `Q(s, a_i) ← r + γ max Q(s', a_i)`
   다른 에이전트 j가 동시에 정책을 변경할 때 이 업데이트에 무슨 일이 일어나나요?
3. IQL이 수렴에 완전히 실패하는 구체적인 예시(예: 2인 조정 게임)를 드세요.
4. CTDE 패러다임이 학습 중 비정상성을 어떻게 해결하는지 설명하세요. 실행이 여전히 분산화될 수 있는 이유는 무엇인가요?
5. CTDE가 비정상성을 완전히 해결하나요, 아니면 일부 잔여 비정상성이 남아있나요? 근거를 제시하세요.

### 연습 2: VDN vs QMIX 표현 능력

VDN과 QMIX의 표현력을 비교하세요.

1. VDN은 Q_tot = Σ Q_i를 가정합니다. 이 덧셈 가정이 위반되는 간단한 2-에이전트, 2-행동 협력 게임을 구성하세요(보수 행렬을 작성하세요) — 즉, VDN에서 개별 탐욕적 선택으로는 최적 공동 행동을 복원할 수 없는 경우를 만드세요.
2. QMIX 단조성(monotonicity) 조건 ∂Q_tot/∂Q_i ≥ 0을 설명하세요. 이것이 Q_tot에서의 argmax가 개별 argmax 연산과 일치함을 보장하는 이유는 무엇인가요(IGM 속성)?
3. VDN이 단조성 조건을 만족함을 보이세요 — ∂(ΣQ_i)/∂Q_j = 1 ≥ 0임을 대수적으로 증명하세요.
4. QMIX는 표현할 수 있지만 VDN은 표현할 수 없는 협력 게임을 구성하세요. 어떤 구조적 속성이 QMIX로 표현 가능하게 만드나요?
5. QMIX도 표현할 수 없는 협력 게임의 부류는 무엇인가요? 어떤 알고리즘이 필요할까요?

### 연습 3: 간단한 CTDE 시스템 구현

Section 3.1의 CTDE 패턴을 활용하여 두 에이전트 협력 시스템을 만드세요.

1. 두 에이전트가 동시에 반대쪽 모서리에 도달해야 보상 +1을 받는 (그렇지 않으면 0) 간단한 2D 격자 환경을 만드세요. 각 에이전트는 자신의 위치만 관측합니다.
2. 로컬 관측을 입력으로 받는 두 개의 액터(에이전트당 하나씩)를 구현하세요.
3. 두 에이전트의 관측을 연결(concatenate)하여 입력으로 받는 중앙집중 크리틱(centralized critic)을 구현하세요.
4. 중앙집중 크리틱의 Q값으로 액터 경사를 계산하는 간단한 MADDPG 방식의 업데이트로 학습시키세요.
5. 같은 환경에서 두 개의 독립 Q-학습 에이전트와 비교하세요. 어느 쪽이 더 빠르게 수렴하나요? 어느 쪽이 더 높은 보상을 달성하나요?

```python
# 환경 뼈대 코드
class CoopGridEnv:
    def __init__(self, size=5):
        self.size = size
        self.n_agents = 2
        # 에이전트 0 목표: 우상단 모서리, 에이전트 1 목표: 좌하단 모서리
        self.goals = [(size-1, size-1), (0, 0)]

    def reset(self):
        # 두 에이전트의 무작위 시작 위치
        ...

    def step(self, actions):
        # actions: 각 에이전트의 (dx, dy) 리스트
        ...
```

### 연습 4: 자기 대전(Self-Play) 커리큘럼 설계

경쟁 게임을 위한 자기 대전 학습 커리큘럼을 설계하세요.

1. Section 8.1의 `SelfPlayTrainer` 뼈대를 사용하여 간단한 1대1 게임(예: 틱택토 또는 단순화된 퐁)을 설정하세요.
2. 현재 정책을 주기적으로 동결하여 상대 풀(opponent pool)에 추가하는 `save_snapshot()` 메서드를 구현하세요.
3. 두 가지 상대 샘플링 전략을 실험하세요:
   - **균등(Uniform)**: 과거 스냅샷을 동일한 확률로 샘플링
   - **우선순위(Prioritized)**: 최근 스냅샷을 더 자주 샘플링 (예: 최신성으로 가중치 부여)
4. 각 전략으로 10,000 에피소드 학습하세요. 500 에피소드마다 고정된 무작위 상대에 대한 승률을 추적하세요.
5. 분석: 어느 샘플링 전략이 더 빠른 향상을 가져오나요? 상대 풀의 크기가 결과에 영향을 미치나요?

### 연습 5: PettingZoo 협력 태스크

PettingZoo를 활용하여 협력 태스크에 MAPPO를 적용하세요.

1. PettingZoo를 설치하세요: `pip install pettingzoo[mpe]`
2. `simple_spread_v2` 환경을 설정하세요 (3개의 에이전트가 협력하여 3개의 랜드마크를 차지해야 함).
3. Section 7.1의 MAPPO 아키텍처를 구현하세요: 각 에이전트는 로컬 액터를 가지고, 각 에이전트의 크리틱은 글로벌 상태(모든 에이전트와 랜드마크의 위치를 연결한 것)를 받습니다.
4. 50,000 타임스텝 동안 학습하고 에피소드별 팀 보상을 기록하세요.
5. 세 개의 독립 PPO 에이전트(각자의 액터와 크리틱을 가지며 로컬 관측만 사용)와 비교하세요. 다음을 측정하세요:
   - 최종 평균 팀 보상
   - 최대 가능 보상의 50%에 도달하는 데 걸린 타임스텝 수
   - 정성적 행동: 에이전트들이 분산하는 법을 학습했나요?

---

## 다음 단계

- [12_Practical_RL_Project.md](./12_Practical_RL_Project.md) - 실전 프로젝트
