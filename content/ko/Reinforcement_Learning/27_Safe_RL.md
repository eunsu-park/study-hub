[이전: Imitation Learning](./26_Imitation_Learning.md)

---

# 27. 안전 강화학습

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 명시적 안전 제약이 있는 안전 RL을 위한 제약 MDP 정식화
2. RL 학습 중 제약 만족을 위한 라그랑주 방법 구현
3. 치명적인 행동을 피하는 안전 탐색 전략 구축
4. 최악의 경우 보장을 위한 위험 민감 RL과 CVaR 최적화 이해
5. 실제 배포를 위한 안전 레이어와 차폐 적용

---

## 목차

1. [왜 안전 RL인가?](#1-왜-안전-rl인가)
2. [제약 MDP (CMDP)](#2-제약-mdp-cmdp)
3. [라그랑주 방법](#3-라그랑주-방법)
4. [안전 탐색](#4-안전-탐색)
5. [위험 민감 RL](#5-위험-민감-rl)
6. [안전 레이어와 차폐](#6-안전-레이어와-차폐)
7. [평가 및 검증](#7-평가-및-검증)
8. [연습문제](#8-연습문제)

---

## 1. 왜 안전 RL인가?

### 1.1 안전은 선택이 아닙니다

```
실제 세계에서의 RL에는 결과가 따릅니다:

자율주행:
  제약 없는 RL: "빨간 신호를 무시하면 더 빠른 경로를 찾았습니다!"
  안전 RL: "교통법규 준수를 조건으로 속도를 최대화"

의료 치료:
  제약 없는 RL: "최적 투여량은 정상의 10배" (보상 해킹)
  안전 RL: "투여 안전 한계를 조건으로 결과 최적화"

로보틱스:
  제약 없는 RL: "가장 빠른 경로는 벽을 통과합니다"
  안전 RL: "충돌 회피를 조건으로 이동"

핵심 통찰: 보상뿐만 아니라 제약이 필요합니다.
```

### 1.2 안전의 유형

```
안전 분류:

1. 제약 만족
   "속도 제한을 초과하지 않기"
   수학적: E[Σ c(s,a)] ≤ d (비용 제약)

2. 안전 탐색
   "학습 중 위험한 상태를 방문하지 않기"
   학습 중에도 치명적인 행동 회피

3. 견고성
   "교란 하에서도 올바르게 작동"
   모델 불확실성, 적대적 교란 처리

4. 정렬
   "인간이 실제로 원하는 것을 하기"
   보상 해킹 회피, 의도 유지
```

---

## 2. 제약 MDP (CMDP)

### 2.1 CMDP 정식화

```
표준 MDP: max_π E[Σ γᵗ r(sₜ, aₜ)]

CMDP는 K개의 제약을 추가합니다:
  max_π  E[Σ γᵗ r(sₜ, aₜ)]           (보상 목적)
  s.t.   E[Σ γᵗ cₖ(sₜ, aₜ)] ≤ dₖ    k = 1, ..., K에 대해

여기서:
  cₖ(s, a) = 제약 k에 대한 비용 함수
  dₖ = 제약 k에 대한 예산

예시 (자율주행):
  보상: 목적지까지의 진행
  제약 1: E[충돌_비용] ≤ 0   (충돌 없음)
  제약 2: E[속도_위반] ≤ 0  (속도 제한 준수)
  제약 3: E[차선_위반] ≤ 0.1 (대부분 차선 유지)
```

```text
┌─────────────────────────────────────────────────────────────────┐
│            제약 MDP (CMDP) 구조                                  │
│                                                                 │
│   실현 가능한 정책 공간                                           │
│   ┌───────────────────────────────────────────────────────┐     │
│   │                                                       │     │
│   │   모든 정책 π                                          │     │
│   │                                                       │     │
│   │   ┌───────────────────────────────────┐               │     │
│   │   │ 제약을 만족하는 정책               │               │     │
│   │   │  C_1(π) ≤ d_1                    │               │     │
│   │   │  C_2(π) ≤ d_2                    │               │     │
│   │   │  ...                              │               │     │
│   │   │                                   │               │     │
│   │   │          ★ π* (최적 안전 정책)    │               │     │
│   │   └───────────────────────────────────┘               │     │
│   │                                                       │     │
│   │        ✗ π_비제약 (보상만 최대화)                      │     │
│   │          (제약 위반 가능)                               │     │
│   └───────────────────────────────────────────────────────┘     │
│                                                                 │
│   목표: 실현 가능 영역 내에서 J(π)를 최대화하는 π* 탐색          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 코드로 표현한 CMDP

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedEnvironment:
    """Environment wrapper that provides cost signals."""

    def __init__(self, base_env, cost_functions):
        self.env = base_env
        self.cost_functions = cost_functions  # List of cost functions
        self.n_constraints = len(cost_functions)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute costs for each constraint
        costs = []
        for cost_fn in self.cost_functions:
            cost = cost_fn(obs, action, info)
            costs.append(cost)

        info['costs'] = np.array(costs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Example cost functions for robotics
def collision_cost(obs, action, info):
    """Cost = 1 if collision detected."""
    return float(info.get('collision', False))

def velocity_cost(obs, action, info, max_vel=2.0):
    """Cost = max(0, velocity - max_vel)."""
    velocity = np.linalg.norm(obs['velocity'])
    return max(0, velocity - max_vel)

def torque_cost(obs, action, info, max_torque=10.0):
    """Cost for exceeding torque limits."""
    return max(0, np.abs(action).max() - max_torque)
```

---

## 3. 라그랑주 방법

### 3.1 라그랑주 접근법

```
라그랑주 승수를 사용하여 제약 최적화를 비제약으로 변환:

원래: max_π J(π)  s.t. C_k(π) ≤ d_k

라그랑지안: max_π min_λ≥0  J(π) - Σ_k λ_k (C_k(π) - d_k)

교대 최적화:
1. λ 고정, π 최적화 (수정된 보상으로 표준 RL)
   r_modified(s,a) = r(s,a) - Σ_k λ_k c_k(s,a)

2. π 고정, λ 최적화 (승수에 대한 경사 상승)
   λ_k ← max(0, λ_k + α_λ (C_k(π) - d_k))

직관: λ_k는 제약 k가 위반될 때 증가하여,
에이전트가 해당 비용 회피에 더 많은 주의를 기울이게 합니다.
```

```text
┌─────────────────────────────────────────────────────────────────┐
│              라그랑주 안전 RL 학습 루프                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 롤아웃 수집: (s, a, r, c_1, ..., c_K, s', done)          │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 수정된 보상 계산:                                         │    │
│  │   r̃(s,a) = r(s,a) - λ_1·c_1(s,a) - ... - λ_K·c_K(s,a) │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                     │
│             ┌─────────────┴──────────────┐                      │
│             ▼                            ▼                      │
│  ┌─────────────────────┐    ┌─────────────────────────────┐     │
│  │ 정책 π 업데이트       │    │ 승수 λ_k 업데이트            │     │
│  │ (r̃로 PPO/SAC)       │    │ λ_k ← max(0, λ_k + α(C_k-d_k)) │  │
│  └─────────────────────┘    │                             │     │
│                             │  C_k > d_k → λ_k ↑ (처벌 강화)  │
│                             │  C_k < d_k → λ_k ↓ (처벌 완화)  │
│                             └─────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 라그랑주 PPO 구현

```python
class LagrangianPPO:
    """PPO with Lagrangian constraint handling."""

    def __init__(self, state_dim, action_dim, n_constraints,
                 cost_limits, hidden_dim=256, lr=3e-4,
                 lambda_lr=5e-3, gamma=0.99, clip_ratio=0.2):
        self.n_constraints = n_constraints
        self.cost_limits = torch.FloatTensor(cost_limits)  # d_k thresholds
        self.gamma = gamma
        self.clip_ratio = clip_ratio

        # Policy
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

        # Value function for reward
        self.reward_critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Value functions for each cost (separate critics)
        self.cost_critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            ) for _ in range(n_constraints)
        ])

        # Lagrange multipliers (log scale for positivity)
        self.log_lambdas = nn.Parameter(torch.zeros(n_constraints))

        self.policy_optimizer = torch.optim.Adam(
            list(self.actor.parameters()) +
            [self.action_mean.weight, self.action_mean.bias, self.action_log_std],
            lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.reward_critic.parameters()) +
            list(self.cost_critics.parameters()),
            lr=lr
        )
        self.lambda_optimizer = torch.optim.Adam(
            [self.log_lambdas], lr=lambda_lr
        )

    @property
    def lambdas(self):
        return self.log_lambdas.exp()

    def get_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        features = self.actor(state_t)
        mean = self.action_mean(features)
        std = self.action_log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.squeeze(0).detach().numpy(), log_prob.item()

    def update(self, rollout_data):
        """Update policy, critics, and Lagrange multipliers."""
        states = torch.FloatTensor(rollout_data['states'])
        actions = torch.FloatTensor(rollout_data['actions'])
        rewards = torch.FloatTensor(rollout_data['rewards'])
        costs = torch.FloatTensor(rollout_data['costs'])  # (T, n_constraints)
        old_log_probs = torch.FloatTensor(rollout_data['log_probs'])
        dones = torch.FloatTensor(rollout_data['dones'])

        # Compute advantages for reward
        reward_values = self.reward_critic(states).squeeze(-1)
        reward_advantages = self._compute_gae(rewards, reward_values, dones)

        # Compute advantages for each cost
        cost_advantages = []
        for k in range(self.n_constraints):
            cost_values = self.cost_critics[k](states).squeeze(-1)
            cost_adv = self._compute_gae(costs[:, k], cost_values, dones)
            cost_advantages.append(cost_adv)
        cost_advantages = torch.stack(cost_advantages, dim=-1)  # (T, K)

        # Combined advantage: reward - Σ λ_k * cost_k
        lambdas = self.lambdas.detach()
        combined_advantages = reward_advantages - (cost_advantages * lambdas).sum(-1)

        # PPO policy update with combined advantages
        features = self.actor(states)
        mean = self.action_mean(features)
        std = self.action_log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(-1)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * combined_advantages.detach()
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio,
                            1 + self.clip_ratio) * combined_advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Critic updates
        reward_pred = self.reward_critic(states).squeeze(-1)
        reward_returns = reward_advantages + reward_values.detach()
        critic_loss = F.mse_loss(reward_pred, reward_returns)

        for k in range(self.n_constraints):
            cost_pred = self.cost_critics[k](states).squeeze(-1)
            cost_returns = cost_advantages[:, k] + \
                self.cost_critics[k](states).squeeze(-1).detach()
            critic_loss += F.mse_loss(cost_pred, cost_returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Lagrange multiplier update
        # Increase λ_k if constraint k is violated (C_k > d_k)
        with torch.no_grad():
            episode_costs = costs.sum(dim=0) / (1 - dones).sum()

        lambda_loss = -(self.log_lambdas * (episode_costs - self.cost_limits)).sum()

        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'lambdas': self.lambdas.detach().numpy(),
            'episode_costs': episode_costs.numpy(),
        }

    def _compute_gae(self, rewards, values, dones, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = torch.zeros(T)
        last_gae = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * (1 - dones[t]) * next_value - values[t]
            advantages[t] = last_gae = delta + \
                self.gamma * gae_lambda * (1 - dones[t]) * last_gae

        return advantages
```

---

## 4. 안전 탐색

### 4.1 학습 중 안전

```
학습 중에도 일부 상태/행동은 허용할 수 없습니다:

엄격한 제약 (절대 위반 불가):
  - 로봇 관절 한계
  - 중요 시스템에서의 충돌 회피
  - 약물 투여량 한계

유연한 제약 (가끔 위반 가능):
  - 효율성 목표
  - 편안함 지표
  - 비중요 성능 경계

접근법:
1. 행동 마스킹: 선택 전에 안전하지 않은 행동 제거
2. 안전 레이어: 안전한 집합으로 행동 투영
3. 배리어 함수: 안전하지 않은 상태에서 에이전트를 밀어냄
4. 교사 개입: 위험이 높을 때 전문가가 인수
```

### 4.2 행동 안전 레이어

```python
class SafetyLayer(nn.Module):
    """Project RL actions onto the nearest safe action."""

    def __init__(self, state_dim, action_dim, constraint_model=None):
        super().__init__()
        # Learn constraint model: c(s, a) ≤ 0 means safe
        self.constraint_model = constraint_model or nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, proposed_action, correction_lr=0.1, n_steps=10):
        """Project proposed action onto safe set."""
        safe_action = proposed_action.clone().detach().requires_grad_(True)

        for _ in range(n_steps):
            constraint = self.constraint_model(
                torch.cat([state, safe_action], dim=-1)
            )

            if (constraint <= 0).all():
                break  # Already safe

            # Gradient descent to minimize constraint violation
            violation = F.relu(constraint).sum()
            grad = torch.autograd.grad(violation, safe_action)[0]
            safe_action = (safe_action - correction_lr * grad).detach()
            safe_action.requires_grad_(True)

            # Clip to action bounds
            safe_action = safe_action.clamp(-1, 1).detach().requires_grad_(True)

        return safe_action.detach()


class ControlBarrierFunction:
    """Control Barrier Function for continuous-time safety."""

    def __init__(self, barrier_fn, alpha=1.0):
        self.barrier_fn = barrier_fn  # h(x) > 0 means safe
        self.alpha = alpha

    def is_safe(self, state):
        return self.barrier_fn(state) > 0

    def safe_action(self, state, proposed_action, dynamics_fn):
        """Modify action to satisfy CBF constraint:
        dh/dt + α·h(x) ≥ 0
        """
        h = self.barrier_fn(state)

        if h > 0.5:  # Safely away from boundary
            return proposed_action

        # Need to ensure h doesn't decrease too fast
        # Solve QP: min ||a - a_proposed||² s.t. Lf_h + Lg_h·a + α·h ≥ 0
        # Simplified: project onto half-space
        return proposed_action  # Placeholder for QP solution
```

---

## 5. 위험 민감 RL

### 5.1 기대값을 넘어서

```python
def risk_sensitive_objectives():
    """Common risk-sensitive RL objectives."""
    objectives = {
        'Expected Value': {
            'formula': 'max E[R]',
            'risk': 'neutral',
            'use_case': 'When average performance matters most',
        },
        'CVaR (Conditional Value at Risk)': {
            'formula': 'max E[R | R ≤ F⁻¹(α)]',
            'risk': 'averse',
            'use_case': 'Healthcare, finance (worst-case focus)',
        },
        'Mean-Variance': {
            'formula': 'max E[R] - λ·Var[R]',
            'risk': 'averse',
            'use_case': 'Balanced risk-return tradeoff',
        },
        'Worst-Case (Minimax)': {
            'formula': 'max min_ξ E[R | ξ]',
            'risk': 'extremely averse',
            'use_case': 'Safety-critical systems',
        },
        'Entropic Risk': {
            'formula': 'max (1/β)·log E[exp(β·R)]',
            'risk': 'adjustable (β < 0: averse, β > 0: seeking)',
            'use_case': 'Tunable risk sensitivity',
        },
    }
    return objectives
```

### 5.2 CVaR 정책 최적화

```python
class CVaRPolicyGradient:
    """Policy gradient optimizing CVaR instead of expected return."""

    def __init__(self, policy, alpha=0.25, lr=3e-4):
        self.policy = policy
        self.alpha = alpha  # CVaR level (lower = more conservative)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def update(self, episodes):
        """Update policy to maximize CVaR_α."""
        returns = torch.FloatTensor([ep['return'] for ep in episodes])

        # CVaR: average of bottom α returns
        n_bottom = max(1, int(self.alpha * len(returns)))
        sorted_returns, sorted_indices = returns.sort()
        bottom_returns = sorted_returns[:n_bottom]
        bottom_indices = sorted_indices[:n_bottom]

        # Policy gradient only on bottom-α episodes
        policy_loss = 0
        for idx in bottom_indices:
            ep = episodes[idx.item()]
            log_probs = torch.stack(ep['log_probs'])
            advantages = bottom_returns.mean() - ep['return']  # Simplified

            policy_loss -= (log_probs * advantages).sum()

        policy_loss /= n_bottom

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return {
            'cvar': bottom_returns.mean().item(),
            'mean_return': returns.mean().item(),
            'worst_return': returns.min().item(),
        }
```

---

## 6. 안전 레이어와 차폐

### 6.1 런타임 안전 모니터

```python
class SafetyMonitor:
    """Runtime safety monitor that can override RL policy."""

    def __init__(self, constraint_checks, fallback_policy):
        self.checks = constraint_checks
        self.fallback = fallback_policy
        self.violation_count = 0
        self.total_steps = 0

    def safe_action(self, state, proposed_action):
        """Check safety and override if necessary."""
        self.total_steps += 1

        for check_name, check_fn in self.checks.items():
            if not check_fn(state, proposed_action):
                self.violation_count += 1
                safe_action = self.fallback(state)
                return safe_action, {'overridden': True, 'reason': check_name}

        return proposed_action, {'overridden': False}

    def safety_rate(self):
        return 1 - (self.violation_count / max(self.total_steps, 1))


# Example usage
monitor = SafetyMonitor(
    constraint_checks={
        'joint_limits': lambda s, a: np.all(np.abs(a) < 0.95),
        'velocity_limit': lambda s, a: np.linalg.norm(s['vel']) < 2.0,
        'workspace_bounds': lambda s, a: np.all(np.abs(s['pos']) < 1.5),
    },
    fallback_policy=lambda s: np.zeros(action_dim)  # Stop
)
```

---

## 7. 평가 및 검증

### 7.1 안전 지표

```python
def evaluate_safe_agent(agent, env, n_episodes=100, cost_threshold=0.0):
    """Comprehensive safety evaluation."""
    returns = []
    costs_per_ep = []
    violations = []
    max_single_cost = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_return = 0
        ep_cost = 0
        ep_violations = 0
        ep_max_cost = 0
        done = False

        while not done:
            action = agent.get_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            ep_return += reward

            cost = info.get('cost', 0)
            ep_cost += cost
            ep_max_cost = max(ep_max_cost, cost)
            if cost > cost_threshold:
                ep_violations += 1

            done = terminated or truncated

        returns.append(ep_return)
        costs_per_ep.append(ep_cost)
        violations.append(ep_violations)
        max_single_cost.append(ep_max_cost)

    print("=== Safety Evaluation ===")
    print(f"Return:         {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"Episode cost:   {np.mean(costs_per_ep):.3f} +/- {np.std(costs_per_ep):.3f}")
    print(f"Violation rate: {np.mean([v > 0 for v in violations])*100:.1f}%")
    print(f"Avg violations: {np.mean(violations):.2f} per episode")
    print(f"Max single cost:{np.max(max_single_cost):.4f}")
    print(f"Cost at 95th %: {np.percentile(costs_per_ep, 95):.4f}")

    return {
        'mean_return': np.mean(returns),
        'mean_cost': np.mean(costs_per_ep),
        'violation_rate': np.mean([v > 0 for v in violations]),
        'cost_cvar_95': np.mean(sorted(costs_per_ep)[-5:]),
    }
```

---

## 8. 연습문제

### 연습문제 1: 제약 CartPole

안전 제약이 있는 CMDP를 CartPole에 구현하세요:
1. CartPole을 제약으로 래핑: 막대 각도가 +/- 10도 이내여야 함
2. 비용 크리틱을 사용한 라그랑주 PPO 구현
3. 제약 없는 PPO vs 라그랑주 PPO 비교
4. 트레이드오프 시연: 제약된 에이전트는 보상이 낮지만 위반이 적음
5. 학습 과정에서 제약 만족 그래프

### 연습문제 2: 안전 레이어 구현

안전 레이어를 구축하고 테스트하세요:
1. 장애물이 있는 2D 내비게이션 환경 생성
2. 표준 RL 정책 학습 (장애물과 충돌 가능)
3. 행동을 안전한 집합으로 투영하는 학습된 안전 레이어 추가
4. 충돌률 비교: 안전 레이어 유무
5. 안전의 성능 비용 측정 (보상 감소)

### 연습문제 3: CVaR 최적화

CVaR 정책 그래디언트를 구현하세요:
1. 드문 치명적 사건이 있는 확률적 환경 생성
2. 위험 중립 에이전트를 학습시키고 치명적 사건 빈도 측정
3. CVaR 최적화 구현 (alpha = 0.1)
4. 비교: 평균 리턴, 최악의 경우 리턴, 재해 빈도
5. 파레토 프론티어 그래프: 평균 리턴 vs 안전 지표

### 연습문제 4: 라그랑주 승수 역학

라그랑주 승수 행동을 연구하세요:
1. 다양한 난이도의 3개 제약이 있는 CMDP
2. 라그랑주 PPO를 학습시키고 시간에 따른 lambda 값 기록
3. 위반된 제약에 대해 lambda가 증가하는 것을 시연
4. 제약 예산을 변경하고 lambda 적응 관찰
5. 제약 만족에 기반한 적응형 lambda 학습률 구현

### 연습문제 5: 교사와 함께하는 안전 탐색

안전 탐색 시스템을 구축하세요:
1. "위험 지대" (높은 비용 영역)가 있는 환경 생성
2. 위험 지대를 피하는 간단한 교사 정책 구현
3. 학습 중 에이전트가 위험에 진입하면 교사가 인수
4. 에이전트가 학습함에 따라 점진적으로 교사 개입 감소
5. 비교: 학습 중 제약 위반 횟수

---

*27강 끝*
