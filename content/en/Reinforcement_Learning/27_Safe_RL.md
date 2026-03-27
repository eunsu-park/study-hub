[Previous: Imitation Learning](./26_Imitation_Learning.md)

---

# 27. Safe Reinforcement Learning

## Learning Objectives

After completing this lesson, you will be able to:

1. Formulate constrained MDPs for safe RL with explicit safety constraints
2. Implement Lagrangian methods for constraint satisfaction during RL training
3. Build safe exploration strategies that avoid catastrophic actions
4. Understand risk-sensitive RL and CVaR optimization for worst-case guarantees
5. Apply safety layers and shielding for real-world deployment

---

## Table of Contents

1. [Why Safe RL?](#1-why-safe-rl)
2. [Constrained MDPs (CMDPs)](#2-constrained-mdps-cmdps)
3. [Lagrangian Methods](#3-lagrangian-methods)
4. [Safe Exploration](#4-safe-exploration)
5. [Risk-Sensitive RL](#5-risk-sensitive-rl)
6. [Safety Layers and Shielding](#6-safety-layers-and-shielding)
7. [Evaluation and Verification](#7-evaluation-and-verification)
8. [Exercises](#8-exercises)

---

## 1. Why Safe RL?

### 1.1 Safety Is Not Optional

```
RL in the real world has consequences:

Autonomous driving:
  Unconstrained RL: "Found a faster route by running red lights!"
  Safe RL: "Maximize speed SUBJECT TO traffic law compliance"

Medical treatment:
  Unconstrained RL: "Optimal dose is 10x normal" (reward hacking)
  Safe RL: "Optimize outcomes SUBJECT TO dose safety limits"

Robotics:
  Unconstrained RL: "Fastest path goes through the wall"
  Safe RL: "Navigate SUBJECT TO collision avoidance"

Key insight: We need CONSTRAINTS, not just rewards.
```

### 1.2 Types of Safety

```
Safety taxonomy:

1. Constraint satisfaction
   "Never exceed velocity limit"
   Mathematical: E[Σ c(s,a)] ≤ d (cost constraint)

2. Safe exploration
   "Don't visit dangerous states while learning"
   Even during training, avoid catastrophic actions

3. Robustness
   "Work correctly under perturbations"
   Handle model uncertainty, adversarial disturbances

4. Alignment
   "Do what the human actually wants"
   Avoid reward hacking, maintain intent
```

---

## 2. Constrained MDPs (CMDPs)

### 2.1 CMDP Formulation

```
Standard MDP: max_π E[Σ γᵗ r(sₜ, aₜ)]

CMDP adds K constraints:
  max_π  E[Σ γᵗ r(sₜ, aₜ)]           (reward objective)
  s.t.   E[Σ γᵗ cₖ(sₜ, aₜ)] ≤ dₖ    for k = 1, ..., K

Where:
  cₖ(s, a) = cost function for constraint k
  dₖ = budget for constraint k

Example (autonomous driving):
  Reward: progress toward destination
  Constraint 1: E[collision_cost] ≤ 0   (no collisions)
  Constraint 2: E[speed_violation] ≤ 0  (obey speed limits)
  Constraint 3: E[lane_violation] ≤ 0.1 (mostly stay in lane)
```

```text
┌─────────────────────────────────────────────────────────────────┐
│            Constrained MDP (CMDP) Structure                     │
│                                                                 │
│   Feasible Policy Space                                         │
│   ┌───────────────────────────────────────────────────────┐     │
│   │                                                       │     │
│   │   All policies π                                      │     │
│   │                                                       │     │
│   │   ┌───────────────────────────────────┐               │     │
│   │   │ Constraint-satisfying policies    │               │     │
│   │   │  C_1(π) ≤ d_1                    │               │     │
│   │   │  C_2(π) ≤ d_2                    │               │     │
│   │   │  ...                              │               │     │
│   │   │                                   │               │     │
│   │   │          ★ π* (optimal safe)      │               │     │
│   │   └───────────────────────────────────┘               │     │
│   │                                                       │     │
│   │        ✗ π_unconstrained (maximizes reward only)      │     │
│   │          (may violate constraints)                     │     │
│   └───────────────────────────────────────────────────────┘     │
│                                                                 │
│   Goal: find π* = argmax J(π) inside the feasible region        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 CMDP in Code

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

## 3. Lagrangian Methods

### 3.1 The Lagrangian Approach

```
Convert constrained optimization to unconstrained using Lagrange multipliers:

Original: max_π J(π)  s.t. C_k(π) ≤ d_k

Lagrangian: max_π min_λ≥0  J(π) - Σ_k λ_k (C_k(π) - d_k)

Alternating optimization:
1. Fix λ, optimize π (standard RL with modified reward)
   r_modified(s,a) = r(s,a) - Σ_k λ_k c_k(s,a)

2. Fix π, optimize λ (gradient ascent on multipliers)
   λ_k ← max(0, λ_k + α_λ (C_k(π) - d_k))

Intuition: λ_k increases when constraint k is violated,
making the agent pay more attention to avoiding that cost.
```

```text
┌─────────────────────────────────────────────────────────────────┐
│              Lagrangian Safe RL Training Loop                   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Collect rollout: (s, a, r, c_1, ..., c_K, s', done)     │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Compute modified reward:                                 │    │
│  │   r̃(s,a) = r(s,a) - λ_1·c_1(s,a) - ... - λ_K·c_K(s,a) │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                     │
│             ┌─────────────┴──────────────┐                      │
│             ▼                            ▼                      │
│  ┌─────────────────────┐    ┌─────────────────────────────┐     │
│  │ Update policy π      │    │ Update multipliers λ_k      │     │
│  │ (PPO/SAC on r̃)      │    │ λ_k ← max(0, λ_k + α(C_k-d_k)) │  │
│  └─────────────────────┘    │                             │     │
│                             │  C_k > d_k → λ_k ↑ (penalize)   │
│                             │  C_k < d_k → λ_k ↓ (relax)      │
│                             └─────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Lagrangian PPO Implementation

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

## 4. Safe Exploration

### 4.1 Safety During Learning

```
Even during training, some states/actions are unacceptable:

Hard constraints (must never violate):
  - Robot joint limits
  - Collision avoidance in critical systems
  - Medication dosage limits

Soft constraints (occasional violation OK):
  - Efficiency targets
  - Comfort metrics
  - Non-critical performance bounds

Approaches:
1. Action masking: Remove unsafe actions before selection
2. Safety layer: Project actions onto safe set
3. Barrier functions: Repel agent from unsafe states
4. Teacher intervention: Expert takes over when risk is high
```

### 4.2 Action Safety Layer

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

## 5. Risk-Sensitive RL

### 5.1 Beyond Expected Value

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

### 5.2 CVaR Policy Optimization

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

## 6. Safety Layers and Shielding

### 6.1 Runtime Safety Monitors

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

## 7. Evaluation and Verification

### 7.1 Safety Metrics

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

## 8. Exercises

### Exercise 1: Constrained CartPole

Implement CMDP for constPole with safety constraints:
1. Wrap CartPole with a constraint: pole angle must stay within +/- 10 degrees
2. Implement Lagrangian PPO with cost critic
3. Compare unconstrained PPO vs Lagrangian PPO
4. Show the tradeoff: constrained agent has lower reward but fewer violations
5. Plot constraint satisfaction over training

### Exercise 2: Safety Layer Implementation

Build and test a safety layer:
1. Create a 2D navigation environment with obstacles
2. Train a standard RL policy (may collide with obstacles)
3. Add a learned safety layer that projects actions to safe set
4. Compare collision rate: with and without safety layer
5. Measure the performance cost of safety (reward reduction)

### Exercise 3: CVaR Optimization

Implement CVaR policy gradient:
1. Create a stochastic environment with rare catastrophic events
2. Train risk-neutral agent and measure catastrophic event frequency
3. Implement CVaR optimization (alpha = 0.1)
4. Compare: mean return, worst-case return, catastrophe frequency
5. Plot the Pareto frontier: mean return vs safety metrics

### Exercise 4: Lagrange Multiplier Dynamics

Study Lagrange multiplier behavior:
1. CMDP with 3 constraints of varying difficulty
2. Train Lagrangian PPO and log lambda values over time
3. Show that lambdas increase for violated constraints
4. Vary constraint budgets and observe lambda adaptation
5. Implement adaptive lambda learning rate based on constraint satisfaction

### Exercise 5: Safe Exploration with Teacher

Build a safe exploration system:
1. Create an environment with "danger zones" (high cost regions)
2. Implement a simple teacher policy that avoids danger zones
3. During training, teacher overrides when agent enters danger
4. Gradually reduce teacher intervention as agent learns
5. Compare: number of constraint violations during training

---

*End of Lesson 27*
