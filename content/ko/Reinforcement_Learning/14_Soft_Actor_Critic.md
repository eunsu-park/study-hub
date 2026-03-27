# 14. Soft Actor-Critic (SAC)

**난이도: ⭐⭐⭐⭐ (고급)**

## 학습 목표
- 최대 엔트로피 강화학습(maximum entropy reinforcement learning) 이해
- 연속 행동 공간을 위한 SAC 알고리즘 구현
- 자동 온도(alpha) 튜닝 학습
- SAC와 PPO, TD3 비교
- 실용적인 연속 제어 태스크에 SAC 적용

---

## 목차

1. [최대 엔트로피 강화학습](#1-최대-엔트로피-강화학습)
2. [SAC 알고리즘](#2-sac-알고리즘)
3. [SAC 구현](#3-sac-구현)
4. [자동 온도 튜닝](#4-자동-온도-튜닝)
5. [SAC vs 다른 알고리즘](#5-sac-vs-다른-알고리즘)
6. [실용적 팁](#6-실용적-팁)
7. [연습 문제](#7-연습-문제)

---

## 1. 최대 엔트로피 강화학습

### 1.1 표준 강화학습 vs 최대 엔트로피 강화학습

```
┌─────────────────────────────────────────────────────────────────┐
│              Maximum Entropy Framework                            │
│                                                                 │
│  Standard RL objective:                                         │
│  π* = argmax_π  E [ Σ γ^t r_t ]                               │
│                  → maximize expected return only                 │
│                                                                 │
│  Maximum Entropy RL objective:                                  │
│  π* = argmax_π  E [ Σ γ^t (r_t + α H(π(·|s_t))) ]            │
│                  → maximize return + policy entropy             │
│                                                                 │
│  Where:                                                         │
│  • H(π(·|s)) = -E[log π(a|s)] is the policy entropy           │
│  • α (temperature) controls exploration-exploitation balance    │
│                                                                 │
│  Benefits of maximum entropy:                                   │
│  1. Encourages exploration (higher entropy = more random)       │
│  2. Captures multiple modes (doesn't collapse to one solution)  │
│  3. More robust to perturbations                                │
│  4. Better transfer and fine-tuning                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 소프트 벨만 방정식(Soft Bellman Equation)

```
┌─────────────────────────────────────────────────────────────────┐
│              Soft Value Functions                                 │
│                                                                 │
│  Soft state value:                                              │
│  V(s) = E_a~π [ Q(s,a) - α log π(a|s) ]                      │
│                                                                 │
│  Soft Q-value (Bellman equation):                               │
│  Q(s,a) = r(s,a) + γ E_s' [ V(s') ]                          │
│         = r(s,a) + γ E_s' [ E_a'~π [ Q(s',a') - α log π(a'|s') ] ]
│                                                                 │
│  Soft policy improvement:                                       │
│  π_new = argmin_π  D_KL( π(·|s) || exp(Q(s,·)/α) / Z(s) )   │
│                                                                 │
│  In practice: π outputs mean and std of Gaussian               │
│  a ~ tanh(μ + σ · ε),  ε ~ N(0, I)                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. SAC 알고리즘

### 2.1 SAC 구성 요소

```
┌─────────────────────────────────────────────────────────────────┐
│              SAC Architecture                                    │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │                  Actor (Policy)                    │           │
│  │  π_φ(a|s): Squashed Gaussian                     │           │
│  │  Input: state s                                   │           │
│  │  Output: μ(s), σ(s) → a = tanh(μ + σ·ε)         │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │              Twin Critics (Q1, Q2)                 │           │
│  │  Q_θ1(s, a), Q_θ2(s, a)                          │           │
│  │  Input: state s, action a                         │           │
│  │  Output: Q-value                                  │           │
│  │  → Use min(Q1, Q2) to prevent overestimation     │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │              Target Networks (Q1', Q2')            │           │
│  │  Soft update: θ' ← τθ + (1-τ)θ'                  │           │
│  │  Provides stable targets for critic training      │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │              Temperature (α)                       │           │
│  │  Controls entropy bonus                           │           │
│  │  Can be fixed or automatically tuned              │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 SAC 업데이트 규칙

```
┌─────────────────────────────────────────────────────────────────┐
│              SAC Training Steps                                  │
│                                                                 │
│  For each gradient step:                                        │
│                                                                 │
│  1. Sample batch (s, a, r, s', done) from replay buffer         │
│                                                                 │
│  2. Compute target:                                             │
│     a' ~ π_φ(·|s')                                              │
│     y = r + γ(1-done) × [min(Q'₁(s',a'), Q'₂(s',a'))          │
│                           - α log π_φ(a'|s')]                   │
│                                                                 │
│  3. Update Critics (minimize MSE):                              │
│     L_Q = E[(Q_θi(s,a) - y)²]  for i = 1, 2                  │
│                                                                 │
│  4. Update Actor (maximize):                                    │
│     ã ~ π_φ(·|s)  (reparameterization trick)                   │
│     L_π = E[α log π_φ(ã|s) - min(Q_θ1(s,ã), Q_θ2(s,ã))]    │
│                                                                 │
│  5. Update Temperature (if auto-tuning):                        │
│     L_α = E[-α (log π_φ(ã|s) + H_target)]                    │
│                                                                 │
│  6. Soft update target networks:                                │
│     θ'i ← τ θi + (1-τ) θ'i                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. SAC 구현

### 3.1 액터 네트워크(Squashed Gaussian Policy)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # 재파라미터화 트릭(reparameterization trick)으로 rsample을 사용하는 이유:
        # 확률적 샘플링 단계를 통해 그래디언트가 흐를 수 있게 한다 — z = μ + σ·ε로
        # 무작위성을 고정된 ε ~ N(0,I)로 이동시켜 actor 손실을 μ와 σ에 대해 미분할 수 있다
        z = normal.rsample()

        # tanh 스쿼싱(squashing)을 사용하는 이유: 무한한 가우시안 샘플을 (-1, 1) 범위로 매핑하여
        # 연속 제어에서 흔한 유계 행동 공간(예: 관절 토크)에 맞춘다;
        # 스쿼싱 없이는 정책이 액추에이터를 포화시키는 임의로 큰 행동을 생성할 수 있다
        action = torch.tanh(z)

        # 보정 항 -log(1 - action^2 + 1e-6)이 필요한 이유: 로그 확률의 변수 변환 공식에서는
        # tanh의 로그 절대 야코비안(log absolute Jacobian)을 빼야 한다; 이를 생략하면
        # log_prob이 부정확해져 엔트로피 추정과 actor 그래디언트가 잘못된다
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic_action(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)
```

### 3.2 크리틱 네트워크

```python
class TwinQCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)
```

### 3.3 SAC 에이전트

```python
import copy

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 auto_alpha=True, target_entropy=None):

        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha

        # Networks
        self.actor = GaussianActor(state_dim, action_dim, hidden_dim)
        # 트윈 크리틱(twin critics)을 사용하는 이유: min(Q1, Q2)를 타깃으로 사용하면
        # 단일 Q-네트워크 사용 시 발생하는 과대추정(overestimation) 편향을 방지한다 —
        # 과대추정은 actor가 잘못된 높은 Q-값을 활용하게 하여 정책 붕괴를 일으킨다;
        # 트윈 크리틱은 TD3에서 처음 효과가 입증되었다
        self.critic = TwinQCritic(state_dim, action_dim, hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)

        # 타깃 파라미터를 동결하는 이유: 타깃 네트워크는 critic 업데이트 중 안정적인
        # 회귀 타깃을 제공한다; 타깃으로 그래디언트가 흐르면 이동 타깃(moving-target) 문제가
        # 발생하여 학습을 불안정하게 만든다
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Temperature (alpha)
        if auto_alpha:
            # 기본 목표 엔트로피를 -action_dim으로 설정하는 이유: 정책이 거의 균일해지지 않으면서
            # 모든 고가치 행동을 탐험할 충분한 자유를 갖도록 "행동 차원당 1비트"를 설정하는 경험적 방법
            self.target_entropy = target_entropy or -action_dim
            # log_alpha를 최적화하는 이유 (alpha 직접 대신): log_alpha는 제약이 없고 그래디언트가
            # 잘 작동한다; 지수화(exponentiating)를 통해 클리핑이나 사영(projection) 없이 alpha > 0을 보장한다
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic_action(state)
            else:
                action, _ = self.actor.sample(state)
        return action.squeeze(0).numpy()

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch

        # --- Update Critics ---
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            # 트윈 타깃의 최솟값을 사용하는 이유: 벨만 타깃에서 Q1과 Q2의 최솟값을 취하면
            # 비관적 Q-추정값이 전파되어, critic이 자신의 (종종 낙관적인) 예측을 부트스트랩할 때
            # 누적되는 상향 편향을 상쇄한다
            q_target = torch.min(q1_target, q2_target)
            # 타깃에서 alpha * log_prob을 빼는 이유: 이것이 소프트 벨만 백업(soft Bellman backup)이다 —
            # 엔트로피 보너스는 정책의 로그 확률로 모든 보상을 증강하여, 정책이 확률적으로 유지되고
            # 모든 고가치 행동을 탐험하도록 장려한다
            target = rewards + self.gamma * (1 - dones) * \
                     (q_target - self.alpha * next_log_probs)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Temperature ---
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- Soft Update Target Networks ---
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            # 작은 τ(0.005)로 소프트 업데이트를 사용하는 이유: 온라인 가중치를 타깃으로 천천히
            # 블렌딩(θ' ← τθ + (1-τ)θ')하면 연속적인 그래디언트 스텝에서 타깃이 안정적으로 유지된다 —
            # 이는 매 스텝마다 actor가 업데이트되는 SAC, TD3 같은 연속 행동 알고리즘에 매우 중요하다;
            # 반면 DQN은 이산 행동 공간의 정책이 작은 타깃 변동에 덜 민감하기 때문에
            # 일반적으로 N 스텝마다 하드 업데이트(τ=1)를 사용한다
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'entropy': -log_probs.mean().item()
        }
```

### 3.4 리플레이 버퍼

```python
import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=1_000_000):
        self.capacity = capacity
        self.idx = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idxs]),
            torch.FloatTensor(self.actions[idxs]),
            torch.FloatTensor(self.rewards[idxs]),
            torch.FloatTensor(self.next_states[idxs]),
            torch.FloatTensor(self.dones[idxs])
        )

    def __len__(self):
        return self.size
```

### 3.5 학습 루프

```python
import gymnasium as gym

def train_sac(env_name='Pendulum-v1', total_steps=100_000,
              batch_size=256, start_steps=5000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = env.action_space.high[0]

    agent = SACAgent(state_dim, action_dim)
    buffer = ReplayBuffer(state_dim, action_dim)

    state, _ = env.reset()
    episode_reward = 0
    episode_rewards = []

    for step in range(total_steps):
        # Random actions for initial exploration
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state) * action_scale

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(state, action / action_scale, reward, next_state, float(terminated))

        state = next_state
        episode_reward += reward

        if done:
            episode_rewards.append(episode_reward)
            state, _ = env.reset()
            episode_reward = 0

        # Update after collecting enough data
        if step >= start_steps and len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            metrics = agent.update(batch)

            if step % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                print(f"Step {step}: avg_reward={avg_reward:.1f}, "
                      f"alpha={metrics['alpha']:.3f}, "
                      f"entropy={metrics['entropy']:.3f}")

    return agent, episode_rewards
```

---

## 4. 자동 온도 튜닝

### 4.1 자동 튜닝의 중요성

```
┌─────────────────────────────────────────────────────────────────┐
│              Temperature (α) Effect                              │
│                                                                 │
│  α too high:                     α too low:                     │
│  ┌─────────────────────┐         ┌─────────────────────┐       │
│  │ Entropy dominates    │         │ Return dominates     │       │
│  │ → nearly random     │         │ → premature          │       │
│  │ → slow learning     │         │   convergence        │       │
│  │ → poor performance  │         │ → poor exploration   │       │
│  └─────────────────────┘         └─────────────────────┘       │
│                                                                 │
│  Auto-tuning:                                                   │
│  ┌──────────────────────────────────────────────────┐           │
│  │ Constraint: H(π(·|s)) ≥ H_target               │           │
│  │ If entropy < target: increase α (explore more)   │           │
│  │ If entropy > target: decrease α (exploit more)   │           │
│  │ H_target = -dim(A) (heuristic for continuous)    │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                 │
│  Typical α trajectory during training:                          │
│  α                                                              │
│  │                                                              │
│  │  ╲                                                           │
│  │   ╲                                                          │
│  │    ╲___                                                      │
│  │        ╲____                                                 │
│  │             ╲_________                                       │
│  │                       ─────                                  │
│  └──────────────────────────────▶ steps                        │
│  (starts high for exploration, decreases as policy converges)   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 온도 손실 함수

```python
# Automatic temperature tuning objective
# L(α) = E_a~π [-α log π(a|s) - α H_target]
# = E_a~π [-α (log π(a|s) + H_target)]

# In the update step:
alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

# Intuition:
# When entropy (-log_probs) < H_target: log_probs + H_target > 0
# → gradient pushes log_alpha up → α increases → more entropy encouraged
# When entropy > H_target: log_probs + H_target < 0
# → gradient pushes log_alpha down → α decreases
```

---

## 5. SAC vs 다른 알고리즘

### 5.1 비교 표

```
┌───────────────┬──────────┬──────────┬──────────┬───────────────┐
│               │ SAC      │ PPO      │ TD3      │ DDPG          │
├───────────────┼──────────┼──────────┼──────────┼───────────────┤
│ Policy type   │ Stochas. │ Stochas. │ Determin.│ Deterministic │
│ On/Off policy │ Off      │ On       │ Off      │ Off           │
│ Action space  │ Contin.  │ Both     │ Contin.  │ Continuous    │
│ Entropy reg.  │ Yes      │ Yes      │ No       │ No            │
│ Twin critics  │ Yes      │ No       │ Yes      │ No            │
│ Sample eff.   │ High     │ Low      │ High     │ Medium        │
│ Stability     │ High     │ High     │ Medium   │ Low           │
│ Hyperparams   │ Few      │ Many     │ Medium   │ Many          │
│ Auto-tuning   │ α tuning │ No       │ No       │ No            │
└───────────────┴──────────┴──────────┴──────────┴───────────────┘
```

### 5.2 SAC 사용 시기

```
Use SAC when:
✓ Continuous action spaces (robotics, control)
✓ Sample efficiency matters (real-world, expensive simulation)
✓ You want stable training with minimal tuning
✓ Multi-modal optimal policies exist

Use PPO instead when:
✓ Discrete action spaces
✓ On-policy learning is preferred
✓ Simulation is cheap (can generate many samples)
✓ Distributed training (PPO scales better)

Use TD3 instead when:
✓ Deterministic policy is preferred
✓ Simpler implementation needed
✓ No entropy regularization wanted
```

---

## 6. 실용적 팁

### 6.1 하이퍼파라미터

```
┌────────────────────────┬──────────────┬──────────────────────────┐
│ Hyperparameter         │ Default      │ Notes                    │
├────────────────────────┼──────────────┼──────────────────────────┤
│ Learning rate          │ 3e-4         │ Same for actor & critic  │
│ Discount (γ)           │ 0.99         │ Standard                 │
│ Soft update (τ)        │ 0.005        │ Slow target updates      │
│ Batch size             │ 256          │ Larger is more stable    │
│ Buffer size            │ 1M           │ Large replay buffer      │
│ Hidden layers          │ (256, 256)   │ 2 layers is standard     │
│ Start steps            │ 5000-10000   │ Random exploration first │
│ Target entropy         │ -dim(A)      │ Heuristic, works well    │
│ Gradient steps/env step│ 1            │ 1:1 ratio is standard    │
└────────────────────────┴──────────────┴──────────────────────────┘
```

### 6.2 일반적인 문제와 해결 방법

```
Issue: Training instability / Q-values diverge
→ Check reward scale (normalize if needed)
→ Reduce learning rate
→ Increase batch size

Issue: Low entropy (premature convergence)
→ Enable auto alpha tuning
→ Increase initial alpha
→ Check action bounds

Issue: Slow learning
→ Increase start_steps for better initial exploration
→ Try larger networks
→ Check reward shaping

Issue: Action values saturating at bounds
→ Ensure proper action scaling
→ Check tanh squashing implementation
→ Verify log_prob correction term
```

---

## 7. 연습 문제

### 연습 1: Pendulum에서 SAC
`Pendulum-v1`에서 SAC를 학습하고 학습 곡선을 그리세요.

```python
# Train SAC
agent, rewards = train_sac('Pendulum-v1', total_steps=50_000)

# Plot learning curve
import matplotlib.pyplot as plt
window = 10
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(smoothed)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('SAC on Pendulum-v1')
plt.show()

# Expected: converges to ~-200 within 20K steps
```

### 연습 2: SAC vs PPO 비교
연속 제어 태스크에서 SAC와 PPO를 모두 학습하고 샘플 효율성을 비교하세요.

```python
# Use HalfCheetah-v4 or Hopper-v4
# Plot reward vs environment steps for both algorithms
# Expected: SAC reaches same performance in ~5x fewer environment steps
# But PPO may have lower wall-clock time per step
```

### 연습 3: 절제 연구(Ablation Study)
다음 변형으로 SAC를 실행하고 비교하세요:
1. 고정 alpha = 0.2 (자동 튜닝 없음)
2. 자동 alpha (기본값)
3. 엔트로피 항 없음 (alpha = 0, TD3와 유사)
4. 단일 Q-네트워크 (트윈 크리틱 없음)

```python
# Expected findings:
# - Auto alpha > fixed alpha (adapts to task)
# - With entropy > without (better exploration)
# - Twin critics > single (prevents overestimation)
```

### 연습 4: 커스텀 환경
커스텀 연속 제어 태스크에 SAC를 적용하세요.

```python
# Example: reaching task
import gymnasium as gym
from gymnasium import spaces

class ReachingEnv(gym.Env):
    """2D reaching task: move arm tip to target."""

    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,))
        self.target = np.array([0.5, 0.5])

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.pos = np.random.uniform(-1, 1, size=2)
        return np.concatenate([self.pos, self.target]), {}

    def step(self, action):
        self.pos = np.clip(self.pos + action * 0.1, -1, 1)
        dist = np.linalg.norm(self.pos - self.target)
        reward = -dist
        done = dist < 0.05
        return np.concatenate([self.pos, self.target]), reward, done, False, {}
```

---

## 요약

```
┌─────────────────────────────────────────────────────────────────┐
│              SAC Key Components                                  │
│                                                                 │
│  1. Maximum entropy objective: reward + α × entropy             │
│  2. Squashed Gaussian policy: a = tanh(μ + σε)                 │
│  3. Twin Q-critics: min(Q1, Q2) prevents overestimation        │
│  4. Automatic temperature: α adapts to maintain target entropy  │
│  5. Off-policy: high sample efficiency via replay buffer        │
│                                                                 │
│  SAC is the go-to algorithm for continuous control tasks         │
│  due to its stability, sample efficiency, and minimal tuning.   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 참고 문헌

- [SAC Paper (v1)](https://arxiv.org/abs/1801.01290) — Haarnoja et al. 2018
- [SAC Paper (v2, auto-alpha)](https://arxiv.org/abs/1812.05905) — Haarnoja et al. 2018
- [Spinning Up: SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [CleanRL SAC Implementation](https://docs.cleanrl.dev/rl-algorithms/sac/)

---

[다음: 커리큘럼 학습](./15_Curriculum_Learning.md)
