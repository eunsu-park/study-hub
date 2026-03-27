[Previous: Hierarchical RL](./16_Hierarchical_RL.md)

---

# 18. Distributional Reinforcement Learning

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the distributional perspective on reinforcement learning and why modeling full return distributions matters
2. Implement the C51 algorithm with categorical distribution projection
3. Build Quantile Regression DQN (QR-DQN) and understand quantile Huber loss
4. Implement Implicit Quantile Networks (IQN) for risk-sensitive control
5. Compare distributional methods on standard benchmarks and analyze their advantages

---

## Table of Contents

1. [Why Distributional RL?](#1-why-distributional-rl)
2. [The C51 Algorithm](#2-the-c51-algorithm)
3. [Quantile Regression DQN (QR-DQN)](#3-quantile-regression-dqn-qr-dqn)
4. [Implicit Quantile Networks (IQN)](#4-implicit-quantile-networks-iqn)
5. [Distributional Policy Gradients](#5-distributional-policy-gradients)
6. [Risk-Sensitive Control](#6-risk-sensitive-control)
7. [Practical Implementation Guide](#7-practical-implementation-guide)
8. [Exercises](#8-exercises)

---

## 1. Why Distributional RL?

### 1.1 Beyond Expected Values

Traditional RL algorithms learn the *expected* return from each state-action pair. But expected values throw away valuable information about uncertainty.

```
Traditional Q-Learning:
  Q(s, a) = E[R₁ + γR₂ + γ²R₃ + ...]  ← single scalar

Distributional RL:
  Z(s, a) = R₁ + γR₂ + γ²R₃ + ...      ← full random variable

  Example: Two slot machines
  Machine A: always pays $5           → E[R] = $5
  Machine B: pays $0 or $10 with 50/50 → E[R] = $5

  Same expected value, very different distributions!
  A risk-averse agent should prefer Machine A.
  A risk-seeking agent might prefer Machine B.
```

### 1.2 The Return Distribution

Instead of learning Q(s,a) = E[Z(s,a)], we learn the full distribution of Z(s,a).

```
                    Traditional RL          Distributional RL
                    ┌─────────────┐         ┌─────────────────┐
State-Action ──────▶│  Q(s,a) = 5 │         │  Z(s,a):        │
  (s, a)           └─────────────┘         │  ▓▓░░▓▓░░▓▓    │
                    Single number           │  Probability     │
                                            │  distribution    │
                                            └─────────────────┘
```

### 1.3 Distributional Bellman Equation

The standard Bellman equation operates on expectations:

```
Q(s, a) = E[R + γ max_a' Q(s', a')]
```

The distributional Bellman equation operates on distributions:

```
Z(s, a) =ᵈ R + γ Z(s', a*)    where a* = argmax_a' E[Z(s', a')]
                                 =ᵈ means "equal in distribution"
```

This is a contraction in the Wasserstein metric (p-Wasserstein distance):

```python
import numpy as np

def wasserstein_distance(p, q, support_p, support_q):
    """Compute 1-Wasserstein distance between two discrete distributions."""
    # CDF-based computation
    all_points = np.sort(np.unique(np.concatenate([support_p, support_q])))
    cdf_p = np.zeros_like(all_points, dtype=float)
    cdf_q = np.zeros_like(all_points, dtype=float)

    for i, x in enumerate(all_points):
        cdf_p[i] = np.sum(p[support_p <= x])
        cdf_q[i] = np.sum(q[support_q <= x])

    # W₁ = integral |CDF_p - CDF_q|
    dx = np.diff(all_points, prepend=all_points[0])
    return np.sum(np.abs(cdf_p - cdf_q) * dx)
```

### 1.4 Why Distributions Help

| Benefit | Explanation |
|---------|-------------|
| **Richer signal** | Distribution provides more gradient information than a scalar |
| **Auxiliary learning** | Predicting distributions is a harder, more informative task |
| **Risk sensitivity** | Can optimize for CVaR, variance, or other risk measures |
| **Better exploration** | Epistemic uncertainty can drive exploration |
| **Multimodality** | Captures multiple possible outcomes (e.g., win/lose) |

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_return_distributions():
    """Show why distributional RL is more informative."""
    np.random.seed(42)

    # Two actions with same expected return but different distributions
    returns_safe = np.random.normal(5.0, 0.5, 10000)
    returns_risky = np.concatenate([
        np.random.normal(2.0, 0.5, 5000),
        np.random.normal(8.0, 0.5, 5000)
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(returns_safe, bins=50, alpha=0.7, color='blue', density=True)
    axes[0].axvline(np.mean(returns_safe), color='red', linestyle='--',
                    label=f'E[Z] = {np.mean(returns_safe):.2f}')
    axes[0].set_title('Safe Action: Z(s, a_safe)')
    axes[0].legend()

    axes[1].hist(returns_risky, bins=50, alpha=0.7, color='orange', density=True)
    axes[1].axvline(np.mean(returns_risky), color='red', linestyle='--',
                    label=f'E[Z] = {np.mean(returns_risky):.2f}')
    axes[1].set_title('Risky Action: Z(s, a_risky)')
    axes[1].legend()

    plt.suptitle('Same E[Q] ~ 5.0, Very Different Distributions')
    plt.tight_layout()
    plt.savefig('distributional_comparison.png', dpi=150)
    plt.show()

# visualize_return_distributions()
```

---

## 2. The C51 Algorithm

### 2.1 Categorical Distribution Representation

C51 (Categorical with 51 atoms) represents the return distribution as a categorical distribution over a fixed set of evenly-spaced "atoms":

```
Atoms:  z₁=V_MIN, z₂, z₃, ..., z_N=V_MAX    (N=51 by default)

             p(zᵢ | s, a) = probability of return being zᵢ
                  │
                  ▼
  Probability
    ▓
    ▓ ▓
    ▓ ▓ ▓
    ▓ ▓ ▓ ▓
    ▓ ▓ ▓ ▓ ▓ ▓
  ──────────────── Return value
  V_MIN         V_MAX

  Q(s,a) = Σᵢ zᵢ · p(zᵢ | s, a)   ← Expected value recovered
```

### 2.2 Network Architecture

```
            ┌──────────────┐
  State s ──▶  Shared CNN  │
            │  /FC Layers  │──────┐
            └──────────────┘      │
                                  ▼
                    ┌─────────────────────────┐
                    │  Per-action output heads │
                    │                         │
                    │  Action 0: [p₁...p₅₁]  │  (softmax)
                    │  Action 1: [p₁...p₅₁]  │  (softmax)
                    │  ...                    │
                    │  Action K: [p₁...p₅₁]  │  (softmax)
                    └─────────────────────────┘
```

### 2.3 Projection Step

When computing the target distribution, the Bellman update shifts and scales the atoms. Since the resulting atoms may not align with our fixed support, we need to *project* back:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class C51Network(nn.Module):
    """C51 distributional DQN network."""

    def __init__(self, state_dim, action_dim, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Support: fixed set of atoms
        self.register_buffer(
            'support', torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Network layers
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim * n_atoms)

    def forward(self, state):
        """Return probability distributions for each action."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape to (batch, actions, atoms) and apply softmax over atoms
        x = x.view(-1, self.action_dim, self.n_atoms)
        probs = F.softmax(x, dim=-1)
        return probs

    def get_q_values(self, state):
        """Compute Q-values as expected values of distributions."""
        probs = self.forward(state)
        q_values = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return q_values


def c51_projection(next_probs, rewards, dones, support, gamma, v_min, v_max, n_atoms):
    """
    Project the Bellman-updated distribution onto the fixed support.

    Args:
        next_probs: (batch, n_atoms) - target distribution
        rewards: (batch,) - immediate rewards
        dones: (batch,) - terminal flags
        support: (n_atoms,) - atom locations
        gamma: discount factor
    Returns:
        projected: (batch, n_atoms) - projected distribution
    """
    batch_size = rewards.shape[0]
    delta_z = (v_max - v_min) / (n_atoms - 1)

    # Compute Tz = r + γz for each atom
    Tz = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * support.unsqueeze(0)
    Tz = Tz.clamp(v_min, v_max)

    # Compute projection indices
    b = (Tz - v_min) / delta_z  # fractional index
    l = b.floor().long()         # lower index
    u = (l + 1).clamp(max=n_atoms - 1)  # upper index
    l = l.clamp(min=0)

    # Distribute probability
    projected = torch.zeros(batch_size, n_atoms, device=rewards.device)

    # Lower bound contribution
    projected.scatter_add_(1, l, next_probs * (u.float() - b))
    # Upper bound contribution
    projected.scatter_add_(1, u, next_probs * (b - l.float()))

    return projected
```

### 2.4 C51 Training Loop

```python
class C51Agent:
    """Complete C51 agent with experience replay."""

    def __init__(self, state_dim, action_dim, n_atoms=51,
                 v_min=-10, v_max=10, lr=2.5e-4, gamma=0.99):
        self.gamma = gamma
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.network = C51Network(state_dim, action_dim, n_atoms, v_min, v_max)
        self.target_network = C51Network(state_dim, action_dim, n_atoms, v_min, v_max)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.support = self.network.support

    def select_action(self, state, epsilon=0.0):
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.network.get_q_values(state_t)
            return q_values.argmax(dim=-1).item()

    def train_step(self, batch):
        """One training step on a batch from replay buffer."""
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current distribution: p(s, a)
        current_probs = self.network(states)
        current_probs = current_probs[
            torch.arange(len(states)), actions
        ]

        with torch.no_grad():
            # Next state: select action using online network (Double DQN style)
            next_q = self.network.get_q_values(next_states)
            next_actions = next_q.argmax(dim=-1)

            # Get target distribution for selected action
            next_probs = self.target_network(next_states)
            next_probs = next_probs[
                torch.arange(len(next_states)), next_actions
            ]

            # Project target distribution
            target_probs = c51_projection(
                next_probs, rewards, dones,
                self.support, self.gamma,
                self.v_min, self.v_max, self.n_atoms
            )

        # Cross-entropy loss between projected target and current
        loss = -(target_probs * torch.log(current_probs + 1e-8)).sum(dim=-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()
```

### 2.5 C51 Results and Analysis

```
Performance on Atari Games (from Bellemare et al., 2017):

Game           | DQN     | C51     | Improvement
─────────────────────────────────────────────────
Asterix        | 8,503   | 406,211 | 47.8x
Breakout       | 401     | 748     | 1.9x
Pong           | 20.9    | 20.9    | 1.0x
Seaquest       | 5,286   | 266,434 | 50.4x
Space Invaders | 1,976   | 5,747   | 2.9x

Key insight: Biggest gains in stochastic/multi-modal environments.
```

---

## 3. Quantile Regression DQN (QR-DQN)

### 3.1 From Fixed Atoms to Fixed Probabilities

C51 fixes the atom locations and learns probabilities. QR-DQN does the opposite: fixes the probabilities (uniform quantiles) and learns the atom locations.

```
C51:
  Fixed:   z₁, z₂, ..., z₅₁  (atom locations)
  Learned: p₁, p₂, ..., p₅₁  (probabilities)

QR-DQN:
  Fixed:   τ₁=1/2N, τ₂=3/2N, ..., τ_N=(2N-1)/2N  (quantile midpoints)
  Learned: θ₁, θ₂, ..., θ_N  (quantile values)

  Advantage: No V_MIN/V_MAX hyperparameters needed!
```

### 3.2 Quantile Huber Loss

QR-DQN uses the quantile Huber loss, which combines quantile regression with Huber loss for stability:

```python
def quantile_huber_loss(predictions, targets, taus, kappa=1.0):
    """
    Compute quantile Huber loss.

    Args:
        predictions: (batch, N) - predicted quantile values
        targets: (batch, N) - target quantile values
        taus: (N,) - quantile midpoints
        kappa: Huber loss threshold
    Returns:
        loss: scalar
    """
    # Pairwise TD errors: (batch, N_pred, N_target)
    td_errors = targets.unsqueeze(1) - predictions.unsqueeze(2)

    # Huber loss element
    huber = torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors ** 2,
        kappa * (td_errors.abs() - 0.5 * kappa)
    )

    # Quantile weight: asymmetric weighting
    taus_expanded = taus.unsqueeze(0).unsqueeze(2)
    quantile_weight = torch.abs(
        taus_expanded - (td_errors < 0).float()
    )

    loss = (quantile_weight * huber).sum(dim=-1).mean(dim=-1)
    return loss.mean()
```

### 3.3 QR-DQN Network

```python
class QRDQNNetwork(nn.Module):
    """Quantile Regression DQN network."""

    def __init__(self, state_dim, action_dim, n_quantiles=200):
        super().__init__()
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles

        # Quantile midpoints
        taus = torch.arange(1, n_quantiles + 1, dtype=torch.float32)
        self.register_buffer('taus', (2 * taus - 1) / (2 * n_quantiles))

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim * n_quantiles)

    def forward(self, state):
        """Return quantile values for each action."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        quantiles = x.view(-1, self.action_dim, self.n_quantiles)
        return quantiles

    def get_q_values(self, state):
        """Q-values as mean of quantile values."""
        quantiles = self.forward(state)
        return quantiles.mean(dim=-1)
```

### 3.4 QR-DQN vs C51 Comparison

| Feature | C51 | QR-DQN |
|---------|-----|--------|
| **Representation** | Fixed support, learned probs | Fixed probs, learned support |
| **Hyperparameters** | V_MIN, V_MAX, N_atoms | N_quantiles only |
| **Loss function** | Cross-entropy | Quantile Huber |
| **Convergence metric** | KL divergence | Wasserstein distance |
| **Flexibility** | Limited by support range | Unbounded support |
| **Typical N** | 51 | 200 |

---

## 4. Implicit Quantile Networks (IQN)

### 4.1 From Fixed to Sampled Quantiles

IQN goes further than QR-DQN by sampling quantile levels at random during training, rather than using a fixed set:

```
QR-DQN:  Fixed τ = {0.025, 0.075, ..., 0.975}  (N=20 quantiles)
IQN:     Sample τ ~ Uniform(0, 1)                (any quantile on demand)

This means IQN can approximate the FULL quantile function F⁻¹(τ).
```

### 4.2 Quantile Embedding

IQN embeds the quantile level τ using a cosine basis:

```python
class QuantileEmbedding(nn.Module):
    """Embed quantile levels using cosine basis functions."""

    def __init__(self, embedding_dim=64, n_cos=64):
        super().__init__()
        self.n_cos = n_cos
        self.embedding = nn.Linear(n_cos, embedding_dim)

        self.register_buffer(
            'i_pi',
            torch.arange(1, n_cos + 1, dtype=torch.float32) * np.pi
        )

    def forward(self, taus):
        """
        Embed quantile levels.
        Args:
            taus: (batch, N) quantile levels in [0, 1]
        Returns:
            embedding: (batch, N, embedding_dim)
        """
        cos_features = torch.cos(taus.unsqueeze(-1) * self.i_pi)
        embedding = F.relu(self.embedding(cos_features))
        return embedding


class IQNNetwork(nn.Module):
    """Implicit Quantile Network."""

    def __init__(self, state_dim, action_dim, embedding_dim=64, n_cos=64):
        super().__init__()
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        # State encoder
        self.state_fc1 = nn.Linear(state_dim, embedding_dim)
        self.state_fc2 = nn.Linear(embedding_dim, embedding_dim)

        # Quantile embedding
        self.quantile_embed = QuantileEmbedding(embedding_dim, n_cos)

        # Combined layers
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, state, taus):
        """
        Compute quantile values for given state and quantile levels.

        Args:
            state: (batch, state_dim)
            taus: (batch, N) sampled quantile levels
        Returns:
            quantile_values: (batch, N, action_dim)
        """
        # Encode state
        state_feat = F.relu(self.state_fc1(state))
        state_feat = F.relu(self.state_fc2(state_feat))

        # Embed quantiles
        tau_feat = self.quantile_embed(taus)

        # Element-wise product
        combined = state_feat.unsqueeze(1) * tau_feat

        # Output quantile values for each action
        x = F.relu(self.fc1(combined))
        quantile_values = self.fc2(x)

        return quantile_values

    def get_q_values(self, state, n_quantiles=32):
        """Estimate Q by sampling quantiles and averaging."""
        batch_size = state.shape[0]
        taus = torch.rand(batch_size, n_quantiles, device=state.device)
        quantile_values = self.forward(state, taus)
        return quantile_values.mean(dim=1)
```

### 4.3 Risk-Sensitive Policies with IQN

A major advantage of IQN: we can implement different risk attitudes by choosing which quantiles to evaluate:

```python
class RiskSensitiveIQN:
    """IQN agent with configurable risk attitude."""

    def __init__(self, network, risk_level='neutral'):
        self.network = network
        self.risk_level = risk_level

    def select_action(self, state, n_quantiles=32):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)

            if self.risk_level == 'neutral':
                # Uniform in [0, 1] -> expected value
                taus = torch.rand(1, n_quantiles)

            elif self.risk_level == 'averse':
                # Lower quantiles only: [0, 0.25] -> pessimistic (CVaR)
                taus = torch.rand(1, n_quantiles) * 0.25

            elif self.risk_level == 'seeking':
                # Upper quantiles only: [0.75, 1] -> optimistic
                taus = 0.75 + torch.rand(1, n_quantiles) * 0.25

            elif self.risk_level == 'cvar_50':
                # CVaR at 50%: average of lower 50% quantiles
                taus = torch.rand(1, n_quantiles) * 0.5

            quantile_values = self.network(state_t, taus)
            q_values = quantile_values.mean(dim=1)
            return q_values.argmax(dim=-1).item()
```

### 4.4 Comparison: C51 vs QR-DQN vs IQN

```
                C51              QR-DQN           IQN
              ──────────       ──────────       ──────────
Support:      Fixed atoms      Learned atoms    Implicit (any τ)
Probabilities: Learned         Fixed (uniform)  Sampled
Flexibility:   ★★              ★★★              ★★★★★
Risk control:  Limited         Limited          Full CVaR/CPT
Computation:   Low             Medium           Medium-High
Performance:   Good            Better           Best
```

---

## 5. Distributional Policy Gradients

### 5.1 Extending to Continuous Actions

Distributional methods were originally developed for DQN (discrete actions). Extending to continuous actions requires distributional policy gradient methods.

```python
class DistributionalCritic(nn.Module):
    """QR-DQN style critic for continuous actions."""

    def __init__(self, state_dim, action_dim, n_quantiles=25, hidden_dim=256):
        super().__init__()
        self.n_quantiles = n_quantiles

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_quantiles),
        )

    def forward(self, state, action):
        """Return quantile values for (state, action) pair."""
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
```

### 5.2 D4PG: Distributed Distributional DDPG

D4PG combines:
- Distributional critics (C51-style)
- Distributed training (multiple actors)
- N-step returns
- Prioritized experience replay

```
D4PG Architecture:

  Actor 1 ─────┐
  Actor 2 ─────┤
  Actor 3 ─────┼──▶ Prioritized    ──▶ Learner
  ...          │    Replay Buffer       (C51 Critic + DDPG Actor)
  Actor K ─────┘

  Each actor runs in parallel, collecting diverse experience.
  Learner updates distributional critic with cross-entropy loss.
```

### 5.3 Distributional SAC

```python
class DistributionalSAC:
    """
    Distributional Soft Actor-Critic combining SAC with distributional critics.
    Uses QR-DQN style critics for continuous action spaces.
    """

    def __init__(self, state_dim, action_dim, n_quantiles=25,
                 hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau_soft = tau
        self.alpha = alpha
        self.n_quantiles = n_quantiles

        # Quantile midpoints
        taus = torch.arange(1, n_quantiles + 1, dtype=torch.float32)
        self.taus = (2 * taus - 1) / (2 * n_quantiles)

        # Twin distributional critics
        self.critic1 = DistributionalCritic(state_dim, action_dim, n_quantiles, hidden_dim)
        self.critic2 = DistributionalCritic(state_dim, action_dim, n_quantiles, hidden_dim)

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )

    def update_critics(self, states, actions, rewards, next_states, dones,
                       next_actions, next_log_probs):
        """Update distributional critics with quantile regression."""
        with torch.no_grad():
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_quantiles = torch.min(target_q1, target_q2)

            target_quantiles = rewards.unsqueeze(1) + \
                self.gamma * (1 - dones.unsqueeze(1)) * \
                (target_quantiles - self.alpha * next_log_probs.unsqueeze(1))

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        loss1 = quantile_huber_loss(current_q1, target_quantiles, self.taus)
        loss2 = quantile_huber_loss(current_q2, target_quantiles, self.taus)

        critic_loss = loss1 + loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()
```

---

## 6. Risk-Sensitive Control

### 6.1 Risk Measures

With access to full return distributions, we can optimize various risk measures:

```
Risk Measures from Finance applied to RL:

1. Expected Value (risk-neutral):
   ρ(Z) = E[Z]

2. Variance:
   ρ(Z) = E[Z] - λ·Var(Z)     (mean-variance)

3. CVaR (Conditional Value at Risk):
   CVaR_α(Z) = E[Z | Z ≤ F⁻¹(α)]
   "Average return in the worst α% of cases"

4. Wang Risk Measure:
   Distorts the quantile function: g(τ) = Φ(Φ⁻¹(τ) + η)
   η > 0: risk-averse, η < 0: risk-seeking

5. Cumulative Prospect Theory (CPT):
   Different weighting for gains vs losses
   Overweights rare extreme events
```

### 6.2 CVaR Optimization with IQN

```python
def cvar_action_selection(iqn_network, state, alpha=0.25, n_samples=64):
    """
    Select action maximizing CVaR_α.

    CVaR_α = average of bottom α quantiles of return distribution.
    Lower α -> more conservative (worst-case focus).
    """
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)

        # Sample quantiles only from [0, α] for CVaR
        taus = torch.rand(1, n_samples) * alpha

        quantile_values = iqn_network(state_t, taus)
        cvar_values = quantile_values.mean(dim=1)

        return cvar_values.argmax(dim=-1).item()


def evaluate_risk_policies(env, iqn_network, n_episodes=100):
    """Compare different risk attitudes on the same environment."""
    risk_levels = {
        'CVaR 10% (very conservative)': 0.10,
        'CVaR 25% (conservative)': 0.25,
        'CVaR 50% (moderate)': 0.50,
        'Risk-neutral (CVaR 100%)': 1.00,
    }

    results = {}
    for name, alpha in risk_levels.items():
        returns = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_return = 0
            done = False

            while not done:
                action = cvar_action_selection(iqn_network, state, alpha)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                done = terminated or truncated

            returns.append(episode_return)

        results[name] = {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'cvar_10': np.percentile(returns, 10),
        }

    return results
```

### 6.3 Risk-Reward Tradeoffs

```
           Mean Return
              ▲
              │         ○ Risk-neutral
              │       ○
              │     ○ CVaR-50%
              │   ○
              │ ○ CVaR-25%
              │○ CVaR-10%
              └──────────────────────▶ Safety (higher CVaR₁₀)

  As we become more conservative:
  - Mean return decreases (pay a premium for safety)
  - Worst-case performance improves significantly
  - Standard deviation typically decreases
```

---

## 7. Practical Implementation Guide

### 7.1 Choosing the Right Algorithm

```
Decision tree for distributional RL:

  Need risk sensitivity?
  ├── Yes → IQN (full quantile function)
  └── No
       ├── Discrete actions?
       │   ├── Simple setup → C51 (well-understood, reliable)
       │   └── No V_MIN/V_MAX → QR-DQN (fewer hyperparameters)
       └── Continuous actions?
           └── D4PG or Distributional SAC
```

### 7.2 Hyperparameter Guidelines

| Parameter | C51 | QR-DQN | IQN |
|-----------|-----|--------|-----|
| **N (atoms/quantiles)** | 51 | 200 | 64 (sampled) |
| **V_MIN, V_MAX** | Task-dependent | N/A | N/A |
| **Learning rate** | 2.5e-4 | 5e-5 | 5e-5 |
| **Huber kappa** | N/A | 1.0 | 1.0 |
| **Cosine embedding dim** | N/A | N/A | 64 |
| **Batch size** | 32 | 32 | 32 |
| **Target update** | 8000 steps | 8000 steps | 8000 steps |

### 7.3 Common Pitfalls

```python
# Pitfall 1: Wrong V_MIN/V_MAX for C51
# If returns exceed [V_MIN, V_MAX], distribution gets clipped
# Solution: Set range wider than expected returns

# Pitfall 2: Numerical instability in log probabilities
# Bad:
loss = -(target * torch.log(predicted)).sum()
# Good:
loss = -(target * torch.log(predicted + 1e-8)).sum()

# Pitfall 3: Forgetting to sort quantiles for QR-DQN
# Quantile values should be roughly sorted
# The loss naturally encourages this, but initialization matters

# Pitfall 4: Too few quantile samples in IQN during evaluation
# Training: N=8 samples is fine (stochastic is OK)
# Evaluation: Use N=32+ for stable Q-value estimates
```

### 7.4 Full Training Example

```python
import gymnasium as gym
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones, dtype=float))

    def __len__(self):
        return len(self.buffer)


def train_c51(env_name='CartPole-v1', n_episodes=500, n_atoms=51):
    """Train C51 agent on a Gymnasium environment."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = C51Agent(state_dim, action_dim, n_atoms=n_atoms)
    buffer = ReplayBuffer()
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 64
    target_update_freq = 500
    step_count = 0

    episode_returns = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_return = 0
        done = False

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_return += reward
            step_count += 1

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                agent.train_step(batch)

            if step_count % target_update_freq == 0:
                agent.target_network.load_state_dict(
                    agent.network.state_dict()
                )

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_returns.append(episode_return)

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_returns[-50:])
            print(f"Episode {episode+1}, Avg Return: {avg:.1f}, "
                  f"Epsilon: {epsilon:.3f}")

    env.close()
    return agent, episode_returns
```

---

## 8. Exercises

### Exercise 1: Implement C51 from Scratch

Build a complete C51 agent for CartPole-v1:
1. Implement the C51Network with configurable N_atoms
2. Implement the categorical projection step
3. Train for 500 episodes and plot the learning curve
4. Visualize the learned return distributions for different states
5. Compare N_atoms = {11, 21, 51} and analyze the effect

### Exercise 2: QR-DQN Implementation

Implement QR-DQN and compare with C51:
1. Build the QR-DQN network with quantile Huber loss
2. Train on CartPole-v1 and LunarLander-v2
3. Plot the learned quantile functions for key states
4. Compare convergence speed and final performance vs C51
5. Experiment with N_quantiles = {10, 25, 50, 200}

### Exercise 3: IQN with Risk Sensitivity

Build an IQN agent with risk-sensitive policies:
1. Implement the cosine embedding for quantile levels
2. Train on a stochastic environment (e.g., modified CartPole with random wind)
3. Implement CVaR action selection at different alpha levels
4. Compare risk-neutral vs CVaR-25% policies: plot return distributions
5. Show that conservative policies sacrifice mean return for lower variance

### Exercise 4: Distribution Visualization Dashboard

Create a visualization tool for distributional RL:
1. During training, save return distributions at key states every 100 episodes
2. Create animated plots showing how distributions evolve
3. Show the projection step visually: before and after projection
4. Plot the Wasserstein distance between consecutive distribution estimates
5. Identify states where the distribution is bimodal and explain why

### Exercise 5: Distributional Rainbow

Combine distributional RL with other DQN improvements:
1. Start with C51 as the base
2. Add prioritized experience replay (prioritize by distributional TD error)
3. Add n-step returns with distribution projection
4. Add noisy networks (replace epsilon-greedy)
5. Compare: DQN, C51, QR-DQN, and your distributional Rainbow on Atari

---

*End of Lesson 18*
