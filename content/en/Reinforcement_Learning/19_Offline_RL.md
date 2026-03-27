[Previous: Distributional RL](./18_Distributional_RL.md)

---

# 19. Offline Reinforcement Learning

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the offline RL problem setting and distribution shift challenges
2. Implement Conservative Q-Learning (CQL) with lower-bound Q-value estimation
3. Build a Decision Transformer that casts RL as sequence modeling
4. Understand behavior cloning limits and when it suffices vs full offline RL
5. Compare BCQ, BEAR, CQL, and Decision Transformer on D4RL benchmarks

---

## Table of Contents

1. [The Offline RL Problem](#1-the-offline-rl-problem)
2. [Distribution Shift and Extrapolation Error](#2-distribution-shift-and-extrapolation-error)
3. [Behavior Cloning Revisited](#3-behavior-cloning-revisited)
4. [Conservative Q-Learning (CQL)](#4-conservative-q-learning-cql)
5. [Decision Transformer](#5-decision-transformer)
6. [Other Offline Methods: BCQ and BEAR](#6-other-offline-methods-bcq-and-bear)
7. [Practical Guide and D4RL Benchmarks](#7-practical-guide-and-d4rl-benchmarks)
8. [Exercises](#8-exercises)

---

## 1. The Offline RL Problem

### 1.1 Online vs Offline RL

```
Online RL:
  Agent ──▶ Environment ──▶ Agent ──▶ Environment ──▶ ...
  (act)     (observe)       (learn)   (explore more)
  + Can explore freely
  - Expensive/dangerous in real world (robotics, healthcare, autonomous driving)

Offline RL (Batch RL):
  Fixed Dataset D = {(s, a, r, s')₁, (s, a, r, s')₂, ...}
        │
        ▼
  Learn policy π from D without any further interaction
  + Safe: no risky exploration
  + Leverage existing logged data (hospital records, driving logs)
  - Distribution shift: cannot query actions outside dataset
```

### 1.2 Why Is Offline RL Hard?

The fundamental challenge is **distribution shift**: the learned policy may visit state-action pairs not covered by the dataset.

```python
import numpy as np

def demonstrate_distribution_shift():
    """Show why naive off-policy learning fails offline."""
    # Suppose behavior policy collects data near action=0
    n_samples = 1000
    states = np.random.uniform(-1, 1, n_samples)
    actions = np.random.normal(0, 0.3, n_samples)  # centered near 0
    rewards = -(states ** 2) - (actions - states) ** 2  # optimal: a = s

    print("Dataset statistics:")
    print(f"  Actions: mean={actions.mean():.2f}, std={actions.std():.2f}")
    print(f"  Action range: [{actions.min():.2f}, {actions.max():.2f}]")
    print()
    print("Problem: Q(s, a=5) was never observed in dataset.")
    print("Q-learning may extrapolate arbitrarily high values!")
    print("The learned policy then selects these out-of-distribution actions.")

demonstrate_distribution_shift()
```

### 1.3 Extrapolation Error

```
Q-value landscape:

  Q(s,a)
    ▲        / Extrapolated (wrong!)
    │      //
    │    //
    │  //      In-distribution      Out-of-distribution
    │ /   ●●●●●●●●●●●              ???
    │/    (data covers this)        (no data here)
    └─────────────────────────────▶ action
         a_min        a_max

  Standard DQN picks argmax Q -> selects out-of-distribution action -> disaster
```

---

## 2. Distribution Shift and Extrapolation Error

### 2.1 Formal Analysis

Let π_β be the behavior policy (that collected the data) and π be the learned policy.

```
Offline RL Objective:
  max_π E_{s~d^π} [Σ γᵗ r(sₜ, π(sₜ))]

But we only have data from d^{π_β} (behavior policy's state distribution).

When π ≠ π_β:
  - π visits states where d^{π_β}(s) ≈ 0 -> no training data
  - Q(s, a) for unseen (s, a) is unreliable
  - Errors compound over multi-step rollouts
```

### 2.2 Measuring OOD Degree

```python
from sklearn.neighbors import KernelDensity

def measure_ood_degree(dataset_actions, policy_actions, bandwidth=0.1):
    """
    Estimate how out-of-distribution the policy's actions are
    relative to the behavior dataset.
    """
    # Fit KDE on dataset actions
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(dataset_actions.reshape(-1, 1))

    # Score policy actions
    log_density = kde.score_samples(policy_actions.reshape(-1, 1))
    density = np.exp(log_density)

    print(f"Dataset action density at policy actions:")
    print(f"  Mean density: {density.mean():.4f}")
    print(f"  Min density:  {density.min():.6f}")
    print(f"  % below threshold (OOD): "
          f"{(density < 0.01).mean()*100:.1f}%")

    return density
```

### 2.3 Solutions Taxonomy

```
Offline RL Methods:

1. Policy Constraint Methods
   ├── BCQ (Batch-Constrained Q-learning)
   │     └── Only consider actions "close" to dataset
   ├── BEAR (Bootstrapping Error Accumulation Reduction)
   │     └── MMD constraint on action distribution
   └── TD3+BC
         └── Simple behavior cloning regularization

2. Value Pessimism Methods
   ├── CQL (Conservative Q-Learning)
   │     └── Push down Q-values for OOD actions
   └── PBRL (Pessimistic Bellman Reinforcement Learning)
         └── Lower confidence bound on Q

3. Model-Based Offline RL
   ├── MOPO (Model-based Offline Policy Optimization)
   │     └── Penalize uncertainty in learned dynamics
   └── MOReL
         └── Construct pessimistic MDP

4. Sequence Modeling
   └── Decision Transformer
         └── Cast RL as conditional sequence generation
```

---

## 3. Behavior Cloning Revisited

### 3.1 When Is BC Enough?

Behavior cloning (supervised learning on expert actions) is the simplest approach.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviorCloningPolicy(nn.Module):
    """Simple behavior cloning via supervised learning."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)

    def get_action(self, state):
        with torch.no_grad():
            return self.forward(torch.FloatTensor(state)).numpy()


def train_bc(dataset, state_dim, action_dim, epochs=100, batch_size=256):
    """Train behavior cloning policy."""
    policy = BehaviorCloningPolicy(state_dim, action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    states = torch.FloatTensor(dataset['states'])
    actions = torch.FloatTensor(dataset['actions'])
    n = len(states)

    for epoch in range(epochs):
        indices = torch.randperm(n)
        total_loss = 0

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            s_batch = states[batch_idx]
            a_batch = actions[batch_idx]

            predicted = policy(s_batch)
            loss = F.mse_loss(predicted, a_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / (n // batch_size)
            print(f"Epoch {epoch+1}, BC Loss: {avg_loss:.4f}")

    return policy
```

### 3.2 BC Limitations: Compounding Errors

```
The problem with BC in sequential decisions:

Step 1: Small error ε -> slightly off-track
Step 2: Now in unfamiliar state -> error grows to 2ε
Step 3: Further off -> error 3ε
...
Step T: Error ~ T·ε  (linear compounding!)

Expert trajectory:  ● → ● → ● → ● → ● → ● → GOAL
BC trajectory:      ● → ●↗ → ●↗↗ → ???  → ???  → CRASH

DAgger (Dataset Aggregation) addresses this by iteratively querying
the expert in states visited by the learned policy.
But in offline RL, we can't query the expert!
```

### 3.3 When BC Works vs When You Need Offline RL

| Scenario | BC Sufficient? | Why? |
|----------|---------------|------|
| Expert-only data, short horizon | Yes | Low compounding error |
| Expert-only data, long horizon | Maybe | Use with DAgger if possible |
| Mixed-quality data | No | BC averages over bad+good demos |
| Sub-optimal data | No | BC can only match data quality |
| Need to exceed data quality | No | Need value-based offline RL |

---

## 4. Conservative Q-Learning (CQL)

### 4.1 The CQL Idea

CQL adds a regularizer that pushes Q-values down for out-of-distribution actions while keeping them up for in-distribution actions.

```
Standard Q-learning:
  Minimize: E_{(s,a,r,s')~D} [(Q(s,a) - (r + γ max_a' Q(s',a')))²]

CQL adds:
  + α · E_{s~D} [log Σ_a exp(Q(s,a))]     <- push down all Q(s,a)
  - α · E_{(s,a)~D} [Q(s,a)]               <- push up Q for in-data actions

Net effect: Q-values for OOD actions are conservatively low.
```

### 4.2 CQL Implementation

```python
class QNetwork(nn.Module):
    """Simple Q-network for continuous state-action spaces."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous actions."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state, n_samples=1):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        if n_samples > 1:
            # Return (batch, n_samples, action_dim)
            actions = dist.rsample((n_samples,)).permute(1, 0, 2)
            log_probs = dist.log_prob(actions.permute(1, 0, 2)).sum(-1).permute(1, 0)
            return actions, log_probs
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class CQLAgent:
    """Conservative Q-Learning for offline RL."""

    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 lr=3e-4, gamma=0.99, cql_alpha=1.0, tau=0.005,
                 n_random_actions=10):
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.action_dim = action_dim
        self.n_random_actions = n_random_actions

        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim)

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr
        )

    def compute_cql_loss(self, states, actions):
        """Compute CQL regularization term."""
        batch_size = states.shape[0]

        # Sample random actions uniformly
        random_actions = torch.FloatTensor(
            batch_size, self.n_random_actions, self.action_dim
        ).uniform_(-1, 1)

        # Sample actions from current policy
        with torch.no_grad():
            policy_actions, policy_log_probs = self.policy.sample(
                states, n_samples=self.n_random_actions
            )

        # Q-values for random and policy actions
        random_q1 = self._get_q_batch(self.q1, states, random_actions)
        random_q2 = self._get_q_batch(self.q2, states, random_actions)
        policy_q1 = self._get_q_batch(self.q1, states, policy_actions)
        policy_q2 = self._get_q_batch(self.q2, states, policy_actions)

        # LogSumExp over sampled actions
        cat_q1 = torch.cat([random_q1, policy_q1], dim=1)
        cat_q2 = torch.cat([random_q2, policy_q2], dim=1)

        logsumexp_q1 = torch.logsumexp(cat_q1, dim=1).mean()
        logsumexp_q2 = torch.logsumexp(cat_q2, dim=1).mean()

        # Subtract data Q-values
        data_q1 = self.q1(states, actions).mean()
        data_q2 = self.q2(states, actions).mean()

        cql_loss = (logsumexp_q1 - data_q1) + (logsumexp_q2 - data_q2)
        return cql_loss

    def _get_q_batch(self, q_net, states, action_samples):
        """Get Q-values for multiple action samples per state."""
        batch_size = states.shape[0]
        n_samples = action_samples.shape[1]

        states_expanded = states.unsqueeze(1).expand(-1, n_samples, -1)
        states_flat = states_expanded.reshape(-1, states.shape[-1])
        actions_flat = action_samples.reshape(-1, self.action_dim)

        q_values = q_net(states_flat, actions_flat)
        return q_values.reshape(batch_size, n_samples)

    def train_step(self, batch):
        """One training step on offline batch."""
        states, actions, rewards, next_states, dones = [
            torch.FloatTensor(x) for x in batch
        ]

        # Standard TD loss
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            target_q = torch.min(
                self.target_q1(next_states, next_actions),
                self.target_q2(next_states, next_actions)
            )
            target_value = rewards + self.gamma * (1 - dones) * target_q.squeeze()

        current_q1 = self.q1(states, actions).squeeze()
        current_q2 = self.q2(states, actions).squeeze()

        td_loss = F.mse_loss(current_q1, target_value) + \
                  F.mse_loss(current_q2, target_value)

        # CQL regularization
        cql_loss = self.compute_cql_loss(states, actions)

        # Total critic loss
        critic_loss = td_loss + self.cql_alpha * cql_loss

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # Policy update
        new_actions, log_probs = self.policy.sample(states)
        q_new = torch.min(
            self.q1(states, new_actions),
            self.q2(states, new_actions)
        )
        policy_loss = -q_new.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft target update
        for param, target_param in zip(self.q1.parameters(),
                                        self.target_q1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(self.q2.parameters(),
                                        self.target_q2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'policy_loss': policy_loss.item(),
        }
```

### 4.3 CQL Variants

```
CQL variants and their regularizers:

CQL(H):  α · (E_s[log Σ_a exp(Q(s,a))] - E_{(s,a)~D}[Q(s,a)])
          Automatic α via Lagrangian: adjust to maintain target gap

CQL(ρ):  α · (E_{s,a~ρ}[Q(s,a)] - E_{(s,a)~D}[Q(s,a)])
          Where ρ can be uniform, policy, or mixture

Key insight: CQL provably learns a lower bound on Q^π
  Q_CQL(s,a) ≤ Q^π(s,a) for all (s,a) with high probability
  -> Safe policy improvement guarantee!
```

---

## 5. Decision Transformer

### 5.1 RL as Sequence Modeling

Decision Transformer recasts offline RL as a sequence prediction problem:

```
Traditional RL: Learn Q/π -> plan via value maximization
Decision Transformer: Learn to predict actions given desired returns

Input sequence:
  (R̂₁, s₁, a₁, R̂₂, s₂, a₂, ..., R̂ₜ, sₜ, ???)
                                              ↑
                                    Predict aₜ given target return R̂ₜ

R̂ₜ = "return-to-go" = desired cumulative return from timestep t onward

At test time: set R̂₁ = high target -> model generates expert-like actions
```

### 5.2 Architecture

```
                    Return-to-go  State    Action
                    Embedding     Embedding Embedding
                         │           │        │
                         ▼           ▼        ▼
Timestep ──▶ [R̂₁ s₁ a₁ | R̂₂ s₂ a₂ | R̂₃ s₃ ???]
Embedding                                     ↑
                         │                     │
                    ┌────▼─────────────────────┐
                    │    GPT-2 Transformer     │
                    │    (causal attention)     │
                    └──────────────────────────┘
                                    │
                                    ▼
                              Predicted a₃
```

### 5.3 Implementation

```python
class DecisionTransformer(nn.Module):
    """Decision Transformer for offline RL."""

    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 n_heads=4, n_layers=3, max_length=20,
                 max_ep_length=1000, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Embeddings for each modality
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.return_embed = nn.Linear(1, hidden_dim)

        # Positional (timestep) embedding
        self.timestep_embed = nn.Embedding(max_ep_length, hidden_dim)

        # Layer norm
        self.embed_ln = nn.LayerNorm(hidden_dim)

        # GPT-2 style transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Prediction head
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, returns_to_go, states, actions, timesteps):
        """
        Args:
            returns_to_go: (batch, T, 1)
            states: (batch, T, state_dim)
            actions: (batch, T, action_dim)
            timesteps: (batch, T)
        Returns:
            predicted_actions: (batch, T, action_dim)
        """
        batch_size, T = states.shape[0], states.shape[1]

        # Embed each modality
        state_embeddings = self.state_embed(states)
        action_embeddings = self.action_embed(actions)
        return_embeddings = self.return_embed(returns_to_go)

        # Add timestep embeddings
        time_embeddings = self.timestep_embed(timesteps)
        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        return_embeddings += time_embeddings

        # Interleave: [R1, s1, a1, R2, s2, a2, ...]
        stacked = torch.stack(
            [return_embeddings, state_embeddings, action_embeddings],
            dim=2
        ).reshape(batch_size, 3 * T, self.hidden_dim)

        stacked = self.embed_ln(stacked)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(3 * T, 3 * T, device=states.device) * float('-inf'),
            diagonal=1
        )

        # Transformer forward
        output = self.transformer(stacked, mask=causal_mask)

        # Extract state position outputs (predict action at each timestep)
        state_outputs = output[:, 1::3, :]

        predicted_actions = self.predict_action(state_outputs)
        return predicted_actions
```

### 5.4 Inference with Decision Transformer

```python
def evaluate_decision_transformer(model, env, target_return,
                                  max_ep_length=1000, context_length=20):
    """Evaluate Decision Transformer with a target return."""
    model.eval()

    state, _ = env.reset()
    states = [state]
    actions = []
    returns_to_go = [target_return]
    timesteps = [0]

    episode_return = 0

    for t in range(max_ep_length):
        K = min(t + 1, context_length)

        rtg_input = torch.FloatTensor(returns_to_go[-K:]).reshape(1, K, 1)
        state_input = torch.FloatTensor(np.array(states[-K:])).unsqueeze(0)

        if len(actions) > 0:
            action_input = torch.FloatTensor(
                np.array(actions[-(K-1):] + [[0]*model.action_dim])
            ).unsqueeze(0)
        else:
            action_input = torch.zeros(1, K, model.action_dim)

        timestep_input = torch.LongTensor(timesteps[-K:]).unsqueeze(0)

        with torch.no_grad():
            predicted = model(rtg_input, state_input, action_input,
                            timestep_input)
            action = predicted[0, -1].numpy()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += reward

        states.append(next_state)
        actions.append(action)
        returns_to_go.append(returns_to_go[-1] - reward)
        timesteps.append(t + 1)

        if done:
            break

    return episode_return
```

### 5.5 Decision Transformer Strengths and Limitations

| Aspect | Strength | Limitation |
|--------|----------|------------|
| **Simplicity** | No Bellman updates, no value estimation | Cannot stitch trajectories |
| **Conditioning** | Easy to specify desired returns | Requires knowing good target returns |
| **Stability** | No bootstrapping -> no divergence | May underperform on stitching tasks |
| **Scaling** | Leverages transformer scaling laws | Computationally expensive |
| **Multi-task** | One model, different return targets | Sensitive to return scale |

---

## 6. Other Offline Methods: BCQ and BEAR

### 6.1 Batch-Constrained Q-Learning (BCQ)

BCQ constrains the learned policy to only select actions within the support of the behavior policy:

```python
class BCQAgent:
    """Batch-Constrained Q-learning agent."""

    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 lr=3e-4, gamma=0.99, tau=0.005, phi=0.05):
        self.gamma = gamma
        self.tau = tau
        self.phi = phi  # action perturbation range

        # Generative model: VAE to model behavior policy
        # self.vae = VAE(state_dim, action_dim, hidden_dim, latent_dim=action_dim*2)

        # Perturbation model: small adjustments within [-phi, phi]
        # self.perturbation = PerturbationNetwork(state_dim, action_dim, hidden_dim, phi)

        # Twin Q-networks
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)

    def select_action(self, state, vae, perturbation, n_candidates=100):
        """Select action using BCQ procedure."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)

            # 1. Sample candidate actions from VAE (behavior model)
            state_repeated = state_t.repeat(n_candidates, 1)
            candidates = vae.decode(state_repeated)

            # 2. Perturb each candidate slightly
            perturbed = perturbation(state_repeated, candidates)

            # 3. Pick the one with highest Q-value
            q1 = self.q1(state_repeated, perturbed)
            q2 = self.q2(state_repeated, perturbed)
            q = torch.min(q1, q2)

            best_idx = q.argmax(dim=0)
            return perturbed[best_idx].cpu().numpy()
```

### 6.2 BEAR: Bootstrapping Error Accumulation Reduction

BEAR constrains the learned policy using Maximum Mean Discrepancy (MMD):

```python
def mmd_loss(policy_actions, dataset_actions, kernel='laplacian', sigma=20.0):
    """
    Compute MMD between policy action distribution and dataset actions.
    MMD^2(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    """
    if kernel == 'laplacian':
        def k(x, y):
            diff = (x.unsqueeze(1) - y.unsqueeze(0)).abs().sum(-1)
            return torch.exp(-diff / sigma)
    elif kernel == 'gaussian':
        def k(x, y):
            diff = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1)
            return torch.exp(-diff / (2 * sigma ** 2))

    kpp = k(policy_actions, policy_actions).mean()
    kdd = k(dataset_actions, dataset_actions).mean()
    kpd = k(policy_actions, dataset_actions).mean()

    mmd_squared = kpp + kdd - 2 * kpd
    return mmd_squared
```

### 6.3 Method Comparison on D4RL

```
D4RL Benchmark Results (normalized scores, higher is better):

Environment           | BC   | BCQ  | BEAR | CQL  | DT
------------------------------------------------------
halfcheetah-medium    | 42.6 | 40.7 | 41.7 | 44.0 | 42.6
hopper-medium         | 52.9 | 54.5 | 52.1 | 58.5 | 67.6
walker2d-medium       | 75.3 | 53.1 | 59.1 | 72.5 | 74.0
halfcheetah-med-expert| 55.2 | 64.7 | 53.4 | 91.6 | 86.8
hopper-med-expert     | 52.5 | 110.9| 96.3 | 105.4| 107.6
walker2d-med-expert   | 107.5| 57.5 | 40.1 | 108.8| 108.1

Key findings:
- CQL strong across the board, especially on mixed-quality data
- DT excels when data has high-return trajectories
- BCQ/BEAR good but can struggle with mixed data
- BC surprisingly competitive on expert-only data
```

---

## 7. Practical Guide and D4RL Benchmarks

### 7.1 Algorithm Selection Guide

```
Which offline RL method should you use?

Data quality?
├── Expert-only -> Behavior Cloning (simplest, often sufficient)
├── Mixed quality -> CQL or IQL (handles multi-modal data well)
├── Suboptimal-only -> CQL (can improve beyond data)
└── Random -> Hard for all methods; consider model-based

Key considerations:
├── Need trajectory stitching? -> CQL/IQL (DT cannot stitch)
├── Simple implementation? -> TD3+BC (just add BC regularizer to TD3)
├── Sequence modeling fan? -> Decision Transformer
└── Continuous control? -> CQL-SAC or IQL
```

### 7.2 Common Offline RL Pitfalls

```python
# Pitfall 1: Evaluating with only mean performance
# Offline RL can have high variance; report confidence intervals
def evaluate_properly(agent, env, n_episodes=100):
    returns = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_return = 0
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)

    print(f"Mean: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"Median: {np.median(returns):.1f}")
    print(f"Min/Max: {np.min(returns):.1f} / {np.max(returns):.1f}")
    return returns

# Pitfall 2: Hyperparameter sensitivity
# CQL's alpha is critical: too high -> overly conservative, too low -> OOD issues
# Solution: Use automatic alpha tuning via Lagrangian

# Pitfall 3: Not normalizing rewards/states
def normalize_dataset(dataset):
    state_mean = dataset['observations'].mean(axis=0)
    state_std = dataset['observations'].std(axis=0) + 1e-6
    dataset['observations'] = (dataset['observations'] - state_mean) / state_std
    dataset['next_observations'] = (
        dataset['next_observations'] - state_mean
    ) / state_std
    return dataset, state_mean, state_std
```

### 7.3 Offline-to-Online Fine-tuning

```python
def offline_to_online(agent, env, offline_steps=50000,
                      online_steps=50000, batch_size=256):
    """
    Two-phase training:
    Phase 1: Offline pre-training on static dataset
    Phase 2: Online fine-tuning with environment interaction
    """
    # Phase 1: Offline
    print("Phase 1: Offline pre-training...")
    dataset = env.get_dataset()
    for step in range(offline_steps):
        batch = sample_batch(dataset, batch_size)
        agent.train_step(batch)

        if (step + 1) % 10000 == 0:
            scores = evaluate_properly(agent, env, n_episodes=10)
            print(f"  Offline step {step+1}: {np.mean(scores):.1f}")

    # Phase 2: Online fine-tuning
    print("Phase 2: Online fine-tuning...")
    from collections import deque
    replay_buffer = deque(maxlen=online_steps)

    state, _ = env.reset()
    for step in range(online_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.append((state, action, reward, next_state, float(done)))
        state = next_state if not done else env.reset()[0]

        if len(replay_buffer) >= batch_size:
            import random
            batch = random.sample(replay_buffer, batch_size)
            batch = [np.array(x) for x in zip(*batch)]
            agent.train_step(batch)

        if (step + 1) % 10000 == 0:
            scores = evaluate_properly(agent, env, n_episodes=10)
            print(f"  Online step {step+1}: {np.mean(scores):.1f}")
```

---

## 8. Exercises

### Exercise 1: Implement Behavior Cloning Baseline

Build and evaluate a BC baseline on a custom offline dataset:
1. Collect data from a trained CartPole agent at different skill levels (random, medium, expert)
2. Train BC on each dataset and compare performance
3. Measure compounding error: compare BC performance at different horizon lengths
4. Plot learning curves as a function of dataset size

### Exercise 2: Conservative Q-Learning from Scratch

Implement CQL for discrete actions:
1. Create an offline dataset from a medium-quality CartPole policy
2. Implement CQL with the conservative regularizer
3. Compare CQL with naive offline DQN (no constraint)
4. Ablate CQL alpha: plot performance vs alpha = {0.1, 0.5, 1.0, 5.0, 10.0}
5. Show that CQL learns conservative Q-values (compare with true Q)

### Exercise 3: Decision Transformer

Build a minimal Decision Transformer:
1. Collect trajectory dataset from multiple CartPole episodes
2. Implement the causal transformer with return-to-go conditioning
3. Train and evaluate with different target returns
4. Show that higher target returns produce better policies (up to data quality limit)
5. Visualize attention patterns: what does the model attend to?

### Exercise 4: Offline RL Data Quality Study

Systematically study how data quality affects offline RL:
1. Generate datasets with varying quality: random, 25%, 50%, 75%, expert
2. Also vary dataset sizes: 1K, 10K, 100K, 1M transitions
3. Train BC, CQL, and DT on each combination
4. Create a heatmap: rows=quality, columns=dataset size, cells=performance
5. Identify crossover points where offline RL beats BC

### Exercise 5: Trajectory Stitching Demonstration

Demonstrate the "stitching" ability that separates value-based offline RL from BC/DT:
1. Create a simple 2D navigation environment with 2 rooms
2. Collect data: some trajectories go A->B, others go B->C, none go A->C
3. Show that BC/DT cannot learn A->C (never seen in data)
4. Show that CQL can stitch sub-trajectories to discover A->C path
5. Visualize the value function learned by CQL showing the stitched path

---

*End of Lesson 19*
