[Previous: World Models](./25_World_Models.md)

---

# 26. Imitation Learning

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the imitation learning spectrum from behavioral cloning to interactive methods
2. Implement DAgger for iterative dataset aggregation with expert queries
3. Build adversarial imitation learning methods beyond GAIL
4. Understand observation-only imitation learning (learning from videos)
5. Compare imitation approaches on sample efficiency and generalization

---

## Table of Contents

1. [Imitation Learning Fundamentals](#1-imitation-learning-fundamentals)
2. [Behavioral Cloning Deep Dive](#2-behavioral-cloning-deep-dive)
3. [DAgger and Interactive Methods](#3-dagger-and-interactive-methods)
4. [Adversarial Imitation Learning](#4-adversarial-imitation-learning)
5. [Observation-Only Imitation](#5-observation-only-imitation)
6. [Few-Shot and One-Shot Imitation](#6-few-shot-and-one-shot-imitation)
7. [Practical Imitation Pipelines](#7-practical-imitation-pipelines)
8. [Exercises](#8-exercises)

---

## 1. Imitation Learning Fundamentals

### 1.1 The Imitation Spectrum

```
Imitation Learning Methods (ordered by expert access needed):

1. Behavioral Cloning (BC)
   Expert access: Offline dataset of demonstrations
   Method: Supervised learning π(a|s) = argmax P(a|s)
   Pros: Simple, no environment needed
   Cons: Compounding errors, distribution shift

2. DAgger (Dataset Aggregation)
   Expert access: Can query expert during training
   Method: Iteratively collect data where learner visits
   Pros: Handles distribution shift
   Cons: Needs online expert access

3. IRL / GAIL
   Expert access: Demonstrations only
   Method: Learn reward / occupancy matching
   Pros: Can generalize beyond demonstrations
   Cons: Requires environment access

4. Observation-only IL
   Expert access: Videos only (no actions!)
   Method: Learn state mapping or inverse dynamics
   Pros: Many "demonstrations" freely available
   Cons: Action inference is hard
```

### 1.2 Problem Formulation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard notation:
# Expert demonstrations: D_E = {(s_i, a_i)}_{i=1}^N
# Expert policy: π_E(a|s)
# Learner policy: π_θ(a|s)
# Environment dynamics: T(s'|s,a)

# Objective varies by method:
# BC:     min_θ E_{(s,a)~D_E} [L(π_θ(s), a)]
# DAgger: min_θ E_{s~d^{π_θ}} [L(π_θ(s), π_E(s))]
# GAIL:   min_π max_D E_E[log D(s,a)] + E_π[log(1-D(s,a))]
```

---

## 2. Behavioral Cloning Deep Dive

### 2.1 Advanced BC Architectures

```python
class TransformerBCPolicy(nn.Module):
    """Transformer-based behavioral cloning with history."""

    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 n_heads=4, n_layers=2, context_length=10):
        super().__init__()
        self.context_length = context_length

        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.pos_embed = nn.Embedding(context_length, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=4*hidden_dim, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state_history):
        """
        Args:
            state_history: (batch, T, state_dim) last T states
        Returns:
            action: (batch, action_dim) predicted action
        """
        T = state_history.shape[1]
        positions = torch.arange(T, device=state_history.device)

        x = self.state_embed(state_history) + self.pos_embed(positions)

        mask = torch.triu(
            torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1
        )
        x = self.transformer(x, mask=mask)

        return self.action_head(x[:, -1])  # Predict from last position


class GaussianBCPolicy(nn.Module):
    """BC with Gaussian action prediction (for continuous actions)."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.net(state)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-5, 2)
        return mean, log_std

    def loss(self, states, expert_actions):
        """Negative log-likelihood loss."""
        mean, log_std = self.forward(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return -dist.log_prob(expert_actions).sum(dim=-1).mean()
```

### 2.2 Data Augmentation for BC

```python
class BCDataAugmentation:
    """Data augmentation to improve BC robustness."""

    def __init__(self, noise_std=0.01, action_noise_std=0.0):
        self.noise_std = noise_std
        self.action_noise_std = action_noise_std

    def augment_state(self, states):
        """Add Gaussian noise to states."""
        noise = torch.randn_like(states) * self.noise_std
        return states + noise

    def augment_continuous_actions(self, actions):
        """Small noise on continuous actions."""
        noise = torch.randn_like(actions) * self.action_noise_std
        return actions + noise

    def temporal_augment(self, trajectory, p_drop=0.1):
        """Randomly drop/duplicate timesteps."""
        augmented = []
        for t in range(len(trajectory)):
            if np.random.random() > p_drop:
                augmented.append(trajectory[t])
                if np.random.random() < p_drop:
                    augmented.append(trajectory[t])  # duplicate
        return augmented
```

---

## 3. DAgger and Interactive Methods

### 3.1 DAgger Algorithm

```
DAgger (Dataset Aggregation, Ross et al., 2011):

1. Initialize D ← expert demonstrations
2. Train π₁ on D (behavioral cloning)
3. For i = 1, 2, ..., N:
   a. Execute π_i in environment, collect states S_i
   b. Query expert π_E on S_i to get labels
   c. D ← D ∪ {(s, π_E(s)) for s in S_i}
   d. Train π_{i+1} on D

Key insight: By training on states VISITED BY THE LEARNER
(not just the expert), DAgger handles distribution shift!
```

### 3.2 DAgger Implementation

```python
class DAggerTrainer:
    """DAgger: Dataset Aggregation for imitation learning."""

    def __init__(self, policy, expert_fn, env, lr=1e-3,
                 mixing_decay=0.99):
        self.policy = policy
        self.expert = expert_fn  # Can query expert for any state
        self.env = env
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.dataset = []  # Aggregated dataset
        self.beta = 1.0  # Mixing coefficient (start with expert)
        self.mixing_decay = mixing_decay

    def collect_episode(self, max_steps=200):
        """Collect episode, mixing learner and expert actions."""
        state, _ = self.env.reset()
        episode_data = []

        for step in range(max_steps):
            # Mix learner and expert actions
            if np.random.random() < self.beta:
                action = self.expert(state)  # Expert action
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action = self.policy(state_t).squeeze(0).numpy()

            # ALWAYS label with expert action (regardless of who acted)
            expert_action = self.expert(state)
            episode_data.append((state.copy(), expert_action.copy()))

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            state = next_state

            if terminated or truncated:
                break

        return episode_data

    def train(self, n_iterations=50, episodes_per_iter=10,
              train_epochs=10, batch_size=64):
        """Full DAgger training loop."""
        performance_history = []

        for iteration in range(n_iterations):
            # Collect data with current policy
            new_data = []
            for _ in range(episodes_per_iter):
                episode = self.collect_episode()
                new_data.extend(episode)

            self.dataset.extend(new_data)

            # Train on aggregated dataset
            states = torch.FloatTensor([d[0] for d in self.dataset])
            actions = torch.FloatTensor([d[1] for d in self.dataset])

            for epoch in range(train_epochs):
                indices = torch.randperm(len(self.dataset))
                for i in range(0, len(self.dataset), batch_size):
                    batch_idx = indices[i:i+batch_size]
                    s_batch = states[batch_idx]
                    a_batch = actions[batch_idx]

                    pred_actions = self.policy(s_batch)
                    loss = F.mse_loss(pred_actions, a_batch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Decay mixing coefficient
            self.beta *= self.mixing_decay

            # Evaluate
            avg_return = self.evaluate()
            performance_history.append(avg_return)
            print(f"Iter {iteration+1}: Return={avg_return:.1f}, "
                  f"Dataset={len(self.dataset)}, Beta={self.beta:.3f}")

        return performance_history

    def evaluate(self, n_episodes=10):
        """Evaluate current policy without expert."""
        returns = []
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            ep_return = 0
            done = False

            while not done:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action = self.policy(state_t).squeeze(0).numpy()

                state, reward, terminated, truncated, _ = self.env.step(action)
                ep_return += reward
                done = terminated or truncated

            returns.append(ep_return)

        return np.mean(returns)
```

### 3.3 DAgger Variants

```
DAgger variants:

1. SafeDAgger: Only query expert when learner is uncertain
   -> Reduces expert queries significantly

2. EnsembleDAgger: Use ensemble disagreement for uncertainty
   -> Query expert when ensemble disagrees

3. ThriftyDAgger: Budget-constrained expert queries
   -> Fixed total number of expert queries

4. HG-DAgger: Human-gated DAgger
   -> Human decides when to intervene

5. LazyDAgger: Switch between expert and learner based on safety
```

---

## 4. Adversarial Imitation Learning

### 4.1 Beyond GAIL

```python
class AIRL(nn.Module):
    """
    Adversarial Inverse RL (AIRL).
    Unlike GAIL, AIRL recovers a transferable reward function.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, gamma=0.99):
        super().__init__()
        self.gamma = gamma

        # Reward function: r(s,a)
        self.reward = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Shaping function: Φ(s) for PBRS
        self.shaping = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, next_state, done):
        """
        AIRL discriminator: f(s,a,s') = r(s) + γΦ(s') - Φ(s)
        """
        r = self.reward(state)
        phi_s = self.shaping(state)
        phi_s_next = self.shaping(next_state)

        f = r + self.gamma * (1 - done.float().unsqueeze(1)) * phi_s_next - phi_s
        return f

    def get_reward(self, state):
        """Get the disentangled reward (transferable!)."""
        return self.reward(state)
```

### 4.2 ValueDICE

```python
class ValueDICE:
    """
    ValueDICE: Offline imitation learning without environment interaction.
    Estimates the divergence between expert and offline policy.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4):
        # Discriminator (nu function)
        self.nu = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.optimizer = torch.optim.Adam(self.nu.parameters(), lr=lr)

    def compute_loss(self, expert_states, expert_actions,
                     offline_states, offline_actions, offline_next_states,
                     gamma=0.99):
        """ValueDICE loss for offline imitation."""
        # Expert nu values
        expert_input = torch.cat([expert_states, expert_actions], dim=-1)
        expert_nu = self.nu(expert_input).mean()

        # Offline data terms
        offline_input = torch.cat([offline_states, offline_actions], dim=-1)
        current_nu = self.nu(offline_input)

        # This implements the Fenchel dual of f-divergence
        loss = expert_nu - torch.logsumexp(current_nu.squeeze(), dim=0)

        return loss
```

---

## 5. Observation-Only Imitation

### 5.1 Learning from Videos

```
Can we learn from watching videos? (No action labels!)

Challenge: Videos show WHAT happened, not HOW (no actions).

Approaches:
1. Inverse dynamics: Learn f(s_t, s_{t+1}) -> a_t from own experience
   Then infer actions for expert video

2. State-matching: Match state visitation distributions
   Don't need actions at all!

3. Temporal alignment: Learn correspondence between
   own experience and expert video
```

### 5.2 Inverse Dynamics for Action Inference

```python
class InverseDynamicsModel(nn.Module):
    """Predict action from state transition."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=-1)
        return self.net(x)


class ObservationOnlyIL:
    """Imitation learning from observation-only demonstrations."""

    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.inverse_model = InverseDynamicsModel(state_dim, action_dim)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.inv_optimizer = torch.optim.Adam(
            self.inverse_model.parameters(), lr=lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr
        )

    def train_inverse_model(self, own_transitions, epochs=50):
        """Train inverse dynamics on agent's own experience."""
        states = torch.FloatTensor(own_transitions['states'])
        next_states = torch.FloatTensor(own_transitions['next_states'])
        actions = torch.FloatTensor(own_transitions['actions'])

        for epoch in range(epochs):
            pred_actions = self.inverse_model(states, next_states)
            loss = F.mse_loss(pred_actions, actions)

            self.inv_optimizer.zero_grad()
            loss.backward()
            self.inv_optimizer.step()

    def infer_and_train(self, expert_states):
        """Infer actions from expert states, then BC."""
        states = torch.FloatTensor(expert_states[:-1])
        next_states = torch.FloatTensor(expert_states[1:])

        # Infer expert actions
        with torch.no_grad():
            inferred_actions = self.inverse_model(states, next_states)

        # Behavioral cloning on inferred actions
        pred_actions = self.policy(states)
        loss = F.mse_loss(pred_actions, inferred_actions)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()
```

---

## 6. Few-Shot and One-Shot Imitation

### 6.1 Meta-Imitation Learning

```
Goal: Learn to imitate from just ONE demonstration!

Approach:
  Train on MANY tasks, each with few demonstrations
  At test time: given one demo of new task, imitate it

Meta-learning for IL:
  During meta-training:
    Task i: given K demos, learn to imitate
    Loss: how well does policy perform on task i

  During meta-testing:
    New task: given 1 demo, adapt and perform
```

### 6.2 Task-Conditioned Policy

```python
class OneShotiImitationPolicy(nn.Module):
    """Policy that takes a demonstration as input."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Demo encoder: process the demonstration sequence
        self.demo_encoder = nn.GRU(
            state_dim + action_dim, hidden_dim, batch_first=True
        )

        # Policy: conditioned on demo embedding + current state
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def encode_demo(self, demo_states, demo_actions):
        """Encode demonstration into a task embedding."""
        demo_input = torch.cat([demo_states, demo_actions], dim=-1)
        _, h_n = self.demo_encoder(demo_input.unsqueeze(0))
        return h_n.squeeze(0)  # (hidden_dim,)

    def forward(self, state, demo_embedding):
        """Predict action given current state and task embedding."""
        x = torch.cat([state, demo_embedding.expand(len(state), -1)], dim=-1)
        return self.policy(x)
```

---

## 7. Practical Imitation Pipelines

### 7.1 When to Use What

```
Decision guide:

Can query expert at will?
├── Yes → DAgger (best theoretical guarantees)
│         Or: HG-DAgger if expert time is limited
└── No
    ├── Have action labels?
    │   ├── Small dataset (< 1000 demos)
    │   │   └── BC with data augmentation + ensembles
    │   └── Large dataset (> 10000 demos)
    │       └── BC works well; or GAIL if environment available
    └── No action labels (videos only)
        ├── Have own robot data?
        │   └── Inverse dynamics + BC
        └── No own data
            └── Very hard. State-matching or manual reward design
```

### 7.2 Evaluation Protocol

```python
def evaluate_imitation(agent, env, expert_fn, n_episodes=100):
    """Comprehensive evaluation of imitation learning agent."""
    agent_returns = []
    expert_returns = []
    action_errors = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        agent_return = 0
        expert_return = 0
        ep_action_errors = []
        done = False

        while not done:
            agent_action = agent.get_action(state)
            expert_action = expert_fn(state)

            action_error = np.linalg.norm(agent_action - expert_action)
            ep_action_errors.append(action_error)

            state, reward, terminated, truncated, _ = env.step(agent_action)
            agent_return += reward
            done = terminated or truncated

        agent_returns.append(agent_return)
        action_errors.append(np.mean(ep_action_errors))

    print(f"Agent return:  {np.mean(agent_returns):.1f} +/- {np.std(agent_returns):.1f}")
    print(f"Action error:  {np.mean(action_errors):.4f}")
    print(f"Expert return: {np.mean(expert_returns):.1f} (reference)")
    print(f"Normalized:    {np.mean(agent_returns)/np.mean(expert_returns)*100:.1f}%")
```

---

## 8. Exercises

### Exercise 1: BC vs DAgger Comparison

Compare behavioral cloning with DAgger:
1. Create an expert policy for CartPole (train with PPO to ~500 reward)
2. Collect 50 expert demonstrations for BC
3. Implement DAgger with the same total expert queries (50 episodes)
4. Compare: BC with all data upfront vs DAgger with iterative collection
5. Plot performance vs expert queries for both methods

### Exercise 2: Gaussian BC with Uncertainty

Build BC with uncertainty estimation:
1. Implement GaussianBCPolicy that predicts mean and std
2. Train on expert demonstrations
3. Visualize uncertainty: high uncertainty in unfamiliar states
4. Use uncertainty for active querying (query expert on uncertain states)
5. Compare with deterministic BC on out-of-distribution states

### Exercise 3: Observation-Only Imitation

Learn from expert videos (no action labels):
1. Train an inverse dynamics model from agent's random exploration
2. Record expert state trajectories (without actions)
3. Use inverse dynamics to infer actions, then BC
4. Compare with standard BC that has action labels
5. Measure: how does inverse model accuracy affect final policy?

### Exercise 4: DAgger with Budget Constraints

Implement budget-aware DAgger:
1. DAgger with a fixed budget of 100 expert queries total
2. Strategy 1: Query uniformly across iterations
3. Strategy 2: Query more in early iterations
4. Strategy 3: Query only when ensemble uncertainty is high
5. Compare all strategies on final policy performance

### Exercise 5: Multi-Task Imitation

Build a task-conditioned imitation learner:
1. Create 5 simple navigation tasks (different goal positions)
2. Collect 10 demonstrations per task
3. Train a demo-conditioned policy
4. Test on held-out tasks (new goal positions)
5. Compare with: (a) separate BC per task, (b) single BC on all data

---

*End of Lesson 26*
