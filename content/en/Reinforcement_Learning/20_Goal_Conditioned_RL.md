[Previous: Offline RL](./19_Offline_RL.md)

---

# 20. Goal-Conditioned Reinforcement Learning

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the goal-conditioned RL framework and universal value functions
2. Implement Hindsight Experience Replay (HER) for sparse reward environments
3. Build goal-conditioned policies with relabeling strategies
4. Understand multi-goal learning and goal representation design
5. Apply goal-conditioned methods to robotic manipulation tasks

---

## Table of Contents

1. [Goal-Conditioned Framework](#1-goal-conditioned-framework)
2. [Universal Value Function Approximators](#2-universal-value-function-approximators)
3. [Hindsight Experience Replay (HER)](#3-hindsight-experience-replay-her)
4. [Goal Relabeling Strategies](#4-goal-relabeling-strategies)
5. [Goal Representation Learning](#5-goal-representation-learning)
6. [Robotic Manipulation Applications](#6-robotic-manipulation-applications)
7. [Advanced Goal-Conditioned Methods](#7-advanced-goal-conditioned-methods)
8. [Exercises](#8-exercises)

---

## 1. Goal-Conditioned Framework

### 1.1 From Fixed to Variable Goals

Standard RL learns a single policy for a single objective. Goal-conditioned RL learns a policy that can achieve *any* specified goal.

```
Standard RL:
  π(a | s)          Policy depends only on state
  Goal is implicit (maximize cumulative reward)

Goal-Conditioned RL:
  π(a | s, g)       Policy depends on state AND goal
  One policy can achieve many different goals!

Example: Robot arm
  Standard: Learn to reach position (5, 3)
  Goal-Conditioned: Learn to reach ANY position (x, y)

  π(a | s, g=(5,3)) -> actions to reach (5,3)
  π(a | s, g=(1,7)) -> actions to reach (1,7)
  Same policy, different goals!
```

### 1.2 Goal-Conditioned MDP

```
Standard MDP:    (S, A, T, R, γ)
Goal-Conditioned: (S, A, G, T, R_g, γ)

Where:
  G = goal space (could be same as state space or different)
  R_g(s, a, g) = reward function that depends on goal

Common reward functions:
  Sparse:  R(s, a, g) = 1 if ||s' - g|| < ε,  0 otherwise
  Dense:   R(s, a, g) = -||s' - g||₂           (negative distance)
  Binary:  R(s, a, g) = -1 if not at goal       (penalty per step)
```

### 1.3 Why Goal Conditioning Matters

```python
import numpy as np

def demonstrate_gc_advantage():
    """Show sample efficiency of goal-conditioned learning."""
    # Without GC: Need to learn separate policy for each goal
    n_goals = 100
    episodes_per_goal = 1000
    total_standard = n_goals * episodes_per_goal  # 100,000 episodes

    # With GC: Single policy learns all goals simultaneously
    # + HER multiplies learning signal from each episode
    total_gc = 10000  # 10,000 episodes, HER provides multi-goal learning
    her_multiplier = 4  # 4 relabeled goals per real episode
    effective_gc = total_gc * (1 + her_multiplier)  # 50,000 effective episodes

    print(f"Standard approach: {total_standard:,} episodes")
    print(f"GC + HER approach: {total_gc:,} episodes "
          f"({effective_gc:,} effective)")
    print(f"Efficiency gain: {total_standard / total_gc:.0f}x")

demonstrate_gc_advantage()
```

---

## 2. Universal Value Function Approximators

### 2.1 UVFA Architecture

Universal Value Function Approximators (UVFA) extend Q-functions to condition on goals:

```
Standard Q:  Q(s, a)     -> scalar value
UVFA Q:      Q(s, a, g)  -> scalar value (conditioned on goal)

Architecture options:

Option A: Concatenation
  [s, a, g] -> MLP -> Q(s,a,g)

Option B: Separate encoders + combination
  s -> encoder_s -> φ(s)
  g -> encoder_g -> ψ(g)     -> combine -> Q
  a -> encoder_a -> α(a)

Option C: Relational (attention-based)
  s, g -> cross-attention -> Q
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalConditionedQNetwork(nn.Module):
    """Q-network conditioned on goals."""

    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.net(x)


class GoalConditionedPolicy(nn.Module):
    """Deterministic policy conditioned on goals."""

    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim=256,
                 max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        return self.net(x) * self.max_action
```

### 2.2 Training UVFA

```python
class GoalConditionedDDPG:
    """DDPG agent with goal conditioning."""

    def __init__(self, state_dim, action_dim, goal_dim,
                 hidden_dim=256, lr=1e-3, gamma=0.98, tau=0.005):
        self.gamma = gamma
        self.tau = tau

        self.actor = GoalConditionedPolicy(state_dim, goal_dim, action_dim, hidden_dim)
        self.critic = GoalConditionedQNetwork(state_dim, action_dim, goal_dim, hidden_dim)
        self.target_actor = GoalConditionedPolicy(state_dim, goal_dim, action_dim, hidden_dim)
        self.target_critic = GoalConditionedQNetwork(state_dim, action_dim, goal_dim, hidden_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, goal, noise=0.1):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        goal_t = torch.FloatTensor(goal).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_t, goal_t).squeeze(0).numpy()

        action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def train_step(self, states, actions, rewards, next_states, goals, dones):
        """Train on a batch with goal information."""
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        goals = torch.FloatTensor(goals)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Critic update
        with torch.no_grad():
            next_actions = self.target_actor(next_states, goals)
            target_q = self.target_critic(next_states, next_actions, goals)
            target = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(states, actions, goals)
        critic_loss = F.mse_loss(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        pred_actions = self.actor(states, goals)
        actor_loss = -self.critic(states, pred_actions, goals).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft target update
        for p, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return critic_loss.item(), actor_loss.item()
```

---

## 3. Hindsight Experience Replay (HER)

### 3.1 The Sparse Reward Problem

```
The core problem:
  Goal: reach position (5, 3)
  Reward: +1 only when ||position - (5,3)|| < 0.1
  Otherwise: 0

  With random exploration, probability of accidentally reaching goal ≈ 0
  No reward signal -> no learning!

  Even with 1 million episodes, the agent may never reach (5,3).
```

### 3.2 The HER Insight

**Key insight**: Even a failed episode teaches us something. If we tried to reach (5,3) but ended at (2,7), we can relabel the goal to (2,7) and say "we successfully reached (2,7)!"

```
Original experience:
  Goal: (5,3)
  Trajectory: s₀ -> s₁ -> s₂ -> s₃ = (2,7)
  Reward: 0, 0, 0, 0 (never reached (5,3))
  Learning signal: NONE

HER relabeling:
  Relabeled goal: (2,7)  [where we actually ended up]
  Trajectory: s₀ -> s₁ -> s₂ -> s₃ = (2,7)
  Reward: 0, 0, 0, 1 (reached the relabeled goal!)
  Learning signal: Learns how to reach (2,7)

  Over many relabeled episodes, the agent learns to reach MANY goals.
  Eventually, it learns to reach the actual desired goal too!
```

### 3.3 HER Implementation

```python
class HindsightExperienceReplay:
    """HER buffer that augments episodes with hindsight goals."""

    def __init__(self, capacity=1_000_000, goal_strategy='future',
                 n_sampled_goals=4, reward_fn=None):
        self.capacity = capacity
        self.goal_strategy = goal_strategy
        self.n_sampled_goals = n_sampled_goals
        self.reward_fn = reward_fn or self._default_reward

        self.episodes = []
        self.transitions = []  # flat list for sampling

    @staticmethod
    def _default_reward(achieved_goal, desired_goal, threshold=0.05):
        """Sparse reward: 0 if achieved, -1 otherwise."""
        dist = np.linalg.norm(achieved_goal - desired_goal)
        return 0.0 if dist < threshold else -1.0

    def store_episode(self, episode):
        """
        Store an episode and generate HER relabeled transitions.

        episode: list of dicts with keys:
            'state', 'action', 'next_state', 'achieved_goal',
            'desired_goal', 'done'
        """
        T = len(episode)

        # Store original transitions
        for t, transition in enumerate(episode):
            self.transitions.append({
                'state': transition['state'],
                'action': transition['action'],
                'reward': self.reward_fn(
                    transition['achieved_goal'],
                    transition['desired_goal']
                ),
                'next_state': transition['next_state'],
                'goal': transition['desired_goal'],
                'done': transition['done'],
            })

            # Generate HER goals
            her_goals = self._sample_her_goals(episode, t)

            for goal in her_goals:
                her_reward = self.reward_fn(
                    transition['achieved_goal'], goal
                )
                her_done = (her_reward == 0.0)

                self.transitions.append({
                    'state': transition['state'],
                    'action': transition['action'],
                    'reward': her_reward,
                    'next_state': transition['next_state'],
                    'goal': goal,
                    'done': her_done,
                })

        # Trim if over capacity
        if len(self.transitions) > self.capacity:
            self.transitions = self.transitions[-self.capacity:]

    def _sample_her_goals(self, episode, current_idx):
        """Sample goals using the specified strategy."""
        T = len(episode)
        goals = []

        if self.goal_strategy == 'future':
            # Sample from future achieved goals in this episode
            future_indices = list(range(current_idx + 1, T))
            if not future_indices:
                return goals

            n = min(self.n_sampled_goals, len(future_indices))
            selected = np.random.choice(future_indices, n, replace=False)

            for idx in selected:
                goals.append(episode[idx]['achieved_goal'].copy())

        elif self.goal_strategy == 'final':
            # Use the final achieved goal
            goals.append(episode[-1]['achieved_goal'].copy())

        elif self.goal_strategy == 'episode':
            # Sample from any achieved goal in the episode
            indices = np.random.randint(0, T, self.n_sampled_goals)
            for idx in indices:
                goals.append(episode[idx]['achieved_goal'].copy())

        return goals

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = np.random.randint(0, len(self.transitions), batch_size)
        batch = [self.transitions[i] for i in indices]

        return {
            'states': np.array([t['state'] for t in batch]),
            'actions': np.array([t['action'] for t in batch]),
            'rewards': np.array([t['reward'] for t in batch]),
            'next_states': np.array([t['next_state'] for t in batch]),
            'goals': np.array([t['goal'] for t in batch]),
            'dones': np.array([t['done'] for t in batch], dtype=float),
        }
```

### 3.4 Training Loop with HER

```python
def train_with_her(env, agent, n_epochs=50, n_cycles=50,
                   n_episodes=16, n_train_batches=40, batch_size=256):
    """Training loop for goal-conditioned agent with HER."""
    her_buffer = HindsightExperienceReplay(
        goal_strategy='future',
        n_sampled_goals=4
    )

    success_rates = []

    for epoch in range(n_epochs):
        epoch_successes = 0
        epoch_total = 0

        for cycle in range(n_cycles):
            # Collect episodes
            for _ in range(n_episodes):
                episode = collect_episode(env, agent)
                her_buffer.store_episode(episode)

                # Check if goal was achieved
                final_dist = np.linalg.norm(
                    episode[-1]['achieved_goal'] - episode[0]['desired_goal']
                )
                epoch_successes += int(final_dist < 0.05)
                epoch_total += 1

            # Train on batches
            for _ in range(n_train_batches):
                batch = her_buffer.sample(batch_size)
                agent.train_step(
                    batch['states'], batch['actions'],
                    batch['rewards'], batch['next_states'],
                    batch['goals'], batch['dones']
                )

        success_rate = epoch_successes / epoch_total
        success_rates.append(success_rate)
        print(f"Epoch {epoch+1}/{n_epochs}, Success Rate: {success_rate:.2%}")

    return success_rates


def collect_episode(env, agent, max_steps=50):
    """Collect one episode with goal information."""
    obs = env.reset()
    state = obs['observation']
    desired_goal = obs['desired_goal']
    episode = []

    for step in range(max_steps):
        action = agent.select_action(state, desired_goal)
        next_obs, reward, terminated, truncated, info = env.step(action)

        episode.append({
            'state': state.copy(),
            'action': action.copy(),
            'next_state': next_obs['observation'].copy(),
            'achieved_goal': next_obs['achieved_goal'].copy(),
            'desired_goal': desired_goal.copy(),
            'done': terminated or truncated,
        })

        state = next_obs['observation']
        if terminated or truncated:
            break

    return episode
```

---

## 4. Goal Relabeling Strategies

### 4.1 Strategy Comparison

```
HER Goal Relabeling Strategies:

1. 'future' (default, best performance):
   Select goals from future states in the same episode
   Pro: Most informative (learns forward progress)
   Con: Biased toward end of episode

2. 'final':
   Always use the final state as the relabeled goal
   Pro: Simple, always provides positive signal
   Con: Less diverse goals

3. 'episode':
   Sample from any state in the episode
   Pro: Maximum diversity
   Con: May relabel with already-achieved goals (less informative)

4. 'random':
   Sample from any previously seen achieved goal
   Pro: Diverse goals across episodes
   Con: May be too far from current trajectory

Performance ranking (typical):
  future > episode > final > random
```

### 4.2 Curriculum-Based Goal Relabeling

```python
class CurriculumHER(HindsightExperienceReplay):
    """HER with curriculum-based goal selection."""

    def __init__(self, *args, initial_difficulty=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.difficulty = initial_difficulty  # 0=easy, 1=hard

    def _sample_her_goals(self, episode, current_idx):
        """Sample goals with distance-based curriculum."""
        T = len(episode)
        goals = []

        # Mix of nearby (easy) and far (hard) goals
        for _ in range(self.n_sampled_goals):
            if np.random.random() < self.difficulty:
                # Hard: sample from late in episode (far future)
                idx = np.random.randint(max(current_idx + 1, T // 2), T)
            else:
                # Easy: sample from near future
                max_idx = min(current_idx + 5, T)
                idx = np.random.randint(current_idx + 1, max(max_idx, current_idx + 2))

            idx = min(idx, T - 1)
            goals.append(episode[idx]['achieved_goal'].copy())

        return goals

    def increase_difficulty(self, success_rate, threshold=0.7):
        """Increase difficulty when agent is succeeding."""
        if success_rate > threshold:
            self.difficulty = min(1.0, self.difficulty + 0.1)
            print(f"Difficulty increased to {self.difficulty:.1f}")
```

---

## 5. Goal Representation Learning

### 5.1 State-Based vs Learned Goals

```
Goal representations:

1. State-based (simple):
   g = desired state (or subset of state)
   Works when goal space = state space
   Example: g = (x, y) target position

2. Image-based:
   g = target image showing desired configuration
   Requires learning goal embeddings

3. Language-based:
   g = "put the red block on the blue block"
   Requires language grounding

4. Learned latent goals:
   g = z ∈ R^d, learned representation
   Can capture abstract goals
```

### 5.2 Contrastive Goal Representations

```python
class ContrastiveGoalEncoder(nn.Module):
    """Learn goal representations using contrastive learning."""

    def __init__(self, state_dim, embedding_dim=64, temperature=0.1):
        super().__init__()
        self.temperature = temperature

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, state):
        return F.normalize(self.encoder(state), dim=-1)

    def contrastive_loss(self, states, goals, negative_goals):
        """
        InfoNCE loss for goal representation learning.
        Positive pairs: (state, achieved_goal) from same trajectory
        Negative pairs: (state, random_goal) from different trajectories
        """
        state_embed = self.forward(states)        # (B, D)
        goal_embed = self.forward(goals)           # (B, D)
        neg_embed = self.forward(negative_goals)   # (B, K, D)

        # Positive similarity
        pos_sim = (state_embed * goal_embed).sum(dim=-1) / self.temperature

        # Negative similarity
        neg_sim = torch.bmm(
            neg_embed, state_embed.unsqueeze(-1)
        ).squeeze(-1) / self.temperature  # (B, K)

        # InfoNCE
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(len(states), dtype=torch.long, device=states.device)
        loss = F.cross_entropy(logits, labels)

        return loss
```

---

## 6. Robotic Manipulation Applications

### 6.1 Gymnasium Robotics Environments

```python
# Goal-conditioned environments from Gymnasium Robotics
# pip install gymnasium-robotics

import gymnasium as gym

# FetchReach: Move gripper to target position
env = gym.make('FetchReach-v3')

# FetchPush: Push a block to target position
env = gym.make('FetchPush-v3')

# FetchSlide: Slide a puck to target (beyond reach)
env = gym.make('FetchSlide-v3')

# FetchPickAndPlace: Pick up and place object
env = gym.make('FetchPickAndPlace-v3')

# Observation structure:
obs, info = env.reset()
print(f"Observation: {obs['observation'].shape}")      # robot state
print(f"Achieved goal: {obs['achieved_goal'].shape}")   # current object pos
print(f"Desired goal: {obs['desired_goal'].shape}")     # target position
```

### 6.2 FetchReach Example

```python
def train_fetch_reach():
    """Train goal-conditioned agent on FetchReach."""
    env = gym.make('FetchReach-v3')

    obs, _ = env.reset()
    state_dim = obs['observation'].shape[0]
    goal_dim = obs['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]

    agent = GoalConditionedDDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        lr=1e-3,
        gamma=0.98,
    )

    success_rates = train_with_her(
        env, agent,
        n_epochs=50,
        n_cycles=50,
        n_episodes=16,
    )

    print(f"Final success rate: {success_rates[-1]:.2%}")
    return agent, success_rates
```

### 6.3 Multi-Goal Evaluation

```python
def evaluate_multi_goal(agent, env, n_goals=100):
    """Evaluate agent on multiple random goals."""
    successes = 0

    for _ in range(n_goals):
        obs, _ = env.reset()
        state = obs['observation']
        goal = obs['desired_goal']

        for step in range(50):
            action = agent.select_action(state, goal, noise=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            state = obs['observation']

            if info.get('is_success', False):
                successes += 1
                break

            if terminated or truncated:
                break

    success_rate = successes / n_goals
    print(f"Multi-goal success rate: {success_rate:.2%} ({successes}/{n_goals})")
    return success_rate
```

---

## 7. Advanced Goal-Conditioned Methods

### 7.1 Automatic Goal Generation

```python
class GoalGAN:
    """Generate goals at the frontier of agent's capability."""

    def __init__(self, goal_dim, hidden_dim=128, noise_dim=4):
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.noise_dim = noise_dim

    def generate_goals(self, n_goals):
        """Generate goals at appropriate difficulty."""
        noise = torch.randn(n_goals, self.noise_dim)
        with torch.no_grad():
            goals = self.generator(noise)
        return goals.numpy()

    def label_goals(self, goals, agent, env, n_attempts=5):
        """Label goals by whether agent can achieve them."""
        labels = []
        for goal in goals:
            successes = 0
            for _ in range(n_attempts):
                success = attempt_goal(agent, env, goal)
                successes += int(success)

            # GOID (Goals of Intermediate Difficulty)
            success_rate = successes / n_attempts
            # Label 1 if goal is at the frontier (achievable ~50% of time)
            labels.append(0.2 < success_rate < 0.8)

        return np.array(labels, dtype=float)
```

### 7.2 Hindsight Goal Ranking (HGR)

```python
def hindsight_goal_ranking(episode, n_goals=4, method='energy'):
    """
    Select the most informative hindsight goals.
    Instead of random future goals, rank by learning utility.
    """
    T = len(episode)
    achieved_goals = [ep['achieved_goal'] for ep in episode]

    if method == 'energy':
        # Prefer goals that are diverse and at medium distance
        scores = []
        for i in range(T):
            diversity = np.mean([
                np.linalg.norm(achieved_goals[i] - achieved_goals[j])
                for j in range(T) if j != i
            ])
            scores.append(diversity)

        # Select top-k diverse goals
        top_indices = np.argsort(scores)[-n_goals:]
        return [achieved_goals[i] for i in top_indices]

    elif method == 'td_error':
        # Prefer goals where the agent has high TD error
        # (most learning potential)
        pass

    return [achieved_goals[i] for i in
            np.random.choice(T, min(n_goals, T), replace=False)]
```

### 7.3 RIG: Reinforcement Learning with Imagined Goals

```
RIG framework:
1. Train a VAE on observed states
2. Sample latent goals z ~ prior
3. Decode z to goal state for visualization
4. Train goal-conditioned policy in latent space

Benefits:
- Can imagine goals never seen before
- Compact goal representation
- Works with image observations

Pipeline:
  Image obs -> VAE encoder -> z_current
  Sample z_goal from prior
  Policy: π(a | z_current, z_goal)
```

---

## 8. Exercises

### Exercise 1: Implement HER from Scratch

Build HER for a simple 2D reaching task:
1. Create a 2D environment where agent moves to target position
2. Use sparse binary reward (success threshold = 0.05)
3. Implement HER with 'future' strategy and n_sampled_goals=4
4. Compare learning curves: with HER vs without HER
5. Show that without HER, the agent never learns (sparse reward too hard)

### Exercise 2: Goal Relabeling Strategy Comparison

Systematically compare HER strategies:
1. Implement all four strategies: future, final, episode, random
2. Train each on the same reaching environment for 100 epochs
3. Plot success rate curves for all strategies
4. Measure goal diversity for each strategy (distribution of relabeled goals)
5. Explain why 'future' typically works best

### Exercise 3: Goal-Conditioned DDPG on FetchReach

Build the full goal-conditioned DDPG pipeline:
1. Set up FetchReach-v3 environment from Gymnasium Robotics
2. Implement GoalConditionedDDPG with twin critics
3. Integrate HER buffer with 'future' strategy
4. Train and report success rate over 50 epochs
5. Visualize the learned policy: plot trajectories to different goals

### Exercise 4: Curriculum Goal Generation

Implement automatic curriculum for goal difficulty:
1. Start with easy goals (close to initial state)
2. Gradually increase goal distance as agent improves
3. Track success rate at each difficulty level
4. Compare with uniform goal sampling
5. Show that curriculum leads to faster learning on hard goals

### Exercise 5: Multi-Goal Transfer

Demonstrate transfer between goal-conditioned tasks:
1. Train agent on FetchReach (gripper positioning)
2. Transfer the goal-conditioned policy to FetchPush
3. Measure: how much does pre-training help?
4. Fine-tune on FetchPush with HER
5. Compare learning curves: from scratch vs with transfer

---

*End of Lesson 20*
