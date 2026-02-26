[Previous: Model Saving and Deployment](./41_Model_Saving_Deployment.md)

---

# 42. Reinforcement Learning Introduction

## Learning Objectives

- Understand basic concepts and terminology of reinforcement learning
- MDP (Markov Decision Process) framework
- Q-Learning and Value-based methods
- Policy Gradient overview
- Deep RL basics (DQN)
- PyTorch implementation and practice

---

## 1. Reinforcement Learning Overview

### Definition and Characteristics

```
Reinforcement Learning: Agent learns to maximize rewards through environment interaction

Characteristics:
1. Trial and Error Learning
2. Delayed Reward
3. Exploration-Exploitation tradeoff
4. Sequential Decision Making
```

### Supervised Learning vs Reinforcement Learning

```
┌─────────────────────────────────────────────────────────────┐
│           Supervised Learning vs Reinforcement Learning      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Supervised Learning                                         │
│  ┌─────────┐   answer    ┌─────────┐                        │
│  │ Input x │ ─────────→ │ Label y │                        │
│  └─────────┘             └─────────┘                        │
│  Immediate feedback, correct answer provided                 │
│                                                              │
│  Reinforcement Learning                                      │
│  ┌─────────┐  action  ┌─────────┐  reward  ┌─────────┐     │
│  │ State s │ ──────→ │ Action a│ ──────→ │ Reward r│     │
│  └─────────┘         └─────────┘         └─────────┘     │
│       ↑                    │                   │             │
│       └────────────────────┴───────────────────┘             │
│  Delayed feedback, exploration required                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Applications of Reinforcement Learning

```
Games: AlphaGo, Atari, StarCraft II
Robotics: Robot control, autonomous driving
Finance: Portfolio optimization, algorithmic trading
Recommendation: Personalized recommendations, dialogue systems
Resource management: Datacenter cooling, network optimization
```

---

## 2. MDP (Markov Decision Process)

### Components

```
MDP = (S, A, P, R, γ)

S: State (state set)
   - Environment state observed by agent
   - e.g., game screen, robot position/velocity

A: Action (action set)
   - Actions agent can take
   - e.g., move up/down/left/right, motor torque

P: Transition Probability
   - P(s'|s, a): probability of transitioning to s' when taking action a in state s

R: Reward (reward function)
   - R(s, a, s'): reward received during state transition

γ: Discount Factor
   - Present value of future rewards (0 < γ ≤ 1)
```

### Markov Property

```
Future depends only on current state (independent of past):

P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, ..., s_t, a_t)

Meaning: Current state contains sufficient information
```

### Interaction Loop

```python
# RL basic loop
def rl_loop(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 1. Agent selects action
            action = agent.select_action(state)

            # 2. Environment returns next state and reward
            next_state, reward, done, info = env.step(action)

            # 3. Agent learns from experience
            agent.learn(state, action, reward, next_state, done)

            # 4. Update state
            state = next_state
            total_reward += reward

        print(f"Episode {episode}: Total Reward = {total_reward}")
```

---

## 3. Value Functions

### State Value Function (V)

```
V^π(s) = E[G_t | S_t = s, π]

G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
    = Σ_{k=0}^∞ γ^k R_{t+k+1}

Meaning: Expected cumulative reward when following policy π from state s
```

### Action Value Function (Q)

```
Q^π(s, a) = E[G_t | S_t = s, A_t = a, π]

Meaning: Expected reward when taking action a in state s, then following π
```

### Bellman Equation

```python
# Bellman equation (core!)

# Value Function
V(s) = max_a [ R(s, a) + γ * Σ P(s'|s,a) * V(s') ]

# Q Function
Q(s, a) = R(s, a) + γ * Σ P(s'|s,a) * max_a' Q(s', a')

# Meaning: Current value = immediate reward + discounted future value
```

---

## 4. Q-Learning

### Algorithm Overview

```
Q-Learning: Model-free, Off-policy algorithm

Features:
1. No environment model (P) required
2. Learn optimal policy from data collected with different policy (ε-greedy)
3. Store Q values in table form
```

### Q-Learning Update

```python
# Q-Learning update rule

Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

# Breakdown:
# TD Target: r + γ * max_a' Q(s', a')  (target)
# TD Error: TD Target - Q(s, a)        (error)
# α: Learning Rate
```

### PyTorch Implementation

```python
import numpy as np

class QLearningAgent:
    """Q-Learning Agent (Tabular) (⭐⭐⭐)"""
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Initialize Q-Table
        self.q_table = np.zeros((state_size, action_size))

    def select_action(self, state):
        """ε-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)  # explore
        else:
            return np.argmax(self.q_table[state])       # exploit

    def learn(self, state, action, reward, next_state, done):
        """Update Q-Table"""
        # TD Target
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])

        # TD Error
        td_error = td_target - self.q_table[state, action]

        # Update
        self.q_table[state, action] += self.lr * td_error

        # Epsilon Decay
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """Return learned policy"""
        return np.argmax(self.q_table, axis=1)
```

---

## 5. Deep Q-Network (DQN)

### Core Idea

```
Problem: Q-Table infeasible for large state spaces
Solution: Approximate Q(s, a) with neural network

Q(s, a; θ) ≈ Q*(s, a)

Key techniques:
1. Experience Replay: Improve efficiency by reusing experiences
2. Target Network: Stabilize training
```

### DQN Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DQN Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  State s                                                     │
│     │                                                        │
│     ▼                                                        │
│  ┌─────────────────────────────────────┐                    │
│  │  Neural Network (CNN/MLP)           │                    │
│  │  Input: State s                     │                    │
│  │  Output: Q(s, a) for all actions    │                    │
│  └─────────────────────────────────────┘                    │
│     │                                                        │
│     ▼                                                        │
│  [Q(s, a_1), Q(s, a_2), ..., Q(s, a_n)]                     │
│     │                                                        │
│     ▼                                                        │
│  Action = argmax_a Q(s, a)                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Experience Replay

```python
from collections import deque
import random

class ReplayBuffer:
    """Experience Replay Buffer (⭐⭐⭐)"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store experience"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Random sampling — why random?
        Consecutive experiences are highly correlated (similar states,
        similar actions).  Training on correlated batches causes the
        network to overfit to recent experience and forget earlier
        lessons.  Random sampling breaks these temporal correlations,
        producing i.i.d.-like mini-batches that stabilize SGD."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)
```

### DQN PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class QNetwork(nn.Module):
    """Q-Network (MLP) (⭐⭐⭐)"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN Agent (⭐⭐⭐⭐)"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.learn_step = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Q-Networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = copy.deepcopy(self.q_network)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        """ε-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """DQN training"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (with target network)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss and update
        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()
```

### DQN Training Loop

```python
def train_dqn(env, agent, episodes=500):
    """DQN Training Loop (⭐⭐⭐)"""
    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):  # gym 0.26+
            state = state[0]

        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            result = env.step(action)
            if len(result) == 5:  # gym 0.26+
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_state, reward, done, info = result

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn()

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

    return rewards_history
```

---

## 6. Policy Gradient

### Idea

```
Value-based (DQN): Learn Q function → Derive policy indirectly
Policy-based: Learn policy directly

Policy = π_θ(a|s) = P(a|s; θ)

Advantages:
1. Handle continuous action spaces
2. Learn stochastic policies
3. Guaranteed convergence (to local optimum)
```

### Policy Gradient Theorem

```python
# Objective: maximize J(θ) = E[Σ R_t]

# Gradient:
∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t|s_t) * G_t ]

# G_t: Cumulative reward (Return) from time t
# log π_θ: Log probability of policy
```

### REINFORCE Algorithm

```python
class PolicyNetwork(nn.Module):
    """Policy Network (⭐⭐⭐)"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class REINFORCEAgent:
    """REINFORCE (Monte Carlo Policy Gradient) (⭐⭐⭐⭐)"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Episode buffer
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """Stochastic action selection"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state)

        # Sample from Categorical distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))

        return action.item()

    def store_reward(self, reward):
        """Store reward"""
        self.rewards.append(reward)

    def learn(self):
        """Learn at end of episode"""
        # Compute returns (from end to beginning)
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize (baseline effect)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy Gradient loss
        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss -= log_prob * G  # negative: gradient ascent

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffer
        self.log_probs = []
        self.rewards = []

        return loss.item()
```

### REINFORCE Training

```python
def train_reinforce(env, agent, episodes=1000):
    """REINFORCE Training (⭐⭐⭐)"""
    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            agent.store_reward(reward)

            state = next_state
            total_reward += reward

        # Learn at episode end
        agent.learn()
        rewards_history.append(total_reward)

        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")

    return rewards_history
```

---

## 7. Actor-Critic

### Idea

```
REINFORCE problem: High variance (Monte Carlo estimation)
Solution: Estimate value with Critic → reduce variance

Actor: Policy π_θ (decide actions)
Critic: Value V_φ (evaluate states)
```

### Advantage Function

```python
# Advantage = Q(s,a) - V(s)
# Meaning: How much better is this action compared to average

# Estimated with TD Error:
A(s, a) ≈ r + γV(s') - V(s)
```

### Actor-Critic Implementation

```python
class ActorCritic(nn.Module):
    """Actor-Critic Network (⭐⭐⭐⭐)"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (Policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic (Value)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        policy = self.actor(features)
        value = self.critic(features)
        return policy, value


class A2CAgent:
    """Advantage Actor-Critic (⭐⭐⭐⭐)"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        policy, _ = self.network(state)
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def learn(self, state, action, reward, next_state, done, log_prob):
        """One-step Actor-Critic Update"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        _, value = self.network(state)
        _, next_value = self.network(next_state)

        # TD Target and Advantage
        td_target = reward + self.gamma * next_value * (1 - done)
        advantage = td_target - value

        # Actor Loss (policy gradient with advantage)
        actor_loss = -log_prob * advantage.detach()

        # Critic Loss (value function)
        critic_loss = advantage.pow(2)

        # Total Loss
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

---

## 8. Environments and Experiments

### Using OpenAI Gym

```python
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1')

# Environment info
print(f"State space: {env.observation_space}")      # Box(4,)
print(f"Action space: {env.action_space}")          # Discrete(2)
print(f"State dim: {env.observation_space.shape}")  # (4,)
print(f"Action dim: {env.action_space.n}")          # 2

# Run episode
state, _ = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # random action
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state

env.close()
```

### Experiment Example: CartPole

```python
def run_experiment():
    """CartPole experiment (⭐⭐⭐)"""
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n              # 2

    # DQN agent
    agent = DQNAgent(state_dim, action_dim)

    # Training
    rewards = train_dqn(env, agent, episodes=500)

    # Visualize results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN on CartPole-v1')

    # Moving average
    window = 50
    ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards)), ma, 'r-', linewidth=2)

    plt.savefig('cartpole_dqn.png')
    plt.close()

    env.close()
    return rewards
```

---

## 9. Algorithm Comparison

### Main Algorithm Characteristics

| Algorithm | Type | On/Off-Policy | Features |
|-----------|------|---------------|----------|
| Q-Learning | Value-based | Off-policy | Table, simple |
| DQN | Value-based | Off-policy | Neural network, experience replay |
| REINFORCE | Policy-based | On-policy | Monte Carlo, high variance |
| A2C/A3C | Actor-Critic | On-policy | Advantage, parallelization |
| PPO | Actor-Critic | On-policy | Stable, practical |
| SAC | Actor-Critic | Off-policy | Continuous actions, entropy |

### Selection Guide

```
Discrete action space:
- Simple problems: DQN
- Complex problems: PPO

Continuous action space:
- Stable: SAC
- Fast learning: PPO

Resource constraints:
- A2C (single machine)

Large-scale parallel:
- A3C, PPO
```

---

## 10. Advanced Topics Overview

### Double DQN

```python
# DQN problem: Q value overestimation
# Solution: Use different networks for action selection and evaluation

# Original DQN:
target_q = reward + gamma * target_net(next_state).max()

# Double DQN:
best_action = q_net(next_state).argmax()
target_q = reward + gamma * target_net(next_state)[best_action]
```

### Dueling DQN

```python
# Q = V + A (Value + Advantage)
# Separate state value and action advantage

class DuelingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, action_dim)

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q
```

### Prioritized Experience Replay

```python
# Sample important experiences (high TD Error) more frequently
# P(i) ∝ |TD_error_i|^α

# Use Sum Tree data structure for implementation
```

---

## Summary

### Key Concepts

1. **MDP**: Define problem with states, actions, rewards, transitions
2. **Bellman Equation**: Current value = immediate reward + future value
3. **Q-Learning**: Learn Q function with TD
4. **DQN**: Neural network + experience replay + target network
5. **Policy Gradient**: Direct policy optimization
6. **Actor-Critic**: Actor + Critic to reduce variance

### Practical Tips

```python
# 1. Reward design is key
# - Sparse reward → difficult learning
# - Shaped reward → helps learning (but can bias)

# 2. Hyperparameter tuning
# - Learning rate: 1e-4 ~ 1e-3
# - Gamma: 0.99
# - Epsilon decay: slowly

# 3. Debugging
# - Check reward curve
# - Monitor Q value distribution
# - Visualize learned policy
```

---

## Exercises

### Exercise 1: Implement Q-Learning on FrozenLake

Using the `QLearningAgent` and the `FrozenLake-v1` environment from Gymnasium:
1. The FrozenLake grid has 16 states and 4 actions. Initialize the Q-table as all zeros.
2. Train for 10,000 episodes with `lr=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995`.
3. After training, print the learned Q-table (4×16 matrix) and visualize the greedy policy as arrows on the 4×4 grid.
4. Compute the success rate (fraction of episodes reaching the goal) over the last 1,000 episodes. What does it mean if the success rate is near 0.7? Is Q-learning guaranteed to find the optimal policy for this environment?

### Exercise 2: Train DQN on CartPole-v1 and Measure Convergence

Using the provided `DQNAgent` and `train_dqn`:
1. Train on `CartPole-v1` for 500 episodes with default hyperparameters.
2. Plot the episode reward curve with a 50-episode moving average overlay.
3. CartPole is "solved" when the moving average reaches 475. How many episodes does your DQN need?
4. Experiment: remove the target network (set `target_update=1`, which copies weights every step). How does training stability change? Explain why the target network is necessary using the concepts of a "moving target" in supervised learning.

### Exercise 3: Ablation Study on Experience Replay

Investigate the importance of the replay buffer:
1. Train DQN on CartPole with the full replay buffer (capacity=10,000, random sampling).
2. Train DQN without experience replay: process each transition immediately and discard it (batch_size=1, buffer as a deque of size 1).
3. Plot both reward curves and compare final performance and convergence speed.
4. Explain the instability in case (2) using the concept of correlated samples: why does training on consecutive transitions violate the i.i.d. assumption required by SGD?

### Exercise 4: Implement Double DQN

The standard DQN overestimates Q-values because the same network selects and evaluates the best next action. Double DQN decouples these steps:
1. In the `DQNAgent.learn` method, modify the target computation:
   - Use `q_network` to select the best next action: `best_action = q_network(next_states).argmax(dim=1)`.
   - Use `target_network` to evaluate it: `next_q = target_network(next_states).gather(1, best_action.unsqueeze(1)).squeeze(1)`.
2. Train both standard DQN and Double DQN on CartPole for 500 episodes.
3. Compare their average Q-value estimates during training (log `current_q.mean().item()` per episode). Does Double DQN produce lower, more accurate Q estimates?

### Exercise 5: Train REINFORCE on CartPole and Compare with DQN

Using the provided `REINFORCEAgent`:
1. Train on `CartPole-v1` for 1,000 episodes.
2. Plot the episode reward curve alongside the DQN curve from Exercise 2.
3. REINFORCE is typically more unstable than DQN. Observe and explain: what causes the high variance in REINFORCE's reward curve? How does the return normalization step (`(returns - returns.mean()) / returns.std()`) act as a baseline to reduce variance?
4. Implement a simple Actor-Critic by adding a baseline: instead of using the full return `G_t`, subtract the value estimate `V(s_t)` predicted by a small value network. Verify that this reduces the variance of the gradient estimates.

---

## References

- Sutton & Barto: http://incompleteideas.net/book/the-book.html
- DQN: https://arxiv.org/abs/1312.5602
- Policy Gradient: https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
- OpenAI Spinning Up: https://spinningup.openai.com/
- Gymnasium: https://gymnasium.farama.org/
