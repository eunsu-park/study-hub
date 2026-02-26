# 07. Deep Q-Network (DQN)

**Difficulty: ⭐⭐⭐ (Intermediate)**

## Learning Objectives
- Understand DQN's core ideas and structure
- Grasp Experience Replay principles and implementation
- Understand the need for Target Network and its operation
- Learn improvement techniques like Double DQN, Dueling DQN
- Implement DQN with PyTorch

---

## 1. Limitations of Q-Learning and DQN

### 1.1 Limitations of Tabular Q-Learning

```
Problems:
1. Cannot store table for large state spaces (Atari: 256^(84*84*4) states)
2. Cannot handle continuous state spaces
3. No generalization between similar states
```

### 1.2 Function Approximation

```python
# Approximate Q function with neural network instead of table
# Q(s, a) ≈ Q(s, a; θ)

import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Why a neural network for Q: tabular Q-learning requires storing a value for every
    (state, action) pair — impossible when the state space is large or continuous.
    A neural network generalizes across similar states, enabling learning in high-
    dimensional environments like Atari (84x84x4 pixel inputs)."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # Why output all action values at once: a single forward pass gives Q(s,a)
        # for every action, making argmax selection O(1) instead of requiring
        # one forward pass per action
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)  # Output Q values for all actions simultaneously
```

---

## 2. Core Techniques of DQN

### 2.1 Experience Replay

Store experiences in a buffer and sample randomly for learning.

**Advantages:**
- Improved data efficiency (reuse experiences)
- Remove correlation between consecutive samples
- Stabilize learning

```python
from collections import deque
import random

class ReplayBuffer:
    # Why experience replay: consecutive transitions are highly correlated (s_t and s_{t+1}
    # are nearly identical). Training on correlated batches causes the network to overfit
    # to recent experience and forget older knowledge. Random sampling from a large buffer
    # breaks this correlation and provides a more i.i.d.-like training distribution.
    def __init__(self, capacity=100000):
        # Why deque with maxlen: automatically discards oldest experiences when full,
        # ensuring the buffer stays within memory limits while keeping recent data
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)
```

### 2.2 Target Network

Use a separate target network to stabilize learning.

**Problem:** When updating Q(s,a;θ), target y = r + γ max Q(s',a';θ) also changes
**Solution:** Fix target network θ⁻ and update periodically

**DQN Loss Function:**

$$L(\theta) = \mathbb{E}\left[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2\right]$$

where:
- $\theta$ = online network parameters (updated every step)
- $\theta^-$ = target network parameters (frozen copy of $\theta$, updated every N steps)
- $r + \gamma \max_{a'} Q(s', a'; \theta^-)$ = TD target (computed with frozen $\theta^-$ for stability)

**Why a target network?** Without it, both the prediction $Q(s,a;\theta)$ and the target $r + \gamma \max_{a'} Q(s',a';\theta)$ shift simultaneously on each gradient step. This creates a "moving target" problem — the network chases a target that keeps changing, causing oscillation or divergence. Freezing $\theta^-$ decouples the target from the current parameters, turning the problem into a supervised regression task for N steps at a time.

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)

        # Why identical initialization: ensures the target network starts at the same
        # point, so initial TD targets are consistent with the online network
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = 0.99

    def update_target_network(self):
        """Hard update: copy all weights at once every N steps.
        Why hard update: simple and ensures the target is stable for exactly N steps,
        giving the online network a fixed regression target to converge toward."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update_target(self, tau=0.005):
        """Soft update: blend weights incrementally every step.
        Why soft update (Polyak averaging): avoids the sudden shift of hard updates,
        providing a smoother target that can improve stability in some environments."""
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
```

---

## 3. Complete DQN Implementation

### 3.1 Agent Class

```python
import numpy as np

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step = 0

        # Networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def choose_action(self, state, training=True):
        # Why epsilon-greedy with decay: early on, Q estimates are random, so heavy
        # exploration discovers diverse transitions. As Q improves, we shift toward
        # exploitation. At test time (training=False), we use pure greedy.
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        # Why minimum buffer check: training on very small batches before the buffer
        # fills produces high-variance gradients from correlated recent transitions
        if len(self.buffer) < self.batch_size:
            return None

        # Why random batch sampling: breaks temporal correlation between consecutive
        # transitions, approximating an i.i.d. training set
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Why gather: the network outputs Q for all actions; gather extracts only
        # the Q values for the actions that were actually taken
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Why target network here: using frozen theta^- for the TD target prevents
        # the "moving target" problem — the target stays fixed for target_update_freq
        # steps, turning this into a standard supervised regression problem
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            # Why (1 - dones): terminal states have no future reward, so we zero out
            # the bootstrapped value to prevent the network from hallucinating future returns
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # MSE loss: L(theta) = E[(target_q - current_q)^2]
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Why gradient clipping: deep networks can produce exploding gradients,
        # especially early in training when Q estimates are poor. Clipping keeps
        # updates bounded, preventing catastrophic parameter jumps.
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        # Why periodic hard update: copies online weights to target network every N
        # steps. More frequent updates track the online network faster but reduce
        # stability; less frequent updates are more stable but slower to incorporate
        # new knowledge.
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Why epsilon floor at epsilon_min: ensures the agent always retains some
        # exploration, preventing permanent lock-in to a suboptimal policy
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()
```

### 3.2 Training Loop

```python
import gymnasium as gym

def train_dqn(env_name='CartPole-v1', n_episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    rewards_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn()

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")

    return agent, rewards_history
```

---

## 4. DQN Improvements

### 4.1 Double DQN

Solves Q-value overestimation problem in vanilla DQN.

```python
# Vanilla DQN: y = r + γ max_a' Q(s', a'; θ⁻)
# Double DQN: y = r + γ Q(s', argmax_a' Q(s', a'; θ); θ⁻)

def compute_double_dqn_target(self, rewards, next_states, dones):
    # Why Double DQN: vanilla DQN uses max_a' Q(s', a'; theta^-) for both action
    # selection and evaluation. The max operator is positively biased — when Q
    # estimates have noise, max picks the noisiest overestimate. By decoupling
    # selection (online network) from evaluation (target network), the bias is
    # greatly reduced, leading to more accurate Q values and better policies.
    with torch.no_grad():
        # Why online network for selection: the online network has the freshest
        # estimates, making it better at identifying the best action
        next_actions = self.q_network(next_states).argmax(1, keepdim=True)

        # Why target network for evaluation: even if the selected action is slightly
        # wrong, the target network provides a less biased Q estimate for it
        next_q = self.target_network(next_states).gather(1, next_actions).squeeze()

        target_q = rewards + self.gamma * next_q * (1 - dones)

    return target_q
```

### 4.2 Dueling DQN

Decompose Q function into V (state value) and A (advantage).

```
Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))
```

```python
class DuelingQNetwork(nn.Module):
    """Why Dueling architecture: many states have similar values regardless of
    which action is taken (e.g., when no obstacle is near, all actions are
    roughly equal). Separating V(s) from A(s,a) lets the network learn the
    state value independently, improving sample efficiency in states where
    action choice matters little."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Why shared features: both V and A need to understand the state, so sharing
        # early layers avoids redundant computation and improves feature reuse
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream (V)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Why subtract mean(A): makes the decomposition identifiable — without
        # centering, V and A are not uniquely determined (any constant could shift
        # between them). Subtracting the mean forces A to have zero mean, so V
        # truly represents the state value.
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
```

### 4.3 Prioritized Experience Replay (PER)

Sample experiences with higher TD error more frequently.

```python
class PrioritizedReplayBuffer:
    """Why prioritized replay: uniform sampling wastes time replaying transitions
    the network already predicts well. Sampling proportional to TD error focuses
    learning on surprising or poorly-predicted transitions, accelerating convergence."""

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        # Why alpha: controls how much prioritization is used (alpha=0 → uniform,
        # alpha=1 → fully proportional to TD error). 0.6 is a common balance.
        self.alpha = alpha
        # Why beta: importance sampling correction — prioritized sampling introduces
        # bias by oversampling high-error transitions. Beta anneals from 0.4 to 1.0
        # during training to fully correct this bias by end of training.
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0

    def push(self, *experience, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        total_priority = self.priorities[:len(self.buffer)].sum()
        probs = self.priorities[:len(self.buffer)] / total_priority

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        return batch, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha
```

---

## 5. CNN-based DQN (Atari)

### 5.1 Image Input Network

```python
class AtariDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        # Why 4-frame stack: a single frame gives no velocity information —
        # stacking 4 consecutive frames lets the network infer motion (direction
        # and speed of objects), which is critical for games like Pong and Breakout
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # x shape: (batch, 4, 84, 84)
        # Why normalize to [0,1]: pixel values in [0, 255] would produce very large
        # activations; dividing by 255 keeps inputs in a stable range for gradient descent
        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

### 5.2 Frame Preprocessing

```python
import cv2

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess_frame(self, frame):
        """Convert to 84x84 grayscale"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, frame):
        processed = self.preprocess_frame(frame)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        return np.array(self.frames)

    def step(self, frame):
        processed = self.preprocess_frame(frame)
        self.frames.append(processed)
        return np.array(self.frames)
```

---

## 6. Practice: CartPole-v1

```python
def main():
    # Environment setup
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # DQN agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=100
    )

    # Training
    n_episodes = 300
    scores = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        score = 0

        for t in range(500):
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done or truncated)
            agent.learn()

            state = next_state
            score += reward

            if done or truncated:
                break

        scores.append(score)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Score: {np.mean(scores[-10:]):.2f}")

        # Solved condition
        if np.mean(scores[-100:]) >= 475:
            print(f"Solved in {episode + 1} episodes!")
            break

    env.close()
    return agent, scores

if __name__ == "__main__":
    agent, scores = main()
```

---

## Summary

| Technique | Purpose | Key Idea |
|-----------|---------|----------|
| Experience Replay | Data efficiency, remove correlation | Random sampling from buffer |
| Target Network | Stabilize learning | Fix target, periodic updates |
| Double DQN | Prevent overestimation | Separate action selection/evaluation |
| Dueling DQN | Efficient learning | Separate V and A |
| PER | Efficient sampling | Priority based on TD error |

---

## Next Steps

- [08_Policy_Gradient.md](./08_Policy_Gradient.md) - Policy-based methods

---

## References

- Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
- Mnih et al., "Human-level control through deep reinforcement learning" (Nature 2015)
- van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2015)
- Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning" (2016)
- Schaul et al., "Prioritized Experience Replay" (2015)
