# 12. Practical RL Project

**Difficulty: ⭐⭐⭐⭐ (Advanced)**

## Learning Objectives
- Master Gymnasium environment usage
- Understand complete RL project structure
- Learn training monitoring and debugging techniques
- Implement Atari game agents
- Save and evaluate trained models

---

## 1. Project Structure

### 1.1 Recommended Directory Structure

```
rl_project/
├── config/
│   ├── default.yaml
│   └── atari.yaml
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── dqn.py
│   └── ppo.py
├── networks/
│   ├── __init__.py
│   ├── mlp.py
│   └── cnn.py
├── utils/
│   ├── __init__.py
│   ├── buffer.py
│   ├── logger.py
│   └── wrappers.py
├── envs/
│   └── custom_env.py
├── train.py
├── evaluate.py
└── requirements.txt
```

### 1.2 Configuration File

```yaml
# config/default.yaml
env:
  name: "CartPole-v1"
  n_envs: 4

agent:
  type: "PPO"
  lr: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  epochs: 10
  batch_size: 64

training:
  total_timesteps: 100000
  eval_freq: 10000
  save_freq: 50000
  log_freq: 1000

logging:
  use_wandb: true
  project_name: "rl-project"
```

---

## 2. Gymnasium Environment

### 2.1 Basic Usage

```python
import gymnasium as gym
import numpy as np

def basic_usage():
    # Create environment
    env = gym.make("CartPole-v1", render_mode="human")

    # Environment info
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run episode
    observation, info = env.reset(seed=42)

    for _ in range(1000):
        action = env.action_space.sample()  # random action
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
```

### 2.2 Vectorized Environments (Parallel Processing)

```python
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

def make_env(env_name, seed):
    def _init():
        env = gym.make(env_name)
        env.reset(seed=seed)
        return env
    return _init

def vectorized_envs():
    n_envs = 4
    env_name = "CartPole-v1"

    # Asynchronous environments (each environment in a separate process)
    envs = AsyncVectorEnv([
        make_env(env_name, seed=i) for i in range(n_envs)
    ])

    # Reset all environments simultaneously
    observations, infos = envs.reset()
    print(f"Observations shape: {observations.shape}")

    # Step all environments simultaneously
    actions = envs.action_space.sample()
    observations, rewards, terminateds, truncateds, infos = envs.step(actions)

    envs.close()
```

### 2.3 Environment Wrappers

```python
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class FrameStack(gym.Wrapper):
    """Stack consecutive frames"""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

        # Modify observation space
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(n_frames, *obs_shape),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return np.array(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.array(self.frames), reward, terminated, truncated, info


class RewardWrapper(gym.RewardWrapper):
    """Reward scaling/clipping"""
    def reward(self, reward):
        return np.clip(reward, -1, 1)


class NormalizeObservation(gym.ObservationWrapper):
    """Observation normalization"""
    def __init__(self, env):
        super().__init__(env)
        self.mean = 0
        self.var = 1
        self.count = 1e-4

    def observation(self, obs):
        self.update_stats(obs)
        return (obs - self.mean) / np.sqrt(self.var + 1e-8)

    def update_stats(self, obs):
        batch_mean = np.mean(obs)
        batch_var = np.var(obs)
        batch_count = obs.size

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        self.var = (self.var * self.count + batch_var * batch_count) / total_count
        self.count = total_count
```

---

## 3. Complete PPO Project

### 3.1 Network Definition

```python
# networks/mlp.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(64, 64)):
        super().__init__()

        # Why shared layers: actor and critic share a feature extractor —
        # they both need to understand the environment state, so sharing
        # lower-level features reduces parameters and speeds up learning
        layers = []
        prev_size = obs_dim
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.Tanh()
            ])
            prev_size = size

        self.shared = nn.Sequential(*layers)

        # Actor and Critic heads
        self.actor = nn.Linear(prev_size, action_dim)
        self.critic = nn.Linear(prev_size, 1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Why orthogonal initialization with sqrt(2): preserves gradient
                # magnitude through deep networks; empirically more stable for
                # RL than Xavier/Kaiming because it prevents early policy collapse
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        features = self.shared(obs)
        return self.actor(features), self.critic(features)

    def get_action_and_value(self, obs, action=None):
        logits, value = self.forward(obs)
        # Why Categorical with logits (not probs): numerically stable — avoids
        # explicit softmax which can overflow/underflow; logits are passed
        # directly to the log-sum-exp trick inside the distribution
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)
```

### 3.2 PPO Agent

```python
# agents/ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO:
    def __init__(
        self,
        env,
        network,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64,
        device="cpu"
    ):
        self.env = env
        self.network = network.to(device)
        self.optimizer = optim.Adam(network.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

    def collect_rollout(self, n_steps):
        """Collect experience"""
        obs_buf = []
        act_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        logp_buf = []

        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)

        for _ in range(n_steps):
            with torch.no_grad():
                action, logp, _, value = self.network.get_action_and_value(obs)

            next_obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = terminated or truncated

            obs_buf.append(obs.cpu().numpy())
            act_buf.append(action.cpu().numpy())
            rew_buf.append(reward)
            done_buf.append(done)
            val_buf.append(value.cpu().numpy())
            logp_buf.append(logp.cpu().numpy())

            obs = torch.FloatTensor(next_obs).to(self.device)
            if done:
                obs, _ = self.env.reset()
                obs = torch.FloatTensor(obs).to(self.device)

        # Estimate last value
        with torch.no_grad():
            _, _, _, last_value = self.network.get_action_and_value(obs)

        return {
            'obs': np.array(obs_buf),
            'actions': np.array(act_buf),
            'rewards': np.array(rew_buf),
            'dones': np.array(done_buf),
            'values': np.array(val_buf),
            'log_probs': np.array(logp_buf),
            'last_value': last_value.cpu().numpy()
        }

    def compute_gae(self, rollout):
        """Compute GAE"""
        rewards = rollout['rewards']
        values = rollout['values']
        dones = rollout['dones']
        last_value = rollout['last_value']

        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        last_gae = 0

        # Why reversed iteration: GAE is defined recursively as
        # A_t = delta_t + (gamma*lambda) * A_{t+1}, so we must compute
        # future advantages before current ones — a backward pass is O(n)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            # Why multiply by next_non_terminal: masks out the TD error at
            # episode boundaries so that value from a new episode doesn't
            # "bleed into" the advantage estimate of the previous episode
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            # Why gae_lambda: controls the bias-variance trade-off; lambda=1
            # gives unbiased but high-variance MC returns, lambda=0 gives
            # low-variance but biased 1-step TD; lambda=0.95 is a sweet spot
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """PPO update"""
        advantages, returns = self.compute_gae(rollout)

        # Why normalize advantages: removes scale dependence so that the same
        # clip_epsilon works across tasks with very different reward magnitudes;
        # also improves gradient conditioning and training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs = torch.FloatTensor(rollout['obs']).to(self.device)
        actions = torch.LongTensor(rollout['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Why multiple epochs on the same rollout: PPO's clipped objective
        # bounds how far the new policy can deviate from the old one, making
        # it safe to reuse each batch of experience for several gradient steps —
        # this improves sample efficiency without the instability of TRPO
        n_samples = len(obs)
        indices = np.arange(n_samples)

        total_loss = 0
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    obs[batch_idx], actions[batch_idx]
                )

                # Why ratio in log space: exp(log_new - log_old) is numerically
                # identical to new_prob / old_prob but avoids floating-point
                # underflow when probabilities are very small
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])

                # Why take min of surr1 and surr2: the clipped surrogate
                # prevents large policy updates in either direction — surr2
                # caps the benefit of moving toward high-advantage actions,
                # making the objective a pessimistic lower bound on true performance
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns[batch_idx])

                # Why negative entropy as a loss term: maximizing entropy encourages
                # the policy to remain stochastic and continue exploring; the small
                # entropy_coef (0.01) prevents premature convergence to a deterministic policy
                entropy_loss = -entropy.mean()

                # Total loss
                loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                # Why clip gradients: PPO can produce large gradients when the
                # policy ratio deviates significantly; clipping prevents destructive
                # parameter updates that would break the rollout collection policy
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss / (self.n_epochs * (n_samples // self.batch_size))

    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
```

---

## 4. Training Script

```python
# train.py
import gymnasium as gym
import numpy as np
import torch
from agents.ppo import PPO
from networks.mlp import ActorCriticMLP
from utils.logger import Logger

def train(config):
    # Create environment
    env = gym.make(config['env']['name'])

    # Create network
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    network = ActorCriticMLP(obs_dim, action_dim)

    # Create agent
    agent = PPO(
        env=env,
        network=network,
        **config['agent']
    )

    # Logger
    logger = Logger(config['logging'])

    # Training loop
    total_timesteps = config['training']['total_timesteps']
    n_steps = config['training']['n_steps']
    timesteps = 0
    episode_rewards = []
    current_episode_reward = 0

    while timesteps < total_timesteps:
        # Collect rollout
        rollout = agent.collect_rollout(n_steps)
        timesteps += n_steps

        # Track episode rewards
        for r, d in zip(rollout['rewards'], rollout['dones']):
            current_episode_reward += r
            if d:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0

        # Update
        loss = agent.update(rollout)

        # Logging
        if len(episode_rewards) > 0:
            logger.log({
                'timesteps': timesteps,
                'loss': loss,
                'mean_reward': np.mean(episode_rewards[-10:]),
                'episodes': len(episode_rewards)
            })

        # Save checkpoint
        if timesteps % config['training']['save_freq'] == 0:
            agent.save(f"checkpoints/ppo_{timesteps}.pt")

    env.close()
    return agent

if __name__ == "__main__":
    import yaml
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    train(config)
```

---

## 5. Evaluation Script

```python
# evaluate.py
import gymnasium as gym
import torch
import numpy as np

def evaluate(agent, env_name, n_episodes=10, render=False):
    """Evaluate trained agent"""
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)

    episode_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = agent.network.get_action_and_value(obs_tensor)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: {total_reward}")

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return episode_rewards
```

---

## 6. Logging and Visualization

### 6.1 Weights & Biases Integration

```python
# utils/logger.py
import wandb
import matplotlib.pyplot as plt
from collections import deque

class Logger:
    def __init__(self, config):
        self.use_wandb = config.get('use_wandb', False)
        self.rewards_buffer = deque(maxlen=100)

        if self.use_wandb:
            wandb.init(
                project=config.get('project_name', 'rl-project'),
                config=config
            )

    def log(self, metrics):
        if 'mean_reward' in metrics:
            self.rewards_buffer.append(metrics['mean_reward'])

        if self.use_wandb:
            wandb.log(metrics)
        else:
            print(f"Step {metrics.get('timesteps', 0)}: "
                  f"Reward={metrics.get('mean_reward', 0):.2f}")

    def plot_rewards(self, rewards, save_path=None):
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def close(self):
        if self.use_wandb:
            wandb.finish()
```

---

## 7. Atari Project

### 7.1 CNN Network

```python
# networks/cnn.py
class AtariNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.actor = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x / 255.0  # normalize
        features = self.conv(x)
        return self.actor(features), self.critic(features)
```

### 7.2 Atari Wrappers

```python
from gymnasium.wrappers import AtariPreprocessing, FrameStack

def make_atari_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False
    )
    env = FrameStack(env, 4)
    return env
```

---

## 8. Debugging Tips

### 8.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Reward not increasing | Learning rate too high/low | Grid search learning rate |
| Unstable training | Gradient explosion | Gradient clipping |
| Sudden performance drop | Policy change too drastic | Reduce clip_epsilon |
| Out of memory | Buffer size | Adjust batch size |

### 8.2 Debugging Code

```python
def debug_training(agent):
    """Debug training"""
    # Check gradients
    for name, param in agent.network.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm={grad_norm:.6f}")

    # Check policy entropy
    obs = torch.randn(1, obs_dim)
    logits, _ = agent.network(obs)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.log()).sum()
    print(f"Policy entropy: {entropy.item():.4f}")
```

---

## Summary

**Project Checklist:**
- [ ] Environment setup and testing
- [ ] Define network architecture
- [ ] Implement agent
- [ ] Write training loop
- [ ] Configure logging
- [ ] Hyperparameter tuning
- [ ] Save/load model
- [ ] Evaluation and visualization

**Key Tools:**
- Gymnasium: Environments
- PyTorch: Neural networks
- Weights & Biases: Experiment tracking
- NumPy: Numerical operations

---

## Exercises

### Exercise 1: Custom Gymnasium Environment

Build a custom environment following the Gymnasium API.

1. Create a `GridWorldEnv` class that inherits from `gym.Env` and implements a 5×5 grid navigation task:
   - The agent starts at (0, 0) and must reach the goal at (4, 4).
   - Actions: 0=Up, 1=Down, 2=Left, 3=Right.
   - Reward: +10 at goal, −0.1 per step, episode ends at goal or after 100 steps.
   - Observation: a length-2 vector [row, col] (normalized to [0, 1]).
2. Implement `__init__`, `reset`, `step`, `render`, and `close` methods with correct type signatures.
3. Register the environment: `gym.register(id='GridWorld-v0', entry_point='envs.custom_env:GridWorldEnv')`.
4. Verify with `gymnasium.utils.env_checker.check_env(env)` — fix any reported issues.
5. Train a REINFORCE agent on the environment for 500 episodes and visualize the learned path.

### Exercise 2: Environment Wrapper Pipeline

Build a pipeline of custom wrappers for an Atari-style task.

1. Using the `FrameStack` and `NormalizeObservation` wrappers from Section 2.3, create a composed wrapper for `CartPole-v1`:
   ```python
   env = gym.make('CartPole-v1')
   env = NormalizeObservation(env)  # normalize observations online
   env = RecordEpisodeStatistics(env)  # track episode length and reward
   ```
2. Implement a new `TimeLimit` wrapper that terminates an episode after `max_steps` steps regardless of the environment's own done signal.
3. Implement a `ClipReward` wrapper that clips rewards to [−1, 1] (as used in DQN Atari training).
4. Chain all four wrappers together and confirm that the `observation_space` and `action_space` are preserved correctly.
5. Verify that the `NormalizeObservation` wrapper produces approximately zero-mean, unit-variance observations after 1000 steps of random play.

### Exercise 3: Experiment Tracking with W&B

Instrument a full PPO training run with Weights & Biases logging.

1. Install wandb: `pip install wandb` and create a free account at wandb.ai.
2. Initialize a W&B run at the start of `train.py`:
   ```python
   wandb.init(project="rl-study", config=config)
   ```
3. Log the following metrics every rollout:
   - `actor_loss`, `critic_loss`, `entropy`
   - `mean_reward` (average of last 10 episodes)
   - `policy_ratio_mean` and `policy_ratio_max` (from the PPO update)
4. Log a histogram of advantages before normalization every 10 rollouts.
5. Run three seeds (42, 123, 456) on CartPole-v1 and use W&B's grouping feature to visualize mean ± std across seeds. What does the variance across seeds tell you about PPO's stability?

### Exercise 4: Hyperparameter Sweep

Conduct a systematic hyperparameter search using the config file structure.

1. Use the `config/default.yaml` structure from Section 1.2. Define a sweep over:
   - `lr` ∈ {1e-4, 3e-4, 1e-3}
   - `clip_epsilon` ∈ {0.1, 0.2, 0.3}
   - `gae_lambda` ∈ {0.9, 0.95, 1.0}
2. This gives 27 configurations — run each for 100,000 timesteps on CartPole-v1 (or use W&B sweeps to run them automatically).
3. Record `mean_reward` of the last 20 episodes for each configuration.
4. Identify the top 3 configurations and the bottom 3. What patterns do you observe?
5. Plot a heatmap of `mean_reward` vs. `lr` and `clip_epsilon` (averaging over `gae_lambda`). Which hyperparameter has the largest effect?

### Exercise 5: End-to-End Atari Agent

Build and train a complete PPO agent on a simple Atari game.

1. Set up the Atari environment with the preprocessing wrappers from Section 7.2:
   ```python
   env = make_atari_env('ALE/Pong-v5')
   ```
2. Instantiate the `AtariNetwork` CNN from Section 7.1 and verify that the output shape of the convolutional layers is 3136 by running a forward pass with a dummy input of shape `(1, 4, 84, 84)`.
3. Train PPO for 1,000,000 timesteps using the hyperparameters from Section 7 of the PPO lesson (lr=2.5e-4, n_steps=128, clip_epsilon=0.1).
4. Save a checkpoint every 200,000 timesteps and evaluate each checkpoint for 10 episodes.
5. Plot the evaluation reward vs. timesteps. At what point does the agent first consistently beat the random baseline (reward > −20 in Pong)?

---

## Additional Learning Resources

- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [RLlib](https://docs.ray.io/en/latest/rllib/)
