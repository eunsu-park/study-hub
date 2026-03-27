[Previous: Safe RL](./27_Safe_RL.md)

---

# 28. Capstone: Training an RL Agent End-to-End

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a complete RL training pipeline from environment selection to deployment
2. Apply modern techniques: distributional RL, world models, or RLHF in a full project
3. Implement proper evaluation protocols with statistical significance
4. Debug common RL training failures systematically
5. Document and reproduce RL experiments following best practices

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Selection and Design](#2-environment-selection-and-design)
3. [Algorithm Selection and Implementation](#3-algorithm-selection-and-implementation)
4. [Training Pipeline](#4-training-pipeline)
5. [Debugging RL Systems](#5-debugging-rl-systems)
6. [Evaluation and Benchmarking](#6-evaluation-and-benchmarking)
7. [Reproducibility and Documentation](#7-reproducibility-and-documentation)
8. [Capstone Projects](#8-capstone-projects)

---

## 1. Project Overview

### 1.1 What Makes a Good RL Project

```
A strong RL project demonstrates:

1. Problem Understanding
   - Clear motivation: why RL? Why not supervised learning?
   - Well-defined objective and success criteria
   - Understanding of domain constraints

2. Technical Depth
   - Appropriate algorithm selection (justified!)
   - Proper implementation with modern techniques
   - Ablation studies showing what matters

3. Rigorous Evaluation
   - Multiple random seeds (at least 5)
   - Confidence intervals on all reported numbers
   - Comparison with baselines
   - Analysis of failure modes

4. Reproducibility
   - All hyperparameters documented
   - Code organized and runnable
   - Environment and dependencies specified
```

### 1.2 Project Timeline

```
Week 1: Setup and Exploration
  □ Select environment
  □ Implement basic random/heuristic baseline
  □ Set up logging (TensorBoard/W&B)
  □ Define evaluation protocol

Week 2: Core Algorithm
  □ Implement chosen algorithm
  □ Get first training run working
  □ Identify and fix obvious bugs
  □ Tune basic hyperparameters

Week 3: Improvements and Ablations
  □ Add advanced techniques (distributional, world model, etc.)
  □ Run ablation experiments
  □ Compare with baselines
  □ Run multiple seeds

Week 4: Analysis and Documentation
  □ Generate plots and analysis
  □ Write project report
  □ Clean up code
  □ Ensure reproducibility
```

---

## 2. Environment Selection and Design

### 2.1 Environment Difficulty Guide

```python
import gymnasium as gym

# Difficulty tiers for RL projects:

# Tier 1: Getting Started (1-2 days to solve)
easy_envs = {
    'CartPole-v1': 'Discrete, dense reward, easy',
    'MountainCar-v0': 'Discrete, sparse reward, needs exploration',
    'Pendulum-v1': 'Continuous, dense reward',
    'LunarLander-v2': 'Discrete/continuous, shaped reward',
}

# Tier 2: Moderate (1-2 weeks)
medium_envs = {
    'BipedalWalker-v3': 'Continuous, locomotion',
    'HalfCheetah-v4': 'MuJoCo, continuous, standard benchmark',
    'Hopper-v4': 'MuJoCo, balance + locomotion',
    'FetchReach-v3': 'Goal-conditioned, sparse reward',
}

# Tier 3: Challenging (2-4 weeks)
hard_envs = {
    'Humanoid-v4': 'High-dim continuous, complex locomotion',
    'FetchPickAndPlace-v3': 'Manipulation, sparse, needs HER',
    'Ant-v4': 'Multi-legged, high dimensional',
}

# Tier 4: Research-level
research_envs = {
    'Atari games': 'Pixel observations, various difficulty',
    'DM Control Suite': 'Continuous control from pixels',
    'Custom environments': 'Tailored to research question',
}
```

### 2.2 Custom Environment Template

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomRLEnvironment(gym.Env):
    """Template for custom RL environments."""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, difficulty='medium'):
        super().__init__()
        self.render_mode = render_mode
        self.difficulty = difficulty

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Environment state
        self.state = None
        self.step_count = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(-0.1, 0.1, size=8).astype(np.float32)
        self.step_count = 0

        info = {'difficulty': self.difficulty}
        return self.state.copy(), info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1

        # Environment dynamics
        self.state = self._dynamics(self.state, action)

        # Reward
        reward = self._compute_reward(self.state, action)

        # Termination conditions
        terminated = self._is_terminal(self.state)
        truncated = self.step_count >= self.max_steps

        info = {
            'step': self.step_count,
            'state_norm': np.linalg.norm(self.state),
        }

        return self.state.copy(), reward, terminated, truncated, info

    def _dynamics(self, state, action):
        """Define your environment dynamics here."""
        # Simple example: linear dynamics with action influence
        next_state = state.copy()
        next_state[:2] += action * 0.1
        next_state[2:4] = action  # velocity = action
        return next_state

    def _compute_reward(self, state, action):
        """Define your reward function here."""
        goal = np.array([1.0, 1.0])
        distance = np.linalg.norm(state[:2] - goal)
        return -distance - 0.01 * np.linalg.norm(action)

    def _is_terminal(self, state):
        """Define termination conditions."""
        goal = np.array([1.0, 1.0])
        return np.linalg.norm(state[:2] - goal) < 0.05
```

---

## 3. Algorithm Selection and Implementation

### 3.1 Algorithm Selection Guide

```
Choose your algorithm:

Discrete actions?
├── Yes
│   ├── Simple/fast → DQN with prioritized replay
│   ├── Want distribution → C51 or QR-DQN
│   └── Best performance → Rainbow DQN
└── No (continuous)
    ├── On-policy preferred → PPO
    ├── Sample efficiency → SAC
    ├── With safety constraints → Lagrangian PPO
    ├── With world model → Dreamer
    └── From demonstrations → GAIL + PPO

Multi-agent?
└── Independent PPO or MAPPO

Goal-conditioned?
└── DDPG/SAC + HER

Offline data available?
└── CQL or Decision Transformer
```

### 3.2 Modular Algorithm Components

```python
class RLComponents:
    """Reusable components for RL algorithms."""

    @staticmethod
    def build_mlp(input_dim, output_dim, hidden_dims=[256, 256],
                  activation=nn.ReLU, output_activation=None):
        """Build MLP with specified architecture."""
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), activation()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation:
            layers.append(output_activation())
        return nn.Sequential(*layers)

    @staticmethod
    def polyak_update(source, target, tau=0.005):
        """Soft update target network."""
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    @staticmethod
    def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
        """Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = np.zeros(T)
        last_gae = 0

        for t in reversed(range(T)):
            next_val = values[t + 1] if t < T - 1 else 0
            delta = rewards[t] + gamma * (1 - dones[t]) * next_val - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

        returns = advantages + values[:T]
        return advantages, returns
```

---

## 4. Training Pipeline

### 4.1 Complete Training Loop

```python
import time
import json
from pathlib import Path


class RLTrainingPipeline:
    """Complete RL training pipeline with logging and checkpoints."""

    def __init__(self, agent, env, config, log_dir='./runs'):
        self.agent = agent
        self.env = env
        self.config = config
        self.log_dir = Path(log_dir) / f"run_{int(time.time())}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        self.episode_count = 0
        self.total_steps = 0
        self.best_return = float('-inf')
        self.metrics_history = []

    def train(self, total_steps, eval_interval=10000, save_interval=50000):
        """Main training loop."""
        state, _ = self.env.reset()
        episode_return = 0
        episode_length = 0
        episode_start = time.time()

        while self.total_steps < total_steps:
            # Select action
            action = self.agent.select_action(state)

            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)

            # Update agent
            if self.agent.ready_to_update():
                update_info = self.agent.update()

            state = next_state
            episode_return += reward
            episode_length += 1
            self.total_steps += 1

            if done:
                # Log episode
                episode_time = time.time() - episode_start
                self.metrics_history.append({
                    'episode': self.episode_count,
                    'step': self.total_steps,
                    'return': episode_return,
                    'length': episode_length,
                    'time': episode_time,
                })

                self.episode_count += 1
                state, _ = self.env.reset()
                episode_return = 0
                episode_length = 0
                episode_start = time.time()

            # Periodic evaluation
            if self.total_steps % eval_interval == 0:
                eval_return = self.evaluate()
                print(f"Step {self.total_steps:,}: Eval Return = {eval_return:.1f}")

                if eval_return > self.best_return:
                    self.best_return = eval_return
                    self.save_checkpoint('best')

            # Periodic save
            if self.total_steps % save_interval == 0:
                self.save_checkpoint(f'step_{self.total_steps}')

        # Final save
        self.save_checkpoint('final')
        self.save_metrics()

    def evaluate(self, n_episodes=10):
        """Evaluate agent without exploration noise."""
        returns = []
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            ep_return = 0
            done = False

            while not done:
                action = self.agent.select_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                ep_return += reward
                done = terminated or truncated

            returns.append(ep_return)

        return np.mean(returns)

    def save_checkpoint(self, name):
        """Save agent checkpoint."""
        path = self.log_dir / f'checkpoint_{name}.pt'
        self.agent.save(path)

    def save_metrics(self):
        """Save training metrics."""
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics_history, f)
```

### 4.2 Hyperparameter Sweep

```python
def hyperparameter_sweep(env_name, algorithm, param_grid, n_seeds=3):
    """Run hyperparameter sweep with multiple seeds."""
    results = []

    for params in param_grid:
        seed_returns = []

        for seed in range(n_seeds):
            env = gym.make(env_name)
            env.reset(seed=seed)

            agent = algorithm(env, seed=seed, **params)
            pipeline = RLTrainingPipeline(agent, env, params)
            pipeline.train(total_steps=params.get('total_steps', 100000))

            final_return = pipeline.evaluate(n_episodes=50)
            seed_returns.append(final_return)

        result = {
            'params': params,
            'mean_return': np.mean(seed_returns),
            'std_return': np.std(seed_returns),
            'seeds': seed_returns,
        }
        results.append(result)

        print(f"Params: {params}")
        print(f"  Return: {result['mean_return']:.1f} "
              f"+/- {result['std_return']:.1f}")

    # Find best
    best = max(results, key=lambda r: r['mean_return'])
    print(f"\nBest: {best['params']} -> {best['mean_return']:.1f}")

    return results
```

---

## 5. Debugging RL Systems

### 5.1 Common RL Bugs

```
RL Debugging Checklist:

1. REWARD BUGS (most common!)
   □ Print reward statistics every 100 episodes
   □ Verify reward sign (positive for good, negative for bad?)
   □ Check for reward clipping issues
   □ Test: does a random agent get ~0 reward?

2. OBSERVATION BUGS
   □ Print observation range and statistics
   □ Are observations normalized? (important for neural nets)
   □ Check for NaN/Inf values
   □ Verify observation matches documentation

3. ACTION BUGS
   □ Are actions clipped to valid range?
   □ For discrete: is action space correct?
   □ For continuous: is action scale appropriate?
   □ Test: do random actions produce varied behavior?

4. NEURAL NETWORK BUGS
   □ Are gradients flowing? (check for zero/exploding gradients)
   □ Is loss decreasing? (at least for value/critic)
   □ Are target networks being updated?
   □ Check weight initialization

5. ALGORITHM BUGS
   □ Is discount factor γ correct? (0.99 typical)
   □ Is experience replay working? (check buffer contents)
   □ Are advantages normalized?
   □ For PPO: is clipping working? (ratio should be near 1)
```

### 5.2 Diagnostic Tools

```python
class RLDiagnostics:
    """Diagnostic tools for debugging RL training."""

    @staticmethod
    def check_environment(env, n_steps=1000):
        """Verify environment is working correctly."""
        print("=== Environment Check ===")
        obs, _ = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")

        rewards = []
        obs_list = []

        for _ in range(n_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            obs_list.append(obs)

            if terminated or truncated:
                obs, _ = env.reset()

        obs_array = np.array(obs_list)
        print(f"\nObservation stats:")
        print(f"  Mean: {obs_array.mean(axis=0)}")
        print(f"  Std:  {obs_array.std(axis=0)}")
        print(f"  Min:  {obs_array.min(axis=0)}")
        print(f"  Max:  {obs_array.max(axis=0)}")

        print(f"\nReward stats:")
        print(f"  Mean: {np.mean(rewards):.4f}")
        print(f"  Std:  {np.std(rewards):.4f}")
        print(f"  Min:  {np.min(rewards):.4f}")
        print(f"  Max:  {np.max(rewards):.4f}")

    @staticmethod
    def check_gradients(model, sample_input):
        """Check gradient flow in neural network."""
        output = model(sample_input)
        loss = output.mean()
        loss.backward()

        print("=== Gradient Check ===")
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                ratio = grad_norm / (param_norm + 1e-8)
                status = "OK" if 1e-7 < grad_norm < 100 else "WARNING"
                print(f"  {name}: grad={grad_norm:.6f}, "
                      f"param={param_norm:.4f}, ratio={ratio:.6f} [{status}]")
            else:
                print(f"  {name}: NO GRADIENT!")

    @staticmethod
    def check_value_accuracy(critic, env, policy, n_episodes=20, gamma=0.99):
        """Check if critic predictions match actual returns."""
        predicted_values = []
        actual_returns = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            states = [state]
            rewards = []
            done = False

            while not done:
                action = policy(state)
                state, reward, terminated, truncated, _ = env.step(action)
                rewards.append(reward)
                states.append(state)
                done = terminated or truncated

            # Compute actual returns
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            # Compare with critic predictions
            for s, ret in zip(states[:-1], returns):
                with torch.no_grad():
                    v = critic(torch.FloatTensor(s).unsqueeze(0)).item()
                predicted_values.append(v)
                actual_returns.append(ret)

        correlation = np.corrcoef(predicted_values, actual_returns)[0, 1]
        mse = np.mean((np.array(predicted_values) - np.array(actual_returns)) ** 2)
        print(f"=== Value Accuracy ===")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  MSE: {mse:.4f}")
```

---

## 6. Evaluation and Benchmarking

### 6.1 Statistical Evaluation

```python
from scipy import stats


def evaluate_with_confidence(agent, env, n_episodes=100,
                             confidence=0.95):
    """Evaluate with proper confidence intervals."""
    returns = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_return = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)

    mean = np.mean(returns)
    se = stats.sem(returns)
    ci = stats.t.interval(confidence, len(returns)-1, loc=mean, scale=se)

    print(f"Mean Return: {mean:.1f}")
    print(f"{confidence*100:.0f}% CI: [{ci[0]:.1f}, {ci[1]:.1f}]")
    print(f"Median: {np.median(returns):.1f}")
    print(f"Min/Max: {np.min(returns):.1f} / {np.max(returns):.1f}")

    return {'mean': mean, 'ci': ci, 'all_returns': returns}


def compare_algorithms(results_dict, metric='mean'):
    """Statistical comparison of multiple algorithms."""
    names = list(results_dict.keys())

    print("=== Algorithm Comparison ===")
    for name, results in results_dict.items():
        returns = results['all_returns']
        print(f"{name}: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")

    # Pairwise significance tests
    print("\nPairwise t-tests (p-values):")
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                t_stat, p_val = stats.ttest_ind(
                    results_dict[name_i]['all_returns'],
                    results_dict[name_j]['all_returns']
                )
                sig = "*" if p_val < 0.05 else ""
                print(f"  {name_i} vs {name_j}: p={p_val:.4f} {sig}")
```

---

## 7. Reproducibility and Documentation

### 7.1 Experiment Configuration

```python
DEFAULT_CONFIG = {
    # Environment
    'env_name': 'HalfCheetah-v4',
    'max_episode_steps': 1000,

    # Algorithm
    'algorithm': 'SAC',
    'gamma': 0.99,
    'tau': 0.005,
    'lr': 3e-4,
    'hidden_dims': [256, 256],
    'batch_size': 256,
    'buffer_size': 1_000_000,
    'learning_starts': 10000,
    'update_frequency': 1,

    # Training
    'total_steps': 1_000_000,
    'eval_interval': 10000,
    'n_eval_episodes': 10,
    'n_seeds': 5,

    # Logging
    'log_dir': './experiments',
    'save_checkpoints': True,
}


def set_all_seeds(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: full determinism may require additional settings
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
```

---

## 8. Capstone Projects

### Project A: Atari Agent with Modern Techniques

Train a high-performing Atari agent:
1. Select 3 Atari games of varying difficulty (e.g., Pong, Breakout, Montezuma)
2. Implement Rainbow DQN (DQN + C51 + PER + n-step + noisy nets + dueling)
3. Add RND exploration bonus for Montezuma's Revenge
4. Train each game for 10M frames (5 seeds each)
5. Report: learning curves, final scores, comparison with vanilla DQN
6. Analysis: Which Rainbow components help most for each game?

### Project B: MuJoCo Locomotion with World Models

Build a Dreamer-style agent for locomotion:
1. Implement RSSM world model for HalfCheetah/Ant
2. Train actor-critic in imagination (Dreamer approach)
3. Compare sample efficiency: Dreamer vs SAC vs PPO
4. Ablation: imagination horizon, model capacity, KL weighting
5. Visualize imagined trajectories vs real trajectories
6. Report: samples to reach target performance for each method

### Project C: RLHF for Text Summarization

Apply RLHF to improve text summarization:
1. Fine-tune GPT-2 small on summarization (SFT phase)
2. Create synthetic preference data (prefer concise, accurate summaries)
3. Train reward model on preference data
4. Implement PPO or DPO for RLHF fine-tuning
5. Compare: SFT baseline vs RLHF with PPO vs DPO
6. Analyze: reward model quality, KL divergence, output quality

### Project D: Safe Robot Navigation

Build a safe navigation agent:
1. Create 2D navigation with obstacles and danger zones
2. Formulate as CMDP: maximize speed, constrain collisions
3. Implement Lagrangian PPO with safety constraints
4. Add safety layer for hard constraint enforcement
5. Compare: unconstrained PPO, Lagrangian PPO, PPO + safety layer
6. Metrics: reward, collision rate, constraint satisfaction, safety during training

### Project E: Multi-Task Goal-Conditioned Agent

Train a single policy for multiple manipulation tasks:
1. Set up 3 Gymnasium Robotics tasks: FetchReach, FetchPush, FetchSlide
2. Implement goal-conditioned SAC with HER
3. Train a single multi-task policy (shared encoder, task-specific heads)
4. Evaluate transfer: does multi-task training help vs separate training?
5. Curriculum: start with easy (Reach), gradually add harder tasks
6. Report: success rates, learning curves, transfer analysis

---

## Final Checklist

Before submitting your capstone project, verify:

```
Code Quality:
  □ Code is clean and well-commented
  □ Configuration is separate from algorithm code
  □ Runs with a single command (with config file)

Experiments:
  □ At least 3 random seeds for each experiment
  □ Proper baselines included
  □ Ablation study on key design choices
  □ Statistical significance reported

Documentation:
  □ README with setup instructions
  □ Hyperparameters listed completely
  □ Learning curves with confidence bands
  □ Clear discussion of results and limitations

Analysis:
  □ What worked and what didn't?
  □ What would you do differently?
  □ What are the key takeaways?
  □ Future work and extensions
```

---

*End of Lesson 28 - Congratulations on completing the Reinforcement Learning course!*
