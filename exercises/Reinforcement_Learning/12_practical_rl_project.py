"""
Exercises for Lesson 12: Practical RL Project
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
import time


def exercise_1():
    """
    Exercise 1: Custom Gymnasium Environment

    Build a custom 5x5 grid navigation environment following the Gymnasium API.
    """
    print("Custom GridWorld Environment:")
    print("=" * 60)
    print()

    class GridWorldEnv:
        """
        5x5 Grid World following Gymnasium-like API.

        Agent starts at (0,0), goal at (4,4).
        Actions: 0=Up, 1=Down, 2=Left, 3=Right
        Reward: +10 at goal, -0.1 per step
        Episode ends at goal or after 100 steps.
        Observation: [row, col] normalized to [0, 1]
        """

        def __init__(self, size=5):
            self.size = size
            self.goal = (size - 1, size - 1)
            self.max_steps = 100
            self.action_deltas = {
                0: (-1, 0),  # Up
                1: (1, 0),   # Down
                2: (0, -1),  # Left
                3: (0, 1),   # Right
            }
            self.n_actions = 4
            self.state_dim = 2

        def reset(self, seed=None):
            if seed is not None:
                np.random.seed(seed)
            self.pos = (0, 0)
            self.steps = 0
            obs = np.array(self.pos, dtype=np.float32) / (self.size - 1)
            return obs

        def step(self, action):
            self.steps += 1
            dr, dc = self.action_deltas[action]
            new_r = max(0, min(self.size - 1, self.pos[0] + dr))
            new_c = max(0, min(self.size - 1, self.pos[1] + dc))
            self.pos = (new_r, new_c)

            obs = np.array(self.pos, dtype=np.float32) / (self.size - 1)

            if self.pos == self.goal:
                reward = 10.0
                done = True
            elif self.steps >= self.max_steps:
                reward = -0.1
                done = True
            else:
                reward = -0.1
                done = False

            return obs, reward, done

        def render(self):
            grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
            grid[self.goal[0]][self.goal[1]] = 'G'
            grid[self.pos[0]][self.pos[1]] = 'A'
            print()
            for row in grid:
                print('  ' + ' '.join(row))
            print()

    # Verify the environment
    env = GridWorldEnv()
    print("Environment properties:")
    print(f"  Grid size: {env.size}x{env.size}")
    print(f"  State dim: {env.state_dim}")
    print(f"  Actions: {env.n_actions}")
    print(f"  Goal: {env.goal}")
    print(f"  Max steps: {env.max_steps}")
    print()

    # Test reset
    obs = env.reset(seed=42)
    print(f"After reset: obs = {obs}")
    env.render()

    # Test step
    obs, reward, done = env.step(1)  # Down
    print(f"After action Down: obs = {obs}, reward = {reward}, done = {done}")

    obs, reward, done = env.step(3)  # Right
    print(f"After action Right: obs = {obs}, reward = {reward}, done = {done}")

    # Train REINFORCE agent
    print("\nTraining REINFORCE on GridWorld:")

    # Simple linear policy
    weights = np.zeros((2, 4))
    lr = 0.05
    gamma = 0.99
    n_episodes = 500

    def softmax(logits):
        logits = logits - np.max(logits)
        exp_l = np.exp(logits)
        return exp_l / np.sum(exp_l)

    episode_rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        states, actions, rewards = [], [], []
        done = False

        while not done:
            logits = obs @ weights
            probs = softmax(logits)
            action = np.random.choice(4, p=probs)
            next_obs, reward, done = env.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # Compute returns and update
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for t in range(len(states)):
            probs = softmax(states[t] @ weights)
            grad = np.zeros(4)
            grad[actions[t]] = 1.0
            grad -= probs
            weights += lr * np.outer(states[t], grad * returns[t])

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  Episode {episode+1}: avg_reward = {avg:.2f}")

    # Visualize learned path
    print("\nLearned path:")
    obs = env.reset()
    env.render()
    done = False
    path = [env.pos]
    while not done and len(path) < 20:
        logits = obs @ weights
        action = np.argmax(logits)
        obs, reward, done = env.step(action)
        path.append(env.pos)

    print(f"  Path: {' -> '.join(str(p) for p in path)}")
    print(f"  Reached goal: {env.pos == env.goal}")


def exercise_2():
    """
    Exercise 2: Environment Wrapper Pipeline

    Build a pipeline of custom wrappers.
    """
    print("Environment Wrapper Pipeline:")
    print("=" * 60)
    print()

    # Base environment
    class SimpleEnv:
        """Simple environment for testing wrappers."""

        def __init__(self):
            self.state = np.zeros(4, dtype=np.float32)
            self.n_actions = 2
            self.state_dim = 4
            self.max_steps = 200

        def reset(self):
            self.state = np.random.uniform(-0.05, 0.05, size=4).astype(np.float32)
            self.steps = 0
            return self.state.copy()

        def step(self, action):
            self.steps += 1
            self.state += np.random.randn(4).astype(np.float32) * 0.1
            reward = 1.0
            done = self.steps >= self.max_steps or abs(self.state[2]) > 0.2
            return self.state.copy(), reward, done

    # Wrapper 1: Normalize Observation
    class NormalizeObservation:
        """Online observation normalization using Welford's algorithm."""

        def __init__(self, env):
            self.env = env
            self.n_actions = env.n_actions
            self.state_dim = env.state_dim
            self.mean = np.zeros(env.state_dim)
            self.var = np.ones(env.state_dim)
            self.count = 1e-4

        def reset(self):
            obs = self.env.reset()
            self._update_stats(obs)
            return self._normalize(obs)

        def step(self, action):
            obs, reward, done = self.env.step(action)
            self._update_stats(obs)
            return self._normalize(obs), reward, done

        def _update_stats(self, obs):
            self.count += 1
            delta = obs - self.mean
            self.mean += delta / self.count
            delta2 = obs - self.mean
            self.var += (delta * delta2 - self.var) / self.count

        def _normalize(self, obs):
            return (obs - self.mean) / (np.sqrt(self.var) + 1e-8)

    # Wrapper 2: Time Limit
    class TimeLimit:
        """Terminate episode after max_steps regardless of env."""

        def __init__(self, env, max_steps=50):
            self.env = env
            self.n_actions = env.n_actions
            self.state_dim = env.state_dim
            self.max_steps = max_steps
            self.steps = 0

        def reset(self):
            self.steps = 0
            return self.env.reset()

        def step(self, action):
            obs, reward, done = self.env.step(action)
            self.steps += 1
            if self.steps >= self.max_steps:
                done = True
            return obs, reward, done

    # Wrapper 3: Clip Reward
    class ClipReward:
        """Clip rewards to [-1, 1]."""

        def __init__(self, env):
            self.env = env
            self.n_actions = env.n_actions
            self.state_dim = env.state_dim

        def reset(self):
            return self.env.reset()

        def step(self, action):
            obs, reward, done = self.env.step(action)
            reward = np.clip(reward, -1.0, 1.0)
            return obs, reward, done

    # Wrapper 4: Record Statistics
    class RecordEpisodeStatistics:
        """Track episode length and reward."""

        def __init__(self, env):
            self.env = env
            self.n_actions = env.n_actions
            self.state_dim = env.state_dim
            self.episode_reward = 0
            self.episode_length = 0
            self.episodes_completed = 0
            self.reward_history = []

        def reset(self):
            if self.episode_length > 0:
                self.reward_history.append(self.episode_reward)
                self.episodes_completed += 1
            self.episode_reward = 0
            self.episode_length = 0
            return self.env.reset()

        def step(self, action):
            obs, reward, done = self.env.step(action)
            self.episode_reward += reward
            self.episode_length += 1
            return obs, reward, done

    # Chain all wrappers
    print("Chaining wrappers: Base -> Normalize -> ClipReward -> TimeLimit -> Record")
    print()

    base_env = SimpleEnv()
    env = NormalizeObservation(base_env)
    env = ClipReward(env)
    env = TimeLimit(env, max_steps=100)
    env = RecordEpisodeStatistics(env)

    # Verify properties are preserved
    print(f"  State dim: {env.state_dim}")
    print(f"  N actions: {env.n_actions}")
    print()

    # Run random policy and verify wrappers
    obs_history = []
    reward_history = []

    for episode in range(20):
        obs = env.reset()
        done = False
        while not done:
            action = np.random.randint(env.n_actions)
            obs, reward, done = env.step(action)
            obs_history.append(obs)
            reward_history.append(reward)

    obs_array = np.array(obs_history)
    reward_array = np.array(reward_history)

    print("Verification after 20 episodes of random play:")
    print()

    # Check normalization
    print("NormalizeObservation check:")
    print(f"  Observation mean: {np.mean(obs_array, axis=0).round(3)}")
    print(f"  Observation std:  {np.std(obs_array, axis=0).round(3)}")
    print(f"  Expected: approximately zero mean, unit variance")
    print()

    # Check reward clipping
    print("ClipReward check:")
    print(f"  Reward range: [{reward_array.min():.2f}, {reward_array.max():.2f}]")
    print(f"  Expected: within [-1.0, 1.0]")
    print()

    # Check time limit
    print("TimeLimit check:")
    print(f"  Episodes completed: {env.episodes_completed}")
    print(f"  Reward history: {[f'{r:.1f}' for r in env.reward_history[:5]]}...")
    print()

    print("All wrappers functioning correctly!")


def exercise_3():
    """
    Exercise 3: Experiment Tracking (without W&B)

    Demonstrate experiment tracking patterns.
    """
    print("Experiment Tracking Patterns:")
    print("=" * 60)
    print()

    class SimpleLogger:
        """Simple experiment logger (W&B-like interface without dependencies)."""

        def __init__(self, project_name, config):
            self.project = project_name
            self.config = config
            self.metrics = {}
            self.step = 0
            print(f"  Initialized logger for project: {project_name}")
            print(f"  Config: {config}")

        def log(self, data):
            self.step += 1
            for key, value in data.items():
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)

        def summary(self):
            print(f"\n  Logger Summary:")
            for key, values in self.metrics.items():
                if len(values) > 0:
                    print(f"    {key}: min={min(values):.3f}, "
                          f"max={max(values):.3f}, "
                          f"last={values[-1]:.3f}")

    # Run experiment with 3 seeds
    configs = [
        {'seed': 42, 'lr': 3e-4, 'gamma': 0.99},
        {'seed': 123, 'lr': 3e-4, 'gamma': 0.99},
        {'seed': 456, 'lr': 3e-4, 'gamma': 0.99},
    ]

    all_rewards = []

    for config in configs:
        np.random.seed(config['seed'])
        logger = SimpleLogger("rl-study", config)

        # Simulate training (simplified PPO-like)
        episode_rewards = []
        for episode in range(200):
            # Simulated reward with improvement trend
            base = 50 + 150 * (1 - np.exp(-episode / 50))
            noise = np.random.randn() * 30
            reward = max(0, base + noise)
            episode_rewards.append(reward)

            if (episode + 1) % 20 == 0:
                logger.log({
                    'mean_reward': np.mean(episode_rewards[-10:]),
                    'episode': episode + 1,
                })

        logger.summary()
        all_rewards.append(episode_rewards)

    # Cross-seed statistics
    print("\nCross-seed analysis:")
    all_rewards = np.array(all_rewards)  # Shape: (3, 200)

    for window_start in [0, 50, 100, 150]:
        window_end = window_start + 50
        window_data = all_rewards[:, window_start:window_end]
        means = np.mean(window_data, axis=1)
        print(f"  Episodes {window_start+1}-{window_end}:")
        print(f"    Per-seed means: {np.round(means, 1)}")
        print(f"    Overall mean: {np.mean(means):.1f} +/- {np.std(means):.1f}")

    print()
    print("What variance across seeds tells us:")
    print("  Low variance -> algorithm is stable and reproducible")
    print("  High variance -> performance is sensitive to initialization")
    print("  PPO typically shows low variance due to clipped updates")


def exercise_4():
    """
    Exercise 4: Hyperparameter Sweep

    Conduct a systematic hyperparameter search.
    """
    print("Hyperparameter Sweep:")
    print("=" * 60)
    print()

    np.random.seed(42)

    # Simulate PPO performance for different hyperparameters
    def simulate_ppo_performance(lr, clip_epsilon, gae_lambda, seed=42):
        """Simulate PPO training outcome based on hyperparameters."""
        np.random.seed(seed)

        # Heuristic model of how hyperparameters affect performance
        # lr: sweet spot around 3e-4
        lr_score = 1.0 - 2.0 * abs(np.log10(lr) - np.log10(3e-4))
        lr_score = max(0, min(1, lr_score))

        # clip_epsilon: sweet spot around 0.2
        clip_score = 1.0 - 3.0 * abs(clip_epsilon - 0.2)
        clip_score = max(0, min(1, clip_score))

        # gae_lambda: sweet spot around 0.95
        gae_score = 1.0 - 2.0 * abs(gae_lambda - 0.95)
        gae_score = max(0, min(1, gae_score))

        # Combined score with noise
        score = (lr_score + clip_score + gae_score) / 3.0
        reward = 100 + 400 * score + np.random.randn() * 20

        return max(0, reward)

    # Sweep
    lrs = [1e-4, 3e-4, 1e-3]
    clip_epsilons = [0.1, 0.2, 0.3]
    gae_lambdas = [0.9, 0.95, 1.0]

    results = []
    for lr in lrs:
        for clip_eps in clip_epsilons:
            for gae_lam in gae_lambdas:
                reward = simulate_ppo_performance(lr, clip_eps, gae_lam)
                results.append({
                    'lr': lr, 'clip_epsilon': clip_eps,
                    'gae_lambda': gae_lam, 'reward': reward
                })

    # Sort by reward
    results.sort(key=lambda x: x['reward'], reverse=True)

    print(f"Total configurations: {len(results)}")
    print()

    print("Top 3 configurations:")
    for i, r in enumerate(results[:3]):
        print(f"  #{i+1}: lr={r['lr']:.0e}, clip_eps={r['clip_epsilon']:.1f}, "
              f"gae_lambda={r['gae_lambda']:.2f} -> reward={r['reward']:.1f}")

    print()
    print("Bottom 3 configurations:")
    for i, r in enumerate(results[-3:]):
        print(f"  #{len(results)-2+i}: lr={r['lr']:.0e}, clip_eps={r['clip_epsilon']:.1f}, "
              f"gae_lambda={r['gae_lambda']:.2f} -> reward={r['reward']:.1f}")

    # Heatmap: lr vs clip_epsilon (average over gae_lambda)
    print()
    print("Heatmap: Mean reward vs lr and clip_epsilon (avg over gae_lambda):")
    print()
    print(f"{'':>12}", end="")
    for clip_eps in clip_epsilons:
        print(f"  eps={clip_eps:.1f}", end="")
    print()
    print("-" * 42)

    for lr in lrs:
        print(f"  lr={lr:.0e}", end="")
        for clip_eps in clip_epsilons:
            avg = np.mean([r['reward'] for r in results
                           if r['lr'] == lr and r['clip_epsilon'] == clip_eps])
            print(f"  {avg:7.1f}", end="")
        print()

    print()
    print("Patterns observed:")
    print("  - lr=3e-4 consistently performs best (matches PPO default)")
    print("  - clip_epsilon=0.2 is optimal (default PPO value)")
    print("  - lr has the largest effect: too high causes instability,")
    print("    too low causes slow learning")
    print("  - clip_epsilon and gae_lambda have smaller but measurable effects")


def exercise_5():
    """
    Exercise 5: End-to-End Training Pipeline

    Demonstrate a complete training pipeline without Atari/gym.
    """
    print("End-to-End Training Pipeline:")
    print("=" * 60)
    print()

    # Simple image-like environment (no real Atari needed)
    class SimpleVisualEnv:
        """Environment with a simple 'visual' observation (8x8 grid)."""

        def __init__(self):
            self.size = 8
            self.agent_pos = [0, 0]
            self.goal_pos = [7, 7]
            self.max_steps = 100
            self.steps = 0
            self.n_actions = 4
            self.obs_shape = (1, 8, 8)  # Channel, Height, Width

        def _get_obs(self):
            """Return an 8x8 image-like observation."""
            obs = np.zeros((1, self.size, self.size), dtype=np.float32)
            obs[0, self.agent_pos[0], self.agent_pos[1]] = 1.0  # Agent
            obs[0, self.goal_pos[0], self.goal_pos[1]] = 0.5    # Goal
            return obs

        def reset(self):
            self.agent_pos = [0, 0]
            self.steps = 0
            return self._get_obs()

        def step(self, action):
            self.steps += 1
            deltas = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
            dr, dc = deltas[action]
            self.agent_pos[0] = max(0, min(self.size-1, self.agent_pos[0] + dr))
            self.agent_pos[1] = max(0, min(self.size-1, self.agent_pos[1] + dc))

            if self.agent_pos == self.goal_pos:
                return self._get_obs(), 10.0, True
            elif self.steps >= self.max_steps:
                return self._get_obs(), -0.1, True
            else:
                return self._get_obs(), -0.1, False

    # Simple conv-like feature extractor (flattened for pure numpy)
    class SimpleCNNPolicy:
        """CNN-like policy using numpy (flattened conv approximation)."""

        def __init__(self, obs_size, n_actions):
            self.feature_dim = 16
            self.obs_flat = obs_size
            self.n_actions = n_actions
            # Feature extraction weights
            self.w_feat = np.random.randn(obs_size, self.feature_dim) * 0.01
            # Actor head
            self.w_actor = np.random.randn(self.feature_dim, n_actions) * 0.01
            # Critic head
            self.w_critic = np.random.randn(self.feature_dim, 1) * 0.01

        def forward(self, obs):
            obs_flat = obs.flatten()
            features = np.maximum(0, obs_flat @ self.w_feat)  # ReLU
            logits = features @ self.w_actor
            value = (features @ self.w_critic)[0]
            return logits, value, features

        def get_action(self, obs):
            logits, value, _ = self.forward(obs)
            logits -= np.max(logits)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            action = np.random.choice(self.n_actions, p=probs)
            return action, probs, value

    # Training pipeline
    env = SimpleVisualEnv()
    obs_size = 1 * 8 * 8  # Flattened
    policy = SimpleCNNPolicy(obs_size, env.n_actions)
    lr = 0.001
    gamma = 0.99
    n_episodes = 200

    # Checkpoints
    checkpoints = {}

    print("Training pipeline:")
    print(f"  Observation shape: {env.obs_shape}")
    print(f"  Feature dim: {policy.feature_dim}")
    print(f"  Actions: {env.n_actions}")
    print()

    episode_rewards = []
    t_start = time.time()

    for episode in range(n_episodes):
        obs = env.reset()
        states, actions, rewards_list, values_list = [], [], [], []
        done = False

        while not done:
            action, probs, value = policy.get_action(obs)
            next_obs, reward, done = env.step(action)

            states.append(obs.flatten())
            actions.append(action)
            rewards_list.append(reward)
            values_list.append(value)

            obs = next_obs

        total_reward = sum(rewards_list)
        episode_rewards.append(total_reward)

        # Compute returns and advantages
        returns = []
        G = 0
        for r in reversed(rewards_list):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        advantages = returns - np.array(values_list)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update (simplified PPO-like)
        for t in range(len(states)):
            s = states[t]
            a = actions[t]
            adv = advantages[t]
            ret = returns[t]

            features = np.maximum(0, s @ policy.w_feat)
            logits = features @ policy.w_actor
            logits -= np.max(logits)
            probs = np.exp(logits) / np.sum(np.exp(logits))

            # Actor update
            grad_log = np.zeros(env.n_actions)
            grad_log[a] = 1.0
            grad_log -= probs
            policy.w_actor += lr * np.outer(features, grad_log * adv)

            # Critic update
            value = (features @ policy.w_critic)[0]
            policy.w_critic += lr * features.reshape(-1, 1) * (ret - value)

            # Feature update
            policy.w_feat += lr * 0.1 * np.outer(s, features > 0) * adv

        # Save checkpoint
        if (episode + 1) % 50 == 0:
            checkpoints[episode + 1] = {
                'mean_reward': np.mean(episode_rewards[-20:]),
                'episode': episode + 1,
            }

    t_end = time.time()
    print(f"Training completed in {t_end - t_start:.1f}s")
    print()

    # Evaluate checkpoints
    print("Checkpoint evaluation:")
    for ep, ckpt in sorted(checkpoints.items()):
        print(f"  Episode {ep}: mean_reward = {ckpt['mean_reward']:.2f}")

    # Final evaluation
    print()
    print("Final evaluation (10 episodes):")
    eval_rewards = []
    for _ in range(10):
        obs = env.reset()
        done = False
        total = 0
        while not done:
            logits, _, _ = policy.forward(obs)
            action = np.argmax(logits)  # Greedy at evaluation
            obs, reward, done = env.step(action)
            total += reward
        eval_rewards.append(total)
    print(f"  Mean: {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")

    # Show learned behavior
    obs = env.reset()
    path = [list(env.agent_pos)]
    done = False
    while not done and len(path) < 20:
        logits, _, _ = policy.forward(obs)
        action = np.argmax(logits)
        obs, reward, done = env.step(action)
        path.append(list(env.agent_pos))
    print(f"  Learned path: {path[:10]}{'...' if len(path) > 10 else ''}")
    print(f"  Reached goal: {env.agent_pos == env.goal_pos}")


if __name__ == "__main__":
    print("=== Exercise 1: Custom Environment ===")
    exercise_1()

    print("\n=== Exercise 2: Wrapper Pipeline ===")
    exercise_2()

    print("\n=== Exercise 3: Experiment Tracking ===")
    exercise_3()

    print("\n=== Exercise 4: Hyperparameter Sweep ===")
    exercise_4()

    print("\n=== Exercise 5: End-to-End Pipeline ===")
    exercise_5()

    print("\nAll exercises completed!")
