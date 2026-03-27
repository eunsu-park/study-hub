"""
Capstone: Training an RL Agent End-to-End — Example Code

Demonstrates a complete RL training pipeline:
  1. Environment setup with proper wrappers
  2. Agent implementation (tabular Q-learning as baseline)
  3. Training loop with logging and evaluation
  4. Hyperparameter sensitivity analysis
  5. Reproducibility checklist

No external dependencies required.
"""

import numpy as np
from collections import defaultdict
import time


# ============================================================
# 1. Environment + Wrappers
# ============================================================

class CliffWalkEnv:
    """
    Classic Cliff Walking environment (Sutton & Barto Example 6.6).
    4x12 grid. Cliff at bottom row columns 1-10.
    Start: (3,0), Goal: (3,11).
    """

    ROWS, COLS = 4, 12
    START = (3, 0)
    GOAL = (3, 11)
    CLIFF = {(3, c) for c in range(1, 11)}
    ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U

    def reset(self):
        self.pos = list(self.START)
        self.steps = 0
        return self._state()

    def _state(self):
        return (self.pos[0], self.pos[1])

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        r = max(0, min(self.ROWS - 1, self.pos[0] + dr))
        c = max(0, min(self.COLS - 1, self.pos[1] + dc))
        self.pos = [r, c]
        self.steps += 1

        state = self._state()
        if state in self.CLIFF:
            # Fall off cliff: big penalty, return to start
            self.pos = list(self.START)
            return self._state(), -100.0, False
        elif state == self.GOAL:
            return state, -1.0, True
        else:
            return state, -1.0, False

    def all_states(self):
        return [(r, c) for r in range(self.ROWS) for c in range(self.COLS)]


class EpisodeLimitWrapper:
    """Wraps environment to limit episode length."""

    def __init__(self, env, max_steps=200):
        self.env = env
        self.max_steps = max_steps
        self._steps = 0

    def reset(self):
        self._steps = 0
        return self.env.reset()

    def step(self, action):
        self._steps += 1
        s, r, done = self.env.step(action)
        if self._steps >= self.max_steps:
            done = True
        return s, r, done


# ============================================================
# 2. Agent
# ============================================================

class QLearningAgent:
    """
    Tabular Q-Learning agent with epsilon-greedy exploration.
    """

    def __init__(self, n_actions=4, gamma=0.99, lr=0.1,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, ns, done):
        target = r if done else r + self.gamma * np.max(self.Q[ns])
        self.Q[s][a] += self.lr * (target - self.Q[s][a])
        self.total_steps += 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        return {s: int(np.argmax(v)) for s, v in self.Q.items()}


# ============================================================
# 3. Training Loop
# ============================================================

class TrainingLogger:
    """Tracks metrics during training."""

    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_log = []

    def log(self, reward, length, epsilon):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.epsilon_log.append(epsilon)

    def rolling_mean(self, window=100):
        if len(self.episode_rewards) < window:
            return np.mean(self.episode_rewards)
        return np.mean(self.episode_rewards[-window:])

    def summary(self, window=100):
        n = len(self.episode_rewards)
        return {
            'total_episodes': n,
            'mean_reward_last': self.rolling_mean(window),
            'mean_length_last': np.mean(self.episode_lengths[-window:]),
            'final_epsilon': self.epsilon_log[-1] if self.epsilon_log else None,
        }


def evaluate_agent(agent, env_class, n_eval=200, seed=9999):
    """Evaluate agent in greedy mode (epsilon=0)."""
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0

    rewards = []
    np.random.seed(seed)
    for _ in range(n_eval):
        env = EpisodeLimitWrapper(env_class(), max_steps=100)
        state = env.reset()
        total_r = 0.0
        done = False
        while not done:
            a = agent.select_action(state)
            state, r, done = env.step(a)
            total_r += r
        rewards.append(total_r)

    agent.epsilon = saved_epsilon
    return np.mean(rewards), np.std(rewards)


def train(env_class, agent, n_episodes=1000, seed=42):
    """Full training loop."""
    np.random.seed(seed)
    logger = TrainingLogger()

    for ep in range(n_episodes):
        env = EpisodeLimitWrapper(env_class(), max_steps=200)
        state = env.reset()
        total_r = 0.0
        done = False
        step = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_r += reward
            step += 1

        agent.decay_epsilon()
        logger.log(total_r, step, agent.epsilon)

    return logger


def demonstrate_training_pipeline():
    """Run the full RL training pipeline with logging."""
    print("=" * 60)
    print("1. Complete RL Training Pipeline")
    print("=" * 60)

    agent = QLearningAgent(n_actions=4, gamma=0.99, lr=0.1,
                           epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.998)

    print("\nTraining Q-Learning on CliffWalking...")
    start_time = time.time()
    logger = train(CliffWalkEnv, agent, n_episodes=1000)
    elapsed = time.time() - start_time

    # Progress report
    window = 200
    print(f"\n  Training progress ({window}-ep rolling average):")
    print(f"  {'Episodes':>12} | {'Mean Reward':>12} | {'Epsilon':>8}")
    print("  " + "-" * 40)
    for i in range(0, 1000, window):
        end = i + window
        mr = np.mean(logger.episode_rewards[i:end])
        eps = np.mean(logger.epsilon_log[i:end])
        print(f"  {i+1:>5}-{end:<5}   | {mr:>12.2f} | {eps:>8.4f}")

    # Evaluation
    mean_r, std_r = evaluate_agent(agent, CliffWalkEnv)
    print(f"\n  Final evaluation (greedy, 200 episodes):")
    print(f"  Mean reward = {mean_r:.2f} ± {std_r:.2f}")
    print(f"  Training time: {elapsed:.2f}s")

    # Optimal path via greedy policy
    print(f"\n  Optimal policy actions from start to goal:")
    env = CliffWalkEnv()
    state = env.reset()
    path = [state]
    action_names = ['→', '←', '↓', '↑']
    actions_taken = []
    for _ in range(30):
        a = int(np.argmax(agent.Q[state]))
        state, _, done = env.step(a)
        path.append(state)
        actions_taken.append(action_names[a])
        if done:
            break
    print(f"  Path: {' '.join(actions_taken)}")
    print(f"  Length: {len(path)} steps")

    return agent, logger


# ============================================================
# 4. Hyperparameter Sensitivity
# ============================================================

def hyperparameter_sweep():
    """Run ablation over key hyperparameters."""
    print("\n" + "=" * 60)
    print("2. Hyperparameter Sensitivity Analysis")
    print("=" * 60)

    n_seeds = 3
    n_episodes = 500
    eval_window = 100

    configs = [
        {'lr': 0.01, 'gamma': 0.99},
        {'lr': 0.10, 'gamma': 0.99},
        {'lr': 0.50, 'gamma': 0.99},
        {'lr': 0.10, 'gamma': 0.50},
        {'lr': 0.10, 'gamma': 0.95},
    ]

    print(f"\n  {'lr':>6} | {'gamma':>6} | {'Mean Reward':>12} | {'Std Reward':>12}")
    print("  " + "-" * 48)

    for cfg in configs:
        seed_rewards = []
        for seed in range(n_seeds):
            agent = QLearningAgent(n_actions=4, lr=cfg['lr'], gamma=cfg['gamma'],
                                   epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995)
            logger = train(CliffWalkEnv, agent, n_episodes=n_episodes, seed=seed)
            seed_rewards.append(np.mean(logger.episode_rewards[-eval_window:]))

        print(f"  {cfg['lr']:>6.2f} | {cfg['gamma']:>6.2f} | "
              f"{np.mean(seed_rewards):>12.2f} | {np.std(seed_rewards):>12.2f}")

    print("\n  Key insights:")
    print("  - Too small lr: slow learning")
    print("  - Too large lr: unstable Q-values")
    print("  - Low gamma: myopic agent ignores future rewards")


# ============================================================
# 5. Reproducibility Checklist
# ============================================================

def reproducibility_checklist():
    """Demonstrate reproducibility practices."""
    print("\n" + "=" * 60)
    print("3. Reproducibility Checklist")
    print("=" * 60)

    # Same seed -> same results
    results = []
    for _ in range(3):
        agent = QLearningAgent(n_actions=4, lr=0.1, gamma=0.99,
                               epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995)
        logger = train(CliffWalkEnv, agent, n_episodes=200, seed=42)
        results.append(np.mean(logger.episode_rewards[-50:]))

    print(f"\n  Three runs with seed=42: {[f'{r:.4f}' for r in results]}")
    print(f"  All identical: {len(set(results)) == 1}")

    # Different seed -> different results
    diff_seeds = []
    for s in [0, 1, 2, 3, 4]:
        agent = QLearningAgent(n_actions=4, lr=0.1, gamma=0.99,
                               epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995)
        logger = train(CliffWalkEnv, agent, n_episodes=300, seed=s)
        diff_seeds.append(np.mean(logger.episode_rewards[-50:]))

    mean = np.mean(diff_seeds)
    std = np.std(diff_seeds)
    print(f"\n  5 independent seeds: {[f'{r:.1f}' for r in diff_seeds]}")
    print(f"  Mean ± Std: {mean:.2f} ± {std:.2f}")
    print(f"\n  Best practices:")
    print(f"  1. Report mean ± std over >= 5 seeds")
    print(f"  2. Fix all random seeds (numpy, environment)")
    print(f"  3. Log all hyperparameters")
    print(f"  4. Save checkpoints at regular intervals")
    print(f"  5. Separate training and evaluation environments")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    agent, logger = demonstrate_training_pipeline()
    hyperparameter_sweep()
    reproducibility_checklist()

    print("\n" + "=" * 60)
    print("Capstone RL Agent examples complete!")
    print("=" * 60)
