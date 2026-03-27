"""
Practical RL Project: Complete Trading Environment Implementation
Includes project structure, training pipeline, model save/load, evaluation, and visualization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import json
import os


# =============================================================================
# 1. Custom Environment: Simple Trading Environment
# =============================================================================

class SimpleTradingEnv:
    """
    Simple stock trading environment
    - Observation: Current price, moving averages, shares held
    - Actions: 0(sell), 1(hold), 2(buy)
    - Reward: Portfolio value change
    """

    def __init__(self, initial_balance=10000, stock_dim=1, max_steps=100):
        self.initial_balance = initial_balance
        self.stock_dim = stock_dim
        self.max_steps = max_steps

        # Observation space: [price, 5-day MA, 20-day MA, shares held, cash ratio]
        self.obs_dim = 5
        # Action space: sell(0), hold(1), buy(2)
        self.action_dim = 3

        self.reset()

    def _generate_price_series(self):
        """Generate price time series (random walk + trend)"""
        trend = np.random.choice([-1, 0, 1])  # down, sideways, up
        prices = [100.0]

        for _ in range(self.max_steps):
            # Random walk + trend
            change = np.random.randn() * 2 + trend * 0.5
            new_price = max(50.0, prices[-1] + change)  # Minimum price limit
            prices.append(new_price)

        return np.array(prices)

    def reset(self):
        """Reset the environment"""
        self.prices = self._generate_price_series()
        self.current_step = 0

        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0

        return self._get_observation()

    def _get_observation(self):
        """Return current observation"""
        # Price information
        current_price = self.prices[self.current_step]

        # Moving average calculation
        start_5 = max(0, self.current_step - 5)
        start_20 = max(0, self.current_step - 20)
        ma5 = np.mean(self.prices[start_5:self.current_step + 1])
        ma20 = np.mean(self.prices[start_20:self.current_step + 1])

        # Portfolio information
        total_value = self.balance + self.shares_held * current_price
        cash_ratio = self.balance / total_value if total_value > 0 else 0

        obs = np.array([
            current_price / 100.0,  # Normalized
            ma5 / 100.0,
            ma20 / 100.0,
            self.shares_held / 100.0,
            cash_ratio
        ], dtype=np.float32)

        return obs

    def step(self, action):
        """Environment step"""
        current_price = self.prices[self.current_step]
        prev_value = self.balance + self.shares_held * current_price

        # Execute action
        if action == 0:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price * 0.99  # 1% commission
                self.total_shares_sold += self.shares_held
                self.shares_held = 0

        elif action == 2:  # Buy
            shares_to_buy = self.balance // current_price
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * 1.01  # 1% commission
                if cost <= self.balance:
                    self.shares_held += shares_to_buy
                    self.balance -= cost
                    self.total_shares_bought += shares_to_buy

        # Advance to next step
        self.current_step += 1

        # Compute reward: portfolio value change
        next_price = self.prices[self.current_step]
        current_value = self.balance + self.shares_held * next_price
        reward = (current_value - prev_value) / prev_value

        # Termination condition
        done = self.current_step >= self.max_steps - 1

        # Final reward bonus
        if done:
            # Bonus/penalty based on final return
            total_return = (current_value - self.initial_balance) / self.initial_balance
            reward += total_return * 10

        return self._get_observation(), reward, done, {}


# =============================================================================
# 2. Network Architecture
# =============================================================================

class TradingPolicyNetwork(nn.Module):
    """Trading policy network"""

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        features = self.feature_extractor(obs)
        return F.softmax(self.actor(features), dim=-1), self.critic(features)

    def get_action_and_value(self, obs, action=None):
        probs, value = self.forward(obs)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)


# =============================================================================
# 3. PPO Agent (for project)
# =============================================================================

class PPOAgent:
    """PPO Agent for project"""

    def __init__(self, config):
        self.config = config

        # Create network
        self.network = TradingPolicyNetwork(
            obs_dim=config['obs_dim'],
            action_dim=config['action_dim'],
            hidden_dim=config.get('hidden_dim', 128)
        )

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.get('lr', 3e-4)
        )

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.n_epochs = config.get('n_epochs', 10)
        self.batch_size = config.get('batch_size', 64)

    def collect_rollout(self, env, n_steps):
        """Collect experience"""
        rollout = {
            'obs': [], 'actions': [], 'rewards': [], 'dones': [],
            'values': [], 'log_probs': []
        }

        obs = env.reset()

        for _ in range(n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)

            next_obs, reward, done, _ = env.step(action.item())

            rollout['obs'].append(obs)
            rollout['actions'].append(action.item())
            rollout['rewards'].append(reward)
            rollout['dones'].append(done)
            rollout['values'].append(value.item())
            rollout['log_probs'].append(log_prob.item())

            obs = next_obs if not done else env.reset()

        # Estimate last value
        with torch.no_grad():
            _, _, _, last_value = self.network.get_action_and_value(
                torch.FloatTensor(obs).unsqueeze(0)
            )
            rollout['last_value'] = last_value.item()

        # Convert to NumPy arrays
        for key in ['obs', 'actions', 'rewards', 'dones', 'values', 'log_probs']:
            rollout[key] = np.array(rollout[key])

        return rollout

    def compute_gae(self, rollout):
        """Compute GAE"""
        rewards = rollout['rewards']
        values = rollout['values']
        dones = rollout['dones']
        last_value = rollout['last_value']

        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_val = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """PPO update"""
        advantages, returns = self.compute_gae(rollout)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs = torch.FloatTensor(rollout['obs'])
        actions = torch.LongTensor(rollout['actions'])
        old_log_probs = torch.FloatTensor(rollout['log_probs'])
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)

        # Train for multiple epochs
        total_loss = 0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = np.random.permutation(len(obs))

            for start in range(0, len(obs), self.batch_size):
                idx = indices[start:start + self.batch_size]

                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    obs[idx], actions[idx]
                )

                # PPO loss
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages_tensor[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor[idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, returns_tensor[idx])
                entropy_loss = -entropy.mean()

                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        return total_loss / n_updates if n_updates > 0 else 0

    def save(self, filepath):
        """Save model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)

    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# =============================================================================
# 4. Training Pipeline
# =============================================================================

class TrainingLogger:
    """Training logger"""

    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.metrics = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'returns': []
        }

    def log(self, episode, reward, loss, portfolio_return):
        """Record metrics"""
        self.metrics['episodes'].append(episode)
        self.metrics['rewards'].append(reward)
        self.metrics['losses'].append(loss)
        self.metrics['returns'].append(portfolio_return)

    def save(self):
        """Save logs"""
        filepath = os.path.join(self.log_dir, 'training_log.json')
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def plot(self):
        """Visualize training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Episode rewards
        axes[0, 0].plot(self.metrics['episodes'], self.metrics['rewards'], alpha=0.3)
        if len(self.metrics['rewards']) > 10:
            window = min(50, len(self.metrics['rewards']) // 10)
            smoothed = np.convolve(self.metrics['rewards'], np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.metrics['rewards'])), smoothed, linewidth=2)
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        axes[0, 1].plot(self.metrics['episodes'], self.metrics['losses'])
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # Portfolio return
        axes[1, 0].plot(self.metrics['episodes'], self.metrics['returns'], alpha=0.3)
        if len(self.metrics['returns']) > 10:
            window = min(50, len(self.metrics['returns']) // 10)
            smoothed = np.convolve(self.metrics['returns'], np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(self.metrics['returns'])), smoothed, linewidth=2)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[1, 0].set_title('Portfolio Return')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Return (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # Return distribution
        axes[1, 1].hist(self.metrics['returns'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Return Distribution')
        axes[1, 1].set_xlabel('Return (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'), dpi=100, bbox_inches='tight')
        print(f"Training graph saved: {self.log_dir}/training_progress.png")


def train_agent(config):
    """Train agent"""
    # Create environment
    env = SimpleTradingEnv(
        initial_balance=config['initial_balance'],
        max_steps=config['max_steps']
    )

    # Create agent
    agent_config = {
        'obs_dim': env.obs_dim,
        'action_dim': env.action_dim,
        'hidden_dim': config['hidden_dim'],
        'lr': config['lr'],
        'gamma': config['gamma'],
        'gae_lambda': config['gae_lambda'],
        'clip_epsilon': config['clip_epsilon'],
        'n_epochs': config['n_epochs'],
        'batch_size': config['batch_size']
    }
    agent = PPOAgent(agent_config)

    # Logger
    logger = TrainingLogger(log_dir=config['log_dir'])

    # Training loop
    n_episodes = config['n_episodes']
    n_steps = config['n_steps']

    print("Starting training...\n")

    for episode in range(n_episodes):
        # Collect rollout
        rollout = agent.collect_rollout(env, n_steps)

        # Episode statistics
        episode_reward = rollout['rewards'].sum()
        final_obs = rollout['obs'][-1]
        portfolio_return = (final_obs[3] * 100 + final_obs[4] * config['initial_balance'] - config['initial_balance']) / config['initial_balance'] * 100

        # Update
        loss = agent.update(rollout)

        # Logging
        logger.log(episode, episode_reward, loss, portfolio_return)

        if (episode + 1) % config['log_interval'] == 0:
            avg_reward = np.mean(logger.metrics['rewards'][-config['log_interval']:])
            avg_return = np.mean(logger.metrics['returns'][-config['log_interval']:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Return: {avg_return:.2f}% | "
                  f"Loss: {loss:.4f}")

        # Save checkpoint
        if (episode + 1) % config['save_interval'] == 0:
            save_path = os.path.join(config['checkpoint_dir'], f'agent_ep{episode + 1}.pt')
            agent.save(save_path)
            print(f"  Checkpoint saved: {save_path}")

    # Save final model
    final_path = os.path.join(config['checkpoint_dir'], 'agent_final.pt')
    agent.save(final_path)

    # Save logs and visualize
    logger.save()
    logger.plot()

    print("\nTraining complete!")
    return agent, logger


# =============================================================================
# 5. Evaluation
# =============================================================================

def evaluate_agent(agent, n_episodes=10, render=False):
    """Evaluate the trained agent"""
    env = SimpleTradingEnv()
    episode_returns = []

    print("\n=== Agent Evaluation ===\n")

    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                action, _, _, _ = agent.network.get_action_and_value(obs_tensor)

            obs, reward, done, _ = env.step(action.item())
            total_reward += reward

        # Final return
        final_value = env.balance + env.shares_held * env.prices[env.current_step]
        portfolio_return = (final_value - env.initial_balance) / env.initial_balance * 100

        episode_returns.append(portfolio_return)
        print(f"Episode {episode + 1}: Return = {portfolio_return:.2f}%")

    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)

    print(f"\nAverage return: {mean_return:.2f}% +/- {std_return:.2f}%")

    return episode_returns


# =============================================================================
# 6. Main
# =============================================================================

if __name__ == "__main__":
    # Project configuration
    config = {
        # Environment
        'initial_balance': 10000,
        'max_steps': 100,

        # Network
        'hidden_dim': 128,

        # Training
        'lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'n_epochs': 10,
        'batch_size': 64,

        # Training parameters
        'n_episodes': 500,
        'n_steps': 100,
        'log_interval': 50,
        'save_interval': 100,

        # Directories
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints'
    }

    # Create directories
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    print("=" * 60)
    print("Practical RL Project: Trading Agent")
    print("=" * 60)

    # Print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Training
    agent, logger = train_agent(config)

    # Evaluation
    returns = evaluate_agent(agent, n_episodes=20)

    print("\nProject complete!")
    print("\nGenerated files:")
    print(f"  - {config['log_dir']}/training_log.json")
    print(f"  - {config['log_dir']}/training_progress.png")
    print(f"  - {config['checkpoint_dir']}/agent_final.pt")

    print("\nKey takeaways:")
    print("  1. Custom environment implementation (Gymnasium-style)")
    print("  2. Modular PPO agent")
    print("  3. Training pipeline (collect-update-log)")
    print("  4. Model save/load")
    print("  5. Evaluation and visualization")
    print("  6. Project structure best practices")
