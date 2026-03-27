"""
Multi-Agent RL: IQL and Simple Cooperative/Competitive Environments
Basic concepts of multi-agent reinforcement learning implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class IQLAgent:
    """Independent Q-Learning Agent"""

    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.1):
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

    def choose_action(self, obs):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(obs))
            return q_values.argmax().item()

    def update(self, obs, action, reward, next_obs, done):
        """Q-learning update"""
        obs_tensor = torch.FloatTensor(obs)
        next_obs_tensor = torch.FloatTensor(next_obs)

        current_q = self.q_network(obs_tensor)[action]

        with torch.no_grad():
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * self.q_network(next_obs_tensor).max()

        loss = (current_q - target_q) ** 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class SimpleGridWorld:
    """
    Simple 2-agent grid environment
    - Grid: 5x5
    - Goal: Both agents reach their respective goal positions
    - Cooperative element: Bonus reward if agents share the same cell
    """

    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.n_agents = 2

        # Actions: up, down, left, right, stay
        self.action_dim = 5
        self.obs_dim = 4  # (x, y, goal_x, goal_y)

        self.reset()

    def reset(self):
        """Reset the environment"""
        # Random initial agent positions
        self.agent_pos = [
            [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
            for _ in range(self.n_agents)
        ]

        # Fixed goal positions
        self.goals = [
            [0, self.grid_size - 1],  # Agent 0's goal
            [self.grid_size - 1, 0]   # Agent 1's goal
        ]

        self.steps = 0
        return self.get_observations()

    def get_observations(self):
        """Return each agent's observation"""
        observations = []
        for i in range(self.n_agents):
            obs = [
                self.agent_pos[i][0] / self.grid_size,
                self.agent_pos[i][1] / self.grid_size,
                self.goals[i][0] / self.grid_size,
                self.goals[i][1] / self.grid_size
            ]
            observations.append(np.array(obs, dtype=np.float32))
        return observations

    def step(self, actions):
        """Environment step"""
        self.steps += 1
        rewards = [0.0, 0.0]

        # Move each agent
        for i, action in enumerate(actions):
            x, y = self.agent_pos[i]

            # Apply action: up(0), down(1), left(2), right(3), stay(4)
            if action == 0:  # up
                x = max(0, x - 1)
            elif action == 1:  # down
                x = min(self.grid_size - 1, x + 1)
            elif action == 2:  # left
                y = max(0, y - 1)
            elif action == 3:  # right
                y = min(self.grid_size - 1, y + 1)
            # action == 4: stay

            self.agent_pos[i] = [x, y]

            # Goal reached reward
            if self.agent_pos[i] == self.goals[i]:
                rewards[i] += 10.0

            # Small penalty per step
            rewards[i] -= 0.01

        # Cooperation bonus: agents in the same cell
        if self.agent_pos[0] == self.agent_pos[1]:
            rewards[0] += 1.0
            rewards[1] += 1.0

        # Termination condition: both reached their goals or max steps
        done = (
            (self.agent_pos[0] == self.goals[0] and self.agent_pos[1] == self.goals[1])
            or self.steps >= 50
        )

        observations = self.get_observations()
        return observations, rewards, done


class CompetitiveGridWorld:
    """
    Competitive environment: Two agents compete for a single reward
    The first agent to reach the goal takes the reward
    """

    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.n_agents = 2
        self.action_dim = 5
        self.obs_dim = 4

        self.reset()

    def reset(self):
        """Reset the environment"""
        # Agent initial positions
        self.agent_pos = [
            [0, 0],
            [self.grid_size - 1, self.grid_size - 1]
        ]

        # Shared goal position (center)
        self.goal = [self.grid_size // 2, self.grid_size // 2]
        self.goal_taken = False
        self.steps = 0

        return self.get_observations()

    def get_observations(self):
        """Return each agent's observation"""
        observations = []
        for i in range(self.n_agents):
            obs = [
                self.agent_pos[i][0] / self.grid_size,
                self.agent_pos[i][1] / self.grid_size,
                self.goal[0] / self.grid_size,
                self.goal[1] / self.grid_size
            ]
            observations.append(np.array(obs, dtype=np.float32))
        return observations

    def step(self, actions):
        """Environment step"""
        self.steps += 1
        rewards = [0.0, 0.0]

        # Move each agent
        for i, action in enumerate(actions):
            x, y = self.agent_pos[i]

            if action == 0:  # up
                x = max(0, x - 1)
            elif action == 1:  # down
                x = min(self.grid_size - 1, x + 1)
            elif action == 2:  # left
                y = max(0, y - 1)
            elif action == 3:  # right
                y = min(self.grid_size - 1, y + 1)

            self.agent_pos[i] = [x, y]

            # Goal reached check (only the first agent gets the reward)
            if not self.goal_taken and self.agent_pos[i] == self.goal:
                rewards[i] += 10.0
                self.goal_taken = True

            # Small penalty per step
            rewards[i] -= 0.01

        # Termination condition
        done = self.goal_taken or self.steps >= 50

        observations = self.get_observations()
        return observations, rewards, done


class IQLSystem:
    """Multi-agent IQL system"""

    def __init__(self, n_agents, obs_dim, action_dim):
        self.agents = [
            IQLAgent(obs_dim, action_dim)
            for _ in range(n_agents)
        ]
        self.n_agents = n_agents

    def choose_actions(self, observations):
        """Select actions for all agents"""
        return [
            agent.choose_action(obs)
            for agent, obs in zip(self.agents, observations)
        ]

    def update(self, observations, actions, rewards, next_observations, done):
        """Update all agents"""
        losses = []
        for i, agent in enumerate(self.agents):
            loss = agent.update(
                observations[i], actions[i],
                rewards[i], next_observations[i], done
            )
            losses.append(loss)
        return losses


def train_cooperative():
    """IQL training in cooperative environment"""
    print("=== Cooperative Environment Training ===\n")

    env = SimpleGridWorld(grid_size=5)
    system = IQLSystem(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim
    )

    n_episodes = 1000
    episode_rewards = []

    for episode in range(n_episodes):
        observations = env.reset()
        total_rewards = [0.0, 0.0]
        done = False

        while not done:
            actions = system.choose_actions(observations)
            next_observations, rewards, done = env.step(actions)

            system.update(observations, actions, rewards, next_observations, done)

            observations = next_observations
            total_rewards[0] += rewards[0]
            total_rewards[1] += rewards[1]

        episode_rewards.append(sum(total_rewards) / 2)

        # Decay epsilon
        for agent in system.agents:
            agent.epsilon = max(0.01, agent.epsilon * 0.995)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

    return episode_rewards


def train_competitive():
    """IQL training in competitive environment"""
    print("\n=== Competitive Environment Training ===\n")

    env = CompetitiveGridWorld(grid_size=5)
    system = IQLSystem(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim
    )

    n_episodes = 1000
    agent0_wins = []
    agent1_wins = []

    for episode in range(n_episodes):
        observations = env.reset()
        episode_rewards = [0.0, 0.0]
        done = False

        while not done:
            actions = system.choose_actions(observations)
            next_observations, rewards, done = env.step(actions)

            system.update(observations, actions, rewards, next_observations, done)

            observations = next_observations
            episode_rewards[0] += rewards[0]
            episode_rewards[1] += rewards[1]

        # Record winner
        agent0_wins.append(1 if episode_rewards[0] > episode_rewards[1] else 0)
        agent1_wins.append(1 if episode_rewards[1] > episode_rewards[0] else 0)

        # Decay epsilon
        for agent in system.agents:
            agent.epsilon = max(0.01, agent.epsilon * 0.995)

        if (episode + 1) % 100 == 0:
            win_rate_0 = np.mean(agent0_wins[-100:]) * 100
            win_rate_1 = np.mean(agent1_wins[-100:]) * 100
            print(f"Episode {episode + 1}, Win Rate - Agent0: {win_rate_0:.1f}%, Agent1: {win_rate_1:.1f}%")

    return agent0_wins, agent1_wins


def visualize_results(coop_rewards, comp_wins):
    """Visualize training results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cooperative environment rewards
    axes[0].plot(coop_rewards, alpha=0.3, color='blue')
    window = 50
    smoothed = np.convolve(coop_rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(smoothed, color='blue', linewidth=2)
    axes[0].set_title('Cooperative Environment: Average Reward')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].grid(True, alpha=0.3)

    # Competitive environment win rates
    agent0_wins, agent1_wins = comp_wins
    window = 50
    win_rate_0 = np.convolve(agent0_wins, np.ones(window)/window, mode='valid') * 100
    win_rate_1 = np.convolve(agent1_wins, np.ones(window)/window, mode='valid') * 100

    axes[1].plot(win_rate_0, label='Agent 0', linewidth=2)
    axes[1].plot(win_rate_1, label='Agent 1', linewidth=2)
    axes[1].axhline(y=50, color='r', linestyle='--', alpha=0.3, label='Equilibrium')
    axes[1].set_title('Competitive Environment: Win Rate')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('multi_agent_results.png', dpi=100, bbox_inches='tight')
    print("\nGraph saved: multi_agent_results.png")


def demonstrate_ctde_concept():
    """
    CTDE (Centralized Training, Decentralized Execution) concept explanation
    Uses global information during training but only local observations during execution
    """
    print("\n=== CTDE Paradigm Concept ===\n")
    print("Training phase:")
    print("  - Critic: Has access to all agents' observations + actions")
    print("  - Learns value function with global state")
    print("\nExecution phase:")
    print("  - Actor: Uses only local observations")
    print("  - Decentralized execution, no communication needed")
    print("\nAdvantages:")
    print("  - Easier to discover cooperative patterns during training")
    print("  - Good scalability during execution")
    print("  - Works in partially observable environments")


if __name__ == "__main__":
    print("Multi-Agent Reinforcement Learning Examples\n")

    # Cooperative environment training
    coop_rewards = train_cooperative()

    # Competitive environment training
    agent0_wins, agent1_wins = train_competitive()

    # CTDE concept explanation
    demonstrate_ctde_concept()

    # Visualize results
    visualize_results(coop_rewards, (agent0_wins, agent1_wins))

    print("\nTraining complete!")
    print("\nKey concepts:")
    print("1. IQL: Each agent independently performs Q-learning")
    print("2. Non-stationarity: Environment is dynamic due to other agents' changing policies")
    print("3. Cooperation vs Competition: Different learning dynamics depending on reward structure")
    print("4. CTDE: Centralized Training, Decentralized Execution paradigm")
