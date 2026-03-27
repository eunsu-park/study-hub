"""
TD Learning (Temporal Difference Learning) Implementation
Includes TD(0), SARSA, Q-Learning, Expected SARSA
"""
import numpy as np
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt


class TD0Prediction:
    """TD(0) Policy Evaluation Algorithm"""

    def __init__(self, alpha=0.1, gamma=0.99):
        self.V = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, state, reward, next_state, done):
        """TD(0) value function update"""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.V[next_state]

        td_error = td_target - self.V[state]
        self.V[state] += self.alpha * td_error
        return td_error

    def get_value(self, state):
        return self.V[state]


class SARSA:
    """SARSA (On-policy TD Control)"""

    def __init__(self, n_actions, alpha=0.5, gamma=0.99, epsilon=0.1):
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        """Epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update: Q(s,a) <- Q(s,a) + alpha[r + gamma*Q(s',a') - Q(s,a)]"""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.Q[next_state][next_action]

        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        return td_error


class QLearning:
    """Q-Learning (Off-policy TD Control)"""

    def __init__(self, n_actions, alpha=0.5, gamma=0.99, epsilon=0.1):
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        """Epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        """Q-Learning update: Q(s,a) <- Q(s,a) + alpha[r + gamma * max Q(s',a') - Q(s,a)]"""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        return td_error


class ExpectedSARSA:
    """Expected SARSA"""

    def __init__(self, n_actions, alpha=0.5, gamma=0.99, epsilon=0.1):
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        """Epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        """Expected SARSA update: uses the expected value under the policy for the next state"""
        if done:
            td_target = reward
        else:
            # Compute expected value under epsilon-greedy policy
            best_action = np.argmax(self.Q[next_state])
            expected_q = 0.0
            for a in range(self.n_actions):
                if a == best_action:
                    prob = 1 - self.epsilon + self.epsilon / self.n_actions
                else:
                    prob = self.epsilon / self.n_actions
                expected_q += prob * self.Q[next_state][a]

            td_target = reward + self.gamma * expected_q

        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        return td_error


def train_sarsa(env_name='CliffWalking-v0', n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    """SARSA training"""
    env = gym.make(env_name)
    agent = SARSA(env.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon)

    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            total_reward += reward

        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"SARSA - Episode {episode + 1}: avg_reward = {avg:.1f}")

    env.close()
    return agent, episode_rewards


def train_qlearning(env_name='CliffWalking-v0', n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    """Q-Learning training"""
    env = gym.make(env_name)
    agent = QLearning(env.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon)

    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Q-Learning - Episode {episode + 1}: avg_reward = {avg:.1f}")

    env.close()
    return agent, episode_rewards


def compare_td_methods():
    """Compare TD methods: SARSA vs Q-Learning vs Expected SARSA"""
    print("=== Comparing TD Methods on CliffWalking ===\n")

    # SARSA (prefers the safe path)
    print("Training SARSA...")
    _, sarsa_rewards = train_sarsa(n_episodes=500)

    # Q-Learning (learns the optimal path, more risky)
    print("\nTraining Q-Learning...")
    _, qlearning_rewards = train_qlearning(n_episodes=500)

    # Expected SARSA
    print("\nTraining Expected SARSA...")
    env = gym.make('CliffWalking-v0')
    expected_sarsa = ExpectedSARSA(env.action_space.n, alpha=0.5, gamma=1.0, epsilon=0.1)
    expected_rewards = []

    for episode in range(500):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = expected_sarsa.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            expected_sarsa.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        expected_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(expected_rewards[-100:])
            print(f"Expected SARSA - Episode {episode + 1}: avg_reward = {avg:.1f}")

    env.close()

    # Learning curve visualization
    plot_comparison(sarsa_rewards, qlearning_rewards, expected_rewards)

    return sarsa_rewards, qlearning_rewards, expected_rewards


def plot_comparison(sarsa_rewards, qlearning_rewards, expected_rewards):
    """Visualize learning curve comparison"""
    window = 10

    def smooth(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(12, 5))

    # Raw data
    plt.subplot(1, 2, 1)
    plt.plot(sarsa_rewards, alpha=0.3, label='SARSA (raw)')
    plt.plot(qlearning_rewards, alpha=0.3, label='Q-Learning (raw)')
    plt.plot(expected_rewards, alpha=0.3, label='Expected SARSA (raw)')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('TD Methods Comparison - Raw Data')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Smoothed data
    plt.subplot(1, 2, 2)
    plt.plot(smooth(sarsa_rewards, window), label='SARSA (smoothed)', linewidth=2)
    plt.plot(smooth(qlearning_rewards, window), label='Q-Learning (smoothed)', linewidth=2)
    plt.plot(smooth(expected_rewards, window), label='Expected SARSA (smoothed)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward (smoothed)')
    plt.title(f'TD Methods Comparison - Smoothed (window={window})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('td_methods_comparison.png', dpi=150)
    print("\nLearning curves saved to 'td_methods_comparison.png'.")


def visualize_policy(agent, env_name='CliffWalking-v0'):
    """Visualize the learned policy (CliffWalking only)"""
    if env_name != 'CliffWalking-v0':
        print("Policy visualization is only supported for the CliffWalking environment.")
        return

    print("\n=== Learned Policy (4x12 grid) ===")
    arrows = {0: '^', 1: '>', 2: 'v', 3: '<'}

    for row in range(4):
        line = ""
        for col in range(12):
            state = row * 12 + col
            if state == 36:  # Start
                line += " S "
            elif state == 47:  # Goal
                line += " G "
            elif 37 <= state <= 46:  # Cliff
                line += " C "
            else:
                action = np.argmax(agent.Q[state])
                line += f" {arrows[action]} "
        print(line)

    print("\n(S: Start, G: Goal, C: Cliff, ^>v<: action direction)")


if __name__ == "__main__":
    # Compare TD methods
    sarsa_rewards, qlearning_rewards, expected_rewards = compare_td_methods()

    # SARSA policy visualization
    print("\n" + "="*50)
    env = gym.make('CliffWalking-v0')
    sarsa_agent = SARSA(env.action_space.n)

    # Retrain (for visualization)
    for episode in range(500):
        state, _ = env.reset()
        action = sarsa_agent.choose_action(state)
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = sarsa_agent.choose_action(next_state)
            sarsa_agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action

    print("\nSARSA training complete - prefers the safe path")
    visualize_policy(sarsa_agent)

    env.close()
