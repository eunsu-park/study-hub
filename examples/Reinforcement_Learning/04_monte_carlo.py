"""
Monte Carlo Methods Implementation
- First-visit MC Prediction
- Every-visit MC Prediction
- MC Control (Exploring Starts)
- On-policy MC Control (epsilon-greedy)
- Off-policy MC (Importance Sampling)
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym


def calculate_returns(episode, gamma=0.99):
    """
    Compute returns at each time step from an episode

    Args:
        episode: List of [(state, action, reward), ...]
        gamma: Discount factor

    Returns:
        returns: [(state, G), ...] returns at each time step
    """
    G = 0  # Initialize return
    returns = []

    # Compute in reverse order (efficient computation)
    for t in range(len(episode) - 1, -1, -1):
        state, action, reward = episode[t]
        G = reward + gamma * G  # Discounted return
        returns.insert(0, (state, G))

    return returns


def first_visit_mc_prediction(env, policy, n_episodes=10000, gamma=0.99):
    """
    First-visit MC Policy Evaluation

    Args:
        env: Gymnasium environment
        policy: Policy function policy(state) -> action
        n_episodes: Number of episodes
        gamma: Discount factor

    Returns:
        V: State value function
    """
    # Sum of returns and visit counts for each state
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)

    for episode_num in range(n_episodes):
        # Generate an episode
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        # First-visit: find the first visit index for each state
        visited = set()
        G = 0

        # Compute returns in reverse order
        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]

            G = gamma * G + reward_t

            # First-visit check
            if state_t not in visited:
                visited.add(state_t)
                returns_sum[state_t] += G
                returns_count[state_t] += 1
                V[state_t] = returns_sum[state_t] / returns_count[state_t]

        if (episode_num + 1) % 2000 == 0:
            print(f"Episode {episode_num + 1}/{n_episodes}")

    return dict(V)


def every_visit_mc_prediction(env, policy, n_episodes=10000, gamma=0.99):
    """
    Every-visit MC Policy Evaluation

    Counts all visits
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)

    for episode_num in range(n_episodes):
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        G = 0

        # Every-visit: update at every visit
        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]

            G = gamma * G + reward_t

            # Count all visits
            returns_sum[state_t] += G
            returns_count[state_t] += 1
            V[state_t] = returns_sum[state_t] / returns_count[state_t]

        if (episode_num + 1) % 2000 == 0:
            print(f"Episode {episode_num + 1}/{n_episodes}")

    return dict(V)


def epsilon_greedy_policy(Q, state, n_actions, epsilon=0.1):
    """
    Epsilon-greedy action selection

    Args:
        Q: Action value function
        state: Current state
        n_actions: Number of actions
        epsilon: Exploration probability

    Returns:
        action: Selected action
    """
    if np.random.random() < epsilon:
        # Exploration: random action
        return np.random.randint(n_actions)
    else:
        # Exploitation: best action
        return np.argmax(Q[state])


def mc_on_policy_control(env, n_episodes=100000, gamma=0.99,
                         epsilon=0.1, epsilon_decay=0.9999):
    """
    On-policy MC Control (epsilon-greedy)

    Args:
        env: Gymnasium environment
        n_episodes: Number of episodes
        gamma: Discount factor
        epsilon: Exploration rate
        epsilon_decay: Epsilon decay rate

    Returns:
        Q: Action value function
        policy: Learned policy
        episode_rewards: Rewards per episode
    """
    n_actions = env.action_space.n

    Q = defaultdict(lambda: np.zeros(n_actions))
    returns_sum = defaultdict(lambda: np.zeros(n_actions))
    returns_count = defaultdict(lambda: np.zeros(n_actions))

    episode_rewards = []

    print("Starting MC On-Policy Control training...")
    for episode_num in range(n_episodes):
        episode = []
        state, _ = env.reset()
        done = False
        total_reward = 0

        # Generate episode with epsilon-greedy policy
        while not done:
            action = epsilon_greedy_policy(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode.append((state, action, reward))
            total_reward += reward

            state = next_state
            done = terminated or truncated

        episode_rewards.append(total_reward)

        # Q update (First-visit)
        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            if (state_t, action_t) not in visited:
                visited.add((state_t, action_t))
                returns_sum[state_t][action_t] += G
                returns_count[state_t][action_t] += 1
                Q[state_t][action_t] = (returns_sum[state_t][action_t] /
                                        returns_count[state_t][action_t])

        # Decay epsilon
        epsilon = max(0.01, epsilon * epsilon_decay)

        if (episode_num + 1) % 10000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            print(f"Episode {episode_num + 1}: avg_reward = {avg_reward:.3f}, "
                  f"epsilon = {epsilon:.4f}")

    # Final greedy policy
    policy = {}
    for state in Q:
        policy[state] = np.argmax(Q[state])

    return dict(Q), policy, episode_rewards


def mc_off_policy_control(env, n_episodes=100000, gamma=0.99):
    """
    Off-policy MC Control (Weighted Importance Sampling)

    Behavior policy: epsilon-greedy (exploration)
    Target policy: greedy (exploitation)

    Returns:
        Q: Action value function
        target_policy: Target policy
        episode_rewards: Rewards per episode
    """
    n_actions = env.action_space.n

    Q = defaultdict(lambda: np.zeros(n_actions))
    C = defaultdict(lambda: np.zeros(n_actions))  # Cumulative weights

    episode_rewards = []
    epsilon = 0.1  # Behavior policy epsilon

    print("Starting MC Off-Policy Control training...")
    for episode_num in range(n_episodes):
        # Generate episode with behavior policy (epsilon-greedy)
        episode = []
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy_policy(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            total_reward += reward
            state = next_state
            done = terminated or truncated

        episode_rewards.append(total_reward)

        G = 0
        W = 1.0  # Importance sampling weight

        # Process in reverse order
        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            # Weighted importance sampling update
            C[state_t][action_t] += W
            Q[state_t][action_t] += (W / C[state_t][action_t] *
                                     (G - Q[state_t][action_t]))

            # Action under target policy (greedy)
            target_action = np.argmax(Q[state_t])

            # Break if action differs from target policy
            if action_t != target_action:
                break

            # Update importance ratio
            # pi(a|s) = 1 (deterministic), b(a|s) = (1-epsilon) + epsilon/|A| or epsilon/|A|
            if action_t == target_action:
                b_prob = (1 - epsilon) + epsilon / n_actions
            else:
                b_prob = epsilon / n_actions

            W = W * 1.0 / b_prob

        if (episode_num + 1) % 10000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            print(f"Episode {episode_num + 1}: avg_reward = {avg_reward:.3f}")

    # Final greedy policy
    target_policy = {}
    for state in Q:
        target_policy[state] = np.argmax(Q[state])

    return dict(Q), target_policy, episode_rewards


def blackjack_example():
    """MC learning in the Blackjack environment"""
    print("\n" + "=" * 60)
    print("Blackjack Example - MC On-Policy Control")
    print("=" * 60)

    env = gym.make('Blackjack-v1', sab=True)

    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions))
    returns_sum = defaultdict(lambda: np.zeros(n_actions))
    returns_count = defaultdict(lambda: np.zeros(n_actions))

    n_episodes = 500000
    gamma = 1.0
    epsilon = 0.1

    wins = 0
    losses = 0
    draws = 0

    print(f"\nTraining for {n_episodes} episodes...")
    for ep in range(n_episodes):
        episode = []
        state, _ = env.reset()
        done = False

        # Generate episode
        while not done:
            action = epsilon_greedy_policy(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        # Record result
        final_reward = episode[-1][2]
        if final_reward == 1:
            wins += 1
        elif final_reward == -1:
            losses += 1
        else:
            draws += 1

        # Q update
        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t

            if (state_t, action_t) not in visited:
                visited.add((state_t, action_t))
                returns_sum[state_t][action_t] += G
                returns_count[state_t][action_t] += 1
                Q[state_t][action_t] = (returns_sum[state_t][action_t] /
                                        returns_count[state_t][action_t])

        if (ep + 1) % 100000 == 0:
            win_rate = wins / (ep + 1)
            print(f"Episode {ep + 1}: win rate = {win_rate:.3f}")

    env.close()

    # Final statistics
    print("\nTraining complete!")
    print(f"Total episodes: {n_episodes}")
    print(f"Wins: {wins} ({wins/n_episodes*100:.1f}%)")
    print(f"Losses: {losses} ({losses/n_episodes*100:.1f}%)")
    print(f"Draws: {draws} ({draws/n_episodes*100:.1f}%)")
    print(f"Number of learned state-action pairs: {len(Q)}")

    # Policy visualization
    visualize_blackjack_policy(Q)

    return Q


def visualize_blackjack_policy(Q):
    """Visualize the Blackjack policy"""
    print("\n" + "=" * 60)
    print("Learned Blackjack Policy")
    print("=" * 60)
    print("H: Hit (draw a card), S: Stick (hold)")

    print("\n=== Without a usable ace ===")
    print("       Dealer's Card")
    print("Sum    A  2  3  4  5  6  7  8  9  10")
    print("-" * 50)

    for player_sum in range(21, 11, -1):
        row = f"{player_sum:2d}:   "
        for dealer in range(1, 11):
            state = (player_sum, dealer, False)
            if state in Q:
                action = np.argmax(Q[state])
                row += "H  " if action == 1 else "S  "
            else:
                row += "?  "
        print(row)

    print("\n=== With a usable ace ===")
    print("       Dealer's Card")
    print("Sum    A  2  3  4  5  6  7  8  9  10")
    print("-" * 50)

    for player_sum in range(21, 11, -1):
        row = f"{player_sum:2d}:   "
        for dealer in range(1, 11):
            state = (player_sum, dealer, True)
            if state in Q:
                action = np.argmax(Q[state])
                row += "H  " if action == 1 else "S  "
            else:
                row += "?  "
        print(row)


def plot_learning_curve(episode_rewards, window=1000):
    """Visualize the learning curve"""
    # Compute moving average
    moving_avg = []
    for i in range(len(episode_rewards)):
        start = max(0, i - window + 1)
        moving_avg.append(np.mean(episode_rewards[start:i+1]))

    plt.figure(figsize=(12, 6))
    plt.plot(moving_avg, label=f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Monte Carlo Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mc_learning_curve.png', dpi=150)
    print("Learning curve saved: mc_learning_curve.png")


def compare_mc_methods():
    """Compare MC methods"""
    print("=" * 60)
    print("Monte Carlo Methods Comparison")
    print("=" * 60)

    env = gym.make('Blackjack-v1', sab=True)

    # 1. First-visit MC Prediction
    print("\n[1] First-visit MC Prediction (random policy)")
    print("-" * 60)

    def random_policy(state):
        return env.action_space.sample()

    V_first = first_visit_mc_prediction(env, random_policy, n_episodes=10000)
    print(f"Number of estimated states: {len(V_first)}")
    print(f"Sample state values: {list(V_first.items())[:5]}")

    # 2. Every-visit MC Prediction
    print("\n[2] Every-visit MC Prediction (random policy)")
    print("-" * 60)
    V_every = every_visit_mc_prediction(env, random_policy, n_episodes=10000)
    print(f"Number of estimated states: {len(V_every)}")

    # 3. On-policy MC Control
    print("\n[3] On-policy MC Control (epsilon-greedy)")
    print("-" * 60)
    Q_on, policy_on, rewards_on = mc_on_policy_control(
        env, n_episodes=50000, gamma=1.0, epsilon=0.1
    )
    print(f"Number of learned state-action pairs: {len(Q_on)}")
    print(f"Final average reward: {np.mean(rewards_on[-1000:]):.3f}")

    # 4. Off-policy MC Control
    print("\n[4] Off-policy MC Control (Importance Sampling)")
    print("-" * 60)
    Q_off, policy_off, rewards_off = mc_off_policy_control(
        env, n_episodes=50000, gamma=1.0
    )
    print(f"Number of learned state-action pairs: {len(Q_off)}")
    print(f"Final average reward: {np.mean(rewards_off[-1000:]):.3f}")

    env.close()

    # Learning curve comparison
    plt.figure(figsize=(12, 6))

    window = 1000
    moving_avg_on = [np.mean(rewards_on[max(0, i-window+1):i+1])
                     for i in range(len(rewards_on))]
    moving_avg_off = [np.mean(rewards_off[max(0, i-window+1):i+1])
                      for i in range(len(rewards_off))]

    plt.plot(moving_avg_on, label='On-policy MC', alpha=0.7)
    plt.plot(moving_avg_off, label='Off-policy MC', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('On-policy vs Off-policy MC Control')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mc_comparison.png', dpi=150)
    print("\nComparison graph saved: mc_comparison.png")

    return Q_on, policy_on, Q_off, policy_off


if __name__ == "__main__":
    # MC methods comparison
    try:
        Q_on, policy_on, Q_off, policy_off = compare_mc_methods()

        # Blackjack example
        Q_blackjack = blackjack_example()

    except Exception as e:
        print(f"\nExecution failed: {e}")
        print("Make sure the gymnasium package is installed: pip install gymnasium")

    print("\n" + "=" * 60)
    print("Monte Carlo methods examples complete!")
    print("=" * 60)
