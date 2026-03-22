"""
Exercises for Lesson 42: Reinforcement Learning Introduction
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random


# === Exercise 1: Q-Learning on FrozenLake ===
# Problem: Implement Q-Learning for a simple grid world.

def exercise_1():
    """Q-Learning on a simple 4x4 grid world (simulated FrozenLake)."""
    np.random.seed(42)

    # Simple 4x4 grid: 16 states, 4 actions (up, down, left, right)
    n_states = 16
    n_actions = 4
    grid_size = 4

    # Define transitions (deterministic for simplicity)
    # State = row * 4 + col
    # Goal at state 15, hole at states 5, 7, 11, 12
    holes = {5, 7, 11, 12}
    goal = 15

    def step(state, action):
        row, col = state // grid_size, state % grid_size
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(grid_size - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(grid_size - 1, col + 1)
        next_state = row * grid_size + col
        if next_state == goal:
            return next_state, 1.0, True
        elif next_state in holes:
            return next_state, 0.0, True
        return next_state, 0.0, False

    # Q-Learning
    Q = np.zeros((n_states, n_actions))
    lr = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    n_episodes = 10000

    success_count = 0
    for episode in range(n_episodes):
        state = 0
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = Q[state].argmax()

            next_state, reward, done = step(state, action)
            Q[state, action] += lr * (reward + gamma * Q[next_state].max() - Q[state, action])
            state = next_state

        if state == goal:
            success_count += 1

        epsilon = max(0.01, epsilon * epsilon_decay)

        if episode >= n_episodes - 1000 and state == goal:
            success_count += 0  # Already counted

    # Evaluate
    last_1000_success = 0
    for _ in range(1000):
        state = 0
        done = False
        while not done:
            action = Q[state].argmax()
            state, _, done = step(state, action)
        if state == goal:
            last_1000_success += 1

    success_rate = last_1000_success / 1000
    print(f"  Success rate (last 1000 episodes): {success_rate:.3f}")

    # Print policy
    arrows = ["^", "v", "<", ">"]
    print(f"\n  Learned policy:")
    for row in range(grid_size):
        line = ""
        for col in range(grid_size):
            state = row * grid_size + col
            if state == goal:
                line += "  G "
            elif state in holes:
                line += "  H "
            else:
                line += f"  {arrows[Q[state].argmax()]} "
        print(f"    {line}")


# === Exercise 2: DQN on CartPole ===
# Problem: Train DQN on a simple balancing task.

def exercise_2():
    """DQN on simplified CartPole-like environment."""
    torch.manual_seed(42)
    np.random.seed(42)

    class SimpleCartPole:
        """Simplified CartPole: balance a pole by choosing left/right."""
        def __init__(self):
            self.reset()

        def reset(self):
            self.state = np.random.uniform(-0.05, 0.05, size=4).astype(np.float32)
            self.steps = 0
            return self.state.copy()

        def step(self, action):
            x, x_dot, theta, theta_dot = self.state
            force = 1.0 if action == 1 else -1.0
            # Simplified physics
            theta_dot += (force * 0.1 - 0.5 * np.sin(theta)) * 0.02
            theta += theta_dot * 0.02
            x_dot += force * 0.01
            x += x_dot * 0.02
            self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
            self.steps += 1
            done = abs(theta) > 0.2 or abs(x) > 2.4 or self.steps >= 500
            reward = 1.0 if not done else 0.0
            return self.state.copy(), reward, done

    class DQN(nn.Module):
        def __init__(self, state_dim=4, action_dim=2, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )

        def forward(self, x):
            return self.net(x)

    env = SimpleCartPole()
    q_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)

    replay_buffer = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    target_update = 10

    episode_rewards = []

    for episode in range(300):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.tensor(state))
                    action = q_vals.argmax().item()

            next_state, reward, done = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states = torch.tensor(np.array([b[0] for b in batch]))
                actions = torch.tensor([b[1] for b in batch]).unsqueeze(1)
                rewards = torch.tensor([b[2] for b in batch])
                next_states = torch.tensor(np.array([b[3] for b in batch]))
                dones = torch.tensor([b[4] for b in batch]).float()

                current_q = q_net(states).gather(1, actions).squeeze()
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + gamma * next_q * (1 - dones)

                loss = F.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if (episode + 1) % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

    # Moving average
    window = 50
    moving_avg = [np.mean(episode_rewards[max(0, i - window):i + 1])
                  for i in range(len(episode_rewards))]

    print(f"  DQN training on CartPole:")
    print(f"  Episode 50  avg reward: {moving_avg[49]:.1f}")
    print(f"  Episode 150 avg reward: {moving_avg[149]:.1f}")
    print(f"  Episode 300 avg reward: {moving_avg[-1]:.1f}")


# === Exercise 3: Experience Replay Ablation ===
# Problem: Compare training with and without replay buffer.

def exercise_3():
    """Effect of experience replay on training stability."""
    torch.manual_seed(42)

    # Simple environment: state is 1D, optimal action depends on state sign
    class SimpleEnv:
        def __init__(self):
            self.reset()

        def reset(self):
            self.state = np.random.uniform(-1, 1, size=2).astype(np.float32)
            self.steps = 0
            return self.state.copy()

        def step(self, action):
            # Optimal: action=1 if state[0]>0, else action=0
            correct = 1 if self.state[0] > 0 else 0
            reward = 1.0 if action == correct else -1.0
            self.state = np.random.uniform(-1, 1, size=2).astype(np.float32)
            self.steps += 1
            done = self.steps >= 50
            return self.state.copy(), reward, done

    class MiniDQN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2)
            )

        def forward(self, x):
            return self.net(x)

    for mode in ["With Replay", "Without Replay"]:
        torch.manual_seed(42)
        env = SimpleEnv()
        model = MiniDQN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        buffer = deque(maxlen=5000) if mode == "With Replay" else deque(maxlen=1)

        total_rewards = []
        for episode in range(200):
            state = env.reset()
            ep_reward = 0
            for _ in range(50):
                if random.random() < 0.1:
                    action = random.randint(0, 1)
                else:
                    with torch.no_grad():
                        action = model(torch.tensor(state)).argmax().item()

                next_state, reward, done = env.step(action)
                buffer.append((state, action, reward, next_state, done))
                state = next_state
                ep_reward += reward

                if len(buffer) >= 32:
                    batch = random.sample(buffer, min(32, len(buffer)))
                    states = torch.tensor(np.array([b[0] for b in batch]))
                    actions = torch.tensor([b[1] for b in batch]).unsqueeze(1)
                    rewards = torch.tensor([b[2] for b in batch])
                    next_s = torch.tensor(np.array([b[3] for b in batch]))
                    dones = torch.tensor([b[4] for b in batch]).float()

                    q = model(states).gather(1, actions).squeeze()
                    with torch.no_grad():
                        next_q = model(next_s).max(1)[0]
                    target = rewards + 0.99 * next_q * (1 - dones)
                    loss = F.mse_loss(q, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if done:
                    break

            total_rewards.append(ep_reward)

        last_50_avg = np.mean(total_rewards[-50:])
        reward_std = np.std(total_rewards[-50:])
        print(f"  {mode}: avg_reward={last_50_avg:.1f}, std={reward_std:.1f}")

    print("\n  Without replay, consecutive transitions are correlated,")
    print("  violating i.i.d. assumption and causing unstable training.")


# === Exercise 4: Double DQN ===
# Problem: Implement Double DQN to reduce Q-value overestimation.

def exercise_4():
    """Double DQN reduces Q-value overestimation."""
    torch.manual_seed(42)

    state_dim = 4
    action_dim = 2

    class DQN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 32), nn.ReLU(),
                nn.Linear(32, 32), nn.ReLU(),
                nn.Linear(32, action_dim),
            )

        def forward(self, x):
            return self.net(x)

    # Compare standard DQN vs Double DQN target computation
    q_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(q_net.state_dict())

    # Add slight differences to simulate training
    with torch.no_grad():
        for p in q_net.parameters():
            p += torch.randn_like(p) * 0.1

    states = torch.randn(100, state_dim)
    next_states = torch.randn(100, state_dim)

    with torch.no_grad():
        # Standard DQN: use target_net for both selection and evaluation
        standard_next_q = target_net(next_states).max(1)[0]

        # Double DQN: use q_net to select, target_net to evaluate
        best_actions = q_net(next_states).argmax(1, keepdim=True)
        double_next_q = target_net(next_states).gather(1, best_actions).squeeze(1)

    print(f"  Standard DQN avg next Q: {standard_next_q.mean().item():.4f}")
    print(f"  Double DQN avg next Q:   {double_next_q.mean().item():.4f}")
    print(f"  Standard DQN max next Q: {standard_next_q.max().item():.4f}")
    print(f"  Double DQN max next Q:   {double_next_q.max().item():.4f}")
    print("\n  Double DQN typically produces lower (more accurate) Q estimates")
    print("  by decoupling action selection (q_net) from evaluation (target_net).")


# === Exercise 5: REINFORCE vs DQN ===
# Problem: Implement REINFORCE policy gradient and compare with DQN.

def exercise_5():
    """REINFORCE policy gradient on a simple environment."""
    torch.manual_seed(42)

    class SimpleEnv:
        def __init__(self):
            self.reset()

        def reset(self):
            self.state = np.random.uniform(-1, 1, size=4).astype(np.float32)
            self.steps = 0
            return self.state.copy()

        def step(self, action):
            correct = 1 if self.state[0] > 0 else 0
            reward = 1.0 if action == correct else -0.5
            self.state = np.random.uniform(-1, 1, size=4).astype(np.float32)
            self.steps += 1
            done = self.steps >= 100
            return self.state.copy(), reward, done

    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim=4, action_dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 32), nn.ReLU(),
                nn.Linear(32, action_dim),
            )

        def forward(self, x):
            return F.softmax(self.net(x), dim=-1)

    env = SimpleEnv()
    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    gamma = 0.99

    episode_rewards = []

    for episode in range(500):
        state = env.reset()
        states, actions, rewards = [], [], []

        done = False
        while not done:
            state_t = torch.tensor(state)
            probs = policy(state_t)
            action = torch.multinomial(probs, 1).item()

            next_state, reward, done = env.step(action)
            states.append(state_t)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        # Normalize returns (baseline)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient
        loss = 0
        for state_t, action, G in zip(states, actions, returns):
            probs = policy(state_t)
            log_prob = torch.log(probs[action])
            loss -= log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(sum(rewards))

    window = 50
    moving_avg = [np.mean(episode_rewards[max(0, i - window):i + 1])
                  for i in range(len(episode_rewards))]

    print(f"  REINFORCE training:")
    print(f"  Episode 50  avg reward: {moving_avg[49]:.1f}")
    print(f"  Episode 250 avg reward: {moving_avg[249]:.1f}")
    print(f"  Episode 500 avg reward: {moving_avg[-1]:.1f}")
    print(f"  Reward variance (last 50): {np.std(episode_rewards[-50:]):.2f}")

    print("\n  REINFORCE has higher variance than DQN because:")
    print("  1. Full episode must complete before any learning")
    print("  2. Policy gradient estimates are inherently noisy")
    print("  Return normalization (baseline) reduces but doesn't eliminate variance.")


if __name__ == "__main__":
    print("=== Exercise 1: Q-Learning on Grid World ===")
    exercise_1()
    print("\n=== Exercise 2: DQN on CartPole ===")
    exercise_2()
    print("\n=== Exercise 3: Experience Replay Ablation ===")
    exercise_3()
    print("\n=== Exercise 4: Double DQN ===")
    exercise_4()
    print("\n=== Exercise 5: REINFORCE vs DQN ===")
    exercise_5()
    print("\nAll exercises completed!")
