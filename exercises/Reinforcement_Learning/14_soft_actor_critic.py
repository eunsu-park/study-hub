"""
Exercises for Lesson 14: Soft Actor-Critic (SAC)
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# Shared: Simple Pendulum Environment (no gym dependency)
# ============================================================
class SimplePendulum:
    """
    Pendulum-like continuous control environment.

    State: [cos(theta), sin(theta), theta_dot]  (3-dim)
    Action: torque in [-2, 2]  (1-dim, continuous)
    Reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
    Goal: swing up and balance at theta=0 (upright)

    This replicates the dynamics of Pendulum-v1 without requiring
    gymnasium.
    """

    def __init__(self):
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.state_dim = 3
        self.action_dim = 1
        self.theta = 0.0
        self.theta_dot = 0.0
        self.steps = 0
        self.max_steps = 200

    def reset(self):
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.theta_dot = np.random.uniform(-1.0, 1.0)
        self.steps = 0
        return self._get_obs()

    def step(self, action):
        # Clip action to valid range
        u = np.clip(action, -self.max_torque, self.max_torque)
        if isinstance(u, np.ndarray):
            u = u[0]

        # Pendulum dynamics
        theta_ddot = (-3.0 * self.g / (2.0 * self.l) *
                      np.sin(self.theta + np.pi) +
                      3.0 / (self.m * self.l ** 2) * u)

        self.theta_dot = np.clip(
            self.theta_dot + theta_ddot * self.dt,
            -self.max_speed, self.max_speed
        )
        self.theta += self.theta_dot * self.dt
        self.theta = ((self.theta + np.pi) % (2 * np.pi)) - np.pi

        # Reward: penalize angle, velocity, and torque
        reward = -(self.theta ** 2 + 0.1 * self.theta_dot ** 2 +
                   0.001 * u ** 2)

        self.steps += 1
        done = self.steps >= self.max_steps
        return self._get_obs(), reward, done

    def _get_obs(self):
        return np.array([np.cos(self.theta), np.sin(self.theta),
                         self.theta_dot], dtype=np.float32)


# ============================================================
# Shared: Replay Buffer
# ============================================================
class ReplayBuffer:
    """Circular replay buffer for off-policy learning."""

    def __init__(self, state_dim, action_dim, capacity=100000):
        self.capacity = capacity
        self.idx = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self.size


# ============================================================
# Shared: Simple MLP (numpy only)
# ============================================================
class SimpleMLP:
    """
    2-layer MLP with ReLU activation, implemented in pure numpy.
    Supports forward pass and gradient-based parameter updates.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, lr=3e-4):
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.W3 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * scale2
        self.b3 = np.zeros(output_dim, dtype=np.float32)
        self.lr = lr
        # Cache for backprop
        self._x = None
        self._z1 = None
        self._a1 = None
        self._z2 = None
        self._a2 = None

    def forward(self, x):
        self._x = x
        self._z1 = x @ self.W1 + self.b1
        self._a1 = np.maximum(0, self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = np.maximum(0, self._z2)
        out = self._a2 @ self.W3 + self.b3
        return out

    def backward(self, dout):
        """Backprop given gradient of loss w.r.t. output."""
        batch_size = dout.shape[0]

        dW3 = self._a2.T @ dout / batch_size
        db3 = dout.mean(axis=0)

        da2 = dout @ self.W3.T
        da2 = da2 * (self._z2 > 0)
        dW2 = self._a1.T @ da2 / batch_size
        db2 = da2.mean(axis=0)

        da1 = da2 @ self.W2.T
        da1 = da1 * (self._z1 > 0)
        dW1 = self._x.T @ da1 / batch_size
        db1 = da1.mean(axis=0)

        # Gradient clipping
        max_norm = 1.0
        for grad in [dW1, db1, dW2, db2, dW3, db3]:
            np.clip(grad, -max_norm, max_norm, out=grad)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    def copy_from(self, other):
        """Copy parameters from another MLP."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()

    def soft_update(self, source, tau=0.005):
        """Polyak averaging: theta_target <- tau*theta + (1-tau)*theta_target."""
        self.W1 = tau * source.W1 + (1 - tau) * self.W1
        self.b1 = tau * source.b1 + (1 - tau) * self.b1
        self.W2 = tau * source.W2 + (1 - tau) * self.W2
        self.b2 = tau * source.b2 + (1 - tau) * self.b2
        self.W3 = tau * source.W3 + (1 - tau) * self.W3
        self.b3 = tau * source.b3 + (1 - tau) * self.b3


# ============================================================
# Shared: SAC Agent (numpy implementation)
# ============================================================
class SACAgent:
    """
    Soft Actor-Critic agent implemented in pure numpy.

    Components:
    - Actor: outputs mean and log_std of a Gaussian; action = tanh(sample)
    - Twin Critics: Q1(s,a), Q2(s,a) to prevent overestimation
    - Target critics: slowly updated copies for stable Bellman targets
    - Alpha: temperature parameter (optionally auto-tuned)
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 auto_alpha=True, target_entropy=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha

        # Actor: outputs [mean, log_std] (2 * action_dim outputs)
        self.actor = SimpleMLP(state_dim, hidden_dim, 2 * action_dim, lr=lr)

        # Twin critics: Q(s, a) -> scalar
        self.q1 = SimpleMLP(state_dim + action_dim, hidden_dim, 1, lr=lr)
        self.q2 = SimpleMLP(state_dim + action_dim, hidden_dim, 1, lr=lr)

        # Target critics
        self.q1_target = SimpleMLP(state_dim + action_dim, hidden_dim, 1, lr=lr)
        self.q2_target = SimpleMLP(state_dim + action_dim, hidden_dim, 1, lr=lr)
        self.q1_target.copy_from(self.q1)
        self.q2_target.copy_from(self.q2)

        # Temperature
        if auto_alpha:
            self.target_entropy = target_entropy if target_entropy is not None else -action_dim
            self.log_alpha = 0.0  # log(alpha), optimized directly
            self.alpha = np.exp(self.log_alpha)
            self.alpha_lr = lr
        else:
            self.alpha = alpha

    def _actor_sample(self, states):
        """
        Sample actions from the squashed Gaussian policy.
        Returns actions in [-1, 1] and their log probabilities.
        """
        out = self.actor.forward(states)
        mean = out[:, :self.action_dim]
        log_std = out[:, self.action_dim:]
        # Clamp log_std for numerical stability
        log_std = np.clip(log_std, -20, 2)
        std = np.exp(log_std)

        # Reparameterization: z = mean + std * epsilon
        epsilon = np.random.randn(*mean.shape).astype(np.float32)
        z = mean + std * epsilon

        # Squash through tanh
        action = np.tanh(z)

        # Log probability with tanh correction
        # log_prob = log N(z; mean, std) - log(1 - tanh(z)^2)
        log_prob = -0.5 * ((z - mean) / (std + 1e-8)) ** 2 - np.log(std + 1e-8) - 0.5 * np.log(2 * np.pi)
        log_prob -= np.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(axis=-1, keepdims=True)

        return action, log_prob, mean, std, epsilon

    def select_action(self, state, deterministic=False):
        """Select action for a single state."""
        state = state.reshape(1, -1)
        out = self.actor.forward(state)
        mean = out[0, :self.action_dim]
        log_std = np.clip(out[0, self.action_dim:], -20, 2)

        if deterministic:
            return np.tanh(mean)
        else:
            std = np.exp(log_std)
            z = mean + std * np.random.randn(self.action_dim).astype(np.float32)
            return np.tanh(z)

    def update(self, batch, use_twin=True):
        """
        Perform one gradient step on all components.

        Args:
            batch: tuple of (states, actions, rewards, next_states, dones)
            use_twin: if True, use min(Q1, Q2); if False, use Q1 only
        """
        states, actions, rewards, next_states, dones = batch
        batch_size = states.shape[0]

        # --- Compute target Q value ---
        next_actions, next_log_probs, _, _, _ = self._actor_sample(next_states)
        sa_next = np.concatenate([next_states, next_actions], axis=1)

        q1_target_val = self.q1_target.forward(sa_next)
        if use_twin:
            q2_target_val = self.q2_target.forward(sa_next)
            q_target = np.minimum(q1_target_val, q2_target_val)
        else:
            q_target = q1_target_val

        # Soft Bellman target: r + gamma * (1 - done) * (Q_target - alpha * log_pi)
        target = rewards.reshape(-1, 1) + self.gamma * (1.0 - dones.reshape(-1, 1)) * (
            q_target - self.alpha * next_log_probs
        )

        # --- Update Critics ---
        sa = np.concatenate([states, actions], axis=1)

        # Q1 update
        q1_pred = self.q1.forward(sa)
        q1_error = q1_pred - target
        self.q1.backward(2.0 * q1_error)

        # Q2 update (if using twin critics)
        if use_twin:
            q2_pred = self.q2.forward(sa)
            q2_error = q2_pred - target
            self.q2.backward(2.0 * q2_error)

        # --- Update Actor ---
        # Sample new actions from current policy
        new_actions, log_probs, mean, std, epsilon = self._actor_sample(states)
        sa_new = np.concatenate([states, new_actions], axis=1)

        q1_new = self.q1.forward(sa_new)
        if use_twin:
            q2_new = self.q2.forward(sa_new)
            q_new = np.minimum(q1_new, q2_new)
        else:
            q_new = q1_new

        # Actor loss: E[alpha * log_pi - Q]
        # We approximate the gradient through the reparameterization trick.
        # For the numpy implementation, we use a simple policy gradient approach:
        # dL/d_actor_params approx= (alpha * log_prob - Q) * d_log_prob/d_params
        # This is simplified; in PyTorch, autograd handles this exactly.
        actor_loss_signal = self.alpha * log_probs - q_new
        # Gradient through the actor output (simplified)
        dout = np.zeros_like(self.actor.forward(states))
        # Push mean toward actions that maximize Q - alpha*log_pi
        # Using the chain rule through tanh and the Gaussian
        dtanh = 1 - new_actions ** 2 + 1e-6
        # d(actor_loss)/d(mean) ~ -sign(Q_new) * dtanh * sign(mean direction)
        # Simplified gradient: use actor_loss_signal to modulate
        d_mean = actor_loss_signal * dtanh * epsilon / (std + 1e-8)
        d_log_std = actor_loss_signal * (epsilon ** 2 - 1.0)
        dout[:, :self.action_dim] = d_mean
        dout[:, self.action_dim:] = d_log_std * 0.1  # Scale down std gradient
        self.actor.backward(dout)

        # --- Update Temperature (alpha) ---
        entropy = -log_probs.mean()
        if self.auto_alpha:
            # L(alpha) = -alpha * (log_pi + H_target)
            # d/d(log_alpha) = -(log_pi + H_target)
            alpha_grad = -(log_probs.mean() + self.target_entropy)
            self.log_alpha -= self.alpha_lr * alpha_grad
            self.log_alpha = np.clip(self.log_alpha, -5, 2)
            self.alpha = np.exp(self.log_alpha)

        # --- Soft update target networks ---
        self.q1_target.soft_update(self.q1, self.tau)
        if use_twin:
            self.q2_target.soft_update(self.q2, self.tau)

        critic_loss = float(np.mean(q1_error ** 2))
        actor_loss_val = float(actor_loss_signal.mean())
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss_val,
            'alpha': self.alpha,
            'entropy': float(entropy),
        }


def exercise_1():
    """
    SAC on Pendulum: Train SAC on a pendulum-like environment and
    plot the learning curve.

    Expected: The agent should learn to swing up and balance the pendulum,
    achieving reward around -200 to -300 within ~200 episodes.
    """
    print("Exercise 1: SAC on Pendulum")
    print("=" * 60)

    env = SimplePendulum()
    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        auto_alpha=True,
    )
    buffer = ReplayBuffer(env.state_dim, env.action_dim, capacity=50000)

    n_episodes = 200
    batch_size = 64
    start_steps = 1000  # Random exploration before training
    episode_rewards = []
    total_steps = 0

    print("\nTraining SAC on SimplePendulum...")
    print(f"{'Episode':>8} | {'Reward':>10} | {'Avg(10)':>10} | "
          f"{'Alpha':>8} | {'Entropy':>10}")
    print("-" * 60)

    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        metrics = {}

        while not done:
            # Random exploration for initial data collection
            if total_steps < start_steps:
                action = np.random.uniform(-2, 2, size=env.action_dim).astype(np.float32)
            else:
                action = agent.select_action(state) * env.max_torque

            next_state, reward, done = env.step(action)
            # Store normalized action in buffer
            buffer.add(state, action / env.max_torque, reward, next_state, done)
            state = next_state
            ep_reward += reward
            total_steps += 1

            # Train after initial exploration phase
            if total_steps >= start_steps and len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                metrics = agent.update(batch)

        episode_rewards.append(ep_reward)

        # Print progress
        if (ep + 1) % 20 == 0 or ep == 0:
            avg_10 = np.mean(episode_rewards[-10:])
            alpha = metrics.get('alpha', agent.alpha)
            entropy = metrics.get('entropy', 0)
            print(f"{ep+1:>8} | {ep_reward:>10.1f} | {avg_10:>10.1f} | "
                  f"{alpha:>8.4f} | {entropy:>10.3f}")

    # Summary statistics
    print()
    print("Learning Curve Summary:")
    windows = [10, 50]
    for w in windows:
        if len(episode_rewards) >= w:
            first_w = np.mean(episode_rewards[:w])
            last_w = np.mean(episode_rewards[-w:])
            print(f"  First {w} episodes avg: {first_w:.1f}")
            print(f"  Last  {w} episodes avg: {last_w:.1f}")
            print(f"  Improvement: {last_w - first_w:.1f}")
            print()

    print("Key observations:")
    print("  1. Initial episodes show poor reward (random exploration).")
    print("  2. After start_steps, the agent begins learning and reward improves.")
    print("  3. Alpha (temperature) typically decreases as the policy becomes")
    print("     more confident, shifting from exploration to exploitation.")
    print("  4. The entropy tracks the randomness of the policy — it should")
    print("     decrease as the agent converges to a good strategy.")

    return episode_rewards


def exercise_2():
    """
    Compare SAC vs PPO-like on Pendulum: Demonstrate sample efficiency
    difference between off-policy (SAC) and on-policy (REINFORCE baseline).

    Since we're working without gym, we compare:
    - SAC (off-policy, replay buffer, entropy regularization)
    - REINFORCE with baseline (on-policy, no replay buffer)

    Expected: SAC learns faster per environment step (sample efficient)
    because it reuses past experience from the replay buffer.
    """
    print("\nExercise 2: SAC vs On-Policy (REINFORCE) Comparison")
    print("=" * 60)

    # --- REINFORCE with baseline (on-policy comparison) ---
    class REINFORCEAgent:
        """Simple REINFORCE with learned value baseline."""

        def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3):
            self.policy = SimpleMLP(state_dim, hidden_dim, 2 * action_dim, lr=lr)
            self.value = SimpleMLP(state_dim, hidden_dim, 1, lr=lr)
            self.action_dim = action_dim
            self.gamma = 0.99

        def select_action(self, state):
            state = state.reshape(1, -1)
            out = self.policy.forward(state)
            mean = out[0, :self.action_dim]
            log_std = np.clip(out[0, self.action_dim:], -2, 1)
            std = np.exp(log_std)
            z = mean + std * np.random.randn(self.action_dim).astype(np.float32)
            return np.tanh(z) * 2.0  # Scale to [-2, 2]

        def update_from_episode(self, states, actions, rewards):
            """Update policy and value from a complete episode."""
            T = len(rewards)
            if T == 0:
                return

            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)

            # Compute discounted returns
            returns = np.zeros(T, dtype=np.float32)
            G = 0
            for t in reversed(range(T)):
                G = rewards[t] + self.gamma * G
                returns[t] = G

            # Normalize returns
            if np.std(returns) > 0:
                returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

            # Value baseline
            values = self.value.forward(states).flatten()
            advantages = returns - values

            # Update value network
            v_error = (values - returns).reshape(-1, 1)
            self.value.backward(2.0 * v_error)

            # Update policy (simplified REINFORCE gradient)
            out = self.policy.forward(states)
            mean = out[:, :self.action_dim]
            log_std = np.clip(out[:, self.action_dim:], -2, 1)
            std = np.exp(log_std)
            normalized_actions = actions / 2.0  # Map [-2,2] to [-1,1]
            z = np.arctanh(np.clip(normalized_actions, -0.999, 0.999))
            # Gradient direction: push mean toward actions with positive advantage
            d_mean = -advantages.reshape(-1, 1) * (z - mean) / (std ** 2 + 1e-8)
            d_log_std = -advantages.reshape(-1, 1) * ((z - mean) ** 2 / (std ** 2 + 1e-8) - 1)
            dout = np.zeros_like(out)
            dout[:, :self.action_dim] = d_mean
            dout[:, self.action_dim:] = d_log_std * 0.1
            self.policy.backward(dout)

    # --- Run both algorithms ---
    n_episodes = 200
    n_runs = 3

    sac_rewards_all = np.zeros((n_runs, n_episodes))
    reinforce_rewards_all = np.zeros((n_runs, n_episodes))

    for run in range(n_runs):
        np.random.seed(run * 42)

        # SAC
        env_sac = SimplePendulum()
        sac_agent = SACAgent(
            state_dim=3, action_dim=1, hidden_dim=64,
            lr=3e-4, gamma=0.99, auto_alpha=True,
        )
        sac_buffer = ReplayBuffer(3, 1, capacity=50000)
        total_steps = 0

        for ep in range(n_episodes):
            state = env_sac.reset()
            ep_reward = 0
            done = False
            while not done:
                if total_steps < 500:
                    action = np.random.uniform(-2, 2, size=1).astype(np.float32)
                else:
                    action = sac_agent.select_action(state) * 2.0
                next_state, reward, done = env_sac.step(action)
                sac_buffer.add(state, action / 2.0, reward, next_state, done)
                state = next_state
                ep_reward += reward
                total_steps += 1
                if total_steps >= 500 and len(sac_buffer) >= 64:
                    batch = sac_buffer.sample(64)
                    sac_agent.update(batch)
            sac_rewards_all[run, ep] = ep_reward

        # REINFORCE
        env_rf = SimplePendulum()
        rf_agent = REINFORCEAgent(state_dim=3, action_dim=1, hidden_dim=64, lr=1e-3)

        for ep in range(n_episodes):
            state = env_rf.reset()
            ep_states, ep_actions, ep_rewards = [], [], []
            done = False
            while not done:
                action = rf_agent.select_action(state)
                next_state, reward, done = env_rf.step(action)
                ep_states.append(state)
                ep_actions.append(action)
                ep_rewards.append(reward)
                state = next_state
            rf_agent.update_from_episode(ep_states, ep_actions, ep_rewards)
            reinforce_rewards_all[run, ep] = sum(ep_rewards)

    # Average across runs
    sac_mean = sac_rewards_all.mean(axis=0)
    reinforce_mean = reinforce_rewards_all.mean(axis=0)

    # Display comparison
    print()
    print(f"Average episode reward ({n_runs} runs):")
    print(f"{'Episode':>8} | {'SAC':>12} | {'REINFORCE':>12} | {'Difference':>12}")
    print("-" * 52)

    window = 10
    sac_smooth = np.convolve(sac_mean, np.ones(window) / window, mode='valid')
    rf_smooth = np.convolve(reinforce_mean, np.ones(window) / window, mode='valid')

    for idx in [0, 19, 49, 99, 149, 189]:
        if idx < len(sac_smooth):
            diff = sac_smooth[idx] - rf_smooth[idx]
            print(f"{idx + window:>8} | {sac_smooth[idx]:>12.1f} | "
                  f"{rf_smooth[idx]:>12.1f} | {diff:>12.1f}")

    # Summary
    print()
    print(f"Final 50 episodes average:")
    sac_final = sac_mean[-50:].mean()
    rf_final = reinforce_mean[-50:].mean()
    print(f"  SAC:       {sac_final:.1f}")
    print(f"  REINFORCE: {rf_final:.1f}")

    print()
    print("Key observations:")
    print("  1. SAMPLE EFFICIENCY: SAC typically learns faster per episode")
    print("     because it reuses transitions from the replay buffer.")
    print("     REINFORCE discards each episode's data after one update.")
    print("  2. STABILITY: SAC's twin critics and entropy regularization")
    print("     provide more stable learning than vanilla REINFORCE.")
    print("  3. ENTROPY BONUS: SAC maintains exploration throughout training")
    print("     via the entropy term, preventing premature convergence.")
    print("  4. OFF-POLICY ADVANTAGE: SAC can perform many gradient updates")
    print("     per environment step; REINFORCE is limited to one update")
    print("     per episode (on-policy constraint).")


def exercise_3():
    """
    Ablation Study: Compare SAC variants to isolate the contribution
    of each component.

    Variants:
    1. Fixed alpha = 0.2 (no auto-tuning)
    2. Auto alpha (default SAC)
    3. No entropy term (alpha = 0, TD3-like)
    4. Single Q-network (no twin critics, prone to overestimation)
    """
    print("\nExercise 3: SAC Ablation Study")
    print("=" * 60)

    configs = {
        "Fixed alpha=0.2": {"auto_alpha": False, "alpha": 0.2, "use_twin": True},
        "Auto alpha": {"auto_alpha": True, "alpha": 0.2, "use_twin": True},
        "No entropy (a=0)": {"auto_alpha": False, "alpha": 0.0, "use_twin": True},
        "Single critic": {"auto_alpha": True, "alpha": 0.2, "use_twin": False},
    }

    n_episodes = 150
    n_runs = 3
    batch_size = 64
    start_steps = 500

    all_results = {}

    for config_name, config in configs.items():
        print(f"\n  Running: {config_name}...")
        run_rewards = np.zeros((n_runs, n_episodes))

        for run in range(n_runs):
            np.random.seed(run * 123)
            env = SimplePendulum()

            agent = SACAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=64,
                lr=3e-4,
                gamma=0.99,
                tau=0.005,
                auto_alpha=config["auto_alpha"],
                alpha=config["alpha"],
            )
            buffer = ReplayBuffer(env.state_dim, env.action_dim, capacity=50000)
            total_steps = 0

            for ep in range(n_episodes):
                state = env.reset()
                ep_reward = 0
                done = False

                while not done:
                    if total_steps < start_steps:
                        action = np.random.uniform(-2, 2, size=1).astype(np.float32)
                    else:
                        action = agent.select_action(state) * env.max_torque

                    next_state, reward, done = env.step(action)
                    buffer.add(state, action / env.max_torque, reward, next_state, done)
                    state = next_state
                    ep_reward += reward
                    total_steps += 1

                    if total_steps >= start_steps and len(buffer) >= batch_size:
                        batch = buffer.sample(batch_size)
                        agent.update(batch, use_twin=config["use_twin"])

                run_rewards[run, ep] = ep_reward

        all_results[config_name] = run_rewards.mean(axis=0)

    # Display comparison
    print()
    print(f"Ablation Results (avg over {n_runs} runs):")
    print(f"{'Config':>20} | {'First 50':>10} | {'Last 50':>10} | {'Improvement':>12}")
    print("-" * 60)

    for name, rewards in all_results.items():
        first_50 = rewards[:50].mean()
        last_50 = rewards[-50:].mean()
        improvement = last_50 - first_50
        print(f"{name:>20} | {first_50:>10.1f} | {last_50:>10.1f} | {improvement:>12.1f}")

    # Detailed comparison at key episodes
    print()
    print("Average reward at key episodes (smoothed, window=10):")
    window = 10
    episodes_to_show = [10, 30, 50, 100, 150]
    header = f"{'Episode':>8} |"
    for name in configs:
        header += f" {name:>16} |"
    print(header)
    print("-" * (10 + 19 * len(configs)))

    for ep in episodes_to_show:
        if ep <= n_episodes:
            row = f"{ep:>8} |"
            for name in configs:
                smoothed = np.convolve(all_results[name],
                                       np.ones(window) / window, mode='valid')
                idx = min(ep - window, len(smoothed) - 1)
                if idx >= 0:
                    row += f" {smoothed[idx]:>16.1f} |"
                else:
                    row += f" {'N/A':>16} |"
            print(row)

    print()
    print("Expected findings and analysis:")
    print()
    print("  1. AUTO ALPHA vs FIXED ALPHA:")
    print("     - Auto alpha adapts the temperature to the task difficulty.")
    print("     - Fixed alpha=0.2 may be too high (over-exploration) or")
    print("       too low (under-exploration) for a given environment.")
    print("     - Auto-tuning typically outperforms or matches fixed alpha.")
    print()
    print("  2. WITH ENTROPY (alpha > 0) vs WITHOUT (alpha = 0):")
    print("     - The entropy bonus encourages continued exploration,")
    print("       preventing the policy from collapsing to a single")
    print("       deterministic action prematurely.")
    print("     - Without entropy, the agent may converge faster initially")
    print("       but get stuck in local optima more often.")
    print()
    print("  3. TWIN CRITICS vs SINGLE CRITIC:")
    print("     - Twin critics use min(Q1, Q2) in the Bellman target,")
    print("       preventing the systematic upward bias in Q-value estimates.")
    print("     - Single critic tends to overestimate Q-values, causing the")
    print("       actor to exploit incorrect high-value actions.")
    print("     - This overestimation can lead to instability or divergence.")


def exercise_4():
    """
    Custom Environment: Apply SAC to a 2D reaching task where an agent
    must move a point mass to a target position.

    Environment:
    - State: [x, y, target_x, target_y] (4-dim)
    - Action: [dx, dy] in [-1, 1] (2-dim continuous)
    - Reward: -distance to target
    - Done: when distance < 0.05 or max steps reached
    """
    print("\nExercise 4: SAC on Custom Reaching Environment")
    print("=" * 60)

    class ReachingEnv:
        """
        2D reaching task: move a point to a target position.
        The target is randomly placed at reset.
        """

        def __init__(self):
            self.state_dim = 4  # [x, y, target_x, target_y]
            self.action_dim = 2  # [dx, dy]
            self.max_steps = 100
            self.step_size = 0.1
            self.success_threshold = 0.05
            self.pos = np.zeros(2, dtype=np.float32)
            self.target = np.zeros(2, dtype=np.float32)
            self.steps = 0

        def reset(self):
            self.pos = np.random.uniform(-1, 1, size=2).astype(np.float32)
            self.target = np.random.uniform(-0.5, 0.5, size=2).astype(np.float32)
            self.steps = 0
            return self._get_obs()

        def step(self, action):
            action = np.clip(action, -1, 1)
            if action.ndim == 0:
                action = np.array([action])
            self.pos = np.clip(self.pos + action[:2] * self.step_size, -1, 1)
            self.steps += 1

            dist = np.linalg.norm(self.pos - self.target)
            reward = -dist  # Dense reward
            success = dist < self.success_threshold
            done = success or self.steps >= self.max_steps

            # Bonus for reaching target
            if success:
                reward += 10.0

            return self._get_obs(), reward, done

        def _get_obs(self):
            return np.concatenate([self.pos, self.target]).astype(np.float32)

    # --- Train SAC ---
    env = ReachingEnv()
    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        auto_alpha=True,
    )
    buffer = ReplayBuffer(env.state_dim, env.action_dim, capacity=50000)

    n_episodes = 300
    batch_size = 64
    start_steps = 500
    total_steps = 0

    episode_rewards = []
    success_count = 0
    success_history = []  # Track success rate over time

    print("\nTraining SAC on ReachingEnv...")
    print(f"{'Episode':>8} | {'Reward':>10} | {'Avg(20)':>10} | "
          f"{'Success%':>10} | {'Alpha':>8}")
    print("-" * 60)

    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        metrics = {}

        while not done:
            if total_steps < start_steps:
                action = np.random.uniform(-1, 1, size=env.action_dim).astype(np.float32)
            else:
                action = agent.select_action(state)

            next_state, reward, done = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            total_steps += 1

            if total_steps >= start_steps and len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                metrics = agent.update(batch)

        episode_rewards.append(ep_reward)

        # Check if episode was successful (reached target)
        dist = np.linalg.norm(env.pos - env.target)
        success = dist < env.success_threshold
        if success:
            success_count += 1
        success_history.append(success)

        if (ep + 1) % 30 == 0 or ep == 0:
            avg_20 = np.mean(episode_rewards[-20:])
            recent_success = np.mean(success_history[-30:]) * 100
            alpha = metrics.get('alpha', agent.alpha)
            print(f"{ep+1:>8} | {ep_reward:>10.1f} | {avg_20:>10.1f} | "
                  f"{recent_success:>9.1f}% | {alpha:>8.4f}")

    # Evaluate the trained policy deterministically
    print("\nEvaluating trained policy (deterministic, 50 episodes):")
    eval_rewards = []
    eval_successes = 0
    eval_steps_to_reach = []

    for _ in range(50):
        state = env.reset()
        ep_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done = env.step(action)
            state = next_state
            ep_reward += reward
            steps += 1

        eval_rewards.append(ep_reward)
        dist = np.linalg.norm(env.pos - env.target)
        if dist < env.success_threshold:
            eval_successes += 1
            eval_steps_to_reach.append(steps)

    print(f"  Average reward: {np.mean(eval_rewards):.1f} +/- {np.std(eval_rewards):.1f}")
    print(f"  Success rate: {eval_successes}/50 = {eval_successes * 2}%")
    if eval_steps_to_reach:
        print(f"  Avg steps to reach (when successful): "
              f"{np.mean(eval_steps_to_reach):.1f}")

    # Show a sample trajectory
    print("\nSample trajectory (deterministic policy):")
    state = env.reset()
    print(f"  Start:  pos=({env.pos[0]:.3f}, {env.pos[1]:.3f}), "
          f"target=({env.target[0]:.3f}, {env.target[1]:.3f})")
    trajectory = [env.pos.copy()]
    done = False
    steps = 0

    while not done and steps < 100:
        action = agent.select_action(state, deterministic=True)
        state, reward, done = env.step(action)
        trajectory.append(env.pos.copy())
        steps += 1

    final_dist = np.linalg.norm(env.pos - env.target)
    print(f"  End:    pos=({env.pos[0]:.3f}, {env.pos[1]:.3f}), "
          f"dist={final_dist:.4f}, steps={steps}")

    # ASCII visualization of trajectory
    print("\n  Trajectory (ASCII, 20x20 grid):")
    grid_size = 20
    grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]

    def pos_to_grid(p):
        """Map position [-1, 1] to grid coordinates [0, grid_size-1]."""
        gx = int(np.clip((p[0] + 1) / 2 * (grid_size - 1), 0, grid_size - 1))
        gy = int(np.clip((1 - (p[1] + 1) / 2) * (grid_size - 1), 0, grid_size - 1))
        return gy, gx

    # Draw trajectory
    for i, pos in enumerate(trajectory):
        r, c = pos_to_grid(pos)
        if grid[r][c] == '.':
            grid[r][c] = 'o'

    # Mark start and end
    sr, sc = pos_to_grid(trajectory[0])
    er, ec = pos_to_grid(trajectory[-1])
    tr, tc = pos_to_grid(env.target)
    grid[sr][sc] = 'S'
    grid[er][ec] = 'E'
    grid[tr][tc] = 'T'

    for row in grid:
        print("  " + " ".join(row))

    print()
    print("  Legend: S=start, T=target, E=end, o=path")
    print()
    print("Key observations:")
    print("  1. The reaching task has a clear optimal strategy: move directly")
    print("     toward the target. SAC discovers this through exploration.")
    print("  2. The dense reward signal (-distance) provides useful gradient")
    print("     information at every step, aiding learning.")
    print("  3. The success bonus (+10) helps the agent learn that reaching")
    print("     the target is especially valuable.")
    print("  4. With auto-alpha, SAC balances exploration (trying different")
    print("     approach angles) with exploitation (moving straight to target).")


if __name__ == "__main__":
    print("=== Exercise 1: SAC on Pendulum ===")
    exercise_1()

    print("\n=== Exercise 2: SAC vs On-Policy Comparison ===")
    exercise_2()

    print("\n=== Exercise 3: SAC Ablation Study ===")
    exercise_3()

    print("\n=== Exercise 4: SAC on Custom Reaching Environment ===")
    exercise_4()

    print("\nAll exercises completed!")
