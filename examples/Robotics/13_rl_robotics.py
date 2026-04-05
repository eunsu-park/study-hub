"""
Reinforcement Learning for Robotic Arm Reaching
=================================================
Train a simple policy to control a 2-link planar arm to reach target positions.

Reinforcement Learning (RL) offers a fundamentally different approach to robot
control: instead of deriving control laws from physics models, the robot learns
through trial and error. The key components are:

  - State: what the robot observes (joint angles, velocities, target position)
  - Action: what the robot does (joint torques)
  - Reward: scalar feedback signal (negative distance to target)
  - Policy: mapping from states to actions (what we want to learn)

We implement two RL approaches:
  1. REINFORCE (policy gradient): directly optimize a parameterized policy
     by following the gradient of expected cumulative reward
  2. Q-learning with discretized actions: learn a value function over
     state-action pairs using temporal difference updates

Both methods learn to reach arbitrary target positions using only reward
signals, with no explicit knowledge of kinematics or dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Environment: 2-link planar arm reaching task
# ---------------------------------------------------------------------------
class ReachingEnv:
    """Simple 2-link arm environment for RL experiments.

    The arm lives in 2D. Each joint has a limited angular range and torque
    limit. The goal is to move the end-effector to a target position.

    State (6D): [cos(q1), sin(q1), cos(q2), sin(q2), target_x, target_y]
        We use cos/sin encoding to avoid angle wraparound discontinuities,
        a standard trick in robot RL.

    Action (2D): [torque1, torque2], continuous and bounded

    Reward: -distance_to_target (per step) + bonus for reaching close
    """

    def __init__(self, L1=1.0, L2=0.8, max_torque=2.0, dt=0.05):
        self.L1, self.L2 = L1, L2
        self.max_torque = max_torque
        self.dt = dt
        self.max_steps = 100

        # State
        self.q = np.zeros(2)       # joint angles
        self.qd = np.zeros(2)      # joint velocities
        self.target = np.zeros(2)  # target (x, y)
        self.step_count = 0

    def forward_kinematics(self, q):
        """Compute end-effector position from joint angles."""
        x = self.L1 * np.cos(q[0]) + self.L2 * np.cos(q[0] + q[1])
        y = self.L1 * np.sin(q[0]) + self.L2 * np.sin(q[0] + q[1])
        return np.array([x, y])

    def _get_obs(self):
        """Build observation vector with cos/sin encoding."""
        return np.array([
            np.cos(self.q[0]), np.sin(self.q[0]),
            np.cos(self.q[1]), np.sin(self.q[1]),
            self.target[0], self.target[1]
        ])

    def reset(self, rng=None):
        """Reset arm to random configuration with random target.

        We sample targets within the reachable workspace to ensure the
        task is feasible. The workspace radius is L1 + L2.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Random initial joint angles (small range near zero)
        self.q = rng.uniform(-0.5, 0.5, size=2)
        self.qd = np.zeros(2)

        # Random target within reachable workspace
        max_reach = (self.L1 + self.L2) * 0.8  # 80% of max to stay feasible
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0.3, max_reach)
        self.target = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        """Apply torques and advance one time step.

        Uses simple double-integrator dynamics (ignoring gravity and coupling
        for simplicity — the RL agent must learn to handle this).
        """
        # Clip torques
        tau = np.clip(action, -self.max_torque, self.max_torque)

        # Simplified dynamics: M*qdd = tau - damping*qd
        # Using unit inertia and light damping for learning tractability
        damping = 0.1
        qdd = tau - damping * self.qd

        # Euler integration
        self.qd += qdd * self.dt
        self.qd = np.clip(self.qd, -5.0, 5.0)  # velocity limit
        self.q += self.qd * self.dt

        # Wrap angles to [-pi, pi]
        self.q = (self.q + np.pi) % (2 * np.pi) - np.pi

        self.step_count += 1

        # Compute reward
        ee_pos = self.forward_kinematics(self.q)
        dist = np.linalg.norm(ee_pos - self.target)

        reward = -dist  # primary reward: negative distance
        if dist < 0.1:
            reward += 5.0  # bonus for reaching close to target

        done = (self.step_count >= self.max_steps) or (dist < 0.05)

        return self._get_obs(), reward, done


# ---------------------------------------------------------------------------
# REINFORCE (policy gradient)
# ---------------------------------------------------------------------------
class GaussianPolicy:
    """Linear Gaussian policy for continuous actions.

    pi(a|s) = N(mu(s), sigma^2)  where  mu(s) = W @ s + b

    This is the simplest continuous policy: the mean is a linear function
    of the state, and the standard deviation is a learnable parameter.
    The policy gradient theorem tells us how to update W, b, and log_sigma
    to maximize expected return.
    """

    def __init__(self, state_dim, action_dim, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        scale = 0.1
        self.W = rng.normal(0, scale, size=(action_dim, state_dim))
        self.b = np.zeros(action_dim)
        self.log_sigma = np.zeros(action_dim)  # log std dev

    def forward(self, state):
        """Compute mean action."""
        return self.W @ state + self.b

    def sample(self, state, rng):
        """Sample action from Gaussian policy."""
        mu = self.forward(state)
        sigma = np.exp(self.log_sigma)
        action = mu + sigma * rng.normal(size=mu.shape)
        return action

    def log_prob(self, state, action):
        """Compute log probability of action given state.

        log pi(a|s) = -0.5 * sum((a - mu)^2 / sigma^2 + 2*log(sigma) + log(2*pi))
        """
        mu = self.forward(state)
        sigma = np.exp(self.log_sigma)
        return -0.5 * np.sum(
            ((action - mu) / sigma) ** 2 + 2 * self.log_sigma + np.log(2 * np.pi)
        )

    def log_prob_grad(self, state, action):
        """Compute gradients of log_prob w.r.t. policy parameters.

        These are the "score function" gradients used in REINFORCE:
          d/dW log pi = (a - mu) / sigma^2 * s^T
          d/db log pi = (a - mu) / sigma^2
          d/d(log_sigma) log pi = (a - mu)^2 / sigma^2 - 1
        """
        mu = self.forward(state)
        sigma = np.exp(self.log_sigma)
        diff = (action - mu) / (sigma ** 2)

        dW = np.outer(diff, state)
        db = diff
        dlog_sigma = ((action - mu) ** 2) / (sigma ** 2) - 1.0

        return dW, db, dlog_sigma


def reinforce_train(n_episodes=500, gamma=0.99, lr=0.001):
    """Train a reaching policy using REINFORCE.

    REINFORCE (Williams, 1992) is the simplest policy gradient method:
      1. Collect a full episode using the current policy
      2. Compute discounted returns G_t for each time step
      3. Update parameters: theta += lr * G_t * grad(log pi(a_t|s_t))

    We use a baseline (mean return) to reduce variance, which is critical
    for stable learning.
    """
    rng = np.random.default_rng(42)
    env = ReachingEnv()
    policy = GaussianPolicy(state_dim=6, action_dim=2, rng=rng)

    episode_rewards = []

    for ep in range(n_episodes):
        # Collect one episode
        states, actions, rewards = [], [], []
        obs = env.reset(rng=rng)

        done = False
        while not done:
            action = policy.sample(obs, rng)
            states.append(obs)
            actions.append(action)
            obs, reward, done = env.step(action)
            rewards.append(reward)

        # Compute discounted returns (reward-to-go)
        T = len(rewards)
        returns = np.zeros(T)
        G = 0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G

        # Baseline: subtract mean return to reduce variance
        returns -= np.mean(returns)
        std = np.std(returns)
        if std > 1e-8:
            returns /= std

        # Policy gradient update
        dW_total = np.zeros_like(policy.W)
        db_total = np.zeros_like(policy.b)
        dls_total = np.zeros_like(policy.log_sigma)

        for t in range(T):
            dW, db, dls = policy.log_prob_grad(states[t], actions[t])
            dW_total += returns[t] * dW
            db_total += returns[t] * db
            dls_total += returns[t] * dls

        # Gradient ascent (maximizing expected return)
        policy.W += lr * dW_total / T
        policy.b += lr * db_total / T
        policy.log_sigma += lr * dls_total / T

        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  Episode {ep+1:4d}: avg reward (last 50) = {avg:.2f}")

    return policy, episode_rewards


# ---------------------------------------------------------------------------
# Q-learning with discretized actions
# ---------------------------------------------------------------------------
class QLearningAgent:
    """Tabular Q-learning with tile coding for continuous states.

    Since our state space is continuous, we use tile coding to discretize it:
    multiple overlapping grids (tilings) cover the state space, and each
    state activates one tile per tiling. The Q-value is the sum of weights
    across active tiles.

    This provides generalization (nearby states share tiles) while keeping
    the simplicity of tabular methods.
    """

    def __init__(self, n_tilings=8, tiles_per_dim=4, n_actions=9, rng=None):
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.n_actions = n_actions

        # Discretized action set: 9 combinations of {-1, 0, +1} per joint
        torques = np.array([-1.0, 0.0, 1.0])
        self.action_set = np.array(np.meshgrid(torques, torques)).T.reshape(-1, 2)

        # Tile coding weight vector
        total_tiles = n_tilings * (tiles_per_dim ** 6) * n_actions
        # Use hash-based indexing to manage memory
        self.table_size = 100000
        self.weights = np.zeros(self.table_size)

        # Tiling offsets for generalization
        if rng is None:
            rng = np.random.default_rng(42)
        self.offsets = rng.uniform(0, 1.0 / tiles_per_dim, size=(n_tilings, 6))

    def _get_tiles(self, state, action_idx):
        """Compute tile indices for a state-action pair using hash-based coding."""
        # Normalize state to [0, 1] range
        s_min = np.array([-1, -1, -1, -1, -1.5, -1.5])
        s_max = np.array([1, 1, 1, 1, 1.5, 1.5])
        s_norm = (state - s_min) / (s_max - s_min + 1e-8)
        s_norm = np.clip(s_norm, 0, 0.999)

        tiles = []
        for tiling in range(self.n_tilings):
            shifted = s_norm + self.offsets[tiling]
            coords = (shifted * self.tiles_per_dim).astype(int)
            # Hash the coordinates with tiling and action index
            h = hash((tiling, action_idx, tuple(coords))) % self.table_size
            tiles.append(h)
        return tiles

    def q_value(self, state, action_idx):
        """Compute Q(s, a) as sum of weights at active tiles."""
        tiles = self._get_tiles(state, action_idx)
        return sum(self.weights[t] for t in tiles)

    def best_action(self, state):
        """Select the action with highest Q-value."""
        q_vals = [self.q_value(state, a) for a in range(self.n_actions)]
        return int(np.argmax(q_vals))

    def select_action(self, state, epsilon, rng):
        """Epsilon-greedy action selection."""
        if rng.random() < epsilon:
            return rng.integers(self.n_actions)
        return self.best_action(state)

    def update(self, state, action_idx, reward, next_state, done, alpha, gamma):
        """Q-learning update: Q(s,a) += alpha * (r + gamma*max_a Q(s',a') - Q(s,a))."""
        current_q = self.q_value(state, action_idx)

        if done:
            target = reward
        else:
            next_q = max(self.q_value(next_state, a) for a in range(self.n_actions))
            target = reward + gamma * next_q

        td_error = target - current_q
        tiles = self._get_tiles(state, action_idx)

        for t in tiles:
            self.weights[t] += alpha * td_error / self.n_tilings


def qlearning_train(n_episodes=500, alpha=0.05, gamma=0.99,
                    epsilon_start=1.0, epsilon_end=0.05):
    """Train a reaching policy using Q-learning with tile coding."""
    rng = np.random.default_rng(42)
    env = ReachingEnv()
    agent = QLearningAgent(rng=rng)

    episode_rewards = []

    for ep in range(n_episodes):
        obs = env.reset(rng=rng)
        ep_reward = 0
        done = False

        # Linear epsilon decay
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * ep / n_episodes

        while not done:
            action_idx = agent.select_action(obs, epsilon, rng)
            action = agent.action_set[action_idx]

            next_obs, reward, done = env.step(action)
            agent.update(obs, action_idx, reward, next_obs, done, alpha, gamma)

            obs = next_obs
            ep_reward += reward

        episode_rewards.append(ep_reward)

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  Episode {ep+1:4d}: avg reward (last 50) = {avg:.2f}, "
                  f"epsilon = {epsilon:.3f}")

    return agent, episode_rewards


# ---------------------------------------------------------------------------
# Evaluation and visualization
# ---------------------------------------------------------------------------
def evaluate_policy(policy_type, policy_or_agent, n_eval=20):
    """Evaluate a trained policy on random reaching tasks.

    Returns end-effector trajectories and final distances for visualization.
    """
    rng = np.random.default_rng(123)
    env = ReachingEnv()
    results = []

    for _ in range(n_eval):
        obs = env.reset(rng=rng)
        trajectory = [env.forward_kinematics(env.q).copy()]
        target = env.target.copy()

        done = False
        while not done:
            if policy_type == "reinforce":
                action = policy_or_agent.forward(obs)  # use mean (no noise)
            else:
                action_idx = policy_or_agent.best_action(obs)
                action = policy_or_agent.action_set[action_idx]

            obs, _, done = env.step(action)
            trajectory.append(env.forward_kinematics(env.q).copy())

        final_dist = np.linalg.norm(trajectory[-1] - target)
        results.append({
            "trajectory": np.array(trajectory),
            "target": target,
            "final_dist": final_dist
        })

    return results


def demo_rl_robotics():
    """Demonstrate reinforcement learning for robotic arm reaching."""
    print("=" * 60)
    print("Reinforcement Learning for Robotic Arm Reaching")
    print("=" * 60)

    # --- Train REINFORCE ---
    print("\n--- Training REINFORCE (Policy Gradient) ---")
    pg_policy, pg_rewards = reinforce_train(n_episodes=500)

    # --- Train Q-learning ---
    print("\n--- Training Q-Learning (Tile Coding) ---")
    ql_agent, ql_rewards = qlearning_train(n_episodes=500)

    # --- Evaluate both ---
    print("\n--- Evaluating trained policies ---")
    pg_results = evaluate_policy("reinforce", pg_policy)
    ql_results = evaluate_policy("qlearning", ql_agent)

    pg_dists = [r["final_dist"] for r in pg_results]
    ql_dists = [r["final_dist"] for r in ql_results]
    print(f"  REINFORCE  — mean final dist: {np.mean(pg_dists):.3f} "
          f"(+/- {np.std(pg_dists):.3f})")
    print(f"  Q-learning — mean final dist: {np.mean(ql_dists):.3f} "
          f"(+/- {np.std(ql_dists):.3f})")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (0,0) Learning curves
    window = 20
    pg_smooth = np.convolve(pg_rewards, np.ones(window)/window, mode='valid')
    ql_smooth = np.convolve(ql_rewards, np.ones(window)/window, mode='valid')
    axes[0, 0].plot(pg_smooth, color='#1f77b4', linewidth=1.5, label='REINFORCE')
    axes[0, 0].plot(ql_smooth, color='#ff7f0e', linewidth=1.5, label='Q-learning')
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Episode Reward (smoothed)")
    axes[0, 0].set_title("Learning Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1) REINFORCE reaching trajectories
    for r in pg_results[:8]:
        traj = r["trajectory"]
        axes[0, 1].plot(traj[:, 0], traj[:, 1], alpha=0.5, linewidth=1)
        axes[0, 1].plot(r["target"][0], r["target"][1], 'r*', markersize=10)
        axes[0, 1].plot(traj[-1, 0], traj[-1, 1], 'ko', markersize=4)
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].set_title("REINFORCE: End-Effector Trajectories")
    axes[0, 1].set_aspect('equal')
    axes[0, 1].grid(True, alpha=0.3)

    # (1,0) Q-learning reaching trajectories
    for r in ql_results[:8]:
        traj = r["trajectory"]
        axes[1, 0].plot(traj[:, 0], traj[:, 1], alpha=0.5, linewidth=1)
        axes[1, 0].plot(r["target"][0], r["target"][1], 'r*', markersize=10)
        axes[1, 0].plot(traj[-1, 0], traj[-1, 1], 'ko', markersize=4)
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    axes[1, 0].set_title("Q-Learning: End-Effector Trajectories")
    axes[1, 0].set_aspect('equal')
    axes[1, 0].grid(True, alpha=0.3)

    # (1,1) Final distance comparison
    axes[1, 1].boxplot([pg_dists, ql_dists], labels=['REINFORCE', 'Q-learning'])
    axes[1, 1].set_ylabel("Final Distance to Target")
    axes[1, 1].set_title("Reaching Accuracy Comparison")
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("RL for Robotic Arm Reaching: REINFORCE vs Q-Learning", fontsize=14)
    plt.tight_layout()
    plt.savefig("13_rl_robotics.png", dpi=120)
    plt.show()

    # --- Summary ---
    print("\n--- Summary ---")
    print("  REINFORCE (policy gradient):")
    print("    - Directly optimizes the policy (no value function)")
    print("    - Continuous actions, smooth policy")
    print("    - High variance in gradient estimates")
    print("  Q-learning (tile coding):")
    print("    - Learns state-action values, derives policy from argmax")
    print("    - Discretized actions (coarser control)")
    print("    - Lower variance but limited by discretization")


if __name__ == "__main__":
    demo_rl_robotics()
