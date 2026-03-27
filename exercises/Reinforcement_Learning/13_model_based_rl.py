"""
Exercises for Lesson 13: Model-Based Reinforcement Learning
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# Shared: Simple GridWorld Environment (no gym dependency)
# ============================================================
class GridWorld:
    """
    5x5 GridWorld with walls for Dyna-Q experiments.

    Layout:
      S . . . .
      . W . W .
      . W . W .
      . . . W .
      . . . . G

    S = start (0,0), G = goal (4,4), W = wall
    Actions: 0=up, 1=right, 2=down, 3=left
    Reward: +1 at goal, 0 otherwise
    """

    def __init__(self, size=5):
        self.size = size
        self.walls = {(1, 1), (1, 3), (2, 1), (2, 3), (3, 3)}
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.n_states = size * size
        self.n_actions = 4
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self._state_to_idx(self.state)

    def step(self, action):
        row, col = self.state
        if action == 0:    # up
            row = max(row - 1, 0)
        elif action == 1:  # right
            col = min(col + 1, self.size - 1)
        elif action == 2:  # down
            row = min(row + 1, self.size - 1)
        elif action == 3:  # left
            col = max(col - 1, 0)

        new_state = (row, col)
        # If hit a wall, stay in place
        if new_state in self.walls:
            new_state = self.state

        self.state = new_state
        done = (self.state == self.goal)
        reward = 1.0 if done else 0.0
        return self._state_to_idx(self.state), reward, done

    def _state_to_idx(self, state):
        return state[0] * self.size + state[1]

    def _idx_to_state(self, idx):
        return (idx // self.size, idx % self.size)


# ============================================================
# Shared: Dyna-Q Agent
# ============================================================
class DynaQ:
    """Tabular Dyna-Q agent with configurable planning steps."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=0.1, n_planning=5):
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}      # (s, a) -> (r, s', done)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning
        self.visited = []    # track visited (s, a) pairs

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, s, a, r, s_next, done):
        # Direct RL update
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

        # Model learning
        self.model[(s, a)] = (r, s_next, done)
        if (s, a) not in self.visited:
            self.visited.append((s, a))

        # Planning: n simulated updates from the learned model
        for _ in range(self.n_planning):
            if len(self.visited) == 0:
                break
            idx = np.random.randint(len(self.visited))
            sim_s, sim_a = self.visited[idx]
            sim_r, sim_s_next, sim_done = self.model[(sim_s, sim_a)]

            sim_target = sim_r if sim_done else sim_r + self.gamma * np.max(self.Q[sim_s_next])
            self.Q[sim_s, sim_a] += self.alpha * (sim_target - self.Q[sim_s, sim_a])


def exercise_1():
    """
    Dyna-Q on GridWorld: Compare performance with different planning steps
    (n=0, 5, 50).

    n=0 is pure model-free Q-learning (no planning).
    n=5 uses 5 simulated updates per real step.
    n=50 uses 50 simulated updates per real step.

    Expected: More planning steps -> faster convergence (fewer real
    environment steps needed), at the cost of more computation per step.
    """
    print("Exercise 1: Dyna-Q with Different Planning Steps")
    print("=" * 60)

    env = GridWorld(size=5)
    n_episodes = 300
    n_runs = 5  # Average over multiple runs for reliability
    planning_values = [0, 5, 50]

    # Store steps-per-episode for each planning setting
    all_results = {}

    for n_planning in planning_values:
        run_steps = np.zeros((n_runs, n_episodes))

        for run in range(n_runs):
            agent = DynaQ(
                n_states=env.n_states,
                n_actions=env.n_actions,
                alpha=0.1,
                gamma=0.95,
                epsilon=0.1,
                n_planning=n_planning
            )

            for ep in range(n_episodes):
                state = env.reset()
                steps = 0
                done = False

                while not done and steps < 500:  # max 500 steps per episode
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    steps += 1

                run_steps[run, ep] = steps

        all_results[n_planning] = run_steps.mean(axis=0)

    # Display results
    print()
    print(f"Average steps per episode (over {n_runs} runs):")
    print(f"{'Episode':>8} | {'n=0':>8} | {'n=5':>8} | {'n=50':>8}")
    print("-" * 42)

    # Show key episodes
    for ep in [0, 9, 19, 49, 99, 199, 299]:
        if ep < n_episodes:
            row = f"{ep+1:>8} |"
            for n in planning_values:
                row += f" {all_results[n][ep]:>8.1f} |"
            print(row)

    # Show average over last 50 episodes
    print()
    print("Average steps over last 50 episodes:")
    for n in planning_values:
        avg = all_results[n][-50:].mean()
        print(f"  n_planning={n:>2}: {avg:.1f} steps")

    # Show when each setting first reaches near-optimal
    print()
    print("Episode when average steps first drops below 20:")
    for n in planning_values:
        # Use a moving average window of 10
        window = 10
        smoothed = np.convolve(all_results[n], np.ones(window) / window, mode='valid')
        found = False
        for i, val in enumerate(smoothed):
            if val < 20:
                print(f"  n_planning={n:>2}: episode {i + window}")
                found = True
                break
        if not found:
            print(f"  n_planning={n:>2}: did not reach threshold in {n_episodes} episodes")

    print()
    print("Key observations:")
    print("  1. n=0 (pure Q-learning) learns slowly — each real transition")
    print("     only produces a single Q-update.")
    print("  2. n=5 learns significantly faster — 5 extra simulated updates")
    print("     per real step helps propagate value information.")
    print("  3. n=50 learns fastest — near-optimal paths are found quickly")
    print("     because planning 'rehearses' many state transitions per step.")
    print("  4. More planning = more CPU time per real step, but fewer real")
    print("     steps needed. In sample-expensive environments (robotics),")
    print("     the trade-off strongly favors more planning.")


def exercise_2():
    """
    Neural Dynamics Model: Train a simple dynamics model on a continuous
    control task (simulated pendulum) and evaluate prediction accuracy.

    Shows that:
    - 1-step prediction error is small
    - Multi-step rollout error grows with horizon (compounding errors)
    - Ensemble disagreement correlates with actual error
    """
    print("\nExercise 2: Neural Dynamics Model")
    print("=" * 60)

    # --- Simple Pendulum Environment (no gym dependency) ---
    class SimplePendulum:
        """
        Simplified pendulum dynamics.
        State: [cos(theta), sin(theta), theta_dot]
        Action: torque in [-2, 2]
        """

        def __init__(self):
            self.max_speed = 8.0
            self.max_torque = 2.0
            self.dt = 0.05
            self.g = 10.0
            self.m = 1.0
            self.l = 1.0
            self.theta = 0.0
            self.theta_dot = 0.0

        def reset(self):
            self.theta = np.random.uniform(-np.pi, np.pi)
            self.theta_dot = np.random.uniform(-1, 1)
            return self._get_obs()

        def step(self, action):
            action = np.clip(action, -self.max_torque, self.max_torque)
            # Pendulum dynamics: theta_ddot = -3g/(2l) * sin(theta + pi) + 3/(ml^2) * u
            theta_ddot = (-3.0 * self.g / (2.0 * self.l) *
                          np.sin(self.theta + np.pi) +
                          3.0 / (self.m * self.l ** 2) * action)
            self.theta_dot = np.clip(
                self.theta_dot + theta_ddot * self.dt,
                -self.max_speed, self.max_speed
            )
            self.theta += self.theta_dot * self.dt
            # Normalize theta to [-pi, pi]
            self.theta = ((self.theta + np.pi) % (2 * np.pi)) - np.pi

            reward = -(self.theta ** 2 + 0.1 * self.theta_dot ** 2 +
                       0.001 * action ** 2)
            return self._get_obs(), reward

        def _get_obs(self):
            return np.array([np.cos(self.theta), np.sin(self.theta),
                             self.theta_dot])

    # --- Collect data from random policy ---
    print("\n1. Collecting data from random policy...")
    env = SimplePendulum()
    states, actions, next_states, rewards = [], [], [], []

    for _ in range(200):  # 200 episodes
        obs = env.reset()
        for _ in range(200):  # 200 steps per episode
            action = np.random.uniform(-2, 2)
            next_obs, reward = env.step(action)
            states.append(obs)
            actions.append([action])
            next_states.append(next_obs)
            rewards.append(reward)
            obs = next_obs

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    print(f"   Collected {len(states)} transitions")

    # --- Simple Neural Network (numpy only) ---
    class SimpleNeuralNet:
        """
        2-layer MLP for dynamics prediction.
        Predicts next_state - state (residual) and reward.
        """

        def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
            # Xavier initialization
            scale1 = np.sqrt(2.0 / input_dim)
            scale2 = np.sqrt(2.0 / hidden_dim)
            self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
            self.b1 = np.zeros(hidden_dim, dtype=np.float32)
            self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale2
            self.b2 = np.zeros(hidden_dim, dtype=np.float32)
            self.W3 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * scale2
            self.b3 = np.zeros(output_dim, dtype=np.float32)
            self.lr = lr

        def forward(self, x):
            """Forward pass with ReLU activations."""
            self.z1 = x @ self.W1 + self.b1
            self.a1 = np.maximum(0, self.z1)  # ReLU
            self.z2 = self.a1 @ self.W2 + self.b2
            self.a2 = np.maximum(0, self.z2)  # ReLU
            self.out = self.a2 @ self.W3 + self.b3
            return self.out

        def backward(self, x, target):
            """Backpropagation with MSE loss."""
            batch_size = x.shape[0]
            pred = self.forward(x)
            error = pred - target  # (batch, output_dim)
            loss = np.mean(error ** 2)

            # Gradient of output layer
            dout = 2.0 * error / batch_size
            dW3 = self.a2.T @ dout
            db3 = dout.sum(axis=0)

            # Gradient through second hidden layer
            da2 = dout @ self.W3.T
            da2 = da2 * (self.z2 > 0)  # ReLU derivative
            dW2 = self.a1.T @ da2
            db2 = da2.sum(axis=0)

            # Gradient through first hidden layer
            da1 = da2 @ self.W2.T
            da1 = da1 * (self.z1 > 0)  # ReLU derivative
            dW1 = x.T @ da1
            db1 = da1.sum(axis=0)

            # Gradient clipping to prevent explosion
            for grad in [dW1, db1, dW2, db2, dW3, db3]:
                np.clip(grad, -1.0, 1.0, out=grad)

            # Update weights
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W3 -= self.lr * dW3
            self.b3 -= self.lr * db3

            return loss

        def predict(self, state, action):
            """Predict next_state given state and action."""
            x = np.concatenate([state, action], axis=-1)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            residual = self.forward(x)
            # Output: residual (next_state - state) and reward
            next_state_pred = state + residual[:, :state.shape[-1]]
            reward_pred = residual[:, state.shape[-1]:]
            return next_state_pred, reward_pred

    # --- Train ensemble of dynamics models ---
    print("\n2. Training ensemble of 5 dynamics models...")
    state_dim = 3
    action_dim = 1
    n_models = 5
    input_dim = state_dim + action_dim
    output_dim = state_dim + 1  # residual state + reward

    # Compute residuals (next_state - state) as targets
    residuals = next_states - states
    targets = np.concatenate([residuals, rewards.reshape(-1, 1)], axis=1)

    # Normalize inputs and targets
    inputs = np.concatenate([states, actions], axis=1)
    input_mean, input_std = inputs.mean(axis=0), inputs.std(axis=0) + 1e-8
    target_mean, target_std = targets.mean(axis=0), targets.std(axis=0) + 1e-8

    inputs_norm = (inputs - input_mean) / input_std
    targets_norm = (targets - target_mean) / target_std

    ensemble = []
    for i in range(n_models):
        model = SimpleNeuralNet(input_dim, 64, output_dim, lr=3e-4)
        # Each model trained on a different bootstrap sample
        n_data = len(inputs_norm)
        for epoch in range(50):
            # Bootstrap sample (sampling with replacement gives diversity)
            idx = np.random.randint(0, n_data, size=min(512, n_data))
            batch_x = inputs_norm[idx]
            batch_y = targets_norm[idx]
            loss = model.backward(batch_x, batch_y)
        ensemble.append(model)
        print(f"   Model {i+1}: final training loss = {loss:.6f}")

    # --- Evaluate 1-step prediction accuracy ---
    print("\n3. Evaluating 1-step prediction accuracy...")
    # Use a separate test set (last 5000 transitions)
    test_inputs = inputs_norm[-5000:]
    test_targets = targets_norm[-5000:]
    test_states_raw = states[-5000:]
    test_next_states_raw = next_states[-5000:]

    one_step_errors = []
    ensemble_preds = []
    for model in ensemble:
        preds = model.forward(test_inputs)
        # Denormalize
        preds_denorm = preds * target_std + target_mean
        pred_next_states = test_states_raw + preds_denorm[:, :state_dim]
        mse = np.mean((pred_next_states - test_next_states_raw) ** 2)
        one_step_errors.append(mse)
        ensemble_preds.append(pred_next_states)

    print(f"   1-step MSE per model: {[f'{e:.6f}' for e in one_step_errors]}")
    print(f"   Ensemble mean 1-step MSE: {np.mean(one_step_errors):.6f}")

    # --- Evaluate multi-step rollout error ---
    print("\n4. Evaluating multi-step rollout error (error compounding)...")
    rollout_lengths = [1, 5, 10, 20]
    rollout_mses = {}

    for H in rollout_lengths:
        errors = []
        n_rollouts = 100

        for _ in range(n_rollouts):
            # Pick a random starting point
            start_idx = np.random.randint(0, len(states) - H - 1)
            current_state = states[start_idx].copy()

            # Roll out H steps using ensemble mean prediction
            for step in range(H):
                action = actions[start_idx + step]
                x = np.concatenate([current_state, action]).reshape(1, -1)
                x_norm = (x - input_mean) / input_std

                # Average prediction from ensemble
                pred_residuals = []
                for model in ensemble:
                    pred = model.forward(x_norm)
                    pred_denorm = pred * target_std + target_mean
                    pred_residuals.append(pred_denorm[0, :state_dim])

                mean_residual = np.mean(pred_residuals, axis=0)
                current_state = current_state + mean_residual

            # Compare with actual state after H steps
            actual_state = states[start_idx + H] if start_idx + H < len(states) else next_states[-1]
            error = np.mean((current_state - actual_state) ** 2)
            errors.append(error)

        rollout_mses[H] = np.mean(errors)

    print(f"\n   Rollout Length vs MSE:")
    print(f"   {'H':>5} | {'MSE':>12} | {'Ratio to H=1':>15}")
    print(f"   {'-'*40}")
    base_mse = rollout_mses[1]
    for H in rollout_lengths:
        ratio = rollout_mses[H] / base_mse if base_mse > 0 else float('inf')
        print(f"   {H:>5} | {rollout_mses[H]:>12.6f} | {ratio:>15.2f}x")

    # --- Ensemble disagreement vs actual error ---
    print("\n5. Ensemble disagreement correlation with actual error...")
    disagreements = []
    actual_errors = []

    for i in range(min(1000, len(test_inputs))):
        x = test_inputs[i:i+1]
        preds_list = []
        for model in ensemble:
            pred = model.forward(x)
            preds_list.append(pred[0])
        preds_arr = np.array(preds_list)
        # Disagreement = std across models
        disagreement = np.mean(np.std(preds_arr, axis=0))
        disagreements.append(disagreement)

        # Actual error = distance from mean prediction to true target
        mean_pred = np.mean(preds_arr, axis=0)
        actual_error = np.mean((mean_pred - test_targets[i]) ** 2)
        actual_errors.append(actual_error)

    disagreements = np.array(disagreements)
    actual_errors = np.array(actual_errors)

    # Compute correlation
    if np.std(disagreements) > 0 and np.std(actual_errors) > 0:
        correlation = np.corrcoef(disagreements, actual_errors)[0, 1]
    else:
        correlation = 0.0

    # Split into quartiles by disagreement and show error
    sorted_idx = np.argsort(disagreements)
    n = len(sorted_idx)
    quartile_size = n // 4

    print(f"\n   Correlation between disagreement and actual error: {correlation:.4f}")
    print()
    print(f"   {'Quartile':>10} | {'Avg Disagreement':>18} | {'Avg Error':>12}")
    print(f"   {'-'*50}")
    for q in range(4):
        start = q * quartile_size
        end = (q + 1) * quartile_size if q < 3 else n
        q_idx = sorted_idx[start:end]
        avg_dis = disagreements[q_idx].mean()
        avg_err = actual_errors[q_idx].mean()
        labels = ["Low", "Med-Low", "Med-High", "High"]
        print(f"   {labels[q]:>10} | {avg_dis:>18.6f} | {avg_err:>12.6f}")

    print()
    print("Key observations:")
    print("  1. 1-step prediction error is small — the model learns local dynamics well.")
    print("  2. Multi-step error grows significantly with rollout length due to")
    print("     compounding: each step's error shifts the state distribution,")
    print("     pushing the model into regions it hasn't been trained on.")
    print("  3. Ensemble disagreement correlates positively with actual error,")
    print("     confirming it as a useful uncertainty proxy.")
    print("  4. MBPO mitigates compounding by using short rollouts (H=1-5)")
    print("     branched from real states, keeping the model close to the")
    print("     training distribution.")


def exercise_3():
    """
    Compare Sample Efficiency: Model-Free Q-learning vs Dyna-Q
    (model-based) on a stochastic GridWorld.

    This demonstrates the core advantage of model-based RL: reaching
    the same performance with far fewer real environment interactions.
    """
    print("\nExercise 3: Model-Free vs Model-Based Sample Efficiency")
    print("=" * 60)

    # --- Stochastic GridWorld (slightly harder) ---
    class StochasticGridWorld:
        """
        6x6 GridWorld with stochastic transitions.
        With probability 0.8, the intended action is taken.
        With probability 0.2, a random action is taken instead.
        """

        def __init__(self, size=6):
            self.size = size
            self.walls = {(1, 1), (1, 2), (2, 4), (3, 1), (3, 4), (4, 2)}
            self.start = (0, 0)
            self.goal = (size - 1, size - 1)
            self.n_states = size * size
            self.n_actions = 4
            self.stochasticity = 0.2
            self.state = self.start

        def reset(self):
            self.state = self.start
            return self._state_to_idx(self.state)

        def step(self, action):
            # Stochastic: 80% intended action, 20% random
            if np.random.random() < self.stochasticity:
                action = np.random.randint(self.n_actions)

            row, col = self.state
            if action == 0:    # up
                row = max(row - 1, 0)
            elif action == 1:  # right
                col = min(col + 1, self.size - 1)
            elif action == 2:  # down
                row = min(row + 1, self.size - 1)
            elif action == 3:  # left
                col = max(col - 1, 0)

            new_state = (row, col)
            if new_state in self.walls:
                new_state = self.state

            self.state = new_state
            done = (self.state == self.goal)
            reward = 1.0 if done else -0.01  # Small step penalty
            return self._state_to_idx(self.state), reward, done

        def _state_to_idx(self, state):
            return state[0] * self.size + state[1]

    # --- Run comparison ---
    n_episodes = 500
    n_runs = 10
    max_steps = 200

    configs = {
        "Q-Learning (n=0)": 0,
        "Dyna-Q (n=5)": 5,
        "Dyna-Q (n=20)": 20,
        "Dyna-Q (n=50)": 50,
    }

    results = {}
    cumulative_steps = {}

    for name, n_planning in configs.items():
        run_rewards = np.zeros((n_runs, n_episodes))
        run_cum_steps = np.zeros((n_runs, n_episodes))

        for run in range(n_runs):
            env = StochasticGridWorld()
            agent = DynaQ(
                n_states=env.n_states,
                n_actions=env.n_actions,
                alpha=0.1,
                gamma=0.95,
                epsilon=0.1,
                n_planning=n_planning
            )

            total_real_steps = 0
            for ep in range(n_episodes):
                state = env.reset()
                ep_reward = 0
                ep_steps = 0
                done = False

                while not done and ep_steps < max_steps:
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    ep_reward += reward
                    ep_steps += 1
                    total_real_steps += 1

                run_rewards[run, ep] = ep_reward
                run_cum_steps[run, ep] = total_real_steps

        results[name] = run_rewards.mean(axis=0)
        cumulative_steps[name] = run_cum_steps.mean(axis=0)

    # Display results
    print()
    print(f"Average reward over last 50 episodes ({n_runs} runs):")
    for name in configs:
        avg = results[name][-50:].mean()
        total_steps = cumulative_steps[name][-1]
        print(f"  {name:>20}: avg_reward = {avg:.3f}, "
              f"total_real_steps = {total_steps:.0f}")

    # Show sample efficiency: steps to reach threshold reward
    print()
    threshold = 0.5
    window = 20
    print(f"Real environment steps to reach avg reward > {threshold} (window={window}):")
    for name in configs:
        smoothed = np.convolve(results[name], np.ones(window) / window, mode='valid')
        found = False
        for i, val in enumerate(smoothed):
            if val > threshold:
                # Find cumulative steps at this episode
                ep_idx = i + window - 1
                steps_needed = cumulative_steps[name][ep_idx]
                print(f"  {name:>20}: {steps_needed:.0f} real steps "
                      f"(episode {ep_idx + 1})")
                found = True
                break
        if not found:
            print(f"  {name:>20}: did not reach threshold")

    # Show computation trade-off
    print()
    print("Computation trade-off (Q-updates per real step):")
    for name, n in configs.items():
        q_updates = 1 + n  # 1 direct + n planning
        print(f"  {name:>20}: {q_updates} Q-updates per real environment step")

    print()
    print("Key observations:")
    print("  1. SAMPLE EFFICIENCY: Dyna-Q reaches good performance with fewer")
    print("     real environment interactions than pure Q-learning.")
    print("  2. DIMINISHING RETURNS: n=5 gives a big jump over n=0, but")
    print("     n=50 vs n=20 offers smaller marginal benefit.")
    print("  3. COMPUTATION COST: Each real step requires (1 + n) Q-updates.")
    print("     More planning = more CPU, but fewer real interactions.")
    print("  4. STOCHASTICITY: The tabular model stores only one (r, s')")
    print("     per (s, a), so it approximates stochastic environments.")
    print("     Neural network models can capture distributions better.")
    print("  5. PRACTICAL IMPLICATION: In robotics or other domains where")
    print("     real interactions are expensive/dangerous, the extra")
    print("     computation of model-based RL is a worthwhile trade-off.")


if __name__ == "__main__":
    print("=== Exercise 1: Dyna-Q on GridWorld ===")
    exercise_1()

    print("\n=== Exercise 2: Neural Dynamics Model ===")
    exercise_2()

    print("\n=== Exercise 3: Model-Free vs Model-Based Sample Efficiency ===")
    exercise_3()

    print("\nAll exercises completed!")
