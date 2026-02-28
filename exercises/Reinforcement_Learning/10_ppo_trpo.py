"""
Exercises for Lesson 10: PPO and TRPO
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: TRPO vs PPO Trade-off Analysis

    Analyze the theoretical and practical differences between TRPO and PPO.
    """
    print("TRPO vs PPO Trade-off Analysis:")
    print("=" * 60)
    print()

    print("1. Why TRPO requires the Hessian of the KL divergence:")
    print("   TRPO solves a constrained optimization problem:")
    print("     maximize L(theta) s.t. KL(pi_old || pi_theta) <= delta")
    print()
    print("   Using Lagrangian methods and a second-order Taylor expansion of KL,")
    print("   the natural policy gradient direction requires solving:")
    print("     F * x = g")
    print("   where F is the Fisher Information Matrix (= Hessian of KL at theta_old),")
    print("   g is the policy gradient, and x is the update direction.")
    print()
    print("   Computational complexity of the Hessian:")
    print("   - Computing: O(n^2) where n = number of parameters")
    print("   - Storing: O(n^2) memory")
    print("   - Inverting: O(n^3)")
    print("   - For a network with 1M parameters: 1 trillion elements in Hessian!")
    print("   - TRPO uses conjugate gradient to avoid explicit inversion: O(n*k)")
    print("     where k is the number of CG iterations (~10-20)")
    print()

    print("2. How PPO's clipped objective approximates the TRPO constraint:")
    print("   PPO replaces the hard KL constraint with a soft clipping mechanism:")
    print("     L^CLIP = min(r(theta)*A, clip(r(theta), 1-eps, 1+eps)*A)")
    print("   where r(theta) = pi_new(a|s) / pi_old(a|s)")
    print()
    print("   This effectively limits how much the policy ratio can deviate from 1,")
    print("   which is a heuristic approximation of bounding the KL divergence.")
    print("   When the ratio is within [1-eps, 1+eps], the gradient flows normally.")
    print("   Outside this range, the gradient is zeroed out, preventing large updates.")
    print()

    print("3. Comparison table:")
    print()
    print(f"{'Property':<30} {'TRPO':<25} {'PPO-Clip':<25}")
    print("-" * 80)
    properties = [
        ("Constraint type", "Hard KL constraint", "Soft clipping heuristic"),
        ("Gradient order", "Second-order", "First-order only"),
        ("Implementation complexity", "High (CG, line search)", "Low (standard SGD)"),
        ("Memory cost", "O(n^2) or O(n*k)", "O(n)"),
        ("Typical wall-clock speed", "Slower (2-5x)", "Faster"),
    ]
    for prop, trpo, ppo in properties:
        print(f"{prop:<30} {trpo:<25} {ppo:<25}")
    print()

    print("4. When TRPO's hard constraint is strictly necessary:")
    print("   - Safety-critical applications (robotics, autonomous driving)")
    print("     where a bad policy update could cause physical damage.")
    print("   - When PPO's clipping fails to prevent catastrophic updates")
    print("     (can happen with very high advantage estimates).")
    print("   - Theoretical guarantees: TRPO provides monotonic improvement")
    print("     guarantees, while PPO's clipping is a heuristic.")
    print("   - In practice, PPO works well enough for most applications.")


def exercise_2():
    """
    Exercise 2: Clipping Behavior Visualization

    Visualize how the PPO clipped objective responds to different ratio values.
    """
    print("PPO Clipping Behavior Visualization:")
    print("=" * 60)
    print()

    eps = 0.2
    r_values = np.linspace(0.5, 1.5, 101)

    def l_clip(r, A, eps=0.2):
        """Compute PPO clipped objective for each ratio value."""
        clipped = np.clip(r, 1 - eps, 1 + eps)
        return np.minimum(r * A, clipped * A)

    advantage_values = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

    print(f"Epsilon = {eps}")
    print(f"Clipping boundaries: r = {1-eps} and r = {1+eps}")
    print()

    for A in advantage_values:
        values = l_clip(r_values, A, eps)
        # Show key points
        r_08 = l_clip(np.array([0.8]), A, eps)[0]
        r_10 = l_clip(np.array([1.0]), A, eps)[0]
        r_12 = l_clip(np.array([1.2]), A, eps)[0]
        r_05 = l_clip(np.array([0.5]), A, eps)[0]
        r_15 = l_clip(np.array([1.5]), A, eps)[0]

        print(f"  A = {A:+.1f}: L(r=0.5)={r_05:+6.2f}, L(r=0.8)={r_08:+6.2f}, "
              f"L(r=1.0)={r_10:+6.2f}, L(r=1.2)={r_12:+6.2f}, L(r=1.5)={r_15:+6.2f}")

    print()
    print("ASCII Plot of L^CLIP for A=+1.0 and A=-1.0:")
    print("-" * 60)

    # Simple ASCII visualization
    for A_val, label in [(1.0, "A=+1.0"), (-1.0, "A=-1.0")]:
        print(f"\n  {label}:")
        values = l_clip(r_values, A_val, eps)
        min_v, max_v = np.min(values), np.max(values)
        width = 40

        for i in range(0, len(r_values), 5):
            r = r_values[i]
            v = values[i]
            if max_v - min_v > 0:
                bar_len = int((v - min_v) / (max_v - min_v) * width)
            else:
                bar_len = width // 2
            marker = "|" if abs(r - 0.8) < 0.03 or abs(r - 1.2) < 0.03 else " "
            print(f"  r={r:.2f} {marker} {'#' * bar_len:<{width}} {v:+.2f}")

    print()
    print("Explanation of flat regions:")
    print("  When A > 0 (good action):")
    print("    - For r > 1+eps: L^CLIP is flat at (1+eps)*A")
    print("    - The gradient is zero, preventing further increase in pi_new")
    print("    - This caps the 'benefit' of reinforcing a good action")
    print()
    print("  When A < 0 (bad action):")
    print("    - For r < 1-eps: L^CLIP is flat at (1-eps)*A")
    print("    - The gradient is zero, preventing further decrease in pi_new")
    print("    - This caps the 'punishment' of a bad action")
    print()
    print("  The min() operation is pessimistic: it always takes the more")
    print("  conservative estimate, ensuring stable updates.")


def exercise_3():
    """
    Exercise 3: GAE Lambda Sensitivity

    Investigate how the GAE lambda parameter affects bias-variance trade-off.
    """
    print("GAE Lambda Sensitivity Analysis:")
    print("=" * 60)
    print()

    np.random.seed(42)

    # Simple environment: 1D random walk with stochastic rewards
    class RandomWalkEnv:
        def __init__(self, n_states=10):
            self.n_states = n_states
            self.goal = n_states - 1

        def reset(self):
            self.state = self.n_states // 2
            self.steps = 0
            return self.state

        def step(self, action):
            self.steps += 1
            if action == 1:  # right
                self.state = min(self.state + 1, self.n_states - 1)
            else:  # left
                self.state = max(self.state - 1, 0)

            reward = 1.0 if self.state == self.goal else -0.01
            done = self.state == self.goal or self.state == 0 or self.steps >= 50
            return self.state, reward, done

    def compute_gae(rewards, values, next_values, dones, gamma, lam):
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros(len(rewards))
        gae = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + gamma * next_values[t] - values[t]
                gae = delta + gamma * lam * gae
            advantages[t] = gae
        return advantages

    # Simple tabular value function
    def train_with_lambda(lam, n_episodes=500, gamma=0.99):
        env = RandomWalkEnv(n_states=10)
        V = np.zeros(10)
        lr_v = 0.1
        lr_p = 0.01

        # Simple softmax policy
        policy_logits = np.zeros((10, 2))
        episode_rewards = []

        for episode in range(n_episodes):
            state = env.reset()
            states, actions, rewards_list, next_states, dones = [], [], [], [], []
            total_reward = 0

            done = False
            while not done:
                # Softmax policy
                logits = policy_logits[state]
                probs = np.exp(logits - np.max(logits))
                probs /= probs.sum()
                action = np.random.choice(2, p=probs)

                next_state, reward, done = env.step(action)

                states.append(state)
                actions.append(action)
                rewards_list.append(reward)
                next_states.append(next_state)
                dones.append(done)
                total_reward += reward

                state = next_state

            episode_rewards.append(total_reward)

            # Compute values and GAE
            values = np.array([V[s] for s in states])
            next_vals = np.array([V[s] if not d else 0.0
                                  for s, d in zip(next_states, dones)])

            advantages = compute_gae(
                np.array(rewards_list), values, next_vals,
                np.array(dones), gamma, lam
            )

            # Update value function
            returns = advantages + values
            for t in range(len(states)):
                V[states[t]] += lr_v * (returns[t] - V[states[t]])

            # Update policy
            for t in range(len(states)):
                s, a = states[t], actions[t]
                adv = advantages[t]
                probs = np.exp(policy_logits[s] - np.max(policy_logits[s]))
                probs /= probs.sum()
                grad = np.zeros(2)
                grad[a] = 1.0
                grad -= probs
                policy_logits[s] += lr_p * grad * adv

        return episode_rewards

    # Test different lambda values
    lambda_values = [0.0, 0.5, 0.9, 0.95, 1.0]
    results = {}

    for lam in lambda_values:
        rewards = train_with_lambda(lam)
        last_50_mean = np.mean(rewards[-50:])
        last_50_std = np.std(rewards[-50:])
        results[lam] = (last_50_mean, last_50_std, rewards)

    print(f"{'lambda':>8} {'Mean(last 50)':>15} {'Std(last 50)':>15} {'Description'}")
    print("-" * 70)
    descriptions = {
        0.0: "Pure 1-step TD (high bias, low var)",
        0.5: "Moderate blend",
        0.9: "Mostly MC-like",
        0.95: "Default PPO value",
        1.0: "Pure MC (no bias, high var)",
    }
    for lam in lambda_values:
        mean, std, _ = results[lam]
        print(f"{lam:>8.2f} {mean:>15.3f} {std:>15.3f}   {descriptions[lam]}")

    print()
    print("Analysis:")
    print("  lambda=0 (1-step TD): Uses only one reward + bootstrap.")
    print("    Low variance but high bias if V is inaccurate.")
    print()
    print("  lambda=1 (MC): Uses the full episode return.")
    print("    Unbiased but high variance, especially for long episodes.")
    print()
    print("  lambda=0.95 (PPO default): Good balance between bias and variance.")
    print("    Uses mostly real rewards with a small bootstrap correction.")
    print("    This typically matches or exceeds other lambda values in practice.")


def exercise_4():
    """
    Exercise 4: Implement PPO-Penalty

    Implement the KL-penalty variant of PPO and compare with clipping.
    """
    print("PPO-Penalty Implementation:")
    print("=" * 60)
    print()

    np.random.seed(42)

    def softmax(logits):
        logits = logits - np.max(logits)
        exp_l = np.exp(logits)
        return exp_l / np.sum(exp_l)

    def kl_divergence(old_probs, new_probs):
        """KL(old || new)"""
        return np.sum(old_probs * np.log(old_probs / (new_probs + 1e-10) + 1e-10))

    # Simple bandit environment for demonstration
    n_actions = 4
    true_rewards = [1.0, 3.0, 5.0, 2.0]  # Action 2 is optimal

    def ppo_clip_train(n_steps=1000, eps=0.2, lr=0.05):
        logits = np.zeros(n_actions)
        rewards_hist = []

        for step in range(n_steps):
            old_probs = softmax(logits)
            action = np.random.choice(n_actions, p=old_probs)
            reward = np.random.normal(true_rewards[action], 1.0)
            rewards_hist.append(reward)

            # PPO-Clip update
            new_probs = softmax(logits)
            ratio = new_probs[action] / (old_probs[action] + 1e-10)
            clipped_ratio = np.clip(ratio, 1 - eps, 1 + eps)
            advantage = reward - np.sum(old_probs * true_rewards)

            # Gradient for softmax policy
            grad = np.zeros(n_actions)
            grad[action] = 1.0
            grad -= new_probs

            effective_ratio = min(ratio * advantage, clipped_ratio * advantage)
            logits += lr * grad * effective_ratio

        return rewards_hist, softmax(logits)

    def ppo_penalty_train(n_steps=1000, target_kl=0.01, initial_beta=1.0, lr=0.05):
        logits = np.zeros(n_actions)
        beta = initial_beta
        rewards_hist = []
        beta_hist = []

        for step in range(n_steps):
            old_probs = softmax(logits)
            action = np.random.choice(n_actions, p=old_probs)
            reward = np.random.normal(true_rewards[action], 1.0)
            rewards_hist.append(reward)

            # PPO-Penalty update
            advantage = reward - np.sum(old_probs * true_rewards)

            # Gradient for policy loss - beta * KL
            grad = np.zeros(n_actions)
            grad[action] = 1.0
            grad -= softmax(logits)

            logits += lr * grad * advantage

            # Measure KL after update
            new_probs = softmax(logits)
            kl = kl_divergence(old_probs, new_probs)

            # Adaptive beta adjustment
            if kl > 1.5 * target_kl:
                beta *= 2.0  # KL too large, increase penalty
            elif kl < target_kl / 1.5:
                beta /= 2.0  # KL too small, decrease penalty

            beta_hist.append(beta)

            # Apply KL penalty to pull logits back
            kl_grad = new_probs - old_probs  # Approximate KL gradient
            logits -= lr * beta * kl_grad

        return rewards_hist, softmax(logits), beta_hist

    # Train both
    clip_rewards, clip_probs = ppo_clip_train()
    penalty_rewards, penalty_probs, beta_hist = ppo_penalty_train()

    print("Results after 1000 steps:")
    print(f"  True rewards: {true_rewards}")
    print(f"  Optimal action: 2 (reward = {true_rewards[2]})")
    print()

    print("PPO-Clip:")
    print(f"  Final probs: [{', '.join(f'{p:.3f}' for p in clip_probs)}]")
    print(f"  Mean reward (last 100): {np.mean(clip_rewards[-100:]):.3f}")
    print()

    print("PPO-Penalty:")
    print(f"  Final probs: [{', '.join(f'{p:.3f}' for p in penalty_probs)}]")
    print(f"  Mean reward (last 100): {np.mean(penalty_rewards[-100:]):.3f}")
    print(f"  Final beta: {beta_hist[-1]:.4f}")
    print(f"  Beta range: [{min(beta_hist):.4f}, {max(beta_hist):.4f}]")
    print()

    print("Adaptive beta behavior:")
    for start in range(0, 1000, 200):
        end = start + 200
        avg_beta = np.mean(beta_hist[start:end])
        print(f"  Steps {start}-{end}: avg_beta = {avg_beta:.4f}")

    print()
    print("Does adaptive beta effectively control KL?")
    print("  Yes -- when KL exceeds target, beta increases to penalize large")
    print("  policy changes. When KL is small, beta decreases to allow faster")
    print("  learning. This self-tuning mechanism provides trust-region-like")
    print("  behavior without requiring second-order optimization.")


def exercise_5():
    """
    Exercise 5: Vectorized Environment PPO

    Demonstrate the concept of parallel environments for faster data collection.
    """
    print("Vectorized Environment PPO:")
    print("=" * 60)
    print()

    np.random.seed(42)

    # Simple grid navigation environment
    class SimpleGridEnv:
        def __init__(self, size=5):
            self.size = size
            self.goal = (size-1, size-1)

        def reset(self):
            self.pos = (0, 0)
            self.steps = 0
            return np.array(self.pos, dtype=np.float32) / self.size

        def step(self, action):
            self.steps += 1
            deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
            dr, dc = deltas[action]
            r, c = self.pos
            self.pos = (max(0, min(self.size-1, r+dr)),
                        max(0, min(self.size-1, c+dc)))
            reward = 1.0 if self.pos == self.goal else -0.01
            done = self.pos == self.goal or self.steps >= 50
            return np.array(self.pos, dtype=np.float32) / self.size, reward, done

    # Vectorized environment wrapper
    class VectorizedEnv:
        def __init__(self, env_class, n_envs, **kwargs):
            self.envs = [env_class(**kwargs) for _ in range(n_envs)]
            self.n_envs = n_envs

        def reset(self):
            return np.array([env.reset() for env in self.envs])

        def step(self, actions):
            results = [env.step(a) for env, a in zip(self.envs, actions)]
            obs = np.array([r[0] for r in results])
            rewards = np.array([r[1] for r in results])
            dones = np.array([r[2] for r in results])

            # Auto-reset completed environments
            for i, done in enumerate(dones):
                if done:
                    obs[i] = self.envs[i].reset()

            return obs, rewards, dones

    # Compare single vs vectorized
    import time

    def collect_rollout_single(n_steps, env):
        state = env.reset()
        transitions = 0
        for _ in range(n_steps):
            action = np.random.randint(4)
            state, reward, done = env.step(action)
            transitions += 1
            if done:
                state = env.reset()
        return transitions

    def collect_rollout_vectorized(n_steps, vec_env):
        states = vec_env.reset()
        transitions = 0
        for _ in range(n_steps):
            actions = np.random.randint(4, size=vec_env.n_envs)
            states, rewards, dones = vec_env.step(actions)
            transitions += vec_env.n_envs
        return transitions

    n_steps = 256
    n_envs = 4

    # Single environment
    single_env = SimpleGridEnv()
    t0 = time.time()
    single_transitions = collect_rollout_single(n_steps, single_env)
    t1 = time.time()
    single_time = t1 - t0

    # Vectorized environment
    vec_env = VectorizedEnv(SimpleGridEnv, n_envs=n_envs)
    t0 = time.time()
    vec_transitions = collect_rollout_vectorized(n_steps, vec_env)
    t1 = time.time()
    vec_time = t1 - t0

    print(f"Single environment (1 env, {n_steps} steps):")
    print(f"  Transitions collected: {single_transitions}")
    print(f"  Time: {single_time*1000:.2f} ms")
    print()

    print(f"Vectorized environment ({n_envs} envs, {n_steps} steps):")
    print(f"  Transitions collected: {vec_transitions}")
    print(f"  Time: {vec_time*1000:.2f} ms")
    print()

    print(f"Transition ratio: {vec_transitions / single_transitions:.1f}x more transitions")
    print(f"  ({n_envs} envs collect {n_envs}x transitions per step)")
    print()

    print("Analysis:")
    print(f"  Using {n_envs} environments produces {n_envs}x more transitions")
    print(f"  for the same number of steps.")
    print()
    print("  Does it change learning QUALITY or just SPEED?")
    print("  - It primarily changes SPEED (more data per wall-clock second).")
    print("  - It also improves QUALITY slightly because:")
    print("    1. Diverse starting states: Each env may be at a different state,")
    print("       providing more diverse training data in each rollout.")
    print("    2. Reduced correlation: Transitions from different envs are")
    print("       independent, reducing the correlation in training batches.")
    print("    3. Better gradient estimates: More transitions per update")
    print("       reduces variance in the policy gradient estimate.")


if __name__ == "__main__":
    print("=== Exercise 1: TRPO vs PPO Analysis ===")
    exercise_1()

    print("\n=== Exercise 2: Clipping Behavior ===")
    exercise_2()

    print("\n=== Exercise 3: GAE Lambda Sensitivity ===")
    exercise_3()

    print("\n=== Exercise 4: PPO-Penalty ===")
    exercise_4()

    print("\n=== Exercise 5: Vectorized Environment ===")
    exercise_5()

    print("\nAll exercises completed!")
