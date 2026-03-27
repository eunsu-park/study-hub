"""
Exercises for Lesson 08: Policy Gradient
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Log-Derivative Trick Derivation

    Derive the policy gradient theorem from first principles.
    """
    print("Policy Gradient Theorem Derivation:")
    print("=" * 60)
    print()
    print("Step 1: Start from the objective")
    print("  J(theta) = E_{tau ~ pi_theta}[R(tau)]")
    print("           = integral p(tau; theta) R(tau) d_tau")
    print("  where tau = (s0, a0, r0, s1, a1, r1, ...) is a trajectory")
    print()

    print("Step 2: Take the gradient")
    print("  grad J(theta) = integral grad p(tau; theta) R(tau) d_tau")
    print()
    print("  Apply the log-derivative identity:")
    print("  grad p(tau; theta) = p(tau; theta) * grad log p(tau; theta)")
    print()
    print("  Proof of identity:")
    print("    grad log f(x) = grad f(x) / f(x)")
    print("    => grad f(x) = f(x) * grad log f(x)")
    print()

    print("Step 3: Factor the trajectory probability")
    print("  p(tau; theta) = p(s0) * prod_{t=0}^{T-1} [pi_theta(a_t|s_t) * P(s_{t+1}|s_t,a_t)]")
    print()
    print("  log p(tau; theta) = log p(s0)")
    print("                    + sum_{t=0}^{T-1} log pi_theta(a_t|s_t)")
    print("                    + sum_{t=0}^{T-1} log P(s_{t+1}|s_t,a_t)")
    print()

    print("Step 4: Take gradient of log p(tau; theta)")
    print("  grad log p(tau; theta) = sum_{t=0}^{T-1} grad log pi_theta(a_t|s_t)")
    print()
    print("  KEY INSIGHT: p(s0) and P(s'|s,a) do NOT depend on theta!")
    print("  Therefore their gradients are zero and they disappear.")
    print("  This is crucial: we don't need to know the environment dynamics!")
    print()

    print("Step 5: Final form")
    print("  grad J(theta) = E_{tau ~ pi_theta}[sum_{t=0}^T grad log pi_theta(a_t|s_t) * G_t]")
    print()
    print("  In words: 'Increase the probability of actions that led to")
    print("  high returns, decrease the probability of actions that led")
    print("  to low returns.'")
    print()

    # Numerical verification using a simple 2-action problem
    print("Numerical Verification (2-action MDP):")
    print("-" * 50)

    np.random.seed(42)

    # Simple softmax policy with one parameter theta
    def softmax_policy(theta):
        """pi(a=1|theta) = exp(theta) / (1 + exp(theta))"""
        p1 = 1 / (1 + np.exp(-theta))
        return np.array([1 - p1, p1])

    def log_grad(theta, action):
        """grad log pi(a|theta)"""
        p = softmax_policy(theta)
        if action == 1:
            return 1 - p[1]  # d/dtheta log(sigmoid(theta)) = 1 - sigmoid(theta)
        else:
            return -p[1]     # d/dtheta log(1 - sigmoid(theta)) = -sigmoid(theta)

    # Rewards: action 0 gives R=1, action 1 gives R=3
    rewards = {0: 1.0, 1: 3.0}

    theta = 0.0  # Initialize
    n_samples = 10000

    # Estimate gradient via REINFORCE
    grad_estimate = 0
    for _ in range(n_samples):
        probs = softmax_policy(theta)
        action = np.random.choice(2, p=probs)
        R = rewards[action]
        grad_estimate += log_grad(theta, action) * R

    grad_estimate /= n_samples

    # Analytical gradient
    probs = softmax_policy(theta)
    analytical_grad = sum(probs[a] * log_grad(theta, a) * rewards[a] for a in [0, 1])

    print(f"  theta = {theta}")
    print(f"  Policy: pi(a=0) = {probs[0]:.4f}, pi(a=1) = {probs[1]:.4f}")
    print(f"  REINFORCE gradient estimate: {grad_estimate:.4f}")
    print(f"  Analytical gradient:         {analytical_grad:.4f}")
    print(f"  Gradient is positive -> increase theta -> more probability on a=1 (reward=3)")


def exercise_2():
    """
    Exercise 2: Implement Vanilla REINFORCE

    Implement and train a REINFORCE agent on a simple environment
    (CartPole-like simulation without gym dependency).
    """
    print("Vanilla REINFORCE on a Simple Balancing Task:")
    print("=" * 60)
    print()

    # Simple CartPole-like environment (no gym needed)
    class SimpleCartPole:
        """Simplified cart-pole: state = [position, velocity, angle, angular_vel]"""

        def __init__(self):
            self.gravity = 9.8
            self.masscart = 1.0
            self.masspole = 0.1
            self.length = 0.5
            self.dt = 0.02
            self.max_steps = 200

        def reset(self):
            self.state = np.random.uniform(-0.05, 0.05, size=4)
            self.steps = 0
            return self.state.copy()

        def step(self, action):
            x, x_dot, theta, theta_dot = self.state
            force = 10.0 if action == 1 else -10.0

            costheta = np.cos(theta)
            sintheta = np.sin(theta)
            total_mass = self.masscart + self.masspole

            temp = (force + self.masspole * self.length * theta_dot**2 * sintheta) / total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0/3.0 - self.masspole * costheta**2 / total_mass))
            xacc = temp - self.masspole * self.length * thetaacc * costheta / total_mass

            x += self.dt * x_dot
            x_dot += self.dt * xacc
            theta += self.dt * theta_dot
            theta_dot += self.dt * thetaacc

            self.state = np.array([x, x_dot, theta, theta_dot])
            self.steps += 1

            done = (abs(x) > 2.4 or abs(theta) > 0.2095 or self.steps >= self.max_steps)
            reward = 1.0 if not done else 0.0

            return self.state.copy(), reward, done

    # Simple linear softmax policy
    class LinearPolicy:
        def __init__(self, state_dim, n_actions):
            self.weights = np.zeros((state_dim, n_actions))
            self.state_dim = state_dim
            self.n_actions = n_actions

        def get_probs(self, state):
            logits = state @ self.weights
            logits -= np.max(logits)  # numerical stability
            exp_logits = np.exp(logits)
            return exp_logits / np.sum(exp_logits)

        def get_action(self, state):
            probs = self.get_probs(state)
            action = np.random.choice(self.n_actions, p=probs)
            return action, probs

    # REINFORCE training
    env = SimpleCartPole()
    policy = LinearPolicy(state_dim=4, n_actions=2)
    gamma = 0.99
    lr = 0.01
    n_episodes = 500

    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        total_reward = 0

        # Collect episode
        done = False
        while not done:
            action, probs = policy.get_action(state)
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            total_reward += reward

            state = next_state

        episode_rewards.append(total_reward)

        # Compute returns (NO normalization - raw G_t values)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # Policy gradient update
        for t in range(len(states)):
            s = states[t]
            a = actions[t]
            G_t = returns[t]

            probs = policy.get_probs(s)
            # Gradient of log pi(a|s) for softmax: e_a - pi
            grad_log = np.zeros(policy.n_actions)
            grad_log[a] = 1.0
            grad_log -= probs

            # Update: theta <- theta + lr * grad_log_pi * G
            policy.weights += lr * np.outer(s, grad_log * G_t)

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  Episode {episode+1}: avg reward (last 50) = {avg:.1f}")

    # Show variance in learning curve
    print(f"\nLearning curve statistics:")
    for start in range(0, n_episodes, 100):
        end = start + 100
        chunk = episode_rewards[start:end]
        print(f"  Episodes {start+1}-{end}: mean={np.mean(chunk):.1f}, "
              f"std={np.std(chunk):.1f}")

    print(f"\nObservation: The learning curve shows HIGH VARIANCE.")
    print(f"This is expected because REINFORCE uses full episode returns,")
    print(f"which vary significantly from episode to episode.")


def exercise_3():
    """
    Exercise 3: Baseline Effect on Variance

    Compare the effect of different baselines on gradient variance.
    """
    print("Baseline Effect on Gradient Variance:")
    print("=" * 60)
    print()

    np.random.seed(42)

    # Simple 2-action MDP for clear demonstration
    # Action 0: reward ~ N(5, 2), Action 1: reward ~ N(7, 2)
    # Optimal: always action 1

    n_actions = 2
    true_means = [5.0, 7.0]
    reward_std = 2.0

    def simulate_episode(policy_probs):
        action = np.random.choice(n_actions, p=policy_probs)
        reward = np.random.normal(true_means[action], reward_std)
        return action, reward

    # Fixed policy for comparison
    policy_probs = np.array([0.4, 0.6])

    n_samples = 10000

    # Collect samples
    actions = np.zeros(n_samples, dtype=int)
    rewards = np.zeros(n_samples)
    for i in range(n_samples):
        actions[i], rewards[i] = simulate_episode(policy_probs)

    # Compute policy gradient estimates with different baselines
    # grad log pi(a|s) for softmax with 2 actions
    def grad_log_pi(action, probs):
        grad = np.zeros(n_actions)
        grad[action] = 1.0
        grad -= probs
        return grad

    # Method A: No baseline (raw returns)
    gradients_A = []
    for i in range(n_samples):
        g = grad_log_pi(actions[i], policy_probs) * rewards[i]
        gradients_A.append(g)
    gradients_A = np.array(gradients_A)
    var_A = np.var(gradients_A, axis=0)

    # Method B: Return normalization (zero-mean, unit-variance)
    normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    gradients_B = []
    for i in range(n_samples):
        g = grad_log_pi(actions[i], policy_probs) * normalized_rewards[i]
        gradients_B.append(g)
    gradients_B = np.array(gradients_B)
    var_B = np.var(gradients_B, axis=0)

    # Method C: Learned value baseline V(s) = E[R] under current policy
    baseline = np.sum(policy_probs * true_means)
    gradients_C = []
    for i in range(n_samples):
        advantage = rewards[i] - baseline
        g = grad_log_pi(actions[i], policy_probs) * advantage
        gradients_C.append(g)
    gradients_C = np.array(gradients_C)
    var_C = np.var(gradients_C, axis=0)

    print(f"Policy: pi(a=0)={policy_probs[0]:.1f}, pi(a=1)={policy_probs[1]:.1f}")
    print(f"True means: R(a=0)={true_means[0]}, R(a=1)={true_means[1]}")
    print(f"Baseline V = {baseline:.2f}")
    print()

    print(f"Method A (no baseline):")
    print(f"  Mean gradient: {np.mean(gradients_A, axis=0)}")
    print(f"  Variance:      {var_A}")
    print(f"  Total variance: {np.sum(var_A):.4f}")
    print()

    print(f"Method B (normalized returns):")
    print(f"  Mean gradient: {np.mean(gradients_B, axis=0)}")
    print(f"  Variance:      {var_B}")
    print(f"  Total variance: {np.sum(var_B):.4f}")
    print()

    print(f"Method C (value baseline):")
    print(f"  Mean gradient: {np.mean(gradients_C, axis=0)}")
    print(f"  Variance:      {var_C}")
    print(f"  Total variance: {np.sum(var_C):.4f}")
    print()

    print(f"Variance reduction:")
    print(f"  B vs A: {np.sum(var_A)/np.sum(var_B):.2f}x")
    print(f"  C vs A: {np.sum(var_A)/np.sum(var_C):.2f}x")
    print()

    print("Why baselines don't change the expected gradient:")
    print("  E[grad log pi(a|s) * b] = b * E[grad log pi(a|s)]")
    print("                          = b * sum_a pi(a|s) * grad log pi(a|s)")
    print("                          = b * grad sum_a pi(a|s)")
    print("                          = b * grad 1 = 0")
    print("  Since probabilities sum to 1, subtracting any constant b")
    print("  from the return doesn't change the expected gradient direction,")
    print("  but it can dramatically reduce variance by centering the signal.")


def exercise_4():
    """
    Exercise 4: Gaussian Policy for Continuous Control

    Demonstrate properties of a Gaussian policy for continuous actions.
    """
    print("Gaussian Policy for Continuous Control:")
    print("=" * 60)
    print()

    # Simple continuous control environment: 1D target reaching
    # State: [position, target], Action: force in [-2, 2]
    class SimpleReachEnv:
        def __init__(self):
            self.max_steps = 100

        def reset(self):
            self.pos = np.random.uniform(-1, 1)
            self.target = np.random.uniform(-1, 1)
            self.steps = 0
            return np.array([self.pos, self.target])

        def step(self, action):
            action = np.clip(action, -2.0, 2.0)
            self.pos = np.clip(self.pos + 0.1 * action, -2, 2)
            self.steps += 1
            dist = abs(self.pos - self.target)
            reward = -dist
            done = dist < 0.05 or self.steps >= self.max_steps
            return np.array([self.pos, self.target]), reward, done

    # Gaussian policy with learnable mean and log_std
    class GaussianPolicy:
        def __init__(self, state_dim, action_dim=1):
            self.mean_weights = np.zeros((state_dim, action_dim))
            self.log_std = 0.0  # Initialized to 0 -> std = exp(0) = 1

        @property
        def std(self):
            return np.exp(self.log_std)

        def get_action(self, state):
            mean = state @ self.mean_weights
            std = self.std
            action = np.random.normal(mean, std)
            return action.item(), mean.item(), std

        def log_prob(self, state, action):
            mean = (state @ self.mean_weights).item()
            std = self.std
            return -0.5 * ((action - mean) / std)**2 - np.log(std) - 0.5 * np.log(2 * np.pi)

    # Why log_std initialized to 0 (std=1):
    print("Why log_std is initialized to 0 (std = exp(0) = 1):")
    print("  - std=1 provides moderate initial exploration")
    print("  - Too small std -> insufficient exploration at the start")
    print("  - Too large std -> random actions, slow learning")
    print("  - log_std is learned (not std directly) to ensure std > 0 always")
    print()

    # Train and track std evolution
    env = SimpleReachEnv()
    policy = GaussianPolicy(state_dim=2)
    gamma = 0.99
    lr_mean = 0.01
    lr_std = 0.005
    n_episodes = 300

    episode_rewards = []
    std_history = []

    for episode in range(n_episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        done = False

        while not done:
            action, mean, std = policy.get_action(state)
            action_clipped = np.clip(action, -2.0, 2.0)
            next_state, reward, done = env.step(action_clipped)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        std_history.append(policy.std)

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # Normalize returns
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update policy
        for t in range(len(states)):
            s = states[t]
            a = actions[t]
            G_t = returns[t]

            mean = (s @ policy.mean_weights).item()
            std = policy.std

            # Gradient of log N(a; mean, std) w.r.t. mean weights
            grad_mean = s.reshape(-1, 1) * (a - mean) / (std**2)
            policy.mean_weights += lr_mean * grad_mean * G_t

            # Gradient of log N(a; mean, std) w.r.t. log_std
            grad_log_std = ((a - mean)**2 / std**2) - 1
            policy.log_std += lr_std * grad_log_std * G_t

    print("Training results:")
    for start in range(0, n_episodes, 60):
        end = min(start + 60, n_episodes)
        chunk_r = episode_rewards[start:end]
        chunk_std = std_history[start:end]
        print(f"  Episodes {start+1}-{end}: "
              f"avg_reward={np.mean(chunk_r):.2f}, "
              f"avg_std={np.mean(chunk_std):.4f}")

    print(f"\nInitial std: {std_history[0]:.4f}")
    print(f"Final std:   {std_history[-1]:.4f}")
    print()
    print("What the change in std tells us:")
    if std_history[-1] < std_history[0]:
        print("  std DECREASED over training, meaning the agent became more")
        print("  confident in its actions. This is expected: as the policy")
        print("  improves, it needs less random exploration and can commit")
        print("  to actions it has learned are effective.")
    else:
        print("  std did not decrease significantly. The task may need more")
        print("  training or the environment requires continued exploration.")


def exercise_5():
    """
    Exercise 5: Entropy Regularization Ablation

    Investigate how entropy coefficient affects exploration and convergence.
    """
    print("Entropy Regularization Ablation:")
    print("=" * 60)
    print()

    np.random.seed(42)

    # Simple multi-armed bandit with entropy regularization
    # This demonstrates the concept without gym dependency
    n_actions = 5
    true_means = [1.0, 2.0, 5.0, 3.0, 1.5]
    reward_std = 1.0

    def softmax(logits):
        logits = logits - np.max(logits)
        exp_l = np.exp(logits)
        return exp_l / np.sum(exp_l)

    def entropy(probs):
        """H(pi) = -sum pi * log(pi)"""
        return -np.sum(probs * np.log(probs + 1e-10))

    results = {}

    for entropy_coef in [0.0, 0.01, 0.1, 0.5]:
        np.random.seed(42)
        logits = np.zeros(n_actions)
        lr = 0.05
        n_steps = 500
        rewards_history = []
        entropy_history = []

        for step in range(n_steps):
            probs = softmax(logits)
            action = np.random.choice(n_actions, p=probs)
            reward = np.random.normal(true_means[action], reward_std)
            rewards_history.append(reward)
            entropy_history.append(entropy(probs))

            # Policy gradient with entropy bonus
            # grad log pi(a) for softmax
            grad_log = np.zeros(n_actions)
            grad_log[action] = 1.0
            grad_log -= probs

            # Entropy gradient: -grad sum(pi * log pi) = -(log pi + 1)
            # For softmax: grad_entropy = -sum_a (log pi(a) + 1) * grad pi(a)/d logits
            # Simplified: entropy bonus encourages uniform distribution
            entropy_grad = -np.log(probs + 1e-10) - 1
            entropy_grad -= np.mean(entropy_grad)  # Center

            logits += lr * (grad_log * reward + entropy_coef * entropy_grad)

        final_probs = softmax(logits)
        avg_reward_last50 = np.mean(rewards_history[-50:])
        best_action = np.argmax(final_probs)

        results[entropy_coef] = {
            'avg_reward': avg_reward_last50,
            'final_probs': final_probs,
            'best_action': best_action,
            'final_entropy': entropy(final_probs),
        }

    print(f"True reward means: {true_means}")
    print(f"Optimal action: 2 (mean reward = {true_means[2]})")
    print()

    print(f"{'entropy_coef':>12} {'avg_reward':>12} {'best_action':>12} "
          f"{'entropy':>10} {'probs':>30}")
    print("-" * 80)

    for ec, r in results.items():
        probs_str = '[' + ', '.join(f'{p:.3f}' for p in r['final_probs']) + ']'
        print(f"{ec:>12.2f} {r['avg_reward']:>12.2f} {r['best_action']:>12} "
              f"{r['final_entropy']:>10.4f} {probs_str}")

    print()
    print("Analysis of trade-offs:")
    print()
    print("entropy_coef = 0.0 (no regularization):")
    print("  The policy may converge quickly to a single action,")
    print("  potentially getting stuck if it commits to a suboptimal one early.")
    print()
    print("entropy_coef = 0.01 (small):")
    print("  Slight exploration encouragement. Good balance -- the agent")
    print("  explores enough to find the best action but still converges.")
    print()
    print("entropy_coef = 0.1 (moderate):")
    print("  More exploration, slower convergence. May maintain diversity")
    print("  longer than needed, but more robust to local optima.")
    print()
    print("entropy_coef = 0.5 (large):")
    print("  TOO MUCH entropy regularization! The entropy term dominates")
    print("  the reward signal, forcing the policy to stay nearly uniform.")
    print("  The agent barely learns from rewards and performs poorly.")


if __name__ == "__main__":
    print("=== Exercise 1: Log-Derivative Trick ===")
    exercise_1()

    print("\n=== Exercise 2: Vanilla REINFORCE ===")
    exercise_2()

    print("\n=== Exercise 3: Baseline Effect on Variance ===")
    exercise_3()

    print("\n=== Exercise 4: Gaussian Policy ===")
    exercise_4()

    print("\n=== Exercise 5: Entropy Regularization ===")
    exercise_5()

    print("\nAll exercises completed!")
