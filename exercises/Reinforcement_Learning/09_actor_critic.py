"""
Exercises for Lesson 09: Actor-Critic
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: TD Error as Advantage Estimate

    Show that the 1-step TD error delta = r + gamma*V(s') - V(s)
    is an unbiased estimate of the advantage A(s,a) = Q(s,a) - V(s).
    """
    print("TD Error as Advantage Estimate:")
    print("=" * 60)

    print()
    print("Derivation:")
    print("  A(s, a) = Q(s, a) - V(s)")
    print()
    print("  Q(s, a) = E[r + gamma * V(s') | s, a]")
    print("            (by Bellman equation under current policy)")
    print()
    print("  Therefore:")
    print("  A(s, a) = E[r + gamma * V(s') | s, a] - V(s)")
    print("          = E[r + gamma * V(s') - V(s) | s, a]")
    print("          = E[delta | s, a]")
    print()
    print("  The 1-step TD error delta = r + gamma*V(s') - V(s)")
    print("  is a STOCHASTIC estimate of the advantage.")
    print("  Its expectation under a given (s, a) equals A(s, a).")
    print()

    # Numerical verification
    np.random.seed(0)

    # Known environment: state 0, action 0 leads to reward ~ N(2, 1) then state 1
    # V(0) = 3.0 (known), V(1) = 1.0 (known), gamma = 0.9
    V = {0: 3.0, 1: 1.0}
    gamma = 0.9
    true_Q_s0_a0 = 2.0 + gamma * V[1]  # E[r] + gamma * V(s')
    true_A = true_Q_s0_a0 - V[0]

    n_samples = 10000
    td_errors = []
    for _ in range(n_samples):
        r = np.random.normal(2.0, 1.0)  # stochastic reward
        delta = r + gamma * V[1] - V[0]
        td_errors.append(delta)

    mean_td = np.mean(td_errors)
    print(f"  True A(s=0, a=0) = {true_A:.4f}")
    print(f"  E[TD error] (empirical, {n_samples} samples) = {mean_td:.4f}")
    print(f"  Difference: {abs(true_A - mean_td):.6f}")
    print()
    print("  => TD error is an UNBIASED but HIGH-VARIANCE estimate of A(s,a).")
    print("     This is why the critic (V estimate) is crucial — it reduces variance.")


def exercise_2():
    """
    Exercise 2: Actor-Critic vs REINFORCE

    Implement both on a simple bandit task and compare variance.
    """
    print("\nActor-Critic vs REINFORCE — Variance Comparison:")
    print("=" * 60)

    np.random.seed(42)

    # Simple k-armed bandit: k actions, rewards ~ N(mu_a, 1)
    k = 5
    true_means = np.array([1.0, 2.0, 5.0, 3.0, 1.5])

    def softmax(logits):
        logits = logits - np.max(logits)
        e = np.exp(logits)
        return e / e.sum()

    n_steps = 2000
    lr_actor = 0.05
    lr_critic = 0.1

    # --- REINFORCE (no baseline) ---
    logits_r = np.zeros(k)
    gradient_norms_r = []

    for step in range(n_steps):
        pi = softmax(logits_r)
        action = np.random.choice(k, p=pi)
        reward = np.random.normal(true_means[action], 1.0)

        grad_log = np.zeros(k)
        grad_log[action] = 1.0
        grad_log -= pi
        grad = grad_log * reward
        gradient_norms_r.append(np.linalg.norm(grad))
        logits_r += lr_actor * grad

    # --- Actor-Critic (with critic as baseline) ---
    logits_ac = np.zeros(k)
    V = np.zeros(k)  # per-action critic (simple)
    V_mean = 0.0     # state value (baseline) for bandit
    gradient_norms_ac = []

    for step in range(n_steps):
        pi = softmax(logits_ac)
        action = np.random.choice(k, p=pi)
        reward = np.random.normal(true_means[action], 1.0)

        # Critic update
        V_mean += lr_critic * (reward - V_mean)

        # Actor update with advantage = r - V(s)
        advantage = reward - V_mean
        grad_log = np.zeros(k)
        grad_log[action] = 1.0
        grad_log -= pi
        grad = grad_log * advantage
        gradient_norms_ac.append(np.linalg.norm(grad))
        logits_ac += lr_actor * grad

    print(f"\n  Gradient norm statistics (proxy for variance):")
    print(f"  {'Method':>18} | {'Mean':>8} | {'Std':>8} | {'Max':>8}")
    print("  " + "-" * 50)
    print(f"  {'REINFORCE':>18} | {np.mean(gradient_norms_r):>8.4f} | "
          f"{np.std(gradient_norms_r):>8.4f} | {np.max(gradient_norms_r):>8.4f}")
    print(f"  {'Actor-Critic':>18} | {np.mean(gradient_norms_ac):>8.4f} | "
          f"{np.std(gradient_norms_ac):>8.4f} | {np.max(gradient_norms_ac):>8.4f}")

    var_reduction = np.std(gradient_norms_r) / np.std(gradient_norms_ac)
    print(f"\n  Variance reduction factor (std): {var_reduction:.2f}x")
    print(f"\n  Final learned probabilities:")
    pi_r = softmax(logits_r)
    pi_ac = softmax(logits_ac)
    print(f"  REINFORCE:    {pi_r.round(3)}")
    print(f"  Actor-Critic: {pi_ac.round(3)}")
    print(f"  Optimal (action 2): maximize probability on index 2 (mean={true_means[2]})")


def exercise_3():
    """
    Exercise 3: Bootstrapping Depth

    Compare n-step returns for bias-variance trade-off.
    n=1 (TD): low variance, high bias
    n=inf (MC): zero bias, high variance
    """
    print("\nN-Step Returns — Bias-Variance Trade-off:")
    print("=" * 60)

    np.random.seed(5)

    # Simple environment: deterministic rewards except noise at terminal
    def run_episode(max_steps=20):
        """Returns a trajectory of (reward, V_next) pairs."""
        # Simulate: state value decreases toward goal
        rewards = []
        values = []
        for t in range(max_steps):
            r = np.random.normal(0.5, 1.0)  # stochastic reward
            v_next = max(0, (max_steps - t - 1) * 0.5)  # crude V estimate
            rewards.append(r)
            values.append(v_next)
            if t == max_steps - 1:  # terminal
                break
        return rewards, values

    def n_step_return(rewards, values, n, gamma=0.99):
        """Compute n-step return for first state in trajectory."""
        G = sum(gamma**i * r for i, r in enumerate(rewards[:n]))
        if n < len(rewards):
            G += gamma**n * values[n]
        return G

    n_trials = 5000
    gamma = 0.99

    # True return: average of Monte Carlo estimates
    true_returns = []
    for _ in range(n_trials):
        r, v = run_episode()
        true_returns.append(sum(gamma**i * ri for i, ri in enumerate(r)))
    true_return = np.mean(true_returns)

    print(f"\n  True expected return (MC estimate): {true_return:.4f}")
    print(f"\n  {'n':>4} | {'Mean estimate':>14} | {'Bias':>8} | {'Std':>8}")
    print("  " + "-" * 44)

    for n in [1, 2, 5, 10, 20]:
        estimates = []
        for _ in range(n_trials):
            r, v = run_episode()
            estimates.append(n_step_return(r, v, n, gamma))
        mean_est = np.mean(estimates)
        bias = mean_est - true_return
        std = np.std(estimates)
        print(f"  {n:>4} | {mean_est:>14.4f} | {bias:>8.4f} | {std:>8.4f}")

    print()
    print("  n=1 (TD):  High bias (V-function is imperfect), low variance.")
    print("  n=inf (MC): Zero bias, high variance (full trajectory noise).")
    print("  Best n:    Problem-dependent; typically 3-20 in practice (GAE uses lambda).")


def exercise_4():
    """
    Exercise 4: Generalized Advantage Estimation (GAE)

    Show that GAE interpolates between TD(0) and MC with lambda.
    """
    print("\nGeneralized Advantage Estimation (GAE):")
    print("=" * 60)

    print()
    print("GAE formula:")
    print("  A_t^GAE(lambda) = sum_{k=0}^{inf} (gamma*lambda)^k * delta_{t+k}")
    print("  where delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)")
    print()
    print("  lambda=0: A = delta_t = r_t + gamma*V(s') - V(s)  [1-step TD]")
    print("  lambda=1: A = sum_k gamma^k * r_{t+k} - V(s_t)    [MC advantage]")
    print()

    np.random.seed(3)
    n_steps = 15  # trajectory length
    gamma = 0.99

    # Known trajectory
    rewards = np.random.normal(1.0, 0.5, n_steps)
    # V(s_t) decreases toward terminal
    V = np.array([(n_steps - t) * 0.5 for t in range(n_steps + 1)])

    def compute_gae(rewards, V, gamma, lam):
        """Compute GAE advantages for a trajectory."""
        td_errors = rewards + gamma * V[1:] - V[:-1]
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = td_errors[t] + gamma * lam * gae
            advantages[t] = gae
        return advantages

    # Compute GAE for different lambda values
    print(f"  Advantages for first 5 steps (trajectory length={n_steps}):")
    print(f"  {'Step':>5}", end="")
    lambdas = [0.0, 0.5, 0.9, 1.0]
    for lam in lambdas:
        print(f" | lambda={lam}", end="")
    print()
    print("  " + "-" * 60)

    all_advantages = {lam: compute_gae(rewards, V, gamma, lam) for lam in lambdas}

    for t in range(5):
        print(f"  {t:>5}", end="")
        for lam in lambdas:
            print(f" | {all_advantages[lam][t]:>10.4f}", end="")
        print()

    # Variance comparison
    print(f"\n  Variance across trajectory (proxy for estimation noise):")
    for lam in lambdas:
        var = np.var(all_advantages[lam])
        print(f"  lambda={lam}: variance = {var:.6f}")


if __name__ == "__main__":
    print("=== Exercise 1: TD Error as Advantage ===")
    exercise_1()

    print("\n=== Exercise 2: Actor-Critic vs REINFORCE ===")
    exercise_2()

    print("\n=== Exercise 3: N-Step Returns ===")
    exercise_3()

    print("\n=== Exercise 4: GAE ===")
    exercise_4()

    print("\nAll exercises completed!")
