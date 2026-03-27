"""
Exercises for Lesson 25: World Models
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


def exercise_1():
    """
    Exercise 1: World Model vs Model-Free — Sample Efficiency

    Show that using a world model requires fewer real environment
    interactions to achieve the same performance.
    """
    print("World Model Sample Efficiency:")
    print("=" * 60)

    np.random.seed(42)

    # Simple grid: 4x4, action moves right/down toward (3,3)
    n_states, n_actions = 16, 4
    ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]
    SIZE = 4

    def encode(r, c): return r * SIZE + c
    def decode(s): return divmod(s, SIZE)

    def env_step(s, a):
        r, c = decode(s)
        dr, dc = ACTIONS[a]
        nr, nc = max(0, min(SIZE-1, r+dr)), max(0, min(SIZE-1, c+dc))
        ns = encode(nr, nc)
        done = ns == n_states - 1
        return ns, (1.0 if done else -0.01), done

    # Model-free Q-learning
    def train_model_free(n_real_steps, seed):
        np.random.seed(seed)
        Q = np.zeros((n_states, n_actions))
        state = 0
        for _ in range(n_real_steps):
            a = np.random.randint(n_actions) if np.random.random() < 0.3 else int(np.argmax(Q[state]))
            ns, r, done = env_step(state, a)
            Q[state, a] += 0.1 * (r + 0.99 * np.max(Q[ns]) * (not done) - Q[state, a])
            state = ns if not done else 0
        return Q

    # Dyna-Q: model-based with imaginary rollouts
    def train_dyna(n_real_steps, n_imaginary, seed):
        np.random.seed(seed)
        Q = np.zeros((n_states, n_actions))
        model = defaultdict(lambda: defaultdict(int))
        R_model = defaultdict(float)
        state = 0
        for _ in range(n_real_steps):
            a = np.random.randint(n_actions) if np.random.random() < 0.3 else int(np.argmax(Q[state]))
            ns, r, done = env_step(state, a)
            Q[state, a] += 0.1 * (r + 0.99 * np.max(Q[ns]) * (not done) - Q[state, a])
            model[(state, a)][ns] += 1
            R_model[(state, a)] += 0.1 * (r - R_model[(state, a)])
            state = ns if not done else 0
            for _ in range(n_imaginary):
                if not model: break
                sa = list(model.keys())[np.random.randint(len(model))]
                s_p, a_p = sa
                ns_counts = model[sa]
                total = sum(ns_counts.values())
                ns_list = list(ns_counts.keys())
                ns_p = ns_list[np.random.choice(len(ns_list), p=[v/total for v in ns_counts.values()])]
                r_p = R_model[sa]
                done_p = (ns_p == n_states - 1)
                Q[s_p, a_p] += 0.1 * (r_p + 0.99 * np.max(Q[ns_p]) * (not done_p) - Q[s_p, a_p])
        return Q

    def eval_policy(Q, n_eval=500):
        successes = 0
        for _ in range(n_eval):
            s = np.random.randint(n_states // 2)
            for _ in range(20):
                a = int(np.argmax(Q[s]))
                s, _, done = env_step(s, a)
                if done:
                    successes += 1
                    break
        return successes / n_eval

    print(f"\n  {'Real steps':>12} | {'Model-Free':>12} | {'Dyna (k=10)':>14}")
    print("  " + "-" * 44)
    for n_real in [50, 200, 500, 1000]:
        rates_mf = [eval_policy(train_model_free(n_real, s)) for s in range(5)]
        rates_dy = [eval_policy(train_dyna(n_real, 10, s)) for s in range(5)]
        print(f"  {n_real:>12} | {np.mean(rates_mf):>10.1%}±{np.std(rates_mf):.2f} | "
              f"{np.mean(rates_dy):>12.1%}±{np.std(rates_dy):.2f}")

    print("\n  Dyna achieves higher performance at same # real steps.")
    print("  Model provides 'free' imaginary training data.")


def exercise_2():
    """
    Exercise 2: Model Predictive Control

    Implement MPC and show planning horizon vs. performance trade-off.
    """
    print("\nModel Predictive Control (MPC):")
    print("=" * 60)

    np.random.seed(7)

    # 1-D balance task: state = angle, action = torque
    # Dynamics: a_next = a + 0.1*torque - 0.05*sign(a) [simplified]
    def true_dynamics(angle, torque):
        return angle + 0.1 * torque - 0.05 * np.sign(angle)

    def true_reward(angle, torque):
        return -(angle**2) - 0.01 * torque**2

    # Learned model (slightly inaccurate)
    def learned_dynamics(angle, torque):
        return angle + 0.09 * torque - 0.04 * np.sign(angle)  # slightly off

    def mpc(angle, horizon, n_candidates=100):
        best_return = -1e10
        best_torque = 0.0
        for _ in range(n_candidates):
            torques = np.random.uniform(-2, 2, horizon)
            a = angle
            total_r = 0.0
            for t, tau in enumerate(torques):
                total_r += (0.99**t) * true_reward(a, tau)
                a = learned_dynamics(a, tau)
            if total_r > best_return:
                best_return = total_r
                best_torque = torques[0]
        return best_torque

    def evaluate_mpc(horizon, n_episodes=100):
        total_rewards = []
        for _ in range(n_episodes):
            angle = np.random.uniform(-1.0, 1.0)
            ep_r = 0.0
            for step in range(30):
                torque = mpc(angle, horizon)
                ep_r += (0.99**step) * true_reward(angle, torque)
                angle = true_dynamics(angle, torque)
            total_rewards.append(ep_r)
        return np.mean(total_rewards), np.std(total_rewards)

    def random_policy_eval(n_episodes=100):
        total_rewards = []
        for _ in range(n_episodes):
            angle = np.random.uniform(-1.0, 1.0)
            ep_r = 0.0
            for step in range(30):
                torque = np.random.uniform(-2, 2)
                ep_r += (0.99**step) * true_reward(angle, torque)
                angle = true_dynamics(angle, torque)
            total_rewards.append(ep_r)
        return np.mean(total_rewards), np.std(total_rewards)

    r_rand, std_rand = random_policy_eval()
    print(f"\n  Random policy: {r_rand:.3f} ± {std_rand:.3f}")
    print(f"\n  {'Horizon':>8} | {'Mean Reward':>12} | {'Std':>8}")
    print("  " + "-" * 36)
    for H in [1, 3, 5, 10]:
        mean_r, std_r = evaluate_mpc(H)
        print(f"  {H:>8} | {mean_r:>12.3f} | {std_r:>8.3f}")

    print("\n  Longer horizon = better planning but more computation.")
    print("  Model inaccuracy limits the benefit of long horizons.")


def exercise_3():
    """
    Exercise 3: Latent Dynamics Model

    Show the benefit of a compact latent state representation
    for world model learning.
    """
    print("\nLatent Dynamics Model:")
    print("=" * 60)

    np.random.seed(1)

    # High-dim observation: 16-dim but only 2-dim latent matters
    obs_dim = 16
    latent_dim = 2

    # Encoder: project to latent (known true encoder for simulation)
    W_enc = np.zeros((latent_dim, obs_dim))
    W_enc[0, 0] = 1.0  # first latent = first feature
    W_enc[1, 1] = 1.0  # second latent = second feature

    def true_encode(obs):
        return W_enc @ obs  # true 2-D latent

    def generate_obs(latent):
        obs = np.zeros(obs_dim)
        obs[:latent_dim] = latent
        obs[latent_dim:] = np.random.randn(obs_dim - latent_dim) * 0.1  # noise dimensions
        return obs

    def latent_transition(latent, action):
        """Simple linear dynamics in latent space."""
        A = np.array([[0.9, 0.1], [-0.1, 0.9]])
        B = np.array([[0.5, 0.0], [0.0, 0.5]])
        return A @ latent + B @ action

    # Learn model in: (1) raw observation space, (2) latent space
    n_data = 2000
    actions = np.random.randn(n_data, 2)
    latents = [np.random.randn(latent_dim)]
    for a in actions:
        latents.append(latent_transition(latents[-1], a))
    latents = np.array(latents)
    obs_data = np.array([generate_obs(l) for l in latents])

    # Model in obs space: learn obs_dim x obs_dim transition
    # Model in latent space: learn latent_dim x latent_dim transition

    def fit_linear(X, Y):
        """Fit Y = W * X^T by least squares."""
        return Y.T @ X @ np.linalg.pinv(X.T @ X)

    X_obs = obs_data[:-1].T      # (obs_dim, N)
    Y_obs = obs_data[1:].T       # (obs_dim, N)
    X_lat = latents[:-1].T       # (latent_dim, N)
    Y_lat = latents[1:].T        # (latent_dim, N)

    W_obs = fit_linear(X_obs.T, Y_obs.T)
    W_lat = fit_linear(X_lat.T, Y_lat.T)

    # Prediction error
    pred_obs = W_obs @ X_obs
    pred_lat = W_lat @ X_lat
    err_obs = np.mean((pred_obs - Y_obs)**2)
    err_lat = np.mean((pred_lat - Y_lat)**2)

    print(f"\n  Observation dim: {obs_dim}, True latent dim: {latent_dim}")
    print(f"  Model parameters: obs-space={obs_dim**2}, latent-space={latent_dim**2}")
    print(f"\n  Prediction MSE:")
    print(f"    Obs-space model:    {err_obs:.6f}")
    print(f"    Latent-space model: {err_lat:.6f}")
    print(f"  Latent model is {err_obs/err_lat:.1f}x more accurate with "
          f"{obs_dim**2//latent_dim**2}x fewer parameters.")
    print("\n  Compact latent representations reduce model complexity and improve accuracy.")


def exercise_4():
    """
    Exercise 4: Model Error Propagation

    Show how model errors compound over long imaginary rollouts.
    """
    print("\nModel Error Propagation:")
    print("=" * 60)

    np.random.seed(5)

    # True 1-D dynamics: x_{t+1} = 0.9*x_t + action
    # Learned model: slight overestimation of decay (0.95 instead of 0.9)
    true_decay = 0.9
    learned_decay = 0.95  # model is slightly inaccurate

    n_trials = 1000

    print(f"\n  True dynamics: x_{{t+1}} = {true_decay}*x_t + action")
    print(f"  Learned model: x_{{t+1}} = {learned_decay}*x_t + action (small error)")
    print(f"\n  Mean absolute prediction error (MAE) at each horizon:")
    print(f"  {'Horizon':>8} | {'MAE':>10} | {'Error growth':>14}")
    print("  " + "-" * 40)

    prev_mae = None
    for H in [1, 2, 5, 10, 20]:
        errors = []
        for _ in range(n_trials):
            x0 = np.random.uniform(-2, 2)
            actions = np.random.uniform(-0.5, 0.5, H)

            # True trajectory
            x_true = x0
            for a in actions:
                x_true = true_decay * x_true + a

            # Model trajectory (with error)
            x_model = x0
            for a in actions:
                x_model = learned_decay * x_model + a

            errors.append(abs(x_true - x_model))

        mae = np.mean(errors)
        growth = mae / prev_mae if prev_mae else 1.0
        print(f"  {H:>8} | {mae:>10.5f} | {growth:>14.3f}x")
        prev_mae = mae

    print("\n  Prediction error grows with horizon (compounding model errors).")
    print("  This limits the effective planning horizon in practice.")
    print("  Short-horizon MPC with frequent replanning mitigates this issue.")


if __name__ == "__main__":
    print("=== Exercise 1: Sample Efficiency ===")
    exercise_1()

    print("\n=== Exercise 2: MPC Planning Horizon ===")
    exercise_2()

    print("\n=== Exercise 3: Latent Dynamics Model ===")
    exercise_3()

    print("\n=== Exercise 4: Model Error Propagation ===")
    exercise_4()

    print("\nAll exercises completed!")
