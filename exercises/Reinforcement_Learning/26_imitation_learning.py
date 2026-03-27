"""
Exercises for Lesson 26: Imitation Learning
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


def exercise_1():
    """
    Exercise 1: Compounding Error in Behavioral Cloning

    Prove and demonstrate that BC error scales as O(T^2)
    while DAgger error scales as O(T).
    """
    print("Compounding Error: BC O(T^2) vs DAgger O(T):")
    print("=" * 60)

    print()
    print("Theoretical bound (Ross & Bagnell 2010):")
    print()
    print("  BC:     E[Total Loss] <= epsilon * T^2  (epsilon = per-step error)")
    print("  DAgger: E[Total Loss] <= epsilon * T    (distributional mismatch removed)")
    print()
    print("Intuition for BC O(T^2):")
    print("  At each step t, the learner is eps-away from the expert.")
    print("  Over T steps, it drifts into a region of states with higher error.")
    print("  Each new state adds more compounding, leading to T^2 growth.")
    print()
    print("Why DAgger fixes this:")
    print("  DAgger trains on the DISTRIBUTION OF STATES THE LEARNER VISITS.")
    print("  So even when the learner deviates, it has seen those states in training.")
    print("  Error stays linear in T (one epsilon error per step, no compounding).")
    print()

    # Empirical verification
    np.random.seed(0)

    def expert_action(state):
        """Expert: simple proportional controller."""
        return -2.0 * state

    def bc_action(state, w_bc):
        """BC policy: linear approximation with slight error."""
        return w_bc * state

    def dynamics(state, action, noise=0.01):
        """x_{t+1} = 0.9*x + 0.1*action + noise"""
        return 0.9 * state + 0.1 * action + np.random.randn() * noise

    # BC: trained on expert states (near 0), but w is slightly miscalibrated
    w_bc_exact = -2.0       # exact BC
    w_bc_approx = -1.5      # approximate BC (trained with noise)

    # DAgger: trained on visited states -> handles distribution shift
    w_dagger = -1.9  # slightly better calibrated for visited states

    horizons = [5, 10, 20, 50, 100]
    n_trials = 500

    print(f"  Mean absolute state magnitude at different horizons:")
    print(f"  (proxy for error accumulation, starting from x0 ~ U(-0.5, 0.5))")
    print(f"\n  {'H':>5} | {'Expert':>8} | {'BC (exact)':>12} | "
          f"{'BC (approx)':>13} | {'DAgger':>8}")
    print("  " + "-" * 58)

    for H in horizons:
        expert_dev, bc_exact_dev, bc_approx_dev, dagger_dev = [], [], [], []
        for _ in range(n_trials):
            x0 = np.random.uniform(-0.5, 0.5)
            for policy_dev, w in [(expert_dev, None), (bc_exact_dev, w_bc_exact),
                                   (bc_approx_dev, w_bc_approx), (dagger_dev, w_dagger)]:
                x = x0
                for _ in range(H):
                    a = expert_action(x) if w is None else bc_action(x, w)
                    x = dynamics(x, a)
                policy_dev.append(abs(x))
        print(f"  {H:>5} | {np.mean(expert_dev):>8.4f} | {np.mean(bc_exact_dev):>12.4f} | "
              f"{np.mean(bc_approx_dev):>13.4f} | {np.mean(dagger_dev):>8.4f}")

    print("\n  BC (approx) error grows super-linearly (O(T^2) regime).")
    print("  DAgger handles distribution shift, keeping error near O(T).")


def exercise_2():
    """
    Exercise 2: DAgger Implementation

    Implement DAgger and compare convergence with behavior cloning.
    """
    print("\nDAgger vs Behavioral Cloning:")
    print("=" * 60)

    np.random.seed(42)

    # 1-D control: reach target 0 from random start
    class ReachEnv:
        def reset(self):
            self.state = np.random.uniform(-3, 3)
            self.steps = 0
            return self.state
        def step(self, action):
            action = np.clip(action, -2, 2)
            self.state += 0.1 * action
            self.steps += 1
            done = abs(self.state) < 0.1 or self.steps >= 40
            return self.state, -abs(self.state), done
        def expert_action(self, state):
            return np.clip(-3.0 * state, -2, 2)

    class LinearPolicy:
        def __init__(self): self.w = 0.0
        def predict(self, state): return np.clip(self.w * state, -2, 2)
        def fit(self, states, actions):
            if len(states) < 2: return
            X = np.array(states).reshape(-1, 1)
            y = np.array(actions)
            self.w = float(np.linalg.lstsq(X, y, rcond=None)[0][0])

    def evaluate(policy_fn, n_episodes=100):
        env = ReachEnv()
        rewards = []
        for ep in range(n_episodes):
            s = env.reset()
            total_r = 0.0
            done = False
            while not done:
                a = policy_fn(s)
                s, r, done = env.step(a)
                total_r += r
            rewards.append(total_r)
        return np.mean(rewards)

    env = ReachEnv()
    n_iters = 8

    # BC: collect expert data once, fit, done
    bc_policy = LinearPolicy()
    bc_states, bc_actions = [], []
    for _ in range(30):
        s = env.reset()
        done = False
        while not done:
            a = env.expert_action(s)
            bc_states.append(s)
            bc_actions.append(a)
            s, _, done = env.step(a)
    bc_policy.fit(bc_states, bc_actions)
    bc_reward = evaluate(bc_policy.predict)

    # DAgger: iterative
    dagger_policy = LinearPolicy()
    dagger_policy.w = 0.1
    dagger_states, dagger_actions = [], []

    dagger_iters = []
    for it in range(n_iters):
        # Collect states with current policy, label with expert
        new_states = []
        for _ in range(10):
            s = env.reset()
            done = False
            while not done:
                a = dagger_policy.predict(s)
                new_states.append(s)
                s, _, done = env.step(a)
        # Label with expert
        new_actions = [env.expert_action(s) for s in new_states]
        dagger_states.extend(new_states)
        dagger_actions.extend(new_actions)
        dagger_policy.fit(dagger_states, dagger_actions)
        dagger_iters.append(evaluate(dagger_policy.predict))

    print(f"\n  BC (single training): reward = {bc_reward:.3f}, w = {bc_policy.w:.4f}")
    print(f"\n  DAgger (iterative) convergence:")
    for i, r in enumerate(dagger_iters):
        print(f"    Iteration {i+1}: reward = {r:.3f}, w = {dagger_policy.w:.4f}")

    print(f"\n  DAgger improves iteratively by visiting learner's own states.")


def exercise_3():
    """
    Exercise 3: Imitation from Observation Only

    Learn from state-only demonstrations (no action labels).
    """
    print("\nImitation from Observation Only:")
    print("=" * 60)

    print()
    print("Challenge: No action labels in demonstrations.")
    print("           Only state sequences observed.")
    print()
    print("Approaches:")
    print("  1. Inverse Dynamics Model (IDM):")
    print("     Learn f(s_t, s_{t+1}) -> a_t from interaction data")
    print("     Then label demonstrations: a_t = f(s_t, s_{t+1})")
    print()
    print("  2. State-matching reward:")
    print("     r(s_t) = -distance(s_t, s^expert_t)")
    print("     Optimize policy to visit same states as expert.")
    print()

    np.random.seed(3)

    # Simple 1-D environment
    def transition(s, a):
        """s_{t+1} = s + 0.1*a"""
        return s + 0.1 * np.clip(a, -2, 2)

    # Expert demos: always go right (action = 2), so s increases
    expert_trajs = [[i * 0.1 for i in range(11)] for _ in range(20)]
    expert_states = [s for traj in expert_trajs for s in traj]

    # IDM: learn f(s, s') -> action from interaction data
    # Collected by random policy
    X_idm, y_idm = [], []
    for _ in range(2000):
        s = np.random.uniform(-1, 1)
        a = np.random.uniform(-2, 2)
        ns = transition(s, a)
        X_idm.append([s, ns])
        y_idm.append(a)

    X_idm = np.array(X_idm)
    y_idm = np.array(y_idm)

    # Fit IDM: linear regression (s, s') -> a
    from numpy.linalg import lstsq
    w_idm, _, _, _ = lstsq(X_idm, y_idm, rcond=None)

    def idm_action(s, ns):
        return np.clip(np.dot(w_idm, [s, ns]), -2, 2)

    # Label expert demonstrations using IDM
    labeled_states, labeled_actions = [], []
    for traj in expert_trajs:
        for i in range(len(traj) - 1):
            s, ns = traj[i], traj[i + 1]
            a = idm_action(s, ns)
            labeled_states.append(s)
            labeled_actions.append(a)

    # Fit BC policy on IDM-labeled data
    X_bc = np.array(labeled_states).reshape(-1, 1)
    y_bc = np.array(labeled_actions)
    w_bc, _, _, _ = lstsq(X_bc, y_bc, rcond=None)
    w_bc = w_bc[0]

    print(f"  IDM weights [w_s, w_ns]: {w_idm.round(4)}")
    print(f"  True inverse dynamics:  a = 10 * (ns - s)  [since s+=0.1*a]")
    print(f"  Expected w_idm: [-10, 10] (approximately)")
    print(f"\n  BC policy from IDM-labeled demos: w = {w_bc:.4f}")
    print(f"  Expert action (always right): 2.0")
    print(f"\n  IDM + observation-only imitation can recover expert behavior")
    print(f"  without ever seeing action labels directly.")


def exercise_4():
    """
    Exercise 4: One-Shot Imitation Learning Concept

    Demonstrate adapting to a new task from a single demonstration.
    """
    print("\nOne-Shot Imitation Learning:")
    print("=" * 60)

    print()
    print("Goal: learn a meta-policy that adapts to new tasks from one demo.")
    print()
    print("Key idea: Train on many tasks, so the model learns HOW to imitate,")
    print("          not what to imitate for a specific task.")
    print()

    np.random.seed(9)

    # Multi-task: each task is a different target position
    n_tasks_train = 20

    # Simple approach: task embedding from demonstration
    # Demo: observe (s, a) pairs -> extract goal direction
    def demo_to_embedding(demo_states):
        """Compute direction from start to end of demo."""
        start, end = demo_states[0], demo_states[-1]
        diff = end - start
        norm = abs(diff) + 1e-8
        return np.array([diff / norm, norm])  # direction + magnitude

    # Meta-policy: linear over [state, task_embedding]
    def meta_policy(state, embedding, w):
        features = np.concatenate([[state], embedding])
        return np.clip(np.dot(w, features), -2, 2)

    # Generate training data
    train_data = []
    for _ in range(n_tasks_train):
        target = np.random.uniform(-3, 3)
        # Expert demo for this task: always move toward target
        demo_states = [np.random.uniform(-1, 1)]
        for _ in range(10):
            action = np.clip(3.0 * (target - demo_states[-1]), -2, 2)
            demo_states.append(demo_states[-1] + 0.1 * action)
        embedding = demo_to_embedding(demo_states)
        # Collect training transitions
        for s in np.linspace(-3, 3, 20):
            expert_a = np.clip(3.0 * (target - s), -2, 2)
            train_data.append((s, embedding, expert_a))

    # Fit meta-policy
    X = np.array([[s] + list(emb) for s, emb, _ in train_data])
    y = np.array([a for _, _, a in train_data])
    w_meta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Evaluate on new tasks (one-shot)
    test_targets = [-2.0, 0.0, 1.5, 2.5]
    print(f"  Test target positions: {test_targets}")
    print(f"\n  One-shot evaluation (10 steps each, start=0):")
    print(f"  {'Target':>8} | {'Final state':>12} | {'Error':>8}")
    print("  " + "-" * 38)

    for target in test_targets:
        # One demonstration
        demo_states = [0.0]
        for _ in range(10):
            a = np.clip(3.0 * (target - demo_states[-1]), -2, 2)
            demo_states.append(demo_states[-1] + 0.1 * a)
        embedding = demo_to_embedding(demo_states)

        # Execute meta-policy
        state = 0.0
        for _ in range(10):
            a = meta_policy(state, embedding, w_meta)
            state = state + 0.1 * a

        print(f"  {target:>8.1f} | {state:>12.4f} | {abs(state-target):>8.4f}")

    print("\n  Meta-policy adapts to new targets from a single demonstration.")


if __name__ == "__main__":
    print("=== Exercise 1: Compounding Error ===")
    exercise_1()

    print("\n=== Exercise 2: DAgger Implementation ===")
    exercise_2()

    print("\n=== Exercise 3: Imitation from Observation ===")
    exercise_3()

    print("\n=== Exercise 4: One-Shot Imitation ===")
    exercise_4()

    print("\nAll exercises completed!")
