"""
Exercises for Lesson 19: Offline Reinforcement Learning
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


def exercise_1():
    """
    Exercise 1: Distribution Shift — Why Offline RL Is Hard

    Demonstrate that a policy can visit states outside the dataset,
    leading to unreliable Q-value estimates.
    """
    print("Distribution Shift in Offline RL:")
    print("=" * 60)

    np.random.seed(42)

    # Dataset: collected by a behavior policy that only visits states 0-5
    n_states = 10
    behavior_policy_states = list(range(6))  # only visits 0-5

    dataset = []
    for _ in range(1000):
        s = np.random.choice(behavior_policy_states)
        a = np.random.randint(2)
        ns = min(n_states - 1, s + (1 if a == 1 else 0))
        r = 1.0 if ns == n_states - 1 else 0.0
        dataset.append((s, a, r, ns))

    # Offline Q-learning (ignoring distribution shift)
    Q = np.zeros((n_states, 2))
    for _ in range(50):
        for s, a, r, ns in dataset:
            Q[s, a] += 0.1 * (r + 0.99 * np.max(Q[ns]) - Q[s, a])

    # Extract policy
    policy = [int(np.argmax(Q[s])) for s in range(n_states)]

    print(f"\n  Dataset only covers states: {behavior_policy_states}")
    print(f"\n  Learned Q-values for ALL states (including unvisited 6-9):")
    print(f"  {'State':>6} | {'Q(s,left)':>10} | {'Q(s,right)':>10} | {'Policy':>8} | {'In Dataset':>10}")
    print("  " + "-" * 56)
    for s in range(n_states):
        in_data = s in behavior_policy_states
        print(f"  {s:>6} | {Q[s,0]:>10.4f} | {Q[s,1]:>10.4f} | "
              f"{'right' if policy[s] else 'left':>8} | {str(in_data):>10}")

    print(f"\n  States 6-9 were NEVER in the dataset, yet have Q-values.")
    print(f"  These values are EXTRAPOLATED (unreliable).")
    print(f"  A policy using these values may take catastrophically wrong actions.")


def exercise_2():
    """
    Exercise 2: Conservative Q-Learning (CQL) Penalty

    Show that CQL's conservative penalty pulls down Q-values for
    out-of-distribution actions.
    """
    print("\nConservative Q-Learning (CQL) Penalty:")
    print("=" * 60)

    np.random.seed(7)
    n_states = 5
    n_actions = 3

    # Dataset only has action=1 for all states
    dataset = [(s, 1, np.random.normal(1.0, 0.1), min(n_states-1, s+1))
               for s in range(n_states) for _ in range(100)]

    def train(alpha_cql, n_epochs=30):
        Q = np.zeros((n_states, n_actions))
        for _ in range(n_epochs):
            np.random.shuffle(dataset)
            for s, a, r, ns in dataset:
                # Standard TD
                td_target = r + 0.99 * np.max(Q[ns])
                Q[s, a] += 0.1 * (td_target - Q[s, a])

                if alpha_cql > 0:
                    # CQL: penalize logsumexp, add back behavior Q
                    log_sum = np.log(np.sum(np.exp(Q[s] - np.max(Q[s]))) + 1e-10) + np.max(Q[s])
                    for a2 in range(n_actions):
                        Q[s, a2] -= 0.1 * alpha_cql * np.exp(Q[s, a2] - log_sum)
                    Q[s, a] += 0.1 * alpha_cql * (1.0 / n_actions)
        return Q

    print(f"\n  Dataset: only action=1 is observed (actions 0, 2 are OOD)")
    print(f"\n  Q-values for state s=2:")
    print(f"  {'alpha_cql':>10} | {'Q(s,a=0)':>10} | {'Q(s,a=1)':>10} | {'Q(s,a=2)':>10}")
    print("  " + "-" * 48)

    for alpha in [0.0, 0.5, 1.0, 2.0]:
        Q = train(alpha)
        print(f"  {alpha:>10.1f} | {Q[2,0]:>10.4f} | {Q[2,1]:>10.4f} | {Q[2,2]:>10.4f}")

    print(f"\n  With alpha_cql=0: OOD actions (0, 2) have similar Q to action 1.")
    print(f"  With large alpha_cql: OOD Q-values are pushed DOWN (conservative).")
    print(f"  => CQL penalizes actions not in the dataset, preventing extrapolation.")


def exercise_3():
    """
    Exercise 3: Behavior Cloning as Offline RL Baseline

    Compare BC vs offline Q-learning on policy quality.
    """
    print("\nBehavior Cloning vs Offline Q-Learning:")
    print("=" * 60)

    np.random.seed(0)

    # Simple chain: 0 -> 1 -> 2 -> 3 (goal)
    # Behavior policy: mostly right but sometimes left (epsilon=0.3)
    n_states = 4
    n_actions = 2  # 0=left, 1=right

    def collect_data(epsilon, n_episodes=200):
        data = []
        for _ in range(n_episodes):
            s = 0
            for _ in range(10):
                if np.random.random() < epsilon:
                    a = np.random.randint(2)
                else:
                    a = 1  # behavior: go right
                ns = min(n_states-1, s+1) if a == 1 else max(0, s-1)
                r = 10.0 if ns == n_states-1 else -0.1
                data.append((s, a, r, ns, ns == n_states-1))
                s = ns
                if s == n_states-1:
                    break
        return data

    def train_bc(data):
        counts = np.zeros((n_states, n_actions))
        for s, a, r, ns, d in data:
            counts[s, a] += 1
        return np.array([int(np.argmax(counts[s])) if counts[s].sum() > 0 else 1
                         for s in range(n_states)])

    def train_offline_ql(data, n_epochs=50):
        Q = np.zeros((n_states, n_actions))
        for _ in range(n_epochs):
            for s, a, r, ns, done in data:
                Q[s, a] += 0.1 * ((r if done else r + 0.99 * np.max(Q[ns])) - Q[s, a])
        return np.array([int(np.argmax(Q[s])) for s in range(n_states)])

    def evaluate(policy, n_trials=500):
        successes = 0
        for _ in range(n_trials):
            s = 0
            for _ in range(10):
                a = policy[s]
                s = min(n_states-1, s+1) if a == 1 else max(0, s-1)
                if s == n_states-1:
                    successes += 1
                    break
        return successes / n_trials

    for eps in [0.1, 0.3, 0.6]:
        data = collect_data(epsilon=eps)
        bc_pol = train_bc(data)
        ql_pol = train_offline_ql(data)
        bc_rate = evaluate(bc_pol)
        ql_rate = evaluate(ql_pol)
        print(f"\n  Behavior epsilon={eps}: BC={bc_rate:.1%}, Offline QL={ql_rate:.1%}")
        print(f"    BC policy:  {bc_pol}")
        print(f"    QL policy:  {ql_pol}")

    print(f"\n  BC ceiling = behavior policy quality.")
    print(f"  Offline QL can sometimes exceed BC when the dataset is informative.")


def exercise_4():
    """
    Exercise 4: Decision Transformer Concept

    Show how RL as sequence modeling (return-conditioned BC) works.
    """
    print("\nDecision Transformer — Return-Conditioned Behavior Cloning:")
    print("=" * 60)

    print()
    print("  Decision Transformer key idea:")
    print("  Instead of learning Q-values, treat RL as sequence modeling:")
    print()
    print("  Input:  (R_to_go_1, s_1, a_1, R_to_go_2, s_2, a_2, ...)")
    print("  Output: predict a_t given (R_to_go_t, s_t, context)")
    print()
    print("  At test time: set R_to_go = desired_return")
    print("  The model selects actions that historically led to that return.")
    print()

    np.random.seed(1)
    n_states = 5
    n_actions = 2

    # Collect dataset with different quality trajectories
    def run_trajectory(policy_quality):
        """quality=1.0: optimal, 0.0: random"""
        traj = []
        s = 0
        total_r = 0.0
        for _ in range(20):
            if np.random.random() < policy_quality:
                a = 1  # go right (optimal)
            else:
                a = np.random.randint(2)
            ns = min(n_states-1, s+1) if a == 1 else max(0, s-1)
            r = 10.0 if ns == n_states-1 else -0.1
            traj.append((s, a, r))
            total_r += r
            s = ns
            if s == n_states-1:
                break
        return traj, total_r

    # Collect trajectories of varying quality
    all_trajs = []
    for quality in [0.0, 0.5, 1.0]:
        for _ in range(100):
            traj, ret = run_trajectory(quality)
            all_trajs.append((traj, ret))

    # Return-conditioned action selection: given (state, desired_return_to_go),
    # find the most common action in the dataset for similar (s, R_to_go) pairs
    def return_conditioned_action(state, desired_rtg, dataset, rtg_tolerance=2.0):
        action_votes = np.zeros(n_actions)
        for traj, ret in dataset:
            # Compute returns-to-go along the trajectory
            for i, (s, a, r) in enumerate(traj):
                rtg = sum(rr for _, _, rr in traj[i:])
                if s == state and abs(rtg - desired_rtg) < rtg_tolerance:
                    action_votes[a] += 1
        if action_votes.sum() == 0:
            return 1  # default
        return int(np.argmax(action_votes))

    print("  Return-conditioned action selection from dataset:")
    print(f"  {'State':>6} | {'Desired RTG':>12} | {'Chosen action':>14}")
    print("  " + "-" * 40)
    for s in range(3):
        for desired in [1.0, 5.0, 9.0]:
            a = return_conditioned_action(s, desired, all_trajs, rtg_tolerance=2.0)
            action_name = 'right' if a == 1 else 'left'
            print(f"  {s:>6} | {desired:>12.1f} | {action_name:>14}")

    print()
    print("  With high desired RTG: model selects optimal (right) actions.")
    print("  This is the key insight of Decision Transformer:")
    print("  condition on HIGH desired return at test time -> optimal behavior.")


if __name__ == "__main__":
    print("=== Exercise 1: Distribution Shift ===")
    exercise_1()

    print("\n=== Exercise 2: CQL Penalty ===")
    exercise_2()

    print("\n=== Exercise 3: BC vs Offline QL ===")
    exercise_3()

    print("\n=== Exercise 4: Decision Transformer ===")
    exercise_4()

    print("\nAll exercises completed!")
