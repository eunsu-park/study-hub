"""
Exercises for Lesson 06: Q-Learning and SARSA
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


def exercise_1():
    """
    Exercise 1: Manual Q-Learning Update

    Given the following Q-table transition:
      State s=0, Action a=1, Reward r=5, Next state s'=2
      Q(0,1) = 3.0, Q(2,*) = [1.5, 4.0, 2.0]
      alpha=0.1, gamma=0.9

    Compute the Q-learning update step.
    """
    print("Q-Learning Update:")
    print("=" * 60)

    Q_s_a = 3.0
    reward = 5.0
    gamma = 0.9
    alpha = 0.1
    Q_ns = np.array([1.5, 4.0, 2.0])

    # Q-Learning target uses max over next-state actions (off-policy)
    max_Q_ns = np.max(Q_ns)
    td_target = reward + gamma * max_Q_ns
    td_error = td_target - Q_s_a
    new_Q = Q_s_a + alpha * td_error

    print(f"  Q(s=0, a=1) = {Q_s_a}")
    print(f"  R = {reward}, gamma = {gamma}, alpha = {alpha}")
    print(f"  Q(s'=2, *) = {Q_ns}")
    print(f"  max Q(s',*) = {max_Q_ns}")
    print()
    print(f"  TD Target  = R + gamma * max Q(s',*)")
    print(f"             = {reward} + {gamma} * {max_Q_ns} = {td_target}")
    print(f"  TD Error   = {td_target} - {Q_s_a} = {td_error}")
    print(f"  New Q(0,1) = {Q_s_a} + {alpha} * {td_error:.4f} = {new_Q:.4f}")


def exercise_2():
    """
    Exercise 2: Q-Learning vs SARSA on Cliff Walking

    Implement both algorithms and compare their paths.
    Q-Learning: off-policy, tends to find optimal but risky path.
    SARSA: on-policy, finds a safer path under epsilon-greedy.
    """
    print("\nQ-Learning vs SARSA on Cliff Walking:")
    print("=" * 60)

    # Cliff Walking environment
    class CliffWalk:
        ROWS, COLS = 4, 12
        START, GOAL = (3, 0), (3, 11)
        CLIFF = {(3, c) for c in range(1, 11)}
        ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]

        def reset(self):
            self.pos = list(self.START)
            return tuple(self.pos)

        def step(self, a):
            dr, dc = self.ACTIONS[a]
            r = max(0, min(self.ROWS-1, self.pos[0]+dr))
            c = max(0, min(self.COLS-1, self.pos[1]+dc))
            self.pos = [r, c]
            s = tuple(self.pos)
            if s in self.CLIFF:
                self.pos = list(self.START)
                return tuple(self.pos), -100.0, False
            done = s == self.GOAL
            return s, -1.0, done

    def train(method, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1, seed=42):
        np.random.seed(seed)
        Q = defaultdict(lambda: np.zeros(4))
        env = CliffWalk()
        episode_rewards = []

        for ep in range(n_episodes):
            state = env.reset()
            total_r = 0.0
            done = False
            action = (np.random.randint(4) if np.random.random() < epsilon
                      else int(np.argmax(Q[state])))

            for _ in range(200):
                next_state, reward, done = env.step(action)
                next_action = (np.random.randint(4) if np.random.random() < epsilon
                               else int(np.argmax(Q[next_state])))

                if method == 'qlearning':
                    target = reward + gamma * np.max(Q[next_state]) * (not done)
                else:  # sarsa
                    target = reward + gamma * Q[next_state][next_action] * (not done)

                Q[state][action] += alpha * (target - Q[state][action])
                total_r += reward
                state, action = next_state, next_action
                if done:
                    break

            episode_rewards.append(total_r)

        return Q, episode_rewards

    Q_ql, r_ql = train('qlearning')
    Q_sa, r_sa = train('sarsa')

    print(f"\n  Method       | Mean Reward (last 100 ep)")
    print(f"  -------------|-------------------------")
    print(f"  Q-Learning   | {np.mean(r_ql[-100:]):>8.2f}")
    print(f"  SARSA        | {np.mean(r_sa[-100:]):>8.2f}")

    print("\n  Interpretation:")
    print("  Q-Learning learns the OPTIMAL policy (close to cliff)")
    print("  but evaluation with epsilon-greedy is sometimes penalized")
    print("  by accidental cliff falls.")
    print("  SARSA accounts for exploration in its value estimates,")
    print("  learning the SAFER upper path (avoids cliff during training).")


def exercise_3():
    """
    Exercise 3: Double Q-Learning — Maximization Bias

    Show that standard Q-Learning has maximization bias (overestimates Q-values).
    Double Q-Learning corrects this.
    """
    print("\nDouble Q-Learning — Maximization Bias:")
    print("=" * 60)

    np.random.seed(42)

    # Environment: 2 actions, both reward from N(0, 1) (true Q = 0 for both)
    # Standard Q-Learning should overestimate because max(noisy estimates) > 0
    n_actions = 5
    true_q = np.zeros(n_actions)  # True Q-values are all 0

    def sample_reward(action):
        return np.random.normal(true_q[action], 1.0)

    # Standard Q-Learning estimate
    Q_single = np.zeros(n_actions)
    counts = np.ones(n_actions)  # Start with 1 to avoid div-by-zero

    n_steps = 2000
    estimates_single = []
    estimates_double = []

    # Double Q-Learning: two Q-tables
    Q_A = np.zeros(n_actions)
    Q_B = np.zeros(n_actions)
    counts_A = np.ones(n_actions)
    counts_B = np.ones(n_actions)

    for step in range(n_steps):
        # Standard Q-Learning: action = argmax Q, update with max
        a = int(np.argmax(Q_single))
        r = sample_reward(a)
        counts[a] += 1
        Q_single[a] += (r - Q_single[a]) / counts[a]
        estimates_single.append(np.max(Q_single))

        # Double Q-Learning: use A to select, B to evaluate
        if np.random.random() < 0.5:
            a_A = int(np.argmax(Q_A))
            r_A = sample_reward(a_A)
            counts_A[a_A] += 1
            Q_A[a_A] += (r_A - Q_A[a_A]) / counts_A[a_A]
        else:
            a_B = int(np.argmax(Q_B))
            r_B = sample_reward(a_B)
            counts_B[a_B] += 1
            Q_B[a_B] += (r_B - Q_B[a_B]) / counts_B[a_B]

        # Double estimate: average of max Q from both tables
        estimates_double.append((np.max(Q_A) + np.max(Q_B)) / 2)

    print(f"\n  True Q-values: all 0.0 ({n_actions} actions)")
    print(f"\n  After {n_steps} steps:")
    print(f"  Standard Q-Learning max estimate: {estimates_single[-1]:.4f} "
          f"(expected ~{0.0})")
    print(f"  Double Q-Learning max estimate:   {estimates_double[-1]:.4f} "
          f"(expected ~{0.0})")

    # Rolling average to see bias clearly
    print(f"\n  Rolling max-Q estimate (window=200):")
    print(f"  {'Steps':>8} | {'Standard QL':>12} | {'Double QL':>12}")
    print("  " + "-" * 40)
    for start in range(0, n_steps, 200):
        end = start + 200
        sq = np.mean(estimates_single[start:end])
        dq = np.mean(estimates_double[start:end])
        print(f"  {start+1:>5}-{end:<3} | {sq:>12.4f} | {dq:>12.4f}")

    print("\n  Standard Q-Learning overestimates because max of noisy estimates")
    print("  > true max. Double Q-Learning decorrelates selection from evaluation.")


def exercise_4():
    """
    Exercise 4: Epsilon Decay Schedule

    Compare constant vs decaying epsilon in Q-learning.
    """
    print("\nEpsilon Decay Schedule Comparison:")
    print("=" * 60)

    class GridEnv:
        SIZE = 5
        ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]
        def reset(self): self.pos = [0,0]; return (0,0)
        def step(self, a):
            dr,dc = self.ACTIONS[a]
            self.pos = [max(0,min(self.SIZE-1,self.pos[0]+dr)),
                        max(0,min(self.SIZE-1,self.pos[1]+dc))]
            s = tuple(self.pos)
            done = s == (self.SIZE-1, self.SIZE-1)
            return s, (1.0 if done else -0.01), done

    def train_with_schedule(schedule, n_episodes=500, seed=0):
        np.random.seed(seed)
        Q = defaultdict(lambda: np.zeros(4))
        env = GridEnv()
        rewards = []
        for ep in range(n_episodes):
            eps = schedule(ep)
            state = env.reset()
            total_r = 0.0
            done = False
            for _ in range(50):
                a = np.random.randint(4) if np.random.random() < eps else int(np.argmax(Q[state]))
                ns, r, done = env.step(a)
                Q[state][a] += 0.1 * (r + 0.99 * np.max(Q[ns]) * (not done) - Q[state][a])
                state = ns; total_r += r
                if done: break
            rewards.append(total_r)
        return rewards

    schedules = {
        'constant 0.3':  lambda ep: 0.3,
        'constant 0.1':  lambda ep: 0.1,
        'linear decay':  lambda ep: max(0.05, 1.0 - ep / 400),
        'exp decay':     lambda ep: max(0.05, 1.0 * (0.99 ** ep)),
    }

    print(f"\n  {'Schedule':>18} | {'Final 100 ep mean':>18}")
    print("  " + "-" * 42)
    for name, sched in schedules.items():
        r = train_with_schedule(sched)
        print(f"  {name:>18} | {np.mean(r[-100:]):>18.4f}")

    print("\n  Decaying epsilon: explore early, exploit later.")
    print("  Constant epsilon: always maintains some exploration")
    print("  (good for non-stationary environments).")


if __name__ == "__main__":
    print("=== Exercise 1: Q-Learning Update ===")
    exercise_1()

    print("\n=== Exercise 2: Q-Learning vs SARSA ===")
    exercise_2()

    print("\n=== Exercise 3: Double Q-Learning ===")
    exercise_3()

    print("\n=== Exercise 4: Epsilon Decay ===")
    exercise_4()

    print("\nAll exercises completed!")
