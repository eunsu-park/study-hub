"""
Exercises for Lesson 20: Goal-Conditioned Reinforcement Learning
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


def exercise_1():
    """
    Exercise 1: Goal-Conditioned MDP Formulation

    Formally define the GCMDP and show why a single policy
    π(a|s,g) is more data-efficient than one policy per goal.
    """
    print("Goal-Conditioned MDP Formulation:")
    print("=" * 60)

    print()
    print("Standard MDP:     (S, A, T, R, gamma)")
    print("Goal-Cond MDP:    (S, A, G, T, R_g, gamma)")
    print()
    print("Where:")
    print("  G = goal space (e.g., target positions)")
    print("  R_g(s, a, s', g) = reward given goal g")
    print("  π(a | s, g) = policy conditioned on goal")
    print()
    print("Data efficiency comparison:")
    print()

    # Show exponential scaling benefit
    grid_sizes = [2, 5, 10, 20]
    print(f"  {'Grid size':>10} | {'#Goals':>8} | {'Single-policy':>15} | {'N policies':>12}")
    print("  " + "-" * 52)
    for g in grid_sizes:
        n_goals = g * g
        single_policy_params = g * g * 4  # Q-table: states x actions
        n_policies_params = n_goals * single_policy_params
        print(f"  {g:>10} | {n_goals:>8} | {single_policy_params:>15} | "
              f"{n_policies_params:>12}")

    print()
    print("  Single goal-conditioned policy scales as O(|S| * |G| * |A|),")
    print("  while separate policies scale as O(|G| * |S| * |A|).")
    print("  More importantly, the goal-conditioned policy GENERALIZES")
    print("  across goals through shared parameters — separate policies cannot.")


def exercise_2():
    """
    Exercise 2: Implement HER with Different Strategies

    Implement and compare HER relabeling strategies:
    - 'final': relabel with the final achieved state
    - 'future': relabel with a random future state in the episode
    """
    print("\nHER Relabeling Strategies:")
    print("=" * 60)

    class ReachEnv:
        SIZE = 4
        ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]

        def reset(self, goal):
            self.pos = np.array([0, 0])
            self.goal = np.array(goal)
            return tuple(self.pos), tuple(self.goal)

        def step(self, a):
            dr, dc = self.ACTIONS[a]
            self.pos = np.clip(self.pos + [dr, dc], 0, self.SIZE-1)
            s = tuple(self.pos)
            done = np.array_equal(self.pos, self.goal)
            return s, (1.0 if done else 0.0), done

        def encode(self, pos):
            return pos[0] * self.SIZE + pos[1]

    def train_her(strategy, n_episodes=3000, seed=42):
        np.random.seed(seed)
        n_pos = 16
        Q = np.zeros((n_pos, n_pos, 4))
        env = ReachEnv()
        successes = []

        for ep in range(n_episodes):
            goal = tuple(np.random.randint(0, env.SIZE, 2))
            s, g = env.reset(goal)
            s_idx, g_idx = env.encode(s), env.encode(g)
            trajectory = []
            ep_success = False

            for _ in range(env.SIZE * 3):
                eps = max(0.05, 0.5 - ep / 1500)
                a = np.random.randint(4) if np.random.random() < eps else int(np.argmax(Q[s_idx, g_idx]))
                ns, r, done = env.step(a)
                ns_idx = env.encode(ns)
                trajectory.append((s_idx, g_idx, a, r, ns_idx, done))
                Q[s_idx, g_idx, a] += 0.1 * ((r if done else r + 0.99 * np.max(Q[ns_idx, g_idx])) - Q[s_idx, g_idx, a])
                s_idx = ns_idx
                if done:
                    ep_success = True
                    break

            # HER relabeling
            if trajectory:
                if strategy == 'final':
                    hindsight_goals = [trajectory[-1][4]]
                else:  # 'future'
                    hindsight_goals = [trajectory[np.random.randint(i, len(trajectory))][4]
                                       for i in range(len(trajectory))]
                    hindsight_goals = list(set(hindsight_goals))

                for hg_idx in hindsight_goals:
                    for (s2, _, a2, _, ns2, _) in trajectory:
                        her_done = (ns2 == hg_idx)
                        her_r = 1.0 if her_done else 0.0
                        Q[s2, hg_idx, a2] += 0.1 * ((her_r if her_done else her_r + 0.99 * np.max(Q[ns2, hg_idx])) - Q[s2, hg_idx, a2])

            successes.append(float(ep_success))

        return successes

    hist_final = train_her('final')
    hist_future = train_her('future')

    window = 500
    print(f"\n  Success rate (rolling {window}-episode window):")
    print(f"  {'Episodes':>12} | {'HER-final':>12} | {'HER-future':>12}")
    print("  " + "-" * 44)
    for start in range(0, 3000, window):
        end = start + window
        hf = np.mean(hist_final[start:end])
        hfu = np.mean(hist_future[start:end])
        print(f"  {start+1:>5}-{end:<5}   | {hf:>10.1%} | {hfu:>10.1%}")

    print("\n  'future' strategy relabels with any future state in episode,")
    print("  providing more diverse hindsight goals than 'final' alone.")


def exercise_3():
    """
    Exercise 3: Universal Value Function Properties

    Demonstrate that Q(s, g, a) generalizes across goals.
    """
    print("\nUniversal Value Function Generalization:")
    print("=" * 60)

    np.random.seed(3)

    # Simple 1-D reach task: state in {0,...,9}, goal in {0,...,9}
    n_states = 10
    n_goals = 10
    n_actions = 2  # 0=left, 1=right

    Q = np.zeros((n_states, n_goals, n_actions))

    def transition(s, a, g):
        ns = min(n_states-1, s+1) if a == 1 else max(0, s-1)
        done = (ns == g)
        r = 1.0 if done else -0.1
        return ns, r, done

    # Train on a subset of goals (0, 2, 4, 6, 8)
    training_goals = [0, 2, 4, 6, 8]
    test_goals = [1, 3, 5, 7, 9]

    for ep in range(5000):
        g = np.random.choice(training_goals)
        s = np.random.randint(n_states)
        for _ in range(15):
            eps = max(0.05, 0.5 - ep / 2500)
            a = np.random.randint(2) if np.random.random() < eps else int(np.argmax(Q[s, g]))
            ns, r, done = transition(s, a, g)
            Q[s, g, a] += 0.1 * ((r if done else r + 0.99 * np.max(Q[ns, g])) - Q[s, g, a])
            s = ns
            if done:
                break

    def evaluate_goal(g, n_trials=100):
        successes = 0
        for _ in range(n_trials):
            s = np.random.randint(n_states)
            for _ in range(15):
                a = int(np.argmax(Q[s, g]))
                s2, _, done = transition(s, a, g)
                s = s2
                if done:
                    successes += 1
                    break
        return successes / n_trials

    print(f"\n  Training goals: {training_goals}")
    print(f"  Test goals:     {test_goals}")
    print(f"\n  Success rate for each goal:")
    print(f"  {'Goal':>6} | {'Success Rate':>14} | {'In Training':>12}")
    print("  " + "-" * 38)

    for g in range(n_goals):
        rate = evaluate_goal(g)
        in_train = g in training_goals
        print(f"  {g:>6} | {rate:>14.1%} | {str(in_train):>12}")

    print("\n  UVFA learns to generalize across goals.")
    print("  Even unseen test goals achieve non-zero success rates.")


def exercise_4():
    """
    Exercise 4: Sparse vs Dense Reward in Goal-Conditioned Settings

    Show that HER makes sparse reward tractable.
    Without HER, sparse rewards converge much more slowly.
    """
    print("\nSparse vs Dense Reward with HER:")
    print("=" * 60)

    np.random.seed(88)

    class GoalGridEnv:
        SIZE = 5
        ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]
        def reset(self, goal):
            self.pos = [0, 0]
            self.goal = goal
            return tuple(self.pos)
        def step(self, a):
            dr, dc = self.ACTIONS[a]
            self.pos = [max(0,min(self.SIZE-1,self.pos[0]+dr)),
                        max(0,min(self.SIZE-1,self.pos[1]+dc))]
            s = tuple(self.pos)
            reached = s == tuple(self.goal)
            return s, (1.0 if reached else 0.0), reached

    def train_policy(use_her, use_dense, n_episodes=1500):
        Q = defaultdict(lambda: np.zeros(4))
        env = GoalGridEnv()
        successes = []

        for ep in range(n_episodes):
            goal = tuple(np.random.randint(0, env.SIZE, 2))
            state = env.reset(goal)
            traj = []
            ep_success = False

            for _ in range(env.SIZE * 3):
                eps = max(0.05, 0.5 - ep / 750)
                sa_key = (state, goal)
                a = np.random.randint(4) if np.random.random() < eps else int(np.argmax(Q[sa_key]))
                ns, r_sparse, done = env.step(a)
                if use_dense:
                    # Dense: negative Manhattan distance
                    r = -(abs(ns[0]-goal[0]) + abs(ns[1]-goal[1])) * 0.1 + (1.0 if done else 0.0)
                else:
                    r = r_sparse
                nsk = (ns, goal)
                Q[sa_key][a] += 0.1 * ((r if done else r + 0.99 * np.max(Q[nsk])) - Q[sa_key][a])
                traj.append((state, goal, a, r_sparse, ns, done))
                state = ns
                if done:
                    ep_success = True
                    break

            if use_her and traj:
                hg = traj[-1][4]  # final achieved state
                for (s2, _, a2, _, ns2, _) in traj:
                    her_done = (ns2 == hg)
                    her_r = 1.0 if her_done else 0.0
                    hsk = (s2, hg)
                    hsnk = (ns2, hg)
                    Q[hsk][a2] += 0.1 * ((her_r if her_done else her_r + 0.99 * np.max(Q[hsnk])) - Q[hsk][a2])

            successes.append(float(ep_success))
        return successes

    configs = [
        ("Sparse, no HER", False, False),
        ("Sparse, with HER", True, False),
        ("Dense, no HER", False, True),
    ]

    print(f"\n  Success rate (last 300 episodes of 1500):")
    for name, use_her, use_dense in configs:
        hist = train_policy(use_her, use_dense)
        final_rate = np.mean(hist[-300:])
        print(f"  {name:<20}: {final_rate:.1%}")

    print("\n  HER makes sparse reward competitive with dense reward engineering.")


if __name__ == "__main__":
    print("=== Exercise 1: GCMDP Formulation ===")
    exercise_1()

    print("\n=== Exercise 2: HER Strategies ===")
    exercise_2()

    print("\n=== Exercise 3: UVFA Generalization ===")
    exercise_3()

    print("\n=== Exercise 4: Sparse vs Dense Reward ===")
    exercise_4()

    print("\nAll exercises completed!")
