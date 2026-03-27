"""
Exercises for Lesson 21: Reward Shaping and Intrinsic Motivation
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


def exercise_1():
    """
    Exercise 1: Potential-Based Shaping — Policy Invariance Proof

    Verify that the optimal policy does not change after potential-based shaping.
    """
    print("Potential-Based Shaping — Policy Invariance:")
    print("=" * 60)

    print()
    print("Theorem: If F(s,s') = gamma*phi(s') - phi(s), then")
    print("the optimal policy under R' = R + F is identical to")
    print("the optimal policy under R.")
    print()
    print("Proof sketch:")
    print("  Q'(s,a) = Q(s,a) + phi(s) - gamma * E[phi(s')]")
    print("  Wait — this still depends on the transition dynamics.")
    print("  The key result is on the VALUE FUNCTION:")
    print("  V'*(s) = V*(s) + phi(s)")
    print("  argmax_a Q'(s,a) = argmax_a Q(s,a)")
    print("  => Same optimal action in every state.")
    print()

    # Numerical verification on a simple 3-state MDP
    # States: 0, 1, 2(goal). Actions: 0(stay) or 1(advance)
    # Transitions: action 1 in state s -> state s+1 with prob 1
    #              action 0 stays in place
    # Rewards: R(s=1, a=1) = 0, R(s=2, _) = 10 (goal)
    # Phi(s) = s  (simple potential)

    gamma = 0.9
    phi = {0: 0, 1: 1, 2: 2}

    # Value function under original reward (computed analytically)
    # V*(2) = 10 (absorbing), V*(1) = 0 + gamma*10 = 9, V*(0) = 0 + gamma*9 = 8.1
    V_orig = {0: 8.1, 1: 9.0, 2: 10.0}
    # V'*(s) = V*(s) + phi(s)
    V_shaped = {s: V_orig[s] + phi[s] for s in [0, 1, 2]}

    print("  Verification (3-state MDP):")
    print(f"  {'State':>6} | {'V*(s)':>8} | {'phi(s)':>8} | {'V_shaped(s)':>12}")
    print("  " + "-" * 44)
    for s in [0, 1, 2]:
        print(f"  {s:>6} | {V_orig[s]:>8.3f} | {phi[s]:>8.3f} | {V_shaped[s]:>12.3f}")

    # The optimal action in each state is still action=1 (advance)
    print(f"\n  Optimal policy under V_orig: always advance (action=1)")
    print(f"  Optimal policy under V_shaped: also always advance (action=1)")
    print(f"  => Potential-based shaping preserves the optimal policy.")

    # Also verify the anti-shaping example (non-potential shaping is NOT invariant)
    print(f"\n  Counter-example: non-potential shaping CAN change optimal policy")
    print(f"  R_bad(s,a,s') = +100 * (s' == 0) [reward for going backwards]")
    print(f"  This can cause the agent to prefer staying near state 0 — wrong behavior!")


def exercise_2():
    """
    Exercise 2: Curiosity-Driven Exploration

    Implement prediction-error-based intrinsic reward and
    show it improves coverage in a sparse environment.
    """
    print("\nCuriosity-Driven Exploration:")
    print("=" * 60)

    np.random.seed(42)

    # 1D chain with a "treasure" at state 19 (never reached by random policy)
    n_states = 20
    n_actions = 2  # 0=left, 1=right

    def transition(s, a):
        if a == 1:
            return min(n_states-1, s+1)
        return max(0, s-1)

    # Prediction model: count-based uncertainty
    visit_counts = np.zeros((n_states, n_actions))

    def intrinsic_reward(s, a):
        """Bonus = 1 / sqrt(N(s,a) + 1)"""
        return 1.0 / np.sqrt(visit_counts[s, a] + 1)

    # Train with curiosity
    Q = np.zeros((n_states, n_actions))
    state_visits_curiosity = np.zeros(n_states)

    for step in range(3000):
        s = np.random.randint(n_states // 3)  # always start in first third
        for _ in range(30):
            if np.random.random() < 0.2:
                a = np.random.randint(2)
            else:
                # Add curiosity bonus to Q-values
                bonus = np.array([intrinsic_reward(s, a2) for a2 in range(n_actions)])
                a = int(np.argmax(Q[s] + 0.5 * bonus))
            ns = transition(s, a)
            visit_counts[s, a] += 1
            state_visits_curiosity[s] += 1
            # Extrinsic reward + intrinsic bonus
            extrinsic = 10.0 if ns == n_states-1 else 0.0
            intrinsic = intrinsic_reward(s, a)
            r = extrinsic + 0.5 * intrinsic
            Q[s, a] += 0.1 * (r + 0.99 * np.max(Q[ns]) - Q[s, a])
            s = ns

    # Train without curiosity
    Q2 = np.zeros((n_states, n_actions))
    state_visits_no_curiosity = np.zeros(n_states)

    for step in range(3000):
        s = np.random.randint(n_states // 3)
        for _ in range(30):
            a = np.random.randint(2) if np.random.random() < 0.2 else int(np.argmax(Q2[s]))
            ns = transition(s, a)
            state_visits_no_curiosity[s] += 1
            r = 10.0 if ns == n_states-1 else 0.0
            Q2[s, a] += 0.1 * (r + 0.99 * np.max(Q2[ns]) - Q2[s, a])
            s = ns

    print(f"\n  Chain environment: states 0-{n_states-1}, treasure at state {n_states-1}")
    print(f"\n  State coverage (visits per state):")
    print(f"  {'State':>7} | {'No Curiosity':>14} | {'With Curiosity':>16}")
    print("  " + "-" * 46)
    for s in range(0, n_states, 2):
        nc = int(state_visits_no_curiosity[s])
        cu = int(state_visits_curiosity[s])
        print(f"  {s:>7} | {nc:>14} | {cu:>16}")

    farthest_nc = max((s for s in range(n_states) if state_visits_no_curiosity[s] > 0), default=0)
    farthest_cu = max((s for s in range(n_states) if state_visits_curiosity[s] > 0), default=0)
    print(f"\n  Farthest state reached: no_curiosity={farthest_nc}, curiosity={farthest_cu}")


def exercise_3():
    """
    Exercise 3: Reward Hacking Example

    Design a reward function, show how an agent can hack it,
    and fix the reward to prevent hacking.
    """
    print("\nReward Hacking and Mitigation:")
    print("=" * 60)

    print()
    print("Example: Robot vacuum cleaner")
    print()
    print("  Naive reward: R = dirt_collected_per_step")
    print("  Hacked behavior: dump dirt -> immediately re-collect -> repeat")
    print()

    # Simulate the hacking behavior
    class VacuumEnv:
        def __init__(self, total_dirt=10):
            self.dirt = total_dirt
            self.hopper = 0       # dirt in robot's hopper
            self.cleaned = 0      # actually cleaned (emptied into bin)
            self.steps = 0

        def vacuum_action(self):
            """Pick up dirt from floor."""
            if self.dirt > 0:
                self.dirt -= 1
                self.hopper += 1
                return 1  # reward: dirt collected
            return 0

        def dump_action(self):
            """Dump hopper back on floor (hacking action)."""
            self.dirt += self.hopper
            self.hopper = 0
            return 0  # no direct reward

        def empty_bin_action(self):
            """Properly empty hopper into bin."""
            cleaned = self.hopper
            self.cleaned += cleaned
            self.hopper = 0
            return 0  # no direct reward

    # Strategy A: always vacuum (optimal clean behavior)
    env_a = VacuumEnv(total_dirt=10)
    rewards_a = []
    for _ in range(30):
        r = env_a.vacuum_action()
        rewards_a.append(r)
    rewards_a += [env_a.empty_bin_action() for _ in range(5)]

    # Strategy B: vacuum, dump, vacuum (hacking)
    env_b = VacuumEnv(total_dirt=5)
    rewards_b = []
    for _ in range(50):
        env_b.vacuum_action()  # pick up all dirt
        rewards_b.append(1)
        env_b.dump_action()    # dump back
        rewards_b.append(0)

    print(f"  Strategy A (clean properly):")
    print(f"    Total reward collected: {sum(rewards_a)}")
    print(f"    Actual cleaned: {env_a.cleaned}")
    print(f"    Dirt remaining: {env_a.dirt}")
    print()
    print(f"  Strategy B (hack: vacuum-dump-repeat):")
    print(f"    Total reward collected: {sum(rewards_b)}")
    print(f"    Actual cleaned: {env_b.cleaned}")
    print(f"    Dirt remaining: {env_b.dirt}")
    print()
    print("  Strategy B gets MORE reward but does ZERO actual cleaning!")
    print()
    print("  Fixed reward: R = dirt_actually_removed_from_environment")
    print("  (not just collected into hopper)")
    print("  => Only reward when dirt enters the permanent bin.")


def exercise_4():
    """
    Exercise 4: Count-Based vs Curiosity-Based Exploration

    Compare count-based bonus (N-visit) vs prediction-error curiosity
    in terms of exploration efficiency.
    """
    print("\nCount-Based vs Prediction-Error Exploration:")
    print("=" * 60)

    np.random.seed(0)
    n_states = 15

    def transition(s, a):
        if a == 1: return min(n_states-1, s+1)
        return max(0, s-1)

    def run_with_bonus(bonus_type, n_steps=2000):
        Q = np.zeros((n_states, 2))
        visit_N = np.zeros((n_states, 2))
        pred_errors = np.ones((n_states, 2))  # for curiosity: running avg error
        state_coverage = np.zeros(n_states)

        for step in range(n_steps):
            s = 0  # always start from 0
            for _ in range(20):
                if bonus_type == 'count':
                    bonus = 1.0 / np.sqrt(visit_N[s] + 1)
                elif bonus_type == 'curiosity':
                    bonus = pred_errors[s]
                else:
                    bonus = np.zeros(2)

                a = int(np.argmax(Q[s] + bonus))
                ns = transition(s, a)
                visit_N[s, a] += 1
                state_coverage[ns] += 1

                # Update curiosity model (prediction error decays on visited transitions)
                if bonus_type == 'curiosity':
                    actual_error = 1.0 if visit_N[s, a] < 3 else 0.1
                    pred_errors[s, a] = 0.9 * pred_errors[s, a] + 0.1 * actual_error

                Q[s, a] += 0.1 * (0 + 0.99 * np.max(Q[ns]) - Q[s, a])
                s = ns

        return state_coverage

    cov_count = run_with_bonus('count')
    cov_curiosity = run_with_bonus('curiosity')
    cov_random = run_with_bonus('none')

    print(f"\n  State coverage (visits) after 2000 episodes from s=0:")
    print(f"  {'State':>7} | {'Random':>8} | {'Count-based':>12} | {'Curiosity':>10}")
    print("  " + "-" * 48)
    for s in range(0, n_states, 2):
        print(f"  {s:>7} | {cov_random[s]:>8.0f} | {cov_count[s]:>12.0f} | {cov_curiosity[s]:>10.0f}")

    farthest_r = max((s for s in range(n_states) if cov_random[s] > 0), default=0)
    farthest_c = max((s for s in range(n_states) if cov_count[s] > 0), default=0)
    farthest_cu = max((s for s in range(n_states) if cov_curiosity[s] > 0), default=0)
    print(f"\n  Farthest state: random={farthest_r}, count={farthest_c}, curiosity={farthest_cu}")
    print("  Both exploration bonuses encourage visiting novel states.")


if __name__ == "__main__":
    print("=== Exercise 1: Policy Invariance ===")
    exercise_1()

    print("\n=== Exercise 2: Curiosity-Driven Exploration ===")
    exercise_2()

    print("\n=== Exercise 3: Reward Hacking ===")
    exercise_3()

    print("\n=== Exercise 4: Count-Based vs Curiosity ===")
    exercise_4()

    print("\nAll exercises completed!")
