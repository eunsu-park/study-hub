"""
Exercises for Lesson 28: Capstone — Training an RL Agent End-to-End
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict
import time


def exercise_1():
    """
    Exercise 1: Project Planning

    Design a complete RL project plan for a given problem.
    Problem: Train an agent to navigate a 10x10 grid to collect items
             while avoiding dynamic obstacles.
    """
    print("RL Project Planning:")
    print("=" * 60)

    project_plan = {
        "1. Problem Analysis": {
            "Why RL?": "Sequential decisions, no ground truth labels, maximize collection",
            "State space": "10x10 grid, agent pos, item positions, obstacle positions",
            "Action space": "4 directions (up/down/left/right), discrete",
            "Reward function": "+5 for collecting item, -1 for hitting obstacle, -0.01 per step",
            "Episode length": "Max 200 steps or all items collected",
            "Success metric": "Items collected per episode (mean over 100 episodes)",
        },
        "2. Algorithm Selection": {
            "Baseline": "Tabular Q-learning (small state space, interpretable)",
            "Primary": "DQN (if feature representation needed)",
            "Why not PPO?": "Discrete actions + small grid -> Q-learning suffices",
            "Hyperparameters": "gamma=0.99, lr=0.001, epsilon decay: 1.0->0.05 over 2000 eps",
        },
        "3. Training Pipeline": {
            "Warmup": "1000 random episodes to fill replay buffer",
            "Training": "10000 episodes with epsilon decay",
            "Evaluation": "Every 500 episodes, evaluate 100 greedy episodes",
            "Checkpointing": "Save model every 1000 episodes",
        },
        "4. Debugging Checklist": {
            "Sanity check": "Random policy should score ~0.5 items; verify reward scale",
            "Learning check": "Q-values should increase over training",
            "Exploration": "Log epsilon; verify gradual decay",
            "Common issues": "Reward too sparse, learning rate too large, discount too small",
        },
        "5. Evaluation": {
            "Seeds": "5 independent seeds minimum",
            "Baselines": "Random policy, greedy heuristic",
            "Metrics": "Mean ± std items collected, steps to first item, success rate",
            "Ablations": "gamma, lr, epsilon schedule",
        },
    }

    for section, items in project_plan.items():
        print(f"\n  {section}")
        for key, val in items.items():
            print(f"    {key}: {val}")


def exercise_2():
    """
    Exercise 2: RL Debugging — Diagnosing Common Failures

    Identify and fix common RL training problems.
    """
    print("\nRL Debugging — Common Failures and Fixes:")
    print("=" * 60)

    np.random.seed(42)

    # Minimal grid for debugging
    class DebugGrid:
        SIZE = 4
        GOAL = (3, 3)
        ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]
        def reset(self):
            self.pos = [0, 0]
            return tuple(self.pos)
        def step(self, a):
            dr, dc = self.ACTIONS[a]
            self.pos = [max(0,min(self.SIZE-1,self.pos[0]+dr)),
                        max(0,min(self.SIZE-1,self.pos[1]+dc))]
            s = tuple(self.pos)
            done = s == self.GOAL
            return s, (1.0 if done else -0.01), done

    def run_and_diagnose(lr, gamma, epsilon_init, epsilon_final, n_episodes=500):
        Q = defaultdict(lambda: np.zeros(4))
        env = DebugGrid()
        rewards = []
        q_magnitudes = []
        epsilon = epsilon_init

        for ep in range(n_episodes):
            s = env.reset()
            total_r = 0.0
            done = False
            for _ in range(50):
                a = np.random.randint(4) if np.random.random() < epsilon else int(np.argmax(Q[s]))
                ns, r, done = env.step(a)
                Q[s][a] += lr * (r + gamma * np.max(Q[ns]) * (not done) - Q[s][a])
                s = ns
                total_r += r
                if done: break
            rewards.append(total_r)
            epsilon = max(epsilon_final, epsilon * 0.997)
            q_magnitudes.append(np.mean([np.max(np.abs(v)) for v in Q.values()]))

        return np.mean(rewards[-100:]), np.mean(q_magnitudes[-100:])

    configs = [
        ("Normal",         0.1, 0.99, 1.0, 0.05, "Baseline"),
        ("LR too high",    0.9, 0.99, 1.0, 0.05, "Unstable: Q-values oscillate"),
        ("LR too low",     0.001, 0.99, 1.0, 0.05, "Slow: needs more episodes"),
        ("Low gamma",      0.1, 0.2,  1.0, 0.05, "Myopic: ignores future reward"),
        ("No exploration", 0.1, 0.99, 0.1, 0.05, "Stuck: poor initial exploration"),
        ("No decay",       0.1, 0.99, 0.5, 0.5,  "Always exploring: poor late perf"),
    ]

    print(f"\n  {'Config':>18} | {'Avg Reward':>12} | {'Q Magnitude':>12} | {'Diagnosis':>35}")
    print("  " + "-" * 90)

    for name, lr, gamma, eps_i, eps_f, diagnosis in configs:
        mean_r, mean_q = run_and_diagnose(lr, gamma, eps_i, eps_f)
        print(f"  {name:>18} | {mean_r:>12.4f} | {mean_q:>12.4f} | {diagnosis}")

    print("\n  Debugging tips:")
    print("  1. Check reward scale: if Q-values explode, reduce LR or reward")
    print("  2. Plot epsilon decay: ensure enough exploration early")
    print("  3. Monitor Q-value magnitude: should grow slowly then stabilize")
    print("  4. Check gamma: too low => myopic, too high => unstable with function approx")


def exercise_3():
    """
    Exercise 3: Reproducibility and Multi-Seed Evaluation

    Demonstrate proper multi-seed evaluation with statistical reporting.
    """
    print("\nReproducibility — Multi-Seed Evaluation:")
    print("=" * 60)

    class SimpleEnv:
        SIZE = 5
        ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]
        def reset(self): self.pos = [0,0]; return (0,0)
        def step(self, a):
            dr,dc = self.ACTIONS[a]
            self.pos = [max(0,min(self.SIZE-1,self.pos[0]+dr)),
                        max(0,min(self.SIZE-1,self.pos[1]+dc))]
            s = tuple(self.pos)
            done = s == (self.SIZE-1,self.SIZE-1)
            return s, (1.0 if done else -0.01), done

    def train_and_eval(seed, n_episodes=300):
        np.random.seed(seed)
        Q = defaultdict(lambda: np.zeros(4))
        env = SimpleEnv()
        epsilon = 1.0
        for ep in range(n_episodes):
            s = env.reset()
            done = False
            for _ in range(50):
                a = np.random.randint(4) if np.random.random() < epsilon else int(np.argmax(Q[s]))
                ns, r, done = env.step(a)
                Q[s][a] += 0.1 * (r + 0.99 * np.max(Q[ns]) * (not done) - Q[s][a])
                s = ns
                if done: break
            epsilon = max(0.05, epsilon * 0.99)

        # Greedy evaluation
        success = 0
        for _ in range(200):
            s = env.reset()
            for _ in range(30):
                a = int(np.argmax(Q[s]))
                s, _, done = env.step(a)
                if done: success += 1; break
        return success / 200

    n_seeds = 10
    results = []
    print(f"\n  Running {n_seeds} independent seeds...")
    for seed in range(n_seeds):
        r = train_and_eval(seed)
        results.append(r)
        print(f"    Seed {seed:2d}: {r:.1%}")

    mean = np.mean(results)
    std = np.std(results)
    ci_95 = 1.96 * std / np.sqrt(n_seeds)

    print(f"\n  Statistical summary ({n_seeds} seeds):")
    print(f"    Mean:           {mean:.3f} ({mean:.1%})")
    print(f"    Std:            {std:.3f}")
    print(f"    95% CI:         [{mean-ci_95:.3f}, {mean+ci_95:.3f}]")
    print(f"    Min / Max:      {min(results):.3f} / {max(results):.3f}")
    print(f"\n  Best practice: always report mean ± std over ≥5 seeds.")
    print(f"  Single-seed results can be misleading (lucky or unlucky init).")

    # Same seed -> identical results
    r1 = train_and_eval(42)
    r2 = train_and_eval(42)
    print(f"\n  Determinism check (seed=42): {r1:.4f} == {r2:.4f}: {r1 == r2}")


def exercise_4():
    """
    Exercise 4: Hyperparameter Ablation Study

    Conduct a systematic ablation to understand which hyperparameters
    matter most for performance.
    """
    print("\nHyperparameter Ablation Study:")
    print("=" * 60)

    class SimpleEnv:
        SIZE = 5
        ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]
        def reset(self): self.pos = [0,0]; return (0,0)
        def step(self, a):
            dr,dc = self.ACTIONS[a]
            self.pos = [max(0,min(self.SIZE-1,self.pos[0]+dr)),
                        max(0,min(self.SIZE-1,self.pos[1]+dc))]
            s = tuple(self.pos)
            done = s == (self.SIZE-1,self.SIZE-1)
            return s, (1.0 if done else -0.01), done

    def run(lr, gamma, epsilon_decay, n_seeds=3, n_ep=300):
        rates = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            Q = defaultdict(lambda: np.zeros(4))
            env = SimpleEnv()
            epsilon = 1.0
            for ep in range(n_ep):
                s = env.reset()
                done = False
                for _ in range(50):
                    a = np.random.randint(4) if np.random.random() < epsilon else int(np.argmax(Q[s]))
                    ns, r, done = env.step(a)
                    Q[s][a] += lr * (r + gamma * np.max(Q[ns]) * (not done) - Q[s][a])
                    s = ns
                    if done: break
                epsilon = max(0.05, epsilon * epsilon_decay)
            success = sum(
                1 for _ in range(100) for _ in [0]
                if (lambda s2: any(
                    (s2 := env.step(int(np.argmax(Q[s2])))[0]) == (4,4)
                    for _ in range(20)
                ))(env.reset())
            ) / 100
            rates.append(success)
        return np.mean(rates), np.std(rates)

    # Simplified evaluation
    def quick_eval(lr, gamma, epsilon_decay, n_seeds=3, n_ep=300):
        rates = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            Q = defaultdict(lambda: np.zeros(4))
            env = SimpleEnv()
            epsilon = 1.0
            for ep in range(n_ep):
                s = env.reset()
                done = False
                for _ in range(50):
                    a = np.random.randint(4) if np.random.random() < epsilon else int(np.argmax(Q[s]))
                    ns, r, done = env.step(a)
                    Q[s][a] += lr * (r + gamma * np.max(Q[ns]) * (not done) - Q[s][a])
                    s = ns
                    if done: break
                epsilon = max(0.05, epsilon * epsilon_decay)
            # Quick success rate
            successes = 0
            for _ in range(100):
                s = env.reset()
                for _ in range(30):
                    a = int(np.argmax(Q[s]))
                    s, _, done = env.step(a)
                    if done: successes += 1; break
            rates.append(successes / 100)
        return np.mean(rates), np.std(rates)

    print(f"\n  Baseline: lr=0.1, gamma=0.99, epsilon_decay=0.99")
    base_mean, base_std = quick_eval(0.1, 0.99, 0.99)
    print(f"  Baseline: {base_mean:.3f} ± {base_std:.3f}")

    print(f"\n  Learning Rate Ablation:")
    for lr in [0.01, 0.05, 0.1, 0.3, 0.5]:
        m, s = quick_eval(lr, 0.99, 0.99)
        print(f"    lr={lr:.2f}: {m:.3f} ± {s:.3f}  {'<- BEST' if m == max([quick_eval(l, 0.99, 0.99)[0] for l in [0.01,0.05,0.1,0.3,0.5]]) else ''}")

    print(f"\n  Gamma Ablation:")
    for gamma in [0.5, 0.8, 0.9, 0.95, 0.99]:
        m, s = quick_eval(0.1, gamma, 0.99)
        print(f"    gamma={gamma:.2f}: {m:.3f} ± {s:.3f}")

    print(f"\n  Epsilon Decay Ablation:")
    for ed in [0.99, 0.995, 0.997, 0.999]:
        m, s = quick_eval(0.1, 0.99, ed)
        print(f"    decay={ed}: {m:.3f} ± {s:.3f}")

    print(f"\n  Key findings:")
    print(f"  - Learning rate: moderate values work best (too high = unstable)")
    print(f"  - Gamma: near 1.0 for tasks with delayed reward")
    print(f"  - Epsilon decay: too fast = poor exploration, too slow = slow convergence")


if __name__ == "__main__":
    print("=== Exercise 1: Project Planning ===")
    exercise_1()

    print("\n=== Exercise 2: RL Debugging ===")
    exercise_2()

    print("\n=== Exercise 3: Reproducibility ===")
    exercise_3()

    print("\n=== Exercise 4: Ablation Study ===")
    exercise_4()

    print("\nAll exercises completed!")
