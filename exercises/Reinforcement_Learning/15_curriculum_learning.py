"""
Exercises for Lesson 15: Curriculum Learning for RL
Topic: Reinforcement_Learning

Solutions to practice problems covering stage curricula,
automatic difficulty scoring, reward shaping, domain randomization,
and HER.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# Shared: GridWorld Environment
# ============================================================
class GridWorld:
    """Configurable GridWorld for curriculum learning experiments."""

    def __init__(self, size=5, num_obstacles=0, goal=None, seed=None):
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.obstacles = set()
        self.goal = goal or (size - 1, size - 1)
        while len(self.obstacles) < num_obstacles:
            pos = (self.rng.randint(size), self.rng.randint(size))
            if pos != (0, 0) and pos != self.goal:
                self.obstacles.add(pos)
        self.state = (0, 0)
        self.steps = 0
        self.max_steps = size * size * 2

    def reset(self):
        self.state = (0, 0)
        self.steps = 0
        return self.state

    def step(self, action):
        self.steps += 1
        r, c = self.state
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            if (nr, nc) not in self.obstacles:
                self.state = (nr, nc)
        done = self.state == self.goal or self.steps >= self.max_steps
        reward = 1.0 if self.state == self.goal else -0.01
        return self.state, reward, done


class QLearner:
    """Tabular Q-learning agent."""

    def __init__(self, n_actions=4, lr=0.1, gamma=0.99, epsilon=0.2):
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * np.max(self.Q[s2])
        self.Q[s][a] += self.lr * (target - self.Q[s][a])


def exercise_1():
    """
    Stage curriculum for 20×20 GridWorld with obstacles.
    Compare curriculum training vs direct training on hardest stage.
    """
    print("=== Exercise 1: Stage Curriculum for GridWorld ===\n")

    # Define 4 stages with increasing difficulty
    stages = [
        {"name": "5×5, 0 obs", "size": 5, "obstacles": 0, "episodes": 200},
        {"name": "10×10, 3 obs", "size": 10, "obstacles": 3, "episodes": 300},
        {"name": "15×15, 6 obs", "size": 15, "obstacles": 6, "episodes": 400},
        {"name": "20×20, 10 obs", "size": 20, "obstacles": 10, "episodes": 500},
    ]

    total_episodes = sum(s["episodes"] for s in stages)

    # Curriculum training
    agent_cur = QLearner(lr=0.1, epsilon=0.3)
    print("  Curriculum training:")
    for stage in stages:
        successes = []
        for ep in range(stage["episodes"]):
            env = GridWorld(stage["size"], stage["obstacles"], seed=ep)
            s = env.reset()
            done = False
            while not done:
                a = agent_cur.act(s)
                s2, r, done = env.step(a)
                agent_cur.update(s, a, r, s2, done)
                s = s2
            successes.append(r > 0)
        rate = np.mean(successes[-50:]) if len(successes) >= 50 else np.mean(successes)
        print(f"    {stage['name']}: {rate:.0%} success")

    # Direct training on hardest
    agent_dir = QLearner(lr=0.1, epsilon=0.3)
    successes_dir = []
    for ep in range(total_episodes):
        env = GridWorld(20, 10, seed=ep)
        s = env.reset()
        done = False
        while not done:
            a = agent_dir.act(s)
            s2, r, done = env.step(a)
            agent_dir.update(s, a, r, s2, done)
            s = s2
        successes_dir.append(r > 0)

    # Evaluate both on hard task
    agent_cur.epsilon = 0.05
    agent_dir.epsilon = 0.05
    cur_eval, dir_eval = [], []
    for ep in range(100):
        env = GridWorld(20, 10, seed=ep + 10000)
        s = env.reset()
        done = False
        while not done:
            a = agent_cur.act(s)
            s2, r, done = env.step(a)
            s = s2
        cur_eval.append(r > 0)

        env = GridWorld(20, 10, seed=ep + 10000)
        s = env.reset()
        done = False
        while not done:
            a = agent_dir.act(s)
            s2, r, done = env.step(a)
            s = s2
        dir_eval.append(r > 0)

    print(f"\n  Evaluation on 20×20, 10 obstacles (100 episodes):")
    print(f"    Curriculum: {np.mean(cur_eval):.0%}")
    print(f"    Direct:     {np.mean(dir_eval):.0%}")
    print(f"    Both used {total_episodes} total training episodes")
    print()


def exercise_2():
    """
    Automatic difficulty scoring with frontier sampling.
    """
    print("=== Exercise 2: Automatic Difficulty Scoring ===\n")

    # 20 task configurations: varying grid size + obstacles
    tasks = []
    for i in range(20):
        size = 4 + i
        obs = max(0, i - 2)
        tasks.append({"id": i, "size": size, "obstacles": obs})

    agent = QLearner(lr=0.1, epsilon=0.25)
    history = defaultdict(list)
    target = 0.5

    task_sample_count = defaultdict(int)

    for episode in range(600):
        # Score each task
        scores = []
        for t in tasks:
            h = history[t["id"]]
            if len(h) < 3:
                scores.append(1.0)
            else:
                rate = np.mean(h[-15:])
                scores.append(max(0.05, 1.0 - abs(rate - target) * 2))

        probs = np.array(scores) / sum(scores)
        idx = np.random.choice(len(tasks), p=probs)
        task = tasks[idx]
        task_sample_count[idx] += 1

        env = GridWorld(task["size"], task["obstacles"], seed=episode)
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s)
            s2, r, done = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2
        history[task["id"]].append(float(r > 0))

    print("  Discovered task ordering (by success rate):")
    print(f"  {'Task':>6} | {'Size':>6} | {'Obs':>5} | {'Success':>8} | "
          f"{'Samples':>8}")
    print("  " + "-" * 50)

    task_rates = []
    for t in tasks:
        h = history[t["id"]]
        rate = np.mean(h) if h else 0
        task_rates.append((t, rate))

    for t, rate in sorted(task_rates, key=lambda x: -x[1]):
        print(f"  {t['id']:6d} | {t['size']:6d} | {t['obstacles']:5d} | "
              f"{rate:8.0%} | {task_sample_count[t['id']]:8d}")

    print(f"\n  The algorithm naturally samples frontier tasks (~50% success)")
    print(f"  Easy tasks get fewer samples, hard tasks get explored")
    print()


def exercise_3():
    """
    Reward shaping curriculum: dense → sparse transition.
    """
    print("=== Exercise 3: Reward Shaping Curriculum ===\n")

    size = 12
    goal = (11, 11)

    def dense_r(state, goal, size):
        r, c = state
        dist = abs(r - goal[0]) + abs(c - goal[1])
        return -dist / (2 * (size - 1))

    def sparse_r(state, goal):
        return 1.0 if state == goal else -0.01

    configs = {
        "Sparse only": lambda s, ep: sparse_r(s, goal),
        "Dense only": lambda s, ep: dense_r(s, goal, size),
        "Linear blend": lambda s, ep: (
            max(0, 1 - ep / 300) * dense_r(s, goal, size) +
            min(1, ep / 300) * sparse_r(s, goal)
        ),
    }

    for name, reward_fn in configs.items():
        agent = QLearner(lr=0.1, epsilon=0.3)
        successes = []
        for ep in range(500):
            env = GridWorld(size, 4, goal=goal, seed=ep)
            s = env.reset()
            done = False
            while not done:
                a = agent.act(s)
                s2, raw_r, done = env.step(a)
                shaped = reward_fn(s2, ep)
                agent.update(s, a, shaped, s2, done)
                s = s2
            successes.append(raw_r > 0)

        agent.epsilon = 0.05
        eval_succ = []
        for ep in range(100):
            env = GridWorld(size, 4, goal=goal, seed=ep + 5000)
            s = env.reset()
            done = False
            while not done:
                a = agent.act(s)
                s2, r, done = env.step(a)
                s = s2
            eval_succ.append(r > 0)

        print(f"  {name:<15}: {np.mean(eval_succ):.0%} eval success")

    print(f"\n  Dense reward guides exploration but may learn wrong policy")
    print(f"  Sparse reward is correct but hard to learn from")
    print(f"  Curriculum blend: best of both worlds")
    print()


def exercise_4():
    """
    Domain randomization: randomize environment parameters
    and measure transfer to unseen configurations.
    """
    print("=== Exercise 4: Domain Randomization ===\n")

    def train_with_randomization(agent, episodes, width):
        """Train with parameter randomization of given width."""
        for ep in range(episodes):
            # Randomize: size in [8±width*4], obstacles in [3±width*3]
            size = int(np.clip(8 + np.random.uniform(-width, width) * 4,
                               5, 15))
            n_obs = int(np.clip(3 + np.random.uniform(-width, width) * 3,
                                0, 8))
            env = GridWorld(size, n_obs, seed=ep)
            s = env.reset()
            done = False
            while not done:
                a = agent.act(s)
                s2, r, done = env.step(a)
                agent.update(s, a, r, s2, done)
                s = s2

    # Test configurations (unseen)
    test_configs = [
        {"size": 6, "obstacles": 1},
        {"size": 9, "obstacles": 4},
        {"size": 12, "obstacles": 6},
        {"size": 7, "obstacles": 2},
        {"size": 10, "obstacles": 5},
    ]

    widths = [0.0, 0.3, 0.6, 1.0]
    print(f"  {'Width':>8} | {'Source Acc':>10} | {'Transfer Acc':>12}")
    print("  " + "-" * 38)

    for width in widths:
        agent = QLearner(lr=0.1, epsilon=0.25)
        train_with_randomization(agent, 500, width)
        agent.epsilon = 0.05

        # Source evaluation (default config: 8×8, 3 obstacles)
        src_succ = []
        for ep in range(50):
            env = GridWorld(8, 3, seed=ep + 9000)
            s = env.reset()
            done = False
            while not done:
                a = agent.act(s)
                s2, r, done = env.step(a)
                s = s2
            src_succ.append(r > 0)

        # Transfer evaluation
        transfer_succ = []
        for cfg in test_configs:
            for ep in range(20):
                env = GridWorld(cfg["size"], cfg["obstacles"],
                                seed=ep + 8000)
                s = env.reset()
                done = False
                while not done:
                    a = agent.act(s)
                    s2, r, done = env.step(a)
                    s = s2
                transfer_succ.append(r > 0)

        print(f"  {width:8.1f} | {np.mean(src_succ):10.0%} | "
              f"{np.mean(transfer_succ):12.0%}")

    print(f"\n  Narrow randomization = good source, poor transfer")
    print(f"  Wide randomization = slightly worse source, better transfer")
    print()


def exercise_5():
    """
    HER (Hindsight Experience Replay) for goal-reaching.
    """
    print("=== Exercise 5: Hindsight Experience Replay ===\n")

    class GoalGridWorld:
        """GridWorld with explicit goal parameter."""

        def __init__(self, size=8, seed=None):
            self.size = size
            self.rng = np.random.RandomState(seed)
            self.state = (0, 0)
            self.steps = 0
            self.max_steps = size * 3

        def reset(self, goal=None):
            self.state = (0, 0)
            self.goal = goal or (self.size - 1, self.size - 1)
            self.steps = 0
            return self.state

        def step(self, action):
            self.steps += 1
            r, c = self.state
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                self.state = (nr, nc)
            done = (self.state == self.goal or
                    self.steps >= self.max_steps)
            reward = 0.0 if self.state == self.goal else -1.0
            return self.state, reward, done

    # Agent with goal-conditioned Q-table: Q[(state, goal)][action]
    Q_her = defaultdict(lambda: np.zeros(4))
    Q_no_her = defaultdict(lambda: np.zeros(4))
    lr, gamma, epsilon = 0.1, 0.99, 0.3

    def select_action(Q, state, goal, epsilon):
        key = (state, goal)
        if np.random.random() < epsilon:
            return np.random.randint(4)
        return int(np.argmax(Q[key]))

    def q_update(Q, s, g, a, r, s2, done):
        key = (s, g)
        target = r if done else r + gamma * np.max(Q[(s2, g)])
        Q[key][a] += lr * (target - Q[key][a])

    # Goals of increasing difficulty
    goals = [(3, 3), (5, 5), (7, 7)]
    size = 8

    for goal in goals:
        # Reset Q-tables for fair comparison
        Q_her_local = defaultdict(lambda: np.zeros(4))
        Q_no_local = defaultdict(lambda: np.zeros(4))

        for ep in range(500):
            env = GoalGridWorld(size, seed=ep)
            s = env.reset(goal=goal)
            trajectory = [(s, None, None)]

            done = False
            while not done:
                a = select_action(Q_her_local, s, goal, epsilon)
                s2, r, done = env.step(a)
                trajectory.append((s2, a, r))

                # Standard update (with and without HER)
                q_update(Q_no_local, s, goal, a, r, s2, done)
                q_update(Q_her_local, s, goal, a, r, s2, done)
                s = s2

            # HER: relabel with achieved states as goals
            states_in_traj = [t[0] for t in trajectory]
            for i in range(1, len(trajectory)):
                s_i, a_i, _ = trajectory[i]
                s_prev = trajectory[i-1][0]
                # Sample future achieved state as alternative goal
                if i < len(trajectory) - 1:
                    future_idx = np.random.randint(i, len(trajectory))
                    new_goal = states_in_traj[future_idx]
                    new_r = 0.0 if s_i == new_goal else -1.0
                    new_done = s_i == new_goal
                    q_update(Q_her_local, s_prev, new_goal, a_i,
                             new_r, s_i, new_done)

        # Evaluate
        eps_eval = 0.05
        her_succ, no_succ = [], []
        for ep in range(50):
            env = GoalGridWorld(size, seed=ep + 5000)
            s = env.reset(goal=goal)
            done = False
            while not done:
                a = select_action(Q_her_local, s, goal, eps_eval)
                s, _, done = env.step(a)
            her_succ.append(s == goal)

            env = GoalGridWorld(size, seed=ep + 5000)
            s = env.reset(goal=goal)
            done = False
            while not done:
                a = select_action(Q_no_local, s, goal, eps_eval)
                s, _, done = env.step(a)
            no_succ.append(s == goal)

        print(f"  Goal {goal}: HER={np.mean(her_succ):.0%}, "
              f"No HER={np.mean(no_succ):.0%}")

    print(f"\n  HER turns failures into training data by relabeling goals")
    print(f"  Biggest benefit on harder (distant) goals where standard RL fails")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
