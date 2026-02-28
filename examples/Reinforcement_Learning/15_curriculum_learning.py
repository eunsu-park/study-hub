"""
Curriculum Learning for RL — Example Code

Demonstrates curriculum learning concepts:
  1. Stage-based curriculum for GridWorld
  2. Automatic difficulty scoring (frontier sampling)
  3. Self-paced continuous difficulty
  4. Reward shaping curriculum (dense → sparse transition)

All environments are self-contained — no gym/gymnasium dependency.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# 1. GridWorld Environment with Configurable Difficulty
# ============================================================
class GridWorld:
    """GridWorld with configurable size, obstacles, and goal position.

    Difficulty increases with grid size, number of obstacles,
    and distance from start to goal.
    """

    def __init__(self, size=5, num_obstacles=0, goal=None, seed=None):
        self.size = size
        self.rng = np.random.RandomState(seed)

        # Place obstacles
        self.obstacles = set()
        while len(self.obstacles) < num_obstacles:
            pos = (self.rng.randint(size), self.rng.randint(size))
            if pos != (0, 0) and pos != (goal or (size-1, size-1)):
                self.obstacles.add(pos)

        self.goal = goal or (size - 1, size - 1)
        self.state = (0, 0)
        self.steps = 0
        self.max_steps = size * size * 2
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right

    def reset(self):
        self.state = (0, 0)
        self.steps = 0
        return self._state_id()

    def step(self, action):
        self.steps += 1
        r, c = self.state
        # 0=up, 1=down, 2=left, 3=right
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nr, nc = r + dr, c + dc

        # Boundary check
        if 0 <= nr < self.size and 0 <= nc < self.size:
            if (nr, nc) not in self.obstacles:
                self.state = (nr, nc)

        done = (self.state == self.goal) or (self.steps >= self.max_steps)
        reward = 1.0 if self.state == self.goal else -0.01
        return self._state_id(), reward, done

    def _state_id(self):
        return self.state[0] * self.size + self.state[1]


# ============================================================
# 2. Q-Learning Agent
# ============================================================
class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99,
                 epsilon=0.2):
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.lr * (target - self.Q[state][action])


# ============================================================
# 3. Stage-Based Curriculum
# ============================================================
def demo_stage_curriculum():
    """Compare curriculum vs direct training on hard GridWorld."""
    print("=" * 60)
    print("Demo 1: Stage-Based Curriculum for GridWorld")
    print("=" * 60)

    stages = [
        {"name": "Easy (5×5, 0 obs)", "size": 5, "obstacles": 0},
        {"name": "Medium (8×8, 3 obs)", "size": 8, "obstacles": 3},
        {"name": "Hard (12×12, 8 obs)", "size": 12, "obstacles": 8},
    ]

    # --- Curriculum training ---
    print("\n--- Curriculum Training ---")
    agent_curriculum = QLearningAgent(144, 4, lr=0.1, epsilon=0.3)
    total_episodes = 0

    for stage in stages:
        env = GridWorld(stage["size"], stage["obstacles"], seed=42)
        agent_curriculum.n_actions = env.n_actions
        successes = []

        for ep in range(300):
            state = env.reset()
            done = False
            while not done:
                action = agent_curriculum.select_action(state)
                next_state, reward, done = env.step(action)
                agent_curriculum.update(state, action, reward,
                                        next_state, done)
                state = next_state
            successes.append(reward > 0)
            total_episodes += 1

        rate = sum(successes[-50:]) / 50 if len(successes) >= 50 else 0
        print(f"  {stage['name']}: {rate:.0%} success after 300 eps")

    # Evaluate on hard task
    env_hard = GridWorld(12, 8, seed=42)
    agent_curriculum.epsilon = 0.05
    curriculum_successes = []
    for _ in range(100):
        state = env_hard.reset()
        done = False
        while not done:
            action = agent_curriculum.select_action(state)
            next_state, reward, done = env_hard.step(action)
            state = next_state
        curriculum_successes.append(reward > 0)

    # --- Direct training on hard task ---
    print("\n--- Direct Training on Hard Task ---")
    agent_direct = QLearningAgent(144, 4, lr=0.1, epsilon=0.3)
    env_hard = GridWorld(12, 8, seed=42)

    for ep in range(900):  # same total episodes
        state = env_hard.reset()
        done = False
        while not done:
            action = agent_direct.select_action(state)
            next_state, reward, done = env_hard.step(action)
            agent_direct.update(state, action, reward, next_state, done)
            state = next_state

    agent_direct.epsilon = 0.05
    direct_successes = []
    env_hard = GridWorld(12, 8, seed=42)
    for _ in range(100):
        state = env_hard.reset()
        done = False
        while not done:
            action = agent_direct.select_action(state)
            next_state, reward, done = env_hard.step(action)
            state = next_state
        direct_successes.append(reward > 0)

    print(f"\n  Results on hard task (12×12, 8 obstacles):")
    print(f"    Curriculum: {sum(curriculum_successes):.0%} success")
    print(f"    Direct:     {sum(direct_successes):.0%} success")
    print(f"    Both used {total_episodes} total training episodes")


# ============================================================
# 4. Automatic Difficulty Scoring
# ============================================================
def demo_automatic_curriculum():
    """Automatic task sampling at the learning frontier."""
    print("\n" + "=" * 60)
    print("Demo 2: Automatic Curriculum (Frontier Sampling)")
    print("=" * 60)

    # Define task space: grid sizes from 4 to 15
    task_space = []
    for size in range(4, 16):
        n_obs = max(0, (size - 4))
        task_space.append({
            "id": f"grid_{size}",
            "size": size,
            "obstacles": n_obs,
        })

    # Track success rates
    task_history = defaultdict(list)
    agent = QLearningAgent(225, 4, lr=0.1, epsilon=0.3)

    target_success = 0.5  # sample tasks with ~50% success

    print(f"\nTraining with frontier sampling (target success = "
          f"{target_success:.0%}):")
    print(f"{'Episode':>8} | {'Task Sampled':>15} | {'Success':>8} | "
          f"{'Task Success Rate':>18}")
    print("-" * 60)

    for episode in range(500):
        # Compute sampling scores
        scores = []
        for task in task_space:
            history = task_history[task["id"]]
            if len(history) < 3:
                score = 1.0  # explore unknown tasks
            else:
                rate = np.mean(history[-20:])
                score = max(0.05, 1.0 - abs(rate - target_success) * 2)
            scores.append(score)

        probs = np.array(scores) / sum(scores)
        idx = np.random.choice(len(task_space), p=probs)
        task = task_space[idx]

        # Train on selected task
        env = GridWorld(task["size"], task["obstacles"], seed=episode)
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        success = reward > 0
        task_history[task["id"]].append(float(success))

        if (episode + 1) % 100 == 0:
            rate = np.mean(task_history[task["id"]][-20:])
            print(f"{episode+1:8d} | {task['id']:>15} | "
                  f"{'Yes' if success else 'No':>8} | {rate:18.0%}")

    # Show learned difficulty ordering
    print("\nDiscovered difficulty ordering:")
    print(f"  {'Task':>10} | {'Success Rate':>12} | {'Episodes':>8}")
    print("  " + "-" * 40)
    for task in task_space:
        h = task_history[task["id"]]
        if h:
            rate = np.mean(h)
            print(f"  {task['id']:>10} | {rate:12.0%} | {len(h):8d}")


# ============================================================
# 5. Reward Shaping Curriculum
# ============================================================
def demo_reward_shaping():
    """Demonstrate dense → sparse reward transition."""
    print("\n" + "=" * 60)
    print("Demo 3: Reward Shaping Curriculum (Dense → Sparse)")
    print("=" * 60)

    size = 10
    goal = (9, 9)

    def dense_reward(state_id, goal, size):
        """Distance-based dense reward (always available)."""
        r, c = state_id // size, state_id % size
        gr, gc = goal
        dist = abs(r - gr) + abs(c - gc)
        max_dist = 2 * (size - 1)
        return -dist / max_dist  # normalize to [-1, 0]

    def sparse_reward(state_id, goal, size):
        """Sparse reward: +1 at goal, -0.01 otherwise."""
        r, c = state_id // size, state_id % size
        return 1.0 if (r, c) == goal else -0.01

    # Train with three reward strategies
    strategies = {
        "Sparse only": lambda s, ep: sparse_reward(s, goal, size),
        "Dense only": lambda s, ep: dense_reward(s, goal, size),
        "Curriculum": lambda s, ep: (
            # Blend: starts dense, transitions to sparse over 300 eps
            max(0, 1 - ep / 300) * dense_reward(s, goal, size) +
            min(1, ep / 300) * sparse_reward(s, goal, size)
        ),
    }

    results = {}
    for name, reward_fn in strategies.items():
        agent = QLearningAgent(size * size, 4, lr=0.1, epsilon=0.3)
        env = GridWorld(size, num_obstacles=5, goal=goal, seed=42)
        successes = []

        for ep in range(500):
            state = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, raw_reward, done = env.step(action)
                shaped_reward = reward_fn(next_state, ep)
                agent.update(state, action, shaped_reward,
                             next_state, done)
                state = next_state
            successes.append(raw_reward > 0)

        # Evaluate
        agent.epsilon = 0.05
        env = GridWorld(size, num_obstacles=5, goal=goal, seed=42)
        eval_successes = []
        for _ in range(100):
            state = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                state = next_state
            eval_successes.append(reward > 0)

        results[name] = sum(eval_successes) / 100

    print(f"\nResults on 10×10 GridWorld with 5 obstacles:")
    for name, rate in results.items():
        print(f"  {name:<15}: {rate:.0%} success")
    print("\n  Curriculum blends dense exploration signal with true objective")


# ============================================================
# 6. Self-Paced Difficulty Progression
# ============================================================
def demo_self_paced():
    """Self-paced learning with continuous difficulty adjustment."""
    print("\n" + "=" * 60)
    print("Demo 4: Self-Paced Learning")
    print("=" * 60)

    difficulty = 0.0  # starts easy
    step_up = 0.05
    step_down = 0.1
    target_return = 0.6

    agent = QLearningAgent(400, 4, lr=0.1, epsilon=0.25)
    recent_successes = []
    difficulty_history = []

    print(f"\n{'Episode':>8} | {'Difficulty':>10} | {'Grid Size':>10} | "
          f"{'Obstacles':>10} | {'Success Rate':>12}")
    print("-" * 65)

    for ep in range(600):
        # Map difficulty to environment parameters
        grid_size = int(5 + difficulty * 10)  # 5 to 15
        n_obstacles = int(difficulty * 8)      # 0 to 8

        env = GridWorld(grid_size, n_obstacles, seed=ep)
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        success = reward > 0
        recent_successes.append(float(success))
        if len(recent_successes) > 50:
            recent_successes.pop(0)
        difficulty_history.append(difficulty)

        # Adjust difficulty
        if len(recent_successes) >= 10:
            avg = np.mean(recent_successes[-10:])
            if avg >= target_return:
                difficulty = min(difficulty + step_up, 1.0)
            elif avg < target_return * 0.5:
                difficulty = max(difficulty - step_down, 0.0)

        if (ep + 1) % 100 == 0:
            rate = np.mean(recent_successes[-50:]) if recent_successes else 0
            print(f"{ep+1:8d} | {difficulty:10.2f} | {grid_size:10d} | "
                  f"{n_obstacles:10d} | {rate:12.0%}")

    print(f"\nFinal difficulty reached: {difficulty:.2f}")
    print(f"Difficulty progressed from 0.00 to {max(difficulty_history):.2f}")


if __name__ == "__main__":
    demo_stage_curriculum()
    demo_automatic_curriculum()
    demo_reward_shaping()
    demo_self_paced()
