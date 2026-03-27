"""
Hierarchical Reinforcement Learning — Example Code

Demonstrates HRL concepts:
  1. Options framework with SMDP Q-learning in a 4-room GridWorld
  2. Goal-conditioned hierarchical agent (simplified HIRO)
  3. Feudal manager-worker architecture

All environments are self-contained — no gym/gymnasium dependency.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# 1. Four-Room GridWorld
# ============================================================
class FourRoomGrid:
    """Four-room GridWorld for hierarchical RL experiments.

    Layout (13x13):
      #############
      #     #     #
      #     #     #
      #           #
      #     #     #
      #     #     #
      ######+######
      #     #     #
      #     #     #
      #           #
      #     #     #
      #     #     #
      #############

    Four rooms connected by hallways ('+' marks doors).
    """

    def __init__(self, goal=None, seed=None):
        self.size = 13
        self.rng = np.random.RandomState(seed)

        # Build walls
        self.walls = set()
        # Outer walls
        for i in range(self.size):
            self.walls.add((0, i))
            self.walls.add((self.size-1, i))
            self.walls.add((i, 0))
            self.walls.add((i, self.size-1))
        # Middle horizontal wall
        for c in range(1, self.size - 1):
            if c != 6:  # door at column 6
                self.walls.add((6, c))
        # Middle vertical wall
        for r in range(1, self.size - 1):
            if r != 3 and r != 9:  # doors at rows 3 and 9
                self.walls.add((r, 6))

        self.doors = [(6, 6), (3, 6), (9, 6)]  # hallway positions
        self.goal = goal or (11, 11)
        self.state = (1, 1)
        self.steps = 0
        self.max_steps = 500
        self.n_actions = 4

    def reset(self):
        self.state = (1, 1)
        self.steps = 0
        return self.state

    def step(self, action):
        self.steps += 1
        r, c = self.state
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nr, nc = r + dr, c + dc

        if (nr, nc) not in self.walls:
            self.state = (nr, nc)

        done = (self.state == self.goal) or (self.steps >= self.max_steps)
        reward = 1.0 if self.state == self.goal else -0.01
        return self.state, reward, done

    def get_room(self, state):
        """Return which room a state belongs to (0-3)."""
        r, c = state
        if r <= 6 and c <= 6:
            return 0  # top-left
        elif r <= 6 and c > 6:
            return 1  # top-right
        elif r > 6 and c <= 6:
            return 2  # bottom-left
        else:
            return 3  # bottom-right


# ============================================================
# 2. Options Framework Implementation
# ============================================================
class GoToHallwayOption:
    """Option: navigate to a specific hallway (door) position.

    Terminates when the agent reaches the target hallway
    or exceeds the step limit.
    """

    def __init__(self, target_hallway, name):
        self.target = target_hallway
        self.name = name
        self.max_steps = 100
        self.Q = defaultdict(lambda: np.zeros(4))
        self.trained = False

    def can_initiate(self, state):
        """Can start from any non-wall state."""
        return True

    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(4)
        return int(np.argmax(self.Q[state]))

    def should_terminate(self, state, steps):
        return state == self.target or steps >= self.max_steps

    def train(self, env_class, episodes=500, lr=0.2, gamma=0.99):
        """Pre-train option policy to reach target hallway."""
        for _ in range(episodes):
            env = env_class(goal=self.target, seed=None)
            state = env.reset()
            # Random start position
            while True:
                r = np.random.randint(1, 12)
                c = np.random.randint(1, 12)
                if (r, c) not in env.walls:
                    state = (r, c)
                    env.state = state
                    break

            for step in range(self.max_steps):
                action = self.get_action(state, epsilon=0.3)
                next_state, _, done = env.step(action)
                # Dense reward for getting closer to target
                dist_before = abs(state[0] - self.target[0]) + abs(
                    state[1] - self.target[1])
                dist_after = abs(next_state[0] - self.target[0]) + abs(
                    next_state[1] - self.target[1])
                reward = (dist_before - dist_after) * 0.1
                if next_state == self.target:
                    reward = 1.0
                    done = True

                target_q = reward
                if not done:
                    target_q += gamma * np.max(self.Q[next_state])
                self.Q[state][action] += lr * (
                    target_q - self.Q[state][action])
                state = next_state
                if done:
                    break
        self.trained = True


def demo_options_framework():
    """Options framework with SMDP Q-learning in 4-room GridWorld."""
    print("=" * 60)
    print("Demo 1: Options Framework — 4-Room GridWorld")
    print("=" * 60)

    # Define options (one per hallway/door)
    hallways = [(6, 6), (3, 6), (9, 6)]
    options = []
    for hw in hallways:
        opt = GoToHallwayOption(hw, f"go_to_{hw}")
        print(f"  Training option: go to hallway {hw}...", end=" ")
        opt.train(FourRoomGrid, episodes=300)
        print("done")
        options.append(opt)

    # Add primitive actions as options (for fine-grained control)
    # (omitted for simplicity — hallway options suffice)

    # SMDP Q-learning over options
    Q_options = defaultdict(lambda: np.zeros(len(options)))
    gamma = 0.99
    lr = 0.1

    print("\nTraining policy over options...")
    successes = []
    for episode in range(300):
        env = FourRoomGrid(goal=(11, 11), seed=episode)
        state = env.reset()
        done = False

        while not done:
            # Epsilon-greedy option selection
            if np.random.random() < 0.2:
                opt_idx = np.random.randint(len(options))
            else:
                opt_idx = int(np.argmax(Q_options[state]))

            option = options[opt_idx]
            start_state = state
            total_reward = 0.0
            discount = 1.0
            steps = 0

            # Execute option
            while not done:
                action = option.get_action(state, epsilon=0.05)
                next_state, reward, done = env.step(action)
                total_reward += discount * reward
                discount *= gamma
                steps += 1
                state = next_state
                if option.should_terminate(state, steps):
                    break

            # SMDP Q-learning update
            if done:
                target = total_reward
            else:
                target = total_reward + (gamma ** steps) * np.max(
                    Q_options[state])
            Q_options[start_state][opt_idx] += lr * (
                target - Q_options[start_state][opt_idx])

        successes.append(env.state == env.goal)

        if (episode + 1) % 100 == 0:
            rate = np.mean(successes[-50:])
            print(f"  Episode {episode+1}: success rate = {rate:.0%}")

    # Compare with flat Q-learning
    print("\nFlat Q-learning comparison...")
    flat_Q = defaultdict(lambda: np.zeros(4))
    flat_successes = []
    for episode in range(300):
        env = FourRoomGrid(goal=(11, 11), seed=episode)
        state = env.reset()
        done = False
        while not done:
            if np.random.random() < 0.2:
                action = np.random.randint(4)
            else:
                action = int(np.argmax(flat_Q[state]))
            next_state, reward, done = env.step(action)
            target = reward
            if not done:
                target += gamma * np.max(flat_Q[next_state])
            flat_Q[state][action] += lr * (
                target - flat_Q[state][action])
            state = next_state
        flat_successes.append(env.state == env.goal)

    print(f"\n  Results after 300 episodes:")
    print(f"    Options (HRL):    {np.mean(successes[-50:]):.0%} success")
    print(f"    Flat Q-learning:  {np.mean(flat_successes[-50:]):.0%} success")


# ============================================================
# 3. Goal-Conditioned Hierarchical Agent (simplified HIRO)
# ============================================================
class GoalConditioned2D:
    """Simple 2D continuous environment for goal-conditioned HRL.

    Agent moves in a 2D plane. High-level sets sub-goals,
    low-level moves toward sub-goals.
    """

    def __init__(self, goal=None, seed=None):
        self.rng = np.random.RandomState(seed)
        self.pos = np.array([0.0, 0.0])
        self.final_goal = goal if goal is not None else np.array([8.0, 8.0])
        self.steps = 0
        self.max_steps = 200

    def reset(self):
        self.pos = np.array([0.0, 0.0])
        self.steps = 0
        return self.pos.copy()

    def step(self, action):
        """Action: 2D velocity vector clipped to [-1, 1]."""
        action = np.clip(action, -1.0, 1.0)
        self.pos = self.pos + action * 0.5
        self.pos = np.clip(self.pos, -2.0, 12.0)
        self.steps += 1
        dist = np.linalg.norm(self.pos - self.final_goal)
        done = dist < 0.5 or self.steps >= self.max_steps
        reward = 1.0 if dist < 0.5 else 0.0  # sparse reward
        return self.pos.copy(), reward, done


def demo_goal_conditioned():
    """Simplified HIRO: high-level goals + low-level controller."""
    print("\n" + "=" * 60)
    print("Demo 2: Goal-Conditioned Hierarchy (HIRO-style)")
    print("=" * 60)

    final_goal = np.array([8.0, 8.0])
    subgoal_horizon = 10  # high-level sets goal every 10 steps

    # Simple low-level policy: move toward sub-goal with noise
    def low_level_action(pos, subgoal, noise=0.3):
        direction = subgoal - pos
        norm = np.linalg.norm(direction) + 1e-8
        action = direction / norm  # unit vector toward subgoal
        action += np.random.randn(2) * noise  # exploration noise
        return np.clip(action, -1, 1)

    # High-level: learn which sub-goals lead to final goal
    # Discretize sub-goal space for simplicity
    subgoal_grid = []
    for x in range(0, 10, 2):
        for y in range(0, 10, 2):
            subgoal_grid.append(np.array([float(x), float(y)]))

    # Q-values for (position_bucket, subgoal_idx)
    Q_high = defaultdict(lambda: np.zeros(len(subgoal_grid)))

    def bucket_pos(pos):
        return (int(np.clip(pos[0], 0, 9)),
                int(np.clip(pos[1], 0, 9)))

    print("\nTraining hierarchical agent...")
    successes = []
    gamma = 0.99
    lr_high = 0.1

    for episode in range(500):
        env = GoalConditioned2D(goal=final_goal, seed=episode)
        pos = env.reset()
        done = False
        total_steps = 0

        while not done:
            # High-level: select sub-goal
            bucket = bucket_pos(pos)
            if np.random.random() < 0.3:
                sg_idx = np.random.randint(len(subgoal_grid))
            else:
                sg_idx = int(np.argmax(Q_high[bucket]))
            subgoal = subgoal_grid[sg_idx]

            start_pos = pos.copy()
            sub_reward = 0.0
            sub_steps = 0

            # Low-level: move toward sub-goal for c steps
            for _ in range(subgoal_horizon):
                action = low_level_action(pos, subgoal,
                                          noise=max(0.1, 0.5 - episode/500))
                pos, reward, done = env.step(action)
                sub_reward += reward
                sub_steps += 1
                total_steps += 1

                # Low-level intrinsic reward: distance to subgoal
                if np.linalg.norm(pos - subgoal) < 1.0:
                    break
                if done:
                    break

            # High-level Q-learning update
            next_bucket = bucket_pos(pos)
            # Reward: reaching final goal or getting closer
            dist_before = np.linalg.norm(start_pos - final_goal)
            dist_after = np.linalg.norm(pos - final_goal)
            high_reward = sub_reward + (dist_before - dist_after) * 0.1

            target = high_reward
            if not done:
                target += (gamma ** sub_steps) * np.max(Q_high[next_bucket])
            Q_high[bucket][sg_idx] += lr_high * (
                target - Q_high[bucket][sg_idx])

        successes.append(np.linalg.norm(pos - final_goal) < 0.5)

        if (episode + 1) % 100 == 0:
            rate = np.mean(successes[-50:])
            print(f"  Episode {episode+1}: success = {rate:.0%}")

    # Compare with flat agent
    print("\nFlat agent comparison...")
    flat_successes = []
    for episode in range(500):
        env = GoalConditioned2D(goal=final_goal, seed=episode)
        pos = env.reset()
        done = False
        while not done:
            # Random exploration (no hierarchy)
            action = np.random.randn(2) * 0.5
            action = np.clip(action, -1, 1)
            pos, reward, done = env.step(action)
        flat_successes.append(np.linalg.norm(pos - final_goal) < 0.5)

    print(f"\n  Results (reaching goal at [8, 8]):")
    print(f"    Hierarchical: {np.mean(successes[-100:]):.0%}")
    print(f"    Flat random:  {np.mean(flat_successes[-100:]):.0%}")


# ============================================================
# 4. Manager-Worker (Feudal) Demo
# ============================================================
def demo_feudal():
    """Simplified feudal manager-worker architecture."""
    print("\n" + "=" * 60)
    print("Demo 3: Feudal Manager-Worker Architecture")
    print("=" * 60)

    env = FourRoomGrid(goal=(11, 11))
    manager_interval = 5  # manager acts every 5 steps

    # Manager: selects direction (up/down/left/right) as a goal
    directions = {
        0: np.array([-1, 0]),  # up
        1: np.array([1, 0]),   # down
        2: np.array([0, -1]),  # left
        3: np.array([0, 1]),   # right
    }

    Q_manager = defaultdict(lambda: np.zeros(4))
    Q_worker = defaultdict(lambda: np.zeros(4))
    gamma = 0.99
    lr = 0.15

    print("\nTraining feudal agent...")
    successes = []

    for episode in range(400):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Manager decides direction
            room = env.get_room(state)
            if np.random.random() < 0.2:
                mgr_action = np.random.randint(4)
            else:
                mgr_action = int(np.argmax(Q_manager[room]))
            goal_dir = directions[mgr_action]

            mgr_start_state = state
            mgr_start_room = room
            mgr_reward = 0.0
            mgr_steps = 0

            # Worker follows manager's direction for several steps
            for _ in range(manager_interval):
                # Worker action: move in direction closest to goal_dir
                worker_key = (state, tuple(goal_dir))
                if np.random.random() < 0.15:
                    w_action = np.random.randint(4)
                else:
                    w_action = int(np.argmax(Q_worker[worker_key]))

                next_state, reward, done = env.step(w_action)
                mgr_reward += reward
                mgr_steps += 1
                total_reward += reward

                # Worker intrinsic reward: cosine similarity with goal_dir
                actual_dir = np.array(next_state) - np.array(state)
                if np.linalg.norm(actual_dir) > 0:
                    cos_sim = np.dot(actual_dir, goal_dir) / (
                        np.linalg.norm(actual_dir) * np.linalg.norm(goal_dir)
                        + 1e-8)
                    w_reward = cos_sim * 0.5
                else:
                    w_reward = -0.1

                # Worker Q update
                next_worker_key = (next_state, tuple(goal_dir))
                w_target = w_reward
                if not done:
                    w_target += gamma * np.max(Q_worker[next_worker_key])
                Q_worker[worker_key][w_action] += lr * (
                    w_target - Q_worker[worker_key][w_action])

                state = next_state
                if done:
                    break

            # Manager Q update
            next_room = env.get_room(state)
            m_target = mgr_reward
            if not done:
                m_target += (gamma ** mgr_steps) * np.max(
                    Q_manager[next_room])
            Q_manager[mgr_start_room][mgr_action] += lr * (
                m_target - Q_manager[mgr_start_room][mgr_action])

        successes.append(env.state == env.goal)
        if (episode + 1) % 100 == 0:
            rate = np.mean(successes[-50:])
            print(f"  Episode {episode+1}: success = {rate:.0%}")

    # Show manager's learned preferences per room
    print("\nManager's preferred directions per room:")
    dir_names = {0: "up", 1: "down", 2: "left", 3: "right"}
    for room in range(4):
        best = int(np.argmax(Q_manager[room]))
        room_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
        print(f"  Room {room} ({room_names[room]}): "
              f"go {dir_names[best]} "
              f"(Q-values: {Q_manager[room].round(2)})")


if __name__ == "__main__":
    demo_options_framework()
    demo_goal_conditioned()
    demo_feudal()
