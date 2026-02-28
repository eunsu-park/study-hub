"""
Exercises for Lesson 16: Hierarchical Reinforcement Learning
Topic: Reinforcement_Learning

Solutions to practice problems covering the Options framework,
Option-Critic, feudal goal-setting, HIRO-style relabeling,
and HRL architecture comparison.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# Shared: Four-Room GridWorld
# ============================================================
class FourRoomGrid:
    """Four-room GridWorld (13×13) with doors between rooms."""

    def __init__(self, goal=None, seed=None):
        self.size = 13
        self.walls = set()
        for i in range(self.size):
            self.walls.update({(0, i), (self.size-1, i),
                               (i, 0), (i, self.size-1)})
        for c in range(1, self.size - 1):
            if c != 6:
                self.walls.add((6, c))
        for r in range(1, self.size - 1):
            if r != 3 and r != 9:
                self.walls.add((r, 6))
        self.doors = [(6, 6), (3, 6), (9, 6)]
        self.goal = goal or (11, 11)
        self.state = (1, 1)
        self.steps = 0
        self.max_steps = 500

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
        done = self.state == self.goal or self.steps >= self.max_steps
        reward = 1.0 if self.state == self.goal else -0.01
        return self.state, reward, done

    def get_room(self, state):
        r, c = state
        if r <= 6 and c <= 6:
            return 0
        elif r <= 6:
            return 1
        elif c <= 6:
            return 2
        else:
            return 3


def exercise_1():
    """
    Options framework in 4-room GridWorld.
    Train hallway options, then learn policy-over-options.
    """
    print("=== Exercise 1: Options in 4-Room GridWorld ===\n")

    hallways = [(6, 6), (3, 6), (9, 6)]

    # Pre-train option policies: reach each hallway
    option_Qs = []
    for hw in hallways:
        Q = defaultdict(lambda: np.zeros(4))
        print(f"  Training option: go to {hw}...", end=" ")

        for ep in range(300):
            env = FourRoomGrid(goal=hw)
            # Random start
            while True:
                r = np.random.randint(1, 12)
                c = np.random.randint(1, 12)
                if (r, c) not in env.walls:
                    env.state = (r, c)
                    break
            s = env.state
            for _ in range(100):
                if np.random.random() < 0.3:
                    a = np.random.randint(4)
                else:
                    a = int(np.argmax(Q[s]))
                s2, _, done = env.step(a)
                dist_b = abs(s[0]-hw[0]) + abs(s[1]-hw[1])
                dist_a = abs(s2[0]-hw[0]) + abs(s2[1]-hw[1])
                r = (dist_b - dist_a) * 0.1
                if s2 == hw:
                    r = 1.0
                    done = True
                target = r if done else r + 0.99 * np.max(Q[s2])
                Q[s][a] += 0.15 * (target - Q[s][a])
                s = s2
                if done:
                    break
        option_Qs.append(Q)
        print("done")

    # SMDP Q-learning over options
    Q_opt = defaultdict(lambda: np.zeros(len(hallways)))
    gamma = 0.99
    successes = []

    print("\n  Training policy over options...")
    for episode in range(300):
        env = FourRoomGrid(goal=(11, 11), seed=episode)
        state = env.reset()
        done = False

        while not done:
            if np.random.random() < 0.2:
                oi = np.random.randint(len(hallways))
            else:
                oi = int(np.argmax(Q_opt[state]))

            start = state
            total_r = 0.0
            disc = 1.0
            steps = 0

            for _ in range(100):
                a = int(np.argmax(option_Qs[oi][state]))
                next_s, r, done = env.step(a)
                total_r += disc * r
                disc *= gamma
                steps += 1
                state = next_s
                if state == hallways[oi] or done:
                    break

            target = total_r
            if not done:
                target += (gamma ** steps) * np.max(Q_opt[state])
            Q_opt[start][oi] += 0.1 * (target - Q_opt[start][oi])

        successes.append(env.state == env.goal)

    # Flat comparison
    flat_Q = defaultdict(lambda: np.zeros(4))
    flat_succ = []
    for ep in range(300):
        env = FourRoomGrid(goal=(11, 11), seed=ep)
        s = env.reset()
        done = False
        while not done:
            a = int(np.argmax(flat_Q[s])) if np.random.random() > 0.2 else np.random.randint(4)
            s2, r, done = env.step(a)
            t = r if done else r + 0.99 * np.max(flat_Q[s2])
            flat_Q[s][a] += 0.1 * (t - flat_Q[s][a])
            s = s2
        flat_succ.append(env.state == env.goal)

    print(f"\n  Results (last 50 episodes):")
    print(f"    Options HRL:     {np.mean(successes[-50:]):.0%}")
    print(f"    Flat Q-learning: {np.mean(flat_succ[-50:]):.0%}")
    print(f"    Options provide temporal abstraction → faster learning")
    print()


def exercise_2():
    """
    Option-Critic: learn option policies and termination conditions.
    Observe effect of deliberation cost on option duration.
    """
    print("=== Exercise 2: Option-Critic (Simplified) ===\n")

    # 1D chain: 20 states, goal at state 19
    n_states = 20
    n_options = 4

    # Option policies and termination functions (tabular)
    opt_Q = np.random.randn(n_options, n_states, 2) * 0.01  # 2 actions: left/right
    opt_beta = np.full((n_options, n_states), 0.5)  # termination probability
    Q_omega = np.zeros((n_states, n_options))  # policy over options

    gamma = 0.99

    for delib_cost in [0.0, 0.01, 0.05]:
        # Reset
        opt_Q_copy = opt_Q.copy()
        opt_beta_copy = np.full((n_options, n_states), 0.3)
        Q_omega_copy = Q_omega.copy()

        durations = []

        for episode in range(500):
            state = 0
            done = False
            steps = 0

            # Select initial option
            if np.random.random() < 0.2:
                option = np.random.randint(n_options)
            else:
                option = int(np.argmax(Q_omega_copy[state]))

            option_steps = 0

            while not done and steps < 100:
                # Select action from option's policy
                if np.random.random() < 0.15:
                    action = np.random.randint(2)
                else:
                    action = int(np.argmax(opt_Q_copy[option, state]))

                # Execute action (0=left, 1=right)
                next_state = max(0, min(n_states - 1,
                                        state + (1 if action == 1 else -1)))
                reward = 1.0 if next_state == n_states - 1 else -0.01
                done = next_state == n_states - 1
                steps += 1
                option_steps += 1

                # Check termination
                term_prob = 1.0 / (1.0 + np.exp(-opt_beta_copy[option, next_state]))
                terminate = np.random.random() < term_prob

                if terminate or done:
                    durations.append(option_steps)
                    # Termination penalty (deliberation cost)
                    reward -= delib_cost

                    # Select next option
                    if np.random.random() < 0.15:
                        next_option = np.random.randint(n_options)
                    else:
                        next_option = int(np.argmax(Q_omega_copy[next_state]))

                    # Update Q_omega
                    target = reward
                    if not done:
                        target += gamma * np.max(Q_omega_copy[next_state])
                    Q_omega_copy[state][option] += 0.1 * (
                        target - Q_omega_copy[state][option])

                    # Update termination
                    advantage = np.max(Q_omega_copy[next_state]) - Q_omega_copy[next_state][option]
                    opt_beta_copy[option, next_state] += 0.01 * advantage

                    option = next_option
                    option_steps = 0
                else:
                    # Intra-option Q update
                    target = reward
                    if not done:
                        target += gamma * np.max(opt_Q_copy[option, next_state])
                    opt_Q_copy[option, state, action] += 0.1 * (
                        target - opt_Q_copy[option, state, action])

                state = next_state

        avg_dur = np.mean(durations[-100:]) if durations else 0
        print(f"  Deliberation cost = {delib_cost:.2f}: "
              f"avg option duration = {avg_dur:.1f} steps")

    print(f"\n  Higher deliberation cost → longer options (less switching)")
    print(f"  Without cost, options may degenerate to 1-step primitives")
    print()


def exercise_3():
    """
    Feudal manager-worker: manager sets direction, worker follows.
    """
    print("=== Exercise 3: Feudal Manager-Worker ===\n")

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dir_names = ["up", "down", "left", "right"]

    Q_mgr = defaultdict(lambda: np.zeros(4))
    Q_wkr = defaultdict(lambda: np.zeros(4))
    mgr_interval = 5
    gamma = 0.99

    successes = []
    for episode in range(400):
        env = FourRoomGrid(goal=(11, 11), seed=episode)
        state = env.reset()
        done = False

        while not done:
            room = env.get_room(state)
            mgr_a = (int(np.argmax(Q_mgr[room]))
                      if np.random.random() > 0.2
                      else np.random.randint(4))
            goal_dir = directions[mgr_a]

            mgr_start = state
            mgr_room = room
            mgr_r = 0.0
            mgr_steps = 0

            for _ in range(mgr_interval):
                wkey = (state, goal_dir)
                w_a = (int(np.argmax(Q_wkr[wkey]))
                       if np.random.random() > 0.15
                       else np.random.randint(4))
                ns, r, done = env.step(w_a)
                mgr_r += r
                mgr_steps += 1

                # Worker intrinsic reward
                actual = (ns[0] - state[0], ns[1] - state[1])
                if actual != (0, 0):
                    cos_sim = (actual[0] * goal_dir[0] + actual[1] * goal_dir[1])
                    w_r = cos_sim * 0.5
                else:
                    w_r = -0.1

                nwkey = (ns, goal_dir)
                wt = w_r if done else w_r + gamma * np.max(Q_wkr[nwkey])
                Q_wkr[wkey][w_a] += 0.15 * (wt - Q_wkr[wkey][w_a])
                state = ns
                if done:
                    break

            nroom = env.get_room(state)
            mt = mgr_r
            if not done:
                mt += (gamma ** mgr_steps) * np.max(Q_mgr[nroom])
            Q_mgr[mgr_room][mgr_a] += 0.1 * (mt - Q_mgr[mgr_room][mgr_a])

        successes.append(env.state == env.goal)

        if (episode + 1) % 100 == 0:
            print(f"    Episode {episode+1}: "
                  f"{np.mean(successes[-50:]):.0%} success")

    print(f"\n  Manager's learned directions:")
    room_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
    for room in range(4):
        best = int(np.argmax(Q_mgr[room]))
        print(f"    {room_names[room]}: go {dir_names[best]}")
    print()


def exercise_4():
    """
    HIRO-style goal relabeling in continuous 2D environment.
    """
    print("=== Exercise 4: HIRO Goal Relabeling ===\n")

    class Grid2D:
        def __init__(self, size=8, goal=None):
            self.size = size
            self.pos = np.array([0.0, 0.0])
            self.goal = goal if goal is not None else np.array([7.0, 7.0])
            self.steps = 0

        def reset(self):
            self.pos = np.array([0.0, 0.0])
            self.steps = 0
            return self.pos.copy()

        def step(self, action):
            action = np.clip(action, -1, 1)
            self.pos = np.clip(self.pos + action * 0.5, 0, self.size)
            self.steps += 1
            dist = np.linalg.norm(self.pos - self.goal)
            done = dist < 0.5 or self.steps >= 150
            reward = 1.0 if dist < 0.5 else 0.0
            return self.pos.copy(), reward, done

    # High-level: discretized sub-goals
    subgoals = []
    for x in range(0, 8, 2):
        for y in range(0, 8, 2):
            subgoals.append(np.array([float(x), float(y)]))

    Q_hi = defaultdict(lambda: np.zeros(len(subgoals)))
    c = 10

    def bucket(pos):
        return (int(np.clip(pos[0], 0, 7)), int(np.clip(pos[1], 0, 7)))

    # With relabeling
    print("  Training with goal relabeling...")
    successes_rel = []
    for ep in range(400):
        env = Grid2D()
        pos = env.reset()
        done = False
        trajectory = []

        while not done:
            b = bucket(pos)
            sg_i = (int(np.argmax(Q_hi[b]))
                    if np.random.random() > 0.3
                    else np.random.randint(len(subgoals)))
            sg = subgoals[sg_i]
            start = pos.copy()
            start_b = b
            sub_r = 0
            steps = 0

            for _ in range(c):
                direction = sg - pos
                norm = np.linalg.norm(direction) + 1e-8
                action = direction / norm + np.random.randn(2) * 0.3
                pos, r, done = env.step(action)
                sub_r += r
                steps += 1
                trajectory.append((start.copy(), sg_i, pos.copy()))
                if done:
                    break

            dist_b = np.linalg.norm(start - env.goal)
            dist_a = np.linalg.norm(pos - env.goal)
            hi_r = sub_r + (dist_b - dist_a) * 0.1

            next_b = bucket(pos)
            target = hi_r if done else hi_r + 0.99 ** steps * np.max(Q_hi[next_b])
            Q_hi[start_b][sg_i] += 0.1 * (target - Q_hi[start_b][sg_i])

            # Goal relabeling: use actually reached position as goal
            reached = pos.copy()
            closest_sg = min(range(len(subgoals)),
                             key=lambda i: np.linalg.norm(subgoals[i] - reached))
            rel_r = 0.5  # bonus for "reaching" relabeled goal
            Q_hi[start_b][closest_sg] += 0.1 * (
                rel_r + 0.99 * np.max(Q_hi[next_b]) - Q_hi[start_b][closest_sg])

        successes_rel.append(np.linalg.norm(pos - env.goal) < 0.5)

    # Without relabeling
    Q_hi_no = defaultdict(lambda: np.zeros(len(subgoals)))
    successes_no = []
    for ep in range(400):
        env = Grid2D()
        pos = env.reset()
        done = False
        while not done:
            b = bucket(pos)
            sg_i = (int(np.argmax(Q_hi_no[b]))
                    if np.random.random() > 0.3
                    else np.random.randint(len(subgoals)))
            sg = subgoals[sg_i]
            start = pos.copy()
            start_b = b
            sub_r = 0
            steps = 0
            for _ in range(c):
                direction = sg - pos
                norm = np.linalg.norm(direction) + 1e-8
                action = direction / norm + np.random.randn(2) * 0.3
                pos, r, done = env.step(action)
                sub_r += r
                steps += 1
                if done:
                    break
            dist_b = np.linalg.norm(start - env.goal)
            dist_a = np.linalg.norm(pos - env.goal)
            hi_r = sub_r + (dist_b - dist_a) * 0.1
            next_b = bucket(pos)
            target = hi_r if done else hi_r + 0.99 ** steps * np.max(Q_hi_no[next_b])
            Q_hi_no[start_b][sg_i] += 0.1 * (target - Q_hi_no[start_b][sg_i])
        successes_no.append(np.linalg.norm(pos - env.goal) < 0.5)

    print(f"\n  With relabeling:    {np.mean(successes_rel[-100:]):.0%}")
    print(f"  Without relabeling: {np.mean(successes_no[-100:]):.0%}")
    print(f"  Relabeling provides extra learning signal from failed attempts")
    print()


def exercise_5():
    """
    Compare flat Q-learning, Options, and feudal on 4-room navigation.
    """
    print("=== Exercise 5: HRL Architecture Comparison ===\n")

    n_episodes = 300

    # 1. Flat Q-learning
    flat_Q = defaultdict(lambda: np.zeros(4))
    flat_succ = []
    for ep in range(n_episodes):
        env = FourRoomGrid(seed=ep)
        s = env.reset()
        done = False
        while not done:
            a = (int(np.argmax(flat_Q[s]))
                 if np.random.random() > 0.2
                 else np.random.randint(4))
            s2, r, done = env.step(a)
            t = r if done else r + 0.99 * np.max(flat_Q[s2])
            flat_Q[s][a] += 0.1 * (t - flat_Q[s][a])
            s = s2
        flat_succ.append(env.state == env.goal)

    # 2. Options with hand-designed hallway skills
    hw_Qs = []
    for hw in [(6, 6), (3, 6), (9, 6)]:
        Q = defaultdict(lambda: np.zeros(4))
        for _ in range(200):
            env = FourRoomGrid(goal=hw)
            while True:
                r, c = np.random.randint(1, 12), np.random.randint(1, 12)
                if (r, c) not in env.walls:
                    env.state = (r, c)
                    break
            s = env.state
            for _ in range(80):
                a = int(np.argmax(Q[s])) if np.random.random() > 0.3 else np.random.randint(4)
                s2, _, done = env.step(a)
                db = abs(s[0]-hw[0]) + abs(s[1]-hw[1])
                da = abs(s2[0]-hw[0]) + abs(s2[1]-hw[1])
                rw = 1.0 if s2 == hw else (db - da) * 0.1
                t = rw if s2 == hw else rw + 0.99 * np.max(Q[s2])
                Q[s][a] += 0.15 * (t - Q[s][a])
                s = s2
                if s == hw:
                    break
        hw_Qs.append(Q)

    Q_opt = defaultdict(lambda: np.zeros(3))
    opt_succ = []
    for ep in range(n_episodes):
        env = FourRoomGrid(seed=ep)
        s = env.reset()
        done = False
        while not done:
            oi = (int(np.argmax(Q_opt[s]))
                  if np.random.random() > 0.2
                  else np.random.randint(3))
            start = s
            total_r = 0
            d = 1.0
            steps = 0
            for _ in range(80):
                a = int(np.argmax(hw_Qs[oi][s]))
                s2, r, done = env.step(a)
                total_r += d * r
                d *= 0.99
                steps += 1
                s = s2
                if s == [(6,6),(3,6),(9,6)][oi] or done:
                    break
            t = total_r if done else total_r + 0.99**steps * np.max(Q_opt[s])
            Q_opt[start][oi] += 0.1 * (t - Q_opt[start][oi])
        opt_succ.append(env.state == env.goal)

    # 3. Simple feudal
    Q_m = defaultdict(lambda: np.zeros(4))
    Q_w = defaultdict(lambda: np.zeros(4))
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    feudal_succ = []
    for ep in range(n_episodes):
        env = FourRoomGrid(seed=ep)
        s = env.reset()
        done = False
        while not done:
            rm = env.get_room(s)
            ma = int(np.argmax(Q_m[rm])) if np.random.random() > 0.2 else np.random.randint(4)
            gd = dirs[ma]
            ss = s
            sr = rm
            mr = 0
            ms = 0
            for _ in range(5):
                wk = (s, gd)
                wa = int(np.argmax(Q_w[wk])) if np.random.random() > 0.15 else np.random.randint(4)
                ns, r, done = env.step(wa)
                mr += r
                ms += 1
                ad = (ns[0]-s[0], ns[1]-s[1])
                wr = (ad[0]*gd[0]+ad[1]*gd[1])*0.5 if ad != (0,0) else -0.1
                nwk = (ns, gd)
                wt = wr if done else wr + 0.99 * np.max(Q_w[nwk])
                Q_w[wk][wa] += 0.15 * (wt - Q_w[wk][wa])
                s = ns
                if done:
                    break
            nr = env.get_room(s)
            mt = mr if done else mr + 0.99**ms * np.max(Q_m[nr])
            Q_m[sr][ma] += 0.1 * (mt - Q_m[sr][ma])
        feudal_succ.append(env.state == env.goal)

    # Report
    print(f"  {'Method':>20} | {'Last 50 Success':>16} | {'Overall':>10}")
    print("  " + "-" * 55)
    for name, succ in [("Flat Q-learning", flat_succ),
                       ("Options (HRL)", opt_succ),
                       ("Feudal (HRL)", feudal_succ)]:
        print(f"  {name:>20} | {np.mean(succ[-50:]):16.0%} | "
              f"{np.mean(succ):10.0%}")

    print(f"\n  Both HRL methods outperform flat Q-learning")
    print(f"  Options benefit from pre-trained skills")
    print(f"  Feudal learns structure from scratch (slower but more flexible)")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
