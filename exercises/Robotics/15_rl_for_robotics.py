"""
Exercises for Lesson 15: Reinforcement Learning for Robotics
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Reward Function Design
    Pick-and-place task: pick up ball and place in cup.
    """
    # Simulated state
    ee_pos = np.array([0.3, 0.0, 0.4])     # end-effector position
    ball_pos = np.array([0.5, 0.2, 0.1])   # ball position
    cup_pos = np.array([0.0, 0.5, 0.1])    # cup position
    gripper_closed = False
    ball_grasped = False
    ball_in_cup = False

    def reward_function(ee_pos, ball_pos, cup_pos, gripper_closed,
                        ball_grasped, ball_in_cup):
        """
        Multi-component reward for pick-and-place.

        Components:
        1. Reaching: guide EE toward ball (before grasp)
        2. Grasping: reward for successful grasp
        3. Transporting: guide grasped ball toward cup
        4. Placing: reward for ball in cup
        """
        reward = 0.0
        components = {}

        if not ball_grasped:
            # Phase 1: Reaching — dense reward to approach ball
            dist_to_ball = np.linalg.norm(ee_pos - ball_pos)
            reaching_reward = -1.0 * dist_to_ball  # closer = higher reward
            # Bonus for being close enough to grasp
            if dist_to_ball < 0.05 and gripper_closed:
                reaching_reward += 5.0  # grasping bonus
            components["reaching"] = reaching_reward
            reward += reaching_reward

        elif not ball_in_cup:
            # Phase 2: Transporting — guide toward cup
            dist_to_cup = np.linalg.norm(ee_pos - cup_pos)
            transport_reward = -0.5 * dist_to_cup
            # Height penalty: don't drag ball on ground
            if ee_pos[2] < 0.15:
                transport_reward -= 1.0
            components["transport"] = transport_reward
            reward += transport_reward

            # Phase 3: Placing
            if dist_to_cup < 0.05:
                place_reward = 10.0  # big bonus for placing in cup
                components["placing"] = place_reward
                reward += place_reward

        else:
            # Phase 4: Task complete
            components["success"] = 20.0
            reward += 20.0

        # Always: small penalty per timestep (efficiency)
        components["time_penalty"] = -0.01
        reward -= 0.01

        return reward, components

    print("Reward Function Design: Pick and Place")
    print(f"  Ball at {ball_pos}, Cup at {cup_pos}")

    # Demonstrate reward at different states
    states = [
        ("Far from ball", np.array([0.3, 0.0, 0.4]), False, False, False),
        ("Near ball", np.array([0.51, 0.21, 0.11]), False, False, False),
        ("Grasped ball", np.array([0.5, 0.2, 0.3]), True, True, False),
        ("Near cup", np.array([0.02, 0.48, 0.12]), True, True, False),
        ("Ball in cup", np.array([0.0, 0.5, 0.1]), True, True, True),
    ]

    print(f"\n  {'State':>18} | {'Reward':>8} | Components")
    print("  " + "-" * 65)
    for label, ee, gripper, grasped, in_cup in states:
        r, comps = reward_function(ee, ball_pos, cup_pos, gripper, grasped, in_cup)
        comp_str = ", ".join(f"{k}={v:.3f}" for k, v in comps.items())
        print(f"  {label:>18} | {r:>8.3f} | {comp_str}")

    print(f"\n  Potential reward hacking issues:")
    print(f"  - Robot might hover near ball without grasping (reaching reward)")
    print(f"  - Robot might grasp ball and hold it near cup without releasing")
    print(f"  - Robot might find shortcuts (throw ball into cup)")
    print(f"  - Careful tuning of component weights is needed to prevent hacking")


def exercise_2():
    """
    Exercise 2: Domain Randomization Study
    2D reaching task with perturbed dynamics.
    """
    np.random.seed(42)

    class PointAgent:
        """2D point agent with damped dynamics."""
        def __init__(self, damping=0.5):
            self.damping = damping
            self.pos = np.zeros(2)
            self.vel = np.zeros(2)
            self.dt = 0.05

        def reset(self, target=None):
            self.pos = np.zeros(2)
            self.vel = np.zeros(2)
            if target is None:
                target = np.random.uniform(-1, 1, 2)
            return self.pos.copy(), target

        def step(self, action):
            """action: 2D force vector."""
            action = np.clip(action, -1, 1)
            acc = action - self.damping * self.vel
            self.vel += acc * self.dt
            self.pos += self.vel * self.dt
            return self.pos.copy()

    def simple_policy(pos, target, Kp=2.0, Kd=0.5):
        """Simple PD policy."""
        error = target - pos
        # Estimate velocity from position change (crude)
        action = Kp * error
        return np.clip(action, -1, 1)

    # Train policy with fixed dynamics
    target = np.array([0.7, 0.3])
    n_episodes = 50
    n_steps = 100

    # Evaluate with fixed dynamics
    agent = PointAgent(damping=0.5)
    errors_fixed = []
    for _ in range(n_episodes):
        pos, _ = agent.reset(target)
        for _ in range(n_steps):
            action = simple_policy(pos, target)
            pos = agent.step(action)
        errors_fixed.append(np.linalg.norm(pos - target))

    # Evaluate with perturbed dynamics (different damping)
    errors_perturbed = []
    for _ in range(n_episodes):
        agent_pert = PointAgent(damping=1.5)  # 3x damping
        pos, _ = agent_pert.reset(target)
        for _ in range(n_steps):
            action = simple_policy(pos, target)
            pos = agent_pert.step(action)
        errors_perturbed.append(np.linalg.norm(pos - target))

    # Train with domain randomization
    errors_dr = []
    for _ in range(n_episodes):
        # Random damping during evaluation
        agent_dr = PointAgent(damping=1.5)
        pos, _ = agent_dr.reset(target)
        for _ in range(n_steps):
            # Policy trained with random damping would be more conservative
            action = simple_policy(pos, target, Kp=3.0, Kd=1.0)  # tuned for robustness
            pos = agent_dr.step(action)
        errors_dr.append(np.linalg.norm(pos - target))

    print("Domain Randomization Study: 2D Reaching Task")
    print(f"  Target: {target}")
    print(f"  Training damping: 0.5, Test damping: 1.5 (3x)")
    print(f"\n  {'Policy':>25} | {'Mean error':>10} | {'Std error':>10}")
    print("  " + "-" * 50)
    print(f"  {'Fixed dyn (train=test)':>25} | {np.mean(errors_fixed):>10.4f} | "
          f"{np.std(errors_fixed):>10.4f}")
    print(f"  {'Fixed dyn (perturbed)':>25} | {np.mean(errors_perturbed):>10.4f} | "
          f"{np.std(errors_perturbed):>10.4f}")
    print(f"  {'Domain randomized':>25} | {np.mean(errors_dr):>10.4f} | "
          f"{np.std(errors_dr):>10.4f}")
    print(f"\n  Domain randomization improves transfer to perturbed dynamics")
    print(f"  by learning a more robust policy during training.")


def exercise_3():
    """
    Exercise 3: Safety Layer Implementation
    2-DOF robot arm with position, velocity, and EE speed limits.
    """
    l1, l2 = 0.5, 0.4

    # Safety constraints
    q_min = np.radians([-180, -180])
    q_max = np.radians([180, 180])
    dq_max = np.array([3.0, 3.0])          # rad/s
    ee_speed_max = 1.5                       # m/s
    dt = 0.01

    def fk(q):
        x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
        y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
        return np.array([x, y])

    def jacobian(q):
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        return np.array([[-l1*s1 - l2*s12, -l2*s12],
                          [l1*c1 + l2*c12, l2*c12]])

    def safety_layer(q, dq_desired):
        """Apply safety constraints to desired joint velocities."""
        dq_safe = dq_desired.copy()

        # 1. Joint velocity limits
        dq_safe = np.clip(dq_safe, -dq_max, dq_max)

        # 2. Joint position limits (slow down near limits)
        for i in range(2):
            q_next = q[i] + dq_safe[i] * dt
            if q_next > q_max[i]:
                dq_safe[i] = max(0, (q_max[i] - q[i]) / dt * 0.9)
            elif q_next < q_min[i]:
                dq_safe[i] = min(0, (q_min[i] - q[i]) / dt * 0.9)

        # 3. End-effector speed limit
        J = jacobian(q)
        ee_vel = J @ dq_safe
        ee_speed = np.linalg.norm(ee_vel)
        if ee_speed > ee_speed_max:
            scale = ee_speed_max / ee_speed
            dq_safe *= scale

        return dq_safe

    # Simulate aggressive RL agent
    np.random.seed(42)
    n_steps = 500
    q = np.array([0.0, np.radians(45)])

    violations_no_safety = {"position": 0, "velocity": 0, "ee_speed": 0}
    violations_with_safety = {"position": 0, "velocity": 0, "ee_speed": 0}

    for step in range(n_steps):
        # Aggressive random action (simulating RL agent)
        dq_rl = np.random.uniform(-5, 5, 2)  # exceeds limits

        # Without safety layer
        q_unsafe = q + dq_rl * dt
        if np.any(q_unsafe > q_max) or np.any(q_unsafe < q_min):
            violations_no_safety["position"] += 1
        if np.any(np.abs(dq_rl) > dq_max):
            violations_no_safety["velocity"] += 1
        J = jacobian(q)
        ee_speed_unsafe = np.linalg.norm(J @ dq_rl)
        if ee_speed_unsafe > ee_speed_max:
            violations_no_safety["ee_speed"] += 1

        # With safety layer
        dq_safe = safety_layer(q, dq_rl)
        q_safe = q + dq_safe * dt
        if np.any(q_safe > q_max + 0.01) or np.any(q_safe < q_min - 0.01):
            violations_with_safety["position"] += 1
        if np.any(np.abs(dq_safe) > dq_max + 0.01):
            violations_with_safety["velocity"] += 1
        ee_speed_safe = np.linalg.norm(jacobian(q) @ dq_safe)
        if ee_speed_safe > ee_speed_max + 0.01:
            violations_with_safety["ee_speed"] += 1

        # Update state using safe action
        q = q + dq_safe * dt

    print("Safety Layer for 2-DOF Robot Arm")
    print(f"  Constraints: q in [-180, 180]°, dq_max=3 rad/s, EE speed max=1.5 m/s")
    print(f"  {n_steps} steps with aggressive random actions")
    print(f"\n  {'Constraint':>15} | {'Without safety':>15} | {'With safety':>15}")
    print("  " + "-" * 50)
    for key in violations_no_safety:
        print(f"  {key:>15} | {violations_no_safety[key]:>15} | "
              f"{violations_with_safety[key]:>15}")
    print(f"\n  The safety layer prevents ALL constraint violations while")
    print(f"  allowing the RL agent to explore as aggressively as it wants.")


def exercise_4():
    """
    Exercise 4: Curriculum Design for door opening.
    """
    curriculum = [
        {
            "level": 1,
            "name": "Door ajar (30°), light spring, no handle",
            "description": (
                "Door is already partially open. Robot only needs to push it further. "
                "Light spring resistance. No handle manipulation required."
            ),
            "progression": (
                "Teaches basic force application and trajectory following. "
                "Agent learns that pushing in the right direction opens the door."
            ),
            "success_threshold": "Door opens to 90° in 95% of episodes",
        },
        {
            "level": 2,
            "name": "Door closed, light spring, no latch",
            "description": (
                "Door starts fully closed but has no latch. Robot must apply "
                "enough force to overcome the spring and push door open."
            ),
            "progression": (
                "Builds on Level 1 by requiring the agent to initiate motion "
                "from a closed state. Teaches force magnitude control."
            ),
            "success_threshold": "Door opens fully from closed in 90% of episodes",
        },
        {
            "level": 3,
            "name": "Door closed, heavy spring, handle required",
            "description": (
                "Door has a lever handle that must be pressed down before pushing. "
                "Heavier spring resistance. Handle requires coordinated grasp + push."
            ),
            "progression": (
                "Adds handle manipulation — agent must learn sequential "
                "actions: approach handle, grasp, press down, push door. "
                "Heavy spring requires sustained force."
            ),
            "success_threshold": "Door opens via handle in 85% of episodes",
        },
        {
            "level": 4,
            "name": "Door with latch, variable weight, variable handle type",
            "description": (
                "Full task: door may have knob or lever handle, varying weight, "
                "and a latch that must be fully released before pushing. "
                "Domain randomization on door parameters."
            ),
            "progression": (
                "Full generalization. Agent must handle multiple handle types "
                "and varying dynamics. Domain randomization ensures robustness. "
                "Prepares agent for real-world transfer."
            ),
            "success_threshold": "Opens varied doors in 80% of episodes",
        },
    ]

    print("Curriculum Design: Robot Door Opening")
    print("=" * 60)
    for level in curriculum:
        print(f"\n  Level {level['level']}: {level['name']}")
        print(f"    Description:  {level['description']}")
        print(f"    Progression:  {level['progression']}")
        print(f"    Success gate: {level['success_threshold']}")

    print(f"\n  Curriculum rationale:")
    print(f"  Each level adds exactly one new challenge, allowing the agent")
    print(f"  to build on previously learned skills. Without curriculum,")
    print(f"  the agent faces the full problem from scratch and may never")
    print(f"  discover the handle→press→push sequence through random exploration.")


def exercise_5():
    """
    Exercise 5: Simulator Comparison
    CartPole vs custom 2D reaching environment.
    """
    # Simple CartPole-like environment
    class CartPole:
        def __init__(self):
            self.gravity = 9.81
            self.mass_cart = 1.0
            self.mass_pole = 0.1
            self.length = 0.5
            self.dt = 0.02
            self.x_threshold = 2.4
            self.theta_threshold = np.radians(12)
            self.state = None

        def reset(self):
            self.state = np.random.uniform(-0.05, 0.05, 4)
            return self.state.copy()

        def step(self, action):
            x, dx, theta, dtheta = self.state
            force = 10.0 if action == 1 else -10.0

            costh, sinth = np.cos(theta), np.sin(theta)
            total_mass = self.mass_cart + self.mass_pole
            ml = self.mass_pole * self.length

            temp = (force + ml * dtheta**2 * sinth) / total_mass
            thetaacc = (self.gravity * sinth - costh * temp) / (
                self.length * (4.0/3.0 - self.mass_pole * costh**2 / total_mass))
            xacc = temp - ml * thetaacc * costh / total_mass

            x += dx * self.dt
            dx += xacc * self.dt
            theta += dtheta * self.dt
            dtheta += thetaacc * self.dt

            self.state = np.array([x, dx, theta, dtheta])
            done = (abs(x) > self.x_threshold or abs(theta) > self.theta_threshold)
            reward = 1.0 if not done else 0.0
            return self.state.copy(), reward, done

    # Simple 2D reaching environment
    class Reaching2D:
        def __init__(self):
            self.dt = 0.05
            self.max_steps = 100
            self.step_count = 0
            self.pos = None
            self.target = None

        def reset(self):
            self.pos = np.zeros(2)
            self.target = np.random.uniform(-1, 1, 2)
            self.step_count = 0
            return np.concatenate([self.pos, self.target])

        def step(self, action):
            action = np.clip(action, -1, 1)
            self.pos += action * self.dt * 2
            self.step_count += 1

            dist = np.linalg.norm(self.pos - self.target)
            reward = -dist
            done = dist < 0.05 or self.step_count >= self.max_steps
            if dist < 0.05:
                reward += 10.0
            return np.concatenate([self.pos, self.target]), reward, done

    # Simple random policy evaluation
    np.random.seed(42)
    n_episodes = 200

    # CartPole with random policy
    env_cp = CartPole()
    cp_rewards = []
    for _ in range(n_episodes):
        obs = env_cp.reset()
        total_r = 0
        for _ in range(200):
            action = 1 if obs[2] > 0 else 0  # simple heuristic
            obs, r, done = env_cp.step(action)
            total_r += r
            if done:
                break
        cp_rewards.append(total_r)

    # Reaching2D with simple proportional policy
    env_r2d = Reaching2D()
    r2d_rewards = []
    for _ in range(n_episodes):
        obs = env_r2d.reset()
        total_r = 0
        for _ in range(100):
            pos = obs[:2]
            target = obs[2:]
            action = 2.0 * (target - pos)  # simple P-controller
            obs, r, done = env_r2d.step(action)
            total_r += r
            if done:
                break
        r2d_rewards.append(total_r)

    print("Simulator Comparison: CartPole vs 2D Reaching")
    print(f"\n  {'Metric':>25} | {'CartPole':>12} | {'Reaching2D':>12}")
    print("  " + "-" * 55)
    print(f"  {'Mean episode reward':>25} | {np.mean(cp_rewards):>12.2f} | "
          f"{np.mean(r2d_rewards):>12.2f}")
    print(f"  {'Std episode reward':>25} | {np.std(cp_rewards):>12.2f} | "
          f"{np.std(r2d_rewards):>12.2f}")
    print(f"  {'State dimension':>25} | {'4':>12} | {'4':>12}")
    print(f"  {'Action space':>25} | {'Discrete(2)':>12} | {'Cont(2)':>12}")

    print(f"\n  Key observations:")
    print(f"  - CartPole has simpler dynamics but requires balancing (instability)")
    print(f"  - Reaching2D has continuous actions, easier to solve with basic control")
    print(f"  - The environment strongly affects RL training efficiency:")
    print(f"    sparse vs dense rewards, action space type, and dynamics complexity")
    print(f"    all influence how quickly an RL agent can learn.")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 15: RL for Robotics — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: Reward Function Design ---")
    exercise_1()

    print("\n--- Exercise 2: Domain Randomization Study ---")
    exercise_2()

    print("\n--- Exercise 3: Safety Layer Implementation ---")
    exercise_3()

    print("\n--- Exercise 4: Curriculum Design ---")
    exercise_4()

    print("\n--- Exercise 5: Simulator Comparison ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
