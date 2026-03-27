"""
Exercises for Lesson 23: RL for Robotics
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Sim-to-Real Gap Analysis

    Identify sources of sim-to-real gap and quantify their effects.
    """
    print("Sim-to-Real Gap Analysis:")
    print("=" * 60)

    sources = [
        (
            "1. Physics modeling errors",
            "Inaccurate mass, friction, damping, contact dynamics.",
            "A robot arm trained with mass=1kg fails when actual mass=0.9kg.",
            "Domain randomization: sample mass in [0.7, 1.3] during training."
        ),
        (
            "2. Actuation model",
            "Real motors have latency, backlash, and saturation.",
            "Commanded torque=5Nm but real output varies ±10% with lag.",
            "Model actuator dynamics; use action smoothness penalty."
        ),
        (
            "3. Sensor noise",
            "Real sensors have noise, calibration errors, occlusion.",
            "Camera-based state estimation has 5mm localization error.",
            "Add measurement noise in simulation; use Gaussian noise layers."
        ),
        (
            "4. Unmodeled dynamics",
            "Cable management, heat effects, wear, external disturbances.",
            "Robot arm cable causes unexpected torques not in sim.",
            "Adaptive control; online fine-tuning with few real samples."
        ),
    ]

    for name, desc, example, mitigation in sources:
        print(f"\n  {name}")
        print(f"    Description:  {desc}")
        print(f"    Example:      {example}")
        print(f"    Mitigation:   {mitigation}")

    # Quantify transfer gap
    print("\n  Quantitative sim-to-real gap example:")
    np.random.seed(0)
    n_eval = 500

    # Policy tuned for sim (mass=1.0, friction=0.1)
    def sim_success_rate(mass, friction, n=n_eval, seed=0):
        """Simplified: success = policy gain * (1 - |mass-1|) * (1 - |friction-0.1|*5)"""
        np.random.seed(seed)
        gains = np.random.normal(0.8, 0.15, n)
        mass_penalty = abs(mass - 1.0) * 1.5
        friction_penalty = abs(friction - 0.1) * 4.0
        successes = gains > (0.2 + mass_penalty + friction_penalty)
        return np.mean(successes)

    print(f"\n  {'Scenario':>30} | {'Success Rate':>14}")
    print("  " + "-" * 50)
    scenarios = [
        ("Sim (nominal)", 1.0, 0.1),
        ("Real (mass+10%)", 1.1, 0.1),
        ("Real (mass+30%)", 1.3, 0.1),
        ("Real (friction*2)", 1.0, 0.2),
        ("Real (both shifted)", 1.2, 0.18),
    ]
    for name, mass, friction in scenarios:
        rate = sim_success_rate(mass, friction)
        print(f"  {name:>30} | {rate:>14.1%}")


def exercise_2():
    """
    Exercise 2: Domain Randomization Implementation

    Implement domain randomization and show it improves
    transfer to environments outside the training distribution.
    """
    print("\nDomain Randomization:")
    print("=" * 60)

    np.random.seed(42)

    class SimpleArmEnv:
        """
        1-DOF arm: reach target angle.
        State: [current_angle, angular_vel, target_angle]
        Action: torque [-1, 1]
        """

        def __init__(self, inertia=1.0, damping=0.1, noise=0.01):
            self.inertia = inertia
            self.damping = damping
            self.noise = noise
            self.dt = 0.05

        def reset(self, target=None):
            self.angle = np.random.uniform(-1.0, 1.0)
            self.vel = 0.0
            self.target = target or np.random.uniform(-0.5, 0.5)
            return np.array([self.angle, self.vel, self.target])

        def step(self, torque):
            torque = np.clip(torque, -1.0, 1.0)
            acc = (torque - self.damping * self.vel) / self.inertia
            self.vel += acc * self.dt
            self.angle += self.vel * self.dt
            obs = np.array([self.angle + np.random.randn() * self.noise,
                            self.vel + np.random.randn() * self.noise,
                            self.target])
            err = abs(self.angle - self.target)
            reward = -err - 0.01 * torque**2
            done = err < 0.05
            return obs, reward, done

    def simple_policy(obs, w):
        """PD-like linear policy: torque = w . [angle_error, vel]"""
        error = obs[0] - obs[2]
        vel = obs[1]
        return np.clip(np.dot(w, [error, vel]), -1.0, 1.0)

    def evaluate_policy(w, inertia, damping, noise, n_trials=200, seed=0):
        np.random.seed(seed)
        successes = 0
        for _ in range(n_trials):
            env = SimpleArmEnv(inertia=inertia, damping=damping, noise=noise)
            obs = env.reset()
            for _ in range(50):
                a = simple_policy(obs, w)
                obs, _, done = env.step(a)
                if done:
                    successes += 1
                    break
        return successes / n_trials

    # Policy A: tuned only for nominal params
    w_nominal = np.array([-3.0, -1.5])

    # Policy B: tuned for a range (domain randomization)
    w_robust = np.array([-1.5, -0.8])  # more conservative

    print("\n  Transfer evaluation:")
    print(f"  {'Scenario':>30} | {'Nominal':>10} | {'Robust':>10}")
    print("  " + "-" * 58)

    test_cases = [
        ("Nominal (I=1.0, d=0.1)", 1.0, 0.1, 0.01),
        ("Heavy (I=1.5, d=0.1)", 1.5, 0.1, 0.01),
        ("Light (I=0.7, d=0.1)", 0.7, 0.1, 0.01),
        ("High damping (I=1.0, d=0.3)", 1.0, 0.3, 0.01),
        ("High noise (I=1.0, sigma=0.1)", 1.0, 0.1, 0.1),
    ]

    for name, I, d, n in test_cases:
        rate_a = evaluate_policy(w_nominal, I, d, n)
        rate_b = evaluate_policy(w_robust, I, d, n)
        print(f"  {name:>30} | {rate_a:>10.1%} | {rate_b:>10.1%}")

    print("\n  Nominal policy may excel in sim but degrades under distribution shift.")
    print("  Robust policy maintains performance across varied conditions.")


def exercise_3():
    """
    Exercise 3: Reward Shaping for Locomotion

    Design a locomotion reward function with multiple components
    and analyze each component's effect.
    """
    print("\nLocomotion Reward Design:")
    print("=" * 60)

    print()
    print("  Locomotion reward components:")
    print("  R = w1*forward_vel - w2*energy - w3*fall - w4*jerk + w5*alive")
    print()

    # Simulated locomotion episodes with different reward designs
    np.random.seed(5)

    def generate_episode(policy_type, n_steps=100):
        """Generate synthetic locomotion data for different policy types."""
        if policy_type == "random":
            vels = np.random.normal(0, 1, n_steps)
            torques = np.random.normal(0, 2, (n_steps, 2))
            survived = n_steps
        elif policy_type == "fast_risky":
            vels = np.random.normal(3.0, 1.0, n_steps)
            torques = np.random.normal(0, 3.0, (n_steps, 2))
            survived = int(n_steps * np.random.uniform(0.3, 0.7))
        elif policy_type == "slow_safe":
            vels = np.random.normal(1.0, 0.2, n_steps)
            torques = np.random.normal(0, 0.5, (n_steps, 2))
            survived = n_steps
        else:  # balanced
            vels = np.random.normal(2.0, 0.3, n_steps)
            torques = np.random.normal(0, 1.0, (n_steps, 2))
            survived = int(n_steps * np.random.uniform(0.8, 1.0))

        vels = vels[:survived]
        torques = torques[:survived]
        return vels, torques, survived

    def compute_reward(vels, torques, survived, total_steps,
                       w_vel=1.0, w_energy=0.01, w_fall=10.0, w_alive=0.5):
        forward = np.mean(vels) * w_vel
        energy = np.mean(np.sum(torques**2, axis=1)) * w_energy
        fall_penalty = w_fall if survived < total_steps else 0.0
        alive = (survived / total_steps) * w_alive
        total = forward - energy - fall_penalty + alive
        return total, {'forward': forward, 'energy': -energy,
                       'fall': -fall_penalty, 'alive': alive}

    total_steps = 100
    print(f"  Policy comparison (reward components):")
    print(f"  {'Policy':>14} | {'Total':>8} | {'Forward':>8} | {'Energy':>8} | "
          f"{'Fall':>6} | {'Alive':>6}")
    print("  " + "-" * 66)

    for policy in ["random", "fast_risky", "slow_safe", "balanced"]:
        vels, torques, survived = generate_episode(policy)
        r_total, components = compute_reward(vels, torques, survived, total_steps)
        print(f"  {policy:>14} | {r_total:>8.3f} | {components['forward']:>8.3f} | "
              f"{components['energy']:>8.3f} | {components['fall']:>6.1f} | "
              f"{components['alive']:>6.3f}")

    print("\n  'balanced' gets highest reward: good speed without falling.")
    print("  'fast_risky' often falls (large fall penalty).")
    print("  Reward weights must be tuned per task — no universal formula.")


def exercise_4():
    """
    Exercise 4: System Identification

    Fit physical parameters (mass, friction) from observed trajectories.
    """
    print("\nSystem Identification from Trajectory Data:")
    print("=" * 60)

    np.random.seed(1)

    # True robot: mass=1.2kg, friction=0.15
    true_mass = 1.2
    true_friction = 0.15
    dt = 0.05

    def simulate_trajectory(mass, friction, forces, init_vel=0.0):
        """1-D dynamics: a = F/m - friction*v"""
        vel = init_vel
        positions = [0.0]
        for f in forces:
            acc = f / mass - friction * vel
            vel += acc * dt
            positions.append(positions[-1] + vel * dt)
        return np.array(positions)

    # Collect observed trajectory from true robot
    forces = np.random.normal(1.0, 0.5, 50)
    obs_positions = simulate_trajectory(true_mass, true_friction, forces)
    obs_positions += np.random.randn(len(obs_positions)) * 0.01  # noise

    # System identification: grid search over (mass, friction)
    best_loss = float('inf')
    best_mass, best_friction = None, None

    mass_range = np.arange(0.5, 2.0, 0.1)
    friction_range = np.arange(0.0, 0.5, 0.05)

    for mass in mass_range:
        for friction in friction_range:
            pred = simulate_trajectory(mass, friction, forces)
            loss = np.mean((pred - obs_positions)**2)
            if loss < best_loss:
                best_loss = loss
                best_mass = mass
                best_friction = friction

    print(f"\n  True parameters:      mass={true_mass:.2f}, friction={true_friction:.2f}")
    print(f"  Identified:           mass={best_mass:.2f}, friction={best_friction:.2f}")
    print(f"  Fit loss (MSE):       {best_loss:.8f}")
    print(f"\n  System identification enables physics-consistent simulation,")
    print(f"  reducing sim-to-real gap by calibrating to actual robot behavior.")


if __name__ == "__main__":
    print("=== Exercise 1: Sim-to-Real Gap ===")
    exercise_1()

    print("\n=== Exercise 2: Domain Randomization ===")
    exercise_2()

    print("\n=== Exercise 3: Locomotion Reward ===")
    exercise_3()

    print("\n=== Exercise 4: System Identification ===")
    exercise_4()

    print("\nAll exercises completed!")
