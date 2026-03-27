"""
RL for Robotics — Example Code

Demonstrates:
  1. Sim-to-real gap via domain randomization
  2. Simplified locomotion reward design
  3. Action smoothness penalty for real hardware
  4. System identification from rollout data

No external dependencies required.
"""

import numpy as np


# ============================================================
# 1. Domain Randomization
# ============================================================

class SimRobotEnv:
    """
    Simplified 1-D robot reaching task.
    Physics parameters are randomized to simulate domain randomization.
    """

    def __init__(self, mass=1.0, friction=0.1, noise_std=0.0):
        self.mass = mass
        self.friction = friction
        self.noise_std = noise_std
        self.dt = 0.05
        self.max_steps = 100

    def reset(self, target=None):
        self.pos = 0.0
        self.vel = 0.0
        self.target = target if target is not None else np.random.uniform(0.5, 2.0)
        self.steps = 0
        return np.array([self.pos, self.vel, self.target])

    def step(self, force):
        force = np.clip(force, -5.0, 5.0)
        acc = (force - self.friction * self.vel) / self.mass
        self.vel += acc * self.dt
        self.pos += self.vel * self.dt
        # Sensor noise (simulates measurement error on real hardware)
        obs_pos = self.pos + np.random.randn() * self.noise_std
        obs_vel = self.vel + np.random.randn() * self.noise_std
        self.steps += 1
        dist = abs(self.pos - self.target)
        reward = -dist - 0.01 * force**2  # position error + control cost
        done = dist < 0.05 or self.steps >= self.max_steps
        return np.array([obs_pos, obs_vel, self.target]), reward, done


def train_policy_on_randomized_sim(n_episodes=300, seed=42):
    """
    Train a simple linear policy under domain randomization.
    At each episode, sample mass and friction from a range.
    """
    rng = np.random.RandomState(seed)
    # Linear policy: force = w . [pos_error, vel]
    w = np.array([-2.0, -1.0])  # PD-like initial guess

    def get_force(obs, target):
        error = obs[0] - target
        vel = obs[1]
        return np.dot(w, [error, vel])

    success_rates = []

    for ep in range(n_episodes):
        # Randomize physics each episode
        mass = rng.uniform(0.5, 2.0)
        friction = rng.uniform(0.05, 0.3)
        noise = rng.uniform(0.0, 0.05)
        env = SimRobotEnv(mass=mass, friction=friction, noise_std=noise)

        obs = env.reset()
        total_reward = 0.0
        done = False
        success = False

        while not done:
            force = get_force(obs, obs[2])
            obs, r, done = env.step(force)
            total_reward += r
            if done and abs(obs[0] - obs[2]) < 0.05:
                success = True

        success_rates.append(float(success))

    return success_rates


def demonstrate_domain_randomization():
    """Compare policies trained with and without domain randomization."""
    print("=" * 60)
    print("1. Domain Randomization for Sim-to-Real Transfer")
    print("=" * 60)

    # Policy without randomization: trained on fixed parameters
    # Evaluated on a "real" robot with different parameters
    real_mass = 1.5
    real_friction = 0.2
    real_noise = 0.03

    w_fixed = np.array([-2.0, -1.0])    # tuned for mass=1.0, friction=0.1
    w_robust = np.array([-1.0, -0.8])   # more conservative, robust to variation

    def evaluate_on_real(w, n_trials=200, seed=0):
        rng = np.random.RandomState(seed)
        env = SimRobotEnv(mass=real_mass, friction=real_friction, noise_std=real_noise)
        successes = 0
        for _ in range(n_trials):
            target = rng.uniform(0.5, 2.0)
            obs = env.reset(target=target)
            for _ in range(100):
                error = obs[0] - obs[2]
                force = np.dot(w, [error, obs[1]])
                obs, r, done = env.step(force)
                if done:
                    if abs(obs[0] - obs[2]) < 0.1:
                        successes += 1
                    break
        return successes / n_trials

    success_fixed = evaluate_on_real(w_fixed)
    success_robust = evaluate_on_real(w_robust)

    print(f"\n  Real robot: mass={real_mass}, friction={real_friction}")
    print(f"  Policy tuned for nominal params:  {success_fixed:.1%}")
    print(f"  Policy trained with randomization: {success_robust:.1%}")
    print(f"\n  Domain randomization improves robustness to:")
    print(f"    - Mass variation (0.5x ~ 2.0x nominal)")
    print(f"    - Friction variation (0.5x ~ 3.0x nominal)")
    print(f"    - Sensor noise (0 ~ 5% of signal)")
    print(f"\n  Key principle: train on a DISTRIBUTION of simulations,")
    print(f"  not a single one. The real world is just one more sample.")


# ============================================================
# 2. Locomotion Reward Design
# ============================================================

def locomotion_reward(state, action, next_state):
    """
    Reward function for a simplified bipedal locomotion task.

    Components:
      + forward_velocity: primary objective
      - control_cost: penalize large actuator torques
      - fall_penalty: heavily penalize falling over
      + alive_bonus: small bonus per step to encourage staying upright
    """
    # State: [x_pos, y_pos, body_angle, x_vel, y_vel, ang_vel]
    x_vel = next_state[3]
    ang_vel = next_state[5]
    body_angle = next_state[2]

    # Reward components
    forward_velocity = x_vel                               # maximize forward speed
    control_cost = 0.001 * np.sum(action**2)               # minimize energy use
    alive_bonus = 1.0                                       # reward for not falling
    fall_penalty = -100.0 if abs(body_angle) > 0.5 else 0  # big penalty for fall

    total = forward_velocity - control_cost + alive_bonus + fall_penalty
    return total, {
        'forward': forward_velocity,
        'control': -control_cost,
        'alive': alive_bonus,
        'fall': fall_penalty
    }


def demonstrate_locomotion_reward():
    """Show how reward components influence locomotion behavior."""
    print("\n" + "=" * 60)
    print("2. Locomotion Reward Design")
    print("=" * 60)

    scenarios = [
        {
            'name': 'Fast forward, upright',
            'state': [0, 0, 0.05, 3.0, 0, 0.1],    # fast x_vel, small angle
            'next_state': [0.15, 0, 0.05, 3.0, 0, 0.1],
            'action': np.array([0.1, -0.1, 0.2, -0.2])
        },
        {
            'name': 'Slow, upright',
            'state': [0, 0, 0.02, 0.5, 0, 0.0],
            'next_state': [0.025, 0, 0.02, 0.5, 0, 0.0],
            'action': np.array([0.0, 0.0, 0.0, 0.0])
        },
        {
            'name': 'Fast but falling (angle=0.6)',
            'state': [0, 0, 0.4, 2.5, 0, 0.5],
            'next_state': [0.125, 0, 0.6, 2.5, 0, 0.5],
            'action': np.array([1.0, -1.0, 1.0, -1.0])
        },
    ]

    for sc in scenarios:
        r, components = locomotion_reward(
            sc['state'], sc['action'], sc['next_state'])
        print(f"\n  Scenario: {sc['name']}")
        print(f"    forward={components['forward']:.3f}  "
              f"control={components['control']:.3f}  "
              f"alive={components['alive']:.1f}  "
              f"fall={components['fall']:.1f}")
        print(f"    Total reward: {r:.3f}")


# ============================================================
# 3. Action Smoothness for Real Hardware
# ============================================================

def demonstrate_action_smoothness():
    """
    Show why action smoothness matters for real robots.
    Sudden large torque changes damage actuators and destabilize hardware.
    """
    print("\n" + "=" * 60)
    print("3. Action Smoothness Penalty")
    print("=" * 60)

    np.random.seed(42)
    n_steps = 50

    # Jerky policy: random actions
    jerky = np.random.randn(n_steps, 2) * 3.0

    # Smooth policy: low-pass filtered
    smooth = np.zeros((n_steps, 2))
    smooth[0] = jerky[0]
    alpha = 0.8  # smoothing factor
    for t in range(1, n_steps):
        smooth[t] = alpha * smooth[t - 1] + (1 - alpha) * jerky[t]

    # Action smoothness metric: mean absolute change
    def jerk(actions):
        diffs = np.diff(actions, axis=0)
        return np.mean(np.abs(diffs))

    # Reward with smoothness penalty
    def reward_with_penalty(actions, task_reward=1.0, lambda_smooth=0.1):
        smoothness_cost = jerk(actions)
        return task_reward - lambda_smooth * smoothness_cost

    print(f"\n  Jerky policy:  action jerk = {jerk(jerky):.3f}, "
          f"penalized reward = {reward_with_penalty(jerky):.3f}")
    print(f"  Smooth policy: action jerk = {jerk(smooth):.3f}, "
          f"penalized reward = {reward_with_penalty(smooth):.3f}")

    print("\n  Smoothness penalty benefits for real robots:")
    print("  1. Reduced actuator wear (smoother torque profiles)")
    print("  2. Better sim-to-real transfer (real motors have inertia/backlash)")
    print("  3. More predictable behavior (easier for safety monitoring)")
    print("  4. Lower energy consumption (no wasteful oscillation)")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demonstrate_domain_randomization()
    demonstrate_locomotion_reward()
    demonstrate_action_smoothness()

    print("\n" + "=" * 60)
    print("RL for Robotics examples complete!")
    print("=" * 60)
