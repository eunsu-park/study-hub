# 15. Reinforcement Learning for Robotics

[← Previous: ROS2 Navigation Stack](14_ROS2_Navigation.md) | [Next: Multi-Robot Systems and Swarms →](16_Multi_Robot_Systems.md)

---

## Learning Objectives

1. Explain why reinforcement learning is valuable for robotic tasks that are hard to program explicitly
2. Understand the sim-to-real transfer problem and key techniques (domain randomization, system identification)
3. Design reward functions for robotic manipulation and locomotion tasks
4. Describe safe RL concepts: constrained optimization, safe exploration strategies
5. Recognize how RL policies are applied to grasping, assembly, and locomotion
6. Compare major simulation environments for robot RL: MuJoCo, Isaac Gym, Gymnasium

---

Throughout this course, we have designed robot controllers using mathematical models: computed torque control uses the equations of motion, impedance control uses a desired mass-spring-damper model, and navigation uses costmaps and planning algorithms. These model-based approaches work beautifully when we have accurate models. But what about tasks where the model is too complex to write down, the environment is too unpredictable, or the optimal strategy is unknown?

Consider teaching a robot hand to manipulate a Rubik's cube. The contact dynamics between fingers and the cube involve friction, deformation, slipping, and dozens of interacting surfaces. Writing a physics-based controller for this is extraordinarily difficult. But a reinforcement learning agent can discover effective policies through trial and error — learning from millions of simulated attempts what no human engineer could program by hand.

This lesson bridges robotics and RL, focusing on the unique challenges that arise when RL agents must operate in the physical world: transferring from simulation to reality, ensuring safety during learning, and designing reward functions that lead to desired robot behaviors.

> **Analogy**: Sim-to-real transfer is like practicing on a flight simulator before flying a real aircraft. The virtual experience transfers to the real cockpit if the simulation is realistic enough. But if the simulator has perfect weather while reality has turbulence, or if the simulated controls respond differently from the real ones, the pilot's training may not transfer. Domain randomization is like training the pilot in many different simulated conditions — fog, rain, crosswind — so they are prepared for anything reality throws at them.

---

## 1. Why RL for Robotics?

### 1.1 Limitations of Model-Based Control

Traditional robotics assumes we can:
1. Write down the dynamics equations ($M\ddot{q} + C\dot{q} + g = \tau$)
2. Model the environment (obstacle positions, surface properties)
3. Specify the optimal behavior (trajectory, impedance parameters)

These assumptions break down for many real-world tasks:

| Challenge | Why Models Fail | RL Advantage |
|-----------|----------------|--------------|
| Contact-rich manipulation | Friction, deformation are hard to model | Learns from interaction experience |
| Deformable objects | Infinite-dimensional state | Learns compact representations |
| Novel objects | No prior model available | Generalizes from training distribution |
| Complex environments | Too many variables to model | Adapts through trial and error |
| Undefined objectives | "Handle gently" is hard to formalize | Reward shaping captures intent |

### 1.2 The RL Framework for Robotics

A robot RL problem is formulated as a Markov Decision Process (MDP):

- **State** $s_t$: Joint angles, velocities, object poses, sensor readings
- **Action** $a_t$: Joint torques, velocity commands, gripper force
- **Transition** $s_{t+1} \sim p(s_{t+1} | s_t, a_t)$: Robot dynamics + environment
- **Reward** $r_t = R(s_t, a_t, s_{t+1})$: Task success signal
- **Policy** $\pi(a_t | s_t)$: Learned mapping from observations to actions

The goal is to find the policy that maximizes the expected cumulative reward:

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

### 1.3 On-Robot vs. In-Simulation Learning

| Approach | Pros | Cons |
|----------|------|------|
| Learn on real robot | No sim-to-real gap | Slow (real-time), expensive, can damage robot |
| Learn in simulation | Fast (1000x real-time), safe, parallelizable | Sim-to-real gap |
| Hybrid | Best of both | Complex pipeline |

**Current best practice**: Train in simulation, fine-tune on the real robot. The simulation phase does the heavy lifting (millions of episodes), and the real-world phase bridges the gap (hundreds of episodes).

---

## 2. Sim-to-Real Transfer

### 2.1 The Sim-to-Real Gap

A policy trained in simulation may fail on the real robot because of differences between the simulator and reality:

| Gap Type | Examples |
|----------|---------|
| **Visual** | Textures, lighting, reflections differ |
| **Dynamic** | Friction, damping, mass distribution differ |
| **Actuator** | Motor response, backlash, delays differ |
| **Sensor** | Noise characteristics, calibration differ |
| **Contact** | Surface properties, deformation differ |

### 2.2 Domain Randomization

**Domain randomization** trains the policy across a wide range of simulation parameters so it learns to be robust to any specific realization — including reality.

$$\theta_{sim} \sim \text{Uniform}(\theta_{min}, \theta_{max})$$

Parameters to randomize:

```python
import numpy as np

class DomainRandomizer:
    """Randomizes simulation parameters for sim-to-real transfer.

    Why randomize? If we train in one specific simulation, the policy
    overfits to that simulation's parameters (exact friction = 0.5,
    exact mass = 1.0 kg). By randomizing these parameters across a
    wide range, the policy must work for friction = 0.2 to 0.8 and
    mass = 0.7 to 1.3. Reality falls somewhere in this range, so the
    robust policy transfers.
    """

    def __init__(self):
        # Define randomization ranges
        self.ranges = {
            # Dynamics
            'friction': (0.2, 1.0),
            'damping': (0.5, 2.0),
            'mass_scale': (0.8, 1.2),       # Scale factor for link masses
            'com_offset': (-0.02, 0.02),     # Center of mass offset [m]

            # Actuator
            'motor_strength': (0.85, 1.15),  # Scale factor for max torque
            'action_delay': (0, 3),          # Control delay in timesteps
            'action_noise': (0.0, 0.05),     # Noise added to actions

            # Sensor
            'observation_noise': (0.0, 0.02),
            'encoder_offset': (-0.01, 0.01), # Systematic bias [rad]

            # Environment
            'gravity': (9.7, 9.9),           # Gravity variation
            'ground_friction': (0.3, 1.0),
        }

    def sample(self):
        """Sample a random parameter set for one episode.

        Why per-episode randomization? Each training episode uses a
        different set of parameters. Over millions of episodes, the
        policy experiences the full range and learns a strategy that
        works across all of them — including the real world.
        """
        params = {}
        for name, (low, high) in self.ranges.items():
            if isinstance(low, int):
                params[name] = np.random.randint(low, high + 1)
            else:
                params[name] = np.random.uniform(low, high)
        return params

    def apply_to_env(self, env, params):
        """Apply randomized parameters to the simulation environment.

        Each simulator (MuJoCo, Isaac Gym, etc.) has its own API for
        modifying physical parameters. This method translates our
        abstract parameter set to simulator-specific calls.
        """
        # Pseudocode — actual API depends on simulator
        # env.model.geom_friction[:] *= params['friction']
        # env.model.body_mass[:] *= params['mass_scale']
        # env.sim.gravity[2] = -params['gravity']
        pass
```

### 2.3 System Identification

**System identification** takes the opposite approach from domain randomization: instead of making the policy robust to all parameters, it estimates the real-world parameters and configures the simulator to match reality.

Steps:
1. Collect real-world trajectory data $(s_t^{real}, a_t^{real}, s_{t+1}^{real})$
2. Optimize simulation parameters to minimize:

$$\theta^* = \arg\min_\theta \sum_t \| f_{sim}(s_t, a_t; \theta) - s_{t+1}^{real} \|^2$$

3. Train RL policy in the calibrated simulator

```python
def system_identification_concept(real_trajectories, sim_env):
    """Estimate simulation parameters to match real-world data.

    Why system ID? Domain randomization trains a policy that is
    'jack of all trades, master of none.' System identification
    produces a simulation that closely matches reality, enabling
    a more specialized (and often better-performing) policy.

    The downside: requires real-world data collection, and the
    identified parameters may not generalize to new conditions.
    """
    from scipy.optimize import minimize

    def simulation_error(params_flat):
        """Compute mismatch between sim and real trajectories."""
        # Unpack parameters
        friction = params_flat[0]
        damping = params_flat[1]
        mass_scale = params_flat[2]

        # Configure simulator
        # sim_env.set_params(friction, damping, mass_scale)

        total_error = 0.0
        for traj in real_trajectories:
            for s, a, s_next_real in traj:
                # Simulate one step
                # s_next_sim = sim_env.step(s, a)
                # total_error += np.sum((s_next_sim - s_next_real)**2)
                pass

        return total_error

    # Initial guess
    x0 = np.array([0.5, 1.0, 1.0])

    # Optimize
    result = minimize(simulation_error, x0, method='Nelder-Mead')
    return result.x
```

### 2.4 Combining Both Approaches

The most robust approach combines domain randomization with system identification:

1. **System ID** to center the randomization range around realistic values
2. **Domain randomization** around the identified parameters for robustness
3. **Fine-tuning** on the real robot to close the remaining gap

---

## 3. Reward Shaping for Robotic Tasks

### 3.1 The Reward Design Challenge

The reward function is arguably the most critical design choice in robot RL. A poorly designed reward leads to:

- **Reward hacking**: The agent finds a way to get high reward without achieving the intended behavior
- **Sparse reward**: The agent almost never receives any reward signal (e.g., reward = 1 only when task is complete)
- **Unintended behavior**: The reward encourages something different from what the designer intended

### 3.2 Reward Shaping Strategies

**Dense vs. Sparse rewards**:

$$r_{sparse}(s) = \begin{cases} 1 & \text{if task complete} \\ 0 & \text{otherwise} \end{cases}$$

$$r_{dense}(s) = -d(s, s_{goal}) + r_{bonus}(s)$$

Dense rewards provide continuous guidance but are harder to design correctly.

```python
class ManipulationReward:
    """Reward function for a robotic grasping task.

    Why multi-component rewards? A single sparse signal ('did you grasp it?')
    is too infrequent for learning. Breaking the reward into components
    guides the agent through sub-goals: approach the object, close the
    gripper, lift. Each component shapes one aspect of the desired behavior.
    """

    def __init__(self):
        self.reach_weight = 1.0
        self.grasp_weight = 5.0
        self.lift_weight = 10.0
        self.success_bonus = 100.0

    def compute(self, gripper_pos, object_pos, gripper_closed,
                object_height, target_height):
        """Compute multi-component reward for grasping.

        Why weighted components? Different phases of the task have
        different importance. Lifting the object is worth more than
        reaching because it's the actual goal. But without the
        reaching reward, the agent would never discover that it
        needs to approach the object first.
        """
        reward = 0.0

        # Component 1: Reaching — encourage gripper to approach object
        distance = np.linalg.norm(gripper_pos - object_pos)
        reward += self.reach_weight * (1.0 - np.tanh(5.0 * distance))

        # Component 2: Grasping — reward closing gripper near object
        if distance < 0.05:  # Close enough to grasp
            if gripper_closed:
                reward += self.grasp_weight

        # Component 3: Lifting — reward lifting the object
        if gripper_closed and object_height > 0.02:
            height_progress = min(object_height / target_height, 1.0)
            reward += self.lift_weight * height_progress

        # Bonus: task completion
        if object_height >= target_height * 0.95:
            reward += self.success_bonus

        return reward


class LocomotionReward:
    """Reward function for legged robot locomotion.

    Why penalize energy and joint torques? Without these penalties,
    the agent learns bizarre, energy-wasteful gaits — spinning joints,
    high-frequency oscillations — that maximize forward velocity but
    would destroy a real robot's actuators. Energy penalties encourage
    natural, efficient gaits similar to biological locomotion.
    """

    def __init__(self, target_velocity=1.0):
        self.target_velocity = target_velocity

    def compute(self, forward_velocity, body_height, joint_torques,
                body_orientation, feet_contact):
        """Compute locomotion reward.

        The reward balances multiple objectives:
        - Move forward at the target speed
        - Stay upright (penalize body tilt)
        - Minimize energy consumption (penalize joint torques)
        - Maintain stability (penalize excessive body height variation)
        """
        reward = 0.0

        # Velocity tracking: reward matching target speed
        vel_error = abs(forward_velocity - self.target_velocity)
        reward += 2.0 * np.exp(-2.0 * vel_error)

        # Stability: penalize body tilt (roll and pitch)
        roll, pitch = body_orientation[0], body_orientation[1]
        reward -= 0.5 * (roll**2 + pitch**2)

        # Energy efficiency: penalize squared torques
        reward -= 0.001 * np.sum(joint_torques**2)

        # Height maintenance: penalize deviation from nominal
        nominal_height = 0.3  # meters
        reward -= 1.0 * (body_height - nominal_height)**2

        # Alive bonus: constant reward for not falling
        if body_height > 0.15:
            reward += 0.5
        else:
            reward -= 10.0  # Heavy penalty for falling

        return reward
```

### 3.3 Curriculum Learning

For difficult tasks, training with the final reward from the start may be too hard. **Curriculum learning** gradually increases difficulty:

```python
def curriculum_schedule(epoch, total_epochs):
    """Curriculum for a manipulation task.

    Why curriculum? Imagine learning to juggle 5 balls starting from
    scratch. You'd fail millions of times before accidentally
    succeeding. It's much more efficient to first learn with 1 ball,
    then 2, then 3... This is curriculum learning: start easy,
    gradually increase difficulty as the agent improves.
    """
    progress = epoch / total_epochs

    if progress < 0.3:
        # Phase 1: Object starts close to gripper, wide success threshold
        return {
            'initial_distance': 0.05,
            'success_threshold': 0.10,
            'object_mass': 0.1,
        }
    elif progress < 0.7:
        # Phase 2: Moderate distance, tighter threshold
        return {
            'initial_distance': 0.15,
            'success_threshold': 0.05,
            'object_mass': 0.3,
        }
    else:
        # Phase 3: Full difficulty
        return {
            'initial_distance': 0.30,
            'success_threshold': 0.02,
            'object_mass': 0.5,
        }
```

---

## 4. Safe RL for Robotics

### 4.1 Why Safety Matters

RL's exploration mechanism (random actions to discover new behaviors) is dangerous on real robots:

- A random joint torque could slam the arm into the table
- An aggressive locomotion policy could flip the robot
- An exploratory action near a human could cause injury

Safety is not optional in robotics — it is a hard constraint.

### 4.2 Constrained RL

**Constrained Markov Decision Process (CMDP)** adds safety constraints to the standard RL objective:

$$\max_\pi \mathbb{E}_\pi \left[ \sum_t \gamma^t r_t \right] \quad \text{subject to} \quad \mathbb{E}_\pi \left[ \sum_t \gamma^t c_t^{(i)} \right] \leq d_i \quad \forall i$$

where $c_t^{(i)}$ is the cost for constraint $i$ (e.g., joint torque limit, collision penalty) and $d_i$ is the constraint budget.

**Lagrangian relaxation** converts the CMDP into an unconstrained problem:

$$\mathcal{L}(\pi, \lambda) = \mathbb{E}_\pi\left[\sum_t r_t\right] - \sum_i \lambda_i \left(\mathbb{E}_\pi\left[\sum_t c_t^{(i)}\right] - d_i\right)$$

The dual variable $\lambda_i$ is updated to enforce the constraint.

### 4.3 Safe Exploration Strategies

```python
class SafetyLayer:
    """Safety layer that modifies RL actions to prevent constraint violations.

    Why a separate safety layer? It decouples learning from safety.
    The RL agent can explore freely, learning from a wide range of
    actions. The safety layer then projects unsafe actions to the
    nearest safe action before they reach the robot. This way, the
    robot is always safe, even during exploration.
    """

    def __init__(self, joint_limits, torque_limits, velocity_limits):
        self.q_min, self.q_max = joint_limits
        self.tau_max = torque_limits
        self.qd_max = velocity_limits

    def filter_action(self, action, current_state):
        """Project action to the safe set.

        Safety constraints:
        1. Joint position limits: don't command motion beyond limits
        2. Torque limits: don't exceed actuator capacity
        3. Velocity limits: don't exceed safe speed
        4. Self-collision: don't command configurations that cause collision
        """
        q, qd = current_state[:len(self.q_min)], current_state[len(self.q_min):]

        # Clamp torque commands
        safe_action = np.clip(action, -self.tau_max, self.tau_max)

        # Check if action would drive joint beyond limits
        # Simple: reduce torque if joint is near limit and moving toward it
        for i in range(len(q)):
            if q[i] < self.q_min[i] + 0.1 and safe_action[i] < 0:
                # Near lower limit, trying to go further — block
                safe_action[i] = max(safe_action[i], 0.0)
            if q[i] > self.q_max[i] - 0.1 and safe_action[i] > 0:
                # Near upper limit, trying to go further — block
                safe_action[i] = min(safe_action[i], 0.0)

        # Velocity damping near limits
        for i in range(len(qd)):
            if abs(qd[i]) > self.qd_max[i] * 0.9:
                # Approaching velocity limit — reduce command
                safe_action[i] *= 0.5

        return safe_action
```

### 4.4 Sim-to-Real Safety Pipeline

```
┌─────────────────────────────────────────────┐
│ Phase 1: Simulation Training                 │
│ - Unconstrained exploration (safe in sim)    │
│ - Domain randomization                       │
│ - Reward shaping + curriculum learning       │
├─────────────────────────────────────────────┤
│ Phase 2: Simulation Validation               │
│ - Test in simulation with safety constraints │
│ - Verify constraint satisfaction             │
│ - Statistical analysis of failure modes      │
├─────────────────────────────────────────────┤
│ Phase 3: Cautious Real-World Transfer        │
│ - Start with conservative action limits      │
│ - Gradually increase operational envelope    │
│ - Safety layer always active                 │
│ - Human supervisor with emergency stop       │
├─────────────────────────────────────────────┤
│ Phase 4: Real-World Fine-Tuning              │
│ - Safe fine-tuning with constrained RL       │
│ - Small batch updates (few episodes)         │
│ - Continuous safety monitoring               │
└─────────────────────────────────────────────┘
```

---

## 5. RL for Manipulation

### 5.1 Grasping Policies

Robot grasping is one of the most successful applications of RL in robotics:

**Observation space**: RGB-D image (or point cloud), gripper position/orientation, joint angles
**Action space**: Gripper pose delta ($\Delta x, \Delta y, \Delta z, \Delta\theta$) + open/close
**Reward**: Grasp success (did the object remain in the gripper after lifting?)

```python
class GraspingEnvironment:
    """Simplified grasping environment for RL training.

    Why learn grasping with RL instead of analytical methods?
    Analytical grasp planners (force closure, form closure) require
    exact object geometry and friction coefficients. RL-based grasping
    works from raw sensor input (images/point clouds) and generalizes
    to novel objects never seen during training — if the training
    distribution is diverse enough.
    """

    def __init__(self, n_objects=5):
        self.n_objects = n_objects
        self.workspace = {'x': (-0.3, 0.3), 'y': (-0.3, 0.3), 'z': (0.0, 0.3)}

    def reset(self):
        """Reset environment with random objects.

        Why randomize objects? The goal is a general grasping policy,
        not one that works for a specific object. Training with diverse
        objects (different shapes, sizes, masses, textures) produces
        a policy that generalizes to new objects at test time.
        """
        self.objects = []
        for _ in range(self.n_objects):
            obj = {
                'position': np.random.uniform(
                    [self.workspace['x'][0], self.workspace['y'][0], 0.02],
                    [self.workspace['x'][1], self.workspace['y'][1], 0.02]
                ),
                'size': np.random.uniform(0.02, 0.08),
                'mass': np.random.uniform(0.05, 0.5),
                'shape': np.random.choice(['box', 'cylinder', 'sphere']),
            }
            self.objects.append(obj)

        self.gripper_pos = np.array([0.0, 0.0, 0.3])
        self.gripper_open = True
        return self._get_observation()

    def step(self, action):
        """Execute grasping action and return next observation, reward."""
        dx, dy, dz, d_close = action

        # Move gripper
        self.gripper_pos += np.array([dx, dy, dz]) * 0.05  # Scale actions

        # Gripper open/close
        if d_close > 0.5:
            self.gripper_open = False
        else:
            self.gripper_open = True

        # Check grasp success
        reward = self._compute_reward()
        done = self._check_done()

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """Return observation: gripper state + simplified scene."""
        obs = np.concatenate([
            self.gripper_pos,
            [float(self.gripper_open)],
            # In a real system: RGB-D image, point cloud, or feature vector
        ])
        return obs

    def _compute_reward(self):
        """Reward for grasping."""
        reward = 0.0
        for obj in self.objects:
            dist = np.linalg.norm(self.gripper_pos[:2] - obj['position'][:2])
            reward += 0.1 * (1.0 - np.tanh(5.0 * dist))  # Approach reward
        return reward

    def _check_done(self):
        """Check if episode is over."""
        return False  # Simplified
```

### 5.2 Assembly Policies

RL for assembly tasks (peg-in-hole, connector insertion) typically uses:

- **Force/torque observations**: The contact forces provide critical information about alignment
- **Impedance/compliance actions**: Rather than commanding positions directly, the RL agent commands impedance parameters (stiffness, damping)
- **Residual RL**: Start with a classical controller and learn a residual policy that corrects its behavior

```python
def residual_policy_concept(classical_action, learned_residual):
    """Residual RL: classical controller + learned correction.

    Why residual? Starting from scratch, RL for assembly would need
    millions of episodes to discover that the peg should move toward
    the hole. A classical controller already knows this — it handles
    90% of the task. The RL agent only needs to learn the remaining
    10%: the subtle force adjustments during insertion that are hard
    to program analytically.

    final_action = classical_controller(state) + alpha * rl_policy(state)
    """
    alpha = 0.1  # Limit RL's influence for safety
    return classical_action + alpha * learned_residual
```

---

## 6. RL for Locomotion

### 6.1 Legged Robot Locomotion

RL has achieved remarkable results in legged locomotion:

- **MIT Mini Cheetah**: RL-trained policy runs at 3.7 m/s
- **ANYmal**: RL policy traverses stairs, slopes, and rough terrain
- **Atlas** (Boston Dynamics): Uses RL for parkour and acrobatic movements

**Observation space** for a quadruped:

| Component | Dimension | Description |
|-----------|-----------|-------------|
| Base orientation | 3 | Roll, pitch, yaw |
| Base angular velocity | 3 | From IMU |
| Joint positions | 12 | 3 joints x 4 legs |
| Joint velocities | 12 | From encoders |
| Previous actions | 12 | For smoothness |
| Command velocity | 3 | Desired $v_x, v_y, \omega_z$ |

**Action space**: Target joint positions (12 DOF), executed by a low-level PD controller at each joint.

### 6.2 Training Pipeline

```python
class LocomotionTrainer:
    """Conceptual training pipeline for legged robot locomotion.

    Why PPO for locomotion? PPO (Proximal Policy Optimization) is
    the de facto standard for robot locomotion RL because:
    1. Stable training (clipped surrogate objective prevents large updates)
    2. Works with continuous action spaces
    3. Parallelizes well (thousands of environments simultaneously)
    4. Handles the high-dimensional observation/action spaces of legged robots

    PPO's clipped surrogate objective:
      L_CLIP = E_t[min(r_t(theta) * A_hat_t,
                       clip(r_t(theta), 1-eps, 1+eps) * A_hat_t)]
    where:
      r_t(theta) = pi_new(a_t|s_t) / pi_old(a_t|s_t)  — probability ratio
        between new and old policies; measures how much the policy changed
      A_hat_t — advantage estimate (how much better action a_t was than
        average, typically computed via GAE: Generalized Advantage Estimation)
      eps — clip range (typically 0.2), prevents r_t from deviating too
        far from 1.0, ensuring small, stable policy updates
    """

    def __init__(self, n_envs=4096):
        self.n_envs = n_envs  # Parallel environments

        # Training typically runs for 1-10 billion environment steps
        # At 4096 parallel envs x 50 Hz, this takes 6-60 hours on GPU

    def train_config(self):
        """Standard training configuration for quadruped locomotion."""
        return {
            'algorithm': 'PPO',
            'n_parallel_envs': 4096,
            'episode_length': 1000,     # 20 seconds at 50 Hz
            'learning_rate': 3e-4,
            'entropy_coefficient': 0.01, # Encourage exploration
            'clip_range': 0.2,
            'n_epochs_per_update': 5,
            'minibatch_size': 4096,

            # Domain randomization
            'randomize_friction': (0.3, 1.2),
            'randomize_mass': (0.8, 1.2),
            'randomize_motor_strength': (0.85, 1.15),
            'randomize_terrain': True,

            # Curriculum: start on flat ground, progress to rough terrain
            'terrain_curriculum': True,
            'max_terrain_difficulty': 0.8,
        }

    def reward_function(self, state, action, next_state):
        """Multi-objective locomotion reward.

        Why so many reward terms? Without careful reward design, the
        agent finds degenerate solutions: dragging instead of walking,
        vibrating joints to move forward, or galloping in a way that
        would destroy real actuators. Each penalty term rules out a
        class of undesirable behaviors.
        """
        reward = 0.0

        # Primary: track commanded velocity
        vel_cmd = state['command_velocity']
        vel_actual = next_state['base_linear_velocity']
        reward += 2.0 * np.exp(-4.0 * np.sum((vel_actual[:2] - vel_cmd[:2])**2))

        # Penalties for unnatural behavior
        reward -= 0.001 * np.sum(action**2)           # Energy
        reward -= 0.01 * np.sum(state['joint_velocities']**2)  # Smoothness
        reward -= 0.1 * abs(next_state['base_orientation'][0])  # Roll
        reward -= 0.1 * abs(next_state['base_orientation'][1])  # Pitch
        reward -= 0.5 * max(0, 0.15 - next_state['base_height'])  # Don't fall

        # Foot clearance: encourage lifting feet
        for foot in range(4):
            if not next_state['foot_contact'][foot]:
                reward += 0.01 * max(0, next_state['foot_height'][foot] - 0.02)

        return reward
```

---

## 7. Simulation Environments

### 7.1 MuJoCo

**MuJoCo** (Multi-Joint dynamics with Contact) is the most widely used physics simulator for robot RL:

- Fast, accurate contact simulation
- Differentiable (MuJoCo XLA)
- Supports soft contacts, tendons, and complex mechanisms
- Standard benchmark tasks (Ant, Humanoid, Hand, etc.)

### 7.2 Isaac Gym / Isaac Sim

**NVIDIA Isaac Gym** enables massively parallel simulation on GPUs:

- 4096+ environments simultaneously on a single GPU
- GPU-accelerated physics and rendering
- 10-100x faster than CPU-based simulators
- Tight integration with PyTorch for end-to-end GPU training

```python
def isaac_gym_concept():
    """Conceptual Isaac Gym training setup.

    Why GPU simulation? Traditional RL for robotics is bottlenecked
    by simulation speed: each environment step takes milliseconds on
    CPU. Isaac Gym moves the entire pipeline to GPU — physics, reward
    computation, policy inference, and gradient updates — achieving
    orders-of-magnitude speedup. Training that took days on CPU takes
    hours or minutes on GPU.
    """
    # Pseudocode — actual Isaac Gym API
    # env = IsaacGymEnv(
    #     num_envs=4096,
    #     env_spacing=2.0,
    #     sim_device='cuda:0',
    #     compute_device='cuda:0',
    # )
    #
    # # All 4096 environments step simultaneously on GPU
    # obs = env.reset()
    # for step in range(total_steps):
    #     actions = policy(obs)           # GPU inference
    #     obs, rewards, dones, infos = env.step(actions)  # GPU physics
    #     policy.update(obs, rewards)     # GPU gradient update
    pass
```

### 7.3 Gymnasium (Farama Foundation)

**Gymnasium** (formerly OpenAI Gym) is the standard API for RL environments:

```python
import gymnasium as gym

def gymnasium_robot_example():
    """Using Gymnasium for robot RL.

    Gymnasium provides a consistent API that decouples the RL
    algorithm from the environment. The same PPO implementation
    can train on CartPole, a MuJoCo robot, or a real robot — only
    the environment changes.
    """
    # Classic control (for learning RL basics)
    env = gym.make('CartPole-v1')

    # MuJoCo robots (requires mujoco package)
    # env = gym.make('Ant-v4')
    # env = gym.make('Humanoid-v4')

    obs, info = env.reset()

    for step in range(1000):
        action = env.action_space.sample()  # Random policy
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
```

### 7.4 Simulator Comparison

| Feature | MuJoCo | Isaac Gym | PyBullet | Gazebo |
|---------|--------|-----------|----------|--------|
| Speed | Fast (CPU) | Very fast (GPU) | Moderate (CPU) | Slow |
| Parallelism | Multi-process | GPU-native (4096+) | Multi-process | Limited |
| Contact accuracy | Excellent | Good | Good | Moderate |
| Rendering | Built-in | GPU-accelerated | OpenGL | Gazebo/Ogre |
| ROS2 integration | Limited | Via Isaac Sim | Via ROS bridge | Native |
| License | Free (Apache 2.0) | Free (NVIDIA) | Free (BSD) | Free (Apache 2.0) |
| Best for | Research, benchmarks | Large-scale training | Quick prototyping | Full robot system |

---

## 8. Practical Considerations

### 8.1 Observation Design

What the agent observes dramatically affects learning performance:

| Observation Choice | Pros | Cons |
|-------------------|------|------|
| Joint angles + velocities | Low-dimensional, easy to learn | No environment awareness |
| Raw images (pixels) | Rich information | High-dimensional, slow to learn |
| Point clouds | 3D geometry | Variable size, requires special architectures |
| Privileged info (in sim) + distilled | Fast training | Two-phase training |

**Observation normalization** is critical for training stability. Raw observations often have wildly different scales (joint angles in radians vs. velocities in rad/s vs. forces in Newtons). Normalizing with running statistics `obs = (obs - obs_mean) / (obs_std + 1e-8)` centers the inputs and scales them to unit variance, which helps the neural network learn efficiently. The small constant `1e-8` prevents division by zero when a feature has zero variance (e.g., a constant sensor reading in early training).

**Asymmetric actor-critic**: Use privileged information (exact object pose, contact forces) for the critic during training, but only use real-sensor-observable features for the actor. This speeds up learning while maintaining deployability.

### 8.2 Action Space Design

| Action Type | Description | Stability | Precision |
|-------------|-------------|-----------|-----------|
| Joint torques | Direct torque commands | Low (dangerous) | High (theoretically) |
| Joint position targets | PD controller tracks targets | High (PD stabilizes) | Moderate |
| End-effector deltas | Cartesian velocity commands | High | Task-dependent |
| Impedance parameters | Stiffness/damping setpoints | Very high | Task-dependent |

**Best practice**: Use joint position targets with a PD controller. The RL agent outputs desired joint positions, and a low-level PD controller computes the torques. This is much safer than direct torque control and provides a natural action smoothness.

**A note on SAC (Soft Actor-Critic)**: An alternative to PPO for continuous-action robotics tasks, SAC maximizes the entropy-regularized objective $J(\pi) = \sum_t \mathbb{E}[r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))]$, where $\alpha$ is the **temperature parameter** that controls the exploration-exploitation trade-off. Physically, $\alpha$ determines how much the agent values randomness: high $\alpha$ encourages broad exploration (useful in early training or contact-rich tasks), while low $\alpha$ focuses on exploiting known good actions. Modern SAC implementations auto-tune $\alpha$ to maintain a target entropy, removing the need for manual adjustment.

---

## Summary

| Concept | Key Idea |
|---------|----------|
| RL for robotics | Learn from interaction when models are too complex to derive analytically |
| Sim-to-real gap | Physical parameter mismatches between simulator and reality |
| Domain randomization | Train across wide parameter range so policy is robust to reality |
| System identification | Estimate real parameters to calibrate simulator |
| Reward shaping | Multi-component dense rewards guide learning; sparse rewards are insufficient |
| Curriculum learning | Start easy, increase difficulty as the agent improves |
| Safe RL | Constrained MDP + safety layers to prevent dangerous exploration |
| Residual RL | Learn corrections to a classical controller for the best of both worlds |
| GPU simulation | Isaac Gym enables 4096+ parallel environments for fast training |
| Action space design | Joint position targets with PD controller for safety and stability |

---

## Exercises

1. **Reward function design**: Design a reward function for a robot arm task: pick up a ball and place it in a cup. Define at least 4 reward components (reaching, grasping, transporting, placing). Implement the reward function and explain how each component guides the desired behavior. What could go wrong (reward hacking)?

2. **Domain randomization study**: Implement a simple 2D reaching task in Python (a point agent reaching a target). Train a policy with fixed dynamics, then evaluate it when the dynamics are perturbed (different damping). Repeat with domain randomization during training. Compare the transfer performance.

3. **Safety layer implementation**: Implement a safety layer for a 2-DOF robot arm that enforces: (a) joint position limits, (b) joint velocity limits, and (c) maximum end-effector speed. Show that an RL agent with the safety layer never violates the constraints, while one without it does.

4. **Curriculum design**: Design a curriculum for teaching a simulated robot to open a door. Identify 3-4 difficulty levels (e.g., door ajar vs. fully closed, light vs. heavy door, with/without handle). Explain the progression and why each level prepares the agent for the next.

5. **Simulator comparison**: Using Gymnasium, create the same reaching task in two environments (e.g., CartPole and a custom 2D environment). Train PPO on both and compare: training time, final performance, and sensitivity to hyperparameters. What does this tell you about the role of the environment in RL training efficiency?

---

## Further Reading

- Tobin, J. et al. "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." *IROS*, 2017. (Foundational domain randomization paper)
- Peng, X. B. et al. "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization." *ICRA*, 2018. (Dynamics randomization for locomotion)
- OpenAI. "Solving Rubik's Cube with a Robot Hand." 2019. (Landmark sim-to-real result)
- Rudin, N. et al. "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning." *CoRL*, 2022. (Isaac Gym for locomotion)
- Garcıa, J. and Fernandez, F. "A Comprehensive Survey on Safe Reinforcement Learning." *JMLR*, 2015. (Safe RL survey)
- Johannink, T. et al. "Residual Reinforcement Learning for Robot Manipulators." *ICRA*, 2019. (Residual RL)

---

[← Previous: ROS2 Navigation Stack](14_ROS2_Navigation.md) | [Next: Multi-Robot Systems and Swarms →](16_Multi_Robot_Systems.md)
