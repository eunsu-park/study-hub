[Previous: Inverse RL](./22_Inverse_RL.md)

---

# 23. RL for Robotics

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the sim-to-real gap and why direct RL on real robots is challenging
2. Implement domain randomization for robust sim-to-real transfer
3. Describe MuJoCo and Isaac Gym simulation environments for robotic RL
4. Build RL pipelines for locomotion and manipulation tasks
5. Apply system identification and fine-tuning techniques for real robot deployment

---

## Table of Contents

1. [RL Meets Robotics](#1-rl-meets-robotics)
2. [Simulation Environments](#2-simulation-environments)
3. [Sim-to-Real Transfer](#3-sim-to-real-transfer)
4. [Domain Randomization](#4-domain-randomization)
5. [RL for Locomotion](#5-rl-for-locomotion)
6. [RL for Manipulation](#6-rl-for-manipulation)
7. [Real-World Deployment](#7-real-world-deployment)
8. [Exercises](#8-exercises)

---

## 1. RL Meets Robotics

### 1.1 Why RL for Robotics?

```
Traditional Robotics:
  Hand-designed controllers for each task
  + Precise, reliable for known scenarios
  - Brittle to new situations
  - Enormous engineering effort per task

RL Robotics:
  Learn controllers from experience
  + Adapts to new situations
  + One algorithm, many tasks
  - Sample efficiency (millions of trials)
  - Safety during training
  - Sim-to-real gap
```

### 1.2 Challenges Unique to Robotics

```
Challenge              | Impact              | Solution
-----------------------|---------------------|-------------------
Sample efficiency      | Can't do 10M trials | Sim-to-real, model-based RL
Safety                 | Robot can break      | Sim training, safe RL
Partial observability  | Sensors are noisy    | Recurrent policies, filtering
High-dimensional action| 20+ joints           | Curriculum, hierarchical RL
Continuous control     | Smooth movements     | SAC, PPO with continuous actions
Contact dynamics       | Hard to simulate     | Domain randomization
Latency               | 10-50ms delays        | Include delay in training
```

### 1.3 The Sim-to-Real Pipeline

```
Standard pipeline:

1. Build simulator (MuJoCo, Isaac Gym, PyBullet)
      │
2. Train policy in simulation (millions of episodes, fast)
      │
3. Domain randomization (vary physics, visuals)
      │
4. Transfer to real robot (zero-shot or fine-tune)
      │
5. Optional: Fine-tune on real robot (few episodes)
```

---

## 2. Simulation Environments

### 2.1 MuJoCo

```python
import gymnasium as gym

# MuJoCo environments (now free and open source!)
# pip install mujoco gymnasium[mujoco]

# Locomotion
ant_env = gym.make('Ant-v4')
humanoid_env = gym.make('Humanoid-v4')
half_cheetah_env = gym.make('HalfCheetah-v4')

# Manipulation
# Gymnasium Robotics provides Fetch and Shadow Hand
fetch_env = gym.make('FetchReach-v3')

# MuJoCo state: joint positions + velocities
obs, _ = ant_env.reset()
print(f"Ant observation dim: {obs.shape}")  # (27,)
print(f"Ant action dim: {ant_env.action_space.shape}")  # (8,)
```

### 2.2 Isaac Gym (GPU-Accelerated)

```python
# Isaac Gym runs THOUSANDS of environments in parallel on GPU
# Speedup: 100-1000x over CPU simulation

# Conceptual example (actual API differs)
class IsaacGymConfig:
    num_envs = 4096          # 4096 parallel environments
    sim_device = 'cuda:0'
    physics_engine = 'PhysX'

    # Physics parameters
    gravity = [0, 0, -9.81]
    dt = 1.0 / 60.0
    substeps = 2

    # Domain randomization built-in
    randomize_friction = True
    friction_range = [0.5, 1.5]
    randomize_mass = True
    mass_range = [0.8, 1.2]


# Training throughput comparison:
# MuJoCo (CPU):      ~1,000 steps/second
# IsaacGym (1 GPU):  ~1,000,000 steps/second
# IsaacGym (8 GPU):  ~8,000,000 steps/second
```

### 2.3 Environment Comparison

| Feature | MuJoCo | Isaac Gym | PyBullet |
|---------|--------|-----------|----------|
| **Speed** | ~1K steps/s | ~1M steps/s | ~500 steps/s |
| **GPU** | No | Yes (native) | Optional |
| **Parallel envs** | Process-based | GPU-native | Process-based |
| **Contact physics** | Excellent | Good | Good |
| **License** | Free (Apache 2.0) | Free (NVIDIA) | Free (zlib) |
| **Rendering** | MuJoCo renderer | NVIDIA renderer | OpenGL |
| **Ecosystem** | Largest | Growing fast | Medium |

---

## 3. Sim-to-Real Transfer

### 3.1 The Sim-to-Real Gap

```
Why does a policy trained in simulation fail on real robots?

Dynamics gap:
  Simulation: Perfect rigid bodies, simple friction
  Reality: Flexible materials, complex contacts, wear

Observation gap:
  Simulation: Clean state vectors, perfect images
  Reality: Noisy sensors, occlusions, calibration error

Action gap:
  Simulation: Perfect actuation, zero latency
  Reality: Motor delays, backlash, torque limits

Visual gap:
  Simulation: Perfect lighting, textures
  Reality: Variable lighting, reflections, clutter
```

### 3.2 Transfer Methods Taxonomy

```python
def sim_to_real_methods():
    """Overview of sim-to-real transfer approaches."""
    methods = {
        'Domain Randomization': {
            'idea': 'Randomize sim parameters so policy is robust',
            'effort': 'Low (just vary parameters)',
            'effectiveness': 'Good for dynamics, great for vision',
        },
        'System Identification': {
            'idea': 'Make simulation match reality precisely',
            'effort': 'High (measure real parameters)',
            'effectiveness': 'Excellent when done well',
        },
        'Domain Adaptation': {
            'idea': 'Learn to map sim observations to real',
            'effort': 'Medium (needs some real data)',
            'effectiveness': 'Good for visual transfer',
        },
        'Real-World Fine-tuning': {
            'idea': 'Fine-tune sim-trained policy on real robot',
            'effort': 'Medium (needs real robot time)',
            'effectiveness': 'Best final performance',
        },
        'Sim-to-Sim Transfer': {
            'idea': 'Transfer between different simulators first',
            'effort': 'Low',
            'effectiveness': 'Good sanity check',
        },
    }

    for name, info in methods.items():
        print(f"{name}:")
        print(f"  Idea:          {info['idea']}")
        print(f"  Effort:        {info['effort']}")
        print(f"  Effectiveness: {info['effectiveness']}")
        print()

sim_to_real_methods()
```

---

## 4. Domain Randomization

### 4.1 Physics Randomization

```python
import numpy as np


class DomainRandomizer:
    """Randomize simulation parameters for robust transfer."""

    def __init__(self):
        self.params = {
            # Dynamics
            'gravity': {'default': -9.81, 'range': (-11.0, -8.0)},
            'friction': {'default': 1.0, 'range': (0.5, 2.0)},
            'mass_scale': {'default': 1.0, 'range': (0.8, 1.2)},
            'damping': {'default': 0.5, 'range': (0.1, 1.0)},

            # Actuator
            'motor_strength': {'default': 1.0, 'range': (0.8, 1.2)},
            'action_delay': {'default': 0, 'range': (0, 3)},  # steps
            'action_noise': {'default': 0.0, 'range': (0.0, 0.05)},

            # Sensor
            'observation_noise': {'default': 0.0, 'range': (0.0, 0.02)},
            'sensor_delay': {'default': 0, 'range': (0, 2)},  # steps
        }

    def sample(self):
        """Sample a random set of parameters."""
        sampled = {}
        for name, config in self.params.items():
            low, high = config['range']
            if isinstance(config['default'], int):
                sampled[name] = np.random.randint(low, high + 1)
            else:
                sampled[name] = np.random.uniform(low, high)
        return sampled

    def apply_to_env(self, env, params):
        """Apply randomized parameters to environment."""
        # Implementation depends on simulator API
        if hasattr(env, 'model'):  # MuJoCo
            model = env.model
            # Scale body masses
            for i in range(model.nbody):
                model.body_mass[i] *= params['mass_scale']

            # Set friction
            for i in range(model.ngeom):
                model.geom_friction[i][0] = params['friction']

            # Set gravity
            model.opt.gravity[2] = params['gravity']


class RandomizedEnvWrapper:
    """Wrapper that randomizes environment at each reset."""

    def __init__(self, env_fn, randomizer):
        self.env_fn = env_fn
        self.randomizer = randomizer
        self.env = env_fn()

    def reset(self, **kwargs):
        # Re-create environment with new random parameters
        params = self.randomizer.sample()
        self.env = self.env_fn()
        self.randomizer.apply_to_env(self.env, params)
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
```

### 4.2 Visual Domain Randomization

```python
class VisualRandomizer:
    """Randomize visual appearance for sim-to-real transfer."""

    def __init__(self):
        self.params = {
            'light_position': {'range': [(-2, 2), (-2, 2), (1, 4)]},
            'light_intensity': {'range': (0.3, 1.0)},
            'camera_position_noise': {'range': (-0.05, 0.05)},
            'texture_randomize': True,
            'background_randomize': True,
            'object_color_range': {'range': (0.0, 1.0)},
        }

    def randomize_scene(self, scene):
        """Apply visual randomizations to the scene."""
        # Random lighting
        for light in scene.lights:
            light.position += np.random.uniform(
                *self.params['light_position']['range']
            )
            light.intensity = np.random.uniform(
                *self.params['light_intensity']['range']
            )

        # Random object colors
        for obj in scene.objects:
            obj.color = np.random.uniform(0, 1, 3)

        # Random camera perturbation
        noise = np.random.uniform(
            *self.params['camera_position_noise']['range'], size=3
        )
        scene.camera.position += noise

        return scene
```

### 4.3 Automatic Domain Randomization (ADR)

```
ADR (OpenAI, 2019 - used for Rubik's cube solving):

Instead of hand-tuning randomization ranges:
1. Start with narrow ranges (close to default)
2. If policy succeeds > threshold: WIDEN ranges
3. If policy fails too often: NARROW ranges
4. Automatically finds the right level of randomization

Result: Policy trained with ADR can handle
real-world variations it never explicitly saw.
```

```python
class AutomaticDomainRandomization:
    """ADR: Automatically adjust randomization ranges."""

    def __init__(self, initial_ranges, step_size=0.02, success_threshold=0.8):
        self.ranges = {k: list(v) for k, v in initial_ranges.items()}
        self.step_size = step_size
        self.success_threshold = success_threshold
        self.success_history = {k: [] for k in initial_ranges}

    def update(self, param_name, success):
        """Update range based on success/failure."""
        self.success_history[param_name].append(success)

        # Use last 100 episodes
        recent = self.success_history[param_name][-100:]
        if len(recent) < 100:
            return

        success_rate = np.mean(recent)
        low, high = self.ranges[param_name]

        if success_rate > self.success_threshold:
            # Too easy -> widen range
            self.ranges[param_name] = [
                low - self.step_size,
                high + self.step_size
            ]
        elif success_rate < self.success_threshold - 0.2:
            # Too hard -> narrow range
            self.ranges[param_name] = [
                low + self.step_size * 0.5,
                high - self.step_size * 0.5
            ]

    def sample_params(self):
        """Sample parameters from current ranges."""
        params = {}
        for name, (low, high) in self.ranges.items():
            params[name] = np.random.uniform(low, high)
        return params
```

---

## 5. RL for Locomotion

### 5.1 Locomotion Tasks

```
Common locomotion benchmarks:

HalfCheetah: 2D running (6 joints)
  Reward: forward velocity - control cost
  Typical PPO return: ~6000

Ant: 4-legged walking (8 joints)
  Reward: forward velocity - control cost - contact cost
  Typical PPO return: ~5000

Humanoid: Bipedal walking (17 joints)
  Reward: forward velocity - control cost
  Typical PPO return: ~6000
  Much harder to train than quadrupeds!

Walker2d: 2D bipedal walking (6 joints)
  Reward: forward velocity - control cost
  Typical PPO return: ~5000
```

### 5.2 Locomotion Policy Training

```python
import torch
import torch.nn as nn
import numpy as np


class LocomotionPolicy(nn.Module):
    """MLP policy for locomotion tasks."""

    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ELU()])
            prev_dim = dim

        self.backbone = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        features = self.backbone(obs)
        mean = self.mean(features)
        std = self.log_std.exp()
        return torch.distributions.Normal(mean, std)

    def get_action(self, obs, deterministic=False):
        dist = self.forward(obs)
        if deterministic:
            return dist.mean
        return dist.sample()


def train_locomotion(env_name='Ant-v4', total_steps=10_000_000):
    """Train locomotion policy with PPO."""
    import gymnasium as gym

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = LocomotionPolicy(obs_dim, act_dim)

    # PPO hyperparameters for locomotion
    config = {
        'lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'epochs_per_update': 10,
        'batch_size': 64,
        'steps_per_update': 2048,
    }

    print(f"Training {env_name}: obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"Total steps: {total_steps:,}")
    # ... PPO training loop ...

    return policy
```

### 5.3 Terrain-Aware Locomotion

```python
class TerrainGenerator:
    """Generate diverse terrains for robust locomotion training."""

    def __init__(self, terrain_types=None):
        self.terrain_types = terrain_types or [
            'flat', 'rough', 'slopes', 'stairs', 'gaps'
        ]

    def generate(self, terrain_type='random'):
        if terrain_type == 'random':
            terrain_type = np.random.choice(self.terrain_types)

        heightmap = np.zeros((100, 100))

        if terrain_type == 'flat':
            pass  # already flat

        elif terrain_type == 'rough':
            heightmap += np.random.normal(0, 0.03, heightmap.shape)

        elif terrain_type == 'slopes':
            x = np.linspace(0, 1, 100)
            slope = np.random.uniform(0.1, 0.3)
            heightmap += np.outer(slope * x, np.ones(100))

        elif terrain_type == 'stairs':
            step_height = np.random.uniform(0.05, 0.15)
            for i in range(0, 100, 10):
                heightmap[i:, :] += step_height

        elif terrain_type == 'gaps':
            for i in range(0, 100, 20):
                gap_width = np.random.randint(2, 5)
                heightmap[i:i+gap_width, :] = -0.5

        return heightmap, terrain_type
```

---

## 6. RL for Manipulation

### 6.1 Dexterous Manipulation

```
Dexterous hand manipulation is one of the hardest RL problems:

Shadow Hand:  24 joints, 20 actuated
Task: In-hand object rotation (e.g., Rubik's cube)

Challenges:
  - Very high dimensional action space
  - Contact-rich dynamics (fingertips on object)
  - Precise coordination needed
  - Object can be dropped (catastrophic failure)

OpenAI solution (2019):
  - PPO + LSTM policy
  - Automatic Domain Randomization
  - 100+ CPU years of simulation
  - Transferred to real Shadow Hand
```

### 6.2 Reward Design for Manipulation

```python
def manipulation_reward(state, action, goal, info):
    """Multi-component reward for pick-and-place."""
    gripper_pos = state['gripper_pos']
    object_pos = state['object_pos']
    goal_pos = goal

    # Component 1: Reach the object
    reach_dist = np.linalg.norm(gripper_pos - object_pos)
    reach_reward = -reach_dist

    # Component 2: Grasp the object
    is_grasped = info.get('is_grasped', False)
    grasp_reward = 1.0 if is_grasped else 0.0

    # Component 3: Move to goal (only when grasped)
    if is_grasped:
        place_dist = np.linalg.norm(object_pos - goal_pos)
        place_reward = -place_dist
    else:
        place_reward = 0.0

    # Component 4: Success bonus
    success = np.linalg.norm(object_pos - goal_pos) < 0.05
    success_reward = 10.0 if success else 0.0

    # Combine with potential-based shaping for reach and place
    total = (0.1 * reach_reward + 0.5 * grasp_reward +
             1.0 * place_reward + success_reward)

    return total
```

---

## 7. Real-World Deployment

### 7.1 Deployment Checklist

```
Before deploying RL policy on real robot:

[ ] Safety
    [ ] Emergency stop accessible
    [ ] Joint/torque limits enforced in hardware
    [ ] Action clipping in policy
    [ ] Slow initial execution (reduce speed)

[ ] Policy validation
    [ ] Tested in multiple sim variations
    [ ] Success rate > 90% in hardest randomization
    [ ] Robust to observation noise injection
    [ ] Handles action delay gracefully

[ ] Sensor calibration
    [ ] Camera intrinsics/extrinsics calibrated
    [ ] Joint encoders verified
    [ ] Force/torque sensors zeroed

[ ] Gradual deployment
    [ ] Start with simple tasks
    [ ] Increase complexity gradually
    [ ] Monitor for distribution shift
    [ ] Have fallback controller ready
```

### 7.2 Online Adaptation

```python
class RealWorldAdapter:
    """Adapt sim-trained policy to real robot online."""

    def __init__(self, sim_policy, adaptation_lr=1e-4):
        self.policy = sim_policy
        self.adaptation_buffer = []
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=adaptation_lr
        )

    def collect_and_adapt(self, real_env, n_episodes=5):
        """Collect real data and fine-tune."""
        for ep in range(n_episodes):
            state, _ = real_env.reset()
            episode_data = []

            for step in range(100):
                action = self.policy.get_action(
                    torch.FloatTensor(state), deterministic=True
                ).detach().numpy()

                next_state, reward, done, truncated, info = real_env.step(action)
                episode_data.append((state, action, reward, next_state, done))
                state = next_state

                if done or truncated:
                    break

            self.adaptation_buffer.extend(episode_data)

        # Fine-tune with collected real data
        self._fine_tune(n_updates=100)

    def _fine_tune(self, n_updates=100, batch_size=32):
        """Fine-tune policy on real-world data."""
        for _ in range(n_updates):
            batch = self._sample_batch(batch_size)
            # Standard RL update (e.g., SAC or PPO)
            # ... update policy ...
```

---

## 8. Exercises

### Exercise 1: Domain Randomization Study

Systematically study domain randomization effects:
1. Train a HalfCheetah policy WITHOUT randomization
2. Add physics randomization (mass, friction, gravity)
3. Compare performance on "nominal" and "perturbed" environments
4. Plot success rate vs randomization range width
5. Find the sweet spot: too little randomization = no transfer, too much = poor nominal

### Exercise 2: Sim-to-Sim Transfer

Practice transfer between simulators:
1. Train a locomotion policy in one environment setting
2. Evaluate on environments with different physics parameters
3. Apply domain randomization and re-evaluate
4. Measure the "transferability gap" with and without randomization
5. Implement ADR and show it automatically finds good ranges

### Exercise 3: Locomotion on Diverse Terrain

Build an adaptive locomotion agent:
1. Create 5 terrain types: flat, rough, slopes, stairs, gaps
2. Train a single policy on randomized terrain selection
3. Compare with: (a) flat-only training, (b) per-terrain specialists
4. Measure adaptation speed when encountering a new terrain type
5. Visualize gait changes across different terrains

### Exercise 4: Pick-and-Place Pipeline

Build an end-to-end manipulation pipeline:
1. Set up FetchPickAndPlace environment
2. Design a multi-component reward function
3. Train with HER and goal-conditioned DDPG
4. Add domain randomization for object properties
5. Report success rate across object sizes and positions

### Exercise 5: Real-World Simulation Challenge

Simulate a real-world deployment scenario:
1. Train policy in "clean" simulation
2. Create a "real" environment with added noise, delays, and model mismatch
3. Measure zero-shot transfer performance
4. Apply domain randomization and measure improvement
5. Implement online adaptation with 10 "real" episodes and show recovery

---

*End of Lesson 23*
