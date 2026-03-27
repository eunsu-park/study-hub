[이전: Inverse RL](./22_Inverse_RL.md)

---

# 23. 로보틱스를 위한 RL

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 시뮬레이션-실제 간극과 실제 로봇에서 직접 RL이 어려운 이유 설명
2. 견고한 시뮬레이션-실제 전이를 위한 도메인 랜덤화 구현
3. 로봇 RL을 위한 MuJoCo와 Isaac Gym 시뮬레이션 환경 설명
4. 이동 및 조작 과제를 위한 RL 파이프라인 구축
5. 실제 로봇 배포를 위한 시스템 식별 및 미세 조정 기법 적용

---

## 목차

1. [RL과 로보틱스의 만남](#1-rl과-로보틱스의-만남)
2. [시뮬레이션 환경](#2-시뮬레이션-환경)
3. [시뮬레이션-실제 전이](#3-시뮬레이션-실제-전이)
4. [도메인 랜덤화](#4-도메인-랜덤화)
5. [이동을 위한 RL](#5-이동을-위한-rl)
6. [조작을 위한 RL](#6-조작을-위한-rl)
7. [실제 배포](#7-실제-배포)
8. [연습문제](#8-연습문제)

---

## 1. RL과 로보틱스의 만남

### 1.1 왜 로보틱스에 RL을 사용하는가?

```
전통적 로보틱스:
  각 과제에 대한 수동 설계 제어기
  + 정확하고, 알려진 시나리오에서 신뢰할 수 있음
  - 새로운 상황에 취약
  - 과제당 엄청난 엔지니어링 노력

RL 로보틱스:
  경험으로부터 제어기 학습
  + 새로운 상황에 적응
  + 하나의 알고리즘, 많은 과제
  - 샘플 효율성 (수백만 번의 시도)
  - 학습 중 안전
  - 시뮬레이션-실제 간극
```

### 1.2 로보틱스 고유의 도전

```
도전                    | 영향                  | 해결책
-----------------------|----------------------|-------------------
샘플 효율성             | 천만 번 시도 불가      | 시뮬-실제, 모델 기반 RL
안전                    | 로봇이 파손될 수 있음   | 시뮬 학습, 안전 RL
부분 관찰성             | 센서가 잡음이 많음      | 순환 정책, 필터링
고차원 행동             | 20개 이상의 관절        | 커리큘럼, 계층적 RL
연속 제어               | 부드러운 움직임         | SAC, 연속 행동 PPO
접촉 역학               | 시뮬레이션이 어려움     | 도메인 랜덤화
지연                    | 10-50ms 지연           | 학습에 지연 포함
```

### 1.3 시뮬레이션-실제 파이프라인

```
표준 파이프라인:

1. 시뮬레이터 구축 (MuJoCo, Isaac Gym, PyBullet)
      │
2. 시뮬레이션에서 정책 학습 (수백만 에피소드, 빠름)
      │
3. 도메인 랜덤화 (물리, 시각적 요소 변화)
      │
4. 실제 로봇으로 전이 (제로샷 또는 미세 조정)
      │
5. 선택사항: 실제 로봇에서 미세 조정 (소수 에피소드)
```

---

## 2. 시뮬레이션 환경

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

### 2.2 Isaac Gym (GPU 가속)

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

### 2.3 환경 비교

| 기능 | MuJoCo | Isaac Gym | PyBullet |
|---------|--------|-----------|----------|
| **속도** | ~1K steps/s | ~1M steps/s | ~500 steps/s |
| **GPU** | 아니오 | 예 (네이티브) | 선택적 |
| **병렬 환경** | 프로세스 기반 | GPU 네이티브 | 프로세스 기반 |
| **접촉 물리** | 우수 | 좋음 | 좋음 |
| **라이선스** | 무료 (Apache 2.0) | 무료 (NVIDIA) | 무료 (zlib) |
| **렌더링** | MuJoCo 렌더러 | NVIDIA 렌더러 | OpenGL |
| **생태계** | 가장 큼 | 빠르게 성장 | 중간 |

---

## 3. 시뮬레이션-실제 전이

### 3.1 시뮬레이션-실제 간극

```
시뮬레이션에서 학습된 정책이 실제 로봇에서 실패하는 이유:

역학 간극:
  시뮬레이션: 완벽한 강체, 단순한 마찰
  현실: 유연한 재료, 복잡한 접촉, 마모

관찰 간극:
  시뮬레이션: 깨끗한 상태 벡터, 완벽한 이미지
  현실: 잡음이 있는 센서, 가려짐, 보정 오류

행동 간극:
  시뮬레이션: 완벽한 구동, 지연 없음
  현실: 모터 지연, 백래시, 토크 제한

시각적 간극:
  시뮬레이션: 완벽한 조명, 텍스처
  현실: 가변 조명, 반사, 어수선함
```

### 3.2 전이 방법 분류

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

## 4. 도메인 랜덤화

### 4.1 물리 랜덤화

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

### 4.2 시각적 도메인 랜덤화

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

### 4.3 자동 도메인 랜덤화 (ADR)

```
ADR (OpenAI, 2019 - 루빅큐브 풀기에 사용):

랜덤화 범위를 수동으로 조정하는 대신:
1. 좁은 범위로 시작 (기본값에 가까움)
2. 정책이 임계값 이상 성공하면: 범위 확장
3. 정책이 너무 자주 실패하면: 범위 축소
4. 자동으로 적절한 랜덤화 수준을 찾음

결과: ADR로 학습된 정책이 명시적으로 보지 못한
실제 세계의 변화를 처리할 수 있음.
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

## 5. 이동을 위한 RL

### 5.1 이동 과제

```
일반적인 이동 벤치마크:

HalfCheetah: 2D 달리기 (6 관절)
  보상: 전진 속도 - 제어 비용
  일반적인 PPO 리턴: ~6000

Ant: 4다리 걷기 (8 관절)
  보상: 전진 속도 - 제어 비용 - 접촉 비용
  일반적인 PPO 리턴: ~5000

Humanoid: 이족 보행 (17 관절)
  보상: 전진 속도 - 제어 비용
  일반적인 PPO 리턴: ~6000
  사족보행보다 학습이 훨씬 어려움!

Walker2d: 2D 이족 보행 (6 관절)
  보상: 전진 속도 - 제어 비용
  일반적인 PPO 리턴: ~5000
```

### 5.2 이동 정책 학습

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

### 5.3 지형 인식 이동

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

## 6. 조작을 위한 RL

### 6.1 민첩한 조작

```
민첩한 손 조작은 가장 어려운 RL 문제 중 하나:

Shadow Hand:  24 관절, 20 구동
과제: 손 안에서 물체 회전 (예: 루빅큐브)

도전:
  - 매우 높은 차원의 행동 공간
  - 접촉이 풍부한 역학 (손끝에서 물체)
  - 정밀한 조정 필요
  - 물체를 떨어뜨릴 수 있음 (치명적 실패)

OpenAI 해결책 (2019):
  - PPO + LSTM 정책
  - 자동 도메인 랜덤화
  - 100+ CPU 년의 시뮬레이션
  - 실제 Shadow Hand로 전이
```

### 6.2 조작을 위한 보상 설계

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

## 7. 실제 배포

### 7.1 배포 체크리스트

```
실제 로봇에 RL 정책을 배포하기 전:

[ ] 안전
    [ ] 비상 정지 접근 가능
    [ ] 관절/토크 한계가 하드웨어에서 적용됨
    [ ] 정책에서 행동 클리핑
    [ ] 느린 초기 실행 (속도 감소)

[ ] 정책 검증
    [ ] 여러 시뮬 변형에서 테스트 완료
    [ ] 가장 어려운 랜덤화에서 성공률 > 90%
    [ ] 관찰 잡음 주입에 대한 견고성
    [ ] 행동 지연을 우아하게 처리

[ ] 센서 보정
    [ ] 카메라 내부/외부 매개변수 보정
    [ ] 관절 인코더 검증
    [ ] 힘/토크 센서 영점 조정

[ ] 점진적 배포
    [ ] 간단한 과제부터 시작
    [ ] 복잡도를 점진적으로 증가
    [ ] 분포 이동 모니터링
    [ ] 대체 제어기 준비
```

### 7.2 온라인 적응

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

## 8. 연습문제

### 연습문제 1: 도메인 랜덤화 연구

도메인 랜덤화 효과를 체계적으로 연구하세요:
1. 랜덤화 없이 HalfCheetah 정책 학습
2. 물리 랜덤화 추가 (질량, 마찰, 중력)
3. "기본" 및 "교란된" 환경에서 성능 비교
4. 랜덤화 범위 폭 대비 성공률 그래프
5. 최적점 찾기: 너무 적은 랜덤화 = 전이 불가, 너무 많으면 = 기본 성능 저하

### 연습문제 2: 시뮬-시뮬 전이

시뮬레이터 간 전이를 연습하세요:
1. 하나의 환경 설정에서 이동 정책 학습
2. 다른 물리 매개변수를 가진 환경에서 평가
3. 도메인 랜덤화를 적용하고 재평가
4. 랜덤화 유무에 따른 "전이 가능성 격차" 측정
5. ADR을 구현하고 자동으로 좋은 범위를 찾는 것을 시연

### 연습문제 3: 다양한 지형에서의 이동

적응형 이동 에이전트를 구축하세요:
1. 5가지 지형 유형 생성: 평탄, 거친, 경사, 계단, 틈
2. 랜덤화된 지형 선택으로 단일 정책 학습
3. 비교: (a) 평탄만 학습, (b) 지형별 전문가
4. 새로운 지형 유형을 만날 때 적응 속도 측정
5. 다양한 지형에 따른 보행 변화 시각화

### 연습문제 4: 집어놓기 파이프라인

종단 간 조작 파이프라인을 구축하세요:
1. FetchPickAndPlace 환경 설정
2. 다중 구성 요소 보상 함수 설계
3. HER과 목표 조건부 DDPG로 학습
4. 물체 속성에 대한 도메인 랜덤화 추가
5. 물체 크기와 위치에 따른 성공률 보고

### 연습문제 5: 실제 세계 시뮬레이션 챌린지

실제 배포 시나리오를 시뮬레이션하세요:
1. "깨끗한" 시뮬레이션에서 정책 학습
2. 잡음, 지연, 모델 불일치가 추가된 "실제" 환경 생성
3. 제로샷 전이 성능 측정
4. 도메인 랜덤화를 적용하고 개선 측정
5. 10개의 "실제" 에피소드로 온라인 적응을 구현하고 회복 시연

---

*23강 끝*
