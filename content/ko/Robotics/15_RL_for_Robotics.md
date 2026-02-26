# 15. 로보틱스를 위한 강화학습(Reinforcement Learning for Robotics)

[← 이전: ROS2 내비게이션 스택](14_ROS2_Navigation.md) | [다음: 다중 로봇 시스템과 군집 →](16_Multi_Robot_Systems.md)

---

## 학습 목표

1. 명시적으로 프로그래밍하기 어려운 로봇 작업에 강화학습(Reinforcement Learning)이 왜 유용한지 설명한다
2. 시뮬레이션-실제 전이(sim-to-real transfer) 문제와 핵심 기법(도메인 무작위화, 시스템 식별)을 이해한다
3. 로봇 조작 및 보행 작업을 위한 보상 함수(reward function)를 설계한다
4. 안전한 RL 개념을 설명한다: 제약 최적화(constrained optimization), 안전한 탐색 전략
5. RL 정책(policy)이 파지(grasping), 조립(assembly), 보행(locomotion)에 어떻게 적용되는지 인식한다
6. 로봇 RL을 위한 주요 시뮬레이션 환경을 비교한다: MuJoCo, Isaac Gym, Gymnasium

---

이 강좌 전반에 걸쳐 우리는 수학적 모델을 사용하여 로봇 컨트롤러를 설계했다: 계산 토크 제어(computed torque control)는 운동 방정식을 사용하고, 임피던스 제어(impedance control)는 원하는 질량-스프링-댐퍼 모델을 사용하며, 내비게이션은 코스트맵(costmap)과 경로 계획 알고리즘을 사용한다. 이러한 모델 기반(model-based) 접근 방식은 정확한 모델이 있을 때 훌륭하게 작동한다. 하지만 모델이 너무 복잡하여 명시적으로 작성할 수 없거나, 환경이 너무 예측 불가능하거나, 최적 전략을 알 수 없는 작업은 어떻게 해야 할까?

루빅스 큐브를 조작하도록 로봇 손을 훈련시키는 경우를 생각해 보자. 손가락과 큐브 사이의 접촉 역학에는 마찰, 변형, 미끄러짐, 그리고 수십 개의 상호 작용하는 표면이 관련된다. 이를 위한 물리 기반 컨트롤러를 작성하는 것은 극도로 어렵다. 하지만 강화학습 에이전트(agent)는 시행착오를 통해 효과적인 정책을 발견할 수 있다 — 어떤 인간 엔지니어도 직접 프로그래밍할 수 없는 것을 수백만 번의 시뮬레이션 시도를 통해 학습한다.

이 레슨은 로보틱스와 RL을 연결하며, RL 에이전트가 물리적 세계에서 작동해야 할 때 발생하는 고유한 과제들에 집중한다: 시뮬레이션에서 현실로의 전이, 학습 중 안전 보장, 그리고 원하는 로봇 행동을 유도하는 보상 함수 설계.

> **비유**: 시뮬레이션-실제 전이(sim-to-real transfer)는 실제 비행기를 조종하기 전에 비행 시뮬레이터로 연습하는 것과 같다. 시뮬레이션이 충분히 현실적이라면 가상 경험이 실제 조종석으로 전이된다. 하지만 시뮬레이터에는 완벽한 날씨만 있고 현실에는 난기류가 있거나, 시뮬레이션의 조종 장치가 실제와 다르게 반응한다면 파일럿의 훈련이 전이되지 않을 수 있다. 도메인 무작위화(domain randomization)는 파일럿을 안개, 비, 측풍 등 다양한 시뮬레이션 조건에서 훈련시켜 현실이 어떤 상황을 던져도 준비가 되도록 하는 것과 같다.

---

## 1. 로보틱스에 RL이 필요한 이유

### 1.1 모델 기반 제어의 한계

전통적인 로보틱스는 다음을 가정한다:
1. 동역학 방정식($M\ddot{q} + C\dot{q} + g = \tau$)을 작성할 수 있다
2. 환경을 모델링할 수 있다 (장애물 위치, 표면 특성)
3. 최적 행동을 명시할 수 있다 (궤적, 임피던스 매개변수)

이러한 가정들은 많은 실제 세계 작업에서 무너진다:

| 과제 | 모델이 실패하는 이유 | RL의 장점 |
|------|---------------------|-----------|
| 접촉이 많은 조작(contact-rich manipulation) | 마찰, 변형을 모델링하기 어려움 | 상호작용 경험으로부터 학습 |
| 변형 가능한 물체(deformable objects) | 무한 차원 상태 | 압축 표현(compact representation) 학습 |
| 새로운 물체 | 사전 모델 없음 | 훈련 분포(training distribution)로부터 일반화 |
| 복잡한 환경 | 너무 많은 변수 | 시행착오를 통한 적응 |
| 정의되지 않은 목표 | "부드럽게 다루기"는 형식화하기 어려움 | 보상 형성(reward shaping)으로 의도 포착 |

### 1.2 로보틱스를 위한 RL 프레임워크

로봇 RL 문제는 마르코프 결정 과정(Markov Decision Process, MDP)으로 공식화된다:

- **상태(state)** $s_t$: 관절 각도, 속도, 물체 자세, 센서 판독값
- **행동(action)** $a_t$: 관절 토크, 속도 명령, 그리퍼 힘
- **전이(transition)** $s_{t+1} \sim p(s_{t+1} | s_t, a_t)$: 로봇 동역학 + 환경
- **보상(reward)** $r_t = R(s_t, a_t, s_{t+1})$: 작업 성공 신호
- **정책(policy)** $\pi(a_t | s_t)$: 관찰에서 행동으로의 학습된 매핑

목표는 기대 누적 보상(expected cumulative reward)을 최대화하는 정책을 찾는 것이다:

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

### 1.3 실제 로봇 vs. 시뮬레이션 학습

| 접근 방식 | 장점 | 단점 |
|-----------|------|------|
| 실제 로봇에서 학습 | 시뮬레이션-실제 격차 없음 | 느림(실시간), 비용 많이 듦, 로봇 손상 가능 |
| 시뮬레이션에서 학습 | 빠름(실시간의 1000배), 안전, 병렬화 가능 | 시뮬레이션-실제 격차 |
| 하이브리드(hybrid) | 두 방식의 장점 | 복잡한 파이프라인 |

**현재 최선의 관행(best practice)**: 시뮬레이션에서 훈련하고 실제 로봇에서 미세 조정(fine-tune)한다. 시뮬레이션 단계가 주요 작업을 수행하고(수백만 에피소드), 실제 세계 단계가 격차를 메운다(수백 에피소드).

---

## 2. 시뮬레이션-실제 전이(Sim-to-Real Transfer)

### 2.1 시뮬레이션-실제 격차(Sim-to-Real Gap)

시뮬레이션에서 훈련된 정책은 시뮬레이터와 현실 사이의 차이로 인해 실제 로봇에서 실패할 수 있다:

| 격차 유형 | 예시 |
|-----------|------|
| **시각적(visual)** | 텍스처, 조명, 반사가 다름 |
| **동적(dynamic)** | 마찰, 감쇠, 질량 분포가 다름 |
| **액추에이터(actuator)** | 모터 응답, 백래시(backlash), 지연이 다름 |
| **센서(sensor)** | 노이즈 특성, 캘리브레이션이 다름 |
| **접촉(contact)** | 표면 특성, 변형이 다름 |

### 2.2 도메인 무작위화(Domain Randomization)

**도메인 무작위화**는 광범위한 시뮬레이션 매개변수에 걸쳐 정책을 훈련하여, 현실을 포함한 어떤 특정 실현에도 강건(robust)하도록 학습시킨다.

$$\theta_{sim} \sim \text{Uniform}(\theta_{min}, \theta_{max})$$

무작위화할 매개변수:

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

### 2.3 시스템 식별(System Identification)

**시스템 식별**은 도메인 무작위화와 반대 접근 방식을 취한다: 모든 매개변수에 강건한 정책을 만드는 대신, 실제 세계 매개변수를 추정하여 현실에 맞게 시뮬레이터를 구성한다.

단계:
1. 실제 세계 궤적 데이터 수집 $(s_t^{real}, a_t^{real}, s_{t+1}^{real})$
2. 다음을 최소화하도록 시뮬레이션 매개변수 최적화:

$$\theta^* = \arg\min_\theta \sum_t \| f_{sim}(s_t, a_t; \theta) - s_{t+1}^{real} \|^2$$

3. 캘리브레이션된 시뮬레이터에서 RL 정책 훈련

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

### 2.4 두 접근 방식의 결합

가장 강건한 접근 방식은 도메인 무작위화와 시스템 식별을 결합한다:

1. **시스템 식별(System ID)**로 무작위화 범위를 현실적인 값 주위로 중심화
2. 식별된 매개변수 주위에서 **도메인 무작위화**를 통해 강건성 확보
3. 남은 격차를 메우기 위해 실제 로봇에서 **미세 조정(fine-tuning)**

---

## 3. 로봇 작업을 위한 보상 형성(Reward Shaping)

### 3.1 보상 설계의 과제

보상 함수(reward function)는 로봇 RL에서 가장 중요한 설계 선택이다. 잘못 설계된 보상은 다음을 초래한다:

- **보상 해킹(reward hacking)**: 에이전트가 의도한 행동을 달성하지 않고 높은 보상을 얻는 방법을 찾음
- **희소 보상(sparse reward)**: 에이전트가 거의 어떤 보상 신호도 받지 못함 (예: 작업 완료 시에만 보상 = 1)
- **의도하지 않은 행동**: 보상이 설계자가 의도한 것과 다른 것을 장려함

### 3.2 보상 형성 전략

**밀집 보상(dense) vs. 희소 보상(sparse)**:

$$r_{sparse}(s) = \begin{cases} 1 & \text{if task complete} \\ 0 & \text{otherwise} \end{cases}$$

$$r_{dense}(s) = -d(s, s_{goal}) + r_{bonus}(s)$$

밀집 보상은 지속적인 안내를 제공하지만 올바르게 설계하기가 더 어렵다.

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

### 3.3 커리큘럼 학습(Curriculum Learning)

어려운 작업의 경우, 처음부터 최종 보상으로 훈련하는 것이 너무 어려울 수 있다. **커리큘럼 학습**은 난이도를 점진적으로 높여 간다:

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

## 4. 로보틱스를 위한 안전한 RL(Safe RL)

### 4.1 안전이 중요한 이유

RL의 탐색 메커니즘(새로운 행동을 발견하기 위한 무작위 행동)은 실제 로봇에서 위험하다:

- 무작위 관절 토크가 로봇 팔을 테이블에 충돌시킬 수 있다
- 공격적인 보행 정책이 로봇을 전복시킬 수 있다
- 인간 근처에서의 탐색적 행동이 부상을 초래할 수 있다

로보틱스에서 안전은 선택 사항이 아니다 — 하드 제약(hard constraint)이다.

### 4.2 제약 RL(Constrained RL)

**제약 마르코프 결정 과정(Constrained Markov Decision Process, CMDP)**은 표준 RL 목표에 안전 제약을 추가한다:

$$\max_\pi \mathbb{E}_\pi \left[ \sum_t \gamma^t r_t \right] \quad \text{subject to} \quad \mathbb{E}_\pi \left[ \sum_t \gamma^t c_t^{(i)} \right] \leq d_i \quad \forall i$$

여기서 $c_t^{(i)}$는 제약 $i$에 대한 비용(예: 관절 토크 한계, 충돌 패널티)이고 $d_i$는 제약 예산이다.

**라그랑지안 완화(Lagrangian relaxation)**는 CMDP를 비제약 문제로 변환한다:

$$\mathcal{L}(\pi, \lambda) = \mathbb{E}_\pi\left[\sum_t r_t\right] - \sum_i \lambda_i \left(\mathbb{E}_\pi\left[\sum_t c_t^{(i)}\right] - d_i\right)$$

이중 변수(dual variable) $\lambda_i$는 제약을 강제하기 위해 업데이트된다.

### 4.3 안전한 탐색 전략

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

### 4.4 시뮬레이션-실제 안전 파이프라인

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

## 5. 조작을 위한 RL(RL for Manipulation)

### 5.1 파지 정책(Grasping Policies)

로봇 파지(grasping)는 로보틱스에서 RL의 가장 성공적인 응용 사례 중 하나다:

**관찰 공간(observation space)**: RGB-D 이미지 (또는 포인트 클라우드), 그리퍼 위치/방향, 관절 각도
**행동 공간(action space)**: 그리퍼 포즈 변화량(delta) ($\Delta x, \Delta y, \Delta z, \Delta\theta$) + 열기/닫기
**보상(reward)**: 파지 성공 여부 (물체를 들어 올린 후 그리퍼 안에 남아있는가?)

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

### 5.2 조립 정책(Assembly Policies)

조립 작업(핀-구멍 삽입, 커넥터 삽입)을 위한 RL은 일반적으로 다음을 사용한다:

- **힘/토크 관찰(force/torque observations)**: 접촉력이 정렬에 관한 중요한 정보를 제공한다
- **임피던스/컴플라이언스 행동(impedance/compliance actions)**: RL 에이전트가 위치를 직접 명령하는 대신 임피던스 매개변수(강성, 감쇠)를 명령한다
- **잔차 RL(residual RL)**: 고전적 컨트롤러로 시작하고 그 행동을 보정하는 잔차 정책을 학습한다

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

## 6. 보행을 위한 RL(RL for Locomotion)

### 6.1 다리 달린 로봇 보행(Legged Robot Locomotion)

RL은 다리 달린 보행에서 놀라운 결과를 달성했다:

- **MIT Mini Cheetah**: RL 훈련된 정책이 3.7 m/s로 달림
- **ANYmal**: RL 정책이 계단, 경사면, 험한 지형을 통과
- **Atlas** (Boston Dynamics): 파쿠르(parkour)와 곡예 동작에 RL 사용

사족 보행 로봇의 **관찰 공간**:

| 구성 요소 | 차원 | 설명 |
|-----------|------|------|
| 본체 방향(base orientation) | 3 | 롤(roll), 피치(pitch), 요(yaw) |
| 본체 각속도(base angular velocity) | 3 | IMU로부터 |
| 관절 위치(joint positions) | 12 | 다리 4개 × 관절 3개 |
| 관절 속도(joint velocities) | 12 | 엔코더로부터 |
| 이전 행동(previous actions) | 12 | 부드러움을 위해 |
| 명령 속도(command velocity) | 3 | 원하는 $v_x, v_y, \omega_z$ |

**행동 공간**: 목표 관절 위치 (12 DOF), 각 관절에서 저수준 PD 컨트롤러가 실행.

### 6.2 훈련 파이프라인

```python
class LocomotionTrainer:
    """Conceptual training pipeline for legged robot locomotion.

    Why PPO for locomotion? PPO (Proximal Policy Optimization) is
    the de facto standard for robot locomotion RL because:
    1. Stable training (clipped surrogate objective prevents large updates)
    2. Works with continuous action spaces
    3. Parallelizes well (thousands of environments simultaneously)
    4. Handles the high-dimensional observation/action spaces of legged robots

    PPO의 클리핑된 대리 목적 함수(clipped surrogate objective):
      L_CLIP = E_t[min(r_t(theta) * A_hat_t,
                       clip(r_t(theta), 1-eps, 1+eps) * A_hat_t)]
    각 기호의 의미:
      r_t(theta) = pi_new(a_t|s_t) / pi_old(a_t|s_t) — 새 정책과 이전 정책
        사이의 확률비(probability ratio); 정책이 얼마나 변했는지 측정
      A_hat_t — 이점 추정값(advantage estimate, 행동 a_t가 평균보다 얼마나
        좋았는지, 일반적으로 GAE(Generalized Advantage Estimation)로 계산)
      eps — 클리핑 범위(일반적으로 0.2), r_t가 1.0에서 너무 벗어나지 않게
        하여 작고 안정적인 정책 업데이트를 보장
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

## 7. 시뮬레이션 환경(Simulation Environments)

### 7.1 MuJoCo

**MuJoCo** (Multi-Joint dynamics with Contact)는 로봇 RL에 가장 널리 사용되는 물리 시뮬레이터다:

- 빠르고 정확한 접촉 시뮬레이션
- 미분 가능(differentiable) (MuJoCo XLA)
- 소프트 접촉(soft contacts), 힘줄(tendons), 복잡한 메커니즘 지원
- 표준 벤치마크 작업 (Ant, Humanoid, Hand 등)

### 7.2 Isaac Gym / Isaac Sim

**NVIDIA Isaac Gym**은 GPU에서 대규모 병렬 시뮬레이션을 가능하게 한다:

- 단일 GPU에서 4096개 이상의 환경 동시 실행
- GPU 가속 물리 및 렌더링
- CPU 기반 시뮬레이터보다 10-100배 빠름
- 엔드-투-엔드 GPU 훈련을 위한 PyTorch와의 긴밀한 통합

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

**Gymnasium** (구 OpenAI Gym)은 RL 환경의 표준 API다:

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

### 7.4 시뮬레이터 비교

| 기능 | MuJoCo | Isaac Gym | PyBullet | Gazebo |
|------|--------|-----------|----------|--------|
| 속도 | 빠름 (CPU) | 매우 빠름 (GPU) | 보통 (CPU) | 느림 |
| 병렬성 | 멀티프로세스 | GPU 네이티브 (4096+) | 멀티프로세스 | 제한적 |
| 접촉 정확도 | 우수 | 좋음 | 좋음 | 보통 |
| 렌더링 | 내장 | GPU 가속 | OpenGL | Gazebo/Ogre |
| ROS2 통합 | 제한적 | Isaac Sim 경유 | ROS 브리지 경유 | 네이티브 |
| 라이센스 | 무료 (Apache 2.0) | 무료 (NVIDIA) | 무료 (BSD) | 무료 (Apache 2.0) |
| 최적 용도 | 연구, 벤치마크 | 대규모 훈련 | 빠른 프로토타이핑 | 풀 로봇 시스템 |

---

## 8. 실용적 고려 사항

### 8.1 관찰 설계(Observation Design)

에이전트가 관찰하는 것이 학습 성능에 극적인 영향을 미친다:

| 관찰 선택 | 장점 | 단점 |
|-----------|------|------|
| 관절 각도 + 속도 | 저차원, 학습하기 쉬움 | 환경 인식 없음 |
| 원시 이미지 (픽셀) | 풍부한 정보 | 고차원, 학습 느림 |
| 포인트 클라우드(point clouds) | 3D 기하학 | 가변 크기, 특수 아키텍처 필요 |
| 특권 정보(privileged info, 시뮬레이션) + 증류(distilled) | 빠른 훈련 | 2단계 훈련 필요 |

**관찰 정규화(observation normalization)**는 훈련 안정성에 매우 중요하다. 원시 관찰(raw observations)은 종종 극히 다른 스케일을 가진다 (관절 각도는 라디안, 속도는 rad/s, 힘은 뉴턴). 이동 통계량으로 정규화하면 `obs = (obs - obs_mean) / (obs_std + 1e-8)` 입력을 중앙화하고 단위 분산으로 스케일링하여 신경망이 효율적으로 학습하도록 돕는다. 작은 상수 `1e-8`은 특징의 분산이 0일 때(예: 초기 훈련에서 일정한 센서 판독값) 0으로 나누는 것을 방지한다.

**비대칭 액터-크리틱(asymmetric actor-critic)**: 훈련 중 크리틱(critic)에는 특권 정보(정확한 물체 자세, 접촉력)를 사용하지만 액터(actor)에는 실제 센서로 관찰 가능한 특징만 사용한다. 이렇게 하면 배포 가능성을 유지하면서 학습 속도를 높인다.

### 8.2 행동 공간 설계(Action Space Design)

| 행동 유형 | 설명 | 안정성 | 정밀도 |
|-----------|------|--------|--------|
| 관절 토크(joint torques) | 직접 토크 명령 | 낮음 (위험) | 높음 (이론적) |
| 관절 위치 목표(joint position targets) | PD 컨트롤러가 목표 추적 | 높음 (PD가 안정화) | 보통 |
| 말단 효과기 변화량(end-effector deltas) | 데카르트 속도 명령 | 높음 | 작업 의존적 |
| 임피던스 매개변수(impedance parameters) | 강성/감쇠 설정점 | 매우 높음 | 작업 의존적 |

**최선의 관행**: PD 컨트롤러와 함께 관절 위치 목표를 사용한다. RL 에이전트가 원하는 관절 위치를 출력하고 저수준 PD 컨트롤러가 토크를 계산한다. 이는 직접 토크 제어보다 훨씬 안전하며 자연스러운 행동 부드러움을 제공한다.

**SAC(Soft Actor-Critic)에 대한 참고**: 연속 행동 공간(continuous-action) 로보틱스 작업을 위한 PPO의 대안으로, SAC는 엔트로피 정규화 목적 함수 $J(\pi) = \sum_t \mathbb{E}[r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))]$를 최대화한다. 여기서 $\alpha$는 탐색-활용 트레이드오프(exploration-exploitation trade-off)를 제어하는 **온도 매개변수(temperature parameter)**다. 물리적으로 $\alpha$는 에이전트가 무작위성을 얼마나 중시하는지 결정한다: 높은 $\alpha$는 넓은 탐색을 장려하고(초기 훈련이나 접촉이 많은 작업에 유용), 낮은 $\alpha$는 알려진 좋은 행동을 활용하는 데 집중한다. 현대 SAC 구현은 $\alpha$를 목표 엔트로피(target entropy)에 맞춰 자동 조정하여 수동 조절이 불필요하다.

---

## 요약

| 개념 | 핵심 아이디어 |
|------|--------------|
| 로보틱스를 위한 RL | 모델이 분석적으로 도출하기 너무 복잡할 때 상호작용으로부터 학습 |
| 시뮬레이션-실제 격차(sim-to-real gap) | 시뮬레이터와 현실 사이의 물리적 매개변수 불일치 |
| 도메인 무작위화(domain randomization) | 광범위한 매개변수 범위에서 훈련하여 현실에 강건한 정책 생성 |
| 시스템 식별(system identification) | 실제 매개변수 추정으로 시뮬레이터 캘리브레이션 |
| 보상 형성(reward shaping) | 다중 구성 요소 밀집 보상으로 학습 안내; 희소 보상은 불충분 |
| 커리큘럼 학습(curriculum learning) | 쉽게 시작하여 에이전트가 향상됨에 따라 난이도 증가 |
| 안전한 RL(safe RL) | 위험한 탐색 방지를 위한 제약 MDP + 안전 레이어 |
| 잔차 RL(residual RL) | 두 세계의 장점을 위한 고전적 컨트롤러에 대한 보정 학습 |
| GPU 시뮬레이션 | Isaac Gym이 빠른 훈련을 위한 4096개 이상의 병렬 환경 지원 |
| 행동 공간 설계 | 안전성과 안정성을 위한 PD 컨트롤러와 함께 관절 위치 목표 사용 |

---

## 연습 문제

1. **보상 함수 설계**: 로봇 팔 작업을 위한 보상 함수를 설계하라: 공을 집어 컵에 넣기. 최소 4개의 보상 구성 요소를 정의하라 (접근하기, 파지하기, 운반하기, 놓기). 보상 함수를 구현하고 각 구성 요소가 원하는 행동을 어떻게 안내하는지 설명하라. 무엇이 잘못될 수 있는가 (보상 해킹)?

2. **도메인 무작위화 연구**: Python으로 간단한 2D 도달 작업을 구현하라 (목표를 향해 도달하는 점 에이전트). 고정된 동역학으로 정책을 훈련한 다음, 동역학이 교란되었을 때 (다른 감쇠) 평가하라. 훈련 중 도메인 무작위화를 사용하여 반복하라. 전이 성능을 비교하라.

3. **안전 레이어 구현**: 다음을 강제하는 2-DOF 로봇 팔을 위한 안전 레이어를 구현하라: (a) 관절 위치 한계, (b) 관절 속도 한계, (c) 최대 말단 효과기 속도. 안전 레이어가 있는 RL 에이전트는 절대 제약을 위반하지 않는 반면 없는 에이전트는 위반함을 보여라.

4. **커리큘럼 설계**: 시뮬레이션 로봇이 문을 여는 것을 가르치기 위한 커리큘럼을 설계하라. 3-4개의 난이도 수준을 식별하라 (예: 문이 열려 있는 것 vs. 완전히 닫힌 것, 가벼운 vs. 무거운 문, 손잡이 있음/없음). 진행 과정과 각 수준이 에이전트를 다음 수준을 위해 어떻게 준비시키는지 설명하라.

5. **시뮬레이터 비교**: Gymnasium을 사용하여 두 환경에서 동일한 도달 작업을 생성하라 (예: CartPole과 커스텀 2D 환경). 두 환경에서 PPO를 훈련하고 비교하라: 훈련 시간, 최종 성능, 하이퍼파라미터에 대한 민감도. 이것이 RL 훈련 효율성에서 환경의 역할에 대해 무엇을 알려주는가?

---

## 추가 참고자료

- Tobin, J. et al. "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." *IROS*, 2017. (도메인 무작위화의 근본적인 논문)
- Peng, X. B. et al. "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization." *ICRA*, 2018. (보행을 위한 동역학 무작위화)
- OpenAI. "Solving Rubik's Cube with a Robot Hand." 2019. (획기적인 시뮬레이션-실제 전이 결과)
- Rudin, N. et al. "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning." *CoRL*, 2022. (보행을 위한 Isaac Gym)
- Garcıa, J. and Fernandez, F. "A Comprehensive Survey on Safe Reinforcement Learning." *JMLR*, 2015. (안전한 RL 서베이)
- Johannink, T. et al. "Residual Reinforcement Learning for Robot Manipulators." *ICRA*, 2019. (잔차 RL)

---

[← 이전: ROS2 내비게이션 스택](14_ROS2_Navigation.md) | [다음: 다중 로봇 시스템과 군집 →](16_Multi_Robot_Systems.md)
