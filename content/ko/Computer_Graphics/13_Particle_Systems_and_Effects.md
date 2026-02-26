# 13. 파티클 시스템과 이펙트

[← 이전: 12. 애니메이션과 골격 시스템](12_Animation_and_Skeletal_Systems.md) | [다음: 14. GPU 컴퓨팅 →](14_GPU_Computing.md)

---

## 학습 목표

1. 파티클 시스템(Particle System) 아키텍처 이해: 이미터(Emitter), 파티클 풀(Particle Pool), 업데이트(Update), 렌더(Render) 단계
2. 파티클 속성(Particle Properties)과 수명 동안의 변화 정의
3. 중력(Gravity), 바람(Wind), 항력(Drag), 점 인력기(Point Attractor) 등 일반적인 힘 구현
4. 파티클 시뮬레이션을 위한 수치 적분(Numerical Integration) 방법(오일러(Euler), 벨렛(Verlet)) 비교
5. 카메라를 향하는 파티클 쿼드(Quad)를 위한 빌보드 렌더링(Billboard Rendering) 설명
6. 트랜스폼 피드백(Transform Feedback)과 컴퓨트 셰이더(Compute Shader)를 이용한 GPU 파티클 시스템 이해
7. 체적 이펙트(Volumetric Effects)(연기, 불, 안개)와 레이 마칭(Ray Marching) 기초 설명
8. Python으로 완전한 파티클 시스템 시뮬레이션 구현

---

## 왜 중요한가

파티클 시스템은 게임과 영화의 시각 이펙트(Visual Effects)를 구현하는 핵심 도구입니다. 인터랙티브 미디어에서 보이는 모든 폭발, 연기 궤적, 빗줄기, 불꽃, 반짝임, 마법 주문, 분수는 파티클로 만들어집니다. 윌리엄 리브스(William Reeves)는 1983년 루카스필름(Lucasfilm)에서 *스타 트렉 II: 칸의 분노(Star Trek II: The Wrath of Khan)*의 "제네시스 이펙트(Genesis effect)"를 만들기 위해 파티클 시스템을 도입했으며, 이후 핵심 그래픽스 기법으로 자리잡았습니다.

파티클 시스템의 강점은 단순함에 있습니다. 각 파티클은 몇 가지 속성(위치, 속도, 나이)을 가진 점에 불과하지만, 수천 개가 모이면 복잡하고 유기적인 현상을 만들어냅니다. 파티클 시스템을 이해하면 물리학, 게임, 과학적 시각화 전반에 적용되는 시뮬레이션(Simulation), 수치 적분(Numerical Integration), GPU 기반 렌더링(GPU-driven Rendering)의 기초를 배울 수 있습니다.

---

## 1. 파티클 시스템 아키텍처

### 1.1 개요

파티클 시스템은 다음으로 구성됩니다:

```
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Emitter  │────▶│ Particle  │────▶│  Update   │────▶│  Render  │
│ (spawn)  │     │   Pool    │     │ (physics) │     │ (draw)   │
└─────────┘     └──────────┘     └──────────┘     └──────────┘
     │                                  │
     │           ┌──────────┐           │
     └──────────▶│  Forces  │◀──────────┘
                 └──────────┘
```

**이미터(Emitter)**: 초기 속성(위치, 속도, 색상, 크기)을 가진 새 파티클을 생성합니다. 방출 방식은 다음과 같습니다:
- **연속(Continuous)**: 초당 N개 파티클
- **버스트(Burst)**: 한 번에 모든 파티클 방출 (폭발)
- **공간적(Spatial)**: 점, 선, 원, 구, 메시(Mesh) 표면 등에서 방출

**파티클 풀(Particle pool)**: 고정 크기의 파티클 배열입니다. 죽은 파티클은 메모리 할당 오버헤드를 피하기 위해 재사용됩니다. 이는 고전적인 **오브젝트 풀(Object Pool)** 패턴입니다.

**업데이트(Update)**: 매 프레임마다 $\Delta t$ 만큼 시뮬레이션을 진행합니다: 힘을 적용하고, 속도/위치를 적분하며, 파티클의 나이를 증가시키고, 만료된 파티클을 제거합니다.

**렌더(Render)**: 살아있는 각 파티클을 텍스처 쿼드(Textured Quad), 포인트 스프라이트(Point Sprite), 또는 메시(Mesh)로 그립니다.

### 1.2 파티클 생명 주기

```
Born (emitter spawns)
  │
  ▼
Alive (position, velocity, age update each frame)
  │  age += dt
  │  apply forces
  │  integrate motion
  ▼
Dead (age >= lifetime)  →  returned to pool
```

---

## 2. 파티클 속성

각 파티클은 수명 동안 변화하는 속성들을 가집니다:

| 속성 | 타입 | 설명 |
|------|------|------|
| `position` | vec3 | 현재 월드 위치 |
| `velocity` | vec3 | 현재 속도 |
| `acceleration` | vec3 | 누적 힘 / 질량 |
| `age` | float | 생성 후 경과 시간 (초) |
| `lifetime` | float | 소멸 전 최대 나이 |
| `color` | vec4 | RGBA 색상 (알파는 페이드아웃용) |
| `size` | float | 파티클 크기 (반지름 또는 쿼드 너비) |
| `rotation` | float | 회전 각도 (텍스처 쿼드용) |
| `mass` | float | 힘 계산에 사용 |

### 2.1 속성 커브

속성들은 파티클 수명 동안 **커브(Curve)**를 사용해 변화하는 경우가 많습니다:

- **알파(Alpha)**: 생성 시 페이드인, 수명 중 완전 불투명, 소멸 전 페이드아웃
- **크기(Size)**: 작게 시작, 성장, 그 후 축소
- **색상(Color)**: 흰색 → 노란색 → 주황색 → 빨간색 → 검정색으로 전환 (불꽃)

이 커브들은 정규화된 나이 $t = \text{age} / \text{lifetime} \in [0, 1]$로 매개변수화됩니다.

```python
def fade_in_out(t, fade_in=0.1, fade_out=0.3):
    """
    Alpha curve: quick fade-in, sustained, gradual fade-out.
    Why smooth fading: prevents harsh pop-in/pop-out artifacts.
    """
    if t < fade_in:
        return t / fade_in
    elif t > 1.0 - fade_out:
        return (1.0 - t) / fade_out
    else:
        return 1.0
```

---

## 3. 힘

### 3.1 중력 (Gravity)

가장 단순하고 흔한 힘:

$$\mathbf{F}_{\text{gravity}} = m \mathbf{g}$$

여기서 $\mathbf{g} = (0, -9.81, 0)$ m/s$^2$는 지구 중력입니다. 양식화된 이펙트의 경우 $\mathbf{g}$는 임의의 벡터와 크기를 가질 수 있습니다.

### 3.2 바람 (Wind)

일정하거나 공간적으로 변하는 힘:

$$\mathbf{F}_{\text{wind}} = \mathbf{w}(t)$$

펄린(Perlin) 또는 심플렉스(Simplex) 노이즈를 사용해 난류(Turbulent) 바람을 시뮬레이션할 수 있습니다:

$$\mathbf{F}_{\text{turbulence}}(\mathbf{p}, t) = A \cdot \text{noise}(\mathbf{p} \cdot s + t \cdot f)$$

여기서 $A$는 진폭(Amplitude), $s$는 공간 주파수(Spatial Frequency), $f$는 시간 주파수(Temporal Frequency)입니다.

### 3.3 항력 (Drag, 공기 저항)

항력은 운동에 반대되며 속도에 비례합니다:

$$\mathbf{F}_{\text{drag}} = -c_d \cdot \mathbf{v}$$

선형 항력의 경우, 또는:

$$\mathbf{F}_{\text{drag}} = -c_d \cdot \|\mathbf{v}\| \cdot \mathbf{v}$$

이차 항력(고속에서 물리적으로 더 정확)의 경우.

**항력 계수(Drag Coefficient)** $c_d$는 파티클이 얼마나 빨리 감속하는지를 결정합니다:
- 낮은 항력: 파티클이 멀리 이동 (스파크, 총알)
- 높은 항력: 파티클이 빨리 감속 (연기, 먼지)

### 3.4 점 인력기와 반발기 (Point Attractors and Repellers)

위치 $\mathbf{a}$의 점 인력기(Point Attractor)는 파티클을 그쪽으로 끌어당깁니다:

$$\mathbf{F}_{\text{attract}} = k \cdot \frac{\mathbf{a} - \mathbf{p}}{\|\mathbf{a} - \mathbf{p}\|^2 + \epsilon}$$

$\epsilon$은 인력기 위치에서의 특이점(Singularity)을 방지합니다. 음수 $k$는 반발기(Repeller)를 만듭니다.

### 3.5 소용돌이 힘 (Vortex Force)

소용돌이 힘(Vortex Force)은 축 $\hat{\mathbf{a}}$ 주위에 회오리 운동을 만듭니다:

$$\mathbf{F}_{\text{vortex}} = k \cdot (\hat{\mathbf{a}} \times (\mathbf{p} - \mathbf{c}))$$

여기서 $\mathbf{c}$는 소용돌이 중심입니다. 토네이도, 소용돌이, 나선형 이펙트에 유용합니다.

---

## 4. 수치 적분

### 4.1 오일러 법 (Euler Method)

가장 단순한 적분기입니다. 위치 $\mathbf{x}$, 속도 $\mathbf{v}$, 가속도 $\mathbf{a}$가 주어졌을 때:

$$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \mathbf{a}(t) \cdot \Delta t$$
$$\mathbf{x}(t + \Delta t) = \mathbf{x}(t) + \mathbf{v}(t) \cdot \Delta t$$

**심플렉틱 오일러(Symplectic Euler)**(반암묵적(Semi-implicit) 오일러라고도 함)는 먼저 속도를 갱신한 후 *새로운* 속도로 위치를 계산합니다:

$$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \mathbf{a}(t) \cdot \Delta t$$
$$\mathbf{x}(t + \Delta t) = \mathbf{x}(t) + \mathbf{v}(t + \Delta t) \cdot \Delta t$$

이 작은 변화가 에너지 보존(Energy Conservation)을 극적으로 개선하며, 게임 물리 엔진의 표준입니다.

### 4.2 벨렛 적분 (Verlet Integration)

**스토르머-벨렛(Stormer-Verlet)**은 명시적인 속도 없이 현재와 이전 시간 단계의 위치를 사용합니다:

$$\mathbf{x}(t + \Delta t) = 2\mathbf{x}(t) - \mathbf{x}(t - \Delta t) + \mathbf{a}(t) \cdot \Delta t^2$$

속도는 암묵적입니다: $\mathbf{v}(t) \approx \frac{\mathbf{x}(t) - \mathbf{x}(t - \Delta t)}{\Delta t}$.

**장점**: 2차 정확도, 심플렉틱(에너지 보존), 제약 조건 만족(천, 로프)에 탁월합니다.

**단점**: 감쇠(Damping) 또는 속도 의존 힘 추가가 덜 자연스럽습니다.

### 4.3 비교

| 방법 | 차수 | 에너지 보존 | 복잡도 | 최적 용도 |
|------|------|------------|--------|----------|
| 오일러(Euler) | 1차 | 나쁨 (에너지 발산) | 매우 낮음 | 빠른 프로토타이핑 |
| 심플렉틱 오일러(Symplectic Euler) | 1차 | 좋음 | 매우 낮음 | 게임, 파티클 |
| 벨렛(Verlet) | 2차 | 탁월 | 낮음 | 제약 조건, 천 |
| RK4 | 4차 | 좋음 | 중간 | 고정밀 필요 시 |

파티클 시스템에는 **심플렉틱 오일러(Symplectic Euler)**가 거의 항상 충분합니다.

---

## 5. 빌보드 렌더링 (Billboard Rendering)

### 5.1 문제

파티클은 점이지만 텍스처가 있는 형태(원, 연기 덩어리, 스파크)로 표시하고 싶습니다. 각 파티클을 항상 카메라를 향하는 **쿼드(Quad)**(두 개의 삼각형)로 렌더링합니다.

### 5.2 카메라를 향하는 쿼드

위치 $\mathbf{p}$, 크기 $s$인 각 파티클에 대해 카메라의 오른쪽 벡터 $\mathbf{r}$과 위 벡터 $\mathbf{u}$로 쿼드를 구성합니다:

$$\mathbf{v}_0 = \mathbf{p} + s(-\mathbf{r} + \mathbf{u}), \quad \mathbf{v}_1 = \mathbf{p} + s(\mathbf{r} + \mathbf{u})$$
$$\mathbf{v}_2 = \mathbf{p} + s(\mathbf{r} - \mathbf{u}), \quad \mathbf{v}_3 = \mathbf{p} + s(-\mathbf{r} - \mathbf{u})$$

이 네 꼭짓점은 뷰 방향에 관계없이 항상 카메라를 향하는 화면 정렬 쿼드(Screen-aligned Quad)를 형성합니다.

### 5.3 렌더링 고려사항

- **블렌딩(Blending)**: 파티클은 일반적으로 **가산 블렌딩(Additive Blending)**(불, 스파크, 마법) 또는 **알파 블렌딩(Alpha Blending)**(연기, 먼지)을 사용합니다. 가산: $C_{\text{out}} = C_{\text{src}} + C_{\text{dst}}$. 알파: $C_{\text{out}} = \alpha C_{\text{src}} + (1-\alpha)C_{\text{dst}}$.
- **깊이 테스트(Depth Testing)**: z-버퍼 쓰기는 비활성화(파티클이 겹침), z-테스트는 활성화(파티클이 솔리드 지오메트리에 가려짐).
- **정렬(Sorting)**: 알파 블렌딩의 경우 파티클을 뒤에서 앞으로 정렬합니다. 가산 블렌딩은 가환적(Commutative)이므로 정렬이 불필요합니다.
- **소프트 파티클(Soft Particles)**: 파티클 깊이를 깊이 버퍼와 비교해 솔리드 지오메트리와 교차하는 파티클을 페이드 처리합니다. 경계선이 딱딱하게 보이는 현상을 방지합니다.

### 5.4 포인트 스프라이트 (Point Sprites)

현대 GPU는 **포인트 스프라이트(Point Sprites)**를 지원합니다: 단일 꼭짓점을 자동으로 생성된 화면 정렬 쿼드로 렌더링합니다. OpenGL에서: `glEnable(GL_PROGRAM_POINT_SIZE)`로 활성화하고 버텍스 셰이더에서 `gl_PointSize`를 설정합니다. CPU에서 쿼드 지오메트리를 생성하는 것보다 효율적입니다.

---

## 6. Python 구현: 파티클 시스템

```python
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional

@dataclass
class Particle:
    """A single particle with physics properties."""
    position: np.ndarray      # (3,) world position
    velocity: np.ndarray      # (3,) velocity
    color: np.ndarray         # (4,) RGBA
    size: float               # Radius or quad half-size
    age: float = 0.0          # Time since birth
    lifetime: float = 2.0     # Maximum age
    mass: float = 1.0
    alive: bool = True

    @property
    def normalized_age(self):
        """Age as fraction of lifetime [0, 1]."""
        return min(self.age / self.lifetime, 1.0)


class Emitter:
    """Spawns particles with randomized initial properties."""

    def __init__(self, position, rate=50.0, lifetime_range=(1.0, 3.0),
                 speed_range=(1.0, 3.0), size_range=(0.05, 0.15),
                 color_start=np.array([1, 0.8, 0.2, 1]),
                 color_end=np.array([1, 0, 0, 0]),
                 spread_angle=30.0, direction=np.array([0, 1, 0])):
        self.position = np.array(position, dtype=float)
        self.rate = rate               # Particles per second
        self.lifetime_range = lifetime_range
        self.speed_range = speed_range
        self.size_range = size_range
        self.color_start = color_start
        self.color_end = color_end
        self.spread_angle = spread_angle  # Cone half-angle in degrees
        self.direction = direction / np.linalg.norm(direction)
        self._accumulator = 0.0        # Fractional particle accumulator

    def emit(self, dt) -> List[Particle]:
        """Generate new particles for this time step."""
        # Why accumulator: if rate*dt < 1, we still need to emit
        # particles over multiple frames
        self._accumulator += self.rate * dt
        count = int(self._accumulator)
        self._accumulator -= count

        particles = []
        for _ in range(count):
            # Random direction within cone
            vel_dir = self._random_cone_direction()
            speed = np.random.uniform(*self.speed_range)

            lifetime = np.random.uniform(*self.lifetime_range)
            size = np.random.uniform(*self.size_range)

            p = Particle(
                position=self.position.copy(),
                velocity=vel_dir * speed,
                color=self.color_start.copy(),
                size=size,
                lifetime=lifetime,
            )
            particles.append(p)

        return particles

    def _random_cone_direction(self):
        """Generate a random direction within a cone around self.direction."""
        # Build local frame
        d = self.direction
        if abs(d[0]) < 0.9:
            tangent = np.cross(np.array([1, 0, 0]), d)
        else:
            tangent = np.cross(np.array([0, 1, 0]), d)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(d, tangent)

        # Random angle within cone
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(
            np.cos(np.radians(self.spread_angle)), 1.0
        )
        sin_theta = np.sqrt(1 - cos_theta ** 2)

        # Direction in local frame, then transform to world
        local = np.array([sin_theta * np.cos(phi),
                          sin_theta * np.sin(phi),
                          cos_theta])
        world = local[0] * tangent + local[1] * bitangent + local[2] * d
        return world / np.linalg.norm(world)


class ForceField:
    """Base class for forces applied to particles."""

    def apply(self, particle: Particle, dt: float) -> np.ndarray:
        """Return force vector (3,)."""
        raise NotImplementedError


class Gravity(ForceField):
    def __init__(self, g=np.array([0, -9.81, 0])):
        self.g = np.array(g, dtype=float)

    def apply(self, particle, dt):
        return particle.mass * self.g


class Wind(ForceField):
    def __init__(self, direction=np.array([1, 0, 0]), strength=2.0):
        self.force = np.array(direction, dtype=float) * strength

    def apply(self, particle, dt):
        return self.force


class Drag(ForceField):
    def __init__(self, coefficient=0.5):
        self.cd = coefficient

    def apply(self, particle, dt):
        # Why linear drag: simple and effective for visual particle systems
        return -self.cd * particle.velocity


class PointAttractor(ForceField):
    def __init__(self, position, strength=10.0, epsilon=0.1):
        self.pos = np.array(position, dtype=float)
        self.strength = strength
        self.epsilon = epsilon

    def apply(self, particle, dt):
        diff = self.pos - particle.position
        dist_sq = np.dot(diff, diff) + self.epsilon
        return self.strength * diff / dist_sq


class ParticleSystem:
    """
    Complete particle system: manages emitters, forces, and particle pool.
    Uses symplectic Euler integration.
    """

    def __init__(self, max_particles=5000):
        self.max_particles = max_particles
        self.particles: List[Particle] = []
        self.emitters: List[Emitter] = []
        self.forces: List[ForceField] = []
        self.time = 0.0

    def add_emitter(self, emitter: Emitter):
        self.emitters.append(emitter)

    def add_force(self, force: ForceField):
        self.forces.append(force)

    def update(self, dt: float):
        """Advance the simulation by dt seconds."""
        self.time += dt

        # Emit new particles
        for emitter in self.emitters:
            new_particles = emitter.emit(dt)
            # Respect pool limit
            available = self.max_particles - len(self.particles)
            self.particles.extend(new_particles[:available])

        # Update existing particles
        alive_particles = []
        for p in self.particles:
            # Age the particle
            p.age += dt

            # Kill expired particles
            if p.age >= p.lifetime:
                p.alive = False
                continue

            # Accumulate forces
            total_force = np.zeros(3)
            for force in self.forces:
                total_force += force.apply(p, dt)

            # Symplectic Euler integration
            # Why symplectic: better energy conservation than standard Euler
            acceleration = total_force / p.mass
            p.velocity += acceleration * dt       # Update velocity first
            p.position += p.velocity * dt         # Then use new velocity

            # Update visual properties based on normalized age
            t = p.normalized_age

            # Color interpolation (start -> end over lifetime)
            # Why we store both: allows different start/end per emitter
            emitter = self.emitters[0] if self.emitters else None
            if emitter:
                p.color = (1 - t) * emitter.color_start + t * emitter.color_end

            # Size: grow then shrink
            p.size *= (1.0 - 0.3 * dt)  # Gradual shrink

            alive_particles.append(p)

        self.particles = alive_particles

    def get_positions(self) -> np.ndarray:
        """Return all particle positions as (N, 3) array."""
        if not self.particles:
            return np.zeros((0, 3))
        return np.array([p.position for p in self.particles])

    def get_colors(self) -> np.ndarray:
        """Return all particle colors as (N, 4) array."""
        if not self.particles:
            return np.zeros((0, 4))
        return np.array([p.color for p in self.particles])

    def stats(self) -> dict:
        return {
            "alive": len(self.particles),
            "max": self.max_particles,
            "time": self.time,
        }


# --- Demo: Fire-like particle system ---

system = ParticleSystem(max_particles=2000)

# Emitter: fire shooting upward
fire_emitter = Emitter(
    position=[0, 0, 0],
    rate=200,
    lifetime_range=(0.5, 2.0),
    speed_range=(1.0, 4.0),
    size_range=(0.05, 0.2),
    color_start=np.array([1.0, 0.9, 0.3, 1.0]),   # Bright yellow
    color_end=np.array([0.8, 0.1, 0.0, 0.0]),      # Dark red, transparent
    spread_angle=20.0,
    direction=np.array([0, 1, 0]),
)
system.add_emitter(fire_emitter)

# Forces
system.add_force(Gravity(np.array([0, -2.0, 0])))  # Weak gravity (fire rises)
system.add_force(Wind(np.array([0.5, 0, 0]), strength=0.5))
system.add_force(Drag(coefficient=0.8))

# Simulate
dt = 1.0 / 60.0  # 60 FPS
print("Simulating fire particle system...")
for frame in range(180):  # 3 seconds
    system.update(dt)
    if frame % 30 == 0:
        s = system.stats()
        positions = system.get_positions()
        if len(positions) > 0:
            avg_y = np.mean(positions[:, 1])
            max_y = np.max(positions[:, 1])
            print(f"  Frame {frame:3d}: {s['alive']:4d} particles, "
                  f"avg_y={avg_y:.2f}, max_y={max_y:.2f}")

# Visualization (if matplotlib available)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    positions = system.get_positions()
    colors = system.get_colors()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 2], positions[:, 1],
               c=colors[:, :3], s=colors[:, 3] * 20, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title(f'Fire Particle System ({len(positions)} particles)')
    plt.tight_layout()
    plt.savefig('particle_fire.png', dpi=150)
    plt.close()
    print("Saved particle_fire.png")
except ImportError:
    print("Install matplotlib for visualization")
```

---

## 7. GPU 파티클 시스템

### 7.1 동기

CPU 파티클 시스템은 수만 개의 파티클로 제한됩니다. GPU 파티클 시스템은 시뮬레이션 전체를 GPU에서 실행해 **수백만 개**를 처리할 수 있습니다.

### 7.2 트랜스폼 피드백 방식 (OpenGL)

1. 버텍스 버퍼 오브젝트(VBO, Vertex Buffer Object)에 파티클 데이터 저장
2. 버텍스 셰이더에서 힘 적용과 적분 수행
3. **트랜스폼 피드백(Transform Feedback)**을 사용해 갱신된 위치/속도를 두 번째 VBO에 기록
4. 매 프레임마다 두 VBO를 교환(핑퐁, Ping-pong)
5. 출력 VBO를 포인트 스프라이트로 렌더링

```glsl
// Vertex shader for particle update (transform feedback)
#version 330

in vec3 in_position;
in vec3 in_velocity;
in float in_age;
in float in_lifetime;

out vec3 out_position;
out vec3 out_velocity;
out float out_age;
out float out_lifetime;

uniform float dt;
uniform vec3 gravity;

void main() {
    out_age = in_age + dt;
    out_lifetime = in_lifetime;

    if (out_age >= out_lifetime) {
        // Respawn logic would go here (or use a compute shader)
        out_position = vec3(0.0);
        out_velocity = vec3(0.0);
        out_age = 0.0;
    } else {
        vec3 accel = gravity;
        out_velocity = in_velocity + accel * dt;
        out_position = in_position + out_velocity * dt;
    }
}
```

### 7.3 컴퓨트 셰이더 방식

현대 GPU는 임의의 버퍼를 읽고 쓸 수 있는 **컴퓨트 셰이더(Compute Shader)**를 지원합니다:

1. **셰이더 스토리지 버퍼 오브젝트(SSBO, Shader Storage Buffer Object)**에 파티클 데이터 저장
2. 파티클당 하나의 스레드로 컴퓨트 셰이더 디스패치
3. 각 스레드가 독립적으로 파티클을 갱신
4. 핑퐁 불필요 -- SSBO 내 원자적 연산(Atomic Operations) 또는 이중 버퍼링

```glsl
// Compute shader for particle simulation
#version 430

layout(local_size_x = 256) in;

struct Particle {
    vec4 position;    // xyz = pos, w = size
    vec4 velocity;    // xyz = vel, w = age
    vec4 color;
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

uniform float dt;
uniform vec3 gravity;
uniform int numParticles;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= numParticles) return;

    Particle p = particles[idx];
    float age = p.velocity.w + dt;

    if (age >= p.position.w) {  // position.w stores lifetime
        // Dead particle -- reset or skip
        age = 0.0;
        // ... respawn logic ...
    }

    vec3 vel = p.velocity.xyz + gravity * dt;
    vec3 pos = p.position.xyz + vel * dt;

    p.position.xyz = pos;
    p.velocity.xyz = vel;
    p.velocity.w = age;

    particles[idx] = p;
}
```

### 7.4 성능 비교

| 방식 | 파티클 수 | 병목 |
|------|----------|------|
| CPU (Python) | ~1,000-5,000 | Python 오버헤드 |
| CPU (C++) | ~50,000-100,000 | 캐시 미스, 단일 스레드 |
| GPU (트랜스폼 피드백) | ~500,000 | VBO 핑퐁 오버헤드 |
| GPU (컴퓨트 셰이더) | ~1,000,000-10,000,000 | 메모리 대역폭 |

---

## 8. 체적 이펙트 (Volumetric Effects)

### 8.1 연기와 안개

연기, 안개, 구름은 개별 파티클로 잘 표현되지 않습니다(공간을 채우기 때문입니다). 두 가지 접근 방식이 있습니다:

**파티클 기반**: 부드러운 텍스처를 가진 크고 반투명한 파티클 다수를 겹쳐서 사용합니다. 먼 거리의 연기 기둥에는 효과적이지만 가까이서 보면 "덩어리져(Blobby)" 보입니다.

**볼륨 렌더링(Volume Rendering) (레이 마칭)**: 3D 볼륨(밀도 필드)을 통과하는 레이를 투사하고 레이를 따라 색상과 불투명도를 누적합니다:

$$C_{\text{out}} = \sum_{k=0}^{N} T_k \cdot \sigma(\mathbf{x}_k) \cdot c(\mathbf{x}_k) \cdot \Delta s$$

여기서 $T_k = \exp\left(-\sum_{j=0}^{k-1} \sigma(\mathbf{x}_j) \Delta s\right)$는 투과율(Transmittance), $\sigma$는 밀도(소광 계수, Extinction Coefficient), $c$는 색상/방출(Emission)입니다.

### 8.2 레이 마칭 (Ray Marching)

**레이 마칭(Ray Marching)**은 일정한 간격으로 레이를 따라 이동하며 각 단계에서 밀도 필드를 샘플링합니다:

```
function ray_march(ray_origin, ray_direction, max_steps, step_size):
    color = (0, 0, 0)
    transmittance = 1.0

    for i in range(max_steps):
        pos = ray_origin + i * step_size * ray_direction
        density = sample_density_field(pos)

        if density > 0:
            // Beer-Lambert extinction
            extinction = exp(-density * step_size)
            // Light contribution at this point
            light = compute_lighting(pos, density)
            // Accumulate
            color += transmittance * (1 - extinction) * light
            transmittance *= extinction

        if transmittance < 0.01:
            break  // Early termination: fully opaque

    return color
```

### 8.3 불꽃 렌더링 (Fire Rendering)

불꽃은 파티클 기반과 체적 기반 접근 방식을 결합합니다:

1. **온도 필드(Temperature Field)**: 연소를 밀도 + 온도 필드로 시뮬레이션
2. **색상 매핑(Color Mapping)**: 온도를 방출 색상(흑체 복사, Black Body Radiation)에 매핑:
   - 낮은 온도: 어두운 빨강/주황
   - 높은 온도: 노랑/흰색
   - 매우 높은 온도: 파랑/흰색
3. **노이즈(Noise)**: 밀도 필드에 절차적 노이즈(펄린(Perlin), 심플렉스(Simplex)) 추가로 난류(Turbulent) 외형 구현
4. **위쪽 이동**: 밀도 필드를 위쪽으로 이동(부력, Buoyancy)

### 8.4 실용적인 불꽃 공식

정규화된 온도 $T$ (0~1)에서 간단한 불꽃 색상:

$$R = \min(1, 1.5T), \quad G = \min(1, 1.5T^2), \quad B = T^4$$

이는 낮은 온도를 빨강, 중간을 노랑, 높은 온도를 흰색으로 매핑합니다.

---

## 9. 고급 파티클 기법

### 9.1 충돌 감지 (Collision Detection)

파티클은 씬(Scene) 지오메트리와 충돌할 수 있습니다:
- **평면 충돌(Plane Collision)**: 파티클이 평면을 통과하면 법선 방향의 속도 성분을 반사합니다: $v_n' = -e \cdot v_n$, 여기서 $e$는 반발 계수(Coefficient of Restitution)
- **구 충돌(Sphere Collision)**: 구의 표면 법선을 사용해 유사하게 처리
- **깊이 버퍼 충돌(Depth Buffer Collision)**: 파티클 깊이를 씬 깊이 버퍼와 비교 (GPU 친화적)

### 9.2 궤적 렌더링 (Trail Rendering)

파티클은 과거 위치를 기록해 궤적을 남길 수 있습니다:
- **선 궤적(Line Trails)**: 현재와 이전 위치를 연결 (스파크, 불꽃놀이)
- **리본 궤적(Ribbon Trails)**: 방출 순서대로 파티클을 텍스처 스트립으로 연결 (검 베기, 미사일 궤적)

### 9.3 하위 이미터 (Sub-Emitters)

파티클이 죽을 때 **하위 이미터(Sub-Emitter)** 버스트를 생성할 수 있습니다:
- 불꽃놀이: 메인 파티클이 상승하고 사멸 시 하위 파티클로 폭발
- 스파크 연쇄: 충돌 스파크가 더 작은 스파크를 생성
- 폭발 후 연기: 불 파티클이 연기 하위 파티클을 방출

### 9.4 파티클 LOD (Level of Detail)

원거리 파티클 시스템은 단순화할 수 있습니다:
- 카메라와의 거리에 따라 방출률 감소
- 파티클 크기 증가, 수 감소
- 매우 먼 이펙트는 단일 빌보드 텍스처로 전환

---

## 10. 이펙트 레시피

| 이펙트 | 파티클 수 | 수명 | 힘 | 블렌딩 | 비고 |
|--------|----------|------|-----|--------|------|
| 불꽃(Fire) | 200-1000 | 0.5-2초 | 약한 상향 중력, 항력, 난류 | 가산 | 색상: 노랑→빨강→검정 |
| 연기(Smoke) | 100-500 | 2-5초 | 부력, 항력, 바람 | 알파 | 크고, 느리고, 낮은 알파 |
| 스파크(Sparks) | 50-200 | 0.3-1초 | 강한 중력, 항력 | 가산 | 작고, 밝고, 빠름 |
| 비(Rain) | 500-2000 | 1-3초 | 강한 중력 | 알파 | 수직 줄기 |
| 눈(Snow) | 200-1000 | 3-8초 | 약한 중력, 바람 | 알파 | 느리고, 흩날림 |
| 폭발(Explosion) | 500 버스트 | 0.5-2초 | 방사형 속도, 중력, 항력 | 가산 | 버스트 방출, 하위 이미터 |
| 마법/주문(Magic/Spell) | 100-500 | 1-3초 | 인력기, 소용돌이 | 가산 | 회오리, 다채로운 색상 |
| 먼지(Dust) | 20-100 | 1-4초 | 바람, 항력 | 알파 | 갈색, 크고, 느림 |

---

## 요약

| 개념 | 핵심 아이디어 |
|------|-------------|
| 파티클 생명 주기 | 이미터 생성 → 업데이트 (힘 + 적분) → 렌더 → age > lifetime 시 소멸 |
| 오브젝트 풀 | 고정 크기 배열; 메모리 할당 방지를 위해 죽은 파티클 재사용 |
| 힘 | 중력, 바람, 항력 ($-c_d \mathbf{v}$), 인력기 ($k/r^2$), 소용돌이 |
| 심플렉틱 오일러 | 먼저 속도 갱신, 그 다음 위치; 더 나은 에너지 보존 |
| 벨렛 적분 | 2차 정확도; 현재와 이전 위치 사용; 제약 조건에 적합 |
| 빌보드 | 카메라를 향하는 쿼드; 카메라의 오른쪽/위 벡터로 구성 |
| GPU 파티클 | 트랜스폼 피드백 (VBO 핑퐁) 또는 컴퓨트 셰이더; 수백만 파티클 |
| 레이 마칭 | 레이를 따라 이동, 밀도와 색상 누적; 연기, 안개, 불꽃용 |
| 비어-람베르트 법칙 | 투과율 $T = e^{-\sigma \cdot s}$; 볼륨 내 빛 흡수를 모델링 |

## 연습 문제

1. **파티클 분수**: 초기 속도로 위쪽으로 파티클을 방출하고, 중력을 적용하며, 파티클이 지면 평면(y = 0)에서 70% 반발 계수로 튕기는 파티클 시스템을 만드세요.

2. **소용돌이 인력기**: 불꽃 파티클 시스템에 소용돌이 힘을 추가하세요. 다양한 강도와 축 방향을 실험해보세요. 시각적 이펙트는 어떻게 변하나요?

3. **적분 비교**: (a) 오일러, (b) 심플렉틱 오일러, (c) 벨렛 적분을 사용해 중력 하에서 파티클을 시뮬레이션하세요. 10초 동안의 궤적과 에너지 보존을 비교하세요. 각 방법에 대해 총 에너지 $E = \frac{1}{2}mv^2 + mgh$의 시간 변화를 플롯하세요.

4. **불꽃놀이**: 불꽃놀이 파티클 시스템을 구현하세요: 파티클 하나가 상승한 후 방사형으로 퍼지는 100개의 하위 파티클로 폭발하고, 각 하위 파티클은 페이드아웃되며 중력의 영향을 받습니다.

5. **간단한 레이 마처**: 원형 안개 볼륨을 렌더링하는 2D 레이 마처를 구현하세요. 밀도는 중심으로부터의 거리에 따라 감소해야 합니다(가우시안 프로파일). 결과를 1D 스캔라인으로 시각화하세요.

6. **GPU 시뮬레이션 설계**: 100만 개의 파티클을 지원하는 GPU 컴퓨트 셰이더 파티클 시스템의 데이터 레이아웃을 설계하세요(이론적으로). 버퍼 구조, 워크 그룹 크기, 방출과 소멸 처리 방법을 명시하세요.

## 더 읽어보기

- Reeves, W.T. "Particle Systems -- A Technique for Modeling a Class of Fuzzy Objects." *SIGGRAPH*, 1983. (파티클 시스템 원조 논문)
- Latta, L. "Building a Million Particle System." *GDC*, 2004. (실용적인 GPU 파티클 구현)
- Stam, J. "Real-Time Fluid Dynamics for Games." *GDC*, 2003. (연기/불꽃을 위한 유체 시뮬레이션; Jos Stam의 안정적 유체)
- Bridson, R. *Fluid Simulation for Computer Graphics*, 2nd ed. CRC Press, 2015. (이펙트를 위한 포괄적인 유체 역학)
- McGuire, M. "The Graphics Codex." Online, 2024. (뛰어난 파티클 시스템 및 빌보드 렌더링 레퍼런스)
