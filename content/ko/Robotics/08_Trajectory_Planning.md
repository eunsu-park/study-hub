# 궤적 계획과 실행(Trajectory Planning and Execution)

[← 이전: 운동 계획](07_Motion_Planning.md) | [다음: 로봇 제어 →](09_Robot_Control.md)

## 학습 목표

1. 경로(path)와 궤적(trajectory)을 구별하고, 시간 매개변수화(time parameterization)가 로봇 실행에 필수적인 이유를 설명한다
2. 지정된 경계 조건을 가진 3차(cubic) 및 5차(quintic) 다항식 보간을 사용하여 점대점(point-to-point) 궤적을 설계한다
3. 다중 경유점 운동을 위한 포물선 블렌드를 가진 선형 세그먼트(LSPB, Linear Segments with Parabolic Blends)를 사용하여 경유점 궤적을 구성한다
4. 부드럽고 인간적인 운동을 위한 최소 저크(minimum-jerk) 궤적을 유도하고 구현한다
5. 관절 공간(joint-space)과 태스크 공간(task-space) 궤적 계획을 비교하고 각각이 적합한 상황을 파악한다
6. 역동역학(inverse dynamics)을 이용한 순방향 제어(feedforward control)로 궤적 추적을 구현한다

---

## 왜 중요한가

운동 계획(레슨 7)은 충돌 없는 경로 — C-space에서의 구성 시퀀스 — 를 제공한다. 하지만 경로는 *시간*의 개념이 없다. 로봇이 각 구성에 언제 있어야 하는지, 얼마나 빠르게 이동해야 하는지, 또는 어떻게 가속하고 감속해야 하는지를 지정하지 않는다. 궤적(trajectory)은 이 시간 차원을 추가한다: 경로에 시간 일정을 더한 것이다.

적절한 궤적 계획 없이는 로봇이 너무 빠르게 가속하거나(모터 토크 한계 초과), 너무 많이 저크하거나(진동 및 마모 유발), 또는 잘못된 시간에 도착할 수 있다(이동 목표를 놓치거나 동기화된 기계와 충돌). 궤적 계획은 추상적인 기하학의 세계와 액추에이터 제약, 매끄러움 요구사항, 타이밍 조정의 물리적 세계가 만나는 곳이다. 컨트롤러가 제어를 넘겨받기 전 마지막 단계다.

> **비유**: 궤적은 안무된 춤과 같다 — 단순히 "어디로" 가는지가 아니라 "언제"와 "얼마나 빠르게"다. 경로는 무대 위에서 댄서가 가야 할 위치 순서를 알려주고; 궤적은 템포, 각 동작으로의 가속, 그리고 모든 스텝의 정확한 타이밍을 알려준다.

---

## 경로(Path) vs 궤적(Trajectory)

### 정의

- **경로(Path)**: 스칼라 $s \in [0, 1]$로 매개변수화된 C-space의 기하학적 곡선: $\mathbf{q}(s)$
- **궤적(Trajectory)**: 시간 매개변수화된 경로: $t \in [t_0, t_f]$에 대한 $\mathbf{q}(t)$

경로는 "로봇이 어떤 구성을 거쳐 지나가는가?"에 답한다.
궤적은 "로봇이 각 구성에 언제 도달하며, 속도와 가속도는 얼마인가?"에 답한다.

### 시간이 중요한 이유

| 시간 없이 (경로만) | 시간 있음 (궤적) |
|--------------------------|----------------------|
| 속도 프로파일 없음 | 각 순간의 속도 |
| 가속도 한계 없음 | 모터 토크 한계 준수 |
| 매끄러움 보장 없음 | 제어된 저크(마모, 진동) |
| 외부 이벤트와 동기화 불가 | 다른 로봇, 컨베이어와 조정 가능 |

```python
import numpy as np

def path_to_trajectory(path, velocities=None, total_time=None):
    """Convert a geometric path to a trajectory by adding time stamps.

    Why not just use constant speed? Because constant speed creates
    infinite acceleration at waypoints (instantaneous velocity change).
    Real motors cannot do this. We need smooth velocity profiles that
    start and stop at zero velocity (unless blending through waypoints).
    """
    n_waypoints = len(path)

    if total_time is None:
        # Estimate time from path length assuming unit average speed
        total_length = sum(np.linalg.norm(path[i+1] - path[i])
                          for i in range(n_waypoints - 1))
        total_time = total_length  # 1 unit/s average speed

    # Simple approach: distribute time proportionally to segment length
    segment_lengths = [np.linalg.norm(path[i+1] - path[i])
                       for i in range(n_waypoints - 1)]
    total_length = sum(segment_lengths)

    timestamps = [0.0]
    for length in segment_lengths:
        dt = total_time * (length / total_length) if total_length > 0 else 0
        timestamps.append(timestamps[-1] + dt)

    return np.array(timestamps), path
```

---

## 점대점 궤적(Point-to-Point Trajectories)

가장 단순한 경우: 시간 $T$ 내에 구성 $\mathbf{q}_0$에서 $\mathbf{q}_f$로 이동. 각 관절은 독립적으로 계획된다.

### 3차 다항식(Cubic Polynomial)

시간에 대한 3차 다항식:

$$q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3$$

4개의 계수로 4개의 경계 조건을 만족시킬 수 있다:
- $q(0) = q_0$ (시작 위치)
- $q(T) = q_f$ (끝 위치)
- $\dot{q}(0) = v_0$ (시작 속도, 일반적으로 0)
- $\dot{q}(T) = v_f$ (끝 속도, 일반적으로 0)

선형 시스템을 풀면:

$$a_0 = q_0, \quad a_1 = v_0$$
$$a_2 = \frac{3(q_f - q_0) - (2v_0 + v_f)T}{T^2}$$
$$a_3 = \frac{-2(q_f - q_0) + (v_0 + v_f)T}{T^3}$$

```python
class CubicTrajectory:
    """Cubic polynomial trajectory between two configurations.

    Why cubic? It's the minimum-degree polynomial that can specify
    both position and velocity at start and end. This gives smooth
    (C^1 continuous) motion with zero velocity at endpoints.

    Limitation: we cannot control acceleration at the boundaries.
    The acceleration at t=0 and t=T is determined by the polynomial
    and may be discontinuous when chaining segments.
    """
    def __init__(self, q0, qf, T, v0=0, vf=0):
        """
        q0, qf: start and end positions (scalar or array)
        T: duration
        v0, vf: start and end velocities
        """
        self.q0 = np.atleast_1d(q0).astype(float)
        self.qf = np.atleast_1d(qf).astype(float)
        self.T = T
        self.v0 = np.atleast_1d(v0).astype(float)
        self.vf = np.atleast_1d(vf).astype(float)

        # Compute coefficients
        self.a0 = self.q0
        self.a1 = self.v0
        self.a2 = (3*(self.qf - self.q0) - (2*self.v0 + self.vf)*T) / T**2
        self.a3 = (-2*(self.qf - self.q0) + (self.v0 + self.vf)*T) / T**3

    def position(self, t):
        """Position at time t."""
        t = np.clip(t, 0, self.T)
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3

    def velocity(self, t):
        """Velocity at time t."""
        t = np.clip(t, 0, self.T)
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2

    def acceleration(self, t):
        """Acceleration at time t."""
        t = np.clip(t, 0, self.T)
        return 2*self.a2 + 6*self.a3*t

    def evaluate(self, n_points=100):
        """Evaluate trajectory at n_points equally spaced times."""
        times = np.linspace(0, self.T, n_points)
        positions = np.array([self.position(t) for t in times])
        velocities = np.array([self.velocity(t) for t in times])
        accelerations = np.array([self.acceleration(t) for t in times])
        return times, positions, velocities, accelerations


# Example: move joint from 0 to 90 degrees in 2 seconds
traj = CubicTrajectory(q0=0, qf=np.radians(90), T=2.0)

print("=== Cubic Trajectory (0 -> 90 deg in 2s) ===")
for t in [0, 0.5, 1.0, 1.5, 2.0]:
    q = np.degrees(traj.position(t))
    v = np.degrees(traj.velocity(t))
    a = np.degrees(traj.acceleration(t))
    print(f"t={t:.1f}s: q={q:.1f} deg, v={v:.1f} deg/s, a={a:.1f} deg/s^2")
```

### 5차 다항식(Quintic Polynomial)

5차(5th degree) 다항식:

$$q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$

6개의 계수로 6개의 경계 조건을 지정할 수 있다:
- 시작과 끝에서의 위치, 속도, **가속도**

이는 $C^2$ 연속성을 제공한다 — 매끄러운 가속도 프로파일, 즉 모터에서 순간적인 힘 점프가 없음을 의미한다.

경계 조건 $q(0) = q_0$, $\dot{q}(0) = v_0$, $\ddot{q}(0) = a_0$, $q(T) = q_f$, $\dot{q}(T) = v_f$, $\ddot{q}(T) = a_f$를 풀면:

```python
class QuinticTrajectory:
    """Quintic polynomial trajectory with full boundary conditions.

    Why quintic over cubic? Because we can also specify boundary
    accelerations. Starting and ending with zero acceleration means
    the motor torque transitions smoothly — no sudden jumps that would
    cause vibrations and mechanical stress.

    This is the standard choice for industrial robot joint trajectories.
    """
    def __init__(self, q0, qf, T, v0=0, vf=0, a0=0, af=0):
        self.q0 = np.atleast_1d(q0).astype(float)
        self.qf = np.atleast_1d(qf).astype(float)
        self.T = T

        v0 = np.atleast_1d(v0).astype(float)
        vf = np.atleast_1d(vf).astype(float)
        a0 = np.atleast_1d(a0).astype(float)
        af = np.atleast_1d(af).astype(float)

        # Solve for coefficients using the 6 boundary conditions
        # Organized as a linear system: M * [a0..a5]^T = b
        T2 = T**2
        T3 = T**3
        T4 = T**4
        T5 = T**5

        self.a0 = self.q0
        self.a1 = v0
        self.a2 = a0 / 2

        # The remaining 3 coefficients from the end conditions
        # Solving the 3x3 system:
        M = np.array([[  T3,    T4,    T5],
                       [3*T2,  4*T3,  5*T4],
                       [6*T,  12*T2, 20*T3]])

        b_pos = self.qf - self.q0 - v0*T - (a0/2)*T2
        b_vel = vf - v0 - a0*T
        b_acc = af - a0

        # Handle multi-joint case
        if self.q0.ndim == 0:
            b = np.array([b_pos, b_vel, b_acc])
            coeffs = np.linalg.solve(M, b)
            self.a3, self.a4, self.a5 = coeffs
        else:
            n = len(self.q0)
            self.a3 = np.zeros(n)
            self.a4 = np.zeros(n)
            self.a5 = np.zeros(n)
            for j in range(n):
                b = np.array([b_pos[j], b_vel[j], b_acc[j]])
                coeffs = np.linalg.solve(M, b)
                self.a3[j], self.a4[j], self.a5[j] = coeffs

    def position(self, t):
        t = np.clip(t, 0, self.T)
        return (self.a0 + self.a1*t + self.a2*t**2 +
                self.a3*t**3 + self.a4*t**4 + self.a5*t**5)

    def velocity(self, t):
        t = np.clip(t, 0, self.T)
        return (self.a1 + 2*self.a2*t + 3*self.a3*t**2 +
                4*self.a4*t**3 + 5*self.a5*t**4)

    def acceleration(self, t):
        t = np.clip(t, 0, self.T)
        return (2*self.a2 + 6*self.a3*t +
                12*self.a4*t**2 + 20*self.a5*t**3)

    def jerk(self, t):
        """Third derivative — rate of change of acceleration.

        Why care about jerk? Because jerk translates to the rate of
        change of motor torque, which causes mechanical vibrations.
        Minimizing jerk reduces wear and produces smoother motion.
        """
        t = np.clip(t, 0, self.T)
        return 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2


# Compare cubic and quintic
print("\n=== Cubic vs Quintic Comparison ===")
T = 2.0
q0, qf = 0, np.radians(90)

cubic = CubicTrajectory(q0, qf, T)
quintic = QuinticTrajectory(q0, qf, T)

print(f"{'Time':>5s} | {'Cubic pos':>10s} {'vel':>8s} {'acc':>8s} | "
      f"{'Quintic pos':>11s} {'vel':>8s} {'acc':>8s}")
print("-" * 75)
for t in [0, 0.5, 1.0, 1.5, 2.0]:
    qc = np.degrees(cubic.position(t))
    vc = np.degrees(cubic.velocity(t))
    ac = np.degrees(cubic.acceleration(t))
    qq = np.degrees(quintic.position(t))
    vq = np.degrees(quintic.velocity(t))
    aq = np.degrees(quintic.acceleration(t))
    print(f"{t:5.1f} | {qc:10.2f} {vc:8.2f} {ac:8.2f} | "
          f"{qq:11.2f} {vq:8.2f} {aq:8.2f}")

# Key difference: acceleration at boundaries
print(f"\nCubic accel at t=0:  {np.degrees(cubic.acceleration(0)):.2f} deg/s^2")
print(f"Quintic accel at t=0: {np.degrees(quintic.acceleration(0)):.2f} deg/s^2")
print("(Quintic starts with zero acceleration — smoother!)")
```

---

## 경유점 궤적(Via-Point Trajectories)

### 다중 경유점 문제

실제 작업 대부분은 여러 경유점(via points)을 통과해야 한다 — 단순히 A에서 B로 가는 것이 아니라, A에서 B, C, D를 거쳐 E로 가야 한다. 필요한 궤적:

1. 각 경유점을 통과하거나 근접하게 지나간다
2. 매끄러운 속도를 유지한다 (요청하지 않는 한 갑작스러운 정지 없음)
3. 관절 속도 및 가속도 한계를 준수한다

### 포물선 블렌드를 가진 선형 세그먼트(LSPB, Linear Segments with Parabolic Blends)

LSPB(**사다리꼴 속도 프로파일**이라고도 함)는 산업에서 가장 일반적인 궤적 유형이다:

- **선형 세그먼트(Linear segments)**: 경유점 사이에서 일정 속도
- **포물선 블렌드(Parabolic blends)**: 전환을 부드럽게 하기 위한 경유점에서의 2차 위치 (일정 가속도)

속도 프로파일은 사다리꼴처럼 보인다: 가속, 순항, 감속.

```python
class LSPBTrajectory:
    """Linear Segment with Parabolic Blend (trapezoidal velocity profile).

    Why LSPB? It's the workhorse of industrial robotics because:
    1. Simple to compute and implement on real-time controllers
    2. Respects velocity AND acceleration limits
    3. Time-optimal among trapezoidal profiles
    4. Easy to synchronize multi-axis motion

    The velocity profile is trapezoidal: ramp up, cruise, ramp down.
    The position profile is: parabola (acceleration), line (cruise),
    parabola (deceleration).
    """
    def __init__(self, q0, qf, v_max, a_max, T=None):
        """
        q0, qf: start and end positions
        v_max: maximum velocity (absolute)
        a_max: maximum acceleration (absolute)
        T: total duration (optional — computed if None)
        """
        self.q0 = float(q0)
        self.qf = float(qf)
        self.v_max = abs(v_max)
        self.a_max = abs(a_max)

        h = self.qf - self.q0  # total displacement
        self.sign = np.sign(h) if h != 0 else 1
        h_abs = abs(h)

        # Check if we can reach maximum velocity
        # Time to accelerate to v_max: t_acc = v_max / a_max
        # Distance during acceleration: d_acc = 0.5 * a_max * t_acc^2 = v_max^2 / (2*a_max)
        d_acc = self.v_max**2 / (2 * self.a_max)

        if 2 * d_acc >= h_abs:
            # Cannot reach max velocity — triangular profile
            # Peak velocity: v_peak = sqrt(h_abs * a_max)
            self.v_cruise = np.sqrt(h_abs * self.a_max)
            self.t_acc = self.v_cruise / self.a_max
            self.t_cruise = 0
            self.t_dec = self.t_acc
            self.T = 2 * self.t_acc
            self.is_triangular = True
        else:
            # Full trapezoidal profile
            self.v_cruise = self.v_max
            self.t_acc = self.v_cruise / self.a_max
            d_cruise = h_abs - 2 * d_acc
            self.t_cruise = d_cruise / self.v_cruise
            self.t_dec = self.t_acc
            self.T = self.t_acc + self.t_cruise + self.t_dec
            self.is_triangular = False

        if T is not None and T > self.T:
            # If user specifies longer duration, reduce cruise speed
            self.T = T
            self._recompute_for_duration(h_abs, T)

    def _recompute_for_duration(self, h_abs, T):
        """Recompute profile for a specified total duration.

        Why allow user-specified duration? For multi-joint synchronization.
        All joints must finish at the same time, so slower joints must
        stretch their profiles to match the slowest joint.
        """
        # Given T and a_max, find v_cruise such that h = v*T - v^2/a
        # This is a quadratic in v: v^2/a - v*T + h = 0
        a = 1.0 / self.a_max
        b = -T
        c = h_abs
        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            # Impossible with given a_max and T — just use triangular
            return

        v = (-b - np.sqrt(discriminant)) / (2*a)
        if v > self.v_max:
            v = self.v_max

        self.v_cruise = v
        self.t_acc = v / self.a_max
        self.t_cruise = T - 2 * self.t_acc
        self.t_dec = self.t_acc
        self.is_triangular = (self.t_cruise <= 0)

    def position(self, t):
        t = np.clip(t, 0, self.T)
        s = self.sign
        v = self.v_cruise
        a = self.a_max

        if t <= self.t_acc:
            # Acceleration phase (parabolic)
            return self.q0 + s * 0.5 * a * t**2
        elif t <= self.t_acc + self.t_cruise:
            # Cruise phase (linear)
            dt = t - self.t_acc
            q_acc_end = self.q0 + s * 0.5 * a * self.t_acc**2
            return q_acc_end + s * v * dt
        else:
            # Deceleration phase (parabolic)
            dt = t - self.t_acc - self.t_cruise
            q_cruise_end = (self.q0 + s * 0.5 * a * self.t_acc**2 +
                           s * v * self.t_cruise)
            return q_cruise_end + s * v * dt - s * 0.5 * a * dt**2

    def velocity(self, t):
        t = np.clip(t, 0, self.T)
        s = self.sign
        a = self.a_max

        if t <= self.t_acc:
            return s * a * t
        elif t <= self.t_acc + self.t_cruise:
            return s * self.v_cruise
        else:
            dt = t - self.t_acc - self.t_cruise
            return s * (self.v_cruise - a * dt)

    def acceleration(self, t):
        t = np.clip(t, 0, self.T)
        s = self.sign

        if t < self.t_acc:
            return s * self.a_max
        elif t < self.t_acc + self.t_cruise:
            return 0.0
        elif t < self.T:
            return -s * self.a_max
        else:
            return 0.0


# Example: trapezoidal profile
print("\n=== LSPB (Trapezoidal) Trajectory ===")
lspb = LSPBTrajectory(q0=0, qf=np.radians(90), v_max=1.0, a_max=2.0)
print(f"Profile: {'Triangular' if lspb.is_triangular else 'Trapezoidal'}")
print(f"Duration: {lspb.T:.3f}s")
print(f"t_acc={lspb.t_acc:.3f}s, t_cruise={lspb.t_cruise:.3f}s, "
      f"t_dec={lspb.t_dec:.3f}s")

print(f"\n{'Time':>5s} {'Position (deg)':>14s} {'Velocity':>10s} {'Accel':>10s}")
for t in np.linspace(0, lspb.T, 9):
    q = np.degrees(lspb.position(t))
    v = np.degrees(lspb.velocity(t))
    a = np.degrees(lspb.acceleration(t))
    print(f"{t:5.2f} {q:14.2f} {v:10.2f} {a:10.2f}")
```

### 다중 경유점 LSPB

여러 경유점에 대해, 각 경유점에서 포물선 블렌드를 가진 LSPB 세그먼트를 연결한다:

```python
class MultiWaypointLSPB:
    """LSPB trajectory through multiple waypoints.

    Why chain segments? Because industrial tasks are rarely just A-to-B.
    A pick-and-place task might require: home → approach → grasp →
    lift → move → place → release → home. Each transition must be smooth.
    """
    def __init__(self, waypoints, times, v_max, a_max):
        """
        waypoints: list of joint positions
        times: list of arrival times (same length as waypoints)
        v_max: max velocity per segment
        a_max: max acceleration
        """
        self.waypoints = np.array(waypoints, dtype=float)
        self.times = np.array(times, dtype=float)
        self.n_segments = len(waypoints) - 1
        self.v_max = v_max
        self.a_max = a_max

        # Build individual segments
        self.segments = []
        for i in range(self.n_segments):
            dt = self.times[i+1] - self.times[i]
            seg = LSPBTrajectory(
                self.waypoints[i], self.waypoints[i+1],
                v_max, a_max, T=dt
            )
            self.segments.append(seg)

    def position(self, t):
        """Evaluate position at time t."""
        t = np.clip(t, self.times[0], self.times[-1])

        # Find active segment
        for i in range(self.n_segments):
            if t <= self.times[i+1]:
                local_t = t - self.times[i]
                return self.segments[i].position(local_t)

        return self.waypoints[-1]

    def velocity(self, t):
        t = np.clip(t, self.times[0], self.times[-1])
        for i in range(self.n_segments):
            if t <= self.times[i+1]:
                local_t = t - self.times[i]
                return self.segments[i].velocity(local_t)
        return 0.0


# Example: 4-waypoint trajectory
waypoints = np.radians([0, 45, 90, 30])
times = [0, 1.0, 2.5, 4.0]

multi = MultiWaypointLSPB(waypoints, times, v_max=1.5, a_max=3.0)

print("\n=== Multi-Waypoint LSPB ===")
print("Waypoints:", np.degrees(waypoints).round(1), "deg")
print("Times:", times, "s")
print()
for t in np.linspace(0, 4.0, 17):
    q = np.degrees(multi.position(t))
    v = np.degrees(multi.velocity(t))
    print(f"t={t:.2f}s: q={q:>7.2f} deg, v={v:>7.2f} deg/s")
```

---

## 최소 저크(Minimum-Jerk) 궤적

### 동기

**저크(jerk)**는 위치의 3차 미분(가속도의 변화율)이다. 저크를 최소화하면 가장 매끄러운 가능한 운동이 생성된다 — 이는 다음에 중요하다:
- 기계적 진동 감소
- 편안한 인간-로봇 상호작용 (인간은 저크가 많은 운동을 불안하게 느낀다)
- 진동이 정확도를 저하시키는 정밀 응용

### 유도

최소 저크 궤적은 다음을 최소화한다:

$$J = \int_0^T \left(\frac{d^3 q}{dt^3}\right)^2 dt$$

$t = 0$과 $t = T$에서 위치, 속도, 가속도에 대한 경계 조건을 만족하면서.

해는 특정 계수를 가진 **5차 다항식**이다:

영 경계 속도 및 가속도에 대해:

$$q(t) = q_0 + (q_f - q_0) \left[ 10\left(\frac{t}{T}\right)^3 - 15\left(\frac{t}{T}\right)^4 + 6\left(\frac{t}{T}\right)^5 \right]$$

```python
class MinimumJerkTrajectory:
    """Minimum-jerk trajectory — the smoothest possible motion.

    Why minimize jerk? Flash & Hogan (1985) showed that human arm
    movements closely follow minimum-jerk profiles. This makes
    minimum-jerk trajectories ideal for:
    - Human-robot collaboration (predictable, non-threatening motion)
    - Precision tasks (minimal vibration)
    - General-purpose smooth trajectories

    The math: minimizing integral of jerk^2 with boundary conditions
    on position, velocity, and acceleration yields a 5th-order polynomial.
    """
    def __init__(self, q0, qf, T, v0=0, vf=0, a0=0, af=0):
        self.q0 = np.atleast_1d(q0).astype(float)
        self.qf = np.atleast_1d(qf).astype(float)
        self.T = T
        self.v0 = np.atleast_1d(v0).astype(float)
        self.vf = np.atleast_1d(vf).astype(float)
        self.a0 = np.atleast_1d(a0).astype(float)
        self.af = np.atleast_1d(af).astype(float)

    def _tau(self, t):
        """Normalized time: tau = t / T, in [0, 1]."""
        return np.clip(t / self.T, 0, 1)

    def position(self, t):
        tau = self._tau(t)
        h = self.qf - self.q0
        T = self.T

        if np.allclose(self.v0, 0) and np.allclose(self.vf, 0) and \
           np.allclose(self.a0, 0) and np.allclose(self.af, 0):
            # Simplified form for zero boundary conditions
            s = 10*tau**3 - 15*tau**4 + 6*tau**5
            return self.q0 + h * s
        else:
            # General form (same as quintic with specific coefficients)
            return QuinticTrajectory(
                self.q0, self.qf, T,
                self.v0, self.vf, self.a0, self.af
            ).position(t)

    def velocity(self, t):
        tau = self._tau(t)
        h = self.qf - self.q0
        T = self.T

        if np.allclose(self.v0, 0) and np.allclose(self.vf, 0) and \
           np.allclose(self.a0, 0) and np.allclose(self.af, 0):
            s_dot = (30*tau**2 - 60*tau**3 + 30*tau**4) / T
            return h * s_dot
        else:
            return QuinticTrajectory(
                self.q0, self.qf, T,
                self.v0, self.vf, self.a0, self.af
            ).velocity(t)

    def acceleration(self, t):
        tau = self._tau(t)
        h = self.qf - self.q0
        T = self.T

        if np.allclose(self.v0, 0) and np.allclose(self.vf, 0) and \
           np.allclose(self.a0, 0) and np.allclose(self.af, 0):
            s_ddot = (60*tau - 180*tau**2 + 120*tau**3) / T**2
            return h * s_ddot
        else:
            return QuinticTrajectory(
                self.q0, self.qf, T,
                self.v0, self.vf, self.a0, self.af
            ).acceleration(t)

    def jerk(self, t):
        tau = self._tau(t)
        h = self.qf - self.q0
        T = self.T
        s_dddot = (60 - 360*tau + 360*tau**2) / T**3
        return h * s_dddot


# Compare trajectory smoothness
print("\n=== Minimum-Jerk vs LSPB ===")
q0, qf, T = 0, np.radians(90), 2.0

mjt = MinimumJerkTrajectory(q0, qf, T)
lspb = LSPBTrajectory(q0, qf, v_max=1.5, a_max=3.0, T=T)

# Compute total jerk for each
n_eval = 1000
times = np.linspace(0, T, n_eval)
dt = T / n_eval

jerk_mj = sum(mjt.jerk(t)**2 for t in times) * dt
jerk_lspb = sum(lspb.acceleration(t)**2 for t in times) * dt  # approx jerk via accel changes

# Estimate LSPB jerk (infinite at transition points, so we measure accel discontinuity)
accel_changes_lspb = sum(abs(lspb.acceleration(times[i+1]) - lspb.acceleration(times[i]))
                        for i in range(len(times)-1))

print(f"Min-jerk: integrated jerk^2 = {jerk_mj:.4f}")
print(f"LSPB: acceleration discontinuities (sum) = {np.degrees(accel_changes_lspb):.1f} deg/s^2")
print(f"\nMin-jerk has continuous derivatives everywhere;")
print(f"LSPB has discontinuous acceleration at blend boundaries.")
```

---

## 시간 최적 궤적(Time-Optimal Trajectories)

### 개념

**시간 최적 궤적(Time-optimal trajectory)**은 관절 속도, 가속도, 토크 한계를 준수하면서 최소 시간 내에 경로를 횡단한다. 이는 최적 제어 문제다:

$$\min_{s(t)} \int_0^{T_f} dt = T_f$$

다음 조건에서:
- $\mathbf{q}(t) = \mathbf{q}(s(t))$가 기하학적 경로를 따름
- $|\dot{q}_i| \leq \dot{q}_{i,max}$
- $|\ddot{q}_i| \leq \ddot{q}_{i,max}$
- $|\tau_i| \leq \tau_{i,max}$

### 위상 평면 방법(Phase-Plane Method) (간략 개요)

시간 최적 문제는 $s$가 경로 매개변수인 위상 평면 $(\dot{s}, s)$를 분석함으로써 풀 수 있다. 핵심 통찰: 경로의 각 점에서 토크 한계에 의해 결정되는 최대 및 최소 가속도가 있다. 최적 궤적은 최종 조건을 만족시키기 위해 최대 감속으로 전환해야 할 때까지 최대 가속도 곡선을 따른다.

```python
def time_optimal_estimate(waypoints, v_limits, a_limits):
    """Estimate time-optimal duration for a multi-waypoint trajectory.

    Why estimate? Full time-optimal trajectory computation requires
    solving a nonlinear optimal control problem, which is complex.
    This simple estimate gives a lower bound on the execution time
    that's useful for task planning and scheduling.

    For each segment, the minimum time is:
    T_min = max over all joints of min time for that joint
    (because all joints must be synchronized)
    """
    n_segments = len(waypoints) - 1
    total_time = 0

    for seg in range(n_segments):
        segment_times = []
        for j in range(len(waypoints[0])):
            dq = abs(waypoints[seg+1][j] - waypoints[seg][j])

            # Time for triangular profile (if can't reach max velocity)
            t_tri = 2 * np.sqrt(dq / a_limits[j])

            # Time for trapezoidal profile
            v_peak = np.sqrt(dq * a_limits[j])
            if v_peak <= v_limits[j]:
                t_seg = t_tri
            else:
                t_acc = v_limits[j] / a_limits[j]
                d_acc = 0.5 * a_limits[j] * t_acc**2
                d_cruise = dq - 2 * d_acc
                t_cruise = d_cruise / v_limits[j]
                t_seg = 2 * t_acc + t_cruise

            segment_times.append(t_seg)

        # Slowest joint determines segment time
        total_time += max(segment_times)

    return total_time

# Example: 2-joint robot moving through 3 waypoints
waypoints = [
    np.radians([0, 0]),
    np.radians([90, 45]),
    np.radians([45, 90]),
]
v_limits = np.radians([180, 180])   # 180 deg/s max
a_limits = np.radians([360, 360])   # 360 deg/s^2 max

t_opt = time_optimal_estimate(waypoints, v_limits, a_limits)
print(f"\nEstimated time-optimal duration: {t_opt:.3f}s")
```

---

## 태스크 공간 vs 관절 공간 계획

### 관절 공간 계획(Joint-Space Planning)

관절 좌표 $\mathbf{q}(t)$에서 궤적을 계획한다. 각 관절은 독립적으로 자체 다항식/LSPB를 따른다.

**장점**:
- 단순하고 빠른 계산
- 관절 한계를 자연스럽게 준수
- 특이점(singularity) 문제 없음
- 각 관절을 독립적으로 계획 가능

**단점**:
- 작업 공간에서 엔드 이펙터 경로가 직관적이지 않음 (곡선형, 예상치 못한 모양)
- 도구의 직선 운동을 보장할 수 없음

### 태스크 공간 계획(Task-Space Planning)

태스크 좌표 $\mathbf{x}(t) = (x, y, z, \phi, \theta, \psi)^T$에서 궤적을 계획하고 각 시간 스텝에서 역기구학(IK)을 사용한다.

**장점**:
- 엔드 이펙터 경로의 직접 제어 (예: 직선, 호)
- 작업 공간에서 정의된 작업에 직관적 (용접 솔기, 도색)

**단점**:
- 각 시간 스텝에서 IK 필요 (계산 비용)
- 특이점에 직면할 수 있음
- 관절 한계를 위반할 수 있음
- IK에 해가 없을 수 있음 (경로가 작업 공간 외부를 지남)

```python
def task_space_trajectory(fk_func, ik_func, x_start, x_end, T,
                          n_points=100, traj_type='quintic'):
    """Plan a straight-line trajectory in task space.

    Why straight-line in task space? Many industrial tasks require it:
    welding along a seam, painting a surface, or inserting a peg into
    a hole. Joint-space planning would produce a curved path in workspace.

    The catch: we must solve IK at every point along the trajectory.
    If any point is singular or unreachable, the trajectory fails.
    """
    times = np.linspace(0, T, n_points)
    x_traj = []
    q_traj = []

    for t in times:
        # Interpolate in task space
        tau = t / T
        if traj_type == 'quintic':
            s = 10*tau**3 - 15*tau**4 + 6*tau**5
        elif traj_type == 'linear':
            s = tau
        else:
            s = tau  # default linear

        x_desired = x_start + s * (x_end - x_start)
        x_traj.append(x_desired)

        # Solve IK for each task-space point
        q = ik_func(x_desired)
        if q is None:
            print(f"IK failed at t={t:.3f}s, x={x_desired}")
            return None, None, None
        q_traj.append(q)

    return times, np.array(x_traj), np.array(q_traj)

def joint_space_trajectory(q_start, q_end, T, n_points=100):
    """Plan independently in joint space using quintic profiles.

    Why independent joint planning? Because it's guaranteed to work —
    no IK failures, no singularities, no workspace boundary issues.
    The resulting end-effector path will be curved, but that's
    acceptable for many tasks (pick-and-place, repositioning).
    """
    traj = QuinticTrajectory(q_start, q_end, T)
    times = np.linspace(0, T, n_points)
    q_traj = np.array([traj.position(t) for t in times])
    return times, q_traj
```

### 태스크 공간과 관절 공간 선택

| 기준 | 관절 공간 | 태스크 공간 |
|-----------|-------------|------------|
| 단순 위치 변경 | 선호 | 불필요 |
| 직선 도구 경로 필요 | 부적합 | 필수 |
| 특이점 근처 | 안전 | 위험 |
| 작업 공간 경계 근처 | 안전 | 실패 가능 |
| 계산 비용 | 낮음 | 높음 (스텝당 IK) |
| 직관적인 엔드 이펙터 운동 | 아니오 | 예 |

---

## 궤적 추적과 순방향 제어(Trajectory Tracking and Feedforward Control)

### 추적 문제

궤적 $\mathbf{q}_d(t)$, $\dot{\mathbf{q}}_d(t)$, $\ddot{\mathbf{q}}_d(t)$가 있으면, 컨트롤러가 이를 정확하게 따르도록 해야 한다. 레슨 6의 **계산 토크(computed torque)** 컨트롤러가 이를 수행한다:

$$\boldsymbol{\tau} = M(\mathbf{q})\left[\ddot{\mathbf{q}}_d + K_d(\dot{\mathbf{q}}_d - \dot{\mathbf{q}}) + K_p(\mathbf{q}_d - \mathbf{q})\right] + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

핵심 통찰: 궤적이 $\ddot{\mathbf{q}}_d$를 **순방향(feedforward)**으로 제공하여, 피드백만 사용하는 제어에 비해 추적 성능을 극적으로 향상시킨다.

```python
class TrajectoryTracker:
    """Complete trajectory tracking system with feedforward + feedback.

    Why feedforward? Consider following a fast trajectory. With
    feedback-only (PD) control, the controller only reacts AFTER
    an error appears. With feedforward, the controller proactively
    applies the torques needed for the desired acceleration BEFORE
    the error grows. The feedback term only corrects small residual errors.

    This is the standard architecture for industrial robot controllers.
    """
    def __init__(self, arm, Kp, Kd, use_feedforward=True):
        self.arm = arm
        self.Kp = Kp
        self.Kd = Kd
        self.use_feedforward = use_feedforward

    def compute_torque(self, q, q_dot, q_des, q_dot_des, q_ddot_des):
        """Compute control torque for trajectory tracking."""
        # Error
        e = q_des - q
        e_dot = q_dot_des - q_dot

        # Feedback: PD control
        a_fb = self.Kp @ e + self.Kd @ e_dot

        if self.use_feedforward:
            # Feedforward: desired acceleration + feedback correction
            a = q_ddot_des + a_fb
            # Computed torque
            M = self.arm.inertia_matrix(q)
            C = self.arm.coriolis_matrix(q, q_dot)
            g = self.arm.gravity_vector(q)
            tau = M @ a + C @ q_dot + g
        else:
            # Feedback only (PD + gravity compensation)
            g = self.arm.gravity_vector(q)
            tau = self.Kp @ e + self.Kd @ e_dot + g

        return tau

    def execute_trajectory(self, trajectory, dt=0.001):
        """Execute a trajectory and return tracking results.

        Why simulate? To evaluate trajectory quality BEFORE sending
        it to the real robot. Simulation catches issues like:
        - Torque limit violations
        - Excessive tracking error
        - Instability from poor gain tuning
        """
        times, positions, velocities, accelerations = trajectory

        q = positions[0].copy()
        q_dot = velocities[0].copy()
        n_steps = len(times)

        results = {
            'times': [], 'q_actual': [], 'q_desired': [],
            'error': [], 'torques': []
        }

        for i in range(n_steps):
            t = times[i]
            q_des = positions[i]
            q_dot_des = velocities[i]
            q_ddot_des = accelerations[i]

            # Compute control torque
            tau = self.compute_torque(q, q_dot, q_des, q_dot_des, q_ddot_des)

            # Record results
            results['times'].append(t)
            results['q_actual'].append(q.copy())
            results['q_desired'].append(q_des.copy())
            results['error'].append(np.linalg.norm(q - q_des))
            results['torques'].append(tau.copy())

            # Simulate forward dynamics
            q_ddot = self.arm.forward_dynamics(q, q_dot, tau)
            q_dot = q_dot + q_ddot * dt
            q = q + q_dot * dt

        # Convert to arrays
        for key in results:
            results[key] = np.array(results[key])

        return results


# Demonstrate tracking with feedforward vs feedback-only
# Import the arm dynamics from Lesson 6
class TwoLinkDynamics:
    def __init__(self, m1, m2, l1, l2, lc1, lc2, I1, I2, g=9.81):
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.lc1, self.lc2 = lc1, lc2
        self.I1, self.I2 = I1, I2
        self.g = g

    def inertia_matrix(self, q):
        c2 = np.cos(q[1])
        a = self.m1*self.lc1**2 + self.m2*(self.l1**2 + self.lc2**2 + \
            2*self.l1*self.lc2*c2) + self.I1 + self.I2
        b = self.m2*(self.lc2**2 + self.l1*self.lc2*c2) + self.I2
        d = self.m2*self.lc2**2 + self.I2
        return np.array([[a, b], [b, d]])

    def coriolis_matrix(self, q, q_dot):
        h = self.m2 * self.l1 * self.lc2 * np.sin(q[1])
        return np.array([[-h*q_dot[1], -h*(q_dot[0] + q_dot[1])],
                          [h*q_dot[0], 0]])

    def gravity_vector(self, q):
        g = self.g
        g1 = (self.m1*self.lc1 + self.m2*self.l1)*g*np.cos(q[0]) + \
              self.m2*self.lc2*g*np.cos(q[0] + q[1])
        g2 = self.m2*self.lc2*g*np.cos(q[0] + q[1])
        return np.array([g1, g2])

    def forward_dynamics(self, q, q_dot, tau):
        M = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        g = self.gravity_vector(q)
        return np.linalg.solve(M, tau - C @ q_dot - g)


arm = TwoLinkDynamics(m1=5, m2=3, l1=0.5, l2=0.4, lc1=0.25, lc2=0.2, I1=0.1, I2=0.05)

# Create a quintic trajectory
q0 = np.radians([0, 0])
qf = np.radians([60, 45])
T = 1.5
traj = QuinticTrajectory(q0, qf, T)
times = np.linspace(0, T, 500)
positions = np.array([traj.position(t) for t in times])
velocities = np.array([traj.velocity(t) for t in times])
accelerations = np.array([traj.acceleration(t) for t in times])
trajectory_data = (times, positions, velocities, accelerations)

# Track with feedforward
Kp = np.diag([100, 80])
Kd = np.diag([20, 15])
tracker_ff = TrajectoryTracker(arm, Kp, Kd, use_feedforward=True)
results_ff = tracker_ff.execute_trajectory(trajectory_data)

# Track without feedforward
tracker_fb = TrajectoryTracker(arm, Kp, Kd, use_feedforward=False)
results_fb = tracker_fb.execute_trajectory(trajectory_data)

print("\n=== Trajectory Tracking Comparison ===")
print(f"{'Metric':<30s} {'With FF':>12s} {'Without FF':>12s}")
print("-" * 55)
max_err_ff = np.max(results_ff['error'])
max_err_fb = np.max(results_fb['error'])
avg_err_ff = np.mean(results_ff['error'])
avg_err_fb = np.mean(results_fb['error'])
max_tau_ff = np.max(np.abs(results_ff['torques']))
max_tau_fb = np.max(np.abs(results_fb['torques']))

print(f"{'Max tracking error (rad)':<30s} {max_err_ff:12.6f} {max_err_fb:12.6f}")
print(f"{'Avg tracking error (rad)':<30s} {avg_err_ff:12.6f} {avg_err_fb:12.6f}")
print(f"{'Max torque (N*m)':<30s} {max_tau_ff:12.3f} {max_tau_fb:12.3f}")
print(f"{'Final error (rad)':<30s} "
      f"{results_ff['error'][-1]:12.6f} {results_fb['error'][-1]:12.6f}")
```

---

## 다중 관절 동기화(Multi-Joint Synchronization)

여러 관절에 대한 궤적을 동시에 계획할 때, 모든 관절이 동시에 시작하고 끝나야 한다. 해법: 가장 느린 관절을 찾고 나머지 관절을 그에 맞게 늘린다.

```python
def synchronized_lspb(q_start, q_end, v_limits, a_limits):
    """Compute synchronized LSPB trajectories for all joints.

    Why synchronize? Because an unsynchronized multi-joint trajectory
    means some joints finish before others. The end-effector path
    becomes unpredictable — the arm could swing through obstacles
    during the time the fast joints wait for the slow joints.

    Strategy: compute the minimum time for each joint independently,
    take the maximum, then stretch all joints to that duration.
    """
    n_joints = len(q_start)
    joint_times = []

    # Compute minimum time for each joint independently
    for j in range(n_joints):
        seg = LSPBTrajectory(q_start[j], q_end[j], v_limits[j], a_limits[j])
        joint_times.append(seg.T)

    # Synchronize to the slowest joint
    T_sync = max(joint_times)
    print(f"Individual joint times: {[f'{t:.3f}' for t in joint_times]}")
    print(f"Synchronized duration: {T_sync:.3f}s")

    # Create synchronized trajectories
    trajectories = []
    for j in range(n_joints):
        seg = LSPBTrajectory(q_start[j], q_end[j], v_limits[j], a_limits[j], T=T_sync)
        trajectories.append(seg)

    return trajectories, T_sync


# Example: synchronize 3-joint motion
q_start = np.radians([0, 0, 0])
q_end = np.radians([90, 30, 120])  # joint 3 has the largest motion
v_limits = np.radians([180, 180, 180])
a_limits = np.radians([360, 360, 360])

print("\n=== Synchronized Multi-Joint LSPB ===")
trajs, T_sync = synchronized_lspb(q_start, q_end, v_limits, a_limits)

# Verify all joints finish at the same time
print(f"\nAt t={T_sync:.3f}s:")
for j, traj in enumerate(trajs):
    q_final = np.degrees(traj.position(T_sync))
    q_target = np.degrees(q_end[j])
    print(f"  Joint {j+1}: position={q_final:.2f} deg (target: {q_target:.1f})")
```

---

## 요약

- **궤적(trajectory)**은 기하학적 경로에 시간 매개변수화를 추가하여 각 순간의 위치, 속도, 가속도를 지정한다
- **3차 다항식(Cubic polynomials)**은 위치와 속도 경계 조건을 만족시킨다 ($C^1$ 매끄러움)
- **5차 다항식(Quintic polynomials)**은 가속도 경계 조건도 만족시킨다 ($C^2$ 매끄러움), 산업 응용에 선호됨
- **LSPB** (사다리꼴 속도)는 단순성과 속도/가속도 한계의 직접적인 적용으로 산업 표준임
- **최소 저크(Minimum-jerk)** 궤적은 자연스러운 인간 팔 운동과 일치하는 가장 매끄러운 운동을 생성한다
- **시간 최적(Time-optimal)** 궤적은 액추에이터 제약 하에서 실행 시간을 최소화한다
- **관절 공간(Joint-space)** 계획은 더 단순하고 특이점이 없다; **태스크 공간(task-space)** 계획은 직관적인 엔드 이펙터 경로를 제공한다
- 원하는 궤적의 가속도를 사용한 **순방향 제어(feedforward control)**는 피드백만 사용하는 접근에 비해 추적 정확도를 극적으로 향상시킨다
- **다중 관절 동기화(Multi-joint synchronization)**는 예측 가능한 엔드 이펙터 운동을 위해 모든 관절이 동시에 끝나도록 보장한다

---

## 연습문제

### 연습문제 1: 다항식 궤적

$q_0 = 0°$에서 $q_f = 120°$로 $T = 3$초 내에 이동하는 단일 관절에 대해:
1. 3차 및 5차 궤적을 계산한다 (영 경계 속도/가속도)
2. 두 궤적에 대한 위치, 속도, 가속도, 저크를 평가하고 그래프를 그린다
3. 최대 속도와 가속도를 비교한다. 어느 것이 더 큰가?
4. 최대 속도 한계가 60 deg/s라면, 어느 궤적이 이를 위반하는가? 그렇다면 필요한 최소 $T$는 얼마인가?

### 연습문제 2: LSPB 설계

다음 조건을 가진 관절에 대한 LSPB 궤적을 설계한다:
- $q_0 = -45°$, $q_f = 90°$
- $v_{max} = 100°/s$
- $a_{max} = 200°/s^2$

1. 프로파일이 사다리꼴인가 삼각형인가?
2. 총 지속 시간은 얼마인가?
3. 순항 속도는 얼마인가?
4. 이제 $T = 3$초를 지정한다 (최소보다 길다). 새로운 순항 속도는 얼마인가?

### 연습문제 3: 다중 경유점 궤적

2-자유도 로봇이 다음 경유점을 방문해야 한다:
- $q_A = (0°, 0°)$, $t = 0$ s
- $q_B = (60°, 30°)$, $t = 1.5$ s
- $q_C = (30°, 90°)$, $t = 3.0$ s
- $q_D = (0°, 0°)$, $t = 5.0$ s

1. 각 세그먼트에 5차 다항식을 사용하여 계획한다 (각 경유점에서 영 속도)
2. 각 세그먼트에 LSPB를 사용하여 계획한다
3. 매끄러움 비교: 두 접근법에 대한 가속도 프로파일을 그래프로 그린다
4. 경유점에서 속도 불연속성이 있는가 (있다면 얼마나)?

### 연습문제 4: 최소 저크 vs 5차 다항식

1. 같은 운동에 대해 최소 저크 궤적과 5차 다항식 궤적을 모두 구현한다
2. 두 궤적에 대해 적분된 저크 제곱 ($\int \dddot{q}^2 dt$)을 계산한다
3. 최소 저크 궤적이 더 낮은 적분 저크를 가짐을 검증한다
4. 저크 프로파일을 나란히 그래프로 그린다

### 연습문제 5: 궤적 추적 시뮬레이션

레슨 6의 2링크 팔 동역학을 사용하여:
1. $q = (0, 0)$에서 $q = (90°, 45°)$로 2초 내에 이동하는 5차 궤적을 생성한다
2. $K_p = 100 I$, $K_d = 20 I$로 계산 토크 제어를 구현한다
3. 추적을 시뮬레이션하고 시간에 따른 오차를 그래프로 그린다
4. 질량 파라미터에 10% 오차를 도입한다 (모델 불일치). 영향을 관찰한다
5. 제어 이득을 두 배로 늘린다. 모델 불일치 시 추적이 향상되는가 아니면 악화되는가?

---

[← 이전: 운동 계획](07_Motion_Planning.md) | [다음: 로봇 제어 →](09_Robot_Control.md)
