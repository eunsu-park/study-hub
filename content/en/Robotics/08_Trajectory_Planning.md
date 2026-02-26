# Trajectory Planning and Execution

[← Previous: Motion Planning](07_Motion_Planning.md) | [Next: Robot Control →](09_Robot_Control.md)

## Learning Objectives

1. Distinguish between a path and a trajectory, and explain why time parameterization is essential for robot execution
2. Design point-to-point trajectories using cubic and quintic polynomial interpolation with specified boundary conditions
3. Construct via-point trajectories using Linear Segments with Parabolic Blends (LSPB) for multi-waypoint motion
4. Derive and implement minimum-jerk trajectories for smooth, human-like motion
5. Compare joint-space and task-space trajectory planning and identify when each approach is appropriate
6. Implement trajectory tracking with feedforward control using inverse dynamics

---

## Why This Matters

Motion planning (Lesson 7) gives us a collision-free path — a sequence of configurations in C-space. But a path has no notion of *time*. It does not specify when the robot should be at each configuration, how fast it should be moving, or how it accelerates and decelerates. A trajectory adds this time dimension: it is a path plus a time schedule.

Without proper trajectory planning, a robot might accelerate too fast (exceeding motor torque limits), jerk too much (causing vibrations and wear), or arrive at the wrong time (missing a moving target or colliding with a synchronized machine). Trajectory planning is where the abstract world of geometry meets the physical world of actuator constraints, smoothness requirements, and timing coordination. It is the last step before the controller takes over.

> **Analogy**: A trajectory is like a choreographed dance — not just "where" to go, but "when" and "how fast." A path tells the dancer the sequence of positions on stage; a trajectory tells the dancer the tempo, the acceleration into each move, and the precise timing of every step.

---

## Path vs Trajectory

### Definitions

- **Path**: A geometric curve in C-space, parameterized by a scalar $s \in [0, 1]$: $\mathbf{q}(s)$
- **Trajectory**: A path with time parameterization: $\mathbf{q}(t)$ for $t \in [t_0, t_f]$

A path answers: "Through which configurations does the robot pass?"
A trajectory answers: "At what time does the robot reach each configuration, and with what velocity and acceleration?"

### Why Time Matters

| Without time (path only) | With time (trajectory) |
|--------------------------|----------------------|
| No velocity profile | Velocity at each instant |
| No acceleration limits | Respects motor torque limits |
| No smoothness guarantee | Controlled jerk (wear, vibration) |
| Cannot synchronize with external events | Can coordinate with other robots, conveyors |

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

## Point-to-Point Trajectories

The simplest case: move from configuration $\mathbf{q}_0$ to $\mathbf{q}_f$ in time $T$. Each joint is planned independently.

### Cubic Polynomial

A cubic polynomial in time:

$$q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3$$

With 4 coefficients, we can satisfy 4 boundary conditions:
- $q(0) = q_0$ (start position)
- $q(T) = q_f$ (end position)
- $\dot{q}(0) = v_0$ (start velocity, typically 0)
- $\dot{q}(T) = v_f$ (end velocity, typically 0)

Solving the linear system:

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

### Quintic Polynomial

A quintic (5th degree) polynomial:

$$q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$

With 6 coefficients, we can specify 6 boundary conditions:
- Position, velocity, and **acceleration** at both start and end

This gives $C^2$ continuity — smooth acceleration profiles, which means no instantaneous force jumps at the motors.

Solving the boundary conditions $q(0) = q_0$, $\dot{q}(0) = v_0$, $\ddot{q}(0) = a_0$, $q(T) = q_f$, $\dot{q}(T) = v_f$, $\ddot{q}(T) = a_f$:

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

## Via-Point Trajectories

### The Multi-Waypoint Problem

Most real tasks require passing through multiple waypoints (via points) — not just going from A to B, but from A through B, C, D to E. We need a trajectory that:

1. Passes through (or near) each waypoint
2. Maintains smooth velocity (no sudden stops unless requested)
3. Respects joint velocity and acceleration limits

### Linear Segments with Parabolic Blends (LSPB)

LSPB (also called **trapezoidal velocity profile**) is the most common industrial trajectory type:

- **Linear segments**: Constant velocity between waypoints
- **Parabolic blends**: Quadratic position (constant acceleration) at waypoints to smooth transitions

The velocity profile looks like a trapezoid: accelerate, cruise, decelerate.

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

### Multi-Waypoint LSPB

For multiple waypoints, we chain LSPB segments with parabolic blends at each waypoint:

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

## Minimum-Jerk Trajectory

### Motivation

The **jerk** is the third derivative of position (rate of change of acceleration). Minimizing jerk produces the smoothest possible motion — this is important for:
- Reducing mechanical vibrations
- Comfortable human-robot interaction (humans find jerky motion unsettling)
- Precision applications where vibrations degrade accuracy

### Derivation

The minimum-jerk trajectory minimizes:

$$J = \int_0^T \left(\frac{d^3 q}{dt^3}\right)^2 dt$$

subject to boundary conditions on position, velocity, and acceleration at $t = 0$ and $t = T$.

The solution is a **5th-degree polynomial** (quintic) with specific coefficients:

For zero boundary velocities and accelerations:

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

## Time-Optimal Trajectories

### Concept

A **time-optimal trajectory** traverses a path in minimum time while respecting joint velocity, acceleration, and torque limits. This is an optimal control problem:

$$\min_{s(t)} \int_0^{T_f} dt = T_f$$

subject to:
- $\mathbf{q}(t) = \mathbf{q}(s(t))$ follows the geometric path
- $|\dot{q}_i| \leq \dot{q}_{i,max}$
- $|\ddot{q}_i| \leq \ddot{q}_{i,max}$
- $|\tau_i| \leq \tau_{i,max}$

### Phase-Plane Method (Brief Overview)

The time-optimal problem can be solved by analyzing the phase plane $(\dot{s}, s)$ where $s$ is the path parameter. The key insight: at each point on the path, there is a maximum and minimum acceleration determined by the torque limits. The optimal trajectory follows the maximum acceleration curve until it must switch to maximum deceleration to satisfy the final conditions.

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

## Task-Space vs Joint-Space Planning

### Joint-Space Planning

Plan trajectories in joint coordinates $\mathbf{q}(t)$. Each joint follows its own polynomial/LSPB independently.

**Advantages**:
- Simple and fast computation
- Naturally respects joint limits
- No singularity issues
- Each joint can be planned independently

**Disadvantages**:
- End-effector path in workspace is not intuitive (curved, possibly unexpected)
- Cannot guarantee straight-line motion of the tool

### Task-Space Planning

Plan trajectories in task coordinates $\mathbf{x}(t) = (x, y, z, \phi, \theta, \psi)^T$ and use IK at each time step.

**Advantages**:
- Direct control of end-effector path (e.g., straight lines, arcs)
- Intuitive for tasks defined in workspace (welding seams, painting)

**Disadvantages**:
- Requires IK at each time step (computational cost)
- May encounter singularities
- May violate joint limits
- IK may have no solution (path passes outside workspace)

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

### Choosing Between Task-Space and Joint-Space

| Criterion | Joint-Space | Task-Space |
|-----------|-------------|------------|
| Simple repositioning | Preferred | Unnecessary |
| Straight-line tool path needed | Not suitable | Required |
| Near singularities | Safe | Dangerous |
| Near workspace boundaries | Safe | May fail |
| Computational cost | Low | Higher (IK per step) |
| Intuitive end-effector motion | No | Yes |

---

## Trajectory Tracking and Feedforward Control

### The Tracking Problem

Once we have a trajectory $\mathbf{q}_d(t)$, $\dot{\mathbf{q}}_d(t)$, $\ddot{\mathbf{q}}_d(t)$, we need the controller to follow it accurately. From Lesson 6, the **computed torque** controller does this:

$$\boldsymbol{\tau} = M(\mathbf{q})\left[\ddot{\mathbf{q}}_d + K_d(\dot{\mathbf{q}}_d - \dot{\mathbf{q}}) + K_p(\mathbf{q}_d - \mathbf{q})\right] + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

The key insight: the trajectory provides $\ddot{\mathbf{q}}_d$ as **feedforward**, which dramatically improves tracking compared to feedback-only control.

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

## Multi-Joint Synchronization

When planning trajectories for multiple joints simultaneously, all joints must start and finish at the same time. The solution: find the slowest joint and stretch all other joints to match.

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

## Summary

- A **trajectory** adds time parameterization to a geometric path, specifying position, velocity, and acceleration at each instant
- **Cubic polynomials** satisfy position and velocity boundary conditions ($C^1$ smooth)
- **Quintic polynomials** also satisfy acceleration boundary conditions ($C^2$ smooth), preferred for industrial applications
- **LSPB** (trapezoidal velocity) is the industrial standard for its simplicity and direct enforcement of velocity/acceleration limits
- **Minimum-jerk** trajectories produce the smoothest motion, matching natural human arm movements
- **Time-optimal** trajectories minimize execution time under actuator constraints
- **Joint-space** planning is simpler and singularity-free; **task-space** planning provides intuitive end-effector paths
- **Feedforward control** using the desired trajectory's acceleration dramatically improves tracking accuracy over feedback-only approaches
- **Multi-joint synchronization** ensures all joints finish simultaneously for predictable end-effector motion

---

## Exercises

### Exercise 1: Polynomial Trajectories

For a single joint moving from $q_0 = 0°$ to $q_f = 120°$ in $T = 3$ seconds:
1. Compute cubic and quintic trajectories (zero boundary velocities/accelerations)
2. Evaluate and plot position, velocity, acceleration, and jerk for both
3. Compare the maximum velocity and acceleration. Which is larger?
4. If the maximum velocity limit is 60 deg/s, does either trajectory violate it? If so, what minimum $T$ is needed?

### Exercise 2: LSPB Design

Design an LSPB trajectory for a joint with:
- $q_0 = -45°$, $q_f = 90°$
- $v_{max} = 100°/s$
- $a_{max} = 200°/s^2$

1. Is the profile trapezoidal or triangular?
2. What is the total duration?
3. What is the cruise velocity?
4. Now specify $T = 3$ seconds (longer than minimum). What is the new cruise velocity?

### Exercise 3: Multi-Waypoint Trajectory

A 2-DOF robot must visit waypoints:
- $q_A = (0°, 0°)$ at $t = 0$ s
- $q_B = (60°, 30°)$ at $t = 1.5$ s
- $q_C = (30°, 90°)$ at $t = 3.0$ s
- $q_D = (0°, 0°)$ at $t = 5.0$ s

1. Plan using quintic polynomials for each segment (with zero velocity at each waypoint)
2. Plan using LSPB for each segment
3. Compare smoothness: plot acceleration profiles for both approaches
4. What are the velocity discontinuities (if any) at the waypoints?

### Exercise 4: Minimum-Jerk vs Quintic

1. Implement both minimum-jerk and quintic trajectories for the same motion
2. Compute the integrated jerk squared ($\int \dddot{q}^2 dt$) for both
3. Verify that the minimum-jerk trajectory has lower integrated jerk
4. Plot the jerk profiles side by side

### Exercise 5: Trajectory Tracking Simulation

Using the 2-link arm dynamics from Lesson 6:
1. Generate a quintic trajectory from $q = (0, 0)$ to $q = (90°, 45°)$ in 2 seconds
2. Implement computed torque control with $K_p = 100 I$, $K_d = 20 I$
3. Simulate tracking and plot the error over time
4. Introduce a 10% error in the mass parameters (model mismatch) and observe the effect
5. Double the control gains. Does this improve or worsen tracking with model mismatch?

---

[← Previous: Motion Planning](07_Motion_Planning.md) | [Next: Robot Control →](09_Robot_Control.md)
