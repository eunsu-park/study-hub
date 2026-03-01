"""
Trajectory Generation for Robotic Manipulators
===============================================
Polynomial and trapezoidal velocity profiles for smooth motion.

A trajectory specifies not just the path (geometric shape) but also the
time history of motion along that path: position q(t), velocity q̇(t),
and acceleration q̈(t). Good trajectories ensure:
  - Smooth motion (continuous velocity, bounded acceleration)
  - Respect actuator limits (max velocity, max acceleration)
  - Minimize vibration (low jerk = d³q/dt³)

Methods implemented:
  1. Cubic polynomial: 4 coefficients, matches position + velocity at endpoints
  2. Quintic polynomial: 6 coefficients, also matches acceleration at endpoints
  3. Minimum-jerk: special quintic that minimizes integral of jerk² (biological motion model)
  4. Linear Segment with Parabolic Blends (LSPB): trapezoidal velocity profile,
     widely used in industrial robotics for its simplicity and predictability
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# ---------------------------------------------------------------------------
# Cubic polynomial trajectory
# ---------------------------------------------------------------------------
def cubic_trajectory(t: np.ndarray, t0: float, tf: float,
                      q0: float, qf: float,
                      v0: float = 0.0, vf: float = 0.0) -> Tuple[np.ndarray, ...]:
    """Generate a cubic polynomial trajectory.

    q(t) = a0 + a1*t + a2*t^2 + a3*t^3

    4 boundary conditions → 4 coefficients:
      q(t0) = q0, q(tf) = qf, q̇(t0) = v0, q̇(tf) = vf

    Why cubic? It is the minimum-degree polynomial that can match position
    and velocity at both endpoints. However, acceleration is NOT zero at
    endpoints, which causes discontinuous jerk if chaining segments.

    Args:
        t: Time array
        t0, tf: Start and final times
        q0, qf: Start and final positions
        v0, vf: Start and final velocities
    """
    T = tf - t0  # Duration
    # Solve the 4x4 linear system for coefficients
    # Using normalized time s = (t - t0) / T for numerical stability
    a0 = q0
    a1 = v0 * T
    a2 = 3 * (qf - q0) - 2 * v0 * T - vf * T
    a3 = -2 * (qf - q0) + v0 * T + vf * T

    s = np.clip((t - t0) / T, 0, 1)  # Normalized time
    pos = a0 + a1 * s + a2 * s**2 + a3 * s**3
    vel = (a1 + 2 * a2 * s + 3 * a3 * s**2) / T
    acc = (2 * a2 + 6 * a3 * s) / T**2

    return pos, vel, acc


# ---------------------------------------------------------------------------
# Quintic polynomial trajectory
# ---------------------------------------------------------------------------
def quintic_trajectory(t: np.ndarray, t0: float, tf: float,
                        q0: float, qf: float,
                        v0: float = 0.0, vf: float = 0.0,
                        a0: float = 0.0, af: float = 0.0) -> Tuple[np.ndarray, ...]:
    """Generate a quintic polynomial trajectory.

    q(t) = c0 + c1*s + c2*s^2 + c3*s^3 + c4*s^4 + c5*s^5

    6 boundary conditions → 6 coefficients:
      q, q̇, q̈ at both endpoints

    Why quintic? By matching accelerations at endpoints, we get continuous
    acceleration → smoother motion than cubic. The extra smoothness reduces
    vibration and mechanical wear. The trade-off is higher peak velocities
    for the same duration.
    """
    T = tf - t0
    s = np.clip((t - t0) / T, 0, 1)

    c0 = q0
    c1 = v0 * T
    c2 = 0.5 * a0 * T**2
    c3 = 10 * (qf - q0) - 6 * v0 * T - 4 * a0 * T**2 / 2 - 4 * vf * T + af * T**2 / 2
    c4 = -15 * (qf - q0) + 8 * v0 * T + 7 * a0 * T**2 / 2 + 7 * vf * T - af * T**2
    c5 = 6 * (qf - q0) - 3 * (v0 + vf) * T - (a0 - af) * T**2 / 2

    pos = c0 + c1*s + c2*s**2 + c3*s**3 + c4*s**4 + c5*s**5
    vel = (c1 + 2*c2*s + 3*c3*s**2 + 4*c4*s**3 + 5*c5*s**4) / T
    acc = (2*c2 + 6*c3*s + 12*c4*s**2 + 20*c5*s**3) / T**2

    return pos, vel, acc


# ---------------------------------------------------------------------------
# Minimum-jerk trajectory
# ---------------------------------------------------------------------------
def minimum_jerk_trajectory(t: np.ndarray, t0: float, tf: float,
                              q0: float, qf: float) -> Tuple[np.ndarray, ...]:
    """Generate a minimum-jerk trajectory.

    This is a special case of the quintic polynomial where the trajectory
    minimizes the integral of jerk squared:  min integral(q̈̇^2 dt)

    The result is: q(t) = q0 + (qf - q0) * (10s^3 - 15s^4 + 6s^5)

    Why minimize jerk?
      - Flash & Hogan (1985) showed human arm movements follow this profile
      - Produces the smoothest possible motion for a given duration
      - Zero velocity, acceleration, AND jerk at endpoints
      - Ideal for precision tasks where vibration must be minimized
    """
    T = tf - t0
    s = np.clip((t - t0) / T, 0, 1)
    delta = qf - q0

    # The famous minimum-jerk formula
    pos = q0 + delta * (10 * s**3 - 15 * s**4 + 6 * s**5)
    vel = delta * (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    acc = delta * (60 * s - 180 * s**2 + 120 * s**3) / T**2

    return pos, vel, acc


# ---------------------------------------------------------------------------
# LSPB (Linear Segment with Parabolic Blends) — Trapezoidal velocity
# ---------------------------------------------------------------------------
def lspb_trajectory(t: np.ndarray, t0: float, tf: float,
                     q0: float, qf: float,
                     v_max: float = None) -> Tuple[np.ndarray, ...]:
    """Generate an LSPB (trapezoidal velocity) trajectory.

    The velocity profile has three phases:
      1. Acceleration (parabolic position) — ramp up
      2. Cruise (linear position) — constant velocity
      3. Deceleration (parabolic position) — ramp down

    Why LSPB in industry?
      - Simple and predictable: easy to set velocity and acceleration limits
      - Maximum speed is reached quickly (time-efficient)
      - Widely used in CNC machines, industrial robots, 3D printers

    Drawback: acceleration has discontinuities (step changes at phase transitions),
    which can cause vibration. S-curve profiles fix this but are more complex.
    """
    T = tf - t0
    delta = qf - q0
    sign = np.sign(delta)

    if v_max is None:
        # Default: use 1.5x the average velocity to ensure a cruise phase exists
        v_max = 1.5 * abs(delta) / T

    v_max = abs(v_max) * sign  # Match direction

    # Blend time: solve for tb such that the area under the trapezoid = delta
    # delta = v_max * (T - tb), where tb = v_max / a, and a = v_max / tb
    # This gives: tb = T - delta / v_max
    tb = T - delta / v_max

    if tb < 0 or tb > T / 2:
        # If blend time is invalid, use triangular profile (no cruise phase)
        tb = T / 2
        v_max = delta / (T / 2)

    a = v_max / tb  # Acceleration magnitude

    pos = np.zeros_like(t)
    vel = np.zeros_like(t)
    acc = np.zeros_like(t)

    for i, ti in enumerate(t):
        s = ti - t0
        if s <= 0:
            pos[i] = q0
        elif s <= tb:
            # Acceleration phase (parabolic)
            pos[i] = q0 + 0.5 * a * s**2
            vel[i] = a * s
            acc[i] = a
        elif s <= T - tb:
            # Cruise phase (linear)
            pos[i] = q0 + 0.5 * a * tb**2 + v_max * (s - tb)
            vel[i] = v_max
            acc[i] = 0
        elif s <= T:
            # Deceleration phase (parabolic)
            s_rem = T - s  # Time remaining
            pos[i] = qf - 0.5 * a * s_rem**2
            vel[i] = a * s_rem
            acc[i] = -a
        else:
            pos[i] = qf

    return pos, vel, acc


# ---------------------------------------------------------------------------
# Multi-segment via-point trajectory
# ---------------------------------------------------------------------------
def via_point_trajectory(t: np.ndarray, via_times: np.ndarray,
                          via_points: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Generate a multi-segment trajectory through via points using cubic splines.

    For each segment, we use cubic polynomials with continuity constraints:
      - Position matches at via points
      - Velocity is continuous at via points (computed from neighbors)
      - Start/end velocities are zero

    This is a simplified version; industrial systems use more sophisticated
    blending (e.g., time-optimal multi-segment with velocity limits).
    """
    n_segments = len(via_points) - 1
    pos = np.zeros_like(t)
    vel = np.zeros_like(t)
    acc = np.zeros_like(t)

    # Estimate velocities at via points using central differences
    vias_vel = np.zeros(len(via_points))
    for i in range(1, len(via_points) - 1):
        # Average slope of neighboring segments
        dt_prev = via_times[i] - via_times[i - 1]
        dt_next = via_times[i + 1] - via_times[i]
        slope_prev = (via_points[i] - via_points[i - 1]) / dt_prev
        slope_next = (via_points[i + 1] - via_points[i]) / dt_next
        vias_vel[i] = (slope_prev + slope_next) / 2

    # Generate each segment
    for seg in range(n_segments):
        mask = (t >= via_times[seg]) & (t < via_times[seg + 1])
        if seg == n_segments - 1:
            mask = (t >= via_times[seg]) & (t <= via_times[seg + 1])

        t_seg = t[mask]
        p, v, a = cubic_trajectory(t_seg, via_times[seg], via_times[seg + 1],
                                     via_points[seg], via_points[seg + 1],
                                     vias_vel[seg], vias_vel[seg + 1])
        pos[mask] = p
        vel[mask] = v
        acc[mask] = a

    return pos, vel, acc


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo_trajectory():
    """Demonstrate and compare different trajectory generation methods."""
    print("=" * 60)
    print("Trajectory Generation Demo")
    print("=" * 60)

    t0, tf = 0.0, 2.0
    q0, qf = 0.0, np.pi / 2  # 90 degree motion
    t = np.linspace(t0, tf, 500)

    # Compute all trajectory types
    p_cubic, v_cubic, a_cubic = cubic_trajectory(t, t0, tf, q0, qf)
    p_quint, v_quint, a_quint = quintic_trajectory(t, t0, tf, q0, qf)
    p_mj, v_mj, a_mj = minimum_jerk_trajectory(t, t0, tf, q0, qf)
    p_lspb, v_lspb, a_lspb = lspb_trajectory(t, t0, tf, q0, qf)

    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    labels = ['Cubic', 'Quintic', 'Min-Jerk', 'LSPB']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    styles = ['-', '--', '-.', ':']

    for data, label, color, style in zip(
        [(p_cubic, v_cubic, a_cubic), (p_quint, v_quint, a_quint),
         (p_mj, v_mj, a_mj), (p_lspb, v_lspb, a_lspb)],
        labels, colors, styles
    ):
        axes[0].plot(t, np.degrees(data[0]), style, color=color, linewidth=2, label=label)
        axes[1].plot(t, data[1], style, color=color, linewidth=2, label=label)
        axes[2].plot(t, data[2], style, color=color, linewidth=2, label=label)

    axes[0].set_ylabel("Position (deg)")
    axes[0].set_title("Trajectory Comparison: Point-to-Point Motion (0 → 90 deg)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Acceleration (rad/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("07_trajectory_comparison.png", dpi=120)
    plt.show()

    # Print key metrics
    print("\n--- Trajectory Metrics ---")
    for name, (_, v, a) in zip(labels, [
        (p_cubic, v_cubic, a_cubic), (p_quint, v_quint, a_quint),
        (p_mj, v_mj, a_mj), (p_lspb, v_lspb, a_lspb)
    ]):
        print(f"  {name:10s}: peak vel = {np.max(np.abs(v)):.3f} rad/s, "
              f"peak acc = {np.max(np.abs(a)):.3f} rad/s²")

    # --- Multi-segment via-point trajectory ---
    print("\n--- Via-Point Trajectory ---")
    via_times = np.array([0.0, 1.0, 2.5, 3.5, 5.0])
    via_points = np.radians(np.array([0, 45, 30, 80, 60]))  # Degrees → radians

    t_via = np.linspace(via_times[0], via_times[-1], 500)
    p_via, v_via, a_via = via_point_trajectory(t_via, via_times, via_points)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t_via, np.degrees(p_via), 'b-', linewidth=2)
    axes[0].plot(via_times, np.degrees(via_points), 'ro', markersize=10, label='Via points')
    axes[0].set_ylabel("Position (deg)")
    axes[0].set_title("Via-Point Trajectory (Cubic Spline Segments)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_via, v_via, 'b-', linewidth=2)
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_via, a_via, 'b-', linewidth=2)
    axes[2].set_ylabel("Acceleration (rad/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    # Draw vertical lines at via points
    for ax in axes:
        for vt in via_times:
            ax.axvline(x=vt, color='gray', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig("07_trajectory_via_points.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    demo_trajectory()
