"""
Exercises for Lesson 08: Trajectory Planning
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np


class CubicTrajectory:
    """Cubic polynomial trajectory with zero boundary velocities."""
    def __init__(self, q0, qf, T):
        self.q0, self.qf, self.T = float(q0), float(qf), float(T)
        # q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # BCs: q(0)=q0, q(T)=qf, dq(0)=0, dq(T)=0
        self.a0 = self.q0
        self.a1 = 0
        self.a2 = 3 * (self.qf - self.q0) / T**2
        self.a3 = -2 * (self.qf - self.q0) / T**3

    def position(self, t):
        t = np.clip(t, 0, self.T)
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3

    def velocity(self, t):
        t = np.clip(t, 0, self.T)
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2

    def acceleration(self, t):
        t = np.clip(t, 0, self.T)
        return 2*self.a2 + 6*self.a3*t

    def jerk(self, t):
        t = np.clip(t, 0, self.T)
        return 6*self.a3 + 0*t  # constant jerk for cubic


class QuinticTrajectory:
    """Quintic polynomial trajectory with zero boundary vel/accel."""
    def __init__(self, q0, qf, T, v0=0, vf=0, a0=0, af=0):
        self.q0, self.qf, self.T = float(q0), float(qf), float(T)
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5
        self.a0 = q0
        self.a1 = v0
        self.a2 = a0 / 2
        M = np.array([[T3, T4, T5],
                       [3*T2, 4*T3, 5*T4],
                       [6*T, 12*T2, 20*T3]])
        b = np.array([qf - q0 - v0*T - (a0/2)*T2,
                       vf - v0 - a0*T,
                       af - a0])
        coeffs = np.linalg.solve(M, b)
        self.a3, self.a4, self.a5 = coeffs

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
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3

    def jerk(self, t):
        t = np.clip(t, 0, self.T)
        return 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2


class LSPBTrajectory:
    """Linear Segment with Parabolic Blend trajectory."""
    def __init__(self, q0, qf, v_max, a_max, T=None):
        self.q0, self.qf = float(q0), float(qf)
        self.v_max, self.a_max = abs(v_max), abs(a_max)
        h = self.qf - self.q0
        self.sign = np.sign(h) if h != 0 else 1
        h_abs = abs(h)
        d_acc = self.v_max**2 / (2 * self.a_max)

        if 2 * d_acc >= h_abs:
            # Triangular profile
            self.v_cruise = np.sqrt(h_abs * self.a_max)
            self.t_acc = self.v_cruise / self.a_max
            self.t_cruise = 0
            self.T = 2 * self.t_acc
            self.is_triangular = True
        else:
            # Trapezoidal profile
            self.v_cruise = self.v_max
            self.t_acc = self.v_cruise / self.a_max
            d_cruise = h_abs - 2 * d_acc
            self.t_cruise = d_cruise / self.v_cruise
            self.T = self.t_acc + self.t_cruise + self.t_acc
            self.is_triangular = False

        if T is not None and T > self.T:
            self.T = T
            a_ = 1.0 / self.a_max
            b_ = -T
            c_ = h_abs
            disc = b_**2 - 4*a_*c_
            if disc >= 0:
                v = (-b_ - np.sqrt(disc)) / (2*a_)
                self.v_cruise = min(v, self.v_max)
                self.t_acc = self.v_cruise / self.a_max
                self.t_cruise = T - 2 * self.t_acc
                self.is_triangular = (self.t_cruise <= 0)

    def position(self, t):
        t = np.clip(t, 0, self.T)
        s = self.sign
        t1 = self.t_acc
        t2 = self.t_acc + self.t_cruise

        if t <= t1:
            return self.q0 + s * 0.5 * self.a_max * t**2
        elif t <= t2:
            d_acc = 0.5 * self.a_max * t1**2
            return self.q0 + s * (d_acc + self.v_cruise * (t - t1))
        else:
            dt = t - t2
            d_total = (0.5 * self.a_max * t1**2 + self.v_cruise * self.t_cruise
                       + self.v_cruise * dt - 0.5 * self.a_max * dt**2)
            return self.q0 + s * d_total


def exercise_1():
    """
    Exercise 1: Polynomial Trajectories
    q0=0° to qf=120° in T=3 seconds.
    """
    q0 = np.radians(0)
    qf = np.radians(120)
    T = 3.0

    cubic = CubicTrajectory(q0, qf, T)
    quintic = QuinticTrajectory(q0, qf, T)

    t_samples = np.linspace(0, T, 100)

    # Evaluate profiles
    print(f"Cubic vs Quintic: 0° → 120° in {T}s")
    print(f"\n{'t':>5} | {'Cubic':>40} | {'Quintic':>40}")
    print(f"{'':>5} | {'pos(°)':>8} {'vel(°/s)':>10} {'acc(°/s²)':>10} {'jerk':>10} | "
          f"{'pos(°)':>8} {'vel(°/s)':>10} {'acc(°/s²)':>10} {'jerk':>10}")
    print("-" * 105)
    for t in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        pc = np.degrees(cubic.position(t))
        vc = np.degrees(cubic.velocity(t))
        ac = np.degrees(cubic.acceleration(t))
        jc = np.degrees(cubic.jerk(t))
        pq = np.degrees(quintic.position(t))
        vq = np.degrees(quintic.velocity(t))
        aq = np.degrees(quintic.acceleration(t))
        jq = np.degrees(quintic.jerk(t))
        print(f"{t:5.1f} | {pc:8.2f} {vc:10.2f} {ac:10.2f} {jc:10.2f} | "
              f"{pq:8.2f} {vq:10.2f} {aq:10.2f} {jq:10.2f}")

    # Maximum velocity and acceleration
    v_cubic = [abs(np.degrees(cubic.velocity(t))) for t in t_samples]
    v_quintic = [abs(np.degrees(quintic.velocity(t))) for t in t_samples]
    a_cubic = [abs(np.degrees(cubic.acceleration(t))) for t in t_samples]
    a_quintic = [abs(np.degrees(quintic.acceleration(t))) for t in t_samples]

    print(f"\nMax velocity:     cubic={max(v_cubic):.2f}°/s, quintic={max(v_quintic):.2f}°/s")
    print(f"Max acceleration: cubic={max(a_cubic):.2f}°/s², quintic={max(a_quintic):.2f}°/s²")
    print(f"Quintic has higher peak velocity (to compensate for smoother transitions)")

    # Velocity limit check
    v_limit = 60.0  # deg/s
    cubic_violates = max(v_cubic) > v_limit
    quintic_violates = max(v_quintic) > v_limit
    print(f"\nVelocity limit = {v_limit}°/s")
    print(f"  Cubic violates: {cubic_violates} (max={max(v_cubic):.2f})")
    print(f"  Quintic violates: {quintic_violates} (max={max(v_quintic):.2f})")

    if quintic_violates:
        # Find minimum T for quintic: max vel ≈ 15/8 * (qf-q0)/T for quintic
        # Exact: v_max_quintic = (15/8)*(qf-q0)/T
        T_min_quintic = (15.0 / 8.0) * np.degrees(abs(qf - q0)) / v_limit
        print(f"  Minimum T for quintic: {T_min_quintic:.2f} s")
    if cubic_violates:
        # Cubic: v_max = (3/2)*(qf-q0)/T
        T_min_cubic = (3.0 / 2.0) * np.degrees(abs(qf - q0)) / v_limit
        print(f"  Minimum T for cubic: {T_min_cubic:.2f} s")


def exercise_2():
    """
    Exercise 2: LSPB Design
    q0=-45°, qf=90°, v_max=100°/s, a_max=200°/s².
    """
    q0 = np.radians(-45)
    qf = np.radians(90)
    v_max = np.radians(100)
    a_max = np.radians(200)

    lspb = LSPBTrajectory(q0, qf, v_max, a_max)

    profile_type = "triangular" if lspb.is_triangular else "trapezoidal"
    print(f"LSPB: {-45}° → {90}°, v_max={100}°/s, a_max={200}°/s²")
    print(f"\n1. Profile type: {profile_type}")
    print(f"2. Total duration: {lspb.T:.4f} s")
    print(f"3. Cruise velocity: {np.degrees(lspb.v_cruise):.2f}°/s")
    print(f"   Acceleration time: {lspb.t_acc:.4f} s")
    print(f"   Cruise time: {lspb.t_cruise:.4f} s")

    # With T = 3 seconds
    lspb3 = LSPBTrajectory(q0, qf, v_max, a_max, T=3.0)
    print(f"\n4. With T = 3.0 s (longer than minimum):")
    print(f"   New cruise velocity: {np.degrees(lspb3.v_cruise):.2f}°/s")
    print(f"   New acceleration time: {lspb3.t_acc:.4f} s")
    print(f"   New cruise time: {lspb3.t_cruise:.4f} s")

    # Verify endpoints
    print(f"\n   Verification: q(0) = {np.degrees(lspb3.position(0)):.2f}°, "
          f"q(3) = {np.degrees(lspb3.position(3.0)):.2f}°")


def exercise_3():
    """
    Exercise 3: Multi-Waypoint Trajectory (2-DOF)
    Waypoints: A=(0,0), B=(60,30), C=(30,90), D=(0,0).
    """
    waypoints = [
        (np.radians([0, 0]), 0.0),
        (np.radians([60, 30]), 1.5),
        (np.radians([30, 90]), 3.0),
        (np.radians([0, 0]), 5.0),
    ]

    print("Multi-waypoint trajectory (2-DOF)")
    print("  Waypoints: A=(0,0)° B=(60,30)° C=(30,90)° D=(0,0)°")
    print("  Times:     0s       1.5s        3.0s        5.0s")

    # Quintic segments with zero velocity at waypoints
    print("\n--- Quintic segments (zero velocity at each waypoint) ---")
    for seg in range(len(waypoints) - 1):
        q_start = waypoints[seg][0]
        q_end = waypoints[seg + 1][0]
        t_start = waypoints[seg][1]
        t_end = waypoints[seg + 1][1]
        T = t_end - t_start

        for joint in range(2):
            traj = QuinticTrajectory(q_start[joint], q_end[joint], T)
            mid = T / 2
            print(f"  Segment {seg + 1}, Joint {joint + 1}: "
                  f"{np.degrees(q_start[joint]):.0f}° → {np.degrees(q_end[joint]):.0f}° "
                  f"in {T:.1f}s, "
                  f"max|v| = {abs(np.degrees(traj.velocity(mid))):.1f}°/s, "
                  f"max|a| = {max(abs(np.degrees(traj.acceleration(0))), abs(np.degrees(traj.acceleration(T)))):.1f}°/s²")

    # LSPB segments
    print("\n--- LSPB segments ---")
    v_max_lspb = np.radians(120)
    a_max_lspb = np.radians(300)
    for seg in range(len(waypoints) - 1):
        q_start = waypoints[seg][0]
        q_end = waypoints[seg + 1][0]
        t_start = waypoints[seg][1]
        t_end = waypoints[seg + 1][1]
        T = t_end - t_start

        for joint in range(2):
            lspb = LSPBTrajectory(q_start[joint], q_end[joint], v_max_lspb, a_max_lspb, T)
            profile = "trapezoidal" if not lspb.is_triangular else "triangular"
            print(f"  Segment {seg + 1}, Joint {joint + 1}: {profile}, "
                  f"v_cruise={np.degrees(lspb.v_cruise):.1f}°/s")

    # Velocity discontinuities
    print("\n  Velocity discontinuities at waypoints:")
    print("  With zero velocity at each waypoint (quintic): NO discontinuities")
    print("  With LSPB (segment-by-segment): velocity is zero at start/end of each")
    print("  segment, so no discontinuities, but the motion pauses at each waypoint.")
    print("  For continuous through-motion, blending between segments is needed.")


def exercise_4():
    """
    Exercise 4: Minimum-Jerk vs Quintic
    """
    q0, qf = np.radians(0), np.radians(90)
    T = 2.0

    # Minimum-jerk: same polynomial form as quintic with zero BCs
    # For zero boundary v, a: minimum-jerk IS the quintic with zero BCs
    # But let's show this explicitly
    quintic = QuinticTrajectory(q0, qf, T)

    # Minimum-jerk with explicit formula:
    # q(t) = q0 + (qf-q0) * [10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5]
    h = qf - q0

    def min_jerk_pos(t):
        s = t / T
        return q0 + h * (10*s**3 - 15*s**4 + 6*s**5)

    def min_jerk_jerk(t):
        s = t / T
        return h * (60/T**3 - 360*s/T**3 + 360*s**2/T**3)

    # Compare
    t_samples = np.linspace(0, T, 1000)
    dt = t_samples[1] - t_samples[0]

    # Integrated jerk squared
    jerk_sq_quintic = sum(quintic.jerk(t)**2 * dt for t in t_samples)
    jerk_sq_mj = sum(min_jerk_jerk(t)**2 * dt for t in t_samples)

    print(f"Minimum-Jerk vs Quintic: 0° → 90° in {T}s")
    print(f"\nNote: For zero boundary velocities and accelerations,")
    print(f"the minimum-jerk trajectory IS the quintic polynomial.")
    print(f"They produce identical trajectories.")

    # Verify
    max_diff = max(abs(quintic.position(t) - min_jerk_pos(t)) for t in t_samples)
    print(f"\nMax position difference: {np.degrees(max_diff):.2e}°")

    print(f"\nIntegrated jerk squared (∫ j² dt):")
    print(f"  Quintic:      {jerk_sq_quintic:.6f}")
    print(f"  Minimum-jerk: {jerk_sq_mj:.6f}")

    # Now compare with a different quintic (nonzero boundary velocity)
    quintic_v = QuinticTrajectory(q0, qf, T, v0=np.radians(10), vf=np.radians(-5))
    jerk_sq_qv = sum(quintic_v.jerk(t)**2 * dt for t in t_samples)
    print(f"\n  Quintic with v0=10°/s, vf=-5°/s:")
    print(f"  Integrated jerk squared: {jerk_sq_qv:.6f}")
    print(f"  This is higher than minimum-jerk, as expected.")


def exercise_5():
    """
    Exercise 5: Trajectory Tracking Simulation with 2-link arm.
    """
    # 2-link arm parameters
    m1, m2 = 3.0, 2.0
    l1, l2 = 0.5, 0.4
    lc1, lc2 = 0.25, 0.2
    I1 = m1 * l1**2 / 12
    I2 = m2 * l2**2 / 12
    g = 9.81

    def dynamics(q, dq, tau, m1_actual, m2_actual):
        """Compute acceleration with possibly incorrect model params."""
        I1_a = m1_actual * l1**2 / 12
        I2_a = m2_actual * l2**2 / 12
        c2 = np.cos(q[1])
        s2 = np.sin(q[1])
        M11 = I1_a + I2_a + m1_actual*lc1**2 + m2_actual*(l1**2 + lc2**2 + 2*l1*lc2*c2)
        M12 = I2_a + m2_actual*(lc2**2 + l1*lc2*c2)
        M22 = I2_a + m2_actual*lc2**2
        M = np.array([[M11, M12], [M12, M22]])
        h = m2_actual * l1 * lc2 * s2
        C_dq = np.array([-h*dq[1]*(2*dq[0]+dq[1]), h*dq[0]**2])
        g_vec = np.array([
            (m1_actual*lc1 + m2_actual*l1)*g*np.cos(q[0]) + m2_actual*lc2*g*np.cos(q[0]+q[1]),
            m2_actual*lc2*g*np.cos(q[0]+q[1])
        ])
        return np.linalg.solve(M, tau - C_dq - g_vec)

    def computed_torque_model(q, dq, qdd_des, m1_model, m2_model):
        """Computed torque using model parameters (may differ from actual)."""
        I1_m = m1_model * l1**2 / 12
        I2_m = m2_model * l2**2 / 12
        c2 = np.cos(q[1])
        s2 = np.sin(q[1])
        M11 = I1_m + I2_m + m1_model*lc1**2 + m2_model*(l1**2 + lc2**2 + 2*l1*lc2*c2)
        M12 = I2_m + m2_model*(lc2**2 + l1*lc2*c2)
        M22 = I2_m + m2_model*lc2**2
        M = np.array([[M11, M12], [M12, M22]])
        h = m2_model * l1 * lc2 * s2
        C_dq = np.array([-h*dq[1]*(2*dq[0]+dq[1]), h*dq[0]**2])
        g_vec = np.array([
            (m1_model*lc1 + m2_model*l1)*g*np.cos(q[0]) + m2_model*lc2*g*np.cos(q[0]+q[1]),
            m2_model*lc2*g*np.cos(q[0]+q[1])
        ])
        return M @ qdd_des + C_dq + g_vec

    # Generate quintic trajectory
    T = 2.0
    dt = 0.001
    steps = int(T / dt)

    traj = [QuinticTrajectory(0, np.radians(90), T),
            QuinticTrajectory(0, np.radians(45), T)]

    def simulate(Kp, Kd, m1_model, m2_model, m1_actual, m2_actual, label):
        q = np.array([0.0, 0.0])
        dq = np.array([0.0, 0.0])
        max_err = 0.0

        for step in range(steps):
            t = step * dt
            q_d = np.array([traj[0].position(t), traj[1].position(t)])
            dq_d = np.array([traj[0].velocity(t), traj[1].velocity(t)])
            ddq_d = np.array([traj[0].acceleration(t), traj[1].acceleration(t)])

            e = q_d - q
            de = dq_d - dq
            qdd_des = ddq_d + Kp * e + Kd * de
            tau = computed_torque_model(q, dq, qdd_des, m1_model, m2_model)
            qdd = dynamics(q, dq, tau, m1_actual, m2_actual)
            dq = dq + qdd * dt
            q = q + dq * dt

            err = np.degrees(np.linalg.norm(q_d - q))
            max_err = max(max_err, err)

        final_err = np.degrees(np.linalg.norm(
            np.array([traj[0].position(T), traj[1].position(T)]) - q))
        return max_err, final_err

    Kp, Kd = 100.0, 20.0

    # Case 1: Perfect model
    max_e, final_e = simulate(Kp, Kd, m1, m2, m1, m2, "Perfect model")
    print(f"Trajectory tracking: (0,0) → (90°,45°) in {T}s")
    print(f"  Kp={Kp}, Kd={Kd}")
    print(f"\n1. Perfect model:")
    print(f"   Max error: {max_e:.4f}°, Final error: {final_e:.4f}°")

    # Case 2: 10% mass error
    m1_err, m2_err = m1 * 1.1, m2 * 1.1
    max_e2, final_e2 = simulate(Kp, Kd, m1, m2, m1_err, m2_err, "10% mass error")
    print(f"\n2. With 10% mass error (model uses nominal, plant has +10%):")
    print(f"   Max error: {max_e2:.4f}°, Final error: {final_e2:.4f}°")

    # Case 3: Doubled gains with mismatch
    Kp2, Kd2 = 200.0, 40.0
    max_e3, final_e3 = simulate(Kp2, Kd2, m1, m2, m1_err, m2_err, "Doubled gains + mismatch")
    print(f"\n3. Doubled gains (Kp={Kp2}, Kd={Kd2}) with 10% mass error:")
    print(f"   Max error: {max_e3:.4f}°, Final error: {final_e3:.4f}°")
    print(f"\n   Higher gains reduce tracking error even with model mismatch,")
    print(f"   because the PD terms in the computed torque controller act as")
    print(f"   a robust feedback loop that compensates for model errors.")
    print(f"   However, very high gains can cause oscillations or instability.")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 08: Trajectory Planning — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: Polynomial Trajectories ---")
    exercise_1()

    print("\n--- Exercise 2: LSPB Design ---")
    exercise_2()

    print("\n--- Exercise 3: Multi-Waypoint Trajectory ---")
    exercise_3()

    print("\n--- Exercise 4: Minimum-Jerk vs Quintic ---")
    exercise_4()

    print("\n--- Exercise 5: Trajectory Tracking Simulation ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
