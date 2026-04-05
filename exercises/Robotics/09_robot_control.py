"""
Exercises for Lesson 09: Robot Control
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: PID Tuning Experiment
    Single-joint robot (pendulum with motor) under gravity.
    """
    m = 2.0      # kg
    l = 0.5      # m
    lc = 0.25    # m
    I = m * l**2 / 12 + m * lc**2  # parallel axis
    g = 9.81
    dt = 0.001
    T = 5.0
    steps = int(T / dt)
    q_d = np.radians(45)

    def simulate_pid(Kp, Kd, Ki, gravity_comp=False):
        q, dq = 0.0, 0.0
        integral = 0.0
        errors = []
        for step in range(steps):
            e = q_d - q
            integral += e * dt
            de = -dq

            tau = Kp * e + Kd * de + Ki * integral
            if gravity_comp:
                tau += m * g * lc * np.sin(q)

            qdd = (tau - m * g * lc * np.sin(q)) / I
            dq += qdd * dt
            q += dq * dt
            errors.append(np.degrees(abs(e)))

        return errors[-1], max(errors)

    print("PID Tuning: Single-joint pendulum, target = 45°")
    print(f"  Kp=50, Kd=10, varying Ki\n")

    # Without gravity compensation
    print("Without gravity compensation:")
    print(f"  {'Ki':>6} | {'SS error (°)':>12} | {'Max error (°)':>13}")
    print("  " + "-" * 40)
    for Ki in [0, 1, 5, 10, 20]:
        ss, mx = simulate_pid(50, 10, Ki, gravity_comp=False)
        print(f"  {Ki:>6} | {ss:>12.4f} | {mx:>13.2f}")

    # With gravity compensation
    print("\nWith gravity compensation:")
    print(f"  {'Ki':>6} | {'SS error (°)':>12} | {'Max error (°)':>13}")
    print("  " + "-" * 40)
    for Ki in [0, 1, 5, 10, 20]:
        ss, mx = simulate_pid(50, 10, Ki, gravity_comp=True)
        print(f"  {Ki:>6} | {ss:>12.4f} | {mx:>13.2f}")

    print("\nExplanation:")
    print("  Without gravity compensation, there is a steady-state error because")
    print("  gravity acts as a constant disturbance. Increasing Ki reduces this")
    print("  error but can cause oscillations (integral windup).")
    print("  With gravity compensation, the feedforward term cancels gravity,")
    print("  and Ki becomes unnecessary — PD alone achieves zero SS error.")


def exercise_2():
    """
    Exercise 2: Computed Torque vs PID
    Fast sinusoidal trajectory (omega=5 rad/s) for 2-link arm.
    """
    m1, m2 = 3.0, 2.0
    l1, l2 = 0.5, 0.4
    lc1, lc2 = 0.25, 0.2
    I1 = m1 * l1**2 / 12
    I2 = m2 * l2**2 / 12
    g = 9.81
    dt = 0.001
    T = 4.0
    omega = 5.0
    steps = int(T / dt)

    Kp = np.diag([100.0, 100.0])
    Kd = np.diag([20.0, 20.0])

    def M_matrix(q):
        c2 = np.cos(q[1])
        M11 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2)
        M12 = I2 + m2*(lc2**2 + l1*lc2*c2)
        M22 = I2 + m2*lc2**2
        return np.array([[M11, M12], [M12, M22]])

    def gravity_vec(q):
        return np.array([
            (m1*lc1 + m2*l1)*g*np.cos(q[0]) + m2*lc2*g*np.cos(q[0]+q[1]),
            m2*lc2*g*np.cos(q[0]+q[1])
        ])

    def coriolis_vec(q, dq):
        h = m2 * l1 * lc2 * np.sin(q[1])
        return np.array([-h*dq[1]*(2*dq[0]+dq[1]), h*dq[0]**2])

    def dynamics(q, dq, tau):
        M = M_matrix(q)
        return np.linalg.solve(M, tau - coriolis_vec(q, dq) - gravity_vec(q))

    def desired(t):
        q_d = np.radians(30) * np.sin(omega * t) * np.array([1.0, 0.5])
        dq_d = np.radians(30) * omega * np.cos(omega * t) * np.array([1.0, 0.5])
        ddq_d = -np.radians(30) * omega**2 * np.sin(omega * t) * np.array([1.0, 0.5])
        return q_d, dq_d, ddq_d

    # PD + gravity
    q, dq = np.array([0.0, 0.0]), np.array([0.0, 0.0])
    errors_pd = []
    for step in range(steps):
        t = step * dt
        q_d, dq_d, _ = desired(t)
        e = q_d - q
        de = dq_d - dq
        tau = Kp @ e + Kd @ de + gravity_vec(q)
        qdd = dynamics(q, dq, tau)
        dq += qdd * dt
        q += dq * dt
        errors_pd.append(np.degrees(np.linalg.norm(e)))

    # Computed torque
    q, dq = np.array([0.0, 0.0]), np.array([0.0, 0.0])
    errors_ct = []
    for step in range(steps):
        t = step * dt
        q_d, dq_d, ddq_d = desired(t)
        e = q_d - q
        de = dq_d - dq
        qdd_des = ddq_d + Kp @ e + Kd @ de
        tau = M_matrix(q) @ qdd_des + coriolis_vec(q, dq) + gravity_vec(q)
        qdd = dynamics(q, dq, tau)
        dq += qdd * dt
        q += dq * dt
        errors_ct.append(np.degrees(np.linalg.norm(e)))

    print(f"Sinusoidal trajectory tracking (omega={omega} rad/s)")
    print(f"\n  {'Controller':>20} | {'Max error (°)':>13} | {'RMS error (°)':>13}")
    print("  " + "-" * 55)
    print(f"  {'PD + gravity':>20} | {max(errors_pd):>13.4f} | "
          f"{np.sqrt(np.mean(np.array(errors_pd)**2)):>13.4f}")
    print(f"  {'Computed torque':>20} | {max(errors_ct):>13.4f} | "
          f"{np.sqrt(np.mean(np.array(errors_ct)**2)):>13.4f}")
    print(f"\n  The performance gap increases with speed because at high speeds,")
    print(f"  Coriolis/centrifugal forces become significant. PD+gravity only")
    print(f"  compensates for static gravity, while computed torque compensates")
    print(f"  for all nonlinear dynamic terms (M, C, G).")


def exercise_3():
    """
    Exercise 3: Impedance Control Design for surface polishing.
    """
    print("Impedance control for surface polishing")
    print("  Surface normal: z-axis")
    print("  Tangential: x, y axes")

    # Desired impedance parameters
    Kd_x, Kd_y = 50.0, 50.0       # Low stiffness tangential (compliant)
    Kd_z = 500.0                    # High stiffness normal (maintain force)
    Bd_x, Bd_y = 10.0, 10.0        # Moderate damping tangential
    Bd_z = 50.0                     # High damping normal

    F_desired_z = 10.0  # N (desired contact force)

    print(f"\n  Impedance parameters:")
    print(f"    x (tangential): Kd={Kd_x} N/m, Bd={Bd_x} N*s/m")
    print(f"    y (tangential): Kd={Kd_y} N/m, Bd={Bd_y} N*s/m")
    print(f"    z (normal):     Kd={Kd_z} N/m, Bd={Bd_z} N*s/m")
    print(f"    Desired normal force: {F_desired_z} N")

    # Simulate: EE follows circular path in x-y while maintaining z-force
    dt = 0.001
    T = 3.0
    steps = int(T / dt)
    radius = 0.05  # 5cm circle
    omega = 2 * np.pi / T  # one full circle

    # Simple simulation
    x, y, z = 0.0, 0.0, 0.0  # position relative to desired
    dx, dy, dz = 0.0, 0.0, 0.0
    surface_z = -0.001  # surface slightly below desired position

    max_force_z = 0
    for step in range(steps):
        t = step * dt
        # Desired path (circle in x-y, constant z)
        x_d = radius * np.cos(omega * t)
        y_d = radius * np.sin(omega * t)
        z_d = surface_z + F_desired_z / Kd_z  # offset to achieve desired force

        # Position error
        ex, ey, ez = x_d - x, y_d - y, z_d - z

        # Impedance control force
        Fx = Kd_x * ex + Bd_x * (-dx)
        Fy = Kd_y * ey + Bd_y * (-dy)
        Fz = Kd_z * ez + Bd_z * (-dz)

        # Environment reaction (simple spring surface at z=surface_z)
        Fe_z = max(0, -5000 * (z - surface_z)) if z < surface_z else 0

        # Simple mass model
        mass = 1.0
        ddx = (Fx) / mass
        ddy = (Fy) / mass
        ddz = (Fz - Fe_z) / mass

        dx += ddx * dt
        dy += ddy * dt
        dz += ddz * dt
        x += dx * dt
        y += dy * dt
        z += dz * dt

        contact_force = Fe_z
        max_force_z = max(max_force_z, contact_force)

    print(f"\n  Simulation results (circular polishing path, r={radius*100:.0f}cm):")
    print(f"    Max contact force: {max_force_z:.2f} N")
    print(f"    Design principle: low tangential stiffness allows the robot to")
    print(f"    follow the surface contour, while high normal stiffness maintains")
    print(f"    consistent contact force for uniform polishing pressure.")


def exercise_4():
    """
    Exercise 4: Hybrid Position/Force Controller for peg-in-hole.
    """
    print("Hybrid position/force control: peg-in-hole insertion")
    print("  x, y: position control (center the peg)")
    print("  z: force control (maintain 5N insertion force)")

    F_insert = 5.0  # N desired insertion force
    Kp_xy = 500.0   # position gain for x, y
    Kd_xy = 50.0    # damping for x, y
    Ki_f = 10.0     # force integral gain for z
    Kp_f = 1.0      # force proportional gain for z

    dt = 0.001
    T = 3.0
    steps = int(T / dt)

    # State
    x, y, z = 0.002, -0.001, 0.0  # small initial offset
    dx, dy, dz = 0.0, 0.0, 0.0
    force_integral = 0.0
    mass = 0.5

    z_positions = []
    forces = []

    for step in range(steps):
        # Position control (x, y) — target is (0, 0)
        Fx = Kp_xy * (0 - x) + Kd_xy * (0 - dx)
        Fy = Kp_xy * (0 - y) + Kd_xy * (0 - dy)

        # Friction model: opposing z-motion
        friction = -2.0 * dz if abs(dz) > 0 else 0

        # Contact force (spring-like resistance in z-direction once z > 0.01)
        F_contact = 50.0 * max(0, z - 0.01) if z > 0.01 else 0

        # Force control (z) — target force = F_insert
        F_measured = F_contact + friction  # simulated force measurement
        force_error = F_insert - F_measured
        force_integral += force_error * dt
        Fz = Kp_f * force_error + Ki_f * force_integral

        # Dynamics
        ddx = Fx / mass
        ddy = Fy / mass
        ddz = (Fz - F_contact - friction) / mass

        dx += ddx * dt
        dy += ddy * dt
        dz += ddz * dt
        x += dx * dt
        y += dy * dt
        z += dz * dt

        z_positions.append(z * 1000)  # mm
        forces.append(F_measured)

    print(f"\n  Simulation results:")
    print(f"    Initial offset: x={2:.1f}mm, y={-1:.1f}mm")
    print(f"    Final x position: {x*1000:.4f} mm (target: 0)")
    print(f"    Final y position: {y*1000:.4f} mm (target: 0)")
    print(f"    Final z depth: {z*1000:.2f} mm")
    print(f"    Final contact force: {forces[-1]:.2f} N (target: {F_insert} N)")
    print(f"    Position control centers the peg while force control pushes it in.")


def exercise_5():
    """
    Exercise 5: Sliding Mode Controller with/without boundary layer.
    """
    # Single joint pendulum
    m = 2.0
    lc = 0.25
    I = 0.02 + m * lc**2
    g = 9.81
    dt = 0.001
    T = 3.0
    steps = int(T / dt)
    q_d = np.radians(45)
    lam = 10.0  # sliding surface parameter

    def simulate_smc(phi=None):
        """phi=None: no boundary layer (sign). phi>0: boundary layer (sat)."""
        q, dq = 0.0, 0.0
        torques = []
        errors = []
        K = 5.0  # switching gain

        for step in range(steps):
            e = q_d - q
            de = -dq
            s = de + lam * e  # sliding surface

            if phi is None:
                # Pure sliding mode (sign function)
                u_switch = K * np.sign(s)
            else:
                # Boundary layer (saturation function)
                u_switch = K * np.clip(s / phi, -1, 1)

            # Equivalent control + switching
            tau_eq = I * (lam * de) + m * g * lc * np.sin(q)
            tau = tau_eq + u_switch

            qdd = (tau - m * g * lc * np.sin(q)) / I
            dq += qdd * dt
            q += dq * dt

            torques.append(tau)
            errors.append(np.degrees(abs(e)))

        return torques, errors

    # Without boundary layer
    tau_sign, err_sign = simulate_smc(phi=None)
    # With boundary layer
    tau_sat_01, err_sat_01 = simulate_smc(phi=0.1)
    tau_sat_05, err_sat_05 = simulate_smc(phi=0.5)
    tau_sat_10, err_sat_10 = simulate_smc(phi=1.0)

    print("Sliding Mode Controller: chattering analysis")
    print(f"  lambda={lam}, K=5.0\n")

    def analyze_chattering(torques, label):
        tau = np.array(torques)
        # Chattering: high-frequency oscillation in control signal
        dtau = np.diff(tau)
        sign_changes = np.sum(np.diff(np.sign(dtau)) != 0)
        amplitude = np.std(tau[len(tau)//2:])  # steady-state std
        return sign_changes, amplitude

    print(f"  {'Method':>20} | {'SS error (°)':>12} | {'Chatter freq':>12} | {'Torque std':>10}")
    print("  " + "-" * 65)

    for label, tau, err in [
        ("sign (no BL)", tau_sign, err_sign),
        ("sat (phi=0.1)", tau_sat_01, err_sat_01),
        ("sat (phi=0.5)", tau_sat_05, err_sat_05),
        ("sat (phi=1.0)", tau_sat_10, err_sat_10),
    ]:
        changes, amplitude = analyze_chattering(tau, label)
        print(f"  {label:>20} | {err[-1]:>12.4f} | {changes:>12d} | {amplitude:>10.4f}")

    print(f"\n  Trade-off: larger phi reduces chattering amplitude and frequency")
    print(f"  but introduces a steady-state error (tracking is not exact within")
    print(f"  the boundary layer). Without the boundary layer, the sign function")
    print(f"  causes high-frequency switching (chattering) that can damage actuators.")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 09: Robot Control — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: PID Tuning Experiment ---")
    exercise_1()

    print("\n--- Exercise 2: Computed Torque vs PID ---")
    exercise_2()

    print("\n--- Exercise 3: Impedance Control Design ---")
    exercise_3()

    print("\n--- Exercise 4: Hybrid Controller ---")
    exercise_4()

    print("\n--- Exercise 5: Sliding Mode Chattering ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
