"""
Exercises for Lesson 06: Robot Dynamics
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Single Link Dynamics (Pendulum)
    m=2kg, l=0.5m, lc=0.25m, I=0.02 kg*m^2.
    """
    m = 2.0     # kg
    l = 0.5     # m (link length)
    lc = 0.25   # m (center of mass)
    I = 0.02    # kg*m^2 (about CoM)
    g = 9.81    # m/s^2

    # Euler-Lagrange: (I + m*lc^2) * theta_dd + m*g*lc*sin(theta) = tau
    I_eff = I + m * lc**2

    print("Single-link pendulum dynamics:")
    print(f"  EoM: {I_eff:.4f} * theta_dd + {m * g * lc:.4f} * sin(theta) = tau")
    print(f"  I_eff = I + m*lc^2 = {I} + {m}*{lc}^2 = {I_eff:.4f} kg*m^2")

    # Gravity torque at various angles
    print(f"\nGravity torque (tau_g = m*g*lc*sin(theta)):")
    for theta_deg in [0, 45, 90]:
        theta = np.radians(theta_deg)
        tau_g = m * g * lc * np.sin(theta)
        print(f"  theta = {theta_deg:3d}°: tau_g = {tau_g:.4f} N*m")

    # Torque to hold at 45°
    tau_hold = m * g * lc * np.sin(np.radians(45))
    print(f"\nTorque to hold at 45°: {tau_hold:.4f} N*m")

    # Simulate free fall from 90° using Euler integration
    print(f"\nFree fall from 90° (Euler integration, dt=0.001s, 1s):")
    dt = 0.001
    theta = np.radians(90)
    omega = 0.0
    t = 0.0

    print(f"  {'t (s)':>6} | {'theta (°)':>10} | {'omega (°/s)':>12}")
    print("  " + "-" * 38)
    for step in range(1001):
        if step % 200 == 0:
            print(f"  {t:6.3f} | {np.degrees(theta):10.2f} | {np.degrees(omega):12.2f}")
        # theta_dd = -m*g*lc*sin(theta) / I_eff
        alpha = -m * g * lc * np.sin(theta) / I_eff
        omega += alpha * dt
        theta += omega * dt
        t += dt


def exercise_2():
    """
    Exercise 2: Inertia Matrix Analysis for 2-link arm.
    """
    # Parameters
    m1, m2 = 3.0, 2.0
    l1, l2 = 0.5, 0.4
    lc1, lc2 = 0.25, 0.2
    I1 = m1 * l1**2 / 12  # rod inertia
    I2 = m2 * l2**2 / 12

    def inertia_matrix(q2):
        """2-link arm inertia matrix M(q)."""
        c2 = np.cos(q2)
        M11 = (I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2))
        M12 = I2 + m2 * (lc2**2 + l1 * lc2 * c2)
        M22 = I2 + m2 * lc2**2
        return np.array([[M11, M12], [M12, M22]])

    print("2-link arm inertia matrix analysis")
    print(f"  m1={m1}, m2={m2}, l1={l1}, l2={l2}")
    print(f"  lc1={lc1}, lc2={lc2}")

    print(f"\n{'q2 (°)':>8} | {'M11':>8} {'M12':>8} {'M22':>8} | {'eig1':>8} {'eig2':>8}")
    print("-" * 65)
    for q2_deg in [0, 90, 180]:
        q2 = np.radians(q2_deg)
        M = inertia_matrix(q2)
        eigvals = np.linalg.eigvalsh(M)
        print(f"{q2_deg:>7}° | {M[0, 0]:>8.4f} {M[0, 1]:>8.4f} {M[1, 1]:>8.4f} | "
              f"{eigvals[0]:>8.4f} {eigvals[1]:>8.4f}")

    # M11 max/min
    # M11 = const + 2*m2*l1*lc2*cos(q2)
    # Max when cos(q2) = 1 (q2=0), min when cos(q2) = -1 (q2=180)
    M_0 = inertia_matrix(0)
    M_pi = inertia_matrix(np.pi)
    print(f"\nM11 is maximized at q2=0° ({M_0[0, 0]:.4f}): arm fully extended")
    print(f"  (maximum effective inertia at joint 1)")
    print(f"M11 is minimized at q2=180° ({M_pi[0, 0]:.4f}): arm fully folded")
    print(f"  (link 2 CoM closest to joint 1 axis)")

    # Verify positive definiteness at random configs
    np.random.seed(42)
    n_test = 100
    all_pd = True
    for _ in range(n_test):
        q2_rand = np.random.uniform(-np.pi, np.pi)
        M = inertia_matrix(q2_rand)
        eigvals = np.linalg.eigvalsh(M)
        if np.any(eigvals <= 0):
            all_pd = False
            break
    print(f"\nPositive definiteness check ({n_test} random configs): "
          f"{'ALL positive definite' if all_pd else 'FAILED'}")


def exercise_3():
    """
    Exercise 3: Coriolis Effects
    """
    m1, m2 = 3.0, 2.0
    l1, l2 = 0.5, 0.4
    lc1, lc2 = 0.25, 0.2
    I1 = m1 * l1**2 / 12
    I2 = m2 * l2**2 / 12

    def coriolis_matrix(q, dq):
        """Coriolis/centrifugal matrix C(q, dq)."""
        q2 = q[1]
        dq1, dq2 = dq
        h = m2 * l1 * lc2 * np.sin(q2)
        C = np.array([
            [-h * dq2, -h * (dq1 + dq2)],
            [h * dq1, 0]
        ])
        return C

    def inertia_matrix(q):
        c2 = np.cos(q[1])
        M11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2)
        M12 = I2 + m2 * (lc2**2 + l1 * lc2 * c2)
        M22 = I2 + m2 * lc2**2
        return np.array([[M11, M12], [M12, M22]])

    q = np.radians([45.0, 60.0])
    dq = np.array([2.0, 0.0])  # only joint 1 moving

    C = coriolis_matrix(q, dq)
    tau_coriolis = C @ dq

    print(f"Configuration: q = (45°, 60°), dq = (2, 0) rad/s")
    print(f"\n1. Coriolis matrix C(q, dq):")
    print(f"   {C.round(6)}")
    print(f"\n   Coriolis torque on joint 2: tau_c2 = {tau_coriolis[1]:.6f} N*m")

    print(f"\n2. Physical explanation:")
    print(f"   When joint 1 rotates, link 2 experiences a centrifugal-like effect.")
    print(f"   The rotation of joint 1 creates a 'virtual force' on link 2's CoM")
    print(f"   that tends to straighten the arm (push link 2 outward). This creates")
    print(f"   a torque about joint 2 even though joint 2 is not being driven.")

    # Verify skew-symmetry: N = M_dot - 2C should be skew-symmetric
    print(f"\n3. Skew-symmetry verification (N = dM/dt - 2C):")
    delta = 1e-7
    configs = [
        (np.radians([30, 45]), np.array([1.0, 0.5])),
        (np.radians([60, -30]), np.array([-0.5, 2.0])),
        (np.radians([0, 90]), np.array([1.5, -1.0])),
    ]

    for q_test, dq_test in configs:
        M = inertia_matrix(q_test)
        C = coriolis_matrix(q_test, dq_test)

        # M_dot via finite difference: dM/dt = dM/dq * dq/dt
        M_plus1 = inertia_matrix(q_test + np.array([delta, 0]))
        M_plus2 = inertia_matrix(q_test + np.array([0, delta]))
        dM_dq1 = (M_plus1 - M) / delta
        dM_dq2 = (M_plus2 - M) / delta
        M_dot = dM_dq1 * dq_test[0] + dM_dq2 * dq_test[1]

        N = M_dot - 2 * C
        skew_check = np.max(np.abs(N + N.T))
        print(f"   q=({np.degrees(q_test[0]):.0f}°,{np.degrees(q_test[1]):.0f}°), "
              f"dq=({dq_test[0]:.1f},{dq_test[1]:.1f}): "
              f"||N+N^T|| = {skew_check:.2e} {'(skew-symmetric)' if skew_check < 1e-6 else ''}")


def exercise_4():
    """
    Exercise 4: Controller Comparison
    PD, PD+gravity, computed torque for 2-link arm.
    """
    # Parameters
    m1, m2 = 3.0, 2.0
    l1, l2 = 0.5, 0.4
    lc1, lc2 = 0.25, 0.2
    I1 = m1 * l1**2 / 12
    I2 = m2 * l2**2 / 12
    g = 9.81
    dt = 0.001
    T = 3.0
    steps = int(T / dt)

    q_d = np.radians([45.0, 30.0])
    Kp = np.diag([100.0, 100.0])
    Kd = np.diag([20.0, 20.0])

    def dynamics(q, dq, tau):
        """Compute acceleration from tau = M*qdd + C*dq + g."""
        c2 = np.cos(q[1])
        s2 = np.sin(q[1])
        M11 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2)
        M12 = I2 + m2*(lc2**2 + l1*lc2*c2)
        M22 = I2 + m2*lc2**2
        M = np.array([[M11, M12], [M12, M22]])

        h = m2 * l1 * lc2 * s2
        C_dq = np.array([-h*dq[1]*(2*dq[0]+dq[1]), h*dq[0]**2])

        g_vec = np.array([
            (m1*lc1 + m2*l1)*g*np.cos(q[0]) + m2*lc2*g*np.cos(q[0]+q[1]),
            m2*lc2*g*np.cos(q[0]+q[1])
        ])

        qdd = np.linalg.solve(M, tau - C_dq - g_vec)
        return qdd

    def gravity_vec(q):
        return np.array([
            (m1*lc1 + m2*l1)*g*np.cos(q[0]) + m2*lc2*g*np.cos(q[0]+q[1]),
            m2*lc2*g*np.cos(q[0]+q[1])
        ])

    def simulate(controller_name, control_fn):
        q = np.array([0.0, 0.0])
        dq = np.array([0.0, 0.0])
        errors = []

        for step in range(steps):
            e = q_d - q
            de = -dq
            tau = control_fn(q, dq, e, de)
            qdd = dynamics(q, dq, tau)
            dq = dq + qdd * dt
            q = q + dq * dt
            errors.append(np.degrees(np.linalg.norm(e)))

        ss_error = errors[-1]
        max_error = max(errors)
        return ss_error, max_error

    # Controller 1: PD only
    def pd_control(q, dq, e, de):
        return Kp @ e + Kd @ de

    # Controller 2: PD + gravity compensation
    def pd_gravity(q, dq, e, de):
        return Kp @ e + Kd @ de + gravity_vec(q)

    # Controller 3: Computed torque
    def computed_torque(q, dq, e, de):
        c2 = np.cos(q[1])
        M11 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2)
        M12 = I2 + m2*(lc2**2 + l1*lc2*c2)
        M22 = I2 + m2*lc2**2
        M = np.array([[M11, M12], [M12, M22]])
        s2 = np.sin(q[1])
        h = m2 * l1 * lc2 * s2
        C_dq = np.array([-h*dq[1]*(2*dq[0]+dq[1]), h*dq[0]**2])
        g_v = gravity_vec(q)

        # qdd_des = Kp*e + Kd*de (desired acceleration)
        qdd_des = Kp @ e + Kd @ de
        return M @ qdd_des + C_dq + g_v

    controllers = [
        ("PD only", pd_control),
        ("PD + gravity", pd_gravity),
        ("Computed torque", computed_torque),
    ]

    print(f"Step input: q = (0,0) → q_d = (45°, 30°)")
    print(f"Kp = diag(100, 100), Kd = diag(20, 20)")
    print(f"\n{'Controller':>20} | {'SS error (°)':>12} | {'Max error (°)':>13}")
    print("-" * 55)

    for name, ctrl_fn in controllers:
        ss_err, max_err = simulate(name, ctrl_fn)
        print(f"{name:>20} | {ss_err:>12.4f} | {max_err:>13.4f}")

    print(f"\nObservations:")
    print(f"  - PD only: significant steady-state error due to uncompensated gravity")
    print(f"  - PD + gravity: eliminates steady-state error; gravity term provides")
    print(f"    the needed feedforward torque")
    print(f"  - Computed torque: best transient response; linearizes the dynamics,")
    print(f"    making the error dynamics behave like a linear PD system")


def exercise_5():
    """
    Exercise 5: Newton-Euler Implementation
    """
    m1, m2 = 3.0, 2.0
    l1, l2 = 0.5, 0.4
    lc1, lc2 = 0.25, 0.2
    I1 = m1 * l1**2 / 12
    I2 = m2 * l2**2 / 12
    g = 9.81

    def newton_euler_2link(q, dq, ddq):
        """Recursive Newton-Euler for 2-link planar arm."""
        # Forward pass: compute link velocities and accelerations
        # Link 1
        omega1 = dq[0]
        alpha1 = ddq[0]
        ac1_x = -lc1 * (alpha1 * np.sin(q[0]) + dq[0]**2 * np.cos(q[0]))
        ac1_y = lc1 * (alpha1 * np.cos(q[0]) - dq[0]**2 * np.sin(q[0])) - g

        # Link 2
        omega2 = dq[0] + dq[1]
        alpha2 = ddq[0] + ddq[1]
        # Acceleration of joint 2
        aj2_x = -l1 * (alpha1 * np.sin(q[0]) + dq[0]**2 * np.cos(q[0]))
        aj2_y = l1 * (alpha1 * np.cos(q[0]) - dq[0]**2 * np.sin(q[0])) - g
        # Acceleration of CoM2
        s12 = np.sin(q[0] + q[1])
        c12 = np.cos(q[0] + q[1])
        ac2_x = aj2_x - lc2 * (alpha2 * s12 + omega2**2 * c12)
        ac2_y = aj2_y + lc2 * (alpha2 * c12 - omega2**2 * s12)

        # Backward pass: compute forces and torques
        # Link 2
        f2_x = m2 * ac2_x
        f2_y = m2 * ac2_y + m2 * g  # add back gravity component
        tau2 = I2 * alpha2 + lc2 * (f2_x * (-s12) + f2_y * c12)

        # Link 1
        f1_x = m1 * ac1_x + f2_x
        f1_y = m1 * ac1_y + m1 * g + f2_y
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        tau1 = (I1 * alpha1 + lc1 * (f1_x * (-s1) + f1_y * c1)
                + l1 * (f2_x * (-s1) + f2_y * c1) + tau2)

        return np.array([tau1, tau2])

    def euler_lagrange_2link(q, dq, ddq):
        """Euler-Lagrange inverse dynamics."""
        c2 = np.cos(q[1])
        s2 = np.sin(q[1])
        M11 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2)
        M12 = I2 + m2*(lc2**2 + l1*lc2*c2)
        M22 = I2 + m2*lc2**2
        M = np.array([[M11, M12], [M12, M22]])

        h = m2 * l1 * lc2 * s2
        C_dq = np.array([-h*dq[1]*(2*dq[0]+dq[1]), h*dq[0]**2])

        g_vec = np.array([
            (m1*lc1 + m2*l1)*g*np.cos(q[0]) + m2*lc2*g*np.cos(q[0]+q[1]),
            m2*lc2*g*np.cos(q[0]+q[1])
        ])
        return M @ ddq + C_dq + g_vec

    # Verify at random configurations
    np.random.seed(42)
    print("Newton-Euler vs Euler-Lagrange comparison (2-link arm):")
    print(f"\n{'Config':>8} | {'NE tau':>20} | {'EL tau':>20} | {'Max diff':>10}")
    print("-" * 70)

    import time
    n_configs = 5

    for i in range(n_configs):
        q = np.random.uniform(-np.pi, np.pi, 2)
        dq = np.random.uniform(-2, 2, 2)
        ddq = np.random.uniform(-5, 5, 2)

        tau_ne = newton_euler_2link(q, dq, ddq)
        tau_el = euler_lagrange_2link(q, dq, ddq)
        diff = np.max(np.abs(tau_ne - tau_el))
        print(f"  {i + 1:>5} | [{tau_ne[0]:>8.4f}, {tau_ne[1]:>8.4f}] | "
              f"[{tau_el[0]:>8.4f}, {tau_el[1]:>8.4f}] | {diff:>10.2e}")

    # Timing comparison
    n_timing = 10000
    t0 = time.perf_counter()
    for _ in range(n_timing):
        q = np.random.uniform(-np.pi, np.pi, 2)
        dq = np.random.uniform(-2, 2, 2)
        ddq = np.random.uniform(-5, 5, 2)
        newton_euler_2link(q, dq, ddq)
    t_ne = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_timing):
        q = np.random.uniform(-np.pi, np.pi, 2)
        dq = np.random.uniform(-2, 2, 2)
        ddq = np.random.uniform(-5, 5, 2)
        euler_lagrange_2link(q, dq, ddq)
    t_el = time.perf_counter() - t0

    print(f"\nTiming ({n_timing} evaluations):")
    print(f"  Newton-Euler:    {t_ne * 1000:.1f} ms")
    print(f"  Euler-Lagrange:  {t_el * 1000:.1f} ms")
    print(f"  For 2 links they are similar. NE scales O(n) while EL scales O(n^3+).")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 06: Robot Dynamics — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: Single Link Dynamics ---")
    exercise_1()

    print("\n--- Exercise 2: Inertia Matrix Analysis ---")
    exercise_2()

    print("\n--- Exercise 3: Coriolis Effects ---")
    exercise_3()

    print("\n--- Exercise 4: Controller Comparison ---")
    exercise_4()

    print("\n--- Exercise 5: Newton-Euler Implementation ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
