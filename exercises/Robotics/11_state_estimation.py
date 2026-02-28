"""
Exercises for Lesson 11: State Estimation and Filtering
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: 1D Kalman Filter — constant velocity model.
    State: [x, x_dot], GPS-like measurements at 1Hz.
    """
    np.random.seed(42)
    dt = 1.0       # measurement interval (1 Hz GPS)
    T = 50.0       # total time
    steps = int(T / dt)

    # State transition: x_k = F * x_{k-1} + w
    F = np.array([[1, dt],
                   [0, 1]])
    # Process noise (random acceleration)
    q_acc = 0.5  # acceleration noise std
    Q = np.array([[dt**4 / 4, dt**3 / 2],
                   [dt**3 / 2, dt**2]]) * q_acc**2
    # Measurement: z = H * x + v
    H = np.array([[1, 0]])  # observe position only
    R = np.array([[4.0]])   # GPS noise variance (sigma = 2m)

    # Ground truth: constant velocity + random acceleration perturbations
    x_true = np.zeros((steps + 1, 2))
    x_true[0] = [0, 2.0]  # start at x=0, v=2 m/s

    for k in range(steps):
        acc = np.random.normal(0, q_acc)
        x_true[k + 1, 0] = x_true[k, 0] + x_true[k, 1] * dt + 0.5 * acc * dt**2
        x_true[k + 1, 1] = x_true[k, 1] + acc * dt

    # Generate measurements
    measurements = x_true[1:, 0] + np.random.normal(0, 2.0, steps)

    # Kalman filter
    x_est = np.array([0.0, 0.0])  # initial estimate
    P = np.diag([10.0, 10.0])     # initial covariance (uncertain)

    kf_states = [x_est.copy()]
    kf_sigmas = [np.sqrt(P[0, 0])]

    for k in range(steps):
        # Predict
        x_pred = F @ x_est
        P_pred = F @ P @ F.T + Q

        # Update
        z = measurements[k]
        y = z - H @ x_pred  # innovation
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_est = x_pred + K.flatten() * y
        P = (np.eye(2) - K @ H) @ P_pred

        kf_states.append(x_est.copy())
        kf_sigmas.append(np.sqrt(P[0, 0]))

    kf_states = np.array(kf_states)
    kf_sigmas = np.array(kf_sigmas)

    # Results
    print("1D Kalman Filter: Constant Velocity Model")
    print(f"  GPS noise: sigma = 2.0 m")
    print(f"  Process noise: sigma_acc = {q_acc} m/s^2")

    print(f"\n  {'Time':>5} | {'True x':>8} | {'Meas':>8} | {'KF est':>8} | "
          f"{'+-2σ':>8} | {'Error':>8}")
    print("  " + "-" * 60)
    for k in [0, 5, 10, 20, 30, 40, 49]:
        idx = k + 1
        true_x = x_true[idx, 0]
        meas = measurements[k] if k < steps else float('nan')
        est = kf_states[idx, 0]
        sigma = kf_sigmas[idx]
        err = abs(true_x - est)
        print(f"  {(k+1)*dt:>5.0f} | {true_x:>8.2f} | {meas:>8.2f} | "
              f"{est:>8.2f} | {2*sigma:>8.2f} | {err:>8.2f}")

    # RMS error comparison
    meas_errors = [abs(measurements[k] - x_true[k+1, 0]) for k in range(steps)]
    kf_errors = [abs(kf_states[k+1, 0] - x_true[k+1, 0]) for k in range(steps)]
    print(f"\n  RMS position error:")
    print(f"    Measurements: {np.sqrt(np.mean(np.array(meas_errors)**2)):.2f} m")
    print(f"    KF estimate:  {np.sqrt(np.mean(np.array(kf_errors)**2)):.2f} m")
    print(f"    KF reduces error by fusing prediction + measurement.")


def exercise_2():
    """
    Exercise 2: EKF for Differential-Drive Robot Localization.
    """
    np.random.seed(42)
    dt = 0.1
    T = 30.0
    steps = int(T / dt)

    # 6 known landmarks
    landmarks = np.array([
        [2, 2], [8, 2], [5, 8], [1, 7], [9, 7], [5, 1]
    ], dtype=float)

    # Noise parameters
    sigma_v = 0.1    # velocity noise
    sigma_w = 0.05   # angular velocity noise
    sigma_r = 0.5    # range noise
    sigma_phi = np.radians(5)  # bearing noise
    max_range = 6.0  # sensor range

    Q = np.diag([0.01, 0.01, 0.005])  # process noise
    R = np.diag([sigma_r**2, sigma_phi**2])  # measurement noise

    # True trajectory: circle
    v_true = 1.0     # m/s
    w_true = 0.2     # rad/s

    x_true = np.zeros((steps + 1, 3))
    x_true[0] = [5, 5, 0]  # start at center

    x_dr = np.zeros((steps + 1, 3))  # dead reckoning
    x_dr[0] = x_true[0].copy()

    x_ekf = np.zeros((steps + 1, 3))
    x_ekf[0] = x_true[0].copy()
    P = np.diag([0.1, 0.1, 0.01])

    for k in range(steps):
        # True motion
        v = v_true + np.random.normal(0, sigma_v)
        w = w_true + np.random.normal(0, sigma_w)

        theta = x_true[k, 2]
        x_true[k+1, 0] = x_true[k, 0] + v * np.cos(theta) * dt
        x_true[k+1, 1] = x_true[k, 1] + v * np.sin(theta) * dt
        x_true[k+1, 2] = x_true[k, 2] + w * dt

        # Dead reckoning (uses noisy inputs, no correction)
        theta_dr = x_dr[k, 2]
        x_dr[k+1, 0] = x_dr[k, 0] + v * np.cos(theta_dr) * dt
        x_dr[k+1, 1] = x_dr[k, 1] + v * np.sin(theta_dr) * dt
        x_dr[k+1, 2] = x_dr[k, 2] + w * dt

        # EKF Predict
        theta_e = x_ekf[k, 2]
        x_pred = np.array([
            x_ekf[k, 0] + v_true * np.cos(theta_e) * dt,
            x_ekf[k, 1] + v_true * np.sin(theta_e) * dt,
            x_ekf[k, 2] + w_true * dt
        ])

        F = np.array([
            [1, 0, -v_true * np.sin(theta_e) * dt],
            [0, 1,  v_true * np.cos(theta_e) * dt],
            [0, 0, 1]
        ])
        P = F @ P @ F.T + Q

        # EKF Update with visible landmarks
        for lm in landmarks:
            dx = lm[0] - x_true[k+1, 0]
            dy = lm[1] - x_true[k+1, 1]
            r_true = np.sqrt(dx**2 + dy**2)
            if r_true > max_range:
                continue

            # Noisy measurement
            r_meas = r_true + np.random.normal(0, sigma_r)
            phi_meas = (np.arctan2(dy, dx) - x_true[k+1, 2]
                        + np.random.normal(0, sigma_phi))

            # Expected measurement
            dx_e = lm[0] - x_pred[0]
            dy_e = lm[1] - x_pred[1]
            r_exp = np.sqrt(dx_e**2 + dy_e**2)
            phi_exp = np.arctan2(dy_e, dx_e) - x_pred[2]

            # Jacobian
            H = np.array([
                [-dx_e/r_exp, -dy_e/r_exp, 0],
                [dy_e/r_exp**2, -dx_e/r_exp**2, -1]
            ])

            innovation = np.array([r_meas - r_exp, phi_meas - phi_exp])
            innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi

            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x_pred = x_pred + K @ innovation
            P = (np.eye(3) - K @ H) @ P

        x_ekf[k+1] = x_pred

    # Results
    dr_errors = np.sqrt((x_dr[:, 0] - x_true[:, 0])**2 + (x_dr[:, 1] - x_true[:, 1])**2)
    ekf_errors = np.sqrt((x_ekf[:, 0] - x_true[:, 0])**2 + (x_ekf[:, 1] - x_true[:, 1])**2)

    print("EKF Localization: Differential-Drive Robot")
    print(f"  {len(landmarks)} landmarks, max range = {max_range}m")
    print(f"\n  {'Time':>5} | {'DR error (m)':>12} | {'EKF error (m)':>13}")
    print("  " + "-" * 38)
    for k in [0, 50, 100, 150, 200, 250, 299]:
        print(f"  {k * dt:>5.1f} | {dr_errors[k]:>12.4f} | {ekf_errors[k]:>13.4f}")

    print(f"\n  Final errors: DR = {dr_errors[-1]:.4f}m, EKF = {ekf_errors[-1]:.4f}m")
    print(f"  EKF reduces drift by using landmark observations for correction.")


def exercise_3():
    """
    Exercise 3: Particle Filter for Global Localization.
    """
    np.random.seed(42)
    n_particles = 500
    env_size = 20.0

    # 3 known landmarks
    landmarks = np.array([[5, 5], [15, 5], [10, 15]], dtype=float)
    sigma_obs = 1.0

    # True robot position
    x_true = np.array([7.0, 8.0, np.radians(45)])

    # Initialize particles uniformly
    particles = np.random.uniform(0, env_size, (n_particles, 2))
    weights = np.ones(n_particles) / n_particles

    print("Particle Filter: Global Localization")
    print(f"  Environment: {env_size}m x {env_size}m")
    print(f"  {n_particles} particles, {len(landmarks)} landmarks")
    print(f"  True position: ({x_true[0]:.1f}, {x_true[1]:.1f})")

    # Multiple observation steps (robot stays still, takes measurements)
    n_observations = 5
    for obs in range(n_observations):
        # Simulate observations from true position
        obs_ranges = []
        for lm in landmarks:
            r_true = np.linalg.norm(lm - x_true[:2])
            r_meas = r_true + np.random.normal(0, sigma_obs)
            obs_ranges.append(r_meas)

        # Update weights
        for i in range(n_particles):
            log_w = 0
            for j, lm in enumerate(landmarks):
                r_exp = np.linalg.norm(lm - particles[i])
                # Gaussian likelihood
                log_w += -0.5 * ((obs_ranges[j] - r_exp) / sigma_obs)**2
            weights[i] = np.exp(log_w)

        # Normalize
        weights /= np.sum(weights)

        # Effective sample size
        n_eff = 1.0 / np.sum(weights**2)

        # Resample (systematic resampling)
        if n_eff < n_particles / 2:
            cumsum = np.cumsum(weights)
            r = np.random.uniform(0, 1.0 / n_particles)
            indices = []
            j = 0
            for i in range(n_particles):
                u = r + i / n_particles
                while u > cumsum[j]:
                    j += 1
                indices.append(j)
            particles = particles[indices]
            # Add small noise after resampling
            particles += np.random.normal(0, 0.1, particles.shape)
            weights = np.ones(n_particles) / n_particles

        # Statistics
        mean = np.average(particles, weights=weights, axis=0)
        std = np.sqrt(np.average((particles - mean)**2, weights=weights, axis=0))
        error = np.linalg.norm(mean - x_true[:2])

        print(f"\n  Observation {obs + 1}:")
        print(f"    Mean estimate: ({mean[0]:.2f}, {mean[1]:.2f})")
        print(f"    Std:           ({std[0]:.2f}, {std[1]:.2f})")
        print(f"    Error:         {error:.4f} m")
        print(f"    N_eff:         {n_eff:.0f} / {n_particles}")

    print(f"\n  The particle cloud converges from uniform to tight cluster")
    print(f"  around the true position as more observations are incorporated.")


def exercise_4():
    """
    Exercise 4: Filter Comparison (EKF vs UKF vs PF).
    """
    np.random.seed(42)

    # Nonlinear system: range-bearing localization
    landmark = np.array([5.0, 5.0])
    sigma_r, sigma_b = 0.5, np.radians(5)
    R = np.diag([sigma_r**2, sigma_b**2])

    # True state
    x_true = np.array([2.0, 3.0])
    # Measurement function
    def h(x):
        dx = landmark[0] - x[0]
        dy = landmark[1] - x[1]
        return np.array([np.sqrt(dx**2 + dy**2), np.arctan2(dy, dx)])

    import time

    # Generate measurements
    n_steps = 50
    measurements = []
    for _ in range(n_steps):
        z_true = h(x_true)
        z = z_true + np.array([np.random.normal(0, sigma_r),
                                np.random.normal(0, sigma_b)])
        measurements.append(z)

    # EKF
    def run_ekf(x0, P0):
        x = x0.copy()
        P = P0.copy()
        t0 = time.perf_counter()
        errors = []
        for z in measurements:
            dx = landmark[0] - x[0]
            dy = landmark[1] - x[1]
            r = np.sqrt(dx**2 + dy**2)
            H = np.array([
                [-dx/r, -dy/r],
                [dy/r**2, -dx/r**2]
            ])
            z_pred = h(x)
            inn = z - z_pred
            inn[1] = (inn[1] + np.pi) % (2*np.pi) - np.pi
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ inn
            P = (np.eye(2) - K @ H) @ P
            errors.append(np.linalg.norm(x - x_true))
        elapsed = time.perf_counter() - t0
        return errors, elapsed

    # Simple UKF
    def run_ukf(x0, P0):
        n = 2
        alpha, beta, kappa = 1e-3, 2, 0
        lam = alpha**2 * (n + kappa) - n
        x = x0.copy()
        P = P0.copy()
        t0 = time.perf_counter()
        errors = []

        for z in measurements:
            # Generate sigma points
            sqrt_P = np.linalg.cholesky((n + lam) * P)
            sigma_pts = np.zeros((2*n+1, n))
            sigma_pts[0] = x
            for i in range(n):
                sigma_pts[i+1] = x + sqrt_P[i]
                sigma_pts[n+i+1] = x - sqrt_P[i]

            # Weights
            w_m = np.full(2*n+1, 1/(2*(n+lam)))
            w_c = np.full(2*n+1, 1/(2*(n+lam)))
            w_m[0] = lam/(n+lam)
            w_c[0] = lam/(n+lam) + (1-alpha**2+beta)

            # Transform sigma points through measurement function
            z_pts = np.array([h(sp) for sp in sigma_pts])
            z_pred = np.average(z_pts, weights=w_m, axis=0)

            Pzz = sum(w_c[i] * np.outer(z_pts[i]-z_pred, z_pts[i]-z_pred)
                       for i in range(2*n+1)) + R
            Pxz = sum(w_c[i] * np.outer(sigma_pts[i]-x, z_pts[i]-z_pred)
                       for i in range(2*n+1))

            K = Pxz @ np.linalg.inv(Pzz)
            inn = z - z_pred
            inn[1] = (inn[1] + np.pi) % (2*np.pi) - np.pi
            x = x + K @ inn
            P = P - K @ Pzz @ K.T
            errors.append(np.linalg.norm(x - x_true))

        elapsed = time.perf_counter() - t0
        return errors, elapsed

    # Particle filter
    def run_pf(x0, P0, n_particles=200):
        particles = np.random.multivariate_normal(x0, P0, n_particles)
        weights = np.ones(n_particles) / n_particles
        t0 = time.perf_counter()
        errors = []

        for z in measurements:
            for i in range(n_particles):
                z_pred = h(particles[i])
                inn = z - z_pred
                inn[1] = (inn[1] + np.pi) % (2*np.pi) - np.pi
                log_w = -0.5 * inn @ np.linalg.inv(R) @ inn
                weights[i] = np.exp(log_w)
            weights /= np.sum(weights)

            # Resample
            indices = np.random.choice(n_particles, n_particles, p=weights)
            particles = particles[indices] + np.random.normal(0, 0.01, (n_particles, 2))
            weights = np.ones(n_particles) / n_particles

            mean = np.mean(particles, axis=0)
            errors.append(np.linalg.norm(mean - x_true))

        elapsed = time.perf_counter() - t0
        return errors, elapsed

    # Run comparison
    x0_near = np.array([2.5, 3.5])  # near true state
    P0 = np.diag([1.0, 1.0])

    print("Filter Comparison: Nonlinear range-bearing localization")
    print(f"  True state: {x_true}")
    print(f"  Initial estimate: {x0_near}")

    results = {}
    for name, runner in [("EKF", lambda: run_ekf(x0_near, P0)),
                          ("UKF", lambda: run_ukf(x0_near, P0)),
                          ("PF (200)", lambda: run_pf(x0_near, P0))]:
        errs, t = runner()
        results[name] = (errs, t)

    print(f"\n  {'Filter':>10} | {'Final err (m)':>13} | {'Mean err (m)':>12} | {'Time (ms)':>10}")
    print("  " + "-" * 55)
    for name, (errs, t) in results.items():
        print(f"  {name:>10} | {errs[-1]:>13.4f} | {np.mean(errs):>12.4f} | {t*1000:>10.2f}")


def exercise_5():
    """
    Exercise 5: Multi-rate Sensor Fusion.
    """
    np.random.seed(42)
    dt_sim = 0.001  # 1000 Hz simulation
    T = 60.0
    steps = int(T / dt_sim)

    # Sensor rates
    imu_rate = 100      # Hz
    odom_rate = 50      # Hz
    gps_rate = 1        # Hz
    gps_loss_start = 20.0  # GPS lost at 20s
    gps_loss_end = 50.0    # GPS returns at 50s

    # True trajectory: figure-8
    def true_state(t):
        x = 10 * np.sin(0.2 * t)
        y = 5 * np.sin(0.4 * t)
        vx = 10 * 0.2 * np.cos(0.2 * t)
        vy = 5 * 0.4 * np.cos(0.4 * t)
        return np.array([x, y, vx, vy])

    # Simple KF state: [x, y, vx, vy]
    F = np.eye(4)
    dt_imu = 1.0 / imu_rate
    F[0, 2] = dt_imu
    F[1, 3] = dt_imu

    Q_imu = np.diag([0.01, 0.01, 0.1, 0.1])
    R_odom = np.diag([0.05, 0.05, 0.2, 0.2])
    R_gps = np.diag([2.0, 2.0])

    H_odom = np.eye(4)
    H_gps = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    x_est = np.array([0.0, 0.0, 2.0, 2.0])
    P = np.diag([5.0, 5.0, 1.0, 1.0])

    errors_with_gps = []
    errors_without_gps = []
    times_log = []
    last_imu = 0
    last_odom = 0
    last_gps = 0

    for step in range(steps):
        t = step * dt_sim
        s_true = true_state(t)

        # IMU update (predict step)
        if t - last_imu >= 1.0 / imu_rate:
            # Predict with IMU
            ax_true = -10 * 0.2**2 * np.sin(0.2 * t)
            ay_true = -5 * 0.4**2 * np.sin(0.4 * t)
            ax = ax_true + np.random.normal(0, 0.5)
            ay = ay_true + np.random.normal(0, 0.5)

            x_est[2] += ax * dt_imu
            x_est[3] += ay * dt_imu
            x_est[0] += x_est[2] * dt_imu
            x_est[1] += x_est[3] * dt_imu

            F_k = np.eye(4)
            F_k[0, 2] = dt_imu
            F_k[1, 3] = dt_imu
            P = F_k @ P @ F_k.T + Q_imu
            last_imu = t

        # Odometry update
        if t - last_odom >= 1.0 / odom_rate:
            z_odom = s_true + np.random.normal(0, [0.1, 0.1, 0.3, 0.3])
            y_inn = z_odom - H_odom @ x_est
            S = H_odom @ P @ H_odom.T + R_odom
            K = P @ H_odom.T @ np.linalg.inv(S)
            x_est = x_est + K @ y_inn
            P = (np.eye(4) - K @ H_odom) @ P
            last_odom = t

        # GPS update (if available)
        gps_available = not (gps_loss_start <= t <= gps_loss_end)
        if gps_available and t - last_gps >= 1.0 / gps_rate:
            z_gps = s_true[:2] + np.random.normal(0, 2.0, 2)
            y_inn = z_gps - H_gps @ x_est
            S = H_gps @ P @ H_gps.T + R_gps
            K = P @ H_gps.T @ np.linalg.inv(S)
            x_est = x_est + K @ y_inn
            P = (np.eye(4) - K @ H_gps) @ P
            last_gps = t

        # Log every 0.1s
        if step % 100 == 0:
            err = np.linalg.norm(x_est[:2] - s_true[:2])
            times_log.append(t)
            if gps_available:
                errors_with_gps.append(err)
                errors_without_gps.append(None)
            else:
                errors_with_gps.append(None)
                errors_without_gps.append(err)

    print("Multi-rate Sensor Fusion:")
    print(f"  IMU: {imu_rate}Hz, Odometry: {odom_rate}Hz, GPS: {gps_rate}Hz")
    print(f"  GPS lost: {gps_loss_start}-{gps_loss_end}s")

    print(f"\n  {'Time (s)':>8} | {'GPS?':>5} | {'Position error (m)':>18}")
    print("  " + "-" * 38)
    for i, t in enumerate(times_log):
        if i % 50 == 0:
            err = errors_with_gps[i] if errors_with_gps[i] is not None else errors_without_gps[i]
            gps_str = "YES" if errors_with_gps[i] is not None else "NO"
            if err is not None:
                print(f"  {t:>8.1f} | {gps_str:>5} | {err:>18.4f}")

    # Summary statistics
    err_gps = [e for e in errors_with_gps if e is not None]
    err_nogps = [e for e in errors_without_gps if e is not None]
    print(f"\n  Average error WITH GPS:    {np.mean(err_gps):.4f} m")
    print(f"  Average error WITHOUT GPS: {np.mean(err_nogps):.4f} m")
    print(f"  Error ratio: {np.mean(err_nogps)/np.mean(err_gps):.1f}x worse without GPS")
    print(f"\n  Without GPS, the estimate degrades as IMU/odometry drift accumulates.")
    print(f"  GPS provides absolute position reference that bounds the drift.")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 11: State Estimation and Filtering — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: Kalman Filter Implementation ---")
    exercise_1()

    print("\n--- Exercise 2: EKF for Differential-Drive Robot ---")
    exercise_2()

    print("\n--- Exercise 3: Particle Filter Global Localization ---")
    exercise_3()

    print("\n--- Exercise 4: Filter Comparison ---")
    exercise_4()

    print("\n--- Exercise 5: Multi-rate Fusion ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
