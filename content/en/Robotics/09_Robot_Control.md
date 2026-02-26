# 9. Robot Control

[← Previous: Trajectory Planning and Execution](08_Trajectory_Planning.md) | [Next: Sensors and Perception →](10_Sensors_and_Perception.md)

---

## Learning Objectives

1. Design and tune joint-space PID controllers for robotic manipulators
2. Derive and implement computed torque control using inverse dynamics and PD feedback
3. Understand impedance control and its spring-damper model for compliant interaction
4. Distinguish between force control and hybrid position/force control strategies
5. Recognize adaptive control principles for handling model uncertainties
6. Analyze robustness and disturbance rejection in robot control systems

---

A robot that can plan beautiful trajectories is useless if it cannot execute them accurately in the physical world. Control is the bridge between planned motion and actual motion — it is the discipline of computing the right torques, forces, and commands so that the robot's joints follow desired trajectories despite gravity, friction, inertia, and unexpected disturbances. In the previous lesson, we learned *what* the robot should do (trajectory planning). In this lesson, we learn *how* to make it actually do it.

Robot control is especially challenging because manipulators are highly nonlinear, coupled, multi-input multi-output (MIMO) systems. A torque applied at one joint affects every other joint through inertial coupling. The effective inertia of each joint changes with the robot's configuration. Gravity loads shift as the arm moves. These complexities demand control strategies that go beyond the simple PID controllers found in most introductory courses — though we will start there, because PID remains the workhorse of industrial robotics.

---

## 1. Joint-Space PID Control

### 1.1 The Independent Joint Control Paradigm

The simplest approach to robot control treats each joint as an independent single-input, single-output (SISO) system. Each joint has its own PID controller that computes a torque command based on the joint-level tracking error.

For joint $i$, the control law is:

$$\tau_i = K_{p,i} e_i(t) + K_{i,i} \int_0^t e_i(\sigma) \, d\sigma + K_{d,i} \dot{e}_i(t)$$

where $e_i(t) = q_{d,i}(t) - q_i(t)$ is the position error, $q_{d,i}$ is the desired position, and $q_i$ is the measured position.

In vector form for all $n$ joints:

$$\boldsymbol{\tau} = K_p \mathbf{e}(t) + K_i \int_0^t \mathbf{e}(\sigma) \, d\sigma + K_d \dot{\mathbf{e}}(t)$$

where $K_p$, $K_i$, $K_d$ are diagonal $n \times n$ gain matrices.

```python
import numpy as np

class JointPIDController:
    """Independent joint PID controller for an n-DOF robot.

    Why diagonal gain matrices? Each joint is treated independently,
    which simplifies tuning but ignores inter-joint coupling effects.
    This works well when gear ratios are high (reducing coupling)
    or when the robot moves slowly (small dynamic effects).
    """

    def __init__(self, n_joints, kp, ki, kd, dt=0.001):
        # Gain matrices are diagonal — one set of gains per joint
        self.Kp = np.diag(kp)  # Proportional gains [Nm/rad]
        self.Ki = np.diag(ki)  # Integral gains [Nm/(rad*s)]
        self.Kd = np.diag(kd)  # Derivative gains [Nm*s/rad]
        self.dt = dt

        # Integral accumulator and previous error for derivative
        self.integral_error = np.zeros(n_joints)
        self.prev_error = np.zeros(n_joints)

    def compute(self, q_desired, q_actual, qd_desired=None, qd_actual=None):
        """Compute joint torques from tracking error.

        Why accept velocity signals separately? When velocity measurements
        are available (e.g., from tachometers), we can compute derivative
        action more accurately than by differencing position readings.
        """
        error = q_desired - q_actual
        self.integral_error += error * self.dt

        # Use measured velocities if available; otherwise differentiate error
        if qd_desired is not None and qd_actual is not None:
            derror = qd_desired - qd_actual
        else:
            derror = (error - self.prev_error) / self.dt

        self.prev_error = error.copy()

        tau = self.Kp @ error + self.Ki @ self.integral_error + self.Kd @ derror
        return tau

    def reset(self):
        """Reset integrator state — important when switching setpoints."""
        self.integral_error[:] = 0.0
        self.prev_error[:] = 0.0
```

### 1.2 PID Tuning for Robot Joints

Tuning PID gains for robot joints is more nuanced than for simple linear systems because the effective plant dynamics change with configuration. A gain set that works well when the arm is extended may be sluggish or oscillatory when the arm is tucked in.

**Practical tuning procedure**:

1. **Start with PD only** (set $K_i = 0$). This avoids integrator windup during initial tuning.
2. **Increase $K_p$** until the joint responds quickly but starts to oscillate.
3. **Add $K_d$** to damp the oscillations. The derivative term acts as a virtual damper.
4. **Add small $K_i$** only if steady-state error is unacceptable. In many robotic applications, PD control is sufficient because gravity compensation (see below) eliminates the primary source of steady-state error.
5. **Test across the workspace** — move the arm to different configurations and verify stability.

**Gravity compensation** is a critical enhancement to joint PID:

$$\boldsymbol{\tau} = K_p \mathbf{e} + K_d \dot{\mathbf{e}} + \mathbf{g}(\mathbf{q})$$

where $\mathbf{g}(\mathbf{q})$ is the gravity torque vector. By adding a feedforward gravity term, we remove the constant disturbance that would otherwise require large $K_i$ gains to overcome.

```python
def pd_gravity_compensation(q_desired, q, qd, Kp, Kd, gravity_func):
    """PD control with gravity compensation.

    Why add gravity feedforward? Without it, the PD controller must use
    its proportional term to fight gravity, leading to a permanent
    position error (droop). The gravity term cancels this load exactly,
    so the PD only needs to handle dynamic tracking errors.
    """
    error = q_desired - q
    derror = -qd  # Desired velocity is zero for regulation

    tau = Kp @ error + Kd @ derror + gravity_func(q)
    return tau
```

### 1.3 Limitations of Independent Joint PID

Independent joint PID ignores the coupled dynamics of the robot:

- **Configuration-dependent inertia**: The effective inertia seen by each joint motor changes as the robot moves, making a fixed set of gains a compromise.
- **Coriolis and centrifugal coupling**: Fast motions at one joint create torques at other joints that the independent PID must treat as disturbances.
- **Limited performance envelope**: Independent joint PID works well for slow, precise movements (typical of industrial pick-and-place) but struggles with fast, dynamic motions.

These limitations motivate model-based control approaches.

---

## 2. Computed Torque Control

### 2.1 The Idea: Cancel the Nonlinearities

Recall from Lesson 6 that the robot dynamics are:

$$M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$$

where $M$ is the inertia matrix, $C$ contains Coriolis and centrifugal terms, and $\mathbf{g}$ is the gravity vector.

**Computed torque control** (also called **inverse dynamics control** or **feedback linearization**) uses the dynamic model to cancel the nonlinear terms, then applies a simple linear control law to the resulting "linearized" system.

The control law is:

$$\boldsymbol{\tau} = M(\mathbf{q})\mathbf{u} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

where $\mathbf{u}$ is an auxiliary control input. Substituting into the dynamics:

$$M(\mathbf{q})\ddot{\mathbf{q}} = M(\mathbf{q})\mathbf{u}$$

Since $M(\mathbf{q})$ is always positive definite (invertible), we get:

$$\ddot{\mathbf{q}} = \mathbf{u}$$

This is a set of $n$ decoupled double integrators — a linear system! We can now design $\mathbf{u}$ using any linear control technique. The standard choice is PD control on the error:

$$\mathbf{u} = \ddot{\mathbf{q}}_d + K_d \dot{\mathbf{e}} + K_p \mathbf{e}$$

where $\mathbf{e} = \mathbf{q}_d - \mathbf{q}$.

The closed-loop error dynamics become:

$$\ddot{\mathbf{e}} + K_d \dot{\mathbf{e}} + K_p \mathbf{e} = \mathbf{0}$$

This is a stable second-order system for any positive definite $K_p$ and $K_d$.

### 2.2 Choosing Gains

The error dynamics for each joint are:

$$\ddot{e}_i + 2\zeta_i \omega_{n,i} \dot{e}_i + \omega_{n,i}^2 e_i = 0$$

Comparing with the general form, we set:

$$K_{p,i} = \omega_{n,i}^2, \qquad K_{d,i} = 2\zeta_i \omega_{n,i}$$

For critical damping ($\zeta = 1$), if we want a natural frequency of $\omega_n = 50$ rad/s:

$$K_p = 2500, \qquad K_d = 100$$

```python
class ComputedTorqueController:
    """Computed torque (inverse dynamics + PD) controller.

    Why use the full dynamic model? By computing and canceling the
    nonlinear dynamics (inertia coupling, Coriolis, gravity), we
    convert the control problem from a complex nonlinear MIMO problem
    into n independent linear double-integrator problems.
    """

    def __init__(self, robot_model, kp, kd):
        self.robot = robot_model
        self.Kp = np.diag(kp)
        self.Kd = np.diag(kd)

    def compute(self, q_desired, qd_desired, qdd_desired, q, qd):
        """Compute control torques using inverse dynamics + PD.

        The three components:
        1. M(q)*u: Feedforward + feedback through the inertia matrix
        2. C(q,qd)*qd: Cancel velocity-dependent forces
        3. g(q): Cancel gravitational forces
        """
        error = q_desired - q
        derror = qd_desired - qd

        # Auxiliary input: desired acceleration + PD correction
        u = qdd_desired + self.Kp @ error + self.Kd @ derror

        # Full inverse dynamics
        M = self.robot.inertia_matrix(q)
        C = self.robot.coriolis_matrix(q, qd)
        g = self.robot.gravity_vector(q)

        tau = M @ u + C @ qd + g
        return tau
```

### 2.3 When Computed Torque Fails

Computed torque assumes **perfect knowledge** of $M$, $C$, and $\mathbf{g}$. In practice:

- **Parameter uncertainty**: Link masses, inertias, and center-of-mass locations are never known exactly.
- **Unmodeled dynamics**: Joint flexibility, actuator dynamics, gear backlash, and friction are absent from the rigid-body model.
- **Computational cost**: Evaluating the full dynamics at servo rate (1 kHz+) requires efficient algorithms (e.g., recursive Newton-Euler).

When the model is inaccurate, the nonlinear cancellation is imperfect, and the closed-loop system is no longer a set of perfect double integrators. The residual nonlinearities appear as disturbances. This motivates **robust** and **adaptive** control strategies.

---

## 3. Impedance Control

### 3.1 From Position Control to Interaction Control

Consider a robot that must sand a surface, insert a peg into a hole, or hand an object to a human. In these tasks, the robot inevitably makes contact with the environment. A pure position controller will either:

- Apply excessive force if the surface is slightly closer than expected, or
- Fail to maintain contact if the surface is slightly farther.

We need a controller that manages the **relationship between motion and force** rather than controlling each independently.

> **Analogy**: Impedance control is like holding an egg — you adjust stiffness to be firm enough to grip but gentle enough not to crush. A position controller is like a rigid clamp: it goes to a position regardless of what is in the way. An impedance controller behaves like a spring-damper system: it yields when encountering resistance and pushes when displaced from equilibrium.

### 3.2 Impedance Control Law

**Impedance control** specifies the desired dynamic relationship between the deviation from a reference position and the force exerted on the environment:

$$\mathbf{F} = M_d(\ddot{\mathbf{x}}_d - \ddot{\mathbf{x}}) + B_d(\dot{\mathbf{x}}_d - \dot{\mathbf{x}}) + K_d(\mathbf{x}_d - \mathbf{x})$$

where:
- $M_d$ is the desired **inertia** matrix (how the robot resists acceleration)
- $B_d$ is the desired **damping** matrix (viscous friction behavior)
- $K_d$ is the desired **stiffness** matrix (spring-like behavior)
- $\mathbf{x}_d$ is the reference position, $\mathbf{x}$ is the actual position

**Physical interpretation and units** of the impedance parameters:

| Symbol | Name | Unit (linear / rotational) | Physical meaning |
|--------|------|---------------------------|-----------------|
| $M_d$ | Desired inertia | kg / kg$\cdot$m$^2$ | Apparent mass felt by the environment; larger $M_d$ makes the robot sluggish to accelerate |
| $B_d$ | Desired damping | N$\cdot$s/m / N$\cdot$m$\cdot$s/rad | Energy dissipation rate; larger $B_d$ suppresses oscillation but slows response |
| $K_d$ | Desired stiffness | N/m / N$\cdot$m/rad | Restoring force per unit displacement; larger $K_d$ means tighter position tracking but harder contact |

In the Laplace domain, the impedance transfer function is $Z_d(s) = M_d s^2 + B_d s + K_d$, mapping displacement to force. Choosing these three parameters fully specifies how the robot feels to the environment at every frequency — $K_d$ dominates at low frequencies (static stiffness), $B_d$ at mid frequencies (damping), and $M_d$ at high frequencies (inertia).

The robot behaves as if it were a **mass-spring-damper system** connected to the reference trajectory. When in free space, it tracks $\mathbf{x}_d$ closely. When it contacts the environment, it deviates from $\mathbf{x}_d$ and exerts a force proportional to the deviation.

### 3.3 Stiffness and Damping Design

The choice of $K_d$ and $B_d$ depends on the task:

| Task | Stiffness | Damping | Rationale |
|------|-----------|---------|-----------|
| Polishing | Low | High | Conform to surface, avoid bouncing |
| Assembly (peg-in-hole) | Low in insertion dir, high in lateral | Moderate | Compliant along insertion, precise laterally |
| Handover to human | Very low | Low | Easy for human to guide |
| Precise positioning | High | Critical damping | Stiff tracking, no oscillation |

**Critical damping** for each Cartesian DOF:

$$B_{d,i} = 2\sqrt{K_{d,i} \cdot M_{d,i}}$$

```python
class ImpedanceController:
    """Cartesian impedance controller.

    Why control impedance instead of position? When the robot interacts
    with an environment that has its own dynamics (stiffness, mass),
    the combined robot-environment system must be stable. Impedance
    control guarantees passivity: the robot absorbs energy on contact
    rather than injecting energy that could cause instability.
    """

    def __init__(self, Md, Bd, Kd):
        # Desired impedance parameters (6x6 for full Cartesian space)
        self.Md = np.array(Md)  # Desired inertia [kg, kg*m^2]
        self.Bd = np.array(Bd)  # Desired damping [Ns/m, Ns*m/rad]
        self.Kd = np.array(Kd)  # Desired stiffness [N/m, Nm/rad]

    def compute_force(self, x_desired, x_actual, xd_desired, xd_actual,
                      xdd_desired=None):
        """Compute desired Cartesian force from impedance model.

        Why separate from joint torque computation? The impedance model
        operates in Cartesian space. We convert to joint torques using
        the Jacobian: tau = J^T * F.
        """
        pos_error = x_desired - x_actual
        vel_error = xd_desired - xd_actual

        F = self.Kd @ pos_error + self.Bd @ vel_error

        if xdd_desired is not None:
            F += self.Md @ xdd_desired

        return F

    def compute_torque(self, F_cartesian, jacobian, q, qd, robot_model):
        """Convert Cartesian force command to joint torques.

        Why add gravity compensation? The impedance model defines the
        desired behavior in task space. Gravity is a joint-space
        disturbance that must be canceled separately.
        """
        # Map Cartesian force to joint torques
        tau = jacobian.T @ F_cartesian

        # Add gravity compensation
        tau += robot_model.gravity_vector(q)

        return tau
```

### 3.4 Impedance vs. Admittance Control

There are two dual approaches to interaction control:

- **Impedance control**: Measures motion, commands force. Input: displacement → Output: force.
  - Best for robots with good torque control (direct-drive, series elastic actuators).
- **Admittance control**: Measures force, commands motion. Input: force → Output: displacement.
  - Best for stiff robots with position control (industrial robots with high gear ratios).

$$\text{Impedance: } \mathbf{F} = Z(\mathbf{x}_d - \mathbf{x})$$
$$\text{Admittance: } \mathbf{x} = Y(\mathbf{F}_{ext})$$

where $Z$ is the impedance operator and $Y = Z^{-1}$ is the admittance operator.

**When to choose which?** The selection depends on the robot's natural dynamics and actuation. If the robot is naturally backdrivable (low gear ratio, direct-drive), it can accurately command torques, so impedance control (measure motion, output force) is natural. If the robot has high gear ratios (stiff position-controlled industrial arms), commanding precise positions is easy but commanding precise torques is not, so admittance control (measure force via a sensor, output position corrections) is preferred. Mathematically, the combined robot-environment impedance must remain passive (positive real) for stability; choosing the control mode that complements the hardware's inherent impedance ensures this condition is met with simpler controller design.

---

## 4. Force Control

### 4.1 Direct Force Control

In some tasks, we need to control the contact force directly. For example, a robot grinding a surface must maintain a specific normal force regardless of surface irregularities.

The simplest force controller uses PI feedback on the force error:

$$\boldsymbol{\tau}_f = K_{fp}(\mathbf{F}_d - \mathbf{F}) + K_{fi} \int_0^t (\mathbf{F}_d - \mathbf{F}(\sigma)) \, d\sigma$$

where $\mathbf{F}_d$ is the desired force and $\mathbf{F}$ is the measured force (from a force/torque sensor).

**Challenges of pure force control**:

- Requires a force/torque sensor (cost, noise, drift).
- When the robot loses contact, force error goes to $\mathbf{F}_d$ and the controller drives the robot into the surface aggressively — this needs explicit contact detection and mode switching.
- Force measurement noise limits the bandwidth of force control.

### 4.2 Hybrid Position/Force Control

**Hybrid control** (Raibert and Craig, 1981) separates the task space into position-controlled and force-controlled subspaces using a **selection matrix** $S$:

$$\boldsymbol{\tau} = J^T\left[S \cdot \mathbf{F}_{force} + (I - S) \cdot \mathbf{F}_{position}\right]$$

where:
- $S$ is a diagonal selection matrix with 1s for force-controlled DOFs and 0s for position-controlled DOFs
- $\mathbf{F}_{force}$ comes from a force controller
- $\mathbf{F}_{position}$ comes from a position/impedance controller

**Example**: Wiping a table surface:

| DOF | Control Mode | Rationale |
|-----|-------------|-----------|
| $x$ (along surface) | Position | Move across the surface |
| $y$ (along surface) | Position | Move across the surface |
| $z$ (normal to surface) | Force | Maintain constant pressing force |
| $\theta_x, \theta_y$ | Force (zero) | Allow tilting to conform to surface |
| $\theta_z$ | Position | Control orientation of wiper |

```python
class HybridController:
    """Hybrid position/force controller.

    Why split DOFs between position and force control? In contact tasks,
    some directions are constrained by the environment (e.g., you can't
    move through a wall), so force control is natural. Other directions
    are free (e.g., sliding along the wall), so position control makes sense.
    """

    def __init__(self, pos_controller, force_controller, selection_matrix):
        self.pos_ctrl = pos_controller
        self.force_ctrl = force_controller
        # S: diagonal matrix, 1 = force-controlled, 0 = position-controlled
        self.S = np.diag(selection_matrix)

    def compute(self, x_d, xd_d, F_d, x, xd, F_measured, jacobian, q, robot):
        """Compute joint torques from hybrid position/force control.

        Why use a selection matrix? It provides a clean mathematical
        framework for assigning control modes per DOF. In practice,
        the selection can change dynamically based on the task phase
        (e.g., approach → contact → slide).
        """
        # Force-controlled DOFs
        F_force = self.force_ctrl.compute(F_d, F_measured)

        # Position-controlled DOFs
        F_pos = self.pos_ctrl.compute_force(x_d, x, xd_d, xd)

        # Combine using selection matrix
        F_task = self.S @ F_force + (np.eye(len(self.S)) - self.S) @ F_pos

        # Map to joint torques
        tau = jacobian.T @ F_task + robot.gravity_vector(q)
        return tau
```

---

## 5. Adaptive Control

### 5.1 Motivation

Computed torque control requires accurate knowledge of the dynamic parameters:

$$\boldsymbol{\theta} = [m_1, m_1 l_{c1}, I_1, m_2, m_2 l_{c2}, I_2, \ldots]^T$$

In practice, these parameters are uncertain. **Adaptive control** estimates these parameters online while controlling the robot.

### 5.2 Linearity in the Parameters

A key property of robot dynamics is that the equations are **linear in the dynamic parameters**:

$$M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = Y(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}) \boldsymbol{\theta}$$

where $Y$ is the **regressor matrix** (depends on kinematics, not on the unknown parameters) and $\boldsymbol{\theta}$ is the parameter vector.

This allows us to design an adaptive law that uses tracking errors to update $\hat{\boldsymbol{\theta}}$.

### 5.3 Adaptive Computed Torque

The adaptive version of computed torque control replaces the true parameters with estimated parameters:

$$\boldsymbol{\tau} = \hat{M}(\mathbf{q})\mathbf{u} + \hat{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \hat{\mathbf{g}}(\mathbf{q})$$

where $\hat{M}$, $\hat{C}$, $\hat{\mathbf{g}}$ use the current parameter estimates $\hat{\boldsymbol{\theta}}$.

The parameter update law:

$$\dot{\hat{\boldsymbol{\theta}}} = \Gamma Y^T(\mathbf{q}, \dot{\mathbf{q}}, \dot{\mathbf{q}}_r, \ddot{\mathbf{q}}_r) \mathbf{s}$$

where:
- $\Gamma > 0$ is the adaptation gain matrix
- $\mathbf{s} = \dot{\mathbf{e}} + \Lambda \mathbf{e}$ is the sliding variable
- $\dot{\mathbf{q}}_r = \dot{\mathbf{q}}_d + \Lambda \mathbf{e}$ is the reference velocity
- $\Lambda > 0$ is a design parameter

```python
class AdaptiveController:
    """Adaptive computed torque controller.

    Why adapt online? In real robots, payloads change (picking up objects),
    joints wear (friction changes), and the environment varies. Adaptive
    control automatically adjusts the internal model to maintain performance
    without manual re-identification.
    """

    def __init__(self, n_joints, n_params, Kd, Lambda, Gamma):
        self.Kd = np.diag(Kd)           # PD gain
        self.Lambda = np.diag(Lambda)    # Sliding surface slope
        self.Gamma = np.diag(Gamma)      # Adaptation rate
        self.theta_hat = np.zeros(n_params)  # Parameter estimates

    def compute(self, q_d, qd_d, qdd_d, q, qd, regressor_func, dt):
        """Compute torque and update parameter estimates.

        Why use a sliding variable s? It combines position and velocity
        errors into a single measure. When s → 0, both the position
        and velocity errors converge to zero. This simplifies the
        stability analysis (Lyapunov-based).
        """
        e = q_d - q
        ed = qd_d - qd

        # Reference velocity and acceleration
        qd_r = qd_d + self.Lambda @ e
        qdd_r = qdd_d + self.Lambda @ ed

        # Sliding variable
        s = ed + self.Lambda @ e

        # Regressor matrix: dynamics = Y(q, qd, qd_r, qdd_r) * theta
        Y = regressor_func(q, qd, qd_r, qdd_r)

        # Control torque using current parameter estimates
        tau = Y @ self.theta_hat + self.Kd @ s

        # Update parameter estimates
        self.theta_hat += self.Gamma @ Y.T @ s * dt

        return tau
```

### 5.4 Practical Considerations

- **Persistent excitation**: Parameters converge only if the robot motion is "rich enough" to excite all dynamic parameters. In practice, this means the robot must move through various configurations.
- **Parameter projection**: Estimated parameters must remain physically meaningful (e.g., masses must be positive). Projection algorithms enforce these constraints.
- **Convergence speed vs. noise sensitivity**: Higher adaptation gain $\Gamma$ means faster convergence but more sensitivity to measurement noise.

---

## 6. Robustness and Disturbance Rejection

### 6.1 Sources of Disturbances

Real robots face numerous disturbances:

| Source | Character | Typical Magnitude |
|--------|-----------|-------------------|
| Joint friction | Nonlinear (Coulomb + viscous) | 5-15% of rated torque |
| Payload uncertainty | Parametric | Varies with task |
| External contact | Impulsive or sustained | Task-dependent |
| Sensor noise | High-frequency | 0.1-1% of range |
| Model mismatch | Structured uncertainty | 10-30% of model terms |

### 6.2 Robust Control: Sliding Mode

**Sliding mode control** provides robustness to bounded model uncertainties by adding a discontinuous switching term:

$$\boldsymbol{\tau} = \hat{M}\mathbf{u} + \hat{C}\dot{\mathbf{q}} + \hat{\mathbf{g}} + K_{robust} \cdot \text{sgn}(\mathbf{s})$$

where $\text{sgn}(\mathbf{s})$ is the signum function applied element-wise.

The switching term pushes the system toward the **sliding surface** $\mathbf{s} = \dot{\mathbf{e}} + \Lambda \mathbf{e} = \mathbf{0}$, despite uncertainties.

**Chattering problem**: The discontinuous $\text{sgn}$ function causes high-frequency switching (chattering) in practice. Solutions include:

1. **Boundary layer**: Replace $\text{sgn}(s)$ with $\text{sat}(s/\phi)$ where $\phi$ is the boundary layer thickness
2. **Super-twisting algorithm**: A higher-order sliding mode that produces continuous control
3. **Adaptive boundary**: Adjust $\phi$ based on the observed chattering

```python
def sliding_mode_control(q_d, qd_d, qdd_d, q, qd, robot, Lambda, K_robust, phi):
    """Sliding mode controller with boundary layer.

    Why use a boundary layer? Pure sliding mode causes chattering —
    high-frequency oscillation around the sliding surface. The boundary
    layer replaces the hard switching with a smooth approximation,
    trading some robustness for practical smoothness.
    """
    e = q_d - q
    ed = qd_d - qd

    # Sliding variable
    s = ed + Lambda @ e

    # Reference trajectory
    qd_r = qd_d + Lambda @ e
    qdd_r = qdd_d + Lambda @ ed

    # Nominal computed torque
    M = robot.inertia_matrix(q)
    C = robot.coriolis_matrix(q, qd)
    g = robot.gravity_vector(q)

    u = qdd_r  # Could add PD here for faster convergence
    tau_nominal = M @ u + C @ qd + g

    # Robust switching term with boundary layer (saturation function)
    # sat(s/phi) = s/phi if |s| < phi, else sgn(s)
    sat = np.clip(s / phi, -1.0, 1.0)
    tau_robust = K_robust @ sat

    return tau_nominal + tau_robust
```

### 6.3 Disturbance Observer (DOB)

A **disturbance observer** estimates the lumped disturbance acting on the system and cancels it:

$$\hat{d} = Q(s) \left[\boldsymbol{\tau} - M_n \ddot{\mathbf{q}}\right]$$

where $M_n$ is the nominal model and $Q(s)$ is a low-pass filter. The estimated disturbance $\hat{d}$ includes model uncertainties, friction, and external forces. We subtract $\hat{d}$ from the control command to reject these disturbances.

**Advantages over sliding mode**: Continuous control signal (no chattering), works with any inner-loop controller, provides a disturbance estimate useful for fault detection.

---

## 7. Control Architecture Overview

### 7.1 Hierarchical Control

Modern robot control systems use a hierarchical architecture:

```
┌──────────────────────────────────┐
│  Task-Level Planner (1-10 Hz)    │  ← Trajectory waypoints
├──────────────────────────────────┤
│  Cartesian Controller (100 Hz)   │  ← Impedance/force control
├──────────────────────────────────┤
│  Joint Controller (1 kHz)        │  ← PID/computed torque
├──────────────────────────────────┤
│  Motor Driver (10 kHz)           │  ← Current control
└──────────────────────────────────┘
```

Each level operates at a different rate, with higher levels providing setpoints to lower levels.

### 7.2 Comparison of Control Strategies

| Strategy | Model Required | Handles Contact | Robustness | Complexity |
|----------|---------------|-----------------|------------|------------|
| Joint PID | None (minimal) | Poor | Low | Low |
| PD + Gravity | Gravity model | Poor | Moderate | Low |
| Computed Torque | Full dynamics | Poor | Low (sensitive) | High |
| Impedance | Full dynamics | Excellent | Moderate | High |
| Hybrid | Full dynamics | Excellent | Moderate | Very high |
| Adaptive | Regressor form | Moderate | High | Very high |
| Sliding Mode | Bounds on uncertainty | Moderate | Very high | High |

### 7.3 Practical Recommendations

- **Industrial pick-and-place**: PD + gravity compensation (simple, reliable, sufficient for slow motions with high-ratio gears).
- **High-speed manipulation**: Computed torque or adaptive control (must compensate dynamic effects).
- **Human-robot collaboration**: Impedance control (safety through compliance).
- **Contact-rich tasks**: Hybrid position/force or impedance control.
- **Uncertain environments**: Adaptive or sliding mode control.

---

## 8. Putting It All Together: Simulation Example

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_2link_control(controller_type='pid', T=5.0, dt=0.001):
    """Simulate control of a 2-link planar robot.

    Why simulate before deploying? Control parameters that look
    good on paper can fail catastrophically on a real robot.
    Simulation lets us tune gains, test edge cases (singularities,
    joint limits), and verify stability safely.
    """
    # Robot parameters
    m1, m2 = 1.0, 1.0  # Link masses [kg]
    l1, l2 = 1.0, 1.0  # Link lengths [m]
    lc1, lc2 = 0.5, 0.5  # Center of mass distances [m]
    I1 = m1 * l1**2 / 12  # Moments of inertia [kg*m^2]
    I2 = m2 * l2**2 / 12
    g_acc = 9.81

    def dynamics(q, qd, tau):
        """2-link planar robot dynamics (Euler-Lagrange).
        Returns joint accelerations given state and torques.
        """
        q1, q2 = q
        qd1, qd2 = qd

        # Inertia matrix entries
        d11 = m1*lc1**2 + I1 + m2*(l1**2 + lc2**2 + 2*l1*lc2*np.cos(q2)) + I2
        d12 = m2*(lc2**2 + l1*lc2*np.cos(q2)) + I2
        d22 = m2*lc2**2 + I2

        M = np.array([[d11, d12], [d12, d22]])

        # Coriolis/centrifugal
        h = m2 * l1 * lc2 * np.sin(q2)
        C = np.array([[-h * qd2, -h * (qd1 + qd2)],
                       [h * qd1, 0.0]])

        # Gravity
        g = np.array([
            (m1*lc1 + m2*l1) * g_acc * np.cos(q1) + m2*lc2*g_acc*np.cos(q1+q2),
            m2 * lc2 * g_acc * np.cos(q1 + q2)
        ])

        qdd = np.linalg.solve(M, tau - C @ qd - g)
        return qdd

    def gravity_vec(q):
        q1, q2 = q
        return np.array([
            (m1*lc1 + m2*l1)*g_acc*np.cos(q1) + m2*lc2*g_acc*np.cos(q1+q2),
            m2*lc2*g_acc*np.cos(q1+q2)
        ])

    # Desired trajectory: sinusoidal joint motion
    def desired_trajectory(t):
        q_d = np.array([np.sin(t), 0.5 * np.sin(2 * t)])
        qd_d = np.array([np.cos(t), np.cos(2 * t)])
        qdd_d = np.array([-np.sin(t), -2 * np.sin(2 * t)])
        return q_d, qd_d, qdd_d

    # Initialize
    n_steps = int(T / dt)
    q = np.array([0.0, 0.0])
    qd = np.array([0.0, 0.0])

    q_history = np.zeros((n_steps, 2))
    qd_history = np.zeros((n_steps, 2))
    error_history = np.zeros((n_steps, 2))
    tau_history = np.zeros((n_steps, 2))

    # Controller gains
    Kp = np.diag([100.0, 100.0])
    Kd = np.diag([20.0, 20.0])

    for i in range(n_steps):
        t = i * dt
        q_d, qd_d, qdd_d = desired_trajectory(t)

        e = q_d - q
        ed = qd_d - qd

        if controller_type == 'pid':
            tau = Kp @ e + Kd @ ed + gravity_vec(q)
        elif controller_type == 'computed_torque':
            # Full computed torque (we reuse dynamics components)
            q1, q2 = q
            qd1, qd2 = qd
            d11 = m1*lc1**2+I1+m2*(l1**2+lc2**2+2*l1*lc2*np.cos(q2))+I2
            d12 = m2*(lc2**2+l1*lc2*np.cos(q2))+I2
            d22 = m2*lc2**2+I2
            M = np.array([[d11, d12], [d12, d22]])
            h = m2*l1*lc2*np.sin(q2)
            C = np.array([[-h*qd2, -h*(qd1+qd2)], [h*qd1, 0.0]])
            g = gravity_vec(q)
            u = qdd_d + Kp @ e + Kd @ ed
            tau = M @ u + C @ qd + g
        else:
            raise ValueError(f"Unknown controller: {controller_type}")

        # Simulate dynamics (Euler integration)
        qdd = dynamics(q, qd, tau)
        qd = qd + qdd * dt
        q = q + qd * dt

        q_history[i] = q
        error_history[i] = e
        tau_history[i] = tau

    return q_history, error_history, tau_history

# Run comparison
# q_pid, e_pid, _ = simulate_2link_control('pid')
# q_ct, e_ct, _ = simulate_2link_control('computed_torque')
# Plot tracking errors to see the dramatic improvement of computed torque
```

---

## Summary

| Concept | Key Idea |
|---------|----------|
| Joint PID | Simple, treats joints independently; add gravity compensation for steady-state accuracy |
| Computed torque | Cancel nonlinear dynamics using the model; converts to linear double-integrator control |
| Impedance control | Controls the force-motion relationship; robot behaves as a programmable spring-damper |
| Force control | Direct regulation of contact forces using force/torque sensor feedback |
| Hybrid control | Partition task space into position-controlled and force-controlled subspaces |
| Adaptive control | Online parameter estimation to handle model uncertainty |
| Sliding mode | Discontinuous switching for robustness to bounded uncertainties |
| DOB | Estimate and cancel lumped disturbances using a low-pass filtered inverse model |

---

## Exercises

1. **PID tuning experiment**: Simulate a single-joint robot (pendulum with motor) under gravity. Start with $K_p = 50$, $K_d = 10$, $K_i = 0$. Gradually increase $K_i$ and observe the effect on steady-state error and oscillation. Then add gravity compensation and repeat — explain why $K_i$ becomes unnecessary.

2. **Computed torque vs. PID**: Using the 2-link simulation code above, compare tracking errors for a fast sinusoidal trajectory ($\omega = 5$ rad/s) between PD+gravity and computed torque control. Plot the position errors for both joints. Why does the performance gap increase with speed?

3. **Impedance control design**: A robot must polish a surface. Design impedance parameters ($K_d$, $B_d$) for 3 DOFs: two tangential (along the surface) and one normal. The robot should maintain approximately 10 N of normal force while tracking a circular path along the surface. Implement and simulate the controller.

4. **Hybrid controller**: Implement the hybrid position/force controller for a peg-in-hole task. The peg is aligned with the $z$-axis. Use force control for $z$ (maintain 5 N insertion force) and position control for $x$ and $y$ (center the peg). Simulate with a simple environment model.

5. **Sliding mode chattering**: Implement the sliding mode controller with and without the boundary layer. Plot the control torques for both cases. Measure the chattering amplitude and frequency. How does the boundary layer thickness $\phi$ affect the trade-off between robustness and smoothness?

---

## Further Reading

- Siciliano, B. et al. *Robotics: Modelling, Planning and Control*. Springer, 2009. Chapters 8-9. (Comprehensive treatment of robot control)
- Slotine, J.-J. E. and Li, W. *Applied Nonlinear Control*. Prentice Hall, 1991. (Adaptive and sliding mode control)
- Hogan, N. "Impedance Control: An Approach to Manipulation." *ASME Journal of Dynamic Systems*, 1985. (Original impedance control paper)
- Raibert, M. H. and Craig, J. J. "Hybrid Position/Force Control of Manipulators." *ASME Journal of Dynamic Systems*, 1981.

---

[← Previous: Trajectory Planning and Execution](08_Trajectory_Planning.md) | [Next: Sensors and Perception →](10_Sensors_and_Perception.md)
