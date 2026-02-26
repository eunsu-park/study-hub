# Velocity Kinematics and the Jacobian

[← Previous: Inverse Kinematics](04_Inverse_Kinematics.md) | [Next: Robot Dynamics →](06_Robot_Dynamics.md)

## Learning Objectives

1. Define angular and linear velocity for rigid bodies and express them in different reference frames
2. Derive the manipulator Jacobian and understand its role as the fundamental velocity mapping between joint space and task space
3. Distinguish between the geometric Jacobian and the analytical Jacobian, and know when to use each
4. Analyze singularities through the Jacobian's singular values and identify their physical meaning
5. Apply the force/torque mapping (statics duality) to relate end-effector forces to joint torques
6. Compute and interpret manipulability ellipsoids and dexterity measures for evaluating robot configurations

---

## Why This Matters

The Jacobian is arguably the single most important mathematical object in robotics. It appears everywhere: in velocity control (mapping joint speeds to end-effector speed), in force control (mapping end-effector forces to joint torques), in singularity analysis (determining where the robot loses dexterity), in trajectory tracking (computing joint accelerations), and in dynamics (building the equations of motion).

Understanding the Jacobian transforms you from someone who can only solve kinematic puzzles into someone who can design controllers, analyze workspace quality, and reason about what a robot can and cannot do at any configuration. If forward kinematics tells you "where," the Jacobian tells you "how fast and in which directions."

> **Analogy**: The Jacobian is like a gear ratio — it translates joint speeds into end-effector speeds. Just as a car's transmission converts engine RPM into wheel speed (and the ratio changes with gear selection), the Jacobian converts joint velocities into end-effector velocity (and the "ratio" changes with the robot's configuration).

---

## Angular and Linear Velocity

### Linear Velocity

For a point $\mathbf{p}(t)$ moving in space, the linear velocity is simply:

$$\mathbf{v} = \dot{\mathbf{p}} = \frac{d\mathbf{p}}{dt}$$

### Angular Velocity

A rotating rigid body has an **angular velocity vector** $\boldsymbol{\omega}$ that describes both the axis and rate of rotation. For a rotation matrix $R(t)$:

$$\dot{R}(t) = [\boldsymbol{\omega}]_\times R(t) \quad \text{(body frame)}$$
$$\dot{R}(t) = [\boldsymbol{\omega}_s]_\times R(t) \quad \text{where } \boldsymbol{\omega}_s = R \boldsymbol{\omega} \text{ (space frame)}$$

where $[\boldsymbol{\omega}]_\times$ is the skew-symmetric matrix of $\boldsymbol{\omega}$.

**Key property**: Angular velocities add as vectors (unlike finite rotations, which do not commute):

$$\boldsymbol{\omega}_{total} = \boldsymbol{\omega}_1 + \boldsymbol{\omega}_2$$

### Velocity of a Point on a Rigid Body

For a point $P$ on a rigid body with origin $O$, angular velocity $\boldsymbol{\omega}$, and linear velocity $\mathbf{v}_O$:

$$\mathbf{v}_P = \mathbf{v}_O + \boldsymbol{\omega} \times \mathbf{r}_{OP}$$

where $\mathbf{r}_{OP}$ is the position of $P$ relative to $O$.

```python
import numpy as np

def skew(w):
    """Skew-symmetric matrix (cross-product matrix).

    Why a matrix for cross product? Because it lets us express
    omega x r as a matrix-vector product [omega]_x * r, which
    integrates cleanly into the Jacobian formulation.
    """
    return np.array([[    0, -w[2],  w[1]],
                     [ w[2],     0, -w[0]],
                     [-w[1],  w[0],     0]])

def velocity_of_point(v_origin, omega, r_OP):
    """Velocity of point P on a rigid body.

    v_P = v_O + omega x r_OP

    This is the fundamental kinematic equation for rigid bodies.
    Every column of the Jacobian is essentially this equation
    applied to one joint's contribution.
    """
    return v_origin + np.cross(omega, r_OP)

# Example: a wheel rotating at 10 rad/s about z-axis
omega = np.array([0, 0, 10])  # rad/s about z
r = np.array([0.3, 0, 0])     # point on the rim, 0.3m from center
v_point = velocity_of_point(np.zeros(3), omega, r)
print(f"Rim velocity: {v_point} m/s")  # [0, 3, 0] — tangential velocity
print(f"|v| = {np.linalg.norm(v_point):.1f} m/s")  # omega * r = 3 m/s
```

---

## The Manipulator Jacobian

### Definition

The **manipulator Jacobian** $J(\mathbf{q})$ is a matrix that maps joint velocities to end-effector velocities:

$$\begin{bmatrix} \mathbf{v} \\ \boldsymbol{\omega} \end{bmatrix} = J(\mathbf{q}) \, \dot{\mathbf{q}}$$

where:
- $\mathbf{v} \in \mathbb{R}^3$ is the end-effector linear velocity
- $\boldsymbol{\omega} \in \mathbb{R}^3$ is the end-effector angular velocity
- $\dot{\mathbf{q}} \in \mathbb{R}^n$ is the vector of joint velocities
- $J(\mathbf{q}) \in \mathbb{R}^{6 \times n}$ (for full 6-DOF task space)

### Jacobian Columns: Contribution of Each Joint

The $i$-th column of $J$ describes the end-effector velocity produced by unit velocity at joint $i$ (with all other joints stationary). For an $n$-DOF serial manipulator:

$$J = \begin{bmatrix} J_{v_1} & J_{v_2} & \cdots & J_{v_n} \\ J_{\omega_1} & J_{\omega_2} & \cdots & J_{\omega_n} \end{bmatrix}$$

For a **revolute** joint $i$:
$$J_i = \begin{bmatrix} J_{v_i} \\ J_{\omega_i} \end{bmatrix} = \begin{bmatrix} \hat{z}_{i-1} \times (\mathbf{p}_n - \mathbf{p}_{i-1}) \\ \hat{z}_{i-1} \end{bmatrix}$$

For a **prismatic** joint $i$:
$$J_i = \begin{bmatrix} J_{v_i} \\ J_{\omega_i} \end{bmatrix} = \begin{bmatrix} \hat{z}_{i-1} \\ \mathbf{0} \end{bmatrix}$$

where:
- $\hat{z}_{i-1}$ is the joint axis (z-axis of frame $\{i-1\}$) expressed in the base frame
- $\mathbf{p}_n$ is the end-effector position in the base frame
- $\mathbf{p}_{i-1}$ is the origin of frame $\{i-1\}$ in the base frame

```python
def compute_geometric_jacobian(joint_types, z_axes, origins, p_end):
    """Compute the 6xN geometric Jacobian.

    Why 'geometric'? Because this Jacobian uses the geometric relationship
    between joint axes and end-effector position. It directly gives
    linear + angular velocity of the end-effector in the base frame.

    This is the most commonly used Jacobian in robotics.

    Parameters:
        joint_types: list of 'revolute' or 'prismatic'
        z_axes: list of joint axis directions in base frame (z_{i-1})
        origins: list of joint origin positions in base frame (p_{i-1})
        p_end: end-effector position in base frame (p_n)

    Returns:
        J: 6xN Jacobian matrix
    """
    n = len(joint_types)
    J = np.zeros((6, n))

    for i in range(n):
        z = z_axes[i]    # joint axis direction
        p = origins[i]   # joint origin position

        if joint_types[i] == 'revolute':
            # Linear: z x (p_end - p_joint)
            J[:3, i] = np.cross(z, p_end - p)
            # Angular: z
            J[3:, i] = z
        else:  # prismatic
            # Linear: z (translation along joint axis)
            J[:3, i] = z
            # Angular: 0 (translation doesn't create rotation)
            J[3:, i] = 0

    return J
```

### Jacobian from DH / FK

In practice, we compute the Jacobian from the forward kinematics transforms:

```python
class RobotJacobian:
    """Compute the geometric Jacobian from DH parameters and FK.

    Why combine with FK? Because the Jacobian depends on the current
    configuration — we need FK to know where each joint axis is
    in the base frame. This class reuses the FK computation.
    """
    def __init__(self, dh_params):
        self.dh_params = dh_params
        self.n = len(dh_params)

    def dh_transform(self, theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [ 0,     sa,     ca,    d],
            [ 0,      0,      0,    1]
        ])

    def fk(self, q):
        """Forward kinematics returning all intermediate transforms."""
        T = np.eye(4)
        transforms = [T.copy()]  # T_00 = I (base frame)

        for i in range(self.n):
            p = self.dh_params[i]
            if p['type'] == 'revolute':
                theta = p['theta'] + q[i]
                d = p['d']
            else:
                theta = p['theta']
                d = p['d'] + q[i]

            Ti = self.dh_transform(theta, d, p['a'], p['alpha'])
            T = T @ Ti
            transforms.append(T.copy())

        return T, transforms

    def jacobian(self, q):
        """Compute the 6xN geometric Jacobian at configuration q.

        The Jacobian changes with configuration — it must be recomputed
        at each time step in a control loop. For a 6-DOF robot at 1 kHz
        control rate, that's 1000 Jacobian computations per second.
        """
        T_end, transforms = self.fk(q)
        p_end = T_end[:3, 3]  # end-effector position

        J = np.zeros((6, self.n))

        for i in range(self.n):
            T_i = transforms[i]  # T_{0,i} — frame i-1 in base
            z_i = T_i[:3, 2]     # z-axis of frame i-1
            p_i = T_i[:3, 3]     # origin of frame i-1

            if self.dh_params[i]['type'] == 'revolute':
                J[:3, i] = np.cross(z_i, p_end - p_i)
                J[3:, i] = z_i
            else:
                J[:3, i] = z_i
                J[3:, i] = 0

        return J

    def end_effector_velocity(self, q, q_dot):
        """Compute end-effector velocity from joint velocities."""
        J = self.jacobian(q)
        v_ee = J @ q_dot
        return v_ee[:3], v_ee[3:]  # (linear, angular)


# Example: 2-link planar robot Jacobian
planar_2link_dh = [
    {'theta': 0, 'd': 0, 'a': 1.0, 'alpha': 0, 'type': 'revolute'},
    {'theta': 0, 'd': 0, 'a': 0.8, 'alpha': 0, 'type': 'revolute'},
]

robot = RobotJacobian(planar_2link_dh)

q = np.radians([30, 45])
J = robot.jacobian(q)
print("Jacobian at q=[30, 45] deg:")
print(np.round(J, 4))

# Verify: joint 1 at 1 rad/s should produce both linear and angular velocity
q_dot = np.array([1.0, 0.0])
v_lin, v_ang = robot.end_effector_velocity(q, q_dot)
print(f"\nJoint 1 only (1 rad/s):")
print(f"  Linear velocity:  {np.round(v_lin, 4)} m/s")
print(f"  Angular velocity: {np.round(v_ang, 4)} rad/s")
```

### Analytical Jacobian for 2-Link Planar Robot

For the 2-link planar robot, we can derive the Jacobian analytically by differentiating the FK equations:

$$x = l_1 \cos\theta_1 + l_2 \cos(\theta_1 + \theta_2)$$
$$y = l_1 \sin\theta_1 + l_2 \sin(\theta_1 + \theta_2)$$

The Jacobian (position part only, since we're in 2D):

$$J = \begin{bmatrix} \frac{\partial x}{\partial \theta_1} & \frac{\partial x}{\partial \theta_2} \\ \frac{\partial y}{\partial \theta_1} & \frac{\partial y}{\partial \theta_2} \end{bmatrix} = \begin{bmatrix} -l_1 s_1 - l_2 s_{12} & -l_2 s_{12} \\ l_1 c_1 + l_2 c_{12} & l_2 c_{12} \end{bmatrix}$$

where $s_1 = \sin\theta_1$, $c_1 = \cos\theta_1$, $s_{12} = \sin(\theta_1 + \theta_2)$, $c_{12} = \cos(\theta_1 + \theta_2)$.

```python
def jacobian_2link_analytical(theta1, theta2, l1, l2):
    """Analytical Jacobian for 2-link planar robot (2x2 position Jacobian).

    Why derive analytically? For simple robots, the analytical form
    reveals the structure directly. For instance, we can immediately see
    that det(J) = l1*l2*sin(theta2), telling us singularities occur
    at theta2 = 0 or pi — the arm is fully extended or folded.
    """
    s1 = np.sin(theta1)
    c1 = np.cos(theta1)
    s12 = np.sin(theta1 + theta2)
    c12 = np.cos(theta1 + theta2)

    J = np.array([
        [-l1*s1 - l2*s12, -l2*s12],
        [ l1*c1 + l2*c12,  l2*c12]
    ])
    return J

# Verify against numerical Jacobian
def numerical_jacobian(fk_func, q, delta=1e-7):
    """Numerical Jacobian via finite differences.

    Why also compute numerically? Cross-checking analytical against
    numerical Jacobian catches derivation errors. This is a standard
    verification technique in robotics.
    """
    x0 = fk_func(q)
    n = len(q)
    m = len(x0)
    J = np.zeros((m, n))

    for i in range(n):
        q_plus = q.copy()
        q_plus[i] += delta
        q_minus = q.copy()
        q_minus[i] -= delta
        J[:, i] = (fk_func(q_plus) - fk_func(q_minus)) / (2 * delta)

    return J

# FK function for the numerical Jacobian
l1, l2 = 1.0, 0.8
def fk_2link(q):
    return np.array([
        l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]),
        l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    ])

q = np.radians([30, 45])
J_analytical = jacobian_2link_analytical(q[0], q[1], l1, l2)
J_numerical = numerical_jacobian(fk_2link, q)

print("Analytical Jacobian:")
print(np.round(J_analytical, 6))
print("\nNumerical Jacobian:")
print(np.round(J_numerical, 6))
print(f"\nMatch: {np.allclose(J_analytical, J_numerical, atol=1e-5)}")
```

---

## Geometric vs Analytical Jacobian

There are two types of Jacobian, and the distinction matters:

### Geometric Jacobian $J_g$

Maps joint velocities to **spatial twist** (linear velocity + angular velocity):

$$\begin{bmatrix} \mathbf{v} \\ \boldsymbol{\omega} \end{bmatrix} = J_g(\mathbf{q}) \, \dot{\mathbf{q}}$$

The angular velocity $\boldsymbol{\omega}$ is a well-defined physical quantity — the instantaneous rotation axis and speed.

### Analytical Jacobian $J_a$

Maps joint velocities to **time derivatives of a pose parameterization** (e.g., position + Euler angles):

$$\dot{\mathbf{x}} = J_a(\mathbf{q}) \, \dot{\mathbf{q}}$$

where $\mathbf{x} = (x, y, z, \phi, \theta, \psi)^T$ with Euler angles.

### Relationship

The linear velocity parts are identical. The angular parts differ:

$$\boldsymbol{\omega} = B(\boldsymbol{\phi}) \, \dot{\boldsymbol{\phi}}$$

where $B$ is a matrix that depends on the Euler angle convention and current orientation. Therefore:

$$J_a = \begin{bmatrix} I & 0 \\ 0 & B^{-1} \end{bmatrix} J_g$$

```python
def euler_zyx_rate_matrix(phi, theta, psi):
    """Matrix B that maps Euler angle rates to angular velocity.

    omega = B * [phi_dot, theta_dot, psi_dot]^T

    Why does this matrix exist? Because Euler angle rates are NOT
    the same as angular velocity components. The angular velocity
    is a vector in 3D; Euler angle rates are derivatives of three
    successive rotation angles about different (moving) axes.

    WARNING: B is singular when theta = +/- 90 deg (gimbal lock!).
    This is another reason to prefer the geometric Jacobian.
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    st = np.sin(theta)

    B = np.array([
        [1, 0,     -st    ],
        [0, cp,  sp * ct  ],
        [0, -sp, cp * ct  ]
    ])
    return B

# Demonstration: Euler angle rates vs angular velocity
phi, theta, psi = np.radians([10, 30, 20])
B = euler_zyx_rate_matrix(phi, theta, psi)
print("Euler rate matrix B:")
print(np.round(B, 4))
print(f"det(B) = {np.linalg.det(B):.4f}")  # Non-zero: no gimbal lock here

# At gimbal lock (theta = 90 deg):
B_singular = euler_zyx_rate_matrix(0, np.pi/2, 0)
print(f"\nAt theta=90 deg, det(B) = {np.linalg.det(B_singular):.6f}")  # ~0
```

### When to Use Which?

| Jacobian | Use Case |
|----------|----------|
| Geometric $J_g$ | Velocity/force control, manipulability analysis, singularity analysis |
| Analytical $J_a$ | Task-space trajectory tracking using Euler angle error, optimization |

**Rule of thumb**: Use the geometric Jacobian unless you specifically need Euler angle rates.

---

## Singularity Analysis

### Revisiting Singularities Through the SVD

The **Singular Value Decomposition** (SVD) of the Jacobian provides the deepest insight into robot behavior:

$$J = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{m \times m}$: columns are **task-space singular directions**
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$
- $V \in \mathbb{R}^{n \times n}$: columns are **joint-space singular directions**

**Physical interpretation**:
- $\sigma_i$: the "gain" in the $i$-th singular direction (how much joint motion produces task-space motion)
- A singularity occurs when $\sigma_i = 0$ for some $i$
- Near-singularity: $\sigma_i \approx 0$ (large joint motion needed for small task motion)

```python
def singularity_analysis(J, verbose=True):
    """Complete singularity analysis via SVD.

    Why SVD? Because it decomposes the Jacobian into independent
    'channels' of motion. Each singular value is the gain of one
    channel. When a singular value goes to zero, that channel of
    end-effector motion shuts down — the robot can't move in that
    direction no matter how fast the joints spin.
    """
    U, sigma, Vt = np.linalg.svd(J)
    rank = np.sum(sigma > 1e-10)

    # Condition number: ratio of largest to smallest singular value
    # Measures how 'distorted' the mapping is
    if sigma[-1] > 1e-10:
        condition = sigma[0] / sigma[-1]
    else:
        condition = np.inf

    # Manipulability: product of singular values (Yoshikawa measure)
    manipulability = np.prod(sigma[:rank])

    if verbose:
        print(f"Singular values: {np.round(sigma, 6)}")
        print(f"Rank: {rank} / {min(J.shape)}")
        print(f"Condition number: {condition:.1f}")
        print(f"Manipulability (Yoshikawa): {manipulability:.6f}")

        if rank < min(J.shape):
            print("\n*** SINGULAR CONFIGURATION ***")
            # Identify lost directions
            for i in range(rank, len(sigma)):
                lost_dir = U[:, i]
                print(f"Lost task-space direction: {np.round(lost_dir, 4)}")

    return sigma, rank, condition, manipulability


# Example: 2-link robot at various configurations
l1, l2 = 1.0, 0.8

# Regular configuration
print("=== q = [30, 45] deg (regular) ===")
J_regular = jacobian_2link_analytical(np.radians(30), np.radians(45), l1, l2)
singularity_analysis(J_regular)

# Near-singular
print("\n=== q = [30, 5] deg (near-singular) ===")
J_near = jacobian_2link_analytical(np.radians(30), np.radians(5), l1, l2)
singularity_analysis(J_near)

# Singular (fully extended)
print("\n=== q = [30, 0] deg (SINGULAR) ===")
J_singular = jacobian_2link_analytical(np.radians(30), np.radians(0), l1, l2)
singularity_analysis(J_singular)
```

### Determinant Analysis for Square Jacobians

For a non-redundant robot (square Jacobian), the determinant provides a quick singularity test:

For the 2-link planar robot:

$$\det(J) = l_1 l_2 \sin\theta_2$$

This is zero when $\theta_2 = 0$ (arm extended) or $\theta_2 = \pi$ (arm folded).

```python
def plot_manipulability_landscape(l1, l2, n_points=100):
    """Compute manipulability over the joint space.

    Why map the entire landscape? Because it reveals which regions of
    joint space are well-conditioned (good for control) and which are
    dangerous (near singularity). Motion planners use this information
    to prefer paths through well-conditioned regions.
    """
    theta1_range = np.linspace(-np.pi, np.pi, n_points)
    theta2_range = np.linspace(-np.pi, np.pi, n_points)

    manipulability = np.zeros((n_points, n_points))

    for i, t1 in enumerate(theta1_range):
        for j, t2 in enumerate(theta2_range):
            J = jacobian_2link_analytical(t1, t2, l1, l2)
            manipulability[j, i] = abs(np.linalg.det(J))

    # Report statistics
    print(f"Manipulability range: [{manipulability.min():.4f}, "
          f"{manipulability.max():.4f}]")
    print(f"Max manipulability at theta2 = +/- 90 deg "
          f"(= {l1*l2:.4f})")
    print(f"Zero manipulability at theta2 = 0 or 180 deg")

    return manipulability, theta1_range, theta2_range

w, _, _ = plot_manipulability_landscape(1.0, 0.8)
```

---

## Force/Torque Mapping (Statics Duality)

### The Principle of Virtual Work

One of the most elegant results in robotics: the Jacobian transpose maps end-effector forces to joint torques.

If the end-effector exerts a wrench (force + torque) $\mathbf{F} = (\mathbf{f}, \boldsymbol{\tau}_{ee})^T$ on the environment, the corresponding joint torques are:

$$\boldsymbol{\tau} = J^T(\mathbf{q}) \, \mathbf{F}$$

This is the **statics duality** — the same Jacobian that maps velocities forward maps forces backward.

**Derivation via virtual work**: For virtual displacements $\delta \mathbf{q}$ and $\delta \mathbf{x} = J \delta \mathbf{q}$:

$$\delta W = \mathbf{F}^T \delta \mathbf{x} = \mathbf{F}^T J \delta \mathbf{q} = \boldsymbol{\tau}^T \delta \mathbf{q}$$

Since this holds for all $\delta \mathbf{q}$: $\boldsymbol{\tau} = J^T \mathbf{F}$.

```python
def compute_joint_torques(J, F_ee):
    """Map end-effector wrench to joint torques.

    Why J transpose (not J inverse)? Because the force mapping goes
    in the opposite direction from velocity. Velocity: joints → EE
    uses J. Force: EE → joints uses J^T. This is the statics duality,
    derived from energy conservation (virtual work).

    This is fundamental to force control: if you want the end-effector
    to exert a specific force, compute the required joint torques using J^T.
    """
    return J.T @ F_ee

# Example: 2-link arm pressing down with 10N force
l1, l2 = 1.0, 0.8
q = np.radians([45, 30])

# 2D position Jacobian
J = jacobian_2link_analytical(q[0], q[1], l1, l2)

# Force at end-effector: 10N downward
F_ee = np.array([0, -10])  # Fx=0, Fy=-10 N

# Required joint torques
tau = compute_joint_torques(J, F_ee)
print(f"Configuration: q = [{np.degrees(q[0]):.0f}, {np.degrees(q[1]):.0f}] deg")
print(f"End-effector force: F = {F_ee} N")
print(f"Required joint torques: tau = {np.round(tau, 3)} N*m")

# Physical check: at this configuration, what are the moment arms?
# Joint 1 must support the entire force at a moment arm equal to the
# horizontal distance from joint 1 to the end-effector
x_ee, y_ee = l1*np.cos(q[0]) + l2*np.cos(q[0]+q[1]), \
             l1*np.sin(q[0]) + l2*np.sin(q[0]+q[1])
print(f"\nEnd-effector position: ({x_ee:.4f}, {y_ee:.4f})")
print(f"Moment arm for joint 1: x_ee = {x_ee:.4f} m")
print(f"Expected tau_1: {x_ee * 10:.4f} N*m")  # Should match tau[0]
```

### Force Ellipsoid

The force ellipsoid shows which end-effector forces can be achieved with bounded joint torques ($\|\boldsymbol{\tau}\| \leq 1$):

$$\mathbf{F}^T (J J^T)^{-1} \mathbf{F} \leq 1$$

The axes of the force ellipsoid are aligned with the columns of $U$ from the SVD, and their lengths are the singular values $\sigma_i$.

> **Key insight**: The velocity ellipsoid and force ellipsoid are **inverses** of each other. Directions where the robot moves fast (large $\sigma$) are directions where it can exert little force, and vice versa. This is the robotic analog of a lever: long lever = fast tip speed but small force; short lever = slow tip speed but large force.

---

## Manipulability Ellipsoid

### Definition

The **manipulability ellipsoid** visualizes the end-effector velocities achievable with unit joint velocity ($\|\dot{\mathbf{q}}\| \leq 1$):

$$\mathbf{v}^T (J J^T)^{-1} \mathbf{v} \leq 1$$

The ellipsoid's principal axes are the left singular vectors of $J$, and their semi-axis lengths are the singular values.

```python
def compute_manipulability_ellipse(J):
    """Compute the 2D manipulability ellipse parameters.

    Why an ellipse (not a circle)? Because the Jacobian maps a unit
    sphere in joint space to an ellipsoid in task space. The shape of
    this ellipsoid tells us:
    - Long axis: direction of easy/fast motion
    - Short axis: direction of difficult/slow motion
    - Ratio (condition number): isotropy of the mapping
    - Area (proportional to manipulability): overall dexterity
    """
    U, sigma, Vt = np.linalg.svd(J)

    # Ellipse parameters
    semi_axes = sigma  # lengths of semi-axes
    directions = U[:, :len(sigma)]  # directions of semi-axes
    angle = np.arctan2(U[1, 0], U[0, 0])  # angle of major axis

    return semi_axes, directions, angle

# Visualize at different configurations
l1, l2 = 1.0, 0.8
configs = {
    'Regular (30, 90)': np.radians([30, 90]),
    'Near-singular (30, 10)': np.radians([30, 10]),
    'Symmetric (45, -90)': np.radians([45, -90]),
}

for name, q in configs.items():
    J = jacobian_2link_analytical(q[0], q[1], l1, l2)
    semi_axes, directions, angle = compute_manipulability_ellipse(J)
    w = np.prod(semi_axes)
    cond = semi_axes[0] / max(semi_axes[1], 1e-10)

    print(f"\n{name}:")
    print(f"  Semi-axes: {np.round(semi_axes, 4)}")
    print(f"  Manipulability: {w:.4f}")
    print(f"  Condition number: {cond:.1f}")
    print(f"  Ellipse angle: {np.degrees(angle):.1f} deg")
```

### Dexterity Measures

Several scalar measures quantify manipulability:

| Measure | Formula | Interpretation |
|---------|---------|----------------|
| **Yoshikawa manipulability** | $w = \sqrt{\det(JJ^T)} = \prod \sigma_i$ | Volume of manipulability ellipsoid (0 at singularity) |
| **Condition number** | $\kappa = \sigma_{max} / \sigma_{min}$ | Isotropy ($\kappa = 1$ is ideal; $\kappa = \infty$ at singularity) |
| **Minimum singular value** | $\sigma_{min}$ | Worst-case velocity gain |
| **Isotropy index** | $1/\kappa$ | Normalized isotropy ($1$ = isotropic, $0$ = singular) |

```python
def dexterity_measures(J):
    """Compute all standard dexterity measures.

    Why multiple measures? Because no single number captures all
    aspects of dexterity. Yoshikawa's measure can be high even if
    the ellipsoid is very elongated (good in one direction, bad in
    another). The condition number captures isotropy but not overall
    magnitude. Use them together for a complete picture.
    """
    U, sigma, Vt = np.linalg.svd(J)

    measures = {
        'yoshikawa': np.prod(sigma),
        'condition_number': sigma[0] / max(sigma[-1], 1e-15),
        'min_singular_value': sigma[-1],
        'isotropy': sigma[-1] / max(sigma[0], 1e-15),
    }

    return measures

# Optimal configuration for isotropy (2-link planar)
# The condition number is minimized (isotropy maximized) when the
# manipulability ellipse is closest to a circle.
# For a 2-link robot, this happens when theta2 = +/- 90 degrees
# and additionally depends on the link length ratio.

best_config = None
best_isotropy = 0

for t1 in np.linspace(-np.pi, np.pi, 100):
    for t2 in np.linspace(-np.pi, np.pi, 100):
        J = jacobian_2link_analytical(t1, t2, l1, l2)
        m = dexterity_measures(J)
        if m['isotropy'] > best_isotropy:
            best_isotropy = m['isotropy']
            best_config = (t1, t2)

print(f"\nBest isotropy: {best_isotropy:.4f} at "
      f"q = ({np.degrees(best_config[0]):.1f}, {np.degrees(best_config[1]):.1f}) deg")
```

---

## Jacobian for a 6-DOF Robot

For a complete 6-DOF manipulator, the Jacobian is 6x6. Here is a full implementation:

```python
def jacobian_6dof(dh_params, q):
    """Complete 6x6 Jacobian for a 6-DOF serial manipulator.

    This function extracts joint axes and origins from FK,
    then applies the geometric Jacobian formula for each joint.

    For real-time control at 1 kHz, this computation takes
    roughly 10-50 microseconds on modern hardware — fast enough.
    """
    n = len(dh_params)
    assert n == 6, "This function is for 6-DOF robots"

    # Compute FK to get all frame transforms
    T = np.eye(4)
    transforms = [T.copy()]

    for i in range(n):
        p = dh_params[i]
        ct = np.cos(p['theta'] + (q[i] if p['type'] == 'revolute' else 0))
        st = np.sin(p['theta'] + (q[i] if p['type'] == 'revolute' else 0))
        d = p['d'] + (q[i] if p['type'] == 'prismatic' else 0)
        ca = np.cos(p['alpha'])
        sa = np.sin(p['alpha'])
        a = p['a']

        Ti = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [ 0,     sa,     ca,    d],
            [ 0,      0,      0,    1]
        ])
        T = T @ Ti
        transforms.append(T.copy())

    p_end = transforms[-1][:3, 3]
    J = np.zeros((6, n))

    for i in range(n):
        z_i = transforms[i][:3, 2]
        p_i = transforms[i][:3, 3]

        if dh_params[i]['type'] == 'revolute':
            J[:3, i] = np.cross(z_i, p_end - p_i)
            J[3:, i] = z_i
        else:
            J[:3, i] = z_i
            J[3:, i] = 0

    return J

# PUMA 560-like DH parameters
puma_dh = [
    {'theta': 0, 'd': 0,    'a': 0,    'alpha': -np.pi/2, 'type': 'revolute'},
    {'theta': 0, 'd': 0,    'a': 0.4318, 'alpha': 0,       'type': 'revolute'},
    {'theta': 0, 'd': 0.15, 'a': 0.0203, 'alpha': -np.pi/2, 'type': 'revolute'},
    {'theta': 0, 'd': 0.4318, 'a': 0, 'alpha': np.pi/2,    'type': 'revolute'},
    {'theta': 0, 'd': 0,    'a': 0,    'alpha': -np.pi/2, 'type': 'revolute'},
    {'theta': 0, 'd': 0,    'a': 0,    'alpha': 0,        'type': 'revolute'},
]

q = np.radians([30, -45, 60, 0, 30, 0])
J_puma = jacobian_6dof(puma_dh, q)

print("PUMA 560 Jacobian (6x6):")
print(np.round(J_puma, 4))

print("\nSingularity analysis:")
sigma, rank, cond, manip = singularity_analysis(J_puma)
```

---

## Summary

- The **Jacobian** $J(\mathbf{q})$ is the fundamental velocity mapping: $\dot{\mathbf{x}} = J \dot{\mathbf{q}}$
- For **revolute** joints, each Jacobian column involves $\hat{z} \times (\mathbf{p}_{ee} - \mathbf{p}_{joint})$ (linear) and $\hat{z}$ (angular)
- For **prismatic** joints, the column is $\hat{z}$ (linear) and $\mathbf{0}$ (angular)
- The **geometric Jacobian** gives spatial twist; the **analytical Jacobian** gives Euler angle rates
- **Singularities** ($\det(J) = 0$) represent lost DOF — the robot cannot move in certain directions
- **Statics duality**: $\boldsymbol{\tau} = J^T \mathbf{F}$ maps end-effector forces to joint torques
- The **manipulability ellipsoid** visualizes the velocity/force capability at each configuration
- **SVD** of the Jacobian provides the most complete analysis of robot behavior

---

## Exercises

### Exercise 1: Jacobian Derivation

For a 3-link planar robot with link lengths $l_1$, $l_2$, $l_3$:
1. Derive the 2x3 position Jacobian analytically
2. Compute the Jacobian at $q = (30°, 45°, -60°)$ for $l_1 = l_2 = l_3 = 0.5$ m
3. Verify your result using finite differences
4. What is the null space at this configuration? What physical motion does it represent?

### Exercise 2: Singularity Analysis

For the 2-link planar robot ($l_1 = 1.0$, $l_2 = 0.8$):
1. Find all singular configurations analytically from $\det(J) = 0$
2. At $q = (45°, 0°)$: compute the SVD and identify the lost direction of motion
3. Attempt to move the end-effector in the lost direction using the pseudo-inverse — what happens to joint velocities?
4. Repeat with DLS (different $\lambda$ values) and compare

### Exercise 3: Force Mapping

A 2-link planar arm ($l_1 = l_2 = 0.5$ m) at $q = (0°, 90°)$ must exert a 20 N force pointing to the right (+x direction) at the end-effector.
1. Compute the required joint torques using $\tau = J^T F$
2. If joint 1 has a maximum torque of 15 N*m, can this force be achieved?
3. What is the maximum force in the +x direction the robot can exert at this configuration (assuming $|\tau_i| \leq 15$ N*m)?

### Exercise 4: Manipulability Ellipsoid

For the 2-link robot ($l_1 = 1.0$, $l_2 = 0.5$):
1. Compute and compare manipulability ellipses at: $\theta_2 = 30°$, $60°$, $90°$, $120°$, $150°$ (with $\theta_1 = 0$)
2. At which $\theta_2$ is the ellipse most circular (best isotropy)?
3. Does the optimal $\theta_2$ for isotropy depend on $\theta_1$? Why or why not?

### Exercise 5: Numerical Jacobian Verification

Implement a general numerical Jacobian (using finite differences) for the PUMA 560-like robot:
1. Compare with the geometric Jacobian at 5 random configurations
2. What step size $\delta$ gives the best accuracy? (Try $10^{-4}$ to $10^{-10}$)
3. Why does accuracy degrade for very small $\delta$? (Hint: floating-point arithmetic)

---

[← Previous: Inverse Kinematics](04_Inverse_Kinematics.md) | [Next: Robot Dynamics →](06_Robot_Dynamics.md)
