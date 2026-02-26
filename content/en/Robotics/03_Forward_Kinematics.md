# Forward Kinematics

[← Previous: Rigid Body Transformations](02_Rigid_Body_Transformations.md) | [Next: Inverse Kinematics →](04_Inverse_Kinematics.md)

## Learning Objectives

1. Describe kinematic chains and distinguish between revolute and prismatic joint types
2. Apply the Denavit-Hartenberg (DH) convention to systematically assign coordinate frames and extract the four DH parameters for each joint
3. Build individual transformation matrices from DH parameters and multiply them to obtain the forward kinematics
4. Compute the forward kinematics for common robot configurations: 2-link planar, 3-DOF spatial, and 6-DOF manipulators
5. Analyze the workspace of a manipulator by sweeping through joint configurations
6. Implement a general-purpose FK solver in Python

---

## Why This Matters

Forward kinematics answers the most fundamental question in robotics: **"If I set my joints to these angles, where does the end-effector end up?"** Every time a robot moves, the controller computes FK thousands of times per second to track where the tool actually is. Without FK, you have a collection of motors spinning blindly — with FK, you have a machine that knows its own geometry.

FK is also the foundation for everything that follows. Inverse kinematics (Lesson 4) inverts FK. The Jacobian (Lesson 5) is the derivative of FK. Dynamics (Lesson 6) requires FK to compute link positions and velocities. Understanding FK deeply — not just the formula, but the systematic procedure for deriving it — is essential for any robotics practitioner.

> **Analogy**: DH parameters are like GPS coordinates for robot joints — 4 numbers uniquely place each joint relative to the previous one. Just as GPS uses (latitude, longitude, altitude, heading) to pinpoint a location on Earth, DH uses $(d, \theta, a, \alpha)$ to pinpoint one joint frame relative to the next in a kinematic chain.

---

## Kinematic Chains

### Links and Joints

A **kinematic chain** is a series of rigid bodies (**links**) connected by **joints**. The two fundamental joint types in robotics are:

| Joint Type | Motion | DOF | Variable | Symbol |
|-----------|--------|-----|----------|--------|
| **Revolute (R)** | Rotation about an axis | 1 | Angle $\theta$ | ![revolute](revolute) |
| **Prismatic (P)** | Translation along an axis | 1 | Displacement $d$ | ![prismatic](prismatic) |

Other joint types can be decomposed into combinations of these:
- **Universal joint** (U) = 2 revolute joints with intersecting, perpendicular axes (2 DOF)
- **Spherical joint** (S) = 3 revolute joints with coincident axes (3 DOF)
- **Cylindrical joint** (C) = 1 revolute + 1 prismatic on the same axis (2 DOF)

### Open vs Closed Chains

- **Open chain** (serial): each link is connected to at most two other links. The base is fixed, and the end-effector is free. (Example: industrial robot arm)
- **Closed chain** (parallel): at least one link is connected to more than two others, forming loops. (Example: Stewart platform)

This lesson focuses on open chains, where FK is straightforward via chain multiplication.

### Numbering Convention

For an $n$-DOF serial manipulator:
- **Links**: numbered $0$ (base/ground) to $n$ (end-effector link)
- **Joints**: numbered $1$ to $n$; joint $i$ connects link $i-1$ to link $i$
- **Frames**: frame $\{i\}$ is attached to link $i$

```
{0} ──[Joint 1]── {1} ──[Joint 2]── {2} ── ... ──[Joint n]── {n}
base                                                        end-effector
```

---

## The Denavit-Hartenberg (DH) Convention

### Motivation

For an $n$-DOF robot, the FK requires describing the transformation from frame $\{i-1\}$ to frame $\{i\}$ for each joint. Without a systematic convention, every robot would require custom derivation. The DH convention standardizes this with exactly **4 parameters per joint**.

### The Four DH Parameters

For joint $i$ connecting link $i-1$ to link $i$:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| **Link length** | $a_i$ | Distance from $z_{i-1}$ to $z_i$ along $x_i$ |
| **Link twist** | $\alpha_i$ | Angle from $z_{i-1}$ to $z_i$ about $x_i$ |
| **Link offset** | $d_i$ | Distance from $x_{i-1}$ to $x_i$ along $z_{i-1}$ |
| **Joint angle** | $\theta_i$ | Angle from $x_{i-1}$ to $x_i$ about $z_{i-1}$ |

For a **revolute** joint, $\theta_i$ is the joint variable (the one that changes).
For a **prismatic** joint, $d_i$ is the joint variable.

### DH Frame Assignment Rules

1. **$z_i$ axis**: aligned with the axis of joint $i+1$
2. **$x_i$ axis**: along the common normal from $z_{i-1}$ to $z_i$ (i.e., $x_i = z_{i-1} \times z_i / \|z_{i-1} \times z_i\|$)
3. **$y_i$ axis**: completes the right-handed frame ($y_i = z_i \times x_i$)
4. **Origin of frame $\{i\}$**: at the intersection of $z_i$ and $x_i$

**Special cases**:
- If $z_{i-1}$ and $z_i$ are parallel: $x_i$ direction is ambiguous; choose for simplicity (typically along the line connecting origins)
- If $z_{i-1}$ and $z_i$ intersect: $x_i$ is perpendicular to both (their cross product)

### Step-by-Step DH Procedure

1. Number all joints from 1 to $n$
2. Assign $z_i$ axes along each joint axis
3. Assign $x_i$ axes according to the common normal rules
4. Locate frame origins at the intersection of $x_i$ and $z_i$
5. Read off the four DH parameters for each joint
6. Build the transformation matrix for each joint
7. Multiply all matrices together

### The DH Transformation Matrix

The transformation from frame $\{i-1\}$ to frame $\{i\}$ is:

$${}^{i-1}_{i}T = Rot_z(\theta_i) \cdot Trans_z(d_i) \cdot Trans_x(a_i) \cdot Rot_x(\alpha_i)$$

In matrix form:

$${}^{i-1}_{i}T = \begin{bmatrix} \cos\theta_i & -\sin\theta_i \cos\alpha_i & \sin\theta_i \sin\alpha_i & a_i \cos\theta_i \\ \sin\theta_i & \cos\theta_i \cos\alpha_i & -\cos\theta_i \sin\alpha_i & a_i \sin\theta_i \\ 0 & \sin\alpha_i & \cos\alpha_i & d_i \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

```python
import numpy as np

def dh_transform(theta, d, a, alpha):
    """Compute the 4x4 homogeneous transformation from DH parameters.

    Why this specific order (Rz, Tz, Tx, Rx)? It follows from the DH
    convention: first rotate about z_{i-1} by theta, then translate along
    z_{i-1} by d, then translate along x_i by a, then rotate about x_i
    by alpha. This specific decomposition ensures that exactly 4 parameters
    suffice for any joint configuration.

    Parameters:
        theta: joint angle (revolute variable) [rad]
        d: link offset (prismatic variable) [m]
        a: link length [m]
        alpha: link twist [rad]

    Returns:
        4x4 homogeneous transformation matrix
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ])
```

---

## Example 1: 2-Link Planar Robot

The simplest non-trivial manipulator: two revolute joints in a plane.

```
        Joint 2
          o───────── End-effector
         /  Link 2
        /   (length l2)
Joint 1
  o────────o
  |  Link 1
  |  (length l1)
  |
 ===  Base (fixed)
```

### DH Parameters

| Joint $i$ | $\theta_i$ | $d_i$ | $a_i$ | $\alpha_i$ |
|-----------|-----------|-------|-------|------------|
| 1 | $\theta_1$ (variable) | 0 | $l_1$ | 0 |
| 2 | $\theta_2$ (variable) | 0 | $l_2$ | 0 |

All $\alpha_i = 0$ (all joint axes are parallel, pointing out of the plane) and all $d_i = 0$ (planar — no offset along $z$).

### Forward Kinematics

$${}^{0}_{2}T = {}^{0}_{1}T \cdot {}^{1}_{2}T$$

The end-effector position:

$$x = l_1 \cos\theta_1 + l_2 \cos(\theta_1 + \theta_2)$$
$$y = l_1 \sin\theta_1 + l_2 \sin(\theta_1 + \theta_2)$$
$$\phi = \theta_1 + \theta_2 \quad \text{(end-effector orientation)}$$

```python
def fk_2link_planar(theta1, theta2, l1, l2):
    """Forward kinematics for a 2-link planar robot.

    Why derive the closed-form? For simple robots, the closed-form is
    faster than matrix multiplication and gives physical insight.
    But for complex robots, we always use the general DH approach.
    """
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    phi = theta1 + theta2  # orientation angle

    return x, y, phi

def fk_2link_planar_dh(theta1, theta2, l1, l2):
    """Same FK using the general DH method (for verification).

    Why verify both ways? In robotics, cross-checking is essential.
    A bug in FK propagates to every downstream computation.
    """
    T01 = dh_transform(theta1, 0, l1, 0)
    T12 = dh_transform(theta2, 0, l2, 0)
    T02 = T01 @ T12
    x, y = T02[0, 3], T02[1, 3]
    phi = np.arctan2(T02[1, 0], T02[0, 0])
    return x, y, phi

# Test with specific joint angles
l1, l2 = 1.0, 0.8  # link lengths in meters
theta1 = np.radians(30)
theta2 = np.radians(45)

x1, y1, phi1 = fk_2link_planar(theta1, theta2, l1, l2)
x2, y2, phi2 = fk_2link_planar_dh(theta1, theta2, l1, l2)

print(f"Closed-form: x={x1:.4f}, y={y1:.4f}, phi={np.degrees(phi1):.1f} deg")
print(f"DH method:   x={x2:.4f}, y={y2:.4f}, phi={np.degrees(phi2):.1f} deg")
print(f"Match: {np.allclose([x1,y1,phi1], [x2,y2,phi2])}")
```

---

## Example 2: 3-DOF Spatial Robot (RPR)

A more interesting example: Revolute-Prismatic-Revolute configuration with non-zero $\alpha$ values.

### DH Parameters

| Joint $i$ | $\theta_i$ | $d_i$ | $a_i$ | $\alpha_i$ |
|-----------|-----------|-------|-------|------------|
| 1 | $\theta_1$ (var) | $d_1$ | 0 | $-90°$ |
| 2 | $-90°$ | $d_2$ (var) | 0 | $90°$ |
| 3 | $\theta_3$ (var) | 0 | 0 | 0 |

```python
def fk_3dof_rpr(theta1, d2, theta3, d1=0.5):
    """FK for a 3-DOF RPR spatial manipulator.

    Why this configuration? RPR (Revolute-Prismatic-Revolute) is common
    in cylindrical robots. The first revolute provides base rotation,
    the prismatic joint provides vertical reach, and the second revolute
    provides wrist rotation.
    """
    T01 = dh_transform(theta1, d1, 0, -np.pi/2)
    T12 = dh_transform(-np.pi/2, d2, 0, np.pi/2)
    T23 = dh_transform(theta3, 0, 0, 0)

    T03 = T01 @ T12 @ T23

    position = T03[:3, 3]
    orientation = T03[:3, :3]

    return T03, position, orientation

# Example configuration
T, pos, ori = fk_3dof_rpr(
    theta1=np.radians(30),
    d2=0.4,
    theta3=np.radians(60)
)
print(f"End-effector position: {np.round(pos, 4)}")
print(f"End-effector orientation:\n{np.round(ori, 4)}")
```

---

## Example 3: 6-DOF Articulated Robot (PUMA-like)

The standard industrial robot: 6 revolute joints arranged in a specific geometry. This is similar to the classic PUMA 560 configuration.

### DH Parameters (PUMA 560-like)

| Joint | $\theta_i$ | $d_i$ | $a_i$ | $\alpha_i$ |
|-------|-----------|-------|-------|------------|
| 1 | $\theta_1$ | 0 | 0 | $-90°$ |
| 2 | $\theta_2$ | 0 | $a_2$ | 0 |
| 3 | $\theta_3$ | $d_3$ | $a_3$ | $-90°$ |
| 4 | $\theta_4$ | $d_4$ | 0 | $90°$ |
| 5 | $\theta_5$ | 0 | 0 | $-90°$ |
| 6 | $\theta_6$ | 0 | 0 | 0 |

Joints 4, 5, 6 form a **spherical wrist** — their axes intersect at a common point. This is an extremely important design choice because it **decouples** position from orientation, making inverse kinematics tractable (Lesson 4).

```python
class SerialRobot:
    """General-purpose serial robot with DH parameterization.

    Why a class? Because we'll reuse this for FK, IK, Jacobian, and dynamics.
    Encapsulating the DH parameters and FK computation in one place avoids
    duplicated code and ensures consistency.
    """
    def __init__(self, name, dh_params):
        """
        dh_params: list of dicts, each with keys:
            'theta': nominal angle (for revolute: this is the offset, variable added at runtime)
            'd': link offset
            'a': link length
            'alpha': link twist
            'type': 'revolute' or 'prismatic'
        """
        self.name = name
        self.dh_params = dh_params
        self.n_joints = len(dh_params)

    def fk(self, q):
        """Compute forward kinematics for joint values q.

        Parameters:
            q: array of joint values (angles for revolute, displacements for prismatic)

        Returns:
            T: 4x4 homogeneous transformation (base to end-effector)
            transforms: list of intermediate transforms [{0}_T_{1}, {0}_T_{2}, ...]

        Why return intermediate transforms? They're needed for:
        - Jacobian computation (Lesson 5)
        - Dynamics computation (Lesson 6)
        - Visualization
        - Collision checking
        """
        assert len(q) == self.n_joints, \
            f"Expected {self.n_joints} joint values, got {len(q)}"

        T = np.eye(4)
        transforms = []

        for i, (params, qi) in enumerate(zip(self.dh_params, q)):
            if params['type'] == 'revolute':
                theta = params['theta'] + qi  # qi is the joint angle
                d = params['d']
            else:  # prismatic
                theta = params['theta']
                d = params['d'] + qi  # qi is the displacement

            Ti = dh_transform(theta, d, params['a'], params['alpha'])
            T = T @ Ti
            transforms.append(T.copy())

        return T, transforms

    def joint_positions(self, q):
        """Get 3D positions of all joints (for visualization).

        Returns list of (x, y, z) positions from base to end-effector.
        """
        _, transforms = self.fk(q)
        positions = [np.array([0, 0, 0])]  # base position
        for T in transforms:
            positions.append(T[:3, 3])
        return positions


# Define PUMA 560-like robot
puma_dh = [
    {'theta': 0, 'd': 0,    'a': 0,    'alpha': -np.pi/2, 'type': 'revolute'},
    {'theta': 0, 'd': 0,    'a': 0.4318, 'alpha': 0,       'type': 'revolute'},
    {'theta': 0, 'd': 0.15, 'a': 0.0203, 'alpha': -np.pi/2, 'type': 'revolute'},
    {'theta': 0, 'd': 0.4318, 'a': 0, 'alpha': np.pi/2,    'type': 'revolute'},
    {'theta': 0, 'd': 0,    'a': 0,    'alpha': -np.pi/2, 'type': 'revolute'},
    {'theta': 0, 'd': 0,    'a': 0,    'alpha': 0,        'type': 'revolute'},
]

puma = SerialRobot("PUMA 560", puma_dh)

# Home configuration (all zeros)
q_home = np.zeros(6)
T_home, _ = puma.fk(q_home)
print(f"PUMA 560 home position: {np.round(T_home[:3, 3], 4)}")

# Some non-trivial configuration
q_test = np.radians([30, -45, 60, 0, 30, 0])
T_test, _ = puma.fk(q_test)
print(f"PUMA 560 test position: {np.round(T_test[:3, 3], 4)}")
print(f"Test orientation:\n{np.round(T_test[:3, :3], 4)}")
```

---

## Workspace Analysis

The **workspace** of a robot is the set of all positions reachable by the end-effector. We can visualize it by sweeping through joint configurations.

### Types of Workspace

- **Reachable workspace**: All points the end-effector tip can reach with *at least one* orientation
- **Dexterous workspace**: All points reachable with *every* orientation (subset of reachable workspace)
- **Cross-section workspace**: A 2D slice of the 3D workspace (useful for visualization)

### Workspace Computation

```python
def compute_workspace_2d(robot, n_samples=50):
    """Sample the workspace of a planar robot by grid search.

    Why grid search? For 2-DOF robots, uniform grid sampling gives
    a clear picture. For higher DOF, random sampling is more efficient
    because the grid grows exponentially.
    """
    positions = []

    # Create grid of joint values
    for i in range(robot.n_joints):
        pass  # We'll create ranges below

    # For a 2-DOF planar robot
    if robot.n_joints == 2:
        for q1 in np.linspace(-np.pi, np.pi, n_samples):
            for q2 in np.linspace(-np.pi, np.pi, n_samples):
                T, _ = robot.fk(np.array([q1, q2]))
                positions.append(T[:3, 3])

    return np.array(positions)

def compute_workspace_random(robot, joint_limits, n_samples=10000):
    """Sample workspace by random joint configurations.

    Why random sampling? For robots with > 3 DOF, grid sampling is
    impractical (e.g., 50 samples per joint x 6 joints = 15.6 billion
    configurations). Random sampling provides a reasonable approximation.
    """
    positions = []

    for _ in range(n_samples):
        q = np.array([
            np.random.uniform(lo, hi)
            for lo, hi in joint_limits
        ])
        T, _ = robot.fk(q)
        positions.append(T[:3, 3])

    return np.array(positions)

# Define 2-link planar robot using our SerialRobot class
planar_2link_dh = [
    {'theta': 0, 'd': 0, 'a': 1.0, 'alpha': 0, 'type': 'revolute'},
    {'theta': 0, 'd': 0, 'a': 0.8, 'alpha': 0, 'type': 'revolute'},
]
planar_2link = SerialRobot("2-Link Planar", planar_2link_dh)

# Compute workspace boundaries analytically
l1, l2 = 1.0, 0.8
r_max = l1 + l2      # fully extended
r_min = abs(l1 - l2)  # fully folded
print(f"Workspace: annular ring with r_min={r_min}, r_max={r_max}")
print(f"(When both joints have full rotation range)")

# Numerical verification
positions = compute_workspace_random(
    planar_2link,
    joint_limits=[(-np.pi, np.pi), (-np.pi, np.pi)],
    n_samples=5000
)
distances = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
print(f"Sampled distance range: [{distances.min():.3f}, {distances.max():.3f}]")
```

### Workspace Factors

Several factors affect the workspace:

1. **Link lengths**: Longer links = larger workspace
2. **Joint limits**: Reduced range = smaller workspace
3. **Joint types**: Prismatic joints extend workspace linearly; revolute joints create curved boundaries
4. **Number of DOF**: More DOF = generally larger dexterous workspace
5. **Configuration**: Serial robots have large workspace-to-footprint ratio; parallel robots have smaller relative workspace

---

## Modified DH Convention

There are actually **two** DH conventions in common use:

| Aspect | Standard (Classic) DH | Modified (Craig) DH |
|--------|----------------------|---------------------|
| Frame placement | Frame $\{i\}$ at the end of link $i$ | Frame $\{i\}$ at the beginning of link $i$ |
| Transform order | $Rot_z \cdot Trans_z \cdot Trans_x \cdot Rot_x$ | $Rot_x \cdot Trans_x \cdot Rot_z \cdot Trans_z$ |
| Reference | Denavit & Hartenberg (1955) | Craig's textbook |

The modified DH convention (Craig) reverses the order:

$${}^{i-1}_{i}T = Rot_x(\alpha_{i-1}) \cdot Trans_x(a_{i-1}) \cdot Rot_z(\theta_i) \cdot Trans_z(d_i)$$

```python
def dh_transform_modified(alpha_prev, a_prev, theta, d):
    """Modified DH transformation (Craig convention).

    Why two conventions? Historical reasons. Both give the same final
    result for FK if parameters are assigned consistently. However,
    the modified convention places frame {i} at the proximal end of
    link i, which some find more intuitive for dynamics computation.

    IMPORTANT: Never mix conventions! Check which one a paper or
    textbook uses before copying DH parameters.
    """
    ca = np.cos(alpha_prev)
    sa = np.sin(alpha_prev)
    ct = np.cos(theta)
    st = np.sin(theta)

    return np.array([
        [     ct,      -st,   0,    a_prev],
        [st * ca,  ct * ca, -sa, -sa * d  ],
        [st * sa,  ct * sa,  ca,  ca * d  ],
        [      0,        0,   0,        1 ]
    ])
```

> **Warning**: The two conventions use different parameter assignments for the same robot. Never mix DH parameters from different sources without checking the convention.

---

## Verifying Forward Kinematics

Verification is critical in robotics — a bug in FK propagates to every downstream computation.

### Verification Strategies

```python
def verify_fk(robot, test_cases):
    """Verify FK against known configurations.

    Why formal verification? Because FK bugs can cause real robots to
    crash into obstacles or themselves. These test cases are like unit
    tests for your robot model.
    """
    all_passed = True

    for name, q, expected_pos, tol in test_cases:
        T, _ = robot.fk(q)
        actual_pos = T[:3, 3]
        error = np.linalg.norm(actual_pos - expected_pos)

        if error > tol:
            print(f"FAIL: {name} - error={error:.6f} > tol={tol}")
            print(f"  Expected: {expected_pos}")
            print(f"  Actual:   {np.round(actual_pos, 6)}")
            all_passed = False
        else:
            print(f"PASS: {name} - error={error:.6f}")

    return all_passed

# Test cases for 2-link planar robot
test_cases = [
    # (name, q, expected_position, tolerance)
    ("Home (both zero)", np.array([0, 0]),
     np.array([1.8, 0, 0]), 1e-10),

    ("Joint 1 = 90 deg, Joint 2 = 0", np.array([np.pi/2, 0]),
     np.array([0, 1.8, 0]), 1e-10),

    ("Both 90 deg", np.array([np.pi/2, np.pi/2]),
     np.array([-0.8, 1.0, 0]), 1e-10),

    ("Fully folded back", np.array([0, np.pi]),
     np.array([0.2, 0, 0]), 1e-10),
]

verify_fk(planar_2link, test_cases)
```

### Consistency Checks

1. **Home position**: At $q = 0$, the result should match the "zero configuration" geometry
2. **Single joint motion**: Moving one joint at a time should produce predictable motion
3. **Symmetry**: For symmetric robots, symmetric joint configurations should give symmetric poses
4. **Workspace boundaries**: Fully extended and fully retracted should match $l_1 + l_2$ and $|l_1 - l_2|$
5. **Cross-validation**: Compare with a known FK implementation (e.g., Robotics Toolbox for Python)

---

## General FK Solver

Putting it all together, here is a complete, reusable FK implementation:

```python
class RobotFK:
    """Complete forward kinematics solver with DH parameterization.

    This class supports both standard and modified DH conventions,
    arbitrary combinations of revolute and prismatic joints,
    and provides intermediate frame transforms for downstream use.
    """
    def __init__(self, dh_params, convention='standard'):
        """
        dh_params: list of dicts with keys:
            For standard DH: 'theta', 'd', 'a', 'alpha', 'type'
            For modified DH: 'alpha_prev', 'a_prev', 'theta', 'd', 'type'
        convention: 'standard' or 'modified'
        """
        self.dh_params = dh_params
        self.convention = convention
        self.n = len(dh_params)

    def compute(self, q):
        """Compute FK for given joint configuration.

        Returns:
            T_0n: base-to-end-effector transform
            T_list: list of base-to-frame-i transforms (i=1..n)
            T_individual: list of frame-to-frame transforms (i-1 to i)
        """
        T_0i = np.eye(4)
        T_list = []
        T_individual = []

        for i in range(self.n):
            p = self.dh_params[i]

            if self.convention == 'standard':
                theta = p['theta'] + (q[i] if p['type'] == 'revolute' else 0)
                d = p['d'] + (q[i] if p['type'] == 'prismatic' else 0)
                Ti = dh_transform(theta, d, p['a'], p['alpha'])
            else:
                theta = p['theta'] + (q[i] if p['type'] == 'revolute' else 0)
                d = p['d'] + (q[i] if p['type'] == 'prismatic' else 0)
                Ti = dh_transform_modified(p['alpha_prev'], p['a_prev'], theta, d)

            T_individual.append(Ti)
            T_0i = T_0i @ Ti
            T_list.append(T_0i.copy())

        return T_0i, T_list, T_individual

    def end_effector_pose(self, q):
        """Convenience method: returns position and rotation matrix."""
        T, _, _ = self.compute(q)
        return T[:3, 3], T[:3, :3]

    def joint_origins(self, q):
        """Get all joint origin positions in the base frame."""
        _, T_list, _ = self.compute(q)
        origins = [np.zeros(3)]  # base
        for T in T_list:
            origins.append(T[:3, 3])
        return origins

    def print_joint_frames(self, q):
        """Print all intermediate frame poses (useful for debugging)."""
        _, T_list, _ = self.compute(q)
        for i, T in enumerate(T_list):
            pos = T[:3, 3]
            # Extract ZYX Euler angles for readability
            R = T[:3, :3]
            beta = -np.arcsin(np.clip(R[2, 0], -1, 1))
            alpha = np.arctan2(R[1, 0], R[0, 0])
            gamma = np.arctan2(R[2, 1], R[2, 2])
            print(f"Frame {i+1}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), "
                  f"rpy=({np.degrees(gamma):.1f}, {np.degrees(beta):.1f}, "
                  f"{np.degrees(alpha):.1f}) deg")


# Usage example: SCARA robot (RRP + R for wrist)
scara_dh = [
    {'theta': 0, 'd': 0.4, 'a': 0.35, 'alpha': 0, 'type': 'revolute'},
    {'theta': 0, 'd': 0,   'a': 0.25, 'alpha': np.pi, 'type': 'revolute'},
    {'theta': 0, 'd': 0,   'a': 0,    'alpha': 0, 'type': 'prismatic'},
    {'theta': 0, 'd': 0,   'a': 0,    'alpha': 0, 'type': 'revolute'},
]

scara = RobotFK(scara_dh, convention='standard')

q_scara = np.array([np.radians(30), np.radians(-45), 0.1, np.radians(90)])
T_scara, _, _ = scara.compute(q_scara)
print("\nSCARA FK result:")
scara.print_joint_frames(q_scara)
print(f"\nEnd-effector:\n{np.round(T_scara, 4)}")
```

---

## Common Pitfalls

### 1. Convention Confusion

Mixing standard and modified DH parameters is the most common FK error. Always verify the convention before using DH tables from any source.

### 2. Angle Units

DH tables may list angles in degrees, but trigonometric functions require radians. Always convert.

### 3. Frame Assignment Ambiguity

When consecutive $z$ axes are parallel, the $x$ axis direction is arbitrary. Different choices lead to different DH parameters but the same FK result (the parameters compensate).

### 4. Tool Frame

The DH convention gives the transformation to the last joint frame, not the tool. Add a **tool transform** $T_{tool}$ at the end:

$$T_{0,tool} = T_{0,n} \cdot T_{n,tool}$$

```python
# Adding a tool frame
# The tool tip may be offset from the last joint frame
T_tool = np.eye(4)
T_tool[2, 3] = 0.1  # tool extends 10 cm along z of last frame

T_0n, _, _ = scara.compute(q_scara)
T_0_tool = T_0n @ T_tool
print(f"Last joint position:  {np.round(T_0n[:3, 3], 4)}")
print(f"Tool tip position:    {np.round(T_0_tool[:3, 3], 4)}")
```

---

## Summary

- **Kinematic chains** consist of links connected by revolute (rotation) or prismatic (translation) joints
- The **DH convention** assigns coordinate frames using 4 parameters per joint: $(\theta, d, a, \alpha)$
- For revolute joints, $\theta$ is the variable; for prismatic joints, $d$ is the variable
- **Forward kinematics** is the product of all joint transformation matrices: $T_{0n} = T_{01} \cdot T_{12} \cdots T_{(n-1)n}$
- The **workspace** is the set of all reachable end-effector positions, determined by link lengths, joint types, and joint limits
- Two DH conventions exist (standard and modified) — **never mix them**
- Always **verify** FK with known test configurations before using it in a real system

---

## Exercises

### Exercise 1: DH Parameter Assignment

For a 3-DOF planar robot (3 revolute joints, all in the same plane) with link lengths $l_1 = 0.5$ m, $l_2 = 0.4$ m, $l_3 = 0.3$ m:
1. Draw the robot and assign DH frames
2. Write the DH parameter table
3. Compute the FK symbolically (end-effector $x$, $y$, $\phi$)
4. Verify numerically for $\theta_1 = 30°$, $\theta_2 = 45°$, $\theta_3 = -30°$

### Exercise 2: SCARA Robot

A SCARA robot has the DH parameters:

| Joint | $\theta$ | $d$ | $a$ | $\alpha$ | Type |
|-------|---------|-----|-----|---------|------|
| 1 | $\theta_1$ | 0.5 | 0.4 | 0 | R |
| 2 | $\theta_2$ | 0 | 0.3 | $\pi$ | R |
| 3 | 0 | $d_3$ | 0 | 0 | P |
| 4 | $\theta_4$ | 0 | 0 | 0 | R |

1. Implement FK for this robot
2. Compute the end-effector pose for $\theta_1 = 45°$, $\theta_2 = -30°$, $d_3 = 0.15$ m, $\theta_4 = 60°$
3. What is the workspace shape? Sketch a top-down view

### Exercise 3: 6-DOF Verification

Using the PUMA 560-like robot defined in the lesson:
1. Compute FK for $q = [0, -90°, 0, 0, 0, 0]$ (arm pointing straight up)
2. Compute FK for $q = [90°, 0, 0, 0, 0, 0]$ (base rotated 90 degrees)
3. For each case, verify that the result makes geometric sense

### Exercise 4: Workspace Visualization

1. For the 2-link planar robot ($l_1 = 1.0$, $l_2 = 0.8$), sample the workspace using a grid of joint angles
2. Plot the sampled points to visualize the annular workspace
3. Now restrict joint 2 to $[-90°, 90°]$. Plot the new workspace
4. Calculate the area of both workspaces (numerically)

### Exercise 5: Mixed Joint Robot

Design a 3-DOF robot with the following structure: Revolute-Prismatic-Revolute.
1. Choose DH parameters that give a useful workspace (you decide what "useful" means)
2. Implement FK and verify with at least 3 test configurations
3. Compare the workspace shape with the all-revolute 3-DOF planar robot from Exercise 1

---

[← Previous: Rigid Body Transformations](02_Rigid_Body_Transformations.md) | [Next: Inverse Kinematics →](04_Inverse_Kinematics.md)
