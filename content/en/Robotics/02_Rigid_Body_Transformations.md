# Rigid Body Transformations

[← Previous: Robotics Overview](01_Robotics_Overview.md) | [Next: Forward Kinematics →](03_Forward_Kinematics.md)

## Learning Objectives

1. Construct rotation matrices for 2D and 3D rotations and understand the properties of the SO(3) group
2. Describe orientations using Euler angles, identify the gimbal lock problem, and explain why it limits their usefulness
3. Apply the axis-angle representation and derive rotations via Rodrigues' formula
4. Define quaternions, perform quaternion multiplication, and convert between quaternions and rotation matrices
5. Build homogeneous transformation matrices in SE(3) to represent combined rotation and translation
6. Compose multiple transformations to describe the pose of one frame relative to another through intermediate frames

---

## Why This Matters

Every robotic system must answer a deceptively simple question: "Where is this thing, and which way is it pointing?" Whether "this thing" is a joint, an end-effector, a camera, or an obstacle, we need a precise mathematical language to describe its **position** and **orientation** — collectively called its **pose**.

Rigid body transformations provide that language. They are the foundation upon which all of kinematics (Lessons 3-5), dynamics (Lesson 6), and planning (Lessons 7-8) are built. A rotation matrix tells you how a gripper is oriented. A homogeneous transformation matrix tells you where a camera is relative to the robot base. Getting these representations right — and understanding their subtleties — is not optional. It is the entry ticket to robotics mathematics.

> **Analogy**: Quaternions are like compass bearings in 4D — they avoid the "confusion" of gimbal lock that plagues Euler angles, just as a magnetic compass avoids the confusion of trying to describe a direction using only "left turns" and "right turns."

---

## Coordinate Frames and Notation

In robotics, we attach **coordinate frames** (right-handed Cartesian coordinate systems) to objects: the world, the robot base, each link, the end-effector, sensors, and obstacles.

We use the following notation:
- $\{A\}$, $\{B\}$: coordinate frames
- ${}^{A}\mathbf{p}$: a point expressed in frame $\{A\}$
- ${}^{A}_{B}\mathbf{R}$: rotation matrix that transforms vectors from frame $\{B\}$ to frame $\{A\}$
- ${}^{A}_{B}\mathbf{T}$: homogeneous transformation from frame $\{B\}$ to frame $\{A\}$

The convention is: **the superscript is the frame the vector is expressed in, the subscript is the frame being described.**

---

## 2D Rotations

### Rotation Matrix in 2D

A 2D rotation by angle $\theta$ (counterclockwise) is represented by:

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

This matrix rotates a vector $\mathbf{v}$ in the plane:

$$\mathbf{v}' = R(\theta) \mathbf{v}$$

```python
import numpy as np

def rot2d(theta):
    """2D rotation matrix.

    Why a matrix? Because rotation is a linear transformation,
    and matrices are the natural representation of linear maps.
    Composing two rotations is just matrix multiplication.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

# Rotate a point (1, 0) by 90 degrees
theta = np.pi / 2
p = np.array([1.0, 0.0])
p_rotated = rot2d(theta) @ p
print(f"Original: {p}")
print(f"Rotated by 90 deg: {p_rotated}")  # [0, 1] (approximately)

# Key property: composition
# Rotating by 30 deg then by 60 deg = rotating by 90 deg
R_30 = rot2d(np.radians(30))
R_60 = rot2d(np.radians(60))
R_90 = rot2d(np.radians(90))
print(f"R(30)*R(60) == R(90): {np.allclose(R_30 @ R_60, R_90)}")  # True
```

### Properties of 2D Rotation Matrices

1. **Orthogonal**: $R^T R = I$ (columns are orthonormal)
2. **Determinant 1**: $\det(R) = 1$ (preserves orientation — no reflection)
3. **Inverse = Transpose**: $R^{-1} = R^T$ (computationally efficient!)
4. **Composition**: $R(\alpha) R(\beta) = R(\alpha + \beta)$

---

## 3D Rotations and SO(3)

### Elementary Rotation Matrices

Rotation about the three principal axes:

**Rotation about $x$-axis** by angle $\alpha$:

$$R_x(\alpha) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\alpha & -\sin\alpha \\ 0 & \sin\alpha & \cos\alpha \end{bmatrix}$$

**Rotation about $y$-axis** by angle $\beta$:

$$R_y(\beta) = \begin{bmatrix} \cos\beta & 0 & \sin\beta \\ 0 & 1 & 0 \\ -\sin\beta & 0 & \cos\beta \end{bmatrix}$$

**Rotation about $z$-axis** by angle $\gamma$:

$$R_z(\gamma) = \begin{bmatrix} \cos\gamma & -\sin\gamma & 0 \\ \sin\gamma & \cos\gamma & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

```python
def rotx(alpha):
    """Rotation about x-axis."""
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

def roty(beta):
    """Rotation about y-axis.

    Why is the sign pattern different? The y-axis rotation has +sin
    in the (0,2) position and -sin in the (2,0) position — this is
    because the cyclic order x→y→z requires this sign convention
    to maintain a right-handed coordinate system.
    """
    c, s = np.cos(beta), np.sin(beta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rotz(gamma):
    """Rotation about z-axis."""
    c, s = np.cos(gamma), np.sin(gamma)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])
```

### The Special Orthogonal Group SO(3)

The set of all 3D rotation matrices forms a mathematical group called **SO(3)** — the Special Orthogonal group in 3 dimensions.

**Properties of SO(3)**:
- **Closure**: If $R_1, R_2 \in SO(3)$, then $R_1 R_2 \in SO(3)$
- **Identity**: $I \in SO(3)$
- **Inverse**: $R^{-1} = R^T \in SO(3)$
- **Associativity**: $(R_1 R_2) R_3 = R_1 (R_2 R_3)$
- **NOT commutative**: $R_1 R_2 \neq R_2 R_1$ in general!

The non-commutativity is crucial — the order of rotations matters:

```python
# Demonstration: rotation order matters!
# Rotate 90 deg about x, then 90 deg about z
R1 = rotx(np.pi/2) @ rotz(np.pi/2)
# Rotate 90 deg about z, then 90 deg about x
R2 = rotz(np.pi/2) @ rotx(np.pi/2)

print("Rx(90)*Rz(90):")
print(np.round(R1, 3))
print("\nRz(90)*Rx(90):")
print(np.round(R2, 3))
print(f"\nAre they equal? {np.allclose(R1, R2)}")  # False!
```

> **Intuition**: Hold a book in front of you. Rotate it 90 degrees about the vertical axis (like opening a door), then 90 degrees about the axis pointing toward you (tilting it). Now start over: tilt first, then rotate. The final orientations are different!

### Interpreting a Rotation Matrix

A rotation matrix $R$ can be interpreted in two equivalent ways:

1. **Column interpretation**: The columns of ${}^{A}_{B}R$ are the unit vectors of frame $\{B\}$'s axes expressed in frame $\{A\}$:

$${}^{A}_{B}R = \begin{bmatrix} {}^{A}\hat{x}_B & {}^{A}\hat{y}_B & {}^{A}\hat{z}_B \end{bmatrix}$$

2. **Transformation interpretation**: $R$ transforms coordinates from frame $\{B\}$ to frame $\{A\}$:

$${}^{A}\mathbf{p} = {}^{A}_{B}R \cdot {}^{B}\mathbf{p}$$

---

## Euler Angles

### Definition

Euler angles parameterize a rotation as a sequence of three elementary rotations about coordinate axes. There are 12 possible conventions (choosing 3 axes from $\{x, y, z\}$ with no two adjacent axes the same).

The two most common in robotics:

| Convention | Sequence | Common Name | Typical Use |
|-----------|----------|-------------|-------------|
| ZYX | $R_z(\psi) R_y(\theta) R_x(\phi)$ | Yaw-Pitch-Roll | Aerospace, mobile robots |
| ZXZ | $R_z(\alpha) R_x(\beta) R_z(\gamma)$ | Proper Euler | Wrist orientations |

### ZYX (Yaw-Pitch-Roll)

$$R_{ZYX}(\psi, \theta, \phi) = R_z(\psi) \cdot R_y(\theta) \cdot R_x(\phi)$$

where:
- $\psi$ = yaw (rotation about $z$)
- $\theta$ = pitch (rotation about $y'$)
- $\phi$ = roll (rotation about $x''$)

```python
def euler_zyx_to_rotation(yaw, pitch, roll):
    """Convert ZYX Euler angles to rotation matrix.

    Why ZYX? In aerospace and mobile robotics, we think of orientation
    as: first face a compass heading (yaw), then tilt up/down (pitch),
    then tilt sideways (roll). This matches our physical intuition.
    """
    return rotz(yaw) @ roty(pitch) @ rotx(roll)

def rotation_to_euler_zyx(R):
    """Extract ZYX Euler angles from rotation matrix.

    Why the atan2 function? Unlike atan, atan2 returns the correct
    quadrant for the angle, handling the full [-pi, pi] range.

    WARNING: This fails at pitch = +/- 90 degrees (gimbal lock).
    """
    # Check for gimbal lock
    if abs(R[2, 0]) >= 1.0 - 1e-10:
        # Gimbal lock: pitch = +/- 90 degrees
        yaw = np.arctan2(-R[0, 1], R[0, 2])  # or any consistent choice
        pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
        roll = 0.0  # arbitrary — one DOF is lost
        print("WARNING: Gimbal lock detected!")
    else:
        pitch = -np.arcsin(R[2, 0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])

    return yaw, pitch, roll

# Example: yaw=30 deg, pitch=45 deg, roll=60 deg
yaw, pitch, roll = np.radians(30), np.radians(45), np.radians(60)
R = euler_zyx_to_rotation(yaw, pitch, roll)
yaw_r, pitch_r, roll_r = rotation_to_euler_zyx(R)
print(f"Original: yaw={np.degrees(yaw):.1f}, pitch={np.degrees(pitch):.1f}, "
      f"roll={np.degrees(roll):.1f}")
print(f"Recovered: yaw={np.degrees(yaw_r):.1f}, pitch={np.degrees(pitch_r):.1f}, "
      f"roll={np.degrees(roll_r):.1f}")
```

### The Gimbal Lock Problem

**Gimbal lock** occurs when two rotation axes align, causing a loss of one degree of freedom. For ZYX Euler angles, this happens when pitch $\theta = \pm 90°$.

At $\theta = 90°$:

$$R = R_z(\psi) R_y(90°) R_x(\phi) = \begin{bmatrix} 0 & \sin(\phi-\psi) & \cos(\phi-\psi) \\ 0 & -\cos(\phi-\psi) & \sin(\phi-\psi) \\ -1 & 0 & 0 \end{bmatrix}$$

Notice: only the *difference* $\phi - \psi$ appears — we cannot determine $\phi$ and $\psi$ individually. One degree of rotational freedom is lost.

```python
# Gimbal lock demonstration
# At pitch = 90 degrees, changing yaw and roll have the same effect

R1 = euler_zyx_to_rotation(
    yaw=np.radians(30), pitch=np.radians(90), roll=np.radians(0))
R2 = euler_zyx_to_rotation(
    yaw=np.radians(0), pitch=np.radians(90), roll=np.radians(-30))

# These produce the SAME rotation matrix!
print("Are R1 and R2 equal?", np.allclose(R1, R2))  # True
# This means yaw=30,roll=0 and yaw=0,roll=-30 are indistinguishable
# at pitch=90 — we've lost one DOF.
```

> **Real-world consequence**: The Apollo 11 spacecraft used Euler angles for its guidance system. Engineers had to ensure the spacecraft never approached gimbal lock orientations, adding constraints to mission planning. Modern systems use quaternions to avoid this entirely.

---

## Axis-Angle Representation

### Euler's Rotation Theorem

Any rotation in 3D can be described as a single rotation by angle $\theta$ about a fixed axis $\hat{\omega}$ (unit vector). This is **Euler's rotation theorem**.

The axis-angle representation is: $(\hat{\omega}, \theta)$ where $\hat{\omega} \in \mathbb{R}^3$, $\|\hat{\omega}\| = 1$, and $\theta \in [0, \pi]$.

Alternatively, we can combine them into a single vector $\boldsymbol{\omega} = \theta \hat{\omega}$ (the rotation vector), where the direction encodes the axis and the magnitude encodes the angle.

### Rodrigues' Rotation Formula

Given axis $\hat{\omega}$ and angle $\theta$, the rotation matrix is:

$$R = I + \sin\theta \, [\hat{\omega}]_\times + (1 - \cos\theta) \, [\hat{\omega}]_\times^2$$

where $[\hat{\omega}]_\times$ is the **skew-symmetric matrix** of $\hat{\omega}$:

$$[\hat{\omega}]_\times = \begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix}$$

The skew-symmetric matrix implements the cross product: $[\hat{\omega}]_\times \mathbf{v} = \hat{\omega} \times \mathbf{v}$.

```python
def skew(w):
    """Create skew-symmetric matrix from 3D vector.

    Why skew-symmetric? It encodes the cross product as matrix multiplication.
    This is the bridge between the Lie algebra so(3) and the Lie group SO(3),
    which becomes important in advanced robotics (screw theory, exponential maps).
    """
    return np.array([[    0, -w[2],  w[1]],
                     [ w[2],     0, -w[0]],
                     [-w[1],  w[0],     0]])

def rodrigues(omega_hat, theta):
    """Rodrigues' rotation formula: axis-angle to rotation matrix.

    Why this formula? It provides a direct, singularity-free way to
    compute a rotation matrix from a physically intuitive representation
    (axis + angle). No gimbal lock possible.
    """
    K = skew(omega_hat)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def rotation_to_axis_angle(R):
    """Extract axis-angle from rotation matrix.

    Uses the fact that:
    - theta = arccos((trace(R) - 1) / 2)
    - omega is the eigenvector of R with eigenvalue 1
    """
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if abs(theta) < 1e-10:
        # No rotation — axis is undefined
        return np.array([0, 0, 1]), 0.0

    if abs(theta - np.pi) < 1e-10:
        # theta = pi: use eigenvector method
        # R + I has rank 1; any non-zero column is proportional to omega
        M = R + np.eye(3)
        for col in range(3):
            if np.linalg.norm(M[:, col]) > 1e-10:
                omega_hat = M[:, col] / np.linalg.norm(M[:, col])
                return omega_hat, theta

    # General case
    omega_hat = np.array([R[2,1] - R[1,2],
                          R[0,2] - R[2,0],
                          R[1,0] - R[0,1]]) / (2 * np.sin(theta))
    return omega_hat, theta

# Example: 120 degrees about the [1,1,1]/sqrt(3) axis
omega_hat = np.array([1, 1, 1]) / np.sqrt(3)
theta = np.radians(120)
R = rodrigues(omega_hat, theta)
print("Rotation matrix:")
print(np.round(R, 4))

# Verify: recover axis-angle
omega_recovered, theta_recovered = rotation_to_axis_angle(R)
print(f"\nRecovered axis: {np.round(omega_recovered, 4)}")
print(f"Recovered angle: {np.degrees(theta_recovered):.1f} degrees")
```

### The Exponential Map

Rodrigues' formula is actually the **matrix exponential** of a skew-symmetric matrix:

$$R = e^{[\hat{\omega}]_\times \theta} = I + \sin\theta \, [\hat{\omega}]_\times + (1 - \cos\theta) \, [\hat{\omega}]_\times^2$$

This connection between skew-symmetric matrices (the **Lie algebra** $\mathfrak{so}(3)$) and rotation matrices (the **Lie group** SO(3)) is fundamental to modern robotics, particularly in screw theory and optimization on manifolds.

```python
from scipy.linalg import expm

# The matrix exponential gives the same result as Rodrigues' formula
omega_hat = np.array([0, 0, 1])  # z-axis
theta = np.radians(45)
K = skew(omega_hat)

R_rodrigues = rodrigues(omega_hat, theta)
R_expm = expm(K * theta)  # matrix exponential

print(f"Rodrigues == expm? {np.allclose(R_rodrigues, R_expm)}")  # True
```

---

## Quaternions

### Motivation

Euler angles suffer from gimbal lock. Rotation matrices use 9 numbers (with 6 constraints) for 3 DOF. Quaternions use 4 numbers (with 1 constraint) — more compact, singularity-free, and numerically stable for interpolation.

### Definition

A **unit quaternion** $\mathbf{q}$ is a 4-dimensional vector with unit norm:

$$\mathbf{q} = q_w + q_x \mathbf{i} + q_y \mathbf{j} + q_z \mathbf{k} = (q_w, \mathbf{q}_v)$$

where:
- $q_w$ is the **scalar** (real) part
- $\mathbf{q}_v = (q_x, q_y, q_z)$ is the **vector** (imaginary) part
- $\|\mathbf{q}\| = \sqrt{q_w^2 + q_x^2 + q_y^2 + q_z^2} = 1$

The imaginary units satisfy Hamilton's equations:

$$\mathbf{i}^2 = \mathbf{j}^2 = \mathbf{k}^2 = \mathbf{ijk} = -1$$

### Quaternion from Axis-Angle

Given rotation axis $\hat{\omega}$ and angle $\theta$:

$$\mathbf{q} = \left(\cos\frac{\theta}{2}, \, \sin\frac{\theta}{2} \, \hat{\omega}\right)$$

Note the half-angle! This is because the unit quaternion double-covers SO(3): $\mathbf{q}$ and $-\mathbf{q}$ represent the same rotation.

```python
class Quaternion:
    """Unit quaternion for 3D rotation.

    Why a class? Quaternion arithmetic (especially multiplication) has
    specific rules that differ from regular vector operations. Encapsulating
    these in a class prevents errors and makes code readable.

    Convention: q = (w, x, y, z) where w is the scalar part.
    """
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self._normalize()

    def _normalize(self):
        """Ensure unit norm. Numerical drift can violate this."""
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm > 1e-10:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm

    @classmethod
    def from_axis_angle(cls, axis, angle):
        """Create quaternion from axis-angle representation.

        Why half-angle? Because quaternion rotation uses q * p * q_conj,
        which applies the rotation twice (once from each side), so each
        side contributes half the angle.
        """
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)  # ensure unit vector
        half = angle / 2
        w = np.cos(half)
        x, y, z = np.sin(half) * axis
        return cls(w, x, y, z)

    def conjugate(self):
        """Quaternion conjugate — negates the vector part.

        For unit quaternions, the conjugate equals the inverse.
        This represents the reverse rotation.
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other):
        """Hamilton product (quaternion multiplication).

        This is the core operation: composing two rotations.
        The formula follows directly from i^2 = j^2 = k^2 = ijk = -1.
        """
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return Quaternion(w, x, y, z)

    def rotate_vector(self, v):
        """Rotate a 3D vector using this quaternion.

        The rotation formula: v' = q * v_quat * q_conj
        where v_quat = (0, v_x, v_y, v_z) — a "pure" quaternion.
        """
        v_quat = Quaternion(0, v[0], v[1], v[2])
        result = self * v_quat * self.conjugate()
        return np.array([result.x, result.y, result.z])

    def to_rotation_matrix(self):
        """Convert to 3x3 rotation matrix.

        Why provide this? Many algorithms (FK, Jacobian) work with
        rotation matrices. Quaternions are better for storage, interpolation,
        and composition; matrices are better for transforming many points.
        """
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
        ])

    def __repr__(self):
        return f"Q({self.w:.4f}, {self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


# Example: 90 degrees about z-axis
q = Quaternion.from_axis_angle([0, 0, 1], np.pi/2)
print(f"Quaternion: {q}")

# Rotate (1, 0, 0) — should give (0, 1, 0)
v = np.array([1.0, 0.0, 0.0])
v_rotated = q.rotate_vector(v)
print(f"Rotated vector: {np.round(v_rotated, 4)}")  # [0, 1, 0]

# Compose two 90-degree rotations about z — should give 180 degrees
q2 = q * q
print(f"Double rotation: {q2}")
v_double = q2.rotate_vector(v)
print(f"After 180 deg: {np.round(v_double, 4)}")  # [-1, 0, 0]

# Verify: quaternion → matrix matches Rodrigues
R_quat = q.to_rotation_matrix()
R_rodrigues = rodrigues(np.array([0, 0, 1]), np.pi/2)
print(f"Matrix match: {np.allclose(R_quat, R_rodrigues)}")  # True
```

### Quaternion Interpolation (SLERP)

One of the biggest advantages of quaternions is smooth interpolation. **SLERP** (Spherical Linear Interpolation) produces a constant-velocity rotation between two orientations:

$$\text{slerp}(\mathbf{q}_0, \mathbf{q}_1, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} \mathbf{q}_0 + \frac{\sin(t\Omega)}{\sin\Omega} \mathbf{q}_1$$

where $\Omega = \arccos(\mathbf{q}_0 \cdot \mathbf{q}_1)$ and $t \in [0, 1]$.

```python
def slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions.

    Why SLERP instead of linear interpolation? Linear interpolation of
    rotation matrices or Euler angles produces non-uniform angular velocity
    and can pass through non-rotation states. SLERP follows the shortest
    path on the unit sphere at constant angular velocity.
    """
    # Ensure shortest path (q and -q are the same rotation)
    dot = q0.w*q1.w + q0.x*q1.x + q0.y*q1.y + q0.z*q1.z
    if dot < 0:
        q1 = Quaternion(-q1.w, -q1.x, -q1.y, -q1.z)
        dot = -dot

    dot = np.clip(dot, -1, 1)

    if dot > 0.9995:
        # Very close — use linear interpolation to avoid numerical issues
        w = q0.w + t * (q1.w - q0.w)
        x = q0.x + t * (q1.x - q0.x)
        y = q0.y + t * (q1.y - q0.y)
        z = q0.z + t * (q1.z - q0.z)
        return Quaternion(w, x, y, z)

    omega = np.arccos(dot)
    sin_omega = np.sin(omega)

    s0 = np.sin((1 - t) * omega) / sin_omega
    s1 = np.sin(t * omega) / sin_omega

    w = s0 * q0.w + s1 * q1.w
    x = s0 * q0.x + s1 * q1.x
    y = s0 * q0.y + s1 * q1.y
    z = s0 * q0.z + s1 * q1.z

    return Quaternion(w, x, y, z)

# Interpolate from identity (no rotation) to 90 deg about z
q_start = Quaternion(1, 0, 0, 0)  # identity
q_end = Quaternion.from_axis_angle([0, 0, 1], np.pi/2)

for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    q_t = slerp(q_start, q_end, t)
    v = q_t.rotate_vector(np.array([1, 0, 0]))
    angle = np.degrees(np.arctan2(v[1], v[0]))
    print(f"t={t:.2f}: angle = {angle:.1f} deg")
```

### Comparison of Rotation Representations

| Representation | Parameters | Singularity-free? | Composition | Interpolation | Use Case |
|---------------|------------|-------------------|-------------|---------------|----------|
| Rotation Matrix | 9 (6 constraints) | Yes | Matrix multiply | Poor (not on SO(3)) | FK, Jacobian |
| Euler Angles | 3 | No (gimbal lock) | Complex formula | Poor | Human interface |
| Axis-Angle | 4 (1 constraint) | Near $\theta=0$ | Rodrigues | Moderate | Visualization |
| Quaternion | 4 (1 constraint) | Yes | Hamilton product | SLERP (excellent) | Storage, interpolation |

---

## Homogeneous Transformations — SE(3)

### Combining Rotation and Translation

A rigid body transformation in 3D consists of a rotation $R$ and a translation $\mathbf{d}$:

$${}^{A}\mathbf{p} = {}^{A}_{B}R \cdot {}^{B}\mathbf{p} + {}^{A}\mathbf{d}_{B}$$

We encode this compactly as a **4x4 homogeneous transformation matrix**:

$${}^{A}_{B}T = \begin{bmatrix} {}^{A}_{B}R & {}^{A}\mathbf{d}_B \\ \mathbf{0}^T & 1 \end{bmatrix} = \begin{bmatrix} r_{11} & r_{12} & r_{13} & d_x \\ r_{21} & r_{22} & r_{23} & d_y \\ r_{31} & r_{32} & r_{33} & d_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

We use **homogeneous coordinates**: a 3D point $(x, y, z)$ becomes $(x, y, z, 1)^T$.

$$\begin{bmatrix} {}^{A}\mathbf{p} \\ 1 \end{bmatrix} = {}^{A}_{B}T \begin{bmatrix} {}^{B}\mathbf{p} \\ 1 \end{bmatrix}$$

```python
def make_transform(R, d):
    """Create 4x4 homogeneous transformation matrix.

    Why 4x4? Because combining rotation and translation into a single
    matrix allows us to compose multiple transformations by simple
    matrix multiplication — the same operation for rotation alone.
    This is the key insight of homogeneous coordinates.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = d
    return T

def transform_point(T, p):
    """Transform a 3D point using a homogeneous transformation."""
    p_h = np.array([p[0], p[1], p[2], 1.0])  # homogeneous coordinates
    p_transformed = T @ p_h
    return p_transformed[:3]  # back to 3D

def inverse_transform(T):
    """Compute the inverse of a homogeneous transformation.

    Why not just np.linalg.inv? Because we can exploit the structure:
    T_inv = [[R^T, -R^T * d], [0, 1]]
    This is faster and more numerically stable.
    """
    R = T[:3, :3]
    d = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ d
    return T_inv

# Example: frame B is rotated 90 deg about z and translated (1, 0, 0.5)
R = rotz(np.pi/2)
d = np.array([1.0, 0.0, 0.5])
T_AB = make_transform(R, d)

print("T_AB:")
print(np.round(T_AB, 4))

# Transform a point (1, 0, 0) in frame B to frame A
p_B = np.array([1.0, 0.0, 0.0])
p_A = transform_point(T_AB, p_B)
print(f"\nPoint in B: {p_B}")
print(f"Point in A: {np.round(p_A, 4)}")  # [1, 1, 0.5]

# Verify inverse
T_BA = inverse_transform(T_AB)
p_B_recovered = transform_point(T_BA, p_A)
print(f"Recovered in B: {np.round(p_B_recovered, 4)}")  # [1, 0, 0]
```

### The SE(3) Group

The set of all homogeneous transformation matrices forms **SE(3)** — the Special Euclidean group in 3 dimensions. It represents all rigid body motions (rotation + translation) in 3D space.

**Properties**:
- **Closure**: $T_1 \cdot T_2 \in SE(3)$
- **Identity**: $I_{4\times4} \in SE(3)$
- **Inverse**: $T^{-1} \in SE(3)$ (as computed above)
- **NOT commutative**: order of transformations matters

---

## Composition of Transformations

### Chain Rule

If we know the pose of frame $\{C\}$ relative to $\{B\}$, and $\{B\}$ relative to $\{A\}$, then:

$${}^{A}_{C}T = {}^{A}_{B}T \cdot {}^{B}_{C}T$$

This is the **chain rule** and it is the mathematical foundation of forward kinematics (Lesson 3).

```python
# Example: a robotic arm with base frame {0}, elbow frame {1}, and tool frame {2}

# Link 1: rotate 45 deg about z, translate 1m along new x
T_01 = make_transform(rotz(np.radians(45)), np.array([1.0, 0.0, 0.0]))

# Link 2: rotate 30 deg about z, translate 0.8m along new x
T_12 = make_transform(rotz(np.radians(30)), np.array([0.8, 0.0, 0.0]))

# Compose: tool frame in base frame
T_02 = T_01 @ T_12

print("T_01 (base to elbow):")
print(np.round(T_01, 4))
print("\nT_12 (elbow to tool):")
print(np.round(T_12, 4))
print("\nT_02 (base to tool) = T_01 * T_12:")
print(np.round(T_02, 4))

# Where is the tool origin in the base frame?
tool_position = T_02[:3, 3]
print(f"\nTool position in base frame: {np.round(tool_position, 4)}")

# What orientation does the tool have?
total_angle = np.degrees(np.arctan2(T_02[1, 0], T_02[0, 0]))
print(f"Tool orientation (z-rotation): {total_angle:.1f} degrees")  # 75 deg
```

### Fixed Frame vs Body Frame Rotations

An important subtlety: the order of matrix multiplication depends on whether rotations are about **fixed (world) axes** or **body (current) axes**.

- **Body frame** (post-multiply): $T = T_1 \cdot T_2 \cdot T_3$ — each rotation is about the *current* frame's axis
- **Fixed frame** (pre-multiply): $T = T_3 \cdot T_2 \cdot T_1$ — each rotation is about the *fixed world* frame's axis

In DH convention (Lesson 3), we use body frame (post-multiply) convention.

```python
# Body frame vs fixed frame rotation
# Rotate 90 deg about z, then 90 deg about y (body frame)
T_body = rotz(np.pi/2) @ roty(np.pi/2)

# The same final orientation using fixed frame:
# must reverse the order
T_fixed = roty(np.pi/2) @ rotz(np.pi/2)

# These are different!
print("Body frame (Rz then Ry_body):")
print(np.round(T_body, 3))
print("\nFixed frame (Ry_fixed then Rz_fixed):")
print(np.round(T_fixed, 3))
print(f"\nSame? {np.allclose(T_body, T_fixed)}")  # False in general
```

---

## Practical Considerations

### Numerical Issues with Rotation Matrices

Repeated multiplication of rotation matrices causes numerical drift — the result may no longer be exactly orthogonal. Periodically **re-orthogonalize** using SVD:

```python
def reorthogonalize(R):
    """Re-orthogonalize a rotation matrix using SVD.

    Why SVD? It finds the closest orthogonal matrix in the Frobenius norm
    sense. This is crucial in real-time robot control where thousands of
    matrix multiplications accumulate floating-point errors.
    """
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    # Ensure det = +1 (not a reflection)
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
    return R_clean

# Simulate numerical drift
R = rotz(0.01)  # tiny rotation
R_accumulated = np.eye(3)
for _ in range(100000):
    R_accumulated = R_accumulated @ R

print(f"det(R) after 100k multiplications: {np.linalg.det(R_accumulated):.10f}")
print(f"R^T R == I? {np.allclose(R_accumulated.T @ R_accumulated, np.eye(3))}")

R_clean = reorthogonalize(R_accumulated)
print(f"\nAfter re-orthogonalization:")
print(f"det(R) = {np.linalg.det(R_clean):.10f}")
print(f"R^T R == I? {np.allclose(R_clean.T @ R_clean, np.eye(3))}")
```

### When to Use Which Representation

| Scenario | Best Representation | Reason |
|----------|-------------------|--------|
| Forward kinematics | Rotation matrices / SE(3) | Direct chain multiplication |
| Storing orientations | Quaternions | Compact (4 numbers), no singularity |
| Interpolating orientations | Quaternions (SLERP) | Smooth, constant-velocity |
| Human-readable display | Euler angles | Intuitive (yaw/pitch/roll) |
| Optimization on SO(3) | Axis-angle / Lie algebra | Minimal parameterization |
| Real-time control loop | Quaternions | Fast composition, easy re-normalization |

---

## Summary

- **Rotation matrices** are 3x3 orthogonal matrices with determinant +1, forming the SO(3) group
- **Euler angles** use 3 parameters but suffer from **gimbal lock** at specific configurations
- **Axis-angle** (Rodrigues' formula) provides a singularity-free rotation from a physical axis and angle
- **Quaternions** use 4 parameters, are singularity-free, and enable smooth interpolation via SLERP
- **Homogeneous transformations** (SE(3)) combine rotation and translation in a single 4x4 matrix
- **Composition** of transformations is achieved by matrix multiplication, forming the basis of forward kinematics

---

## Exercises

### Exercise 1: Rotation Matrix Properties

Given $R = R_z(30°) \cdot R_x(45°)$:
1. Compute $R$ numerically
2. Verify that $R^T R = I$ and $\det(R) = 1$
3. What is $R^{-1}$? Verify by computing $R \cdot R^{-1}$

### Exercise 2: Gimbal Lock Investigation

1. Write a function that takes ZYX Euler angles and returns the rotation matrix
2. Set pitch = 89 degrees and compute the matrix for (yaw=10, pitch=89, roll=20) and (yaw=15, pitch=89, roll=15). How different are the results?
3. Now set pitch = 90 degrees exactly. Show that the extracted yaw and roll are ambiguous by finding two different (yaw, roll) pairs that give the same rotation matrix

### Exercise 3: Quaternion Operations

1. Create quaternions for: (a) 90 deg about x-axis, (b) 90 deg about y-axis
2. Compose them: first (a) then (b), and first (b) then (a). Are the results the same?
3. Use SLERP to interpolate between the identity quaternion and a 180-degree rotation about the z-axis at 5 equally spaced values of $t$

### Exercise 4: Transformation Chains

A camera is mounted on a robot end-effector. Given:
- ${}^{0}_{ee}T$: end-effector in base frame (rotation by 45 deg about z, translation $(0.5, 0.3, 0.8)$)
- ${}^{ee}_{cam}T$: camera in end-effector frame (rotation by 180 deg about x, translation $(0, 0, 0.1)$)

1. Compute the camera's pose in the base frame ${}^{0}_{cam}T$
2. A point $\mathbf{p}_{cam} = (0.2, 0.1, 1.0)$ is detected in camera coordinates. Find its position in the base frame
3. If the robot moves so that ${}^{0}_{ee}T$ changes (new rotation: 90 deg about z, same translation), recompute the camera pose

### Exercise 5: Representation Conversion Round-Trip

1. Start with Euler angles: yaw = 25 deg, pitch = 40 deg, roll = -15 deg
2. Convert to rotation matrix
3. Convert rotation matrix to axis-angle
4. Convert axis-angle to quaternion
5. Convert quaternion back to rotation matrix
6. Convert rotation matrix back to Euler angles
7. Verify you recover the original angles (within numerical precision)

---

[← Previous: Robotics Overview](01_Robotics_Overview.md) | [Next: Forward Kinematics →](03_Forward_Kinematics.md)
