# 12. Animation and Skeletal Systems

[← Previous: Path Tracing and Global Illumination](11_Path_Tracing_and_Global_Illumination.md) | [Next: Particle Systems and Effects →](13_Particle_Systems_and_Effects.md)

---

## Learning Objectives

1. Understand keyframe animation and the role of interpolation between poses
2. Implement linear interpolation (lerp) and spherical linear interpolation (slerp) for rotations
3. Use Bezier curves and splines to create smooth animation paths
4. Describe skeletal animation: bones, joints, bind pose, and the bone hierarchy
5. Implement forward kinematics (FK) to compute world-space joint positions
6. Understand inverse kinematics (IK) algorithms: CCD and FABRIK
7. Explain vertex skinning methods: linear blend skinning (LBS) and dual quaternion skinning
8. Apply morph targets (blend shapes) for facial animation and deformations

---

## Why This Matters

Static 3D scenes are impressive, but **animation** brings them to life. From a character walking across a game world to a robot arm assembling parts in a simulation, animation is the bridge between geometry and storytelling. The techniques in this lesson are used in every animated film, video game, VR experience, and robotics simulator.

Skeletal animation, in particular, is one of the most elegant solutions in computer graphics: instead of animating thousands of vertices individually, we animate a skeleton of perhaps 50-200 bones, and the mesh follows automatically through **skinning**. This separation of control (skeleton) from detail (mesh) is what makes character animation practical.

---

## 1. Keyframe Animation

### 1.1 Concept

In **keyframe animation**, the animator specifies the state of an object (position, rotation, scale, color, etc.) at specific moments in time called **keyframes**. The system automatically computes intermediate states through **interpolation**.

```
Time:     0s         0.5s        1.0s        1.5s        2.0s
          |           |           |           |           |
Key:    [start]                 [peak]                 [end]
          ●───────────────────────●───────────────────────●
          pos=(0,0,0)           pos=(0,3,0)           pos=(5,0,0)
```

The animator only sets the keyframes; interpolation fills in every frame in between.

### 1.2 Animation Curves

Each animated property has an **animation curve** (also called an f-curve) that maps time $t$ to value $v(t)$. The shape of this curve determines the feel of the motion:

- **Linear**: Constant velocity; mechanical feel
- **Ease-in**: Starts slow, accelerates
- **Ease-out**: Starts fast, decelerates
- **Ease-in-out**: Smooth start and end (most natural for character motion)

These are typically controlled by cubic Bezier curves or Hermite splines on the curve itself.

---

## 2. Interpolation Methods

### 2.1 Linear Interpolation (Lerp)

The simplest interpolation between two values $a$ and $b$:

$$\text{lerp}(a, b, t) = (1 - t) \cdot a + t \cdot b, \quad t \in [0, 1]$$

For vectors (positions):

$$\mathbf{p}(t) = (1 - t)\mathbf{p}_0 + t\mathbf{p}_1$$

Lerp produces constant-velocity motion between keyframes. It is fast and sufficient for many properties (position, color, scale), but **not suitable for rotations** represented as matrices or quaternions.

### 2.2 Why Lerp Fails for Rotations

Consider interpolating between two rotation matrices. Lerp would compute $(1-t)\mathbf{R}_0 + t\mathbf{R}_1$, but the result is generally **not a rotation matrix** (it loses orthogonality). The interpolated matrix may include shearing or scaling artifacts.

Even with Euler angles, lerp causes **gimbal lock** and non-uniform rotation speed.

### 2.3 Quaternion Representation

A **quaternion** $q = w + xi + yj + zk$ represents a rotation of angle $\theta$ around axis $\mathbf{u}$:

$$q = \cos\frac{\theta}{2} + \sin\frac{\theta}{2}(u_x i + u_y j + u_z k)$$

Unit quaternions ($\|q\| = 1$) form a **double cover** of 3D rotations: $q$ and $-q$ represent the same rotation. They avoid gimbal lock and compose efficiently.

### 2.4 Spherical Linear Interpolation (Slerp)

**Slerp** interpolates along the shortest arc on the 4D unit sphere, producing constant angular velocity:

$$\text{slerp}(q_0, q_1, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} q_0 + \frac{\sin(t\Omega)}{\sin\Omega} q_1$$

where $\Omega = \arccos(q_0 \cdot q_1)$ is the angle between the quaternions.

**Important**: If $q_0 \cdot q_1 < 0$, negate one quaternion first to ensure interpolation takes the short path (less than 180 degrees).

**When $\Omega$ is very small** (nearly identical rotations), slerp has numerical issues. Fall back to normalized lerp (nlerp) in this case:

$$\text{nlerp}(q_0, q_1, t) = \frac{(1-t)q_0 + tq_1}{\|(1-t)q_0 + tq_1\|}$$

Nlerp does not have constant angular velocity but is fast and avoids the singularity.

### 2.5 Implementation

```python
import numpy as np

def lerp(a, b, t):
    """Linear interpolation between scalars, vectors, or arrays."""
    return (1.0 - t) * a + t * b


def quaternion_dot(q1, q2):
    """Dot product of two quaternions (as 4D vectors)."""
    return np.dot(q1, q2)


def quaternion_normalize(q):
    """Normalize a quaternion to unit length."""
    n = np.linalg.norm(q)
    return q / n if n > 1e-10 else q


def slerp(q0, q1, t):
    """
    Spherical Linear Interpolation between two unit quaternions.
    Quaternion format: [w, x, y, z].
    """
    # Ensure shortest path: if dot product is negative, negate one
    dot = quaternion_dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot

    # If quaternions are very close, fall back to normalized lerp
    # Why: sin(omega) approaches 0, causing division-by-zero
    if dot > 0.9995:
        result = (1.0 - t) * q0 + t * q1
        return quaternion_normalize(result)

    omega = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_omega = np.sin(omega)

    s0 = np.sin((1.0 - t) * omega) / sin_omega
    s1 = np.sin(t * omega) / sin_omega

    return s0 * q0 + s1 * q1


def quaternion_from_axis_angle(axis, angle_deg):
    """Create a quaternion from an axis and angle (degrees)."""
    angle_rad = np.radians(angle_deg)
    half = angle_rad / 2.0
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(half),
                     axis[0] * np.sin(half),
                     axis[1] * np.sin(half),
                     axis[2] * np.sin(half)])


def quaternion_to_matrix(q):
    """Convert a unit quaternion [w,x,y,z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


# Demo: interpolate between two rotations
q_start = quaternion_from_axis_angle(np.array([0, 1, 0]), 0)     # No rotation
q_end   = quaternion_from_axis_angle(np.array([0, 1, 0]), 120)   # 120 degrees around Y

for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    q = slerp(q_start, q_end, t)
    # Extract the effective rotation angle
    angle = 2 * np.degrees(np.arccos(np.clip(q[0], -1, 1)))
    print(f"  t={t:.2f}: angle = {angle:.1f} degrees, q = [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]")
```

Output:
```
  t=0.00: angle = 0.0 degrees, q = [1.000, 0.000, 0.000, 0.000]
  t=0.25: angle = 30.0 degrees, q = [0.966, 0.000, 0.259, 0.000]
  t=0.50: angle = 60.0 degrees, q = [0.866, 0.000, 0.500, 0.000]
  t=0.75: angle = 90.0 degrees, q = [0.707, 0.000, 0.707, 0.000]
  t=1.00: angle = 120.0 degrees, q = [0.500, 0.000, 0.866, 0.000]
```

Notice the angle increases uniformly (30 degrees per step) -- this is the constant angular velocity property of slerp.

---

## 3. Bezier Curves and Splines

### 3.1 Cubic Bezier Curves

A **cubic Bezier curve** is defined by four control points $\mathbf{P}_0, \mathbf{P}_1, \mathbf{P}_2, \mathbf{P}_3$:

$$\mathbf{B}(t) = (1-t)^3\mathbf{P}_0 + 3(1-t)^2 t\mathbf{P}_1 + 3(1-t)t^2\mathbf{P}_2 + t^3\mathbf{P}_3, \quad t \in [0,1]$$

Properties:
- The curve starts at $\mathbf{P}_0$ and ends at $\mathbf{P}_3$
- The tangent at $\mathbf{P}_0$ points toward $\mathbf{P}_1$; the tangent at $\mathbf{P}_3$ points from $\mathbf{P}_2$
- The curve lies within the convex hull of the control points
- $\mathbf{P}_1$ and $\mathbf{P}_2$ act as "handles" that shape the curve without it passing through them

### 3.2 Animation Paths

Bezier curves define smooth motion paths in 3D space. An object follows the curve as $t$ advances from 0 to 1:

```python
def bezier_cubic(P0, P1, P2, P3, t):
    """
    Evaluate a cubic Bezier curve at parameter t.
    The curve smoothly moves from P0 to P3, shaped by control points P1, P2.
    """
    u = 1.0 - t
    # Why De Casteljau form: numerically more stable than expanding the polynomial
    return (u**3 * P0 + 3 * u**2 * t * P1
            + 3 * u * t**2 * P2 + t**3 * P3)


def bezier_tangent(P0, P1, P2, P3, t):
    """Tangent (velocity) of the cubic Bezier at parameter t."""
    u = 1.0 - t
    return (3 * u**2 * (P1 - P0) + 6 * u * t * (P2 - P1)
            + 3 * t**2 * (P3 - P2))
```

### 3.3 Catmull-Rom Splines

When the animation path passes through a sequence of keyframe positions, **Catmull-Rom splines** are convenient because they **interpolate** the control points (the curve passes through all of them):

Given points $\mathbf{P}_{i-1}, \mathbf{P}_i, \mathbf{P}_{i+1}, \mathbf{P}_{i+2}$, the segment between $\mathbf{P}_i$ and $\mathbf{P}_{i+1}$ is:

$$\mathbf{C}(t) = 0.5 \begin{bmatrix} 1 & t & t^2 & t^3 \end{bmatrix}
\begin{bmatrix} 0 & 2 & 0 & 0 \\ -1 & 0 & 1 & 0 \\ 2 & -5 & 4 & -1 \\ -1 & 3 & -3 & 1 \end{bmatrix}
\begin{bmatrix} \mathbf{P}_{i-1} \\ \mathbf{P}_i \\ \mathbf{P}_{i+1} \\ \mathbf{P}_{i+2} \end{bmatrix}$$

The tangent at each point is automatically computed from the neighboring points, ensuring $C^1$ continuity.

---

## 4. Skeletal Animation

### 4.1 Skeleton Structure

A **skeleton** is a hierarchy (tree) of **bones** (also called **joints**). Each bone has:
- A **parent** bone (except the root)
- A **local transform** (rotation + translation relative to parent)
- A **rest/bind pose** -- the default configuration when the character is in T-pose or A-pose

```
                    [Root / Hips]
                    /            \
            [Spine]              [Left Leg]
           /       \                  \
     [Chest]    [Right Leg]     [Left Knee]
      /    \                          \
[Left Arm] [Right Arm]          [Left Foot]
    |           |
[Left Elbow] [Right Elbow]
    |           |
[Left Hand] [Right Hand]
```

### 4.2 Bind Pose and Inverse Bind Matrix

The **bind pose** is the skeleton configuration in which the mesh was modeled. Each bone $j$ has a **bind-pose world transform** $\mathbf{B}_j$ that maps from bone-local space to world space.

The **inverse bind matrix** $\mathbf{B}_j^{-1}$ transforms a vertex from world space into the bone's local space (in the bind pose). This is precomputed once:

$$\mathbf{B}_j^{-1} = (\mathbf{B}_j)^{-1}$$

During animation, the bone's current world transform is $\mathbf{W}_j$. The **skinning matrix** for bone $j$ is:

$$\mathbf{S}_j = \mathbf{W}_j \cdot \mathbf{B}_j^{-1}$$

This matrix takes a vertex from its bind-pose position to its animated position.

### 4.3 Animated Pose

An **animation clip** stores keyframes for each bone's local transform (typically as quaternion rotation + translation). At each frame:

1. Interpolate between keyframes (using slerp for rotation, lerp for position)
2. Compute each bone's local transform
3. Propagate through the hierarchy to get world transforms: $\mathbf{W}_j = \mathbf{W}_{\text{parent}(j)} \cdot \mathbf{L}_j$

---

## 5. Forward Kinematics (FK)

### 5.1 Definition

**Forward kinematics** computes the position and orientation of each bone given the joint angles (local rotations). Starting from the root, we multiply transforms down the chain:

$$\mathbf{W}_0 = \mathbf{L}_0$$
$$\mathbf{W}_j = \mathbf{W}_{\text{parent}(j)} \cdot \mathbf{L}_j$$

The end-effector position (e.g., the hand) is:

$$\mathbf{p}_{\text{end}} = \mathbf{W}_n \cdot \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}$$

### 5.2 FK Implementation

```python
import numpy as np

class Bone:
    """A bone in a skeletal hierarchy."""

    def __init__(self, name, length, parent=None):
        self.name = name
        self.length = length         # Bone length along local X axis
        self.parent = parent
        self.children = []

        # Local transform: rotation angle (degrees) around Z axis (2D simplification)
        self.local_angle = 0.0

        # Computed during FK
        self.world_transform = np.eye(3)  # 2D: 3x3 homogeneous
        self.world_position = np.zeros(2)
        self.world_end = np.zeros(2)

        if parent:
            parent.children.append(self)

    def local_matrix(self):
        """Build the local transform: rotate by angle, then translate by length."""
        theta = np.radians(self.local_angle)
        c, s = np.cos(theta), np.sin(theta)
        # Why rotation then translation: rotation happens at the joint,
        # then we offset along the rotated bone direction
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])
        T = np.array([[1, 0, self.length],
                      [0, 1, 0],
                      [0, 0, 1]])
        return R @ T


def forward_kinematics(bone, parent_transform=None):
    """
    Recursively compute world transforms for the skeleton.
    Each bone's world transform = parent's world * local transform.
    """
    if parent_transform is None:
        parent_transform = np.eye(3)

    # The joint position is at the origin of the parent transform
    bone.world_transform = parent_transform @ bone.local_matrix()

    # Joint start position: where the parent ends
    bone.world_position = parent_transform[:2, 2]

    # Joint end position: transform the local endpoint
    bone.world_end = bone.world_transform[:2, 2]

    for child in bone.children:
        forward_kinematics(child, bone.world_transform)


# Build a 3-bone arm (2D)
root = Bone("Shoulder", length=0.0)       # Root at origin, no length
upper_arm = Bone("UpperArm", length=2.0, parent=root)
forearm = Bone("Forearm", length=1.5, parent=upper_arm)
hand = Bone("Hand", length=1.0, parent=forearm)

# Set joint angles
root.local_angle = 0.0
upper_arm.local_angle = 45.0    # Shoulder rotated 45 degrees
forearm.local_angle = -30.0     # Elbow bent -30 degrees
hand.local_angle = 10.0         # Wrist slightly rotated

# Compute FK
forward_kinematics(root)

# Print results
for bone in [root, upper_arm, forearm, hand]:
    print(f"  {bone.name:>10}: start=({bone.world_position[0]:.2f}, {bone.world_position[1]:.2f})"
          f"  end=({bone.world_end[0]:.2f}, {bone.world_end[1]:.2f})")
```

---

## 6. Inverse Kinematics (IK)

### 6.1 The Problem

**Inverse kinematics** is the reverse of FK: given a desired end-effector position (e.g., "put the hand here"), find the joint angles that achieve it.

FK: joint angles $\rightarrow$ end-effector position (straightforward)
IK: end-effector position $\rightarrow$ joint angles (underdetermined, multiple solutions)

IK is harder because:
- The system is typically **underdetermined** (more DOF than constraints)
- The mapping is **nonlinear** (involves trigonometric functions)
- Joint limits must be respected

### 6.2 Analytical IK (2-Bone Case)

For a simple 2-bone chain (e.g., upper arm + forearm), we can solve analytically using the **law of cosines**:

Given bone lengths $L_1, L_2$ and target distance $d = \|\mathbf{target} - \mathbf{root}\|$:

$$\cos\theta_2 = \frac{d^2 - L_1^2 - L_2^2}{2L_1 L_2}$$

$$\theta_1 = \text{atan2}(t_y, t_x) - \text{atan2}(L_2 \sin\theta_2, L_1 + L_2\cos\theta_2)$$

This gives an exact solution (or two solutions: elbow-up and elbow-down).

### 6.3 CCD (Cyclic Coordinate Descent)

**CCD** is an iterative algorithm for arbitrary-length chains:

1. Start from the **last** joint in the chain
2. Rotate it so the end-effector points toward the target
3. Move to the next joint toward the root and repeat
4. Cycle through all joints until the end-effector is close enough to the target

```
function CCD(chain, target, max_iterations):
    for iter in range(max_iterations):
        for j from last_joint to first_joint:
            // Vector from joint j to end-effector
            to_end = end_effector_pos - joint_pos[j]
            // Vector from joint j to target
            to_target = target - joint_pos[j]
            // Rotate joint j to align to_end with to_target
            angle = angle_between(to_end, to_target)
            joint_angle[j] += angle
            // Recompute FK
            update_chain()
        if distance(end_effector, target) < threshold:
            break
```

**Pros**: Simple, handles long chains, easy to add joint limits.
**Cons**: Can converge slowly, may produce unnatural poses (excessive winding).

### 6.4 FABRIK (Forward And Backward Reaching Inverse Kinematics)

**FABRIK** (Aristidou & Lasenby, 2011) works by iteratively adjusting joint positions rather than angles:

**Forward pass** (from end-effector to root):
1. Move end-effector to the target
2. Move the second-to-last joint toward the end-effector, maintaining bone length
3. Continue toward the root

**Backward pass** (from root to end-effector):
1. Move root back to its original position
2. Move the next joint toward the root, maintaining bone length
3. Continue toward the end-effector

Repeat passes until convergence.

```python
def fabrik(joint_positions, bone_lengths, target, tolerance=0.01, max_iter=50):
    """
    FABRIK inverse kinematics solver.

    joint_positions: list of 2D positions [(x,y), ...]
    bone_lengths: list of distances between consecutive joints
    target: desired end-effector position (x,y)
    """
    n = len(joint_positions)
    positions = [np.array(p, dtype=float) for p in joint_positions]
    target = np.array(target, dtype=float)

    # Check if target is reachable
    total_length = sum(bone_lengths)
    root_to_target = np.linalg.norm(target - positions[0])
    if root_to_target > total_length:
        # Target unreachable: stretch toward it
        direction = (target - positions[0]) / root_to_target
        for i in range(1, n):
            positions[i] = positions[i-1] + direction * bone_lengths[i-1]
        return positions

    root_pos = positions[0].copy()

    for iteration in range(max_iter):
        # Check convergence
        error = np.linalg.norm(positions[-1] - target)
        if error < tolerance:
            break

        # --- Forward pass: move end-effector to target ---
        positions[-1] = target.copy()
        for i in range(n - 2, -1, -1):
            # Move joint i toward joint i+1, maintaining bone length
            direction = positions[i] - positions[i+1]
            dist = np.linalg.norm(direction)
            if dist > 1e-10:
                direction /= dist
            positions[i] = positions[i+1] + direction * bone_lengths[i]

        # --- Backward pass: move root back to original position ---
        positions[0] = root_pos.copy()
        for i in range(1, n):
            # Move joint i toward joint i-1, maintaining bone length
            direction = positions[i] - positions[i-1]
            dist = np.linalg.norm(direction)
            if dist > 1e-10:
                direction /= dist
            positions[i] = positions[i-1] + direction * bone_lengths[i-1]

    return positions


# Demo: 4-joint chain reaching for a target
joints = [(0, 0), (2, 0), (3.5, 0), (4.5, 0)]
lengths = [2.0, 1.5, 1.0]
target = (3.0, 3.0)

result = fabrik(joints, lengths, target, tolerance=0.001)
print("FABRIK result:")
for i, pos in enumerate(result):
    print(f"  Joint {i}: ({pos[0]:.3f}, {pos[1]:.3f})")
end_error = np.linalg.norm(result[-1] - np.array(target))
print(f"  End-effector error: {end_error:.6f}")
```

### 6.5 CCD vs. FABRIK

| Property | CCD | FABRIK |
|----------|-----|--------|
| Works with | Angles (rotational joints) | Positions (then derive angles) |
| Convergence speed | Moderate | Fast (often 3-5 iterations) |
| Natural-looking | Can produce winding | Generally more natural |
| Joint constraints | Easy to add per-joint | Possible but requires projection |
| Implementation | Simple | Very simple |
| Use cases | Games, robotics | Games, procedural animation |

---

## 7. Vertex Skinning

### 7.1 The Problem

Once the skeleton is posed (via FK, IK, or keyframe animation), we need to **deform the mesh** to follow the bones. Each vertex may be influenced by multiple bones (e.g., a vertex near the elbow is affected by both the upper arm and forearm bones).

### 7.2 Linear Blend Skinning (LBS)

**LBS** (also called "smooth skinning" or "skeletal subspace deformation") is the standard technique:

$$\mathbf{v}' = \sum_{j=1}^{n} w_j \cdot \mathbf{S}_j \cdot \mathbf{v}$$

where:
- $\mathbf{v}$ is the vertex position in bind pose
- $\mathbf{S}_j = \mathbf{W}_j \cdot \mathbf{B}_j^{-1}$ is the skinning matrix for bone $j$
- $w_j$ is the weight of bone $j$ on this vertex ($\sum_j w_j = 1$)
- $\mathbf{v}'$ is the deformed vertex position

Typically, each vertex is influenced by at most 4 bones (a GPU-friendly limit).

**The "candy wrapper" artifact**: LBS linearly blends transformation matrices. When a joint rotates significantly (e.g., a forearm twist of 180 degrees), the interpolated matrices produce a **volume collapse** that looks like a candy wrapper twist. This is the main limitation of LBS.

### 7.3 Dual Quaternion Skinning

**Dual quaternion skinning** (Kavan et al., 2007) replaces matrix blending with dual quaternion blending, which preserves volume:

A dual quaternion $\hat{q} = q_r + \epsilon q_d$ represents a rigid transformation (rotation $q_r$ + translation encoded in $q_d$). Blending dual quaternions avoids the volume collapse:

$$\hat{q}_{\text{blend}} = \frac{\sum_j w_j \hat{q}_j}{\|\sum_j w_j \hat{q}_j\|}$$

This is more expensive than LBS but eliminates candy-wrapper artifacts. Most modern game engines offer it as an option.

### 7.4 Comparison

| Property | LBS | Dual Quaternion |
|----------|-----|-----------------|
| Volume preservation | No (candy wrapper) | Yes |
| Speed | Very fast | Slightly slower |
| GPU support | Native in all engines | Widely supported |
| Implementation | Simple matrix blend | Dual quaternion math |
| Scaling support | Yes | Requires extra handling |

---

## 8. Morph Targets (Blend Shapes)

### 8.1 Concept

**Morph targets** (also called **blend shapes**) store complete deformed versions of a mesh. The final mesh is a weighted blend of the base mesh and one or more targets:

$$\mathbf{v}_{\text{final}} = \mathbf{v}_{\text{base}} + \sum_{k=1}^{K} \alpha_k (\mathbf{v}_{\text{target}_k} - \mathbf{v}_{\text{base}})$$

where $\alpha_k \in [0, 1]$ is the blend weight for target $k$.

### 8.2 Facial Animation

Morph targets are the standard for **facial animation**:
- Each target represents a facial expression or phoneme: "smile", "frown", "blink", "mouth_open", "jaw_left", etc.
- The FACS (Facial Action Coding System) defines standardized **action units** that correspond to individual muscle movements
- Animators keyframe the blend weights to create speech and expressions

A typical game character face has 30-60 blend shapes. Film characters may have 200+.

### 8.3 Implementation

```python
def apply_blend_shapes(base_vertices, targets, weights):
    """
    Compute blended vertex positions from base mesh and morph targets.

    base_vertices: (N, 3) array of base mesh positions
    targets: list of (N, 3) arrays, each a morph target
    weights: list of floats, blend weight for each target
    """
    result = base_vertices.copy()
    for target, weight in zip(targets, weights):
        if abs(weight) > 1e-6:
            # Why store deltas: saves memory and makes blending a simple add
            delta = target - base_vertices
            result += weight * delta
    return result


# Demo: simple face mesh (triangle of 3 vertices)
base = np.array([[0.0, 0.0, 0.0],    # Left corner of mouth
                  [1.0, 0.0, 0.0],    # Right corner of mouth
                  [0.5, 0.5, 0.0]])   # Upper lip

smile_target = np.array([[-.1, 0.2, 0.0],  # Left corner up
                          [1.1, 0.2, 0.0],  # Right corner up
                          [0.5, 0.6, 0.0]]) # Upper lip slightly up

frown_target = np.array([[0.1, -0.2, 0.0], # Left corner down
                          [0.9, -0.2, 0.0], # Right corner down
                          [0.5, 0.3, 0.0]]) # Upper lip slightly down

# Half smile + slight frown = smirk
result = apply_blend_shapes(base, [smile_target, frown_target], [0.5, 0.2])
print("Blended vertices (smirk):")
for i, v in enumerate(result):
    print(f"  v{i}: ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")
```

### 8.4 Morph Targets vs. Skeletal Animation

| Aspect | Morph Targets | Skeletal Animation |
|--------|---------------|-------------------|
| Use case | Facial, soft deformation | Body, rigid limbs |
| Storage | Full mesh per target | Skeleton + weights |
| Expressiveness | Any arbitrary deformation | Limited by skeleton topology |
| Performance | Vertex additions | Matrix multiplies |
| Combinability | Additive blending | FK/IK chain |
| Common combo | Face (blend shapes) + body (skeleton) |

---

## 9. Keyframe Interpolation Pipeline

The complete animation pipeline at each frame:

```
1. Determine current time t within animation clip
2. For each bone:
   a. Find surrounding keyframes (before and after t)
   b. Compute interpolation factor: alpha = (t - t_prev) / (t_next - t_prev)
   c. Interpolate rotation: slerp(q_prev, q_next, alpha)
   d. Interpolate position: lerp(p_prev, p_next, alpha)
   e. Build local transform matrix from interpolated rotation + position
3. Forward Kinematics: propagate local transforms to world transforms
4. Compute skinning matrices: S_j = W_j * B_j_inverse
5. Vertex skinning: deform each vertex using weighted skinning matrices
6. Apply blend shapes (if any)
7. Upload deformed vertices to GPU for rendering
```

Steps 4-5 are typically performed on the **GPU** via vertex shaders, with bone matrices uploaded as uniform arrays.

---

## 10. Animation Blending

### 10.1 Cross-Fade

Smoothly transition between two animations (e.g., walk to run) by interpolating their outputs:

$$\mathbf{L}_j^{\text{blend}} = \text{slerp}(\mathbf{L}_j^{\text{walk}}, \mathbf{L}_j^{\text{run}}, \alpha)$$

where $\alpha$ ramps from 0 to 1 over the transition period.

### 10.2 Layered Animation

Combine animations additively:
- **Base layer**: Full-body locomotion (walk, run, jump)
- **Upper body layer**: Aim weapon, wave hand
- **Facial layer**: Blend shapes for expression

Each layer can have a **mask** that limits which bones it affects.

### 10.3 Animation State Machines

Games use state machines to manage animation transitions:

```
[Idle] --move--> [Walk] --speed_up--> [Run]
  |                 |                    |
  v                 v                    v
[Jump]           [Jump]              [Jump]
  |                 |                    |
  v                 v                    v
[Land]           [Land]              [Land]
```

Transitions specify blend durations and conditions (speed thresholds, input events).

---

## Summary

| Concept | Key Idea |
|---------|----------|
| Keyframe animation | Define poses at key times; interpolate between them |
| Lerp | $\text{lerp}(a, b, t) = (1-t)a + tb$ -- linear, not suitable for rotations |
| Slerp | Constant angular velocity interpolation on quaternion hypersphere |
| Bezier curves | Smooth parametric curves defined by control points; used for animation paths |
| Skeletal animation | Hierarchy of bones; animate bones, mesh follows |
| Forward kinematics | Joint angles $\to$ end-effector position; multiply transforms down the chain |
| Inverse kinematics | End-effector position $\to$ joint angles; CCD, FABRIK, analytical |
| LBS | Weighted sum of skinning matrices; fast but candy-wrapper artifact |
| Dual quaternion | Volume-preserving skinning; eliminates candy-wrapper at modest cost |
| Morph targets | Blend between stored mesh deformations; standard for facial animation |

## Exercises

1. **Slerp implementation**: Implement slerp and verify that it produces constant angular velocity by measuring the angle between successive interpolated quaternions.

2. **2-Bone analytical IK**: Implement the analytical 2-bone IK solver using the law of cosines. Visualize both the "elbow-up" and "elbow-down" solutions.

3. **FABRIK 3D**: Extend the FABRIK implementation to 3D. Create a 5-bone chain and solve for several target positions. Visualize the convergence over iterations.

4. **FK chain animation**: Create a 4-bone arm. Animate the joint angles using sinusoidal functions at different frequencies. Visualize the resulting end-effector path.

5. **Blend shape face**: Create a simplified face mesh (10-20 vertices). Define 4 morph targets (smile, frown, surprise, blink). Animate blend weights over time to create a sequence of expressions.

6. **LBS artifact**: Implement LBS for a cylindrical mesh (tube) with two bones. Rotate one bone by 0, 45, 90, 135, and 180 degrees. Observe and document the candy-wrapper collapse. Then implement dual quaternion skinning and compare.

## Further Reading

- Parent, R. *Computer Animation: Algorithms and Techniques*, 3rd ed. Morgan Kaufmann, 2012. (Comprehensive animation reference)
- Aristidou, A. and Lasenby, J. "FABRIK: A Fast, Iterative Solver for the Inverse Kinematics Problem." *Graphical Models*, 2011. (The FABRIK algorithm)
- Kavan, L. et al. "Skinning with Dual Quaternions." *I3D*, 2007. (Volume-preserving skinning)
- Shoemake, K. "Animating Rotation with Quaternion Curves." *SIGGRAPH*, 1985. (Introduction of slerp to graphics)
- Lewis, J.P. et al. "Practice and Theory of Blendshape Facial Models." *Eurographics STAR*, 2014. (Survey of blend shape techniques)
