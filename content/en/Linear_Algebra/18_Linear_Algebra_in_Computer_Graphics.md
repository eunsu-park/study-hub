# Lesson 18: Linear Algebra in Computer Graphics

[Previous: Lesson 17](./17_Linear_Algebra_in_Deep_Learning.md) | [Overview](./00_Overview.md) | [Next: Lesson 19](./19_Randomized_Linear_Algebra.md)

---

## Learning Objectives

- Represent 2D and 3D geometric transformations (rotation, scaling, translation, shearing) as matrices
- Understand homogeneous coordinates and why they unify affine transformations into matrix multiplication
- Derive and apply perspective and orthographic projection matrices
- Explain quaternions and their advantages over Euler angles for representing 3D rotations
- Build a complete camera transform pipeline (model, view, projection)
- Implement transformations in Python with NumPy

---

## 1. 2D Transformations

### 1.1 Linear Transformations as Matrices

Every linear transformation in 2D can be represented as multiplication by a $2 \times 2$ matrix:

| Transformation | Matrix | Effect |
|---|---|---|
| Scaling | $\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$ | Scale axes independently |
| Rotation | $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ | Rotate by $\theta$ counterclockwise |
| Reflection (x-axis) | $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$ | Flip vertically |
| Shear (x) | $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$ | Shear along x-axis |

```python
import numpy as np
import matplotlib.pyplot as plt

def apply_transform(points, M):
    """Apply 2x2 transformation matrix M to a set of 2D points."""
    return (M @ points.T).T

def plot_shape(ax, points, label, color='blue', fill=True):
    """Plot a closed polygon."""
    pts = np.vstack([points, points[0]])
    if fill:
        ax.fill(pts[:, 0], pts[:, 1], alpha=0.3, color=color)
    ax.plot(pts[:, 0], pts[:, 1], '-o', color=color, markersize=4, label=label)

# Define a simple shape (unit square)
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

# Transformations
transformations = {
    'Scaling (2x, 0.5y)': np.array([[2, 0], [0, 0.5]]),
    'Rotation (45 deg)': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                    [np.sin(np.pi/4),  np.cos(np.pi/4)]]),
    'Shear (k=0.5)': np.array([[1, 0.5], [0, 1]]),
    'Reflection (y-axis)': np.array([[-1, 0], [0, 1]]),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, M) in zip(axes.flat, transformations.items()):
    transformed = apply_transform(square, M)
    plot_shape(ax, square, 'Original', 'blue')
    plot_shape(ax, transformed, 'Transformed', 'red')
    ax.set_title(name)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 2.5)

plt.tight_layout()
plt.show()
```

### 1.2 Composition of Transformations

Composing transformations is just matrix multiplication. The order matters because matrix multiplication is not commutative.

```python
# Rotate then scale vs Scale then rotate
theta = np.pi / 6  # 30 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
S = np.array([[2, 0],
              [0, 1]])

# Compose in different orders
RS = R @ S  # First scale, then rotate
SR = S @ R  # First rotate, then scale

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
triangle = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)

plot_shape(axes[0], triangle, 'Original', 'blue')
axes[0].set_title('Original')

plot_shape(axes[1], triangle, 'Original', 'blue')
plot_shape(axes[1], apply_transform(triangle, RS), 'R(S(x))', 'red')
axes[1].set_title('Scale then Rotate (R @ S)')

plot_shape(axes[2], triangle, 'Original', 'blue')
plot_shape(axes[2], apply_transform(triangle, SR), 'S(R(x))', 'green')
axes[2].set_title('Rotate then Scale (S @ R)')

for ax in axes:
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1, 2)

plt.tight_layout()
plt.show()

print(f"R @ S = S @ R? {np.allclose(RS, SR)}")
print(f"R @ S:\n{np.round(RS, 3)}")
print(f"S @ R:\n{np.round(SR, 3)}")
```

---

## 2. Homogeneous Coordinates

### 2.1 The Problem with Translation

Translation $\mathbf{x}' = \mathbf{x} + \mathbf{t}$ is an **affine** transformation, not a linear one. It cannot be represented as $\mathbf{x}' = M\mathbf{x}$ with a $2 \times 2$ matrix.

**Homogeneous coordinates** solve this by adding a third coordinate. A 2D point $(x, y)$ becomes $(x, y, 1)$, and all affine transformations become $3 \times 3$ matrix multiplications.

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

```python
def to_homogeneous(points):
    """Convert 2D points to homogeneous coordinates."""
    return np.hstack([points, np.ones((len(points), 1))])

def from_homogeneous(points_h):
    """Convert from homogeneous back to 2D."""
    return points_h[:, :2] / points_h[:, 2:3]

def translation_matrix_2d(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])

def rotation_matrix_2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def scaling_matrix_2d(sx, sy):
    return np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0,  0, 1]])

# Rotate around a specific point (not the origin)
def rotate_around_point(theta, cx, cy):
    """Rotate by theta around point (cx, cy)."""
    T1 = translation_matrix_2d(-cx, -cy)   # Translate to origin
    R = rotation_matrix_2d(theta)            # Rotate
    T2 = translation_matrix_2d(cx, cy)       # Translate back
    return T2 @ R @ T1

# Example: rotate a square around its center
square = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=float)
center = square.mean(axis=0)

fig, ax = plt.subplots(figsize=(8, 8))
plot_shape(ax, square, 'Original', 'blue')

for angle_deg in [30, 60, 90]:
    theta = np.radians(angle_deg)
    M = rotate_around_point(theta, center[0], center[1])
    square_h = to_homogeneous(square)
    rotated_h = (M @ square_h.T).T
    rotated = from_homogeneous(rotated_h)
    plot_shape(ax, rotated, f'{angle_deg} deg', plt.cm.Set1(angle_deg / 120))

ax.plot(*center, 'k*', markersize=15, label='Center')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Rotation Around Center Using Homogeneous Coordinates')
plt.tight_layout()
plt.show()
```

---

## 3. 3D Transformations

### 3.1 Basic 3D Matrices

In 3D, transformations use $4 \times 4$ homogeneous matrices:

```python
def translation_3d(tx, ty, tz):
    return np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]])

def scaling_3d(sx, sy, sz):
    return np.array([[sx, 0,  0,  0],
                     [0,  sy, 0,  0],
                     [0,  0,  sz, 0],
                     [0,  0,  0,  1]])

def rotation_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0,  0, 0],
                     [0, c, -s, 0],
                     [0, s,  c, 0],
                     [0, 0,  0, 1]])

def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])

def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]])

# Rotation around arbitrary axis (Rodrigues' formula)
def rotation_axis_angle(axis, theta):
    """Rotation matrix around an arbitrary axis by angle theta."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    M = np.eye(4)
    M[:3, :3] = R
    return M

# Example: compound 3D transformation
M = translation_3d(1, 2, 3) @ rotation_z(np.pi / 4) @ scaling_3d(2, 2, 2)
print(f"Compound transformation (4x4):\n{np.round(M, 3)}")

# Apply to a 3D point
p = np.array([1, 0, 0, 1])  # Homogeneous
p_transformed = M @ p
print(f"\nOriginal:    {p[:3]}")
print(f"Transformed: {p_transformed[:3]}")
```

### 3.2 Euler Angles and Gimbal Lock

**Euler angles** represent a rotation as three sequential rotations around coordinate axes. The most common convention is roll-pitch-yaw ($R = R_z(\psi) R_y(\theta) R_x(\phi)$).

The critical problem is **gimbal lock**: when the pitch angle is $\pm 90°$, two axes align and one degree of freedom is lost.

```python
# Demonstrate gimbal lock
def euler_to_rotation(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to rotation matrix."""
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)

# Normal case: all three axes are independent
R_normal = euler_to_rotation(np.radians(30), np.radians(20), np.radians(45))
print(f"Normal rotation (rank 3 of 3x3 submatrix):")
print(f"  R[:3,:3] rank: {np.linalg.matrix_rank(R_normal[:3,:3])}")

# Gimbal lock: pitch = 90 degrees
R_gimbal = euler_to_rotation(np.radians(30), np.radians(90), np.radians(45))
print(f"\nGimbal lock (pitch = 90 deg):")
print(f"  R[:3,:3]:\n{np.round(R_gimbal[:3,:3], 4)}")
print("  Roll and yaw now affect the same axis!")

# Show that changing roll or yaw produces the same rotation
R1 = euler_to_rotation(np.radians(50), np.radians(90), np.radians(25))
R2 = euler_to_rotation(np.radians(25), np.radians(90), np.radians(50))
print(f"\n  R(roll=50, pitch=90, yaw=25) == R(roll=25, pitch=90, yaw=50)?")
print(f"  {np.allclose(R1[:3,:3], R2[:3,:3])}")
```

---

## 4. Quaternions

### 4.1 Quaternion Basics

A **quaternion** is a 4D number $q = w + xi + yj + zk$ where $i^2 = j^2 = k^2 = ijk = -1$. Written as $q = (w, \mathbf{v})$ where $w$ is the scalar part and $\mathbf{v} = (x, y, z)$ is the vector part.

A unit quaternion ($\|q\| = 1$) represents a rotation by angle $\theta$ around axis $\hat{\mathbf{u}}$:

$$q = \left(\cos\frac{\theta}{2}, \sin\frac{\theta}{2} \hat{\mathbf{u}}\right)$$

```python
class Quaternion:
    def __init__(self, w, x, y, z):
        self.q = np.array([w, x, y, z], dtype=float)

    @property
    def w(self): return self.q[0]
    @property
    def x(self): return self.q[1]
    @property
    def y(self): return self.q[2]
    @property
    def z(self): return self.q[3]

    @staticmethod
    def from_axis_angle(axis, angle):
        """Create quaternion from axis-angle representation."""
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        half = angle / 2
        return Quaternion(np.cos(half),
                         np.sin(half) * axis[0],
                         np.sin(half) * axis[1],
                         np.sin(half) * axis[2])

    def normalize(self):
        norm = np.linalg.norm(self.q)
        return Quaternion(*(self.q / norm))

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other):
        """Hamilton product."""
        w1, x1, y1, z1 = self.q
        w2, x2, y2, z2 = other.q
        return Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )

    def rotate_vector(self, v):
        """Rotate a 3D vector using this quaternion."""
        v_quat = Quaternion(0, v[0], v[1], v[2])
        result = self * v_quat * self.conjugate()
        return np.array([result.x, result.y, result.z])

    def to_rotation_matrix(self):
        """Convert quaternion to 3x3 rotation matrix."""
        w, x, y, z = self.q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
        ])

    def __repr__(self):
        return f"Q({self.w:.4f}, {self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

# Example: 90 degree rotation around z-axis
q = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
print(f"Quaternion: {q}")

v = np.array([1, 0, 0])
v_rotated = q.rotate_vector(v)
print(f"Rotate [1,0,0] by 90 deg around z: {np.round(v_rotated, 4)}")
# Expected: [0, 1, 0]

# Compare with rotation matrix
R = q.to_rotation_matrix()
v_rotated_matrix = R @ v
print(f"Using rotation matrix: {np.round(v_rotated_matrix, 4)}")
print(f"Match: {np.allclose(v_rotated, v_rotated_matrix)}")
```

### 4.2 SLERP (Spherical Linear Interpolation)

Quaternions enable smooth interpolation between rotations via **SLERP**:

$$\text{slerp}(q_0, q_1, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} q_0 + \frac{\sin(t\Omega)}{\sin\Omega} q_1$$

where $\cos\Omega = q_0 \cdot q_1$.

```python
def slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions."""
    dot = np.dot(q0.q, q1.q)

    # If dot is negative, negate one quaternion (shortest path)
    if dot < 0:
        q1 = Quaternion(*(-q1.q))
        dot = -dot

    if dot > 0.9995:
        # Very close: use linear interpolation
        result = Quaternion(*(q0.q + t * (q1.q - q0.q)))
        return result.normalize()

    omega = np.arccos(np.clip(dot, -1, 1))
    sin_omega = np.sin(omega)
    s0 = np.sin((1 - t) * omega) / sin_omega
    s1 = np.sin(t * omega) / sin_omega
    return Quaternion(*(s0 * q0.q + s1 * q1.q))

# Interpolate between two rotations
q_start = Quaternion.from_axis_angle([0, 0, 1], 0)          # No rotation
q_end = Quaternion.from_axis_angle([0, 0, 1], np.pi / 2)    # 90 deg around z

print("SLERP interpolation:")
v = np.array([1, 0, 0])
for t in np.linspace(0, 1, 5):
    q_t = slerp(q_start, q_end, t)
    v_rot = q_t.rotate_vector(v)
    angle = np.degrees(np.arccos(np.clip(np.dot(v, v_rot), -1, 1)))
    print(f"  t={t:.2f}: angle={angle:.1f} deg, vector={np.round(v_rot, 3)}")
```

### 4.3 Quaternion vs Euler vs Rotation Matrix

| Representation | Parameters | Gimbal lock | Interpolation | Composition | Normalization |
|---|---|---|---|---|---|
| Euler angles | 3 | Yes | Poor | Slow | Not needed |
| Rotation matrix | 9 | No | Poor | Matrix multiply | Re-orthogonalize |
| Quaternion | 4 | No | SLERP (smooth) | Hamilton product | Normalize to unit |

---

## 5. Projection Matrices

### 5.1 Orthographic Projection

Orthographic projection maps 3D coordinates to 2D by dropping one coordinate. It preserves parallel lines and ratios of distances.

$$P_{\text{ortho}} = \begin{bmatrix} \frac{2}{r-l} & 0 & 0 & -\frac{r+l}{r-l} \\ 0 & \frac{2}{t-b} & 0 & -\frac{t+b}{t-b} \\ 0 & 0 & -\frac{2}{f-n} & -\frac{f+n}{f-n} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

```python
def orthographic_projection(left, right, bottom, top, near, far):
    """Orthographic projection matrix."""
    return np.array([
        [2/(right-left), 0, 0, -(right+left)/(right-left)],
        [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
        [0, 0, -2/(far-near), -(far+near)/(far-near)],
        [0, 0, 0, 1]
    ])

P_ortho = orthographic_projection(-1, 1, -1, 1, 0.1, 100)
print(f"Orthographic projection matrix:\n{np.round(P_ortho, 4)}")
```

### 5.2 Perspective Projection

Perspective projection creates the illusion of depth: distant objects appear smaller. This is a **projective** (non-linear) transformation implemented via homogeneous coordinates.

$$P_{\text{persp}} = \begin{bmatrix} \frac{2n}{r-l} & 0 & \frac{r+l}{r-l} & 0 \\ 0 & \frac{2n}{t-b} & \frac{t+b}{t-b} & 0 \\ 0 & 0 & -\frac{f+n}{f-n} & -\frac{2fn}{f-n} \\ 0 & 0 & -1 & 0 \end{bmatrix}$$

After multiplication, we must divide by the $w$ component (perspective divide).

```python
def perspective_projection(fov_y, aspect, near, far):
    """Perspective projection matrix (OpenGL-style).

    Args:
        fov_y: vertical field of view in radians
        aspect: width / height
        near, far: near and far clipping planes
    """
    f = 1.0 / np.tan(fov_y / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]
    ])

P_persp = perspective_projection(np.radians(60), 16/9, 0.1, 100)
print(f"Perspective projection matrix:\n{np.round(P_persp, 4)}")

# Project a 3D point
point_3d = np.array([2, 1, -5, 1])  # In camera space (z is negative = in front)
point_clip = P_persp @ point_3d

# Perspective divide
point_ndc = point_clip[:3] / point_clip[3]
print(f"\n3D point: {point_3d[:3]}")
print(f"Clip coordinates: {np.round(point_clip, 4)}")
print(f"NDC (after perspective divide): {np.round(point_ndc, 4)}")
```

### 5.3 Depth and Perspective

```python
# Demonstrate perspective: objects at different depths
points = np.array([
    [1, 0, -2, 1],   # Close
    [1, 0, -5, 1],   # Medium
    [1, 0, -10, 1],  # Far
    [1, 0, -50, 1],  # Very far
])

P = perspective_projection(np.radians(60), 1.0, 0.1, 100)

print("Perspective effect on same-sized objects at different depths:")
print(f"{'Depth':>8s}  {'Projected X':>12s}  {'Apparent size':>14s}")
for p in points:
    p_clip = P @ p
    p_ndc = p_clip[:2] / p_clip[3]
    print(f"{p[2]:8.1f}  {p_ndc[0]:12.4f}  {abs(p_ndc[0]):14.4f}")
```

---

## 6. Camera Transform Pipeline

### 6.1 The Full Pipeline

The complete transformation from world coordinates to screen pixels involves:

1. **Model matrix** ($M$): object space to world space
2. **View matrix** ($V$): world space to camera space
3. **Projection matrix** ($P$): camera space to clip coordinates
4. **Perspective divide**: clip to NDC (normalized device coordinates)
5. **Viewport transform**: NDC to screen pixels

$$\mathbf{p}_{\text{clip}} = P \cdot V \cdot M \cdot \mathbf{p}_{\text{object}}$$

```python
def look_at(eye, target, up):
    """Compute the view matrix (camera transform).

    Args:
        eye: camera position in world space
        target: point the camera is looking at
        up: world up direction
    """
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    up = np.array(up, dtype=float)

    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    up_actual = np.cross(right, forward)

    V = np.eye(4)
    V[0, :3] = right
    V[1, :3] = up_actual
    V[2, :3] = -forward  # Camera looks along -z in OpenGL convention
    V[:3, 3] = -V[:3, :3] @ eye

    return V

def viewport_transform(width, height):
    """Map NDC [-1,1] x [-1,1] to screen [0,width] x [0,height]."""
    return np.array([
        [width/2, 0, 0, width/2],
        [0, height/2, 0, height/2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# Full pipeline example
# 1. Model: place a cube at world position (3, 0, -5), rotated 30 deg
M = translation_3d(3, 0, -5) @ rotation_y(np.radians(30))

# 2. View: camera at (0, 2, 0), looking at (3, 0, -5)
V = look_at(eye=[0, 2, 0], target=[3, 0, -5], up=[0, 1, 0])

# 3. Projection
P = perspective_projection(np.radians(60), 16/9, 0.1, 100)

# 4. Combined MVP matrix
MVP = P @ V @ M

# Transform cube vertices
cube_vertices = np.array([
    [-1, -1, -1, 1], [1, -1, -1, 1], [1, 1, -1, 1], [-1, 1, -1, 1],
    [-1, -1,  1, 1], [1, -1,  1, 1], [1, 1,  1, 1], [-1, 1,  1, 1],
], dtype=float)

clip_coords = (MVP @ cube_vertices.T).T
ndc = clip_coords[:, :3] / clip_coords[:, 3:4]

# Viewport
VP = viewport_transform(1920, 1080)
screen = (VP @ np.hstack([ndc, np.ones((len(ndc), 1))]).T).T

print("Cube vertices through the pipeline:")
print(f"{'Vertex':>8s}  {'World':>20s}  {'Screen (x, y)':>16s}")
for i, v in enumerate(cube_vertices):
    world = (M @ v)[:3]
    scr = screen[i, :2]
    print(f"{i:8d}  ({world[0]:5.1f}, {world[1]:5.1f}, {world[2]:5.1f})  "
          f"({scr[0]:7.1f}, {scr[1]:7.1f})")
```

### 6.2 Wireframe Rendering

```python
# Render a wireframe cube
cube_edges = [
    (0,1), (1,2), (2,3), (3,0),  # Front face
    (4,5), (5,6), (6,7), (7,4),  # Back face
    (0,4), (1,5), (2,6), (3,7),  # Connecting edges
]

fig, ax = plt.subplots(figsize=(10, 6))

for i, j in cube_edges:
    ax.plot([screen[i, 0], screen[j, 0]],
            [screen[i, 1], screen[j, 1]], 'b-', linewidth=2)

ax.scatter(screen[:, 0], screen[:, 1], c='red', s=50, zorder=5)
for i in range(len(screen)):
    ax.annotate(str(i), (screen[i, 0] + 10, screen[i, 1] + 10))

ax.set_xlim(0, 1920)
ax.set_ylim(0, 1080)
ax.set_aspect('equal')
ax.invert_yaxis()  # Screen coordinates: y increases downward
ax.set_title('Wireframe Cube (Perspective Projection)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 7. Normal Transformation

When transforming geometry, surface normals must be transformed differently from vertices. If vertices are transformed by $M$, normals must be transformed by $(M^{-1})^T$ (the inverse transpose).

This is because normals are **covectors** (they belong to the dual space), not vectors.

```python
# Normal transformation
def transform_normal(normal, M):
    """Transform a surface normal using the inverse-transpose of M.

    For a 4x4 model matrix M, normals use (M^{-1})^T.
    """
    M_3x3 = M[:3, :3]
    normal_matrix = np.linalg.inv(M_3x3).T
    transformed = normal_matrix @ normal
    return transformed / np.linalg.norm(transformed)

# Example: non-uniform scaling breaks normals if not using inverse-transpose
normal = np.array([0, 1, 0])  # Pointing up
M_scale = scaling_3d(2, 1, 1)[:3, :3]  # Scale x by 2

# Wrong: just apply M
wrong_normal = M_scale @ normal
wrong_normal = wrong_normal / np.linalg.norm(wrong_normal)

# Right: apply (M^{-1})^T
right_normal = transform_normal(normal, np.eye(4))
right_normal_scaled = transform_normal(normal, scaling_3d(2, 1, 1))

print(f"Original normal: {normal}")
print(f"After scaling x by 2:")
print(f"  Wrong (using M):        {np.round(wrong_normal, 4)}")
print(f"  Right (using (M^-1)^T): {np.round(right_normal_scaled, 4)}")

# For uniform scaling, both methods agree:
M_uniform = scaling_3d(3, 3, 3)
n1 = (M_uniform[:3, :3] @ normal)
n1 = n1 / np.linalg.norm(n1)
n2 = transform_normal(normal, M_uniform)
print(f"\nUniform scaling (3x): both methods agree: {np.allclose(n1, n2)}")
```

---

## Exercises

### Exercise 1: 2D Transformation Sequence

Create a Python animation (or series of plots) that shows a triangle undergoing:
1. Translation by (2, 1)
2. Rotation by 45 degrees around its centroid
3. Scaling by (1.5, 0.5) relative to its centroid

Compute the single combined matrix that performs all three at once and verify it produces the same result.

### Exercise 2: Quaternion Operations

1. Implement quaternion multiplication, conjugate, and inverse
2. Convert a quaternion to an Euler angle triplet and back
3. Demonstrate SLERP by interpolating between two rotations and plotting the trajectory of a rotated vector on the unit sphere

### Exercise 3: Camera System

Build a first-person camera system:
1. Implement `look_at` to compute the view matrix
2. Move the camera forward, backward, left, right
3. Rotate the camera (yaw and pitch)
4. Render a wireframe scene (multiple cubes at different positions) from the camera's perspective

### Exercise 4: Projection Comparison

For a scene with objects at depths $z = 1, 5, 10, 50$:
1. Compute orthographic and perspective projections
2. Plot the projected positions side by side
3. Explain visually why perspective gives the impression of depth

### Exercise 5: Normal Transformation Proof

Prove (computationally) that using the inverse transpose is correct for normal transformation:
1. Define a triangle with vertices and a surface normal
2. Apply a non-uniform scaling transformation
3. Show that $M \cdot n$ is no longer perpendicular to the transformed surface
4. Show that $(M^{-1})^T \cdot n$ is perpendicular to the transformed surface

---

[Previous: Lesson 17](./17_Linear_Algebra_in_Deep_Learning.md) | [Overview](./00_Overview.md) | [Next: Lesson 19](./19_Randomized_Linear_Algebra.md)

**License**: CC BY-NC 4.0
