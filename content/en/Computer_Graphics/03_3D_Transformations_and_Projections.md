# 03. 3D Transformations and Projections

[&larr; Previous: 2D Transformations](02_2D_Transformations.md) | [Next: Rasterization &rarr;](04_Rasterization.md)

---

## Learning Objectives

1. Extend 2D transformation concepts to 4x4 matrices for 3D homogeneous coordinates
2. Construct the Model matrix that positions objects in the world
3. Derive the View matrix (lookAt) that represents the camera's viewpoint
4. Understand and derive both perspective and orthographic projection matrices
5. Explain the full MVP (Model-View-Projection) pipeline from object space to screen space
6. Describe the viewport transform that maps NDC to pixel coordinates
7. Recognize the problems with Euler angles (gimbal lock) and understand quaternions as an alternative
8. Implement the complete MVP matrix chain in Python

---

## Why This Matters

In Lesson 02, we mastered 2D transformations using 3x3 matrices. Now we extend these ideas to 3D, where the stakes are higher: we must position objects in a 3D world, simulate a camera, and project the 3D scene onto a 2D screen. The **Model-View-Projection (MVP)** matrix chain is arguably the most important concept in real-time 3D graphics -- every single vertex in every 3D application passes through it. Understanding how a 3D point becomes a 2D pixel is the key to understanding everything from camera controls to shadow mapping to VR rendering.

---

## 1. Homogeneous Coordinates in 3D

Just as we used 3x3 matrices for 2D (adding a $w$ component), we use **4x4 matrices** for 3D:

$$\mathbf{p} = \begin{bmatrix} x \\ y \\ z \end{bmatrix} \quad \rightarrow \quad \mathbf{p}_h = \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

A general 3D affine transformation:

$$\mathbf{M} = \begin{bmatrix} & & & t_x \\ & \mathbf{R}_{3\times3} & & t_y \\ & & & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

Where $\mathbf{R}_{3\times3}$ encodes rotation/scale/shear and $(t_x, t_y, t_z)$ is the translation.

---

## 2. Basic 3D Transformations

### 2.1 Translation

$$\mathbf{T}(t_x, t_y, t_z) = \begin{bmatrix} 1 & 0 & 0 & t_x \\ 0 & 1 & 0 & t_y \\ 0 & 0 & 1 & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

### 2.2 Scaling

$$\mathbf{S}(s_x, s_y, s_z) = \begin{bmatrix} s_x & 0 & 0 & 0 \\ 0 & s_y & 0 & 0 \\ 0 & 0 & s_z & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

### 2.3 Rotation

Unlike 2D (one rotation axis), 3D has three principal rotation axes:

**Rotation about x-axis** by angle $\theta$:

$$\mathbf{R}_x(\theta) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta & 0 \\ 0 & \sin\theta & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

**Rotation about y-axis** by angle $\theta$:

$$\mathbf{R}_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta & 0 \\ 0 & 1 & 0 & 0 \\ -\sin\theta & 0 & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

**Rotation about z-axis** by angle $\theta$:

$$\mathbf{R}_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 & 0 \\ \sin\theta & \cos\theta & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

> **Note**: The sign pattern for $\mathbf{R}_y$ looks "swapped" compared to $\mathbf{R}_x$ and $\mathbf{R}_z$. This is because the $y$-axis rotation follows the right-hand rule with the cyclic ordering $y \rightarrow z \rightarrow x$.

### 2.4 Rotation About an Arbitrary Axis

To rotate by angle $\theta$ about a unit vector $\hat{\mathbf{u}} = (u_x, u_y, u_z)$, the **Rodrigues' rotation formula** gives:

$$\mathbf{R}(\hat{\mathbf{u}}, \theta) = \cos\theta \cdot \mathbf{I} + (1 - \cos\theta)(\hat{\mathbf{u}} \otimes \hat{\mathbf{u}}) + \sin\theta \cdot [\hat{\mathbf{u}}]_\times$$

Where $[\hat{\mathbf{u}}]_\times$ is the skew-symmetric cross-product matrix:

$$[\hat{\mathbf{u}}]_\times = \begin{bmatrix} 0 & -u_z & u_y \\ u_z & 0 & -u_x \\ -u_y & u_x & 0 \end{bmatrix}$$

And $\hat{\mathbf{u}} \otimes \hat{\mathbf{u}}$ is the outer product:

$$\hat{\mathbf{u}} \otimes \hat{\mathbf{u}} = \begin{bmatrix} u_x^2 & u_x u_y & u_x u_z \\ u_x u_y & u_y^2 & u_y u_z \\ u_x u_z & u_y u_z & u_z^2 \end{bmatrix}$$

---

## 3. The Model Matrix

The **Model matrix** $\mathbf{M}_{\text{model}}$ transforms vertices from **object space** (local to the 3D model) to **world space** (the shared scene coordinate system).

Typically composed of scale, rotation, and translation:

$$\mathbf{M}_{\text{model}} = \mathbf{T} \cdot \mathbf{R} \cdot \mathbf{S}$$

> **Order convention**: Scale first (change size in local space), then rotate (orient the object), then translate (position in the world). Applied right-to-left: $\mathbf{p}_{\text{world}} = \mathbf{T} \cdot \mathbf{R} \cdot \mathbf{S} \cdot \mathbf{p}_{\text{object}}$.

```python
import numpy as np

def make_translation(tx, ty, tz):
    """Create a 4x4 translation matrix."""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0,  1]
    ], dtype=float)

def make_scale(sx, sy, sz):
    """Create a 4x4 scaling matrix."""
    return np.array([
        [sx,  0,  0, 0],
        [ 0, sy,  0, 0],
        [ 0,  0, sz, 0],
        [ 0,  0,  0, 1]
    ], dtype=float)

def make_rotation_y(theta_deg):
    """Create a 4x4 rotation matrix about the y-axis."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ], dtype=float)

# Example: place a cube at position (5, 0, -3), rotated 45 deg around Y,
# scaled to half size
model_matrix = (make_translation(5, 0, -3)
                @ make_rotation_y(45)
                @ make_scale(0.5, 0.5, 0.5))
print("Model matrix:\n", model_matrix)
```

---

## 4. The View Matrix (Camera)

The **View matrix** $\mathbf{M}_{\text{view}}$ transforms from **world space** to **camera (eye) space**, where the camera is at the origin looking along the negative z-axis (by convention in OpenGL).

### 4.1 The LookAt Construction

Given:
- $\mathbf{eye}$: camera position in world space
- $\mathbf{target}$: point the camera looks at
- $\mathbf{up}$: world "up" direction (usually $(0, 1, 0)$)

We construct an orthonormal basis for the camera:

$$\mathbf{f} = \text{normalize}(\mathbf{target} - \mathbf{eye}) \quad \text{(forward direction)}$$

$$\mathbf{r} = \text{normalize}(\mathbf{f} \times \mathbf{up}) \quad \text{(right direction)}$$

$$\mathbf{u} = \mathbf{r} \times \mathbf{f} \quad \text{(true up direction)}$$

The view matrix combines a rotation (aligning camera axes with world axes) and a translation (moving the camera to the origin):

$$\mathbf{M}_{\text{view}} = \begin{bmatrix} r_x & r_y & r_z & -\mathbf{r} \cdot \mathbf{eye} \\ u_x & u_y & u_z & -\mathbf{u} \cdot \mathbf{eye} \\ -f_x & -f_y & -f_z & \mathbf{f} \cdot \mathbf{eye} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

> **Why negative $\mathbf{f}$?** In OpenGL convention, the camera looks along $-z$ in eye space. So the forward direction maps to $-z$, which means we negate $\mathbf{f}$ in the third row.

```python
def normalize(v):
    """Normalize a vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v  # Avoid division by zero
    return v / n

def look_at(eye, target, up):
    """
    Construct a view (camera) matrix.

    Why this works: the view matrix is the INVERSE of the camera's
    model matrix. Instead of computing a 4x4 inverse, we exploit the
    fact that rotation matrices are orthogonal (inverse = transpose)
    and combine with translation analytically.

    Parameters:
        eye: camera position (3D)
        target: point the camera looks at (3D)
        up: world up vector (3D)

    Returns:
        4x4 view matrix
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    # Camera basis vectors
    f = normalize(target - eye)     # Forward (into the screen)
    r = normalize(np.cross(f, up))  # Right
    u = np.cross(r, f)              # True up (may differ from input 'up')

    # Build view matrix: rotation part transposes the camera basis,
    # translation part dots with -eye to account for camera position
    view = np.array([
        [r[0],  r[1],  r[2],  -np.dot(r, eye)],
        [u[0],  u[1],  u[2],  -np.dot(u, eye)],
        [-f[0], -f[1], -f[2],  np.dot(f, eye)],
        [0,     0,     0,     1]
    ], dtype=float)

    return view

# Camera at (0, 2, 5), looking at origin, world up is +Y
view_matrix = look_at(
    eye=[0, 2, 5],
    target=[0, 0, 0],
    up=[0, 1, 0]
)
print("View matrix:\n", np.round(view_matrix, 4))
```

### 4.2 Intuition: "Moving the Camera" vs "Moving the World"

Moving the camera right is equivalent to moving the entire world left. The view matrix does the latter: it transforms all world-space vertices so that the camera appears to be at the origin. This simplifies subsequent projection calculations.

---

## 5. Projection Matrices

Projection transforms 3D eye-space coordinates to 2D. There are two main types.

### 5.1 Orthographic Projection

Orthographic projection maps a rectangular box (the view volume) to the NDC cube $[-1, 1]^3$. There is **no perspective foreshortening** -- distant objects appear the same size as near objects.

Given a view volume defined by left $l$, right $r$, bottom $b$, top $t$, near $n$, far $f$:

$$\mathbf{P}_{\text{ortho}} = \begin{bmatrix} \frac{2}{r-l} & 0 & 0 & -\frac{r+l}{r-l} \\ 0 & \frac{2}{t-b} & 0 & -\frac{t+b}{t-b} \\ 0 & 0 & \frac{-2}{f-n} & -\frac{f+n}{f-n} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

This is simply a translation (centering the box at origin) followed by a scale (making it unit-sized).

```python
def orthographic(left, right, bottom, top, near, far):
    """
    Create an orthographic projection matrix (OpenGL convention).

    Why no perspective? In orthographic projection, parallel lines
    remain parallel. This is useful for CAD, 2D games, isometric views,
    and shadow maps (directional lights use ortho projection).
    """
    return np.array([
        [2/(right-left), 0, 0, -(right+left)/(right-left)],
        [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
        [0, 0, -2/(far-near),  -(far+near)/(far-near)],
        [0, 0, 0, 1]
    ], dtype=float)
```

### 5.2 Perspective Projection

Perspective projection simulates how the human eye and cameras see: **distant objects appear smaller**. It maps a truncated pyramid (frustum) to the NDC cube.

```
       Near Plane          Far Plane
       ┌───────┐          ┌─────────────┐
       │       │         ╱│             │╲
       │  eye  │────────╱ │   Frustum   │ ╲
       │ (0,0) │────────╲ │   Volume    │ ╱
       │       │         ╲│             │╱
       └───────┘          └─────────────┘
       z = -n              z = -f
```

Given field-of-view angle $\text{fov}$ (vertical), aspect ratio $a = \frac{w}{h}$, near plane $n$, far plane $f$:

$$t = n \cdot \tan\left(\frac{\text{fov}}{2}\right), \quad r = t \cdot a$$

The perspective projection matrix (OpenGL convention, mapping to $z \in [-1, 1]$):

$$\mathbf{P}_{\text{persp}} = \begin{bmatrix} \frac{n}{r} & 0 & 0 & 0 \\ 0 & \frac{n}{t} & 0 & 0 \\ 0 & 0 & \frac{-(f+n)}{f-n} & \frac{-2fn}{f-n} \\ 0 & 0 & -1 & 0 \end{bmatrix}$$

### 5.3 Derivation of the Perspective Matrix

The key insight is that perspective projection divides $x$ and $y$ by $-z$ (objects farther away have larger $|z|$, so they shrink). Let us derive the matrix step by step.

**Step 1**: A point at $(x, y, z)$ in eye space should project to screen coordinates:

$$x_{\text{proj}} = \frac{n \cdot x}{-z}, \quad y_{\text{proj}} = \frac{n \cdot y}{-z}$$

(We use $-z$ because the camera looks along $-z$; the near plane is at $z = -n$.)

**Step 2**: We want this as a matrix multiplication + perspective division. The trick is to use the $w$ component of homogeneous coordinates. Set $w' = -z$, then:

$$\begin{bmatrix} x' \\ y' \\ z' \\ w' \end{bmatrix} = \mathbf{P} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

After perspective division: $\left(\frac{x'}{w'}, \frac{y'}{w'}, \frac{z'}{w'}\right)$.

**Step 3**: We need $\frac{x'}{w'} = \frac{n \cdot x}{-z}$. If $w' = -z$, then $x' = n \cdot x$. This gives us the first row: $[n, 0, 0, 0]$.

After normalizing by the frustum extent (dividing by $r$ and $t$), and handling the $z$-mapping to $[-1, 1]$, we arrive at the full matrix.

**Step 4**: The $z$-mapping must satisfy:
- $z = -n$ maps to $z_{\text{NDC}} = -1$
- $z = -f$ maps to $z_{\text{NDC}} = +1$

Solving the linear system in $1/z$ gives the third row: $[0, 0, -(f+n)/(f-n), -2fn/(f-n)]$.

```python
def perspective(fov_deg, aspect, near, far):
    """
    Create a perspective projection matrix (OpenGL convention).

    Why fov and aspect instead of l/r/b/t? This is the more intuitive
    parameterization: fov controls "zoom level" and aspect matches
    the screen's width/height ratio.

    Parameters:
        fov_deg: vertical field of view in degrees
        aspect: width / height ratio
        near: distance to near clipping plane (positive)
        far: distance to far clipping plane (positive)
    """
    fov = np.radians(fov_deg)
    t = near * np.tan(fov / 2)     # Half-height of near plane
    r = t * aspect                  # Half-width of near plane

    return np.array([
        [near/r, 0,      0,                     0],
        [0,      near/t, 0,                     0],
        [0,      0,      -(far+near)/(far-near), -2*far*near/(far-near)],
        [0,      0,      -1,                     0]
    ], dtype=float)
```

### 5.4 Depth Precision

The perspective matrix maps $z$ non-linearly: there is much more precision near the near plane than near the far plane. The NDC depth is:

$$z_{\text{NDC}} = \frac{-(f+n)}{f-n} + \frac{2fn}{(f-n)(-z)}$$

This is a hyperbolic function of $-z$. Consequences:
- A near/far ratio of 1:1000 wastes most depth precision in the first 10% of the depth range
- **Practical advice**: Keep the near plane as far as possible and the far plane as close as possible
- **Reversed-Z**: A modern technique that flips the depth mapping to use floating-point precision more evenly

---

## 6. The MVP Pipeline

The complete transformation chain from object space to clip space:

$$\mathbf{p}_{\text{clip}} = \mathbf{P} \cdot \mathbf{V} \cdot \mathbf{M} \cdot \mathbf{p}_{\text{object}}$$

Where:
- $\mathbf{M}$ = Model matrix (object &rarr; world)
- $\mathbf{V}$ = View matrix (world &rarr; eye/camera)
- $\mathbf{P}$ = Projection matrix (eye &rarr; clip)

After clipping, **perspective division** converts clip to NDC:

$$\mathbf{p}_{\text{NDC}} = \left(\frac{x_c}{w_c}, \frac{y_c}{w_c}, \frac{z_c}{w_c}\right)$$

```
Object      Model       World      View       Eye       Projection    Clip
Space  ────────────▶  Space  ────────────▶  Space  ──────────────▶  Space
                                                                       │
                                                              Perspective
                                                              Division
                                                                       │
                                                                       ▼
Screen     Viewport      NDC
Space  ◀────────────   Space
(pixels)              [-1,1]^3
```

---

## 7. Viewport Transform

The **viewport transform** maps NDC coordinates $[-1, 1]^2$ to screen pixel coordinates:

$$x_{\text{screen}} = \frac{w}{2} \cdot x_{\text{NDC}} + \frac{w}{2} + x_0$$

$$y_{\text{screen}} = \frac{h}{2} \cdot y_{\text{NDC}} + \frac{h}{2} + y_0$$

Where $(x_0, y_0)$ is the viewport offset (usually $(0, 0)$) and $(w, h)$ is the viewport size in pixels.

As a matrix:

$$\mathbf{M}_{\text{viewport}} = \begin{bmatrix} \frac{w}{2} & 0 & 0 & x_0 + \frac{w}{2} \\ 0 & \frac{h}{2} & 0 & y_0 + \frac{h}{2} \\ 0 & 0 & \frac{1}{2} & \frac{1}{2} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

```python
def viewport(x0, y0, width, height):
    """
    Create the viewport transformation matrix.

    Maps NDC [-1,1]^2 to screen coordinates [x0, x0+width] x [y0, y0+height].
    Depth is mapped from [-1,1] to [0,1] for the depth buffer.
    """
    return np.array([
        [width/2,  0,        0,   x0 + width/2],
        [0,        height/2, 0,   y0 + height/2],
        [0,        0,        0.5, 0.5],
        [0,        0,        0,   1]
    ], dtype=float)
```

---

## 8. Euler Angles and Gimbal Lock

### 8.1 Euler Angles

**Euler angles** represent a 3D orientation as three sequential rotations:
- **Yaw** ($\psi$): rotation about y-axis (look left/right)
- **Pitch** ($\theta$): rotation about x-axis (look up/down)
- **Roll** ($\phi$): rotation about z-axis (tilt head)

$$\mathbf{R} = \mathbf{R}_y(\psi) \cdot \mathbf{R}_x(\theta) \cdot \mathbf{R}_z(\phi)$$

(The order of rotations varies by convention -- this is one common choice.)

### 8.2 Gimbal Lock

**Gimbal lock** occurs when two rotation axes align, causing a loss of one degree of freedom. For the Yaw-Pitch-Roll convention, when pitch = $\pm 90°$, the yaw and roll axes become parallel -- changes to either produce the same rotation.

**Example**: In an airplane, if you pitch up 90 degrees (nose pointing straight up), yaw and roll both rotate about the same axis. You have lost the ability to independently control one rotation axis.

```
Normal state:              Gimbal lock (pitch = 90°):
  Yaw  ↻  (Y axis)           Yaw  ↻  (Y axis)
  Pitch ↻ (X axis)           Pitch ↻ (Z axis aligned!)
  Roll  ↻ (Z axis)           Roll  ↻ (Z axis)
  [3 independent axes]       [Yaw and Roll = same axis!]
```

### 8.3 Quaternions: The Solution

**Quaternions** are 4D numbers that elegantly represent 3D rotations without gimbal lock:

$$q = w + xi + yj + zk$$

Or equivalently: $q = (w, \mathbf{v})$ where $\mathbf{v} = (x, y, z)$.

A rotation of angle $\theta$ about unit axis $\hat{\mathbf{u}}$ is represented as:

$$q = \left(\cos\frac{\theta}{2}, \sin\frac{\theta}{2} \cdot \hat{\mathbf{u}}\right)$$

**Key properties**:
- **Unit quaternions** ($|q| = 1$) represent rotations
- **Composition**: $q_{\text{combined}} = q_2 \cdot q_1$ (multiply quaternions)
- **Interpolation**: SLERP (Spherical Linear Interpolation) smoothly blends between orientations
- **No gimbal lock**: Quaternions have no singularities
- **Compact**: 4 numbers instead of 9 (3x3 matrix)

```python
def quaternion_from_axis_angle(axis, angle_deg):
    """
    Create a unit quaternion from an axis-angle rotation.

    Why quaternions over Euler angles?
    1. No gimbal lock -- all orientations are reachable
    2. Smooth interpolation (SLERP)
    3. Compact (4 floats vs 9 for matrix)
    4. Numerically stable composition
    """
    axis = normalize(np.asarray(axis, dtype=float))
    half_angle = np.radians(angle_deg) / 2
    w = np.cos(half_angle)
    xyz = np.sin(half_angle) * axis
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quaternion_to_matrix(q):
    """
    Convert a unit quaternion to a 4x4 rotation matrix.

    This avoids trigonometric functions entirely -- the conversion
    uses only multiplications and additions, which is more efficient.
    """
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y),   0],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x),   0],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y), 0],
        [0,             0,             0,              1]
    ], dtype=float)


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions (compose rotations).

    q1 * q2 applies q2 first, then q1 (same as matrix convention).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def slerp(q1, q2, t):
    """
    Spherical Linear Interpolation between two quaternions.

    SLERP produces constant-speed rotation along the shortest path
    between two orientations. This is essential for smooth animations.

    Parameters:
        q1, q2: unit quaternions (start and end orientations)
        t: interpolation parameter in [0, 1]
    """
    dot = np.dot(q1, q2)

    # If dot product is negative, negate one quaternion to take the shorter path
    # (q and -q represent the same rotation)
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If quaternions are very close, use linear interpolation to avoid division by zero
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    return w1 * q1 + w2 * q2
```

---

## 9. Complete MVP Implementation

```python
"""
Complete Model-View-Projection pipeline demonstration.

Transforms a 3D cube from object space through every coordinate space
to final screen pixel positions.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# Helper functions (from earlier sections)
# ═══════════════════════════════════════════════════════════════

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v

def make_translation(tx, ty, tz):
    return np.array([[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]], dtype=float)

def make_rotation_y(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=float)

def make_scale(sx, sy, sz):
    return np.array([[sx,0,0,0],[0,sy,0,0],[0,0,sz,0],[0,0,0,1]], dtype=float)

def look_at(eye, target, up):
    eye, target, up = [np.asarray(v, dtype=float) for v in [eye, target, up]]
    f = normalize(target - eye)
    r = normalize(np.cross(f, up))
    u = np.cross(r, f)
    return np.array([
        [r[0], r[1], r[2], -np.dot(r, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [-f[0],-f[1],-f[2], np.dot(f, eye)],
        [0, 0, 0, 1]
    ], dtype=float)

def perspective(fov_deg, aspect, near, far):
    fov = np.radians(fov_deg)
    t = near * np.tan(fov / 2)
    r = t * aspect
    return np.array([
        [near/r, 0, 0, 0],
        [0, near/t, 0, 0],
        [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
        [0, 0, -1, 0]
    ], dtype=float)

# ═══════════════════════════════════════════════════════════════
# Define scene
# ═══════════════════════════════════════════════════════════════

# A unit cube centered at origin (8 vertices)
cube_vertices = np.array([
    [-0.5, -0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5],
    [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5],
    [-0.5,  0.5,  0.5],
], dtype=float)

# ═══════════════════════════════════════════════════════════════
# Build MVP matrices
# ═══════════════════════════════════════════════════════════════

# Model: scale by 2, rotate 30 deg around Y, translate to (0, 1, -5)
M = make_translation(0, 1, -5) @ make_rotation_y(30) @ make_scale(2, 2, 2)

# View: camera at (0, 3, 5) looking at origin
V = look_at(eye=[0, 3, 5], target=[0, 0, 0], up=[0, 1, 0])

# Projection: 60 degree FOV, 16:9 aspect, near=0.1, far=100
P = perspective(fov_deg=60, aspect=16/9, near=0.1, far=100)

# Combined MVP
MVP = P @ V @ M

# ═══════════════════════════════════════════════════════════════
# Transform vertices through the pipeline
# ═══════════════════════════════════════════════════════════════

print("=== MVP Pipeline Demonstration ===\n")
print(f"Processing {len(cube_vertices)} vertices of a unit cube...\n")

# Screen dimensions
screen_w, screen_h = 1920, 1080

for i, v in enumerate(cube_vertices):
    # Step 1: Object space -> Clip space (via MVP)
    p_obj = np.array([v[0], v[1], v[2], 1.0])
    p_clip = MVP @ p_obj

    # Step 2: Perspective division -> NDC
    w = p_clip[3]
    p_ndc = p_clip[:3] / w

    # Step 3: Viewport transform -> Screen coordinates
    sx = (p_ndc[0] + 1) * 0.5 * screen_w
    sy = (1 - p_ndc[1]) * 0.5 * screen_h  # Flip Y (screen Y goes down)
    depth = (p_ndc[2] + 1) * 0.5  # Depth in [0, 1]

    if i < 4:  # Print first 4 vertices as examples
        print(f"Vertex {i}: object={v}")
        print(f"  clip=({p_clip[0]:.3f}, {p_clip[1]:.3f}, "
              f"{p_clip[2]:.3f}, {p_clip[3]:.3f})")
        print(f"  NDC=({p_ndc[0]:.3f}, {p_ndc[1]:.3f}, {p_ndc[2]:.3f})")
        print(f"  screen=({sx:.1f}, {sy:.1f}), depth={depth:.4f}")
        print()

print("... (remaining vertices follow the same process)")
```

---

## 10. Normal Transformation

When transforming geometry, normals require special treatment. If the model matrix includes non-uniform scaling, simply applying $\mathbf{M}$ to normals produces incorrect results.

The correct **normal matrix** is the inverse-transpose of the upper-left 3x3 of the model matrix:

$$\mathbf{N} = (\mathbf{M}_{3\times3}^{-1})^T$$

**Why?** Normals are not positions -- they are perpendicular to surfaces. Non-uniform scaling changes the surface but should preserve perpendicularity. The inverse-transpose ensures that the transformed normal remains perpendicular to the transformed surface.

**Proof sketch**: If $\mathbf{t}$ is a tangent vector (lies in the surface plane), then $\mathbf{n} \cdot \mathbf{t} = 0$. After transformation: $\mathbf{n}' \cdot \mathbf{t}' = (\mathbf{N}\mathbf{n})^T (\mathbf{M}\mathbf{t}) = \mathbf{n}^T \mathbf{N}^T \mathbf{M} \mathbf{t}$. For this to equal zero, we need $\mathbf{N}^T \mathbf{M} = \mathbf{I}$, giving $\mathbf{N} = (\mathbf{M}^{-1})^T$.

```python
def compute_normal_matrix(model_matrix):
    """
    Compute the normal transformation matrix.

    If the model matrix has only rotation and uniform scale,
    the normal matrix equals the model's upper-left 3x3 (since
    rotation is orthogonal and uniform scale cancels out).
    But for non-uniform scaling, we MUST use the inverse-transpose.
    """
    linear = model_matrix[:3, :3]
    return np.linalg.inv(linear).T
```

---

## Summary

| Coordinate Space | Range | Comes After |
|-----------------|-------|-------------|
| Object (local) | Arbitrary | Model definition |
| World | Arbitrary | Model matrix $\mathbf{M}$ |
| Eye (camera) | Camera at origin, looks along $-z$ | View matrix $\mathbf{V}$ |
| Clip | $-w \leq x,y,z \leq w$ | Projection matrix $\mathbf{P}$ |
| NDC | $[-1, 1]^3$ | Perspective division ($\div w$) |
| Screen | $[0, W] \times [0, H]$ pixels | Viewport transform |

**Key takeaways**:
- The MVP chain $\mathbf{P} \cdot \mathbf{V} \cdot \mathbf{M}$ is the central transformation in all 3D rendering
- The View matrix is constructed from camera position + look direction using the **lookAt** formula
- Perspective projection creates foreshortening by dividing by $w = -z$
- Depth precision is non-linear and concentrated near the near plane
- Euler angles suffer from **gimbal lock**; quaternions provide a robust alternative
- Normals must be transformed by the **inverse-transpose** of the model matrix

---

## Exercises

1. **MVP Construction**: Given a camera at $(3, 3, 3)$ looking at the origin with up vector $(0, 1, 0)$, construct the full MVP matrix for a unit cube at position $(1, 0, -2)$. Transform all 8 vertices to screen coordinates (1920x1080 viewport).

2. **Orthographic vs Perspective**: Render the same scene using both orthographic and perspective projections. Describe the visual differences. When would you prefer orthographic?

3. **Depth Buffer Values**: For a perspective projection with $n = 0.1$, $f = 100$: compute the NDC depth for objects at $z = -1$, $z = -10$, $z = -50$, and $z = -100$. Plot the function $z_{\text{NDC}}(z)$ and discuss why this non-linear distribution causes precision problems.

4. **Gimbal Lock Demonstration**: Implement an Euler angle rotation system. Show that when pitch = 90 degrees, yaw and roll produce the same rotation. Then show the same rotations using quaternions without the lock.

5. **SLERP Animation**: Using quaternions, implement a smooth rotation from "looking forward" to "looking 180 degrees right and 45 degrees up." Sample 10 intermediate orientations using SLERP and convert each to a rotation matrix.

6. **Normal Transformation**: Create a model matrix with non-uniform scaling $(2, 1, 0.5)$ and rotation. Show that applying the model matrix directly to a normal vector produces an incorrect result, while the inverse-transpose produces the correct perpendicular normal.

---

## Further Reading

1. Marschner, S. & Shirley, P. *Fundamentals of Computer Graphics* (5th ed.), Ch. 7 -- "Viewing"
2. Akenine-Moller, T. et al. *Real-Time Rendering* (4th ed.), Ch. 4 -- "Transforms"
3. [Learn OpenGL -- Coordinate Systems](https://learnopengl.com/Getting-started/Coordinate-Systems) -- Interactive explanation of the MVP pipeline
4. [Quaternions and Spatial Rotation](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation) -- Mathematical details
5. [The Depth Buffer Explained (blog)](https://developer.nvidia.com/content/depth-precision-visualized) -- NVIDIA's depth precision visualization
