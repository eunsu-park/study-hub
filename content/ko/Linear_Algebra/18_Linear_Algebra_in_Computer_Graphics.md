# 18강: 컴퓨터 그래픽스에서의 선형대수 (Linear Algebra in Computer Graphics)

[이전: 17강](./17_Linear_Algebra_in_Deep_Learning.md) | [개요](./00_Overview.md) | [다음: 19강](./19_Randomized_Linear_Algebra.md)

---

## 학습 목표

- 2D 및 3D 기하 변환(회전, 스케일링, 이동, 전단)을 행렬로 표현할 수 있다
- 동차 좌표와 아핀 변환을 행렬 곱셈으로 통일하는 이유를 이해할 수 있다
- 투시 및 직교 투영 행렬을 유도하고 적용할 수 있다
- 쿼터니언과 오일러 각 대비 3D 회전 표현의 장점을 설명할 수 있다
- 완전한 카메라 변환 파이프라인(모델, 뷰, 투영)을 구축할 수 있다
- NumPy를 사용하여 Python으로 변환을 구현할 수 있다

---

## 1. 2D 변환

### 1.1 행렬로서의 선형 변환

2D의 모든 선형 변환은 $2 \times 2$ 행렬 곱셈으로 표현할 수 있습니다:

| 변환 | 행렬 | 효과 |
|---|---|---|
| 스케일링 | $\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$ | 축을 독립적으로 스케일 |
| 회전 | $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ | $\theta$만큼 반시계 방향 회전 |
| 반사 (x축) | $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$ | 수직으로 뒤집기 |
| 전단 (x) | $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$ | x축을 따라 전단 |

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

### 1.2 변환의 합성

변환의 합성은 행렬 곱셈입니다. 행렬 곱셈은 교환법칙이 성립하지 않으므로 순서가 중요합니다.

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

## 2. 동차 좌표

### 2.1 이동 변환의 문제

이동 $\mathbf{x}' = \mathbf{x} + \mathbf{t}$는 **아핀** 변환이며, 선형 변환이 아닙니다. $2 \times 2$ 행렬로 $\mathbf{x}' = M\mathbf{x}$로 표현할 수 없습니다.

**동차 좌표**는 세 번째 좌표를 추가하여 이 문제를 해결합니다. 2D 점 $(x, y)$는 $(x, y, 1)$이 되고, 모든 아핀 변환이 $3 \times 3$ 행렬 곱셈이 됩니다.

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

## 3. 3D 변환

### 3.1 기본 3D 행렬

3D에서는 $4 \times 4$ 동차 행렬을 사용합니다:

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

### 3.2 오일러 각과 짐벌 락

**오일러 각**은 회전을 좌표축에 대한 세 번의 순차적 회전으로 표현합니다. 가장 일반적인 규약은 롤-피치-요 ($R = R_z(\psi) R_y(\theta) R_x(\phi)$)입니다.

핵심 문제는 **짐벌 락**입니다: 피치 각이 $\pm 90°$일 때 두 축이 정렬되어 자유도 하나를 잃게 됩니다.

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

## 4. 쿼터니언

### 4.1 쿼터니언 기초

**쿼터니언**은 4차원 수 $q = w + xi + yj + zk$이며, 여기서 $i^2 = j^2 = k^2 = ijk = -1$입니다. $q = (w, \mathbf{v})$로 쓸 수 있으며, $w$는 스칼라 부분, $\mathbf{v} = (x, y, z)$는 벡터 부분입니다.

단위 쿼터니언($\|q\| = 1$)은 축 $\hat{\mathbf{u}}$에 대한 각도 $\theta$의 회전을 나타냅니다:

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

### 4.2 SLERP (구면 선형 보간)

쿼터니언은 **SLERP**를 통해 회전 간 부드러운 보간을 가능하게 합니다:

$$\text{slerp}(q_0, q_1, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} q_0 + \frac{\sin(t\Omega)}{\sin\Omega} q_1$$

여기서 $\cos\Omega = q_0 \cdot q_1$입니다.

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

### 4.3 쿼터니언 vs 오일러 vs 회전 행렬

| 표현 방식 | 매개변수 | 짐벌 락 | 보간 | 합성 | 정규화 |
|---|---|---|---|---|---|
| 오일러 각 | 3 | 있음 | 불량 | 느림 | 필요 없음 |
| 회전 행렬 | 9 | 없음 | 불량 | 행렬 곱셈 | 재직교화 필요 |
| 쿼터니언 | 4 | 없음 | SLERP (부드러움) | Hamilton 곱 | 단위 정규화 |

---

## 5. 투영 행렬

### 5.1 직교 투영

직교 투영은 하나의 좌표를 버려 3D 좌표를 2D로 매핑합니다. 평행선과 거리 비율을 보존합니다.

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

### 5.2 투시 투영

투시 투영은 깊이의 착각을 만듭니다: 먼 물체가 더 작게 보입니다. 이는 동차 좌표를 통해 구현되는 **사영**(비선형) 변환입니다.

$$P_{\text{persp}} = \begin{bmatrix} \frac{2n}{r-l} & 0 & \frac{r+l}{r-l} & 0 \\ 0 & \frac{2n}{t-b} & \frac{t+b}{t-b} & 0 \\ 0 & 0 & -\frac{f+n}{f-n} & -\frac{2fn}{f-n} \\ 0 & 0 & -1 & 0 \end{bmatrix}$$

곱셈 후 $w$ 성분으로 나누어야 합니다(투시 나눗셈).

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

### 5.3 깊이와 원근

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

## 6. 카메라 변환 파이프라인

### 6.1 전체 파이프라인

월드 좌표에서 화면 픽셀까지의 전체 변환은 다음을 포함합니다:

1. **모델 행렬** ($M$): 오브젝트 공간에서 월드 공간으로
2. **뷰 행렬** ($V$): 월드 공간에서 카메라 공간으로
3. **투영 행렬** ($P$): 카메라 공간에서 클립 좌표로
4. **투시 나눗셈**: 클립에서 NDC(정규화 장치 좌표)로
5. **뷰포트 변환**: NDC에서 화면 픽셀로

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

### 6.2 와이어프레임 렌더링

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

## 7. 법선 벡터 변환

기하 형상을 변환할 때 표면 법선은 정점과 다르게 변환되어야 합니다. 정점이 $M$으로 변환되면 법선은 $(M^{-1})^T$ (역전치 행렬)로 변환되어야 합니다.

이는 법선이 벡터가 아닌 **쌍대벡터**(쌍대 공간에 속함)이기 때문입니다.

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

## 연습 문제

### 문제 1: 2D 변환 시퀀스

삼각형이 다음 변환을 거치는 Python 애니메이션(또는 일련의 플롯)을 만드시오:
1. (2, 1)만큼 이동
2. 무게중심을 기준으로 45도 회전
3. 무게중심을 기준으로 (1.5, 0.5) 스케일링

세 가지를 한 번에 수행하는 단일 결합 행렬을 계산하고 동일한 결과를 생성하는지 검증하시오.

### 문제 2: 쿼터니언 연산

1. 쿼터니언 곱셈, 켤레, 역을 구현하시오
2. 쿼터니언을 오일러 각 삼중항으로 변환하고 다시 변환하시오
3. 두 회전 사이를 보간하는 SLERP를 시연하고, 회전된 벡터의 궤적을 단위 구 위에 그리시오

### 문제 3: 카메라 시스템

1인칭 카메라 시스템을 구축하시오:
1. 뷰 행렬을 계산하는 `look_at`을 구현하시오
2. 카메라를 전후좌우로 이동하시오
3. 카메라를 회전하시오 (요와 피치)
4. 카메라의 시점에서 와이어프레임 장면(다른 위치에 있는 여러 큐브)을 렌더링하시오

### 문제 4: 투영 비교

깊이 $z = 1, 5, 10, 50$에 물체가 있는 장면에 대해:
1. 직교 및 투시 투영을 계산하시오
2. 투영된 위치를 나란히 그리시오
3. 투시 투영이 깊이 인상을 주는 이유를 시각적으로 설명하시오

### 문제 5: 법선 변환 증명

역전치를 사용하는 것이 법선 변환에 올바른지 (계산적으로) 증명하시오:
1. 정점과 표면 법선을 가진 삼각형을 정의하시오
2. 비균일 스케일링 변환을 적용하시오
3. $M \cdot n$이 변환된 표면에 더 이상 수직이 아님을 보이시오
4. $(M^{-1})^T \cdot n$이 변환된 표면에 수직임을 보이시오

---

[이전: 17강](./17_Linear_Algebra_in_Deep_Learning.md) | [개요](./00_Overview.md) | [다음: 19강](./19_Randomized_Linear_Algebra.md)

**License**: CC BY-NC 4.0
