#!/bin/bash
# Exercises for Lesson 16: Linear Algebra in Graphics
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Model-View-Projection ===
# Problem: Construct model, view, and projection matrices for a simple
# 3D scene and transform a set of vertices.
exercise_1() {
    echo "=== Exercise 1: Model-View-Projection ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# Model matrix: rotate 45 deg around Y, then translate by (2, 0, -5)
def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(4)
    R[0,0], R[0,2] = c, s
    R[2,0], R[2,2] = -s, c
    return R

def translation(tx, ty, tz):
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T

M_model = translation(2, 0, -5) @ rotation_y(np.pi / 4)
print(f"Model matrix:\n{np.round(M_model, 4)}")

# View matrix: camera at (0, 2, 0) looking at (0, 0, -5)
# Simple look-at (no roll)
M_view = translation(0, -2, 0)  # Move world so camera is at origin
print(f"\nView matrix:\n{np.round(M_view, 4)}")

# Projection: simple perspective (FOV = 90 deg)
n, f = 0.1, 100  # near, far
fov = np.pi / 2
aspect = 16 / 9
t = n * np.tan(fov / 2)
r = t * aspect
M_proj = np.array([
    [n/r, 0,    0,              0],
    [0,   n/t,  0,              0],
    [0,   0,   -(f+n)/(f-n),   -2*f*n/(f-n)],
    [0,   0,   -1,              0]
])
print(f"Projection matrix:\n{np.round(M_proj, 4)}")

# MVP matrix
MVP = M_proj @ M_view @ M_model
print(f"\nMVP matrix:\n{np.round(MVP, 4)}")

# Transform a vertex
v = np.array([0, 0, 0, 1])  # Origin of model
v_clip = MVP @ v
v_ndc = v_clip[:3] / v_clip[3]  # Perspective divide
print(f"\nVertex (0,0,0) in clip space: {np.round(v_clip, 4)}")
print(f"In NDC: {np.round(v_ndc, 4)}")
SOLUTION
}

# === Exercise 2: Quaternion Rotation ===
# Problem: Rotate vector [1,0,0] by 120 degrees around axis [1,1,1]
# using quaternions.
exercise_2() {
    echo "=== Exercise 2: Quaternion Rotation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

def quat_from_axis_angle(axis, theta):
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(theta/2),
                     np.sin(theta/2) * axis[0],
                     np.sin(theta/2) * axis[1],
                     np.sin(theta/2) * axis[2]])

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_rotate(q, v):
    """Rotate vector v by quaternion q: q v q*"""
    v_quat = np.array([0, v[0], v[1], v[2]])
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)
    return result[1:]

axis = np.array([1, 1, 1])
theta = 2 * np.pi / 3  # 120 degrees
v = np.array([1, 0, 0])

q = quat_from_axis_angle(axis, theta)
v_rotated = quat_rotate(q, v)

print(f"Axis: {axis / np.linalg.norm(axis)}")
print(f"Angle: {np.degrees(theta)} degrees")
print(f"Quaternion: {np.round(q, 4)}")
print(f"||q|| = {np.linalg.norm(q):.4f}")
print(f"\n[1,0,0] rotated: {np.round(v_rotated, 4)}")
print(f"Expected: [0, 1, 0] (120 deg around [1,1,1] cycles x->y->z)")
SOLUTION
}

# === Exercise 3: Homogeneous Coordinates ===
# Problem: Transform a triangle with vertices at (0,0), (1,0), (0.5, 1)
# by scaling 2x, rotating 30 degrees, then translating by (3, 2).
exercise_3() {
    echo "=== Exercise 3: Homogeneous Coordinates ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# Vertices (2D homogeneous)
triangle = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [0.5, 1, 1]
]).T  # (3, 3) columns are vertices

# Scale 2x
S = np.array([[2, 0, 0],
              [0, 2, 0],
              [0, 0, 1]])

# Rotate 30 degrees
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta),  np.cos(theta), 0],
              [0, 0, 1]])

# Translate by (3, 2)
T = np.array([[1, 0, 3],
              [0, 1, 2],
              [0, 0, 1]])

# Combined: T @ R @ S (scale first, then rotate, then translate)
M = T @ R @ S
print(f"Scale:\n{S}")
print(f"\nRotate (30 deg):\n{np.round(R, 4)}")
print(f"\nTranslate:\n{T}")
print(f"\nCombined M = T @ R @ S:\n{np.round(M, 4)}")

# Transform
transformed = M @ triangle
print(f"\nOriginal vertices:\n{triangle[:2].T}")
print(f"Transformed vertices:\n{np.round(transformed[:2].T, 4)}")
SOLUTION
}

# === Exercise 4: Ray-Plane Intersection ===
# Problem: Find the intersection of ray r(t) = o + t*d with the plane
# defined by normal n = [0,1,0] and point p0 = [0,2,0].
exercise_4() {
    echo "=== Exercise 4: Ray-Plane Intersection ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# Ray: origin and direction
o = np.array([1, 0, 1], dtype=float)  # Ray origin
d = np.array([0, 1, 0], dtype=float)  # Ray direction (toward +y)

# Plane: n . (p - p0) = 0
n = np.array([0, 1, 0], dtype=float)  # Normal
p0 = np.array([0, 2, 0], dtype=float) # Point on plane

# Intersection: t = n . (p0 - o) / (n . d)
denom = np.dot(n, d)
print(f"Ray: r(t) = {o} + t * {d}")
print(f"Plane: {n} . (p - {p0}) = 0")

if abs(denom) < 1e-10:
    print("Ray is parallel to plane (no intersection)")
else:
    t = np.dot(n, p0 - o) / denom
    intersection = o + t * d
    print(f"\nt = {t}")
    print(f"Intersection point: {intersection}")

    # Verify point is on plane
    print(f"On plane: n . (p - p0) = {np.dot(n, intersection - p0):.10f}")

    # Verify point is on ray
    print(f"On ray (t >= 0): {t >= 0}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 16: Linear Algebra in Graphics"
echo "=============================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
