# 02. 2D Transformations

[&larr; Previous: Graphics Pipeline Overview](01_Graphics_Pipeline_Overview.md) | [Next: 3D Transformations and Projections &rarr;](03_3D_Transformations_and_Projections.md)

---

## Learning Objectives

1. Represent 2D translation, rotation, and scaling as matrices
2. Understand why homogeneous coordinates require 3x3 matrices for 2D transformations
3. Compose multiple transformations by matrix multiplication and recognize that order matters
4. Apply reflection and shear transformations
5. Distinguish between affine and projective transformations
6. Implement all 2D transformations from scratch using NumPy
7. Visualize the geometric effect of each transformation on shapes
8. Build intuition for transformation composition through the "instruction card" analogy

---

## Why This Matters

Transformations are the mathematical language of motion and change in computer graphics. Every time you drag an icon on your desktop, rotate an image in a photo editor, or watch a 2D game character run across the screen, 2D transformations are at work. Mastering 2D transformations first -- before tackling the more complex 3D case -- builds a solid foundation because the same principles (matrix representation, composition, coordinate systems) carry over directly to 3D.

Think of each transformation as an **instruction card telling every point where to move**. A rotation card says "every point, rotate 45 degrees around the origin." A scaling card says "every point, move twice as far from the origin." When you stack instruction cards, you compose transformations -- and the order you stack them in profoundly affects the result.

---

## 1. Points and Vectors in 2D

A **point** in 2D is a position: $\mathbf{p} = (x, y)$.
A **vector** in 2D is a displacement: $\mathbf{v} = (v_x, v_y)$.

In graphics, we often treat both as column vectors and apply matrix transformations to them:

$$\mathbf{p} = \begin{bmatrix} x \\ y \end{bmatrix}$$

A **transformation** $T$ maps points to new positions:

$$T: \mathbb{R}^2 \rightarrow \mathbb{R}^2, \quad \mathbf{p}' = T(\mathbf{p})$$

If $T$ can be expressed as a matrix multiplication, we say it is a **linear transformation**:

$$\mathbf{p}' = \mathbf{M} \cdot \mathbf{p}$$

---

## 2. Basic 2D Transformations

### 2.1 Scaling

Scaling changes the size of an object. A **non-uniform** scale allows different factors along each axis:

$$\mathbf{S}(s_x, s_y) = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

Applying the scale:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} s_x \cdot x \\ s_y \cdot y \end{bmatrix}$$

- **Uniform scaling**: $s_x = s_y$ (preserves shape, changes size)
- **Non-uniform scaling**: $s_x \neq s_y$ (stretches/compresses along axes)
- $s = 1$: no change; $s > 1$: enlarge; $0 < s < 1$: shrink; $s < 0$: reflect + scale

### 2.2 Rotation

Rotation by angle $\theta$ (counterclockwise, about the origin):

$$\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**Derivation**: A point at $(r, 0)$ rotated by $\theta$ moves to $(r\cos\theta, r\sin\theta)$. A point at $(0, r)$ rotated by $\theta$ moves to $(-r\sin\theta, r\cos\theta)$. These form the columns of the rotation matrix.

Properties of rotation matrices:
- **Orthogonal**: $\mathbf{R}^T = \mathbf{R}^{-1}$ (the inverse is just the transpose)
- **Determinant** = 1 (preserves area and orientation)
- **Rotation composition**: $\mathbf{R}(\alpha) \cdot \mathbf{R}(\beta) = \mathbf{R}(\alpha + \beta)$

### 2.3 Translation

Translation shifts every point by a fixed offset:

$$\mathbf{p}' = \mathbf{p} + \mathbf{t} = \begin{bmatrix} x + t_x \\ y + t_y \end{bmatrix}$$

**The problem**: Translation is *not* a linear transformation -- it cannot be expressed as a 2x2 matrix multiplication. This is because linear transformations always map the origin to itself: $\mathbf{M} \cdot \mathbf{0} = \mathbf{0}$. But translation moves the origin.

This motivates **homogeneous coordinates**.

---

## 3. Homogeneous Coordinates

### 3.1 The Key Idea

To unify all affine transformations (including translation) into matrix multiplications, we add a third coordinate. A 2D point $(x, y)$ becomes:

$$\mathbf{p}_h = \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Now we can represent translation as a 3x3 matrix:

$$\mathbf{T}(t_x, t_y) = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$$

Verification:

$$\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} x + t_x \\ y + t_y \\ 1 \end{bmatrix}$$

### 3.2 Why 3x3 for 2D?

The fundamental insight: **a translation in $n$-dimensional space is a linear transformation in $(n+1)$-dimensional homogeneous space**. By embedding 2D points into 3D homogeneous space, we gain the ability to represent all affine transformations as matrix multiplications.

The general form of a 2D affine transformation in homogeneous coordinates:

$$\mathbf{M} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix}$$

Where:
- $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$ encodes rotation, scaling, shear
- $\begin{bmatrix} t_x \\ t_y \end{bmatrix}$ encodes translation
- The bottom row $[0, 0, 1]$ ensures the $w$-coordinate remains 1

### 3.3 Points vs Vectors in Homogeneous Coordinates

| Entity | Homogeneous Form | Why? |
|--------|-----------------|------|
| Point | $(x, y, 1)$ | Points *should* be translated |
| Vector | $(v_x, v_y, 0)$ | Vectors *should not* be translated |

Multiplying a translation matrix by a vector $(v_x, v_y, 0)^T$ leaves it unchanged (the $t_x, t_y$ terms multiply the 0 in the third component). This correctly models that vectors represent directions, which are independent of position.

### 3.4 All Transformations in Homogeneous Form

**Translation**:
$$\mathbf{T}(t_x, t_y) = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$$

**Scaling** (about the origin):
$$\mathbf{S}(s_x, s_y) = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Rotation** (about the origin):
$$\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

---

## 4. Composition of Transformations

### 4.1 Matrix Multiplication = Transformation Composition

The beauty of the matrix representation: composing transformations is simply matrix multiplication. To first rotate, then translate:

$$\mathbf{p}' = \mathbf{T} \cdot \mathbf{R} \cdot \mathbf{p}$$

> **Reading order**: Transformations apply **right to left**. In $\mathbf{T} \cdot \mathbf{R} \cdot \mathbf{p}$, the rotation $\mathbf{R}$ is applied first, then the translation $\mathbf{T}$.

### 4.2 Order Matters!

Transformation composition is **not commutative**: $\mathbf{T} \cdot \mathbf{R} \neq \mathbf{R} \cdot \mathbf{T}$ in general.

**Example**: Consider rotating 90 degrees and translating by $(5, 0)$.

**Rotate then translate** ($\mathbf{T} \cdot \mathbf{R}$):
1. Rotate the point around the origin
2. Then shift everything right by 5

**Translate then rotate** ($\mathbf{R} \cdot \mathbf{T}$):
1. Shift the point right by 5
2. Then rotate around the origin (the shifted point sweeps in a circle)

These produce very different results! The second case orbits the translated point around the origin.

```python
import numpy as np

def make_translation(tx, ty):
    """Create a 2D translation matrix in homogeneous coordinates."""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0,  1]
    ], dtype=float)

def make_rotation(theta_deg):
    """Create a 2D rotation matrix (counterclockwise) in homogeneous coordinates."""
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=float)

# Demonstrate that order matters
T = make_translation(5, 0)
R = make_rotation(90)
point = np.array([1, 0, 1])  # Point at (1, 0)

# Rotate first, then translate
result_RT = T @ R @ point
print(f"Rotate then Translate: ({result_RT[0]:.2f}, {result_RT[1]:.2f})")
# Output: Rotate then Translate: (5.00, 1.00)

# Translate first, then rotate
result_TR = R @ T @ point
print(f"Translate then Rotate: ({result_TR[0]:.2f}, {result_TR[1]:.2f})")
# Output: Translate then Rotate: (0.00, 6.00)
```

### 4.3 Rotation About an Arbitrary Point

To rotate around a point $\mathbf{c} = (c_x, c_y)$ instead of the origin:

1. Translate so that $\mathbf{c}$ is at the origin: $\mathbf{T}(-c_x, -c_y)$
2. Rotate: $\mathbf{R}(\theta)$
3. Translate back: $\mathbf{T}(c_x, c_y)$

$$\mathbf{M} = \mathbf{T}(c_x, c_y) \cdot \mathbf{R}(\theta) \cdot \mathbf{T}(-c_x, -c_y)$$

This pattern -- "translate to origin, apply transformation, translate back" -- recurs frequently in graphics.

```python
def make_rotation_about_point(theta_deg, cx, cy):
    """
    Rotate around an arbitrary center point.

    Why this pattern? Rotation matrices rotate about the origin.
    To rotate about (cx, cy), we temporarily move (cx, cy) to the origin,
    perform the rotation, and move back.
    """
    T_to_origin = make_translation(-cx, -cy)
    R = make_rotation(theta_deg)
    T_back = make_translation(cx, cy)

    # Compose: apply right to left (to_origin first, rotate, then back)
    return T_back @ R @ T_to_origin
```

### 4.4 Scaling About an Arbitrary Point

Similarly, to scale about a point $\mathbf{c}$:

$$\mathbf{M} = \mathbf{T}(c_x, c_y) \cdot \mathbf{S}(s_x, s_y) \cdot \mathbf{T}(-c_x, -c_y)$$

---

## 5. Reflection

Reflection mirrors geometry across an axis.

### 5.1 Reflection Across Axes

**Reflect across x-axis** (flip y):
$$\mathbf{M}_x = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Reflect across y-axis** (flip x):
$$\mathbf{M}_y = \begin{bmatrix} -1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Reflect across origin** (flip both):
$$\mathbf{M}_o = \begin{bmatrix} -1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

### 5.2 Reflection Across an Arbitrary Line

To reflect across a line through the origin at angle $\alpha$:

$$\mathbf{M}_\alpha = \begin{bmatrix} \cos 2\alpha & \sin 2\alpha & 0 \\ \sin 2\alpha & -\cos 2\alpha & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

For a line not through the origin: translate, reflect, translate back (same pattern as rotation about an arbitrary point).

---

## 6. Shear

Shear "slants" objects along one axis proportional to the other axis.

### 6.1 Shear Along X

$$\mathbf{Sh}_x(k) = \begin{bmatrix} 1 & k & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

This shifts each point's $x$-coordinate by $k \cdot y$:

$$x' = x + k \cdot y, \quad y' = y$$

Imagine a deck of cards: the bottom card stays still, and each higher card slides a bit to the right.

### 6.2 Shear Along Y

$$\mathbf{Sh}_y(k) = \begin{bmatrix} 1 & 0 & 0 \\ k & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

This shifts each point's $y$-coordinate by $k \cdot x$.

### 6.3 Decomposition Insight

Any 2D affine transformation can be decomposed into a sequence of:
- Rotations
- Scalings (non-uniform)
- Translations

Alternatively, shear is sometimes used as a computational building block because it is simpler to implement in hardware than rotation.

---

## 7. Affine vs Projective Transformations

### 7.1 Affine Transformations

An **affine transformation** preserves:
- **Collinearity**: Points on a line remain on a line
- **Ratios of distances**: The midpoint of a segment remains the midpoint
- **Parallelism**: Parallel lines remain parallel

All transformations we have seen so far (translation, rotation, scaling, reflection, shear) are affine. In homogeneous coordinates, affine transformations have the form:

$$\mathbf{M}_{\text{affine}} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix}$$

The bottom row is always $[0, 0, 1]$.

### 7.2 Projective Transformations

A **projective transformation** (also called a **homography** or **perspective transformation**) allows the bottom row to be non-trivial:

$$\mathbf{M}_{\text{proj}} = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & 1 \end{bmatrix}$$

After multiplication, we must divide by the $w$-component to get back to 2D:

$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \mathbf{M}_{\text{proj}} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad \text{result} = \left(\frac{x'}{w'}, \frac{y'}{w'}\right)$$

Projective transformations:
- **Do not** preserve parallelism (parallel lines can converge)
- **Do** preserve collinearity (lines remain lines)
- Are used for perspective projection in 3D graphics (Lesson 03)
- Are used in image stitching and augmented reality

### 7.3 Transformation Hierarchy

```
Rigid (Euclidean)  ⊂  Similarity  ⊂  Affine  ⊂  Projective
     │                    │              │            │
  Rotation +          Rigid +        Similarity +   Affine +
  Translation        Uniform Scale   Non-uniform    Perspective
                                     Scale + Shear
     │                    │              │            │
  Preserves:          Preserves:     Preserves:    Preserves:
  - Distances         - Angles       - Parallelism - Collinearity
  - Angles            - Ratios       - Ratios      (only)
  - Ratios
```

---

## 8. Implementation: Complete 2D Transform Library

```python
"""
Complete 2D transformation library using NumPy.

All transformations operate in homogeneous coordinates (3x3 matrices).
Points are represented as (x, y, 1) column vectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# ═══════════════════════════════════════════════════════════════
# Core Transformation Matrices
# ═══════════════════════════════════════════════════════════════

def translate(tx, ty):
    """
    Create a 2D translation matrix.

    Why homogeneous coords? Without them, translation requires addition
    (p' = p + t), making it incompatible with other transforms that use
    multiplication. Homogeneous coords unify everything as multiplication.
    """
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0,  1]
    ], dtype=float)


def rotate(theta_deg):
    """
    Create a 2D rotation matrix (counterclockwise about origin).

    The rotation matrix is orthogonal: R^T = R^{-1}.
    This means the inverse rotation is simply the transpose,
    which is computationally cheaper than a general matrix inverse.
    """
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=float)


def scale(sx, sy=None):
    """
    Create a 2D scaling matrix.

    If sy is not given, uniform scaling is applied (sx = sy).
    Negative scale values produce reflection + scaling.
    """
    if sy is None:
        sy = sx  # Uniform scaling
    return np.array([
        [sx,  0, 0],
        [ 0, sy, 0],
        [ 0,  0, 1]
    ], dtype=float)


def reflect_x():
    """Reflect across the x-axis (y coordinates are negated)."""
    return scale(1, -1)


def reflect_y():
    """Reflect across the y-axis (x coordinates are negated)."""
    return scale(-1, 1)


def reflect_line(angle_deg):
    """
    Reflect across a line through the origin at the given angle.

    Derivation: rotate so the line aligns with x-axis, reflect across
    x-axis, rotate back. This simplifies to the formula below.
    """
    a = np.radians(angle_deg)
    c2, s2 = np.cos(2 * a), np.sin(2 * a)
    return np.array([
        [ c2, s2, 0],
        [ s2, -c2, 0],
        [  0,   0, 1]
    ], dtype=float)


def shear_x(k):
    """
    Shear along x-axis: x' = x + k*y, y' = y.

    Visualize a deck of cards: each card slides horizontally
    proportional to its height.
    """
    return np.array([
        [1, k, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float)


def shear_y(k):
    """Shear along y-axis: x' = x, y' = y + k*x."""
    return np.array([
        [1, 0, 0],
        [k, 1, 0],
        [0, 0, 1]
    ], dtype=float)


# ═══════════════════════════════════════════════════════════════
# Compound Transformations
# ═══════════════════════════════════════════════════════════════

def rotate_about(theta_deg, cx, cy):
    """
    Rotate about an arbitrary center point (cx, cy).

    Pattern: translate center to origin -> rotate -> translate back.
    This "sandwich" pattern is fundamental in graphics.
    """
    return translate(cx, cy) @ rotate(theta_deg) @ translate(-cx, -cy)


def scale_about(sx, sy, cx, cy):
    """Scale about an arbitrary center point (cx, cy)."""
    return translate(cx, cy) @ scale(sx, sy) @ translate(-cx, -cy)


# ═══════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════

def transform_points(matrix, points):
    """
    Apply a 3x3 transformation matrix to an array of 2D points.

    Parameters:
        matrix: 3x3 transformation matrix
        points: Nx2 array of (x, y) coordinates

    Returns:
        Nx2 array of transformed (x', y') coordinates

    Why we add w=1: each point (x,y) becomes (x,y,1) in homogeneous
    coordinates. After transformation, we extract (x', y') and discard w'.
    For affine transforms w' is always 1; for projective transforms,
    we would need to divide by w'.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]

    # Convert to homogeneous: (x, y) -> (x, y, 1)
    ones = np.ones((n, 1))
    homogeneous = np.hstack([points, ones])  # Nx3

    # Apply transformation: each row is a point, so we transpose
    # M @ p for column vectors = (p^T @ M^T)^T for row vectors
    transformed = (matrix @ homogeneous.T).T  # Nx3

    # Convert back from homogeneous: divide by w (handles projective case)
    w = transformed[:, 2:3]
    return transformed[:, :2] / w


def compose(*transforms):
    """
    Compose multiple transformations (applied right to left).

    compose(A, B, C) produces A @ B @ C, meaning C is applied first.
    This matches the mathematical convention p' = A * B * C * p.
    """
    result = np.eye(3)
    for t in transforms:
        result = result @ t
    return result


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════

def plot_shape(ax, points, color='blue', alpha=0.3, label=None):
    """Plot a 2D polygon defined by its vertices."""
    polygon = Polygon(points, closed=True)
    p = PatchCollection([polygon], alpha=alpha, facecolors=[color],
                        edgecolors=[color], linewidths=2)
    ax.add_collection(p)
    if label:
        centroid = points.mean(axis=0)
        ax.annotate(label, centroid, fontsize=10, ha='center',
                    fontweight='bold', color=color)


def demo_transformations():
    """Demonstrate key 2D transformations visually."""
    # Define a simple house shape
    house = np.array([
        [0, 0], [2, 0], [2, 2], [1, 3], [0, 2]  # Square base + triangle roof
    ])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('2D Transformations', fontsize=16, fontweight='bold')

    transforms = [
        ("Translation (3, 1)", translate(3, 1)),
        ("Rotation (45 deg)", rotate(45)),
        ("Scale (1.5, 0.8)", scale(1.5, 0.8)),
        ("Reflect (y-axis)", reflect_y()),
        ("Shear X (k=0.5)", shear_x(0.5)),
        ("Rotate about (1,1) 45 deg", rotate_about(45, 1, 1)),
    ]

    for ax, (title, matrix) in zip(axes.flat, transforms):
        ax.set_xlim(-4, 6)
        ax.set_ylim(-4, 6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(title)

        # Draw original shape
        plot_shape(ax, house, color='blue', alpha=0.2, label='Original')

        # Draw transformed shape
        transformed = transform_points(matrix, house)
        plot_shape(ax, transformed, color='red', alpha=0.4, label='Transformed')

    plt.tight_layout()
    plt.savefig('2d_transforms_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════
# Demonstration
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # --- Basic transforms ---
    print("=== 2D Transformation Examples ===\n")

    point = np.array([[3, 2]])  # A single point at (3, 2)

    # Translation
    T = translate(5, -1)
    print(f"Original: {point[0]}")
    print(f"After translate(5, -1): {transform_points(T, point)[0]}")
    # Expected: (8, 1)

    # Rotation by 90 degrees
    R = rotate(90)
    print(f"After rotate(90): {transform_points(R, point)[0]}")
    # Expected: (-2, 3) -- 90 deg CCW

    # Composition: rotate then translate
    M = compose(translate(5, -1), rotate(90))
    print(f"After rotate(90) then translate(5,-1): {transform_points(M, point)[0]}")
    # Expected: (3, 2) -- rotate to (-2,3), then translate to (3, 2)

    # --- Order matters ---
    print("\n=== Order Matters ===")
    p = np.array([[1, 0]])
    M1 = compose(translate(5, 0), rotate(90))  # Rotate first, then translate
    M2 = compose(rotate(90), translate(5, 0))  # Translate first, then rotate

    r1 = transform_points(M1, p)[0]
    r2 = transform_points(M2, p)[0]
    print(f"Rotate then Translate: ({r1[0]:.2f}, {r1[1]:.2f})")
    print(f"Translate then Rotate: ({r2[0]:.2f}, {r2[1]:.2f})")
    print(f"Same result? {np.allclose(r1, r2)}")  # False!

    # --- Transform a triangle ---
    print("\n=== Transforming a Triangle ===")
    triangle = np.array([[0, 0], [1, 0], [0.5, 1]])
    M = compose(
        translate(2, 3),       # 3. Move to final position
        rotate(45),            # 2. Rotate 45 degrees
        scale(2, 2)            # 1. Double the size
    )
    transformed = transform_points(M, triangle)
    print(f"Original triangle:\n{triangle}")
    print(f"After scale(2) -> rotate(45) -> translate(2,3):\n"
          f"{np.round(transformed, 3)}")

    # --- Visualization ---
    print("\n=== Generating visualization... ===")
    demo_transformations()
```

---

## 9. Matrix Properties and Inverses

Understanding inverse transformations is crucial for "undoing" operations.

### 9.1 Inverse of Common Transforms

| Transformation | Forward Matrix | Inverse |
|---------------|---------------|---------|
| Translation $\mathbf{T}(t_x, t_y)$ | $\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$ | $\mathbf{T}(-t_x, -t_y)$ |
| Rotation $\mathbf{R}(\theta)$ | Rotation by $\theta$ | $\mathbf{R}(-\theta) = \mathbf{R}(\theta)^T$ |
| Scale $\mathbf{S}(s_x, s_y)$ | Scale by $(s_x, s_y)$ | $\mathbf{S}(1/s_x, 1/s_y)$ |
| Composition $\mathbf{A} \cdot \mathbf{B}$ | Apply B then A | $\mathbf{B}^{-1} \cdot \mathbf{A}^{-1}$ |

### 9.2 Inverse of Composition

The inverse of a composed transformation reverses the order:

$$(\mathbf{A} \cdot \mathbf{B} \cdot \mathbf{C})^{-1} = \mathbf{C}^{-1} \cdot \mathbf{B}^{-1} \cdot \mathbf{A}^{-1}$$

Think of it like putting on and taking off clothes: you put on socks, then shoes (compose). To undo, you take off shoes first, then socks (reverse order).

### 9.3 Determinant and Area Change

The **determinant** of a 2D transformation matrix (the upper-left 2x2 block) tells us how areas change:

$$\text{Area ratio} = |\det(\mathbf{M}_{2 \times 2})|$$

- $|\det| = 1$: area preserved (rotations, reflections)
- $|\det| > 1$: area increases (enlargement)
- $|\det| < 1$: area decreases (shrinking)
- $\det < 0$: orientation is reversed (reflection)

```python
def analyze_transform(matrix):
    """Analyze properties of a 2D transformation matrix."""
    # Extract the linear part (upper-left 2x2)
    linear = matrix[:2, :2]
    det = np.linalg.det(linear)

    # Extract translation
    tx, ty = matrix[0, 2], matrix[1, 2]

    print(f"Matrix:\n{matrix}")
    print(f"Translation: ({tx:.3f}, {ty:.3f})")
    print(f"Determinant: {det:.3f}")
    print(f"Area scale factor: {abs(det):.3f}")
    print(f"Preserves orientation: {det > 0}")
    print(f"Is orthogonal: {np.allclose(linear @ linear.T, np.eye(2))}")
    print()

# Example analyses
print("Pure rotation (45 deg):")
analyze_transform(rotate(45))

print("Non-uniform scale (2, 0.5):")
analyze_transform(scale(2, 0.5))

print("Reflection across y-axis:")
analyze_transform(reflect_y())
```

---

## 10. Common Pitfalls

### 10.1 Rotation Direction Convention

Different systems use different conventions:
- **Math convention**: counterclockwise is positive (what we use)
- **Screen convention**: y-axis often points down, so "counterclockwise" in math appears clockwise on screen

Always verify your rotation direction matches your coordinate system!

### 10.2 Transformation About the Origin

Scaling and rotation matrices operate **about the origin** by default. If your object is not centered at the origin, the result may be unexpected:

```python
# A square centered at (5, 5), not the origin
square = np.array([[4, 4], [6, 4], [6, 6], [4, 6]])

# Scaling by 2 about the origin moves the square away!
S_origin = scale(2)
wrong = transform_points(S_origin, square)
print(f"Scale about origin: {wrong}")  # Points at (8,8) to (12,12)

# Correct: scale about the square's center (5, 5)
S_center = scale_about(2, 2, 5, 5)
correct = transform_points(S_center, square)
print(f"Scale about center: {correct}")  # Points at (3,3) to (7,7)
```

### 10.3 Floating-Point Accumulation

Repeatedly composing rotation matrices can accumulate floating-point errors, causing the matrix to "drift" away from being a proper rotation (orthogonal) matrix. For long-running animations, periodically re-orthogonalize:

```python
def orthogonalize(matrix):
    """
    Re-orthogonalize a rotation matrix that has accumulated floating-point errors.

    Uses the Gram-Schmidt process on the linear part.
    Without this, after thousands of small rotations, the matrix
    may introduce slight scaling or skew artifacts.
    """
    linear = matrix[:2, :2].copy()

    # Normalize first column
    col0 = linear[:, 0]
    col0 = col0 / np.linalg.norm(col0)

    # Make second column orthogonal to first, then normalize
    col1 = linear[:, 1]
    col1 = col1 - np.dot(col1, col0) * col0
    col1 = col1 / np.linalg.norm(col1)

    result = matrix.copy()
    result[:2, 0] = col0
    result[:2, 1] = col1
    return result
```

---

## Summary

| Transformation | Matrix (Homogeneous) | Preserves |
|---------------|---------------------|-----------|
| Translation $(t_x, t_y)$ | $\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$ | Shape, size, orientation |
| Rotation $\theta$ | $\begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$ | Shape, size |
| Scale $(s_x, s_y)$ | $\begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}$ | Shape (if uniform) |
| Shear $k$ (x-axis) | $\begin{bmatrix} 1 & k & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ | Area |
| Reflection (y-axis) | $\begin{bmatrix} -1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ | Shape, size |

**Key takeaways**:
- Homogeneous coordinates (3x3 matrices for 2D) unify all affine transformations into matrix multiplication
- Transformation composition is matrix multiplication, applied **right to left**
- **Order matters**: $\mathbf{T} \cdot \mathbf{R} \neq \mathbf{R} \cdot \mathbf{T}$ in general
- To transform about a non-origin point: translate to origin, transform, translate back
- Affine transforms preserve parallelism; projective transforms do not

---

## Exercises

1. **Matrix Construction**: Write the single 3x3 matrix that: (a) scales by factor 2, (b) rotates 30 degrees counterclockwise, (c) translates by $(3, -1)$, applied in that order. Verify by applying it to the point $(1, 0)$.

2. **Order Investigation**: For a square with corners at $(0,0), (1,0), (1,1), (0,1)$, compute the result of: (a) rotate 45 degrees then scale by 2, and (b) scale by 2 then rotate 45 degrees. Plot both results and explain the visual difference.

3. **Arbitrary Rotation**: Derive the composite matrix for rotating 60 degrees about the point $(3, 4)$. Verify that the point $(3, 4)$ maps to itself under this transformation.

4. **Inverse Transform**: An object has been transformed by $\mathbf{M} = \mathbf{T}(2, 3) \cdot \mathbf{R}(45) \cdot \mathbf{S}(2, 1)$. Write the inverse matrix $\mathbf{M}^{-1}$ that would undo this transformation. Verify that $\mathbf{M}^{-1} \cdot \mathbf{M} = \mathbf{I}$.

5. **Shear Decomposition**: Show that a rotation by angle $\theta$ can be decomposed into three shears. Implement this and verify that the result matches the standard rotation matrix.

6. **Projective Transform**: A projective transformation maps the unit square $\{(0,0), (1,0), (1,1), (0,1)\}$ to the quadrilateral $\{(0,0), (2,0), (1.5, 1), (0.5, 1)\}$. Set up the system of equations to find the 3x3 projective matrix (8 unknowns since we normalize $h_{33} = 1$).

---

## Further Reading

1. Marschner, S. & Shirley, P. *Fundamentals of Computer Graphics* (5th ed.), Ch. 6 -- "Transformation Matrices"
2. Hughes, J.F. et al. *Computer Graphics: Principles and Practice* (3rd ed.), Ch. 11 -- "2D Transformations"
3. [3Blue1Brown - Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) -- Beautiful visual intuition for matrices and transformations
4. Strang, G. *Introduction to Linear Algebra* -- Chapters on linear transformations
5. [Immersive Linear Algebra](http://immersivemath.com/ila/) -- Interactive online textbook with 2D/3D visualizations
