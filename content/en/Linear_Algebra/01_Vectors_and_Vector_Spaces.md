# Lesson 1: Vectors and Vector Spaces

## Learning Objectives

- Define vectors geometrically and algebraically, and perform addition, scalar multiplication, and dot product in Python
- Determine whether a set of vectors is linearly independent using rank tests
- Explain the concepts of basis, span, and dimension, and express vectors in alternate bases
- Identify and verify subspaces of $\mathbb{R}^n$
- Compute cross products in $\mathbb{R}^3$ and interpret the result geometrically

---

## 1. What Is a Vector?

A vector is a quantity that has both **magnitude** and **direction**. It can be viewed from two complementary perspectives:

1. **Geometric perspective** -- an arrow in space pointing from one location to another.
2. **Algebraic perspective** -- an ordered list (tuple) of numbers.

An $n$-dimensional real vector $\mathbf{v}$ is written as a column:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

In NumPy, we represent vectors as one-dimensional arrays:

```python
import numpy as np
import matplotlib.pyplot as plt

# Create 2D and 3D vectors
v2 = np.array([3, 2])
v3 = np.array([1, -4, 7])

print(f"2D vector: {v2}, shape: {v2.shape}")
print(f"3D vector: {v3}, shape: {v3.shape}")
```

### Visualizing 2D Vectors

```python
fig, ax = plt.subplots(figsize=(6, 6))

v = np.array([3, 2])
w = np.array([-1, 4])

ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
          color='tab:red', width=0.008, label=r'$\mathbf{v} = (3,2)$')
ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1,
          color='tab:blue', width=0.008, label=r'$\mathbf{w} = (-1,4)$')

ax.set_xlim(-3, 5)
ax.set_ylim(-1, 6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_title('Two vectors in the plane')
plt.show()
```

---

## 2. Vector Operations

### 2.1 Vector Addition

Two vectors of the same dimension are added component-wise. Geometrically, this follows the **parallelogram law** -- place the tail of the second vector at the head of the first.

$$\mathbf{v} + \mathbf{w} = \begin{bmatrix} v_1 + w_1 \\ v_2 + w_2 \\ \vdots \\ v_n + w_n \end{bmatrix}$$

```python
v = np.array([3, 2])
w = np.array([1, 4])
s = v + w
print(f"v + w = {s}")  # [4, 6]

# Visualize addition
fig, ax = plt.subplots(figsize=(7, 7))
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
          color='tab:red', width=0.007, label='v')
ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1,
          color='tab:blue', width=0.007, label='w')
ax.quiver(0, 0, s[0], s[1], angles='xy', scale_units='xy', scale=1,
          color='purple', width=0.007, label='v + w')
# Parallelogram sides
ax.quiver(v[0], v[1], w[0], w[1], angles='xy', scale_units='xy', scale=1,
          color='gray', width=0.004, alpha=0.5, linestyle='--')
ax.quiver(w[0], w[1], v[0], v[1], angles='xy', scale_units='xy', scale=1,
          color='gray', width=0.004, alpha=0.5, linestyle='--')
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Vector Addition (Parallelogram Law)')
plt.show()
```

### 2.2 Scalar Multiplication

Multiplying a vector by a scalar $c \in \mathbb{R}$ scales its magnitude and, if $c < 0$, reverses its direction.

$$c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$$

```python
v = np.array([2, 1])
scales = [0.5, 1, 2, -1]

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:orange']
for c, col in zip(scales, colors):
    cv = c * v
    ax.quiver(0, 0, cv[0], cv[1], angles='xy', scale_units='xy', scale=1,
              color=col, width=0.006, label=f'{c}v = {cv}')
ax.set_xlim(-3, 5)
ax.set_ylim(-2, 3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Scalar Multiplication')
plt.show()
```

### 2.3 Dot Product (Inner Product)

The dot product of two vectors yields a scalar:

$$\mathbf{v} \cdot \mathbf{w} = \sum_{i=1}^{n} v_i w_i = v_1 w_1 + v_2 w_2 + \cdots + v_n w_n$$

**Geometric interpretation**:

$$\mathbf{v} \cdot \mathbf{w} = \|\mathbf{v}\| \, \|\mathbf{w}\| \cos\theta$$

where $\theta$ is the angle between the two vectors.

| Dot product value | Angle relationship |
|---|---|
| $\mathbf{v} \cdot \mathbf{w} > 0$ | Acute angle (less than 90 degrees) |
| $\mathbf{v} \cdot \mathbf{w} = 0$ | Right angle (orthogonal) |
| $\mathbf{v} \cdot \mathbf{w} < 0$ | Obtuse angle (greater than 90 degrees) |

```python
v = np.array([3, 1])
w = np.array([1, 3])

# Compute dot product
dot = np.dot(v, w)
print(f"v . w = {dot}")  # 6

# Compute angle between vectors
norm_v = np.linalg.norm(v)
norm_w = np.linalg.norm(w)
cos_theta = dot / (norm_v * norm_w)
theta_rad = np.arccos(np.clip(cos_theta, -1, 1))
theta_deg = np.degrees(theta_rad)
print(f"||v|| = {norm_v:.4f}")
print(f"||w|| = {norm_w:.4f}")
print(f"Angle = {theta_deg:.2f} degrees")

# Orthogonal check
u = np.array([1, 0])
u_perp = np.array([0, 1])
print(f"u . u_perp = {np.dot(u, u_perp)}")  # 0
```

### 2.4 Cross Product (3D Only)

The cross product is defined only in $\mathbb{R}^3$ and returns a vector **perpendicular** to both inputs:

$$\mathbf{v} \times \mathbf{w} = \begin{bmatrix} v_2 w_3 - v_3 w_2 \\ v_3 w_1 - v_1 w_3 \\ v_1 w_2 - v_2 w_1 \end{bmatrix}$$

Its magnitude equals the area of the parallelogram spanned by $\mathbf{v}$ and $\mathbf{w}$:

$$\|\mathbf{v} \times \mathbf{w}\| = \|\mathbf{v}\| \, \|\mathbf{w}\| \sin\theta$$

```python
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])

cross = np.cross(v, w)
print(f"v x w = {cross}")

# Verify perpendicularity
print(f"(v x w) . v = {np.dot(cross, v)}")  # 0
print(f"(v x w) . w = {np.dot(cross, w)}")  # 0

# Parallelogram area
area = np.linalg.norm(cross)
print(f"Parallelogram area = {area:.4f}")
```

---

## 3. Vector Spaces

### 3.1 Definition

A **vector space** $V$ over $\mathbb{R}$ is a set equipped with two operations -- vector addition and scalar multiplication -- that satisfy the following eight axioms for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and $a, b \in \mathbb{R}$:

1. Commutativity: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
2. Associativity of addition: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
3. Additive identity: there exists $\mathbf{0}$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$
4. Additive inverse: for each $\mathbf{v}$ there exists $-\mathbf{v}$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$
5. Multiplicative identity: $1 \cdot \mathbf{v} = \mathbf{v}$
6. Compatibility: $a(b\mathbf{v}) = (ab)\mathbf{v}$
7. Distributivity over vector addition: $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$
8. Distributivity over scalar addition: $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$

**Common examples**:
- $\mathbb{R}^n$ -- the space of all $n$-tuples of real numbers
- $P_n$ -- all polynomials of degree at most $n$
- $C[a, b]$ -- all continuous functions on $[a, b]$

### 3.2 Subspaces

A non-empty subset $W \subseteq V$ is a **subspace** if it is itself a vector space under the same operations. The subspace test requires only three checks:

1. $\mathbf{0} \in W$
2. Closed under addition: $\mathbf{u}, \mathbf{v} \in W \implies \mathbf{u} + \mathbf{v} \in W$
3. Closed under scalar multiplication: $\mathbf{v} \in W,\, c \in \mathbb{R} \implies c\mathbf{v} \in W$

```python
# Example: the plane z = 0 in R^3 is a subspace
u = np.array([2, -1, 0])
v = np.array([3, 5, 0])

# Check closure
print(f"u + v = {u + v}")      # z-component is still 0
print(f"3 * u = {3 * u}")      # z-component is still 0
print(f"Contains zero? {np.allclose(0 * u, np.zeros(3))}")

# Counter-example: the plane z = 1 is NOT a subspace
w = np.array([1, 0, 1])
print(f"2 * w = {2 * w}")  # z-component is 2, not 1 => not closed
```

---

## 4. Linear Combinations and Span

### 4.1 Linear Combinations

A **linear combination** of vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ is any expression of the form:

$$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_k \mathbf{v}_k, \quad c_i \in \mathbb{R}$$

```python
v1 = np.array([1, 0])
v2 = np.array([0, 1])

# Some linear combinations
combo1 = 3 * v1 + 2 * v2   # [3, 2]
combo2 = -1 * v1 + 4 * v2  # [-1, 4]
print(f"3*v1 + 2*v2 = {combo1}")
print(f"-1*v1 + 4*v2 = {combo2}")
```

### 4.2 Span

The **span** of a set of vectors is the set of all possible linear combinations:

$$\mathrm{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \left\{ \sum_{i=1}^k c_i \mathbf{v}_i \;\middle|\; c_i \in \mathbb{R} \right\}$$

The span is always a subspace of the ambient vector space.

```python
from mpl_toolkits.mplot3d import Axes3D

# Span of two non-parallel vectors in R^3 forms a plane
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

# Sample many linear combinations
t = np.linspace(-2, 2, 30)
points = []
for c1 in t:
    for c2 in t:
        points.append(c1 * v1 + c2 * v2)
points = np.array(points)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.4)
ax.quiver(0, 0, 0, *v1, color='r', arrow_length_ratio=0.1, label='v1')
ax.quiver(0, 0, 0, *v2, color='b', arrow_length_ratio=0.1, label='v2')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('span(v1, v2) = the xy-plane')
ax.legend()
plt.show()
```

---

## 5. Linear Independence

### 5.1 Definition

Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are **linearly independent** if the only solution to

$$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_k \mathbf{v}_k = \mathbf{0}$$

is $c_1 = c_2 = \cdots = c_k = 0$.

If there exists a non-trivial solution (not all $c_i$ zero), the vectors are **linearly dependent** -- at least one vector can be written as a linear combination of the others.

### 5.2 Practical Test

Arrange the vectors as columns of a matrix $A$. Then:

- **Linearly independent** $\iff$ $\mathrm{rank}(A) = k$ (number of vectors).

```python
# Independent set
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
A_indep = np.column_stack([v1, v2, v3])
print(f"Rank = {np.linalg.matrix_rank(A_indep)}, #vectors = 3 => independent")

# Dependent set (v3 = v1 + v2)
w1 = np.array([1, 0, 0])
w2 = np.array([0, 1, 0])
w3 = np.array([1, 1, 0])
A_dep = np.column_stack([w1, w2, w3])
print(f"Rank = {np.linalg.matrix_rank(A_dep)}, #vectors = 3 => dependent")
```

### 5.3 Geometric Intuition

- In $\mathbb{R}^2$, two vectors are independent if and only if they do not lie on the same line through the origin.
- In $\mathbb{R}^3$, three vectors are independent if and only if they do not lie in the same plane through the origin.
- You can never have more than $n$ linearly independent vectors in $\mathbb{R}^n$.

```python
# Visual: two independent vs two dependent vectors in R^2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Independent
v = np.array([2, 1])
w = np.array([1, 3])
ax1.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
           color='r', width=0.008, label='v')
ax1.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1,
           color='b', width=0.008, label='w')
ax1.set_xlim(-1, 4); ax1.set_ylim(-1, 4)
ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)
ax1.legend(); ax1.set_title('Linearly Independent')

# Dependent (w = 2v)
v = np.array([1, 2])
w = np.array([2, 4])
ax2.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
           color='r', width=0.008, label='v')
ax2.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1,
           color='b', width=0.008, label='w = 2v')
ax2.set_xlim(-1, 5); ax2.set_ylim(-1, 5)
ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
ax2.legend(); ax2.set_title('Linearly Dependent')

plt.tight_layout()
plt.show()
```

---

## 6. Basis and Dimension

### 6.1 Basis

A **basis** for a vector space $V$ is a set of vectors $\{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ that is:

1. **Linearly independent**, and
2. **Spans** $V$.

Every vector in $V$ can be written as a **unique** linear combination of the basis vectors.

### 6.2 Standard Basis

The **standard basis** for $\mathbb{R}^n$ consists of the unit vectors:

$$\mathbf{e}_1 = \begin{bmatrix}1\\0\\\vdots\\0\end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix}0\\1\\\vdots\\0\end{bmatrix}, \quad \ldots, \quad \mathbf{e}_n = \begin{bmatrix}0\\0\\\vdots\\1\end{bmatrix}$$

### 6.3 Dimension

The **dimension** of a vector space is the number of vectors in any basis. All bases of a given vector space have the same cardinality.

- $\dim(\mathbb{R}^n) = n$
- $\dim(P_k) = k + 1$ (polynomials up to degree $k$)

### 6.4 Change of Basis

If $B = [\mathbf{b}_1 \mid \cdots \mid \mathbf{b}_n]$ is a matrix whose columns form a new basis, then the coordinates of $\mathbf{v}$ in that basis are:

$$[\mathbf{v}]_B = B^{-1} \mathbf{v}$$

```python
# Standard basis representation
v = np.array([5, 3, -2])
e1, e2, e3 = np.eye(3)
print(f"v = {v[0]}*e1 + {v[1]}*e2 + {v[2]}*e3")

# Change of basis
b1 = np.array([1, 1, 0])
b2 = np.array([0, 1, 1])
b3 = np.array([1, 0, 1])
B = np.column_stack([b1, b2, b3])

# Coordinates in the new basis
coords_B = np.linalg.solve(B, v)
print(f"Coordinates in new basis: {coords_B}")

# Verify: reconstruct from new basis
v_reconstructed = coords_B[0] * b1 + coords_B[1] * b2 + coords_B[2] * b3
print(f"Reconstructed: {v_reconstructed}")
print(f"Match: {np.allclose(v, v_reconstructed)}")
```

---

## 7. Summary of Key Concepts

| Concept | Definition | NumPy |
|---------|-----------|-------|
| Vector addition | $\mathbf{v} + \mathbf{w}$ | `v + w` |
| Scalar multiplication | $c\mathbf{v}$ | `c * v` |
| Dot product | $\mathbf{v} \cdot \mathbf{w}$ | `np.dot(v, w)` |
| Cross product | $\mathbf{v} \times \mathbf{w}$ | `np.cross(v, w)` |
| Norm | $\|\mathbf{v}\|$ | `np.linalg.norm(v)` |
| Linear independence test | $\mathrm{rank}(A) = k$ | `np.linalg.matrix_rank(A)` |
| Change of basis | $B^{-1}\mathbf{v}$ | `np.linalg.solve(B, v)` |

---

## Exercises

### Exercise 1: Vector Arithmetic

Given $\mathbf{u} = [2, -1, 3]^T$ and $\mathbf{v} = [-1, 4, 2]^T$:

1. Compute $\mathbf{u} + \mathbf{v}$, $3\mathbf{u} - 2\mathbf{v}$, and $\mathbf{u} \cdot \mathbf{v}$.
2. Find the angle between $\mathbf{u}$ and $\mathbf{v}$ in degrees.
3. Compute $\mathbf{u} \times \mathbf{v}$ and verify it is perpendicular to both inputs.

### Exercise 2: Linear Independence

Determine whether the following sets are linearly independent. Justify your answer.

(a) $\{[1, 2]^T, [3, 6]^T\}$

(b) $\{[1, 0, 1]^T, [0, 1, 1]^T, [1, 1, 0]^T\}$

(c) $\{[1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T, [1, 1, 1]^T\}$

### Exercise 3: Basis and Coordinates

Show that $\mathbf{b}_1 = [1, 1]^T$ and $\mathbf{b}_2 = [1, -1]^T$ form a basis for $\mathbb{R}^2$. Then express $\mathbf{v} = [5, 3]^T$ in this basis.

### Exercise 4: Subspace Verification

Which of the following are subspaces of $\mathbb{R}^3$?

(a) $W_1 = \{[x, y, z]^T : x + y + z = 0\}$

(b) $W_2 = \{[x, y, z]^T : x^2 + y^2 + z^2 \le 1\}$

(c) $W_3 = \{[x, y, z]^T : x = 2y\}$

### Exercise 5: Span

Find the span of $\{[1, 2, 3]^T, [4, 5, 6]^T, [7, 8, 9]^T\}$. What is the dimension of this span? (Hint: check the rank of the matrix formed by these vectors.)

---

[Overview](00_Overview.md) | [Next: Lesson 2 - Matrices and Operations >>](02_Matrices_and_Operations.md)

**License**: CC BY-NC 4.0
