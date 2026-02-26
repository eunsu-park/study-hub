# 01. Vectors and Matrices

## Learning Objectives

- Understand the geometric meaning and algebraic operations of vectors and implement them in Python
- Understand the concepts of vector spaces, basis, and dimension, and determine linear independence
- Interpret matrices as linear transformations and visualize their geometric meaning
- Understand the concepts of determinants, inverse matrices, and rank, and solve systems of equations
- Give concrete examples of how vectors and matrices are used in machine learning

---

## 1. Vector Definition and Operations

### 1.1 What is a Vector?

A vector is a quantity with magnitude and direction. Vectors can be understood from two perspectives:

1. **Geometric perspective**: An arrow. Displacement from starting point to endpoint
2. **Algebraic perspective**: An ordered list of numbers

An n-dimensional vector $\mathbf{v}$ is represented as:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Create 2D vectors
v = np.array([3, 2])
w = np.array([1, 3])

# Visualize vectors
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', width=0.006, label='v')
ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='b', width=0.006, label='w')
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title('2D Vectors')
plt.show()
```

### 1.2 Vector Operations

**Vector addition**: Adding two vectors follows the parallelogram law.

$$\mathbf{v} + \mathbf{w} = \begin{bmatrix} v_1 + w_1 \\ v_2 + w_2 \\ \vdots \\ v_n + w_n \end{bmatrix}$$

**Scalar multiplication**: Multiplying a scalar $c$ by a vector maintains direction and only changes magnitude.

$$c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$$

```python
# Vector addition and scalar multiplication
v_plus_w = v + w
c = 2
cv = c * v

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', width=0.006, label='v')
ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='b', width=0.006, label='w')
ax.quiver(0, 0, v_plus_w[0], v_plus_w[1], angles='xy', scale_units='xy', scale=1,
          color='purple', width=0.006, label='v+w')
ax.quiver(v[0], v[1], w[0], w[1], angles='xy', scale_units='xy', scale=1,
          color='gray', width=0.004, linestyle='--', alpha=0.5)
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title('Vector Addition (Parallelogram Law)')
plt.show()
```

### 1.3 Dot Product

The dot product of two vectors returns a scalar value:

$$\mathbf{v} \cdot \mathbf{w} = \sum_{i=1}^n v_i w_i = v_1w_1 + v_2w_2 + \cdots + v_nw_n$$

Geometric interpretation:

$$\mathbf{v} \cdot \mathbf{w} = \|\mathbf{v}\| \|\mathbf{w}\| \cos\theta$$

where $\theta$ is the angle between the two vectors.

**Meaning of the dot product**:
- $\mathbf{v} \cdot \mathbf{w} > 0$: Acute angle (same direction)
- $\mathbf{v} \cdot \mathbf{w} = 0$: Right angle (orthogonal)
- $\mathbf{v} \cdot \mathbf{w} < 0$: Obtuse angle (opposite direction)

```python
# Dot product calculation
dot_product = np.dot(v, w)
print(f"v · w = {dot_product}")

# Angle calculation
norm_v = np.linalg.norm(v)
norm_w = np.linalg.norm(w)
cos_theta = dot_product / (norm_v * norm_w)
theta = np.arccos(cos_theta)
print(f"Angle between v and w: {np.degrees(theta):.2f} degrees")

# Orthogonal vector example
u = np.array([1, 0])
v_orth = np.array([0, 1])
print(f"u · v_orth = {np.dot(u, v_orth)}")  # 0 (orthogonal)
```

### 1.4 Cross Product - 3D Only

Defined only for 3D vectors, the result is a vector perpendicular to both:

$$\mathbf{v} \times \mathbf{w} = \begin{bmatrix} v_2w_3 - v_3w_2 \\ v_3w_1 - v_1w_3 \\ v_1w_2 - v_2w_1 \end{bmatrix}$$

The magnitude is the area of the parallelogram: $\|\mathbf{v} \times \mathbf{w}\| = \|\mathbf{v}\| \|\mathbf{w}\| \sin\theta$

```python
# 3D cross product
v3d = np.array([1, 2, 3])
w3d = np.array([4, 5, 6])
cross_product = np.cross(v3d, w3d)
print(f"v × w = {cross_product}")

# Verify that the cross product is perpendicular to both vectors
print(f"(v × w) · v = {np.dot(cross_product, v3d)}")  # ~0
print(f"(v × w) · w = {np.dot(cross_product, w3d)}")  # ~0
```

## 2. Vector Spaces

### 2.1 Vector Space Definition

A vector space $V$ is a set where the following two operations are defined:
1. **Vector addition**: $\mathbf{u} + \mathbf{v} \in V$
2. **Scalar multiplication**: $c\mathbf{v} \in V$

And it must satisfy 8 axioms (commutativity, associativity, identity, inverse, distributivity, etc.).

**Examples**:
- $\mathbb{R}^n$: All n-dimensional real vectors
- Polynomial space: All polynomials of degree $n$ or less
- Function space: All continuous functions on interval $[a, b]$

### 2.2 Subspace

For a subset $W$ of vector space $V$ to be a subspace:
1. Contains zero vector: $\mathbf{0} \in W$
2. Closed under addition: $\mathbf{u}, \mathbf{v} \in W \Rightarrow \mathbf{u} + \mathbf{v} \in W$
3. Closed under scalar multiplication: $\mathbf{v} \in W, c \in \mathbb{R} \Rightarrow c\mathbf{v} \in W$

**Example**: In $\mathbb{R}^3$, a plane or line through the origin is a subspace.

```python
# Subspace example: the plane z=0 in R^3 (xy-plane)
# This plane is a subspace of R^3

# Two vectors on the plane
u = np.array([1, 2, 0])
v = np.array([3, -1, 0])

# Check closure under addition
print(f"u + v = {u + v}")  # [4, 1, 0] - still on the z=0 plane
print(f"2*u = {2*u}")      # [2, 4, 0] - still on the z=0 plane
```

### 2.3 Span and Linear Combinations

A linear combination of vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$$

Span: The set of all possible linear combinations

$$\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \{c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k : c_i \in \mathbb{R}\}$$

```python
# Span visualization: span of two vectors forms a plane
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

# Generate various linear combinations
combinations = []
for c1 in np.linspace(-2, 2, 20):
    for c2 in np.linspace(-2, 2, 20):
        combinations.append(c1 * v1 + c2 * v2)

combinations = np.array(combinations)

# 3D plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(combinations[:, 0], combinations[:, 1], combinations[:, 2],
           alpha=0.3, s=1)
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', arrow_length_ratio=0.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Span of two vectors in R^3')
plt.show()
```

### 2.4 Linear Independence

Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are linearly independent if:

$$c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \Rightarrow c_1 = \cdots = c_k = 0$$

That is, the zero vector cannot be formed by a non-trivial linear combination.

**Test**: If the rank of the matrix with vectors as columns equals the number of vectors, they are linearly independent.

```python
# Linear independence test
vectors_independent = np.array([[1, 2], [2, 3], [3, 4]]).T  # each column is a vector
vectors_dependent = np.array([[1, 2], [2, 4], [3, 6]]).T

rank_indep = np.linalg.matrix_rank(vectors_independent)
rank_dep = np.linalg.matrix_rank(vectors_dependent)

print(f"Independent vectors rank: {rank_indep} (# vectors: 3)")
print(f"Dependent vectors rank: {rank_dep} (# vectors: 3)")
print(f"First set is linearly independent: {rank_indep == 3}")
print(f"Second set is linearly independent: {rank_dep == 3}")
```

### 2.5 Basis and Dimension

**Basis**: A set of linearly independent vectors that span vector space $V$

**Dimension**: The number of vectors in a basis (all bases have the same count)

Standard basis of $\mathbb{R}^n$:

$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix}, \quad \ldots, \quad \mathbf{e}_n = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$

```python
# Standard basis
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

# Express an arbitrary vector in the basis
v = np.array([5, 3, -2])
print(f"v = {v[0]}*e1 + {v[1]}*e2 + {v[2]}*e3")
print(f"Verification: {v[0]*e1 + v[1]*e2 + v[2]*e3}")

# Using a different basis
b1 = np.array([1, 1, 0])
b2 = np.array([0, 1, 1])
b3 = np.array([1, 0, 1])

# To express v in the new basis, solve a system of equations
B = np.column_stack([b1, b2, b3])
coords = np.linalg.solve(B, v)
print(f"v in new basis: {coords}")
print(f"Verification: {coords[0]*b1 + coords[1]*b2 + coords[2]*b3}")
```

## 3. Matrix Fundamentals

### 3.1 Matrix Definition and Operations

An $m \times n$ matrix $A$ is an array of numbers with $m$ rows and $n$ columns:

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

**Matrix addition**: Only possible for matrices of the same size

$$C = A + B \quad \Rightarrow \quad c_{ij} = a_{ij} + b_{ij}$$

**Matrix multiplication**: When $A$ is $m \times n$ and $B$ is $n \times p$, $C = AB$ is $m \times p$

$$(AB)_{ij} = \sum_{k=1}^n a_{ik}b_{kj}$$

```python
# Matrix creation and operations
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

print(f"A shape: {A.shape}")  # (2, 3)
print(f"B shape: {B.shape}")  # (3, 2)

# Matrix multiplication
C = A @ B  # or np.matmul(A, B) or np.dot(A, B)
print(f"C = A @ B:\n{C}")
print(f"C shape: {C.shape}")  # (2, 2)

# Element-wise multiplication (Hadamard product)
D = np.array([[1, 2], [3, 4]])
E = np.array([[5, 6], [7, 8]])
F = D * E  # element-wise
print(f"D * E (element-wise):\n{F}")
```

### 3.2 Transpose, Inverse, and Determinant

**Transpose**: Matrix with rows and columns swapped

$$A^T_{ij} = A_{ji}$$

**Inverse**: $AA^{-1} = A^{-1}A = I$ (exists only for square matrices with full rank)

**Determinant**: Scalar value defined for square matrices
- $\det(A) \neq 0 \Leftrightarrow A$ has an inverse

```python
# Transpose
A = np.array([[1, 2, 3],
              [4, 5, 6]])
A_T = A.T
print(f"A^T:\n{A_T}")

# Inverse matrix
B = np.array([[1, 2],
              [3, 4]])
B_inv = np.linalg.inv(B)
print(f"B inverse:\n{B_inv}")
print(f"B @ B_inv:\n{B @ B_inv}")  # identity matrix

# Determinant
det_B = np.linalg.det(B)
print(f"det(B) = {det_B}")

# Singular matrix (no inverse)
C = np.array([[1, 2],
              [2, 4]])
print(f"det(C) = {np.linalg.det(C):.10f}")  # ~0
# C_inv = np.linalg.inv(C)  # raises LinAlgError
```

### 3.3 Special Matrices

**Identity matrix**: $I_n$

$$I_n = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}$$

**Symmetric matrix**: $A = A^T$

**Orthogonal matrix**: $Q^TQ = QQ^T = I$ (columns form an orthonormal basis)

**Diagonal matrix**: All zeros except on diagonal

**Triangular matrix**: Upper or lower triangular

```python
# Creating special matrices
I = np.eye(3)  # 3x3 identity matrix
print(f"Identity matrix:\n{I}")

# Diagonal matrix
D = np.diag([1, 2, 3])
print(f"Diagonal matrix:\n{D}")

# Symmetric matrix
S = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])
print(f"Is symmetric? {np.allclose(S, S.T)}")

# Orthogonal matrix (rotation matrix)
theta = np.pi / 4
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
print(f"Q^T @ Q:\n{Q.T @ Q}")  # identity matrix
```

## 4. Linear Transformations

### 4.1 Matrix = Linear Transformation

A matrix $A$ can be viewed as a function that transforms one vector into another:

$$T(\mathbf{v}) = A\mathbf{v}$$

Two properties of linear transformations:
1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. $T(c\mathbf{v}) = cT(\mathbf{v})$

```python
# Linear transformation example
def linear_transform(A, v):
    return A @ v

# Transformation matrix
A = np.array([[2, 0],
              [0, 0.5]])

# Unit square
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

# After transformation
transformed = A @ square

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(square[0], square[1], 'b-o')
ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.5, 2.5)
ax1.set_aspect('equal')
ax1.grid(True)
ax1.set_title('Original')

ax2.plot(transformed[0], transformed[1], 'r-o')
ax2.set_xlim(-0.5, 2.5)
ax2.set_ylim(-0.5, 2.5)
ax2.set_aspect('equal')
ax2.grid(True)
ax2.set_title(f'After transformation by A')

plt.show()
```

### 4.2 Geometric Transformations

**Scaling**:

$$\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

**Rotation**:

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**Reflection**: Reflection about x-axis

$$\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$

**Shear**: Shear in x direction

$$\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$$

```python
# Visualize various transformations
theta = np.pi / 6  # 30 degrees

transformations = {
    'Scaling': np.array([[2, 0], [0, 0.5]]),
    'Rotation': np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]]),
    'Reflection': np.array([[1, 0], [0, -1]]),
    'Shear': np.array([[1, 0.5], [0, 1]])
}

# Original shape
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()

for idx, (name, A) in enumerate(transformations.items()):
    transformed = A @ square

    axes[idx].plot(square[0], square[1], 'b-o', label='Original', alpha=0.5)
    axes[idx].plot(transformed[0], transformed[1], 'r-o', label='Transformed')
    axes[idx].set_xlim(-1.5, 2.5)
    axes[idx].set_ylim(-1.5, 2.5)
    axes[idx].set_aspect('equal')
    axes[idx].grid(True)
    axes[idx].legend()
    axes[idx].set_title(name)

plt.tight_layout()
plt.show()
```

### 4.3 Projection

Projection of vector $\mathbf{v}$ onto vector $\mathbf{u}$:

$$\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\mathbf{v} \cdot \mathbf{u}}{\mathbf{u} \cdot \mathbf{u}} \mathbf{u}$$

Projection onto subspaces can be represented as matrices.

```python
# Vector projection
def project(v, u):
    """Project vector v onto u"""
    return (np.dot(v, u) / np.dot(u, u)) * u

v = np.array([3, 2])
u = np.array([1, 0])

proj_v = project(v, u)

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
          color='blue', width=0.006, label='v')
ax.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1,
          color='red', width=0.006, label='u')
ax.quiver(0, 0, proj_v[0], proj_v[1], angles='xy', scale_units='xy', scale=1,
          color='green', width=0.006, label='proj_u(v)')
ax.plot([proj_v[0], v[0]], [proj_v[1], v[1]], 'k--', alpha=0.5)
ax.set_xlim(-0.5, 4)
ax.set_ylim(-0.5, 3)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title('Vector Projection')
plt.show()
```

## 5. Rank and Systems of Equations

### 5.1 Matrix Rank

The rank of a matrix is the maximum number of linearly independent rows (or columns).

**Properties**:
- $\text{rank}(A) \leq \min(m, n)$ for $m \times n$ matrix
- $\text{rank}(A) = \text{rank}(A^T)$
- Full rank: $\text{rank}(A) = \min(m, n)$

```python
# Rank computation
A_full = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(f"rank(A_full) = {np.linalg.matrix_rank(A_full)}")  # 2 (not full rank)

A_rank2 = np.array([[1, 2],
                    [3, 4],
                    [5, 6]])
print(f"rank(A_rank2) = {np.linalg.matrix_rank(A_rank2)}")  # 2 (full rank)
```

### 5.2 System of Equations $A\mathbf{x} = \mathbf{b}$

**Solution existence**:
- Solution exists $\Leftrightarrow$ $\mathbf{b} \in \text{col}(A)$ (column space of A)
- If $\text{rank}(A) = \text{rank}([A|\mathbf{b}])$, solution exists

**Solution uniqueness**:
- Unique solution $\Leftrightarrow$ A has full column rank
- Infinitely many solutions $\Leftrightarrow$ A does not have full column rank

| $\text{rank}(A)$ | $\text{rank}([A\|\mathbf{b}])$ | Number of solutions |
|------------------|--------------------------------|---------------------|
| $r$ | $r$ | Infinitely many (if $r < n$) or unique (if $r = n$) |
| $r$ | $r+1$ | No solution |

```python
# Case with a unique solution
A = np.array([[2, 1],
              [1, 3]])
b = np.array([5, 8])
x = np.linalg.solve(A, b)
print(f"Solution: x = {x}")
print(f"Verification: Ax = {A @ x}")

# Case with no solution (find least squares solution)
A_overdetermined = np.array([[1, 1],
                             [1, 2],
                             [1, 3]])
b_overdetermined = np.array([1, 2, 4])  # no exact solution exists

# Least squares solution
x_lstsq = np.linalg.lstsq(A_overdetermined, b_overdetermined, rcond=None)[0]
print(f"Least squares solution: {x_lstsq}")
print(f"Residual: {A_overdetermined @ x_lstsq - b_overdetermined}")
```

### 5.3 Gaussian Elimination

An algorithm for solving systems of equations. Transforms the matrix into row echelon form.

```python
def gaussian_elimination(A, b):
    """Solve Ax = b using Gaussian elimination"""
    n = len(b)
    # Create augmented matrix
    Ab = np.column_stack([A.astype(float), b.astype(float)])

    # Forward elimination
    for i in range(n):
        # Choose pivot (partial pivoting)
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        Ab[[i, max_row]] = Ab[[max_row, i]]

        # Elimination
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]

    return x

A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

x_gauss = gaussian_elimination(A, b)
print(f"Solution by Gaussian elimination: {x_gauss}")
print(f"Verification: {A @ x_gauss}")
```

## 6. Vectors and Matrices in ML

### 6.1 Feature Vectors

In machine learning, each data point is represented as a vector.

```python
# Example: Iris dataset
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # (150, 4) - 150 samples, 4 features
y = iris.target

print(f"Data shape: {X.shape}")
print(f"First sample (feature vector):\n{X[0]}")
print(f"Feature names: {iris.feature_names}")
```

### 6.2 Data Matrix and Batch Processing

**Data matrix**: Each row is one sample (convention)

$$X = \begin{bmatrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_m^T \end{bmatrix} = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1n} \\ x_{21} & x_{22} & \cdots & x_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{m1} & x_{m2} & \cdots & x_{mn} \end{bmatrix}$$

Matrix operations allow processing entire batches at once.

```python
# Batch processing example: linear regression
np.random.seed(42)
X = np.random.randn(100, 3)  # 100 samples, 3 features
true_w = np.array([2, -1, 0.5])
y = X @ true_w + np.random.randn(100) * 0.1

# Solve with normal equation: w = (X^T X)^{-1} X^T y
w_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"True weights: {true_w}")
print(f"Estimated weights: {w_hat}")

# Prediction (batch processing)
y_pred = X @ w_hat
print(f"Predictions for first 5 samples:\n{y_pred[:5]}")
```

### 6.3 Weight Matrices

Each layer of a neural network is represented as a weight matrix.

```python
# Simple neural network layer
class DenseLayer:
    def __init__(self, input_dim, output_dim):
        # He initialization
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2/input_dim)
        self.b = np.zeros(output_dim)

    def forward(self, X):
        """X: (batch_size, input_dim)"""
        return X @ self.W + self.b

# Example
layer = DenseLayer(input_dim=10, output_dim=5)
X_batch = np.random.randn(32, 10)  # batch size 32
output = layer.forward(X_batch)
print(f"Input shape: {X_batch.shape}")
print(f"Weight matrix shape: {layer.W.shape}")
print(f"Output shape: {output.shape}")
```

### 6.4 Cosine Similarity and Distance

Methods for measuring similarity between vectors:

**Cosine similarity**:

$$\text{similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \cos\theta$$

**Euclidean distance**:

$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\| = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}$$

```python
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Document vector example (TF-IDF)
doc1 = np.array([1, 2, 0, 1])
doc2 = np.array([2, 1, 1, 0])
doc3 = np.array([0, 0, 1, 1])

docs = np.array([doc1, doc2, doc3])

# Cosine similarity
cos_sim = cosine_similarity(docs)
print(f"Cosine similarity matrix:\n{cos_sim}")

# Euclidean distance
eucl_dist = euclidean_distances(docs)
print(f"Euclidean distance matrix:\n{eucl_dist}")
```

## Practice Problems

### Problem 1: Vector Spaces and Basis
Determine whether the following vectors form a basis for $\mathbb{R}^3$. If they do, express vector $\mathbf{v} = [1, 2, 3]^T$ in this basis.

$$\mathbf{b}_1 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \quad \mathbf{b}_2 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}, \quad \mathbf{b}_3 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$$

### Problem 2: Linear Transformation Visualization
Find and visualize the final transformation matrix when applying the following transformations in order:
1. 45 degree rotation
2. Scale by 2 in x direction
3. Reflection about x-axis

### Problem 3: Projection Matrix
Find the projection matrix $P$ onto the xy-plane (i.e., the plane $z=0$). That is, find a $3 \times 3$ matrix $P$ such that $P\mathbf{v}$ projects $\mathbf{v}$ onto the xy-plane.

### Problem 4: Least Squares Problem
Find the least squares solution to the following overdetermined system:

$$\begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 2 \\ 3 \\ 5 \\ 6 \end{bmatrix}$$

### Problem 5: ML Application
There is a dataset with 100 samples and 5 features. You want to train a linear regression model.
1. Specify the dimensions of data matrix $X$ and weight vector $\mathbf{w}$
2. What is the dimension of predicted value $\hat{\mathbf{y}} = X\mathbf{w}$?
3. Verify the dimensions of each matrix in the normal equation $\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$

## References

### Textbooks
1. **Strang, G.** (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.
   - Classic linear algebra text with excellent geometric intuition.
2. **Axler, S.** (2015). *Linear Algebra Done Right* (3rd ed.). Springer.
   - Abstract but rigorous approach.
3. **Boyd, S., & Vandenberghe, L.** (2018). *Introduction to Applied Linear Algebra*. Cambridge University Press.
   - Modern, application-focused approach.

### Online Resources
1. **3Blue1Brown - Essence of Linear Algebra**: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
   - The textbook of visualization
2. **MIT 18.06 - Linear Algebra (Gilbert Strang)**: https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/
3. **Khan Academy - Linear Algebra**: https://www.khanacademy.org/math/linear-algebra

### Practice Tools
1. **NumPy Documentation**: https://numpy.org/doc/stable/
2. **Matrix Calculator**: https://matrixcalc.org/
3. **Wolfram Alpha**: https://www.wolframalpha.com/

---

**Next lesson**: [02. Matrix Decompositions](02_Matrix_Decompositions.md) to learn eigenvalue decomposition and SVD.
