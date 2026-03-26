# Lesson 5: Linear Transformations

## Learning Objectives

- Define linear transformations and verify the linearity conditions
- Represent linear transformations as matrices and compose transformations via matrix multiplication
- Compute the kernel (null space) and image (column space) of a transformation
- State and apply the rank-nullity theorem
- Visualize common 2D transformations: rotation, scaling, reflection, shear, and projection

---

## 1. Definition of a Linear Transformation

### 1.1 What Is a Linear Transformation?

A function $T : \mathbb{R}^n \to \mathbb{R}^m$ is a **linear transformation** if it satisfies two properties for all $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ and $c \in \mathbb{R}$:

1. **Additivity**: $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **Homogeneity**: $T(c\mathbf{v}) = c \, T(\mathbf{v})$

Equivalently, $T$ is linear if and only if:

$$T(\alpha\mathbf{u} + \beta\mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})$$

A linear transformation always maps the zero vector to the zero vector: $T(\mathbf{0}) = \mathbf{0}$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Example: T(x, y) = (2x + y, x - y)
def T(v):
    return np.array([2*v[0] + v[1], v[0] - v[1]])

# Verify linearity
u = np.array([1, 3])
v = np.array([2, -1])
alpha, beta = 3, -2

lhs = T(alpha * u + beta * v)
rhs = alpha * T(u) + beta * T(v)
print(f"T(alpha*u + beta*v) = {lhs}")
print(f"alpha*T(u) + beta*T(v) = {rhs}")
print(f"Linear? {np.allclose(lhs, rhs)}")

# Counter-example: T(v) = v + [1, 0] is NOT linear
def T_affine(v):
    return v + np.array([1, 0])

lhs = T_affine(u + v)
rhs = T_affine(u) + T_affine(v)
print(f"\nAffine T(u+v) = {lhs}")
print(f"T(u) + T(v) = {rhs}")
print(f"Linear? {np.allclose(lhs, rhs)}")  # False
```

---

## 2. Matrix Representation

### 2.1 Every Linear Transformation Has a Matrix

For any linear transformation $T : \mathbb{R}^n \to \mathbb{R}^m$, there exists a unique $m \times n$ matrix $A$ such that:

$$T(\mathbf{v}) = A\mathbf{v}$$

The columns of $A$ are the images of the standard basis vectors:

$$A = \begin{bmatrix} T(\mathbf{e}_1) & T(\mathbf{e}_2) & \cdots & T(\mathbf{e}_n) \end{bmatrix}$$

```python
# Find the matrix of T(x, y) = (2x + y, x - y)
e1 = np.array([1, 0])
e2 = np.array([0, 1])

col1 = T(e1)  # [2, 1]
col2 = T(e2)  # [1, -1]

A = np.column_stack([col1, col2])
print(f"Matrix A:\n{A}")

# Verify
v = np.array([3, 5])
print(f"T(v) = {T(v)}")
print(f"A @ v = {A @ v}")
```

### 2.2 Common 2D Transformation Matrices

| Transformation | Matrix | Effect |
|----------------|--------|--------|
| Rotation by $\theta$ | $\begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$ | Rotates counterclockwise |
| Scaling | $\begin{bmatrix}s_x & 0 \\ 0 & s_y\end{bmatrix}$ | Scales along axes |
| Reflection (x-axis) | $\begin{bmatrix}1 & 0 \\ 0 & -1\end{bmatrix}$ | Flips vertically |
| Reflection (y-axis) | $\begin{bmatrix}-1 & 0 \\ 0 & 1\end{bmatrix}$ | Flips horizontally |
| Shear (x-direction) | $\begin{bmatrix}1 & k \\ 0 & 1\end{bmatrix}$ | Slants horizontally |
| Projection onto x-axis | $\begin{bmatrix}1 & 0 \\ 0 & 0\end{bmatrix}$ | Collapses y-component |

```python
theta = np.pi / 6  # 30 degrees

transforms = {
    'Rotation (30 deg)': np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta),  np.cos(theta)]]),
    'Scale (2x, 0.5y)': np.array([[2, 0],
                                   [0, 0.5]]),
    'Reflect (x-axis)':  np.array([[1,  0],
                                   [0, -1]]),
    'Shear (k=0.5)':     np.array([[1, 0.5],
                                   [0, 1]]),
    'Projection (x)':    np.array([[1, 0],
                                   [0, 0]]),
}

# Original shape: unit square
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, (name, M) in zip(axes, transforms.items()):
    transformed = M @ square

    ax.fill(square[0], square[1], alpha=0.3, color='blue', label='Original')
    ax.fill(transformed[0], transformed[1], alpha=0.3, color='red', label='Transformed')
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(name, fontsize=9)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.show()
```

---

## 3. Composition of Transformations

### 3.1 Composing Two Transformations

If $T_1(\mathbf{v}) = A\mathbf{v}$ and $T_2(\mathbf{v}) = B\mathbf{v}$, then the composition $T_2 \circ T_1$ is:

$$(T_2 \circ T_1)(\mathbf{v}) = B(A\mathbf{v}) = (BA)\mathbf{v}$$

**Note the order**: Apply $A$ first, then $B$, but the combined matrix is $BA$ (right to left).

```python
# Rotate 45 degrees, then scale by 2 in x
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
S = np.array([[2, 0],
              [0, 1]])

# Composition: first R, then S
M_compose = S @ R

# Verify
v = np.array([1, 0])
step1 = R @ v
step2 = S @ step1
combined = M_compose @ v

print(f"Step-by-step: R@v = {step1}, S@(R@v) = {step2}")
print(f"Combined:     (S@R)@v = {combined}")
print(f"Match? {np.allclose(step2, combined)}")

# Visualize
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].fill(square[0], square[1], alpha=0.3, color='blue')
axes[0].set_title('Original')

rotated = R @ square
axes[1].fill(rotated[0], rotated[1], alpha=0.3, color='green')
axes[1].set_title('After Rotation (45 deg)')

composed = M_compose @ square
axes[2].fill(composed[0], composed[1], alpha=0.3, color='red')
axes[2].set_title('After Rotation + Scale')

for ax in axes:
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1, 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.2 Order Matters

Since matrix multiplication is not commutative, the order of composition matters:

$$T_1 \circ T_2 \neq T_2 \circ T_1 \quad \text{(in general)}$$

```python
# Rotation then scale vs scale then rotation
SR = S @ R  # rotate first, then scale
RS = R @ S  # scale first, then rotate

print(f"S @ R =\n{np.round(SR, 4)}")
print(f"R @ S =\n{np.round(RS, 4)}")
print(f"Same? {np.allclose(SR, RS)}")
```

---

## 4. Kernel (Null Space)

### 4.1 Definition

The **kernel** (or **null space**) of a linear transformation $T(\mathbf{v}) = A\mathbf{v}$ is:

$$\ker(T) = \mathrm{null}(A) = \{\mathbf{v} \in \mathbb{R}^n : A\mathbf{v} = \mathbf{0}\}$$

The kernel is always a subspace of $\mathbb{R}^n$. Its dimension is called the **nullity** of $A$.

### 4.2 Interpretation

- If $\ker(T) = \{\mathbf{0}\}$, the transformation is **injective** (one-to-one): distinct inputs map to distinct outputs.
- If $\ker(T)$ is non-trivial, the transformation "collapses" part of the domain.

```python
from scipy.linalg import null_space

# Projection onto x-axis: kills the y-component
P = np.array([[1, 0],
              [0, 0]], dtype=float)

ns = null_space(P)
print(f"Kernel of projection:\n{ns}")
print(f"Nullity = {ns.shape[1]}")

# Rank-2 matrix in R^3 -> R^3 (has non-trivial kernel)
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

ns_A = null_space(A)
print(f"\nKernel of A:\n{ns_A}")
print(f"Nullity = {ns_A.shape[1]}")

# Verify kernel vector is in null space
if ns_A.shape[1] > 0:
    print(f"A @ kernel_vec = {A @ ns_A[:, 0]}")  # should be ~0
```

---

## 5. Image (Column Space)

### 5.1 Definition

The **image** (or **column space**, or **range**) of a linear transformation $T(\mathbf{v}) = A\mathbf{v}$ is:

$$\mathrm{im}(T) = \mathrm{col}(A) = \{A\mathbf{v} : \mathbf{v} \in \mathbb{R}^n\}$$

This is the span of the columns of $A$. Its dimension is the **rank** of $A$.

### 5.2 Computing a Basis for the Column Space

Reduce $A$ to echelon form and identify the pivot columns. The corresponding columns of the **original** matrix $A$ form a basis for $\mathrm{col}(A)$.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

rank = np.linalg.matrix_rank(A)
print(f"Rank of A: {rank}")

# Use SVD to find an orthonormal basis for the column space
U, s, Vt = np.linalg.svd(A)
col_space_basis = U[:, :rank]
print(f"Orthonormal basis for col(A):\n{col_space_basis}")

# Verify: any column of A should be in the column space
# Project column 2 onto the column space
col2 = A[:, 2]
proj = col_space_basis @ (col_space_basis.T @ col2)
print(f"\nColumn 2 of A: {col2}")
print(f"Projection onto col(A): {proj}")
print(f"In column space? {np.allclose(col2, proj)}")
```

---

## 6. Rank-Nullity Theorem

### 6.1 Statement

For any $m \times n$ matrix $A$:

$$\mathrm{rank}(A) + \mathrm{nullity}(A) = n$$

Equivalently:

$$\dim(\mathrm{im}(T)) + \dim(\ker(T)) = \dim(\text{domain})$$

### 6.2 Intuition

The domain of the transformation is "split" into two parts:
- The part that maps to distinct outputs (dimension = rank)
- The part that collapses to zero (dimension = nullity)

### 6.3 Verification

```python
def verify_rank_nullity(A):
    """Verify the rank-nullity theorem for matrix A."""
    n = A.shape[1]  # number of columns = dimension of domain
    rank = np.linalg.matrix_rank(A)
    ns = null_space(A)
    nullity = ns.shape[1]

    print(f"Matrix shape: {A.shape}")
    print(f"n (domain dim): {n}")
    print(f"Rank:           {rank}")
    print(f"Nullity:        {nullity}")
    print(f"Rank + Nullity = {rank + nullity} == n = {n}: {rank + nullity == n}")
    print()

# Full rank square matrix
A1 = np.array([[1, 2], [3, 4]], dtype=float)
verify_rank_nullity(A1)

# Rank-deficient square matrix
A2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
verify_rank_nullity(A2)

# Rectangular matrix (wide)
A3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
verify_rank_nullity(A3)

# Rectangular matrix (tall)
A4 = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
verify_rank_nullity(A4)
```

---

## 7. The Four Fundamental Subspaces

Every $m \times n$ matrix $A$ defines four fundamental subspaces (due to Gilbert Strang):

| Subspace | Definition | Dimension |
|----------|------------|-----------|
| Column space $\mathrm{col}(A)$ | $\{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}$ | $r$ |
| Null space $\mathrm{null}(A)$ | $\{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ | $n - r$ |
| Row space $\mathrm{col}(A^T)$ | $\{A^T\mathbf{y} : \mathbf{y} \in \mathbb{R}^m\}$ | $r$ |
| Left null space $\mathrm{null}(A^T)$ | $\{\mathbf{y} : A^T\mathbf{y} = \mathbf{0}\}$ | $m - r$ |

where $r = \mathrm{rank}(A)$.

**Key orthogonality relationships**:
- $\mathrm{col}(A) \perp \mathrm{null}(A^T)$ in $\mathbb{R}^m$
- $\mathrm{col}(A^T) \perp \mathrm{null}(A)$ in $\mathbb{R}^n$

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=float)

r = np.linalg.matrix_rank(A)
m, n = A.shape

# Compute all four subspaces via SVD
U, s, Vt = np.linalg.svd(A)

col_space = U[:, :r]           # Column space basis
left_null = U[:, r:]           # Left null space basis
row_space = Vt[:r, :].T       # Row space basis
null_space_A = Vt[r:, :].T    # Null space basis

print(f"Matrix A ({m}x{n}), rank = {r}")
print(f"Column space dim:    {col_space.shape[1]} (in R^{m})")
print(f"Left null space dim: {left_null.shape[1]} (in R^{m})")
print(f"Row space dim:       {row_space.shape[1]} (in R^{n})")
print(f"Null space dim:      {null_space_A.shape[1]} (in R^{n})")

# Verify orthogonality
print(f"\ncol(A)^T @ null(A^T) =\n{np.round(col_space.T @ left_null, 10)}")
print(f"row(A)^T @ null(A) =\n{np.round(row_space.T @ null_space_A, 10)}")
```

---

## 8. Invertible Transformations

A linear transformation $T : \mathbb{R}^n \to \mathbb{R}^n$ is **invertible** if and only if:

- $\ker(T) = \{\mathbf{0}\}$ (injective)
- $\mathrm{im}(T) = \mathbb{R}^n$ (surjective)
- $\mathrm{rank}(A) = n$ (full rank)
- $\det(A) \neq 0$ (non-singular)

All of these conditions are equivalent for square matrices.

```python
# Invertible transformation
A = np.array([[2, 1],
              [1, 3]], dtype=float)
print(f"det(A) = {np.linalg.det(A):.4f}")
print(f"rank(A) = {np.linalg.matrix_rank(A)}")

# Apply and then invert
v = np.array([3, 5])
Tv = A @ v
v_recovered = np.linalg.inv(A) @ Tv
print(f"v = {v}")
print(f"T(v) = {Tv}")
print(f"T^(-1)(T(v)) = {v_recovered}")
print(f"Recovered original? {np.allclose(v, v_recovered)}")
```

---

## 9. Change of Basis as a Transformation

If $B = [\mathbf{b}_1 | \cdots | \mathbf{b}_n]$ is a change-of-basis matrix, then the same linear transformation $T$ has different matrix representations in different bases:

$$A' = B^{-1} A B$$

This is called a **similarity transformation**. $A$ and $A'$ are **similar matrices**.

```python
# Same transformation in two different bases
A = np.array([[3, 1],
              [0, 2]], dtype=float)

# Change of basis
B = np.array([[1, 1],
              [1, -1]], dtype=float)

A_prime = np.linalg.inv(B) @ A @ B
print(f"A in standard basis:\n{A}")
print(f"A in new basis:\n{A_prime}")

# Verify: eigenvalues are preserved under similarity
eig_A = np.sort(np.linalg.eigvals(A))
eig_A_prime = np.sort(np.linalg.eigvals(A_prime))
print(f"\nEigenvalues of A:  {eig_A}")
print(f"Eigenvalues of A': {eig_A_prime}")
print(f"Same? {np.allclose(eig_A, eig_A_prime)}")
```

---

## 10. Summary

| Concept | Description |
|---------|-------------|
| Linear transformation | $T(\alpha\mathbf{u} + \beta\mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})$ |
| Matrix representation | $T(\mathbf{v}) = A\mathbf{v}$ where columns of $A$ are $T(\mathbf{e}_i)$ |
| Composition | $(T_2 \circ T_1)(\mathbf{v}) = (B A)\mathbf{v}$ |
| Kernel (null space) | $\ker(T) = \{\mathbf{v} : A\mathbf{v} = \mathbf{0}\}$ |
| Image (column space) | $\mathrm{im}(T) = \{A\mathbf{v} : \mathbf{v} \in \mathbb{R}^n\}$ |
| Rank-nullity theorem | $\mathrm{rank}(A) + \mathrm{nullity}(A) = n$ |
| Similarity | $A' = B^{-1}AB$ preserves eigenvalues |

---

## Exercises

### Exercise 1: Matrix of a Transformation

Find the matrix representation of $T : \mathbb{R}^3 \to \mathbb{R}^2$ defined by $T(x, y, z) = (x + 2y - z, \; 3x - y + z)$. Verify by computing $T(1, 2, 3)$ both directly and via the matrix.

### Exercise 2: Kernel and Image

For the matrix $A = \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 3 & 6 & 3 \end{bmatrix}$:

1. Find a basis for $\ker(A)$.
2. Find a basis for $\mathrm{im}(A)$.
3. Verify the rank-nullity theorem.

### Exercise 3: Composition

Find the single matrix that represents: first rotate by 90 degrees counterclockwise, then reflect about the x-axis. Apply this to the unit square and plot the result.

### Exercise 4: Four Fundamental Subspaces

For $A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}$, find bases for all four fundamental subspaces and verify the orthogonality relationships.

### Exercise 5: Invertibility

Determine which of the following transformations are invertible. For those that are, find the inverse transformation.

(a) $T(x, y) = (2x - y, \; x + 3y)$

(b) $T(x, y) = (x + 2y, \; 2x + 4y)$

(c) $T(x, y, z) = (x + y, \; y + z, \; x + z)$

---

[<< Previous: Lesson 4 - Vector Norms and Inner Products](04_Vector_Norms_and_Inner_Products.md) | [Overview](00_Overview.md) | [Next: Lesson 6 - Eigenvalues and Eigenvectors >>](06_Eigenvalues_and_Eigenvectors.md)

**License**: CC BY-NC 4.0
