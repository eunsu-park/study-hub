# Lesson 2: Matrices and Operations

## Learning Objectives

- Define matrices and perform addition, scalar multiplication, and matrix multiplication in NumPy
- Compute the transpose, inverse, determinant, and trace of a matrix
- Recognize and construct special matrices: identity, diagonal, symmetric, skew-symmetric, orthogonal, and triangular
- Understand the rules governing matrix algebra (associativity, non-commutativity, etc.)
- Apply matrix operations to solve practical problems

---

## 1. Matrix Fundamentals

### 1.1 Definition

An $m \times n$ matrix $A$ is a rectangular array of real numbers arranged in $m$ rows and $n$ columns:

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

We write $a_{ij}$ or $(A)_{ij}$ for the entry in row $i$, column $j$.

```python
import numpy as np

# Creating matrices
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"A =\n{A}")
print(f"Shape: {A.shape}")      # (2, 3)
print(f"Element (1,2): {A[0, 1]}")  # 2 (0-indexed)

# Useful constructors
Z = np.zeros((3, 4))      # 3x4 zero matrix
O = np.ones((2, 3))       # 2x3 all-ones matrix
R = np.random.randn(3, 3) # 3x3 random matrix
print(f"Random 3x3:\n{R}")
```

### 1.2 Matrix Addition and Scalar Multiplication

Matrices of the **same shape** can be added element-wise, and any matrix can be multiplied by a scalar:

$$C = A + B \implies c_{ij} = a_{ij} + b_{ij}$$
$$D = \alpha A \implies d_{ij} = \alpha \, a_{ij}$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"A + B =\n{A + B}")
print(f"3A =\n{3 * A}")
print(f"A - B =\n{A - B}")
```

---

## 2. Matrix Multiplication

### 2.1 Definition

For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, the product $C = AB \in \mathbb{R}^{m \times p}$ is defined by:

$$c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

The number of **columns** of $A$ must equal the number of **rows** of $B$.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])     # 2x3
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])      # 3x2

C = A @ B   # preferred syntax; also np.matmul(A, B) or np.dot(A, B)
print(f"A @ B =\n{C}")       # 2x2
print(f"Shape: {C.shape}")

# Manual verification of C[0,0]
print(f"C[0,0] = 1*7 + 2*9 + 3*11 = {1*7 + 2*9 + 3*11}")
```

### 2.2 Properties of Matrix Multiplication

| Property | Statement |
|----------|-----------|
| Associativity | $(AB)C = A(BC)$ |
| Left distributivity | $A(B + C) = AB + AC$ |
| Right distributivity | $(A + B)C = AC + BC$ |
| Scalar compatibility | $\alpha(AB) = (\alpha A)B = A(\alpha B)$ |
| **Non-commutativity** | $AB \neq BA$ in general |

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"AB =\n{A @ B}")
print(f"BA =\n{B @ A}")
print(f"AB == BA? {np.allclose(A @ B, B @ A)}")  # False

# Associativity check
C = np.array([[1, 0], [0, 1]])
print(f"(AB)C == A(BC)? {np.allclose((A @ B) @ C, A @ (B @ C))}")  # True
```

### 2.3 Interpreting Matrix Multiplication

There are four equivalent ways to view $C = AB$:

1. **Row-column dot products**: $c_{ij} = (\text{row } i \text{ of } A) \cdot (\text{col } j \text{ of } B)$
2. **Column picture**: Each column of $C$ is a linear combination of columns of $A$
3. **Row picture**: Each row of $C$ is a linear combination of rows of $B$
4. **Outer product sum**: $C = \sum_k (\text{col } k \text{ of } A)(\text{row } k \text{ of } B)$

```python
# Column picture: column j of C is A times column j of B
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A @ B
print(f"Column 0 of C: {C[:, 0]}")
print(f"A @ col 0 of B: {A @ B[:, 0]}")  # same

# Outer product sum
outer_sum = np.outer(A[:, 0], B[0, :]) + np.outer(A[:, 1], B[1, :])
print(f"Outer product sum:\n{outer_sum}")
print(f"Matches C? {np.allclose(C, outer_sum)}")
```

### 2.4 Element-wise (Hadamard) Product

The element-wise product $A \odot B$ multiplies corresponding entries. This is **not** the same as matrix multiplication.

$$(\mathbf{A} \odot \mathbf{B})_{ij} = a_{ij} \, b_{ij}$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

hadamard = A * B   # element-wise in NumPy
matmul = A @ B     # matrix multiplication

print(f"Hadamard product:\n{hadamard}")
print(f"Matrix product:\n{matmul}")
```

---

## 3. Transpose

The **transpose** of $A \in \mathbb{R}^{m \times n}$ is $A^T \in \mathbb{R}^{n \times m}$ defined by:

$$(A^T)_{ij} = A_{ji}$$

### Properties

| Property | Statement |
|----------|-----------|
| Double transpose | $(A^T)^T = A$ |
| Sum | $(A + B)^T = A^T + B^T$ |
| Scalar | $(\alpha A)^T = \alpha A^T$ |
| Product | $(AB)^T = B^T A^T$ (reverse order!) |

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A_T = A.T
print(f"A =\n{A}")
print(f"A^T =\n{A_T}")
print(f"Shape: {A.shape} -> {A_T.shape}")

# Product transpose rule
B = np.array([[1, 0],
              [0, 1],
              [1, 1]])
lhs = (A @ B).T
rhs = B.T @ A.T
print(f"(AB)^T == B^T A^T? {np.allclose(lhs, rhs)}")
```

---

## 4. Determinant

### 4.1 Definition

The **determinant** is a scalar-valued function defined for square matrices. For a $2 \times 2$ matrix:

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

For larger matrices, the determinant can be computed via cofactor expansion or row reduction.

### 4.2 Geometric Interpretation

$|\det(A)|$ gives the **volume scaling factor** of the linear transformation represented by $A$:
- $|\det(A)| = 0$: the transformation collapses space (matrix is singular)
- $\det(A) < 0$: the transformation reverses orientation

### 4.3 Properties

| Property | Statement |
|----------|-----------|
| Product rule | $\det(AB) = \det(A) \det(B)$ |
| Transpose | $\det(A^T) = \det(A)$ |
| Inverse | $\det(A^{-1}) = 1 / \det(A)$ |
| Scalar | $\det(\alpha A) = \alpha^n \det(A)$ for $n \times n$ matrix |
| Triangular | determinant = product of diagonal entries |

```python
# 2x2 determinant
A = np.array([[3, 1],
              [2, 4]])
det_A = np.linalg.det(A)
print(f"det(A) = {det_A:.4f}")  # 3*4 - 1*2 = 10

# 3x3 determinant
B = np.array([[1, 2, 3],
              [0, 4, 5],
              [0, 0, 6]])
print(f"det(B) = {np.linalg.det(B):.4f}")  # 1*4*6 = 24 (triangular)

# Singular matrix
C = np.array([[1, 2],
              [2, 4]])
print(f"det(C) = {np.linalg.det(C):.10f}")  # ~0

# Product rule
print(f"det(A)*det(B[:2,:2]) = {det_A * np.linalg.det(B[:2,:2]):.4f}")
```

### 4.4 Visualizing the Determinant

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Unit square vertices
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

matrices = [
    (np.array([[2, 0], [0, 1]]), "det = 2 (stretch)"),
    (np.array([[1, 1], [0, 1]]), "det = 1 (shear)"),
    (np.array([[1, 2], [0.5, 1]]), "det = 0 (singular)"),
]

for ax, (M, title) in zip(axes, matrices):
    d = np.linalg.det(M)
    transformed = (M @ square.T).T

    ax.fill(*square.T, alpha=0.3, color='blue', label='Original')
    ax.fill(*transformed.T, alpha=0.3, color='red', label='Transformed')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"{title}\n|det| = {abs(d):.2f}")

plt.tight_layout()
plt.show()
```

---

## 5. Inverse Matrix

### 5.1 Definition

For a square matrix $A$, the **inverse** $A^{-1}$ satisfies:

$$AA^{-1} = A^{-1}A = I$$

An inverse exists if and only if $\det(A) \neq 0$ (the matrix is **non-singular** or **invertible**).

### 5.2 Properties

| Property | Statement |
|----------|-----------|
| Uniqueness | The inverse, if it exists, is unique |
| Involution | $(A^{-1})^{-1} = A$ |
| Product | $(AB)^{-1} = B^{-1}A^{-1}$ (reverse order!) |
| Transpose | $(A^T)^{-1} = (A^{-1})^T$ |

### 5.3 Computing the Inverse

For a $2 \times 2$ matrix:

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

```python
A = np.array([[4, 7],
              [2, 6]])

A_inv = np.linalg.inv(A)
print(f"A^(-1) =\n{A_inv}")
print(f"A @ A^(-1) =\n{A @ A_inv}")       # identity
print(f"A^(-1) @ A =\n{A_inv @ A}")       # identity

# Verify with 2x2 formula
det_A = np.linalg.det(A)
A_inv_formula = (1 / det_A) * np.array([[6, -7], [-2, 4]])
print(f"Formula matches? {np.allclose(A_inv, A_inv_formula)}")

# Singular matrix -- no inverse
B = np.array([[1, 2], [2, 4]])
try:
    np.linalg.inv(B)
except np.linalg.LinAlgError as e:
    print(f"Cannot invert singular matrix: {e}")
```

---

## 6. Trace

The **trace** of a square matrix is the sum of its diagonal entries:

$$\mathrm{tr}(A) = \sum_{i=1}^n a_{ii}$$

### Properties

| Property | Statement |
|----------|-----------|
| Linearity | $\mathrm{tr}(A + B) = \mathrm{tr}(A) + \mathrm{tr}(B)$ |
| Scalar | $\mathrm{tr}(\alpha A) = \alpha \, \mathrm{tr}(A)$ |
| Transpose | $\mathrm{tr}(A^T) = \mathrm{tr}(A)$ |
| Cyclic property | $\mathrm{tr}(ABC) = \mathrm{tr}(CAB) = \mathrm{tr}(BCA)$ |
| Eigenvalue connection | $\mathrm{tr}(A) = \sum_i \lambda_i$ |

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(f"tr(A) = {np.trace(A)}")  # 1 + 5 + 9 = 15

# Cyclic property
B = np.random.randn(3, 3)
C = np.random.randn(3, 3)
print(f"tr(ABC) = {np.trace(A @ B @ C):.6f}")
print(f"tr(CAB) = {np.trace(C @ A @ B):.6f}")
print(f"tr(BCA) = {np.trace(B @ C @ A):.6f}")

# Trace equals sum of eigenvalues
eigenvalues = np.linalg.eigvals(A)
print(f"Sum of eigenvalues = {np.sum(eigenvalues).real:.6f}")
print(f"tr(A) = {np.trace(A)}")
```

---

## 7. Special Matrices

### 7.1 Identity Matrix

The $n \times n$ identity matrix $I_n$ has ones on the diagonal and zeros elsewhere. It is the multiplicative identity: $AI = IA = A$.

```python
I3 = np.eye(3)
print(f"I_3 =\n{I3}")

A = np.random.randn(3, 3)
print(f"A @ I == A? {np.allclose(A @ I3, A)}")
```

### 7.2 Diagonal Matrix

A diagonal matrix $D$ has non-zero entries only on the main diagonal:

$$D = \mathrm{diag}(d_1, d_2, \ldots, d_n)$$

Diagonal matrices are easy to invert, exponentiate, and multiply.

```python
d = np.array([2, 3, 5])
D = np.diag(d)
print(f"D =\n{D}")

# Inverse of diagonal matrix
D_inv = np.diag(1.0 / d)
print(f"D^(-1) =\n{D_inv}")
print(f"D @ D^(-1) =\n{D @ D_inv}")

# Powers of diagonal matrix
D_cubed = np.diag(d**3)
print(f"D^3 =\n{D_cubed}")
```

### 7.3 Symmetric Matrix

A matrix is **symmetric** if $A = A^T$, meaning $a_{ij} = a_{ji}$.

Symmetric matrices have several important properties:
- All eigenvalues are real
- Eigenvectors corresponding to distinct eigenvalues are orthogonal
- They can be diagonalized by an orthogonal matrix

```python
# Creating a symmetric matrix
A = np.random.randn(3, 3)
S = (A + A.T) / 2   # guaranteed symmetric
print(f"S =\n{S}")
print(f"S == S^T? {np.allclose(S, S.T)}")

# Eigenvalues are real
eigenvalues = np.linalg.eigvals(S)
print(f"Eigenvalues: {eigenvalues}")
print(f"All real? {np.all(np.isreal(eigenvalues))}")
```

### 7.4 Skew-Symmetric Matrix

A matrix is **skew-symmetric** if $A = -A^T$. The diagonal entries of a skew-symmetric matrix are always zero.

```python
A = np.random.randn(3, 3)
K = (A - A.T) / 2   # guaranteed skew-symmetric
print(f"K =\n{np.round(K, 4)}")
print(f"K == -K^T? {np.allclose(K, -K.T)}")
print(f"Diagonal: {np.diag(K)}")  # all ~0
```

### 7.5 Orthogonal Matrix

A square matrix $Q$ is **orthogonal** if $Q^T Q = Q Q^T = I$. Equivalently, $Q^{-1} = Q^T$.

Orthogonal matrices preserve lengths and angles -- they represent rotations and reflections.

```python
# 2D rotation matrix
theta = np.pi / 4  # 45 degrees
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print(f"Q =\n{Q}")
print(f"Q^T @ Q =\n{np.round(Q.T @ Q, 10)}")   # identity
print(f"det(Q) = {np.linalg.det(Q):.6f}")        # +1 (rotation)

# Orthogonal matrices preserve vector norms
v = np.array([3, 4])
Qv = Q @ v
print(f"||v|| = {np.linalg.norm(v):.4f}")
print(f"||Qv|| = {np.linalg.norm(Qv):.4f}")  # same
```

### 7.6 Triangular Matrices

An **upper triangular** matrix has all entries below the diagonal equal to zero. A **lower triangular** matrix has all entries above the diagonal equal to zero.

$$U = \begin{bmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{bmatrix}, \quad L = \begin{bmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{bmatrix}$$

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

U = np.triu(A)  # upper triangular
L = np.tril(A)  # lower triangular

print(f"Upper triangular:\n{U}")
print(f"Lower triangular:\n{L}")

# Determinant of triangular matrix = product of diagonal
print(f"det(U) = {np.linalg.det(U):.4f}")
print(f"Product of diagonal: {np.prod(np.diag(U))}")
```

---

## 8. Matrix Rank

The **rank** of a matrix is the maximum number of linearly independent rows (or equivalently, columns).

$$\mathrm{rank}(A) \le \min(m, n) \quad \text{for } A \in \mathbb{R}^{m \times n}$$

A matrix has **full rank** when $\mathrm{rank}(A) = \min(m, n)$.

```python
# Full rank matrix
A = np.array([[1, 2],
              [3, 4]])
print(f"rank(A) = {np.linalg.matrix_rank(A)}")  # 2

# Rank-deficient matrix
B = np.array([[1, 2, 3],
              [4, 5, 6],
              [5, 7, 9]])  # row 3 = row 1 + row 2
print(f"rank(B) = {np.linalg.matrix_rank(B)}")  # 2

# Rank properties
C = np.random.randn(3, 5)
print(f"C is 3x5, rank(C) = {np.linalg.matrix_rank(C)}")  # at most 3
print(f"rank(C) = rank(C^T)? {np.linalg.matrix_rank(C) == np.linalg.matrix_rank(C.T)}")
```

---

## 9. Block Matrices

Matrices can be partitioned into **blocks** (sub-matrices). Block multiplication follows the same rules as scalar multiplication, provided the block sizes are compatible.

$$\begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix} \begin{bmatrix} B_{11} \\ B_{21} \end{bmatrix} = \begin{bmatrix} A_{11}B_{11} + A_{12}B_{21} \\ A_{21}B_{11} + A_{22}B_{21} \end{bmatrix}$$

```python
# Block matrix construction
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
Z = np.zeros((2, 2))
I = np.eye(2)

# Build a 4x4 block matrix
M = np.block([[A, B],
              [Z, I]])
print(f"Block matrix M =\n{M}")
print(f"Shape: {M.shape}")
```

---

## 10. Summary

| Operation | Notation | NumPy |
|-----------|----------|-------|
| Matrix multiplication | $AB$ | `A @ B` |
| Element-wise product | $A \odot B$ | `A * B` |
| Transpose | $A^T$ | `A.T` |
| Inverse | $A^{-1}$ | `np.linalg.inv(A)` |
| Determinant | $\det(A)$ | `np.linalg.det(A)` |
| Trace | $\mathrm{tr}(A)$ | `np.trace(A)` |
| Rank | $\mathrm{rank}(A)$ | `np.linalg.matrix_rank(A)` |
| Identity | $I_n$ | `np.eye(n)` |
| Diagonal | $\mathrm{diag}(d)$ | `np.diag(d)` |

---

## Exercises

### Exercise 1: Matrix Arithmetic

Let $A = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$ and $B = \begin{bmatrix} 5 & 7 \\ 6 & 8 \end{bmatrix}$.

Compute $AB$, $BA$, $A^T B$, and verify that $AB \neq BA$.

### Exercise 2: Determinant and Invertibility

For the matrix $M = \begin{bmatrix} 2 & 1 & 0 \\ 1 & 3 & 1 \\ 0 & 1 & 2 \end{bmatrix}$:

1. Compute $\det(M)$.
2. Determine whether $M$ is invertible.
3. If invertible, compute $M^{-1}$ and verify $M M^{-1} = I$.

### Exercise 3: Special Matrices

1. Show that any square matrix $A$ can be written as $A = S + K$ where $S$ is symmetric and $K$ is skew-symmetric.
2. Verify this decomposition numerically for a random $4 \times 4$ matrix.

### Exercise 4: Trace Properties

For random $3 \times 3$ matrices $A$, $B$, $C$, numerically verify:

1. $\mathrm{tr}(AB) = \mathrm{tr}(BA)$
2. $\mathrm{tr}(ABC) = \mathrm{tr}(BCA) = \mathrm{tr}(CAB)$
3. $\mathrm{tr}(A)$ equals the sum of eigenvalues of $A$

### Exercise 5: Rank Exploration

Construct matrices with the following properties and verify using NumPy:

1. A $3 \times 3$ matrix with rank 1
2. A $4 \times 2$ matrix with rank 2
3. A $3 \times 3$ matrix with rank 2 whose determinant is 0

---

[<< Previous: Lesson 1 - Vectors and Vector Spaces](01_Vectors_and_Vector_Spaces.md) | [Overview](00_Overview.md) | [Next: Lesson 3 - Systems of Linear Equations >>](03_Systems_of_Linear_Equations.md)

**License**: CC BY-NC 4.0
