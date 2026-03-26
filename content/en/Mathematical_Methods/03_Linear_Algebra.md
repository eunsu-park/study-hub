# 03. Linear Algebra

> **Boas Chapter 3** — In physical sciences, linear algebra forms the foundation for nearly every field, including inertia tensors in mechanics, matrix formalism in quantum mechanics, and coupled oscillation analysis.

---

## Learning Objectives

After completing this lesson, you will be able to:

- Perform **matrix operations** (addition, multiplication, transpose, inverse) and calculate determinants
- Solve systems of linear equations using **Gaussian elimination** and **Cramer's rule**
- Find **eigenvalues/eigenvectors** and perform matrix diagonalization
- Understand the spectral theorem for **symmetric and Hermitian matrices**, and utilize properties of orthogonal/unitary matrices
- Determine the definiteness (positive/negative definite) of **quadratic forms**
- **Physics applications**: Handle principal axis transformation of inertia tensors, normal modes of coupled oscillations, and matrix mechanics in quantum mechanics

---

## 1. Matrix Fundamentals

### 1.1 Matrices and Basic Operations

A **matrix** is a rectangular array of numbers. For an $m \times n$ matrix $A$, elements are denoted as $a_{ij}$:

$$
A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}
$$

**Matrix addition**: Element-wise addition for matrices of the same size

$$
(A + B)_{ij} = a_{ij} + b_{ij}
$$

**Scalar multiplication**: Multiply all elements by a scalar

$$
(cA)_{ij} = c \cdot a_{ij}
$$

**Matrix multiplication**: If $A$ is $m \times n$ and $B$ is $n \times p$, the result is $m \times p$

$$
(AB)_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}
$$

> **Note**: Matrix multiplication generally does not commute: $AB \neq BA$

```python
import numpy as np

# Basic matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A =\n", A)
print("B =\n", B)

# Matrix multiplication
print("\nAB =\n", A @ B)
print("BA =\n", B @ A)
print("AB ≠ BA:", not np.allclose(A @ B, B @ A))

# Properties of matrix multiplication
C = np.array([[1, 0], [2, 3]])
print("\n(AB)C =\n", (A @ B) @ C)
print("A(BC) =\n", A @ (B @ C))
print("Associativity holds:", np.allclose((A @ B) @ C, A @ (B @ C)))
```

### 1.2 Transpose and Conjugate Transpose

**Transpose** $A^T$: Exchange rows and columns

$$
(A^T)_{ij} = a_{ji}
$$

**Properties**:
- $(AB)^T = B^T A^T$ (order reversed)
- $(A^T)^T = A$
- $(A + B)^T = A^T + B^T$

**Conjugate transpose** (adjoint) $A^\dagger$: Transpose + complex conjugate

$$
(A^\dagger)_{ij} = \overline{a_{ji}}
$$

```python
# Transpose matrix
A = np.array([[1, 2, 3], [4, 5, 6]])
print("A =\n", A)
print("A^T =\n", A.T)

# Conjugate transpose of a complex matrix
Z = np.array([[1+2j, 3-1j], [4j, 2+3j]])
print("\nZ =\n", Z)
print("Z† =\n", Z.conj().T)

# Verify (AB)^T = B^T A^T
B = np.array([[1, 2], [3, 4], [5, 6]])
print("\n(AB)^T =\n", (A @ B).T)
print("B^T A^T =\n", B.T @ A.T)
```

### 1.3 Special Matrices

**Identity matrix** $I$: Diagonal elements are all 1

$$
I_{ij} = \delta_{ij} = \begin{cases} 1 & (i = j) \\ 0 & (i \neq j) \end{cases}
$$

**Diagonal matrix**: Matrix with only diagonal elements non-zero

**Symmetric matrix**: $A^T = A$, i.e., $a_{ij} = a_{ji}$

**Antisymmetric matrix**: $A^T = -A$

**Hermitian matrix**: $A^\dagger = A$ (complex extension of symmetric matrices)

**Orthogonal matrix**: $A^T A = AA^T = I$, i.e., $A^{-1} = A^T$

**Unitary matrix**: $A^\dagger A = AA^\dagger = I$ (complex extension of orthogonal matrices)

```python
# Orthogonal matrix example: 2D rotation matrix
theta = np.pi / 4  # 45-degree rotation
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print("R =\n", R)
print("R^T R =\n", np.round(R.T @ R, 10))
print("det(R) =", np.linalg.det(R))  # det = +1 (proper rotation)

# Hermitian matrix example
H = np.array([[2, 1-1j], [1+1j, 3]])
print("\nH =\n", H)
print("H† =\n", H.conj().T)
print("Hermitian:", np.allclose(H, H.conj().T))
```

---

## 2. Determinants

### 2.1 Definition and Basic Properties

The **determinant** $\det(A)$ of an $n \times n$ square matrix $A$ is a scalar value.

**2×2 determinant**:

$$
\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc
$$

**3×3 determinant** (Sarrus's rule or cofactor expansion):

$$
\det\begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}
= a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})
$$

**Key properties**:

1. $\det(AB) = \det(A) \cdot \det(B)$
2. $\det(A^T) = \det(A)$
3. $\det(cA) = c^n \det(A)$ ($n \times n$ matrix)
4. Exchanging rows (or columns) changes the sign
5. If one row is a constant multiple of another, $\det = 0$
6. $\det(A^{-1}) = 1/\det(A)$

### 2.2 Cofactor Expansion

**Minor** $M_{ij}$: Determinant of the $(n-1) \times (n-1)$ matrix obtained by deleting the $i$-th row and $j$-th column

**Cofactor**: $C_{ij} = (-1)^{i+j} M_{ij}$

**Cofactor expansion of determinant** (expanding along the $i$-th row):

$$
\det(A) = \sum_{j=1}^{n} a_{ij} C_{ij} = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} M_{ij}
$$

```python
import numpy as np
from numpy.linalg import det

# Determinant computation
A = np.array([[2, 1, 3],
              [0, -1, 2],
              [4, 3, 1]])

print("det(A) =", det(A))

# Manual cofactor expansion (along row 1)
# det = 2*(-1-6) - 1*(0-8) + 3*(0+4)
manual = 2*(-1*1 - 2*3) - 1*(0*1 - 2*4) + 3*(0*3 - (-1)*4)
print("Manual computation:", manual)

# Verify properties of the determinant
B = np.array([[1, 2, 0],
              [3, 1, -1],
              [2, 0, 4]])

print(f"\ndet(A) = {det(A):.4f}")
print(f"det(B) = {det(B):.4f}")
print(f"det(AB) = {det(A @ B):.4f}")
print(f"det(A)*det(B) = {det(A)*det(B):.4f}")
```

### 2.3 Geometric Interpretation of Determinants

The absolute value $|\det(A)|$ of the determinant of an $n \times n$ matrix $A$ is the **volume of the parallelepiped** formed by the row vectors (or column vectors).

- 2D: $|\det(A)|$ = area of the parallelogram formed by two vectors
- 3D: $|\det(A)|$ = volume of the parallelepiped formed by three vectors

The **sign** of $\det(A)$ indicates orientation:
- $\det > 0$: Right-handed coordinate system preserved
- $\det < 0$: Coordinate system inverted (includes mirror reflection)

```python
import matplotlib.pyplot as plt

# Determinant = area in 2D
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before transformation: unit square
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T

A = np.array([[2, 1], [0, 3]])  # det = 6

# After transformation
transformed = A @ square

for ax, shape, title in [(axes[0], square, 'Unit square (area=1)'),
                          (axes[1], transformed, f'After transform (area=|det|={abs(det(A)):.0f})')]:
    ax.fill(shape[0], shape[1], alpha=0.3, color='blue')
    ax.plot(shape[0], shape[1], 'b-', linewidth=2)
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('determinant_area.png', dpi=100, bbox_inches='tight')
plt.close()
print(f"det(A) = {det(A):.0f}: area scaled by {abs(det(A)):.0f}x")
```

---

## 3. Inverse Matrices and Systems of Linear Equations

### 3.1 Inverse Matrix

The **inverse matrix** $A^{-1}$ of a square matrix $A$ satisfies:

$$
AA^{-1} = A^{-1}A = I
$$

**Necessary and sufficient condition** for the inverse to exist: $\det(A) \neq 0$ (non-singular matrix)

**Inverse using cofactors**:

$$
A^{-1} = \frac{1}{\det(A)} \text{adj}(A)
$$

where $\text{adj}(A)$ is the **adjugate matrix**, the transpose of the cofactor matrix: $\text{adj}(A)_{ij} = C_{ji}$

**2×2 inverse**:

$$
\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1}
= \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$

```python
A = np.array([[2, 1], [5, 3]])
A_inv = np.linalg.inv(A)

print("A =\n", A)
print("A^{-1} =\n", A_inv)
print("A @ A^{-1} =\n", np.round(A @ A_inv, 10))

# Manual computation using the adjugate matrix
d = det(A)  # 2*3 - 1*5 = 1
adj_A = np.array([[3, -1], [-5, 2]])  # transpose of cofactor matrix
manual_inv = adj_A / d
print("\nManual computation:\n", manual_inv)
```

### 3.2 Gaussian Elimination

Solve the system of linear equations $A\mathbf{x} = \mathbf{b}$ by performing row operations on the **augmented matrix** $(A | \mathbf{b})$.

**Allowed elementary row operations**:
1. Exchange two rows
2. Multiply a row by a non-zero constant
3. Add a constant multiple of one row to another row

**Goal**: Transform to upper triangular form (or reduced row echelon form) followed by back substitution

```python
def gauss_eliminate(A_aug, verbose=True):
    """Solve a system of equations by Gaussian elimination.

    A_aug: augmented matrix [A | b]
    """
    A = A_aug.astype(float).copy()
    n = A.shape[0]

    # Forward elimination
    for col in range(n):
        # Pivot selection (partial pivoting)
        max_row = col + np.argmax(np.abs(A[col:, col]))
        if max_row != col:
            A[[col, max_row]] = A[[max_row, col]]

        if abs(A[col, col]) < 1e-12:
            print(f"Warning: pivot is near zero (column {col})")
            continue

        # Eliminate rows below
        for row in range(col + 1, n):
            factor = A[row, col] / A[col, col]
            A[row] -= factor * A[col]

    if verbose:
        print("Upper triangular matrix:\n", np.round(A, 4))

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (A[i, -1] - A[i, i+1:n] @ x[i+1:n]) / A[i, i]

    return x

# Example: 3x3 system of equations
# 2x + y - z = 8
# -3x - y + 2z = -11
# -2x + y + 2z = -3
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

A_aug = np.column_stack([A, b])
x = gauss_eliminate(A_aug)
print(f"\nSolution: x = {x}")
print(f"Verify Ax = {A @ x}")

# Compare with NumPy built-in
x_np = np.linalg.solve(A, b)
print(f"NumPy solve: {x_np}")
```

### 3.3 Cramer's Rule

For a system of linear equations $A\mathbf{x} = \mathbf{b}$, if $\det(A) \neq 0$, each unknown is:

$$
x_i = \frac{\det(A_i)}{\det(A)}
$$

where $A_i$ is the matrix obtained by replacing the $i$-th column of $A$ with $\mathbf{b}$.

> **Note**: Cramer's rule is theoretically important but computationally less efficient than Gaussian elimination. For large $n$, Gaussian elimination ($O(n^3)$) is much faster than Cramer's rule ($O(n \cdot n!)$).

```python
def cramer(A, b):
    """Solve Ax = b using Cramer's rule."""
    n = len(b)
    d = det(A)
    if abs(d) < 1e-12:
        raise ValueError("Determinant is 0: no unique solution exists")

    x = np.zeros(n)
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        x[i] = det(A_i) / d
    return x

# Example
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

x = cramer(A, b)
print(f"Cramer's rule solution: {x}")
```

### 3.4 Rank of a Matrix

The **rank** of a matrix is the maximum number of linearly independent rows (or columns).

$$
\text{rank}(A) = \text{dim}(\text{Col}(A)) = \text{dim}(\text{Row}(A))
$$

**Existence of solutions for systems of linear equations**:
- $\text{rank}(A) = \text{rank}(A|b) = n$: Unique solution
- $\text{rank}(A) = \text{rank}(A|b) < n$: Infinitely many solutions
- $\text{rank}(A) < \text{rank}(A|b)$: No solution (inconsistent)

```python
# Rank examples
A_full = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])  # rank 3
A_deficient = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # rank 2 (row3 != row1 + 2*row2, but 1+3=2*2)

print(f"rank(A_full) = {np.linalg.matrix_rank(A_full)}")
print(f"rank(A_deficient) = {np.linalg.matrix_rank(A_deficient)}")
print(f"det(A_full) = {det(A_full):.4f}")
print(f"det(A_deficient) = {det(A_deficient):.4f}")
```

---

## 4. Eigenvalues and Eigenvectors

### 4.1 Definition and Characteristic Equation

For a square matrix $A$, find a non-zero vector $\mathbf{v}$ and scalar $\lambda$ that satisfy:

$$
A\mathbf{v} = \lambda\mathbf{v}
$$

- $\lambda$: **eigenvalue**
- $\mathbf{v}$: **eigenvector**

Geometric interpretation: Under the linear transformation by $A$, an eigenvector is a special vector whose **direction remains unchanged**. The eigenvalue is the scaling factor in that direction.

**Characteristic equation**:

$$
\det(A - \lambda I) = 0
$$

This is an $n$-th degree polynomial in $\lambda$, called the **characteristic polynomial**.

```python
import numpy as np
from numpy.linalg import eig

# Eigenvalues/eigenvectors of a 2x2 matrix
A = np.array([[4, 2], [1, 3]])

eigenvalues, eigenvectors = eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors (as columns):\n", eigenvectors)

# Verify: Av = λv
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    Av = A @ v
    lam_v = lam * v
    print(f"\nλ_{i+1} = {lam:.4f}")
    print(f"v_{i+1} = {v}")
    print(f"Av = {Av}")
    print(f"λv = {lam_v}")
    print(f"Av ≈ λv: {np.allclose(Av, lam_v)}")
```

### 4.2 Manual Calculation Example

Find eigenvalues/eigenvectors of $A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$:

**Characteristic equation**:
$$
\det(A - \lambda I) = \det\begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix} = (3-\lambda)(2-\lambda) = 0
$$

$\lambda_1 = 3$, $\lambda_2 = 2$

**Eigenvector** ($\lambda_1 = 3$):
$$
(A - 3I)\mathbf{v} = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}\mathbf{v} = 0 \implies v_2 = 0 \implies \mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}
$$

**Eigenvector** ($\lambda_2 = 2$):
$$
(A - 2I)\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\mathbf{v} = 0 \implies v_1 + v_2 = 0 \implies \mathbf{v}_2 = \begin{pmatrix} -1 \\ 1 \end{pmatrix}
$$

```python
import sympy as sp

# Analytic computation using SymPy
A_sym = sp.Matrix([[3, 1], [0, 2]])
print("Characteristic polynomial:", A_sym.charpoly().as_expr())
print("Eigenvalues:", A_sym.eigenvals())  # {3: 1, 2: 1} (eigenvalue: multiplicity)
print("Eigenvectors:", A_sym.eigenvects())
```

### 4.3 Matrix Diagonalization

An $n \times n$ matrix $A$ is **diagonalizable** if it has $n$ linearly independent eigenvectors.

Construct a matrix $P = [\mathbf{v}_1 | \mathbf{v}_2 | \cdots | \mathbf{v}_n]$ with eigenvectors as columns:

$$
P^{-1}AP = D = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)
$$

**Usefulness of diagonalization**:

- **Powers**: $A^k = P D^k P^{-1}$, where $D^k$ is computed by raising diagonal elements to the $k$-th power
- **Exponential**: $e^{At} = P \, e^{Dt} \, P^{-1}$, where $e^{Dt} = \text{diag}(e^{\lambda_1 t}, \ldots, e^{\lambda_n t})$
- **Coupled ODEs**: Decouple $\dot{\mathbf{x}} = A\mathbf{x}$ into independent scalar ODEs

```python
# Diagonalization example
A = np.array([[4, 2], [1, 3]])
eigenvalues, P = eig(A)
D = np.diag(eigenvalues)

print("P (eigenvector matrix):\n", P)
print("D (diagonal matrix):\n", D)

# Verify: A = P D P^{-1}
P_inv = np.linalg.inv(P)
A_reconstructed = P @ D @ P_inv
print("\nP D P^{-1} =\n", np.round(A_reconstructed, 10))
print("Matches A:", np.allclose(A, A_reconstructed))

# Application: compute A^10
A_power_10 = P @ np.diag(eigenvalues**10) @ P_inv
print(f"\nA^10 =\n{np.round(A_power_10.real, 2)}")
print(f"Direct computation:\n{np.round(np.linalg.matrix_power(A, 10).astype(float), 2)}")
```

### 4.4 Properties of Symmetric and Hermitian Matrices

**Spectral Theorem**:

1. **Real symmetric matrices** ($A^T = A$):
   - All eigenvalues are **real**
   - Eigenvectors corresponding to distinct eigenvalues are **orthogonal**
   - Diagonalizable by an orthogonal matrix $Q$: $Q^T A Q = D$, $Q^T = Q^{-1}$

2. **Hermitian matrices** ($A^\dagger = A$):
   - All eigenvalues are **real**
   - Eigenvectors corresponding to distinct eigenvalues are **orthogonal**
   - Diagonalizable by a unitary matrix $U$: $U^\dagger A U = D$, $U^\dagger = U^{-1}$

```python
# Properties of real symmetric matrices
A_sym = np.array([[4, 1, 2],
                   [1, 3, 1],
                   [2, 1, 5]])

eigenvalues, eigenvectors = eig(A_sym)
print("Eigenvalues of symmetric matrix:", eigenvalues)
print("All real:", np.allclose(eigenvalues.imag, 0))

# Verify orthogonality of eigenvectors
print("\nEigenvector dot products:")
for i in range(3):
    for j in range(i+1, 3):
        dot = eigenvectors[:, i] @ eigenvectors[:, j]
        print(f"  v{i+1} · v{j+1} = {dot:.6e}")

# Diagonalization with orthogonal matrix
Q = eigenvectors
print(f"\nQ^T Q =\n{np.round(Q.T @ Q, 10)}")
print("Q is orthogonal:", np.allclose(Q.T @ Q, np.eye(3)))
```

### 4.5 Summary of Major Matrix Decompositions

| Decomposition | Form | Condition | Physics Application |
|--------|------|------|------------|
| Eigenvalue decomposition | $A = PDP^{-1}$ | $n$ independent eigenvectors | Oscillation modes, stability |
| Orthogonal diagonalization | $A = QDQ^T$ | Symmetric matrix | Principal axis theorem, inertia tensor |
| SVD | $A = U\Sigma V^T$ | Any $m \times n$ | Data reduction, least squares |
| LU decomposition | $A = LU$ | Square matrix | Efficient solution of linear systems |
| Cholesky | $A = LL^T$ | Positive definite symmetric | Statistics, optimization |

---

## 5. Quadratic Forms and Definiteness

### 5.1 Definition of Quadratic Forms

A **quadratic form** for a vector $\mathbf{x} \in \mathbb{R}^n$:

$$
Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i,j} a_{ij} x_i x_j
$$

where $A$ can always be treated as a symmetric matrix (the antisymmetric part does not contribute to the quadratic form, so replace $A \to (A + A^T)/2$).

**2-variable example**:

$$
Q(x, y) = \begin{pmatrix} x & y \end{pmatrix} \begin{pmatrix} a & b \\ b & c \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = ax^2 + 2bxy + cy^2
$$

### 5.2 Determining Definiteness

**Definiteness** of a symmetric matrix $A$:

| Classification | Condition | Eigenvalue Condition |
|------|------|------------|
| Positive definite | $\mathbf{x}^T A \mathbf{x} > 0$ (for all $\mathbf{x} \neq 0$) | All $\lambda_i > 0$ |
| Positive semidefinite | $\mathbf{x}^T A \mathbf{x} \geq 0$ | All $\lambda_i \geq 0$ |
| Negative definite | $\mathbf{x}^T A \mathbf{x} < 0$ (for all $\mathbf{x} \neq 0$) | All $\lambda_i < 0$ |
| Indefinite | Can be both positive and negative | Mixed positive/negative eigenvalues |

**Sylvester's criterion**: A symmetric matrix is positive definite if and only if all **leading principal minors** are positive:

$$
a_{11} > 0, \quad \det\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix} > 0, \quad \det\begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix} > 0, \quad \ldots
$$

```python
def check_definiteness(A):
    """Determine the definiteness of a symmetric matrix."""
    eigenvalues = np.linalg.eigvalsh(A)  # specialized for symmetric matrices
    print(f"Eigenvalues: {eigenvalues}")

    if all(eigenvalues > 0):
        return "positive definite"
    elif all(eigenvalues >= 0):
        return "positive semidefinite"
    elif all(eigenvalues < 0):
        return "negative definite"
    elif all(eigenvalues <= 0):
        return "negative semidefinite"
    else:
        return "indefinite"

# Examples
matrices = {
    "positive definite": np.array([[2, -1], [-1, 2]]),
    "indefinite": np.array([[1, 3], [3, 1]]),
    "positive semidefinite": np.array([[1, 1], [1, 1]]),
}

for name, M in matrices.items():
    result = check_definiteness(M)
    print(f"{name} matrix: {result}\n")
```

### 5.3 Principal Axis Theorem

When a quadratic form is transformed to **principal axis coordinates**, cross terms disappear.

If a symmetric matrix $A = Q D Q^T$:

$$
Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \mathbf{y}^T D \mathbf{y} = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \cdots + \lambda_n y_n^2
$$

where $\mathbf{y} = Q^T \mathbf{x}$ (principal axis coordinates).

> This is directly applied in physics to **principal axis transformation of inertia tensors**, **principal stress directions of stress tensors**, etc.

```python
import matplotlib.pyplot as plt

# Visualization of quadratic surface and principal axis transformation
A = np.array([[5, 2], [2, 2]])
eigenvalues, Q = np.linalg.eigh(A)

print(f"Original quadratic form: 5x² + 4xy + 2y²")
print(f"Eigenvalues: λ₁={eigenvalues[0]:.2f}, λ₂={eigenvalues[1]:.2f}")
print(f"Principal axis coordinates: {eigenvalues[0]:.2f}y₁² + {eigenvalues[1]:.2f}y₂²")

# Ellipse visualization
theta = np.linspace(0, 2*np.pi, 200)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Ellipse in original coordinates (5x² + 4xy + 2y² = 1)
# Using parametric representation
t = np.linspace(0, 2*np.pi, 200)
# Ellipse in principal axis coordinates: λ₁y₁² + λ₂y₂² = 1
y1 = np.cos(t) / np.sqrt(eigenvalues[0])
y2 = np.sin(t) / np.sqrt(eigenvalues[1])
# Transform back to original coordinates: x = Q y
xy = Q @ np.array([y1, y2])

axes[0].plot(xy[0], xy[1], 'b-', linewidth=2)
# Principal axis directions
for i in range(2):
    v = Q[:, i] / np.sqrt(eigenvalues[i])
    axes[0].arrow(0, 0, v[0], v[1], head_width=0.03, color=['red', 'green'][i],
                  linewidth=2, label=f'Principal axis {i+1} (λ={eigenvalues[i]:.2f})')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].set_title('Original coordinates: 5x² + 4xy + 2y² = 1')

# Ellipse in principal axis coordinates (axis-aligned)
y1 = np.cos(t) / np.sqrt(eigenvalues[0])
y2 = np.sin(t) / np.sqrt(eigenvalues[1])
axes[1].plot(y1, y2, 'b-', linewidth=2)
axes[1].axhline(y=0, color='green', linewidth=1.5, label=f'y₂-axis (λ={eigenvalues[1]:.2f})')
axes[1].axvline(x=0, color='red', linewidth=1.5, label=f'y₁-axis (λ={eigenvalues[0]:.2f})')
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].set_title(f'Principal axis coordinates: {eigenvalues[0]:.2f}y₁² + {eigenvalues[1]:.2f}y₂² = 1')

plt.tight_layout()
plt.savefig('principal_axes.png', dpi=100, bbox_inches='tight')
plt.close()
```

---

## 6. Physics Applications

### 6.1 Moment of Inertia Tensor and Principal Axes

The **moment of inertia tensor** of a rigid body is a $3 \times 3$ symmetric matrix:

$$
I = \begin{pmatrix} I_{xx} & -I_{xy} & -I_{xz} \\ -I_{yx} & I_{yy} & -I_{yz} \\ -I_{zx} & -I_{zy} & I_{zz} \end{pmatrix}
$$

where:
- $I_{xx} = \sum m_i (y_i^2 + z_i^2)$ (moment of inertia)
- $I_{xy} = \sum m_i x_i y_i$ (product of inertia)

**Principal axis transformation**: Diagonalizing $I$ yields the **principal moments of inertia** $I_1, I_2, I_3$ and **principal axes**.

$$
Q^T I Q = \text{diag}(I_1, I_2, I_3)
$$

In the principal axis coordinate system, angular momentum simplifies to $L_i = I_i \omega_i$.

```python
# Inertia tensor example: masses at vertices of a cube
# Cube vertex coordinates (side length 2, center at origin)
vertices = np.array([
    [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
    [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1]
], dtype=float)

m = 1.0  # mass at each vertex

# Compute inertia tensor
I_tensor = np.zeros((3, 3))
for r in vertices:
    I_tensor += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

print("Inertia tensor:\n", I_tensor)

# Diagonalization → principal moments of inertia
eigenvalues, eigenvectors = np.linalg.eigh(I_tensor)
print(f"\nPrincipal moments: I₁={eigenvalues[0]:.2f}, I₂={eigenvalues[1]:.2f}, I₃={eigenvalues[2]:.2f}")
print("Principal axes:\n", eigenvectors)
```

### 6.2 Coupled Oscillations

A system with two masses connected by springs:

$$
M\ddot{\mathbf{x}} = -K\mathbf{x}
$$

where $M$ is the mass matrix and $K$ is the stiffness matrix.

**Normal frequencies** and **normal modes** form a generalized eigenvalue problem:

$$
K\mathbf{v} = \omega^2 M\mathbf{v}
$$

```python
# Coupled oscillation example: two mass-spring system
# m₁ = m₂ = m, spring constants k₁ = k₂ = k₃ = k
# Equations of motion: m*x₁'' = -k*x₁ + k*(x₂-x₁) = -2k*x₁ + k*x₂
#                      m*x₂'' = -k*(x₂-x₁) - k*x₂ = k*x₁ - 2k*x₂

m, k = 1.0, 1.0

M_mat = m * np.eye(2)
K_mat = np.array([[2*k, -k],
                   [-k, 2*k]])

# Generalized eigenvalue problem: K v = ω² M v
# Since M = mI, this becomes (1/m)K v = ω² v
eigenvalues, eigenvectors = np.linalg.eigh(K_mat / m)
omega = np.sqrt(eigenvalues)

print("Stiffness matrix K:\n", K_mat)
print(f"\nNatural frequencies: ω₁ = {omega[0]:.4f}, ω₂ = {omega[1]:.4f}")
print(f"Periods: T₁ = {2*np.pi/omega[0]:.4f}, T₂ = {2*np.pi/omega[1]:.4f}")
print(f"\nMode 1 (in-phase): {eigenvectors[:, 0]}")
print(f"Mode 2 (out-of-phase): {eigenvectors[:, 1]}")

# Time evolution visualization
t = np.linspace(0, 20, 500)
# Initial conditions: x₁(0) = 1, x₂(0) = 0, zero velocity
# Decompose into modal coordinates
eta = eigenvectors.T @ np.array([1, 0])
x = np.zeros((2, len(t)))
for i in range(2):
    x += np.outer(eigenvectors[:, i], eta[i] * np.cos(omega[i] * t))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, x[0], label='x₁(t)', linewidth=1.5)
ax.plot(t, x[1], label='x₂(t)', linewidth=1.5)
ax.set_xlabel('Time t')
ax.set_ylabel('Displacement')
ax.set_title('Coupled oscillations: beat phenomenon')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('coupled_oscillations.png', dpi=100, bbox_inches='tight')
plt.close()
```

### 6.3 Matrices in Quantum Mechanics

In quantum mechanics, observables are represented as **Hermitian operators**, which in finite dimensions are Hermitian matrices.

**Pauli matrices for spin-1/2 particles**:

$$
\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

These are Hermitian ($\sigma_i^\dagger = \sigma_i$), and each has eigenvalues $\pm 1$.

```python
# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

for name, sigma in [('σ_x', sigma_x), ('σ_y', sigma_y), ('σ_z', sigma_z)]:
    vals, vecs = np.linalg.eigh(sigma)
    print(f"{name}:")
    print(f"  Hermitian: {np.allclose(sigma, sigma.conj().T)}")
    print(f"  Eigenvalues: {vals}")
    print(f"  tr(σ) = {np.trace(sigma):.0f}, det(σ) = {np.linalg.det(sigma):.0f}")
    print()

# Commutation relations: [σ_i, σ_j] = 2i ε_ijk σ_k
comm_xy = sigma_x @ sigma_y - sigma_y @ sigma_x
print("[σ_x, σ_y] = 2i σ_z:")
print(np.round(comm_xy, 10))
print("2i σ_z:")
print(2j * sigma_z)
print("Matches:", np.allclose(comm_xy, 2j * sigma_z))
```

---

## 7. Linear Algebra with Python

### 7.1 Using NumPy and SciPy

```python
import numpy as np
from scipy import linalg

# ===== Basic operations =====
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])

# Determinant and inverse
print(f"det(A) = {np.linalg.det(A):.4f}")
print(f"A^(-1) =\n{np.linalg.inv(A)}")

# Eigenvalue decomposition
vals, vecs = np.linalg.eig(A)
print(f"\nEigenvalues: {vals}")

# ===== Various matrix decompositions =====

# LU decomposition
P, L, U = linalg.lu(A)
print(f"\nLU decomposition:")
print(f"P =\n{P}")
print(f"L =\n{np.round(L, 4)}")
print(f"U =\n{np.round(U, 4)}")
print(f"PLU = A: {np.allclose(P @ L @ U, A)}")

# SVD (Singular Value Decomposition)
U_svd, s, Vt = np.linalg.svd(A)
print(f"\nSingular values: {s}")

# QR decomposition
Q, R = np.linalg.qr(A)
print(f"\nQR decomposition: Q orthogonality {np.allclose(Q.T @ Q, np.eye(3))}")

# Solving a system of equations
b = np.array([1, 2, 3])
x = np.linalg.solve(A, b)
print(f"\nSolution of Ax = b: {x}")
print(f"Verify: Ax = {A @ x}")
```

### 7.2 Symbolic Computation with SymPy

```python
import sympy as sp

# Symbolic matrix
a, b, c, d = sp.symbols('a b c d')
M = sp.Matrix([[a, b], [c, d]])

print("General 2x2 matrix:")
print(f"  det = {M.det()}")
print(f"  trace = {M.trace()}")
print(f"  Characteristic polynomial: {M.charpoly().as_expr()}")

# Cayley-Hamilton theorem: every matrix satisfies its own characteristic polynomial
# A² - (a+d)A + (ad-bc)I = 0
lam = sp.Symbol('lambda')
char_poly = M.charpoly(lam)
print(f"\nCharacteristic polynomial p(λ) = {char_poly.as_expr()}")
print("Cayley-Hamilton: verify p(M) = 0")
result = M**2 - (a+d)*M + (a*d - b*c)*sp.eye(2)
print(f"  p(M) = {sp.simplify(result)}")
```

---

## Exercises

### Basic Problems

**1.** Find the determinant and inverse of the following matrix:
$$
A = \begin{pmatrix} 1 & 2 & 1 \\ 3 & 1 & -1 \\ 2 & 0 & 3 \end{pmatrix}
$$

**2.** Solve the following system of equations using Gaussian elimination:
$$
x + 2y + z = 4, \quad 3x + y - z = 2, \quad 2x + 3z = 7
$$

**3.** Find the eigenvalues and eigenvectors of the following matrix:
$$
B = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}
$$

### Applied Problems

**4.** Determine whether the quadratic form $Q(x,y) = 3x^2 + 2xy + 3y^2$ is positive definite/negative definite/indefinite, and perform a principal axis transformation.

**5.** Two objects with masses $m_1 = 1$, $m_2 = 2$ are connected by three springs with spring constants $k_1 = 3$, $k_2 = 4$, $k_3 = 2$ (wall–k₁–m₁–k₂–m₂–k₃–wall). Find the normal frequencies and normal modes.

**6.** For a spin-1/2 measurement operator in the magnetic field direction $\hat{n} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$, $\hat{S}_n = \frac{\hbar}{2}(\sigma_x \sin\theta\cos\phi + \sigma_y \sin\theta\sin\phi + \sigma_z \cos\theta)$, find the eigenvalues and eigenvectors.

---

## References

- Boas, *Mathematical Methods in the Physical Sciences*, Chapter 3
- Strang, *Introduction to Linear Algebra*
- Arfken & Weber, *Mathematical Methods for Physicists*, Chapters 3-4
- Code for this lesson: See `examples/Math_for_AI/01_vector_matrix_ops.py`, `02_svd_pca.py`

---

## Next Lesson

[04. Partial Differentiation](./04_Partial_Differentiation.md) covers calculus of multivariable functions.
