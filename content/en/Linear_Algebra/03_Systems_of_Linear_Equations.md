# Lesson 3: Systems of Linear Equations

## Learning Objectives

- Represent a system of linear equations in matrix form $A\mathbf{x} = \mathbf{b}$
- Perform Gaussian elimination by hand and in code, reducing to row echelon form (REF) and reduced row echelon form (RREF)
- Determine whether a system has zero, one, or infinitely many solutions using the rank condition
- Compute the LU decomposition and use it to solve systems efficiently
- Apply these techniques to practical problems including circuit analysis and interpolation

---

## 1. Matrix Form of Linear Systems

A system of $m$ linear equations in $n$ unknowns:

$$\begin{aligned}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n &= b_2 \\
&\;\;\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m
\end{aligned}$$

can be written compactly as:

$$A\mathbf{x} = \mathbf{b}$$

where $A \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{b} \in \mathbb{R}^m$.

```python
import numpy as np
from scipy import linalg

# Example: 3 equations, 3 unknowns
# 2x + y - z = 8
# -3x - y + 2z = -11
# -2x + y + 2z = -3
A = np.array([[ 2,  1, -1],
              [-3, -1,  2],
              [-2,  1,  2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

x = np.linalg.solve(A, b)
print(f"Solution: x = {x}")
print(f"Verification: Ax = {A @ x}")
```

---

## 2. Gaussian Elimination

### 2.1 The Algorithm

Gaussian elimination transforms the augmented matrix $[A \mid \mathbf{b}]$ into **row echelon form** (REF) using three elementary row operations:

1. **Swap** two rows
2. **Scale** a row by a non-zero scalar
3. **Add** a multiple of one row to another

### 2.2 Row Echelon Form (REF)

A matrix is in REF if:
- All zero rows are at the bottom
- The leading entry (pivot) of each non-zero row is to the right of the pivot in the row above
- All entries below each pivot are zero

### 2.3 Implementation

```python
def gaussian_elimination(A, b, verbose=False):
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix
    Ab = np.column_stack([A.astype(float), b.astype(float)])

    if verbose:
        print("Augmented matrix:")
        print(Ab)
        print()

    # Forward elimination
    for col in range(n):
        # Partial pivoting: find row with largest absolute value in column
        max_row = col + np.argmax(np.abs(Ab[col:, col]))
        if max_row != col:
            Ab[[col, max_row]] = Ab[[max_row, col]]
            if verbose:
                print(f"Swap rows {col} and {max_row}")

        if abs(Ab[col, col]) < 1e-12:
            raise ValueError(f"Zero pivot encountered at column {col}")

        # Eliminate entries below the pivot
        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]
            if verbose:
                print(f"R{row} <- R{row} - ({factor:.4f}) * R{col}")

        if verbose:
            print(Ab)
            print()

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]

    return x

# Test
A = np.array([[ 2,  1, -1],
              [-3, -1,  2],
              [-2,  1,  2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

x = gaussian_elimination(A, b, verbose=True)
print(f"Solution: {x}")
```

---

## 3. Reduced Row Echelon Form (RREF)

### 3.1 Definition

RREF is REF with two additional conditions:
- Each pivot is equal to 1
- Each pivot is the only non-zero entry in its column

### 3.2 Implementation

```python
def rref(A, tol=1e-12):
    """Compute the reduced row echelon form of matrix A."""
    M = A.astype(float).copy()
    m, n = M.shape
    pivot_row = 0

    pivot_cols = []

    for col in range(n):
        if pivot_row >= m:
            break

        # Find pivot
        max_row = pivot_row + np.argmax(np.abs(M[pivot_row:, col]))
        if abs(M[max_row, col]) < tol:
            continue  # skip this column

        # Swap rows
        M[[pivot_row, max_row]] = M[[max_row, pivot_row]]

        # Scale pivot to 1
        M[pivot_row] = M[pivot_row] / M[pivot_row, col]

        # Eliminate all other entries in this column
        for row in range(m):
            if row != pivot_row and abs(M[row, col]) > tol:
                M[row] -= M[row, col] * M[pivot_row]

        pivot_cols.append(col)
        pivot_row += 1

    return M, pivot_cols

# Example
A_aug = np.array([[ 1,  2, -1,  3],
                   [ 2,  4,  1,  6],
                   [ 3,  6,  0,  9]], dtype=float)

R, pivots = rref(A_aug)
print(f"RREF:\n{R}")
print(f"Pivot columns: {pivots}")
```

### 3.3 Using SymPy for Exact RREF

```python
import sympy

# SymPy gives exact rational arithmetic
M = sympy.Matrix([[1, 2, -1, 3],
                   [2, 4,  1, 6],
                   [3, 6,  0, 9]])

R, pivots = M.rref()
print(f"RREF:\n{R}")
print(f"Pivot columns: {pivots}")
```

---

## 4. Existence and Uniqueness of Solutions

### 4.1 The Rank Condition

For the system $A\mathbf{x} = \mathbf{b}$ with $A \in \mathbb{R}^{m \times n}$:

| Condition | Result |
|-----------|--------|
| $\mathrm{rank}(A) < \mathrm{rank}([A \mid \mathbf{b}])$ | **No solution** (inconsistent) |
| $\mathrm{rank}(A) = \mathrm{rank}([A \mid \mathbf{b}]) = n$ | **Unique solution** |
| $\mathrm{rank}(A) = \mathrm{rank}([A \mid \mathbf{b}]) < n$ | **Infinitely many solutions** |

### 4.2 Geometric Interpretation

- **Unique solution**: $n$ hyperplanes intersect at exactly one point
- **No solution**: at least two hyperplanes are parallel (inconsistent)
- **Infinite solutions**: hyperplanes intersect along a line, plane, or higher-dimensional flat

```python
# Case 1: Unique solution
A1 = np.array([[1, 1], [1, -1]], dtype=float)
b1 = np.array([4, 2], dtype=float)
Ab1 = np.column_stack([A1, b1])
print("Case 1 (unique):")
print(f"  rank(A) = {np.linalg.matrix_rank(A1)}, rank([A|b]) = {np.linalg.matrix_rank(Ab1)}, n = {A1.shape[1]}")
print(f"  Solution: {np.linalg.solve(A1, b1)}")

# Case 2: No solution
A2 = np.array([[1, 1], [1, 1]], dtype=float)
b2 = np.array([2, 3], dtype=float)
Ab2 = np.column_stack([A2, b2])
print("\nCase 2 (no solution):")
print(f"  rank(A) = {np.linalg.matrix_rank(A2)}, rank([A|b]) = {np.linalg.matrix_rank(Ab2)}")

# Case 3: Infinitely many solutions
A3 = np.array([[1, 2], [2, 4]], dtype=float)
b3 = np.array([3, 6], dtype=float)
Ab3 = np.column_stack([A3, b3])
print("\nCase 3 (infinite solutions):")
print(f"  rank(A) = {np.linalg.matrix_rank(A3)}, rank([A|b]) = {np.linalg.matrix_rank(Ab3)}, n = {A3.shape[1]}")
```

### 4.3 Visualizing 2D Systems

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

x_vals = np.linspace(-1, 5, 100)

# Unique solution: x + y = 4, x - y = 2
ax = axes[0]
ax.plot(x_vals, 4 - x_vals, label='x + y = 4')
ax.plot(x_vals, x_vals - 2, label='x - y = 2')
ax.plot(3, 1, 'ro', markersize=8)
ax.set_title('Unique solution (3, 1)')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

# No solution: x + y = 2, x + y = 3 (parallel lines)
ax = axes[1]
ax.plot(x_vals, 2 - x_vals, label='x + y = 2')
ax.plot(x_vals, 3 - x_vals, label='x + y = 3')
ax.set_title('No solution (parallel)')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

# Infinite solutions: x + 2y = 3, 2x + 4y = 6 (same line)
ax = axes[2]
ax.plot(x_vals, (3 - x_vals) / 2, label='x + 2y = 3', linewidth=3, alpha=0.5)
ax.plot(x_vals, (6 - 2*x_vals) / 4, label='2x + 4y = 6', linewidth=1, linestyle='--')
ax.set_title('Infinite solutions (same line)')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

for ax in axes:
    ax.set_xlim(-1, 5); ax.set_ylim(-2, 4)
    ax.set_xlabel('x'); ax.set_ylabel('y')

plt.tight_layout()
plt.show()
```

---

## 5. LU Decomposition

### 5.1 Concept

LU decomposition factors a matrix $A$ into:

$$A = LU$$

where $L$ is **lower triangular** (with ones on the diagonal) and $U$ is **upper triangular**.

With partial pivoting, we get:

$$PA = LU$$

where $P$ is a permutation matrix.

### 5.2 Why LU?

Once we have $A = LU$, solving $A\mathbf{x} = \mathbf{b}$ reduces to two triangular solves:

1. **Forward substitution**: Solve $L\mathbf{y} = \mathbf{b}$ for $\mathbf{y}$
2. **Back substitution**: Solve $U\mathbf{x} = \mathbf{y}$ for $\mathbf{x}$

This is especially efficient when solving the same system with **multiple right-hand sides**, since the decomposition is computed only once.

### 5.3 Implementation

```python
from scipy.linalg import lu, lu_factor, lu_solve

A = np.array([[ 2,  1, -1],
              [-3, -1,  2],
              [-2,  1,  2]], dtype=float)

# Full LU decomposition
P, L, U = lu(A)
print(f"P =\n{P}")
print(f"L =\n{L}")
print(f"U =\n{U}")

# Verify: PA = LU  =>  A = P^T L U
print(f"P^T L U =\n{P.T @ L @ U}")
print(f"Matches A? {np.allclose(A, P.T @ L @ U)}")

# Solve using LU
b = np.array([8, -11, -3], dtype=float)
lu_piv = lu_factor(A)
x = lu_solve(lu_piv, b)
print(f"Solution: {x}")

# Solve with multiple right-hand sides
B = np.array([[8, 1],
              [-11, 2],
              [-3, 3]], dtype=float)
X = lu_solve(lu_piv, B)
print(f"Solutions for two RHS:\n{X}")
```

### 5.4 Manual LU (Without Pivoting)

```python
def lu_no_pivot(A):
    """LU decomposition without pivoting (for educational purposes)."""
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()

    for col in range(n - 1):
        for row in range(col + 1, n):
            factor = U[row, col] / U[col, col]
            L[row, col] = factor
            U[row, col:] -= factor * U[col, col:]

    return L, U

A = np.array([[2, 1, -1],
              [4, 5, -3],
              [6, 10, 1]], dtype=float)

L, U = lu_no_pivot(A)
print(f"L =\n{L}")
print(f"U =\n{U}")
print(f"L @ U =\n{L @ U}")
print(f"Matches A? {np.allclose(A, L @ U)}")
```

### 5.5 Computational Cost

| Operation | Cost (flops) |
|-----------|-------------|
| LU decomposition | $\frac{2}{3}n^3$ |
| Forward substitution | $n^2$ |
| Back substitution | $n^2$ |
| Total for one solve | $\frac{2}{3}n^3 + 2n^2$ |
| Each additional RHS | $2n^2$ |

Compare with computing $A^{-1}$ explicitly: $\sim 2n^3$ flops. LU is much more efficient, and numerically more stable.

---

## 6. Homogeneous Systems

A system is **homogeneous** if $\mathbf{b} = \mathbf{0}$:

$$A\mathbf{x} = \mathbf{0}$$

Homogeneous systems always have at least the **trivial solution** $\mathbf{x} = \mathbf{0}$.

A non-trivial solution exists if and only if $\mathrm{rank}(A) < n$ (number of unknowns).

The set of all solutions forms the **null space** (or **kernel**) of $A$:

$$\mathrm{null}(A) = \{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0}\}$$

```python
# Matrix with rank < n => non-trivial null space
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

print(f"rank(A) = {np.linalg.matrix_rank(A)}")  # 2 < 3

# Find null space using SVD
from scipy.linalg import null_space
ns = null_space(A)
print(f"Null space basis:\n{ns}")
print(f"A @ null_vector =\n{A @ ns}")  # should be ~0
```

---

## 7. Overdetermined and Underdetermined Systems

### 7.1 Overdetermined Systems ($m > n$)

More equations than unknowns. Typically no exact solution exists. The **least squares** solution minimizes $\|A\mathbf{x} - \mathbf{b}\|^2$:

$$\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$$

```python
# Overdetermined: 4 equations, 2 unknowns
A = np.array([[1, 1],
              [1, 2],
              [1, 3],
              [1, 4]], dtype=float)
b = np.array([2.1, 2.9, 4.2, 4.8], dtype=float)

# Least squares solution
x_lstsq, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
print(f"Least squares solution: {x_lstsq}")
print(f"Residual norm: {np.linalg.norm(A @ x_lstsq - b):.6f}")

# Via normal equation
x_normal = np.linalg.solve(A.T @ A, A.T @ b)
print(f"Normal equation solution: {x_normal}")
```

### 7.2 Underdetermined Systems ($m < n$)

Fewer equations than unknowns. If a solution exists, there are infinitely many. The **minimum norm** solution is the one with the smallest $\|\mathbf{x}\|$:

```python
# Underdetermined: 2 equations, 3 unknowns
A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=float)
b = np.array([6, 15], dtype=float)

# Minimum norm solution via pseudoinverse
x_min_norm = np.linalg.pinv(A) @ b
print(f"Minimum norm solution: {x_min_norm}")
print(f"Verification: Ax = {A @ x_min_norm}")
print(f"||x|| = {np.linalg.norm(x_min_norm):.6f}")
```

---

## 8. Application: Polynomial Interpolation

Given $n+1$ data points $(x_0, y_0), \ldots, (x_n, y_n)$, find the polynomial $p(x) = c_0 + c_1 x + \cdots + c_n x^n$ that passes through all of them. This leads to the **Vandermonde system**:

$$\begin{bmatrix} 1 & x_0 & x_0^2 & \cdots & x_0^n \\ 1 & x_1 & x_1^2 & \cdots & x_1^n \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_n & x_n^2 & \cdots & x_n^n \end{bmatrix} \begin{bmatrix} c_0 \\ c_1 \\ \vdots \\ c_n \end{bmatrix} = \begin{bmatrix} y_0 \\ y_1 \\ \vdots \\ y_n \end{bmatrix}$$

```python
import matplotlib.pyplot as plt

# Data points
x_data = np.array([0, 1, 2, 3, 4], dtype=float)
y_data = np.array([1, 3, 2, 5, 4], dtype=float)

# Build Vandermonde matrix
n = len(x_data) - 1
V = np.vander(x_data, increasing=True)
print(f"Vandermonde matrix:\n{V}")

# Solve for coefficients
coeffs = np.linalg.solve(V, y_data)
print(f"Polynomial coefficients: {coeffs}")

# Plot
x_fine = np.linspace(-0.5, 4.5, 200)
V_fine = np.vander(x_fine, N=n+1, increasing=True)
y_fine = V_fine @ coeffs

plt.figure(figsize=(8, 5))
plt.plot(x_data, y_data, 'ro', markersize=8, label='Data points')
plt.plot(x_fine, y_fine, 'b-', label=f'Degree-{n} polynomial')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.grid(True, alpha=0.3)
plt.title('Polynomial Interpolation via Vandermonde System')
plt.show()
```

---

## 9. Pivoting Strategies

### 9.1 Why Pivot?

Without pivoting, Gaussian elimination can fail or produce wildly inaccurate results when a pivot is zero or very small relative to other entries.

### 9.2 Partial Pivoting

At each step, swap the current row with the row below that has the largest absolute value in the pivot column. This is the standard strategy used in practice and is what NumPy/SciPy implement.

### 9.3 Complete Pivoting

Searches the entire remaining sub-matrix for the largest entry and swaps both rows and columns. More stable but more expensive. Rarely needed in practice.

```python
def gaussian_partial_pivot(A, b):
    """Gaussian elimination with partial pivoting -- production quality."""
    n = len(b)
    Ab = np.column_stack([A.astype(float), b.astype(float)])

    for col in range(n):
        # Partial pivoting
        max_idx = col + np.argmax(np.abs(Ab[col:, col]))
        Ab[[col, max_idx]] = Ab[[max_idx, col]]

        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:]) / Ab[i, i]
    return x

# Test with a matrix that needs pivoting
A = np.array([[1e-20, 1],
              [1,     1]], dtype=float)
b = np.array([1, 2], dtype=float)

x_pivot = gaussian_partial_pivot(A, b)
x_numpy = np.linalg.solve(A, b)
print(f"With pivoting: {x_pivot}")
print(f"NumPy:         {x_numpy}")
```

---

## 10. Summary

| Concept | Description |
|---------|-------------|
| $A\mathbf{x} = \mathbf{b}$ | Matrix form of a linear system |
| Gaussian elimination | Row reduction to echelon form |
| REF / RREF | Row echelon and reduced row echelon forms |
| Rank condition | Determines existence/uniqueness of solutions |
| LU decomposition | $PA = LU$ for efficient repeated solves |
| Null space | $\{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ |
| Least squares | $\hat{\mathbf{x}} = (A^TA)^{-1}A^T\mathbf{b}$ for overdetermined systems |
| Partial pivoting | Swap rows for numerical stability |

---

## Exercises

### Exercise 1: Gaussian Elimination

Solve the following system by hand using Gaussian elimination, then verify with NumPy:

$$\begin{cases} x + 2y + z = 9 \\ 2x - y + 3z = 8 \\ 3x + y - z = 3 \end{cases}$$

### Exercise 2: RREF Analysis

Find the RREF of $\begin{bmatrix} 1 & 2 & 3 & 4 \\ 2 & 4 & 7 & 9 \\ 3 & 6 & 10 & 13 \end{bmatrix}$ and determine the rank, pivot columns, and free variables.

### Exercise 3: LU Decomposition

1. Compute the LU decomposition of $A = \begin{bmatrix} 2 & 4 & -2 \\ 4 & 9 & -3 \\ -2 & -3 & 7 \end{bmatrix}$.
2. Use your decomposition to solve $A\mathbf{x} = \mathbf{b}$ for $\mathbf{b} = [2, 8, 10]^T$ and $\mathbf{b} = [4, 8, -2]^T$.

### Exercise 4: Least Squares Line Fitting

Fit a line $y = ax + b$ to the data points $(1, 2.1)$, $(2, 3.9)$, $(3, 6.2)$, $(4, 7.8)$, $(5, 10.1)$ using the normal equation. Plot the data and the fitted line.

### Exercise 5: Null Space

Find a basis for the null space of $A = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 2 & 4 & 6 & 8 \\ 1 & 2 & 4 & 6 \end{bmatrix}$. What is the dimension of the null space?

---

[<< Previous: Lesson 2 - Matrices and Operations](02_Matrices_and_Operations.md) | [Overview](00_Overview.md) | [Next: Lesson 4 - Vector Norms and Inner Products >>](04_Vector_Norms_and_Inner_Products.md)

**License**: CC BY-NC 4.0
