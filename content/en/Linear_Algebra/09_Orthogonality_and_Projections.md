# Lesson 9: Orthogonality and Projections

## Learning Objectives

- Compute orthogonal projections onto vectors, lines, and subspaces
- Derive and apply the normal equations for least-squares problems
- Implement the Gram-Schmidt process and understand its connection to QR decomposition
- Compute the QR decomposition and use it to solve least-squares problems
- Apply least squares to linear regression and polynomial fitting

---

## 1. Orthogonal Projection onto a Vector

### 1.1 Projection Formula

The orthogonal projection of $\mathbf{b}$ onto $\mathbf{a}$ is the point on the line spanned by $\mathbf{a}$ that is closest to $\mathbf{b}$:

$$\mathrm{proj}_{\mathbf{a}}(\mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{a} \cdot \mathbf{a}} \mathbf{a} = \frac{\mathbf{a}^T \mathbf{b}}{\mathbf{a}^T \mathbf{a}} \mathbf{a}$$

The **projection matrix** for projecting onto $\mathbf{a}$ is:

$$P = \frac{\mathbf{a}\mathbf{a}^T}{\mathbf{a}^T \mathbf{a}}$$

### 1.2 Properties of Projection Matrices

A projection matrix $P$ satisfies:
- **Idempotent**: $P^2 = P$ (projecting twice gives the same result)
- **Symmetric**: $P = P^T$ (for orthogonal projections)
- The **error** (residual) $\mathbf{e} = \mathbf{b} - P\mathbf{b}$ is orthogonal to the projection: $P\mathbf{b} \perp \mathbf{e}$

```python
import numpy as np
import matplotlib.pyplot as plt

a = np.array([2, 1], dtype=float)
b = np.array([1, 3], dtype=float)

# Projection of b onto a
proj = (np.dot(a, b) / np.dot(a, a)) * a
error = b - proj

print(f"proj_a(b) = {proj}")
print(f"error = {error}")
print(f"proj . error = {np.dot(proj, error):.10f}")  # ~0 (orthogonal)

# Projection matrix
P = np.outer(a, a) / np.dot(a, a)
print(f"\nProjection matrix P:\n{P}")
print(f"P^2 == P? {np.allclose(P @ P, P)}")
print(f"P == P^T? {np.allclose(P, P.T)}")
print(f"P @ b = {P @ b}")  # same as proj

# Visualize
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1,
          color='red', width=0.008, label='a')
ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1,
          color='blue', width=0.008, label='b')
ax.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1,
          color='green', width=0.008, label='proj_a(b)')
ax.plot([proj[0], b[0]], [proj[1], b[1]], 'k--', linewidth=1, label='error (perpendicular)')
ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend()
ax.set_title('Orthogonal Projection onto a Vector')
plt.show()
```

---

## 2. Projection onto a Subspace

### 2.1 General Formula

To project $\mathbf{b}$ onto the column space of a matrix $A$ (which has linearly independent columns):

$$\hat{\mathbf{b}} = A(A^T A)^{-1} A^T \mathbf{b}$$

The **projection matrix** is:

$$P = A(A^T A)^{-1} A^T$$

### 2.2 Derivation

We seek $\hat{\mathbf{b}} = A\hat{\mathbf{x}}$ such that the error $\mathbf{e} = \mathbf{b} - A\hat{\mathbf{x}}$ is orthogonal to the column space of $A$:

$$A^T(\mathbf{b} - A\hat{\mathbf{x}}) = \mathbf{0}$$

This gives the **normal equations**:

$$A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$$

Solving: $\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$, and $\hat{\mathbf{b}} = A\hat{\mathbf{x}}$.

```python
# Project b onto the plane spanned by a1 and a2
a1 = np.array([1, 0, 0], dtype=float)
a2 = np.array([0, 1, 0], dtype=float)
b = np.array([1, 2, 3], dtype=float)

A = np.column_stack([a1, a2])

# Projection
P = A @ np.linalg.inv(A.T @ A) @ A.T
b_hat = P @ b
error = b - b_hat

print(f"Projection: {b_hat}")
print(f"Error: {error}")
print(f"Error orthogonal to a1? {np.isclose(np.dot(error, a1), 0)}")
print(f"Error orthogonal to a2? {np.isclose(np.dot(error, a2), 0)}")
print(f"P is idempotent? {np.allclose(P @ P, P)}")
print(f"P is symmetric? {np.allclose(P, P.T)}")
```

### 2.3 Complementary Projection

If $P$ projects onto a subspace $S$, then $I - P$ projects onto $S^{\perp}$ (the orthogonal complement):

```python
# Complementary projection
P_perp = np.eye(3) - P

b_perp = P_perp @ b
print(f"Projection onto complement: {b_perp}")
print(f"b = proj + proj_perp? {np.allclose(b, b_hat + b_perp)}")
print(f"proj . proj_perp = {np.dot(b_hat, b_perp):.10f}")  # ~0
```

---

## 3. Least Squares

### 3.1 The Problem

Given an **overdetermined** system $A\mathbf{x} = \mathbf{b}$ (more equations than unknowns), there is typically no exact solution. The **least squares** solution minimizes:

$$\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2$$

### 3.2 Normal Equations

Setting the gradient to zero yields the **normal equations**:

$$A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$$

If $A$ has full column rank, the unique solution is:

$$\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$$

### 3.3 Linear Regression Example

```python
# Linear regression: fit y = ax + b
np.random.seed(42)
x_data = np.linspace(0, 10, 50)
y_data = 2.5 * x_data + 1.0 + np.random.randn(50) * 2

# Design matrix (column of ones + column of x values)
A = np.column_stack([np.ones_like(x_data), x_data])
b = y_data

# Normal equation
x_hat = np.linalg.solve(A.T @ A, A.T @ b)
print(f"Intercept: {x_hat[0]:.4f}, Slope: {x_hat[1]:.4f}")
print(f"True: Intercept = 1.0, Slope = 2.5")

# Prediction and residual
y_pred = A @ x_hat
residual = b - y_pred
print(f"Residual norm: {np.linalg.norm(residual):.4f}")

# Compare with np.linalg.lstsq
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"lstsq result: {x_lstsq}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(x_data, y_data, alpha=0.5, s=20, label='Data')
ax1.plot(x_data, y_pred, 'r-', linewidth=2,
         label=f'Fit: y = {x_hat[1]:.2f}x + {x_hat[0]:.2f}')
ax1.set_xlabel('x'); ax1.set_ylabel('y')
ax1.set_title('Least Squares Linear Regression')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.stem(range(len(residual)), residual, linefmt='b-', markerfmt='bo', basefmt='r-')
ax2.set_xlabel('Sample index'); ax2.set_ylabel('Residual')
ax2.set_title('Residuals')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.4 Polynomial Fitting

```python
# Polynomial regression: fit y = a0 + a1*x + a2*x^2 + a3*x^3
np.random.seed(42)
x = np.linspace(-2, 2, 60)
y_true = 0.5 * x**3 - 2 * x + 1
y = y_true + np.random.randn(60) * 0.5

degrees = [1, 3, 10]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, deg in zip(axes, degrees):
    # Vandermonde matrix
    A = np.vander(x, N=deg+1, increasing=True)
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]

    x_fine = np.linspace(-2.2, 2.2, 200)
    A_fine = np.vander(x_fine, N=deg+1, increasing=True)
    y_fit = A_fine @ coeffs

    ax.scatter(x, y, alpha=0.4, s=15, label='Data')
    ax.plot(x_fine, y_fit, 'r-', linewidth=2, label=f'Degree {deg} fit')
    ax.set_ylim(-8, 8)
    ax.set_title(f'Polynomial Degree {deg}')
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 4. Gram-Schmidt Process

### 4.1 The Algorithm

Given linearly independent vectors $\mathbf{a}_1, \ldots, \mathbf{a}_n$, the Gram-Schmidt process produces an **orthonormal** set $\mathbf{q}_1, \ldots, \mathbf{q}_n$ that spans the same subspace.

**Steps**:

$$\mathbf{u}_1 = \mathbf{a}_1, \quad \mathbf{q}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|}$$

$$\mathbf{u}_k = \mathbf{a}_k - \sum_{j=1}^{k-1} (\mathbf{q}_j^T \mathbf{a}_k) \mathbf{q}_j, \quad \mathbf{q}_k = \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|}$$

### 4.2 Implementation

```python
def gram_schmidt(A):
    """Classical Gram-Schmidt orthonormalization.

    A: matrix whose columns are the input vectors.
    Returns Q (orthonormal columns) and R (upper triangular).
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        # Subtract projections onto previous basis vectors
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-10:
            raise ValueError("Columns are linearly dependent")
        Q[:, j] = v / R[j, j]

    return Q, R

# Test
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

Q, R = gram_schmidt(A)

print(f"Q (orthonormal):\n{np.round(Q, 4)}")
print(f"R (upper triangular):\n{np.round(R, 4)}")
print(f"Q^T Q (should be I):\n{np.round(Q.T @ Q, 10)}")
print(f"QR == A? {np.allclose(Q @ R, A)}")
```

### 4.3 Modified Gram-Schmidt

The classical algorithm can suffer from numerical instability. The **modified** Gram-Schmidt algorithm is more stable:

```python
def modified_gram_schmidt(A):
    """Modified Gram-Schmidt -- more numerically stable."""
    m, n = A.shape
    Q = A.astype(float).copy()
    R = np.zeros((n, n))

    for j in range(n):
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / R[j, j]

        for k in range(j + 1, n):
            R[j, k] = np.dot(Q[:, j], Q[:, k])
            Q[:, k] -= R[j, k] * Q[:, j]

    return Q, R

# Compare classical vs modified on a mildly ill-conditioned matrix
np.random.seed(42)
A_ill = np.random.randn(100, 50) + 1e-8

Q_classical, R_classical = gram_schmidt(A_ill)
Q_modified, R_modified = modified_gram_schmidt(A_ill)

# Orthogonality quality
orth_error_classical = np.linalg.norm(Q_classical.T @ Q_classical - np.eye(50))
orth_error_modified = np.linalg.norm(Q_modified.T @ Q_modified - np.eye(50))

print(f"Orthogonality error (classical): {orth_error_classical:.2e}")
print(f"Orthogonality error (modified):  {orth_error_modified:.2e}")
```

---

## 5. QR Decomposition

### 5.1 Definition

Every $m \times n$ matrix $A$ (with $m \ge n$ and full column rank) can be factored as:

$$A = QR$$

where:
- $Q \in \mathbb{R}^{m \times n}$ has orthonormal columns ($Q^TQ = I_n$)
- $R \in \mathbb{R}^{n \times n}$ is upper triangular with positive diagonal entries

### 5.2 Computing QR with NumPy

```python
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

Q, R = np.linalg.qr(A)

print(f"Q:\n{np.round(Q, 4)}")
print(f"R:\n{np.round(R, 4)}")
print(f"Q^T Q:\n{np.round(Q.T @ Q, 10)}")
print(f"QR == A? {np.allclose(Q @ R, A)}")
```

### 5.3 Solving Least Squares with QR

The QR decomposition provides a numerically stable way to solve least squares:

$$A\mathbf{x} = \mathbf{b} \implies QR\mathbf{x} = \mathbf{b} \implies R\mathbf{x} = Q^T\mathbf{b}$$

This avoids forming $A^TA$ (which squares the condition number).

```python
# Least squares via QR
np.random.seed(42)
m, n = 100, 3
A = np.random.randn(m, n)
b = np.random.randn(m)

# Method 1: Normal equations (less stable)
x_normal = np.linalg.solve(A.T @ A, A.T @ b)

# Method 2: QR decomposition (more stable)
Q, R = np.linalg.qr(A)
x_qr = np.linalg.solve(R, Q.T @ b)

# Method 3: NumPy lstsq (uses SVD internally)
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]

print(f"Normal equations: {x_normal}")
print(f"QR decomposition: {x_qr}")
print(f"NumPy lstsq:      {x_lstsq}")
print(f"All match? {np.allclose(x_normal, x_qr) and np.allclose(x_qr, x_lstsq)}")

# Condition number comparison
print(f"\ncond(A):     {np.linalg.cond(A):.4f}")
print(f"cond(A^T A): {np.linalg.cond(A.T @ A):.4f}")  # squared!
```

---

## 6. Orthogonal Complements

### 6.1 Definition

The **orthogonal complement** of a subspace $S$ in $\mathbb{R}^n$ is:

$$S^{\perp} = \{\mathbf{v} \in \mathbb{R}^n : \mathbf{v} \cdot \mathbf{s} = 0 \text{ for all } \mathbf{s} \in S\}$$

Key facts:
- $\dim(S) + \dim(S^{\perp}) = n$
- $(S^{\perp})^{\perp} = S$
- Every $\mathbf{v} \in \mathbb{R}^n$ can be uniquely written as $\mathbf{v} = \mathbf{v}_S + \mathbf{v}_{S^{\perp}}$

### 6.2 Fundamental Subspace Relationships

For a matrix $A$:
- $\mathrm{col}(A)^{\perp} = \mathrm{null}(A^T)$ (in $\mathbb{R}^m$)
- $\mathrm{row}(A)^{\perp} = \mathrm{null}(A)$ (in $\mathbb{R}^n$)

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=float)

# Column space (via QR)
Q_col, _ = np.linalg.qr(A)

# Null space of A^T (left null space)
from scipy.linalg import null_space
left_null = null_space(A.T)

print(f"Column space basis (Q):\n{np.round(Q_col, 4)}")
print(f"Left null space basis:\n{np.round(left_null, 4)}")

# Verify orthogonality
print(f"\nQ^T @ left_null:\n{np.round(Q_col.T @ left_null, 10)}")  # should be ~0
```

---

## 7. Application: Weighted Least Squares

When different observations have different reliabilities, use **weighted least squares**:

$$\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} (A\mathbf{x} - \mathbf{b})^T W (A\mathbf{x} - \mathbf{b})$$

where $W$ is a diagonal weight matrix. The solution is:

$$\hat{\mathbf{x}} = (A^T W A)^{-1} A^T W \mathbf{b}$$

```python
# Weighted least squares
np.random.seed(42)
x_data = np.linspace(0, 10, 50)
y_true = 2 * x_data + 1

# Add heteroscedastic noise (more noise for larger x)
noise_std = 0.5 + 0.5 * x_data
y_data = y_true + np.random.randn(50) * noise_std

# Design matrix
A = np.column_stack([np.ones_like(x_data), x_data])

# Weights: inverse of noise variance
weights = 1.0 / noise_std**2
W = np.diag(weights)

# Ordinary least squares
x_ols = np.linalg.solve(A.T @ A, A.T @ y_data)

# Weighted least squares
x_wls = np.linalg.solve(A.T @ W @ A, A.T @ W @ y_data)

print(f"OLS: intercept = {x_ols[0]:.4f}, slope = {x_ols[1]:.4f}")
print(f"WLS: intercept = {x_wls[0]:.4f}, slope = {x_wls[1]:.4f}")
print(f"True: intercept = 1.0, slope = 2.0")

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(x_data, y_data, yerr=noise_std, fmt='o', alpha=0.3, markersize=3)
plt.plot(x_data, A @ x_ols, 'b-', linewidth=2, label=f'OLS: {x_ols[1]:.2f}x + {x_ols[0]:.2f}')
plt.plot(x_data, A @ x_wls, 'r-', linewidth=2, label=f'WLS: {x_wls[1]:.2f}x + {x_wls[0]:.2f}')
plt.plot(x_data, y_true, 'k--', linewidth=1, label='True: 2.0x + 1.0')
plt.xlabel('x'); plt.ylabel('y')
plt.legend(); plt.grid(True, alpha=0.3)
plt.title('Ordinary vs Weighted Least Squares')
plt.show()
```

---

## 8. Householder Reflections (Advanced)

NumPy internally uses **Householder reflections** rather than Gram-Schmidt for QR. A Householder reflection is:

$$H = I - 2\mathbf{v}\mathbf{v}^T \quad (\|\mathbf{v}\| = 1)$$

It reflects vectors across the hyperplane perpendicular to $\mathbf{v}$.

```python
def householder_qr(A):
    """QR decomposition using Householder reflections."""
    m, n = A.shape
    R = A.astype(float).copy()
    Q = np.eye(m)

    for j in range(min(m - 1, n)):
        # Select the column below the diagonal
        x = R[j:, j]

        # Householder vector
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * (-1 if x[0] < 0 else 1)
        v = x + e1
        v = v / np.linalg.norm(v)

        # Apply reflection to R
        R[j:, j:] -= 2 * np.outer(v, v @ R[j:, j:])

        # Accumulate Q
        Q[j:, :] -= 2 * np.outer(v, v @ Q[j:, :])

    return Q.T, R

# Test
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

Q_hh, R_hh = householder_qr(A)
print(f"Householder Q:\n{np.round(Q_hh, 4)}")
print(f"Householder R:\n{np.round(R_hh, 4)}")
print(f"QR == A? {np.allclose(Q_hh @ R_hh, A)}")
```

---

## 9. Summary

| Concept | Formula / Description |
|---------|----------------------|
| Projection onto vector | $\frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}}\mathbf{a}$ |
| Projection matrix (subspace) | $P = A(A^TA)^{-1}A^T$ |
| Normal equations | $A^TA\hat{\mathbf{x}} = A^T\mathbf{b}$ |
| Gram-Schmidt | Produces orthonormal basis from independent vectors |
| QR decomposition | $A = QR$, $Q$ orthonormal, $R$ upper triangular |
| Least squares via QR | $R\hat{\mathbf{x}} = Q^T\mathbf{b}$ (more stable than normal equations) |
| NumPy | `np.linalg.qr(A)`, `np.linalg.lstsq(A, b)` |

---

## Exercises

### Exercise 1: Projection Matrices

Compute the projection matrix $P$ that projects $\mathbb{R}^3$ onto the plane spanned by $\mathbf{a}_1 = [1, 0, 1]^T$ and $\mathbf{a}_2 = [0, 1, 1]^T$. Verify that $P$ is idempotent and symmetric. Project $\mathbf{b} = [1, 2, 3]^T$ and check that the error is orthogonal to both $\mathbf{a}_1$ and $\mathbf{a}_2$.

### Exercise 2: Least Squares Fitting

Fit a quadratic $y = ax^2 + bx + c$ to the data:

| x | -2 | -1 | 0 | 1 | 2 | 3 |
|---|----|----|---|---|---|---|
| y | 7  | 2  | 1 | 2 | 7 | 14|

Set up and solve the normal equations. Compute the residual.

### Exercise 3: Gram-Schmidt

Apply the Gram-Schmidt process (by hand and in code) to $\{\mathbf{a}_1 = [1, 1, 1]^T, \mathbf{a}_2 = [1, 1, 0]^T, \mathbf{a}_3 = [1, 0, 0]^T\}$. Verify the result is orthonormal and that $A = QR$.

### Exercise 4: QR vs Normal Equations

Generate a $200 \times 10$ matrix $A$ with a large condition number (use `np.linalg.svd` to set specific singular values). Solve $A\mathbf{x} = \mathbf{b}$ using both the normal equations and QR. Compare the accuracy of the two solutions against the known true solution. When does the normal equation approach break down?

### Exercise 5: Weighted Least Squares

A sensor makes measurements at different times with known measurement uncertainties $\sigma_i$. Use weighted least squares to fit a linear model, and show that it gives better parameter estimates than ordinary least squares when the noise is heteroscedastic.

---

[<< Previous: Lesson 8 - Principal Component Analysis](08_Principal_Component_Analysis.md) | [Overview](00_Overview.md) | [Next: Lesson 10 - Matrix Decompositions >>](10_Matrix_Decompositions.md)

**License**: CC BY-NC 4.0
