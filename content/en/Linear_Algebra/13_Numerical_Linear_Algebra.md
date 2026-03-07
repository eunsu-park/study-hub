# Lesson 13: Numerical Linear Algebra

[Previous: Lesson 12](./12_Sparse_Matrices.md) | [Overview](./00_Overview.md) | [Next: Lesson 14](./14_Iterative_Methods.md)

---

## Learning Objectives

- Understand IEEE 754 floating-point representation and its implications for linear algebra
- Define machine epsilon and demonstrate its effect on computation
- Compute and interpret condition numbers to assess problem sensitivity
- Distinguish between forward stability, backward stability, and numerical stability
- Identify and avoid catastrophic cancellation in numerical algorithms
- Evaluate the numerical stability of fundamental algorithms (Gaussian elimination, Gram-Schmidt, Householder)

---

## 1. Floating-Point Arithmetic

### 1.1 IEEE 754 Representation

Every floating-point number is stored as:

$$x = (-1)^s \times m \times 2^e$$

where $s$ is the sign bit, $m$ is the mantissa (significand), and $e$ is the exponent.

| Format | Total bits | Mantissa bits | Exponent bits | Machine epsilon |
|---|---|---|---|---|
| float16 (half) | 16 | 10 | 5 | $\approx 9.77 \times 10^{-4}$ |
| float32 (single) | 32 | 23 | 8 | $\approx 1.19 \times 10^{-7}$ |
| float64 (double) | 64 | 52 | 11 | $\approx 2.22 \times 10^{-16}$ |

```python
import numpy as np

# Inspect floating-point properties
for dtype in [np.float16, np.float32, np.float64]:
    info = np.finfo(dtype)
    print(f"{dtype.__name__:>10}: eps={info.eps:.3e}, "
          f"min={info.tiny:.3e}, max={info.max:.3e}, "
          f"mantissa bits={info.nmant}")
```

### 1.2 Key Floating-Point Properties

Floating-point arithmetic does not obey the same algebraic rules as real arithmetic. The critical difference is that the result of every operation is **rounded** to the nearest representable number.

```python
# 1. Associativity fails
a = 1e16
b = -1e16
c = 1.0
print(f"(a + b) + c = {(a + b) + c}")  # 1.0
print(f"a + (b + c) = {a + (b + c)}")  # 0.0 (c is lost when added to a)

# 2. Distributivity fails
x = 1e-15
y = 1.0
z = -1.0
print(f"\nx * (y + z) = {x * (y + z)}")   # 0.0
print(f"x*y + x*z = {x * y + x * z}")     # 0.0 (but intermediate values differ)

# 3. Not all decimal fractions are representable
print(f"\n0.1 + 0.2 = {0.1 + 0.2}")  # 0.30000000000000004
print(f"0.1 + 0.2 == 0.3: {0.1 + 0.2 == 0.3}")  # False

# 4. Spacing between consecutive floats
x = 1.0
eps = np.spacing(x)
print(f"\nSpacing at 1.0: {eps}")
print(f"1.0 + eps/2 == 1.0: {1.0 + eps / 2 == 1.0}")  # True (rounded down)
print(f"1.0 + eps == 1.0: {1.0 + eps == 1.0}")          # False
```

### 1.3 The Standard Model of Floating-Point Arithmetic

The IEEE 754 standard guarantees that for any arithmetic operation $\circ \in \{+, -, \times, /\}$:

$$\text{fl}(a \circ b) = (a \circ b)(1 + \delta), \quad |\delta| \leq \epsilon_{\text{mach}}$$

where $\epsilon_{\text{mach}}$ is the machine epsilon. This means each individual operation introduces a relative error of at most $\epsilon_{\text{mach}}$, but errors can accumulate over many operations.

---

## 2. Machine Epsilon

### 2.1 Definition

Machine epsilon ($\epsilon_{\text{mach}}$) is the smallest positive number such that:

$$\text{fl}(1 + \epsilon_{\text{mach}}) > 1$$

Equivalently, it is half the spacing between 1.0 and the next representable float.

```python
def compute_machine_epsilon(dtype=np.float64):
    """Compute machine epsilon by successive halving."""
    eps = dtype(1.0)
    while dtype(1.0) + eps > dtype(1.0):
        eps_prev = eps
        eps = dtype(eps / 2.0)
    return eps_prev

# Compute for different precisions
for dtype in [np.float16, np.float32, np.float64]:
    computed = compute_machine_epsilon(dtype)
    actual = np.finfo(dtype).eps
    print(f"{dtype.__name__:>10}: computed={computed:.3e}, "
          f"numpy={actual:.3e}, match={np.isclose(computed, actual)}")
```

### 2.2 Relative vs Absolute Error

For a computed value $\hat{x}$ approximating the true value $x$:

- **Absolute error**: $|x - \hat{x}|$
- **Relative error**: $\frac{|x - \hat{x}|}{|x|}$ (for $x \neq 0$)

Floating-point arithmetic controls **relative error**, not absolute error. This means small numbers have tiny absolute error, while large numbers have large absolute error.

```python
# Relative error is bounded, absolute error scales with magnitude
for magnitude in [1e-10, 1e0, 1e10]:
    x = magnitude
    x_plus = np.nextafter(x, np.inf)
    abs_error = x_plus - x
    rel_error = abs_error / x
    print(f"x = {x:.0e}: abs_error = {abs_error:.3e}, "
          f"rel_error = {rel_error:.3e}")
```

---

## 3. Condition Number

### 3.1 Definition

The **condition number** measures how sensitive a problem's output is to small perturbations in its input. For a matrix $A$, the condition number (with respect to the 2-norm) is:

$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

where $\sigma_{\max}$ and $\sigma_{\min}$ are the largest and smallest singular values.

| $\kappa(A)$ | Interpretation |
|---|---|
| $\approx 1$ | Well-conditioned |
| $10^3$ | Modest; 3 digits of accuracy lost |
| $10^{16}$ | Ill-conditioned; essentially singular in float64 |
| $\infty$ | Singular matrix |

```python
# Condition number examples
matrices = {
    "Identity": np.eye(4),
    "Well-conditioned": np.array([[2, 1], [1, 2]]),
    "Moderate": np.array([[1, 1], [1, 1.0001]]),
    "Hilbert (ill-conditioned)": np.array([
        [1, 1/2, 1/3, 1/4],
        [1/2, 1/3, 1/4, 1/5],
        [1/3, 1/4, 1/5, 1/6],
        [1/4, 1/5, 1/6, 1/7]
    ]),
}

for name, A in matrices.items():
    cond = np.linalg.cond(A)
    sv = np.linalg.svd(A, compute_uv=False)
    print(f"{name:30s}: cond = {cond:.2e}, "
          f"sigma_max = {sv[0]:.4f}, sigma_min = {sv[-1]:.4e}")
```

### 3.2 Condition Number and Solution Accuracy

For a linear system $Ax = b$, a perturbation $\delta b$ in the right-hand side causes a perturbation $\delta x$ in the solution bounded by:

$$\frac{\|\delta x\|}{\|x\|} \leq \kappa(A) \frac{\|\delta b\|}{\|b\|}$$

This means that if $\kappa(A) = 10^k$, you lose approximately $k$ digits of accuracy.

```python
# Demonstrate condition number effect on solution accuracy
from scipy.linalg import hilbert

for n in [3, 5, 8, 10, 12]:
    H = hilbert(n)
    x_true = np.ones(n)
    b = H @ x_true

    # Solve
    x_computed = np.linalg.solve(H, b)

    # Errors
    rel_error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
    cond = np.linalg.cond(H)
    digits_lost = np.log10(cond)

    print(f"n={n:2d}: cond(H) = {cond:.2e}, "
          f"rel_error = {rel_error:.2e}, "
          f"digits lost ~ {digits_lost:.1f}")
```

### 3.3 Condition Number of Common Operations

```python
# Condition number of matrix multiplication
A = np.random.randn(4, 4)
B = np.random.randn(4, 4)

print(f"cond(A) = {np.linalg.cond(A):.2f}")
print(f"cond(B) = {np.linalg.cond(B):.2f}")
print(f"cond(AB) = {np.linalg.cond(A @ B):.2f}")
print(f"cond(A) * cond(B) = {np.linalg.cond(A) * np.linalg.cond(B):.2f}")
print("Note: cond(AB) <= cond(A) * cond(B)")

# Condition number of eigenvalue problems
# Eigenvalues of symmetric matrices are well-conditioned
# Eigenvalues of non-symmetric matrices can be ill-conditioned
A_sym = np.array([[2, 1], [1, 3]])
A_nonsym = np.array([[1, 1000], [0, 2]])

print(f"\nSymmetric: cond = {np.linalg.cond(A_sym):.2f}")
print(f"Non-symmetric: cond = {np.linalg.cond(A_nonsym):.2f}")
```

---

## 4. Numerical Stability

### 4.1 Forward and Backward Stability

An algorithm is:

- **Forward stable** if the computed result $\hat{y}$ satisfies $\|\hat{y} - y\| / \|y\| = O(\epsilon_{\text{mach}})$ for the exact input.
- **Backward stable** if the computed result is the exact answer to a slightly perturbed problem: $\hat{y} = f(x + \delta x)$ where $\|\delta x\| / \|x\| = O(\epsilon_{\text{mach}})$.

Backward stability is the gold standard: it says the algorithm introduces errors no worse than those already present in the input data.

### 4.2 Example: Summation Algorithms

```python
def naive_sum(x):
    """Left-to-right summation. Error grows as O(n * eps)."""
    s = 0.0
    for xi in x:
        s += xi
    return s

def kahan_sum(x):
    """Kahan compensated summation. Error is O(eps) regardless of n."""
    s = 0.0
    c = 0.0  # Running compensation for lost low-order bits
    for xi in x:
        y = xi - c
        t = s + y
        c = (t - s) - y  # Algebraically zero, but captures rounding error
        s = t
    return s

def pairwise_sum(x):
    """Pairwise (recursive) summation. Error grows as O(log(n) * eps)."""
    n = len(x)
    if n <= 256:
        return sum(x)
    mid = n // 2
    return pairwise_sum(x[:mid]) + pairwise_sum(x[mid:])

# Test with a known-difficult case
np.random.seed(42)
n = 100000
# Mix of large and small numbers
x = np.concatenate([
    np.random.randn(n // 2) * 1e8,
    np.random.randn(n // 2) * 1e-8
])
np.random.shuffle(x)

# "True" sum using higher precision
true_sum = np.float64(np.sum(x.astype(np.float128)))

results = {
    "np.sum": np.sum(x),
    "naive_sum": naive_sum(x),
    "kahan_sum": kahan_sum(x),
    "pairwise_sum": pairwise_sum(x),
}

for name, result in results.items():
    rel_error = abs(result - true_sum) / abs(true_sum)
    print(f"{name:15s}: result = {result:.10e}, rel_error = {rel_error:.2e}")
```

---

## 5. Catastrophic Cancellation

### 5.1 What Is Catastrophic Cancellation?

**Catastrophic cancellation** occurs when subtracting two nearly equal numbers. Although each number may have full precision, their difference has far fewer significant digits.

```python
# Classic example: quadratic formula
# Solve x^2 - 200x + 1 = 0
a, b, c = 1.0, -200.0, 1.0

# Standard formula (suffers from cancellation for one root)
disc = np.sqrt(b**2 - 4*a*c)
x1_standard = (-b + disc) / (2*a)
x2_standard = (-b - disc) / (2*a)

# True roots (computed with higher precision)
# x = 100 +/- sqrt(9999) = 100 +/- 99.99500...
x1_true = 199.994999874993750
x2_true = 0.005000125006250

print("Standard quadratic formula:")
print(f"  x1 = {x1_standard:.15f} (error = {abs(x1_standard - x1_true):.2e})")
print(f"  x2 = {x2_standard:.15f} (error = {abs(x2_standard - x2_true):.2e})")

# Numerically stable alternative: use Vieta's formula for the small root
# x1 * x2 = c/a, so x2 = c / (a * x1)
x2_stable = c / (a * x1_standard)
print(f"\nStable formula:")
print(f"  x2 = {x2_stable:.15f} (error = {abs(x2_stable - x2_true):.2e})")
```

### 5.2 More Examples of Cancellation

```python
# Example 1: Computing variance
# Naive: Var = E[X^2] - (E[X])^2 can suffer from cancellation
np.random.seed(42)
x = np.random.randn(10000) + 1e8  # Large mean, small variance

# Naive formula
mean_x = np.mean(x)
var_naive = np.mean(x**2) - mean_x**2
var_stable = np.var(x)  # Uses centered formula: mean((x - mean)^2)

print(f"Naive variance:  {var_naive:.6f}")
print(f"Stable variance: {var_stable:.6f}")
print(f"Relative diff: {abs(var_naive - var_stable) / var_stable:.2e}")

# Example 2: exp(x) - 1 for small x
x_small = 1e-15
print(f"\nexp({x_small}) - 1:")
print(f"  Naive:  {np.exp(x_small) - 1:.6e}")  # Cancellation!
print(f"  expm1:  {np.expm1(x_small):.6e}")     # Numerically stable

# Example 3: log(1 + x) for small x
print(f"\nlog(1 + {x_small}):")
print(f"  Naive:  {np.log(1 + x_small):.6e}")   # Cancellation!
print(f"  log1p:  {np.log1p(x_small):.6e}")      # Numerically stable
```

### 5.3 Avoiding Cancellation in Linear Algebra

```python
# Cancellation in determinants
# det(A) can be catastrophically inaccurate for ill-conditioned matrices

# Wilkinson matrix: nearly singular
n = 10
A = np.eye(n)
A[:, -1] = 1
A[-1, :] = -1 * np.ones(n)
A[-1, -1] = 1

# Compare determinant methods
det_numpy = np.linalg.det(A)
sign, logdet = np.linalg.slogdet(A)  # log-space computation
det_slog = sign * np.exp(logdet)

print(f"np.linalg.det: {det_numpy:.6e}")
print(f"slogdet:       {det_slog:.6e}")
print(f"log|det|:      {logdet:.6f}")

# For large matrices, always use slogdet
print("\nFor products of many matrices, log-determinant avoids overflow:")
A = np.random.randn(100, 100)
sign, logdet = np.linalg.slogdet(A)
print(f"sign = {sign}, log|det| = {logdet:.2f}")
```

---

## 6. Stability of Fundamental Algorithms

### 6.1 Gaussian Elimination with Partial Pivoting

Without pivoting, Gaussian elimination can be unstable. **Partial pivoting** (swapping rows to put the largest entry on the diagonal) is backward stable in practice.

```python
from scipy.linalg import lu, solve

# Without pivoting: potential instability
A_bad = np.array([[1e-20, 1.0],
                  [1.0, 1.0]])
b = np.array([1.0, 2.0])

# LU with pivoting (default in scipy)
P, L, U = lu(A_bad)
print("LU with pivoting:")
print(f"P:\n{P}")
print(f"L:\n{L}")
print(f"U:\n{U}")

x = solve(A_bad, b)
print(f"Solution: {x}")
print(f"Residual: {np.linalg.norm(A_bad @ x - b):.2e}")

# Growth factor: ratio of largest entry in U to largest in A
growth = np.max(np.abs(U)) / np.max(np.abs(A_bad))
print(f"Growth factor: {growth:.2e}")
```

### 6.2 Classical vs Modified Gram-Schmidt

The Classical Gram-Schmidt (CGS) process loses orthogonality for ill-conditioned matrices. Modified Gram-Schmidt (MGS) is more stable, and Householder reflections are backward stable.

```python
def classical_gram_schmidt(A):
    """Classical Gram-Schmidt: numerically unstable."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R

def modified_gram_schmidt(A):
    """Modified Gram-Schmidt: better numerical stability."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy().astype(float)
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j]
        for i in range(j + 1, n):
            R[j, i] = Q[:, j] @ V[:, i]
            V[:, i] -= R[j, i] * Q[:, j]
    return Q, R

# Compare on ill-conditioned matrix
n = 50
# Construct an ill-conditioned matrix
singular_values = np.logspace(0, -12, n)
U, _ = np.linalg.qr(np.random.randn(n, n))
Vt, _ = np.linalg.qr(np.random.randn(n, n))
A = U @ np.diag(singular_values) @ Vt
print(f"Condition number: {np.linalg.cond(A):.2e}")

# Classical Gram-Schmidt
Q_cgs, R_cgs = classical_gram_schmidt(A)
orth_error_cgs = np.linalg.norm(Q_cgs.T @ Q_cgs - np.eye(n))

# Modified Gram-Schmidt
Q_mgs, R_mgs = modified_gram_schmidt(A)
orth_error_mgs = np.linalg.norm(Q_mgs.T @ Q_mgs - np.eye(n))

# Householder (NumPy's QR, backward stable)
Q_hh, R_hh = np.linalg.qr(A)
orth_error_hh = np.linalg.norm(Q_hh.T @ Q_hh - np.eye(n))

print(f"\nOrthogonality error (||Q^T Q - I||):")
print(f"  Classical GS:     {orth_error_cgs:.2e}")
print(f"  Modified GS:      {orth_error_mgs:.2e}")
print(f"  Householder (QR): {orth_error_hh:.2e}")
```

### 6.3 Householder Reflections

Householder reflections are the backbone of backward-stable QR factorization. A Householder reflector is an orthogonal matrix of the form:

$$H = I - 2\frac{\mathbf{v}\mathbf{v}^T}{\mathbf{v}^T\mathbf{v}}$$

```python
def householder_qr(A):
    """QR factorization using Householder reflections (backward stable)."""
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)

    for j in range(min(m - 1, n)):
        # Compute Householder vector
        x = R[j:, j]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * np.sign(x[0])
        v = x + e1
        v = v / np.linalg.norm(v)

        # Apply reflection: R[j:, j:] = R[j:, j:] - 2 v (v^T R[j:, j:])
        R[j:, j:] -= 2 * np.outer(v, v @ R[j:, j:])

        # Accumulate Q
        Q[j:, :] -= 2 * np.outer(v, v @ Q[j:, :])

    return Q.T, R

# Test
A = np.random.randn(5, 4)
Q, R = householder_qr(A)
print(f"||QR - A|| = {np.linalg.norm(Q @ R - A):.2e}")
print(f"||Q^T Q - I|| = {np.linalg.norm(Q.T @ Q - np.eye(5)):.2e}")
```

---

## 7. Mixed Precision and Error Analysis

### 7.1 Mixed Precision Arithmetic

Modern hardware supports multiple floating-point precisions. Using lower precision (float16, bfloat16) for parts of a computation can dramatically improve speed while maintaining accuracy through **iterative refinement**.

```python
# Effect of precision on linear algebra
A = np.random.randn(100, 100)
b = np.random.randn(100)
x_true = np.linalg.solve(A, b)

for dtype in [np.float16, np.float32, np.float64]:
    A_d = A.astype(dtype)
    b_d = b.astype(dtype)

    if dtype == np.float16:
        # float16 does not have good LAPACK support; use manual solve
        x_d = np.linalg.solve(A_d.astype(np.float32), b_d.astype(np.float32))
    else:
        x_d = np.linalg.solve(A_d, b_d)

    rel_error = np.linalg.norm(x_d - x_true) / np.linalg.norm(x_true)
    residual = np.linalg.norm(A @ x_d.astype(np.float64) - b)
    print(f"{dtype.__name__:>10}: rel_error = {rel_error:.2e}, "
          f"residual = {residual:.2e}")
```

### 7.2 Iterative Refinement

Iterative refinement solves a system in low precision, then corrects the solution using the residual computed in high precision:

1. Solve $A\hat{x} = b$ in low precision
2. Compute residual $r = b - A\hat{x}$ in high precision
3. Solve $A\delta x = r$ in low precision
4. Update $\hat{x} \leftarrow \hat{x} + \delta x$
5. Repeat until convergence

```python
def iterative_refinement(A, b, solve_low, max_iter=10, tol=1e-14):
    """Iterative refinement using low-precision solve with high-precision residual."""
    x = solve_low(A, b)
    residuals = []

    for i in range(max_iter):
        r = b - A @ x  # High precision residual
        residual_norm = np.linalg.norm(r) / np.linalg.norm(b)
        residuals.append(residual_norm)

        if residual_norm < tol:
            print(f"Converged in {i+1} iterations")
            break

        dx = solve_low(A, r)
        x = x + dx

    return x, residuals

# Simulate low-precision solve by rounding
def solve_low_precision(A, b):
    """Simulate float32 solve."""
    return np.linalg.solve(A.astype(np.float32), b.astype(np.float32)).astype(np.float64)

np.random.seed(42)
A = np.random.randn(200, 200)
b = np.random.randn(200)

x_refined, residuals = iterative_refinement(A, b, solve_low_precision)
x_direct = np.linalg.solve(A, b)

print(f"\nDirect solve error:  {np.linalg.norm(A @ x_direct - b):.2e}")
print(f"Refined solve error: {np.linalg.norm(A @ x_refined - b):.2e}")

# Plot convergence
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.semilogy(residuals, 'bo-')
plt.xlabel('Iteration')
plt.ylabel('Relative residual')
plt.title('Iterative Refinement Convergence')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 8. Practical Guidelines

### 8.1 Rules of Thumb

1. **Never compare floats with `==`**. Use `np.isclose()` or `np.allclose()`.
2. **Check the condition number** before solving a linear system. If $\kappa \cdot \epsilon_{\text{mach}} \approx 1$, the answer is meaningless.
3. **Prefer stable algorithms**: Householder QR over Gram-Schmidt, pivoted LU over unpivoted.
4. **Avoid computing $A^{-1}$** explicitly. Solve $Ax = b$ instead.
5. **Use `slogdet` instead of `det`** for large matrices to avoid overflow.
6. **Use `expm1`, `log1p`** for expressions near cancellation boundaries.
7. **Scale your data** before computation to avoid overflow/underflow.

```python
# Rule 1: Never use == for floats
x = 0.1 + 0.2
print(f"0.1 + 0.2 == 0.3: {x == 0.3}")
print(f"np.isclose(0.1+0.2, 0.3): {np.isclose(x, 0.3)}")

# Rule 4: Solve Ax=b, don't compute A^{-1}b
A = np.random.randn(500, 500)
b = np.random.randn(500)

# BAD: x = A^{-1} b
x_inv = np.linalg.inv(A) @ b
res_inv = np.linalg.norm(A @ x_inv - b)

# GOOD: solve directly
x_solve = np.linalg.solve(A, b)
res_solve = np.linalg.norm(A @ x_solve - b)

print(f"\nUsing A^(-1)b: residual = {res_inv:.2e}")
print(f"Using solve:   residual = {res_solve:.2e}")

# Rule 7: Scaling
# Problem: compute x^T A x when x has entries of very different magnitude
x_unscaled = np.array([1e15, 1e-15])
A_small = np.array([[1, 0.5], [0.5, 1]])

# Without scaling (overflow risk)
Q_unscaled = x_unscaled @ A_small @ x_unscaled
print(f"\nUnscaled Q: {Q_unscaled:.6e}")

# With scaling
scale = np.linalg.norm(x_unscaled)
x_scaled = x_unscaled / scale
Q_scaled = (scale**2) * (x_scaled @ A_small @ x_scaled)
print(f"Scaled Q:   {Q_scaled:.6e}")
```

---

## Exercises

### Exercise 1: Machine Epsilon Exploration

1. Write a function that computes machine epsilon for `float32` using successive halving
2. Verify that `1.0 + eps/2` rounds to `1.0` but `1.0 + eps` does not (for your computed eps)
3. Compute the spacing between consecutive floats at $x = 1000$ and at $x = 0.001$. How do they differ?

### Exercise 2: Condition Number and Solution Accuracy

For Hilbert matrices of sizes $n = 2, 4, 6, 8, 10, 12, 14$:

1. Compute the condition number
2. Solve $Hx = b$ where $x_{\text{true}} = \mathbf{1}$ and $b = Hx_{\text{true}}$
3. Plot the relative error versus $n$ and overlay the theoretical bound $\kappa(H) \cdot \epsilon_{\text{mach}}$

### Exercise 3: Catastrophic Cancellation

The function $f(x) = \frac{1 - \cos(x)}{x^2}$ suffers from cancellation for small $x$. Rewrite it using a numerically stable formula and compare the two for $x = 10^{-k}$, $k = 1, \ldots, 16$.

### Exercise 4: Gram-Schmidt Comparison

Implement classical Gram-Schmidt, modified Gram-Schmidt, and use NumPy's Householder QR. For a matrix with condition number $10^{12}$, measure:

1. Orthogonality loss: $\|Q^TQ - I\|$
2. Factorization error: $\|QR - A\|$

Which method is most stable? Which is fastest?

### Exercise 5: Iterative Refinement

Solve a linear system with a moderately ill-conditioned matrix ($\kappa \approx 10^8$) using:

1. Direct solve in float32
2. Direct solve in float64
3. Float32 solve + iterative refinement (residuals in float64)

Compare the relative errors and residuals after each refinement step.

---

[Previous: Lesson 12](./12_Sparse_Matrices.md) | [Overview](./00_Overview.md) | [Next: Lesson 14](./14_Iterative_Methods.md)

**License**: CC BY-NC 4.0
