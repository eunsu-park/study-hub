# Numerical Analysis Basics

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain IEEE 754 floating-point representation and identify how round-off and truncation errors arise.
2. Define absolute error, relative error, and machine epsilon, and analyze their impact on numerical computations.
3. Implement numerical differentiation using finite difference approximations and assess their accuracy.
4. Apply numerical integration methods (trapezoidal rule, Simpson's rule) and compare their convergence rates.
5. Analyze the trade-off between step size and numerical error in differentiation and integration.

---

## Overview

Numerical analysis studies methods for approximately solving mathematical problems using computers. We will learn about floating-point representation, error analysis, numerical differentiation and integration, which form the foundation of simulation.

**Why This Lesson Matters:** Every numerical simulation is built on floating-point arithmetic, and every floating-point operation introduces a tiny error. These errors can accumulate, cancel, or amplify in ways that produce wildly incorrect results. Understanding how computers represent numbers, where errors come from, and how to control them is not optional -- it is the difference between a simulation that gives trustworthy predictions and one that silently produces garbage.

---

## 1. Floating-Point Representation

### 1.1 IEEE 754 Standard

```python
import numpy as np
import struct

# Check floating-point bit representation
def float_to_bits(f):
    """Convert float64 to bit string"""
    packed = struct.pack('>d', f)
    bits = ''.join(f'{b:08b}' for b in packed)
    return f"Sign: {bits[0]} | Exponent: {bits[1:12]} | Mantissa: {bits[12:]}"

print(float_to_bits(1.0))
print(float_to_bits(-1.0))
print(float_to_bits(0.1))

# Machine epsilon
print(f"\nfloat64 machine epsilon: {np.finfo(np.float64).eps}")
print(f"float32 machine epsilon: {np.finfo(np.float32).eps}")
```

### 1.2 Numerical Limits

```python
# Overflow and underflow
print("float64 range:")
print(f"  Minimum: {np.finfo(np.float64).min}")
print(f"  Maximum: {np.finfo(np.float64).max}")
print(f"  Smallest positive: {np.finfo(np.float64).tiny}")

# Precision loss example
a = 1e16
b = 1.0
print(f"\n1e16 + 1 - 1e16 = {(a + b) - a}")  # 0.0 (precision loss)
print(f"1 + 1e16 - 1e16 = {b + (a - a)}")    # 1.0 (correct result)
```

### 1.3 Rounding Error

```python
# 0.1 cannot be represented exactly
x = 0.1
print(f"0.1 actual value: {x:.20f}")
print(f"0.1 + 0.2 = 0.3? {0.1 + 0.2 == 0.3}")  # False

# Use tolerance for comparison
print(f"np.isclose: {np.isclose(0.1 + 0.2, 0.3)}")
```

---

## 2. Error Analysis

### 2.1 Error Types

```python
def analyze_error(true_value, approx_value):
    """Calculate absolute and relative error"""
    abs_error = abs(true_value - approx_value)
    rel_error = abs_error / abs(true_value) if true_value != 0 else float('inf')
    return abs_error, rel_error

# Example: π approximation
import math
approximations = [
    ("22/7", 22/7),
    ("355/113", 355/113),
    ("3.14159", 3.14159),
]

print("π approximation error analysis:")
for name, approx in approximations:
    abs_e, rel_e = analyze_error(math.pi, approx)
    print(f"  {name:10}: Absolute error={abs_e:.2e}, Relative error={rel_e:.2e}")
```

### 2.2 Numerical Stability

```python
# Unstable computation example: small difference from large number
def unstable_subtract(x):
    """Numerically unstable subtraction"""
    return (1 + x) - 1

def stable_subtract(x):
    """Numerically stable form"""
    return x

x_values = [1e-15, 1e-16, 1e-17]
print("Small number subtraction comparison:")
for x in x_values:
    print(f"  x={x}: Unstable={unstable_subtract(x):.2e}, Stable={stable_subtract(x):.2e}")
```

### 2.3 Condition Number

```python
# Matrix condition number
def analyze_condition_number():
    # Well-conditioned matrix
    A_good = np.array([[1, 0], [0, 1]])

    # Poorly-conditioned matrix
    A_bad = np.array([[1, 1], [1, 1.0001]])

    print("Condition number analysis:")
    print(f"  Identity matrix: {np.linalg.cond(A_good):.2f}")
    print(f"  Nearly singular matrix: {np.linalg.cond(A_bad):.2f}")

analyze_condition_number()
```

---

## 3. Numerical Differentiation

Differentiation is the foundation of all PDE solvers. We approximate derivatives using values at discrete grid points. The three main formulas differ in accuracy and the points they use:

- **Forward difference**: $f'(x) \approx \frac{f(x+h) - f(x)}{h} + O(h)$ -- first-order accurate
- **Backward difference**: $f'(x) \approx \frac{f(x) - f(x-h)}{h} + O(h)$ -- first-order accurate
- **Central difference**: $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} + O(h^2)$ -- second-order accurate

The central difference is more accurate because the first-order error terms from the Taylor expansion cancel by symmetry.

### 3.1 Finite Difference Method

```python
def numerical_derivatives(f, x, h=1e-5):
    """Various finite difference formulas"""
    # Forward difference: uses f at x and x+h only (one-sided, O(h))
    forward = (f(x + h) - f(x)) / h

    # Backward difference: uses f at x and x-h only (one-sided, O(h))
    backward = (f(x) - f(x - h)) / h

    # Central difference: symmetric about x, error terms cancel → O(h²)
    central = (f(x + h) - f(x - h)) / (2 * h)

    return forward, backward, central

# Test: f(x) = sin(x), f'(x) = cos(x)
x = np.pi / 4
true_deriv = np.cos(x)

forward, backward, central = numerical_derivatives(np.sin, x)

print(f"Derivative of sin(x) at x = π/4:")
print(f"  True value: {true_deriv:.10f}")
print(f"  Forward difference: {forward:.10f}, Error: {abs(forward - true_deriv):.2e}")
print(f"  Backward difference: {backward:.10f}, Error: {abs(backward - true_deriv):.2e}")
print(f"  Central difference: {central:.10f}, Error: {abs(central - true_deriv):.2e}")
```

### 3.2 Higher-Order Derivatives

```python
def second_derivative(f, x, h=1e-5):
    """Second derivative (central difference)"""
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2

# f(x) = sin(x), f''(x) = -sin(x)
x = np.pi / 4
true_second = -np.sin(x)
approx_second = second_derivative(np.sin, x)

print(f"\nSecond derivative:")
print(f"  True value: {true_second:.10f}")
print(f"  Approximation: {approx_second:.10f}")
print(f"  Error: {abs(approx_second - true_second):.2e}")
```

### 3.3 Effect of Step Size

There is a fundamental tension between truncation error (decreases with smaller $h$) and rounding error (increases with smaller $h$ because we subtract nearly equal numbers). The optimal step size balances these two: for the central difference, $h_{\text{opt}} \approx \varepsilon_{\text{mach}}^{1/3} \approx 6 \times 10^{-6}$ for float64.

```python
def analyze_step_size():
    """Error analysis based on step size"""
    f = np.sin
    x = 1.0
    true_deriv = np.cos(x)

    h_values = np.logspace(-1, -15, 15)
    errors = []

    for h in h_values:
        central = (f(x + h) - f(x - h)) / (2 * h)
        errors.append(abs(central - true_deriv))

    return h_values, errors

h_values, errors = analyze_step_size()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors, 'bo-')
plt.xlabel('Step size h')
plt.ylabel('Error')
plt.title('Central Difference Error vs Step Size')
plt.grid(True)
plt.axvline(x=1e-8, color='r', linestyle='--', label='Near optimal')
plt.legend()
plt.show()
# Too small h: increased rounding error
# Too large h: increased truncation error
```

---

## 4. Numerical Integration

Numerical integration (quadrature) is needed whenever an integral cannot be evaluated analytically. The basic idea: approximate the integrand by a polynomial on each subinterval and integrate the polynomial exactly. Higher-order polynomials give faster convergence: the trapezoidal rule uses linear interpolation ($O(h^2)$), Simpson's rule uses quadratic interpolation ($O(h^4)$), and Gauss quadrature achieves even higher orders.

### 4.1 Trapezoidal Rule

The trapezoidal rule approximates the area under the curve by connecting adjacent points with straight lines. On each subinterval $[x_i, x_{i+1}]$, the area is $h \cdot (f_i + f_{i+1})/2$, like a trapezoid.

```python
def trapezoidal(f, a, b, n):
    """Integration using trapezoidal rule"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Trapezoidal rule
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return integral

# Test: ∫₀^π sin(x) dx = 2
result = trapezoidal(np.sin, 0, np.pi, 100)
print(f"∫₀^π sin(x) dx:")
print(f"  True value: 2.0")
print(f"  Trapezoidal (n=100): {result:.10f}")
print(f"  Error: {abs(result - 2.0):.2e}")
```

### 4.2 Simpson's Rule

Simpson's rule fits a quadratic through each group of three consecutive points, achieving $O(h^4)$ convergence -- two orders better than the trapezoidal rule for the same number of points. This dramatic improvement comes because quadratics exactly integrate cubic polynomials (a happy cancellation in the error analysis).

```python
def simpson(f, a, b, n):
    """Simpson's 1/3 rule (n must be even)"""
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Simpson's rule: (h/3) * [y₀ + 4y₁ + 2y₂ + 4y₃ + ... + yₙ]
    integral = h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])
    return integral

result_trap = trapezoidal(np.sin, 0, np.pi, 10)
result_simp = simpson(np.sin, 0, np.pi, 10)

print(f"\nComparison at n=10:")
print(f"  Trapezoidal: {result_trap:.10f}, Error: {abs(result_trap - 2.0):.2e}")
print(f"  Simpson: {result_simp:.10f}, Error: {abs(result_simp - 2.0):.2e}")
```

### 4.3 SciPy Integration

```python
from scipy import integrate

# 1D integration
result, error = integrate.quad(np.sin, 0, np.pi)
print(f"\nscipy.integrate.quad:")
print(f"  Result: {result:.15f}")
print(f"  Estimated error: {error:.2e}")

# 2D integration
def f_2d(y, x):
    return x * y

result_2d, error_2d = integrate.dblquad(f_2d, 0, 1, 0, 1)
print(f"\n∫∫ xy dxdy (0~1):")
print(f"  Result: {result_2d:.10f}")  # 0.25
```

### 4.4 Convergence Analysis

```python
def convergence_analysis():
    """Integration convergence analysis"""
    true_value = 2.0  # ∫₀^π sin(x) dx
    n_values = [4, 8, 16, 32, 64, 128, 256]

    trap_errors = []
    simp_errors = []

    for n in n_values:
        trap_errors.append(abs(trapezoidal(np.sin, 0, np.pi, n) - true_value))
        simp_errors.append(abs(simpson(np.sin, 0, np.pi, n) - true_value))

    # Estimate convergence order
    print("Convergence analysis:")
    print(f"{'n':>6} {'Trapezoidal':>12} {'Simpson':>12}")
    for i, n in enumerate(n_values):
        print(f"{n:>6} {trap_errors[i]:>12.2e} {simp_errors[i]:>12.2e}")

    # Trapezoidal: O(h²), Simpson: O(h⁴)
    return n_values, trap_errors, simp_errors

convergence_analysis()
```

---

## 5. Practice Problems

### Problem 1: Numerical Differentiation
Calculate the derivative of f(x) = e^(-x²) at x=0.5 with various step sizes and analyze the error.

```python
def exercise_1():
    f = lambda x: np.exp(-x**2)
    f_prime = lambda x: -2*x * np.exp(-x**2)  # Analytical derivative

    x = 0.5
    true_value = f_prime(x)

    # Solution
    h_values = np.logspace(-1, -12, 12)
    for h in h_values:
        approx = (f(x + h) - f(x - h)) / (2 * h)
        print(f"h={h:.0e}: Error={abs(approx - true_value):.2e}")

exercise_1()
```

### Problem 2: Numerical Integration
Calculate ∫₀^1 e^(-x²) dx using trapezoidal and Simpson's rules.

```python
def exercise_2():
    f = lambda x: np.exp(-x**2)

    # scipy reference value
    true_val, _ = integrate.quad(f, 0, 1)

    # Solution
    for n in [10, 50, 100]:
        trap = trapezoidal(f, 0, 1, n)
        simp = simpson(f, 0, 1, n)
        print(f"n={n}: Trapezoidal={trap:.8f}, Simpson={simp:.8f}")

    print(f"True value: {true_val:.8f}")

exercise_2()
```

---

## Summary

| Concept | Key Content |
|------|----------|
| Floating-point | IEEE 754, machine epsilon, precision limits |
| Error types | Truncation error, rounding error, condition number |
| Numerical differentiation | Forward/backward/central difference, step size selection |
| Numerical integration | Trapezoidal (O(h²)), Simpson (O(h⁴)) |

## Exercises

### Exercise 1: Machine Epsilon and Precision Loss

Explain what machine epsilon (ε_mach) represents for `float64` and `float32`. Then demonstrate the catastrophic cancellation problem: compute `(a + b) - a` for `a = 1e15` and `b = 1.0`, explain why the result is incorrect, and show how to rewrite the expression to avoid the issue.

<details>
<summary>Show Answer</summary>

Machine epsilon is the smallest positive number ε such that `1.0 + ε ≠ 1.0` in floating-point arithmetic. It bounds the relative rounding error of a single operation.

```python
import numpy as np

# Machine epsilon values
print(f"float64 machine epsilon: {np.finfo(np.float64).eps:.2e}")  # ~2.22e-16
print(f"float32 machine epsilon: {np.finfo(np.float32).eps:.2e}")  # ~1.19e-07

# Catastrophic cancellation
a = 1e15
b = 1.0

# Incorrect: a and b differ by 15 orders of magnitude
result_bad = (a + b) - a
print(f"\n(a + b) - a = {result_bad}")   # 0.0  -- precision of b is lost

# Correct: subtract before adding
result_good = b + (a - a)
print(f"b + (a - a) = {result_good}")    # 1.0  -- mathematically equivalent, numerically stable

# When a >> b, use stable_form = b directly
# The key insight: reorder operations to avoid adding small to large
```

The root cause: when `a` is stored as a float64, it has only about 15-16 significant decimal digits. Adding `b = 1.0` to `a = 1e15` requires aligning decimal points, pushing `b`'s bits below the representable range, so `b` is lost entirely.

</details>

### Exercise 2: Forward vs. Central Difference Error Scaling

For `f(x) = cos(x)` at `x = 0.7`, compute numerical derivatives using the forward difference (`O(h)`) and central difference (`O(h²)`) formulas at step sizes `h = 10⁻¹, 10⁻³, 10⁻⁵, 10⁻⁷`. For each method, compute the error relative to the analytical value `f'(x) = -sin(x)` and verify that the error ratios match the expected convergence orders when `h` is reduced by a factor of 100.

<details>
<summary>Show Answer</summary>

```python
import numpy as np

x = 0.7
true_val = -np.sin(x)
h_values = [1e-1, 1e-3, 1e-5, 1e-7]

print(f"True derivative: {true_val:.10f}\n")
print(f"{'h':>8}  {'Forward error':>14}  {'Central error':>14}")
print("-" * 42)

prev_fwd, prev_cen = None, None
for h in h_values:
    fwd  = (np.cos(x + h) - np.cos(x)) / h
    cen  = (np.cos(x + h) - np.cos(x - h)) / (2 * h)
    e_fwd = abs(fwd - true_val)
    e_cen = abs(cen - true_val)
    print(f"{h:8.0e}  {e_fwd:14.2e}  {e_cen:14.2e}", end="")
    if prev_fwd is not None:
        # Ratio should be ~100 (O(h)) and ~10000 (O(h²)) when h shrinks 100x
        print(f"  [fwd ratio: {prev_fwd/e_fwd:.0f}, cen ratio: {prev_cen/e_cen:.0f}]", end="")
    print()
    prev_fwd, prev_cen = e_fwd, e_cen
```

Expected output pattern:
- Forward difference error ratios ≈ 100 (error shrinks linearly with h → O(h))
- Central difference error ratios ≈ 10000 (error shrinks quadratically with h → O(h²))

Note: at very small h (≈ 1e-7 for forward, ≈ 1e-5 for central) rounding errors begin to dominate and the ratios break down.

</details>

### Exercise 3: Composite Simpson's Rule Convergence

Implement composite Simpson's rule and numerically verify its O(h⁴) convergence order by computing `∫₀^1 x³ e^x dx` for n = 4, 8, 16, 32, 64 subintervals. The exact value is `e - 2 ≈ 0.71828...`. Compute the estimated convergence order from successive error ratios.

<details>
<summary>Show Answer</summary>

```python
import numpy as np
from scipy.integrate import quad

def simpson(f, a, b, n):
    """Composite Simpson's 1/3 rule (n must be even)."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])

f = lambda x: x**3 * np.exp(x)
exact, _ = quad(f, 0, 1)   # ≈ e - 2

print(f"Exact value: {exact:.10f}")
print(f"\n{'n':>6}  {'Approx':>14}  {'Error':>12}  {'Order':>8}")
print("-" * 46)

prev_error = None
for n in [4, 8, 16, 32, 64]:
    approx = simpson(f, 0, 1, n)
    error  = abs(approx - exact)
    if prev_error is not None:
        order = np.log2(prev_error / error)
        print(f"{n:>6}  {approx:14.10f}  {error:12.2e}  {order:8.2f}")
    else:
        print(f"{n:>6}  {approx:14.10f}  {error:12.2e}  {'—':>8}")
    prev_error = error
```

The convergence orders printed should be close to **4.0**, confirming O(h⁴) = O((1/n)⁴). Each doubling of n reduces the error by a factor of ~16.

</details>

### Exercise 4: Condition Number and Linear System Sensitivity

Create a nearly-singular 2×2 matrix with condition number > 10⁵ and solve `Ax = b` twice: once with the exact `b` and once with `b` perturbed by a small vector of magnitude ≈ 10⁻⁶. Show that the relative error in the solution can be much larger than the relative error in `b`, quantify this amplification using the condition number bound, and verify with NumPy.

<details>
<summary>Show Answer</summary>

```python
import numpy as np

# Hilbert matrix (notoriously ill-conditioned)
n = 5
A = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
cond = np.linalg.cond(A)
print(f"Condition number of 5x5 Hilbert matrix: {cond:.2e}")

# Exact right-hand side: x_true = [1, 1, 1, 1, 1]
x_true = np.ones(n)
b_exact = A @ x_true

# Perturbed right-hand side
rng = np.random.default_rng(0)
delta_b = rng.standard_normal(n)
delta_b *= 1e-6 / np.linalg.norm(delta_b)   # ||δb|| ≈ 1e-6

b_perturbed = b_exact + delta_b

# Solve both systems
x_exact     = np.linalg.solve(A, b_exact)
x_perturbed = np.linalg.solve(A, b_perturbed)

rel_b_error = np.linalg.norm(delta_b) / np.linalg.norm(b_exact)
rel_x_error = np.linalg.norm(x_perturbed - x_exact) / np.linalg.norm(x_exact)

print(f"\nRelative error in b:  {rel_b_error:.2e}")
print(f"Relative error in x:  {rel_x_error:.2e}")
print(f"Amplification factor: {rel_x_error / rel_b_error:.1f}")
print(f"Condition number:     {cond:.2e}")
print(f"\nTheory: ||δx||/||x|| ≤ κ(A) * ||δb||/||b||")
print(f"Bound:  {cond * rel_b_error:.2e}  (actual: {rel_x_error:.2e})")
```

The relative error in `x` can be up to `κ(A)` times larger than the relative error in `b`. For the 5×5 Hilbert matrix, κ ≈ 5×10⁵, so a perturbation of magnitude 10⁻⁶ in `b` can produce an error of order 10⁻¹ in `x`.

</details>

### Exercise 5: Optimal Step Size for Central Difference

For `f(x) = sin(x)` at `x = 1.0`, use central differences with step sizes ranging from `h = 10⁻¹` to `h = 10⁻¹⁵` and plot error vs. h on a log-log scale. Identify the approximate optimal step size where truncation error and rounding error balance, and derive the theoretical optimal `h_opt ≈ (ε_mach)^(1/3)`.

<details>
<summary>Show Answer</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

f      = np.sin
x      = 1.0
f_true = np.cos(x)   # exact derivative

h_values = np.logspace(-1, -15, 150)
errors   = [abs((f(x + h) - f(x - h)) / (2 * h) - f_true) for h in h_values]

# Theoretical analysis:
# Truncation error  ≈ h² |f'''(x)| / 6   → grows with h²
# Rounding error    ≈ ε_mach |f(x)| / h  → grows as 1/h
# Optimal balance:  h_opt ≈ (ε_mach)^(1/3) ≈ 6e-6

eps   = np.finfo(float).eps
h_opt = eps ** (1/3)
print(f"Machine epsilon:  {eps:.2e}")
print(f"Theoretical h_opt ≈ ε^(1/3) = {h_opt:.2e}")

plt.figure(figsize=(9, 5))
plt.loglog(h_values, errors, 'b-', linewidth=1.5, label='Actual error')
plt.axvline(h_opt, color='r', linestyle='--', label=f'Theoretical h_opt ≈ {h_opt:.0e}')
plt.xlabel('Step size h')
plt.ylabel('Absolute error')
plt.title('Central Difference Error vs Step Size')
plt.legend()
plt.grid(True, which='both', alpha=0.4)
plt.show()

# Observed minimum
min_idx = np.argmin(errors)
print(f"Observed optimal h ≈ {h_values[min_idx]:.2e}  (min error: {errors[min_idx]:.2e})")
```

The error curve shows two regimes:
- **Left of minimum** (large h): dominated by truncation error, slope ≈ +2 on log-log plot.
- **Right of minimum** (small h): dominated by rounding error, slope ≈ −1 on log-log plot.

The theoretical optimal step is `h_opt = (ε_mach)^{1/3} ≈ 6×10⁻⁶`, giving a minimum error near `ε_mach^{2/3} ≈ 4×10⁻¹¹`.

</details>
