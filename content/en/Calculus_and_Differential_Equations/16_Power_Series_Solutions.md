# Power Series Solutions of ODE

## Learning Objectives

- Classify singular points of an ODE as ordinary, regular singular, or irregular singular
- Construct power series solutions near ordinary points by finding recurrence relations for coefficients
- Apply the Frobenius method to find series solutions near regular singular points
- Recognize how Bessel's equation and Legendre's equation arise and connect series solutions to special functions
- Verify series solutions numerically using Python and compare with known special functions

## Prerequisites

Before studying this lesson, you should be comfortable with:
- Power series, radius of convergence, Taylor series (Lessons 3-4)
- Second-order linear ODE (Lessons 10-12)
- Laplace transform methods as an alternative approach (Lesson 15)

## Motivation: When Standard Methods Fail

For constant-coefficient ODE, we found solutions using the characteristic equation. For certain non-constant coefficient equations, the Laplace transform or variation of parameters works. But many important equations in physics and engineering have **variable coefficients** that resist all these methods:

$$x^2 y'' + xy' + (x^2 - n^2)y = 0 \quad \text{(Bessel's equation)}$$

$$(1 - x^2)y'' - 2xy' + \ell(\ell+1)y = 0 \quad \text{(Legendre's equation)}$$

$$xy'' + (1-x)y' + ny = 0 \quad \text{(Laguerre's equation)}$$

These equations appear everywhere: vibrating circular membranes (drums), gravitational potentials, quantum mechanics. The **power series method** lets us find solutions by assuming $y = \sum a_n x^n$ and determining the coefficients.

## Ordinary and Singular Points

Consider the standard form of a second-order linear ODE:

$$y'' + P(x)y' + Q(x)y = 0$$

**Ordinary point**: $x = x_0$ is an ordinary point if both $P(x)$ and $Q(x)$ are analytic (have convergent Taylor series) at $x_0$.

**Singular point**: $x = x_0$ is a singular point if $P(x)$ or $Q(x)$ is not analytic at $x_0$.

For the equation written as $a_2(x)y'' + a_1(x)y' + a_0(x)y = 0$, divide by $a_2(x)$ to get standard form. Singular points occur where $a_2(x_0) = 0$.

### Regular vs Irregular Singular Points

A singular point $x_0$ is **regular** if:

$$(x - x_0)P(x) \quad \text{and} \quad (x - x_0)^2 Q(x)$$

are both analytic at $x_0$. Otherwise, it is an **irregular** singular point.

**Why this matters**: At a regular singular point, the Frobenius method guarantees at least one series solution. At an irregular singular point, no such guarantee exists.

**Example**: Classify the singular points of $x^2 y'' + xy' + (x^2 - n^2)y = 0$ (Bessel's equation).

Standard form: $y'' + \frac{1}{x}y' + \frac{x^2 - n^2}{x^2}y = 0$

So $P(x) = 1/x$ and $Q(x) = (x^2 - n^2)/x^2$. The singular point is $x = 0$.

Check: $xP(x) = 1$ (analytic) and $x^2 Q(x) = x^2 - n^2$ (analytic). Therefore $x = 0$ is a **regular singular point**.

## Power Series Solution Near Ordinary Points

### The Method

If $x_0$ is an ordinary point, assume a solution of the form:

$$y = \sum_{n=0}^{\infty} a_n (x - x_0)^n$$

The radius of convergence is at least the distance from $x_0$ to the nearest singular point in the complex plane.

### Step-by-Step Procedure

1. Assume $y = \sum_{n=0}^{\infty} a_n x^n$ (taking $x_0 = 0$ for simplicity)
2. Compute $y' = \sum_{n=1}^{\infty} n a_n x^{n-1}$ and $y'' = \sum_{n=2}^{\infty} n(n-1) a_n x^{n-2}$
3. Substitute into the ODE
4. Shift indices so all sums start at the same power of $x$
5. Collect coefficients of each power $x^k$ and set them to zero
6. Solve the resulting **recurrence relation** for $a_n$

### Worked Example: Airy's Equation

Solve $y'' - xy = 0$ near $x_0 = 0$.

$x = 0$ is an ordinary point (no singular points for this equation).

**Step 1-2**: Let $y = \sum_{n=0}^{\infty} a_n x^n$, so $y'' = \sum_{n=2}^{\infty} n(n-1)a_n x^{n-2}$.

**Step 3**: Substitute:

$$\sum_{n=2}^{\infty} n(n-1)a_n x^{n-2} - x \sum_{n=0}^{\infty} a_n x^n = 0$$

$$\sum_{n=2}^{\infty} n(n-1)a_n x^{n-2} - \sum_{n=0}^{\infty} a_n x^{n+1} = 0$$

**Step 4**: Shift indices. In the first sum, let $m = n - 2$ (so $n = m + 2$):

$$\sum_{m=0}^{\infty} (m+2)(m+1)a_{m+2} x^{m} - \sum_{n=0}^{\infty} a_n x^{n+1} = 0$$

In the second sum, let $m = n + 1$ (so $n = m - 1$):

$$\sum_{m=0}^{\infty} (m+2)(m+1)a_{m+2} x^{m} - \sum_{m=1}^{\infty} a_{m-1} x^{m} = 0$$

**Step 5**: The $m = 0$ term from the first sum gives $2 \cdot 1 \cdot a_2 = 0$, so $a_2 = 0$.

For $m \geq 1$: $(m+2)(m+1)a_{m+2} - a_{m-1} = 0$.

**Step 6**: The recurrence relation:

$$a_{m+2} = \frac{a_{m-1}}{(m+2)(m+1)}, \quad m \geq 1$$

This connects $a_{m+2}$ to $a_{m-1}$, stepping by 3. So we get three independent chains:

- Chain starting from $a_0$: $a_0, a_3, a_6, a_9, \ldots$
- Chain starting from $a_1$: $a_1, a_4, a_7, a_{10}, \ldots$
- Chain starting from $a_2 = 0$: $a_2 = a_5 = a_8 = \cdots = 0$

Computing:
- $a_3 = \frac{a_0}{3 \cdot 2}$, $a_6 = \frac{a_3}{6 \cdot 5} = \frac{a_0}{6 \cdot 5 \cdot 3 \cdot 2}$
- $a_4 = \frac{a_1}{4 \cdot 3}$, $a_7 = \frac{a_4}{7 \cdot 6} = \frac{a_1}{7 \cdot 6 \cdot 4 \cdot 3}$

The general solution is $y = a_0 y_1(x) + a_1 y_2(x)$ where $y_1$ and $y_2$ are the two linearly independent Airy functions.

## Frobenius Method (Regular Singular Points)

When $x_0$ is a **regular singular** point, the power series method may fail (the first coefficient could be zero). The **Frobenius method** generalizes the approach by assuming:

$$y = \sum_{n=0}^{\infty} a_n (x - x_0)^{n+r}$$

where $r$ is an unknown exponent to be determined. The key difference: the exponent $r$ need not be an integer.

### The Indicial Equation

Substituting the Frobenius series into the ODE and collecting the lowest power of $x$ yields the **indicial equation** -- a quadratic in $r$:

$$r(r - 1) + p_0 r + q_0 = 0$$

where $p_0 = \lim_{x \to x_0} (x - x_0)P(x)$ and $q_0 = \lim_{x \to x_0} (x - x_0)^2 Q(x)$.

The two roots $r_1 \geq r_2$ determine the form of the solutions:

| Case | Condition | Solutions |
|------|-----------|-----------|
| Distinct, non-integer difference | $r_1 - r_2 \notin \mathbb{Z}$ | Two Frobenius series |
| Equal roots | $r_1 = r_2$ | One Frobenius series + logarithmic solution |
| Integer difference | $r_1 - r_2 \in \mathbb{Z}^+$ | First solution with $r_1$; second may need logarithm |

### Worked Example: Bessel's Equation of Order Zero

$$x^2 y'' + xy' + x^2 y = 0$$

Divide by $x^2$: $y'' + \frac{1}{x}y' + y = 0$. Here $x = 0$ is a regular singular point.

$p_0 = \lim_{x \to 0} x \cdot (1/x) = 1$, $q_0 = \lim_{x \to 0} x^2 \cdot 1 = 0$.

**Indicial equation**: $r(r-1) + r + 0 = r^2 = 0$, so $r = 0$ (double root).

This means one solution is a Frobenius series with $r = 0$ (which is actually a regular power series), and the second solution involves a logarithmic term. The first solution is the famous **Bessel function** $J_0(x)$:

$$J_0(x) = \sum_{m=0}^{\infty} \frac{(-1)^m}{(m!)^2} \left(\frac{x}{2}\right)^{2m} = 1 - \frac{x^2}{4} + \frac{x^4}{64} - \cdots$$

## Introduction to Special Functions

Many of the most important functions in mathematics and physics arise as series solutions of specific ODE.

### Bessel Functions

Bessel's equation of order $\nu$:

$$x^2 y'' + xy' + (x^2 - \nu^2)y = 0$$

The Frobenius method gives $J_\nu(x)$, the **Bessel function of the first kind**. These functions:
- Describe vibrations of circular membranes (drum modes)
- Appear in cylindrical coordinate solutions of the wave equation
- Model diffraction patterns through circular apertures

### Legendre Polynomials

Legendre's equation:

$$(1 - x^2)y'' - 2xy' + \ell(\ell+1)y = 0$$

When $\ell$ is a non-negative integer, one solution is a **polynomial** $P_\ell(x)$:
- $P_0(x) = 1$, $P_1(x) = x$, $P_2(x) = \frac{1}{2}(3x^2 - 1)$

These polynomials:
- Describe gravitational and electric potentials in spherical coordinates
- Form the angular part of hydrogen atom wavefunctions (with spherical harmonics)
- Are the basis for Gauss-Legendre quadrature in numerical integration

For comprehensive coverage of these special functions, see [Mathematical Methods - Special Functions](../Mathematical_Methods/11_Special_Functions.md).

## Python Implementation

```python
"""
Power Series Solutions of ODE.

This script demonstrates:
1. Computing series coefficients via recurrence relations (Airy equation)
2. Frobenius method for Bessel's equation
3. Comparing series approximations with scipy.special functions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy, jv  # Airy functions and Bessel J_v
from scipy.integrate import solve_ivp


# ── 1. Airy Equation: y'' - xy = 0 ──────────────────────────
def airy_series_coefficients(N, a0=1.0, a1=0.0):
    """
    Compute the first N coefficients of the power series solution
    to y'' - xy = 0.

    The recurrence is: a_{m+2} = a_{m-1} / ((m+2)(m+1)) for m >= 1
    with a_2 = 0 always.

    Parameters:
        N: number of coefficients to compute
        a0, a1: initial coefficients (free parameters)

    Returns:
        array of coefficients [a_0, a_1, ..., a_{N-1}]
    """
    a = np.zeros(N)
    a[0] = a0
    if N > 1:
        a[1] = a1
    # a[2] = 0 is already set by np.zeros

    # Apply recurrence: a_{m+2} = a_{m-1} / ((m+2)(m+1))
    for m in range(1, N - 2):
        a[m + 2] = a[m - 1] / ((m + 2) * (m + 1))

    return a


def evaluate_power_series(coeffs, x_vals):
    """Evaluate y = sum(a_n * x^n) at given x values."""
    result = np.zeros_like(x_vals)
    for n, a_n in enumerate(coeffs):
        result += a_n * x_vals**n
    return result


# Compare our series with scipy's Airy functions
N_terms = 30  # Number of series terms (more terms = better approximation)
x_range = np.linspace(-10, 5, 500)

# Ai(x) corresponds to a specific linear combination of a0 and a1
# scipy returns (Ai, Ai', Bi, Bi')
Ai, Ai_prime, Bi, Bi_prime = airy(x_range)

# Our series approximation for y_1 (a0=1, a1=0) and y_2 (a0=0, a1=1)
coeffs_y1 = airy_series_coefficients(N_terms, a0=1.0, a1=0.0)
coeffs_y2 = airy_series_coefficients(N_terms, a0=0.0, a1=1.0)

y1_series = evaluate_power_series(coeffs_y1, x_range)
y2_series = evaluate_power_series(coeffs_y2, x_range)

# Ai(x) is a specific combination: Ai(x) = c1*y1 + c2*y2
# At x=0: Ai(0) = 1/(3^{2/3} Gamma(2/3)) and Ai'(0) = -1/(3^{1/3} Gamma(1/3))
from scipy.special import gamma
c1 = 1.0 / (3**(2/3) * gamma(2/3))  # coefficient for y1
c2 = -1.0 / (3**(1/3) * gamma(1/3))  # coefficient for y2
Ai_series = c1 * y1_series + c2 * y2_series

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Airy function comparison
axes[0].plot(x_range, Ai, 'b-', linewidth=2, label='Ai(x) [scipy]')
axes[0].plot(x_range, Ai_series, 'r--', linewidth=1.5,
             label=f'Series ({N_terms} terms)')
axes[0].set_xlim(-10, 5)
axes[0].set_ylim(-0.6, 0.6)
axes[0].set_xlabel('x')
axes[0].set_ylabel('Ai(x)')
axes[0].set_title("Airy Function: Series vs scipy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── 2. Bessel Function J_0(x) ────────────────────────────────
def bessel_J0_series(x, N_terms=20):
    """
    Compute J_0(x) from its power series:
    J_0(x) = sum_{m=0}^{N} (-1)^m / (m!)^2 * (x/2)^{2m}

    This series comes from the Frobenius method applied to
    Bessel's equation with nu=0 and indicial root r=0.
    """
    result = np.zeros_like(x, dtype=float)
    for m in range(N_terms):
        # Each term: (-1)^m * (x/2)^{2m} / (m!)^2
        term = ((-1)**m / (np.math.factorial(m))**2) * (x / 2)**(2*m)
        result += term
    return result


x_bessel = np.linspace(0, 20, 500)
J0_exact = jv(0, x_bessel)  # scipy's exact Bessel function

# Compare different numbers of series terms
for n_terms in [5, 10, 20]:
    J0_approx = bessel_J0_series(x_bessel, N_terms=n_terms)
    axes[1].plot(x_bessel, J0_approx, '--',
                 label=f'Series ({n_terms} terms)', alpha=0.7)

axes[1].plot(x_bessel, J0_exact, 'k-', linewidth=2, label='J_0(x) [scipy]')
axes[1].set_xlabel('x')
axes[1].set_ylabel('J_0(x)')
axes[1].set_title("Bessel Function J_0: Series Convergence")
axes[1].set_ylim(-0.5, 1.1)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('power_series_solutions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to power_series_solutions.png")

# ── 3. Legendre Polynomials ──────────────────────────────────
print("\n=== Legendre Polynomials (from series solution) ===")
from numpy.polynomial.legendre import Legendre

x_leg = np.linspace(-1, 1, 200)
fig2, ax2 = plt.subplots(figsize=(8, 5))

for ell in range(5):
    # Legendre class uses coefficients in the Legendre basis
    # We use a simpler construction: scipy provides legendre via special
    from scipy.special import legendre
    P_ell = legendre(ell)
    ax2.plot(x_leg, P_ell(x_leg), linewidth=2, label=f'$P_{ell}(x)$')

ax2.set_xlabel('x')
ax2.set_ylabel('P_l(x)')
ax2.set_title('Legendre Polynomials P_0 through P_4')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig('legendre_polynomials.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to legendre_polynomials.png")
```

## Summary

| Concept | Key Idea |
|---------|----------|
| Ordinary point | $P(x)$, $Q(x)$ analytic; standard power series works |
| Regular singular point | $(x-x_0)P$, $(x-x_0)^2 Q$ analytic; Frobenius method works |
| Indicial equation | Determines exponent $r$ in the Frobenius series |
| Recurrence relation | Algebraic formula linking $a_{n+k}$ to earlier coefficients |
| Special functions | Bessel, Legendre, Laguerre, Hermite -- all arise from series solutions |

The power series method is not just an abstract technique. It is the historical origin of many functions that are as fundamental to applied mathematics as sine and cosine. When you call `scipy.special.jv(0, x)`, you are evaluating a function that was first discovered by inserting a series into an ODE and turning the crank.

## Practice Problems

1. **Ordinary point classification**: For the equation $(1 + x^2)y'' + 2xy' + 4y = 0$, identify all singular points and classify them as regular or irregular. What is the guaranteed radius of convergence for a series solution centered at $x_0 = 0$?

2. **Series solution**: Find the first six nonzero terms of the power series solution to $y'' + xy' + y = 0$ with $y(0) = 1$, $y'(0) = 0$. Write out the recurrence relation explicitly.

3. **Frobenius method**: Apply the Frobenius method to the Euler equation $x^2 y'' + 3xy' + y = 0$. Find the indicial equation and both solutions. (Hint: This equation also has exact solutions of the form $x^r$. Verify that your Frobenius series reduces to the exact answer.)

4. **Bessel function computation**: Write a Python function that computes $J_1(x)$ using its series representation. Compare your result with `scipy.special.jv(1, x)` over the interval $[0, 15]$. How many terms are needed for 8-digit accuracy at $x = 10$?

5. **Legendre's equation**: Starting from Legendre's equation with $\ell = 3$, derive the series solution and show that it terminates (becomes a polynomial). Verify that the resulting polynomial is a scalar multiple of $P_3(x) = \frac{1}{2}(5x^3 - 3x)$ by normalizing so that $P_3(1) = 1$.

---

*Previous: [Laplace Transform for ODE](./15_Laplace_Transform_for_ODE.md) | Next: [Introduction to Partial Differential Equations](./17_Introduction_to_PDE.md)*
