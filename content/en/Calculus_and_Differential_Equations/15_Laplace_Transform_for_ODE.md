# Laplace Transform for ODE

## Learning Objectives

- Define the Laplace transform and compute transforms of elementary functions from the integral definition
- Apply key properties (linearity, shifting, differentiation, convolution) to simplify transform computations
- Perform inverse Laplace transforms using partial fraction decomposition
- Solve initial value problems by transforming ODE into algebraic equations in the s-domain
- Model discontinuous forcing using the Heaviside step function and Dirac delta function

## Prerequisites

Before studying this lesson, you should be comfortable with:
- Second-order linear ODE and initial value problems (Lessons 10-12)
- Systems of differential equations (Lesson 14)
- Improper integrals from calculus

## Motivation: Why the Laplace Transform?

Solving differential equations with the methods of undetermined coefficients or variation of parameters works well for constant-coefficient ODE with smooth forcing functions. But what if the forcing function is **discontinuous** -- a switch flipping on at time $t = 3$, or an impulsive hammer strike at $t = 0$? What about initial value problems where we want the solution in one systematic step rather than finding the general solution first and then fitting constants?

The **Laplace transform** converts a differential equation in the time domain into an **algebraic equation** in a frequency domain (the $s$-domain). We solve the algebra, then transform back. Think of it like using logarithms to turn multiplication into addition: we change the problem into an easier one, solve it, and convert back.

```
  Time Domain              s-Domain
  ──────────              ────────
  Differential Eq.  ──L──>  Algebraic Eq.
       │                        │
       │  (hard)                │  (easy)
       ▼                        ▼
  Solution y(t)   <──L⁻¹──  Solution Y(s)
```

## Definition of the Laplace Transform

Given a function $f(t)$ defined for $t \geq 0$, its **Laplace transform** is:

$$\mathcal{L}\{f(t)\} = F(s) = \int_0^{\infty} e^{-st} f(t) \, dt$$

where:
- $f(t)$: the original function in the **time domain** (input signal, forcing function, etc.)
- $F(s)$: the transformed function in the **$s$-domain** (also called the frequency domain)
- $s$: a complex variable, $s = \sigma + j\omega$, chosen so that the integral converges
- $e^{-st}$: the **kernel** that "weighs" the function by an exponential decay

The integral converges when $\text{Re}(s)$ is large enough to overpower the growth of $f(t)$. The minimum value of $\text{Re}(s)$ for convergence is called the **abscissa of convergence**.

### Computing Transforms from the Definition

**Example 1: $f(t) = 1$ (constant function)**

$$\mathcal{L}\{1\} = \int_0^{\infty} e^{-st} \cdot 1 \, dt = \left[-\frac{1}{s} e^{-st}\right]_0^{\infty} = 0 - \left(-\frac{1}{s}\right) = \frac{1}{s}, \quad s > 0$$

**Example 2: $f(t) = e^{at}$ (exponential)**

$$\mathcal{L}\{e^{at}\} = \int_0^{\infty} e^{-st} e^{at} \, dt = \int_0^{\infty} e^{-(s-a)t} \, dt = \frac{1}{s - a}, \quad s > a$$

**Example 3: $f(t) = t^n$ (power function)**

By repeated integration by parts (or recognizing the Gamma function):

$$\mathcal{L}\{t^n\} = \frac{n!}{s^{n+1}}, \quad s > 0, \quad n = 0, 1, 2, \ldots$$

## Table of Common Laplace Pairs

This table is your essential reference. Every entry can be derived from the definition, but having them memorized (or at hand) speeds up problem solving enormously.

| $f(t)$ | $F(s) = \mathcal{L}\{f(t)\}$ | Region of Convergence |
|---------|-------------------------------|----------------------|
| $1$ | $\dfrac{1}{s}$ | $s > 0$ |
| $t^n$ | $\dfrac{n!}{s^{n+1}}$ | $s > 0$ |
| $e^{at}$ | $\dfrac{1}{s-a}$ | $s > a$ |
| $\sin(\omega t)$ | $\dfrac{\omega}{s^2 + \omega^2}$ | $s > 0$ |
| $\cos(\omega t)$ | $\dfrac{s}{s^2 + \omega^2}$ | $s > 0$ |
| $e^{at}\sin(\omega t)$ | $\dfrac{\omega}{(s-a)^2 + \omega^2}$ | $s > a$ |
| $e^{at}\cos(\omega t)$ | $\dfrac{s-a}{(s-a)^2 + \omega^2}$ | $s > a$ |
| $t \cdot e^{at}$ | $\dfrac{1}{(s-a)^2}$ | $s > a$ |
| $u(t-a)$ (Heaviside) | $\dfrac{e^{-as}}{s}$ | $s > 0$ |
| $\delta(t-a)$ (Dirac) | $e^{-as}$ | all $s$ |

## Key Properties

### 1. Linearity

$$\mathcal{L}\{af(t) + bg(t)\} = aF(s) + bG(s)$$

This is inherited directly from the linearity of integration. It means we can transform each term of an ODE separately.

### 2. First Shifting Theorem (s-shifting)

$$\mathcal{L}\{e^{at}f(t)\} = F(s - a)$$

Multiplying by $e^{at}$ in the time domain shifts $F(s)$ by $a$ in the $s$-domain. This is why $\mathcal{L}\{e^{at}\cos(\omega t)\}$ has $s$ replaced by $(s-a)$ in the cosine transform.

### 3. Second Shifting Theorem (t-shifting)

$$\mathcal{L}\{u(t-a)f(t-a)\} = e^{-as}F(s)$$

where $u(t-a)$ is the Heaviside step function. A time delay of $a$ seconds multiplies the transform by $e^{-as}$.

### 4. Differentiation Property (The Key to Solving ODE)

$$\mathcal{L}\{f'(t)\} = sF(s) - f(0)$$

$$\mathcal{L}\{f''(t)\} = s^2 F(s) - sf(0) - f'(0)$$

More generally:

$$\mathcal{L}\{f^{(n)}(t)\} = s^n F(s) - s^{n-1}f(0) - s^{n-2}f'(0) - \cdots - f^{(n-1)}(0)$$

This is the **crucial property**: derivatives become multiplication by $s$, and initial conditions appear automatically. This is why the Laplace transform is tailor-made for initial value problems.

### 5. Integration Property

$$\mathcal{L}\left\{\int_0^t f(\tau) \, d\tau\right\} = \frac{F(s)}{s}$$

Integration in time corresponds to division by $s$ in the frequency domain.

### 6. Convolution Theorem

$$\mathcal{L}\{(f * g)(t)\} = F(s) \cdot G(s)$$

where the convolution is $(f * g)(t) = \int_0^t f(\tau)g(t - \tau) \, d\tau$. Multiplication in the $s$-domain corresponds to convolution in time. This is useful for inverse transforms of products.

## Inverse Laplace Transform

The inverse Laplace transform recovers $f(t)$ from $F(s)$:

$$f(t) = \mathcal{L}^{-1}\{F(s)\}$$

In practice, we rarely use the complex integral formula. Instead, we use:

1. **Table lookup**: Match $F(s)$ to a known pair
2. **Partial fraction decomposition**: Break a rational $F(s)$ into simpler terms
3. **Completing the square**: For irreducible quadratics in the denominator

### Partial Fractions Method

Given $F(s) = \frac{P(s)}{Q(s)}$ where $\deg(P) < \deg(Q)$:

**Distinct real roots**: If $Q(s) = (s - r_1)(s - r_2) \cdots (s - r_n)$:

$$F(s) = \frac{A_1}{s - r_1} + \frac{A_2}{s - r_2} + \cdots + \frac{A_n}{s - r_n}$$

Each term inverts to $A_k e^{r_k t}$.

**Repeated roots**: If $Q(s)$ has $(s - r)^m$:

$$\frac{A_1}{s - r} + \frac{A_2}{(s - r)^2} + \cdots + \frac{A_m}{(s - r)^m}$$

**Complex roots**: For irreducible $s^2 + bs + c$, complete the square and match to shifted sine/cosine forms.

### Worked Example: Inverse Transform

Find $\mathcal{L}^{-1}\left\{\dfrac{5s + 3}{s^2 + 4s + 13}\right\}$.

**Step 1**: Complete the square in the denominator:

$$s^2 + 4s + 13 = (s + 2)^2 + 9 = (s + 2)^2 + 3^2$$

**Step 2**: Rewrite the numerator to match the shifted forms:

$$\frac{5s + 3}{(s+2)^2 + 9} = \frac{5(s+2) - 7}{(s+2)^2 + 9} = 5\cdot\frac{s+2}{(s+2)^2 + 9} - \frac{7}{3}\cdot\frac{3}{(s+2)^2 + 9}$$

**Step 3**: Invert using the table:

$$f(t) = 5e^{-2t}\cos(3t) - \frac{7}{3}e^{-2t}\sin(3t)$$

## Solving IVPs with Laplace Transform

### The Systematic Procedure

1. Take the Laplace transform of both sides of the ODE
2. Substitute initial conditions (they appear from the differentiation property)
3. Solve for $Y(s)$ algebraically
4. Find $y(t) = \mathcal{L}^{-1}\{Y(s)\}$ using partial fractions and table lookup

### Worked Example: Second-Order IVP

Solve: $y'' + 5y' + 6y = 2e^{-t}$, with $y(0) = 1$, $y'(0) = 0$.

**Step 1**: Transform both sides.

$$[s^2 Y - sy(0) - y'(0)] + 5[sY - y(0)] + 6Y = \frac{2}{s + 1}$$

**Step 2**: Substitute $y(0) = 1$, $y'(0) = 0$:

$$s^2 Y - s + 5sY - 5 + 6Y = \frac{2}{s+1}$$

$$(s^2 + 5s + 6)Y = \frac{2}{s+1} + s + 5$$

**Step 3**: Factor and solve for $Y$:

$$(s+2)(s+3)Y = \frac{2 + (s+5)(s+1)}{s+1} = \frac{s^2 + 6s + 7}{s+1}$$

$$Y(s) = \frac{s^2 + 6s + 7}{(s+1)(s+2)(s+3)}$$

**Step 4**: Partial fractions:

$$\frac{s^2 + 6s + 7}{(s+1)(s+2)(s+3)} = \frac{A}{s+1} + \frac{B}{s+2} + \frac{C}{s+3}$$

Setting $s = -1$: $A = \frac{1 - 6 + 7}{(1)(2)} = 1$

Setting $s = -2$: $B = \frac{4 - 12 + 7}{(-1)(1)} = -\frac{-1}{-1} = -1$

Wait -- let us be careful: $B = \frac{4 - 12 + 7}{(-2+1)(-2+3)} = \frac{-1}{(-1)(1)} = 1$

Setting $s = -3$: $C = \frac{9 - 18 + 7}{(-3+1)(-3+2)} = \frac{-2}{(-2)(-1)} = \frac{-2}{2} = -1$

$$Y(s) = \frac{1}{s+1} + \frac{1}{s+2} - \frac{1}{s+3}$$

**Step 5**: Invert:

$$y(t) = e^{-t} + e^{-2t} - e^{-3t}$$

**Verification**: At $t = 0$: $y(0) = 1 + 1 - 1 = 1$ (correct). Computing $y'(0) = -1 - 2 + 3 = 0$ (correct).

## Heaviside Step Function

The **Heaviside step function** (or unit step function) models sudden switches:

$$u(t - a) = \begin{cases} 0, & t < a \\ 1, & t \geq a \end{cases}$$

It "turns on" at time $t = a$. Any piecewise function can be written using step functions:

$$f(t) = \begin{cases} 0, & t < 1 \\ 3, & 1 \leq t < 4 \\ 0, & t \geq 4 \end{cases} = 3[u(t-1) - u(t-4)]$$

**Laplace transform**: $\mathcal{L}\{u(t-a)\} = \dfrac{e^{-as}}{s}$

For a shifted function: $\mathcal{L}\{u(t-a)f(t-a)\} = e^{-as}F(s)$ (second shifting theorem).

## Dirac Delta Function

The **Dirac delta function** $\delta(t - a)$ models an instantaneous impulse at time $t = a$:

- $\delta(t - a) = 0$ for $t \neq a$
- $\int_{-\infty}^{\infty} \delta(t - a) f(t) \, dt = f(a)$ (sifting property)

Think of it as the limit of a very tall, very narrow pulse whose total area is 1. It models a hammer strike, a sudden voltage spike, or a point force.

**Laplace transform**: $\mathcal{L}\{\delta(t - a)\} = e^{-as}$

For $a = 0$: $\mathcal{L}\{\delta(t)\} = 1$. This means the impulse response of a system is obtained by inverse-transforming the transfer function directly.

## Transfer Functions (Introduction)

For a linear constant-coefficient ODE with zero initial conditions:

$$a_n y^{(n)} + \cdots + a_1 y' + a_0 y = f(t)$$

The **transfer function** is:

$$H(s) = \frac{Y(s)}{F(s)} = \frac{1}{a_n s^n + \cdots + a_1 s + a_0}$$

$H(s)$ characterizes the system independently of the input. The output for any input is $Y(s) = H(s) F(s)$, which in the time domain is the convolution $y(t) = (h * f)(t)$ where $h(t) = \mathcal{L}^{-1}\{H(s)\}$ is the **impulse response**.

This concept is central to control theory (see Control_Theory topic) and signal processing (see Signal_Processing topic).

## Python Implementation

```python
"""
Laplace Transform for solving ODE using SymPy.

This script demonstrates:
1. Computing Laplace transforms symbolically
2. Solving an IVP via the Laplace method
3. Visualizing the solution and step-function forcing
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import (
    symbols, Function, laplace_transform, inverse_laplace_transform,
    exp, sin, cos, Heaviside, DiracDelta, apart, oo,
    dsolve, Eq, pprint, simplify
)

# Define symbolic variables
t, s = symbols('t s', positive=True)
# positive=True helps SymPy assume t > 0 for Laplace integrals

# ── 1. Basic Laplace Transforms ──────────────────────────────
print("=== Basic Laplace Transforms ===\n")

functions = [1, t, t**2, exp(-3*t), sin(2*t), cos(2*t)]
for f in functions:
    # laplace_transform returns (transform, convergence_region, condition)
    F, region, cond = laplace_transform(f, t, s)
    print(f"L{{{f}}} = {F},  (converges for {cond})")

# ── 2. Properties demonstration ──────────────────────────────
print("\n=== Shifting Property ===")
# L{e^{at} f(t)} = F(s-a)
# L{e^{-2t} sin(3t)} should be 3/((s+2)^2 + 9)
f_shifted = exp(-2*t) * sin(3*t)
F_shifted, _, _ = laplace_transform(f_shifted, t, s)
print(f"L{{e^(-2t) sin(3t)}} = {F_shifted}")

# ── 3. Solving an IVP with Laplace Transform ────────────────
print("\n=== Solving IVP: y'' + 5y' + 6y = 2e^(-t), y(0)=1, y'(0)=0 ===\n")

# Method: Manual algebraic approach
# After transforming: Y(s) = (s^2 + 6s + 7) / ((s+1)(s+2)(s+3))
Y_s = (s**2 + 6*s + 7) / ((s + 1) * (s + 2) * (s + 3))

# Partial fraction decomposition — apart() does the heavy lifting
Y_partial = apart(Y_s, s)
print(f"Y(s) = {Y_s}")
print(f"Partial fractions: Y(s) = {Y_partial}")

# Inverse Laplace transform to get y(t)
y_solution = inverse_laplace_transform(Y_s, s, t)
print(f"y(t) = {simplify(y_solution)}")

# ── 4. Verify with dsolve (direct ODE solver) ───────────────
print("\n=== Verification with dsolve ===")
y = Function('y')
t_sym = symbols('t')
ode = Eq(y(t_sym).diff(t_sym, 2) + 5*y(t_sym).diff(t_sym) + 6*y(t_sym),
         2*exp(-t_sym))
# dsolve finds the general solution; we apply initial conditions
sol = dsolve(ode, y(t_sym), ics={y(0): 1, y(t_sym).diff(t_sym).subs(t_sym, 0): 0})
print(f"dsolve result: {sol}")

# ── 5. Visualization ─────────────────────────────────────────
t_vals = np.linspace(0, 5, 300)

# Our Laplace-derived solution: y(t) = e^{-t} + e^{-2t} - e^{-3t}
y_vals = np.exp(-t_vals) + np.exp(-2*t_vals) - np.exp(-3*t_vals)

# Forcing function: 2e^{-t}
f_vals = 2 * np.exp(-t_vals)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot the solution
axes[0].plot(t_vals, y_vals, 'b-', linewidth=2, label=r'$y(t) = e^{-t} + e^{-2t} - e^{-3t}$')
axes[0].axhline(y=0, color='gray', linewidth=0.5)
axes[0].set_xlabel('t')
axes[0].set_ylabel('y(t)')
axes[0].set_title("Solution of y'' + 5y' + 6y = 2e^{-t}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot with Heaviside forcing
# Solve y'' + 4y = u(t-2), y(0)=0, y'(0)=0
# The step function turns on a constant force at t=2
t_vals2 = np.linspace(0, 10, 500)
# Analytical solution using Laplace:
#   Y(s) = e^{-2s} / (s(s^2+4))
#   y(t) = u(t-2) * [1/4 - (1/4)cos(2(t-2))]
y_step = np.where(t_vals2 >= 2,
                  0.25 * (1 - np.cos(2*(t_vals2 - 2))),
                  0.0)
# The forcing function
f_step = np.where(t_vals2 >= 2, 1.0, 0.0)

axes[1].plot(t_vals2, y_step, 'b-', linewidth=2, label=r'$y(t)$')
axes[1].plot(t_vals2, f_step, 'r--', linewidth=1.5, alpha=0.6, label=r'$u(t-2)$ (forcing)')
axes[1].axhline(y=0, color='gray', linewidth=0.5)
axes[1].axvline(x=2, color='gray', linewidth=0.5, linestyle=':')
axes[1].set_xlabel('t')
axes[1].set_ylabel('y(t)')
axes[1].set_title("Response to Step Function: y'' + 4y = u(t-2)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('laplace_transform_solutions.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to laplace_transform_solutions.png")
```

## Summary

The Laplace transform is a powerful algebraic tool for solving initial value problems:

| Concept | Key Idea |
|---------|----------|
| Definition | $F(s) = \int_0^\infty e^{-st} f(t) \, dt$ |
| Differentiation | Derivatives become powers of $s$; ICs appear automatically |
| Inverse | Partial fractions + table lookup |
| Step function | Models sudden switches; uses $e^{-as}$ shifting |
| Delta function | Models impulses; $\mathcal{L}\{\delta(t)\} = 1$ |
| Transfer function | $H(s) = Y(s)/F(s)$ characterizes the system |

For a deeper treatment of the Laplace transform including the Bromwich integral and applications in complex analysis, see [Mathematical Methods - Laplace Transform](../Mathematical_Methods/15_Laplace_Transform.md).

## Practice Problems

1. **Basic transforms**: Compute $\mathcal{L}\{t^3 e^{-2t}\}$ and $\mathcal{L}\{e^{3t}\cos(4t)\}$ using properties (not the integral definition). Verify your answers with SymPy.

2. **Inverse transform**: Find $\mathcal{L}^{-1}\left\{\dfrac{3s + 7}{(s+1)(s^2 + 4)}\right\}$ using partial fractions. Identify the transient and oscillatory parts of the solution.

3. **IVP solving**: Solve $y'' + 4y' + 4y = e^{-2t}$, $y(0) = 0$, $y'(0) = 1$ using the Laplace transform. Note that the denominator has a repeated root -- how does this affect the partial fractions?

4. **Step function**: A spring-mass system satisfies $y'' + 9y = 5u(t - \pi)$ with $y(0) = 0$, $y'(0) = 0$. Find the response $y(t)$. Sketch the solution and explain physically what happens at $t = \pi$.

5. **Impulse response**: For the system $y'' + 2y' + 5y = \delta(t)$, $y(0^-) = 0$, $y'(0^-) = 0$, find the impulse response $h(t)$. Then use convolution to find the response to $f(t) = e^{-t}$ (without re-solving the ODE).

---

*Previous: [Systems of Ordinary Differential Equations](./14_Systems_of_ODE.md) | Next: [Power Series Solutions](./16_Power_Series_Solutions.md)*
