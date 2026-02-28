# Derivatives Fundamentals

## Learning Objectives

- **Define** the derivative as the limit of a difference quotient and interpret it as the instantaneous rate of change
- **Apply** differentiation rules (power, product, quotient, chain) to compute derivatives of composite functions
- **Perform** implicit differentiation on equations not solved for $y$
- **Compute** higher-order derivatives and understand their physical meaning (velocity, acceleration, jerk)
- **Implement** both numerical and symbolic differentiation in Python and compare their accuracy

## Introduction

If calculus is the mathematics of change, then the derivative is its most fundamental tool. Think of it this way: a speedometer in your car does not compute your average speed over the whole trip -- it tells you how fast you are going *right now*, at this exact instant. The derivative formalizes this idea of an "instantaneous rate of change."

Historically, Newton and Leibniz independently developed the derivative in the late 17th century. Newton thought of it as "fluxions" (rates of flowing quantities), while Leibniz developed the notation $\frac{dy}{dx}$ that we still use today. Both were trying to solve the same practical problems: how do planets move, and how do curves behave?

## The Derivative as a Limit

### The Difference Quotient

Given a function $f(x)$, the **average rate of change** between $x = a$ and $x = a + h$ is:

$$\frac{f(a + h) - f(a)}{h}$$

This is the slope of the **secant line** connecting the points $(a, f(a))$ and $(a+h, f(a+h))$.

### The Derivative Definition

The **derivative** of $f$ at $x = a$ is the limit of the difference quotient as $h \to 0$:

$$f'(a) = \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}$$

- $f'(a)$: the derivative of $f$ at $a$ (Newton's prime notation)
- $\frac{df}{dx}\bigg|_{x=a}$: the same thing in Leibniz notation
- $h$: a small increment that shrinks to zero
- The limit transforms the *average* rate into the *instantaneous* rate

**Geometric interpretation:** The derivative $f'(a)$ is the slope of the **tangent line** to the graph of $f$ at the point $(a, f(a))$.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_derivative(f, a, h_values=[1.0, 0.5, 0.1], x_range=(-1, 4)):
    """
    Show how secant lines approach the tangent line as h -> 0.

    This visualization makes the limit definition concrete: each secant
    line connects two points on the curve, and as h shrinks, the secant
    rotates to become the tangent.
    """
    x = np.linspace(*x_range, 500)
    y = f(x)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x, y, 'b-', linewidth=2, label='$f(x) = x^2$')

    colors = ['red', 'orange', 'green']
    for h, color in zip(h_values, colors):
        # Slope of secant line through (a, f(a)) and (a+h, f(a+h))
        slope = (f(a + h) - f(a)) / h
        # Equation of secant line: y - f(a) = slope * (x - a)
        y_secant = f(a) + slope * (x - a)
        ax.plot(x, y_secant, '--', color=color, linewidth=1.5,
                label=f'Secant (h={h}), slope={slope:.2f}')
        ax.plot([a, a + h], [f(a), f(a + h)], 'o', color=color, markersize=6)

    # True tangent line (derivative of x^2 at x=a is 2a)
    true_slope = 2 * a
    y_tangent = f(a) + true_slope * (x - a)
    ax.plot(x, y_tangent, 'k-', linewidth=2.5,
            label=f'Tangent, slope={true_slope:.2f}')
    ax.plot(a, f(a), 'ko', markersize=8, zorder=5)

    ax.set_xlim(*x_range)
    ax.set_ylim(-2, 12)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title(f'Secant lines approaching tangent at $x = {a}$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('derivative_visualization.png', dpi=150)
    plt.show()

f = lambda x: x**2
visualize_derivative(f, a=1.5, h_values=[1.5, 0.8, 0.3, 0.05])
```

## Basic Differentiation Rules

Computing derivatives from the limit definition every time would be tedious. Fortunately, a handful of rules cover most functions we encounter.

### Power Rule

$$\frac{d}{dx} x^n = n x^{n-1}$$

where $n$ is any real number. This single rule handles polynomials, roots ($x^{1/2}$), and reciprocals ($x^{-1}$).

**Examples:**
- $\frac{d}{dx} x^3 = 3x^2$
- $\frac{d}{dx} \sqrt{x} = \frac{d}{dx} x^{1/2} = \frac{1}{2} x^{-1/2} = \frac{1}{2\sqrt{x}}$
- $\frac{d}{dx} \frac{1}{x^2} = \frac{d}{dx} x^{-2} = -2x^{-3} = \frac{-2}{x^3}$

### Sum and Constant Multiple Rules

$$\frac{d}{dx} [f(x) + g(x)] = f'(x) + g'(x) \qquad \frac{d}{dx} [c \cdot f(x)] = c \cdot f'(x)$$

Differentiation is **linear**: you can differentiate term by term and pull constants out.

### Product Rule

When two functions are multiplied, their derivative is not simply the product of derivatives:

$$\frac{d}{dx} [f(x) \cdot g(x)] = f'(x) \cdot g(x) + f(x) \cdot g'(x)$$

**Mnemonic:** "First times derivative of second, plus second times derivative of first."

**Example:** $\frac{d}{dx} [x^2 \sin x] = 2x \sin x + x^2 \cos x$

### Quotient Rule

$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{f'(x) g(x) - f(x) g'(x)}{[g(x)]^2}$$

**Mnemonic:** "Low d-high minus high d-low, over low-low" (where "low" is the denominator).

**Example:**
$$\frac{d}{dx} \left[\frac{x^2}{x+1}\right] = \frac{2x(x+1) - x^2(1)}{(x+1)^2} = \frac{x^2 + 2x}{(x+1)^2}$$

### Chain Rule

The chain rule is arguably the most important rule in calculus, especially for machine learning (backpropagation is the chain rule applied recursively).

If $y = f(g(x))$, then:

$$\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$$

In Leibniz notation, if $y = f(u)$ and $u = g(x)$:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**Intuition:** If $u$ changes 3 times as fast as $x$, and $y$ changes 5 times as fast as $u$, then $y$ changes $15 = 3 \times 5$ times as fast as $x$. Rates of change multiply through a "chain" of compositions.

**Examples:**
- $\frac{d}{dx} (3x + 1)^5 = 5(3x+1)^4 \cdot 3 = 15(3x+1)^4$
- $\frac{d}{dx} \sin(x^2) = \cos(x^2) \cdot 2x$
- $\frac{d}{dx} e^{-x^2} = e^{-x^2} \cdot (-2x)$

### Common Derivatives Table

| Function $f(x)$ | Derivative $f'(x)$ |
|------------------|---------------------|
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln x$ | $1/x$ |
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x$ |
| $a^x$ | $a^x \ln a$ |
| $\arcsin x$ | $1/\sqrt{1-x^2}$ |
| $\arctan x$ | $1/(1+x^2)$ |

## Implicit Differentiation

Not all relationships are given as $y = f(x)$. For example, the equation of a circle is:

$$x^2 + y^2 = 25$$

Here $y$ is not explicitly written as a function of $x$. To find $\frac{dy}{dx}$, we differentiate both sides with respect to $x$, treating $y$ as an implicit function of $x$:

$$\frac{d}{dx}(x^2) + \frac{d}{dx}(y^2) = \frac{d}{dx}(25)$$

$$2x + 2y \frac{dy}{dx} = 0$$

$$\frac{dy}{dx} = -\frac{x}{y}$$

This tells us the slope at any point $(x, y)$ on the circle. At $(3, 4)$: slope $= -3/4$. At $(0, 5)$: slope $= 0$ (top of the circle, as expected).

```python
import sympy as sp

# Implicit differentiation with SymPy
x, y = sp.symbols('x y')

# Circle: x^2 + y^2 = 25
# SymPy's idiff handles implicit differentiation
circle_eq = x**2 + y**2 - 25

# Method 1: Using idiff (implicit differentiation)
dydx = sp.idiff(circle_eq, y, x)
print(f"dy/dx for circle: {dydx}")
# Output: -x/y

# Method 2: Manual approach -- differentiate and solve
# Treat y as a function of x: y(x)
y_func = sp.Function('y')(x)
eq_implicit = x**2 + y_func**2 - 25

# Differentiate with respect to x
diff_eq = sp.diff(eq_implicit, x)
print(f"After differentiating: {diff_eq}")

# Solve for dy/dx
dydx_manual = sp.solve(diff_eq, sp.diff(y_func, x))[0]
print(f"dy/dx (manual): {dydx_manual}")
```

## Higher-Order Derivatives

The derivative of a derivative is the **second derivative**, written $f''(x)$ or $\frac{d^2y}{dx^2}$:

$$f''(x) = \frac{d}{dx}\left[f'(x)\right]$$

**Physical interpretation for position $s(t)$:**
- $s'(t) = v(t)$: velocity (rate of position change)
- $s''(t) = v'(t) = a(t)$: acceleration (rate of velocity change)
- $s'''(t) = a'(t) = j(t)$: jerk (rate of acceleration change -- what you feel in an elevator)

```python
import sympy as sp

x = sp.Symbol('x')

# Compute derivatives of increasing order
f = sp.sin(x) * sp.exp(-x)
print(f"f(x) = {f}")
print(f"f'(x) = {sp.diff(f, x)}")
print(f"f''(x) = {sp.simplify(sp.diff(f, x, 2))}")
print(f"f'''(x) = {sp.simplify(sp.diff(f, x, 3))}")
print(f"f''''(x) = {sp.simplify(sp.diff(f, x, 4))}")

# Note: the nth derivative of sin(x)*exp(-x) has a beautiful pattern
# related to complex exponentials, which we explore in later lessons
```

## Numerical vs. Symbolic Differentiation

In practice, we use two approaches to compute derivatives:

### Forward Difference (Numerical)

$$f'(x) \approx \frac{f(x + h) - f(x)}{h}$$

Error: $O(h)$ -- accuracy improves linearly with smaller $h$.

### Central Difference (Numerical)

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

Error: $O(h^2)$ -- much more accurate for the same $h$.

### Symbolic (Exact)

Libraries like SymPy manipulate expressions algebraically to find exact derivatives.

```python
import numpy as np
import sympy as sp

# Compare numerical and symbolic approaches for f(x) = sin(x) at x = 1
def f_numeric(x):
    return np.sin(x)

x_val = 1.0
exact = np.cos(1.0)  # True derivative of sin(x) is cos(x)

# Numerical derivatives with decreasing h
print(f"{'h':<12} {'Forward':>14} {'Central':>14} {'Fwd Error':>12} {'Ctr Error':>12}")
print("-" * 66)
for h in [0.1, 0.01, 0.001, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    fwd = (f_numeric(x_val + h) - f_numeric(x_val)) / h
    ctr = (f_numeric(x_val + h) - f_numeric(x_val - h)) / (2 * h)
    print(f"{h:<12.0e} {fwd:>14.10f} {ctr:>14.10f} "
          f"{abs(fwd - exact):>12.2e} {abs(ctr - exact):>12.2e}")

# Note: Very small h causes floating-point cancellation errors.
# For h=1e-12, the central difference becomes LESS accurate.
# This demonstrates the tension between approximation error and
# machine precision -- a key theme in numerical computing.

# Symbolic (exact) approach
x = sp.Symbol('x')
f_sym = sp.sin(x)
df_sym = sp.diff(f_sym, x)  # cos(x) -- exact
print(f"\nSymbolic derivative: {df_sym}")
print(f"Evaluated at x=1: {float(df_sym.subs(x, 1)):.15f}")
print(f"Exact cos(1):      {exact:.15f}")
```

## Summary

- The **derivative** $f'(a)$ is defined as $\lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$ -- the instantaneous rate of change
- **Differentiation rules** (power, product, quotient, chain) eliminate the need to compute limits each time
- The **chain rule** is especially important: it tells us how to differentiate compositions, and it is the mathematical foundation of backpropagation in neural networks
- **Implicit differentiation** handles equations where $y$ is not isolated
- **Higher-order derivatives** capture acceleration, curvature, and higher-order rates of change
- **Numerical differentiation** (forward/central difference) approximates derivatives from function values, but floating-point errors limit accuracy for very small $h$
- **Symbolic differentiation** (SymPy) gives exact results but requires algebraic expressions

## Practice Problems

### Problem 1: Applying Differentiation Rules

Compute the derivative of each function:

(a) $f(x) = 3x^4 - 2x^3 + 7x - 9$

(b) $g(x) = x^2 e^x \sin x$ (use the product rule twice)

(c) $h(x) = \frac{\ln x}{x^2 + 1}$

(d) $k(x) = \cos(\sqrt{x^2 + 1})$ (use the chain rule carefully -- there are three nested functions)

### Problem 2: Implicit Differentiation

Find $\frac{dy}{dx}$ for the ellipse $\frac{x^2}{9} + \frac{y^2}{4} = 1$. At what points is the tangent line horizontal? Vertical?

### Problem 3: Numerical Accuracy Investigation

Write a Python script that computes the derivative of $f(x) = e^x$ at $x = 0$ using:
- Forward difference with $h = 10^{-k}$ for $k = 1, 2, \ldots, 16$
- Central difference with the same $h$ values

Plot the absolute error vs. $h$ on a log-log scale. Explain why the error decreases and then increases as $h$ gets very small.

### Problem 4: Derivative from First Principles

Using only the limit definition (not the rules), prove that $\frac{d}{dx}[\sin x] = \cos x$.

(Hint: You will need the angle-addition formula $\sin(a+b) = \sin a \cos b + \cos a \sin b$ and the limits $\lim_{h \to 0} \frac{\sin h}{h} = 1$ and $\lim_{h \to 0} \frac{\cos h - 1}{h} = 0$.)

### Problem 5: Chain Rule in Machine Learning

In a simple neural network, the loss function is $L = (y - \hat{y})^2$ where $\hat{y} = \sigma(wx + b)$ and $\sigma(z) = \frac{1}{1+e^{-z}}$ is the sigmoid function.

(a) Use the chain rule to compute $\frac{\partial L}{\partial w}$.

(b) Verify your answer with SymPy by defining the symbolic expression and differentiating.

## References

- Stewart, *Calculus: Early Transcendentals*, Ch. 3 (Differentiation Rules)
- [3Blue1Brown: Derivative formulas through geometry](https://www.youtube.com/watch?v=S0_qX4VJhMQ)
- [MIT OCW 18.01: Derivatives](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/)

---

[Previous: Limits and Continuity](./01_Limits_and_Continuity.md) | [Next: Applications of Derivatives](./03_Applications_of_Derivatives.md)
