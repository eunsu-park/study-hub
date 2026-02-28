# Limits and Continuity

## Learning Objectives

- **Define** the limit of a function using the epsilon-delta formulation and explain its geometric meaning
- **Evaluate** limits using algebraic techniques, the Squeeze Theorem, and L'Hopital's rule (preview)
- **Classify** types of discontinuity (removable, jump, infinite) and determine where a function is continuous
- **Apply** the Intermediate Value Theorem to prove the existence of solutions
- **Implement** numerical limit estimation and epsilon-delta visualization in Python

## Introduction

Imagine driving a car toward a tunnel entrance. You can describe your position at every moment *before* reaching the tunnel, and you can predict exactly where you will be when you arrive -- even if, for some reason, you never actually enter. This is the essence of a limit: it describes the value a function *approaches* as the input approaches a particular point, regardless of whether the function is actually defined there.

Limits are the foundational concept of calculus. Every derivative is defined as a limit. Every integral is defined as a limit. Understanding limits rigorously is what separates calculus from mere algebraic manipulation.

## Intuitive Notion of Limits

Consider the function:

$$f(x) = \frac{x^2 - 1}{x - 1}$$

At $x = 1$, this function is undefined (division by zero). But we can ask: what value does $f(x)$ approach as $x$ gets closer and closer to 1?

Factoring the numerator: $f(x) = \frac{(x-1)(x+1)}{x-1} = x + 1$ for $x \neq 1$.

As $x \to 1$, we get $f(x) \to 2$. We write:

$$\lim_{x \to 1} \frac{x^2 - 1}{x - 1} = 2$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Numerically approach x = 1 from both sides
x_left = 1 - np.array([0.1, 0.01, 0.001, 0.0001, 0.00001])
x_right = 1 + np.array([0.1, 0.01, 0.001, 0.0001, 0.00001])

# f(x) = (x^2 - 1) / (x - 1) -- undefined at x=1, but approaches 2
f = lambda x: (x**2 - 1) / (x - 1)

print("Approaching from the left:")
for x in x_left:
    print(f"  f({x:.5f}) = {f(x):.10f}")

print("\nApproaching from the right:")
for x in x_right:
    print(f"  f({x:.5f}) = {f(x):.10f}")
# Both sides converge to 2.0, confirming lim_{x->1} f(x) = 2
```

## The Epsilon-Delta Definition

The informal statement "f(x) approaches L as x approaches a" is made rigorous by the epsilon-delta definition:

$$\lim_{x \to a} f(x) = L$$

means: for every $\varepsilon > 0$, there exists a $\delta > 0$ such that

$$0 < |x - a| < \delta \implies |f(x) - L| < \varepsilon$$

**Symbol meanings:**
- $\varepsilon$ (epsilon): the tolerance -- how close we require $f(x)$ to be to $L$
- $\delta$ (delta): the corresponding restriction -- how close $x$ must be to $a$
- $0 < |x - a|$: we never evaluate at $x = a$ itself; we only look *near* $a$

**Geometric interpretation:** Given any horizontal band of width $2\varepsilon$ centered at $y = L$, we can find a vertical band of width $2\delta$ centered at $x = a$ such that the graph of $f$ stays within the horizontal band whenever $x$ is within the vertical band (excluding $x = a$ itself).

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_epsilon_delta(f, a, L, epsilon, delta, x_range=(0, 3)):
    """
    Visualize the epsilon-delta definition of a limit.

    The shaded regions show the epsilon-band (horizontal, around L)
    and the delta-band (vertical, around a). If the function's graph
    stays inside the epsilon-band whenever x is in the delta-band,
    the limit condition is satisfied for these particular values.
    """
    x = np.linspace(*x_range, 1000)
    # Remove the point x=a to show it's excluded from consideration
    x = x[np.abs(x - a) > 0.001]
    y = f(x)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the function
    ax.plot(x, y, 'b-', linewidth=2, label='$f(x)$')

    # Epsilon band (horizontal)
    ax.axhspan(L - epsilon, L + epsilon, alpha=0.2, color='green',
               label=f'$\\varepsilon$-band: $({L-epsilon:.1f}, {L+epsilon:.1f})$')
    ax.axhline(y=L, color='green', linestyle='--', alpha=0.5)

    # Delta band (vertical)
    ax.axvspan(a - delta, a + delta, alpha=0.2, color='red',
               label=f'$\\delta$-band: $({a-delta:.2f}, {a+delta:.2f})$')
    ax.axvline(x=a, color='red', linestyle='--', alpha=0.5)

    # Mark the limit point (open circle since f may not be defined there)
    ax.plot(a, L, 'o', color='green', markersize=10, markerfacecolor='white',
            markeredgewidth=2, zorder=5)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title(f'$\\varepsilon$-$\\delta$ visualization: '
                 f'$\\varepsilon={epsilon}$, $\\delta={delta}$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('epsilon_delta.png', dpi=150)
    plt.show()

# Example: lim_{x->1} (x^2 - 1)/(x - 1) = 2
f = lambda x: (x**2 - 1) / (x - 1)
visualize_epsilon_delta(f, a=1, L=2, epsilon=0.5, delta=0.4, x_range=(0, 3))
```

## Limit Laws

If $\lim_{x \to a} f(x) = L$ and $\lim_{x \to a} g(x) = M$, then:

| Law | Statement |
|-----|-----------|
| **Sum** | $\lim_{x \to a} [f(x) + g(x)] = L + M$ |
| **Difference** | $\lim_{x \to a} [f(x) - g(x)] = L - M$ |
| **Product** | $\lim_{x \to a} [f(x) \cdot g(x)] = L \cdot M$ |
| **Quotient** | $\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{L}{M}$, provided $M \neq 0$ |
| **Power** | $\lim_{x \to a} [f(x)]^n = L^n$ |
| **Constant** | $\lim_{x \to a} c = c$ |

These laws let us break complex limits into simpler pieces. For example:

$$\lim_{x \to 2} (3x^2 + 5x - 1) = 3(4) + 5(2) - 1 = 21$$

## One-Sided Limits

Sometimes a function behaves differently from the left and right:

$$\lim_{x \to a^-} f(x) \quad \text{(left-hand limit: approach from below)}$$
$$\lim_{x \to a^+} f(x) \quad \text{(right-hand limit: approach from above)}$$

**Key theorem:** $\lim_{x \to a} f(x) = L$ exists if and only if both one-sided limits exist and are equal:

$$\lim_{x \to a^-} f(x) = \lim_{x \to a^+} f(x) = L$$

**Example:** The Heaviside step function $H(x) = \begin{cases} 0 & x < 0 \\ 1 & x \geq 0 \end{cases}$

Here $\lim_{x \to 0^-} H(x) = 0$ and $\lim_{x \to 0^+} H(x) = 1$. Since these differ, $\lim_{x \to 0} H(x)$ does not exist.

## The Squeeze Theorem

If $g(x) \leq f(x) \leq h(x)$ near $x = a$ (except possibly at $a$), and

$$\lim_{x \to a} g(x) = \lim_{x \to a} h(x) = L$$

then $\lim_{x \to a} f(x) = L$.

**Classic application:** Prove $\lim_{x \to 0} x \sin(1/x) = 0$.

Since $-1 \leq \sin(1/x) \leq 1$, we have $-|x| \leq x\sin(1/x) \leq |x|$.

Both $\lim_{x \to 0} (-|x|) = 0$ and $\lim_{x \to 0} |x| = 0$, so by the Squeeze Theorem:

$$\lim_{x \to 0} x \sin(1/x) = 0$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize the Squeeze Theorem for x*sin(1/x)
x = np.linspace(-0.5, 0.5, 10000)
x = x[x != 0]  # Remove x=0 to avoid division by zero

y = x * np.sin(1/x)
y_upper = np.abs(x)   # Upper bound: |x|
y_lower = -np.abs(x)  # Lower bound: -|x|

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, 'b-', linewidth=0.5, label='$x \\sin(1/x)$')
ax.plot(x, y_upper, 'r--', linewidth=1.5, label='$|x|$ (upper bound)')
ax.plot(x, y_lower, 'g--', linewidth=1.5, label='$-|x|$ (lower bound)')
ax.plot(0, 0, 'ko', markersize=8, zorder=5, label='Limit = 0')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Squeeze Theorem: $-|x| \\leq x\\sin(1/x) \\leq |x|$')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('squeeze_theorem.png', dpi=150)
plt.show()
```

## Limits at Infinity

We also consider what happens as $x$ grows without bound:

$$\lim_{x \to \infty} f(x) = L$$

means $f(x)$ can be made arbitrarily close to $L$ by taking $x$ sufficiently large.

**Useful technique for rational functions:** Divide numerator and denominator by the highest power of $x$ in the denominator.

$$\lim_{x \to \infty} \frac{3x^2 + 2x}{5x^2 - 1} = \lim_{x \to \infty} \frac{3 + 2/x}{5 - 1/x^2} = \frac{3}{5}$$

**Rule of thumb for rational functions** $\frac{p(x)}{q(x)}$:
- If $\deg(p) < \deg(q)$: limit is 0
- If $\deg(p) = \deg(q)$: limit is the ratio of leading coefficients
- If $\deg(p) > \deg(q)$: limit is $\pm\infty$ (does not exist as a finite value)

## Continuity

A function $f$ is **continuous at** $x = a$ if three conditions hold:

1. $f(a)$ is defined
2. $\lim_{x \to a} f(x)$ exists
3. $\lim_{x \to a} f(x) = f(a)$

In everyday terms: you can draw the graph without lifting your pen.

### Types of Discontinuity

| Type | Description | Example |
|------|-------------|---------|
| **Removable** | Limit exists but $f(a)$ is missing or wrong | $f(x) = \frac{x^2-1}{x-1}$ at $x=1$ |
| **Jump** | One-sided limits exist but differ | Heaviside function at $x=0$ |
| **Infinite** | Function blows up to $\pm\infty$ | $f(x) = 1/x$ at $x=0$ |
| **Oscillatory** | No limit due to oscillation | $f(x) = \sin(1/x)$ at $x=0$ |

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Demonstrate different types of discontinuity
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Removable discontinuity: (x^2 - 1)/(x - 1) at x = 1
x1 = np.linspace(-1, 3, 1000)
x1 = x1[np.abs(x1 - 1) > 0.01]
y1 = (x1**2 - 1) / (x1 - 1)
axes[0, 0].plot(x1, y1, 'b-', linewidth=2)
axes[0, 0].plot(1, 2, 'o', color='blue', markersize=10,
                markerfacecolor='white', markeredgewidth=2)  # Open circle at hole
axes[0, 0].set_title('Removable: $(x^2-1)/(x-1)$ at $x=1$')
axes[0, 0].grid(True, alpha=0.3)

# 2. Jump discontinuity: Heaviside function at x = 0
x2 = np.linspace(-2, 2, 1000)
y2 = np.heaviside(x2, 0.5)
axes[0, 1].plot(x2[x2 < 0], y2[x2 < 0], 'b-', linewidth=2)
axes[0, 1].plot(x2[x2 > 0], y2[x2 > 0], 'b-', linewidth=2)
axes[0, 1].plot(0, 0, 'o', color='blue', markersize=10,
                markerfacecolor='white', markeredgewidth=2)
axes[0, 1].plot(0, 1, 'o', color='blue', markersize=10, markeredgewidth=2)
axes[0, 1].set_title('Jump: Heaviside at $x=0$')
axes[0, 1].grid(True, alpha=0.3)

# 3. Infinite discontinuity: 1/x at x = 0
x3 = np.linspace(-2, 2, 1000)
x3 = x3[np.abs(x3) > 0.05]
y3 = 1 / x3
axes[1, 0].plot(x3, y3, 'b-', linewidth=2)
axes[1, 0].set_ylim(-10, 10)
axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Infinite: $1/x$ at $x=0$')
axes[1, 0].grid(True, alpha=0.3)

# 4. Oscillatory discontinuity: sin(1/x) at x = 0
x4 = np.linspace(-0.5, 0.5, 100000)
x4 = x4[np.abs(x4) > 0.001]
y4 = np.sin(1 / x4)
axes[1, 1].plot(x4, y4, 'b-', linewidth=0.3)
axes[1, 1].set_title('Oscillatory: $\\sin(1/x)$ at $x=0$')
axes[1, 1].grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

plt.tight_layout()
plt.savefig('discontinuity_types.png', dpi=150)
plt.show()
```

## The Intermediate Value Theorem (IVT)

**Theorem:** If $f$ is continuous on $[a, b]$ and $N$ is any number between $f(a)$ and $f(b)$, then there exists at least one $c \in (a, b)$ such that $f(c) = N$.

**Intuition:** A continuous function cannot "jump over" a value. If you start below sea level and end above it, you must cross sea level at some point.

**Practical application:** The IVT guarantees the existence of roots. If $f(a) < 0$ and $f(b) > 0$ (with $f$ continuous), then $f(c) = 0$ for some $c \in (a, b)$.

```python
import numpy as np

def bisection_method(f, a, b, tol=1e-10, max_iter=100):
    """
    Find a root of f in [a, b] using the bisection method.

    The IVT guarantees a root exists when f(a) and f(b) have
    opposite signs. Bisection repeatedly halves the interval,
    choosing the half where the sign change persists.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    for i in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c, i + 1  # Return root and iteration count
        if f(a) * f(c) < 0:
            b = c  # Root is in left half
        else:
            a = c  # Root is in right half

    return c, max_iter

# Example: find sqrt(2) as a root of f(x) = x^2 - 2
f = lambda x: x**2 - 2
root, iterations = bisection_method(f, 1, 2)
print(f"Root found: {root:.15f}")
print(f"Actual sqrt(2): {np.sqrt(2):.15f}")
print(f"Iterations: {iterations}")
print(f"Error: {abs(root - np.sqrt(2)):.2e}")
```

## Symbolic Limits with SymPy

Python's SymPy library can compute limits symbolically, giving exact answers:

```python
import sympy as sp

x = sp.Symbol('x')

# Basic limit
expr1 = (x**2 - 1) / (x - 1)
print(f"lim (x^2-1)/(x-1) as x->1: {sp.limit(expr1, x, 1)}")
# Output: 2

# Limit at infinity
expr2 = (3*x**2 + 2*x) / (5*x**2 - 1)
print(f"lim (3x^2+2x)/(5x^2-1) as x->inf: {sp.limit(expr2, x, sp.oo)}")
# Output: 3/5

# One-sided limits
expr3 = 1 / x
print(f"lim 1/x as x->0+: {sp.limit(expr3, x, 0, '+')}")  # oo
print(f"lim 1/x as x->0-: {sp.limit(expr3, x, 0, '-')}")  # -oo

# The famous limit: sin(x)/x as x -> 0
expr4 = sp.sin(x) / x
print(f"lim sin(x)/x as x->0: {sp.limit(expr4, x, 0)}")
# Output: 1

# Squeeze theorem example
expr5 = x * sp.sin(1/x)
print(f"lim x*sin(1/x) as x->0: {sp.limit(expr5, x, 0)}")
# Output: 0
```

## Summary

- A **limit** describes the value a function approaches, not necessarily the value it takes
- The **epsilon-delta definition** makes "approaching" rigorous: for any tolerance $\varepsilon$, we can find a neighborhood $\delta$
- **Limit laws** let us decompose complex limits into simpler parts
- **One-sided limits** must agree for the two-sided limit to exist
- The **Squeeze Theorem** handles oscillating functions by bounding them
- **Continuity** means the limit equals the function value -- no gaps, jumps, or holes
- The **Intermediate Value Theorem** guarantees that continuous functions cannot skip values, enabling root-finding algorithms

## Practice Problems

### Problem 1: Algebraic Limit Evaluation

Evaluate the following limits algebraically (show your work):

(a) $\lim_{x \to 3} \frac{x^2 - 9}{x - 3}$

(b) $\lim_{x \to 0} \frac{\sqrt{1 + x} - 1}{x}$ (Hint: rationalize the numerator)

(c) $\lim_{x \to \infty} \frac{2x^3 - x + 5}{4x^3 + 3x^2}$

### Problem 2: Epsilon-Delta Proof

Using the epsilon-delta definition, prove that $\lim_{x \to 2} (3x + 1) = 7$.

(Hint: Start with $|f(x) - L| < \varepsilon$ and work backward to find $\delta$ in terms of $\varepsilon$.)

### Problem 3: Discontinuity Classification

For each function, determine the type of discontinuity at the given point:

(a) $f(x) = \frac{\sin x}{x}$ at $x = 0$

(b) $f(x) = \lfloor x \rfloor$ (floor function) at $x = 2$

(c) $f(x) = \frac{1}{(x-1)^2}$ at $x = 1$

### Problem 4: IVT Application

Show that the equation $x^5 - 3x + 1 = 0$ has at least one root in the interval $[0, 1]$. Then use the bisection method (modify the code above) to find this root to 8 decimal places.

### Problem 5: Squeeze Theorem

Use the Squeeze Theorem to evaluate $\lim_{x \to 0} x^2 \cos(1/x^2)$. Then write Python code to visualize the function and its bounding functions on $[-0.5, 0.5]$.

## References

- Stewart, *Calculus: Early Transcendentals*, Ch. 2 (Limits and Derivatives)
- [3Blue1Brown: Limits](https://www.youtube.com/watch?v=kfF40MiS7zA)
- [Paul's Online Notes: Limits](https://tutorial.math.lamar.edu/Classes/CalcI/Limits.aspx)

---

[Previous: Course Overview](./00_Overview.md) | [Next: Derivatives Fundamentals](./02_Derivatives_Fundamentals.md)
