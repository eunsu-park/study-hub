# Applications of Derivatives

## Learning Objectives

- **Identify** critical points and classify them as local maxima, minima, or saddle points using the first and second derivative tests
- **Solve** optimization problems by translating real-world constraints into mathematical formulations
- **Apply** related rates techniques to problems where multiple quantities change simultaneously
- **Evaluate** indeterminate forms using L'Hopital's rule
- **Construct** linear approximations and Taylor polynomials to approximate function values near a point

## Introduction

Knowing how to compute derivatives is like knowing how to read a map. The real power comes from using that skill to navigate -- to find the highest mountain, predict how fast a shadow moves, or estimate a function's value without a calculator. This lesson covers the major applications of derivatives that appear throughout science and engineering.

## Critical Points and Extrema

### Finding Critical Points

A **critical point** of $f$ is a value $c$ in the domain where either $f'(c) = 0$ or $f'(c)$ does not exist.

**Why these matter:** The Extreme Value Theorem guarantees that a continuous function on a closed interval $[a, b]$ attains both a maximum and a minimum. These extreme values occur either at critical points or at the endpoints.

### First Derivative Test

Examine the sign of $f'(x)$ on either side of a critical point $c$:

| $f'$ before $c$ | $f'$ after $c$ | Conclusion |
|------------------|-----------------|------------|
| $+$ (increasing) | $-$ (decreasing) | Local **maximum** at $c$ |
| $-$ (decreasing) | $+$ (increasing) | Local **minimum** at $c$ |
| Same sign | Same sign | Neither (inflection possible) |

### Second Derivative Test

If $f'(c) = 0$ and $f''$ exists at $c$:

- $f''(c) > 0$: local **minimum** (curve is concave up, like a bowl)
- $f''(c) < 0$: local **maximum** (curve is concave down, like a hill)
- $f''(c) = 0$: test is **inconclusive** (use first derivative test instead)

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = sp.Symbol('x')

# Example: f(x) = x^3 - 3x^2 + 1
f = x**3 - 3*x**2 + 1
f_prime = sp.diff(f, x)
f_double_prime = sp.diff(f, x, 2)

print(f"f(x) = {f}")
print(f"f'(x) = {f_prime}")
print(f"f''(x) = {f_double_prime}")

# Find critical points: where f'(x) = 0
critical_points = sp.solve(f_prime, x)
print(f"\nCritical points: {critical_points}")

for cp in critical_points:
    second_deriv_val = f_double_prime.subs(x, cp)
    if second_deriv_val > 0:
        classification = "local minimum"
    elif second_deriv_val < 0:
        classification = "local maximum"
    else:
        classification = "inconclusive"
    print(f"  x = {cp}: f''({cp}) = {second_deriv_val} --> {classification}")
    print(f"    f({cp}) = {f.subs(x, cp)}")

# Visualization
x_vals = np.linspace(-1, 4, 500)
f_np = lambda t: t**3 - 3*t**2 + 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Function plot
ax1.plot(x_vals, f_np(x_vals), 'b-', linewidth=2, label='$f(x) = x^3 - 3x^2 + 1$')
for cp in critical_points:
    cp_float = float(cp)
    ax1.plot(cp_float, f_np(cp_float), 'ro', markersize=10, zorder=5)
    ax1.annotate(f'({cp_float}, {f_np(cp_float):.0f})',
                 (cp_float, f_np(cp_float)), textcoords="offset points",
                 xytext=(15, 10), fontsize=11)
ax1.set_ylabel('$f(x)$')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('Critical Points and Extrema')

# First derivative plot
f_prime_np = lambda t: 3*t**2 - 6*t
ax2.plot(x_vals, f_prime_np(x_vals), 'r-', linewidth=2, label="$f'(x) = 3x^2 - 6x$")
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.fill_between(x_vals, f_prime_np(x_vals), 0,
                  where=(f_prime_np(x_vals) > 0), alpha=0.2, color='green',
                  label='$f\' > 0$ (increasing)')
ax2.fill_between(x_vals, f_prime_np(x_vals), 0,
                  where=(f_prime_np(x_vals) < 0), alpha=0.2, color='red',
                  label='$f\' < 0$ (decreasing)')
ax2.set_xlabel('$x$')
ax2.set_ylabel("$f'(x)$")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('critical_points.png', dpi=150)
plt.show()
```

## Optimization Problems

Optimization is perhaps the most practical application of derivatives. The strategy is always the same:

1. **Draw a picture** and identify variables
2. **Write** the quantity to optimize as a function of one variable
3. **Find** the domain (physical constraints)
4. **Differentiate**, set equal to zero, solve
5. **Verify** the result is a max/min (second derivative test or endpoint check)

### Example: Maximizing Enclosed Area

A farmer has 200 meters of fencing and wants to enclose the largest possible rectangular area against a river (no fence needed on the river side).

**Setup:** Let $x$ = width (perpendicular to river), $y$ = length (parallel to river).

Constraint: $2x + y = 200$, so $y = 200 - 2x$.

Objective: Maximize $A = xy = x(200 - 2x) = 200x - 2x^2$.

$$A'(x) = 200 - 4x = 0 \implies x = 50$$

$$A''(x) = -4 < 0 \quad \text{(confirms maximum)}$$

So $x = 50$ m, $y = 100$ m, $A_{\max} = 5000$ m$^2$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Optimization: maximize area A(x) = x(200 - 2x)
x = np.linspace(0, 100, 500)
A = x * (200 - 2*x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, A, 'b-', linewidth=2)
ax.plot(50, 5000, 'ro', markersize=10, zorder=5)
ax.annotate('Maximum: (50, 5000)', (50, 5000),
            textcoords="offset points", xytext=(20, -20),
            fontsize=12, arrowprops=dict(arrowstyle='->', color='red'))
ax.set_xlabel('Width $x$ (meters)')
ax.set_ylabel('Area $A$ (m$^2$)')
ax.set_title('Fencing Problem: Area vs Width')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('optimization_fencing.png', dpi=150)
plt.show()
```

### Example: Minimizing Material Cost

Design an open-top cylindrical can of volume $V = 500$ cm$^3$ that uses the least material.

**Setup:** Variables are radius $r$ and height $h$. The can has a bottom ($\pi r^2$) and a side ($2\pi r h$).

Constraint: $\pi r^2 h = 500$, so $h = \frac{500}{\pi r^2}$.

Surface area: $S = \pi r^2 + 2\pi r h = \pi r^2 + \frac{1000}{r}$

$$S'(r) = 2\pi r - \frac{1000}{r^2} = 0 \implies r^3 = \frac{500}{\pi} \implies r = \left(\frac{500}{\pi}\right)^{1/3}$$

```python
import numpy as np
import sympy as sp

r = sp.Symbol('r', positive=True)
V = 500

# Surface area as a function of r only (h eliminated using volume constraint)
S = sp.pi * r**2 + 1000 / r
dS = sp.diff(S, r)
r_opt = sp.solve(dS, r)[0]
h_opt = V / (sp.pi * r_opt**2)

print(f"Optimal radius: r = {r_opt} = {float(r_opt):.4f} cm")
print(f"Optimal height: h = {float(h_opt):.4f} cm")
print(f"Minimum surface area: {float(S.subs(r, r_opt)):.4f} cm^2")
print(f"Ratio h/r = {float(h_opt / r_opt):.4f}")
# Notice: h/r = 2 at the optimum -- the height equals the diameter!
```

## Related Rates

In related rates problems, several quantities change with time, and we know the rate of one and want to find the rate of another.

**Strategy:** Write an equation relating the quantities, differentiate both sides with respect to time $t$ using the chain rule, then substitute known values.

### Example: Expanding Circle

A stone dropped in a pond creates a circular ripple whose radius increases at 3 cm/s. How fast is the area increasing when $r = 10$ cm?

**Equation:** $A = \pi r^2$

**Differentiate with respect to $t$:**

$$\frac{dA}{dt} = 2\pi r \frac{dr}{dt}$$

- $\frac{dr}{dt} = 3$ cm/s (given)
- $r = 10$ cm (given instant)

$$\frac{dA}{dt} = 2\pi(10)(3) = 60\pi \approx 188.5 \text{ cm}^2/\text{s}$$

**Intuition:** The area grows faster as the circle gets larger because the circumference (the "perimeter being pushed outward") is longer.

### Example: Ladder Problem

A 10-meter ladder leans against a wall. The bottom slides away at 1 m/s. How fast is the top sliding down when the bottom is 6 m from the wall?

$$x^2 + y^2 = 100$$

$$2x \frac{dx}{dt} + 2y \frac{dy}{dt} = 0$$

When $x = 6$: $y = \sqrt{100 - 36} = 8$.

$$2(6)(1) + 2(8)\frac{dy}{dt} = 0 \implies \frac{dy}{dt} = -\frac{3}{4} \text{ m/s}$$

The negative sign means $y$ is decreasing (the top slides *down*).

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize the ladder problem over time
# x(t) = initial_x + t (bottom moves at 1 m/s)
# y(t) = sqrt(100 - x(t)^2) (Pythagorean constraint)

t = np.linspace(0, 3.5, 100)
x0 = 6.0
x_t = x0 + t
y_t = np.sqrt(np.maximum(100 - x_t**2, 0))

# Rate of top sliding: dy/dt = -x * (dx/dt) / y
dx_dt = 1.0
dy_dt = -x_t * dx_dt / np.where(y_t > 0, y_t, np.nan)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Position plot
for i in range(0, len(t), 15):
    alpha = 0.3 + 0.7 * i / len(t)
    ax1.plot([x_t[i], 0], [0, y_t[i]], 'b-', alpha=alpha, linewidth=2)
ax1.set_xlabel('$x$ (m)')
ax1.set_ylabel('$y$ (m)')
ax1.set_title('Ladder sliding along wall')
ax1.set_aspect('equal')
ax1.set_xlim(-0.5, 11)
ax1.set_ylim(-0.5, 11)
ax1.grid(True, alpha=0.3)

# Rate plot -- shows dy/dt accelerates as the ladder falls
ax2.plot(t, dy_dt, 'r-', linewidth=2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('$dy/dt$ (m/s)')
ax2.set_title('Rate of top sliding down')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linewidth=0.5)
# Note: the top accelerates downward -- the rate becomes more negative
# over time. As y -> 0, the speed approaches infinity (the ladder
# "slaps" the ground), which is physically unrealistic but mathematically exact.

plt.tight_layout()
plt.savefig('related_rates_ladder.png', dpi=150)
plt.show()
```

## L'Hopital's Rule

When a limit yields an **indeterminate form** ($\frac{0}{0}$ or $\frac{\infty}{\infty}$), L'Hopital's rule provides an elegant escape:

$$\text{If } \lim_{x \to a} \frac{f(x)}{g(x)} \text{ is } \frac{0}{0} \text{ or } \frac{\infty}{\infty}, \text{ then } \lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$

provided the limit on the right exists.

**Examples:**

$$\lim_{x \to 0} \frac{\sin x}{x} = \lim_{x \to 0} \frac{\cos x}{1} = 1$$

$$\lim_{x \to \infty} \frac{\ln x}{x} = \lim_{x \to \infty} \frac{1/x}{1} = 0$$

$$\lim_{x \to 0} \frac{e^x - 1 - x}{x^2} \stackrel{\frac{0}{0}}{=} \lim_{x \to 0} \frac{e^x - 1}{2x} \stackrel{\frac{0}{0}}{=} \lim_{x \to 0} \frac{e^x}{2} = \frac{1}{2}$$

**Caution:** Other indeterminate forms ($0 \cdot \infty$, $\infty - \infty$, $0^0$, $1^\infty$, $\infty^0$) must first be rewritten into $\frac{0}{0}$ or $\frac{\infty}{\infty}$ form.

```python
import sympy as sp

x = sp.Symbol('x')

# Verify L'Hopital examples with SymPy
examples = [
    (sp.sin(x) / x, 0, "sin(x)/x"),
    (sp.log(x) / x, sp.oo, "ln(x)/x"),
    ((sp.exp(x) - 1 - x) / x**2, 0, "(e^x - 1 - x)/x^2"),
    ((1 - sp.cos(x)) / x**2, 0, "(1 - cos(x))/x^2"),
]

for expr, point, name in examples:
    result = sp.limit(expr, x, point)
    print(f"lim {name} as x->{point}: {result}")
```

## Linear Approximation and Taylor Polynomials

### Linear Approximation

Near $x = a$, a differentiable function is well approximated by its tangent line:

$$f(x) \approx f(a) + f'(a)(x - a) \quad \text{for } x \text{ near } a$$

This is called the **linearization** or **first-order Taylor approximation**. It is the basis for why differential equations can often be solved by linearization, and why Newton's method works.

**Example:** Estimate $\sqrt{4.1}$ without a calculator.

Let $f(x) = \sqrt{x}$, $a = 4$. Then $f(4) = 2$, $f'(x) = \frac{1}{2\sqrt{x}}$, $f'(4) = \frac{1}{4}$.

$$\sqrt{4.1} \approx 2 + \frac{1}{4}(4.1 - 4) = 2 + 0.025 = 2.025$$

Actual value: $\sqrt{4.1} = 2.02485...$. The linear approximation is accurate to 3 decimal places.

### Taylor Polynomials

For better accuracy, we include higher-order terms. The **$n$th-degree Taylor polynomial** of $f$ centered at $a$ is:

$$T_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x - a)^k = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots$$

- $f^{(k)}(a)$: the $k$th derivative evaluated at $a$
- $k!$: factorial, ensuring the coefficients match the function's behavior
- $(x - a)^k$: each term captures finer local detail

When $a = 0$, this is called a **Maclaurin polynomial**.

### Newton's Method

Newton's method uses linear approximation iteratively to find roots:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

**Geometric view:** At each step, we replace the curve with its tangent line and find where the tangent crosses the $x$-axis.

```python
import numpy as np
import matplotlib.pyplot as plt

def newtons_method(f, df, x0, tol=1e-12, max_iter=50, verbose=True):
    """
    Newton's method for finding roots of f(x) = 0.

    Each iteration replaces f with its tangent line at the current
    estimate and solves for where the tangent crosses zero. This
    gives quadratic convergence -- the number of correct digits
    roughly doubles each iteration.
    """
    x = x0
    history = [x]

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-15:
            print("WARNING: derivative near zero, method may fail")
            break
        x_new = x - fx / dfx
        history.append(x_new)

        if verbose:
            print(f"  Iter {i+1}: x = {x_new:.15f}, f(x) = {f(x_new):.2e}")

        if abs(x_new - x) < tol:
            break
        x = x_new

    return x_new, history

# Find sqrt(2) using Newton's method on f(x) = x^2 - 2
print("Finding sqrt(2):")
f = lambda x: x**2 - 2
df = lambda x: 2*x
root, hist = newtons_method(f, df, x0=1.0)
print(f"\nResult: {root:.15f}")
print(f"Actual: {np.sqrt(2):.15f}")

# Visualize convergence -- note how fast it converges
errors = [abs(x - np.sqrt(2)) for x in hist]
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(range(len(errors)), errors, 'bo-', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('|Error|')
ax.set_title("Newton's Method: Quadratic Convergence")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('newtons_method.png', dpi=150)
plt.show()
```

## Summary

- **Critical points** occur where $f'(x) = 0$ or $f'$ doesn't exist; classify using the first or second derivative test
- **Optimization** translates real-world constraints into calculus: express the objective as a function of one variable, then find critical points
- **Related rates** use the chain rule to connect rates of change of linked quantities
- **L'Hopital's rule** resolves $\frac{0}{0}$ and $\frac{\infty}{\infty}$ indeterminate forms by differentiating numerator and denominator
- **Linear approximation** ($f(x) \approx f(a) + f'(a)(x-a)$) is the simplest Taylor polynomial and the foundation of Newton's method
- **Newton's method** achieves quadratic convergence, roughly doubling the number of correct digits each iteration

## Practice Problems

### Problem 1: Finding and Classifying Extrema

Find all critical points of $f(x) = x^4 - 4x^3 + 4x^2$ and classify each as a local maximum, local minimum, or neither. Sketch the function to verify.

### Problem 2: Optimization

A rectangular box with a square base and open top must have a volume of 32,000 cm$^3$. Find the dimensions that minimize the amount of material used (i.e., minimize the surface area).

### Problem 3: Related Rates

Air is pumped into a spherical balloon at a rate of 100 cm$^3$/s. How fast is the radius increasing when the diameter is 50 cm? ($V = \frac{4}{3}\pi r^3$)

### Problem 4: L'Hopital's Rule

Evaluate each limit:

(a) $\lim_{x \to 0} \frac{e^x - 1 - x - x^2/2}{x^3}$

(b) $\lim_{x \to 0^+} x \ln x$ (Hint: rewrite as $\frac{\ln x}{1/x}$)

(c) $\lim_{x \to \infty} x^{1/x}$ (Hint: let $y = x^{1/x}$ and take $\ln$)

### Problem 5: Taylor Polynomial Approximation

(a) Write the 4th-degree Maclaurin polynomial for $f(x) = e^x$. Use it to estimate $e^{0.5}$ and compare with the true value.

(b) Write Python code to plot $\sin(x)$ along with its Taylor polynomials of degree 1, 3, 5, 7, and 9 on $[-2\pi, 2\pi]$. Observe how higher-degree polynomials approximate the function over a wider range.

## References

- Stewart, *Calculus: Early Transcendentals*, Ch. 4 (Applications of Differentiation)
- [3Blue1Brown: Optimization](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- [Paul's Online Notes: Applications of Derivatives](https://tutorial.math.lamar.edu/Classes/CalcI/DerivAppsIntro.aspx)

---

[Previous: Derivatives Fundamentals](./02_Derivatives_Fundamentals.md) | [Next: Integration Fundamentals](./04_Integration_Fundamentals.md)
