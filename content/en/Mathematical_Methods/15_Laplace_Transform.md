# 15. Laplace Transform

## Learning Objectives

- Understand the definition and existence conditions of the Laplace transform, and explain the concept of the region of convergence
- Derive Laplace transforms of basic functions and use transform tables to calculate transforms of complex functions
- Prove and apply major properties of the Laplace transform: shifting theorems, differentiation properties, convolution theorem, etc.
- Systematically solve initial value problems for ordinary differential equations using partial fraction decomposition and inverse Laplace transforms
- Understand the basic principles of linear system analysis and stability determination using transfer functions
- Apply Laplace transforms to physics and engineering problems such as RLC circuits and damped oscillations

> **Importance in physics and engineering**: The Laplace transform is a core tool that enables systematic solution of initial value problems by converting differential equations into algebraic equations. It is essential in nearly all engineering fields including circuit analysis, control engineering, signal processing, mechanical vibration, and heat conduction. As a generalization of the Fourier transform, it is particularly powerful for analyzing transient responses.

---

## 1. Definition and Existence Conditions

### 1.1 Definition

The **Laplace transform** of a function $f(t)$ ($t \geq 0$) is defined as:

$$F(s) = \mathcal{L}\{f(t)\} = \int_0^\infty f(t) e^{-st} \, dt$$

where $s = \sigma + i\omega$ is a complex variable. $F(s)$ is defined in the region of $s$ where this integral converges.

Intuitively, the Laplace transform converts a function $f(t)$ in the time domain to a function $F(s)$ in the complex frequency domain. The real part $\sigma$ represents exponential decay/growth, and the imaginary part $\omega$ represents oscillation.

### 1.2 Existence Conditions

**Sufficient conditions** for the existence of the Laplace transform:

1. **Piecewise continuous**: $f(t)$ is continuous except at a finite number of discontinuities on any finite interval $[0, T]$
2. **Exponential order**: There exist constants $M > 0$, $a \in \mathbb{R}$, $T > 0$ such that

$$|f(t)| \leq M e^{at}, \quad t > T$$

If these conditions are satisfied, the integral converges for all $s$ with $\text{Re}(s) > a$.

### 1.3 Region of Convergence

The region of convergence (ROC) is the region in the complex plane where the Laplace integral converges:

$$\text{ROC} = \{ s \in \mathbb{C} \mid \text{Re}(s) > \sigma_c \}$$

where $\sigma_c$ is called the **abscissa of convergence**. For example:
- $f(t) = 1$: $\sigma_c = 0$ (i.e., $\text{Re}(s) > 0$)
- $f(t) = e^{at}$: $\sigma_c = a$ (i.e., $\text{Re}(s) > a$)
- $f(t) = e^{-t}\sin t$: $\sigma_c = -1$

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Computing Laplace transforms using SymPy
t, s = sp.symbols('t s', positive=True)

# Laplace transforms of basic functions
funcs = {
    '1': 1,
    't': t,
    't^2': t**2,
    'exp(-2t)': sp.exp(-2*t),
    'sin(3t)': sp.sin(3*t),
    'cos(3t)': sp.cos(3*t),
}

print("=" * 50)
print("Laplace transforms of basic functions")
print("=" * 50)
for name, f in funcs.items():
    F = sp.laplace_transform(f, t, s, noconds=True)
    print(f"L{{{name}}} = {F}")
```

### 1.4 Relationship with Fourier Transform

The Laplace transform is a generalization of the Fourier transform. Setting $s = i\omega$ (and if $f(t) = 0$ for $t < 0$):

$$F(i\omega) = \int_0^\infty f(t) e^{-i\omega t} \, dt$$

This is identical to the one-sided Fourier transform. The Laplace transform multiplies by an additional convergence factor $e^{-\sigma t}$, making it applicable to functions where the Fourier transform doesn't exist (e.g., $f(t) = e^{2t}$).

---

## 2. Laplace Transforms of Basic Functions

### 2.1 Transform Table

The table below summarizes the most important Laplace transform pairs:

| $f(t)$ ($t \geq 0$) | $F(s) = \mathcal{L}\{f(t)\}$ | Convergence condition |
|---|---|---|
| $1$ | $\dfrac{1}{s}$ | $\text{Re}(s) > 0$ |
| $t^n$ ($n = 0, 1, 2, \ldots$) | $\dfrac{n!}{s^{n+1}}$ | $\text{Re}(s) > 0$ |
| $e^{at}$ | $\dfrac{1}{s - a}$ | $\text{Re}(s) > a$ |
| $\sin(bt)$ | $\dfrac{b}{s^2 + b^2}$ | $\text{Re}(s) > 0$ |
| $\cos(bt)$ | $\dfrac{s}{s^2 + b^2}$ | $\text{Re}(s) > 0$ |
| $\sinh(bt)$ | $\dfrac{b}{s^2 - b^2}$ | $\text{Re}(s) > |b|$ |
| $\cosh(bt)$ | $\dfrac{s}{s^2 - b^2}$ | $\text{Re}(s) > |b|$ |
| $t^n e^{at}$ | $\dfrac{n!}{(s-a)^{n+1}}$ | $\text{Re}(s) > a$ |
| $e^{at}\sin(bt)$ | $\dfrac{b}{(s-a)^2 + b^2}$ | $\text{Re}(s) > a$ |
| $e^{at}\cos(bt)$ | $\dfrac{s-a}{(s-a)^2 + b^2}$ | $\text{Re}(s) > a$ |

**Example derivation**: Computing $\mathcal{L}\{e^{at}\}$ directly

$$\int_0^\infty e^{at} e^{-st} \, dt = \int_0^\infty e^{-(s-a)t} \, dt = \left[ \frac{e^{-(s-a)t}}{-(s-a)} \right]_0^\infty = \frac{1}{s-a}$$

provided that $\text{Re}(s) > a$ so the exponential converges to 0 as $t \to \infty$.

### 2.2 Unit Step Function (Heaviside Function)

**Heaviside function** definition:

$$u(t - a) = \begin{cases} 0, & t < a \\ 1, & t \geq a \end{cases}$$

Laplace transform:

$$\mathcal{L}\{u(t-a)\} = \int_a^\infty e^{-st} \, dt = \frac{e^{-as}}{s}, \quad a \geq 0$$

### 2.3 Dirac Delta Function

$\delta(t - a)$ represents a unit impulse at $t = a$:

$$\mathcal{L}\{\delta(t - a)\} = \int_0^\infty \delta(t - a) e^{-st} \, dt = e^{-as}, \quad a \geq 0$$

In particular, when $a = 0$, $\mathcal{L}\{\delta(t)\} = 1$. This means the Laplace transform of an impulse input is the constant 1.

```python
import sympy as sp

t, s, a = sp.symbols('t s a', positive=True)

# Laplace transform of the unit step function
heaviside_transform = sp.laplace_transform(sp.Heaviside(t - a), t, s, noconds=True)
print(f"L{{u(t-a)}} = {heaviside_transform}")

# Laplace transform of the Dirac delta function
delta_transform = sp.laplace_transform(sp.DiracDelta(t - a), t, s, noconds=True)
print(f"L{{delta(t-a)}} = {delta_transform}")

# Numerical verification of basic transform pair: L{sin(3t)} = 3/(s^2+9)
import numpy as np
from scipy import integrate

def numerical_laplace(f, s_val, upper=50):
    """Numerical computation of the Laplace transform"""
    integrand = lambda tau: f(tau) * np.exp(-s_val * tau)
    result, _ = integrate.quad(integrand, 0, upper)
    return result

s_test = 2.0
# Numerical result
numerical = numerical_laplace(lambda tau: np.sin(3*tau), s_test)
# Analytical result: 3/(s^2+9)
analytical = 3 / (s_test**2 + 9)

print(f"\nL{{sin(3t)}} verification (s={s_test}):")
print(f"  Numerical integration: {numerical:.6f}")
print(f"  Analytical:            {analytical:.6f}")
print(f"  Error:                 {abs(numerical - analytical):.2e}")
```

---

## 3. Properties of the Laplace Transform

### 3.1 Linearity

The Laplace transform is a **linear operator**:

$$\mathcal{L}\{\alpha f(t) + \beta g(t)\} = \alpha F(s) + \beta G(s)$$

This follows directly from the linearity of integration.

### 3.2 First Shifting Theorem (s-shift)

If $\mathcal{L}\{f(t)\} = F(s)$, then:

$$\mathcal{L}\{e^{at}f(t)\} = F(s - a)$$

**Proof**:

$$\mathcal{L}\{e^{at}f(t)\} = \int_0^\infty e^{at}f(t)e^{-st} \, dt = \int_0^\infty f(t)e^{-(s-a)t} \, dt = F(s-a)$$

**Application example**: To find $\mathcal{L}\{e^{-2t}\cos(3t)\}$

$$\mathcal{L}\{\cos(3t)\} = \frac{s}{s^2 + 9}$$

Replace $s$ with $s + 2$:

$$\mathcal{L}\{e^{-2t}\cos(3t)\} = \frac{s+2}{(s+2)^2 + 9}$$

### 3.3 Second Shifting Theorem (t-shift)

If $\mathcal{L}\{f(t)\} = F(s)$, then:

$$\mathcal{L}\{f(t-a)\,u(t-a)\} = e^{-as}F(s), \quad a > 0$$

**Proof**: Using the substitution $\tau = t - a$

$$\int_0^\infty f(t-a)\,u(t-a)\,e^{-st} \, dt = \int_a^\infty f(t-a)\,e^{-st} \, dt = \int_0^\infty f(\tau)\,e^{-s(\tau+a)} \, d\tau = e^{-as}F(s)$$

This theorem is fundamental for transforms of time-delayed signals.

### 3.4 Differentiation Property

The most powerful property of the Laplace transform is that **it converts differentiation into algebraic operations**:

$$\mathcal{L}\{f'(t)\} = sF(s) - f(0)$$

$$\mathcal{L}\{f''(t)\} = s^2 F(s) - sf(0) - f'(0)$$

Generally, the transform of the $n$-th derivative is:

$$\mathcal{L}\{f^{(n)}(t)\} = s^n F(s) - s^{n-1}f(0) - s^{n-2}f'(0) - \cdots - f^{(n-1)}(0)$$

**Proof** (first derivative): Applying integration by parts

$$\int_0^\infty f'(t)e^{-st} \, dt = \left[f(t)e^{-st}\right]_0^\infty + s\int_0^\infty f(t)e^{-st} \, dt = -f(0) + sF(s)$$

### 3.5 Integration Property

$$\mathcal{L}\left\{\int_0^t f(\tau) \, d\tau\right\} = \frac{F(s)}{s}$$

Integration in the time domain corresponds to division by $s$ in the $s$-domain.

### 3.6 Convolution Theorem

The **convolution** of two functions is:

$$(f * g)(t) = \int_0^t f(\tau) \, g(t - \tau) \, d\tau$$

**Convolution theorem**:

$$\mathcal{L}\{f * g\} = F(s) \cdot G(s)$$

That is, convolution in the time domain corresponds to multiplication in the $s$-domain.

**Proof**: Using interchange of integration order

$$\mathcal{L}\{f * g\} = \int_0^\infty \left(\int_0^t f(\tau)g(t-\tau) \, d\tau \right) e^{-st} \, dt$$

Substituting $u = t - \tau$ and interchanging integration order yields $F(s) \cdot G(s)$.

### 3.7 Initial Value and Final Value Theorems

**Initial value theorem**: $f(0^+) = \lim_{s \to \infty} sF(s)$

**Final value theorem**: $\lim_{t \to \infty} f(t) = \lim_{s \to 0} sF(s)$

However, the final value theorem is valid only when all poles of $sF(s)$ are in the left half of the complex plane (i.e., when the system is stable).

```python
import sympy as sp

t, s = sp.symbols('t s')
a_sym = sp.Symbol('a', positive=True)

# First shifting theorem verification: L{e^(-2t)*cos(3t)}
f1 = sp.exp(-2*t) * sp.cos(3*t)
F1_direct = sp.laplace_transform(f1, t, s, noconds=True)
F1_shift = (s + 2) / ((s + 2)**2 + 9)  # applying s-shift
print("=== First shifting theorem verification ===")
print(f"Direct transform: {F1_direct}")
print(f"Shifting theorem: {F1_shift}")
print(f"Identical:        {sp.simplify(F1_direct - F1_shift) == 0}")

# Differentiation property verification: L{f'(t)} = sF(s) - f(0)
# f(t) = t*exp(-t), f(0) = 0
f2 = t * sp.exp(-t)
f2_prime = sp.diff(f2, t)  # (1-t)*exp(-t)

F2 = sp.laplace_transform(f2, t, s, noconds=True)  # 1/(s+1)^2
F2_prime_direct = sp.laplace_transform(f2_prime, t, s, noconds=True)
F2_prime_property = s * F2 - 0  # f(0) = 0

print("\n=== Differentiation property verification ===")
print(f"L{{f'(t)}} direct:  {F2_prime_direct}")
print(f"sF(s) - f(0):      {sp.simplify(F2_prime_property)}")

# Convolution theorem verification: L{sin(t) * sin(t)} = [1/(s^2+1)]^2
conv_result = sp.laplace_transform(
    sp.integrate(sp.sin(t - sp.Symbol('tau')) * sp.sin(sp.Symbol('tau')),
                 (sp.Symbol('tau'), 0, t)), t, s, noconds=True
)
product_result = 1 / (s**2 + 1)**2
print("\n=== Convolution theorem ===")
print(f"F(s)*G(s) = 1/(s^2+1)^2 = {product_result}")
```

---

## 4. Inverse Laplace Transform

### 4.1 Partial Fraction Decomposition

The key technique for inverse Laplace transforms is to decompose $F(s)$ into partial fractions and then apply the transform table in reverse.

**Case 1**: Distinct real roots

$$\frac{P(s)}{(s-a_1)(s-a_2)\cdots(s-a_n)} = \frac{A_1}{s-a_1} + \frac{A_2}{s-a_2} + \cdots + \frac{A_n}{s-a_n}$$

**Case 2**: Repeated roots

$$\frac{P(s)}{(s-a)^n} = \frac{A_1}{s-a} + \frac{A_2}{(s-a)^2} + \cdots + \frac{A_n}{(s-a)^n}$$

**Case 3**: Complex conjugate roots

$$\frac{P(s)}{s^2 + bs + c} = \frac{As + B}{s^2 + bs + c}$$

Then complete the square and apply $\sin$, $\cos$ transform pairs.

### 4.2 Heaviside Cover-up Method

For distinct first-order factors, coefficients can be quickly found. The coefficient of the term with denominator $(s - a_k)$ in $F(s)$:

$$A_k = \left[(s - a_k) F(s)\right]_{s = a_k}$$

**Example**: Find the inverse transform of $F(s) = \dfrac{3s + 2}{(s+1)(s-2)}$

$$A_1 = \left[\frac{3s+2}{s-2}\right]_{s=-1} = \frac{-1}{-3} = \frac{1}{3}$$

$$A_2 = \left[\frac{3s+2}{s+1}\right]_{s=2} = \frac{8}{3}$$

Therefore $f(t) = \frac{1}{3}e^{-t} + \frac{8}{3}e^{2t}$

### 4.3 Bromwich Integral

The rigorous formula for the inverse Laplace transform is given by a complex integral:

$$f(t) = \mathcal{L}^{-1}\{F(s)\} = \frac{1}{2\pi i} \int_{\gamma - i\infty}^{\gamma + i\infty} F(s) e^{st} \, ds$$

where $\gamma$ is a real number greater than the real part of all singularities of $F(s)$. This integral can be computed using the residue theorem, a direct application of complex analysis from Chapter 12.

```python
import sympy as sp

s, t = sp.symbols('s t')

print("=== Partial fraction decomposition and inverse Laplace transform ===\n")

# Example 1: distinct real roots
F1 = (3*s + 2) / ((s + 1) * (s - 2))
print(f"F(s) = {F1}")
pf1 = sp.apart(F1, s)
print(f"Partial fractions: {pf1}")
f1 = sp.inverse_laplace_transform(F1, s, t)
print(f"f(t) = {f1}\n")

# Example 2: repeated root
F2 = (2*s + 3) / (s + 1)**2
print(f"F(s) = {F2}")
pf2 = sp.apart(F2, s)
print(f"Partial fractions: {pf2}")
f2 = sp.inverse_laplace_transform(F2, s, t)
print(f"f(t) = {f2}\n")

# Example 3: complex conjugate roots
F3 = (s + 3) / (s**2 + 2*s + 5)
print(f"F(s) = {F3}")
# Complete the square: (s+1)^2 + 4 -> e^(-t)cos(2t) + e^(-t)sin(2t)
f3 = sp.inverse_laplace_transform(F3, s, t)
print(f"f(t) = {f3}\n")

# Example 4: higher-order denominator
F4 = 1 / (s * (s**2 + 4))
print(f"F(s) = {F4}")
pf4 = sp.apart(F4, s)
print(f"Partial fractions: {pf4}")
f4 = sp.inverse_laplace_transform(F4, s, t)
print(f"f(t) = {f4}")
```

---

## 5. Solving Ordinary Differential Equations

### 5.1 Solution Procedure

General procedure for solving ODEs using the Laplace transform:

1. Apply the Laplace transform to both sides of the ODE
2. Use the differentiation property to substitute initial conditions
3. Solve algebraically for $Y(s)$
4. Find $y(t)$ by inverse Laplace transform

### 5.2 Second-Order Constant Coefficient ODE

**Example**: Solve the initial value problem

$$y'' + 3y' + 2y = 0, \quad y(0) = 1, \quad y'(0) = 0$$

**Solution**:

Apply Laplace transform to both sides:

$$[s^2 Y(s) - sy(0) - y'(0)] + 3[sY(s) - y(0)] + 2Y(s) = 0$$

Substitute initial conditions:

$$s^2 Y - s + 3sY - 3 + 2Y = 0$$

Solve for $Y(s)$:

$$Y(s)(s^2 + 3s + 2) = s + 3$$

$$Y(s) = \frac{s + 3}{s^2 + 3s + 2} = \frac{s + 3}{(s+1)(s+2)}$$

Partial fraction decomposition: $\frac{s+3}{(s+1)(s+2)} = \frac{2}{s+1} - \frac{1}{s+2}$

Inverse transform:

$$y(t) = 2e^{-t} - e^{-2t}$$

### 5.3 Nonhomogeneous ODE

**Example**: $y'' + y = \sin(2t)$, $y(0) = 0$, $y'(0) = 0$

Apply Laplace transform to both sides:

$$s^2 Y(s) + Y(s) = \frac{2}{s^2 + 4}$$

$$Y(s) = \frac{2}{(s^2 + 1)(s^2 + 4)}$$

Partial fraction decomposition:

$$\frac{2}{(s^2+1)(s^2+4)} = \frac{2}{3} \cdot \frac{1}{s^2+1} - \frac{2}{3} \cdot \frac{1}{s^2+4}$$

Inverse transform:

$$y(t) = \frac{2}{3}\sin t - \frac{1}{3}\sin 2t$$

### 5.4 Coupled Differential Equations

**Example**: Coupled system

$$\begin{cases} x' = 2x - y, \quad x(0) = 1 \\ y' = x, \quad\quad\;\;\; y(0) = 0 \end{cases}$$

Apply Laplace transform:

$$sX - 1 = 2X - Y \quad \Rightarrow \quad (s-2)X + Y = 1$$

$$sY = X \quad \Rightarrow \quad X = sY$$

Substituting:

$$s(s-2)Y + Y = 1 \quad \Rightarrow \quad Y(s) = \frac{1}{s^2 - 2s + 1} = \frac{1}{(s-1)^2}$$

$$X(s) = sY(s) = \frac{s}{(s-1)^2} = \frac{1}{s-1} + \frac{1}{(s-1)^2}$$

Inverse transform:

$$x(t) = e^t + te^t = (1 + t)e^t, \quad y(t) = te^t$$

```python
import sympy as sp

t, s = sp.symbols('t s')
Y = sp.Function('Y')

print("=== ODE solution: y'' + 3y' + 2y = 0, y(0)=1, y'(0)=0 ===\n")

# Method 1: Solve directly with SymPy's dsolve
y = sp.Function('y')
ode = sp.Eq(y(t).diff(t, 2) + 3*y(t).diff(t) + 2*y(t), 0)
sol = sp.dsolve(ode, y(t), ics={y(0): 1, y(t).diff(t).subs(t, 0): 0})
print(f"dsolve result: {sol}\n")

# Method 2: Step-by-step Laplace transform
print("--- Step-by-step Laplace transform solution ---")

# Find Y(s)
Ys = (s + 3) / (s**2 + 3*s + 2)
print(f"Y(s) = {Ys}")

# Partial fraction decomposition
pf = sp.apart(Ys, s)
print(f"Partial fractions: {pf}")

# Inverse Laplace transform
yt = sp.inverse_laplace_transform(Ys, s, t)
print(f"y(t) = {yt}\n")

# Verification: check initial conditions and ODE satisfaction
print("--- Verification ---")
print(f"y(0) = {yt.subs(t, 0)}")
print(f"y'(0) = {sp.diff(yt, t).subs(t, 0)}")
residual = sp.simplify(sp.diff(yt, t, 2) + 3*sp.diff(yt, t) + 2*yt)
print(f"y'' + 3y' + 2y = {residual}")

print("\n=== Nonhomogeneous ODE: y'' + y = sin(2t) ===\n")
Ys2 = 2 / ((s**2 + 1) * (s**2 + 4))
pf2 = sp.apart(Ys2, s)
print(f"Partial fractions: {pf2}")
yt2 = sp.inverse_laplace_transform(Ys2, s, t)
print(f"y(t) = {yt2}")
```

---

## 6. Transfer Functions and System Analysis

### 6.1 Transfer Function Definition

In linear time-invariant (LTI) systems, the **transfer function** is the ratio of the Laplace transforms of output to input:

$$H(s) = \frac{Y(s)}{X(s)}$$

where all initial conditions are assumed to be zero. The transfer function represents the intrinsic characteristics of the system, independent of the input.

### 6.2 Poles and Zeros

- **Zeros**: Values of $s$ where $H(s) = 0$ (roots of the numerator)
- **Poles**: Values of $s$ where $H(s) \to \infty$ (roots of the denominator)

The locations of the poles determine the dynamic characteristics of the system.

### 6.3 Impulse Response and Step Response

- **Impulse response** $h(t) = \mathcal{L}^{-1}\{H(s)\}$: output when input is $\delta(t)$
- **Step response**: output when input is $u(t)$, $Y(s) = H(s)/s$

### 6.4 Stability Analysis

For a system to be **BIBO stable (Bounded-Input Bounded-Output stable)**:

> All poles of the transfer function $H(s)$ must be located in the **left half of the complex plane** (i.e., $\text{Re}(s) < 0$).

Response characteristics based on pole locations:
- **Left half-plane** ($\text{Re}(s) < 0$): Decay → stable
- **Imaginary axis** ($\text{Re}(s) = 0$): Sustained oscillation → marginally stable
- **Right half-plane** ($\text{Re}(s) > 0$): Divergence → unstable

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Second-order system: H(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
omega_n = 2.0  # natural frequency
zeta_values = [0.1, 0.3, 0.7, 1.0, 2.0]  # damping ratios

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for zeta in zeta_values:
    # Define transfer function
    num = [omega_n**2]
    den = [1, 2*zeta*omega_n, omega_n**2]
    sys = signal.TransferFunction(num, den)

    # Step response
    t_step, y_step = signal.step(sys)
    axes[0].plot(t_step, y_step, label=f'zeta={zeta}')

    # Plot poles
    poles = np.roots(den)
    axes[1].plot(poles.real, poles.imag, 'x', markersize=10,
                 label=f'zeta={zeta}: {poles[0]:.2f}')

axes[0].set_xlabel('Time t')
axes[0].set_ylabel('y(t)')
axes[0].set_title('Step response (variation with damping ratio)')
axes[0].legend()
axes[0].grid(True)
axes[0].axhline(y=1, color='k', linestyle='--', alpha=0.3)

axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[1].set_xlabel('Re(s)')
axes[1].set_ylabel('Im(s)')
axes[1].set_title('Pole locations (s-plane)')
axes[1].legend()
axes[1].grid(True)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('transfer_function_analysis.png', dpi=150)
plt.show()
print("As damping ratio increases, poles approach the real axis and response stabilizes faster")
```

---

## 7. Physical Applications

### 7.1 RLC Circuit Analysis

Applying Kirchhoff's law to a series RLC circuit:

$$L\frac{di}{dt} + Ri + \frac{1}{C}\int_0^t i(\tau) \, d\tau = V(t)$$

In terms of charge $q$ (where $i = dq/dt$):

$$L q'' + R q' + \frac{q}{C} = V(t)$$

Applying Laplace transform (with $q(0) = 0$, $q'(0) = 0$):

$$\left(Ls^2 + Rs + \frac{1}{C}\right) Q(s) = V(s)$$

**Transfer function**:

$$H(s) = \frac{Q(s)}{V(s)} = \frac{1}{Ls^2 + Rs + \frac{1}{C}} = \frac{1/L}{s^2 + \frac{R}{L}s + \frac{1}{LC}}$$

Setting natural frequency $\omega_0 = 1/\sqrt{LC}$ and damping ratio $\zeta = R/(2\sqrt{L/C})$ gives the same form as a standard second-order system.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# RLC circuit parameters
R = 10       # Ohm (resistance)
L = 0.1      # H (inductance)
C = 1e-3     # F (capacitance)

omega_0 = 1 / np.sqrt(L * C)  # natural frequency
zeta = R / (2 * np.sqrt(L / C))  # damping ratio
print(f"RLC circuit parameters:")
print(f"  Natural frequency omega_0 = {omega_0:.2f} rad/s")
print(f"  Damping ratio zeta = {zeta:.4f}")
print(f"  State: {'overdamped' if zeta > 1 else 'critically damped' if zeta == 1 else 'underdamped'}")

# Transfer function: H(s) = (1/L) / (s^2 + (R/L)s + 1/(LC))
num = [1/L]
den = [1, R/L, 1/(L*C)]
sys_rlc = signal.TransferFunction(num, den)

# Response to unit step voltage input (switch ON)
t_sim = np.linspace(0, 0.1, 1000)
t_out, q_out = signal.step(sys_rlc, T=t_sim)

# Current i(t) = dq/dt
i_out = np.gradient(q_out, t_out)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(t_out * 1000, q_out * 1e6, 'b-', linewidth=2)
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Charge q (uC)')
axes[0].set_title('RLC series circuit - step response')
axes[0].grid(True)

axes[1].plot(t_out * 1000, i_out * 1000, 'r-', linewidth=2)
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Current i (mA)')
axes[1].set_title('Current response')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('rlc_circuit_response.png', dpi=150)
plt.show()
```

### 7.2 Damped Oscillation (Mass-Spring-Damper System)

A system with mass $m$, spring constant $k$, and damping coefficient $c$:

$$m\ddot{x} + c\dot{x} + kx = F(t)$$

For free vibration ($F = 0$) with initial displacement $x_0$:

$$ms^2 X - msx_0 + csX - cx_0 + kX = 0$$

$$X(s) = \frac{(ms + c)x_0}{ms^2 + cs + k} = \frac{x_0(s + c/m)}{s^2 + (c/m)s + k/m}$$

With damping ratio $\zeta = c/(2\sqrt{mk})$, natural frequency $\omega_n = \sqrt{k/m}$, and damped frequency $\omega_d = \omega_n\sqrt{1-\zeta^2}$:

$$x(t) = x_0 e^{-\zeta\omega_n t}\left(\cos\omega_d t + \frac{\zeta}{\sqrt{1-\zeta^2}}\sin\omega_d t\right)$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Mass-spring-damper parameters
m = 1.0   # kg
k = 100.0  # N/m
x0 = 0.05  # m (initial displacement 5cm)

# Various damping conditions
c_values = [0.5, 5.0, 20.0, 25.0]  # N*s/m

t = np.linspace(0, 3, 1000)
plt.figure(figsize=(10, 6))

for c in c_values:
    omega_n = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(m * k))

    if zeta < 1:  # underdamped
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        x = x0 * np.exp(-zeta * omega_n * t) * (
            np.cos(omega_d * t) +
            (zeta / np.sqrt(1 - zeta**2)) * np.sin(omega_d * t)
        )
        label = f'c={c} (zeta={zeta:.2f}, underdamped)'
    elif zeta == 1:  # critically damped
        x = x0 * (1 + omega_n * t) * np.exp(-omega_n * t)
        label = f'c={c} (zeta={zeta:.2f}, critically damped)'
    else:  # overdamped
        r1 = -zeta * omega_n + omega_n * np.sqrt(zeta**2 - 1)
        r2 = -zeta * omega_n - omega_n * np.sqrt(zeta**2 - 1)
        A = x0 * r2 / (r2 - r1)
        B = -x0 * r1 / (r2 - r1)
        x = A * np.exp(r1 * t) + B * np.exp(r2 * t)
        label = f'c={c} (zeta={zeta:.2f}, overdamped)'

    plt.plot(t, x * 100, linewidth=2, label=label)

plt.xlabel('Time (s)')
plt.ylabel('Displacement (cm)')
plt.title('Free vibration of mass-spring-damper system')
plt.legend()
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.savefig('damped_oscillation.png', dpi=150)
plt.show()
```

### 7.3 Heat Conduction Problem

Heat conduction equation when temperature $T_0$ is suddenly applied to one end of a rod of length $L$:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

Applying Laplace transform with respect to time (with $u(x, 0) = 0$):

$$sU(x, s) = \alpha \frac{d^2 U}{dx^2}$$

This is an ODE in $x$:

$$U(x, s) = A e^{-x\sqrt{s/\alpha}} + B e^{x\sqrt{s/\alpha}}$$

Applying boundary conditions $U(0, s) = T_0/s$, $U(\infty, s) = 0$:

$$U(x, s) = \frac{T_0}{s} e^{-x\sqrt{s/\alpha}}$$

Inverse transform:

$$u(x, t) = T_0 \, \text{erfc}\left(\frac{x}{2\sqrt{\alpha t}}\right)$$

where $\text{erfc}$ is the complementary error function.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Heat conduction parameters
T0 = 100.0        # boundary temperature (deg C)
alpha = 1.14e-4   # thermal diffusivity (m^2/s, iron)

# Temperature distribution at different times
x = np.linspace(0, 0.1, 200)  # position (m)
times = [1, 10, 60, 300, 1800]  # time (s)

plt.figure(figsize=(10, 6))
for t_val in times:
    u = T0 * erfc(x / (2 * np.sqrt(alpha * t_val)))
    plt.plot(x * 100, u, linewidth=2, label=f't = {t_val}s')

plt.xlabel('Position x (cm)')
plt.ylabel('Temperature u (deg C)')
plt.title('Heat conduction in a semi-infinite rod (Laplace transform solution)')
plt.legend()
plt.grid(True)
plt.savefig('heat_conduction_laplace.png', dpi=150)
plt.show()
```

---

## 8. Numerical Inverse Laplace Transform

### 8.1 Need

In many practical problems, the analytical inverse transform of $F(s)$ is impossible or very complex. In such cases, numerical inverse Laplace transform algorithms are needed.

### 8.2 Stehfest Algorithm

The Stehfest algorithm approximates $f(t)$ using only values of $F(s)$ on the real axis:

$$f(t) \approx \frac{\ln 2}{t} \sum_{k=1}^{N} V_k \, F\left(\frac{k \ln 2}{t}\right)$$

where the weights $V_k$ are computed using binomial coefficients. $N$ is chosen to be even (typically $N = 10 \sim 18$).

### 8.3 Talbot Method

The Talbot method is a deformation of the Bromwich integral path, numerically integrating along a parabolic path in the complex plane. It is generally more accurate than the Stehfest method.

```python
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def stehfest_weights(N):
    """Compute weights for the Stehfest algorithm"""
    V = np.zeros(N)
    for k in range(1, N + 1):
        s = 0
        for j in range(int((k + 1) / 2), min(k, N // 2) + 1):
            numer = j**(N // 2) * factorial(2 * j)
            denom = (factorial(N // 2 - j) *
                     factorial(j) *
                     factorial(j - 1) *
                     factorial(k - j) *
                     factorial(2 * j - k))
            s += numer / denom
        V[k - 1] = (-1)**(k + N // 2) * s
    return V

def numerical_inverse_laplace(F_func, t_values, N=12):
    """Numerical inverse Laplace transform using the Stehfest algorithm

    Parameters:
        F_func: function that computes F(s)
        t_values: array of time values at which to compute the inverse transform
        N: Stehfest order (even, default 12)
    """
    V = stehfest_weights(N)
    ln2 = np.log(2)
    result = np.zeros_like(t_values, dtype=float)

    for i, t in enumerate(t_values):
        if t <= 0:
            result[i] = 0
            continue
        s_vals = np.arange(1, N + 1) * ln2 / t
        F_vals = np.array([F_func(sv) for sv in s_vals])
        result[i] = (ln2 / t) * np.sum(V * F_vals)

    return result

# Verification: F(s) = 1/(s+1) -> f(t) = e^(-t)
F_exp = lambda sv: 1.0 / (sv + 1.0)
t_test = np.linspace(0.01, 5, 200)
f_numerical = numerical_inverse_laplace(F_exp, t_test)
f_exact = np.exp(-t_test)

plt.figure(figsize=(10, 6))
plt.plot(t_test, f_exact, 'b-', linewidth=2, label='Analytical: exp(-t)')
plt.plot(t_test, f_numerical, 'r--', linewidth=2, label='Stehfest (N=12)')
plt.xlabel('Time t')
plt.ylabel('f(t)')
plt.title('Numerical inverse Laplace transform verification')
plt.legend()
plt.grid(True)
plt.savefig('numerical_inverse_laplace.png', dpi=150)
plt.show()

# Error analysis
max_error = np.max(np.abs(f_numerical - f_exact))
print(f"Maximum error: {max_error:.2e}")

# More complex example: F(s) = 1/(s^2+1) -> f(t) = sin(t)
F_sin = lambda sv: 1.0 / (sv**2 + 1.0)
f_sin_numerical = numerical_inverse_laplace(F_sin, t_test)
f_sin_exact = np.sin(t_test)

print(f"\nMaximum error for sin(t) inverse transform: {np.max(np.abs(f_sin_numerical - f_sin_exact)):.2e}")
```

---

## Practice Problems

### Basic

**Problem 1.** Find the Laplace transform of the following functions.

(a) $f(t) = 3t^2 - 2e^{-t} + 5\cos(4t)$

(b) $f(t) = t^3 e^{2t}$

(c) $f(t) = e^{-3t}\sin(5t)$

**Problem 2.** Find the inverse Laplace transform of the following $F(s)$.

(a) $F(s) = \dfrac{5}{s^3}$

(b) $F(s) = \dfrac{2s + 1}{s^2 + 4s + 13}$

(c) $F(s) = \dfrac{3}{(s-1)(s+2)(s-3)}$

**Problem 3.** Use the convolution theorem to find $\mathcal{L}^{-1}\left\{\dfrac{1}{s(s+1)}\right\}$.

### Intermediate

**Problem 4.** Solve the following initial value problem using the Laplace transform.

$$y'' - 4y' + 4y = e^{2t}, \quad y(0) = 0, \quad y'(0) = 1$$

**Problem 5.** Use the second shifting theorem to find the Laplace transform of:

$$f(t) = \begin{cases} 0, & 0 \leq t < 2 \\ t - 2, & t \geq 2 \end{cases}$$

**Problem 6.** Solve the following coupled differential equations using the Laplace transform.

$$x' + y = e^t, \quad x + y' = 0, \quad x(0) = 1, \quad y(0) = 0$$

**Problem 7.** For the transfer function $H(s) = \dfrac{s + 3}{s^2 + 4s + 8}$, find the poles and zeros, determine system stability, and find the impulse response $h(t)$.

### Advanced

**Problem 8.** Solve the integral equation using the Laplace transform.

$$y(t) = 1 + \int_0^t y(\tau) \sin(t - \tau) \, d\tau$$

**Problem 9.** For a series RLC circuit ($R = 4\,\Omega$, $L = 1\,$H, $C = 1/5\,$F) driven by $V(t) = 10u(t)$ (unit step voltage), find the current $i(t)$ using the Laplace transform.

**Problem 10.** Use the final value theorem to find the steady-state value of the unit step response for a system with transfer function:

$$H(s) = \frac{10(s + 2)}{s^2 + 5s + 6}$$

---

## Advanced Topics

### Bilateral Laplace Transform

The standard Laplace transform is unilateral, defined only for $t \geq 0$, but the **bilateral Laplace transform** is:

$$F(s) = \int_{-\infty}^{\infty} f(t) e^{-st} \, dt$$

The bilateral transform has a strip-shaped region of convergence and is important in signal processing and probability theory.

### Relationship with Z-Transform

The discrete-time counterpart of the Laplace transform is the **Z-transform**:

$$X(z) = \sum_{n=0}^{\infty} x[n] z^{-n}$$

Setting $z = e^{sT}$ (where $T$ is the sampling period) establishes the relationship between the two transforms. It plays a key role in digital signal processing and discrete control systems.

### References

**Textbooks**:
- **Boas, M. L.** *Mathematical Methods in the Physical Sciences*, 3rd ed., Ch. 8 (Sec. 10-12)
- **Arfken, Weber** *Mathematical Methods for Physicists*, Ch. 15

**Supplementary materials**:
- **Schiff, J. L.** *The Laplace Transform: Theory and Applications* - Balanced coverage of theory and applications
- **Dyke, P. P. G.** *An Introduction to Laplace Transforms and Fourier Series* - Mathematical physics approach

### Key Formula Summary

| Property | Time domain | $s$-domain |
|------|-----------|----------|
| Definition | $f(t)$ | $F(s) = \int_0^\infty f(t)e^{-st}\,dt$ |
| Linearity | $\alpha f + \beta g$ | $\alpha F + \beta G$ |
| First shift | $e^{at}f(t)$ | $F(s-a)$ |
| Second shift | $f(t-a)u(t-a)$ | $e^{-as}F(s)$ |
| Differentiation | $f'(t)$ | $sF(s) - f(0)$ |
| Integration | $\int_0^t f(\tau)\,d\tau$ | $F(s)/s$ |
| Convolution | $(f*g)(t)$ | $F(s) \cdot G(s)$ |
| Initial value | $f(0^+)$ | $\lim_{s\to\infty} sF(s)$ |
| Final value | $\lim_{t\to\infty} f(t)$ | $\lim_{s\to 0} sF(s)$ |

---

**Previous**: [14. Complex Analysis](14_Complex_Analysis.md)
**Next**: [16. Green's Functions](16_Greens_Functions.md)
