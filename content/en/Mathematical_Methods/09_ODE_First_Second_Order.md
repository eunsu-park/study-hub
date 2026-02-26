# 09. Ordinary Differential Equations: First and Second Order

## Learning Objectives

- Master key solution techniques for 1st order ODEs (separable, integrating factor, exact, substitution methods)
- Understand homogeneous and non-homogeneous solution methods for 2nd order constant-coefficient ODEs
- Model and analyze damped harmonic oscillators and RLC circuits using ODEs
- Grasp the meaning of existence and uniqueness theorems and the Wronskian
- Use SymPy and SciPy to verify analytical solutions and obtain numerical solutions

---

## 1. First Order ODEs

The general form of a first order ordinary differential equation is:

$$\frac{dy}{dx} = f(x, y)$$

When an initial condition $y(x_0) = y_0$ is given, it becomes an **Initial Value Problem (IVP)**.

### 1.1 Separable Equations

If $f(x,y)$ can be separated as $g(x) \cdot h(y)$, the equation is **separable**:

$$\frac{dy}{dx} = g(x)\,h(y) \quad\Longrightarrow\quad \frac{dy}{h(y)} = g(x)\,dx$$

Integrating both sides yields the solution.

**Example: Population Growth Model (Logistic Equation)**

$$\frac{dP}{dt} = rP\!\left(1 - \frac{P}{K}\right)$$

where $r$ is the intrinsic growth rate and $K$ is the carrying capacity.

Using partial fraction decomposition to separate:

$$\frac{dP}{P(1 - P/K)} = r\,dt \quad\Longrightarrow\quad \frac{1}{P} + \frac{1/K}{1 - P/K}\,dP = r\,dt$$

Integrating gives the solution:

$$P(t) = \frac{K}{1 + \left(\frac{K}{P_0} - 1\right)e^{-rt}}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, dsolve, Eq, exp

# Solve logistic equation with SymPy
t = symbols('t')
P = Function('P')
r_val, K_val, P0_val = 0.5, 100, 10

ode = Eq(P(t).diff(t), r_val * P(t) * (1 - P(t) / K_val))
sol = dsolve(ode, P(t), ics={P(0): P0_val})
print("Analytical solution:", sol)

# Numerical verification (SciPy)
from scipy.integrate import solve_ivp

def logistic(t, y):
    return r_val * y[0] * (1 - y[0] / K_val)

t_span = (0, 20)
t_eval = np.linspace(0, 20, 200)
result = solve_ivp(logistic, t_span, [P0_val], t_eval=t_eval)

plt.figure(figsize=(8, 5))
plt.plot(result.t, result.y[0], 'b-', label='Numerical solution (solve_ivp)')
plt.axhline(y=K_val, color='r', linestyle='--', label=f'Carrying capacity K={K_val}')
plt.xlabel('Time t')
plt.ylabel('Population P(t)')
plt.title('Logistic Growth Model')
plt.legend()
plt.grid(True)
plt.show()
```

### 1.2 Linear First Order ODEs and Integrating Factor

**Standard form**:

$$\frac{dy}{dx} + P(x)\,y = Q(x)$$

Define the **integrating factor** $\mu(x)$ as:

$$\mu(x) = e^{\int P(x)\,dx}$$

Multiplying both sides by $\mu(x)$ makes the left side a perfect derivative:

$$\frac{d}{dx}\bigl[\mu(x)\,y\bigr] = \mu(x)\,Q(x)$$

Therefore, the general solution is:

$$y = \frac{1}{\mu(x)}\left[\int \mu(x)\,Q(x)\,dx + C\right]$$

**Example: Newton's Law of Cooling**

$$\frac{dT}{dt} = -k(T - T_{\text{env}})$$

Converting to standard form: $\frac{dT}{dt} + kT = kT_{\text{env}}$, so $\mu = e^{kt}$.

$$T(t) = T_{\text{env}} + (T_0 - T_{\text{env}})\,e^{-kt}$$

```python
from sympy import symbols, Function, dsolve, Eq, exp

t, k, T_env, T0 = symbols('t k T_env T_0', positive=True)
T = Function('T')

ode = Eq(T(t).diff(t), -k * (T(t) - T_env))
sol = dsolve(ode, T(t), ics={T(0): T0})
print("Newton's law of cooling solution:", sol)
# T(t) = T_env + (T_0 - T_env)*exp(-k*t)
```

### 1.3 Exact Equations

An equation of the form:

$$M(x,y)\,dx + N(x,y)\,dy = 0$$

is **exact** if and only if:

$$\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}$$

In this case, a potential function $F(x,y)$ exists such that:

$$\frac{\partial F}{\partial x} = M, \quad \frac{\partial F}{\partial y} = N$$

The solution is $F(x,y) = C$ (implicit form).

**Solution procedure:**
1. Verify $\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}$
2. Compute $F = \int M\,dx + g(y)$ (where $g(y)$ is a function of $y$ only)
3. Determine $g'(y)$ from $\frac{\partial F}{\partial y} = N$
4. Write $F(x,y) = C$

**Example:** $(2xy + 3)\,dx + (x^2 + 4y)\,dy = 0$

- $M = 2xy + 3$, $N = x^2 + 4y$
- $\frac{\partial M}{\partial y} = 2x = \frac{\partial N}{\partial x}$ → exact
- $F = \int (2xy + 3)\,dx = x^2 y + 3x + g(y)$
- $\frac{\partial F}{\partial y} = x^2 + g'(y) = x^2 + 4y$ → $g'(y) = 4y$ → $g(y) = 2y^2$
- **Solution:** $x^2 y + 3x + 2y^2 = C$

### 1.4 Substitution Methods

Equations that are neither separable nor linear can often be solved by appropriate substitutions.

**Homogeneous equation:**

If $\frac{dy}{dx} = f\!\left(\frac{y}{x}\right)$, use the substitution $v = y/x$ (i.e., $y = vx$).

$$x\frac{dv}{dx} + v = f(v) \quad\Longrightarrow\quad \frac{dv}{f(v) - v} = \frac{dx}{x}$$

**Bernoulli equation:**

$$\frac{dy}{dx} + P(x)\,y = Q(x)\,y^n \quad (n \neq 0, 1)$$

The substitution $w = y^{1-n}$ transforms it into a linear first order ODE:

$$\frac{dw}{dx} + (1-n)P(x)\,w = (1-n)Q(x)$$

```python
from sympy import symbols, Function, dsolve, Eq

x = symbols('x')
y = Function('y')

# Bernoulli equation: y' + y/x = x*y^2
bernoulli_ode = Eq(y(x).diff(x) + y(x)/x, x * y(x)**2)
sol = dsolve(bernoulli_ode, y(x))
print("Bernoulli equation solution:", sol)
```

---

## 2. Second Order Constant Coefficient ODEs

General form of a second order constant coefficient linear ODE:

$$a\,y'' + b\,y' + c\,y = f(x)$$

where $a, b, c$ are constants. If $f(x) = 0$, it's **homogeneous**; if $f(x) \neq 0$, it's **non-homogeneous**.

### 2.1 Homogeneous Equations and Characteristic Equation

$$a\,y'' + b\,y' + c\,y = 0$$

Substituting $y = e^{rx}$ yields the **characteristic equation**:

$$ar^2 + br + c = 0$$

Quadratic formula: $r = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$

Three cases arise depending on the discriminant $D = b^2 - 4ac$.

### 2.2 Three Cases: Distinct Real Roots, Repeated Roots, Complex Roots

**Case 1: $D > 0$ — Two distinct real roots $r_1, r_2$**

$$y = C_1 e^{r_1 x} + C_2 e^{r_2 x}$$

**Case 2: $D = 0$ — Repeated root $r_1 = r_2 = r$**

$$y = (C_1 + C_2 x)\,e^{rx}$$

($xe^{rx}$ is the second independent solution — prevents degeneracy)

**Case 3: $D < 0$ — Complex roots $r = \alpha \pm i\beta$**

$$y = e^{\alpha x}\bigl(C_1 \cos\beta x + C_2 \sin\beta x\bigr)$$

This is the real form using Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$.

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, dsolve, Eq, cos, sin, exp

x = symbols('x')
y = Function('y')

# Case 1: y'' - 5y' + 6y = 0 → r = 2, 3
sol1 = dsolve(Eq(y(x).diff(x, 2) - 5*y(x).diff(x) + 6*y(x), 0), y(x))
print("Distinct real roots:", sol1)

# Case 2: y'' - 4y' + 4y = 0 → r = 2 (repeated root)
sol2 = dsolve(Eq(y(x).diff(x, 2) - 4*y(x).diff(x) + 4*y(x), 0), y(x))
print("Repeated root:", sol2)

# Case 3: y'' + 2y' + 5y = 0 → r = -1 ± 2i
sol3 = dsolve(Eq(y(x).diff(x, 2) + 2*y(x).diff(x) + 5*y(x), 0), y(x))
print("Complex roots:", sol3)
```

### 2.3 Non-homogeneous Equations: Method of Undetermined Coefficients

$$a\,y'' + b\,y' + c\,y = f(x)$$

**General solution** = homogeneous solution $y_h$ + particular solution $y_p$

The method of undetermined coefficients applies when $f(x)$ is a polynomial, exponential, trigonometric function, or their combinations.

| Form of $f(x)$ | Trial form for $y_p$ |
|---|---|
| $P_n(x)$ (n-th degree polynomial) | $A_n x^n + A_{n-1}x^{n-1} + \cdots + A_0$ |
| $e^{\alpha x}$ | $A e^{\alpha x}$ |
| $\cos\beta x$ or $\sin\beta x$ | $A\cos\beta x + B\sin\beta x$ |
| $e^{\alpha x}\cos\beta x$ | $e^{\alpha x}(A\cos\beta x + B\sin\beta x)$ |

**Note:** If the trial $y_p$ is contained in $y_h$, multiply by $x$ (modification rule for duplication).

**Example:** $y'' + 4y = 3\sin 2x$

Homogeneous solution: $y_h = C_1\cos 2x + C_2\sin 2x$ (characteristic roots $r = \pm 2i$)

Since $\sin 2x$ is in $y_h$, we try $y_p = x(A\cos 2x + B\sin 2x)$.

After substitution and comparing coefficients:

$$y_p = -\frac{3}{4}x\cos 2x$$

```python
from sympy import symbols, Function, dsolve, Eq, sin, cos

x = symbols('x')
y = Function('y')

# y'' + 4y = 3*sin(2x)
ode = Eq(y(x).diff(x, 2) + 4*y(x), 3*sin(2*x))
sol = dsolve(ode, y(x))
print("Undetermined coefficients solution:", sol)
```

### 2.4 Non-homogeneous Equations: Variation of Parameters

This is a **general method** for $f(x)$ where undetermined coefficients don't apply.

When homogeneous solutions $y_1(x), y_2(x)$ are known, the particular solution is:

$$y_p = u_1(x)\,y_1(x) + u_2(x)\,y_2(x)$$

where $u_1', u_2'$ are determined by the system:

$$u_1' y_1 + u_2' y_2 = 0$$
$$u_1' y_1' + u_2' y_2' = \frac{f(x)}{a}$$

Using the **Wronskian** $W = y_1 y_2' - y_2 y_1'$:

$$u_1' = -\frac{y_2 f(x)}{aW}, \quad u_2' = \frac{y_1 f(x)}{aW}$$

**Example:** $y'' + y = \sec x$

- Homogeneous solutions: $y_1 = \cos x$, $y_2 = \sin x$
- $W = \cos x \cdot \cos x - \sin x \cdot (-\sin x) = 1$
- $u_1' = -\sin x \cdot \sec x = -\tan x$ → $u_1 = \ln|\cos x|$
- $u_2' = \cos x \cdot \sec x = 1$ → $u_2 = x$
- $y_p = \cos x \ln|\cos x| + x\sin x$

```python
from sympy import symbols, Function, dsolve, Eq, sec

x = symbols('x')
y = Function('y')

# y'' + y = sec(x)
ode = Eq(y(x).diff(x, 2) + y(x), sec(x))
sol = dsolve(ode, y(x), hint='variation_of_parameters')
print("Variation of parameters solution:", sol)
```

---

## 3. Damped Harmonic Oscillator

This is the most important application of second order ODEs in physics.

For a system with mass $m$, damping coefficient $\gamma$, and spring constant $k$:

$$m\ddot{x} + \gamma\dot{x} + kx = F(t)$$

Define $\omega_0 = \sqrt{k/m}$ (natural frequency) and $\beta = \gamma/(2m)$ (damping constant):

$$\ddot{x} + 2\beta\dot{x} + \omega_0^2 x = \frac{F(t)}{m}$$

### 3.1 Free Oscillation

When there's no external force ($F(t) = 0$):

$$\ddot{x} + 2\beta\dot{x} + \omega_0^2 x = 0$$

Characteristic equation: $r^2 + 2\beta r + \omega_0^2 = 0$

$$r = -\beta \pm \sqrt{\beta^2 - \omega_0^2}$$

### 3.2 Overdamped, Critically Damped, and Underdamped

Three oscillation modes arise depending on the sign of the discriminant $\beta^2 - \omega_0^2$:

**1) Underdamped: $\beta < \omega_0$**

$$x(t) = A e^{-\beta t}\cos(\omega_d t + \phi)$$

where the damped frequency $\omega_d = \sqrt{\omega_0^2 - \beta^2}$

Oscillates while decaying exponentially. Most physical systems fall into this case.

**2) Critically Damped: $\beta = \omega_0$**

$$x(t) = (C_1 + C_2 t)\,e^{-\beta t}$$

Returns to equilibrium fastest without oscillation. Used in door dampers, etc.

**3) Overdamped: $\beta > \omega_0$**

$$x(t) = C_1 e^{r_1 t} + C_2 e^{r_2 t}$$

Both real roots are negative, so it slowly approaches equilibrium.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

omega0 = 5.0   # natural frequency
x0, v0 = 1.0, 0.0  # initial conditions: x(0)=1, v(0)=0

fig, ax = plt.subplots(figsize=(10, 6))
t_eval = np.linspace(0, 5, 500)

for label, beta in [('Underdamped (β=1)', 1.0),
                     ('Critically damped (β=5)', 5.0),
                     ('Overdamped (β=8)', 8.0)]:
    def damped_osc(t, y, b=beta):
        return [y[1], -2*b*y[1] - omega0**2 * y[0]]

    sol = solve_ivp(damped_osc, (0, 5), [x0, v0], t_eval=t_eval)
    ax.plot(sol.t, sol.y[0], label=label)

ax.set_xlabel('Time t (s)')
ax.set_ylabel('Displacement x(t)')
ax.set_title('Damped Harmonic Oscillator: Three Damping Regimes')
ax.legend()
ax.grid(True)
ax.axhline(y=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()
```

### 3.3 Forced Oscillation and Resonance

When an external force $F(t) = F_0 \cos\omega t$ is applied:

$$\ddot{x} + 2\beta\dot{x} + \omega_0^2 x = \frac{F_0}{m}\cos\omega t$$

Steady-state particular solution:

$$x_p(t) = A(\omega)\cos(\omega t - \delta)$$

where the amplitude and phase are:

$$A(\omega) = \frac{F_0/m}{\sqrt{(\omega_0^2 - \omega^2)^2 + 4\beta^2\omega^2}}$$

$$\tan\delta = \frac{2\beta\omega}{\omega_0^2 - \omega^2}$$

**Resonance condition:** Driving frequency that maximizes amplitude $A(\omega)$:

$$\omega_{\text{res}} = \sqrt{\omega_0^2 - 2\beta^2}$$

As $\beta \to 0$, $\omega_{\text{res}} \to \omega_0$ (undamped resonance). Without damping, amplitude diverges to infinity.

**Q-factor (Quality Factor):**

$$Q = \frac{\omega_0}{2\beta}$$

Higher $Q$ means sharper resonance peak and less energy loss.

```python
import numpy as np
import matplotlib.pyplot as plt

omega0 = 10.0
F0_over_m = 1.0
omega = np.linspace(0.1, 20, 500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for beta in [0.2, 0.5, 1.0, 2.0]:
    A = F0_over_m / np.sqrt((omega0**2 - omega**2)**2 + (2*beta*omega)**2)
    delta = np.arctan2(2*beta*omega, omega0**2 - omega**2)
    Q = omega0 / (2*beta)
    ax1.plot(omega, A, label=f'β={beta}, Q={Q:.1f}')
    ax2.plot(omega, np.degrees(delta), label=f'β={beta}')

ax1.set_xlabel('Driving frequency ω')
ax1.set_ylabel('Amplitude A(ω)')
ax1.set_title('Forced Oscillation: Resonance Curve')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Driving frequency ω')
ax2.set_ylabel('Phase difference δ (°)')
ax2.set_title('Forced Oscillation: Phase Response')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()
```

---

## 4. Electrical Circuit Analysis

### 4.1 RLC Circuit Equation

Applying Kirchhoff's voltage law (KVL) to a series RLC circuit:

$$L\frac{dI}{dt} + RI + \frac{Q}{C} = V(t)$$

Using $I = dQ/dt$, we get a second order ODE for charge $Q$:

$$L\frac{d^2Q}{dt^2} + R\frac{dQ}{dt} + \frac{1}{C}Q = V(t)$$

**Correspondence with damped oscillator:**

| Mechanical system | Electrical circuit |
|--------|----------|
| Mass $m$ | Inductance $L$ |
| Damping coefficient $\gamma$ | Resistance $R$ |
| Spring constant $k$ | $1/C$ |
| Displacement $x$ | Charge $Q$ |
| Velocity $\dot{x}$ | Current $I$ |
| External force $F(t)$ | Source voltage $V(t)$ |

Natural frequency: $\omega_0 = 1/\sqrt{LC}$, damping constant: $\beta = R/(2L)$

### 4.2 Transient and Steady-State Response

The total response divides into two parts:

- **Transient response:** Homogeneous solution $Q_h(t)$ — decays and disappears over time
- **Steady-state response:** Particular solution $Q_p(t)$ — continues oscillating at driving frequency

For $V(t) = V_0 \cos\omega t$, steady-state current:

$$I_{\text{ss}}(t) = \frac{V_0}{Z}\cos(\omega t - \phi)$$

where $Z$ is the impedance and $\phi$ is the phase angle.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# RLC circuit parameters
L = 0.5      # inductance (H)
R = 10.0     # resistance (Ω)
C = 100e-6   # capacitance (F)
V0 = 12.0    # source amplitude (V)
omega_drive = 100.0  # driving angular frequency (rad/s)

omega0 = 1 / np.sqrt(L * C)
beta = R / (2 * L)
print(f"Natural frequency: ω₀ = {omega0:.1f} rad/s")
print(f"Damping constant: β = {beta:.1f} s⁻¹")

# Q'' + (R/L)Q' + (1/LC)Q = V(t)/L
def rlc_circuit(t, y):
    Q, I = y
    dQ = I
    dI = (V0 * np.cos(omega_drive * t) - R * I - Q / C) / L
    return [dQ, dI]

t_span = (0, 0.5)
t_eval = np.linspace(0, 0.5, 2000)
sol = solve_ivp(rlc_circuit, t_span, [0.0, 0.0], t_eval=t_eval,
                method='RK45', max_step=1e-4)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(sol.t * 1000, sol.y[0] * 1e6, 'b-')
ax1.set_ylabel('Charge Q (μC)')
ax1.set_title('Series RLC Circuit Response')
ax1.grid(True)

ax2.plot(sol.t * 1000, sol.y[1] * 1000, 'r-')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Current I (mA)')
ax2.grid(True)
plt.tight_layout()
plt.show()
```

### 4.3 Impedance and Complex Number Solution

In AC circuits, using complex impedance transforms ODEs into algebraic equations.

$$V(t) = V_0 e^{i\omega t} \quad\text{gives}$$

Complex impedance of each component:
- Resistor: $Z_R = R$
- Inductor: $Z_L = i\omega L$
- Capacitor: $Z_C = \frac{1}{i\omega C} = -\frac{i}{\omega C}$

Series combined impedance:

$$Z = R + i\!\left(\omega L - \frac{1}{\omega C}\right)$$

Magnitude and phase of impedance:

$$|Z| = \sqrt{R^2 + \left(\omega L - \frac{1}{\omega C}\right)^2}$$

$$\phi = \arctan\frac{\omega L - 1/(\omega C)}{R}$$

Steady-state current amplitude: $I_0 = V_0 / |Z|$

**Resonance condition:** When $\omega L = 1/(\omega C)$, i.e., $\omega = \omega_0 = 1/\sqrt{LC}$, $|Z| = R$ is minimum, current maximum.

```python
import numpy as np
import matplotlib.pyplot as plt

L, R, C = 0.5, 10.0, 100e-6
V0 = 12.0
omega = np.linspace(10, 500, 1000)
omega0 = 1 / np.sqrt(L * C)

# Complex impedance
Z = R + 1j * (omega * L - 1 / (omega * C))
I_amp = V0 / np.abs(Z)
phase = np.angle(Z, deg=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(omega, I_amp, 'b-')
ax1.axvline(x=omega0, color='r', linestyle='--', label=f'Resonance ω₀={omega0:.1f}')
ax1.set_ylabel('Current amplitude I₀ (A)')
ax1.set_title('RLC Circuit Frequency Response')
ax1.legend()
ax1.grid(True)

ax2.plot(omega, phase, 'g-')
ax2.axvline(x=omega0, color='r', linestyle='--')
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.set_xlabel('Angular frequency ω (rad/s)')
ax2.set_ylabel('Phase angle φ (°)')
ax2.grid(True)
plt.tight_layout()
plt.show()
```

---

## 5. Existence and Uniqueness of Solutions

### 5.1 Picard-Lindelöf Theorem

**Theorem (Picard-Lindelöf):**

For the initial value problem $\frac{dy}{dx} = f(x, y)$, $y(x_0) = y_0$, if $f(x,y)$ and $\frac{\partial f}{\partial y}$ are continuous in a rectangular region containing $(x_0, y_0)$, then a solution **exists and is unique** in a neighborhood of $x_0$.

The key is the **Lipschitz condition** for $f$:

$$|f(x, y_1) - f(x, y_2)| \leq L|y_1 - y_2|$$

If $\frac{\partial f}{\partial y}$ is bounded, the Lipschitz condition is automatically satisfied.

**Counterexample:** $\frac{dy}{dx} = y^{1/3}$, $y(0) = 0$

For $f(x,y) = y^{1/3}$, at $y = 0$, $\frac{\partial f}{\partial y} = \frac{1}{3}y^{-2/3} \to \infty$, breaking the Lipschitz condition. Indeed, both $y = 0$ and $y = \left(\frac{2x}{3}\right)^{3/2}$ are solutions (uniqueness fails).

**Picard Iteration:**

The iterative method used in proving the theorem is also numerically useful:

$$y_{n+1}(x) = y_0 + \int_{x_0}^{x} f(t, y_n(t))\,dt$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Rational, Piecewise, integrate

# Picard iteration: y' = x + y, y(0) = 1
# Exact solution: y = 2e^x - x - 1

x_sym = symbols('x')
y_exact = 2 * np.e**np.linspace(0, 2, 200) - np.linspace(0, 2, 200) - 1

# Numerical Picard iteration
x_vals = np.linspace(0, 2, 200)

def picard_iteration(f, x0, y0, x_vals, n_iter=6):
    """Approximate ODE solution using Picard iteration."""
    from scipy.integrate import cumulative_trapezoid
    y_n = np.full_like(x_vals, y0, dtype=float)
    results = [y_n.copy()]

    for _ in range(n_iter):
        integrand = f(x_vals, y_n)
        integral = cumulative_trapezoid(integrand, x_vals, initial=0)
        y_n = y0 + integral
        results.append(y_n.copy())
    return results

f = lambda x, y: x + y
iterations = picard_iteration(f, 0, 1, x_vals, n_iter=6)

plt.figure(figsize=(10, 6))
for i, y_approx in enumerate(iterations[1:], 1):
    if i in [1, 2, 3, 6]:
        plt.plot(x_vals, y_approx, '--', label=f'Picard iteration {i}')
plt.plot(x_vals, y_exact, 'k-', linewidth=2, label='Exact solution')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Convergence of Picard Iteration')
plt.legend()
plt.grid(True)
plt.show()
```

### 5.2 Wronskian and Linear Independence

For two solutions $y_1, y_2$ of the second order ODE $y'' + P(x)y' + Q(x)y = 0$, the **Wronskian** is:

$$W(y_1, y_2) = \begin{vmatrix} y_1 & y_2 \\ y_1' & y_2' \end{vmatrix} = y_1 y_2' - y_2 y_1'$$

**Key theorems:**

1. $y_1, y_2$ are linearly independent ⟺ $W \neq 0$ (for solutions of the ODE)
2. **Abel's Theorem:** $W(x) = W(x_0)\,\exp\!\left(-\int_{x_0}^{x} P(s)\,ds\right)$
3. $W$ is either identically 0 or never 0 (for solutions of the ODE)

**Example:** $y_1 = e^x$, $y_2 = e^{2x}$

$$W = e^x \cdot 2e^{2x} - e^{2x} \cdot e^x = 2e^{3x} - e^{3x} = e^{3x} \neq 0$$

Therefore linearly independent, forming a basis for the general solution $y = C_1 e^x + C_2 e^{2x}$.

**Generalization to n-th order:**

For $n$ functions $y_1, \ldots, y_n$:

$$W = \begin{vmatrix}
y_1 & y_2 & \cdots & y_n \\
y_1' & y_2' & \cdots & y_n' \\
\vdots & \vdots & \ddots & \vdots \\
y_1^{(n-1)} & y_2^{(n-1)} & \cdots & y_n^{(n-1)}
\end{vmatrix}$$

```python
from sympy import symbols, exp, Matrix, simplify

x = symbols('x')

# Wronskian calculation
y1 = exp(x)
y2 = exp(2*x)

W_matrix = Matrix([
    [y1, y2],
    [y1.diff(x), y2.diff(x)]
])
W = simplify(W_matrix.det())
print(f"W(e^x, e^(2x)) = {W}")  # e^(3x)

# Wronskian of 3 functions
y3 = exp(3*x)
W3 = Matrix([
    [y1, y2, y3],
    [y1.diff(x), y2.diff(x), y3.diff(x)],
    [y1.diff(x, 2), y2.diff(x, 2), y3.diff(x, 2)]
])
print(f"W(e^x, e^(2x), e^(3x)) = {simplify(W3.det())}")  # 2*e^(6x)
```

---

## Practice Problems

### Basic Problems

**1.** Solve the following separable ODE: $\frac{dy}{dx} = \frac{x^2}{1 + y^2}$, $y(0) = 0$

**2.** Solve using integrating factor: $\frac{dy}{dx} + 2xy = x$

**3.** Check if the following is an exact equation, and solve if exact:
$(3x^2 y + y^3)\,dx + (x^3 + 3xy^2)\,dy = 0$

**4.** Solve using characteristic equation: $y'' - 6y' + 9y = 0$, $y(0) = 2$, $y'(0) = 5$

**5.** Solve using undetermined coefficients: $y'' + 3y' + 2y = 4e^{-x}$

### Application Problems

**6.** A damped oscillator has mass $m = 0.5\,\text{kg}$, spring constant $k = 8\,\text{N/m}$, damping coefficient $\gamma = 2\,\text{kg/s}$, starting from $x(0) = 0.1\,\text{m}$, $\dot{x}(0) = 0$.
   - (a) Determine damping type (overdamped/critical/underdamped)
   - (b) Find analytical solution $x(t)$
   - (c) Obtain numerical solution with Python and compare

**7.** In a series RLC circuit with $L = 0.1\,\text{H}$, $R = 20\,\Omega$, $C = 50\,\mu\text{F}$ and $V(t) = 10\cos(100t)\,\text{V}$:
   - (a) Find natural frequency $\omega_0$ and damping constant $\beta$
   - (b) Find steady-state current amplitude and phase
   - (c) What $\omega$ gives resonance?

**8.** Solve using variation of parameters: $y'' + 4y = \frac{1}{\sin 2x}$

### Advanced Problems

**9.** Use the Wronskian to show $\{1, x, x^2\}$ are linearly independent.

**10.** Apply Picard iteration to $y' = y$, $y(0) = 1$ for the first 5 iterations and compare with Taylor series of $e^x$.

```python
# Problem 6 solution code (skeleton)
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

m, gamma_val, k = 0.5, 2.0, 8.0
omega0 = np.sqrt(k / m)
beta_val = gamma_val / (2 * m)
print(f"ω₀ = {omega0:.2f}, β = {beta_val:.2f}")
print(f"β² - ω₀² = {beta_val**2 - omega0**2:.2f}")
# β=2, ω₀=4 → β < ω₀ → underdamped

def damped_system(t, y):
    return [y[1], -(gamma_val/m)*y[1] - (k/m)*y[0]]

t_span = (0, 5)
t_eval = np.linspace(0, 5, 500)
sol = solve_ivp(damped_system, t_span, [0.1, 0.0], t_eval=t_eval)

# Analytical solution: x(t) = A*exp(-β*t)*cos(ωd*t + φ)
omega_d = np.sqrt(omega0**2 - beta_val**2)
A = 0.1 / np.cos(np.arctan(beta_val / omega_d))  # from initial conditions
phi = np.arctan(beta_val / omega_d)
x_analytic = A * np.exp(-beta_val * t_eval) * np.cos(omega_d * t_eval + phi)

plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], 'b-', label='Numerical solution')
plt.plot(t_eval, x_analytic, 'r--', label='Analytical solution')
plt.xlabel('Time (s)')
plt.ylabel('Displacement x(t) (m)')
plt.title('Underdamped Harmonic Oscillator')
plt.legend()
plt.grid(True)
plt.show()
```

---

## References

- **Mary L. Boas**, *Mathematical Methods in the Physical Sciences*, 3rd Edition, Chapter 8
- **George B. Arfken**, *Mathematical Methods for Physicists*, Chapter 9
- **Erwin Kreyszig**, *Advanced Engineering Mathematics*, Chapters 1-3
- **SymPy ODE documentation**: https://docs.sympy.org/latest/modules/solvers/ode.html
- **SciPy solve_ivp documentation**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- **3Blue1Brown**: "Differential Equations" series (visual intuition)

---

## Next Lesson

[08. Power Series and Frobenius Method](./08_Power_Series_Frobenius.md) — First half of Boas Chapter 12. Learn to solve ODEs using power series near singular points, and discover the origins of Bessel functions and Legendre polynomials.
