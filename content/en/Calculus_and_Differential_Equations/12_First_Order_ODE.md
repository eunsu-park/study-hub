# 12. First-Order Ordinary Differential Equations

## Learning Objectives

- Classify ODEs by order, linearity, and type, and explain what a solution means
- Visualize solution behavior through direction fields and approximate solutions with Euler's method
- Solve separable, linear, and exact first-order ODEs using analytical techniques
- State the Picard-Lindelof theorem and identify when uniqueness fails
- Model real-world phenomena (population growth, mixing, cooling) with first-order ODEs and verify solutions numerically

---

## 1. What Is an ODE?

An **ordinary differential equation (ODE)** is an equation involving a function $y(x)$ and its derivatives:

$$F\!\left(x,\, y,\, y',\, y'',\, \ldots,\, y^{(n)}\right) = 0$$

- **Order:** The highest derivative that appears. $y' + y = 0$ is first-order; $y'' + y = 0$ is second-order.
- **Linear:** The unknown $y$ and its derivatives appear to the first power with no products between them. $y' + xy = x^2$ is linear; $y' + y^2 = 0$ is nonlinear.
- **Solution:** A function $y(x)$ that satisfies the equation on some interval. An **Initial Value Problem (IVP)** adds a condition $y(x_0) = y_0$.

**Analogy:** An ODE is a "rule about rates of change." If $y$ is your bank balance, $y' = 0.05y$ says "the rate of growth is 5% of the current balance" -- this is the ODE for continuous compound interest.

**Why study ODEs?** Almost every dynamical system in science is described by differential equations: population growth, radioactive decay, electrical circuits, planetary orbits, chemical reactions, neural networks, and more.

---

## 2. Direction Fields and Euler's Method

### 2.1 Direction Fields

Before solving an ODE analytically, we can **visualize** the solution. For $y' = f(x, y)$, at every point $(x, y)$ the equation tells us the slope. Drawing short line segments with these slopes creates a **direction field** (or slope field).

**How to read a direction field:** Drop a ball anywhere on the field; it "rolls" along the slopes, tracing out a solution curve. Different starting points give different solutions -- the field shows them all simultaneously.

### 2.2 Euler's Method

The simplest numerical method: step forward using the tangent line.

Given $y' = f(x, y)$, $y(x_0) = y_0$, and step size $h$:

$$y_{n+1} = y_n + h \cdot f(x_n, y_n), \quad x_{n+1} = x_n + h$$

- $h$: step size (smaller = more accurate but more computation)
- $f(x_n, y_n)$: slope at the current point
- $y_{n+1}$: predicted value at the next step

**Intuition:** Walk along the tangent line for a small step, then re-evaluate the slope. It is like navigating with a compass that only shows the direction at your current location -- you keep checking and adjusting.

**Error:** Euler's method has **first-order accuracy**: the global error is $O(h)$. Halving the step size halves the error. More sophisticated methods like Runge-Kutta achieve $O(h^4)$.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Direction field and Euler's method ---

def f(x, y):
    """RHS of ODE: y' = x - y (a simple linear first-order ODE)."""
    return x - y

# Direction field
x_grid = np.linspace(-2, 4, 25)
y_grid = np.linspace(-2, 4, 25)
X, Y = np.meshgrid(x_grid, y_grid)
slopes = f(X, Y)

# Normalize arrows for uniform length (direction only)
magnitude = np.sqrt(1 + slopes**2)
U = 1 / magnitude
V = slopes / magnitude

fig, ax = plt.subplots(figsize=(10, 8))
ax.quiver(X, Y, U, V, color='lightblue', alpha=0.6, scale=30)

# Euler's method with different step sizes
def euler_method(f, x0, y0, h, x_end):
    """Solve y' = f(x, y) using Euler's method."""
    xs = [x0]
    ys = [y0]
    x, y = x0, y0
    while x < x_end - 1e-12:
        y = y + h * f(x, y)
        x = x + h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Exact solution for y' = x - y, y(0) = 0: y = x - 1 + e^(-x)
x_exact = np.linspace(0, 4, 200)
y_exact = x_exact - 1 + np.exp(-x_exact)

ax.plot(x_exact, y_exact, 'k-', linewidth=2.5, label='Exact solution')

for h, color in [(0.5, 'red'), (0.2, 'orange'), (0.05, 'green')]:
    xe, ye = euler_method(f, 0, 0, h, 4)
    ax.plot(xe, ye, 'o-', color=color, markersize=3, label=f'Euler h={h}')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Direction Field and Euler's Method for $y' = x - y$")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 4)
ax.set_ylim(-2, 4)
plt.tight_layout()
plt.show()
```

---

## 3. Separable Equations

A first-order ODE is **separable** if it can be written as:

$$\frac{dy}{dx} = g(x)\,h(y)$$

**Solution method:** Separate variables and integrate both sides:

$$\int \frac{dy}{h(y)} = \int g(x)\, dx + C$$

**Think of it as:** Moving all the $y$'s to one side and all the $x$'s to the other.

### 3.1 Example: Exponential Growth and Decay

$$\frac{dy}{dx} = ky$$

Separate: $\frac{dy}{y} = k\,dx$. Integrate: $\ln|y| = kx + C$, so $y = y_0 e^{kx}$.

- $k > 0$: exponential growth (population, compound interest)
- $k < 0$: exponential decay (radioactive decay, Newton's cooling)

### 3.2 Example: Logistic Growth

$$\frac{dP}{dt} = rP\!\left(1 - \frac{P}{K}\right)$$

- $P$: population
- $r$: intrinsic growth rate
- $K$: carrying capacity (maximum sustainable population)

Separating and using partial fractions:

$$P(t) = \frac{K}{1 + \left(\frac{K}{P_0} - 1\right)e^{-rt}}$$

The population starts growing exponentially but levels off at $K$ -- an S-shaped (sigmoidal) curve.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, Function, dsolve, Eq, exp

# --- Logistic growth: analytical and numerical ---
t_sym = symbols('t')
P = Function('P')
r, K, P0 = 0.5, 1000, 10

# Analytical solution via SymPy
ode = Eq(P(t_sym).diff(t_sym), r * P(t_sym) * (1 - P(t_sym) / K))
sol = dsolve(ode, P(t_sym), ics={P(0): P0})
print(f"Analytical solution: {sol}")

# Numerical solution via SciPy
t_span = (0, 25)
t_eval = np.linspace(0, 25, 300)

def logistic(t, y):
    """Logistic growth ODE: dP/dt = r*P*(1 - P/K)."""
    return r * y[0] * (1 - y[0] / K)

result = solve_ivp(logistic, t_span, [P0], t_eval=t_eval)

# Analytical formula
P_exact = K / (1 + (K / P0 - 1) * np.exp(-r * t_eval))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_eval, P_exact, 'b-', linewidth=2, label='Analytical')
ax.plot(result.t, result.y[0], 'r--', linewidth=2, label='Numerical (solve_ivp)')
ax.axhline(y=K, color='gray', linestyle=':', label=f'Carrying capacity K={K}')
ax.set_xlabel('Time t')
ax.set_ylabel('Population P(t)')
ax.set_title('Logistic Growth Model')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
```

---

## 4. Linear First-Order ODEs

### 4.1 Standard Form and Integrating Factor

A **linear first-order ODE** has the standard form:

$$\frac{dy}{dx} + P(x)\,y = Q(x)$$

The **integrating factor** method is systematic and always works:

1. Compute the integrating factor: $\mu(x) = e^{\int P(x)\,dx}$
2. Multiply both sides by $\mu$: $\frac{d}{dx}[\mu\,y] = \mu\,Q$
3. Integrate: $\mu\,y = \int \mu\,Q\, dx + C$
4. Solve for $y$: $y = \frac{1}{\mu}\left[\int \mu\,Q\, dx + C\right]$

**Why does this work?** Multiplying by $\mu$ makes the left side a **perfect derivative** (product rule in reverse). The magic of $\mu = e^{\int P\,dx}$ is that $\mu' = P\mu$, which is exactly what's needed.

### 4.2 Example: Mixing Problem

A tank initially contains 100 L of pure water. Brine with 2 g/L of salt flows in at 3 L/min, and the well-mixed solution flows out at 3 L/min.

Let $y(t)$ = grams of salt at time $t$. The volume stays at 100 L.

$$\frac{dy}{dt} = \underbrace{2 \cdot 3}_{\text{salt in}} - \underbrace{\frac{y}{100} \cdot 3}_{\text{salt out}} = 6 - \frac{3y}{100}$$

Standard form: $y' + 0.03y = 6$

Integrating factor: $\mu = e^{0.03t}$

$$y(t) = 200(1 - e^{-0.03t})$$

As $t \to \infty$, $y \to 200$ g (equilibrium: concentration approaches 2 g/L).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Mixing problem ---
def mixing(t, y):
    """dy/dt = rate_in - rate_out = 6 - 0.03*y."""
    return 6 - 0.03 * y[0]

t_span = (0, 200)
t_eval = np.linspace(0, 200, 500)
result = solve_ivp(mixing, t_span, [0], t_eval=t_eval)

# Analytical solution
y_exact = 200 * (1 - np.exp(-0.03 * t_eval))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Salt amount over time
ax1.plot(t_eval, y_exact, 'b-', linewidth=2, label='Salt amount y(t)')
ax1.axhline(y=200, color='red', linestyle='--', label='Equilibrium (200 g)')
ax1.set_xlabel('Time (min)')
ax1.set_ylabel('Salt (g)')
ax1.set_title('Mixing Problem: Salt Amount')
ax1.legend()
ax1.grid(True)

# Concentration over time
conc = y_exact / 100  # g/L
ax2.plot(t_eval, conc, 'g-', linewidth=2, label='Concentration (g/L)')
ax2.axhline(y=2.0, color='red', linestyle='--', label='Inflow concentration (2 g/L)')
ax2.set_xlabel('Time (min)')
ax2.set_ylabel('Concentration (g/L)')
ax2.set_title('Mixing Problem: Concentration')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

---

## 5. Exact Equations

### 5.1 Definition

An equation $M(x,y)\,dx + N(x,y)\,dy = 0$ is **exact** if there exists a function $\Psi(x,y)$ such that:

$$\frac{\partial\Psi}{\partial x} = M, \quad \frac{\partial\Psi}{\partial y} = N$$

The solution is the implicit equation $\Psi(x, y) = C$.

### 5.2 Test for Exactness

By Clairaut's theorem, the equation is exact if and only if:

$$\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}$$

### 5.3 Solution Method

If exact:
1. Integrate $M$ with respect to $x$: $\Psi = \int M\,dx + g(y)$
2. Differentiate with respect to $y$ and set equal to $N$: $\frac{\partial}{\partial y}\int M\,dx + g'(y) = N$
3. Solve for $g'(y)$ and integrate to find $g(y)$

**Example:** $(2xy + 3)\,dx + (x^2 + 4y)\,dy = 0$

Check: $M_y = 2x = N_x$ -- exact.

$\Psi = \int (2xy + 3)\,dx = x^2 y + 3x + g(y)$

$\Psi_y = x^2 + g'(y) = x^2 + 4y \implies g'(y) = 4y \implies g(y) = 2y^2$

Solution: $x^2 y + 3x + 2y^2 = C$

```python
from sympy import symbols, Function, Eq, dsolve, classify_ode

x = symbols('x')
y = Function('y')

# --- Exact equation solved with SymPy ---
# (2xy + 3) + (x^2 + 4y) y' = 0
# Rewrite as: y' = -(2xy + 3) / (x^2 + 4y)

ode = Eq(y(x).diff(x), -(2*x*y(x) + 3) / (x**2 + 4*y(x)))

# SymPy can identify and solve exact equations
classifications = classify_ode(ode, y(x))
print(f"ODE classification: {classifications}")

sol = dsolve(ode, y(x))
print(f"Solution: {sol}")
```

---

## 6. Existence and Uniqueness

### 6.1 The Picard-Lindelof Theorem

**Theorem:** For the IVP $y' = f(x, y)$, $y(x_0) = y_0$:

If $f$ and $\partial f / \partial y$ are continuous in a rectangle $R$ containing $(x_0, y_0)$, then there exists a **unique** solution in some neighborhood of $x_0$.

The key condition is the **Lipschitz condition**:

$$|f(x, y_1) - f(x, y_2)| \le L |y_1 - y_2|$$

for some constant $L$. If $\partial f/\partial y$ is bounded on $R$, the Lipschitz condition holds automatically (by the Mean Value Theorem).

### 6.2 When Uniqueness Fails

**Classic counterexample:** $y' = y^{1/3}$, $y(0) = 0$

Here $f(x, y) = y^{1/3}$, and $\partial f / \partial y = \frac{1}{3}y^{-2/3} \to \infty$ as $y \to 0$.

Both $y(x) = 0$ and $y(x) = \left(\frac{2x}{3}\right)^{3/2}$ are valid solutions through $(0, 0)$ -- uniqueness fails precisely because the Lipschitz condition breaks down at the initial point.

### 6.3 Why This Matters

Existence and uniqueness are not just theoretical curiosities:
- **Numerical methods** rely on uniqueness to guarantee convergence
- **Physical models** should have unique solutions (a ball can't be in two places)
- When uniqueness fails, the model may need refinement (e.g., the physics at the singular point isn't captured correctly)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Uniqueness failure: y' = y^(1/3), y(0) = 0 ---
# Two valid solutions: y=0 and y = (2x/3)^(3/2)

x_vals = np.linspace(-0.5, 3, 200)

# Solution 1: trivial
y_trivial = np.zeros_like(x_vals)

# Solution 2: nontrivial (only valid for x >= 0)
y_nontrivial = np.where(x_vals >= 0,
                         (2 * x_vals / 3)**1.5,
                         0)

# What does a numerical solver do?
def ode_func(x, y):
    """y' = y^(1/3), handling y=0 carefully."""
    return np.sign(y) * np.abs(y)**(1/3)

# Starting exactly at y=0, the solver might find either solution
result = solve_ivp(ode_func, (0, 3), [0.0], t_eval=np.linspace(0, 3, 200),
                   method='RK45', max_step=0.01)

# Starting slightly above 0
result2 = solve_ivp(ode_func, (0, 3), [1e-10], t_eval=np.linspace(0, 3, 200),
                    method='RK45', max_step=0.01)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_vals, y_trivial, 'b-', linewidth=2, label='$y = 0$ (trivial)')
ax.plot(x_vals, y_nontrivial, 'r-', linewidth=2,
        label='$y = (2x/3)^{3/2}$ (nontrivial)')
ax.plot(result.t, result.y[0], 'g--', linewidth=2,
        label='Numerical from $y(0)=0$')
ax.plot(result2.t, result2.y[0], 'm:', linewidth=2,
        label='Numerical from $y(0)=10^{-10}$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Uniqueness Failure: $y' = y^{1/3}$, $y(0) = 0$\n"
             "Two valid solutions through the same point!")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
```

---

## 7. Applications

### 7.1 Newton's Law of Cooling

$$\frac{dT}{dt} = -k(T - T_{\text{env}})$$

- $T(t)$: object temperature
- $T_{\text{env}}$: ambient temperature (constant)
- $k > 0$: cooling constant

Solution: $T(t) = T_{\text{env}} + (T_0 - T_{\text{env}})e^{-kt}$

The temperature exponentially approaches the ambient temperature.

### 7.2 RC Circuit

$$R\frac{dQ}{dt} + \frac{Q}{C} = V(t)$$

- $Q$: charge on capacitor
- $R$: resistance, $C$: capacitance
- $V(t)$: source voltage

This is a linear first-order ODE with time constant $\tau = RC$.

### 7.3 Terminal Velocity

For an object falling with drag:

$$m\frac{dv}{dt} = mg - bv$$

where $b$ is the drag coefficient. Solution: $v(t) = \frac{mg}{b}(1 - e^{-bt/m})$

Terminal velocity: $v_{\text{term}} = mg/b$ (when drag equals gravity).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Three first-order ODE applications ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
t_eval = np.linspace(0, 10, 200)

# 1. Newton's Cooling: hot coffee cooling from 90C in 20C room
T_env, T0, k = 20, 90, 0.3
T_cool = T_env + (T0 - T_env) * np.exp(-k * t_eval)

axes[0].plot(t_eval, T_cool, 'r-', linewidth=2)
axes[0].axhline(y=T_env, color='blue', linestyle='--', label=f'$T_{{env}}={T_env}$°C')
axes[0].set_xlabel('Time (min)')
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title("Newton's Law of Cooling")
axes[0].legend()
axes[0].grid(True)

# 2. RC circuit: charging capacitor (V=5V, R=1kΩ, C=1mF)
R_val, C_val, V0 = 1000, 1e-3, 5.0
tau = R_val * C_val  # time constant = 1 s
t_rc = np.linspace(0, 5 * tau, 200)
Q_charge = C_val * V0 * (1 - np.exp(-t_rc / tau))
V_cap = Q_charge / C_val

axes[1].plot(t_rc, V_cap, 'b-', linewidth=2)
axes[1].axhline(y=V0, color='red', linestyle='--', label=f'$V_0={V0}$ V')
axes[1].axvline(x=tau, color='gray', linestyle=':', label=f'$\\tau={tau}$ s')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Capacitor Voltage (V)')
axes[1].set_title('RC Circuit Charging')
axes[1].legend()
axes[1].grid(True)

# 3. Terminal velocity: skydiver (m=80kg, b=15 kg/s)
m, b_drag, g = 80, 15, 9.8
v_term = m * g / b_drag
t_fall = np.linspace(0, 40, 200)
v_fall = v_term * (1 - np.exp(-b_drag * t_fall / m))

axes[2].plot(t_fall, v_fall, 'g-', linewidth=2)
axes[2].axhline(y=v_term, color='red', linestyle='--',
                label=f'$v_{{term}}={v_term:.1f}$ m/s')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Velocity (m/s)')
axes[2].set_title('Terminal Velocity (Skydiver)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
```

---

## 8. Cross-References

- **Mathematical Methods Lesson 09** provides more advanced first-order ODE techniques (Bernoulli equations, substitution methods) and a deeper treatment of existence theorems.
- **Lesson 13 (Second-Order ODE)** extends the methods here to second-order equations, with physical applications to vibrations.
- **Numerical Simulation Lesson 01** covers more sophisticated numerical ODE solvers (Runge-Kutta, adaptive methods).

---

## Practice Problems

**1.** Solve the separable ODE $\frac{dy}{dx} = \frac{x(1 + y^2)}{y(1 + x^2)}$ with $y(0) = 1$.

**2.** A tank contains 500 L of water with 10 kg of dissolved salt. Fresh water flows in at 5 L/min and the well-mixed solution flows out at 5 L/min. Find the amount of salt $y(t)$ at time $t$ and determine how long until the salt concentration drops below 0.01 kg/L.

**3.** Determine whether the equation $(3x^2 + y\cos x)\,dx + (\sin x - 4y^3)\,dy = 0$ is exact. If so, find the solution.

**4.** Apply Euler's method with step sizes $h = 0.2$ and $h = 0.1$ to solve $y' = y - x^2 + 1$, $y(0) = 0.5$ on $[0, 2]$. Compare with the exact solution $y = (x + 1)^2 - 0.5e^x$ and compute the error at $x = 2$.

**5.** A cup of coffee at 95°C is placed in a room at 22°C. After 5 minutes, the temperature is 70°C.
   - (a) Find the cooling constant $k$.
   - (b) When will the coffee reach 40°C (drinkable temperature)?
   - (c) Use `solve_ivp` to plot the cooling curve and verify analytically.

---

## References

- **William E. Boyce & Richard C. DiPrima**, *Elementary Differential Equations and Boundary Value Problems*, 11th Edition, Chapters 1-2
- **James Stewart**, *Calculus: Early Transcendentals*, 9th Edition, Chapter 9
- **Erwin Kreyszig**, *Advanced Engineering Mathematics*, 10th Edition, Chapter 1
- **SciPy solve_ivp**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- **3Blue1Brown**, "Differential Equations" (visual intuition)

---

[Previous: Vector Calculus](./11_Vector_Calculus.md) | [Next: Second-Order Ordinary Differential Equations](./13_Second_Order_ODE.md)
