# 13. Second-Order Ordinary Differential Equations

## Learning Objectives

- Solve homogeneous second-order ODEs with constant coefficients using the characteristic equation
- Handle all three cases: distinct real roots, repeated roots, and complex conjugate roots
- Find particular solutions using undetermined coefficients and variation of parameters
- Model mechanical vibrations: free, damped, and forced oscillations, including resonance
- Simulate spring-mass systems numerically and visualize resonance phenomena with Python

---

## 1. Homogeneous Equations with Constant Coefficients

### 1.1 The General Form

A **second-order linear ODE with constant coefficients** is:

$$ay'' + by' + cy = g(x)$$

where $a, b, c$ are constants and $g(x)$ is a given function.

When $g(x) = 0$, the equation is **homogeneous**:

$$ay'' + by' + cy = 0$$

### 1.2 The Characteristic Equation

The key insight is to guess $y = e^{rx}$. Substituting:

$$ar^2 e^{rx} + bre^{rx} + ce^{rx} = 0$$

Since $e^{rx} \neq 0$, we divide through to get the **characteristic equation**:

$$ar^2 + br + c = 0$$

This is a quadratic in $r$. Its roots determine the form of the solution entirely.

**Analogy:** The characteristic equation translates a differential equation problem into an algebra problem. Instead of solving a differential equation, we solve a quadratic -- a much simpler task.

### 1.3 Three Cases

**Case 1: Two distinct real roots $r_1 \neq r_2$**

$$y = C_1 e^{r_1 x} + C_2 e^{r_2 x}$$

**Example:** $y'' - 5y' + 6y = 0$

Characteristic equation: $r^2 - 5r + 6 = (r-2)(r-3) = 0$, so $r = 2, 3$.

$$y = C_1 e^{2x} + C_2 e^{3x}$$

**Case 2: Repeated root $r_1 = r_2 = r$**

We only get one independent solution $e^{rx}$. The second is found by multiplying by $x$:

$$y = C_1 e^{rx} + C_2\, x\, e^{rx} = (C_1 + C_2 x)e^{rx}$$

**Example:** $y'' - 4y' + 4y = 0$

Characteristic equation: $r^2 - 4r + 4 = (r-2)^2 = 0$, so $r = 2$ (repeated).

$$y = (C_1 + C_2 x)e^{2x}$$

**Why $xe^{rx}$?** When the characteristic polynomial has a repeated root, the standard exponential gives only one solution. The method of **reduction of order** shows that $xe^{rx}$ is the missing independent solution.

**Case 3: Complex conjugate roots $r = \alpha \pm i\beta$**

Using Euler's formula $e^{i\beta x} = \cos\beta x + i\sin\beta x$:

$$y = e^{\alpha x}(C_1\cos\beta x + C_2\sin\beta x)$$

**Example:** $y'' + 2y' + 5y = 0$

Characteristic equation: $r^2 + 2r + 5 = 0$, so $r = -1 \pm 2i$.

$$y = e^{-x}(C_1\cos 2x + C_2\sin 2x)$$

This represents **oscillation** ($\cos$ and $\sin$) with **exponential decay** ($e^{-x}$) -- exactly what a damped vibration looks like.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Visualize all three cases ---
x = np.linspace(0, 4, 300)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Case 1: distinct real roots (r=2, r=-1)
# y'' - y' - 2y = 0, y(0)=1, y'(0)=0
# y = (1/3)e^(2x) + (2/3)e^(-x)
y1 = (1/3) * np.exp(2*x) + (2/3) * np.exp(-x)
axes[0].plot(x, y1, 'b-', linewidth=2)
axes[0].set_title('Case 1: Distinct Real Roots\n$r = 2, -1$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_ylim(-1, 10)
axes[0].grid(True)

# Case 2: repeated root (r=2)
# y'' - 4y' + 4y = 0, y(0)=1, y'(0)=0
# y = (1 - 2x)e^(2x)
x2 = np.linspace(0, 2, 300)
y2 = (1 - 2*x2) * np.exp(2*x2)
axes[1].plot(x2, y2, 'r-', linewidth=2)
axes[1].axhline(y=0, color='gray', linewidth=0.5)
axes[1].set_title('Case 2: Repeated Root\n$r = 2$ (double)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].grid(True)

# Case 3: complex roots (r = -1 ± 2i)
# y'' + 2y' + 5y = 0, y(0)=1, y'(0)=0
# y = e^(-x)(cos(2x) + (1/2)sin(2x))
y3 = np.exp(-x) * (np.cos(2*x) + 0.5*np.sin(2*x))
envelope = np.exp(-x) * np.sqrt(1 + 0.25)
axes[2].plot(x, y3, 'g-', linewidth=2, label='Solution')
axes[2].plot(x, envelope, 'r--', alpha=0.5, label='Envelope $e^{-x}\\sqrt{5/4}$')
axes[2].plot(x, -envelope, 'r--', alpha=0.5)
axes[2].axhline(y=0, color='gray', linewidth=0.5)
axes[2].set_title('Case 3: Complex Roots\n$r = -1 \\pm 2i$')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
```

---

## 2. Nonhomogeneous Equations

### 2.1 General Solution Structure

For $ay'' + by' + cy = g(x)$, the general solution is:

$$y = y_h + y_p$$

- $y_h$: **homogeneous (complementary) solution** -- solves $ay'' + by' + cy = 0$
- $y_p$: **particular solution** -- any one solution of the full equation

**Analogy:** Finding the general solution is like finding all points on a line. You need one specific point on the line ($y_p$) plus the direction of the line ($y_h$, which captures the "freedom" in the solution).

### 2.2 Method of Undetermined Coefficients

This method works when $g(x)$ is a polynomial, exponential, sine, cosine, or a combination.

**The idea:** Guess a solution with the same form as $g(x)$ but with unknown coefficients. Substitute into the ODE and solve for the coefficients.

| $g(x)$ | Guess for $y_p$ |
|---------|-----------------|
| $P_n(x)$ (polynomial of degree $n$) | $A_n x^n + A_{n-1}x^{n-1} + \cdots + A_0$ |
| $e^{\alpha x}$ | $Ae^{\alpha x}$ |
| $\cos\beta x$ or $\sin\beta x$ | $A\cos\beta x + B\sin\beta x$ |
| $e^{\alpha x}\cos\beta x$ | $e^{\alpha x}(A\cos\beta x + B\sin\beta x)$ |

**Modification rule:** If the guess duplicates a term in $y_h$, multiply by $x$ (or $x^2$ if both terms match).

**Example:** $y'' + 3y' + 2y = 4e^{-x}$

$y_h = C_1 e^{-x} + C_2 e^{-2x}$ (roots $r = -1, -2$)

Initial guess $y_p = Ae^{-x}$ duplicates $C_1 e^{-x}$. Modify: $y_p = Axe^{-x}$.

Substituting: $y_p'' + 3y_p' + 2y_p = Ae^{-x}(-1 + 3(-1) + 2\cdot 0) \ldots$ (after careful algebra) gives $A = -4$.

$$y_p = -4xe^{-x}$$

### 2.3 Variation of Parameters

A more general method that works for **any** $g(x)$, not just special forms.

Given homogeneous solutions $y_1, y_2$, we seek $y_p = u_1 y_1 + u_2 y_2$ where:

$$u_1' = -\frac{y_2 g(x)}{aW}, \quad u_2' = \frac{y_1 g(x)}{aW}$$

and $W = y_1 y_2' - y_2 y_1'$ is the **Wronskian**.

**When to use which method:**
- **Undetermined coefficients:** Quick and easy when $g(x)$ has a suitable form (polynomial, exponential, trig)
- **Variation of parameters:** Always works, but requires computing the Wronskian and integrating -- use when undetermined coefficients doesn't apply

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, dsolve, Eq, exp, cos, sin

x = symbols('x')
y = Function('y')

# --- Undetermined coefficients example ---
# y'' + 3y' + 2y = 10*cos(x)
ode = Eq(y(x).diff(x, 2) + 3*y(x).diff(x) + 2*y(x), 10*cos(x))
sol = dsolve(ode, y(x))
print(f"General solution: {sol}")

# With initial conditions y(0) = 0, y'(0) = 0
sol_ivp = dsolve(ode, y(x), ics={y(0): 0, y(x).diff(x).subs(x, 0): 0})
print(f"IVP solution:     {sol_ivp}")

# --- Variation of parameters example ---
# y'' + y = sec(x)  (cannot use undetermined coefficients!)
ode2 = Eq(y(x).diff(x, 2) + y(x), 1/cos(x))
sol2 = dsolve(ode2, y(x))
print(f"\nVariation of parameters: {sol2}")
```

---

## 3. Mechanical Vibrations

The most important physical application of second-order ODEs is the **mass-spring system**.

### 3.1 The Equation of Motion

For a mass $m$ on a spring (constant $k$) with damping ($\gamma$) and external force $F(t)$:

$$m\ddot{x} + \gamma\dot{x} + kx = F(t)$$

Define:
- $\omega_0 = \sqrt{k/m}$: **natural frequency** (how the system wants to oscillate)
- $\beta = \gamma/(2m)$: **damping coefficient**

The equation becomes:

$$\ddot{x} + 2\beta\dot{x} + \omega_0^2 x = \frac{F(t)}{m}$$

### 3.2 Free Undamped Oscillation ($\beta = 0$, $F = 0$)

$$\ddot{x} + \omega_0^2 x = 0 \implies x(t) = A\cos(\omega_0 t + \phi)$$

This is **simple harmonic motion** -- sinusoidal oscillation at the natural frequency, forever. The amplitude $A$ and phase $\phi$ are determined by initial conditions.

### 3.3 Free Damped Oscillation ($\beta > 0$, $F = 0$)

$$\ddot{x} + 2\beta\dot{x} + \omega_0^2 x = 0$$

Characteristic roots: $r = -\beta \pm \sqrt{\beta^2 - \omega_0^2}$

| Regime | Condition | Behavior |
|--------|-----------|----------|
| **Underdamped** | $\beta < \omega_0$ | Oscillates with decaying amplitude: $x = Ae^{-\beta t}\cos(\omega_d t + \phi)$ |
| **Critically damped** | $\beta = \omega_0$ | Returns to equilibrium fastest without oscillation: $x = (C_1 + C_2 t)e^{-\beta t}$ |
| **Overdamped** | $\beta > \omega_0$ | Slowly returns without oscillation: $x = C_1 e^{r_1 t} + C_2 e^{r_2 t}$ |

where $\omega_d = \sqrt{\omega_0^2 - \beta^2}$ is the **damped frequency** (slightly lower than $\omega_0$).

**Real-world examples:**
- **Underdamped:** Car hitting a pothole (bounces a few times before settling)
- **Critically damped:** Door closer mechanism (returns quickly without slamming)
- **Overdamped:** Heavy door in thick oil (returns very slowly)

### 3.4 Forced Oscillation and Resonance ($F(t) = F_0\cos\omega t$)

$$\ddot{x} + 2\beta\dot{x} + \omega_0^2 x = \frac{F_0}{m}\cos\omega t$$

The steady-state response has the same frequency as the driving force:

$$x_p(t) = A(\omega)\cos(\omega t - \delta)$$

where the **amplitude** and **phase lag** are:

$$A(\omega) = \frac{F_0/m}{\sqrt{(\omega_0^2 - \omega^2)^2 + 4\beta^2\omega^2}}$$

$$\delta = \arctan\frac{2\beta\omega}{\omega_0^2 - \omega^2}$$

**Resonance** occurs near $\omega = \omega_0$ (more precisely at $\omega_{\text{res}} = \sqrt{\omega_0^2 - 2\beta^2}$), where the amplitude peaks.

**The Q-factor** (quality factor) measures the sharpness of resonance:

$$Q = \frac{\omega_0}{2\beta}$$

Higher $Q$ means a sharper, taller resonance peak and slower energy dissipation.

**Resonance disasters:** The Tacoma Narrows Bridge collapse (1940), wine glass shattering from sound, and mechanical failures in poorly designed structures all involve resonance.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Spring-mass system: three damping regimes ---
omega0 = 5.0  # natural frequency
x0, v0 = 1.0, 0.0  # initial displacement 1, initial velocity 0

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Three damping regimes
t_eval = np.linspace(0, 6, 600)
for label, beta, color in [('Underdamped ($\\beta=1$)', 1.0, 'blue'),
                             ('Critically damped ($\\beta=5$)', 5.0, 'green'),
                             ('Overdamped ($\\beta=8$)', 8.0, 'red')]:
    def osc(t, y, b=beta):
        return [y[1], -2*b*y[1] - omega0**2 * y[0]]
    sol = solve_ivp(osc, (0, 6), [x0, v0], t_eval=t_eval)
    axes[0, 0].plot(sol.t, sol.y[0], color=color, linewidth=2, label=label)

axes[0, 0].axhline(y=0, color='gray', linewidth=0.5)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Displacement x(t)')
axes[0, 0].set_title('Free Damped Oscillation')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Top-right: Resonance curves (amplitude vs driving frequency)
omega = np.linspace(0.1, 10, 500)
F0_m = 1.0

for beta in [0.2, 0.5, 1.0, 2.0]:
    A = F0_m / np.sqrt((omega0**2 - omega**2)**2 + (2*beta*omega)**2)
    Q = omega0 / (2*beta)
    axes[0, 1].plot(omega, A, linewidth=2, label=f'$\\beta={beta}$, Q={Q:.1f}')

axes[0, 1].axvline(x=omega0, color='gray', linestyle=':', alpha=0.5)
axes[0, 1].set_xlabel('Driving frequency $\\omega$ (rad/s)')
axes[0, 1].set_ylabel('Amplitude $A(\\omega)$')
axes[0, 1].set_title('Resonance Curves')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Bottom-left: Forced oscillation (transient + steady-state)
beta_f = 0.3
omega_drive = 4.8  # near resonance

def forced_osc(t, y):
    """Forced damped oscillator."""
    return [y[1],
            F0_m * np.cos(omega_drive * t) - 2*beta_f*y[1] - omega0**2*y[0]]

t_long = np.linspace(0, 30, 2000)
sol_f = solve_ivp(forced_osc, (0, 30), [0, 0], t_eval=t_long)

axes[1, 0].plot(sol_f.t, sol_f.y[0], 'b-', linewidth=1)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Displacement x(t)')
axes[1, 0].set_title(f'Forced Oscillation ($\\omega_d={omega_drive}$, near resonance)')
axes[1, 0].grid(True)

# Bottom-right: Beat phenomenon (undamped, omega close to omega0)
omega_beat = 4.5
def beat_osc(t, y):
    """Undamped forced oscillator -- produces beats."""
    return [y[1], F0_m * np.cos(omega_beat * t) - omega0**2 * y[0]]

sol_beat = solve_ivp(beat_osc, (0, 40), [0, 0],
                      t_eval=np.linspace(0, 40, 3000))

axes[1, 1].plot(sol_beat.t, sol_beat.y[0], 'purple', linewidth=1)
# Beat envelope
delta_omega = omega0 - omega_beat
envelope = F0_m / (omega0**2 - omega_beat**2) * 2
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Displacement x(t)')
axes[1, 1].set_title(f'Beat Phenomenon ($\\omega_d={omega_beat}$, $\\omega_0={omega0}$)')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

---

## 4. Phase Lag and Energy

### 4.1 Phase Response

The phase lag $\delta$ tells us how much the response lags behind the driving force:

- At low frequencies ($\omega \ll \omega_0$): $\delta \approx 0$ -- the system follows the force in phase
- At resonance ($\omega = \omega_0$): $\delta = \pi/2$ -- the response lags by a quarter cycle
- At high frequencies ($\omega \gg \omega_0$): $\delta \approx \pi$ -- the response is nearly opposite to the force

### 4.2 Energy in Damped Oscillations

The total mechanical energy of a damped oscillator decreases over time:

$$E(t) = \frac{1}{2}m\dot{x}^2 + \frac{1}{2}kx^2$$

$$\frac{dE}{dt} = -\gamma\dot{x}^2 \le 0$$

The energy loss rate equals the damping force times velocity -- the power dissipated as heat.

For an underdamped oscillator: $E(t) \approx E_0\, e^{-2\beta t}$

The **decay time** (time for energy to drop to $1/e$) is $\tau = 1/(2\beta)$, and the Q-factor is $Q = \omega_0\tau/2 = \omega_0/(2\beta)$.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Energy decay in underdamped oscillation ---
omega0 = 5.0
beta = 0.3
m = 1.0
k = omega0**2 * m

def damped_osc(t, y):
    return [y[1], -2*beta*y[1] - omega0**2 * y[0]]

t_eval = np.linspace(0, 20, 2000)
sol = solve_ivp(damped_osc, (0, 20), [1.0, 0.0], t_eval=t_eval)
x_vals = sol.y[0]
v_vals = sol.y[1]

# Compute kinetic and potential energy
KE = 0.5 * m * v_vals**2
PE = 0.5 * k * x_vals**2
E_total = KE + PE

# Theoretical envelope
E_envelope = 0.5 * k * np.exp(-2 * beta * t_eval)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(t_eval, x_vals, 'b-', linewidth=1.5)
ax1.plot(t_eval, np.exp(-beta * t_eval), 'r--', alpha=0.6, label='Envelope')
ax1.plot(t_eval, -np.exp(-beta * t_eval), 'r--', alpha=0.6)
ax1.set_ylabel('Displacement x(t)')
ax1.set_title('Underdamped Oscillation and Energy Decay')
ax1.legend()
ax1.grid(True)

ax2.plot(t_eval, KE, 'r-', alpha=0.7, linewidth=1, label='Kinetic Energy')
ax2.plot(t_eval, PE, 'b-', alpha=0.7, linewidth=1, label='Potential Energy')
ax2.plot(t_eval, E_total, 'k-', linewidth=2, label='Total Energy')
ax2.plot(t_eval, E_envelope, 'g--', linewidth=2, label='$E_0 e^{-2\\beta t}$')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Energy (J)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

---

## 5. Summary Table

| Type | Equation | Characteristic Roots | Solution Form |
|------|----------|---------------------|---------------|
| Distinct real | $ay'' + by' + cy = 0$ | $r_1 \neq r_2$ (real) | $C_1 e^{r_1 x} + C_2 e^{r_2 x}$ |
| Repeated real | $ay'' + by' + cy = 0$ | $r_1 = r_2 = r$ | $(C_1 + C_2 x)e^{rx}$ |
| Complex | $ay'' + by' + cy = 0$ | $\alpha \pm i\beta$ | $e^{\alpha x}(C_1\cos\beta x + C_2\sin\beta x)$ |
| Nonhomogeneous | $ay'' + by' + cy = g(x)$ | -- | $y_h + y_p$ |

---

## 6. Cross-References

- **Mathematical Methods Lesson 10** covers higher-order ODEs, systems of ODEs, and the power series method (Frobenius).
- **Lesson 12 (First-Order ODE)** provides the foundation for this lesson, including existence-uniqueness and first-order techniques.
- **Lesson 14 (Systems of ODE)** shows how to convert second-order ODEs to systems and analyze them using eigenvalue methods and phase planes.
- **Electrodynamics Lesson 08** applies second-order ODE theory to RLC circuits (the electrical analogue of the spring-mass system).

---

## Practice Problems

**1.** Solve the IVP: $y'' + 6y' + 8y = 0$, $y(0) = 2$, $y'(0) = -1$. Classify the roots and sketch the solution.

**2.** Find the general solution of $y'' + 4y' + 4y = 3e^{-2x}$. (Hint: the guess $Ae^{-2x}$ duplicates both homogeneous solutions -- you need $Ax^2 e^{-2x}$.)

**3.** Use variation of parameters to solve $y'' + y = \tan x$.

**4.** A spring-mass system has $m = 2\,\text{kg}$, $k = 50\,\text{N/m}$, $\gamma = 4\,\text{kg/s}$.
   - (a) Is the system underdamped, critically damped, or overdamped?
   - (b) If $x(0) = 0.1\,\text{m}$ and $\dot{x}(0) = 0$, find $x(t)$.
   - (c) Plot $x(t)$ using `solve_ivp` and overlay the analytical solution.
   - (d) How long until the amplitude drops to 1% of the initial displacement?

**5.** For the forced oscillator $\ddot{x} + 0.4\dot{x} + 25x = 5\cos\omega t$:
   - (a) Find the natural frequency $\omega_0$ and Q-factor.
   - (b) At what driving frequency $\omega$ is the steady-state amplitude maximized?
   - (c) Plot the amplitude $A(\omega)$ and phase $\delta(\omega)$ as functions of $\omega$.
   - (d) Simulate the system at resonance and at $\omega = 1$ (far from resonance) using `solve_ivp`. Compare the steady-state amplitudes.

---

## References

- **William E. Boyce & Richard C. DiPrima**, *Elementary Differential Equations*, 11th Edition, Chapters 3-4
- **Erwin Kreyszig**, *Advanced Engineering Mathematics*, 10th Edition, Chapter 2
- **Mary L. Boas**, *Mathematical Methods in the Physical Sciences*, 3rd Edition, Chapter 8
- **A.P. French**, *Vibrations and Waves*, MIT Introductory Physics Series
- **3Blue1Brown**, "Differential Equations" series

---

[Previous: First-Order Ordinary Differential Equations](./12_First_Order_ODE.md) | [Next: Systems of Ordinary Differential Equations](./14_Systems_of_ODE.md)
