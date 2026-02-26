# Maxwell's Equations — Differential Form

[← Previous: 06. Electromagnetic Induction](06_Electromagnetic_Induction.md) | [Next: 08. Maxwell's Equations — Integral Form →](08_Maxwells_Equations_Integral.md)

---

## Learning Objectives

1. Explain the displacement current and why Maxwell added it to Ampere's law
2. Write all four Maxwell equations in differential form and explain each one physically
3. Derive the electromagnetic wave equation from Maxwell's equations
4. Calculate the speed of light from $\epsilon_0$ and $\mu_0$
5. Introduce the scalar and vector gauge potentials $\phi$ and $\mathbf{A}$ in the time-dependent case
6. Explain the Lorenz gauge and its advantages for wave propagation
7. Demonstrate the wave equation numerically using finite differences

---

This lesson is the climax of classical electrodynamics. We assemble the four laws developed in the preceding lessons — Gauss's law, the no-monopole condition, Faraday's law, and Ampere's law — into a unified set: Maxwell's equations. But there is a surprise. Ampere's law as stated in magnetostatics is inconsistent with charge conservation. Maxwell's genius was to recognize this inconsistency and fix it by adding the **displacement current**. The corrected equations predict something astonishing: electromagnetic waves that travel at the speed of light. Light itself is an electromagnetic wave. With this single deduction, optics, electricity, and magnetism are unified into one theory.

---

## The Inconsistency of Ampere's Law

The magnetostatic form of Ampere's law is:

$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$$

Take the divergence of both sides:

$$\nabla \cdot (\nabla \times \mathbf{B}) = \mu_0 \nabla \cdot \mathbf{J}$$

The left side is identically zero (divergence of a curl vanishes). So this demands:

$$\nabla \cdot \mathbf{J} = 0$$

But the continuity equation (charge conservation) says:

$$\nabla \cdot \mathbf{J} = -\frac{\partial \rho}{\partial t}$$

For magnetostatics ($\partial \rho/\partial t = 0$), this is fine. But for time-varying fields, $\nabla \cdot \mathbf{J} \neq 0$ in general. A charging capacitor is the classic example: current flows into one plate and out of the other, but no current flows between the plates. Yet the B field must exist between the plates.

**Ampere's law is wrong!** (For time-dependent fields.)

---

## The Displacement Current

Maxwell's fix: add a term to Ampere's law to restore consistency. From Gauss's law:

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0} \implies \frac{\partial \rho}{\partial t} = \epsilon_0 \frac{\partial}{\partial t}(\nabla \cdot \mathbf{E}) = \epsilon_0 \nabla \cdot \frac{\partial \mathbf{E}}{\partial t}$$

So the continuity equation becomes:

$$\nabla \cdot \mathbf{J} + \epsilon_0 \nabla \cdot \frac{\partial \mathbf{E}}{\partial t} = 0 \implies \nabla \cdot \left(\mathbf{J} + \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}\right) = 0$$

The modified Ampere's law:

$$\boxed{\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0\epsilon_0 \frac{\partial \mathbf{E}}{\partial t}}$$

The term $\mathbf{J}_d = \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}$ is called the **displacement current density**. It is not a current of real charges — it is a changing electric field that produces a magnetic field just as a real current does.

> **Analogy**: Imagine a relay race where runners (real current $\mathbf{J}$) pass a baton between legs. At the handoff zone (the gap in a charging capacitor), no runner is present, but the baton (the electromagnetic effect) is still being transmitted by the changing electric field. The displacement current is this "phantom runner" that keeps the relay going without a gap.

```python
import numpy as np
import matplotlib.pyplot as plt

# Displacement current in a charging parallel-plate capacitor
# Why this example: it's the canonical case that motivated Maxwell's correction

epsilon_0 = 8.854e-12
mu_0 = 4 * np.pi * 1e-7

# Capacitor parameters
A_plate = 0.01       # plate area (100 cm²)
d = 0.002            # plate separation (2 mm)
C = epsilon_0 * A_plate / d   # capacitance
R = 1000             # charging resistance (Ω)
V0 = 10.0            # source voltage (V)
tau = R * C           # RC time constant

t = np.linspace(0, 5 * tau, 1000)

# Charging current
I_real = (V0 / R) * np.exp(-t / tau)

# Electric field between plates: E = V_cap/(d) = (V₀/d)(1-e^(-t/τ))
E = (V0 / d) * (1 - np.exp(-t / tau))

# Displacement current density: J_d = ε₀ ∂E/∂t
# Why ε₀ ∂E/∂t: this is Maxwell's displacement current
dE_dt = (V0 / d) * (1 / tau) * np.exp(-t / tau)
J_d = epsilon_0 * dE_dt

# Total displacement current through the capacitor gap = J_d × A
I_displacement = J_d * A_plate

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Real current in the wire
axes[0].plot(t / tau, I_real * 1e6, 'b-', linewidth=2)
axes[0].set_xlabel('t / τ')
axes[0].set_ylabel('I (μA)')
axes[0].set_title('Real Current in Wire')
axes[0].grid(True, alpha=0.3)

# Displacement current in the gap
axes[1].plot(t / tau, I_displacement * 1e6, 'r-', linewidth=2)
axes[1].set_xlabel('t / τ')
axes[1].set_ylabel('$I_d$ (μA)')
axes[1].set_title('Displacement Current in Gap')
axes[1].grid(True, alpha=0.3)

# Compare: they should be identical!
axes[2].plot(t / tau, I_real * 1e6, 'b-', linewidth=2, label='Real current $I$')
axes[2].plot(t / tau, I_displacement * 1e6, 'r--', linewidth=2, label='Displacement current $I_d$')
axes[2].set_xlabel('t / τ')
axes[2].set_ylabel('Current (μA)')
axes[2].set_title('$I_{real} = I_{displacement}$ (Continuity!)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle(f'Displacement Current in Charging Capacitor (τ = {tau*1e9:.1f} ns)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('displacement_current.png', dpi=150)
plt.show()

print(f"RC time constant: τ = {tau*1e9:.2f} ns")
print(f"Max real current: I₀ = {V0/R*1e6:.1f} μA")
print(f"Max displacement current: I_d = {epsilon_0 * V0/(d*tau) * A_plate * 1e6:.1f} μA")
```

---

## Maxwell's Equations — The Complete Set

The four Maxwell equations in differential form, in vacuum:

$$\boxed{
\begin{aligned}
(i) \quad & \nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0} && \text{(Gauss's law)} \\[8pt]
(ii) \quad & \nabla \cdot \mathbf{B} = 0 && \text{(No monopoles)} \\[8pt]
(iii) \quad & \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t} && \text{(Faraday's law)} \\[8pt]
(iv) \quad & \nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t} && \text{(Ampere-Maxwell law)}
\end{aligned}
}$$

### Physical Interpretation

| Equation | What it says |
|---|---|
| (i) Gauss | Electric field lines begin/end on charges |
| (ii) No monopoles | Magnetic field lines always close on themselves |
| (iii) Faraday | Changing magnetic fields create circulating electric fields |
| (iv) Ampere-Maxwell | Currents and changing electric fields create circulating magnetic fields |

### The Beautiful Symmetry

Equations (iii) and (iv) reveal a remarkable reciprocity:
- Changing $\mathbf{B}$ produces $\mathbf{E}$ (Faraday)
- Changing $\mathbf{E}$ produces $\mathbf{B}$ (displacement current)

This mutual generation is the mechanism by which electromagnetic waves propagate: an oscillating $\mathbf{E}$ field creates an oscillating $\mathbf{B}$ field, which creates an oscillating $\mathbf{E}$ field, and so on — bootstrapping each other through empty space at the speed of light.

### In Matter

In a linear, isotropic, homogeneous medium:

$$\nabla \cdot \mathbf{D} = \rho_f, \quad \nabla \cdot \mathbf{B} = 0, \quad \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}, \quad \nabla \times \mathbf{H} = \mathbf{J}_f + \frac{\partial \mathbf{D}}{\partial t}$$

where $\mathbf{D} = \epsilon\mathbf{E}$ and $\mathbf{H} = \mathbf{B}/\mu$.

---

## Deriving the Wave Equation

In vacuum with no charges or currents ($\rho = 0$, $\mathbf{J} = 0$), Maxwell's equations become:

$$\nabla \cdot \mathbf{E} = 0, \quad \nabla \cdot \mathbf{B} = 0$$
$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}, \quad \nabla \times \mathbf{B} = \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t}$$

Take the curl of Faraday's law:

$$\nabla \times (\nabla \times \mathbf{E}) = -\frac{\partial}{\partial t}(\nabla \times \mathbf{B}) = -\mu_0\epsilon_0\frac{\partial^2 \mathbf{E}}{\partial t^2}$$

Use the vector identity $\nabla \times (\nabla \times \mathbf{E}) = \nabla(\nabla \cdot \mathbf{E}) - \nabla^2\mathbf{E}$ and $\nabla \cdot \mathbf{E} = 0$:

$$\boxed{\nabla^2\mathbf{E} = \mu_0\epsilon_0\frac{\partial^2\mathbf{E}}{\partial t^2}}$$

Similarly for $\mathbf{B}$:

$$\boxed{\nabla^2\mathbf{B} = \mu_0\epsilon_0\frac{\partial^2\mathbf{B}}{\partial t^2}}$$

These are **wave equations**! Comparing with the standard wave equation $\nabla^2 f = \frac{1}{v^2}\frac{\partial^2 f}{\partial t^2}$:

$$v = \frac{1}{\sqrt{\mu_0\epsilon_0}} = \frac{1}{\sqrt{(4\pi\times10^{-7})(8.854\times10^{-12})}} = 2.998 \times 10^8 \text{ m/s}$$

This is the **speed of light**! Maxwell wrote in 1864: "This velocity is so nearly that of light that it seems we have strong reason to conclude that light itself is an electromagnetic disturbance."

```python
import numpy as np

# Calculate the speed of light from ε₀ and μ₀
# Why from first principles: this was Maxwell's greatest triumph

mu_0 = 4 * np.pi * 1e-7     # permeability of free space (T·m/A)
epsilon_0 = 8.854187817e-12  # permittivity of free space (F/m)

c_calculated = 1 / np.sqrt(mu_0 * epsilon_0)
c_measured = 299_792_458     # exact (by definition since 1983), m/s

print("Speed of Light from Maxwell's Equations")
print("=" * 50)
print(f"μ₀ = {mu_0:.10e} T·m/A")
print(f"ε₀ = {epsilon_0:.10e} F/m")
print(f"")
print(f"c = 1/√(μ₀ε₀) = {c_calculated:.6f} m/s")
print(f"c (measured)    = {c_measured} m/s")
print(f"Relative error  = {abs(c_calculated - c_measured)/c_measured:.2e}")
print(f"")
print(f"This agreement was the smoking gun that light is an EM wave!")
```

---

## Gauge Potentials for Time-Dependent Fields

In the time-dependent case, we still have $\nabla \cdot \mathbf{B} = 0$, so:

$$\mathbf{B} = \nabla \times \mathbf{A}$$

But now Faraday's law gives $\nabla \times \mathbf{E} = -\partial\mathbf{B}/\partial t = -\nabla \times (\partial\mathbf{A}/\partial t)$, so:

$$\nabla \times \left(\mathbf{E} + \frac{\partial \mathbf{A}}{\partial t}\right) = 0$$

A curl-free vector is a gradient:

$$\mathbf{E} + \frac{\partial \mathbf{A}}{\partial t} = -\nabla V$$

Therefore:

$$\boxed{\mathbf{E} = -\nabla V - \frac{\partial \mathbf{A}}{\partial t}}$$

$$\boxed{\mathbf{B} = \nabla \times \mathbf{A}}$$

In electrostatics ($\partial\mathbf{A}/\partial t = 0$), we recover $\mathbf{E} = -\nabla V$.

### Gauge Freedom (Time-Dependent)

The gauge transformation generalizes to:

$$\mathbf{A} \to \mathbf{A}' = \mathbf{A} + \nabla\lambda, \qquad V \to V' = V - \frac{\partial\lambda}{\partial t}$$

Both $\mathbf{E}$ and $\mathbf{B}$ are invariant under this transformation.

---

## The Lorenz Gauge

Substituting the potentials into Maxwell's equations, we get coupled equations for $V$ and $\mathbf{A}$. The **Lorenz gauge** decouples them:

$$\boxed{\nabla \cdot \mathbf{A} + \mu_0\epsilon_0 \frac{\partial V}{\partial t} = 0} \qquad \text{(Lorenz gauge condition)}$$

With this choice, the equations for $V$ and $\mathbf{A}$ become four independent wave equations:

$$\nabla^2 V - \mu_0\epsilon_0 \frac{\partial^2 V}{\partial t^2} = -\frac{\rho}{\epsilon_0}$$

$$\nabla^2 \mathbf{A} - \mu_0\epsilon_0 \frac{\partial^2 \mathbf{A}}{\partial t^2} = -\mu_0 \mathbf{J}$$

Or, using the d'Alembertian operator $\Box^2 \equiv \nabla^2 - \mu_0\epsilon_0\frac{\partial^2}{\partial t^2}$:

$$\Box^2 V = -\rho/\epsilon_0, \qquad \Box^2 \mathbf{A} = -\mu_0\mathbf{J}$$

These are beautiful: each potential satisfies a wave equation driven by its source.

> **Analogy**: The Lorenz gauge is like choosing coordinates that decouple the equations of motion. In classical mechanics, normal modes decouple coupled oscillators. The Lorenz gauge does the same for the electromagnetic potentials — each one evolves independently, driven only by its own source.

---

## Numerical Demonstration: 1D Wave Equation

We can solve the 1D electromagnetic wave equation using finite differences:

$$\frac{\partial^2 E}{\partial t^2} = c^2 \frac{\partial^2 E}{\partial x^2}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Solve the 1D electromagnetic wave equation using FDTD
# Why FDTD: it directly solves Maxwell's equations on a grid

# Domain
L = 10.0              # domain length (in arbitrary units where c = 1)
c = 1.0               # speed of light (normalized)
N = 500               # spatial grid points
dx = L / N
dt = 0.5 * dx / c     # CFL condition: dt < dx/c for stability
# Why CFL: the wave must not travel more than one cell per time step

x = np.linspace(0, L, N)
T_total = 8.0         # total simulation time
N_t = int(T_total / dt)

# Initialize E field: Gaussian pulse
# Why Gaussian: it's a localized disturbance that clearly shows wave propagation
sigma = 0.3
x_center = L / 2
E = np.exp(-((x - x_center) / sigma)**2)
E_prev = E.copy()     # E at previous time step (for leapfrog)

# Store snapshots for visualization
snapshots = []
snapshot_times = [0, 1, 2, 3, 4, 5, 6, 7]
snap_idx = 0

for n in range(N_t):
    t_now = n * dt

    # Save snapshots
    if snap_idx < len(snapshot_times) and t_now >= snapshot_times[snap_idx]:
        snapshots.append((t_now, E.copy()))
        snap_idx += 1

    # Finite difference update: E(t+dt) = 2E(t) - E(t-dt) + (c*dt/dx)² * [E(x+dx) - 2E(x) + E(x-dx)]
    # Why leapfrog: second-order accurate in both space and time
    r = (c * dt / dx)**2
    E_next = np.zeros_like(E)
    E_next[1:-1] = 2*E[1:-1] - E_prev[1:-1] + r * (E[2:] - 2*E[1:-1] + E[:-2])

    # Absorbing boundary conditions (first-order Mur)
    # Why absorbing: we want waves to leave the domain without reflection
    E_next[0] = E[1] + (c*dt - dx)/(c*dt + dx) * (E_next[1] - E[0])
    E_next[-1] = E[-2] + (c*dt - dx)/(c*dt + dx) * (E_next[-2] - E[-1])

    E_prev = E.copy()
    E = E_next.copy()

fig, axes = plt.subplots(4, 2, figsize=(14, 14))
axes_flat = axes.flatten()

for i, (t_snap, E_snap) in enumerate(snapshots):
    if i < len(axes_flat):
        axes_flat[i].plot(x, E_snap, 'b-', linewidth=1.5)
        axes_flat[i].set_ylim(-1.2, 1.2)
        axes_flat[i].set_xlabel('x')
        axes_flat[i].set_ylabel('E')
        axes_flat[i].set_title(f't = {t_snap:.1f}')
        axes_flat[i].grid(True, alpha=0.3)
        axes_flat[i].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

plt.suptitle('1D EM Wave Propagation (FDTD)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wave_equation_1d.png', dpi=150)
plt.show()
```

---

## Retarded Potentials

The solutions to the wave equations for the potentials in the Lorenz gauge are the **retarded potentials**:

$$V(\mathbf{r}, t) = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}', t_r)}{|\mathbf{r}-\mathbf{r}'|}\,d\tau'$$

$$\mathbf{A}(\mathbf{r}, t) = \frac{\mu_0}{4\pi}\int \frac{\mathbf{J}(\mathbf{r}', t_r)}{|\mathbf{r}-\mathbf{r}'|}\,d\tau'$$

where $t_r = t - |\mathbf{r}-\mathbf{r}'|/c$ is the **retarded time** — the time at which a signal traveling at speed $c$ must leave the source point $\mathbf{r}'$ to arrive at the field point $\mathbf{r}$ at time $t$.

The retarded potentials embody **causality**: the fields at $(\mathbf{r}, t)$ depend on what the sources were doing at an earlier time, not what they are doing "now." Electromagnetic effects propagate at the speed of light — they do not act instantaneously.

Note: There are also "advanced" potentials using $t_a = t + |\mathbf{r}-\mathbf{r}'|/c$ (future sources affect present fields). While mathematically valid, they violate causality and are normally discarded. Their existence reflects the time-reversal symmetry of Maxwell's equations.

```python
import numpy as np
import matplotlib.pyplot as plt

# Retarded potential: visualize the causal structure
# Why light cones: they show which source events can influence a given field point

c = 1.0  # normalized speed of light

fig, ax = plt.subplots(figsize=(8, 8))

# Draw the light cone at the observation point (x=0, t=5)
t_obs = 5.0
x_obs = 0.0

# Past light cone: events that can influence (x_obs, t_obs)
t_past = np.linspace(0, t_obs, 100)
# Why ±c(t_obs - t): the light cone boundary in 1+1D spacetime
x_left = x_obs - c * (t_obs - t_past)
x_right = x_obs + c * (t_obs - t_past)

ax.fill_betweenx(t_past, x_left, x_right, alpha=0.15, color='blue',
                  label='Past light cone (causal region)')
ax.plot(x_left, t_past, 'b-', linewidth=1.5)
ax.plot(x_right, t_past, 'b-', linewidth=1.5)
ax.plot(x_obs, t_obs, 'ro', markersize=10, zorder=5, label='Observation event')

# Source events
sources = [(-2, 2), (1, 3), (3, 1), (-4, 4.5)]
for sx, st in sources:
    # Check if inside past light cone
    inside = abs(sx - x_obs) <= c * (t_obs - st) and st < t_obs
    color = 'green' if inside else 'red'
    marker = 's' if inside else 'x'
    label = 'Can influence' if inside else 'Cannot influence'
    ax.plot(sx, st, marker, color=color, markersize=12, zorder=5)
    t_r = t_obs - abs(sx - x_obs) / c
    if inside:
        ax.annotate(f't_r = {t_r:.1f}', (sx, st), textcoords="offset points",
                   xytext=(10, 5), fontsize=9)

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title('Causal Structure: Retarded Potentials Use Past Light Cone')
ax.set_xlim(-6, 6)
ax.set_ylim(0, 6)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('retarded_potentials.png', dpi=150)
plt.show()
```

---

## Electromagnetic Duality

In the absence of charges ($\rho = 0$, $\mathbf{J} = 0$), Maxwell's equations exhibit a remarkable **duality** between $\mathbf{E}$ and $\mathbf{B}$:

If we make the replacements $\mathbf{E} \to c\mathbf{B}$ and $c\mathbf{B} \to -\mathbf{E}$, the source-free Maxwell equations map into themselves. This symmetry would become exact if magnetic monopoles existed, with the full symmetric form:

$$\nabla \cdot \mathbf{E} = \rho_e/\epsilon_0, \quad \nabla \cdot \mathbf{B} = \mu_0\rho_m$$
$$\nabla \times \mathbf{E} = -\mu_0\mathbf{J}_m - \frac{\partial\mathbf{B}}{\partial t}, \quad \nabla \times \mathbf{B} = \mu_0\mathbf{J}_e + \mu_0\epsilon_0\frac{\partial\mathbf{E}}{\partial t}$$

While magnetic monopoles remain unobserved, this duality is a powerful theoretical tool and appears in various guises throughout modern physics.

---

## Dimensional Analysis and Units

It is instructive to verify the units in Maxwell's equations. In SI units:

| Quantity | Symbol | SI Unit |
|---|---|---|
| Electric field | $\mathbf{E}$ | V/m = kg$\cdot$m/(A$\cdot$s$^3$) |
| Magnetic field | $\mathbf{B}$ | T = kg/(A$\cdot$s$^2$) |
| Charge density | $\rho$ | C/m$^3$ = A$\cdot$s/m$^3$ |
| Current density | $\mathbf{J}$ | A/m$^2$ |
| Permittivity | $\epsilon_0$ | F/m = A$^2\cdot$s$^4$/(kg$\cdot$m$^3$) |
| Permeability | $\mu_0$ | H/m = kg$\cdot$m/(A$^2\cdot$s$^2$) |

**Checking** $\nabla \times \mathbf{B} = \mu_0\epsilon_0\partial\mathbf{E}/\partial t$:

- Left: $[\nabla \times \mathbf{B}] = \text{T/m}$
- Right: $[\mu_0\epsilon_0][\mathbf{E}]/[\text{s}] = \text{s}^2/\text{m}^2 \cdot \text{V}/(\text{m}\cdot\text{s}) = \text{T/m}$ ✓

The product $\mu_0\epsilon_0 = 1/c^2$ has units of s$^2$/m$^2$ — the reciprocal of a velocity squared.

---

## Historical Note

Maxwell published his equations in 1865 in "A Dynamical Theory of the Electromagnetic Field." The four equations as we know them today were distilled by Oliver Heaviside in the 1880s from Maxwell's original set of twenty equations using the vector calculus notation that Heaviside himself helped develop. Heinrich Hertz experimentally confirmed electromagnetic waves in 1887, twenty years after Maxwell's prediction.

---

## Summary

| Concept | Key Equation |
|---|---|
| Displacement current | $\mathbf{J}_d = \epsilon_0 \partial\mathbf{E}/\partial t$ |
| Gauss's law | $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$ |
| No monopoles | $\nabla \cdot \mathbf{B} = 0$ |
| Faraday's law | $\nabla \times \mathbf{E} = -\partial\mathbf{B}/\partial t$ |
| Ampere-Maxwell | $\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\epsilon_0\partial\mathbf{E}/\partial t$ |
| Wave equation | $\nabla^2\mathbf{E} = \mu_0\epsilon_0\,\partial^2\mathbf{E}/\partial t^2$ |
| Speed of light | $c = 1/\sqrt{\mu_0\epsilon_0}$ |
| Potentials | $\mathbf{E} = -\nabla V - \partial\mathbf{A}/\partial t$, $\mathbf{B} = \nabla\times\mathbf{A}$ |
| Lorenz gauge | $\nabla\cdot\mathbf{A} + \mu_0\epsilon_0\,\partial V/\partial t = 0$ |
| d'Alembertian | $\Box^2 V = -\rho/\epsilon_0$, $\Box^2\mathbf{A} = -\mu_0\mathbf{J}$ |

---

## Exercises

### Exercise 1: Displacement Current Magnitude
A circular parallel-plate capacitor (radius $R = 5$ cm, gap $d = 2$ mm) is being charged by a current $I = 0.5$ A. Calculate the displacement current density between the plates and the magnetic field at the edge of the plates. Compare with the B field you would get from a real current.

### Exercise 2: Wave Equation Derivation
Starting from Maxwell's equations, derive the wave equation for $\mathbf{B}$ (parallel to the derivation for $\mathbf{E}$ given in this lesson). Verify that both $\mathbf{E}$ and $\mathbf{B}$ propagate at the same speed.

### Exercise 3: 2D Wave Simulation
Extend the 1D FDTD simulation to 2D. Start with a point-like initial perturbation and observe the circular wave spreading outward. Verify that the wave speed matches $c$.

### Exercise 4: Lorenz Gauge Verification
For a time-varying point charge $q(t) = q_0 \sin(\omega t)$ at the origin, the retarded potentials are known. Write them down and verify that they satisfy the Lorenz gauge condition.

### Exercise 5: Gauge Transformation
Start from the Lorenz gauge potentials for a uniformly moving charge. Apply a gauge transformation with $\lambda = f(x - vt)$ and verify that the fields $\mathbf{E}$ and $\mathbf{B}$ are unchanged.

---

[← Previous: 06. Electromagnetic Induction](06_Electromagnetic_Induction.md) | [Next: 08. Maxwell's Equations — Integral Form →](08_Maxwells_Equations_Integral.md)
