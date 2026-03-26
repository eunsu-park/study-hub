# Electromagnetic Induction

[← Previous: 05. Magnetic Vector Potential](05_Magnetic_Vector_Potential.md) | [Next: 07. Maxwell's Equations — Differential Form →](07_Maxwells_Equations_Differential.md)

---

## Learning Objectives

1. State Faraday's law of electromagnetic induction in both integral and differential forms
2. Explain Lenz's law and its connection to energy conservation
3. Compute motional EMF for moving conductors in magnetic fields
4. Define self-inductance and mutual inductance and calculate them for standard geometries
5. Derive the energy stored in a magnetic field and the magnetic energy density
6. Analyze RL circuit transients using inductance
7. Simulate electromagnetic induction phenomena numerically in Python

---

Until now, we have treated electric and magnetic fields as entirely separate phenomena — electric fields from charges, magnetic fields from currents, with no interaction between the two. In 1831, Michael Faraday shattered this separation with a simple but revolutionary experiment: he showed that a changing magnetic field produces an electric field. This discovery of electromagnetic induction is the bridge that connects electricity to magnetism and ultimately leads to Maxwell's equations and the prediction of electromagnetic waves. It is also the principle behind every electric generator, transformer, and wireless charger in the modern world.

---

## Faraday's Law

### The Experiment

Faraday observed that an EMF (electromotive force) is induced in a circuit whenever the magnetic flux through the circuit changes with time. The faster the change, the larger the EMF.

### Integral Form

$$\boxed{\mathcal{E} = -\frac{d\Phi_B}{dt}}$$

where the magnetic flux through a surface $S$ bounded by the circuit $C$ is:

$$\Phi_B = \int_S \mathbf{B} \cdot d\mathbf{a}$$

The EMF around the circuit is:

$$\mathcal{E} = \oint_C \mathbf{E} \cdot d\mathbf{l}$$

So Faraday's law in full integral form reads:

$$\oint_C \mathbf{E} \cdot d\mathbf{l} = -\frac{d}{dt}\int_S \mathbf{B} \cdot d\mathbf{a}$$

### Differential Form

Using Stokes' theorem on the left side:

$$\int_S (\nabla \times \mathbf{E}) \cdot d\mathbf{a} = -\int_S \frac{\partial \mathbf{B}}{\partial t} \cdot d\mathbf{a}$$

Since this must hold for any surface $S$:

$$\boxed{\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}}$$

This is a profound upgrade from electrostatics: in electrostatics, $\nabla \times \mathbf{E} = 0$ (conservative field). Now, with time-varying $\mathbf{B}$, the electric field develops a curl — it is no longer conservative. The electric field can push charges around closed loops.

> **Analogy**: Imagine a whirlpool that only appears when you stir the water (change the magnetic field). In electrostatics, the electric "water" flows straight downhill (from high V to low V). With electromagnetic induction, the changing magnetic field acts like a stirring spoon, creating circular currents of electric field that have no beginning or end — just like a whirlpool.

---

## Lenz's Law

The negative sign in Faraday's law is called **Lenz's law**:

> The induced EMF opposes the change in flux that produces it.

This is a consequence of **energy conservation**. If the induced current enhanced the flux change instead of opposing it, the system would run away — producing ever-increasing currents and fields, violating conservation of energy.

**Practical rule**: If the flux through a loop is increasing, the induced current flows in the direction that produces a magnetic field opposing the increase (trying to maintain the original flux). If the flux is decreasing, the current flows to maintain it.

```python
import numpy as np
import matplotlib.pyplot as plt

# Faraday's law: EMF induced by a time-varying magnetic field through a loop
# Why simulation: seeing the relationship between dΦ/dt and EMF builds intuition

# A circular loop of radius R is in a time-varying uniform B field
R_loop = 0.1        # loop radius (10 cm)
A_loop = np.pi * R_loop**2   # loop area

t = np.linspace(0, 2, 1000)  # time in seconds

# Case 1: Linearly increasing B
B_linear = 0.5 * t    # B in Tesla, increasing at 0.5 T/s
Phi_linear = B_linear * A_loop
EMF_linear = -np.gradient(Phi_linear, t)   # EMF = -dΦ/dt

# Case 2: Sinusoidal B
omega = 2 * np.pi * 2    # 2 Hz
B_sin = 0.5 * np.sin(omega * t)
Phi_sin = B_sin * A_loop
EMF_sin = -np.gradient(Phi_sin, t)

# Case 3: Exponentially decaying B
tau = 0.5       # decay time constant
B_exp = 0.5 * np.exp(-t / tau)
Phi_exp = B_exp * A_loop
EMF_exp = -np.gradient(Phi_exp, t)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Why paired plots: seeing B(t) alongside EMF(t) makes Faraday's law visual
for idx, (B, EMF, Phi, label) in enumerate([
    (B_linear, EMF_linear, Phi_linear, 'Linear B(t) = 0.5t'),
    (B_sin, EMF_sin, Phi_sin, 'Sinusoidal B(t) = 0.5sin(ωt)'),
    (B_exp, EMF_exp, Phi_exp, 'Exponential B(t) = 0.5e^{-t/τ}')
]):
    axes[idx, 0].plot(t, B, 'b-', linewidth=2, label='B(t)')
    axes[idx, 0].plot(t, Phi * 100, 'g--', linewidth=1.5, label='Φ(t) ×100')
    axes[idx, 0].set_ylabel('B (T) / Φ×100 (Wb)')
    axes[idx, 0].set_title(f'{label}')
    axes[idx, 0].legend()
    axes[idx, 0].grid(True, alpha=0.3)

    axes[idx, 1].plot(t, EMF * 1e3, 'r-', linewidth=2)
    axes[idx, 1].set_ylabel('EMF (mV)')
    axes[idx, 1].set_title(f'EMF = -dΦ/dt')
    axes[idx, 1].grid(True, alpha=0.3)
    axes[idx, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

for ax in axes[-1, :]:
    ax.set_xlabel('Time (s)')

plt.suptitle("Faraday's Law: EMF = -dΦ/dt", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('faraday_law.png', dpi=150)
plt.show()
```

---

## Motional EMF

When a conductor moves through a magnetic field, the free charges inside it experience a magnetic force $\mathbf{F} = q\mathbf{v} \times \mathbf{B}$. This force drives current and produces an EMF.

### Sliding Bar

A conducting bar of length $l$ slides with velocity $v$ along two parallel rails in a uniform magnetic field $\mathbf{B} = B\hat{z}$ (perpendicular to the plane of the rails):

$$\mathcal{E} = Blv$$

This is consistent with Faraday's law: as the bar moves, the area of the circuit increases at rate $dA/dt = lv$, so $d\Phi/dt = BlV$ and $|\mathcal{E}| = Blv$.

### General Motional EMF

For an arbitrary conducting loop moving with velocity $\mathbf{v}$ in a magnetic field:

$$\mathcal{E}_{\text{motional}} = \oint (\mathbf{v} \times \mathbf{B}) \cdot d\mathbf{l}$$

### Faraday's Law Unifies Both Cases

Faraday's law $\mathcal{E} = -d\Phi_B/dt$ applies universally:
1. **Transformer EMF**: $\mathbf{B}$ changes, loop is stationary $\to$ $\nabla \times \mathbf{E} = -\partial\mathbf{B}/\partial t$
2. **Motional EMF**: Loop moves, $\mathbf{B}$ is static $\to$ force from $\mathbf{v} \times \mathbf{B}$
3. **Both**: Both effects contribute

```python
import numpy as np
import matplotlib.pyplot as plt

# Motional EMF: conducting bar sliding on rails in a uniform B field
# Why simulate: watching the current build up and the bar decelerate is instructive

B = 0.5          # magnetic field (T)
l = 0.2          # bar length (m)
R = 1.0          # circuit resistance (Ω)
m = 0.01         # bar mass (kg)
v0 = 5.0         # initial velocity (m/s)

# Time integration
# Why fine dt: the exponential decay requires good resolution
dt = 1e-4
t_max = 0.1
t = np.arange(0, t_max, dt)
N = len(t)

v = np.zeros(N)
x = np.zeros(N)
v[0] = v0

for i in range(N - 1):
    # EMF from motion
    emf = B * l * v[i]

    # Current in the circuit: I = EMF/R
    I = emf / R

    # Force on the bar: F = -BIl (Lenz's law — opposes motion)
    # Why negative: the induced current creates a force opposing the velocity
    F = -B * I * l

    # Update velocity and position
    v[i+1] = v[i] + (F / m) * dt
    x[i+1] = x[i] + v[i+1] * dt

# Analytic solution: v(t) = v₀ exp(-B²l²t/(mR))
# Why exponential: the braking force is proportional to velocity
tau_decay = m * R / (B * l)**2
v_analytic = v0 * np.exp(-t / tau_decay)
emf_values = B * l * v
I_values = emf_values / R

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(t * 1000, v, 'b-', linewidth=2, label='Numerical')
axes[0, 0].plot(t * 1000, v_analytic, 'r--', linewidth=2, label='Analytic')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Velocity (m/s)')
axes[0, 0].set_title('Bar Velocity')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t * 1000, emf_values * 1000, 'g-', linewidth=2)
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('EMF (mV)')
axes[0, 1].set_title('Induced EMF = Blv')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(t * 1000, I_values * 1000, 'r-', linewidth=2)
axes[1, 0].set_xlabel('Time (ms)')
axes[1, 0].set_ylabel('Current (mA)')
axes[1, 0].set_title('Induced Current = EMF/R')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t * 1000, x * 100, 'm-', linewidth=2)
axes[1, 1].set_xlabel('Time (ms)')
axes[1, 1].set_ylabel('Position (cm)')
axes[1, 1].set_title('Bar Position')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'Motional EMF: Sliding Bar (τ = {tau_decay*1000:.2f} ms)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('motional_emf.png', dpi=150)
plt.show()

print(f"Decay time constant: τ = {tau_decay*1000:.2f} ms")
print(f"Initial EMF: {B*l*v0*1000:.1f} mV")
print(f"At t = τ, v = {v0/np.e:.3f} m/s (v₀/e)")
```

---

## Self-Inductance

When a current flows in a circuit, it produces a magnetic flux through the circuit itself. If the current changes, this flux changes, inducing an EMF that opposes the change. This phenomenon is **self-induction**.

The **self-inductance** $L$ is defined by:

$$\Phi_B = LI$$

and the back-EMF is:

$$\mathcal{E} = -L\frac{dI}{dt}$$

The unit of inductance is the **henry** (H): 1 H = 1 V$\cdot$s/A = 1 Wb/A.

### Inductance of Standard Geometries

**Solenoid** (length $l$, $N$ turns, cross-section area $A$):
$$L = \mu_0 \frac{N^2}{l} A = \mu_0 n^2 l A$$

**Toroid** ($N$ turns, inner radius $a$, outer radius $b$, height $h$):
$$L = \frac{\mu_0 N^2 h}{2\pi} \ln\frac{b}{a}$$

**Coaxial cable** (inner radius $a$, outer radius $b$, length $l$):
$$L = \frac{\mu_0 l}{2\pi} \ln\frac{b}{a}$$

> **Analogy**: Inductance is the electromagnetic analogue of inertia. Just as a massive object resists changes in velocity ($F = ma = m \, dv/dt$), an inductor resists changes in current ($\mathcal{E} = -L \, dI/dt$). A coil with large $L$ is "heavy" — it takes a large EMF to change the current quickly.

---

## Mutual Inductance

When two circuits are near each other, the current in one produces flux through the other. The **mutual inductance** $M$ relates the flux through circuit 2 due to current in circuit 1:

$$\Phi_{21} = M_{21} I_1$$

**Neumann formula** (for two loops):
$$M = \frac{\mu_0}{4\pi} \oint_{C_1} \oint_{C_2} \frac{d\mathbf{l}_1 \cdot d\mathbf{l}_2}{|\mathbf{r}_1 - \mathbf{r}_2|}$$

**Key property**: $M_{12} = M_{21} \equiv M$ — mutual inductance is symmetric! The proof uses the Neumann formula, which is manifestly symmetric in the two loops.

### Transformer

A transformer exploits mutual inductance. Two coils wound on a common core:

$$\frac{V_2}{V_1} = \frac{N_2}{N_1}$$

This voltage ratio is exact for an ideal transformer (perfect coupling, no resistance).

```python
import numpy as np
import matplotlib.pyplot as plt

# Mutual inductance: two coaxial loops
# Why Neumann formula: it gives M directly from the geometry

mu_0 = 4 * np.pi * 1e-7

def mutual_inductance_coaxial_loops(R1, R2, d, N_segments=1000):
    """
    Compute mutual inductance between two coaxial circular loops.

    Parameters:
        R1, R2: radii of the two loops
        d: axial separation between loop centers
        N_segments: number of segments per loop

    Returns:
        M: mutual inductance (H)
    """
    # Discretize both loops
    phi1 = np.linspace(0, 2*np.pi, N_segments, endpoint=False)
    phi2 = np.linspace(0, 2*np.pi, N_segments, endpoint=False)
    dphi = 2 * np.pi / N_segments

    # Neumann formula: M = (μ₀/4π) ∮∮ dl₁·dl₂ / |r₁-r₂|
    # Why double integral: each segment of loop 1 interacts with each of loop 2
    M = 0.0
    for p1 in phi1:
        # Position and direction of current element on loop 1 (at z=0)
        x1 = R1 * np.cos(p1)
        y1 = R1 * np.sin(p1)
        z1 = 0.0
        dl1_x = -R1 * np.sin(p1) * dphi
        dl1_y = R1 * np.cos(p1) * dphi

        # All segments of loop 2 (at z=d)
        x2 = R2 * np.cos(phi2)
        y2 = R2 * np.sin(phi2)
        z2 = d

        dl2_x = -R2 * np.sin(phi2) * dphi
        dl2_y = R2 * np.cos(phi2) * dphi

        # Separation
        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # dl₁ · dl₂
        dot = dl1_x * dl2_x + dl1_y * dl2_y

        M += np.sum(dot / r)

    M *= mu_0 / (4 * np.pi)
    return M

# Compute M as a function of separation
R1 = R2 = 0.1   # both loops have radius 10 cm
separations = np.linspace(0.01, 0.5, 50)
M_values = np.array([mutual_inductance_coaxial_loops(R1, R2, d, N_segments=500)
                      for d in separations])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(separations * 100, M_values * 1e6, 'b-', linewidth=2)
ax.set_xlabel('Separation d (cm)')
ax.set_ylabel('M (μH)')
ax.set_title(f'Mutual Inductance of Two Coaxial Loops (R = {R1*100:.0f} cm)')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('mutual_inductance.png', dpi=150)
plt.show()

print(f"M at d = R:     {mutual_inductance_coaxial_loops(R1, R2, R1, 500)*1e6:.4f} μH")
print(f"M at d = 0.01m: {M_values[0]*1e6:.4f} μH")
```

---

## Energy Stored in a Magnetic Field

The energy stored in an inductor carrying current $I$:

$$W = \frac{1}{2}LI^2$$

More generally, the energy stored in a magnetic field is:

$$\boxed{W = \frac{1}{2\mu_0}\int_{\text{all space}} |\mathbf{B}|^2 \, d\tau}$$

The **magnetic energy density** is:

$$u_B = \frac{B^2}{2\mu_0} \quad [\text{J/m}^3]$$

Compare with the electric energy density $u_E = \frac{\epsilon_0}{2}E^2$. In an electromagnetic field, the total energy density is:

$$u = \frac{1}{2}\left(\epsilon_0 E^2 + \frac{B^2}{\mu_0}\right)$$

### Example: Energy in a Solenoid

Inside a solenoid ($B = \mu_0 nI$, volume $= Al$):

$$W = \frac{B^2}{2\mu_0}(Al) = \frac{(\mu_0 nI)^2}{2\mu_0}Al = \frac{1}{2}\mu_0 n^2 Al \cdot I^2 = \frac{1}{2}LI^2$$

Consistent!

---

## RL Circuits

An RL circuit (resistor $R$ + inductor $L$ in series with a voltage source $V_0$):

$$V_0 = IR + L\frac{dI}{dt}$$

### Charging (Switch closed at $t=0$):

$$I(t) = \frac{V_0}{R}\left(1 - e^{-t/\tau}\right), \qquad \tau = \frac{L}{R}$$

The time constant $\tau = L/R$ governs how quickly the current rises. Larger $L$ (more "inertia") means slower rise.

### Discharging (Source removed):

$$I(t) = I_0 \, e^{-t/\tau}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# RL circuit transients: charging and discharging
# Why both analytic and numerical: the numerical approach generalizes to nonlinear circuits

R = 10.0        # resistance (Ω)
L = 0.1         # inductance (H)
V0 = 5.0        # source voltage (V)
tau = L / R      # time constant

t = np.linspace(0, 5 * tau, 1000)
dt = t[1] - t[0]

# Analytic solutions
I_charge_analytic = (V0 / R) * (1 - np.exp(-t / tau))
I_discharge_analytic = (V0 / R) * np.exp(-t / tau)

# Numerical solution using Euler method
I_charge_numerical = np.zeros_like(t)
I_discharge_numerical = np.zeros_like(t)
I_discharge_numerical[0] = V0 / R

for i in range(len(t) - 1):
    # Charging: V₀ = IR + L dI/dt → dI/dt = (V₀ - IR)/L
    dI_dt_c = (V0 - I_charge_numerical[i] * R) / L
    I_charge_numerical[i+1] = I_charge_numerical[i] + dI_dt_c * dt

    # Discharging: 0 = IR + L dI/dt → dI/dt = -IR/L
    dI_dt_d = -I_discharge_numerical[i] * R / L
    I_discharge_numerical[i+1] = I_discharge_numerical[i] + dI_dt_d * dt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Charging
axes[0].plot(t / tau, I_charge_analytic * 1000, 'b-', linewidth=2, label='Analytic')
axes[0].plot(t / tau, I_charge_numerical * 1000, 'r--', linewidth=2, label='Numerical')
axes[0].axhline(y=V0/R * 1000, color='gray', linestyle=':', label=f'I_max = {V0/R*1000:.0f} mA')
axes[0].axvline(x=1, color='green', linestyle='--', alpha=0.5, label=f'τ = {tau*1000:.1f} ms')
axes[0].set_xlabel('t / τ')
axes[0].set_ylabel('I (mA)')
axes[0].set_title('RL Charging')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Discharging
axes[1].plot(t / tau, I_discharge_analytic * 1000, 'b-', linewidth=2, label='Analytic')
axes[1].plot(t / tau, I_discharge_numerical * 1000, 'r--', linewidth=2, label='Numerical')
axes[1].axvline(x=1, color='green', linestyle='--', alpha=0.5, label=f'τ = {tau*1000:.1f} ms')
axes[1].set_xlabel('t / τ')
axes[1].set_ylabel('I (mA)')
axes[1].set_title('RL Discharging')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'RL Circuit (R = {R} Ω, L = {L*1000:.0f} mH, τ = {tau*1000:.1f} ms)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rl_circuit.png', dpi=150)
plt.show()
```

---

## Summary

| Concept | Key Equation |
|---|---|
| Faraday's law (integral) | $\mathcal{E} = -d\Phi_B/dt$ |
| Faraday's law (differential) | $\nabla \times \mathbf{E} = -\partial\mathbf{B}/\partial t$ |
| Lenz's law | Induced EMF opposes the change |
| Motional EMF | $\mathcal{E} = \oint (\mathbf{v}\times\mathbf{B})\cdot d\mathbf{l}$ |
| Self-inductance | $\mathcal{E} = -L\,dI/dt$ |
| Solenoid inductance | $L = \mu_0 n^2 l A$ |
| Mutual inductance | $\Phi_{21} = MI_1$ |
| Neumann formula | $M = \frac{\mu_0}{4\pi}\oint\oint \frac{d\mathbf{l}_1\cdot d\mathbf{l}_2}{|\mathbf{r}_1-\mathbf{r}_2|}$ |
| Magnetic energy | $W = \frac{1}{2}LI^2 = \frac{1}{2\mu_0}\int B^2\,d\tau$ |
| Magnetic energy density | $u_B = B^2/(2\mu_0)$ |
| RL time constant | $\tau = L/R$ |

---

## Exercises

### Exercise 1: AC Generator
A rectangular loop of area $A$ rotates with angular frequency $\omega$ in a uniform field $\mathbf{B}$. Show that the induced EMF is $\mathcal{E} = NBA\omega\sin(\omega t)$ for $N$ turns. Plot the EMF and instantaneous power for realistic values.

### Exercise 2: Eddy Current Braking
A conducting disk rotates in a magnetic field. Model the braking torque from eddy currents and simulate the deceleration. How does the braking torque depend on angular velocity?

### Exercise 3: Coupled RL Circuits
Two coils with self-inductances $L_1$, $L_2$ and mutual inductance $M$ are connected in series. Derive the equations of motion and compute the effective inductance. Compare the cases of aiding ($L_{\text{eff}} = L_1 + L_2 + 2M$) and opposing ($L_{\text{eff}} = L_1 + L_2 - 2M$) configurations.

### Exercise 4: Energy Conservation in Induction
For the sliding bar problem, verify energy conservation numerically: show that the kinetic energy lost by the bar equals the total energy dissipated in the resistance ($\int I^2 R \, dt$).

### Exercise 5: Inductance of a Toroid
A toroidal coil with $N = 500$ turns, inner radius $a = 10$ cm, outer radius $b = 15$ cm, and height $h = 3$ cm. Compute its inductance analytically. Then model it as $N$ circular current loops and compute the inductance numerically via the Neumann formula. Compare the results.

---

[← Previous: 05. Magnetic Vector Potential](05_Magnetic_Vector_Potential.md) | [Next: 07. Maxwell's Equations — Differential Form →](07_Maxwells_Equations_Differential.md)
