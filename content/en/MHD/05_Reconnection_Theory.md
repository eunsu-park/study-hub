# 5. Magnetic Reconnection Theory

## Learning Objectives

By the end of this lesson, you should be able to:

1. Explain why magnetic reconnection is fundamental to plasma physics and astrophysics
2. Derive the Sweet-Parker reconnection rate and understand its limitations
3. Analyze the Petschek model and explain how it achieves faster reconnection
4. Understand the role of Hall physics in collisionless reconnection
5. Describe X-point geometry and the structure of reconnection regions
6. Calculate and interpret dimensionless reconnection rates
7. Implement numerical models of reconnection rate scaling

## 1. Introduction to Magnetic Reconnection

### 1.1 What is Magnetic Reconnection?

Magnetic reconnection is a fundamental plasma process in which magnetic field topology changes and magnetic energy is rapidly converted to plasma kinetic and thermal energy. This process is responsible for some of the most explosive phenomena in the universe.

```
Key concept: In ideal MHD, magnetic field lines are "frozen in" to the plasma
(flux freezing). Reconnection breaks this constraint, allowing field lines to
break and reconnect in a different configuration.
```

The process occurs in thin current sheets where the frozen-in condition breaks down due to non-ideal effects (resistivity, Hall physics, or kinetic effects). The changing topology allows magnetic energy to be released, often explosively.

### 1.2 Why Reconnection Matters

Magnetic reconnection is crucial for understanding:

**Solar Physics:**
- Solar flares: Energy release of ~10³²–10³³ erg in minutes
- Coronal mass ejections (CMEs): Billion-ton plasma eruptions
- Coronal heating: Maintaining million-degree corona

**Space Physics:**
- Magnetospheric substorms: Auroral brightening events
- Solar wind-magnetosphere coupling: Energy transfer at magnetopause
- Particle acceleration: Non-thermal particle populations

**Laboratory Plasmas:**
- Tokamak sawtooth crashes: Rapid core temperature drops
- Disruptions: Catastrophic loss of confinement
- Spheromak and RFP relaxation: Magnetic self-organization

**Astrophysics:**
- Pulsar magnetospheres: High-energy radiation
- Accretion disk coronae: X-ray emission
- Gamma-ray burst jets: Relativistic outflows

The key puzzle that drove reconnection theory: How can reconnection be fast enough to explain observations when naive estimates predict extremely slow rates?

### 1.3 The Frozen-In Field Theorem and Its Breakdown

In ideal MHD, the electric field is:

$$\mathbf{E} = -\mathbf{v} \times \mathbf{B}$$

This leads to the frozen-in flux theorem: the magnetic flux through any closed loop moving with the plasma is conserved. Field lines can be thought of as moving with the plasma.

When resistivity is included, Ohm's law becomes:

$$\mathbf{E} = -\mathbf{v} \times \mathbf{B} + \eta \mathbf{J}$$

where $\eta$ is the resistivity and $\mathbf{J}$ is the current density. The resistive term allows field lines to "slip" through the plasma. However, as we'll see, classical resistivity alone is too small to explain observed reconnection rates.

The induction equation becomes:

$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) + \eta \nabla^2 \mathbf{B}$$

The relative importance of the ideal and resistive terms is measured by the magnetic Reynolds number:

$$R_m = \frac{Lv_A}{\eta}$$

where $L$ is a characteristic length scale and $v_A = B/\sqrt{\mu_0 \rho}$ is the Alfvén velocity. For most astrophysical and space plasmas, $R_m \gg 1$, making reconnection a challenging problem.

## 2. Sweet-Parker Reconnection

### 2.1 Model Setup

The Sweet-Parker model, independently developed by Sweet (1958) and Parker (1957), was the first quantitative theory of magnetic reconnection. It assumes:

1. **Steady state**: Time-independent configuration
2. **Two-dimensional geometry**: Invariance along the current direction
3. **Thin current sheet**: Length $L \gg$ width $\delta$
4. **Symmetric inflow**: Plasma flows in from both sides
5. **Classical resistivity**: $\eta$ is constant

```
                    Inflow v_in
                        ↓
                        ↓
        B_in ←  ←  ←  ←┼→  →  →  → B_in
                        ↓
        ================╋================ Current sheet
                        ↓                 (length L, width δ)
        B_in ←  ←  ←  ←┼→  →  →  → B_in
                        ↓
                        ↓ Outflow v_out
```

The reconnecting field is antiparallel: $\mathbf{B} = B_{in} \hat{x}$ above the sheet and $\mathbf{B} = -B_{in} \hat{x}$ below. The reconnection electric field $E_z$ is uniform and points out of the page.

### 2.2 Derivation of the Reconnection Rate

We derive the reconnection rate by applying conservation laws to the diffusion region.

**Mass Conservation:**

The mass inflow rate must equal the mass outflow rate:

$$\rho v_{in} L \approx \rho v_{out} \delta$$

Therefore:

$$v_{out} \approx v_{in} \frac{L}{\delta}$$

**Momentum Conservation:**

The magnetic pressure drives the outflow. Balancing magnetic pressure with dynamic pressure:

$$\frac{B_{in}^2}{2\mu_0} \approx \rho v_{out}^2$$

This gives the outflow velocity as approximately the Alfvén speed:

$$v_{out} \approx v_A = \frac{B_{in}}{\sqrt{\mu_0 \rho}}$$

**Ohm's Law at the X-point:**

At the center of the diffusion region, the reconnection electric field is:

$$E_z = \eta J_z$$

where $J_z \approx B_{in}/(\mu_0 \delta)$ from Ampère's law. Outside the diffusion region (in the ideal region), Ohm's law gives:

$$E_z = v_{in} B_{in}$$

Equating these:

$$v_{in} B_{in} = \eta \frac{B_{in}}{\mu_0 \delta}$$

Solving for the width:

$$\delta = \frac{\eta}{\mu_0 v_{in}}$$

**Solving for $v_{in}$:**

Combining $v_{out} = v_{in} L/\delta$ with $v_{out} = v_A$ and the expression for $\delta$:

$$v_A = v_{in} \frac{L}{\eta/(\mu_0 v_{in})} = v_{in} \frac{L \mu_0 v_{in}}{\eta}$$

$$v_A = \frac{L \mu_0 v_{in}^2}{\eta}$$

$$v_{in} = \sqrt{\frac{\eta v_A}{L \mu_0}} = v_A \sqrt{\frac{\eta}{L v_A \mu_0}}$$

**Dimensionless Reconnection Rate:**

Define the Lundquist number:

$$S = \frac{L v_A \mu_0}{\eta} = \frac{L v_A}{\eta/\mu_0}$$

This is the magnetic Reynolds number based on the sheet length. The dimensionless reconnection rate is:

$$M_A = \frac{v_{in}}{v_A} = S^{-1/2}$$

Also, the aspect ratio is:

$$\frac{\delta}{L} = S^{-1}$$

### 2.3 The Sweet-Parker Problem

For a solar flare, typical parameters are:

- $L \sim 10^9$ m (10,000 km)
- $B \sim 0.01$ T (100 G)
- $n \sim 10^{16}$ m⁻³
- $T \sim 10^7$ K
- Spitzer resistivity: $\eta \sim 10^{-4}$ Ω·m

This gives:

$$v_A \sim 10^6 \text{ m/s}$$

$$S \sim \frac{10^9 \times 10^6}{10^{-4}/\mu_0} \sim 10^{14}$$

$$M_A \sim 10^{-7}$$

$$v_{in} \sim 10^{-1} \text{ m/s}$$

**The problem:** At this rate, reconnecting the field would take:

$$t \sim \frac{L}{v_{in}} \sim 10^{10} \text{ s} \sim 300 \text{ years}$$

But solar flares release energy in **minutes to hours**! Observed reconnection rates are $M_A \sim 0.01$–$0.1$, about 100,000 times faster than Sweet-Parker predicts.

This is the **reconnection rate problem**: Sweet-Parker reconnection is far too slow to explain observations.

### 2.4 Limitations of Sweet-Parker

The Sweet-Parker model has several limitations:

1. **Too slow**: As shown above, $M_A \sim S^{-1/2}$ gives unrealistically slow rates for large $S$.

2. **Assumes uniform resistivity**: In reality, resistivity can be enhanced by anomalous processes (turbulence, waves).

3. **Steady state**: Real reconnection is often time-dependent and bursty.

4. **2D**: Three-dimensional effects can be important.

5. **Classical resistivity**: Collisionless plasmas require kinetic physics.

Despite these limitations, Sweet-Parker remains a useful benchmark and describes some aspects of reconnection in certain regimes.

## 3. Petschek Reconnection

### 3.1 The Petschek Model

Petschek (1964) proposed a radical modification: instead of a long diffusion region, reconnection occurs in a small resistive region near the X-point, with most of the energy conversion happening in extended slow-mode MHD shocks.

```
                    Inflow
                      ↓
         ╲            ↓            ╱
          ╲           ↓           ╱   Slow shock
           ╲          ↓          ╱
            ╲         ↓         ╱
             ╲        ↓        ╱
              ╲      ┏━┓      ╱
               ╲     ┃ ┃     ╱        Small diffusion
        ════════╲════┃X┃════╱═══════  region (size δ)
                 ╲   ┃ ┃   ╱
                  ╲  ┗━┛  ╱
                   ╲  ↓  ╱
                    ╲ ↓ ╱
                     ╲↓╱
                      ↓ Outflow
```

**Key features:**

1. **Small diffusion region**: Size ~$\delta \sim \eta/(v_A \mu_0)$, independent of $L$
2. **Slow MHD shocks**: Extend from diffusion region to distance ~$L$
3. **Fast reconnection**: Rate depends only logarithmically on $S$

### 3.2 Slow-Mode MHD Shocks

Slow-mode shocks are one of three types of MHD shocks (fast, slow, intermediate). Their properties:

- **Velocity jump**: Flow accelerates across shock
- **Magnetic field**: $B_{\perp}$ decreases, $B_{\parallel}$ can increase or decrease
- **Density**: Increases across shock (compression)
- **Temperature**: Increases (entropy generation)

The slow shocks convert magnetic energy to thermal and kinetic energy. The shocks make an angle $\psi$ with the current sheet:

$$\psi \sim \frac{\delta}{L} \sim \frac{\eta}{L v_A \mu_0}$$

### 3.3 Petschek Reconnection Rate

Petschek's analysis gives a reconnection rate:

$$M_A \sim \frac{\pi}{8 \ln S}$$

For $S = 10^{14}$:

$$M_A \sim \frac{\pi}{8 \ln(10^{14})} \sim \frac{3.14}{8 \times 32} \sim 0.012$$

This is remarkably close to observed rates! The logarithmic dependence on $S$ makes the rate nearly independent of resistivity for large $S$.

**Derivation sketch:**

The reconnection electric field is $E_z = v_{in} B_{in}$. In the diffusion region:

$$E_z = \eta J_z \sim \eta \frac{B_{in}}{\mu_0 \delta}$$

Equating:

$$v_{in} B_{in} \sim \eta \frac{B_{in}}{\mu_0 \delta}$$

$$v_{in} \sim \frac{\eta}{\mu_0 \delta}$$

The diffusion region size is set by local physics:

$$\delta \sim \frac{\eta}{\mu_0 v_A}$$

So:

$$v_{in} \sim \frac{\eta}{\mu_0 \eta/(\mu_0 v_A)} = v_A$$

But this would give $M_A = 1$, which is too fast (causality violation). The constraint comes from matching the slow shock angle to the diffusion region size:

$$\psi \sim \frac{1}{\ln S}$$

The reconnection rate becomes:

$$M_A \sim \frac{1}{\ln S}$$

with a numerical prefactor $\pi/8$ from detailed shock analysis.

### 3.4 Stability of Petschek Reconnection

A major problem was discovered by Biskamp (1986): **Petschek reconnection is unstable for uniform resistivity**.

Numerical simulations showed that with uniform $\eta$, the system evolves to a Sweet-Parker configuration, not a Petschek one. The slow shocks do not form.

**When does Petschek work?**

Petschek reconnection can occur when:

1. **Localized resistivity**: $\eta$ enhanced near the X-point (e.g., by anomalous resistivity)
2. **Time-dependent**: Transient fast reconnection before evolving to Sweet-Parker
3. **Kinetic effects**: Collisionless reconnection (Hall MHD, two-fluid, kinetic)

In resistive MHD with uniform $\eta$, Sweet-Parker is the stable steady state. However, nature rarely provides uniform resistivity, and collisionless effects dominate in many plasmas.

## 4. Hall MHD and Collisionless Reconnection

### 4.1 The Hall Term

In a collisionless plasma, ions and electrons can move independently on scales smaller than the ion inertial length (ion skin depth):

$$d_i = \frac{c}{\omega_{pi}} = \frac{1}{\sqrt{\mu_0 n e^2 / m_i}} \approx 2.28 \times 10^7 \sqrt{\frac{10^6 \text{ cm}^{-3}}{n}} \text{ cm}$$

For typical solar corona parameters ($n \sim 10^{10}$ cm⁻³), $d_i \sim 70$ km, much smaller than the global scale $L \sim 10^4$ km.

On scales smaller than $d_i$, the Hall term in the generalized Ohm's law becomes important:

$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \eta \mathbf{J} + \frac{1}{ne} \mathbf{J} \times \mathbf{B} + \text{other terms}$$

The Hall term is $\mathbf{J} \times \mathbf{B}/(ne)$.

**Physical interpretation:**

- On scales $\gg d_i$: Electrons and ions move together (single-fluid MHD)
- On scales $\sim d_i$: Ions decouple from the magnetic field
- On scales $< d_i$: Electrons carry current and control dynamics

### 4.2 Hall MHD Equations

Hall MHD consists of:

**Continuity:**

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$

**Momentum:**

$$\rho \frac{d\mathbf{v}}{dt} = -\nabla p + \mathbf{J} \times \mathbf{B}$$

**Induction (with Hall term):**

$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) - \nabla \times \left( \frac{1}{ne} \mathbf{J} \times \mathbf{B} \right) + \eta \nabla^2 \mathbf{B}$$

**Ampère's law:**

$$\mathbf{J} = \frac{1}{\mu_0} \nabla \times \mathbf{B}$$

The Hall term can be rewritten as:

$$\nabla \times \left( \frac{1}{ne} \mathbf{J} \times \mathbf{B} \right) = \nabla \times \left( \frac{1}{\mu_0 ne} (\nabla \times \mathbf{B}) \times \mathbf{B} \right)$$

Defining the ion skin depth $d_i = c/\omega_{pi}$, this term scales as $v_A B/d_i$.

### 4.3 Structure of Hall Reconnection

Hall reconnection has a **two-scale structure**:

1. **Outer region** ($r > d_i$): Ion-controlled, obeys standard MHD
   - Ions and electrons frozen to field
   - Scale: $L \sim 100 d_i$ or larger

2. **Inner diffusion region** ($r \sim d_i$): Electron-controlled
   - Ions decoupled, electrons frozen to field
   - Out-of-plane (Hall) magnetic field generated
   - Current carried by electrons

**Quadrupolar Hall field:**

The Hall term generates an out-of-plane (guide field direction) magnetic field with quadrupolar symmetry:

```
        B_z > 0  |  B_z < 0
                 |
      ━━━━━━━━━━X━━━━━━━━━━
                 |
        B_z < 0  |  B_z > 0
```

This quadrupolar $B_z$ structure is a **smoking gun signature** of Hall reconnection, observed in:
- Magnetotail reconnection (Cluster, MMS satellites)
- Laboratory reconnection experiments (MRX, VTF)
- Simulations (GEM Challenge)

### 4.4 Hall Reconnection Rate

The key result: **Hall reconnection gives a fast rate independent of resistivity** for large $S$.

**Scaling arguments:**

In the ion diffusion region, the reconnection electric field is:

$$E_z \sim v_{in} B_{in}$$

In the electron diffusion region (size $\delta_e \sim d_e$, electron skin depth), electron physics dominates. The reconnection rate is set by electron dynamics, giving:

$$v_{in} \sim 0.1 v_A$$

The rate $M_A \sim 0.1$ is observed in many Hall MHD simulations and is independent of $S$ (as long as $S$ is large enough that $d_i < L$).

**Why is it fast?**

The ion decoupling allows a much shorter diffusion region ($\delta \sim d_i$ instead of $\delta \sim L/S^{1/2}$). The aspect ratio is:

$$\frac{\delta}{L} \sim \frac{d_i}{L}$$

not $1/S$, so the reconnection rate becomes independent of resistivity.

### 4.5 GEM Reconnection Challenge

The Geospace Environmental Modeling (GEM) Magnetic Reconnection Challenge (Birn et al. 2001) was a community effort to benchmark reconnection simulations.

**Setup:**

- Harris current sheet equilibrium
- 2D periodic domain
- Various codes: Hall MHD, two-fluid, hybrid, full PIC
- Compare reconnection rate, structure, time evolution

**Key results:**

1. **Fast reconnection**: All codes found $M_A \sim 0.1$, independent of $\eta$ for large $S$
2. **Quadrupolar Hall field**: Confirmed in all Hall/kinetic models
3. **Two-scale structure**: Ion and electron diffusion regions clearly separated
4. **Electron heating**: Temperature anisotropy develops near X-point
5. **Plasmoid formation**: At very large $S$, secondary islands form (see Lesson 7)

The GEM Challenge established that collisionless reconnection is generically fast, resolving the reconnection rate problem for space plasmas.

## 5. X-Point Geometry and Magnetic Topology

### 5.1 X-Point Configuration

An X-point is a **magnetic null** where $\mathbf{B} = 0$. Near an X-point in 2D, the field can be Taylor-expanded:

$$B_x \approx B_0 \frac{x}{L}$$

$$B_y \approx -B_0 \frac{y}{L}$$

where $B_0$ and $L$ are characteristic field strength and length scale.

**Field lines** near the X-point are hyperbolas:

$$xy = \text{const}$$

**Separatrices** are the field lines passing through the X-point (where const = 0):
- $x = 0$: vertical separatrix
- $y = 0$: horizontal separatrix

These separate regions of different magnetic topology.

### 5.2 Magnetic Topology and Topology Change

**Magnetic topology** refers to the connectivity of field lines. In 2D, topology is characterized by:
- Which field lines connect to which boundaries
- Location of X-points (nulls) and O-points (extrema)

**Ideal MHD preserves topology:**

In ideal MHD, the frozen-in theorem guarantees that field line connectivity is conserved. If two plasma elements are on the same field line initially, they remain on the same field line.

**Reconnection changes topology:**

Reconnection allows field lines to break and reconnect, changing connectivity. This is the essence of reconnection.

```
Before reconnection:        After reconnection:

    A ════════════ B            A ════╗
                                      ║
         X (no flow)                  X (reconnection)
                                      ║
    C ════════════ D            C ════╝

    A connects to B             A connects to D
    C connects to D             C connects to B
```

The rate of topology change is measured by the reconnection electric field $E_z$.

### 5.3 Separatrices and Magnetic Islands

**Separatrices** divide regions of different topology. Plasma crossing a separatrix changes field line connection.

**Magnetic islands (O-points):**

When reconnection is not complete, closed field lines can form, creating magnetic islands (plasmoids). An O-point is a local maximum/minimum of the flux function $\psi$.

In a current sheet, multiple X-points and O-points can form:

```
    ────────O────X────O────X────O────
```

This chain structure is important in turbulent or high-$S$ reconnection (plasmoid instability, Lesson 7).

### 5.4 The Vector Potential and Flux Function

In 2D, the magnetic field can be written in terms of a flux function $\psi$:

$$\mathbf{B} = \nabla \psi \times \hat{z}$$

or in components:

$$B_x = \frac{\partial \psi}{\partial y}, \quad B_y = -\frac{\partial \psi}{\partial x}$$

Field lines are contours of $\psi$. The X-point is a saddle point of $\psi$, and O-points are local extrema.

The induction equation becomes an evolution equation for $\psi$:

$$\frac{\partial \psi}{\partial t} = -E_z + \eta J_z$$

where $J_z = -\nabla^2 \psi / \mu_0$.

At an X-point, $\nabla \psi = 0$ (since $\mathbf{B} = 0$), so:

$$\left( \frac{\partial \psi}{\partial t} \right)_{X} = -E_z$$

The reconnection electric field directly measures the rate of change of flux at the X-point.

## 6. Measuring Reconnection Rates

### 6.1 Dimensionless Reconnection Rate

The standard measure is the **Alfvénic Mach number**:

$$M_A = \frac{v_{in}}{v_A}$$

where $v_{in}$ is the inflow velocity into the diffusion region and $v_A = B_{in}/\sqrt{\mu_0 \rho}$ is the Alfvén speed based on the upstream field.

**Typical values:**

- Sweet-Parker: $M_A \sim S^{-1/2} \sim 10^{-7}$ for solar corona
- Petschek: $M_A \sim (\ln S)^{-1} \sim 0.01$
- Hall/collisionless: $M_A \sim 0.1$

### 6.2 Reconnection Electric Field

An equivalent measure is the reconnection electric field $E_{rec}$:

$$E_{rec} = v_{in} B_{in}$$

Normalized by the characteristic electric field $v_A B_{in}$:

$$\tilde{E} = \frac{E_{rec}}{v_A B_{in}} = M_A$$

In steady state, $E_{rec}$ is uniform throughout the reconnection region.

### 6.3 Flux Transfer Rate

The rate of magnetic flux reconnected per unit time (per unit length in the third dimension):

$$\frac{d\Phi}{dt} = E_{rec} \cdot (\text{length in } z)$$

In 2D simulations, this is often plotted as a function of time to diagnose the reconnection phase.

### 6.4 Observational Measures

In observations (e.g., spacecraft data), reconnection rates are inferred from:

1. **Inflow velocity**: Measured by Doppler shifts or particle instruments
2. **Outflow velocity**: Often near $v_A$, confirming Alfvénic reconnection
3. **Hall magnetic field**: Quadrupolar $B_z$ signature (MMS observations)
4. **Energetic particles**: Accelerated particles indicate reconnection

For solar flares, the reconnection rate is estimated from:

$$M_A \sim \frac{v_{up}}{v_A}$$

where $v_{up}$ is the velocity of upward-moving flare ribbons (tracing reconnection footpoints), typically $\sim 10$–100 km/s, giving $M_A \sim 0.01$–$0.1$.

## 7. Python Examples

### 7.1 Sweet-Parker vs Petschek Scaling

```python
import numpy as np
import matplotlib.pyplot as plt

# Lundquist number S = Lv_A/η spans many orders of magnitude: S~10⁶ for
# laboratory plasmas, S~10¹⁴ for the solar corona.  The wide range motivates
# using logspace so that all regimes are equally represented on the log plot.
S = np.logspace(4, 16, 100)

# M_SP = S^(-1/2): Sweet-Parker rate falls steeply with S because the diffusion
# region must extend the full current-sheet length L, making reconnection
# extremely slow at high S.  For S~10¹⁴ this gives M_A~10⁻⁷, orders of
# magnitude below observed rates — the core of the reconnection rate problem.
M_SP = S**(-0.5)

# M_P = π/(8 ln S): Petschek's weak (logarithmic) dependence on S comes from
# the slow-shock geometry where the diffusion region size is fixed at δ~η/v_A
# independent of L, and the slow shocks carry most of the energy conversion.
# At S=10¹⁴, ln(10¹⁴)≈32, giving M_P≈0.012 — close to observed flare rates.
M_P = np.pi / (8 * np.log(S))

# Hall reconnection rate ~0.1 is independent of S once S is large enough that
# d_i ≪ L: the ion decoupling creates a diffusion region of fixed size ~d_i
# regardless of resistivity, so the reconnection rate is set by ion dynamics
# rather than by resistive diffusion — resolving the reconnection rate problem.
M_Hall = 0.1 * np.ones_like(S)

# Plot
plt.figure(figsize=(10, 6))
plt.loglog(S, M_SP, label='Sweet-Parker ($S^{-1/2}$)', linewidth=2)
plt.loglog(S, M_P, label=r'Petschek ($\pi/(8\ln S)$)', linewidth=2)
plt.loglog(S, M_Hall, label='Hall (collisionless)', linewidth=2, linestyle='--')

# Mark typical regimes
plt.axvline(1e8, color='gray', linestyle=':', alpha=0.5)
plt.text(1e8, 0.5, 'Laboratory', rotation=90, va='bottom', ha='right', alpha=0.5)
plt.axvline(1e14, color='gray', linestyle=':', alpha=0.5)
plt.text(1e14, 0.5, 'Solar corona', rotation=90, va='bottom', ha='right', alpha=0.5)

plt.xlabel('Lundquist number $S$', fontsize=14)
plt.ylabel('Reconnection rate $M_A = v_{in}/v_A$', fontsize=14)
plt.title('Reconnection Rate Scaling', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(1e4, 1e16)
plt.ylim(1e-8, 1)
plt.tight_layout()
plt.savefig('reconnection_rate_scaling.png', dpi=150)
plt.show()

# Print example values
S_examples = [1e6, 1e10, 1e14]
print("\nReconnection rates for different regimes:")
print(f"{'S':>10} {'Sweet-Parker':>15} {'Petschek':>15} {'Hall':>15}")
print("-" * 60)
for s in S_examples:
    sp = s**(-0.5)
    p = np.pi / (8 * np.log(s))
    h = 0.1
    print(f"{s:>10.1e} {sp:>15.2e} {p:>15.4f} {h:>15.2f}")
```

### 7.2 X-Point Magnetic Field Structure

```python
import numpy as np
import matplotlib.pyplot as plt

# Create grid
x = np.linspace(-2, 2, 40)
y = np.linspace(-2, 2, 40)
X, Y = np.meshgrid(x, y)

# Bx = X, By = -Y is the simplest X-point magnetic field, obtained by
# Taylor-expanding any 2D null to first order.  The signs ensure ∇·B = 0
# (∂Bx/∂x + ∂By/∂y = 1 - 1 = 0) while creating the saddle-point topology
# that characterizes an X-point: field lines are hyperbolas xy = const.
Bx = X
By = -Y

# B_mag vanishes exactly at the origin (the null point) and grows linearly
# with distance; visualizing it as a color map immediately shows where the
# diffusion region (low-B region) must form during reconnection.
B_mag = np.sqrt(Bx**2 + By**2)

# Create figure with field lines and strength
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Field lines
ax = axes[0]
# Plot field lines using streamplot
ax.streamplot(X, Y, Bx, By, color=B_mag, cmap='viridis',
              linewidth=1.5, density=1.5, arrowsize=1.5)
ax.plot(0, 0, 'rx', markersize=15, markeredgewidth=3, label='X-point')
ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Separatrices')
ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.set_xlabel('$x/L$', fontsize=14)
ax.set_ylabel('$y/L$', fontsize=14)
ax.set_title('X-Point Magnetic Field Lines', fontsize=16)
ax.legend(fontsize=11)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Right panel: Field strength
ax = axes[1]
contour = ax.contourf(X, Y, B_mag, levels=20, cmap='plasma')
ax.contour(X, Y, B_mag, levels=10, colors='black', alpha=0.3, linewidths=0.5)
ax.plot(0, 0, 'wx', markersize=15, markeredgewidth=3)
ax.axhline(0, color='white', linestyle='--', alpha=0.7, linewidth=2)
ax.axvline(0, color='white', linestyle='--', alpha=0.7, linewidth=2)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('$|\\mathbf{B}|/B_0$', fontsize=14)
ax.set_xlabel('$x/L$', fontsize=14)
ax.set_ylabel('$y/L$', fontsize=14)
ax.set_title('Magnetic Field Strength', fontsize=16)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('xpoint_structure.png', dpi=150)
plt.show()

# Plot hyperbolic field lines explicitly
fig, ax = plt.subplots(figsize=(8, 8))
t = np.linspace(-2, 2, 200)

# Field lines are xy = const
constants = [-1.5, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 1.5]
for c in constants:
    if c > 0:
        x_pos = t[t > 0]
        y_pos = c / x_pos
        ax.plot(x_pos, y_pos, 'b-', linewidth=1.5)
        ax.plot(-x_pos, -y_pos, 'b-', linewidth=1.5)
    elif c < 0:
        x_neg = t[t > 0]
        y_neg = c / x_neg
        ax.plot(x_neg, y_neg, 'r-', linewidth=1.5)
        ax.plot(-x_neg, -y_neg, 'r-', linewidth=1.5)

# Separatrices
ax.axhline(0, color='green', linestyle='--', linewidth=2.5, label='Separatrices', alpha=0.7)
ax.axvline(0, color='green', linestyle='--', linewidth=2.5, alpha=0.7)
ax.plot(0, 0, 'ko', markersize=12, label='X-point (null)')

ax.set_xlabel('$x/L$', fontsize=14)
ax.set_ylabel('$y/L$', fontsize=14)
ax.set_title('Hyperbolic Field Lines Near X-Point', fontsize=16)
ax.legend(fontsize=12)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('xpoint_hyperbolic.png', dpi=150)
plt.show()
```

### 7.3 Sweet-Parker Diffusion Region Aspect Ratio

```python
import numpy as np
import matplotlib.pyplot as plt

# Lundquist number
S = np.logspace(2, 14, 100)

# δ/L = S⁻¹ is the Sweet-Parker aspect ratio: the diffusion region width δ
# must be thin enough that resistive diffusion balances the advection of field
# into the layer.  At S=10¹⁴ (solar corona), δ/L ~ 10⁻¹⁴ — an astronomically
# thin sheet, which is physically implausible and motivates collisionless models.
delta_over_L = S**(-1)

# L/δ = S is the inverse aspect ratio: the larger this number, the more
# elongated the current sheet and the slower the reconnection, because
# outflow must travel a longer distance before the field can relax.
L_over_delta = S

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: aspect ratio
ax = axes[0]
ax.loglog(S, delta_over_L, linewidth=2, color='blue')
ax.axhline(0.1, color='red', linestyle='--', label='$\\delta/L = 0.1$', alpha=0.7)
ax.axhline(0.01, color='orange', linestyle='--', label='$\\delta/L = 0.01$', alpha=0.7)
ax.set_xlabel('Lundquist number $S$', fontsize=14)
ax.set_ylabel('Aspect ratio $\\delta/L$', fontsize=14)
ax.set_title('Sweet-Parker Diffusion Region Aspect Ratio', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Right: reconnection rate vs aspect ratio
ax = axes[1]
# M_A = S⁻¹/² = (δ/L)^(1/2): the reconnection rate equals the square root
# of the aspect ratio, showing that a wider (less elongated) diffusion region
# reconnects faster — this is the geometric origin of the Sweet-Parker bottleneck.
M_A = S**(-0.5)
ax.loglog(delta_over_L, M_A, linewidth=2, color='green')
ax.set_xlabel('Aspect ratio $\\delta/L$', fontsize=14)
ax.set_ylabel('Reconnection rate $M_A$', fontsize=14)
ax.set_title('$M_A$ vs $\\delta/L$ (Sweet-Parker)', fontsize=16)
ax.grid(True, alpha=0.3)
# Add scaling annotation
ax.text(1e-6, 1e-2, '$M_A = \\delta/L = S^{-1/2}$', fontsize=13,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('sweet_parker_aspect_ratio.png', dpi=150)
plt.show()

# Print examples
print("\nSweet-Parker diffusion region properties:")
print(f"{'S':>12} {'δ/L':>12} {'L/δ':>12} {'M_A':>12}")
print("-" * 50)
S_vals = [1e4, 1e6, 1e8, 1e10, 1e12, 1e14]
for s in S_vals:
    d_L = s**(-1)
    L_d = s
    M = s**(-0.5)
    print(f"{s:>12.0e} {d_L:>12.2e} {L_d:>12.2e} {M:>12.2e}")
```

### 7.4 Hall Reconnection Quadrupolar Field

```python
import numpy as np
import matplotlib.pyplot as plt

# Grid
x = np.linspace(-3, 3, 60)
y = np.linspace(-2, 2, 40)
X, Y = np.meshgrid(x, y)

# tanh(Y) produces a Harris-sheet-like in-plane field: it transitions smoothly
# from -B₀ (below the sheet) to +B₀ (above), matching the standard current
# sheet equilibrium used in GEM Challenge simulations.
Bx = np.tanh(Y)
# The exp(-Y²) envelope ensures By decays away from the current sheet,
# confining the X-point structure to a localized reconnection region rather
# than extending to infinity.
By = -np.tanh(X / 2) * np.exp(-Y**2)

# The quadrupolar Hall field B_z ∝ X·Y is the most important observational
# signature of Hall reconnection: it arises because the J×B Hall term in
# Ohm's law generates out-of-plane currents with opposite signs in each
# quadrant of the X-point, creating the characteristic four-lobe pattern
# confirmed by Cluster and MMS spacecraft data.
r2 = X**2 + Y**2
Bz = X * Y * np.exp(-r2 / 2)

# J_z = -tanh(Y)/cosh²(Y) is the Harris sheet current profile: it is
# concentrated near y=0 (the current layer) and decays exponentially away,
# reflecting the self-consistent kinetic equilibrium of a thin current sheet.
J_z = -np.tanh(Y) / np.cosh(Y)**2

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: In-plane field
ax = axes[0, 0]
ax.streamplot(X, Y, Bx, By, color='black', linewidth=1, density=1.5, arrowsize=1.2)
ax.plot(0, 0, 'rx', markersize=15, markeredgewidth=3)
ax.set_xlabel('$x/d_i$', fontsize=13)
ax.set_ylabel('$y/d_i$', fontsize=13)
ax.set_title('In-Plane Magnetic Field ($B_x$, $B_y$)', fontsize=14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Panel 2: Out-of-plane Hall field
ax = axes[0, 1]
levels = np.linspace(-0.5, 0.5, 21)
contour = ax.contourf(X, Y, Bz, levels=levels, cmap='RdBu_r', extend='both')
ax.contour(X, Y, Bz, levels=levels[::2], colors='black', alpha=0.3, linewidths=0.5)
ax.plot(0, 0, 'kx', markersize=15, markeredgewidth=3)
ax.axhline(0, color='black', linestyle='--', alpha=0.5)
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('$B_z/B_0$', fontsize=13)
ax.set_xlabel('$x/d_i$', fontsize=13)
ax.set_ylabel('$y/d_i$', fontsize=13)
ax.set_title('Out-of-Plane Hall Field ($B_z$)', fontsize=14)
ax.set_aspect('equal')

# Panel 3: Current density
ax = axes[1, 0]
contour = ax.contourf(X, Y, J_z, levels=20, cmap='coolwarm')
ax.contour(X, Y, J_z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('$J_z/J_0$', fontsize=13)
ax.set_xlabel('$x/d_i$', fontsize=13)
ax.set_ylabel('$y/d_i$', fontsize=13)
ax.set_title('Current Density ($J_z$)', fontsize=14)
ax.set_aspect('equal')

# Panel 4: Line plot of Bz along x-axis
ax = axes[1, 1]
y_cuts = [0.5, 1.0, 1.5]
for y_cut in y_cuts:
    idx = np.argmin(np.abs(y - y_cut))
    ax.plot(x, Bz[idx, :], linewidth=2, label=f'$y/d_i = {y_cut}$')
ax.axhline(0, color='black', linestyle='--', alpha=0.5)
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('$x/d_i$', fontsize=13)
ax.set_ylabel('$B_z/B_0$', fontsize=13)
ax.set_title('Hall Field Profile Along $x$ (Quadrupolar Structure)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hall_reconnection_field.png', dpi=150)
plt.show()

# Schematic of two-scale structure
fig, ax = plt.subplots(figsize=(10, 8))

# Outer ion diffusion region
theta = np.linspace(0, 2*np.pi, 100)
x_ion = 2 * np.cos(theta)
y_ion = 1.5 * np.sin(theta)
ax.fill(x_ion, y_ion, color='lightblue', alpha=0.5, label='Ion diffusion region ($\\sim d_i$)')
ax.plot(x_ion, y_ion, 'b-', linewidth=2)

# Inner electron diffusion region
x_elec = 0.5 * np.cos(theta)
y_elec = 0.3 * np.sin(theta)
ax.fill(x_elec, y_elec, color='salmon', alpha=0.5, label='Electron diffusion region ($\\sim d_e$)')
ax.plot(x_elec, y_elec, 'r-', linewidth=2)

# X-point
ax.plot(0, 0, 'kx', markersize=20, markeredgewidth=4)

# Separatrices
ax.plot([-3, 3], [0, 0], 'k--', linewidth=2, alpha=0.7)
ax.plot([0, 0], [-2.5, 2.5], 'k--', linewidth=2, alpha=0.7)

# Inflow/outflow arrows
ax.annotate('', xy=(0, -1.5), xytext=(0, -2.5),
            arrowprops=dict(arrowstyle='->', lw=3, color='green'))
ax.text(0.2, -2, 'Inflow', fontsize=13, color='green')

ax.annotate('', xy=(2.5, 0), xytext=(1.5, 0),
            arrowprops=dict(arrowstyle='->', lw=3, color='orange'))
ax.text(2, 0.2, 'Outflow', fontsize=13, color='orange')

ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$y$', fontsize=14)
ax.set_title('Two-Scale Structure of Hall Reconnection', fontsize=16)
ax.legend(fontsize=12, loc='upper right')
ax.set_xlim(-3, 3)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hall_two_scale_structure.png', dpi=150)
plt.show()
```

### 7.5 Reconnection Rate Evolution

```python
import numpy as np
import matplotlib.pyplot as plt

# Time array
t = np.linspace(0, 100, 500)

# Sweet-Parker approach to steady state (slow)
M_SP = 0.001 * (1 - np.exp(-t / 30))

# Petschek burst (faster)
M_P = 0.02 * np.exp(-((t - 20) / 10)**2) * (t > 10)

# Hall reconnection (fast, sustained)
M_Hall = 0.1 * (1 - np.exp(-t / 5)) * (t > 15)

# Combined example: onset, burst, quasi-steady
M_combined = M_SP.copy()
M_combined += M_P
M_combined += M_Hall * 0.5

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Panel 1: Individual models
ax = axes[0]
ax.plot(t, M_SP, label='Sweet-Parker (slow)', linewidth=2)
ax.plot(t, M_P, label='Petschek burst', linewidth=2)
ax.plot(t, M_Hall, label='Hall (fast)', linewidth=2)
ax.set_xlabel('Time ($t v_A / L$)', fontsize=13)
ax.set_ylabel('Reconnection rate $M_A$', fontsize=13)
ax.set_title('Idealized Reconnection Rate Profiles', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.15)

# Panel 2: Combined realistic scenario
ax = axes[1]
ax.plot(t, M_combined, linewidth=2.5, color='darkblue')
ax.axhline(0.1, color='red', linestyle='--', alpha=0.6, label='Typical Hall rate (~0.1)')
ax.axhline(0.01, color='orange', linestyle='--', alpha=0.6, label='Typical Petschek rate (~0.01)')
ax.fill_between(t, 0, M_combined, alpha=0.3, color='skyblue')

# Annotate phases
ax.annotate('Onset', xy=(10, 0.005), fontsize=12,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax.annotate('Burst', xy=(25, 0.08), fontsize=12,
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
ax.annotate('Quasi-steady', xy=(60, 0.06), fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax.set_xlabel('Time ($t v_A / L$)', fontsize=13)
ax.set_ylabel('Reconnection rate $M_A$', fontsize=13)
ax.set_title('Realistic Time-Dependent Reconnection', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.12)

plt.tight_layout()
plt.savefig('reconnection_rate_evolution.png', dpi=150)
plt.show()
```

## Summary

Magnetic reconnection is a fundamental process that changes magnetic topology and converts magnetic energy to plasma energy. We covered:

1. **Sweet-Parker model**: Steady-state reconnection in a long, thin current sheet. The rate $M_A \sim S^{-1/2}$ is far too slow for astrophysical applications when $S \sim 10^{14}$.

2. **Petschek model**: Reconnection with slow MHD shocks, giving a rate $M_A \sim (\ln S)^{-1} \sim 0.01$. However, this requires localized resistivity and is unstable with uniform $\eta$.

3. **Hall MHD reconnection**: In collisionless plasmas, ions decouple on scales $\sim d_i$ (ion skin depth), leading to a two-scale structure. The reconnection rate is fast ($M_A \sim 0.1$) and independent of resistivity. The quadrupolar Hall magnetic field is a key observational signature.

4. **X-point geometry**: Magnetic nulls where $\mathbf{B} = 0$, with hyperbolic field line structure. Separatrices divide regions of different topology. Reconnection changes field line connectivity.

5. **Reconnection rate measures**: The dimensionless rate $M_A = v_{in}/v_A$ is the standard measure. Observed rates in solar flares and magnetospheric substorms are $\sim 0.01$–$0.1$, consistent with Petschek and Hall reconnection.

The resolution of the reconnection rate problem came from recognizing that collisionless effects (Hall physics, kinetic effects) dominate in most natural plasmas, enabling fast reconnection independent of classical resistivity. The GEM Challenge confirmed that kinetic reconnection generically gives $M_A \sim 0.1$.

## Practice Problems

1. **Sweet-Parker scaling**:
   a) Derive the Sweet-Parker reconnection rate starting from mass conservation, momentum balance, and Ohm's law.
   b) For Earth's magnetotail ($L = 10^7$ m, $B = 20$ nT, $n = 10^6$ m⁻³, $\eta = 10^{-2}$ Ω·m), calculate $S$ and $M_A$.
   c) Estimate the reconnection time scale. Is this consistent with observed substorm onset times (~1 hour)?

2. **Petschek vs Sweet-Parker**:
   a) At what Lundquist number does the Petschek rate exceed the Sweet-Parker rate?
   b) Plot the ratio $M_P / M_{SP}$ as a function of $S$ for $S = 10^2$ to $10^{16}$.
   c) Explain physically why Petschek is faster.

3. **Hall reconnection**:
   a) For solar corona parameters ($n = 10^{16}$ m⁻³), calculate the ion skin depth $d_i$.
   b) If the global scale is $L = 10^9$ m, what is the scale separation $L/d_i$?
   c) Sketch the two-scale structure (ion and electron diffusion regions).

4. **X-point field**:
   a) For the field $\mathbf{B} = B_0 (x \hat{x} - y \hat{y})/L$, find the field lines (contours of $\psi$).
   b) Calculate the magnetic field strength $|\mathbf{B}|$ as a function of position.
   c) Where is $|\mathbf{B}|$ maximum? Minimum?

5. **Reconnection electric field**:
   a) If $v_{in} = 0.1 v_A$ and $B_{in} = 0.01$ T, what is the reconnection electric field $E_{rec}$?
   b) If the Alfvén speed is $v_A = 10^6$ m/s, calculate $E_{rec}$ in V/m.
   c) How much magnetic flux reconnects per second across a length of 1000 km in the $z$-direction?

6. **Quadrupolar Hall field**:
   a) Explain physically why the Hall term $\mathbf{J} \times \mathbf{B}/(ne)$ generates an out-of-plane field.
   b) Sketch the quadrupolar $B_z$ structure for an X-point in the $xy$-plane.
   c) How would spacecraft traversing the diffusion region observe this field?

7. **Simulation analysis**:
   a) In a 2D MHD simulation, you measure $v_{in} = 0.05 v_A$ during the reconnection phase. What is $M_A$?
   b) If the simulation has $\eta = 10^{-4}$ (code units), $L = 10$, $v_A = 1$, calculate $S$.
   c) Is this reconnection rate consistent with Sweet-Parker, Petschek, or Hall reconnection?

8. **Energy conversion**:
   a) The magnetic energy inflow rate per unit volume is $\sim v_{in} B^2/(2\mu_0)$. If $M_A = 0.1$, express this in terms of $v_A$ and $B$.
   b) Compare the magnetic energy inflow rate to the kinetic energy outflow rate $\sim \rho v_{out}^3 / 2$.
   c) Where does the "missing" energy go?

9. **GEM Challenge**:
   a) Research the GEM Reconnection Challenge setup (Harris sheet, perturbation, boundary conditions).
   b) What were the key findings regarding reconnection rate from different codes?
   c) How did resistive MHD results differ from Hall MHD and kinetic results?

10. **Observational signatures**:
    a) List three observational signatures of magnetic reconnection in the magnetotail.
    b) How does the MMS (Magnetospheric Multiscale) mission measure the Hall fields?
    c) What would you expect to observe if a spacecraft crossed the ion diffusion region but not the electron diffusion region?

## Navigation

Previous: [Current-Driven Instabilities](./04_Current_Driven_Instabilities.md) | Next: [Reconnection Applications](./06_Reconnection_Applications.md)
