# Lesson 2: Mathematical Modeling of Physical Systems

## Learning Objectives

- Derive differential equation models for mechanical, electrical, and electromechanical systems
- Identify analogies between physical domains
- Linearize nonlinear models around equilibrium points
- Convert physical models to standard forms suitable for control analysis

## 1. Why Mathematical Modeling?

Control design requires a **mathematical model** — a set of equations that describes the dynamic behavior of the plant. The model captures the essential physics while being simple enough for analysis and controller synthesis.

**Modeling approaches:**
- **First principles**: Derive from physics (Newton's laws, Kirchhoff's laws, conservation laws)
- **System identification**: Fit models to measured input-output data (covered in advanced courses)
- **Hybrid**: Combine physics-based structure with data-fitted parameters

We focus on first-principles modeling in this lesson.

## 2. Mechanical Systems

### 2.1 Translational Systems

Three fundamental elements:

| Element | Law | Equation |
|---------|-----|----------|
| Mass $m$ | Newton's 2nd law | $F = m\ddot{x}$ |
| Damper $b$ | Viscous friction | $F = b\dot{x}$ |
| Spring $k$ | Hooke's law | $F = kx$ |

**Example: Mass-Spring-Damper**

```
    F(t)
    →  ┌───┐
   ────┤ m ├────┬──── wall
       └───┘    │
          ├──┤b├──┤  (damper)
          ├──/\/\──┤  (spring, k)
```

Applying Newton's second law:

$$m\ddot{x}(t) + b\dot{x}(t) + kx(t) = F(t)$$

This is a **second-order linear ODE** with constant coefficients.

**Standard form** (dividing by $m$):

$$\ddot{x} + 2\zeta\omega_n \dot{x} + \omega_n^2 x = \frac{F(t)}{m}$$

where $\omega_n = \sqrt{k/m}$ is the **natural frequency** and $\zeta = b/(2\sqrt{mk})$ is the **damping ratio**.

### 2.2 Rotational Systems

| Element | Law | Equation |
|---------|-----|----------|
| Moment of inertia $J$ | Newton's rotational law | $\tau = J\ddot{\theta}$ |
| Rotational damper $B$ | Viscous friction | $\tau = B\dot{\theta}$ |
| Torsional spring $K$ | Hooke's law (rotational) | $\tau = K\theta$ |

**Example: Simple Pendulum (small angle)**

For a pendulum of length $l$ and mass $m$:

$$ml^2 \ddot{\theta} + mgl\sin\theta = \tau(t)$$

Linearizing for small angles ($\sin\theta \approx \theta$):

$$ml^2 \ddot{\theta} + mgl\theta = \tau(t)$$

### 2.3 Gear Trains

A gear train with gear ratio $N = N_2/N_1$ (teeth ratio) transforms torque and speed:

$$\theta_2 = \frac{N_1}{N_2}\theta_1, \quad \tau_2 = \frac{N_2}{N_1}\tau_1$$

The reflected inertia seen at the input shaft:

$$J_{\text{eff}} = J_1 + \left(\frac{N_1}{N_2}\right)^2 J_2$$

## 3. Electrical Systems

### 3.1 Passive Elements

| Element | Voltage-Current | Impedance $Z(s)$ |
|---------|----------------|-------------------|
| Resistor $R$ | $v = Ri$ | $R$ |
| Inductor $L$ | $v = L\frac{di}{dt}$ | $Ls$ |
| Capacitor $C$ | $v = \frac{1}{C}\int i \, dt$ | $\frac{1}{Cs}$ |

### 3.2 Kirchhoff's Laws

- **KVL** (Kirchhoff's Voltage Law): Sum of voltages around a loop = 0
- **KCL** (Kirchhoff's Current Law): Sum of currents at a node = 0

**Example: Series RLC Circuit**

Applying KVL:

$$L\frac{di}{dt} + Ri + \frac{1}{C}\int i \, dt = v_{\text{in}}(t)$$

Differentiating and substituting $v_C = \frac{1}{C}\int i \, dt$:

$$LC\ddot{v}_C + RC\dot{v}_C + v_C = v_{\text{in}}(t)$$

This has the **same form** as the mass-spring-damper system!

### 3.3 Op-Amp Circuits

Ideal op-amp assumptions: infinite input impedance, zero output impedance, infinite gain.

**Inverting amplifier:**

$$v_{\text{out}} = -\frac{R_f}{R_{\text{in}}} v_{\text{in}}$$

**Integrator:**

$$v_{\text{out}} = -\frac{1}{R_{\text{in}}C_f} \int v_{\text{in}} \, dt$$

**Differentiator:**

$$v_{\text{out}} = -R_f C_{\text{in}} \frac{dv_{\text{in}}}{dt}$$

Op-amp circuits are commonly used to implement analog controllers (PID).

## 4. Electromechanical Systems

### 4.1 DC Motor

A DC motor converts electrical energy to mechanical energy. The key equations are:

**Electrical (armature circuit):**

$$L_a \frac{di_a}{dt} + R_a i_a + K_b \dot{\theta} = v_a(t)$$

**Mechanical (rotor):**

$$J\ddot{\theta} + B\dot{\theta} = K_t i_a$$

where:
- $v_a$: applied voltage
- $i_a$: armature current
- $K_b$: back-EMF constant
- $K_t$: torque constant ($K_t = K_b$ in SI units)
- $R_a, L_a$: armature resistance and inductance

**Transfer function** (from $V_a(s)$ to $\Theta(s)$, assuming $L_a \approx 0$):

$$\frac{\Theta(s)}{V_a(s)} = \frac{K_t}{s(JR_a s + BR_a + K_t K_b)}$$

The DC motor is one of the most important plants in control engineering — used in robotics, disk drives, printers, and countless other applications.

## 5. Analogies Between Physical Domains

The mathematical structure of physical systems is often identical across domains:

| Mechanical (Trans.) | Mechanical (Rot.) | Electrical | Fluid | Thermal |
|---------------------|-------------------|------------|-------|---------|
| Force $F$ | Torque $\tau$ | Voltage $v$ | Pressure $P$ | Temp. $T$ |
| Velocity $\dot{x}$ | Angular vel. $\dot{\theta}$ | Current $i$ | Flow rate $Q$ | Heat flow $q$ |
| Mass $m$ | Inertia $J$ | Inductance $L$ | Inertance | — |
| Damper $b$ | Rot. damper $B$ | Resistance $R$ | Fluid resistance | Thermal resist. |
| Spring $k$ | Torsion spring $K$ | Elastance $1/C$ | Fluid capacitance | Thermal cap. |
| Displacement $x$ | Angle $\theta$ | Charge $q$ | Volume | Heat $Q$ |

These analogies allow techniques developed for one domain to be applied directly to another.

## 6. Linearization

Most real systems are **nonlinear**. Linear control theory applies only after **linearization** around an **operating point** (equilibrium).

### 6.1 Equilibrium Point

An equilibrium $(\bar{x}, \bar{u})$ satisfies $\dot{x} = f(\bar{x}, \bar{u}) = 0$.

### 6.2 Taylor Series Linearization

Given a nonlinear system $\dot{x} = f(x, u)$, define perturbation variables:

$$\delta x = x - \bar{x}, \quad \delta u = u - \bar{u}$$

Expanding $f$ in a Taylor series and keeping first-order terms:

$$\delta\dot{x} \approx \frac{\partial f}{\partial x}\bigg|_{(\bar{x},\bar{u})} \delta x + \frac{\partial f}{\partial u}\bigg|_{(\bar{x},\bar{u})} \delta u$$

This yields a **linear approximation** valid near the equilibrium.

### 6.3 Example: Nonlinear Pendulum

The full nonlinear equation:

$$\ddot{\theta} + \frac{g}{l}\sin\theta = \frac{\tau}{ml^2}$$

**Equilibrium:** $\bar{\theta} = 0$, $\bar{\tau} = 0$

**Linearization:** $\sin\theta \approx \theta$ for small $\theta$:

$$\ddot{\theta} + \frac{g}{l}\theta = \frac{\tau}{ml^2}$$

**Equilibrium:** $\bar{\theta} = \pi$ (inverted pendulum), $\bar{\tau} = 0$

Setting $\delta\theta = \theta - \pi$: $\sin(\pi + \delta\theta) = -\sin(\delta\theta) \approx -\delta\theta$:

$$\delta\ddot{\theta} - \frac{g}{l}\delta\theta = \frac{\tau}{ml^2}$$

Note the sign change — the inverted equilibrium is **unstable** (positive coefficient on $\delta\theta$), while the hanging equilibrium is **stable** (negative coefficient).

## 7. From Physical Model to Transfer Function

The standard workflow for obtaining a transfer function:

1. **Identify** physical elements and their interconnections
2. **Apply** physical laws (Newton's, Kirchhoff's, etc.)
3. **Derive** the differential equation(s)
4. **Linearize** if necessary
5. **Take the Laplace transform** (assume zero initial conditions)
6. **Solve** for $G(s) = Y(s)/U(s)$

**Example:** For the mass-spring-damper $m\ddot{x} + b\dot{x} + kx = F$:

$$ms^2 X(s) + bsX(s) + kX(s) = F(s)$$

$$G(s) = \frac{X(s)}{F(s)} = \frac{1}{ms^2 + bs + k}$$

## Practice Exercises

### Exercise 1: Mechanical Modeling

A two-mass system has masses $m_1$ and $m_2$ connected by a spring $k_{12}$ and damper $b_{12}$. Mass $m_1$ is connected to a wall by spring $k_1$ and damper $b_1$. Force $F(t)$ is applied to $m_2$.

1. Draw the free-body diagram for each mass
2. Write the coupled differential equations
3. Find the transfer function $X_2(s)/F(s)$

### Exercise 2: Electrical-Mechanical Analogy

For the series RLC circuit with input voltage $v_{\text{in}}$ and output $v_C$:

1. Write the differential equation
2. Identify the mechanical analogy (which mechanical element corresponds to each electrical element?)
3. Find the transfer function $V_C(s)/V_{\text{in}}(s)$ and identify $\omega_n$ and $\zeta$

### Exercise 3: Linearization

A tank system has nonlinear dynamics:

$$A\dot{h} = q_{\text{in}} - c\sqrt{h}$$

where $h$ is the water level, $A$ is the cross-sectional area, $q_{\text{in}}$ is the input flow rate, and $c$ is a valve coefficient.

1. Find the equilibrium $\bar{h}$ as a function of a constant input $\bar{q}_{\text{in}}$
2. Linearize around this equilibrium
3. Find the transfer function from $\delta q_{\text{in}}$ to $\delta h$

---

*Previous: [Lesson 1 — Introduction to Control Systems](01_Introduction_to_Control_Systems.md) | Next: [Lesson 3 — Transfer Functions and Block Diagrams](03_Transfer_Functions_and_Block_Diagrams.md)*
