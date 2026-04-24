# Lesson 2: Mathematical Modeling of Physical Systems

## Learning Objectives

- Derive differential equation models for mechanical, electrical, and electromechanical systems
- Identify analogies between physical domains
- Linearize nonlinear models around equilibrium points
- Convert physical models to standard forms suitable for control analysis
- Validate models numerically and recognize when a model has been linearized too aggressively

## 0. The Modeler's Job

Every controller in this course assumes a model. The model is never the system — it is a deliberately simplified mathematical surrogate that captures the dynamics that matter at the time scale and operating range of interest. Picking what to keep and what to discard is the modeling decision, and it has more impact on closed-loop performance than the controller choice in most projects.

Three guiding principles you will reach for repeatedly:

- **Keep the dominant physics.** A car's lateral dynamics are dominated by tire slip, mass, and yaw inertia. The fact that the windshield is curved is irrelevant. The fact that the front and rear axles are different is essential. Modeling fidelity grows by adding the next-most-important physics, not by adding everything.
- **Linearize at the operating point.** Nonlinear physics ($\sin\theta$, $v^2$ drag, hysteresis) makes design hard. Almost every control design reduces to "linearize, design, simulate the original nonlinear system, iterate." Section 6 below is the workhorse procedure.
- **Honor the analogy across domains.** A mass on a spring and an LC tank circuit obey identical equations. Recognizing this saves you weeks of derivation when you switch fields. Section 5 is the cheat sheet.

The output of this lesson — a linear ODE for your plant — feeds Lesson 3's transfer function, which feeds every analysis lesson after that. Get this right and the rest of the course is bookkeeping.

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

> **Why this saves so much time**: PID tuning rules developed for chemical-reactor temperature loops in the 1940s transfer directly to motor speed control today, because both reduce to the same canonical second-order plant. The wisdom from one industry is reusable in another only because the math is identical.

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

### 6.4 How Far Does the Linearization Hold?

A useful rule of thumb: linearization is "good enough" while the nonlinear term's relative error is below ~10%. For the pendulum, $\sin\theta = \theta - \theta^3/6 + \cdots$. The relative error is $\theta^2/6$, so a 10% threshold gives $\theta \leq \sqrt{0.6} \approx 0.77$ rad ≈ 44°. Beyond that, the linear model lies and the controller may misbehave.

For a centrifugal speed regulator with $\omega^2$ nonlinearity around $\bar{\omega}$, the local slope is $2\bar{\omega}$ and the second-order term grows quadratically with the perturbation — the same 10% rule says you can perturb by up to ~22% of $\bar{\omega}$ before the linear model fails.

In practice: design with the linear model, simulate with the nonlinear one, plot the divergence. If the simulated response stays within 10% of the linear prediction across your operating envelope, you are safe.

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

### 7.1 Building the Same Model in Python

`sympy` derives the transfer function automatically — useful for catching algebra mistakes:

```python
import sympy as sp

s, m, b, k = sp.symbols('s m b k', positive=True)
X, F = sp.symbols('X F')

# Equation: m s^2 X + b s X + k X = F
eq = sp.Eq(m * s**2 * X + b * s * X + k * X, F)
G = sp.simplify(sp.solve(eq, X)[0] / F)
print("G(s) =", G)         # 1 / (m s^2 + b s + k)

# Plug in numerical values: m=1, b=2, k=10
G_num = G.subs({m: 1, b: 2, k: 10})
print("Numeric G(s) =", G_num)

# Standard form: omega_n, zeta
omega_n = sp.sqrt(k / m)
zeta    = b / (2 * sp.sqrt(m * k))
print("omega_n =", omega_n.subs({m: 1, k: 10}).evalf())
print("zeta    =", zeta.subs({m: 1, b: 2, k: 10}).evalf())
```

The sympy expression `G = 1/(m*s**2 + b*s + k)` matches the hand derivation. For more complex multi-element circuits or motors, this becomes invaluable — sympy will simplify $C(sI-A)^{-1}B$ symbolically without arithmetic mistakes.

## 8. Common Pitfalls

1. **Modeling too much.** Adding the next physical effect always tightens the model — and triples the design complexity. Stop adding effects when their contribution is below the dominant ones by 10× at the relevant frequencies.
2. **Linearizing at the wrong operating point.** A pendulum linearized at $\theta = 0$ behaves nothing like one linearized at $\theta = \pi$. The signs of physical parameters can FLIP. Always identify the operating point explicitly and verify the equilibrium condition $f(\bar x, \bar u) = 0$.
3. **Mixing units silently.** SI is mandatory for back-EMF / torque-constant identities ($K_t = K_b$). Mix units (e.g., RPM with rad/s) and the equations no longer balance. Convert everything to SI before deriving.
4. **Ignoring sensor and actuator dynamics.** Modeling the plant as if you have direct access to its state is fine for a textbook. In real hardware, the sensor has its own time constant (thermocouples are slow, encoders are fast) and the actuator has its own bandwidth (a hydraulic ram is slow, a piezo is fast). Both belong in the model.
5. **Treating the linearized model as exact.** Within the operating envelope, the linear model is a useful tool. Outside, it is wrong. Always run the nonlinear simulation as the final check before committing the design to hardware.
6. **Forgetting to honor the analogy.** Engineers often re-derive the same canonical second-order system in five different domains over their career. Recognize the form once and you skip 80% of the algebra forever.

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

### Exercise 4: Sympy Drill

Use the Python snippet from Section 7.1 to derive the transfer function $\Theta(s)/V_a(s)$ for the DC motor (Section 4.1). Plug in plausible numerical values ($J = 0.01, B = 0.1, K_t = K_b = 0.05, R_a = 1$) and find the closed-loop poles. Verify they are in the LHP.

### Exercise 5: Linearization Range

For the pendulum equation $\ddot\theta + (g/l)\sin\theta = u$, simulate the nonlinear ODE with initial conditions $\theta_0 \in \{0.1, 0.5, 1.0\}$ rad and zero input. Plot each trajectory against the linear model's prediction $\theta(t) = \theta_0 \cos(\sqrt{g/l}\,t)$. At which $\theta_0$ does the linear prediction diverge from truth by more than 10%? Compare to the rule-of-thumb estimate from Section 6.4.

---

*Previous: [Lesson 1 — Introduction to Control Systems](01_Introduction_to_Control_Systems.md) | Next: [Lesson 3 — Transfer Functions and Block Diagrams](03_Transfer_Functions_and_Block_Diagrams.md)*
