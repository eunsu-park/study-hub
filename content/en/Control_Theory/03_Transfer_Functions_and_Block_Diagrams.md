# Lesson 3: Transfer Functions and Block Diagrams

## Learning Objectives

- Define and compute transfer functions for LTI systems
- Identify poles, zeros, and their effect on system behavior
- Manipulate block diagrams using algebra rules
- Reduce complex block diagrams to a single transfer function
- Apply Mason's gain formula to signal flow graphs
- Build transfer-function objects in Python and recognize the algebraic gotchas that hand reduction is prone to

## 0. Why the Transfer Function Earns Its Own Lesson

The transfer function is the most economical representation of an LTI system: a ratio of polynomials in $s$. Three reasons every classical-control technique is built on it:

- **Composition is multiplication.** Cascading two plants $G_1$ and $G_2$ in series produces $G_1 G_2$ — a polynomial multiplication, not a convolution. This single property is why frequency-domain methods scale to circuits with hundreds of components.
- **Stability is a glance at pole locations.** The poles of $G(s)$ are the roots of its denominator. Plot them on the $s$-plane; LHP poles mean stable, RHP poles mean unstable, on the imaginary axis means marginally stable. No simulation required.
- **Block diagrams compose visually.** A complex system reduces to a single $G(s)$ via four algebraic rules. The block diagram is a sketch of the math; the algebra is the simplification you do on paper.

The transfer function trades one thing for all this: it loses internal state. A system with hidden modes (uncontrollable or unobservable, see Lesson 12) can be misrepresented by its transfer function. For most classical control work, that trade is acceptable.

Mental picture: the transfer function is a contract — given any input frequency, here is the gain and phase the plant produces. Everything in Lessons 4-10 is about reading that contract intelligently.

## 1. Transfer Function Definition

The **transfer function** of an LTI system is the ratio of the Laplace transform of the output to the Laplace transform of the input, assuming zero initial conditions:

$$G(s) = \frac{Y(s)}{U(s)} = \frac{b_m s^m + b_{m-1}s^{m-1} + \cdots + b_0}{a_n s^n + a_{n-1}s^{n-1} + \cdots + a_0}$$

**Key properties:**
- Defined only for LTI systems
- Independent of the input signal
- Contains all information about the input-output behavior (but not internal dynamics)
- The system is **proper** if $m \leq n$ (degree of numerator $\leq$ degree of denominator)
- The system is **strictly proper** if $m < n$

## 2. Poles and Zeros

### 2.1 Definitions

Factoring the transfer function:

$$G(s) = K \frac{(s - z_1)(s - z_2)\cdots(s - z_m)}{(s - p_1)(s - p_2)\cdots(s - p_n)}$$

- **Zeros** $z_1, \ldots, z_m$: roots of the numerator — values of $s$ where $G(s) = 0$
- **Poles** $p_1, \ldots, p_n$: roots of the denominator — values of $s$ where $G(s) \to \infty$
- **Gain** $K$: ratio of leading coefficients

### 2.2 Pole Locations and Time-Domain Behavior

The poles determine the natural response of the system:

| Pole Location | Time Response | Stability |
|--------------|---------------|-----------|
| Real, negative ($s = -a$) | $e^{-at}$ (decaying exponential) | Stable |
| Real, positive ($s = +a$) | $e^{at}$ (growing exponential) | Unstable |
| Complex pair $s = -\sigma \pm j\omega$ | $e^{-\sigma t}\sin(\omega t + \phi)$ (damped oscillation) | Stable if $\sigma > 0$ |
| Imaginary pair $s = \pm j\omega$ | $\sin(\omega t + \phi)$ (sustained oscillation) | Marginally stable |
| $s = 0$ | Constant (integrator) | Marginally stable |

**The dominant poles** (closest to the imaginary axis) primarily determine the transient response.

### 2.3 Effect of Zeros

Zeros do not affect stability but influence the transient response:
- **Left half-plane zeros**: Speed up the response
- **Right half-plane (RHP) zeros**: Cause initial inverse response (undershoot before tracking), limit achievable bandwidth
- **Zeros near poles**: Approximately cancel the pole's contribution

> **The RHP-zero penalty**: any plant with an RHP zero has a hard limit on closed-loop bandwidth — roughly half the zero's magnitude. Aircraft pitch dynamics, voltage-step-up converters, and the cart position of an inverted pendulum all share this limitation. No clever control fixes it; you can only design around it.

## 3. Standard Transfer Function Forms

### 3.1 First-Order System

$$G(s) = \frac{K}{\tau s + 1}$$

- **DC gain**: $K = G(0)$
- **Time constant**: $\tau$ (time to reach 63.2% of final value)
- Single real pole at $s = -1/\tau$

### 3.2 Second-Order System

$$G(s) = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}$$

- **Natural frequency**: $\omega_n$
- **Damping ratio**: $\zeta$
- Poles: $s = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}$

| Damping | $\zeta$ Range | Pole Type | Response |
|---------|--------------|-----------|----------|
| Overdamped | $\zeta > 1$ | Two real negative | Slow, no oscillation |
| Critically damped | $\zeta = 1$ | Repeated real | Fastest without oscillation |
| Underdamped | $0 < \zeta < 1$ | Complex conjugate pair | Oscillatory, decaying |
| Undamped | $\zeta = 0$ | Pure imaginary | Sustained oscillation |

### 3.3 Higher-Order Systems

Higher-order systems can often be approximated by their **dominant poles** — the poles closest to the imaginary axis dominate the transient response if other poles are at least 5 times farther from the imaginary axis.

## 4. Block Diagram Algebra

### 4.1 Basic Connections

**Series (cascade):**

```
U → [G₁] → [G₂] → Y
```

$$G(s) = G_1(s) G_2(s)$$

**Parallel:**

```
    ┌→ [G₁] →┐
U → ┤         ├→(+)→ Y
    └→ [G₂] →┘
```

$$G(s) = G_1(s) + G_2(s)$$

**Negative feedback:**

```
R →(+)→ [G] → Y
    ↑          |
    └── [H] ←──┘
```

$$\frac{Y(s)}{R(s)} = \frac{G(s)}{1 + G(s)H(s)}$$

**Positive feedback:**

$$\frac{Y(s)}{R(s)} = \frac{G(s)}{1 - G(s)H(s)}$$

### 4.2 Block Diagram Reduction Rules

| Operation | Before | After |
|-----------|--------|-------|
| Move summing point ahead of block | $G(E_1 + E_2)$ | $GE_1 + GE_2$ |
| Move summing point past block | $GE_1 + E_2$ | $G(E_1 + E_2/G)$ |
| Move takeoff point ahead of block | Takeoff after $G$ | Takeoff before $G$, insert $G$ in branch |
| Move takeoff point past block | Takeoff before $G$ | Takeoff after $G$, insert $1/G$ in branch |
| Swap summing points | $(\pm A \pm B) \pm C$ | $(\pm A \pm C) \pm B$ |

### 4.3 Reduction Example

Consider the system:

```
R →(+)→ [G₁] →(+)→ [G₂] → Y
    ↑              ↑        |
    |    D(s) ─────┘        |
    └────── [H] ←───────────┘
```

**Step 1:** Inner loop — combine the disturbance path:
$$Y(s) = G_2(s)[G_1(s)E(s) + D(s)]$$

**Step 2:** Close the outer loop with $E(s) = R(s) - H(s)Y(s)$:

$$Y(s) = \frac{G_1(s)G_2(s)}{1 + G_1(s)G_2(s)H(s)} R(s) + \frac{G_2(s)}{1 + G_1(s)G_2(s)H(s)} D(s)$$

This shows how feedback attenuates the disturbance by the factor $\frac{1}{1 + G_1 G_2 H}$.

### 4.4 Block Reduction in Python

`python-control` does the algebra for you, with no chance of a hand error:

```python
from control import TransferFunction, feedback, series, parallel

G1 = TransferFunction([1], [1, 1])      # 1 / (s + 1)
G2 = TransferFunction([2], [1, 3])      # 2 / (s + 3)
H  = TransferFunction([1], [1, 5])      # 1 / (s + 5)

forward = series(G1, G2)                # G1 * G2
T = feedback(forward, H)                # forward / (1 + forward * H)
print("Closed loop:", T)
print("Poles:", T.pole())
print("Zeros:", T.zero())
```

For diagrams with branches and disturbance inputs, build each path separately (forward path, disturbance-to-output path) and add them with `parallel`. The result of any reduction is a single `TransferFunction` — exactly what every analysis routine in Lessons 4-10 consumes.

## 5. Closed-Loop Transfer Functions

For a standard feedback system with forward path $G(s)$ and feedback path $H(s)$:

### 5.1 Key Transfer Functions

| Transfer Function | Name | Formula |
|-------------------|------|---------|
| $T(s) = \frac{Y}{R}$ | Closed-loop (complementary sensitivity) | $\frac{GH_{\text{ff}}}{1+GH}$ or $\frac{G}{1+GH}$ for unity feedback |
| $S(s) = \frac{E}{R}$ | Sensitivity | $\frac{1}{1+GH}$ |
| $\frac{Y}{D}$ | Disturbance-to-output | $\frac{G_2}{1+G_1 G_2 H}$ |

### 5.2 Fundamental Constraint

$$S(s) + T(s) = 1$$

This means sensitivity and complementary sensitivity **always trade off**: you cannot make both small simultaneously. This is one of the most fundamental limitations in feedback control.

> **What this constraint really says**: at any frequency, you can have either great reference tracking (small $S$, large $T$) OR great noise rejection (small $T$, large $S$) — never both. Engineers split frequency bands: tight tracking at low frequencies (where references live), tight noise rejection at high frequencies (where sensor noise lives). The crossover band is the design compromise.

## 6. Signal Flow Graphs and Mason's Formula

### 6.1 Signal Flow Graphs (SFGs)

An alternative to block diagrams — nodes represent signals, directed edges (branches) represent gains.

**Conversion from block diagram:**
- Each signal becomes a **node**
- Each block becomes a **branch** with the transfer function as gain

### 6.2 Mason's Gain Formula

For a signal flow graph, the transfer function from input to output is:

$$T = \frac{\sum_k P_k \Delta_k}{\Delta}$$

where:
- $P_k$: gain of the $k$-th forward path
- $\Delta = 1 - \sum L_i + \sum L_iL_j - \sum L_iL_jL_k + \cdots$
  - $L_i$: gain of the $i$-th individual loop
  - $L_iL_j$: product of gains of two non-touching loops
  - $L_iL_jL_k$: product of gains of three non-touching loops
- $\Delta_k$: cofactor of $\Delta$ for the $k$-th forward path (remove all loops touching path $k$)

**Two loops are non-touching** if they share no common nodes.

### 6.3 Mason's Formula Example

Consider a system with forward paths and loops:
- Forward path 1: $P_1 = G_1 G_2 G_3$ (touches loops $L_1$ and $L_2$)
- Forward path 2: $P_2 = G_4$ (touches loop $L_1$)
- Loop 1: $L_1 = -G_1 G_2 H_1$
- Loop 2: $L_2 = -G_2 G_3 H_2$
- Loops $L_1$ and $L_2$ are touching (share node after $G_2$)

$$\Delta = 1 - (L_1 + L_2) = 1 + G_1 G_2 H_1 + G_2 G_3 H_2$$

$$\Delta_1 = 1 \quad \text{(all loops touch path 1)}$$
$$\Delta_2 = 1 - L_2 = 1 + G_2 G_3 H_2 \quad \text{(loop 2 doesn't touch path 2)}$$

$$T = \frac{G_1 G_2 G_3 + G_4(1 + G_2 G_3 H_2)}{1 + G_1 G_2 H_1 + G_2 G_3 H_2}$$

> **When Mason beats block reduction**: for a system with 3+ feedback loops or several forward paths, hand block reduction takes pages and is error-prone. Mason's formula reduces it to "list paths, list loops, plug in." For circuit-style diagrams with many crossings, it is dramatically faster.

## 7. Characteristic Equation

The **characteristic equation** is obtained by setting the denominator of the closed-loop transfer function to zero:

$$1 + G(s)H(s) = 0$$

or equivalently, the denominator polynomial of $T(s)$:

$$a_n s^n + a_{n-1}s^{n-1} + \cdots + a_0 = 0$$

The roots of the characteristic equation are the **closed-loop poles**. All closed-loop stability analysis techniques (Routh-Hurwitz, root locus, Nyquist) fundamentally analyze this equation.

## 8. Common Pitfalls

1. **Cancelling pole-zero pairs without checking the cancellation is exact.** A near-cancellation in $G(s)$ leaves a slow mode that the input cannot excite (uncontrollable) but is still in the model. The transfer function loses it; the state-space view (Lessons 11-12) keeps it. For safety-critical systems, never trust pole-zero cancellation.
2. **Forgetting that $G(s)$ assumes zero initial conditions.** When you simulate a system with nonzero initial state, the transfer function does not capture it — you need state-space (Lesson 11) for that.
3. **Treating an improper transfer function as physically realizable.** $G(s) = s^2$ has more zeros than poles; it amplifies high frequencies without bound. Real plants are always proper. If your derivation gives an improper transfer function, you have an algebra mistake or an unrealistic model.
4. **Reducing block diagrams in the wrong order.** Inner loops first, then outer. Trying to reduce the outer loop while inner loops still exist usually produces a wrong (or at least messier) result. Always identify the innermost loop and collapse it before moving outward.
5. **Reading dominant-pole approximations as exact.** "Pole 5× farther is negligible" is a useful heuristic for sketching, but for hard performance specs (settling within 0.1%), the non-dominant pole IS in the math. Use the full transfer function for the final design check.
6. **Mixing $G(s)$ from different conventions silently.** Some textbooks define $G(s)$ as $Y/U$, others as $U/Y$ (impedance vs admittance style). Some put the feedback inside (negative feedback by default), others spell it out. State your conventions before deriving.

## Practice Exercises

### Exercise 1: Poles and Zeros

For the transfer function:

$$G(s) = \frac{2(s + 3)}{(s + 1)(s^2 + 4s + 8)}$$

1. Find all poles and zeros
2. Identify whether the poles are real or complex
3. Determine if the system is stable
4. Sketch the pole-zero plot in the $s$-plane

### Exercise 2: Block Diagram Reduction

Reduce the following block diagram to find $Y(s)/R(s)$:

```
R →(+)→ [G₁] →(+)→ [G₂] →(+)→ [G₃] → Y
    ↑              ↑              ↑     |
    |              └── [H₂] ←────┘     |
    └───────────── [H₁] ←──────────────┘
```

### Exercise 3: Mason's Gain Formula

A system has the following signal flow graph properties:
- Forward paths: $P_1 = ABCD$, $P_2 = AEFD$
- Loops: $L_1 = -BG$, $L_2 = -CH$, $L_3 = -EFHG$
- $L_1$ and $L_2$ are non-touching; all other pairs are touching

Find $T = Y/R$ using Mason's formula.

### Exercise 4: Python Verification

For the system in Exercise 1, use `python-control` to:
1. Construct the transfer function `G = TransferFunction([2, 6], [1, 5, 12, 8])`
2. List poles and zeros via `G.pole()` and `G.zero()`
3. Compare to your hand answer.
4. Simulate the impulse response and confirm stability visually.

### Exercise 5: Sensitivity-Complementary Sensitivity Trade-off

For a unity-feedback loop with $G(s) = K / (s(s+5))$, plot $|S(j\omega)|$ and $|T(j\omega)|$ for $K \in \{1, 10, 100\}$. Identify the frequency where $|S| = |T| = 1/2$ (called the "crossover frequency"). How does the trade-off shift as $K$ grows? Relate this to the engineering principle "low frequency for tracking, high frequency for noise rejection."

---

*Previous: [Lesson 2 — Mathematical Modeling of Physical Systems](02_Mathematical_Modeling.md) | Next: [Lesson 4 — Time-Domain Analysis](04_Time_Domain_Analysis.md)*
