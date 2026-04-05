# Lesson 6: Root Locus Method

## Learning Objectives

- Understand the concept of root locus as a graphical tool for control design
- Apply the rules for sketching root loci by hand
- Use root locus to select gain for desired closed-loop pole locations
- Analyze the effect of adding poles and zeros to the open-loop transfer function
- Apply root locus to controller design

## 1. Introduction to Root Locus

The **root locus** is the set of all possible closed-loop pole locations as a parameter (usually gain $K$) varies from $0$ to $\infty$.

For a unity-feedback system with open-loop transfer function $KG(s)H(s)$, the characteristic equation is:

$$1 + KG(s)H(s) = 0$$

$$KG(s)H(s) = -1$$

This requires two conditions simultaneously:
- **Magnitude condition:** $|KG(s)H(s)| = 1$
- **Angle condition:** $\angle G(s)H(s) = (2k+1) \times 180°$ for $k = 0, \pm1, \pm2, \ldots$

The root locus consists of all points $s$ satisfying the angle condition. The gain $K$ at any point on the locus is found from the magnitude condition.

## 2. Rules for Sketching Root Locus

Let $G(s)H(s) = \frac{N(s)}{D(s)}$ with $n$ poles and $m$ zeros ($n \geq m$).

### Rule 1: Starting and Ending Points
- The locus **starts** ($K = 0$) at the **open-loop poles**
- The locus **ends** ($K \to \infty$) at the **open-loop zeros** (including $n - m$ zeros at infinity)

### Rule 2: Number of Branches
- There are $n$ branches (one for each open-loop pole)

### Rule 3: Symmetry
- The root locus is **symmetric about the real axis** (complex poles come in conjugate pairs)

### Rule 4: Real-Axis Segments
- A point on the real axis is on the root locus if the **total number of real poles and zeros to its right is odd**

### Rule 5: Asymptotes
The $n - m$ branches going to infinity approach asymptotes:
- **Angles:** $\theta_a = \frac{(2k+1) \times 180°}{n-m}$ for $k = 0, 1, \ldots, n-m-1$
- **Centroid** (intersection point on real axis): $\sigma_a = \frac{\sum p_i - \sum z_j}{n - m}$

### Rule 6: Breakaway and Break-in Points
Points where branches leave (or enter) the real axis are found from:

$$\frac{dK}{ds} = 0 \quad \text{where} \quad K = -\frac{1}{G(s)H(s)}$$

### Rule 7: Departure and Arrival Angles
- **Departure angle** from a complex pole $p_i$: $\theta_d = 180° - \sum \angle(p_i - p_j) + \sum \angle(p_i - z_k)$ (summing over all other poles and zeros)
- **Arrival angle** at a complex zero $z_i$: $\theta_a = 180° + \sum \angle(z_i - p_j) - \sum \angle(z_i - z_k)$

### Rule 8: Imaginary Axis Crossings
Found by substituting $s = j\omega$ into the characteristic equation and solving for $\omega$ and $K$ (or using the Routh criterion to find the critical $K$).

## 3. Root Locus Example

**Example:** $G(s)H(s) = \frac{1}{s(s+1)(s+3)}$

- **Poles:** $s = 0, -1, -3$ ($n = 3$, $m = 0$)
- **Zeros:** None finite → 3 zeros at infinity

**Applying the rules:**

1. **Branches:** 3, starting at $s = 0, -1, -3$
2. **Asymptotes:** $n - m = 3$
   - Angles: $60°, 180°, 300°$
   - Centroid: $\sigma_a = (0 + (-1) + (-3))/3 = -4/3$
3. **Real-axis segments:** $(-\infty, -3)$ and $(-1, 0)$
4. **Breakaway point:** Solve $\frac{d}{ds}[s(s+1)(s+3)] = 0$: $3s^2 + 8s + 3 = 0 \Rightarrow s = -0.451, -2.215$
   - $s = -0.451$ is on the locus (between $-1$ and $0$) → breakaway point
   - $s = -2.215$ is not on the locus
5. **Imaginary crossing:** Using Routh on $s^3 + 4s^2 + 3s + K = 0$: critical $K = 12$, $\omega = \sqrt{3}$

The locus shows three branches diverging from the real axis — as $K$ increases from 0, the system becomes unstable at $K = 12$.

## 4. Effect of Adding Poles and Zeros

### 4.1 Adding a Pole

Adding an open-loop pole:
- Pushes the root locus **toward the right half-plane** (destabilizing)
- Reduces the range of $K$ for stability
- Adds another branch going to infinity

### 4.2 Adding a Zero

Adding an open-loop zero:
- Pulls the root locus **toward the left half-plane** (stabilizing)
- Increases the range of $K$ for stability
- Provides a termination point for a locus branch

### 4.3 Lead and Lag Compensation Preview

- **Lead compensator** (zero to the left of pole): bends the root locus left → improves transient response
- **Lag compensator** (pole close to origin, zero nearby): modifies low-frequency gain without significantly affecting the root locus shape → reduces steady-state error

These are developed fully in Lesson 10.

## 5. Using Root Locus for Design

### 5.1 Gain Selection

To achieve a desired closed-loop pole location $s_d$:

1. Verify $s_d$ lies on the root locus (check the angle condition)
2. Calculate $K$ from the magnitude condition: $K = \frac{1}{|G(s_d)H(s_d)|}$
3. Check that all other closed-loop poles are acceptable

### 5.2 Design to Specifications

Given time-domain specifications ($t_s$, $M_p$, etc.):

1. Convert to desired $\zeta$ and $\omega_n$ (or equivalently, desired $\sigma$ and $\omega_d$)
2. Identify the target region in the $s$-plane
3. If a point in the target region lies on the root locus, select the gain
4. If not, add compensation (poles/zeros) to reshape the locus

### 5.3 Example: Gain Design

For $G(s) = \frac{K}{s(s+4)}$, design for $M_p \leq 20\%$ and $t_s \leq 2$ s.

**Specifications → $s$-plane:**
- $M_p \leq 20\%$: $\zeta \geq 0.456$
- $t_s \leq 2$ s: $\sigma = \zeta\omega_n \geq 2$

The root locus is a vertical line at $s = -2$ (breakaway from the real axis midpoint). We need poles with $\text{Re}(s) \leq -2$ and $\zeta \geq 0.456$.

At $s = -2 + j3.88$ (satisfying $\zeta = 0.456$): $K = |s(s+4)| = |(-2+j3.88)(-2+j3.88+4)| = 19.4$

## 6. Root Locus for Negative $K$ (Complementary Root Locus)

When $K$ varies from $-\infty$ to $0$, the angle condition becomes:

$$\angle G(s)H(s) = 2k \times 180°$$

The complementary root locus uses slightly different rules:
- Real-axis segments: total number of poles and zeros to the right must be **even** (including zero)
- Asymptote angles: $\theta_a = \frac{2k \times 180°}{n-m}$

This is relevant for positive feedback systems.

## Practice Exercises

### Exercise 1: Root Locus Sketch

Sketch the root locus for $G(s) = \frac{K}{s(s+2)(s+4)}$.

1. Apply all rules to determine real-axis segments, asymptotes, breakaway points, and imaginary axis crossings
2. Find the value of $K$ at the imaginary axis crossing
3. For what range of $K$ is the system stable?

### Exercise 2: Design with Root Locus

For $G(s) = \frac{K(s+3)}{s(s+1)(s+5)}$:

1. Sketch the root locus
2. Find the value of $K$ that places a closed-loop pole at $s = -2$
3. Find all closed-loop poles at this value of $K$

### Exercise 3: Effect of Compensation

A system has $G(s) = \frac{K}{s(s+2)}$. Compare the root loci when a compensator zero is added at:
1. $s = -1$ (between the poles)
2. $s = -5$ (to the left of both poles)

Sketch both root loci and discuss how the zero location affects the achievable closed-loop pole positions.

---

*Previous: [Lesson 5 — Stability Analysis](05_Stability_Analysis.md) | Next: [Lesson 7 — Frequency Response: Bode Plots](07_Bode_Plots.md)*
