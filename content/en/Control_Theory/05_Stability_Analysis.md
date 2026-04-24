# Lesson 5: Stability Analysis

## Learning Objectives

- Define BIBO stability and asymptotic stability for LTI systems
- Apply the Routh-Hurwitz criterion to determine stability from the characteristic polynomial
- Use the Routh array to find the range of a parameter for stability
- Identify marginally stable and unstable systems
- Understand the connection between pole locations and stability
- Verify stability conclusions numerically and recognize the pitfalls that the Routh criterion is designed to avoid

## 0. Motivation — Why Routh When We Have `numpy.roots`?

A 2026 student reasonably asks: if a three-line Python call factors any polynomial, why spend pages on a tabular algorithm invented in 1875?

Three reasons the Routh array still earns its place:

- **Stability as a function of a parameter.** Root-finding tells you "at $K = 12$ the poles are at …". Routh answers "for what range of $K$ is this system stable?" in a single symbolic pass. That parametric question is the one control engineers ask most often, and Python's numerical root-finder cannot answer it directly.
- **Analytical insight.** The Routh conditions for a third-order system collapse to $a_2 a_1 > a_0$ — a relationship you can remember and apply from a napkin. The numerical answer for one $K$ gives you no such intuition.
- **It catches symbolic mistakes.** When you are deriving a characteristic polynomial by hand, a sign error in one coefficient flips the stability conclusion. Routh's necessary condition (all coefficients same sign) is a 10-second sanity check — far faster than setting up a numerical test.

Keep a concrete picture: imagine a ball sitting in the bottom of a bowl vs. balanced on top of a hill. The bowl is stable (push the ball, it returns); the hilltop is unstable. Poles in the left half-plane are bowl-shaped exponentials that decay toward equilibrium; poles in the right half-plane are hilltop exponentials that run away. The imaginary axis is the razor's edge — pure oscillation forever, neither decaying nor growing.

## 1. Stability Concepts

Stability is the **most fundamental requirement** for any control system. An unstable system is at best useless and at worst dangerous.

### 1.1 BIBO Stability

A system is **bounded-input, bounded-output (BIBO) stable** if every bounded input produces a bounded output:

$$|u(t)| \leq M_u < \infty \quad \Rightarrow \quad |y(t)| \leq M_y < \infty$$

For an LTI system with impulse response $g(t)$, BIBO stability is equivalent to:

$$\int_0^\infty |g(t)| \, dt < \infty$$

### 1.2 Asymptotic Stability

A system is **asymptotically stable** if the free response (zero input) decays to zero:

$$\lim_{t \to \infty} y_{\text{free}}(t) = 0$$

### 1.3 Stability from Pole Locations

For LTI systems, both conditions reduce to a simple pole test:

| Stability | Condition |
|-----------|-----------|
| **Asymptotically stable** | All poles in the open left half-plane: $\text{Re}(p_i) < 0 \; \forall i$ |
| **Marginally stable** | No poles in the RHP, at least one simple pole on the imaginary axis |
| **Unstable** | At least one pole in the RHP, or repeated poles on the imaginary axis |

For closed-loop stability, we analyze the roots of the **characteristic equation**:

$$1 + G(s)H(s) = 0 \quad \Leftrightarrow \quad \Delta(s) = 0$$

> **Caution**: BIBO stability and asymptotic stability are identical for an LTI system's input-output pair *only when* there are no pole-zero cancellations at unstable locations. A hidden RHP pole cancelled by a zero passes the BIBO test but still makes the internal state diverge. This is why modern analysis checks poles of the **state-space** $A$ matrix, not just the transfer function — covered in Lesson 12.

## 2. The Routh-Hurwitz Criterion

Finding the roots of a polynomial of degree $>2$ is generally difficult. The Routh-Hurwitz criterion determines stability **without computing the roots**.

### 2.1 Necessary Condition

**Theorem:** A necessary condition for all roots of $\Delta(s) = a_n s^n + \cdots + a_1 s + a_0$ to have negative real parts is that all coefficients $a_i > 0$ (assuming $a_n > 0$).

If any coefficient is zero or negative, the system is **not** stable. (This is necessary but not sufficient for $n \geq 3$.)

### 2.2 Routh Array Construction

Given the characteristic polynomial:

$$\Delta(s) = a_n s^n + a_{n-1} s^{n-1} + a_{n-2} s^{n-2} + \cdots + a_0$$

Construct the Routh array:

| $s^n$ | $a_n$ | $a_{n-2}$ | $a_{n-4}$ | $\cdots$ |
|-------|-------|-----------|-----------|----------|
| $s^{n-1}$ | $a_{n-1}$ | $a_{n-3}$ | $a_{n-5}$ | $\cdots$ |
| $s^{n-2}$ | $b_1$ | $b_2$ | $b_3$ | $\cdots$ |
| $s^{n-3}$ | $c_1$ | $c_2$ | $c_3$ | $\cdots$ |
| $\vdots$ | | | | |
| $s^0$ | | | | |

where:

$$b_1 = \frac{a_{n-1}a_{n-2} - a_n a_{n-3}}{a_{n-1}}, \quad b_2 = \frac{a_{n-1}a_{n-4} - a_n a_{n-5}}{a_{n-1}}, \quad \ldots$$

$$c_1 = \frac{b_1 a_{n-3} - a_{n-1} b_2}{b_1}, \quad \ldots$$

### 2.3 Routh-Hurwitz Stability Criterion

**Theorem (Routh-Hurwitz):** The number of roots of $\Delta(s)$ with positive real parts equals the **number of sign changes** in the first column of the Routh array.

**Corollary:** The system is stable if and only if **all entries in the first column are positive** (assuming $a_n > 0$).

### 2.4 Example

$$\Delta(s) = s^4 + 2s^3 + 3s^2 + 4s + 5$$

Routh array:

| Row | Col 1 | Col 2 | Col 3 |
|-----|-------|-------|-------|
| $s^4$ | $1$ | $3$ | $5$ |
| $s^3$ | $2$ | $4$ | $0$ |
| $s^2$ | $\frac{2 \cdot 3 - 1 \cdot 4}{2} = 1$ | $\frac{2 \cdot 5 - 1 \cdot 0}{2} = 5$ | |
| $s^1$ | $\frac{1 \cdot 4 - 2 \cdot 5}{1} = -6$ | | |
| $s^0$ | $5$ | | |

First column: $1, 2, 1, -6, 5$

Sign changes: $1 \to -6$ and $-6 \to 5$ → **2 sign changes** → 2 RHP roots → **Unstable**.

### 2.5 The Array-Filling Recipe, Written Mechanically

Once you see the pattern the array is mechanical. For each new row:

1. Look at the **two rows immediately above** (call them "upper" and "above-upper").
2. For each column $j$ of the new row, compute a 2×2 determinant built from columns $0$ and $j+1$ of those two upper rows, divided by the pivot (column 0 of "upper"):
   $$\text{new}_j = \frac{\text{upper}[0] \cdot \text{above-upper}[j+1] - \text{above-upper}[0] \cdot \text{upper}[j+1]}{\text{upper}[0]}$$
   Notice this is always the **same 2×2 pattern** with "upper" as the denominator.
3. Stop when a row has only one nonzero column (that is the $s^0$ row).

A worked spreadsheet-style mnemonic:

```
[above-upper]:   A  B  C
[upper]:         D  E  F
[new row]:      (D*B - A*E)/D   (D*C - A*F)/D   ...
```

If you remember the 2×2 "cross minus cross, divided by the pivot D" pattern, you never need to look up the formula again.

## 3. Special Cases in the Routh Array

### 3.1 Zero in the First Column

If the first element of a row is zero (but the row is not all zeros), replace the zero with a small positive number $\epsilon > 0$ and continue. After completing the array, examine the signs as $\epsilon \to 0^+$.

**Example:** $\Delta(s) = s^3 + s^2 + 2s + 2$

| $s^3$ | $1$ | $2$ |
| $s^2$ | $1$ | $2$ |
| $s^1$ | $\frac{1\cdot 2 - 1\cdot 2}{1} = 0 \to \epsilon$ | |
| $s^0$ | $2$ | |

First column: $1, 1, \epsilon, 2$ — no sign changes as $\epsilon \to 0^+$ → stable?

Actually, the zero row indicates **imaginary axis roots**: $\Delta(s) = (s^2+2)(s+1)$, so poles at $s = \pm j\sqrt{2}$ → **marginally stable**, not asymptotically stable.

### 3.2 Entire Row of Zeros

If an entire row becomes zero, it indicates that the characteristic polynomial has **symmetric root pairs** (roots that are negatives of each other: $\pm\sigma$, $\pm j\omega$, or $\pm\sigma \pm j\omega$).

**Procedure:**
1. Form the **auxiliary polynomial** $P(s)$ from the row **above** the zero row
2. Replace the zero row with the coefficients of $\frac{dP}{ds}$
3. Continue the Routh array

The roots of $P(s)$ are the symmetric root pairs and include the imaginary axis roots.

### 3.3 Why the $\epsilon$ Trick Works

The $\epsilon \to 0^+$ substitution is not a hack — it is a limit argument. The Routh conditions are continuous in the polynomial coefficients except right at the singularity where the pivot is zero. Perturbing by a tiny positive $\epsilon$ moves you a hair off the singular locus, the counts remain valid, and the limit recovers the correct sign pattern.

If the sign pattern DIFFERS as $\epsilon \to 0^+$ vs. $\epsilon \to 0^-$, you have a marginally stable system — the polynomial has roots exactly on the imaginary axis. This is the flag Section 3.1 caught implicitly.

## 4. Stability Ranges Using Routh-Hurwitz

One of the most powerful applications: finding the range of a parameter (typically gain $K$) for which the system is stable.

### 4.1 Example: Finding the Stability Range of $K$

A unity-feedback system with $G(s) = \frac{K}{s(s+1)(s+5)}$.

Characteristic equation: $s^3 + 6s^2 + 5s + K = 0$

Routh array:

| $s^3$ | $1$ | $5$ |
| $s^2$ | $6$ | $K$ |
| $s^1$ | $\frac{30 - K}{6}$ | |
| $s^0$ | $K$ | |

For stability, all first-column entries must be positive:
- $6 > 0$ ✓
- $\frac{30-K}{6} > 0 \Rightarrow K < 30$
- $K > 0$

**Stability range:** $0 < K < 30$.

At $K = 30$: the $s^1$ row becomes zero → **marginally stable** with sustained oscillation. The auxiliary polynomial from the $s^2$ row: $6s^2 + 30 = 0 \Rightarrow s = \pm j\sqrt{5}$. The frequency of oscillation is $\omega = \sqrt{5}$ rad/s.

### 4.2 Verification in Python

The symbolic answer above says "stable for $0 < K < 30$." A 20-line numerical sweep confirms this and builds confidence:

```python
import numpy as np

def is_stable(K):
    # char poly: s^3 + 6s^2 + 5s + K
    roots = np.roots([1, 6, 5, K])
    return np.all(roots.real < 0)

for K in [0.1, 10, 20, 29, 30, 30.01, 100]:
    print(f"K = {K:>7.2f}  stable = {is_stable(K)}")
```

Expected output:

```
K =    0.10  stable = True
K =   10.00  stable = True
K =   20.00  stable = True
K =   29.00  stable = True
K =   30.00  stable = False   # ← exactly the boundary
K =   30.01  stable = False
K =  100.00  stable = False
```

Notice `K = 30` reports `False`: at the boundary, `numpy.roots` computes poles with tiny-but-nonzero real parts because of floating-point noise. The Routh analysis is the ground truth here — the system is marginally stable (pure oscillation at $\omega = \sqrt{5}$), which is neither "stable" in the asymptotic sense nor RHP-unstable. This mismatch is exactly why the symbolic method is worth knowing: numerical tools blur the boundary, while Routh nails it.

## 5. Hurwitz Determinants (Alternative Formulation)

The Hurwitz criterion provides the same information through determinants. For $\Delta(s) = a_n s^n + \cdots + a_0$ with $a_n > 0$, all roots have negative real parts if and only if the **Hurwitz determinants** are all positive:

$$D_1 = a_{n-1} > 0$$

$$D_2 = \begin{vmatrix} a_{n-1} & a_n \\ a_{n-3} & a_{n-2} \end{vmatrix} > 0$$

$$D_3 = \begin{vmatrix} a_{n-1} & a_n & 0 \\ a_{n-3} & a_{n-2} & a_{n-1} \\ a_{n-5} & a_{n-4} & a_{n-3} \end{vmatrix} > 0$$

For low-order systems this can be simpler than the full Routh array.

### 5.1 Special Cases

**Second-order** $s^2 + a_1 s + a_0$: Stable if and only if $a_1 > 0$ and $a_0 > 0$.

**Third-order** $s^3 + a_2 s^2 + a_1 s + a_0$: Stable if and only if $a_2 > 0$, $a_0 > 0$, and $a_2 a_1 > a_0$.

### 5.2 Physical Example: Spring-Mass-Damper

For $m\ddot{x} + b\dot{x} + kx = F$, the characteristic polynomial is $ms^2 + bs + k$, i.e. a second-order form with $a_1 = b/m$ and $a_0 = k/m$.

Routh / Hurwitz says stable iff $b > 0$ and $k > 0$. Translated to physical parameters:

- $k > 0$ means the spring pulls back (positive restoring force). A $k = 0$ spring is a free mass; $k < 0$ is a repulsive force (inverted pendulum before linearization) — both unstable.
- $b > 0$ means the damper dissipates energy. $b = 0$ is a lossless oscillator (imaginary-axis poles, marginally stable); $b < 0$ would be an energy-injecting damper — unstable.

So the Routh criteria map exactly onto the physical intuition: you need a restoring force AND energy dissipation. The mathematics is just bookkeeping for this intuition.

## 6. Relative Stability

The Routh criterion tells us only if poles are in the LHP. For **relative stability** — how far the poles are from the imaginary axis — we can use a shifted variable.

**Method:** To determine if all poles have $\text{Re}(p_i) < -\sigma_0$, substitute $s = \hat{s} - \sigma_0$ into $\Delta(s)$ and apply Routh to the new polynomial in $\hat{s}$.

If all entries in the first column of the shifted Routh array are positive, all original poles satisfy $\text{Re}(p_i) < -\sigma_0$.

> Why this matters: a system with all poles just inside the LHP is "technically stable" but will take a very long time to settle and amplifies noise. Engineering rules typically demand $\text{Re}(p_i) < -\sigma_0$ for some explicit $\sigma_0$ — 0.5 or 1.0 for slow processes, 5–10 for servos.

## 7. Common Pitfalls

1. **Using the necessary condition as sufficient.** "All coefficients are positive, therefore stable" is WRONG for $n \geq 3$. It is a quick first check, but the full Routh array must be completed.
2. **Forgetting that the Routh test assumes $a_n > 0$.** If your leading coefficient is negative, multiply the polynomial by $-1$ first. Omitting this reverses every sign change count.
3. **Misreading a zero row as "stable because no sign changes".** A zero row signals imaginary-axis roots — marginal stability, not asymptotic stability. Always apply the auxiliary-polynomial procedure.
4. **Mixing up Routh for $K$ ranges.** Students routinely derive "$K > 0$" and stop, forgetting the upper bound from the row where $K$ enters negatively. Check **every** first-column entry as a function of $K$.
5. **Over-trusting numerical root finders at the boundary.** As Section 4.2 showed, floating-point errors near marginal stability can give misleading "unstable" verdicts. Use Routh for boundary-of-stability questions.
6. **Pole-zero cancellation hiding instability.** A transfer-function Routh test cannot see a pole cancelled by a zero at the same RHP location. The state-space eigenvalue test (Lesson 12) is the only safe guarantee; rely on that for safety-critical systems.

## Practice Exercises

### Exercise 1: Routh-Hurwitz Application

Apply the Routh-Hurwitz criterion to determine the stability of systems with these characteristic polynomials:

1. $s^4 + 3s^3 + 5s^2 + 4s + 2$
2. $s^4 + s^3 + 2s^2 + 2s + 1$
3. $s^5 + 2s^4 + 3s^3 + 6s^2 + 2s + 1$

### Exercise 2: Gain Range

A unity-feedback system has $G(s) = \frac{K(s+2)}{s(s+1)(s+3)(s+4)}$.

1. Determine the range of $K > 0$ for stability
2. Find the frequency of oscillation at the critical value of $K$

### Exercise 3: Relative Stability

For the characteristic polynomial $s^3 + 10s^2 + 31s + 30$:

1. Verify that the system is stable
2. Determine whether all poles satisfy $\text{Re}(p_i) < -1$
3. Find the actual poles and verify your answers from parts 1 and 2

### Exercise 4: Numerical Sweep

Write a small Python script that sweeps $K$ from $-5$ to $50$ in steps of $0.1$ for the system in Section 4.1 and plots the rightmost-pole real part as a function of $K$. Mark the critical $K$ where the curve crosses zero. Compare to the Routh prediction.

### Exercise 5: Hidden RHP Pole

Construct a closed-loop transfer function that appears stable by the Routh test on its denominator but has a hidden RHP pole canceled by a matching zero. Show the consequence by applying a small disturbance to the state directly (bypass the input) and observing that the output diverges despite the BIBO-stable transfer function.

---

*Previous: [Lesson 4 — Time-Domain Analysis](04_Time_Domain_Analysis.md) | Next: [Lesson 6 — Root Locus Method](06_Root_Locus.md)*
