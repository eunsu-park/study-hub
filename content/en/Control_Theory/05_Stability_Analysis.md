# Lesson 5: Stability Analysis

## Learning Objectives

- Define BIBO stability and asymptotic stability for LTI systems
- Apply the Routh-Hurwitz criterion to determine stability from the characteristic polynomial
- Use the Routh array to find the range of a parameter for stability
- Identify marginally stable and unstable systems
- Understand the connection between pole locations and stability

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

## 5. Hurwitz Determinants (Alternative Formulation)

The Hurwitz criterion provides the same information through determinants. For $\Delta(s) = a_n s^n + \cdots + a_0$ with $a_n > 0$, all roots have negative real parts if and only if the **Hurwitz determinants** are all positive:

$$D_1 = a_{n-1} > 0$$

$$D_2 = \begin{vmatrix} a_{n-1} & a_n \\ a_{n-3} & a_{n-2} \end{vmatrix} > 0$$

$$D_3 = \begin{vmatrix} a_{n-1} & a_n & 0 \\ a_{n-3} & a_{n-2} & a_{n-1} \\ a_{n-5} & a_{n-4} & a_{n-3} \end{vmatrix} > 0$$

For low-order systems this can be simpler than the full Routh array.

### 5.1 Special Cases

**Second-order** $s^2 + a_1 s + a_0$: Stable if and only if $a_1 > 0$ and $a_0 > 0$.

**Third-order** $s^3 + a_2 s^2 + a_1 s + a_0$: Stable if and only if $a_2 > 0$, $a_0 > 0$, and $a_2 a_1 > a_0$.

## 6. Relative Stability

The Routh criterion tells us only if poles are in the LHP. For **relative stability** — how far the poles are from the imaginary axis — we can use a shifted variable.

**Method:** To determine if all poles have $\text{Re}(p_i) < -\sigma_0$, substitute $s = \hat{s} - \sigma_0$ into $\Delta(s)$ and apply Routh to the new polynomial in $\hat{s}$.

If all entries in the first column of the shifted Routh array are positive, all original poles satisfy $\text{Re}(p_i) < -\sigma_0$.

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

---

*Previous: [Lesson 4 — Time-Domain Analysis](04_Time_Domain_Analysis.md) | Next: [Lesson 6 — Root Locus Method](06_Root_Locus.md)*
