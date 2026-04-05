# Lesson 7: Frequency Response — Bode Plots

## Learning Objectives

- Understand the concept of frequency response for LTI systems
- Construct Bode magnitude and phase plots from a transfer function
- Use asymptotic approximations for quick sketching
- Read gain margin and phase margin from Bode plots
- Apply frequency-domain specifications to assess system performance

## 1. Frequency Response Concept

The **frequency response** describes how a system responds to sinusoidal inputs at different frequencies.

For a stable LTI system with transfer function $G(s)$, if the input is:

$$u(t) = A\sin(\omega t)$$

then the steady-state output is:

$$y_{ss}(t) = A|G(j\omega)|\sin(\omega t + \angle G(j\omega))$$

The system **scales** the amplitude by $|G(j\omega)|$ and **shifts** the phase by $\angle G(j\omega)$.

### 1.1 Frequency Response Function

$$G(j\omega) = |G(j\omega)| e^{j\angle G(j\omega)}$$

- **Magnitude:** $|G(j\omega)| = \sqrt{[\text{Re}(G(j\omega))]^2 + [\text{Im}(G(j\omega))]^2}$
- **Phase:** $\angle G(j\omega) = \tan^{-1}\frac{\text{Im}(G(j\omega))}{\text{Re}(G(j\omega))}$

## 2. Bode Plot Basics

A **Bode plot** consists of two graphs:
1. **Magnitude plot**: $20\log_{10}|G(j\omega)|$ in decibels (dB) vs. $\log_{10}\omega$
2. **Phase plot**: $\angle G(j\omega)$ in degrees vs. $\log_{10}\omega$

### 2.1 Advantages of Logarithmic Plots

- Products of transfer functions become **sums** of individual Bode plots
- Powers become **multiplications**: $|G^n| \text{ dB} = n \times |G| \text{ dB}$
- Straight-line asymptotic approximations are easy to draw
- Covers a wide range of frequencies (decades)

### 2.2 Decibel Scale

$$|G|_{\text{dB}} = 20\log_{10}|G(j\omega)|$$

| Linear | dB |
|--------|-----|
| $1$ | $0$ dB |
| $2$ | $6$ dB |
| $10$ | $20$ dB |
| $100$ | $40$ dB |
| $0.1$ | $-20$ dB |
| $0.01$ | $-40$ dB |

## 3. Bode Plots of Basic Elements

Any transfer function can be decomposed into basic elements:

$$G(s) = \frac{K \prod(1 + s/z_i) \prod(1 + 2\zeta_k s/\omega_{n_k} + s^2/\omega_{n_k}^2)}{s^N \prod(1 + s/p_j) \prod(1 + 2\zeta_l s/\omega_{n_l} + s^2/\omega_{n_l}^2)}$$

### 3.1 Constant Gain $K$

- Magnitude: $20\log_{10}|K|$ dB (horizontal line)
- Phase: $0°$ if $K > 0$, $-180°$ if $K < 0$

### 3.2 Integrator/Differentiator $s^{\pm N}$

**Integrator** $1/s$:
- Magnitude: $-20$ dB/decade (line through 0 dB at $\omega = 1$)
- Phase: $-90°$ (constant)

**Double integrator** $1/s^2$:
- Magnitude: $-40$ dB/decade
- Phase: $-180°$

### 3.3 First-Order Factor $(1 + s/\omega_0)^{\pm 1}$

**First-order zero** $(1 + s/\omega_0)$:

| Frequency | Magnitude Asymptote | Phase |
|-----------|-------------------|-------|
| $\omega \ll \omega_0$ | $0$ dB | $0°$ |
| $\omega = \omega_0$ | $+3$ dB | $+45°$ |
| $\omega \gg \omega_0$ | $+20$ dB/decade | $+90°$ |

- **Corner (break) frequency:** $\omega_0$
- Phase transition: from $0.1\omega_0$ to $10\omega_0$

**First-order pole** $1/(1 + s/\omega_0)$: magnitude and phase are the negative of the zero.

### 3.4 Second-Order Factor

**Underdamped quadratic** $\frac{1}{1 + 2\zeta s/\omega_n + s^2/\omega_n^2}$:

| Frequency | Magnitude Asymptote | Phase |
|-----------|-------------------|-------|
| $\omega \ll \omega_n$ | $0$ dB | $0°$ |
| $\omega = \omega_n$ | $-20\log_{10}(2\zeta)$ dB (resonance peak) | $-90°$ |
| $\omega \gg \omega_n$ | $-40$ dB/decade | $-180°$ |

The resonance peak height depends on $\zeta$:
- $\zeta = 0.1$: $+14$ dB peak
- $\zeta = 0.3$: $+5.7$ dB peak
- $\zeta = 0.707$: $0$ dB (no peak, $-3$ dB at $\omega_n$)
- $\zeta \geq 0.707$: no resonance peak

**Resonant frequency:** $\omega_r = \omega_n\sqrt{1 - 2\zeta^2}$ (for $\zeta < 1/\sqrt{2}$)

**Resonant peak magnitude:** $M_r = \frac{1}{2\zeta\sqrt{1-\zeta^2}}$

## 4. Composite Bode Plot Construction

### 4.1 Procedure

1. Rewrite $G(s)$ in **time-constant form**: factor out constants so each term is $(1 + s/\omega_i)$ or $(1 + 2\zeta s/\omega_n + s^2/\omega_n^2)$
2. Identify all corner frequencies and sort them
3. Plot each element's contribution separately
4. **Add** all magnitude contributions (dB add)
5. **Add** all phase contributions

### 4.2 Example

$$G(s) = \frac{100(s+1)}{s(s+10)} = \frac{10(1+s)}{s(1+s/10)}$$

The DC gain (after factoring out the integrator and evaluating at $\omega \to 0$) normalization: at $\omega = 1$, the magnitude is $10/\omega = 10$ → $20$ dB.

**Elements:**
- Gain $10$: $+20$ dB
- Integrator $1/s$: $-20$ dB/dec, passes through $0$ dB at $\omega = 1$
- Zero at $\omega = 1$: break upward
- Pole at $\omega = 10$: break downward

**Asymptotic magnitude:**
- $\omega < 1$: slope $-20$ dB/dec (integrator only)
- $1 < \omega < 10$: slope $0$ dB/dec (integrator + zero cancel)
- $\omega > 10$: slope $-20$ dB/dec (integrator + zero + pole)

**Phase:**
- $\omega \to 0$: $-90°$ (integrator) + $0°$ + $0°$ = $-90°$
- $\omega = 1$: $-90°$ + $45°$ + $(-5.7°)$ ≈ $-51°$
- $\omega = 10$: $-90°$ + $84.3°$ + $(-45°)$ ≈ $-51°$
- $\omega \to \infty$: $-90°$ + $90°$ + $(-90°)$ = $-90°$

## 5. Stability Margins from Bode Plots

### 5.1 Gain Margin (GM)

The **gain margin** is the amount (in dB) by which the gain can be increased before the system becomes unstable:

$$GM = -20\log_{10}|G(j\omega_{pc})| \quad \text{dB}$$

where $\omega_{pc}$ is the **phase crossover frequency** (where $\angle G(j\omega) = -180°$).

**Interpretation:** The gain margin tells us by how many dB we can increase $K$ before the loop gain reaches 0 dB at the $-180°$ phase crossing.

### 5.2 Phase Margin (PM)

The **phase margin** is the additional phase lag needed to reach $-180°$ at the gain crossover:

$$PM = 180° + \angle G(j\omega_{gc})$$

where $\omega_{gc}$ is the **gain crossover frequency** (where $|G(j\omega)| = 0$ dB, i.e., unity gain).

### 5.3 Stability from Margins

| Condition | Stable? |
|-----------|---------|
| $GM > 0$ dB and $PM > 0°$ | Yes |
| $GM < 0$ dB or $PM < 0°$ | No |
| $GM = 0$ dB or $PM = 0°$ | Marginally stable |

**Typical design targets:**
- $GM \geq 6$ dB (factor of 2 gain tolerance)
- $PM \geq 30°$ to $60°$
- $PM \approx 60°$ gives $\zeta \approx 0.6$ (good balance of speed and damping)

### 5.4 Relationship Between Phase Margin and Damping

For a second-order system, the phase margin and damping ratio are approximately related by:

$$PM \approx 100\zeta \quad \text{degrees (for } \zeta < 0.7\text{)}$$

More precisely: $\zeta \approx PM/100$ for $PM < 70°$.

## 6. Frequency-Domain Specifications

| Specification | Symbol | Meaning |
|--------------|--------|---------|
| **Bandwidth** | $\omega_{BW}$ | Frequency where $|T(j\omega)|$ drops to $-3$ dB |
| **Resonant peak** | $M_r$ | Maximum value of $|T(j\omega)|$ |
| **Gain crossover frequency** | $\omega_{gc}$ | Approximates closed-loop bandwidth |

**Connections to time domain:**
- Larger $\omega_{BW}$ → faster response (smaller $t_r$)
- Larger $M_r$ → more overshoot
- $\omega_{BW} \approx \omega_n\sqrt{(1-2\zeta^2) + \sqrt{4\zeta^4 - 4\zeta^2 + 2}}$ for second-order systems

## Practice Exercises

### Exercise 1: Bode Plot Construction

Sketch the asymptotic Bode plot (magnitude and phase) for:

$$G(s) = \frac{50(s+5)}{s(s+2)(s+50)}$$

1. Rewrite in time-constant form
2. Identify all corner frequencies
3. Draw asymptotic magnitude and phase plots
4. Determine the gain and phase margins

### Exercise 2: System Identification from Bode Plot

A system's Bode magnitude plot shows:
- Slope of $-20$ dB/dec for $\omega < 2$
- Slope of $-40$ dB/dec for $2 < \omega < 10$
- Slope of $-60$ dB/dec for $\omega > 10$
- Magnitude is $20$ dB at $\omega = 1$

Determine the transfer function.

### Exercise 3: Stability Margin Analysis

For $G(s) = \frac{K}{s(0.1s+1)(0.01s+1)}$:

1. With $K = 10$, find the gain margin and phase margin
2. Find the maximum value of $K$ for which the system is stable
3. What value of $K$ gives a phase margin of $45°$?

---

*Previous: [Lesson 6 — Root Locus Method](06_Root_Locus.md) | Next: [Lesson 8 — Nyquist Stability Criterion](08_Nyquist_Stability.md)*
