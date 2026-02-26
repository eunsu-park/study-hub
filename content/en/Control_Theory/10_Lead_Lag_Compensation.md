# Lesson 10: Lead-Lag Compensation

## Learning Objectives

- Design lead compensators to improve transient response and stability margins
- Design lag compensators to reduce steady-state error without degrading stability
- Combine lead and lag compensation for comprehensive performance improvement
- Apply frequency-domain (Bode) design methods for compensator synthesis
- Understand the trade-offs between different compensation strategies

## 1. Compensation Overview

When the plant $G_p(s)$ with a simple gain $K$ cannot meet all design specifications, we add a **compensator** $G_c(s)$ to reshape the system's frequency response or root locus.

```
R →(+)→ [G_c(s)] → [G_p(s)] → Y
    ↑                          |
    └──────────────────────────┘
```

**Types of compensation:**

| Type | Structure | Primary Effect |
|------|-----------|----------------|
| Lead | Zero to the left of pole | Adds phase lead → improves PM, speeds up response |
| Lag | Pole close to origin, zero nearby | Increases low-freq. gain → reduces $e_{ss}$ |
| Lead-Lag | Combination | Improves both transient and steady-state |

## 2. Lead Compensator

### 2.1 Transfer Function

$$G_{\text{lead}}(s) = K_c \frac{s + z}{s + p} = K_c \frac{\tau s + 1}{\alpha\tau s + 1} \quad (p > z, \; 0 < \alpha < 1)$$

where:
- $z = 1/\tau$: zero location
- $p = 1/(\alpha\tau)$: pole location ($p > z$, so pole is farther left)
- $\alpha = z/p < 1$: determines the maximum phase lead

### 2.2 Frequency Response

The lead compensator provides positive phase between the zero and pole frequencies:

$$\phi_{\max} = \sin^{-1}\frac{1-\alpha}{1+\alpha}$$

at the geometric mean frequency:

$$\omega_{\max} = \frac{1}{\tau\sqrt{\alpha}} = \sqrt{zp}$$

**Design insight:** Smaller $\alpha$ gives more phase lead but also more gain increase at high frequencies (noise amplification). Practical limit: $\alpha \geq 0.05$ ($\phi_{\max} \leq 65°$). If more lead is needed, use two cascaded lead compensators.

### 2.3 Bode Design Procedure

**Given:** Plant $G_p(s)$, desired phase margin $PM_d$, and steady-state error requirement.

**Steps:**

1. **Set the gain** $K$ to satisfy the steady-state error requirement (using error constants $K_p$, $K_v$, $K_a$)
2. **Evaluate** the uncompensated system: find the current $PM$ with the gain $K$
3. **Determine required phase lead:** $\phi_{\max} = PM_d - PM_{\text{current}} + \text{margin}$ (add 5°-12° margin because the compensator shifts $\omega_{gc}$)
4. **Calculate** $\alpha = \frac{1 - \sin\phi_{\max}}{1 + \sin\phi_{\max}}$
5. **Find new $\omega_{gc}$**: the frequency where $|KG_p(j\omega)| = -10\log_{10}(1/\alpha)$ dB (so that after compensation, the gain is 0 dB at $\omega_{\max}$)
6. **Set** $\omega_{\max} = \omega_{gc,\text{new}}$ and compute $\tau = 1/(\omega_{\max}\sqrt{\alpha})$
7. **Compute** $K_c = K/\alpha$ (or adjust so the overall DC gain is correct)
8. **Verify** the design by plotting the compensated Bode plot

### 2.4 Example

**Plant:** $G_p(s) = \frac{1}{s(s+1)}$

**Specifications:** $K_v = 10$, $PM \geq 45°$

1. For $K_v = 10$: $K_v = \lim_{s\to 0} sKG_p(s) = K \Rightarrow K = 10$
2. Uncompensated: $KG_p(s) = \frac{10}{s(s+1)}$. At $\omega_{gc} \approx 2.9$ rad/s, $PM \approx 18°$
3. Required lead: $\phi_{\max} = 45° - 18° + 10° = 37°$
4. $\alpha = \frac{1 - \sin 37°}{1 + \sin 37°} = \frac{1 - 0.602}{1 + 0.602} = 0.249$
5. New $\omega_{gc}$: where $|10/(j\omega)(j\omega + 1)| = 10\log_{10}(1/0.249) = 6$ dB → $\omega_{gc} \approx 3.8$ rad/s
6. $\tau = 1/(3.8\sqrt{0.249}) = 0.527$ → zero at $1/\tau = 1.9$, pole at $1/(\alpha\tau) = 7.6$

Lead compensator: $G_c(s) = \frac{10}{0.249} \cdot \frac{0.527s + 1}{0.131s + 1} = 40.2\frac{s + 1.9}{s + 7.6}$

## 3. Lag Compensator

### 3.1 Transfer Function

$$G_{\text{lag}}(s) = K_c \frac{s + z}{s + p} = K_c \frac{\tau s + 1}{\beta\tau s + 1} \quad (z > p, \; \beta > 1)$$

where $z > p$ (zero is farther from origin than the pole), so $\beta = z/p > 1$.

### 3.2 Principle of Operation

The lag compensator works by:
- **Increasing the low-frequency gain** (below the zero frequency) by a factor of $\beta$
- **Not significantly affecting** the phase near the gain crossover frequency (the pole-zero pair is placed well below $\omega_{gc}$)
- The small phase lag at $\omega_{gc}$ is the price paid

### 3.3 Bode Design Procedure

1. **Set gain** $K$ so that the current $\omega_{gc}$ provides the desired $PM$ (ignoring $e_{ss}$)
2. **Determine** the additional gain needed: $\beta = \frac{K_{\text{required}}}{K_{\text{current}}}$ (ratio of gain needed for $e_{ss}$ to gain used for PM)
3. **Place the zero** at $\omega_z = \omega_{gc}/10$ (one decade below crossover, to minimize phase impact)
4. **Place the pole** at $\omega_p = \omega_z/\beta$
5. **Verify** PM and $e_{ss}$ of the compensated system

### 3.4 Example

**Plant:** $G_p(s) = \frac{1}{s(s+1)(s+2)}$

**Specifications:** $K_v = 5$, $PM \geq 40°$

1. For $PM = 40°$ without lag: at $\omega_{gc} = 0.5$, $PM \approx 63°$. The gain at $\omega = 0.5$ is $|G_p(j0.5)| = 0.89 \Rightarrow K = 1/0.89 = 1.12$ → $K_v = K/2 = 0.56$ (too small)
2. $\beta = 5/0.56 = 8.93$
3. Zero at $\omega_z = 0.5/10 = 0.05$ → $z = 0.05$
4. Pole at $\omega_p = 0.05/8.93 = 0.0056$ → $p = 0.0056$

Lag compensator: $G_c(s) = 1.12 \frac{s + 0.05}{s + 0.0056} = 1.12 \frac{20s + 1}{178.6s + 1}$

## 4. Lead-Lag Compensator

### 4.1 Structure

$$G_{c}(s) = K_c \frac{(s + z_1)(s + z_2)}{(s + p_1)(s + p_2)}$$

where $(s + z_1)/(s + p_1)$ is the lead part and $(s + z_2)/(s + p_2)$ is the lag part.

### 4.2 Design Strategy

1. Design the **lag** part first to meet the steady-state error requirement
2. Design the **lead** part to meet the phase margin requirement
3. Combine and verify

Alternatively:
1. Design **lead** first for desired PM
2. Design **lag** to boost low-frequency gain for $e_{ss}$
3. Iterate if the lag section's phase contribution affects the gain crossover

### 4.3 When to Use Each

| Specification | Compensator |
|--------------|-------------|
| Insufficient PM, adequate $e_{ss}$ | Lead |
| Adequate PM, insufficient $e_{ss}$ | Lag |
| Both PM and $e_{ss}$ need improvement | Lead-Lag |
| Very large PM improvement needed ($> 65°$) | Double lead or lead-lag |

## 5. Comparison with PID

The classical compensators are closely related to PID:

| PID Term | Compensator Equivalent |
|----------|----------------------|
| P | Gain adjustment |
| PI | Lag compensator (pole at origin, zero nearby) |
| PD | Lead compensator (zero, pole far away) |
| PID | Lead-lag with integrator |

**Advantages of lead-lag over PID:**
- More flexibility in pole/zero placement
- Better noise handling (lead has a finite high-frequency gain)
- Systematic frequency-domain design

**Advantages of PID:**
- Simpler to implement and tune in the field
- Well-established industrial tuning rules
- Easier to understand for non-specialists

## 6. Design Verification

After designing a compensator, verify by computing:

1. **Step response**: Check $M_p$, $t_s$, $t_r$, $e_{ss}$
2. **Bode plot**: Verify $GM$, $PM$, bandwidth
3. **Root locus**: Check closed-loop pole locations
4. **Sensitivity**: Ensure $|S(j\omega)| < M_s^{\max}$ at all frequencies
5. **Control effort**: Ensure $|u(t)| < u_{\max}$ (actuator saturation check)

## Practice Exercises

### Exercise 1: Lead Compensator Design

Design a lead compensator for $G_p(s) = \frac{5}{s(s+2)}$ to achieve $PM \geq 50°$ while maintaining $K_v = 10$.

1. Determine the required gain
2. Find the uncompensated phase margin
3. Design the lead compensator using the Bode method
4. Verify the phase margin of the compensated system

### Exercise 2: Lag Compensator Design

For $G_p(s) = \frac{1}{s(s+1)(s+5)}$, design a lag compensator such that $K_v = 10$ and $PM \geq 40°$.

### Exercise 3: Lead-Lag Design

A plant $G_p(s) = \frac{10}{(s+1)(s+5)}$ requires:
- Zero steady-state error for step input ($K_p = \infty$, need integrator)
- $PM \geq 50°$

Design a lead-lag compensator (with an integrator in the lag section) to meet both requirements.

---

*Previous: [Lesson 9 — PID Control](09_PID_Control.md) | Next: [Lesson 11 — State-Space Representation](11_State_Space_Representation.md)*
