# 13. Adaptive Filters

**Previous**: [12. Multirate Signal Processing](./12_Multirate_Signal_Processing.md) | **Next**: [14. Time-Frequency Analysis](./14_Time_Frequency_Analysis.md)

---

Adaptive filters are filters whose coefficients are adjusted automatically according to an optimization algorithm. Unlike fixed filters designed with complete a priori knowledge of signal and noise statistics, adaptive filters can operate in unknown or time-varying environments by continuously updating their parameters from data. They are the workhorses behind noise cancellation headphones, echo cancellation in phones, channel equalization in modems, and many other real-world systems.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: FIR/IIR filter design, linear algebra, basic optimization concepts

**Learning Objectives**:
- Derive the Wiener filter as the optimal MMSE linear filter
- Understand the method of steepest descent and its convergence properties
- Derive and implement the LMS algorithm and analyze its convergence behavior
- Implement the Normalized LMS (NLMS) algorithm for improved convergence
- Derive and implement the RLS algorithm using the matrix inversion lemma
- Compare LMS and RLS in terms of complexity, convergence, and tracking
- Apply adaptive filters to system identification, noise cancellation, echo cancellation, and equalization

---

## Table of Contents

1. [Why Adaptive Filtering?](#1-why-adaptive-filtering)
2. [The Wiener Filter: Optimal MMSE Solution](#2-the-wiener-filter-optimal-mmse-solution)
3. [Method of Steepest Descent](#3-method-of-steepest-descent)
4. [The LMS Algorithm](#4-the-lms-algorithm)
5. [LMS Convergence Analysis](#5-lms-convergence-analysis)
6. [Normalized LMS (NLMS)](#6-normalized-lms-nlms)
7. [The RLS Algorithm](#7-the-rls-algorithm)
8. [Comparison: LMS vs RLS](#8-comparison-lms-vs-rls)
9. [Application: System Identification](#9-application-system-identification)
10. [Application: Noise Cancellation](#10-application-noise-cancellation)
11. [Application: Echo Cancellation](#11-application-echo-cancellation)
12. [Application: Channel Equalization](#12-application-channel-equalization)
13. [Application: Adaptive Beamforming](#13-application-adaptive-beamforming)
14. [Python Implementation: Complete Adaptive Filtering Toolkit](#14-python-implementation-complete-adaptive-filtering-toolkit)
15. [Exercises](#15-exercises)
16. [Summary](#16-summary)
17. [References](#17-references)

---

## 1. Why Adaptive Filtering?

### 1.1 Limitations of Fixed Filters

Conventional FIR and IIR filters require complete knowledge of the signal and noise characteristics at design time. Consider the challenges when:

- **Statistics are unknown**: You cannot design an optimal filter if you don't know the spectral characteristics of the noise.
- **Statistics are time-varying**: A wireless channel changes as the transmitter or receiver moves. A filter designed for one channel realization becomes suboptimal moments later.
- **Real-time operation is required**: Some environments demand continuous adaptation without offline design phases.

### 1.2 The Adaptive Filtering Framework

An adaptive filter consists of two parts:

1. **A parameterized filter structure** (usually FIR): computes the output $y(n)$ from the input $x(n)$.
2. **An adaptation algorithm**: adjusts the filter coefficients $\mathbf{w}(n)$ to minimize some cost function.

```
                    ┌──────────────────────┐
     x(n) ────────▶│   Adaptive Filter    │────────▶ y(n)
                    │   w(n)               │
                    └──────────┬───────────┘
                               │
                               │  e(n) = d(n) - y(n)
                               │
     d(n) ─────────────────────┴─────────▶ Error
     (desired signal)                      Computation
                                              │
                                              ▼
                                     Adaptation Algorithm
                                     (update w(n+1))
```

The **error signal** is:

$$e(n) = d(n) - y(n) = d(n) - \mathbf{w}^T(n) \mathbf{x}(n)$$

where:
- $d(n)$ is the **desired (reference) signal**
- $\mathbf{x}(n) = [x(n), x(n-1), \ldots, x(n-M+1)]^T$ is the input vector
- $\mathbf{w}(n) = [w_0(n), w_1(n), \ldots, w_{M-1}(n)]^T$ is the filter weight vector
- $M$ is the filter order

### 1.3 Common Configurations

Adaptive filters are used in four primary configurations:

| Configuration | Input $x(n)$ | Desired $d(n)$ | Objective |
|---|---|---|---|
| System identification | Input to unknown system | Output of unknown system | Model the unknown system |
| Inverse modeling | Output of unknown system | Delayed input | Equalize the channel |
| Noise cancellation | Correlated noise reference | Signal + noise | Extract the signal |
| Prediction | Delayed version of signal | Current signal | Predict future values |

---

## 2. The Wiener Filter: Optimal MMSE Solution

### 2.1 Cost Function

The **minimum mean square error (MMSE)** criterion minimizes the expected squared error:

$$J(\mathbf{w}) = E\left[|e(n)|^2\right] = E\left[|d(n) - \mathbf{w}^T \mathbf{x}(n)|^2\right]$$

Expanding:

$$J(\mathbf{w}) = E[d^2(n)] - 2\mathbf{w}^T E[d(n)\mathbf{x}(n)] + \mathbf{w}^T E[\mathbf{x}(n)\mathbf{x}^T(n)] \mathbf{w}$$

Define:
- **Autocorrelation matrix**: $\mathbf{R} = E[\mathbf{x}(n)\mathbf{x}^T(n)]$ (an $M \times M$ positive-definite matrix)
- **Cross-correlation vector**: $\mathbf{p} = E[d(n)\mathbf{x}(n)]$ (an $M \times 1$ vector)
- $\sigma_d^2 = E[d^2(n)]$

The cost function becomes a **quadratic bowl**:

$$J(\mathbf{w}) = \sigma_d^2 - 2\mathbf{w}^T \mathbf{p} + \mathbf{w}^T \mathbf{R} \mathbf{w}$$

### 2.2 The Wiener-Hopf Equation

Taking the gradient and setting it to zero:

$$\nabla_{\mathbf{w}} J = -2\mathbf{p} + 2\mathbf{R}\mathbf{w} = \mathbf{0}$$

This yields the **Wiener-Hopf equation** (normal equation):

$$\boxed{\mathbf{R}\mathbf{w}_{opt} = \mathbf{p}}$$

The optimal (Wiener) filter is:

$$\mathbf{w}_{opt} = \mathbf{R}^{-1}\mathbf{p}$$

The **minimum MSE** at the optimal solution is:

$$J_{min} = \sigma_d^2 - \mathbf{p}^T \mathbf{R}^{-1} \mathbf{p}$$

### 2.3 Performance Surface

Since $\mathbf{R}$ is positive-definite, the cost function $J(\mathbf{w})$ is a convex quadratic -- a bowl-shaped surface (an elliptic paraboloid in the $M$-dimensional weight space). Any descent algorithm will converge to the unique global minimum.

Using the eigendecomposition $\mathbf{R} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$, the cost function in the rotated coordinate system $\mathbf{v} = \mathbf{Q}^T(\mathbf{w} - \mathbf{w}_{opt})$ becomes:

$$J(\mathbf{v}) = J_{min} + \sum_{k=0}^{M-1} \lambda_k v_k^2$$

where $\lambda_k$ are the eigenvalues of $\mathbf{R}$. The contours of $J$ are ellipses whose axes are aligned with the eigenvectors and whose extents are determined by the eigenvalues.

### 2.4 Limitations of the Wiener Solution

The Wiener filter requires:
1. Knowledge of $\mathbf{R}$ and $\mathbf{p}$ (second-order statistics)
2. Stationarity of the signals
3. Computation of $\mathbf{R}^{-1}$ ($O(M^3)$ operations)

In practice, these are rarely available exactly, motivating iterative and adaptive approaches.

---

## 3. Method of Steepest Descent

### 3.1 Gradient Descent on the MSE Surface

Instead of solving the Wiener-Hopf equation directly, we can reach $\mathbf{w}_{opt}$ iteratively using gradient descent:

$$\mathbf{w}(n+1) = \mathbf{w}(n) - \mu \nabla_{\mathbf{w}} J(n)$$

The true gradient of the MSE cost function is:

$$\nabla_{\mathbf{w}} J = -2\mathbf{p} + 2\mathbf{R}\mathbf{w}(n)$$

So the update rule is:

$$\boxed{\mathbf{w}(n+1) = \mathbf{w}(n) + 2\mu\left(\mathbf{p} - \mathbf{R}\mathbf{w}(n)\right)}$$

This is the **steepest descent** algorithm. Note it still requires knowledge of $\mathbf{R}$ and $\mathbf{p}$, so it is not truly adaptive yet.

### 3.2 Convergence Analysis

Define the weight error vector: $\boldsymbol{\epsilon}(n) = \mathbf{w}(n) - \mathbf{w}_{opt}$

Substituting into the update:

$$\boldsymbol{\epsilon}(n+1) = (\mathbf{I} - 2\mu\mathbf{R})\boldsymbol{\epsilon}(n)$$

Using the eigendecomposition $\mathbf{R} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$, in the rotated coordinates $\mathbf{v}(n) = \mathbf{Q}^T \boldsymbol{\epsilon}(n)$:

$$v_k(n+1) = (1 - 2\mu\lambda_k) v_k(n)$$

For convergence, we need $|1 - 2\mu\lambda_k| < 1$ for all $k$, which gives:

$$\boxed{0 < \mu < \frac{1}{\lambda_{max}}}$$

where $\lambda_{max}$ is the largest eigenvalue of $\mathbf{R}$.

### 3.3 Convergence Speed and Eigenvalue Spread

The rate of convergence of each mode $v_k$ is determined by $|1 - 2\mu\lambda_k|$. The optimal step size for each mode would be $\mu_k = 1/(2\lambda_k)$, but since we use a single $\mu$:

- The fastest-converging mode corresponds to $\lambda_{max}$
- The slowest-converging mode corresponds to $\lambda_{min}$

The **eigenvalue spread** (condition number):

$$\chi(\mathbf{R}) = \frac{\lambda_{max}}{\lambda_{min}}$$

governs the overall convergence rate. A large eigenvalue spread means slow convergence -- the algorithm "zig-zags" across the narrow valley of the performance surface.

### 3.4 Learning Curve

The MSE as a function of iteration is the **learning curve**:

$$J(n) = J_{min} + \sum_{k=0}^{M-1} \lambda_k v_k^2(0)(1 - 2\mu\lambda_k)^{2n}$$

Each mode decays geometrically with time constant:

$$\tau_k = \frac{-1}{2\ln|1 - 2\mu\lambda_k|} \approx \frac{1}{4\mu\lambda_k} \quad \text{for small } \mu$$

The slowest mode has time constant $\tau_{max} \approx 1/(4\mu\lambda_{min})$.

---

## 4. The LMS Algorithm

### 4.1 Derivation

The steepest descent algorithm requires the true gradient $\nabla J = -2\mathbf{p} + 2\mathbf{R}\mathbf{w}(n)$. The key insight of Widrow and Hoff (1960) is to replace the true gradient with an **instantaneous estimate**:

$$\hat{\nabla} J(n) = -2e(n)\mathbf{x}(n)$$

This is obtained by replacing the expectation with the instantaneous sample:
- $\mathbf{R}\mathbf{w}(n) \approx \mathbf{x}(n)\mathbf{x}^T(n)\mathbf{w}(n) = \mathbf{x}(n)y(n)$
- $\mathbf{p} \approx d(n)\mathbf{x}(n)$

The **LMS algorithm** is:

$$\boxed{\mathbf{w}(n+1) = \mathbf{w}(n) + \mu \, e(n) \, \mathbf{x}(n)}$$

where $e(n) = d(n) - \mathbf{w}^T(n)\mathbf{x}(n)$.

### 4.2 Algorithm Summary

```
LMS Algorithm
─────────────────────────────────────────────
Initialize: w(0) = 0 (or small random values)
Parameters: step size μ, filter order M

For each new sample n = 0, 1, 2, ...
  1. Form input vector: x(n) = [x(n), x(n-1), ..., x(n-M+1)]^T
  2. Compute output:    y(n) = w^T(n) x(n)
  3. Compute error:     e(n) = d(n) - y(n)
  4. Update weights:    w(n+1) = w(n) + μ e(n) x(n)
```

**Computational complexity**: $O(M)$ multiplications and additions per sample -- remarkably efficient.

### 4.3 Properties of LMS

1. **Simplicity**: No matrix inversions, no autocorrelation estimation
2. **Low complexity**: $2M$ multiplications per iteration
3. **Stochastic gradient**: The gradient estimate is noisy but unbiased: $E[\hat{\nabla}J] = \nabla J$
4. **Self-tuning**: Automatically adjusts to track slow changes in signal statistics

---

## 5. LMS Convergence Analysis

### 5.1 Mean Convergence

Taking expectations of the LMS update (under the **independence assumption** -- $\mathbf{x}(n)$ is independent of $\mathbf{w}(n)$):

$$E[\mathbf{w}(n+1)] = E[\mathbf{w}(n)] + \mu E[e(n)\mathbf{x}(n)]$$

After some algebra:

$$E[\boldsymbol{\epsilon}(n+1)] = (\mathbf{I} - 2\mu\mathbf{R}) E[\boldsymbol{\epsilon}(n)]$$

This is the same recurrence as steepest descent, so the **mean convergence** condition is:

$$0 < \mu < \frac{1}{\lambda_{max}}$$

In practice, we use:

$$0 < \mu < \frac{1}{\text{tr}(\mathbf{R})} = \frac{1}{M \cdot \sigma_x^2}$$

since $\text{tr}(\mathbf{R}) = \sum_k \lambda_k \geq \lambda_{max}$, and $\text{tr}(\mathbf{R}) = M\sigma_x^2$ for stationary inputs.

### 5.2 Mean-Square Convergence

The condition for the MSE to converge (mean-square stability) is more restrictive:

$$0 < \mu < \frac{2}{\lambda_{max} + \text{tr}(\mathbf{R})}$$

For practical purposes, a safe choice is:

$$\mu < \frac{1}{3 \, \text{tr}(\mathbf{R})} = \frac{1}{3M\sigma_x^2}$$

### 5.3 Excess MSE and Misadjustment

Even after convergence, the LMS algorithm does not reach $J_{min}$ because the stochastic gradient introduces noise into the weight updates. The **excess MSE** is:

$$J_{excess} = J_{steady-state} - J_{min}$$

The **misadjustment** is:

$$\mathcal{M} = \frac{J_{excess}}{J_{min}} \approx \mu \, \text{tr}(\mathbf{R}) = \mu M \sigma_x^2$$

This reveals a fundamental **tradeoff**:
- **Large $\mu$**: Fast convergence but large misadjustment (noisy steady state)
- **Small $\mu$**: Slow convergence but small misadjustment (accurate steady state)

### 5.4 Step Size Selection Guidelines

| Criterion | Step Size |
|---|---|
| Stability (mean) | $\mu < 1/\lambda_{max}$ |
| Stability (mean-square) | $\mu < 2/(\lambda_{max} + \text{tr}(\mathbf{R}))$ |
| Practical rule | $\mu \in [0.01, 0.1] / (M \sigma_x^2)$ |
| Misadjustment $\leq$ 10% | $\mu \leq 0.1 / (M \sigma_x^2)$ |

### 5.5 Convergence Time

The approximate time constant for the slowest mode is:

$$\tau_{mse} \approx \frac{1}{4\mu\lambda_{min}}$$

Combined with the misadjustment constraint $\mathcal{M} = \mu M \sigma_x^2$:

$$\tau_{mse} \approx \frac{M \sigma_x^2}{4\mathcal{M}\lambda_{min}} = \frac{\chi(\mathbf{R})}{4\mathcal{M}} \cdot \frac{M\sigma_x^2}{\lambda_{max}}$$

Large eigenvalue spread $\chi(\mathbf{R})$ requires many iterations for convergence at a given misadjustment.

---

## 6. Normalized LMS (NLMS)

### 6.1 Motivation

The standard LMS has a fixed step size $\mu$, which means the effective adaptation rate depends on the input power $\|\mathbf{x}(n)\|^2$. When the input power varies, LMS can become unstable or converge too slowly.

### 6.2 Derivation

The NLMS algorithm is obtained by normalizing the step size by the input power:

$$\boxed{\mathbf{w}(n+1) = \mathbf{w}(n) + \frac{\tilde{\mu}}{\|\mathbf{x}(n)\|^2 + \delta} \, e(n) \, \mathbf{x}(n)}$$

where:
- $\tilde{\mu} \in (0, 2)$ is the normalized step size
- $\delta > 0$ is a small regularization constant to prevent division by zero

### 6.3 Derivation from Constrained Optimization

NLMS can be derived by solving the constrained optimization problem:

$$\min_{\mathbf{w}(n+1)} \|\mathbf{w}(n+1) - \mathbf{w}(n)\|^2 \quad \text{subject to} \quad \mathbf{w}^T(n+1)\mathbf{x}(n) = d(n)$$

That is: find the closest weight vector to the current one that perfectly fits the latest data point. Using Lagrange multipliers, one obtains the NLMS update with $\tilde{\mu} = 1$.

### 6.4 Advantages of NLMS

1. **Robust convergence**: Step size automatically adapts to input power
2. **Simpler tuning**: Only one parameter $\tilde{\mu} \in (0, 2)$ to set
3. **Better for non-stationary inputs**: Works well with varying signal levels
4. **Minimal extra cost**: One additional inner product per iteration

### 6.5 NLMS Convergence

The convergence condition for NLMS is simply:

$$0 < \tilde{\mu} < 2$$

The misadjustment is approximately:

$$\mathcal{M}_{NLMS} \approx \frac{\tilde{\mu}}{2 - \tilde{\mu}} \cdot \frac{1}{M}$$

A typical choice is $\tilde{\mu} \in [0.1, 1.0]$.

---

## 7. The RLS Algorithm

### 7.1 Motivation

While LMS estimates the gradient stochastically (one sample at a time), the **Recursive Least Squares (RLS)** algorithm minimizes a deterministic cost function over all past data:

$$J_{RLS}(n) = \sum_{i=0}^{n} \lambda^{n-i} |e(i)|^2$$

where $\lambda \in (0, 1]$ is the **forgetting factor** (typically $0.95 \leq \lambda \leq 1.0$). Recent samples are weighted more heavily than older ones, providing tracking capability for non-stationary environments.

### 7.2 The Normal Equations for Weighted LS

The cost function is minimized by:

$$\mathbf{w}(n) = \boldsymbol{\Phi}^{-1}(n) \boldsymbol{\theta}(n)$$

where:
- $\boldsymbol{\Phi}(n) = \sum_{i=0}^{n} \lambda^{n-i} \mathbf{x}(i)\mathbf{x}^T(i)$ is the weighted sample correlation matrix
- $\boldsymbol{\theta}(n) = \sum_{i=0}^{n} \lambda^{n-i} d(i)\mathbf{x}(i)$ is the weighted cross-correlation vector

Both have recursive updates:

$$\boldsymbol{\Phi}(n) = \lambda \boldsymbol{\Phi}(n-1) + \mathbf{x}(n)\mathbf{x}^T(n)$$

$$\boldsymbol{\theta}(n) = \lambda \boldsymbol{\theta}(n-1) + d(n)\mathbf{x}(n)$$

### 7.3 Matrix Inversion Lemma

To avoid recomputing $\boldsymbol{\Phi}^{-1}(n)$ at each step ($O(M^3)$), we use the **matrix inversion lemma** (Woodbury identity):

$$(\mathbf{A} + \mathbf{u}\mathbf{v}^T)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{u}\mathbf{v}^T\mathbf{A}^{-1}}{1 + \mathbf{v}^T\mathbf{A}^{-1}\mathbf{u}}$$

Define $\mathbf{P}(n) = \boldsymbol{\Phi}^{-1}(n)$. Then:

$$\mathbf{P}(n) = \lambda^{-1}\mathbf{P}(n-1) - \lambda^{-1}\mathbf{k}(n)\mathbf{x}^T(n)\mathbf{P}(n-1)$$

where the **gain vector** is:

$$\mathbf{k}(n) = \frac{\lambda^{-1}\mathbf{P}(n-1)\mathbf{x}(n)}{1 + \lambda^{-1}\mathbf{x}^T(n)\mathbf{P}(n-1)\mathbf{x}(n)}$$

### 7.4 RLS Algorithm Summary

```
RLS Algorithm
─────────────────────────────────────────────
Initialize: w(0) = 0, P(0) = δ^{-1} I (δ small, e.g., 0.01)
Parameters: forgetting factor λ (e.g., 0.99), regularization δ

For each new sample n = 1, 2, ...
  1. Compute gain vector:
     k(n) = P(n-1) x(n) / [λ + x^T(n) P(n-1) x(n)]

  2. Compute a priori error:
     e(n) = d(n) - w^T(n-1) x(n)

  3. Update weights:
     w(n) = w(n-1) + k(n) e(n)

  4. Update inverse correlation matrix:
     P(n) = λ^{-1} [P(n-1) - k(n) x^T(n) P(n-1)]
```

**Computational complexity**: $O(M^2)$ per sample (due to the $\mathbf{P}$ matrix update).

### 7.5 Forgetting Factor

The forgetting factor $\lambda$ determines the **effective memory** of the algorithm:

$$N_{eff} = \frac{1}{1 - \lambda}$$

| $\lambda$ | $N_{eff}$ | Behavior |
|---|---|---|
| 1.0 | $\infty$ | Growing window (stationary environments) |
| 0.99 | 100 | Good for slowly varying statistics |
| 0.95 | 20 | Good for rapidly varying statistics |
| 0.9 | 10 | Very fast tracking, but noisy |

### 7.6 Properties of RLS

1. **Fast convergence**: Converges in approximately $2M$ iterations (independent of eigenvalue spread)
2. **No eigenvalue spread problem**: The $\mathbf{P}$ matrix whitens the input
3. **Higher complexity**: $O(M^2)$ vs $O(M)$ for LMS
4. **Numerical sensitivity**: The $\mathbf{P}$ matrix can lose positive-definiteness; stabilized versions exist (e.g., QR-RLS, lattice RLS)

---

## 8. Comparison: LMS vs RLS

| Feature | LMS | NLMS | RLS |
|---|---|---|---|
| Complexity per sample | $O(M)$ | $O(M)$ | $O(M^2)$ |
| Memory | $O(M)$ | $O(M)$ | $O(M^2)$ |
| Convergence speed | Slow (depends on $\chi$) | Moderate | Fast ($\sim 2M$ iterations) |
| Misadjustment | Higher | Moderate | Lower |
| Tracking ability | Moderate | Moderate | Good |
| Numerical stability | Excellent | Excellent | Can be poor |
| Eigenvalue spread sensitivity | High | Moderate | None |
| Step size parameter | $\mu$ (tricky to set) | $\tilde{\mu} \in (0,2)$ | $\lambda$ (easier to set) |

**Rule of thumb**: Use LMS/NLMS when computational cost is paramount or the filter is long. Use RLS when fast convergence is essential and the filter order is moderate.

---

## 9. Application: System Identification

### 9.1 Problem Statement

```
                    ┌────────────────────┐
     x(n) ────────▶│  Unknown System     │────────▶ d(n) = h*x(n) + v(n)
         │          │  h = [h0, h1, ...]  │
         │          └────────────────────┘
         │
         │          ┌────────────────────┐
         └────────▶│  Adaptive Filter   │────────▶ y(n) = w^T x(n)
                    │  w(n)              │
                    └────────────────────┘
                                                    e(n) = d(n) - y(n) → 0
```

The adaptive filter learns the impulse response of the unknown system. When the algorithm converges, $\mathbf{w}_{opt} \approx \mathbf{h}$.

### 9.2 When to Use

- **Plant modeling**: Control systems need a model of the system
- **Acoustic path identification**: Know the room impulse response
- **Adaptive inverse control**: Once you identify the forward model, invert it

---

## 10. Application: Noise Cancellation

### 10.1 The Adaptive Noise Canceller (ANC)

```
     Signal s(n) + Noise n0(n) = d(n)    (primary input)

     Noise reference n1(n)               (reference input, correlated with n0)
              │
              ▼
     ┌────────────────────┐
     │  Adaptive Filter   │ ──▶ ŷ(n) ≈ n0(n)
     │  w(n)              │
     └────────────────────┘
                                         e(n) = d(n) - ŷ(n) ≈ s(n)
```

**Key insight**: The reference input $n_1(n)$ is correlated with the noise $n_0(n)$ but uncorrelated with the signal $s(n)$. The adaptive filter transforms $n_1(n)$ into an estimate of $n_0(n)$. The error signal is then an estimate of the clean signal $s(n)$.

### 10.2 Mathematical Justification

The MSE is:

$$E[e^2(n)] = E[(s(n) + n_0(n) - \hat{y}(n))^2]$$

Since $s(n)$ is uncorrelated with both $n_0(n)$ and $n_1(n)$:

$$E[e^2(n)] = E[s^2(n)] + E[(n_0(n) - \hat{y}(n))^2]$$

Minimizing $E[e^2(n)]$ with respect to $\mathbf{w}$ minimizes $E[(n_0(n) - \hat{y}(n))^2]$, which means $\hat{y}(n) \to n_0(n)$ and $e(n) \to s(n)$.

**The signal is extracted as a byproduct of the noise estimation.**

---

## 11. Application: Echo Cancellation

### 11.1 Acoustic Echo Cancellation (AEC)

In speakerphone systems, the far-end speech is played through a loudspeaker, bounces around the room, and is picked up by the microphone. The adaptive filter models the acoustic path from loudspeaker to microphone.

```
Far-end ──▶ Loudspeaker ──▶ Room ──▶ Microphone ──▶ Near-end + Echo
  x(n)                   h(n)                        d(n) = s(n) + h*x(n)
    │
    │         ┌──────────────────┐
    └────────▶│  Adaptive Filter │──▶ ŷ(n) ≈ h*x(n)
              │  w(n) ≈ h        │
              └──────────────────┘
                                       e(n) = d(n) - ŷ(n) ≈ s(n)
```

**Challenges**:
- The acoustic impulse response can be very long (100-500 ms at 8 kHz = 800-4000 taps)
- Double-talk: both speakers active simultaneously
- Non-stationarity: people move, doors open

### 11.2 Network Echo Cancellation

In telephone networks, impedance mismatches at the hybrid (2-wire to 4-wire conversion) create electrical echoes. The echo path is shorter but the requirements are strict (>40 dB echo return loss enhancement).

---

## 12. Application: Channel Equalization

### 12.1 Problem

A transmitted signal $a(n)$ passes through a dispersive channel $c(n)$, producing inter-symbol interference (ISI):

$$x(n) = \sum_k c(k) a(n-k) + v(n)$$

The equalizer is an adaptive filter that undoes the channel distortion:

$$\hat{a}(n - \Delta) = \mathbf{w}^T(n) \mathbf{x}(n)$$

where $\Delta$ is a decision delay chosen so that $w(n) * c(n) \approx \delta(n - \Delta)$.

### 12.2 Training and Decision-Directed Modes

- **Training mode**: A known sequence is transmitted; $d(n) = a(n-\Delta)$
- **Decision-directed mode**: After initial convergence, use the slicer output $\hat{a}(n-\Delta)$ as $d(n)$

---

## 13. Application: Adaptive Beamforming

### 13.1 Problem

An array of $M$ sensors receives signals from multiple directions. The goal is to steer a beam toward the desired signal while nulling interferers.

The received signal at the array is:

$$\mathbf{x}(n) = s(n)\mathbf{a}(\theta_s) + \sum_{k=1}^{K} i_k(n)\mathbf{a}(\theta_k) + \mathbf{v}(n)$$

where $\mathbf{a}(\theta)$ is the **steering vector** for direction $\theta$.

### 13.2 Minimum Variance Distortionless Response (MVDR)

The Capon beamformer solves:

$$\min_{\mathbf{w}} \mathbf{w}^H \mathbf{R} \mathbf{w} \quad \text{subject to} \quad \mathbf{w}^H \mathbf{a}(\theta_s) = 1$$

Solution:

$$\mathbf{w}_{MVDR} = \frac{\mathbf{R}^{-1}\mathbf{a}(\theta_s)}{\mathbf{a}^H(\theta_s)\mathbf{R}^{-1}\mathbf{a}(\theta_s)}$$

Adaptive variants estimate $\mathbf{R}$ recursively using RLS-like updates.

---

## 14. Python Implementation: Complete Adaptive Filtering Toolkit

### 14.1 LMS, NLMS, and RLS Implementations

```python
import numpy as np
import matplotlib.pyplot as plt


def lms_filter(x, d, M, mu):
    """
    LMS adaptive filter.

    Parameters
    ----------
    x : ndarray
        Input signal
    d : ndarray
        Desired (reference) signal
    M : int
        Filter order (number of taps)
    mu : float
        Step size

    Returns
    -------
    y : ndarray
        Filter output
    e : ndarray
        Error signal
    w_history : ndarray
        Weight history (N x M)
    """
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    w_history = np.zeros((N, M))

    for n in range(M, N):
        x_vec = x[n:n-M:-1] if M > 1 else np.array([x[n]])
        # Proper construction of input vector
        x_vec = x[n-M+1:n+1][::-1]

        y[n] = np.dot(w, x_vec)
        e[n] = d[n] - y[n]
        w = w + mu * e[n] * x_vec
        w_history[n] = w

    return y, e, w_history


def nlms_filter(x, d, M, mu_tilde, delta=1e-6):
    """
    Normalized LMS adaptive filter.

    Parameters
    ----------
    x : ndarray
        Input signal
    d : ndarray
        Desired (reference) signal
    M : int
        Filter order
    mu_tilde : float
        Normalized step size (0 < mu_tilde < 2)
    delta : float
        Regularization constant

    Returns
    -------
    y, e, w_history : ndarrays
    """
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    w_history = np.zeros((N, M))

    for n in range(M, N):
        x_vec = x[n-M+1:n+1][::-1]

        y[n] = np.dot(w, x_vec)
        e[n] = d[n] - y[n]

        norm_sq = np.dot(x_vec, x_vec) + delta
        w = w + (mu_tilde / norm_sq) * e[n] * x_vec
        w_history[n] = w

    return y, e, w_history


def rls_filter(x, d, M, lam=0.99, delta=0.01):
    """
    Recursive Least Squares adaptive filter.

    Parameters
    ----------
    x : ndarray
        Input signal
    d : ndarray
        Desired (reference) signal
    M : int
        Filter order
    lam : float
        Forgetting factor (0 < lambda <= 1)
    delta : float
        Regularization for P initialization

    Returns
    -------
    y, e, w_history : ndarrays
    """
    N = len(x)
    w = np.zeros(M)
    P = (1.0 / delta) * np.eye(M)
    y = np.zeros(N)
    e = np.zeros(N)
    w_history = np.zeros((N, M))

    for n in range(M, N):
        x_vec = x[n-M+1:n+1][::-1]

        # Gain vector
        Px = P @ x_vec
        denom = lam + x_vec @ Px
        k = Px / denom

        # A priori error
        y[n] = np.dot(w, x_vec)
        e[n] = d[n] - y[n]

        # Weight update
        w = w + k * e[n]

        # Inverse correlation matrix update
        P = (1.0 / lam) * (P - np.outer(k, x_vec @ P))

        w_history[n] = w

    return y, e, w_history
```

### 14.2 System Identification Example

```python
# System Identification Demo
np.random.seed(42)

# Unknown system (FIR)
h_true = np.array([0.5, 1.2, -0.8, 0.3, -0.1])
M = len(h_true)

# Generate input signal (white noise)
N = 2000
x = np.random.randn(N)

# System output + measurement noise
d = np.convolve(x, h_true, mode='full')[:N] + 0.01 * np.random.randn(N)

# Run adaptive filters
mu_lms = 0.01
_, e_lms, w_lms = lms_filter(x, d, M, mu_lms)
_, e_nlms, w_nlms = nlms_filter(x, d, M, mu_tilde=0.5)
_, e_rls, w_rls = rls_filter(x, d, M, lam=0.99)

# Plot learning curves
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# MSE learning curves (smoothed)
window = 50
mse_lms = np.convolve(e_lms**2, np.ones(window)/window, mode='valid')
mse_nlms = np.convolve(e_nlms**2, np.ones(window)/window, mode='valid')
mse_rls = np.convolve(e_rls**2, np.ones(window)/window, mode='valid')

axes[0].semilogy(mse_lms, label='LMS', alpha=0.8)
axes[0].semilogy(mse_nlms, label='NLMS', alpha=0.8)
axes[0].semilogy(mse_rls, label='RLS', alpha=0.8)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('MSE')
axes[0].set_title('Learning Curves: System Identification')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Final weight comparison
x_pos = np.arange(M)
width = 0.2
axes[1].bar(x_pos - 1.5*width, h_true, width, label='True', color='black')
axes[1].bar(x_pos - 0.5*width, w_lms[-1], width, label='LMS')
axes[1].bar(x_pos + 0.5*width, w_nlms[-1], width, label='NLMS')
axes[1].bar(x_pos + 1.5*width, w_rls[-1], width, label='RLS')
axes[1].set_xlabel('Tap index')
axes[1].set_ylabel('Weight value')
axes[1].set_title('Identified Impulse Response')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('system_identification.png', dpi=150, bbox_inches='tight')
plt.show()

# Print final weights
print("True system:  ", h_true)
print("LMS weights:  ", np.round(w_lms[-1], 4))
print("NLMS weights: ", np.round(w_nlms[-1], 4))
print("RLS weights:  ", np.round(w_rls[-1], 4))
```

### 14.3 Noise Cancellation Demo

```python
# Adaptive Noise Cancellation Demo
np.random.seed(42)

N = 5000
t = np.arange(N) / 1000.0  # 1 kHz sampling rate

# Clean signal: sum of sinusoids
s = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# Noise source
noise_source = np.random.randn(N)

# Noise that corrupts the signal (filtered version of noise source)
noise_path = np.array([1.0, -0.5, 0.3, -0.1])
n0 = np.convolve(noise_source, noise_path, mode='full')[:N]

# Primary input: signal + noise
d = s + n0

# Reference input: correlated with noise but not with signal
# (different path from the noise source)
ref_path = np.array([0.8, -0.4, 0.2])
n1 = np.convolve(noise_source, ref_path, mode='full')[:N]

# Apply adaptive noise canceller
M = 8  # Filter order (longer than the noise path to be safe)
mu = 0.01

y_lms, e_lms, _ = lms_filter(n1, d, M, mu)
y_nlms, e_nlms, _ = nlms_filter(n1, d, M, mu_tilde=0.5)
y_rls, e_rls, _ = rls_filter(n1, d, M, lam=0.995)

# Plot results
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

axes[0].plot(t[:500], s[:500], 'g', linewidth=1.5, label='Clean signal')
axes[0].set_title('Original Clean Signal')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t[:500], d[:500], 'r', alpha=0.7, label='Signal + Noise')
axes[1].set_title('Noisy Signal (Primary Input)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t[:500], e_nlms[:500], 'b', alpha=0.7, label='NLMS output')
axes[2].plot(t[:500], s[:500], 'g--', alpha=0.5, label='Clean (reference)')
axes[2].set_title('Recovered Signal (NLMS Noise Canceller)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# SNR improvement over time
window = 200
snr_input = 10 * np.log10(
    np.convolve(s**2, np.ones(window)/window, mode='same') /
    np.convolve(n0**2, np.ones(window)/window, mode='same') + 1e-10
)
residual_nlms = e_nlms - s
snr_output = 10 * np.log10(
    np.convolve(s**2, np.ones(window)/window, mode='same') /
    np.convolve(residual_nlms**2, np.ones(window)/window, mode='same') + 1e-10
)

axes[3].plot(t, snr_input, 'r', alpha=0.7, label='Input SNR')
axes[3].plot(t, snr_output, 'b', alpha=0.7, label='Output SNR (NLMS)')
axes[3].set_xlabel('Time (s)')
axes[3].set_ylabel('SNR (dB)')
axes[3].set_title('SNR Improvement')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('noise_cancellation.png', dpi=150, bbox_inches='tight')
plt.show()

# Compute overall SNR improvement
snr_in = 10 * np.log10(np.mean(s[M:]**2) / np.mean(n0[M:]**2))
snr_out_nlms = 10 * np.log10(
    np.mean(s[1000:]**2) / np.mean((e_nlms[1000:] - s[1000:])**2)
)
print(f"Input SNR:       {snr_in:.1f} dB")
print(f"Output SNR (NLMS): {snr_out_nlms:.1f} dB")
print(f"SNR improvement:   {snr_out_nlms - snr_in:.1f} dB")
```

### 14.4 Tracking a Time-Varying System

```python
# Tracking a time-varying system
np.random.seed(42)

N = 4000
x = np.random.randn(N)

# Time-varying system: coefficients change at n=2000
h1 = np.array([1.0, 0.5, -0.3])
h2 = np.array([0.2, -0.8, 1.0])
M = 3

d = np.zeros(N)
for n in range(M, N):
    x_vec = x[n-M+1:n+1][::-1]
    if n < 2000:
        d[n] = np.dot(h1, x_vec) + 0.01 * np.random.randn()
    else:
        d[n] = np.dot(h2, x_vec) + 0.01 * np.random.randn()

# Compare algorithms
_, e_lms, w_lms = lms_filter(x, d, M, mu=0.05)
_, e_nlms, w_nlms = nlms_filter(x, d, M, mu_tilde=0.8)
_, e_rls, w_rls = rls_filter(x, d, M, lam=0.98)

# Plot weight trajectories
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

titles = ['LMS', 'NLMS', 'RLS']
w_histories = [w_lms, w_nlms, w_rls]
colors = ['tab:blue', 'tab:orange', 'tab:green']

for ax, title, w_hist in zip(axes, titles, w_histories):
    for i in range(M):
        ax.plot(w_hist[:, i], label=f'w[{i}]', alpha=0.8)
    # Plot true values
    ax.axhline(y=h1[0], color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=h1[1], color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=h1[2], color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=2000, color='red', linestyle='--', alpha=0.5, label='System change')
    ax.set_title(f'{title} Weight Tracking')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Iteration')
plt.tight_layout()
plt.savefig('tracking_demo.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 15. Exercises

### Exercise 1: Wiener Filter

Consider a system where $x(n)$ is white noise with variance $\sigma_x^2 = 1$ and the desired signal is $d(n) = 0.8x(n) + 0.5x(n-1) - 0.3x(n-2) + v(n)$, where $v(n)$ is white noise with variance $\sigma_v^2 = 0.1$, independent of $x(n)$.

(a) Compute the autocorrelation matrix $\mathbf{R}$ for a 3-tap Wiener filter.

(b) Compute the cross-correlation vector $\mathbf{p}$.

(c) Find the optimal Wiener filter $\mathbf{w}_{opt}$ by solving $\mathbf{R}\mathbf{w}_{opt} = \mathbf{p}$.

(d) Compute the minimum MSE $J_{min}$.

### Exercise 2: LMS Convergence

An LMS filter with $M = 10$ taps is applied to an input signal with autocorrelation matrix having eigenvalues $\lambda_{max} = 5.0$ and $\lambda_{min} = 0.1$.

(a) What is the maximum step size for mean convergence?

(b) What is the condition number $\chi(\mathbf{R})$?

(c) If $\mu = 0.01$, compute the misadjustment $\mathcal{M}$, given $\text{tr}(\mathbf{R}) = 10$.

(d) Estimate the convergence time constant $\tau_{mse}$ for the slowest mode.

(e) Explain qualitatively how the convergence would change if you whitened the input before applying LMS.

### Exercise 3: NLMS vs LMS

Implement both LMS and NLMS for noise cancellation with a non-stationary input signal whose power alternates between 0.1 and 10.0 every 500 samples. Use a filter order of $M = 16$.

(a) Show that LMS with a fixed step size either diverges during high-power segments or converges too slowly during low-power segments.

(b) Demonstrate that NLMS handles the power variation gracefully.

(c) Plot the MSE learning curves for both algorithms.

### Exercise 4: RLS Implementation

Implement RLS with forgetting factor $\lambda = 0.99$ for identifying a system with impulse response $h = [1, -0.5, 0.25, -0.125]$.

(a) Plot the convergence of each weight to its true value. Compare with LMS and NLMS.

(b) Vary $\lambda$ from 0.9 to 1.0 and plot the steady-state MSE vs convergence time tradeoff.

(c) Introduce a system change at $n = 1000$ (change $h$ to $[0.5, 0.3, -0.2, 0.1]$). Compare the tracking performance of LMS, NLMS, and RLS.

### Exercise 5: Echo Cancellation Simulation

Simulate an acoustic echo cancellation scenario:

(a) Generate a "far-end speech" signal as a sum of sinusoids with varying frequencies.

(b) Create a room impulse response (use an exponentially decaying random sequence of length 100).

(c) Add near-end noise.

(d) Apply NLMS with filter order 128. Plot the echo return loss enhancement (ERLE) over time:

$$\text{ERLE}(n) = 10 \log_{10} \frac{E[d^2(n)]}{E[e^2(n)]}$$

(e) Investigate the effect of double-talk (adding near-end speech) on the adaptive filter.

### Exercise 6: Adaptive Equalization

A digital communication channel has impulse response $c = [0.5, 1.0, 0.5]$ (introduces ISI).

(a) Generate a random BPSK signal ($a(n) \in \{-1, +1\}$) and pass it through the channel. Add noise at SNR = 20 dB.

(b) Design an adaptive equalizer using LMS with $M = 11$ taps and decision delay $\Delta = 5$.

(c) Plot the bit error rate (BER) as a function of training length.

(d) Switch to decision-directed mode after 500 training symbols and verify that the BER remains stable.

(e) Compare the eye diagram before and after equalization.

### Exercise 7: Effect of Filter Order

For the system identification problem with true system $h = [0.5, 1.2, -0.8, 0.3, -0.1]$:

(a) Run LMS with filter orders $M = 3, 5, 7, 10, 20$ and compare the steady-state MSE.

(b) Explain what happens when $M < 5$ (under-modeling) and $M > 5$ (over-modeling).

(c) Plot the identified impulse responses for each $M$.

---

## 16. Summary

| Concept | Key Formula / Idea |
|---|---|
| Wiener filter | $\mathbf{w}_{opt} = \mathbf{R}^{-1}\mathbf{p}$ (optimal MMSE) |
| Steepest descent | $\mathbf{w}(n+1) = \mathbf{w}(n) + 2\mu(\mathbf{p} - \mathbf{R}\mathbf{w}(n))$ |
| Convergence condition | $0 < \mu < 1/\lambda_{max}$ |
| LMS update | $\mathbf{w}(n+1) = \mathbf{w}(n) + \mu \, e(n) \, \mathbf{x}(n)$ |
| LMS misadjustment | $\mathcal{M} = \mu \, \text{tr}(\mathbf{R})$ |
| NLMS update | $\mathbf{w}(n+1) = \mathbf{w}(n) + \frac{\tilde{\mu}}{\|\mathbf{x}\|^2+\delta} e(n)\mathbf{x}(n)$ |
| RLS gain | $\mathbf{k}(n) = \frac{\mathbf{P}(n-1)\mathbf{x}(n)}{\lambda + \mathbf{x}^T(n)\mathbf{P}(n-1)\mathbf{x}(n)}$ |
| Forgetting factor memory | $N_{eff} = 1/(1-\lambda)$ |
| Noise cancellation | Error signal $e(n) = d(n) - \hat{y}(n) \approx s(n)$ |
| Tradeoff | Fast convergence vs low misadjustment |

**Key takeaways**:
1. The Wiener filter provides the theoretical optimum but requires known statistics.
2. LMS approximates the gradient with instantaneous estimates -- simple, robust, $O(M)$.
3. NLMS normalizes by input power -- better stability with varying signal levels.
4. RLS uses all past data with exponential weighting -- fast convergence at $O(M^2)$ cost.
5. The misadjustment-convergence tradeoff is fundamental to all adaptive algorithms.
6. Adaptive filters power numerous applications from noise cancellation to equalization.

---

## 17. References

1. S. Haykin, *Adaptive Filter Theory*, 5th ed., Pearson, 2014.
2. A.H. Sayed, *Adaptive Filters*, Wiley-IEEE Press, 2008.
3. P.S.R. Diniz, *Adaptive Filtering: Algorithms and Practical Implementation*, 4th ed., Springer, 2013.
4. B. Widrow and S.D. Stearns, *Adaptive Signal Processing*, Pearson, 1985.
5. S. Haykin, "Adaptive filter theory," in *Proc. IEEE*, vol. 90, no. 2, pp. 211-259, 2002.
6. B. Farhang-Boroujeny, *Adaptive Filters: Theory and Applications*, 2nd ed., Wiley, 2013.

---

**Previous**: [12. Multirate Signal Processing](./12_Multirate_Signal_Processing.md) | **Next**: [14. Time-Frequency Analysis](./14_Time_Frequency_Analysis.md)
