# Control Theory

## Overview

This topic covers the analysis and design of feedback control systems — from classical transfer-function methods to modern state-space techniques. Control theory provides the mathematical framework for making systems behave as desired: tracking references, rejecting disturbances, and maintaining stability.

## Prerequisites

- Calculus and linear algebra (matrix operations, eigenvalues)
- Ordinary differential equations (Mathematical_Methods L09-L10)
- Laplace transforms (Mathematical_Methods L15)
- Recommended: LTI systems and frequency concepts (Signal_Processing L02, L07)

## Learning Path

### Part I: Foundations (L01-L03)

| Lesson | Title | Key Concepts |
|--------|-------|--------------|
| 01 | Introduction to Control Systems | Open-loop vs. closed-loop, feedback, block diagrams, historical context |
| 02 | Mathematical Modeling of Physical Systems | Spring-mass-damper, DC motor, thermal systems, linearization |
| 03 | Transfer Functions and Block Diagrams | Laplace-domain representation, poles/zeros, block diagram algebra |

### Part II: Classical Analysis (L04-L06)

| Lesson | Title | Key Concepts |
|--------|-------|--------------|
| 04 | Time-Domain Analysis | Step/impulse response, 2nd-order specifications, steady-state error |
| 05 | Stability Analysis | Routh-Hurwitz criterion, BIBO stability, relative stability |
| 06 | Root Locus Method | Construction rules, gain selection, design via root locus |

### Part III: Frequency-Domain Methods (L07-L08)

| Lesson | Title | Key Concepts |
|--------|-------|--------------|
| 07 | Frequency Response — Bode Plots | Magnitude/phase plots, asymptotic approximation, system identification |
| 08 | Nyquist Stability and Robustness | Nyquist criterion, gain and phase margins, sensitivity functions |

### Part IV: Controller Design (L09-L10)

| Lesson | Title | Key Concepts |
|--------|-------|--------------|
| 09 | PID Control | P/PI/PD/PID actions, Ziegler-Nichols tuning, anti-windup, practical guidelines |
| 10 | Compensation Design | Lead, lag, and lead-lag compensators, frequency-domain design |

### Part V: Modern Control (L11-L14)

| Lesson | Title | Key Concepts |
|--------|-------|--------------|
| 11 | State-Space Representation | State variables, state equations, conversion to/from transfer functions |
| 12 | State-Space Analysis | Controllability, observability, canonical forms, minimal realizations |
| 13 | State Feedback and Observer Design | Pole placement, Luenberger observer, separation principle |
| 14 | Optimal Control | Linear Quadratic Regulator (LQR), Kalman filter, LQG |

### Part VI: Extensions (L15-L16)

| Lesson | Title | Key Concepts |
|--------|-------|--------------|
| 15 | Digital Control Systems | Sampling, zero-order hold, z-domain analysis, discrete PID |
| 16 | Nonlinear Control and Advanced Topics | Linearization, Lyapunov stability, phase portraits, model predictive control |

## Connections to Other Topics

- **Mathematical_Methods**: ODE solutions, Laplace transforms, phase-plane analysis (L09-L10, L15)
- **Signal_Processing**: LTI systems, frequency response, Z-transforms (L02, L07-L10)
- **Numerical_Simulation**: ODE solvers for simulating controlled systems (L04-L06)
- **Reinforcement_Learning**: MDP/optimal control duality, model-based RL (L03, L13)
- **Math_for_AI**: Optimization theory connects to optimal control (L05-L07)

## Example Code

Executable Python examples are in [`examples/Control_Theory/`](../../../examples/Control_Theory/). They use `numpy`, `scipy`, `matplotlib`, and the `control` library for transfer function manipulation, simulation, and controller design.

## References

- *Modern Control Engineering* by Katsuhiko Ogata (primary reference)
- *Modern Control Systems* by Richard C. Dorf and Robert H. Bishop
- *Feedback Control of Dynamic Systems* by Gene F. Franklin, J. David Powell, Abbas Emami-Naeini
- *Linear Systems Theory* by João P. Hespanha
