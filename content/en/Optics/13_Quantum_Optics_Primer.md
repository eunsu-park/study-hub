# 13. Quantum Optics Primer

[← Previous: 12. Nonlinear Optics](12_Nonlinear_Optics.md) | [Next: 14. Computational Optics →](14_Computational_Optics.md)

---

For most of this course, we have treated light as a classical electromagnetic wave — and this description works beautifully for lasers, fiber optics, diffraction, and even nonlinear optics. But light is ultimately quantum mechanical, and there are phenomena that classical optics simply cannot explain: the granularity of photodetection (individual clicks of a photon counter), correlations between distant photons that violate classical probability theory, and noise levels below the classical vacuum limit.

Quantum optics is where Maxwell meets quantum mechanics. It provides the language for describing single photons, entangled photon pairs, and non-classical states of light that are reshaping technology through quantum communication, quantum computing, and quantum sensing. This lesson offers a primer — enough to understand the core concepts, appreciate the key experiments, and connect to the rapidly evolving field of quantum information.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Explain the quantization of the electromagnetic field and describe Fock (number) states
2. Define coherent states and show they correspond to classical laser light with Poissonian photon statistics
3. Distinguish Poissonian, sub-Poissonian, and super-Poissonian photon statistics and their physical origins
4. Describe single-photon sources and the Hong-Ou-Mandel effect as a signature of quantum interference
5. Explain squeezed states and their application in precision measurement beyond the standard quantum limit
6. Define Bell states and describe how entangled photon pairs are generated and tested
7. Outline the BB84 quantum key distribution protocol and the photonic approach to quantum computing

---

## Table of Contents

1. [Quantization of Light](#1-quantization-of-light)
2. [Fock States (Number States)](#2-fock-states-number-states)
3. [Coherent States](#3-coherent-states)
4. [Photon Statistics](#4-photon-statistics)
5. [Single-Photon Sources](#5-single-photon-sources)
6. [The Hong-Ou-Mandel Effect](#6-the-hong-ou-mandel-effect)
7. [Squeezed States](#7-squeezed-states)
8. [Entanglement and Bell States](#8-entanglement-and-bell-states)
9. [Quantum Key Distribution (BB84)](#9-quantum-key-distribution-bb84)
10. [Quantum Computing with Photons](#10-quantum-computing-with-photons)
11. [Python Examples](#11-python-examples)
12. [Summary](#12-summary)
13. [Exercises](#13-exercises)
14. [References](#14-references)

---

## 1. Quantization of Light

### 1.1 From Classical to Quantum

In classical electromagnetism, the energy of a light field can take any value — it is continuous. Planck (1900) and Einstein (1905) showed that energy exchange between light and matter occurs in discrete quanta $E = h\nu = \hbar\omega$. But quantum optics goes further: the electromagnetic field itself is quantized.

### 1.2 The Quantum Harmonic Oscillator

Each mode of the electromagnetic field (characterized by wavevector $\mathbf{k}$ and polarization $\hat{e}$) is mathematically equivalent to a quantum harmonic oscillator. The Hamiltonian for a single mode is:

$$\hat{H} = \hbar\omega\!\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right) = \hbar\omega\!\left(\hat{n} + \frac{1}{2}\right)$$

where:
- $\hat{a}^\dagger$ is the **creation operator** (adds one photon)
- $\hat{a}$ is the **annihilation operator** (removes one photon)
- $\hat{n} = \hat{a}^\dagger\hat{a}$ is the **number operator** (counts photons)
- $\frac{1}{2}\hbar\omega$ is the **zero-point energy** (vacuum fluctuations)

The commutation relation $[\hat{a}, \hat{a}^\dagger] = 1$ encodes the fundamental quantum nature of light.

### 1.3 The Electric Field Operator

The quantized electric field for a single mode is:

$$\hat{E}(z, t) = \mathcal{E}_0\!\left(\hat{a}\,e^{i(kz - \omega t)} + \hat{a}^\dagger\,e^{-i(kz - \omega t)}\right)$$

where $\mathcal{E}_0 = \sqrt{\hbar\omega/(2\epsilon_0 V)}$ is the **electric field per photon** — the field strength associated with a single quantum of excitation in a mode of volume $V$.

For a typical optical mode in a $1\,\text{cm}^3$ cavity at $\lambda = 500\,\text{nm}$:

$$\mathcal{E}_0 \approx \sqrt{\frac{(1.055 \times 10^{-34})(3.77 \times 10^{15})}{2(8.854 \times 10^{-12})(10^{-6})}} \approx 1.5\,\text{V/m}$$

This is an extraordinarily small field — but it represents the fundamental quantum of electromagnetic energy, and its effects are measurable with modern detectors.

> **Analogy**: Think of quantization like water flowing through a pipe. In the classical picture (a wide river), water flows smoothly and continuously. But zoom in to the molecular level, and you see individual water molecules. Similarly, a bright laser beam looks like a smooth electromagnetic wave, but at low intensities — or with sensitive enough detectors — you see individual photon "molecules." The creation and annihilation operators are like faucets that add or remove one photon at a time.

---

## 2. Fock States (Number States)

### 2.1 Definition

**Fock states** (or number states) $|n\rangle$ are eigenstates of the photon number operator:

$$\hat{n}|n\rangle = n|n\rangle, \quad n = 0, 1, 2, 3, \ldots$$

The state $|n\rangle$ contains exactly $n$ photons. The energy is:

$$E_n = \hbar\omega\!\left(n + \frac{1}{2}\right)$$

### 2.2 The Vacuum State

The ground state $|0\rangle$ is the **vacuum** — zero photons, but nonzero energy $E_0 = \hbar\omega/2$. This vacuum energy has real physical consequences:
- **Casimir effect**: Attractive force between parallel conducting plates due to modified vacuum modes
- **Spontaneous emission**: An excited atom decays because it couples to vacuum fluctuations
- **Lamb shift**: Small energy shift in hydrogen due to vacuum field fluctuations

### 2.3 Creation and Annihilation

$$\hat{a}^\dagger|n\rangle = \sqrt{n+1}\,|n+1\rangle \quad (\text{create one photon})$$

$$\hat{a}|n\rangle = \sqrt{n}\,|n-1\rangle \quad (\text{destroy one photon})$$

$$\hat{a}|0\rangle = 0 \quad (\text{cannot remove a photon from vacuum})$$

Fock states can be built from the vacuum: $|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle$.

### 2.4 Properties of Fock States

- **Definite photon number**: $\Delta n = 0$ — the number of photons is exactly known
- **Completely uncertain phase**: By the number-phase uncertainty relation $\Delta n \cdot \Delta\phi \geq 1/2$, a state with $\Delta n = 0$ has completely random phase
- **Non-classical**: Fock states with $n \geq 1$ cannot be described by any classical wave — they are genuinely quantum

### 2.5 Difficulty of Preparation

Fock states are extremely difficult to prepare. A single-photon state $|1\rangle$ is the most commonly achieved; states with $n \geq 2$ are much harder. This is a fundamental challenge in quantum optics.

---

## 3. Coherent States

### 3.1 Definition

**Coherent states** $|\alpha\rangle$, introduced by Glauber (Nobel Prize, 2005), are eigenstates of the annihilation operator:

$$\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$$

where $\alpha$ is a complex number. They are the quantum states that most closely resemble a classical electromagnetic wave.

### 3.2 Expansion in Fock States

$$|\alpha\rangle = e^{-|\alpha|^2/2}\sum_{n=0}^{\infty}\frac{\alpha^n}{\sqrt{n!}}|n\rangle$$

The probability of finding $n$ photons is:

$$P(n) = |\langle n|\alpha\rangle|^2 = \frac{|\alpha|^{2n}}{n!}e^{-|\alpha|^2}$$

This is a **Poisson distribution** with mean $\bar{n} = |\alpha|^2$.

### 3.3 Properties

- **Mean photon number**: $\langle\hat{n}\rangle = |\alpha|^2$
- **Photon number variance**: $\text{Var}(\hat{n}) = |\alpha|^2 = \bar{n}$
- **Fano factor**: $F = \text{Var}(n)/\bar{n} = 1$ (Poissonian)
- **Minimum uncertainty**: $\Delta X_1 \cdot \Delta X_2 = 1/4$ (equal uncertainty in both quadratures)
- **Not orthogonal**: $\langle\alpha|\beta\rangle = e^{-|\alpha-\beta|^2/2} \neq 0$ (but approximately orthogonal for $|\alpha - \beta| \gg 1$)

### 3.4 Why Lasers Produce Coherent States

A laser above threshold produces light that is well described by a coherent state. The gain saturation mechanism acts as a stabilizing feedback that maintains a definite amplitude and phase, while the quantum noise manifests as Poissonian fluctuations in photon number. This connection between laser physics (Lesson 8) and quantum field theory was one of Glauber's key insights.

### 3.5 The Displacement Operator

A coherent state is a displaced vacuum:

$$|\alpha\rangle = \hat{D}(\alpha)|0\rangle, \quad \hat{D}(\alpha) = e^{\alpha\hat{a}^\dagger - \alpha^*\hat{a}}$$

In phase space (the optical quadrature plane $X_1$-$X_2$), the vacuum state is a circular blob of uncertainty centered at the origin. A coherent state is the same blob displaced to the point $(\text{Re}(\alpha), \text{Im}(\alpha))$.

---

## 4. Photon Statistics

### 4.1 Three Regimes

The photon number distribution reveals the quantum nature of a light source. The **Fano factor** $F = \text{Var}(n)/\bar{n}$ classifies:

**Poissonian** ($F = 1$): Coherent light (laser). Photon arrivals are independent random events, like radioactive decay.

**Super-Poissonian** ($F > 1$): Thermal/chaotic light (incandescent bulb, LED). Photons tend to "bunch" — they arrive in clusters. This is the Hanbury Brown-Twiss (HBT) effect.

**Sub-Poissonian** ($F < 1$): Non-classical light (single-photon sources, squeezed light). Photons are more evenly spaced than random — they "antibunch." This has no classical explanation.

### 4.2 Thermal Light Statistics

For a single mode of thermal (blackbody) radiation at temperature $T$:

$$P(n) = \frac{\bar{n}^n}{(1+\bar{n})^{n+1}}, \quad \bar{n} = \frac{1}{e^{\hbar\omega/k_BT} - 1}$$

This is the **Bose-Einstein distribution**. The variance is:

$$\text{Var}(n) = \bar{n}^2 + \bar{n} = \bar{n}(\bar{n} + 1)$$

So $F = \bar{n} + 1 > 1$: super-Poissonian.

### 4.3 The Second-Order Correlation Function

The standard experimental measure of photon statistics is the **second-order correlation function** $g^{(2)}(\tau)$:

$$g^{(2)}(\tau) = \frac{\langle\hat{a}^\dagger(t)\hat{a}^\dagger(t+\tau)\hat{a}(t+\tau)\hat{a}(t)\rangle}{\langle\hat{a}^\dagger\hat{a}\rangle^2}$$

At zero delay:

| Source | $g^{(2)}(0)$ | Statistics |
|--------|---------------|------------|
| Coherent (laser) | 1 | Poissonian |
| Thermal | 2 | Super-Poissonian (bunching) |
| Single-photon | 0 | Sub-Poissonian (antibunching) |
| $n$-photon Fock | $1 - 1/n$ | Sub-Poissonian |

$g^{(2)}(0) < 1$ is impossible for any classical field — it is a definitive signature of quantum light.

### 4.4 Measurement: Hanbury Brown-Twiss Setup

```
                    50:50
   Source ─────── Beam ─────── Detector A
                  splitter
                    │
                    └────────── Detector B
                                    │
                          Coincidence counter
                          (measures g⁽²⁾(τ))
```

By measuring the rate of coincident detection events as a function of time delay $\tau$, we directly obtain $g^{(2)}(\tau)$.

---

## 5. Single-Photon Sources

### 5.1 Why Single Photons Matter

Single photons are the elementary carriers of quantum information. For quantum key distribution, quantum computing, and quantum networking, we need sources that emit exactly one photon at a time (on demand).

### 5.2 Types of Single-Photon Sources

**Attenuated laser**: Reduce laser intensity until $\bar{n} \ll 1$. The photon statistics remain Poissonian — there is always a probability of emitting 0 or 2 photons. Not a true single-photon source (still $g^{(2)}(0) = 1$), but simple and widely used.

**Heralded SPDC**: Spontaneous parametric down-conversion produces photon pairs. Detecting one photon "heralds" the presence of its partner. $g^{(2)}(0) \approx 0$ for the heralded photon. Probabilistic but well-characterized.

**Quantum dots**: Semiconductor nanostructures that behave as artificial atoms. Under pulsed excitation, they emit single photons with high purity ($g^{(2)}(0) < 0.01$) and can be deterministic. Leading platform for photonic quantum technology.

**Nitrogen-vacancy (NV) centers in diamond**: Atomic-scale defects that emit single photons at room temperature. $g^{(2)}(0) \sim 0.1$. Used in quantum sensing and communication.

**Trapped ions/atoms**: Excellent single-photon emitters with high indistinguishability, but complex setups.

### 5.3 Figures of Merit

- **Purity**: $g^{(2)}(0) \to 0$ (low multi-photon probability)
- **Indistinguishability**: Photons from successive emissions must be identical (same frequency, polarization, temporal profile). Measured via Hong-Ou-Mandel visibility.
- **Brightness**: High collection and extraction efficiency
- **Repetition rate**: Fast triggering for high data rates

---

## 6. The Hong-Ou-Mandel Effect

### 6.1 Two-Photon Interference

The Hong-Ou-Mandel (HOM) effect (1987) is one of the most striking demonstrations of quantum optics. When two **identical** single photons enter a 50:50 beam splitter from different ports, they **always exit together** from the same port — never one from each port.

```
   Photon 1 ──→ ┌────────┐ ──→ Both photons go to C or D
                 │  50:50  │
   Photon 2 ──→ │   BS    │ ──→ (but NEVER one to C and one to D)
                 └────────┘
```

### 6.2 Quantum Explanation

The beam splitter transforms the input modes ($\hat{a}_1, \hat{a}_2$) into output modes ($\hat{a}_3, \hat{a}_4$):

$$\hat{a}_3 = \frac{1}{\sqrt{2}}(\hat{a}_1 + i\hat{a}_2), \quad \hat{a}_4 = \frac{1}{\sqrt{2}}(i\hat{a}_1 + \hat{a}_2)$$

Input state: one photon in each port: $|1\rangle_1|1\rangle_2 = \hat{a}_1^\dagger\hat{a}_2^\dagger|0\rangle$.

Computing the output:

$$\hat{a}_1^\dagger\hat{a}_2^\dagger = \frac{1}{2}(\hat{a}_3^\dagger - i\hat{a}_4^\dagger)(-i\hat{a}_3^\dagger + \hat{a}_4^\dagger)$$

$$= \frac{1}{2}(-i\hat{a}_3^{\dagger2} + \hat{a}_3^\dagger\hat{a}_4^\dagger - i^2\hat{a}_4^\dagger\hat{a}_3^\dagger + (-i)\hat{a}_4^{\dagger2})$$

The cross terms $\hat{a}_3^\dagger\hat{a}_4^\dagger$ and $-\hat{a}_4^\dagger\hat{a}_3^\dagger$ cancel (since bosonic operators commute: $\hat{a}_3^\dagger\hat{a}_4^\dagger = \hat{a}_4^\dagger\hat{a}_3^\dagger$, and the signs are opposite). The output is:

$$\frac{i}{2}(-\hat{a}_3^{\dagger 2} + \hat{a}_4^{\dagger 2})|0\rangle = \frac{i}{\sqrt{2}}(|2,0\rangle - |0,2\rangle)$$

**Both photons exit the same port** — the coincidence rate drops to zero. This is called the **HOM dip**.

### 6.3 Physical Interpretation

The two-photon interference occurs because there are two indistinguishable paths to the same outcome: both photons reflected, or both transmitted. These two amplitudes have opposite signs (due to the $\pi/2$ phase shift upon reflection) and cancel perfectly — but only if the photons are truly identical (indistinguishable).

### 6.4 The HOM Dip

In practice, the two photons arrive with a relative time delay $\tau$. When $\tau = 0$ (perfect overlap), the coincidence rate drops to zero. As $|\tau|$ increases, the photons become distinguishable, and the coincidence rate recovers to the classical level. The width of the dip corresponds to the coherence time of the photons.

The **HOM visibility** $V = (C_{\max} - C_{\min})/C_{\max}$ measures the indistinguishability of the photon pair. $V = 1$ means perfectly indistinguishable; $V > 0.5$ is impossible with classical light.

---

## 7. Squeezed States

### 7.1 Quantum Noise and the Uncertainty Principle

The electromagnetic field can be decomposed into two **quadratures** (like the real and imaginary parts of the complex amplitude):

$$\hat{X}_1 = \frac{1}{2}(\hat{a} + \hat{a}^\dagger), \quad \hat{X}_2 = \frac{1}{2i}(\hat{a} - \hat{a}^\dagger)$$

The Heisenberg uncertainty principle requires:

$$\Delta X_1 \cdot \Delta X_2 \geq \frac{1}{4}$$

For a coherent state (and the vacuum), $\Delta X_1 = \Delta X_2 = 1/2$ — the **standard quantum limit (SQL)**.

### 7.2 Squeezing

A **squeezed state** has reduced uncertainty in one quadrature at the expense of increased uncertainty in the other:

$$\Delta X_1 = \frac{1}{2}e^{-r}, \quad \Delta X_2 = \frac{1}{2}e^{+r}$$

where $r > 0$ is the squeezing parameter. The uncertainty product remains at the minimum: $\Delta X_1 \cdot \Delta X_2 = 1/4$.

In phase space, the circular uncertainty blob of a coherent state is "squeezed" into an ellipse — hence the name.

### 7.3 Generation

Squeezed light is generated using parametric down-conversion (from Lesson 12) — specifically, a degenerate OPA (optical parametric amplifier) below threshold. The $\chi^{(2)}$ process correlates photon pairs, reducing fluctuations in one quadrature.

State-of-the-art squeezing: ~15 dB below the SQL (the noise in the squeezed quadrature is $10^{-1.5} \approx 3\%$ of the vacuum noise level).

### 7.4 Application: Gravitational Wave Detection

LIGO (Laser Interferometer Gravitational-Wave Observatory) uses squeezed vacuum states injected into its dark port to improve sensitivity beyond the shot noise limit. Since 2019, squeezed light has been routinely used in LIGO, improving the detection range for binary neutron star mergers by ~50%.

> **Analogy**: Imagine measuring the position of a marble on a table by looking at its shadow. The standard quantum limit is like a slightly blurry shadow — there is always some uncertainty in your measurement due to the wave nature of the photons you use to illuminate the marble. Squeezing is like reshaping the illumination beam so that the shadow is sharper in the horizontal direction (at the cost of being blurrier vertically). If you only care about horizontal position, squeezing gives you a more precise measurement than the standard limit allows.

---

## 8. Entanglement and Bell States

### 8.1 Quantum Entanglement

Two quantum systems are **entangled** when the quantum state of the combined system cannot be written as a product of individual states. For entangled photons:

$$|\Psi\rangle \neq |\psi_A\rangle \otimes |\psi_B\rangle$$

Entangled photons exhibit correlations stronger than any classical system can produce. Einstein famously called this "spooky action at a distance."

### 8.2 Bell States

The four **Bell states** are maximally entangled states of two qubits (e.g., two photon polarizations):

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|HH\rangle + |VV\rangle)$$

$$|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|HH\rangle - |VV\rangle)$$

$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|HV\rangle + |VH\rangle)$$

$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|HV\rangle - |VH\rangle)$$

where $|H\rangle$ and $|V\rangle$ are horizontal and vertical polarization.

### 8.3 Generation via SPDC

Type II SPDC naturally produces the state $|\Psi^-\rangle$: the signal and idler photons have orthogonal polarizations, and their exact assignment is quantum-mechanically undefined until measurement.

### 8.4 Bell's Theorem and Experimental Tests

Bell's inequality (1964) provides a quantitative test: any local hidden variable theory satisfies:

$$|S| \leq 2 \quad (\text{CHSH inequality})$$

Quantum mechanics predicts violations up to $|S| = 2\sqrt{2} \approx 2.83$.

Experiments by Aspect (1982), and more recently loophole-free tests by Hensen et al. (2015) and others, have conclusively demonstrated $|S| > 2$ — ruling out local hidden variable theories. Aspect, Clauser, and Zeilinger received the 2022 Nobel Prize in Physics for these experiments.

---

## 9. Quantum Key Distribution (BB84)

### 9.1 The Idea

BB84 (Bennett & Brassard, 1984) is a protocol for distributing a secret encryption key between two parties (Alice and Bob) with security guaranteed by quantum mechanics — not computational complexity.

### 9.2 Protocol

1. **Alice** sends single photons, each randomly prepared in one of four polarization states from two bases:
   - **Rectilinear basis** ($+$): $|H\rangle$ = "0", $|V\rangle$ = "1"
   - **Diagonal basis** ($\times$): $|D\rangle = |+45°\rangle$ = "0", $|A\rangle = |-45°\rangle$ = "1"

2. **Bob** randomly chooses a measurement basis ($+$ or $\times$) for each photon.

3. **Sifting**: Alice and Bob publicly compare their basis choices (not results). They keep only the bits where they used the same basis (~50% of bits).

4. **Error estimation**: They sacrifice a fraction of shared bits to estimate the error rate. If the error rate is too high, an eavesdropper (Eve) is detected.

5. **Privacy amplification**: Post-processing to distill a shorter, perfectly secret key.

### 9.3 Security

If Eve intercepts a photon and measures it, she must guess which basis Alice used. If she guesses wrong, her measurement disturbs the photon's state, introducing errors that Alice and Bob can detect. The **no-cloning theorem** guarantees that Eve cannot copy the quantum state without disturbing it.

### 9.4 Practical QKD

Commercial QKD systems operate over fiber (up to ~100 km without repeaters) and through free space (satellite QKD demonstrated by China's Micius satellite in 2017 over 1,200 km).

Key rates: typically kbit/s to Mbit/s over metropolitan distances.

---

## 10. Quantum Computing with Photons

### 10.1 Photonic Qubits

Photons can encode qubits in several degrees of freedom:
- **Polarization**: $|H\rangle$ and $|V\rangle$ as $|0\rangle$ and $|1\rangle$
- **Path (dual-rail)**: Photon in mode $a$ or mode $b$
- **Time-bin**: Early or late arrival time
- **Frequency**: Different spectral modes

### 10.2 Linear Optical Quantum Computing (LOQC)

Knill, Laflamme, and Milburn (2001) showed that universal quantum computing is possible using only:
- Single-photon sources
- Linear optical elements (beam splitters, phase shifters)
- Photon detectors
- Classical feedforward

The key insight: photon detection (a nonlinear operation) provides the nonlinearity needed for universal gates, through **measurement-induced nonlinearity**.

### 10.3 Boson Sampling

Boson sampling (Aaronson & Arkhipov, 2011) is a computational task that photonic systems can perform efficiently but classical computers (probably) cannot. Send $n$ identical photons into an $m$-mode interferometer and sample the output distribution. This was one of the first candidates for demonstrating quantum computational advantage.

In 2020, the Chinese experiment "Jiuzhang" demonstrated boson sampling with 76 detected photons — a task estimated to take classical supercomputers $10^{10}$ years.

### 10.4 Photonic Quantum Advantage

Advantages of photonic qubits:
- No decoherence at room temperature (photons do not interact with the environment)
- Natural carriers for quantum communication (compatible with fiber networks)
- High-speed operation (THz clock rates in principle)

Challenges:
- Photon loss (probabilistic detection)
- Lack of deterministic photon-photon interactions
- Need for ultra-efficient single-photon sources and detectors

Companies like PsiQuantum, Xanadu, and QuiX are building photonic quantum processors.

---

## 11. Python Examples

### 11.1 Photon Statistics Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

def poisson_dist(n, n_mean):
    """Poisson distribution: coherent light (laser)."""
    return n_mean**n * np.exp(-n_mean) / factorial(n, exact=False)

def bose_einstein_dist(n, n_mean):
    """Bose-Einstein distribution: thermal/chaotic light."""
    return n_mean**n / (1 + n_mean)**(n + 1)

def fock_dist(n, n_exact):
    """Fock (number) state: definite photon number."""
    return np.where(n == n_exact, 1.0, 0.0)

# Compare distributions for mean photon number = 5
n_mean = 5
n = np.arange(0, 20)

P_coherent = poisson_dist(n, n_mean)
P_thermal = bose_einstein_dist(n, n_mean)
P_fock = fock_dist(n, n_mean)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Coherent (Poisson)
axes[0].bar(n, P_coherent, color='steelblue', alpha=0.8, edgecolor='black')
axes[0].set_title(f'Coherent State |α|² = {n_mean}\n(Poissonian, F = 1)', fontsize=11)
axes[0].set_xlabel('Photon number n')
axes[0].set_ylabel('P(n)')
var_c = n_mean
axes[0].text(0.95, 0.95, f'⟨n⟩ = {n_mean}\nVar = {var_c}\nF = {var_c/n_mean:.1f}',
             transform=axes[0].transAxes, va='top', ha='right',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Thermal (Bose-Einstein)
axes[1].bar(n, P_thermal, color='indianred', alpha=0.8, edgecolor='black')
axes[1].set_title(f'Thermal Light ⟨n⟩ = {n_mean}\n(Super-Poissonian, F > 1)', fontsize=11)
axes[1].set_xlabel('Photon number n')
var_t = n_mean**2 + n_mean
axes[1].text(0.95, 0.95, f'⟨n⟩ = {n_mean}\nVar = {var_t}\nF = {var_t/n_mean:.1f}',
             transform=axes[1].transAxes, va='top', ha='right',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Fock state
axes[2].bar(n, P_fock, color='forestgreen', alpha=0.8, edgecolor='black')
axes[2].set_title(f'Fock State |n={n_mean}⟩\n(Sub-Poissonian, F = 0)', fontsize=11)
axes[2].set_xlabel('Photon number n')
axes[2].text(0.95, 0.95, f'⟨n⟩ = {n_mean}\nVar = 0\nF = 0',
             transform=axes[2].transAxes, va='top', ha='right',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

for ax in axes:
    ax.set_xlim(-0.5, 19.5)
    ax.set_ylim(0, max(P_thermal.max(), P_coherent.max(), 1.05) * 1.1)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Photon Number Distributions: Three Types of Light', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('photon_statistics.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 11.2 Hong-Ou-Mandel Dip Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

def hom_dip(tau, tau_c):
    """
    Simulate the Hong-Ou-Mandel dip.

    The coincidence rate at a 50:50 beam splitter drops to zero
    when two identical photons arrive simultaneously (τ=0).
    The dip width is determined by the photon coherence time τ_c.

    For distinguishable photons (|τ| >> τ_c), the coincidence rate
    is 0.5 (classical: each photon independently chooses an output port).
    For indistinguishable photons (τ → 0), quantum interference causes
    both photons to exit the same port, and coincidences vanish.
    """
    # Gaussian single-photon wavepackets
    # Overlap integral determines the visibility
    visibility = np.exp(-tau**2 / (2 * tau_c**2))

    # Classical coincidence rate = 0.5 (random 50:50 choice for each photon)
    # Quantum coincidence rate drops by the overlap squared
    coincidence = 0.5 * (1 - visibility**2)

    return coincidence

# Photon coherence time (related to bandwidth)
tau_c = 1e-12  # 1 ps (typical for SPDC photons with ~1 nm bandwidth)

tau = np.linspace(-5e-12, 5e-12, 1000)
R_coin = hom_dip(tau, tau_c)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(tau * 1e12, R_coin, 'b-', linewidth=2.5)
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5,
           label='Classical limit (distinguishable photons)')
ax.axhline(y=0.25, color='orange', linestyle=':', linewidth=1.5,
           label='50% visibility (partial indistinguishability)')

# Fill the dip region
ax.fill_between(tau * 1e12, R_coin, 0.5, alpha=0.15, color='blue')

ax.set_xlabel('Relative delay τ (ps)', fontsize=12)
ax.set_ylabel('Coincidence rate (normalized)', fontsize=12)
ax.set_title('Hong-Ou-Mandel Dip', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.02, 0.6)

# Annotate
ax.annotate('Quantum interference:\nboth photons exit\nthe same port',
            xy=(0, 0), xytext=(2, 0.15),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='blue'))

plt.tight_layout()
plt.savefig('hom_dip.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 11.3 BB84 Protocol Simulation

```python
import numpy as np

def bb84_simulation(n_bits, eve_present=False):
    """
    Simulate the BB84 quantum key distribution protocol.

    Alice prepares random polarization states, Bob measures in
    random bases. When their bases match, they share a bit.
    If Eve intercepts and measures (intercept-resend attack),
    she introduces ~25% errors in the sifted key — detectable
    by Alice and Bob through error rate estimation.
    """
    # Step 1: Alice prepares random bits and bases
    alice_bits = np.random.randint(0, 2, n_bits)
    alice_bases = np.random.randint(0, 2, n_bits)  # 0 = rectilinear, 1 = diagonal

    # Step 2: Eve intercepts (if present)
    if eve_present:
        eve_bases = np.random.randint(0, 2, n_bits)
        # Eve measures in her random basis
        # If Eve's basis matches Alice's, she gets the correct bit
        # If not, she gets a random result and disturbs the state
        eve_bits = np.where(
            eve_bases == alice_bases,
            alice_bits,  # Correct measurement
            np.random.randint(0, 2, n_bits)  # Random result
        )
        # Eve re-sends in her basis — this is the "resend" part
        # The state Eve sends may not match Alice's original state
        transmitted_bits = eve_bits
        transmitted_bases = eve_bases  # Eve's measurement collapses to her basis
    else:
        transmitted_bits = alice_bits
        transmitted_bases = alice_bases

    # Step 3: Bob measures in random bases
    bob_bases = np.random.randint(0, 2, n_bits)

    if eve_present:
        # Bob's result depends on whether his basis matches Eve's resend basis
        bob_bits = np.where(
            bob_bases == transmitted_bases,
            transmitted_bits,
            np.random.randint(0, 2, n_bits)
        )
    else:
        bob_bits = np.where(
            bob_bases == alice_bases,
            alice_bits,
            np.random.randint(0, 2, n_bits)
        )

    # Step 4: Sifting — keep only matching bases
    matching_bases = alice_bases == bob_bases
    sifted_alice = alice_bits[matching_bases]
    sifted_bob = bob_bits[matching_bases]

    # Step 5: Error estimation
    n_sifted = len(sifted_alice)
    if n_sifted > 0:
        errors = np.sum(sifted_alice != sifted_bob)
        error_rate = errors / n_sifted
    else:
        error_rate = 0

    return {
        'n_sent': n_bits,
        'n_sifted': n_sifted,
        'sifting_rate': n_sifted / n_bits,
        'error_rate': error_rate,
        'key_bits': sifted_alice[:20]  # First 20 bits of the key
    }

# Run simulations
np.random.seed(42)
n_bits = 10000

print("=" * 60)
print("BB84 Quantum Key Distribution Simulation")
print("=" * 60)

# Without eavesdropper
result_no_eve = bb84_simulation(n_bits, eve_present=False)
print(f"\n--- Without Eavesdropper ---")
print(f"Bits sent: {result_no_eve['n_sent']}")
print(f"Sifted key length: {result_no_eve['n_sifted']} ({result_no_eve['sifting_rate']:.1%})")
print(f"Error rate: {result_no_eve['error_rate']:.4f}")
print(f"First 20 key bits: {result_no_eve['key_bits']}")

# With eavesdropper
result_eve = bb84_simulation(n_bits, eve_present=True)
print(f"\n--- With Eavesdropper (intercept-resend) ---")
print(f"Bits sent: {result_eve['n_sent']}")
print(f"Sifted key length: {result_eve['n_sifted']} ({result_eve['sifting_rate']:.1%})")
print(f"Error rate: {result_eve['error_rate']:.4f}")
print(f"Expected error rate from Eve: ~0.25 (25%)")
print(f"First 20 key bits: {result_eve['key_bits']}")

# Security check
threshold = 0.11  # Typical threshold for BB84
print(f"\n--- Security Decision ---")
print(f"Error threshold: {threshold:.0%}")
for label, result in [("No Eve", result_no_eve), ("With Eve", result_eve)]:
    secure = "SECURE" if result['error_rate'] < threshold else "ABORT (eavesdropper detected!)"
    print(f"  {label}: error = {result['error_rate']:.4f} → {secure}")
```

---

## 12. Summary

| Concept | Key Formula / Idea |
|---------|--------------------|
| Field quantization | $\hat{H} = \hbar\omega(\hat{n} + 1/2)$; energy is discrete |
| Fock states | $\|n\rangle$: exactly $n$ photons; $\Delta n = 0$ |
| Coherent states | $\hat{a}\|\alpha\rangle = \alpha\|\alpha\rangle$; Poissonian statistics; classical-like |
| Photon statistics | Poissonian ($F=1$, laser), super-P ($F>1$, thermal), sub-P ($F<1$, quantum) |
| $g^{(2)}(0)$ | 1 (coherent), 2 (thermal), 0 (single photon) |
| Single-photon sources | Quantum dots, NV centers, heralded SPDC |
| HOM effect | Two identical photons at 50:50 BS always exit same port |
| Squeezed states | $\Delta X_1 < 1/2$ (below SQL), $\Delta X_1 \Delta X_2 = 1/4$ |
| Bell states | Maximally entangled: $\|\Phi^\pm\rangle$, $\|\Psi^\pm\rangle$ |
| BB84 | QKD protocol: security from no-cloning theorem |
| Photonic QC | Linear optics + single photons + detection = universal QC |

---

## 13. Exercises

### Exercise 1: Coherent State Properties

For a coherent state $|\alpha\rangle$ with $\alpha = 3 + 4i$:

(a) Calculate the mean photon number $\bar{n}$.
(b) Calculate the photon number standard deviation.
(c) What is the probability of detecting exactly 25 photons?
(d) Plot the photon number distribution for $n = 0$ to $n = 50$.

### Exercise 2: $g^{(2)}$ Measurement

A light source produces the following detection record at a Hanbury Brown-Twiss setup: in $T = 100\,\text{s}$, detector A registers $N_A = 5000$ counts, detector B registers $N_B = 4800$ counts, and $N_{AB} = 150$ coincidences are recorded within a coincidence window of $\Delta t = 10\,\text{ns}$.

(a) Estimate $g^{(2)}(0) \approx \frac{N_{AB} T}{N_A N_B \Delta t}$.
(b) Is this source classical or quantum?
(c) What type of source could produce this $g^{(2)}(0)$?

### Exercise 3: HOM Visibility

In a Hong-Ou-Mandel experiment, the photons have a spectral bandwidth of $\Delta\lambda = 2\,\text{nm}$ centered at $\lambda = 810\,\text{nm}$.

(a) Calculate the coherence time $\tau_c = \lambda^2/(c\Delta\lambda)$.
(b) What time delay $\tau$ reduces the HOM visibility to $1/e$ of its peak?
(c) If the measured visibility is 92%, what does this imply about the indistinguishability of the photon source?

### Exercise 4: BB84 Key Rate

A BB84 QKD system operates at 1 GHz pulse rate with the following parameters: source efficiency 0.5, fiber loss 0.2 dB/km over 50 km, detector efficiency 10%, dark count rate $10^{-6}$ per gate.

(a) Calculate the photon arrival rate at Bob's detector.
(b) Estimate the raw sifted key rate.
(c) At what distance does the key rate drop to zero (the dark counts dominate)?

### Exercise 5: Squeezing and LIGO

LIGO's arm length is $L = 4\,\text{km}$, operating at $\lambda = 1064\,\text{nm}$ with circulating power $P = 750\,\text{kW}$.

(a) What is the shot-noise-limited displacement sensitivity $\delta x_{\text{SQL}}$ for a measurement time of 1 ms?
(b) If 10 dB of squeezing is injected, what is the improved sensitivity?
(c) The smallest gravitational wave strain measured by LIGO is $h \sim 10^{-21}$. Convert this to a displacement $\delta L = hL/2$ and compare with your answer to (b).

---

## 14. References

1. Fox, M. (2006). *Quantum Optics: An Introduction*. Oxford University Press. — Excellent introductory text.
2. Gerry, C. C., & Knight, P. L. (2023). *Introductory Quantum Optics* (2nd ed.). Cambridge University Press.
3. Walls, D. F., & Milburn, G. J. (2008). *Quantum Optics* (2nd ed.). Springer.
4. Glauber, R. J. (1963). "The quantum theory of optical coherence." *Physical Review*, 130, 2529.
5. Hong, C. K., Ou, Z. Y., & Mandel, L. (1987). "Measurement of subpicosecond time intervals between two photons by interference." *Physical Review Letters*, 59, 2044.
6. Bennett, C. H., & Brassard, G. (1984). "Quantum cryptography: Public key distribution and coin tossing." *Proceedings of IEEE International Conference on Computers, Systems and Signal Processing*, 175-179.
7. Aspect, A. (2022). Nobel Lecture: "From Einstein, Bohr and Bell to Quantum Technologies."

---

[← Previous: 12. Nonlinear Optics](12_Nonlinear_Optics.md) | [Next: 14. Computational Optics →](14_Computational_Optics.md)
