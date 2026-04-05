# 14. Time-Frequency Analysis

**Previous**: [13. Adaptive Filters](./13_Adaptive_Filters.md) | **Next**: [15. Image Signal Processing](./15_Image_Signal_Processing.md)

---

The Fourier transform reveals what frequencies are present in a signal, but not when they occur. For non-stationary signals -- music, speech, seismic events, biological rhythms -- we need tools that describe how spectral content evolves over time. This lesson develops the two principal frameworks for time-frequency analysis: the Short-Time Fourier Transform (STFT) and the Wavelet Transform, each with fundamentally different approaches to resolving the tension between time and frequency localization.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: DFT/FFT, windowing, convolution, basic linear algebra

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why the Fourier transform is inadequate for non-stationary signals
2. Derive and compute the STFT and understand the time-frequency resolution tradeoff
3. State and interpret the Heisenberg uncertainty principle for signals
4. Compute and interpret spectrograms
5. Understand the Wigner-Ville distribution and its properties
6. Define and compute the Continuous Wavelet Transform (CWT) using standard mother wavelets
7. Explain multiresolution analysis and implement the Discrete Wavelet Transform (DWT)
8. Apply Mallat's algorithm for wavelet decomposition and reconstruction
9. Compare STFT and wavelet approaches for real-world signal analysis

---

## Table of Contents

1. [Limitations of the Fourier Transform](#1-limitations-of-the-fourier-transform)
2. [Short-Time Fourier Transform (STFT)](#2-short-time-fourier-transform-stft)
3. [Time-Frequency Resolution and the Uncertainty Principle](#3-time-frequency-resolution-and-the-uncertainty-principle)
4. [The Spectrogram](#4-the-spectrogram)
5. [Window Selection for STFT](#5-window-selection-for-stft)
6. [Wigner-Ville Distribution](#6-wigner-ville-distribution)
7. [Introduction to Wavelet Analysis](#7-introduction-to-wavelet-analysis)
8. [Continuous Wavelet Transform (CWT)](#8-continuous-wavelet-transform-cwt)
9. [Mother Wavelets](#9-mother-wavelets)
10. [Multiresolution Analysis (MRA)](#10-multiresolution-analysis-mra)
11. [Discrete Wavelet Transform (DWT)](#11-discrete-wavelet-transform-dwt)
12. [Mallat's Algorithm](#12-mallats-algorithm)
13. [Wavelet Decomposition and Reconstruction](#13-wavelet-decomposition-and-reconstruction)
14. [STFT vs Wavelet Transform](#14-stft-vs-wavelet-transform)
15. [Applications](#15-applications)
16. [Python Implementation](#16-python-implementation)
17. [Exercises](#17-exercises)
18. [Summary](#18-summary)
19. [References](#19-references)

---

## 1. Limitations of the Fourier Transform

### 1.1 The Problem with Global Frequency Analysis

The Fourier transform computes frequency content over the entire duration of a signal:

$$X(f) = \int_{-\infty}^{\infty} x(t) \, e^{-j2\pi ft} \, dt$$

Consider two signals:
1. A constant 440 Hz tone for 2 seconds
2. A signal that plays 220 Hz for the first second, then 440 Hz for the next second

Both signals have the same Fourier magnitude spectrum (they contain the same frequencies for the same total duration), but they are perceptually and physically very different. The Fourier transform discards all temporal information about **when** each frequency occurs.

### 1.2 Non-Stationary Signals

A signal is **stationary** if its statistical properties do not change over time. Most real-world signals are non-stationary:

- **Speech**: Phonemes change rapidly (10-50 ms segments)
- **Music**: Notes, chords, and dynamics evolve continuously
- **Seismic signals**: P-waves, S-waves, and surface waves arrive at different times with different frequency content
- **Biomedical**: Heart rhythms, brain waves, and muscle signals vary with physiological state
- **Radar/Sonar**: Chirp signals have instantaneous frequency that changes with time

For these signals, we need a **joint time-frequency representation**.

### 1.3 The Instantaneous Frequency Concept

For a signal of the form $x(t) = A(t)\cos(\phi(t))$, the **instantaneous frequency** is:

$$f_i(t) = \frac{1}{2\pi}\frac{d\phi(t)}{dt}$$

For a linear chirp $x(t) = \cos(2\pi(f_0 t + \frac{1}{2}\beta t^2))$, the instantaneous frequency is $f_i(t) = f_0 + \beta t$. The Fourier transform cannot capture this time-varying nature.

---

## 2. Short-Time Fourier Transform (STFT)

### 2.1 Definition

The STFT localizes the Fourier analysis in time by applying a window function $w(t)$ centered at time $\tau$:

$$\boxed{X_{STFT}(\tau, f) = \int_{-\infty}^{\infty} x(t) \, w(t - \tau) \, e^{-j2\pi ft} \, dt}$$

The idea is simple: multiply the signal by a short window centered at $\tau$ to extract a local segment, then take its Fourier transform. By sliding $\tau$ across the signal, we get a sequence of local spectra.

### 2.2 Discrete STFT

For a discrete-time signal $x[n]$ with window $w[n]$ of length $L$:

$$X_{STFT}[m, k] = \sum_{n=0}^{L-1} x[n + mH] \, w[n] \, e^{-j2\pi kn/N}$$

where:
- $m$ is the time frame index
- $k$ is the frequency bin index
- $H$ is the hop size (stride between successive windows)
- $N$ is the FFT size (zero-padded if $N > L$)

### 2.3 Interpretation

The STFT maps a 1D signal into a 2D time-frequency plane. At each time instant $\tau$, we get a full spectrum. The STFT is a function of two variables: time $\tau$ and frequency $f$.

```
Frequency ▲
           │  ┌──────┐
           │  │      │   (high frequency events)
           │  │      │
           │  └──────┘
           │           ┌──────────────┐
           │           │              │  (mid frequency events)
           │           └──────────────┘
           │  ┌──────────┐
           │  │          │               (low frequency events)
           │  └──────────┘
           └──────────────────────────────────▶ Time
```

### 2.4 Overlap and Hop Size

Common choices for the hop size $H$ relative to window length $L$:
- **$H = L$**: No overlap, may miss events at window boundaries
- **$H = L/2$**: 50% overlap, good tradeoff (most common)
- **$H = L/4$**: 75% overlap, smoother time evolution, more computation

For perfect reconstruction (important in synthesis applications), the window and hop size must satisfy the **constant overlap-add (COLA) condition**:

$$\sum_m w[n - mH] = \text{constant} \quad \forall n$$

---

## 3. Time-Frequency Resolution and the Uncertainty Principle

### 3.1 Time and Frequency Resolution

The STFT's time and frequency resolution are determined by the window:

- **Time resolution**: $\Delta t \approx$ effective width of $w(t)$
- **Frequency resolution**: $\Delta f \approx$ effective bandwidth of $W(f)$ (the Fourier transform of the window)

A short window gives good time resolution but poor frequency resolution. A long window gives good frequency resolution but poor time resolution.

### 3.2 The Heisenberg-Gabor Uncertainty Principle

For any window function $w(t)$, the time-bandwidth product satisfies:

$$\boxed{\Delta t \cdot \Delta f \geq \frac{1}{4\pi}}$$

where the spreads are defined as:

$$\Delta t^2 = \frac{\int (t - \bar{t})^2 |w(t)|^2 \, dt}{\int |w(t)|^2 \, dt}, \qquad \Delta f^2 = \frac{\int (f - \bar{f})^2 |W(f)|^2 \, df}{\int |W(f)|^2 \, df}$$

This is a fundamental limit: **no window can achieve arbitrary precision in both time and frequency simultaneously.**

The **Gaussian window** $w(t) = e^{-\alpha t^2}$ achieves equality (minimum uncertainty). The resulting STFT is called the **Gabor transform**.

### 3.3 Tiles in the Time-Frequency Plane

The STFT partitions the time-frequency plane into **tiles** of fixed size $\Delta t \times \Delta f$:

```
STFT Tiling (fixed resolution):

Frequency ▲
           │ ┌──┬──┬──┬──┬──┬──┬──┬──┐
           │ │  │  │  │  │  │  │  │  │
           │ ├──┼──┼──┼──┼──┼──┼──┼──┤  ← Same Δt × Δf everywhere
           │ │  │  │  │  │  │  │  │  │
           │ ├──┼──┼──┼──┼──┼──┼──┼──┤
           │ │  │  │  │  │  │  │  │  │
           │ ├──┼──┼──┼──┼──┼──┼──┼──┤
           │ │  │  │  │  │  │  │  │  │
           │ └──┴──┴──┴──┴──┴──┴──┴──┘
           └──────────────────────────▶ Time
```

This fixed tiling is the fundamental limitation of the STFT: the resolution is the same everywhere. For many signals, we want better time resolution at high frequencies and better frequency resolution at low frequencies -- this is exactly what wavelets provide.

---

## 4. The Spectrogram

### 4.1 Definition

The **spectrogram** is the squared magnitude of the STFT:

$$S(\tau, f) = |X_{STFT}(\tau, f)|^2$$

It represents the **power spectral density** localized in time. The spectrogram is always real and non-negative.

### 4.2 Properties

1. **Energy conservation** (Moyal's formula):
$$\int\int |X_{STFT}(\tau,f)|^2 \, d\tau \, df = \int |x(t)|^2 \, dt$$

2. **Time marginal**: $\int S(\tau,f)\,df$ gives the local energy around time $\tau$ (depends on window)

3. **Frequency marginal**: $\int S(\tau,f)\,d\tau$ gives the energy spectral density (smeared by window)

### 4.3 Spectrogram as a Smoothed Wigner-Ville Distribution

The spectrogram can be shown to be a 2D smoothing of the Wigner-Ville distribution (Section 6):

$$S_x(\tau, f) = \iint W_x(t, \nu) \, W_w(\tau - t, f - \nu) \, dt \, d\nu$$

where $W_x$ is the Wigner-Ville distribution of $x$ and $W_w$ is the Wigner-Ville distribution of the window. The window acts as a 2D smoothing kernel, reducing cross-term interference at the cost of resolution.

---

## 5. Window Selection for STFT

### 5.1 Window Properties

The choice of window affects both resolution and spectral leakage:

| Window | Mainlobe Width (-3dB) | Sidelobe Level | Best For |
|---|---|---|---|
| Rectangular | $0.89/L$ | -13 dB | Maximum frequency resolution |
| Hann | $1.44/L$ | -31 dB | General purpose |
| Hamming | $1.33/L$ | -43 dB | Spectral analysis |
| Blackman | $1.68/L$ | -58 dB | Low sidelobe requirements |
| Gaussian | $\sim 1.2/L$ | varies with $\sigma$ | Optimal time-frequency product |
| Kaiser | adjustable | adjustable | Flexible design |

### 5.2 Practical Guidelines

- **Speech**: Window length 20-40 ms (captures pitch periods), Hamming window, 50% overlap
- **Music**: Window length 50-100 ms (better frequency resolution for musical notes), Hann window
- **Transient detection**: Short windows (1-5 ms) for better time resolution
- **General rule**: Match the window length to the time scale of the features you want to resolve

### 5.3 Zero-Padding

Zero-padding the windowed segment before the FFT (using $N > L$) does **not** improve frequency resolution but does provide interpolation in the frequency domain, making the spectrogram appear smoother. The true frequency resolution is always determined by the window length $L$.

---

## 6. Wigner-Ville Distribution

### 6.1 Definition

The **Wigner-Ville distribution (WVD)** is a quadratic time-frequency distribution:

$$W_x(t, f) = \int_{-\infty}^{\infty} x\!\left(t + \frac{\tau}{2}\right) x^*\!\left(t - \frac{\tau}{2}\right) e^{-j2\pi f\tau} \, d\tau$$

### 6.2 Properties

1. **Excellent resolution**: The WVD achieves the best possible time-frequency concentration
2. **Correct marginals**:
   - $\int W_x(t,f)\,df = |x(t)|^2$ (instantaneous power)
   - $\int W_x(t,f)\,dt = |X(f)|^2$ (power spectral density)
3. **Instantaneous frequency**: For a mono-component signal, the first moment in frequency gives the instantaneous frequency

### 6.3 The Cross-Term Problem

The WVD is a **bilinear** (quadratic) representation. For a signal with two components $x = x_1 + x_2$:

$$W_x = W_{x_1} + W_{x_2} + 2\Re\{W_{x_1,x_2}\}$$

The cross-term $W_{x_1,x_2}$ creates **oscillating interference patterns** between the true signal components. For $N$ components, there are $N(N-1)/2$ cross-terms, which can overwhelm the true signal content.

### 6.4 Cohen's Class

The **Cohen's class** of distributions provides a general framework for time-frequency representations that satisfy the correct marginal properties:

$$C_x(t, f) = \iiint e^{-j2\pi(\theta\tau + f\nu - \theta u)} \Phi(\theta, \tau) x\!\left(u+\frac{\tau}{2}\right) x^*\!\left(u-\frac{\tau}{2}\right) du \, d\tau \, d\theta$$

where $\Phi(\theta, \tau)$ is the **kernel function**. Different kernels yield different distributions:
- $\Phi = 1$: Wigner-Ville distribution
- $\Phi = $ Gaussian: Smoothed pseudo-Wigner-Ville (reduces cross-terms)
- $\Phi = e^{-\sigma^2 \theta^2 \tau^2}$: Choi-Williams distribution

---

## 7. Introduction to Wavelet Analysis

### 7.1 Motivation: Variable Resolution

The STFT uses a fixed window, giving the same resolution at all frequencies. But many signals have:
- **High-frequency events** that are short in time (transients, clicks, edges)
- **Low-frequency events** that are long in time (trends, fundamental tones)

We want:
- **Short windows** (good time resolution) at **high frequencies**
- **Long windows** (good frequency resolution) at **low frequencies**

This is exactly what the wavelet transform provides.

### 7.2 From STFT to Wavelets

In the STFT, we modulate a fixed window to different frequencies:

$$\text{STFT}: \quad g_{\tau,f}(t) = w(t-\tau) \, e^{j2\pi ft}$$

In the wavelet transform, we **scale** and **translate** a single prototype function (the mother wavelet):

$$\text{CWT}: \quad \psi_{a,b}(t) = \frac{1}{\sqrt{|a|}} \psi\!\left(\frac{t-b}{a}\right)$$

- **Translation** $b$: localizes in time
- **Scale** $a$: controls the time-frequency resolution
  - Small $a$: compressed wavelet (short in time, captures high frequencies)
  - Large $a$: stretched wavelet (long in time, captures low frequencies)

### 7.3 Wavelet Tiling

```
Wavelet Tiling (variable resolution):

Frequency ▲
           │ ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐
           │ ├┴┬┴┬┴┬┴┬┴┬┴┬┴┬┴┬┴┬┴┬┴┬┴┬┘  ← High freq: narrow Δt, wide Δf
           │ ├──┬──┬──┬──┬──┬──┬──┬──┤
           │ │  │  │  │  │  │  │  │  │     ← Mid freq
           │ ├────┬────┬────┬────┤
           │ │    │    │    │    │          ← Low freq: wide Δt, narrow Δf
           │ ├────────┬────────┤
           │ │        │        │
           │ └────────┴────────┘
           └──────────────────────────▶ Time
```

Each tile has the same area $\Delta t \cdot \Delta f = \text{const}$, but the aspect ratio varies: tall and narrow at high frequencies, short and wide at low frequencies.

---

## 8. Continuous Wavelet Transform (CWT)

### 8.1 Definition

The **Continuous Wavelet Transform** of $x(t)$ with mother wavelet $\psi(t)$ is:

$$\boxed{W_x(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \, \psi^*\!\left(\frac{t-b}{a}\right) dt}$$

where:
- $a > 0$ is the **scale** parameter (inversely related to frequency)
- $b$ is the **translation** (time shift) parameter
- $\psi^*$ is the complex conjugate of the mother wavelet
- $1/\sqrt{|a|}$ normalizes the energy across scales

### 8.2 Relationship Between Scale and Frequency

For a mother wavelet with center frequency $f_c$, the pseudo-frequency at scale $a$ is:

$$f_a = \frac{f_c}{a}$$

So **large scale** corresponds to **low frequency** and **small scale** to **high frequency**.

### 8.3 Admissibility Condition

For the CWT to be invertible, the mother wavelet must satisfy the **admissibility condition**:

$$C_\psi = \int_0^{\infty} \frac{|\hat{\Psi}(f)|^2}{f} \, df < \infty$$

where $\hat{\Psi}(f)$ is the Fourier transform of $\psi(t)$. This requires $\hat{\Psi}(0) = 0$, meaning the wavelet must have **zero mean**:

$$\int_{-\infty}^{\infty} \psi(t) \, dt = 0$$

### 8.4 Inverse CWT

The signal can be reconstructed from its CWT:

$$x(t) = \frac{1}{C_\psi} \int_0^{\infty} \int_{-\infty}^{\infty} W_x(a,b) \, \frac{1}{\sqrt{a}} \psi\!\left(\frac{t-b}{a}\right) \frac{db \, da}{a^2}$$

### 8.5 Scalogram

The **scalogram** is the wavelet analogue of the spectrogram:

$$\text{Scalogram}(a, b) = |W_x(a, b)|^2$$

---

## 9. Mother Wavelets

### 9.1 Haar Wavelet

The simplest wavelet, proposed by Alfred Haar in 1909:

$$\psi_{Haar}(t) = \begin{cases} 1 & 0 \leq t < 1/2 \\ -1 & 1/2 \leq t < 1 \\ 0 & \text{otherwise} \end{cases}$$

**Properties**:
- Compact support (finite duration)
- Discontinuous (poor frequency localization)
- One vanishing moment: $\int t^0 \psi(t)\,dt = 0$
- Good for detecting sharp transitions and edges

### 9.2 Morlet Wavelet

A complex wavelet consisting of a complex exponential modulated by a Gaussian:

$$\psi_{Morlet}(t) = C_\sigma \, \pi^{-1/4} \, e^{j\omega_0 t} \, e^{-t^2/2}$$

where $\omega_0$ (typically $\omega_0 \approx 5$ or $2\pi$) is the center frequency and $C_\sigma$ is a normalization constant.

**Properties**:
- Complex-valued (provides both amplitude and phase)
- Infinite support (but effectively finite due to Gaussian decay)
- Excellent time-frequency localization (Gaussian envelope)
- No compact support, no orthogonality
- Most commonly used for continuous wavelet analysis

### 9.3 Mexican Hat Wavelet (Ricker Wavelet)

The negative normalized second derivative of a Gaussian:

$$\psi_{mhat}(t) = \frac{2}{\sqrt{3}\pi^{1/4}} (1 - t^2) \, e^{-t^2/2}$$

**Properties**:
- Real-valued, symmetric
- Two vanishing moments: $\int t^k \psi(t)\,dt = 0$ for $k = 0, 1$
- Related to the Laplacian of Gaussian (LoG) used in image processing
- Good for detecting peaks and valleys

### 9.4 Daubechies Wavelets

Ingrid Daubechies (1988) constructed a family of wavelets $\psi_{Db-N}$ with $N$ vanishing moments and compact support of length $2N - 1$:

- **Db1** = Haar wavelet
- **Db2**: Support length 3, 2 vanishing moments
- **Db4**: Support length 7, 4 vanishing moments (commonly used)
- **Db10**: Support length 19, 10 vanishing moments

**Properties**:
- Orthogonal and compactly supported
- No closed-form expression (defined by filter coefficients)
- Higher $N$: smoother wavelet, longer support, better frequency localization
- The only orthogonal wavelets with compact support and maximum vanishing moments

### 9.5 Other Important Wavelets

| Wavelet | Type | Key Feature |
|---|---|---|
| Symlets (sym$N$) | Orthogonal | Near-symmetric version of Daubechies |
| Coiflets (coif$N$) | Orthogonal | Near-symmetric, vanishing moments in both $\psi$ and $\phi$ |
| Meyer | Orthogonal | Defined in frequency domain, infinitely smooth |
| Shannon | Orthogonal | Ideal bandpass, sinc-based |
| Gabor | Non-orthogonal | Gaussian-modulated sinusoid |
| Paul | Complex | Analytic, good time localization |

---

## 10. Multiresolution Analysis (MRA)

### 10.1 Concept

Multiresolution analysis, introduced by Mallat (1989) and Meyer, provides the mathematical framework for the DWT. The idea is to decompose a signal into successive **approximations** at different resolutions, with **detail** captured at each level.

Think of looking at a landscape:
- **Coarse resolution**: See mountains and valleys
- **Medium resolution**: See individual trees and buildings
- **Fine resolution**: See leaves and bricks

### 10.2 Nested Approximation Spaces

An MRA consists of a nested sequence of closed subspaces of $L^2(\mathbb{R})$:

$$\cdots \subset V_{-2} \subset V_{-1} \subset V_0 \subset V_1 \subset V_2 \subset \cdots$$

satisfying:
1. **Nesting**: $V_j \subset V_{j+1}$
2. **Density**: $\overline{\bigcup_j V_j} = L^2(\mathbb{R})$
3. **Separation**: $\bigcap_j V_j = \{0\}$
4. **Scaling**: $f(t) \in V_j \Leftrightarrow f(2t) \in V_{j+1}$
5. **Shift invariance**: $f(t) \in V_0 \Leftrightarrow f(t-k) \in V_0$ for all $k \in \mathbb{Z}$
6. **Riesz basis**: There exists a **scaling function** $\phi(t)$ such that $\{\phi(t-k)\}_{k \in \mathbb{Z}}$ forms a Riesz basis for $V_0$

### 10.3 Scaling Function and Wavelet Function

The **scaling function** (father wavelet) $\phi(t)$ satisfies the **refinement equation** (dilation equation):

$$\phi(t) = \sqrt{2} \sum_k h[k] \, \phi(2t - k)$$

where $h[k]$ are the **scaling coefficients** (lowpass filter).

The **wavelet function** $\psi(t)$ is defined by:

$$\psi(t) = \sqrt{2} \sum_k g[k] \, \phi(2t - k)$$

where $g[k] = (-1)^k h[1-k]$ are the **wavelet coefficients** (highpass filter, related to $h$ by the quadrature mirror relationship).

### 10.4 Detail Spaces

The **detail space** $W_j$ is the orthogonal complement of $V_j$ in $V_{j+1}$:

$$V_{j+1} = V_j \oplus W_j$$

The wavelets at scale $j$ span $W_j$:

$$W_j = \text{span}\{\psi_{j,k}(t) = 2^{j/2}\psi(2^j t - k)\}_{k \in \mathbb{Z}}$$

Signal decomposition:

$$V_J = V_0 \oplus W_0 \oplus W_1 \oplus \cdots \oplus W_{J-1}$$

$$f(t) = \sum_k c_{0,k}\phi_{0,k}(t) + \sum_{j=0}^{J-1}\sum_k d_{j,k}\psi_{j,k}(t)$$

where $c_{j,k}$ are **approximation coefficients** and $d_{j,k}$ are **detail coefficients**.

---

## 11. Discrete Wavelet Transform (DWT)

### 11.1 Dyadic Sampling

The CWT is highly redundant: it computes coefficients at all scales $a$ and translations $b$. The DWT samples on a **dyadic grid**:

$$a = 2^j, \quad b = k \cdot 2^j$$

giving:

$$W_x(j, k) = 2^{-j/2} \int x(t) \, \psi(2^{-j}t - k) \, dt$$

### 11.2 Connection to Filter Banks

The DWT can be computed efficiently using a pair of filters:
- **Lowpass filter** $h[n]$: associated with the scaling function $\phi$
- **Highpass filter** $g[n]$: associated with the wavelet $\psi$

These form a **quadrature mirror filter (QMF)** bank:

$$g[n] = (-1)^n h[1-n]$$

**Perfect reconstruction** conditions:
$$H(z)H(z^{-1}) + H(-z)H(-z^{-1}) = 2$$

### 11.3 Analysis and Synthesis

**Analysis** (decomposition):
- Approximation coefficients: $c_{j+1}[k] = \sum_n h[n-2k] \, c_j[n]$ (lowpass filter + downsample by 2)
- Detail coefficients: $d_{j+1}[k] = \sum_n g[n-2k] \, c_j[n]$ (highpass filter + downsample by 2)

**Synthesis** (reconstruction):
- $c_j[n] = \sum_k c_{j+1}[k] \, h[n-2k] + \sum_k d_{j+1}[k] \, g[n-2k]$ (upsample by 2 + filter + sum)

---

## 12. Mallat's Algorithm

### 12.1 The Fast Wavelet Transform

Mallat's algorithm computes the DWT using iterated filter banks, analogous to how the FFT computes the DFT. The computational cost is $O(N)$ -- even faster than the FFT!

```
Mallat's Algorithm (Analysis/Decomposition):

Level 0:  c₀[n] = x[n]  (original signal, N samples)
              │
              ├──── h[n] ──▶ ↓2 ──▶ c₁[k]  (approximation, N/2 samples)
              │                        │
              │                        ├──── h[n] ──▶ ↓2 ──▶ c₂[k]  (N/4)
              │                        │                        │
              │                        │                        ├──── h[n]──▶↓2──▶ c₃
              │                        │                        │
              │                        │                        └──── g[n]──▶↓2──▶ d₃
              │                        │
              │                        └──── g[n] ──▶ ↓2 ──▶ d₂[k]  (detail, N/4)
              │
              └──── g[n] ──▶ ↓2 ──▶ d₁[k]  (detail, N/2 samples)
```

### 12.2 Reconstruction (Synthesis)

```
Mallat's Algorithm (Synthesis/Reconstruction):

c₃ ──▶ ↑2 ──▶ h̃[n] ──┐
                         ├──▶ + ──▶ c₂ ──▶ ↑2 ──▶ h̃[n] ──┐
d₃ ──▶ ↑2 ──▶ g̃[n] ──┘                                     ├──▶ + ──▶ c₁ ──▶ ...
                                   d₂ ──▶ ↑2 ──▶ g̃[n] ──┘
```

### 12.3 Computational Complexity

For a signal of length $N$ and $J$ decomposition levels:
- Each level: $O(N/2^j)$ operations (filter + downsample)
- Total: $O(N) + O(N/2) + O(N/4) + \cdots = O(2N) = O(N)$

Compare with:
- FFT: $O(N \log N)$
- STFT: $O(N \cdot L \cdot \log L)$ where $L$ is the window length

---

## 13. Wavelet Decomposition and Reconstruction

### 13.1 Multi-Level Decomposition

At decomposition level $J$, the signal is represented as:

$$x[n] = \underbrace{c_J[k]}_{\text{coarse approx.}} + \underbrace{d_J[k]}_{\text{finest detail lost}} + d_{J-1}[k] + \cdots + d_1[k]$$

Each level of detail coefficients captures a specific frequency band:
- $d_1$: highest frequencies ($f_s/4$ to $f_s/2$)
- $d_2$: next band ($f_s/8$ to $f_s/4$)
- $d_j$: band ($f_s/2^{j+1}$ to $f_s/2^j$)
- $c_J$: lowest frequencies (0 to $f_s/2^{J+1}$)

### 13.2 Wavelet Denoising (Thresholding)

One of the most powerful applications of the DWT is **denoising by thresholding**:

1. Compute the DWT of the noisy signal
2. Apply a threshold to the detail coefficients
3. Reconstruct using the inverse DWT

**Hard thresholding**:
$$\hat{d}[k] = \begin{cases} d[k] & |d[k]| \geq \lambda \\ 0 & |d[k]| < \lambda \end{cases}$$

**Soft thresholding** (shrinkage):
$$\hat{d}[k] = \text{sgn}(d[k]) \max(|d[k]| - \lambda, 0)$$

**Universal threshold** (Donoho and Johnstone, 1994):
$$\lambda = \sigma \sqrt{2 \ln N}$$

where $\sigma$ is the noise standard deviation, estimated from the finest detail level:
$$\hat{\sigma} = \frac{\text{median}(|d_1|)}{0.6745}$$

### 13.3 Choosing the Number of Levels

The maximum number of decomposition levels for a signal of length $N$ with filter length $L$ is:

$$J_{max} = \lfloor \log_2(N / (L-1)) \rfloor$$

In practice, 3-6 levels suffice for most applications.

---

## 14. STFT vs Wavelet Transform

### 14.1 Resolution Comparison

| Feature | STFT | Wavelet Transform |
|---|---|---|
| Time-frequency tiling | Uniform rectangles | Variable (dyadic) |
| Low-frequency resolution | Same as high freq | Better frequency resolution |
| High-frequency resolution | Same as low freq | Better time resolution |
| Basis functions | Modulated windows | Scaled/translated wavelets |
| Computation (discrete) | $O(N \log N)$ per frame | $O(N)$ total |
| Redundancy (continuous) | High (depends on overlap) | High (CWT) or critical (DWT) |
| Phase information | Yes (complex STFT) | Depends on wavelet |
| Invertibility | Yes (COLA condition) | Yes (admissibility) |

### 14.2 When to Use What

**Use STFT/Spectrogram when:**
- The signal has relatively uniform time-frequency characteristics
- You need constant frequency resolution across all frequencies
- Musical pitch analysis (equal frequency bins match musical perception)
- The signal is well-described by sinusoidal components

**Use Wavelet Transform when:**
- The signal has both transient and slowly-varying components
- Different frequency bands require different time resolutions
- You need to detect singularities or sharp transitions
- Denoising is the primary goal
- Multi-scale analysis is needed (fractals, turbulence)

### 14.3 Comparison for a Chirp Signal

For a linear chirp $x(t) = \cos(2\pi(f_0 t + \frac{\beta}{2}t^2))$:

- **STFT**: Shows a straight line in the time-frequency plane, but the line width is constant (determined by the window)
- **Wavelet**: Shows a curve (since scale $\neq$ frequency linearly), but with better time resolution at high frequencies and better frequency resolution at low frequencies

---

## 15. Applications

### 15.1 Music and Audio Analysis

- **Spectrogram**: Fundamental tool for music information retrieval, showing melody, harmonics, and rhythm
- **Constant-Q Transform** (CQT): Wavelet-like transform with logarithmic frequency spacing matching musical scales
- **Onset detection**: Wavelet analysis detects note onsets (transients) with good time precision
- **Pitch tracking**: STFT with harmonic analysis

### 15.2 Biomedical Signal Processing

- **ECG analysis**: Wavelet decomposition separates QRS complex (high freq) from baseline wander (low freq)
- **EEG spectral analysis**: STFT reveals alpha (8-12 Hz), beta (12-30 Hz), and gamma (>30 Hz) rhythms over time
- **EMG processing**: Time-frequency analysis of muscle activation patterns

### 15.3 Vibration Analysis and Fault Detection

- **Bearing fault detection**: Wavelet decomposition reveals periodic impulses hidden in noise
- **Rotating machinery**: Order tracking with time-frequency analysis
- **Structural health monitoring**: Wavelet-based damage detection in bridges and buildings

### 15.4 Geophysics and Seismology

- **Seismic event detection**: Multi-scale wavelet analysis separates local and teleseismic events
- **Dispersion analysis**: Group velocity measurement using wavelet transforms
- **Time-frequency filtering**: Remove specific time-frequency regions of interference

### 15.5 Image Processing

- **Wavelet compression**: JPEG2000 uses the DWT (Daubechies 9/7 or Le Gall 5/3 wavelets)
- **Edge detection**: Wavelet maxima correspond to edges at different scales
- **Texture analysis**: Wavelet energy in different sub-bands characterizes textures

---

## 16. Python Implementation

### 16.1 STFT and Spectrogram

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig


def compute_stft(x, fs, window_length, hop_size, nfft=None, window='hann'):
    """
    Compute the Short-Time Fourier Transform.

    Parameters
    ----------
    x : ndarray
        Input signal
    fs : float
        Sampling frequency
    window_length : int
        Window length in samples
    hop_size : int
        Hop size in samples
    nfft : int
        FFT size (zero-padded if > window_length)
    window : str
        Window type

    Returns
    -------
    t : ndarray
        Time axis
    f : ndarray
        Frequency axis
    Zxx : ndarray
        STFT matrix (frequency x time)
    """
    if nfft is None:
        nfft = window_length

    # Create window
    win = sig.get_window(window, window_length)

    # Number of frames
    n_frames = 1 + (len(x) - window_length) // hop_size

    # STFT matrix
    Zxx = np.zeros((nfft // 2 + 1, n_frames), dtype=complex)

    for m in range(n_frames):
        start = m * hop_size
        segment = x[start:start + window_length] * win
        spectrum = np.fft.rfft(segment, n=nfft)
        Zxx[:, m] = spectrum

    # Time and frequency axes
    t = np.arange(n_frames) * hop_size / fs
    f = np.arange(nfft // 2 + 1) * fs / nfft

    return t, f, Zxx


# Generate test signal: chirp + impulse
fs = 1000  # Sampling frequency
duration = 2.0
t = np.arange(0, duration, 1/fs)
N = len(t)

# Linear chirp from 50 Hz to 400 Hz
chirp = sig.chirp(t, f0=50, f1=400, t1=duration, method='linear')

# Add impulses at specific times
impulse_signal = np.zeros(N)
for t_imp in [0.3, 0.8, 1.5]:
    idx = int(t_imp * fs)
    impulse_signal[idx:idx+10] = 1.0

# Combined signal
x = chirp + 0.5 * impulse_signal + 0.1 * np.random.randn(N)

# Compute spectrograms with different window lengths
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time-domain signal
axes[0, 0].plot(t, x, 'b', alpha=0.7)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].set_title('Signal: Chirp + Impulses')
axes[0, 0].grid(True, alpha=0.3)

# Short window (good time resolution)
window_lengths = [32, 128, 512]
titles = ['Short Window (32 samples)', 'Medium Window (128 samples)',
          'Long Window (512 samples)']

for idx, (wl, title) in enumerate(zip(window_lengths, titles)):
    ax = axes[(idx + 1) // 2, (idx + 1) % 2]
    f_stft, t_stft, Sxx = sig.spectrogram(
        x, fs, window='hann', nperseg=wl, noverlap=wl//2, nfft=max(wl, 512)
    )
    ax.pcolormesh(t_stft, f_stft, 10*np.log10(Sxx + 1e-10),
                  shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Spectrogram: {title}')
    ax.set_ylim([0, 500])

plt.tight_layout()
plt.savefig('stft_resolution_tradeoff.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 16.2 Wavelet Analysis with PyWavelets

```python
import pywt
import numpy as np
import matplotlib.pyplot as plt


# Generate a non-stationary signal
fs = 1024
t = np.arange(0, 2, 1/fs)
N = len(t)

# Signal with time-varying frequency content
x = np.zeros(N)
x[:N//4] = np.sin(2 * np.pi * 10 * t[:N//4])           # 10 Hz
x[N//4:N//2] = np.sin(2 * np.pi * 50 * t[N//4:N//2])   # 50 Hz
x[N//2:3*N//4] = np.sin(2 * np.pi * 100 * t[N//2:3*N//4])  # 100 Hz
x[3*N//4:] = np.sin(2 * np.pi * 200 * t[3*N//4:])      # 200 Hz

# Add some noise
x += 0.3 * np.random.randn(N)

# --- Continuous Wavelet Transform (CWT) ---
scales = np.arange(1, 128)
wavelet = 'cmor1.5-1.0'  # Complex Morlet wavelet
coefficients, frequencies = pywt.cwt(x, scales, wavelet, sampling_period=1/fs)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Time domain
axes[0].plot(t, x, 'b', alpha=0.7)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Non-stationary Signal (10→50→100→200 Hz)')
axes[0].grid(True, alpha=0.3)

# CWT scalogram
im = axes[1].pcolormesh(t, frequencies, np.abs(coefficients),
                         shading='gouraud', cmap='jet')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('CWT Scalogram (Morlet Wavelet)')
axes[1].set_ylim([0, 300])
plt.colorbar(im, ax=axes[1], label='|CWT|')

# Compare with STFT spectrogram
from scipy import signal as sig
f_stft, t_stft, Sxx = sig.spectrogram(x, fs, nperseg=128, noverlap=96)
axes[2].pcolormesh(t_stft, f_stft, 10*np.log10(Sxx + 1e-10),
                   shading='gouraud', cmap='jet')
axes[2].set_ylabel('Frequency (Hz)')
axes[2].set_xlabel('Time (s)')
axes[2].set_title('STFT Spectrogram (128-sample Hann window)')
axes[2].set_ylim([0, 300])

plt.tight_layout()
plt.savefig('cwt_vs_stft.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 16.3 Discrete Wavelet Transform (DWT) and Denoising

```python
import pywt
import numpy as np
import matplotlib.pyplot as plt


# Create a clean signal
fs = 1000
t = np.arange(0, 1, 1/fs)
clean = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*50*t)

# Add noise
np.random.seed(42)
noise_level = 0.8
noisy = clean + noise_level * np.random.randn(len(t))

# DWT decomposition
wavelet = 'db4'
level = 5
coeffs = pywt.wavedec(noisy, wavelet, level=level)

# Estimate noise level from finest detail coefficients
sigma = np.median(np.abs(coeffs[-1])) / 0.6745
print(f"Estimated noise std: {sigma:.3f} (true: {noise_level:.3f})")

# Apply soft thresholding
threshold = sigma * np.sqrt(2 * np.log(len(noisy)))  # Universal threshold
print(f"Threshold: {threshold:.3f}")

coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
for i in range(1, len(coeffs)):
    coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))

# Reconstruct
denoised = pywt.waverec(coeffs_thresh, wavelet)
denoised = denoised[:len(t)]  # Trim to original length

# Plot results
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

axes[0].plot(t, clean, 'g', linewidth=2)
axes[0].set_title('Clean Signal')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, noisy, 'r', alpha=0.7)
axes[1].set_title(f'Noisy Signal (SNR = {10*np.log10(np.var(clean)/noise_level**2):.1f} dB)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, denoised, 'b', linewidth=1.5)
axes[2].set_title('Denoised Signal (Wavelet Soft Thresholding, db4)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(t, clean, 'g--', alpha=0.5, label='Clean')
axes[3].plot(t, denoised, 'b', alpha=0.7, label='Denoised')
axes[3].set_title('Comparison: Clean vs Denoised')
axes[3].set_xlabel('Time (s)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wavelet_denoising.png', dpi=150, bbox_inches='tight')
plt.show()

# Compute SNR improvement
snr_noisy = 10 * np.log10(np.sum(clean**2) / np.sum((noisy - clean)**2))
snr_denoised = 10 * np.log10(np.sum(clean**2) / np.sum((denoised - clean)**2))
print(f"\nSNR (noisy):    {snr_noisy:.1f} dB")
print(f"SNR (denoised): {snr_denoised:.1f} dB")
print(f"SNR improvement: {snr_denoised - snr_noisy:.1f} dB")
```

### 16.4 DWT Decomposition Visualization

```python
import pywt
import numpy as np
import matplotlib.pyplot as plt


# Generate signal with multi-scale features
fs = 1024
t = np.arange(0, 2, 1/fs)
N = len(t)

# Low frequency trend + medium frequency oscillation + high frequency bursts
x = (0.5 * np.sin(2*np.pi*2*t)                      # 2 Hz trend
     + np.sin(2*np.pi*30*t)                           # 30 Hz oscillation
     + 0.3 * np.sin(2*np.pi*150*t) * (t > 0.5) * (t < 1.0)  # 150 Hz burst
     + 0.2 * np.random.randn(N))                      # noise

# Multi-level DWT decomposition
wavelet = 'db4'
level = 6
coeffs = pywt.wavedec(x, wavelet, level=level)

# Reconstruct individual components
details = []
for i in range(1, level + 1):
    # Zero out all coefficients except the i-th detail
    c_temp = [np.zeros_like(c) for c in coeffs]
    c_temp[i] = coeffs[i].copy()
    detail_i = pywt.waverec(c_temp, wavelet)[:N]
    details.append(detail_i)

# Approximation
c_approx = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
approx = pywt.waverec(c_approx, wavelet)[:N]

# Plot decomposition
fig, axes = plt.subplots(level + 2, 1, figsize=(14, 16), sharex=True)

axes[0].plot(t, x, 'k', alpha=0.7)
axes[0].set_ylabel('Signal')
axes[0].set_title(f'DWT Decomposition ({wavelet}, {level} levels)')

axes[1].plot(t, approx, 'b', alpha=0.7)
axes[1].set_ylabel(f'A{level}')
freq_band = f'0-{fs/2**(level+1):.0f} Hz'
axes[1].annotate(freq_band, xy=(0.98, 0.8), xycoords='axes fraction',
                 ha='right', fontsize=9, color='gray')

for i, detail in enumerate(details):
    ax = axes[i + 2]
    ax.plot(t, detail, 'r' if i == 0 else 'orange' if i < 3 else 'g', alpha=0.7)
    level_idx = level - i
    ax.set_ylabel(f'D{level_idx}')
    low_f = fs / 2**(level_idx + 1)
    high_f = fs / 2**level_idx
    freq_band = f'{low_f:.0f}-{high_f:.0f} Hz'
    ax.annotate(freq_band, xy=(0.98, 0.8), xycoords='axes fraction',
                ha='right', fontsize=9, color='gray')

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('dwt_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 16.5 Manual CWT Implementation

```python
import numpy as np
import matplotlib.pyplot as plt


def morlet_wavelet(t, omega0=5.0):
    """Morlet wavelet (simplified, without correction term)."""
    return np.pi**(-0.25) * np.exp(1j * omega0 * t) * np.exp(-t**2 / 2)


def manual_cwt(x, scales, dt=1.0, omega0=5.0):
    """
    Manual CWT implementation using convolution.

    Parameters
    ----------
    x : ndarray
        Input signal
    scales : ndarray
        Array of scales
    dt : float
        Sampling period
    omega0 : float
        Center frequency of Morlet wavelet

    Returns
    -------
    W : ndarray
        CWT coefficient matrix (n_scales x N)
    freqs : ndarray
        Pseudo-frequencies corresponding to each scale
    """
    N = len(x)
    n_scales = len(scales)
    W = np.zeros((n_scales, N), dtype=complex)

    for i, scale in enumerate(scales):
        # Create wavelet at this scale
        # Wavelet support: we need enough points to capture the wavelet
        M = min(10 * int(scale / dt), N)
        t_wav = np.arange(-M, M + 1) * dt
        wavelet = (1.0 / np.sqrt(scale)) * morlet_wavelet(t_wav / scale, omega0)
        wavelet = np.conj(wavelet)  # Conjugate for cross-correlation

        # Convolve (cross-correlate)
        conv_result = np.convolve(x, wavelet, mode='same') * dt
        W[i, :] = conv_result

    # Pseudo-frequencies
    freqs = omega0 / (2 * np.pi * scales * dt)

    return W, freqs


# Test with a chirp signal
fs = 500
t = np.arange(0, 3, 1/fs)
# Chirp from 5 Hz to 100 Hz
x = np.cos(2 * np.pi * (5*t + 47.5*t**2/3))

scales = np.arange(1, 100)
W, freqs = manual_cwt(x, scales, dt=1/fs)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(t, x, 'b', alpha=0.7)
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Chirp Signal (5 Hz to 100 Hz)')
axes[0].grid(True, alpha=0.3)

im = axes[1].pcolormesh(t, freqs, np.abs(W), shading='gouraud', cmap='magma')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('Manual CWT Scalogram (Morlet)')
axes[1].set_ylim([0, 120])
plt.colorbar(im, ax=axes[1], label='|W(a,b)|')

plt.tight_layout()
plt.savefig('manual_cwt.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 17. Exercises

### Exercise 1: STFT Resolution Tradeoff

Generate a signal consisting of two simultaneous sinusoids at 100 Hz and 105 Hz (very close in frequency) that each last for 0.5 seconds, followed by 0.5 seconds of silence.

(a) Compute and display the spectrogram with window lengths of 32, 64, 128, 256, and 512 samples ($f_s = 1000$ Hz).

(b) For each window length, determine whether the two frequencies are resolved (i.e., show as separate peaks). What is the minimum window length needed?

(c) Add a sharp click at $t = 0.3$ s. For which window lengths can you identify both the click's timing and the two frequencies?

(d) Verify the uncertainty principle: compute $\Delta t$ and $\Delta f$ for each window and check that $\Delta t \cdot \Delta f \geq 1/(4\pi)$.

### Exercise 2: Window Comparison

Using a signal with a known chirp (50 to 200 Hz over 1 second, $f_s = 1000$ Hz):

(a) Compute spectrograms using Rectangular, Hann, Hamming, Blackman, and Gaussian windows (all length 128).

(b) Compare the spectral leakage visible in each spectrogram. Which window produces the cleanest chirp trajectory?

(c) For each window, measure the -3 dB mainlobe width and the highest sidelobe level.

### Exercise 3: Wavelet Families

(a) Plot the scaling function $\phi(t)$ and wavelet function $\psi(t)$ for Haar, db4, db8, sym4, and coif2 wavelets (use `pywt.Wavelet(...).wavefun(level=8)`).

(b) Compute and plot the frequency response of the associated decomposition filters $h[n]$ and $g[n]$.

(c) For each wavelet, decompose the test signal from Exercise 1 and compare the time-frequency resolution.

(d) Which wavelet best resolves the two close sinusoids? Which best localizes the click?

### Exercise 4: Wavelet Denoising

Generate a signal $x(t) = \text{sign}(\sin(2\pi \cdot 3 \cdot t))$ (square wave at 3 Hz) corrupted by Gaussian noise at SNR = 5 dB.

(a) Apply wavelet denoising with hard and soft thresholding using the universal threshold. Compare the results.

(b) Try different wavelets (Haar, db4, sym8) and compare the denoising quality (SNR).

(c) Vary the number of decomposition levels from 1 to 8. Plot the output SNR vs level number.

(d) Compare with a simple lowpass filter. Under what conditions does wavelet denoising outperform frequency-domain filtering?

### Exercise 5: CWT vs DWT

For a signal composed of:
- A 5 Hz sinusoid for the full duration
- A 50 Hz burst from $t = 0.3$ to $t = 0.5$ s
- A 200 Hz burst from $t = 0.7$ to $t = 0.72$ s (very short)

(a) Compute the CWT using the Morlet wavelet and display the scalogram.

(b) Compute a 6-level DWT using db4 and display the decomposition.

(c) Which representation better localizes the 200 Hz burst in time? Why?

(d) Which representation better separates the 5 Hz and 50 Hz components in frequency? Why?

### Exercise 6: Music Analysis

Load or synthesize a short musical passage (e.g., a piano scale: C4, D4, E4, F4, G4, A4, B4, C5, each note lasting 0.25 seconds).

(a) Compute and display the spectrogram. Can you identify each note and its harmonics?

(b) Compute the CWT and display the scalogram. Compare with the spectrogram.

(c) Implement a simple note detector by finding the fundamental frequency in each time frame.

(d) Compute the constant-Q transform (CQT) using logarithmically spaced frequency bins. How does it compare with the linear-frequency spectrogram for music analysis?

### Exercise 7: Wigner-Ville Distribution

(a) Implement the discrete Wigner-Ville distribution for a single-component linear chirp. Verify that the WVD shows a clean, narrow line.

(b) Now compute the WVD for the sum of two linear chirps (one increasing, one decreasing in frequency). Identify the cross-terms. Where do they appear in the time-frequency plane?

(c) Apply a smoothing kernel to the WVD (smoothed pseudo-Wigner-Ville) and show that the cross-terms are reduced but the resolution degrades.

---

## 18. Summary

| Concept | Key Formula / Idea |
|---|---|
| STFT | $X(\tau,f) = \int x(t)w(t-\tau)e^{-j2\pi ft}dt$ |
| Spectrogram | $S(\tau,f) = |X_{STFT}(\tau,f)|^2$ |
| Uncertainty principle | $\Delta t \cdot \Delta f \geq 1/(4\pi)$ |
| CWT | $W(a,b) = \frac{1}{\sqrt{a}}\int x(t)\psi^*(\frac{t-b}{a})dt$ |
| Scale-frequency relation | $f_a = f_c / a$ |
| Admissibility | $\int\psi(t)dt = 0$ (zero mean) |
| Refinement equation | $\phi(t) = \sqrt{2}\sum_k h[k]\phi(2t-k)$ |
| QMF relation | $g[n] = (-1)^n h[1-n]$ |
| DWT computation | Mallat's algorithm: $O(N)$ |
| Denoising threshold | $\lambda = \sigma\sqrt{2\ln N}$ (universal) |
| Noise estimation | $\hat{\sigma} = \text{median}(|d_1|)/0.6745$ |

**Key takeaways**:
1. The Fourier transform gives global frequency information; STFT and wavelets add temporal localization.
2. The uncertainty principle sets a fundamental limit on joint time-frequency resolution.
3. The STFT uses a fixed window, giving uniform resolution -- good for signals with consistent characteristics.
4. Wavelets use variable resolution: narrow windows at high frequencies, wide at low frequencies.
5. The DWT via Mallat's algorithm is computationally efficient ($O(N)$) and produces a non-redundant representation.
6. Wavelet denoising by thresholding is a powerful, principled approach to noise removal.
7. The choice between STFT and wavelets depends on the signal structure and the analysis goal.

---

## 19. References

1. S. Mallat, *A Wavelet Tour of Signal Processing*, 3rd ed., Academic Press, 2009.
2. I. Daubechies, *Ten Lectures on Wavelets*, SIAM, 1992.
3. L. Cohen, *Time-Frequency Analysis*, Prentice Hall, 1995.
4. C.K. Chui, *An Introduction to Wavelets*, Academic Press, 1992.
5. D.L. Donoho and I.M. Johnstone, "Ideal spatial adaptation by wavelet shrinkage," *Biometrika*, vol. 81, pp. 425-455, 1994.
6. A. Boggess and F.J. Narcowich, *A First Course in Wavelets with Fourier Analysis*, 2nd ed., Wiley, 2009.
7. M. Vetterli and J. Kovacevic, *Wavelets and Subband Coding*, Prentice Hall, 1995.

---

**Previous**: [13. Adaptive Filters](./13_Adaptive_Filters.md) | **Next**: [15. Image Signal Processing](./15_Image_Signal_Processing.md)
