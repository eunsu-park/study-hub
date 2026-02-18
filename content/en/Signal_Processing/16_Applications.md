# 16. Applications of Signal Processing

**Previous**: [15. Image Signal Processing](./15_Image_Signal_Processing.md) | [Overview](./00_Overview.md)

---

Signal processing is not merely a theoretical discipline -- it is an engineering backbone that powers systems we use every day. This final lesson surveys four major application domains: audio, communications, radar/sonar, and biomedical signal processing. For each domain, we develop the key signal processing concepts, provide mathematical foundations, and implement working Python demonstrations. The goal is to show how the tools from the previous fifteen lessons come together in real systems.

**Difficulty**: ⭐⭐⭐

**Prerequisites**: DFT/FFT, filtering (FIR/IIR), modulation basics, correlation, spectral analysis

**Learning Objectives**:
- Understand audio signal representation and implement digital audio effects
- Implement pitch detection using autocorrelation and cepstrum methods
- Describe analog and digital modulation schemes (AM, FM, ASK, FSK, PSK, QAM)
- Explain pulse shaping, matched filtering, and OFDM fundamentals
- Apply matched filtering and chirp compression to radar signal processing
- Compute and interpret the ambiguity function for radar waveform design
- Process biomedical signals (ECG, EEG) using the techniques from this course
- Implement working demonstrations of each application in Python

---

## Table of Contents

1. [Audio Signal Processing](#1-audio-signal-processing)
2. [Audio Representation and Formats](#2-audio-representation-and-formats)
3. [Digital Audio Effects](#3-digital-audio-effects)
4. [Pitch Detection](#4-pitch-detection)
5. [Audio Equalization](#5-audio-equalization)
6. [Speech Coding: Linear Predictive Coding (LPC)](#6-speech-coding-linear-predictive-coding-lpc)
7. [Communications: Analog Modulation](#7-communications-analog-modulation)
8. [Communications: Digital Modulation](#8-communications-digital-modulation)
9. [Pulse Shaping and Matched Filtering](#9-pulse-shaping-and-matched-filtering)
10. [Channel Models and Equalization](#10-channel-models-and-equalization)
11. [OFDM Fundamentals](#11-ofdm-fundamentals)
12. [Radar Signal Processing](#12-radar-signal-processing)
13. [Chirp Signals and Pulse Compression](#13-chirp-signals-and-pulse-compression)
14. [The Ambiguity Function](#14-the-ambiguity-function)
15. [Biomedical Signal Processing: ECG](#15-biomedical-signal-processing-ecg)
16. [Biomedical Signal Processing: EEG](#16-biomedical-signal-processing-eeg)
17. [Heart Rate Variability (HRV)](#17-heart-rate-variability-hrv)
18. [Python Implementations](#18-python-implementations)
19. [Exercises](#19-exercises)
20. [Summary](#20-summary)
21. [References](#21-references)

---

## 1. Audio Signal Processing

### 1.1 The Audio Signal Chain

```
Sound Source → Microphone → ADC → Digital Processing → DAC → Amplifier → Speaker
  (analog)     (transducer)       (DSP algorithms)        (transducer)    (analog)
```

Audio signal processing operates on digital representations of sound waves. The key parameters are:

- **Sampling rate** ($f_s$): 44.1 kHz (CD), 48 kHz (professional), 96/192 kHz (high-res)
- **Bit depth**: 16-bit (CD), 24-bit (professional), 32-bit float (internal processing)
- **Channels**: Mono (1), Stereo (2), Surround (5.1, 7.1), Spatial (Ambisonics)

### 1.2 Frequency Ranges

| Range | Frequency | Musical Context |
|---|---|---|
| Sub-bass | 20-60 Hz | Felt more than heard |
| Bass | 60-250 Hz | Kick drum, bass guitar |
| Low mid | 250-500 Hz | Warmth, body of instruments |
| Mid | 500 Hz - 2 kHz | Vocal clarity, presence |
| Upper mid | 2-4 kHz | Perceived loudness, attack |
| Presence | 4-6 kHz | Definition, clarity |
| Brilliance | 6-20 kHz | Air, sparkle, sibilance |

Human hearing spans approximately 20 Hz to 20 kHz, with maximum sensitivity around 2-5 kHz (the ear canal resonance).

---

## 2. Audio Representation and Formats

### 2.1 Pulse Code Modulation (PCM)

PCM is the standard uncompressed digital audio representation:

1. **Sample**: Analog signal sampled at rate $f_s$
2. **Quantize**: Each sample mapped to one of $2^B$ levels (for $B$-bit resolution)
3. **Encode**: Binary representation of each quantized sample

The **signal-to-quantization-noise ratio (SQNR)** for a full-scale sinusoid:

$$\text{SQNR} = 6.02B + 1.76 \text{ dB}$$

For 16-bit audio: $\text{SQNR} \approx 98$ dB. For 24-bit: $\text{SQNR} \approx 146$ dB.

### 2.2 Data Rate

Uncompressed data rate:

$$R = f_s \times B \times C$$

where $C$ is the number of channels.

For CD audio (44.1 kHz, 16-bit, stereo): $R = 44100 \times 16 \times 2 = 1.41$ Mbps.

### 2.3 Dithering

When reducing bit depth, **dithering** adds a small amount of noise before quantization to convert distortion (harmonic) into noise (broadband). This trades a slight increase in noise floor for the elimination of quantization distortion artifacts.

The dither noise is typically triangular probability density function (TPDF) with amplitude of $\pm 1$ LSB.

---

## 3. Digital Audio Effects

### 3.1 Delay-Based Effects

Most audio effects are built from delay lines:

$$y[n] = x[n] + g \cdot x[n - D]$$

where $D$ is the delay in samples and $g$ is the feedback/feedforward gain.

#### Echo/Delay

Simple delay with feedback:

$$y[n] = x[n] + g \cdot y[n - D]$$

For an audible echo: $D > 50$ ms ($D > 0.05 f_s$ samples).

#### Reverb

Reverb simulates the acoustic response of a room. The simplest model uses comb filters and allpass filters (Schroeder reverberator):

**Comb filter** (feedback):
$$y[n] = x[n] + g \cdot y[n - D]$$

**Allpass filter**:
$$y[n] = -g \cdot x[n] + x[n - D] + g \cdot y[n - D]$$

The Schroeder reverberator cascades 4 parallel comb filters followed by 2 series allpass filters.

More sophisticated reverb uses **convolution reverb**: convolve the dry signal with a measured room impulse response (RIR).

### 3.2 Modulation-Based Effects

#### Chorus

The chorus effect simulates multiple instruments playing the same note by mixing the original signal with a delayed copy whose delay is slowly modulated:

$$y[n] = x[n] + g \cdot x[n - D(n)]$$

where $D(n) = D_0 + A \sin(2\pi f_{LFO} n / f_s)$ with $D_0 \approx 20$-30 ms, $A \approx 1$-5 ms, $f_{LFO} \approx 0.5$-3 Hz.

#### Flanger

Similar to chorus but with shorter delays and feedback:

$$y[n] = x[n] + g \cdot y[n - D(n)]$$

with $D_0 \approx 1$-10 ms, creating a sweeping comb filter effect.

#### Vibrato

Pure pitch modulation (no dry signal mixed in):

$$y[n] = x[n - D(n)]$$

with sinusoidal delay modulation.

### 3.3 Dynamics Processing

#### Compressor

Reduces the dynamic range by attenuating signals above a threshold:

$$g_{dB}[n] = \begin{cases} 0 & x_{dB}[n] < T \\ (1 - 1/R)(T - x_{dB}[n]) & x_{dB}[n] \geq T \end{cases}$$

where $T$ is the threshold and $R$ is the compression ratio.

**Parameters**:
- **Threshold**: Level above which compression begins
- **Ratio**: Amount of compression (2:1, 4:1, $\infty$:1 = limiter)
- **Attack**: How quickly the compressor responds
- **Release**: How quickly the compressor returns to unity gain
- **Knee**: Hard or soft transition at the threshold

The attack and release are implemented with an envelope follower using exponential smoothing:

$$x_{env}[n] = \begin{cases} \alpha_a x_{env}[n-1] + (1-\alpha_a)|x[n]| & |x[n]| > x_{env}[n-1] \\ \alpha_r x_{env}[n-1] + (1-\alpha_r)|x[n]| & |x[n]| \leq x_{env}[n-1] \end{cases}$$

where $\alpha_a = e^{-1/(f_s \cdot t_{attack})}$ and $\alpha_r = e^{-1/(f_s \cdot t_{release})}$.

---

## 4. Pitch Detection

### 4.1 Autocorrelation Method

The fundamental frequency of a periodic signal can be detected from the autocorrelation function:

$$R_{xx}[\tau] = \sum_{n=0}^{N-1} x[n] \cdot x[n + \tau]$$

For a periodic signal with period $T_0$ (in samples), $R_{xx}[\tau]$ has peaks at $\tau = 0, T_0, 2T_0, \ldots$. The fundamental frequency is:

$$f_0 = \frac{f_s}{T_0}$$

**Algorithm**:
1. Window the signal (Hann window, 20-50 ms frame)
2. Compute the autocorrelation
3. Find the first significant peak after the origin
4. The peak location gives $T_0$

**Pitch range constraint**: For speech, $f_0 \in [80, 400]$ Hz, so search for peaks in $\tau \in [f_s/400, f_s/80]$.

### 4.2 Cepstrum Method

The **cepstrum** (anagram of "spectrum") is defined as:

$$c[n] = \text{IDFT}\{\log|\text{DFT}\{x[n]\}|\}$$

The independent variable $n$ in the cepstrum domain is called **quefrency** (anagram of "frequency") and has units of samples (or time).

For a voiced speech signal, the cepstrum separates the **vocal tract envelope** (low quefrency) from the **pitch excitation** (peak at quefrency = pitch period):

```
Speech spectrum = Vocal tract envelope × Pitch harmonics
    (smooth)              (fine structure)

log(Speech spectrum) = log(Envelope) + log(Harmonics)
    cepstrum domain:    low quefrency   + peak at T₀
```

**Algorithm**:
1. Compute DFT of windowed frame
2. Take log of magnitude spectrum
3. Compute IDFT (cepstrum)
4. Find peak in the expected pitch range
5. Peak quefrency = pitch period $T_0$

### 4.3 Comparison

| Method | Pros | Cons |
|---|---|---|
| Autocorrelation | Robust, simple | Octave errors, broad peaks |
| Cepstrum | Good separation of source/filter | Sensitivity to noise, resolution limited |
| YIN (improved AC) | State-of-the-art accuracy | More complex |
| pYIN | Probabilistic, very robust | Computational cost |

---

## 5. Audio Equalization

### 5.1 Graphic Equalizer

A graphic equalizer consists of parallel bandpass filters at fixed center frequencies (typically at octave or 1/3-octave intervals). Each band has an adjustable gain.

Standard octave-band center frequencies: 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000 Hz.

### 5.2 Parametric Equalizer

A parametric equalizer provides adjustable **center frequency**, **gain**, and **bandwidth** (Q factor) for each band:

**Peaking (bell) filter** (second-order IIR):

$$H(z) = \frac{1 + \alpha A \cdot z^{-1} \cdot b_1 + \alpha z^{-2}}{1 + \alpha/A \cdot z^{-1} \cdot a_1 + \alpha z^{-2}}$$

where $A = 10^{G_{dB}/40}$, $\omega_0 = 2\pi f_0/f_s$, $\alpha = \sin(\omega_0)/(2Q)$.

Cookbook biquad coefficients for a peaking EQ:

$$b_0 = 1 + \alpha A, \quad b_1 = -2\cos\omega_0, \quad b_2 = 1 - \alpha A$$
$$a_0 = 1 + \alpha/A, \quad a_1 = -2\cos\omega_0, \quad a_2 = 1 - \alpha/A$$

### 5.3 Shelving Filters

- **Low shelf**: Boosts/cuts frequencies below $f_0$
- **High shelf**: Boosts/cuts frequencies above $f_0$

Used for bass and treble controls on audio equipment.

---

## 6. Speech Coding: Linear Predictive Coding (LPC)

### 6.1 The Source-Filter Model

Speech production is modeled as:

$$\text{Excitation source} \to \text{Vocal tract filter} \to \text{Speech signal}$$

- **Voiced sounds** (vowels): Excitation = periodic pulse train at pitch frequency
- **Unvoiced sounds** (fricatives): Excitation = white noise
- **Vocal tract**: All-pole filter (resonant tube model)

### 6.2 Linear Prediction

The current sample is predicted from past samples:

$$\hat{x}[n] = \sum_{k=1}^{p} a_k x[n-k]$$

The prediction error (residual) is:

$$e[n] = x[n] - \hat{x}[n] = x[n] - \sum_{k=1}^{p} a_k x[n-k]$$

The LPC coefficients $\{a_k\}$ are chosen to minimize $E[e^2[n]]$.

### 6.3 Solving for LPC Coefficients

Minimizing the prediction error leads to the **Yule-Walker (normal) equations**:

$$\mathbf{R}\mathbf{a} = \mathbf{r}$$

where $\mathbf{R}$ is the Toeplitz autocorrelation matrix and $\mathbf{r}$ is the autocorrelation vector.

The **Levinson-Durbin algorithm** solves this in $O(p^2)$ operations, exploiting the Toeplitz structure.

### 6.4 LPC Vocoder

The LPC vocoder transmits only:
1. **LPC coefficients** ($\{a_k\}$, typically $p = 10$-16 for narrowband speech)
2. **Pitch period** (for voiced frames)
3. **Voiced/unvoiced flag**
4. **Gain** (energy of the frame)

This achieves very low bit rates (2.4 kbps for LPC-10) at the cost of speech quality.

### 6.5 LPC Spectrum

The LPC model represents the vocal tract as an all-pole filter:

$$H(z) = \frac{G}{1 - \sum_{k=1}^{p} a_k z^{-k}} = \frac{G}{A(z)}$$

The LPC spectrum $|H(e^{j\omega})|^2$ provides a smooth spectral envelope of the speech signal. The poles of $H(z)$ correspond to the **formant frequencies** (resonances of the vocal tract).

---

## 7. Communications: Analog Modulation

### 7.1 Why Modulation?

Modulation shifts a baseband signal to a higher carrier frequency for:
1. **Antenna efficiency**: Antenna size $\sim \lambda/4$; higher frequency = smaller antenna
2. **Frequency division multiplexing**: Multiple signals share the spectrum
3. **Noise performance**: Some modulation schemes provide noise improvement

### 7.2 Amplitude Modulation (AM)

The AM signal is:

$$x_{AM}(t) = [1 + m \cdot x(t)] \cos(2\pi f_c t)$$

where:
- $f_c$ is the carrier frequency
- $m$ is the modulation index ($0 < m \leq 1$ for no distortion)
- $x(t)$ is the normalized message signal ($|x(t)| \leq 1$)

**Spectrum**: The AM spectrum consists of the carrier plus upper and lower sidebands:
$$X_{AM}(f) = \frac{1}{2}\delta(f - f_c) + \frac{m}{4}[X(f - f_c) + X(f + f_c)]$$

**Bandwidth**: $B_{AM} = 2W$ where $W$ is the message bandwidth.

**Demodulation**: Envelope detection (simple diode + RC circuit).

### 7.3 Frequency Modulation (FM)

The FM signal is:

$$x_{FM}(t) = A_c \cos\!\left(2\pi f_c t + 2\pi k_f \int_0^t x(\tau) \, d\tau\right)$$

where $k_f$ is the frequency deviation constant (Hz/V).

The **instantaneous frequency** is:

$$f_i(t) = f_c + k_f x(t)$$

**Bandwidth** (Carson's rule):

$$B_{FM} \approx 2(\Delta f + W) = 2W(\beta + 1)$$

where $\Delta f = k_f \max|x(t)|$ is the peak frequency deviation and $\beta = \Delta f / W$ is the modulation index.

**Advantages of FM over AM**:
- Constant envelope (no amplitude variation) -- robust to nonlinear amplifiers
- Better noise performance (FM captures noise as amplitude variation, which is removed by limiting)
- Capture effect: stronger signal suppresses weaker interferers

### 7.4 Phase Modulation (PM)

$$x_{PM}(t) = A_c \cos\!\left(2\pi f_c t + k_p x(t)\right)$$

PM and FM are related: FM of $x(t)$ is equivalent to PM of $\int x(t)dt$.

---

## 8. Communications: Digital Modulation

### 8.1 Digital Modulation Overview

Digital modulation maps discrete symbols to analog waveforms for transmission:

| Scheme | What Varies | Constellation |
|---|---|---|
| ASK (Amplitude Shift Keying) | Amplitude | Points on real axis |
| FSK (Frequency Shift Keying) | Frequency | Orthogonal signals |
| PSK (Phase Shift Keying) | Phase | Points on unit circle |
| QAM (Quadrature Amplitude Modulation) | Amplitude + Phase | Grid in I-Q plane |

### 8.2 Binary Phase Shift Keying (BPSK)

The simplest PSK scheme maps bits to antipodal signals:

$$s(t) = \begin{cases} +A\cos(2\pi f_c t) & \text{bit } 1 \\ -A\cos(2\pi f_c t) & \text{bit } 0 \end{cases}$$

**Bit error rate** in AWGN:

$$P_b = Q\!\left(\sqrt{\frac{2E_b}{N_0}}\right) = \frac{1}{2}\text{erfc}\!\left(\sqrt{\frac{E_b}{N_0}}\right)$$

where $E_b$ is the energy per bit and $N_0$ is the noise power spectral density.

### 8.3 Quadrature Phase Shift Keying (QPSK)

QPSK maps pairs of bits (dibits) to four phase states:

$$s_k(t) = A\cos\!\left(2\pi f_c t + \frac{\pi}{4} + \frac{k\pi}{2}\right), \quad k = 0, 1, 2, 3$$

**Constellation**: 4 points at angles $\{\pi/4, 3\pi/4, 5\pi/4, 7\pi/4\}$.

QPSK has the **same BER as BPSK** but carries **twice the data rate** in the same bandwidth.

### 8.4 Quadrature Amplitude Modulation (QAM)

$M$-QAM uses both amplitude and phase to map $\log_2 M$ bits per symbol:

$$s(t) = A_I \cos(2\pi f_c t) - A_Q \sin(2\pi f_c t)$$

where $(A_I, A_Q)$ are chosen from a regular grid.

**16-QAM**: 4 bits per symbol, 16 constellation points in a $4 \times 4$ grid.

**64-QAM**: 6 bits per symbol (used in WiFi, cable TV).

**256-QAM**: 8 bits per symbol (high-throughput WiFi).

The tradeoff: higher $M$ gives higher spectral efficiency but requires higher SNR.

### 8.5 I-Q Representation

Any bandpass signal can be represented as:

$$s(t) = I(t)\cos(2\pi f_c t) - Q(t)\sin(2\pi f_c t)$$

where $I(t)$ is the **in-phase** component and $Q(t)$ is the **quadrature** component. The complex baseband equivalent is:

$$\tilde{s}(t) = I(t) + jQ(t)$$

All digital modulation schemes can be described as mapping symbols to points in the $I$-$Q$ plane.

---

## 9. Pulse Shaping and Matched Filtering

### 9.1 The Inter-Symbol Interference (ISI) Problem

When digital symbols are transmitted as pulses, the tails of one symbol's pulse can interfere with adjacent symbols:

$$r(t) = \sum_k a_k \, p(t - kT_s)$$

where $a_k$ are symbol values, $p(t)$ is the pulse shape, and $T_s$ is the symbol period.

### 9.2 Nyquist ISI Criterion

A pulse $p(t)$ causes zero ISI at the sampling instants if:

$$p(kT_s) = \begin{cases} 1 & k = 0 \\ 0 & k \neq 0 \end{cases}$$

The sinc pulse $p(t) = \text{sinc}(t/T_s)$ achieves this with minimum bandwidth $W = 1/(2T_s)$, but it decays slowly and is impractical.

### 9.3 Raised Cosine Pulse

The raised cosine pulse provides zero ISI with faster decay:

$$P(f) = \begin{cases} T_s & |f| \leq \frac{1-\alpha}{2T_s} \\[6pt] \frac{T_s}{2}\left[1 + \cos\!\left(\frac{\pi T_s}{\alpha}\left(|f| - \frac{1-\alpha}{2T_s}\right)\right)\right] & \frac{1-\alpha}{2T_s} < |f| \leq \frac{1+\alpha}{2T_s} \\[6pt] 0 & |f| > \frac{1+\alpha}{2T_s} \end{cases}$$

where $\alpha \in [0, 1]$ is the **rolloff factor**.

**Bandwidth**: $W = \frac{1+\alpha}{2T_s}$

In practice, the **root-raised cosine (RRC)** filter is split between transmitter and receiver: $P_{RRC}(f) = \sqrt{P_{RC}(f)}$. This way, $P_{TX}(f) \cdot P_{RX}(f) = P_{RC}(f)$, and the receiver filter is the matched filter for the transmitted pulse.

### 9.4 Matched Filter

The **matched filter** maximizes the output SNR for a known pulse $s(t)$ in additive white noise:

$$h_{matched}(t) = s^*(T - t)$$

The matched filter is the time-reversed, conjugated version of the signal, delayed by $T$.

**Output SNR** of the matched filter:

$$\text{SNR}_{max} = \frac{2E_s}{N_0}$$

where $E_s = \int |s(t)|^2 \, dt$ is the signal energy. This is the maximum achievable SNR regardless of the filter.

---

## 10. Channel Models and Equalization

### 10.1 Additive White Gaussian Noise (AWGN) Channel

$$r(t) = s(t) + n(t)$$

where $n(t)$ is white Gaussian noise with PSD $N_0/2$.

### 10.2 Multipath Channel

$$r(t) = \sum_{k=0}^{L-1} h_k \, s(t - \tau_k) + n(t)$$

The discrete-time equivalent:

$$r[n] = \sum_{k=0}^{L-1} h[k] \, s[n-k] + v[n] = (h * s)[n] + v[n]$$

Multipath causes:
- **ISI**: Delayed copies of the signal interfere with current symbols
- **Frequency-selective fading**: Some frequencies are attenuated more than others

### 10.3 Zero-Forcing (ZF) Equalizer

The ZF equalizer inverts the channel:

$$W_{ZF}(f) = \frac{1}{H(f)}$$

**Problem**: Noise enhancement at frequencies where $|H(f)|$ is small.

### 10.4 MMSE Equalizer

$$W_{MMSE}(f) = \frac{H^*(f)}{|H(f)|^2 + N_0/E_s}$$

The MMSE equalizer balances ISI elimination with noise suppression. When $N_0 = 0$, it reduces to the ZF equalizer.

### 10.5 Adaptive Equalization

In practice, the channel is unknown and time-varying. Adaptive equalizers (LMS, RLS from Lesson 13) are used with:
- **Training sequence**: Known symbols for initial convergence
- **Decision-directed mode**: Use detected symbols as reference after convergence

---

## 11. OFDM Fundamentals

### 11.1 Motivation

Orthogonal Frequency Division Multiplexing (OFDM) is the dominant modulation scheme for broadband wireless (WiFi, LTE, 5G) and wired (DSL, cable) communications.

The key idea: instead of transmitting one high-rate stream through a frequency-selective channel, transmit many low-rate streams on parallel narrowband subcarriers, each experiencing flat fading.

### 11.2 OFDM System Model

```
Transmitter:
  Data → S/P → QAM Map → IFFT → Add CP → P/S → DAC → Channel

Receiver:
  ADC → S/P → Remove CP → FFT → QAM Demap → P/S → Data
```

The transmitted OFDM symbol (discrete time):

$$x[n] = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} X[k] \, e^{j2\pi kn/N}, \quad n = 0, 1, \ldots, N-1$$

This is simply the **IFFT** of the frequency-domain data symbols $X[k]$.

### 11.3 Cyclic Prefix (CP)

The cyclic prefix copies the last $L_{CP}$ samples to the beginning of the OFDM symbol:

$$\tilde{x}[n] = x[n \mod N], \quad n = -L_{CP}, \ldots, N-1$$

Purpose: converts linear convolution (channel) into **circular convolution**, which is diagonalized by the DFT. This eliminates inter-symbol interference between OFDM symbols.

**Condition**: $L_{CP} \geq L_{channel} - 1$ (CP length must be at least as long as the channel impulse response minus 1).

### 11.4 One-Tap Equalization

After removing the CP and taking the FFT at the receiver:

$$Y[k] = H[k] \cdot X[k] + V[k]$$

Each subcarrier sees a **flat** (scalar) channel $H[k]$. Equalization is trivial:

$$\hat{X}[k] = \frac{Y[k]}{H[k]}$$

This is the enormous advantage of OFDM: it reduces a complicated frequency-selective equalization problem to $N$ independent single-tap equalizations.

### 11.5 OFDM Parameters (WiFi Example)

| Parameter | 802.11a/g (20 MHz) |
|---|---|
| Subcarriers ($N$) | 64 |
| Data subcarriers | 48 |
| Pilot subcarriers | 4 |
| Subcarrier spacing | 312.5 kHz |
| Symbol duration | 3.2 $\mu$s |
| CP duration | 0.8 $\mu$s |
| Total symbol | 4.0 $\mu$s |

---

## 12. Radar Signal Processing

### 12.1 Radar Basics

**RADAR**: Radio Detection and Ranging. A radar transmits a pulse, and the echo from a target reveals:

- **Range**: $R = \frac{c \cdot \tau}{2}$ where $\tau$ is the round-trip delay and $c$ is the speed of light
- **Velocity**: Via the Doppler shift $f_d = \frac{2v_r}{\lambda}$ where $v_r$ is the radial velocity and $\lambda$ is the wavelength

### 12.2 Range Resolution

The range resolution is determined by the bandwidth of the transmitted pulse:

$$\Delta R = \frac{c}{2B}$$

where $B$ is the signal bandwidth.

For a simple rectangular pulse of duration $\tau_p$:
- Bandwidth: $B \approx 1/\tau_p$
- Range resolution: $\Delta R = c\tau_p/2$

**The dilemma**: Good range resolution requires a short pulse (wide bandwidth), but a short pulse has low energy, limiting detection range. Pulse compression resolves this.

### 12.3 Doppler Processing

For a moving target, the received signal has a frequency shift:

$$f_d = \frac{2v_r f_c}{c} = \frac{2v_r}{\lambda}$$

The velocity resolution from a coherent processing interval (CPI) of duration $T_{CPI}$:

$$\Delta v = \frac{\lambda}{2T_{CPI}}$$

### 12.4 Matched Filter for Radar

The radar receiver uses a matched filter to maximize the output SNR:

$$h_{MF}[n] = s^*[N-1-n]$$

The output of the matched filter is the **cross-correlation** of the received signal with the transmitted waveform:

$$y[n] = \sum_k r[k] \, s^*[k - n]$$

For a simple pulse, the matched filter output is a triangle with peak at the target delay.

---

## 13. Chirp Signals and Pulse Compression

### 13.1 The Linear FM Chirp

A **chirp** (linear frequency modulated pulse) sweeps frequency linearly across the pulse duration:

$$s(t) = \text{rect}\!\left(\frac{t}{\tau_p}\right) \exp\!\left(j\pi \frac{B}{\tau_p} t^2\right) \exp(j2\pi f_c t)$$

The instantaneous frequency varies from $f_c - B/2$ to $f_c + B/2$ over the pulse duration $\tau_p$.

**Key property**: The chirp has bandwidth $B$ and duration $\tau_p$, so its **time-bandwidth product** is:

$$\text{TBP} = B \tau_p \gg 1$$

### 13.2 Pulse Compression

The matched filter for a chirp compresses the long pulse into a short peak:

- **Input pulse**: Duration $\tau_p$, bandwidth $B$
- **Compressed pulse**: Duration $\approx 1/B$, peak amplitude $\approx \sqrt{B\tau_p}$

The **compression ratio** is:

$$\text{CR} = B\tau_p$$

The **processing gain** is:

$$G_p = 10\log_{10}(B\tau_p) \text{ dB}$$

Example: $\tau_p = 10$ $\mu$s, $B = 10$ MHz $\Rightarrow$ TBP = 100 $\Rightarrow$ processing gain = 20 dB.

### 13.3 Range Resolution After Compression

After pulse compression, the range resolution is:

$$\Delta R = \frac{c}{2B}$$

This is determined by the bandwidth $B$, not the pulse duration $\tau_p$. A long chirp pulse achieves the same range resolution as a short CW pulse, but with much more energy.

### 13.4 Sidelobes and Windowing

The compressed pulse has sidelobes (analogous to the sidelobes of a sinc function). Window functions (Hamming, Taylor, etc.) applied to the matched filter reduce sidelobes at the cost of slightly broader mainlobe (degraded resolution).

---

## 14. The Ambiguity Function

### 14.1 Definition

The **ambiguity function** describes a radar waveform's ability to simultaneously resolve targets in range and velocity:

$$\chi(\tau, f_d) = \int_{-\infty}^{\infty} s(t) \, s^*(t - \tau) \, e^{j2\pi f_d t} \, dt$$

where $\tau$ is the time delay (range) and $f_d$ is the Doppler frequency (velocity).

### 14.2 Properties

1. **Maximum at origin**: $|\chi(0, 0)| = E_s$ (signal energy)
2. **Volume invariance**: $\iint |\chi(\tau, f_d)|^2 \, d\tau \, df_d = E_s^2$
3. **Symmetry**: $|\chi(\tau, f_d)| = |\chi(-\tau, -f_d)|$

The volume invariance property means that the ambiguity function is a fixed-volume surface: sharpening in one dimension necessarily broadens another. This is the radar analogue of the uncertainty principle.

### 14.3 Ambiguity Functions of Common Waveforms

**CW pulse** (rectangular):
- Thumbtack-like along $\tau$ axis (width $\sim \tau_p$)
- Sinc-like along $f_d$ axis (width $\sim 1/\tau_p$)
- Good Doppler resolution, poor range resolution (for long $\tau_p$)

**Linear FM chirp**:
- Narrow along a diagonal ridge (range-Doppler coupling)
- Good range resolution (determined by $B$)
- The ridge means a Doppler shift looks like a range shift -- must be compensated

**Phase-coded waveforms** (e.g., Barker codes):
- Thumbtack-like ambiguity function
- Low sidelobes in both range and Doppler
- Limited to specific code lengths

### 14.4 Waveform Design

The ideal ambiguity function would be a single spike at the origin (perfect resolution in both range and velocity), but the volume invariance property forbids this. Waveform design involves choosing the shape that best suits the application:

- **Surveillance radar**: Need good Doppler resolution $\to$ long CW pulse
- **Tracking radar**: Need good range resolution $\to$ chirp
- **Pulse-Doppler radar**: Need both $\to$ coherent train of chirp pulses

---

## 15. Biomedical Signal Processing: ECG

### 15.1 The ECG Signal

The electrocardiogram (ECG or EKG) records the electrical activity of the heart. A single cardiac cycle produces the characteristic PQRST waveform:

```
    R
    ▲
    │╲
    │ ╲
    │  ╲
    │   ╲      T
P   │    ╲    ╱╲
╱╲  │     ╲  ╱  ╲
    │      ╲╱    ╲
────┼──────S──────────── baseline
    Q

P wave: Atrial depolarization        (0.08-0.10 s)
QRS:    Ventricular depolarization    (0.06-0.10 s)
T wave: Ventricular repolarization    (0.16 s)
PR interval: AV conduction time       (0.12-0.20 s)
QT interval: Ventricular activity     (0.30-0.44 s)
```

### 15.2 ECG Signal Characteristics

| Parameter | Value |
|---|---|
| Amplitude | 0.1 - 5 mV |
| Bandwidth | 0.05 - 150 Hz |
| Typical sampling rate | 250 - 1000 Hz |
| Heart rate | 60-100 bpm (1-1.67 Hz) |

### 15.3 ECG Noise Sources

1. **Baseline wander**: Low-frequency drift from respiration, body movement (< 0.5 Hz)
2. **Powerline interference**: 50/60 Hz and harmonics
3. **Muscle noise (EMG)**: Broadband, 20-500 Hz
4. **Motion artifacts**: Electrode movement, broadband

### 15.4 ECG Preprocessing Pipeline

```
Raw ECG → Baseline removal → Notch filter → Bandpass filter → QRS detection
           (highpass 0.5 Hz)  (50/60 Hz)    (0.5-40 Hz)      (Pan-Tompkins)
```

**Baseline wander removal**: Highpass filter with cutoff 0.5 Hz (or median filter with 200 ms and 600 ms windows).

**Powerline interference**: Notch filter at 50 or 60 Hz (narrow bandstop IIR filter).

### 15.5 QRS Detection: Pan-Tompkins Algorithm

The Pan-Tompkins algorithm (1985) is the standard QRS detector:

1. **Bandpass filter** (5-15 Hz): Maximizes QRS energy while suppressing P/T waves and noise
2. **Differentiation**: Emphasizes the steep QRS slopes: $y[n] = \frac{1}{8}(-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2])$
3. **Squaring**: $z[n] = y[n]^2$ (makes all values positive, emphasizes large slopes)
4. **Moving window integration**: $w[n] = \frac{1}{N}\sum_{k=0}^{N-1} z[n-k]$ with $N \approx 150$ ms
5. **Adaptive thresholding**: Two thresholds that adapt to the signal and noise levels

---

## 16. Biomedical Signal Processing: EEG

### 16.1 The EEG Signal

The electroencephalogram (EEG) records brain electrical activity from scalp electrodes. EEG signals are much weaker than ECG.

| Parameter | Value |
|---|---|
| Amplitude | 1 - 200 $\mu$V |
| Bandwidth | 0.5 - 100 Hz |
| Sampling rate | 256 - 1024 Hz |
| Channels | 1-256 (10-20 system) |

### 16.2 EEG Frequency Bands

| Band | Frequency | State |
|---|---|---|
| Delta ($\delta$) | 0.5-4 Hz | Deep sleep |
| Theta ($\theta$) | 4-8 Hz | Drowsiness, light sleep, meditation |
| Alpha ($\alpha$) | 8-13 Hz | Relaxed, eyes closed |
| Beta ($\beta$) | 13-30 Hz | Active thinking, focus |
| Gamma ($\gamma$) | 30-100 Hz | Higher cognitive functions, perception |

### 16.3 EEG Spectral Analysis

The power spectral density (PSD) of EEG reveals the dominant brain states:

$$S_{xx}(f) = \frac{1}{N}|X(f)|^2$$

**Band power** is computed by integrating the PSD over each frequency band:

$$P_{band} = \int_{f_1}^{f_2} S_{xx}(f) \, df$$

**Relative band power**: $P_{rel} = P_{band} / P_{total}$ indicates the proportion of total power in each band.

### 16.4 Event-Related Potentials (ERPs)

ERPs are obtained by averaging many EEG trials time-locked to a stimulus:

$$\text{ERP}[n] = \frac{1}{K}\sum_{k=1}^{K} x_k[n]$$

Averaging reduces the background noise by $\sqrt{K}$ while preserving the time-locked response.

### 16.5 Spectral Analysis Methods for EEG

- **Welch's method**: Averaged periodograms for robust PSD estimation
- **Multitaper method**: Multiple orthogonal tapers for reduced variance
- **Short-Time Fourier Transform**: Time-varying spectral content (event-related spectral perturbation)
- **Wavelet analysis**: Multi-scale time-frequency analysis of transient brain events

---

## 17. Heart Rate Variability (HRV)

### 17.1 What is HRV?

Heart Rate Variability is the variation in the time intervals between consecutive heartbeats (RR intervals). HRV is a marker of autonomic nervous system function.

### 17.2 Time-Domain Measures

From the sequence of RR intervals $\{RR_i\}$:

| Measure | Formula | Meaning |
|---|---|---|
| SDNN | $\sqrt{\frac{1}{N}\sum(RR_i - \overline{RR})^2}$ | Overall variability |
| RMSSD | $\sqrt{\frac{1}{N-1}\sum(RR_{i+1} - RR_i)^2}$ | Short-term variability |
| pNN50 | $\frac{\#\{|RR_{i+1} - RR_i| > 50\text{ms}\}}{N-1} \times 100\%$ | Parasympathetic activity |

### 17.3 Frequency-Domain Measures

The PSD of the RR interval time series reveals autonomic regulation:

| Band | Frequency | Origin |
|---|---|---|
| VLF | 0.003-0.04 Hz | Thermoregulation, RAAS |
| LF | 0.04-0.15 Hz | Sympathetic + Parasympathetic |
| HF | 0.15-0.40 Hz | Parasympathetic (respiratory sinus arrhythmia) |

The **LF/HF ratio** is a (controversial) marker of sympathovagal balance.

### 17.4 HRV Analysis Pipeline

1. **QRS detection**: Pan-Tompkins algorithm
2. **RR interval extraction**: Time between consecutive R-peaks
3. **Artifact removal**: Remove ectopic beats and missed detections
4. **Interpolation**: Resample RR intervals to uniform time grid (e.g., 4 Hz, cubic spline)
5. **PSD estimation**: Welch's method or AR model
6. **Band power calculation**: Integrate PSD in VLF, LF, HF bands

---

## 18. Python Implementations

### 18.1 Digital Audio Effects

```python
import numpy as np
import matplotlib.pyplot as plt


def generate_audio_signal(duration=2.0, fs=44100):
    """Generate a test audio signal (guitar-like pluck)."""
    t = np.arange(0, duration, 1/fs)
    # Karplus-Strong synthesis approximation
    f0 = 220  # A3
    signal = np.zeros(len(t))
    # Harmonics with decay
    for k in range(1, 8):
        decay = np.exp(-k * 1.5 * t)
        signal += (1/k) * np.sin(2*np.pi*k*f0*t) * decay
    signal = signal / np.max(np.abs(signal))
    return t, signal


def delay_effect(x, fs, delay_ms=300, feedback=0.5, mix=0.5):
    """Apply delay/echo effect."""
    delay_samples = int(delay_ms * fs / 1000)
    y = np.zeros(len(x) + delay_samples * 5)
    y[:len(x)] = x.copy()

    for i in range(delay_samples, len(y)):
        y[i] += feedback * y[i - delay_samples]

    return mix * y[:len(x)] + (1 - mix) * x


def chorus_effect(x, fs, depth_ms=3, rate_hz=1.5, mix=0.5):
    """Apply chorus effect with LFO-modulated delay."""
    N = len(x)
    t = np.arange(N) / fs
    base_delay = int(25 * fs / 1000)  # 25 ms base delay
    depth = int(depth_ms * fs / 1000)

    y = np.zeros(N)
    for n in range(base_delay + depth, N):
        # LFO modulates the delay
        mod = depth * np.sin(2 * np.pi * rate_hz * t[n])
        delay = base_delay + int(mod)

        # Linear interpolation for fractional delay
        d_int = int(delay)
        d_frac = delay - d_int

        if n - d_int - 1 >= 0:
            delayed = (1 - d_frac) * x[n - d_int] + d_frac * x[n - d_int - 1]
        else:
            delayed = x[n - d_int]

        y[n] = (1 - mix) * x[n] + mix * delayed

    return y


def compressor(x, fs, threshold_db=-20, ratio=4, attack_ms=5, release_ms=50):
    """Apply dynamic range compression."""
    N = len(x)
    threshold = 10**(threshold_db / 20)
    alpha_a = np.exp(-1 / (fs * attack_ms / 1000))
    alpha_r = np.exp(-1 / (fs * release_ms / 1000))

    env = np.zeros(N)
    gain = np.ones(N)
    y = np.zeros(N)

    for n in range(1, N):
        # Envelope follower
        abs_x = np.abs(x[n])
        if abs_x > env[n-1]:
            env[n] = alpha_a * env[n-1] + (1 - alpha_a) * abs_x
        else:
            env[n] = alpha_r * env[n-1] + (1 - alpha_r) * abs_x

        # Compute gain
        if env[n] > threshold:
            env_db = 20 * np.log10(env[n] + 1e-10)
            thresh_db = 20 * np.log10(threshold)
            gain_db = thresh_db + (env_db - thresh_db) / ratio - env_db
            gain[n] = 10**(gain_db / 20)
        else:
            gain[n] = 1.0

        y[n] = gain[n] * x[n]

    return y, gain


# Demo
fs = 44100
t, x = generate_audio_signal(2.0, fs)

# Apply effects
x_delay = delay_effect(x, fs, delay_ms=200, feedback=0.4)
x_chorus = chorus_effect(x, fs, depth_ms=3, rate_hz=1.5)
x_comp, gain = compressor(x, fs, threshold_db=-12, ratio=4)

# Plot
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

axes[0].plot(t[:4000], x[:4000], 'b', alpha=0.7)
axes[0].set_title('Original Signal')
axes[0].set_ylabel('Amplitude')

axes[1].plot(t[:4000], x_delay[:4000], 'r', alpha=0.7)
axes[1].set_title('Delay Effect (200ms, feedback=0.4)')
axes[1].set_ylabel('Amplitude')

axes[2].plot(t[:4000], x_chorus[:4000], 'g', alpha=0.7)
axes[2].set_title('Chorus Effect')
axes[2].set_ylabel('Amplitude')

axes[3].plot(t[:4000], x[:4000], 'b', alpha=0.3, label='Original')
axes[3].plot(t[:4000], x_comp[:4000], 'r', alpha=0.7, label='Compressed')
axes[3].set_title('Compressor (threshold=-12dB, ratio=4:1)')
axes[3].set_ylabel('Amplitude')
axes[3].set_xlabel('Time (s)')
axes[3].legend()

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('audio_effects.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 18.2 Digital Modulation

```python
import numpy as np
import matplotlib.pyplot as plt


def qpsk_modulate(bits, samples_per_symbol=20):
    """QPSK modulation."""
    # Ensure even number of bits
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)

    n_symbols = len(bits) // 2
    symbols = np.zeros(n_symbols, dtype=complex)

    # Gray coding: 00->pi/4, 01->3pi/4, 11->5pi/4, 10->7pi/4
    mapping = {(0, 0): np.exp(1j * np.pi/4),
               (0, 1): np.exp(1j * 3*np.pi/4),
               (1, 1): np.exp(1j * 5*np.pi/4),
               (1, 0): np.exp(1j * 7*np.pi/4)}

    for i in range(n_symbols):
        dibit = (bits[2*i], bits[2*i+1])
        symbols[i] = mapping[dibit]

    # Upsample
    signal = np.zeros(n_symbols * samples_per_symbol, dtype=complex)
    signal[::samples_per_symbol] = symbols

    # Pulse shaping (raised cosine)
    t = np.arange(-4*samples_per_symbol, 4*samples_per_symbol + 1)
    alpha = 0.35
    Ts = samples_per_symbol
    rc_pulse = np.sinc(t/Ts) * np.cos(np.pi*alpha*t/Ts) / (1 - (2*alpha*t/Ts)**2 + 1e-10)
    rc_pulse /= np.sqrt(np.sum(rc_pulse**2))

    signal = np.convolve(signal, rc_pulse, mode='same')
    return symbols, signal


def qam16_constellation():
    """Generate 16-QAM constellation."""
    levels = [-3, -1, 1, 3]
    constellation = []
    for i in levels:
        for q in levels:
            constellation.append(complex(i, q))
    return np.array(constellation) / np.sqrt(10)  # Normalize


# Generate random bits
np.random.seed(42)
bits = np.random.randint(0, 2, 200)

# QPSK modulation
symbols, signal = qpsk_modulate(bits)

# Add AWGN noise at different SNR levels
snr_values = [5, 15, 25]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, snr_db in enumerate(snr_values):
    noise_power = 10**(-snr_db/10)
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(symbols))
                                       + 1j * np.random.randn(len(symbols)))
    received = symbols + noise

    # Constellation diagram
    ax = axes[0, idx]
    ax.scatter(received.real, received.imag, s=10, alpha=0.5, label='Received')
    ax.scatter(symbols.real, symbols.imag, s=100, c='red', marker='x',
               linewidths=2, label='Ideal')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'QPSK: SNR = {snr_db} dB')
    ax.set_xlabel('In-phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.legend(fontsize=8)

# 16-QAM constellation
qam_const = qam16_constellation()
for idx, snr_db in enumerate(snr_values):
    noise_power = 10**(-snr_db/10)
    n_sym = 500
    # Random symbols from constellation
    sym_idx = np.random.randint(0, 16, n_sym)
    tx_symbols = qam_const[sym_idx]
    noise = np.sqrt(noise_power/2) * (np.random.randn(n_sym)
                                       + 1j * np.random.randn(n_sym))
    rx_symbols = tx_symbols + noise

    ax = axes[1, idx]
    ax.scatter(rx_symbols.real, rx_symbols.imag, s=5, alpha=0.3, label='Received')
    ax.scatter(qam_const.real, qam_const.imag, s=100, c='red', marker='x',
               linewidths=2, label='Ideal')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'16-QAM: SNR = {snr_db} dB')
    ax.set_xlabel('In-phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('digital_modulation.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 18.3 Radar: Chirp Pulse Compression

```python
import numpy as np
import matplotlib.pyplot as plt


def generate_chirp(tau_p, B, fs):
    """Generate a linear FM chirp pulse."""
    t = np.arange(0, tau_p, 1/fs)
    chirp_rate = B / tau_p
    s = np.exp(1j * np.pi * chirp_rate * t**2)
    return t, s


def matched_filter(received, template):
    """Apply matched filter (cross-correlation)."""
    mf = np.conj(template[::-1])
    output = np.convolve(received, mf, mode='full')
    return output


# Radar parameters
c = 3e8           # Speed of light (m/s)
fc = 10e9         # Carrier frequency (10 GHz, X-band)
tau_p = 10e-6     # Pulse width (10 μs)
B = 5e6           # Bandwidth (5 MHz)
fs = 20e6         # Sampling frequency

# Range resolution
delta_R = c / (2 * B)
print(f"Range resolution: {delta_R:.1f} m")
print(f"Time-bandwidth product: {B * tau_p:.0f}")
print(f"Processing gain: {10*np.log10(B*tau_p):.1f} dB")

# Generate chirp
t_chirp, chirp = generate_chirp(tau_p, B, fs)

# Simulate two targets at different ranges
R1 = 5000    # Target 1 at 5 km
R2 = 5060    # Target 2 at 5.06 km (60 m apart)
A1 = 1.0     # Target 1 amplitude
A2 = 0.5     # Target 2 amplitude (weaker)

# Convert ranges to time delays
tau1 = 2 * R1 / c
tau2 = 2 * R2 / c

# Create received signal
N_total = int(2 * 15000 / c * fs)  # Enough samples for 15 km range
received = np.zeros(N_total, dtype=complex)

# Add noise
noise_level = 0.3
received += noise_level * (np.random.randn(N_total) + 1j * np.random.randn(N_total))

# Add target returns
idx1 = int(tau1 * fs)
idx2 = int(tau2 * fs)
if idx1 + len(chirp) < N_total:
    received[idx1:idx1+len(chirp)] += A1 * chirp
if idx2 + len(chirp) < N_total:
    received[idx2:idx2+len(chirp)] += A2 * chirp

# Matched filter output
mf_output = matched_filter(received, chirp)

# Convert to range
range_axis = np.arange(len(mf_output)) * c / (2 * fs) / 1000  # km

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Chirp waveform
axes[0].plot(t_chirp * 1e6, chirp.real, 'b', alpha=0.7)
axes[0].set_xlabel('Time (μs)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title(f'Linear FM Chirp Pulse (B={B/1e6:.0f} MHz, τ={tau_p*1e6:.0f} μs)')
axes[0].grid(True, alpha=0.3)

# Chirp spectrogram
from scipy import signal
f_stft, t_stft, Sxx = signal.spectrogram(
    chirp.real, fs, nperseg=64, noverlap=56, nfft=256
)
axes[1].pcolormesh(t_stft*1e6, f_stft/1e6, 10*np.log10(Sxx + 1e-10),
                   shading='gouraud', cmap='viridis')
axes[1].set_xlabel('Time (μs)')
axes[1].set_ylabel('Frequency (MHz)')
axes[1].set_title('Chirp Spectrogram')

# Matched filter output
mf_db = 20 * np.log10(np.abs(mf_output) / np.max(np.abs(mf_output)) + 1e-10)
mask = (range_axis > 4.5) & (range_axis < 5.5)
axes[2].plot(range_axis[mask], mf_db[mask], 'b', linewidth=1.5)
axes[2].axvline(x=R1/1000, color='r', linestyle='--', alpha=0.5,
                label=f'Target 1: {R1/1000:.1f} km')
axes[2].axvline(x=R2/1000, color='g', linestyle='--', alpha=0.5,
                label=f'Target 2: {R2/1000:.3f} km')
axes[2].set_xlabel('Range (km)')
axes[2].set_ylabel('Amplitude (dB)')
axes[2].set_title(f'Matched Filter Output (ΔR = {delta_R:.0f} m)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([-40, 5])

plt.tight_layout()
plt.savefig('radar_pulse_compression.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 18.4 Biomedical: ECG Processing

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def generate_synthetic_ecg(duration=10, fs=360, heart_rate=72):
    """
    Generate a synthetic ECG signal.

    Parameters
    ----------
    duration : float
        Duration in seconds
    fs : int
        Sampling rate
    heart_rate : int
        Heart rate in bpm

    Returns
    -------
    t : ndarray
        Time axis
    ecg : ndarray
        Synthetic ECG signal
    r_peaks : ndarray
        Indices of R-peaks
    """
    t = np.arange(0, duration, 1/fs)
    N = len(t)
    ecg = np.zeros(N)
    rr_interval = 60 / heart_rate  # seconds
    rr_samples = int(rr_interval * fs)

    r_peaks = []

    # Generate each heartbeat
    beat_start = int(0.1 * fs)
    while beat_start + rr_samples < N:
        # P wave
        p_center = beat_start + int(0.16 * rr_samples)
        p_width = int(0.04 * fs)
        t_local = np.arange(-3*p_width, 3*p_width+1)
        p_wave = 0.15 * np.exp(-t_local**2 / (2*p_width**2))
        start = max(0, p_center - 3*p_width)
        end = min(N, p_center + 3*p_width + 1)
        ecg[start:end] += p_wave[:end-start]

        # QRS complex
        r_center = beat_start + int(0.25 * rr_samples)
        r_peaks.append(r_center)

        # Q wave
        q_center = r_center - int(0.02 * fs)
        q_width = int(0.01 * fs)
        t_local = np.arange(-3*q_width, 3*q_width+1)
        q_wave = -0.1 * np.exp(-t_local**2 / (2*q_width**2))
        start = max(0, q_center - 3*q_width)
        end = min(N, q_center + 3*q_width + 1)
        ecg[start:end] += q_wave[:end-start]

        # R wave
        r_width = int(0.008 * fs)
        t_local = np.arange(-3*r_width, 3*r_width+1)
        r_wave = 1.0 * np.exp(-t_local**2 / (2*r_width**2))
        start = max(0, r_center - 3*r_width)
        end = min(N, r_center + 3*r_width + 1)
        ecg[start:end] += r_wave[:end-start]

        # S wave
        s_center = r_center + int(0.02 * fs)
        s_width = int(0.012 * fs)
        t_local = np.arange(-3*s_width, 3*s_width+1)
        s_wave = -0.2 * np.exp(-t_local**2 / (2*s_width**2))
        start = max(0, s_center - 3*s_width)
        end = min(N, s_center + 3*s_width + 1)
        ecg[start:end] += s_wave[:end-start]

        # T wave
        t_center = beat_start + int(0.5 * rr_samples)
        t_width = int(0.06 * fs)
        t_local = np.arange(-3*t_width, 3*t_width+1)
        t_wave = 0.3 * np.exp(-t_local**2 / (2*t_width**2))
        start = max(0, t_center - 3*t_width)
        end = min(N, t_center + 3*t_width + 1)
        ecg[start:end] += t_wave[:end-start]

        # Add some RR variability
        rr_var = rr_samples + int(np.random.randn() * 0.02 * rr_samples)
        beat_start += rr_var

    return t, ecg, np.array(r_peaks)


def pan_tompkins_qrs(ecg, fs):
    """
    Simplified Pan-Tompkins QRS detector.

    Parameters
    ----------
    ecg : ndarray
        ECG signal
    fs : int
        Sampling rate

    Returns
    -------
    r_peaks : ndarray
        Detected R-peak indices
    """
    # Step 1: Bandpass filter (5-15 Hz)
    nyq = fs / 2
    b_bp, a_bp = signal.butter(2, [5/nyq, 15/nyq], btype='band')
    filtered = signal.filtfilt(b_bp, a_bp, ecg)

    # Step 2: Derivative
    h_diff = np.array([-1, -2, 0, 2, 1]) / 8
    diff = np.convolve(filtered, h_diff, mode='same')

    # Step 3: Squaring
    squared = diff**2

    # Step 4: Moving window integration (150 ms)
    win_size = int(0.15 * fs)
    integrator = np.convolve(squared, np.ones(win_size)/win_size, mode='same')

    # Step 5: Thresholding
    threshold = 0.4 * np.max(integrator)
    peaks_mask = integrator > threshold

    # Find peaks
    r_peaks = []
    min_distance = int(0.3 * fs)  # Minimum 300 ms between beats

    i = 0
    while i < len(peaks_mask):
        if peaks_mask[i]:
            # Find the actual R-peak (maximum in the raw ECG)
            start = max(0, i - win_size//2)
            end = min(len(ecg), i + win_size//2)
            r_idx = start + np.argmax(ecg[start:end])

            if len(r_peaks) == 0 or (r_idx - r_peaks[-1]) > min_distance:
                r_peaks.append(r_idx)
            i = r_idx + min_distance
        else:
            i += 1

    return np.array(r_peaks)


# Generate and process ECG
np.random.seed(42)
fs = 360
t, clean_ecg, true_r_peaks = generate_synthetic_ecg(duration=10, fs=fs, heart_rate=72)

# Add noise
baseline_wander = 0.1 * np.sin(2*np.pi*0.15*t) + 0.05*np.sin(2*np.pi*0.3*t)
powerline = 0.05 * np.sin(2*np.pi*60*t)
muscle_noise = 0.03 * np.random.randn(len(t))
noisy_ecg = clean_ecg + baseline_wander + powerline + muscle_noise

# Preprocessing
# Baseline removal (highpass 0.5 Hz)
b_hp, a_hp = signal.butter(2, 0.5/(fs/2), btype='high')
ecg_no_baseline = signal.filtfilt(b_hp, a_hp, noisy_ecg)

# Notch filter for 60 Hz
b_notch, a_notch = signal.iirnotch(60, 30, fs)
ecg_notched = signal.filtfilt(b_notch, a_notch, ecg_no_baseline)

# Bandpass 0.5-40 Hz
b_bp, a_bp = signal.butter(3, [0.5/(fs/2), 40/(fs/2)], btype='band')
ecg_filtered = signal.filtfilt(b_bp, a_bp, noisy_ecg)

# QRS detection
detected_r_peaks = pan_tompkins_qrs(ecg_filtered, fs)

# Plot results
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

axes[0].plot(t, clean_ecg, 'g', alpha=0.7)
axes[0].plot(t[true_r_peaks], clean_ecg[true_r_peaks], 'rv', markersize=10)
axes[0].set_title('Clean ECG with True R-peaks')
axes[0].set_ylabel('Amplitude (mV)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, noisy_ecg, 'r', alpha=0.5)
axes[1].set_title('Noisy ECG (baseline wander + 60Hz + muscle noise)')
axes[1].set_ylabel('Amplitude (mV)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, ecg_filtered, 'b', alpha=0.7)
axes[2].set_title('Filtered ECG (0.5-40 Hz bandpass)')
axes[2].set_ylabel('Amplitude (mV)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(t, ecg_filtered, 'b', alpha=0.5)
axes[3].plot(t[detected_r_peaks], ecg_filtered[detected_r_peaks],
             'rv', markersize=10, label='Detected R-peaks')
axes[3].set_title(f'QRS Detection (Pan-Tompkins): {len(detected_r_peaks)} beats detected')
axes[3].set_xlabel('Time (s)')
axes[3].set_ylabel('Amplitude (mV)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ecg_processing.png', dpi=150, bbox_inches='tight')
plt.show()

# HRV Analysis
rr_intervals = np.diff(detected_r_peaks) / fs * 1000  # Convert to ms
print(f"\nHRV Time-Domain Measures:")
print(f"  Mean RR: {np.mean(rr_intervals):.1f} ms")
print(f"  SDNN:    {np.std(rr_intervals):.1f} ms")
print(f"  RMSSD:   {np.sqrt(np.mean(np.diff(rr_intervals)**2)):.1f} ms")
print(f"  Mean HR: {60000/np.mean(rr_intervals):.1f} bpm")
```

### 18.5 Pitch Detection Demo

```python
import numpy as np
import matplotlib.pyplot as plt


def autocorrelation_pitch(frame, fs, fmin=80, fmax=500):
    """
    Pitch detection using autocorrelation.

    Parameters
    ----------
    frame : ndarray
        Signal frame
    fs : float
        Sampling rate
    fmin, fmax : float
        Min/max expected pitch frequency

    Returns
    -------
    f0 : float
        Estimated pitch frequency (0 if unvoiced)
    """
    N = len(frame)
    # Compute autocorrelation
    r = np.correlate(frame, frame, mode='full')
    r = r[N-1:]  # Take positive lags only
    r = r / r[0]  # Normalize

    # Search range
    lag_min = int(fs / fmax)
    lag_max = min(int(fs / fmin), N - 1)

    # Find the first significant peak
    if lag_max >= len(r):
        lag_max = len(r) - 1

    r_search = r[lag_min:lag_max+1]
    if len(r_search) == 0:
        return 0

    peak_idx = np.argmax(r_search) + lag_min

    # Voiced/unvoiced decision
    if r[peak_idx] > 0.3:  # Threshold
        return fs / peak_idx
    else:
        return 0


def cepstrum_pitch(frame, fs, fmin=80, fmax=500):
    """Pitch detection using cepstrum."""
    N = len(frame)
    # Compute cepstrum
    spectrum = np.fft.fft(frame, n=2*N)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.real(np.fft.ifft(log_spectrum))

    # Search range (quefrency = period in samples)
    q_min = int(fs / fmax)
    q_max = min(int(fs / fmin), N - 1)

    cep_search = cepstrum[q_min:q_max+1]
    if len(cep_search) == 0:
        return 0

    peak_idx = np.argmax(cep_search) + q_min

    if cepstrum[peak_idx] > 0.1:
        return fs / peak_idx
    else:
        return 0


# Generate a test signal with known pitch
fs = 16000
duration = 1.0
t = np.arange(0, duration, 1/fs)

# Create a signal with time-varying pitch (glissando)
f0_start = 150
f0_end = 300
phase = 2 * np.pi * (f0_start * t + (f0_end - f0_start) * t**2 / (2 * duration))
signal_test = np.zeros(len(t))
for k in range(1, 6):
    signal_test += (1/k) * np.sin(k * phase)
signal_test = signal_test / np.max(np.abs(signal_test))
signal_test += 0.05 * np.random.randn(len(t))  # Add noise

# Frame-by-frame pitch detection
frame_length = int(0.04 * fs)  # 40 ms
hop = int(0.01 * fs)  # 10 ms
n_frames = (len(signal_test) - frame_length) // hop

pitches_ac = np.zeros(n_frames)
pitches_cep = np.zeros(n_frames)
frame_times = np.zeros(n_frames)

window = np.hanning(frame_length)

for i in range(n_frames):
    start = i * hop
    frame = signal_test[start:start+frame_length] * window
    frame_times[i] = (start + frame_length/2) / fs

    pitches_ac[i] = autocorrelation_pitch(frame, fs)
    pitches_cep[i] = cepstrum_pitch(frame, fs)

# True pitch trajectory
true_pitch = f0_start + (f0_end - f0_start) * frame_times / duration

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(t, signal_test, 'b', alpha=0.5)
axes[0].set_title('Test Signal (Glissando 150-300 Hz)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

axes[1].plot(frame_times, true_pitch, 'k-', linewidth=2, label='True pitch')
axes[1].plot(frame_times, pitches_ac, 'ro', markersize=3, alpha=0.7,
             label='Autocorrelation')
axes[1].plot(frame_times, pitches_cep, 'bx', markersize=3, alpha=0.7,
             label='Cepstrum')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_title('Pitch Detection Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 500])

# Autocorrelation and cepstrum of one frame
frame_idx = n_frames // 2
start = frame_idx * hop
frame = signal_test[start:start+frame_length] * window

r = np.correlate(frame, frame, mode='full')
r = r[len(frame)-1:]
r = r / r[0]

spectrum = np.fft.fft(frame, n=2*frame_length)
cepstrum_vals = np.real(np.fft.ifft(np.log(np.abs(spectrum) + 1e-10)))

lag = np.arange(len(r)) / fs * 1000  # ms
axes[2].plot(lag[:int(fs/80)], r[:int(fs/80)], 'b', alpha=0.7,
             label='Autocorrelation')
axes[2].set_xlabel('Lag (ms)')
axes[2].set_ylabel('Normalized Autocorrelation')
axes[2].set_title('Autocorrelation of Middle Frame')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pitch_detection.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 19. Exercises

### Exercise 1: Audio Effects Chain

(a) Implement a Schroeder reverberator using 4 parallel comb filters (with delays 29.7, 37.1, 41.1, 43.7 ms and gains 0.742, 0.733, 0.715, 0.697) and 2 series allpass filters (delays 5.0, 1.7 ms, gain 0.7). Apply to a short impulse and plot the resulting impulse response.

(b) Generate a simple melody (e.g., "Twinkle Twinkle Little Star" using sine waves at the appropriate frequencies). Apply the reverb and listen (save as WAV file if possible).

(c) Implement a flanger effect and demonstrate the characteristic comb-filter sweep by plotting the time-varying magnitude response.

### Exercise 2: Pitch Detection Robustness

(a) Generate a pure tone at 220 Hz and verify that both autocorrelation and cepstrum methods detect it correctly.

(b) Add harmonic content (1st through 5th harmonics with decreasing amplitude). Do the pitch detectors still find the fundamental?

(c) Add noise at SNR = 20, 10, 5, and 0 dB. Plot the pitch detection accuracy vs SNR for both methods.

(d) Create a signal with missing fundamental (only harmonics 2, 3, 4 of 100 Hz). Can the autocorrelation method still detect 100 Hz? What about the cepstrum?

### Exercise 3: Digital Modulation BER

(a) Implement BPSK, QPSK, and 16-QAM modulation and demodulation.

(b) Simulate transmission over an AWGN channel for $E_b/N_0$ from 0 to 20 dB. Plot the BER curves for all three schemes.

(c) Compare with the theoretical BER:
- BPSK: $P_b = Q(\sqrt{2E_b/N_0})$
- QPSK: Same as BPSK (per bit)
- 16-QAM: $P_b \approx \frac{3}{8}Q(\sqrt{4E_b/(5N_0)})$

(d) Implement Gray coding for 16-QAM and show that it improves the BER compared to natural binary mapping.

### Exercise 4: OFDM System

(a) Implement a basic OFDM transmitter and receiver with 64 subcarriers, 16-QAM modulation, and a cyclic prefix of length 16.

(b) Simulate transmission through a multipath channel $h = [1, 0, 0.5, 0, 0.2]$ with AWGN.

(c) Show that the cyclic prefix enables simple one-tap equalization. Plot the constellation diagram before and after equalization.

(d) Remove the cyclic prefix and demonstrate the resulting ISI.

### Exercise 5: Radar Waveform Design

(a) Generate chirp pulses with time-bandwidth products of 10, 50, and 200. Apply matched filtering and compare the compressed pulse widths and sidelobe levels.

(b) Apply a Hamming window to the matched filter. How does it affect the mainlobe width and sidelobe level?

(c) Simulate a scenario with two targets at ranges 10 km and 10.03 km. For each TBP, determine whether the targets are resolved.

(d) Compute and plot the 2D ambiguity function $|\chi(\tau, f_d)|$ for a chirp pulse with TBP = 50. Identify the range-Doppler coupling ridge.

### Exercise 6: ECG Analysis Pipeline

(a) Generate a synthetic ECG with heart rate 75 bpm and add:
- Baseline wander (0.2 Hz sinusoid, amplitude 0.3 mV)
- 50 Hz powerline noise (amplitude 0.1 mV)
- Random noise (SNR = 20 dB)

(b) Implement the full preprocessing pipeline: highpass filter (0.5 Hz), notch filter (50 Hz), bandpass (1-40 Hz). Show the signal after each stage.

(c) Implement the Pan-Tompkins QRS detector. Count the number of detected beats and compare with ground truth.

(d) Extract RR intervals and compute SDNN, RMSSD, and pNN50. Compare with the known heart rate variability.

(e) Compute the HRV power spectrum and identify the LF and HF bands.

### Exercise 7: EEG Spectral Analysis

(a) Simulate an EEG signal as a sum of:
- Alpha band (10 Hz): dominant when eyes closed
- Beta band (20 Hz): present during concentration
- Background noise (1/f spectrum)

(b) Compute the power spectral density using Welch's method (2-second windows, 50% overlap). Identify the alpha and beta peaks.

(c) Simulate an "eyes open/closed" experiment: alpha power increases from $t = 2$-4 s (eyes closed) and decreases elsewhere. Use the STFT to show the time-varying alpha band power.

(d) Compute the relative band power (alpha/total, beta/total) over time using a sliding window.

---

## 20. Summary

| Domain | Key Technique | Signal Processing Tool |
|---|---|---|
| Audio | Delay effects, reverb | Delay lines, comb/allpass filters |
| Audio | Pitch detection | Autocorrelation, cepstrum |
| Audio | Equalization | Parametric biquad filters |
| Audio | Speech coding | Linear prediction (LPC) |
| Communications | Analog modulation | AM, FM, PM |
| Communications | Digital modulation | PSK, QAM (I-Q processing) |
| Communications | Pulse shaping | Raised cosine, matched filter |
| Communications | OFDM | FFT/IFFT, cyclic prefix |
| Communications | Equalization | ZF, MMSE, adaptive (LMS/RLS) |
| Radar | Range detection | Matched filter |
| Radar | Pulse compression | Chirp, matched filter |
| Radar | Waveform design | Ambiguity function |
| Biomedical | ECG preprocessing | Bandpass, notch filter |
| Biomedical | QRS detection | Pan-Tompkins algorithm |
| Biomedical | EEG analysis | PSD, band power, STFT |
| Biomedical | HRV | Time/frequency domain measures |

**Key takeaways**:
1. Audio effects are built from fundamental building blocks: delay lines, filters, and modulators.
2. Pitch detection reduces to finding periodicity via autocorrelation or spectral analysis via the cepstrum.
3. Digital modulation maps bits to points in the I-Q plane; higher-order schemes trade SNR for spectral efficiency.
4. OFDM converts a frequency-selective channel into parallel flat channels using the FFT.
5. Radar pulse compression achieves both high energy (long pulse) and fine range resolution (wide bandwidth).
6. The ambiguity function completely characterizes a radar waveform's range-velocity resolution.
7. ECG processing combines bandpass filtering, notch filtering, and the Pan-Tompkins algorithm for reliable QRS detection.
8. EEG spectral analysis reveals brain states through the power distribution across frequency bands.
9. Every application in this lesson builds directly on the fundamentals from Lessons 1-15.

---

## 21. References

1. J.O. Smith III, *Physical Audio Signal Processing*, W3K Publishing, 2010. (Available online)
2. J.G. Proakis and M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008.
3. M.A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014.
4. M.I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill, 2001.
5. J. Pan and W.J. Tompkins, "A real-time QRS detection algorithm," *IEEE Trans. Biomedical Engineering*, vol. BME-32, no. 3, pp. 230-236, 1985.
6. R. Rangayyan, *Biomedical Signal Analysis*, 2nd ed., Wiley-IEEE Press, 2015.
7. S. Haykin and M. Moher, *Communication Systems*, 5th ed., Wiley, 2009.
8. A.V. Oppenheim and R.W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010.
9. U.R. Acharya et al., "Heart rate variability: a review," *Medical and Biological Engineering and Computing*, vol. 44, pp. 1031-1051, 2006.

---

**Previous**: [15. Image Signal Processing](./15_Image_Signal_Processing.md) | [Overview](./00_Overview.md)
