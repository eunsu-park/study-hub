# Signal Processing - Overview

## Introduction

Signal processing is the mathematical framework for analyzing, modifying, and synthesizing signals — quantities that convey information about the behavior or attributes of a physical phenomenon. From speech recognition and audio engineering to radar systems and medical imaging, signal processing provides the foundational tools that power modern technology.

This course takes a systematic journey through signal processing, beginning with the continuous-time and discrete-time descriptions of signals and systems, proceeding through the major transform-domain techniques (Fourier, Z, and wavelet), covering the design and implementation of digital filters, and culminating in applications such as spectral estimation, adaptive filtering, image processing, and communications.

Each lesson pairs rigorous mathematical derivation with Python implementations using NumPy, SciPy, and Matplotlib, enabling direct computation and visualization of the concepts.

---

## Learning Objectives

By the end of this course, you will be able to:

1. **Classify and characterize** signals (continuous/discrete, deterministic/random, energy/power) and systems (LTI, causal, stable)
2. **Apply transform-domain analysis** using Fourier series, Fourier transform, DFT/FFT, and Z-transform
3. **Design and implement** both FIR and IIR digital filters for specific frequency-response specifications
4. **Understand sampling theory** including the Nyquist theorem, aliasing, and reconstruction
5. **Perform spectral estimation** using classical and parametric methods
6. **Implement adaptive filters** for real-time signal processing tasks
7. **Analyze non-stationary signals** using time-frequency techniques (STFT, wavelets)
8. **Apply signal processing** to practical domains: audio, image, communications, and radar

---

## Prerequisites

| Topic | Where to Study | Why Needed |
|-------|---------------|------------|
| Calculus (single and multivariable) | University calculus course | Integrals, derivatives, limits used throughout |
| Linear algebra | [Mathematical_Methods L03](../Mathematical_Methods/03_Linear_Algebra.md) | Vector spaces, eigenvalues, matrix operations |
| Complex numbers | [Mathematical_Methods L02](../Mathematical_Methods/02_Complex_Numbers.md) | Euler's formula, phasor representation |
| Fourier series & transforms | [Mathematical_Methods L07-L08](../Mathematical_Methods/07_Fourier_Series.md) | Mathematical foundation (this course applies them to signal processing) |
| Laplace transform | [Mathematical_Methods L15](../Mathematical_Methods/15_Laplace_Transform.md) | Transfer functions, system analysis |
| Basic Python (NumPy, Matplotlib) | [Python topic](../Python/00_Overview.md) | All code examples use Python |

---

## Lesson List

| No. | Filename | Title | Key Topics |
|-----|----------|-------|------------|
| 00 | 00_Overview.md | Overview | Course introduction and study guide |
| 01 | [01_Signals_and_Systems.md](./01_Signals_and_Systems.md) | Signals and Systems | Continuous/discrete signals, energy/power, basic signals, system properties |
| 02 | [02_LTI_Systems_and_Convolution.md](./02_LTI_Systems_and_Convolution.md) | LTI Systems and Convolution | Linearity, time-invariance, convolution integral/sum, impulse response |
| 03 | [03_Fourier_Series_and_Applications.md](./03_Fourier_Series_and_Applications.md) | Fourier Series and Applications | Periodic signal decomposition, spectrum, Parseval's theorem |
| 04 | [04_Continuous_Fourier_Transform.md](./04_Continuous_Fourier_Transform.md) | Continuous Fourier Transform | CTFT properties, frequency domain analysis, filtering |
| 05 | 05_Sampling_and_Reconstruction.md | Sampling and Reconstruction | Nyquist theorem, aliasing, anti-aliasing, DAC reconstruction |
| 06 | 06_Discrete_Fourier_Transform.md | Discrete Fourier Transform | DFT definition, FFT algorithm, zero-padding, leakage |
| 07 | 07_Z_Transform.md | Z-Transform | Z-transform, ROC, inverse Z-transform, transfer functions |
| 08 | 08_Digital_Filter_Fundamentals.md | Digital Filter Fundamentals | FIR vs IIR, frequency response, linear/nonlinear phase |
| 09 | 09_FIR_Filter_Design.md | FIR Filter Design | Windowing method, Parks-McClellan, linear phase constraints |
| 10 | 10_IIR_Filter_Design.md | IIR Filter Design | Butterworth, Chebyshev, elliptic, bilinear transform |
| 11 | 11_Multirate_Processing.md | Multirate Signal Processing | Decimation, interpolation, polyphase filters, sample rate conversion |
| 12 | 12_Spectral_Analysis.md | Spectral Analysis | Periodogram, Welch method, AR/ARMA parametric models |
| 13 | 13_Adaptive_Filters.md | Adaptive Filters | LMS, NLMS, RLS, echo cancellation, noise cancellation |
| 14 | 14_Time_Frequency_Analysis.md | Time-Frequency Analysis | STFT, spectrogram, wavelet transform, CWT, DWT |
| 15 | 15_Image_Signal_Processing.md | Image Signal Processing | 2D DFT, spatial filters, edge detection, image enhancement |
| 16 | 16_Applications.md | Applications | Audio processing, communications, radar/sonar, biomedical signals |

---

## Required Libraries

```bash
pip install numpy scipy matplotlib
```

- **NumPy**: Array operations, FFT, linear algebra
- **SciPy**: Signal processing (`scipy.signal`), filter design, spectral analysis, wavelets
- **Matplotlib**: Signal visualization, spectrograms, frequency response plots

### Optional Libraries

```bash
pip install soundfile PyWavelets librosa
```

- **soundfile**: Reading/writing audio files
- **PyWavelets** (`pywt`): Wavelet transforms (CWT, DWT)
- **librosa**: Audio feature extraction and analysis

---

## Recommended Study Path

### Phase 1: Foundations (Lessons 01-04) — 2-3 weeks

```
01 Signals and Systems
        │
        ▼
02 LTI Systems and Convolution
        │
        ▼
03 Fourier Series ──▶ 04 Continuous Fourier Transform
```

- Signal classification and fundamental system properties
- Convolution as the core operation for LTI systems
- Frequency-domain decomposition of periodic and aperiodic signals
- Transform properties and their physical interpretation

**Goal**: Build a solid time-domain and frequency-domain foundation

### Phase 2: Discrete-Time Framework (Lessons 05-07) — 2-3 weeks

```
04 CTFT ──▶ 05 Sampling and Reconstruction
                        │
                        ▼
              06 DFT and FFT ──▶ 07 Z-Transform
```

- Bridge from continuous to discrete through sampling theory
- Computational frequency analysis with the DFT and FFT
- Z-transform as the discrete-time counterpart of Laplace

**Goal**: Master the mathematical tools for discrete-time signal analysis

### Phase 3: Digital Filter Design (Lessons 08-10) — 2-3 weeks

```
07 Z-Transform ──▶ 08 Digital Filter Fundamentals
                            │
                    ┌───────┴───────┐
                    ▼               ▼
           09 FIR Design     10 IIR Design
```

- Understand the tradeoffs between FIR and IIR structures
- Design filters to meet precise frequency-response specifications
- Implement filters in Python using `scipy.signal`

**Goal**: Be able to design and implement digital filters for any specification

### Phase 4: Advanced Analysis (Lessons 11-14) — 2-3 weeks

```
11 Multirate Processing    12 Spectral Analysis
                                    │
                                    ▼
                           13 Adaptive Filters
                                    │
                                    ▼
                        14 Time-Frequency Analysis
```

- Sample rate conversion and efficient multirate structures
- Classical and parametric spectral estimation
- Adaptive algorithms for real-time tracking
- Joint time-frequency representations (STFT, wavelets)

**Goal**: Handle advanced analysis scenarios including non-stationary and real-time signals

### Phase 5: Applications (Lessons 15-16) — 1-2 weeks

```
15 Image Signal Processing ──▶ 16 Applications
```

- Extend 1D processing to 2D images
- Capstone applications in audio, communications, radar, and biomedical

**Goal**: Apply signal processing techniques to real-world problems

---

## Cross-References to Other Topics

| Related Topic | Connection |
|---------------|-----------|
| [Mathematical Methods](../Mathematical_Methods/00_Overview.md) | Fourier analysis (L07-08), Laplace transform (L15), complex analysis (L14) |
| [Numerical Simulation](../Numerical_Simulation/00_Overview.md) | PDE discretization (L07-08), spectral methods (L21), FDTD (L15-16) |
| [Deep Learning](../Deep_Learning/00_Overview.md) | 1D/2D convolution in CNNs, attention as weighted filtering |
| [Computer Vision](../Computer_Vision/00_Overview.md) | Image filtering, edge detection, frequency-domain image processing |
| [Data Science](../Data_Science/00_Overview.md) | Time series analysis, spectral density estimation |
| [Plasma Physics](../Plasma_Physics/00_Overview.md) | Fourier analysis of plasma waves, spectral methods in simulation |

---

## References

### Textbooks

1. **Oppenheim, A. V. & Willsky, A. S.** *Signals and Systems* (2nd ed.), Prentice Hall, 1997 — The classic reference for continuous and discrete signals and systems
2. **Oppenheim, A. V. & Schafer, R. W.** *Discrete-Time Signal Processing* (3rd ed.), Pearson, 2010 — Definitive treatment of discrete-time theory and filter design
3. **Haykin, S. & Van Veen, B.** *Signals and Systems* (2nd ed.), Wiley, 2003 — Excellent balance of theory and applications
4. **Proakis, J. G. & Manolakis, D. G.** *Digital Signal Processing* (4th ed.), Pearson, 2006 — Comprehensive DSP reference with MATLAB/Python examples
5. **Mallat, S.** *A Wavelet Tour of Signal Processing* (3rd ed.), Academic Press, 2009 — Authoritative reference on wavelet theory

### Online Resources

- [MIT OpenCourseWare 6.003 — Signals and Systems](https://ocw.mit.edu/courses/6-003-signals-and-systems-fall-2011/)
- [MIT OpenCourseWare 6.341 — Discrete-Time Signal Processing](https://ocw.mit.edu/courses/6-341-discrete-time-signal-processing-fall-2005/)
- [SciPy Signal Processing Documentation](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [Think DSP — Free Python DSP textbook](https://greenteapress.com/wp/think-dsp/)
