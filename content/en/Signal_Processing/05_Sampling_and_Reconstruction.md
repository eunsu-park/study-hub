# Sampling and Reconstruction

## Overview

Sampling is the bridge between continuous-time (analog) and discrete-time (digital) signals. Understanding the sampling process, its limitations, and how to faithfully reconstruct the original signal is fundamental to all digital signal processing. This lesson covers the Nyquist-Shannon sampling theorem, aliasing, anti-aliasing strategies, reconstruction techniques, and practical ADC/DAC considerations.

**Learning Objectives:**
- Understand the mathematical framework for ideal sampling
- State and prove the Nyquist-Shannon sampling theorem
- Identify and prevent aliasing
- Implement reconstruction using sinc interpolation and hold circuits
- Analyze practical ADC/DAC systems and the benefits of oversampling

**Prerequisites:** [04. Fourier Transform and Frequency Domain](04_Fourier_Transform.md)

---

## 1. Sampling of Continuous-Time Signals

### 1.1 Why Sample?

Digital systems (computers, DSP chips, microcontrollers) cannot process continuous-time signals directly. We must convert analog signals to sequences of numbers. The key question: **under what conditions can we perfectly recover the original signal from its samples?**

### 1.2 The Sampling Process

Given a continuous-time signal $x(t)$, sampling at interval $T_s$ (the sampling period) produces a discrete-time signal:

$$x[n] = x(nT_s), \quad n \in \mathbb{Z}$$

The sampling frequency (sample rate) is:

$$f_s = \frac{1}{T_s} \quad \text{(Hz)}$$

The angular sampling frequency is:

$$\Omega_s = 2\pi f_s = \frac{2\pi}{T_s} \quad \text{(rad/s)}$$

```python
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_sampling():
    """Demonstrate the basic sampling process."""
    # Continuous-time signal (simulated with dense grid)
    t_cont = np.linspace(0, 1, 10000)
    f_signal = 5  # 5 Hz sine wave
    x_cont = np.sin(2 * np.pi * f_signal * t_cont)

    # Sample at different rates
    sample_rates = [10, 20, 50]  # Hz

    fig, axes = plt.subplots(len(sample_rates), 1, figsize=(12, 10))

    for ax, fs in zip(axes, sample_rates):
        Ts = 1.0 / fs
        n_samples = np.arange(0, 1, Ts)
        x_samples = np.sin(2 * np.pi * f_signal * n_samples)

        ax.plot(t_cont, x_cont, 'b-', alpha=0.5, label='Continuous signal')
        ax.stem(n_samples, x_samples, linefmt='r-', markerfmt='ro',
                basefmt='k-', label=f'Samples (fs={fs} Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Sampling at fs = {fs} Hz (Ts = {Ts:.3f} s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sampling_basic.png', dpi=150)
    plt.show()

demonstrate_sampling()
```

---

## 2. Ideal Sampling with Impulse Train

### 2.1 Mathematical Model

Ideal sampling multiplies $x(t)$ by an impulse train (Dirac comb):

$$s(t) = \sum_{n=-\infty}^{\infty} \delta(t - nT_s)$$

The sampled signal is:

$$x_s(t) = x(t) \cdot s(t) = \sum_{n=-\infty}^{\infty} x(nT_s) \, \delta(t - nT_s)$$

### 2.2 Frequency Domain Analysis

The Fourier transform of the impulse train is another impulse train:

$$S(j\Omega) = \frac{2\pi}{T_s} \sum_{k=-\infty}^{\infty} \delta\!\left(\Omega - k\Omega_s\right)$$

Since multiplication in time corresponds to convolution in frequency:

$$X_s(j\Omega) = \frac{1}{2\pi} X(j\Omega) * S(j\Omega)$$

Substituting and simplifying:

$$\boxed{X_s(j\Omega) = \frac{1}{T_s} \sum_{k=-\infty}^{\infty} X\!\left(j(\Omega - k\Omega_s)\right)}$$

This is the **fundamental result of sampling**: the spectrum of the sampled signal is a periodic repetition of the original spectrum $X(j\Omega)$, shifted by integer multiples of $\Omega_s$ and scaled by $1/T_s$.

### 2.3 Visualizing Spectral Replication

```python
def visualize_spectral_replication():
    """Show how sampling replicates the spectrum periodically."""
    # Original signal: bandlimited with max frequency B
    B = 5  # Hz (bandwidth)
    f = np.linspace(-20, 20, 2000)

    # Triangular spectrum (simplified representation)
    X_orig = np.maximum(0, 1 - np.abs(f) / B)

    # Sampling frequency
    fs_values = [15, 10, 7]  # Hz
    titles = [
        f'fs = 15 Hz > 2B = 10 Hz (No aliasing)',
        f'fs = 10 Hz = 2B (Critical sampling)',
        f'fs = 7 Hz < 2B (Aliasing!)'
    ]

    fig, axes = plt.subplots(len(fs_values) + 1, 1, figsize=(14, 12))

    # Original spectrum
    axes[0].plot(f, X_orig, 'b-', linewidth=2)
    axes[0].fill_between(f, X_orig, alpha=0.3)
    axes[0].set_title('Original Spectrum X(f)')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('|X(f)|')
    axes[0].set_xlim(-20, 20)
    axes[0].grid(True, alpha=0.3)

    for ax, fs, title in zip(axes[1:], fs_values, titles):
        # Sum of shifted replicas
        X_sampled = np.zeros_like(f)
        for k in range(-5, 6):
            X_sampled += np.maximum(0, 1 - np.abs(f - k * fs) / B)

        ax.plot(f, X_sampled, 'r-', linewidth=2)
        ax.fill_between(f, X_sampled, alpha=0.3, color='red')

        # Show individual replicas as dashed
        for k in range(-3, 4):
            replica = np.maximum(0, 1 - np.abs(f - k * fs) / B)
            ax.plot(f, replica, 'b--', alpha=0.4, linewidth=1)

        # Mark Nyquist boundaries
        ax.axvline(fs / 2, color='green', linestyle=':', linewidth=1.5,
                    label=f'fs/2 = {fs/2} Hz')
        ax.axvline(-fs / 2, color='green', linestyle=':', linewidth=1.5)

        ax.set_title(title)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('|Xs(f)|')
        ax.set_xlim(-20, 20)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('spectral_replication.png', dpi=150)
    plt.show()

visualize_spectral_replication()
```

---

## 3. The Nyquist-Shannon Sampling Theorem

### 3.1 Statement of the Theorem

**Theorem (Nyquist-Shannon):** A bandlimited signal $x(t)$ with no frequency components above $B$ Hz (i.e., $X(j\Omega) = 0$ for $|\Omega| > 2\pi B$) is completely determined by its samples $x[n] = x(nT_s)$ if the sampling rate satisfies:

$$\boxed{f_s > 2B}$$

or equivalently:

$$T_s < \frac{1}{2B}$$

### 3.2 Key Definitions

| Term | Definition | Expression |
|------|-----------|------------|
| **Nyquist rate** | Minimum sampling rate to avoid aliasing | $f_{\text{Nyquist}} = 2B$ |
| **Nyquist frequency** | Highest frequency representable at sampling rate $f_s$ | $f_{\text{max}} = f_s / 2$ |
| **Nyquist interval** | Maximum spacing between samples | $T_{\text{Nyquist}} = 1/(2B)$ |

> **Important distinction:** The Nyquist *rate* depends on the signal ($2B$), while the Nyquist *frequency* depends on the sampling rate ($f_s/2$). They are equal only at the critical sampling condition.

### 3.3 Proof Sketch

1. The sampled spectrum is $X_s(j\Omega) = \frac{1}{T_s} \sum_k X(j(\Omega - k\Omega_s))$.
2. If $\Omega_s > 2(2\pi B)$, the shifted replicas do not overlap.
3. We can recover $X(j\Omega)$ by applying an ideal lowpass filter with cutoff $\Omega_s / 2$.
4. Therefore $x(t)$ is fully recoverable from its samples.

### 3.4 Reconstruction Formula

When the sampling theorem is satisfied, the original signal can be reconstructed exactly:

$$\boxed{x(t) = \sum_{n=-\infty}^{\infty} x[n] \, \operatorname{sinc}\!\left(\frac{t - nT_s}{T_s}\right)}$$

where $\operatorname{sinc}(u) = \frac{\sin(\pi u)}{\pi u}$.

This is the **Whittaker-Shannon interpolation formula** (ideal sinc interpolation).

```python
def demonstrate_nyquist_theorem():
    """Demonstrate the sampling theorem with different rates."""
    # Signal: sum of sinusoids
    f1, f2 = 3, 7  # Hz
    B = 7  # Bandwidth
    nyquist_rate = 2 * B  # 14 Hz

    t_cont = np.linspace(0, 1, 10000)
    x_cont = np.sin(2 * np.pi * f1 * t_cont) + 0.5 * np.sin(2 * np.pi * f2 * t_cont)

    sample_rates = [30, 14, 10]  # Above, at, below Nyquist rate
    labels = [
        f'fs = 30 Hz > 2B = {nyquist_rate} Hz (Well sampled)',
        f'fs = 14 Hz = 2B = {nyquist_rate} Hz (Critical)',
        f'fs = 10 Hz < 2B = {nyquist_rate} Hz (Undersampled!)'
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    for ax, fs, label in zip(axes, sample_rates, labels):
        Ts = 1.0 / fs
        n_samples = np.arange(0, 1, Ts)
        x_samples = np.sin(2 * np.pi * f1 * n_samples) + \
                    0.5 * np.sin(2 * np.pi * f2 * n_samples)

        # Sinc interpolation to reconstruct
        t_recon = np.linspace(0, 1, 10000)
        x_recon = np.zeros_like(t_recon)
        for i, ns in enumerate(n_samples):
            x_recon += x_samples[i] * np.sinc((t_recon - ns) / Ts)

        ax.plot(t_cont, x_cont, 'b-', alpha=0.4, linewidth=1, label='Original')
        ax.plot(t_recon, x_recon, 'r-', linewidth=1.5, label='Reconstructed')
        ax.stem(n_samples, x_samples, linefmt='g-', markerfmt='go',
                basefmt='k-', label='Samples')
        ax.set_title(label)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nyquist_theorem.png', dpi=150)
    plt.show()

demonstrate_nyquist_theorem()
```

---

## 4. Aliasing

### 4.1 What Is Aliasing?

Aliasing occurs when the sampling rate is insufficient ($f_s < 2B$), causing the spectral replicas to overlap. High-frequency components "fold back" and appear as lower frequencies that are **indistinguishable** from genuine low-frequency content.

The aliased frequency of a component at frequency $f_0$ sampled at rate $f_s$ is:

$$f_{\text{alias}} = |f_0 - k \cdot f_s|, \quad k = \operatorname{round}(f_0 / f_s)$$

More precisely, the apparent frequency after sampling is:

$$f_{\text{apparent}} = \left| \left( (f_0 + f_s/2) \bmod f_s \right) - f_s/2 \right|$$

### 4.2 Aliasing Examples

```python
def demonstrate_aliasing():
    """Show aliasing with concrete sinusoidal examples."""
    fs = 20  # Sampling frequency: 20 Hz
    nyquist = fs / 2  # 10 Hz

    # Three signals: below, at, and above Nyquist
    freqs = [3, 10, 17]  # Hz
    # f=3 Hz: well below Nyquist, sampled correctly
    # f=10 Hz: at Nyquist, ambiguous
    # f=17 Hz: above Nyquist, aliases to |17 - 20| = 3 Hz

    t_cont = np.linspace(0, 1, 10000)
    Ts = 1.0 / fs
    t_samples = np.arange(0, 1, Ts)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    for ax, f in zip(axes, freqs):
        x_cont = np.cos(2 * np.pi * f * t_cont)
        x_samples = np.cos(2 * np.pi * f * t_samples)

        ax.plot(t_cont, x_cont, 'b-', alpha=0.5, linewidth=1,
                label=f'Original: {f} Hz')

        # Show the aliased frequency
        if f > nyquist:
            f_alias = fs - f  # For f < fs
            x_alias = np.cos(2 * np.pi * f_alias * t_cont)
            ax.plot(t_cont, x_alias, 'r--', linewidth=1.5,
                    label=f'Alias: {f_alias} Hz')

        ax.stem(t_samples, x_samples, linefmt='g-', markerfmt='go',
                basefmt='k-', label='Samples')
        ax.set_title(f'f = {f} Hz, fs = {fs} Hz, Nyquist = {nyquist} Hz')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('aliasing_demo.png', dpi=150)
    plt.show()

demonstrate_aliasing()
```

### 4.3 Aliasing in Real-World Scenarios

| Scenario | Cause | Effect |
|----------|-------|--------|
| **Audio** | Recording > Nyquist/2 frequencies | Harsh, metallic artifacts |
| **Video** | Spinning wheels in film (24 fps) | Wheels appear to rotate backward |
| **Photography** | Fine patterns (Moire effect) | False color/pattern artifacts |
| **Medical imaging** | Insufficient spatial sampling | Spatial aliasing in MRI/CT |
| **Radar** | PRF too low for target velocity | Velocity ambiguity |

### 4.4 The Wagon Wheel Effect

A classic example of temporal aliasing is the "wagon wheel effect" in film:

- A wheel rotates at $f_{\text{wheel}}$ Hz (revolutions per second)
- Camera samples at $f_s = 24$ fps
- If $f_{\text{wheel}} > 12$ Hz, the wheel appears to rotate backward or at the wrong speed

```python
def wagon_wheel_effect():
    """Simulate the wagon wheel aliasing effect."""
    fps = 24  # Camera frame rate

    # Wheel rotation frequencies
    rotation_freqs = [5, 12, 23, 25, 47, 48]

    print("Wagon Wheel Effect (Camera: 24 fps)")
    print("=" * 50)

    for f_rot in rotation_freqs:
        # Apparent frequency after aliasing
        f_apparent = f_rot % fps
        if f_apparent > fps / 2:
            f_apparent = f_apparent - fps
        print(f"  Actual: {f_rot:3d} Hz -> Apparent: {f_apparent:+6.1f} Hz "
              f"({'backward' if f_apparent < 0 else 'forward'})")

wagon_wheel_effect()
```

Output:
```
Wagon Wheel Effect (Camera: 24 fps)
==================================================
  Actual:   5 Hz -> Apparent:   +5.0 Hz (forward)
  Actual:  12 Hz -> Apparent:  +12.0 Hz (forward)
  Actual:  23 Hz -> Apparent:   -1.0 Hz (backward)
  Actual:  25 Hz -> Apparent:   +1.0 Hz (forward)
  Actual:  47 Hz -> Apparent:   -1.0 Hz (backward)
  Actual:  48 Hz -> Apparent:   +0.0 Hz (forward)
```

---

## 5. Anti-Aliasing Filters

### 5.1 Purpose

An **anti-aliasing filter** is a lowpass filter applied to the continuous-time signal *before* sampling. It removes (attenuates) frequency components above $f_s/2$ to prevent aliasing.

### 5.2 Ideal vs. Practical Anti-Aliasing

| Property | Ideal Filter | Practical Filter |
|----------|-------------|-----------------|
| Cutoff | Brick-wall at $f_s/2$ | Gradual roll-off |
| Transition band | Zero width | Finite width |
| Stopband attenuation | Infinite | Finite (e.g., -80 dB) |
| Causality | Non-causal | Causal |
| Implementation | Impossible | Analog circuit |

### 5.3 Practical Considerations

Since ideal brick-wall filters are unrealizable, practical anti-aliasing involves:

1. **Guard band**: Sample somewhat higher than $2B$ to allow for filter roll-off
2. **Filter order**: Higher-order filters have steeper roll-off but more phase distortion
3. **Common choices**: Butterworth (maximally flat), Chebyshev (steeper roll-off), Elliptic (steepest)

```python
from scipy import signal

def design_anti_aliasing_filter():
    """Design and compare anti-aliasing filters."""
    fs = 1000  # Sampling frequency
    f_nyquist = fs / 2
    f_cutoff = 400  # Desired signal bandwidth

    # Normalized cutoff for analog filter design
    Wn = 2 * np.pi * f_cutoff  # Angular frequency

    orders = [2, 4, 8]
    f_plot = np.logspace(1, 3.5, 1000)  # 10 Hz to ~3 kHz
    w_plot = 2 * np.pi * f_plot

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Butterworth filters of different orders
    for order in orders:
        b, a = signal.butter(order, Wn, btype='low', analog=True)
        w, H = signal.freqs(b, a, worN=w_plot)

        mag_db = 20 * np.log10(np.abs(H))
        phase_deg = np.angle(H, deg=True)

        axes[0].semilogx(w / (2 * np.pi), mag_db,
                         label=f'Butterworth order {order}')
        axes[1].semilogx(w / (2 * np.pi), phase_deg,
                         label=f'Butterworth order {order}')

    # Mark important frequencies
    for ax in axes:
        ax.axvline(f_cutoff, color='green', linestyle='--', alpha=0.7,
                   label=f'Cutoff = {f_cutoff} Hz')
        ax.axvline(f_nyquist, color='red', linestyle='--', alpha=0.7,
                   label=f'Nyquist = {f_nyquist} Hz')
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('Anti-Aliasing Filter: Magnitude Response')
    axes[0].set_ylim(-80, 5)

    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].set_title('Anti-Aliasing Filter: Phase Response')

    plt.tight_layout()
    plt.savefig('anti_aliasing_filter.png', dpi=150)
    plt.show()

design_anti_aliasing_filter()
```

---

## 6. Reconstruction

### 6.1 Ideal Reconstruction (Sinc Interpolation)

The ideal reconstruction uses the Whittaker-Shannon interpolation formula:

$$x_r(t) = \sum_{n=-\infty}^{\infty} x[n] \, \operatorname{sinc}\!\left(\frac{t - nT_s}{T_s}\right)$$

In the frequency domain, this is equivalent to applying an ideal lowpass filter with:
- Gain: $T_s$ in the passband
- Cutoff frequency: $\Omega_s / 2$

**Properties of ideal sinc interpolation:**
- Passes through all sample points exactly: $x_r(nT_s) = x[n]$
- Perfect reconstruction if sampling theorem is satisfied
- Non-causal (requires all past and future samples)
- Impractical due to infinite support of sinc function

```python
def sinc_interpolation_demo():
    """Demonstrate ideal sinc interpolation for reconstruction."""
    # Original signal
    f1, f2 = 3, 7
    t_true = np.linspace(0, 1, 10000)
    x_true = np.sin(2 * np.pi * f1 * t_true) + 0.5 * np.cos(2 * np.pi * f2 * t_true)

    # Sample at fs = 20 Hz (Nyquist rate = 14 Hz, so well sampled)
    fs = 20
    Ts = 1.0 / fs
    n_samples = np.arange(0, 1, Ts)
    x_samples = np.sin(2 * np.pi * f1 * n_samples) + \
                0.5 * np.cos(2 * np.pi * f2 * n_samples)

    # Sinc interpolation
    t_recon = np.linspace(0, 1, 10000)
    x_recon = np.zeros_like(t_recon)

    for n, (tn, xn) in enumerate(zip(n_samples, x_samples)):
        x_recon += xn * np.sinc((t_recon - tn) / Ts)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Reconstruction
    axes[0].plot(t_true, x_true, 'b-', linewidth=1, alpha=0.5, label='Original')
    axes[0].plot(t_recon, x_recon, 'r-', linewidth=1.5, label='Sinc reconstructed')
    axes[0].stem(n_samples, x_samples, linefmt='g-', markerfmt='go',
                 basefmt='k-', label='Samples')
    axes[0].set_title(f'Sinc Interpolation (fs = {fs} Hz)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction error
    # Interpolate true signal at same points for comparison
    x_true_at_recon = np.sin(2 * np.pi * f1 * t_recon) + \
                      0.5 * np.cos(2 * np.pi * f2 * t_recon)
    error = x_true_at_recon - x_recon

    axes[1].plot(t_recon, error, 'k-', linewidth=0.5)
    axes[1].set_title(f'Reconstruction Error (max = {np.max(np.abs(error)):.6f})')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Error')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sinc_interpolation.png', dpi=150)
    plt.show()

sinc_interpolation_demo()
```

### 6.2 Individual Sinc Contributions

```python
def visualize_sinc_contributions():
    """Show how individual sinc functions combine for reconstruction."""
    fs = 5  # Low rate for visualization
    Ts = 1.0 / fs
    n_samples = np.arange(0, 1, Ts)
    x_samples = np.array([0.0, 0.8, 0.3, -0.5, -0.9])

    t = np.linspace(-0.2, 1.2, 1000)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Individual sinc contributions
    total = np.zeros_like(t)
    colors = plt.cm.tab10(np.linspace(0, 1, len(n_samples)))

    for i, (tn, xn) in enumerate(zip(n_samples, x_samples)):
        sinc_contrib = xn * np.sinc((t - tn) / Ts)
        total += sinc_contrib
        axes[0].plot(t, sinc_contrib, '--', color=colors[i], alpha=0.6,
                     label=f'x[{i}]={xn:.1f} * sinc')
        axes[0].plot(tn, xn, 'o', color=colors[i], markersize=8)

    axes[0].set_title('Individual Sinc Contributions')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Total reconstruction
    axes[1].plot(t, total, 'r-', linewidth=2, label='Reconstructed signal')
    axes[1].stem(n_samples, x_samples, linefmt='g-', markerfmt='go',
                 basefmt='k-', label='Samples')
    axes[1].set_title('Sum of All Sinc Functions')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sinc_contributions.png', dpi=150)
    plt.show()

visualize_sinc_contributions()
```

---

## 7. Zero-Order Hold and First-Order Hold

### 7.1 Zero-Order Hold (ZOH)

The simplest practical reconstruction: hold each sample value constant until the next sample.

$$x_{\text{ZOH}}(t) = x[n], \quad nT_s \leq t < (n+1)T_s$$

Equivalently, convolution with a rectangular pulse:

$$x_{\text{ZOH}}(t) = \sum_{n} x[n] \, p(t - nT_s)$$

where $p(t) = \begin{cases} 1 & 0 \leq t < T_s \\ 0 & \text{otherwise} \end{cases}$

The frequency response of the ZOH is:

$$H_{\text{ZOH}}(j\Omega) = T_s \, e^{-j\Omega T_s/2} \, \operatorname{sinc}\!\left(\frac{\Omega T_s}{2\pi}\right)$$

This introduces:
- **Amplitude distortion**: sinc envelope attenuates higher frequencies
- **Phase distortion**: half-sample delay ($T_s/2$)

### 7.2 First-Order Hold (FOH)

Linear interpolation between consecutive samples:

$$x_{\text{FOH}}(t) = x[n] + \frac{x[n+1] - x[n]}{T_s}(t - nT_s), \quad nT_s \leq t < (n+1)T_s$$

Equivalently, convolution with a triangular pulse:

$$\Lambda(t) = \begin{cases} 1 - |t|/T_s & |t| \leq T_s \\ 0 & \text{otherwise} \end{cases}$$

The FOH provides better high-frequency approximation than ZOH but introduces one full sample delay.

### 7.3 Comparison of Reconstruction Methods

```python
def compare_reconstruction_methods():
    """Compare ZOH, FOH, and sinc reconstruction."""
    # Original signal
    f_sig = 3  # Hz
    t_true = np.linspace(0, 1, 10000)
    x_true = np.sin(2 * np.pi * f_sig * t_true) + \
             0.3 * np.sin(2 * np.pi * 7 * t_true)

    # Sample at fs = 20 Hz
    fs = 20
    Ts = 1.0 / fs
    t_samples = np.arange(0, 1, Ts)
    x_samples = np.sin(2 * np.pi * f_sig * t_samples) + \
                0.3 * np.sin(2 * np.pi * 7 * t_samples)

    t_recon = np.linspace(0, 1, 10000)

    # 1. Zero-Order Hold
    x_zoh = np.zeros_like(t_recon)
    for i in range(len(t_samples)):
        if i < len(t_samples) - 1:
            mask = (t_recon >= t_samples[i]) & (t_recon < t_samples[i + 1])
        else:
            mask = t_recon >= t_samples[i]
        x_zoh[mask] = x_samples[i]

    # 2. First-Order Hold (linear interpolation)
    x_foh = np.interp(t_recon, t_samples, x_samples)

    # 3. Sinc interpolation
    x_sinc = np.zeros_like(t_recon)
    for tn, xn in zip(t_samples, x_samples):
        x_sinc += xn * np.sinc((t_recon - tn) / Ts)

    # Plot comparison
    methods = [
        ('Zero-Order Hold (ZOH)', x_zoh, 'orange'),
        ('First-Order Hold (FOH)', x_foh, 'purple'),
        ('Sinc Interpolation (Ideal)', x_sinc, 'red'),
    ]

    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 12))

    for ax, (name, x_rec, color) in zip(axes, methods):
        ax.plot(t_true, x_true, 'b-', alpha=0.4, linewidth=1, label='Original')
        ax.plot(t_recon, x_rec, '-', color=color, linewidth=1.5, label=name)
        ax.stem(t_samples, x_samples, linefmt='g-', markerfmt='go',
                basefmt='k-', label='Samples')

        # Compute error
        x_true_interp = np.sin(2 * np.pi * f_sig * t_recon) + \
                        0.3 * np.sin(2 * np.pi * 7 * t_recon)
        error = np.sqrt(np.mean((x_true_interp - x_rec) ** 2))

        ax.set_title(f'{name}  |  RMSE = {error:.4f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=150)
    plt.show()

compare_reconstruction_methods()
```

---

## 8. Practical ADC/DAC Considerations

### 8.1 Analog-to-Digital Conversion (ADC)

A practical ADC involves three steps:

1. **Anti-aliasing filter**: Analog lowpass filter before sampling
2. **Sample-and-hold**: Captures the instantaneous analog value
3. **Quantization**: Maps continuous amplitude to discrete levels

Key ADC specifications:

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| Resolution | Number of bits | 8, 12, 16, 24 bits |
| Sampling rate | Samples per second | 44.1 kHz (audio), 1 MSPS (general) |
| SNR | Signal-to-noise ratio | $\approx 6.02N + 1.76$ dB for $N$ bits |
| ENOB | Effective number of bits | Accounts for all noise sources |
| INL/DNL | Integral/differential nonlinearity | Ideally < 1 LSB |

### 8.2 Quantization

Quantization maps a continuous amplitude to one of $2^N$ discrete levels for an $N$-bit ADC.

**Uniform quantizer:**

$$x_q = \Delta \cdot \left\lfloor \frac{x}{\Delta} + 0.5 \right\rfloor$$

where $\Delta = \frac{x_{\max} - x_{\min}}{2^N}$ is the quantization step size (LSB).

**Quantization error (noise):**

$$e_q = x_q - x, \quad -\frac{\Delta}{2} \leq e_q < \frac{\Delta}{2}$$

For uniformly distributed quantization noise, the **signal-to-quantization-noise ratio (SQNR)**:

$$\text{SQNR} \approx 6.02 N + 1.76 \text{ dB}$$

```python
def demonstrate_quantization():
    """Demonstrate ADC quantization effects."""
    # Original signal
    t = np.linspace(0, 0.01, 10000)
    f_sig = 440  # Hz (A4 note)
    x = np.sin(2 * np.pi * f_sig * t)

    bit_depths = [2, 4, 8, 16]

    fig, axes = plt.subplots(len(bit_depths), 1, figsize=(14, 12))

    for ax, bits in zip(axes, bit_depths):
        n_levels = 2 ** bits
        delta = 2.0 / n_levels  # Range [-1, 1]

        # Quantize
        x_q = np.round(x / delta) * delta
        x_q = np.clip(x_q, -1.0, 1.0)

        # Quantization error
        error = x_q - x

        ax.plot(t * 1000, x, 'b-', alpha=0.4, linewidth=1, label='Original')
        ax.plot(t * 1000, x_q, 'r-', linewidth=1, label=f'{bits}-bit quantized')

        sqnr = 6.02 * bits + 1.76
        actual_sqnr = 10 * np.log10(np.mean(x**2) / np.mean(error**2)) \
            if np.mean(error**2) > 0 else float('inf')

        ax.set_title(f'{bits}-bit Quantization ({n_levels} levels) | '
                     f'SQNR: theoretical={sqnr:.1f} dB, actual={actual_sqnr:.1f} dB')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quantization.png', dpi=150)
    plt.show()

demonstrate_quantization()
```

### 8.3 Digital-to-Analog Conversion (DAC)

A DAC converts digital samples back to an analog signal:

1. **D/A conversion**: Convert digital code to analog voltage/current
2. **Reconstruction filter**: Lowpass filter to smooth the staircase output

Most DACs inherently use ZOH, producing a staircase output that requires a post-DAC reconstruction filter to remove the spectral images.

### 8.4 Compensation for ZOH Distortion

Since ZOH introduces a sinc roll-off, a **sinc compensation filter** (inverse sinc) can be applied digitally before the DAC:

$$H_{\text{comp}}(e^{j\omega}) = \frac{\omega T_s / 2}{\sin(\omega T_s / 2)}$$

This pre-equalizes the digital signal so that after ZOH, the output is flat in the passband.

---

## 9. Oversampling and Its Benefits

### 9.1 What Is Oversampling?

Oversampling means sampling at a rate significantly higher than the Nyquist rate:

$$f_s = M \cdot 2B$$

where $M > 1$ is the **oversampling ratio**.

### 9.2 Benefits of Oversampling

1. **Relaxed anti-aliasing filter requirements**: Wider transition band allows simpler analog filters
2. **Increased SNR through noise shaping**: Quantization noise is spread over a wider bandwidth, and filtering can remove out-of-band noise
3. **SNR improvement**: Oversampling by factor $M$ gives $10 \log_{10}(M)$ dB improvement (before noise shaping)
4. **Easier reconstruction**: Wider spectral gap simplifies the reconstruction filter

### 9.3 Sigma-Delta ($\Sigma\Delta$) Oversampling ADC

Modern high-resolution ADCs (audio, instrumentation) use $\Sigma\Delta$ modulation:

1. Oversample at $M \times f_s$ (e.g., $M = 64$ or $128$)
2. Use a 1-bit quantizer (comparator)
3. Shape quantization noise to high frequencies (noise shaping)
4. Digitally filter and decimate to the desired rate

Effective resolution: $N_{\text{eff}} \approx N + \frac{(2L+1)}{2} \log_2(M) - \frac{1}{2} \log_2\!\left(\frac{\pi^{2L}}{2L+1}\right)$

where $L$ is the order of the noise shaping loop and $N$ is the quantizer resolution.

```python
def demonstrate_oversampling():
    """Show the benefits of oversampling."""
    # Original signal: 100 Hz sinusoid
    f_sig = 100  # Hz
    B = 200      # Signal bandwidth

    # Nyquist rate
    f_nyquist = 2 * B  # 400 Hz

    oversampling_ratios = [1, 2, 4, 8, 16]

    print("Oversampling Benefits")
    print("=" * 60)
    print(f"Signal bandwidth B = {B} Hz, Nyquist rate = {f_nyquist} Hz")
    print(f"{'M':>4s} | {'fs (Hz)':>10s} | {'SNR gain (dB)':>15s} | "
          f"{'Transition band':>18s}")
    print("-" * 60)

    for M in oversampling_ratios:
        fs = M * f_nyquist
        snr_gain = 10 * np.log10(M) if M > 1 else 0
        transition = fs / 2 - B
        print(f"{M:4d} | {fs:10d} | {snr_gain:15.2f} | "
              f"{transition:14.0f} Hz")

    # Visualize filter requirements
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # M = 1: tight filter needed
    f = np.linspace(0, 800, 1000)
    fs1 = 400
    spec = np.where(f <= B, 1, 0)
    axes[0].fill_between(f, spec, alpha=0.3, color='blue', label='Signal')
    axes[0].axvline(fs1 / 2, color='red', linestyle='--',
                    label=f'Nyquist freq = {fs1/2} Hz')
    axes[0].axvline(fs1 - B, color='orange', linestyle='--',
                    label=f'First alias starts = {fs1-B} Hz')

    # Anti-aliasing region (narrow!)
    axes[0].axvspan(B, fs1 / 2, alpha=0.2, color='green',
                    label=f'Transition band = {fs1/2 - B} Hz')
    axes[0].set_title(f'M = 1 (fs = {fs1} Hz): Narrow transition band')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # M = 4: relaxed filter
    fs4 = 4 * 400
    axes[1].fill_between(f, spec, alpha=0.3, color='blue', label='Signal')
    axes[1].axvline(fs4 / 2, color='red', linestyle='--',
                    label=f'Nyquist freq = {fs4/2} Hz')

    f_ext = np.linspace(0, 2000, 2000)
    spec_ext = np.where(f_ext <= B, 1, 0)
    axes[1].fill_between(f_ext[:800], spec_ext[:800], alpha=0.3, color='blue')
    axes[1].axvspan(B, fs4 / 2, alpha=0.2, color='green',
                    label=f'Transition band = {fs4/2 - B} Hz')
    axes[1].set_title(f'M = 4 (fs = {fs4} Hz): Wide transition band')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_xlim(0, 800)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('oversampling.png', dpi=150)
    plt.show()

demonstrate_oversampling()
```

### 9.4 Decimation After Oversampling

After oversampling and digital filtering, the sample rate can be reduced (decimated) to the desired output rate:

1. Apply a sharp digital lowpass filter (easy to implement precisely)
2. Downsample by factor $M$ (keep every $M$-th sample)

This is the basis of modern $\Sigma\Delta$ ADC architecture.

---

## 10. Multirate Signal Processing Preview

### 10.1 Decimation (Downsampling)

Reduce sample rate by factor $M$:

$$y[n] = x[nM]$$

Must apply anti-aliasing filter first to prevent aliasing.

### 10.2 Interpolation (Upsampling)

Increase sample rate by factor $L$:

$$w[n] = \begin{cases} x[n/L] & n = 0, \pm L, \pm 2L, \ldots \\ 0 & \text{otherwise} \end{cases}$$

followed by a lowpass (anti-imaging) filter.

### 10.3 Sample Rate Conversion

Rational rate change by factor $L/M$:

$$\text{Upsample by } L \;\rightarrow\; \text{Filter} \;\rightarrow\; \text{Downsample by } M$$

```python
def demonstrate_decimation_interpolation():
    """Show basic multirate operations: decimation and interpolation."""
    # Original signal
    fs = 1000
    t = np.arange(0, 0.1, 1 / fs)
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

    # Decimation by M=4
    M = 4
    # Anti-aliasing filter before decimation
    b_dec, a_dec = signal.butter(8, 1.0 / M, btype='low')
    x_filtered = signal.filtfilt(b_dec, a_dec, x)
    x_dec = x_filtered[::M]
    t_dec = t[::M]

    # Interpolation by L=4
    L = 4
    x_up = np.zeros(len(x_dec) * L)
    x_up[::L] = x_dec
    # Anti-imaging filter after upsampling
    b_int, a_int = signal.butter(8, 1.0 / L, btype='low')
    x_interp = L * signal.filtfilt(b_int, a_int, x_up)
    t_interp = np.arange(len(x_interp)) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(t * 1000, x, 'b-', linewidth=1)
    axes[0].set_title(f'Original Signal (fs = {fs} Hz, {len(x)} samples)')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(t_dec * 1000, x_dec, linefmt='r-', markerfmt='ro',
                 basefmt='k-')
    axes[1].set_title(f'Decimated by M={M} (fs = {fs//M} Hz, '
                      f'{len(x_dec)} samples)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_interp * 1000, x_interp, 'g-', linewidth=1,
                 label='Interpolated')
    axes[2].plot(t[:len(x_interp)] * 1000, x[:len(x_interp)], 'b--',
                 alpha=0.5, linewidth=1, label='Original')
    axes[2].set_title(f'Interpolated back to fs = {fs} Hz')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('multirate.png', dpi=150)
    plt.show()

demonstrate_decimation_interpolation()
```

---

## 11. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                  Sampling & Reconstruction                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Sampling:                                                       │
│    x[n] = x(nTs)                                                │
│    Spectrum: Xs(jΩ) = (1/Ts) Σ X(j(Ω - kΩs))                  │
│                                                                  │
│  Nyquist-Shannon Theorem:                                        │
│    fs > 2B  →  perfect reconstruction possible                  │
│    fs ≤ 2B  →  aliasing (irreversible distortion)               │
│                                                                  │
│  Anti-Aliasing:                                                  │
│    Analog LPF before sampling to remove f > fs/2                │
│    Practical: guard band + filter roll-off                      │
│                                                                  │
│  Reconstruction Methods:                                         │
│    Ideal: sinc interpolation (impractical)                      │
│    ZOH:   staircase, sinc distortion (simple)                   │
│    FOH:   linear interpolation (better)                         │
│    Post-filter: smooth staircase output                         │
│                                                                  │
│  Oversampling (fs >> 2B):                                        │
│    - Relaxes anti-aliasing filter requirements                  │
│    - SNR gain: 10 log10(M) dB                                   │
│    - Enables Sigma-Delta ADC architecture                       │
│                                                                  │
│  Practical ADC/DAC:                                              │
│    ADC: anti-alias → sample-hold → quantize                     │
│    DAC: D/A → ZOH → reconstruction filter                      │
│    SQNR ≈ 6.02N + 1.76 dB                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Exercises

### Exercise 1: Sampling Rate Determination

A signal contains components at 100 Hz, 250 Hz, and 500 Hz.

**(a)** What is the minimum sampling rate to avoid aliasing?

**(b)** If the signal is sampled at 800 Hz, which components will alias, and to what frequencies?

**(c)** What sampling rate would you recommend in practice, and why?

### Exercise 2: Aliasing Analysis

A 1 kHz sinusoidal signal is sampled at 1.5 kHz.

**(a)** At what frequency does the alias appear?

**(b)** Write Python code to demonstrate this aliasing visually with a plot showing both the original and aliased signals.

**(c)** Design an anti-aliasing filter (specify type, order, and cutoff) that would prevent this problem if the desired bandwidth is 600 Hz.

### Exercise 3: Sinc Interpolation Implementation

Implement a complete sinc interpolation function:

```python
def sinc_reconstruct(samples, Ts, t_recon):
    """
    Reconstruct a continuous-time signal from samples using sinc interpolation.

    Parameters:
        samples: array of sample values x[n]
        Ts: sampling period
        t_recon: array of time points for reconstruction

    Returns:
        x_recon: reconstructed signal values
    """
    # Your implementation here
    pass
```

**(a)** Test your function with a known bandlimited signal and verify perfect reconstruction.

**(b)** Measure the reconstruction error as a function of the number of sinc terms used (truncation).

**(c)** Compare computation time vs. accuracy for different truncation lengths.

### Exercise 4: Quantization Noise Analysis

**(a)** Derive the SQNR formula $\text{SQNR} = 6.02N + 1.76$ dB for a uniform quantizer with $N$ bits and a full-scale sinusoidal input.

**(b)** Write Python code to verify this formula empirically for $N = 4, 8, 12, 16$ bits.

**(c)** How does the SQNR change if the signal only uses half the quantizer range?

### Exercise 5: ZOH Frequency Response

**(a)** Plot the magnitude and phase response of the ZOH reconstruction for $f_s = 8000$ Hz.

**(b)** Calculate the attenuation at $f = 3000$ Hz due to the sinc roll-off.

**(c)** Design a digital sinc compensation filter (provide FIR coefficients for 31 taps) and show that it corrects the passband droop.

### Exercise 6: Oversampling vs. Bit Depth Trade-off

An audio application requires an effective resolution of 16 bits (SQNR $\approx$ 98 dB).

**(a)** If using a 12-bit ADC with oversampling (no noise shaping), what oversampling ratio $M$ is needed?

**(b)** With first-order noise shaping ($\Sigma\Delta$), what oversampling ratio suffices?

**(c)** Write Python code to simulate a first-order $\Sigma\Delta$ modulator and verify the noise shaping behavior by plotting the output spectrum.

### Exercise 7: Complete ADC/DAC Pipeline

Build a complete sampling-reconstruction pipeline in Python:

1. Generate a test signal (sum of sinusoids)
2. Apply an anti-aliasing filter
3. Sample at a specified rate
4. Quantize to $N$ bits
5. Reconstruct using (a) ZOH, (b) linear interpolation, (c) sinc interpolation
6. Apply a reconstruction lowpass filter
7. Compare all three methods: RMSE, SNR, visual quality

---

## 13. Further Reading

- Oppenheim, Willsky, Nawab. *Signals and Systems*, 2nd ed. Chapter 7.
- Oppenheim, Schafer. *Discrete-Time Signal Processing*, 3rd ed. Chapters 4-5.
- Proakis, Manolakis. *Digital Signal Processing*, 4th ed. Chapter 6.
- Lyons, R. G. *Understanding Digital Signal Processing*, 3rd ed. Chapters 2-3.
- Smith, S. W. *The Scientist and Engineer's Guide to Digital Signal Processing*, Chapters 3-4. (Free online: dspguide.com)

---

**Previous**: [04. Fourier Transform and Frequency Domain](04_Fourier_Transform.md) | **Next**: [06. Discrete Fourier Transform](06_Discrete_Fourier_Transform.md)
