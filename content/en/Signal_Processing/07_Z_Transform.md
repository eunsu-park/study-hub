# Z-Transform

## Overview

The Z-transform is the discrete-time counterpart of the Laplace transform. It converts difference equations into algebraic equations, enabling analysis of discrete-time LTI systems in the complex $z$-plane. The Z-transform provides powerful tools for determining system stability, frequency response, and transfer functions. This lesson covers Z-transform theory, properties, inverse methods, and applications to digital system analysis.

**Learning Objectives:**
- Define and compute bilateral and unilateral Z-transforms
- Determine the Region of Convergence (ROC) and its implications
- Apply Z-transform properties for system analysis
- Compute inverse Z-transforms using multiple methods
- Analyze LTI systems using transfer functions, poles, and zeros
- Relate the Z-transform to the DTFT and Laplace transform

**Prerequisites:** [06. Discrete Fourier Transform](06_Discrete_Fourier_Transform.md)

---

## 1. Definition of the Z-Transform

### 1.1 Bilateral Z-Transform

The bilateral (two-sided) Z-transform of a discrete-time signal $x[n]$ is:

$$\boxed{X(z) = \mathcal{Z}\{x[n]\} = \sum_{n=-\infty}^{\infty} x[n] \, z^{-n}}$$

where $z$ is a complex variable: $z = r \, e^{j\omega}$.

### 1.2 Unilateral Z-Transform

The unilateral (one-sided) Z-transform is used for causal signals and systems with initial conditions:

$$X(z) = \sum_{n=0}^{\infty} x[n] \, z^{-n}$$

This form is particularly useful for solving difference equations with non-zero initial conditions.

### 1.3 Intuition: What Is z?

The complex variable $z = r \, e^{j\omega}$ can be decomposed:
- $|z| = r$: Radial distance from the origin (controls convergence via exponential weighting)
- $\angle z = \omega$: Angle (corresponds to frequency)
- On the unit circle ($r = 1$, $z = e^{j\omega}$): Z-transform reduces to the DTFT

### 1.4 Common Z-Transform Pairs

| Signal $x[n]$ | Z-Transform $X(z)$ | ROC |
|-------------|------------------|-----|
| $\delta[n]$ | $1$ | All $z$ |
| $u[n]$ (unit step) | $\frac{z}{z-1} = \frac{1}{1-z^{-1}}$ | $|z| > 1$ |
| $a^n u[n]$ | $\frac{z}{z-a} = \frac{1}{1-az^{-1}}$ | $|z| > |a|$ |
| $-a^n u[-n-1]$ | $\frac{z}{z-a} = \frac{1}{1-az^{-1}}$ | $|z| < |a|$ |
| $n a^n u[n]$ | $\frac{az}{(z-a)^2} = \frac{az^{-1}}{(1-az^{-1})^2}$ | $|z| > |a|$ |
| $\cos(\omega_0 n) u[n]$ | $\frac{z(z-\cos\omega_0)}{z^2 - 2z\cos\omega_0 + 1}$ | $|z| > 1$ |
| $r^n \cos(\omega_0 n) u[n]$ | $\frac{z(z - r\cos\omega_0)}{z^2 - 2rz\cos\omega_0 + r^2}$ | $|z| > r$ |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def compute_z_transform_examples():
    """Compute and verify common Z-transform pairs."""
    # Example 1: x[n] = (0.8)^n * u[n]
    a = 0.8
    N = 50
    n = np.arange(N)
    x = a ** n  # Causal exponential

    # Z-transform: X(z) = 1 / (1 - 0.8 * z^{-1}), |z| > 0.8
    # Evaluate on unit circle (should give DTFT)
    omega = np.linspace(-np.pi, np.pi, 1024)
    z_unit = np.exp(1j * omega)

    # X(z) on unit circle
    X_formula = 1.0 / (1.0 - a * z_unit ** (-1))

    # DTFT (direct computation from samples)
    X_dtft = np.zeros(len(omega), dtype=complex)
    for k in range(N):
        X_dtft += x[k] * np.exp(-1j * omega * k)

    # Compare
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(omega / np.pi, np.abs(X_formula), 'b-', linewidth=2,
                 label='Z-transform formula')
    axes[0].plot(omega / np.pi, np.abs(X_dtft), 'r--', linewidth=1,
                 label=f'DTFT (N={N} terms)')
    axes[0].set_title(r'$x[n] = 0.8^n u[n]$: Magnitude on Unit Circle')
    axes[0].set_xlabel(r'$\omega / \pi$')
    axes[0].set_ylabel(r'$|X(e^{j\omega})|$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(omega / np.pi, np.angle(X_formula), 'b-', linewidth=2,
                 label='Z-transform formula')
    axes[1].plot(omega / np.pi, np.angle(X_dtft), 'r--', linewidth=1,
                 label=f'DTFT (N={N} terms)')
    axes[1].set_title('Phase on Unit Circle')
    axes[1].set_xlabel(r'$\omega / \pi$')
    axes[1].set_ylabel('Phase (radians)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ztransform_unit_circle.png', dpi=150)
    plt.show()

compute_z_transform_examples()
```

---

## 2. Region of Convergence (ROC)

### 2.1 Definition

The ROC is the set of all $z$ values for which the Z-transform sum converges:

$$\text{ROC} = \left\{ z \in \mathbb{C} : \sum_{n=-\infty}^{\infty} |x[n]| \, |z|^{-n} < \infty \right\}$$

The ROC is always an annular region in the z-plane (a ring between two concentric circles centered at the origin):

$$R^{-} < |z| < R^{+}$$

### 2.2 ROC Properties

1. **ROC does not contain any poles** of $X(z)$
2. **Finite-duration signals**: ROC is the entire z-plane (possibly excluding $z = 0$ and/or $z = \infty$)
3. **Right-sided signals** ($x[n] = 0$ for $n < N_1$): ROC is the exterior of a circle: $|z| > R^{-}$
4. **Left-sided signals** ($x[n] = 0$ for $n > N_2$): ROC is the interior of a circle: $|z| < R^{+}$
5. **Two-sided signals**: ROC is an annular ring
6. **DTFT exists** if and only if the ROC includes the unit circle $|z| = 1$
7. **Causal and stable system**: ROC includes and extends outside the unit circle; all poles are inside the unit circle

### 2.3 ROC and Signal Type: The Same X(z), Different Signals

The Z-transform $X(z) = \frac{1}{1 - az^{-1}}$ can correspond to **two different signals** depending on the ROC:

- ROC: $|z| > |a|$ $\implies$ $x[n] = a^n u[n]$ (causal, right-sided)
- ROC: $|z| < |a|$ $\implies$ $x[n] = -a^n u[-n-1]$ (anti-causal, left-sided)

> This is why the ROC is essential: $X(z)$ alone (without the ROC) does not uniquely determine $x[n]$.

```python
def visualize_roc():
    """Visualize ROC for different signal types."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    theta = np.linspace(0, 2 * np.pi, 200)

    # Case 1: Causal signal x[n] = 0.7^n u[n], ROC: |z| > 0.7
    ax = axes[0]
    # Unit circle
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
    # Pole at z = 0.7
    ax.plot(0.7, 0, 'rx', markersize=12, markeredgewidth=2, label='Pole')
    # ROC: |z| > 0.7 (shade exterior)
    r_roc = 0.7
    circle = plt.Circle((0, 0), r_roc, fill=True, color='lightblue',
                         alpha=0.5, label=f'ROC: |z| > {r_roc}')
    ax.add_patch(circle)
    ax.fill_between(np.cos(theta) * 2, np.sin(theta) * 2,
                    alpha=0.2, color='green')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title(r'Causal: $0.7^n u[n]$' + '\nROC: |z| > 0.7')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Case 2: Anti-causal signal, ROC: |z| < 0.7
    ax = axes[1]
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
    ax.plot(0.7, 0, 'rx', markersize=12, markeredgewidth=2, label='Pole')
    circle = plt.Circle((0, 0), r_roc, fill=True, color='lightgreen',
                         alpha=0.5, label=f'ROC: |z| < {r_roc}')
    ax.add_patch(circle)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title(r'Anti-causal: $-0.7^n u[-n-1]$' + '\nROC: |z| < 0.7')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Case 3: Two-sided signal, ROC: annular ring
    ax = axes[2]
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
    ax.plot(0.5, 0, 'rx', markersize=12, markeredgewidth=2, label='Poles')
    ax.plot(1.5, 0, 'rx', markersize=12, markeredgewidth=2)
    # ROC: 0.5 < |z| < 1.5
    for r in [0.5, 1.5]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'b--', linewidth=1)
    # Shade annular region
    theta_fill = np.linspace(0, 2 * np.pi, 100)
    r_inner, r_outer = 0.5, 1.5
    ax.fill_between(
        np.concatenate([r_inner * np.cos(theta_fill),
                        r_outer * np.cos(theta_fill[::-1])]),
        np.concatenate([r_inner * np.sin(theta_fill),
                        r_outer * np.sin(theta_fill[::-1])]),
        alpha=0.3, color='yellow', label='ROC: 0.5 < |z| < 1.5'
    )
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Two-sided signal\nROC: 0.5 < |z| < 1.5')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_visualization.png', dpi=150)
    plt.show()

visualize_roc()
```

---

## 3. Properties of the Z-Transform

### 3.1 Linearity

$$a \, x_1[n] + b \, x_2[n] \quad \xleftrightarrow{\mathcal{Z}} \quad a \, X_1(z) + b \, X_2(z)$$

ROC: at least $\text{ROC}_1 \cap \text{ROC}_2$ (may be larger if pole-zero cancellations occur).

### 3.2 Time Shifting

$$x[n - n_0] \quad \xleftrightarrow{\mathcal{Z}} \quad z^{-n_0} X(z)$$

ROC: same as $X(z)$ (possibly with addition/removal of $z = 0$ or $z = \infty$).

> The delay operator $z^{-1}$ is fundamental in digital systems. A delay of one sample is represented by multiplication by $z^{-1}$.

### 3.3 Scaling in the z-Domain

$$a^n x[n] \quad \xleftrightarrow{\mathcal{Z}} \quad X(z/a)$$

ROC: $|a| \cdot R^{-} < |z| < |a| \cdot R^{+}$ (ROC scaled by $|a|$).

### 3.4 Time Reversal

$$x[-n] \quad \xleftrightarrow{\mathcal{Z}} \quad X(z^{-1})$$

ROC: $1/R^{+} < |z| < 1/R^{-}$ (ROC inverted).

### 3.5 Differentiation in z-Domain

$$n \, x[n] \quad \xleftrightarrow{\mathcal{Z}} \quad -z \frac{dX(z)}{dz}$$

Useful for deriving transforms involving $n \cdot a^n$.

### 3.6 Convolution

$$x_1[n] * x_2[n] \quad \xleftrightarrow{\mathcal{Z}} \quad X_1(z) \cdot X_2(z)$$

ROC: at least $\text{ROC}_1 \cap \text{ROC}_2$.

This is the most important property for system analysis: convolution in time becomes multiplication in the z-domain.

### 3.7 Initial Value Theorem (Causal Signals)

$$x[0] = \lim_{z \to \infty} X(z)$$

### 3.8 Final Value Theorem

If $(1 - z^{-1})X(z)$ has all poles inside the unit circle:

$$\lim_{n \to \infty} x[n] = \lim_{z \to 1} (1 - z^{-1}) X(z)$$

### 3.9 Properties Summary Table

| Property | Time Domain | Z-Domain | ROC |
|----------|------------|----------|-----|
| Linearity | $ax_1 + bx_2$ | $aX_1 + bX_2$ | $\supseteq R_1 \cap R_2$ |
| Time shift | $x[n-n_0]$ | $z^{-n_0}X(z)$ | $R$ (maybe $\pm$ 0, $\infty$) |
| Scaling | $a^n x[n]$ | $X(z/a)$ | $|a| \cdot R$ |
| Reversal | $x[-n]$ | $X(1/z)$ | $1/R$ |
| Differentiation | $nx[n]$ | $-z\frac{dX}{dz}$ | $R$ |
| Convolution | $x_1 * x_2$ | $X_1 X_2$ | $\supseteq R_1 \cap R_2$ |
| Accumulation | $\sum_{k=-\infty}^{n} x[k]$ | $\frac{X(z)}{1-z^{-1}}$ | $R \cap \{|z|>1\}$ |

```python
def demonstrate_z_properties():
    """Numerically verify Z-transform properties."""
    # Test signal: x[n] = 0.8^n * u[n], truncated to 100 samples
    N = 100
    a = 0.8
    n = np.arange(N)
    x = a ** n

    # Evaluate Z-transforms on the unit circle
    omega = np.linspace(-np.pi, np.pi, 512)
    z = np.exp(1j * omega)

    def zt_on_circle(signal, omega_vals):
        """Compute Z-transform on unit circle (= DTFT)."""
        z_vals = np.exp(1j * omega_vals)
        result = np.zeros(len(omega_vals), dtype=complex)
        for k in range(len(signal)):
            result += signal[k] * z_vals ** (-k)
        return result

    # Property 1: Time shift
    m = 5
    x_shifted = np.zeros(N + m)
    x_shifted[m:m + N] = x

    X_orig = zt_on_circle(x, omega)
    X_shifted_direct = zt_on_circle(x_shifted, omega)
    X_shifted_property = np.exp(-1j * omega * m) * X_orig

    # Property 2: Convolution
    h = 0.5 ** n  # Another causal exponential
    y_conv = np.convolve(x[:50], h[:50])  # Linear convolution

    X_x = zt_on_circle(x[:50], omega)
    H_z = zt_on_circle(h[:50], omega)
    Y_product = X_x * H_z
    Y_conv_direct = zt_on_circle(y_conv, omega)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time shift verification
    axes[0, 0].plot(omega / np.pi, np.abs(X_shifted_direct), 'b-',
                    linewidth=2, label='Direct')
    axes[0, 0].plot(omega / np.pi, np.abs(X_shifted_property), 'r--',
                    linewidth=1, label='z^{-m} X(z)')
    axes[0, 0].set_title(f'Time Shift Property (m={m}): Magnitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(omega / np.pi,
                    np.abs(X_shifted_direct - X_shifted_property), 'k-')
    axes[0, 1].set_title(f'Time Shift Error')
    axes[0, 1].set_ylabel('|Error|')
    axes[0, 1].grid(True, alpha=0.3)

    # Convolution verification
    axes[1, 0].plot(omega / np.pi, np.abs(Y_conv_direct), 'b-',
                    linewidth=2, label='Z{x*h}')
    axes[1, 0].plot(omega / np.pi, np.abs(Y_product), 'r--',
                    linewidth=1, label='X(z)H(z)')
    axes[1, 0].set_title('Convolution Property: Magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(omega / np.pi, np.abs(Y_conv_direct - Y_product), 'k-')
    axes[1, 1].set_title('Convolution Property Error')
    axes[1, 1].set_ylabel('|Error|')
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel(r'$\omega / \pi$')

    plt.tight_layout()
    plt.savefig('z_properties.png', dpi=150)
    plt.show()

demonstrate_z_properties()
```

---

## 4. Inverse Z-Transform

### 4.1 Formal Definition

$$x[n] = \frac{1}{2\pi j} \oint_C X(z) \, z^{n-1} \, dz$$

where $C$ is a counterclockwise contour within the ROC encircling the origin.

In practice, three methods are commonly used:

### 4.2 Method 1: Partial Fraction Expansion

For rational $X(z) = B(z)/A(z)$, decompose into simple fractions:

$$X(z) = \sum_i \frac{A_i}{1 - p_i z^{-1}} + \cdots$$

Each term has a known inverse Z-transform, and the ROC determines whether each is causal or anti-causal.

**Example:**

$$X(z) = \frac{1}{(1 - 0.5z^{-1})(1 - 0.8z^{-1})}, \quad |z| > 0.8$$

Partial fractions:

$$X(z) = \frac{A}{1 - 0.5z^{-1}} + \frac{B}{1 - 0.8z^{-1}}$$

Solving: $A = \frac{-5}{3}$, $B = \frac{8}{3}$

Since ROC is $|z| > 0.8$ (both poles inside ROC), both terms are causal:

$$x[n] = \left(-\frac{5}{3}(0.5)^n + \frac{8}{3}(0.8)^n\right) u[n]$$

```python
def partial_fraction_inverse_z():
    """Inverse Z-transform via partial fraction expansion."""
    # X(z) = 1 / ((1 - 0.5 z^{-1})(1 - 0.8 z^{-1}))
    # Numerator: [1] (in z^{-1} form)
    # Denominator: (1 - 0.5 z^{-1})(1 - 0.8 z^{-1})
    #            = 1 - 1.3 z^{-1} + 0.4 z^{-2}

    # Using scipy.signal for partial fractions
    # Express as H(z) = B(z)/A(z) in descending powers of z
    # B(z) = 1
    # A(z) = 1 - 1.3 z^{-1} + 0.4 z^{-2}

    b = [1.0]                   # Numerator coefficients
    a = [1.0, -1.3, 0.4]       # Denominator coefficients

    # Partial fraction expansion
    # scipy uses z (not z^{-1}), so we need to be careful
    # Convert to z-form: multiply num/den by z^2
    b_z = [0, 0, 1]            # z^0 (need to match length)
    a_z = [1, -1.3, 0.4]       # z^2 - 1.3z + 0.4

    residues, poles, remainder = signal.residuez(b, a)

    print("Partial Fraction Expansion")
    print("=" * 50)
    print(f"X(z) = 1 / ((1 - 0.5z^-1)(1 - 0.8z^-1))")
    print(f"\nPoles: {poles}")
    print(f"Residues: {residues}")
    print(f"Remainder: {remainder}")

    # Reconstruct x[n]
    N = 30
    n = np.arange(N)

    # From partial fractions (causal, ROC: |z| > 0.8)
    x_pf = np.zeros(N)
    for r, p in zip(residues, poles):
        x_pf += np.real(r * p ** n)

    # Direct computation via scipy.signal (impulse response)
    _, x_impulse = signal.dimpulse(signal.dlti(b, a, dt=1), n=N)
    x_impulse = np.squeeze(x_impulse)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stem(n, x_pf, linefmt='b-', markerfmt='bo', basefmt='k-',
            label='Partial fractions')
    ax.plot(n, x_impulse, 'rx', markersize=8, label='scipy dimpulse')
    ax.set_title('Inverse Z-Transform via Partial Fractions')
    ax.set_xlabel('n')
    ax.set_ylabel('x[n]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('inverse_z_partial.png', dpi=150)
    plt.show()

partial_fraction_inverse_z()
```

### 4.3 Method 2: Long Division (Power Series Expansion)

Divide $B(z^{-1})$ by $A(z^{-1})$ to get coefficients of $z^{-n}$, which are the values $x[n]$.

**Example:**

$$X(z) = \frac{1}{1 - 1.5z^{-1} + 0.5z^{-2}}$$

Long division in $z^{-1}$:

$$\frac{1}{1 - 1.5z^{-1} + 0.5z^{-2}} = 1 + 1.5z^{-1} + 1.75z^{-2} + 1.875z^{-3} + \cdots$$

So $x[0] = 1$, $x[1] = 1.5$, $x[2] = 1.75$, $x[3] = 1.875$, ...

```python
def long_division_inverse_z():
    """Inverse Z-transform via long division."""
    # X(z) = B(z^{-1}) / A(z^{-1})
    b = np.array([1.0])
    a = np.array([1.0, -1.5, 0.5])

    N = 20
    x = np.zeros(N)

    # Long division algorithm
    remainder = np.zeros(len(b) + N)
    remainder[:len(b)] = b

    for n in range(N):
        x[n] = remainder[0] / a[0]
        for k in range(len(a)):
            if k < len(remainder):
                remainder[k] -= x[n] * a[k]
        remainder = np.roll(remainder, -1)
        remainder[-1] = 0

    # Verify with scipy
    _, x_scipy = signal.dimpulse(signal.dlti(b, a, dt=1), n=N)
    x_scipy = np.squeeze(x_scipy)

    print("Long Division Inverse Z-Transform")
    print("=" * 40)
    print(f"X(z) = 1 / (1 - 1.5z^(-1) + 0.5z^(-2))")
    print(f"\n{'n':>4s} | {'x[n] (long div)':>16s} | {'x[n] (scipy)':>14s}")
    print("-" * 40)
    for i in range(min(10, N)):
        print(f"{i:4d} | {x[i]:16.6f} | {x_scipy[i]:14.6f}")

long_division_inverse_z()
```

### 4.4 Method 3: Contour Integration (Residue Theorem)

$$x[n] = \sum_{\text{poles } p_k \text{ inside } C} \text{Res}\left[X(z) z^{n-1}, p_k\right]$$

For a simple pole at $z = p_k$:

$$\text{Res}\left[X(z)z^{n-1}, p_k\right] = \lim_{z \to p_k} (z - p_k) X(z) z^{n-1}$$

---

## 5. Transfer Function H(z)

### 5.1 Definition

For an LTI system described by the difference equation:

$$\sum_{k=0}^{N} a_k \, y[n-k] = \sum_{k=0}^{M} b_k \, x[n-k]$$

The transfer function is:

$$\boxed{H(z) = \frac{Y(z)}{X(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{\sum_{k=0}^{N} a_k z^{-k}} = \frac{B(z)}{A(z)}}$$

The output in the z-domain is simply:

$$Y(z) = H(z) \cdot X(z)$$

### 5.2 Impulse Response and Transfer Function

The impulse response $h[n]$ is the inverse Z-transform of $H(z)$:

$$h[n] = \mathcal{Z}^{-1}\{H(z)\}$$

Since $Y(z) = H(z) X(z)$ and multiplication in z-domain corresponds to convolution in time:

$$y[n] = h[n] * x[n] = \sum_{k=-\infty}^{\infty} h[k] \, x[n-k]$$

### 5.3 Example: First-Order System

$$y[n] = 0.9 \, y[n-1] + x[n]$$

Transfer function:

$$H(z) = \frac{1}{1 - 0.9z^{-1}} = \frac{z}{z - 0.9}$$

- One pole at $z = 0.9$
- One zero at $z = 0$ (trivial, from $z^{-1}$ formulation)

```python
def transfer_function_example():
    """Analyze a first-order digital system."""
    # y[n] = 0.9 * y[n-1] + x[n]
    # H(z) = 1 / (1 - 0.9 z^{-1})
    b = [1.0]
    a = [1.0, -0.9]

    # Create discrete-time system
    sys = signal.dlti(b, a, dt=1)

    # Impulse response
    t_imp, h = signal.dimpulse(sys, n=40)
    h = np.squeeze(h)

    # Step response
    t_step, s = signal.dstep(sys, n=40)
    s = np.squeeze(s)

    # Frequency response
    w, H = signal.freqz(b, a, worN=1024)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Impulse response
    axes[0, 0].stem(np.arange(len(h)), h, linefmt='b-', markerfmt='bo',
                    basefmt='k-')
    axes[0, 0].set_title('Impulse Response h[n]')
    axes[0, 0].set_xlabel('n')
    axes[0, 0].set_ylabel('h[n]')
    axes[0, 0].grid(True, alpha=0.3)

    # Step response
    axes[0, 1].stem(np.arange(len(s)), s, linefmt='r-', markerfmt='ro',
                    basefmt='k-')
    axes[0, 1].set_title('Step Response')
    axes[0, 1].set_xlabel('n')
    axes[0, 1].set_ylabel('y[n]')
    axes[0, 1].grid(True, alpha=0.3)

    # Magnitude response
    axes[1, 0].plot(w / np.pi, 20 * np.log10(np.abs(H)), 'b-', linewidth=2)
    axes[1, 0].set_title('Magnitude Response |H(e^jw)|')
    axes[1, 0].set_xlabel(r'$\omega / \pi$')
    axes[1, 0].set_ylabel('Magnitude (dB)')
    axes[1, 0].grid(True, alpha=0.3)

    # Phase response
    axes[1, 1].plot(w / np.pi, np.unwrap(np.angle(H)), 'r-', linewidth=2)
    axes[1, 1].set_title('Phase Response')
    axes[1, 1].set_xlabel(r'$\omega / \pi$')
    axes[1, 1].set_ylabel('Phase (radians)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(r'System: $y[n] = 0.9\,y[n-1] + x[n]$', fontsize=14)
    plt.tight_layout()
    plt.savefig('transfer_function.png', dpi=150)
    plt.show()

transfer_function_example()
```

---

## 6. Poles and Zeros in the z-Plane

### 6.1 Definition

For a rational transfer function:

$$H(z) = \frac{b_0 + b_1 z^{-1} + \cdots + b_M z^{-M}}{a_0 + a_1 z^{-1} + \cdots + a_N z^{-N}} = G \cdot \frac{\prod_{k=1}^{M}(z - z_k)}{\prod_{k=1}^{N}(z - p_k)}$$

- **Zeros** ($z_k$): Values of $z$ where $H(z) = 0$ (numerator roots)
- **Poles** ($p_k$): Values of $z$ where $H(z) \to \infty$ (denominator roots)
- $G$: Gain factor

### 6.2 Pole-Zero Plot

```python
def pole_zero_analysis():
    """Analyze a system using pole-zero plots."""
    # Second-order system (resonator)
    # H(z) = 1 / (1 - 2r cos(w0) z^{-1} + r^2 z^{-2})
    r = 0.9       # Pole radius (< 1 for stability)
    w0 = np.pi / 4  # Resonant frequency (pi/4 = fs/8)

    b = [1.0]
    a = [1.0, -2 * r * np.cos(w0), r ** 2]

    # Find poles and zeros
    zeros = np.roots(b)
    poles = np.roots(a)

    print("Pole-Zero Analysis")
    print("=" * 50)
    print(f"Zeros: {zeros}")
    print(f"Poles: {poles}")
    print(f"Pole magnitudes: {np.abs(poles)}")
    print(f"Pole angles: {np.angle(poles) / np.pi} * pi")
    print(f"Stable: {all(np.abs(poles) < 1)}")

    # Pole-zero plot and frequency response
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pole-zero plot
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1,
            label='Unit circle')

    # Plot zeros
    if len(zeros) > 0:
        ax.plot(np.real(zeros), np.imag(zeros), 'bo', markersize=10,
                label=f'Zeros ({len(zeros)})')

    # Plot poles
    ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=12,
            markeredgewidth=2, label=f'Poles ({len(poles)})')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Pole-Zero Plot')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Frequency response
    ax = axes[1]
    w, H = signal.freqz(b, a, worN=1024)
    ax.plot(w / np.pi, 20 * np.log10(np.abs(H)), 'b-', linewidth=2)
    ax.axvline(w0 / np.pi, color='red', linestyle='--', alpha=0.5,
               label=f'Resonant freq = {w0/np.pi:.2f}pi')
    ax.set_title('Magnitude Response')
    ax.set_xlabel(r'$\omega / \pi$')
    ax.set_ylabel('Magnitude (dB)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pole_zero.png', dpi=150)
    plt.show()

pole_zero_analysis()
```

### 6.3 Effect of Pole and Zero Locations

| Location | Effect |
|----------|--------|
| **Pole near unit circle** | Sharp peak in frequency response at the pole angle |
| **Zero on unit circle** | Null (zero gain) at the zero angle |
| **Pole inside unit circle** | Stable, decaying impulse response |
| **Pole outside unit circle** | Unstable, growing impulse response |
| **Pole on unit circle** | Marginally stable, sustained oscillation |
| **Pole at origin** | Pure delay (FIR behavior) |
| **Complex conjugate poles** | Resonance (oscillatory decay) |

### 6.4 Interactive Pole-Zero Exploration

```python
def pole_zero_effects():
    """Show how pole/zero locations affect frequency response."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    configurations = [
        {
            'title': 'Lowpass (pole near z=1)',
            'b': [1.0], 'a': [1.0, -0.9]
        },
        {
            'title': 'Highpass (pole near z=-1)',
            'b': [1.0, -1.0], 'a': [1.0, -0.9]
        },
        {
            'title': 'Bandpass (complex conjugate poles)',
            'b': [1.0, 0, -1.0],
            'a': [1.0, -2 * 0.9 * np.cos(np.pi / 4), 0.81]
        },
        {
            'title': 'Notch (zeros on unit circle)',
            'b': [1.0, -2 * np.cos(np.pi / 4), 1.0],
            'a': [1.0, -2 * 0.9 * np.cos(np.pi / 4), 0.81]
        },
        {
            'title': 'All-pass (poles/zeros reciprocal)',
            'b': [0.5, 1.0],
            'a': [1.0, 0.5]
        },
        {
            'title': 'Comb filter (pole at z^N=r^N)',
            'b': [1.0],
            'a': np.concatenate([[1.0], np.zeros(7), [-0.8]])
        },
    ]

    for ax_row, config in zip(axes.flat, configurations):
        b, a = np.array(config['b']), np.array(config['a'])

        w, H = signal.freqz(b, a, worN=1024)
        ax_row.plot(w / np.pi, 20 * np.log10(np.abs(H) + 1e-12),
                    'b-', linewidth=2)
        ax_row.set_title(config['title'])
        ax_row.set_xlabel(r'$\omega / \pi$')
        ax_row.set_ylabel('Magnitude (dB)')
        ax_row.set_ylim(-40, 30)
        ax_row.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pole_zero_effects.png', dpi=150)
    plt.show()

pole_zero_effects()
```

---

## 7. Stability Analysis

### 7.1 BIBO Stability

A causal LTI system is **Bounded-Input Bounded-Output (BIBO) stable** if and only if:

$$\sum_{n=0}^{\infty} |h[n]| < \infty$$

In terms of the Z-transform, a causal system is BIBO stable if and only if:

$$\boxed{\text{All poles of } H(z) \text{ lie strictly inside the unit circle: } |p_k| < 1 \, \forall k}$$

### 7.2 Stability Conditions

| Pole Location | Stability | Impulse Response |
|--------------|-----------|-----------------|
| All $|p_k| < 1$ | Stable | Decays to zero |
| Some $|p_k| = 1$ (simple) | Marginally stable | Bounded, non-decaying |
| Any $|p_k| > 1$ | Unstable | Grows without bound |
| $|p_k| = 1$ (repeated) | Unstable | Grows like $n^{m-1}$ |

### 7.3 Stability Checking

```python
def stability_analysis():
    """Analyze stability of several systems."""
    systems = [
        {
            'name': 'Stable: y[n] = 0.5*y[n-1] + x[n]',
            'b': [1.0], 'a': [1.0, -0.5]
        },
        {
            'name': 'Marginally stable: y[n] = y[n-1] + x[n]',
            'b': [1.0], 'a': [1.0, -1.0]
        },
        {
            'name': 'Unstable: y[n] = 1.1*y[n-1] + x[n]',
            'b': [1.0], 'a': [1.0, -1.1]
        },
        {
            'name': 'Stable oscillator: r=0.9, w0=pi/4',
            'b': [1.0],
            'a': [1.0, -2 * 0.9 * np.cos(np.pi / 4), 0.81]
        },
        {
            'name': 'Unstable oscillator: r=1.05, w0=pi/4',
            'b': [1.0],
            'a': [1.0, -2 * 1.05 * np.cos(np.pi / 4), 1.05**2]
        },
    ]

    fig, axes = plt.subplots(len(systems), 2, figsize=(14, 3 * len(systems)))

    for i, sys_info in enumerate(systems):
        b, a = np.array(sys_info['b']), np.array(sys_info['a'])
        poles = np.roots(a)
        max_pole_mag = np.max(np.abs(poles))

        stability = "STABLE" if max_pole_mag < 1 else \
                    "MARGINALLY STABLE" if np.isclose(max_pole_mag, 1) else \
                    "UNSTABLE"

        # Pole-zero plot
        ax_pz = axes[i, 0]
        theta = np.linspace(0, 2 * np.pi, 200)
        ax_pz.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
        ax_pz.plot(np.real(poles), np.imag(poles), 'rx', markersize=10,
                   markeredgewidth=2)
        ax_pz.set_xlim(-1.5, 1.5)
        ax_pz.set_ylim(-1.5, 1.5)
        ax_pz.set_aspect('equal')
        ax_pz.set_title(f'{sys_info["name"]}\n[{stability}] max|pole|={max_pole_mag:.3f}')
        ax_pz.axhline(0, color='gray', linewidth=0.5)
        ax_pz.axvline(0, color='gray', linewidth=0.5)
        ax_pz.grid(True, alpha=0.3)

        # Impulse response
        ax_ir = axes[i, 1]
        N = 40
        h = np.zeros(N)
        h[0] = b[0] / a[0]
        for n in range(1, N):
            h[n] = (b[n] if n < len(b) else 0)
            for k in range(1, min(n + 1, len(a))):
                h[n] -= a[k] * h[n - k]
            h[n] /= a[0]

        color = 'green' if stability == "STABLE" else \
                'orange' if stability == "MARGINALLY STABLE" else 'red'
        ax_ir.stem(np.arange(N), h, linefmt=f'{color[0]}-',
                   markerfmt=f'{color[0]}o', basefmt='k-')
        ax_ir.set_title(f'Impulse Response')
        ax_ir.set_xlabel('n')
        ax_ir.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stability.png', dpi=150)
    plt.show()

stability_analysis()
```

### 7.4 Jury Stability Test

For a polynomial $A(z) = a_0 z^N + a_1 z^{N-1} + \cdots + a_N$, the Jury stability test provides necessary and sufficient conditions for all roots to be inside the unit circle, without explicitly computing the roots.

**Necessary conditions (quick checks):**
1. $A(1) > 0$
2. $(-1)^N A(-1) > 0$
3. $|a_N| < |a_0|$

```python
def jury_test(a):
    """Perform Jury stability test on polynomial coefficients."""
    a = np.array(a, dtype=float)
    N = len(a) - 1  # Polynomial order

    print("Jury Stability Test")
    print("=" * 50)
    print(f"Polynomial order: {N}")
    print(f"Coefficients: {a}")

    # Necessary conditions
    A_1 = np.polyval(a, 1)
    A_neg1 = np.polyval(a, -1)
    cond1 = A_1 > 0
    cond2 = ((-1) ** N * A_neg1) > 0
    cond3 = abs(a[-1]) < abs(a[0])

    print(f"\nNecessary conditions:")
    print(f"  A(1) = {A_1:.4f} > 0 ? {cond1}")
    print(f"  (-1)^N * A(-1) = {(-1)**N * A_neg1:.4f} > 0 ? {cond2}")
    print(f"  |a_N| = {abs(a[-1]):.4f} < |a_0| = {abs(a[0]):.4f} ? {cond3}")

    if not (cond1 and cond2 and cond3):
        print("\n  => UNSTABLE (necessary condition violated)")
        return False

    # Verify with actual roots
    roots = np.roots(a)
    max_mag = np.max(np.abs(roots))
    stable = max_mag < 1
    print(f"\nVerification: max|root| = {max_mag:.6f}")
    print(f"System is {'STABLE' if stable else 'UNSTABLE'}")
    return stable

# Test examples
print("System 1:")
jury_test([1, -1.3, 0.4])  # Stable (poles at 0.5, 0.8)
print("\nSystem 2:")
jury_test([1, -2.0, 1.1])  # Unstable
```

---

## 8. Relationship to DTFT and Laplace Transform

### 8.1 Z-Transform and DTFT

The DTFT is the Z-transform evaluated on the unit circle:

$$\boxed{X(e^{j\omega}) = X(z)\big|_{z=e^{j\omega}}}$$

This relationship holds if the ROC of $X(z)$ includes the unit circle.

**Consequence:** The frequency response of a discrete-time system is:

$$H(e^{j\omega}) = H(z)\big|_{z=e^{j\omega}}$$

### 8.2 Z-Transform and Laplace Transform

For a sampled signal $x_s(t) = \sum_n x(nT_s) \delta(t - nT_s)$, the relationship between the Laplace transform $X_s(s)$ and the Z-transform $X(z)$ is:

$$\boxed{z = e^{sT_s}}$$

or equivalently:

$$s = \frac{1}{T_s} \ln z$$

This maps:
- Left half of s-plane ($\text{Re}(s) < 0$) $\to$ Interior of unit circle ($|z| < 1$)
- Imaginary axis ($\text{Re}(s) = 0$) $\to$ Unit circle ($|z| = 1$)
- Right half of s-plane ($\text{Re}(s) > 0$) $\to$ Exterior of unit circle ($|z| > 1$)

### 8.3 Frequency Response from H(z)

To compute the frequency response at a specific physical frequency $f$ (Hz):

$$\omega = 2\pi f / f_s \quad \text{(normalized digital frequency)}$$
$$H(e^{j\omega}) = H(z)\big|_{z = e^{j2\pi f/f_s}}$$

```python
def frequency_response_from_hz():
    """Compute frequency response from H(z) by evaluating on unit circle."""
    # System: H(z) = (1 + z^{-1}) / (1 - 0.5 z^{-1})
    b = [1.0, 1.0]
    a = [1.0, -0.5]

    fs = 8000  # Hz

    # Method 1: Using scipy.signal.freqz
    w, H_scipy = signal.freqz(b, a, worN=1024)
    f_hz = w * fs / (2 * np.pi)

    # Method 2: Direct evaluation on unit circle
    omega = np.linspace(0, np.pi, 1024)
    z = np.exp(1j * omega)
    H_direct = (b[0] + b[1] * z**(-1)) / (a[0] + a[1] * z**(-1))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(f_hz, 20 * np.log10(np.abs(H_scipy)), 'b-',
                 linewidth=2, label='scipy.signal.freqz')
    axes[0].plot(f_hz, 20 * np.log10(np.abs(H_direct)), 'r--',
                 linewidth=1, label='Direct evaluation')
    axes[0].set_title(r'Frequency Response: $H(z) = (1+z^{-1})/(1-0.5z^{-1})$')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f_hz, np.unwrap(np.angle(H_scipy)) * 180 / np.pi, 'b-',
                 linewidth=2, label='scipy.signal.freqz')
    axes[1].plot(f_hz, np.unwrap(np.angle(H_direct)) * 180 / np.pi, 'r--',
                 linewidth=1, label='Direct evaluation')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('freq_response_hz.png', dpi=150)
    plt.show()

frequency_response_from_hz()
```

### 8.4 s-Plane to z-Plane Mapping

```python
def s_to_z_mapping():
    """Visualize the mapping from s-plane to z-plane."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    Ts = 1.0  # Normalized sampling period

    # s-plane
    ax = axes[0]
    # Stability boundary (imaginary axis)
    sigma = np.linspace(-3, 3, 100)
    omega_s = np.linspace(-np.pi / Ts, np.pi / Ts, 100)

    ax.axvline(0, color='red', linewidth=2, label='Stability boundary')
    ax.fill_betweenx([-4, 4], -4, 0, alpha=0.1, color='green',
                      label='Stable region')
    ax.fill_betweenx([-4, 4], 0, 4, alpha=0.1, color='red',
                      label='Unstable region')

    # Constant sigma lines
    for sig in [-2, -1, -0.5, 0.5, 1, 2]:
        ax.axvline(sig, color='gray', linestyle=':', alpha=0.3)

    # Constant omega lines
    for om in np.linspace(-3, 3, 7):
        ax.axhline(om, color='gray', linestyle=':', alpha=0.3)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-4, 4)
    ax.set_title('s-Plane')
    ax.set_xlabel(r'$\sigma$ (Real)')
    ax.set_ylabel(r'$j\omega$ (Imaginary)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # z-plane
    ax = axes[1]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2,
            label='Unit circle (stability)')

    # Map constant sigma lines
    for sig in [-2, -1, -0.5, 0, 0.5, 1, 2]:
        r = np.exp(sig * Ts)
        ax.plot(r * np.cos(theta), r * np.sin(theta), '--',
                alpha=0.4, label=f'sigma={sig}' if sig in [-1, 0, 1] else '')

    ax.fill_between(np.cos(theta), np.sin(theta), alpha=0.1, color='green')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title(r'z-Plane ($z = e^{sT_s}$)')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('s_to_z_mapping.png', dpi=150)
    plt.show()

s_to_z_mapping()
```

---

## 9. Geometric Interpretation of Frequency Response

### 9.1 Frequency Response as Vector Product

The frequency response at frequency $\omega$ can be computed geometrically:

$$|H(e^{j\omega})| = |G| \cdot \frac{\prod_{k=1}^{M} |e^{j\omega} - z_k|}{\prod_{k=1}^{N} |e^{j\omega} - p_k|}$$

$$\angle H(e^{j\omega}) = \angle G + \sum_{k=1}^{M} \angle(e^{j\omega} - z_k) - \sum_{k=1}^{N} \angle(e^{j\omega} - p_k)$$

As $\omega$ sweeps from $0$ to $\pi$, the point $e^{j\omega}$ traces the upper half of the unit circle. The magnitude is the product of distances to zeros divided by the product of distances to poles.

```python
def geometric_frequency_response():
    """Visualize the geometric interpretation of frequency response."""
    # System with poles and zeros
    zeros = [0.5 + 0.5j, 0.5 - 0.5j]
    poles = [0.8 * np.exp(1j * np.pi / 3), 0.8 * np.exp(-1j * np.pi / 3)]

    # Build transfer function from poles and zeros
    b = np.real(np.poly(zeros))
    a = np.real(np.poly(poles))

    # Animate for a specific frequency
    omega_target = np.pi / 3  # Target frequency
    z_point = np.exp(1j * omega_target)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pole-zero plot with vectors
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)

    # Zeros
    for z in zeros:
        ax.plot(np.real(z), np.imag(z), 'bo', markersize=10)
        # Vector from zero to point on unit circle
        ax.annotate('', xy=(np.real(z_point), np.imag(z_point)),
                    xytext=(np.real(z), np.imag(z)),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # Poles
    for p in poles:
        ax.plot(np.real(p), np.imag(p), 'rx', markersize=12, markeredgewidth=2)
        ax.annotate('', xy=(np.real(z_point), np.imag(z_point)),
                    xytext=(np.real(p), np.imag(p)),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Point on unit circle
    ax.plot(np.real(z_point), np.imag(z_point), 'g*', markersize=15,
            label=f'e^(j*{omega_target/np.pi:.2f}*pi)')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f'Geometric Vectors at omega = {omega_target/np.pi:.2f}*pi')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Frequency response
    ax = axes[1]
    w, H = signal.freqz(b, a, worN=1024)
    ax.plot(w / np.pi, 20 * np.log10(np.abs(H) + 1e-12), 'b-', linewidth=2)
    ax.axvline(omega_target / np.pi, color='green', linestyle='--',
               linewidth=2, label=f'omega = {omega_target/np.pi:.2f}*pi')

    H_at_target = np.polyval(b, z_point) / np.polyval(a, z_point)
    ax.plot(omega_target / np.pi, 20 * np.log10(np.abs(H_at_target)),
            'g*', markersize=15)

    ax.set_title('Magnitude Response')
    ax.set_xlabel(r'$\omega / \pi$')
    ax.set_ylabel('Magnitude (dB)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('geometric_freq_response.png', dpi=150)
    plt.show()

geometric_frequency_response()
```

---

## 10. Comprehensive Example: System Analysis Pipeline

```python
def complete_system_analysis():
    """Complete analysis of a digital system from difference equation to
    frequency response."""
    print("Complete System Analysis")
    print("=" * 60)

    # Difference equation:
    # y[n] - 1.2y[n-1] + 0.72y[n-2] = x[n] - 0.5x[n-1]
    b = [1.0, -0.5]
    a = [1.0, -1.2, 0.72]

    # 1. Transfer function
    print("\n1. Transfer function:")
    print(f"   H(z) = ({b[0]} + {b[1]}z^-1) / "
          f"({a[0]} + {a[1]}z^-1 + {a[2]}z^-2)")

    # 2. Poles and zeros
    zeros = np.roots(b)
    poles = np.roots(a)
    print(f"\n2. Zeros: {zeros}")
    print(f"   Poles: {poles}")
    print(f"   Pole magnitudes: {np.abs(poles)}")
    print(f"   Pole angles: {np.angle(poles) * 180 / np.pi} degrees")

    # 3. Stability
    stable = all(np.abs(poles) < 1)
    print(f"\n3. Stability: {'STABLE' if stable else 'UNSTABLE'}")
    print(f"   Max |pole| = {np.max(np.abs(poles)):.4f}")

    # 4. Partial fraction expansion
    r, p, k = signal.residuez(b, a)
    print(f"\n4. Partial fractions:")
    print(f"   Residues: {r}")
    print(f"   Poles: {p}")
    print(f"   Direct term: {k}")

    # 5. Impulse response (first 50 samples)
    _, h = signal.dimpulse(signal.dlti(b, a, dt=1), n=50)
    h = np.squeeze(h)

    # 6. Frequency response
    w, H = signal.freqz(b, a, worN=1024)

    # Plot everything
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2)

    # Pole-zero plot
    ax1 = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
    ax1.plot(np.real(zeros), np.imag(zeros), 'bo', markersize=10, label='Zeros')
    ax1.plot(np.real(poles), np.imag(poles), 'rx', markersize=12,
             markeredgewidth=2, label='Poles')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Pole-Zero Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Impulse response
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.stem(np.arange(len(h)), h, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax2.set_title('Impulse Response h[n]')
    ax2.set_xlabel('n')
    ax2.grid(True, alpha=0.3)

    # Magnitude response
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(w / np.pi, 20 * np.log10(np.abs(H) + 1e-12), 'b-', linewidth=2)
    ax3.set_title('Magnitude Response')
    ax3.set_xlabel(r'$\omega / \pi$')
    ax3.set_ylabel('dB')
    ax3.grid(True, alpha=0.3)

    # Phase response
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(w / np.pi, np.unwrap(np.angle(H)) * 180 / np.pi, 'r-',
             linewidth=2)
    ax4.set_title('Phase Response')
    ax4.set_xlabel(r'$\omega / \pi$')
    ax4.set_ylabel('Degrees')
    ax4.grid(True, alpha=0.3)

    # Group delay
    ax5 = fig.add_subplot(gs[2, 0])
    _, gd = signal.group_delay((b, a), w=1024)
    ax5.plot(w / np.pi, gd, 'g-', linewidth=2)
    ax5.set_title('Group Delay')
    ax5.set_xlabel(r'$\omega / \pi$')
    ax5.set_ylabel('Samples')
    ax5.grid(True, alpha=0.3)

    # Step response
    ax6 = fig.add_subplot(gs[2, 1])
    _, s = signal.dstep(signal.dlti(b, a, dt=1), n=50)
    s = np.squeeze(s)
    ax6.stem(np.arange(len(s)), s, linefmt='m-', markerfmt='mo', basefmt='k-')
    ax6.set_title('Step Response')
    ax6.set_xlabel('n')
    ax6.grid(True, alpha=0.3)

    plt.suptitle(r'System: $y[n] - 1.2y[n-1] + 0.72y[n-2] = x[n] - 0.5x[n-1]$',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('complete_analysis.png', dpi=150)
    plt.show()

complete_system_analysis()
```

---

## 11. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                      Z-Transform                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Definition:                                                     │
│    X(z) = Σ x[n] z^{-n}     z = r * e^{jω}                    │
│                                                                  │
│  ROC (Region of Convergence):                                    │
│    - Determines signal uniquely (same X(z), different ROC →     │
│      different x[n])                                            │
│    - Causal: |z| > R⁻  (exterior of circle)                    │
│    - Anti-causal: |z| < R⁺ (interior of circle)                │
│    - Two-sided: R⁻ < |z| < R⁺ (annular ring)                  │
│                                                                  │
│  Key Properties:                                                 │
│    - Time shift: x[n-m] ↔ z^{-m} X(z)                         │
│    - Convolution: x₁*x₂ ↔ X₁(z)·X₂(z)                        │
│    - z^{-1} = one-sample delay                                  │
│                                                                  │
│  Transfer Function:                                              │
│    H(z) = Y(z)/X(z) = B(z)/A(z)                                │
│    Poles → denominator roots                                    │
│    Zeros → numerator roots                                      │
│                                                                  │
│  Stability (causal systems):                                     │
│    ALL poles inside unit circle ↔ BIBO stable                   │
│                                                                  │
│  Relationships:                                                  │
│    Z-transform on unit circle = DTFT: X(e^{jω}) = X(z)|_{z=e^{jω}}│
│    z = e^{sTs} connects Z-transform to Laplace                 │
│                                                                  │
│  Inverse Methods:                                                │
│    1. Partial fractions → known pairs                           │
│    2. Long division → power series                              │
│    3. Contour integration → residue theorem                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Exercises

### Exercise 1: Z-Transform Computation

Compute the Z-transform and specify the ROC for each:

**(a)** $x[n] = (0.5)^n u[n] + (0.8)^n u[n]$

**(b)** $x[n] = (0.6)^n u[n] - 2(0.6)^n u[n-3]$

**(c)** $x[n] = n(0.9)^n u[n]$

**(d)** $x[n] = (0.7)^{|n|}$ (two-sided)

### Exercise 2: Inverse Z-Transform

Find $x[n]$ for each using partial fractions:

**(a)** $X(z) = \frac{z}{(z-0.5)(z-0.8)}, \quad |z| > 0.8$

**(b)** $X(z) = \frac{z}{(z-0.5)(z-0.8)}, \quad |z| < 0.5$

**(c)** $X(z) = \frac{z^2}{(z-0.5)(z-0.8)}, \quad |z| > 0.8$

**(d)** $X(z) = \frac{1+2z^{-1}}{1-z^{-1}+0.5z^{-2}}, \quad |z| > 0.707$

### Exercise 3: System Analysis

Given the difference equation: $y[n] = 0.8y[n-1] - 0.64y[n-2] + x[n] + x[n-1]$

**(a)** Find $H(z)$ and sketch the pole-zero plot.

**(b)** Is the system stable? Justify your answer.

**(c)** Find the impulse response $h[n]$ using partial fractions.

**(d)** Compute and plot the frequency response (magnitude and phase).

**(e)** Is this system a lowpass, highpass, bandpass, or bandstop filter?

### Exercise 4: Stability Determination

For each system, determine stability without computing roots:

**(a)** $H(z) = \frac{1}{1 - 0.5z^{-1} + 0.06z^{-2}}$

**(b)** $H(z) = \frac{z^2}{z^2 - 1.4z + 0.85}$

**(c)** $H(z) = \frac{1}{1 - 1.8\cos(0.4\pi)z^{-1} + 0.81z^{-2}}$

Use the Jury stability test and verify with Python.

### Exercise 5: ROC and Signal Determination

The Z-transform is $X(z) = \frac{2z^2 - 1.5z}{z^2 - 0.9z + 0.2}$.

**(a)** Find all possible ROCs.

**(b)** For each ROC, determine the corresponding signal $x[n]$ (causal, anti-causal, or two-sided).

**(c)** For which ROC does the DTFT exist?

### Exercise 6: Transfer Function Design

Design a second-order digital system with the following specifications:
- Resonant frequency at $\omega_0 = \pi/3$ rad/sample
- Bandwidth $\Delta\omega \approx 0.1$ rad/sample
- Unity gain at resonance

**(a)** Place poles at $z = re^{\pm j\omega_0}$ and choose $r$ for the desired bandwidth. (Hint: bandwidth $\approx 2(1-r)$ for $r$ close to 1.)

**(b)** Place zeros to achieve unity gain at resonance.

**(c)** Implement in Python and plot the frequency response.

**(d)** Test with a signal containing the resonant frequency and other frequencies.

### Exercise 7: Implementing a Digital Oscillator

A digital oscillator can be created using the difference equation:

$$y[n] = 2\cos(\omega_0) y[n-1] - y[n-2]$$

**(a)** Find $H(z)$ and its poles. Where are the poles located?

**(b)** Why is this system marginally stable? What happens in practice with finite-precision arithmetic?

**(c)** Implement the oscillator in Python and generate a 440 Hz tone at 8000 Hz sampling rate. Compare with a direct sine computation after 10000 samples.

---

## 13. Further Reading

- Oppenheim, Schafer. *Discrete-Time Signal Processing*, 3rd ed. Chapters 3, 5.
- Proakis, Manolakis. *Digital Signal Processing*, 4th ed. Chapters 3-4.
- Haykin, Van Veen. *Signals and Systems*, 2nd ed. Chapter 8.
- Roberts, M. J. *Signals and Systems*, Chapter 10.
- Phillips, Parr, Riskin. *Signals, Systems, and Transforms*, 5th ed. Chapters 8-9.

---

**Previous**: [06. Discrete Fourier Transform](06_Discrete_Fourier_Transform.md) | **Next**: [08. Digital Filter Fundamentals](08_Digital_Filter_Fundamentals.md)
