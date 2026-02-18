# 15. Image Signal Processing

**Previous**: [14. Time-Frequency Analysis](./14_Time_Frequency_Analysis.md) | **Next**: [16. Applications](./16_Applications.md)

---

An image is a two-dimensional signal: a function of spatial coordinates rather than time. Nearly every concept from 1D signal processing -- convolution, Fourier transforms, filtering, sampling -- has a natural 2D extension. This lesson develops the signal processing foundations of image analysis, covering spatial and frequency domain operations, edge detection, enhancement, and compression. We use NumPy and SciPy throughout, focusing on the underlying signal processing principles rather than high-level computer vision APIs.

**Difficulty**: ⭐⭐⭐

**Prerequisites**: 1D DFT, convolution, FIR filter design, sampling theorem

**Learning Objectives**:
- Represent images as 2D discrete signals and understand their sampling structure
- Compute and interpret the 2D Discrete Fourier Transform
- Apply 2D convolution for spatial domain filtering (smoothing, sharpening, median)
- Design and apply frequency domain filters (lowpass, highpass, bandpass)
- Implement gradient-based and Laplacian edge detectors, including the Canny algorithm
- Apply histogram equalization and contrast stretching for image enhancement
- Understand the DCT-based JPEG compression pipeline
- Explain 2D sampling, resolution, and the Nyquist criterion in two dimensions

---

## Table of Contents

1. [Images as 2D Signals](#1-images-as-2d-signals)
2. [2D Discrete Fourier Transform](#2-2d-discrete-fourier-transform)
3. [2D Convolution](#3-2d-convolution)
4. [Spatial Domain Filtering: Smoothing](#4-spatial-domain-filtering-smoothing)
5. [Spatial Domain Filtering: Sharpening](#5-spatial-domain-filtering-sharpening)
6. [Median Filtering](#6-median-filtering)
7. [Frequency Domain Filtering: Lowpass](#7-frequency-domain-filtering-lowpass)
8. [Frequency Domain Filtering: Highpass and Bandpass](#8-frequency-domain-filtering-highpass-and-bandpass)
9. [Edge Detection: Gradient Operators](#9-edge-detection-gradient-operators)
10. [Edge Detection: Laplacian and LoG](#10-edge-detection-laplacian-and-log)
11. [Canny Edge Detector](#11-canny-edge-detector)
12. [Image Enhancement](#12-image-enhancement)
13. [Image Compression: JPEG Overview](#13-image-compression-jpeg-overview)
14. [Sampling in 2D](#14-sampling-in-2d)
15. [Python Implementation](#15-python-implementation)
16. [Exercises](#16-exercises)
17. [Summary](#17-summary)
18. [References](#18-references)

---

## 1. Images as 2D Signals

### 1.1 Digital Image Representation

A grayscale digital image is a 2D function $f[m, n]$ where:
- $m \in \{0, 1, \ldots, M-1\}$ is the row index (vertical coordinate)
- $n \in \{0, 1, \ldots, N-1\}$ is the column index (horizontal coordinate)
- $f[m, n] \in [0, 255]$ for an 8-bit image (0 = black, 255 = white)

A color image has three channels: $f_R[m,n]$, $f_G[m,n]$, $f_B[m,n]$.

### 1.2 Image Formation

An image results from sampling a continuous light intensity field $f(x, y)$:

$$f[m, n] = f(m \Delta x, n \Delta y)$$

where $\Delta x$ and $\Delta y$ are the spatial sampling intervals. The reciprocals $1/\Delta x$ and $1/\Delta y$ are the spatial sampling frequencies.

### 1.3 Image Properties from a Signal Processing Perspective

| Property | 1D Signal | 2D Image |
|---|---|---|
| Independent variable | Time $t$ | Spatial coordinates $(x, y)$ |
| Sampling | Temporal sampling rate $f_s$ | Spatial resolution (pixels per unit length) |
| Frequency | Temporal frequency (Hz) | Spatial frequency (cycles/pixel or cycles/mm) |
| Filtering | Temporal filters | Spatial filters (kernels) |
| Transform | 1D DFT | 2D DFT |

### 1.4 Spatial Frequency

Spatial frequency describes how rapidly pixel values change across the image:
- **Low spatial frequencies**: Smooth regions, gradual intensity changes (sky, walls)
- **High spatial frequencies**: Edges, textures, fine details (hair, text, noise)

A sinusoidal pattern in an image: $f[m,n] = \cos(2\pi u_0 m + 2\pi v_0 n)$ has spatial frequencies $u_0$ (vertical, cycles/pixel) and $v_0$ (horizontal, cycles/pixel).

---

## 2. 2D Discrete Fourier Transform

### 2.1 Definition

The **2D DFT** of an $M \times N$ image $f[m, n]$ is:

$$\boxed{F[k, l] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} f[m, n] \, e^{-j2\pi(km/M + ln/N)}}$$

where $k \in \{0, \ldots, M-1\}$ and $l \in \{0, \ldots, N-1\}$.

The **inverse 2D DFT** is:

$$f[m, n] = \frac{1}{MN} \sum_{k=0}^{M-1} \sum_{l=0}^{N-1} F[k, l] \, e^{j2\pi(km/M + ln/N)}$$

### 2.2 Separability

The 2D DFT is **separable**: it can be computed as 1D DFTs along rows followed by 1D DFTs along columns (or vice versa):

$$F[k, l] = \sum_{m=0}^{M-1} \left(\sum_{n=0}^{N-1} f[m,n] \, e^{-j2\pi ln/N}\right) e^{-j2\pi km/M}$$

This reduces the complexity from $O(M^2 N^2)$ to $O(MN(\log M + \log N))$ using the FFT.

### 2.3 Interpreting the 2D Spectrum

The magnitude $|F[k,l]|$ shows the strength of each spatial frequency component:
- **Center** ($k = 0, l = 0$): DC component (average intensity)
- **Horizontal axis** ($k = 0$): Horizontal frequency components
- **Vertical axis** ($l = 0$): Vertical frequency components
- **Diagonal**: Diagonal frequency components
- **Distance from center**: Overall spatial frequency magnitude

The phase $\angle F[k,l]$ encodes the spatial position of features. Remarkably, the phase often carries more perceptual information than the magnitude.

### 2.4 Properties of the 2D DFT

| Property | Spatial Domain | Frequency Domain |
|---|---|---|
| Linearity | $af_1 + bf_2$ | $aF_1 + bF_2$ |
| Shift | $f[m-m_0, n-n_0]$ | $F[k,l] \, e^{-j2\pi(km_0/M + ln_0/N)}$ |
| Convolution | $f * g$ | $F \cdot G$ |
| Correlation | $f \star g$ | $F^* \cdot G$ |
| Parseval's | $\sum|f|^2 = \frac{1}{MN}\sum|F|^2$ | (Energy conservation) |
| Conjugate symmetry | $f$ real | $F[k,l] = F^*[-k,-l]$ |

### 2.5 Centering the Spectrum

By default, `np.fft.fft2` places the DC component at corner $(0,0)$. To center it (for visualization), use `np.fft.fftshift`, which swaps quadrants. Equivalently, multiply the spatial image by $(-1)^{m+n}$ before the DFT.

---

## 3. 2D Convolution

### 3.1 Definition

The 2D discrete convolution of image $f[m,n]$ with kernel $h[m,n]$ (size $K \times L$) is:

$$\boxed{g[m, n] = (f * h)[m, n] = \sum_{i=0}^{K-1} \sum_{j=0}^{L-1} f[m-i, n-j] \, h[i, j]}$$

In practice, we usually use **correlation** (flipped kernel), which is equivalent to convolution for symmetric kernels:

$$(f \star h)[m, n] = \sum_{i=0}^{K-1} \sum_{j=0}^{L-1} f[m+i-K/2, n+j-L/2] \, h[i, j]$$

### 3.2 Boundary Handling

When the kernel extends beyond the image boundary, we must handle the edges:

| Method | Description | Effect |
|---|---|---|
| Zero-padding | Assume 0 outside | Dark borders |
| Reflect | Mirror the image | Natural-looking |
| Wrap | Periodic extension | Circular convolution |
| Replicate | Extend edge pixels | Minimal artifacts |
| Crop | Output only valid region | Smaller output |

### 3.3 Convolution Theorem in 2D

$$f * h \xleftrightarrow{\text{DFT}} F \cdot H$$

Frequency domain filtering is efficient for large kernels:
- **Spatial convolution**: $O(MN \cdot KL)$ operations
- **Frequency domain**: $O(MN \log(MN))$ (FFT + pointwise multiply + IFFT)

Frequency domain filtering is faster when $KL > \log(MN)$.

---

## 4. Spatial Domain Filtering: Smoothing

### 4.1 Averaging (Box) Filter

The simplest smoothing filter replaces each pixel with the average of its neighbors:

$$h_{avg} = \frac{1}{K^2} \begin{bmatrix} 1 & 1 & \cdots & 1 \\ 1 & 1 & \cdots & 1 \\ \vdots & & & \vdots \\ 1 & 1 & \cdots & 1 \end{bmatrix}_{K \times K}$$

For $3 \times 3$:
$$h_{3\times3} = \frac{1}{9}\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

**Effect**: Blurs the image, reduces noise, but also blurs edges.

**Frequency response**: The 2D box filter is a 2D sinc function in the frequency domain -- it has significant sidelobe leakage.

### 4.2 Gaussian Filter

A better smoothing filter uses a 2D Gaussian kernel:

$$h_{Gauss}[m, n] = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{m^2 + n^2}{2\sigma^2}\right)$$

The Gaussian filter is:
- **Separable**: $h[m,n] = g[m] \cdot g[n]$ where $g[k] = \exp(-k^2/(2\sigma^2))$
- **Isotropic**: Same smoothing in all directions
- **No ringing**: Gaussian in frequency domain too (no sidelobes)
- **Optimal**: Minimizes the space-frequency uncertainty product

**Separability** reduces computation from $O(K^2)$ to $O(2K)$ per pixel.

Common kernel sizes: $\sigma = 1 \to 5 \times 5$, $\sigma = 2 \to 9 \times 9$ (choose size $\geq 6\sigma + 1$).

### 4.3 Weighted Averaging

A compromise between box and Gaussian:

$$h_{weighted} = \frac{1}{16}\begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$

This is an approximation to a Gaussian with $\sigma \approx 0.85$.

---

## 5. Spatial Domain Filtering: Sharpening

### 5.1 The Laplacian Operator

The **Laplacian** is a second-order derivative operator that highlights regions of rapid intensity change:

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

Discrete approximation (4-connected):
$$\nabla^2 f[m,n] \approx f[m+1,n] + f[m-1,n] + f[m,n+1] + f[m,n-1] - 4f[m,n]$$

Kernel:
$$h_{Lap4} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

Including diagonals (8-connected):
$$h_{Lap8} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & -8 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

### 5.2 Laplacian Sharpening

Sharpening is achieved by subtracting the Laplacian from the original image:

$$g[m,n] = f[m,n] - c \cdot \nabla^2 f[m,n]$$

where $c > 0$ controls the sharpening strength. The combined kernel for $c = 1$ (4-connected Laplacian):

$$h_{sharp} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

### 5.3 Unsharp Masking

A classic technique from photography:

1. Blur the image: $f_{blur} = f * h_{Gauss}$
2. Compute the "unsharp mask": $\text{mask} = f - f_{blur}$
3. Add a scaled mask back: $g = f + k \cdot \text{mask}$

where $k > 0$ is the sharpening factor.

$$g = (1 + k) f - k (f * h_{Gauss}) = f + k(f - f * h_{Gauss})$$

In the frequency domain: $G = F + k(F - F \cdot H_{Gauss}) = F(1 + k - kH_{Gauss})$

This is a **highpass emphasis filter**: it boosts high frequencies while leaving the DC component unchanged.

---

## 6. Median Filtering

### 6.1 Definition

The **median filter** replaces each pixel with the median of pixel values in a local neighborhood:

$$g[m,n] = \text{median}\{f[m+i, n+j] : (i,j) \in \mathcal{W}\}$$

where $\mathcal{W}$ is the filter window (typically $3 \times 3$ or $5 \times 5$).

### 6.2 Properties

- **Non-linear**: Cannot be expressed as a convolution
- **Edge-preserving**: Smooths noise while maintaining sharp edges
- **Excellent for impulse noise**: Removes salt-and-pepper noise completely
- **Rank-order filter**: Special case of the more general class of order-statistic filters

### 6.3 Comparison with Linear Filters

| Feature | Gaussian Filter | Median Filter |
|---|---|---|
| Type | Linear | Non-linear |
| Gaussian noise | Good | Fair |
| Salt-and-pepper noise | Poor (blurs spikes) | Excellent (removes spikes) |
| Edge preservation | Blurs edges | Preserves edges |
| Computation | $O(K^2)$ per pixel | $O(K^2 \log K)$ per pixel (sorting) |
| Frequency domain | Lowpass (Gaussian) | No simple frequency interpretation |

---

## 7. Frequency Domain Filtering: Lowpass

### 7.1 General Procedure

1. Compute 2D DFT: $F = \text{FFT2}(f)$
2. Center the spectrum: $F_c = \text{fftshift}(F)$
3. Multiply by filter: $G_c = F_c \cdot H$
4. Un-center: $G = \text{ifftshift}(G_c)$
5. Inverse DFT: $g = \text{IFFT2}(G)$

### 7.2 Ideal Lowpass Filter

$$H_{ideal}[k, l] = \begin{cases} 1 & \sqrt{k^2 + l^2} \leq D_0 \\ 0 & \text{otherwise} \end{cases}$$

where $D_0$ is the cutoff frequency (in pixels from center).

**Problem**: The ideal filter has infinite spatial extent (2D sinc), causing **ringing artifacts** (Gibbs phenomenon).

### 7.3 Butterworth Lowpass Filter

$$H_{Butter}[k, l] = \frac{1}{1 + \left(\frac{D(k,l)}{D_0}\right)^{2n}}$$

where $D(k,l) = \sqrt{k^2 + l^2}$ is the distance from the center and $n$ is the filter order.

- $n = 1$: Gentle rolloff, minimal ringing
- $n = 2$: Good tradeoff (commonly used)
- $n \to \infty$: Approaches ideal filter

### 7.4 Gaussian Lowpass Filter

$$H_{Gauss}[k, l] = \exp\!\left(-\frac{D(k,l)^2}{2D_0^2}\right)$$

**Properties**:
- No ringing (Gaussian in frequency $\to$ Gaussian in space)
- Smooth rolloff
- $D_0$ corresponds to the frequency where $H = e^{-1/2} \approx 0.607$

### 7.5 Comparison of Lowpass Filters

```
|H(D)|
  1 ─── ─────────┐
  │    Ideal ──────┘
  │    Butter(n=2) ─────╲
  │    Gaussian ──────────╲
  │                         ╲
  0 ────────────────────────────▶ D
                   D₀
```

The Gaussian filter provides the smoothest spatial response (no ringing) at the cost of the least sharp frequency cutoff.

---

## 8. Frequency Domain Filtering: Highpass and Bandpass

### 8.1 Highpass Filters

A highpass filter is the complement of a lowpass filter:

$$H_{HP}(k, l) = 1 - H_{LP}(k, l)$$

Highpass filtering preserves edges and fine detail while removing smooth, slowly varying regions.

**Ideal highpass**:
$$H_{ideal,HP}[k,l] = \begin{cases} 0 & D(k,l) \leq D_0 \\ 1 & D(k,l) > D_0 \end{cases}$$

**Gaussian highpass**:
$$H_{Gauss,HP}[k,l] = 1 - \exp\!\left(-\frac{D(k,l)^2}{2D_0^2}\right)$$

### 8.2 High-Boost Filter (Highpass Emphasis)

To retain some of the low-frequency content:

$$H_{boost} = a + (1-a) \cdot H_{HP} = a \cdot H_{LP} + H_{HP}$$

For $a > 1$: emphasizes high frequencies while preserving the overall image.

This is equivalent to unsharp masking in the frequency domain.

### 8.3 Bandpass and Bandstop (Notch) Filters

**Bandpass filter** passes frequencies in a ring between $D_1$ and $D_2$:

$$H_{BP}[k,l] = \begin{cases} 1 & D_1 \leq D(k,l) \leq D_2 \\ 0 & \text{otherwise} \end{cases}$$

**Notch filter** removes specific frequency components (useful for removing periodic noise patterns):

$$H_{notch}[k,l] = \prod_{i=1}^{Q} \frac{1}{1 + \left(\frac{D_{0i}}{D_i(k,l)}\right)^{2n}}$$

where $D_i(k,l)$ is the distance from the $i$-th notch center.

**Application**: Removing periodic interference patterns (e.g., moire patterns, scanning artifacts).

---

## 9. Edge Detection: Gradient Operators

### 9.1 Image Gradient

The gradient of a 2D image $f(x,y)$ is:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} G_x \\ G_y \end{bmatrix}$$

The **gradient magnitude** gives edge strength:
$$|\nabla f| = \sqrt{G_x^2 + G_y^2} \approx |G_x| + |G_y|$$

The **gradient direction** gives edge orientation:
$$\theta = \arctan\!\left(\frac{G_y}{G_x}\right)$$

### 9.2 Sobel Operator

The Sobel operator computes smoothed gradient estimates:

$$G_x = h_x * f = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * f$$

$$G_y = h_y * f = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * f$$

**Design rationale**: The Sobel kernel is separable: $h_x = \begin{bmatrix}1\\2\\1\end{bmatrix} \begin{bmatrix}-1&0&1\end{bmatrix}$. The first factor smooths in the $y$-direction, and the second computes the derivative in the $x$-direction. The smoothing reduces noise sensitivity.

### 9.3 Prewitt Operator

Similar to Sobel but with uniform weighting:

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix} * f, \qquad G_y = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix} * f$$

### 9.4 Scharr Operator

An improved gradient approximation with better rotational symmetry:

$$G_x = \begin{bmatrix} -3 & 0 & 3 \\ -10 & 0 & 10 \\ -3 & 0 & 3 \end{bmatrix} * f$$

### 9.5 Roberts Cross Operator

The simplest gradient operator, using $2 \times 2$ kernels:

$$G_x = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} * f, \qquad G_y = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} * f$$

Sensitive to noise due to the small kernel size.

---

## 10. Edge Detection: Laplacian and LoG

### 10.1 Laplacian Edge Detection

The Laplacian detects edges as **zero crossings** of the second derivative:

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

At an edge: the first derivative has a peak, and the second derivative crosses zero.

**Advantages**: Isotropic (detects edges in all directions equally).

**Disadvantages**: Highly sensitive to noise (amplifies high-frequency noise even more than first-order operators).

### 10.2 Laplacian of Gaussian (LoG)

To mitigate noise sensitivity, first smooth with a Gaussian, then apply the Laplacian:

$$\text{LoG}(x, y) = \nabla^2 [G_\sigma * f] = [\nabla^2 G_\sigma] * f$$

The LoG kernel (also called the "Mexican hat" in 1D) is:

$$\text{LoG}(x, y) = -\frac{1}{\pi\sigma^4}\left[1 - \frac{x^2 + y^2}{2\sigma^2}\right] \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

**Marr-Hildreth edge detector**: Find zero crossings of $\text{LoG} * f$.

### 10.3 Difference of Gaussians (DoG)

An efficient approximation to the LoG:

$$\text{DoG}(x,y) = G_{\sigma_1}(x,y) - G_{\sigma_2}(x,y) \approx (\sigma_2 - \sigma_1) \cdot \nabla^2 G_{\bar{\sigma}}$$

Typically $\sigma_2 / \sigma_1 \approx 1.6$. The DoG is the basis of the SIFT feature detector.

---

## 11. Canny Edge Detector

### 11.1 Design Criteria

John Canny (1986) formulated edge detection as an optimization problem with three criteria:

1. **Good detection**: Low probability of missing real edges and low probability of false edges
2. **Good localization**: Detected edges should be as close as possible to the true edges
3. **Single response**: Only one response per edge (no multiple detections)

### 11.2 Algorithm Steps

```
Step 1: Gaussian smoothing
────────────────────────
f_smooth = f * G_σ

Step 2: Gradient computation (Sobel)
────────────────────────────────────
G_x = h_x * f_smooth
G_y = h_y * f_smooth
Magnitude = sqrt(G_x² + G_y²)
Direction θ = atan2(G_y, G_x)

Step 3: Non-maximum suppression
───────────────────────────────
For each pixel:
  - Quantize θ to 0°, 45°, 90°, 135°
  - Compare magnitude with neighbors along gradient direction
  - Suppress (set to 0) if not a local maximum

Step 4: Double thresholding
───────────────────────────
Strong edges:  magnitude > T_high
Weak edges:    T_low < magnitude ≤ T_high
Non-edges:     magnitude ≤ T_low

Step 5: Edge tracking by hysteresis
────────────────────────────────────
- Strong edges are always edges
- Weak edges are edges only if connected to a strong edge
- Non-edges are discarded
```

### 11.3 Non-Maximum Suppression

This step thins the edges to 1-pixel width by suppressing pixels that are not the local maximum along the gradient direction:

```
Gradient direction:          Neighbors to compare:
       0° (horizontal)       left, right
      45° (diagonal)         upper-right, lower-left
      90° (vertical)         above, below
     135° (anti-diagonal)    upper-left, lower-right
```

### 11.4 Hysteresis Thresholding

Using two thresholds ($T_{low}$ and $T_{high}$, typically $T_{low} = 0.4 T_{high}$) prevents:
- Broken edges (that a single high threshold would cause)
- Excessive false edges (that a single low threshold would cause)

The hysteresis step connects weak edge pixels to strong edge pixels using connectivity analysis (e.g., BFS or DFS).

### 11.5 Parameter Selection

| Parameter | Effect | Typical Range |
|---|---|---|
| $\sigma$ (Gaussian) | Noise suppression vs edge localization | 1.0 - 3.0 |
| $T_{high}$ | Strong edge threshold | 70th-90th percentile of gradient |
| $T_{low}$ | Weak edge threshold | $0.3 T_{high}$ to $0.5 T_{high}$ |

---

## 12. Image Enhancement

### 12.1 Histogram

The **histogram** of an image counts the number of pixels at each intensity level:

$$h[k] = \text{number of pixels with intensity } k, \quad k = 0, 1, \ldots, L-1$$

The **normalized histogram** (probability mass function):

$$p[k] = \frac{h[k]}{M \cdot N}$$

### 12.2 Histogram Equalization

Histogram equalization redistributes pixel intensities to achieve a (approximately) uniform histogram, maximizing contrast.

**Continuous case**: The transformation $s = T(r)$ that produces a uniform distribution is:

$$s = T(r) = \int_0^r p_r(\rho) \, d\rho = \text{CDF}(r)$$

**Discrete case**: For an $L$-level image:

$$s_k = T(r_k) = (L-1) \sum_{j=0}^{k} p[j] = (L-1) \cdot \text{CDF}[k]$$

**Algorithm**:
1. Compute the histogram $h[k]$
2. Compute the CDF: $\text{CDF}[k] = \sum_{j=0}^{k} p[j]$
3. Map each pixel: $f_{eq}[m,n] = \text{round}((L-1) \cdot \text{CDF}[f[m,n]])$

### 12.3 Contrast Limited Adaptive Histogram Equalization (CLAHE)

Global histogram equalization can over-enhance noise in smooth regions. CLAHE addresses this by:

1. Dividing the image into small tiles (e.g., $8 \times 8$)
2. Applying histogram equalization to each tile with a **clip limit** that caps the histogram (redistributing excess counts)
3. Interpolating between tiles for seamless results

### 12.4 Contrast Stretching

**Linear contrast stretching** maps the intensity range $[r_{min}, r_{max}]$ to the full range $[0, L-1]$:

$$s = \frac{L-1}{r_{max} - r_{min}} (r - r_{min})$$

**Percentile-based stretching**: Use the 2nd and 98th percentiles instead of min/max to avoid outlier influence.

### 12.5 Gamma Correction

A power-law transformation:

$$s = c \cdot r^\gamma$$

- $\gamma < 1$: Brightens dark regions (expands dark tones)
- $\gamma = 1$: No change
- $\gamma > 1$: Darkens bright regions (expands bright tones)

---

## 13. Image Compression: JPEG Overview

### 13.1 The JPEG Pipeline

JPEG (Joint Photographic Experts Group) is the most widely used image compression standard. It exploits two properties:
1. **Spatial redundancy**: Nearby pixels are correlated
2. **Psychovisual redundancy**: Humans are less sensitive to high-frequency detail

```
JPEG Encoding Pipeline:
────────────────────────────────────────────────────────────

RGB → YCbCr → Chroma    → 8×8 Block → DCT → Quantize → Entropy
              Subsampling  Partition              ↓       Encoding
                                             Q-Table     (Huffman
                                                          or
                                                          Arithmetic)
```

### 13.2 Color Space Conversion

Convert from RGB to YCbCr:
- **Y**: Luminance (brightness) -- the most important component
- **Cb**: Blue-difference chrominance
- **Cr**: Red-difference chrominance

$$\begin{bmatrix} Y \\ C_b \\ C_r \end{bmatrix} = \begin{bmatrix} 0.299 & 0.587 & 0.114 \\ -0.169 & -0.331 & 0.500 \\ 0.500 & -0.419 & -0.081 \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix} + \begin{bmatrix} 0 \\ 128 \\ 128 \end{bmatrix}$$

### 13.3 Chroma Subsampling

Human vision has lower spatial resolution for color than for brightness. JPEG exploits this by subsampling the chrominance channels:
- **4:4:4**: No subsampling (full resolution)
- **4:2:2**: Horizontal subsampling by 2
- **4:2:0**: Both horizontal and vertical subsampling by 2 (most common)

### 13.4 The 2D Discrete Cosine Transform (DCT)

Each $8 \times 8$ block is transformed using the Type-II DCT:

$$F[u, v] = \frac{1}{4} C_u C_v \sum_{m=0}^{7} \sum_{n=0}^{7} f[m,n] \cos\!\left(\frac{(2m+1)u\pi}{16}\right) \cos\!\left(\frac{(2n+1)v\pi}{16}\right)$$

where $C_u = 1/\sqrt{2}$ for $u = 0$, $C_u = 1$ otherwise.

The DCT is preferred over the DFT because:
- Real-valued (no complex arithmetic)
- Better energy compaction (more energy concentrated in fewer coefficients)
- No discontinuity artifacts at block boundaries (the DCT implicitly assumes an even extension)

### 13.5 Quantization

The DCT coefficients are divided by a **quantization matrix** $Q[u,v]$ and rounded:

$$F_Q[u,v] = \text{round}\!\left(\frac{F[u,v]}{Q[u,v]}\right)$$

The quantization matrix has larger values for high-frequency components (more aggressive compression of frequencies humans are less sensitive to). This is the **lossy** step -- information is irreversibly discarded.

**Quality factor**: Scaling the $Q$ matrix controls the quality-size tradeoff.

### 13.6 Entropy Coding

The quantized coefficients are encoded using:
1. **Zigzag scanning**: Reads the $8 \times 8$ block in a zigzag pattern to group low-frequency (large) coefficients first and high-frequency (often zero) coefficients last
2. **Run-length encoding (RLE)**: Efficiently encodes runs of zeros
3. **Huffman coding** (or arithmetic coding): Variable-length coding for further compression

### 13.7 JPEG Artifacts

At high compression ratios, JPEG produces characteristic artifacts:
- **Blocking**: Visible $8 \times 8$ grid boundaries
- **Mosquito noise**: Ringing around sharp edges
- **Color bleeding**: Chrominance artifacts near high-contrast edges

---

## 14. Sampling in 2D

### 14.1 2D Sampling Theorem

A 2D continuous signal $f(x, y)$ bandlimited to $[-u_{max}, u_{max}] \times [-v_{max}, v_{max}]$ can be perfectly reconstructed from its samples if:

$$\frac{1}{\Delta x} > 2u_{max}, \quad \frac{1}{\Delta y} > 2v_{max}$$

where $\Delta x$ and $\Delta y$ are the sampling intervals.

### 14.2 Aliasing in 2D

When the sampling condition is violated, high-frequency content folds back and overlaps with low-frequency content. In images, aliasing manifests as:
- **Moire patterns**: Regular patterns that don't exist in the scene
- **Jagged edges** (staircasing): Diagonal lines appear as staircase steps
- **Wagon wheel effect**: In video, rotating wheels appear to spin backward

### 14.3 Anti-Aliasing

Before downsampling an image, apply a lowpass filter to remove frequencies above the new Nyquist frequency:

1. Filter: $f_{filtered} = f * h_{LP}$
2. Downsample: $f_{down}[m,n] = f_{filtered}[Dm, Dn]$

where $D$ is the downsampling factor and the cutoff of $h_{LP}$ is $\pi/D$.

### 14.4 Image Interpolation (Upsampling)

Common interpolation methods for image resizing:
- **Nearest neighbor**: Block artifacts, sharp
- **Bilinear**: Smooth, slight blurring
- **Bicubic**: Sharper than bilinear, slight ringing possible
- **Sinc (ideal)**: Perfect reconstruction but infinite support

Bilinear interpolation:

$$f(x, y) = (1-a)(1-b)f[m,n] + a(1-b)f[m,n+1] + (1-a)b f[m+1,n] + ab \, f[m+1,n+1]$$

where $a = x - m$ and $b = y - n$ are the fractional parts.

---

## 15. Python Implementation

### 15.1 2D DFT Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def create_test_image(size=256):
    """Create a test image with various spatial frequencies."""
    img = np.zeros((size, size))

    # Horizontal bars (vertical frequency)
    y = np.arange(size)
    for freq in [5, 15, 40]:
        stripe = np.sin(2 * np.pi * freq * y / size)
        img[:, size//6:size//3] += stripe[:, np.newaxis] * (freq == 5)
        img[:, size//3:size//2] += stripe[:, np.newaxis] * (freq == 15)
        img[:, size//2:2*size//3] += stripe[:, np.newaxis] * (freq == 40)

    # Add a rectangle
    img[size//4:3*size//4, 3*size//4:7*size//8] = 1.0

    # Add a circle
    yy, xx = np.mgrid[:size, :size]
    circle = ((xx - size//8)**2 + (yy - 3*size//4)**2) < (size//10)**2
    img[circle] = 1.0

    return img


# Create test image
img = create_test_image(256)

# Compute 2D DFT
F = np.fft.fft2(img)
F_shifted = np.fft.fftshift(F)
magnitude = np.log1p(np.abs(F_shifted))  # Log scale for visualization
phase = np.angle(F_shifted)

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(magnitude, cmap='hot')
axes[1].set_title('Magnitude Spectrum (log scale)')
axes[1].axis('off')

axes[2].imshow(phase, cmap='hsv')
axes[2].set_title('Phase Spectrum')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('2d_dft_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 15.2 Spatial Filtering: Smoothing and Sharpening

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal


def generate_noisy_image(size=256, noise_std=20):
    """Generate a test image with noise."""
    # Create a clean image with geometric shapes
    img = np.zeros((size, size))

    # Rectangle
    img[50:200, 30:100] = 180

    # Circle
    yy, xx = np.mgrid[:size, :size]
    circle = ((xx - 180)**2 + (yy - 130)**2) < 50**2
    img[circle] = 220

    # Triangle
    for i in range(80):
        img[170+i, 130+i//2:230-i//2] = 140

    # Add Gaussian noise
    noisy = img + noise_std * np.random.randn(size, size)
    return img.clip(0, 255), noisy.clip(0, 255)


np.random.seed(42)
clean, noisy = generate_noisy_image()

# Define kernels
box_3x3 = np.ones((3, 3)) / 9
box_5x5 = np.ones((5, 5)) / 25

# Gaussian kernel
sigma = 1.5
size_g = int(6*sigma + 1) | 1  # Ensure odd
ax = np.arange(-size_g//2 + 1, size_g//2 + 1)
xx, yy = np.meshgrid(ax, ax)
gaussian_kernel = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
gaussian_kernel /= gaussian_kernel.sum()

# Laplacian sharpening kernel
laplacian = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

# Apply filters
smoothed_box = ndimage.convolve(noisy, box_5x5)
smoothed_gauss = ndimage.convolve(noisy, gaussian_kernel)
sharpened = ndimage.convolve(clean, laplacian)  # Sharpen the clean image

# Unsharp mask
blurred = ndimage.gaussian_filter(clean, sigma=2.0)
unsharp = clean + 1.5 * (clean - blurred)
unsharp = np.clip(unsharp, 0, 255)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(clean, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Clean Image')

axes[0, 1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('Noisy Image (σ=20)')

axes[0, 2].imshow(smoothed_box, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title('Box Filter 5×5')

axes[1, 0].imshow(smoothed_gauss, cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title(f'Gaussian Filter (σ={sigma})')

axes[1, 1].imshow(np.clip(sharpened, 0, 255), cmap='gray', vmin=0, vmax=255)
axes[1, 1].set_title('Laplacian Sharpening')

axes[1, 2].imshow(unsharp, cmap='gray', vmin=0, vmax=255)
axes[1, 2].set_title('Unsharp Masking (k=1.5)')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('spatial_filtering.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 15.3 Frequency Domain Filtering

```python
import numpy as np
import matplotlib.pyplot as plt


def frequency_filter(img, filter_func, *args):
    """Apply a frequency domain filter to an image."""
    M, N = img.shape
    # Compute centered 2D DFT
    F = np.fft.fftshift(np.fft.fft2(img))

    # Create frequency grid
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)

    # Apply filter
    H = filter_func(D, *args)
    G = F * H

    # Inverse DFT
    g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    return g, H


def ideal_lowpass(D, D0):
    return (D <= D0).astype(float)

def butterworth_lowpass(D, D0, n=2):
    return 1.0 / (1 + (D / D0)**(2*n))

def gaussian_lowpass(D, D0):
    return np.exp(-D**2 / (2 * D0**2))


# Create test image
np.random.seed(42)
size = 256
img = np.zeros((size, size))
img[80:180, 60:200] = 200
yy, xx = np.mgrid[:size, :size]
circle = ((xx - 128)**2 + (yy - 128)**2) < 40**2
img[circle] = 255
img += 15 * np.random.randn(size, size)
img = np.clip(img, 0, 255)

# Apply different lowpass filters
D0 = 30  # Cutoff frequency

filtered_ideal, H_ideal = frequency_filter(img, ideal_lowpass, D0)
filtered_butter, H_butter = frequency_filter(img, butterworth_lowpass, D0, 2)
filtered_gauss, H_gauss = frequency_filter(img, gaussian_lowpass, D0)

# Highpass (complement of Gaussian lowpass)
filtered_hp, H_hp = frequency_filter(
    img, lambda D, D0: 1 - gaussian_lowpass(D, D0), D0
)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Top row: filters
axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original')

axes[0, 1].imshow(H_ideal, cmap='gray')
axes[0, 1].set_title(f'Ideal LP (D₀={D0})')

axes[0, 2].imshow(H_butter, cmap='gray')
axes[0, 2].set_title(f'Butterworth LP (D₀={D0}, n=2)')

axes[0, 3].imshow(H_gauss, cmap='gray')
axes[0, 3].set_title(f'Gaussian LP (D₀={D0})')

# Bottom row: filtered images
axes[1, 0].imshow(np.clip(filtered_ideal, 0, 255), cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title('Ideal LP (ringing)')

axes[1, 1].imshow(np.clip(filtered_butter, 0, 255), cmap='gray', vmin=0, vmax=255)
axes[1, 1].set_title('Butterworth LP')

axes[1, 2].imshow(np.clip(filtered_gauss, 0, 255), cmap='gray', vmin=0, vmax=255)
axes[1, 2].set_title('Gaussian LP (smooth)')

axes[1, 3].imshow(np.clip(filtered_hp + 128, 0, 255), cmap='gray')
axes[1, 3].set_title('Gaussian HP')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('frequency_domain_filtering.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 15.4 Edge Detection

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def sobel_edge_detection(img):
    """Manual Sobel edge detection."""
    # Sobel kernels
    Kx = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)
    Ky = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]], dtype=float)

    Gx = ndimage.convolve(img, Kx)
    Gy = ndimage.convolve(img, Ky)

    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.arctan2(Gy, Gx)

    return magnitude, direction, Gx, Gy


def canny_edge_detector(img, sigma=1.0, low_thresh=0.1, high_thresh=0.3):
    """
    Manual Canny edge detector implementation.

    Parameters
    ----------
    img : ndarray
        Input grayscale image (float, 0-1 range)
    sigma : float
        Gaussian smoothing parameter
    low_thresh : float
        Low threshold (fraction of max gradient)
    high_thresh : float
        High threshold (fraction of max gradient)

    Returns
    -------
    edges : ndarray
        Binary edge map
    """
    # Step 1: Gaussian smoothing
    smoothed = ndimage.gaussian_filter(img, sigma)

    # Step 2: Gradient computation (Sobel)
    magnitude, direction, _, _ = sobel_edge_detection(smoothed)

    # Normalize
    magnitude = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude

    # Step 3: Non-maximum suppression
    M, N = magnitude.shape
    nms = np.zeros_like(magnitude)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # Determine neighbor direction
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = magnitude[i, j-1], magnitude[i, j+1]
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[i-1, j+1], magnitude[i+1, j-1]
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[i-1, j], magnitude[i+1, j]
            else:
                n1, n2 = magnitude[i-1, j-1], magnitude[i+1, j+1]

            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                nms[i, j] = magnitude[i, j]

    # Step 4: Double thresholding
    strong = nms >= high_thresh
    weak = (nms >= low_thresh) & (nms < high_thresh)

    # Step 5: Edge tracking by hysteresis (simplified using dilation)
    edges = strong.copy()
    # Connect weak edges adjacent to strong edges
    for _ in range(10):  # Iterate to propagate connectivity
        dilated = ndimage.binary_dilation(edges)
        edges = edges | (weak & dilated)

    return edges.astype(float)


# Create a test image
np.random.seed(42)
size = 256
img = np.zeros((size, size))
# Rectangle
img[60:200, 40:220] = 0.8
# Inner rectangle
img[90:170, 70:190] = 0.3
# Circle
yy, xx = np.mgrid[:size, :size]
circle = ((xx - 128)**2 + (yy - 128)**2) < 30**2
img[circle] = 1.0
# Add mild noise
img += 0.02 * np.random.randn(size, size)
img = np.clip(img, 0, 1)

# Edge detection methods
magnitude, direction, Gx, Gy = sobel_edge_detection(img)

# Laplacian
laplacian = ndimage.laplace(ndimage.gaussian_filter(img, 1.0))

# LoG
log_result = ndimage.gaussian_laplace(img, sigma=2.0)

# Canny
canny_edges = canny_edge_detector(img, sigma=1.5, low_thresh=0.05, high_thresh=0.15)

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')

axes[0, 1].imshow(magnitude, cmap='gray')
axes[0, 1].set_title('Sobel Magnitude')

axes[0, 2].imshow(direction, cmap='hsv')
axes[0, 2].set_title('Sobel Direction')

axes[1, 0].imshow(np.abs(laplacian), cmap='gray')
axes[1, 0].set_title('Laplacian (after Gaussian)')

axes[1, 1].imshow(np.abs(log_result), cmap='gray')
axes[1, 1].set_title('Laplacian of Gaussian (σ=2)')

axes[1, 2].imshow(canny_edges, cmap='gray')
axes[1, 2].set_title('Canny Edge Detector')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('edge_detection.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 15.5 Histogram Equalization

```python
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(img):
    """
    Apply histogram equalization to an 8-bit grayscale image.

    Parameters
    ----------
    img : ndarray
        Input image (uint8 or float [0, 255])

    Returns
    -------
    equalized : ndarray
        Histogram-equalized image
    """
    img_int = img.astype(np.uint8)
    L = 256

    # Compute histogram
    hist, bins = np.histogram(img_int, bins=L, range=(0, L))

    # Compute CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalize to [0, 1]

    # Map pixel values
    equalized = np.round((L - 1) * cdf_normalized[img_int]).astype(np.uint8)

    return equalized


# Create a low-contrast test image
np.random.seed(42)
size = 256
yy, xx = np.mgrid[:size, :size]

# Low-contrast image (values concentrated in [80, 170])
img = 80 + 90 * (np.sin(2*np.pi*3*xx/size) * np.cos(2*np.pi*2*yy/size) + 1) / 2
circle = ((xx - 128)**2 + (yy - 128)**2) < 60**2
img[circle] = 150
img += 5 * np.random.randn(size, size)
img = np.clip(img, 0, 255).astype(np.uint8)

# Apply histogram equalization
equalized = histogram_equalization(img)

# Gamma correction examples
gamma_low = np.clip(255 * (img / 255.0)**0.5, 0, 255).astype(np.uint8)   # Brighten
gamma_high = np.clip(255 * (img / 255.0)**2.0, 0, 255).astype(np.uint8)  # Darken

# Plot
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Images
axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original (low contrast)')

axes[0, 1].imshow(equalized, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('Histogram Equalized')

axes[0, 2].imshow(gamma_low, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title('Gamma = 0.5 (brighten)')

axes[0, 3].imshow(gamma_high, cmap='gray', vmin=0, vmax=255)
axes[0, 3].set_title('Gamma = 2.0 (darken)')

# Histograms
axes[1, 0].hist(img.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.7)
axes[1, 0].set_title('Original Histogram')
axes[1, 0].set_xlim([0, 255])

axes[1, 1].hist(equalized.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
axes[1, 1].set_title('Equalized Histogram')
axes[1, 1].set_xlim([0, 255])

axes[1, 2].hist(gamma_low.ravel(), bins=256, range=(0, 255), color='orange', alpha=0.7)
axes[1, 2].set_title('Gamma 0.5 Histogram')
axes[1, 2].set_xlim([0, 255])

axes[1, 3].hist(gamma_high.ravel(), bins=256, range=(0, 255), color='red', alpha=0.7)
axes[1, 3].set_title('Gamma 2.0 Histogram')
axes[1, 3].set_xlim([0, 255])

for ax in axes[0]:
    ax.axis('off')

plt.tight_layout()
plt.savefig('histogram_equalization.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 15.6 DCT-Based Compression Demo

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn


def jpeg_compress_block(block, quality=50):
    """
    Simulate JPEG compression on a single 8x8 block.

    Parameters
    ----------
    block : ndarray
        8x8 pixel block
    quality : int
        Quality factor (1-100)

    Returns
    -------
    reconstructed : ndarray
        Reconstructed block after compression
    n_nonzero : int
        Number of non-zero quantized coefficients
    """
    # Standard JPEG luminance quantization table
    Q_base = np.array([
        [16, 11, 10, 16,  24,  40,  51,  61],
        [12, 12, 14, 19,  26,  58,  60,  55],
        [14, 13, 16, 24,  40,  57,  69,  56],
        [14, 17, 22, 29,  51,  87,  80,  62],
        [18, 22, 37, 56,  68, 109, 103,  77],
        [24, 35, 55, 64,  81, 104, 113,  92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103,  99]
    ])

    # Scale quantization table by quality
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    Q = np.clip(np.round(Q_base * scale / 100), 1, 255)

    # Shift to center around 0
    block_shifted = block.astype(float) - 128

    # Forward DCT
    dct_block = dctn(block_shifted, type=2, norm='ortho')

    # Quantize
    quantized = np.round(dct_block / Q)
    n_nonzero = np.count_nonzero(quantized)

    # Dequantize
    dequantized = quantized * Q

    # Inverse DCT
    reconstructed = idctn(dequantized, type=2, norm='ortho') + 128

    return np.clip(reconstructed, 0, 255), n_nonzero


def jpeg_compress_image(img, quality=50):
    """Simulate JPEG compression on a full image."""
    M, N = img.shape
    # Pad to multiple of 8
    M_pad = int(np.ceil(M / 8)) * 8
    N_pad = int(np.ceil(N / 8)) * 8
    padded = np.zeros((M_pad, N_pad))
    padded[:M, :N] = img

    result = np.zeros_like(padded)
    total_nonzero = 0

    for i in range(0, M_pad, 8):
        for j in range(0, N_pad, 8):
            block = padded[i:i+8, j:j+8]
            result[i:i+8, j:j+8], nz = jpeg_compress_block(block, quality)
            total_nonzero += nz

    total_coeffs = (M_pad // 8) * (N_pad // 8) * 64
    return result[:M, :N], total_nonzero / total_coeffs


# Create test image
np.random.seed(42)
size = 256
yy, xx = np.mgrid[:size, :size]
img = np.zeros((size, size))
img[40:220, 30:230] = 180
circle = ((xx - 128)**2 + (yy - 128)**2) < 50**2
img[circle] = 240
img += np.sin(2*np.pi*5*xx/size) * 30
img = np.clip(img + 10*np.random.randn(size, size), 0, 255)

# Compress at different quality levels
qualities = [5, 20, 50, 90]
fig, axes = plt.subplots(2, len(qualities)+1, figsize=(18, 8))

axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# Show DCT of one block
block = img[40:48, 30:38]
dct_block = dctn(block - 128, type=2, norm='ortho')
axes[1, 0].imshow(np.log1p(np.abs(dct_block)), cmap='hot')
axes[1, 0].set_title('DCT of 8x8 block\n(log magnitude)')
axes[1, 0].axis('off')

for idx, q in enumerate(qualities):
    compressed, ratio = jpeg_compress_image(img, quality=q)
    psnr = 10 * np.log10(255**2 / np.mean((img - compressed)**2))
    error = np.abs(img - compressed)

    axes[0, idx+1].imshow(compressed, cmap='gray', vmin=0, vmax=255)
    axes[0, idx+1].set_title(f'Q={q}\nPSNR={psnr:.1f} dB')
    axes[0, idx+1].axis('off')

    axes[1, idx+1].imshow(error, cmap='hot', vmin=0, vmax=50)
    axes[1, idx+1].set_title(f'Error (non-zero: {ratio:.1%})')
    axes[1, idx+1].axis('off')

plt.suptitle('JPEG Compression Simulation (DCT-based)', fontsize=14)
plt.tight_layout()
plt.savefig('jpeg_compression.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 16. Exercises

### Exercise 1: 2D DFT Properties

(a) Create a $256 \times 256$ image containing a single white rectangle (64x32 pixels) at the center. Compute and display its 2D DFT magnitude spectrum. Explain the pattern you see in terms of the 2D sinc function.

(b) Shift the rectangle 50 pixels to the right. Compute the new magnitude and phase spectra. What changes and what stays the same? Verify the shift theorem.

(c) Rotate the rectangle by 45 degrees and compute the magnitude spectrum. How does the rotation in the spatial domain affect the frequency domain?

(d) Swap the magnitude of one image with the phase of another. Display the results. Which carries more visual information -- magnitude or phase?

### Exercise 2: Smoothing Filter Comparison

Generate a test image containing:
- A region with additive Gaussian noise ($\sigma = 30$)
- A region with salt-and-pepper noise (5% probability)

(a) Apply box filters of size $3 \times 3$, $5 \times 5$, and $7 \times 7$. Measure the PSNR for each.

(b) Apply Gaussian filters with $\sigma = 1, 2, 3$. Compare with the box filters.

(c) Apply median filters of size $3 \times 3$ and $5 \times 5$. Which noise type benefits most from median filtering?

(d) For each filter, compute the edge preservation metric: the ratio of gradient energy in the filtered image to gradient energy in the original clean image.

### Exercise 3: Frequency Domain Design

(a) Design and implement an ideal bandpass filter that passes spatial frequencies between $D_1 = 20$ and $D_2 = 60$ pixels (from center). Apply to a test image and observe the result.

(b) Create an image with periodic vertical stripes (noise) added to a photograph. Design a notch filter to remove the stripes without affecting the underlying image.

(c) Implement a homomorphic filter: take the log of the image, apply a highpass filter in the frequency domain, then exponentiate. This enhances local contrast by normalizing the illumination. Compare with histogram equalization.

### Exercise 4: Edge Detection Comparison

Create a test image with:
- Strong edges (high contrast)
- Weak edges (low contrast)
- Texture regions
- Gaussian noise ($\sigma = 10$)

(a) Apply Sobel, Prewitt, and Scharr operators. Compare the edge maps and gradient magnitudes.

(b) Apply the Laplacian and LoG with $\sigma = 1, 2, 3$. How does $\sigma$ affect the detected edges?

(c) Implement the full Canny edge detector (all 5 steps). Vary the thresholds and Gaussian $\sigma$ to see their effects.

(d) Quantitatively compare the detectors using precision-recall curves against a ground-truth edge map.

### Exercise 5: Histogram Processing

(a) Generate a dark image (histogram concentrated in [0, 50]) and a bright image (histogram in [200, 255]). Apply histogram equalization to both and compare.

(b) Implement CLAHE: divide the image into $8 \times 8$ tiles, compute the clipped histogram for each tile, and use bilinear interpolation for smooth transitions. Compare with global histogram equalization.

(c) Implement histogram matching (specification): given a target histogram shape, transform the image so its histogram approximates the target.

### Exercise 6: JPEG Compression Analysis

(a) Implement a complete JPEG compression simulator: color space conversion (for color images), $8 \times 8$ block DCT, quantization, zigzag scan, and RLE. Measure the compression ratio and PSNR at quality factors 10, 30, 50, 70, and 90.

(b) Plot the PSNR vs compression ratio curve (rate-distortion curve).

(c) Visualize the blocking artifacts at low quality. Implement a simple deblocking filter (lowpass filter applied only across block boundaries).

(d) Compare DCT compression with wavelet compression: apply the 2D DWT, threshold small coefficients, and reconstruct. Plot the rate-distortion curves for both methods.

### Exercise 7: 2D Sampling and Aliasing

(a) Create a high-resolution ($512 \times 512$) image with a zone plate pattern: $f[m,n] = \cos(\alpha(m^2 + n^2))$. Downsample by factors of 2, 4, and 8 (with and without anti-aliasing). Identify the aliasing artifacts.

(b) Implement bilinear and bicubic interpolation from scratch. Upsample a small ($32 \times 32$) image by a factor of 8 using nearest-neighbor, bilinear, and bicubic methods. Compare the results.

(c) Given a natural image, compute its 2D power spectrum and estimate the effective bandwidth. Determine the minimum sampling rate needed to avoid aliasing.

---

## 17. Summary

| Concept | Key Formula / Idea |
|---|---|
| 2D DFT | $F[k,l] = \sum_m\sum_n f[m,n]e^{-j2\pi(km/M + ln/N)}$ |
| 2D convolution | $g = f * h$ (spatial) $\Leftrightarrow$ $G = F \cdot H$ (frequency) |
| Gaussian filter | $h[m,n] = \frac{1}{2\pi\sigma^2}e^{-(m^2+n^2)/(2\sigma^2)}$; separable |
| Sobel operator | Smoothed gradient: $|\nabla f| = \sqrt{G_x^2 + G_y^2}$ |
| Laplacian | $\nabla^2 f$; detect edges as zero crossings |
| LoG | $\nabla^2(G_\sigma * f) = (\nabla^2 G_\sigma) * f$ |
| Canny | Smooth $\to$ Gradient $\to$ NMS $\to$ Hysteresis |
| Histogram equalization | $s_k = (L-1)\sum_{j=0}^k p[j]$ (CDF mapping) |
| DCT (JPEG) | Real-valued, better energy compaction than DFT |
| JPEG quantization | $F_Q = \text{round}(F/Q)$ (lossy step) |
| 2D Nyquist | $1/\Delta x > 2u_{max}$, $1/\Delta y > 2v_{max}$ |

**Key takeaways**:
1. Images are 2D signals; all 1D signal processing concepts extend naturally to 2D.
2. The 2D DFT is separable, enabling efficient computation via row-column FFTs.
3. Spatial filtering (convolution with kernels) handles smoothing, sharpening, and edge detection.
4. Frequency domain filtering is efficient for large kernels and provides intuitive design (lowpass, highpass, notch).
5. Edge detection combines differentiation (gradient or Laplacian) with noise suppression (Gaussian smoothing).
6. The Canny edge detector optimally balances detection accuracy, localization, and single-response criteria.
7. Histogram equalization is a powerful, automatic contrast enhancement technique.
8. JPEG compression exploits the DCT's energy compaction and human visual insensitivity to high-frequency detail.
9. Anti-aliasing (lowpass before downsampling) is critical for artifact-free image resizing.

---

## 18. References

1. R.C. Gonzalez and R.E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018.
2. A.K. Jain, *Fundamentals of Digital Image Processing*, Prentice Hall, 1989.
3. W.K. Pratt, *Digital Image Processing*, 4th ed., Wiley, 2007.
4. J.F. Canny, "A computational approach to edge detection," *IEEE Trans. PAMI*, vol. 8, no. 6, pp. 679-698, 1986.
5. G.K. Wallace, "The JPEG still picture compression standard," *IEEE Trans. Consumer Electronics*, vol. 38, no. 1, pp. xviii-xxxiv, 1992.
6. A.V. Oppenheim and R.W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010 (Chapter 8: The Discrete Fourier Transform).

---

**Previous**: [14. Time-Frequency Analysis](./14_Time_Frequency_Analysis.md) | **Next**: [16. Applications](./16_Applications.md)
