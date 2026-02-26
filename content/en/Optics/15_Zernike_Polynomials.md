# 15. Zernike Polynomials

[← Previous: 14. Computational Optics](14_Computational_Optics.md) | [Next: 16. Adaptive Optics →](16_Adaptive_Optics.md)

---

When an optical system fails to produce a perfect image, the cause is almost always a distorted wavefront. Whether the culprit is a slightly misaligned telescope mirror, a lens with manufacturing imperfections, or the turbulent atmosphere above an observatory, the resulting wavefront error — the deviation from an ideal sphere — degrades image quality. To diagnose and correct these errors, we need a precise mathematical language for describing wavefront shapes over a circular aperture.

Zernike polynomials, introduced by Frits Zernike in the 1930s (the same physicist who invented phase contrast microscopy), provide exactly this language. They form a complete, orthonormal set of polynomials defined on the unit circle — the natural geometry of most optical apertures. Each Zernike mode corresponds to a recognizable optical aberration (tilt, defocus, coma, astigmatism, spherical aberration), making them far more physically intuitive than other basis sets like Fourier modes. Today, Zernike polynomials are the standard tool for wavefront analysis in optical testing, adaptive optics, ophthalmology, and optical design.

This lesson develops Zernike polynomials from their mathematical definition through practical wavefront analysis and atmospheric turbulence modeling, building on the brief introduction in Lesson 14 (Computational Optics, §5.3).

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Define Zernike polynomials on the unit circle and derive the radial and angular components
2. Apply the Noll indexing convention to enumerate Zernike modes with a single index
3. Prove orthonormality of Zernike polynomials over the unit circle and explain its significance for wavefront decomposition
4. Identify the physical meaning of the first 21 Zernike modes and their effect on point spread functions
5. Perform wavefront decomposition using inner products and least-squares fitting from slope measurements
6. Describe Kolmogorov turbulence statistics and derive the Noll covariance matrix for atmospheric phase screens
7. Calculate RMS wavefront error from Zernike coefficients and estimate Strehl ratio via the Maréchal approximation
8. Implement Zernike mode generation, wavefront fitting, and turbulence phase screen simulation in Python

---

## Table of Contents

1. [Introduction to Wavefront Aberrations](#1-introduction-to-wavefront-aberrations)
2. [Definition of Zernike Polynomials](#2-definition-of-zernike-polynomials)
3. [Noll Indexing Convention](#3-noll-indexing-convention)
4. [Orthogonality and Completeness](#4-orthogonality-and-completeness)
5. [Physical Interpretation of Modes](#5-physical-interpretation-of-modes)
6. [Wavefront Decomposition and Fitting](#6-wavefront-decomposition-and-fitting)
7. [Atmospheric Turbulence and Kolmogorov Statistics](#7-atmospheric-turbulence-and-kolmogorov-statistics)
8. [RMS Wavefront Error and Strehl Ratio](#8-rms-wavefront-error-and-strehl-ratio)
9. [Python Examples](#9-python-examples)
10. [Summary](#10-summary)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. Introduction to Wavefront Aberrations

### 1.1 The Ideal Wavefront

An ideal imaging system converts a point source into a converging spherical wavefront that collapses to a diffraction-limited point at the image plane. The wavefront — the surface of constant optical phase — is a perfect sphere centered on the image point. The resulting image is the Airy pattern, the best that diffraction allows.

### 1.2 Optical Path Difference and Wavefront Error

In practice, every optical element introduces deviations from the ideal sphere. We define the **wavefront error** $W(\rho, \theta)$ as the optical path difference (OPD) between the actual wavefront and the ideal reference sphere, measured in waves or micrometers:

$$W(\rho, \theta) = \text{OPD}(\rho, \theta) = n \cdot \Delta z(\rho, \theta)$$

where $\rho$ and $\theta$ are polar coordinates on the exit pupil (normalized so that $\rho \in [0, 1]$), $n$ is the refractive index, and $\Delta z$ is the physical surface deviation.

### 1.3 Why a Circular Basis?

Most optical systems have circular apertures — telescope primaries, camera lenses, the human eye. A wavefront basis defined on a circle maps naturally to these geometries. While Fourier modes work on rectangular domains and Legendre polynomials on intervals, **Zernike polynomials are the unique set of polynomials that are orthogonal over the unit disk** and separate neatly into radial and angular factors.

> **Analogy**: Just as Fourier series decompose a time signal into sine and cosine harmonics — each with a clear frequency interpretation — Zernike polynomials decompose a wavefront into "aberration harmonics" over a circular aperture. Each mode has a clear optical interpretation: the "fundamental" modes are tilt and defocus, while higher-order modes represent progressively finer wavefront structures like coma and spherical aberration.

---

## 2. Definition of Zernike Polynomials

### 2.1 Radial Polynomials

The Zernike polynomials are defined on the unit circle $\rho \in [0, 1]$, $\theta \in [0, 2\pi)$ as products of a radial part and an angular part. The radial polynomial of order $n$ and azimuthal frequency $m$ is:

$$R_n^{|m|}(\rho) = \sum_{s=0}^{(n-|m|)/2} \frac{(-1)^s (n-s)!}{s! \left(\frac{n+|m|}{2}-s\right)! \left(\frac{n-|m|}{2}-s\right)!} \rho^{n-2s}$$

The indices satisfy two constraints:
- $n \geq 0$ is the radial order
- $|m| \leq n$ and $n - |m|$ is even (so that $(n - |m|)/2$ is a non-negative integer)

For example:
- $R_0^0(\rho) = 1$ (constant)
- $R_1^1(\rho) = \rho$ (linear)
- $R_2^0(\rho) = 2\rho^2 - 1$ (quadratic, related to defocus)
- $R_2^2(\rho) = \rho^2$ (quadratic, related to astigmatism)
- $R_3^1(\rho) = 3\rho^3 - 2\rho$ (cubic, related to coma)
- $R_4^0(\rho) = 6\rho^4 - 6\rho^2 + 1$ (quartic, related to spherical aberration)

### 2.2 Full Zernike Polynomials

The complete Zernike polynomial combines the radial function with a trigonometric angular dependence:

$$Z_n^m(\rho, \theta) = \begin{cases} N_n^m R_n^{|m|}(\rho) \cos(m\theta) & \text{if } m \geq 0 \\ N_n^{|m|} R_n^{|m|}(\rho) \sin(|m|\theta) & \text{if } m < 0 \end{cases}$$

where the normalization factor ensures orthonormality:

$$N_n^m = \sqrt{\frac{2(n+1)}{1 + \delta_{m0}}}$$

Here $\delta_{m0}$ is the Kronecker delta ($\delta_{m0} = 1$ if $m = 0$, else $0$). The factor of $2(n+1)$ comes from the radial integral, and the $(1 + \delta_{m0})$ accounts for the angular average (cosine and sine modes integrate to $\pi$, but the $m = 0$ case integrates to $2\pi$).

### 2.3 Index Ranges and Mode Count

For a given maximum radial order $n_{\max}$, the total number of Zernike modes is:

$$N_{\text{modes}} = \frac{(n_{\max}+1)(n_{\max}+2)}{2}$$

| $n_{\max}$ | Modes | Includes |
|:-----------:|:-----:|----------|
| 1 | 3 | Piston, tip, tilt |
| 2 | 6 | + defocus, astigmatism |
| 3 | 10 | + coma, trefoil |
| 4 | 15 | + spherical, secondary astigmatism, quadrafoil |
| 5 | 21 | + secondary coma, secondary trefoil, pentafoil |
| 6 | 28 | + secondary spherical, ... |

---

## 3. Noll Indexing Convention

### 3.1 The Single-Index Problem

The double-index $(n, m)$ notation is mathematically natural but inconvenient for ordering modes in a vector. Several single-index schemes exist; the most widely used is the **Noll convention** (Noll, 1976), which assigns a single index $j = 1, 2, 3, \ldots$ to each mode.

### 3.2 Noll Ordering Rules

Noll's scheme orders modes by increasing radial order $n$, with the following rules within each order:

1. Even $m$ (cosine) modes come before odd $m$ (sine) modes for the same $|m|$
2. Within the same $n$, modes are ordered by increasing $|m|$
3. For the same $n$ and $|m|$, the cosine term (even $j$) precedes the sine term (odd $j$)

The mapping from $j$ to $(n, m)$ for the first 21 modes:

| $j$ | $n$ | $m$ | Name | Expression |
|:---:|:---:|:---:|------|------------|
| 1 | 0 | 0 | Piston | $1$ |
| 2 | 1 | 1 | Tilt (x) | $2\rho\cos\theta$ |
| 3 | 1 | −1 | Tilt (y) | $2\rho\sin\theta$ |
| 4 | 2 | 0 | Defocus | $\sqrt{3}(2\rho^2 - 1)$ |
| 5 | 2 | −2 | Astigmatism (oblique) | $\sqrt{6}\rho^2\sin 2\theta$ |
| 6 | 2 | 2 | Astigmatism (vertical) | $\sqrt{6}\rho^2\cos 2\theta$ |
| 7 | 3 | −1 | Coma (vertical) | $\sqrt{8}(3\rho^3 - 2\rho)\sin\theta$ |
| 8 | 3 | 1 | Coma (horizontal) | $\sqrt{8}(3\rho^3 - 2\rho)\cos\theta$ |
| 9 | 3 | −3 | Trefoil (vertical) | $\sqrt{8}\rho^3\sin 3\theta$ |
| 10 | 3 | 3 | Trefoil (oblique) | $\sqrt{8}\rho^3\cos 3\theta$ |
| 11 | 4 | 0 | Spherical | $\sqrt{5}(6\rho^4 - 6\rho^2 + 1)$ |
| 12 | 4 | 2 | Secondary astigmatism (v) | $\sqrt{10}(4\rho^4 - 3\rho^2)\cos 2\theta$ |
| 13 | 4 | −2 | Secondary astigmatism (o) | $\sqrt{10}(4\rho^4 - 3\rho^2)\sin 2\theta$ |
| 14 | 4 | 4 | Quadrafoil (v) | $\sqrt{10}\rho^4\cos 4\theta$ |
| 15 | 4 | −4 | Quadrafoil (o) | $\sqrt{10}\rho^4\sin 4\theta$ |
| 16 | 5 | 1 | Secondary coma (h) | $\sqrt{12}(10\rho^5 - 12\rho^3 + 3\rho)\cos\theta$ |
| 17 | 5 | −1 | Secondary coma (v) | $\sqrt{12}(10\rho^5 - 12\rho^3 + 3\rho)\sin\theta$ |
| 18 | 5 | 3 | Secondary trefoil (o) | $\sqrt{12}(5\rho^5 - 4\rho^3)\cos 3\theta$ |
| 19 | 5 | −3 | Secondary trefoil (v) | $\sqrt{12}(5\rho^5 - 4\rho^3)\sin 3\theta$ |
| 20 | 5 | 5 | Pentafoil (h) | $\sqrt{12}\rho^5\cos 5\theta$ |
| 21 | 5 | −5 | Pentafoil (v) | $\sqrt{12}\rho^5\sin 5\theta$ |

### 3.3 Conversion Algorithm

The conversion from Noll index $j$ to $(n, m)$ follows these steps:

1. Find $n$ such that $n(n+1)/2 < j \leq (n+1)(n+2)/2$
2. Compute the position within the row: $k = j - n(n+1)/2$
3. Determine $|m|$ from $k$ and the parity of $n$
4. Assign the sign of $m$ based on whether $j$ is even (cosine, $m > 0$) or odd (sine, $m < 0$) — with the exception that $m = 0$ modes have $j$ even

> **Note**: Other indexing conventions exist. The **ANSI/OSA standard** (Thibos et al., 2002) uses a different ordering where modes are sorted by $|m|$ within each $n$, and negative $m$ precedes positive $m$. Always verify which convention is being used when reading papers or interfacing with optical software.

---

## 4. Orthogonality and Completeness

### 4.1 The Orthonormality Relation

The defining property of Zernike polynomials is their orthonormality over the unit disk:

$$\int_0^1 \int_0^{2\pi} Z_j(\rho, \theta) Z_{j'}(\rho, \theta) \, \rho \, d\rho \, d\theta = \pi \, \delta_{jj'}$$

This means:
- Different modes are orthogonal: their overlap integral is zero
- Each mode is normalized to $\pi$ (the area of the unit disk)

Some references normalize to unity by dividing by $\pi$, giving $\langle Z_j, Z_{j'} \rangle = \delta_{jj'}$.

### 4.2 Proof Sketch

The orthogonality follows from two independent factors:

**Angular orthogonality**: The trigonometric functions satisfy:
$$\int_0^{2\pi} \cos(m\theta)\cos(m'\theta)\,d\theta = \pi(1+\delta_{m0})\delta_{mm'}$$
$$\int_0^{2\pi} \sin(m\theta)\sin(m'\theta)\,d\theta = \pi\delta_{mm'} \quad (m, m' \neq 0)$$
$$\int_0^{2\pi} \cos(m\theta)\sin(m'\theta)\,d\theta = 0$$

**Radial orthogonality**: For polynomials with the same $|m|$:
$$\int_0^1 R_n^{|m|}(\rho) R_{n'}^{|m|}(\rho) \, \rho \, d\rho = \frac{\delta_{nn'}}{2(n+1)}$$

This result is non-trivial and follows from the connection between Zernike radial polynomials and Jacobi polynomials:

$$R_n^{|m|}(\rho) = (-1)^{(n-|m|)/2} P_{(n-|m|)/2}^{(0, |m|)}(1 - 2\rho^2)$$

where $P_k^{(\alpha, \beta)}$ are Jacobi polynomials, which are orthogonal with respect to the weight $(1-x)^\alpha(1+x)^\beta$. The change of variable $x = 1 - 2\rho^2$ converts the radial integral into the standard Jacobi orthogonality relation.

### 4.3 Completeness

The Zernike polynomials form a **complete** basis for square-integrable functions on the unit disk. Any wavefront $W(\rho, \theta)$ with $\int\!\int |W|^2 \rho\,d\rho\,d\theta < \infty$ can be expanded as:

$$W(\rho, \theta) = \sum_{j=1}^{\infty} a_j Z_j(\rho, \theta)$$

In practice, we truncate at some maximum order $j_{\max}$, and the truncation error decreases as $j_{\max}$ increases.

### 4.4 Why Not Just Use Fourier Modes?

Fourier modes (sines and cosines on $x, y$) are orthogonal on a *rectangle*, but when restricted to a circle, they lose orthogonality. This means Fourier coefficients on a circular aperture are coupled — changing one coefficient affects the interpretation of others. Zernike polynomials avoid this problem entirely because they are specifically designed for circular geometry.

| Property | Zernike | Fourier |
|----------|---------|---------|
| Domain | Unit circle | Rectangle |
| Orthogonality on circle | Yes | No |
| Physical interpretation | Each mode = named aberration | No direct aberration meaning |
| Computational cost | $O(j_{\max}^2)$ for all modes | $O(N \log N)$ via FFT |
| Annular apertures | Modified Zernike needed | Still works |

---

## 5. Physical Interpretation of Modes

### 5.1 Low-Order Aberrations

Each Zernike mode corresponds to a classical optical aberration. Understanding these modes is essential for optical engineers, as they connect mathematical coefficients to physical effects on image quality.

**Piston** ($j = 1$, $Z_1 = 1$): A constant phase offset across the aperture. This has no effect on image quality (it shifts the phase of the entire wavefront uniformly) and is usually ignored. However, in interferometry, piston matters because it determines fringe position.

**Tip and Tilt** ($j = 2, 3$): These linear modes shift the image laterally in the focal plane without changing its shape. Tip ($\rho\cos\theta$) shifts horizontally and tilt ($\rho\sin\theta$) shifts vertically. In astronomical observations, atmospheric tip-tilt causes the "dancing" of star images and is the dominant source of image degradation at long exposures.

**Defocus** ($j = 4$, $Z_4 = \sqrt{3}(2\rho^2 - 1)$): A quadratic wavefront curvature that shifts the best focus axially. The image blurs symmetrically — the PSF expands into a uniform disk. Corrected by adjusting the focus position.

**Astigmatism** ($j = 5, 6$): The wavefront has different curvatures along two perpendicular axes, creating a cross-shaped PSF. There is no single best focus — images of horizontal and vertical lines focus at different distances. Common in the human eye.

### 5.2 Third-Order (Seidel) Aberrations

**Coma** ($j = 7, 8$, $n = 3$, $|m| = 1$): A comet-shaped PSF flare, typically caused by off-axis imaging. The wavefront error varies as $\rho^3$, creating an asymmetric image with a bright core and a diffuse tail pointing away from the optical axis.

**Trefoil** ($j = 9, 10$, $n = 3$, $|m| = 3$): A three-fold symmetric aberration that produces a triangular PSF. Less commonly encountered in simple optical systems but important in segmented-mirror telescopes (from segment alignment errors).

**Spherical Aberration** ($j = 11$, $n = 4$, $m = 0$): The most important rotationally symmetric aberration after defocus. Rays at the edge of the aperture focus at a different distance than paraxial rays, creating a halo around the PSF core. This is the aberration that plagued the Hubble Space Telescope before COSTAR corrective optics were installed.

### 5.3 Higher-Order Modes

| Order $n$ | Modes | Physical Significance |
|:---------:|:-----:|----------------------|
| 5 | Secondary coma, secondary trefoil, pentafoil | Fine-structure corrections; important in large telescopes |
| 6 | Secondary spherical, ... | Atmospheric turbulence residuals after low-order AO correction |
| 7–10 | Tertiary aberrations | Relevant for extreme AO systems on 8-30 m telescopes |
| >10 | High-frequency structure | "Scintillation" regime; difficult to correct |

> **Analogy**: Think of Zernike modes as the harmonics of a circular drumhead. The lowest mode ($j = 1$, piston) is the drum at rest. Tip and tilt ($j = 2, 3$) are the drum tilting like a see-saw. Defocus ($j = 4$) is the center bulging up while the edge goes down (or vice versa). Higher modes have increasingly complex nodal patterns — exactly like the vibrational modes you can visualize with sand on a Chladni plate, but on a circular rather than square plate.

### 5.4 Effect on the PSF

The wavefront error $W(\rho, \theta)$ modifies the complex pupil function:

$$P(\rho, \theta) = A(\rho, \theta) \exp\left[\frac{2\pi i}{\lambda} W(\rho, \theta)\right]$$

where $A$ is the aperture amplitude. The PSF is the squared modulus of the Fourier transform of $P$:

$$\text{PSF}(\mathbf{u}) = \left|\mathcal{F}\{P(\rho, \theta)\}\right|^2$$

Each Zernike mode produces a characteristic PSF distortion:

| Mode | PSF Effect |
|------|-----------|
| Defocus | Symmetric broadening (donut at large defocus) |
| Astigmatism | Elongation along one axis; cross-pattern at large amplitudes |
| Coma | Comet-shaped tail; bright core + diffuse fan |
| Trefoil | Triangular symmetry; three-pointed star |
| Spherical | Bright central core + diffuse halo |

---

## 6. Wavefront Decomposition and Fitting

### 6.1 Modal Decomposition via Inner Product

Given a measured wavefront $W(\rho, \theta)$ on the unit disk, the Zernike coefficient for mode $j$ is obtained by projecting onto the basis:

$$a_j = \frac{1}{\pi} \int_0^1 \int_0^{2\pi} W(\rho, \theta) Z_j(\rho, \theta) \, \rho \, d\rho \, d\theta$$

For discrete data on an $N \times N$ grid, this becomes a weighted sum:

$$a_j = \frac{1}{\sum_k w_k} \sum_{k \in \text{pupil}} w_k W_k Z_j(\rho_k, \theta_k)$$

where $w_k$ are quadrature weights (often uniform for equally-spaced pixels within the circular aperture).

### 6.2 Least-Squares Fitting

In practice, wavefront sensors often measure **slopes** (local wavefront gradients) rather than the wavefront itself. The Shack-Hartmann sensor, for example, measures $\partial W / \partial x$ and $\partial W / \partial y$ at a grid of subaperture locations.

The fitting problem becomes: given slope measurements $\mathbf{s} = [s_{x,1}, s_{y,1}, s_{x,2}, s_{y,2}, \ldots]^T$, find the Zernike coefficients $\mathbf{a} = [a_1, a_2, \ldots, a_J]^T$ that minimize:

$$\|\mathbf{s} - \mathbf{D}\mathbf{a}\|^2$$

where $\mathbf{D}$ is the **derivative matrix** with elements:

$$D_{2k-1, j} = \frac{\partial Z_j}{\partial x}\bigg|_{(\rho_k, \theta_k)}, \quad D_{2k, j} = \frac{\partial Z_j}{\partial y}\bigg|_{(\rho_k, \theta_k)}$$

The least-squares solution is:

$$\hat{\mathbf{a}} = (\mathbf{D}^T \mathbf{D})^{-1} \mathbf{D}^T \mathbf{s} = \mathbf{D}^+ \mathbf{s}$$

where $\mathbf{D}^+$ is the Moore-Penrose pseudo-inverse. In practice, singular value decomposition (SVD) is used to compute $\mathbf{D}^+$ robustly, discarding singular values below a noise threshold.

### 6.3 Fitting Error and Mode Selection

The residual wavefront error after fitting $J$ modes is:

$$W_{\text{res}}(\rho, \theta) = W(\rho, \theta) - \sum_{j=1}^{J} a_j Z_j(\rho, \theta)$$

The RMS residual decreases with increasing $J$, but noise amplification also increases (higher modes are more sensitive to measurement noise). The optimal number of modes balances aberration fitting against noise propagation — a classic bias-variance tradeoff.

### 6.4 Zernike Derivatives

The partial derivatives of Zernike polynomials with respect to Cartesian coordinates are needed for slope-based fitting. These can be computed analytically or via recurrence relations. For the Noll-indexed mode $Z_j(\rho, \theta)$:

$$\frac{\partial Z_j}{\partial x} = \frac{\partial Z_j}{\partial \rho}\cos\theta - \frac{1}{\rho}\frac{\partial Z_j}{\partial \theta}\sin\theta$$

$$\frac{\partial Z_j}{\partial y} = \frac{\partial Z_j}{\partial \rho}\sin\theta + \frac{1}{\rho}\frac{\partial Z_j}{\partial \theta}\cos\theta$$

The derivatives of the radial polynomial $R_n^{|m|}$ and trigonometric factors follow standard calculus rules.

---

## 7. Atmospheric Turbulence and Kolmogorov Statistics

### 7.1 The Kolmogorov Model

Atmospheric turbulence is driven by temperature fluctuations that create random variations in the refractive index $n(\mathbf{r})$. Kolmogorov (1941) showed that for scales between the **inner scale** $l_0$ (a few millimeters, where viscous dissipation dominates) and the **outer scale** $L_0$ (tens of meters, the energy injection scale), the refractive index structure function follows a power law:

$$D_n(r) = \langle [n(\mathbf{r}') - n(\mathbf{r}' + \mathbf{r})]^2 \rangle = C_n^2 r^{2/3}$$

where $C_n^2$ is the refractive index structure constant (units: $\text{m}^{-2/3}$), which characterizes the turbulence strength and varies with altitude.

### 7.2 The Fried Parameter

The **Fried parameter** $r_0$ (also called the coherence length) is the aperture diameter at which the atmospheric resolution equals the diffraction limit:

$$r_0 = \left[0.423 k^2 \sec(\gamma) \int_0^{\infty} C_n^2(h) \, dh\right]^{-3/5}$$

where $k = 2\pi/\lambda$ and $\gamma$ is the zenith angle. Typical values:

| Condition | $r_0$ at 500 nm |
|-----------|:---------------:|
| Excellent site (Mauna Kea) | 20–30 cm |
| Good site | 10–15 cm |
| Average site | 5–10 cm |
| Poor seeing | < 5 cm |

The atmospheric resolution (seeing) is $\theta_{\text{seeing}} \approx 0.98 \lambda / r_0$, and a telescope of diameter $D \gg r_0$ has its resolution limited to $\lambda/r_0$ rather than $\lambda/D$.

### 7.3 Wavefront Phase Structure Function

The phase structure function for a plane wave propagating through Kolmogorov turbulence is:

$$D_\phi(r) = 6.88 \left(\frac{r}{r_0}\right)^{5/3}$$

This can be expressed in terms of the phase power spectral density:

$$\Phi_\phi(\kappa) = 0.023 \, r_0^{-5/3} \kappa^{-11/3}$$

where $\kappa$ is the spatial frequency. This $-11/3$ power law is the hallmark of Kolmogorov turbulence.

### 7.4 Noll Covariance Matrix

When the atmospheric wavefront is decomposed into Zernike modes, the covariance between coefficients $a_j$ and $a_{j'}$ is (Noll, 1976):

$$\langle a_j a_{j'} \rangle = K_{jj'} \left(\frac{D}{r_0}\right)^{5/3}$$

where $D$ is the telescope diameter and $K_{jj'}$ is a matrix that depends only on the mode indices. Key properties:

- **Diagonal dominance**: Most of the variance is concentrated in low-order modes (tip-tilt alone accounts for ~87% of the total variance)
- **Off-diagonal coupling**: Modes with the same $|m|$ and same parity of $n$ are correlated
- **Power-law decay**: The variance of mode $j$ decreases roughly as $j^{-11/6}$ for large $j$

The total wavefront variance is:

$$\sigma_\phi^2 = 1.0299 \left(\frac{D}{r_0}\right)^{5/3} \quad [\text{rad}^2]$$

After removing the first $J$ Zernike modes (e.g., through adaptive optics), the residual variance is:

$$\sigma_J^2 \approx 0.2944 \, J^{-\sqrt{3}/2} \left(\frac{D}{r_0}\right)^{5/3} \quad [\text{rad}^2] \quad \text{(for large } J\text{)}$$

### 7.5 Phase Screen Generation

To simulate atmospheric turbulence, we generate random phase screens with the Kolmogorov power spectrum. The **FFT method** works as follows:

1. Generate a grid of complex Gaussian random numbers $\hat{c}(\kappa_x, \kappa_y)$
2. Filter by the square root of the Kolmogorov power spectrum: $\hat{\phi}(\boldsymbol{\kappa}) = \hat{c}(\boldsymbol{\kappa}) \sqrt{\Phi_\phi(\kappa)}$
3. Inverse FFT to obtain $\phi(x, y)$

The resulting phase screen has the correct structure function statistics. For more accurate low-frequency content, **subharmonic addition** can be used (Lane et al., 1992).

---

## 8. RMS Wavefront Error and Strehl Ratio

### 8.1 RMS from Zernike Coefficients

Thanks to orthonormality, the RMS wavefront error has a simple expression in terms of Zernike coefficients:

$$\sigma_W = \sqrt{\frac{1}{\pi}\int\!\!\int_{\text{pupil}} W^2 \rho\,d\rho\,d\theta} = \sqrt{\sum_{j=2}^{J} a_j^2}$$

Note that piston ($j = 1$) is excluded since it does not affect image quality. This is one of the great advantages of the Zernike basis: the RMS is simply the root-sum-square of the coefficients.

### 8.2 The Maréchal Approximation

For small wavefront errors ($\sigma_W \ll \lambda$), the **Strehl ratio** (peak intensity relative to the diffraction limit) is approximated by:

$$S \approx \exp\left[-(2\pi\sigma_W)^2\right] = \exp\left[-\left(\frac{2\pi\sigma_W}{\lambda}\right)^2\right]$$

where $\sigma_W$ is in units of waves. This is valid for $S \gtrsim 0.1$ (i.e., $\sigma_W \lesssim \lambda/4$).

| $\sigma_W$ (waves) | $\sigma_W$ (nm at 550 nm) | Strehl Ratio |
|:---:|:---:|:---:|
| 0 | 0 | 1.000 |
| $\lambda/20$ | 27.5 | 0.905 |
| $\lambda/14$ | 39.3 | 0.815 |
| $\lambda/10$ | 55.0 | 0.674 |
| $\lambda/7$ | 78.6 | 0.444 |
| $\lambda/4$ | 137.5 | 0.081 |

> **The Rayleigh criterion** ($\lambda/4$ peak-to-valley OPD, corresponding to $\sigma_W \approx \lambda/14$ RMS for defocus) gives a Strehl ratio of ~0.8 and is traditionally considered the threshold for a "diffraction-limited" system.

### 8.3 Partial Correction Analysis

If an adaptive optics system corrects the first $J$ Zernike modes, the residual Strehl ratio depends only on the uncorrected modes:

$$S_J \approx \exp\left[-(2\pi)^2 \sum_{j=J+1}^{\infty} \langle a_j^2 \rangle\right]$$

Using the Noll residual variance formula:

| Modes Corrected | Removed Aberrations | Residual Variance (rad²) | Improvement Factor |
|:---:|---|:---:|:---:|
| 1 (piston) | — | $1.030(D/r_0)^{5/3}$ | 1.0× |
| 3 (tip-tilt) | Tip, tilt | $0.134(D/r_0)^{5/3}$ | 7.7× |
| 6 | + defocus, astigmatism | $0.058(D/r_0)^{5/3}$ | 17.8× |
| 10 | + coma, trefoil | $0.034(D/r_0)^{5/3}$ | 30.3× |
| 21 | Through order 5 | $0.016(D/r_0)^{5/3}$ | 64.4× |

This table shows the dramatic improvement from correcting just the first few modes — a key motivation for adaptive optics.

---

## 9. Python Examples

### 9.1 Zernike Mode Generation

```python
import numpy as np

def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index j (starting at 1) to radial order n and
    azimuthal frequency m.

    The Noll convention orders modes by increasing n, with even-j
    assigned to cosine (m >= 0) and odd-j to sine (m < 0) terms.
    """
    # Find radial order n: j falls in the range (n(n+1)/2, (n+1)(n+2)/2]
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    # Position within this order
    k = j - n * (n + 1) // 2
    # Determine |m|
    if n % 2 == 0:
        m_abs = 2 * ((k + 1) // 2)
    else:
        m_abs = 2 * (k // 2) + 1
    # Sign convention: even j -> cosine (m >= 0), odd j -> sine (m < 0)
    if m_abs == 0:
        m = 0
    elif j % 2 == 0:
        m = m_abs
    else:
        m = -m_abs
    return n, m


def zernike_radial(n: int, m_abs: int, rho: np.ndarray) -> np.ndarray:
    """Compute radial Zernike polynomial R_n^|m|(rho).

    Uses the explicit factorial formula. The polynomial is zero outside
    the unit circle.

    Parameters
    ----------
    n : int  — Radial order (>= 0)
    m_abs : int  — Absolute azimuthal frequency (0 <= m_abs <= n, n - m_abs even)
    rho : ndarray  — Radial coordinate(s), 0 <= rho <= 1
    """
    R = np.zeros_like(rho, dtype=float)
    for s in range((n - m_abs) // 2 + 1):
        coeff = ((-1) ** s * np.math.factorial(n - s)
                 / (np.math.factorial(s)
                    * np.math.factorial((n + m_abs) // 2 - s)
                    * np.math.factorial((n - m_abs) // 2 - s)))
        R += coeff * rho ** (n - 2 * s)
    return R


def zernike(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate the Noll-indexed Zernike polynomial Z_j at (rho, theta).

    Returns the normalized polynomial value. Points outside the unit
    circle are set to NaN.
    """
    n, m = noll_to_nm(j)
    m_abs = abs(m)
    # Normalization factor
    if m == 0:
        norm = np.sqrt(n + 1)
    else:
        norm = np.sqrt(2 * (n + 1))
    R = zernike_radial(n, m_abs, rho)
    if m >= 0:
        Z = norm * R * np.cos(m_abs * theta)
    else:
        Z = norm * R * np.sin(m_abs * theta)
    # Mask outside unit circle
    Z[rho > 1.0] = np.nan
    return Z
```

### 9.2 Wavefront Fitting from Discrete Data

```python
def zernike_fit(wavefront: np.ndarray, n_modes: int,
                rho: np.ndarray, theta: np.ndarray,
                mask: np.ndarray) -> np.ndarray:
    """Fit Zernike coefficients to a wavefront on a discrete grid.

    This builds the design matrix [Z_1, Z_2, ..., Z_J] at the valid
    (in-pupil) pixel locations and solves the normal equations via SVD.

    Parameters
    ----------
    wavefront : 2D array  — Measured wavefront (same shape as rho, theta)
    n_modes : int  — Number of Zernike modes to fit (j = 1..n_modes)
    rho, theta : 2D arrays  — Polar coordinates at each pixel
    mask : 2D bool array  — True inside the pupil

    Returns
    -------
    coeffs : 1D array of length n_modes  — Fitted Zernike coefficients
    """
    # Flatten valid pixels
    w = wavefront[mask].ravel()
    r = rho[mask].ravel()
    t = theta[mask].ravel()
    # Build design matrix: each column is Z_j evaluated at valid pixels
    A = np.column_stack([zernike(j, r, t) for j in range(1, n_modes + 1)])
    # Solve via least squares (SVD-based)
    coeffs, _, _, _ = np.linalg.lstsq(A, w, rcond=None)
    return coeffs
```

### 9.3 Kolmogorov Phase Screen (FFT Method)

```python
def kolmogorov_phase_screen(N: int, r0: float, L: float,
                            seed: int | None = None) -> np.ndarray:
    """Generate a Kolmogorov turbulence phase screen using the FFT method.

    The screen has the correct D_phi(r) = 6.88 (r/r0)^(5/3) structure
    function for separations between the grid spacing and the screen size.

    Parameters
    ----------
    N : int  — Grid size (NxN pixels)
    r0 : float  — Fried parameter in physical units (e.g., meters)
    L : float  — Physical side length of the screen (same units as r0)
    seed : int or None  — Random seed for reproducibility

    Returns
    -------
    phi : 2D array (N, N)  — Phase screen in radians
    """
    rng = np.random.default_rng(seed)
    # Spatial frequency grid
    df = 1.0 / L  # frequency spacing
    fx = np.fft.fftfreq(N, d=L / N)
    fy = np.fft.fftfreq(N, d=L / N)
    Fx, Fy = np.meshgrid(fx, fy)
    f_mag = np.sqrt(Fx**2 + Fy**2)
    f_mag[0, 0] = 1.0  # avoid division by zero; DC is set to zero later
    # Kolmogorov PSD: Phi(f) = 0.023 * r0^(-5/3) * (2*pi*f)^(-11/3)
    # In terms of spatial frequency f (not angular frequency kappa):
    psd = 0.023 * r0**(-5.0/3) * (2 * np.pi * f_mag)**(-11.0/3)
    psd[0, 0] = 0.0  # remove piston
    # Generate random complex field weighted by sqrt(PSD)
    # The factor accounts for the discrete FT normalization
    cn = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    cn *= np.sqrt(psd) * (2 * np.pi / L)
    # Inverse FFT to get phase screen
    phi = np.real(np.fft.ifft2(cn)) * N**2
    return phi
```

### 9.4 Strehl Ratio Calculator

```python
def strehl_marechal(rms_waves: float) -> float:
    """Estimate Strehl ratio using the Maréchal approximation.

    S ≈ exp(-(2π σ)²), valid for σ ≲ λ/4 (Strehl ≳ 0.1).

    Parameters
    ----------
    rms_waves : float  — RMS wavefront error in units of waves (λ)

    Returns
    -------
    S : float  — Estimated Strehl ratio (0 to 1)
    """
    return np.exp(-(2 * np.pi * rms_waves)**2)


def rms_from_zernike(coeffs: np.ndarray, exclude_piston: bool = True) -> float:
    """Compute RMS wavefront error from Zernike coefficients.

    Thanks to orthonormality: σ = sqrt(Σ a_j²), excluding piston.
    """
    start = 1 if exclude_piston else 0  # coeffs[0] = a_1 (piston)
    return np.sqrt(np.sum(coeffs[start:]**2))
```

---

## 10. Summary

| Concept | Key Formula / Idea |
|---------|--------------------|
| Zernike polynomial | $Z_n^m(\rho, \theta) = N_n^m R_n^{|m|}(\rho) \times \{\cos, \sin\}(|m|\theta)$ |
| Radial polynomial | $R_n^{|m|}(\rho) = \sum_s \frac{(-1)^s(n-s)!}{s!(\ldots)!(\ldots)!}\rho^{n-2s}$ |
| Normalization | $N_n^m = \sqrt{2(n+1)/(1+\delta_{m0})}$ |
| Orthonormality | $\langle Z_j, Z_{j'} \rangle = \pi\delta_{jj'}$ |
| Mode count to order $n$ | $(n+1)(n+2)/2$ |
| Noll indexing | $j = 1, 2, 3, \ldots$ mapping to $(n, m)$ per Noll (1976) |
| Wavefront decomposition | $a_j = \frac{1}{\pi}\int\!\int W Z_j \rho\,d\rho\,d\theta$ |
| RMS wavefront error | $\sigma_W = \sqrt{\sum_{j=2}^J a_j^2}$ |
| Kolmogorov PSD | $\Phi_\phi(\kappa) = 0.023\,r_0^{-5/3}\kappa^{-11/3}$ |
| Fried parameter | $r_0 = [0.423 k^2 \sec\gamma \int C_n^2\,dh]^{-3/5}$ |
| Maréchal approximation | $S \approx e^{-(2\pi\sigma_W)^2}$ |
| Residual after $J$ modes | $\sigma_J^2 \approx 0.2944\,J^{-\sqrt{3}/2}(D/r_0)^{5/3}$ |

---

## 11. Exercises

### Exercise 1: Radial Polynomial Verification

(a) Show by direct computation that $R_4^0(\rho) = 6\rho^4 - 6\rho^2 + 1$ using the explicit Zernike radial polynomial formula. (b) Verify orthogonality numerically: compute $\int_0^1 R_2^0(\rho) R_4^0(\rho) \rho\,d\rho$ on a fine grid and confirm it equals zero. (c) Repeat for $R_3^1$ and $R_5^1$. (d) Compute $R_6^0(\rho)$ and plot it. What is the physical interpretation of this mode?

### Exercise 2: Wavefront Analysis

A wavefront sensor returns the following Noll Zernike coefficients (in waves): $a_2 = 0.05$, $a_3 = -0.08$, $a_4 = 0.30$, $a_5 = -0.15$, $a_6 = 0.10$, $a_7 = 0.12$, $a_8 = -0.06$, $a_{11} = 0.20$. (a) Compute the total RMS wavefront error. (b) Which single mode contributes most to the RMS? (c) Estimate the Strehl ratio using the Maréchal approximation. (d) If you could perfectly correct the three worst modes, what would the new Strehl ratio be? (e) Plot the wavefront error map and the corresponding PSF (hint: use FFT of the pupil function).

### Exercise 3: Atmospheric Phase Screen

Generate a Kolmogorov phase screen for a 4-meter telescope with $r_0 = 15$ cm at $\lambda = 500$ nm on a 512×512 grid. (a) Compute and plot the phase structure function $D_\phi(r)$ from the generated screen and compare with the theoretical $6.88(r/r_0)^{5/3}$. (b) Fit the first 36 Zernike modes and display the coefficient magnitudes. (c) Verify that tip-tilt modes dominate the total variance. (d) Compare the residual variance after removing 10 and 36 modes with the Noll formula.

### Exercise 4: Annular Aperture Extension

Many telescopes have a central obscuration (secondary mirror). (a) On a grid representing an annular aperture with obscuration ratio $\epsilon = 0.3$, evaluate the first 15 standard Zernike modes and verify that they are **not** orthogonal (compute the Gram matrix $G_{jj'} = \langle Z_j, Z_{j'} \rangle_\text{annulus}$). (b) Use Gram-Schmidt orthogonalization to construct the first 10 annular Zernike polynomials. (c) Decompose a synthetic wavefront (coma + spherical) on both the standard and annular bases. How do the coefficients differ? (d) Discuss when annular Zernike polynomials are necessary versus when standard Zernike polynomials on the full circle are a sufficient approximation.

---

## 12. References

1. Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence." *Journal of the Optical Society of America*, 66(3), 207–211. — The standard reference for Noll indexing and atmospheric Zernike statistics.
2. Born, M., & Wolf, E. (2019). *Principles of Optics* (7th expanded ed.). Cambridge University Press. — Chapter 9 on aberrations; Appendix VII on Zernike polynomials.
3. Mahajan, V. N. (2013). *Optical Imaging and Aberrations, Part III: Wavefront Analysis*. SPIE Press. — Comprehensive treatment of Zernike-based wavefront analysis.
4. Hardy, J. W. (1998). *Adaptive Optics for Astronomical Telescopes*. Oxford University Press. — Chapters 3–4 on atmospheric turbulence and Zernike decomposition.
5. Roddier, F. (Ed.) (1999). *Adaptive Optics in Astronomy*. Cambridge University Press. — Chapter 2 on atmospheric turbulence theory.
6. Thibos, L. N., Applegate, R. A., Schwiegerling, J. T., & Webb, R. (2002). "Standards for reporting the optical aberrations of eyes." *Journal of Refractive Surgery*, 18(5), S652–S660. — ANSI/OSA Zernike standard for ophthalmology.
7. Lane, R. G., Glindemann, A., & Dainty, J. C. (1992). "Simulation of a Kolmogorov phase screen." *Waves in Random Media*, 2(3), 209–224. — FFT-based phase screen generation with subharmonics.

---

[← Previous: 14. Computational Optics](14_Computational_Optics.md) | [Next: 16. Adaptive Optics →](16_Adaptive_Optics.md)
