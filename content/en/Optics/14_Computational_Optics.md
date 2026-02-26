# 14. Computational Optics

[← Previous: 13. Quantum Optics Primer](13_Quantum_Optics_Primer.md) | [Next: 15. Zernike Polynomials →](15_Zernike_Polynomials.md)

---

Throughout this course, we have studied how light propagates, interferes, diffracts, and interacts with matter — and we have largely relied on analytical formulas to describe these phenomena. But real optical systems are far too complex for closed-form solutions: a camera lens has ten or more surfaces with aspherical profiles, a photonic crystal waveguide has subwavelength features that scatter light in intricate patterns, and adaptive optics systems must correct for atmospheric turbulence in real time. **Computational optics** is the discipline that bridges the gap between optical theory and practical engineering by harnessing numerical methods to design, simulate, and even replace traditional optical components.

The field has two complementary faces. On one side, computers simulate light propagation through optical systems — from simple ray tracing through complex lenses to full-wave Maxwell solvers for nanophotonic devices. On the other side, computation replaces physical optics: phase retrieval algorithms recover information that detectors cannot measure directly, computational photography techniques synthesize images that no single exposure could capture, and machine learning models design optical elements that human intuition would never conceive. Together, these approaches have transformed optics from a craft of polishing glass into a computational science.

This lesson surveys the major computational methods in optics, from classical ray tracing to modern machine-learning-driven inverse design, providing both the theoretical foundations and practical implementations.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Implement sequential ray tracing through multi-surface optical systems using Snell's law and the ABCD transfer matrix method
2. Derive the beam propagation method (BPM) from the paraxial wave equation and implement the split-step Fourier algorithm
3. Explain the FDTD method for solving Maxwell's equations on a Yee grid, including stability conditions and absorbing boundaries
4. Implement the Gerchberg-Saxton algorithm for iterative phase retrieval and explain the transport of intensity equation (TIE)
5. Describe wavefront sensing with Shack-Hartmann sensors and characterize aberrations using Zernike polynomials
6. Explain computational photography techniques including light field imaging, HDR, and coded apertures
7. Describe digital holography reconstruction using the angular spectrum method and phase unwrapping
8. Identify applications of machine learning in optics, including learned phase retrieval, super-resolution, and inverse design

---

## Table of Contents

1. [Geometric Ray Tracing](#1-geometric-ray-tracing)
2. [Beam Propagation Method (BPM)](#2-beam-propagation-method-bpm)
3. [FDTD for Optics](#3-fdtd-for-optics)
4. [Phase Retrieval](#4-phase-retrieval)
5. [Wavefront Sensing](#5-wavefront-sensing)
6. [Computational Photography](#6-computational-photography)
7. [Digital Holography](#7-digital-holography)
8. [Machine Learning in Optics](#8-machine-learning-in-optics)
9. [Python Examples](#9-python-examples)
10. [Summary](#10-summary)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. Geometric Ray Tracing

### 1.1 The Ray Tracing Paradigm

Ray tracing is the oldest and most widely used computational method in optical design. It treats light as a collection of rays that travel in straight lines through homogeneous media and refract at interfaces according to Snell's law. Despite ignoring diffraction and interference, geometric ray tracing remains the workhorse of lens design — a modern camera lens is designed almost entirely by tracing millions of rays through its surfaces.

> **Analogy**: Ray tracing is like planning a road trip through a series of countries with different speed limits. At each border (optical surface), you recalculate your direction based on the local "speed of light" (refractive index). The ABCD matrix method is like having a GPS that multiplies a single transfer matrix for each leg of the journey — at the end, one matrix multiplication tells you where you end up and at what angle, without retracing the route step by step.

### 1.2 Sequential Ray Tracing

In a sequential optical system (light passes through surfaces in a fixed order), each ray is defined by its height $y$ above the optical axis and its angle $\theta$ (or slope $u = \tan\theta \approx \theta$ in the paraxial limit) at each surface.

At a refracting surface with radius of curvature $R$, separating media with refractive indices $n_1$ and $n_2$, Snell's law in vector form gives:

$$n_2 \hat{\mathbf{s}}_2 = n_1 \hat{\mathbf{s}}_1 + (n_2\cos\theta_2 - n_1\cos\theta_1)\hat{\mathbf{n}}$$

where $\hat{\mathbf{s}}_1, \hat{\mathbf{s}}_2$ are unit direction vectors of the incident and refracted rays, $\hat{\mathbf{n}}$ is the surface normal, and the angles are measured from the normal. For exact (non-paraxial) tracing, you compute the ray-surface intersection, find the local normal, and apply Snell's law without any small-angle approximation.

### 1.3 The ABCD Matrix Method

In the **paraxial approximation** ($\sin\theta \approx \theta$), the relationship between the ray parameters $(y, u)$ at input and output planes is linear:

$$\begin{pmatrix} y_{\text{out}} \\ u_{\text{out}} \end{pmatrix} = \begin{pmatrix} A & B \\ C & D \end{pmatrix} \begin{pmatrix} y_{\text{in}} \\ u_{\text{in}} \end{pmatrix}$$

The key matrices (already introduced in Lesson 8 for Gaussian beams) are:

**Free-space propagation** (distance $d$ in medium of index $n$):

$$M_{\text{prop}} = \begin{pmatrix} 1 & d/n \\ 0 & 1 \end{pmatrix}$$

**Refraction at a spherical surface** (radius $R$, from $n_1$ to $n_2$):

$$M_{\text{refr}} = \begin{pmatrix} 1 & 0 \\ -(n_2 - n_1)/R & 1 \end{pmatrix}$$

Note: this uses the sign convention where $R > 0$ means the center of curvature is to the right.

**Thin lens** (focal length $f$):

$$M_{\text{lens}} = \begin{pmatrix} 1 & 0 \\ -1/f & 1 \end{pmatrix}$$

For a complete system, multiply the matrices in reverse order (right to left):

$$M_{\text{sys}} = M_N \cdot M_{N-1} \cdots M_2 \cdot M_1$$

The system matrix encodes everything about paraxial imaging: the effective focal length is $f = -1/C$, and the image position can be found from the condition $B = 0$.

### 1.4 Spot Diagrams and Ray Fans

A **spot diagram** shows where a bundle of rays from a single object point lands on the image plane. For a perfect lens, all rays converge to a single point; in reality, aberrations spread them out. The shape of the spot reveals the dominant aberration:
- Circular spread → spherical aberration
- Comet-shaped → coma
- Cross-shaped → astigmatism

A **ray fan** (or ray aberration plot) plots the transverse ray error $\Delta y$ as a function of entrance pupil height $h$. Different curves reveal different Seidel aberrations:
- $\Delta y \propto h^3$ → spherical aberration
- $\Delta y \propto h^2$ → coma
- $\Delta y \propto h$ → defocus or field curvature

### 1.5 Beyond Paraxial: Real Ray Tracing

For real lens design, the paraxial approximation is insufficient. Software like Zemax, Code V, and OpticStudio traces exact rays using the full form of Snell's law. Each ray-surface intersection is found iteratively (Newton's method for aspherical surfaces), and the refraction is computed without any small-angle approximation. Millions of rays are traced to build a statistical picture of image quality.

---

## 2. Beam Propagation Method (BPM)

### 2.1 When Rays Are Not Enough

Ray tracing fails when diffraction effects are important — when feature sizes approach the wavelength, or when light propagates over long distances in guiding structures (optical fibers, integrated waveguides). The **Beam Propagation Method (BPM)** solves the paraxial wave equation numerically, capturing both diffraction and the effect of spatially varying refractive index.

### 2.2 The Paraxial Wave Equation

Starting from the Helmholtz equation for a monochromatic field $U(\mathbf{r})e^{-i\omega t}$:

$$\nabla^2 U + k_0^2 n^2(x, y, z) U = 0$$

We write $U = \psi(x, y, z) e^{ikz}$, where $\psi$ is a slowly varying envelope propagating along $z$ and $k = k_0 n_0$ is the reference wavenumber ($n_0$ is the background refractive index). Substituting and neglecting $\partial^2\psi/\partial z^2$ (the **paraxial** or **slowly varying envelope** approximation):

$$\frac{\partial \psi}{\partial z} = \frac{i}{2k}\nabla_\perp^2 \psi + \frac{ik_0}{2n_0}\Delta n^2(x,y,z)\,\psi$$

where $\nabla_\perp^2 = \partial^2/\partial x^2 + \partial^2/\partial y^2$ is the transverse Laplacian and $\Delta n^2 = n^2 - n_0^2 \approx 2n_0\Delta n$ for small index perturbations.

This equation has the form:

$$\frac{\partial\psi}{\partial z} = (i\hat{D} + i\hat{N})\psi$$

where $\hat{D} = \nabla_\perp^2/(2k)$ is the **diffraction operator** and $\hat{N} = k_0\Delta n^2/(2n_0)$ is the **refractive index operator**.

> **Analogy**: The BPM is like simulating a ball rolling on a tilted, bumpy surface. The diffraction operator $\hat{D}$ handles the ball's natural tendency to spread out (like a wave diffracting), while the refractive index operator $\hat{N}$ handles the bumps and valleys that steer the ball (like a waveguide confining light). The split-step method alternates between handling each effect separately — first let the ball spread freely for a tiny step, then adjust for the bumps, then repeat.

### 2.3 Split-Step Fourier Method

The split-step method advances $\psi$ by a small step $\Delta z$ by alternating between the two operators:

$$\psi(z + \Delta z) \approx e^{i\hat{N}\Delta z}\,\mathcal{F}^{-1}\!\left\{e^{i\hat{D}_k \Delta z}\,\mathcal{F}\{\psi(z)\}\right\}$$

where $\hat{D}_k = -(k_x^2 + k_y^2)/(2k)$ is the diffraction operator in the spatial frequency domain. The algorithm is:

1. **Forward FFT**: Transform $\psi(x,y)$ to the spatial frequency domain $\tilde{\psi}(k_x, k_y)$
2. **Diffraction step**: Multiply by $\exp\!\left[-i(k_x^2 + k_y^2)\Delta z/(2k)\right]$
3. **Inverse FFT**: Transform back to spatial domain
4. **Refraction step**: Multiply by $\exp\!\left[ik_0\Delta n(x,y,z)\Delta z / n_0\right]$ (using $\Delta n^2 \approx 2n_0\Delta n$)
5. Repeat for each $z$-step

The key insight is that the diffraction operator is diagonal in the Fourier domain (each spatial frequency propagates independently in free space), while the refractive index operator is diagonal in real space (it acts locally at each point).

### 2.4 Angular Spectrum Propagation

A closely related method — the **angular spectrum method** — propagates an arbitrary field $U(x,y,z_0)$ to a new plane $z_0 + d$ without the paraxial approximation:

$$U(x, y, z_0 + d) = \mathcal{F}^{-1}\!\left\{\tilde{U}(k_x, k_y, z_0) \cdot e^{ik_z d}\right\}$$

where $k_z = \sqrt{k^2 - k_x^2 - k_y^2}$ (for propagating waves; evanescent for $k_x^2 + k_y^2 > k^2$). This is exact within scalar diffraction theory and is the foundation for numerical propagation in digital holography (Section 7).

### 2.5 Applications

BPM is the standard tool for simulating:
- **Optical fiber modes and propagation** in graded-index and photonic crystal fibers
- **Integrated photonic waveguides**: directional couplers, Y-junctions, ring resonators
- **Laser beam propagation** through turbulent atmosphere
- **Graded-index (GRIN) lenses** with continuously varying refractive index

---

## 3. FDTD for Optics

### 3.1 Full-Wave Simulation

When the paraxial approximation breaks down — at subwavelength features, sharp bends, or metallic nanostructures — we need to solve Maxwell's equations directly. The **Finite-Difference Time-Domain (FDTD)** method, introduced by Kane Yee in 1966, does exactly this: it discretizes Maxwell's curl equations on a grid and steps forward in time.

### 3.2 Maxwell's Curl Equations

In source-free linear media:

$$\frac{\partial \mathbf{H}}{\partial t} = -\frac{1}{\mu}\nabla \times \mathbf{E}$$

$$\frac{\partial \mathbf{E}}{\partial t} = \frac{1}{\epsilon}\nabla \times \mathbf{H}$$

FDTD discretizes these equations using central differences in both space and time.

### 3.3 The Yee Grid

Yee's key insight was to stagger the $\mathbf{E}$ and $\mathbf{H}$ field components in both space and time:

- $E_x, E_y, E_z$ and $H_x, H_y, H_z$ are positioned at different points on the unit cell
- $\mathbf{E}$ is updated at integer time steps, $\mathbf{H}$ at half-integer time steps

This staggering naturally satisfies Gauss's laws ($\nabla \cdot \mathbf{E} = 0$, $\nabla \cdot \mathbf{B} = 0$) and provides second-order accuracy in both space and time. In 1D, the update equations for $E_x$ and $H_y$ are:

$$H_y^{n+1/2}(i+\tfrac{1}{2}) = H_y^{n-1/2}(i+\tfrac{1}{2}) - \frac{\Delta t}{\mu\Delta x}\left[E_x^n(i+1) - E_x^n(i)\right]$$

$$E_x^{n+1}(i) = E_x^n(i) + \frac{\Delta t}{\epsilon(i)\Delta x}\left[H_y^{n+1/2}(i+\tfrac{1}{2}) - H_y^{n+1/2}(i-\tfrac{1}{2})\right]$$

where superscripts denote time steps and arguments denote spatial grid points.

### 3.4 Stability: The Courant Condition

The time step $\Delta t$ must satisfy the **Courant-Friedrichs-Lewy (CFL)** condition for numerical stability:

$$c\Delta t \leq \frac{1}{\sqrt{1/\Delta x^2 + 1/\Delta y^2 + 1/\Delta z^2}}$$

In 1D, this simplifies to $c\Delta t \leq \Delta x$. In 3D with uniform grid spacing $\Delta$: $c\Delta t \leq \Delta/\sqrt{3}$.

The spatial grid must also resolve the shortest wavelength in the simulation. A common rule of thumb is $\Delta \leq \lambda_{\min}/20$ for accurate results in dielectric structures, and $\Delta \leq \lambda_{\min}/40$ near metallic surfaces where fields vary rapidly.

### 3.5 Absorbing Boundary Conditions: PML

FDTD simulations require finite computational domains, but outgoing waves should not reflect back from the boundaries. **Perfectly Matched Layers (PML)**, introduced by Berenger (1994), surround the computational domain with an artificial absorbing medium that absorbs waves at all angles of incidence without reflection. The PML works by analytically continuing the spatial coordinates into the complex plane:

$$\tilde{x} = \int_0^x s_x(x')\,dx', \quad s_x(x') = 1 + \frac{\sigma_x(x')}{i\omega\epsilon_0}$$

where $\sigma_x$ is a conductivity profile that increases from zero at the PML boundary to a maximum value deep inside the PML. A polynomial profile $\sigma_x(x) = \sigma_{\max}(x/d)^m$ with $m = 3\text{-}4$ works well in practice.

### 3.6 Applications in Photonics

FDTD is the tool of choice for:
- **Photonic crystals**: Band structure calculation, defect modes, slow light
- **Plasmonic nanostructures**: Field enhancement near metallic nanoparticles, nanoantennas
- **Metamaterials**: Negative index, cloaking, perfect absorbers
- **Nanophotonic devices**: Waveguide bends, grating couplers, resonators
- **Solar cells**: Light trapping in nanostructured thin films

Popular FDTD software includes Lumerical FDTD, Meep (open source, from MIT), and COMSOL (frequency-domain FEM, but often compared with FDTD).

> **Analogy**: If BPM is like watching a river flow downstream (only forward propagation), FDTD is like watching the entire ocean — waves can go in any direction, reflect, scatter, and interfere, and you capture every detail of the time evolution. The price is computational cost: FDTD requires discretizing all three spatial dimensions plus time, while BPM only steps along one axis.

---

## 4. Phase Retrieval

### 4.1 The Phase Problem

Optical detectors — cameras, CCDs, photodiodes — measure **intensity** $I = |U|^2$. The **phase** $\phi$ of the complex field $U = |U|e^{i\phi}$ is lost. Yet phase carries critical information: it encodes wavefront shape, depth, and refractive index variations. Recovering phase from intensity measurements is the **phase problem**, one of the most important inverse problems in optics.

The phase problem arises in:
- **Astronomy**: Measuring aberrations of telescope optics
- **Electron microscopy**: Imaging atomic-scale structures
- **X-ray crystallography**: Determining molecular structure (where it earned several Nobel Prizes)
- **Adaptive optics**: Correcting atmospheric turbulence
- **Coherent diffractive imaging**: Imaging without lenses at nanometer resolution

### 4.2 The Gerchberg-Saxton Algorithm

The **Gerchberg-Saxton (GS) algorithm** (1972) is the foundational iterative phase retrieval method. It recovers the phase of a wavefront given two intensity measurements: one in the object plane ($I_{\text{obj}} = |U_{\text{obj}}|^2$) and one in the far field or Fourier plane ($I_{\text{far}} = |\tilde{U}|^2$).

**Algorithm**:

1. Start with a random phase guess: $U_{\text{obj}}^{(0)} = \sqrt{I_{\text{obj}}}\,e^{i\phi_{\text{random}}}$
2. Propagate to the Fourier plane: $\tilde{U}^{(k)} = \mathcal{F}\{U_{\text{obj}}^{(k)}\}$
3. **Apply Fourier constraint**: Replace amplitude with measured $\sqrt{I_{\text{far}}}$, keep phase:
   $$\tilde{U}_{\text{corrected}}^{(k)} = \sqrt{I_{\text{far}}}\,\frac{\tilde{U}^{(k)}}{|\tilde{U}^{(k)}|}$$
4. Propagate back: $U_{\text{obj}}^{(k)} = \mathcal{F}^{-1}\{\tilde{U}_{\text{corrected}}^{(k)}\}$
5. **Apply object constraint**: Replace amplitude with measured $\sqrt{I_{\text{obj}}}$, keep phase:
   $$U_{\text{obj}}^{(k+1)} = \sqrt{I_{\text{obj}}}\,\frac{U_{\text{obj}}^{(k)}}{|U_{\text{obj}}^{(k)}|}$$
6. Repeat until convergence (the phase stabilizes)

The GS algorithm is an example of **alternating projections**: each step projects the current estimate onto the set of fields consistent with one measurement. It converges monotonically (the error never increases), but may stagnate at local minima.

> **Analogy**: Imagine you know the shadow of an object from two different angles, but you need to reconstruct the 3D shape. You start with a guess, adjust it to match shadow #1, then adjust to match shadow #2, and keep alternating. Each adjustment brings you closer to the true shape. The Gerchberg-Saxton algorithm does exactly this, with the two "shadows" being the intensity patterns in two different planes (object and Fourier).

### 4.3 Transport of Intensity Equation (TIE)

A non-iterative alternative is the **Transport of Intensity Equation**. It relates the axial derivative of intensity to the phase:

$$-k\frac{\partial I}{\partial z} = \nabla_\perp \cdot (I \nabla_\perp \phi)$$

where $I(x,y)$ is the intensity and $\phi(x,y)$ is the phase at a given plane. By measuring the intensity at three closely spaced planes ($z - \delta z$, $z$, $z + \delta z$), we estimate $\partial I/\partial z$ by finite differences and solve the resulting Poisson-like equation for $\phi$.

**Advantages**: Deterministic (no iteration), unique solution (for simply connected domains), works with partially coherent light.
**Limitations**: Requires very precise intensity measurements, sensitive to noise, assumes small propagation distances.

### 4.4 Modern Extensions

- **Hybrid Input-Output (HIO)**: Fienup's modification of GS that provides faster convergence by applying a feedback parameter $\beta$ to the object constraint
- **Ptychography**: Overlapping scanning diffraction patterns provide redundant information, making phase retrieval much more robust and uniquely determined
- **Single-shot phase retrieval**: Using structured illumination or coded apertures to recover phase from a single intensity measurement

---

## 5. Wavefront Sensing

### 5.1 Why Measure Wavefronts?

A perfect optical system produces a flat (or perfectly spherical) wavefront. Aberrations — from atmospheric turbulence, manufacturing errors, or thermal effects — distort the wavefront, degrading image quality. Wavefront sensing measures these distortions so they can be corrected (adaptive optics) or characterized (optical testing).

### 5.2 Shack-Hartmann Wavefront Sensor

The **Shack-Hartmann sensor** is the most widely used wavefront sensor. It consists of a microlens array placed in front of a camera. Each microlens samples a small patch of the incoming wavefront and focuses it to a spot on the detector.

```
Incoming wavefront (aberrated)
  ↓   ↓   ↓   ↓   ↓   ↓
┌───┬───┬───┬───┬───┬───┐  ← Microlens array
│ · │ · │ · │ · │ · │ · │
│  ·│·  │ · │·  │  ·│ · │  ← Focal spots on detector
└───┴───┴───┴───┴───┴───┘
     Spot displacement ∝ local wavefront slope
```

For a flat wavefront, each spot falls at the center of its subaperture. For an aberrated wavefront, spots are displaced proportionally to the local wavefront slope:

$$\frac{\partial W}{\partial x} \approx \frac{\Delta x_{\text{spot}}}{f_{\text{lenslet}}}, \quad \frac{\partial W}{\partial y} \approx \frac{\Delta y_{\text{spot}}}{f_{\text{lenslet}}}$$

where $W(x,y)$ is the wavefront error and $f_{\text{lenslet}}$ is the microlens focal length. The wavefront $W$ is reconstructed by integrating the measured slopes — a problem equivalent to solving Poisson's equation $\nabla^2 W = \nabla \cdot (\text{measured slopes})$.

### 5.3 Zernike Polynomials

The standard basis for describing wavefront aberrations over a circular aperture is the **Zernike polynomials** $Z_n^m(\rho, \theta)$. They are orthogonal over the unit disk:

$$\int_0^1\int_0^{2\pi} Z_n^m(\rho, \theta)\,Z_{n'}^{m'}(\rho, \theta)\,\rho\,d\rho\,d\theta = \frac{\pi}{2(n+1)}\delta_{nn'}\delta_{mm'}$$

Each Zernike mode corresponds to a familiar aberration:

| Noll index $j$ | $n$ | $m$ | Name | Formula |
|:---:|:---:|:---:|------|---------|
| 1 | 0 | 0 | Piston | $1$ |
| 2 | 1 | 1 | Tilt (x) | $2\rho\cos\theta$ |
| 3 | 1 | -1 | Tilt (y) | $2\rho\sin\theta$ |
| 4 | 2 | 0 | Defocus | $\sqrt{3}(2\rho^2 - 1)$ |
| 5 | 2 | -2 | Astigmatism (oblique) | $\sqrt{6}\,\rho^2\sin 2\theta$ |
| 6 | 2 | 2 | Astigmatism (vertical) | $\sqrt{6}\,\rho^2\cos 2\theta$ |
| 7 | 3 | -1 | Coma (y) | $\sqrt{8}(3\rho^3 - 2\rho)\sin\theta$ |
| 8 | 3 | 1 | Coma (x) | $\sqrt{8}(3\rho^3 - 2\rho)\cos\theta$ |
| 11 | 4 | 0 | Spherical | $\sqrt{5}(6\rho^4 - 6\rho^2 + 1)$ |

The wavefront is expressed as $W(\rho, \theta) = \sum_j a_j Z_j(\rho, \theta)$, where the coefficients $a_j$ are determined by fitting to the measured slope data. The root-mean-square wavefront error is simply $\sigma_W = \sqrt{\sum_j a_j^2}$ (excluding piston).

### 5.4 Adaptive Optics

In astronomical adaptive optics (AO), a wavefront sensor measures the atmospheric distortion using light from a guide star (natural or laser-generated). A deformable mirror — with actuators that push and pull its surface — corrects the wavefront in real time (typically at 500-1000 Hz). The Strehl ratio (ratio of peak PSF intensity to diffraction-limited peak) improves from ~0.01 (seeing-limited) to >0.5 (near diffraction-limited) with AO.

Modern AO systems on 8-10 m telescopes routinely achieve diffraction-limited imaging at near-infrared wavelengths, and multi-conjugate AO (MCAO) extends the corrected field of view by using multiple deformable mirrors conjugated to different atmospheric layers.

---

## 6. Computational Photography

### 6.1 Beyond the Single Photograph

Traditional photography captures a single 2D projection of a 3D scene. **Computational photography** uses a combination of optics, sensors, and algorithms to transcend the limitations of a single conventional exposure — capturing more information about the scene (depth, dynamic range, spectral content) and synthesizing images that would be physically impossible for any single lens-detector system.

### 6.2 Light Field Cameras

A **light field camera** (plenoptic camera) captures not just the intensity at each pixel, but the direction from which light arrives. This is achieved by placing a microlens array before the image sensor, similar to a Shack-Hartmann sensor but used for imaging.

The light field $L(x, y, u, v)$ is a 4D function: $(x, y)$ is the spatial position on the sensor, and $(u, v)$ is the angular coordinate (which part of the main lens the ray passed through). With the full 4D light field, one can:

- **Computationally refocus** after capture: shift-and-add subaperture images with different offsets to focus at different depths
- **Change the viewpoint**: synthesize images from slightly different perspectives
- **Estimate depth**: from the parallax between subaperture images
- **Adjust depth of field**: digitally change the aperture size

The Lytro camera (2012) was the first commercial light field camera. The computational refocusing formula is:

$$I_\alpha(x, y) = \iint L\!\left(x + (1 - \alpha)u,\; y + (1 - \alpha)v,\; u, v\right)du\,dv$$

For $\alpha = 1$: focus at the microlens plane. For $\alpha \neq 1$: focus at other depths.

### 6.3 High Dynamic Range (HDR) Imaging

Real-world scenes have dynamic ranges exceeding $10^6$:1, but camera sensors typically capture only $10^3$:1 (12-bit). **HDR imaging** recovers the full dynamic range by merging multiple exposures:

$$\ln E(x,y) = \sum_j w(Z_j)\left[\ln Z_j - \ln \Delta t_j\right] \bigg/ \sum_j w(Z_j)$$

where $Z_j$ is the pixel value at exposure time $\Delta t_j$, and $w$ is a weighting function that emphasizes well-exposed pixels. The recovered radiance map $E$ is then **tone-mapped** for display on a standard monitor.

### 6.4 Coded Apertures

A **coded aperture** replaces the circular lens opening with a specifically designed pattern (e.g., a random binary mask or a broadband optimized pattern). This modifies the point spread function (PSF) in a controlled way, enabling:

- **Extended depth of field**: A cubic phase mask produces a PSF that is nearly invariant to defocus; deconvolution then yields a sharp image over a large depth range
- **Depth estimation**: The defocus blur depends on both depth and the aperture pattern; a well-designed pattern makes this relationship invertible
- **Motion deblur**: Flutter shutter (random open/close during exposure) creates a broadband temporal code that makes motion blur invertible

### 6.5 Single-Pixel Cameras

A **single-pixel camera** (compressive sensing camera) uses a spatial light modulator (SLM) to project a sequence of random patterns onto the scene and measures the total intensity with a single detector for each pattern:

$$y_i = \sum_{j} \Phi_{ij} x_j$$

where $x$ is the scene (vectorized), $\Phi_i$ is the $i$-th measurement pattern, and $y_i$ is the measured intensity. With $M \ll N$ measurements (where $N$ is the number of pixels), the scene can be reconstructed by solving the underdetermined system using sparsity-based optimization (compressive sensing):

$$\hat{x} = \arg\min_x \|\Psi x\|_1 \quad \text{subject to} \quad y = \Phi x$$

where $\Psi$ is a sparsifying transform (e.g., wavelet). This is especially powerful at wavelengths where detector arrays are expensive (infrared, terahertz).

---

## 7. Digital Holography

### 7.1 From Optical to Digital

In classical holography (Lesson 11), the hologram is recorded on photographic film and optically reconstructed by illumination with the reference beam. **Digital holography** replaces the film with a CCD/CMOS sensor and performs reconstruction numerically. This brings quantitative phase measurement, no chemical processing, and the full power of digital signal processing.

### 7.2 Recording

A digital hologram is recorded as an interference pattern between the object wave $U_o$ and a reference wave $U_r$ on a digital sensor:

$$I(x, y) = |U_o + U_r|^2 = |U_r|^2 + |U_o|^2 + U_r^*U_o + U_rU_o^*$$

The last two terms contain the holographic information. In **off-axis** geometry, the reference wave is tilted so that these terms are separated in the Fourier domain, enabling isolation by spatial filtering.

### 7.3 Numerical Reconstruction: Angular Spectrum Method

To reconstruct the object wave at a distance $d$ from the hologram plane:

1. Isolate the $U_r^*U_o$ term by Fourier filtering (off-axis) or phase stepping (on-axis)
2. Multiply by the digital reference wave: $U_h(x,y) = U_r^*U_o \cdot U_r = |U_r|^2 U_o \propto U_o$
3. Propagate to the object plane using the angular spectrum method:

$$U_{\text{obj}}(x,y) = \mathcal{F}^{-1}\!\left\{\mathcal{F}\{U_h\} \cdot e^{ik_z d}\right\}$$

where $k_z = \sqrt{k^2 - k_x^2 - k_y^2}$.

The reconstructed complex field gives both the **amplitude** $|U_{\text{obj}}|$ (intensity image) and the **phase** $\arg(U_{\text{obj}})$ (quantitative phase map).

### 7.4 Phase Unwrapping

The measured phase $\phi_{\text{wrapped}} = \arg(U)$ is wrapped to the interval $(-\pi, \pi]$. For samples with optical path differences exceeding one wavelength, the true phase must be **unwrapped** — the $2\pi$ jumps must be resolved to obtain a continuous phase map.

Common algorithms include:
- **Path-following**: Itoh's method — integrate phase differences along a path; fails near residues (inconsistent pixels)
- **Quality-guided**: Unwrap high-quality (low-noise) regions first
- **Least-squares**: Minimize $\sum|\nabla\phi_{\text{unwrapped}} - \text{wrapped gradient}|^2$; robust but smooths discontinuities
- **Goldstein's algorithm**: Identify residues (sources of unwrapping errors) and place branch cuts between them to prevent error propagation

### 7.5 Applications of Digital Holography

- **Quantitative phase imaging (QPI)**: Measure cell thickness, refractive index, and dry mass in biology without staining
- **Digital holographic microscopy (DHM)**: Numerical refocusing allows 3D reconstruction from a single hologram
- **Vibration analysis**: Time-resolved holograms reveal surface deformation at nanometer precision
- **Particle tracking**: 3D positions of microscopic particles in a volume

---

## 8. Machine Learning in Optics

### 8.1 The Deep Learning Revolution

Machine learning, particularly deep neural networks, has emerged as a transformative tool in computational optics since ~2017. Neural networks can learn complex mappings from data — mappings that are difficult or impossible to express analytically.

### 8.2 Learned Phase Retrieval

Traditional phase retrieval algorithms (GS, HIO) are iterative and may converge slowly or stagnate at local minima. Neural networks can learn to map a single diffraction pattern directly to the phase:

$$\phi_{\text{predicted}} = f_\theta(I_{\text{measured}})$$

where $f_\theta$ is a trained convolutional neural network. Architectures like U-Net, ResNet, and physics-informed networks (which incorporate the wave propagation operator into the loss function) achieve faster and more accurate phase retrieval than iterative methods, especially for noisy data.

### 8.3 Computational Imaging: Denoising and Super-Resolution

**Denoising**: Neural networks trained on pairs of noisy and clean images learn to remove noise while preserving features. In fluorescence microscopy, this enables imaging with 10-100x lower light doses, reducing phototoxicity.

**Super-resolution**: Networks learn to reconstruct high-resolution images from low-resolution measurements, surpassing the diffraction limit in specific contexts:
- Single-image super-resolution (SISR) using deep CNNs
- Structured illumination microscopy (SIM) reconstruction using neural networks
- Localization microscopy (PALM/STORM) with learned particle detection

### 8.4 Inverse Design of Optical Elements

Perhaps the most exciting application: using gradient-based optimization (adjoint methods) or generative models to **design** optical structures with desired performance:

- **Metasurface design**: Neural networks predict the far-field response of nanostructured surfaces, enabling rapid inverse design of metalenses, beam splitters, and holograms
- **Topology optimization**: Gradient descent on the refractive index distribution of a photonic device (the FDTD simulator is differentiated through the adjoint method) to achieve desired transmission/reflection
- **Diffractive optical neural networks (DONNs)**: Optical elements that physically perform neural network inference at the speed of light, designed by training the phase profile of each diffractive layer

### 8.5 Physics-Informed Neural Networks (PINNs)

PINNs incorporate physical laws (Maxwell's equations, the wave equation) into the loss function:

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda\,\mathcal{L}_{\text{physics}}$$

where $\mathcal{L}_{\text{physics}}$ penalizes violations of the governing equations. This regularization dramatically reduces the amount of training data required and ensures physically plausible solutions.

> **Analogy**: Traditional optical design is like an architect drawing blueprints by hand, relying on experience and rules of thumb. ML-driven inverse design is like giving the architect a magic sketchpad that explores millions of designs simultaneously, finding structures that no human would have imagined — flat lenses made of nanoscale pillars, waveguides with fractal-like cross-sections, diffractive elements that sort light by wavelength with near-perfect efficiency. The "magic" comes from automatic differentiation: the computer can compute how every tiny change in the design affects performance, and follow the gradient to the optimum.

---

## 9. Python Examples

### 9.1 Sequential Ray Tracer

```python
import numpy as np
import matplotlib.pyplot as plt

class Surface:
    """A spherical refracting surface in a sequential optical system."""
    def __init__(self, z_pos, radius, n_before, n_after):
        self.z = z_pos          # position along optical axis
        self.R = radius         # radius of curvature (inf for flat)
        self.n1 = n_before      # refractive index before surface
        self.n2 = n_after       # refractive index after surface

def trace_paraxial(y_in, u_in, surfaces):
    """
    Trace a paraxial ray through a sequence of surfaces using ABCD matrices.

    Why ABCD matrices? For paraxial rays, refraction and propagation are
    linear operations on (y, u). This lets us chain arbitrary sequences
    of surfaces as simple 2x2 matrix multiplications — far faster than
    solving Snell's law iteratively for each ray-surface intersection.

    Parameters
    ----------
    y_in : float — initial ray height above axis
    u_in : float — initial ray angle (radians, paraxial)
    surfaces : list of Surface objects in order of propagation

    Returns
    -------
    positions : list of (z, y) tuples for plotting the ray path
    """
    y, u = y_in, u_in
    positions = [(surfaces[0].z, y)]

    for i, surf in enumerate(surfaces):
        # Propagate from previous surface (or input) to this surface
        if i > 0:
            d = surf.z - surfaces[i-1].z
            # Free-space propagation: y' = y + d*u (paraxial, n=1 between lenses)
            y = y + d * u
        positions.append((surf.z, y))

        # Refract at this surface using the lensmaker's equation form
        # The ABCD refraction matrix has C = -(n2 - n1) / R
        if np.isinf(surf.R):
            power = 0.0
        else:
            power = (surf.n2 - surf.n1) / surf.R
        u = u - y * power  # paraxial Snell's law

    # Propagate to a final observation plane beyond the last surface
    d_final = 50.0
    y_final = y + d_final * u
    positions.append((surfaces[-1].z + d_final, y_final))

    return positions

# Define a simple two-lens system (doublet + singlet)
# This demonstrates how compound systems are built from elementary surfaces
surfaces = [
    Surface(z_pos=0,   radius=100.0,    n_before=1.0,  n_after=1.52),   # Front of lens 1
    Surface(z_pos=5,   radius=-100.0,   n_before=1.52, n_after=1.0),    # Back of lens 1
    Surface(z_pos=50,  radius=80.0,     n_before=1.0,  n_after=1.67),   # Front of lens 2
    Surface(z_pos=53,  radius=-200.0,   n_before=1.67, n_after=1.0),    # Back of lens 2
]

# Trace a fan of rays at different heights
fig, ax = plt.subplots(figsize=(12, 5))

heights = np.linspace(-15, 15, 11)
for y0 in heights:
    positions = trace_paraxial(y0, 0.0, surfaces)
    zs, ys = zip(*positions)
    ax.plot(zs, ys, 'b-', alpha=0.6, linewidth=0.8)

# Draw the lens surfaces as vertical lines
for surf in surfaces:
    ax.axvline(x=surf.z, color='gray', linestyle='--', alpha=0.5)
    ax.text(surf.z, 17, f'R={surf.R:.0f}', fontsize=7, ha='center')

ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_xlabel('z (mm)')
ax.set_ylabel('Ray height y (mm)')
ax.set_title('Paraxial Ray Trace Through a Two-Lens System')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ray_trace.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.2 Beam Propagation Method (Split-Step)

```python
import numpy as np
import matplotlib.pyplot as plt

def bpm_split_step(psi0, x, z_array, n_profile, k0, n0):
    """
    Beam Propagation Method using the split-step Fourier algorithm.

    Why split-step? The paraxial wave equation has two terms: diffraction
    (diagonal in k-space) and refraction (diagonal in real space). By
    alternating between Fourier domain and real-space operations, we handle
    each term where it is simplest — this is both accurate and efficient (FFT
    gives O(N log N) cost per step instead of O(N^2) for direct convolution).

    Parameters
    ----------
    psi0 : 1D array — initial field envelope
    x : 1D array — transverse coordinate (m)
    z_array : 1D array — propagation positions (m)
    n_profile : callable — n_profile(x, z) returns refractive index array
    k0 : float — free-space wavenumber 2*pi/lambda
    n0 : float — background refractive index

    Returns
    -------
    field : 2D array (len(z_array), len(x)) — field at each z
    """
    dx = x[1] - x[0]
    N = len(x)
    k = k0 * n0  # reference wavenumber

    # Spatial frequency axis — must match FFT conventions
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    # Store field at each z position for visualization
    field = np.zeros((len(z_array), N), dtype=complex)
    field[0] = psi0
    psi = psi0.copy()

    for i in range(1, len(z_array)):
        dz = z_array[i] - z_array[i-1]
        z_mid = 0.5 * (z_array[i] + z_array[i-1])

        # Step 1: Diffraction in Fourier domain
        # The free-space propagator exp(-i*kx^2*dz/(2k)) spreads the beam;
        # each spatial frequency acquires a phase shift proportional to kx^2
        psi_k = np.fft.fft(psi)
        diffraction_phase = np.exp(-1j * kx**2 * dz / (2 * k))
        psi_k *= diffraction_phase
        psi = np.fft.ifft(psi_k)

        # Step 2: Refraction in real space
        # The local index variation dn bends the wavefront — this is where
        # the waveguide confinement enters the simulation
        n = n_profile(x, z_mid)
        dn = n - n0
        refraction_phase = np.exp(1j * k0 * dn * dz)
        psi *= refraction_phase

        field[i] = psi

    return field

# Example: Gaussian beam in a GRIN (graded-index) waveguide
# The parabolic index profile n(x) = n0 * sqrt(1 - alpha^2 * x^2)
# guides the beam by continuously bending rays toward the axis
lambda0 = 1.0e-6           # wavelength: 1 um
k0 = 2 * np.pi / lambda0
n0 = 1.5                   # core index
alpha = 200.0              # GRIN parameter (1/m)

# Transverse and longitudinal grids
x = np.linspace(-50e-6, 50e-6, 512)
z = np.linspace(0, 2e-3, 500)

# GRIN index profile: parabolic approximation
def grin_profile(x, z):
    return n0 * np.sqrt(np.maximum(1 - (alpha * x)**2, 0.5))

# Launch a Gaussian beam offset from the axis to excite oscillation
w0 = 10e-6  # beam waist
x_offset = 15e-6
psi0 = np.exp(-(x - x_offset)**2 / w0**2)

# Run the BPM simulation
field = bpm_split_step(psi0, x, z, grin_profile, k0, n0)

# Visualize the propagation — intensity shows the beam oscillating
# in the parabolic potential, analogous to a mass on a spring
fig, ax = plt.subplots(figsize=(12, 5))
extent = [z[0]*1e3, z[-1]*1e3, x[0]*1e6, x[-1]*1e6]
im = ax.imshow(np.abs(field.T)**2, aspect='auto', extent=extent,
               origin='lower', cmap='hot')
ax.set_xlabel('Propagation z (mm)')
ax.set_ylabel('Transverse x (um)')
ax.set_title('BPM: Gaussian Beam in a GRIN Waveguide')
plt.colorbar(im, label='Intensity |psi|^2')
plt.tight_layout()
plt.savefig('bpm_grin.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.3 Gerchberg-Saxton Phase Retrieval

```python
import numpy as np
import matplotlib.pyplot as plt

def gerchberg_saxton(target_amplitude, source_amplitude, n_iter=100):
    """
    Gerchberg-Saxton algorithm for phase retrieval.

    Why iterative? We know the amplitude in two planes (source and target)
    but not the phase in either. A single Fourier transform cannot recover
    phase from amplitude alone — the problem is underdetermined. By
    alternating between planes and enforcing the known amplitude at each,
    we progressively narrow down the consistent phase distribution.

    Parameters
    ----------
    target_amplitude : 2D array — desired amplitude in the Fourier plane
    source_amplitude : 2D array — known amplitude in the source plane
    n_iter : int — number of iterations

    Returns
    -------
    phase : 2D array — recovered phase in the source plane
    errors : list — RMS error at each iteration for convergence monitoring
    """
    # Initialize with random phase — the algorithm will iteratively
    # refine this guess using the two amplitude constraints
    phase = 2 * np.pi * np.random.random(source_amplitude.shape)
    errors = []

    for k in range(n_iter):
        # Source plane: combine known amplitude with current phase estimate
        u_source = source_amplitude * np.exp(1j * phase)

        # Propagate to Fourier plane (forward FFT with proper centering)
        u_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_source)))

        # Record error: how well does the Fourier amplitude match the target?
        error = np.sqrt(np.mean((np.abs(u_fourier) - target_amplitude)**2))
        errors.append(error)

        # Fourier constraint: replace amplitude, keep phase
        # This is the "projection onto the Fourier amplitude constraint set"
        fourier_phase = np.angle(u_fourier)
        u_fourier = target_amplitude * np.exp(1j * fourier_phase)

        # Propagate back to source plane (inverse FFT)
        u_source = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u_fourier)))

        # Source constraint: replace amplitude, keep phase
        phase = np.angle(u_source)

    return phase, errors

# Example: Design a phase mask that produces a ring pattern in the far field.
# This is a practical beam shaping problem — we want to redistribute a
# uniform laser beam into a ring, using only a phase-only spatial light
# modulator (SLM). We can control phase but NOT amplitude.
N = 256
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)

# Target: ring pattern in Fourier plane
target_amp = np.exp(-((R - 0.3) / 0.05)**2)  # ring at radius 0.3
target_amp /= target_amp.max()

# Source: uniform amplitude (representing a flat-top laser beam)
source_amp = np.ones((N, N))
source_amp[R > 0.8] = 0  # circular aperture

# Run the GS algorithm
np.random.seed(42)
recovered_phase, errors = gerchberg_saxton(target_amp, source_amp, n_iter=200)

# Verify: propagate the recovered phase and check the far-field pattern
u_result = source_amp * np.exp(1j * recovered_phase)
far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_result)))

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(source_amp, cmap='gray')
axes[0].set_title('Source Amplitude')

axes[1].imshow(recovered_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
axes[1].set_title('Recovered Phase')

axes[2].imshow(np.abs(far_field)**2, cmap='hot')
axes[2].set_title('Reconstructed Far Field')

axes[3].semilogy(errors)
axes[3].set_xlabel('Iteration')
axes[3].set_ylabel('RMS Error')
axes[3].set_title('Convergence')
axes[3].grid(True, alpha=0.3)

for ax in axes[:3]:
    ax.axis('off')

plt.suptitle('Gerchberg-Saxton Phase Retrieval', fontsize=13)
plt.tight_layout()
plt.savefig('gs_phase_retrieval.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.4 Zernike Polynomials

```python
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def zernike_radial(n, m, rho):
    """
    Compute the radial component R_n^m(rho) of the Zernike polynomial.

    Why not just use a lookup table? The radial polynomial formula works
    for arbitrary (n, m) pairs, making this function general enough to
    compute any Zernike mode up to arbitrary order — essential for
    high-order adaptive optics with hundreds of modes.
    """
    m_abs = abs(m)
    if (n - m_abs) % 2 != 0:
        return np.zeros_like(rho)

    R = np.zeros_like(rho, dtype=float)
    for s in range(int((n - m_abs) / 2) + 1):
        # Each term in the sum contributes a power of rho;
        # the alternating signs produce the oscillatory radial pattern
        coef = ((-1)**s * factorial(n - s)
                / (factorial(s)
                   * factorial(int((n + m_abs) / 2) - s)
                   * factorial(int((n - m_abs) / 2) - s)))
        R += coef * rho**(n - 2*s)
    return R

def zernike(n, m, rho, theta):
    """
    Compute Zernike polynomial Z_n^m(rho, theta) over the unit disk.

    The normalization follows the standard Noll convention so that
    each mode has unit RMS over the pupil — this means the Zernike
    coefficient a_j directly gives the RMS contribution of that mode
    to the total wavefront error.
    """
    # Normalization factor (Noll convention)
    if m == 0:
        norm = np.sqrt(n + 1)
    else:
        norm = np.sqrt(2 * (n + 1))

    R = zernike_radial(n, m, rho)

    if m >= 0:
        Z = norm * R * np.cos(m * theta)
    else:
        Z = norm * R * np.sin(abs(m) * theta)

    # Mask outside the unit disk — Zernike polynomials are undefined there
    Z[rho > 1.0] = np.nan
    return Z

# Generate a grid over the unit disk
N = 300
x = np.linspace(-1.1, 1.1, N)
X, Y = np.meshgrid(x, x)
rho = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

# Plot the first 15 Zernike modes (Noll ordering)
# These are the modes most relevant to optical testing:
# low-order = alignment errors, mid-order = manufacturing, high-order = turbulence
noll_modes = [
    (0, 0, 'Piston'),
    (1, 1, 'Tilt X'), (1, -1, 'Tilt Y'),
    (2, 0, 'Defocus'), (2, -2, 'Astig 45'), (2, 2, 'Astig 0'),
    (3, -1, 'Coma Y'), (3, 1, 'Coma X'),
    (3, -3, 'Trefoil Y'), (3, 3, 'Trefoil X'),
    (4, 0, 'Spherical'), (4, 2, 'Sec Astig 0'),
    (4, -2, 'Sec Astig 45'), (4, 4, 'Tetrafoil 0'),
    (4, -4, 'Tetrafoil 45'),
]

fig, axes = plt.subplots(3, 5, figsize=(16, 10))
axes = axes.ravel()

for idx, (n, m, name) in enumerate(noll_modes):
    Z = zernike(n, m, rho, theta)
    im = axes[idx].imshow(Z, cmap='RdBu_r', extent=[-1.1, 1.1, -1.1, 1.1])
    axes[idx].set_title(f'Z({n},{m:+d})\n{name}', fontsize=9)
    axes[idx].axis('off')
    plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

plt.suptitle('Zernike Polynomials (First 15 Modes)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('zernike_modes.png', dpi=150, bbox_inches='tight')
plt.show()

# Example: Synthesize an aberrated wavefront and decompose it
# This simulates what a Shack-Hartmann sensor + Zernike fitting does:
# measure the wavefront, express it as a sum of Zernike modes, and
# identify which aberrations dominate
coefficients = {
    (2, 0): 0.5,    # defocus — the largest contributor
    (2, 2): 0.3,    # astigmatism
    (3, 1): -0.2,   # coma X
    (4, 0): 0.1,    # spherical aberration
}

W = np.zeros_like(rho)
for (n, m), a in coefficients.items():
    W += a * zernike(n, m, rho, theta)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(W, cmap='RdBu_r', extent=[-1.1, 1.1, -1.1, 1.1])
ax.set_title('Synthesized Aberrated Wavefront\n'
             r'($0.5\lambda$ defocus + $0.3\lambda$ astig + '
             r'$0.2\lambda$ coma + $0.1\lambda$ spherical)')
ax.axis('off')
plt.colorbar(im, label='Wavefront error (waves)')
plt.tight_layout()
plt.savefig('aberrated_wavefront.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 10. Summary

| Method | Domain | Key Equation / Idea | Typical Application |
|--------|--------|---------------------|---------------------|
| Ray tracing | Geometric optics | Snell's law + ABCD matrices | Lens design, cameras |
| BPM | Paraxial wave optics | Split-step Fourier: $\psi(z+\Delta z) = e^{i\hat{N}\Delta z}\mathcal{F}^{-1}\{e^{i\hat{D}\Delta z}\mathcal{F}\{\psi\}\}$ | Waveguides, fibers |
| FDTD | Full-wave Maxwell | Yee grid + leapfrog time stepping; CFL stability | Photonic crystals, plasmonics |
| Phase retrieval | Inverse problems | Gerchberg-Saxton: alternating projections in two planes | Wavefront correction, CDI |
| TIE | Deterministic phase | $-k\,\partial I/\partial z = \nabla \cdot (I\nabla\phi)$ | Microscopy, electron imaging |
| Wavefront sensing | Aberration measurement | Shack-Hartmann spots → Zernike decomposition | Adaptive optics, optical testing |
| Computational photography | Enhanced imaging | Light fields, HDR, coded apertures, compressive sensing | Consumer cameras, microscopy |
| Digital holography | Quantitative phase | Angular spectrum reconstruction + phase unwrapping | DHM, vibration analysis |
| ML in optics | Data-driven | Neural networks: $\phi = f_\theta(I)$ | Phase retrieval, inverse design |

**Key takeaway**: Computational optics has evolved from simple ray tracing to a discipline where algorithms are as important as lenses. The trend is unmistakable — optics is becoming a computational science, where software and hardware are co-designed to achieve imaging performance that neither could achieve alone.

---

## 11. Exercises

### Exercise 1: ABCD Ray Tracing

Design a two-lens optical relay (4f system) with $f_1 = 50\,\text{mm}$ and $f_2 = 100\,\text{mm}$. (a) Write the system ABCD matrix and verify that the magnification is $M = -f_2/f_1 = -2$. (b) Trace a fan of 11 rays with heights from $-10$ to $+10\,\text{mm}$ at angle $u = 0$ through the system. (c) Add a third lens with $f_3 = -30\,\text{mm}$ placed 20 mm after the second lens and plot the new ray fan. What happens to the focal point?

### Exercise 2: BPM Waveguide Modes

Use the BPM code to simulate a step-index slab waveguide with core index $n_{\text{core}} = 1.50$ and cladding index $n_{\text{clad}} = 1.48$, core half-width $a = 5\,\mu\text{m}$, at $\lambda = 1.55\,\mu\text{m}$. (a) Launch a Gaussian beam centered on the waveguide and propagate for 5 mm. How many modes are excited? (b) Calculate the V-number and predict the number of guided modes analytically. (c) Launch a tilted Gaussian and observe the radiation of unguided light.

### Exercise 3: Phase Retrieval Challenge

Generate a complex test object: $U = A(x,y)\exp[i\phi(x,y)]$ where $A$ is the silhouette of the letter "F" and $\phi$ is a random smooth phase (sum of low-order Zernike polynomials). (a) Compute $I_{\text{obj}} = |U|^2$ and $I_{\text{far}} = |\mathcal{F}\{U\}|^2$. (b) Run the GS algorithm for 500 iterations and plot the convergence curve. (c) Try starting with different random phases — does the algorithm always converge to the same solution? (d) Implement the HIO algorithm (Fienup, $\beta = 0.9$) and compare its convergence speed.

### Exercise 4: Zernike Wavefront Analysis

A Shack-Hartmann sensor measures the following Zernike coefficients (in waves) for a telescope primary mirror: $a_4 = 0.35$ (defocus), $a_5 = -0.12$ (astigmatism), $a_7 = 0.08$ (coma), $a_{11} = 0.15$ (spherical). (a) Synthesize and plot the wavefront error map. (b) Calculate the total RMS wavefront error. (c) Estimate the Strehl ratio using the Marechal approximation $S \approx e^{-(2\pi\sigma)^2}$. (d) If you could correct only one mode, which would give the greatest improvement in Strehl ratio?

### Exercise 5: Digital Holography Simulation

Generate a digital hologram of a point object located 10 mm from the sensor ($\lambda = 633\,\text{nm}$, sensor size $5\,\text{mm} \times 5\,\text{mm}$, 1024 $\times$ 1024 pixels, plane-wave reference at $3^\circ$ off-axis). (a) Compute the hologram intensity pattern. (b) Reconstruct the object using the angular spectrum method at distances of 5, 10, and 15 mm. (c) At which distance is the point object best focused? (d) Add Gaussian noise (SNR = 20 dB) to the hologram and repeat the reconstruction — how does the noise affect the result?

### Exercise 6: Comparing Computational Methods

For a dielectric slab of thickness $d = 2\lambda$ and index $n = 2.0$ illuminated by a normally incident plane wave: (a) Calculate the transmittance analytically using the Fabry-Perot formula. (b) Simulate the same configuration using BPM. (c) Discuss why BPM gives an approximate result (what physics does it miss?). (d) Under what conditions would FDTD be necessary instead of BPM for this problem?

---

## 12. References

1. Goodman, J. W. (2017). *Introduction to Fourier Optics* (4th ed.). W.H. Freeman. — The definitive reference for Fourier optics and propagation methods (Chapters 3-4 on angular spectrum, Chapter 5 on coherent imaging).
2. Saleh, B. E. A., & Teich, M. C. (2019). *Fundamentals of Photonics* (3rd ed.). Wiley. — Chapter 4 (ray optics matrices), Chapter 7 (beam propagation).
3. Taflove, A., & Hagness, S. C. (2005). *Computational Electrodynamics: The Finite-Difference Time-Domain Method* (3rd ed.). Artech House. — The standard FDTD reference.
4. Fienup, J. R. (1982). "Phase retrieval algorithms: a comparison." *Applied Optics*, 21(15), 2758-2769. — Classic paper on GS, HIO, and error-reduction algorithms.
5. Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence." *JOSA*, 66(3), 207-211. — Standard Zernike indexing for wavefront analysis.
6. Kim, M. K. (2010). "Principles and techniques of digital holographic microscopy." *SPIE Reviews*, 1(1), 018005. — Comprehensive review of digital holography.
7. Rivenson, Y., Zhang, Y., Gunaydin, H., Teng, D., & Ozcan, A. (2018). "Phase recovery and holographic image reconstruction using deep learning in neural networks." *Light: Science & Applications*, 7, 17141. — Pioneering work on deep learning for holographic reconstruction.
8. Molesky, S., et al. (2018). "Inverse design in nanophotonics." *Nature Photonics*, 12, 659-670. — Review of computational inverse design of photonic structures.

---

[← Previous: 13. Quantum Optics Primer](13_Quantum_Optics_Primer.md) | [Next: 15. Zernike Polynomials →](15_Zernike_Polynomials.md)
