# 15. Multipole Expansion

[← Previous: 14. Relativistic Electrodynamics](14_Relativistic_Electrodynamics.md) | [Next: 16. Computational Electrodynamics →](16_Computational_Electrodynamics.md)

## Learning Objectives

1. Understand the multipole expansion as a systematic approximation for distant fields
2. Derive the monopole, dipole, and quadrupole terms in powers of $1/r$
3. Connect the expansion to Legendre polynomials and spherical harmonics
4. Compute electric and magnetic multipole moments for given charge and current distributions
5. Analyze multipole radiation: electric dipole, magnetic dipole, and electric quadrupole
6. Understand selection rules governing which multipoles radiate
7. Implement multipole calculations and visualize radiation patterns in Python

When observing a charge distribution from far away, the fine details of its shape become less important — what matters is a sequence of coarser and coarser descriptions: the total charge (monopole), the separation of positive and negative charges (dipole), the asymmetry of that separation (quadrupole), and so on. The multipole expansion provides this hierarchy, decomposing an arbitrary charge distribution into a series of ever-more-refined contributions that fall off as successively higher powers of $1/r$. For radiation, the multipole expansion reveals that accelerating charges radiate predominantly as dipoles, with quadrupole and higher-order corrections becoming important when the source is comparable in size to the wavelength.

> **Analogy**: Think of viewing a city from an airplane. At 30,000 feet (monopole level), you see only that "there is a city here" — a single point. Descend to 10,000 feet (dipole level), and you can distinguish "the city is elongated north-south." At 5,000 feet (quadrupole level), you see that "it has a dense center with suburbs extending in four directions." Each level of the multipole expansion adds finer spatial detail, like progressively zooming in.

---

## 1. The Multipole Expansion for Electrostatics

### 1.1 Setup

Consider a localized charge distribution $\rho(\mathbf{r}')$ confined to a region of size $d$. We want the potential at a point $\mathbf{r}$ far from the source ($r \gg d$).

The exact potential is:

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d^3r'$$

### 1.2 Expansion of $1/|\mathbf{r} - \mathbf{r}'|$

The key mathematical identity is:

$$\frac{1}{|\mathbf{r} - \mathbf{r}'|} = \sum_{\ell=0}^{\infty} \frac{r'^{\ell}}{r^{\ell+1}} P_\ell(\cos\alpha)$$

where $\alpha$ is the angle between $\mathbf{r}$ and $\mathbf{r}'$, and $P_\ell$ are the **Legendre polynomials**. This is valid for $r > r'$.

The first few Legendre polynomials are:

$$P_0(x) = 1, \quad P_1(x) = x, \quad P_2(x) = \frac{1}{2}(3x^2 - 1), \quad P_3(x) = \frac{1}{2}(5x^3 - 3x)$$

### 1.3 The Expansion Term by Term

Substituting into the potential:

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0}\sum_{\ell=0}^{\infty} \frac{1}{r^{\ell+1}} \int r'^{\ell} P_\ell(\cos\alpha) \, \rho(\mathbf{r}') \, d^3r'$$

**Monopole ($\ell = 0$)**:

$$V_0 = \frac{1}{4\pi\epsilon_0} \frac{Q}{r}, \quad Q = \int \rho(\mathbf{r}') \, d^3r'$$

This is the total charge — it looks like a point charge from far away.

**Dipole ($\ell = 1$)**:

$$V_1 = \frac{1}{4\pi\epsilon_0} \frac{\mathbf{p} \cdot \hat{r}}{r^2}, \quad \mathbf{p} = \int \mathbf{r}' \rho(\mathbf{r}') \, d^3r'$$

The dipole moment $\mathbf{p}$ measures the separation of positive and negative charge.

**Quadrupole ($\ell = 2$)**:

$$V_2 = \frac{1}{4\pi\epsilon_0} \frac{1}{r^3} \sum_{ij} \frac{1}{2} Q_{ij} \hat{r}_i \hat{r}_j$$

where the quadrupole moment tensor is:

$$Q_{ij} = \int (3r'_i r'_j - r'^2 \delta_{ij}) \rho(\mathbf{r}') \, d^3r'$$

The quadrupole tensor is symmetric and traceless (5 independent components).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

def multipole_potential(charges, positions, r_eval, theta_eval, max_ell=10):
    """
    Compute electrostatic potential using multipole expansion.

    Parameters:
        charges   : list of charges
        positions : list of position vectors [[x,y,z], ...]
        r_eval    : radial distances for evaluation
        theta_eval: polar angles for evaluation
        max_ell   : maximum multipole order

    Why multipole expansion: for distant fields, only the first few
    terms contribute significantly, giving both physical insight
    (what "shape" does the charge distribution have?) and computational
    efficiency (O(L) vs O(N) for N charges).
    """
    eps_0 = 8.854e-12
    charges = np.array(charges)
    positions = np.array(positions)

    V = np.zeros((len(r_eval), len(theta_eval)))

    for ell in range(max_ell + 1):
        P_ell = legendre(ell)

        # Compute multipole moment: q_ell = sum_i q_i * r_i^ell * P_ell(cos theta_i)
        for q, pos in zip(charges, positions):
            r_i = np.linalg.norm(pos)
            if r_i < 1e-15:
                cos_alpha_i = 0
            else:
                cos_alpha_i = pos[2] / r_i  # z/r for alignment with z-axis

            q_ell = q * r_i**ell * P_ell(cos_alpha_i)

            for j, theta in enumerate(theta_eval):
                V[:, j] += (1 / (4 * np.pi * eps_0)) * q_ell * P_ell(np.cos(theta)) / r_eval**(ell + 1)

    return V

def compare_exact_vs_multipole():
    """
    Compare exact potential with successive multipole approximations
    for a simple charge distribution.
    """
    eps_0 = 8.854e-12

    # Two charges: +q at (0,0,d/2) and -q at (0,0,-d/2)
    # This is a pure dipole
    d = 0.1   # separation (m)
    q = 1e-9  # charge (C)

    charges = [q, -q]
    positions = [[0, 0, d/2], [0, 0, -d/2]]

    # Evaluate along different angles at fixed r
    theta = np.linspace(0.01, np.pi - 0.01, 200)
    r_eval = np.array([1.0])  # 1 meter away

    # Exact potential
    V_exact = np.zeros(len(theta))
    for qi, pos in zip(charges, positions):
        for j, th in enumerate(theta):
            r_point = np.array([r_eval[0] * np.sin(th), 0, r_eval[0] * np.cos(th)])
            dist = np.linalg.norm(r_point - np.array(pos))
            V_exact[j] += qi / (4 * np.pi * eps_0 * dist)

    # Multipole approximations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Angular pattern comparison
    ax = axes[0]
    ax.plot(np.degrees(theta), V_exact * 1e9, 'k-', linewidth=2, label='Exact')

    for max_l, color, style in [(0, 'red', '--'), (1, 'blue', '-.'), (2, 'green', ':')]:
        V_approx = np.zeros(len(theta))
        for ell in range(max_l + 1):
            P_ell = legendre(ell)
            for qi, pos in zip(charges, positions):
                r_i = np.linalg.norm(pos)
                cos_alpha_i = pos[2] / r_i if r_i > 0 else 0
                q_ell = qi * r_i**ell * P_ell(cos_alpha_i)
                for j, th in enumerate(theta):
                    V_approx[j] += q_ell * P_ell(np.cos(th)) / (4 * np.pi * eps_0 * r_eval[0]**(ell + 1))

        label = f'Up to $\\ell$ = {max_l} ({"monopole" if max_l==0 else "dipole" if max_l==1 else "quadrupole"})'
        ax.plot(np.degrees(theta), V_approx * 1e9, color=color, linestyle=style,
                linewidth=1.5, label=label)

    ax.set_xlabel('Polar angle $\\theta$ (degrees)')
    ax.set_ylabel('Potential (nV at r = 1 m)')
    ax.set_title('Multipole Approximation: Dipole Configuration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Radial falloff comparison
    ax = axes[1]
    r_range = np.logspace(-0.5, 1.5, 100)  # 0.3 to 30 meters
    theta_fixed = np.pi / 4  # 45 degrees

    V_exact_r = np.zeros(len(r_range))
    V_mono_r = np.zeros(len(r_range))
    V_dipo_r = np.zeros(len(r_range))

    for i, r in enumerate(r_range):
        r_point = np.array([r * np.sin(theta_fixed), 0, r * np.cos(theta_fixed)])
        for qi, pos in zip(charges, positions):
            dist = np.linalg.norm(r_point - np.array(pos))
            V_exact_r[i] += qi / (4 * np.pi * eps_0 * dist)

        # Dipole approximation: p = q*d along z
        p = q * d
        V_dipo_r[i] = p * np.cos(theta_fixed) / (4 * np.pi * eps_0 * r**2)

    ax.loglog(r_range, np.abs(V_exact_r), 'k-', linewidth=2, label='Exact')
    ax.loglog(r_range, np.abs(V_dipo_r), 'b--', linewidth=1.5, label='Dipole approx')
    ax.loglog(r_range, np.abs(V_exact_r[0]) * (r_range[0]/r_range)**2,
              'gray', linestyle=':', alpha=0.5, label='$\\sim 1/r^2$ guide')
    ax.set_xlabel('Distance r (m)')
    ax.set_ylabel('|Potential| (V)')
    ax.set_title('Radial Falloff at $\\theta$ = 45°')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig("multipole_expansion.png", dpi=150)
    plt.show()

compare_exact_vs_multipole()
```

---

## 2. Spherical Harmonics

### 2.1 Addition Theorem

The Legendre polynomial can be decomposed using the **addition theorem**:

$$P_\ell(\cos\alpha) = \frac{4\pi}{2\ell + 1}\sum_{m=-\ell}^{\ell} Y_\ell^{m*}(\theta', \phi') \, Y_\ell^m(\theta, \phi)$$

This allows us to express the multipole expansion in terms of spherical harmonics:

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0}\sum_{\ell=0}^{\infty}\sum_{m=-\ell}^{\ell} \frac{4\pi}{2\ell + 1} \frac{q_{\ell m}}{r^{\ell+1}} Y_\ell^m(\theta, \phi)$$

where the **spherical multipole moments** are:

$$q_{\ell m} = \int r'^{\ell} Y_\ell^{m*}(\theta', \phi') \, \rho(\mathbf{r}') \, d^3r'$$

### 2.2 Properties of Spherical Harmonics

The spherical harmonics $Y_\ell^m(\theta, \phi)$ form a complete orthonormal set on the sphere:

$$\int_0^{2\pi}\int_0^{\pi} Y_\ell^{m*}(\theta, \phi) Y_{\ell'}^{m'}(\theta, \phi) \sin\theta \, d\theta \, d\phi = \delta_{\ell\ell'}\delta_{mm'}$$

The first several are:

$$Y_0^0 = \frac{1}{\sqrt{4\pi}}, \quad Y_1^0 = \sqrt{\frac{3}{4\pi}}\cos\theta, \quad Y_1^{\pm 1} = \mp\sqrt{\frac{3}{8\pi}}\sin\theta \, e^{\pm i\phi}$$

$$Y_2^0 = \sqrt{\frac{5}{16\pi}}(3\cos^2\theta - 1), \quad Y_2^{\pm 1} = \mp\sqrt{\frac{15}{8\pi}}\sin\theta\cos\theta \, e^{\pm i\phi}$$

### 2.3 Number of Independent Components

At each order $\ell$, there are $2\ell + 1$ independent moments:
- $\ell = 0$ (monopole): 1 component ($Q$)
- $\ell = 1$ (dipole): 3 components ($p_x, p_y, p_z$)
- $\ell = 2$ (quadrupole): 5 components (traceless symmetric tensor)
- $\ell = 3$ (octupole): 7 components

```python
from scipy.special import sph_harm

def plot_spherical_harmonics(max_ell=3):
    """
    Visualize spherical harmonics Y_l^m on the sphere.

    Why spherical harmonics: they are the angular basis functions
    for the multipole expansion, analogous to sine and cosine
    being the basis for Fourier series. Each Y_l^m represents
    a specific angular pattern of the potential.
    """
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 200)
    THETA, PHI = np.meshgrid(theta, phi)

    fig, axes = plt.subplots(max_ell + 1, 2 * max_ell + 1, figsize=(20, 12),
                              subplot_kw={'projection': 'polar'})

    for ell in range(max_ell + 1):
        for m_idx, m in enumerate(range(-ell, ell + 1)):
            col = m_idx + (max_ell - ell)  # center alignment
            ax = axes[ell, col]

            # Compute Y_l^m
            # scipy sph_harm uses (m, l, phi, theta) convention
            Y = sph_harm(abs(m), ell, PHI, THETA)
            if m < 0:
                Y = np.sqrt(2) * (-1)**m * Y.imag
            elif m > 0:
                Y = np.sqrt(2) * (-1)**m * Y.real
            else:
                Y = Y.real

            # Plot in polar coordinates (theta is the radial axis)
            # Average over phi for a 2D cross-section
            Y_cross = Y[0, :]  # phi = 0 cross-section
            r_plot = np.abs(Y_cross)

            ax.plot(theta, r_plot, 'b-', linewidth=1.5)
            ax.fill_between(theta, 0, r_plot,
                           where=(Y_cross >= 0), alpha=0.3, color='blue')
            ax.fill_between(theta, 0, r_plot,
                           where=(Y_cross < 0), alpha=0.3, color='red')
            ax.set_title(f'$Y_{{{ell}}}^{{{m}}}$', fontsize=10, pad=5)
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        # Hide unused subplots
        for m_idx in range(2 * max_ell + 1):
            if m_idx < (max_ell - ell) or m_idx > (max_ell + ell):
                axes[ell, m_idx].set_visible(False)

    plt.suptitle('Spherical Harmonics $Y_\\ell^m$ (cross-sections at $\\phi=0$)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("spherical_harmonics.png", dpi=150)
    plt.show()

plot_spherical_harmonics(max_ell=3)
```

---

## 3. Magnetic Multipoles

### 3.1 Magnetic Vector Potential Expansion

The magnetic vector potential of a localized current distribution is:

$$\mathbf{A}(\mathbf{r}) = \frac{\mu_0}{4\pi}\int \frac{\mathbf{J}(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d^3r'$$

Expanding $1/|\mathbf{r} - \mathbf{r}'|$ as before, the leading terms are:

**Magnetic monopole ($\ell = 0$)**: Vanishes identically because $\nabla \cdot \mathbf{B} = 0$ — there are no magnetic charges.

**Magnetic dipole ($\ell = 1$)**:

$$\mathbf{A}_{\text{dip}} = \frac{\mu_0}{4\pi}\frac{\mathbf{m} \times \hat{r}}{r^2}$$

where the magnetic dipole moment is:

$$\mathbf{m} = \frac{1}{2}\int \mathbf{r}' \times \mathbf{J}(\mathbf{r}') \, d^3r'$$

For a planar current loop: $\mathbf{m} = I \mathbf{A}$ (current times area vector).

### 3.2 Magnetic Quadrupole

The magnetic quadrupole tensor is:

$$M_{ij} = \frac{1}{3}\int (r'_i J_j + r'_j J_i) \, d^3r'$$

Unlike the electric quadrupole, the magnetic quadrupole is not traceless in general, and its radiation properties are fundamentally different.

---

## 4. Multipole Radiation

### 4.1 Radiation Hierarchy

For an oscillating source of characteristic size $d$ and wavelength $\lambda$, the radiated power from each multipole order scales as:

$$P_\ell \sim \left(\frac{d}{\lambda}\right)^{2\ell}$$

Since typically $d \ll \lambda$ (long-wavelength limit), the radiation is dominated by the lowest-order nonvanishing multipole:

1. **Electric dipole (E1)**: $P \propto |\ddot{\mathbf{p}}|^2 \propto \omega^4 p_0^2$ — dominant for most antennas and atomic transitions
2. **Magnetic dipole (M1)**: $P \propto |\ddot{\mathbf{m}}|^2/c^2 \propto \omega^4 m_0^2/c^2$ — suppressed by $(v/c)^2$
3. **Electric quadrupole (E2)**: $P \propto |\dddot{Q}|^2/c^2 \propto \omega^6 Q_0^2 d^2/c^2$ — suppressed by $(d/\lambda)^2$

### 4.2 Radiation Patterns

Each multipole has a characteristic angular distribution:

| Multipole | Power pattern | Parity | Polarization |
|-----------|--------------|--------|-------------|
| E1 | $\sin^2\theta$ | Odd ($-1$) | $\hat{\theta}$ |
| M1 | $\sin^2\theta$ | Even ($+1$) | $\hat{\phi}$ |
| E2 | $\sin^2\theta\cos^2\theta$ (for $m=0$) | Even ($+1$) | Mixed |

The E1 and M1 patterns look the same in shape, but they have different **parities** — the electric field of M1 radiation has the opposite sign under $\mathbf{r} \to -\mathbf{r}$ compared to E1.

### 4.3 Electric Quadrupole Radiation

For an oscillating linear quadrupole (three charges: $+q$ at $z = \pm d/2$, $-2q$ at origin):

$$P_{\text{E2}} = \frac{\mu_0 \omega^6 Q_0^2}{1440\pi c^3}$$

where $Q_0 = qd^2$ is the quadrupole moment amplitude.

```python
def multipole_radiation_patterns():
    """
    Compare radiation patterns of different multipole orders.

    Why compare: understanding the angular patterns helps identify
    the dominant multipole in experimental measurements (e.g., from
    the angular distribution of emitted radiation in atomic physics).
    """
    theta = np.linspace(0, 2 * np.pi, 500)

    # Power patterns for different multipoles
    E1 = np.sin(theta)**2                          # Electric dipole
    M1 = np.sin(theta)**2                          # Magnetic dipole (same shape!)
    E2_m0 = np.sin(theta)**2 * np.cos(theta)**2   # Electric quadrupole (m=0)
    E2_m1 = (1 + np.cos(theta)**2)**2 * np.sin(theta)**2 / 4  # E2 (m=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                              subplot_kw={'projection': 'polar'})

    patterns = [
        (E1, 'Electric Dipole (E1)', 'blue'),
        (M1, 'Magnetic Dipole (M1)', 'red'),
        (E2_m0, 'Electric Quadrupole E2 (m=0)', 'green'),
        (E2_m1, 'Electric Quadrupole E2 (m=1)', 'purple')
    ]

    for ax, (pattern, title, color) in zip(axes.flat, patterns):
        pattern_norm = pattern / pattern.max() if pattern.max() > 0 else pattern
        ax.plot(theta, pattern_norm, color=color, linewidth=2)
        ax.fill(theta, pattern_norm, alpha=0.2, color=color)
        ax.set_title(title, pad=15, fontsize=11)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    plt.suptitle('Multipole Radiation Patterns (Power)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("multipole_radiation_patterns.png", dpi=150)
    plt.show()

multipole_radiation_patterns()
```

---

## 5. Selection Rules

### 5.1 Parity Selection Rules

Each multipole has a definite parity under spatial inversion ($\mathbf{r} \to -\mathbf{r}$):

- **Electric multipole of order $\ell$**: parity $(-1)^\ell$
- **Magnetic multipole of order $\ell$**: parity $(-1)^{\ell+1}$

In quantum mechanics, transitions between states of definite parity require that the emitted photon carry away the appropriate parity:

$$\text{E1}: \Delta\ell = \pm 1, \quad \Delta m = 0, \pm 1, \quad \text{parity change}$$

$$\text{M1}: \Delta\ell = 0, \quad \Delta m = 0, \pm 1, \quad \text{no parity change}$$

$$\text{E2}: \Delta\ell = 0, \pm 2, \quad \Delta m = 0, \pm 1, \pm 2, \quad \text{no parity change}$$

### 5.2 Why Selection Rules Matter

- The $2s \to 1s$ transition in hydrogen is **forbidden** for E1 (same parity). It occurs via two-photon emission (very slow).
- The $2p \to 1s$ transition is **allowed** for E1 (parity change, $\Delta\ell = 1$). It has a lifetime of $\sim 1.6$ ns.
- Nuclear gamma-ray transitions often proceed through M1 or E2 because nuclear structure favors higher multipoles.

### 5.3 Relative Strengths

For atomic transitions ($d \sim a_0 \sim 0.5$ angstrom, $\lambda \sim 5000$ angstrom):

$$\frac{P_{\text{M1}}}{P_{\text{E1}}} \sim \left(\frac{v}{c}\right)^2 \sim \alpha^2 \approx 5 \times 10^{-5}$$

$$\frac{P_{\text{E2}}}{P_{\text{E1}}} \sim \left(\frac{d}{\lambda}\right)^2 \sim \left(\frac{a_0}{\lambda}\right)^2 \approx 10^{-7}$$

where $\alpha \approx 1/137$ is the fine-structure constant.

```python
def multipole_power_scaling():
    """
    Show how radiated power scales with frequency and source size
    for different multipole orders.

    Why scaling matters: it explains why E1 dominates in atomic physics,
    why nuclear transitions are often M1 or E2, and why radio antennas
    are almost always analyzed as E1 radiators.
    """
    # Ratio d/lambda
    d_over_lambda = np.logspace(-4, 0, 100)

    # Relative power (normalized to E1 = 1 at d/lambda = 0.01)
    E1 = np.ones_like(d_over_lambda)  # reference
    M1 = 0.1 * np.ones_like(d_over_lambda)  # suppressed by (v/c)^2 ~ 0.1 for illustration
    E2 = d_over_lambda**2 / d_over_lambda[0]**2
    M2 = 0.1 * d_over_lambda**2 / d_over_lambda[0]**2
    E3 = d_over_lambda**4 / d_over_lambda[0]**4

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(d_over_lambda, E1, 'b-', linewidth=2, label='E1 (electric dipole)')
    ax.loglog(d_over_lambda, M1, 'r--', linewidth=2, label='M1 (magnetic dipole)')
    ax.loglog(d_over_lambda, E2, 'g-.', linewidth=2, label='E2 (electric quadrupole)')
    ax.loglog(d_over_lambda, M2, 'm:', linewidth=2, label='M2 (magnetic quadrupole)')
    ax.loglog(d_over_lambda, E3, 'k--', linewidth=1.5, alpha=0.5,
              label='E3 (electric octupole)')

    # Mark typical regimes
    ax.axvspan(1e-4, 1e-2, alpha=0.1, color='blue', label='Atoms ($d/\\lambda \\sim 10^{-3}$)')
    ax.axvspan(1e-2, 1e-1, alpha=0.1, color='green', label='Nuclei ($d/\\lambda \\sim 10^{-2}$)')

    ax.set_xlabel('Source size / wavelength ($d / \\lambda$)')
    ax.set_ylabel('Relative radiated power')
    ax.set_title('Multipole Radiation Power Scaling')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(1e-10, 1e3)

    plt.tight_layout()
    plt.savefig("multipole_scaling.png", dpi=150)
    plt.show()

multipole_power_scaling()
```

---

## 6. Computing Multipole Moments

### 6.1 Example: Linear Quadrupole

Consider three charges on the $z$-axis: $+q$ at $z = +d$, $-2q$ at the origin, $+q$ at $z = -d$.

- Monopole: $Q = q - 2q + q = 0$
- Dipole: $\mathbf{p} = q(d\hat{z}) + (-2q)(0) + q(-d\hat{z}) = 0$
- Quadrupole: $Q_{zz} = q(3d^2 - d^2) + 0 + q(3d^2 - d^2) = 4qd^2$

The potential goes as $1/r^3$ — the first nonvanishing term.

### 6.2 Example: Square Quadrupole

Four charges at the corners of a square of side $a$ in the $xy$-plane: $+q$ at $(\pm a/2, a/2, 0)$ and $-q$ at $(\pm a/2, -a/2, 0)$.

- Monopole: $Q = 0$
- Dipole: $\mathbf{p} = 2qa\hat{y}$ (net dipole along $y$)

This configuration has a nonzero dipole moment, so the quadrupole contribution is subdominant.

```python
def compute_multipole_moments(charges, positions, max_ell=4):
    """
    Compute electric multipole moments q_{ell,m} for a discrete charge distribution.

    Why compute moments: they characterize the far-field behavior of
    any charge distribution. The moments are the "fingerprint" of the
    source as seen from a distance.
    """
    from scipy.special import sph_harm

    moments = {}

    for ell in range(max_ell + 1):
        for m in range(-ell, ell + 1):
            q_lm = 0 + 0j
            for q, pos in zip(charges, positions):
                r = np.linalg.norm(pos)
                if r < 1e-15:
                    # Y_l^m(0,0) is nonzero only for m=0
                    if m == 0:
                        theta_i, phi_i = 0, 0
                    else:
                        continue
                else:
                    theta_i = np.arccos(pos[2] / r)
                    phi_i = np.arctan2(pos[1], pos[0])

                Y_lm_conj = np.conj(sph_harm(m, ell, phi_i, theta_i))
                q_lm += q * r**ell * Y_lm_conj

            moments[(ell, m)] = q_lm

    return moments

# Linear quadrupole
d = 0.1  # meters
q = 1e-9  # coulombs

charges_lq = [q, -2*q, q]
positions_lq = [[0, 0, d], [0, 0, 0], [0, 0, -d]]

print("Linear Quadrupole (charges: +q, -2q, +q along z-axis)")
print("=" * 60)
moments_lq = compute_multipole_moments(charges_lq, positions_lq)
for (ell, m), val in sorted(moments_lq.items()):
    if abs(val) > 1e-25:
        print(f"  q({ell},{m:+d}) = {val:.4e}")

print("\n")

# Dipole
charges_dp = [q, -q]
positions_dp = [[0, 0, d/2], [0, 0, -d/2]]

print("Pure Dipole (charges: +q, -q along z-axis)")
print("=" * 60)
moments_dp = compute_multipole_moments(charges_dp, positions_dp)
for (ell, m), val in sorted(moments_dp.items()):
    if abs(val) > 1e-25:
        print(f"  q({ell},{m:+d}) = {val:.4e}")
```

---

## 7. Applications of the Multipole Expansion

### 7.1 Gravitational Multipoles

The multipole expansion applies to any $1/r$ field, including gravity. The Earth's gravitational potential is:

$$U = -\frac{GM}{r}\left[1 - \sum_{\ell=2}^{\infty} \left(\frac{R_E}{r}\right)^\ell J_\ell P_\ell(\cos\theta)\right]$$

The $J_2 \approx 1.08 \times 10^{-3}$ term (oblateness) causes satellite orbit precession.

### 7.2 Nuclear Multipole Moments

Nuclear charge distributions are characterized by their multipole moments:
- **Quadrupole moment**: Measures nuclear deformation. Prolate nuclei have $Q > 0$, oblate have $Q < 0$.
- **Magnetic dipole moment**: Related to nuclear spin and composition.

### 7.3 Antenna Design

Antenna engineers use multipole expansion to characterize antenna far-field patterns. The coefficients $a_{\ell m}$ and $b_{\ell m}$ (for E and M multipoles) completely determine the radiation pattern.

---

## Summary

| Concept | Key Formula | Physical Meaning |
|---------|-------------|------------------|
| Monopole potential | $V_0 = Q/(4\pi\epsilon_0 r)$ | Total charge; $\sim 1/r$ |
| Dipole potential | $V_1 = \mathbf{p}\cdot\hat{r}/(4\pi\epsilon_0 r^2)$ | Charge separation; $\sim 1/r^2$ |
| Quadrupole potential | $V_2 \sim Q_{ij}\hat{r}_i\hat{r}_j / r^3$ | Charge asymmetry; $\sim 1/r^3$ |
| Spherical harmonics | $q_{\ell m} = \int r'^\ell Y_\ell^{m*} \rho \, d^3r'$ | Angular decomposition of source |
| E1 radiation power | $P \propto \omega^4 p_0^2$ | Dominant for $d \ll \lambda$ |
| Power scaling | $P_\ell \propto (d/\lambda)^{2\ell}$ | Higher multipoles suppressed |
| E1 selection rule | $\Delta\ell = \pm 1$, parity change | Governs atomic transitions |

---

## Exercises

### Exercise 1: Quadrupole Potential Map
Create a 2D contour plot of the electrostatic potential from a linear quadrupole ($+q$, $-2q$, $+q$ along the $z$-axis with spacing $d$). Compare with the exact potential computed from Coulomb's law at distances $r = 2d, 5d, 10d$. At what distance does the quadrupole approximation become accurate to within 5%?

### Exercise 2: Multipole Moments of a Ring
A uniformly charged ring of radius $R$ and total charge $Q$ lies in the $xy$-plane. (a) Compute all multipole moments $q_{\ell 0}$ up to $\ell = 6$ (by symmetry, $q_{\ell m} = 0$ for $m \neq 0$). (b) Show that only even $\ell$ contribute. (c) Compare the exact potential on the axis with the multipole series truncated at $\ell = 4$.

### Exercise 3: Magnetic Dipole Radiation
An electron in a hydrogen atom transitions from the $2s$ to $1s$ state. Explain why E1 radiation is forbidden (both states have $\ell = 0$). Estimate the M1 transition rate relative to the $2p \to 1s$ E1 rate. What is the actual mechanism for $2s \to 1s$ decay?

### Exercise 4: E2 Radiation Pattern
Plot the full 3D radiation pattern of an electric quadrupole oscillating along the $z$-axis. Use the formula $dP/d\Omega \propto \sin^2\theta\cos^2\theta$ for $m = 0$. Compare with the E1 pattern and identify the null directions.

### Exercise 5: Earth's Gravitational Multipoles
The Earth has $J_2 = 1.0826 \times 10^{-3}$ and $J_4 = -1.62 \times 10^{-6}$. (a) Compute the gravitational potential at the surface at the equator and at the pole, including up to the $J_4$ term. (b) What is the geoid height difference between equator and pole? (c) How does $J_2$ affect the orbital precession rate of a low-Earth orbit satellite?

---

[← Previous: 14. Relativistic Electrodynamics](14_Relativistic_Electrodynamics.md) | [Next: 16. Computational Electrodynamics →](16_Computational_Electrodynamics.md)
