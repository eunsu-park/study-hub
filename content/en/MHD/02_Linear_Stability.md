# 2. Linear Stability Theory

## Learning Objectives

- Understand the linearization of ideal MHD equations around an equilibrium
- Derive the force operator and formulate the eigenvalue problem for normal modes
- Apply the energy principle to determine stability without solving eigenvalue problem
- Compute growth rates and stability boundaries for specific equilibria
- Understand the Kruskal-Shafranov criterion for external kink stability
- Apply the Suydam criterion for local interchange stability
- Implement numerical eigenvalue solvers for MHD stability analysis

## 1. Introduction to MHD Stability

An MHD equilibrium, while satisfying force balance, may be **unstable** to small perturbations. Stability analysis determines whether perturbations grow (unstable) or decay/oscillate (stable).

**Linear stability theory** examines the evolution of infinitesimal perturbations:

```
Equilibrium state: (p₀, ρ₀, B₀, v₀=0)
Perturbed state:   (p₀+p₁, ρ₀+ρ₁, B₀+B₁, v₁)

where |perturbations| << |equilibrium|
```

Key questions:
- Does a perturbation grow exponentially? (Unstable)
- Does it oscillate without growth? (Marginally stable)
- Does it decay? (Stable)

The **growth rate** $\gamma$ or **frequency** $\omega$ is determined by solving an eigenvalue problem.

## 2. Linearization of Ideal MHD

### 2.1 Perturbation Expansion

Consider a plasma displacement $\boldsymbol{\xi}(\mathbf{x}, t)$ from equilibrium. The Lagrangian displacement relates perturbed quantities:

$$
\mathbf{x}' = \mathbf{x} + \boldsymbol{\xi}(\mathbf{x}, t)
$$

Perturbed quantities (Eulerian description):

$$
\begin{aligned}
\rho_1 &= -\nabla\cdot(\rho_0\boldsymbol{\xi}) \\
\mathbf{v}_1 &= \frac{\partial\boldsymbol{\xi}}{\partial t} \\
\mathbf{B}_1 &= \nabla\times(\boldsymbol{\xi}\times\mathbf{B}_0) \\
p_1 &= -\boldsymbol{\xi}\cdot\nabla p_0 - \gamma p_0\nabla\cdot\boldsymbol{\xi}
\end{aligned}
$$

where $\gamma$ is the adiabatic index.

### 2.2 Linearized Momentum Equation

The ideal MHD momentum equation:

$$
\rho\frac{\partial\mathbf{v}}{\partial t} = -\nabla p + \mathbf{J}\times\mathbf{B}
$$

Linearizing (keeping only first-order terms):

$$
\rho_0\frac{\partial^2\boldsymbol{\xi}}{\partial t^2} = -\nabla p_1 + \mathbf{J}_1\times\mathbf{B}_0 + \mathbf{J}_0\times\mathbf{B}_1
$$

Using $\mathbf{J} = \nabla\times\mathbf{B}/\mu_0$:

$$
\rho_0\frac{\partial^2\boldsymbol{\xi}}{\partial t^2} = -\nabla p_1 + \frac{1}{\mu_0}(\nabla\times\mathbf{B}_1)\times\mathbf{B}_0 + \frac{1}{\mu_0}(\nabla\times\mathbf{B}_0)\times\mathbf{B}_1
$$

This can be written compactly as:

$$
\rho_0\frac{\partial^2\boldsymbol{\xi}}{\partial t^2} = \mathbf{F}(\boldsymbol{\xi})
$$

where $\mathbf{F}(\boldsymbol{\xi})$ is the **force operator** (linear in $\boldsymbol{\xi}$).

### 2.3 Normal Mode Analysis

Assume normal mode time dependence:

$$
\boldsymbol{\xi}(\mathbf{x}, t) = \hat{\boldsymbol{\xi}}(\mathbf{x})e^{-i\omega t}
$$

Substituting:

$$
-\omega^2\rho_0\hat{\boldsymbol{\xi}} = \mathbf{F}(\hat{\boldsymbol{\xi}})
$$

This is an **eigenvalue problem**:

$$
\mathbf{F}(\hat{\boldsymbol{\xi}}) = -\omega^2\rho_0\hat{\boldsymbol{\xi}}
$$

- Eigenvalue: $\omega^2$
- Eigenfunction: $\hat{\boldsymbol{\xi}}(\mathbf{x})$

**Stability criterion**:
- If all $\omega^2 > 0$: **stable** (oscillatory modes)
- If any $\omega^2 < 0$: **unstable** (exponential growth with $\gamma = \sqrt{-\omega^2}$)
- If $\omega^2 = 0$: **marginal stability**

### 2.4 Self-Adjoint Property of Force Operator

The force operator $\mathbf{F}$ has a crucial property: it is **self-adjoint** (Hermitian):

$$
\int \boldsymbol{\xi}_1^*\cdot\mathbf{F}(\boldsymbol{\xi}_2)\, dV = \int \boldsymbol{\xi}_2^*\cdot\mathbf{F}(\boldsymbol{\xi}_1)\, dV
$$

**Consequence**: All eigenvalues $\omega^2$ are **real**.

Proof: Consider eigenmode $\mathbf{F}(\hat{\boldsymbol{\xi}}) = -\omega^2\rho_0\hat{\boldsymbol{\xi}}$.

Taking inner product with $\hat{\boldsymbol{\xi}}^*$:

$$
\int \hat{\boldsymbol{\xi}}^*\cdot\mathbf{F}(\hat{\boldsymbol{\xi}})\, dV = -\omega^2\int \rho_0|\hat{\boldsymbol{\xi}}|^2\, dV
$$

If $\mathbf{F}$ is self-adjoint, the LHS is real, hence $\omega^2$ is real.

This means modes are either:
- **Oscillatory**: $\omega$ real, $\boldsymbol{\xi} \propto e^{-i\omega t}$
- **Growing/decaying**: $\omega$ imaginary, $\boldsymbol{\xi} \propto e^{\gamma t}$ with $\gamma = |\omega|$

## 3. Energy Principle

### 3.1 Potential Energy Functional

Rather than solving the eigenvalue problem directly, **Bernstein, Frieman, Kruskal, and Kulsrud (1958)** showed that stability can be determined from the sign of a potential energy functional.

Define the **perturbed potential energy**:

$$
\delta W = -\frac{1}{2}\int \boldsymbol{\xi}^*\cdot\mathbf{F}(\boldsymbol{\xi})\, dV
$$

From the eigenvalue equation:

$$
\delta W = \frac{1}{2}\omega^2\int \rho_0|\boldsymbol{\xi}|^2\, dV = \frac{1}{2}\omega^2 K
$$

where $K$ is the kinetic energy.

**Energy Principle**:
- If $\delta W > 0$ for all allowed $\boldsymbol{\xi}$: **stable** ($\omega^2 > 0$)
- If $\delta W < 0$ for some $\boldsymbol{\xi}$: **unstable** ($\omega^2 < 0$)
- If $\min(\delta W) = 0$: **marginal stability**

### 3.2 Explicit Form of $\delta W$

The potential energy can be decomposed:

$$
\delta W = \delta W_F + \delta W_S + \delta W_V
$$

where:
- $\delta W_F$: fluid (bulk) contribution
- $\delta W_S$: surface contribution
- $\delta W_V$: vacuum contribution

**Fluid contribution**:

$$
\delta W_F = \frac{1}{2}\int\left[\frac{|\mathbf{Q}|^2}{\mu_0} + \gamma p_0|\nabla\cdot\boldsymbol{\xi}|^2 + (\boldsymbol{\xi}\cdot\nabla p_0)(\nabla\cdot\boldsymbol{\xi}^*)\right]dV
$$

where:

$$
\mathbf{Q} = \nabla\times(\boldsymbol{\xi}\times\mathbf{B}_0) = \mathbf{B}_1
$$

is the perturbed magnetic field.

### 3.3 Physical Interpretation

Decompose $\delta W_F$ into physical contributions:

$$
\delta W_F = \delta W_{compression} + \delta W_{tension} + \delta W_{pressure}
$$

**Magnetic compression**:

$$
\delta W_{compression} = \frac{1}{2\mu_0}\int |\mathbf{B}_1|^2\, dV
$$

Compressing/stretching magnetic field lines costs energy (stabilizing).

**Magnetic tension**:

From $\mathbf{B}_1 = \nabla\times(\boldsymbol{\xi}\times\mathbf{B}_0)$, bending field lines creates restoring force.

**Pressure drive**:

Pressure gradient term can drive instability if curvature is unfavorable.

### 3.4 Intuitive Form

For incompressible perturbations ($\nabla\cdot\boldsymbol{\xi} = 0$):

$$
\delta W \approx \frac{1}{2}\int\left[\frac{|\nabla\times(\boldsymbol{\xi}\times\mathbf{B}_0)|^2}{\mu_0} + \boldsymbol{\xi}_\perp\cdot\nabla p_0\, \nabla\cdot\boldsymbol{\xi}_\perp\right]dV
$$

The first term (magnetic bending) is always positive (stabilizing).

The second term (pressure drive) can be negative if $\nabla p_0$ and $\nabla\cdot\boldsymbol{\xi}_\perp$ have the same sign.

**Unfavorable curvature**: Pressure gradient pointing toward convex field line curvature → unstable.

## 4. Kruskal-Shafranov Criterion

### 4.1 External Kink Instability

The **external kink** is a large-scale displacement of the entire plasma column. It is the dominant instability in toroidal confinement.

**Configuration**: Cylindrical plasma with helical perturbation:

$$
\boldsymbol{\xi} = \hat{\boldsymbol{\xi}}(r)e^{i(m\theta - nz/R_0)}
$$

where:
- $m$: poloidal mode number
- $n$: toroidal mode number (for toroidal systems)
- $(m,n) = (1,1)$ is the most dangerous mode

### 4.2 Derivation for Sharp Boundary

Consider a sharp-boundary model:
- Uniform current density inside $r < a$: $\mathbf{J} = J_0\hat{\mathbf{z}}$
- Vacuum outside $r > a$

The safety factor at the edge:

$$
q(a) = \frac{aB_z}{R_0 B_θ(a)}
$$

**Stability condition** (Kruskal and Shafranov, 1958):

$$
q(a) > \frac{m}{n}
$$

For the $(m,n) = (1,1)$ mode:

$$
q(a) > 1
$$

### 4.3 Physical Interpretation

The kink instability arises from the competition between:
- **Destabilizing**: Magnetic pressure imbalance when column bends
- **Stabilizing**: Axial field $B_z$ provides tension resisting bending

```
Kink Instability (m=1):
======================

Before:          After:
   │                ╱│╲
   │               ╱ │ ╲
   │     →        │  │  │  (column bends)
   │               ╲ │ ╱
   │                ╲│╱

Stabilization: B_z provides tension
               (higher B_z → higher q → stable)
```

For $q(a) < 1$, the field lines complete less than one toroidal turn per poloidal turn, and the configuration cannot resist the kink.

### 4.4 Generalization

For a diffuse current profile $J(r)$:

$$
q(a) > \frac{m}{n}\quad\text{(necessary but not sufficient)}
$$

Additional conditions involve current profile shape, pressure gradients, and conducting wall position.

## 5. Suydam Criterion

### 5.1 Local Interchange Instability

The **Suydam criterion** gives a **necessary condition** for stability against localized interchange modes.

**Interchange instability**: Adjacent flux tubes exchange positions driven by pressure gradient and magnetic curvature.

### 5.2 Derivation

Consider a cylindrical plasma in equilibrium. The Suydam criterion states:

$$
\frac{r}{4}\left(\frac{q'}{q}\right)^2 + \frac{2\mu_0 p'}{B_z^2} > 0
$$

for all $r$.

**Physical interpretation**:

- First term: Magnetic shear $q'/q$ stabilizes
- Second term: Pressure gradient can destabilize

If $p' < 0$ (pressure decreasing outward) and shear is weak, violation → instability.

### 5.3 Application

For a screw pinch with:
- $B_z = \text{const}$
- $B_θ(r) = B_{θ0}r/a$ (linear profile)

The safety factor:

$$
q(r) = \frac{rB_z}{R_0 B_θ} = \frac{B_z a}{R_0 B_{θ0}}\frac{1}{r}
$$

Thus:

$$
\frac{q'}{q} = -\frac{1}{r}
$$

Suydam criterion becomes:

$$
\frac{1}{4r^2} + \frac{2\mu_0 p'}{B_z^2} > 0
$$

If $|p'|$ is too large, the criterion is violated.

### 5.4 Limitation

Suydam criterion is **necessary** but **not sufficient**. Violating it guarantees instability, but satisfying it does not guarantee stability (global modes may still exist).

## 6. Growth Rate Calculation

### 6.1 Analytic Growth Rate for Simple Geometry

For a sharp-boundary Z-pinch with $m=0$ (sausage mode), the growth rate is:

$$
\gamma^2 = \frac{k_z^2 B_θ^2(a)}{\mu_0\rho_0}
$$

where $k_z$ is the axial wavenumber.

**Interpretation**: No stabilization → always unstable for $k_z \neq 0$.

Adding axial field $B_z$ stabilizes short wavelengths:

$$
\gamma^2 = \frac{B_θ^2(a)}{\mu_0\rho_0}\left(k_z^2 - \frac{B_z^2}{B_θ^2(a)}k_z^2\right)
$$

Critical wavenumber for stabilization:

$$
k_z > k_c = \frac{B_θ(a)}{B_z}
$$

### 6.2 Eigenvalue Problem Formulation

For general geometry, the eigenvalue problem:

$$
\mathbf{F}(\hat{\boldsymbol{\xi}}) = -\omega^2\rho_0\hat{\boldsymbol{\xi}}
$$

must be discretized and solved numerically.

**Method**:
1. Choose basis functions or grid
2. Project $\mathbf{F}$ onto discrete space → matrix $\mathbf{A}$
3. Solve matrix eigenvalue problem: $\mathbf{A}\mathbf{x} = \lambda\mathbf{x}$
4. Eigenvalue $\lambda = -\omega^2$
5. If $\lambda > 0$: unstable with $\gamma = \sqrt{\lambda}$

## 7. Numerical Implementation

### 7.1 Finite Element Discretization

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

class MHDStabilitySolver:
    """
    Solve MHD stability eigenvalue problem for cylindrical geometry
    """

    def __init__(self, nr, r_max, equilibrium):
        """
        nr: number of radial grid points
        r_max: maximum radius
        equilibrium: dict with Bz(r), Btheta(r), p(r), rho(r)
        """
        self.nr = nr
        self.r_max = r_max
        self.r = np.linspace(0, r_max, nr)
        self.dr = self.r[1] - self.r[0]

        self.equilibrium = equilibrium
        self.mu0 = 4*np.pi*1e-7

    def compute_force_operator(self, m, kz):
        """
        Compute force operator matrix for mode (m, kz)
        Simplified 1D radial eigenvalue problem
        """
        nr = self.nr
        r = self.r

        # Get equilibrium profiles
        Bz = np.array([self.equilibrium['Bz'](ri) for ri in r])
        Btheta = np.array([self.equilibrium['Btheta'](ri) for ri in r])
        p = np.array([self.equilibrium['p'](ri) for ri in r])
        rho = np.array([self.equilibrium['rho'](ri) for ri in r])

        # Build matrix (simplified model)
        # Full MHD stability is very complex; here we implement a toy model

        A = np.zeros((nr, nr))

        for i in range(1, nr-1):
            # The Laplacian structure d²ξ/dr² + (1/r)dξ/dr - (m²/r²)ξ
            # comes from projecting the force operator onto the m-th Fourier
            # harmonic; the -m²/r² term is the centrifugal cost of bending
            # the displacement around the cylinder in the azimuthal direction.

            # The magnetic tension coefficient B²/(μ₀ρ) equals v_A² (Alfvén
            # speed squared): field-line bending creates a restoring force
            # proportional to v_A², which stabilizes high-m short-wavelength
            # perturbations — this is the fundamental stabilizing mechanism.
            tension_coef = (Bz[i]**2 + Btheta[i]**2) / (self.mu0 * rho[i])

            # Diagonal
            A[i, i] = -2*tension_coef/self.dr**2 - m**2*tension_coef/r[i]**2

            # The asymmetric off-diagonal term tension/(2r dr) comes from the
            # cylindrical (1/r)dξ/dr part of the Laplacian; it is first-order
            # and vanishes in Cartesian geometry where r → ∞.
            A[i, i+1] = tension_coef/self.dr**2 + tension_coef/(2*r[i]*self.dr)
            A[i, i-1] = tension_coef/self.dr**2 - tension_coef/(2*r[i]*self.dr)

            # The pressure gradient term -∇p/ρ acts as a destabilizing drive:
            # if pressure decreases outward (dp/dr < 0), this term is positive
            # in A, reducing the restoring force and potentially driving growth.
            if i > 0:
                dpdx = (p[i+1] - p[i-1]) / (2*self.dr)
                A[i, i] += -dpdx / rho[i]

        # Dirichlet conditions ξ(0) = ξ(r_max) = 0 enforce regularity at the
        # axis (no singular displacement) and no perturbation at the boundary
        # (plasma surface is held fixed in this simplified model).
        A[0, 0] = 1.0
        A[-1, -1] = 1.0

        return A

    def solve_stability(self, m, kz):
        """
        Solve eigenvalue problem for mode (m, kz)
        Returns: eigenvalues (growth rates squared), eigenvectors
        """
        A = self.compute_force_operator(m, kz)

        # scipy.linalg.eigh is used instead of eig because the force operator
        # F is self-adjoint (Hermitian), guaranteeing real eigenvalues ω².
        # eigh exploits this symmetry for faster, more numerically stable
        # computation compared to the general complex eigensolver.
        # A ξ = λ ξ, where λ = -ω²
        eigenvalues, eigenvectors = eigh(A)

        # λ > 0 means ω² < 0, i.e., ω is purely imaginary → exponential growth.
        # The sign convention (λ = -ω²) comes from the eigenvalue equation
        # F(ξ) = -ω²ρξ: a positive definite F (restoring force) gives λ < 0
        # (stable oscillation), while a negative semidefinite F gives λ > 0 (unstable).
        growth_rates_squared = eigenvalues

        return growth_rates_squared, eigenvectors

    def stability_scan(self, m_values, kz_values):
        """
        Scan stability over mode numbers
        Returns: growth rate map
        """
        growth_map = np.zeros((len(m_values), len(kz_values)))

        for i, m in enumerate(m_values):
            for j, kz in enumerate(kz_values):
                eigenvalues, _ = self.solve_stability(m, kz)

                # Maximum growth rate for this mode
                max_growth_sq = np.max(eigenvalues)
                growth_map[i, j] = np.sqrt(max(max_growth_sq, 0))

        return growth_map

    def plot_growth_rate(self, m_values, kz_values, growth_map):
        """Plot growth rate map"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Contour plot
        KZ, M = np.meshgrid(kz_values, m_values)
        levels = 20

        CS = ax.contourf(KZ, M, growth_map, levels=levels, cmap='hot')
        ax.contour(KZ, M, growth_map, levels=[0], colors='cyan', linewidths=2)

        cbar = plt.colorbar(CS, ax=ax)
        cbar.set_label('Growth rate γ [1/s]', fontsize=12)

        ax.set_xlabel('Axial wavenumber kz [1/m]', fontsize=12)
        ax.set_ylabel('Poloidal mode number m', fontsize=12)
        ax.set_title('MHD Instability Growth Rate Map', fontsize=14)

        plt.tight_layout()
        return fig

    def plot_eigenmode(self, m, kz, mode_index=0):
        """Plot eigenmode structure"""
        eigenvalues, eigenvectors = self.solve_stability(m, kz)

        # Sort by eigenvalue (most unstable first)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select mode
        eigenvalue = eigenvalues[mode_index]
        eigenmode = eigenvectors[:, mode_index]

        gamma = np.sqrt(max(eigenvalue, 0))

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.r, eigenmode.real, 'b-', linewidth=2, label='Real part')
        ax.plot(self.r, eigenmode.imag, 'r--', linewidth=2, label='Imaginary part')

        ax.set_xlabel('Radius r [m]', fontsize=12)
        ax.set_ylabel('Displacement ξ (normalized)', fontsize=12)
        ax.set_title(f'Eigenmode (m={m}, kz={kz:.2f}): γ = {gamma:.2e} s⁻¹', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

# Example: Z-pinch stability
def example_zpinch_stability():
    """Analyze stability of a Z-pinch"""

    # Equilibrium parameters
    a = 0.1  # Plasma radius [m]
    I = 1e6  # Current [A]
    n = 1e20 # Density [m^-3]
    T = 1e6  # Temperature [K]
    mu0 = 4*np.pi*1e-7
    kB = 1.38e-23
    mp = 1.67e-27

    # Equilibrium profiles
    def Bz(r):
        return 0.5  # Weak axial field [T]

    def Btheta(r):
        # Enclosed current
        if r < a:
            I_enc = I * (r/a)**2
        else:
            I_enc = I
        return mu0 * I_enc / (2*np.pi*r) if r > 0 else 0

    def p(r):
        # Parabolic pressure
        p0 = 2 * n * kB * T
        if r < a:
            return p0 * (1 - (r/a)**2)
        else:
            return 0

    def rho(r):
        # Uniform density
        return n * mp if r < a else 0.01 * n * mp

    equilibrium = {
        'Bz': Bz,
        'Btheta': Btheta,
        'p': p,
        'rho': rho
    }

    # Solver
    solver = MHDStabilitySolver(nr=100, r_max=2*a, equilibrium=equilibrium)

    print("=== Z-Pinch Stability Analysis ===")
    print(f"Plasma radius: {a*100} cm")
    print(f"Current: {I/1e6} MA")
    print(f"Density: {n:.2e} m^-3")
    print(f"Temperature: {T/1e6} MK")
    print(f"Axial field: {Bz(0)} T")
    print(f"Azimuthal field at edge: {Btheta(a):.3f} T")

    # Stability scan
    m_values = [0, 1, 2, 3]  # Poloidal mode numbers
    kz_values = np.linspace(1, 100, 50)  # Axial wavenumbers [1/m]

    print("\nScanning stability...")
    growth_map = solver.stability_scan(m_values, kz_values)

    # Find most unstable mode
    max_idx = np.unravel_index(np.argmax(growth_map), growth_map.shape)
    max_growth = growth_map[max_idx]
    m_unstable = m_values[max_idx[0]]
    kz_unstable = kz_values[max_idx[1]]

    print(f"\nMost unstable mode: m={m_unstable}, kz={kz_unstable:.1f} m^-1")
    print(f"Growth rate: γ = {max_growth:.2e} s^-1")
    print(f"Growth time: τ = {1/max_growth:.2e} s")

    # Plot growth rate map
    fig1 = solver.plot_growth_rate(m_values, kz_values, growth_map)
    plt.savefig('/tmp/zpinch_growth_map.png', dpi=150)
    print("\nGrowth rate map saved to /tmp/zpinch_growth_map.png")

    # Plot unstable eigenmode
    fig2 = solver.plot_eigenmode(m_unstable, kz_unstable, mode_index=0)
    plt.savefig('/tmp/zpinch_eigenmode.png', dpi=150)
    print("Eigenmode structure saved to /tmp/zpinch_eigenmode.png")

    plt.close('all')

if __name__ == "__main__":
    example_zpinch_stability()
```

### 7.2 Safety Factor Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_q_profile(r, Bz, Btheta, R0):
    """
    Compute safety factor q(r) = r*Bz / (R0*Btheta)
    """
    q = r * Bz / (R0 * Btheta + 1e-10)  # Avoid division by zero
    return q

def check_kruskal_shafranov(q_edge, m, n):
    """
    Check Kruskal-Shafranov criterion: q(a) > m/n
    """
    q_crit = m / n
    margin = q_edge - q_crit

    stable = margin > 0

    return stable, margin, q_crit

def plot_q_and_stability(r, q, R0, a):
    """
    Plot q-profile with stability boundaries
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # q-profile
    ax.plot(r/a, q, 'b-', linewidth=2, label='q(r)')

    # Rational surfaces
    rational_q = [1, 1.5, 2, 2.5, 3]
    colors = ['red', 'orange', 'green', 'cyan', 'magenta']

    for q_val, color in zip(rational_q, colors):
        ax.axhline(y=q_val, color=color, linestyle='--', alpha=0.7,
                   label=f'q = {q_val}')

        # Find radial location
        idx = np.argmin(np.abs(q - q_val))
        if idx > 0 and idx < len(r)-1:
            r_res = r[idx] / a
            ax.plot(r_res, q_val, 'o', color=color, markersize=8)

    ax.set_xlabel('r/a (normalized radius)', fontsize=12)
    ax.set_ylabel('Safety factor q', fontsize=12)
    ax.set_title('Safety Factor Profile and Rational Surfaces', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim([0, max(q)*1.1])

    plt.tight_layout()
    return fig

def example_tokamak_q_profile():
    """
    Compute and analyze q-profile for a tokamak
    """
    # Tokamak parameters
    R0 = 3.0  # Major radius [m]
    a = 1.0   # Minor radius [m]
    Bt0 = 5.0 # Toroidal field [T]
    Ip = 2e6  # Plasma current [A]

    mu0 = 4*np.pi*1e-7

    # Radial grid
    r = np.linspace(0.01, a, 200)

    # Current density profile (parabolic)
    alpha = 2.0
    j0 = Ip / (np.pi * a**2 * (1 - 1/(alpha+1)))

    def j_profile(r_val):
        return j0 * (1 - (r_val/a)**alpha)

    # Enclosed current
    def I_enclosed(r_val):
        # Integrate j(r') from 0 to r
        from scipy.integrate import quad
        result, _ = quad(lambda rp: j_profile(rp) * 2*np.pi*rp, 0, r_val)
        return result

    # Poloidal field
    I_enc = np.array([I_enclosed(ri) for ri in r])
    Btheta = mu0 * I_enc / (2*np.pi*r)

    # Toroidal field (1/R dependence)
    Bz = Bt0 * R0 / (R0 + r)

    # Safety factor
    q = compute_q_profile(r, Bz, Btheta, R0)

    print("=== Tokamak q-Profile Analysis ===")
    print(f"Major radius R0 = {R0} m")
    print(f"Minor radius a = {a} m")
    print(f"Aspect ratio A = {R0/a}")
    print(f"Toroidal field Bt0 = {Bt0} T")
    print(f"Plasma current Ip = {Ip/1e6} MA")

    print(f"\nq(0) ≈ {q[0]:.2f}")
    print(f"q(a) = {q[-1]:.2f}")

    # Check Kruskal-Shafranov for (1,1) mode
    stable_11, margin_11, q_crit_11 = check_kruskal_shafranov(q[-1], 1, 1)

    print(f"\nKruskal-Shafranov (m=1, n=1):")
    print(f"  Critical q: {q_crit_11}")
    print(f"  Edge q: {q[-1]:.2f}")
    print(f"  Margin: {margin_11:.2f}")
    print(f"  Status: {'STABLE' if stable_11 else 'UNSTABLE'}")

    # Check for other modes
    modes = [(1, 1), (2, 1), (3, 1), (3, 2)]
    print("\nStability check for resonant modes:")
    for m, n in modes:
        stable, margin, q_crit = check_kruskal_shafranov(q[-1], m, n)
        status = "✓ STABLE" if stable else "✗ UNSTABLE"
        print(f"  (m={m}, n={n}): q_crit={q_crit:.2f}, margin={margin:+.2f} → {status}")

    # Find rational surfaces
    print("\nRational surface locations (r/a):")
    for q_rational in [1, 2, 3]:
        idx = np.argmin(np.abs(q - q_rational))
        if q[idx] > 0.8*q_rational and q[idx] < 1.2*q_rational:
            print(f"  q = {q_rational}: r/a ≈ {r[idx]/a:.3f}")
        else:
            print(f"  q = {q_rational}: not found in plasma")

    # Plot
    fig = plot_q_and_stability(r, q, R0, a)
    plt.savefig('/tmp/tokamak_q_profile.png', dpi=150)
    print("\nq-profile plot saved to /tmp/tokamak_q_profile.png")
    plt.close()

if __name__ == "__main__":
    example_tokamak_q_profile()
```

### 7.3 Energy Principle Calculation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

def compute_delta_W(r, xi_r, xi_theta, Bz, Btheta, p, rho, m, kz, gamma_adiabatic=5/3):
    """
    Compute potential energy δW for a given displacement

    Parameters:
    -----------
    r: radial grid
    xi_r, xi_theta: displacement components
    Bz, Btheta: equilibrium fields
    p: pressure
    rho: density
    m: poloidal mode number
    kz: axial wavenumber
    gamma_adiabatic: adiabatic index

    Returns:
    --------
    delta_W: potential energy
    """
    mu0 = 4*np.pi*1e-7

    # Perturbed magnetic field (simplified)
    # B1_r ~ -ikz * xi_r * Bz + (im/r) * xi_theta * Bz
    # This is a simplified model; full calculation is complex

    # div_xi measures how much the fluid compresses: a non-zero divergence
    # costs energy through both magnetic compression and acoustic compression,
    # so the most dangerous (unstable) perturbations tend to be incompressible
    # (div_xi → 0) since they avoid these stabilizing terms.
    dxi_r_dr = np.gradient(xi_r, r)
    div_xi = dxi_r_dr + xi_r/r + (1j*m/r)*xi_theta

    # The kz*Bz and (m/r)*Bz terms capture field-line bending: displacing the
    # plasma along kz or around the m-th azimuthal harmonic stretches and
    # bends field lines, doing positive work (B1_perp_sq > 0), which is
    # always stabilizing — this is why B_z makes Z-pinches more stable.
    B1_perp_sq = np.abs((kz*Bz)**2 * xi_r**2 + (m*Bz/r)**2 * xi_theta**2)

    delta_W_magnetic = 0.5 * simps(B1_perp_sq / mu0 * 2*np.pi*r, r)

    # Adiabatic compression δW_p = (γp/2)|div_ξ|² integrates the acoustic
    # restoring force: compressing the plasma adiabatically raises its pressure,
    # opposing further compression.  This term is always positive (stabilizing).
    delta_W_pressure = 0.5 * simps(gamma_adiabatic * p * np.abs(div_xi)**2 * 2*np.pi*r, r)

    # The pressure gradient drive ξ_r (dp/dr) div_ξ can be negative when
    # dp/dr < 0 (pressure decreasing outward) and the displacement is outward
    # (ξ_r > 0 with div_ξ > 0): this is the interchange/ballooning mechanism
    # where low-pressure plasma flows outward, releasing free energy.
    dp_dr = np.gradient(p, r)
    delta_W_drive = simps(xi_r * dp_dr * div_xi.real * 2*np.pi*r, r)

    delta_W = delta_W_magnetic + delta_W_pressure + delta_W_drive

    return delta_W

def example_energy_principle():
    """
    Use energy principle to assess stability
    """
    # Set up equilibrium (similar to previous examples)
    a = 0.1
    r = np.linspace(0.01, a, 100)

    # Fields
    Bz0 = 1.0
    Btheta0 = 0.5

    Bz = Bz0 * np.ones_like(r)
    Btheta = Btheta0 * r / a

    # Pressure
    p0 = 1e5
    p = p0 * (1 - (r/a)**2)**2

    # Density
    rho = 1e-3 * np.ones_like(r)

    # Test displacement (trial function)
    # Try a simple form: xi_r ~ sin(πr/a)
    xi_r = np.sin(np.pi * r / a)
    xi_theta = np.zeros_like(r)

    # Mode numbers
    m = 1
    kz = 10  # [1/m]

    # Compute δW
    delta_W = compute_delta_W(r, xi_r, xi_theta, Bz, Btheta, p, rho, m, kz)

    print("=== Energy Principle Stability Test ===")
    print(f"Mode: m={m}, kz={kz} m^-1")
    print(f"Trial displacement: ξ_r ~ sin(πr/a)")
    print(f"\nPotential energy: δW = {delta_W:.3e} J")

    if delta_W > 0:
        print("Result: STABLE (δW > 0)")
    elif delta_W < 0:
        print("Result: UNSTABLE (δW < 0)")
    else:
        print("Result: MARGINAL (δW = 0)")

    # Try multiple trial functions
    print("\n=== Testing Multiple Trial Functions ===")

    trial_functions = {
        'sin(πr/a)': lambda r: np.sin(np.pi*r/a),
        'sin(2πr/a)': lambda r: np.sin(2*np.pi*r/a),
        '(r/a)(1-r/a)': lambda r: (r/a)*(1-r/a),
        '(1-(r/a)²)': lambda r: 1-(r/a)**2,
    }

    results = []

    for name, func in trial_functions.items():
        xi_r_trial = func(r)
        xi_theta_trial = np.zeros_like(r)

        dW = compute_delta_W(r, xi_r_trial, xi_theta_trial, Bz, Btheta, p, rho, m, kz)

        results.append((name, dW))
        status = "STABLE" if dW > 0 else "UNSTABLE"
        print(f"  {name:20s}: δW = {dW:+.3e} J → {status}")

    # Find most dangerous (minimum δW)
    min_idx = np.argmin([dW for _, dW in results])
    min_name, min_dW = results[min_idx]

    print(f"\nMost dangerous trial function: {min_name}")
    print(f"Minimum δW: {min_dW:.3e} J")

    if min_dW < 0:
        print("⚠ UNSTABLE configuration detected!")
    else:
        print("✓ Stable for all tested trial functions")

if __name__ == "__main__":
    example_energy_principle()
```

## 8. Advanced Topics

### 8.1 Resistive Instabilities

In **resistive MHD**, finite resistivity allows magnetic reconnection. The ideal MHD stability analysis must be modified.

**Tearing mode**: Reconnection at rational surfaces where $q = m/n$ leads to magnetic island formation (covered in Lesson 4).

### 8.2 Kinetic Effects

MHD assumes fluid approximation. For wavelengths comparable to particle gyroradii, kinetic effects (Landau damping, wave-particle resonances) modify stability.

### 8.3 Conducting Wall Stabilization

Placing a conducting wall close to the plasma can stabilize external kink modes by inducing image currents.

**With wall**: modes grow on resistive wall timescale (slow)
**Without wall**: modes grow on Alfvén timescale (fast)

## Summary

In this lesson, we have developed the theory of MHD linear stability:

1. **Linearization**: Perturbation analysis leads to eigenvalue problem $\mathbf{F}(\hat{\boldsymbol{\xi}}) = -\omega^2\rho_0\hat{\boldsymbol{\xi}}$ with self-adjoint force operator.

2. **Energy Principle**: Stability determined by sign of potential energy $\delta W$ without solving eigenvalue problem. Decomposition into magnetic compression, tension, and pressure drive.

3. **Kruskal-Shafranov Criterion**: External kink stability requires $q(a) > m/n$. For tokamaks, $q(a) > 1$ is necessary.

4. **Suydam Criterion**: Local interchange stability requires sufficient magnetic shear to overcome pressure gradient drive.

5. **Growth Rates**: Numerical eigenvalue solvers compute growth rates and mode structures.

6. **Numerical Implementation**: Discretization of force operator, eigenvalue solvers, and energy principle calculations.

These tools form the basis for understanding and predicting MHD instabilities in fusion plasmas, which constrain operating regimes and motivate control strategies.

## Practice Problems

### Problem 1: Energy Principle for Sausage Mode

Consider a Z-pinch with uniform current density and sharp boundary at $r=a$. The equilibrium has:
- $B_θ(r) = \frac{\mu_0 I r}{2\pi a^2}$ for $r < a$
- $B_θ(r) = \frac{\mu_0 I}{2\pi r}$ for $r > a$
- $p(r) = p_0$ (constant) for $r < a$

For the $m=0$ (sausage) mode with trial displacement $\xi_r = \xi_0\sin(\pi r/a)$, $\xi_θ = \xi_z = 0$:

**(a)** Compute $\nabla\cdot\boldsymbol{\xi}$.

**(b)** Estimate the perturbed magnetic field $\mathbf{B}_1$.

**(c)** Calculate the magnetic compression energy $\delta W_{mag} = \frac{1}{2\mu_0}\int |\mathbf{B}_1|^2 dV$.

**(d)** Calculate the pressure compression energy $\delta W_p = \frac{\gamma}{2}\int p_0 |\nabla\cdot\boldsymbol{\xi}|^2 dV$.

**(e)** Determine if $\delta W > 0$ (stable) or $\delta W < 0$ (unstable).

### Problem 2: Kruskal-Shafranov for Tokamak

A tokamak has:
- Major radius $R_0 = 2$ m
- Minor radius $a = 0.5$ m
- Toroidal field $B_t = 4$ T (constant)
- Current density $J_z(r) = J_0(1 - r^2/a^2)$

**(a)** Find the total plasma current $I_p = \int J_z(r) 2\pi r\, dr$.

**(b)** Compute the poloidal field $B_θ(r) = \mu_0 I(r)/(2\pi r)$ where $I(r)$ is enclosed current.

**(c)** Calculate the safety factor $q(r) = rB_t/(R_0 B_θ(r))$.

**(d)** Evaluate $q(0)$ (on-axis, using l'Hôpital's rule) and $q(a)$ (edge).

**(e)** Check the Kruskal-Shafranov criterion for $(m,n) = (1,1)$ mode. Is the configuration stable?

**(f)** What minimum edge current is required to achieve $q(a) = 3$?

### Problem 3: Suydam Criterion Application

A screw pinch has:
- $B_z = 1$ T (constant)
- $B_θ(r) = B_{θ0}(r/a)$ (linear)
- $p(r) = p_0(1 - r^2/a^2)$
- Major radius $R_0 = 10a$

**(a)** Compute the safety factor $q(r)$ and its derivative $q'(r)$.

**(b)** Compute the pressure gradient $p'(r)$.

**(c)** Evaluate the Suydam criterion:
$$
\frac{r}{4}\left(\frac{q'}{q}\right)^2 + \frac{2\mu_0 p'}{B_z^2}
$$
at $r = a/2$.

**(d)** Determine the maximum allowed $p_0$ for Suydam stability at all radii.

**(e)** What happens if $p_0$ exceeds this limit?

### Problem 4: Growth Rate Estimate

For an unstable mode with $\delta W < 0$, the growth rate can be estimated by:

$$
\gamma^2 \sim \frac{|\delta W|}{K}
$$

where $K = \frac{1}{2}\int\rho_0|\boldsymbol{\xi}|^2 dV$ is the kinetic energy of the perturbation.

Consider a cylindrical plasma with radius $a = 0.1$ m, length $L = 1$ m, density $\rho_0 = 10^{-6}$ kg/m³.

The displacement is $\boldsymbol{\xi} = \xi_0\sin(\pi r/a)\hat{\mathbf{r}}$ with $\xi_0 = 0.01$ m.

From energy principle calculation: $\delta W = -10^3$ J.

**(a)** Compute the kinetic energy $K$.

**(b)** Estimate the growth rate $\gamma$.

**(c)** Calculate the growth time $\tau = 1/\gamma$.

**(d)** If the Alfvén speed is $v_A = 10^6$ m/s, compare $\gamma$ to the Alfvén frequency $\omega_A = v_A/a$.

**(e)** Is this a fast (Alfvén timescale) or slow instability?

### Problem 5: Eigenvalue Problem Setup

Set up (but do not solve) the eigenvalue problem for a cylindrical plasma with:
- $B_z = B_0 = \text{const}$
- $B_θ(r) = 0$
- $p(r) = p_0(1-r^2/a^2)$
- $\rho = \rho_0 = \text{const}$

For an $m=1$ perturbation $\boldsymbol{\xi} = \xi_r(r)e^{i(\theta - kz z)}\hat{\mathbf{r}} + \xi_θ(r)e^{i(\theta - kz z)}\hat{\boldsymbol{\theta}}$:

**(a)** Write the linearized momentum equation in component form.

**(b)** Express $\mathbf{B}_1$ in terms of $\xi_r, \xi_θ$.

**(c)** Derive coupled ODEs for $\xi_r(r)$ and $\xi_θ(r)$.

**(d)** What are the boundary conditions at $r=0$ and $r=a$?

**(e)** How would you discretize this system for numerical solution?

---

**Previous**: [MHD Equilibria](./01_MHD_Equilibria.md) | **Next**: [Pressure-Driven Instabilities](./03_Pressure_Driven_Instabilities.md)
