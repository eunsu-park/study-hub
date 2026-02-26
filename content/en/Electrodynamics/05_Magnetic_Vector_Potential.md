# Magnetic Vector Potential

[← Previous: 04. Magnetostatics](04_Magnetostatics.md) | [Next: 06. Electromagnetic Induction →](06_Electromagnetic_Induction.md)

---

## Learning Objectives

1. Define the magnetic vector potential $\mathbf{A}$ and show that $\mathbf{B} = \nabla \times \mathbf{A}$
2. Explain gauge freedom and the physical significance of choosing a gauge
3. Apply the Coulomb gauge to simplify magnetostatic calculations
4. Calculate the vector potential of standard current distributions (wire, solenoid)
5. Describe the Aharonov-Bohm effect and its implications for the reality of potentials
6. Develop the multipole expansion of the vector potential
7. Compute vector potentials numerically and verify $\mathbf{B} = \nabla \times \mathbf{A}$

---

In electrostatics, the scalar potential $V$ was a computational convenience — easier to compute than $\mathbf{E}$ because it is a scalar rather than a vector. The magnetic vector potential $\mathbf{A}$ plays an analogous role in magnetostatics, though it is itself a vector. Yet $\mathbf{A}$ is more than a mathematical trick. Quantum mechanics reveals that $\mathbf{A}$ has direct physical significance: in the Aharonov-Bohm effect, the potential affects a charged particle even in regions where $\mathbf{B} = 0$. This lesson develops the vector potential formalism, explores gauge freedom, and shows how $\mathbf{A}$ encodes the physics of magnetic fields. Along the way, we encounter one of the most beautiful results in physics: the Aharonov-Bohm effect, which shows that the potential is not merely a mathematical convenience but a physical entity in its own right.

---

## Why a Vector Potential?

Since $\nabla \cdot \mathbf{B} = 0$ (no magnetic monopoles), the divergence theorem guarantees that $\mathbf{B}$ can be written as the curl of some vector field:

$$\boxed{\mathbf{B} = \nabla \times \mathbf{A}}$$

This is guaranteed by a theorem in vector calculus: a divergence-free vector field can always be expressed as a curl.

Compare with electrostatics:
- $\nabla \times \mathbf{E} = 0 \implies \mathbf{E} = -\nabla V$ (curl-free $\implies$ gradient of scalar)
- $\nabla \cdot \mathbf{B} = 0 \implies \mathbf{B} = \nabla \times \mathbf{A}$ (divergence-free $\implies$ curl of vector)

The mathematical theorems behind these statements are the **Helmholtz decomposition** (any vector field can be decomposed into a curl-free part and a divergence-free part) and the **Poincare lemma** (a closed form is locally exact).

### The Equation for A

Substituting $\mathbf{B} = \nabla \times \mathbf{A}$ into Ampere's law $\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$:

$$\nabla \times (\nabla \times \mathbf{A}) = \mu_0 \mathbf{J}$$

Using the vector identity $\nabla \times (\nabla \times \mathbf{A}) = \nabla(\nabla \cdot \mathbf{A}) - \nabla^2 \mathbf{A}$:

$$\nabla(\nabla \cdot \mathbf{A}) - \nabla^2 \mathbf{A} = \mu_0 \mathbf{J}$$

This is more complex than we would like — but gauge freedom comes to the rescue.

---

## Gauge Freedom

If $\mathbf{A}$ gives the correct $\mathbf{B}$, so does $\mathbf{A}' = \mathbf{A} + \nabla \lambda$ for any scalar function $\lambda$, because:

$$\nabla \times \mathbf{A}' = \nabla \times \mathbf{A} + \nabla \times (\nabla \lambda) = \nabla \times \mathbf{A} = \mathbf{B}$$

(The curl of a gradient is always zero.)

This means $\mathbf{A}$ is not unique — it is defined only up to the gradient of an arbitrary scalar. This ambiguity is called **gauge freedom**, and choosing a specific $\lambda$ is called **fixing the gauge**.

> **Analogy**: Gauge freedom is like choosing the zero of altitude. Whether you measure heights above sea level or above your kitchen floor, all differences in altitude (the physically meaningful quantity) remain the same. The absolute "potential height" is ambiguous, but relative heights (analogous to $\mathbf{B} = \nabla \times \mathbf{A}$) are unambiguous.

### Coulomb Gauge

The most common choice in magnetostatics is the **Coulomb gauge**:

$$\nabla \cdot \mathbf{A} = 0$$

With this choice, the equation for $\mathbf{A}$ simplifies beautifully:

$$-\nabla^2 \mathbf{A} = \mu_0 \mathbf{J}$$

This is three copies of Poisson's equation — one for each component of $\mathbf{A}$! The solution (by analogy with $V = \frac{1}{4\pi\epsilon_0}\int \frac{\rho}{r}\,d\tau'$) is:

$$\boxed{\mathbf{A}(\mathbf{r}) = \frac{\mu_0}{4\pi} \int \frac{\mathbf{J}(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d\tau'}$$

For a line current:

$$\mathbf{A}(\mathbf{r}) = \frac{\mu_0 I}{4\pi} \int \frac{d\mathbf{l}'}{|\mathbf{r} - \mathbf{r}'|}$$

Note the contrast with Biot-Savart:
- Biot-Savart has a cross product and $1/r^2$ — harder to compute
- The vector potential formula has no cross product and $1/r$ — easier to integrate

---

## Vector Potential of an Infinite Straight Wire

For an infinite straight wire carrying current $I$ along the z-axis:

$$\mathbf{A} = -\frac{\mu_0 I}{2\pi} \ln(s/s_0) \, \hat{z}$$

where $s$ is the perpendicular distance from the wire and $s_0$ is a reference distance (corresponding to the arbitrary constant in the potential).

Verify: $\mathbf{B} = \nabla \times \mathbf{A}$. In cylindrical coordinates with $\mathbf{A} = A_z(s)\hat{z}$:

$$\nabla \times \mathbf{A} = -\frac{\partial A_z}{\partial s}\hat{\phi} = \frac{\mu_0 I}{2\pi s}\hat{\phi}$$

This is indeed the correct magnetic field of an infinite wire.

### Vector Potential of a Circular Loop (On-Axis)

For a circular loop of radius $R$ carrying current $I$, the vector potential on the axis ($\rho = 0$, arbitrary $z$) vanishes by symmetry: $\mathbf{A}_{\text{axis}} = 0$. This is because the contributions from opposite sides of the loop cancel.

However, slightly off-axis, the vector potential is nonzero and has only a $\phi$-component. For $\rho \ll R$ (near the axis):

$$A_\phi \approx \frac{\mu_0 I R^2 \rho}{4(R^2 + z^2)^{3/2}}$$

The full off-axis expression involves elliptic integrals — one of many situations where the potential is easier to compute numerically than analytically.

### Relation Between A and Magnetic Flux

A beautiful identity connects the vector potential to the magnetic flux:

$$\Phi_B = \int_S \mathbf{B} \cdot d\mathbf{a} = \int_S (\nabla \times \mathbf{A}) \cdot d\mathbf{a} = \oint_C \mathbf{A} \cdot d\mathbf{l}$$

The magnetic flux through any surface is the line integral of $\mathbf{A}$ around the boundary. This is Stokes' theorem applied to the vector potential, and it provides a practical method for computing $\mathbf{A}$ when the flux is known (as in the solenoid example below).

---

## Vector Potential of a Solenoid

An infinite solenoid of radius $R$, with $n$ turns per unit length carrying current $I$, has $\mathbf{B} = \mu_0 n I \hat{z}$ inside and $\mathbf{B} = 0$ outside.

By symmetry, $\mathbf{A} = A_\phi(s)\hat{\phi}$ (circulates around the z-axis). We can find it using $\oint \mathbf{A} \cdot d\mathbf{l} = \int \mathbf{B} \cdot d\mathbf{a}$ (since $\int (\nabla \times \mathbf{A}) \cdot d\mathbf{a} = \oint \mathbf{A} \cdot d\mathbf{l}$ by Stokes' theorem):

**Inside** ($s < R$):
$$A_\phi (2\pi s) = \mu_0 n I (\pi s^2) \implies A_\phi = \frac{\mu_0 n I}{2} s$$

**Outside** ($s > R$):
$$A_\phi (2\pi s) = \mu_0 n I (\pi R^2) \implies A_\phi = \frac{\mu_0 n I R^2}{2s}$$

Note: Outside the solenoid, $\mathbf{B} = 0$ but $\mathbf{A} \neq 0$! The vector potential is nonzero even where the magnetic field vanishes. This remarkable fact — that the potential can exist where the field does not — has profound physical consequences through the Aharonov-Bohm effect, discussed below.

The form $A_\phi \propto 1/s$ outside the solenoid is identical to the vector potential of a magnetic monopole string — a topological feature that appears in various contexts in theoretical physics.

```python
import numpy as np
import matplotlib.pyplot as plt

# Vector potential of a solenoid: A_φ as a function of distance from axis
# Why important: A ≠ 0 outside the solenoid even though B = 0 there

mu_0 = 4 * np.pi * 1e-7
n = 1000     # turns per meter
I = 1.0      # current (A)
R = 0.05     # solenoid radius (5 cm)

s = np.linspace(0.001, 0.15, 500)

# A_φ: linear inside, 1/s outside
A_phi = np.where(
    s < R,
    mu_0 * n * I * s / 2,           # inside: grows linearly
    mu_0 * n * I * R**2 / (2 * s)   # outside: falls as 1/s
)

# B_z: uniform inside, zero outside
B_z = np.where(s < R, mu_0 * n * I, 0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Vector potential A_φ
axes[0].plot(s * 100, A_phi * 1e6, 'b-', linewidth=2)
axes[0].axvline(x=R*100, color='gray', linestyle='--', label=f'R = {R*100:.0f} cm')
axes[0].set_xlabel('s (cm)')
axes[0].set_ylabel(r'$A_\phi$ ($\mu$T·m)')
axes[0].set_title(r'Vector Potential $A_\phi(s)$')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].annotate('A ≠ 0 here!\n(but B = 0)', xy=(R*100 + 2, A_phi[300]*1e6),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

# Magnetic field B_z
axes[1].plot(s * 100, B_z * 1e3, 'r-', linewidth=2)
axes[1].axvline(x=R*100, color='gray', linestyle='--', label=f'R = {R*100:.0f} cm')
axes[1].set_xlabel('s (cm)')
axes[1].set_ylabel('$B_z$ (mT)')
axes[1].set_title(r'Magnetic Field $B_z(s)$')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Solenoid: A is nonzero where B is zero!', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('solenoid_vector_potential.png', dpi=150)
plt.show()

# Verify B = curl A numerically
# In cylindrical coordinates with A = A_φ(s) φ̂:
# B_z = (1/s) d(sA_φ)/ds
ds = s[1] - s[0]
sA = s * A_phi
# Why numerical derivative: to verify B = curl A
B_numerical = np.gradient(sA, ds) / s

print("Verification: B_z from curl(A)")
print(f"  Analytic B inside:  {mu_0 * n * I * 1e3:.4f} mT")
idx_inside = len(s) // 5   # point well inside
print(f"  Numerical B inside: {B_numerical[idx_inside] * 1e3:.4f} mT")
idx_outside = 4 * len(s) // 5  # point well outside
print(f"  Numerical B outside: {B_numerical[idx_outside] * 1e3:.6f} mT (should be ~0)")
```

---

## The Aharonov-Bohm Effect

The Aharonov-Bohm effect (1959) is one of the most profound results in quantum mechanics. It demonstrates that the vector potential $\mathbf{A}$ has direct physical consequences, even in regions where $\mathbf{B} = 0$.

### Setup

An electron beam is split into two paths that pass on opposite sides of a solenoid and then recombine. The solenoid is completely shielded — the electrons never experience any magnetic field ($\mathbf{B} = 0$ outside the solenoid).

### Observation

Despite $\mathbf{B} = 0$ everywhere the electrons travel, the interference pattern shifts when the current in the solenoid changes. The phase difference between the two paths is:

$$\Delta\phi = \frac{e}{\hbar}\oint \mathbf{A} \cdot d\mathbf{l} = \frac{e}{\hbar}\Phi_B$$

where $\Phi_B = \int \mathbf{B} \cdot d\mathbf{a}$ is the magnetic flux through the solenoid.

### Significance

- In classical physics, $\mathbf{A}$ is just a mathematical convenience; only $\mathbf{B}$ is "real"
- In quantum mechanics, $\mathbf{A}$ directly affects the phase of the electron wavefunction
- This suggests that **potentials are more fundamental than fields**
- The effect has been confirmed experimentally (Tonomura et al., 1986, using a toroidal magnet covered with a superconducting shield to ensure $\mathbf{B} = 0$ outside)

### Connection to Berry Phase

The Aharonov-Bohm phase is an example of a **geometric phase** (Berry phase): a phase acquired by a quantum state when the parameters of its Hamiltonian are cycled around a closed loop. The AB phase depends on the topology of the path (whether it encircles the solenoid) rather than local field values. This topological character is why it cannot be explained by any local field effect — it is inherently non-local.

```python
import numpy as np
import matplotlib.pyplot as plt

# Aharonov-Bohm effect: interference pattern shift with enclosed flux
# Why simulate: visualizing the fringe shift makes the abstract effect concrete

hbar = 1.055e-34      # reduced Planck constant (J·s)
e = 1.6e-19            # electron charge (C)
lambda_dB = 1e-10      # de Broglie wavelength (1 Å, typical for electrons)
k = 2 * np.pi / lambda_dB

# Screen position (angular coordinate)
theta = np.linspace(-0.01, 0.01, 1000)  # small angles (radians)

# Baseline: path length difference → standard double-slit pattern
d_slit = 1e-6     # slit separation (1 μm)
delta_path = d_slit * np.sin(theta)  # path length difference
# Why k*delta_path: the phase difference from geometry alone
phase_geom = k * delta_path

# Aharonov-Bohm phase: additional phase from vector potential
# Φ_B = magnetic flux through solenoid (in units of flux quantum Φ₀ = h/e)
phi_0 = 2 * np.pi * hbar / e  # flux quantum

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, flux_ratio in enumerate([0, 0.25, 0.5, 1.0]):
    ax = axes[idx // 2, idx % 2]

    # AB phase shift
    phase_AB = 2 * np.pi * flux_ratio  # in radians

    # Interference: I ∝ cos²((δ_geom + δ_AB)/2)
    # Why cosine squared: this is two-beam interference
    I_pattern = np.cos(0.5 * (phase_geom + phase_AB))**2

    ax.plot(theta * 1e3, I_pattern, 'b-', linewidth=1.5)
    ax.set_xlabel('θ (mrad)')
    ax.set_ylabel('Intensity (arb. units)')
    ax.set_title(f'Φ/Φ₀ = {flux_ratio:.2f}  (AB phase = {phase_AB/np.pi:.1f}π)')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # Mark the central fringe position
    central_shift = -phase_AB / k / d_slit  # shift in angle
    ax.axvline(x=central_shift * 1e3, color='red', linestyle='--', alpha=0.7)

plt.suptitle('Aharonov-Bohm Effect: Interference Pattern vs. Enclosed Flux',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('aharonov_bohm.png', dpi=150)
plt.show()
```

---

## Multipole Expansion of the Vector Potential

Just as the electrostatic potential can be expanded in multipoles (monopole, dipole, quadrupole, ...), so can the vector potential.

For a localized current distribution, the vector potential at large distances ($r \gg r'$) is:

$$\mathbf{A}(\mathbf{r}) = \frac{\mu_0}{4\pi} \frac{1}{r} \int \mathbf{J}(\mathbf{r}') \, d\tau' + \frac{\mu_0}{4\pi} \frac{1}{r^2} \int (\hat{r} \cdot \mathbf{r}') \mathbf{J}(\mathbf{r}') \, d\tau' + \cdots$$

### Monopole Term

$$\mathbf{A}_{\text{mono}} = \frac{\mu_0}{4\pi r} \int \mathbf{J} \, d\tau' = 0$$

The monopole term **always vanishes** for magnetostatics. This is deeply connected to $\nabla \cdot \mathbf{B} = 0$ — there are no magnetic monopoles, so there is no monopole contribution to the potential.

For a current loop: $\int I \, d\mathbf{l}' = I \oint d\mathbf{l}' = 0$ (the integral around a closed loop is zero).

This is fundamentally different from electrostatics, where the monopole term $V_{\text{mono}} = Q/(4\pi\epsilon_0 r)$ is the dominant contribution at large distances. The absence of the magnetic monopole term means the dipole term is the leading long-range contribution — which is why magnetic fields fall off faster ($1/r^3$) than electric fields ($1/r^2$) from neutral charge distributions.

### Dipole Term

$$\mathbf{A}_{\text{dip}} = \frac{\mu_0}{4\pi} \frac{\mathbf{m} \times \hat{r}}{r^2}$$

where $\mathbf{m} = I \int d\mathbf{a}' = I\mathbf{A}$ is the magnetic dipole moment.

This is the leading term at large distances. Taking the curl:

$$\mathbf{B}_{\text{dip}} = \nabla \times \mathbf{A}_{\text{dip}} = \frac{\mu_0}{4\pi r^3}[3(\mathbf{m}\cdot\hat{r})\hat{r} - \mathbf{m}]$$

which we already encountered in Lesson 4.

### Higher Multipoles

Beyond the dipole, the next term is the **magnetic quadrupole**. Its vector potential falls off as $1/r^3$ and its field as $1/r^4$. The quadrupole moment tensor is a second-rank object that captures the next level of detail in the current distribution's geometry.

In atomic and nuclear physics, the magnetic dipole moment is measured with extraordinary precision — the electron's anomalous magnetic moment agrees with QED predictions to better than one part in $10^{12}$. Higher magnetic multipole moments reveal the internal structure of nuclei: a nonzero nuclear magnetic quadrupole moment, for example, indicates that the nucleus deviates from spherical symmetry.

### Comparison: Electric vs. Magnetic Multipoles

| Order | Electric potential | Magnetic potential | Vanishes? |
|---|---|---|---|
| Monopole ($l=0$) | $\sim Q/r$ | Always zero | Yes (always, no monopoles) |
| Dipole ($l=1$) | $\sim p\cos\theta/r^2$ | $\sim m\sin\theta/r^2$ | Only if $\mathbf{p}=0$ or $\mathbf{m}=0$ |
| Quadrupole ($l=2$) | $\sim 1/r^3$ | $\sim 1/r^3$ | Depends on geometry |

> **Analogy**: The multipole expansion is like describing a complex shape from far away. From a great distance, any current loop looks like a magnetic dipole — just as any charged object looks like a point charge. The higher multipoles capture the finer details that only matter up close.

---

## Gauge Transformations in Detail

The freedom to transform $\mathbf{A} \to \mathbf{A} + \nabla\lambda$ is not merely a nuisance — it is a deep symmetry of electromagnetism.

### Different Gauges

| Gauge | Condition | Best for |
|---|---|---|
| Coulomb gauge | $\nabla \cdot \mathbf{A} = 0$ | Magnetostatics, radiation |
| Lorenz gauge | $\nabla \cdot \mathbf{A} = -\mu_0\epsilon_0 \frac{\partial V}{\partial t}$ | Relativistic problems |
| Axial gauge | $A_z = 0$ | Some lattice calculations |
| Temporal gauge | $V = 0$ | Quantum field theory |
| Weyl gauge | $V = 0$ | Canonical quantization |

### Gauge Invariance

Physical observables (forces, energies, interference patterns) must be gauge-invariant. For example:
- $\mathbf{B} = \nabla \times \mathbf{A}$ is gauge-invariant (curl of gradient is zero)
- The AB phase $\oint \mathbf{A} \cdot d\mathbf{l}$ around a closed loop is gauge-invariant (integrating $\nabla\lambda$ around a closed loop gives zero by Stokes' theorem)
- The magnetic flux $\Phi = \oint \mathbf{A}\cdot d\mathbf{l}$ is gauge-invariant (same reasoning)
- Any quantity derived from $\mathbf{E}$ and $\mathbf{B}$ alone (energy density, Poynting vector, forces) is automatically gauge-invariant

### Why Gauge Freedom Matters

One might ask: why not simply fix a gauge once and for all? The reason is practical:

1. **Different gauges simplify different problems.** The Coulomb gauge is ideal for magnetostatics; the Lorenz gauge for radiation problems; the axial gauge for certain lattice calculations.
2. **Gauge invariance constrains the theory.** When constructing new theories, the requirement that results be gauge-invariant is a powerful check — any result that depends on the gauge choice must be wrong.
3. **Gauge symmetry is generative.** In quantum field theory, promoting gauge invariance to a fundamental principle generates the entire theory of electrodynamics from scratch.

The requirement of gauge invariance is one of the most powerful principles in modern physics — it underlies the Standard Model of particle physics.

### From Classical to Quantum Gauge Theory

In quantum mechanics, the gauge transformation extends to the wavefunction:

$$\psi \to \psi' = \psi \, e^{iq\lambda/\hbar}$$

The requirement that physics be invariant under local gauge transformations ($\lambda$ depending on position and time) is the principle that gives rise to electrodynamics itself in quantum field theory. Starting from a free particle and demanding local gauge invariance, one is forced to introduce a gauge field $A_\mu$ — and this is precisely the electromagnetic four-potential! This is the deep reason why electromagnetism exists.

The same logic, applied to more complex symmetry groups, generates the weak and strong nuclear forces — the entire Standard Model of particle physics emerges from gauge invariance.

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate gauge freedom: two different A fields giving the same B
# Why: seeing gauge invariance in action builds confidence in the formalism

# Setup: uniform B = B_0 z_hat in a region
B_0 = 1e-3   # 1 mT

# Grid
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
X, Y = np.meshgrid(x, y)

# Gauge 1: symmetric gauge A = (B₀/2)(-y x̂ + x ŷ)
# Why symmetric: treats x and y equivalently
Ax_1 = -B_0 * Y / 2
Ay_1 = B_0 * X / 2

# Gauge 2: Landau gauge A = B₀ x ŷ  (all in the y-component)
# This is related to gauge 1 by λ = B₀xy/2
Ax_2 = np.zeros_like(X)
Ay_2 = B_0 * X

# Both should give the same B_z = ∂Ay/∂x - ∂Ax/∂y
dx = x[1] - x[0]
dy = y[1] - y[0]

# Numerical curl for gauge 1
dAy1_dx = np.gradient(Ay_1, dx, axis=1)
dAx1_dy = np.gradient(Ax_1, dy, axis=0)
Bz_1 = dAy1_dx - dAx1_dy

# Numerical curl for gauge 2
dAy2_dx = np.gradient(Ay_2, dx, axis=1)
dAx2_dy = np.gradient(Ax_2, dy, axis=0)
Bz_2 = dAy2_dx - dAx2_dy

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gauge 1: symmetric gauge
axes[0].quiver(X, Y, Ax_1, Ay_1, color='blue', alpha=0.7)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Gauge 1 (Symmetric): A = B₀/2 (-y, x, 0)')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# Gauge 2: Landau gauge
axes[1].quiver(X, Y, Ax_2, Ay_2, color='red', alpha=0.7)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Gauge 2 (Landau): A = B₀(0, x, 0)')
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)

# Both give the same B
axes[2].plot(x, Bz_1[10, :] * 1e3, 'bo-', label='Gauge 1', markersize=4)
axes[2].plot(x, Bz_2[10, :] * 1e3, 'r^-', label='Gauge 2', markersize=4)
axes[2].axhline(y=B_0 * 1e3, color='green', linestyle='--', label=f'B₀ = {B_0*1e3} mT')
axes[2].set_xlabel('x')
axes[2].set_ylabel('B_z (mT)')
axes[2].set_title('Same B from Different A')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Gauge Freedom: Different A, Same B', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gauge_freedom.png', dpi=150)
plt.show()

print(f"B_z from Gauge 1: mean = {np.mean(Bz_1)*1e3:.4f} mT, std = {np.std(Bz_1)*1e3:.6f} mT")
print(f"B_z from Gauge 2: mean = {np.mean(Bz_2)*1e3:.4f} mT, std = {np.std(Bz_2)*1e3:.6f} mT")
print(f"Max difference:    {np.max(np.abs(Bz_1 - Bz_2))*1e3:.6f} mT")
```

---

## Summary

| Concept | Key Equation |
|---|---|
| Vector potential | $\mathbf{B} = \nabla \times \mathbf{A}$ |
| Gauge freedom | $\mathbf{A}' = \mathbf{A} + \nabla\lambda$ gives same $\mathbf{B}$ |
| Coulomb gauge | $\nabla \cdot \mathbf{A} = 0$ |
| Poisson form | $\nabla^2 \mathbf{A} = -\mu_0\mathbf{J}$ (in Coulomb gauge) |
| Solution | $\mathbf{A} = \frac{\mu_0}{4\pi}\int \frac{\mathbf{J}}{|\mathbf{r}-\mathbf{r}'|}\,d\tau'$ |
| Infinite wire | $A_z = -\frac{\mu_0 I}{2\pi}\ln(s/s_0)$ |
| Solenoid (inside) | $A_\phi = \frac{\mu_0 n I}{2}s$ |
| Solenoid (outside) | $A_\phi = \frac{\mu_0 n I R^2}{2s}$ |
| Monopole term | $\mathbf{A}_{\text{mono}} = 0$ (always) |
| Dipole term | $\mathbf{A}_{\text{dip}} = \frac{\mu_0}{4\pi}\frac{\mathbf{m}\times\hat{r}}{r^2}$ |
| AB phase | $\Delta\phi = \frac{e}{\hbar}\oint\mathbf{A}\cdot d\mathbf{l}$ |

---

## Exercises

### Exercise 1: Vector Potential of a Finite Solenoid
Numerically compute $\mathbf{A}$ for a solenoid of finite length $L$ and radius $R$, modeled as $N$ circular current loops. Compare with the infinite solenoid formula inside and outside. Where does the approximation break down?

### Exercise 2: Verify Coulomb Gauge
For the vector potential of a circular current loop computed numerically via the Biot-Savart-like formula, verify that $\nabla \cdot \mathbf{A} = 0$ holds at several points.

### Exercise 3: Gauge Transformation
Starting from the symmetric gauge $\mathbf{A}_1 = \frac{B_0}{2}(-y\hat{x} + x\hat{y})$, find the gauge function $\lambda$ that transforms it to the Landau gauge $\mathbf{A}_2 = B_0 x\hat{y}$. Verify your answer.

### Exercise 4: Magnetic Quadrupole
Compute the vector potential of two circular loops with opposite currents (a magnetic quadrupole). How does $\mathbf{A}$ fall off with distance? Compare with the dipole fall-off.

### Exercise 5: AB Effect Flux Quantization
In a superconductor, the magnetic flux through a loop is quantized in units of $\Phi_0 = h/(2e)$ (the factor of 2 is because superconducting carriers are Cooper pairs). Calculate $\Phi_0$ in SI units. If a superconducting ring has area $1\,\text{mm}^2$, what is the maximum $\mathbf{B}$ for a single flux quantum? The flux quantum $\Phi_0 \approx 2.07 \times 10^{-15}$ Wb is a fundamental constant that plays a central role in superconducting quantum devices (SQUIDs) and defines the sensitivity limit of magnetic field measurements.

### Exercise 6: Numerical Vector Potential of a Current Sheet
An infinite current sheet in the $xy$-plane carries surface current density $\mathbf{K} = K_0\hat{x}$ (current per unit length in the $x$-direction). The vector potential is $\mathbf{A} = -\frac{\mu_0 K_0}{2}|z|\hat{x}$. Verify this by computing $\nabla \times \mathbf{A}$ and showing it gives the correct $\mathbf{B}$ (uniform field pointing in opposite directions above and below the sheet).

---

[← Previous: 04. Magnetostatics](04_Magnetostatics.md) | [Next: 06. Electromagnetic Induction →](06_Electromagnetic_Induction.md)
