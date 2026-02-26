# 14. Relativistic Electrodynamics

[← Previous: 13. Radiation and Antennas](13_Radiation_and_Antennas.md) | [Next: 15. Multipole Expansion →](15_Multipole_Expansion.md)

## Learning Objectives

1. Apply Lorentz transformations to electric and magnetic fields and understand their mixing
2. Construct the electromagnetic field tensor $F^{\mu\nu}$ and its dual
3. Write Maxwell's equations in manifestly covariant form using 4-vectors and tensors
4. Understand the 4-potential $A^\mu$, 4-current $J^\mu$, and gauge invariance in covariant language
5. Derive the Lorentz force law from the covariant formulation
6. See how magnetism arises naturally as a relativistic correction to electrostatics
7. Appreciate the Lagrangian formulation of electrodynamics and the stress-energy tensor

Electricity and magnetism are not two separate forces — they are different facets of a single entity, the electromagnetic field, whose appearance depends on the observer's frame of reference. Special relativity reveals this unity with stunning elegance. A purely electric field in one frame acquires a magnetic component when viewed from a moving frame. The electromagnetic field tensor $F^{\mu\nu}$ packages all six components of $\mathbf{E}$ and $\mathbf{B}$ into a single geometric object, and Maxwell's four equations collapse into just two tensor equations. This lesson shows that relativity is not an optional add-on to electrodynamics — it is deeply embedded in its structure.

> **Analogy**: Imagine looking at a flagpole's shadow. When you view it from the south, the shadow points north and has a certain length. Walk to the east, and the shadow now appears to point northwest with a different length. The shadow itself has not changed — only your perspective has. Similarly, $\mathbf{E}$ and $\mathbf{B}$ are like "shadows" of the electromagnetic field tensor projected onto a particular reference frame. Different observers see different $\mathbf{E}$ and $\mathbf{B}$, but the underlying $F^{\mu\nu}$ is the same geometric object.

---

## 1. Lorentz Transformations Review

### 1.1 Spacetime Coordinates

In special relativity, an event is described by the 4-position:

$$x^\mu = (ct, x, y, z) = (x^0, x^1, x^2, x^3)$$

We use the metric signature $(+, -, -, -)$ (particle physics convention). The Minkowski metric is:

$$\eta_{\mu\nu} = \text{diag}(+1, -1, -1, -1)$$

### 1.2 Lorentz Boost

For a boost along the $x$-axis with velocity $v$:

$$\Lambda^\mu_{\ \nu} = \begin{pmatrix} \gamma & -\gamma\beta & 0 & 0 \\ -\gamma\beta & \gamma & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

where $\beta = v/c$ and $\gamma = 1/\sqrt{1-\beta^2}$.

### 1.3 4-Vectors

A **4-vector** $A^\mu$ transforms as $A'^\mu = \Lambda^\mu_{\ \nu} A^\nu$. Important examples:

- **4-velocity**: $u^\mu = \gamma(c, \mathbf{v})$
- **4-momentum**: $p^\mu = m u^\mu = (\gamma mc, \gamma m\mathbf{v}) = (E/c, \mathbf{p})$
- **4-current density**: $J^\mu = (c\rho, \mathbf{J})$
- **4-potential**: $A^\mu = (V/c, \mathbf{A})$

---

## 2. Transformation of Electric and Magnetic Fields

### 2.1 Field Transformation Rules

Under a Lorentz boost with velocity $\mathbf{v} = v\hat{x}$, the fields transform as:

**Components parallel to the boost direction ($x$-axis):**

$$E'_x = E_x, \quad B'_x = B_x$$

**Components perpendicular to the boost direction:**

$$\boxed{E'_y = \gamma(E_y - vB_z), \quad E'_z = \gamma(E_z + vB_y)}$$

$$\boxed{B'_y = \gamma\left(B_y + \frac{v}{c^2}E_z\right), \quad B'_z = \gamma\left(B_z - \frac{v}{c^2}E_y\right)}$$

In compact notation for a general boost $\boldsymbol{\beta} = \mathbf{v}/c$:

$$\mathbf{E}' = \gamma(\mathbf{E} + \boldsymbol{\beta} \times c\mathbf{B}) - (\gamma - 1)(\mathbf{E} \cdot \hat{\beta})\hat{\beta}$$

$$\mathbf{B}' = \gamma\left(\mathbf{B} - \frac{\boldsymbol{\beta}}{c} \times \mathbf{E}\right) - (\gamma - 1)(\mathbf{B} \cdot \hat{\beta})\hat{\beta}$$

### 2.2 Key Implications

1. **A purely electric field acquires a magnetic component** in a moving frame (and vice versa)
2. If $\mathbf{E} = 0$ in one frame, then $\mathbf{E}' = \gamma(\boldsymbol{\beta} \times c\mathbf{B})$ — moving through a magnetic field creates an electric field
3. Two **Lorentz invariants** are preserved in all frames:

$$\boxed{\mathbf{E} \cdot \mathbf{B} = \text{invariant}, \quad E^2 - c^2 B^2 = \text{invariant}}$$

These invariants determine the character of the field: if $E^2 > c^2 B^2$, there exists a frame where $\mathbf{B} = 0$; if $E^2 < c^2 B^2$, there exists a frame where $\mathbf{E} = 0$.

```python
import numpy as np
import matplotlib.pyplot as plt

def lorentz_transform_fields(E, B, beta_vec):
    """
    Transform E and B fields under a Lorentz boost.

    Parameters:
        E        : electric field [Ex, Ey, Ez] (V/m)
        B        : magnetic field [Bx, By, Bz] (T)
        beta_vec : velocity/c [bx, by, bz]

    Returns:
        E', B' in the boosted frame

    Why this matters: the transformation reveals that E and B are not
    independent — they are components of a single entity (the field tensor)
    that mix under changes of reference frame.
    """
    c = 3e8
    E = np.array(E, dtype=float)
    B = np.array(B, dtype=float)
    beta = np.array(beta_vec, dtype=float)
    beta_mag = np.linalg.norm(beta)

    if beta_mag < 1e-15:
        return E.copy(), B.copy()

    gamma = 1.0 / np.sqrt(1 - beta_mag**2)
    beta_hat = beta / beta_mag

    # Parallel and perpendicular components
    E_par = np.dot(E, beta_hat) * beta_hat
    E_perp = E - E_par
    B_par = np.dot(B, beta_hat) * beta_hat
    B_perp = B - B_par

    # Transform
    E_prime = E_par + gamma * (E_perp + c * np.cross(beta, B))
    B_prime = B_par + gamma * (B_perp - np.cross(beta, E) / c)

    return E_prime, B_prime

# Example: Pure magnetic field in lab frame
B_lab = np.array([0, 0, 1.0])     # 1 T in z-direction
E_lab = np.array([0, 0, 0])       # no electric field

# Observe from frame moving at v = 0.5c in x-direction
beta = np.array([0.5, 0, 0])

E_prime, B_prime = lorentz_transform_fields(E_lab, B_lab, beta)

print("Lab frame:")
print(f"  E = {E_lab} V/m")
print(f"  B = {B_lab} T")
print(f"\nMoving frame (v = 0.5c in x-direction):")
print(f"  E' = [{E_prime[0]:.4e}, {E_prime[1]:.4e}, {E_prime[2]:.4e}] V/m")
print(f"  B' = [{B_prime[0]:.4f}, {B_prime[1]:.4f}, {B_prime[2]:.4f}] T")
print(f"\nA magnetic field in the lab becomes electric + magnetic in the moving frame!")
print(f"\nLorentz invariants:")
print(f"  E·B = {np.dot(E_lab, B_lab):.6f} (lab) = {np.dot(E_prime, B_prime):.6f} (moving)")
print(f"  E²-c²B² = {np.dot(E_lab,E_lab) - (3e8)**2 * np.dot(B_lab,B_lab):.4e} (lab)")
print(f"           = {np.dot(E_prime,E_prime) - (3e8)**2 * np.dot(B_prime,B_prime):.4e} (moving)")
```

---

## 3. The Electromagnetic Field Tensor

### 3.1 Construction

The field tensor $F^{\mu\nu}$ is an antisymmetric rank-2 tensor that encodes all six components of $\mathbf{E}$ and $\mathbf{B}$:

$$F^{\mu\nu} = \begin{pmatrix} 0 & -E_x/c & -E_y/c & -E_z/c \\ E_x/c & 0 & -B_z & B_y \\ E_y/c & B_z & 0 & -B_x \\ E_z/c & -B_y & B_x & 0 \end{pmatrix}$$

The tensor transforms as:

$$F'^{\mu\nu} = \Lambda^\mu_{\ \alpha} \Lambda^\nu_{\ \beta} F^{\alpha\beta}$$

This automatically reproduces the field transformation rules from Section 2.

### 3.2 The Dual Tensor

The **dual tensor** $\tilde{F}^{\mu\nu}$ is obtained by the replacement $\mathbf{E} \to c\mathbf{B}$ and $\mathbf{B} \to -\mathbf{E}/c$:

$$\tilde{F}^{\mu\nu} = \frac{1}{2}\epsilon^{\mu\nu\alpha\beta}F_{\alpha\beta} = \begin{pmatrix} 0 & -B_x & -B_y & -B_z \\ B_x & 0 & E_z/c & -E_y/c \\ B_y & -E_z/c & 0 & E_x/c \\ B_z & E_y/c & -E_x/c & 0 \end{pmatrix}$$

where $\epsilon^{\mu\nu\alpha\beta}$ is the Levi-Civita symbol.

### 3.3 Lorentz Invariants from the Tensor

The two Lorentz invariants are elegantly expressed as:

$$F_{\mu\nu}F^{\mu\nu} = 2\left(B^2 - \frac{E^2}{c^2}\right)$$

$$F_{\mu\nu}\tilde{F}^{\mu\nu} = -\frac{4}{c}\mathbf{E} \cdot \mathbf{B}$$

```python
def field_tensor(E, B, c=3e8):
    """
    Construct the electromagnetic field tensor F^{mu,nu}.

    Why the field tensor: it unifies E and B into a single geometric
    object that transforms naturally under Lorentz transformations.
    It makes manifest the relativistic structure of electrodynamics.
    """
    Ex, Ey, Ez = E
    Bx, By, Bz = B

    F = np.array([
        [0,      -Ex/c,  -Ey/c,  -Ez/c],
        [Ex/c,    0,     -Bz,     By   ],
        [Ey/c,    Bz,     0,     -Bx   ],
        [Ez/c,   -By,     Bx,     0    ]
    ])
    return F

def lorentz_boost_matrix(beta_x):
    """4x4 Lorentz boost matrix along x-axis."""
    gamma = 1.0 / np.sqrt(1 - beta_x**2)
    L = np.array([
        [gamma,        -gamma*beta_x, 0, 0],
        [-gamma*beta_x, gamma,         0, 0],
        [0,             0,             1, 0],
        [0,             0,             0, 1]
    ])
    return L

def transform_field_tensor(F, Lambda):
    """Transform F^{mu,nu} under Lorentz transformation."""
    return Lambda @ F @ Lambda.T

# Verify field transformation via the tensor method
E_lab = np.array([0, 0, 0])
B_lab = np.array([0, 0, 1.0])

F_lab = field_tensor(E_lab, B_lab)
Lambda = lorentz_boost_matrix(0.5)
F_prime = transform_field_tensor(F_lab, Lambda)

c = 3e8
print("Field tensor in lab frame:")
print(np.array2string(F_lab, precision=4, suppress_small=True))

print("\nField tensor in boosted frame:")
print(np.array2string(F_prime, precision=4, suppress_small=True))

# Extract E' and B' from F_prime
E_prime_tensor = np.array([F_prime[1,0], F_prime[2,0], F_prime[3,0]]) * c
B_prime_tensor = np.array([F_prime[3,2], F_prime[1,3], F_prime[2,1]])

print(f"\nExtracted fields from F':")
print(f"  E' = {E_prime_tensor}")
print(f"  B' = {B_prime_tensor}")

# Verify invariants
inv1_lab = np.trace(F_lab @ F_lab.T)  # not quite right; need F_{mu nu} F^{mu nu}
# Proper contraction with metric
eta = np.diag([1, -1, -1, -1])
F_lower_lab = eta @ F_lab @ eta
inv1_lab = np.sum(F_lower_lab * F_lab)
F_lower_prime = eta @ F_prime @ eta
inv1_prime = np.sum(F_lower_prime * F_prime)

print(f"\nInvariant F_{{μν}}F^{{μν}}: lab = {inv1_lab:.6f}, boosted = {inv1_prime:.6f}")
```

---

## 4. Covariant Maxwell Equations

### 4.1 The 4-Potential and 4-Current

The electromagnetic potentials form a 4-vector:

$$A^\mu = \left(\frac{V}{c}, \mathbf{A}\right)$$

The field tensor is the curl of the 4-potential:

$$F^{\mu\nu} = \partial^\mu A^\nu - \partial^\nu A^\mu$$

The sources form a 4-current:

$$J^\mu = (c\rho, \mathbf{J})$$

The continuity equation $\nabla \cdot \mathbf{J} + \partial\rho/\partial t = 0$ becomes:

$$\partial_\mu J^\mu = 0$$

### 4.2 Maxwell's Equations in Covariant Form

All four of Maxwell's equations reduce to **just two** tensor equations:

**Inhomogeneous equations** (Gauss's law + Ampere-Maxwell):

$$\boxed{\partial_\mu F^{\mu\nu} = \mu_0 J^\nu}$$

This single equation encodes:
- $\nu = 0$: $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$ (Gauss's law)
- $\nu = 1,2,3$: $\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\epsilon_0 \partial\mathbf{E}/\partial t$ (Ampere-Maxwell)

**Homogeneous equations** (no magnetic monopoles + Faraday):

$$\boxed{\partial_\mu \tilde{F}^{\mu\nu} = 0}$$

Equivalently, using the Bianchi identity:

$$\partial_\lambda F_{\mu\nu} + \partial_\mu F_{\nu\lambda} + \partial_\nu F_{\lambda\mu} = 0$$

This encodes:
- $\nabla \cdot \mathbf{B} = 0$
- $\nabla \times \mathbf{E} = -\partial\mathbf{B}/\partial t$ (Faraday's law)

### 4.3 Gauge Invariance

The transformation:

$$A^\mu \to A^\mu + \partial^\mu \chi$$

leaves $F^{\mu\nu}$ unchanged (since the antisymmetric derivative kills the symmetric part). This is **gauge invariance**, the deep symmetry underlying electrodynamics.

Common gauges:
- **Lorenz gauge**: $\partial_\mu A^\mu = 0$ (manifestly covariant)
- **Coulomb gauge**: $\nabla \cdot \mathbf{A} = 0$ (not covariant, but convenient for non-relativistic problems)

---

## 5. The Lorentz Force in Covariant Form

### 5.1 Covariant Equation of Motion

The Lorentz force law $\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$ becomes:

$$\frac{dp^\mu}{d\tau} = q F^{\mu\nu} u_\nu$$

where $\tau$ is the proper time and $u_\nu = \eta_{\nu\alpha} u^\alpha$ is the covariant 4-velocity.

This is manifestly covariant: the left side is a 4-vector (proper-time derivative of 4-momentum), and the right side is a contraction of a rank-2 tensor with a 4-vector, which is also a 4-vector.

### 5.2 Verification

The spatial components ($\mu = 1, 2, 3$) give:

$$\frac{d(\gamma m\mathbf{v})}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

which is the relativistic Lorentz force law. The temporal component ($\mu = 0$) gives the power equation:

$$\frac{dE}{dt} = q\mathbf{v} \cdot \mathbf{E}$$

which says that only the electric field does work on a charge (the magnetic force is always perpendicular to the velocity).

---

## 6. Magnetism as a Relativistic Effect

### 6.1 The Thought Experiment

Consider a long straight wire carrying current $I$, and a charge $q$ moving parallel to the wire at velocity $v$ (the same drift velocity as the conduction electrons). In the lab frame, the wire is electrically neutral and the charge experiences only a magnetic force.

Now transform to the charge's rest frame. The charge is stationary, so there can be no magnetic force. But the Lorentz contraction of the positive and negative charge densities in the wire is **different** (because they have different velocities in the lab), creating a net **electric field** that provides exactly the same force.

### 6.2 Quantitative Derivation

In the lab frame, the wire has:
- Positive charges (ions): linear charge density $+\lambda$, stationary
- Negative charges (electrons): linear charge density $-\lambda$, drift velocity $v_d$

The magnetic field at distance $s$ is $B = \mu_0 I / (2\pi s)$, and the force on the test charge is:

$$F_{\text{mag}} = qvB = \frac{qv\mu_0 I}{2\pi s}$$

In the charge's rest frame (moving at $v$), Lorentz contraction modifies the charge densities differently:

$$\lambda'_+ = \gamma_v \lambda, \quad \lambda'_- = -\frac{\gamma_v}{\gamma_d'}\lambda$$

where $\gamma_d'$ accounts for the modified electron speed in the new frame. The net charge per unit length creates an electric field, and the resulting electric force exactly matches the magnetic force in the lab frame.

This demonstrates that **magnetism is a relativistic effect of electrostatics** — it arises from the different Lorentz contractions of positive and negative charge distributions.

```python
def magnetism_from_relativity():
    """
    Demonstrate that the magnetic force on a charge moving parallel
    to a current-carrying wire equals the electric force in the
    charge's rest frame due to relativistic length contraction.

    Why this matters: it's one of the most beautiful results in physics.
    The magnetic force — which seems like a fundamentally different
    phenomenon — is nothing but electrostatics viewed from a moving frame.
    """
    c = 3e8
    eps_0 = 8.854e-12
    mu_0 = 4 * np.pi * 1e-7

    # Parameters
    I = 10       # current (A)
    s = 0.01     # distance from wire (m)
    q = 1.6e-19  # test charge (C)

    # Drift velocity of electrons (typically very slow)
    v_d = 1e-4   # m/s (typical for copper wire)

    # Test charge velocity (same as electron drift for simplicity)
    v = v_d

    # Lab frame: magnetic force
    B = mu_0 * I / (2 * np.pi * s)
    F_mag = q * v * B
    print(f"Lab frame (magnetic force):")
    print(f"  B = {B:.6e} T")
    print(f"  F_mag = {F_mag:.6e} N")

    # Now let's show the equivalence for various test charge velocities
    v_test = np.linspace(0.001 * c, 0.9 * c, 100)
    gamma_v = 1.0 / np.sqrt(1 - (v_test / c)**2)

    # Magnetic force in lab frame
    F_magnetic = q * v_test * B

    # Approximate electric force in moving frame
    # The net charge density due to differential Lorentz contraction is:
    # delta_lambda ≈ lambda * v * v_d / c^2 (to first order in v_d/c)
    # This gives E ≈ lambda * v * v_d / (2*pi*eps_0*s*c^2)
    # And F_elec = qE = q*v*mu_0*I/(2*pi*s) = F_mag  (using mu_0 = 1/eps_0*c^2)

    # More precisely, using relativistic velocity addition:
    # lambda_net ≈ gamma_v * lambda * (v*v_d/c^2) for v_d << c
    lambda_line = I / v_d  # linear charge density (C/m)
    delta_lambda = gamma_v * lambda_line * v_test * v_d / c**2
    E_moving = delta_lambda / (2 * np.pi * eps_0 * s)
    F_electric = q * E_moving

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(v_test / c, F_magnetic, 'b-', linewidth=2,
            label='Magnetic force (lab frame)')
    ax.plot(v_test / c, F_electric, 'r--', linewidth=2,
            label='Electric force (moving frame)')
    ax.set_xlabel('Test charge velocity (v/c)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Magnetism as a Relativistic Effect')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(0.5, F_magnetic[50] * 0.7,
            'The two forces are identical:\nmagnetic force = electric force\nfrom different frames!',
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig("magnetism_relativity.png", dpi=150)
    plt.show()

magnetism_from_relativity()
```

---

## 7. Lagrangian Formulation (Overview)

### 7.1 Lagrangian for a Charged Particle

The action for a charged particle in an electromagnetic field is:

$$S = \int \left(-mc \, ds + \frac{q}{c} A_\mu \, dx^\mu\right)$$

where $ds = c \, d\tau = \sqrt{c^2 dt^2 - dx^2 - dy^2 - dz^2}$ is the spacetime interval.

The Lagrangian in terms of coordinate time is:

$$L = -mc^2\sqrt{1 - v^2/c^2} + q\mathbf{v} \cdot \mathbf{A} - qV$$

The Euler-Lagrange equations reproduce the Lorentz force law.

### 7.2 Lagrangian for the Electromagnetic Field

The action for the free electromagnetic field is:

$$S_{\text{field}} = -\frac{1}{4\mu_0}\int F_{\mu\nu}F^{\mu\nu} \, d^4x$$

Adding the coupling to sources:

$$S_{\text{int}} = -\int A_\mu J^\mu \, d^4x$$

Varying $S_{\text{field}} + S_{\text{int}}$ with respect to $A_\mu$ yields the inhomogeneous Maxwell equations.

### 7.3 Stress-Energy Tensor

The energy and momentum of the electromagnetic field are encapsulated in the **stress-energy tensor**:

$$T^{\mu\nu} = \frac{1}{\mu_0}\left(F^{\mu\alpha}F^{\nu}_{\ \alpha} - \frac{1}{4}\eta^{\mu\nu}F_{\alpha\beta}F^{\alpha\beta}\right)$$

Its components have physical meaning:
- $T^{00} = \frac{1}{2}\left(\epsilon_0 E^2 + \frac{B^2}{\mu_0}\right)$ — energy density
- $T^{0i}/c = \epsilon_0(\mathbf{E} \times \mathbf{B})_i$ — momentum density (Poynting vector / $c^2$)
- $T^{ij}$ — Maxwell stress tensor (momentum flux)

Conservation of energy-momentum is expressed as:

$$\partial_\mu T^{\mu\nu} = -F^{\nu\alpha}J_\alpha$$

In the absence of sources, $\partial_\mu T^{\mu\nu} = 0$.

---

## 8. Visualization: Field Transformations

```python
def visualize_field_transformations():
    """
    Show how E and B fields transform as a function of boost velocity.

    Why visualize: seeing the smooth mixing of E and B fields with
    velocity makes the relativistic nature of electromagnetism tangible.
    """
    beta_range = np.linspace(-0.99, 0.99, 500)
    gamma_range = 1.0 / np.sqrt(1 - beta_range**2)
    c = 3e8

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Case 1: Pure B field (1 T in z-direction)
    B0 = 1.0
    Ey_prime = -gamma_range * beta_range * c * B0
    Bz_prime = gamma_range * B0

    axes[0, 0].plot(beta_range, Ey_prime / 1e8, 'b-', linewidth=2, label="$E'_y$ / 10$^8$")
    axes[0, 0].set_ylabel("$E'_y$ (10$^8$ V/m)")
    axes[0, 0].set_title("Pure B-field ($B_z$ = 1 T): Electric field appears")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(beta_range, Bz_prime, 'r-', linewidth=2, label="$B'_z$")
    axes[0, 1].axhline(y=B0, color='gray', linestyle='--', alpha=0.5, label='Lab value')
    axes[0, 1].set_ylabel("$B'_z$ (T)")
    axes[0, 1].set_title("Pure B-field: Magnetic field strengthens")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Case 2: Pure E field (1 MV/m in y-direction)
    E0 = 1e6
    Ey_prime2 = gamma_range * E0
    Bz_prime2 = -gamma_range * beta_range * E0 / c

    axes[1, 0].plot(beta_range, Ey_prime2 / 1e6, 'b-', linewidth=2, label="$E'_y$ / MV/m")
    axes[1, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Lab value')
    axes[1, 0].set_xlabel('$\\beta = v/c$')
    axes[1, 0].set_ylabel("$E'_y$ (MV/m)")
    axes[1, 0].set_title("Pure E-field ($E_y$ = 1 MV/m): Strengthens")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(beta_range, Bz_prime2 * 1e3, 'r-', linewidth=2, label="$B'_z$")
    axes[1, 1].set_xlabel('$\\beta = v/c$')
    axes[1, 1].set_ylabel("$B'_z$ (mT)")
    axes[1, 1].set_title("Pure E-field: Magnetic field appears")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Lorentz Transformation of Electromagnetic Fields', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("field_transformations.png", dpi=150)
    plt.show()

visualize_field_transformations()
```

---

## Summary

| Concept | Key Formula | Physical Meaning |
|---------|-------------|------------------|
| Field tensor | $F^{\mu\nu} = \partial^\mu A^\nu - \partial^\nu A^\mu$ | Unifies E and B into one object |
| Covariant Maxwell (inhomogeneous) | $\partial_\mu F^{\mu\nu} = \mu_0 J^\nu$ | Gauss + Ampere-Maxwell |
| Covariant Maxwell (homogeneous) | $\partial_\mu \tilde{F}^{\mu\nu} = 0$ | No monopoles + Faraday |
| Field transformation | $\mathbf{E}' = \gamma(\mathbf{E} + \boldsymbol{\beta}\times c\mathbf{B}) - (\gamma-1)(\mathbf{E}\cdot\hat{\beta})\hat{\beta}$ | E and B mix under boosts |
| Lorentz invariant 1 | $\mathbf{E} \cdot \mathbf{B} = \text{const}$ | Angle between E and B preserved |
| Lorentz invariant 2 | $E^2 - c^2B^2 = \text{const}$ | Field character preserved |
| Covariant force | $dp^\mu/d\tau = qF^{\mu\nu}u_\nu$ | Relativistic Lorentz force |
| Stress-energy tensor | $T^{00} = \frac{1}{2}(\epsilon_0 E^2 + B^2/\mu_0)$ | Field energy density |

---

## Exercises

### Exercise 1: Relativistic Current Loop
A circular current loop of radius $R$ carrying current $I$ lies in the $xy$-plane. A charge $q$ is at rest at position $(0, 0, d)$ on the loop axis. (a) In the lab frame, what is the force on $q$? (b) Now boost to a frame where $q$ moves with velocity $v\hat{x}$. Compute $\mathbf{E}'$ and $\mathbf{B}'$ at the charge's location using the field transformation. (c) Verify that the spatial force $\mathbf{F}' = q(\mathbf{E}' + \mathbf{v}' \times \mathbf{B}')$ gives the correct result.

### Exercise 2: Field Tensor Invariants
An electromagnetic wave has $\mathbf{E} = E_0 \hat{y} \cos(kz - \omega t)$ and $\mathbf{B} = (E_0/c) \hat{x} \cos(kz - \omega t)$. (a) Compute both Lorentz invariants $\mathbf{E} \cdot \mathbf{B}$ and $E^2 - c^2B^2$. (b) Interpret the result: can you find a frame where only $\mathbf{E}$ or only $\mathbf{B}$ exists? (c) Construct the field tensor and verify the invariants using $F_{\mu\nu}F^{\mu\nu}$.

### Exercise 3: Covariant Continuity
Starting from $\partial_\mu F^{\mu\nu} = \mu_0 J^\nu$, show that charge conservation $\partial_\nu J^\nu = 0$ follows automatically from the antisymmetry of $F^{\mu\nu}$. This is analogous to how $\nabla \cdot (\nabla \times \mathbf{B}) = 0$ implies $\nabla \cdot \mathbf{J} + \partial\rho/\partial t = 0$ in 3D notation.

### Exercise 4: Stress-Energy Tensor for a Plane Wave
For a plane electromagnetic wave propagating in the $z$-direction, compute all 16 components of $T^{\mu\nu}$. Show that $T^{00} = T^{03}/c = T^{33}$ (energy density equals momentum flux), which is the hallmark of massless radiation.

---

[← Previous: 13. Radiation and Antennas](13_Radiation_and_Antennas.md) | [Next: 15. Multipole Expansion →](15_Multipole_Expansion.md)
