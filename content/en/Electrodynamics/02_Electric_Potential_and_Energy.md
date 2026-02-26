# Electric Potential and Energy

[← Previous: 01. Electrostatics Review](01_Electrostatics_Review.md) | [Next: 03. Conductors and Dielectrics →](03_Conductors_and_Dielectrics.md)

---

## Learning Objectives

1. Define the scalar potential $V$ and derive its relationship to the electric field $\mathbf{E}$
2. Compute the potential from point charges and continuous charge distributions
3. Derive and interpret Poisson's and Laplace's equations
4. State and apply the uniqueness theorems for electrostatic boundary value problems
5. Calculate the energy stored in an electrostatic configuration
6. Express electrostatic energy in terms of the field energy density
7. Implement numerical solutions to Laplace's equation using relaxation methods

---

The electric field is a vector — three components at every point in space. But because $\nabla \times \mathbf{E} = 0$ in electrostatics, all the information in the field is captured by a single scalar function: the electric potential $V$. This simplification is not merely computational convenience; it reveals deep physics. The potential is directly tied to energy, and energy arguments are often the fastest route to understanding why charges arrange themselves the way they do. In this lesson we develop the machinery of potential and energy, culminating in Poisson's equation — the fundamental equation that, once solved, gives us everything.

---

## The Scalar Potential

Since $\nabla \times \mathbf{E} = 0$, the electric field can be written as the gradient of a scalar:

$$\mathbf{E} = -\nabla V$$

The negative sign is a convention ensuring that $\mathbf{E}$ points from high potential to low potential (like a ball rolling downhill).

The potential difference between two points $\mathbf{a}$ and $\mathbf{b}$ is:

$$V(\mathbf{b}) - V(\mathbf{a}) = -\int_{\mathbf{a}}^{\mathbf{b}} \mathbf{E} \cdot d\mathbf{l}$$

This integral is **path-independent** (because $\nabla \times \mathbf{E} = 0$), which is why $V$ is well-defined.

> **Analogy**: The electric potential is like altitude on a topographic map. The electric field is the "slope" — it points downhill (from high $V$ to low $V$), and its magnitude tells you how steep the hill is. Just as water flows downhill, positive charges "flow" from high to low potential.

### Potential of a Point Charge

Setting $V = 0$ at infinity (the standard reference):

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0} \frac{q}{|\mathbf{r} - \mathbf{r}'|}$$

For a collection of point charges:

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0} \sum_{i=1}^{N} \frac{q_i}{|\mathbf{r} - \mathbf{r}_i'|}$$

For a continuous distribution:

$$V(\mathbf{r}) = \frac{1}{4\pi\epsilon_0} \int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d\tau'$$

Notice: the potential involves $1/r$ (scalar), not $1/r^2$ (vector). Scalar integrals are much easier to evaluate than vector integrals. This is the primary computational advantage of working with the potential: compute $V$ first (one scalar integral), then differentiate to get $\mathbf{E} = -\nabla V$ (three partial derivatives). The alternative — directly integrating $\mathbf{E}$ as a vector integral — requires three separate integrals, each involving the direction of $\hat{\boldsymbol{\mathscr{r}}}$.

### Units and Dimensions

The potential $V$ has units of volts (V = J/C). It represents the potential energy per unit charge:

$$V = \frac{U}{q} \qquad [\text{V} = \text{J/C} = \text{kg}\cdot\text{m}^2/(\text{A}\cdot\text{s}^3)]$$

The electric field has units V/m, consistent with $\mathbf{E} = -\nabla V$ (gradient introduces $1/\text{m}$).

### Equipotential Surfaces

Surfaces of constant $V$ are called **equipotential surfaces**. Key properties:
- $\mathbf{E}$ is everywhere perpendicular to equipotential surfaces
- No work is done moving a charge along an equipotential
- Conductors in equilibrium are equipotential bodies (we'll prove this in Lesson 3)

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize equipotential surfaces for a dipole
# Why contour plot: equipotentials are curves of constant V in 2D

epsilon_0 = 8.854e-12
k_e = 1 / (4 * np.pi * epsilon_0)

# Dipole: +q at (-d/2, 0), -q at (+d/2, 0)
q = 1e-9
d = 0.2
charges = [(-d/2, 0, q), (d/2, 0, -q)]

x = np.linspace(-0.6, 0.6, 400)
y = np.linspace(-0.6, 0.6, 400)
X, Y = np.meshgrid(x, y)

V = np.zeros_like(X)
for (cx, cy, qi) in charges:
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    r = np.maximum(r, 1e-4)  # avoid singularity at charge locations
    V += k_e * qi / r

# Why clip: potential diverges near charges; clipping makes contours visible
V_clipped = np.clip(V, -500, 500)

fig, ax = plt.subplots(figsize=(8, 8))
levels = np.linspace(-400, 400, 41)
cs = ax.contour(X, Y, V_clipped, levels=levels, cmap='RdBu_r')
ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Equipotential Lines of an Electric Dipole')
ax.set_aspect('equal')

for (cx, cy, qi) in charges:
    color = 'red' if qi > 0 else 'blue'
    ax.plot(cx, cy, 'o', color=color, markersize=10)

plt.tight_layout()
plt.savefig('dipole_equipotentials.png', dpi=150)
plt.show()
```

---

## Poisson's and Laplace's Equations

Combining $\mathbf{E} = -\nabla V$ with Gauss's law $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$:

$$\nabla \cdot (-\nabla V) = \frac{\rho}{\epsilon_0}$$

$$\boxed{\nabla^2 V = -\frac{\rho}{\epsilon_0}} \qquad \text{(Poisson's equation)}$$

In charge-free regions ($\rho = 0$):

$$\boxed{\nabla^2 V = 0} \qquad \text{(Laplace's equation)}$$

These are **second-order partial differential equations**. Poisson's equation is the central equation of electrostatics — solve it with appropriate boundary conditions, and you know the potential (and hence the field) everywhere.

### The Laplacian in Different Coordinates

| Coordinate system | $\nabla^2 V$ |
|---|---|
| Cartesian | $\frac{\partial^2 V}{\partial x^2} + \frac{\partial^2 V}{\partial y^2} + \frac{\partial^2 V}{\partial z^2}$ |
| Spherical | $\frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2 \frac{\partial V}{\partial r}\right) + \frac{1}{r^2\sin\theta}\frac{\partial}{\partial \theta}\left(\sin\theta \frac{\partial V}{\partial \theta}\right) + \frac{1}{r^2\sin^2\theta}\frac{\partial^2 V}{\partial \phi^2}$ |
| Cylindrical | $\frac{1}{s}\frac{\partial}{\partial s}\left(s\frac{\partial V}{\partial s}\right) + \frac{1}{s^2}\frac{\partial^2 V}{\partial \phi^2} + \frac{\partial^2 V}{\partial z^2}$ |

### Key Properties of Laplace's Equation

1. **No local extrema**: A function satisfying $\nabla^2 V = 0$ cannot have a local maximum or minimum in the interior of its domain. This is the **maximum principle** — the maximum and minimum values occur on the boundary.

2. **Mean value property**: The value of $V$ at any point equals the average of $V$ over any sphere centered at that point (provided no charges lie within the sphere):

$$V(\mathbf{r}_0) = \frac{1}{4\pi R^2} \oint_{\text{sphere}} V \, da$$

These properties are not just mathematically elegant — they are the basis of numerical relaxation methods.

---

## Uniqueness Theorems

How do we know our solution is the *right* one? The uniqueness theorems guarantee it.

### First Uniqueness Theorem

The potential $V$ in a volume $\mathcal{V}$ is uniquely determined if:
1. The charge density $\rho$ is specified throughout $\mathcal{V}$, and
2. The value of $V$ is specified on the boundary surface $\mathcal{S}$ (Dirichlet boundary condition)

**Proof sketch**: Suppose $V_1$ and $V_2$ are both solutions. Their difference $V_3 = V_1 - V_2$ satisfies $\nabla^2 V_3 = 0$ in $\mathcal{V}$ with $V_3 = 0$ on $\mathcal{S}$. By the maximum principle, $V_3 = 0$ everywhere, so $V_1 = V_2$.

### Second Uniqueness Theorem

In a volume surrounded by conductors, the electric field is uniquely determined if the total charge on each conductor is specified. (This allows Neumann boundary conditions — specifying $\partial V/\partial n$ rather than $V$ itself.)

These theorems are tremendously powerful: they tell us that **any** method of finding a solution — guessing, symmetry arguments, numerical computation — gives **the** solution.

---

## Numerical Solution: Relaxation Method

The mean value property suggests a numerical algorithm. We discretize space on a grid and iteratively set each grid point to the average of its neighbors:

```python
import numpy as np
import matplotlib.pyplot as plt

# Solve Laplace's equation in 2D using the Jacobi relaxation method
# Why relaxation: it directly exploits the mean-value property of harmonic functions

# Problem: square region, V=100 on top edge, V=0 on other edges
N = 100                          # grid points per side
V = np.zeros((N, N))

# Boundary conditions — these drive the entire solution
V[0, :] = 100.0     # top edge at 100 V
V[-1, :] = 0.0      # bottom edge at 0 V
V[:, 0] = 0.0       # left edge at 0 V
V[:, -1] = 0.0      # right edge at 0 V

# Relaxation iterations
# Why 5000 iterations: convergence is slow for Jacobi; SOR would be faster
n_iter = 5000
for iteration in range(n_iter):
    V_old = V.copy()
    # Update interior points: each point becomes average of 4 neighbors
    # Why average of neighbors: this is the discrete form of ∇²V = 0
    V[1:-1, 1:-1] = 0.25 * (
        V_old[0:-2, 1:-1] +    # top neighbor
        V_old[2:, 1:-1] +      # bottom neighbor
        V_old[1:-1, 0:-2] +    # left neighbor
        V_old[1:-1, 2:]        # right neighbor
    )
    # Re-enforce boundary conditions (they must not change)
    V[0, :] = 100.0
    V[-1, :] = 0.0
    V[:, 0] = 0.0
    V[:, -1] = 0.0

    # Check convergence
    if iteration % 1000 == 0:
        diff = np.max(np.abs(V - V_old))
        print(f"Iteration {iteration}: max change = {diff:.2e}")

# Plot the solution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Contour plot of potential
cs = axes[0].contourf(V, levels=50, cmap='hot')
plt.colorbar(cs, ax=axes[0], label='V (volts)')
axes[0].set_title("Potential V (Laplace's equation)")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# Electric field (negative gradient of V)
# Why np.gradient: numerical differentiation to get E from V
Ey, Ex = np.gradient(-V)   # note: gradient returns (row, col) = (y, x)
E_mag = np.sqrt(Ex**2 + Ey**2)

# Subsample for cleaner arrows
step = 5
xx = np.arange(0, N, step)
yy = np.arange(0, N, step)
XX, YY = np.meshgrid(xx, yy)

axes[1].quiver(XX, YY, Ex[::step, ::step], Ey[::step, ::step],
               E_mag[::step, ::step], cmap='viridis', scale=500)
axes[1].set_title('Electric Field E = -∇V')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

plt.suptitle("Solving Laplace's Equation by Relaxation", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('laplace_relaxation.png', dpi=150)
plt.show()
```

---

## Electrostatic Energy

### Energy of Point Charges

The energy required to assemble a configuration of point charges (bringing them in from infinity):

$$W = \frac{1}{2} \sum_{i=1}^{N} \sum_{\substack{j=1 \\ j \neq i}}^{N} \frac{q_i q_j}{4\pi\epsilon_0 |\mathbf{r}_i - \mathbf{r}_j|}$$

The factor of $1/2$ corrects for double-counting each pair. Equivalently:

$$W = \frac{1}{2} \sum_{i=1}^{N} q_i V(\mathbf{r}_i)$$

where $V(\mathbf{r}_i)$ is the potential at $q_i$ due to all other charges.

### Energy of a Continuous Distribution

For a volume charge density:

$$W = \frac{1}{2} \int \rho \, V \, d\tau$$

### Energy in Terms of the Field

Using Gauss's law, we can rewrite the energy entirely in terms of the electric field:

$$\boxed{W = \frac{\epsilon_0}{2} \int_{\text{all space}} |\mathbf{E}|^2 \, d\tau}$$

This is a remarkable result. The energy is stored **in the field itself**, not just "between" charges. The **energy density** is:

$$u = \frac{\epsilon_0}{2} E^2 \quad [\text{J/m}^3]$$

> **Analogy**: Think of a stretched rubber sheet. The energy is stored in the elastic deformation of the sheet, not at the posts that hold it. Similarly, electrostatic energy is stored in the "deformation" of the electric field throughout space.

### Derivation Outline

Starting from $W = \frac{1}{2}\int \rho V \, d\tau$ and using $\rho = \epsilon_0 \nabla \cdot \mathbf{E}$:

$$W = \frac{\epsilon_0}{2}\int V (\nabla \cdot \mathbf{E}) \, d\tau$$

Apply the product rule $\nabla \cdot (V\mathbf{E}) = V(\nabla \cdot \mathbf{E}) + \mathbf{E} \cdot (\nabla V)$ and note $\nabla V = -\mathbf{E}$:

$$W = \frac{\epsilon_0}{2}\left[\int \nabla \cdot (V\mathbf{E}) \, d\tau + \int E^2 \, d\tau \right]$$

The first integral becomes a surface integral (by the divergence theorem) that vanishes when the surface is pushed to infinity (since $V \sim 1/r$ and $E \sim 1/r^2$, so $VE \sim 1/r^3$ while $da \sim r^2$). This leaves:

$$W = \frac{\epsilon_0}{2}\int E^2 \, d\tau$$

```python
import numpy as np

# Compute electrostatic energy of a uniformly charged sphere two ways
# Why two methods: comparing charge-based and field-based gives confidence

epsilon_0 = 8.854e-12
k_e = 1 / (4 * np.pi * epsilon_0)

Q = 1e-9      # total charge (1 nC)
R = 0.1       # sphere radius (10 cm)

# Method 1: Assembly energy (bringing shells of charge from infinity)
# W = (3/5) * kQ²/R — classic result for uniform sphere
W_assembly = (3 / 5) * k_e * Q**2 / R
print(f"Assembly energy:  W = {W_assembly:.6e} J")

# Method 2: Field energy — integrate (ε₀/2)E² over all space
# Inside: E = kQr/R³, Outside: E = kQ/r²
# Why split integral: E has different functional forms inside and outside

N_r = 100000
r_inner = np.linspace(1e-6, R, N_r)
r_outer = np.linspace(R, 100 * R, N_r)  # integrate far enough

dr_in = r_inner[1] - r_inner[0]
dr_out = r_outer[1] - r_outer[0]

# Inside the sphere: E(r) = kQr/R³
E_in = k_e * Q * r_inner / R**3
u_in = 0.5 * epsilon_0 * E_in**2
# Why 4πr²: spherical shell volume element in radial integration
W_in = np.sum(u_in * 4 * np.pi * r_inner**2 * dr_in)

# Outside the sphere: E(r) = kQ/r²
E_out = k_e * Q / r_outer**2
u_out = 0.5 * epsilon_0 * E_out**2
W_out = np.sum(u_out * 4 * np.pi * r_outer**2 * dr_out)

W_field = W_in + W_out
print(f"Field energy:     W = {W_field:.6e} J")
print(f"Relative error:   {abs(W_field - W_assembly)/W_assembly:.4f}")
print(f"\nEnergy density at surface: {0.5*epsilon_0*(k_e*Q/R**2)**2:.4e} J/m³")
```

---

## The Self-Energy Problem

A subtle issue: the energy of a point charge is **infinite**. Using $W = \frac{\epsilon_0}{2}\int E^2 \, d\tau$ with $E = kq/r^2$:

$$W = \frac{\epsilon_0}{2} \int_0^\infty \left(\frac{q}{4\pi\epsilon_0 r^2}\right)^2 4\pi r^2 \, dr = \frac{q^2}{8\pi\epsilon_0} \int_0^\infty \frac{dr}{r^2} = \infty$$

This "self-energy" divergence is a genuine problem in classical electrodynamics. The resolution involves either:
- Giving the charge a finite size (classical electron radius $r_e = e^2/(4\pi\epsilon_0 m_e c^2) \approx 2.8$ fm)
- Quantum electrodynamics (renormalization)

For practical purposes, we simply exclude self-energy and compute only the **interaction energy** between charges.

The classical electron radius $r_e = e^2/(4\pi\epsilon_0 m_e c^2) \approx 2.82$ fm sets the scale at which classical electrodynamics breaks down. At distances smaller than $r_e$, the self-energy of the electron's field would exceed the electron's rest mass energy — a clear signal that quantum mechanics must take over.

> **Analogy**: The self-energy problem is like asking "how much does it cost to assemble a single coin?" — the question doesn't quite make sense in the same way as asking how much it costs to assemble a stack of coins. The energy concept works beautifully for interactions between charges but becomes problematic for a single point charge by itself.

---

## Potential of Standard Configurations

### Charged Disk

A disk of radius $R$ with uniform surface charge density $\sigma$. On the axis at height $z$:

$$V(z) = \frac{\sigma}{2\epsilon_0}\left(\sqrt{z^2 + R^2} - |z|\right)$$

For $z \gg R$: $V \approx \frac{Q}{4\pi\epsilon_0 z}$ (looks like a point charge, with $Q = \sigma\pi R^2$)

For $z = 0$: $V = \frac{\sigma R}{2\epsilon_0}$ (finite!)

### Charged Line Segment

A uniform line charge of length $2L$ and linear charge density $\lambda$, centered at the origin along the $z$-axis. The potential at a perpendicular distance $s$ on the midplane:

$$V(s) = \frac{\lambda}{4\pi\epsilon_0}\ln\left(\frac{L + \sqrt{L^2 + s^2}}{s}\right) \cdot 2$$

For $s \gg L$: $V \approx \frac{2L\lambda}{4\pi\epsilon_0 s} = \frac{Q}{4\pi\epsilon_0 s}$ (point charge at large distances).

For $s \ll L$: $V \approx \frac{\lambda}{2\pi\epsilon_0}\ln(2L/s)$ (logarithmic, like an infinite line).

```python
import numpy as np
import matplotlib.pyplot as plt

# Potential of a finite line charge — transition from logarithmic to 1/r
# Why study this: it shows how finite geometry interpolates between ideal limits

epsilon_0 = 8.854e-12
k_e = 1 / (4 * np.pi * epsilon_0)

lam = 1e-9        # linear charge density (1 nC/m)
L = 0.2           # half-length of line charge (20 cm)
Q = 2 * L * lam   # total charge

s = np.linspace(0.01, 2.0, 500)

# Exact potential at perpendicular distance s on the midplane
V_exact = 2 * k_e * lam * np.log((L + np.sqrt(L**2 + s**2)) / s)

# Approximation 1: point charge (valid for s >> L)
V_point = k_e * Q / s

# Approximation 2: infinite line (valid for s << L)
# Why reference at s=L: infinite line has arbitrary reference, we match at s=L
V_inf_line = k_e * 2 * lam * np.log(L / s)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(s * 100, V_exact, 'b-', linewidth=2, label='Exact')
ax.plot(s * 100, V_point, 'r--', linewidth=1.5, label='Point charge approx')
ax.plot(s * 100, V_inf_line, 'g:', linewidth=1.5, label='Infinite line approx')
ax.axvline(x=L * 100, color='gray', linestyle='--', alpha=0.5, label=f'L = {L*100:.0f} cm')
ax.set_xlabel('s (cm)')
ax.set_ylabel('V (V)')
ax.set_title('Potential of Finite Line Charge')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('line_charge_potential.png', dpi=150)
plt.show()
```

### Spherical Shell

A shell of radius $R$ with total charge $Q$:

$$V(r) = \begin{cases} \frac{Q}{4\pi\epsilon_0 R} & r < R \text{ (constant inside)} \\ \frac{Q}{4\pi\epsilon_0 r} & r > R \end{cases}$$

The constant interior potential is consistent with $\mathbf{E} = 0$ inside (since $\mathbf{E} = -\nabla V$ and $\nabla(\text{const}) = 0$).

---

## Boundary Conditions for V

At an interface carrying surface charge $\sigma$:

$$V_{\text{above}} = V_{\text{below}} \quad \text{(V is continuous)}$$

$$\frac{\partial V_{\text{above}}}{\partial n} - \frac{\partial V_{\text{below}}}{\partial n} = -\frac{\sigma}{\epsilon_0} \quad \text{(normal derivative is discontinuous)}$$

These conditions, together with Poisson's or Laplace's equation, uniquely determine $V$.

---

## Work and Potential Difference

The work done by the electric field when moving a charge $q$ from point $\mathbf{a}$ to point $\mathbf{b}$:

$$W_{a \to b} = q \int_{\mathbf{a}}^{\mathbf{b}} \mathbf{E} \cdot d\mathbf{l} = q[V(\mathbf{a}) - V(\mathbf{b})]$$

Key insights:
- **Positive work** means the field does work on the charge (charge moves from high to low potential if $q > 0$)
- The work is **path-independent** — only the initial and final positions matter
- The **electron-volt** (eV) = $1.6 \times 10^{-19}$ J is the work done moving one electron through 1 volt

### Connection to Circuits

In a circuit with a battery of EMF $\mathcal{E}$ (electromotive force):
- The battery maintains a potential difference $\Delta V = \mathcal{E}$ across its terminals
- Current flows from high to low potential through the external circuit
- The battery does work $W = q\mathcal{E}$ on charges as they pass through it
- Power delivered: $P = IV = I^2 R = V^2/R$ (Joule's law)

---

## Summary

| Concept | Key Equation |
|---|---|
| Potential definition | $\mathbf{E} = -\nabla V$ |
| Point charge potential | $V = q/(4\pi\epsilon_0 r)$ |
| Poisson's equation | $\nabla^2 V = -\rho/\epsilon_0$ |
| Laplace's equation | $\nabla^2 V = 0$ |
| Energy (discrete) | $W = \frac{1}{2}\sum_i q_i V(\mathbf{r}_i)$ |
| Energy (field) | $W = \frac{\epsilon_0}{2}\int E^2 \, d\tau$ |
| Energy density | $u = \frac{\epsilon_0}{2}E^2$ |
| Mean value property | $V(\mathbf{r}_0)$ = average over surrounding sphere |
| Uniqueness | Solution is unique given $\rho$ + boundary conditions |

---

## Exercises

### Exercise 1: Potential and Field Computation
A thin ring of radius $R = 0.15$ m carries total charge $Q = 2$ nC. Compute and plot $V(z)$ and $E_z(z)$ along the axis. Verify that $E_z = -dV/dz$ numerically.

### Exercise 2: Laplace Solver with Complex Boundaries
Modify the relaxation code to solve Laplace's equation in a square with the following boundary conditions: $V = 100\sin(\pi x/L)$ on the top, $V = 0$ on other edges. Compare your numerical solution with the analytic separation-of-variables result $V(x, y) = 100\sin(\pi x/L)\sinh(\pi y/L)/\sinh(\pi)$.

### Exercise 3: Energy of Concentric Shells
Two concentric spherical shells of radii $a$ and $b$ ($a < b$) carry charges $Q_a$ and $Q_b$. Calculate the total electrostatic energy using the field method. Verify for the special case $Q_a = -Q_b$ (a capacitor).

### Exercise 4: Numerical Poisson Solver
Extend the relaxation method to solve Poisson's equation with a localized charge distribution $\rho(x,y) = \rho_0 \exp(-(x^2+y^2)/w^2)$. Plot the resulting potential and field.

### Exercise 5: Multipole Expansion
Compute the monopole, dipole, and quadrupole terms in the multipole expansion of the potential for a charge distribution consisting of $+q$ at $(0, 0, d)$, $-2q$ at the origin, and $+q$ at $(0, 0, -d)$. This is a **linear quadrupole**. What is the leading far-field term?

---

[← Previous: 01. Electrostatics Review](01_Electrostatics_Review.md) | [Next: 03. Conductors and Dielectrics →](03_Conductors_and_Dielectrics.md)
