# 18. MHD Numerical Methods

## Learning Objectives
- Understand conservative form of MHD equations
- Learn basics of finite volume method
- Understand Godunov-type schemes
- Grasp MHD Riemann problems
- Implement simple MHD shock tube problem
- Learn div B = 0 constraint handling methods

**Why This Lesson Matters:** Numerically solving MHD equations is significantly harder than solving ordinary fluid dynamics (Euler/Navier-Stokes) for two reasons: (1) MHD has 7 characteristic wave speeds instead of 3, making Riemann solvers much more complex, and (2) the divergence-free constraint $\nabla \cdot \mathbf{B} = 0$ has no physical analogue in hydrodynamics and must be maintained numerically to prevent unphysical magnetic monopole forces. The methods developed here -- conservative finite volumes, Godunov schemes, and divergence cleaning -- are the foundation of modern MHD simulation codes used in astrophysics and fusion research.

---

## 1. Conservative Form of MHD Equations

### 1.1 1D MHD Conservative Form

The conservative form $\partial \mathbf{U}/\partial t + \partial \mathbf{F}/\partial x = 0$ is essential for correctly capturing shocks and discontinuities. By the Lax-Wendroff theorem, conservative schemes converge to the correct weak solution, while non-conservative schemes may converge to the wrong shock speed.

```
1D Ideal MHD Conservative Form:

dU/dt + dF/dx = 0

Conserved Variables U:
    [ rho   ]  (density)
    [ rho vx]  (x-momentum)
    [ rho vy]  (y-momentum)
U = [ rho vz]  (z-momentum)
    [ By    ]  (y-magnetic field)
    [ Bz    ]  (z-magnetic field)
    [ E     ]  (total energy)

(Bx = const, in 1D)

Flux F:
    [ rho vx                        ]
    [ rho vx^2 + p* - Bx^2/mu0      ]
    [ rho vx vy - Bx By/mu0         ]
F = [ rho vx vz - Bx Bz/mu0         ]
    [ vx By - vy Bx                 ]
    [ vx Bz - vz Bx                 ]
    [ (E + p*) vx - Bx(v.B)/mu0     ]

Where:
p* = p + B^2/2mu0  (total pressure)
E = p/(gamma-1) + rho v^2/2 + B^2/2mu0  (total energy)
```

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# Physical constants (code units: mu0 = 1)
gamma = 5/3  # Ratio of specific heats

def primitive_to_conservative(rho, vx, vy, vz, Bx, By, Bz, p):
    """Primitive -> Conservative variable conversion"""
    E = p / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2 + vz**2) + \
        0.5 * (Bx**2 + By**2 + Bz**2)

    U = np.array([rho, rho*vx, rho*vy, rho*vz, By, Bz, E])
    return U

def conservative_to_primitive(U, Bx):
    """Conservative -> Primitive variable conversion"""
    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho
    vz = U[3] / rho
    By = U[4]
    Bz = U[5]
    E = U[6]

    p = (gamma - 1) * (E - 0.5 * rho * (vx**2 + vy**2 + vz**2) -
                       0.5 * (Bx**2 + By**2 + Bz**2))

    return rho, vx, vy, vz, Bx, By, Bz, p

def compute_flux(U, Bx):
    """Compute flux"""
    rho, vx, vy, vz, _, By, Bz, p = conservative_to_primitive(U, Bx)

    B2 = Bx**2 + By**2 + Bz**2
    p_star = p + 0.5 * B2  # Total pressure
    E = U[6]
    vB = vx * Bx + vy * By + vz * Bz

    F = np.array([
        rho * vx,
        rho * vx**2 + p_star - Bx**2,
        rho * vx * vy - Bx * By,
        rho * vx * vz - Bx * Bz,
        vx * By - vy * Bx,
        vx * Bz - vz * Bx,
        (E + p_star) * vx - Bx * vB
    ])

    return F

def mhd_wave_speeds(rho, vx, p, Bx, By, Bz):
    """Compute MHD wave speeds"""
    B2 = Bx**2 + By**2 + Bz**2
    cs2 = gamma * p / rho  # Sound speed squared
    ca2 = B2 / rho         # Alfven speed squared
    cax2 = Bx**2 / rho     # x-direction Alfven

    # Fast/slow magnetosonic
    term1 = 0.5 * (cs2 + ca2)
    term2 = 0.5 * np.sqrt((cs2 + ca2)**2 - 4 * cs2 * cax2)

    cf = np.sqrt(term1 + term2)  # Fast
    ca = np.sqrt(cax2)           # Alfven
    cs = np.sqrt(max(term1 - term2, 0))  # Slow

    return cf, ca, cs

print("=" * 60)
print("1D MHD Conservative Form")
print("=" * 60)

# Example: Initial state
rho = 1.0
vx, vy, vz = 0.0, 0.0, 0.0
Bx, By, Bz = 1.0, 0.5, 0.0
p = 1.0

U = primitive_to_conservative(rho, vx, vy, vz, Bx, By, Bz, p)
F = compute_flux(U, Bx)

print("\nPrimitive Variables:")
print(f"  rho={rho}, v=({vx},{vy},{vz}), B=({Bx},{By},{Bz}), p={p}")
print("\nConservative Variables U:")
print(f"  {U}")
print("\nFlux F:")
print(f"  {F}")

cf, ca, cs = mhd_wave_speeds(rho, vx, p, Bx, By, Bz)
print(f"\nWave Speeds: cf={cf:.3f}, ca={ca:.3f}, cs={cs:.3f}")
```

### 1.2 MHD Eigenstructure

```
MHD Characteristics (7 waves):

lambda1 = vx - cf  (fast magnetosonic, left)
lambda2 = vx - ca  (Alfven wave, left)
lambda3 = vx - cs  (slow magnetosonic, left)
lambda4 = vx       (entropy wave)
lambda5 = vx + cs  (slow magnetosonic, right)
lambda6 = vx + ca  (Alfven wave, right)
lambda7 = vx + cf  (fast magnetosonic, right)

Where:
cf = sqrt[(cs^2 + ca^2)/2 + sqrt((cs^2 + ca^2)^2 - 4 cs^2 cax^2)/2]  (fast)
cs = sqrt[(cs^2 + ca^2)/2 - sqrt((cs^2 + ca^2)^2 - 4 cs^2 cax^2)/2]  (slow)
ca = |Bx|/sqrt(rho)                                                   (Alfven)
```

```python
def visualize_mhd_characteristics():
    """Visualize MHD characteristics"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Wave speed vs Bx (By = 0.5, Bz = 0 fixed)
    ax1 = axes[0]

    rho = 1.0
    p = 1.0
    vx = 0.0
    By, Bz = 0.5, 0.0

    Bx_range = np.linspace(0.01, 2.0, 100)

    cf_list, ca_list, cs_list = [], [], []

    for Bx in Bx_range:
        cf, ca, cs = mhd_wave_speeds(rho, vx, p, Bx, By, Bz)
        cf_list.append(cf)
        ca_list.append(ca)
        cs_list.append(cs)

    ax1.plot(Bx_range, cf_list, 'b-', linewidth=2, label='Fast (cf)')
    ax1.plot(Bx_range, ca_list, 'g--', linewidth=2, label='Alfven (ca)')
    ax1.plot(Bx_range, cs_list, 'r-', linewidth=2, label='Slow (cs)')

    # Sound speed reference line
    cs_sound = np.sqrt(gamma * p / rho)
    ax1.axhline(y=cs_sound, color='gray', linestyle=':', label=f'Sound ({cs_sound:.2f})')

    ax1.set_xlabel('Bx')
    ax1.set_ylabel('Wave Speed')
    ax1.set_title('MHD Wave Speed vs Bx')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) x-t diagram (characteristic lines)
    ax2 = axes[1]

    # Characteristic lines at specific state
    Bx = 1.0
    cf, ca, cs = mhd_wave_speeds(rho, vx, p, Bx, By, Bz)

    x0 = 0  # Initial position
    t = np.linspace(0, 1, 50)

    # 7 characteristic lines
    speeds = [-cf, -ca, -cs, 0, cs, ca, cf]
    labels = ['-cf', '-ca', '-cs', 'entropy', '+cs', '+ca', '+cf']
    colors = ['blue', 'green', 'red', 'black', 'red', 'green', 'blue']

    for speed, label, color in zip(speeds, labels, colors):
        x = x0 + speed * t
        ax2.plot(x, t, color=color, linewidth=1.5, label=label)

    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('MHD Characteristic Lines (x-t Diagram)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 2)

    plt.tight_layout()
    plt.savefig('mhd_characteristics.png', dpi=150, bbox_inches='tight')
    plt.show()

# visualize_mhd_characteristics()
```

---

## 2. Finite Volume Method Basics

### 2.1 Integral Form and Cell Averages

```
Finite Volume Method:

Integral Form:
d/dt integral U dx + [F(x2) - F(x1)] = 0

Cell Average:
U_i = (1/dx) integral_{x_{i-1/2}}^{x_{i+1/2}} U dx

Semi-discretization:
dU_i/dt = -(F_{i+1/2} - F_{i-1/2}) / dx

Numerical Flux:
F_{i+1/2} = F(U_i, U_{i+1})  (Riemann solver or approximation)

Advantages:
- Automatically satisfies conservation form
- Natural handling of discontinuities
- Applicable to various grid types
```

```python
def finite_volume_concept():
    """Finite volume method concept visualization"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Cell averages and fluxes
    ax1 = axes[0]

    # Cell boundaries
    x_faces = np.arange(0, 6)
    x_centers = x_faces[:-1] + 0.5

    # Cell average values (example)
    U_avg = np.array([1.0, 0.8, 1.2, 0.6, 0.9])

    # Draw cells
    for i, (x_l, x_r, U) in enumerate(zip(x_faces[:-1], x_faces[1:], U_avg)):
        ax1.fill([x_l, x_r, x_r, x_l], [0, 0, U, U], alpha=0.3,
                color=f'C{i}', edgecolor='black', linewidth=1.5)
        ax1.text((x_l + x_r)/2, U/2, f'$U_{i+1}$', ha='center', va='center', fontsize=11)

    # Flux arrows
    for x in x_faces[1:-1]:
        ax1.annotate('', xy=(x, 1.5), xytext=(x-0.3, 1.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax1.text(x, 1.6, f'$F_{{i+1/2}}$', ha='center', fontsize=10, color='red')

    ax1.set_xlabel('x')
    ax1.set_ylabel('U')
    ax1.set_title('Finite Volume Method: Cell Averages and Numerical Flux')
    ax1.set_xlim(-0.5, 5.5)
    ax1.set_ylim(0, 2)
    ax1.grid(True, alpha=0.3)

    # (2) Riemann problem
    ax2 = axes[1]

    # Initial discontinuity
    x = np.linspace(-1, 1, 200)
    U_L = 1.0
    U_R = 0.5

    U_init = np.where(x < 0, U_L, U_R)
    ax2.plot(x, U_init, 'b-', linewidth=2, label='Initial condition')

    # Riemann solution (conceptual)
    # Shock wave, contact discontinuity, rarefaction wave
    t = 0.3
    x_shock = 0.3  # Shock position

    U_riemann = np.where(x < -0.2, U_L,
                        np.where(x < 0, U_L - (U_L - 0.8) * (x + 0.2) / 0.2,
                                np.where(x < x_shock, 0.8, U_R)))

    ax2.plot(x, U_riemann, 'r--', linewidth=2, label=f't = {t}')

    # Wave markers
    ax2.axvline(x=-0.2, color='green', linestyle=':', label='Rarefaction start')
    ax2.axvline(x=0, color='purple', linestyle=':', label='Contact discontinuity')
    ax2.axvline(x=x_shock, color='orange', linestyle=':', label='Shock wave')

    ax2.set_xlabel('x')
    ax2.set_ylabel('U')
    ax2.set_title('Riemann Problem: Time Evolution of Discontinuous Initial Condition')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('finite_volume.png', dpi=150, bbox_inches='tight')
    plt.show()

# finite_volume_concept()
```

---

## 3. Godunov-type Schemes

### 3.1 Lax-Friedrichs Flux

```
Lax-Friedrichs (LxF) Flux:

F_{i+1/2} = (1/2)[F(U_i) + F(U_{i+1})] - (dx/2dt)(U_{i+1} - U_i)

Or Local LxF:
F_{i+1/2} = (1/2)[F(U_i) + F(U_{i+1})] - (alpha/2)(U_{i+1} - U_i)

alpha = max(|lambda|)  (maximum wave speed)

Characteristics:
- Simple and robust
- 1st order accuracy
- Large numerical diffusion
```

### 3.2 HLL/HLLD Flux

```
HLL (Harten-Lax-van Leer) Flux:

Assumption: Consider only left/right wave speeds SL, SR

         { F_L                              if SL >= 0
F_HLL = { (SR*F_L - SL*F_R + SL*SR*(U_R - U_L))/(SR - SL)  if SL < 0 < SR
         { F_R                              if SR <= 0

Wave speed estimation:
SL = min(vx_L - cf_L, vx_R - cf_R)
SR = max(vx_L + cf_L, vx_R + cf_R)

HLLD (for MHD):
- More refined intermediate states
- Distinguishes contact discontinuity and Alfven waves
- More accurate for MHD
```

```python
def lax_friedrichs_flux(U_L, U_R, Bx, max_speed=None):
    """Lax-Friedrichs flux"""
    F_L = compute_flux(U_L, Bx)
    F_R = compute_flux(U_R, Bx)

    if max_speed is None:
        # Compute maximum wave speed
        rho_L, vx_L, vy_L, vz_L, _, By_L, Bz_L, p_L = conservative_to_primitive(U_L, Bx)
        rho_R, vx_R, vy_R, vz_R, _, By_R, Bz_R, p_R = conservative_to_primitive(U_R, Bx)

        cf_L, _, _ = mhd_wave_speeds(rho_L, vx_L, p_L, Bx, By_L, Bz_L)
        cf_R, _, _ = mhd_wave_speeds(rho_R, vx_R, p_R, Bx, By_R, Bz_R)

        max_speed = max(abs(vx_L) + cf_L, abs(vx_R) + cf_R)

    F = 0.5 * (F_L + F_R) - 0.5 * max_speed * (U_R - U_L)
    return F

def hll_flux(U_L, U_R, Bx):
    """HLL flux"""
    F_L = compute_flux(U_L, Bx)
    F_R = compute_flux(U_R, Bx)

    rho_L, vx_L, vy_L, vz_L, _, By_L, Bz_L, p_L = conservative_to_primitive(U_L, Bx)
    rho_R, vx_R, vy_R, vz_R, _, By_R, Bz_R, p_R = conservative_to_primitive(U_R, Bx)

    cf_L, _, _ = mhd_wave_speeds(rho_L, vx_L, p_L, Bx, By_L, Bz_L)
    cf_R, _, _ = mhd_wave_speeds(rho_R, vx_R, p_R, Bx, By_R, Bz_R)

    SL = min(vx_L - cf_L, vx_R - cf_R)
    SR = max(vx_L + cf_L, vx_R + cf_R)

    if SL >= 0:
        return F_L
    elif SR <= 0:
        return F_R
    else:
        F_HLL = (SR * F_L - SL * F_R + SL * SR * (U_R - U_L)) / (SR - SL)
        return F_HLL
```

---

## 4. MHD Shock Tube Problem

### 4.1 Brio-Wu Shock Tube

```
Brio-Wu Shock Tube (1988):
- Standard test problem for MHD codes
- MHD version of Sod shock tube

Initial Conditions:
Left (x < 0.5):          Right (x >= 0.5):
 rho = 1.0                rho = 0.125
 p = 1.0                  p = 0.1
 vx = vy = vz = 0         vx = vy = vz = 0
 Bx = 0.75                Bx = 0.75
 By = 1.0                 By = -1.0
 Bz = 0                   Bz = 0

Boundary Conditions: Outflow
Final Time: t = 0.1

Solution Structure:
- Fast rarefaction
- Compound wave
- Contact discontinuity
- Slow shock
- Fast rarefaction
```

```python
class MHD_1D_Solver:
    """1D MHD Finite Volume Solver"""

    def __init__(self, Nx=400, x_range=(0, 1), Bx=0.75):
        self.Nx = Nx
        self.x_min, self.x_max = x_range
        self.dx = (self.x_max - self.x_min) / Nx
        self.x = np.linspace(self.x_min + 0.5*self.dx,
                            self.x_max - 0.5*self.dx, Nx)
        self.Bx = Bx

        # Conservative variable array (7 components)
        self.U = np.zeros((7, Nx))

    def set_brio_wu(self):
        """Brio-Wu shock tube initial conditions"""
        for i, x in enumerate(self.x):
            if x < 0.5:
                rho, vx, vy, vz = 1.0, 0.0, 0.0, 0.0
                By, Bz, p = 1.0, 0.0, 1.0
            else:
                rho, vx, vy, vz = 0.125, 0.0, 0.0, 0.0
                By, Bz, p = -1.0, 0.0, 0.1

            self.U[:, i] = primitive_to_conservative(rho, vx, vy, vz,
                                                     self.Bx, By, Bz, p)

    def compute_dt(self, cfl=0.5):
        """Compute time step (CFL condition)"""
        max_speed = 0

        for i in range(self.Nx):
            rho, vx, vy, vz, _, By, Bz, p = conservative_to_primitive(
                self.U[:, i], self.Bx)
            cf, _, _ = mhd_wave_speeds(rho, vx, p, self.Bx, By, Bz)
            max_speed = max(max_speed, abs(vx) + cf)

        return cfl * self.dx / max_speed

    def step(self, dt, flux_func='lxf'):
        """Advance one time step"""
        # Compute fluxes
        F = np.zeros((7, self.Nx + 1))

        for i in range(self.Nx + 1):
            # Boundary treatment (outflow)
            if i == 0:
                U_L = self.U[:, 0]
                U_R = self.U[:, 0]
            elif i == self.Nx:
                U_L = self.U[:, -1]
                U_R = self.U[:, -1]
            else:
                U_L = self.U[:, i-1]
                U_R = self.U[:, i]

            if flux_func == 'lxf':
                F[:, i] = lax_friedrichs_flux(U_L, U_R, self.Bx)
            else:
                F[:, i] = hll_flux(U_L, U_R, self.Bx)

        # Update
        self.U = self.U - dt / self.dx * (F[:, 1:] - F[:, :-1])

    def run(self, t_final, cfl=0.5, flux_func='lxf'):
        """Run simulation"""
        t = 0
        step_count = 0

        while t < t_final:
            dt = self.compute_dt(cfl)
            if t + dt > t_final:
                dt = t_final - t

            self.step(dt, flux_func)
            t += dt
            step_count += 1

        print(f"Completed: {step_count} steps, final t = {t:.4f}")

    def get_primitives(self):
        """Return primitive variables"""
        rho = np.zeros(self.Nx)
        vx = np.zeros(self.Nx)
        vy = np.zeros(self.Nx)
        vz = np.zeros(self.Nx)
        By = np.zeros(self.Nx)
        Bz = np.zeros(self.Nx)
        p = np.zeros(self.Nx)

        for i in range(self.Nx):
            rho[i], vx[i], vy[i], vz[i], _, By[i], Bz[i], p[i] = \
                conservative_to_primitive(self.U[:, i], self.Bx)

        return rho, vx, vy, vz, By, Bz, p


def run_brio_wu_test():
    """Brio-Wu shock tube simulation"""

    # Create solver
    solver = MHD_1D_Solver(Nx=400, x_range=(0, 1), Bx=0.75)
    solver.set_brio_wu()

    # Save initial state
    rho_init, vx_init, vy_init, _, By_init, _, p_init = solver.get_primitives()
    x = solver.x

    # Run simulation
    t_final = 0.1
    solver.run(t_final, cfl=0.5, flux_func='hll')

    # Final state
    rho, vx, vy, _, By, _, p = solver.get_primitives()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Density
    axes[0, 0].plot(x, rho_init, 'b--', alpha=0.5, label='Initial')
    axes[0, 0].plot(x, rho, 'b-', linewidth=1.5, label='Final')
    axes[0, 0].set_ylabel(r'$\rho$')
    axes[0, 0].set_title('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Pressure
    axes[0, 1].plot(x, p_init, 'r--', alpha=0.5, label='Initial')
    axes[0, 1].plot(x, p, 'r-', linewidth=1.5, label='Final')
    axes[0, 1].set_ylabel('p')
    axes[0, 1].set_title('Pressure')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # x-velocity
    axes[0, 2].plot(x, vx_init, 'g--', alpha=0.5, label='Initial')
    axes[0, 2].plot(x, vx, 'g-', linewidth=1.5, label='Final')
    axes[0, 2].set_ylabel(r'$v_x$')
    axes[0, 2].set_title('x-velocity')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # y-velocity
    axes[1, 0].plot(x, vy_init, 'm--', alpha=0.5, label='Initial')
    axes[1, 0].plot(x, vy, 'm-', linewidth=1.5, label='Final')
    axes[1, 0].set_ylabel(r'$v_y$')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_title('y-velocity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # By
    axes[1, 1].plot(x, By_init, 'c--', alpha=0.5, label='Initial')
    axes[1, 1].plot(x, By, 'c-', linewidth=1.5, label='Final')
    axes[1, 1].set_ylabel(r'$B_y$')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_title('y-magnetic field')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Total pressure
    B2 = solver.Bx**2 + By**2
    p_total = p + 0.5 * B2
    B2_init = solver.Bx**2 + By_init**2
    p_total_init = p_init + 0.5 * B2_init

    axes[1, 2].plot(x, p_total_init, 'k--', alpha=0.5, label='Initial')
    axes[1, 2].plot(x, p_total, 'k-', linewidth=1.5, label='Final')
    axes[1, 2].set_ylabel(r'$p + B^2/2$')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_title('Total Pressure')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'Brio-Wu Shock Tube (t = {t_final})', fontsize=14)
    plt.tight_layout()
    plt.savefig('brio_wu_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver

# solver = run_brio_wu_test()
```

---

## 5. div B = 0 Constraint

### 5.1 Problem and Solutions

```
div B = 0 Constraint:

Physical Meaning:
- No magnetic monopoles
- Must always be satisfied

Numerical Problem:
- Discretization errors can cause div B != 0
- "Magnetic monopole" accumulation
- Non-physical forces, instabilities

Solutions:

1. Constrained Transport (CT):
   - Store magnetic field at cell faces
   - Store electric field at cell edges
   - Structurally guarantees div B = 0
   - Similar to Yee grid

2. Projection Method:
   - Hodge decomposition: B = B_sol + grad(phi)
   - Poisson equation: laplacian(phi) = div B
   - Correction: B_new = B - grad(phi)

3. Divergence Cleaning:
   a) Parabolic: dB/dt = -ch^2 grad(div B)
   b) Hyperbolic: dpsi/dt + ch^2 div B = 0,
                  dB/dt + ch grad(psi) = 0
   (GLM: Generalized Lagrange Multiplier)

4. Powell 8-wave:
   - Add source terms: S = -(div B) [0, B, v, v.B]^T
   - Non-conservative, effective in steady state
```

```python
def divergence_cleaning_demo():
    """Divergence cleaning concept demonstration"""

    print("=" * 60)
    print("div B = 0 Constraint Handling Methods")
    print("=" * 60)

    methods = """
    +-------------------------------------------------------------+
    |              div B = 0 Constraint Handling                   |
    +-------------------------------------------------------------+
    |                                                              |
    | 1. Constrained Transport (CT)                                |
    |    - Structurally guarantees div B = 0                       |
    |    - Magnetic field: cell face centers                       |
    |    - Electric field: cell edges                              |
    |    - Automatically satisfied via Stokes theorem              |
    |                                                              |
    | 2. Projection Method                                         |
    |    - Helmholtz decomposition                                 |
    |    - Requires Poisson equation solve                         |
    |    - High computational cost                                 |
    |                                                              |
    | 3. GLM (Hyperbolic Cleaning)                                 |
    |    - Introduce additional scalar variable psi                |
    |    - dpsi/dt + ch^2 div B = -(ch^2/cp^2) psi                |
    |    - dB/dt + ... + ch grad(psi) = 0                         |
    |    - Errors propagate out as waves                           |
    |                                                              |
    | 4. Powell Source Terms                                       |
    |    - Add non-conservative source terms                       |
    |    - dU/dt + dF/dx = -(div B) S                             |
    |    - Simple but violates energy conservation                 |
    |                                                              |
    +-------------------------------------------------------------+
    """
    print(methods)

    # Visualization: CT grid structure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (1) Constrained Transport grid
    ax1 = axes[0]

    # Cell grid
    for i in range(5):
        ax1.axhline(y=i, color='gray', linestyle='-', linewidth=0.5)
        ax1.axvline(x=i, color='gray', linestyle='-', linewidth=0.5)

    # Bx (vertical faces)
    for i in range(5):
        for j in range(4):
            ax1.plot(i, j+0.5, 'b>', markersize=12)

    # By (horizontal faces)
    for i in range(4):
        for j in range(5):
            ax1.plot(i+0.5, j, 'r^', markersize=12)

    # Ez (edges)
    for i in range(5):
        for j in range(5):
            ax1.plot(i, j, 'go', markersize=8)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Constrained Transport Grid')
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 4.5)

    ax1.plot([], [], 'b>', markersize=10, label='Bx (x-face)')
    ax1.plot([], [], 'r^', markersize=10, label='By (y-face)')
    ax1.plot([], [], 'go', markersize=8, label='Ez (edge)')
    ax1.legend(loc='upper right')

    # (2) GLM method concept
    ax2 = axes[1]

    x = np.linspace(0, 10, 100)
    t = 0

    # Initial div B error (Gaussian)
    div_B = np.exp(-(x - 3)**2)

    # Time evolution (propagates in both directions)
    times = [0, 1, 2, 3]
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    for ti, color in zip(times, colors):
        # GLM: error propagates at +-ch speed
        ch = 1.5
        div_B_t = 0.5 * (np.exp(-(x - 3 - ch*ti)**2) +
                        np.exp(-(x - 3 + ch*ti)**2)) * np.exp(-0.5*ti)
        ax2.plot(x, div_B_t, color=color, linewidth=1.5, label=f't = {ti}')

    ax2.set_xlabel('x')
    ax2.set_ylabel(r'$\nabla \cdot B$ error')
    ax2.set_title('GLM: Error Propagates Out of Domain')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('div_b_cleaning.png', dpi=150, bbox_inches='tight')
    plt.show()

# divergence_cleaning_demo()
```

---

## 6. High-Resolution Schemes

### 6.1 MUSCL-Hancock

```
MUSCL-Hancock (2nd Order Accuracy):

1. Linear Reconstruction:
   U_L = U_i + 0.5 * phi(r) * (U_i - U_{i-1})
   U_R = U_{i+1} - 0.5 * phi(r) * (U_{i+2} - U_{i+1})

   phi(r): Slope limiter
   - minmod: phi(r) = max(0, min(1, r))
   - MC: phi(r) = max(0, min((1+r)/2, 2, 2r))
   - van Leer: phi(r) = (r + |r|)/(1 + |r|)

2. Predictor (half-step advance):
   U_L^{n+1/2} = U_L - (dt/2dx)(F(U_L) - F(U_L^-))

3. Riemann solve and flux computation

Advantages:
- 2nd order accuracy (smooth regions)
- TVD (Total Variation Diminishing)
- Oscillation suppression
```

```python
def minmod(a, b):
    """Minmod limiter"""
    if a * b <= 0:
        return 0
    elif abs(a) < abs(b):
        return a
    else:
        return b

def mc_limiter(a, b):
    """MC (Monotonized Central) limiter"""
    if a * b <= 0:
        return 0
    c = 0.5 * (a + b)
    return np.sign(c) * min(abs(c), 2*abs(a), 2*abs(b))

def muscl_reconstruct(U, i, limiter='minmod'):
    """MUSCL reconstruction"""
    if limiter == 'minmod':
        lim_func = minmod
    else:
        lim_func = mc_limiter

    Nx = U.shape[1]

    # Slope calculation
    if i > 0 and i < Nx - 1:
        slope_L = U[:, i] - U[:, i-1]
        slope_R = U[:, i+1] - U[:, i]

        slope = np.array([lim_func(slope_L[k], slope_R[k])
                         for k in range(len(slope_L))])
    else:
        slope = np.zeros(U.shape[0])

    U_L = U[:, i] - 0.5 * slope  # Left state
    U_R = U[:, i] + 0.5 * slope  # Right state

    return U_L, U_R


def run_brio_wu_high_resolution():
    """High-resolution Brio-Wu shock tube"""

    # Solvers (different cell counts)
    solver_lr = MHD_1D_Solver(Nx=200, x_range=(0, 1), Bx=0.75)
    solver_hr = MHD_1D_Solver(Nx=800, x_range=(0, 1), Bx=0.75)

    solver_lr.set_brio_wu()
    solver_hr.set_brio_wu()

    t_final = 0.1

    solver_lr.run(t_final, cfl=0.5, flux_func='hll')
    solver_hr.run(t_final, cfl=0.5, flux_func='hll')

    # Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x_lr = solver_lr.x
    x_hr = solver_hr.x

    rho_lr, vx_lr, _, _, By_lr, _, p_lr = solver_lr.get_primitives()
    rho_hr, vx_hr, _, _, By_hr, _, p_hr = solver_hr.get_primitives()

    axes[0].plot(x_lr, rho_lr, 'b-', linewidth=1.5, label='Nx=200')
    axes[0].plot(x_hr, rho_hr, 'r-', linewidth=1, alpha=0.7, label='Nx=800')
    axes[0].set_ylabel(r'$\rho$')
    axes[0].set_xlabel('x')
    axes[0].set_title('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_lr, vx_lr, 'b-', linewidth=1.5, label='Nx=200')
    axes[1].plot(x_hr, vx_hr, 'r-', linewidth=1, alpha=0.7, label='Nx=800')
    axes[1].set_ylabel(r'$v_x$')
    axes[1].set_xlabel('x')
    axes[1].set_title('x-velocity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x_lr, By_lr, 'b-', linewidth=1.5, label='Nx=200')
    axes[2].plot(x_hr, By_hr, 'r-', linewidth=1, alpha=0.7, label='Nx=800')
    axes[2].set_ylabel(r'$B_y$')
    axes[2].set_xlabel('x')
    axes[2].set_title('y-magnetic field')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Grid Convergence Test (1st Order HLL)', fontsize=14)
    plt.tight_layout()
    plt.savefig('brio_wu_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()

# run_brio_wu_high_resolution()
```

---

## 7. MHD Code Verification

### 7.1 Standard Test Problems

```
Standard Problems for MHD Code Verification:

1. Brio-Wu Shock Tube (1D)
   - Shock waves, contact discontinuity, rarefaction waves
   - Compound wave structure

2. Orszag-Tang Vortex (2D)
   - Highly nonlinear interactions
   - Small-scale structure development
   - Shock wave interactions

3. MHD Rotational Discontinuity (1D)
   - Alfven wave accuracy verification
   - Pure rotation (rho, p constant)

4. Blast Problem (2D/3D)
   - MHD Sedov-Taylor
   - Magnetic field effects

5. Magnetic Loop Advection (2D)
   - div B = 0 preservation verification
   - Circular magnetic structure advection

Accuracy Verification:
- Use problems with analytical solutions
- Grid convergence tests
- Check conserved quantities (mass, momentum, energy)
```

```python
def test_problems_overview():
    """MHD standard test problems overview"""

    print("=" * 60)
    print("MHD Standard Test Problems")
    print("=" * 60)

    tests = """
    1D Tests:
    +-----------------------------------------------------------+
    | Problem       | Verification Target          | Difficulty |
    +---------------+------------------------------+------------+
    | Brio-Wu       | Shock capture, basic waves   | Basic      |
    | Dai-Woodward  | Complete 7-wave structure    | Intermediate|
    | Einfeldt 1203 | Low beta, strong field       | Challenging|
    | Ryu-Jones     | Various MHD waves            | Intermediate|
    +-----------------------------------------------------------+

    2D Tests:
    +-----------------------------------------------------------+
    | Problem       | Verification Target          | Difficulty |
    +---------------+------------------------------+------------+
    | Orszag-Tang   | Nonlinear evolution, turbulence| Standard  |
    | MRI           | Linear growth rate comparison| Physics    |
    | Loop advection| div B, numerical diffusion   | Basic      |
    | Blast problem | Spherical symmetry preservation| Challenging|
    +-----------------------------------------------------------+

    Convergence Tests:
    - Compute L1, L2, L_inf norms
    - Double grid resolution
    - Verify expected convergence order (1st: O(h), 2nd: O(h^2))
    """
    print(tests)

test_problems_overview()
```

---

## 8. Practice Problems

### Exercise 1: Wave Speeds
Calculate fast/slow/Alfven wave speeds for rho = 1, p = 0.5, Bx = 1, By = 0.5, Bz = 0. (gamma = 5/3)

### Exercise 2: Lax-Friedrichs
Apply the Lax-Friedrichs scheme to the 1D linear advection equation and derive the numerical diffusion coefficient.

### Exercise 3: HLL vs LxF
Solve the Brio-Wu problem with both HLL and Lax-Friedrichs fluxes and compare the results.

### Exercise 4: div B Error
Write code to monitor div B error in a 2D MHD simulation.

---

## 9. References

### Key Papers
- Brio & Wu (1988) "An Upwind Differencing Scheme for the Equations of Ideal Magnetohydrodynamics"
- Dedner et al. (2002) "Hyperbolic Divergence Cleaning for the MHD Equations"
- Toth (2000) "The div B = 0 Constraint in Shock-Capturing Magnetohydrodynamics Codes"

### Textbooks
- Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics"
- LeVeque, "Finite Volume Methods for Hyperbolic Problems"

### MHD Codes
- Athena++ (Stone et al.)
- PLUTO (Mignone et al.)
- FLASH (Fryxell et al.)
- Pencil Code (Brandenburg et al.)

---

## Summary

```
MHD Numerical Methods Key Points:

1. Conservative Form:
   dU/dt + dF/dx = 0
   U = [rho, rho v, B, E]^T (7 variables, 1D)

2. Finite Volume Method:
   dU_i/dt = -(F_{i+1/2} - F_{i-1/2})/dx
   Approximate Riemann problem with numerical flux

3. Numerical Fluxes:
   - Lax-Friedrichs: Simple, 1st order, large diffusion
   - HLL: 2-wave approximation, moderate accuracy
   - HLLD: MHD optimized, high accuracy

4. CFL Condition:
   dt <= CFL * dx / (|v| + cf)
   cf: fast magnetosonic speed

5. div B = 0:
   - CT: Structural preservation
   - GLM: Hyperbolic cleaning
   - Projection: Poisson solve

6. High-Resolution Schemes:
   - MUSCL + limiter
   - PPM, WENO
   - 2nd order or higher accuracy

7. Verification:
   - Brio-Wu shock tube
   - Grid convergence tests
   - Check conserved quantities
```

---

## Exercises

### Exercise 1: MHD Wave Speed Calculation
For a plasma state with ρ = 1.0, p = 0.5, Bx = 1.0, By = 0.5, Bz = 0.0, and γ = 5/3, compute all seven MHD characteristic speeds: ±cf, ±ca, ±cs, and the entropy wave speed vx = 0. Use the formulas for fast magnetosonic speed cf, Alfven speed ca = |Bx|/√ρ, and slow magnetosonic speed cs given in Section 1.2. Verify the ordering vx - cf ≤ vx - ca ≤ vx - cs ≤ vx ≤ vx + cs ≤ vx + ca ≤ vx + cf.

### Exercise 2: Lax-Friedrichs Numerical Diffusion
Apply the Lax-Friedrichs scheme to the 1D scalar advection equation ∂u/∂t + a∂u/∂x = 0 with periodic boundary conditions. Using a Gaussian initial condition, run the simulation for 10 wave periods and compare the result to the exact solution (which should be the original Gaussian). Measure the amplitude decay and width broadening as a function of the Courant number S = a·dt/dx ∈ {0.5, 0.8, 0.95}. Show that numerical diffusion is proportional to (1 - S²)/2 × dx.

### Exercise 3: Brio-Wu Shock Tube with HLL vs Lax-Friedrichs
Run the Brio-Wu shock tube test with both the HLL flux and the Lax-Friedrichs flux using Nx = 400 cells and CFL = 0.5 at t = 0.1. Plot density, velocity, and By profiles side by side. Identify the compound wave structure (fast rarefaction, compound wave, contact discontinuity, slow shock, fast rarefaction) in both solutions. Which flux function resolves the intermediate states more sharply? Quantify the difference by measuring the width of the contact discontinuity in each solution.

### Exercise 4: div B Error Monitoring
Modify the MHD_1D_Solver class to add a method that computes the discrete divergence of B at each cell interface using central differences. For 2D, the finite volume divergence at cell (i,j) is: (div B)_{i,j} = (Bx_{i+1/2,j} - Bx_{i-1/2,j})/dx + (By_{i,j+1/2} - By_{i,j-1/2})/dy. Implement a 2D toy problem (e.g., a smooth initial magnetic field vortex advected by a uniform flow) and plot the L2 norm of div B over time. Compare the div B growth rate with and without Powell source term corrections.

### Exercise 5: Grid Convergence Test
Run the Brio-Wu shock tube with grid resolutions Nx = 100, 200, 400, and 800 cells using the HLL flux. At t = 0.1, compute the L1 error in density relative to the Nx = 1600 reference solution: L1 = (1/Nx)Σ|ρ_num(xi) - ρ_ref(xi)|. Plot L1 versus dx on a log-log scale and measure the observed convergence rate. In smooth regions the scheme should approach first order, and near discontinuities the convergence rate should be reduced. Explain why using MUSCL reconstruction would improve convergence in smooth regions.

---

The next lesson covers plasma simulation and the PIC (Particle-In-Cell) method.
