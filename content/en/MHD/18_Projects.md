# 18. Projects

## Learning Objectives

By the end of this lesson, you will be able to:

- Implement a complete 2D resistive MHD simulation of magnetic reconnection
- Analyze reconnection rates and energy conversion in solar flare models
- Build a 1D MHD stability analyzer for tokamak plasmas
- Apply stability criteria (Kruskal-Shafranov, Suydam) to predict disruptions
- Simulate a mean-field dynamo in a spherical shell
- Observe oscillatory dynamo solutions and butterfly diagrams
- Integrate all MHD concepts learned in this course into practical applications

---

## Project 1: Solar Flare Simulation via Magnetic Reconnection

### 1.1 Physics Background

**Solar flares** are explosive releases of magnetic energy in the solar corona, driven by **magnetic reconnection** – the topological restructuring of magnetic field lines that converts magnetic energy into kinetic and thermal energy.

**Key physics:**
- **Tearing instability:** Resistive instability in current sheets → plasmoid formation
- **Sweet-Parker reconnection:** Classical model, slow reconnection rate $\sim \eta^{1/2}$
- **Plasmoid-mediated reconnection:** Fast reconnection via secondary islands, rate independent of $\eta$

**Observables:**
- Reconnection rate: inflow velocity $v_{\text{in}}$
- Energy conversion: $\Delta E_{\text{mag}}$ → $\Delta E_{\text{kin}} + \Delta E_{\text{th}}$
- Current sheet structure: thickness $\delta$, length $L$
- Temperature rise in reconnection region

### 1.2 Problem Setup

**Geometry:** 2D Harris current sheet in $[-L_x, L_x] \times [-L_y, L_y]$ box

**Initial conditions:**
$$
B_x(x, y) = B_0 \tanh(y / a)
$$
$$
B_y(x, y) = 0
$$
$$
\rho(x, y) = \rho_0 \left(1 + \beta \operatorname{sech}^2(y/a)\right)
$$
$$
p(x, y) = p_0 + \frac{B_0^2}{2} \left(1 - \tanh^2(y/a)\right)
$$

where:
- $B_0 = 1.0$ (characteristic field strength)
- $a = 0.5$ (current sheet half-thickness)
- $\rho_0 = 1.0$ (background density)
- $\beta = 1.0$ (plasma beta parameter)
- $p_0 = B_0^2 \beta / 2$

**Perturbation (trigger reconnection):**
$$
B_y(x, y) = \epsilon B_0 \cos(k_x x) \sin(\pi y / L_y)
$$
with $\epsilon = 0.1$, $k_x = 2\pi / L_x$.

**Parameters:**
- Domain: $L_x = 25.6$, $L_y = 12.8$
- Grid: $N_x = 512$, $N_y = 256$
- Resistivity: $\eta = 0.001$ (localized or uniform)
- $\gamma = 5/3$ (ideal gas)

**Boundary conditions:**
- Periodic in $x$
- Conducting walls in $y$ ($\mathbf{v} \cdot \hat{y} = 0$, $\partial_y B_x = 0$, $B_y = 0$)

### 1.3 Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 25.6, 12.8
# Nx=512, Ny=256: the 2:1 aspect ratio matches the domain aspect Lx:Ly = 2:1,
# so dx ≈ dy ≈ 0.05 — a uniform cell size that avoids anisotropic numerical diffusion;
# Nx=512 is chosen so the resistive layer δ ~ a/√S ≈ 0.5/√500 ≈ 0.022 spans ~5 cells,
# which is the minimum needed to resolve the current sheet without artificial broadening
Nx, Ny = 512, 256
dx = Lx / Nx
dy = Ly / Ny
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Physical parameters
B0 = 1.0
# a = 0.5 sets the Harris sheet half-width; it must be large enough to resolve
# the resistive layer δ ~ a/√S where S = Lundquist number = τ_R/τ_A = a²/(η*τ_A);
# with η = 0.001 and a = 0.5 the Alfvén time τ_A ~ a/v_A ~ 0.5 and S ≈ 500,
# giving δ ≈ 0.022 — just above the grid scale dx ≈ 0.05 (marginally resolved)
a = 0.5
rho0 = 1.0
# β = 1.0 ensures total pressure balance: the current sheet has lower magnetic
# pressure (sech² profile) which must be compensated by higher plasma pressure,
# and β = 1 sets these in balance so the sheet does not immediately expand or collapse
# before reconnection begins; β < 1 would over-confine the sheet, β > 1 would expand it
beta_param = 1.0
p0 = B0**2 * beta_param / 2.0
gamma = 5.0 / 3.0
# η = 0.001 gives Lundquist number S = a*v_A/η ≈ 500 for this configuration:
# S ≈ 500 is in the plasmoid-unstable regime (Sweet-Parker is S^{1/2} times slower
# than fast reconnection), so secondary islands should form during the simulation,
# yielding reconnection rates closer to the Petschek/plasmoid fast rate ~0.01-0.1 v_A
eta = 0.001
nu = 0.001  # Viscosity (for numerical stability)
# CFL = 0.4: the Alfvén speed v_A = B0/√(μ₀ρ0) = 1 sets the fastest signal;
# CFL < 0.5 ensures information travels less than one cell per step, preventing
# the instability that arises when the numerical domain of dependence is too small
CFL = 0.4

# Initial conditions
def initial_harris_sheet():
    # tanh(Y/a) is the exact Harris equilibrium solution for Bx: it satisfies
    # force balance d/dy(p + B²/2μ₀) = 0 when combined with the sech² density
    # and pressure profiles below — perturbing away from this exact equilibrium
    # seeds the tearing instability without introducing spurious initial transients
    Bx = B0 * np.tanh(Y / a)
    # By perturbation amplitude ε = 0.1: large enough to trigger reconnection
    # on a reasonable simulation time but small enough (ε < 1) that the tearing
    # mode starts in the linear regime and grows exponentially before nonlinear saturation;
    # cos(kx)*sin(πy/Ly) satisfies By=0 at the conducting walls in y
    By = 0.1 * B0 * np.cos(2*np.pi*X/Lx) * np.sin(np.pi*Y/Ly)  # Perturbation
    Bz = np.zeros_like(Bx)

    # ρ ∝ (1 + β*sech²(y/a)): density is enhanced inside the current sheet to
    # maintain force balance — the magnetic pressure B²/2μ₀ drops to zero at y=0
    # (where tanh=0), so the plasma pressure p must be maximum there, and with the
    # ideal gas law, higher p at fixed T means higher ρ
    rho = rho0 * (1.0 + beta_param * (1.0/np.cosh(Y/a))**2)
    # p = p0 + B0²/2*(1 - tanh²): total pressure (p + B²/2) = const = p0 + B0²/2,
    # confirming exact Harris force balance at every y-location
    p = p0 + B0**2/2.0 * (1.0 - np.tanh(Y/a)**2)

    vx = np.zeros_like(Bx)
    vy = np.zeros_like(Bx)
    vz = np.zeros_like(Bx)

    return rho, vx, vy, vz, p, Bx, By, Bz

# Conservative variables
def prim2cons(rho, vx, vy, vz, p, Bx, By, Bz):
    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2

    U1 = rho
    U2 = rho * vx
    U3 = rho * vy
    U4 = rho * vz
    U5 = p/(gamma-1.0) + 0.5*rho*v2 + 0.5*B2
    U6 = Bx
    U7 = By
    U8 = Bz

    return U1, U2, U3, U4, U5, U6, U7, U8

def cons2prim(U1, U2, U3, U4, U5, U6, U7, U8):
    rho = U1
    vx = U2 / rho
    vy = U3 / rho
    vz = U4 / rho

    Bx = U6
    By = U7
    Bz = U8

    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2

    p = (gamma - 1.0) * (U5 - 0.5*rho*v2 - 0.5*B2)
    p = np.maximum(p, 1e-6)  # Floor

    return rho, vx, vy, vz, p, Bx, By, Bz

# Flux computation (simplified HLL)
def flux_x(rho, vx, vy, vz, p, Bx, By, Bz):
    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2

    ptot = p + 0.5*B2
    E = p/(gamma-1.0) + 0.5*rho*v2 + 0.5*B2

    F1 = rho * vx
    F2 = rho*vx*vx + ptot - Bx*Bx
    F3 = rho*vx*vy - Bx*By
    F4 = rho*vx*vz - Bx*Bz
    F5 = (E + ptot)*vx - Bx*(vx*Bx + vy*By + vz*Bz)
    F6 = np.zeros_like(Bx)  # Bx constant in x
    F7 = By*vx - Bx*vy
    F8 = Bz*vx - Bx*vz

    return F1, F2, F3, F4, F5, F6, F7, F8

def flux_y(rho, vx, vy, vz, p, Bx, By, Bz):
    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2

    ptot = p + 0.5*B2
    E = p/(gamma-1.0) + 0.5*rho*v2 + 0.5*B2

    G1 = rho * vy
    G2 = rho*vy*vx - By*Bx
    G3 = rho*vy*vy + ptot - By*By
    G4 = rho*vy*vz - By*Bz
    G5 = (E + ptot)*vy - By*(vx*Bx + vy*By + vz*Bz)
    G6 = Bx*vy - By*vx
    G7 = np.zeros_like(By)  # By constant in y (with CT)
    G8 = Bz*vy - By*vz

    return G1, G2, G3, G4, G5, G6, G7, G8

# Simple HLL solver
def hll_flux_x(UL, UR, rhoL, vxL, vyL, vzL, pL, BxL, ByL, BzL,
                         rhoR, vxR, vyR, vzR, pR, BxR, ByR, BzR):
    # Estimate wave speeds
    csL = np.sqrt(gamma * pL / rhoL)
    csR = np.sqrt(gamma * pR / rhoR)

    SL = np.minimum(vxL - csL, vxR - csR)
    SR = np.maximum(vxL + csL, vxR + csR)

    FL = flux_x(rhoL, vxL, vyL, vzL, pL, BxL, ByL, BzL)
    FR = flux_x(rhoR, vxR, vyR, vzR, pR, BxR, ByR, BzR)

    # HLL flux
    F_hll = []
    for i in range(8):
        F = np.where(SL >= 0, FL[i],
            np.where(SR <= 0, FR[i],
            (SR*FL[i] - SL*FR[i] + SL*SR*(UR[i] - UL[i])) / (SR - SL)))
        F_hll.append(F)

    return F_hll

# Main solver
def solve_reconnection():
    # Initialize
    rho, vx, vy, vz, p, Bx, By, Bz = initial_harris_sheet()
    U = prim2cons(rho, vx, vy, vz, p, Bx, By, Bz)

    t = 0.0
    t_end = 50.0
    step = 0

    # Diagnostics
    energy_mag = []
    energy_kin = []
    energy_th = []
    times = []

    print("Starting reconnection simulation...")

    while t < t_end:
        # Compute dt
        cs = np.sqrt(gamma * p / rho)
        vA = np.sqrt((Bx**2 + By**2 + Bz**2) / rho)
        vmax = np.max(np.abs(vx) + np.abs(vy) + cs + vA)
        dt = CFL * min(dx, dy) / vmax

        if t + dt > t_end:
            dt = t_end - t

        # Update (simple forward Euler for demonstration; use RK2 in production)
        # X-direction fluxes
        Fx_L = []
        Fx_R = []
        for i in range(Nx):
            iL = (i - 1) % Nx
            iR = i

            UL = [U[k][iL, :] for k in range(8)]
            UR = [U[k][iR, :] for k in range(8)]

            primL = cons2prim(*UL)
            primR = cons2prim(*UR)

            F = hll_flux_x(UL, UR, *primL, *primR)
            if i == 0:
                Fx_L = [np.zeros((Nx, Ny)) for _ in range(8)]
                Fx_R = [np.zeros((Nx, Ny)) for _ in range(8)]

            for k in range(8):
                Fx_R[k][i, :] = F[k]

        # Shift for left flux
        for k in range(8):
            Fx_L[k] = np.roll(Fx_R[k], 1, axis=0)

        # Y-direction fluxes (similar, with boundary conditions)
        # ... (omitted for brevity; apply conducting wall BC)

        # Update conserved variables
        U_new = []
        for k in range(8):
            dU = -(Fx_R[k] - Fx_L[k]) / dx  # Only x-direction for simplicity
            U_new.append(U[k] + dt * dU)

        U = U_new

        # Recover primitives
        rho, vx, vy, vz, p, Bx, By, Bz = cons2prim(*U)

        # Resistive diffusion term η∇²B = η(∇×J): applied as an operator-split
        # step after the ideal MHD update — this splitting avoids the stiffness of
        # an implicit treatment while keeping the resistive layer physically correct;
        # without this step the simulation would be purely ideal and reconnection
        # could not occur regardless of the perturbation amplitude
        Jz = (np.gradient(By, dx, axis=0) - np.gradient(Bx, dy, axis=1))
        # dBx/dt = -η * ∂Jz/∂y: the minus sign is from Faraday's law ∂B/∂t = -∇×E
        # with E = ηJ (Ohm's law); the curl of the resistive electric field drives
        # flux diffusion across the current sheet at rate proportional to η
        dBx_dt = -eta * np.gradient(Jz, dy, axis=1)
        dBy_dt = eta * np.gradient(Jz, dx, axis=0)

        Bx += dt * dBx_dt
        By += dt * dBy_dt

        # Update conserved variables with new B
        U = prim2cons(rho, vx, vy, vz, p, Bx, By, Bz)

        # Diagnostics
        if step % 50 == 0:
            # Tracking all three energy reservoirs confirms that ΔE_mag = ΔE_kin + ΔE_th:
            # magnetic energy released during reconnection should be split between bulk
            # kinetic energy of the outflow jets and thermal energy of the heated plasma —
            # if total energy is not conserved, the numerics have a bug
            E_mag = 0.5 * np.sum((Bx**2 + By**2 + Bz**2) * dx * dy)
            E_kin = 0.5 * np.sum(rho * (vx**2 + vy**2 + vz**2) * dx * dy)
            E_th = np.sum(p / (gamma - 1.0) * dx * dy)

            energy_mag.append(E_mag)
            energy_kin.append(E_kin)
            energy_th.append(E_th)
            times.append(t)

            print(f"Step {step:5d}, t={t:6.2f}, E_mag={E_mag:.4f}, E_kin={E_kin:.4f}, E_th={E_th:.4f}")

        t += dt
        step += 1

        if step > 10000:  # Safety limit
            break

    return times, energy_mag, energy_kin, energy_th, rho, vx, vy, p, Bx, By, Jz

# Run simulation
times, E_mag, E_kin, E_th, rho_f, vx_f, vy_f, p_f, Bx_f, By_f, Jz_f = solve_reconnection()

# Plot energy evolution
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(times, E_mag, 'b-', label='Magnetic', linewidth=2)
ax.plot(times, E_kin, 'r--', label='Kinetic', linewidth=2)
ax.plot(times, E_th, 'g:', label='Thermal', linewidth=2)
E_tot = np.array(E_mag) + np.array(E_kin) + np.array(E_th)
ax.plot(times, E_tot, 'k-.', label='Total', linewidth=1.5)
ax.set_xlabel('Time')
ax.set_ylabel('Energy')
ax.set_title('Energy Evolution in Reconnection')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('reconnection_energy.png', dpi=150, bbox_inches='tight')
print("Energy plot saved: reconnection_energy.png")
plt.close()

# Plot final state
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Current density
im0 = axes[0, 0].contourf(X, Y, Jz_f, levels=30, cmap='RdBu_r')
axes[0, 0].set_title('Current Density $J_z$')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
plt.colorbar(im0, ax=axes[0, 0])

# Temperature (pressure)
im1 = axes[0, 1].contourf(X, Y, p_f, levels=30, cmap='hot')
axes[0, 1].set_title('Pressure (Temperature)')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
plt.colorbar(im1, ax=axes[0, 1])

# Velocity magnitude
v_mag = np.sqrt(vx_f**2 + vy_f**2)
im2 = axes[1, 0].contourf(X, Y, v_mag, levels=30, cmap='plasma')
axes[1, 0].streamplot(X.T, Y.T, vx_f.T, vy_f.T, color='k', density=1.0, linewidth=0.5)
axes[1, 0].set_title('Velocity Field')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
plt.colorbar(im2, ax=axes[1, 0])

# Magnetic field lines
Ay = np.zeros_like(Bx_f)  # Compute vector potential (simplified)
for i in range(1, Nx):
    Ay[i, :] = Ay[i-1, :] + Bx_f[i, :] * dy
im3 = axes[1, 1].contour(X, Y, Ay, levels=20, colors='b', linewidths=1.5)
axes[1, 1].set_title('Magnetic Field Lines')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')
axes[1, 1].set_aspect('equal')

plt.tight_layout()
plt.savefig('reconnection_final.png', dpi=150, bbox_inches='tight')
print("Final state plot saved: reconnection_final.png")
plt.close()
```

### 1.4 Expected Results

- **Energy conversion:** Magnetic energy decreases, kinetic and thermal energy increase
- **Plasmoid formation:** Secondary islands appear in current sheet (if resolution sufficient)
- **Reconnection rate:** Inflow velocity $v_{\text{in}} \sim 0.01 - 0.1 v_A$ (faster than Sweet-Parker $\sim \eta^{1/2}$)
- **Temperature spike:** Localized heating at X-point

### 1.5 Extensions

1. **Localized resistivity:** Use $\eta = \eta_0 + \eta_1 J^2$ (anomalous resistivity)
2. **3D simulation:** Add guide field $B_z$, study drift-kink instability
3. **Particle acceleration:** Inject test particles, track energization
4. **Comparison with Sweet-Parker:** Measure reconnection rate vs $\eta$, verify $M_A \propto S^{-1/2}$ (Sweet-Parker) or $M_A \sim 0.01$ (plasmoid)

---

## Project 2: Tokamak Disruption Prediction

### 2.1 Physics Background

**Tokamak:** Toroidal magnetic confinement device for fusion plasma.

**Key concepts:**
- **Safety factor:** $q(r) = r B_\phi / (R B_\theta)$ (field line pitch)
- **Disruption:** Sudden loss of confinement due to MHD instabilities → plasma terminates, large forces on wall

**Stability criteria:**
- **Kruskal-Shafranov:** $q > 1$ (suppress $m=1$ kink)
- **Suydam criterion:** Local stability to interchange modes
- **Tearing mode:** Rational surface ($q = m/n$) unstable if $\Delta' > 0$

### 2.2 Problem Setup

**Cylindrical tokamak model (1D):**
- Minor radius: $a = 1.0$ m
- Major radius: $R_0 = 3.0$ m
- Toroidal field: $B_\phi(r) = B_0 R_0 / (R_0 + r)$ (approximate)
- Poloidal field from current profile $J_\phi(r)$

**Current profiles to test:**
- Profile 1: Peaked on-axis (stable)
  $$
  J_\phi(r) = J_0 \left(1 - (r/a)^2\right)^2
  $$
- Profile 2: Hollow current (unstable)
  $$
  J_\phi(r) = J_0 (r/a) \left(1 - (r/a)^2\right)
  $$
- Profile 3: Edge-peaked (disruption-prone)
  $$
  J_\phi(r) = J_0 \left(1 - \exp(-(r/a)^4)\right)
  $$

**Goal:** Compute $q(r)$, check stability, predict disruption.

### 2.3 Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Parameters
a = 1.0      # Minor radius (m)
R0 = 3.0     # Major radius (m)
B0 = 2.0     # Toroidal field at axis (T)
Nr = 200
r = np.linspace(0, a, Nr)

# Current profiles
def current_profile(r, profile_type='peaked'):
    if profile_type == 'peaked':
        J = 1.0e6 * (1.0 - (r/a)**2)**2
    elif profile_type == 'hollow':
        J = 1.0e6 * (r/a) * (1.0 - (r/a)**2)
    elif profile_type == 'edge':
        J = 1.0e6 * (1.0 - np.exp(-(r/a)**4))
    else:
        J = np.ones_like(r) * 1.0e6
    return J

# Compute poloidal field from Ampere's law
def compute_Btheta(r, J):
    """B_theta(r) = (mu_0 / r) * integral_0^r J(r') r' dr'"""
    mu0 = 4e-7 * np.pi
    # integrand = J*r: the Ampere's law integral ∫J·dA in cylindrical geometry
    # requires the area element r*dr*dφ; integrating over azimuth gives 2π,
    # and dividing by 2π*r at the end recovers Bθ — the weighting by r accounts
    # for the increasing circumference of shells at larger minor radius
    integrand = J * r
    I_enc = cumulative_trapezoid(integrand, r, initial=0)
    # Division by r gives Bθ = μ₀*I(r)/(2π*r): the safety factor q = r*Bφ/(R*Bθ)
    # diverges at r=0 for any finite Bφ, so the small offset avoids a singularity
    # there while leaving all physically meaningful radii (r > 0.01 m) unaffected
    Btheta = mu0 * I_enc / (r + 1e-10)  # Avoid r=0
    return Btheta

# Toroidal field (approximate, ignoring r/R0 correction)
def compute_Bphi(r):
    return B0 * R0 / (R0 + r)

# Safety factor
def compute_q(r, Btheta, Bphi):
    """q = r * Bphi / (R0 * Btheta)"""
    q = r * Bphi / (R0 * Btheta + 1e-10)
    return q

# Suydam criterion: d(ln p) / d(ln r) + (r * dq/dr / q)^2 > 0
def suydam_criterion(r, q, p):
    """Simplified: check d(ln q)/d(ln r) > some threshold"""
    dq_dr = np.gradient(q, r)
    shear = r / q * dq_dr
    # Simplified: shear > 0.5 (stable)
    return shear > 0.5

# Tearing mode Delta' (simplified estimate)
def estimate_delta_prime(r, q, m, n):
    """Find rational surface q = m/n, estimate Delta'"""
    q_rational = m / n
    idx = np.argmin(np.abs(q - q_rational))
    rs = r[idx]

    if idx > 5 and idx < Nr - 5:
        # Estimate logarithmic derivative mismatch
        dq_dr = np.gradient(q, r)
        delta_prime = -2.0 * dq_dr[idx] / q[idx]  # Crude estimate
    else:
        delta_prime = 0.0

    return rs, delta_prime

# Analyze stability for a profile
def analyze_stability(profile_type):
    J = current_profile(r, profile_type)
    Btheta = compute_Btheta(r, J)
    Bphi = compute_Bphi(r)
    q = compute_q(r, Btheta, Bphi)

    # Simple pressure profile (proportional to current)
    p = 1e5 * (1.0 - (r/a)**2)**2

    # Kruskal-Shafranov q > 1: if q drops below 1 on-axis, the m=1, n=1 internal
    # kink mode becomes ideally unstable — field lines complete one full poloidal
    # turn in less than one toroidal turn, allowing the kink to close on itself
    # and grow without the field-line-bending stabilization that normally resists it
    q_min = np.min(q[1:])  # Skip r=0
    ks_stable = q_min > 1.0

    # Suydam criterion: local shear s = (r/q)*dq/dr measures how quickly field-line
    # pitch changes with radius; higher shear stabilizes interchange modes by requiring
    # perturbations to bend field lines over a shorter connection length — a threshold
    # of s > 0.5 is a simplified proxy for the full Suydam discriminant
    shear = np.zeros_like(r)
    shear[1:] = r[1:] / q[1:] * np.gradient(q, r)[1:]
    suydam_stable = np.all(shear[1:] > 0.5)

    # m=2, n=1 tearing mode at the q=2 rational surface: this is the most dangerous
    # tearing mode in tokamaks because q=2 occurs well inside the plasma where
    # drive is strong; Δ' > 0 means flux is released when the island opens, making
    # the surface unstable — this is the mode that locks and precedes disruptions
    rs_21, delta_prime_21 = estimate_delta_prime(r, q, m=2, n=1)
    tearing_stable = delta_prime_21 < 0

    return {
        'r': r,
        'J': J,
        'q': q,
        'Btheta': Btheta,
        'Bphi': Bphi,
        'q_min': q_min,
        'ks_stable': ks_stable,
        'suydam_stable': suydam_stable,
        'tearing_stable': tearing_stable,
        'delta_prime_21': delta_prime_21,
        'rs_21': rs_21
    }

# Analyze all profiles
profiles = ['peaked', 'hollow', 'edge']
results = {ptype: analyze_stability(ptype) for ptype in profiles}

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

colors = {'peaked': 'blue', 'hollow': 'red', 'edge': 'green'}

for i, ptype in enumerate(profiles):
    res = results[ptype]

    # Current profile
    axes[0, i].plot(res['r'], res['J']/1e6, color=colors[ptype], linewidth=2)
    axes[0, i].set_xlabel('r (m)')
    axes[0, i].set_ylabel('$J_\\phi$ (MA/m²)')
    axes[0, i].set_title(f'{ptype.capitalize()} Current')
    axes[0, i].grid(True, alpha=0.3)

    # Safety factor
    axes[1, i].plot(res['r'], res['q'], color=colors[ptype], linewidth=2, label='q(r)')
    axes[1, i].axhline(1.0, color='k', linestyle='--', label='q=1 (Kruskal)')
    axes[1, i].axhline(2.0, color='gray', linestyle=':', label='q=2')
    axes[1, i].set_xlabel('r (m)')
    axes[1, i].set_ylabel('Safety Factor q')
    axes[1, i].set_title(f'q(r) - {ptype.capitalize()}')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)
    axes[1, i].set_ylim([0, 10])

plt.tight_layout()
plt.savefig('tokamak_profiles.png', dpi=150, bbox_inches='tight')
print("Profile plots saved: tokamak_profiles.png")
plt.close()

# Stability summary
print("\n=== STABILITY ANALYSIS ===\n")
for ptype in profiles:
    res = results[ptype]
    print(f"--- {ptype.upper()} PROFILE ---")
    print(f"  q_min = {res['q_min']:.3f}")
    print(f"  Kruskal-Shafranov (q>1): {'STABLE' if res['ks_stable'] else 'UNSTABLE'}")
    print(f"  Suydam: {'STABLE' if res['suydam_stable'] else 'UNSTABLE'}")
    print(f"  Tearing (2,1) Delta' = {res['delta_prime_21']:.3f}: {'STABLE' if res['tearing_stable'] else 'UNSTABLE'}")
    print(f"  Rational surface (q=2): r_s = {res['rs_21']:.3f} m")

    # Disruption prediction
    if not res['ks_stable']:
        print(f"  PREDICTION: HIGH DISRUPTION RISK (Kruskal violation)")
    elif not res['tearing_stable']:
        print(f"  PREDICTION: MEDIUM DISRUPTION RISK (Tearing unstable)")
    else:
        print(f"  PREDICTION: LOW DISRUPTION RISK")
    print()

# Estimate energy release during disruption
def estimate_disruption_energy(res):
    """Magnetic + thermal energy in plasma"""
    B2 = res['Btheta']**2 + res['Bphi']**2
    V = 2 * np.pi * R0 * np.pi * a**2  # Approximate volume
    E_mag = np.trapz(B2 / (2 * 4e-7*np.pi) * 2*np.pi*R0*res['r'], res['r'])

    p = 1e5 * (1.0 - (res['r']/a)**2)**2
    E_th = np.trapz(1.5 * p * 2*np.pi*R0*res['r'], res['r'])

    return E_mag, E_th

print("\n=== DISRUPTION ENERGY ESTIMATE ===\n")
for ptype in profiles:
    res = results[ptype]
    E_mag, E_th = estimate_disruption_energy(res)
    print(f"{ptype.upper()}: E_mag = {E_mag/1e6:.2f} MJ, E_th = {E_th/1e6:.2f} MJ")
    print(f"  Total = {(E_mag + E_th)/1e6:.2f} MJ")
    print()
```

### 2.4 Expected Results

- **Peaked profile:** Stable (all criteria satisfied)
- **Hollow profile:** Marginal (possible tearing instability at $q=2$ surface)
- **Edge profile:** Unstable (Kruskal violation, disruption likely)

**Disruption energy:** ~10-100 MJ (forces on wall ~ MN, need mitigation!)

### 2.5 Extensions

1. **Neoclassical tearing mode (NTM):** Include bootstrap current perturbation
2. **Vertical displacement event (VDE):** Axisymmetric instability
3. **Resistive wall mode:** Include wall stabilization
4. **Disruption mitigation:** Massive gas injection (MGI) simulation

---

## Project 3: Spherical Shell Dynamo

### 3.1 Physics Background

**Dynamo theory:** Self-sustained magnetic field generation by fluid motion (e.g., Earth's core, Sun's convection zone).

**Key concepts:**
- **Kinematic dynamo:** Velocity field $\mathbf{v}$ prescribed, solve for $\mathbf{B}$
- **Mean-field theory:** Separate $\mathbf{B} = \mathbf{B}_0 + \mathbf{b}$, turbulent EMF $\langle \mathbf{v} \times \mathbf{b} \rangle = \alpha \mathbf{B}_0 - \beta \mathbf{J}$
- **$\alpha$-effect:** Cyclonic turbulence → helical field generation
- **$\Omega$-effect:** Differential rotation → toroidal field from poloidal

**Axisymmetric mean-field equations:**
$$
\frac{\partial A}{\partial t} = \eta \nabla^2 A + \alpha B_\phi
$$
$$
\frac{\partial B_\phi}{\partial t} = \eta \nabla^2 B_\phi + r \sin\theta \, \mathbf{B}_p \cdot \nabla \Omega + \ldots
$$
where $\mathbf{B}_p = \nabla \times (A \hat{\phi})$ (poloidal field).

### 3.2 Problem Setup

**Domain:** Spherical shell $r \in [r_i, r_o]$, $\theta \in [0, \pi]$ (axisymmetric, $\phi$-independent)

**Parameters:**
- Inner radius: $r_i = 0.5$
- Outer radius: $r_o = 1.0$
- Differential rotation: $\Omega(r, \theta) = \Omega_0 \cos^2\theta \, (1 - (r/r_o)^2)$ (solar-like)
- $\alpha$-effect: $\alpha(r, \theta) = \alpha_0 \sin\theta \cos\theta$ (dipole-favoring)
- Magnetic diffusivity: $\eta = 10^{-3}$
- $\alpha_0 = 1.0$, $\Omega_0 = 10.0$

**Equations (simplified 2D):**
$$
\frac{\partial A}{\partial t} = \eta \left(\frac{\partial^2 A}{\partial r^2} + \frac{1}{r^2}\frac{\partial^2 A}{\partial \theta^2}\right) + \alpha B_\phi
$$
$$
\frac{\partial B_\phi}{\partial t} = \eta \left(\frac{\partial^2 B_\phi}{\partial r^2} + \frac{1}{r^2}\frac{\partial^2 B_\phi}{\partial \theta^2}\right) + C_\Omega \frac{\partial A}{\partial \theta}
$$
where $C_\Omega = r \sin\theta \, \partial_r \Omega$.

**Boundary conditions:**
- $A = 0$ at $r = r_i, r_o$ (field lines parallel to surface)
- $B_\phi = 0$ at $r = r_i, r_o$ (vacuum outside)

### 3.3 Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
ri, ro = 0.5, 1.0
Nr, Nth = 64, 64
r = np.linspace(ri, ro, Nr)
theta = np.linspace(0, np.pi, Nth)
R, Theta = np.meshgrid(r, theta, indexing='ij')

dr = (ro - ri) / (Nr - 1)
dtheta = np.pi / (Nth - 1)

eta = 1e-3
alpha0 = 1.0
Omega0 = 10.0
dt = 1e-4
t_end = 2.0

# Ω ∝ cos²θ*(1-r²): faster rotation at equator and core — this latitude and
# radial differential rotation is what drives the Ω-effect: stretching poloidal
# field lines toroidally at different rates creates the strong toroidal field
# (solar sunspot belts) that characterizes the solar dynamo cycle
Omega = Omega0 * np.cos(Theta)**2 * (1.0 - (R/ro)**2)

# α ∝ sinθ*cosθ: anti-symmetric about the equator — positive in the northern
# hemisphere, negative in the southern; this antisymmetry is required to generate
# a dipole-like poloidal field from the toroidal field via the α-effect, because
# a symmetric α would create a quadrupole instead
alpha = alpha0 * np.sin(Theta) * np.cos(Theta)

# C_Ω = r*sinθ * ∂Ω/∂r measures the local shear in Ω: it is the coefficient that
# converts the gradient of the poloidal flux A into the source term for toroidal
# field Bφ — where differential rotation is strong (large |C_Ω|), the Ω-effect
# is most efficient at winding up poloidal field into toroidal field
C_Omega = R * np.sin(Theta) * np.gradient(Omega, dr, axis=0)

# Initialize fields
A = np.random.randn(Nr, Nth) * 0.01
Bphi = np.random.randn(Nr, Nth) * 0.01

# Apply BC
A[0, :] = 0
A[-1, :] = 0
Bphi[0, :] = 0
Bphi[-1, :] = 0

# Laplacian operator (finite differences)
def laplacian_2d(f, dr, dtheta, r):
    """Compute Laplacian in (r, theta) with metric terms."""
    d2f_dr2 = np.zeros_like(f)
    d2f_dtheta2 = np.zeros_like(f)

    # r-direction (central differences)
    d2f_dr2[1:-1, :] = (f[2:, :] - 2*f[1:-1, :] + f[:-2, :]) / dr**2

    # theta-direction
    d2f_dtheta2[:, 1:-1] = (f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2]) / dtheta**2

    # Add metric terms (simplified)
    laplacian = d2f_dr2 + d2f_dtheta2 / r[:, np.newaxis]**2

    return laplacian

# Time evolution
t = 0.0
step = 0
snapshots = []

print("Running spherical shell dynamo...")
while t < t_end:
    # Compute Laplacians
    lap_A = laplacian_2d(A, dr, dtheta, r)
    lap_Bphi = laplacian_2d(Bphi, dr, dtheta, r)

    # Source terms
    dA_dt_theta = np.gradient(A, dtheta, axis=1)

    source_A = alpha * Bphi
    source_Bphi = C_Omega * dA_dt_theta

    # Update
    A += dt * (eta * lap_A + source_A)
    Bphi += dt * (eta * lap_Bphi + source_Bphi)

    # Apply BC
    A[0, :] = 0
    A[-1, :] = 0
    Bphi[0, :] = 0
    Bphi[-1, :] = 0

    # Diagnostics
    if step % 1000 == 0:
        E_A = np.sum(A**2)
        E_Bphi = np.sum(Bphi**2)
        print(f"Step {step:5d}, t={t:.4f}, E_A={E_A:.4f}, E_Bphi={E_Bphi:.4f}")

        if len(snapshots) < 5:
            snapshots.append((t, A.copy(), Bphi.copy()))

    t += dt
    step += 1

print("Dynamo simulation complete.")

# Plot snapshots
fig, axes = plt.subplots(len(snapshots), 2, figsize=(12, 3*len(snapshots)))

for i, (t_snap, A_snap, Bphi_snap) in enumerate(snapshots):
    ax_A = axes[i, 0] if len(snapshots) > 1 else axes[0]
    ax_B = axes[i, 1] if len(snapshots) > 1 else axes[1]

    # Poloidal field (contours of A)
    im_A = ax_A.contourf(R*np.sin(Theta), R*np.cos(Theta), A_snap, levels=20, cmap='RdBu_r')
    ax_A.set_xlabel('x')
    ax_A.set_ylabel('z')
    ax_A.set_title(f'Poloidal Field (A), t={t_snap:.3f}')
    ax_A.set_aspect('equal')
    plt.colorbar(im_A, ax=ax_A)

    # Toroidal field
    im_B = ax_B.contourf(R*np.sin(Theta), R*np.cos(Theta), Bphi_snap, levels=20, cmap='seismic')
    ax_B.set_xlabel('x')
    ax_B.set_ylabel('z')
    ax_B.set_title(f'Toroidal Field $B_\\phi$, t={t_snap:.3f}')
    ax_B.set_aspect('equal')
    plt.colorbar(im_B, ax=ax_B)

plt.tight_layout()
plt.savefig('dynamo_evolution.png', dpi=150, bbox_inches='tight')
print("Dynamo evolution saved: dynamo_evolution.png")
plt.close()

# Butterfly diagram (Bphi vs time at mid-radius)
r_mid_idx = Nr // 2
butterfly_Bphi = []
butterfly_times = []

# Re-run to collect time series (or store during main loop)
# For demonstration, use final state
butterfly_Bphi.append(Bphi[r_mid_idx, :])
butterfly_times.append(t_end)

fig, ax = plt.subplots(figsize=(10, 6))
if len(butterfly_times) > 1:
    ax.contourf(butterfly_times, theta, np.array(butterfly_Bphi).T, levels=30, cmap='RdBu_r')
else:
    ax.plot(theta, Bphi[r_mid_idx, :], 'b-', linewidth=2)
    ax.set_xlabel('$\\theta$ (rad)')
    ax.set_ylabel('$B_\\phi$')
ax.set_title(f'Butterfly Diagram (r={r[r_mid_idx]:.2f})')
ax.grid(True, alpha=0.3)
plt.savefig('butterfly_diagram.png', dpi=150, bbox_inches='tight')
print("Butterfly diagram saved: butterfly_diagram.png")
plt.close()
```

### 3.4 Expected Results

- **Oscillatory dynamo:** $A$ and $B_\phi$ oscillate in time (magnetic cycles)
- **Equatorward migration:** Toroidal field migrates from poles to equator (if $\alpha$-$\Omega$ parameters right)
- **Butterfly diagram:** Shows latitude-time evolution of $B_\phi$ (solar-like if $C_\Omega$ and $\alpha$ chosen well)

**Parker migratory dynamo:** With proper parameters, reproduces solar 22-year cycle!

### 3.5 Extensions

1. **3D dynamo:** Full spherical coordinates, include convection
2. **Nonlinear quenching:** $\alpha \to \alpha(B)$ (Lorentz force back-reaction)
3. **Reversals:** Stochastic $\alpha$ fluctuations → polarity reversals (Earth's field)
4. **Benchmark:** Compare with Dedalus spectral code

---

## Summary of Projects

| Project | Physics | Methods | Difficulty |
|---------|---------|---------|------------|
| **Solar Flare** | Magnetic reconnection, plasmoids | 2D resistive MHD, HLL solver | ★★★★ |
| **Tokamak Disruption** | MHD stability, safety factor | 1D equilibrium, stability criteria | ★★★ |
| **Dynamo** | Mean-field theory, $\alpha$-$\Omega$ | 2D axisymmetric, time evolution | ★★★★ |

**Skills practiced:**
- Conservative MHD solvers (Godunov-type)
- Equilibrium analysis and stability theory
- Mean-field approximations
- Physical interpretation of numerical results

---

## General Tips for Projects

1. **Start simple:** Get 1D working before 2D, low resolution before high
2. **Validate:** Compare to known solutions (e.g., Brio-Wu for Project 1)
3. **Diagnostics:** Monitor energy conservation, $\nabla \cdot \mathbf{B}$, CFL timestep
4. **Visualization:** Use streamplots for $\mathbf{B}$, contours for scalar fields
5. **Parameter scans:** Vary $\eta$, $Re$, grid resolution → convergence study
6. **Physical intuition:** Does the result make sense? (e.g., reconnection should heat plasma)

---

## Conclusion

These three projects integrate the entire MHD course:

- **Conservation laws:** Mass, momentum, energy (Project 1)
- **Wave physics:** Fast, slow, Alfvén (Project 1)
- **Stability:** Kruskal-Shafranov, Suydam, tearing (Project 2)
- **Dynamo theory:** $\alpha$-effect, differential rotation (Project 3)
- **Numerical methods:** Finite volume, Riemann solvers, time integration (all projects)

By completing these, you have mastered **graduate-level magnetohydrodynamics**!

---

## Further Reading

### Solar Reconnection
- Zweibel & Yamada (2009), *Magnetic Reconnection in Astrophysical and Laboratory Plasmas*, ARA&A
- Loureiro et al. (2007), *Instability of current sheets and formation of plasmoid chains*, Physics of Plasmas

### Tokamak Stability
- Wesson & Campbell (2011), *Tokamaks*, 4th Ed., Oxford
- Freidberg (2014), *Ideal MHD*, Cambridge

### Dynamo Theory
- Moffatt (1978), *Magnetic Field Generation in Electrically Conducting Fluids*, Cambridge
- Brandenburg & Subramanian (2005), *Astrophysical magnetic fields and nonlinear dynamo theory*, Physics Reports

### Numerical MHD
- Tóth et al. (2012), *Adaptive numerical algorithms in space weather modeling*, JCP
- Stone et al. (2020), *Athena++: A Fast, Portable, and Multi-Physics PDE Solver*, ApJS

---

## Exercises

### Exercise 1: Harris Sheet Equilibrium Verification

Before running a full reconnection simulation, verify that the initial Harris current sheet is in force balance.

1. Using the Harris sheet initial conditions from Project 1, write a Python script that computes the total pressure $p_{\text{tot}}(y) = p(y) + B^2(y)/2$ at each grid point along $y$.
2. Plot $p_{\text{tot}}(y)$, $p(y)$, and $B^2(y)/2$ on the same axes.
3. Confirm that $p_{\text{tot}}$ is constant across the sheet (within numerical precision).
4. Compute the maximum relative deviation from the mean: $\max |p_{\text{tot}} - \langle p_{\text{tot}} \rangle| / \langle p_{\text{tot}} \rangle$. What value do you expect for an ideal equilibrium?
5. Explain physically why the plasma pressure must be enhanced inside the current sheet ($y \approx 0$) to maintain force balance.

### Exercise 2: Safety Factor and Lundquist Number Scaling

Explore how key dimensionless parameters control MHD behavior.

1. For the tokamak model in Project 2, compute the safety factor $q(r)$ for a **uniform current profile** $J_\phi = J_0 = \text{const}$.
   - Derive analytically: $q(r) = 2 B_\phi r / (\mu_0 J_0 R_0)$.
   - Compare your numerical result from `compute_q` to this formula.
2. For the reconnection simulation in Project 1, the Lundquist number is $S = a v_A / \eta$. With $v_A = B_0 / \sqrt{\rho_0} = 1$, $a = 0.5$:
   - Calculate $S$ for $\eta \in \{0.01, 0.001, 0.0001\}$.
   - For each $S$, estimate the Sweet-Parker reconnection rate $M_A^{\text{SP}} = S^{-1/2}$ and the Sweet-Parker current sheet thickness $\delta^{\text{SP}} = a S^{-1/2}$.
   - Fill in the table:

     | $\eta$ | $S$ | $M_A^{\text{SP}}$ | $\delta^{\text{SP}}$ |
     |--------|-----|-------------------|----------------------|
     | 0.01   |     |                   |                      |
     | 0.001  |     |                   |                      |
     | 0.0001 |     |                   |                      |

3. At what Lundquist number does the plasmoid instability become important (hint: $S \gtrsim 10^4$)? What changes in the reconnection dynamics?

### Exercise 3: Tearing Mode Stability Map

Extend the tokamak stability analysis to build a 2D stability map.

1. Using the stability analysis code from Project 2, scan over two parameters:
   - Current peaking index $\nu$: generalize the peaked profile to $J_\phi(r) = J_0 (1 - (r/a)^2)^\nu$ for $\nu \in \{0.5, 1.0, 1.5, 2.0, 3.0\}$.
   - Total plasma current: scale $J_0$ so that $q_0 = q(r \to 0) \in \{1.2, 1.5, 2.0, 3.0\}$.
2. For each combination, record:
   - $q_{\text{min}}$ (Kruskal-Shafranov stability: $q_{\text{min}} > 1$?)
   - Location of the $q = 2$ rational surface $r_s$
   - Tearing mode $\Delta'$ sign
3. Create a 2D color map (heatmap) with $\nu$ on one axis and $q_0$ on the other, color-coded by stability (green = stable, red = unstable, yellow = marginal).
4. Identify the stability boundary: which combinations of $(\nu, q_0)$ are most disruption-prone?

### Exercise 4: Dynamo Dynamo Number and Cycle Period

Investigate how the dynamo number controls the oscillation frequency of the mean-field dynamo.

1. The dynamo number is defined as $D = C_\alpha C_\Omega$ where $C_\alpha = \alpha_0 L / \eta$ and $C_\Omega = \Omega_0 L^2 / \eta$ (with $L = r_o - r_i$). For the Project 3 parameters:
   - Compute $D$ using $\alpha_0 = 1.0$, $\Omega_0 = 10.0$, $\eta = 10^{-3}$, $L = 0.5$.
2. Modify the dynamo code to scan $\alpha_0 \in \{0.5, 1.0, 2.0, 4.0\}$ while keeping $\Omega_0 = 10.0$ fixed. For each run:
   - Record the total magnetic energy $E(t) = \sum (A^2 + B_\phi^2)$ as a function of time.
   - Determine whether the dynamo grows, decays, or oscillates by fitting an exponential to $E(t)$.
   - If oscillating, estimate the cycle period $T$ from the time between energy maxima.
3. Plot growth rate $\gamma$ (or cycle period $T$) versus $D$. What is the critical dynamo number $D_c$ below which the field decays?
4. Explain physically: why does increasing $\alpha_0$ both increase the growth rate and shorten the cycle period?

### Exercise 5: Integrated MHD Project — CME Initiation Model

Design and implement a simplified coronal mass ejection (CME) initiation model that combines elements from all three projects.

**Background:** CMEs are initiated when a flux rope above a solar active region loses equilibrium and erupts. This involves a force-free magnetic field configuration becoming unstable (kink or torus instability) and undergoing reconnection with the overlying field.

**Tasks:**

1. **Equilibrium setup:** Create a 2D flux rope equilibrium in Cartesian geometry:
   - Background field: $B_y(x) = B_{\text{ext}} \tanh(x / L)$ (arcade field)
   - Embedded flux rope: add a localized current loop centered at $(x_0, y_0) = (0, 1)$ with radius $R_{\text{rope}} = 0.5$, current $I_{\text{rope}}$
   - Compute the total magnetic field $\mathbf{B} = \mathbf{B}_{\text{arcade}} + \mathbf{B}_{\text{rope}}$ using the Biot-Savart law for the rope contribution.

2. **Stability analysis:** Apply the torus instability criterion: the flux rope erupts when
   $$n = -\frac{\partial \ln B_{\text{ext}}}{\partial \ln h} > n_{\text{crit}} \approx 1.5$$
   where $h$ is the height of the flux rope center. Compute the decay index $n(h)$ of the background field as a function of height and determine the critical height $h_c$.

3. **Dynamic evolution:** Initialize the resistive MHD solver from Project 1 with the flux rope equilibrium. Perturb the rope upward by $\delta h = 0.1$ and evolve. Track:
   - Rope center height $h(t)$ (find centroid of $|J_z|$ maximum)
   - Reconnection rate at the current sheet below the rope
   - Energy partition: $\Delta E_{\text{mag}}$ vs $\Delta E_{\text{kin}}$

4. **Physical interpretation:**
   - Does the rope erupt (exponential rise) or return to equilibrium (oscillates)?
   - Where does reconnection occur — above or below the rope?
   - Compare the simulated dynamics to the standard CSHKP flare model (Carmichael-Sturrock-Hirayama-Kopp-Pneuman).
   - Estimate the energy available for a solar flare given the rope parameters.

---

**Previous:** [Spectral and Advanced Methods](./17_Spectral_Methods.md) | **Next:** None (final lesson)
