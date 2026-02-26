"""
Radiation Belt Dynamics: Radial Diffusion of Phase Space Density.

Solves the 1D radial diffusion equation for relativistic electron phase
space density (PSD) in the radiation belts using the Crank-Nicolson finite
difference method. The model captures storm-time enhancement from increased
radial diffusion and outer boundary injection, followed by recovery.

Key physics:
  - Radial diffusion equation (in L-shell coordinates):
    df/dt = L^2 * d/dL * (D_LL / L^2 * df/dL) + S - f/tau_loss
  - D_LL = D0 * L^10 * (Kp/1)^2: radial diffusion coefficient
    Strong L-dependence (L^10) means outer belt diffuses much faster
  - Phase space density f(L, t) at fixed first adiabatic invariant mu
  - Slot region (L ~ 2-3): low PSD due to wave-driven losses
  - Inner belt (L ~ 1.5-2.5): stable, primarily protons
  - Outer belt (L ~ 3-7): dynamic, driven by radial diffusion + local acceleration
  - Crank-Nicolson scheme: unconditionally stable, second-order accurate
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 65)
print("Radiation Belt Radial Diffusion Model (Crank-Nicolson)")
print("=" * 65)


# =========================================================================
# 1. GRID AND PARAMETERS
# =========================================================================
# Spatial grid: L-shell from 1.0 to 7.0
L_min, L_max = 1.0, 7.0
NL = 200                    # number of spatial grid points
L = np.linspace(L_min, L_max, NL)
dL = L[1] - L[0]

# Temporal grid: 5 days (120 hours) in hours
T_total = 120.0             # total simulation time [hours]
dt = 0.1                    # time step [hours]
Nt = int(T_total / dt)
t_arr = np.arange(0, T_total, dt)

print(f"\nGrid parameters:")
print(f"  L range: [{L_min}, {L_max}], NL = {NL}, dL = {dL:.4f}")
print(f"  Time: {T_total} hours, Nt = {Nt}, dt = {dt} hours")


# =========================================================================
# 2. PHYSICAL PARAMETERS
# =========================================================================

# --- Kp index profile (synthetic storm) ---
# Quiet (0-24h) -> Storm (24-48h, Kp peaks at ~6) -> Recovery (48-120h)
def kp_profile(t):
    """Synthetic Kp index as a function of time [hours]."""
    if t < 24:
        return 1.0  # quiet
    elif t < 30:
        return 1.0 + 5.0 * (t - 24) / 6.0  # ramp up
    elif t < 42:
        return 6.0  # storm main phase
    elif t < 54:
        return 6.0 - 4.0 * (t - 42) / 12.0  # declining
    else:
        return 2.0 * np.exp(-(t - 54) / 30.0) + 1.0  # slow recovery


Kp_arr = np.array([kp_profile(t) for t in t_arr])

# --- Radial diffusion coefficient D_LL ---
# D_LL = D0 * L^10 * (Kp/Kp_ref)^2  [R_E^2/hour]
# D0 is chosen so that diffusion timescale at L=5 is ~days during quiet
# and ~hours during storms.
D0 = 1e-10  # base diffusion coefficient [R_E^2/hour / L^10]
Kp_ref = 1.0


def compute_DLL(L_arr, Kp):
    """Compute D_LL at each L for given Kp."""
    return D0 * L_arr**10 * (Kp / Kp_ref)**2


# Print D_LL at key locations
for Kp_val in [1, 3, 6]:
    DLL_test = compute_DLL(np.array([3.0, 5.0, 6.5]), Kp_val)
    print(f"  D_LL at Kp={Kp_val}: L=3: {DLL_test[0]:.2e}, L=5: {DLL_test[1]:.2e}, "
          f"L=6.5: {DLL_test[2]:.2e} R_E^2/h")

# --- Loss timescale ---
# Losses are strongest in the slot region (L ~ 2-3) from whistler-mode waves
# and at very high L from magnetopause shadowing.
def loss_timescale(L_arr):
    """Loss timescale tau_loss [hours] as function of L.

    Short in slot region (L=2-3), longer in outer belt (L=4-6),
    short again near magnetopause (L>6.5).
    """
    tau = np.full_like(L_arr, 200.0)  # default: 200 hours

    # Slot region losses (plasmaspheric hiss)
    slot_mask = (L_arr >= 1.5) & (L_arr <= 3.5)
    tau[slot_mask] = 5.0 + 15.0 * ((L_arr[slot_mask] - 2.5) / 1.0)**2

    # Magnetopause shadowing at high L
    high_L_mask = L_arr > 6.0
    tau[high_L_mask] = 50.0 * np.exp(-(L_arr[high_L_mask] - 6.0) / 0.5)

    return np.maximum(tau, 1.0)  # minimum 1 hour


tau_loss = loss_timescale(L)


# =========================================================================
# 3. INITIAL AND BOUNDARY CONDITIONS
# =========================================================================
# Initial PSD: quiet-time two-belt structure
# Inner belt: small peak at L~1.8 (mostly protons, low electron PSD)
# Slot: minimum at L~2.5
# Outer belt: peak at L~5

def initial_psd(L_arr):
    """Quiet-time phase space density profile f(L).

    Normalized to arbitrary units; represents relativistic electrons
    at a fixed first adiabatic invariant.
    """
    # Inner belt (small, stable population)
    f_inner = 0.1 * np.exp(-0.5 * ((L_arr - 1.8) / 0.3)**2)

    # Slot region (very low PSD)
    f_slot = 0.01

    # Outer belt (main peak)
    f_outer = 1.0 * np.exp(-0.5 * ((L_arr - 5.0) / 1.2)**2)

    # Combine: use maximum to avoid overlap issues
    f = np.maximum(f_inner, f_slot) + f_outer
    return f


f = initial_psd(L)

# Boundary conditions:
# Inner boundary (L=1): fixed at initial value (stable inner belt)
f_inner_bc = f[0]

# Outer boundary (L=7): time-varying (storm injection)
def outer_boundary(t):
    """Time-varying outer boundary PSD.

    During storm, enhanced injection increases PSD at L=7.
    """
    f_quiet = initial_psd(np.array([L_max]))[0]
    if 24 <= t <= 48:
        # Storm enhancement: PSD increases by factor of 5
        enhancement = 1.0 + 4.0 * np.sin(np.pi * (t - 24) / 24)**2
        return f_quiet * enhancement
    return f_quiet


print(f"\nInitial conditions:")
print(f"  Inner belt peak (L=1.8): f = {initial_psd(np.array([1.8]))[0]:.3f}")
print(f"  Slot minimum (L=2.5):    f = {initial_psd(np.array([2.5]))[0]:.3f}")
print(f"  Outer belt peak (L=5.0): f = {initial_psd(np.array([5.0]))[0]:.3f}")
print(f"  Outer BC quiet:          f = {outer_boundary(0):.3f}")
print(f"  Outer BC storm peak:     f = {outer_boundary(36):.3f}")


# =========================================================================
# 4. CRANK-NICOLSON SOLVER
# =========================================================================
# The radial diffusion equation:
#   df/dt = L^2 * d/dL * (D_LL/L^2 * df/dL) - f/tau_loss
#
# Let g(L) = D_LL / L^2. Then the diffusion term is:
#   L^2 * d/dL * (g * df/dL)
#
# Discretize with Crank-Nicolson (average of explicit and implicit):
#   (f^{n+1} - f^n) / dt = 0.5 * [RHS^{n+1} + RHS^n]
#
# This gives a tridiagonal system A * f^{n+1} = B * f^n + rhs_explicit

def build_tridiagonal(L_arr, DLL, tau, dt, NL):
    """Build Crank-Nicolson tridiagonal matrices.

    Returns (A_diag, A_lower, A_upper, B_diag, B_lower, B_upper) for
    the system A * f^{n+1} = B * f^n.

    The diffusion operator L^2 * d/dL(D_LL/L^2 * df/dL) is discretized as:
    L_i^2 * [ (g_{i+1/2} * (f_{i+1}-f_i) - g_{i-1/2} * (f_i-f_{i-1})) / dL^2 ]
    where g = D_LL / L^2.
    """
    dL = L_arr[1] - L_arr[0]
    g = DLL / L_arr**2  # D_LL / L^2

    # Interface values (half-grid)
    g_half_plus = 0.5 * (g[1:] + g[:-1])   # g at i+1/2 (size NL-1)
    g_half_minus = np.zeros(NL)
    g_half_minus[1:] = g_half_plus          # g at i-1/2

    # Diffusion operator coefficients
    # For interior point i: L_i^2 * [g_{i+1/2}*(f_{i+1}-f_i) - g_{i-1/2}*(f_i-f_{i-1})] / dL^2
    alpha = np.zeros(NL)  # coefficient of f_{i-1}
    beta = np.zeros(NL)   # coefficient of f_i
    gamma = np.zeros(NL)  # coefficient of f_{i+1}

    for i in range(1, NL - 1):
        gp = 0.5 * (g[i] + g[min(i+1, NL-1)])  # g at i+1/2
        gm = 0.5 * (g[i] + g[max(i-1, 0)])      # g at i-1/2
        L2 = L_arr[i]**2

        alpha[i] = L2 * gm / dL**2
        gamma[i] = L2 * gp / dL**2
        beta[i] = -(alpha[i] + gamma[i]) - 1.0 / tau[i]  # include loss term

    # Crank-Nicolson: 0.5 weight on each side
    # (I - 0.5*dt*D) * f^{n+1} = (I + 0.5*dt*D) * f^n
    # where D is the operator matrix

    # A matrix (implicit side): I - 0.5*dt*D
    A_lower = -0.5 * dt * alpha[1:]       # sub-diagonal
    A_diag = 1.0 - 0.5 * dt * beta        # main diagonal
    A_upper = -0.5 * dt * gamma[:-1]      # super-diagonal

    # B matrix (explicit side): I + 0.5*dt*D
    B_lower = 0.5 * dt * alpha[1:]
    B_diag = 1.0 + 0.5 * dt * beta
    B_upper = 0.5 * dt * gamma[:-1]

    return (A_diag, A_lower, A_upper, B_diag, B_lower, B_upper)


def solve_tridiagonal(a, b, c, d):
    """Solve tridiagonal system using Thomas algorithm.

    a: sub-diagonal (size n-1)
    b: main diagonal (size n)
    c: super-diagonal (size n-1)
    d: right-hand side (size n)
    """
    n = len(b)
    # Forward sweep
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        m = a[i-1] / (b[i] - a[i-1] * c_prime[i-1]) if i < n else 0
        if i < n - 1:
            m_denom = b[i] - a[i-1] * c_prime[i-1]
            c_prime[i] = c[i] / m_denom
            d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / m_denom
        else:
            m_denom = b[i] - a[i-1] * c_prime[i-1]
            d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / m_denom

    # Back substitution
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x


# --- Time integration ---
# Store snapshots at key times
snapshot_times = [0, 12, 24, 30, 36, 42, 48, 60, 72, 96, 120]
snapshots = {0: f.copy()}

# Store full L-t grid for colormap
f_Lt = np.zeros((Nt, NL))
f_Lt[0] = f.copy()

print(f"\nRunning Crank-Nicolson integration ({Nt} time steps)...")

for n in range(1, Nt):
    t_current = t_arr[n]
    Kp = kp_profile(t_current)

    # Compute D_LL at current Kp
    DLL = compute_DLL(L, Kp)

    # Build tridiagonal system
    A_d, A_l, A_u, B_d, B_l, B_u = build_tridiagonal(L, DLL, tau_loss, dt, NL)

    # Compute RHS: B * f^n
    rhs = B_d * f
    rhs[1:] += B_l * f[:-1]
    rhs[:-1] += B_u * f[1:]

    # Apply boundary conditions
    # Inner BC: f[0] = f_inner_bc (fixed)
    A_d[0] = 1.0
    A_u[0] = 0.0
    rhs[0] = f_inner_bc

    # Outer BC: f[-1] = outer_boundary(t)
    A_d[-1] = 1.0
    A_l[-1] = 0.0
    rhs[-1] = outer_boundary(t_current)

    # Solve tridiagonal system
    f = solve_tridiagonal(A_l, A_d, A_u, rhs)

    # Enforce non-negativity (physical constraint)
    f = np.maximum(f, 0)

    # Store
    f_Lt[n] = f.copy()

    # Save snapshots
    t_h = round(t_current, 1)
    if t_h in snapshot_times and t_h not in snapshots:
        snapshots[t_h] = f.copy()

print("  Integration complete.")

# Print key results
print(f"\nResults at key times:")
print(f"  {'Time [h]':>10} {'Peak PSD':>10} {'Peak L':>8} {'Slot min':>10}")
for t_snap in sorted(snapshots.keys()):
    f_snap = snapshots[t_snap]
    peak_idx = np.argmax(f_snap[10:])  # skip inner belt
    peak_L = L[10 + peak_idx]
    peak_f = f_snap[10 + peak_idx]
    slot_mask = (L >= 2.0) & (L <= 3.0)
    slot_min = f_snap[slot_mask].min()
    print(f"  {t_snap:>10.0f} {peak_f:>10.3f} {peak_L:>8.2f} {slot_min:>10.4f}")


# =========================================================================
# 5. PLOTTING
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Radiation Belt Radial Diffusion Model", fontsize=15, y=0.98)

# --- Panel 1: PSD vs L at different times ---
ax = axes[0, 0]
cmap = plt.cm.coolwarm
for i, t_snap in enumerate(sorted(snapshots.keys())):
    if t_snap in [0, 24, 36, 48, 72, 120]:
        color = cmap(i / (len(snapshots) - 1))
        lw = 2.5 if t_snap in [0, 36, 120] else 1.5
        ls = '-' if t_snap <= 48 else '--'
        ax.plot(L, snapshots[t_snap], color=color, lw=lw, ls=ls,
                label=f't = {t_snap:.0f} h')

# Mark regions
ax.axvspan(1.5, 2.5, color='orange', alpha=0.1, label='Inner belt')
ax.axvspan(2.5, 3.5, color='gray', alpha=0.1, label='Slot region')
ax.axvspan(3.5, 6.5, color='lightblue', alpha=0.1, label='Outer belt')

ax.set_xlabel("L-shell", fontsize=11)
ax.set_ylabel("Phase Space Density f [arb. units]", fontsize=11)
ax.set_title("PSD Profiles at Selected Times", fontsize=12)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 7)
ax.set_ylim(0, None)

# --- Panel 2: L-t colormap ---
ax = axes[0, 1]

# Subsample for plotting
t_plot_idx = np.arange(0, Nt, max(1, Nt // 500))
L_grid, T_grid = np.meshgrid(L, t_arr[t_plot_idx])
f_plot = f_Lt[t_plot_idx]

pcm = ax.pcolormesh(T_grid, L_grid, f_plot, cmap='hot', shading='auto',
                     vmin=0, vmax=2.0)
plt.colorbar(pcm, ax=ax, label='PSD [arb.]')

# Storm phase lines
ax.axvline(24, color='cyan', ls='--', lw=1, label='Storm onset')
ax.axvline(48, color='cyan', ls=':', lw=1, label='Storm end')

ax.set_xlabel("Time [hours]", fontsize=11)
ax.set_ylabel("L-shell", fontsize=11)
ax.set_title("Phase Space Density Evolution (L vs t)", fontsize=12)
ax.legend(fontsize=8, loc='upper right')

# --- Panel 3: Kp index and D_LL at L=5 ---
ax = axes[1, 0]
ax2 = ax.twinx()

l1, = ax.plot(t_arr, Kp_arr, 'b-', lw=1.5, label='Kp index')
DLL_L5 = compute_DLL(np.array([5.0]), Kp_arr)
l2, = ax2.semilogy(t_arr, DLL_L5.flatten(), 'r-', lw=1.5, label='$D_{LL}$ at L=5')

ax.set_xlabel("Time [hours]", fontsize=11)
ax.set_ylabel("Kp index", color='b', fontsize=11)
ax2.set_ylabel("$D_{LL}$ [$R_E^2$/h]", color='r', fontsize=11)
ax.set_title("Kp Index and Diffusion Coefficient", fontsize=12)
ax.legend(handles=[l1, l2], fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.axvspan(24, 48, color='salmon', alpha=0.15, label='_nolegend_')

# --- Panel 4: Loss timescale and D_LL profile ---
ax = axes[1, 1]
ax2 = ax.twinx()

l1, = ax.semilogy(L, tau_loss, 'b-', lw=2, label=r'$\tau_{loss}$ [hours]')
DLL_quiet = compute_DLL(L, 1.0)
DLL_storm = compute_DLL(L, 6.0)
l2, = ax2.semilogy(L, DLL_quiet, 'r--', lw=1.5, label='$D_{LL}$ (Kp=1)')
l3, = ax2.semilogy(L, DLL_storm, 'r-', lw=2, label='$D_{LL}$ (Kp=6)')

ax.set_xlabel("L-shell", fontsize=11)
ax.set_ylabel(r"$\tau_{loss}$ [hours]", color='b', fontsize=11)
ax2.set_ylabel("$D_{LL}$ [$R_E^2$/h]", color='r', fontsize=11)
ax.set_title("Loss Timescale and Diffusion Coefficient", fontsize=12)
ax.legend(handles=[l1, l2, l3], fontsize=8, loc='center right')
ax.grid(True, alpha=0.3)

# Mark regions
ax.axvspan(2.0, 3.5, color='gray', alpha=0.1)
ax.text(2.75, tau_loss.max() * 0.5, 'Slot', fontsize=9, ha='center', style='italic')

plt.tight_layout()
plt.savefig("/opt/projects/01_Personal/03_Study/examples/Space_Weather/06_radiation_belts.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: 06_radiation_belts.png")
