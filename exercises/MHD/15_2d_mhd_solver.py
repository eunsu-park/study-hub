"""
Lesson 15: 2D MHD Solver
Topic: MHD
Description: Exercises on CFL conditions, divergence-free constraints,
             Constrained Transport, Orszag-Tang vortex, Kelvin-Helmholtz
             instability, PLM reconstruction, WENO schemes, and AMR.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """CFL Condition.

    For a 2D MHD simulation with dx = dy = 0.01, fast magnetosonic speed
    c_f = 2, max flow velocity |v| = 1, calculate max timestep for CFL = 0.5.
    """
    dx = 0.01
    dy = 0.01
    c_f = 2.0     # fast magnetosonic speed
    v_max = 1.0   # maximum flow velocity
    CFL = 0.5

    # Maximum signal speed: c_max = c_f + |v|
    c_max = c_f + v_max

    # 2D CFL condition: dt < CFL / (c_max/dx + c_max/dy)
    # Or equivalently: dt < CFL * (1/dx + 1/dy)^(-1) / c_max
    dt_max = CFL / (c_max / dx + c_max / dy)

    # Alternative formulation: dt < CFL * min(dx, dy) / c_max (simpler, more restrictive)
    dt_simple = CFL * min(dx, dy) / c_max

    print(f"  Grid: dx = dy = {dx}")
    print(f"  Fast magnetosonic speed c_f = {c_f}")
    print(f"  Maximum flow velocity |v| = {v_max}")
    print(f"  Maximum signal speed: c_max = c_f + |v| = {c_max}")
    print(f"  CFL number = {CFL}")
    print(f"  2D CFL condition: dt < CFL / (c_max/dx + c_max/dy)")
    print(f"  dt_max = {CFL} / ({c_max}/{dx} + {c_max}/{dy})")
    print(f"  dt_max = {CFL} / ({c_max / dx} + {c_max / dy})")
    print(f"  dt_max = {dt_max:.6f}")
    print(f"  Simpler estimate: dt < CFL * min(dx,dy) / c_max = {dt_simple:.6f}")
    print(f"  For a 256x256 grid at CFL=0.5: ~{int(1.0 / dt_max)} steps per unit time")


def exercise_2():
    """Divergence Preservation.

    Explain why standard finite volume does not preserve div(B) = 0,
    but Constrained Transport does. Sketch the Yee mesh layout.
    """
    print("  Why standard FV fails to preserve div(B) = 0:")
    print("  -------------------------------------------------")
    print("  Standard finite volume updates cell-centered B using numerical fluxes.")
    print("  Each B component is updated independently via Riemann solver fluxes.")
    print("  Discretization errors in the flux computation do not cancel exactly,")
    print("  leading to accumulation of div(B) errors over time.")
    print("  These errors can cause unphysical forces and plasma acceleration.")
    print()
    print("  Why Constrained Transport (CT) preserves div(B) = 0:")
    print("  ---------------------------------------------------")
    print("  CT uses a staggered grid (Yee mesh) where:")
    print("    - B_x is stored at x-face centers (i+1/2, j)")
    print("    - B_y is stored at y-face centers (i, j+1/2)")
    print("    - E_z is computed at cell corners (i+1/2, j+1/2)")
    print()
    print("  The induction equation dB/dt = -curl(E) becomes:")
    print("    dB_x/dt = -dE_z/dy  (at x-face)")
    print("    dB_y/dt = +dE_z/dx  (at y-face)")
    print()
    print("  The discrete divergence:")
    print("    div(B) = (B_x[i+1/2] - B_x[i-1/2])/dx + (B_y[j+1/2] - B_y[j-1/2])/dy")
    print("  is EXACTLY preserved because:")
    print("    d(div(B))/dt = d/dx(-dE_z/dy) + d/dy(dE_z/dx) = 0 (identically!)")
    print("  This is a discrete analog of div(curl) = 0.")
    print()

    # Create Yee mesh diagram
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw cell boundaries
    for i in range(4):
        ax.axhline(i, color='gray', linewidth=1)
        ax.axvline(i, color='gray', linewidth=1)

    # Cell centers (density, pressure, energy)
    for i in range(3):
        for j in range(3):
            ax.plot(i + 0.5, j + 0.5, 'ko', markersize=8)
            ax.annotate(f'$\\rho,p$', (i + 0.5, j + 0.5), fontsize=8,
                        ha='center', va='bottom', color='black')

    # B_x at x-faces (i+1/2, j)
    for i in range(4):
        for j in range(3):
            ax.plot(i, j + 0.5, 'b>', markersize=12)
            if i < 3 and j < 2:
                ax.annotate('$B_x$', (i, j + 0.5), fontsize=9,
                            ha='right', va='center', color='blue')

    # B_y at y-faces (i, j+1/2)
    for i in range(3):
        for j in range(4):
            ax.plot(i + 0.5, j, 'r^', markersize=12)
            if j < 3 and i < 2:
                ax.annotate('$B_y$', (i + 0.5, j), fontsize=9,
                            ha='center', va='top', color='red')

    # E_z at corners
    for i in range(4):
        for j in range(4):
            ax.plot(i, j, 'gs', markersize=6)

    ax.annotate('$E_z$', (3, 3), fontsize=10, ha='left', va='bottom', color='green')

    ax.set_xlim(-0.5, 3.8)
    ax.set_ylim(-0.5, 3.8)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Yee Mesh for Constrained Transport', fontsize=14)
    ax.set_aspect('equal')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Cell center (rho, p, v)'),
        Line2D([0], [0], marker='>', color='w', markerfacecolor='b', markersize=10, label='B_x (x-face)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='r', markersize=10, label='B_y (y-face)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='g', markersize=6, label='E_z (corner)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig('15_yee_mesh.png', dpi=150)
    plt.close()
    print("  Yee mesh diagram saved to 15_yee_mesh.png")


def exercise_3():
    """Orszag-Tang Vortex.

    Why is the Orszag-Tang vortex a good test problem for 2D MHD codes?
    What physical processes does it test?
    """
    print("  The Orszag-Tang Vortex as an MHD Benchmark:")
    print("  ============================================")
    print()
    print("  Initial conditions (on [0, 2*pi]^2, periodic):")
    print("    rho = gamma^2 (uniform)")
    print("    v_x = -sin(y)")
    print("    v_y = sin(x)")
    print("    B_x = -sin(y)")
    print("    B_y = sin(2x)")
    print("    p = gamma (uniform)")
    print()
    print("  Why it is a good test problem (at least 3 reasons):")
    print()
    print("  1. SHOCK INTERACTIONS: Multiple shocks form and interact,")
    print("     testing the Riemann solver's ability to handle MHD shock types")
    print("     (fast, slow, rotational). Shock-shock collisions create complex")
    print("     structures that are sensitive to numerical diffusion.")
    print()
    print("  2. MAGNETIC RECONNECTION: Current sheets form and thin over time,")
    print("     eventually reconnecting. Tests the code's ability to resolve")
    print("     thin current layers and handle magnetic topology changes.")
    print()
    print("  3. MHD TURBULENCE DEVELOPMENT: The smooth initial conditions")
    print("     evolve into a turbulent-like state with a cascade of energy")
    print("     to smaller scales. Tests resolution requirements and dissipation.")
    print()
    print("  4. div(B) PRESERVATION: The evolving field tests whether the code")
    print("     maintains the solenoidal constraint. Div(B) errors would create")
    print("     unphysical forces visible in the solution symmetry.")
    print()
    print("  5. SYMMETRY: The initial conditions have a specific symmetry")
    print("     (invariance under (x,y) -> (x+pi, y+pi)). A correct code")
    print("     must preserve this symmetry throughout the evolution.")

    # Create an illustration of the OT initial conditions
    N = 128
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    y = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    vx = -np.sin(Y)
    vy = np.sin(X)
    Bx = -np.sin(Y)
    By = np.sin(2 * X)
    Jz = 2 * np.cos(2 * X) + np.cos(Y)  # curl(B)_z

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].pcolormesh(X, Y, np.sqrt(vx**2 + vy**2), cmap='viridis', shading='auto')
    axes[0].streamplot(x, y, vx, vy, color='white', linewidth=0.5, density=1.5)
    axes[0].set_title('Velocity Field', fontsize=12)
    plt.colorbar(im0, ax=axes[0], label='|v|')

    im1 = axes[1].pcolormesh(X, Y, np.sqrt(Bx**2 + By**2), cmap='inferno', shading='auto')
    axes[1].streamplot(x, y, Bx, By, color='white', linewidth=0.5, density=1.5)
    axes[1].set_title('Magnetic Field', fontsize=12)
    plt.colorbar(im1, ax=axes[1], label='|B|')

    im2 = axes[2].pcolormesh(X, Y, Jz, cmap='RdBu_r', shading='auto')
    axes[2].set_title('Current Density $J_z$', fontsize=12)
    plt.colorbar(im2, ax=axes[2], label='$J_z$')

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    plt.suptitle('Orszag-Tang Vortex: Initial Conditions', fontsize=14)
    plt.tight_layout()
    plt.savefig('15_orszag_tang_ic.png', dpi=150)
    plt.close()
    print("  Plot saved to 15_orszag_tang_ic.png")


def exercise_4():
    """Kelvin-Helmholtz Stabilization.

    Calculate the minimum magnetic field to suppress KH instability.
    V_0 = 2 m/s, rho = 1 kg/m^3.
    """
    V_0 = 2.0           # m/s (shear velocity)
    rho = 1.0           # kg/m^3
    mu_0 = 4 * np.pi * 1e-7  # H/m

    # KH instability: stabilized when Alfven speed >= shear velocity
    # v_A = B / sqrt(mu_0 * rho) >= V_0
    # B >= V_0 * sqrt(mu_0 * rho)
    B_min = V_0 * np.sqrt(mu_0 * rho)

    print(f"  Shear velocity V_0 = {V_0} m/s")
    print(f"  Density rho = {rho} kg/m^3")
    print(f"  KH instability suppressed when v_A >= V_0")
    print(f"  Minimum B: B_min = V_0 * sqrt(mu_0 * rho)")
    print(f"  B_min = {V_0} * sqrt({mu_0:.3e} * {rho})")
    print(f"  B_min = {B_min:.6e} T")
    print(f"  B_min = {B_min * 1e6:.3f} microT")
    print()

    # Show growth rate vs B
    B_range = np.linspace(0, 2 * B_min, 100)
    v_A_range = B_range / np.sqrt(mu_0 * rho)

    # KH growth rate: gamma ~ k * sqrt(V_0^2 - v_A^2) for v_A < V_0
    # (simplified for equal density on both sides, field parallel to flow)
    gamma_norm = np.where(v_A_range < V_0,
                          np.sqrt(V_0**2 - v_A_range**2) / V_0,
                          0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(B_range / B_min, gamma_norm, 'b-', linewidth=2)
    ax.axvline(1.0, color='red', linestyle='--', label=f'B = B_min = {B_min:.2e} T')
    ax.fill_between(B_range / B_min, 0, gamma_norm, alpha=0.2, color='blue')
    ax.set_xlabel(r'$B / B_{min}$', fontsize=12)
    ax.set_ylabel(r'Normalized growth rate $\gamma / \gamma_0$', fontsize=12)
    ax.set_title('KH Growth Rate vs Magnetic Field', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('15_kh_stabilization.png', dpi=150)
    plt.close()
    print("  Plot saved to 15_kh_stabilization.png")
    print("  For B > B_min, the magnetic tension suppresses the KH instability.")
    print("  The field must be aligned with the shear flow for this stabilization.")


def exercise_5():
    """PLM Reconstruction.

    Given U_{i-1} = 1.0, U_i = 1.5, U_{i+1} = 2.5, compute the slope
    using centered difference and minmod limiter.
    """
    U_im1 = 1.0  # U_{i-1}
    U_i = 1.5    # U_i
    U_ip1 = 2.5  # U_{i+1}
    dx = 1.0     # assume uniform grid spacing

    # (a) Centered difference slope
    sigma_centered = (U_ip1 - U_im1) / (2 * dx)

    # Left and right slopes
    sigma_L = (U_i - U_im1) / dx   # backward difference
    sigma_R = (U_ip1 - U_i) / dx   # forward difference

    # (b) Minmod limiter
    def minmod(a, b):
        if a * b <= 0:
            return 0.0  # opposite signs -> zero slope
        elif abs(a) < abs(b):
            return a
        else:
            return b

    sigma_minmod = minmod(sigma_L, sigma_R)

    # Reconstructed states at i+1/2 interface
    # Left state: U_{i+1/2,L} = U_i + (dx/2) * sigma
    # Right state: U_{i+1/2,R} = U_{i+1} - (dx/2) * sigma_{i+1}

    U_L_centered = U_i + 0.5 * dx * sigma_centered
    U_L_minmod = U_i + 0.5 * dx * sigma_minmod
    U_R = U_ip1 - 0.5 * dx * sigma_R  # simplified (needs U_{i+2})

    print(f"  Cell values: U_{{i-1}} = {U_im1}, U_i = {U_i}, U_{{i+1}} = {U_ip1}")
    print()
    print(f"  Left slope:  sigma_L = (U_i - U_{{i-1}})/dx = {sigma_L:.2f}")
    print(f"  Right slope: sigma_R = (U_{{i+1}} - U_i)/dx = {sigma_R:.2f}")
    print()
    print(f"  (a) Centered difference slope:")
    print(f"    sigma_centered = (U_{{i+1}} - U_{{i-1}})/(2*dx) = {sigma_centered:.2f}")
    print(f"    U_{{i+1/2,L}} = U_i + (dx/2)*sigma = {U_i} + 0.5*{sigma_centered:.2f} = {U_L_centered:.2f}")
    print()
    print(f"  (b) Minmod limiter:")
    print(f"    minmod({sigma_L:.2f}, {sigma_R:.2f}) = {sigma_minmod:.2f}")
    print(f"    U_{{i+1/2,L}} = U_i + (dx/2)*sigma = {U_i} + 0.5*{sigma_minmod:.2f} = {U_L_minmod:.2f}")
    print()
    print(f"  Comparison:")
    print(f"    Centered: U_L = {U_L_centered:.2f} (may introduce oscillations)")
    print(f"    Minmod:   U_L = {U_L_minmod:.2f} (TVD, no new extrema)")
    print(f"    Minmod picks the smaller slope, ensuring monotonicity.")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x_cells = np.array([-1, 0, 1])

    # Centered
    ax1.bar(x_cells, [U_im1, U_i, U_ip1], width=0.9, alpha=0.3, color='blue', label='Cell avg')
    ax1.plot([x_cells[1] - 0.5, x_cells[1] + 0.5],
             [U_i - 0.5 * sigma_centered, U_i + 0.5 * sigma_centered],
             'r-', linewidth=2, label=f'Centered (slope={sigma_centered:.2f})')
    ax1.set_title('Centered Difference', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Cell index')
    ax1.set_ylabel('U')

    # Minmod
    ax2.bar(x_cells, [U_im1, U_i, U_ip1], width=0.9, alpha=0.3, color='blue', label='Cell avg')
    ax2.plot([x_cells[1] - 0.5, x_cells[1] + 0.5],
             [U_i - 0.5 * sigma_minmod, U_i + 0.5 * sigma_minmod],
             'g-', linewidth=2, label=f'Minmod (slope={sigma_minmod:.2f})')
    ax2.set_title('Minmod Limiter', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Cell index')
    ax2.set_ylabel('U')

    plt.suptitle('PLM Reconstruction', fontsize=14)
    plt.tight_layout()
    plt.savefig('15_plm_reconstruction.png', dpi=150)
    plt.close()
    print("  Plot saved to 15_plm_reconstruction.png")


def exercise_6():
    """CT Electric Field.

    Write the formula for E_z at a cell corner using simple averaging
    of velocities and magnetic fields at adjacent face centers.
    """
    print("  Constrained Transport: E_z at cell corner (i+1/2, j+1/2)")
    print("  =========================================================")
    print()
    print("  The electric field E = -v x B gives E_z = -(v_x*B_y - v_y*B_x)")
    print()
    print("  At cell corner (i+1/2, j+1/2), using simple averaging:")
    print()
    print("  E_z(i+1/2, j+1/2) = -1/4 * [")
    print("    (v_x * B_y)(i+1/2, j)   +  (v_x * B_y)(i+1/2, j+1)")
    print("    - (v_y * B_x)(i, j+1/2) -  (v_y * B_x)(i+1, j+1/2)")
    print("  ]")
    print()
    print("  More explicitly:")
    print("  E_z = -1/4 * [")
    print("    v_x(i+1/2,j) * B_y(i+1/2,j)")
    print("    + v_x(i+1/2,j+1) * B_y(i+1/2,j+1)")
    print("    - v_y(i,j+1/2) * B_x(i,j+1/2)")
    print("    - v_y(i+1,j+1/2) * B_x(i+1,j+1/2)")
    print("  ]")
    print()
    print("  Where:")
    print("  - v_x at x-faces: v_x(i+1/2,j) = average of cell-centered v_x")
    print("  - B_y at x-faces: already stored there in the Yee mesh")
    print("  - v_y at y-faces: v_y(i,j+1/2) = average of cell-centered v_y")
    print("  - B_x at y-faces: already stored there in the Yee mesh")
    print()
    print("  The CT update equations:")
    print("  B_x(i,j+1/2)^{n+1} = B_x^n - dt/dy * [E_z(i+1/2,j+1/2) - E_z(i-1/2,j+1/2)]")
    print("  B_y(i+1/2,j)^{n+1} = B_y^n + dt/dx * [E_z(i+1/2,j+1/2) - E_z(i+1/2,j-1/2)]")
    print()
    print("  This ensures d(div(B))/dt = 0 exactly at the discrete level.")


def exercise_7():
    """Dimensional Splitting Error.

    Compare Strang splitting (L_x^{1/2} L_y L_x^{1/2}) accuracy
    to simple splitting (L_x L_y).
    """
    print("  Dimensional Splitting Accuracy:")
    print("  ================================")
    print()
    print("  Simple (Lie) splitting: L_x * L_y")
    print("    - Apply full x-sweep, then full y-sweep")
    print("    - Splitting error: O(dt * [L_x, L_y]) where [,] is commutator")
    print("    - This is FIRST-ORDER accurate in time")
    print("    - Error term: dt * (L_x * L_y - L_y * L_x) / 2")
    print()
    print("  Strang splitting: L_x^{1/2} * L_y * L_x^{1/2}")
    print("    - Half x-sweep, full y-sweep, half x-sweep")
    print("    - Splitting error: O(dt^2 * [[L_x, L_y], L_x + L_y])")
    print("    - This is SECOND-ORDER accurate in time")
    print("    - The symmetric ordering cancels the leading error term")
    print()
    print("  Why Strang splitting is preferred:")
    print("    1. Same cost per step (~1.5x simple, but higher accuracy)")
    print("    2. Matches the temporal order of the base scheme (2nd order)")
    print("    3. Simple splitting reduces overall scheme to 1st order")
    print("    4. Can alternate: L_x^{1/2} L_y L_x^{1/2} and L_y^{1/2} L_x L_y^{1/2}")
    print()

    # Demonstrate splitting error with a simple test
    # Advection in 2D: du/dt + a*du/dx + b*du/dy = 0
    N = 64
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, x)

    # Initial condition: Gaussian pulse
    u_exact = np.exp(-((X - np.pi)**2 + (Y - np.pi)**2))
    a, b = 1.0, 1.0  # advection speeds

    dt = 0.5 * dx / max(abs(a), abs(b))
    N_steps = 100

    # Upwind advection operator in x
    def Lx(u, dt_local):
        u_new = u.copy()
        if a > 0:
            u_new -= a * dt_local / dx * (u - np.roll(u, 1, axis=1))
        else:
            u_new -= a * dt_local / dx * (np.roll(u, -1, axis=1) - u)
        return u_new

    # Upwind advection operator in y
    def Ly(u, dt_local):
        u_new = u.copy()
        if b > 0:
            u_new -= b * dt_local / dx * (u - np.roll(u, 1, axis=0))
        else:
            u_new -= b * dt_local / dx * (np.roll(u, -1, axis=0) - u)
        return u_new

    # Simple splitting
    u_simple = u_exact.copy()
    for _ in range(N_steps):
        u_simple = Ly(Lx(u_simple, dt), dt)

    # Strang splitting
    u_strang = u_exact.copy()
    for _ in range(N_steps):
        u_strang = Lx(Ly(Lx(u_strang, 0.5 * dt), dt), 0.5 * dt)

    # Exact solution (shifted)
    t_final = N_steps * dt
    u_ref = np.exp(-(((X - np.pi - a * t_final) % (2 * np.pi) - np.pi)**2 +
                     ((Y - np.pi - b * t_final) % (2 * np.pi) - np.pi)**2))

    err_simple = np.sqrt(np.mean((u_simple - u_ref)**2))
    err_strang = np.sqrt(np.mean((u_strang - u_ref)**2))

    print(f"  Numerical test (2D advection, N={N}, {N_steps} steps):")
    print(f"    Simple splitting error (L2): {err_simple:.6f}")
    print(f"    Strang splitting error (L2): {err_strang:.6f}")
    print(f"    Ratio: {err_simple / err_strang:.2f}x")


def exercise_8():
    """WENO Advantage.

    Explain why WENO (5th order) is beneficial compared to PLM (2nd order).
    """
    print("  WENO vs PLM Comparison:")
    print("  ========================")
    print()
    print("  PLM (Piecewise Linear Method):")
    print("    - 2nd order accurate in smooth regions")
    print("    - Uses slope limiters (minmod, MC, van Leer) for TVD property")
    print("    - Reduces to 1st order at discontinuities (by design)")
    print("    - Simple to implement, robust")
    print("    - Relatively diffusive (smears contact discontinuities)")
    print()
    print("  WENO (Weighted Essentially Non-Oscillatory):")
    print("    - 5th order accurate in smooth regions (WENO5)")
    print("    - Uses nonlinear weight functions to select smooth stencils")
    print("    - Automatically reduces order near discontinuities")
    print("    - Much less numerical diffusion in smooth regions")
    print("    - Better resolution of contact discontinuities and vortices")
    print()
    print("  Why WENO is beneficial:")
    print("    1. For same grid resolution, WENO captures more small-scale structure")
    print("    2. Current sheets and contact discontinuities are sharper")
    print("    3. Turbulent cascades are better resolved (less numerical dissipation)")
    print("    4. Can use coarser grids for same accuracy -> faster overall")
    print()
    print("  Why not always use WENO:")
    print("    1. ~3x more expensive per cell per timestep than PLM")
    print("    2. More complex to implement, especially in 3D")
    print("    3. Wider stencil (5 vs 3 cells) -> harder for parallel boundaries")
    print("    4. For problems dominated by strong shocks, PLM may be sufficient")
    print()
    print("  Rule of thumb: Use WENO for turbulence, dynamos, and problems with")
    print("  both smooth and discontinuous features. Use PLM for robust shock-")
    print("  dominated flows where simplicity and speed matter more.")


def exercise_9():
    """AMR Refinement Criterion.

    Propose a refinement criterion based on |curl(B)| (current density)
    for resolving current sheets.
    """
    print("  AMR Refinement Criterion for Current Sheets:")
    print("  =============================================")
    print()
    print("  Current density: J = curl(B) / mu_0")
    print("  In 2D: J_z = (dB_y/dx - dB_x/dy) / mu_0")
    print()
    print("  Refinement criterion:")
    print("    Refine cell (i,j) if |J_z(i,j)| * dx > threshold")
    print()
    print("  Mathematical formulation:")
    print("    |J_z| * h > C_ref * B_0 / L_0")
    print()
    print("  where:")
    print("    h = local cell size")
    print("    C_ref = user-chosen threshold (~0.1-0.5)")
    print("    B_0 = reference magnetic field strength")
    print("    L_0 = reference length scale")
    print()
    print("  Alternative (normalized):")
    print("    |J_z(i,j)| / max(|J_z|) > epsilon  (epsilon ~ 0.1-0.3)")
    print()
    print("  Or using the gradient-based approach:")
    print("    |grad(B)|^2 / |B|^2 * h^2 > C_ref^2")
    print()

    # Demonstrate with a current sheet example
    N = 128
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]

    # Harris current sheet: B_x = B_0 * tanh(y / delta)
    delta = 0.05  # current sheet thickness
    B_0 = 1.0
    Bx = B_0 * np.tanh(Y / delta)

    # Current density: J_z = dB_x/dy = B_0 / (delta * cosh^2(y/delta))
    Jz = B_0 / (delta * np.cosh(Y / delta)**2)

    # Refinement flag
    threshold = 0.2 * np.max(np.abs(Jz))
    refine_flag = np.abs(Jz) > threshold

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].pcolormesh(X, Y, Bx, cmap='RdBu_r', shading='auto')
    axes[0].set_title(r'$B_x$', fontsize=12)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Y, Jz, cmap='hot', shading='auto')
    axes[1].set_title(r'$|J_z| = |\nabla \times B|$', fontsize=12)
    plt.colorbar(im1, ax=axes[1])

    axes[2].pcolormesh(X, Y, refine_flag.astype(float), cmap='Greens', shading='auto')
    axes[2].set_title('Refinement Region', fontsize=12)

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    plt.suptitle(f'AMR Refinement: Current Sheet (delta = {delta})', fontsize=14)
    plt.tight_layout()
    plt.savefig('15_amr_criterion.png', dpi=150)
    plt.close()
    print("  Plot saved to 15_amr_criterion.png")
    print(f"  Refinement covers {np.sum(refine_flag) / refine_flag.size * 100:.1f}% of domain")
    print(f"  (Only the current sheet region needs high resolution)")


def exercise_10():
    """Computational Cost.

    Compare cost of 2D MHD (256x256) vs 1D MHD (256 cells).
    """
    N_1D = 256
    N_2D_x = 256
    N_2D_y = 256

    # Operations per cell per timestep (MHD):
    # - Reconstruction: ~20 ops/variable * 8 variables = 160
    # - Riemann solver: ~100 ops per interface
    # - CT update: ~20 ops
    # - Total per cell: ~300 ops (rough estimate)
    ops_per_cell = 300  # approximate

    # 1D: N cells, 2 interfaces per cell (left, right)
    cost_1D = N_1D * ops_per_cell
    # Timestep scales with dx: dt_1D ~ CFL * dx / c_max

    # 2D: N_x * N_y cells, 4 interfaces per cell
    # But also need CT corner computation
    cost_2D = N_2D_x * N_2D_y * ops_per_cell * 2  # factor 2 for additional dimension
    # Timestep: dt_2D ~ CFL / (c_max/dx + c_max/dy) ~ dt_1D (if same dx)

    ratio = cost_2D / cost_1D

    print(f"  Computational Cost Comparison:")
    print(f"  ==============================")
    print(f"  1D MHD: {N_1D} cells")
    print(f"    Operations per timestep: ~{cost_1D:,}")
    print()
    print(f"  2D MHD: {N_2D_x} x {N_2D_y} grid = {N_2D_x * N_2D_y:,} cells")
    print(f"    Operations per timestep: ~{cost_2D:,}")
    print()
    print(f"  Cost ratio (2D / 1D): {ratio:.0f}x")
    print(f"  This is approximately N_y * 2 = {N_2D_y * 2}")
    print(f"  (factor N_y from extra dimension, factor ~2 from extra flux computation)")
    print()
    print(f"  Memory comparison:")
    print(f"    1D: 8 variables * {N_1D} cells = {8 * N_1D:,} doubles = {8 * N_1D * 8 / 1024:.1f} KB")
    mem_2D = 8 * N_2D_x * N_2D_y
    print(f"    2D: 8 variables * {N_2D_x * N_2D_y:,} cells = {mem_2D:,} doubles = {mem_2D * 8 / 1024 / 1024:.1f} MB")
    print()
    print(f"  Scaling to 3D ({N_1D}^3):")
    N_3D = N_1D**3
    cost_3D = N_3D * ops_per_cell * 3
    print(f"    Cells: {N_3D:,}")
    print(f"    Operations: ~{cost_3D:,}")
    print(f"    3D/1D ratio: {cost_3D / cost_1D:,.0f}x")
    print(f"    This is why 3D MHD simulations require supercomputers!")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: CFL Condition", exercise_1),
        ("Exercise 2: Divergence Preservation", exercise_2),
        ("Exercise 3: Orszag-Tang Vortex", exercise_3),
        ("Exercise 4: KH Stabilization", exercise_4),
        ("Exercise 5: PLM Reconstruction", exercise_5),
        ("Exercise 6: CT Electric Field", exercise_6),
        ("Exercise 7: Dimensional Splitting Error", exercise_7),
        ("Exercise 8: WENO Advantage", exercise_8),
        ("Exercise 9: AMR Refinement Criterion", exercise_9),
        ("Exercise 10: Computational Cost", exercise_10),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
        func()
