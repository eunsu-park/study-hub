"""
Geomagnetically Induced Currents (GIC) in Power Grid Networks.

Demonstrates:
- Lehtinen-Pirjola method for GIC calculation in a resistive network
- Simple 4-node power grid with transmission lines and grounding resistances
- Geoelectric field integration along transmission lines
- Network admittance matrix construction and system solution
- Time-varying GIC from rotating geoelectric field (substorm simulation)
- Dependence of GIC on E-field magnitude and direction

Physics:
    During geomagnetic storms, rapid changes in the geomagnetic field (dB/dt)
    induce a geoelectric field E at the Earth's surface via Faraday's law.
    This E-field drives quasi-DC currents through the grounded power grid:

    1. The voltage induced along a transmission line from node i to j is:
       V_ij = integral of E . dl ~ E_x * Dx_ij + E_y * Dy_ij

    2. The Lehtinen-Pirjola (1985) method solves for nodal GIC:
       (Y_n + Y_e) * J = Y_e * V_source

       where:
       - Y_n = network admittance matrix (from line resistances)
       - Y_e = earthing admittance matrix (diagonal, from grounding resistances)
       - V_source = voltage sources at each node from the geoelectric field
       - J = nodal GIC vector [Amps]

    3. The geoelectric field depends on ground conductivity and dB/dt:
       E ~ -dB/dt / (mu_0 * sigma)  (simplified for uniform half-space)

    During substorms, the E-field direction rotates as the electrojet moves
    overhead, causing GIC magnitude and direction to vary at each node.
    Transformer damage occurs when GIC exceeds ~10-100 A (half-cycle saturation).

References:
    - Lehtinen, M. & Pirjola, R. (1985). "Currents produced in earthed
      conductor networks by geomagnetically-induced electric fields."
    - Pulkkinen, A. et al. (2017). "Geomagnetically induced currents:
      Science, engineering, and applications readiness."
    - Boteler, D.H. (2019). "A 21st century view of the March 1989
      magnetic storm." Space Weather, 17, 1427-1441.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 65)
print("GEOMAGNETICALLY INDUCED CURRENTS IN POWER GRID")
print("=" * 65)


# =========================================================================
# 1. DEFINE POWER GRID NETWORK (4-NODE EXAMPLE)
# =========================================================================
# Node positions [km] (geographic coordinates, simplified to Cartesian)
# Represents a small regional grid (e.g., ~200 km extent)
nodes = {
    0: {'name': 'Substation A', 'x': 0.0,   'y': 0.0,   'R_ground': 0.5},
    1: {'name': 'Substation B', 'x': 150.0, 'y': 50.0,  'R_ground': 0.3},
    2: {'name': 'Substation C', 'x': 100.0, 'y': 180.0, 'R_ground': 0.8},
    3: {'name': 'Substation D', 'x': -30.0, 'y': 130.0, 'R_ground': 1.0},
}

# Transmission lines: (from_node, to_node, resistance_per_phase [Ohm])
# Typical HV transmission line: 0.01-0.1 Ohm/km, total 1-10 Ohm
lines = [
    (0, 1, 5.0),   # A-B: ~160 km
    (1, 2, 4.0),   # B-C: ~140 km
    (2, 3, 3.5),   # C-D: ~140 km
    (3, 0, 4.5),   # D-A: ~135 km (ring network)
    (0, 2, 6.0),   # A-C: diagonal (~200 km)
]

N_nodes = len(nodes)
N_lines = len(lines)

print("\n--- Network Topology ---")
print(f"  Nodes: {N_nodes}")
print(f"  Lines: {N_lines}")
for i, node in nodes.items():
    print(f"  Node {i} ({node['name']}): "
          f"({node['x']:.0f}, {node['y']:.0f}) km, "
          f"R_ground = {node['R_ground']:.1f} Ohm")
print()
for (i, j, R) in lines:
    dx = nodes[j]['x'] - nodes[i]['x']
    dy = nodes[j]['y'] - nodes[i]['y']
    length = np.sqrt(dx**2 + dy**2)
    print(f"  Line {i}-{j} ({nodes[i]['name'][:5]}-{nodes[j]['name'][:5]}): "
          f"R = {R:.1f} Ohm, L = {length:.0f} km")


# =========================================================================
# 2. BUILD ADMITTANCE MATRICES
# =========================================================================
def build_network_admittance(nodes, lines):
    """
    Build network admittance matrix Y_n from line resistances.

    Y_n[i,i] = sum of 1/R for all lines connected to node i
    Y_n[i,j] = -1/R_ij  (if line exists between i and j)
    """
    N = len(nodes)
    Y_n = np.zeros((N, N))

    for (i, j, R) in lines:
        conductance = 1.0 / R
        Y_n[i, i] += conductance
        Y_n[j, j] += conductance
        Y_n[i, j] -= conductance
        Y_n[j, i] -= conductance

    return Y_n


def build_earthing_admittance(nodes):
    """
    Build earthing (grounding) admittance matrix Y_e.

    Diagonal matrix: Y_e[i,i] = 1 / R_ground_i
    """
    N = len(nodes)
    Y_e = np.zeros((N, N))
    for i, node in nodes.items():
        Y_e[i, i] = 1.0 / node['R_ground']
    return Y_e


Y_n = build_network_admittance(nodes, lines)
Y_e = build_earthing_admittance(nodes)

print("\n--- Network Admittance Matrix Y_n ---")
for i in range(N_nodes):
    row = "  [" + "  ".join(f"{Y_n[i, j]:7.3f}" for j in range(N_nodes)) + " ]"
    print(row)

print("\n--- Earthing Admittance Matrix Y_e (diagonal) ---")
for i in range(N_nodes):
    print(f"  Node {i}: Y_e = {Y_e[i, i]:.3f} S "
          f"(R_ground = {nodes[i]['R_ground']:.1f} Ohm)")


# =========================================================================
# 3. COMPUTE GIC FOR GIVEN GEOELECTRIC FIELD
# =========================================================================
def compute_gic(nodes, lines, E_x, E_y, Y_n, Y_e):
    """
    Solve for GIC at each node using Lehtinen-Pirjola method.

    Steps:
    1. Compute voltage along each line: V_ij = E_x*Dx + E_y*Dy
    2. Build voltage source vector for each node
    3. Solve: (Y_n + Y_e) * J = Y_e * V_source

    Parameters:
        nodes : dict of node positions and grounding
        lines : list of (from, to, resistance) tuples
        E_x   : eastward geoelectric field [V/km]
        E_y   : northward geoelectric field [V/km]
        Y_n   : network admittance matrix
        Y_e   : earthing admittance matrix

    Returns:
        J_gic : GIC at each node [A] (positive = into ground)
    """
    N = len(nodes)

    # Step 1: Line voltages
    # V_source at each node = sum of (V_ij / R_ij) for lines connected to node i
    # where V_ij = E . dl = E_x * (x_j - x_i) + E_y * (y_j - y_i)
    V_src = np.zeros(N)

    for (i, j, R) in lines:
        dx = nodes[j]['x'] - nodes[i]['x']  # [km]
        dy = nodes[j]['y'] - nodes[i]['y']  # [km]
        V_line = E_x * dx + E_y * dy  # [V] (since E is V/km)

        # Contribute to nodal voltage sources
        V_src[i] += V_line / R
        V_src[j] -= V_line / R

    # Step 2: Solve system
    # (Y_n + Y_e) * J = Y_e * V_nodal
    # But V_nodal is actually derived from V_src through the circuit.
    # The standard LP formulation:
    #   J = (Y_n + Y_e)^{-1} * (Y_n * (Y_n)^{-1} * V_src)
    # Simplified: solve (Y_n + Y_e) * V_nodal = V_src for nodal potentials,
    # then J_i = V_nodal_i / R_ground_i

    # Actually, the correct LP formulation: solve for nodal voltages
    # (Y_n + Y_e) * V = -V_src  (V_src acts as current injection)
    # Wait -- let's use the standard formulation properly:
    # The induced EMFs create current sources. The nodal equation is:
    # (Y_n + Y_e) * V_node = I_src
    # where I_src[i] = sum over lines connected to i of V_line_ij / R_ij
    # Then GIC_i = V_node_i * Y_e[i,i] = V_node_i / R_ground_i

    A_matrix = Y_n + Y_e
    V_node = np.linalg.solve(A_matrix, V_src)

    # GIC at each node (current flowing to ground)
    J_gic = np.array([V_node[i] * Y_e[i, i] for i in range(N)])

    return J_gic, V_node


# === Test with uniform eastward E-field ===
E_x_test = 1.0   # 1 V/km eastward (moderate storm)
E_y_test = 0.0
J_test, V_test = compute_gic(nodes, lines, E_x_test, E_y_test, Y_n, Y_e)

print(f"\n--- GIC for E = ({E_x_test}, {E_y_test}) V/km ---")
print(f"{'Node':<20} {'V_node [V]':<15} {'GIC [A]':<12}")
for i in range(N_nodes):
    print(f"  {nodes[i]['name']:<18} {V_test[i]:<15.2f} {J_test[i]:<12.2f}")
print(f"  Total (should be ~0): {J_test.sum():.6f} A")


# =========================================================================
# 4. GIC VS E-FIELD DIRECTION
# =========================================================================
theta_range = np.linspace(0, 360, 361)  # E-field direction [degrees]
E_mag = 2.0  # V/km (strong storm)

gic_vs_theta = np.zeros((N_nodes, len(theta_range)))
for k, theta_deg in enumerate(theta_range):
    theta_rad = np.radians(theta_deg)
    Ex = E_mag * np.cos(theta_rad)
    Ey = E_mag * np.sin(theta_rad)
    J, _ = compute_gic(nodes, lines, Ex, Ey, Y_n, Y_e)
    gic_vs_theta[:, k] = J

print(f"\n--- Peak GIC at Each Node (|E| = {E_mag} V/km, all directions) ---")
for i in range(N_nodes):
    max_gic = np.max(np.abs(gic_vs_theta[i, :]))
    best_theta = theta_range[np.argmax(np.abs(gic_vs_theta[i, :]))]
    print(f"  {nodes[i]['name']:<18}: max |GIC| = {max_gic:.1f} A "
          f"at theta = {best_theta:.0f} deg")


# =========================================================================
# 5. SUBSTORM SIMULATION (TIME-VARYING E-FIELD)
# =========================================================================
# During a substorm, the electrojet intensifies and the E-field
# direction rotates as the current system evolves (~30-60 min)
dt = 0.5  # time step [minutes]
t_minutes = np.arange(0, 120, dt)
N_t = len(t_minutes)

def substorm_efield(t_min):
    """
    Synthetic substorm geoelectric field.

    Models a substorm with:
    - Growth phase (0-30 min): E slowly increases, mostly northward
    - Onset (30 min): rapid intensification
    - Expansion (30-60 min): E rotates and peaks
    - Recovery (60-120 min): E decreases

    Returns:
        E_x, E_y : geoelectric field components [V/km]
    """
    # Amplitude envelope
    A = np.zeros_like(t_min)
    # Growth phase
    mask1 = t_min < 30
    A[mask1] = 0.5 * (t_min[mask1] / 30.0)
    # Expansion
    mask2 = (t_min >= 30) & (t_min < 45)
    A[mask2] = 0.5 + 4.5 * ((t_min[mask2] - 30) / 15.0)
    # Peak
    mask3 = (t_min >= 45) & (t_min < 55)
    A[mask3] = 5.0
    # Recovery
    mask4 = t_min >= 55
    A[mask4] = 5.0 * np.exp(-(t_min[mask4] - 55) / 30.0)

    # Direction: rotates from north (90 deg) through east during expansion
    direction = np.full_like(t_min, 90.0)  # degrees, 0=east, 90=north
    rot_mask = t_min >= 30
    direction[rot_mask] = 90 - 120 * np.clip((t_min[rot_mask] - 30) / 60, 0, 1)

    theta_rad = np.radians(direction)
    E_x = A * np.cos(theta_rad)
    E_y = A * np.sin(theta_rad)
    return E_x, E_y


E_x_sub, E_y_sub = substorm_efield(t_minutes)
E_mag_sub = np.sqrt(E_x_sub**2 + E_y_sub**2)

# Compute GIC time series
gic_time = np.zeros((N_nodes, N_t))
for k in range(N_t):
    J, _ = compute_gic(nodes, lines, E_x_sub[k], E_y_sub[k], Y_n, Y_e)
    gic_time[:, k] = J

print("\n--- Substorm Simulation Results ---")
print(f"  Peak |E| = {E_mag_sub.max():.1f} V/km "
      f"at t = {t_minutes[np.argmax(E_mag_sub)]:.0f} min")
for i in range(N_nodes):
    max_gic = np.max(np.abs(gic_time[i, :]))
    t_max = t_minutes[np.argmax(np.abs(gic_time[i, :]))]
    print(f"  {nodes[i]['name']:<18}: peak |GIC| = {max_gic:.1f} A "
          f"at t = {t_max:.0f} min")

# Transformer damage threshold
print("\n  Transformer half-cycle saturation thresholds:")
print("    Warning:  |GIC| > 10 A (VAR increase)")
print("    Concern:  |GIC| > 30 A (possible hotspot heating)")
print("    Critical: |GIC| > 75 A (risk of damage)")


# =========================================================================
# 6. PLOTTING
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
node_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# --- Panel 1: Network topology ---
ax = axes[0, 0]
# Draw transmission lines
for (i, j, R) in lines:
    x = [nodes[i]['x'], nodes[j]['x']]
    y = [nodes[i]['y'], nodes[j]['y']]
    ax.plot(x, y, 'k-', linewidth=2, alpha=0.6)
    mx, my = np.mean(x), np.mean(y)
    ax.text(mx, my + 5, f'{R:.1f} $\\Omega$', fontsize=8, ha='center',
            color='gray')

# Draw nodes
for i, node in nodes.items():
    ax.plot(node['x'], node['y'], 'o', color=node_colors[i], markersize=15,
            zorder=5, markeredgecolor='black', markeredgewidth=1.5)
    ax.text(node['x'], node['y'] - 15, f"{node['name']}\n$R_g$={node['R_ground']}$\\Omega$",
            fontsize=8, ha='center', va='top', fontweight='bold')

# Draw E-field arrow
ax.annotate('', xy=(50, -30), xytext=(-10, -30),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='red'))
ax.text(20, -45, 'E-field', fontsize=10, color='red', ha='center',
        fontweight='bold')

ax.set_xlabel('Eastward [km]')
ax.set_ylabel('Northward [km]')
ax.set_title('Power Grid Network Topology')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlim(-80, 200)
ax.set_ylim(-60, 230)

# --- Panel 2: GIC vs E-field direction ---
ax = axes[0, 1]
for i in range(N_nodes):
    ax.plot(theta_range, gic_vs_theta[i, :], color=node_colors[i],
            linewidth=2, label=nodes[i]['name'])

ax.axhline(0, color='black', linewidth=0.5)
ax.axhline(10, color='orange', linestyle='--', alpha=0.5, label='Warning (10 A)')
ax.axhline(-10, color='orange', linestyle='--', alpha=0.5)
ax.set_xlabel('E-field Direction [degrees from East]')
ax.set_ylabel('GIC [A]')
ax.set_title(f'GIC vs E-field Direction (|E| = {E_mag} V/km)')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 360)

# --- Panel 3: E-field during substorm ---
ax = axes[0, 2]
ax.plot(t_minutes, E_x_sub, 'b-', linewidth=2, label='$E_x$ (eastward)')
ax.plot(t_minutes, E_y_sub, 'r-', linewidth=2, label='$E_y$ (northward)')
ax.plot(t_minutes, E_mag_sub, 'k--', linewidth=2, label='$|E|$', alpha=0.7)

# Mark substorm phases
ax.axvspan(0, 30, alpha=0.1, color='green', label='Growth')
ax.axvspan(30, 55, alpha=0.1, color='red', label='Expansion')
ax.axvspan(55, 120, alpha=0.1, color='blue', label='Recovery')

ax.set_xlabel('Time [minutes]')
ax.set_ylabel('Geoelectric Field [V/km]')
ax.set_title('Substorm Geoelectric Field')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# --- Panel 4: GIC time series during substorm ---
ax = axes[1, 0]
for i in range(N_nodes):
    ax.plot(t_minutes, gic_time[i, :], color=node_colors[i],
            linewidth=2, label=nodes[i]['name'])

ax.axhline(0, color='black', linewidth=0.5)
ax.axhspan(10, 100, alpha=0.05, color='orange')
ax.axhspan(-100, -10, alpha=0.05, color='orange')
ax.axhline(10, color='orange', linestyle='--', alpha=0.5)
ax.axhline(-10, color='orange', linestyle='--', alpha=0.5)

ax.set_xlabel('Time [minutes]')
ax.set_ylabel('GIC [A]')
ax.set_title('GIC at Each Node During Substorm')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 5: Polar plot of GIC vs E-field direction ---
ax = axes[1, 1]
ax = fig.add_subplot(2, 3, 5, polar=True)
for i in range(N_nodes):
    theta_rad = np.radians(theta_range)
    ax.plot(theta_rad, np.abs(gic_vs_theta[i, :]), color=node_colors[i],
            linewidth=2, label=nodes[i]['name'])

ax.set_title('|GIC| vs E-field Direction\n(polar)', pad=20)
ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.0))

# --- Panel 6: GIC sensitivity to E-field magnitude ---
ax = axes[1, 2]
E_scan = np.linspace(0, 10, 50)  # V/km
# Fix direction at worst-case for each node
for i in range(N_nodes):
    worst_theta = theta_range[np.argmax(np.abs(gic_vs_theta[i, :]))]
    worst_rad = np.radians(worst_theta)
    gic_vs_E = []
    for E_val in E_scan:
        Ex = E_val * np.cos(worst_rad)
        Ey = E_val * np.sin(worst_rad)
        J, _ = compute_gic(nodes, lines, Ex, Ey, Y_n, Y_e)
        gic_vs_E.append(np.abs(J[i]))
    ax.plot(E_scan, gic_vs_E, color=node_colors[i], linewidth=2,
            label=f"{nodes[i]['name']} ({worst_theta:.0f}°)")

# Damage thresholds
ax.axhline(10, color='orange', linestyle='--', label='Warning (10 A)')
ax.axhline(30, color='red', linestyle='--', label='Concern (30 A)')
ax.axhline(75, color='darkred', linestyle='--', label='Critical (75 A)')

# Reference E-field levels
ax.axvline(1, color='gray', linestyle=':', alpha=0.5)
ax.text(1.1, 5, 'Moderate\nstorm', fontsize=7, color='gray')
ax.axvline(5, color='gray', linestyle=':', alpha=0.5)
ax.text(5.1, 5, 'Major\nstorm', fontsize=7, color='gray')

ax.set_xlabel('|E| Geoelectric Field [V/km]')
ax.set_ylabel('Peak |GIC| [A]')
ax.set_title('GIC Sensitivity to E-field Magnitude')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/opt/projects/01_Personal/03_Study/examples/Space_Weather/10_gic_network.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nKey insights:")
print("  - GIC depends on both E-field magnitude AND direction relative to grid geometry")
print("  - Different nodes are vulnerable to different E-field directions")
print("  - Low grounding resistance increases GIC at that node (more current to ground)")
print("  - During substorms, E-field rotation causes GIC to vary rapidly (~minutes)")
print("  - March 1989 Quebec blackout: E ~ 5 V/km, GIC > 100 A at some transformers")
print("  - GIC scales linearly with E-field magnitude (linear system)")
print("\nPlot saved to 10_gic_network.png")
