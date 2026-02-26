"""
Solar Wind-Magnetosphere Coupling Functions.

Computes and compares the major empirical coupling functions that quantify
energy transfer from the solar wind into the magnetosphere. These functions
predict geomagnetic activity from upstream solar wind parameters.

Key physics:
  - Akasofu epsilon parameter:
    epsilon = (4*pi/mu_0) * V * B^2 * sin^4(theta_c/2) * l_0^2
    where l_0 ~ 7 R_E is an empirical length scale.
  - Newell universal coupling function:
    dPhi_MP/dt = V^(4/3) * B_T^(2/3) * sin^(8/3)(theta_c/2)
    where B_T = sqrt(By^2 + Bz^2) is the transverse IMF.
  - Borovsky (2013) simplified reconnection-rate control:
    R ~ (V * B_s) / (1 + M_A * beta_s^0.5) for southward Bz only
  - Cross-polar cap potential with saturation:
    Phi_PC saturates at ~200 kV due to ionospheric feedback.
  - Clock angle theta_c = arctan(|By|, Bz) measured from +Z in GSM.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Physical constants ---
R_E = 6.371e6              # Earth radius [m]
mu_0 = 4 * np.pi * 1e-7   # permeability of free space [H/m]

print("=" * 65)
print("Solar Wind-Magnetosphere Coupling Functions")
print("=" * 65)


# =========================================================================
# 1. GENERATE SYNTHETIC SOLAR WIND DATA (CME-driven storm)
# =========================================================================
# We simulate a synthetic CME event over 72 hours:
#   - Hours 0-12: quiet pre-storm conditions
#   - Hours 12-14: shock arrival (sudden commencement), speed/density jump
#   - Hours 14-30: sheath region (strong turbulent fields)
#   - Hours 30-48: magnetic cloud (smooth rotation of Bz, strong field)
#   - Hours 48-72: recovery to ambient conditions

np.random.seed(42)  # reproducibility
dt_hours = 0.1      # 6-minute resolution
t_hours = np.arange(0, 72, dt_hours)
N = len(t_hours)

# Solar wind speed V [km/s]
V = np.full(N, 400.0)  # ambient
V[t_hours >= 12] = 400 + 300 * (1 - np.exp(-(t_hours[t_hours >= 12] - 12) / 2))
V[t_hours >= 30] = 600 * np.exp(-(t_hours[t_hours >= 30] - 30) / 30) + 400
V += 15 * np.random.randn(N)  # small fluctuations
V = np.clip(V, 300, 900)

# Proton density n [cm^-3]
n_sw = np.full(N, 5.0)
n_sw[(t_hours >= 12) & (t_hours < 15)] = 25.0  # sheath compression
n_sw[(t_hours >= 15) & (t_hours < 30)] = 12.0  # elevated sheath
n_sw[(t_hours >= 30) & (t_hours < 48)] = 4.0   # magnetic cloud (low density)
n_sw += 1.5 * np.abs(np.random.randn(N))
n_sw = np.clip(n_sw, 1.0, 50.0)

# Dynamic pressure P_dyn [nPa] = 1.67e-6 * n * V^2
# (with n in cm^-3 and V in km/s, P_dyn in nPa)
P_dyn = 1.67e-6 * n_sw * V**2

# IMF components [nT]
# By: fluctuating in sheath, organized in cloud
By = np.full(N, 0.0)
By[(t_hours >= 14) & (t_hours < 30)] = 8 * np.sin(2*np.pi*(t_hours[(t_hours>=14)&(t_hours<30)]-14)/8)
By[(t_hours >= 30) & (t_hours < 48)] = 10 * np.cos(2*np.pi*(t_hours[(t_hours>=30)&(t_hours<48)]-30)/36)
By += 2 * np.random.randn(N)

# Bz: key driver. Strong southward in sheath and first half of cloud
Bz = np.full(N, 1.0)
# Sheath: oscillating with southward bias
sheath_mask = (t_hours >= 14) & (t_hours < 30)
Bz[sheath_mask] = -8 + 6 * np.sin(2*np.pi*(t_hours[sheath_mask]-14)/5)
# Magnetic cloud: smooth rotation from south to north
cloud_mask = (t_hours >= 30) & (t_hours < 48)
cloud_phase = (t_hours[cloud_mask] - 30) / 18.0  # 0 to 1 over cloud
Bz[cloud_mask] = -15 * np.cos(np.pi * cloud_phase)  # south -> north
Bz += 1.5 * np.random.randn(N)

# Total transverse field and clock angle
B_T = np.sqrt(By**2 + Bz**2)
B_total = np.sqrt(By**2 + Bz**2)  # simplified: assume Bx small

# Clock angle: angle of transverse IMF from +Z_GSM
# theta_c = arctan(|By|, Bz) -- ranges from 0 (pure northward) to pi (pure southward)
theta_c = np.arctan2(np.abs(By), Bz)
theta_c = np.clip(theta_c, 0, np.pi)  # ensure [0, pi]

print(f"\nSynthetic CME event: 72-hour duration")
print(f"  Shock arrival: t = 12 h")
print(f"  Peak speed: {V.max():.0f} km/s")
print(f"  Min Bz: {Bz.min():.1f} nT")
print(f"  Max P_dyn: {P_dyn.max():.1f} nPa")


# =========================================================================
# 2. COUPLING FUNCTIONS
# =========================================================================

# --- 2a. Akasofu epsilon parameter [W] ---
# epsilon = (4*pi/mu_0) * V * B^2 * sin^4(theta_c/2) * l_0^2
# V in m/s, B in T, l_0 in m
l_0 = 7.0 * R_E  # empirical length scale
V_mps = V * 1e3   # convert km/s to m/s
B_T_tesla = B_T * 1e-9

epsilon = (4 * np.pi / mu_0) * V_mps * B_T_tesla**2 * np.sin(theta_c/2)**4 * l_0**2
epsilon_GW = epsilon / 1e9  # convert W to GW

print(f"\nAkasofu epsilon:")
print(f"  Peak epsilon = {epsilon.max()/1e12:.2f} TW")
print(f"  Quiet level  ~ {np.mean(epsilon_GW[t_hours < 12]):.1f} GW")

# --- 2b. Newell coupling function [Wb/s or V] ---
# dPhi_MP/dt = V^(4/3) * B_T^(2/3) * sin^(8/3)(theta_c/2)
# V in km/s, B_T in nT -> result is in somewhat arbitrary units
# Newell normalizes to a dimensionless "merging rate" proxy
newell = V**(4.0/3.0) * B_T**(2.0/3.0) * np.sin(theta_c/2)**(8.0/3.0)
# Scale to physically meaningful units: multiply by normalization factor
# Following Newell et al. (2007), factor 1000 gives convenient scale
newell_scaled = newell / 1000.0

print(f"\nNewell coupling function:")
print(f"  Peak dPhi_MP/dt = {newell_scaled.max():.1f} (scaled units)")

# --- 2c. Simplified Borovsky coupling ---
# Borovsky (2013): reconnection rate ~ V * Bs / (1 + beta_s^0.5)
# Bs = max(-Bz, 0) = southward component only
# beta_s = plasma beta of the sheath (simplified as P_dyn / P_mag)
Bs = np.maximum(-Bz, 0) * 1e-9   # southward Bz in Tesla
P_mag = B_total**2 * 1e-18 / (2 * mu_0)  # magnetic pressure [Pa]
P_dyn_Pa = P_dyn * 1e-9  # dynamic pressure in Pa
beta_s = P_dyn_Pa / np.maximum(P_mag, 1e-15)

borovsky = V_mps * Bs / (1.0 + np.sqrt(np.clip(beta_s, 0, 100)))
borovsky_scaled = borovsky / borovsky[borovsky > 0].max() if borovsky.max() > 0 else borovsky

print(f"\nBorovsky coupling (simplified):")
print(f"  Peak reconnection proxy = {borovsky.max():.3e}")

# --- 2d. Cross-polar cap potential with saturation ---
# Phi_PC = a * epsilon^b, but saturates at ~200 kV
# Empirical: Phi_PC = 30 + 0.01 * epsilon_GW, capped at 200 kV
# More physically: Phi_PC = Phi_sat * tanh(epsilon / epsilon_sat)
Phi_sat = 200.0    # saturation potential [kV]
epsilon_sat = 500   # epsilon value at which saturation begins [GW]
Phi_PC = Phi_sat * np.tanh(epsilon_GW / epsilon_sat)

# Also compute unsaturated (linear) version for comparison
Phi_PC_linear = 30 + 0.05 * epsilon_GW

print(f"\nCross-polar cap potential:")
print(f"  Peak Phi_PC (saturated)   = {Phi_PC.max():.1f} kV")
print(f"  Peak Phi_PC (unsaturated) = {Phi_PC_linear.max():.1f} kV")

# --- 2e. Synthetic Dst (for comparison with coupling functions) ---
# Simple Burton-like model: dDst/dt = Q - Dst/tau
# Q proportional to VBs (southward IMF only)
tau_decay = 7.7       # decay time [hours]
Dst_syn = np.zeros(N)
for i in range(1, N):
    VBs = V[i] * max(-Bz[i], 0) / 1000.0  # mV/m
    Q = -4.4 * (VBs - 0.5) if VBs > 0.5 else 0  # injection [nT/h]
    dDst = (Q - Dst_syn[i-1] / tau_decay) * dt_hours
    Dst_syn[i] = Dst_syn[i-1] + dDst


# =========================================================================
# 3. PLOTTING
# =========================================================================
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
fig.suptitle("Solar Wind-Magnetosphere Coupling Functions\n(Synthetic CME Event)",
             fontsize=14, y=0.98)

# Color the storm phases
for ax in axes:
    ax.axvspan(12, 14, color='yellow', alpha=0.1, label='_nolegend_')   # shock
    ax.axvspan(14, 30, color='orange', alpha=0.1, label='_nolegend_')   # sheath
    ax.axvspan(30, 48, color='lightblue', alpha=0.1, label='_nolegend_')  # cloud

# --- Panel 1: Solar wind inputs ---
ax1 = axes[0]
ax1b = ax1.twinx()
ax1.plot(t_hours, V, 'b-', lw=1, label='V [km/s]')
ax1b.plot(t_hours, Bz, 'r-', lw=1, label='$B_z$ [nT]')
ax1.set_ylabel("V [km/s]", color='b', fontsize=11)
ax1b.set_ylabel("$B_z$ [nT]", color='r', fontsize=11)
ax1.set_title("Solar Wind Inputs", fontsize=12)
ax1.legend(loc='upper left', fontsize=8)
ax1b.legend(loc='upper right', fontsize=8)
ax1b.axhline(0, color='gray', ls=':', lw=0.5)

# Phase annotations
ax1.text(6, V.max()*0.95, 'Quiet', fontsize=9, ha='center', style='italic')
ax1.text(13, V.max()*0.95, 'Shock', fontsize=9, ha='center', style='italic')
ax1.text(22, V.max()*0.95, 'Sheath', fontsize=9, ha='center', style='italic')
ax1.text(39, V.max()*0.95, 'MC', fontsize=9, ha='center', style='italic')
ax1.text(60, V.max()*0.95, 'Recovery', fontsize=9, ha='center', style='italic')

# --- Panel 2: Akasofu epsilon ---
axes[1].plot(t_hours, epsilon_GW, 'b-', lw=1.2, label=r'$\varepsilon$ (Akasofu)')
axes[1].set_ylabel(r"$\varepsilon$ [GW]", fontsize=11)
axes[1].set_title("Akasofu Epsilon Parameter", fontsize=12)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# --- Panel 3: Newell coupling ---
axes[2].plot(t_hours, newell_scaled, 'g-', lw=1.2,
             label=r'$d\Phi_{MP}/dt$ (Newell)')
axes[2].set_ylabel("Newell coupling\n[scaled]", fontsize=11)
axes[2].set_title("Newell Universal Coupling Function", fontsize=12)
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

# --- Panel 4: Cross-polar cap potential ---
axes[3].plot(t_hours, Phi_PC, 'purple', lw=1.5, label=r'$\Phi_{PC}$ (saturated)')
axes[3].plot(t_hours, np.clip(Phi_PC_linear, 0, 800), 'purple', lw=1, ls='--',
             alpha=0.5, label=r'$\Phi_{PC}$ (unsaturated)')
axes[3].axhline(Phi_sat, color='red', ls=':', lw=1, label=f'Saturation ({Phi_sat} kV)')
axes[3].set_ylabel("$\\Phi_{PC}$ [kV]", fontsize=11)
axes[3].set_title("Cross-Polar Cap Potential", fontsize=12)
axes[3].legend(fontsize=8)
axes[3].grid(True, alpha=0.3)
axes[3].set_ylim(0, 300)

# --- Panel 5: Synthetic Dst ---
axes[4].plot(t_hours, Dst_syn, 'k-', lw=1.5, label='Dst (Burton model)')
axes[4].axhline(-50, color='orange', ls='--', alpha=0.5, label='Moderate storm')
axes[4].axhline(-100, color='red', ls='--', alpha=0.5, label='Intense storm')
axes[4].set_ylabel("Dst [nT]", fontsize=11)
axes[4].set_xlabel("Time [hours]", fontsize=11)
axes[4].set_title("Synthetic Dst (for comparison)", fontsize=12)
axes[4].legend(fontsize=8)
axes[4].grid(True, alpha=0.3)
axes[4].invert_yaxis()

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/opt/projects/01_Personal/03_Study/examples/Space_Weather/03_coupling_functions.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: 03_coupling_functions.png")
