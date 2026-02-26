"""
Radiative Transfer: Limb Darkening of the Solar Disk.

Demonstrates why the Sun appears darker at its edges (limb) compared to
the center. This is a direct consequence of radiative transfer: we see
deeper (hotter) layers at disk center and shallower (cooler) layers at
the limb.

Key physics:
  - Eddington-Barbier relation: I(mu) ~ S(tau=mu) where mu = cos(theta)
  - Source function increases with depth (hotter deeper layers)
  - Linear limb darkening: I(mu)/I(1) = 1 - u*(1-mu)
  - Wavelength dependence: stronger darkening at shorter wavelengths
  - Related to opacity and temperature gradient in photosphere
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# --- Limb darkening models ---
def limb_darkening_linear(mu, u):
    """Linear limb darkening law: I(mu)/I(1) = 1 - u*(1-mu)."""
    return 1.0 - u * (1.0 - mu)

def limb_darkening_quadratic(mu, a, b):
    """Quadratic limb darkening law: I(mu)/I(1) = 1 - a*(1-mu) - b*(1-mu)^2."""
    return 1.0 - a * (1.0 - mu) - b * (1.0 - mu)**2

def limb_darkening_sqrt(mu, c, d):
    """Square-root limb darkening: I(mu)/I(1) = 1 - c*(1-mu) - d*(1-sqrt(mu))."""
    return 1.0 - c * (1.0 - mu) - d * (1.0 - np.sqrt(mu))

# --- Limb darkening coefficients for different wavelengths ---
# Approximate values from Neckel & Labs (1994) and Claret (2000)
wavelengths = {
    'UV (350 nm)':     {'u': 0.85, 'a': 0.93, 'b': -0.23, 'color': 'violet'},
    'Blue (450 nm)':   {'u': 0.75, 'a': 0.78, 'b': -0.09, 'color': 'blue'},
    'Green (550 nm)':  {'u': 0.60, 'a': 0.62, 'b': -0.03, 'color': 'green'},
    'Red (650 nm)':    {'u': 0.47, 'a': 0.50, 'b':  0.01, 'color': 'red'},
    'IR (800 nm)':     {'u': 0.33, 'a': 0.35, 'b':  0.02, 'color': 'darkred'},
    'Far IR (1600 nm)':{'u': 0.18, 'a': 0.20, 'b':  0.01, 'color': 'maroon'},
}

# --- Effective temperature from Eddington approximation ---
# In grey atmosphere: T^4(tau) = (3/4) T_eff^4 * (tau + 2/3)
# At tau=0 (surface): T_surface = T_eff * (1/2)^(1/4) ~ 0.84 * T_eff
# At tau=2/3: T = T_eff (definition of effective temperature)
T_eff = 5778  # K (solar effective temperature)
print("=" * 60)
print("Limb Darkening and Radiative Transfer")
print(f"  Solar effective temperature: T_eff = {T_eff} K")
print(f"  Surface temperature (tau=0): T_s = {T_eff * 0.5**0.25:.0f} K")
print(f"  T at tau = 2/3: {T_eff:.0f} K (by definition)")
print("=" * 60)

# --- Plot limb darkening curves ---
mu = np.linspace(0.01, 1.0, 500)
theta_deg = np.degrees(np.arccos(mu))

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Solar Limb Darkening", fontsize=14, y=0.98)

# Panel 1: Linear limb darkening at different wavelengths
ax = axes[0, 0]
for name, params in wavelengths.items():
    I_ratio = limb_darkening_linear(mu, params['u'])
    ax.plot(mu, I_ratio, color=params['color'], lw=2, label=f"{name} (u={params['u']:.2f})")
ax.set_xlabel(r"$\mu = \cos\theta$")
ax.set_ylabel(r"$I(\mu) / I(1)$")
ax.set_title("Linear Limb Darkening: $I/I_0 = 1 - u(1-\\mu)$")
ax.legend(fontsize=7, loc='lower right')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)

# Panel 2: Compare linear, quadratic, sqrt laws (green/550nm)
ax = axes[0, 1]
u, a, b = 0.60, 0.62, -0.03
c_sqrt, d_sqrt = 0.38, 0.42  # approximate sqrt law coefficients
ax.plot(mu, limb_darkening_linear(mu, u), 'g-', lw=2, label=f'Linear (u={u})')
ax.plot(mu, limb_darkening_quadratic(mu, a, b), 'b--', lw=2,
        label=f'Quadratic (a={a}, b={b})')
ax.plot(mu, limb_darkening_sqrt(mu, c_sqrt, d_sqrt), 'r:', lw=2,
        label=f'Sqrt (c={c_sqrt}, d={d_sqrt})')
# Eddington grey atmosphere prediction
I_eddington = (2 + 3 * mu) / 5  # I(mu)/I(1) for grey atmosphere
ax.plot(mu, I_eddington, 'k-.', lw=2, label='Eddington grey atm.')
ax.set_xlabel(r"$\mu = \cos\theta$")
ax.set_ylabel(r"$I(\mu) / I(1)$")
ax.set_title("Comparison of Limb Darkening Laws (550 nm)")
ax.legend(fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)

# Panel 3: 2D solar disk with limb darkening
ax = axes[1, 0]
N = 500
x = np.linspace(-1.2, 1.2, N)
y = np.linspace(-1.2, 1.2, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# mu = sqrt(1 - r^2) for disk
mu_disk = np.sqrt(np.maximum(1.0 - R**2, 0))
# Apply green-band limb darkening
u_green = 0.60
I_disk = limb_darkening_linear(mu_disk, u_green)
I_disk[R > 1.0] = 0  # outside disk

# Use solar-like colormap
im = ax.imshow(I_disk, extent=[-1.2, 1.2, -1.2, 1.2],
               cmap='hot', vmin=0, vmax=1.0, origin='lower')
# Draw disk edge
theta_circle = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'w-', lw=1, alpha=0.5)
ax.set_xlabel(r"$x / R_\odot$")
ax.set_ylabel(r"$y / R_\odot$")
ax.set_title("Solar Disk (green, u=0.60)")
ax.set_aspect('equal')
plt.colorbar(im, ax=ax, label=r'$I / I_{center}$', shrink=0.8)

# Panel 4: Radial intensity profile across disk
ax = axes[1, 1]
r_disk = np.linspace(0, 1, 500)
mu_r = np.sqrt(1.0 - r_disk**2)
for name, params in wavelengths.items():
    I_r = limb_darkening_linear(mu_r, params['u'])
    ax.plot(r_disk, I_r, color=params['color'], lw=2, label=name)
ax.set_xlabel(r"$r / R_\odot$ (distance from center)")
ax.set_ylabel(r"$I(r) / I_{center}$")
ax.set_title("Intensity Profile Across Solar Disk")
ax.axvline(np.sin(np.radians(60)), color='gray', ls=':', alpha=0.5, label=r'$\theta=60°$')
ax.legend(fontsize=7, loc='lower left')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/opt/projects/01_Personal/03_Study/examples/Solar_Physics/04_limb_darkening.png",
            dpi=150, bbox_inches='tight')
plt.show()

# --- Print comparative intensities ---
print("\nCenter-to-limb intensity ratio at different mu values:")
print(f"{'mu':>6} {'theta':>6}", end="")
for name in wavelengths:
    short = name.split('(')[0].strip()
    print(f" {short:>10}", end="")
print()
for mu_val in [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]:
    theta_val = np.degrees(np.arccos(mu_val))
    print(f"{mu_val:6.2f} {theta_val:5.1f}°", end="")
    for name, params in wavelengths.items():
        I_val = limb_darkening_linear(mu_val, params['u'])
        print(f" {I_val:10.3f}", end="")
    print()

print("\nPlot saved: 04_limb_darkening.png")
