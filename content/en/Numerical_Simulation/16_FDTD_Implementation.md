# 16. FDTD Implementation

## Learning Objectives
- Complete implementation of 1D FDTD
- Source excitation methods (Gaussian pulse, sinusoidal)
- Absorbing boundary conditions (Simple ABC, Mur ABC)
- 2D FDTD (TM, TE modes)
- PML (Perfectly Matched Layer) concept

**Why This Lesson Matters:** FDTD (Finite-Difference Time-Domain) is the workhorse method for computational electromagnetics. Its appeal lies in simplicity: leapfrog time-stepping on a staggered Yee grid naturally satisfies Gauss's laws and produces second-order accuracy in both time and space. Because it operates directly in the time domain, FDTD can handle broadband sources, nonlinear materials, and complex geometries with a single simulation run.

---

## 1. Complete 1D FDTD Implementation

### 1.1 Basic Structure

```
1D FDTD Algorithm:

Initialization:
- Grid setup (Nx, dx, dt)
- Array initialization (Ey, Hz)
- Material properties (ε, μ, σ)

Time loop:
for n = 1, 2, ..., Nt:
    1. H update: Hz^(n+1/2) = f(Hz^(n-1/2), Ey^n)
    2. Source injection (soft/hard)
    3. E update: Ey^(n+1) = f(Ey^n, Hz^(n+1/2))
    4. Apply boundary conditions (ABC)
    5. Record/output data
```

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Physical constants
c0 = 299792458.0  # Speed of light [m/s]
eps0 = 8.854187817e-12  # Vacuum permittivity [F/m]
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability [H/m]
eta0 = np.sqrt(mu0 / eps0)  # Vacuum impedance [Ω]

class FDTD_1D:
    """1D FDTD Simulator"""

    def __init__(self, Nx=200, dx=1e-3, courant=0.99):
        """
        Parameters:
        - Nx: Number of grid points
        - dx: Spatial interval [m]
        - courant: Courant number (≤ 1)
        """
        self.Nx = Nx
        self.dx = dx
        # CFL condition for EM waves: dt <= dx/c0. The Courant number sets how
        # close to this limit we operate. Values near 1 minimize dispersion error
        # but leave less margin for material interfaces.
        self.dt = courant * dx / c0

        # Field arrays
        self.Ey = np.zeros(Nx)
        self.Hz = np.zeros(Nx)

        # Material property arrays (relative values)
        self.eps_r = np.ones(Nx)
        self.mu_r = np.ones(Nx)
        self.sigma = np.zeros(Nx)  # Electric conductivity
        self.sigma_m = np.zeros(Nx)  # Magnetic conductivity

        # Previous field values for ABC
        self.Ey_left_prev = [0, 0]
        self.Ey_right_prev = [0, 0]

        # Time
        self.time_step = 0

        print(f"1D FDTD Initialization:")
        print(f"  Nx = {Nx}, dx = {dx*1000:.2f} mm")
        print(f"  dt = {self.dt*1e12:.3f} ps")
        print(f"  Courant S = {courant}")

    def set_material(self, start, end, eps_r=1, sigma=0):
        """Set material region"""
        self.eps_r[start:end] = eps_r
        self.sigma[start:end] = sigma

    def compute_coefficients(self):
        """Compute update coefficients"""
        eps = eps0 * self.eps_r
        mu = mu0 * self.mu_r

        # E update coefficients (with loss)
        self.Ca = (1 - self.sigma * self.dt / (2 * eps)) / \
                  (1 + self.sigma * self.dt / (2 * eps))
        self.Cb = (self.dt / (eps * self.dx)) / \
                  (1 + self.sigma * self.dt / (2 * eps))

        # H update coefficients
        self.Da = (1 - self.sigma_m * self.dt / (2 * mu)) / \
                  (1 + self.sigma_m * self.dt / (2 * mu))
        self.Db = (self.dt / (mu * self.dx)) / \
                  (1 + self.sigma_m * self.dt / (2 * mu))

    def update_H(self):
        """H field update"""
        self.Hz[:-1] = (self.Da[:-1] * self.Hz[:-1] -
                       self.Db[:-1] * (self.Ey[1:] - self.Ey[:-1]))

    def update_E(self):
        """E field update"""
        self.Ey[1:-1] = (self.Ca[1:-1] * self.Ey[1:-1] -
                        self.Cb[1:-1] * (self.Hz[1:-1] - self.Hz[:-2]))

    def add_source_soft(self, position, value):
        """Soft source (total field/scattered field boundary)"""
        self.Ey[position] += value

    def add_source_hard(self, position, value):
        """Hard source (forced injection)"""
        self.Ey[position] = value

    def apply_abc_simple(self):
        """Simple absorbing boundary condition (1st order)"""
        # Left boundary
        self.Ey[0] = self.Ey_left_prev[0]
        self.Ey_left_prev[0] = self.Ey_left_prev[1]
        self.Ey_left_prev[1] = self.Ey[1]

        # Right boundary
        self.Ey[-1] = self.Ey_right_prev[0]
        self.Ey_right_prev[0] = self.Ey_right_prev[1]
        self.Ey_right_prev[1] = self.Ey[-2]

    def apply_abc_mur(self):
        """Mur 1st order absorbing boundary condition"""
        coeff = (c0 * self.dt - self.dx) / (c0 * self.dt + self.dx)

        # Left boundary
        self.Ey[0] = self.Ey_left_prev[1] + coeff * (self.Ey[1] - self.Ey_left_prev[0])
        self.Ey_left_prev[0] = self.Ey[0]
        self.Ey_left_prev[1] = self.Ey[1]

        # Right boundary
        self.Ey[-1] = self.Ey_right_prev[1] + coeff * (self.Ey[-2] - self.Ey_right_prev[0])
        self.Ey_right_prev[0] = self.Ey[-1]
        self.Ey_right_prev[1] = self.Ey[-2]

    def step(self, source_func=None, source_pos=None, abc_type='mur'):
        """Advance one time step"""
        self.update_H()

        if source_func is not None and source_pos is not None:
            t = self.time_step * self.dt
            self.add_source_soft(source_pos, source_func(t))

        self.update_E()

        if abc_type == 'simple':
            self.apply_abc_simple()
        elif abc_type == 'mur':
            self.apply_abc_mur()
        else:  # PEC
            self.Ey[0] = 0
            self.Ey[-1] = 0

        self.time_step += 1

    def run(self, n_steps, source_func=None, source_pos=None, abc_type='mur',
           record_interval=1):
        """Run simulation"""
        self.compute_coefficients()

        Ey_history = []
        Hz_history = []

        for n in range(n_steps):
            self.step(source_func, source_pos, abc_type)

            if n % record_interval == 0:
                Ey_history.append(self.Ey.copy())
                Hz_history.append(self.Hz.copy())

        return np.array(Ey_history), np.array(Hz_history)


def gaussian_pulse(t, t0=1e-10, tau=3e-11):
    """Gaussian pulse source"""
    return np.exp(-((t - t0) / tau) ** 2)

def sinusoidal_source(t, freq=3e9, t0=5e-11, tau=2e-11):
    """Modulated sinusoidal source"""
    envelope = 1 - np.exp(-((t - t0) / tau) ** 2) if t < t0 else 1
    return envelope * np.sin(2 * np.pi * freq * t)


def demo_1d_fdtd_basic():
    """1D FDTD basic demonstration"""

    # Create simulator
    fdtd = FDTD_1D(Nx=300, dx=1e-3, courant=0.99)

    # Add dielectric slab
    fdtd.set_material(150, 200, eps_r=4.0)

    # Run simulation
    source_pos = 50
    n_steps = 500

    Ey_history, Hz_history = fdtd.run(
        n_steps,
        source_func=gaussian_pulse,
        source_pos=source_pos,
        abc_type='mur',
        record_interval=5
    )

    # Result visualization
    x = np.arange(fdtd.Nx) * fdtd.dx * 1000  # mm

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Snapshots
    snapshot_indices = [10, 30, 50, 70, 90]

    for idx, snap_idx in enumerate(snapshot_indices):
        if idx < 5:
            ax = axes[idx // 3, idx % 3]
            ax.plot(x, Ey_history[snap_idx], 'b-', linewidth=1.5)

            # Mark dielectric region
            ax.axvspan(150 * fdtd.dx * 1000, 200 * fdtd.dx * 1000,
                      alpha=0.2, color='green', label=r'$\epsilon_r=4$')
            ax.axvline(x=source_pos * fdtd.dx * 1000, color='red',
                      linestyle='--', alpha=0.5)

            ax.set_xlabel('x [mm]')
            ax.set_ylabel('Ey')
            ax.set_title(f'Step {snap_idx * 5}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.2, 1.2)

    # Last subplot: space-time diagram
    ax = axes[1, 2]
    t = np.arange(len(Ey_history)) * 5 * fdtd.dt * 1e9  # ns

    im = ax.pcolormesh(x, t, Ey_history, cmap='RdBu_r', shading='auto',
                      vmin=-0.5, vmax=0.5)
    ax.axvline(x=150 * fdtd.dx * 1000, color='green', linestyle='-', linewidth=2)
    ax.axvline(x=200 * fdtd.dx * 1000, color='green', linestyle='-', linewidth=2)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('t [ns]')
    ax.set_title('Space-Time Diagram')
    plt.colorbar(im, ax=ax, label='Ey')

    plt.suptitle('1D FDTD: Reflection and Transmission at Dielectric Slab', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_1d_dielectric.png', dpi=150, bbox_inches='tight')
    plt.show()

    return fdtd, Ey_history

# fdtd, Ey_history = demo_1d_fdtd_basic()
```

---

## 2. Source Excitation Methods

### 2.1 Hard Source vs Soft Source

```
Source Types:

1. Hard Source:
   Ey[source_pos] = source_value
   - Force field value at that point
   - Reflected wave reflects again at source
   - Simple but causes non-physical reflection

2. Soft Source:
   Ey[source_pos] += source_value
   - Add source to existing field
   - Reflected wave passes through source
   - Used with TF/SF boundary

3. TF/SF (Total-Field/Scattered-Field):
   - Separates incident and scattered waves
   - Accurate plane wave injection
   - Requires additional correction terms
```

```python
def source_comparison():
    """Compare hard source and soft source"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, source_type in enumerate(['hard', 'soft']):
        # Create simulator
        fdtd = FDTD_1D(Nx=200, dx=1e-3, courant=0.99)
        fdtd.compute_coefficients()

        source_pos = 50
        n_steps = 300

        # Add reflector (PEC)
        fdtd.sigma[150:155] = 1e7

        Ey_history = []

        for n in range(n_steps):
            fdtd.update_H()

            t = n * fdtd.dt
            source = gaussian_pulse(t, t0=5e-11, tau=2e-11)

            if source_type == 'hard':
                fdtd.Ey[source_pos] = source
            else:
                fdtd.Ey[source_pos] += source

            fdtd.update_E()
            fdtd.apply_abc_mur()

            if n % 3 == 0:
                Ey_history.append(fdtd.Ey.copy())

        x = np.arange(fdtd.Nx) * fdtd.dx * 1000

        # Snapshots
        for col, snap_idx in enumerate([20, 50, 80]):
            ax = axes[row, col]
            ax.plot(x, Ey_history[snap_idx], 'b-', linewidth=1.5)
            ax.axvline(x=source_pos * fdtd.dx * 1000, color='red', linestyle='--',
                      label='Source')
            ax.axvspan(150 * fdtd.dx * 1000, 155 * fdtd.dx * 1000,
                      alpha=0.3, color='gray', label='PEC')

            ax.set_xlabel('x [mm]')
            ax.set_ylabel('Ey')
            ax.set_title(f'{source_type.capitalize()} Source, Step {snap_idx * 3}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.5, 1.5)
            if col == 0:
                ax.legend()

    plt.suptitle('Hard Source vs Soft Source Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_source_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# source_comparison()
```

### 2.2 Various Source Waveforms

```python
def source_waveforms():
    """Various source waveforms"""

    t = np.linspace(0, 0.5e-9, 1000)  # 0.5 ns

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (1) Gaussian pulse
    ax1 = axes[0, 0]
    t0, tau = 0.15e-9, 0.03e-9
    pulse = np.exp(-((t - t0) / tau) ** 2)
    ax1.plot(t * 1e9, pulse, 'b-', linewidth=2)
    ax1.set_title('Gaussian Pulse')
    ax1.set_xlabel('t [ns]')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)

    # (2) Gaussian derivative (Ricker wavelet)
    ax2 = axes[0, 1]
    ricker = -2 * (t - t0) / tau**2 * np.exp(-((t - t0) / tau) ** 2)
    ax2.plot(t * 1e9, ricker, 'r-', linewidth=2)
    ax2.set_title('Gaussian Derivative (Ricker Wavelet)')
    ax2.set_xlabel('t [ns]')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)

    # (3) Modulated sinusoidal
    ax3 = axes[1, 0]
    freq = 10e9  # 10 GHz
    modulated = np.sin(2 * np.pi * freq * t) * np.exp(-((t - t0) / tau) ** 2)
    ax3.plot(t * 1e9, modulated, 'g-', linewidth=1.5)
    ax3.set_title('Modulated Sinusoidal (10 GHz)')
    ax3.set_xlabel('t [ns]')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)

    # (4) Spectrum
    ax4 = axes[1, 1]
    from scipy.fft import fft, fftfreq

    dt = t[1] - t[0]
    freqs = fftfreq(len(t), dt)
    positive = freqs > 0

    for signal, label, color in [(pulse, 'Gaussian', 'b'),
                                  (ricker, 'Ricker', 'r'),
                                  (modulated, 'Modulated', 'g')]:
        spectrum = np.abs(fft(signal))
        ax4.plot(freqs[positive] * 1e-9, spectrum[positive] / max(spectrum[positive]),
                linewidth=1.5, label=label, color=color)

    ax4.set_xlim(0, 50)
    ax4.set_xlabel('Frequency [GHz]')
    ax4.set_ylabel('Normalized Amplitude')
    ax4.set_title('Frequency Spectrum')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fdtd_source_waveforms.png', dpi=150, bbox_inches='tight')
    plt.show()

# source_waveforms()
```

---

## 3. Absorbing Boundary Conditions (ABC)

### 3.1 Simple ABC

```
1st Order Simple ABC:

Characteristics of 1D wave equation:
(d/dt + c d/dx) Ey = 0  (right-traveling wave)
(d/dt - c d/dx) Ey = 0  (left-traveling wave)

Discretization (right boundary, absorb left-traveling wave):
Ey^(n+1)[Nx-1] = Ey^n[Nx-2]

This is exact only when S = cdt/dx = 1
Reflection occurs when S != 1
```

### 3.2 Mur ABC

```
Mur 1st Order ABC (1981):

Finite difference approximation of 1D wave equation:

Right boundary (x = xmax):
(Ey^(n+1)[Nx-1] - Ey^n[Nx-2]) / (cdt + dx) =
(Ey^n[Nx-1] - Ey^(n+1)[Nx-2]) / (cdt - dx)

Rearranged:
Ey^(n+1)[Nx-1] = Ey^n[Nx-2] +
                 (cdt - dx)/(cdt + dx) * (Ey^(n+1)[Nx-2] - Ey^n[Nx-1])

Advantages:
- More effective over wider range of incidence angles than Simple ABC
- Simple implementation

Limitations:
- Only absorbs normal incidence exactly
- Reflection occurs at oblique incidence
```

```python
def abc_comparison():
    """Absorbing boundary condition comparison"""

    abc_types = ['pec', 'simple', 'mur']
    results = {}

    for abc_type in abc_types:
        fdtd = FDTD_1D(Nx=300, dx=1e-3, courant=0.99)
        fdtd.compute_coefficients()

        source_pos = 100
        Ey_history = []

        for n in range(400):
            fdtd.update_H()

            t = n * fdtd.dt
            source = gaussian_pulse(t, t0=5e-11, tau=2e-11)
            fdtd.Ey[source_pos] += source

            fdtd.update_E()

            if abc_type == 'pec':
                fdtd.Ey[0] = 0
                fdtd.Ey[-1] = 0
            elif abc_type == 'simple':
                fdtd.apply_abc_simple()
            else:
                fdtd.apply_abc_mur()

            if n % 4 == 0:
                Ey_history.append(fdtd.Ey.copy())

        results[abc_type] = np.array(Ey_history)

    # Visualization
    x = np.arange(300) * 1e-3 * 1000

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for row, abc_type in enumerate(abc_types):
        Ey_history = results[abc_type]

        for col, snap_idx in enumerate([20, 50, 80]):
            ax = axes[row, col]
            ax.plot(x, Ey_history[snap_idx], 'b-', linewidth=1.5)
            ax.axvline(x=100, color='red', linestyle='--', alpha=0.5)
            ax.set_ylim(-1.2, 1.2)
            ax.grid(True, alpha=0.3)

            if row == 2:
                ax.set_xlabel('x [mm]')
            if col == 0:
                ax.set_ylabel(f'{abc_type.upper()}\nEy')

            ax.set_title(f'Step {snap_idx * 4}')

    plt.suptitle('Absorbing Boundary Condition Comparison: PEC vs Simple ABC vs Mur ABC', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_abc_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Reflection analysis
    fig, ax = plt.subplots(figsize=(10, 5))

    t = np.arange(len(results['pec'])) * 4 * 1e-3 / c0 * 1e9

    # Field record at specific position (for reflection detection)
    probe_pos = 50

    for abc_type, color in [('pec', 'red'), ('simple', 'blue'), ('mur', 'green')]:
        field = results[abc_type][:, probe_pos]
        ax.plot(t, field, color=color, linewidth=1.5, label=abc_type.upper())

    ax.set_xlabel('t [ns]')
    ax.set_ylabel('Ey at probe')
    ax.set_title('Reflection Comparison at Boundary (probe at x = 50 mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fdtd_abc_reflection.png', dpi=150, bbox_inches='tight')
    plt.show()

# abc_comparison()
```

---

## 4. 2D FDTD Implementation

### 4.1 TM Mode (Ez, Hx, Hy)

```python
class FDTD_2D_TM:
    """2D FDTD TM Mode Simulator"""

    def __init__(self, Nx=100, Ny=100, dx=1e-3, dy=1e-3, courant=0.99):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy

        # Courant condition
        self.dt = courant / (c0 * np.sqrt(1/dx**2 + 1/dy**2))

        # Field arrays
        self.Ez = np.zeros((Ny, Nx))
        self.Hx = np.zeros((Ny, Nx))
        self.Hy = np.zeros((Ny, Nx))

        # Material property arrays
        self.eps_r = np.ones((Ny, Nx))
        self.mu_r = np.ones((Ny, Nx))
        self.sigma = np.zeros((Ny, Nx))

        # Coefficients
        self.Ca = None
        self.Cb = None

        self.time_step = 0

        print(f"2D FDTD TM Mode Initialization:")
        print(f"  Grid: {Nx} x {Ny}")
        print(f"  dx = {dx*1000:.2f} mm, dy = {dy*1000:.2f} mm")
        print(f"  dt = {self.dt*1e12:.3f} ps")

    def set_material_region(self, x1, x2, y1, y2, eps_r=1, sigma=0):
        """Set material region"""
        self.eps_r[y1:y2, x1:x2] = eps_r
        self.sigma[y1:y2, x1:x2] = sigma

    def add_pec_circle(self, cx, cy, radius):
        """Add circular PEC"""
        for j in range(self.Ny):
            for i in range(self.Nx):
                if (i - cx)**2 + (j - cy)**2 < radius**2:
                    self.sigma[j, i] = 1e7

    def compute_coefficients(self):
        """Compute update coefficients"""
        eps = eps0 * self.eps_r

        self.Ca = (1 - self.sigma * self.dt / (2 * eps)) / \
                  (1 + self.sigma * self.dt / (2 * eps))
        self.Cb = (self.dt / eps) / (1 + self.sigma * self.dt / (2 * eps))

    def update_H(self):
        """H field update"""
        # Hx update: Hx = Hx - dt/mu0 * dEz/dy
        self.Hx[:, :-1] -= (self.dt / (mu0 * self.dy)) * \
                          (self.Ez[:, 1:] - self.Ez[:, :-1])

        # Hy update: Hy = Hy + dt/mu0 * dEz/dx
        self.Hy[:-1, :] += (self.dt / (mu0 * self.dx)) * \
                          (self.Ez[1:, :] - self.Ez[:-1, :])

    def update_E(self):
        """E field update"""
        # Ez update: Ez = Ca*Ez + Cb*(dHy/dx - dHx/dy)
        self.Ez[1:-1, 1:-1] = (
            self.Ca[1:-1, 1:-1] * self.Ez[1:-1, 1:-1] +
            self.Cb[1:-1, 1:-1] * (
                (self.Hy[1:-1, 1:-1] - self.Hy[:-2, 1:-1]) / self.dx -
                (self.Hx[1:-1, 1:-1] - self.Hx[1:-1, :-2]) / self.dy
            )
        )

    def apply_pec_boundary(self):
        """PEC boundary condition"""
        self.Ez[0, :] = 0
        self.Ez[-1, :] = 0
        self.Ez[:, 0] = 0
        self.Ez[:, -1] = 0

    def add_point_source(self, x, y, value):
        """Add point source"""
        self.Ez[y, x] += value

    def add_line_source(self, x, value):
        """Add line source (entire y direction)"""
        self.Ez[:, x] += value

    def step(self, source_func=None, source_pos=None, source_type='point'):
        """Advance one time step"""
        self.update_H()

        if source_func is not None and source_pos is not None:
            t = self.time_step * self.dt
            value = source_func(t)

            if source_type == 'point':
                self.add_point_source(source_pos[0], source_pos[1], value)
            elif source_type == 'line':
                self.add_line_source(source_pos, value)

        self.update_E()
        self.apply_pec_boundary()

        self.time_step += 1

    def run(self, n_steps, source_func=None, source_pos=None,
           source_type='point', record_interval=1):
        """Run simulation"""
        self.compute_coefficients()

        Ez_history = []

        for n in range(n_steps):
            self.step(source_func, source_pos, source_type)

            if n % record_interval == 0:
                Ez_history.append(self.Ez.copy())

        return np.array(Ez_history)


def demo_2d_fdtd_tm():
    """2D FDTD TM mode demonstration"""

    # Create simulator
    fdtd = FDTD_2D_TM(Nx=150, Ny=150, dx=1e-3, dy=1e-3, courant=0.9)

    # Add dielectric block
    fdtd.set_material_region(90, 120, 60, 90, eps_r=4)

    # Add PEC circular cylinder
    fdtd.add_pec_circle(50, 75, 15)

    # Run simulation
    source_pos = (20, 75)  # (x, y)

    def source(t):
        return gaussian_pulse(t, t0=1e-10, tau=3e-11)

    Ez_history = fdtd.run(
        n_steps=300,
        source_func=source,
        source_pos=source_pos,
        source_type='point',
        record_interval=5
    )

    # Visualization
    x = np.arange(fdtd.Nx) * fdtd.dx * 1000
    y = np.arange(fdtd.Ny) * fdtd.dy * 1000
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    snapshot_indices = [5, 15, 25, 35, 45, 55]

    for idx, snap_idx in enumerate(snapshot_indices):
        ax = axes[idx // 3, idx % 3]
        vmax = np.max(np.abs(Ez_history[snap_idx])) * 0.7
        if vmax == 0:
            vmax = 1

        im = ax.pcolormesh(X, Y, Ez_history[snap_idx], cmap='RdBu_r',
                          shading='auto', vmin=-vmax, vmax=vmax)

        # Mark material boundaries
        ax.contour(X, Y, fdtd.eps_r, levels=[2], colors='green', linewidths=2)
        ax.contour(X, Y, fdtd.sigma, levels=[1e6], colors='black', linewidths=2)

        # Source position
        ax.plot(source_pos[0] * fdtd.dx * 1000,
               source_pos[1] * fdtd.dy * 1000, 'r*', markersize=10)

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'Step {snap_idx * 5}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Ez')

    plt.suptitle('2D FDTD TM Mode: Scattering Simulation', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_2d_tm_scattering.png', dpi=150, bbox_inches='tight')
    plt.show()

    return fdtd, Ez_history

# fdtd, Ez_history = demo_2d_fdtd_tm()
```

### 4.2 TE Mode (Hz, Ex, Ey)

```python
class FDTD_2D_TE:
    """2D FDTD TE Mode Simulator"""

    def __init__(self, Nx=100, Ny=100, dx=1e-3, dy=1e-3, courant=0.99):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.dt = courant / (c0 * np.sqrt(1/dx**2 + 1/dy**2))

        # Field arrays
        self.Hz = np.zeros((Ny, Nx))
        self.Ex = np.zeros((Ny, Nx))
        self.Ey = np.zeros((Ny, Nx))

        # Material properties
        self.eps_r = np.ones((Ny, Nx))
        self.sigma = np.zeros((Ny, Nx))

        self.time_step = 0

    def compute_coefficients(self):
        eps = eps0 * self.eps_r
        self.Ca = (1 - self.sigma * self.dt / (2 * eps)) / \
                  (1 + self.sigma * self.dt / (2 * eps))
        self.Cb = (self.dt / eps) / (1 + self.sigma * self.dt / (2 * eps))

    def update_E(self):
        """E field update"""
        # Ex = Ca*Ex + Cb * dHz/dy
        self.Ex[1:-1, :] = (
            self.Ca[1:-1, :] * self.Ex[1:-1, :] +
            self.Cb[1:-1, :] * (self.Hz[1:, :] - self.Hz[:-2, :]) / (2 * self.dy)
        )

        # Ey = Ca*Ey - Cb * dHz/dx
        self.Ey[:, 1:-1] = (
            self.Ca[:, 1:-1] * self.Ey[:, 1:-1] -
            self.Cb[:, 1:-1] * (self.Hz[:, 2:] - self.Hz[:, :-2]) / (2 * self.dx)
        )

    def update_H(self):
        """H field update"""
        # Hz = Hz + dt/mu0 * (dEx/dy - dEy/dx)
        self.Hz[1:-1, 1:-1] += (self.dt / mu0) * (
            (self.Ex[2:, 1:-1] - self.Ex[:-2, 1:-1]) / (2 * self.dy) -
            (self.Ey[1:-1, 2:] - self.Ey[1:-1, :-2]) / (2 * self.dx)
        )

    def step(self, source_func=None, source_pos=None):
        self.update_E()

        if source_func is not None and source_pos is not None:
            t = self.time_step * self.dt
            self.Hz[source_pos[1], source_pos[0]] += source_func(t)

        self.update_H()

        # PEC boundary
        self.Ex[0, :] = self.Ex[-1, :] = 0
        self.Ey[:, 0] = self.Ey[:, -1] = 0

        self.time_step += 1
```

---

## 5. PML (Perfectly Matched Layer)

### 5.1 PML Concept

```
PML (Berenger, 1994):

Key Idea:
- Artificial absorbing medium layer
- No reflection at medium boundary (impedance matching)
- Exponential decay inside the layer

Implementation Methods:
1. Split-field PML: Split fields into two components
2. UPML (Uniaxial PML): Anisotropic medium representation
3. CPML (Convolutional PML): Convolutional form

PML Parameters:
- Layer thickness: typically 8-20 cells
- Decay profile: polynomial or exponential
- Maximum conductivity: optimal value exists
```

```python
class FDTD_2D_TM_PML:
    """2D FDTD TM Mode with UPML"""

    def __init__(self, Nx=100, Ny=100, dx=1e-3, dy=1e-3, pml_layers=10, courant=0.9):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.pml_layers = pml_layers

        self.dt = courant / (c0 * np.sqrt(1/dx**2 + 1/dy**2))

        # Field arrays
        self.Ez = np.zeros((Ny, Nx))
        self.Hx = np.zeros((Ny, Nx))
        self.Hy = np.zeros((Ny, Nx))

        # PML auxiliary fields
        self.Psi_Ez_x = np.zeros((Ny, Nx))
        self.Psi_Ez_y = np.zeros((Ny, Nx))
        self.Psi_Hx_y = np.zeros((Ny, Nx))
        self.Psi_Hy_x = np.zeros((Ny, Nx))

        # Material properties
        self.eps_r = np.ones((Ny, Nx))
        self.sigma = np.zeros((Ny, Nx))

        # Initialize PML coefficients
        self._setup_pml()

        self.time_step = 0

        print(f"2D FDTD TM + PML Initialization:")
        print(f"  Grid: {Nx} x {Ny}, PML: {pml_layers} layers")

    def _pml_profile(self, d, d_max, sigma_max, order=3):
        """PML conductivity profile"""
        return sigma_max * (d / d_max) ** order

    def _setup_pml(self):
        """Setup PML coefficients"""
        # Optimal conductivity
        sigma_max = 0.8 * (order + 1) / (eta0 * self.dx) if hasattr(self, 'order') else \
                   0.8 * 4 / (eta0 * self.dx)
        order = 3

        # x-direction PML coefficients
        self.sigma_x = np.zeros(self.Nx)
        self.sigma_x_star = np.zeros(self.Nx)  # dual grid

        for i in range(self.pml_layers):
            # Left PML
            d = self.pml_layers - i
            self.sigma_x[i] = self._pml_profile(d, self.pml_layers, sigma_max, order)
            self.sigma_x_star[i] = self._pml_profile(d - 0.5, self.pml_layers, sigma_max, order)

            # Right PML
            d = i + 1
            self.sigma_x[-(i+1)] = self._pml_profile(d, self.pml_layers, sigma_max, order)
            self.sigma_x_star[-(i+1)] = self._pml_profile(d - 0.5, self.pml_layers, sigma_max, order)

        # y-direction PML coefficients
        self.sigma_y = np.zeros(self.Ny)
        self.sigma_y_star = np.zeros(self.Ny)

        for j in range(self.pml_layers):
            # Bottom PML
            d = self.pml_layers - j
            self.sigma_y[j] = self._pml_profile(d, self.pml_layers, sigma_max, order)
            self.sigma_y_star[j] = self._pml_profile(d - 0.5, self.pml_layers, sigma_max, order)

            # Top PML
            d = j + 1
            self.sigma_y[-(j+1)] = self._pml_profile(d, self.pml_layers, sigma_max, order)
            self.sigma_y_star[-(j+1)] = self._pml_profile(d - 0.5, self.pml_layers, sigma_max, order)

        # Coefficient calculation
        self.b_x = np.exp(-self.sigma_x * self.dt / eps0)
        self.c_x = (1 - self.b_x) / (self.sigma_x * self.dx + 1e-10)
        self.b_x_star = np.exp(-self.sigma_x_star * self.dt / eps0)
        self.c_x_star = (1 - self.b_x_star) / (self.sigma_x_star * self.dx + 1e-10)

        self.b_y = np.exp(-self.sigma_y * self.dt / eps0)
        self.c_y = (1 - self.b_y) / (self.sigma_y * self.dy + 1e-10)
        self.b_y_star = np.exp(-self.sigma_y_star * self.dt / eps0)
        self.c_y_star = (1 - self.b_y_star) / (self.sigma_y_star * self.dy + 1e-10)

    def update_H(self):
        """H field update (with PML)"""
        # Hx update
        dEz_dy = (self.Ez[:, 1:] - self.Ez[:, :-1]) / self.dy
        self.Psi_Hx_y[:, :-1] = (self.b_y[:, np.newaxis] * self.Psi_Hx_y[:, :-1] +
                                self.c_y[:, np.newaxis] * dEz_dy)
        self.Hx[:, :-1] -= self.dt / mu0 * (dEz_dy + self.Psi_Hx_y[:, :-1])

        # Hy update
        dEz_dx = (self.Ez[1:, :] - self.Ez[:-1, :]) / self.dx
        self.Psi_Hy_x[:-1, :] = (self.b_x[np.newaxis, :] * self.Psi_Hy_x[:-1, :] +
                                self.c_x[np.newaxis, :] * dEz_dx)
        self.Hy[:-1, :] += self.dt / mu0 * (dEz_dx + self.Psi_Hy_x[:-1, :])

    def update_E(self):
        """E field update (with PML)"""
        eps = eps0 * self.eps_r

        dHy_dx = (self.Hy[1:-1, 1:-1] - self.Hy[:-2, 1:-1]) / self.dx
        dHx_dy = (self.Hx[1:-1, 1:-1] - self.Hx[1:-1, :-2]) / self.dy

        self.Psi_Ez_x[1:-1, 1:-1] = (self.b_x_star[np.newaxis, 1:-1] * self.Psi_Ez_x[1:-1, 1:-1] +
                                    self.c_x_star[np.newaxis, 1:-1] * dHy_dx)
        self.Psi_Ez_y[1:-1, 1:-1] = (self.b_y_star[1:-1, np.newaxis] * self.Psi_Ez_y[1:-1, 1:-1] +
                                    self.c_y_star[1:-1, np.newaxis] * dHx_dy)

        self.Ez[1:-1, 1:-1] += self.dt / eps[1:-1, 1:-1] * (
            dHy_dx + self.Psi_Ez_x[1:-1, 1:-1] -
            dHx_dy - self.Psi_Ez_y[1:-1, 1:-1]
        )

    def step(self, source_func=None, source_pos=None):
        self.update_H()

        if source_func is not None and source_pos is not None:
            t = self.time_step * self.dt
            self.Ez[source_pos[1], source_pos[0]] += source_func(t)

        self.update_E()

        # PEC outer boundary
        self.Ez[0, :] = self.Ez[-1, :] = 0
        self.Ez[:, 0] = self.Ez[:, -1] = 0

        self.time_step += 1

    def run(self, n_steps, source_func=None, source_pos=None, record_interval=1):
        Ez_history = []

        for n in range(n_steps):
            self.step(source_func, source_pos)

            if n % record_interval == 0:
                Ez_history.append(self.Ez.copy())

        return np.array(Ez_history)


def demo_pml():
    """PML demonstration"""

    # Without PML
    fdtd_no_pml = FDTD_2D_TM(Nx=100, Ny=100, dx=1e-3, dy=1e-3, courant=0.9)
    fdtd_no_pml.compute_coefficients()

    # With PML
    fdtd_pml = FDTD_2D_TM_PML(Nx=100, Ny=100, dx=1e-3, dy=1e-3, pml_layers=10, courant=0.9)

    source_pos = (50, 50)

    def source(t):
        return gaussian_pulse(t, t0=0.15e-9, tau=0.05e-9)

    n_steps = 200

    # Run
    Ez_no_pml = fdtd_no_pml.run(n_steps, source, source_pos, record_interval=10)
    Ez_pml = fdtd_pml.run(n_steps, source, source_pos, record_interval=10)

    # Comparison visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    x = np.arange(100) * 1e-3 * 1000
    y = np.arange(100) * 1e-3 * 1000
    X, Y = np.meshgrid(x, y)

    for col, snap_idx in enumerate([5, 10, 15, 19]):
        # PEC boundary (reflection)
        ax = axes[0, col]
        vmax = 0.3
        ax.pcolormesh(X, Y, Ez_no_pml[snap_idx], cmap='RdBu_r',
                     shading='auto', vmin=-vmax, vmax=vmax)
        ax.set_title(f'PEC Boundary, Step {snap_idx * 10}')
        ax.set_aspect('equal')
        if col == 0:
            ax.set_ylabel('Without PML\ny [mm]')

        # PML boundary (absorption)
        ax = axes[1, col]
        ax.pcolormesh(X, Y, Ez_pml[snap_idx], cmap='RdBu_r',
                     shading='auto', vmin=-vmax, vmax=vmax)

        # Mark PML region
        pml = 10
        ax.axvline(x=pml, color='green', linestyle='--', alpha=0.5)
        ax.axvline(x=100-pml, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=pml, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=100-pml, color='green', linestyle='--', alpha=0.5)

        ax.set_title(f'PML Boundary, Step {snap_idx * 10}')
        ax.set_aspect('equal')
        if col == 0:
            ax.set_ylabel('With PML\ny [mm]')
        ax.set_xlabel('x [mm]')

    plt.suptitle('PEC vs PML Boundary Condition Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_pml_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# demo_pml()
```

---

## 6. Application Example: Waveguide

### 6.1 Rectangular Waveguide Simulation

```python
def rectangular_waveguide():
    """Rectangular waveguide simulation"""

    # Waveguide dimensions (WR-90: a=22.86mm, b=10.16mm)
    # Using simplified dimensions
    a = 30  # Waveguide width (number of cells)
    b = 15  # Waveguide height (number of cells)

    Nx = 150
    Ny = 40
    dx = dy = 1e-3  # 1 mm

    fdtd = FDTD_2D_TM(Nx=Nx, Ny=Ny, dx=dx, dy=dy, courant=0.9)

    # Waveguide walls (PEC)
    wall_y1 = (Ny - b) // 2
    wall_y2 = wall_y1 + b

    # Top/bottom PEC walls
    fdtd.sigma[:wall_y1, :] = 1e7
    fdtd.sigma[wall_y2:, :] = 1e7

    fdtd.compute_coefficients()

    # TE10 mode excitation frequency
    f_c = c0 / (2 * a * dx)  # Cutoff frequency
    f_op = 1.5 * f_c  # Operating frequency

    print(f"Cutoff frequency: {f_c/1e9:.2f} GHz")
    print(f"Operating frequency: {f_op/1e9:.2f} GHz")

    # Source (inside waveguide)
    source_x = 10
    source_y = Ny // 2

    def source(t):
        t0 = 0.2e-9
        tau = 0.05e-9
        return np.sin(2 * np.pi * f_op * t) * (1 - np.exp(-((t - t0)/tau)**2) if t < t0 else 1)

    # Simulation
    n_steps = 500
    Ez_history = []

    for n in range(n_steps):
        fdtd.step(source, (source_x, source_y))

        if n % 5 == 0:
            Ez_history.append(fdtd.Ez.copy())

    Ez_history = np.array(Ez_history)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    x = np.arange(Nx) * dx * 1000
    y = np.arange(Ny) * dy * 1000
    X, Y = np.meshgrid(x, y)

    for idx, snap_idx in enumerate([10, 30, 50, 70, 90]):
        ax = axes[idx // 3, idx % 3]
        vmax = np.max(np.abs(Ez_history[snap_idx])) * 0.5
        if vmax == 0:
            vmax = 1

        im = ax.pcolormesh(X, Y, Ez_history[snap_idx], cmap='RdBu_r',
                          shading='auto', vmin=-vmax, vmax=vmax)

        # Mark waveguide walls
        ax.axhline(y=wall_y1 * dy * 1000, color='black', linewidth=2)
        ax.axhline(y=wall_y2 * dy * 1000, color='black', linewidth=2)

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'Step {snap_idx * 5}')
        plt.colorbar(im, ax=ax)

    # Waveform analysis
    ax = axes[1, 2]
    probe_y = Ny // 2
    probe_x_list = [30, 60, 90, 120]

    for px in probe_x_list:
        signal = Ez_history[:, probe_y, px]
        t = np.arange(len(signal)) * 5 * fdtd.dt * 1e9
        ax.plot(t, signal, label=f'x={px*dx*1000:.0f}mm')

    ax.set_xlabel('t [ns]')
    ax.set_ylabel('Ez')
    ax.set_title('Field at Various Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Rectangular Waveguide (f_op = {f_op/1e9:.1f} GHz > f_c = {f_c/1e9:.1f} GHz)', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_waveguide.png', dpi=150, bbox_inches='tight')
    plt.show()

# rectangular_waveguide()
```

---

## 7. Practice Problems

### Exercise 1: Source Comparison
Compare 1D FDTD results using Gaussian pulse and Ricker wavelet as sources. Analyze frequency response characteristics.

### Exercise 2: ABC Performance
Compare reflection coefficients of 1st order Mur ABC and 2nd order Mur ABC. Analyze performance as a function of incidence angle.

### Exercise 3: PML Optimization
Compare absorption performance for different PML layer thicknesses (5, 10, 15, 20) and polynomial orders (2, 3, 4).

### Exercise 4: Waveguide Modes
Simulate waveguide propagation below and above the TE10 cutoff frequency and analyze the differences.

---

## 8. References

### Key Papers
- Yee (1966) "Numerical Solution of Initial Boundary Value Problems..."
- Mur (1981) "Absorbing Boundary Conditions for the Finite-Difference Approximation..."
- Berenger (1994) "A Perfectly Matched Layer for the Absorption of Electromagnetic Waves"

### Textbooks
- Taflove & Hagness, "Computational Electrodynamics: The FDTD Method"
- Sullivan, "Electromagnetic Simulation Using the FDTD Method"

### Open Source
- MEEP (MIT): Python/C++ FDTD
- gprMax: Ground Penetrating Radar FDTD
- OpenEMS: FDTD + Circuit Simulation

---

## Summary

```
FDTD Implementation Key Points:

1. Algorithm Structure:
   - H update -> Source injection -> E update -> ABC

2. Source Types:
   - Hard: Forced setting (causes reflection)
   - Soft: Addition (+=) (reflected wave passes through)
   - TF/SF: Accurate plane wave

3. Absorbing Boundary Conditions:
   - Simple ABC: Simplest, exact only at S=1
   - Mur ABC: Improved performance, effective for normal incidence
   - PML: Best performance, complex implementation

4. 2D Modes:
   - TM: Ez, Hx, Hy (z polarization)
   - TE: Hz, Ex, Ey (z polarization)

5. PML Elements:
   - Layer thickness: 8-20 cells
   - Polynomial profile (order 3-4)
   - CPML is most effective

6. Numerical Considerations:
   - Courant condition compliance
   - 10-20 cells per wavelength
   - Minimize numerical dispersion
```

---

## Exercises

### Exercise 1: Hard Source vs Soft Source Reflected Field
Using the FDTD_1D class, set up a 1D domain with a PEC reflector at x = 150 mm. Launch a Gaussian pulse from x = 50 mm using both a hard source and a soft source. After the pulse has reflected and returned to the source region, compare the field at the source position for both cases. Explain why the hard source produces an artifact at late times that the soft source does not.

### Exercise 2: Mur ABC Angle Dependence
Implement a 2D FDTD TM simulation with a point source at the center. Replace the PEC boundary on three sides with Mur first-order ABC, and keep PEC on the fourth side. Visually compare how well the ABC absorbs waves arriving at normal incidence versus those arriving at 45°. Estimate the reflection coefficient for each angle by recording the maximum field amplitude at probe points 5 cells from each boundary before and after the pulse arrives.

### Exercise 3: PML Thickness Optimization
Using the FDTD_2D_TM_PML class, run simulations with PML layer counts of 4, 8, 12, and 16 cells. For each, place a probe at the center and measure the maximum spurious reflected field amplitude after the main pulse has left the domain. Plot the reflection level (in dB) versus PML thickness and determine the minimum thickness needed to achieve -40 dB reflection. How does PML order (2, 3, 4) affect this result?

### Exercise 4: Waveguide Mode Cutoff
Using the rectangular waveguide simulation code, set the operating frequency to (a) 0.8×fc, (b) fc, and (c) 1.2×fc where fc is the TE10 cutoff frequency. For each case, plot the Ez field snapshot at n = 300 steps. In cases (a) and (b), observe exponential decay or no propagation; in case (c), observe guided wave propagation. Measure the guide wavelength λg from the standing wave pattern and compare to the theoretical value λg = λ/√(1-(fc/f)²).

### Exercise 5: Gaussian Pulse Spectral Analysis
Run the 1D FDTD with a Gaussian pulse (t0 = 1×10⁻¹⁰ s, tau = 3×10⁻¹¹ s). Record the time series at two probe points: one near the source and one at the far end of the domain (before and after a dielectric slab). Compute the FFT of each recorded time series and plot the magnitude spectrum. Identify: (a) the -3 dB bandwidth of the source spectrum, (b) frequency-dependent attenuation caused by the dielectric slab, and (c) the phase shift introduced by propagation through the slab.

---

The next lesson covers the basics of MHD (Magnetohydrodynamics).
