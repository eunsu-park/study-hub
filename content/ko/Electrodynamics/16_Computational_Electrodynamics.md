# 16. 전산 전자기학 (FDTD)

[← 이전: 15. 다중극 전개](15_Multipole_Expansion.md) | [다음: 17. 전자기 산란 →](17_Electromagnetic_Scattering.md)

## 학습 목표

1. FDTD(시간 영역 유한 차분법, Finite-Difference Time-Domain) 방법을 맥스웰 방정식의 직접 이산화로 이해한다
2. Yee 격자(엇갈린 격자, Staggered Grid)를 구성하고, 이 격자가 발산 조건을 자연스럽게 보존하는 이유를 설명한다
3. 도약 시간 전진법(Leapfrog Time-Stepping)을 구현하고, 쿠란트 안정성 조건(Courant Stability Condition)을 유도한다
4. 소스 주입 및 흡수 경계 조건을 갖춘 완전한 1D FDTD 시뮬레이션을 구축한다
5. 방법을 2D로 확장하고, 완전 정합층(PML, Perfectly Matched Layer) 흡수 경계 조건을 이해한다
6. FDTD 결과를 파동 전파에 대한 해석해와 비교하여 검증한다
7. FDTD를 실용적인 문제에 적용한다: 펄스 전파, 산란, 도파관 시뮬레이션

맥스웰 방정식은 우아하지만, 이상화된 소수의 기하학적 형태에서만 해석적으로 풀 수 있는 연립 편미분 방정식이다. 복잡한 형태의 안테나, 불규칙한 물체에 의한 산란, 불균일 매질에서의 파동과 같은 현실 문제에는 수치 방법이 필요하다. 시간 영역 유한 차분법(FDTD)은 가장 직관적이고 널리 사용되는 접근법이다. FDTD는 맥스웰의 회전 방정식을 엇갈린 격자에서 직접 이산화하여, 시간 단계별로 장을 전진시킨다. 이 레슨에서는 1D부터 시작해 2D로 확장하면서 안정성, 정확도, 경계 조건에 대해 배우며, 처음부터 FDTD 솔버를 직접 구현한다.

> **비유**: 격자 위에 수위 센서를 배치해 연못을 시뮬레이션하는 것을 상상해 보자. 매 시간 단계마다 인접 센서 간의 차이(유한 차분)를 기반으로 각 센서의 값을 갱신한다. 센서 간격이 충분히 좁고 시간 단계가 충분히 작다면, 이산 갱신이 연속적인 파동 물리학을 충실히 재현한다. FDTD는 전자기장에 대해 정확히 이것을 수행하지만, $\mathbf{E}$와 $\mathbf{B}$가 Yee 격자(Yee Lattice) 위에서 약간 엇갈린 격자점에 정의된다는 추가적인 묘미가 있다.

---

## 1. 맥스웰 방정식에서 유한 차분으로

### 1.1 맥스웰 회전 방정식

소스가 없는 선형 매질에서:

$$\frac{\partial \mathbf{B}}{\partial t} = -\nabla \times \mathbf{E}$$

$$\frac{\partial \mathbf{D}}{\partial t} = \nabla \times \mathbf{H}$$

$E_z$와 $H_y$ 성분을 가지며 $x$ 방향으로 전파하는 1D 파동의 경우:

$$\frac{\partial E_z}{\partial t} = \frac{1}{\epsilon}\frac{\partial H_y}{\partial x}$$

$$\frac{\partial H_y}{\partial t} = \frac{1}{\mu}\frac{\partial E_z}{\partial x}$$

### 1.2 유한 차분 근사

도함수에 대한 중심 차분(Central Difference) 근사:

$$\frac{\partial f}{\partial x}\bigg|_{x_i} \approx \frac{f(x_i + \Delta x/2) - f(x_i - \Delta x/2)}{\Delta x} + O(\Delta x^2)$$

이는 **2차 정확도(Second-Order Accurate)**를 가진다 — 오차가 격자 간격의 제곱에 비례하여 감소한다.

### 1.3 Yee 격자 (엇갈린 격자)

Kane Yee의 탁월한 통찰(1966)은 $\mathbf{E}$와 $\mathbf{H}$ 장 성분을 공간과 시간 모두에서 엇갈리게 배치하는 것이었다:

```
1D Yee Grid:

     E_z[0]        E_z[1]        E_z[2]        E_z[3]
      |              |              |              |
      |----H_y[0]----|----H_y[1]----|----H_y[2]----|
      |              |              |              |
     x=0           x=dx          x=2dx          x=3dx
```

- $E_z$는 정수 격자점에 정의된다: $E_z^n[i]$는 $(x = i\Delta x, t = n\Delta t)$에 위치
- $H_y$는 반정수 격자점에 정의된다: $H_y^{n+1/2}[i+1/2]$는 $(x = (i+1/2)\Delta x, t = (n+1/2)\Delta t)$에 위치

이 엇갈림 배치는 중심 차분이 올바른 위치에서 자연스럽게 회전(Curl)을 근사하도록 보장한다.

---

## 2. 도약 알고리즘

### 2.1 갱신 방정식 (1D)

이산화된 맥스웰 방정식은 다음과 같다:

$$H_y^{n+1/2}[i+\tfrac{1}{2}] = H_y^{n-1/2}[i+\tfrac{1}{2}] + \frac{\Delta t}{\mu \Delta x}\left(E_z^n[i+1] - E_z^n[i]\right)$$

$$E_z^{n+1}[i] = E_z^n[i] + \frac{\Delta t}{\epsilon \Delta x}\left(H_y^{n+1/2}[i+\tfrac{1}{2}] - H_y^{n+1/2}[i-\tfrac{1}{2}]\right)$$

$E$ 와 $H$ 장은 교대로 갱신된다 — $H$는 항상 $E$보다 반 시간 단계 앞서 있어, "도약(Leapfrog)"이라는 이름이 붙었다.

### 2.2 쿠란트-프리드리히스-레비 (CFL) 조건

도약법은 **조건부 안정(Conditionally Stable)**이다. 시간 단계는 다음 조건을 만족해야 한다:

$$\boxed{\Delta t \leq \frac{\Delta x}{c \sqrt{d}}}$$

여기서 $d$는 공간 차원 수(1, 2, 또는 3)이고, $c$는 영역 내 최대 파동 속도이다.

**쿠란트 수(Courant Number)**는 $S = c \Delta t / \Delta x$이다. 안정성을 위해 $S \leq 1/\sqrt{d}$이어야 한다.

실용적으로는, 수치 분산(Numerical Dispersion)을 줄이고 안전 마진을 확보하기 위해 $S = 0.5$를 사용한다.

```python
import numpy as np
import matplotlib.pyplot as plt

class FDTD_1D:
    """
    1D FDTD simulation of electromagnetic wave propagation.

    Why build from scratch: understanding the update equations,
    stability conditions, and boundary treatments gives physical
    insight that a black-box solver cannot provide.
    """

    def __init__(self, nx=500, dx=1e-3, courant=0.5):
        """
        Initialize the 1D FDTD domain.

        Parameters:
            nx     : number of spatial cells
            dx     : cell size (m)
            courant: Courant number S = c*dt/dx (must be <= 1)
        """
        self.c = 3e8              # speed of light (m/s)
        self.eps_0 = 8.854e-12    # vacuum permittivity
        self.mu_0 = 4*np.pi*1e-7  # vacuum permeability

        self.nx = nx
        self.dx = dx
        self.dt = courant * dx / self.c
        self.courant = courant

        # Field arrays
        self.Ez = np.zeros(nx)         # E_z at integer points
        self.Hy = np.zeros(nx - 1)     # H_y at half-integer points

        # Material arrays (relative permittivity and permeability)
        self.eps_r = np.ones(nx)
        self.mu_r = np.ones(nx - 1)

        # Update coefficients
        self.cE = self.dt / (self.eps_0 * self.dx)  # for E update
        self.cH = self.dt / (self.mu_0 * self.dx)   # for H update

        self.time_step = 0

    def set_material(self, start_idx, end_idx, eps_r=1.0, mu_r=1.0):
        """Set material properties in a region."""
        self.eps_r[start_idx:end_idx] = eps_r
        if end_idx < self.nx:
            self.mu_r[start_idx:end_idx] = mu_r

    def add_gaussian_source(self, source_idx, t0, spread):
        """Inject a soft Gaussian pulse source at a given location."""
        t = self.time_step * self.dt
        self.Ez[source_idx] += np.exp(-0.5 * ((t - t0) / spread)**2)

    def add_sinusoidal_source(self, source_idx, frequency, amplitude=1.0):
        """Inject a continuous sinusoidal source."""
        t = self.time_step * self.dt
        self.Ez[source_idx] += amplitude * np.sin(2 * np.pi * frequency * t)

    def step(self):
        """
        Advance the simulation by one time step using leapfrog.

        The order matters: H is updated first (from n-1/2 to n+1/2),
        then E is updated (from n to n+1). This preserves the
        time-centering of the central differences.
        """
        # Update H_y (uses E_z at time n)
        self.Hy += (self.cH / self.mu_r) * (self.Ez[1:] - self.Ez[:-1])

        # Update E_z (uses H_y at time n+1/2)
        self.Ez[1:-1] += (self.cE / self.eps_r[1:-1]) * (self.Hy[1:] - self.Hy[:-1])

        # Simple absorbing boundary conditions (Mur first-order)
        # Left boundary
        self.Ez[0] = self.Ez[1]
        # Right boundary
        self.Ez[-1] = self.Ez[-2]

        self.time_step += 1

    def run(self, n_steps, source_idx=None, source_type='gaussian',
            frequency=None, snapshots=None):
        """
        Run the simulation for n_steps.

        Returns field snapshots at specified time steps.
        """
        if snapshots is None:
            snapshots = [n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]

        saved = {}
        t0 = 20 * self.dt * (self.nx // 10)  # delay for Gaussian
        spread = 5 * self.dt * (self.nx // 10)

        for n in range(n_steps):
            if source_idx is not None:
                if source_type == 'gaussian':
                    self.add_gaussian_source(source_idx, t0, spread)
                elif source_type == 'sinusoidal' and frequency is not None:
                    self.add_sinusoidal_source(source_idx, frequency)

            self.step()

            if n in snapshots:
                saved[n] = self.Ez.copy()

        return saved


def demo_1d_fdtd():
    """Demonstrate basic 1D FDTD: Gaussian pulse in free space."""
    sim = FDTD_1D(nx=500, dx=1e-3, courant=0.5)

    # Source at the left quarter of the domain
    source_idx = 100
    n_steps = 600
    snapshots = [100, 200, 300, 500]

    saved = sim.run(n_steps, source_idx=source_idx, source_type='gaussian',
                    snapshots=snapshots)

    x = np.arange(sim.nx) * sim.dx * 1e3  # in mm

    fig, axes = plt.subplots(len(snapshots), 1, figsize=(12, 10), sharex=True)
    for ax, step_n in zip(axes, snapshots):
        ax.plot(x, saved[step_n], 'b-', linewidth=1.5)
        ax.axvline(x=source_idx * sim.dx * 1e3, color='r', linestyle='--',
                   alpha=0.5, label='Source')
        ax.set_ylabel('$E_z$')
        t_ns = step_n * sim.dt * 1e9
        ax.set_title(f'Step {step_n} (t = {t_ns:.2f} ns)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Position (mm)')
    plt.suptitle('1D FDTD: Gaussian Pulse Propagation', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("fdtd_1d_gaussian.png", dpi=150)
    plt.show()

demo_1d_fdtd()
```

---

## 3. 흡수 경계 조건

### 3.1 문제점

FDTD는 유한한 영역을 시뮬레이션하지만, 전자기파는 본래 무한히 전파된다. 격자를 단순히 끊으면 파동이 경계에서 반사되어 해를 오염시킨다.

### 3.2 Mur 흡수 경계 조건

가장 단순한 흡수 경계 조건(ABC, Absorbing Boundary Condition)은 1차 Mur 조건으로, 경계에서의 외향 파동을 근사한다:

$$E_z^{n+1}[0] = E_z^n[1] + \frac{S - 1}{S + 1}\left(E_z^{n+1}[1] - E_z^n[0]\right)$$

여기서 $S$는 쿠란트 수이다. $S = 1$이면 $E_z^{n+1}[0] = E_z^n[1]$로 단순화된다 — 경계 값이 단순히 이웃 격자점의 이전 값이 된다.

### 3.3 완전 정합층 (PML)

Berenger(1994)가 고안한 PML은 경계에 손실성 층을 도입해, 주파수나 입사각에 무관하게 **반사 없이** 입사파를 흡수한다. 이는 자유 공간에 "정합된" 인위적인 전도율(Conductivity)을 도입함으로써 작동한다:

$$\sigma_x(x) = \sigma_{\max}\left(\frac{x}{d_{\text{PML}}}\right)^p$$

여기서 전도율은 PML 내부 경계의 0에서 외부 경계의 $\sigma_{\max}$까지 다항식으로 증가한다. 일반적인 파라미터: $p = 3$, PML 두께 = 10~20개 셀.

```python
class FDTD_1D_PML(FDTD_1D):
    """
    1D FDTD with PML absorbing boundaries.

    Why PML: Mur's ABC works only for normal incidence in 1D and
    degrades for oblique incidence in 2D/3D. PML absorbs waves at
    all angles and frequencies, making it the standard for production
    FDTD codes.
    """

    def __init__(self, nx=500, dx=1e-3, courant=0.5, pml_cells=20):
        super().__init__(nx=nx + 2 * pml_cells, dx=dx, courant=courant)
        self.pml_cells = pml_cells
        self.inner_nx = nx

        # PML conductivity profile (polynomial grading)
        sigma_max = 0.8 * (3 + 1) / (self.dx * np.sqrt(self.mu_0 / self.eps_0))

        # Left PML region
        for i in range(pml_cells):
            depth = (pml_cells - i) / pml_cells
            sigma = sigma_max * depth**3
            self.eps_r[i] = 1.0  # still vacuum permittivity
            # Modify update coefficients to include loss
            # We store the loss factor separately
            pass  # simplified: use exponential decay approach below

        # Use a simpler convolutional PML approach
        self.psi_Ez_left = np.zeros(pml_cells)
        self.psi_Ez_right = np.zeros(pml_cells)
        self.psi_Hy_left = np.zeros(pml_cells)
        self.psi_Hy_right = np.zeros(pml_cells)

        # PML parameters
        sigma = np.zeros(pml_cells)
        for i in range(pml_cells):
            sigma[i] = sigma_max * ((pml_cells - i) / pml_cells)**3

        self.b_pml = np.exp(-sigma * self.dt / self.eps_0)
        self.c_pml = (self.b_pml - 1) * sigma / (sigma**2 + 1e-30) * self.eps_0

    def step(self):
        """Advance one step with PML boundaries."""
        # Standard leapfrog update
        self.Hy += (self.cH / self.mu_r) * (self.Ez[1:] - self.Ez[:-1])
        self.Ez[1:-1] += (self.cE / self.eps_r[1:-1]) * (self.Hy[1:] - self.Hy[:-1])

        # Apply PML damping at boundaries
        pml = self.pml_cells
        decay = 0.98  # simple exponential decay per step in PML

        for i in range(pml):
            factor = decay ** (pml - i)
            self.Ez[i] *= factor
            self.Ez[-(i+1)] *= factor

        self.time_step += 1


def compare_boundaries():
    """Compare Mur ABC vs PML absorbing boundaries."""

    # Simulation with Mur ABC
    sim_mur = FDTD_1D(nx=400, dx=1e-3, courant=0.5)
    saved_mur = sim_mur.run(800, source_idx=200, source_type='gaussian',
                            snapshots=[300, 500, 700])

    # Simulation with PML
    sim_pml = FDTD_1D_PML(nx=400, dx=1e-3, courant=0.5, pml_cells=20)
    saved_pml = sim_pml.run(800, source_idx=220, source_type='gaussian',
                            snapshots=[300, 500, 700])

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex='col')

    for row, step_n in enumerate([300, 500, 700]):
        x_mur = np.arange(sim_mur.nx) * sim_mur.dx * 1e3
        x_pml = np.arange(sim_pml.nx) * sim_pml.dx * 1e3

        axes[row, 0].plot(x_mur, saved_mur[step_n], 'b-', linewidth=1.5)
        axes[row, 0].set_ylabel('$E_z$')
        axes[row, 0].set_title(f'Mur ABC (step {step_n})')
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(x_pml, saved_pml[step_n], 'r-', linewidth=1.5)
        axes[row, 1].set_title(f'PML (step {step_n})')
        axes[row, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Position (mm)')
    axes[-1, 1].set_xlabel('Position (mm)')
    plt.suptitle('Boundary Comparison: Mur ABC vs PML', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("fdtd_boundary_comparison.png", dpi=150)
    plt.show()

compare_boundaries()
```

---

## 4. 2D FDTD

### 4.1 2D에서의 TE 및 TM 모드

2D($xy$ 평면에서의 파동)에서 맥스웰 방정식은 두 개의 독립적인 집합으로 분리된다:

**TM 모드** ($E_z$, $H_x$, $H_y$):

$$\frac{\partial H_x}{\partial t} = -\frac{1}{\mu}\frac{\partial E_z}{\partial y}$$

$$\frac{\partial H_y}{\partial t} = \frac{1}{\mu}\frac{\partial E_z}{\partial x}$$

$$\frac{\partial E_z}{\partial t} = \frac{1}{\epsilon}\left(\frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y}\right)$$

**TE 모드** ($H_z$, $E_x$, $E_y$):

$$\frac{\partial E_x}{\partial t} = \frac{1}{\epsilon}\frac{\partial H_z}{\partial y}$$

$$\frac{\partial E_y}{\partial t} = -\frac{1}{\epsilon}\frac{\partial H_z}{\partial x}$$

$$\frac{\partial H_z}{\partial t} = \frac{1}{\mu}\left(\frac{\partial E_x}{\partial y} - \frac{\partial E_y}{\partial x}\right)$$

### 4.2 2D Yee 격자

```
2D Yee Grid (TM mode):

    Hy---Ez---Hy---Ez
    |         |
    Hx        Hx
    |         |
    Hy---Ez---Hy---Ez
    |         |
    Hx        Hx
    |         |
    Hy---Ez---Hy---Ez
```

$E_z$는 셀 중심에, $H_x$는 셀 가장자리(수평), $H_y$는 셀 가장자리(수직)에 위치한다.

### 4.3 2D에서의 쿠란트 조건

$$\Delta t \leq \frac{1}{c\sqrt{1/\Delta x^2 + 1/\Delta y^2}}$$

정사각 격자($\Delta x = \Delta y = \Delta$)의 경우: $\Delta t \leq \Delta/(c\sqrt{2})$.

```python
class FDTD_2D:
    """
    2D FDTD simulation (TM mode: Ez, Hx, Hy).

    Why 2D: it captures diffraction, scattering, and interference
    effects that are absent in 1D, while remaining computationally
    tractable for interactive exploration.
    """

    def __init__(self, nx=200, ny=200, dx=1e-3, courant=0.5):
        self.c = 3e8
        self.eps_0 = 8.854e-12
        self.mu_0 = 4 * np.pi * 1e-7

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx  # square grid
        self.dt = courant * dx / (self.c * np.sqrt(2))

        # Field arrays (TM mode)
        self.Ez = np.zeros((nx, ny))
        self.Hx = np.zeros((nx, ny - 1))  # half-grid in y
        self.Hy = np.zeros((nx - 1, ny))  # half-grid in x

        # Material properties
        self.eps_r = np.ones((nx, ny))

        # Update coefficients
        self.cE = self.dt / (self.eps_0 * self.dx)
        self.cH = self.dt / (self.mu_0 * self.dx)

        self.time_step = 0

    def set_material_circle(self, cx, cy, radius, eps_r):
        """Set material properties in a circular region."""
        for i in range(self.nx):
            for j in range(self.ny):
                if (i - cx)**2 + (j - cy)**2 < radius**2:
                    self.eps_r[i, j] = eps_r

    def step(self):
        """Advance one time step (leapfrog)."""
        # Update Hx
        self.Hx -= self.cH * (self.Ez[:, 1:] - self.Ez[:, :-1])

        # Update Hy
        self.Hy += self.cH * (self.Ez[1:, :] - self.Ez[:-1, :])

        # Update Ez
        dHy_dx = self.Hy[1:, :] - self.Hy[:-1, :]
        dHx_dy = self.Hx[:, 1:] - self.Hx[:, :-1]

        # Interior update (avoiding boundaries)
        self.Ez[1:-1, 1:-1] += (self.cE / self.eps_r[1:-1, 1:-1]) * \
            (dHy_dx[:, 1:-1] - dHx_dy[1:-1, :])

        # Simple absorbing boundaries
        self.Ez[0, :] = 0
        self.Ez[-1, :] = 0
        self.Ez[:, 0] = 0
        self.Ez[:, -1] = 0

        self.time_step += 1

    def add_point_source(self, ix, iy, frequency=None, t0=None, spread=None):
        """Add a source at (ix, iy)."""
        t = self.time_step * self.dt
        if frequency is not None:
            self.Ez[ix, iy] += np.sin(2 * np.pi * frequency * t)
        elif t0 is not None and spread is not None:
            self.Ez[ix, iy] += np.exp(-0.5 * ((t - t0) / spread)**2)


def demo_2d_fdtd():
    """
    Demonstrate 2D FDTD: point source radiation and scattering
    from a dielectric cylinder.
    """
    sim = FDTD_2D(nx=200, ny=200, dx=0.5e-3, courant=0.5)

    # Add a dielectric cylinder (eps_r = 4) at center-right
    sim.set_material_circle(cx=130, cy=100, radius=20, eps_r=4.0)

    freq = 30e9  # 30 GHz

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    snapshot_steps = [200, 400, 600, 800]

    for n in range(max(snapshot_steps) + 1):
        sim.add_point_source(50, 100, frequency=freq)
        sim.step()

        if n in snapshot_steps:
            idx = snapshot_steps.index(n)
            ax = axes.flat[idx]

            extent = [0, sim.nx * sim.dx * 1e3, 0, sim.ny * sim.dy * 1e3]
            im = ax.imshow(sim.Ez.T, cmap='RdBu_r', origin='lower',
                          extent=extent, vmin=-0.5, vmax=0.5, aspect='equal')

            # Draw the dielectric cylinder
            circle = plt.Circle((130 * sim.dx * 1e3, 100 * sim.dy * 1e3),
                               20 * sim.dx * 1e3, fill=False, color='white',
                               linewidth=2, linestyle='--')
            ax.add_patch(circle)

            # Mark source
            ax.plot(50 * sim.dx * 1e3, 100 * sim.dy * 1e3, 'w*', markersize=10)

            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            t_ns = n * sim.dt * 1e9
            ax.set_title(f'Step {n} (t = {t_ns:.3f} ns)')
            plt.colorbar(im, ax=ax, label='$E_z$', shrink=0.8)

    plt.suptitle('2D FDTD: Point Source + Dielectric Cylinder Scattering',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("fdtd_2d_scattering.png", dpi=150)
    plt.show()

demo_2d_fdtd()
```

---

## 5. 해석해와의 검증

모든 수치 방법에서 중요한 과정은 **검증(Validation)**이다 — 수치 결과를 알려진 해석해와 비교하여 정확성을 확인하는 것이다.

```python
def validate_fdtd_wave_speed():
    """
    Validate FDTD by checking wave propagation speed.

    Why validate: numerical dispersion can cause the simulated wave
    speed to differ from the physical speed. Measuring this error
    is essential for understanding the accuracy of the simulation.
    """
    c = 3e8
    dx = 1e-3
    nx = 1000
    courant_values = [0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for S in courant_values:
        sim = FDTD_1D(nx=nx, dx=dx, courant=S)

        # Gaussian pulse source at center
        source_idx = nx // 4
        t0 = 40 * sim.dt * 10
        spread = 10 * sim.dt * 10

        # Run until pulse reaches 3/4 of the domain
        target_idx = 3 * nx // 4
        target_time = (target_idx - source_idx) * dx / c
        n_steps = int(target_time / sim.dt) + 100

        for n in range(n_steps):
            t = n * sim.dt
            sim.Ez[source_idx] += np.exp(-0.5 * ((t - t0) / spread)**2)
            sim.step()

        # Find pulse peak position
        peak_idx = np.argmax(np.abs(sim.Ez[source_idx:]))
        peak_pos = (source_idx + peak_idx) * dx

        # Expected position
        expected_pos = source_idx * dx + c * n_steps * sim.dt

        x = np.arange(nx) * dx * 1e3
        axes[0].plot(x, sim.Ez, linewidth=1.5, label=f'S = {S}')

    axes[0].set_xlabel('Position (mm)')
    axes[0].set_ylabel('$E_z$')
    axes[0].set_title('Pulse Shape at Same Physical Time (Different Courant Numbers)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Numerical dispersion analysis
    # For a 1D FDTD, the numerical dispersion relation is:
    # sin(omega*dt/2) / dt = c * sin(k*dx/2) / dx
    k_norm = np.linspace(0.01, np.pi, 200)  # k*dx from 0 to pi

    for S in courant_values:
        omega_exact = c * k_norm / dx
        sin_arg = S * np.sin(k_norm / 2)
        sin_arg = np.clip(sin_arg, -1, 1)
        omega_fdtd = 2 * np.arcsin(sin_arg) / (S * dx / c)

        v_phase_ratio = (omega_fdtd / (k_norm / dx)) / c
        axes[1].plot(k_norm / np.pi, v_phase_ratio,
                     linewidth=1.5, label=f'S = {S}')

    axes[1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Normalized wavenumber $k\\Delta x / \\pi$')
    axes[1].set_ylabel('$v_{\\phi}^{\\mathrm{FDTD}} / c$')
    axes[1].set_title('Numerical Dispersion: Phase Velocity Ratio')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.8, 1.05)

    plt.tight_layout()
    plt.savefig("fdtd_validation.png", dpi=150)
    plt.show()

validate_fdtd_wave_speed()
```

---

## 6. 실용적 고려 사항

### 6.1 격자 해상도

경험 법칙은 **파장당 최소 10~20개의 셀** ($\Delta x \leq \lambda / 20$)이다. 물질 경계 근처와 곡면 기하학에서는 더 세밀한 해상도가 필요하다.

### 6.2 수치 분산

FDTD 격자는 **수치 분산(Numerical Dispersion)**을 유발한다 — 시뮬레이션된 위상 속도(Phase Velocity)가 방향과 주파수에 따라 달라진다. 2D에서의 수치 분산 관계식은 다음과 같다:

$$\left(\frac{\sin(\omega\Delta t/2)}{c\Delta t}\right)^2 = \left(\frac{\sin(k_x\Delta x/2)}{\Delta x}\right)^2 + \left(\frac{\sin(k_y\Delta y/2)}{\Delta y}\right)^2$$

이는 격자 축에 대해 45도 방향으로 전파하는 파동이 축 방향 파동보다 약간 느리게 전파됨을 의미한다. 파장당 20개 이상의 셀을 사용하면 이 오차를 1% 미만으로 유지할 수 있다.

### 6.3 소스

- **하드 소스(Hard Source)**: $E_z[i] = f(t)$ — 장 값을 덮어써서 인위적인 반사를 유발한다
- **소프트 소스(Soft Source)**: $E_z[i] \mathrel{+}= f(t)$ — 장에 더하므로 파동에 투명하다
- **전체장/산란장(TF/SF, Total-Field/Scattered-Field)**: 입사파와 산란파를 깔끔하게 분리한다

### 6.4 확장 방법

| 방법 | 설명 | 적용 분야 |
|------|------|-----------|
| FDTD (기본) | 엇갈린 격자, 도약법 | 광대역(Broadband), 시간 영역 |
| ADI-FDTD | 교대 방향 음함수법(Alternating Direction Implicit) | 큰 시간 단계, 무조건 안정 |
| DGTD | 불연속 갤러킨 시간 영역(Discontinuous Galerkin Time Domain) | 복잡한 기하학, 고차 정확도 |
| FEM | 유한 요소법(Finite Element Method) | 주파수 영역, 비정형 메쉬 |
| MoM | 모멘트법(Method of Moments) | 표면 적분, 안테나 설계 |

---

## 요약

| 개념 | 핵심 내용 | 목적 |
|------|-----------|------|
| Yee 격자 | E와 H가 반 셀씩 엇갈림 | $\nabla \cdot \mathbf{B} = 0$을 자동으로 보존 |
| 도약법 | E와 H가 반 시간 단계씩 어긋남 | 2차 정확도, 양함수법(Explicit) |
| 쿠란트 조건 | $S = c\Delta t/\Delta x \leq 1/\sqrt{d}$ | 안정성 요건 |
| Mur ABC | 1차 흡수 경계 조건 | 단순하지만 수직 입사에만 유효 |
| PML | 기울기 흡수층 | 각도·주파수 무관 흡수 |
| 격자 해상도 | $\Delta x \leq \lambda/20$ | 정확도 제어 |
| 수치 분산 | $v_\phi$가 방향에 따라 달라짐 | 격자 고유의 인공 현상 |
| 소프트 소스 | $E_z \mathrel{+}= f(t)$ | 반사파에 투명 |

---

## 연습 문제

### 연습 1: 유전체 슬래브 반사
1D FDTD에서 유전체 슬래브($\epsilon_r = 4$, 두께 $= \lambda/4$)에 수직 입사하는 평면파를 시뮬레이션하라. 반사 및 투과 펄스 진폭을 측정하고, 해석적 프레넬 계수(Fresnel Coefficients)와 비교하라. 슬래브 두께가 $\lambda/2$일 때 결과는 어떻게 달라지는가?

### 연습 2: 쿠란트 수의 영향
$S = 0.1, 0.5, 0.9, 1.0$ 및 $1.1$의 쿠란트 수로 동일한 1D 펄스 전파 시뮬레이션을 실행하라. (a) 펄스 형태를 관찰하라 — 어떤 값에서 분산이 가장 적은가? (b) $S > 1$이면 어떻게 되는가? (c) 각 경우의 수치 분산 관계식을 그래프로 그려라.

### 연습 3: 2D 이중 슬릿
도체 벽의 두 슬릿을 통과하는 평면파의 2D FDTD 시뮬레이션을 구현하라. 슬릿 너비 $= 2\lambda$, 슬릿 간격 $= 5\lambda$로 설정하라. 회절 및 간섭 패턴을 관찰하라. 줄무늬 간격을 해석적 예측값 $\Delta y = \lambda L / d$ ($L$은 슬릿까지의 거리, $d$는 슬릿 간격)와 비교하라.

### 연습 4: 도파관 모드
2D FDTD에서 직사각형 도파관 단면($y = 0$과 $y = b$에 완전 도체(PEC) 경계 조건)을 시뮬레이션하라. 한쪽 끝에서 TE$_{10}$ 모드를 여기(Excite)시키고 전파 상수 $k_z$를 측정하라. 해석적 값 $k_z = \sqrt{k^2 - (\pi/b)^2}$와 비교하라.

### 연습 5: PML 최적화
$p = 1, 2, 3, 4$의 다항식 차수와 5, 10, 20, 40개 셀의 PML 두께를 가진 PML을 구현하라. 반사 펄스 진폭을 입사 펄스와 비교하여 PML의 반사 계수를 측정하라. 각 다항식 차수에 대해 PML 두께의 함수로 반사율을 그래프로 나타내고, 최적 설정을 결정하라.

---

[← 이전: 15. 다중극 전개](15_Multipole_Expansion.md) | [다음: 17. 전자기 산란 →](17_Electromagnetic_Scattering.md)
