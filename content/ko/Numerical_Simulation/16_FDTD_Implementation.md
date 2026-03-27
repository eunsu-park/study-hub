# 16. FDTD 구현 (FDTD Implementation)

## 학습 목표
- 1D FDTD의 완전한 구현
- 소스 여기 방법 (가우시안 펄스, 정현파)
- 흡수 경계조건 (Simple ABC, Mur ABC)
- 2D FDTD (TM, TE 모드)
- PML (Perfectly Matched Layer) 개념

**이 레슨이 중요한 이유:** FDTD(유한차분 시간영역법, Finite-Difference Time-Domain)는 전산전자기학의 핵심 방법입니다. 그 매력은 단순함에 있습니다: 엇갈린 Yee 격자에서의 리프프로그(Leapfrog) 시간 전진은 가우스 법칙을 자연스럽게 만족하며 시간과 공간 모두에서 2차 정확도를 제공합니다. 시간 영역에서 직접 동작하므로, FDTD는 단일 시뮬레이션 실행으로 광대역 소스, 비선형 재료, 복잡한 기하학을 처리할 수 있습니다.

---

## 1. 1D FDTD 완전 구현

### 1.1 기본 구조

```
1D FDTD 알고리즘:

초기화:
- 격자 설정 (Nx, dx, dt)
- 배열 초기화 (Ey, Hz)
- 물성치 설정 (ε, μ, σ)

시간 루프:
for n = 1, 2, ..., Nt:
    1. H 업데이트: Hz^(n+1/2) = f(Hz^(n-1/2), Ey^n)
    2. 소스 주입 (soft/hard)
    3. E 업데이트: Ey^(n+1) = f(Ey^n, Hz^(n+1/2))
    4. 경계조건 적용 (ABC)
    5. 데이터 기록/출력
```

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# 물리 상수
c0 = 299792458.0  # 광속 [m/s]
eps0 = 8.854187817e-12  # 진공 유전율 [F/m]
mu0 = 4 * np.pi * 1e-7  # 진공 투자율 [H/m]
eta0 = np.sqrt(mu0 / eps0)  # 진공 임피던스 [Ω]

class FDTD_1D:
    """1D FDTD 시뮬레이터"""

    def __init__(self, Nx=200, dx=1e-3, courant=0.99):
        """
        Parameters:
        - Nx: 격자점 수
        - dx: 공간 간격 [m]
        - courant: Courant 수 (≤ 1)
        """
        self.Nx = Nx
        self.dx = dx
        self.dt = courant * dx / c0

        # 필드 배열
        self.Ey = np.zeros(Nx)
        self.Hz = np.zeros(Nx)

        # 물성치 배열 (상대값)
        self.eps_r = np.ones(Nx)
        self.mu_r = np.ones(Nx)
        self.sigma = np.zeros(Nx)  # 전기 전도도
        self.sigma_m = np.zeros(Nx)  # 자기 전도도

        # ABC용 이전 필드값
        self.Ey_left_prev = [0, 0]
        self.Ey_right_prev = [0, 0]

        # 시간
        self.time_step = 0

        print(f"1D FDTD 초기화:")
        print(f"  Nx = {Nx}, dx = {dx*1000:.2f} mm")
        print(f"  dt = {self.dt*1e12:.3f} ps")
        print(f"  Courant S = {courant}")

    def set_material(self, start, end, eps_r=1, sigma=0):
        """재료 영역 설정"""
        self.eps_r[start:end] = eps_r
        self.sigma[start:end] = sigma

    def compute_coefficients(self):
        """업데이트 계수 계산"""
        eps = eps0 * self.eps_r
        mu = mu0 * self.mu_r

        # E 업데이트 계수 (손실 포함)
        self.Ca = (1 - self.sigma * self.dt / (2 * eps)) / \
                  (1 + self.sigma * self.dt / (2 * eps))
        self.Cb = (self.dt / (eps * self.dx)) / \
                  (1 + self.sigma * self.dt / (2 * eps))

        # H 업데이트 계수
        self.Da = (1 - self.sigma_m * self.dt / (2 * mu)) / \
                  (1 + self.sigma_m * self.dt / (2 * mu))
        self.Db = (self.dt / (mu * self.dx)) / \
                  (1 + self.sigma_m * self.dt / (2 * mu))

    def update_H(self):
        """H 필드 업데이트"""
        self.Hz[:-1] = (self.Da[:-1] * self.Hz[:-1] -
                       self.Db[:-1] * (self.Ey[1:] - self.Ey[:-1]))

    def update_E(self):
        """E 필드 업데이트"""
        self.Ey[1:-1] = (self.Ca[1:-1] * self.Ey[1:-1] -
                        self.Cb[1:-1] * (self.Hz[1:-1] - self.Hz[:-2]))

    def add_source_soft(self, position, value):
        """소프트 소스 (총 필드/산란 필드 경계)"""
        self.Ey[position] += value

    def add_source_hard(self, position, value):
        """하드 소스 (강제 주입)"""
        self.Ey[position] = value

    def apply_abc_simple(self):
        """간단한 흡수 경계조건 (1차)"""
        # 좌측 경계
        self.Ey[0] = self.Ey_left_prev[0]
        self.Ey_left_prev[0] = self.Ey_left_prev[1]
        self.Ey_left_prev[1] = self.Ey[1]

        # 우측 경계
        self.Ey[-1] = self.Ey_right_prev[0]
        self.Ey_right_prev[0] = self.Ey_right_prev[1]
        self.Ey_right_prev[1] = self.Ey[-2]

    def apply_abc_mur(self):
        """Mur 1차 흡수 경계조건"""
        coeff = (c0 * self.dt - self.dx) / (c0 * self.dt + self.dx)

        # 좌측 경계
        self.Ey[0] = self.Ey_left_prev[1] + coeff * (self.Ey[1] - self.Ey_left_prev[0])
        self.Ey_left_prev[0] = self.Ey[0]
        self.Ey_left_prev[1] = self.Ey[1]

        # 우측 경계
        self.Ey[-1] = self.Ey_right_prev[1] + coeff * (self.Ey[-2] - self.Ey_right_prev[0])
        self.Ey_right_prev[0] = self.Ey[-1]
        self.Ey_right_prev[1] = self.Ey[-2]

    def step(self, source_func=None, source_pos=None, abc_type='mur'):
        """한 시간 단계 전진"""
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
        """시뮬레이션 실행"""
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
    """가우시안 펄스 소스"""
    return np.exp(-((t - t0) / tau) ** 2)

def sinusoidal_source(t, freq=3e9, t0=5e-11, tau=2e-11):
    """변조된 정현파 소스"""
    envelope = 1 - np.exp(-((t - t0) / tau) ** 2) if t < t0 else 1
    return envelope * np.sin(2 * np.pi * freq * t)


def demo_1d_fdtd_basic():
    """1D FDTD 기본 시연"""

    # 시뮬레이터 생성
    fdtd = FDTD_1D(Nx=300, dx=1e-3, courant=0.99)

    # 유전체 슬래브 추가
    fdtd.set_material(150, 200, eps_r=4.0)

    # 시뮬레이션 실행
    source_pos = 50
    n_steps = 500

    Ey_history, Hz_history = fdtd.run(
        n_steps,
        source_func=gaussian_pulse,
        source_pos=source_pos,
        abc_type='mur',
        record_interval=5
    )

    # 결과 시각화
    x = np.arange(fdtd.Nx) * fdtd.dx * 1000  # mm

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 스냅샷
    snapshot_indices = [10, 30, 50, 70, 90]

    for idx, snap_idx in enumerate(snapshot_indices):
        if idx < 5:
            ax = axes[idx // 3, idx % 3]
            ax.plot(x, Ey_history[snap_idx], 'b-', linewidth=1.5)

            # 유전체 영역 표시
            ax.axvspan(150 * fdtd.dx * 1000, 200 * fdtd.dx * 1000,
                      alpha=0.2, color='green', label=r'$\epsilon_r=4$')
            ax.axvline(x=source_pos * fdtd.dx * 1000, color='red',
                      linestyle='--', alpha=0.5)

            ax.set_xlabel('x [mm]')
            ax.set_ylabel('Ey')
            ax.set_title(f'Step {snap_idx * 5}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.2, 1.2)

    # 마지막 subplot: 시공간 다이어그램
    ax = axes[1, 2]
    t = np.arange(len(Ey_history)) * 5 * fdtd.dt * 1e9  # ns

    im = ax.pcolormesh(x, t, Ey_history, cmap='RdBu_r', shading='auto',
                      vmin=-0.5, vmax=0.5)
    ax.axvline(x=150 * fdtd.dx * 1000, color='green', linestyle='-', linewidth=2)
    ax.axvline(x=200 * fdtd.dx * 1000, color='green', linestyle='-', linewidth=2)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('t [ns]')
    ax.set_title('시공간 다이어그램')
    plt.colorbar(im, ax=ax, label='Ey')

    plt.suptitle('1D FDTD: 유전체 슬래브에서의 반사와 투과', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_1d_dielectric.png', dpi=150, bbox_inches='tight')
    plt.show()

    return fdtd, Ey_history

# fdtd, Ey_history = demo_1d_fdtd_basic()
```

---

## 2. 소스 여기 방법

### 2.1 하드 소스 vs 소프트 소스

```
소스 유형:

1. 하드 소스 (Hard Source):
   Ey[source_pos] = source_value
   - 해당 점의 필드 강제 설정
   - 반사파가 소스에서 다시 반사됨
   - 간단하지만 비물리적 반사 발생

2. 소프트 소스 (Soft Source):
   Ey[source_pos] += source_value
   - 기존 필드에 소스 추가
   - 반사파가 소스를 통과
   - TF/SF 경계와 함께 사용

3. TF/SF (Total-Field/Scattered-Field):
   - 입사파와 산란파 분리
   - 정확한 평면파 주입
   - 추가 보정항 필요
```

```python
def source_comparison():
    """하드 소스와 소프트 소스 비교"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, source_type in enumerate(['hard', 'soft']):
        # 시뮬레이터 생성
        fdtd = FDTD_1D(Nx=200, dx=1e-3, courant=0.99)
        fdtd.compute_coefficients()

        source_pos = 50
        n_steps = 300

        # 반사체 (PEC) 추가
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

        # 스냅샷
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

    plt.suptitle('하드 소스 vs 소프트 소스 비교', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_source_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# source_comparison()
```

### 2.2 다양한 소스 파형

```python
def source_waveforms():
    """다양한 소스 파형"""

    t = np.linspace(0, 0.5e-9, 1000)  # 0.5 ns

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (1) 가우시안 펄스
    ax1 = axes[0, 0]
    t0, tau = 0.15e-9, 0.03e-9
    pulse = np.exp(-((t - t0) / tau) ** 2)
    ax1.plot(t * 1e9, pulse, 'b-', linewidth=2)
    ax1.set_title('가우시안 펄스')
    ax1.set_xlabel('t [ns]')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)

    # (2) 가우시안 미분 (Ricker wavelet)
    ax2 = axes[0, 1]
    ricker = -2 * (t - t0) / tau**2 * np.exp(-((t - t0) / tau) ** 2)
    ax2.plot(t * 1e9, ricker, 'r-', linewidth=2)
    ax2.set_title('가우시안 미분 (Ricker Wavelet)')
    ax2.set_xlabel('t [ns]')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)

    # (3) 변조 정현파
    ax3 = axes[1, 0]
    freq = 10e9  # 10 GHz
    modulated = np.sin(2 * np.pi * freq * t) * np.exp(-((t - t0) / tau) ** 2)
    ax3.plot(t * 1e9, modulated, 'g-', linewidth=1.5)
    ax3.set_title('변조 정현파 (10 GHz)')
    ax3.set_xlabel('t [ns]')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)

    # (4) 스펙트럼
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
    ax4.set_title('주파수 스펙트럼')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fdtd_source_waveforms.png', dpi=150, bbox_inches='tight')
    plt.show()

# source_waveforms()
```

---

## 3. 흡수 경계조건 (ABC)

### 3.1 Simple ABC

```
1차 Simple ABC:

1D 파동 방정식의 특성:
(∂/∂t + c ∂/∂x) Ey = 0  (우측 진행파)
(∂/∂t - c ∂/∂x) Ey = 0  (좌측 진행파)

이산화 (우측 경계, 좌측 진행파 흡수):
Ey^(n+1)[Nx-1] = Ey^n[Nx-2]

이것은 S = cΔt/Δx = 1 일 때만 정확
S ≠ 1 이면 반사 발생
```

### 3.2 Mur ABC

```
Mur 1차 ABC (1981):

1D 파동 방정식의 유한차분 근사:

우측 경계 (x = xmax):
(Ey^(n+1)[Nx-1] - Ey^n[Nx-2]) / (cΔt + Δx) =
(Ey^n[Nx-1] - Ey^(n+1)[Nx-2]) / (cΔt - Δx)

정리:
Ey^(n+1)[Nx-1] = Ey^n[Nx-2] +
                 (cΔt - Δx)/(cΔt + Δx) * (Ey^(n+1)[Nx-2] - Ey^n[Nx-1])

장점:
- Simple ABC보다 넓은 입사각에서 유효
- 구현 간단

한계:
- 수직 입사만 정확히 흡수
- 비스듬한 입사에서 반사 발생
```

```python
def abc_comparison():
    """흡수 경계조건 비교"""

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

    # 시각화
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

    plt.suptitle('흡수 경계조건 비교: PEC vs Simple ABC vs Mur ABC', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_abc_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 반사 분석
    fig, ax = plt.subplots(figsize=(10, 5))

    t = np.arange(len(results['pec'])) * 4 * 1e-3 / c0 * 1e9

    # 특정 위치에서의 필드 기록 (반사 검출용)
    probe_pos = 50

    for abc_type, color in [('pec', 'red'), ('simple', 'blue'), ('mur', 'green')]:
        field = results[abc_type][:, probe_pos]
        ax.plot(t, field, color=color, linewidth=1.5, label=abc_type.upper())

    ax.set_xlabel('t [ns]')
    ax.set_ylabel('Ey at probe')
    ax.set_title('경계에서의 반사 비교 (probe at x = 50 mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fdtd_abc_reflection.png', dpi=150, bbox_inches='tight')
    plt.show()

# abc_comparison()
```

---

## 4. 2D FDTD 구현

### 4.1 TM 모드 (Ez, Hx, Hy)

```python
class FDTD_2D_TM:
    """2D FDTD TM 모드 시뮬레이터"""

    def __init__(self, Nx=100, Ny=100, dx=1e-3, dy=1e-3, courant=0.99):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy

        # Courant 조건
        self.dt = courant / (c0 * np.sqrt(1/dx**2 + 1/dy**2))

        # 필드 배열
        self.Ez = np.zeros((Ny, Nx))
        self.Hx = np.zeros((Ny, Nx))
        self.Hy = np.zeros((Ny, Nx))

        # 물성치 배열
        self.eps_r = np.ones((Ny, Nx))
        self.mu_r = np.ones((Ny, Nx))
        self.sigma = np.zeros((Ny, Nx))

        # 계수
        self.Ca = None
        self.Cb = None

        self.time_step = 0

        print(f"2D FDTD TM 모드 초기화:")
        print(f"  격자: {Nx} x {Ny}")
        print(f"  dx = {dx*1000:.2f} mm, dy = {dy*1000:.2f} mm")
        print(f"  dt = {self.dt*1e12:.3f} ps")

    def set_material_region(self, x1, x2, y1, y2, eps_r=1, sigma=0):
        """재료 영역 설정"""
        self.eps_r[y1:y2, x1:x2] = eps_r
        self.sigma[y1:y2, x1:x2] = sigma

    def add_pec_circle(self, cx, cy, radius):
        """원형 PEC 추가"""
        for j in range(self.Ny):
            for i in range(self.Nx):
                if (i - cx)**2 + (j - cy)**2 < radius**2:
                    self.sigma[j, i] = 1e7

    def compute_coefficients(self):
        """업데이트 계수 계산"""
        eps = eps0 * self.eps_r

        self.Ca = (1 - self.sigma * self.dt / (2 * eps)) / \
                  (1 + self.sigma * self.dt / (2 * eps))
        self.Cb = (self.dt / eps) / (1 + self.sigma * self.dt / (2 * eps))

    def update_H(self):
        """H 필드 업데이트"""
        # Hx 업데이트: Hx = Hx - dt/μ₀ * dEz/dy
        self.Hx[:, :-1] -= (self.dt / (mu0 * self.dy)) * \
                          (self.Ez[:, 1:] - self.Ez[:, :-1])

        # Hy 업데이트: Hy = Hy + dt/μ₀ * dEz/dx
        self.Hy[:-1, :] += (self.dt / (mu0 * self.dx)) * \
                          (self.Ez[1:, :] - self.Ez[:-1, :])

    def update_E(self):
        """E 필드 업데이트"""
        # Ez 업데이트: Ez = Ca*Ez + Cb*(dHy/dx - dHx/dy)
        self.Ez[1:-1, 1:-1] = (
            self.Ca[1:-1, 1:-1] * self.Ez[1:-1, 1:-1] +
            self.Cb[1:-1, 1:-1] * (
                (self.Hy[1:-1, 1:-1] - self.Hy[:-2, 1:-1]) / self.dx -
                (self.Hx[1:-1, 1:-1] - self.Hx[1:-1, :-2]) / self.dy
            )
        )

    def apply_pec_boundary(self):
        """PEC 경계조건"""
        self.Ez[0, :] = 0
        self.Ez[-1, :] = 0
        self.Ez[:, 0] = 0
        self.Ez[:, -1] = 0

    def add_point_source(self, x, y, value):
        """점 소스 추가"""
        self.Ez[y, x] += value

    def add_line_source(self, x, value):
        """선 소스 추가 (y 방향 전체)"""
        self.Ez[:, x] += value

    def step(self, source_func=None, source_pos=None, source_type='point'):
        """한 시간 단계 전진"""
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
        """시뮬레이션 실행"""
        self.compute_coefficients()

        Ez_history = []

        for n in range(n_steps):
            self.step(source_func, source_pos, source_type)

            if n % record_interval == 0:
                Ez_history.append(self.Ez.copy())

        return np.array(Ez_history)


def demo_2d_fdtd_tm():
    """2D FDTD TM 모드 시연"""

    # 시뮬레이터 생성
    fdtd = FDTD_2D_TM(Nx=150, Ny=150, dx=1e-3, dy=1e-3, courant=0.9)

    # 유전체 블록 추가
    fdtd.set_material_region(90, 120, 60, 90, eps_r=4)

    # PEC 원형 실린더 추가
    fdtd.add_pec_circle(50, 75, 15)

    # 시뮬레이션 실행
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

    # 시각화
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

        # 재료 경계 표시
        ax.contour(X, Y, fdtd.eps_r, levels=[2], colors='green', linewidths=2)
        ax.contour(X, Y, fdtd.sigma, levels=[1e6], colors='black', linewidths=2)

        # 소스 위치
        ax.plot(source_pos[0] * fdtd.dx * 1000,
               source_pos[1] * fdtd.dy * 1000, 'r*', markersize=10)

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'Step {snap_idx * 5}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Ez')

    plt.suptitle('2D FDTD TM 모드: 산란 시뮬레이션', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_2d_tm_scattering.png', dpi=150, bbox_inches='tight')
    plt.show()

    return fdtd, Ez_history

# fdtd, Ez_history = demo_2d_fdtd_tm()
```

### 4.2 TE 모드 (Hz, Ex, Ey)

```python
class FDTD_2D_TE:
    """2D FDTD TE 모드 시뮬레이터"""

    def __init__(self, Nx=100, Ny=100, dx=1e-3, dy=1e-3, courant=0.99):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.dt = courant / (c0 * np.sqrt(1/dx**2 + 1/dy**2))

        # 필드 배열
        self.Hz = np.zeros((Ny, Nx))
        self.Ex = np.zeros((Ny, Nx))
        self.Ey = np.zeros((Ny, Nx))

        # 물성치
        self.eps_r = np.ones((Ny, Nx))
        self.sigma = np.zeros((Ny, Nx))

        self.time_step = 0

    def compute_coefficients(self):
        eps = eps0 * self.eps_r
        self.Ca = (1 - self.sigma * self.dt / (2 * eps)) / \
                  (1 + self.sigma * self.dt / (2 * eps))
        self.Cb = (self.dt / eps) / (1 + self.sigma * self.dt / (2 * eps))

    def update_E(self):
        """E 필드 업데이트"""
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
        """H 필드 업데이트"""
        # Hz = Hz + dt/μ₀ * (dEx/dy - dEy/dx)
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

        # PEC 경계
        self.Ex[0, :] = self.Ex[-1, :] = 0
        self.Ey[:, 0] = self.Ey[:, -1] = 0

        self.time_step += 1
```

---

## 5. PML (Perfectly Matched Layer)

### 5.1 PML 개념

```
PML (Berenger, 1994):

핵심 아이디어:
- 파동을 흡수하는 인공 매질층
- 매질 경계에서 반사 없음 (임피던스 정합)
- 층 내부에서 지수적 감쇠

구현 방식:
1. Split-field PML: 필드를 두 성분으로 분할
2. UPML (Uniaxial PML): 이방성 매질 표현
3. CPML (Convolutional PML): 컨볼루션 형태

PML 매개변수:
- 층 두께: 보통 8-20 셀
- 감쇠 프로파일: 다항식 또는 지수함수
- 최대 전도도: 최적값 존재
```

```python
class FDTD_2D_TM_PML:
    """2D FDTD TM 모드 with UPML"""

    def __init__(self, Nx=100, Ny=100, dx=1e-3, dy=1e-3, pml_layers=10, courant=0.9):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.pml_layers = pml_layers

        self.dt = courant / (c0 * np.sqrt(1/dx**2 + 1/dy**2))

        # 필드 배열
        self.Ez = np.zeros((Ny, Nx))
        self.Hx = np.zeros((Ny, Nx))
        self.Hy = np.zeros((Ny, Nx))

        # PML 보조 필드
        self.Psi_Ez_x = np.zeros((Ny, Nx))
        self.Psi_Ez_y = np.zeros((Ny, Nx))
        self.Psi_Hx_y = np.zeros((Ny, Nx))
        self.Psi_Hy_x = np.zeros((Ny, Nx))

        # 물성치
        self.eps_r = np.ones((Ny, Nx))
        self.sigma = np.zeros((Ny, Nx))

        # PML 계수 초기화
        self._setup_pml()

        self.time_step = 0

        print(f"2D FDTD TM + PML 초기화:")
        print(f"  격자: {Nx} x {Ny}, PML: {pml_layers} layers")

    def _pml_profile(self, d, d_max, sigma_max, order=3):
        """PML 전도도 프로파일"""
        return sigma_max * (d / d_max) ** order

    def _setup_pml(self):
        """PML 계수 설정"""
        # 최적 전도도
        sigma_max = 0.8 * (order + 1) / (eta0 * self.dx) if hasattr(self, 'order') else \
                   0.8 * 4 / (eta0 * self.dx)
        order = 3

        # x 방향 PML 계수
        self.sigma_x = np.zeros(self.Nx)
        self.sigma_x_star = np.zeros(self.Nx)  # dual grid

        for i in range(self.pml_layers):
            # 좌측 PML
            d = self.pml_layers - i
            self.sigma_x[i] = self._pml_profile(d, self.pml_layers, sigma_max, order)
            self.sigma_x_star[i] = self._pml_profile(d - 0.5, self.pml_layers, sigma_max, order)

            # 우측 PML
            d = i + 1
            self.sigma_x[-(i+1)] = self._pml_profile(d, self.pml_layers, sigma_max, order)
            self.sigma_x_star[-(i+1)] = self._pml_profile(d - 0.5, self.pml_layers, sigma_max, order)

        # y 방향 PML 계수
        self.sigma_y = np.zeros(self.Ny)
        self.sigma_y_star = np.zeros(self.Ny)

        for j in range(self.pml_layers):
            # 하단 PML
            d = self.pml_layers - j
            self.sigma_y[j] = self._pml_profile(d, self.pml_layers, sigma_max, order)
            self.sigma_y_star[j] = self._pml_profile(d - 0.5, self.pml_layers, sigma_max, order)

            # 상단 PML
            d = j + 1
            self.sigma_y[-(j+1)] = self._pml_profile(d, self.pml_layers, sigma_max, order)
            self.sigma_y_star[-(j+1)] = self._pml_profile(d - 0.5, self.pml_layers, sigma_max, order)

        # 계수 계산
        self.b_x = np.exp(-self.sigma_x * self.dt / eps0)
        self.c_x = (1 - self.b_x) / (self.sigma_x * self.dx + 1e-10)
        self.b_x_star = np.exp(-self.sigma_x_star * self.dt / eps0)
        self.c_x_star = (1 - self.b_x_star) / (self.sigma_x_star * self.dx + 1e-10)

        self.b_y = np.exp(-self.sigma_y * self.dt / eps0)
        self.c_y = (1 - self.b_y) / (self.sigma_y * self.dy + 1e-10)
        self.b_y_star = np.exp(-self.sigma_y_star * self.dt / eps0)
        self.c_y_star = (1 - self.b_y_star) / (self.sigma_y_star * self.dy + 1e-10)

    def update_H(self):
        """H 필드 업데이트 (PML 포함)"""
        # Hx 업데이트
        dEz_dy = (self.Ez[:, 1:] - self.Ez[:, :-1]) / self.dy
        self.Psi_Hx_y[:, :-1] = (self.b_y[:, np.newaxis] * self.Psi_Hx_y[:, :-1] +
                                self.c_y[:, np.newaxis] * dEz_dy)
        self.Hx[:, :-1] -= self.dt / mu0 * (dEz_dy + self.Psi_Hx_y[:, :-1])

        # Hy 업데이트
        dEz_dx = (self.Ez[1:, :] - self.Ez[:-1, :]) / self.dx
        self.Psi_Hy_x[:-1, :] = (self.b_x[np.newaxis, :] * self.Psi_Hy_x[:-1, :] +
                                self.c_x[np.newaxis, :] * dEz_dx)
        self.Hy[:-1, :] += self.dt / mu0 * (dEz_dx + self.Psi_Hy_x[:-1, :])

    def update_E(self):
        """E 필드 업데이트 (PML 포함)"""
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

        # PEC 외부 경계
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
    """PML 시연"""

    # PML 없는 경우
    fdtd_no_pml = FDTD_2D_TM(Nx=100, Ny=100, dx=1e-3, dy=1e-3, courant=0.9)
    fdtd_no_pml.compute_coefficients()

    # PML 있는 경우
    fdtd_pml = FDTD_2D_TM_PML(Nx=100, Ny=100, dx=1e-3, dy=1e-3, pml_layers=10, courant=0.9)

    source_pos = (50, 50)

    def source(t):
        return gaussian_pulse(t, t0=0.15e-9, tau=0.05e-9)

    n_steps = 200

    # 실행
    Ez_no_pml = fdtd_no_pml.run(n_steps, source, source_pos, record_interval=10)
    Ez_pml = fdtd_pml.run(n_steps, source, source_pos, record_interval=10)

    # 비교 시각화
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    x = np.arange(100) * 1e-3 * 1000
    y = np.arange(100) * 1e-3 * 1000
    X, Y = np.meshgrid(x, y)

    for col, snap_idx in enumerate([5, 10, 15, 19]):
        # PEC 경계 (반사)
        ax = axes[0, col]
        vmax = 0.3
        ax.pcolormesh(X, Y, Ez_no_pml[snap_idx], cmap='RdBu_r',
                     shading='auto', vmin=-vmax, vmax=vmax)
        ax.set_title(f'PEC 경계, Step {snap_idx * 10}')
        ax.set_aspect('equal')
        if col == 0:
            ax.set_ylabel('Without PML\ny [mm]')

        # PML 경계 (흡수)
        ax = axes[1, col]
        ax.pcolormesh(X, Y, Ez_pml[snap_idx], cmap='RdBu_r',
                     shading='auto', vmin=-vmax, vmax=vmax)

        # PML 영역 표시
        pml = 10
        ax.axvline(x=pml, color='green', linestyle='--', alpha=0.5)
        ax.axvline(x=100-pml, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=pml, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=100-pml, color='green', linestyle='--', alpha=0.5)

        ax.set_title(f'PML 경계, Step {snap_idx * 10}')
        ax.set_aspect('equal')
        if col == 0:
            ax.set_ylabel('With PML\ny [mm]')
        ax.set_xlabel('x [mm]')

    plt.suptitle('PEC vs PML 경계조건 비교', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_pml_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# demo_pml()
```

---

## 6. 응용 예제: 도파관

### 6.1 직사각형 도파관 시뮬레이션

```python
def rectangular_waveguide():
    """직사각형 도파관 시뮬레이션"""

    # 도파관 치수 (WR-90: a=22.86mm, b=10.16mm)
    # 간소화된 치수 사용
    a = 30  # 도파관 폭 (셀 수)
    b = 15  # 도파관 높이 (셀 수)

    Nx = 150
    Ny = 40
    dx = dy = 1e-3  # 1 mm

    fdtd = FDTD_2D_TM(Nx=Nx, Ny=Ny, dx=dx, dy=dy, courant=0.9)

    # 도파관 벽 (PEC)
    wall_y1 = (Ny - b) // 2
    wall_y2 = wall_y1 + b

    # 상단/하단 PEC 벽
    fdtd.sigma[:wall_y1, :] = 1e7
    fdtd.sigma[wall_y2:, :] = 1e7

    fdtd.compute_coefficients()

    # TE10 모드 여기 주파수
    f_c = c0 / (2 * a * dx)  # 차단 주파수
    f_op = 1.5 * f_c  # 운용 주파수

    print(f"차단 주파수: {f_c/1e9:.2f} GHz")
    print(f"운용 주파수: {f_op/1e9:.2f} GHz")

    # 소스 (도파관 내부)
    source_x = 10
    source_y = Ny // 2

    def source(t):
        t0 = 0.2e-9
        tau = 0.05e-9
        return np.sin(2 * np.pi * f_op * t) * (1 - np.exp(-((t - t0)/tau)**2) if t < t0 else 1)

    # 시뮬레이션
    n_steps = 500
    Ez_history = []

    for n in range(n_steps):
        fdtd.step(source, (source_x, source_y))

        if n % 5 == 0:
            Ez_history.append(fdtd.Ez.copy())

    Ez_history = np.array(Ez_history)

    # 시각화
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

        # 도파관 벽 표시
        ax.axhline(y=wall_y1 * dy * 1000, color='black', linewidth=2)
        ax.axhline(y=wall_y2 * dy * 1000, color='black', linewidth=2)

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'Step {snap_idx * 5}')
        plt.colorbar(im, ax=ax)

    # 파형 분석
    ax = axes[1, 2]
    probe_y = Ny // 2
    probe_x_list = [30, 60, 90, 120]

    for px in probe_x_list:
        signal = Ez_history[:, probe_y, px]
        t = np.arange(len(signal)) * 5 * fdtd.dt * 1e9
        ax.plot(t, signal, label=f'x={px*dx*1000:.0f}mm')

    ax.set_xlabel('t [ns]')
    ax.set_ylabel('Ez')
    ax.set_title('다양한 위치에서의 필드')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'직사각형 도파관 (f_op = {f_op/1e9:.1f} GHz > f_c = {f_c/1e9:.1f} GHz)', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_waveguide.png', dpi=150, bbox_inches='tight')
    plt.show()

# rectangular_waveguide()
```

---

## 7. 연습 문제

### 연습 1: 소스(Source) 비교
가우시안 펄스(Gaussian pulse)와 리커 파렛(Ricker wavelet)을 소스로 사용하여 1D FDTD 결과를 비교하시오. 주파수 응답 특성을 분석하시오.

<details><summary>정답 보기</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_fdtd_sources(N=200, Nt=400, dx=1e-3):
    """가우시안 펄스 vs 리커 파렛 소스 비교"""
    dt = dx / (2 * 3e8)  # CFL (S=0.5)

    def run_fdtd(source_type):
        Ez = np.zeros(N); Hy = np.zeros(N)
        src_pos = N // 4
        t0, spread = 30, 10  # 가우시안 파라미터
        f_c = 1e10  # 리커 중심 주파수
        output = []

        for t in range(Nt):
            if source_type == 'gaussian':
                src = np.exp(-0.5 * ((t - t0) / spread)**2)
            else:  # Ricker wavelet
                tau = (t - t0) * dt
                src = (1 - 2*(np.pi*f_c*tau)**2) * np.exp(-(np.pi*f_c*tau)**2)

            Ez[src_pos] += src
            Hy[:-1] += 0.1 * (Ez[1:] - Ez[:-1])
            Ez[1:]  += 0.1 * (Hy[1:] - Hy[:-1])
            output.append(Ez[3*N//4])

        return np.array(output)

    out_gauss  = run_fdtd('gaussian')
    out_ricker = run_fdtd('ricker')

    # 주파수 스펙트럼 비교
    freq = np.fft.fftfreq(Nt, d=dt)
    spec_g = np.abs(np.fft.fft(out_gauss))
    spec_r = np.abs(np.fft.fft(out_ricker))
    pos_mask = freq > 0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].plot(out_gauss);  axes[0,0].set_title('가우시안 펄스 시간 응답')
    axes[0,1].plot(out_ricker); axes[0,1].set_title('리커 파렛 시간 응답')
    axes[1,0].plot(freq[pos_mask]/1e9, spec_g[pos_mask], 'b-')
    axes[1,0].set_xlabel('주파수 (GHz)'); axes[1,0].set_title('가우시안 스펙트럼')
    axes[1,1].plot(freq[pos_mask]/1e9, spec_r[pos_mask], 'r-')
    axes[1,1].set_xlabel('주파수 (GHz)'); axes[1,1].set_title('리커 스펙트럼')
    plt.tight_layout()
    plt.savefig('source_comparison.png', dpi=150)
    plt.close()
    print("가우시안: DC 성분 포함, 저주파 중심의 넓은 스펙트럼")
    print("리커:    DC=0, 중심 주파수 f_c 주변 대역통과 스펙트럼")

compare_fdtd_sources()
```

가우시안 펄스는 DC 성분을 포함한 넓은 스펙트럼을 가지며 시간 응답 분석에 적합합니다. 리커 파렛은 DC 성분이 없고 중심 주파수 f_c 주변에 에너지가 집중되어 특정 주파수 대역 분석에 유리합니다.
</details>

### 연습 2: ABC(Absorbing Boundary Condition) 성능 비교
1차 Mur ABC와 2차 Mur ABC의 반사 계수를 비교하시오. 입사각에 따른 성능 차이를 분석하시오.

<details><summary>정답 보기</summary>

```python
def mur_abc_comparison():
    """1차 vs 2차 Mur ABC 반사 계수 비교"""
    N = 200; Nt = 600
    dx = 1e-3; dt = dx / (3e8 * 2)
    S = 3e8 * dt / dx

    def run_with_abc(order):
        Ez = np.zeros(N + 2)
        Hy = np.zeros(N + 1)
        Ez_prev = np.zeros(N + 2)
        Ez_prev2 = np.zeros(N + 2)
        src_pos = N // 4
        monitor_pos = src_pos - 20
        ref_signal = []

        for t in range(Nt):
            Ez_prev2[:] = Ez_prev[:]
            Ez_prev[:]  = Ez[:]
            if t < 60:
                Ez[src_pos] += np.exp(-0.5*((t-30)/10)**2)
            Hy[:] += S * (Ez[1:] - Ez[:-1])
            Ez[1:-1] += S * (Hy[1:] - Hy[:-1])
            if order == 1:
                Ez[-1] = Ez_prev[-2] + (S-1)/(S+1) * (Ez[-2] - Ez_prev[-1])
            elif order == 2:
                coeff1 = (S - 1) / (S + 1)
                coeff2 = 2 * S / (S + 1)
                Ez[-1] = (-Ez_prev2[-1] + coeff1*(Ez[-2]+Ez_prev2[-2])
                           + coeff2*(Ez_prev[-2]+Ez_prev[-1]))
            ref_signal.append(Ez[monitor_pos])
        return np.array(ref_signal)

    ref1 = run_with_abc(1)
    ref2 = run_with_abc(2)
    R1 = np.max(np.abs(ref1[100:]))
    R2 = np.max(np.abs(ref2[100:]))
    print(f"1차 Mur ABC 반사 계수: R ≈ {R1:.4f} ({20*np.log10(max(R1,1e-16)):.1f} dB)")
    print(f"2차 Mur ABC 반사 계수: R ≈ {R2:.4f} ({20*np.log10(max(R2,1e-16)):.1f} dB)")
    print("2차가 1차보다 약 10~20 dB 더 낮은 반사 계수를 가짐")

mur_abc_comparison()
```

2차 Mur ABC는 더 넓은 입사각 범위에서 우수한 성능을 제공하지만 구현이 복잡합니다. 현재는 PML(완전 정합층)이 가장 널리 사용되는 ABC입니다.
</details>

### 연습 3: PML(Perfectly Matched Layer) 최적화
PML 층 두께(5, 10, 15, 20 셀)와 다항식 차수(2, 3, 4)에 따른 흡수 성능을 비교하시오.

<details><summary>정답 보기</summary>

```python
def pml_optimization_study():
    """PML 두께 및 차수에 따른 흡수 성능 비교"""
    def pml_reflection_coeff(d_pml, poly_order, dx=1e-3):
        R_target = 1e-8
        sigma_max = -(poly_order + 1) / (2 * d_pml * dx) * np.log(R_target)
        sigma_profile = np.array([(i/d_pml)**poly_order * sigma_max
                                   for i in range(d_pml)])
        R_theory = np.exp(-2 * np.sum(sigma_profile) * dx / (3e8 * 8.854e-12))
        return R_theory

    print(f"{'두께(셀)':>8} {'차수':>6} {'반사계수':>14} {'dB':>10}")
    print("-" * 45)
    for d in [5, 10, 15, 20]:
        for order in [2, 3, 4]:
            R = pml_reflection_coeff(d, order)
            dB = 20 * np.log10(max(R, 1e-16))
            print(f"{d:>8} {order:>6} {R:>14.2e} {dB:>10.1f}")
    print("\n권장값: 두께 8~16 셀, 차수 2~3")

pml_optimization_study()
```

두께가 두꺼울수록, 차수가 높을수록 이론적 반사 계수가 낮아지지만, 실제로는 격자 이산화 오차로 최적 성능이 제한됩니다.
</details>

### 연습 4: 도파관 모드(Waveguide Modes)
직사각형 도파관에서 TE10 차단 주파수 이하와 이상의 주파수로 전파를 시뮬레이션하고 결과 차이를 분석하시오.

<details><summary>정답 보기</summary>

```python
def waveguide_mode_simulation():
    """직사각형 도파관 TE10 모드 분석"""
    a = 0.02    # 폭 [m] (2 cm)
    c = 3e8

    f_cutoff = c / (2 * a)
    print(f"TE10 차단 주파수: f_c = {f_cutoff/1e9:.2f} GHz")

    for f_test in [f_cutoff * 0.8, f_cutoff * 1.5]:
        k0 = 2 * np.pi * f_test / c
        kc = np.pi / a
        k_sq = k0**2 - kc**2

        if k_sq > 0:
            beta = np.sqrt(k_sq)
            lambda_g = 2 * np.pi / beta
            v_phase = 2 * np.pi * f_test / beta
            print(f"\nf = {f_test/1e9:.2f} GHz (전파 가능):")
            print(f"  위상 상수 β = {beta:.2f} rad/m")
            print(f"  도파관 파장 λ_g = {lambda_g*100:.2f} cm")
            print(f"  위상 속도 v_p = {v_phase/c:.3f}c")
        else:
            alpha = np.sqrt(-k_sq)
            print(f"\nf = {f_test/1e9:.2f} GHz (에바네슨트, 차단):")
            print(f"  감쇠 상수 α = {alpha:.2f} Np/m")
            print(f"  감쇠율 = {20*alpha*np.log10(np.e):.1f} dB/m")

waveguide_mode_simulation()
```

차단 주파수 이하에서는 에바네슨트(evanescent) 파로 에너지 전달이 불가능합니다. 차단 주파수 이상에서는 TE10 모드가 전파되며, 위상 속도 v_p > c이지만 에너지는 군속도 v_g = c²/v_p < c로 전달됩니다.
</details>

---

## 8. 참고자료

### 핵심 논문
- Yee (1966) "Numerical Solution of Initial Boundary Value Problems..."
- Mur (1981) "Absorbing Boundary Conditions for the Finite-Difference Approximation..."
- Berenger (1994) "A Perfectly Matched Layer for the Absorption of Electromagnetic Waves"

### 교재
- Taflove & Hagness, "Computational Electrodynamics: The FDTD Method"
- Sullivan, "Electromagnetic Simulation Using the FDTD Method"

### 오픈소스
- MEEP (MIT): Python/C++ FDTD
- gprMax: Ground Penetrating Radar FDTD
- OpenEMS: FDTD + 회로 시뮬레이션

---

## 요약

```
FDTD 구현 핵심:

1. 알고리즘 구조:
   - H 업데이트 → 소스 주입 → E 업데이트 → ABC

2. 소스 유형:
   - Hard: 강제 설정 (반사 발생)
   - Soft: 추가 (+= ) (반사파 통과)
   - TF/SF: 정확한 평면파

3. 흡수 경계조건:
   - Simple ABC: 가장 단순, S=1에서만 정확
   - Mur ABC: 개선된 성능, 수직 입사에 효과적
   - PML: 최고 성능, 구현 복잡

4. 2D 모드:
   - TM: Ez, Hx, Hy (z 편광)
   - TE: Hz, Ex, Ey (z 편광)

5. PML 요소:
   - 층 두께: 8-20 셀
   - 다항식 프로파일 (order 3-4)
   - CPML이 가장 효과적

6. 수치적 고려:
   - Courant 조건 준수
   - 파장당 10-20 셀
   - 수치 분산 최소화
```

---

## 연습 문제

### 연습 1: 하드 소스(Hard Source)와 소프트 소스(Soft Source) 반사 필드 비교
FDTD_1D 클래스를 사용하여 x = 150 mm에 완전 도체(PEC) 반사판이 있는 1D 도메인을 설정하세요. x = 50 mm에서 하드 소스(hard source)와 소프트 소스(soft source)를 각각 사용하여 가우시안 펄스(Gaussian pulse)를 입사시키세요. 펄스가 반사되어 소스 영역으로 돌아온 후, 두 경우에 대해 소스 위치에서의 필드를 비교하세요. 하드 소스는 늦은 시간에 아티팩트(artifact)를 생성하지만 소프트 소스는 생성하지 않는 이유를 설명하세요.

### 연습 2: Mur ABC 각도 의존성(Angle Dependence)
중심에 점 소스(point source)가 있는 2D FDTD TM 시뮬레이션을 구현하세요. 세 면의 완전 도체(PEC) 경계를 1차 Mur ABC로 교체하고, 나머지 한 면은 PEC로 유지하세요. ABC가 수직 입사파와 45° 입사파를 각각 얼마나 잘 흡수하는지 시각적으로 비교하세요. 각 경계에서 5셀 떨어진 프로브 위치에서 펄스 도착 전후의 최대 필드 진폭을 기록하여 각도별 반사 계수(reflection coefficient)를 추정하세요.

### 연습 3: PML 두께(Thickness) 최적화
FDTD_2D_TM_PML 클래스를 사용하여 PML 레이어 수 4, 8, 12, 16 셀로 시뮬레이션을 실행하세요. 각 경우에 중앙에 프로브를 배치하고, 주 펄스가 도메인을 빠져나간 후 반사 잡음(spurious reflected field) 최대 진폭을 측정하세요. 반사 수준(dB)을 PML 두께의 함수로 그려보고 -40 dB 반사를 달성하는 데 필요한 최소 두께를 결정하세요. PML 차수(order 2, 3, 4)는 이 결과에 어떤 영향을 주나요?

### 연습 4: 도파관(Waveguide) 모드 차단 주파수(Cutoff Frequency) 검증
직사각형 도파관(rectangular waveguide) 시뮬레이션 코드를 사용하여 동작 주파수를 (a) 0.8×fc, (b) fc, (c) 1.2×fc로 설정하세요 (fc는 TE10 차단 주파수). 각 경우에 300 스텝에서 Ez 필드 스냅샷을 그리세요. 케이스 (a)와 (b)에서는 지수적 감쇠(exponential decay) 또는 비전파(no propagation)를 관찰하고, 케이스 (c)에서는 안내파(guided wave) 전파를 관찰하세요. 정상파(standing wave) 패턴에서 도파관 파장(guide wavelength) λg를 측정하고 이론값 λg = λ/√(1-(fc/f)²)과 비교하세요.

### 연습 5: 가우시안 펄스(Gaussian Pulse) 스펙트럼 분석
가우시안 펄스(t0 = 1×10⁻¹⁰ s, tau = 3×10⁻¹¹ s)로 1D FDTD를 실행하세요. 소스 근처와 도메인 끝(유전체 슬래브 통과 전후)에 있는 두 프로브 위치에서 시계열(time series)을 기록하세요. 각 시계열의 FFT를 계산하고 크기 스펙트럼(magnitude spectrum)을 그리세요. 다음을 확인하세요: (a) 소스 스펙트럼의 -3 dB 대역폭(bandwidth), (b) 유전체 슬래브로 인한 주파수 의존적 감쇠(frequency-dependent attenuation), (c) 슬래브를 통한 전파로 인한 위상 이동(phase shift).

---

다음 레슨에서는 MHD (자기유체역학)의 기초 이론을 다룹니다.
