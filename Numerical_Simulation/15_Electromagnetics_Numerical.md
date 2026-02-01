# 15. 전자기학 수치해석 (Computational Electromagnetics)

## 학습 목표
- Maxwell 방정식의 물리적 의미 복습
- 전자기장의 수치적 이산화 이해
- FDTD (Finite-Difference Time-Domain) 방법 소개
- Yee 격자 구조 파악
- 전자기파의 Courant 조건 학습

---

## 1. Maxwell 방정식 복습

### 1.1 미분 형태

```
Maxwell 방정식 (진공, SI 단위계):

1. Gauss 법칙 (전기):
   ∇·E = ρ/ε₀

2. Gauss 법칙 (자기):
   ∇·B = 0

3. Faraday 법칙:
   ∇×E = -∂B/∂t

4. Ampère-Maxwell 법칙:
   ∇×B = μ₀J + μ₀ε₀ ∂E/∂t

여기서:
- E: 전기장 [V/m]
- B: 자기장 (자속밀도) [T]
- ρ: 전하밀도 [C/m³]
- J: 전류밀도 [A/m²]
- ε₀ = 8.854×10⁻¹² F/m (진공 유전율)
- μ₀ = 4π×10⁻⁷ H/m (진공 투자율)
- c = 1/√(μ₀ε₀) ≈ 3×10⁸ m/s (광속)
```

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 물리 상수
eps0 = 8.854e-12  # 진공 유전율 [F/m]
mu0 = 4 * np.pi * 1e-7  # 진공 투자율 [H/m]
c0 = 1 / np.sqrt(mu0 * eps0)  # 광속 [m/s]

print(f"진공 유전율 ε₀ = {eps0:.3e} F/m")
print(f"진공 투자율 μ₀ = {mu0:.3e} H/m")
print(f"광속 c₀ = {c0:.3e} m/s")

def maxwell_equations_overview():
    """Maxwell 방정식 개요"""

    print("=" * 60)
    print("Maxwell 방정식과 물리적 의미")
    print("=" * 60)

    overview = """
    ┌─────────────────────────────────────────────────────────┐
    │              Maxwell 방정식 체계                         │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  Gauss (전기): ∇·E = ρ/ε₀                              │
    │  → 전하가 전기장의 발산원                                │
    │                                                         │
    │  Gauss (자기): ∇·B = 0                                  │
    │  → 자기 단극자 없음 (항상 N-S 쌍)                        │
    │                                                         │
    │  Faraday: ∇×E = -∂B/∂t                                 │
    │  → 시변 자기장이 전기장 회전 유도                        │
    │                                                         │
    │  Ampère-Maxwell: ∇×B = μ₀J + μ₀ε₀∂E/∂t                 │
    │  → 전류와 시변 전기장이 자기장 회전 유도                 │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    파동 방정식 유도:
    Faraday와 Ampère 법칙을 결합:

    ∇×(∇×E) = -∂(∇×B)/∂t = -μ₀∂J/∂t - μ₀ε₀∂²E/∂t²

    벡터 항등식 ∇×(∇×E) = ∇(∇·E) - ∇²E 적용:

    전하/전류 없는 진공에서:
    ∇²E = μ₀ε₀ ∂²E/∂t² = (1/c²) ∂²E/∂t²

    → 속도 c = 1/√(μ₀ε₀) 인 파동 방정식!
    """
    print(overview)

maxwell_equations_overview()
```

### 1.2 1D 파동 방정식

```python
def electromagnetic_wave_1d():
    """1D 전자기파 해석해 시각화"""

    # 1D TEM 파 (x 방향 전파, y 편광)
    # Ey와 Hz 성분만 존재

    # 공간/시간 설정
    L = 10.0  # 도메인 길이 [m]
    T = 3 * L / c0  # 시뮬레이션 시간

    x = np.linspace(0, L, 500)
    times = [0, L/(3*c0), 2*L/(3*c0), L/c0]

    # 초기 가우시안 펄스
    x0 = L / 4
    sigma = L / 20
    wavelength = L / 5
    k = 2 * np.pi / wavelength
    omega = c0 * k

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, t in enumerate(times):
        ax = axes[idx // 2, idx % 2]

        # 전기장 Ey (가우시안 변조된 사인파)
        envelope = np.exp(-((x - x0 - c0*t) / sigma)**2)
        Ey = envelope * np.sin(k*(x - x0) - omega*t)

        # 자기장 Hz (Ey와 비례, 위상 동일)
        Hz = Ey / (c0 * mu0)  # Hz = Ey/(c*μ₀) for plane wave

        ax.plot(x, Ey, 'b-', linewidth=2, label=r'$E_y$ (전기장)')
        ax.plot(x, Hz * c0 * mu0, 'r--', linewidth=2, label=r'$\mu_0 c H_z$ (자기장)')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('Field amplitude')
        ax.set_title(f't = {t*1e9:.2f} ns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, L)
        ax.set_ylim(-1.5, 1.5)

    plt.suptitle('1D 전자기파 전파 (TEM 모드)', fontsize=14)
    plt.tight_layout()
    plt.savefig('em_wave_1d.png', dpi=150, bbox_inches='tight')
    plt.show()

# electromagnetic_wave_1d()
```

---

## 2. Maxwell 방정식의 이산화

### 2.1 유한차분 접근

```
1D TEM 파동 (Ey, Hz 성분):

Faraday 법칙 (z 성분):
∂Ey/∂x = -∂Bz/∂t = -μ₀ ∂Hz/∂t

Ampère 법칙 (y 성분):
∂Hz/∂x = -ε₀ ∂Ey/∂t

이산화 (중심차분):
∂Ey/∂x ≈ (Ey[i+1] - Ey[i]) / Δx
∂Hz/∂t ≈ (Hz[n+1/2] - Hz[n-1/2]) / Δt

시간 엇갈림 (Leapfrog):
- E는 정수 시간 단계: t = nΔt
- H는 반정수 시간 단계: t = (n+1/2)Δt
```

```python
def fdtd_discretization_concept():
    """FDTD 이산화 개념 시각화"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) 공간 이산화
    ax1 = axes[0]

    # E와 H의 엇갈린 위치
    n_points = 6
    for i in range(n_points):
        # E 점 (정수 위치)
        ax1.plot(i, 0.5, 'bo', markersize=15)
        ax1.text(i, 0.7, f'$E_y^n[{i}]$', ha='center', fontsize=10, color='blue')

        # H 점 (반정수 위치)
        if i < n_points - 1:
            ax1.plot(i + 0.5, 0.5, 'r^', markersize=12)
            ax1.text(i + 0.5, 0.3, f'$H_z^{{n+1/2}}[{i}]$', ha='center', fontsize=9, color='red')

    # 격자선
    for i in range(n_points):
        ax1.axvline(x=i, color='gray', linestyle=':', alpha=0.5)

    ax1.axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlim(-0.5, n_points - 0.5)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Position index i')
    ax1.set_title('공간 엇갈림 (Spatial Staggering)')
    ax1.set_yticks([])

    # 범례
    ax1.plot([], [], 'bo', markersize=10, label='E field')
    ax1.plot([], [], 'r^', markersize=10, label='H field')
    ax1.legend(loc='upper right')

    # (2) 시간 이산화 (Leapfrog)
    ax2 = axes[1]

    n_steps = 5
    for n in range(n_steps):
        # E 점 (정수 시간)
        ax2.plot(0.3, n, 'bo', markersize=15)
        ax2.text(0.5, n, f'$E^{n}$', ha='left', fontsize=12, color='blue')

        # H 점 (반정수 시간)
        ax2.plot(0.3, n + 0.5, 'r^', markersize=12)
        ax2.text(0.5, n + 0.5, f'$H^{{{n}+1/2}}$', ha='left', fontsize=11, color='red')

    # 화살표 (업데이트 순서)
    for n in range(n_steps - 1):
        # E -> H
        ax2.annotate('', xy=(0.3, n + 0.5), xytext=(0.3, n),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        # H -> E
        ax2.annotate('', xy=(0.3, n + 1), xytext=(0.3, n + 0.5),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, n_steps)
    ax2.set_ylabel('Time step n')
    ax2.set_title('시간 엇갈림 (Leapfrog)')
    ax2.set_xticks([])

    plt.tight_layout()
    plt.savefig('fdtd_discretization.png', dpi=150, bbox_inches='tight')
    plt.show()

# fdtd_discretization_concept()
```

### 2.2 업데이트 방정식

```
1D FDTD 업데이트 방정식:

1. H 업데이트 (n → n+1/2):
   Hz^(n+1/2)[i] = Hz^(n-1/2)[i] - (Δt/μ₀Δx)(Ey^n[i+1] - Ey^n[i])

2. E 업데이트 (n+1/2 → n+1):
   Ey^(n+1)[i] = Ey^n[i] - (Δt/ε₀Δx)(Hz^(n+1/2)[i] - Hz^(n+1/2)[i-1])

매개변수화:
C_a = Δt/(μ₀Δx)  (H 계수)
C_b = Δt/(ε₀Δx)  (E 계수)

Hz^(n+1/2)[i] = Hz^(n-1/2)[i] - C_a (Ey^n[i+1] - Ey^n[i])
Ey^(n+1)[i] = Ey^n[i] - C_b (Hz^(n+1/2)[i] - Hz^(n+1/2)[i-1])
```

```python
def simple_1d_fdtd():
    """간단한 1D FDTD 시뮬레이션"""

    # 격자 설정
    Nx = 200
    dx = 1e-3  # 1 mm

    # 시간 설정
    dt = dx / (2 * c0)  # Courant 조건 만족
    n_steps = 500

    # 배열 초기화
    Ey = np.zeros(Nx)
    Hz = np.zeros(Nx)

    # 계수
    Ca = dt / (mu0 * dx)
    Cb = dt / (eps0 * dx)

    # Courant 수
    S = c0 * dt / dx
    print(f"Courant 수 S = {S:.4f}")

    # 소스 위치
    source_pos = Nx // 4

    # 기록용
    Ey_history = []
    times_to_record = [0, 100, 200, 300, 400]

    # 메인 루프
    for n in range(n_steps):
        # H 업데이트
        Hz[:-1] = Hz[:-1] - Ca * (Ey[1:] - Ey[:-1])

        # 소스 (가우시안 펄스)
        t = n * dt
        t0 = 30 * dt
        tau = 10 * dt
        source = np.exp(-((t - t0) / tau) ** 2)
        Ey[source_pos] += source

        # E 업데이트
        Ey[1:] = Ey[1:] - Cb * (Hz[1:] - Hz[:-1])

        # 경계조건 (간단한 흡수)
        Ey[0] = 0
        Ey[-1] = 0

        # 기록
        if n in times_to_record:
            Ey_history.append((n, Ey.copy()))

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    x = np.arange(Nx) * dx * 1000  # mm

    for idx, (n, Ey_snap) in enumerate(Ey_history):
        ax = axes[idx // 3, idx % 3]
        ax.plot(x, Ey_snap, 'b-', linewidth=1.5)
        ax.axvline(x=source_pos * dx * 1000, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('Ey')
        ax.set_title(f'Step n = {n}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.2, 1.2)

    # 마지막 subplot에 설명
    ax = axes[1, 2]
    ax.text(0.5, 0.8, 'FDTD 파라미터:', fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.6, f'Nx = {Nx}', fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.5, f'dx = {dx*1000:.2f} mm', fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, f'dt = {dt*1e12:.2f} ps', fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.3, f'Courant S = {S:.2f}', fontsize=10, ha='center', transform=ax.transAxes)
    ax.axis('off')

    plt.suptitle('1D FDTD 시뮬레이션 - 가우시안 펄스 전파', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_1d_simple.png', dpi=150, bbox_inches='tight')
    plt.show()

    return Ey, Hz

# Ey, Hz = simple_1d_fdtd()
```

---

## 3. Yee 격자 (Yee Lattice)

### 3.1 3D Yee 셀

```
Yee 격자 (1966):
- E와 H 성분이 공간적으로 엇갈림
- 시간적으로도 반 단계 엇갈림
- Maxwell 방정식의 자연스러운 이산화

3D Yee 셀 구조:
- Ex: y-z 면의 변 중심
- Ey: x-z 면의 변 중심
- Ez: x-y 면의 변 중심
- Hx: x 수직 면 중심
- Hy: y 수직 면 중심
- Hz: z 수직 면 중심

각 E 성분은 4개의 H 성분으로 둘러싸임
각 H 성분은 4개의 E 성분으로 둘러싸임
```

```python
def yee_cell_visualization():
    """3D Yee 셀 시각화"""

    fig = plt.figure(figsize=(14, 6))

    # (1) 3D 뷰
    ax1 = fig.add_subplot(121, projection='3d')

    # 큐브 꼭지점
    a = 1.0  # 셀 크기

    # E 필드 위치 (셀 변 중심)
    # Ex: (a, a/2, 0), (a, a/2, a), (0, a/2, 0), (0, a/2, a)
    Ex_pos = [(a, a/2, 0), (a, a/2, a), (0, a/2, 0), (0, a/2, a)]
    # Ey 위치
    Ey_pos = [(a/2, a, 0), (a/2, a, a), (a/2, 0, 0), (a/2, 0, a)]
    # Ez 위치
    Ez_pos = [(0, 0, a/2), (a, 0, a/2), (0, a, a/2), (a, a, a/2)]

    # H 필드 위치 (셀 면 중심)
    Hx_pos = [(a/2, a/2, 0), (a/2, a/2, a)]
    Hy_pos = [(a/2, 0, a/2), (a/2, a, a/2)]
    Hz_pos = [(0, a/2, a/2), (a, a/2, a/2)]

    # 큐브 그리기
    vertices = [
        [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
        [0, 0, a], [a, 0, a], [a, a, a], [0, a, a]
    ]

    # 모서리
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 하단
        [4, 5], [5, 6], [6, 7], [7, 4],  # 상단
        [0, 4], [1, 5], [2, 6], [3, 7]   # 수직
    ]

    for edge in edges:
        points = [vertices[edge[0]], vertices[edge[1]]]
        ax1.plot3D(*zip(*points), 'k-', alpha=0.3)

    # E 필드 (화살표)
    for pos in Ex_pos:
        ax1.quiver(pos[0]-0.15, pos[1], pos[2], 0.3, 0, 0, color='blue', arrow_length_ratio=0.3)
    for pos in Ey_pos[:2]:
        ax1.quiver(pos[0], pos[1]-0.15, pos[2], 0, 0.3, 0, color='blue', arrow_length_ratio=0.3)
    for pos in Ez_pos[:2]:
        ax1.quiver(pos[0], pos[1], pos[2]-0.15, 0, 0, 0.3, color='blue', arrow_length_ratio=0.3)

    # H 필드 (화살표)
    for pos in Hx_pos:
        ax1.quiver(pos[0]-0.15, pos[1], pos[2], 0.3, 0, 0, color='red', arrow_length_ratio=0.3)
    for pos in Hy_pos:
        ax1.quiver(pos[0], pos[1]-0.15, pos[2], 0, 0.3, 0, color='red', arrow_length_ratio=0.3)
    for pos in Hz_pos:
        ax1.quiver(pos[0], pos[1], pos[2]-0.15, 0, 0, 0.3, color='red', arrow_length_ratio=0.3)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('3D Yee 셀')

    # 범례
    ax1.plot([], [], 'b-', linewidth=2, label='E field')
    ax1.plot([], [], 'r-', linewidth=2, label='H field')
    ax1.legend()

    # (2) 2D 뷰 (x-y 평면)
    ax2 = fig.add_subplot(122)

    # 격자
    for i in range(3):
        ax2.axhline(y=i, color='gray', linestyle='-', linewidth=0.5)
        ax2.axvline(x=i, color='gray', linestyle='-', linewidth=0.5)

    # Ez (셀 꼭지점)
    for i in range(3):
        for j in range(3):
            ax2.plot(i, j, 'bo', markersize=12)
            ax2.text(i+0.1, j+0.1, f'Ez({i},{j})', fontsize=8, color='blue')

    # Hx (수평 변 중심)
    for i in range(3):
        for j in range(2):
            ax2.plot(i, j+0.5, 'r>', markersize=10)

    # Hy (수직 변 중심)
    for i in range(2):
        for j in range(3):
            ax2.plot(i+0.5, j, 'r^', markersize=10)

    # Hz (셀 중심)
    for i in range(2):
        for j in range(2):
            ax2.plot(i+0.5, j+0.5, 'rs', markersize=10)
            ax2.text(i+0.6, j+0.5, f'Hz', fontsize=8, color='red')

    ax2.set_xlabel('i')
    ax2.set_ylabel('j')
    ax2.set_title('2D Yee 격자 (TM 모드)')
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.3, 2.5)
    ax2.set_ylim(-0.3, 2.5)

    # 범례
    ax2.plot([], [], 'bo', markersize=10, label='Ez')
    ax2.plot([], [], 'r>', markersize=10, label='Hx')
    ax2.plot([], [], 'r^', markersize=10, label='Hy')
    ax2.plot([], [], 'rs', markersize=10, label='Hz')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('yee_cell.png', dpi=150, bbox_inches='tight')
    plt.show()

# yee_cell_visualization()
```

### 3.2 2D FDTD 모드

```
2D FDTD 분해:

TM 모드 (Transverse Magnetic):
- 성분: Ez, Hx, Hy
- Ez가 z 방향, H가 x-y 평면

방정식:
∂Hx/∂t = -(1/μ) ∂Ez/∂y
∂Hy/∂t = (1/μ) ∂Ez/∂x
∂Ez/∂t = (1/ε)(∂Hy/∂x - ∂Hx/∂y) - σ/ε Ez

TE 모드 (Transverse Electric):
- 성분: Hz, Ex, Ey
- Hz가 z 방향, E가 x-y 평면

방정식:
∂Ex/∂t = (1/ε) ∂Hz/∂y - σ/ε Ex
∂Ey/∂t = -(1/ε) ∂Hz/∂x - σ/ε Ey
∂Hz/∂t = (1/μ)(∂Ex/∂y - ∂Ey/∂x)
```

```python
def fdtd_2d_tm_mode():
    """2D FDTD TM 모드 시뮬레이션"""

    # 격자 설정
    Nx, Ny = 100, 100
    dx = dy = 1e-3  # 1 mm

    # 시간 설정
    dt = 1 / (c0 * np.sqrt(1/dx**2 + 1/dy**2)) * 0.99  # Courant
    n_steps = 300

    # 물성치
    eps = eps0 * np.ones((Ny, Nx))  # 유전율
    mu = mu0 * np.ones((Ny, Nx))    # 투자율
    sigma = np.zeros((Ny, Nx))      # 전도도

    # 배열 초기화
    Ez = np.zeros((Ny, Nx))
    Hx = np.zeros((Ny, Nx))
    Hy = np.zeros((Ny, Nx))

    # 계수
    Ca = (1 - sigma * dt / (2 * eps)) / (1 + sigma * dt / (2 * eps))
    Cb = (dt / eps) / (1 + sigma * dt / (2 * eps))

    # 소스 위치
    source_x, source_y = Nx // 2, Ny // 2

    # Courant 수
    S = c0 * dt * np.sqrt(1/dx**2 + 1/dy**2)
    print(f"2D Courant 수 S = {S:.4f}")

    # 스냅샷 기록
    snapshots = []
    record_steps = [50, 100, 150, 200, 250]

    # 메인 루프
    for n in range(n_steps):
        # H 업데이트
        Hx[:, :-1] = Hx[:, :-1] - dt / (mu[:, :-1] * dy) * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] = Hy[:-1, :] + dt / (mu[:-1, :] * dx) * (Ez[1:, :] - Ez[:-1, :])

        # 소스 (가우시안 펄스)
        t = n * dt
        t0 = 50 * dt
        tau = 20 * dt
        source = np.exp(-((t - t0) / tau) ** 2)

        # E 업데이트
        Ez[1:-1, 1:-1] = (Ca[1:-1, 1:-1] * Ez[1:-1, 1:-1] +
                         Cb[1:-1, 1:-1] * (
                             (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) / dx -
                             (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dy
                         ))

        # 소스 주입
        Ez[source_y, source_x] += source

        # 스냅샷 기록
        if n in record_steps:
            snapshots.append((n, Ez.copy()))

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    x = np.arange(Nx) * dx * 1000
    y = np.arange(Ny) * dy * 1000
    X, Y = np.meshgrid(x, y)

    for idx, (step, Ez_snap) in enumerate(snapshots):
        ax = axes[idx // 3, idx % 3]
        vmax = np.max(np.abs(Ez_snap)) * 0.8
        if vmax == 0:
            vmax = 1

        im = ax.pcolormesh(X, Y, Ez_snap, cmap='RdBu_r', shading='auto',
                          vmin=-vmax, vmax=vmax)
        ax.plot(source_x * dx * 1000, source_y * dy * 1000, 'k*', markersize=10)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'Step n = {step}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Ez')

    # 마지막 subplot
    ax = axes[1, 2]
    ax.text(0.5, 0.7, '2D FDTD TM 모드', fontsize=14, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.5, f'격자: {Nx} x {Ny}', fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, f'dx = dy = {dx*1000:.1f} mm', fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.3, f'Courant S = {S:.2f}', fontsize=11, ha='center', transform=ax.transAxes)
    ax.axis('off')

    plt.suptitle('2D FDTD 시뮬레이션 - 점 소스로부터 원형파 전파', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_2d_tm.png', dpi=150, bbox_inches='tight')
    plt.show()

    return Ez, Hx, Hy

# Ez, Hx, Hy = fdtd_2d_tm_mode()
```

---

## 4. Courant 조건

### 4.1 전자기파의 안정성 조건

```
Courant-Friedrichs-Lewy (CFL) 조건:

1D:
c·Δt/Δx ≤ 1

2D:
c·Δt·√(1/Δx² + 1/Δy²) ≤ 1

3D:
c·Δt·√(1/Δx² + 1/Δy² + 1/Δz²) ≤ 1

물리적 의미:
- 한 시간 단계에서 파동이 한 셀 이상 이동하면 안됨
- 수치적 정보 전파 속도 ≥ 물리적 파동 속도

등방성 격자 (Δx = Δy = Δz = Δ):
1D: Δt ≤ Δ/c
2D: Δt ≤ Δ/(c√2)
3D: Δt ≤ Δ/(c√3)
```

```python
def courant_condition_analysis():
    """Courant 조건 분석"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Courant 수에 따른 안정성
    ax1 = axes[0]

    # 1D FDTD 시뮬레이션 (다양한 Courant 수)
    def run_1d_fdtd(S, n_steps=200):
        Nx = 100
        dx = 1e-3
        dt = S * dx / c0

        Ey = np.zeros(Nx)
        Hz = np.zeros(Nx)

        Ca = dt / (mu0 * dx)
        Cb = dt / (eps0 * dx)

        max_values = []

        for n in range(n_steps):
            Hz[:-1] = Hz[:-1] - Ca * (Ey[1:] - Ey[:-1])

            if n == 0:
                Ey[Nx//4] = 1.0  # 초기 펄스

            Ey[1:] = Ey[1:] - Cb * (Hz[1:] - Hz[:-1])
            Ey[0] = Ey[-1] = 0

            max_values.append(np.max(np.abs(Ey)))

        return max_values

    courant_numbers = [0.5, 0.9, 1.0, 1.01, 1.1]
    colors = ['green', 'blue', 'orange', 'red', 'darkred']

    for S, color in zip(courant_numbers, colors):
        max_vals = run_1d_fdtd(S)
        label = f'S = {S}' + (' (안정)' if S <= 1 else ' (불안정)')
        ax1.semilogy(max_vals, color=color, linewidth=1.5, label=label)

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('max|Ey|')
    ax1.set_title('1D FDTD: Courant 수에 따른 안정성')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-3, 1e10)

    # (2) 분산 관계
    ax2 = axes[1]

    # FDTD 수치 분산 관계
    # ω_num = (2/Δt) arcsin(S sin(kΔx/2))
    # 해석해: ω = ck

    k_norm = np.linspace(0, np.pi, 100)  # kΔx

    for S in [0.5, 0.8, 1.0]:
        omega_exact = k_norm / S  # 정규화된 ωΔt
        omega_fdtd = 2 * np.arcsin(S * np.sin(k_norm / 2))

        # 위상 속도 비율
        vp_ratio = omega_fdtd / omega_exact

        ax2.plot(k_norm / np.pi, vp_ratio, linewidth=2, label=f'S = {S}')

    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Exact')
    ax2.set_xlabel(r'$k\Delta x / \pi$')
    ax2.set_ylabel(r'$v_p^{num} / c$')
    ax2.set_title('FDTD 수치 분산 (위상 속도 비)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.8, 1.05)

    plt.tight_layout()
    plt.savefig('courant_condition.png', dpi=150, bbox_inches='tight')
    plt.show()

# courant_condition_analysis()
```

### 4.2 수치 분산

```
FDTD 수치 분산:

1D 분산 관계:
sin(ωΔt/2) = S·sin(kΔx/2)

여기서 S = cΔt/Δx (Courant 수)

해석해: ω = ck (분산 없음)
FDTD: ω ≠ ck (수치 분산 발생)

문제점:
- 짧은 파장 (고주파)에서 분산 심각
- 파형 왜곡, 위상 오차

해결책:
- 파장당 최소 10-20 셀 사용
- 고차 차분 스킴 사용
- 분산 보정 기법
```

---

## 5. 재료 모델링

### 5.1 유전체와 도체

```
재료 특성 고려:

유전체 (ε > ε₀):
- 파동 속도 감소: v = c/√(εᵣ)
- 파장 감소: λ = λ₀/√(εᵣ)
- 격자 해상도 조정 필요

도체 (σ > 0):
- 전류 유도: J = σE
- 파동 감쇠
- 스킨 깊이: δ = √(2/(ωμσ))

손실 매질:
∂E/∂t 항에 -σE/ε 추가
```

```python
def material_modeling_fdtd():
    """재료 특성을 고려한 2D FDTD"""

    # 격자 설정
    Nx, Ny = 150, 100
    dx = dy = 1e-3

    # 시간 설정
    dt = 1 / (c0 * np.sqrt(1/dx**2 + 1/dy**2)) * 0.99
    n_steps = 400

    # 물성치 배열 초기화
    eps_r = np.ones((Ny, Nx))  # 상대 유전율
    sigma = np.zeros((Ny, Nx))  # 전도도

    # 유전체 블록 추가 (εᵣ = 4)
    eps_r[30:70, 80:100] = 4.0

    # 도체 블록 추가 (PEC 근사)
    sigma[40:60, 40:55] = 1e7  # 높은 전도도

    # 실제 유전율
    eps = eps0 * eps_r

    # 배열 초기화
    Ez = np.zeros((Ny, Nx))
    Hx = np.zeros((Ny, Nx))
    Hy = np.zeros((Ny, Nx))

    # 계수 (손실 포함)
    Ca = (1 - sigma * dt / (2 * eps)) / (1 + sigma * dt / (2 * eps))
    Cb = (dt / eps) / (1 + sigma * dt / (2 * eps))

    # 소스 위치
    source_x, source_y = 20, Ny // 2

    # 스냅샷
    snapshots = []
    record_steps = [50, 100, 200, 300]

    for n in range(n_steps):
        # H 업데이트
        Hx[:, :-1] = Hx[:, :-1] - dt / (mu0 * dy) * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] = Hy[:-1, :] + dt / (mu0 * dx) * (Ez[1:, :] - Ez[:-1, :])

        # 소스
        t = n * dt
        t0 = 50 * dt
        tau = 15 * dt
        freq = 5e9  # 5 GHz
        source = np.exp(-((t - t0) / tau) ** 2) * np.sin(2 * np.pi * freq * t)

        # E 업데이트
        Ez[1:-1, 1:-1] = (Ca[1:-1, 1:-1] * Ez[1:-1, 1:-1] +
                         Cb[1:-1, 1:-1] * (
                             (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) / dx -
                             (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dy
                         ))

        Ez[source_y, source_x] += source

        if n in record_steps:
            snapshots.append((n, Ez.copy()))

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(Nx) * dx * 1000
    y = np.arange(Ny) * dy * 1000
    X, Y = np.meshgrid(x, y)

    for idx, (step, Ez_snap) in enumerate(snapshots):
        ax = axes[idx // 2, idx % 2]
        vmax = np.max(np.abs(Ez_snap)) * 0.5
        if vmax == 0:
            vmax = 1

        im = ax.pcolormesh(X, Y, Ez_snap, cmap='RdBu_r', shading='auto',
                          vmin=-vmax, vmax=vmax)

        # 재료 영역 표시
        ax.contour(X, Y, eps_r, levels=[2], colors='green', linewidths=2)
        ax.contour(X, Y, sigma, levels=[1e6], colors='black', linewidths=2)

        ax.plot(source_x * dx * 1000, source_y * dy * 1000, 'r*', markersize=10)

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'Step n = {step}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Ez')

    # 범례
    axes[0, 0].plot([], [], 'g-', linewidth=2, label=r'유전체 ($\epsilon_r=4$)')
    axes[0, 0].plot([], [], 'k-', linewidth=2, label='도체 (PEC)')
    axes[0, 0].legend(loc='upper right')

    plt.suptitle('재료 특성을 고려한 2D FDTD', fontsize=14)
    plt.tight_layout()
    plt.savefig('fdtd_materials.png', dpi=150, bbox_inches='tight')
    plt.show()

    return Ez

# Ez = material_modeling_fdtd()
```

---

## 6. CEM 방법 비교

### 6.1 주요 수치 기법

```
전산 전자기학 (CEM) 방법들:

1. FDTD (Finite-Difference Time-Domain):
   - 시간 도메인 직접 해석
   - 광대역 특성 한 번에 계산
   - 단순한 구현, 병렬화 용이

2. FEM (Finite Element Method):
   - 복잡한 형상 처리 유리
   - 비구조 격자 사용
   - 고차 요소로 정확도 향상

3. MoM (Method of Moments):
   - 적분 방정식 기반
   - 개방 영역 문제에 적합
   - 안테나, 산란 문제

4. FIT (Finite Integration Technique):
   - 적분형 Maxwell 방정식
   - 에너지 보존 우수
   - 상용 소프트웨어 (CST)

5. FDFD (Finite-Difference Frequency-Domain):
   - 주파수 도메인 정상 상태
   - 특정 주파수 해석에 효율적
```

```python
def cem_methods_comparison():
    """CEM 방법 비교"""

    print("=" * 70)
    print("전산 전자기학 (CEM) 방법 비교")
    print("=" * 70)

    comparison = """
    ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
    │   방법      │    FDTD     │     FEM     │     MoM     │    FDFD     │
    ├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
    │ 도메인      │  시간       │ 시간/주파수  │   주파수    │   주파수    │
    │ 격자        │  구조       │  비구조     │   표면      │   구조      │
    │ 행렬        │  없음       │  희소       │   밀집      │   희소      │
    │ 광대역      │  효율적     │  다중 계산  │  다중 계산  │  다중 계산  │
    │ 비균질      │  용이       │  용이       │   어려움    │   용이      │
    │ 개방 영역   │  ABC/PML    │  무한 요소  │   자동      │   ABC       │
    │ 병렬화      │  매우 용이  │  용이       │   어려움    │   용이      │
    │ 비선형      │  가능       │  가능       │   어려움    │   어려움    │
    └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘

    응용 분야:

    FDTD:
    - 광소자, 안테나, EMC/EMI
    - 생체 전자기학
    - 레이더 단면적 (RCS)

    FEM:
    - 복잡한 형상의 도파관
    - 마이크로파 회로
    - 안테나 급전 구조

    MoM:
    - 와이어 안테나
    - 평면 마이크로스트립
    - 산란 문제

    상용 소프트웨어:
    - FDTD: Lumerical, XFdtd
    - FEM: ANSYS HFSS, COMSOL
    - MoM: FEKO, NEC
    - FIT: CST Studio
    """
    print(comparison)

cem_methods_comparison()
```

---

## 7. 연습 문제

### 연습 1: Maxwell 방정식
Faraday 법칙과 Ampère 법칙을 결합하여 자기장 B에 대한 파동 방정식을 유도하시오.

### 연습 2: 1D FDTD
1D FDTD 코드에서 매질 경계면(ε₁ → ε₂)에서의 반사와 투과를 시뮬레이션하시오. 반사 계수와 투과 계수를 Fresnel 공식과 비교하시오.

### 연습 3: Courant 조건
2D 등방성 격자에서 Courant 수 S = 0.5와 S = 1.0의 수치 분산을 비교하시오. 어느 경우가 더 정확한가?

### 연습 4: Yee 격자
3D Yee 격자에서 Ex 업데이트에 필요한 H 성분들의 위치를 도시하고, 업데이트 방정식을 작성하시오.

---

## 8. 참고자료

### 핵심 논문
- Yee (1966) "Numerical Solution of Initial Boundary Value Problems Involving Maxwell's Equations in Isotropic Media" - FDTD 원논문
- Taflove & Brodwin (1975) - 흡수 경계조건

### 교재
- Taflove & Hagness, "Computational Electrodynamics: The Finite-Difference Time-Domain Method"
- Sullivan, "Electromagnetic Simulation Using the FDTD Method"
- Jin, "The Finite Element Method in Electromagnetics"

### 오픈소스 도구
- MEEP (MIT, FDTD)
- gprMax (Ground Penetrating Radar)
- OpenEMS (FDTD + 회로)

---

## 요약

```
전자기학 수치해석 핵심:

1. Maxwell 방정식:
   - ∇×E = -∂B/∂t (Faraday)
   - ∇×H = J + ∂D/∂t (Ampère)
   - ∇·D = ρ (Gauss 전기)
   - ∇·B = 0 (Gauss 자기)

2. FDTD 핵심:
   - E와 H 공간/시간 엇갈림
   - Yee 격자 구조
   - Leapfrog 시간 전진

3. Courant 조건:
   1D: cΔt/Δx ≤ 1
   2D: cΔt√(1/Δx² + 1/Δy²) ≤ 1
   3D: cΔt√(1/Δx² + 1/Δy² + 1/Δz²) ≤ 1

4. 수치 분산:
   - 짧은 파장에서 심각
   - 파장당 10-20 셀 권장

5. 재료 모델링:
   - 유전체: ε = ε₀εᵣ
   - 도체: σ > 0
   - 손실 매질: Ca, Cb 계수 수정
```

---

다음 레슨에서는 FDTD의 상세 구현과 흡수 경계조건을 다룹니다.
