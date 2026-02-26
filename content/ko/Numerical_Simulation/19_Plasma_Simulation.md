# 19. 플라즈마 시뮬레이션 (Plasma Simulation)

## 학습 목표
- Particle-In-Cell (PIC) 방법의 기본 원리 이해
- 입자 푸시 (Boris 알고리즘) 구현
- 장 풀이 (Poisson 방정식)
- 입자-격자 보간
- 1D 정전기 PIC 시뮬레이션 구현
- Two-stream 불안정성 시뮬레이션

**이 레슨이 중요한 이유:** Particle-In-Cell (PIC) 방법은 유체 모델(속도 분포 함수를 평균화하는)과 완전한 운동론 이론(kinetic theory, 해석적으로 다루기 어려운) 사이의 간극을 연결합니다. 공간 격자 위에서 개별 입자를 추적함으로써, PIC은 유체 모델이 놓치는 비평형(non-equilibrium) 현상 -- 파동-입자 상호작용(wave-particle interaction), 란다우 감쇠(Landau damping), 플라즈마 불안정성(plasma instability) -- 을 포착합니다. PIC은 레이저-플라즈마 상호작용, 입자 가속기, 우주선 대전(spacecraft charging), 우주 플라즈마의 운동론적 난류(kinetic turbulence) 모델링의 주요 도구입니다.

---

## 1. PIC 방법 소개

### 1.1 개념과 원리

```
Particle-In-Cell (PIC) 방법:

핵심 아이디어:
- 플라즈마를 이산적인 "슈퍼 입자"로 표현
- 전자기장은 격자 위에서 계산
- 입자와 격자 사이 보간으로 결합

장점:
- 운동론적 효과 포착 (비평형, 파동-입자 상호작용)
- 상대론적 효과 쉽게 포함
- 복잡한 형상과 경계조건

단점:
- 계산 비용 (많은 입자 필요)
- 통계적 잡음 (유한 입자 수)
- 시간 단계 제한 (플라즈마 주파수)

역사:
- Buneman, Dawson (1960년대)
- Birdsall & Langdon (1991): 표준 교재
```

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# 물리 상수 (정규화 단위)
# 길이: Debye 길이 λD
# 시간: 플라즈마 주기 ωpe^-1
# 속도: 열속도 vth

def pic_introduction():
    """PIC 방법 소개"""

    print("=" * 60)
    print("Particle-In-Cell (PIC) 방법")
    print("=" * 60)

    intro = """
    PIC 알고리즘 사이클:

    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │     ┌─────────┐                    ┌─────────┐         │
    │     │ 입자    │ ──── 보간 ────→   │ 격자    │         │
    │     │ (x, v)  │ ← (전하/전류)     │ (ρ, J)  │         │
    │     └────┬────┘                    └────┬────┘         │
    │          │                              │               │
    │          │                              │ 장 풀이       │
    │   입자   │                              │ (Poisson/    │
    │   푸시   │                              │  Maxwell)    │
    │          │                              │               │
    │     ┌────┴────┐                    ┌────┴────┐         │
    │     │ 입자    │ ←─── 보간 ────    │ 격자    │         │
    │     │ (x, v)  │   (E, B)→가속     │ (E, B)  │         │
    │     └─────────┘                    └─────────┘         │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    시간 전진 (Leapfrog):
    - 위치: t^n → t^{n+1}  (x^{n+1} = x^n + v^{n+1/2}Δt)
    - 속도: t^{n-1/2} → t^{n+1/2}  (Boris 알고리즘)
    - 장: t^n (위치와 동기화)
    """
    print(intro)

pic_introduction()
```

### 1.2 PIC 시간/공간 스케일

```python
def pic_scales():
    """PIC 시뮬레이션 스케일"""

    print("=" * 60)
    print("PIC 시뮬레이션 스케일 조건")
    print("=" * 60)

    scales = """
    공간 해상도:
    Δx < λD (Debye 길이)
    - λD = √(ε₀kT/(n e²)) = vth/ωpe
    - Δx ~ λD 이면 비물리적 가열 발생

    시간 해상도:
    Δt < ωpe^{-1} (플라즈마 주기)
    - ωpe = √(ne²/(ε₀m))
    - Δt ~ 0.1-0.2 × ωpe^{-1} 권장

    CFL 조건:
    Δt × vmax < Δx
    - 정전기: vmax ~ 열속도
    - 전자기: vmax = c (광속)

    입자 수:
    N_ppc (particles per cell) > 50-100
    - 잡음 ~ 1/√N_ppc
    - 더 많을수록 잡음 감소

    시뮬레이션 박스:
    L > 여러 파장 (관심 현상)
    - 주기적 경계: 장파장 제한
    - 열린 경계: 반사 처리 필요
    """
    print(scales)

    # 스케일 관계 시각화
    fig, ax = plt.subplots(figsize=(10, 6))

    # 정규화 단위에서
    # 길이 단위: λD, 시간 단위: ωpe^-1

    n_cells = np.arange(1, 100)
    n_ppc = np.array([10, 50, 100, 500])

    for ppc in n_ppc:
        noise = 1 / np.sqrt(ppc * n_cells)
        ax.loglog(n_cells, noise * 100, linewidth=2, label=f'N_ppc = {ppc}')

    ax.axhline(y=1, color='red', linestyle='--', label='1% noise level')
    ax.set_xlabel('Number of cells')
    ax.set_ylabel('Relative noise [%]')
    ax.set_title('PIC 통계적 잡음 vs 셀 수 및 입자 밀도')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 100)

    plt.tight_layout()
    plt.savefig('pic_noise.png', dpi=150, bbox_inches='tight')
    plt.show()

# pic_scales()
```

---

## 2. 입자 푸시: Boris 알고리즘

Boris 알고리즘은 플라즈마 물리에서 가장 널리 사용되는 입자 적분기(particle integrator)이며, 그럴 만한 이유가 있습니다: 2차 정확도(second-order accurate)이고, 시간 역행 가능(time-reversible)하며, 순수 자기장(전기장 없음)에서 에너지를 정확히 보존합니다. 핵심 아이디어는 Lorentz 힘을 전기장 가속과 자기장 회전으로 분리하여, 각각을 적합한 스킴으로 처리하는 것입니다.

### 2.1 운동 방정식

Lorentz 힘 하에서의 하전 입자 운동 방정식은:

$$m\frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

자기력 $q\mathbf{v} \times \mathbf{B}$는 항상 속도에 수직이므로, 방향은 바꾸지만 속력은 바꾸지 않습니다 -- 이것이 Boris 회전 단계가 $|\mathbf{v}|$를 정확히 보존하도록 설계된 이유입니다. 전기력 $q\mathbf{E}$는 속력과 방향 모두를 변화시키며, 단순한 반단계 가속(half-step acceleration)으로 처리합니다.

```
하전 입자의 운동 방정식:

dx/dt = v

m dv/dt = q(E + v × B)

Boris 알고리즘 (1970):
- 시간 중심 (time-centered) 스킴
- 자기장에서 에너지 보존
- 2차 정확도

단계:
1. 반가속 (E에 의한): v⁻ = v^{n-1/2} + (qE/m)(Δt/2)
2. 회전 (B에 의한): v' → v⁺ (에너지 불변 회전)
3. 반가속: v^{n+1/2} = v⁺ + (qE/m)(Δt/2)
4. 위치 업데이트: x^{n+1} = x^n + v^{n+1/2}Δt
```

```python
def boris_pusher(x, v, E, B, q, m, dt):
    """
    Boris 입자 푸셔

    Parameters:
    - x, v: 입자 위치, 속도 (배열)
    - E, B: 입자 위치에서의 전기/자기장
    - q, m: 전하, 질량
    - dt: 시간 단계

    Returns:
    - x_new, v_new: 업데이트된 위치, 속도
    """
    # 편의를 위한 계수
    qmdt2 = q * dt / (2 * m)

    # 1. 반가속 (E)
    v_minus = v + qmdt2 * E

    # 2. 회전 (B)
    # t = (q/m)(B)(Δt/2)
    t = qmdt2 * B
    s = 2 * t / (1 + np.dot(t, t))

    # v' = v⁻ + v⁻ × t
    v_prime = v_minus + np.cross(v_minus, t)

    # v⁺ = v⁻ + v' × s
    v_plus = v_minus + np.cross(v_prime, s)

    # 3. 반가속 (E)
    v_new = v_plus + qmdt2 * E

    # 4. 위치 업데이트
    x_new = x + v_new * dt

    return x_new, v_new


def boris_demo():
    """Boris 알고리즘 시연: 균일 자기장에서의 입자 운동"""

    # 설정
    q = 1.0   # 전하
    m = 1.0   # 질량
    B0 = 1.0  # 자기장 강도 (z 방향)

    # 사이클로트론 주파수와 주기
    omega_c = q * B0 / m
    T_c = 2 * np.pi / omega_c

    # 시간 설정
    dt = 0.1  # T_c의 약 1/63
    n_steps = int(3 * T_c / dt)

    # 초기 조건
    x = np.array([0.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])  # x 방향 초기 속도
    E = np.array([0.0, 0.0, 0.0])  # 전기장 없음
    B = np.array([0.0, 0.0, B0])   # z 방향 자기장

    # 궤적 기록
    trajectory = [x.copy()]
    velocity = [v.copy()]

    for _ in range(n_steps):
        x, v = boris_pusher(x, v, E, B, q, m, dt)
        trajectory.append(x.copy())
        velocity.append(v.copy())

    trajectory = np.array(trajectory)
    velocity = np.array(velocity)

    # 시각화
    fig = plt.figure(figsize=(14, 5))

    # (1) x-y 평면 궤적
    ax1 = fig.add_subplot(131)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1.5)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='시작')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'r^', markersize=10, label='끝')

    # 해석해 (원)
    r_L = m * v[0] / (q * B0)  # Larmor 반경
    theta = np.linspace(0, 2*np.pi, 100)
    x_exact = r_L * np.sin(theta)
    y_exact = r_L * (1 - np.cos(theta))
    ax1.plot(x_exact, y_exact, 'k--', alpha=0.5, label='해석해')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('x-y 평면 궤적 (사이클로트론 운동)')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # (2) 에너지 보존
    ax2 = fig.add_subplot(132)
    KE = 0.5 * m * np.sum(velocity**2, axis=1)
    t = np.arange(len(KE)) * dt

    ax2.plot(t / T_c, KE / KE[0], 'b-', linewidth=1.5)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel(r't / $T_c$')
    ax2.set_ylabel(r'KE / KE$_0$')
    ax2.set_title('운동 에너지 보존')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.99, 1.01)

    # (3) 속도 성분
    ax3 = fig.add_subplot(133)
    ax3.plot(t / T_c, velocity[:, 0], 'r-', linewidth=1.5, label=r'$v_x$')
    ax3.plot(t / T_c, velocity[:, 1], 'b-', linewidth=1.5, label=r'$v_y$')
    ax3.set_xlabel(r't / $T_c$')
    ax3.set_ylabel('v')
    ax3.set_title('속도 성분')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('boris_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 정확도 분석
    print(f"\nBoris 알고리즘 분석:")
    print(f"  dt = {dt:.3f}, ωc·dt = {omega_c * dt:.3f}")
    print(f"  에너지 변화: {(KE[-1] / KE[0] - 1) * 100:.6f}%")
    print(f"  Larmor 반경: 이론 {r_L:.3f}, 시뮬 {np.max(trajectory[:, 0]):.3f}")

# boris_demo()
```

---

## 3. 장 풀이 (Field Solve)

### 3.1 Poisson 방정식

```
정전기 PIC에서의 장 풀이:

Poisson 방정식:
∇²φ = -ρ/ε₀

1D:
d²φ/dx² = -ρ/ε₀

이산화 (중심차분):
(φ_{i+1} - 2φ_i + φ_{i-1})/Δx² = -ρ_i/ε₀

행렬 형태:
A·φ = b

경계조건:
- 주기적: φ(0) = φ(L)
- Dirichlet: φ = 지정값
- Neumann: dφ/dn = 지정값

전기장:
E = -∇φ
E_i = -(φ_{i+1} - φ_{i-1})/(2Δx)
```

```python
def solve_poisson_1d_periodic(rho, dx, eps0=1.0):
    """
    1D Poisson 방정식 풀이 (주기적 경계조건)
    FFT 사용

    ∇²φ = -ρ/ε₀
    """
    Nx = len(rho)
    L = Nx * dx

    # FFT
    rho_k = fft(rho)

    # 파수
    k = fftfreq(Nx, dx) * 2 * np.pi

    # Poisson 풀이 (k=0 모드 제외)
    phi_k = np.zeros_like(rho_k, dtype=complex)
    phi_k[1:] = rho_k[1:] / (eps0 * k[1:]**2)
    phi_k[0] = 0  # k=0 모드 (평균 포텐셜 = 0)

    # 역 FFT
    phi = np.real(ifft(phi_k))

    return phi

def solve_poisson_1d_dirichlet(rho, dx, phi_left=0, phi_right=0, eps0=1.0):
    """
    1D Poisson 방정식 풀이 (Dirichlet 경계조건)
    삼중대각 행렬
    """
    Nx = len(rho)

    # 삼중대각 행렬 풀이 (Thomas 알고리즘)
    a = np.ones(Nx - 1)         # 하대각
    b = -2 * np.ones(Nx)        # 대각
    c = np.ones(Nx - 1)         # 상대각
    d = -rho * dx**2 / eps0     # 우변

    # 경계조건
    d[0] -= phi_left
    d[-1] -= phi_right

    # Forward sweep
    c_star = np.zeros(Nx - 1)
    d_star = np.zeros(Nx)

    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]

    for i in range(1, Nx - 1):
        c_star[i] = c[i] / (b[i] - a[i-1] * c_star[i-1])

    for i in range(1, Nx):
        d_star[i] = (d[i] - a[i-1] * d_star[i-1]) / (b[i] - a[i-1] * c_star[i-1] if i < Nx-1 else b[i] - a[i-1] * c_star[i-1])

    # Back substitution
    phi = np.zeros(Nx)
    phi[-1] = d_star[-1]

    for i in range(Nx - 2, -1, -1):
        phi[i] = d_star[i] - c_star[i] * phi[i+1]

    return phi

def electric_field_from_potential(phi, dx):
    """포텐셜에서 전기장 계산"""
    Nx = len(phi)
    E = np.zeros(Nx)

    # 중심차분 (주기적 경계조건)
    E[1:-1] = -(phi[2:] - phi[:-2]) / (2 * dx)
    E[0] = -(phi[1] - phi[-1]) / (2 * dx)
    E[-1] = -(phi[0] - phi[-2]) / (2 * dx)

    return E
```

---

## 4. 입자-격자 보간

### 4.1 전하 할당 (Charge Assignment)

```
입자에서 격자로 전하 할당:

1. NGP (Nearest Grid Point):
   - 가장 가까운 격자점에 전체 전하 할당
   - 1차 정확도, 불연속, 잡음 큼

2. CIC (Cloud-In-Cell) / Linear:
   - 선형 가중치로 두 인접 격자점에 분배
   - 가중치: W_i = 1 - |x - x_i|/Δx
   - 2차 정확도, 연속, 잡음 감소

3. TSC (Triangular-Shaped Cloud):
   - 2차 다항식 가중치
   - 3개 격자점에 분배
   - 더 매끄러움

4. Spline 보간:
   - 고차 B-spline
   - 더 많은 격자점 관여
```

```python
def charge_to_grid_cic(x_particles, q_particles, Nx, dx, L):
    """
    CIC (Cloud-In-Cell) 전하 할당

    Parameters:
    - x_particles: 입자 위치 배열
    - q_particles: 입자 전하 배열
    - Nx: 격자점 수
    - dx: 격자 간격
    - L: 도메인 크기

    Returns:
    - rho: 격자 위 전하밀도
    """
    rho = np.zeros(Nx)

    for x, q in zip(x_particles, q_particles):
        # 주기적 경계
        x = x % L

        # 왼쪽 격자점 인덱스
        i = int(x / dx)
        i_next = (i + 1) % Nx

        # CIC 가중치
        frac = (x / dx) - i
        w_left = 1 - frac
        w_right = frac

        # 전하 할당
        rho[i] += q * w_left / dx
        rho[i_next] += q * w_right / dx

    return rho

def field_to_particle_cic(E_grid, x_particle, dx, L):
    """
    CIC 장 보간 (격자 -> 입자)

    Parameters:
    - E_grid: 격자 위 전기장
    - x_particle: 입자 위치
    - dx: 격자 간격
    - L: 도메인 크기

    Returns:
    - E_particle: 입자 위치에서의 전기장
    """
    Nx = len(E_grid)
    x = x_particle % L

    i = int(x / dx)
    i_next = (i + 1) % Nx

    frac = (x / dx) - i
    w_left = 1 - frac
    w_right = frac

    E_particle = w_left * E_grid[i] + w_right * E_grid[i_next]

    return E_particle


def interpolation_demo():
    """보간 방법 시연"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) 전하 할당 비교
    ax1 = axes[0]

    Nx = 20
    dx = 1.0
    L = Nx * dx

    # 단일 입자
    x_particle = 5.3 * dx
    q = 1.0

    # NGP
    rho_ngp = np.zeros(Nx)
    i_ngp = int(round(x_particle / dx)) % Nx
    rho_ngp[i_ngp] = q / dx

    # CIC
    rho_cic = charge_to_grid_cic([x_particle], [q], Nx, dx, L)

    x_grid = np.arange(Nx) * dx

    ax1.bar(x_grid - 0.15, rho_ngp, width=0.3, label='NGP', alpha=0.7)
    ax1.bar(x_grid + 0.15, rho_cic, width=0.3, label='CIC', alpha=0.7)
    ax1.axvline(x=x_particle, color='red', linestyle='--', label='입자 위치')

    ax1.set_xlabel('x')
    ax1.set_ylabel(r'$\rho$')
    ax1.set_title('전하 할당: NGP vs CIC')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) 가중치 함수
    ax2 = axes[1]

    x = np.linspace(-2, 2, 200)

    # NGP
    w_ngp = np.where(np.abs(x) < 0.5, 1, 0)

    # CIC
    w_cic = np.where(np.abs(x) < 1, 1 - np.abs(x), 0)

    # TSC
    w_tsc = np.where(np.abs(x) < 0.5, 0.75 - x**2,
                    np.where(np.abs(x) < 1.5, 0.5 * (1.5 - np.abs(x))**2, 0))

    ax2.plot(x, w_ngp, 'b-', linewidth=2, label='NGP')
    ax2.plot(x, w_cic, 'r-', linewidth=2, label='CIC')
    ax2.plot(x, w_tsc, 'g-', linewidth=2, label='TSC')

    ax2.set_xlabel(r'$(x - x_i) / \Delta x$')
    ax2.set_ylabel('Weight W')
    ax2.set_title('보간 가중치 함수')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pic_interpolation.png', dpi=150, bbox_inches='tight')
    plt.show()

# interpolation_demo()
```

---

## 5. 1D 정전기 PIC 시뮬레이터

### 5.1 완전한 구현

```python
class PIC_1D_Electrostatic:
    """1D 정전기 PIC 시뮬레이터"""

    def __init__(self, Nx, L, dt, n_particles, species_params):
        """
        Parameters:
        - Nx: 격자점 수
        - L: 도메인 크기 (정규화: λD 단위)
        - dt: 시간 단계 (정규화: ωpe^-1 단위)
        - n_particles: 입자 수
        - species_params: 종별 파라미터 리스트
          [{'q': 전하, 'm': 질량, 'n': 입자 수, 'vth': 열속도}]
        """
        self.Nx = Nx
        self.L = L
        self.dx = L / Nx
        self.dt = dt

        # 격자
        self.x_grid = np.linspace(0, L - self.dx, Nx)
        self.rho = np.zeros(Nx)
        self.phi = np.zeros(Nx)
        self.E = np.zeros(Nx)

        # 입자 배열
        self.x = []      # 위치
        self.v = []      # 속도
        self.q = []      # 전하
        self.m = []      # 질량
        self.species = []  # 종 인덱스

        # 종별 입자 초기화
        for sp_idx, params in enumerate(species_params):
            q_sp = params['q']
            m_sp = params['m']
            n_sp = params['n']
            vth_sp = params.get('vth', 1.0)
            v_drift = params.get('v_drift', 0.0)

            # 균일 분포 위치
            x_sp = np.random.uniform(0, L, n_sp)

            # Maxwell 분포 속도 (drift 추가)
            v_sp = np.random.normal(v_drift, vth_sp, n_sp)

            self.x.extend(x_sp)
            self.v.extend(v_sp)
            self.q.extend([q_sp] * n_sp)
            self.m.extend([m_sp] * n_sp)
            self.species.extend([sp_idx] * n_sp)

        self.x = np.array(self.x)
        self.v = np.array(self.v)
        self.q = np.array(self.q)
        self.m = np.array(self.m)
        self.species = np.array(self.species)

        self.n_particles = len(self.x)

        # 슈퍼 입자 가중치 (전하 밀도 정규화)
        self.weight = L / self.n_particles

        print(f"PIC 1D 초기화:")
        print(f"  격자: Nx = {Nx}, dx = {self.dx:.4f}")
        print(f"  시간: dt = {dt}")
        print(f"  입자: N = {self.n_particles}")

    def deposit_charge(self):
        """전하 할당 (CIC)"""
        self.rho = np.zeros(self.Nx)

        for i in range(self.n_particles):
            x = self.x[i] % self.L
            q = self.q[i]

            j = int(x / self.dx)
            j_next = (j + 1) % self.Nx

            frac = (x / self.dx) - j
            w_left = 1 - frac
            w_right = frac

            self.rho[j] += q * w_left * self.weight / self.dx
            self.rho[j_next] += q * w_right * self.weight / self.dx

        # 배경 전하 (중성화)
        self.rho -= np.mean(self.rho)

    def solve_field(self):
        """Poisson 방정식 풀이 (FFT)"""
        self.phi = solve_poisson_1d_periodic(self.rho, self.dx)
        self.E = electric_field_from_potential(self.phi, self.dx)

    def interpolate_field(self):
        """장 보간 (격자 -> 입자)"""
        E_particles = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            x = self.x[i] % self.L
            j = int(x / self.dx)
            j_next = (j + 1) % self.Nx

            frac = (x / self.dx) - j
            w_left = 1 - frac
            w_right = frac

            E_particles[i] = w_left * self.E[j] + w_right * self.E[j_next]

        return E_particles

    def push_particles(self):
        """입자 푸시 (Leapfrog)"""
        E_p = self.interpolate_field()

        # 속도 업데이트: v^{n-1/2} -> v^{n+1/2}
        self.v += (self.q / self.m) * E_p * self.dt

        # 위치 업데이트: x^n -> x^{n+1}
        self.x += self.v * self.dt

        # 주기적 경계
        self.x = self.x % self.L

    def compute_diagnostics(self):
        """진단량 계산"""
        # 운동 에너지
        KE = 0.5 * np.sum(self.m * self.v**2) * self.weight

        # 전기장 에너지 (정전기)
        FE = 0.5 * np.sum(self.E**2) * self.dx

        # 총 에너지
        TE = KE + FE

        return {'KE': KE, 'FE': FE, 'TE': TE}

    def step(self):
        """한 시간 단계"""
        self.deposit_charge()
        self.solve_field()
        self.push_particles()

    def run(self, n_steps, diag_interval=10):
        """시뮬레이션 실행"""
        diagnostics = {'t': [], 'KE': [], 'FE': [], 'TE': []}

        for n in range(n_steps):
            if n % diag_interval == 0:
                diag = self.compute_diagnostics()
                diagnostics['t'].append(n * self.dt)
                diagnostics['KE'].append(diag['KE'])
                diagnostics['FE'].append(diag['FE'])
                diagnostics['TE'].append(diag['TE'])

            self.step()

        return {k: np.array(v) for k, v in diagnostics.items()}
```

### 5.2 Langmuir 파 테스트

```python
def langmuir_wave_test():
    """Langmuir 파 (전자 플라즈마 파) 테스트"""

    # 설정 (정규화 단위)
    Nx = 64
    L = 2 * np.pi * 4  # 4 파장
    dt = 0.1           # ωpe^-1 단위

    # 전자 (이온은 고정 배경)
    n_electrons = 10000
    vth = 1.0  # 열속도 (정규화)

    species = [
        {'q': -1.0, 'm': 1.0, 'n': n_electrons, 'vth': vth, 'v_drift': 0.0}
    ]

    # 시뮬레이터 생성
    pic = PIC_1D_Electrostatic(Nx, L, dt, n_electrons, species)

    # 초기 섭동 (밀도 파)
    k = 2 * np.pi / (L / 4)  # 파수 (4파장)
    amplitude = 0.01

    # 위치 섭동
    pic.x += amplitude * np.sin(k * pic.x) / k

    # 시뮬레이션 실행
    n_steps = 500
    diagnostics = pic.run(n_steps, diag_interval=5)

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (1) 에너지 시간 전개
    ax1 = axes[0, 0]
    t = diagnostics['t']
    ax1.plot(t, diagnostics['KE'], 'b-', label='운동 에너지')
    ax1.plot(t, diagnostics['FE'], 'r-', label='전기장 에너지')
    ax1.plot(t, diagnostics['TE'], 'k--', label='총 에너지')
    ax1.set_xlabel(r't [$\omega_{pe}^{-1}$]')
    ax1.set_ylabel('Energy')
    ax1.set_title('에너지 시간 전개')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) 전기장 에너지 (로그 스케일, 진동 확인)
    ax2 = axes[0, 1]
    ax2.semilogy(t, diagnostics['FE'], 'r-')
    ax2.set_xlabel(r't [$\omega_{pe}^{-1}$]')
    ax2.set_ylabel('Field Energy')
    ax2.set_title('전기장 에너지 (Langmuir 진동)')
    ax2.grid(True, alpha=0.3)

    # (3) 위상 공간
    ax3 = axes[1, 0]
    ax3.scatter(pic.x, pic.v, s=0.5, alpha=0.5)
    ax3.set_xlabel('x')
    ax3.set_ylabel('v')
    ax3.set_title(f'위상 공간 (t = {n_steps * dt:.1f})')
    ax3.grid(True, alpha=0.3)

    # (4) 전하 밀도
    ax4 = axes[1, 1]
    pic.deposit_charge()
    ax4.plot(pic.x_grid, pic.rho, 'b-')
    ax4.set_xlabel('x')
    ax4.set_ylabel(r'$\rho$')
    ax4.set_title('전하 밀도')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Langmuir 파 테스트', fontsize=14)
    plt.tight_layout()
    plt.savefig('langmuir_wave.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 진동수 분석
    from scipy.signal import find_peaks

    FE = diagnostics['FE']
    peaks, _ = find_peaks(FE)

    if len(peaks) > 1:
        T_measured = np.mean(np.diff(t[peaks]))
        omega_measured = 2 * np.pi / T_measured
        print(f"\n측정된 진동수: ω = {omega_measured:.3f} ωpe")
        print(f"이론값: ω ≈ ωpe = 1.0 (k→0 한계)")

# langmuir_wave_test()
```

---

## 6. Two-Stream 불안정성

### 6.1 물리적 배경

```
Two-Stream 불안정성:

설정:
- 두 전자 빔이 반대 방향으로 이동
- 이온은 고정 배경

분산 관계:
1 = ωpe²/2 × [1/(ω - kv₀)² + 1/(ω + kv₀)²]

불안정 조건:
k < kc = ωpe/v₀

성장률 (최대):
γmax ≈ √3/2 × ωpe (at k = ωpe/v₀)

물리적 결과:
- 지수적 장 성장
- 입자 트래핑 (위상 공간 와류)
- 열화 (thermalization)
```

```python
def two_stream_instability():
    """Two-Stream 불안정성 시뮬레이션"""

    # 설정
    Nx = 128
    L = 2 * np.pi * 8  # 8 파장
    dt = 0.1

    # 두 전자 빔
    n_per_beam = 20000
    vth = 0.1  # 작은 열속도
    v_drift = 3.0  # 드리프트 속도

    species = [
        # 빔 1 (오른쪽 이동)
        {'q': -1.0, 'm': 1.0, 'n': n_per_beam, 'vth': vth, 'v_drift': v_drift},
        # 빔 2 (왼쪽 이동)
        {'q': -1.0, 'm': 1.0, 'n': n_per_beam, 'vth': vth, 'v_drift': -v_drift}
    ]

    # 시뮬레이터 생성
    pic = PIC_1D_Electrostatic(Nx, L, dt, n_per_beam * 2, species)

    # 작은 초기 섭동
    np.random.seed(42)
    pic.x += 0.001 * np.random.randn(pic.n_particles)
    pic.x = pic.x % L

    # 시뮬레이션
    n_steps = 600
    diag_interval = 5

    # 위상 공간 스냅샷 저장
    snapshots = []
    snapshot_times = [0, 100, 200, 300, 400, 500]

    diagnostics = {'t': [], 'FE': []}

    for n in range(n_steps):
        if n % diag_interval == 0:
            pic.deposit_charge()
            pic.solve_field()
            FE = 0.5 * np.sum(pic.E**2) * pic.dx
            diagnostics['t'].append(n * dt)
            diagnostics['FE'].append(FE)

        if n in snapshot_times:
            snapshots.append({
                't': n * dt,
                'x': pic.x.copy(),
                'v': pic.v.copy()
            })

        pic.step()

    diagnostics = {k: np.array(v) for k, v in diagnostics.items()}

    # 시각화
    fig = plt.figure(figsize=(16, 12))

    # 위상 공간 스냅샷
    for i, snap in enumerate(snapshots):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.scatter(snap['x'], snap['v'], s=0.2, alpha=0.3, c='blue')
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        ax.set_title(f"t = {snap['t']:.0f}")
        ax.set_xlim(0, L)
        ax.set_ylim(-8, 8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Two-Stream 불안정성: 위상 공간 전개', fontsize=14)
    plt.tight_layout()
    plt.savefig('two_stream_phase_space.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 에너지 및 성장률
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 전기장 에너지 (로그)
    ax1 = axes[0]
    t = diagnostics['t']
    FE = diagnostics['FE']

    ax1.semilogy(t, FE, 'b-', linewidth=1.5)

    # 선형 성장 영역 피팅
    linear_region = (t > 10) & (t < 35)
    if np.any(linear_region) and np.any(FE[linear_region] > 0):
        log_FE = np.log(FE[linear_region])
        t_fit = t[linear_region]
        coeffs = np.polyfit(t_fit, log_FE, 1)
        gamma_measured = coeffs[0] / 2  # FE ∝ exp(2γt)

        ax1.semilogy(t_fit, np.exp(np.polyval(coeffs, t_fit)), 'r--',
                    linewidth=2, label=f'Fit: γ = {gamma_measured:.3f}')
        ax1.legend()

    ax1.set_xlabel(r't [$\omega_{pe}^{-1}$]')
    ax1.set_ylabel('Field Energy')
    ax1.set_title('전기장 에너지 성장')
    ax1.grid(True, alpha=0.3)

    # 이론적 성장률
    ax2 = axes[1]

    k = np.linspace(0.01, 2, 100)
    # 근사 분산 관계에서의 성장률 (최대 성장)
    # γ ≈ ωpe × √(3)/2 × (k v₀/ωpe)^(1/3) for small k
    gamma_theory = 0.866  # √3/2

    ax2.text(0.5, 0.7, f'이론적 최대 성장률:\nγmax ≈ (√3/2)ωpe ≈ {gamma_theory:.3f}',
            transform=ax2.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat'))

    ax2.text(0.5, 0.4, f'측정된 성장률:\nγ ≈ {gamma_measured:.3f}' if 'gamma_measured' in dir() else '',
            transform=ax2.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax2.text(0.1, 0.1, """
Two-Stream 불안정성 특성:
1. 선형 단계: 지수적 성장
2. 비선형 단계: 입자 트래핑
3. 포화 단계: 열화 (thermalization)
    """, transform=ax2.transAxes, fontsize=10, verticalalignment='bottom')

    ax2.axis('off')
    ax2.set_title('불안정성 분석')

    plt.tight_layout()
    plt.savefig('two_stream_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    return pic, diagnostics

# pic, diagnostics = two_stream_instability()
```

---

## 7. 고급 주제

### 7.1 전자기 PIC

```
전자기 PIC 확장:

추가 물리:
- 자기장 (B) 고려
- 전체 Maxwell 방정식 풀이
- 상대론적 입자 운동

Maxwell 방정식:
∂B/∂t = -∇×E
∂E/∂t = c²∇×B - J/ε₀

전류 할당:
J = Σᵢ qᵢvᵢ × (보간 가중치)

시간 전진:
- E, B: FDTD 유사
- 입자: 상대론적 Boris 알고리즘

응용:
- 레이저-플라즈마 상호작용
- 상대론적 충격파
- 입자 가속
```

```python
def em_pic_overview():
    """전자기 PIC 개요"""

    print("=" * 60)
    print("전자기 PIC (Electromagnetic PIC)")
    print("=" * 60)

    overview = """
    전자기 PIC 알고리즘:

    ┌─────────────────────────────────────────────────────────┐
    │ 시간 단계 n → n+1                                       │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ 1. B^n → B^{n+1/2} (반단계)                             │
    │    B^{n+1/2} = B^n - (Δt/2)∇×E^n                        │
    │                                                         │
    │ 2. 입자 푸시: (x^n, v^{n-1/2}) → (x^{n+1}, v^{n+1/2})  │
    │    - 보간: E^n, B^{n+1/2} → 입자                        │
    │    - Boris: 가속 및 회전                                │
    │    - 위치 업데이트                                      │
    │                                                         │
    │ 3. 전류 할당: J^{n+1/2}                                 │
    │    Esirkepov (전하 보존) 또는 Villasenor-Buneman       │
    │                                                         │
    │ 4. E 업데이트: E^n → E^{n+1}                            │
    │    E^{n+1} = E^n + Δt(c²∇×B^{n+1/2} - J^{n+1/2}/ε₀)    │
    │                                                         │
    │ 5. B^{n+1/2} → B^{n+1} (반단계)                         │
    │    B^{n+1} = B^{n+1/2} - (Δt/2)∇×E^{n+1}               │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    코드 예시:
    - EPOCH (영국)
    - OSIRIS (UCLA/IST)
    - SMILEI (프랑스)
    - WarpX (LBNL)
    """
    print(overview)

em_pic_overview()
```

---

## 8. 연습 문제

### 연습 1: Boris 알고리즘(Boris Algorithm) - E×B 드리프트
균일한 전기장 E = E₀x̂와 자기장 B = B₀ẑ에서 Boris 알고리즘으로 입자 운동을 시뮬레이션하시오. E×B 드리프트를 수치적으로 확인하시오.

<details><summary>정답 보기</summary>

E×B 드리프트 이론값: v_drift = E×B/B² = E₀/B₀ (y방향)

```python
import numpy as np
import matplotlib.pyplot as plt

def boris_ExB_drift():
    """Boris 알고리즘으로 E×B 드리프트 시뮬레이션"""
    # 물리 파라미터
    q = 1.0;  m = 1.0    # 전하, 질량
    E0 = 0.1;  B0 = 1.0  # 전기장, 자기장 크기

    # E×B 드리프트 이론값
    v_drift_theory = E0 / B0
    print(f"E×B 드리프트 이론값: v_y = E₀/B₀ = {v_drift_theory:.4f}")

    # 초기 조건
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])

    # 시뮬레이션 파라미터
    dt = 0.01; n_steps = 2000
    omega_c = q * B0 / m  # 사이클로트론 주파수

    positions = [pos.copy()]
    velocities = [vel.copy()]

    for _ in range(n_steps):
        # Boris 알고리즘
        E = np.array([E0, 0.0, 0.0])
        B = np.array([0.0, 0.0, B0])

        # 절반 전기장 킥
        v_minus = vel + (q/m) * E * dt/2

        # 자기장 회전
        t = (q/m) * B * dt/2
        s = 2 * t / (1 + np.dot(t, t))
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus  = v_minus + np.cross(v_prime, s)

        # 나머지 절반 전기장 킥
        vel = v_plus + (q/m) * E * dt/2
        pos = pos + vel * dt

        positions.append(pos.copy())
        velocities.append(vel.copy())

    positions  = np.array(positions)
    velocities = np.array(velocities)

    # 드리프트 속도 측정 (장기 평균)
    v_y_avg = np.mean(velocities[500:, 1])
    print(f"시뮬레이션 E×B 드리프트: v_y ≈ {v_y_avg:.4f}")
    print(f"오차: {abs(v_y_avg - v_drift_theory)/v_drift_theory * 100:.2f}%")

    plt.figure(figsize=(8, 6))
    plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=0.5)
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'E×B 드리프트 궤적 (드리프트 방향: y, v_drift={v_drift_theory:.2f})')
    plt.grid(True); plt.axis('equal')
    plt.tight_layout()
    plt.savefig('ExB_drift.png', dpi=150)
    plt.close()

boris_ExB_drift()
```

Boris 알고리즘의 핵심 장점은 자기 모멘트(magnetic moment) μ = mv²_⊥/(2B)를 보존하는 심플렉틱(symplectic) 구조입니다. E×B 드리프트는 전하 부호와 무관하며, 전자와 이온이 같은 방향으로 드리프트합니다. 이것이 E×B 드리프트가 전류를 발생시키지 않는 이유입니다.
</details>

### 연습 2: CIC vs NGP 전하 할당 비교
같은 입자 분포에 대해 CIC(Cloud-In-Cell)와 NGP(Nearest-Grid-Point) 전하 할당을 비교하시오. 어느 방법이 더 매끄러운 전하 밀도를 제공하는가?

<details><summary>정답 보기</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def charge_assignment_comparison():
    """CIC vs NGP 전하 할당 비교"""
    N_grid = 64    # 격자 점 수
    N_part = 1000  # 입자 수
    L = 1.0        # 도메인 길이
    dx = L / N_grid

    # 임의 입자 위치 (가우시안 분포)
    np.random.seed(42)
    x_part = np.random.normal(L/2, L/10, N_part) % L

    # NGP (Nearest-Grid-Point): 가장 가까운 격자점에 전하 할당
    def ngp_assign(x_part, N_grid, dx, L):
        rho = np.zeros(N_grid)
        for xp in x_part:
            j = int(xp / dx) % N_grid
            rho[j] += 1.0
        return rho / dx

    # CIC (Cloud-In-Cell): 선형 보간으로 인접 격자점에 분배
    def cic_assign(x_part, N_grid, dx, L):
        rho = np.zeros(N_grid)
        for xp in x_part:
            j = int(xp / dx) % N_grid
            frac = (xp / dx) - int(xp / dx)  # 소수 부분
            rho[j]           += (1 - frac)
            rho[(j+1) % N_grid] += frac
        return rho / dx

    rho_ngp = ngp_assign(x_part, N_grid, dx, L)
    rho_cic = cic_assign(x_part, N_grid, dx, L)

    # 이론적 분포 (가우시안)
    x_grid = np.linspace(0, L, N_grid, endpoint=False)
    from scipy.stats import norm
    rho_theory = N_part * norm.pdf(x_grid, L/2, L/10)

    # 고주파 노이즈 비교
    noise_ngp = np.std(rho_ngp - np.convolve(rho_ngp, np.ones(5)/5, mode='same'))
    noise_cic = np.std(rho_cic - np.convolve(rho_cic, np.ones(5)/5, mode='same'))

    plt.figure(figsize=(10, 4))
    plt.plot(x_grid, rho_theory, 'k-', linewidth=2, label='이론 (가우시안)')
    plt.plot(x_grid, rho_ngp, 'r-', alpha=0.7, label=f'NGP (노이즈={noise_ngp:.2f})')
    plt.plot(x_grid, rho_cic, 'b-', alpha=0.7, label=f'CIC (노이즈={noise_cic:.2f})')
    plt.xlabel('x'); plt.ylabel('전하 밀도 ρ')
    plt.title('CIC vs NGP 전하 할당')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig('charge_assignment.png', dpi=150)
    plt.close()

    print(f"NGP 고주파 노이즈: {noise_ngp:.2f}")
    print(f"CIC 고주파 노이즈: {noise_cic:.2f}")
    print(f"CIC가 NGP보다 약 {noise_ngp/noise_cic:.1f}배 더 매끄럽습니다")

charge_assignment_comparison()
```

CIC는 NGP보다 훨씬 매끄러운 전하 밀도를 제공합니다. NGP는 입자가 격자점을 지날 때 격자력(grid heating)이 발생하는 반면, CIC는 연속적인 전하 분배로 이를 완화합니다. 더 높은 차수의 형상 함수(B-spline)를 사용하면 더 매끄러운 결과를 얻을 수 있지만 계산 비용이 증가합니다.
</details>

### 연습 3: 맥스웰 분포 열평형(Thermal Equilibrium) 검증
단일 종 플라즈마 PIC 시뮬레이션에서 초기 맥스웰(Maxwell) 속도 분포가 유지되는지 확인하시오. 수치적 가열(numerical heating)이 발생하는지 조사하시오.

<details><summary>정답 보기</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def thermal_equilibrium_test():
    """PIC 시뮬레이션에서 맥스웰 분포 유지 검증"""
    N_part = 10000; N_grid = 128
    vth = 1.0    # 열속도 (정규화)
    L = 10.0; dx = L / N_grid; dt = 0.1

    np.random.seed(42)
    # 초기 맥스웰 속도 분포
    vx = np.random.normal(0, vth, N_part)
    x  = np.random.uniform(0, L, N_part)

    T_initial = np.mean(vx**2)  # 초기 온도 (kT/m 단위)

    n_steps = 100
    T_history = [T_initial]

    for step in range(n_steps):
        # 간략화된 PIC: 전기장 없이 자유 스트리밍만
        x = (x + vx * dt) % L
        T_current = np.mean(vx**2)
        T_history.append(T_current)

    T_final = T_history[-1]
    heating = (T_final - T_initial) / T_initial * 100

    # 속도 분포 비교 (초기 vs 최종)
    v_bins = np.linspace(-4*vth, 4*vth, 50)
    v_theory = (np.exp(-v_bins**2/(2*vth**2)) /
                (np.sqrt(2*np.pi) * vth) * N_part * (v_bins[1]-v_bins[0]))

    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.hist(vx, bins=v_bins, alpha=0.7, label='시뮬레이션')
    plt.plot(v_bins, v_theory, 'r-', linewidth=2, label='맥스웰 이론')
    plt.xlabel('vx'); plt.ylabel('입자 수')
    plt.title('속도 분포 (t=최종)'); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(T_history, 'b-')
    plt.axhline(T_initial, color='r', linestyle='--', label=f'초기 T={T_initial:.4f}')
    plt.xlabel('시간 스텝'); plt.ylabel('온도 T = <vx²>')
    plt.title(f'수치 가열: {heating:.2f}%'); plt.legend()
    plt.tight_layout()
    plt.savefig('thermal_equilibrium.png', dpi=150)
    plt.close()

    print(f"초기 온도: T = {T_initial:.4f}")
    print(f"최종 온도: T = {T_final:.4f}")
    print(f"수치 가열: {heating:.2f}%")
    print("\n수치 가열 원인:")
    print("1. 유한 격자 크기 → 격자력(aliasing)")
    print("2. 유한 시간 스텝 → 위상 오차")
    print("3. 적은 입자 수 → 통계 노이즈")

thermal_equilibrium_test()
```

자유 스트리밍(전기장 없음)에서는 수치 가열이 최소화됩니다. 실제 PIC에서는 포아송 방정식 풀기 → 전기장 계산 → 입자 밀기 과정에서 격자 주파수 이상의 파동이 에너지를 임의로 흡수하여 수치 가열이 발생합니다. 이를 줄이기 위해 입자당 디바이 길이 해상도 λ_D/dx ≥ 1을 유지해야 합니다.
</details>

### 연습 4: 두 빔 불안정성(Two-Stream Instability) 성장률 측정
두 빔 불안정성(Two-stream instability) 시뮬레이션에서 드리프트 속도 v₀를 변화시키며 성장률 γ를 측정하시오. 이론값과 비교하시오.

<details><summary>정답 보기</summary>

두 빔 불안정성 이론 성장률: γ_max ≈ ω_p/2√2 (비상대론적, 동일 밀도 두 빔)

```python
import numpy as np
import matplotlib.pyplot as plt

def two_stream_growth_rate():
    """두 빔 불안정성 성장률 측정"""
    N_part = 5000; N_grid = 64
    L = 10.0; dx = L / N_grid; dt = 0.05
    omega_p = 1.0  # 플라즈마 주파수 (정규화)

    # 이론 성장률
    gamma_theory = omega_p / (2 * np.sqrt(2))
    print(f"이론 성장률: γ_max = ω_p/(2√2) = {gamma_theory:.4f}")

    def simulate_two_stream(v0, n_steps=500):
        np.random.seed(42)
        vth = 0.1 * v0  # 열속도 (작게)
        # 두 빔: +v0와 -v0
        x   = np.random.uniform(0, L, N_part)
        vx1 = np.random.normal( v0, vth, N_part//2)
        vx2 = np.random.normal(-v0, vth, N_part//2)
        vx  = np.concatenate([vx1, vx2])

        # 전기장 에너지 이력 (불안정성 성장 지표)
        E_history = []

        for step in range(n_steps):
            # 전하 밀도 (CIC)
            rho = np.zeros(N_grid)
            for i, xi in enumerate(x):
                j = int(xi/dx) % N_grid
                frac = (xi/dx) - int(xi/dx)
                rho[j]           += 1 - frac
                rho[(j+1)%N_grid] += frac
            rho = rho / N_part * N_grid - 1.0  # 배경 차감

            # 전기장 (FFT 포아송)
            rho_hat = np.fft.rfft(rho)
            k = np.fft.rfftfreq(N_grid, d=dx/(2*np.pi))
            k[0] = 1e-10  # DC 방지
            E_hat = -1j * rho_hat / k
            Ex = np.fft.irfft(E_hat)

            # 입자 밀기
            for i, xi in enumerate(x):
                j = int(xi/dx) % N_grid
                frac = (xi/dx) - int(xi/dx)
                E_at_p = (1-frac)*Ex[j] + frac*Ex[(j+1)%N_grid]
                vx[i] += (omega_p**2 * dx) * E_at_p * dt
                x[i]   = (xi + vx[i] * dt) % L

            E_field_energy = np.sum(Ex**2) * dx / 2
            E_history.append(E_field_energy)

        return np.array(E_history)

    v0_values = [1.0, 2.0, 3.0]
    plt.figure(figsize=(10, 4))
    for v0 in v0_values:
        E_hist = simulate_two_stream(v0, n_steps=200)
        t = np.arange(len(E_hist)) * dt

        # 성장 구간에서 지수 성장률 추정
        try:
            log_E = np.log(E_hist[10:100])
            slope, _ = np.polyfit(t[10:100], log_E, 1)
            gamma_meas = slope / 2  # 에너지 → 진폭 변환
        except:
            gamma_meas = float('nan')

        plt.semilogy(t, E_hist, label=f'v₀={v0} (γ≈{gamma_meas:.3f})')
        print(f"v₀ = {v0:.1f}: 측정 성장률 γ ≈ {gamma_meas:.4f}, "
              f"이론 γ = {gamma_theory:.4f}")

    plt.xlabel('시간 t'); plt.ylabel('전기장 에너지')
    plt.title('두 빔 불안정성 성장')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig('two_stream_growth.png', dpi=150)
    plt.close()

two_stream_growth_rate()
```

두 빔 불안정성은 서로 반대 방향으로 이동하는 두 전자 빔의 상대 드리프트 에너지가 정전기 파동의 성장으로 전환되는 현상입니다. 이론 성장률 γ_max = ω_p/(2√2)는 드리프트 속도 v₀와 무관하게 일정하지만, 불안정한 파수 범위는 v₀에 따라 달라집니다. PIC 시뮬레이션에서 측정된 성장률이 이론값과 잘 일치하면 코드가 올바르게 구현된 것을 의미합니다.
</details>

---

## 9. 참고자료

### 핵심 교재
- Birdsall & Langdon, "Plasma Physics via Computer Simulation" (표준 교재)
- Hockney & Eastwood, "Computer Simulation Using Particles"
- Arber et al., "Contemporary Particle-In-Cell Approach to Laser-Plasma Modelling"

### PIC 코드
- EPOCH: 레이저 플라즈마
- OSIRIS: 고성능, 상대론적
- SMILEI: 모듈식, 오픈소스
- WarpX: GPU 가속, Exascale

### 온라인 자료
- Plasma Theory Group resources
- UCLA PICKSC 튜토리얼
- LBNL WarpX 문서

---

## 요약

```
PIC 시뮬레이션 핵심:

1. 알고리즘 사이클:
   입자 → 전하/전류 → 장 풀이 → 보간 → 입자 푸시

2. 입자 푸시 (Boris):
   - 반가속 → 회전 → 반가속
   - 에너지 보존, 2차 정확도

3. 장 풀이:
   - 정전기: Poisson (∇²φ = -ρ/ε₀)
   - 전자기: Maxwell (FDTD 유사)

4. 보간:
   - NGP: 0차, 불연속
   - CIC: 1차, 선형
   - TSC: 2차, 매끄러움

5. 스케일 조건:
   - Δx < λD
   - Δt < ωpe^-1
   - N_ppc > 50-100

6. 검증 테스트:
   - Langmuir 파 (플라즈마 진동)
   - Two-stream 불안정성
   - 입자 드리프트

7. 진단:
   - 에너지 보존
   - 위상 공간 분포
   - 분산 관계
```

---

## 연습 문제

### 연습 1: Boris 알고리즘(Boris Algorithm)으로 E×B 드리프트(Drift) 검증
boris_pusher 함수를 사용하여 수직 균일 자기장에서 양성자(proton, 전하 q = +e, 질량 mp = 1.673×10⁻²⁷ kg) 운동을 시뮬레이션하세요: x 방향 전기장 E = 1000 V/m, z 방향 자기장 B = 0.01 T. 이론적 E×B 드리프트 속도는 전하와 질량에 무관하게 y 방향으로 vd = E/B입니다. 10 사이클로트론 주기(cyclotron periods) 동안 시뮬레이션을 실행하고, 안내 중심(guiding center) 궤적을 기록하여 드리프트 속도가 이론값 vd = E×B / B² = E/B와 일치함을 검증하세요. 또한 순수 자기장(전기장 없음)에서 Boris 알고리즘이 운동 에너지(kinetic energy)를 보존함을 확인하세요.

### 연습 2: CIC(Cloud-In-Cell)와 NGP(Nearest Grid Point) 전하 밀도 비교
100개 격자 셀의 1D 도메인에 열속도 vth = 1인 맥스웰-볼츠만(Maxwell-Boltzmann) 속도 분포를 갖는 전자 10,000개를 초기화하세요. NGP와 CIC 보간(interpolation) 방식을 모두 사용하여 격자 위의 전하 밀도를 계산하세요. 두 밀도 프로파일을 같은 축에 그리고 각 방법의 고주파 잡음 수준(표준 편차 대비 평균 밀도)을 계산하세요. 동일한 입자 수에 대해 CIC가 NGP보다 약 √(N_cell)배만큼 잡음을 줄임을 보이세요.

### 연습 3: Langmuir 파(Langmuir Wave) 주파수 측정
Nx = 64 셀, L = 4λD (Debye 길이), N = 10,000 전자, 파수(wavenumber) k = 2π/L에서 진폭 δn/n = 0.01인 정현파(sinusoidal) 밀도 섭동을 갖는 1D 정전기(electrostatic) PIC 시뮬레이션을 초기화하세요. 정전기 필드 에너지를 시간의 함수로 기록하세요. 필드 에너지 시계열의 FFT를 사용하여 진동 주파수 ωpe를 측정하세요. 측정된 주파수를 정규화 단위(normalized units, ωpe = 1)에서의 이론적 플라즈마 주파수(plasma frequency) ωpe = √(ne²/(ε₀me))와 비교하세요. Bohm-Gross 분산 관계(Bohm-Gross dispersion) ω² = ωpe²(1 + 3k²λD²)에서 ωpe보다 10% 보정이 발생하는 파수 k는 얼마인가요?

### 연습 4: 이중 스트림 불안정성(Two-Stream Instability) 성장률 측정
표류 속도(drift velocity) v₀ ∈ {1.0, 2.0, 3.0, 5.0} (열속도로 정규화)로 이중 스트림 불안정성(two-stream instability) 시뮬레이션을 실행하세요. 각 v₀에 대해 초기 지수적 성장 단계 동안 필드 에너지에 지수 함수를 피팅(E_field ∝ exp(2γt))하여 선형 성장률(linear growth rate) γ를 측정하세요. 측정된 γ를 v₀의 함수로 그리고 이중 빔 분산 관계(two-beam dispersion relation)에서 얻은 이론적 최대 성장률 γmax ≈ (√3/2)ωpe와 비교하세요. 어느 v₀에서 성장률이 포화(saturate)되며, 이를 유발하는 물리적 효과는 무엇인가요?

### 연습 5: 입자 수(Particle Number)와 수치 가열(Numerical Heating)
단일 종(single-species) 맥스웰-볼츠만 플라즈마(불안정성 유발 없음)에 대해 N_ppc ∈ {10, 50, 200} 셀당 입자 수로 500 플라즈마 주기(plasma periods) 동안 세 가지 PIC 시뮬레이션을 실행하세요. 이상적인 시뮬레이션에서는 수치 가열(numerical heating)이 없을 때 총 운동 에너지가 일정하게 유지되어야 합니다. 세 경우 모두에 대해 시간에 따른 총 운동 에너지를 그려보세요. 수치 가열률(단위 플라즈마 주기당 에너지 증가)을 측정하고 이것이 1/N_ppc에 비례하여 감소함을 검증하세요. 500 주기 동안 수치 가열을 초기 에너지의 1% 미만으로 유지하려면 최소 N_ppc는 얼마여야 하나요?

---

이것으로 수치 시뮬레이션 시리즈의 CFD, 전자기학, MHD, 플라즈마 주제를 마무리합니다.
