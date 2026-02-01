# 18. MHD 수치해법 (MHD Numerical Methods)

## 학습 목표
- MHD 방정식의 보존 형태 이해
- 유한 체적법 기초 학습
- Godunov 유형 스킴 이해
- MHD Riemann 문제 파악
- 간단한 MHD 충격파 관 문제 구현
- div B = 0 제약조건 처리 방법

---

## 1. MHD 방정식의 보존 형태

### 1.1 1D MHD 보존 형태

```
1D 이상 MHD 보존 형태:

∂U/∂t + ∂F/∂x = 0

보존 변수 U:
    ⎡ ρ     ⎤  (밀도)
    ⎢ ρvx   ⎥  (x-운동량)
    ⎢ ρvy   ⎥  (y-운동량)
U = ⎢ ρvz   ⎥  (z-운동량)
    ⎢ By    ⎥  (y-자기장)
    ⎢ Bz    ⎥  (z-자기장)
    ⎣ E     ⎦  (총 에너지)

(Bx = const, 1D에서)

플럭스 F:
    ⎡ ρvx                           ⎤
    ⎢ ρvx² + p* - Bx²/μ₀            ⎥
    ⎢ ρvxvy - BxBy/μ₀               ⎥
F = ⎢ ρvxvz - BxBz/μ₀               ⎥
    ⎢ vxBy - vyBx                   ⎥
    ⎢ vxBz - vzBx                   ⎥
    ⎣ (E + p*)vx - Bx(v·B)/μ₀       ⎦

여기서:
p* = p + B²/2μ₀  (총 압력)
E = p/(γ-1) + ρv²/2 + B²/2μ₀  (총 에너지)
```

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# 물리 상수 (코드 단위: μ₀ = 1)
gamma = 5/3  # 비열비

def primitive_to_conservative(rho, vx, vy, vz, Bx, By, Bz, p):
    """원시변수 -> 보존변수 변환"""
    E = p / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2 + vz**2) + \
        0.5 * (Bx**2 + By**2 + Bz**2)

    U = np.array([rho, rho*vx, rho*vy, rho*vz, By, Bz, E])
    return U

def conservative_to_primitive(U, Bx):
    """보존변수 -> 원시변수 변환"""
    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho
    vz = U[3] / rho
    By = U[4]
    Bz = U[5]
    E = U[6]

    p = (gamma - 1) * (E - 0.5 * rho * (vx**2 + vy**2 + vz**2) -
                       0.5 * (Bx**2 + By**2 + Bz**2))

    return rho, vx, vy, vz, Bx, By, Bz, p

def compute_flux(U, Bx):
    """플럭스 계산"""
    rho, vx, vy, vz, _, By, Bz, p = conservative_to_primitive(U, Bx)

    B2 = Bx**2 + By**2 + Bz**2
    p_star = p + 0.5 * B2  # 총 압력
    E = U[6]
    vB = vx * Bx + vy * By + vz * Bz

    F = np.array([
        rho * vx,
        rho * vx**2 + p_star - Bx**2,
        rho * vx * vy - Bx * By,
        rho * vx * vz - Bx * Bz,
        vx * By - vy * Bx,
        vx * Bz - vz * Bx,
        (E + p_star) * vx - Bx * vB
    ])

    return F

def mhd_wave_speeds(rho, vx, p, Bx, By, Bz):
    """MHD 파동 속도 계산"""
    B2 = Bx**2 + By**2 + Bz**2
    cs2 = gamma * p / rho  # 음속 제곱
    ca2 = B2 / rho         # Alfven 속도 제곱
    cax2 = Bx**2 / rho     # x-방향 Alfven

    # 빠른/느린 자기음파
    term1 = 0.5 * (cs2 + ca2)
    term2 = 0.5 * np.sqrt((cs2 + ca2)**2 - 4 * cs2 * cax2)

    cf = np.sqrt(term1 + term2)  # Fast
    ca = np.sqrt(cax2)           # Alfven
    cs = np.sqrt(max(term1 - term2, 0))  # Slow

    return cf, ca, cs

print("=" * 60)
print("1D MHD 보존 형태")
print("=" * 60)

# 예제: 초기 상태
rho = 1.0
vx, vy, vz = 0.0, 0.0, 0.0
Bx, By, Bz = 1.0, 0.5, 0.0
p = 1.0

U = primitive_to_conservative(rho, vx, vy, vz, Bx, By, Bz, p)
F = compute_flux(U, Bx)

print("\n원시 변수:")
print(f"  ρ={rho}, v=({vx},{vy},{vz}), B=({Bx},{By},{Bz}), p={p}")
print("\n보존 변수 U:")
print(f"  {U}")
print("\n플럭스 F:")
print(f"  {F}")

cf, ca, cs = mhd_wave_speeds(rho, vx, p, Bx, By, Bz)
print(f"\n파동 속도: cf={cf:.3f}, ca={ca:.3f}, cs={cs:.3f}")
```

### 1.2 MHD 고유 구조

```
MHD 특성 (7개 파동):

λ₁ = vx - cf  (빠른 자기음파, 좌)
λ₂ = vx - ca  (Alfven 파, 좌)
λ₃ = vx - cs  (느린 자기음파, 좌)
λ₄ = vx       (엔트로피 파)
λ₅ = vx + cs  (느린 자기음파, 우)
λ₆ = vx + ca  (Alfven 파, 우)
λ₇ = vx + cf  (빠른 자기음파, 우)

여기서:
cf = √[(cs² + ca²)/2 + √((cs² + ca²)² - 4cs²cax²)/2]  (fast)
cs = √[(cs² + ca²)/2 - √((cs² + ca²)² - 4cs²cax²)/2]  (slow)
ca = |Bx|/√ρ                                          (Alfven)
```

```python
def visualize_mhd_characteristics():
    """MHD 특성 시각화"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) 파동 속도 vs Bx (By = 0.5, Bz = 0 고정)
    ax1 = axes[0]

    rho = 1.0
    p = 1.0
    vx = 0.0
    By, Bz = 0.5, 0.0

    Bx_range = np.linspace(0.01, 2.0, 100)

    cf_list, ca_list, cs_list = [], [], []

    for Bx in Bx_range:
        cf, ca, cs = mhd_wave_speeds(rho, vx, p, Bx, By, Bz)
        cf_list.append(cf)
        ca_list.append(ca)
        cs_list.append(cs)

    ax1.plot(Bx_range, cf_list, 'b-', linewidth=2, label='Fast (cf)')
    ax1.plot(Bx_range, ca_list, 'g--', linewidth=2, label='Alfven (ca)')
    ax1.plot(Bx_range, cs_list, 'r-', linewidth=2, label='Slow (cs)')

    # 음속 참조선
    cs_sound = np.sqrt(gamma * p / rho)
    ax1.axhline(y=cs_sound, color='gray', linestyle=':', label=f'Sound ({cs_sound:.2f})')

    ax1.set_xlabel('Bx')
    ax1.set_ylabel('Wave Speed')
    ax1.set_title('MHD 파동 속도 vs Bx')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) x-t 다이어그램 (특성선)
    ax2 = axes[1]

    # 특정 상태에서의 특성선
    Bx = 1.0
    cf, ca, cs = mhd_wave_speeds(rho, vx, p, Bx, By, Bz)

    x0 = 0  # 초기 위치
    t = np.linspace(0, 1, 50)

    # 7개 특성선
    speeds = [-cf, -ca, -cs, 0, cs, ca, cf]
    labels = ['-cf', '-ca', '-cs', 'entropy', '+cs', '+ca', '+cf']
    colors = ['blue', 'green', 'red', 'black', 'red', 'green', 'blue']

    for speed, label, color in zip(speeds, labels, colors):
        x = x0 + speed * t
        ax2.plot(x, t, color=color, linewidth=1.5, label=label)

    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('MHD 특성선 (x-t 다이어그램)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 2)

    plt.tight_layout()
    plt.savefig('mhd_characteristics.png', dpi=150, bbox_inches='tight')
    plt.show()

# visualize_mhd_characteristics()
```

---

## 2. 유한 체적법 기초

### 2.1 적분 형태와 셀 평균

```
유한 체적법 (Finite Volume Method):

적분 형태:
d/dt ∫ U dx + [F(x₂) - F(x₁)] = 0

셀 평균:
Uᵢ = (1/Δx) ∫_{xᵢ₋₁/₂}^{xᵢ₊₁/₂} U dx

반이산화:
dUᵢ/dt = -(F_{i+1/2} - F_{i-1/2}) / Δx

수치 플럭스:
F_{i+1/2} = F(Uᵢ, Uᵢ₊₁)  (Riemann 솔버 또는 근사)

장점:
- 보존 형태 자동 만족
- 불연속 처리 자연스러움
- 다양한 격자 형태 적용 가능
```

```python
def finite_volume_concept():
    """유한 체적법 개념 시각화"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) 셀 평균과 플럭스
    ax1 = axes[0]

    # 셀 경계
    x_faces = np.arange(0, 6)
    x_centers = x_faces[:-1] + 0.5

    # 셀 평균값 (예시)
    U_avg = np.array([1.0, 0.8, 1.2, 0.6, 0.9])

    # 셀 그리기
    for i, (x_l, x_r, U) in enumerate(zip(x_faces[:-1], x_faces[1:], U_avg)):
        ax1.fill([x_l, x_r, x_r, x_l], [0, 0, U, U], alpha=0.3,
                color=f'C{i}', edgecolor='black', linewidth=1.5)
        ax1.text((x_l + x_r)/2, U/2, f'$U_{i+1}$', ha='center', va='center', fontsize=11)

    # 플럭스 화살표
    for x in x_faces[1:-1]:
        ax1.annotate('', xy=(x, 1.5), xytext=(x-0.3, 1.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax1.text(x, 1.6, f'$F_{{i+1/2}}$', ha='center', fontsize=10, color='red')

    ax1.set_xlabel('x')
    ax1.set_ylabel('U')
    ax1.set_title('유한 체적법: 셀 평균과 수치 플럭스')
    ax1.set_xlim(-0.5, 5.5)
    ax1.set_ylim(0, 2)
    ax1.grid(True, alpha=0.3)

    # (2) Riemann 문제
    ax2 = axes[1]

    # 초기 불연속
    x = np.linspace(-1, 1, 200)
    U_L = 1.0
    U_R = 0.5

    U_init = np.where(x < 0, U_L, U_R)
    ax2.plot(x, U_init, 'b-', linewidth=2, label='초기 조건')

    # Riemann 해 (개념적)
    # 충격파, 접촉 불연속, 팽창파 표시
    t = 0.3
    x_shock = 0.3  # 충격파 위치

    U_riemann = np.where(x < -0.2, U_L,
                        np.where(x < 0, U_L - (U_L - 0.8) * (x + 0.2) / 0.2,
                                np.where(x < x_shock, 0.8, U_R)))

    ax2.plot(x, U_riemann, 'r--', linewidth=2, label=f't = {t}')

    # 파동 표시
    ax2.axvline(x=-0.2, color='green', linestyle=':', label='팽창파 시작')
    ax2.axvline(x=0, color='purple', linestyle=':', label='접촉 불연속')
    ax2.axvline(x=x_shock, color='orange', linestyle=':', label='충격파')

    ax2.set_xlabel('x')
    ax2.set_ylabel('U')
    ax2.set_title('Riemann 문제: 불연속 초기조건의 시간 전개')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('finite_volume.png', dpi=150, bbox_inches='tight')
    plt.show()

# finite_volume_concept()
```

---

## 3. Godunov 유형 스킴

### 3.1 Lax-Friedrichs 플럭스

```
Lax-Friedrichs (LxF) 플럭스:

F_{i+1/2} = (1/2)[F(Uᵢ) + F(Uᵢ₊₁)] - (Δx/2Δt)(Uᵢ₊₁ - Uᵢ)

또는 지역 LxF:
F_{i+1/2} = (1/2)[F(Uᵢ) + F(Uᵢ₊₁)] - (α/2)(Uᵢ₊₁ - Uᵢ)

α = max(|λ|)  (최대 파동 속도)

특징:
- 간단하고 견고함
- 1차 정확도
- 수치 확산 큼
```

### 3.2 HLL/HLLD 플럭스

```
HLL (Harten-Lax-van Leer) 플럭스:

가정: 좌측/우측 파동 속도 SL, SR 만 고려

        ⎧ F_L                          if SL ≥ 0
F_HLL = ⎨ (SR*F_L - SL*F_R + SL*SR*(U_R - U_L))/(SR - SL)  if SL < 0 < SR
        ⎩ F_R                          if SR ≤ 0

파동 속도 추정:
SL = min(vx_L - cf_L, vx_R - cf_R)
SR = max(vx_L + cf_L, vx_R + cf_R)

HLLD (MHD용):
- 중간 상태를 더 세분화
- 접촉 불연속과 Alfven 파 구분
- MHD에서 더 정확
```

```python
def lax_friedrichs_flux(U_L, U_R, Bx, max_speed=None):
    """Lax-Friedrichs 플럭스"""
    F_L = compute_flux(U_L, Bx)
    F_R = compute_flux(U_R, Bx)

    if max_speed is None:
        # 최대 파동 속도 계산
        rho_L, vx_L, vy_L, vz_L, _, By_L, Bz_L, p_L = conservative_to_primitive(U_L, Bx)
        rho_R, vx_R, vy_R, vz_R, _, By_R, Bz_R, p_R = conservative_to_primitive(U_R, Bx)

        cf_L, _, _ = mhd_wave_speeds(rho_L, vx_L, p_L, Bx, By_L, Bz_L)
        cf_R, _, _ = mhd_wave_speeds(rho_R, vx_R, p_R, Bx, By_R, Bz_R)

        max_speed = max(abs(vx_L) + cf_L, abs(vx_R) + cf_R)

    F = 0.5 * (F_L + F_R) - 0.5 * max_speed * (U_R - U_L)
    return F

def hll_flux(U_L, U_R, Bx):
    """HLL 플럭스"""
    F_L = compute_flux(U_L, Bx)
    F_R = compute_flux(U_R, Bx)

    rho_L, vx_L, vy_L, vz_L, _, By_L, Bz_L, p_L = conservative_to_primitive(U_L, Bx)
    rho_R, vx_R, vy_R, vz_R, _, By_R, Bz_R, p_R = conservative_to_primitive(U_R, Bx)

    cf_L, _, _ = mhd_wave_speeds(rho_L, vx_L, p_L, Bx, By_L, Bz_L)
    cf_R, _, _ = mhd_wave_speeds(rho_R, vx_R, p_R, Bx, By_R, Bz_R)

    SL = min(vx_L - cf_L, vx_R - cf_R)
    SR = max(vx_L + cf_L, vx_R + cf_R)

    if SL >= 0:
        return F_L
    elif SR <= 0:
        return F_R
    else:
        F_HLL = (SR * F_L - SL * F_R + SL * SR * (U_R - U_L)) / (SR - SL)
        return F_HLL
```

---

## 4. MHD 충격파 관 문제

### 4.1 Brio-Wu 충격파 관

```
Brio-Wu 충격파 관 (1988):
- MHD 코드 표준 테스트 문제
- Sod 충격파 관의 MHD 버전

초기 조건:
좌측 (x < 0.5):          우측 (x ≥ 0.5):
ρ = 1.0                   ρ = 0.125
p = 1.0                   p = 0.1
vx = vy = vz = 0          vx = vy = vz = 0
Bx = 0.75                 Bx = 0.75
By = 1.0                  By = -1.0
Bz = 0                    Bz = 0

경계 조건: 유출 (outflow)
최종 시간: t = 0.1

해의 구조:
- 빠른 희박파 (fast rarefaction)
- 복합파 (compound wave)
- 접촉 불연속 (contact)
- 느린 충격파 (slow shock)
- 빠른 희박파 (fast rarefaction)
```

```python
class MHD_1D_Solver:
    """1D MHD 유한 체적법 솔버"""

    def __init__(self, Nx=400, x_range=(0, 1), Bx=0.75):
        self.Nx = Nx
        self.x_min, self.x_max = x_range
        self.dx = (self.x_max - self.x_min) / Nx
        self.x = np.linspace(self.x_min + 0.5*self.dx,
                            self.x_max - 0.5*self.dx, Nx)
        self.Bx = Bx

        # 보존 변수 배열 (7 components)
        self.U = np.zeros((7, Nx))

    def set_brio_wu(self):
        """Brio-Wu 충격파 관 초기조건"""
        for i, x in enumerate(self.x):
            if x < 0.5:
                rho, vx, vy, vz = 1.0, 0.0, 0.0, 0.0
                By, Bz, p = 1.0, 0.0, 1.0
            else:
                rho, vx, vy, vz = 0.125, 0.0, 0.0, 0.0
                By, Bz, p = -1.0, 0.0, 0.1

            self.U[:, i] = primitive_to_conservative(rho, vx, vy, vz,
                                                     self.Bx, By, Bz, p)

    def compute_dt(self, cfl=0.5):
        """시간 단계 계산 (CFL 조건)"""
        max_speed = 0

        for i in range(self.Nx):
            rho, vx, vy, vz, _, By, Bz, p = conservative_to_primitive(
                self.U[:, i], self.Bx)
            cf, _, _ = mhd_wave_speeds(rho, vx, p, self.Bx, By, Bz)
            max_speed = max(max_speed, abs(vx) + cf)

        return cfl * self.dx / max_speed

    def step(self, dt, flux_func='lxf'):
        """한 시간 단계 전진"""
        # 플럭스 계산
        F = np.zeros((7, self.Nx + 1))

        for i in range(self.Nx + 1):
            # 경계 처리 (outflow)
            if i == 0:
                U_L = self.U[:, 0]
                U_R = self.U[:, 0]
            elif i == self.Nx:
                U_L = self.U[:, -1]
                U_R = self.U[:, -1]
            else:
                U_L = self.U[:, i-1]
                U_R = self.U[:, i]

            if flux_func == 'lxf':
                F[:, i] = lax_friedrichs_flux(U_L, U_R, self.Bx)
            else:
                F[:, i] = hll_flux(U_L, U_R, self.Bx)

        # 업데이트
        self.U = self.U - dt / self.dx * (F[:, 1:] - F[:, :-1])

    def run(self, t_final, cfl=0.5, flux_func='lxf'):
        """시뮬레이션 실행"""
        t = 0
        step_count = 0

        while t < t_final:
            dt = self.compute_dt(cfl)
            if t + dt > t_final:
                dt = t_final - t

            self.step(dt, flux_func)
            t += dt
            step_count += 1

        print(f"완료: {step_count} steps, final t = {t:.4f}")

    def get_primitives(self):
        """원시 변수 반환"""
        rho = np.zeros(self.Nx)
        vx = np.zeros(self.Nx)
        vy = np.zeros(self.Nx)
        vz = np.zeros(self.Nx)
        By = np.zeros(self.Nx)
        Bz = np.zeros(self.Nx)
        p = np.zeros(self.Nx)

        for i in range(self.Nx):
            rho[i], vx[i], vy[i], vz[i], _, By[i], Bz[i], p[i] = \
                conservative_to_primitive(self.U[:, i], self.Bx)

        return rho, vx, vy, vz, By, Bz, p


def run_brio_wu_test():
    """Brio-Wu 충격파 관 시뮬레이션"""

    # 솔버 생성
    solver = MHD_1D_Solver(Nx=400, x_range=(0, 1), Bx=0.75)
    solver.set_brio_wu()

    # 초기 상태 저장
    rho_init, vx_init, vy_init, _, By_init, _, p_init = solver.get_primitives()
    x = solver.x

    # 시뮬레이션 실행
    t_final = 0.1
    solver.run(t_final, cfl=0.5, flux_func='hll')

    # 최종 상태
    rho, vx, vy, _, By, _, p = solver.get_primitives()

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 밀도
    axes[0, 0].plot(x, rho_init, 'b--', alpha=0.5, label='Initial')
    axes[0, 0].plot(x, rho, 'b-', linewidth=1.5, label='Final')
    axes[0, 0].set_ylabel(r'$\rho$')
    axes[0, 0].set_title('밀도')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 압력
    axes[0, 1].plot(x, p_init, 'r--', alpha=0.5, label='Initial')
    axes[0, 1].plot(x, p, 'r-', linewidth=1.5, label='Final')
    axes[0, 1].set_ylabel('p')
    axes[0, 1].set_title('압력')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # x-속도
    axes[0, 2].plot(x, vx_init, 'g--', alpha=0.5, label='Initial')
    axes[0, 2].plot(x, vx, 'g-', linewidth=1.5, label='Final')
    axes[0, 2].set_ylabel(r'$v_x$')
    axes[0, 2].set_title('x-속도')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # y-속도
    axes[1, 0].plot(x, vy_init, 'm--', alpha=0.5, label='Initial')
    axes[1, 0].plot(x, vy, 'm-', linewidth=1.5, label='Final')
    axes[1, 0].set_ylabel(r'$v_y$')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_title('y-속도')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # By
    axes[1, 1].plot(x, By_init, 'c--', alpha=0.5, label='Initial')
    axes[1, 1].plot(x, By, 'c-', linewidth=1.5, label='Final')
    axes[1, 1].set_ylabel(r'$B_y$')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_title('y-자기장')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 총 압력
    B2 = solver.Bx**2 + By**2
    p_total = p + 0.5 * B2
    B2_init = solver.Bx**2 + By_init**2
    p_total_init = p_init + 0.5 * B2_init

    axes[1, 2].plot(x, p_total_init, 'k--', alpha=0.5, label='Initial')
    axes[1, 2].plot(x, p_total, 'k-', linewidth=1.5, label='Final')
    axes[1, 2].set_ylabel(r'$p + B^2/2$')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_title('총 압력')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'Brio-Wu 충격파 관 (t = {t_final})', fontsize=14)
    plt.tight_layout()
    plt.savefig('brio_wu_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    return solver

# solver = run_brio_wu_test()
```

---

## 5. div B = 0 제약조건

### 5.1 문제점과 해결 방법

```
∇·B = 0 제약조건:

물리적 의미:
- 자기 단극자 없음
- 항상 만족해야 하는 제약

수치적 문제:
- 이산화 오차로 ∇·B ≠ 0 발생 가능
- "자기 단극자" 축적
- 비물리적 힘 발생, 불안정성

해결 방법:

1. Constrained Transport (CT):
   - 자기장을 셀 면에 저장
   - 전기장을 셀 모서리에 저장
   - 구조적으로 ∇·B = 0 보장
   - Yee 격자와 유사

2. Projection Method:
   - Hodge 분해: B = B_sol + ∇φ
   - Poisson 방정식: ∇²φ = ∇·B
   - 수정: B_new = B - ∇φ

3. Divergence Cleaning:
   a) Parabolic: ∂B/∂t = -ch²∇(∇·B)
   b) Hyperbolic: ∂ψ/∂t + ch²∇·B = 0,
                  ∂B/∂t + ch∇ψ = 0
   (GLM: Generalized Lagrange Multiplier)

4. Powell 8-wave:
   - 소스 항 추가: S = -(∇·B) [0, B, v, v·B]ᵀ
   - 비보존적, 정상 상태에서 유효
```

```python
def divergence_cleaning_demo():
    """발산 클리닝 개념 시연"""

    print("=" * 60)
    print("div B = 0 제약조건 처리 방법")
    print("=" * 60)

    methods = """
    ┌─────────────────────────────────────────────────────────────┐
    │              div B = 0 제약조건 처리                         │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │ 1. Constrained Transport (CT)                               │
    │    - 구조적으로 ∇·B = 0 보장                                │
    │    - 자기장: 셀 면 중심                                     │
    │    - 전기장: 셀 모서리                                      │
    │    - Stokes 정리로 자동 만족                                │
    │                                                             │
    │ 2. Projection Method                                        │
    │    - Helmholtz 분해                                         │
    │    - Poisson 방정식 풀이 필요                               │
    │    - 계산 비용 높음                                         │
    │                                                             │
    │ 3. GLM (Hyperbolic Cleaning)                                │
    │    - 추가 스칼라 변수 ψ 도입                                │
    │    - ∂ψ/∂t + ch²∇·B = -(ch²/cp²)ψ                          │
    │    - ∂B/∂t + ... + ch∇ψ = 0                                │
    │    - 오류가 파동으로 전파되어 빠져나감                      │
    │                                                             │
    │ 4. Powell Source Terms                                      │
    │    - 비보존적 소스 항 추가                                  │
    │    - ∂U/∂t + ∂F/∂x = -(∇·B)S                               │
    │    - 간단하지만 에너지 보존 위반                            │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """
    print(methods)

    # 시각화: CT 격자 구조
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (1) Constrained Transport 격자
    ax1 = axes[0]

    # 셀 격자
    for i in range(5):
        ax1.axhline(y=i, color='gray', linestyle='-', linewidth=0.5)
        ax1.axvline(x=i, color='gray', linestyle='-', linewidth=0.5)

    # Bx (수직 면)
    for i in range(5):
        for j in range(4):
            ax1.plot(i, j+0.5, 'b>', markersize=12)

    # By (수평 면)
    for i in range(4):
        for j in range(5):
            ax1.plot(i+0.5, j, 'r^', markersize=12)

    # Ez (모서리)
    for i in range(5):
        for j in range(5):
            ax1.plot(i, j, 'go', markersize=8)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Constrained Transport 격자')
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 4.5)

    ax1.plot([], [], 'b>', markersize=10, label='Bx (x-면)')
    ax1.plot([], [], 'r^', markersize=10, label='By (y-면)')
    ax1.plot([], [], 'go', markersize=8, label='Ez (모서리)')
    ax1.legend(loc='upper right')

    # (2) GLM 방법 개념
    ax2 = axes[1]

    x = np.linspace(0, 10, 100)
    t = 0

    # 초기 div B 오류 (가우시안)
    div_B = np.exp(-(x - 3)**2)

    # 시간 전개 (양쪽으로 전파)
    times = [0, 1, 2, 3]
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    for ti, color in zip(times, colors):
        # GLM: 오류가 ±ch 속도로 전파
        ch = 1.5
        div_B_t = 0.5 * (np.exp(-(x - 3 - ch*ti)**2) +
                        np.exp(-(x - 3 + ch*ti)**2)) * np.exp(-0.5*ti)
        ax2.plot(x, div_B_t, color=color, linewidth=1.5, label=f't = {ti}')

    ax2.set_xlabel('x')
    ax2.set_ylabel(r'$\nabla \cdot B$ error')
    ax2.set_title('GLM: 오류가 도메인 밖으로 전파')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('div_b_cleaning.png', dpi=150, bbox_inches='tight')
    plt.show()

# divergence_cleaning_demo()
```

---

## 6. 고해상도 스킴

### 6.1 MUSCL-Hancock

```
MUSCL-Hancock (2차 정확도):

1. 선형 재구성:
   U_L = Uᵢ + 0.5 * φ(r) * (Uᵢ - Uᵢ₋₁)
   U_R = Uᵢ₊₁ - 0.5 * φ(r) * (Uᵢ₊₂ - Uᵢ₊₁)

   φ(r): 기울기 제한자 (limiter)
   - minmod: φ(r) = max(0, min(1, r))
   - MC: φ(r) = max(0, min((1+r)/2, 2, 2r))
   - van Leer: φ(r) = (r + |r|)/(1 + |r|)

2. Predictor (반단계 전진):
   U_L^{n+1/2} = U_L - (Δt/2Δx)(F(U_L) - F(U_L⁻))

3. Riemann 풀이 및 플럭스 계산

장점:
- 2차 정확도 (매끄러운 영역)
- TVD (Total Variation Diminishing)
- 진동 억제
```

```python
def minmod(a, b):
    """Minmod 리미터"""
    if a * b <= 0:
        return 0
    elif abs(a) < abs(b):
        return a
    else:
        return b

def mc_limiter(a, b):
    """MC (Monotonized Central) 리미터"""
    if a * b <= 0:
        return 0
    c = 0.5 * (a + b)
    return np.sign(c) * min(abs(c), 2*abs(a), 2*abs(b))

def muscl_reconstruct(U, i, limiter='minmod'):
    """MUSCL 재구성"""
    if limiter == 'minmod':
        lim_func = minmod
    else:
        lim_func = mc_limiter

    Nx = U.shape[1]

    # 기울기 계산
    if i > 0 and i < Nx - 1:
        slope_L = U[:, i] - U[:, i-1]
        slope_R = U[:, i+1] - U[:, i]

        slope = np.array([lim_func(slope_L[k], slope_R[k])
                         for k in range(len(slope_L))])
    else:
        slope = np.zeros(U.shape[0])

    U_L = U[:, i] - 0.5 * slope  # 왼쪽 상태
    U_R = U[:, i] + 0.5 * slope  # 오른쪽 상태

    return U_L, U_R


def run_brio_wu_high_resolution():
    """고해상도 Brio-Wu 충격파 관"""

    # 솔버 (더 많은 셀)
    solver_lr = MHD_1D_Solver(Nx=200, x_range=(0, 1), Bx=0.75)
    solver_hr = MHD_1D_Solver(Nx=800, x_range=(0, 1), Bx=0.75)

    solver_lr.set_brio_wu()
    solver_hr.set_brio_wu()

    t_final = 0.1

    solver_lr.run(t_final, cfl=0.5, flux_func='hll')
    solver_hr.run(t_final, cfl=0.5, flux_func='hll')

    # 비교
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x_lr = solver_lr.x
    x_hr = solver_hr.x

    rho_lr, vx_lr, _, _, By_lr, _, p_lr = solver_lr.get_primitives()
    rho_hr, vx_hr, _, _, By_hr, _, p_hr = solver_hr.get_primitives()

    axes[0].plot(x_lr, rho_lr, 'b-', linewidth=1.5, label='Nx=200')
    axes[0].plot(x_hr, rho_hr, 'r-', linewidth=1, alpha=0.7, label='Nx=800')
    axes[0].set_ylabel(r'$\rho$')
    axes[0].set_xlabel('x')
    axes[0].set_title('밀도')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_lr, vx_lr, 'b-', linewidth=1.5, label='Nx=200')
    axes[1].plot(x_hr, vx_hr, 'r-', linewidth=1, alpha=0.7, label='Nx=800')
    axes[1].set_ylabel(r'$v_x$')
    axes[1].set_xlabel('x')
    axes[1].set_title('x-속도')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x_lr, By_lr, 'b-', linewidth=1.5, label='Nx=200')
    axes[2].plot(x_hr, By_hr, 'r-', linewidth=1, alpha=0.7, label='Nx=800')
    axes[2].set_ylabel(r'$B_y$')
    axes[2].set_xlabel('x')
    axes[2].set_title('y-자기장')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('격자 수렴 테스트 (1차 HLL)', fontsize=14)
    plt.tight_layout()
    plt.savefig('brio_wu_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()

# run_brio_wu_high_resolution()
```

---

## 7. MHD 코드 검증

### 7.1 표준 테스트 문제

```
MHD 코드 검증용 표준 문제:

1. Brio-Wu 충격파 관 (1D)
   - 충격파, 접촉 불연속, 희박파
   - 복합파 구조

2. Orszag-Tang 와류 (2D)
   - 고도의 비선형 상호작용
   - 작은 스케일 구조 발생
   - 충격파 상호작용

3. MHD 회전 불연속 (1D)
   - Alfven 파 정확도 검증
   - 순수 회전 (ρ, p 일정)

4. 폭발 문제 (2D/3D)
   - MHD Sedov-Taylor
   - 자기장 영향

5. 자기 루프 이류 (2D)
   - div B = 0 보존 검증
   - 원형 자기 구조 이동

정확도 검증:
- 해석해가 있는 문제 사용
- 격자 수렴 테스트
- 보존량 확인 (질량, 운동량, 에너지)
```

```python
def test_problems_overview():
    """MHD 표준 테스트 문제 개요"""

    print("=" * 60)
    print("MHD 표준 테스트 문제")
    print("=" * 60)

    tests = """
    1D 테스트:
    ┌─────────────────────────────────────────────────────────┐
    │ 문제          │ 검증 대상              │ 난이도        │
    ├───────────────┼────────────────────────┼───────────────┤
    │ Brio-Wu       │ 충격파 포착, 기본 파동 │ 기본          │
    │ Dai-Woodward  │ 7파 구조 전체          │ 중급          │
    │ Einfeldt 1203 │ 낮은 β, 강한 자기장    │ 도전적        │
    │ Ryu-Jones     │ 다양한 MHD 파동        │ 중급          │
    └─────────────────────────────────────────────────────────┘

    2D 테스트:
    ┌─────────────────────────────────────────────────────────┐
    │ 문제          │ 검증 대상              │ 난이도        │
    ├───────────────┼────────────────────────┼───────────────┤
    │ Orszag-Tang   │ 비선형 발전, 난류      │ 표준          │
    │ 자기 회전 불안│ 선형 성장률 비교       │ 물리 검증     │
    │ 루프 이류     │ div B, 수치 확산       │ 기본          │
    │ 폭발 문제     │ 구면 대칭 보존         │ 도전적        │
    └─────────────────────────────────────────────────────────┘

    수렴 테스트:
    - L1, L2, L∞ 노름 계산
    - 격자 해상도 2배씩 증가
    - 예상 수렴 차수 확인 (1차: O(h), 2차: O(h²))
    """
    print(tests)

test_problems_overview()
```

---

## 8. 연습 문제

### 연습 1: 파동 속도
ρ = 1, p = 0.5, Bx = 1, By = 0.5, Bz = 0 일 때 빠른/느린/Alfven 파동 속도를 계산하시오. (γ = 5/3)

### 연습 2: Lax-Friedrichs
1D 선형 이류방정식에 Lax-Friedrichs 스킴을 적용하고, 수치 확산 계수를 유도하시오.

### 연습 3: HLL vs LxF
Brio-Wu 문제를 HLL과 Lax-Friedrichs 플럭스로 각각 풀고 결과를 비교하시오.

### 연습 4: div B 오류
2D MHD 시뮬레이션에서 div B 오류를 모니터링하는 코드를 작성하시오.

---

## 9. 참고자료

### 핵심 논문
- Brio & Wu (1988) "An Upwind Differencing Scheme for the Equations of Ideal Magnetohydrodynamics"
- Dedner et al. (2002) "Hyperbolic Divergence Cleaning for the MHD Equations"
- Toth (2000) "The ∇·B = 0 Constraint in Shock-Capturing Magnetohydrodynamics Codes"

### 교재
- Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics"
- LeVeque, "Finite Volume Methods for Hyperbolic Problems"

### MHD 코드
- Athena++ (Stone et al.)
- PLUTO (Mignone et al.)
- FLASH (Fryxell et al.)
- Pencil Code (Brandenburg et al.)

---

## 요약

```
MHD 수치해법 핵심:

1. 보존 형태:
   ∂U/∂t + ∂F/∂x = 0
   U = [ρ, ρv, B, E]ᵀ (7 변수, 1D)

2. 유한 체적법:
   dUᵢ/dt = -(F_{i+1/2} - F_{i-1/2})/Δx
   수치 플럭스로 Riemann 문제 근사

3. 수치 플럭스:
   - Lax-Friedrichs: 간단, 1차, 확산 큼
   - HLL: 2파 근사, 중간 정확도
   - HLLD: MHD 최적화, 고정확도

4. CFL 조건:
   Δt ≤ CFL × Δx / (|v| + cf)
   cf: 빠른 자기음파 속도

5. div B = 0:
   - CT: 구조적 보존
   - GLM: 쌍곡선형 클리닝
   - Projection: Poisson 풀이

6. 고해상도 스킴:
   - MUSCL + limiter
   - PPM, WENO
   - 2차 이상 정확도

7. 검증:
   - Brio-Wu 충격파 관
   - 격자 수렴 테스트
   - 보존량 확인
```

---

다음 레슨에서는 플라즈마 시뮬레이션과 PIC (Particle-In-Cell) 방법을 다룹니다.
