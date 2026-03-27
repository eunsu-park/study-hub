# 18. 프로젝트

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

- 자기 재결합의 완전한 2D 저항성 MHD 시뮬레이션 구현하기
- 태양 플레어 모델에서 재결합률과 에너지 변환 분석하기
- 토카막 플라즈마를 위한 1D MHD 안정성 분석기 구축하기
- 안정성 기준(Kruskal-Shafranov, Suydam) 적용하여 붕괴 예측하기
- 구형 쉘에서 평균장 다이나모 시뮬레이션하기
- 진동 다이나모 해와 나비 다이어그램 관찰하기
- 이 과정에서 학습한 모든 MHD 개념을 실용 응용에 통합하기

---

## 프로젝트 1: 자기 재결합을 통한 태양 플레어 시뮬레이션

### 1.1 물리 배경

**태양 플레어**는 태양 코로나에서 자기 에너지가 폭발적으로 방출되는 것으로, **자기 재결합**에 의해 구동됩니다 - 자기장 선의 위상학적 재구성으로 자기 에너지를 운동 에너지와 열 에너지로 변환합니다.

**핵심 물리:**
- **티어링 불안정성(Tearing Instability)**: 전류 시트에서 저항성 불안정성 → 플라즈모이드(plasmoid) 형성
- **Sweet-Parker 재결합**: 고전 모델, 느린 재결합률 $\sim \eta^{1/2}$
- **플라즈모이드 매개 재결합**: 2차 아일랜드를 통한 빠른 재결합, $\eta$에 무관한 속도

**관측 가능량:**
- 재결합률: 유입 속도 $v_{\text{in}}$
- 에너지 변환: $\Delta E_{\text{mag}}$ → $\Delta E_{\text{kin}} + \Delta E_{\text{th}}$
- 전류 시트 구조: 두께 $\delta$, 길이 $L$
- 재결합 영역에서 온도 상승

### 1.2 문제 설정

**기하학:** $[-L_x, L_x] \times [-L_y, L_y]$ 상자에서 2D Harris 전류 시트

**초기 조건:**
$$
B_x(x, y) = B_0 \tanh(y / a)
$$
$$
B_y(x, y) = 0
$$
$$
\rho(x, y) = \rho_0 \left(1 + \beta \operatorname{sech}^2(y/a)\right)
$$
$$
p(x, y) = p_0 + \frac{B_0^2}{2} \left(1 - \tanh^2(y/a)\right)
$$

여기서:
- $B_0 = 1.0$ (특성 장 강도)
- $a = 0.5$ (전류 시트 반 두께)
- $\rho_0 = 1.0$ (배경 밀도)
- $\beta = 1.0$ (플라즈마 베타 매개변수)
- $p_0 = B_0^2 \beta / 2$

**섭동 (재결합 트리거):**
$$
B_y(x, y) = \epsilon B_0 \cos(k_x x) \sin(\pi y / L_y)
$$
$\epsilon = 0.1$, $k_x = 2\pi / L_x$와 함께.

**매개변수:**
- 영역: $L_x = 25.6$, $L_y = 12.8$
- 그리드: $N_x = 512$, $N_y = 256$
- 저항: $\eta = 0.001$ (국소 또는 균일)
- $\gamma = 5/3$ (이상 기체)

**경계 조건:**
- $x$에서 주기
- $y$에서 전도 벽 ($\mathbf{v} \cdot \hat{y} = 0$, $\partial_y B_x = 0$, $B_y = 0$)

### 1.3 구현

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 25.6, 12.8
# Nx=512, Ny=256: 2:1 종횡비가 영역 종횡비 Lx:Ly = 2:1과 일치하여
# dx ≈ dy ≈ 0.05 — 비등방성 수치 확산을 피하는 균일한 셀 크기;
# Nx=512는 저항층 δ ~ a/√S ≈ 0.5/√500 ≈ 0.022가 ~5개 셀에 걸치도록 선택 —
# 인공적인 전류 시트 확장 없이 전류 시트를 해상하기 위한 최소값
Nx, Ny = 512, 256
dx = Lx / Nx
dy = Ly / Ny
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Physical parameters
B0 = 1.0
# a = 0.5은 해리스 시트(Harris Sheet) 반 두께를 설정; S = 룬퀴스트 수 = τ_R/τ_A = a²/(η*τ_A)인
# 저항층 δ ~ a/√S를 해상할 수 있을 만큼 충분히 크게 설정해야 함;
# η = 0.001이고 a = 0.5일 때 Alfvén 시간 τ_A ~ a/v_A ~ 0.5이고 S ≈ 500,
# δ ≈ 0.022 — 그리드 스케일 dx ≈ 0.05 바로 위 (한계적으로 해상됨)
a = 0.5
rho0 = 1.0
# β = 1.0은 총 압력 균형을 보장: 전류 시트는 낮은 자기 압력(sech² 프로파일)을 가지며
# 이는 더 높은 플라즈마 압력으로 보완되어야 함,
# β = 1은 재결합이 시작되기 전에 시트가 즉시 팽창하거나 붕괴되지 않도록 균형을 맞춤;
# β < 1이면 시트를 과도하게 가두고, β > 1이면 팽창함
beta_param = 1.0
p0 = B0**2 * beta_param / 2.0
gamma = 5.0 / 3.0
# η = 0.001은 이 구성에서 룬퀴스트 수(Lundquist Number) S = a*v_A/η ≈ 500을 제공:
# S ≈ 500은 플라즈모이드 불안정(Plasmoid-Unstable) 체제; (Sweet-Parker은 S^{1/2}배 느림)
# 따라서 시뮬레이션 중 2차 아일랜드가 형성되어,
# Petschek/플라즈모이드 빠른 재결합률 ~0.01-0.1 v_A에 더 가까운 재결합률이 나타남
eta = 0.001
nu = 0.001  # Viscosity (for numerical stability)
# CFL = 0.4: Alfvén 속도 v_A = B0/√(μ₀ρ0) = 1이 가장 빠른 신호를 설정;
# CFL < 0.5는 정보가 스텝당 한 셀 미만으로 이동하도록 보장하여,
# 수치적 의존 영역이 너무 작을 때 발생하는 불안정성을 방지
CFL = 0.4

# Initial conditions
def initial_harris_sheet():
    # tanh(Y/a)는 Bx에 대한 정확한 해리스 평형(Harris Equilibrium) 해: 아래의 sech²
    # 밀도 및 압력 프로파일과 결합할 때 d/dy(p + B²/2μ₀) = 0을 만족 —
    # 이 정확한 평형에서 벗어나게 섭동을 주면 초기 과도 현상 없이
    # 티어링 불안정성(Tearing Instability)이 씨앗을 받음
    Bx = B0 * np.tanh(Y / a)
    # By 섭동 진폭 ε = 0.1: 합리적인 시뮬레이션 시간에 재결합을 유발할 만큼 충분히 크지만
    # 티어링 모드가 선형 체제에서 시작되어 비선형 포화 전에 지수적으로 성장하도록
    # ε < 1로 충분히 작음; cos(kx)*sin(πy/Ly)는 y에서 전도 벽에서 By=0을 만족
    By = 0.1 * B0 * np.cos(2*np.pi*X/Lx) * np.sin(np.pi*Y/Ly)  # Perturbation
    Bz = np.zeros_like(Bx)

    # ρ ∝ (1 + β*sech²(y/a)): 전류 시트 내부에서 밀도가 증가하여
    # 힘 균형을 유지 — 자기 압력 B²/2μ₀이 y=0(tanh=0인 곳)에서 0으로 감소하므로
    # 플라즈마 압력 p가 그곳에서 최대여야 하며, 이상 기체 법칙으로 고정된 T에서
    # 더 높은 p는 더 높은 ρ를 의미함
    rho = rho0 * (1.0 + beta_param * (1.0/np.cosh(Y/a))**2)
    # p = p0 + B0²/2*(1 - tanh²): 총 압력 (p + B²/2) = 상수 = p0 + B0²/2,
    # 모든 y 위치에서 정확한 해리스 힘 균형을 확인함
    p = p0 + B0**2/2.0 * (1.0 - np.tanh(Y/a)**2)

    vx = np.zeros_like(Bx)
    vy = np.zeros_like(Bx)
    vz = np.zeros_like(Bx)

    return rho, vx, vy, vz, p, Bx, By, Bz

# Conservative variables
def prim2cons(rho, vx, vy, vz, p, Bx, By, Bz):
    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2

    U1 = rho
    U2 = rho * vx
    U3 = rho * vy
    U4 = rho * vz
    U5 = p/(gamma-1.0) + 0.5*rho*v2 + 0.5*B2
    U6 = Bx
    U7 = By
    U8 = Bz

    return U1, U2, U3, U4, U5, U6, U7, U8

def cons2prim(U1, U2, U3, U4, U5, U6, U7, U8):
    rho = U1
    vx = U2 / rho
    vy = U3 / rho
    vz = U4 / rho

    Bx = U6
    By = U7
    Bz = U8

    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2

    p = (gamma - 1.0) * (U5 - 0.5*rho*v2 - 0.5*B2)
    p = np.maximum(p, 1e-6)  # Floor

    return rho, vx, vy, vz, p, Bx, By, Bz

# Flux computation (simplified HLL)
def flux_x(rho, vx, vy, vz, p, Bx, By, Bz):
    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2

    ptot = p + 0.5*B2
    E = p/(gamma-1.0) + 0.5*rho*v2 + 0.5*B2

    F1 = rho * vx
    F2 = rho*vx*vx + ptot - Bx*Bx
    F3 = rho*vx*vy - Bx*By
    F4 = rho*vx*vz - Bx*Bz
    F5 = (E + ptot)*vx - Bx*(vx*Bx + vy*By + vz*Bz)
    F6 = np.zeros_like(Bx)  # Bx constant in x
    F7 = By*vx - Bx*vy
    F8 = Bz*vx - Bx*vz

    return F1, F2, F3, F4, F5, F6, F7, F8

def flux_y(rho, vx, vy, vz, p, Bx, By, Bz):
    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2

    ptot = p + 0.5*B2
    E = p/(gamma-1.0) + 0.5*rho*v2 + 0.5*B2

    G1 = rho * vy
    G2 = rho*vy*vx - By*Bx
    G3 = rho*vy*vy + ptot - By*By
    G4 = rho*vy*vz - By*Bz
    G5 = (E + ptot)*vy - By*(vx*Bx + vy*By + vz*Bz)
    G6 = Bx*vy - By*vx
    G7 = np.zeros_like(By)  # By constant in y (with CT)
    G8 = Bz*vy - By*vz

    return G1, G2, G3, G4, G5, G6, G7, G8

# Simple HLL solver
def hll_flux_x(UL, UR, rhoL, vxL, vyL, vzL, pL, BxL, ByL, BzL,
                         rhoR, vxR, vyR, vzR, pR, BxR, ByR, BzR):
    # Estimate wave speeds
    csL = np.sqrt(gamma * pL / rhoL)
    csR = np.sqrt(gamma * pR / rhoR)

    SL = np.minimum(vxL - csL, vxR - csR)
    SR = np.maximum(vxL + csL, vxR + csR)

    FL = flux_x(rhoL, vxL, vyL, vzL, pL, BxL, ByL, BzL)
    FR = flux_x(rhoR, vxR, vyR, vzR, pR, BxR, ByR, BzR)

    # HLL flux
    F_hll = []
    for i in range(8):
        F = np.where(SL >= 0, FL[i],
            np.where(SR <= 0, FR[i],
            (SR*FL[i] - SL*FR[i] + SL*SR*(UR[i] - UL[i])) / (SR - SL)))
        F_hll.append(F)

    return F_hll

# Main solver
def solve_reconnection():
    # Initialize
    rho, vx, vy, vz, p, Bx, By, Bz = initial_harris_sheet()
    U = prim2cons(rho, vx, vy, vz, p, Bx, By, Bz)

    t = 0.0
    t_end = 50.0
    step = 0

    # Diagnostics
    energy_mag = []
    energy_kin = []
    energy_th = []
    times = []

    print("Starting reconnection simulation...")

    while t < t_end:
        # Compute dt
        cs = np.sqrt(gamma * p / rho)
        vA = np.sqrt((Bx**2 + By**2 + Bz**2) / rho)
        vmax = np.max(np.abs(vx) + np.abs(vy) + cs + vA)
        dt = CFL * min(dx, dy) / vmax

        if t + dt > t_end:
            dt = t_end - t

        # Update (simple forward Euler for demonstration; use RK2 in production)
        # X-direction fluxes
        Fx_L = []
        Fx_R = []
        for i in range(Nx):
            iL = (i - 1) % Nx
            iR = i

            UL = [U[k][iL, :] for k in range(8)]
            UR = [U[k][iR, :] for k in range(8)]

            primL = cons2prim(*UL)
            primR = cons2prim(*UR)

            F = hll_flux_x(UL, UR, *primL, *primR)
            if i == 0:
                Fx_L = [np.zeros((Nx, Ny)) for _ in range(8)]
                Fx_R = [np.zeros((Nx, Ny)) for _ in range(8)]

            for k in range(8):
                Fx_R[k][i, :] = F[k]

        # Shift for left flux
        for k in range(8):
            Fx_L[k] = np.roll(Fx_R[k], 1, axis=0)

        # Y-direction fluxes (similar, with boundary conditions)
        # ... (omitted for brevity; apply conducting wall BC)

        # Update conserved variables
        U_new = []
        for k in range(8):
            dU = -(Fx_R[k] - Fx_L[k]) / dx  # Only x-direction for simplicity
            U_new.append(U[k] + dt * dU)

        U = U_new

        # Recover primitives
        rho, vx, vy, vz, p, Bx, By, Bz = cons2prim(*U)

        # 저항성 확산항(Resistive Diffusion Term) η∇²B = η(∇×J): 이상 MHD 업데이트 후
        # 연산자 분리(Operator-Split) 단계로 적용 — 이 분리는 묵시적 처리의 강성(Stiffness)을
        # 피하면서 저항층을 물리적으로 올바르게 유지함;
        # 이 단계 없이는 시뮬레이션이 순수하게 이상적이어서 섭동 진폭에 관계없이
        # 재결합이 발생하지 않음
        Jz = (np.gradient(By, dx, axis=0) - np.gradient(Bx, dy, axis=1))
        # dBx/dt = -η * ∂Jz/∂y: 마이너스 부호는 E = ηJ (Ohm의 법칙)를 가진
        # Faraday 법칙 ∂B/∂t = -∇×E로부터; 저항성 전기장의 컬(Curl)이
        # η에 비례하는 속도로 전류 시트를 가로질러 플럭스 확산을 유발
        dBx_dt = -eta * np.gradient(Jz, dy, axis=1)
        dBy_dt = eta * np.gradient(Jz, dx, axis=0)

        Bx += dt * dBx_dt
        By += dt * dBy_dt

        # Update conserved variables with new B
        U = prim2cons(rho, vx, vy, vz, p, Bx, By, Bz)

        # Diagnostics
        if step % 50 == 0:
            # 세 에너지 저장소를 모두 추적하면 ΔE_mag = ΔE_kin + ΔE_th 확인:
            # 재결합 중 방출된 자기 에너지는 유출 제트의 체적 운동 에너지와
            # 가열된 플라즈마의 열 에너지로 나뉘어야 함 —
            # 총 에너지가 보존되지 않으면 수치 코드에 버그가 있음
            E_mag = 0.5 * np.sum((Bx**2 + By**2 + Bz**2) * dx * dy)
            E_kin = 0.5 * np.sum(rho * (vx**2 + vy**2 + vz**2) * dx * dy)
            E_th = np.sum(p / (gamma - 1.0) * dx * dy)

            energy_mag.append(E_mag)
            energy_kin.append(E_kin)
            energy_th.append(E_th)
            times.append(t)

            print(f"Step {step:5d}, t={t:6.2f}, E_mag={E_mag:.4f}, E_kin={E_kin:.4f}, E_th={E_th:.4f}")

        t += dt
        step += 1

        if step > 10000:  # Safety limit
            break

    return times, energy_mag, energy_kin, energy_th, rho, vx, vy, p, Bx, By, Jz

# Run simulation
times, E_mag, E_kin, E_th, rho_f, vx_f, vy_f, p_f, Bx_f, By_f, Jz_f = solve_reconnection()

# Plot energy evolution
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(times, E_mag, 'b-', label='Magnetic', linewidth=2)
ax.plot(times, E_kin, 'r--', label='Kinetic', linewidth=2)
ax.plot(times, E_th, 'g:', label='Thermal', linewidth=2)
E_tot = np.array(E_mag) + np.array(E_kin) + np.array(E_th)
ax.plot(times, E_tot, 'k-.', label='Total', linewidth=1.5)
ax.set_xlabel('Time')
ax.set_ylabel('Energy')
ax.set_title('Energy Evolution in Reconnection')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('reconnection_energy.png', dpi=150, bbox_inches='tight')
print("Energy plot saved: reconnection_energy.png")
plt.close()

# Plot final state
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Current density
im0 = axes[0, 0].contourf(X, Y, Jz_f, levels=30, cmap='RdBu_r')
axes[0, 0].set_title('Current Density $J_z$')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
plt.colorbar(im0, ax=axes[0, 0])

# Temperature (pressure)
im1 = axes[0, 1].contourf(X, Y, p_f, levels=30, cmap='hot')
axes[0, 1].set_title('Pressure (Temperature)')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
plt.colorbar(im1, ax=axes[0, 1])

# Velocity magnitude
v_mag = np.sqrt(vx_f**2 + vy_f**2)
im2 = axes[1, 0].contourf(X, Y, v_mag, levels=30, cmap='plasma')
axes[1, 0].streamplot(X.T, Y.T, vx_f.T, vy_f.T, color='k', density=1.0, linewidth=0.5)
axes[1, 0].set_title('Velocity Field')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
plt.colorbar(im2, ax=axes[1, 0])

# Magnetic field lines
Ay = np.zeros_like(Bx_f)  # Compute vector potential (simplified)
for i in range(1, Nx):
    Ay[i, :] = Ay[i-1, :] + Bx_f[i, :] * dy
im3 = axes[1, 1].contour(X, Y, Ay, levels=20, colors='b', linewidths=1.5)
axes[1, 1].set_title('Magnetic Field Lines')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')
axes[1, 1].set_aspect('equal')

plt.tight_layout()
plt.savefig('reconnection_final.png', dpi=150, bbox_inches='tight')
print("Final state plot saved: reconnection_final.png")
plt.close()
```

### 1.4 예상 결과

- **에너지 변환:** 자기 에너지 감소, 운동 및 열 에너지 증가
- **플라즈모이드 형성:** 전류 시트에 2차 아일랜드 나타남 (해상도 충분하면)
- **재결합률:** 유입 속도 $v_{\text{in}} \sim 0.01 - 0.1 v_A$ (Sweet-Parker $\sim \eta^{1/2}$보다 빠름)
- **온도 급상승:** X-점에서 국소 가열

### 1.5 확장

1. **국소 저항**: $\eta = \eta_0 + \eta_1 J^2$ 사용 (비정상 저항)
2. **3D 시뮬레이션**: 가이드 장 $B_z$ 추가, drift-kink 불안정성 연구
3. **입자 가속**: 테스트 입자 주입, 에너지화 추적
4. **Sweet-Parker와 비교**: 재결합률 대 $\eta$ 측정, $M_A \propto S^{-1/2}$ (Sweet-Parker) 또는 $M_A \sim 0.01$ (플라즈모이드) 검증

---

## 프로젝트 2: 토카막 붕괴 예측

### 2.1 물리 배경

**토카막(Tokamak)**: 핵융합 플라즈마를 위한 환형 자기 가둠 장치.

**핵심 개념:**
- **안전 인자(Safety Factor)**: $q(r) = r B_\phi / (R B_\theta)$ (장선 피치)
- **붕괴(Disruption)**: MHD 불안정성으로 인한 가둠의 갑작스런 손실 → 플라즈마 종료, 벽에 큰 힘

**안정성 기준:**
- **Kruskal-Shafranov**: $q > 1$ ($m=1$ kink 억제)
- **Suydam 기준**: 교환 모드에 대한 국소 안정성
- **티어링 모드**: 유리수 표면($q = m/n$)이 $\Delta' > 0$이면 불안정

### 2.2 문제 설정

**원통형 토카막 모델 (1D):**
- 소반경: $a = 1.0$ m
- 대반경: $R_0 = 3.0$ m
- 환형 장: $B_\phi(r) = B_0 R_0 / (R_0 + r)$ (근사)
- 전류 프로파일 $J_\phi(r)$로부터 폴로이달 장

**테스트할 전류 프로파일:**
- 프로파일 1: 축에서 뾰족 (안정)
  $$
  J_\phi(r) = J_0 \left(1 - (r/a)^2\right)^2
  $$
- 프로파일 2: 중공 전류 (불안정)
  $$
  J_\phi(r) = J_0 (r/a) \left(1 - (r/a)^2\right)
  $$
- 프로파일 3: 에지 뾰족 (붕괴 취약)
  $$
  J_\phi(r) = J_0 \left(1 - \exp(-(r/a)^4)\right)
  $$

**목표:** $q(r)$ 계산, 안정성 확인, 붕괴 예측.

### 2.3 구현

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Parameters
a = 1.0      # Minor radius (m)
R0 = 3.0     # Major radius (m)
B0 = 2.0     # Toroidal field at axis (T)
Nr = 200
r = np.linspace(0, a, Nr)

# Current profiles
def current_profile(r, profile_type='peaked'):
    if profile_type == 'peaked':
        J = 1.0e6 * (1.0 - (r/a)**2)**2
    elif profile_type == 'hollow':
        J = 1.0e6 * (r/a) * (1.0 - (r/a)**2)
    elif profile_type == 'edge':
        J = 1.0e6 * (1.0 - np.exp(-(r/a)**4))
    else:
        J = np.ones_like(r) * 1.0e6
    return J

# Compute poloidal field from Ampere's law
def compute_Btheta(r, J):
    """B_theta(r) = (mu_0 / r) * integral_0^r J(r') r' dr'"""
    mu0 = 4e-7 * np.pi
    # integrand = J*r: 원통 좌표계에서 Ampere 법칙의 적분 ∫J·dA는
    # 면적 요소 r*dr*dφ를 필요로 함; 방위각에 대해 적분하면 2π가 나오고,
    # 마지막에 2π*r로 나누면 Bθ가 복원 — r에 의한 가중이 더 큰 소반경에서
    # 쉘의 증가하는 원주를 설명함
    integrand = J * r
    I_enc = cumulative_trapezoid(integrand, r, initial=0)
    # r로 나누면 Bθ = μ₀*I(r)/(2π*r): 안전 인자(Safety Factor) q = r*Bφ/(R*Bθ)는
    # 유한한 Bφ에 대해 r=0에서 발산하므로, 작은 오프셋이 그곳의 특이점을 피하면서
    # 물리적으로 의미 있는 모든 반경(r > 0.01 m)에는 영향을 주지 않음
    Btheta = mu0 * I_enc / (r + 1e-10)  # Avoid r=0
    return Btheta

# Toroidal field (approximate, ignoring r/R0 correction)
def compute_Bphi(r):
    return B0 * R0 / (R0 + r)

# Safety factor
def compute_q(r, Btheta, Bphi):
    """q = r * Bphi / (R0 * Btheta)"""
    q = r * Bphi / (R0 * Btheta + 1e-10)
    return q

# Suydam criterion: d(ln p) / d(ln r) + (r * dq/dr / q)^2 > 0
def suydam_criterion(r, q, p):
    """Simplified: check d(ln q)/d(ln r) > some threshold"""
    dq_dr = np.gradient(q, r)
    shear = r / q * dq_dr
    # Simplified: shear > 0.5 (stable)
    return shear > 0.5

# Tearing mode Delta' (simplified estimate)
def estimate_delta_prime(r, q, m, n):
    """Find rational surface q = m/n, estimate Delta'"""
    q_rational = m / n
    idx = np.argmin(np.abs(q - q_rational))
    rs = r[idx]

    if idx > 5 and idx < Nr - 5:
        # Estimate logarithmic derivative mismatch
        dq_dr = np.gradient(q, r)
        delta_prime = -2.0 * dq_dr[idx] / q[idx]  # Crude estimate
    else:
        delta_prime = 0.0

    return rs, delta_prime

# Analyze stability for a profile
def analyze_stability(profile_type):
    J = current_profile(r, profile_type)
    Btheta = compute_Btheta(r, J)
    Bphi = compute_Bphi(r)
    q = compute_q(r, Btheta, Bphi)

    # Simple pressure profile (proportional to current)
    p = 1e5 * (1.0 - (r/a)**2)**2

    # Kruskal-Shafranov q > 1: q가 축 위에서 1 아래로 떨어지면,
    # m=1, n=1 내부 킨크 모드(Internal Kink Mode)가 이상적으로 불안정해짐 —
    # 장선이 한 번의 환형 회전보다 짧게 폴로이달 완전 회전을 완료하여,
    # 킨크가 자기 장선 굽힘 안정화 없이 성장할 수 있음
    q_min = np.min(q[1:])  # Skip r=0
    ks_stable = q_min > 1.0

    # Suydam 기준(Suydam Criterion): 국소 전단 s = (r/q)*dq/dr이
    # 장선 피치가 반경에 따라 얼마나 빠르게 변하는지 측정;
    # 더 높은 전단이 더 짧은 접속 길이에 걸쳐 섭동이 장선을 구부러지게 해야 하므로
    # 교환 모드(Interchange Modes)를 안정화함 — s > 0.5 임계값은 완전한 Suydam 판별식의 단순화된 대리
    shear = np.zeros_like(r)
    shear[1:] = r[1:] / q[1:] * np.gradient(q, r)[1:]
    suydam_stable = np.all(shear[1:] > 0.5)

    # q=2 유리수 표면(Rational Surface)에서 m=2, n=1 티어링 모드:
    # 이것이 토카막에서 가장 위험한 티어링 모드인 이유는 q=2가 구동이 강한
    # 플라즈마 내부 깊은 곳에 위치하기 때문; Δ' > 0은 아일랜드가 열릴 때
    # 플럭스가 방출됨을 의미하여 표면이 불안정 — 이것이 잠금(Locks)되어 붕괴에 앞서는 모드
    rs_21, delta_prime_21 = estimate_delta_prime(r, q, m=2, n=1)
    tearing_stable = delta_prime_21 < 0

    return {
        'r': r,
        'J': J,
        'q': q,
        'Btheta': Btheta,
        'Bphi': Bphi,
        'q_min': q_min,
        'ks_stable': ks_stable,
        'suydam_stable': suydam_stable,
        'tearing_stable': tearing_stable,
        'delta_prime_21': delta_prime_21,
        'rs_21': rs_21
    }

# Analyze all profiles
profiles = ['peaked', 'hollow', 'edge']
results = {ptype: analyze_stability(ptype) for ptype in profiles}

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

colors = {'peaked': 'blue', 'hollow': 'red', 'edge': 'green'}

for i, ptype in enumerate(profiles):
    res = results[ptype]

    # Current profile
    axes[0, i].plot(res['r'], res['J']/1e6, color=colors[ptype], linewidth=2)
    axes[0, i].set_xlabel('r (m)')
    axes[0, i].set_ylabel('$J_\\phi$ (MA/m²)')
    axes[0, i].set_title(f'{ptype.capitalize()} Current')
    axes[0, i].grid(True, alpha=0.3)

    # Safety factor
    axes[1, i].plot(res['r'], res['q'], color=colors[ptype], linewidth=2, label='q(r)')
    axes[1, i].axhline(1.0, color='k', linestyle='--', label='q=1 (Kruskal)')
    axes[1, i].axhline(2.0, color='gray', linestyle=':', label='q=2')
    axes[1, i].set_xlabel('r (m)')
    axes[1, i].set_ylabel('Safety Factor q')
    axes[1, i].set_title(f'q(r) - {ptype.capitalize()}')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)
    axes[1, i].set_ylim([0, 10])

plt.tight_layout()
plt.savefig('tokamak_profiles.png', dpi=150, bbox_inches='tight')
print("Profile plots saved: tokamak_profiles.png")
plt.close()

# Stability summary
print("\n=== STABILITY ANALYSIS ===\n")
for ptype in profiles:
    res = results[ptype]
    print(f"--- {ptype.upper()} PROFILE ---")
    print(f"  q_min = {res['q_min']:.3f}")
    print(f"  Kruskal-Shafranov (q>1): {'STABLE' if res['ks_stable'] else 'UNSTABLE'}")
    print(f"  Suydam: {'STABLE' if res['suydam_stable'] else 'UNSTABLE'}")
    print(f"  Tearing (2,1) Delta' = {res['delta_prime_21']:.3f}: {'STABLE' if res['tearing_stable'] else 'UNSTABLE'}")
    print(f"  Rational surface (q=2): r_s = {res['rs_21']:.3f} m")

    # Disruption prediction
    if not res['ks_stable']:
        print(f"  PREDICTION: HIGH DISRUPTION RISK (Kruskal violation)")
    elif not res['tearing_stable']:
        print(f"  PREDICTION: MEDIUM DISRUPTION RISK (Tearing unstable)")
    else:
        print(f"  PREDICTION: LOW DISRUPTION RISK")
    print()

# Estimate energy release during disruption
def estimate_disruption_energy(res):
    """Magnetic + thermal energy in plasma"""
    B2 = res['Btheta']**2 + res['Bphi']**2
    V = 2 * np.pi * R0 * np.pi * a**2  # Approximate volume
    E_mag = np.trapz(B2 / (2 * 4e-7*np.pi) * 2*np.pi*R0*res['r'], res['r'])

    p = 1e5 * (1.0 - (res['r']/a)**2)**2
    E_th = np.trapz(1.5 * p * 2*np.pi*R0*res['r'], res['r'])

    return E_mag, E_th

print("\n=== DISRUPTION ENERGY ESTIMATE ===\n")
for ptype in profiles:
    res = results[ptype]
    E_mag, E_th = estimate_disruption_energy(res)
    print(f"{ptype.upper()}: E_mag = {E_mag/1e6:.2f} MJ, E_th = {E_th/1e6:.2f} MJ")
    print(f"  Total = {(E_mag + E_th)/1e6:.2f} MJ")
    print()
```

### 2.4 예상 결과

- **뾰족 프로파일:** 안정 (모든 기준 만족)
- **중공 프로파일:** 한계 ($q=2$ 표면에서 티어링 불안정성 가능)
- **에지 프로파일:** 불안정 (Kruskal 위반, 붕괴 가능성)

**붕괴 에너지:** ~10-100 MJ (벽에 힘 ~ MN, 완화 필요!)

### 2.5 확장

1. **신고전 티어링 모드(NTM)**: 부트스트랩 전류 섭동 포함
2. **수직 변위 사건(VDE)**: 축대칭 불안정성
3. **저항성 벽 모드**: 벽 안정화 포함
4. **붕괴 완화**: 대량 가스 주입(MGI) 시뮬레이션

---

## 프로젝트 3: 구형 쉘 다이나모

### 3.1 물리 배경

**다이나모 이론:** 유체 운동에 의한 자기장 자체 유지 생성 (예: 지구 핵, 태양 대류 영역).

**핵심 개념:**
- **운동학적 다이나모(Kinematic Dynamo)**: 속도장 $\mathbf{v}$ 규정, $\mathbf{B}$ 풀기
- **평균장 이론**: $\mathbf{B} = \mathbf{B}_0 + \mathbf{b}$ 분리, 난류 EMF $\langle \mathbf{v} \times \mathbf{b} \rangle = \alpha \mathbf{B}_0 - \beta \mathbf{J}$
- **$\alpha$-효과**: 사이클론 난류 → 나선형 장 생성
- **$\Omega$-효과**: 미분 회전 → 폴로이달에서 환형 장

**축대칭 평균장 방정식:**
$$
\frac{\partial A}{\partial t} = \eta \nabla^2 A + \alpha B_\phi
$$
$$
\frac{\partial B_\phi}{\partial t} = \eta \nabla^2 B_\phi + r \sin\theta \, \mathbf{B}_p \cdot \nabla \Omega + \ldots
$$
여기서 $\mathbf{B}_p = \nabla \times (A \hat{\phi})$ (폴로이달 장).

### 3.2 문제 설정

**영역:** 구형 쉘 $r \in [r_i, r_o]$, $\theta \in [0, \pi]$ (축대칭, $\phi$-무관)

**매개변수:**
- 내부 반경: $r_i = 0.5$
- 외부 반경: $r_o = 1.0$
- 미분 회전: $\Omega(r, \theta) = \Omega_0 \cos^2\theta \, (1 - (r/r_o)^2)$ (태양형)
- $\alpha$-효과: $\alpha(r, \theta) = \alpha_0 \sin\theta \cos\theta$ (쌍극자 선호)
- 자기 확산도: $\eta = 10^{-3}$
- $\alpha_0 = 1.0$, $\Omega_0 = 10.0$

**방정식 (단순화된 2D):**
$$
\frac{\partial A}{\partial t} = \eta \left(\frac{\partial^2 A}{\partial r^2} + \frac{1}{r^2}\frac{\partial^2 A}{\partial \theta^2}\right) + \alpha B_\phi
$$
$$
\frac{\partial B_\phi}{\partial t} = \eta \left(\frac{\partial^2 B_\phi}{\partial r^2} + \frac{1}{r^2}\frac{\partial^2 B_\phi}{\partial \theta^2}\right) + C_\Omega \frac{\partial A}{\partial \theta}
$$
여기서 $C_\Omega = r \sin\theta \, \partial_r \Omega$.

**경계 조건:**
- $r = r_i, r_o$에서 $A = 0$ (표면에 평행한 장선)
- $r = r_i, r_o$에서 $B_\phi = 0$ (외부 진공)

### 3.3 구현

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
ri, ro = 0.5, 1.0
Nr, Nth = 64, 64
r = np.linspace(ri, ro, Nr)
theta = np.linspace(0, np.pi, Nth)
R, Theta = np.meshgrid(r, theta, indexing='ij')

dr = (ro - ri) / (Nr - 1)
dtheta = np.pi / (Nth - 1)

eta = 1e-3
alpha0 = 1.0
Omega0 = 10.0
dt = 1e-4
t_end = 2.0

# Ω ∝ cos²θ*(1-r²): 적도와 핵에서 더 빠른 회전 — 이 위도 및 반경 방향의
# 미분 회전이 Ω-효과를 구동: 폴로이달 장선을 다른 속도로 환형 방향으로 늘려
# 태양 다이나모 주기를 특징짓는 강한 환형 장(태양 흑점대)을 생성함
Omega = Omega0 * np.cos(Theta)**2 * (1.0 - (R/ro)**2)

# α ∝ sinθ*cosθ: 적도에 대해 반대칭 — 북반구에서는 양수, 남반구에서는 음수;
# 이 반대칭은 α-효과를 통해 환형 장에서 쌍극자형(Dipole-Like) 폴로이달 장을 생성하는 데 필요하며,
# 대칭적인 α는 대신 사중극자(Quadrupole)를 생성할 것임
alpha = alpha0 * np.sin(Theta) * np.cos(Theta)

# C_Ω = r*sinθ * ∂Ω/∂r은 Ω의 국소 전단을 측정: 이것이
# 폴로이달 플럭스 A의 기울기를 환형 장 Bφ의 소스항으로 변환하는 계수 —
# 미분 회전이 강한 곳(|C_Ω|이 큰 곳)에서 Ω-효과가 폴로이달 장을 환형 장으로
# 감는 데 가장 효율적임
C_Omega = R * np.sin(Theta) * np.gradient(Omega, dr, axis=0)

# Initialize fields
A = np.random.randn(Nr, Nth) * 0.01
Bphi = np.random.randn(Nr, Nth) * 0.01

# Apply BC
A[0, :] = 0
A[-1, :] = 0
Bphi[0, :] = 0
Bphi[-1, :] = 0

# Laplacian operator (finite differences)
def laplacian_2d(f, dr, dtheta, r):
    """Compute Laplacian in (r, theta) with metric terms."""
    d2f_dr2 = np.zeros_like(f)
    d2f_dtheta2 = np.zeros_like(f)

    # r-direction (central differences)
    d2f_dr2[1:-1, :] = (f[2:, :] - 2*f[1:-1, :] + f[:-2, :]) / dr**2

    # theta-direction
    d2f_dtheta2[:, 1:-1] = (f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2]) / dtheta**2

    # Add metric terms (simplified)
    laplacian = d2f_dr2 + d2f_dtheta2 / r[:, np.newaxis]**2

    return laplacian

# Time evolution
t = 0.0
step = 0
snapshots = []

print("Running spherical shell dynamo...")
while t < t_end:
    # Compute Laplacians
    lap_A = laplacian_2d(A, dr, dtheta, r)
    lap_Bphi = laplacian_2d(Bphi, dr, dtheta, r)

    # Source terms
    dA_dt_theta = np.gradient(A, dtheta, axis=1)

    source_A = alpha * Bphi
    source_Bphi = C_Omega * dA_dt_theta

    # Update
    A += dt * (eta * lap_A + source_A)
    Bphi += dt * (eta * lap_Bphi + source_Bphi)

    # Apply BC
    A[0, :] = 0
    A[-1, :] = 0
    Bphi[0, :] = 0
    Bphi[-1, :] = 0

    # Diagnostics
    if step % 1000 == 0:
        E_A = np.sum(A**2)
        E_Bphi = np.sum(Bphi**2)
        print(f"Step {step:5d}, t={t:.4f}, E_A={E_A:.4f}, E_Bphi={E_Bphi:.4f}")

        if len(snapshots) < 5:
            snapshots.append((t, A.copy(), Bphi.copy()))

    t += dt
    step += 1

print("Dynamo simulation complete.")

# Plot snapshots
fig, axes = plt.subplots(len(snapshots), 2, figsize=(12, 3*len(snapshots)))

for i, (t_snap, A_snap, Bphi_snap) in enumerate(snapshots):
    ax_A = axes[i, 0] if len(snapshots) > 1 else axes[0]
    ax_B = axes[i, 1] if len(snapshots) > 1 else axes[1]

    # Poloidal field (contours of A)
    im_A = ax_A.contourf(R*np.sin(Theta), R*np.cos(Theta), A_snap, levels=20, cmap='RdBu_r')
    ax_A.set_xlabel('x')
    ax_A.set_ylabel('z')
    ax_A.set_title(f'Poloidal Field (A), t={t_snap:.3f}')
    ax_A.set_aspect('equal')
    plt.colorbar(im_A, ax=ax_A)

    # Toroidal field
    im_B = ax_B.contourf(R*np.sin(Theta), R*np.cos(Theta), Bphi_snap, levels=20, cmap='seismic')
    ax_B.set_xlabel('x')
    ax_B.set_ylabel('z')
    ax_B.set_title(f'Toroidal Field $B_\\phi$, t={t_snap:.3f}')
    ax_B.set_aspect('equal')
    plt.colorbar(im_B, ax=ax_B)

plt.tight_layout()
plt.savefig('dynamo_evolution.png', dpi=150, bbox_inches='tight')
print("Dynamo evolution saved: dynamo_evolution.png")
plt.close()

# Butterfly diagram (Bphi vs time at mid-radius)
r_mid_idx = Nr // 2
butterfly_Bphi = []
butterfly_times = []

# Re-run to collect time series (or store during main loop)
# For demonstration, use final state
butterfly_Bphi.append(Bphi[r_mid_idx, :])
butterfly_times.append(t_end)

fig, ax = plt.subplots(figsize=(10, 6))
if len(butterfly_times) > 1:
    ax.contourf(butterfly_times, theta, np.array(butterfly_Bphi).T, levels=30, cmap='RdBu_r')
else:
    ax.plot(theta, Bphi[r_mid_idx, :], 'b-', linewidth=2)
    ax.set_xlabel('$\\theta$ (rad)')
    ax.set_ylabel('$B_\\phi$')
ax.set_title(f'Butterfly Diagram (r={r[r_mid_idx]:.2f})')
ax.grid(True, alpha=0.3)
plt.savefig('butterfly_diagram.png', dpi=150, bbox_inches='tight')
print("Butterfly diagram saved: butterfly_diagram.png")
plt.close()
```

### 3.4 예상 결과

- **진동 다이나모:** $A$와 $B_\phi$가 시간에 따라 진동 (자기 주기)
- **적도 방향 이동:** 환형 장이 극에서 적도로 이동 ($\alpha$-$\Omega$ 매개변수가 적절하면)
- **나비 다이어그램:** $B_\phi$의 위도-시간 진화 표시 ($C_\Omega$와 $\alpha$가 잘 선택되면 태양형)

**Parker 이동 다이나모:** 적절한 매개변수로 태양 22년 주기 재현!

### 3.5 확장

1. **3D 다이나모**: 완전한 구 좌표계, 대류 포함
2. **비선형 quenching**: $\alpha \to \alpha(B)$ (Lorentz 힘 역반응)
3. **반전**: 확률적 $\alpha$ 변동 → 극성 반전 (지구 장)
4. **벤치마크**: Dedalus 스펙트럴 코드와 비교

---

## 프로젝트 요약

| 프로젝트 | 물리 | 방법 | 난이도 |
|---------|---------|---------|------------|
| **태양 플레어** | 자기 재결합, 플라즈모이드 | 2D 저항성 MHD, HLL 솔버 | ★★★★ |
| **토카막 붕괴** | MHD 안정성, 안전 인자 | 1D 평형, 안정성 기준 | ★★★ |
| **다이나모** | 평균장 이론, $\alpha$-$\Omega$ | 2D 축대칭, 시간 진화 | ★★★★ |

**연습된 기술:**
- 보존형 MHD 솔버 (Godunov 유형)
- 평형 분석과 안정성 이론
- 평균장 근사
- 수치 결과의 물리적 해석

---

## 프로젝트를 위한 일반 팁

1. **간단하게 시작**: 2D 전에 1D를 작동시키고, 고해상도 전에 저해상도
2. **검증**: 알려진 해와 비교 (예: 프로젝트 1을 위한 Brio-Wu)
3. **진단**: 에너지 보존, $\nabla \cdot \mathbf{B}$, CFL 타임스텝 모니터
4. **시각화**: $\mathbf{B}$에 대해 streamplot 사용, 스칼라 장에 대해 윤곽
5. **매개변수 스캔**: $\eta$, $Re$, 그리드 해상도 변화 → 수렴 연구
6. **물리적 직관**: 결과가 의미가 있습니까? (예: 재결합은 플라즈마를 가열해야 함)

---

## 결론

이 세 가지 프로젝트는 전체 MHD 과정을 통합합니다:

- **보존 법칙:** 질량, 운동량, 에너지 (프로젝트 1)
- **파동 물리:** 고속, 저속, Alfvén (프로젝트 1)
- **안정성:** Kruskal-Shafranov, Suydam, 티어링 (프로젝트 2)
- **다이나모 이론:** $\alpha$-효과, 미분 회전 (프로젝트 3)
- **수치 방법:** 유한 체적, Riemann 솔버, 시간 적분 (모든 프로젝트)

이것들을 완료함으로써, **대학원 수준 자기유체역학**을 마스터했습니다!

---

## 추가 읽을거리

### 태양 재결합
- Zweibel & Yamada (2009), *Magnetic Reconnection in Astrophysical and Laboratory Plasmas*, ARA&A
- Loureiro et al. (2007), *Instability of current sheets and formation of plasmoid chains*, Physics of Plasmas

### 토카막 안정성
- Wesson & Campbell (2011), *Tokamaks*, 4th Ed., Oxford
- Freidberg (2014), *Ideal MHD*, Cambridge

### 다이나모 이론
- Moffatt (1978), *Magnetic Field Generation in Electrically Conducting Fluids*, Cambridge
- Brandenburg & Subramanian (2005), *Astrophysical magnetic fields and nonlinear dynamo theory*, Physics Reports

### 수치 MHD
- Tóth et al. (2012), *Adaptive numerical algorithms in space weather modeling*, JCP
- Stone et al. (2020), *Athena++: A Fast, Portable, and Multi-Physics PDE Solver*, ApJS

---

## 연습 문제

### 연습 1: 해리스 시트(Harris Sheet) 평형 검증

완전한 재결합 시뮬레이션을 실행하기 전에, 초기 해리스 전류 시트가 힘 균형(Force Balance) 상태에 있는지 검증하세요.

1. 프로젝트 1의 해리스 시트 초기 조건을 사용하여, $y$ 방향의 각 격자점에서 총 압력 $p_{\text{tot}}(y) = p(y) + B^2(y)/2$를 계산하는 Python 스크립트를 작성하세요.
2. $p_{\text{tot}}(y)$, $p(y)$, $B^2(y)/2$를 같은 축에 그리세요.
3. $p_{\text{tot}}$가 시트 전체에서 (수치 정밀도 범위 내에서) 일정함을 확인하세요.
4. 평균으로부터의 최대 상대 편차 $\max |p_{\text{tot}} - \langle p_{\text{tot}} \rangle| / \langle p_{\text{tot}} \rangle$를 계산하세요. 이상적인 평형에서 어떤 값이 예상됩니까?
5. 전류 시트 내부($y \approx 0$)에서 힘 균형을 유지하기 위해 플라즈마 압력이 높아져야 하는 물리적 이유를 설명하세요.

### 연습 2: 안전 인자(Safety Factor)와 룬퀴스트 수(Lundquist Number) 스케일링

핵심 무차원 매개변수가 MHD 거동을 어떻게 제어하는지 탐구하세요.

1. 프로젝트 2의 토카막 모델에서, **균일 전류 프로파일** $J_\phi = J_0 = \text{상수}$에 대한 안전 인자(Safety Factor) $q(r)$를 계산하세요.
   - 해석적으로 유도하세요: $q(r) = 2 B_\phi r / (\mu_0 J_0 R_0)$.
   - `compute_q`에서 얻은 수치 결과를 이 공식과 비교하세요.
2. 프로젝트 1의 재결합 시뮬레이션에서, 룬퀴스트 수는 $S = a v_A / \eta$입니다. $v_A = B_0 / \sqrt{\rho_0} = 1$, $a = 0.5$일 때:
   - $\eta \in \{0.01, 0.001, 0.0001\}$에 대해 $S$를 계산하세요.
   - 각 $S$에 대해, Sweet-Parker 재결합률 $M_A^{\text{SP}} = S^{-1/2}$와 Sweet-Parker 전류 시트 두께 $\delta^{\text{SP}} = a S^{-1/2}$를 추정하세요.
   - 표를 완성하세요:

     | $\eta$ | $S$ | $M_A^{\text{SP}}$ | $\delta^{\text{SP}}$ |
     |--------|-----|-------------------|----------------------|
     | 0.01   |     |                   |                      |
     | 0.001  |     |                   |                      |
     | 0.0001 |     |                   |                      |

3. 플라즈모이드 불안정성(Plasmoid Instability)이 중요해지는 룬퀴스트 수는 얼마입니까 (힌트: $S \gtrsim 10^4$)? 재결합 역학에서 무엇이 변합니까?

### 연습 3: 티어링 모드(Tearing Mode) 안정성 지도

2D 안정성 지도를 구축하도록 토카막 안정성 분석을 확장하세요.

1. 프로젝트 2의 안정성 분석 코드를 사용하여 두 매개변수를 스캔하세요:
   - 전류 뾰족도 지수 $\nu$: 뾰족 프로파일을 $J_\phi(r) = J_0 (1 - (r/a)^2)^\nu$로 일반화하여 $\nu \in \{0.5, 1.0, 1.5, 2.0, 3.0\}$.
   - 총 플라즈마 전류: $q_0 = q(r \to 0) \in \{1.2, 1.5, 2.0, 3.0\}$이 되도록 $J_0$ 조정.
2. 각 조합에 대해 기록하세요:
   - $q_{\text{min}}$ (Kruskal-Shafranov 안정성: $q_{\text{min}} > 1$?)
   - $q = 2$ 유리수 표면(Rational Surface) $r_s$의 위치
   - 티어링 모드 $\Delta'$ 부호
3. $\nu$를 한 축에, $q_0$를 다른 축에 두고, 안정성으로 색상 코딩된 2D 컬러 맵(히트맵)을 만드세요 (녹색 = 안정, 빨간색 = 불안정, 노란색 = 한계).
4. 안정성 경계를 파악하세요: $(\nu, q_0)$의 어떤 조합이 가장 붕괴에 취약합니까?

### 연습 4: 다이나모 수(Dynamo Number)와 주기

평균장 다이나모의 진동 주파수를 다이나모 수가 어떻게 제어하는지 조사하세요.

1. 다이나모 수는 $D = C_\alpha C_\Omega$로 정의됩니다. 여기서 $C_\alpha = \alpha_0 L / \eta$, $C_\Omega = \Omega_0 L^2 / \eta$ ($L = r_o - r_i$). 프로젝트 3 매개변수에 대해:
   - $\alpha_0 = 1.0$, $\Omega_0 = 10.0$, $\eta = 10^{-3}$, $L = 0.5$를 사용하여 $D$를 계산하세요.
2. $\Omega_0 = 10.0$을 고정한 채 $\alpha_0 \in \{0.5, 1.0, 2.0, 4.0\}$을 스캔하도록 다이나모 코드를 수정하세요. 각 실행에 대해:
   - 총 자기 에너지 $E(t) = \sum (A^2 + B_\phi^2)$를 시간의 함수로 기록하세요.
   - $E(t)$에 지수 함수를 피팅하여 다이나모가 성장, 감쇠, 또는 진동하는지 판단하세요.
   - 진동하면, 에너지 최대값 사이의 시간으로부터 주기 $T$를 추정하세요.
3. 성장률 $\gamma$ (또는 주기 $T$) 대 $D$를 그리세요. 장이 감쇠하는 임계 다이나모 수 $D_c$는 얼마입니까?
4. 물리적으로 설명하세요: 왜 $\alpha_0$를 증가시키면 성장률이 높아지고 동시에 주기가 짧아집니까?

### 연습 5: 통합 MHD 프로젝트 — 코로나 질량 방출(CME, Coronal Mass Ejection) 개시 모델

세 프로젝트의 요소를 결합하여 단순화된 코로나 질량 방출(CME) 개시 모델을 설계하고 구현하세요.

**배경:** CME는 태양 활동 영역 위의 자속 로프(Flux Rope)가 평형을 잃고 폭발할 때 개시됩니다. 이는 힘 없는(Force-Free) 자기장 구성이 불안정해지고(킨크 또는 토러스 불안정성) 위에 있는 장과 재결합을 겪는 과정을 포함합니다.

**과제:**

1. **평형 설정:** 데카르트 기하학에서 2D 자속 로프 평형을 생성하세요:
   - 배경 장: $B_y(x) = B_{\text{ext}} \tanh(x / L)$ (아케이드 장)
   - 내장된 자속 로프: $(x_0, y_0) = (0, 1)$에 중심을 두고, 반경 $R_{\text{rope}} = 0.5$, 전류 $I_{\text{rope}}$인 국소 전류 루프 추가
   - 로프 기여를 위한 비오-사바르(Biot-Savart) 법칙을 사용하여 총 자기장 $\mathbf{B} = \mathbf{B}_{\text{아케이드}} + \mathbf{B}_{\text{로프}}$를 계산하세요.

2. **안정성 분석:** 토러스 불안정성(Torus Instability) 기준을 적용하세요. 자속 로프는 다음 조건에서 폭발합니다:
   $$n = -\frac{\partial \ln B_{\text{ext}}}{\partial \ln h} > n_{\text{crit}} \approx 1.5$$
   여기서 $h$는 자속 로프 중심의 높이입니다. 배경 장의 감쇠 지수(Decay Index) $n(h)$를 높이의 함수로 계산하고 임계 높이 $h_c$를 결정하세요.

3. **동적 진화:** 프로젝트 1의 저항성 MHD 솔버를 자속 로프 평형으로 초기화하세요. 로프를 위쪽으로 $\delta h = 0.1$ 만큼 섭동하고 진화시키세요. 다음을 추적하세요:
   - 로프 중심 높이 $h(t)$ ($|J_z|$ 최대값의 중심을 찾아서)
   - 로프 아래 전류 시트에서의 재결합률
   - 에너지 분배: $\Delta E_{\text{mag}}$ 대 $\Delta E_{\text{kin}}$

4. **물리적 해석:**
   - 로프가 폭발합니까 (지수적 상승) 아니면 평형으로 돌아갑니까 (진동)?
   - 재결합은 로프의 위에서 발생합니까, 아래에서 발생합니까?
   - 시뮬레이션된 역학을 표준 CSHKP 플레어 모델(Carmichael-Sturrock-Hirayama-Kopp-Pneuman)과 비교하세요.
   - 로프 매개변수를 사용하여 태양 플레어에 이용 가능한 에너지를 추정하세요.

---

**이전:** [스펙트럴 및 고급 방법](./17_Spectral_Methods.md) | **다음:** 없음 (마지막 레슨)
