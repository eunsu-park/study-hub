# 13. CFD 기초 (Computational Fluid Dynamics Basics)

## 학습 목표
- 유체역학의 기본 원리와 지배방정식 이해
- 레이놀즈 수와 유동 특성 관계 파악
- Navier-Stokes 방정식의 유도와 의미 이해
- 압축성/비압축성 유동 구분
- 경계층 개념 학습
- 간단한 채널 유동 CFD 구현

**이 레슨이 중요한 이유:** 유체역학은 혈관 내 혈류부터 기상 예측, 항공기 설계에 이르기까지 엄청나게 다양한 현상을 지배합니다. 유체 운동을 기술하는 나비에-스토크스(Navier-Stokes) 방정식은 해석해가 거의 존재하지 않는 비선형 PDE입니다. 전산유체역학(CFD, Computational Fluid Dynamics)은 이러한 방정식을 수치적으로 풀어 이 간극을 메우며, 수치 시뮬레이션의 가장 영향력 있는 응용 분야 중 하나입니다.

---

## 1. 유체역학 기초

### 1.1 연속체 가정

연속체 가정(Continuum Hypothesis)은 유체 모델링에 PDE를 사용할 수 있게 하는 기본 가정입니다. 유동의 길이 스케일이 분자 평균 자유 경로(Mean Free Path)보다 훨씬 큰 경우, 개별 분자를 추적하는 대신 유체를 연속 매질로 취급할 수 있다고 말합니다.

```
연속체 가정 (Continuum Hypothesis):
- 유체를 연속적인 매질로 취급
- 개별 분자 대신 유체 입자(fluid particle) 개념 사용
- Knudsen 수 Kn = λ/L << 1 일 때 유효
  (λ: 평균 자유 경로, L: 특성 길이)
```

### 1.2 기본 물성치

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 유체 물성치 예시
class FluidProperties:
    """유체 물성치 클래스"""

    def __init__(self, name, rho, mu, k=None, cp=None):
        """
        Parameters:
        - name: 유체 이름
        - rho: 밀도 [kg/m³]
        - mu: 동점성계수 [Pa·s]
        - k: 열전도도 [W/(m·K)]
        - cp: 비열 [J/(kg·K)]
        """
        self.name = name
        self.rho = rho      # 밀도
        self.mu = mu        # 동점성계수 (dynamic viscosity)
        self.k = k          # 열전도도
        self.cp = cp        # 정압비열

    @property
    def nu(self):
        """운동점성계수 (kinematic viscosity)"""
        return self.mu / self.rho

    @property
    def alpha(self):
        """열확산계수 (thermal diffusivity)"""
        if self.k and self.cp:
            return self.k / (self.rho * self.cp)
        return None

    @property
    def Pr(self):
        """프란틀 수 (Prandtl number)"""
        if self.cp:
            return self.mu * self.cp / self.k
        return None

    def __repr__(self):
        return f"FluidProperties({self.name}): rho={self.rho}, mu={self.mu}, nu={self.nu:.2e}"

# 일반적인 유체들
water = FluidProperties("Water (20°C)", rho=998, mu=1.002e-3, k=0.598, cp=4182)
air = FluidProperties("Air (20°C)", rho=1.204, mu=1.825e-5, k=0.0257, cp=1007)
oil = FluidProperties("Engine Oil (20°C)", rho=880, mu=0.29, k=0.145, cp=1880)

print(water)
print(f"  Prandtl number: {water.Pr:.2f}")
print(air)
print(f"  Prandtl number: {air.Pr:.2f}")
```

### 1.3 레이놀즈 수 (Reynolds Number)

```
레이놀즈 수 정의:
Re = ρUL/μ = UL/ν = (관성력)/(점성력)

여기서:
- ρ: 유체 밀도
- U: 특성 속도
- L: 특성 길이
- μ: 동점성계수
- ν = μ/ρ: 운동점성계수

유동 특성:
- Re < 2300: 층류 (Laminar flow)
- 2300 < Re < 4000: 천이 영역 (Transition)
- Re > 4000: 난류 (Turbulent flow)
```

```python
def reynolds_number_analysis():
    """레이놀즈 수와 유동 특성 분석"""

    # 관 유동 예제
    D = 0.05  # 관 직경 [m]
    U_range = np.linspace(0.01, 2.0, 100)  # 유속 범위 [m/s]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 물에서의 레이놀즈 수
    ax1 = axes[0]
    Re_water = water.rho * U_range * D / water.mu

    ax1.plot(U_range, Re_water, 'b-', linewidth=2, label='Water')
    ax1.axhline(y=2300, color='orange', linestyle='--', label='Transition start')
    ax1.axhline(y=4000, color='red', linestyle='--', label='Turbulent')

    ax1.fill_between(U_range, 0, 2300, alpha=0.2, color='green', label='Laminar')
    ax1.fill_between(U_range, 2300, 4000, alpha=0.2, color='orange')
    ax1.fill_between(U_range, 4000, max(Re_water), alpha=0.2, color='red')

    ax1.set_xlabel('Flow Velocity U [m/s]')
    ax1.set_ylabel('Reynolds Number Re')
    ax1.set_title(f'Reynolds Number in Pipe Flow (D = {D*100} cm, Water)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 다양한 유체 비교
    ax2 = axes[1]
    fluids = [water, air, oil]
    colors = ['blue', 'cyan', 'brown']

    for fluid, color in zip(fluids, colors):
        Re = fluid.rho * U_range * D / fluid.mu
        ax2.plot(U_range, Re, color=color, linewidth=2, label=fluid.name)

    ax2.axhline(y=2300, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Flow Velocity U [m/s]')
    ax2.set_ylabel('Reynolds Number Re')
    ax2.set_title('Reynolds Number Comparison for Different Fluids')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('reynolds_number.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 실용적 예제 계산
    print("\n=== 실용적 레이놀즈 수 계산 예제 ===")
    print(f"관 직경 D = {D*100} cm")

    test_velocities = [0.1, 0.5, 1.0]
    for U in test_velocities:
        Re = water.rho * U * D / water.mu
        regime = "층류" if Re < 2300 else ("천이" if Re < 4000 else "난류")
        print(f"  U = {U} m/s -> Re = {Re:.0f} ({regime})")

# reynolds_number_analysis()
```

---

## 2. 지배방정식

### 2.1 연속 방정식 (Continuity Equation)

```
질량 보존 법칙:
∂ρ/∂t + ∇·(ρu) = 0

텐서 표기:
∂ρ/∂t + ∂(ρuᵢ)/∂xᵢ = 0

비압축성 유동 (ρ = const):
∇·u = 0
또는: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
```

```python
def visualize_continuity():
    """연속 방정식 시각화 - 검사체적 접근"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 검사체적 개념도
    ax1 = axes[0]

    # 검사체적 (사각형)
    cv = plt.Rectangle((0.3, 0.3), 0.4, 0.4, fill=False,
                       edgecolor='black', linewidth=2)
    ax1.add_patch(cv)

    # 질량 유입/유출 화살표
    # x 방향
    ax1.annotate('', xy=(0.3, 0.5), xytext=(0.1, 0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.text(0.15, 0.55, r'$\rho u A$', fontsize=12, color='blue')

    ax1.annotate('', xy=(0.9, 0.5), xytext=(0.7, 0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.text(0.75, 0.55, r'$\rho u A + \frac{\partial(\rho u)}{\partial x}\Delta x A$', fontsize=10, color='blue')

    # y 방향
    ax1.annotate('', xy=(0.5, 0.3), xytext=(0.5, 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(0.55, 0.15, r'$\rho v A$', fontsize=12, color='red')

    ax1.annotate('', xy=(0.5, 0.9), xytext=(0.5, 0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(0.55, 0.8, r'$\rho v A + ...$', fontsize=10, color='red')

    ax1.text(0.5, 0.5, r'$\Delta V$', fontsize=14, ha='center', va='center')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_title('검사체적과 질량 플럭스')
    ax1.axis('off')

    # 비압축성 유동장 예시 (div u = 0)
    ax2 = axes[1]
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)

    # 유동장: u = y, v = -x (점 소스 회전 유동, div=0)
    U = Y
    V = -X

    # 발산 계산 (수치적)
    div = np.zeros_like(X)

    ax2.streamplot(X, Y, U, V, color='blue', density=1.5, linewidth=1)
    ax2.quiver(X[::2, ::2], Y[::2, ::2], U[::2, ::2], V[::2, ::2],
              color='red', alpha=0.7)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(r'비압축성 유동장 예시: $u=y, v=-x$ ($\nabla \cdot \mathbf{u} = 0$)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('continuity_equation.png', dpi=150, bbox_inches='tight')
    plt.show()

# visualize_continuity()
```

### 2.2 운동량 방정식 (Momentum Equation)

```
Newton의 제2법칙 적용:
ρ(Du/Dt) = -∇p + μ∇²u + ρg + f

여기서:
- Du/Dt = ∂u/∂t + (u·∇)u : 물질 미분
- ∇p: 압력 구배
- μ∇²u: 점성력
- ρg: 중력
- f: 외부 체적력

성분별 (비압축성, 2D):
x: ρ(∂u/∂t + u∂u/∂x + v∂u/∂y) = -∂p/∂x + μ(∂²u/∂x² + ∂²u/∂y²)
y: ρ(∂v/∂t + u∂v/∂x + v∂v/∂y) = -∂p/∂y + μ(∂²v/∂x² + ∂²v/∂y²)
```

### 2.3 Navier-Stokes 방정식 유도

```python
def navier_stokes_derivation():
    """Navier-Stokes 방정식의 각 항 의미 시각화"""

    print("=" * 60)
    print("Navier-Stokes 방정식 유도")
    print("=" * 60)

    derivation = """
    1. 질량 보존 (연속방정식):
       ∂ρ/∂t + ∇·(ρu) = 0

       비압축성: ∇·u = 0

    2. 운동량 보존 (Newton 제2법칙):

       물질 미분 (Lagrangian derivative):
       Du/Dt = ∂u/∂t + (u·∇)u
                ↑         ↑
           국소가속도  대류가속도

       힘의 균형:
       ρ(Du/Dt) = ∑F = -∇p + ∇·τ + ρg
                   ↑      ↑     ↑    ↑
                관성력  압력력 점성력 체적력

       Newton 유체 가정 (τ = μ(∇u + ∇uᵀ)):
       ρ(Du/Dt) = -∇p + μ∇²u + ρg

    3. 비압축성 Navier-Stokes 방정식:

       ∂u/∂t + (u·∇)u = -1/ρ ∇p + ν∇²u + g

       여기서 ν = μ/ρ (운동점성계수)

    4. 무차원화 (Dimensionless form):

       특성 스케일: L(길이), U(속도), T=L/U(시간), P=ρU²(압력)

       ∂u*/∂t* + (u*·∇*)u* = -∇*p* + (1/Re)∇*²u*

       여기서 Re = UL/ν (레이놀즈 수)
    """
    print(derivation)

    # 각 항의 상대적 크기 시각화
    fig, ax = plt.subplots(figsize=(12, 6))

    Re_range = np.logspace(0, 6, 100)

    # 무차원 크기 비교
    inertia = np.ones_like(Re_range)  # O(1)
    viscous = 1 / Re_range            # O(1/Re)
    pressure = np.ones_like(Re_range) # O(1)

    ax.loglog(Re_range, inertia, 'b-', linewidth=2, label='관성항 O(1)')
    ax.loglog(Re_range, viscous, 'r-', linewidth=2, label='점성항 O(1/Re)')
    ax.loglog(Re_range, pressure, 'g--', linewidth=2, label='압력항 O(1)')

    ax.axvline(x=2300, color='gray', linestyle=':', label='층류-난류 천이')
    ax.fill_between(Re_range, 1e-6, 1, where=Re_range < 2300,
                   alpha=0.2, color='green', label='점성 지배')
    ax.fill_between(Re_range, 1e-6, 1, where=Re_range >= 2300,
                   alpha=0.2, color='blue', label='관성 지배')

    ax.set_xlabel('Reynolds Number Re')
    ax.set_ylabel('Relative Magnitude')
    ax.set_title('Navier-Stokes 방정식 각 항의 상대적 크기')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-6, 10)

    plt.tight_layout()
    plt.savefig('navier_stokes_terms.png', dpi=150, bbox_inches='tight')
    plt.show()

# navier_stokes_derivation()
```

---

## 3. 압축성 vs 비압축성 유동

### 3.1 마하 수와 압축성

```
마하 수 정의:
Ma = U/a

여기서:
- U: 유체 속도
- a: 음속 (이상기체: a = √(γRT))

분류:
- Ma < 0.3: 비압축성으로 취급 가능 (밀도 변화 < 5%)
- 0.3 < Ma < 0.8: 아음속 (Subsonic)
- 0.8 < Ma < 1.2: 천음속 (Transonic)
- 1.2 < Ma < 5: 초음속 (Supersonic)
- Ma > 5: 극초음속 (Hypersonic)
```

```python
def compressibility_analysis():
    """압축성 효과 분석"""

    # 등엔트로피 밀도 비
    def density_ratio(Ma, gamma=1.4):
        """ρ/ρ₀ = (1 + (γ-1)/2 Ma²)^(-1/(γ-1))"""
        return (1 + (gamma - 1) / 2 * Ma**2) ** (-1 / (gamma - 1))

    # 압력 비
    def pressure_ratio(Ma, gamma=1.4):
        """p/p₀ = (1 + (γ-1)/2 Ma²)^(-γ/(γ-1))"""
        return (1 + (gamma - 1) / 2 * Ma**2) ** (-gamma / (gamma - 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    Ma = np.linspace(0, 3, 200)

    # 밀도 비
    ax1 = axes[0]
    rho_ratio = density_ratio(Ma)
    ax1.plot(Ma, rho_ratio, 'b-', linewidth=2)
    ax1.axvline(x=0.3, color='green', linestyle='--', label='Ma=0.3 (비압축성 한계)')
    ax1.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5)
    ax1.fill_between(Ma, 0, rho_ratio, where=Ma < 0.3, alpha=0.3, color='green',
                    label='비압축성 영역')

    ax1.set_xlabel('Mach Number Ma')
    ax1.set_ylabel(r'Density Ratio $\rho/\rho_0$')
    ax1.set_title('등엔트로피 유동에서의 밀도 변화')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 1.1)

    # 압축성/비압축성 방정식 비교
    ax2 = axes[1]

    equations = {
        '비압축성\n(Incompressible)': [
            r'$\nabla \cdot \mathbf{u} = 0$',
            r'$\rho \frac{D\mathbf{u}}{Dt} = -\nabla p + \mu \nabla^2 \mathbf{u}$',
            '3개 방정식, 4개 미지수 (u,v,w,p)',
            '압력: Poisson 방정식으로 결정'
        ],
        '압축성\n(Compressible)': [
            r'$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0$',
            r'$\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) = -\nabla p + \nabla \cdot \tau$',
            r'$\frac{\partial E}{\partial t} + \nabla \cdot ((E+p)\mathbf{u}) = \nabla \cdot (k\nabla T + \tau \cdot \mathbf{u})$',
            '5개 방정식, 5개 미지수 (ρ,u,v,w,E) + 상태방정식'
        ]
    }

    ax2.text(0.25, 0.95, '비압축성 (Ma < 0.3)', fontsize=14, fontweight='bold',
            ha='center', transform=ax2.transAxes)
    ax2.text(0.75, 0.95, '압축성 (Ma > 0.3)', fontsize=14, fontweight='bold',
            ha='center', transform=ax2.transAxes)

    y_pos = 0.85
    for eq in equations['비압축성\n(Incompressible)']:
        ax2.text(0.25, y_pos, eq, fontsize=10, ha='center', transform=ax2.transAxes)
        y_pos -= 0.12

    y_pos = 0.85
    for eq in equations['압축성\n(Compressible)']:
        ax2.text(0.75, y_pos, eq, fontsize=10, ha='center', transform=ax2.transAxes)
        y_pos -= 0.12

    ax2.axvline(x=0.5, color='black', linestyle='-', linewidth=2,
               transform=ax2.transAxes)
    ax2.axis('off')
    ax2.set_title('지배방정식 비교')

    plt.tight_layout()
    plt.savefig('compressibility.png', dpi=150, bbox_inches='tight')
    plt.show()

# compressibility_analysis()
```

---

## 4. 경계층 이론

### 4.1 경계층 개념

```
경계층 (Boundary Layer):
- 벽면 근처에서 점성 효과가 지배적인 영역
- 벽면: no-slip 조건 (u = 0)
- 경계층 외부: 자유류 속도 U∞

경계층 두께 δ:
- 속도가 자유류의 99%가 되는 위치
- 층류 평판: δ ~ x/√Re_x (Blasius)
- 난류 평판: δ ~ x/Re_x^(1/5)
```

```python
def boundary_layer_theory():
    """경계층 이론 시각화"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Blasius 층류 경계층 프로파일
    ax1 = axes[0, 0]

    # Blasius 유사해 (근사)
    eta = np.linspace(0, 8, 100)  # 무차원 좌표 η = y√(U∞/νx)

    # f'(η) 근사 (실제는 Blasius 방정식 수치해)
    # 여기서는 간단한 근사 사용
    u_U = np.tanh(eta / 2.5) ** 1.5  # 근사

    ax1.plot(u_U, eta, 'b-', linewidth=2)
    ax1.axhline(y=5.0, color='red', linestyle='--', label=r'$\delta_{99}$ (η ≈ 5)')
    ax1.fill_betweenx(eta, 0, u_U, alpha=0.3)

    ax1.set_xlabel(r'$u/U_\infty$')
    ax1.set_ylabel(r'$\eta = y\sqrt{U_\infty/\nu x}$')
    ax1.set_title('Blasius 층류 경계층 속도 프로파일')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(0, 8)

    # (2) 경계층 두께 성장
    ax2 = axes[0, 1]

    nu = 1.5e-5  # 공기 운동점성계수
    U_inf = 10   # 자유류 속도 [m/s]
    x = np.linspace(0.01, 1, 100)  # 평판 위치 [m]

    Re_x = U_inf * x / nu

    # 층류 경계층 두께 (Blasius)
    delta_lam = 5.0 * x / np.sqrt(Re_x)

    # 난류 경계층 두께 (1/7승 법칙)
    delta_turb = 0.37 * x / Re_x ** 0.2

    # 천이 위치 (Re_x ~ 5×10^5)
    x_trans = 5e5 * nu / U_inf

    ax2.plot(x * 1000, delta_lam * 1000, 'b-', linewidth=2, label='층류')
    ax2.plot(x * 1000, delta_turb * 1000, 'r-', linewidth=2, label='난류')
    ax2.axvline(x=x_trans * 1000, color='green', linestyle='--', label=f'천이점 (x ≈ {x_trans*1000:.0f} mm)')

    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel(r'$\delta$ [mm]')
    ax2.set_title(f'경계층 두께 성장 (U∞ = {U_inf} m/s, 공기)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (3) 벽면 전단응력
    ax3 = axes[1, 0]

    # 층류 벽면 전단응력 계수
    Cf_lam = 0.664 / np.sqrt(Re_x)

    # 난류 벽면 전단응력 계수
    Cf_turb = 0.027 / Re_x ** (1/7)

    ax3.loglog(Re_x, Cf_lam, 'b-', linewidth=2, label='층류 (Blasius)')
    ax3.loglog(Re_x, Cf_turb, 'r-', linewidth=2, label='난류 (1/7승 법칙)')
    ax3.axvline(x=5e5, color='green', linestyle='--', alpha=0.5)

    ax3.set_xlabel(r'$Re_x$')
    ax3.set_ylabel(r'$C_f = \tau_w / (0.5\rho U_\infty^2)$')
    ax3.set_title('벽면 마찰 계수')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (4) 경계층 개념도
    ax4 = axes[1, 1]

    # 평판
    ax4.fill_between([0, 5], [-0.1, -0.1], [0, 0], color='gray', alpha=0.5)

    # 경계층
    x_plate = np.linspace(0, 5, 50)
    delta_vis = 0.5 * np.sqrt(x_plate)  # 단순화된 경계층
    ax4.fill_between(x_plate, 0, delta_vis, alpha=0.3, color='blue', label='경계층')
    ax4.plot(x_plate, delta_vis, 'b-', linewidth=2)

    # 속도 프로파일 화살표
    for x0 in [0.5, 1.5, 3.0, 4.5]:
        y_arrows = np.linspace(0, 0.5 * np.sqrt(x0) * 1.2, 6)
        for y in y_arrows:
            u = min(1, y / (0.5 * np.sqrt(x0))) if x0 > 0 else 0
            ax4.annotate('', xy=(x0 + u * 0.3, y), xytext=(x0, y),
                        arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    # 자유류
    ax4.annotate('', xy=(5, 1.5), xytext=(0, 1.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax4.text(2.5, 1.7, r'$U_\infty$', fontsize=14)

    ax4.text(2.5, 0.3, '경계층', fontsize=12, color='blue')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('평판 위 경계층 발달')
    ax4.set_xlim(-0.5, 5.5)
    ax4.set_ylim(-0.2, 2)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('boundary_layer.png', dpi=150, bbox_inches='tight')
    plt.show()

# boundary_layer_theory()
```

---

## 5. 간단한 CFD 예제: Poiseuille 유동

### 5.1 2D 채널 유동 (Poiseuille Flow)

```
문제 설정:
- 두 평행 평판 사이의 정상 층류 유동
- 압력 구배에 의해 구동
- 해석해 존재 (검증용)

지배방정식 (정상, 완전발달):
d²u/dy² = (1/μ)(dp/dx) = const

경계조건:
- y = 0: u = 0 (no-slip)
- y = H: u = 0 (no-slip)

해석해:
u(y) = -(1/2μ)(dp/dx)y(H-y)

최대 속도 (중심):
u_max = (H²/8μ)|dp/dx|
```

```python
def poiseuille_flow_exact():
    """Poiseuille 유동 해석해"""

    H = 1.0       # 채널 높이 [m]
    mu = 0.01     # 동점성계수 [Pa·s]
    dpdx = -1.0   # 압력 구배 [Pa/m] (음수 = 양의 x 방향 유동)

    y = np.linspace(0, H, 100)
    u_exact = -(1 / (2 * mu)) * dpdx * y * (H - y)

    u_max = H**2 / (8 * mu) * abs(dpdx)
    u_avg = 2 / 3 * u_max

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 속도 프로파일
    ax1 = axes[0]
    ax1.plot(u_exact, y, 'b-', linewidth=2, label='해석해')
    ax1.axhline(y=H/2, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=u_max, color='green', linestyle='--', alpha=0.5, label=f'u_max = {u_max:.2f}')
    ax1.axvline(x=u_avg, color='orange', linestyle='--', alpha=0.5, label=f'u_avg = {u_avg:.2f}')

    ax1.set_xlabel('u [m/s]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Poiseuille 유동 속도 프로파일')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 전단응력 프로파일
    ax2 = axes[1]
    tau = mu * np.gradient(u_exact, y)
    ax2.plot(tau, y, 'r-', linewidth=2)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    ax2.set_xlabel(r'$\tau$ [Pa]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('전단응력 분포')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('poiseuille_exact.png', dpi=150, bbox_inches='tight')
    plt.show()

    return u_exact, y

# u_exact, y = poiseuille_flow_exact()
```

### 5.2 유한차분법을 이용한 CFD 구현

```python
def cfd_channel_flow():
    """
    2D 채널 유동 CFD 시뮬레이션
    - 비정상 Navier-Stokes 방정식
    - 정상 상태까지 시간 전진
    """

    # 격자 설정
    Nx = 50       # x 방향 격자 수
    Ny = 30       # y 방향 격자 수
    Lx = 2.0      # 채널 길이 [m]
    Ly = 1.0      # 채널 높이 [m]

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # 물성치
    rho = 1.0     # 밀도 [kg/m³]
    mu = 0.01     # 동점성계수 [Pa·s]
    nu = mu / rho

    # 압력 구배 (체적력으로 모델링)
    dpdx = -1.0   # [Pa/m]

    # 시간 설정
    dt = 0.001
    n_steps = 2000

    # 초기화
    u = np.zeros((Ny, Nx))  # x-속도
    v = np.zeros((Ny, Nx))  # y-속도
    p = np.zeros((Ny, Nx))  # 압력

    # CFL 조건 확인
    u_max_expected = Ly**2 / (8 * mu) * abs(dpdx)
    CFL = u_max_expected * dt / dx
    print(f"예상 최대 속도: {u_max_expected:.4f} m/s")
    print(f"CFL 수: {CFL:.4f}")

    # 해석해 (검증용)
    u_exact = -(1 / (2 * mu)) * dpdx * y * (Ly - y)

    def apply_boundary_conditions(u, v):
        """경계조건 적용"""
        # 벽면 (no-slip)
        u[0, :] = 0    # 하단 벽
        u[-1, :] = 0   # 상단 벽
        v[0, :] = 0
        v[-1, :] = 0

        # 입구/출구 (주기적 또는 Neumann)
        u[:, 0] = u[:, 1]    # 입구
        u[:, -1] = u[:, -2]  # 출구
        v[:, 0] = 0
        v[:, -1] = v[:, -2]

        return u, v

    def compute_rhs(u, v, p, nu, dx, dy, dpdx, rho):
        """우변 계산 (운동량 방정식)"""
        Ny, Nx = u.shape
        rhs_u = np.zeros_like(u)
        rhs_v = np.zeros_like(v)

        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                # 대류항 (중심차분)
                duudx = (u[i, j+1]**2 - u[i, j-1]**2) / (2 * dx)
                duvdy = (u[i+1, j] * v[i+1, j] - u[i-1, j] * v[i-1, j]) / (2 * dy)

                dvudx = (v[i, j+1] * u[i, j+1] - v[i, j-1] * u[i, j-1]) / (2 * dx)
                dvvdy = (v[i+1, j]**2 - v[i-1, j]**2) / (2 * dy)

                # 확산항 (중심차분)
                d2udx2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dx**2
                d2udy2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dy**2

                d2vdx2 = (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dx**2
                d2vdy2 = (v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dy**2

                # 압력항
                dpdx_local = (p[i, j+1] - p[i, j-1]) / (2 * dx) if j > 0 and j < Nx-1 else 0
                dpdy_local = (p[i+1, j] - p[i-1, j]) / (2 * dy) if i > 0 and i < Ny-1 else 0

                # 운동량 방정식 우변
                rhs_u[i, j] = -duudx - duvdy - dpdx_local/rho + nu * (d2udx2 + d2udy2) - dpdx/rho
                rhs_v[i, j] = -dvudx - dvvdy - dpdy_local/rho + nu * (d2vdx2 + d2vdy2)

        return rhs_u, rhs_v

    # 시간 전진
    history = []

    for n in range(n_steps):
        # 경계조건
        u, v = apply_boundary_conditions(u, v)

        # RHS 계산
        rhs_u, rhs_v = compute_rhs(u, v, p, nu, dx, dy, dpdx, rho)

        # 시간 전진 (Euler)
        u = u + dt * rhs_u
        v = v + dt * rhs_v

        # 수렴 체크
        if n % 200 == 0:
            u_center = u[:, Nx//2]
            error = np.max(np.abs(u_center - u_exact))
            history.append((n, error, np.max(u)))
            print(f"Step {n}: max error = {error:.6f}, max u = {np.max(u):.6f}")

    # 결과 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 속도장 (벡터)
    ax1 = axes[0, 0]
    skip = 2
    ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u[::skip, ::skip], v[::skip, ::skip],
              color='blue', scale=30)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('속도 벡터장')
    ax1.set_aspect('equal')

    # (2) u-속도 등고선
    ax2 = axes[0, 1]
    cf = ax2.contourf(X, Y, u, levels=20, cmap='jet')
    plt.colorbar(cf, ax=ax2, label='u [m/s]')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_title('u-속도 분포')
    ax2.set_aspect('equal')

    # (3) 해석해와 비교
    ax3 = axes[1, 0]
    u_center = u[:, Nx//2]
    ax3.plot(u_center, y, 'bo-', markersize=4, label='CFD')
    ax3.plot(u_exact, y, 'r-', linewidth=2, label='해석해')
    ax3.set_xlabel('u [m/s]')
    ax3.set_ylabel('y [m]')
    ax3.set_title('속도 프로파일 비교 (x = L/2)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (4) 수렴 이력
    ax4 = axes[1, 1]
    steps, errors, u_maxs = zip(*history)
    ax4.semilogy(steps, errors, 'b-o', label='Error')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Max Error')
    ax4.set_title('수렴 이력')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cfd_channel_flow.png', dpi=150, bbox_inches='tight')
    plt.show()

    return u, v, p, X, Y

# u, v, p, X, Y = cfd_channel_flow()
```

---

## 6. CFD의 주요 과제

### 6.1 격자 생성 (Mesh Generation)

```python
def mesh_types_visualization():
    """CFD 격자 유형 시각화"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (1) 구조 격자 (Structured)
    ax1 = axes[0, 0]
    nx, ny = 10, 8
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)

    for i in range(ny):
        ax1.plot(x, np.full_like(x, y[i]), 'b-', linewidth=0.5)
    for j in range(nx):
        ax1.plot(np.full_like(y, x[j]), y, 'b-', linewidth=0.5)

    X, Y = np.meshgrid(x, y)
    ax1.plot(X, Y, 'ko', markersize=3)
    ax1.set_title('구조 격자 (Structured)')
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.1, 2.1)
    ax1.set_ylim(-0.1, 1.1)

    # (2) 비구조 격자 (Unstructured)
    ax2 = axes[0, 1]
    from scipy.spatial import Delaunay

    # 랜덤 포인트
    np.random.seed(42)
    points = np.random.rand(30, 2)
    points[:, 0] *= 2

    # 경계 포인트 추가
    boundary = np.array([[0, 0], [2, 0], [2, 1], [0, 1]])
    for i in range(4):
        edge = np.linspace(boundary[i], boundary[(i+1)%4], 8)[1:-1]
        points = np.vstack([points, edge])

    tri = Delaunay(points)
    ax2.triplot(points[:, 0], points[:, 1], tri.simplices, 'b-', linewidth=0.5)
    ax2.plot(points[:, 0], points[:, 1], 'ko', markersize=3)
    ax2.set_title('비구조 격자 (Unstructured)')
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.1, 2.1)
    ax2.set_ylim(-0.1, 1.1)

    # (3) O-격자 (원통 주위)
    ax3 = axes[1, 0]
    r_inner = 0.3
    r_outer = 1.0
    n_r = 8
    n_theta = 24

    r = np.linspace(r_inner, r_outer, n_r)
    theta = np.linspace(0, 2*np.pi, n_theta)

    for ri in r:
        x_circle = ri * np.cos(theta)
        y_circle = ri * np.sin(theta)
        ax3.plot(x_circle, y_circle, 'b-', linewidth=0.5)

    for ti in theta[:-1]:
        x_radial = r * np.cos(ti)
        y_radial = r * np.sin(ti)
        ax3.plot(x_radial, y_radial, 'b-', linewidth=0.5)

    R, Theta = np.meshgrid(r, theta)
    X_o = R * np.cos(Theta)
    Y_o = R * np.sin(Theta)
    ax3.plot(X_o, Y_o, 'ko', markersize=2)

    ax3.set_title('O-격자 (원통 주위)')
    ax3.set_aspect('equal')
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)

    # (4) 경계층 프리즘 격자
    ax4 = axes[1, 1]

    # 경계층 영역 (프리즘)
    x_wall = np.linspace(0, 2, 20)
    y_layers = [0, 0.02, 0.05, 0.1, 0.2, 0.4]

    for yl in y_layers:
        ax4.plot(x_wall, np.full_like(x_wall, yl), 'b-', linewidth=0.5)

    # 외부 비구조 격자 (삼각형)
    np.random.seed(123)
    outer_points = np.random.rand(20, 2)
    outer_points[:, 0] *= 2
    outer_points[:, 1] = outer_points[:, 1] * 0.5 + 0.4

    tri_outer = Delaunay(outer_points)
    ax4.triplot(outer_points[:, 0], outer_points[:, 1], tri_outer.simplices,
               'g-', linewidth=0.5)

    ax4.fill_between(x_wall, 0, y_layers[-1], alpha=0.2, color='blue',
                    label='경계층 (프리즘)')
    ax4.set_title('하이브리드 격자 (프리즘 + 삼각형)')
    ax4.set_aspect('equal')
    ax4.set_xlim(-0.1, 2.1)
    ax4.set_ylim(-0.05, 1)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('mesh_types.png', dpi=150, bbox_inches='tight')
    plt.show()

# mesh_types_visualization()
```

### 6.2 난류 모델링

```
주요 난류 모델:

1. RANS (Reynolds-Averaged Navier-Stokes):
   - k-ε 모델: 범용, 산업 표준
   - k-ω 모델: 벽면 근처 정확
   - SST 모델: k-ε + k-ω 장점 결합

2. LES (Large Eddy Simulation):
   - 큰 와류는 직접 계산
   - 작은 와류는 모델링 (SGS 모델)

3. DNS (Direct Numerical Simulation):
   - 모든 난류 스케일 직접 계산
   - Re³에 비례하는 계산 비용

선택 기준:
- 정확도: DNS > LES > RANS
- 계산 비용: DNS > LES > RANS
- 실용성: RANS > LES > DNS
```

---

## 7. 연습 문제

### 연습 1: 레이놀즈 수(Reynolds Number) 계산
직경 5cm 관에서 물(20°C)이 평균 속도 2m/s로 흐를 때 레이놀즈 수를 계산하고 유동 상태를 판별하시오.

<details><summary>정답 보기</summary>

물(20°C)의 물성치: ρ = 998 kg/m³, μ = 1.002×10⁻³ Pa·s

```python
rho = 998      # kg/m^3
mu  = 1.002e-3 # Pa·s
U   = 2.0      # m/s
D   = 0.05     # m (5 cm)

Re = rho * U * D / mu
print(f"Reynolds number Re = {Re:.0f}")

if Re < 2300:
    regime = "층류(Laminar)"
elif Re < 4000:
    regime = "천이(Transition)"
else:
    regime = "난류(Turbulent)"
print(f"유동 상태: {regime}")
```

Re = 998 × 2.0 × 0.05 / (1.002×10⁻³) ≈ **99,601**

Re ≫ 4000이므로 **난류(Turbulent)** 유동입니다. 실제 파이프 엔지니어링에서는 이 레이놀즈 수 범위에서 Moody 선도를 이용하여 마찰 계수를 구하고 압력 손실을 계산합니다.
</details>

### 연습 2: Poiseuille 유동(Poiseuille Flow)
Poiseuille 유동에서 평균 속도와 최대 속도의 관계를 유도하시오.

<details><summary>정답 보기</summary>

Poiseuille 유동의 속도 프로파일: u(y) = -(1/2μ)(dp/dx) y(H-y)

최대 속도(중심선, y = H/2):
```
u_max = -(1/2μ)(dp/dx)(H/2)(H - H/2) = -(1/8μ)(dp/dx) H²
```

평균 속도(단면 적분):
```
u_avg = (1/H) ∫₀ᴴ u(y) dy
      = (1/H) · -(1/2μ)(dp/dx) · [Hy²/2 - y³/3]₀ᴴ
      = (1/H) · -(1/2μ)(dp/dx) · (H³/2 - H³/3)
      = -(1/2μ)(dp/dx) · H²/6
      = (2/3) u_max
```

따라서: **u_avg = (2/3) u_max**

```python
H = 1.0; mu = 0.01; dpdx = -1.0
y = np.linspace(0, H, 100)
u = -(1/(2*mu)) * dpdx * y * (H - y)
u_max = np.max(u)
u_avg = np.mean(u)
print(f"u_max = {u_max:.4f}, u_avg = {u_avg:.4f}")
print(f"u_avg / u_max = {u_avg/u_max:.4f} (expected 2/3 = {2/3:.4f})")
```

이 관계는 층류 채널 유동의 핵심 결과로, 유량 측정에 활용됩니다.
</details>

### 연습 3: 경계층 두께(Boundary Layer Thickness)
공기(20°C)가 5m/s로 평판 위를 흐를 때, 선단에서 10cm 떨어진 위치의 층류 경계층 두께를 계산하시오.

<details><summary>정답 보기</summary>

공기(20°C) 물성치: ν = μ/ρ = 1.825×10⁻⁵/1.204 ≈ 1.516×10⁻⁵ m²/s

블라지우스(Blasius) 층류 경계층 공식: δ = 5.0 x / √(Re_x)

```python
nu    = 1.516e-5  # m^2/s (공기 동점성계수)
U_inf = 5.0       # m/s
x     = 0.10      # m (10 cm)

Re_x  = U_inf * x / nu
delta = 5.0 * x / np.sqrt(Re_x)

print(f"Re_x = {Re_x:.0f}")
print(f"경계층 두께 δ = {delta*1000:.2f} mm")

# 유동 상태 확인 (층류 기준: Re_x < 5×10^5)
if Re_x < 5e5:
    print("층류(Laminar) 경계층 확인")
```

Re_x = 5.0 × 0.10 / (1.516×10⁻⁵) ≈ 33,003 (층류)

δ = 5.0 × 0.10 / √33003 ≈ **2.75 mm**

경계층 두께는 x에 비례하고 Re의 제곱근에 반비례합니다. 속도나 거리가 증가할수록 경계층은 상대적으로 얇아집니다.
</details>

### 연습 4: CFD 격자 의존성(Grid Dependence) 검사
채널 유동 CFD 코드를 수정하여 격자 수(Ny = 10, 20, 30)를 변화시키면서 격자 수렴 테스트를 수행하시오. 각 격자에서 계산된 최대 속도를 Poiseuille 해석해와 비교하시오.

<details><summary>정답 보기</summary>

```python
def grid_convergence_test():
    """격자 수렴 테스트: Poiseuille 유동"""
    H = 1.0; mu = 0.01; dpdx = -1.0
    u_max_exact = H**2 / (8*mu) * abs(dpdx)
    print(f"해석해 u_max = {u_max_exact:.4f} m/s\n")

    for Ny in [10, 20, 30, 50]:
        dy = H / (Ny - 1)
        y = np.linspace(0, H, Ny)

        # 1D 확산 방정식: d²u/dy² = (1/mu)(dp/dx)
        # 유한 차분: (u[i+1] - 2u[i] + u[i-1])/dy² = (1/mu)(dp/dx)
        # 내부 절점만 (0과 H에서 경계 조건 u=0)
        n_int = Ny - 2
        A_fd = np.zeros((n_int, n_int))
        b_fd = np.ones(n_int) * (dpdx / mu)

        for i in range(n_int):
            A_fd[i, i] = -2 / dy**2
            if i > 0:       A_fd[i, i-1] = 1 / dy**2
            if i < n_int-1: A_fd[i, i+1] = 1 / dy**2

        u_int = np.linalg.solve(A_fd, b_fd)
        u_full = np.zeros(Ny)
        u_full[1:-1] = u_int
        u_max_num = np.max(u_full)
        error = abs(u_max_num - u_max_exact) / u_max_exact * 100

        print(f"Ny={Ny:3d}: u_max = {u_max_num:.4f}, 오차 = {error:.2f}%")

grid_convergence_test()
```

격자를 세밀하게 할수록 수치해가 해석해에 수렴합니다. 유한 차분법(Finite Difference Method)은 격자 간격 dy에 비례하는 O(dy²) 정확도를 가지므로 격자 수를 2배 늘리면 오차가 약 4배 줄어듭니다.
</details>

---

## 8. 참고자료

### 핵심 교재
- Versteeg & Malalasekera, "An Introduction to Computational Fluid Dynamics"
- Anderson, "Computational Fluid Dynamics: The Basics with Applications"
- Ferziger & Peric, "Computational Methods for Fluid Dynamics"

### CFD 소프트웨어
- OpenFOAM (오픈소스)
- ANSYS Fluent (상용)
- COMSOL Multiphysics (상용)
- SU2 (오픈소스)

### 온라인 자료
- NASA CFD Resources
- CFD Online (포럼, 튜토리얼)
- LearnCAx (무료 강좌)

---

## 요약

```
CFD 기초 핵심:

1. 지배방정식:
   - 연속: ∇·u = 0 (비압축성)
   - 운동량: ρ(Du/Dt) = -∇p + μ∇²u
   - 에너지: (압축성 유동)

2. 무차원 수:
   - Re = ρUL/μ (관성/점성)
   - Ma = U/a (압축성)
   - Pr = ν/α (운동량/열 확산)

3. CFD 절차:
   ① 격자 생성 (전처리)
   ② 이산화 및 계산 (솔버)
   ③ 후처리 (시각화, 분석)

4. 주요 고려사항:
   - 격자 품질 및 수렴
   - 경계조건 설정
   - 난류 모델 선택
   - 수치 안정성 (CFL 조건)
```

---

## 연습 문제

### 연습 1: 레이놀즈 수(Reynolds Number)와 유동 형태
20°C의 물(밀도 998 kg/m³, 동점성계수 1.002×10⁻³ Pa·s)이 지름 5 cm 파이프를 통해 평균 속도 2 m/s로 흐릅니다. 레이놀즈 수(Reynolds number)를 계산하고 유동 형태(층류 또는 난류)를 판별하세요. 그런 다음 동일한 파이프에서 난류(turbulent flow)를 달성하는 데 필요한 최소 속도를 구하세요.

### 연습 2: 나비에-스토크스(Navier-Stokes) 항 분석
2차원 비압축성 유동(incompressible flow)에서 속도 벡터장이 u = sin(x)cos(y), v = -cos(x)sin(y)로 주어질 때, 연속 방정식(continuity equation)이 만족됨을 검증하세요. 그런 다음 와도(vorticity) ω = ∂v/∂x - ∂u/∂y를 계산하고, 이것이 물리적으로 무엇을 의미하는지 설명하세요.

### 연습 3: Poiseuille 유동 비교
채널 높이 H = 1 m, 동점성계수 μ = 0.01 Pa·s, 압력 구배(pressure gradient) dp/dx = -1 Pa/m 조건에서 Poiseuille 유동(Poiseuille flow)의 해석해와 유한 차분(finite difference) CFD 수치해를 모두 구현하세요. 격자 크기 Ny = 10, 20, 40을 사용하여 수치해와 해석해 사이의 L2 오차를 격자 간격의 함수로 그래프로 나타내세요. 관찰되는 수렴 차수는 얼마인가요?

### 연습 4: 경계층(Boundary Layer) 스케일링
공기(20°C, 동점성계수 ν = 1.5×10⁻⁵ m²/s)가 평판 위를 U∞ = 10 m/s로 흐를 때, 0 < x < 0.5 m 범위에서 다음을 계산하고 그래프로 나타내세요: (a) Blasius 공식 δ = 5x/√(Re_x)를 사용한 층류 경계층 두께(laminar boundary layer thickness) δ(x), (b) 국소 마찰 계수(local friction coefficient) Cf(x) = 0.664/√(Re_x), (c) Re_x = 5×10⁵인 천이 위치(transition location). 천이점이 x = 0.1 m로 이동하려면 유속이 얼마여야 하나요?

### 연습 5: CFD 격자 민감도(Mesh Sensitivity) 분석
채널 유동 CFD 코드를 수정하여 체계적인 격자 수렴성(grid convergence) 검토를 수행하세요. Nx×Ny = 25×15, 50×30, 100×60 격자로 시뮬레이션을 실행하세요. 각 격자에 대해 중심선 속도 프로파일을 해석적 Poiseuille 해와 비교하고 최대 절대 오차를 계산하세요. 격자 간격이 절반으로 줄 때 오차가 절반이 되나요(1차 수렴), 아니면 4분의 1로 줄어드나요(2차 수렴)?

---

다음 레슨에서는 비압축성 유동의 대표적 문제인 Lid-Driven Cavity와 SIMPLE 알고리즘을 다룹니다.
