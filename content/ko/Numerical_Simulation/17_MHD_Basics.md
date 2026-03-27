# 17. MHD 기초 이론 (Magnetohydrodynamics Basics)

## 학습 목표
- 자기유체역학(MHD)의 기본 개념 이해
- MHD 가정과 적용 범위 파악
- 이상 MHD 방정식 유도
- Alfven 속도와 MHD 파동 이해
- 자기압과 자기장력 개념 학습

**이 레슨이 중요한 이유:** 자기유체역학(Magnetohydrodynamics, MHD)은 우주에서 가장 에너지가 큰 현상들을 지배합니다: 태양 플레어(solar flare), 천체물리적 제트(astrophysical jet), 항성 다이나모(stellar dynamo), 그리고 핵융합 플라즈마가 그 예입니다. 유체역학과 Maxwell 방정식을 결합함으로써, MHD는 전도성 유체와 자기장이 어떻게 상호작용하는지를 포착합니다 -- 자기장은 유체에 힘을 가하고, 유체의 운동은 자기장을 늘이고 증폭시킵니다. MHD의 이해는 우주 날씨 예측, 제어 핵융합, 천체물리 모델링에 필수적입니다.

---

## 1. MHD 소개

### 1.1 정의와 응용

```
자기유체역학 (Magnetohydrodynamics, MHD):
- 전기 전도성 유체와 전자기장의 상호작용
- 유체역학 + 전자기학의 결합

응용 분야:
1. 천체물리: 태양, 항성, 은하, 성간 매질
2. 핵융합: 토카막, 스텔러레이터 플라즈마 가둠
3. 지구물리: 지구 자기장 다이나모
4. 공학: MHD 발전, 전자기 펌프, 금속 주조
5. 우주물리: 태양풍, 자기권, 우주 날씨

역사:
- Alfvén (1942): MHD 파동 발견 → 1970 노벨 물리학상
```

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 물리 상수
mu0 = 4 * np.pi * 1e-7  # 진공 투자율 [H/m]
eps0 = 8.854e-12        # 진공 유전율 [F/m]
c = 299792458           # 광속 [m/s]
kB = 1.381e-23          # 볼츠만 상수 [J/K]
me = 9.109e-31          # 전자 질량 [kg]
mp = 1.673e-27          # 양성자 질량 [kg]
e = 1.602e-19           # 기본 전하 [C]

def mhd_introduction():
    """MHD 개요"""

    print("=" * 60)
    print("자기유체역학 (MHD) 개요")
    print("=" * 60)

    intro = """
    MHD의 핵심 개념:

    1. 전도성 유체:
       - 플라즈마, 액체 금속, 염수
       - 자유 전하가 전자기장에 반응
       - 전기 전도도 σ > 0

    2. 유체-자기장 상호작용:
       - 자기장이 유체 운동에 힘을 가함 (Lorentz 힘)
       - 유체 운동이 자기장을 변화시킴 (유도)

    3. 결합 방정식:
       - 유체역학: 연속, 운동량, 에너지
       - 전자기학: Maxwell 방정식 (일부)

    ┌─────────────────────────────────────────────────┐
    │                  MHD 영역                        │
    │                                                  │
    │    [유체역학]  ←──── 결합 ────→  [전자기학]      │
    │                                                  │
    │    ρ, v, p         J = σ(E + v×B)      E, B     │
    │    연속/운동량        Ohm's law         Maxwell  │
    │    에너지                              (일부)    │
    └─────────────────────────────────────────────────┘
    """
    print(intro)

mhd_introduction()
```

### 1.2 MHD 시간/공간 스케일

```python
def mhd_scales():
    """MHD 관련 시간/공간 스케일 비교"""

    # 태양 코로나 플라즈마 예시
    n = 1e15       # 밀도 [m^-3]
    T = 1e6        # 온도 [K]
    B = 1e-2       # 자기장 [T]
    L = 1e8        # 특성 길이 [m]

    # 플라즈마 주파수
    omega_pe = np.sqrt(n * e**2 / (eps0 * me))
    omega_pi = np.sqrt(n * e**2 / (eps0 * mp))

    # 사이클로트론 주파수
    omega_ce = e * B / me
    omega_ci = e * B / mp

    # Debye 길이
    lambda_D = np.sqrt(eps0 * kB * T / (n * e**2))

    # 열속도
    v_te = np.sqrt(2 * kB * T / me)
    v_ti = np.sqrt(2 * kB * T / mp)

    # Alfven 속도
    rho = n * mp
    v_A = B / np.sqrt(mu0 * rho)

    # 음속
    gamma = 5/3
    p = n * kB * T
    c_s = np.sqrt(gamma * p / rho)

    print("=" * 60)
    print("플라즈마 스케일 (태양 코로나 예시)")
    print("=" * 60)
    print(f"\n입력 파라미터:")
    print(f"  밀도 n = {n:.2e} m^-3")
    print(f"  온도 T = {T:.2e} K")
    print(f"  자기장 B = {B*1000:.1f} mT")
    print(f"  특성 길이 L = {L/1e6:.0f} Mm")

    print(f"\n주파수:")
    print(f"  전자 플라즈마 주파수 ωpe = {omega_pe:.2e} rad/s")
    print(f"  이온 플라즈마 주파수 ωpi = {omega_pi:.2e} rad/s")
    print(f"  전자 사이클로트론 ωce = {omega_ce:.2e} rad/s")
    print(f"  이온 사이클로트론 ωci = {omega_ci:.2e} rad/s")

    print(f"\n속도:")
    print(f"  전자 열속도 vte = {v_te/1e6:.2f} Mm/s")
    print(f"  이온 열속도 vti = {v_ti/1e3:.2f} km/s")
    print(f"  Alfven 속도 vA = {v_A/1e3:.2f} km/s")
    print(f"  음속 cs = {c_s/1e3:.2f} km/s")

    print(f"\n길이 스케일:")
    print(f"  Debye 길이 λD = {lambda_D:.4f} m")
    print(f"  전자 관성 길이 c/ωpe = {c/omega_pe:.4f} m")
    print(f"  이온 관성 길이 c/ωpi = {c/omega_pi:.2f} m")

    # MHD 유효성 조건 확인
    print(f"\nMHD 유효성 조건:")
    print(f"  L >> λD: {L:.2e} >> {lambda_D:.4f} ✓" if L > 1000*lambda_D else f"  L >> λD: {L} !>> {lambda_D}")
    print(f"  L >> c/ωpi: {L:.2e} >> {c/omega_pi:.2f} ✓" if L > 100*c/omega_pi else f"  L >> c/ωpi 확인 필요")

    # 시각화
    fig, ax = plt.subplots(figsize=(12, 6))

    scales = {
        'λD': lambda_D,
        'c/ωpe': c/omega_pe,
        'c/ωpi': c/omega_pi,
        'vA/ωci': v_A/omega_ci,
        'L (MHD)': L
    }

    y_pos = np.arange(len(scales))
    values = list(scales.values())
    labels = list(scales.keys())

    ax.barh(y_pos, np.log10(values), color=['red', 'orange', 'yellow', 'green', 'blue'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('log₁₀(Length [m])')
    ax.set_title('플라즈마 길이 스케일 비교')
    ax.axvline(x=np.log10(L), color='black', linestyle='--', label='MHD 스케일')
    ax.grid(True, alpha=0.3)

    for i, v in enumerate(values):
        ax.text(np.log10(v) + 0.1, i, f'{v:.2e} m', va='center')

    plt.tight_layout()
    plt.savefig('mhd_scales.png', dpi=150, bbox_inches='tight')
    plt.show()

# mhd_scales()
```

---

## 2. MHD 가정

### 2.1 기본 가정

```
MHD의 핵심 가정:

1. 준중성 (Quasi-neutrality):
   ni ≈ ne = n
   - Debye 길이보다 큰 스케일에서 성립
   - 전하 분리 무시

2. 저주파 근사 (Low-frequency):
   ω << ωci << ωce
   - 변위 전류 무시: ∂D/∂t ≈ 0
   - 전자 관성 무시

3. 유체 근사:
   L >> λmfp (평균 자유 경로)
   - 국소 열평형 가정
   - 운동론적 효과 무시

4. 비상대론적:
   v << c
   - 상대론적 보정 불필요

결과:
- Maxwell 방정식 단순화
- 전기장은 자기장과 속도에서 유도
- 자기 유도 방정식 도출
```

```python
def mhd_assumptions():
    """MHD 가정의 물리적 의미"""

    print("=" * 60)
    print("MHD 가정과 Maxwell 방정식 단순화")
    print("=" * 60)

    assumptions = """
    Maxwell 방정식:
    (1) ∇·E = ρc/ε₀        ← MHD에서: 준중성, ρc ≈ 0
    (2) ∇·B = 0            ← 그대로 유지
    (3) ∇×E = -∂B/∂t       ← 그대로 유지
    (4) ∇×B = μ₀J + μ₀ε₀∂E/∂t  ← 변위 전류 무시

    단순화된 Maxwell 방정식 (MHD):
    (1') ∇·E ≈ 0 (준중성)
    (2') ∇·B = 0
    (3') ∇×E = -∂B/∂t
    (4') ∇×B = μ₀J  (Ampère 법칙)

    일반화된 Ohm의 법칙:
    E + v×B = ηJ + (J×B)/ne - ∇pe/ne + (me/ne²)∂J/∂t
                ↑      ↑        ↑           ↑
             저항   Hall    전자압력    전자관성

    이상 MHD (Ideal MHD):
    E + v×B = 0  (모든 우변 항 무시)
    → 자기장이 유체와 함께 "동결" (frozen-in)

    저항성 MHD (Resistive MHD):
    E + v×B = ηJ  (저항 효과만 포함)
    → 자기장 확산 및 재결합 가능
    """
    print(assumptions)

mhd_assumptions()
```

---

## 3. 이상 MHD 방정식

### 3.1 지배 방정식

```
이상 MHD 방정식 체계:

1. 질량 보존:
   ∂ρ/∂t + ∇·(ρv) = 0

2. 운동량 보존:
   ρ(∂v/∂t + (v·∇)v) = -∇p + J×B + ρg
                            ↑
                        Lorentz 힘

   J×B = (∇×B)×B/μ₀ = (B·∇)B/μ₀ - ∇(B²/2μ₀)
                          ↑           ↑
                      자기장력     자기압

3. 에너지 보존 (단열):
   ∂/∂t(p/ρ^γ) + v·∇(p/ρ^γ) = 0

   또는: ∂p/∂t + v·∇p + γp∇·v = 0

4. 유도 방정식:
   ∂B/∂t = ∇×(v×B)

   (E = -v×B 대입)

5. 발산 조건:
   ∇·B = 0 (항상 유지)
```

```python
def ideal_mhd_equations():
    """이상 MHD 방정식 시각화"""

    print("=" * 60)
    print("이상 MHD 방정식 체계")
    print("=" * 60)

    equations = """
    보존 형태 (Conservative Form):

    ∂U/∂t + ∇·F = S

    여기서:
    ┌─────────────────────────────────────────────────────────┐
    │ 보존 변수 U:                                             │
    │   U = [ρ, ρv, B, E]ᵀ                                    │
    │   E = p/(γ-1) + ρv²/2 + B²/2μ₀ (총 에너지)              │
    ├─────────────────────────────────────────────────────────┤
    │ 플럭스 F (x 방향):                                       │
    │   F₁ = ρvx                        (질량)                │
    │   F₂ = ρvxv - BxB/μ₀ + P*I       (운동량)               │
    │   F₃ = vxB - Bxv                  (자기장)               │
    │   F₄ = (E + P*)vx - Bx(v·B)/μ₀   (에너지)               │
    │                                                         │
    │   P* = p + B²/2μ₀ (총 압력)                             │
    ├─────────────────────────────────────────────────────────┤
    │ 8개 변수: ρ, vx, vy, vz, Bx, By, Bz, p                  │
    │ 8개 방정식: 연속(1), 운동량(3), 유도(3), 에너지(1)       │
    │ + 제약조건: ∇·B = 0                                     │
    └─────────────────────────────────────────────────────────┘
    """
    print(equations)

    # Lorentz 힘 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) 자기 압력
    ax1 = axes[0]

    # 균일 자기장 영역과 외부
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)

    # 자기장 (z 방향, 중심 영역에서 강함)
    Bz = np.exp(-(X**2 + Y**2))

    # 자기 압력 ∇(B²/2μ₀)
    B_pressure = Bz**2 / (2 * mu0)
    grad_Bp_x, grad_Bp_y = np.gradient(B_pressure, x[1]-x[0])

    im = ax1.pcolormesh(X, Y, B_pressure * 1e6, cmap='hot', shading='auto')
    plt.colorbar(im, ax=ax1, label=r'$B^2/2\mu_0$ [μPa]')

    # 자기압 구배 (힘) 표시
    skip = 5
    ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              -grad_Bp_x[::skip, ::skip], -grad_Bp_y[::skip, ::skip],
              color='white', alpha=0.8)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(r'자기 압력 $-\nabla(B^2/2\mu_0)$: 바깥 방향 힘')
    ax1.set_aspect('equal')

    # (2) 자기 장력
    ax2 = axes[1]

    # 곡선 자기장선
    theta = np.linspace(0, 2*np.pi, 100)
    for r in [0.5, 1.0, 1.5]:
        x_line = r * (1 + 0.3 * np.sin(2*theta)) * np.cos(theta)
        y_line = r * (1 + 0.3 * np.sin(2*theta)) * np.sin(theta)
        ax2.plot(x_line, y_line, 'b-', linewidth=1.5)

    # 장력 방향 표시 (곡률 중심 방향)
    for r in [1.0]:
        theta_arrows = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for t in theta_arrows:
            x0 = r * (1 + 0.3 * np.sin(2*t)) * np.cos(t)
            y0 = r * (1 + 0.3 * np.sin(2*t)) * np.sin(t)

            # 곡률 중심 방향 (대략적)
            dx = -x0 * 0.3
            dy = -y0 * 0.3
            ax2.annotate('', xy=(x0+dx, y0+dy), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(r'자기 장력 $(B\cdot\nabla)B/\mu_0$: 곡률 중심 방향')
    ax2.set_aspect('equal')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.grid(True, alpha=0.3)

    # 범례
    ax2.plot([], [], 'b-', linewidth=2, label='자기력선')
    ax2.plot([], [], 'r-', linewidth=2, label='장력 방향')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('mhd_lorentz_force.png', dpi=150, bbox_inches='tight')
    plt.show()

# ideal_mhd_equations()
```

---

## 4. Alfven 속도

### 4.1 정의와 물리적 의미

Alfven 속도는 MHD의 기본 속도 스케일로, 유체역학에서의 음속에 해당합니다. 자기력선을 따라 횡방향 섭동이 전파되는 속도를 나타냅니다:

$$v_A = \frac{B}{\sqrt{\mu_0 \rho}}$$

여기서 $B$는 자기장 세기, $\mu_0$는 진공 투자율(vacuum permeability), $\rho$는 질량 밀도입니다. 핵심 통찰: 이 속도는 자기 장력(magnetic tension, 복원력, $\sim B^2/\mu_0$)과 관성($\sim \rho v^2$) 사이의 균형에서 나옵니다. 플라즈마 베타(plasma beta) $\beta = 2\mu_0 p / B^2$는 열압력 대 자기 압력의 비율로, 어느 힘이 역학을 지배하는지를 결정합니다.

```
Alfven 속도 (Alfvén velocity):

vA = B / √(μ₀ρ)

물리적 의미:
- 자기력선을 따라 전파하는 횡파의 속도
- 자기 에너지와 운동 에너지의 등분배
- B²/2μ₀ ~ ρvA²/2

무차원 파라미터:
- Alfven Mach 수: MA = v/vA
- 플라즈마 베타: β = 2μ₀p/B² = (cs/vA)² × 2/γ

  β << 1: 자기압 지배 (태양 코로나)
  β >> 1: 열압 지배 (태양 대류권)
  β ~ 1: 둘 다 중요
```

```python
def alfven_velocity_analysis():
    """Alfven 속도 분석"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) 다양한 환경에서의 Alfven 속도
    ax1 = axes[0]

    # 환경별 파라미터
    environments = {
        '태양 코로나': {'n': 1e14, 'B': 0.01, 'T': 1e6},
        '태양풍 (1AU)': {'n': 5e6, 'B': 5e-9, 'T': 1e5},
        '성간 매질': {'n': 1e6, 'B': 3e-10, 'T': 1e4},
        '토카막': {'n': 1e20, 'B': 5, 'T': 1e8},
        '액체 나트륨': {'n': 2.5e28, 'B': 0.1, 'T': 400}  # 밀도를 입자 수로 변환
    }

    names = []
    v_A_values = []
    v_s_values = []

    for name, params in environments.items():
        n = params['n']
        B = params['B']
        T = params['T']

        # 이온 질량 (플라즈마는 양성자, 나트륨은 Na)
        if '나트륨' in name:
            m_ion = 23 * mp  # Na 질량
            rho = n * m_ion
        else:
            m_ion = mp
            rho = n * m_ion

        # Alfven 속도
        v_A = B / np.sqrt(mu0 * rho)

        # 음속
        gamma = 5/3
        p = n * kB * T
        c_s = np.sqrt(gamma * p / rho)

        names.append(name)
        v_A_values.append(v_A)
        v_s_values.append(c_s)

    y_pos = np.arange(len(names))

    ax1.barh(y_pos - 0.2, np.log10(v_A_values), 0.4, label=r'$v_A$', color='blue')
    ax1.barh(y_pos + 0.2, np.log10(v_s_values), 0.4, label=r'$c_s$', color='red')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.set_xlabel('log₁₀(velocity [m/s])')
    ax1.set_title('다양한 환경에서의 Alfven 속도와 음속')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) 플라즈마 베타 레짐
    ax2 = axes[1]

    B_range = np.logspace(-10, 1, 100)  # 자기장 범위
    n_values = [1e6, 1e14, 1e20]  # 밀도
    T = 1e6  # 고정 온도

    colors = ['blue', 'green', 'red']
    for n, color in zip(n_values, colors):
        rho = n * mp
        p = n * kB * T

        # 플라즈마 베타
        beta = 2 * mu0 * p / B_range**2

        ax2.loglog(B_range, beta, color=color, linewidth=2, label=f'n = {n:.0e} m⁻³')

    ax2.axhline(y=1, color='black', linestyle='--', label=r'$\beta = 1$')
    ax2.fill_between([1e-10, 1e1], 1e-6, 1, alpha=0.2, color='blue', label=r'$\beta < 1$ (자기압 지배)')
    ax2.fill_between([1e-10, 1e1], 1, 1e6, alpha=0.2, color='red', label=r'$\beta > 1$ (열압 지배)')

    ax2.set_xlabel('B [T]')
    ax2.set_ylabel(r'$\beta = 2\mu_0 p / B^2$')
    ax2.set_title(f'플라즈마 베타 (T = {T:.0e} K)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-6, 1e6)
    ax2.set_xlim(1e-10, 1e1)

    plt.tight_layout()
    plt.savefig('alfven_velocity.png', dpi=150, bbox_inches='tight')
    plt.show()

# alfven_velocity_analysis()
```

---

## 5. MHD 파동

### 5.1 파동 유형

```
이상 MHD의 세 가지 파동 모드:

1. Alfven 파 (전단 Alfven 파):
   - 속도: vA = B₀/√(μ₀ρ)
   - 방향: 자기장 방향으로만 전파
   - 특성: 횡파, 비압축성, 자기력선 진동
   - 속도 섭동: δv ⊥ B₀, k

2. 빠른 자기음파 (Fast Magnetosonic):
   - 속도: vf = √[(vA² + cs²)/2 + √((vA² + cs²)² - 4vA²cs²cos²θ)/2]
   - 특성: 자기압 + 열압 복원력
   - 등방적 전파 (모든 방향)

3. 느린 자기음파 (Slow Magnetosonic):
   - 속도: vs = √[(vA² + cs²)/2 - √((vA² + cs²)² - 4vA²cs²cos²θ)/2]
   - 특성: 자기압과 열압의 반대 작용
   - 자기장 방향 근처로 전파

여기서 θ = ∠(k, B₀)

특수 경우:
- θ = 0 (평행): vf = max(vA, cs), vs = min(vA, cs)
- θ = π/2 (수직): vf = √(vA² + cs²), vs = 0
```

```python
def mhd_wave_speeds():
    """MHD 파동 속도 분산 관계"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) 파동 속도 vs 전파 각도
    ax1 = axes[0]

    theta = np.linspace(0, np.pi/2, 100)

    # vA/cs 비율 (플라즈마 베타 관련)
    ratios = [0.5, 1.0, 2.0]  # vA/cs

    for ratio in ratios:
        vA = ratio
        cs = 1.0

        # 빠른/느린 파
        term1 = (vA**2 + cs**2) / 2
        term2 = np.sqrt((vA**2 + cs**2)**2 - 4 * vA**2 * cs**2 * np.cos(theta)**2) / 2

        v_fast = np.sqrt(term1 + term2)
        v_slow = np.sqrt(np.maximum(term1 - term2, 0))

        # Alfven 파 (성분)
        v_alfven = np.abs(vA * np.cos(theta))

        ax1.plot(np.degrees(theta), v_fast, '-', linewidth=2, label=f'Fast (vA/cs={ratio})')
        ax1.plot(np.degrees(theta), v_slow, '--', linewidth=2, label=f'Slow (vA/cs={ratio})')
        ax1.plot(np.degrees(theta), v_alfven, ':', linewidth=2, label=f'Alfven (vA/cs={ratio})')

    ax1.set_xlabel('θ [degrees]')
    ax1.set_ylabel('Phase velocity / cs')
    ax1.set_title('MHD 파동 속도 vs 전파 각도')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 90)

    # (2) 프리드리히스 다이어그램 (극좌표)
    ax2 = axes[1]

    theta_full = np.linspace(0, 2*np.pi, 360)

    vA = 2.0
    cs = 1.0

    term1 = (vA**2 + cs**2) / 2
    term2 = np.sqrt((vA**2 + cs**2)**2 - 4 * vA**2 * cs**2 * np.cos(theta_full)**2) / 2

    v_fast = np.sqrt(term1 + term2)
    v_slow = np.sqrt(np.maximum(term1 - term2, 0))
    v_alfven = np.abs(vA * np.cos(theta_full))

    # 극좌표 -> 직교좌표
    x_fast = v_fast * np.sin(theta_full)
    y_fast = v_fast * np.cos(theta_full)

    x_slow = v_slow * np.sin(theta_full)
    y_slow = v_slow * np.cos(theta_full)

    x_alf = v_alfven * np.sin(theta_full)
    y_alf = v_alfven * np.cos(theta_full)

    ax2.plot(x_fast, y_fast, 'b-', linewidth=2, label='Fast')
    ax2.plot(x_slow, y_slow, 'r-', linewidth=2, label='Slow')
    ax2.plot(x_alf, y_alf, 'g--', linewidth=2, label='Alfven')

    # B₀ 방향 표시
    ax2.annotate('', xy=(0, 3), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.text(0.2, 2.8, r'$B_0$', fontsize=14)

    ax2.set_xlabel(r'$v_\perp / c_s$')
    ax2.set_ylabel(r'$v_\parallel / c_s$')
    ax2.set_title(f'Friedrichs 다이어그램 (vA/cs = {vA/cs})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig('mhd_waves.png', dpi=150, bbox_inches='tight')
    plt.show()

# mhd_wave_speeds()
```

### 5.2 Alfven 파 시각화

```python
def alfven_wave_visualization():
    """Alfven 파 시각화"""

    fig = plt.figure(figsize=(14, 10))

    # (1) Alfven 파 개념도 (3D)
    ax1 = fig.add_subplot(221, projection='3d')

    z = np.linspace(0, 4*np.pi, 100)
    t = 0

    # 평형 자기장 방향 (z)
    B0 = 1.0

    # 섭동 (y 방향 진동)
    k = 1
    omega = k  # vA = 1로 정규화
    By = 0.3 * np.sin(k*z - omega*t)

    # 자기력선 위치
    x_line = np.zeros_like(z)
    y_line = By

    ax1.plot(x_line, y_line, z, 'b-', linewidth=2, label='자기력선')
    ax1.plot([0]*len(z), [0]*len(z), z, 'k--', alpha=0.5, label='평형 위치')

    # 속도 섭동
    vy = -0.3 * np.sin(k*z - omega*t)  # v ∝ -B (Alfven 관계)
    skip = 10
    ax1.quiver(x_line[::skip], y_line[::skip], z[::skip],
              np.zeros(len(z[::skip])), vy[::skip], np.zeros(len(z[::skip])),
              color='red', length=0.5, arrow_length_ratio=0.3, label='속도 섭동')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z (B₀ 방향)')
    ax1.set_title('Alfven 파: 자기력선 횡방향 진동')
    ax1.legend()

    # (2) 시간 전개
    ax2 = fig.add_subplot(222)

    z = np.linspace(0, 4*np.pi, 200)
    times = [0, 0.5, 1.0, 1.5, 2.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    for t, color in zip(times, colors):
        By = 0.3 * np.sin(k*z - omega*t)
        ax2.plot(z, By, color=color, linewidth=1.5, label=f't = {t:.1f}')

    ax2.set_xlabel('z')
    ax2.set_ylabel(r'$\delta B_y$')
    ax2.set_title('Alfven 파 전파')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # (3) 에너지 분배
    ax3 = fig.add_subplot(223)

    # 자기 에너지와 운동 에너지
    z = np.linspace(0, 4*np.pi, 200)
    t = 0.5

    B_pert = 0.3 * np.sin(k*z - omega*t)
    v_pert = -0.3 * np.sin(k*z - omega*t)

    # 에너지 밀도 (단위 무시, 비례 관계만)
    E_mag = B_pert**2 / 2  # ∝ B²/2μ₀
    E_kin = v_pert**2 / 2  # ∝ ρv²/2

    ax3.plot(z, E_mag, 'b-', linewidth=2, label=r'$\delta B^2/2\mu_0$ (자기)')
    ax3.plot(z, E_kin, 'r--', linewidth=2, label=r'$\rho\delta v^2/2$ (운동)')
    ax3.plot(z, E_mag + E_kin, 'k-', linewidth=2, label='Total')

    ax3.set_xlabel('z')
    ax3.set_ylabel('Energy density')
    ax3.set_title('Alfven 파 에너지 등분배')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (4) 섭동 관계
    ax4 = fig.add_subplot(224)

    # δv = -δB/√(μ₀ρ) (Alfven 관계)
    info_text = """
Alfven 파 특성:

1. 전파 방향: B₀ 방향 (k ∥ B₀)
2. 편광: 횡파 (δv ⊥ B₀, δB ⊥ B₀)
3. 속도: vA = B₀/√(μ₀ρ)

섭동 관계:
δv = ∓ δB/√(μ₀ρ)
(± for k·B₀ ≷ 0)

특징:
- 비압축성: ∇·δv = 0
- 밀도/압력 섭동 없음
- 자기력선이 "기타 줄"처럼 진동
- 장력이 복원력 제공

Alfven 정리 (동결 조건):
이상 MHD에서 자기력선은
유체와 함께 움직임
("frozen-in" 조건)
    """
    ax4.text(0.1, 0.95, info_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig('alfven_wave_detail.png', dpi=150, bbox_inches='tight')
    plt.show()

# alfven_wave_visualization()
```

---

## 6. 자기압과 자기장력

### 6.1 힘의 분해

```
Lorentz 힘 분해:

J×B = (1/μ₀)(∇×B)×B

벡터 항등식 사용:
(∇×B)×B = (B·∇)B - ∇(B²/2)

따라서:
J×B = (B·∇)B/μ₀ - ∇(B²/2μ₀)
        ↑           ↑
     자기장력    자기압 구배

1. 자기 압력 (Magnetic Pressure):
   pm = B²/2μ₀

   - 등방적 (모든 방향 동일)
   - B가 큰 영역에서 작은 영역으로 힘
   - 자기장선 밀집 → 높은 압력

2. 자기 장력 (Magnetic Tension):
   T = (B·∇)B/μ₀ = (B²/μ₀)κ

   - 곡률 κ의 중심 방향
   - 휜 자기력선을 펴려는 힘
   - "기타 줄 장력"과 유사
```

```python
def magnetic_pressure_tension():
    """자기압과 자기장력의 균형 예시"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (1) 자기 압력 평형 (Z-pinch)
    ax1 = axes[0]

    r = np.linspace(0.1, 2, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    R, Theta = np.meshgrid(r, theta)

    # 자기장 (축 방향 전류에 의한 θ 방향 자기장)
    # B_theta ∝ 1/r (외부), B_theta ∝ r (내부)
    r_plasma = 1.0
    B_theta = np.where(R < r_plasma, R, 1/R)

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # 자기압
    P_mag = B_theta**2 / 2

    im = ax1.pcolormesh(X, Y, P_mag, cmap='hot', shading='auto')
    plt.colorbar(im, ax=ax1, label=r'$B^2/2\mu_0$')

    # 자기력선 (동심원)
    for r_line in [0.3, 0.6, 0.9, 1.2, 1.5]:
        circle = plt.Circle((0, 0), r_line, fill=False, color='blue', linewidth=1)
        ax1.add_patch(circle)

    # 압력 구배 방향
    ax1.annotate('', xy=(0.7, 0), xytext=(0.3, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax1.annotate('', xy=(1.7, 0), xytext=(1.3, 0),
                arrowprops=dict(arrowstyle='<-', color='white', lw=2))

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Z-pinch: 자기압이 플라즈마 압축')
    ax1.set_aspect('equal')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)

    # (2) 자기 장력 (휜 자기력선)
    ax2 = axes[1]

    # 휜 자기력선
    x = np.linspace(-2, 2, 100)
    y_lines = [0.3 * np.sin(np.pi * x / 2),
              0.6 * np.sin(np.pi * x / 2),
              0.9 * np.sin(np.pi * x / 2)]

    for y in y_lines:
        ax2.plot(x, y, 'b-', linewidth=2)

    # 장력 방향 (곡률 중심 방향 = 아래)
    x_arrows = [-1, 0, 1]
    for xa in x_arrows:
        idx = np.argmin(np.abs(x - xa))
        y_arrow = 0.6 * np.sin(np.pi * xa / 2)
        # 곡률이 양수면 장력은 아래
        tension_dir = -1 if xa == 0 else (-0.5 if xa > 0 else 0.5)
        ax2.annotate('', xy=(xa, y_arrow + tension_dir * 0.3),
                    xytext=(xa, y_arrow),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('자기 장력: 휜 자기력선을 펴는 힘')
    ax2.set_aspect('equal')
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)

    # 범례
    ax2.plot([], [], 'b-', linewidth=2, label='자기력선')
    ax2.plot([], [], 'r-', linewidth=2, label='장력 방향')
    ax2.legend()

    # (3) 평형 예시
    ax3 = axes[2]

    info_text = """
MHD 평형 조건:

정적 평형에서:
∇p = J×B = (B·∇)B/μ₀ - ∇(B²/2μ₀)

재배열:
∇(p + B²/2μ₀) = (B·∇)B/μ₀
    ↑               ↑
 총 압력        자기 장력

응용 예시:

1. θ-pinch:
   - Bz 만 존재 (직선)
   - 장력 없음, 압력 평형
   - ∂/∂z(p + B²/2μ₀) = 0

2. Z-pinch:
   - Bθ 만 존재 (원형)
   - 압력 + 장력이 열압과 균형
   - (1/r)∂/∂r[r(p + B²/2μ₀)] = Bθ²/μ₀r

3. 스크류 핀치:
   - Bz + Bθ 조합
   - 복잡한 평형 조건
   - 토카막의 기본 형태
    """
    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig('magnetic_pressure_tension.png', dpi=150, bbox_inches='tight')
    plt.show()

# magnetic_pressure_tension()
```

---

## 7. Frozen-in 정리

### 7.1 자기장 동결 조건

```
자기력선 동결 (Frozen-in Theorem):

이상 MHD에서:
E + v×B = 0  (무저항)

유도 방정식:
∂B/∂t = ∇×(v×B)

물리적 의미:
1. 자기력선은 유체 요소와 함께 움직임
2. 자기력선에 "표지"를 하면 유체와 함께 이동
3. 두 유체 요소가 같은 자기력선 위에 있으면
   영원히 같은 자기력선 위에 있음

자기 플럭스 보존:
d/dt ∫∫ B·dS = 0  (움직이는 면에 대해)

위반 조건 (저항성 MHD):
E + v×B = ηJ

유도 방정식:
∂B/∂t = ∇×(v×B) + η/μ₀ ∇²B
                      ↑
                  자기 확산

자기 Reynolds 수:
Rm = μ₀vL/η

Rm >> 1: 동결 조건 성립 (대부분의 천체 플라즈마)
Rm ~ 1: 확산과 대류 경쟁
Rm << 1: 확산 지배
```

```python
def frozen_in_theorem():
    """Frozen-in 정리 시각화"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (1) 이상 MHD: 자기력선이 유체와 함께 이동
    ax1 = axes[0, 0]

    # 초기 격자와 자기력선
    x = np.linspace(0, 2, 6)
    y = np.linspace(0, 1, 6)

    # 초기 상태
    for xi in x:
        ax1.plot([xi, xi], [0, 1], 'b-', linewidth=1, alpha=0.5)
    for yi in y:
        ax1.plot([0, 2], [yi, yi], 'b-', linewidth=1, alpha=0.5)

    # 변형 후 (전단 유동)
    # v = (y, 0) -> x' = x + t*y
    t = 0.5
    for yi in y:
        x_new = x + t * yi
        ax1.plot(x_new, np.full_like(x_new, yi), 'r--', linewidth=1, alpha=0.7)

    for xi in x:
        y_line = np.linspace(0, 1, 20)
        x_line = xi + t * y_line
        ax1.plot(x_line, y_line, 'r--', linewidth=1, alpha=0.7)

    ax1.plot([], [], 'b-', linewidth=2, label='초기 (자기력선)')
    ax1.plot([], [], 'r--', linewidth=2, label='변형 후')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('이상 MHD: 자기력선이 유체와 동결')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # (2) 자기 플럭스 보존
    ax2 = axes[0, 1]

    theta = np.linspace(0, 2*np.pi, 100)

    # 초기 원형 루프
    r0 = 1
    x0 = r0 * np.cos(theta)
    y0 = r0 * np.sin(theta)
    ax2.plot(x0, y0, 'b-', linewidth=2, label='초기 루프')
    ax2.fill(x0, y0, alpha=0.2, color='blue')

    # 변형된 루프 (압축)
    rx, ry = 0.5, 2.0  # 압축/연신
    x1 = rx * np.cos(theta)
    y1 = ry * np.sin(theta)
    ax2.plot(x1, y1, 'r-', linewidth=2, label='변형된 루프')
    ax2.fill(x1, y1, alpha=0.2, color='red')

    # 면적 비교
    A0 = np.pi * r0**2
    A1 = np.pi * rx * ry

    ax2.text(0, 0, f'Φ = ∫B·dA\n보존!', ha='center', va='center', fontsize=11)
    ax2.text(1.5, 0, f'A₀ = {A0:.2f}', fontsize=10)
    ax2.text(0.3, 1.5, f'A₁ = {A1:.2f}', fontsize=10)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('자기 플럭스 보존')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.grid(True, alpha=0.3)

    # (3) 자기 Reynolds 수
    ax3 = axes[1, 0]

    # 다양한 환경의 Rm
    environments = {
        '실험실\n플라즈마': 1e2,
        '태양\n광구': 1e6,
        '태양\n코로나': 1e12,
        '성간\n매질': 1e18,
        '액체\n금속': 1e1
    }

    names = list(environments.keys())
    Rm_values = list(environments.values())

    y_pos = np.arange(len(names))
    colors = ['red' if Rm < 100 else 'green' for Rm in Rm_values]

    ax3.barh(y_pos, np.log10(Rm_values), color=colors)
    ax3.axvline(x=np.log10(100), color='black', linestyle='--', label=r'$R_m = 100$')

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names)
    ax3.set_xlabel(r'log₁₀($R_m$)')
    ax3.set_title('자기 Reynolds 수 비교')
    ax3.grid(True, alpha=0.3)

    # 범례
    ax3.plot([], [], 'g-', linewidth=10, label='동결 조건 성립')
    ax3.plot([], [], 'r-', linewidth=10, label='확산 효과 중요')
    ax3.legend()

    # (4) 개념 정리
    ax4 = axes[1, 1]

    info_text = """
Frozen-in 정리 요약:

조건: 이상 MHD (η = 0, E + v×B = 0)

결과:
1. ∂B/∂t = ∇×(v×B)
2. d/dt ∫∫ B·dS = 0 (이동 면)
3. 자기력선은 유체와 함께 이동

물리적 해석:
- 자기력선에 "동결"된 유체 요소
- 자기장 압축 ↔ 밀도 증가
- B/ρ ∝ 상수 (1D 압축)

위반 시 (η ≠ 0):
- 자기장 확산: τ_diff = μ₀L²/η
- 자기 재결합 가능
- 에너지 변환 (자기 → 운동/열)

중요성:
- 태양 플레어: 재결합
- 토카막: 동결 조건 중요
- 자기권: 재결합 현상
    """
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig('frozen_in_theorem.png', dpi=150, bbox_inches='tight')
    plt.show()

# frozen_in_theorem()
```

---

## 8. 연습 문제

### 연습 1: 알벤 속도(Alfvén Velocity)
태양 코로나에서 B = 10 G, n = 10⁸ cm⁻³일 때 알벤 속도를 계산하시오. 이것을 음속(T = 10⁶ K)과 비교하고 플라즈마 베타(plasma beta)를 구하시오.

<details><summary>정답 보기</summary>

```python
import numpy as np

# 물리 상수
mu_0 = 4 * np.pi * 1e-7   # 진공 투자율 [H/m]
k_B  = 1.38e-23            # 볼츠만 상수 [J/K]
m_p  = 1.67e-27            # 양성자 질량 [kg]

# 태양 코로나 파라미터
B = 10e-4        # 10 G → Tesla (1 G = 10^-4 T)
n = 1e14         # 10^8 cm^-3 → m^-3 (1 cm^-3 = 10^6 m^-3)
T = 1e6          # 온도 [K]
gamma = 5/3      # 단열 지수

# 밀도
rho = n * m_p
print(f"밀도 ρ = {rho:.3e} kg/m³")

# 알벤 속도: v_A = B / sqrt(μ₀ρ)
v_A = B / np.sqrt(mu_0 * rho)
print(f"알벤 속도 v_A = {v_A/1e3:.1f} km/s")

# 음속: c_s = sqrt(γ k_B T / m_p)
c_s = np.sqrt(gamma * k_B * T / m_p)
print(f"음속 c_s = {c_s/1e3:.1f} km/s")

# 플라즈마 베타: β = 열압력 / 자기압력
p_thermal = n * k_B * T        # 열압력
p_magnetic = B**2 / (2 * mu_0) # 자기압력
beta = p_thermal / p_magnetic
print(f"\n열압력 = {p_thermal:.3e} Pa")
print(f"자기압력 = {p_magnetic:.3e} Pa")
print(f"플라즈마 베타 β = {beta:.4f}")
if beta < 1:
    print("β < 1: 자기장 지배 플라즈마 (태양 코로나 전형적)")
```

v_A ≈ 2187 km/s, c_s ≈ 151 km/s → v_A ≫ c_s (자기장 지배 플라즈마)

β ≈ 0.048 ≪ 1: 태양 코로나는 자기압력이 열압력보다 훨씬 크며, 이것이 코로나 구조가 자기장에 의해 지배되는 이유입니다.
</details>

### 연습 2: MHD 파동 속도(MHD Wave Speed)
v_A = 2c_s인 경우, 자기장에 수직한 방향(θ = 90°)으로 전파하는 빠른 자기음파(fast magnetosonic wave)의 위상 속도를 구하시오.

<details><summary>정답 보기</summary>

MHD 파동의 위상 속도 공식:

빠른/느린 자기음파의 위상 속도:
```
v²_{fast,slow} = (1/2){(c_s² + v_A²) ± √[(c_s² + v_A²)² - 4c_s²v_A²cos²θ]}
```

θ = 90°에서 cos θ = 0이므로:
```
v²_{fast} = c_s² + v_A²
v²_{slow} = 0
```

```python
import numpy as np

# v_A = 2c_s 설정
c_s = 1.0       # 정규화
v_A = 2.0 * c_s
theta = np.pi / 2  # 90도

# 빠른/느린 자기음파 위상 속도
def mhd_wave_speeds(c_s, v_A, theta):
    sum_sq  = c_s**2 + v_A**2
    disc    = np.sqrt(sum_sq**2 - 4*c_s**2 * v_A**2 * np.cos(theta)**2)
    v_fast  = np.sqrt((sum_sq + disc) / 2)
    v_slow  = np.sqrt((sum_sq - disc) / 2)
    v_alfven = v_A * abs(np.cos(theta))  # 알벤파
    return v_fast, v_slow, v_alfven

v_f, v_s, v_alf = mhd_wave_speeds(c_s, v_A, theta)
print(f"θ = 90°, v_A = 2c_s:")
print(f"  빠른 자기음파: v_fast = {v_f:.4f} c_s = √(c_s² + v_A²) = √5 c_s")
print(f"  느린 자기음파: v_slow = {v_s:.6f} c_s (≈ 0)")
print(f"  알벤파:        v_A = {v_alf:.4f} c_s (cos90°=0 → 0)")
print(f"\n검증: √(1² + 2²) = √5 = {np.sqrt(5):.4f}")
```

θ = 90°에서 빠른 자기음파의 위상 속도 v_fast = √(c_s² + v_A²) = √(1 + 4)c_s = **√5 c_s ≈ 2.236c_s**

느린 자기음파와 알벤파는 수직 방향으로 전파하지 않습니다(v = 0). 이것은 알벤파와 느린 자기음파가 자기장을 따라 전파하는 특성을 반영합니다.
</details>

### 연습 3: 자기압 평형(Magnetic Pressure Equilibrium)
균일한 자기장 Bz 영역과 무자기장 영역 사이의 경계에서 압력 평형 조건을 구하시오.

<details><summary>정답 보기</summary>

MHD 압력 평형 조건: 총 압력(열압력 + 자기압력)이 경계에서 연속:

```
p₁ + B₁²/(2μ₀) = p₂ + B₂²/(2μ₀)
```

영역 1(자기장 있음): B₁ = Bz, 압력 p₁
영역 2(무자기장): B₂ = 0, 압력 p₂

```python
import numpy as np
mu_0 = 4 * np.pi * 1e-7  # [H/m]

def magnetic_pressure_equilibrium(Bz, T1=1e6, n1=1e18):
    """자기장 영역 vs 무자기장 영역 압력 평형"""
    k_B = 1.38e-23; m_p = 1.67e-27

    # 자기장 영역 (영역 1)
    p_magnetic = Bz**2 / (2 * mu_0)
    p_thermal1 = n1 * k_B * T1
    p_total1 = p_thermal1 + p_magnetic
    beta1 = p_thermal1 / p_magnetic

    print(f"자기장 Bz = {Bz*1e4:.1f} G")
    print(f"  자기압력 = {p_magnetic:.3e} Pa")
    print(f"  열압력₁  = {p_thermal1:.3e} Pa")
    print(f"  플라즈마 β₁ = {beta1:.3f}")

    # 평형 조건: p₂ = p_total1 (B₂=0)
    p_thermal2 = p_total1
    n2 = p_thermal2 / (k_B * T1)  # 같은 온도 가정
    print(f"\n무자기장 영역 (평형 조건):")
    print(f"  필요한 열압력₂ = {p_thermal2:.3e} Pa")
    print(f"  필요한 밀도   n₂ = {n2:.3e} m^-3")
    print(f"  밀도 비 n₂/n₁ = {n2/n1:.3f}")
    print(f"  → 자기장 없는 영역이 더 높은 밀도 필요")

# 태양 흑점 예시: B ≈ 3000 G
magnetic_pressure_equilibrium(Bz=3000e-4, T1=6000, n1=1e23)
```

평형 조건: **p₂ = p₁ + B₁²/(2μ₀)**

자기장이 없는 외부 영역은 자기압력만큼 더 높은 열압력이 필요합니다. 태양 흑점은 강한 자기장(~3000 G)으로 인해 내부 압력이 낮아 주변보다 온도가 낮고(~4500 K vs 5800 K) 어둡게 보입니다.
</details>

### 연습 4: 자기 동결(Frozen-in) 조건
길이 L = 1 Mm, 전도도 σ = 10⁶ S/m인 플라즈마에서 자기 확산 시간(magnetic diffusion time)을 계산하시오. 속도 v = 100 km/s일 때 자기 레이놀즈 수(magnetic Reynolds number)는 얼마인가?

<details><summary>정답 보기</summary>

```python
import numpy as np

# 파라미터
mu_0  = 4 * np.pi * 1e-7  # [H/m]
sigma = 1e6                 # 전도도 [S/m]
L     = 1e6                 # 1 Mm = 10^6 m
v     = 100e3               # 100 km/s [m/s]

# 자기 확산 시간: τ_d = μ₀ σ L²
# (자기장이 저항에 의해 확산되는 시간 척도)
tau_d = mu_0 * sigma * L**2
print(f"자기 확산 시간 τ_d = μ₀σL² = {tau_d:.3e} s")
print(f"                       = {tau_d/(3600*24*365):.1f} 년")

# 알벤 시간 (대류 시간): τ_A = L / v
tau_A = L / v
print(f"\n대류 시간 τ_A = L/v = {tau_A:.1f} s")

# 자기 레이놀즈 수: Rm = μ₀ σ v L = τ_d / τ_A
Rm = mu_0 * sigma * v * L
print(f"\n자기 레이놀즈 수 Rm = μ₀σvL = {Rm:.3e}")
print(f"                      Rm = τ_d/τ_A = {tau_d/tau_A:.3e}")

if Rm > 1:
    print(f"\nRm ≫ 1: 자기장이 플라즈마에 동결(frozen-in)")
    print("→ 자기 확산이 무시될 만큼 느림")
    print("→ 자기 플럭스가 유체 소체와 함께 이동")
else:
    print(f"\nRm < 1: 자기장 확산이 지배적")
```

τ_d ≈ 1.26×10⁶ s ≈ 14.6일, τ_A = 10 s

Rm = μ₀σvL ≈ **1.26×10⁵ ≫ 1**

Rm이 매우 크므로 자기장은 플라즈마에 **동결(frozen-in)**됩니다. 태양 플라즈마에서 Rm ~ 10⁸~10¹², 지구 외핵에서 Rm ~ 10³ 정도로 모두 1보다 훨씬 큽니다. 이 조건에서 자기 플럭스는 보존되고 자기장선은 유체 소체와 함께 이동합니다.
</details>

---

## 9. 참고자료

### 핵심 교재
- Goedbloed & Poedts, "Principles of Magnetohydrodynamics"
- Kulsrud, "Plasma Physics for Astrophysics"
- Freidberg, "Ideal MHD"

### 논문/리뷰
- Alfvén (1942) 원논문 (MHD 파동)
- Priest & Forbes, "Magnetic Reconnection" (재결합)

### 온라인 자료
- Thorne & Blandford, "Modern Classical Physics" (Ch. 19)
- Chen, "Introduction to Plasma Physics" (MHD 장)

---

## 요약

```
MHD 기초 핵심:

1. MHD 가정:
   - 준중성, 저주파, 유체 근사
   - L >> λD, c/ωpi
   - Maxwell 단순화

2. 이상 MHD 방정식:
   - 연속: ∂ρ/∂t + ∇·(ρv) = 0
   - 운동량: ρDv/Dt = -∇p + J×B
   - 에너지: D(p/ρ^γ)/Dt = 0
   - 유도: ∂B/∂t = ∇×(v×B)
   - 제약: ∇·B = 0

3. 주요 속도:
   - Alfven: vA = B/√(μ₀ρ)
   - 음속: cs = √(γp/ρ)
   - 플라즈마 베타: β = 2μ₀p/B²

4. MHD 파동:
   - Alfven 파: vA (자기장 방향)
   - 빠른 자기음파: √(vA² + cs²) (수직)
   - 느린 자기음파: min(vA, cs) (평행)

5. Lorentz 힘:
   J×B = -∇(B²/2μ₀) + (B·∇)B/μ₀
          자기압        자기장력

6. Frozen-in:
   - 이상 MHD: E + v×B = 0
   - 자기력선이 유체와 동결
   - Rm >> 1 일 때 유효
```

---

## 연습 문제

### 연습 1: 알벤 속도(Alfven Velocity)와 플라즈마 베타(Plasma Beta)
자기장 B = 10 G = 10⁻³ T, 수밀도 n = 10⁸ cm⁻³ = 10¹⁴ m⁻³인 태양 코로나(solar corona)에서 다음을 계산하세요 (양성자 플라즈마 가정): (a) ρ = n·mp를 사용한 알벤 속도(Alfven velocity) vA = B/√(μ₀ρ), (b) T = 10⁶ K에서의 음속(sound speed) cs = √(γp/ρ), (c) 플라즈마 베타(plasma beta) β = 2μ₀p/B². 이 환경이 자기 지배(magnetically dominated, β < 1)인지 열 지배(thermally dominated, β > 1)인지 분류하고 파동 전파에 대한 물리적 의미를 설명하세요.

### 연습 2: MHD 파동(Wave) 위상 속도
vA = 2cs인 플라즈마에서 배경 자기장 B₀에 대해 전파 각도 θ = 0°, 30°, 60°, 90°에서 세 가지 MHD 파동 모드 (빠른 자기음파(fast magnetosonic), 알벤파(Alfven wave), 느린 자기음파(slow magnetosonic)) 의 위상 속도를 모두 계산하세요. 5.1절에 제시된 분산 관계(dispersion relation)를 사용하세요. 직교 좌표계에서 v⊥ 대 v‖로 세 파동 모드의 Friedrichs 다이어그램을 그리세요. 빠른 파동이 최대 속도에 도달하는 각도는 어디인가요?

### 연습 3: 자기 압력(Magnetic Pressure) 평형
전류 시트(current sheet)가 자기장 B₁ = 10 G, 열압력 p₁ = 0.5 × 10⁻³ Pa인 영역과 자기장이 없는(B₂ = 0) 열압력 p₂인 영역을 분리합니다. 전류 시트를 가로지르는 전체 압력 균형(total pressure balance) p + B²/(2μ₀) = 상수를 사용하여 p₂를 계산하세요. 그런 다음 전체 압력의 점프가 0임을 검증하세요. 이것이 평형 조건(equilibrium condition)임을 확인하세요.

### 연습 4: 동결-인(Frozen-in) 정리와 자기 확산(Magnetic Diffusion)
전기 전도도(electrical conductivity) σ = 10⁶ S/m, 특성 길이 L = 1 Mm = 10⁶ m인 저항성 플라즈마(resistive plasma)에 대해 다음을 계산하세요: (a) 자기 확산율(magnetic diffusivity) η = 1/(μ₀σ), (b) 자기 확산 타임스케일(magnetic diffusion timescale) τdiff = μ₀σL², (c) 유속 v = 100 km/s에서의 자기 레이놀즈 수(magnetic Reynolds number) Rm = μ₀σvL. 동결-인(frozen-in) 조건이 유효한가요? τdiff를 B = 10 G, n = 10¹⁴ m⁻³를 사용한 알벤 도달 시간(Alfven crossing time) τA = L/vA와 비교하세요.

### 연습 5: 이상 MHD(Ideal MHD) 보존 형식(Conservative Form)
1차원 이상 MHD에서 보존 변수 벡터 U와 플럭스 벡터 F(U)의 모든 7개 성분을 명시적으로 작성하세요. 각 성분의 단위를 확인하여 표현식이 차원적으로 일관성이 있는지 검증하세요. 유도 방정식(induction equation) dB/dt = ∇×(v×B)에서 출발하여, 이것이 보존 형 MHD 시스템에서 사용되는 플럭스 형식과 대수적으로 동치임을 보이세요.

---

다음 레슨에서는 MHD 방정식의 수치해법을 다룹니다.
