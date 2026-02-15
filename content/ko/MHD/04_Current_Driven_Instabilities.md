# 4. 전류 구동 불안정성

## 학습 목표

- 플라즈마 전류에 의해 구동되는 kink 불안정성(외부 및 내부) 이해
- Sausage 불안정성과 그 안정화 분석
- Tearing 모드 이론과 자기 섬 형성 유도
- 저항 벽 모드와 피드백 안정화 연구
- 토카막의 신고전 tearing 모드(NTM) 이해
- Tearing 모드에 대한 Δ' 매개변수와 성장률 계산
- 전류 구동 불안정성에 대한 수치 솔버 구현
- 이론과 실험 관측(sawtooth, disruption)의 연결

## 1. 전류 구동 불안정성 소개

전류 구동 불안정성은 **자유 에너지원**이 플라즈마 전류(또는 등가적으로, 자기장 구성)인 MHD 모드입니다. 압력 구동 모드와 달리, 이것들은 **압력이 0**일 때도 존재할 수 있습니다.

**주요 특징**:
- 전류/자기장 구성에 의해 구동
- 이상적(저항 없음) 또는 저항성(재결합) 가능
- 종종 달성 가능한 최대 플라즈마 전류 제한
- 토카막에서 주요 disruption으로 이어짐

```
전류 구동 불안정성의 분류:
==============================================

이상 MHD (η = 0):
├─ 외부 kink (m=1): 기둥 굽힘, q(a) < 1
├─ 내부 kink (m=1): 코어 kink, q(0) < 1
└─ Sausage (m=0): 조임/팽창, Bz로 안정화

저항 MHD (η ≠ 0):
├─ Tearing 모드: q = m/n에서 자기 재결합
├─ 저항 벽 모드: 벽 안정화 kink
└─ 신고전 tearing 모드 (NTM): Bootstrap 전류
```

## 2. Kink 불안정성

### 2.1 외부 Kink 모드

**외부 kink** (토로이달 시스템에서 m=1, n=1)는 플라즈마 기둥의 전역 변위입니다.

**물리적 그림**:
```
외부 Kink (m=1):
===================

이전:          이후:
   │                ╱
   │               ╱  ← 기둥 굽힘
   │     →        │
   │               ╲
   │                ╲

전체 플라즈마 기둥이
나선형으로 변위
```

**에너지 균형**:
- **불안정화**: 기둥이 굽을 때 자기 압력 불균형
- **안정화**: 축방향 장 $B_z$가 선 굽힘 에너지 제공

**Kruskal-Shafranov 기준** (레슨 2에서):

$$
q(a) > \frac{m}{n}
$$

$(m,n) = (1,1)$에 대해:

$$
q(a) > 1
$$

### 2.2 Sharp-Boundary 모델

다음을 가진 sharp-boundary Z-pinch에 대해:
- 반지름 $a$
- 전류 $I$
- 축방향 장 $B_z$

$m=1$ kink에 대한 분산 관계:

$$
\omega^2 = -\frac{B_\theta^2(a)}{\mu_0\rho}\left(1 - q^2(a)\right)
$$

**안정성**:
- $q(a) < 1$이면: $\omega^2 < 0$ → **불안정**
- $q(a) > 1$이면: $\omega^2 > 0$ → **안정** (진동)

**성장률** (불안정할 때):

$$
\gamma = \frac{B_\theta(a)}{\sqrt{\mu_0\rho}}\sqrt{1 - q^2(a)}
$$

이것은 **Alfvén 시간 척도**에 있습니다: $\tau_A \sim a/v_A$.

### 2.3 내부 Kink 모드

**내부 kink** 모드는 플라즈마 어딘가에서(일반적으로 축 상에서) $q < 1$일 때 발생합니다.

**특징**:
- $q < 1$인 영역에 국소화
- 벽이나 가장자리 불안정성 불필요
- 토카막에서 **sawtooth 진동** 유발

**Sawtooth crash**:
1. 중심 온도 상승
2. 중심 전류 밀도 상승
3. $q(0)$이 1 아래로 하락
4. 내부 kink 촉발
5. 빠른 재결합이 $T$와 $q$ 프로파일 평탄화
6. 주기 반복

```
Sawtooth 진동:
====================

T₀ |     ╱|     ╱|     ╱
   |    ╱ |    ╱ |    ╱
   |   ╱  |   ╱  |   ╱
   |  ╱   |  ╱   |  ╱
   | ╱    | ╱    | ╱
   |╱_____|╱_____|╱_____ 시간
     ↑      ↑      ↑
   Crash  Crash  Crash
   (m=1,n=1 내부 kink)
```

내부 kink 안정성을 위한 **Bussac 기준**:

$$
\beta_p < \frac{0.3}{q_0^2}
$$

여기서 $\beta_p$는 폴로이달 베타이고 $q_0 = q(0)$입니다.

### 2.4 Sausage 불안정성 (m=0)

**Sausage 모드** ($m=0$)는 대칭 조임/팽창입니다:

```
Sausage 모드 (m=0):
==================

이전:        이후:
  ║║║║║         ║║║║║
  ║║║║║    →   ╱║║║║║╲
  ║║║║║         ║║║║║
  ║║║║║        ╱║║║║║╲
  ║║║║║         ║║║║║

기둥이 z를 따라
교대로 조이고 팽창
```

**분산 관계** (Z-pinch, $B_z$ 없음):

$$
\omega^2 = -\frac{B_\theta^2(a)}{\mu_0\rho}k_z^2 a^2
$$

$k_z \neq 0$에 대해 항상 불안정 ($\omega^2 < 0$).

**축방향 장** $B_z$와 함께:

$$
\omega^2 = \frac{B_\theta^2(a)}{\mu_0\rho}k_z^2 a^2\left(\frac{B_z^2}{B_\theta^2(a)} - 1\right)
$$

**안정성**: $B_z > B_\theta(a)$이면 안정, 즉 $q(a) > 1$.

### 2.5 나선형 섭동

일반 나선형 섭동:

$$
\boldsymbol{\xi} = \hat{\boldsymbol{\xi}}(r)e^{i(m\theta + k_z z - \omega t)}
$$

토로이달 식별을 가진 원통에 대해: $k_z = n/R_0$.

**공명 표면**: $\mathbf{k}\cdot\mathbf{B} = 0$인 곳:

$$
mB_\theta(r_s) + k_z B_z = 0 \quad \Rightarrow \quad q(r_s) = \frac{m}{n}
$$

공명 표면에서 장선 굽힘이 사라짐 → **이상 MHD가 특이해짐** → 저항이 필요.

## 3. Tearing 모드 이론

### 3.1 저항 불안정성 기초

**이상 MHD**에서 자기 위상은 보존됩니다(동결 정리). **저항 MHD**에서 유한 저항 $\eta$는 **자기 재결합**을 허용합니다.

**Tearing 모드**: $q = m/n$인 유리수 표면에서의 저항 불안정성.

**물리적 메커니즘**:
1. 섭동이 $q = m/n$ 표면에 전류 시트 생성
2. 저항 확산이 장선 파괴
3. 재결합이 자기 섬 형성
4. 섬이 성장하여 플럭스 표면을 찢음

```
Tearing 모드와 자기 섬:
==================================

초기:           재결합됨:
 ═══════            ═══╱═╲═══
 ═══════    →       ══╱   ╲══  (O-점)
 ═══════            ═╱  ⊗  ╲═
   ↑                 ╲     ╱
유리수              ═══════
표면              (X-점)

섬 폭 w가 시간에 따라 성장
```

### 3.2 Tearing 층 분석

공명 표면 $r = r_s$ 근처에, 폭 $\delta$의 좁은 **tearing 층**이 있습니다.

**층 구조**:
- 내부 층 ($|r - r_s| < \delta$): 저항 중요
- 외부 영역 ($|r - r_s| > \delta$): 이상 MHD

**경계층 방정식** (constant-$\psi$ 영역):

$$
\frac{d^4\psi}{dx^4} - k^2\frac{d^2\psi}{dx^2} + \frac{i\omega\mu_0}{\eta}\psi = 0
$$

여기서 $x = r - r_s$는 공명 표면으로부터의 거리입니다.

**정합 조건**: 내부 해가 외부 이상 해와 정합해야 합니다.

### 3.3 Δ' 매개변수

**Δ' 매개변수**는 공명 표면을 가로지르는 $\psi$의 로그 도함수 점프를 특징화합니다:

$$
\Delta' = \left[\frac{d(\ln\psi')}{dr}\right]_{r_s^+}^{r_s^-} = \left(\frac{1}{\psi'}\frac{d\psi'}{dr}\right)_{r_s^+} - \left(\frac{1}{\psi'}\frac{d\psi'}{dr}\right)_{r_s^-}
$$

여기서 $\psi$는 섭동된 폴로이달 플럭스입니다.

**안정성 기준**:
- $\Delta' > 0$: **불안정** (tearing)
- $\Delta' < 0$: **안정**
- $\Delta' = 0$: **한계**

### 3.4 성장률 스케일링

성장률은 영역에 따라 다릅니다:

**Constant-$\psi$ 영역** (저항 지배):

$$
\gamma \propto \eta^{3/5}(\Delta')^{4/5}
$$

**Non-constant-$\psi$ 영역** (더 느림):

$$
\gamma \propto \eta^{1/3}(\Delta')
$$

**일반적인 값**:
- $\eta \sim 10^{-9}$ (토카막 조건)
- $\gamma \sim 10^2 - 10^4$ s$^{-1}$ (Alfvén보다 훨씬 느림)

### 3.5 자기 섬 폭

포화된 섬 폭은 재결합된 플럭스 $\delta\psi$와 관련됩니다:

$$
w = 4\sqrt{\frac{\delta\psi r_s}{m B_\theta(r_s)}}
$$

**섬 성장**: **Rutherford 방정식**으로 설명:

$$
\frac{d(\delta\psi)}{dt} = \eta J_s \Delta'(w) w
$$

여기서 $\Delta'(w)$는 섬 폭에 의존합니다.

**비선형 진화**:
- 작은 섬: $\Delta'(w) \approx \Delta'(0)$ → 지수 성장
- 큰 섬: $\Delta'(w)$ 감소 → 포화 또는 계속 성장

## 4. 저항 벽 모드

### 4.1 벽 안정화

플라즈마 근처의 **전도 벽**은 섭동을 상쇄하는 **이미지 전류**를 유도하여 이상 외부 kink 모드를 안정화할 수 있습니다.

**이상 벽** (완전 도체, $r = r_w$):
- 섭동이 표면 전류 유도
- 이미지 전류가 외부 장을 정확히 상쇄
- $q(a) < 1$이어도 외부 kink 안정화

**저항 벽** (유한 전도도):
- 이미지 전류가 저항 시간 척도 $\tau_w = \mu_0\sigma d r_w$에서 감쇠
- 모드가 천천히 성장: $\gamma \sim 1/\tau_w$

```
저항 벽 모드:
===================

        플라즈마
          ║
          ║  ← 섭동
          ║
     ════════  저항 벽
       ↓
    유도 전류
    (τw에서 감쇠)

γ ~ 1/τw << γ_ideal
```

### 4.2 성장률

반지름 $r_w$에서 저항 벽에 대해, 성장률:

$$
\gamma_{RWM} \approx \frac{1}{\tau_w}\frac{r_w - a}{r_w}
$$

여기서 $\tau_w = \mu_0\sigma d r_w$는 벽 시간 상수, $\sigma$는 전도도, $d$는 벽 두께입니다.

**일반적인 값**:
- $\tau_w \sim 0.01 - 1$ s (구리 벽)
- $\gamma_{RWM} \sim 1 - 100$ s$^{-1}$ (중간 시간 척도)

### 4.3 피드백 안정화

RWM이 천천히 성장하므로, 피드백 코일로 **능동적으로 안정화**될 수 있습니다:

1. 모드 감지(자기 센서)
2. 보정 장 계산
3. 섭동 상쇄를 위해 코일 구동
4. 피드백이 $\gamma_{RWM}$보다 빨라야 함

**응용**:
- 고베타 토카막 운전
- 고급 시나리오 ($q(0) < 1$)
- 벽 없는 베타 한계를 넘어서 평형 유지

## 5. 신고전 Tearing 모드 (NTM)

### 5.1 Bootstrap 전류 구동

토카막에서 **bootstrap 전류**는 충돌 운동량 보존으로 인한 압력 구배로부터 발생합니다:

$$
J_{bs} \propto \frac{dp}{dr}
$$

자기 섬이 형성될 때:
- 섬이 국소적으로 압력 프로파일 평탄화
- Bootstrap 전류가 섬 내부에서 감소
- 누락된 전류가 나선형 전류 섭동 생성
- 섭동이 추가 섬 성장 구동

**양의 피드백 루프** → **준안정 NTM**

### 5.2 수정된 Rutherford 방정식

Bootstrap 구동을 포함한 섬 진화:

$$
\tau_r\frac{dw}{dt} = r_s\Delta'(w) + r_s\frac{L_q}{L_p}\beta_p\left(\frac{w_d^2}{w^2} - \frac{w_{sat}^2}{w^2 + w_{sat}^2}\right)
$$

여기서:
- $\tau_r$: 저항 확산 시간
- $\Delta'(w)$: 안정성 지수(음수 가능)
- $L_q = q/(dq/dr)$: 전단 길이
- $L_p = p/(dp/dr)$: 압력 스케일 길이
- $\beta_p$: 폴로이달 베타
- $w_d$: 발생을 위한 임계 폭
- $w_{sat}$: 포화 폭

### 5.3 임계값과 준안정성

**주요 특징**: NTM은 성장하기 위해 **씨앗 섬**이 필요합니다.

- 작은 섭동 ($w < w_d$): 섬 감쇠
- 임계값 초과 ($w > w_d$): 섬이 $w_{sat}$로 성장

**촉발 메커니즘**:
- Sawtooth crash
- ELM
- 다른 MHD 사건

```
NTM 준안정성:
=================

dw/dt |        ╱
      |       ╱ ← 성장 영역
      |      ╱
      |_____╱_________ w
      |    ╱  ↑   ↑
      |   ╱  w_d w_sat
      |  ╱
      | ╱ ← 감쇠 영역

두 개의 안정 상태:
1. w = 0 (섬 없음)
2. w = w_sat (포화됨)
```

### 5.4 억제 기법

**NTM 억제 방법**:
1. **전자 사이클로트론 전류 구동(ECCD)**: 섬 O-점에서 전류를 구동하여 누락된 bootstrap 전류 대체
2. **베타 감소**: 압력 구배 낮춤 → 작은 bootstrap 구동
3. **회전 증가**: 속도 전단을 통해 안정화
4. **촉발 회피**: Sawtooth 진폭 감소, ELM 제어

## 6. 수치 구현

### 6.1 Kink 모드 성장률

```python
import numpy as np
import matplotlib.pyplot as plt

def kink_growth_rate(q_edge, Btheta, rho):
    """
    Compute external kink growth rate for cylindrical plasma

    Parameters:
    -----------
    q_edge: safety factor at edge
    Btheta: poloidal field at edge [T]
    rho: plasma density [kg/m^3]

    Returns:
    --------
    gamma: growth rate [1/s] (0 if stable)
    stable: True if stable
    """
    mu0 = 4*np.pi*1e-7

    if q_edge < 1:
        # Unstable
        gamma = (Btheta / np.sqrt(mu0 * rho)) * np.sqrt(1 - q_edge**2)
        stable = False
    else:
        # Stable
        gamma = 0.0
        stable = True

    return gamma, stable

def plot_kink_stability():
    """Plot kink growth rate vs q(a)"""

    # Plasma parameters
    Btheta = 0.5  # T
    rho = 1e-6    # kg/m^3
    mu0 = 4*np.pi*1e-7

    vA = Btheta / np.sqrt(mu0 * rho)  # Alfvén speed

    # Safety factor range
    q_vals = np.linspace(0.1, 2.0, 100)

    gamma_vals = []
    for q in q_vals:
        gamma, stable = kink_growth_rate(q, Btheta, rho)
        gamma_vals.append(gamma)

    gamma_vals = np.array(gamma_vals)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot growth rate
    ax.plot(q_vals, gamma_vals, 'b-', linewidth=2)

    # Mark stability boundary
    ax.axvline(x=1.0, color='r', linestyle='--', linewidth=2,
               label='안정성 경계 (q=1)')

    # Shade unstable region
    unstable_mask = q_vals < 1.0
    if np.any(unstable_mask):
        ax.fill_between(q_vals, 0, gamma_vals, where=unstable_mask,
                        alpha=0.3, color='red', label='불안정')

    # Shade stable region
    stable_mask = q_vals >= 1.0
    ax.fill_between(q_vals[stable_mask], 0, gamma_vals[stable_mask],
                    alpha=0.3, color='green', label='안정')

    ax.set_xlabel('가장자리 안전 인자 q(a)', fontsize=12)
    ax.set_ylabel('성장률 γ [1/s]', fontsize=12)
    ax.set_title(f'외부 Kink 성장률 (vₐ = {vA:.2e} m/s)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig

def example_kink_instability():
    """Analyze external kink for various configurations"""

    print("=== 외부 Kink 불안정성 분석 ===\n")

    # Plasma parameters
    a = 0.5       # Minor radius [m]
    I = 1e6       # Plasma current [A]
    n = 1e20      # Density [m^-3]
    T = 1e7       # Temperature [K]
    mu0 = 4*np.pi*1e-7
    mp = 1.67e-27

    rho = n * mp

    # Poloidal field at edge
    Btheta = mu0 * I / (2*np.pi*a)

    print(f"플라즈마 매개변수:")
    print(f"  반지름: a = {a} m")
    print(f"  전류: I = {I/1e6} MA")
    print(f"  밀도: n = {n:.2e} m^-3")
    print(f"  가장자리 폴로이달 장: Bθ(a) = {Btheta:.3f} T")

    # Test various toroidal fields (varying q)
    R0 = 3.0  # Major radius
    Bt_values = [1.0, 2.0, 3.0, 5.0]  # Toroidal field [T]

    print(f"\n다양한 토로이달 장에 대한 Kink 안정성 (R0 = {R0} m):")
    print("-" * 60)

    for Bt in Bt_values:
        # Safety factor
        q_edge = (a * Bt) / (R0 * Btheta)

        # Growth rate
        gamma, stable = kink_growth_rate(q_edge, Btheta, rho)

        # Alfvén time
        vA = Btheta / np.sqrt(mu0 * rho)
        tau_A = a / vA

        status = "안정" if stable else "불안정"

        print(f"\nBt = {Bt} T:")
        print(f"  q(a) = {q_edge:.2f}")
        print(f"  상태: {status}")

        if not stable:
            print(f"  성장률: γ = {gamma:.2e} s^-1")
            print(f"  성장 시간: τ = {1/gamma:.2e} s")
            print(f"  Alfvén 시간: τA = {tau_A:.2e} s")
            print(f"  γ/ωA = {gamma * tau_A:.2f}")

    # Plot
    fig = plot_kink_stability()
    plt.savefig('/tmp/kink_stability.png', dpi=150)
    print("\n\nKink 안정성 플롯이 /tmp/kink_stability.png에 저장되었습니다")
    plt.close()

if __name__ == "__main__":
    example_kink_instability()
```

### 6.2 Tearing 모드를 위한 Δ' 계산

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def solve_outer_region(r, rs, m, q_profile, psi_rs):
    """
    Solve ideal MHD equation in outer region

    d/dr(r d psi/dr) - m² psi/r = 0 at resonance

    Near resonance: psi ~ C+ (r-rs)  for r > rs
                          C- (r-rs)  for r < rs

    Returns derivative at rs
    """
    # Simplified: assume psi' = const near rs
    # In reality, solve ODE from boundaries

    # For a simple current profile, analytical Δ' exists
    # Here we use a model

    # Estimate from q-profile curvature
    dr = 0.001
    q_rs = q_profile(rs)
    q_p = (q_profile(rs + dr) - q_profile(rs - dr)) / (2*dr)
    q_pp = (q_profile(rs + dr) - 2*q_profile(rs) + q_profile(rs - dr)) / dr**2

    # Approximate Δ' from q''
    # Δ' ≈ (2/r_s) + (q''/q')
    Delta_prime = (2/rs) + (q_pp / q_p) if abs(q_p) > 1e-10 else 0

    return Delta_prime

def tearing_growth_rate(Delta_prime, eta, rs, Btheta):
    """
    Estimate tearing mode growth rate

    γ ~ η^(3/5) Δ'^(4/5)  (constant-psi regime)

    Parameters:
    -----------
    Delta_prime: stability index [1/m]
    eta: resistivity [Ohm*m]
    rs: resonant surface radius [m]
    Btheta: poloidal field [T]

    Returns:
    --------
    gamma: growth rate [1/s]
    """
    mu0 = 4*np.pi*1e-7

    if Delta_prime <= 0:
        return 0.0  # Stable

    # Resistive diffusion time
    tau_R = mu0 * rs**2 / eta

    # Scaling (constant-psi)
    # γ ~ (η/τA)^(3/5) Δ'^(4/5) / rs
    # Simplified estimate
    gamma = (eta / (mu0 * rs**2))**(3/5) * (Delta_prime * rs)**(4/5)

    return gamma

def plot_delta_prime():
    """Plot Δ' for various current profiles"""

    a = 0.5  # Plasma radius
    R0 = 3.0

    # Different current profiles
    profiles = {
        '포물선형': lambda r: 1.0 + 2.0*(r/a)**2,
        '뾰족함': lambda r: 1.0 + 3.0*(r/a)**4,
        '중공형': lambda r: 0.8 + 0.5*(r/a) + 2.0*(r/a)**2,
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Panel 1: q-profiles
    ax = axes[0]
    r_vals = np.linspace(0.01, a, 200)

    for name, q_func in profiles.items():
        q_vals = [q_func(r) for r in r_vals]
        ax.plot(r_vals/a, q_vals, linewidth=2, label=name)

    # Mark rational surfaces
    for q_rational in [1.5, 2.0, 2.5, 3.0]:
        ax.axhline(q_rational, color='gray', linestyle=':', alpha=0.5)
        ax.text(1.05, q_rational, f'q={q_rational}', fontsize=9)

    ax.set_xlabel('r/a', fontsize=12)
    ax.set_ylabel('q(r)', fontsize=12)
    ax.set_title('안전 인자 프로파일', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    # Panel 2: Δ' at q=2 surface
    ax = axes[1]

    m = 2
    q_target = 2.0

    Delta_primes = []
    rs_vals = []

    for name, q_func in profiles.items():
        # Find resonant surface where q = 2
        for r in r_vals:
            if abs(q_func(r) - q_target) < 0.05:
                rs = r
                break
        else:
            rs = None

        if rs is not None:
            Delta_p = solve_outer_region(r_vals, rs, m, q_func, psi_rs=1.0)
            Delta_primes.append(Delta_p)
            rs_vals.append(rs)
        else:
            Delta_primes.append(0)
            rs_vals.append(0)

    colors = ['blue', 'orange', 'green']
    x_pos = np.arange(len(profiles))

    bars = ax.bar(x_pos, Delta_primes, color=colors, alpha=0.7)

    # Mark stability boundary
    ax.axhline(0, color='red', linestyle='--', linewidth=2,
               label='Δ\'=0 (한계)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(profiles.keys())
    ax.set_ylabel('Δ\' [1/m]', fontsize=12)
    ax.set_title(f'q={q_target}에서 Tearing 안정성 지수 (m={m})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate stability
    for i, (dp, bar) in enumerate(zip(Delta_primes, bars)):
        if dp > 0:
            status = "불안정"
            color = 'red'
        else:
            status = "안정"
            color = 'green'

        ax.text(i, dp + 0.1, status, ha='center', fontsize=10,
                color=color, fontweight='bold')

    plt.tight_layout()
    return fig

def example_tearing_mode():
    """Analyze tearing mode for tokamak"""

    print("=== Tearing 모드 분석 ===\n")

    # Parameters
    a = 0.5
    R0 = 3.0
    Bt = 5.0
    Ip = 1e6
    eta = 1e-7  # Resistivity [Ohm*m]

    mu0 = 4*np.pi*1e-7

    # Poloidal field
    Btheta = mu0 * Ip / (2*np.pi*a)

    # q-profile (parabolic)
    def q_profile(r):
        q0 = 1.0
        qa = 3.0
        return q0 + (qa - q0)*(r/a)**2

    # Find q=2 surface
    m = 2
    n = 1
    q_target = m / n

    r_vals = np.linspace(0.01, a, 1000)
    for r in r_vals:
        if abs(q_profile(r) - q_target) < 0.001:
            rs = r
            break

    print(f"플라즈마 매개변수:")
    print(f"  소반지름: a = {a} m")
    print(f"  주요 반지름: R0 = {R0} m")
    print(f"  플라즈마 전류: Ip = {Ip/1e6} MA")
    print(f"  저항: η = {eta:.2e} Ω·m")

    print(f"\n공명 표면 (m={m}, n={n}):")
    print(f"  q = {q_target}")
    print(f"  위치: rs = {rs:.3f} m (rs/a = {rs/a:.2f})")

    # Compute Δ'
    Delta_p = solve_outer_region(r_vals, rs, m, q_profile, psi_rs=1.0)

    print(f"\n안정성 지수:")
    print(f"  Δ' = {Delta_p:.2f} m^-1")

    if Delta_p > 0:
        print("  상태: 불안정 (tearing 모드)")

        # Growth rate
        gamma = tearing_growth_rate(Delta_p, eta, rs, Btheta)

        print(f"\n성장률:")
        print(f"  γ = {gamma:.2e} s^-1")
        print(f"  성장 시간: τ = {1/gamma:.2e} s")

        # Compare to resistive diffusion time
        tau_R = mu0 * rs**2 / eta
        print(f"  저항 시간: τR = {tau_R:.2e} s")
        print(f"  γ τR = {gamma * tau_R:.3f}")

        # Estimate island width (nonlinear)
        # Simplified: w ~ sqrt(Δ' * rs)
        w_estimate = np.sqrt(Delta_p * rs) * 0.1  # Order of magnitude
        print(f"\n추정 포화 섬 폭: w ~ {w_estimate:.3f} m")
        print(f"  (w/rs = {w_estimate/rs:.2f})")

    else:
        print("  상태: 안정")

    # Plot
    fig = plot_delta_prime()
    plt.savefig('/tmp/tearing_delta_prime.png', dpi=150)
    print("\nΔ' 플롯이 /tmp/tearing_delta_prime.png에 저장되었습니다")
    plt.close()

if __name__ == "__main__":
    example_tearing_mode()
```

### 6.3 Rutherford 섬 진화

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rutherford_equation(t, w, Delta_prime_func, eta, Js, rs):
    """
    Rutherford equation for island width evolution

    τR dw/dt = r_s Δ'(w) w

    Parameters:
    -----------
    t: time [s]
    w: island width [m]
    Delta_prime_func: function Δ'(w)
    eta: resistivity [Ohm*m]
    Js: current density at resonant surface [A/m^2]
    rs: resonant surface radius [m]

    Returns:
    --------
    dw/dt: growth rate [m/s]
    """
    mu0 = 4*np.pi*1e-7

    # Resistive time
    tau_R = mu0 * rs**2 / eta

    # Δ'(w)
    Delta_p = Delta_prime_func(w)

    # Rutherford equation
    dwdt = (rs * Delta_p * w) / tau_R

    return dwdt

def Delta_prime_saturating(w, Delta0, w_sat):
    """
    Δ'(w) model with saturation

    Δ'(w) = Δ0 * (1 - w²/w_sat²)

    As w → w_sat, Δ' → 0 → saturation
    """
    if w >= w_sat:
        return 0.0
    else:
        return Delta0 * (1 - (w/w_sat)**2)

def plot_island_evolution():
    """Simulate and plot island evolution"""

    # Parameters
    rs = 0.3      # Resonant surface [m]
    eta = 1e-7    # Resistivity [Ohm*m]
    Js = 1e6      # Current density [A/m^2]
    Delta0 = 5.0  # Initial Δ' [1/m]
    w_sat = 0.1   # Saturation width [m]

    # Δ'(w) function
    Delta_prime_func = lambda w: Delta_prime_saturating(w, Delta0, w_sat)

    # Initial condition
    w0 = 0.001  # Initial seed island [m]

    # Time span
    mu0 = 4*np.pi*1e-7
    tau_R = mu0 * rs**2 / eta
    t_span = (0, 5*tau_R)
    t_eval = np.linspace(0, t_span[1], 500)

    # Solve
    sol = solve_ivp(
        lambda t, w: rutherford_equation(t, w, Delta_prime_func, eta, Js, rs),
        t_span,
        [w0],
        t_eval=t_eval,
        method='RK45'
    )

    t = sol.t
    w = sol.y[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Panel 1: Island width vs time
    ax = axes[0]
    ax.plot(t/tau_R, w*100, 'b-', linewidth=2)
    ax.axhline(w_sat*100, color='r', linestyle='--', linewidth=2,
               label=f'포화 (w_sat = {w_sat*100} cm)')

    ax.set_xlabel('시간 (τR)', fontsize=12)
    ax.set_ylabel('섬 폭 [cm]', fontsize=12)
    ax.set_title('자기 섬 성장 (Rutherford 방정식)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Growth rate vs width
    ax = axes[1]

    # Compute dw/dt
    dwdt = np.array([rutherford_equation(ti, wi, Delta_prime_func, eta, Js, rs)
                     for ti, wi in zip(t, w)])

    ax.plot(w*100, dwdt*100, 'g-', linewidth=2)
    ax.axvline(w_sat*100, color='r', linestyle='--', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', linewidth=1)

    ax.set_xlabel('섬 폭 [cm]', fontsize=12)
    ax.set_ylabel('성장률 dw/dt [cm/s]', fontsize=12)
    ax.set_title('섬 폭 대 성장률', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def example_island_evolution():
    """Simulate magnetic island evolution"""

    print("=== 자기 섬 진화 (Rutherford 방정식) ===\n")

    # Parameters
    rs = 0.3
    eta = 1e-7
    Delta0 = 5.0
    w_sat = 0.1

    mu0 = 4*np.pi*1e-7
    tau_R = mu0 * rs**2 / eta

    print(f"매개변수:")
    print(f"  공명 표면: rs = {rs} m")
    print(f"  저항: η = {eta:.2e} Ω·m")
    print(f"  초기 Δ': Δ0 = {Delta0} m^-1")
    print(f"  포화 폭: w_sat = {w_sat*100} cm")
    print(f"  저항 시간: τR = {tau_R:.2f} s")

    # Initial island
    w0 = 0.001

    print(f"\n초기 씨앗 섬: w0 = {w0*100} cm")

    # Time to reach half saturation
    # Approximate: w(t) ~ w_sat tanh(t/τ_growth)
    # τ_growth ~ τR / (rs Δ0)
    tau_growth = tau_R / (rs * Delta0)

    print(f"성장 시간 척도: τ_growth ~ {tau_growth:.2f} s")
    print(f"  (τ_growth / τR = {tau_growth/tau_R:.3f})")

    # Plot
    fig = plot_island_evolution()
    plt.savefig('/tmp/island_evolution.png', dpi=150)
    print("\n섬 진화 플롯이 /tmp/island_evolution.png에 저장되었습니다")
    plt.close()

if __name__ == "__main__":
    example_island_evolution()
```

## 7. 실험 관측

### 7.1 토카막의 Sawtooth

**Sawtooth 진동**은 내부 kink (m=1, n=1)에 의한 중심 온도의 주기적 붕괴입니다:

- **상승 단계**: 중심 가열 → $T_0$ 상승, $q(0)$ 하락
- **Crash**: $q(0) < 1$일 때, 내부 kink 촉발 → 빠른 재결합
- **평탄화**: 온도와 전류 프로파일 평탄화 → $q(0)$이 1 위로 상승
- **반복**: 주기는 일반적으로 10-100 ms

**영향**:
- 유익함: 불순물 축적 방지, 중심 압력 제한
- 해로움: NTM 촉발, 핵융합 성능 제한

### 7.2 Disruption

**주요 disruption**은 대규모 MHD 불안정성에 의한 재앙적 사건입니다:

**열 소멸** (ms 시간 척도):
- 가둠 손실
- 온도 붕괴
- Runaway 전자 생성

**전류 소멸** (ms 시간 척도):
- 플라즈마 전류 감쇠
- 구조물에서 큰 유도 전압
- 용기에 기계적 힘

**완화 전략** (ITER):
- 대량 가스 주입
- 분쇄된 펠렛 주입
- Disruption 예측 및 회피

### 7.3 신고전 Tearing 모드

고베타 토카막 플라즈마에서 관측됨:
- Sawtooth 또는 ELM에 의해 촉발
- 큰 섬으로 성장(소반지름의 10-20%)
- 가둠 저하
- Disruption으로 이어질 수 있음

**제어 방법**:
- 섬 O-점에서 ECCD
- 회전 제어
- 베타 감소

## 요약

이 강의에서 전류 구동 MHD 불안정성을 연구했습니다:

1. **Kink 불안정성**: 외부 kink ($q(a) < 1$)가 전체 기둥 굽힘; 내부 kink ($q(0) < 1$)가 sawtooth 유발. 충분한 축방향 장($q > 1$)으로 안정화.

2. **Sausage 모드**: m=0 조임/팽창, $B_z$ 없이 항상 불안정. $B_z > B_\theta$일 때 안정화.

3. **Tearing 모드**: 유리수 표면($q = m/n$)에서 저항 재결합. 안정성은 Δ' 매개변수로 결정. 성장률 $\gamma \propto \eta^{3/5}(\Delta')^{4/5}$.

4. **자기 섬**: Tearing 모드로 형성. 폭 진화는 Rutherford 방정식으로 지배. Δ'(w) → 0일 때 포화.

5. **저항 벽 모드**: 저항 벽 시간 척도에서 느리게 성장하는 kink 모드. 피드백 제어로 안정화 가능.

6. **신고전 Tearing 모드**: 누락된 bootstrap 전류에 의해 구동되는 준안정 섬. 성장하기 위해 씨앗 섬 필요. 고베타 운전의 주요 우려사항.

7. **수치 도구**: 성장률 계산, Δ' 추정, 섬 진화 시뮬레이션.

이러한 불안정성은 토카막 성능을 근본적으로 제한하고 ITER 및 미래 원자로를 위한 고급 제어 전략, 프로파일 최적화, disruption 완화 시스템을 동기화합니다.

## 연습 문제

### 문제 1: 외부 Kink 안정성

원통형 Z-pinch가 다음을 가집니다:
- 반지름 $a = 0.1$ m
- 전류 $I = 500$ kA
- 밀도 $\rho = 10^{-6}$ kg/m³
- 처음에 축방향 장 없음

**(a)** 가장자리의 방위각 장 $B_\theta(a)$를 계산하세요.

**(b)** Alfvén 속도 $v_A = B_\theta/\sqrt{\mu_0\rho}$를 계산하세요.

**(c)** $B_z = 0$이므로, $q(a)$는 무엇입니까?

**(d)** 구성이 외부 kink에 대해 안정한가요, 불안정한가요?

**(e)** 불안정하면, 성장률 $\gamma$를 계산하세요.

**(f)** Kink를 안정화하는 데 필요한 최소 $B_z$는 무엇입니까 ($q(a) > 1$)?

### 문제 2: Sawtooth 주기

토카막이 다음을 가집니다:
- 중심 온도 $T_0(t=0) = 5$ keV
- 가열 출력 $P = 10$ MW
- 중심 부피 $V_c \approx 1$ m³
- 입자 밀도 $n = 10^{20}$ m$^{-3}$

중심 온도가 $q(0) = 0.95$가 sawtooth crash를 촉발할 때까지 선형적으로 상승한다고 가정합니다.

**(a)** 가열률 $dT_0/dt = P/(3nV_c k_B)$를 추정하세요.

**(b)** Sawtooth가 $T_0 = 6$ keV일 때 붕괴하면, crash까지의 시간을 계산하세요.

**(c)** Crash 후, $T_0$이 4 keV로 떨어집니다. Sawtooth 주기는 무엇입니까?

**(d)** Crash당 재분배되는 에너지는 얼마입니까?

**(e)** Crash 시간 척도(~ Alfvén 시간 $\tau_A \sim$ 1 μs)를 상승 시간 척도와 비교하세요.

### 문제 3: Tearing 모드 Δ'

전류 밀도가 다음인 원통형 플라즈마에 대해:

$$
J_z(r) = J_0\left(1 - \frac{r^2}{a^2}\right)
$$

안전 인자는:

$$
q(r) = q_0\frac{1 + (r/a)^2}{1 - (r^2/a^2)} \quad (q_0 = q(0))
$$

**(a)** $q(r_s) = 2$인 위치 $r_s$를 찾으세요 ($q_0 = 1$이라고 가정).

**(b)** $q'(r_s)$와 $q''(r_s)$를 수치적 또는 해석적으로 계산하세요.

**(c)** $r_s$에서 $\Delta' \approx \frac{2}{r_s} + \frac{q''}{q'}$를 추정하세요.

**(d)** $m=2, n=1$ 모드가 안정한가요, 불안정한가요?

**(e)** $\eta = 10^{-7}$ Ω·m이고 $a = 0.5$ m이면, 성장률을 추정하세요.

### 문제 4: 자기 섬 폭

Tearing 모드가 공명 표면 $r_s = 0.3$ m에서 재결합된 플럭스 $\delta\psi = 10^{-3}$ Wb를 가집니다.

$r_s$에서 평형 폴로이달 장은 $B_\theta(r_s) = 0.4$ T이고, 모드 수는 $m=2$입니다.

**(a)** 섬 폭을 계산하세요:
$$
w = 4\sqrt{\frac{\delta\psi r_s}{m B_\theta(r_s)}}
$$

**(b)** 섬이 $w = 5$ cm로 성장하면, 새로운 $\delta\psi$는 무엇입니까?

**(c)** 압력 프로파일의 평탄화 영역을 추정하세요 ($r_s$ 주위로 대략 $\pm w/2$).

**(d)** $dp/dr = -10^6$ Pa/m이면, 평탄화로 인해 손실되는 압력 구배는 얼마입니까?

**(e)** Bootstrap 전류와 NTM 구동에 미치는 영향을 논의하세요.

### 문제 5: 저항 벽 모드

토카막이 다음을 가집니다:
- 플라즈마 소반지름 $a = 1$ m
- $r_w = 1.2$ m에서 저항 벽
- 벽 두께 $d = 5$ cm
- 벽 전도도 $\sigma = 5\times 10^7$ S/m (구리)

**(a)** 벽 시간 상수 $\tau_w = \mu_0\sigma d r_w$를 계산하세요.

**(b)** RWM 성장률을 추정하세요:
$$
\gamma_{RWM} \approx \frac{1}{\tau_w}\frac{r_w - a}{r_w}
$$

**(c)** 피드백 제어가 응답 시간 $\tau_{fb} = 10$ ms를 가지면, RWM을 안정화할 수 있습니까?

**(d)** 피드백 시스템에 필요한 대역폭은 무엇입니까?

**(e)** 벽이 이상적(완전 도체)이면, 외부 kink에 무슨 일이 일어납니까?

---

**이전**: [Pressure-Driven Instabilities](./03_Pressure_Driven_Instabilities.md) | **다음**: [Reconnection Theory](./05_Reconnection_Theory.md)
