# 3. 압력 구동 불안정성

## 학습 목표

- 불리한 곡률에 의해 구동되는 교환 불안정성의 물리적 메커니즘 이해
- 자화 플라즈마에서의 Rayleigh-Taylor 불안정성 분석
- 층화된 대기에서의 Parker 불안정성 연구
- 토로이달 기하학에서 높은-n 모드에 대한 ballooning 모드 이론 유도 및 적용
- 국소 교환 안정성에 대한 Mercier 기준 사용
- 압력 구동 불안정성의 수치 시뮬레이션 구현
- 실험 관측(토카막의 ELM)과의 연결 이해

## 1. 압력 구동 불안정성 소개

압력 구동 불안정성은 **압력 구배**가 자기장선 굽힘에 대항하여 유체 운동을 구동할 수 있는 자유 에너지를 제공할 때 발생합니다. 이러한 불안정성은 특히 다음에서 중요합니다:

- **핵융합 플라즈마**: 달성 가능한 압력 제한(베타 한계)
- **천체물리학 플라즈마**: 태양 홍염, 코로나 질량 방출
- **행성 자기권**: 자기꼬리의 자기장 구성

기본 물리학은 다음 사이의 경쟁입니다:
- **불안정화**: 불리한 곡률에서의 압력 구배
- **안정화**: 자기장선 굽힘(장력)

```
압력 구동 불안정성 메커니즘:
=====================================

유리한 곡률                 불리한 곡률
(자기 우물):                (자기 언덕):

    ∇p                            ∇p
     ↓                             ↑
  ═══════  B                   ═══════  B
 (       )                      ‾‾‾‾‾‾‾
  ‾‾‾‾‾‾‾                      (       )

압력이 장 굽힘을              압력이 곡률과
거슬러 밀어냄                 같은 방향으로 밀어냄
→ 안정                       → 불안정
```

## 2. 교환 불안정성

### 2.1 물리적 그림

**교환 불안정성**은 인접한 플럭스 튜브가 위치를 교환할 때 발생합니다. 이것은 유체역학의 Rayleigh-Taylor 불안정성과 유사하지만 중력 대신 자기장이 사용됩니다.

**에너지 고려**:

다음을 가진 다른 반지름의 두 플럭스 튜브를 고려하세요:
- $r_1$에서 튜브 1: 압력 $p_1$, 자기장 $B_1$
- $r_2 > r_1$에서 튜브 2: 압력 $p_2 < p_1$, 자기장 $B_2$

이들을 교환하면 위치 에너지의 변화는:

$$
\delta W \propto (p_1 - p_2)(B_2^2 - B_1^2)
$$

**불안정 조건**: $B$가 $p$보다 빠르게 바깥쪽으로 감소하면, $\delta W < 0$ → 불안정.

### 2.2 곡률과 압력 구배

안정성은 다음의 부호에 달려 있습니다:

$$
\kappa \cdot \nabla p
$$

여기서 $\boldsymbol{\kappa} = \mathbf{b}\cdot\nabla\mathbf{b}$는 장선 곡률이고, $\mathbf{b} = \mathbf{B}/B$입니다.

**유리한 곡률** ($\boldsymbol{\kappa} \cdot \nabla p < 0$):
- 곡률이 높은 압력에서 멀어지는 방향
- 중력에서 가벼운 유체 위의 무거운 유체와 같음(안정)

**불리한 곡률** ($\boldsymbol{\kappa} \cdot \nabla p > 0$):
- 곡률이 높은 압력을 향하는 방향
- 중력에서 무거운 유체 아래의 가벼운 유체와 같음(불안정)

### 2.3 교환 조건

에너지 원리로부터, 교환 불안정성은 다음을 요구합니다:

$$
\int \left[\frac{|\mathbf{B}_1|^2}{\mu_0} - 2(\boldsymbol{\xi}\cdot\boldsymbol{\kappa})(\boldsymbol{\xi}\cdot\nabla p)\right]dV < 0
$$

$\mathbf{B}$에 수직인 비압축성 섭동에 대해:

$$
\delta W \approx -\int (\boldsymbol{\xi}_\perp\cdot\boldsymbol{\kappa})(\boldsymbol{\xi}_\perp\cdot\nabla p)\, dV
$$

$\boldsymbol{\kappa}\cdot\nabla p > 0$이면, $\boldsymbol{\kappa}$와 $\nabla p$ 모두에 평행한 $\boldsymbol{\xi}_\perp$를 선택하여 $\delta W < 0$으로 만들 수 있습니다.

### 2.4 토카막의 좋은 곡률 대 나쁜 곡률

토카막에서 자기장은 토로이달과 폴로이달 성분을 가집니다. 폴로이달 방향으로 돌아갈 때:

**외부 측면** (저자기장 측면, 큰 $R$):
- 장선이 안쪽으로 굽음(플라즈마 쪽으로)
- $\boldsymbol{\kappa}$가 안쪽을 가리킴
- $\nabla p$가 바깥쪽을 가리킴
- $\boldsymbol{\kappa}\cdot\nabla p < 0$ → **유리함** ("좋은 곡률")

**내부 측면** (고자기장 측면, 작은 $R$):
- 장선이 바깥쪽으로 굽음(플라즈마로부터 멀어짐)
- $\boldsymbol{\kappa}$가 바깥쪽을 가리킴
- $\nabla p$가 바깥쪽을 가리킴
- $\boldsymbol{\kappa}\cdot\nabla p > 0$ → **불리함** ("나쁜 곡률")

```
토카막 단면:
=====================

        나쁜 곡률
             ↓
        ═══════════
       ║           ║
       ║  플라즈마  ║  ← 좋은 곡률
       ║           ║
        ═══════════

불안정성은 나쁜 곡률
(내부) 측면에 국소화되는 경향
```

### 2.5 평균 곡률과 안정성

닫힌 장선에 대해, **평균 곡률**이 안정성을 결정합니다:

$$
\bar{\kappa} = \frac{1}{L}\oint \boldsymbol{\kappa}\cdot d\mathbf{l}
$$

$\bar{\kappa}\cdot\nabla p > 0$이면: 잠재적으로 불안정(상세 계산 필요).

**자기 우물**: $\bar{\kappa}\cdot\nabla p < 0$인 모든 곳에서의 구성 → 안정.

## 3. MHD의 Rayleigh-Taylor 불안정성

### 3.1 유체역학적 Rayleigh-Taylor

유체역학에서, 중력장에서 가벼운 유체 위의 무거운 유체는 불안정합니다.

**분산 관계**(자기장 없음):

$$
\omega^2 = -gk\frac{\rho_2 - \rho_1}{\rho_2 + \rho_1}
$$

$\rho_2 > \rho_1$(위에 무거운 것): $\omega^2 < 0$ → 불안정.

성장률: $\gamma = \sqrt{gk}$ (점성도와 무관).

### 3.2 횡방향 장을 가진 MHD Rayleigh-Taylor

중력 $\mathbf{g} = -g\hat{\mathbf{z}}$에 수직인 수평 자기장 $\mathbf{B}_0 = B_0\hat{\mathbf{x}}$를 추가합니다.

**수정된 분산 관계**:

$$
\omega^2 = -gk\frac{\rho_2 - \rho_1}{\rho_2 + \rho_1} + \frac{B_0^2}{\mu_0(\rho_1 + \rho_2)}k_x^2
$$

여기서 $k_x$는 장을 따른 파수입니다.

**안정성 분석**:

- $\mathbf{k} \parallel \mathbf{B}$인 섭동 ($k_x = k$): 다음이면 안정화
  $$
  \frac{B_0^2}{\mu_0} > g(\rho_2 - \rho_1)/k
  $$

- $\mathbf{k} \perp \mathbf{B}$인 섭동 ($k_x = 0$): **항상 불안정** (유체역학과 동일).

안정화를 위한 **임계 파수**:

$$
k_c = \frac{g(\rho_2 - \rho_1)\mu_0}{B_0^2}
$$

짧은 파장 ($k > k_c$)은 안정; 긴 파장은 불안정.

### 3.3 성장률

불안정 모드 ($k < k_c$)에 대해:

$$
\gamma = \sqrt{gk\frac{\rho_2-\rho_1}{\rho_2+\rho_1} - \frac{B_0^2}{\mu_0(\rho_1+\rho_2)}k_x^2}
$$

최대 성장률 ($k_x = 0$에서):

$$
\gamma_{max} = \sqrt{gk\frac{\rho_2-\rho_1}{\rho_2+\rho_1}}
$$

### 3.4 자기 곡률과의 유사성

중력 가속도 $\mathbf{g}$는 자기 곡률로부터의 유효 중력으로 대체될 수 있습니다:

$$
\mathbf{g}_{eff} = \frac{B^2}{\mu_0\rho}\boldsymbol{\kappa}
$$

이것은 RT 불안정성을 교환 불안정성에 연결합니다.

## 4. Parker 불안정성

### 4.1 자기 부력

**Parker 불안정성**(Parker, 1966)은 수평 자기장을 가진 층화된 대기에서 발생합니다. 이것은 **자기 부력**에 의해 구동됩니다: 플라즈마의 무게가 굽은 장선을 따라 미끄러집니다.

**구성**:
- 층화된 대기: $\rho(z)$, $p(z)$
- 수평 장: $\mathbf{B}_0 = B_0(z)\hat{\mathbf{x}}$
- 중력: $\mathbf{g} = -g\hat{\mathbf{z}}$

### 4.2 물리적 메커니즘

장선이 위쪽으로 굽을 때:
1. 플라즈마가 장선을 따라 아래로 미끄러짐(중력 성분이 $\mathbf{B}$를 따름)
2. 꼭지점의 밀도 감소
3. 자기 압력이 꼭지점을 더 위로 밀어냄
4. **폭주 불안정성**

```
Parker 불안정성:
==================

초기:          섭동:
B ────────   B ╱‾‾‾‾╲
             플라즈마 미끄러짐
   ρgh         ╲    ╱
                ╲  ╱
                 ▼▼
             꼭지점의 질량 감소
             → 자기 부력
             → 더 상승
```

### 4.3 분산 관계

스케일 높이 $H = kT/(mg)$를 가진 등온 대기에 대해:

$$
\omega^2 = -\frac{g}{H}\left(1 - \frac{B_0^2}{B_0^2 + \mu_0\rho_0 c_s^2}\right)
$$

여기서 $c_s = \sqrt{\gamma p/\rho}$는 음속입니다.

**불안정 조건**:

$$
\beta = \frac{2\mu_0 p}{B^2} > \frac{2}{\gamma} \approx 1.2 \quad (\gamma=5/3\text{일 때})
$$

플라즈마 베타가 너무 높으면, 자기장이 플라즈마를 지지할 수 없음 → Parker 불안정.

### 4.4 응용

- **성간 매질**: 분자 구름 형성
- **태양 대기**: 홍염 분출
- **은하 역학**: 원반 은하의 수직 구조

## 5. Ballooning 모드

### 5.1 토로이달 기하학의 높은-n 불안정성

**Ballooning 모드**는 토러스의 나쁜 곡률 측면에 국소화되는 높은-$n$(큰 토로이달 모드 수) 압력 구동 불안정성입니다.

**특징**:
- 큰 $n$ → 짧은 수직 파장
- 외부 측면에 국소화(불리한 곡률)
- 압력 구배에 의해 구동
- 자기 전단과 유리한 평균 곡률에 의해 안정화

### 5.2 물리적 그림

모드가 나쁜 곡률 측면에서 "풍선처럼 부풀어", 장이 가장 약한 곳에서 확장하는 풍선과 같습니다.

```
토카막의 Ballooning 모드:
===========================

   평면도:

        n=10 섭동
         ║ ║ ║ ║ ║
    ════╬═╬═╬═╬═╬════  외부 (나쁜 곡률)
         ║ ║ ║ ║ ║

         (국소화됨)

    ═══════════════  내부 (좋은 곡률)
         (약함)
```

### 5.3 장 정렬 좌표계

Ballooning 모드를 분석하기 위해, **장 정렬 좌표계** $(\psi, \theta, \phi)$를 사용합니다. 여기서:
- $\psi$: 플럭스 표면 레이블
- $\theta$: 폴로이달 각(장을 따라)
- $\phi$: 토로이달 각

자기장: $\mathbf{B} = \nabla\phi\times\nabla\psi + q(\psi)\nabla\psi\times\nabla\theta$.

### 5.4 Ballooning 방정식

$n \to \infty$ 극한에서 ballooning 모드 고유값 방정식:

$$
\frac{d}{d\theta}\left[g(\theta)\frac{d\hat{\Phi}}{d\theta}\right] + h(\theta)\hat{\Phi} = \lambda\hat{\Phi}
$$

여기서:
- $g(\theta) = |\nabla\psi|^2/B^2$: 계량 계수
- $h(\theta)$: 압력 구배와 곡률 포함
- $\lambda$: 고유값(성장률과 관련)
- $\hat{\Phi}$: ballooning 진폭

**경계 조건**: $\hat{\Phi}(\theta \pm \infty) = 0$ (국소화).

### 5.5 s-α 다이어그램

안정성은 종종 **s-α 다이어그램**으로 표현됩니다:

- **자기 전단**: $s = (r/q)(dq/dr)$
- **압력 구배**: $\alpha = -(2\mu_0 R_0^2 q^2/B_0^2)(dp/dr)$

```
s-α 안정성 다이어그램:
=====================

α |       불안정
  |      /
  |     /
  |    /
  |   /  ← 안정성 경계
  |  /
  | /____안정____
  |_________________ s
  0

높은 전단 (큰 s): 안정화
높은 압력 구배 (큰 α): 불안정화
```

**근사 안정성 경계**:

$$
\alpha_c \approx 0.6s
$$

### 5.6 ELM과의 연결

토카막에서 ballooning 모드는 **Edge Localized Modes (ELM)**를 촉발한다고 믿어집니다:

- H-모드에서 높은 가장자리 압력 구배
- Ballooning 안정성 경계 초과
- 주기적 분출(ELM)이 압력 구배 완화
- ITER 우려: 큰 ELM은 첫 벽 손상 가능

**ELM 완화 전략**:
- 공명 자기 섭동(RMP)
- 펠렛 주입
- 수직 킥

## 6. Mercier 기준

### 6.1 국소 교환 안정성

**Mercier 기준**(Mercier, 1960)은 토로이달 기하학에서 국소 교환 안정성에 대한 **필요 조건**을 제공합니다.

**기준**:

$$
D_I = D_S + D_W + D_G > \frac{1}{4}
$$

여기서:
- $D_S$: 자기 전단 기여
- $D_W$: 자기 우물 기여
- $D_G$: 측지선 곡률 기여

### 6.2 명시적 형태

큰 종횡비 토카막에 대해:

$$
D_S = \frac{1}{4}\left(\frac{r}{q}\frac{dq}{dr}\right)^2
$$

$$
D_W = \frac{\mu_0 r}{B_p^2}\frac{dp}{dr}\left(1 + 2q^2\right)
$$

$$
D_G \approx \frac{r^2}{R_0 q^2}
$$

**안정성**: $D_I > 1/4$.

### 6.3 물리적 해석

- **$D_S$**: 전단이 플럭스 표면을 분리하여 안정화
- **$D_W$**: 압력 구배가 안정화하면 음수(자기 우물)
- **$D_G$**: 측지선 곡률 효과(일반적으로 안정화)

### 6.4 Suydam 기준과의 관계

원통 기하학에서 Mercier 기준은 **Suydam 기준**(레슨 2)으로 축약됩니다:

$$
\frac{r}{4}\left(\frac{q'}{q}\right)^2 + \frac{2\mu_0 p'}{B_z^2} > 0
$$

### 6.5 한계

Suydam과 마찬가지로, Mercier 기준은 **필요**하지만 **충분**하지 않습니다. 국소 교환만 확인합니다; 전역 모드는 완전한 안정성 분석이 필요합니다.

## 7. 수치 시뮬레이션

### 7.1 Rayleigh-Taylor 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt

def rayleigh_taylor_growth_rate(k, g, rho1, rho2, B0, kx_frac=0):
    """
    Compute growth rate for MHD Rayleigh-Taylor instability

    Parameters:
    -----------
    k: total wavenumber [1/m]
    g: gravitational acceleration [m/s^2]
    rho1: lower fluid density [kg/m^3]
    rho2: upper fluid density [kg/m^3]
    B0: horizontal magnetic field [T]
    kx_frac: fraction of k along B direction (0 to 1)

    Returns:
    --------
    gamma: growth rate [1/s] (or 0 if stable)
    """
    mu0 = 4*np.pi*1e-7

    # Wavenumber along field
    kx = kx_frac * k

    # Atwood number
    A = (rho2 - rho1) / (rho2 + rho1)

    # Alfvén speed squared
    vA2 = B0**2 / (mu0 * (rho1 + rho2))

    # Dispersion relation: ω² = -gkA + vA²kx²
    omega_sq = -g * k * A + vA2 * kx**2

    if omega_sq < 0:
        gamma = np.sqrt(-omega_sq)
    else:
        gamma = 0.0  # Stable (oscillatory)

    return gamma

def plot_rt_growth_rate():
    """Plot RT growth rate vs wavenumber and field strength"""

    # Physical parameters
    g = 10  # m/s^2
    rho1 = 1.0  # kg/m^3 (light fluid)
    rho2 = 2.0  # kg/m^3 (heavy fluid)

    # Wavenumber range
    k_vals = np.logspace(-2, 2, 100)  # [1/m]

    # Magnetic field strengths
    B_vals = [0, 0.1, 0.5, 1.0]  # [T]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Growth rate vs k for different B (k perpendicular to B)
    ax = axes[0]
    for B0 in B_vals:
        gamma_vals = [rayleigh_taylor_growth_rate(k, g, rho1, rho2, B0, kx_frac=0)
                      for k in k_vals]
        ax.loglog(k_vals, gamma_vals, linewidth=2, label=f'B = {B0} T')

    ax.set_xlabel('Wavenumber k [1/m]', fontsize=12)
    ax.set_ylabel('Growth rate γ [1/s]', fontsize=12)
    ax.set_title('RT Growth Rate vs Wavenumber (k ⊥ B)', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    # Panel 2: Growth rate vs angle between k and B
    ax = axes[1]
    k_fixed = 1.0  # Fixed wavenumber
    kx_frac_vals = np.linspace(0, 1, 100)

    for B0 in [0.5, 1.0, 2.0]:
        gamma_vals = [rayleigh_taylor_growth_rate(k_fixed, g, rho1, rho2, B0, kx_frac)
                      for kx_frac in kx_frac_vals]
        ax.plot(kx_frac_vals, gamma_vals, linewidth=2, label=f'B = {B0} T')

    ax.set_xlabel('k_x / k (alignment with B)', fontsize=12)
    ax.set_ylabel('Growth rate γ [1/s]', fontsize=12)
    ax.set_title(f'RT Growth Rate vs Field Alignment (k = {k_fixed} m⁻¹)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig

def example_rt_instability():
    """Example: Rayleigh-Taylor instability analysis"""

    print("=== MHD Rayleigh-Taylor 불안정성 ===\n")

    # Parameters
    g = 10.0
    rho1 = 1.0
    rho2 = 2.0
    k = 1.0

    print(f"위의 무거운 유체: ρ₂ = {rho2} kg/m³")
    print(f"아래의 가벼운 유체:  ρ₁ = {rho1} kg/m³")
    print(f"중력: g = {g} m/s²")
    print(f"파수: k = {k} m⁻¹")

    # No magnetic field
    gamma_0 = rayleigh_taylor_growth_rate(k, g, rho1, rho2, B0=0, kx_frac=0)
    print(f"\n--- 자기장 없음 ---")
    print(f"성장률: γ = {gamma_0:.3f} s⁻¹")
    print(f"성장 시간: τ = {1/gamma_0:.3f} s")

    # With transverse magnetic field (k perpendicular to B)
    B0 = 1.0
    gamma_perp = rayleigh_taylor_growth_rate(k, g, rho1, rho2, B0, kx_frac=0)
    print(f"\n--- 자기장 B = {B0} T (k ⊥ B) ---")
    print(f"성장률: γ = {gamma_perp:.3f} s⁻¹")
    print(f"여전히 불안정!")

    # With field-aligned perturbation
    gamma_par = rayleigh_taylor_growth_rate(k, g, rho1, rho2, B0, kx_frac=1.0)
    print(f"\n--- 자기장 B = {B0} T (k ∥ B) ---")
    if gamma_par > 0:
        print(f"성장률: γ = {gamma_par:.3f} s⁻¹")
    else:
        print("안정 (γ = 0)")

    # Critical wavenumber
    mu0 = 4*np.pi*1e-7
    k_c = g * (rho2 - rho1) * mu0 / B0**2
    print(f"\n임계 파수: k_c = {k_c:.3f} m⁻¹")
    print(f"k > k_c인 모드는 안정 (k ∥ B일 때)")

    # Plot
    fig = plot_rt_growth_rate()
    plt.savefig('/tmp/rt_growth_rate.png', dpi=150)
    print("\n성장률 플롯이 /tmp/rt_growth_rate.png에 저장되었습니다")
    plt.close()

if __name__ == "__main__":
    example_rt_instability()
```

### 7.2 Ballooning 안정성 분석

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def ballooning_stability_boundary(s_vals):
    """
    Approximate ballooning stability boundary in s-α space

    Parameters:
    -----------
    s_vals: array of magnetic shear values

    Returns:
    --------
    alpha_crit: critical alpha for marginal stability
    """
    # Empirical fit: α_crit ≈ 0.6 * s
    alpha_crit = 0.6 * s_vals

    return alpha_crit

def compute_alpha_parameter(r, p, q, B0, R0):
    """
    Compute pressure gradient parameter α

    α = -(2μ₀R₀²q²/B₀²)(dp/dr)

    Parameters:
    -----------
    r: minor radius [m]
    p: pressure [Pa]
    q: safety factor
    B0: toroidal field [T]
    R0: major radius [m]

    Returns:
    --------
    alpha: normalized pressure gradient
    """
    mu0 = 4*np.pi*1e-7

    # Numerical derivative
    dr = 0.001
    dpdx = (p(r + dr) - p(r - dr)) / (2*dr)

    alpha = -(2*mu0*R0**2*q**2/B0**2) * dpdx

    return alpha

def plot_s_alpha_diagram():
    """Plot s-α stability diagram"""

    # Shear range
    s_vals = np.linspace(0, 5, 100)

    # Stability boundary
    alpha_crit = ballooning_stability_boundary(s_vals)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Fill stable and unstable regions
    ax.fill_between(s_vals, 0, alpha_crit, alpha=0.3, color='green', label='안정')
    ax.fill_between(s_vals, alpha_crit, 10, alpha=0.3, color='red', label='불안정')

    # Boundary
    ax.plot(s_vals, alpha_crit, 'b-', linewidth=3, label='한계 안정성')

    # Example operating points
    examples = [
        (1.0, 0.3, '표준 H-모드', 'blue'),
        (2.0, 1.5, 'ELMy H-모드', 'red'),
        (3.0, 1.0, '개선된 가둠', 'orange'),
    ]

    for s, alpha, label, color in examples:
        stable = alpha < ballooning_stability_boundary(np.array([s]))[0]
        marker = 'o' if stable else 'x'
        markersize = 10

        ax.plot(s, alpha, marker, color=color, markersize=markersize,
                label=label, markeredgewidth=2)

    ax.set_xlabel('자기 전단 s = (r/q)(dq/dr)', fontsize=14)
    ax.set_ylabel('압력 매개변수 α', fontsize=14)
    ax.set_title('Ballooning 안정성 다이어그램 (s-α)', fontsize=16)
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 3])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    return fig

def analyze_tokamak_ballooning():
    """Analyze ballooning stability for a tokamak equilibrium"""

    # Tokamak parameters
    R0 = 3.0  # Major radius [m]
    a = 1.0   # Minor radius [m]
    B0 = 5.0  # Toroidal field [T]

    # Profiles
    def p_profile(r):
        p0 = 5e5  # Central pressure [Pa]
        return p0 * (1 - (r/a)**2)**2

    def q_profile(r):
        q0 = 1.0
        qa = 4.0
        return q0 + (qa - q0) * (r/a)**2

    # Compute s and α profiles
    r_vals = np.linspace(0.1*a, 0.9*a, 50)

    s_vals = []
    alpha_vals = []

    for r in r_vals:
        q = q_profile(r)

        # Magnetic shear
        dr = 0.001
        dqdx = (q_profile(r + dr) - q_profile(r - dr)) / (2*dr)
        s = (r / q) * dqdx

        # Alpha parameter
        alpha = compute_alpha_parameter(r, p_profile, q, B0, R0)

        s_vals.append(s)
        alpha_vals.append(alpha)

    s_vals = np.array(s_vals)
    alpha_vals = np.array(alpha_vals)

    # Check stability
    alpha_crit_vals = ballooning_stability_boundary(s_vals)
    stable = alpha_vals < alpha_crit_vals

    print("=== 토카막 Ballooning 안정성 분석 ===\n")
    print(f"주요 반지름: R0 = {R0} m")
    print(f"소반지름: a = {a} m")
    print(f"토로이달 장: B0 = {B0} T")

    # Find most unstable location
    margin = alpha_vals - alpha_crit_vals
    idx_worst = np.argmax(margin)

    print(f"\n가장 불안정한 위치:")
    print(f"  r/a = {r_vals[idx_worst]/a:.2f}")
    print(f"  s = {s_vals[idx_worst]:.2f}")
    print(f"  α = {alpha_vals[idx_worst]:.2f}")
    print(f"  α_crit = {alpha_crit_vals[idx_worst]:.2f}")
    print(f"  여유: {margin[idx_worst]:+.2f}")

    if margin[idx_worst] > 0:
        print("  상태: ballooning 모드에 불안정")
    else:
        print("  상태: 안정")

    # Plot profiles
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Pressure
    ax = axes[0, 0]
    r_plot = np.linspace(0, a, 100)
    p_plot = [p_profile(ri) for ri in r_plot]
    ax.plot(r_plot/a, np.array(p_plot)/1e3, 'b-', linewidth=2)
    ax.set_xlabel('r/a')
    ax.set_ylabel('압력 [kPa]')
    ax.set_title('압력 프로파일')
    ax.grid(True, alpha=0.3)

    # Safety factor
    ax = axes[0, 1]
    q_plot = [q_profile(ri) for ri in r_plot]
    ax.plot(r_plot/a, q_plot, 'g-', linewidth=2)
    ax.set_xlabel('r/a')
    ax.set_ylabel('q')
    ax.set_title('안전 인자 프로파일')
    ax.grid(True, alpha=0.3)

    # s-α trajectory
    ax = axes[1, 0]
    s_boundary = np.linspace(0, max(s_vals)*1.2, 100)
    alpha_boundary = ballooning_stability_boundary(s_boundary)

    ax.fill_between(s_boundary, 0, alpha_boundary, alpha=0.2, color='green')
    ax.fill_between(s_boundary, alpha_boundary, max(alpha_vals)*1.2,
                    alpha=0.2, color='red')

    ax.plot(s_vals, alpha_vals, 'bo-', linewidth=2, markersize=4,
            label='평형 궤적')
    ax.plot(s_boundary, alpha_boundary, 'k--', linewidth=2,
            label='안정성 경계')

    # Mark unstable region
    unstable_mask = ~stable
    if np.any(unstable_mask):
        ax.plot(s_vals[unstable_mask], alpha_vals[unstable_mask], 'ro',
                markersize=6, label='불안정 위치')

    ax.set_xlabel('s')
    ax.set_ylabel('α')
    ax.set_title('s-α 안정성 궤적')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Stability margin
    ax = axes[1, 1]
    ax.plot(r_vals/a, margin, 'r-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.fill_between(r_vals/a, 0, margin, where=(margin>0), alpha=0.3,
                    color='red', label='불안정')
    ax.fill_between(r_vals/a, margin, 0, where=(margin<=0), alpha=0.3,
                    color='green', label='안정')

    ax.set_xlabel('r/a')
    ax.set_ylabel('α - α_crit')
    ax.set_title('Ballooning 안정성 여유')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/ballooning_analysis.png', dpi=150)
    print("\nBallooning 분석 플롯이 /tmp/ballooning_analysis.png에 저장되었습니다")
    plt.close()

    # Also create s-α diagram
    fig2 = plot_s_alpha_diagram()
    plt.savefig('/tmp/s_alpha_diagram.png', dpi=150)
    print("s-α 다이어그램이 /tmp/s_alpha_diagram.png에 저장되었습니다")
    plt.close()

if __name__ == "__main__":
    analyze_tokamak_ballooning()
```

### 7.3 Mercier 기준 평가

```python
import numpy as np
import matplotlib.pyplot as plt

def evaluate_mercier_criterion(r, q, p, Bp, R0):
    """
    Evaluate Mercier criterion: D_I > 1/4

    Parameters:
    -----------
    r: minor radius [m]
    q: safety factor
    p: pressure [Pa]
    Bp: poloidal field [T]
    R0: major radius [m]

    Returns:
    --------
    D_I: Mercier stability parameter
    stable: True if stable, False if unstable
    """
    mu0 = 4*np.pi*1e-7

    # Compute derivatives numerically
    dr = 0.001

    # Shear contribution
    dqdx = (q(r + dr) - q(r - dr)) / (2*dr)
    D_S = 0.25 * ((r / q(r)) * dqdx)**2

    # Magnetic well contribution
    dpdx = (p(r + dr) - p(r - dr)) / (2*dr)
    D_W = (mu0 * r / Bp(r)**2) * dpdx * (1 + 2*q(r)**2)

    # Geodesic curvature
    D_G = r**2 / (R0**2 * q(r)**2)

    # Total
    D_I = D_S + D_W + D_G

    # Stability criterion
    stable = D_I > 0.25

    return D_I, D_S, D_W, D_G, stable

def plot_mercier_stability():
    """Analyze Mercier stability for a tokamak"""

    # Parameters
    R0 = 3.0
    a = 1.0
    p0 = 5e5
    Bp0 = 0.5

    # Profiles
    def q_func(r):
        return 1.0 + 3.0*(r/a)**2

    def p_func(r):
        return p0 * (1 - (r/a)**2)**2

    def Bp_func(r):
        # Approximate: Bp ~ r for parabolic current
        return Bp0 * (r/a) if r > 0 else 1e-6

    # Evaluate over radius
    r_vals = np.linspace(0.1*a, 0.95*a, 100)

    D_I_vals = []
    D_S_vals = []
    D_W_vals = []
    D_G_vals = []
    stable_vals = []

    for r in r_vals:
        D_I, D_S, D_W, D_G, stable = evaluate_mercier_criterion(
            r, q_func, p_func, Bp_func, R0)

        D_I_vals.append(D_I)
        D_S_vals.append(D_S)
        D_W_vals.append(D_W)
        D_G_vals.append(D_G)
        stable_vals.append(stable)

    D_I_vals = np.array(D_I_vals)
    D_S_vals = np.array(D_S_vals)
    D_W_vals = np.array(D_W_vals)
    D_G_vals = np.array(D_G_vals)
    stable_vals = np.array(stable_vals)

    print("=== Mercier 기준 분석 ===\n")
    print(f"주요 반지름: R0 = {R0} m")
    print(f"소반지름: a = {a} m")

    # Check if any location is Mercier unstable
    if np.all(stable_vals):
        print("\n✓ 안정: Mercier 기준이 모든 반지름에서 만족됨")
    else:
        print("\n✗ 불안정: Mercier 기준 위배!")
        unstable_r = r_vals[~stable_vals]
        print(f"  불안정 영역: r/a ∈ [{unstable_r[0]/a:.2f}, {unstable_r[-1]/a:.2f}]")

    # Find minimum D_I
    idx_min = np.argmin(D_I_vals)
    print(f"\n가장 위험한 위치:")
    print(f"  r/a = {r_vals[idx_min]/a:.2f}")
    print(f"  D_I = {D_I_vals[idx_min]:.4f} (임계값: 0.25)")
    print(f"  여유: {D_I_vals[idx_min] - 0.25:+.4f}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Panel 1: Individual contributions
    ax = axes[0]
    ax.plot(r_vals/a, D_S_vals, 'b-', linewidth=2, label='D_S (전단)')
    ax.plot(r_vals/a, D_W_vals, 'r-', linewidth=2, label='D_W (우물)')
    ax.plot(r_vals/a, D_G_vals, 'g-', linewidth=2, label='D_G (측지선)')
    ax.plot(r_vals/a, D_I_vals, 'k-', linewidth=3, label='D_I (전체)')

    ax.set_xlabel('r/a', fontsize=12)
    ax.set_ylabel('Mercier 기여', fontsize=12)
    ax.set_title('Mercier 안정성 매개변수 성분', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Stability margin
    ax = axes[1]
    margin = D_I_vals - 0.25

    ax.plot(r_vals/a, margin, 'b-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.fill_between(r_vals/a, 0, margin, where=(margin>0), alpha=0.3,
                    color='green', label='안정 (D_I > 0.25)')
    ax.fill_between(r_vals/a, margin, 0, where=(margin<=0), alpha=0.3,
                    color='red', label='불안정 (D_I < 0.25)')

    ax.set_xlabel('r/a', fontsize=12)
    ax.set_ylabel('D_I - 0.25', fontsize=12)
    ax.set_title('Mercier 안정성 여유', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/mercier_stability.png', dpi=150)
    print("\nMercier 안정성 플롯이 /tmp/mercier_stability.png에 저장되었습니다")
    plt.close()

if __name__ == "__main__":
    plot_mercier_stability()
```

## 요약

이 강의에서 압력 구동 MHD 불안정성을 탐구했습니다:

1. **교환 불안정성**: 불리한 곡률($\boldsymbol{\kappa}\cdot\nabla p > 0$)에 의해 구동. 자기 전단과 유리한 평균 곡률에 의해 안정화.

2. **Rayleigh-Taylor 불안정성**: 가벼운 유체 위의 무거운 유체. 자기장이 장 방향을 따라 짧은 파장을 안정화하지만 수직 방향은 아님.

3. **Parker 불안정성**: 층화된 대기의 자기 부력. $\beta > 2/\gamma \approx 1.2$일 때 불안정. 천체물리학 플라즈마에 중요.

4. **Ballooning 모드**: 토로이달 기하학의 나쁜 곡률 측면에 국소화된 높은-$n$ 모드. s-α 공간의 안정성 경계: $\alpha_c \approx 0.6s$. 토카막의 ELM과 연결.

5. **Mercier 기준**: 국소 교환 안정성의 필요 조건: $D_I = D_S + D_W + D_G > 1/4$. 전단, 자기 우물, 측지선 곡률 결합.

6. **수치 도구**: RT 성장률, ballooning 안정성 다이어그램, Mercier 기준 평가 구현.

이러한 불안정성은 달성 가능한 플라즈마 압력(베타 한계)을 제한하고 핵융합 장치에서 신중한 평형 설계, 성형, 능동 제어의 필요성을 유발합니다.

## 연습 문제

### 문제 1: 미러의 교환 안정성

자기 미러가 장 강도 $B(z) = B_0(1 + z^2/L^2)$와 균일한 압력 $p = p_0$를 가집니다.

**(a)** 곡률 $\boldsymbol{\kappa} = \mathbf{b}\cdot\nabla\mathbf{b}$를 계산하세요. 여기서 $\mathbf{b} = \mathbf{B}/B$입니다.

**(b)** $\boldsymbol{\kappa}\cdot\nabla p$를 평가하세요.

**(c)** 이 구성이 교환에 대해 안정한가요, 불안정한가요?

**(d)** 압력 구배 $p(z) = p_0 e^{-z^2/L_p^2}$를 추가하면 안정성에 어떤 영향을 미칩니까?

### 문제 2: RT 임계 파장

밀도 $\rho_2 = 10^{-6}$ kg/m³의 플라즈마가 유효 중력 $g_{eff} = 10^4$ m/s² (원심 가속도)에서 수평 자기장 $B_0 = 1$ T에 의해 진공 ($\rho_1 \approx 0$) 위에 지지됩니다.

**(a)** RT 안정성을 위한 임계 파수 $k_c$를 계산하세요.

**(b)** 임계 파장 $\lambda_c = 2\pi/k_c$로 변환하세요.

**(c)** $k = 0.5k_c$에 대해, 성장률 $\gamma$를 계산하세요.

**(d)** 성장 시간 $\tau = 1/\gamma$를 추정하세요.

**(e)** $\tau$를 Alfvén 시간 $\tau_A = \lambda_c/v_A$와 비교하세요.

### 문제 3: Ballooning 안정성 경계

토카막이 중간 반지름에서 자기 전단 $s = 2.5$를 가집니다.

**(a)** 근사 공식 $\alpha_c \approx 0.6s$를 사용하여, 임계 압력 매개변수를 계산하세요.

**(b)** 실제 $\alpha = 2.0$이면, 플라즈마가 ballooning에 대해 안정한가요, 불안정한가요?

**(c)** 한계 안정성을 달성하기 위해 압력 구배를 어느 정도 줄여야 합니까?

**(d)** 대신 $s$를 4.0으로 증가시키면 ($\alpha=2.0$을 유지하면서), 플라즈마가 안정해집니까?

**(e)** 토카막에서 $s$를 증가시키는 두 가지 실험적 방법을 논의하세요.

### 문제 4: 은하 원반의 Parker 불안정성

은하 원반이 다음을 가집니다:
- 가스 밀도 $\rho_0 = 10^{-24}$ kg/m³
- 온도 $T = 10^4$ K
- 자기장 $B = 5\times 10^{-10}$ T
- 스케일 높이 $H = 100$ pc = $3\times 10^{18}$ m

**(a)** 가스 압력 $p = nkT$를 계산하세요 ($n = \rho_0/m_p$라고 가정).

**(b)** 플라즈마 베타 $\beta = 2\mu_0 p/B^2$를 계산하세요.

**(c)** Parker 불안정성 조건 $\beta > 2/\gamma$를 확인하세요 ($\gamma = 5/3$ 사용).

**(d)** 불안정하면, 성장률 $\gamma \sim \sqrt{g/H}$를 추정하세요. 여기서 $g \sim kT/(m_p H)$입니다.

**(e)** 성장 시간을 연 단위로 변환하세요. 이것이 관측된 분자 구름 형성 시간 척도와 일치합니까?

### 문제 5: Mercier 기준 성분

토카막이 $r = 0.5a$에서 다음을 가집니다:
- 안전 인자 $q = 1.5$
- 자기 전단 $(r/q)(dq/dr) = 1.0$
- 압력 $p = 2\times 10^5$ Pa
- 압력 구배 $dp/dr = -10^6$ Pa/m
- 폴로이달 장 $B_p = 0.4$ T
- 주요 반지름 $R_0 = 3$ m

**(a)** 전단 기여 $D_S = \frac{1}{4}\left(\frac{r}{q}\frac{dq}{dr}\right)^2$를 계산하세요.

**(b)** 자기 우물 기여 $D_W = \frac{\mu_0 r}{B_p^2}\frac{dp}{dr}(1 + 2q^2)$를 계산하세요.

**(c)** 측지선 곡률 $D_G = \frac{r^2}{R_0^2 q^2}$를 계산하세요.

**(d)** $D_I = D_S + D_W + D_G$를 평가하세요.

**(e)** $D_I > 0.25$ (Mercier 안정)인지 확인하세요.

**(f)** 어느 항이 안정성/불안정성에 가장 많이 기여합니까?

---

**이전**: [Linear Stability](./02_Linear_Stability.md) | **다음**: [Current-Driven Instabilities](./04_Current_Driven_Instabilities.md)
