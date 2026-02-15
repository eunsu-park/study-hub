# 13. 핵융합 MHD

## 학습 목표

- 자기 confinement 개념 이해: tokamak, stellarator, reversed field pinch (RFP)
- Tokamak 평형 분석: aspect ratio, elongation, triangularity, Shafranov shift
- Beta 한계 유도 및 Troyon limit 이해
- Tokamak의 주요 MHD 불안정성 파악: sawteeth, ELMs, disruptions, NTMs, RWMs
- Disruption 완화 전략 적용 및 물리적 기반 이해
- 정상 상태 핵융합을 위한 stellarator 장점 비교
- Beta limits, sawtooth periods, disruption forces를 위한 Python 모델 구현

## 1. 자기 Confinement 핵융합 소개

자기 confinement 핵융합은 자기장을 사용하여 고온 플라즈마(T ~ 10-20 keV)를 가두어 제어된 열핵융합을 달성하는 것을 목표로 합니다. 주요 과제는 MHD 불안정성에 대한 플라즈마 안정성을 유지하면서 Lawson criterion을 만족하기에 충분한 confinement time과 압력을 달성하는 것입니다.

### 1.1 핵융합 반응 및 요구사항

Deuterium-tritium (D-T) 핵융합 반응이 가장 접근 가능합니다:

```
D + T → He-4 (3.5 MeV) + n (14.1 MeV)
```

핵융합 triple product 요구사항:

```
n T τ_E ≥ 3 × 10²¹ m⁻³ keV s
```

여기서:
- $n$은 플라즈마 밀도
- $T$는 온도
- $\tau_E$는 에너지 confinement time

### 1.2 자기 Confinement 원리

하전 입자는 Larmor 반경으로 자기장선 주위를 회전합니다:

```
r_L = (m v_⊥)/(q B)
```

B = 5 T에서 T = 10 keV인 열적 입자의 경우:
- 전자: $r_{Le} \sim 0.1$ mm
- 이온: $r_{Li} \sim 4$ mm

장치 크기에 비해 작은 Larmor 반경은 confinement를 가능하게 합니다. 그러나 원환 기하학은 drift 운동을 도입하고 신중한 자기장 구성을 필요로 합니다.

## 2. Tokamak 구성

Tokamak은 플라즈마를 원환에 가두기 위해 toroidal 및 poloidal 자기장의 조합을 사용하는 선도적인 자기 confinement 개념입니다.

### 2.1 Tokamak 자기장 구조

Tokamak의 총 자기장은 다음으로 구성됩니다:

1. **Toroidal field** $B_φ$: 외부 toroidal field coils에 의해 생성되는 강한 장
2. **Poloidal field** $B_θ$: 플라즈마 전류 $I_p$ 및 외부 poloidal field coils에 의해 생성되는 약한 장
3. **Vertical field** $B_z$: 플라즈마 위치 및 형상 제어

총 장:

```
B = B_φ e_φ + B_θ e_θ
```

자기장선은 중첩된 flux surfaces (자기 표면) 위에서 원환 주위를 감습니다.

### 2.2 Safety Factor

Safety factor $q$는 자기장선의 pitch를 측정합니다:

```
q = (r B_φ)/(R B_θ)
```

여기서 $R$은 major 반경이고 $r$은 minor 반경입니다.

원형 단면을 가진 큰 aspect ratio tokamak의 경우:

```
q(r) = (r B_0)/(R B_θ(r)) ≈ (2π r² B_0)/(μ₀ R I_p(r))
```

여기서 $I_p(r)$은 반경 $r$ 내부의 플라즈마 전류입니다.

Safety factor 프로파일 $q(r)$은 MHD 안정성에 중요합니다:
- 축에서 $q < 1$이면 sawtooth oscillations 허용
- $q_{edge} < 2$는 disruptions로 이어짐
- $q = m/n$인 rational surfaces는 tearing modes에 취약

### 2.3 플라즈마 전류 프로파일

플라즈마 전류 밀도는 Ampère's law로부터 따릅니다:

```
∇ × B = μ₀ j
```

Tokamak에서 toroidal 전류 밀도:

```
j_φ = (1/μ₀ r) ∂(r B_θ)/∂r
```

일반적인 전류 프로파일:
- **Peaked**: $j(r) = j_0 (1 - r²/a²)^ν$, $ν > 0$
- **Flat**: $ν \approx 1$
- **Hollow**: 전류 밀도 최대값이 축 밖에 위치

전류 프로파일은 $q$-프로파일을 결정하고 안정성에 영향을 줍니다.

### 2.4 Aspect Ratio 및 플라즈마 형상

주요 기하학적 매개변수:

- **Aspect ratio**: $A = R/a$ (일반적으로 2.5-4)
- **Elongation**: $κ = b/a$ (수직/수평 minor 반경, 일반적으로 1.5-2)
- **Triangularity**: $δ$ (D자 형태 단면 특성)

높은 elongation은 플라즈마 부피를 증가시키고 confinement를 개선하지만 vertical displacement events (VDEs)에 대한 취약성을 증가시킵니다.

## 3. Tokamak 평형

### 3.1 Grad-Shafranov 방정식

Tokamak 평형은 축대칭 기하학에서 힘 균형 $j × B = ∇p$로부터 유도된 Grad-Shafranov (GS) 방정식에 의해 지배됩니다.

Poloidal flux 함수 $\psi(R, Z)$ 도입:

```
B_R = -(1/R) ∂ψ/∂Z
B_Z = (1/R) ∂ψ/∂R
```

GS 방정식:

```
Δ* ψ = -μ₀ R² dp/dψ - F dF/dψ
```

여기서 타원 연산자:

```
Δ* ψ = R ∂/∂R (1/R ∂ψ/∂R) + ∂²ψ/∂Z²
```

그리고 $F(ψ) = R B_φ$는 toroidal field 함수입니다.

### 3.2 Shafranov Shift

유한 압력 플라즈마에서 자기 축은 toroidal 효과로 인해 기하학적 중심에서 바깥쪽으로 이동합니다. 이 **Shafranov shift** $\Delta$는 대략:

```
Δ/a ≈ β_p + l_i/2
```

여기서:
- $\beta_p = 2 μ₀ \langle p \rangle / \langle B_θ² \rangle$는 poloidal beta
- $l_i$는 내부 inductance (전류 프로파일에 의존)

일반적인 tokamak 매개변수 ($\beta_p \sim 0.5$, $l_i \sim 1$)의 경우, $\Delta/a \sim 0.5-1$.

Shafranov shift는 압력에 따라 증가하고 평형 한계에 영향을 줍니다.

### 3.3 Beta 한계

플라즈마 beta는 플라즈마 압력 대 자기 압력의 비율입니다:

```
β = 2 μ₀ p / B²
```

여러 정의:
- **Total beta**: $\beta = 2 μ₀ \langle p \rangle / B_0²$
- **Poloidal beta**: $\beta_p = 2 μ₀ \langle p \rangle / \langle B_θ² \rangle$
- **Toroidal beta**: $\beta_t = 2 μ₀ \langle p \rangle / \langle B_φ² \rangle$

**Troyon limit**은 달성 가능한 최대 beta에 대한 경험적 스케일링입니다:

```
β_N = β (%·T·m/MA) = β a B_0 / I_p ≤ β_N^max
```

여기서:
- $a$는 minor 반경 (m)
- $B_0$는 toroidal field (T)
- $I_p$는 플라즈마 전류 (MA)
- $\beta_N^{max} \approx 2.8-4$ (표준 시나리오)

높은 beta는 핵융합 전력 밀도에 바람직하지만, MHD 불안정성 (압력 구동 모드, 외부 kinks)이 한계를 부과합니다.

### 3.4 평형 Beta 한계

큰 aspect ratio tokamak의 경우, 압력 기울기와 자기 장력의 균형:

```
β_t ≤ a/(q R) = 1/(A q)
```

이것은 대략적인 추정입니다. GS 방정식을 사용한 더 정밀한 계산은 Troyon limit를 제공합니다.

## 4. Tokamak의 주요 MHD 불안정성

### 4.1 Sawtooth Oscillations

Sawteeth는 $q_0 < 1$인 tokamak에서 코어 온도와 밀도의 주기적 이완입니다.

**메커니즘:**
1. Ohmic heating이 뾰족한 온도 프로파일 생성
2. $q_0 < 1$일 때 internal kink mode ($m=1, n=1$)가 불안정해짐
3. 자기 재결합이 코어 온도 프로파일을 평평하게 함 (sawtooth crash)
4. Ohmic heating이 뾰족한 프로파일을 재구축하면서 사이클 반복

**Kadomtsev 재결합 모델:**

Internal kink mode는 $q=1$ 표면에서 자기장선을 재결합시켜 mixing 반경 $r_{mix}$ 내부의 온도 프로파일을 평평하게 합니다.

Sawtooth 주기는 다음과 같이 스케일링됩니다:

```
τ_sawtooth ∝ a² / (η S^α)
```

여기서 $S$는 Lundquist 수이고 $α \approx 0.6$ (시뮬레이션으로부터).

**영향:**
- 가장자리로의 주기적 열 펄스
- Neoclassical tearing modes (NTMs) 유발 가능
- 유익: 과도한 peaking 방지, 불순물 배출

**제어 방법:**
- $q=1$ 표면 근처의 Electron cyclotron current drive (ECCD)
- 제어된 crashes를 유발하기 위한 pellet 주입

### 4.2 Edge Localized Modes (ELMs)

ELMs은 high-confinement mode (H-mode)에서 플라즈마 가장자리의 주기적 불안정성입니다. H-mode는 가장자리 근처에 급격한 압력 기울기 (pedestal)를 특징으로 하며, 이는 불안정해질 수 있습니다.

**Peeling-ballooning instability:**

두 가지 구동 메커니즘:
1. **Peeling**: 가장자리 전류 밀도가 외부 kink modes 구동
2. **Ballooning**: 급격한 압력 기울기가 interchange-like modes 구동

$(j_{edge}, \nabla p_{edge})$ 공간에서의 안정성 경계는 "peeling-ballooning" 다이어그램을 형성합니다.

**ELM 유형:**

- **Type I (giant ELMs)**: pedestal 에너지의 5-15%를 배출하는 대규모 주기적 crashes
  - divertor에 상당한 열 플럭스 유발 가능 ($> 10$ MW/m²)
  - 빈도: 10-100 Hz

- **Type III (small ELMs)**: 더 작고 더 빈번함
  - 낮은 pedestal 압력
  - divertor 우려 적음

- **QH-mode (ELM-free)**: edge harmonic oscillation (EHO)을 가진 Quiescent H-mode
  - 대형 ELMs 없이 연속적인 가장자리 입자/에너지 배출
  - 회전 shear 필요, DIII-D에서 관찰됨

**ELMs로부터의 Divertor 열 플럭스:**

```
q_peak ≈ W_ELM / (A_wet τ_ELM)
```

여기서:
- $W_{ELM}$은 ELM당 배출되는 에너지
- $A_{wet}$는 divertor의 wetted 면적
- $\tau_{ELM}$는 ELM 에너지 증착 시간 (0.1-1 ms)

ITER의 경우, 완화되지 않은 Type I ELMs은 재료 한계를 초과하는 $q_{peak} > 20$ MW/m²를 전달할 수 있습니다.

**ELM 완화 전략:**

1. **Resonant Magnetic Perturbations (RMPs)**: 외부 3D 장이 stochastic 가장자리 층 생성
   - DIII-D, ASDEX-U, KSTAR에서 입증됨
   - confinement의 일부 비용으로 ELMs 감소 또는 제거

2. **Pellet pacing**: 작은 pellets 주입이 더 높은 빈도로 ELMs을 유발하여 크기 감소

3. **QH-mode 또는 I-mode**: ELM-free regimes 달성

### 4.3 Disruptions

Disruption은 밀리초 시간 척도에서 발생하는 플라즈마 confinement의 파국적 손실입니다. Disruptions은 ITER와 같은 대형 tokamaks에 주요 과제를 제기합니다.

**원인:**

1. **밀도 한계**: Greenwald 밀도에 접근
   ```
   n_G = I_p / (π a²) (10²⁰ m⁻³ MA⁻¹ m⁻²)
   ```
   $n_G$를 초과하면 복사 붕괴 및 열적 불안정성으로 이어짐.

2. **전류 한계**: 가장자리 safety factor $q_{edge} < 2$는 외부 kink modes로 이어짐

3. **Locked modes**: 오류 장 또는 낮은 회전으로 인해 벽에 고정되는 tearing modes

4. **Beta 한계**: MHD beta limit 초과가 ideal modes 유발

**Disruption 단계:**

1. **열적 quench (TQ)**: 열 에너지 손실 (0.1-1 ms)
   - 온도 붕괴: $T \rightarrow 0$
   - 벽으로의 열 플럭스: 재료 한계 초과 가능
   - 원인: MHD 모드 성장, stochastization

2. **전류 quench (CQ)**: 플라즈마 전류 손실 (1-100 ms)
   - 플라즈마 전류 감쇠: $I_p \rightarrow 0$
   - 전도 구조물에 유도된 전압 및 힘
   - Runaway electron (RE) 생성 위험

3. **Runaway electron beam**: 고도로 상대론적인 전자
   - CQ 동안 유도 전기장에 의해 가속됨
   - 상당한 전류 운반 가능 (MA 수준)
   - 빔이 벽에 충돌하면 고도로 국지화된 열 증착

**Tokamak 구조물에 대한 힘:**

CQ 동안 변화하는 플라즈마 전류는 진공 용기 및 구조물에 와전류를 유도하여 큰 전자기력으로 이어집니다.

**수직 힘:**

```
F_z ~ (dI_p/dt) * (mutual inductance)
```

ITER disruption의 경우: $F_z$는 수 MN에 도달할 수 있습니다.

**Halo currents:**

플라즈마 scrape-off layer를 통해 first wall로, 그 다음 구조물을 통해 플라즈마로 다시 흐르는 전류. 이들은 toroidal 비대칭 힘을 생성합니다.

**Disruption 완화:**

1. **Massive Gas Injection (MGI)**: 대량의 희가스(Ne, Ar) 주입
   - 열 에너지를 더 균일하게 복사
   - runaway 생성을 억제하기 위해 전자 밀도 증가
   - 힘을 감소시키기 위해 전류 quench를 늦춤

2. **Shattered Pellet Injection (SPI)**: 조각으로 부서지는 얼어붙은 pellet 주입
   - MGI보다 더 깊은 침투 및 더 빠른 동화
   - 더 효과적인 복사 분포
   - ITER의 기본 완화 시스템

3. **Disruption 예측 및 회피**: 기계 학습 모델이 수십에서 수백 ms 전에 disruptions 예측
   - disruption 영역을 피하기 위한 실시간 제어
   - 회피 실패 시 완화 트리거

### 4.4 Neoclassical Tearing Modes (NTMs)

NTMs는 자기 섬 내부의 bootstrap 전류에 대한 섭동에 의해 구동되는 resistive tearing modes입니다.

**Bootstrap 전류:**

압력 기울기를 가진 toroidal 플라즈마에서 trapped 입자는 순 toroidal 전류에 기여합니다:

```
j_bs = C(ν*, ε) d p/dr
```

여기서 $\nu^*$는 충돌성이고 $\varepsilon = r/R$은 역 aspect ratio입니다.

**섬 역학:**

Tearing mode가 rational surface $q = m/n$에서 자기 섬을 생성할 때, 압력은 섬 내부에서 평평해지고 국소 bootstrap 전류를 감소시킵니다. 이 누락된 전류가 섬 성장을 구동합니다.

NTM 섬 폭 $w$에 대한 수정된 Rutherford 방정식:

```
τ_R dw/dt = r_s Δ'_{classical} + r_s Δ'_{bs}(w)
```

여기서:

```
Δ'_{bs} = L_{q,p} / w²
```

는 bootstrap 구동 항 (양수, 불안정화)이고 $L_{q,p}$는 압력 및 safety factor 프로파일에 의존합니다.

**NTM 여기 임계값:**

NTMs는 임계 폭을 초과하는 seed 섬 (일반적으로 sawteeth 또는 ELMs로부터)을 필요로 합니다:

```
w_crit ~ sqrt(L_{q,p} / |Δ'_{classical}|)
```

**제어:**

Rational surface에 국지화된 Electron Cyclotron Current Drive (ECCD)는 누락된 bootstrap 전류를 대체하여 NTM 성장을 억제하거나 방지할 수 있습니다.

### 4.5 Resistive Wall Modes (RWMs)

RWMs는 resistive conducting 벽에 의해 부분적으로 안정화된 외부 kink modes입니다.

**전도 벽을 가진 Ideal kink:**

Ideal 외부 kink mode는 플라즈마에 가까운 완전 전도 벽에 의해 안정화될 수 있습니다. Resistive 벽의 경우, 안정화는 일시적입니다: 모드는 벽 resistive 시간 척도 $\tau_{wall}$에서 성장합니다.

**성장률:**

```
γ ≈ τ_wall^{-1}
```

여기서 $\tau_{wall} \sim μ₀ \sigma d_{wall} b_{wall}$ ($\sigma$는 벽 전도도, $d_{wall}$ 두께, $b_{wall}$ 반경).

일반적인 시간 척도: $\tau_{wall} \sim 10-100$ ms (ideal MHD보다 훨씬 느림).

**안정화:**

- **플라즈마 회전**: 플라즈마가 RWM 성장률보다 빠르게 회전하면 모드가 안정화됨
  ```
  ω_rot > γ_RWM
  ```

- **Active feedback control**: 외부 coils이 모드를 감지하고 보정 장 적용

- **Kinetic effects**: energetic 입자의 precession drift와의 공명이 damping 제공 가능

RWMs는 회전 또는 feedback 없이 advanced tokamak 시나리오에서 달성 가능한 beta를 제한합니다.

## 5. Stellarator 구성

Stellarator는 플라즈마 전류에 의존하지 않고 외부 3D 자기장을 사용하여 confinement를 달성하는 tokamak의 대안입니다.

### 5.1 Stellarator 자기장

Stellarator에서 비틀린 자기장선은 rotational transform (tokamak의 1/q와 동등)을 생성하는 외부 coils에 의해 전적으로 생성됩니다.

**장점:**
- **정상 상태**: 전류 구동 필요 없음
- **disruptions 없음**: 큰 플라즈마 전류 없음, 전류 구동 불안정성 없음
- **유연한 최적화**: 안정성 및 confinement를 위해 장 형상 최적화 가능

**과제:**
- **복잡한 3D 기하학**: 설계, 구축 및 분석 어려움
- **Neoclassical transport**: 3D 장에서의 drift orbits이 향상된 transport로 이어질 수 있음
- **제한된 실험 데이터베이스**: tokamaks보다 대형 장치가 적음

### 5.2 Quasi-Symmetry

현대 stellarators는 quasi-symmetry를 목표로 합니다: 자기 좌표계의 특정 방향에서 장 강도 $|B|$가 대략 대칭입니다 (예: quasi-helical, quasi-toroidal, quasi-axisymmetric).

Quasi-symmetry는 입자 drift surfaces가 flux surfaces와 일치하도록 보장하여 neoclassical transport를 감소시킵니다.

**예:**
- W7-X (독일): Quasi-isodynamic, modular coils
- HSX (미국): Quasi-helically symmetric
- NCSX (미국, 취소됨): Quasi-axisymmetric

### 5.3 Stellarators의 MHD 안정성

Stellarators는 저차 rational surfaces를 피하도록 설계될 수 있어 tearing modes에 대한 취약성을 감소시킵니다. 그러나 다른 MHD 안정성 과제에 직면합니다:

- **Interchange modes**: 불리한 곡률 영역이 interchange 불안정성 구동 가능
- **Ballooning modes**: tokamaks와 유사한 압력 구동 불안정성
- **External kinks**: 평형이 최적이 아닌 경우

수치 최적화 코드 (예: 평형을 위한 VMEC, 안정성을 위한 TERPSICHORE)는 stellarator 설계에 필수적입니다.

### 5.4 W7-X 결과

Wendelstein 7-X (W7-X)는 2015년에 first plasma를 달성했고 다음을 입증했습니다:
- 긴 펄스 (최대 101 s)
- tokamaks와 비교 가능한 좋은 에너지 confinement
- 예측과 일치하는 낮은 neoclassical transport
- 섬 및 오류 장 제어

Stellarators는 특히 정상 상태 작동에서 핵융합 반응로에 대한 강력한 후보로 남아 있습니다.

## 6. Reversed Field Pinch (RFP)

RFP는 플라즈마의 외부 영역에서 toroidal 자기장이 방향을 반전시키는 toroidal confinement 개념입니다.

### 6.1 RFP 자기장 구조

RFP는 비교 가능한 toroidal 및 poloidal 장을 가집니다:

```
B_φ(r)는 r = r_reversal에서 부호 변경
B_θ(r) ~ 상수
```

장 구성은 플라즈마 전류 및 MHD dynamo 작용에 의해 유지됩니다.

### 6.2 Taylor Relaxation

Taylor의 가설: 난류 플라즈마는 전역 자기 helicity의 일정한 제약 조건 하에서 최소 에너지 상태로 이완됩니다.

이완된 상태는 다음을 만족합니다:

```
∇ × B = μ B
```

여기서 $\mu$는 상수입니다 (force-free 방정식의 eigenvalue).

실린더에서 이것은 Bessel 함수 프로파일을 생성합니다:

```
B_z(r) = B_0 J_0(μ r)
B_θ(r) = B_0 J_1(μ r)
```

$\mu a$가 $J_0(\mu a) < 0$이 되도록 선택되면 장은 가장자리에서 반전됩니다.

### 6.3 RFP MHD 활동

RFPs는 전류 프로파일을 이완시키고 반전된 장을 유지하는 강한 MHD 변동 (tearing modes)을 나타냅니다. 이 "MHD dynamo"는 RFP 작동에 필수적이지만 confinement를 저하시킵니다.

**최근 개선:**
- **Pulsed Poloidal Current Drive (PPCD)**: MHD 변동 감소, confinement 개선
- **Quasi-single-helicity (QSH) states**: 하나의 지배적인 모드, 감소된 chaos

RFPs는 $\beta \sim 10-20\%$를 달성하여 tokamaks보다 높지만 confinement time은 더 짧습니다.

## 7. Beta 한계 및 안정성 경계

### 7.1 Troyon Beta Limit 유도 (발견적)

큰 aspect ratio tokamak을 고려합니다. 외부 kink mode는 플라즈마 압력 및 전류에 의해 구동됩니다. 불안정화 압력 항과 안정화 자기장선 굽힘의 균형:

```
β ~ 1 / (q a/R)
```

$q \sim a² B_φ / (μ₀ R I_p)$를 사용하여 플라즈마 전류로 표현:

```
β ~ μ₀ I_p / (a B_φ)
```

재배열:

```
β a B_φ / I_p ~ 상수
```

이것은 정규화된 beta $\beta_N$입니다. 더 상세한 계산은 다음을 제공합니다:

```
β_N^{max} ≈ C_T l_i / (A q_cyl)
```

여기서 $C_T \approx 2.8$ (Troyon 계수), $l_i$는 내부 inductance, $A$는 aspect ratio, $q_{cyl}$은 원통형 safety factor입니다.

### 7.2 Ballooning Mode 한계

Ballooning modes는 불리한 곡률 영역에 국지화된 높은 toroidal mode number ($n \rightarrow \infty$) 압력 구동 불안정성입니다.

Mercier criterion은 국소 안정성 조건을 제공합니다:

```
D_I > 0
```

여기서 $D_I$는 압력 기울기, shear 및 자기 well 깊이를 포함합니다.

Tokamak의 경우, ballooning 안정성은 대략 다음을 요구합니다:

```
dp/dr < (임계 기울기)
```

Beta에 대한 ballooning 한계:

```
β_crit ~ (ε/q²) (shear factor)
```

높은 shear ($s = r q'/q$) 및 큰 aspect ratio가 ballooning 안정성을 개선합니다.

### 7.3 Advanced 시나리오의 전역 Beta 한계

Advanced tokamak 시나리오는 높은 beta, 높은 bootstrap 분율 및 정상 상태 작동을 목표로 합니다. 이러한 시나리오는 no-wall ideal kink limit 근처 또는 위에서 작동하지만 with-wall limit 아래에서 작동합니다.

**작동 공간:**
- $\beta_N \sim 3-4$ (no-wall limit ~2.5 위)
- Resistive wall mode 제어 필요 (회전, feedback)
- 높은 $q$-프로파일 (예: $q_{min} > 2$)로 sawteeth 회피 및 NTM 구동 감소

## 8. Python 구현

### 8.1 Troyon Beta Limit

```python
import numpy as np
import matplotlib.pyplot as plt

def troyon_beta_limit(I_p, a, B_0, C_Troyon=2.8):
    """
    Calculate Troyon beta limit.

    Parameters:
    I_p : plasma current (MA)
    a : minor radius (m)
    B_0 : toroidal magnetic field on axis (T)
    C_Troyon : Troyon coefficient (dimensionless, typically 2.8)

    Returns:
    beta_N : normalized beta limit (%)
    beta_percent : absolute beta limit (%)
    """
    beta_N = C_Troyon  # Troyon limit (% T m / MA)
    beta_percent = beta_N * I_p / (a * B_0)
    return beta_N, beta_percent

# Example: ITER-like parameters
I_p_ITER = 15.0  # MA
a_ITER = 2.0     # m
B_0_ITER = 5.3   # T

beta_N_limit, beta_limit = troyon_beta_limit(I_p_ITER, a_ITER, B_0_ITER)
print(f"ITER parameters: I_p = {I_p_ITER} MA, a = {a_ITER} m, B_0 = {B_0_ITER} T")
print(f"Troyon limit: β_N = {beta_N_limit:.2f} % T m / MA")
print(f"Absolute beta limit: β = {beta_limit:.2f} %")

# Scan over plasma current
I_p_scan = np.linspace(5, 20, 50)
beta_scan = [troyon_beta_limit(I_p, a_ITER, B_0_ITER)[1] for I_p in I_p_scan]

plt.figure(figsize=(8, 5))
plt.plot(I_p_scan, beta_scan, 'b-', linewidth=2)
plt.xlabel('Plasma Current (MA)', fontsize=12)
plt.ylabel('Beta Limit (%)', fontsize=12)
plt.title('Troyon Beta Limit vs Plasma Current', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('troyon_beta_limit.png', dpi=150)
plt.show()
```

### 8.2 Sawtooth Period Model

```python
def sawtooth_period(a, T_e, n_e, B, S_exp=0.6):
    """
    Estimate sawtooth period using scaling law.

    Parameters:
    a : minor radius (m)
    T_e : electron temperature (eV)
    n_e : electron density (m^-3)
    B : magnetic field (T)
    S_exp : Lundquist number exponent (typically 0.6)

    Returns:
    tau_sawtooth : sawtooth period (s)
    """
    # Physical constants
    e = 1.602e-19  # C
    m_e = 9.109e-31  # kg
    epsilon_0 = 8.854e-12  # F/m
    mu_0 = 4 * np.pi * 1e-7  # H/m

    # Spitzer resistivity
    ln_Lambda = 15.0  # Coulomb logarithm (approximate)
    eta = (e**2 * ln_Lambda * m_e**0.5) / (12 * np.pi**1.5 * epsilon_0**2 * (e * T_e)**1.5)

    # Lundquist number
    tau_R = mu_0 * a**2 / eta
    tau_A = a / (B / np.sqrt(mu_0 * n_e * m_e * 1836))  # Alfven time (approximation)
    S = tau_R / tau_A

    # Sawtooth period scaling
    tau_sawtooth = tau_R / S**S_exp * 50  # Empirical factor

    return tau_sawtooth, S, eta

# Example: JET-like parameters
a_JET = 1.0  # m
T_e_JET = 2000  # eV (core temperature)
n_e_JET = 5e19  # m^-3
B_JET = 3.0  # T

tau_saw, S_JET, eta_JET = sawtooth_period(a_JET, T_e_JET, n_e_JET, B_JET)
print(f"\nJET parameters: a = {a_JET} m, T_e = {T_e_JET} eV, n_e = {n_e_JET:.1e} m^-3, B = {B_JET} T")
print(f"Spitzer resistivity: η = {eta_JET:.3e} Ω m")
print(f"Lundquist number: S = {S_JET:.2e}")
print(f"Estimated sawtooth period: τ = {tau_saw:.3f} s")

# Scan over temperature
T_e_scan = np.linspace(500, 5000, 50)
tau_scan = [sawtooth_period(a_JET, T_e, n_e_JET, B_JET)[0] for T_e in T_e_scan]

plt.figure(figsize=(8, 5))
plt.plot(T_e_scan, tau_scan, 'r-', linewidth=2)
plt.xlabel('Electron Temperature (eV)', fontsize=12)
plt.ylabel('Sawtooth Period (s)', fontsize=12)
plt.title('Sawtooth Period vs Electron Temperature', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sawtooth_period.png', dpi=150)
plt.show()
```

### 8.3 Disruption Force Estimation

```python
def disruption_forces(I_p, dI_dt, R, a, b_wall):
    """
    Estimate electromagnetic forces during disruption.

    Parameters:
    I_p : initial plasma current (MA)
    dI_dt : current quench rate (MA/s)
    R : major radius (m)
    a : minor radius (m)
    b_wall : wall minor radius (m)

    Returns:
    F_z : vertical force (MN)
    V_loop : loop voltage (V)
    """
    mu_0 = 4 * np.pi * 1e-7

    # Mutual inductance (simple model)
    M = mu_0 * R * (np.log(8 * R / a) - 2 + 0.5)  # H

    # Vertical force (simplified)
    F_z = abs(I_p * 1e6 * dI_dt * 1e6 * M / (2 * np.pi * R)) / 1e6  # MN

    # Loop voltage
    V_loop = abs(M * dI_dt * 1e6)  # V

    return F_z, V_loop

# Example: ITER disruption
I_p_ITER_disr = 15.0  # MA
dI_dt_ITER = -15.0 / 0.15  # MA/s (15 MA in 150 ms)
R_ITER = 6.2  # m
a_ITER_disr = 2.0  # m
b_wall_ITER = 2.3  # m

F_z_ITER, V_loop_ITER = disruption_forces(I_p_ITER_disr, dI_dt_ITER, R_ITER, a_ITER_disr, b_wall_ITER)
print(f"\nITER disruption: I_p = {I_p_ITER_disr} MA, dI/dt = {dI_dt_ITER:.1f} MA/s")
print(f"Estimated vertical force: F_z ~ {F_z_ITER:.2f} MN")
print(f"Estimated loop voltage: V_loop ~ {V_loop_ITER:.1f} V")

# Current quench timescale scan
tau_CQ_scan = np.linspace(0.01, 0.5, 50)  # s
dI_dt_scan = -I_p_ITER_disr / tau_CQ_scan
F_z_scan = [disruption_forces(I_p_ITER_disr, dI_dt, R_ITER, a_ITER_disr, b_wall_ITER)[0] for dI_dt in dI_dt_scan]

plt.figure(figsize=(8, 5))
plt.plot(tau_CQ_scan * 1000, F_z_scan, 'g-', linewidth=2)
plt.xlabel('Current Quench Time (ms)', fontsize=12)
plt.ylabel('Vertical Force (MN)', fontsize=12)
plt.title('Disruption Vertical Force vs Current Quench Time', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('disruption_forces.png', dpi=150)
plt.show()
```

### 8.4 Safety Factor Profile

```python
def safety_factor_profile(r, a, R, B_0, I_p, profile='parabolic', nu=1.0):
    """
    Calculate safety factor profile.

    Parameters:
    r : radial coordinate (m) or array
    a : minor radius (m)
    R : major radius (m)
    B_0 : toroidal field on axis (T)
    I_p : plasma current (MA)
    profile : 'parabolic' or 'flat'
    nu : profile parameter (for parabolic)

    Returns:
    q : safety factor
    """
    r = np.atleast_1d(r)
    mu_0 = 4 * np.pi * 1e-7

    if profile == 'parabolic':
        # j(r) = j_0 (1 - (r/a)^2)^nu
        # I(r) = 2π ∫ j(r') r' dr'
        # For simplicity, approximate q(r)
        q_edge = (a**2 * B_0) / (mu_0 * R * I_p * 1e6) * 2 * np.pi
        q_0 = q_edge / (nu + 1)
        q = q_0 + (q_edge - q_0) * (r / a)**2
    elif profile == 'flat':
        # Flat current profile
        q = (r**2 * B_0) / (mu_0 * R * I_p * 1e6 / (np.pi * a**2)) / (2 * np.pi)
        q[r == 0] = 0  # Avoid singularity
    else:
        raise ValueError("Profile must be 'parabolic' or 'flat'")

    return q

# Plot q-profile for different current profiles
r_array = np.linspace(0, a_ITER, 100)

q_parabolic_1 = safety_factor_profile(r_array, a_ITER, 6.2, B_0_ITER, I_p_ITER, 'parabolic', nu=1.0)
q_parabolic_2 = safety_factor_profile(r_array, a_ITER, 6.2, B_0_ITER, I_p_ITER, 'parabolic', nu=2.0)

plt.figure(figsize=(10, 6))
plt.plot(r_array, q_parabolic_1, 'b-', linewidth=2, label='Parabolic (ν=1)')
plt.plot(r_array, q_parabolic_2, 'r-', linewidth=2, label='Parabolic (ν=2)')
plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='q=1 (sawtooth)')
plt.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='q=2 (disruption)')
plt.xlabel('Minor Radius r (m)', fontsize=12)
plt.ylabel('Safety Factor q', fontsize=12)
plt.title('Safety Factor Profile', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('safety_factor_profile.png', dpi=150)
plt.show()
```

### 8.5 Greenwald Density Limit

```python
def greenwald_density(I_p, a):
    """
    Calculate Greenwald density limit.

    Parameters:
    I_p : plasma current (MA)
    a : minor radius (m)

    Returns:
    n_G : Greenwald density (10^20 m^-3)
    """
    n_G = I_p / (np.pi * a**2)  # 10^20 m^-3
    return n_G

# ITER Greenwald density
n_G_ITER = greenwald_density(I_p_ITER, a_ITER)
print(f"\nITER Greenwald density limit: n_G = {n_G_ITER:.2f} × 10^20 m^-3")

# Scan over current
I_p_scan_greenwald = np.linspace(5, 20, 50)
n_G_scan = [greenwald_density(I_p, a_ITER) for I_p in I_p_scan_greenwald]

plt.figure(figsize=(8, 5))
plt.plot(I_p_scan_greenwald, n_G_scan, 'm-', linewidth=2)
plt.xlabel('Plasma Current (MA)', fontsize=12)
plt.ylabel('Greenwald Density Limit (10²⁰ m⁻³)', fontsize=12)
plt.title('Greenwald Density Limit vs Plasma Current', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('greenwald_density.png', dpi=150)
plt.show()
```

### 8.6 ELM Energy Loss and Divertor Heat Flux

```python
def elm_heat_flux(W_ELM, A_wet, tau_ELM):
    """
    Estimate peak divertor heat flux from ELM.

    Parameters:
    W_ELM : energy expelled per ELM (MJ)
    A_wet : wetted area on divertor (m^2)
    tau_ELM : energy deposition timescale (ms)

    Returns:
    q_peak : peak heat flux (MW/m^2)
    """
    q_peak = W_ELM / (A_wet * tau_ELM * 1e-3)  # MW/m^2
    return q_peak

# ITER Type I ELM (unmitigated)
W_ELM_ITER = 1.0  # MJ (10% of pedestal energy ~ 10 MJ)
A_wet_ITER = 0.5  # m^2 (narrow wetted area)
tau_ELM_ITER = 0.5  # ms

q_peak_ITER = elm_heat_flux(W_ELM_ITER, A_wet_ITER, tau_ELM_ITER)
print(f"\nITER Type I ELM (unmitigated):")
print(f"W_ELM = {W_ELM_ITER} MJ, A_wet = {A_wet_ITER} m^2, τ_ELM = {tau_ELM_ITER} ms")
print(f"Peak heat flux: q_peak ~ {q_peak_ITER:.1f} MW/m^2")

# Mitigation: smaller, more frequent ELMs
W_ELM_mitigated = 0.1  # MJ
n_ELMs = 10  # 10x more frequent

q_peak_mitigated = elm_heat_flux(W_ELM_mitigated, A_wet_ITER, tau_ELM_ITER)
print(f"\nMitigated ELMs:")
print(f"W_ELM = {W_ELM_mitigated} MJ (10x smaller), frequency 10x higher")
print(f"Peak heat flux: q_peak ~ {q_peak_mitigated:.1f} MW/m^2")

# Scan over ELM size
W_ELM_scan = np.linspace(0.05, 2.0, 50)
q_peak_scan = [elm_heat_flux(W, A_wet_ITER, tau_ELM_ITER) for W in W_ELM_scan]

plt.figure(figsize=(8, 5))
plt.plot(W_ELM_scan, q_peak_scan, 'orange', linewidth=2)
plt.axhline(y=10, color='r', linestyle='--', linewidth=2, label='Material limit (~10 MW/m²)')
plt.xlabel('ELM Energy (MJ)', fontsize=12)
plt.ylabel('Peak Heat Flux (MW/m²)', fontsize=12)
plt.title('ELM Divertor Heat Flux vs ELM Energy', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elm_heat_flux.png', dpi=150)
plt.show()
```

### 8.7 Neoclassical Tearing Mode Island Width Evolution

```python
def ntm_island_evolution(w0, Delta_prime_bs, Delta_prime_class, r_s, tau_R, t_max, dt):
    """
    Evolve NTM island width using modified Rutherford equation.

    Parameters:
    w0 : initial island width (m)
    Delta_prime_bs : bootstrap drive (m^-1)
    Delta_prime_class : classical tearing stability parameter (m^-1)
    r_s : radius of rational surface (m)
    tau_R : resistive timescale (s)
    t_max : maximum time (s)
    dt : timestep (s)

    Returns:
    t_array : time array
    w_array : island width evolution
    """
    N_steps = int(t_max / dt)
    t_array = np.zeros(N_steps)
    w_array = np.zeros(N_steps)

    w = w0
    t = 0.0

    for i in range(N_steps):
        t_array[i] = t
        w_array[i] = w

        # Modified Rutherford equation: dw/dt = (r_s/τ_R) * (Δ'_class + L_qp/w^2)
        # Simplified: Δ'_bs ~ L_qp / w^2
        if w > 1e-6:  # Avoid singularity
            dw_dt = (r_s / tau_R) * (Delta_prime_class * w + Delta_prime_bs / w)
        else:
            dw_dt = 0.0

        w += dw_dt * dt
        t += dt

        # Stop if island saturates or decays
        if w < 0:
            w = 0
            break
        if w > 0.5:  # Cap at half minor radius
            break

    return t_array[:i+1], w_array[:i+1]

# Example: NTM at q=3/2 surface
r_s_ntm = 0.6  # m (60% of minor radius)
tau_R_ntm = 1.0  # s
Delta_prime_class_ntm = -0.5  # m^-1 (classically stable)
Delta_prime_bs_ntm = 0.001  # m (bootstrap drive parameter)

# Case 1: Small seed island (below threshold)
w0_small = 0.01  # m
t_small, w_small = ntm_island_evolution(w0_small, Delta_prime_bs_ntm, Delta_prime_class_ntm,
                                         r_s_ntm, tau_R_ntm, 10.0, 0.01)

# Case 2: Large seed island (above threshold)
w0_large = 0.05  # m
t_large, w_large = ntm_island_evolution(w0_large, Delta_prime_bs_ntm, Delta_prime_class_ntm,
                                         r_s_ntm, tau_R_ntm, 10.0, 0.01)

plt.figure(figsize=(10, 6))
plt.plot(t_small, w_small * 100, 'b-', linewidth=2, label=f'Small seed (w₀={w0_small*100:.1f} cm)')
plt.plot(t_large, w_large * 100, 'r-', linewidth=2, label=f'Large seed (w₀={w0_large*100:.1f} cm)')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Island Width (cm)', fontsize=12)
plt.title('NTM Island Width Evolution', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ntm_island_evolution.png', dpi=150)
plt.show()

print(f"\nNTM evolution:")
print(f"Small seed: final width = {w_small[-1]*100:.2f} cm (decays)")
print(f"Large seed: final width = {w_large[-1]*100:.2f} cm (grows)")
```

### 8.8 RFP Taylor State

```python
def rfp_taylor_state(r, a, mu_a):
    """
    Calculate RFP Taylor state magnetic field profiles.

    Parameters:
    r : radial coordinate (array)
    a : minor radius (m)
    mu_a : Taylor eigenvalue * a (dimensionless)

    Returns:
    B_z : toroidal field (normalized)
    B_theta : poloidal field (normalized)
    """
    from scipy.special import jv  # Bessel function

    x = mu_a * r / a
    B_z = jv(0, x)  # J_0
    B_theta = jv(1, x)  # J_1

    return B_z, B_theta

# RFP Taylor state
a_RFP = 0.5  # m
mu_a_RFP = 3.8  # First zero of J_0 is ~2.4, choose higher for reversal

r_RFP = np.linspace(0, a_RFP, 200)
B_z_RFP, B_theta_RFP = rfp_taylor_state(r_RFP, a_RFP, mu_a_RFP)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(r_RFP, B_z_RFP, 'b-', linewidth=2, label='$B_z$ (toroidal)')
ax1.plot(r_RFP, B_theta_RFP, 'r-', linewidth=2, label='$B_θ$ (poloidal)')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('Radius r (m)', fontsize=12)
ax1.set_ylabel('Magnetic Field (normalized)', fontsize=12)
ax1.set_title('RFP Taylor State: Magnetic Field Profiles', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Field line pitch
q_RFP = np.where(np.abs(B_theta_RFP) > 0.01, B_z_RFP / B_theta_RFP * a_RFP / 6.0, np.nan)
ax2.plot(r_RFP, q_RFP, 'g-', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('Radius r (m)', fontsize=12)
ax2.set_ylabel('Safety Factor q', fontsize=12)
ax2.set_title('RFP Safety Factor (approximate)', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-2, 2])

plt.tight_layout()
plt.savefig('rfp_taylor_state.png', dpi=150)
plt.show()

print(f"\nRFP Taylor state: μa = {mu_a_RFP}")
print(f"Field reversal at r/a ~ {r_RFP[B_z_RFP < 0][0] / a_RFP:.2f}")
```

## 9. 요약

이 강의는 자기 confinement 핵융합의 주요 MHD 측면을 다루었습니다:

1. **Tokamak 구성**: Toroidal + poloidal 장, safety factor, 플라즈마 전류
2. **Tokamak 평형**: Grad-Shafranov 방정식, Shafranov shift, beta 한계 (Troyon limit)
3. **주요 불안정성**:
   - **Sawteeth**: $q_0 < 1$, internal kink, Kadomtsev 재결합
   - **ELMs**: Peeling-ballooning modes, Type I/III, 완화 (RMP, pellet pacing, QH-mode)
   - **Disruptions**: 열적 quench, 전류 quench, runaway electrons, 완화 (MGI, SPI)
   - **NTMs**: Bootstrap 구동 섬 성장, ECCD 안정화
   - **RWMs**: Resistive wall modes, 회전 또는 feedback 제어 필요
4. **Stellarator**: 3D 외부 coils, 플라즈마 전류 없음, quasi-symmetry, disruptions 없음
5. **RFP**: 반전된 toroidal 장, Taylor relaxation, MHD dynamo

이러한 MHD 현상을 이해하고 제어하는 것은 실용적인 핵융합 에너지를 달성하는 데 필수적입니다. ITER는 반응로 관련 규모에서 이러한 개념들의 많은 것을 테스트할 것입니다.

## 연습 문제

1. **Troyon limit**: $I_p = 10$ MA, $a = 1.5$ m, $B_0 = 4$ T인 tokamak의 경우, Troyon limit ($\beta_N = 3$)를 사용하여 달성 가능한 최대 beta를 계산하세요. 해당 플라즈마 압력은 얼마입니까?

2. **Safety factor**: tokamak이 $R = 3$ m, $a = 1$ m, $B_0 = 5$ T, $I_p = 5$ MA를 가집니다. 평평한 전류 프로파일을 가정하여 가장자리 safety factor $q_a$를 계산하세요. 이 tokamak은 disruption 위험이 있습니까 ($q_a < 2$)?

3. **Sawtooth period**: $a = 1$ m, $T_e = 3$ keV, $n_e = 5 \times 10^{19}$ m$^{-3}$, $B = 3$ T인 플라즈마의 sawtooth 주기를 추정하세요. 제공된 Python 함수를 사용하세요. $T_e$가 두 배가 되면 주기는 어떻게 변합니까?

4. **Greenwald density**: ITER ($I_p = 15$ MA, $a = 2$ m)의 경우, Greenwald density limit는 $n_G = 1.19 \times 10^{20}$ m$^{-3}$입니다. 평균 밀도가 $n_e = 1.0 \times 10^{20}$ m$^{-3}$인 경우, Greenwald 분율 ($n_e / n_G$)은 얼마입니까? 플라즈마가 밀도 한계에 가깝습니까?

5. **ELM heat flux**: ELM이 $\tau_{ELM} = 1$ ms 동안 $A_{wet} = 1$ m$^2$의 wetted 면적에 걸쳐 $W_{ELM} = 0.5$ MJ를 배출합니다. 최대 열 플럭스를 계산하세요. 이것을 일반적인 재료 한계 10 MW/m$^2$와 비교하세요. 완화가 필요합니까?

6. **Disruption forces**: disruption 동안 플라즈마 전류가 $I_p = 5$ MA에서 $\tau_{CQ} = 100$ ms에 0으로 감쇠합니다. Python 함수를 사용하여 전류 quench 속도 $dI_p/dt$ 및 유도된 loop 전압을 추정하세요 ($R = 3$ m, $a = 1$ m 가정). 수직 힘의 크기는 얼마입니까?

7. **NTM threshold**: NTM이 $\Delta'_{bs} = 0.001$ m인 bootstrap 전류에 의해 구동되고 $\Delta'_{class} = -1$ m$^{-1}$인 classical tearing에 의해 감쇠됩니다. $w_{crit} \sim \sqrt{L_{qp}/|\Delta'_{class}|}$인 임계 섬 폭을 추정하세요. 여기서 $L_{qp} = \Delta'_{bs} / r_s$이고 $r_s = 0.5$ m입니다. NTM을 유발하는 데 필요한 seed 섬 크기는 얼마입니까?

8. **RFP field reversal**: $\mu a = 4.0$인 RFP의 경우, toroidal field $B_z = 0$ (reversal surface)인 반경을 찾으세요. Bessel 함수 $J_0(x)$를 사용하고 첫 번째 영점을 찾으세요. 결과를 $r/a$로 표현하세요.

9. **Stellarator comparison**: 핵융합 반응로를 위한 stellarators의 tokamaks 대비 세 가지 장점과 세 가지 단점을 나열하세요. 어떤 상황에서 stellarator가 선호될 수 있습니까?

10. **Beta optimization**: tokamak이 $\beta_N = 2.5$ (3.0의 Troyon limit 아래)에서 작동합니다. MHD 불안정성을 유발하지 않고 달성 가능한 beta를 증가시키는 두 가지 방법을 제안하세요. 평형 shaping, 전류 프로파일 제어 및 kinetic 안정화를 고려하세요.

---

**이전**: [Accretion Disk MHD](./12_Accretion_Disk_MHD.md) | **다음**: [Space Weather MHD](./14_Space_Weather.md)
