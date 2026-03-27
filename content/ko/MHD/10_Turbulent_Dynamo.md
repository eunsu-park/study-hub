# 10. 난류 Dynamo

## 학습 목표

이 레슨을 마치면 다음을 할 수 있어야 합니다:

- 소규모(변동) 및 대규모(평균장) 난류 dynamos 구별하기
- Kazantsev 이론과 난류에서 자기장의 kinematic 성장 이해하기
- Dynamo 작용에서 자기 Prandtl 수(Pm)의 역할 설명하기
- 자기 helicity 보존과 대규모 dynamo 성장에 대한 제약 분석하기
- 포화 메커니즘과 kinematic에서 dynamic 영역으로의 전환 설명하기
- MHD 난류를 위한 수치 시뮬레이션 접근법(DNS, LES) 이해하기
- 소규모 dynamo 성장과 helicity 진화 모델 구현하기

## 1. 난류 Dynamos 소개

### 1.1 난류 구동 자기장 생성

많은 천체물리학 환경—항성 내부, 강착 원반, 성간 매질, 은하단—에서 흐름은 고도로 **난류**입니다. 난류 dynamos는 층류 dynamos와 몇 가지 핵심 방식에서 다릅니다:

1. **광범위한 스케일 스펙트럼:** 난류는 에너지 주입 스케일 `L`부터 소산 스케일(Kolmogorov 스케일 `η_K` 또는 저항 스케일 `η_R`)까지 광범위한 길이 스케일에 걸친 운동을 포함합니다.

2. **확률론적 특성:** 난류 흐름은 혼돈적이고 시간 의존적이어서 통계적 설명이 필요합니다.

3. **다중 dynamo 메커니즘:** **소규모 dynamo**(변동 장)와 **대규모 dynamo**(평균 장) 모두 동시에 작동할 수 있습니다.

**핵심 질문:**
- Dynamo 시작을 위한 임계 자기 Reynolds 수 `Rm_c`는 무엇인가?
- 자기장은 어떻게 포화되는가?
- 자기장의 구조는 무엇인가(간헐적, 필라멘트형, 평활)?
- 자기 에너지 스펙트럼 `E_B(k)`는 운동 에너지 스펙트럼 `E_K(k)`와 어떻게 비교되는가?

### 1.2 소규모 대 대규모 Dynamos

**소규모(변동) dynamo:**
- 난류 강제 스케일과 **비슷하거나 더 작은** 스케일에서 자기장 증폭
- 난류 연신과 접기에 의해 구동
- Helicity나 대규모 전단 불필요
- 얽힌, 간헐적 자기 구조 생성
- 관련성: ISM, 은하단, 초기 우주

**대규모(평균장) dynamo:**
- 난류 강제 스케일**보다 큰** 스케일에서 자기장 생성
- Helicity(예: 회전과 성층으로부터) 또는 대규모 전단 필요
- 결맞는, 조직화된 장 생성(예: 은하 나선, 태양 쌍극)
- 자기 helicity 보존에 의해 제약
- 관련성: 은하, 항성, 행성

둘 다 동시에 작동할 수 있지만, 대규모 dynamo가 더 제약되고 느립니다.

## 2. 소규모 Dynamo 이론

### 2.1 Kazantsev 이론 (1968)

**Kazantsev**는 무작위, 짧은 상관(시간에서) 속도장에서 자기장의 kinematic 성장에 대한 통계 이론을 개발했습니다.

**가정:**
- 속도장 `v(x,t)`는 Gaussian 무작위장
- 상관 시간 `τ_c ≪ τ_η = ℓ²/η` (짧은 상관, 또는 시간에서 델타 상관)
- 속도 상관 함수:

```
⟨v_i(x,t) v_j(x',t')⟩ = δ(t - t') K_{ij}(x - x')
```

- 등방성, 균질 난류

**유도 방정식:**

```
∂B/∂t = ∇×(v×B) + η∇²B
```

Kinematic 영역에서, `v`는 주어진 것입니다.

**자기장 상관자:**

2점 자기 상관 텐서를 정의:

```
M_{ij}(r,t) = ⟨B_i(x,t) B_j(x+r,t)⟩
```

등방성 난류에서, `M_{ij}`는 `|r|`에만 의존하며 다음으로 분해할 수 있습니다:

```
M_{ij}(r,t) = M_N(r,t) (δ_{ij} - r_i r_j / r²) + M_L(r,t) r_i r_j / r²
```

여기서 `M_N`과 `M_L`은 횡방향 및 종방향 상관 함수입니다.

**Kazantsev 방정식:**

스칼라 상관 함수 `M(r,t) = ⟨B(x,t)·B(x+r,t)⟩`에 대해, Kazantsev는 `r`-공간에서 확산형 방정식을 유도했습니다(델타 상관 속도의 경우):

```
∂M/∂t = (1/r^{d-1}) ∂/∂r [r^{d-1} (D(r) ∂M/∂r - v(r) M)]
```

여기서:
- `d`는 공간 차원(보통 `d=3`)
- `D(r)`는 `r`-공간의 확산 계수, 속도 상관과 관련
- `v(r)`는 drift 항

짧은 상관 시간과 `r → 0` 한계에서:

```
D(r) ≈ D₀ r²
v(r) ≈ v₀ r
```

여기서 `D₀`와 `v₀`는 속도 스펙트럼에 의존하는 상수입니다.

**지수 성장:**

해 `M(r,t) ~ exp(γt) m(r)`를 구합니다. `r ≪ η_K`(작은 스케일)의 경우, 해는:

```
γ ~ (u_rms / ℓ) × (Rm / Rm_c)^{1/2}   for Rm > Rm_c
```

여기서:
- `ℓ`은 난류 상관 스케일
- `u_rms`는 RMS 속도
- `Rm = u_rms ℓ / η`
- `Rm_c`는 임계 자기 Reynolds 수(일반적으로 `Rm_c ~ 50-200`)

**자기 에너지 스펙트럼:**

Kinematic 성장 단계에서, 작은 스케일의 자기 에너지 스펙트럼은:

```
E_B(k) ∝ k^{3/2}   (Kazantsev spectrum)
```

이것은 Kolmogorov 운동 스펙트럼 `E_K(k) ∝ k^{-5/3}`보다 **더 가파르며**, 자기 에너지가 작은 스케일에 집중됨(간헐적 구조)을 나타냅니다.

### 2.2 임계 자기 Reynolds 수

소규모 dynamo 시작은 다음을 요구합니다:

```
Rm > Rm_c
```

**Pm에 대한 의존성:**

임계 `Rm_c`는 **자기 Prandtl 수**에 의존합니다:

```
Pm = ν / η
```

여기서:
- `ν`는 운동 점성계수
- `η`는 자기 확산도

수치 시뮬레이션(예: Schekochihin et al., 2004; Brandenburg & Subramanian, 2005)은 다음을 발견합니다:

- **높은 Pm 영역(`Pm ≫ 1`):** `Rm_c ~ 100` (Pm에 약하게 의존)
  - 점성 차단이 저항 차단 아래: `η_K ≪ η_R`
  - Dynamo가 `η_K`와 `η_R` 사이의 스케일에서 작동

- **낮은 Pm 영역(`Pm ≪ 1`):** Pm이 감소함에 따라 `Rm_c` 증가
  - 저항 차단이 점성 차단 아래: `η_R ≪ η_K`
  - Dynamo가 작은 스케일에서 저항 확산에 의해 억제됨
  - 확산을 극복하기 위해 속도장에 더 많은 파워 필요

- **Pm ~ 1:** `Rm_c ~ 50-100`

**천체물리학적 관련성:**
- **항성:** `Pm ~ 10^{-5} - 10^{-7}` (매우 작음, 시뮬레이션 어려움)
- **ISM, 은하단:** `Pm ≫ 1` (dynamo 더 쉬움)
- **실험실 플라즈마:** `Pm ~ 10^{-6}` (도전적)

### 2.3 연신 메커니즘

소규모 dynamo의 근본적 구동력은 난류 변형률에 의한 **자기력선의 연신**입니다.

**변형률 텐서:**

```
S_{ij} = (1/2)(∂v_i/∂x_j + ∂v_j/∂x_i)
```

**자기장 연신:**

`B`와 정렬된 자기력선 요소 `δℓ`의 진화는 다음을 따릅니다:

```
d(ln|δℓ|)/dt = S_{ij} (δℓ_i δℓ_j) / |δℓ|²
```

양의 **Lyapunov 지수** `λ > 0`를 가진 혼돈 흐름의 경우, 선 요소가 지수적으로 연신됩니다:

```
|δℓ(t)| ~ exp(λt)
```

자기장 강도가 `B ~ B₀ (|δℓ| / |δℓ₀|)` (flux freezing)로 스케일링되므로:

```
B(t) ~ B₀ exp(λt)
```

이것이 소규모 dynamo의 kinematic 성장입니다.

**Anti-dynamo 제약:**

그러나 연신만으로는 충분하지 않습니다. 장선이 흐름과 정렬될 수도 있어(`S_{ij}`의 주 고유벡터를 따라), **포화** 또는 **억제**로 이어집니다. Dynamo는 다음을 요구합니다:

1. **지속적인 연신:** 흐름이 지속적으로 새로운 장 방향을 생성해야 함
2. **접기:** 재연결 또는 위상 재배열이 한 방향으로의 무한정 연신을 방지

### 2.4 포화와 비선형 영역

**Kinematic 영역**에서, 자기장은 지수적으로 성장합니다:

```
B²(t) ~ B₀² exp(2γt)
```

결국, Lorentz 힘이 중요해집니다:

```
J × B / (ρv·∇v) ~ B² / (μ₀ρv²) ~ 1
```

이것이 **비선형(dynamic) 영역**으로의 전환을 표시합니다.

**포화 수준:**

차원 분석은 다음을 시사합니다:

```
B_sat² / (2μ₀) ~ ε_B × (1/2) ρ v²
```

여기서 `ε_B`는 포화에서의 자기-운동 에너지 비입니다.

시뮬레이션은 다음을 발견합니다:
- **높은 Pm:** `ε_B ~ 0.1 - 1` (거의 동등분배)
- **낮은 Pm:** `ε_B ≪ 1` (동등분배 이하, 작은 스케일이 점성에 의해 억제되기 때문)

**장 구조:**

포화에서:
- 자기장이 고도로 **간헐적**(시트, 필라멘트에 집중)
- **자기 Reynolds 응력** `B_iB_j / μ₀`가 속도에 역반응
- 작은 스케일에서 난류 운동 에너지의 효과적 감소
- 자기 에너지 스펙트럼 평탄화: `E_B(k) ~ k^{-1}` to `k^{-3/2}` (Kazantsev보다 덜 가파름)

## 3. 난류에서의 대규모 Dynamo

### 3.1 자기 Helicity의 역 캐스케이드

소규모 dynamo가 작은 스케일에서 장을 증폭하는 반면, **대규모 dynamo**는 강제보다 큰 스케일에서 결맞는 장을 생성합니다.

**핵심 개념:** **자기 helicity**는 보존량(이상 MHD에서)으로 작용하여 제약을 제공합니다.

**자기 helicity:**

```
H_B = ∫ A·B dV
```

여기서 `B = ∇×A`입니다.

**Helicity 보존:**

이상 MHD(`η → 0`)에서, helicity는 보존됩니다:

```
dH_B/dt = 0   (ideal MHD)
```

유한 저항도에서:

```
dH_B/dt = -2η ∫ J·B dV ≈ -2η/ℓ² H_B
```

따라서 helicity는 저항 시간 척도 `τ_η = ℓ²/η`에서 붕괴합니다.

**역 캐스케이드:**

3D MHD 난류에서, 자기 helicity는 **대규모로 캐스케이드**되는 경향이 있고(역 캐스케이드), 자기 에너지는 작은 스케일로 캐스케이드됩니다(순방향 캐스케이드).

**Dynamo에 대한 함의:**

- 소규모 dynamo가 **소규모 helicity**를 가진 소규모 자기장을 생성
- 자기 helicity 역 캐스케이드 → **대규모 helicity** 축적
- 대규모 helicity → 결맞는 대규모 자기장

이 메커니즘은 평균장 언어에서 때때로 **α²-dynamo**라고 불립니다.

### 3.2 대규모 Dynamo에 대한 Helicity 제약

**문제:** 닫힌(주기적 또는 제한된) 시스템에서, 총 자기 helicity가 보존됩니다. 대규모 장이 성장함에 따라, 대규모 helicity를 축적합니다. 총 helicity를 보존하기 위해, 소규모 helicity가 **반대 부호**로 성장해야 합니다. 이 소규모 helicity는 재앙적 quenching을 통해 α-효과를 억제합니다.

**재앙적 α-quenching 재고:**

평균장 이론에서, α-효과는 대규모 장에 의해 quench됩니다:

```
α(B) = α₀ / (1 + Rm (B/B_eq)²)
```

높은 `Rm`에서, 이것은 `α ~ α₀/Rm → 0`로 이어져 dynamo를 차단합니다.

**해결책: Helicity 플럭스**

경계가 **열려** 있으면(예: 항성 표면, 은하 헤일로), 자기 helicity가 경계를 통해 **탈출**할 수 있습니다:

```
dH_B/dt = -2η ∫ J·B dV - ∫ (E × A)·dS
```

여기서 표면 적분은 부피 밖으로의 helicity 플럭스를 나타냅니다.

Helicity 플럭스가 있으면, 제약이 완화됩니다:
- 소규모 helicity가 방출됨
- 대규모 장이 재앙적 quenching 없이 성장 가능
- 포화는 helicity 생성 ~ helicity 플럭스 + 저항 소산일 때 발생

**천체물리학적 응용:**
- **태양 dynamo:** 태양풍과 코로나 질량 방출에 의해 운반되는 Helicity
- **은하 dynamo:** 은하풍을 통해 은하간 매질로의 helicity 탈출
- **강착 원반 dynamo:** 안쪽으로 이류되거나 유출에서 방출되는 Helicity

### 3.3 평균장 난류 Dynamo

**대규모 장 진화:**

평균장 이론을 상기:

```
∂⟨B⟩/∂t = ∇×(⟨v⟩×⟨B⟩) + ∇×(α⟨B⟩) + (η + β)∇²⟨B⟩
```

여기서:
- `α ~ -(1/3)τ_c⟨u·(∇×u)⟩` (helicity 효과)
- `β ~ (1/3)τ_c u²` (난류 확산도)

**Dynamo 수:**

크기 `L`의 영역에서 α² dynamo의 경우:

```
D_α = α L / (η + β)
```

Dynamo 시작: `|D_α| ≳ 10`.

**Helicity 주입:**

회전하는, 성층화된 난류(예: 항성 대류 영역)에서:
- **Coriolis 힘** + **밀도 성층** → cyclonic 소용돌이
- Cyclonic 소용돌이가 순 helicity를 가짐: `⟨u·(∇×u)⟩ ≠ 0`
- Helicity의 부호는 반구에 의존(북과 남에서 반대)

**성장률:**

Kinematic 영역에서:

```
γ ~ α² / (η_eff L)
```

소규모 dynamo 성장률 `γ ~ u/ℓ`보다 훨씬 **느림**.

**포화:**

대규모 dynamo는 다음 때 포화됩니다:
- Lorentz 힘이 흐름을 수정(α 감소)
- Helicity 균형: 생성 ≈ 플럭스 + 소산

## 4. 자기 Prandtl 수 효과

### 4.1 정의와 영역

**자기 Prandtl 수:**

```
Pm = ν / η = (운동량의 분자 확산) / (자기 확산)
```

**Reynolds 수:**

```
Re = UL / ν    (흐름의 Reynolds 수)
Rm = UL / η    (자기 Reynolds 수)

Pm = Rm / Re
```

**천체물리학적 값:**

- **항성 내부:** `Pm ~ 10^{-7} - 10^{-5}`
  - 높은 전도도(낮은 `η`), 낮은 점성(분자 의미의 높은 `ν`, 하지만 난류 `ν_t`는 클 수 있음)
- **액체 금속(실험):** `Pm ~ 10^{-6} - 10^{-5}`
- **성간 매질:** `Pm ≫ 1` (무충돌 플라즈마, 자기 확산이 점성을 지배)
- **은하단:** `Pm ≫ 1`

**스케일 분리:**

- **Kolmogorov 스케일:** `η_K = (ν³/ε)^{1/4}` (운동 에너지가 소산되는 최소 스케일)
- **저항 스케일:** `η_R = (η³/ε)^{1/4}` (자기 에너지가 소산되는 최소 스케일)

비:

```
η_R / η_K = Pm^{-3/4}
```

- `Pm ≫ 1`: `η_R ≪ η_K` (더 작은 스케일에서 자기 소산)
- `Pm ≪ 1`: `η_R ≫ η_K` (더 작은 스케일에서 점성 소산)

### 4.2 높은 Pm 영역의 Dynamo

**특성:**
- 저항 스케일이 점성 스케일 아래: `η_R ≪ η_K`
- 자기장이 `η_K`와 `η_R` 사이의 스케일에서 여기될 수 있음
- 자기장을 위한 **넓은 관성 범위**

**Dynamo 메커니즘:**
- 소규모 dynamo가 효율적으로 작동
- 임계 `Rm_c ~ 100` (상대적으로 낮음)
- 동등분배 근처 포화: `B² / (2μ₀) ~ ρv²/2`

**응용:**
- **성간 매질:** 난류 구름에서 자기장 증폭
- **은하단:** ICM 난류가 `μG` 장 생성

### 4.3 낮은 Pm 영역의 Dynamo

**특성:**
- 점성 스케일이 저항 스케일 아래: `η_K ≪ η_R`
- 자기장이 최소 속도 스케일에 도달하기 전에 소산
- 자기장을 위한 **좁은 관성 범위**

**Dynamo 메커니즘:**
- 소규모 dynamo가 억제됨(더 높은 `Rm_c`)
- 포화가 **동등분배 이하**: `B²/(2μ₀) ≪ ρv²/2`
- 비가 `B²/(μ₀ρv²) ~ Pm^{1/2}`로 스케일링(Schekochihin et al.)

**도전:**
- 낮은 Pm에서의 수치 시뮬레이션은 `η_K`와 `η_R` 둘 다의 해상도 필요 → 계산적으로 비용이 큼
- 대부분의 천체물리학 시스템이 `Pm ≪ 1`을 가지지만, 시뮬레이션은 종종 `Pm ~ 1` 이상 사용

**응용:**
- **항성 dynamos:** 진정한 `Pm ~ 10^{-6}`, 하지만 효과적 난류 `Pm_t`는 1에 더 가까울 수 있음
- **액체 금속 실험:** VKS (von Kármán Sodium) 실험, Riga dynamo

## 5. 난류 Dynamos의 수치 시뮬레이션

### 5.1 직접 수치 시뮬레이션(DNS)

**DNS**는 에너지 주입 스케일 `L`부터 소산 스케일(`η_K`와 `η_R`)까지 모든 스케일을 해상합니다.

**MHD 방정식(비압축성):**

```
∂v/∂t + v·∇v = -∇p + J×B + ν∇²v + f
∂B/∂t = ∇×(v×B) + η∇²B
∇·v = 0
∇·B = 0
```

여기서 `f`는 강제 항(대규모에서 난류를 구동)입니다.

**공간 해상도 요구사항:**

소산 스케일을 해상하려면:

```
N_x ≥ (L / η_K)  for velocity
N_x ≥ (L / η_R)  for magnetic field
```

`Re = 10⁴`와 `Pm = 1`의 경우:

```
η_K ~ L / Re^{3/4} ~ L / 100
η_R ~ L / Rm^{3/4} ~ L / 100

N_x ≥ 100  →  N_total = 100³ = 10^6 grid points (3D)
```

더 높은 `Re` 또는 낮은 `Pm`의 경우, 해상도 요구사항이 폭발합니다.

**스펙트럼 방법:**

일반적으로 **Fourier pseudospectral** 방법 사용:

1. Fourier 공간에서 장 표현: `v(x) ↔ v̂(k)`
2. 실공간에서 비선형 항 `v·∇v`, `v×B` 계산(FFT를 통해)
3. Fourier 공간에서 도함수 계산: `∇ → ik`
4. `∇·v = 0` 강제: solenoidal 부공간으로 투영

**시간 적분:**

- **명시적(RK3, RK4):** 간단하지만 CFL 제약: `Δt ≤ C Δx / |v|_max`
- **암시적(Crank-Nicolson):** 확산 항에 대해, 더 큰 `Δt` 허용
- **IMEX (Implicit-Explicit):** 이류를 명시적으로, 확산을 암시적으로 처리

### 5.2 Large Eddy Simulation (LES)

매우 높은 Reynolds 수(DNS 도달 범위를 넘어서)의 경우, **LES** 사용:

**개념:**
- 큰 스케일만 해상(일부 차단 `k_c`까지)
- 해상되지 않은 작은 스케일의 효과 모델링(서브그리드 스케일, SGS)

**필터링:**

폭 `Δ`의 공간 필터 적용:

```
⟨v⟩(x) = ∫ G(x - x', Δ) v(x') dx'
```

여기서 `G`는 필터 커널(예: Gaussian, box, spectral cutoff)입니다.

**필터링된 MHD 방정식:**

```
∂⟨v⟩/∂t + ⟨v⟩·∇⟨v⟩ = -∇⟨p⟩ + ⟨J⟩×⟨B⟩ + ν∇²⟨v⟩ - ∇·τ_SGS
∂⟨B⟩/∂t = ∇×(⟨v⟩×⟨B⟩) + η∇²⟨B⟩ + ∇×ε_SGS
```

여기서:
- `τ_SGS = ⟨vv⟩ - ⟨v⟩⟨v⟩` (SGS 응력)
- `ε_SGS = ⟨v×B⟩ - ⟨v⟩×⟨B⟩` (SGS EMF)

**서브그리드 모델:**

1. **Eddy 점성/저항도:**

```
τ_SGS ≈ -ν_t(∇⟨v⟩ + (∇⟨v⟩)^T)
ε_SGS ≈ -η_t ∇×⟨B⟩
```

여기서 `ν_t`, `η_t`는 난류 점성/저항도(예: Smagorinsky 모델)입니다.

2. **Gradient 모델:**

```
τ_SGS ≈ C Δ² ∇⟨v⟩·∇⟨v⟩
```

3. **Dynamic 모델:** 해상된 스케일로부터 `C`를 동적으로 계산(Germano identity).

**MHD LES의 도전:**
- SGS 자기장이 강한 역반응을 가질 수 있음(작은 스케일의 dynamo)
- 표준 eddy 점성 모델이 helicity의 역 캐스케이드를 포착하지 못할 수 있음
- 활발한 연구 분야

### 5.3 강제와 경계 조건

**강제:**

통계적으로 정상 난류를 유지하기 위해, 대규모에서 에너지 주입:

```
f(x,t) = F(k, t)  for k in band [k_min, k_max]
```

일반적 방식:
- **확률론적 강제:** 무작위 위상, Gaussian 통계
- **ABC 강제:** Arnold-Beltrami-Childress 흐름(나선형)
- **속도 강제:** 특정 모드에 대해 `|v̂(k)|` 고정, 위상 무작위화

**경계 조건:**

- **주기적:** 가장 간단, 많은 연구에 사용
  - 문제: helicity 보존(플럭스 없음), 재앙적 quenching
- **열림(유출):** Helicity 플럭스 허용
  - 구현: 경계에서 외삽 또는 제로 구배
- **전도 벽:** `B_n` 연속, `E_t = 0` (또는 접선 방향으로 `v×B = 0`)

### 5.4 분석 도구

**에너지 스펙트럼:**

```
E_K(k) = (1/2) Σ_{|k'| ≈ k} |v̂(k')|²
E_B(k) = (1/2μ₀) Σ_{|k'| ≈ k} |B̂(k')|²
```

**Helicity 스펙트럼:**

```
H_K(k) = Σ_{|k'| ≈ k} Re(v̂*(k')·(ik' × v̂(k')))
H_B(k) = Σ_{|k'| ≈ k} Re(Â*(k')·B̂(k'))
```

**구조 함수:**

```
S_p(r) = ⟨|v(x+r) - v(x)|^p⟩
```

간헐성 측정: Kolmogorov의 경우, `S_p(r) ~ r^{ζ_p}`이고 `ζ_p = p/3`; 이탈은 간헐성을 나타냄.

**자기장 PDF:**

```
P(B) = probability distribution of field strength
```

일반적으로 **비Gaussian**, 지수 또는 stretched-지수 꼬리(간헐성).

## 6. 난류 Dynamo의 응용

### 6.1 성간 매질(ISM)

**맥락:**
- ISM은 고도로 난류: 초신성 폭발, 항성풍, 열 불안정성
- 관측된 자기장: `B ~ μG`
- `Pm ≫ 1` (무충돌 플라즈마)

**Dynamo 메커니즘:**
- **소규모 dynamo:** 씨앗 장을 `μG` 수준으로 증폭
- 난류 운동 에너지와 거의 동등분배에서 포화
- 자기장 구조: 필라멘트형, 간헐적

**관측 테스트:**
- Faraday 회전 측정(RM): 시선을 따라 자기장 탐사
- Synchrotron 방출: 총 강도와 편광
- Zeeman 분리: 직접 B 측정(밀집 영역에 제한)

**수치 발견:**
- 소규모 dynamo가 일반적 ISM 난류에 대해 `B ~ 3-10 μG`에서 포화
- 나선 팔, 별 형성 영역의 관측과 일치

### 6.2 은하단

**맥락:**
- 은하단간 매질(ICM): 뜨겁고 희박한 플라즈마
- 합병, AGN 피드백에 의해 구동되는 난류
- 관측된 자기장: `B ~ μG` (RM, 전파 헤일로로부터)

**Dynamo 메커니즘:**
- 소규모 dynamo가 은하단 형성 동안 씨앗 장 증폭
- `Pm ≫ 1` (무충돌)
- 빠른 성장: `τ_dyn ~ Gyr`

**도전:**
- 전도가 소규모 변동을 억제할 수 있음(Braginski 점성)
- 우주선이 dynamo에 영향을 미칠 수 있음

**시뮬레이션:**
- Vazza et al., Miniati, Ryu: MHD를 가진 은하단 형성 시뮬레이션
- Dynamo로부터 `B ~ 0.1 - 1 μG` 발견

### 6.3 강착 원반

**맥락:**
- Magnetorotational 불안정성(MRI)이 난류 생성(Lesson 12 참조)
- 난류 dynamo가 자기장 증폭 및 유지

**Dynamo 메커니즘:**
- 소규모(MRI 난류)와 대규모(MRI 구동 α-효과로부터의 dynamo) 모두
- 원반을 관통하는 수직 장이 증폭될 수 있음
- 원시행성 원반(dead zones)에서 `Pm ≪ 1`, 뜨거운 원반에서 `Pm ~ 1`

**포화:**
- 자기 응력: `⟨B_rB_φ⟩/μ₀ ~ α ⟨p⟩`이고 `α ~ 0.01 - 0.1`
- `B ~ √(αp)` → 열 압력 이하에 해당

**관측 함의:**
- 제트 발사: 대규모 극성 장 필요(dynamo + 이류)
- 원반 바람: 환상 장의 압력

### 6.4 초기 우주

**맥락:**
- 원시 플라즈마의 씨앗 자기장
- 상전이, 원시 밀도 변동으로부터의 난류

**Dynamo:**
- 복사 시대(재결합 전) 동안 소규모 dynamo
- 증폭 인자: 약한 씨앗 장으로부터 `10^{30}` 도달 가능
- 자기장 결맞음 길이: 지평선 또는 감쇠 스케일에 의해 제한

**관련성:**
- 공백과 높은 적색편이 은하에서 관측된 `nG` 장 설명
- 구조 형성에 영향(자기 압력 지지)

## 7. Python 구현

### 7.1 Kazantsev 스펙트럼 모델

```python
import numpy as np
import matplotlib.pyplot as plt

def kazantsev_spectrum():
    """
    Model magnetic energy spectrum in small-scale dynamo.

    Kazantsev prediction: E_B(k) ∝ k^{3/2} in kinematic regime.
    """
    # Wavenumber range
    k = np.logspace(-1, 2, 100)

    # Kinetic energy spectrum (Kolmogorov)
    E_K = k**(-5/3)

    # Magnetic energy spectrum (Kazantsev kinematic)
    E_B_kinematic = k**(3/2)

    # Magnetic energy spectrum (saturated, example: k^{-3/2})
    E_B_saturated = k**(-3/2)

    # Normalize
    E_K /= E_K[len(E_K)//2]
    E_B_kinematic /= E_B_kinematic[len(E_B_kinematic)//2]
    E_B_saturated /= E_B_saturated[len(E_B_saturated)//2]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(k, E_K, 'b-', linewidth=2, label='$E_K(k) \propto k^{-5/3}$ (Kolmogorov)')
    plt.loglog(k, E_B_kinematic, 'r--', linewidth=2, label='$E_B(k) \propto k^{3/2}$ (Kazantsev kinematic)')
    plt.loglog(k, E_B_saturated, 'g-.', linewidth=2, label='$E_B(k) \propto k^{-3/2}$ (Saturated)')

    plt.xlabel('Wavenumber $k$', fontsize=14)
    plt.ylabel('Energy spectrum $E(k)$', fontsize=14)
    plt.title('Kazantsev Spectrum: Small-Scale Dynamo', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('kazantsev_spectrum.png', dpi=150)
    plt.show()

kazantsev_spectrum()
```

### 7.2 소규모 Dynamo 성장 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def small_scale_dynamo_growth():
    """
    Simulate kinematic growth of magnetic energy in small-scale dynamo.

    Model:
      dE_B/dt = 2γ E_B - (E_B/τ_η)

    where:
      γ = growth rate from turbulent stretching
      τ_η = resistive dissipation timescale
    """
    # Parameters
    u_rms = 1.0       # RMS velocity
    ell = 1.0         # Correlation scale
    eta_vals = [0.001, 0.005, 0.01, 0.02]  # Magnetic diffusivity

    # Time array
    t = np.linspace(0, 10, 1000)

    plt.figure(figsize=(12, 6))

    for eta in eta_vals:
        Rm = u_rms * ell / eta
        # Rm_c ~ 60은 저항성 확산이 난류 신장을 이기는 임계값이다.
        # 이 값은 소규모 다이나모 작용에 필요한 최소 자기 레이놀즈 수를 결정하며,
        # 카잔체프(Kazantsev) 이론과 직접 수치 시뮬레이션(DNS)에 의해 확인되었다.
        Rm_c = 60  # Critical magnetic Reynolds number

        if Rm > Rm_c:
            # Growth rate (simplified Kazantsev)
            # γ ∝ √(Rm/Rm_c - 1) — (Rm - Rm_c)에 대한 제곱근 의존성은
            # 초임계 분기(supercritical bifurcation)를 반영한다: 임계값 바로 위에서
            # 다이나모는 약하게 불안정하며(γ 작음), Rm이 Rm_c를 충분히 초과해야
            # 비로소 강력한 증폭이 가능하다.
            gamma = (u_rms / ell) * np.sqrt((Rm - Rm_c) / Rm_c) * 0.1
        else:
            gamma = 0  # No dynamo

        # Resistive timescale
        # τ_η = ℓ²/η는 저항성이 상관 길이 전체에 걸쳐 자기장을 확산시키는 데
        # 걸리는 시간이다. 이 손실항을 포함함으로써 모델이 증폭과 옴(Ohmic) 감쇠
        # 사이의 경쟁을 올바르게 반영한다.
        tau_eta = ell**2 / eta

        # Differential equation: dE_B/dt = 2*gamma*E_B - E_B/tau_eta
        def dE_dt(E, t):
            return 2 * gamma * E - E / tau_eta

        # Initial condition
        E0 = 1e-6

        # Solve ODE
        E_B = odeint(dE_dt, E0, t)

        # Plot
        plt.semilogy(t, E_B, linewidth=2, label=f'Rm={Rm:.1f}, γ={gamma:.3f}')

    plt.xlabel('Time $t$ (in $\ell/u_{rms}$)', fontsize=14)
    plt.ylabel('Magnetic Energy $E_B$', fontsize=14)
    plt.title('Small-Scale Dynamo: Kinematic Growth', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('small_scale_dynamo_growth.png', dpi=150)
    plt.show()

small_scale_dynamo_growth()
```

### 7.3 자기 Helicity 진화

```python
import numpy as np
import matplotlib.pyplot as plt

def magnetic_helicity_evolution():
    """
    Simulate evolution of magnetic helicity in a turbulent dynamo.

    Model helicity production, dissipation, and flux:
      dH_B/dt = Production - Dissipation - Flux
    """
    # Parameters
    L = 1.0           # Domain size
    eta = 0.01        # Magnetic diffusivity
    alpha0 = 0.1      # Alpha effect (helicity production rate coefficient)
    flux_rate = 0.05  # Helicity flux rate (if boundaries are open)

    # Time array
    t = np.linspace(0, 100, 1000)
    dt = t[1] - t[0]

    # Two scenarios: closed vs open boundaries
    # 닫힌/열린 경계 비교는 핵심 진단이다: 닫힌 상자에서는 총 헬리시티가
    # 저항성 감쇠를 제외하면 보존되므로 대규모 자기장이 파국적으로 포화되는 반면,
    # 열린 경계는 소규모 헬리시티가 빠져나갈 수 있어 소광(quenching)을 완화하고
    # 자기장이 더 성장하도록 허용한다.
    scenarios = {
        'Closed (no flux)': 0.0,
        'Open (with flux)': flux_rate
    }

    plt.figure(figsize=(12, 8))

    for i, (label, flux) in enumerate(scenarios.items()):
        # Initialize
        H_B = np.zeros(len(t))
        B_rms = np.zeros(len(t))

        H_B[0] = 0.0
        B_rms[0] = 0.01

        # Time evolution
        for n in range(len(t) - 1):
            # Helicity production (from alpha effect and field growth)
            # α₀ B²는 헬리시티 난류가 대규모 자기장에 헬리시티를 주입하는 속도에 대한
            # 평균장(mean-field) 근사, 즉 ⟨u × b⟩ · B̄이며 α-효과 순환의 일부이다.
            production = alpha0 * B_rms[n]**2

            # Resistive dissipation
            # 2η/L² × H_B는 옴(Ohmic) 헬리시티 감쇠율이다. 계수 2는
            # dH_B/dt = -2η ∫ J·B dV ≈ -2η k² H_B 관계에서 유래하며,
            # 여기서 k ~ 1/L은 대규모 자기장의 지배 파수이다.
            dissipation = (2 * eta / L**2) * H_B[n]

            # Helicity flux (for open boundaries)
            flux_term = flux * H_B[n]

            # Update helicity
            dH_dt = production - dissipation - flux_term
            H_B[n+1] = H_B[n] + dt * dH_dt

            # Simple model for field growth with helicity constraint
            # α-quenching: α_eff = α0 / (1 + |H_B| / H_sat)
            # 이 소광 공식은 역반응(back-reaction)을 포착한다: 성장하는 대규모
            # 헬리시티는 반대 부호의 소규모 헬리시티를 축적하여(총량 보존) α-효과를
            # 억제한다 — 이것이 파국적 소광(catastrophic quenching)의 메커니즘이다.
            H_sat = 0.1
            alpha_eff = alpha0 / (1 + np.abs(H_B[n]) / H_sat)

            # Field growth (simplified)
            gamma = alpha_eff - eta / L**2
            dB_dt = gamma * B_rms[n]
            B_rms[n+1] = B_rms[n] + dt * dB_dt

        # Plot
        plt.subplot(2, 1, 1)
        plt.plot(t, H_B, linewidth=2, label=label)

        plt.subplot(2, 1, 2)
        plt.plot(t, B_rms, linewidth=2, label=label)

    plt.subplot(2, 1, 1)
    plt.xlabel('Time $t$', fontsize=14)
    plt.ylabel('Magnetic Helicity $H_B$', fontsize=14)
    plt.title('Magnetic Helicity Evolution', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.xlabel('Time $t$', fontsize=14)
    plt.ylabel('RMS Magnetic Field $B_{rms}$', fontsize=14)
    plt.title('Magnetic Field Evolution', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('helicity_evolution.png', dpi=150)
    plt.show()

magnetic_helicity_evolution()
```

### 7.4 Dynamo가 있는 난류 캐스케이드

```python
import numpy as np
import matplotlib.pyplot as plt

def turbulent_cascade_with_dynamo():
    """
    Simulate energy cascade in MHD turbulence with dynamo.

    Model shell-averaged energy equations:
      dE_K(k)/dt = T_K(k) + F_K(k) - ν k² E_K(k) - M(k)
      dE_B(k)/dt = T_B(k) + Dynamo(k) - η k² E_B(k) + M(k)

    where:
      T_K, T_B: nonlinear transfer (cascade)
      F_K: forcing
      M: magnetic-kinetic energy exchange
      Dynamo: energy input from stretching
    """
    # Wavenumber bins (logarithmic)
    # 로그 간격 빈을 사용하는 이유는 캐스케이드가 k에서 수십 배에 걸쳐 있기 때문이다.
    # log(k)에서 등간격으로 배치하면 관성 범위(inertial range)의 각 십진 배(decade)가
    # 동일한 수의 쉘로 표현되어 균일한 분해능을 보장한다.
    N_bins = 20
    k = np.logspace(0, 2, N_bins)
    dk = np.diff(np.log(k))
    dk = np.append(dk, dk[-1])

    # Parameters
    nu = 0.01      # Viscosity
    eta = 0.005    # Magnetic diffusivity
    forcing_k = 2  # Forcing wavenumber index

    # Time stepping
    dt = 0.001
    Nt = 5000

    # Initialize
    E_K = np.zeros(N_bins)
    E_B = np.zeros(N_bins)

    # Initial kinetic energy (inject at large scales)
    # 강제 쉘(forcing shell, 대규모)에만 에너지를 주입하는 것은
    # 초기 스펙트럼을 미리 지정하여 결과를 편향시키지 않고,
    # 순방향 캐스케이드(forward cascade)가 자연스럽게 발전하는 것을 관찰하기 위함이다.
    E_K[forcing_k] = 1.0

    # Storage
    E_K_hist = []
    E_B_hist = []

    for n in range(Nt):
        # Forcing
        F_K = np.zeros(N_bins)
        F_K[forcing_k] = 0.1  # Constant energy injection

        # Nonlinear transfer (simplified cascade model)
        # T_K(k) ~ -d/dk(k² E_K)  (dimensional, forward cascade)
        T_K = np.zeros(N_bins)
        T_B = np.zeros(N_bins)

        for i in range(1, N_bins - 1):
            # Forward cascade for kinetic
            # 상향 유한 차분(upwind finite difference)은 높은 k(작은 스케일)로
            # 흐르는 에너지 플럭스를 근사한다. 음의 부호는 플럭스 발산을 각 쉘에서의
            # 손실로 변환한다.
            T_K[i] = -0.5 * (E_K[i] - E_K[i-1]) / dk[i]

            # Forward cascade for magnetic (Iroshnikov-Kraichnan)
            # 자기 에너지에 대해 더 작은 계수(0.3 대 0.5)를 사용하는 이유는
            # MHD에서 유체역학 대비 캐스케이드 속도가 감소하기 때문이다:
            # 알프벤(Alfvén) 파동 전파가 와류(eddy) 상호작용을 약화시키며,
            # 이는 IK 이론 및 더 얕은 -3/2 스펙트럼과 일치한다.
            T_B[i] = -0.3 * (E_B[i] - E_B[i-1]) / dk[i]

        # Dynamo effect: kinetic energy → magnetic energy at small scales
        Dynamo = np.zeros(N_bins)
        for i in range(N_bins):
            if k[i] > k[forcing_k]:
                # 신장률(stretching rate) ~ k × E_K^{1/2} (스케일 1/k에서의 변형률)
                # 포화 인자 (1 - E_B/E_K)는 자기 에너지가 등분배(equipartition)에
                # 가까워질 때 증폭을 차단하며, 역운동학(kinematic) 성장을 멈추는
                # 로렌츠(Lorentz) 힘 역반응을 포착한다.
                Dynamo[i] = 0.1 * k[i] * np.sqrt(E_K[i]) * (1 - E_B[i] / (E_K[i] + 1e-10))

        # Magnetic-kinetic coupling (Lorentz force back-reaction)
        # M 항은 J×B 힘을 통해 자기장에서 운동 에너지로 에너지를 전달하며,
        # 로렌츠(Lorentz) 일이 난류를 변형시키는 동역학적 포화를 포착하는 데 필수적이다.
        M = 0.05 * E_B * np.sqrt(E_K + 1e-10)

        # Dissipation
        D_K = nu * k**2 * E_K
        D_B = eta * k**2 * E_B

        # Update
        dE_K_dt = T_K + F_K - D_K - M
        dE_B_dt = T_B + Dynamo - D_B + M

        E_K += dt * dE_K_dt
        E_B += dt * dE_B_dt

        # Prevent negative energies
        E_K = np.maximum(E_K, 0)
        E_B = np.maximum(E_B, 0)

        # Store snapshots
        if n % 100 == 0:
            E_K_hist.append(E_K.copy())
            E_B_hist.append(E_B.copy())

    # Plot final spectra
    plt.figure(figsize=(10, 6))
    plt.loglog(k, E_K, 'b-o', linewidth=2, markersize=5, label='Kinetic $E_K(k)$')
    plt.loglog(k, E_B, 'r-s', linewidth=2, markersize=5, label='Magnetic $E_B(k)$')

    # Reference slopes
    k_ref = k[5:15]
    plt.loglog(k_ref, 0.1 * k_ref**(-5/3), 'k--', linewidth=1, label='$k^{-5/3}$ (Kolmogorov)')
    plt.loglog(k_ref, 0.01 * k_ref**(-3/2), 'g--', linewidth=1, label='$k^{-3/2}$ (IK or saturated dynamo)')

    plt.xlabel('Wavenumber $k$', fontsize=14)
    plt.ylabel('Energy $E(k)$', fontsize=14)
    plt.title('Energy Spectra in MHD Turbulence with Dynamo', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('turbulent_cascade_dynamo.png', dpi=150)
    plt.show()

    # Animate evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(0, len(E_K_hist), max(1, len(E_K_hist)//10)):
        ax.clear()
        ax.loglog(k, E_K_hist[i], 'b-o', linewidth=2, markersize=5, label='Kinetic')
        ax.loglog(k, E_B_hist[i], 'r-s', linewidth=2, markersize=5, label='Magnetic')
        ax.set_xlabel('Wavenumber $k$', fontsize=14)
        ax.set_ylabel('Energy $E(k)$', fontsize=14)
        ax.set_title(f'Energy Spectra (t = {i*100*dt:.2f})', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, which='both', alpha=0.3)
        plt.pause(0.1)

    plt.show()

turbulent_cascade_with_dynamo()
```

### 7.5 Dynamo 시작의 Pm 의존성

```python
import numpy as np
import matplotlib.pyplot as plt

def Pm_dependence_dynamo():
    """
    Plot critical Rm vs Pm for small-scale dynamo onset.

    Empirical fits from simulations:
      - High Pm: Rm_c ~ 100 (const)
      - Low Pm: Rm_c ~ C * Pm^{-α} (increases as Pm decreases)
    """
    Pm = np.logspace(-3, 2, 100)

    # Empirical model (Schekochihin et al.)
    Rm_c = np.zeros_like(Pm)

    for i, pm in enumerate(Pm):
        if pm >= 1:
            # High Pm regime
            Rm_c[i] = 100
        else:
            # Low Pm regime (example: Rm_c ~ 100 * Pm^{-1/2})
            Rm_c[i] = 100 * pm**(-0.5)

    plt.figure(figsize=(10, 6))
    plt.loglog(Pm, Rm_c, 'b-', linewidth=2.5, label='Critical $Rm_c(Pm)$')

    # Reference lines
    plt.axhline(100, color='k', linestyle='--', linewidth=1, label='$Rm_c \\approx 100$ (High Pm)')
    plt.loglog(Pm[Pm < 1], 100 * Pm[Pm < 1]**(-0.5), 'r--', linewidth=1, label='$Rm_c \propto Pm^{-1/2}$ (Low Pm)')

    plt.xlabel('Magnetic Prandtl Number $Pm = \\nu/\\eta$', fontsize=14)
    plt.ylabel('Critical Magnetic Reynolds Number $Rm_c$', fontsize=14)
    plt.title('Dynamo Onset: Dependence on Magnetic Prandtl Number', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('Pm_dependence_dynamo.png', dpi=150)
    plt.show()

Pm_dependence_dynamo()
```

## 8. 요약

**난류 dynamos**는 광범위한 천체물리학 시스템에서 자기장 생성을 이해하는 데 필수적입니다:

1. **소규모 dynamo:**
   - 난류 강제 스케일 ≤ 스케일에서 자기장 증폭
   - 난류 연신에 의해 구동(Lyapunov 지수)
   - **Kazantsev 이론:** `Rm > Rm_c`에 대해 kinematic 성장률 `γ ~ (u/ℓ) (Rm/Rm_c)^{1/2}`
   - **Kazantsev 스펙트럼:** `E_B(k) ∝ k^{3/2}` (kinematic), 포화 시 평탄화
   - 임계 `Rm_c ~ 50-200`, **Pm**에 의존

2. **자기 Prandtl 수(Pm = ν/η):**
   - **높은 Pm(`Pm ≫ 1`):** 효율적 dynamo, 거의 동등분배 포화
   - **낮은 Pm(`Pm ≪ 1`):** 더 높은 `Rm_c`, 동등분배 이하 포화
   - 대부분의 천체물리학 플라즈마가 `Pm ≪ 1`을 가지지만, 효과적 난류 Pm은 `~ 1`일 수 있음

3. **대규모 dynamo:**
   - 강제 > 스케일에서 장을 생성하기 위해 **helicity**(운동 또는 자기) 필요
   - 자기 helicity의 **역 캐스케이드**가 결맞는 대규모 장 구축
   - **Helicity 제약:** 닫힌 시스템에서, helicity 보존이 **재앙적 α-quenching**으로 이어짐
   - **해결책:** 열린 경계 → helicity 플럭스가 quenching 완화

4. **포화 메커니즘:**
   - Lorentz 힘 역반응이 난류 연신 감소
   - 평균장 그림에서 **α-quenching**
   - 균형: dynamo 생성 ≈ 저항 소산 + helicity 플럭스

5. **수치 시뮬레이션:**
   - **DNS:** 모든 스케일 해상, `Re, Rm ≲ 10^4`로 제한
   - **LES:** 서브그리드 스케일 모델, 더 높은 `Re, Rm` 도달
   - 도전: 낮은 Pm은 `η_K`와 `η_R` 둘 다의 해상 필요

6. **응용:**
   - **ISM:** 소규모 dynamo → `μG` 장(관측됨)
   - **은하단:** 합병 동안 소규모 dynamo
   - **강착 원반:** MRI 구동 난류 dynamo
   - **초기 우주:** 씨앗 장 증폭

난류 dynamos는 포화, helicity 플럭스, 그리고 소규모에서 대규모 dynamos로의 전환을 이해하는 데 있어 진행 중인 연구가 있는 활발한 연구 분야입니다.

## 연습 문제

1. **Kazantsev 성장률:** `u_rms = 10 m/s`, `ℓ = 10⁶ m`, `η = 10⁴ m²/s`인 난류의 경우, `Rm`을 계산하고 `Rm_c = 100`이라고 가정하여 성장률을 추정하십시오.

2. **낮은 Pm에 대한 임계 Rm:** `Pm < 1`에 대해 `Rm_c ~ 100 Pm^{-1/2}`인 경우, `Pm = 10^{-5}`인 액체 나트륨에 대해 `Rm_c`는 무엇입니까?

3. **저항 스케일:** `Re = 10⁴`와 `Pm = 0.01`의 경우, 비 `η_R / η_K`를 계산하십시오. 어느 것이 더 작은 스케일에서 소산됩니까?

4. **동등분배 장:** `ρ = 10^{-21} kg/m³`, `v = 10 km/s`인 ISM에서, 동등분배 자기장을 Gauss로 계산하십시오.

5. **Helicity 소산:** `L = 1 kpc`와 `η = 10^{26} cm²/s`(ISM)인 영역의 경우, 자기 helicity의 저항 붕괴 시간 척도를 추정하십시오.

6. **Kazantsev 스펙트럼:** Kinematic 소규모 dynamo에 대해 예상되는 `E_B(k)`를 그리고 `E_B(k) ∝ k^{-3/2}`인 포화 상태와 비교하십시오. 어떤 파수에서 교차합니까?

7. **Pm 스케일링:** 낮은 Pm에 대해 포화 장 강도가 `B_sat ∝ Pm^{1/2}`로 스케일링되는 경우, `Pm = 1`에서 `Pm = 10^{-6}`로 갈 때 `B_sat`는 얼마나 감소합니까?

8. **Python 연습:** 소규모 dynamo 성장 코드를 수정하여 `γ(B) = γ₀(1 - B²/B_eq²)`를 통해 포화를 포함하십시오. 지수 성장 → 포화 전환을 관찰하십시오.

9. **Helicity 플럭스:** 자기 helicity 진화 코드에서, 플럭스율을 증가시키고 자기장의 포화 수준에 어떻게 영향을 미치는지 관찰하십시오.

10. **고급:** Dynamo가 있는 MHD 난류를 위한 간단한 shell-모델을 구현하십시오. 로그적으로 간격을 둔 파수 shells를 사용하고 shells 사이의 비선형 전달을 모델링하십시오. `Rm`이 변함에 따라 에너지 캐스케이드와 dynamo 시작을 연구하십시오.

---

**이전:** [Dynamo Theory](./09_Dynamo_Theory.md) | **다음:** [Solar MHD](./11_Solar_MHD.md)
