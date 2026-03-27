# 9. Dynamo Theory

## 학습 목표

이 레슨을 마치면 다음을 할 수 있어야 합니다:

- 기본적인 dynamo 문제와 행성 및 별들이 자기장을 유지하기 위해 dynamo 메커니즘이 필요한 이유 설명하기
- MHD 근사에서 자기 유도 방정식 유도하고 해석하기
- Anti-dynamo 정리(Cowling, Zeldovich)와 dynamo 요구사항에 대한 함의 이해하기
- Stretch-twist-fold 메커니즘을 포함한 kinematic dynamo 모델 분석하기
- 평균장 이론을 적용하여 대규모 dynamo 작용(α-효과, β-효과, α-Ω dynamos) 이해하기
- Kinematic과 dynamical dynamo 영역 구별하고 포화 메커니즘 이해하기
- 간단한 dynamos의 수치 모델 구현하고 성장률 분석하기

## 1. Dynamo 문제

### 1.1 왜 Dynamos가 필요한가?

지구, 태양, 그리고 많은 다른 천체 물체들은 수십억 년 동안 지속된 대규모 자기장을 가지고 있습니다. 이것은 근본적인 문제를 제기합니다:

**자유 붕괴 문제:**

생성 메커니즘이 없는 경우, 전도성 유체의 자기장은 저항 시간 척도에서 붕괴합니다:

```
τ_η = L²/η
```

여기서:
- `L`은 특성 길이 척도
- `η = 1/(μ₀σ)`는 자기 확산도
- `σ`는 전기 전도도

지구 핵의 경우:
- `L ~ 10⁶ m`
- `η ~ 1-2 m²/s`
- `τ_η ~ 10⁴ - 10⁵ years`

이것은 지구의 나이(~45억 년)보다 훨씬 짧지만, 지구 자기장은 최소 35억 년 동안 존재했습니다(고지자기 증거로부터). 따라서, **능동적인 생성 메커니즘이 있어야 합니다**.

**정의:** **Dynamo**는 전도성 유체의 운동 에너지를 자기 에너지로 변환하여 저항 소산에 대항하여 자기장을 유지하는 메커니즘입니다.

### 1.2 자기 유도 방정식

운동하는 전도성 유체에서 자기장의 진화는 **자기 유도 방정식**에 의해 지배되며, MHD 근사에서 Maxwell 방정식과 Ohm의 법칙으로부터 유도됩니다:

```
∂B/∂t = ∇ × (v × B) + η∇²B
```

성분 형태:

```
∂B_i/∂t + v_j ∂B_i/∂x_j = B_j ∂v_i/∂x_j + η ∂²B_i/∂x_j∂x_j
```

**물리적 해석:**

1. **이류 항** `v·∇B`: 자기장이 유체에 동결되어 운반됨
2. **연신 항** `B·∇v`: 자기력선이 속도 구배(전단, 변형률)에 의해 연신됨
3. **확산 항** `η∇²B`: 저항 소산

이류/연신 대 확산의 상대적 중요성은 **자기 Reynolds 수**로 측정됩니다:

```
Rm = UL/η
```

여기서:
- `U`는 특성 속도
- `L`은 특성 길이 척도

**Dynamo 영역:**
- `Rm ≪ 1`: 확산 지배, 자기장 붕괴
- `Rm ≫ 1`: 이류 지배, dynamo 작용 가능
- 일반적으로, `Rm_critical ~ O(10)` dynamo 시작

### 1.3 에너지 고려사항

자기 에너지 진화는:

```
dE_B/dt = ∫ B·(∇×(v×B)) dV - ∫ η J² dV
```

여기서:
- 첫 번째 항: 유체 운동에 의한 일(양수 가능 → 증폭)
- 두 번째 항: Ohmic 소산(항상 음수)

지속적인 dynamo의 경우:

```
∫ B·(∇×(v×B)) dV ≥ ∫ η J² dV
```

Dynamo는 저항 손실을 극복하기에 충분한 비율로 운동 에너지를 자기 에너지로 변환합니다.

## 2. Anti-Dynamo 정리

Dynamos가 어떻게 작동하는지 이해하기 전에, 무엇이 **작동할 수 없는지** 아는 것이 중요합니다. Anti-dynamo 정리는 dynamo 메커니즘에 근본적인 제약을 가합니다.

### 2.1 Cowling의 정리 (1934)

**진술:** 축대칭 자기장(원통 또는 구 좌표에서 방위각 φ와 무관)은 dynamo 작용에 의해 유지될 수 없습니다.

**증명 스케치 (원통 좌표):**

축대칭 장의 경우:

```
B = B_r(r,z,t) e_r + B_φ(r,z,t) e_φ + B_z(r,z,t) e_z
```

환상 성분 B_φ는 다음을 만족합니다:

```
∂B_φ/∂t = r(B·∇)(v_φ/r) + (1/r)∂(rB_r)/∂r v_φ + ∂B_z/∂z v_φ + η(∇²B_φ - B_φ/r²)
```

`B_φ = 0`인 중립선의 경우, 우변도 그곳에서 사라져야 합니다. 이 중립선을 따라 `∂B_φ/∂t = 0`이므로, 장은 이를 통과하여 성장할 수 없습니다. 연속성에 의해, 표면에서 초기에 `B_φ = 0`이면, 0으로 유지됩니다 → dynamo 없음.

**함의:** 자기장은 **비축대칭 성분을 가져야** 하며, 평균 장이 축대칭이더라도(예: 지구의 쌍극자 지배 장은 시간 평균된 비축대칭 변동으로부터 발생).

### 2.2 Zeldovich의 정리 (1956)

**진술:** 순수하게 2차원 흐름(속도와 장이 한 좌표, 예를 들어 z와 무관)은 dynamo를 유지할 수 없습니다.

**증명 스케치:**

2D에서, 모든 장선과 유선은 평행한 평면에 놓입니다. x-y 평면의 자기력선을 고려하십시오. 유도 방정식은 다음과 같이 됩니다:

```
∂B/∂t = ∇ × (v × B) + η∇²B
```

`v = v(x,y,t)` 및 `B = B(x,y,t)`와 함께. 장은 다음과 같이 쓸 수 있습니다:

```
B = ∇ × (ψ(x,y,t) e_z)  (B_z = 0의 경우)
```

또는 z 성분과 함께:

```
B = B_z(x,y,t) e_z + ∇ × (ψ(x,y,t) e_z)
```

극성 부분(ψ)의 경우, 유도 방정식은 다음을 제공합니다:

```
∂ψ/∂t = v·∇ψ + η∇²ψ
```

이것은 소스 항이 없는 순수 이류-확산 방정식입니다 → ψ는 붕괴합니다. z 성분은 연신될 수 있지만 재생성되지 않습니다.

**함의:** Dynamo 작용을 위해서는 **3차원 흐름**이 **필요**합니다.

### 2.3 제약 요약

Anti-dynamo 정리로부터:

1. **3D 흐름 필요**: 적어도 하나의 성분이 세 방향 모두에서 변해야 함
2. **비축대칭 성분 필요**: 평균 장이 축대칭이더라도
3. **충분한 복잡성 필요**: 단순한 흐름(예: 균일 회전)은 dynamo가 될 수 없음

이러한 정리는 dynamo 메커니즘의 탐색을 안내합니다: helicity, 차등 회전, 또는 대류 난류를 가진 흐름이 필요합니다.

## 3. Kinematic Dynamo 이론

### 3.1 Kinematic 근사

**Kinematic dynamo 이론**에서, 속도장 `v(x,t)`는 주어진 것(prescribed)이고, 유도 방정식을 자기장 진화에 대해 풀며, 흐름에 대한 **Lorentz 힘 역반응을 무시**합니다.

```
∂B/∂t = ∇ × (v × B) + η∇²B    (v는 주어짐)
```

이것은 다음의 경우 유효합니다:

```
B² / (μ₀ρv²) ≪ 1
```

즉, 자기 에너지 ≪ 운동 에너지.

**고유값 문제:**

다음 형태의 해를 가정합니다:

```
B(x,t) = b(x) exp(γt)
```

여기서 `γ`는 성장률(일반적으로 복소수)입니다. 대입하면:

```
γ b = ∇ × (v × b) + η∇²b
```

이것은 고유값 문제입니다:
- 어떤 고유 모드에 대해 `Re(γ) > 0`이면: **dynamo 작용** (장 성장)
- 모든 모드에 대해 `Re(γ) < 0`이면: 장 붕괴

### 3.2 Stretch-Twist-Fold 메커니즘

Kinematic dynamo 작용의 일반적 메커니즘:

**1. 연신:** 속도 전단이 자기력선을 연신시켜 장 강도를 증가시킴(동결 정리: `B/ρ`는 연신과 함께 증가).

**2. 비틀기:** 나선형 또는 회전 흐름이 장선을 비틀어 극성 ↔ 환상 성분을 변환.

**3. 접기:** 재연결 또는 위상 재배열이 무한정 연신을 방지하여 새로운 장 위상을 생성.

**순환:**

```
극성 장 B_p
    ↓ (차등 회전 → 연신)
환상 장 B_t
    ↓ (나선 운동 → 비틀기)
새로운 극성 장 B_p'
    ↓ (재연결/접기)
강화된 극성 장
```

순환당 순 증폭이 확산 손실을 초과하면, 장은 지수적으로 성장 → dynamo.

### 3.3 Ponomarenko Dynamo (1973)

해석적으로 다룰 수 있는 예: 원통형 전도체에서 나선 흐름.

**설정:**
- 반경 `a`의 무한 원통, 내부는 전도체
- 속도: 원통 좌표 (r, φ, z)에서 `v = (0, rΩ, U)`
  - 회전: `v_φ = rΩ`
  - 병진: `v_z = U`
- 경계: r=a에서 `B` 연속, 외부에서 붕괴

**유도 방정식:**

```
∂B/∂t = ∇ × (v × B) + η∇²B
```

정규 모드를 찾습니다:

```
B ~ exp(γt + imφ + ikz)
```

여기서:
- `m`은 방위 파수
- `k`는 축 파수

**분산 관계 (|m|=1, 작은 k에 대해 단순화):**

```
γ ≈ kU - (k² + 1/a²)η  for small Rm
```

충분히 큰 `Rm = Ua/η`에서, 첫 번째 항(이류)이 확산을 극복합니다:

```
Rm_critical ~ O(10)  (k,m에 따라 다름)
```

**물리적 그림:**
- 나선 흐름이 장선을 나선으로 비틀음
- 축 이류(U)가 확산이 평활화하는 것보다 빠르게 비틀림을 강화
- 파장 ~ a인 모드에서 성장 발생

### 3.4 Roberts Flow Dynamo

수직(z) 성분을 가진 2D 주기적 셀 흐름(x-y 평면)도 dynamo가 될 수 있습니다.

**Roberts 흐름 (Roberts, 1972):**

```
v_x = V₀ sin(ky) cos(kx)
v_y = -V₀ cos(ky) sin(kx)
v_z = √2 V₀ sin(kx) sin(ky)
```

이 흐름은 다음을 가집니다:
- 소용돌이가 있는 셀 구조
- Helicity: `⟨v·(∇×v)⟩ ≠ 0`
- 운동 helicity가 α-효과를 구동(평균장 이론 참조)

**Dynamo 속성:**
- 임계 `Rm_c ~ 5` (매우 효율적)
- 빠른 성장률
- 수치 코드의 테스트 케이스로 사용

## 4. 평균장 Dynamo 이론

### 4.1 Reynolds 분해

난류 흐름(예: 항성 대류 영역)의 경우, 흐름과 장은 평균과 변동 성분을 모두 가집니다:

```
v = ⟨v⟩ + u    (⟨u⟩ = 0)
B = ⟨B⟩ + b    (⟨b⟩ = 0)
```

여기서 `⟨·⟩`는 앙상블 또는 공간 평균을 나타냅니다.

**목표:** **평균 장** `⟨B⟩`만의 방정식을 유도하여, 소규모 난류의 효과를 매개변수화합니다.

### 4.2 평균장 유도 방정식

유도 방정식을 평균화:

```
∂⟨B⟩/∂t = ∇ × (⟨v⟩ × ⟨B⟩) + ∇ × ℰ + η∇²⟨B⟩
```

여기서 **평균 기전력(EMF)**은:

```
ℰ = ⟨u × b⟩
```

이것은 소규모 난류 운동에 의해 유도된 평균 전기장입니다. 도전 과제는 `ℰ`를 `⟨B⟩`와 관련시키는 것입니다.

### 4.3 α-효과

**가정:** 작은 Rm(변동)의 균질, 등방성 난류의 경우, 선형 폐쇄:

```
ℰ ≈ α⟨B⟩ - β∇×⟨B⟩
```

여기서:
- `α`: alpha 계수(의사스칼라, 패리티 아래에서 부호 변경)
- `β`: 난류 확산도(스칼라)

**α-효과 유도 (Steenbeck, Krause, Rädler, 1966):**

상관 시간 `τ_c`와 속도 `u_rms`를 가진 나선 난류의 경우:

```
α ≈ -(1/3) τ_c ⟨u·(∇×u)⟩
   = -(1/3) τ_c ⟨h⟩
```

여기서 `⟨h⟩ = ⟨u·(∇×u)⟩`는 **운동 helicity**입니다.

**물리적 해석:**
- 나선 난류는 선호되는 손잡이(회전 시스템의 사이클론 대류)를 가짐
- 나선 와류에 의한 장선의 비틀림이 환상 → 극성(또는 그 반대)을 변환
- α > 0: 오른손 helicity
- α < 0: 왼손 helicity

**β-효과:**

```
β ≈ (1/3) τ_c u_rms²
```

이것은 난류 혼합에 의한 **강화된 확산도**입니다. 유효 확산도는:

```
η_eff = η + β
```

항성 대류 영역에서, `β ≫ η`이므로, 난류 확산이 지배합니다.

### 4.4 α-Ω Dynamos

회전하는, 차등 회전하는 시스템(예: 태양, 행성)에서, 평균 흐름은 다음을 가집니다:
- 차등 회전: `⟨v⟩ = rΩ(r,θ) e_φ` (환상)
- 나선 난류로부터의 α-효과

**α-Ω dynamo 순환 (구 좌표):**

1. **Ω-효과:** 차등 회전이 극성 장 `⟨B_p⟩`를 환상 장 `⟨B_t⟩`로 전단:

```
∂⟨B_φ⟩/∂t ≈ r sin(θ) (⟨B_r⟩ ∂Ω/∂r + ⟨B_θ⟩/r ∂Ω/∂θ)
```

2. **α-효과:** 나선 난류가 환상으로부터 극성을 재생성:

```
∂⟨B_p⟩/∂t ≈ ∇ × (α⟨B_t⟩ e_φ)
```

**피드백 루프:**

```
⟨B_p⟩ → (Ω-효과) → ⟨B_t⟩ → (α-효과) → ⟨B_p⟩'
```

순환당 순 증폭이 확산을 초과하면, 장이 성장 → dynamo.

### 4.5 α² Dynamos

차등 회전이 약하거나 없지만 helicity가 강한 경우, **α² dynamo**가 작동할 수 있습니다:

```
∂⟨B⟩/∂t = ∇ × (α⟨B⟩) + η_eff ∇²⟨B⟩
```

극성 → 환상 및 환상 → 극성 변환 모두 α에 의해 구동됩니다.

**Dynamo 수:**

반경 R의 구에서 α² dynamos의 경우:

```
D_α = α R / η_eff
```

Dynamo 시작은 일반적으로 `|D_α| ~ 10`에서.

α-Ω dynamos의 경우:

```
D_αΩ = (α ΔΩ R³) / η_eff²
```

여기서 `ΔΩ`는 차등 회전율입니다. 시작은 `|D_αΩ| ~ O(1)`에서.

### 4.6 태양 α-Ω Dynamo

**태양 적용:**

- **차등 회전 (Ω):** 태양진동학으로 측정:
  - 적도가 극보다 빠르게 회전: `Ω(θ)`
  - Tachocline(대류 영역 기저부) 근처의 반경 전단: `∂Ω/∂r`

- **α-효과:** 회전 좌표계에서 사이클론 대류 → helicity
  - 북반구: α < 0 (우세)
  - 남반구: α > 0

**태양 주기:**
- 주기: ~11년(흑점 주기), ~22년(자기 극성 주기)
- Tachocline에서 Ω-효과에 의해 생성된 환상 장
- α-효과(또는 Babcock-Leighton 메커니즘: 기울어진 흑점 쌍)에 의해 재생성된 극성 장
- 환상 장의 적도 방향 전파(나비 다이어그램)
- 극성 장의 극 방향 이동

**도전 과제:**
- 높은 Rm에서 α-quenching(동역학적 효과 참조)
- 자기 부력: 강한 환상 장이 불안정해지고 상승 → 흑점
- 플럭스 수송: 자오선 순환이 주기를 조절

## 5. 동역학적 Dynamo 이론

### 5.1 Lorentz 힘 역반응

Kinematic 영역에서, 자기장은 지수적으로 성장합니다. 그러나 `B² / (μ₀ρv²) → O(1)`이 되면, **Lorentz 힘**이 중요해집니다:

```
ρ(∂v/∂t + v·∇v) = -∇p + J×B + ρν∇²v
```

`J×B` 항이 흐름을 수정하고, 이는 차례로 `∂B/∂t`에 영향을 미칩니다. 이것이 **동역학적 영역**입니다.

**포화:** 장 성장이 느려지고 결국 다음과 같은 수준에서 포화됩니다:

```
입력 전력(흐름으로부터) = Ohmic 소산
```

일반적으로:

```
B_sat² / (2μ₀) ~ ε_B × ρv²/2
```

여기서 `ε_B`는 에너지 변환 효율입니다(종종 `ε_B ~ 0.01 - 0.1`).

### 5.2 α-Quenching

평균장 이론에서, α-효과는 자기장이 성장함에 따라 감소합니다:

**간단한 quenching 공식:**

```
α(B) = α₀ / (1 + (B_eq/B_*)²)
```

여기서:
- `α₀`: kinematic alpha
- `B_eq = √(μ₀ρ) u_rms`: equipartition 장
- `B_*`: quenching 장 강도

**재앙적 quenching:**

매우 높은 자기 Reynolds 수 `Rm → ∞`에서, α-quenching이 심각해집니다:

```
α(B) ~ α₀ / Rm
```

이것은 `Rm → ∞` 한계에서 dynamo가 꺼질 것을 의미하는데, 이는 천체물리학적 물체에 대해 비물리적입니다. 이것은 격렬한 논쟁과 해결책 탐색으로 이어졌습니다:
- 자기 helicity 플럭스(경계 효과)
- 전단 구동 dynamos(α에 덜 의존)
- 역 캐스케이드로부터의 대규모 dynamo

**현재 이해:**
- 닫힌 경계의 경우: 재앙적 quenching은 실제 문제
- 열린 경계의 경우(항성 표면, 디스크 코로나): helicity 플럭스가 quenching을 완화
- 고도 난류 시스템에서: dynamo는 주로 소규모일 수 있음

### 5.3 지구 Dynamo

**지구 dynamo:**

- 위치: 액체 외핵(중심에서 반경 3480 - 6371 km)
- 구성: 철-니켈 합금, σ ~ 10⁶ S/m
- 대류 구동: 내핵의 냉각 및 응고(조성 + 열 부력)
- 회전: Ω = 7.3 × 10⁻⁵ rad/s (Coriolis 지배)

**영역:**
- `Rm ~ 10² - 10³` (난류)
- Ekman 수 `E = ν/(ΩL²) ~ 10⁻¹⁵` (빠른 회전)
- 자기 Prandtl 수 `Pm = ν/η ~ 10⁻⁵` (작은 점성)

**Dynamo 메커니즘:**
- 빠르게 회전하는 구에서 대류 → 나선 흐름
- 사이클론 소용돌이로부터의 α-효과
- 열풍 평형으로부터의 차등 회전
- α-Ω 또는 α² dynamo, 차등 회전의 강도에 따라

**수치 시뮬레이션:**
- Glatzmaier-Roberts (1995): 지구자기 역전을 재현하는 최초의 3D dynamo 시뮬레이션
- 현대 코드: MagIC, Rayleigh, Parody
- 도전 과제: 진정한 지구물리학적 매개변수에 도달할 수 없음(E가 너무 작음)

### 5.4 태양 Dynamo 재방문

역반응이 포함된 경우:

- **Tachocline Ω-효과:** 강한 환상 장 `B_φ ~ 10⁴ G` 생성
- **자기 부력:** 환상 플럭스 튜브가 자기 부력으로 인해 상승:

```
ρg = (B²/2μ₀) / H_p
```

여기서 `H_p`는 압력 척도 높이입니다. `B² / (2μ₀) ~ ρc_s²`일 때 불안정.

- **플럭스 출현:** 상승하는 플럭스 튜브가 표면에서 흑점 형성
- **Babcock-Leighton 메커니즘:** 기울어진 쌍극 흑점(Joy의 법칙) → 표면 확산 및 플럭스 수송을 통한 극성 장
- **자오선 순환:** 표면에서 적도 방향, 기저부에서 극 방향 → 플럭스 수송 dynamo

**인터페이스 dynamo vs. 분포 dynamo:**
- **인터페이스:** Tachocline에서 Ω-효과, 대류 영역에서 α-효과(분리된 층)
- **분포:** 대류 영역 전체에서 dynamo
- 현재 합의: 인터페이스 또는 플럭스 수송 dynamo일 가능성

## 6. Dynamo 시뮬레이션을 위한 수치 방법

### 6.1 스펙트럼 방법

주기적 영역 또는 구 기하학의 경우, **스펙트럼 방법**이 매우 효율적입니다.

**Fourier 표현:**

```
B(x,t) = Σ_k B̂_k(t) exp(ik·x)
```

Fourier 공간에서 유도 방정식:

```
∂B̂_k/∂t = ik × (v̂×B̂)_k - ηk² B̂_k
```

컨벌루션 `(v̂×B̂)_k`는 FFT를 통해 계산할 수 있습니다:
1. `v̂_k, B̂_k → v(x), B(x)` 변환(역 FFT)
2. 실공간에서 `v×B` 계산
3. 역변환: `v×B → (v̂×B̂)_k` (정 FFT)

**구면 조화:**

구 기하학의 경우(예: 별, 행성):

```
B(r,θ,φ,t) = Σ_lm B̂_lm(r,t) Y_lm(θ,φ)
```

여기서 `Y_lm`은 구면 조화입니다. 반경 이산화(유한 차분 또는 Chebyshev 다항식)와 결합.

### 6.2 시간 적분

**명시적 스킴 (예: Runge-Kutta):**

```
B^(n+1) = B^n + Δt × RHS(B^n, v^n)
```

안정성 제약(CFL):

```
Δt ≤ min(Δx / |v|, Δx² / η)
```

**암시적 스킴 (예: Crank-Nicolson):**

제한적인 `Δt ~ Δx²` 제약을 피하기 위해 확산을 암시적으로 처리:

```
(B^(n+1) - B^n)/Δt = (1/2)[RHS(B^(n+1)) + RHS(B^n)]
```

각 시간 단계마다 선형 시스템을 풀어야 하지만, 더 큰 Δt를 허용합니다.

### 6.3 비압축성 제약

`∇·B = 0` 조건은 수치적으로 유지되어야 합니다. 방법:

**1. 벡터 포텐셜:**

```
B = ∇ × A
```

구성에 의해 발산이 없습니다. 다음을 통해 `A`를 진화:

```
∂A/∂t = v × B - ∇ψ + η∇²A
```

여기서 `ψ`는 게이지입니다.

**2. 투영 방법:**

각 시간 단계 후, `B`를 발산이 없는 공간으로 투영:

```
B ← B - ∇(∇⁻²(∇·B))
```

Fourier 공간에서: `B̂_k ← B̂_k - k(k·B̂_k)/k²`.

**3. Constrained transport (CT):**

셀 면에 B를 이산화하여 기계 정밀도까지 ∇·B = 0을 보장.

## 7. Python 구현

### 7.1 α-Ω 평균장 Dynamo (1D)

반경 `r`에서 1D로 단순화된 모델, 축대칭 및 평균 장 가정:

```python
import numpy as np
import matplotlib.pyplot as plt

def alpha_omega_dynamo_1d():
    """
    1D α-Ω mean-field dynamo model.

    Equations (in cylindrical r-z, suppress z for 1D):
      ∂B_φ/∂t = r ∂Ω/∂r B_r + η ∂²B_φ/∂r²
      ∂B_r/∂t = ∂/∂r(α B_φ) + η ∂²B_r/∂r²

    Simplified to 1D in radius with periodic or no-flux boundaries.
    """
    # Parameters
    Nr = 100
    r_max = 1.0
    r = np.linspace(0, r_max, Nr)
    dr = r[1] - r[0]

    # Differential rotation profile: Ω(r) = Ω0(1 - r²)
    # r² 형태는 태양과 유사한 차등 자전을 모방한다: 중심에서 가장 빠르고
    # (적도에 해당), 가장자리에서 느리다(극에 해당). 전단 dΩ/dr = -2Ω0 r이
    # Ω-효과의 구동원이다.
    Omega0 = 1.0
    Omega = Omega0 * (1 - r**2)
    dOmega_dr = -2 * Omega0 * r

    # Alpha profile: α(r) = α0 sin(πr)
    # sin(πr) 프로파일은 양쪽 경계에서 0이 되어 α-효과가 내부에 국한된다
    # — 물리적으로 나선성(helicity)은 경계가 아닌 본체의 회전 대류에서
    # 생성되며, 경계에서는 경계 조건이 물리를 오염시킬 수 있다.
    alpha0 = 0.1
    alpha = alpha0 * np.sin(np.pi * r)

    # Magnetic diffusivity
    eta = 0.01

    # Time stepping
    dt = 0.001
    Nt = 5000

    # Initialize fields
    B_phi = np.zeros(Nr)
    B_r = np.zeros(Nr)

    # Initial perturbation
    # 중점에 단일 점 씨앗을 두는 이유: 대규모 모드를 인위적으로 도입하지
    # 않기 위해서다; 다이나모는 이 국소 초기 조건에서 가장 높은 성장률을
    # 가진 정규 모드(normal mode)를 스스로 증폭해야 한다.
    B_r[Nr//2] = 0.01

    # Storage for plotting
    B_phi_hist = []
    B_r_hist = []
    times = []

    # Time evolution
    for n in range(Nt):
        # Compute second derivatives (finite differences)
        d2B_phi = np.zeros(Nr)
        d2B_r = np.zeros(Nr)

        d2B_phi[1:-1] = (B_phi[2:] - 2*B_phi[1:-1] + B_phi[:-2]) / dr**2
        d2B_r[1:-1] = (B_r[2:] - 2*B_r[1:-1] + B_r[:-2]) / dr**2

        # Boundary conditions: no-flux (∂B/∂r = 0)
        # Neumann 조건은 경계를 가로지르는 전류가 없다는 물리적 제약을
        # 모방한다; Dirichlet 조건(B=0)보다 약하며, 자기장 프로파일이
        # 벽 근처에서 자연스럽게 형성되도록 허용한다.
        d2B_phi[0] = d2B_phi[1]
        d2B_phi[-1] = d2B_phi[-2]
        d2B_r[0] = d2B_r[1]
        d2B_r[-1] = d2B_r[-2]

        # Ω-effect: generates B_φ from B_r
        # 이것이 핵심 소스 항이다: 차등 자전(dΩ/dr)이 반경 방향 자기장 선을
        # 방위각 방향으로 늘려 폴로이달 자속(B_r)을 환상 자속(B_φ)으로
        # 변환한다 — α-Ω 사이클의 첫 번째 절반이다.
        omega_term = r * dOmega_dr * B_r

        # α-effect: generates B_r from B_φ
        # α-효과를 공간 기울기 ∂(αB_φ)/∂r로 계산하는 이유: 평균 EMF
        # (ε = αB̄)의 회전을 올바르게 구동하기 위해서다 — 평균장 유도
        # 방정식이 환상 장으로부터 폴로이달 장을 재생성하여 다이나모 루프를
        # 닫으려면 이 형태가 필요하다.
        alpha_term = np.zeros(Nr)
        alpha_term[1:-1] = (alpha[2:] * B_phi[2:] - alpha[:-2] * B_phi[:-2]) / (2*dr)

        # Update equations
        dB_phi_dt = omega_term + eta * d2B_phi
        dB_r_dt = alpha_term + eta * d2B_r

        B_phi += dt * dB_phi_dt
        B_r += dt * dB_r_dt

        # Store snapshots
        if n % 100 == 0:
            B_phi_hist.append(B_phi.copy())
            B_r_hist.append(B_r.copy())
            times.append(n * dt)

    # Plot evolution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for i in range(0, len(times), len(times)//10):
        ax1.plot(r, B_phi_hist[i], label=f't={times[i]:.2f}')
        ax2.plot(r, B_r_hist[i], label=f't={times[i]:.2f}')

    ax1.set_xlabel('Radius r')
    ax1.set_ylabel('Toroidal field $B_\\phi$')
    ax1.set_title('α-Ω Dynamo: Toroidal Field Evolution')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Radius r')
    ax2.set_ylabel('Radial field $B_r$')
    ax2.set_title('α-Ω Dynamo: Radial Field Evolution')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('alpha_omega_dynamo_1d.png', dpi=150)
    plt.show()

    # Growth rate analysis
    B_total = [np.sqrt(np.mean(Bp**2 + Br**2)) for Bp, Br in zip(B_phi_hist, B_r_hist)]

    plt.figure(figsize=(10, 6))
    plt.semilogy(times, B_total, 'b-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Total field energy (RMS)')
    plt.title('α-Ω Dynamo: Exponential Growth')
    plt.grid(True)
    plt.savefig('alpha_omega_growth.png', dpi=150)
    plt.show()

    # Estimate growth rate
    if len(times) > 10:
        log_B = np.log(np.array(B_total[5:]))  # Exclude initial transient
        t_fit = np.array(times[5:])
        coeffs = np.polyfit(t_fit, log_B, 1)
        growth_rate = coeffs[0]
        print(f"Estimated growth rate γ: {growth_rate:.4f}")

    return r, B_phi_hist, B_r_hist, times

# Run simulation
alpha_omega_dynamo_1d()
```

### 7.2 Ponomarenko Dynamo 분산 관계

Ponomarenko dynamo의 파수 대 성장률 계산:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def ponomarenko_dispersion():
    """
    Solve the Ponomarenko dynamo dispersion relation.

    For helical flow in cylinder:
      v_φ = rΩ, v_z = U

    Simplified dispersion (for small k, |m|=1):
      γ ≈ kU - (k² + π²/a²)η

    More accurate: solve eigenvalue problem numerically.
    """
    # Parameters
    a = 1.0  # Cylinder radius
    Omega = 1.0  # Rotation rate
    U = 1.0  # Axial velocity
    eta_vals = np.array([0.01, 0.02, 0.05, 0.1])  # Magnetic diffusivities

    k_vals = np.linspace(0.1, 5.0, 100)  # Axial wavenumber

    plt.figure(figsize=(10, 6))

    for eta in eta_vals:
        Rm = U * a / eta
        gamma = np.zeros_like(k_vals)

        for i, k in enumerate(k_vals):
            # Simplified growth rate
            gamma[i] = k * U - (k**2 + (np.pi/a)**2) * eta

        plt.plot(k_vals, gamma, label=f'Rm = {Rm:.1f} (η={eta})')

    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('Axial wavenumber k')
    plt.ylabel('Growth rate γ')
    plt.title('Ponomarenko Dynamo Dispersion Relation')
    plt.legend()
    plt.grid(True)
    plt.savefig('ponomarenko_dispersion.png', dpi=150)
    plt.show()

    # Find critical Rm
    print("\nCritical Magnetic Reynolds Numbers:")
    for k in [1.0, 2.0, 3.0]:
        # At marginal stability: γ = 0
        # 0 = kU - (k² + π²/a²)η
        # η_c = kU / (k² + π²/a²)
        eta_c = k * U / (k**2 + (np.pi/a)**2)
        Rm_c = U * a / eta_c
        print(f"  k = {k:.1f}: Rm_c = {Rm_c:.2f}")

ponomarenko_dispersion()
```

### 7.3 태양 나비 다이어그램 시뮬레이션

α-Ω dynamo에서 환상 장의 위도 이동 시뮬레이션:

```python
import numpy as np
import matplotlib.pyplot as plt

def solar_butterfly_diagram():
    """
    Simplified solar butterfly diagram from α-Ω dynamo.

    2D model in (θ, t) where θ is latitude.

    Equations:
      ∂B_φ/∂t = C_Ω ∂²Ω/∂θ² B_θ + η ∂²B_φ/∂θ²
      ∂B_θ/∂t = C_α α(θ) B_φ + η ∂²B_θ/∂θ²

    Use profiles:
      Ω(θ) ~ 1 + δΩ cos²(θ)  (equator faster)
      α(θ) ~ cos(θ)  (sign changes across equator)
    """
    # Parameters
    Ntheta = 100
    theta = np.linspace(-np.pi/2, np.pi/2, Ntheta)  # Latitude
    dtheta = theta[1] - theta[0]

    # Differential rotation: Ω(θ) = Ω0(1 + δΩ cos²θ)
    # cos²θ 프로파일은 태양물리학적 관측(helioseismology)과 일치한다: 적도가
    # 극보다 ~20% 빠르게 회전한다. 2차 미분 d²Ω/dθ²이 Ω-효과를 구동하며
    # (위도 전단(latitudinal shear) ≡ 태양에서 지배적인 환상 자기장 소스).
    Omega0 = 1.0
    delta_Omega = 0.2
    Omega = Omega0 * (1 + delta_Omega * np.cos(theta)**2)
    d2Omega_dtheta2 = -2 * delta_Omega * Omega0 * (np.cos(theta)**2 - np.sin(theta)**2)

    # Alpha effect: α(θ) = α0 cos(θ)
    # cos(θ) 의존성은 회전 대류의 물리를 담는다: Coriolis 유도 나선성은
    # 적도에서 최대이고 극에서 사라지며, 적도면을 가로질러 부호가 바뀐다
    # — 나비 다이어그램이 적도면에 대해 반대칭인 이유다.
    alpha0 = 0.5
    alpha = alpha0 * np.cos(theta)

    # Coefficients
    C_Omega = 10.0  # Ω-effect strength
    C_alpha = 1.0   # α-effect strength
    eta = 0.1       # Diffusivity

    # Time stepping
    dt = 0.01
    Nt = 2000

    # Initialize fields
    B_phi = np.zeros(Ntheta)
    B_theta = np.zeros(Ntheta)

    # Initial perturbation at mid-latitudes
    # 중위도(θ ~ π/4)에서 Gaussian 씨앗을 두는 이유: 관측된 각 새 태양
    # 주기의 시작(~30° 위도)을 모방하기 위해서다; 적도에서 시작하면
    # α(0) ≠ 0이지만 d²Ω/dθ²|₀ ≈ 0이어서 시뮬레이션을 억제하게 된다.
    B_theta += 0.01 * np.exp(-((theta - np.pi/4)**2) / 0.1)

    # Storage
    B_phi_hist = np.zeros((Nt//10, Ntheta))
    times = np.zeros(Nt//10)

    # Time evolution
    for n in range(Nt):
        # Second derivatives
        d2B_phi = np.zeros(Ntheta)
        d2B_theta = np.zeros(Ntheta)

        d2B_phi[1:-1] = (B_phi[2:] - 2*B_phi[1:-1] + B_phi[:-2]) / dtheta**2
        d2B_theta[1:-1] = (B_theta[2:] - 2*B_theta[1:-1] + B_theta[:-2]) / dtheta**2

        # Boundary: zero at poles
        # 극에서 B = 0인 물리적 조건: 환상 자속이 자전축을 통과하지 않는다는
        # 것을 반영한다; 이 조건으로 나비 날개가 고위도에서 무한히 쌓이지
        # 않고 끝나게 된다.
        d2B_phi[0] = 0
        d2B_phi[-1] = 0
        d2B_theta[0] = 0
        d2B_theta[-1] = 0

        # Ω-effect
        omega_term = C_Omega * d2Omega_dtheta2 * B_theta

        # α-effect
        alpha_term = C_alpha * alpha * B_phi

        # Update
        dB_phi_dt = omega_term + eta * d2B_phi
        dB_theta_dt = alpha_term + eta * d2B_theta

        B_phi += dt * dB_phi_dt
        B_theta += dt * dB_theta_dt

        # Store
        if n % 10 == 0:
            B_phi_hist[n//10, :] = B_phi
            times[n//10] = n * dt

    # Plot butterfly diagram
    theta_deg = np.degrees(theta)

    plt.figure(figsize=(12, 6))
    plt.contourf(times, theta_deg, B_phi_hist.T, levels=50, cmap='RdBu_r')
    plt.colorbar(label='Toroidal field $B_\\phi$')
    plt.xlabel('Time (arbitrary units)')
    plt.ylabel('Latitude (degrees)')
    plt.title('Solar Butterfly Diagram (α-Ω Dynamo Simulation)')
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.savefig('butterfly_diagram.png', dpi=150)
    plt.show()

solar_butterfly_diagram()
```

### 7.4 Kinematic Dynamo 성장률 계산

Kinematic dynamos에서 성장률 계산을 위한 일반 프레임워크:

```python
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

def kinematic_dynamo_eigenvalue():
    """
    Compute eigenvalues of the kinematic dynamo operator.

    Discretize the induction equation:
      ∂B/∂t = ∇×(v×B) + η∇²B

    in Fourier space for periodic domain.

    Eigenvalue problem: γ b = L b
    where L is the linear operator.
    """
    # Simplified 1D model for illustration
    # Consider B(x,t) in periodic domain [0, 2π]

    N = 32  # Number of Fourier modes
    k = np.fft.fftfreq(N, d=2*np.pi/N) * 2 * np.pi  # Wavenumbers

    # Prescribed velocity: v(x) = V0 sin(x)
    V0 = 1.0
    eta = 0.01

    # In Fourier space, multiplication by v becomes convolution
    # For simplicity, use a simple shear flow: v = V0 x̂
    # Then (v×B) has components involving derivatives

    # Construct operator matrix (simplified for 1D scalar case)
    # This is a toy model; real dynamos need full 3D vector treatment

    L = np.zeros((N, N), dtype=complex)

    for i in range(N):
        # Diagonal: diffusion term
        L[i, i] = -eta * k[i]**2

        # Off-diagonal: advection/stretching (coupling between modes)
        if i > 0:
            L[i, i-1] = 1j * V0 * k[i]  # Simplified coupling

    # Compute eigenvalues
    eigenvalues, eigenvectors = eig(L)

    # Growth rates are real parts
    growth_rates = np.real(eigenvalues)
    frequencies = np.imag(eigenvalues)

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), c=growth_rates, cmap='RdYlGn')
    plt.colorbar(label='Growth rate Re(γ)')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.xlabel('Re(γ)')
    plt.ylabel('Im(γ)')
    plt.title('Eigenvalue Spectrum')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.stem(np.arange(N), growth_rates)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Mode number')
    plt.ylabel('Growth rate Re(γ)')
    plt.title('Growth Rates by Mode')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('dynamo_eigenvalues.png', dpi=150)
    plt.show()

    max_growth_idx = np.argmax(growth_rates)
    print(f"Maximum growth rate: {growth_rates[max_growth_idx]:.4f}")
    print(f"Corresponding frequency: {frequencies[max_growth_idx]:.4f}")

    if growth_rates[max_growth_idx] > 0:
        print("Dynamo action detected!")
    else:
        print("No dynamo action (all modes decay).")

kinematic_dynamo_eigenvalue()
```

## 8. 요약

**Dynamo 이론**은 천체물리학적 자기장이 어떻게 생성되고 유지되는지를 이해하기 위한 프레임워크를 제공합니다:

1. **Dynamo 문제:** 자기장은 천체물리학적 연령보다 훨씬 짧은 저항 시간 척도에서 붕괴 → 능동적 생성 필요.

2. **유도 방정식:** `∂B/∂t = ∇×(v×B) + η∇²B`가 장 진화를 지배하며, 이류/연신과 확산 사이의 경쟁.

3. **Anti-dynamo 정리:**
   - **Cowling:** 축대칭 dynamo 없음
   - **Zeldovich:** 2D dynamo 없음
   - 함의: 3D, 비축대칭 흐름 필요

4. **Kinematic dynamos:** 주어진 속도, B 성장 해결.
   - **Stretch-twist-fold** 메커니즘
   - **Ponomarenko dynamo:** 원통에서 나선 흐름
   - **Roberts 흐름:** helicity를 가진 셀 흐름

5. **평균장 이론:**
   - **α-효과:** 나선 난류가 환상으로부터 극성을 재생성(그리고 그 반대)
   - **β-효과:** 난류 확산
   - **α-Ω dynamos:** 차등 회전 + α-효과(태양, 행성)
   - **α² dynamos:** α-효과만

6. **동역학적 dynamos:**
   - Lorentz 힘 역반응이 장 성장을 포화
   - **α-quenching:** B가 증가함에 따라 α가 감소
   - **재앙적 quenching:** 높은 Rm에서 심각한 감소(helicity 플럭스에 의해 해결)

7. **응용:**
   - **지구 dynamo:** 지구 외핵의 대류, α-Ω 또는 α² 메커니즘
   - **태양 dynamo:** tachocline에서 α-Ω, 11/22년 주기, 나비 다이어그램
   - 항성 및 은하 dynamos

8. **수치 방법:** 스펙트럼, 유한 차분, 벡터 포텐셜, constrained transport.

Dynamos를 이해하는 것은 행성 자기, 항성 활동 주기, 그리고 은하와 초기 우주의 자화를 설명하는 데 중요합니다.

## 연습 문제

1. **자유 붕괴 시간 척도:** 다음에 대한 자기 확산 시간 척도를 계산하십시오:
   - 지구 핵: `L = 10⁶ m`, `η = 2 m²/s`
   - 태양 대류 영역: `L = 2×10⁸ m`, `η = 10⁴ m²/s` (난류)
   - 연령과 비교하십시오.

2. **자기 Reynolds 수:** `v ~ 100 m/s`, `L ~ 10⁸ m`, `η ~ 10⁴ m²/s`인 태양 대류 영역의 경우, `Rm`을 계산하십시오. Dynamo 작용이 가능합니까?

3. **Cowling의 정리:** 순수 환상 장 `B = B_φ(r,z,t) e_φ`를 고려하십시오. 유도 방정식이 `B_φ`를 유지하기 위해 극성 장으로부터의 소스를 요구함을 보이십시오.

4. **Ponomarenko 성장률:** `a = 1 m`, `U = 1 m/s`, `Ω = 1 rad/s`, `η = 0.05 m²/s`인 원통의 경우, 단순화된 공식을 사용하여 `k = 1 m⁻¹`에 대한 성장률을 추정하십시오.

5. **α-효과 추정:** `u_rms = 10 m/s`, 상관 시간 `τ_c = 10⁴ s`, helicity `⟨h⟩ = 10⁻³ m/s²`인 대류 난류의 경우, α를 추정하십시오.

6. **α-Ω Dynamo 수:** 반경 `R = 10⁸ m`, `α = 1 m/s`, `ΔΩ = 10⁻⁶ rad/s`, `η_eff = 10⁴ m²/s`인 구형 쉘에서, dynamo 수 `D_αΩ`을 계산하십시오. Dynamo가 예상됩니까?

7. **Equipartition 장:** `ρ = 10³ kg/m³`, `v = 100 m/s`인 흐름의 경우, equipartition 자기장 강도를 추정하십시오.

8. **Python 연습:** α-Ω 1D 코드를 수정하여 α-quenching을 포함하십시오: `α(B) = α₀/(1 + B²/B_eq²)`. 지수 성장에서 포화로의 전환을 관찰하십시오.

9. **나비 다이어그램 분석:** 시뮬레이션에서, 진동 주기와 적도 방향 전파 속도를 측정하십시오. `C_Ω`와 `C_α`에 어떻게 의존합니까?

10. **고급:** Fourier 스펙트럼 방법을 사용하여 간단한 2D kinematic dynamo를 구현하십시오. Roberts 흐름을 주고 자기장 진화를 해결하십시오. 성장률을 계산하고 문헌 값과 비교하십시오.

---

**이전:** [MHD Turbulence](./08_MHD_Turbulence.md) | **다음:** [Turbulent Dynamo](./10_Turbulent_Dynamo.md)
