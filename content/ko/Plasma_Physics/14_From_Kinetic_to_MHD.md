# 14. From Kinetic to MHD

## 학습 목표

- 6차원 운동 이론에서 3차원 단일 유체 MHD로의 체계적 축소 이해하기
- 이유체 이론에서 종을 결합하여 단일 유체 MHD 방정식 유도하기
- MHD 근사의 유효 조건과 한계 식별하기
- 무충돌 플라즈마에 대한 CGL (Chew-Goldberger-Low) 이중 단열 모델 설명하기
- 중간 축소로서 drift-kinetic과 gyrokinetic 이론 이해하기
- 다양한 플라즈마 모델 비교하고 각각을 언제 적용할지 알기

## 1. 플라즈마 모델의 계층

### 1.1 개요: 완전 운동 이론에서 MHD까지

플라즈마 물리학은 다양한 근사 수준과 계산 비용을 가진 풍부한 모델 계층을 가지고 있습니다:

```
완전 운동 이론 (Vlasov-Maxwell)
    ↓  [회전에 대한 평균]
Drift-Kinetic (5D)
    ↓  [바운스 운동에 대한 평균 / 섭동 전개]
Gyrokinetic (5D, with FLR)
    ↓  [모멘트 취하기]
이유체 (3D × 2 종)
    ↓  [종 결합]
확장 MHD (Hall, FLR, 등)
    ↓  [작은 항 제거]
단일 유체 MHD (3D)
    ↓  [평형, 선형화]
MHD 파동, 불안정성
```

계층의 각 단계 아래로:
- 차원성 또는 변수 수를 **감소**시킴
- 방정식을 **간소화**함
- 일부 물리를 **손실**함
- 계산 효율성을 **증가**시킴

플라즈마 물리학의 기술은 당면 문제에 적합한 모델을 선택하는 것입니다.

### 1.2 각 모델이 포착하는 것은?

| 모델 | 차원 | 포착 | 놓침 |
|-------|------------|----------|---------|
| **Vlasov-Maxwell** | 6D (r,v,t) | 모든 것: 파동-입자, 비등방성, 운동학적 불안정성 | 계산적으로 금지적 |
| **Drift-Kinetic** | 5D (R,v∥,μ,t) | 평행 역학, 포획 입자, 무충돌 감쇠 | 사이클로트론 공명, gyrophase |
| **Gyrokinetic** | 5D (R,v∥,μ,t) | FLR, 난류, 미세 불안정성 | 빠른 자기음파, 압축성 |
| **이유체** | 3D × 2 종 | Hall 효과, 전자 압력, 별도 종 | 운동학적 효과 (감쇠, 불안정성) |
| **Hall MHD** | 3D | Whistler, 빠른 재결합, 분산 파동 | 운동학적 감쇠, 압력 비등방성 |
| **저항 MHD** | 3D | 재결합, 저항 불안정성 | 작은 스케일에서 빠른 과정 |
| **이상적 MHD** | 3D | Alfvén/자기음파 파동, 총체적 평형 | 재결합, 운동학적 물리, 작은 스케일 |

### 1.3 어떤 모델을 언제 사용할까?

**이상적 MHD 사용 시**:
- 대규모 평형과 안정성 (토카막, 항성 대기)
- 저주파 파동 ($\omega \ll \omega_{ci}$)
- 등방 압력을 가진 충돌 플라즈마
- 자기 Reynolds 수 $R_m \gg 1$

**저항 MHD 사용 시**:
- 자기 재결합 (태양 플레어, 서브스톰)
- 저항 불안정성 (찢어짐 모드)
- 전류 구동 역학

**Hall MHD 사용 시**:
- $d_i$에 접근하는 스케일 (자기권계면, 재결합)
- whistler 유출을 가진 빠른 재결합
- 자기장 생성 (다이나모)

**이유체 사용 시**:
- 별도 전자와 이온 역학이 중요
- 각 종 내의 압력 비등방성
- 운동학적 효과가 부차적

**Gyrokinetic 사용 시**:
- 토카막 난류 (이온-온도-경사 모드, 포획-전자 모드)
- FLR 효과를 가진 미세 불안정성
- 약한 섭동을 가진 무충돌 플라즈마

**완전 운동 이론 사용 시**:
- 파동-입자 공명이 중요 (Landau 감쇠, 사이클로트론 가열)
- 강하게 비-Maxwell 분포 (빔-플라즈마, 폭주 전자)
- 속도-공간 불안정성 (이류, bump-on-tail)

## 2. 이유체에서 단일 유체 MHD로

### 2.1 단일 유체 변수 정의

종 $s$ (전자 $e$, 이온 $i$)에 대한 이유체 방정식을 상기:

**연속**:
$$\frac{\partial n_s}{\partial t} + \nabla \cdot (n_s \mathbf{u}_s) = 0$$

**운동량**:
$$m_s n_s \frac{d \mathbf{u}_s}{dt} = q_s n_s (\mathbf{E} + \mathbf{u}_s \times \mathbf{B}) - \nabla p_s + \mathbf{R}_s$$

**에너지** (단열 닫힘):
$$\frac{d}{dt}\left( \frac{p_s}{n_s^\gamma} \right) = 0$$

단일 유체 MHD를 유도하기 위해, **질량 중심(유체) 변수**를 정의합니다:

**질량 밀도**:
$$\rho = m_i n_i + m_e n_e \approx m_i n$$

(준중성 $n_i \approx n_e \equiv n$과 $m_i \gg m_e$ 사용)

**유체 속도** (질량 중심 속도):
$$\mathbf{v} = \frac{m_i n_i \mathbf{u}_i + m_e n_e \mathbf{u}_e}{\rho} \approx \mathbf{u}_i$$

**총 압력**:
$$p = p_i + p_e$$

**전류 밀도**:
$$\mathbf{J} = e(n_i \mathbf{u}_i - n_e \mathbf{u}_e) \approx en(\mathbf{u}_i - \mathbf{u}_e)$$

**전하 밀도** (준중성):
$$\rho_c = e(n_i - n_e) \approx 0$$

### 2.2 연속 방정식 결합

전자와 이온 연속 방정식을 더합니다:

$$\frac{\partial n_e}{\partial t} + \nabla \cdot (n_e \mathbf{u}_e) = 0$$
$$\frac{\partial n_i}{\partial t} + \nabla \cdot (n_i \mathbf{u}_i) = 0$$

전자 방정식에 $m_e$를 곱하고 이온 방정식에 $m_i$를 곱한 후 더합니다:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (m_e n_e \mathbf{u}_e + m_i n_i \mathbf{u}_i) = 0$$

$\rho \mathbf{v} = m_i n_i \mathbf{u}_i + m_e n_e \mathbf{u}_e \approx m_i n \mathbf{u}_i$ 사용:

$$\boxed{\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0}$$

이것이 단일 유체 MHD에 대한 **질량 연속 방정식**입니다.

### 2.3 운동량 방정식 결합

전자와 이온 운동량 방정식을 더합니다:

$$m_e n_e \frac{d \mathbf{u}_e}{dt} = -e n_e (\mathbf{E} + \mathbf{u}_e \times \mathbf{B}) - \nabla p_e + \mathbf{R}_e$$
$$m_i n_i \frac{d \mathbf{u}_i}{dt} = +e n_i (\mathbf{E} + \mathbf{u}_i \times \mathbf{B}) - \nabla p_i + \mathbf{R}_i$$

충돌 항은 상쇄됩니다: $\mathbf{R}_e + \mathbf{R}_i = 0$ (운동량 보존).

전기장 항은 상쇄됩니다(준중성 사용):
$$-e n_e \mathbf{E} + e n_i \mathbf{E} = e(n_i - n_e) \mathbf{E} \approx 0$$

Lorentz 힘 항은 다음을 제공합니다:
$$-e n_e \mathbf{u}_e \times \mathbf{B} + e n_i \mathbf{u}_i \times \mathbf{B} = e n (\mathbf{u}_i - \mathbf{u}_e) \times \mathbf{B} = \mathbf{J} \times \mathbf{B}$$

관성 항:
$$m_e n_e \frac{d \mathbf{u}_e}{dt} + m_i n_i \frac{d \mathbf{u}_i}{dt} \approx m_i n \frac{d \mathbf{u}_i}{dt} = \rho \frac{d \mathbf{v}}{dt}$$

(전자 관성 항 $m_e n_e d\mathbf{u}_e/dt \ll m_i n_i d\mathbf{u}_i/dt$ 무시).

모두 합치면:

$$\boxed{\rho \frac{d \mathbf{v}}{dt} = \mathbf{J} \times \mathbf{B} - \nabla p}$$

이것이 단일 유체 MHD에 대한 **운동량 방정식**입니다.

### 2.4 이상적 Ohm의 법칙

이상적 MHD의 핵심 단계는 **Ohm의 법칙**을 유도하는 것입니다. Lesson 13에서 일반화된 Ohm의 법칙을 유도했습니다:

$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \eta \mathbf{J} + \frac{1}{en} \mathbf{J} \times \mathbf{B} - \frac{1}{en} \nabla p_e + \frac{m_e}{e^2 n^2} \frac{d \mathbf{J}}{dt}$$

**이상적 MHD**에서, 다음 근사를 합니다:

1. **높은 전도도** ($\eta \to 0$): 저항 항 무시
2. **큰 스케일** ($L \gg d_i$): Hall 항 무시
3. **느린 역학**: 전자 관성 무시
4. **무시할 수 있는 전자 압력 경사** (또는 등방 전자 압력): 압력 항 무시

이것이 **이상적 Ohm의 법칙**을 제공합니다:

$$\boxed{\mathbf{E} + \mathbf{v} \times \mathbf{B} = 0}$$

이것이 **동결 조건**입니다: 자기장은 유체에 동결되어 함께 움직입니다.

### 2.5 Faraday의 법칙과 유도 방정식

Maxwell 방정식에서:
$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

이상적 Ohm의 법칙 $\mathbf{E} = -\mathbf{v} \times \mathbf{B}$ 대입:

$$\nabla \times (-\mathbf{v} \times \mathbf{B}) = -\frac{\partial \mathbf{B}}{\partial t}$$

벡터 항등식 $\nabla \times (\mathbf{A} \times \mathbf{B}) = \mathbf{A}(\nabla \cdot \mathbf{B}) - \mathbf{B}(\nabla \cdot \mathbf{A}) + (\mathbf{B} \cdot \nabla)\mathbf{A} - (\mathbf{A} \cdot \nabla)\mathbf{B}$ 사용:

$$\nabla \times (\mathbf{v} \times \mathbf{B}) = \mathbf{v}(\nabla \cdot \mathbf{B}) - \mathbf{B}(\nabla \cdot \mathbf{v}) + (\mathbf{B} \cdot \nabla)\mathbf{v} - (\mathbf{v} \cdot \nabla)\mathbf{B}$$

$\nabla \cdot \mathbf{B} = 0$이므로:

$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) = (\mathbf{B} \cdot \nabla)\mathbf{v} - \mathbf{B}(\nabla \cdot \mathbf{v}) - (\mathbf{v} \cdot \nabla)\mathbf{B}$$

재배열:

$$\boxed{\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B})}$$

또는 동등하게:

$$\boxed{\frac{d \mathbf{B}}{dt} = (\mathbf{B} \cdot \nabla)\mathbf{v} - \mathbf{B}(\nabla \cdot \mathbf{v})}$$

여기서 $d/dt = \partial/\partial t + \mathbf{v} \cdot \nabla$는 대류 도함수입니다.

이것이 **유도 방정식** (또는 **자기 진화 방정식**)입니다. 플라즈마가 흐를 때 자기장이 어떻게 진화하는지 기술합니다.

### 2.6 요약: 이상적 MHD 방정식

**이상적 MHD 방정식**은:

**질량 연속**:
$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$

**운동량**:
$$\rho \frac{d \mathbf{v}}{dt} = \mathbf{J} \times \mathbf{B} - \nabla p$$

**유도**:
$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B})$$

**에너지** (단열):
$$\frac{d}{dt}\left( \frac{p}{\rho^\gamma} \right) = 0$$

**Ampère의 법칙** (변위 전류 무시):
$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$$

**자기 단극 없음**:
$$\nabla \cdot \mathbf{B} = 0$$

이들은 **8개 미지수**에 대한 **8개 방정식**입니다: $\rho$, $\mathbf{v}$ (3 성분), $p$, $\mathbf{B}$ (3 성분), 제약 $\nabla \cdot \mathbf{B} = 0$ 주어짐.

(전기장 $\mathbf{E}$는 Ohm의 법칙에 의해 결정됩니다: $\mathbf{E} = -\mathbf{v} \times \mathbf{B}$.)

## 3. MHD의 유효 조건

### 3.1 저주파: $\omega \ll \omega_{ci}$

MHD는 **저주파 근사**입니다. 현상의 시간 스케일은 이온 사이클로트론 주기보다 훨씬 길어야 합니다:

$$\omega \ll \omega_{ci} = \frac{eB}{m_i}$$

이는 이온이 개별 입자 거동을 나타내기보다는 유체와 같은 방식으로 장에 회전하고 반응할 시간이 있도록 보장합니다.

**예**: $B = 1$ T의 경우, $\omega_{ci} \approx 10^8$ rad/s ($f \approx 16$ MHz). MHD는 ~10 MHz보다 느린 현상에 유효합니다.

### 3.2 큰 스케일: $L \gg \rho_i$

공간 스케일은 **이온 gyroradius**보다 훨씬 커야 합니다:

$$L \gg \rho_i = \frac{v_{th,i}}{\omega_{ci}}$$

$\lesssim \rho_i$ 스케일에서, 유한 Larmor 반경 (FLR) 효과가 중요해지고, MHD가 붕괴됩니다.

**예**: $T_i = 10$ keV와 $B = 1$ T의 경우, $\rho_i \approx 0.5$ cm. MHD는 $\gg 1$ cm 스케일에 유효합니다.

### 3.3 충돌적: $\lambda_{mfp} \ll L$

등방 압력(이상적 MHD에서 가정)을 위해, 충돌이 분포함수를 등방화할 만큼 충분히 빈번해야 합니다:

$$\lambda_{mfp} = v_{th} \tau \ll L$$

여기서 $\tau$는 충돌 시간입니다.

무충돌 플라즈마에서, 압력 텐서는 **비등방적**입니다 ($p_\parallel \neq p_\perp$), 더 일반적인 닫힘이 필요합니다 (예: 아래에서 논의할 CGL).

**예**: 태양풍에서, $\lambda_{mfp} \sim 1$ AU $\gg L$ 어떤 합리적인 구조에 대해. 표준 MHD는 유효하지 않습니다—CGL 또는 운동학적 모델이 필요합니다.

### 3.4 비상대론적: $v \ll c$

플라즈마 흐름과 열속도는 비상대론적이어야 합니다:

$$v, v_{th} \ll c$$

이는 Ampère의 법칙에서 변위 전류를 무시하고 비상대론적 운동량 방정식을 사용할 수 있게 합니다.

**예**: $T = 10$ keV의 경우, $v_{th,e} \approx 0.04c$ (상대론적 보정 ~몇 퍼센트). 더 높은 온도의 경우, 상대론적 MHD가 필요합니다.

### 3.5 준중성: $n_e \approx n_i$

플라즈마는 관심 스케일에서 준중성이어야 합니다:

$$L \gg \lambda_D = \sqrt{\frac{\epsilon_0 k_B T}{n e^2}}$$

이는 전하 분리를 무시하고 변위 전류를 제거할 수 있게 합니다.

**예**: $n = 10^{19}$ m$^{-3}$와 $T = 10$ eV의 경우, $\lambda_D \approx 10$ μm. MHD는 $L \gg 10$ μm에 유효합니다.

### 3.6 높은 자기 Reynolds 수: $R_m \gg 1$

**이상적 MHD** (동결)의 경우, 자기 Reynolds 수가 커야 합니다:

$$R_m = \frac{\mu_0 V L}{\eta} \gg 1$$

여기서 $V$는 특성 유동 속도, $L$은 길이 스케일, $\eta$는 저항률입니다.

$R_m \sim 1$일 때, 저항률이 중요해집니다 → **저항 MHD**.

**예**: 토카막에서, $V \sim 100$ m/s, $L \sim 1$ m, $\eta \sim 10^{-8}$ Ω·m → $R_m \sim 10^{10}$. 이상적 MHD가 뛰어납니다.

### 3.7 유효 영역 요약

```
이상적 MHD는 다음 모두가 성립할 때 유효합니다:

1. ω << ω_ci           (저주파)
2. L >> ρ_i            (큰 스케일)
3. λ_mfp << L          (충돌적, 등방 p의 경우)
4. v << c              (비상대론적)
5. L >> λ_D            (준중성)
6. R_m >> 1            (동결)

위반 → 확장 MHD 또는 운동학적 모델 필요.
```

## 4. CGL (이중 단열) 모델

### 4.1 동기: 무충돌 자화 플라즈마

많은 천체물리학적 플라즈마(태양풍, 자기권, 은하단)에서, 충돌 평균 자유 경로가 **거대**합니다:

$$\lambda_{mfp} \gg L$$

이러한 플라즈마에서, 입자는 충돌 없이 긴 거리를 이동할 수 있습니다. 압력 텐서는 **비등방적**이 됩니다:

$$\overleftrightarrow{P} = p_\perp \overleftrightarrow{I} + (p_\parallel - p_\perp) \hat{\mathbf{b}} \hat{\mathbf{b}}$$

여기서 $\hat{\mathbf{b}} = \mathbf{B}/B$이고:
- $p_\parallel$: $\mathbf{B}$에 평행한 압력
- $p_\perp$: $\mathbf{B}$에 수직한 압력

표준 MHD는 $p_\parallel = p_\perp = p$ (등방)를 가정하며, 무충돌 플라즈마에서 무효입니다.

### 4.2 Chew-Goldberger-Low (1956) 모델

Chew, Goldberger, Low (CGL)는 **단열 불변량의 보존**을 가정하여 무충돌, 강하게 자화된 플라즈마에 대한 닫힘을 유도했습니다:

**제1 단열 불변량** (자기 모멘트):
$$\mu = \frac{m v_\perp^2}{2B} = \text{const}$$

이는 다음을 의미합니다:
$$\frac{d}{dt}\left( \frac{p_\perp}{n B} \right) = 0$$

**제2 단열 불변량** (종방향 작용):
$$J = \oint v_\parallel ds = \text{const}$$

지역 유체 요소(거울 사이에서 바운스하지 않음)의 경우, 이것은 다음과 같이 됩니다:
$$\frac{d}{dt}\left( \frac{p_\parallel B^2}{n^3} \right) = 0$$

이들이 **CGL 방정식** (또한 **이중 단열** 방정식이라고도 함)입니다.

### 4.3 CGL 닫힘 관계

CGL 방정식은:

$$\boxed{\frac{d}{dt}\left( \frac{p_\perp}{nB} \right) = 0}$$

$$\boxed{\frac{d}{dt}\left( \frac{p_\parallel B^2}{n^3} \right) = 0}$$

이들은 다음과 같이 다시 쓸 수 있습니다:

$$\frac{1}{p_\perp} \frac{dp_\perp}{dt} = \frac{1}{n} \frac{dn}{dt} + \frac{1}{B} \frac{dB}{dt}$$

$$\frac{1}{p_\parallel} \frac{dp_\parallel}{dt} = 3 \frac{1}{n} \frac{dn}{dt} - 2 \frac{1}{B} \frac{dB}{dt}$$

**물리적 해석**:

- 장이 증가할 때($dB/dt > 0$), $p_\perp$가 증가하고(베타트론 가열), $p_\parallel$은 감소합니다(자기 거울 효과).
- 압축($dn/dt > 0$)은 $p_\perp$와 $p_\parallel$ 모두 증가시킵니다.

### 4.4 CGL 압력 텐서

CGL 압력 텐서를 가진 운동량 방정식은 다음과 같이 됩니다:

$$\rho \frac{d \mathbf{v}}{dt} = \mathbf{J} \times \mathbf{B} - \nabla \cdot \overleftrightarrow{P}$$

여기서:
$$\nabla \cdot \overleftrightarrow{P} = \nabla p_\perp + (p_\parallel - p_\perp) \left[ \frac{\nabla \cdot \mathbf{B}}{B} \hat{\mathbf{b}} + \frac{(\mathbf{B} \cdot \nabla) \mathbf{B}}{B^2} \right]$$

$\nabla \cdot \mathbf{B} = 0$과 $(\mathbf{B} \cdot \nabla)\mathbf{B} = B^2 \boldsymbol{\kappa}$ (여기서 $\boldsymbol{\kappa}$는 곡률) 사용:

$$\nabla \cdot \overleftrightarrow{P} = \nabla p_\perp + (p_\parallel - p_\perp) \boldsymbol{\kappa}$$

따라서 운동량 방정식은:

$$\rho \frac{d \mathbf{v}}{dt} = \mathbf{J} \times \mathbf{B} - \nabla p_\perp - (p_\parallel - p_\perp) \boldsymbol{\kappa}$$

비등방성은 장 곡률을 따라 추가 힘 $-(p_\parallel - p_\perp) \boldsymbol{\kappa}$를 만듭니다.

### 4.5 CGL 불안정성

CGL 모델은 다음의 경우 **압력-비등방성-구동 불안정성**을 예측합니다:

1. **거울 불안정성**: $p_\perp / p_\parallel$이 너무 크면
   $$\frac{p_\perp}{p_\parallel} > 1 + \frac{1}{\beta_\perp}$$
   여기서 $\beta_\perp = 2\mu_0 p_\perp / B^2$.

   플라즈마는 $p_\perp$를 줄이기 위해 지역 **자기 거울** (강화된 $B$ 영역)을 만듭니다.

2. **Firehose 불안정성**: $p_\parallel / p_\perp$이 너무 크면
   $$\frac{p_\parallel}{p_\perp} > 1 + \frac{2}{\beta_\parallel}$$
   여기서 $\beta_\parallel = 2\mu_0 p_\parallel / B^2$.

   자기장선이 압력 하에서 소방호스처럼 "꼬입니다".

이러한 불안정성은 태양풍과 지구 자기권계면에서 관찰됩니다.

### 4.6 CGL의 한계

1. **열유속 없음**: CGL은 평행 열전도가 없다고 가정합니다. 실제로, 열유속은 긴 평행 스케일에서 중요합니다.

2. **충돌 없음**: CGL은 무충돌 플라즈마를 위한 것입니다. 약한 충돌을 추가하는 것도 진화를 수정합니다.

3. **지역 근사**: CGL은 제2 단열 불변량이 지역적으로 유지된다고 가정하며, 긴 스케일에서 바운스하는 포획 입자에 대해 붕괴됩니다.

4. **느린 역학**: CGL은 gyro-주기와 바운스 주기에 비해 느린 진화를 가정합니다.

이러한 한계에도 불구하고, CGL은 무충돌 플라즈마에서 비등방 압력의 본질적 물리를 포착하며 우주 물리학에서 널리 사용됩니다.

## 5. MHD를 넘어서: Drift-Kinetic과 Gyrokinetic 이론

### 5.1 Drift-Kinetic 이론

**Drift-kinetic 이론**은 **gyrophase에 대한 평균**을 통해 차원성을 6D에서 **5D**로 줄입니다.

아이디어: 자화 플라즈마에서, 입자는 장선 주위를 빠르게 회전합니다. 느린 역학($\omega \ll \omega_c$)에만 관심이 있다면, 빠른 회전에 대해 평균할 수 있습니다.

**변수**:
- $\mathbf{R}$: 안내 중심 위치 (3D)
- $v_\parallel$: 평행 속도 (1D)
- $\mu$: 자기 모멘트 (단열 불변량, 매개변수)
- 시간 $t$

**분포함수**: $F(\mathbf{R}, v_\parallel, \mu, t)$ (6D 대신 5D)

**Drift-kinetic 방정식** (간소화):
$$\frac{\partial F}{\partial t} + \mathbf{v}_d \cdot \nabla_\mathbf{R} F + \frac{d v_\parallel}{dt} \frac{\partial F}{\partial v_\parallel} = C[F]$$

여기서 $\mathbf{v}_d$는 평행 운동과 수직 표류를 포함합니다:
$$\mathbf{v}_d = v_\parallel \hat{\mathbf{b}} + \mathbf{v}_E + \mathbf{v}_{\nabla B} + \mathbf{v}_\kappa + \cdots$$

**포착하는 것**:
- 평행 운동과 바운스 역학 (포획 입자)
- 모든 안내-중심 표류
- 무충돌 (Landau) 감쇠

**놓치는 것**:
- 사이클로트론 공명 (gyrophase로 평균됨)
- 유한 Larmor 반경 (FLR) 효과

**응용**:
- 신고전 수송 (토카막 충돌 확산)
- 바운스-평균 운동 이론 (포획-입자 불안정성)
- 복사 벨트 역학 (drift-loss-cone)

### 5.2 Gyrokinetic 이론

**Gyrokinetic 이론**은 가장 정교한 축소 모델로, gyrophase에 대한 평균을 하면서 **유한 Larmor 반경 (FLR)** 효과를 포착합니다.

**핵심 혁신**: 작은 매개변수로 전개:
$$\delta = \frac{\rho_i}{L} \sim \frac{\omega}{\omega_{ci}} \sim \frac{\delta f}{f_0} \ll 1$$

이것이 **gyrokinetic 순서**입니다: 느리고, 소진폭, 긴 파장 요동.

**변수** (drift-kinetic과 동일):
- $\mathbf{R}$: gyrocenter 위치
- $v_\parallel$: 평행 속도
- $\mu$: 자기 모멘트

**Gyrokinetic 분포**: $g(\mathbf{R}, v_\parallel, \mu, t)$ (섭동 부분)

**Gyrokinetic 방정식** (개략):
$$\frac{\partial g}{\partial t} + \mathbf{v}_d \cdot \nabla g + \frac{dv_\parallel}{dt} \frac{\partial g}{\partial v_\parallel} = \text{(FLR을 가진 소스 항)}$$

drift-kinetic과의 핵심 차이: **FLR 효과**가 다음을 통해 유지됩니다:
- Gyro평균 전기장: $\langle \phi \rangle_\alpha$ (gyro-궤도에 대한 평균)
- Gyro평균 자기 섭동

**포착하는 것**:
- FLR 효과 (이온 Landau 감쇠, FLR을 가진 파동-입자 공명)
- 미세 불안정성: ITG (이온 온도 경사), TEM (포획 전자 모드), ETG (전자 온도 경사)
- FLR을 가진 난류 캐스케이드

**놓치는 것**:
- 압축 가능한 Alfvén 파동 (빠른 자기음파)
- 저주파 근사: $\omega \ll \omega_{ci}$

**응용**:
- **토카막 난류**: gyrokinetic 시뮬레이션 (GENE, GS2, GYRO)이 난류 수송을 예측하여 제한된 밀폐를 설명
- **미세 불안정성 분석**: ITG, TEM, ETG 모드의 성장률 결정
- **Zonal flows**: 난류를 규제하는 자체 생성 전단 흐름

Gyrokinetic 시뮬레이션은 토카막 물리학의 최첨단이며 세계 최대 슈퍼컴퓨터에서 실행됩니다.

### 5.3 비교: Drift-Kinetic vs. Gyrokinetic

| 특징 | Drift-Kinetic | Gyrokinetic |
|---------|---------------|-------------|
| 차원 | 5D | 5D |
| FLR 효과 | 아니오 | 예 |
| Gyrophase-평균 | 예 | 예 |
| 순서 | 없음 (정확한 gyroaverage) | $\delta \ll 1$ (섭동적) |
| 해결하는 것 | 바운스 운동, 표류 | 난류, 미세 불안정성 |
| 일반적 응용 | 신고전, 복사 벨트 | 토카막 난류, ITG/TEM |
| 계산 비용 | 중간 | 매우 높음 |

## 6. 확장 MHD 모델

### 6.1 Hall MHD

**Hall MHD**는 Ohm의 법칙에 Hall 항을 포함합니다:

$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \frac{1}{en} \mathbf{J} \times \mathbf{B}$$

이는 이온과 전자가 $\sim d_i$ (이온 skin depth) 스케일에서 분리될 수 있게 합니다.

**핵심 특징**:
- 높은 $k$에서 Whistler 파동
- 빠른 자기 재결합 (Petschek 속도)
- 분산적 Alfvén 파동

**응용**:
- 자기 재결합 (자기권계면, 자기권꼬리, 태양 코로나)
- 다이나모 이론 (자기장 생성)
- 우주 플라즈마 난류

### 6.2 이온도 MHD

별도 전자와 이온 온도:

$$\frac{d p_e}{dt} + \gamma p_e \nabla \cdot \mathbf{v} = Q_{ei} + Q_e$$
$$\frac{d p_i}{dt} + \gamma p_i \nabla \cdot \mathbf{v} = -Q_{ei} + Q_i$$

여기서 $Q_{ei}$는 전자-이온 에너지 교환이고, $Q_{e,i}$는 외부 가열입니다.

**응용**:
- 가열과 에너지 분할 (예: 충격파가 처음에 전자보다 이온을 더 가열)
- 복사 냉각 (전자가 더 효율적으로 복사)

### 6.3 FLR-MHD

압력 텐서에 유한 Larmor 반경 보정 포함:

$$\overleftrightarrow{P} = p \overleftrightarrow{I} + \overleftrightarrow{\Pi}^{\text{FLR}}$$

여기서 $\overleftrightarrow{\Pi}^{\text{FLR}}$은 gyroviscosity와 기타 FLR 효과를 포함합니다.

**응용**:
- Kinetic Alfvén 파동
- MHD 불안정성의 FLR 안정화

### 6.4 관성 MHD (전자 MHD)

매우 작은 스케일($d_e$)에서, 전자 관성이 중요해집니다:

$$\mathbf{E} + \mathbf{v}_e \times \mathbf{B} = \frac{m_e}{e^2 n} \frac{d \mathbf{J}}{dt}$$

이것이 **전자 MHD** (EMHD)로, 이온이 정지하고 전자만 움직입니다.

**분산 관계** (EMHD의 whistler):
$$\omega = k^2 V_A d_e$$

**응용**:
- 자기 재결합 확산 영역
- 전자-스케일 난류

## 7. Python 코드 예제

### 7.1 유효 영역 다이어그램

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameter space: length scale vs. frequency
L = np.logspace(-4, 6, 200)  # 0.1 mm to 1000 km
omega = np.logspace(2, 10, 200)  # 100 rad/s to 10 GHz

L_grid, omega_grid = np.meshgrid(L, omega)

# Plasma parameters (typical tokamak)
n = 1e20  # m^-3
B = 2.0   # T
T = 5e3   # eV (5 keV)

e = 1.6e-19
m_i = 1.67e-27
m_e = 9.11e-31
k_B = 1.38e-23

# Characteristic scales and frequencies
omega_ci = e * B / m_i
omega_ce = e * B / m_e
omega_pi = np.sqrt(n * e**2 / (m_i * 8.85e-12))
omega_pe = np.sqrt(n * e**2 / (m_e * 8.85e-12))

v_th_i = np.sqrt(2 * k_B * T * e / m_i)
v_th_e = np.sqrt(2 * k_B * T * e / m_e)

rho_i = v_th_i / omega_ci
rho_e = v_th_e / omega_ce
d_i = 3e8 / omega_pi
d_e = 3e8 / omega_pe
lambda_D = np.sqrt(8.85e-12 * k_B * T * e / (n * e**2))

print("Characteristic scales and frequencies:")
print(f"  Ion gyrofrequency ω_ci = {omega_ci:.2e} rad/s ({omega_ci/(2*np.pi):.2e} Hz)")
print(f"  Ion gyroradius ρ_i = {rho_i*100:.2f} cm")
print(f"  Ion skin depth d_i = {d_i:.2f} m")
print(f"  Electron skin depth d_e = {d_e*100:.2f} cm")
print(f"  Debye length λ_D = {lambda_D*1e6:.2f} μm")
print()

# Define validity regions
# 1. MHD: ω << ω_ci, L >> ρ_i
# 0.1 계수는 보수적인 안전 여유입니다: MHD 순서는 ω/ω_ci → 0을 가정하므로
# 고차 보정(ε = ω/ω_ci)을 ~10% 미만으로 유지하기 위해 최소 한 자릿수의
# 분리가 필요합니다. 유사하게, L > 10ρ_i는 FLR 보정(ε = ρ_i/L)이
# 유체 기술이 성립하기에 충분히 작도록 보장합니다.
MHD = (omega_grid < 0.1 * omega_ci) & (L_grid > 10 * rho_i)

# 2. Hall MHD: ω << ω_ci, L ~ d_i
# Hall 효과는 L ~ d_i에서 O(1)이 됩니다; 상한 L < 100 d_i는
# Hall 항(d_i/L)이 처음으로 ~1% 이상 증가하는 지점을 표시하여
# MHD와 Hall MHD 영역 사이의 전환 구간을 정의합니다.
Hall_MHD = (omega_grid < 0.1 * omega_ci) & (L_grid > 10 * rho_i) & (L_grid < 100 * d_i)

# 3. Two-fluid: ω << ω_ce, L > d_e
# 이유체 이론은 전자 skin depth d_e(전자 관성이 O(1)이 됨)와
# ω ~ ω_ce(전자 사이클로트론 공명이 포함되지 않음)에서 붕괴됩니다.
# 0.1 계수는 MHD와 동일한 한 자릿수 안전 여유를 제공합니다.
Two_Fluid = (omega_grid < 0.1 * omega_ce) & (L_grid > 10 * d_e)

# 4. Gyrokinetic: ω ~ ω_ci, L ~ ρ_i
# Gyrokinetics는 ω_ci 근처와 L ~ ρ_i에서 유효한 섭동 이론입니다;
# 이 특성값들 주변 한 자릿수를 벗어나면 부정확해집니다.
Gyrokinetic = (omega_grid > 0.01 * omega_ci) & (omega_grid < omega_ci) & \
              (L_grid > rho_i) & (L_grid < 100 * rho_i)

# 5. Full kinetic: always valid (but expensive)
Full_Kinetic = np.ones_like(L_grid, dtype=bool)

# Plot
fig, ax = plt.subplots(figsize=(11, 8))

# Color regions
ax.contourf(L_grid, omega_grid, MHD.astype(int), levels=[0.5, 1.5],
            colors=['lightblue'], alpha=0.6)
ax.contourf(L_grid, omega_grid, Hall_MHD.astype(int), levels=[0.5, 1.5],
            colors=['lightcoral'], alpha=0.6)
ax.contourf(L_grid, omega_grid, Gyrokinetic.astype(int), levels=[0.5, 1.5],
            colors=['lightgreen'], alpha=0.6)

# Boundary lines
ax.axhline(omega_ci, color='r', linestyle='--', linewidth=2, label=f'$\omega_{{ci}}$ = {omega_ci:.2e} rad/s')
ax.axhline(omega_ce, color='m', linestyle='--', linewidth=1.5, label=f'$\omega_{{ce}}$ = {omega_ce:.2e} rad/s')

ax.axvline(rho_i, color='b', linestyle='--', linewidth=2, label=f'$\\rho_i$ = {rho_i*100:.1f} cm')
ax.axvline(d_i, color='g', linestyle='--', linewidth=2, label=f'$d_i$ = {d_i:.1f} m')
ax.axvline(d_e, color='orange', linestyle='--', linewidth=1.5, label=f'$d_e$ = {d_e*100:.1f} cm')

# Labels for regions
ax.text(1e0, 1e3, 'MHD', fontsize=16, weight='bold', color='blue')
ax.text(1e-1, 1e4, 'Hall MHD', fontsize=14, weight='bold', color='red')
ax.text(1e-2, 1e7, 'Gyrokinetic', fontsize=14, weight='bold', color='green')
ax.text(1e-3, 1e9, 'Full Kinetic', fontsize=14, weight='bold', color='black')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Length scale L (m)', fontsize=13)
ax.set_ylabel('Frequency ω (rad/s)', fontsize=13)
ax.set_title('Plasma Model Validity Regimes (n=$10^{20}$ m$^{-3}$, B=2 T, T=5 keV)', fontsize=14)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1e-4, 1e6)
ax.set_ylim(1e2, 1e10)

plt.tight_layout()
plt.savefig('validity_regimes.png', dpi=150)
plt.show()
```

### 7.2 CGL vs. 등방 MHD: 거울 불안정성

```python
import numpy as np
import matplotlib.pyplot as plt

def mirror_instability_threshold(beta_perp):
    """
    Mirror instability threshold: p_perp/p_parallel > 1 + 1/beta_perp
    """
    return 1 + 1/beta_perp

# Beta range
beta_perp = np.logspace(-2, 2, 200)

# Threshold
threshold = mirror_instability_threshold(beta_perp)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Threshold curve
ax1.plot(beta_perp, threshold, 'r-', linewidth=3, label='Mirror instability threshold')
ax1.fill_between(beta_perp, 1, threshold, alpha=0.3, color='red', label='Unstable')
ax1.fill_between(beta_perp, threshold, 10, alpha=0.3, color='green', label='Stable')

ax1.set_xscale('log')
ax1.set_xlabel(r'$\beta_\perp = 2\mu_0 p_\perp / B^2$', fontsize=12)
ax1.set_ylabel(r'$p_\perp / p_\parallel$', fontsize=12)
ax1.set_title('Mirror Instability Threshold', fontsize=13)
ax1.set_ylim(1, 10)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Growth rate (simplified)
# γ/Ω_i ~ sqrt(β_perp) * (p_perp/p_parallel - 1 - 1/β_perp) for unstable
beta_example = 1.0
anisotropy = np.linspace(1, 5, 100)
threshold_value = mirror_instability_threshold(beta_example)

# np.where는 인과율을 강제합니다: 성장률은 임계값 미만에서 정확히 0으로,
# 거울 모드가 그 조건에서 선형적으로 안정하기 때문입니다(불안정성의 에너지 원천이 없음).
# sqrt(β_perp) 앞 계수는 자기 압력에 대한 더 높은 플라즈마 압력이
# 불안정성을 위한 더 많은 자유 에너지를 제공함을 반영합니다.
gamma_normalized = np.where(anisotropy > threshold_value,
                             np.sqrt(beta_example) * (anisotropy - threshold_value),
                             0)

ax2.plot(anisotropy, gamma_normalized, 'b-', linewidth=3)
ax2.axvline(threshold_value, color='r', linestyle='--', linewidth=2,
            label=f'Threshold at $\\beta_\\perp$ = {beta_example}')
ax2.fill_between(anisotropy, 0, gamma_normalized, alpha=0.3, color='blue')

ax2.set_xlabel(r'$p_\perp / p_\parallel$', fontsize=12)
ax2.set_ylabel(r'Growth rate $\gamma / \Omega_i$', fontsize=12)
ax2.set_title(f'Mirror Instability Growth Rate ($\\beta_\\perp$ = {beta_example})', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mirror_instability.png', dpi=150)
plt.show()

print(f"Mirror instability:")
print(f"  At β_perp = 0.1: threshold p_perp/p_parallel > {mirror_instability_threshold(0.1):.2f}")
print(f"  At β_perp = 1.0: threshold p_perp/p_parallel > {mirror_instability_threshold(1.0):.2f}")
print(f"  At β_perp = 10:  threshold p_perp/p_parallel > {mirror_instability_threshold(10):.2f}")
print()
print("Physical interpretation:")
print("  High β_perp (strong pressure): easier to go unstable (lower threshold)")
print("  Low β_perp (weak pressure): harder to go unstable (higher threshold)")
```

### 7.3 분산 비교: MHD vs. Hall MHD vs. Kinetic

```python
import numpy as np
import matplotlib.pyplot as plt

# Plasma parameters
n = 1e19
B = 0.1
T_e = 10  # eV
T_i = 10

e = 1.6e-19
m_i = 1.67e-27
m_e = 9.11e-31
mu_0 = 4e-7 * np.pi
k_B = 1.38e-23

# Derived quantities
omega_ci = e * B / m_i
omega_ce = e * B / m_e
omega_pi = np.sqrt(n * e**2 / (m_i * 8.85e-12))

v_A = B / np.sqrt(mu_0 * n * m_i)
c_s = np.sqrt(k_B * (T_e + T_i) * e / m_i)
d_i = 3e8 / omega_pi

v_th_e = np.sqrt(2 * k_B * T_e * e / m_e)
v_th_i = np.sqrt(2 * k_B * T_i * e / m_i)

print("Plasma parameters:")
print(f"  V_A = {v_A:.2e} m/s")
print(f"  c_s = {c_s:.2e} m/s")
print(f"  d_i = {d_i:.2e} m")
print(f"  ω_ci = {omega_ci:.2e} rad/s")
print()

# Wavenumber range
k = np.logspace(-3, 3, 500) / d_i  # normalized to d_i

# 1. MHD Alfvén wave
omega_MHD = k * v_A / omega_ci * (k * d_i)  # normalized to omega_ci

# 2. Hall MHD (whistler)
omega_Hall = k * v_A / omega_ci * (k * d_i) * np.sqrt(1 + (k * d_i)**2)

# 3. Kinetic Alfvén wave (warm plasma, with electron Landau damping)
# Approximate dispersion (electrostatic limit)
k_perp = k / 2  # assume oblique
rho_s = c_s / omega_ci
omega_KAW = k * v_A / omega_ci * (k * d_i) * np.sqrt(1 + (k_perp * d_i * rho_s / d_i)**2)

# 4. Ion acoustic wave
omega_ion_acoustic = k * c_s / omega_ci * (k * d_i)

# Plot
fig, ax = plt.subplots(figsize=(11, 7))

ax.loglog(k * d_i, omega_MHD, 'b-', linewidth=3, label='MHD Alfvén: $\omega = k_\parallel V_A$')
ax.loglog(k * d_i, omega_Hall, 'r--', linewidth=3, label='Hall MHD (whistler): $\omega = k_\parallel V_A \sqrt{1+(kd_i)^2}$')
ax.loglog(k * d_i, omega_KAW, 'g-.', linewidth=3, label='Kinetic Alfvén (warm)')
ax.loglog(k * d_i, omega_ion_acoustic, 'm:', linewidth=3, label='Ion acoustic: $\omega = k c_s$')

# Reference lines
ax.axvline(1, color='k', linestyle=':', alpha=0.5, linewidth=2, label='$k d_i = 1$')
ax.axhline(1, color='gray', linestyle=':', alpha=0.5, linewidth=2, label='$\omega = \omega_{ci}$')

# Asymptotic slopes
k_ref = np.logspace(-2, 0, 50)
ax.loglog(k_ref * d_i, (k_ref * d_i)**1 * 0.01, 'k--', alpha=0.4, label='slope = 1')
ax.loglog(k_ref * d_i * 10, (k_ref * d_i * 10)**2 * 0.001, 'k-.', alpha=0.4, label='slope = 2')

ax.set_xlabel(r'$k d_i$ (normalized wavenumber)', fontsize=13)
ax.set_ylabel(r'$\omega / \omega_{ci}$ (normalized frequency)', fontsize=13)
ax.set_title('Dispersion Relations: MHD vs. Hall MHD vs. Kinetic', fontsize=14)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1e-3, 1e3)
ax.set_ylim(1e-4, 1e2)

plt.tight_layout()
plt.savefig('dispersion_comparison.png', dpi=150)
plt.show()

print("Dispersion relations:")
print("  MHD: ω ∝ k (linear, non-dispersive)")
print("  Hall MHD: ω ∝ k² at k d_i >> 1 (whistler, dispersive)")
print("  Kinetic: includes Landau damping (not shown, requires complex ω)")
```

## 요약

이 수업에서 우리는 운동 이론에서 MHD로의 체계적 축소를 추적했습니다:

1. **이유체에서 단일 유체로**: 전자와 이온 방정식을 결합함으로써, MHD 운동량과 연속 방정식을 얻습니다. 핵심 단계는 일반화된 Ohm의 법칙에서 저항, Hall, 압력, 관성 항을 제거하여 이상적 Ohm의 법칙 $\mathbf{E} + \mathbf{v} \times \mathbf{B} = 0$을 유도하는 것입니다.

2. **유효 조건**: MHD는 저주파($\omega \ll \omega_{ci}$), 큰 스케일($L \gg \rho_i$), 충돌적($\lambda_{mfp} \ll L$), 비상대론적($v \ll c$), 준중성($L \gg \lambda_D$), 높은-$R_m$ 플라즈마에 유효합니다. 위반은 확장 MHD 또는 운동학적 모델이 필요합니다.

3. **CGL 모델**: 무충돌 플라즈마의 경우, 압력은 비등방적입니다($p_\parallel \neq p_\perp$). CGL (이중 단열) 닫힘은 단열 불변량의 보존을 사용합니다: $p_\perp/(nB) = \text{const}$와 $p_\parallel B^2 / n^3 = \text{const}$. 이는 거울과 firehose 불안정성을 예측합니다.

4. **Drift-kinetic과 gyrokinetic**: 이 5D 모델들은 gyrophase에 대한 평균을 하면서 운동학적 효과를 유지합니다. Drift-kinetic은 바운스 역학을 포착하고; gyrokinetic은 FLR 효과를 포함하며 토카막 난류 시뮬레이션에 사용됩니다.

5. **확장 MHD**: Hall MHD, 이온도 MHD, FLR-MHD, 전자 MHD는 복잡성이 증가하는 대가로 추가 물리를 포착하기 위해 표준 MHD를 확장합니다.

6. **모델 비교**: 각 모델은 유효 영역을 가집니다. 선택은 스케일, 주파수, 관심 물리에 의존합니다. MHD는 간단하고 대규모 역학을 포착하고; 운동 이론은 포괄적이지만 계산적으로 비용이 많이 듭니다.

플라즈마 모델의 계층을 이해하는 것은 주어진 문제에 적절한 설명 수준을 선택하는 데 필수적입니다.

## 연습 문제

### 문제 1: 일반화된 Ohm의 법칙에서 이상적 MHD
일반화된 Ohm의 법칙에서 시작:
$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \eta \mathbf{J} + \frac{1}{en} \mathbf{J} \times \mathbf{B} - \frac{1}{en} \nabla p_e + \frac{m_e}{e^2 n^2} \frac{d \mathbf{J}}{dt}$$
$n = 10^{20}$ m$^{-3}$, $T_e = 10$ keV, $B = 5$ T, $L = 1$ m, $V = 100$ m/s인 토카막 플라즈마의 경우:
(a) 특성 시간 스케일 $\tau = L/V$를 계산하십시오.
(b) 좌변에 대한 우변의 각 항의 크기를 추정하십시오.
(c) 이상적 MHD를 얻기 위해 어떤 항을 무시할 수 있습니까? 답을 정당화하십시오.

### 문제 2: CGL 압력 진화
무충돌 플라즈마가 밀도를 일정하게 유지하면서($n = n_0$) 자기장을 $B_0$에서 $2B_0$로 증가시켜 단열적으로 압축됩니다.
(a) CGL 방정식을 사용하여, 초기 값으로 $p_\perp$와 $p_\parallel$의 최종 값을 찾으십시오.
(b) 처음에 $p_{\perp 0} = p_{\parallel 0} = p_0$이면, 압축 후 비등방성 비율 $p_\perp / p_\parallel$은 무엇입니까?
(c) $\beta_{\perp 0} = 0.5$의 경우, 압축된 플라즈마가 거울 불안정성 임계값을 초과합니까?

### 문제 3: 동결 자속
이상적 MHD에서, 유체와 함께 움직이는 임의의 닫힌 루프를 통과하는 자기 자속이 보존됩니다:
$$\frac{d\Phi}{dt} = 0, \quad \text{여기서 } \Phi = \int_S \mathbf{B} \cdot d\mathbf{A}$$
(a) 이상적 Ohm의 법칙과 유도 방정식을 사용하여 이 **동결 정리**를 증명하십시오.
(b) 초기 반경 $r_0 = 10$ cm의 원형 자속관이 자기장 $B_0 = 0.1$ T를 가집니다. 플라즈마가 반경 방향으로 $r = 5$ cm로 압축됩니다. 최종 자기장은 무엇입니까(비압축 흐름 가정)?
(c) "동결"의 물리적 의미는 무엇입니까? 이상적 MHD에서 장선이 재결합할 수 있습니까?

### 문제 4: Gyrokinetic 순서
Gyrokinetic 이론에서, 순서는:
$$\frac{\rho_i}{L} \sim \frac{\omega}{\omega_{ci}} \sim \frac{\delta f}{f_0} \sim \delta \ll 1$$
(a) $L = 1$ m, $\rho_i = 5$ mm인 토카막의 경우, $\delta$는 무엇입니까?
(b) $\omega_{ci} = 10^8$ rad/s이면, gyrokinetics에 의해 해결되는 최대 주파수는 무엇입니까?
(c) 빠른 자기음파가 상한 주파수 제한 없이 $\omega \sim k V_A$를 가집니다. 왜 gyrokinetics가 이 파동을 포착할 수 없습니까?

### 문제 5: Hall MHD 재결합
저항 MHD에서, Sweet-Parker 재결합 속도는:
$$V_{in} \sim \frac{\eta}{L} \sim \frac{V_A}{S^{1/2}}$$
여기서 $S = L V_A / \eta$는 Lundquist 수입니다.

Hall MHD에서, 재결합 속도는 (Petschek):
$$V_{in} \sim 0.1 V_A$$
저항률과 무관합니다!

(a) $B = 0.01$ T, $n = 10^{16}$ m$^{-3}$, $L = 10^4$ km, $T_e = 10^6$ K인 태양 플레어의 경우, Alfvén 속도와 이온 skin depth를 계산하십시오.
(b) Sweet-Parker 재결합 시간 $\tau_{SP} \sim L / V_{in}$을 추정하십시오 (Spitzer 저항률 사용).
(c) Hall MHD 재결합 시간 $\tau_{Hall}$을 추정하십시오.
(d) 태양 플레어는 몇 분의 시간 스케일에서 에너지를 방출합니다. 어떤 모델이 관찰과 일치합니까?

---

**이전**: [Two-Fluid Model](./13_Two_Fluid_Model.md) | **다음**: [Plasma Diagnostics](./15_Plasma_Diagnostics.md)
