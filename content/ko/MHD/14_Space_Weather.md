# 14. 우주 기상 MHD

## 학습 목표

- 지구 자기권 구조 및 태양풍 상호작용 이해
- Magnetopause standoff 거리 및 bow shock 형성 분석
- Dungey cycle 및 자기 재결합 과정 설명
- 자기 폭풍 및 Dst index 모델링
- CME 전파, 행성간 충격파 및 도착 예측 연구
- 지자기 유도 전류 (GIC) 및 우주 기상 영향 평가
- 자기권 물리 및 우주 기상 예보를 위한 Python 모델 구현

## 1. 우주 기상 소개

우주 기상은 우주 기반 및 지상 기반 기술 시스템의 성능과 신뢰성에 영향을 미치고 인간의 생명이나 건강을 위협할 수 있는 태양 및 우주의 가변적인 조건을 의미합니다.

### 1.1 태양-지구 물리

태양-지구 시스템은 태양 코로나 ($T \sim 10^6$ K, $B \sim 1-100$ G)에서 지구 자기권 ($B \sim 0.01-1$ G) 및 전리층 ($n_e \sim 10^{11}$ m$^{-3}$)까지 걸친 결합된 MHD 시스템입니다.

**주요 구성요소:**
- **태양풍**: 태양으로부터의 초음속, super-Alfvénic 플라즈마 흐름
- **행성간 자기장 (IMF)**: 태양풍에 의해 운반되는 frozen-in 자기장
- **지구 자기권**: 지구 자기장이 지배하는 영역
- **Magnetopause**: 태양풍과 자기권 사이의 경계
- **Bow shock**: 태양풍이 자기권을 만나는 충격파

### 1.2 태양풍 매개변수

1 AU (지구 궤도)에서의 일반적인 태양풍 조건:

```
속도:          v_sw ~ 300-800 km/s (느린/빠른 풍)
밀도:          n_sw ~ 5-10 cm^-3
온도:          T_sw ~ 10^5 K
자기장:        B_sw ~ 5 nT
동압:          P_dyn = ρ v² ~ 1-5 nPa
```

태양풍은 특히 태양 폭풍 동안 매우 가변적입니다.

### 1.3 우주 기상 위험

**영향:**
1. **위성 운영**: 복사 손상, 표면 충전, 열권 가열로 인한 항력
2. **통신**: HF 라디오 블랙아웃, GPS 오류
3. **전력망**: 지자기 유도 전류 (GIC)가 변압기 손상 가능
4. **항공**: 고고도에서의 복사 노출, 통신 중단
5. **인간 건강**: 우주 비행사 복사 노출

주요 사건:
- **Carrington Event (1859)**: 기록된 최대 지자기 폭풍
- **Quebec 정전 (1989)**: GIC 유도 전력 정전으로 수백만 명 영향
- **Halloween storms (2003)**: 위성 이상, 전력망 교란
- **Bastille Day storm (2000)**: 통신 중단

## 2. 지구 자기권

### 2.1 쌍극자 자기장

지구의 고유 자기장은 대략 자기 쌍극자입니다:

```
B_r = -2 B_0 (R_E/r)³ cos θ
B_θ = -B_0 (R_E/r)³ sin θ
```

여기서:
- $B_0 \approx 3.12 \times 10^{-5}$ T (적도 표면 장)
- $R_E = 6371$ km (지구 반경)
- $r$은 방사 거리, $\theta$는 자기 여위도

쌍극자 모멘트:

```
M_E ≈ 8 × 10^{22} A m²
```

태양풍이 없으면 쌍극자 장은 무한대까지 확장됩니다. 그러나 태양풍 압력은 낮 쪽에서 장을 압축하고 밤 쪽에서 늘립니다.

### 2.2 Magnetopause

**Magnetopause**는 태양풍 동압이 지구장의 자기 압력과 균형을 이루는 경계입니다.

**압력 균형:**

```
P_dyn = B²/(2 μ₀)
```

여기서 $B$는 magnetopause에서의 자기권 장입니다.

Subsolar point (자기권의 nose)의 경우:

```
ρ_sw v_sw² = B_mp²/(2 μ₀)
```

### 2.3 Magnetopause Standoff 거리

Standoff 거리 $r_{mp}$ (지구 중심에서 subsolar magnetopause까지의 거리)는 압력 균형을 통해 찾아집니다.

단순 쌍극자 모델 사용:

```
B_mp ≈ B_0 (R_E/r_mp)³
```

압력 균형:

```
ρ_sw v_sw² = B_0² (R_E/r_mp)⁶ / (2 μ₀)
```

$r_{mp}$에 대해 풀기:

```
r_mp = R_E (B_0² / (2 μ₀ ρ_sw v_sw²))^(1/6)
```

일반적인 태양풍 조건 ($\rho_{sw} v_{sw}^2 \sim 2$ nPa)의 경우:

```
r_mp ~ 10-12 R_E
```

강한 태양풍 압력 (CME 도착) 동안 $r_{mp}$는 $< 8 R_E$로 압축될 수 있습니다.

### 2.4 Chapman-Ferraro 전류 시스템

Magnetopause는 $B$의 불연속이 아니라 자기권 장을 태양풍으로부터 차폐하는 얇은 전류 시트 (**Chapman-Ferraro 전류**)입니다.

$\nabla \times B = \mu_0 j$로부터 표면 전류 밀도:

```
K = (1/μ₀) ∫ j dl ≈ (B_in - B_out) / μ₀
```

여기서 $B_{in}$은 자기권 장이고 $B_{out}$는 magnetosheath 장 (bow shock 바로 안쪽)입니다.

전류는 magnetopause 주위를 흘러 지구 장을 가두는 폐루프를 생성합니다.

### 2.5 Bow Shock

태양풍은 초음속 ($M_s > 1$)이고 super-Alfvénic ($M_A > 1$)이므로, 초음속 항공기 앞의 충격파와 유사하게 magnetopause 상류에 **bow shock**이 형성됩니다.

충격파를 가로지르는 **점프 조건** (Rankine-Hugoniot 관계):

```
[ρ v_n] = 0  (질량 보존)
[ρ v_n² + p + B_t²/(2μ₀)] = 0  (운동량)
[v_n (E + p/ρ) + (v×B)_n · B_t/μ₀] = 0  (에너지)
[v_n B_t - v_t B_n] = 0  (자기장)
```

여기서 $v_n, v_t$는 normal 및 tangential 속도, $B_n, B_t$는 normal 및 tangential 장입니다.

수직 충격파 ($B \perp v$)의 경우 압축비:

```
ρ₂/ρ₁ = (γ+1) M_s² / ((γ-1) M_s² + 2)
```

$\gamma = 5/3$이고 강한 충격파 ($M_s \gg 1$)의 경우:

```
ρ₂/ρ₁ → 4
```

**Magnetosheath**는 bow shock과 magnetopause 사이의 영역으로, 충격을 받고 압축되고 가열된 태양풍 플라즈마를 포함합니다.

### 2.6 Magnetotail

밤 쪽에서 태양풍은 지구 장을 수백 $R_E$ 하류로 확장되는 긴 **magnetotail**로 늘립니다.

Magnetotail은 다음으로 구성됩니다:
- **Tail lobes**: 지구로부터 떨어져 가는 (북쪽 lobe) 그리고 지구를 향한 (남쪽 lobe) 거의 평행한 자기장선
- **Plasma sheet**: lobes를 분리하는 얇은 전류 시트로, 뜨거운 플라즈마 포함
- **Neutral sheet**: plasma sheet 중심에서 $B_z = 0$인 표면

Tail 전류 ($\pm y$ 방향으로 흐름)가 lobe 장을 유지합니다.

## 3. Dungey Cycle 및 자기 재결합

### 3.1 개방 자기권 개념

**폐쇄 자기권** 모델에서는 태양풍이 magnetopause를 관통하지 않고 지구 주위를 흐릅니다. 그러나 IMF가 남쪽 성분 ($B_z < 0$)을 가질 때, **자기 재결합**이 낮 쪽 magnetopause에서 발생하여 태양풍 플라즈마가 자기권으로 들어갈 수 있게 합니다.

**Dungey cycle** (남쪽 IMF의 경우):

1. **낮 쪽 재결합**: IMF와 자기권 자기장선이 subsolar magnetopause에서 재결합
2. **대류**: 재결합된 자기장선이 태양풍에 의해 극관 위로 쓸려감
3. **Tail 저장**: 자기장선이 magnetotail에 축적되어 에너지 저장
4. **밤 쪽 재결합**: Tail 자기장선이 plasma sheet에서 재결합
5. **복귀 흐름**: 폐쇄된 자기장선이 낮 쪽을 향해 대류하여 플라즈마 복귀

### 3.2 Magnetopause에서의 자기 재결합

재결합이 발생하려면 magnetopause의 반대쪽 자기장이 반평행 성분을 가져야 합니다.

**재결합 속도:**

재결합 전기장 $E_{rec}$는 플럭스가 전달되는 속도를 결정합니다:

```
E_rec ~ 0.1 v_sw B_sw sin²(θ/2)
```

여기서 $\theta$는 IMF clock angle ($yz$ 평면에서 IMF와 북쪽 방향 사이의 각도)입니다.

최대 재결합은 남쪽 IMF ($B_z < 0$, $\theta = 180°$)에서 발생합니다.

**Flux transfer events (FTEs):**

버스트 재결합은 magnetopause를 따라 이동하는 상호 연결된 자기장선의 플럭스 튜브를 생성합니다. 이들은 자기장 크기 및 방향의 갑작스런 펄스로 관찰됩니다.

### 3.3 자기권 대류

재결합은 자기권에서 대규모 플라즈마 순환을 구동합니다:

- **낮 쪽**: 플라즈마와 자기장선이 극쪽으로 이동
- **극관**: 극 위로 antisunward 흐름
- **밤 쪽**: plasma sheet에서 sunward 복귀 흐름
- **Ring current**: 지구 주위의 에너지 입자의 방위각 drift

대류와 관련된 전기장:

```
E_conv = -v × B
```

일반적인 대류 속도: 100-1000 m/s, $E_{conv} \sim 0.01-0.1$ mV/m 제공.

### 3.4 Substorms

**자기권 substorm**은 tail이 자기 플럭스로 과부하될 때 발생하는 magnetotail의 일시적 에너지 방출입니다.

**Substorm 단계:**

1. **성장 단계** (30-60 분): 낮 쪽 재결합이 tail에 에너지 저장, tail lobes 확장, plasma sheet 얇아짐
2. **확장 단계** (10-30 분): 폭발적 방출, 밤 쪽 재결합 (약 10-15 $R_E$ 근처), 오로라 밝아짐, dipolarization (tail 장이 더 쌍극자처럼 됨)
3. **회복 단계** (30-60 분): 시스템이 substorm 전 상태로 이완

**오로라 signature:**

Substorm 시작은 오로라의 갑작스런 밝아짐 및 확장으로 표시되며, 자기장선을 따라 전리층으로 낙하하는 에너지 전자에 의해 발생합니다.

## 4. 자기 폭풍

**지자기 폭풍**은 종종 CME 도착와 관련된 남쪽 IMF의 장기간에 의해 구동되는 지구 자기권의 주요 교란입니다.

### 4.1 Ring Current

폭풍 동안 향상된 자기권 대류는 많은 수의 에너지 이온 (10-200 keV)을 내부 자기권으로 주입합니다. 이러한 입자는 기울기 및 곡률 drift로 인해 지구 주위를 방위각으로 drift합니다:

```
v_drift = (m v_⊥² + v_∥²) / (q B²) (B × ∇B) + (m v_∥²) / (q B³) B × (b · ∇)b
```

이온의 경우: 서쪽으로 drift; 전자의 경우: 동쪽으로 drift.

순 결과는 지구 쌍극자 장에 반대하는 2-7 $R_E$ 고도의 서쪽 **ring current**입니다.

### 4.2 Dst Index

**Dst (Disturbance storm time)** index는 적도 지상 관측소에서 자기장의 수평 성분 감소를 측정하여 ring current 강도의 전역 척도를 제공합니다.

```
Dst = (4개 적도 관측소의 H 성분 편차 합) / 4
```

**일반적인 값:**
- 조용한 조건: $Dst \sim 0$ nT
- 중간 폭풍: $Dst < -50$ nT
- 강한 폭풍: $Dst < -100$ nT
- Super-storm: $Dst < -250$ nT

Carrington event (1859)는 $Dst \sim -1600$ nT에 도달한 것으로 추정됩니다.

**폭풍 단계:**

1. **초기 단계** (몇 시간): Magnetopause 압축, 장의 약간 증가 ($Dst > 0$)
2. **주요 단계** (몇 시간): Ring current 주입, $Dst$의 급격한 감소
3. **회복 단계** (며칠): 전하 교환 및 강수를 통한 ring current 감쇠

### 4.3 Dst-태양풍 결합

경험적 공식 (Burton et al. 1975):

```
dDst/dt = Q(E_sw) - Dst/τ
```

여기서:
- $Q(E_{sw})$는 태양풍 전기장 $E_{sw} = v_{sw} B_s$에 의존하는 주입 함수
- $B_s$는 IMF의 남쪽 성분 ($B_z < 0$이면 $B_s = -B_z$, 그렇지 않으면 0)
- $\tau \sim 8$ 시간은 감쇠 시간 척도

**Burton 공식:**

```
Q = a (v_sw B_s - b)  if v_sw B_s > b, else 0
```

여기서 $a \sim 10^{-3}$ nT/(mV/m)이고 $b \sim 0.5$ mV/m는 경험적 상수입니다.

이 단순 모델은 주요 특징을 포착합니다: 더 높은 $v_{sw}$와 남쪽 $B_z$에 대해 더 강한 폭풍.

### 4.4 폭풍 영향

**복사 벨트 향상:**

폭풍은 외부 복사 벨트 (L=4-7)의 전자를 에너지화하여 위성에 위험을 생성합니다.

**전리층 교란:**

향상된 전류 및 입자 강수가 전리층을 교란하여 라디오 통신 및 GPS 정확도에 영향을 줍니다.

**오로라 확장:**

오로라는 주요 폭풍 동안 중위도 (40-50° 위도)로 확장될 수 있습니다.

## 5. 코로나 질량 방출 (CMEs)

### 5.1 CME 특성

**코로나 질량 방출 (CME)**은 태양 코로나로부터의 플라즈마와 자기장의 분출로, 200-3000 km/s의 속도로 $10^{12}-10^{13}$ kg의 물질을 방출합니다.

**트리거 메커니즘:**
- 자기 플럭스 로프 분출
- 코로나 자기장의 shearing 및 twisting
- 평형 상실 (예: torus 불안정성)

**CME 구조:**
- **Leading edge**: Coronagraph 이미지의 밝은 가장자리
- **Cavity**: 어두운 영역 (낮은 밀도)
- **Core**: 밝은 코어 (prominence 물질)

### 5.2 행성간 CME (ICME)

CME가 행성간 공간으로 전파되면 **ICME**라고 합니다. 지구 (1 AU)에서 ICMEs는 특징적인 signatures를 가집니다:

**Magnetic cloud:**

~1일 동안 자기장 벡터의 부드러운 회전, 낮은 플라즈마 beta ($\beta < 1$), 그리고 낮은 온도를 가진 ICMEs의 부분집합. 이것은 플럭스 로프 구조를 나타냅니다.

**Sheath:**

CME가 빠른 경우 ($v > v_{sw}$), 앞에 충격파를 구동합니다. 충격파와 플럭스 로프 사이의 영역은 **sheath**로, 압축되고 난류적인 태양풍을 포함합니다.

### 5.3 CME 전파 모델

**Drag 기반 모델:**

CME는 태양풍에서 공기역학적 drag를 경험합니다:

```
dv/dt = -γ (v - v_sw)
```

여기서 $\gamma$는 drag 계수입니다:

```
γ ≈ C_d A / (2 M)
```

- $C_d \sim 1$은 drag 계수
- $A$는 CME 단면적
- $M$은 CME 질량

**해석적 해:**

```
v(t) = v_sw + (v_0 - v_sw) exp(-γ t)
```

여기서 $v_0$는 초기 CME 속도입니다.

이동한 거리:

```
r(t) = r_0 + v_sw t + (v_0 - v_sw)/γ (1 - exp(-γ t))
```

**지구까지의 전송 시간:**

$r(t_{arr}) = 1$ AU로 설정하고 $t_{arr}$에 대해 풀기.

일반적인 매개변수 ($v_0 = 1000$ km/s, $v_{sw} = 400$ km/s, $\gamma^{-1} = 1$ day)의 경우:

```
t_arr ~ 2-3 days
```

더 빠른 CMEs는 1-2일에 도착; 더 느린 CMEs는 3-5일.

### 5.4 MHD 시뮬레이션 모델

고급 우주 기상 예보는 3D MHD 코드를 사용하여 CME 전파를 시뮬레이션합니다:

**ENLIL** (NOAA):
- 태양에서 2 AU까지의 3D MHD 모델
- SOHO, ACE의 태양풍 데이터 사용
- 도착 시간, 속도, 밀도 예보 제공

**SUSANOO** (일본):
- 적응 메시 개선을 사용한 전역 MHD 시뮬레이션
- CME-태양풍 상호작용 모델링

**기타 모델:**
- WSA-ENLIL (결합된 코로나-태양권 모델)
- EUHFORIA (유럽 태양권 모델)

이러한 모델은 일반적으로 ~6-12시간 정확도로 도착 시간을 예측합니다.

### 5.5 CME 방향 및 지구효과성

CME의 **지구효과성**은 자기장의 방향에 결정적으로 의존합니다.

**남쪽 장 ($B_z < 0$):**
- 강한 낮 쪽 재결합
- 자기권으로의 효율적인 에너지 전달
- 강한 폭풍 (큰 음의 $Dst$)

**북쪽 장 ($B_z > 0$):**
- 약한 또는 없는 낮 쪽 재결합
- 최소 지자기 활동

CME 장 방향은 L1 (지구로부터 150만 km, ~1시간 경고)에 도착할 때만 측정할 수 있으므로 폭풍 강도 예보는 어렵습니다.

## 6. 지자기 유도 전류 (GIC)

### 6.1 물리적 메커니즘

자기권 자기장의 급격한 변화 (폭풍 또는 substorms 동안)는 Faraday's law를 통해 전도성 지구에 전기장을 유도합니다:

```
∇ × E = -∂B/∂t
```

지면에서 공간적으로 균일한 시간 변화 장 $B(t)$의 경우:

```
E ~ -L ∂B/∂t
```

여기서 $L$은 특성 길이 척도입니다.

이러한 전기장은 전도체: 전력선, 파이프라인, 철도 선로 등에서 전류를 구동합니다.

### 6.2 지면 전도도

유도된 전기장은 지면 전도도 구조에 의존합니다:

```
E(ω) = Z(ω) · H(ω)
```

여기서 $Z(\omega)$는 **표면 임피던스** (지면 전도도 프로파일에 의존)이고 $H$는 수평 자기장 섭동입니다.

높은 저항률을 가진 영역 (예: 캐나다, 스칸디나비아의 Precambrian shield 암석)은 큰 GIC에 더 취약합니다.

### 6.3 전력망의 GIC

GIC는 준-DC 전류로 변압기로 흘러들어 다음을 유발합니다:

1. **Half-cycle saturation**: DC 전류가 변압기 코어를 편향시켜 비대칭 자화로 이어짐
2. **증가된 무효 전력 수요**
3. **가열**: 과도한 가열이 변압기 손상 가능
4. **고조파**: AC 파형 왜곡
5. **전압 불안정성**: cascading failures 유발 가능

**Quebec 정전 (1989년 3월 13일):**

주요 지자기 폭풍이 Hydro-Québec 전력망에 GIC를 유도했습니다. 90초 이내에 전체 전력망이 붕괴하여 600만 명이 최대 9시간 동안 전력 없이 지냈습니다.

### 6.4 GIC 크기 추정

경험적 스케일링:

```
GIC ~ (dB/dt) / R_earth
```

여기서 $R_{earth}$는 지구의 유효 저항 (지면 전도도에 의존)입니다.

$dB/dt \sim 1000$ nT/min이고 저항률 $\rho \sim 1000$ Ω·m인 폭풍의 경우:

```
E ~ 1-10 V/km
```

100 km 송전선에 걸쳐:

```
V ~ 100-1000 V
```

선 저항 $R \sim 0.1$ Ω의 경우:

```
GIC ~ 100-1000 A
```

이것은 AC 전력망에 겹쳐진 준-DC입니다.

### 6.5 GIC 완화

**전략:**
1. **운영 절차**: 폭풍 동안 부하 감소
2. **중성점 차단 장치**: DC를 차단하면서 AC를 통과시키는 커패시터 삽입
3. **네트워크 재구성**: 취약한 연결 개방
4. **개선된 예보**: 사전 경고를 통해 운영자가 준비 가능

최근 초점: 전력망 복원력 계획을 위한 100년 및 500년 GIC 사건 이해.

## 7. 우주 기상 예보

### 7.1 관측 자산

**태양 관측:**
- **SOHO, SDO**: 태양 이미징 (EUV, coronagraph)
- **STEREO**: 3D CME 구조

**태양풍 모니터:**
- **ACE, DSCOVR**: L1 (태양쪽 150만 km)에 위치, ~30-60분 경고 제공
- 지구에 도달하기 전에 태양풍의 $v, n, B$ 측정

**자기권 모니터링:**
- **지상 자력계**: 전역 네트워크 (SuperMAG)
- **위성**: GOES, THEMIS, MMS

### 7.2 예보 워크플로

1. **태양 모니터링**: 플레어, CMEs 감지
2. **CME 전파 모델**: MHD 또는 경험적 모델을 사용하여 도착 시간 추정
3. **L1 데이터 동화**: ICME가 L1에 도달할 때 예측 개선
4. **지자기 지수**: 태양풍 결합 함수를 기반으로 Kp, Dst 예측
5. **영향 평가**: GIC, 복사 벨트, 전리층 효과 추정

### 7.3 예보 기술

**도착 시간**: 일반적으로 CMEs에 대해 ±6-12시간
**강도 (Dst)**: 상관관계 ~0.7-0.8 (알 수 없는 CME $B_z$ 방향에 의해 제한됨)
**확률적 예보**: 앙상블 방법이 신뢰성 개선

### 7.4 운영 센터

- **NOAA Space Weather Prediction Center (SWPC)**: 미국 운영 예보
- **ESA Space Situational Awareness (SSA)**: 유럽 예보
- **UKMO Space Weather Operations Centre (MOSWOC)**: 영국 예보
- **ISES (International Space Environment Service)**: 전역 조정

## 8. Python 구현

### 8.1 Magnetopause Standoff 거리

```python
import numpy as np
import matplotlib.pyplot as plt

def magnetopause_standoff(v_sw, n_sw, B_0=3.12e-5, R_E=6371e3):
    """
    Calculate magnetopause standoff distance.

    Parameters:
    v_sw : solar wind speed (m/s)
    n_sw : solar wind density (m^-3)
    B_0 : Earth's equatorial surface field (T)
    R_E : Earth radius (m)

    Returns:
    r_mp : magnetopause standoff distance (R_E)
    """
    mu_0 = 4 * np.pi * 1e-7
    m_p = 1.673e-27  # proton mass (kg)

    # Dynamic pressure
    rho_sw = n_sw * m_p
    P_dyn = rho_sw * v_sw**2

    # Standoff distance
    r_mp = R_E * (B_0**2 / (2 * mu_0 * P_dyn))**(1/6)

    return r_mp / R_E  # Return in Earth radii

# Typical solar wind conditions
v_typical = 400e3  # m/s
n_typical = 5e6    # m^-3

r_mp_typical = magnetopause_standoff(v_typical, n_typical)
print(f"Typical solar wind: v = {v_typical/1e3:.0f} km/s, n = {n_typical/1e6:.1f} cm^-3")
print(f"Magnetopause standoff: r_mp = {r_mp_typical:.1f} R_E")

# CME impact (enhanced pressure)
v_cme = 800e3  # m/s
n_cme = 20e6   # m^-3

r_mp_cme = magnetopause_standoff(v_cme, n_cme)
print(f"\nCME arrival: v = {v_cme/1e3:.0f} km/s, n = {n_cme/1e6:.1f} cm^-3")
print(f"Magnetopause standoff: r_mp = {r_mp_cme:.1f} R_E (compressed!)")

# Parametric study: vary solar wind speed
v_scan = np.linspace(300e3, 1000e3, 100)
r_mp_scan = [magnetopause_standoff(v, n_typical) for v in v_scan]

plt.figure(figsize=(10, 6))
plt.plot(v_scan/1e3, r_mp_scan, 'b-', linewidth=2)
plt.axhline(y=r_mp_typical, color='g', linestyle='--', linewidth=1.5, label='Typical conditions')
plt.axhline(y=8, color='r', linestyle='--', linewidth=1.5, label='Geosynchronous orbit')
plt.xlabel('Solar Wind Speed (km/s)', fontsize=12)
plt.ylabel('Magnetopause Standoff Distance (R$_E$)', fontsize=12)
plt.title('Magnetopause Position vs Solar Wind Speed', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('magnetopause_standoff.png', dpi=150)
plt.show()
```

### 8.2 Bow Shock 특성

```python
def bow_shock_jump(M_s, gamma=5/3):
    """
    Calculate density compression ratio across bow shock.

    Parameters:
    M_s : sonic Mach number
    gamma : adiabatic index

    Returns:
    r : compression ratio (ρ₂/ρ₁)
    """
    r = (gamma + 1) * M_s**2 / ((gamma - 1) * M_s**2 + 2)
    return r

def sonic_mach_number(v_sw, T_sw):
    """
    Calculate sonic Mach number of solar wind.

    Parameters:
    v_sw : solar wind speed (m/s)
    T_sw : solar wind temperature (K)

    Returns:
    M_s : sonic Mach number
    """
    k_B = 1.381e-23  # J/K
    m_p = 1.673e-27  # kg
    gamma = 5/3

    c_s = np.sqrt(gamma * k_B * T_sw / m_p)
    M_s = v_sw / c_s

    return M_s

# Typical solar wind
T_sw = 1e5  # K
M_s_typical = sonic_mach_number(v_typical, T_sw)
r_typical = bow_shock_jump(M_s_typical)

print(f"\nBow shock (typical solar wind):")
print(f"Sonic Mach number: M_s = {M_s_typical:.1f}")
print(f"Density compression: ρ₂/ρ₁ = {r_typical:.2f}")

# Fast solar wind (CME)
M_s_cme = sonic_mach_number(v_cme, T_sw)
r_cme = bow_shock_jump(M_s_cme)

print(f"\nBow shock (CME):")
print(f"Sonic Mach number: M_s = {M_s_cme:.1f}")
print(f"Density compression: ρ₂/ρ₁ = {r_cme:.2f}")

# Mach number scan
M_s_scan = np.linspace(1.5, 10, 100)
r_scan = [bow_shock_jump(M) for M in M_s_scan]

plt.figure(figsize=(10, 6))
plt.plot(M_s_scan, r_scan, 'r-', linewidth=2)
plt.axhline(y=4, color='k', linestyle='--', linewidth=1.5, label='Strong shock limit (r=4)')
plt.xlabel('Sonic Mach Number $M_s$', fontsize=12)
plt.ylabel('Density Compression Ratio', fontsize=12)
plt.title('Bow Shock Density Jump vs Mach Number', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bow_shock_jump.png', dpi=150)
plt.show()
```

### 8.3 Dst Index Model (Burton)

```python
def dst_evolution(Dst_0, v_sw_series, B_z_series, dt, a=1e-3, b=0.5, tau=8*3600):
    """
    Evolve Dst index using Burton model.

    Parameters:
    Dst_0 : initial Dst (nT)
    v_sw_series : solar wind speed time series (m/s)
    B_z_series : IMF Bz time series (T)
    dt : timestep (s)
    a : injection parameter (nT/(mV/m))
    b : threshold (mV/m)
    tau : decay timescale (s)

    Returns:
    Dst_series : Dst evolution (nT)
    """
    N = len(v_sw_series)
    Dst_series = np.zeros(N)
    Dst = Dst_0

    for i in range(N):
        Dst_series[i] = Dst

        # Solar wind electric field (in mV/m)
        B_s = max(-B_z_series[i], 0) * 1e9  # Convert to nT, take southward component
        E_sw = v_sw_series[i] / 1e3 * B_s / 1e6  # mV/m

        # Injection function
        if E_sw > b:
            Q = a * (E_sw - b)
        else:
            Q = 0

        # Burton equation
        dDst_dt = Q - Dst / tau

        Dst += dDst_dt * dt

    return Dst_series

# Simulate a magnetic storm
t_max = 5 * 24 * 3600  # 5 days
dt = 600  # 10 min
N = int(t_max / dt)
t_series = np.arange(N) * dt / 3600  # hours

# Solar wind scenario: CME arrival at t=12 hours
v_sw_series = np.ones(N) * 400e3  # m/s
B_z_series = np.ones(N) * 2e-9  # T (northward)

# CME arrival: 12-36 hours, enhanced speed and southward field
t_cme_start = int(12 * 3600 / dt)
t_cme_end = int(36 * 3600 / dt)
v_sw_series[t_cme_start:t_cme_end] = 600e3  # m/s
B_z_series[t_cme_start:t_cme_end] = -15e-9  # T (strongly southward)

# Evolve Dst
Dst_0 = 0  # nT (quiet conditions)
Dst_series = dst_evolution(Dst_0, v_sw_series, B_z_series, dt)

print(f"\nDst storm simulation:")
print(f"Minimum Dst: {np.min(Dst_series):.1f} nT at t = {t_series[np.argmin(Dst_series)]:.1f} hours")

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax1.plot(t_series, v_sw_series/1e3, 'b-', linewidth=1.5)
ax1.set_ylabel('Solar Wind Speed (km/s)', fontsize=12)
ax1.set_title('Magnetic Storm Simulation (Burton Dst Model)', fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.plot(t_series, B_z_series*1e9, 'g-', linewidth=1.5)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax2.set_ylabel('IMF $B_z$ (nT)', fontsize=12)
ax2.grid(True, alpha=0.3)

ax3.plot(t_series, Dst_series, 'r-', linewidth=2)
ax3.axhline(y=-50, color='orange', linestyle='--', linewidth=1, label='Moderate storm')
ax3.axhline(y=-100, color='red', linestyle='--', linewidth=1, label='Intense storm')
ax3.set_xlabel('Time (hours)', fontsize=12)
ax3.set_ylabel('Dst (nT)', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dst_storm_simulation.png', dpi=150)
plt.show()
```

### 8.4 CME Transit Time

```python
def cme_transit_time(v_0, v_sw=400e3, gamma_inv=86400, r_target=1.496e11):
    """
    Calculate CME arrival time using drag model.

    Parameters:
    v_0 : initial CME speed (m/s)
    v_sw : solar wind speed (m/s)
    gamma_inv : inverse drag coefficient (s)
    r_target : target distance (m, default 1 AU)

    Returns:
    t_arr : arrival time (hours)
    """
    # Solve r(t) = r_target for t
    # r(t) = v_sw * t + (v_0 - v_sw) * gamma_inv * (1 - exp(-t/gamma_inv))
    # Iterative solution
    t = 0
    dt = 600  # 10 min
    r = 0
    r_0 = 0.1 * r_target  # Start at 0.1 AU (close to Sun)

    while r < r_target:
        v = v_sw + (v_0 - v_sw) * np.exp(-t / gamma_inv)
        r += v * dt
        t += dt

    return t / 3600  # Convert to hours

# CME scenarios
v_0_slow = 500e3  # m/s
v_0_fast = 1200e3  # m/s

t_arr_slow = cme_transit_time(v_0_slow)
t_arr_fast = cme_transit_time(v_0_fast)

print(f"\nCME transit time to Earth (1 AU):")
print(f"Slow CME (v₀ = {v_0_slow/1e3:.0f} km/s): {t_arr_slow:.1f} hours ({t_arr_slow/24:.1f} days)")
print(f"Fast CME (v₀ = {v_0_fast/1e3:.0f} km/s): {t_arr_fast:.1f} hours ({t_arr_fast/24:.1f} days)")

# Parametric study
v_0_scan = np.linspace(400e3, 2000e3, 50)
t_arr_scan = [cme_transit_time(v_0) for v_0 in v_0_scan]

plt.figure(figsize=(10, 6))
plt.plot(v_0_scan/1e3, np.array(t_arr_scan)/24, 'b-', linewidth=2)
plt.xlabel('Initial CME Speed (km/s)', fontsize=12)
plt.ylabel('Transit Time to 1 AU (days)', fontsize=12)
plt.title('CME Arrival Time vs Initial Speed (Drag Model)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cme_transit_time.png', dpi=150)
plt.show()
```

### 8.5 GIC 추정

```python
def gic_estimate(dB_dt, rho_earth=1000, L=100e3):
    """
    Estimate geomagnetically induced current.

    Parameters:
    dB_dt : magnetic field time derivative (nT/min)
    rho_earth : Earth resistivity (Ω·m)
    L : transmission line length (m)

    Returns:
    E : induced electric field (V/km)
    GIC : induced current (A, assuming 0.1 Ω line resistance)
    """
    mu_0 = 4 * np.pi * 1e-7

    # Induced electric field (rough estimate)
    # E ~ (dB/dt) * sqrt(ρ/(2πμ₀f))
    # For quasi-DC, use simplified scaling
    E = (dB_dt * 1e-9 / 60) * np.sqrt(rho_earth / mu_0) / 1000  # V/km

    # Voltage over line
    V = E * L / 1e3  # V

    # Current (assuming line resistance R ~ 0.1 Ω)
    R_line = 0.1  # Ω
    GIC = V / R_line  # A

    return E, GIC

# Quebec blackout scenario
dB_dt_quebec = 480  # nT/min (observed)

E_quebec, GIC_quebec = gic_estimate(dB_dt_quebec)
print(f"\nQuebec blackout (1989-03-13):")
print(f"dB/dt = {dB_dt_quebec} nT/min")
print(f"Induced electric field: E ~ {E_quebec:.2f} V/km")
print(f"GIC (100 km line): ~ {GIC_quebec:.0f} A")

# Carrington event estimate
dB_dt_carrington = 5000  # nT/min (estimated)

E_carrington, GIC_carrington = gic_estimate(dB_dt_carrington)
print(f"\nCarrington event (1859, estimated):")
print(f"dB/dt = {dB_dt_carrington} nT/min")
print(f"Induced electric field: E ~ {E_carrington:.2f} V/km")
print(f"GIC (100 km line): ~ {GIC_carrington:.0f} A")

# Parametric study
dB_dt_scan = np.linspace(10, 2000, 100)
GIC_scan = [gic_estimate(dB_dt)[1] for dB_dt in dB_dt_scan]

plt.figure(figsize=(10, 6))
plt.plot(dB_dt_scan, GIC_scan, 'm-', linewidth=2)
plt.axhline(y=100, color='orange', linestyle='--', linewidth=1.5, label='Concern level (~100 A)')
plt.axvline(x=dB_dt_quebec, color='r', linestyle='--', linewidth=1.5, label='Quebec 1989')
plt.xlabel('dB/dt (nT/min)', fontsize=12)
plt.ylabel('GIC (A)', fontsize=12)
plt.title('Geomagnetically Induced Current vs dB/dt', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gic_estimate.png', dpi=150)
plt.show()
```

### 8.6 Reconnection Electric Field

```python
def reconnection_electric_field(v_sw, B_sw, theta):
    """
    Calculate reconnection electric field at magnetopause.

    Parameters:
    v_sw : solar wind speed (m/s)
    B_sw : IMF magnitude (T)
    theta : IMF clock angle (degrees, 0=northward, 180=southward)

    Returns:
    E_rec : reconnection electric field (mV/m)
    """
    theta_rad = np.deg2rad(theta)
    E_rec = 0.1 * v_sw / 1e3 * B_sw * 1e9 * np.sin(theta_rad / 2)**2  # mV/m
    return E_rec

# Northward IMF
E_rec_north = reconnection_electric_field(v_typical, 5e-9, 0)
print(f"\nReconnection electric field:")
print(f"Northward IMF (θ=0°): E_rec = {E_rec_north:.3f} mV/m (minimal reconnection)")

# Southward IMF
E_rec_south = reconnection_electric_field(v_typical, 5e-9, 180)
print(f"Southward IMF (θ=180°): E_rec = {E_rec_south:.2f} mV/m (strong reconnection)")

# Strong southward IMF (CME)
E_rec_cme = reconnection_electric_field(v_cme, 20e-9, 180)
print(f"CME with southward field: E_rec = {E_rec_cme:.2f} mV/m (very strong!)")

# Clock angle scan
theta_scan = np.linspace(0, 180, 100)
E_rec_scan = [reconnection_electric_field(v_typical, 5e-9, theta) for theta in theta_scan]

plt.figure(figsize=(10, 6))
plt.plot(theta_scan, E_rec_scan, 'purple', linewidth=2)
plt.xlabel('IMF Clock Angle (degrees)', fontsize=12)
plt.ylabel('Reconnection E-field (mV/m)', fontsize=12)
plt.title('Magnetopause Reconnection vs IMF Orientation', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reconnection_efield.png', dpi=150)
plt.show()
```

## 9. 요약

이 강의는 우주 기상의 MHD 물리를 다루었습니다:

1. **지구 자기권**: 태양풍에 의해 압축된 쌍극자 장, magnetopause, bow shock, magnetotail 형성
2. **Magnetopause**: 압력 균형이 standoff 거리 결정 ($r_{mp} \sim 10 R_E$)
3. **Bow shock**: 태양풍을 초음속에서 아음속으로 충격, 밀도를 약 4배 압축
4. **Dungey cycle**: 자기 재결합이 자기권 대류 및 substorms 구동
5. **자기 폭풍**: Ring current 형성이 Dst index 감소시킴, 남쪽 IMF에 의해 구동됨
6. **CMEs**: 태양 플라즈마/장의 분출적 방출, 1-5일에 지구로 전파
7. **GIC**: 급격한 B 변화로부터 전력망의 유도 전류, 정전 유발 가능
8. **예보**: 태양 관측, MHD 모델, L1 모니터의 조합

우주 기상 MHD는 태양에서 지구까지의 규모를 연결하며, 중요한 인프라를 보호하기 위해 전역 모델링 및 실시간 예보를 필요로 합니다.

## 연습 문제

1. **Magnetopause 압축**: CME 동안 태양풍 속도가 $v_{sw} = 900$ km/s로 증가하고 밀도가 $n_{sw} = 30$ cm$^{-3}$로 증가합니다. Magnetopause standoff 거리를 계산하세요. 이것이 geosynchronous orbit (6.6 $R_E$) 안쪽으로 압축됩니까?

2. **Bow shock**: $v_{sw} = 600$ km/s, $T_{sw} = 10^5$ K인 태양풍의 경우, sonic Mach number 및 bow shock를 가로지르는 밀도 압축비를 계산하세요.

3. **Dst 예측**: Burton 모델을 사용하여 6시간 동안 유지되는 태양풍 전기장 $E_{sw} = v_{sw} B_s = 5$ mV/m인 폭풍의 최소 Dst를 추정하세요. 초기 $Dst_0 = 0$, $a = 10^{-3}$ nT/(mV/m), $b = 0.5$ mV/m, $\tau = 8$ 시간을 가정하세요. 폭풍을 분류하세요 (중간: $< -50$ nT; 강한: $< -100$ nT).

4. **CME transit**: CME가 $v_0 = 1500$ km/s로 발사됩니다. $v_{sw} = 400$ km/s이고 $\gamma^{-1} = 1$ day인 drag 모델을 사용하여 지구 (1 AU = 1.5 × 10$^{11}$ m)까지의 도착 시간을 추정하세요. 시간과 일로 표현하세요.

5. **재결합 속도**: (a) 북쪽 IMF: $v_{sw} = 400$ km/s, $B_{sw} = 5$ nT, $\theta = 0°$; (b) 남쪽 IMF: 동일한 매개변수, $\theta = 180°$에 대한 재결합 전기장을 계산하세요. 재결합 효율을 비교하세요.

6. **GIC 위험**: 폭풍 동안 $dB/dt = 1000$ nT/min입니다. 200 km 송전선에서 유도된 전기장 및 GIC를 추정하세요 ($\rho_{earth} = 1000$ Ω·m, $R_{line} = 0.2$ Ω 가정). 이것이 전력망에 우려됩니까?

7. **Substorm 에너지**: substorm이 30분 동안 $10^{15}$ J의 에너지를 방출합니다. 이 에너지가 100 km 고도의 $10^{12}$ m$^2$ 면적에 걸쳐 전리층에 증착되면 에너지 플럭스 (W/m$^2$)를 추정하세요. 태양 상수 (1360 W/m$^2$)와 비교하세요.

8. **Ring current**: Dst index 감소는 ring current 에너지에 비례합니다: $Dst \sim -E_{ring} / (4 \times 10^{14} \text{ J/nT})$. $Dst = -150$ nT의 경우, ring current 에너지를 추정하세요. 줄로 표현하세요.

9. **CME magnetic cloud**: magnetic cloud가 $B \sim 30$ nT, $n \sim 10$ cm$^{-3}$, $T \sim 10^4$ K를 가집니다. 플라즈마 beta $\beta = 2 \mu_0 p / B^2$를 계산하세요. 이것이 플럭스 로프 구조 ($\beta < 1$)와 일치합니까?

10. **우주 기상 예보**: CME 도착 시간이 ~12시간 전에 예측될 수 있지만 폭풍 강도 (Dst 최소값)는 CME가 L1에 도달할 때까지 불확실한 이유를 설명하세요. 중요한 누락 정보는 무엇입니까?

---

**이전**: [Fusion MHD](./13_Fusion_MHD.md) | **다음**: [2D MHD Solver](./15_2D_MHD_Solver.md)
