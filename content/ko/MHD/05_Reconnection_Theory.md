# 5. 자기 재결합 이론

## 학습 목표

이 레슨을 마치면 다음을 할 수 있어야 합니다:

1. 자기 재결합이 플라즈마 물리학 및 천체물리학에서 중요한 이유 설명하기
2. Sweet-Parker 재결합률 유도하고 그 한계 이해하기
3. Petschek 모델을 분석하고 더 빠른 재결합을 달성하는 방법 설명하기
4. 무충돌 재결합에서 Hall 물리학의 역할 이해하기
5. X-점 기하학과 재결합 영역의 구조 설명하기
6. 무차원 재결합률 계산 및 해석하기
7. 재결합률 스케일링의 수치 모델 구현하기

## 1. 자기 재결합 소개

### 1.1 자기 재결합이란 무엇인가?

자기 재결합은 자기장 위상 구조가 변화하고 자기 에너지가 플라즈마 운동 및 열 에너지로 빠르게 변환되는 기본적인 플라즈마 과정입니다. 이 과정은 우주에서 가장 폭발적인 현상들을 일으킵니다.

```
핵심 개념: 이상 MHD에서 자기장 선은 플라즈마에 "동결"되어 있습니다
(자속 동결). 재결합은 이 제약을 깨뜨려 자기장 선이 끊어지고
다른 구성으로 재결합할 수 있게 합니다.
```

이 과정은 비이상 효과(저항, Hall 물리학 또는 운동학적 효과)로 인해 동결 조건이 무너지는 얇은 전류 시트에서 발생합니다. 변화하는 위상 구조는 자기 에너지가 종종 폭발적으로 방출되도록 합니다.

### 1.2 재결합이 중요한 이유

자기 재결합은 다음을 이해하는 데 중요합니다:

**태양 물리학:**
- 태양 플레어: 수 분 내에 ~10³²–10³³ erg의 에너지 방출
- 코로나 질량 방출(CME): 수십억 톤의 플라즈마 분출
- 코로나 가열: 100만도 코로나 유지

**우주 물리학:**
- 자기권 서브스톰: 오로라 밝아짐 이벤트
- 태양풍-자기권 결합: 자기권계면에서의 에너지 전달
- 입자 가속: 비열적 입자 생성

**실험실 플라즈마:**
- 토카막 톱니파 충돌: 급격한 중심 온도 하락
- 붕괴: 가두기의 재앙적 상실
- Spheromak 및 RFP 이완: 자기 자기조직화

**천체물리학:**
- 펄서 자기권: 고에너지 복사
- 강착원반 코로나: X선 방출
- 감마선 폭발 제트: 상대론적 유출

재결합 이론을 발전시킨 핵심 퍼즐: 단순 추정치가 매우 느린 속도를 예측할 때, 어떻게 재결합이 관측을 설명할 만큼 충분히 빠를 수 있는가?

### 1.3 동결 자기장 정리와 그 붕괴

이상 MHD에서 전기장은:

$$\mathbf{E} = -\mathbf{v} \times \mathbf{B}$$

이것은 동결 자속 정리로 이어집니다: 플라즈마와 함께 움직이는 임의의 닫힌 루프를 통과하는 자기 자속이 보존됩니다. 자기장 선은 플라즈마와 함께 움직인다고 생각할 수 있습니다.

저항이 포함되면 Ohm의 법칙은:

$$\mathbf{E} = -\mathbf{v} \times \mathbf{B} + \eta \mathbf{J}$$

여기서 $\eta$는 저항률이고 $\mathbf{J}$는 전류 밀도입니다. 저항 항은 자기장 선이 플라즈마를 통해 "미끄러질" 수 있게 합니다. 그러나 앞으로 보겠지만, 고전적 저항만으로는 관측된 재결합률을 설명하기에는 너무 작습니다.

유도 방정식은:

$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) + \eta \nabla^2 \mathbf{B}$$

이상 항과 저항 항의 상대적 중요성은 자기 Reynolds 수로 측정됩니다:

$$R_m = \frac{Lv_A}{\eta}$$

여기서 $L$은 특성 길이 척도이고 $v_A = B/\sqrt{\mu_0 \rho}$는 Alfvén 속도입니다. 대부분의 천체물리학 및 우주 플라즈마에서 $R_m \gg 1$이므로 재결합은 어려운 문제입니다.

## 2. Sweet-Parker 재결합

### 2.1 모델 설정

Sweet(1958)과 Parker(1957)가 독립적으로 개발한 Sweet-Parker 모델은 자기 재결합의 첫 정량적 이론이었습니다. 가정:

1. **정상 상태**: 시간 독립적 구성
2. **2차원 기하학**: 전류 방향을 따라 불변
3. **얇은 전류 시트**: 길이 $L \gg$ 폭 $\delta$
4. **대칭 유입**: 양쪽에서 플라즈마 유입
5. **고전적 저항**: $\eta$는 상수

```
                    Inflow v_in
                        ↓
                        ↓
        B_in ←  ←  ←  ←┼→  →  →  → B_in
                        ↓
        ================╋================ Current sheet
                        ↓                 (length L, width δ)
        B_in ←  ←  ←  ←┼→  →  →  → B_in
                        ↓
                        ↓ Outflow v_out
```

재결합하는 자기장은 반평행: 시트 위쪽에서 $\mathbf{B} = B_{in} \hat{x}$, 아래쪽에서 $\mathbf{B} = -B_{in} \hat{x}$입니다. 재결합 전기장 $E_z$는 균일하고 지면 밖을 향합니다.

### 2.2 재결합률 유도

확산 영역에 보존 법칙을 적용하여 재결합률을 유도합니다.

**질량 보존:**

질량 유입률은 질량 유출률과 같아야 합니다:

$$\rho v_{in} L \approx \rho v_{out} \delta$$

따라서:

$$v_{out} \approx v_{in} \frac{L}{\delta}$$

**운동량 보존:**

자기 압력이 유출을 구동합니다. 자기 압력과 동적 압력의 균형:

$$\frac{B_{in}^2}{2\mu_0} \approx \rho v_{out}^2$$

이것은 유출 속도가 대략 Alfvén 속도임을 줍니다:

$$v_{out} \approx v_A = \frac{B_{in}}{\sqrt{\mu_0 \rho}}$$

**X-점에서의 Ohm 법칙:**

확산 영역 중심에서 재결합 전기장은:

$$E_z = \eta J_z$$

여기서 Ampère 법칙으로부터 $J_z \approx B_{in}/(\mu_0 \delta)$입니다. 확산 영역 밖(이상 영역)에서 Ohm 법칙은:

$$E_z = v_{in} B_{in}$$

이들을 같다고 놓으면:

$$v_{in} B_{in} = \eta \frac{B_{in}}{\mu_0 \delta}$$

폭을 구하면:

$$\delta = \frac{\eta}{\mu_0 v_{in}}$$

**$v_{in}$ 구하기:**

$v_{out} = v_{in} L/\delta$와 $v_{out} = v_A$ 및 $\delta$에 대한 식을 결합:

$$v_A = v_{in} \frac{L}{\eta/(\mu_0 v_{in})} = v_{in} \frac{L \mu_0 v_{in}}{\eta}$$

$$v_A = \frac{L \mu_0 v_{in}^2}{\eta}$$

$$v_{in} = \sqrt{\frac{\eta v_A}{L \mu_0}} = v_A \sqrt{\frac{\eta}{L v_A \mu_0}}$$

**무차원 재결합률:**

Lundquist 수를 정의:

$$S = \frac{L v_A \mu_0}{\eta} = \frac{L v_A}{\eta/\mu_0}$$

이것은 시트 길이를 기반으로 한 자기 Reynolds 수입니다. 무차원 재결합률은:

$$M_A = \frac{v_{in}}{v_A} = S^{-1/2}$$

또한 종횡비는:

$$\frac{\delta}{L} = S^{-1}$$

### 2.3 Sweet-Parker 문제

태양 플레어의 경우 전형적인 매개변수:

- $L \sim 10^9$ m (10,000 km)
- $B \sim 0.01$ T (100 G)
- $n \sim 10^{16}$ m⁻³
- $T \sim 10^7$ K
- Spitzer 저항: $\eta \sim 10^{-4}$ Ω·m

이것은 다음을 제공합니다:

$$v_A \sim 10^6 \text{ m/s}$$

$$S \sim \frac{10^9 \times 10^6}{10^{-4}/\mu_0} \sim 10^{14}$$

$$M_A \sim 10^{-7}$$

$$v_{in} \sim 10^{-1} \text{ m/s}$$

**문제:** 이 속도로 자기장을 재결합하는 데 걸리는 시간:

$$t \sim \frac{L}{v_{in}} \sim 10^{10} \text{ s} \sim 300 \text{ years}$$

그러나 태양 플레어는 **수 분에서 수 시간** 안에 에너지를 방출합니다! 관측된 재결합률은 $M_A \sim 0.01$–$0.1$로, Sweet-Parker가 예측하는 것보다 약 100,000배 빠릅니다.

이것이 **재결합률 문제**입니다: Sweet-Parker 재결합은 관측을 설명하기에는 너무 느립니다.

### 2.4 Sweet-Parker의 한계

Sweet-Parker 모델에는 몇 가지 한계가 있습니다:

1. **너무 느림**: 위에서 보았듯이, $M_A \sim S^{-1/2}$는 큰 $S$에 대해 비현실적으로 느린 속도를 제공합니다.

2. **균일한 저항 가정**: 실제로 저항은 비정상 과정(난류, 파동)에 의해 증가될 수 있습니다.

3. **정상 상태**: 실제 재결합은 종종 시간 의존적이고 버스트성입니다.

4. **2D**: 3차원 효과가 중요할 수 있습니다.

5. **고전적 저항**: 무충돌 플라즈마는 운동학적 물리학이 필요합니다.

이러한 한계에도 불구하고 Sweet-Parker는 유용한 벤치마크로 남아 있으며 특정 영역에서 재결합의 일부 측면을 설명합니다.

## 3. Petschek 재결합

### 3.1 Petschek 모델

Petschek(1964)은 급진적인 수정을 제안했습니다: 긴 확산 영역 대신, 재결합은 X-점 근처의 작은 저항 영역에서 발생하며, 대부분의 에너지 변환은 확장된 느린 모드 MHD 충격파에서 일어납니다.

```
                    Inflow
                      ↓
         ╲            ↓            ╱
          ╲           ↓           ╱   Slow shock
           ╲          ↓          ╱
            ╲         ↓         ╱
             ╲        ↓        ╱
              ╲      ┏━┓      ╱
               ╲     ┃ ┃     ╱        Small diffusion
        ════════╲════┃X┃════╱═══════  region (size δ)
                 ╲   ┃ ┃   ╱
                  ╲  ┗━┛  ╱
                   ╲  ↓  ╱
                    ╲ ↓ ╱
                     ╲↓╱
                      ↓ Outflow
```

**핵심 특징:**

1. **작은 확산 영역**: 크기 ~$\delta \sim \eta/(v_A \mu_0)$, $L$과 독립적
2. **느린 MHD 충격파**: 확산 영역에서 거리 ~$L$까지 확장
3. **빠른 재결합**: 속도는 $S$에 로그적으로만 의존

### 3.2 느린 모드 MHD 충격파

느린 모드 충격파는 세 가지 유형의 MHD 충격파(빠름, 느림, 중간) 중 하나입니다. 특성:

- **속도 점프**: 충격파를 가로질러 유동이 가속됨
- **자기장**: $B_{\perp}$ 감소, $B_{\parallel}$는 증가하거나 감소할 수 있음
- **밀도**: 충격파를 가로질러 증가(압축)
- **온도**: 증가(엔트로피 생성)

느린 충격파는 자기 에너지를 열 및 운동 에너지로 변환합니다. 충격파는 전류 시트와 각도 $\psi$를 만듭니다:

$$\psi \sim \frac{\delta}{L} \sim \frac{\eta}{L v_A \mu_0}$$

### 3.3 Petschek 재결합률

Petschek의 분석은 재결합률을 제공합니다:

$$M_A \sim \frac{\pi}{8 \ln S}$$

$S = 10^{14}$의 경우:

$$M_A \sim \frac{\pi}{8 \ln(10^{14})} \sim \frac{3.14}{8 \times 32} \sim 0.012$$

이것은 관측된 속도와 놀랍도록 가깝습니다! $S$에 대한 로그 의존성은 큰 $S$에서 속도를 저항률과 거의 독립적으로 만듭니다.

**유도 스케치:**

재결합 전기장은 $E_z = v_{in} B_{in}$입니다. 확산 영역에서:

$$E_z = \eta J_z \sim \eta \frac{B_{in}}{\mu_0 \delta}$$

같다고 놓으면:

$$v_{in} B_{in} \sim \eta \frac{B_{in}}{\mu_0 \delta}$$

$$v_{in} \sim \frac{\eta}{\mu_0 \delta}$$

확산 영역 크기는 국소 물리학으로 설정됩니다:

$$\delta \sim \frac{\eta}{\mu_0 v_A}$$

따라서:

$$v_{in} \sim \frac{\eta}{\mu_0 \eta/(\mu_0 v_A)} = v_A$$

그러나 이것은 $M_A = 1$을 줄 것인데, 너무 빠릅니다(인과율 위반). 제약은 느린 충격파 각도를 확산 영역 크기에 맞추는 것에서 나옵니다:

$$\psi \sim \frac{1}{\ln S}$$

재결합률은:

$$M_A \sim \frac{1}{\ln S}$$

자세한 충격파 분석으로부터 수치 계수 $\pi/8$가 나옵니다.

### 3.4 Petschek 재결합의 안정성

Biskamp(1986)이 발견한 주요 문제: **균일한 저항에 대해 Petschek 재결합은 불안정합니다**.

수치 시뮬레이션은 균일한 $\eta$로 시스템이 Petschek 구성이 아닌 Sweet-Parker 구성으로 진화함을 보여주었습니다. 느린 충격파는 형성되지 않습니다.

**Petschek이 작동하는 경우:**

Petschek 재결합은 다음 경우에 발생할 수 있습니다:

1. **국소화된 저항**: X-점 근처에서 $\eta$ 증가(예: 비정상 저항에 의해)
2. **시간 의존적**: Sweet-Parker로 진화하기 전 과도적 빠른 재결합
3. **운동학적 효과**: 무충돌 재결합(Hall MHD, 이유체, 운동학적)

균일한 $\eta$를 갖는 저항 MHD에서 Sweet-Parker는 안정적인 정상 상태입니다. 그러나 자연은 균일한 저항을 거의 제공하지 않으며, 무충돌 효과가 많은 플라즈마에서 지배적입니다.

## 4. Hall MHD 및 무충돌 재결합

### 4.1 Hall 항

무충돌 플라즈마에서 이온과 전자는 이온 관성 길이(이온 표피 깊이)보다 작은 척도에서 독립적으로 움직일 수 있습니다:

$$d_i = \frac{c}{\omega_{pi}} = \frac{1}{\sqrt{\mu_0 n e^2 / m_i}} \approx 2.28 \times 10^7 \sqrt{\frac{10^6 \text{ cm}^{-3}}{n}} \text{ cm}$$

전형적인 태양 코로나 매개변수($n \sim 10^{10}$ cm⁻³)의 경우, $d_i \sim 70$ km로, 전체 척도 $L \sim 10^4$ km보다 훨씬 작습니다.

$d_i$보다 작은 척도에서 일반화된 Ohm 법칙의 Hall 항이 중요해집니다:

$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \eta \mathbf{J} + \frac{1}{ne} \mathbf{J} \times \mathbf{B} + \text{other terms}$$

Hall 항은 $\mathbf{J} \times \mathbf{B}/(ne)$입니다.

**물리적 해석:**

- 척도 $\gg d_i$: 전자와 이온이 함께 움직임(단일 유체 MHD)
- 척도 $\sim d_i$: 이온이 자기장으로부터 분리
- 척도 $< d_i$: 전자가 전류를 운반하고 동역학 제어

### 4.2 Hall MHD 방정식

Hall MHD는 다음으로 구성됩니다:

**연속:**

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$

**운동량:**

$$\rho \frac{d\mathbf{v}}{dt} = -\nabla p + \mathbf{J} \times \mathbf{B}$$

**유도(Hall 항 포함):**

$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B}) - \nabla \times \left( \frac{1}{ne} \mathbf{J} \times \mathbf{B} \right) + \eta \nabla^2 \mathbf{B}$$

**Ampère 법칙:**

$$\mathbf{J} = \frac{1}{\mu_0} \nabla \times \mathbf{B}$$

Hall 항은 다음과 같이 재작성될 수 있습니다:

$$\nabla \times \left( \frac{1}{ne} \mathbf{J} \times \mathbf{B} \right) = \nabla \times \left( \frac{1}{\mu_0 ne} (\nabla \times \mathbf{B}) \times \mathbf{B} \right)$$

이온 표피 깊이 $d_i = c/\omega_{pi}$를 정의하면, 이 항은 $v_A B/d_i$로 스케일링됩니다.

### 4.3 Hall 재결합의 구조

Hall 재결합은 **2-척도 구조**를 가집니다:

1. **외부 영역** ($r > d_i$): 이온 제어, 표준 MHD를 따름
   - 이온과 전자가 자기장에 동결
   - 척도: $L \sim 100 d_i$ 이상

2. **내부 확산 영역** ($r \sim d_i$): 전자 제어
   - 이온이 분리되고, 전자가 자기장에 동결
   - 면외(Hall) 자기장 생성
   - 전자가 전류를 운반

**사중극 Hall 자기장:**

Hall 항은 사중극 대칭을 갖는 면외(가이드 자기장 방향) 자기장을 생성합니다:

```
        B_z > 0  |  B_z < 0
                 |
      ━━━━━━━━━━X━━━━━━━━━━
                 |
        B_z < 0  |  B_z > 0
```

이 사중극 $B_z$ 구조는 Hall 재결합의 **결정적 증거**로, 다음에서 관측됩니다:
- 자기꼬리 재결합(Cluster, MMS 위성)
- 실험실 재결합 실험(MRX, VTF)
- 시뮬레이션(GEM Challenge)

### 4.4 Hall 재결합률

핵심 결과: **Hall 재결합은 큰 $S$에 대해 저항률과 독립적인 빠른 속도를 제공합니다**.

**스케일링 논증:**

이온 확산 영역에서 재결합 전기장은:

$$E_z \sim v_{in} B_{in}$$

전자 확산 영역(크기 $\delta_e \sim d_e$, 전자 표피 깊이)에서 전자 물리학이 지배적입니다. 재결합률은 전자 동역학으로 설정되어:

$$v_{in} \sim 0.1 v_A$$

속도 $M_A \sim 0.1$은 많은 Hall MHD 시뮬레이션에서 관측되며 $S$와 독립적입니다($S$가 $d_i < L$이 되도록 충분히 크다면).

**왜 빠른가?**

이온 분리는 훨씬 짧은 확산 영역($\delta \sim d_i$ 대신 $\delta \sim L/S^{1/2}$)을 허용합니다. 종횡비는:

$$\frac{\delta}{L} \sim \frac{d_i}{L}$$

$1/S$가 아니므로, 재결합률은 저항률과 독립적이 됩니다.

### 4.5 GEM 재결합 챌린지

GEM(Geospace Environmental Modeling) 자기 재결합 챌린지(Birn et al. 2001)는 재결합 시뮬레이션을 벤치마크하기 위한 커뮤니티 노력이었습니다.

**설정:**

- Harris 전류 시트 평형
- 2D 주기적 영역
- 다양한 코드: Hall MHD, 이유체, 하이브리드, 완전 PIC
- 재결합률, 구조, 시간 진화 비교

**핵심 결과:**

1. **빠른 재결합**: 모든 코드가 큰 $S$에 대해 $\eta$와 독립적인 $M_A \sim 0.1$을 발견
2. **사중극 Hall 자기장**: 모든 Hall/운동학적 모델에서 확인
3. **2-척도 구조**: 이온 및 전자 확산 영역이 명확히 분리
4. **전자 가열**: X-점 근처에서 온도 비등방성 발생
5. **플라스모이드 형성**: 매우 큰 $S$에서 2차 섬 형성(레슨 7 참조)

GEM 챌린지는 무충돌 재결합이 일반적으로 빠르다는 것을 확립하여 우주 플라즈마에 대한 재결합률 문제를 해결했습니다.

## 5. X-점 기하학 및 자기 위상 구조

### 5.1 X-점 구성

X-점은 $\mathbf{B} = 0$인 **자기 영점**입니다. 2D에서 X-점 근처에서 자기장은 Taylor 전개될 수 있습니다:

$$B_x \approx B_0 \frac{x}{L}$$

$$B_y \approx -B_0 \frac{y}{L}$$

여기서 $B_0$와 $L$은 특성 자기장 강도 및 길이 척도입니다.

X-점 근처의 **자기장 선**은 쌍곡선입니다:

$$xy = \text{const}$$

**분리선**은 X-점을 통과하는 자기장 선입니다(const = 0):
- $x = 0$: 수직 분리선
- $y = 0$: 수평 분리선

이들은 서로 다른 자기 위상 구조의 영역을 분리합니다.

### 5.2 자기 위상 구조 및 위상 변화

**자기 위상 구조**는 자기장 선의 연결성을 의미합니다. 2D에서 위상 구조는 다음으로 특징지어집니다:
- 어떤 자기장 선이 어떤 경계와 연결되는지
- X-점(영점) 및 O-점(극값)의 위치

**이상 MHD는 위상 구조를 보존합니다:**

이상 MHD에서 동결 정리는 자기장 선 연결성이 보존됨을 보장합니다. 두 플라즈마 요소가 처음에 같은 자기장 선에 있으면, 계속 같은 자기장 선에 남습니다.

**재결합은 위상 구조를 변화시킵니다:**

재결합은 자기장 선이 끊어지고 재결합하여 연결성을 변경할 수 있게 합니다. 이것이 재결합의 본질입니다.

```
재결합 전:                  재결합 후:

    A ════════════ B            A ════╗
                                      ║
         X (no flow)                  X (reconnection)
                                      ║
    C ════════════ D            C ════╝

    A는 B에 연결                A는 D에 연결
    C는 D에 연결                C는 B에 연결
```

위상 구조 변화 속도는 재결합 전기장 $E_z$로 측정됩니다.

### 5.3 분리선 및 자기 섬

**분리선**은 서로 다른 위상 구조의 영역을 나눕니다. 분리선을 가로지르는 플라즈마는 자기장 선 연결을 변경합니다.

**자기 섬(O-점):**

재결합이 완전하지 않을 때, 닫힌 자기장 선이 형성되어 자기 섬(플라스모이드)을 생성할 수 있습니다. O-점은 자속 함수 $\psi$의 국소 최대/최소입니다.

전류 시트에서 여러 X-점과 O-점이 형성될 수 있습니다:

```
    ────────O────X────O────X────O────
```

이 체인 구조는 난류 또는 고$S$ 재결합에서 중요합니다(플라스모이드 불안정성, 레슨 7).

### 5.4 벡터 포텐셜 및 자속 함수

2D에서 자기장은 자속 함수 $\psi$로 표현될 수 있습니다:

$$\mathbf{B} = \nabla \psi \times \hat{z}$$

또는 성분으로:

$$B_x = \frac{\partial \psi}{\partial y}, \quad B_y = -\frac{\partial \psi}{\partial x}$$

자기장 선은 $\psi$의 등고선입니다. X-점은 $\psi$의 안장점이고, O-점은 국소 극값입니다.

유도 방정식은 $\psi$에 대한 진화 방정식이 됩니다:

$$\frac{\partial \psi}{\partial t} = -E_z + \eta J_z$$

여기서 $J_z = -\nabla^2 \psi / \mu_0$입니다.

X-점에서 $\nabla \psi = 0$($\mathbf{B} = 0$이므로)이므로:

$$\left( \frac{\partial \psi}{\partial t} \right)_{X} = -E_z$$

재결합 전기장은 X-점에서 자속 변화율을 직접 측정합니다.

## 6. 재결합률 측정

### 6.1 무차원 재결합률

표준 측정은 **Alfvénic Mach 수**입니다:

$$M_A = \frac{v_{in}}{v_A}$$

여기서 $v_{in}$은 확산 영역으로의 유입 속도이고 $v_A = B_{in}/\sqrt{\mu_0 \rho}$는 상류 자기장을 기반으로 한 Alfvén 속도입니다.

**전형적인 값:**

- Sweet-Parker: $M_A \sim S^{-1/2} \sim 10^{-7}$ (태양 코로나의 경우)
- Petschek: $M_A \sim (\ln S)^{-1} \sim 0.01$
- Hall/무충돌: $M_A \sim 0.1$

### 6.2 재결합 전기장

동등한 측정은 재결합 전기장 $E_{rec}$입니다:

$$E_{rec} = v_{in} B_{in}$$

특성 전기장 $v_A B_{in}$으로 정규화:

$$\tilde{E} = \frac{E_{rec}}{v_A B_{in}} = M_A$$

정상 상태에서 $E_{rec}$는 재결합 영역 전체에서 균일합니다.

### 6.3 자속 전달률

단위 시간당 재결합되는 자기 자속의 속도(3차원의 단위 길이당):

$$\frac{d\Phi}{dt} = E_{rec} \cdot (\text{length in } z)$$

2D 시뮬레이션에서 이것은 재결합 단계를 진단하기 위해 시간의 함수로 종종 플롯됩니다.

### 6.4 관측적 측정

관측(예: 우주선 데이터)에서 재결합률은 다음으로부터 추론됩니다:

1. **유입 속도**: Doppler 이동 또는 입자 기기로 측정
2. **유출 속도**: 종종 $v_A$ 근처, Alfvénic 재결합 확인
3. **Hall 자기장**: 사중극 $B_z$ 신호(MMS 관측)
4. **에너지 입자**: 가속된 입자는 재결합을 나타냄

태양 플레어의 경우, 재결합률은 다음으로부터 추정됩니다:

$$M_A \sim \frac{v_{up}}{v_A}$$

여기서 $v_{up}$는 위쪽으로 움직이는 플레어 리본의 속도(재결합 발자국 추적)로, 일반적으로 $\sim 10$–100 km/s이며, $M_A \sim 0.01$–$0.1$을 제공합니다.

## 7. Python 예제

### 7.1 Sweet-Parker vs Petschek 스케일링

```python
import numpy as np
import matplotlib.pyplot as plt

# Lundquist 수 S = Lv_A/η는 넓은 범위에 걸쳐 있습니다: 실험실 플라즈마에서
# S~10⁶, 태양 코로나에서 S~10¹⁴. 이 넓은 범위를 고려해 logspace를 사용하면
# 모든 영역이 로그 플롯에서 동등하게 표현됩니다.
S = np.logspace(4, 16, 100)

# M_SP = S^(-1/2): Sweet-Parker 속도는 S가 커질수록 급격히 감소합니다.
# 확산 영역이 전류 시트의 전체 길이 L까지 확장되어야 하므로, 높은 S에서
# 재결합이 매우 느려지기 때문입니다. S~10¹⁴에서 M_A~10⁻⁷으로, 관측값보다
# 수 자릿수 낮습니다 — 이것이 재결합률 문제의 핵심입니다.
M_SP = S**(-0.5)

# M_P = π/(8 ln S): Petschek의 S에 대한 약한(로그) 의존성은 느린 충격파
# 기하학에서 비롯됩니다. 확산 영역의 크기가 L과 무관하게 δ~η/v_A로 고정되고,
# 느린 충격파가 대부분의 에너지 변환을 담당합니다.
# S=10¹⁴에서 ln(10¹⁴)≈32이므로 M_P≈0.012 — 관측된 플레어 속도에 가깝습니다.
M_P = np.pi / (8 * np.log(S))

# Hall 재결합률 ~0.1은 S가 d_i ≪ L을 만족할 만큼 충분히 크면 S와 무관합니다:
# 이온 분리가 저항률과 무관하게 ~d_i 크기의 확산 영역을 생성하므로,
# 재결합률은 저항 확산이 아닌 이온 동역학으로 결정됩니다 — 재결합률 문제 해결.
M_Hall = 0.1 * np.ones_like(S)

# Plot
plt.figure(figsize=(10, 6))
plt.loglog(S, M_SP, label='Sweet-Parker ($S^{-1/2}$)', linewidth=2)
plt.loglog(S, M_P, label=r'Petschek ($\pi/(8\ln S)$)', linewidth=2)
plt.loglog(S, M_Hall, label='Hall (collisionless)', linewidth=2, linestyle='--')

# Mark typical regimes
plt.axvline(1e8, color='gray', linestyle=':', alpha=0.5)
plt.text(1e8, 0.5, 'Laboratory', rotation=90, va='bottom', ha='right', alpha=0.5)
plt.axvline(1e14, color='gray', linestyle=':', alpha=0.5)
plt.text(1e14, 0.5, 'Solar corona', rotation=90, va='bottom', ha='right', alpha=0.5)

plt.xlabel('Lundquist number $S$', fontsize=14)
plt.ylabel('Reconnection rate $M_A = v_{in}/v_A$', fontsize=14)
plt.title('Reconnection Rate Scaling', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(1e4, 1e16)
plt.ylim(1e-8, 1)
plt.tight_layout()
plt.savefig('reconnection_rate_scaling.png', dpi=150)
plt.show()

# Print example values
S_examples = [1e6, 1e10, 1e14]
print("\nReconnection rates for different regimes:")
print(f"{'S':>10} {'Sweet-Parker':>15} {'Petschek':>15} {'Hall':>15}")
print("-" * 60)
for s in S_examples:
    sp = s**(-0.5)
    p = np.pi / (8 * np.log(s))
    h = 0.1
    print(f"{s:>10.1e} {sp:>15.2e} {p:>15.4f} {h:>15.2f}")
```

### 7.2 X-점 자기장 구조

```python
import numpy as np
import matplotlib.pyplot as plt

# Create grid
x = np.linspace(-2, 2, 40)
y = np.linspace(-2, 2, 40)
X, Y = np.meshgrid(x, y)

# Bx = X, By = -Y는 가장 단순한 X-점 자기장으로, 임의의 2D 영점을
# 1차까지 Taylor 전개하면 얻어집니다. 부호는 ∇·B = 0
# (∂Bx/∂x + ∂By/∂y = 1 - 1 = 0)을 보장하면서, X-점을 특징짓는
# 안장점 위상을 생성합니다: 자기장 선은 쌍곡선 xy = const입니다.
Bx = X
By = -Y

# B_mag은 원점(영점)에서 정확히 0이며 거리에 비례하여 선형 증가합니다;
# 컬러 맵으로 시각화하면 재결합 중 확산 영역(낮은-B 영역)이
# 어디에 형성되어야 하는지를 즉시 보여줍니다.
B_mag = np.sqrt(Bx**2 + By**2)

# Create figure with field lines and strength
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Field lines
ax = axes[0]
# Plot field lines using streamplot
ax.streamplot(X, Y, Bx, By, color=B_mag, cmap='viridis',
              linewidth=1.5, density=1.5, arrowsize=1.5)
ax.plot(0, 0, 'rx', markersize=15, markeredgewidth=3, label='X-point')
ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Separatrices')
ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.set_xlabel('$x/L$', fontsize=14)
ax.set_ylabel('$y/L$', fontsize=14)
ax.set_title('X-Point Magnetic Field Lines', fontsize=16)
ax.legend(fontsize=11)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Right panel: Field strength
ax = axes[1]
contour = ax.contourf(X, Y, B_mag, levels=20, cmap='plasma')
ax.contour(X, Y, B_mag, levels=10, colors='black', alpha=0.3, linewidths=0.5)
ax.plot(0, 0, 'wx', markersize=15, markeredgewidth=3)
ax.axhline(0, color='white', linestyle='--', alpha=0.7, linewidth=2)
ax.axvline(0, color='white', linestyle='--', alpha=0.7, linewidth=2)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('$|\\mathbf{B}|/B_0$', fontsize=14)
ax.set_xlabel('$x/L$', fontsize=14)
ax.set_ylabel('$y/L$', fontsize=14)
ax.set_title('Magnetic Field Strength', fontsize=16)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('xpoint_structure.png', dpi=150)
plt.show()

# Plot hyperbolic field lines explicitly
fig, ax = plt.subplots(figsize=(8, 8))
t = np.linspace(-2, 2, 200)

# Field lines are xy = const
constants = [-1.5, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 1.5]
for c in constants:
    if c > 0:
        x_pos = t[t > 0]
        y_pos = c / x_pos
        ax.plot(x_pos, y_pos, 'b-', linewidth=1.5)
        ax.plot(-x_pos, -y_pos, 'b-', linewidth=1.5)
    elif c < 0:
        x_neg = t[t > 0]
        y_neg = c / x_neg
        ax.plot(x_neg, y_neg, 'r-', linewidth=1.5)
        ax.plot(-x_neg, -y_neg, 'r-', linewidth=1.5)

# Separatrices
ax.axhline(0, color='green', linestyle='--', linewidth=2.5, label='Separatrices', alpha=0.7)
ax.axvline(0, color='green', linestyle='--', linewidth=2.5, alpha=0.7)
ax.plot(0, 0, 'ko', markersize=12, label='X-point (null)')

ax.set_xlabel('$x/L$', fontsize=14)
ax.set_ylabel('$y/L$', fontsize=14)
ax.set_title('Hyperbolic Field Lines Near X-Point', fontsize=16)
ax.legend(fontsize=12)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('xpoint_hyperbolic.png', dpi=150)
plt.show()
```

### 7.3 Sweet-Parker 확산 영역 종횡비

```python
import numpy as np
import matplotlib.pyplot as plt

# Lundquist number
S = np.logspace(2, 14, 100)

# δ/L = S⁻¹은 Sweet-Parker 종횡비입니다: 확산 영역 폭 δ는
# 저항 확산이 층 안으로의 자기장 이류와 균형을 이룰 만큼 얇아야 합니다.
# S=10¹⁴(태양 코로나)에서 δ/L ~ 10⁻¹⁴ — 물리적으로 비현실적으로
# 얇은 전류 시트로, 무충돌 모델의 필요성을 동기화합니다.
delta_over_L = S**(-1)

# L/δ = S는 역 종횡비입니다: 이 수가 클수록 전류 시트가 더 길게 늘어나고
# 재결합이 더 느려집니다. 자기장이 이완되기 전에 유출이 더 긴 거리를
# 이동해야 하기 때문입니다.
L_over_delta = S

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: aspect ratio
ax = axes[0]
ax.loglog(S, delta_over_L, linewidth=2, color='blue')
ax.axhline(0.1, color='red', linestyle='--', label='$\\delta/L = 0.1$', alpha=0.7)
ax.axhline(0.01, color='orange', linestyle='--', label='$\\delta/L = 0.01$', alpha=0.7)
ax.set_xlabel('Lundquist number $S$', fontsize=14)
ax.set_ylabel('Aspect ratio $\\delta/L$', fontsize=14)
ax.set_title('Sweet-Parker Diffusion Region Aspect Ratio', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Right: reconnection rate vs aspect ratio
ax = axes[1]
# M_A = S⁻¹/² = (δ/L)^(1/2): 재결합률은 종횡비의 제곱근과 같습니다.
# 더 넓은(덜 늘어난) 확산 영역일수록 더 빠르게 재결합한다는 것을 보여주며 —
# 이것이 Sweet-Parker 병목의 기하학적 원인입니다.
M_A = S**(-0.5)
ax.loglog(delta_over_L, M_A, linewidth=2, color='green')
ax.set_xlabel('Aspect ratio $\\delta/L$', fontsize=14)
ax.set_ylabel('Reconnection rate $M_A$', fontsize=14)
ax.set_title('$M_A$ vs $\\delta/L$ (Sweet-Parker)', fontsize=16)
ax.grid(True, alpha=0.3)
# Add scaling annotation
ax.text(1e-6, 1e-2, '$M_A = \\delta/L = S^{-1/2}$', fontsize=13,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('sweet_parker_aspect_ratio.png', dpi=150)
plt.show()

# Print examples
print("\nSweet-Parker diffusion region properties:")
print(f"{'S':>12} {'δ/L':>12} {'L/δ':>12} {'M_A':>12}")
print("-" * 50)
S_vals = [1e4, 1e6, 1e8, 1e10, 1e12, 1e14]
for s in S_vals:
    d_L = s**(-1)
    L_d = s
    M = s**(-0.5)
    print(f"{s:>12.0e} {d_L:>12.2e} {L_d:>12.2e} {M:>12.2e}")
```

### 7.4 Hall 재결합 사중극 자기장

```python
import numpy as np
import matplotlib.pyplot as plt

# Grid
x = np.linspace(-3, 3, 60)
y = np.linspace(-2, 2, 40)
X, Y = np.meshgrid(x, y)

# tanh(Y)는 Harris 전류 시트와 유사한 면내 자기장을 생성합니다:
# -B₀(시트 아래)에서 +B₀(시트 위)로 부드럽게 전환되며,
# GEM Challenge 시뮬레이션에서 사용되는 표준 전류 시트 평형과 일치합니다.
Bx = np.tanh(Y)
# exp(-Y²) 포락선은 By가 전류 시트에서 멀어질수록 감쇠하게 하여,
# X-점 구조가 무한대까지 확장되지 않고 국소화된 재결합 영역에 한정되도록 합니다.
By = -np.tanh(X / 2) * np.exp(-Y**2)

# 사중극 Hall 자기장 B_z ∝ X·Y는 Hall 재결합의 가장 중요한 관측 신호입니다:
# Ohm 법칙의 J×B Hall 항이 X-점의 각 사분면에서 반대 부호의 면외 전류를 생성하여
# Cluster 및 MMS 우주선 데이터로 확인된 특징적인 네 엽(four-lobe) 패턴을 만들기
# 때문에 발생합니다.
r2 = X**2 + Y**2
Bz = X * Y * np.exp(-r2 / 2)

# J_z = -tanh(Y)/cosh²(Y)는 Harris 전류 시트 전류 프로파일입니다:
# y=0(전류 층) 근처에 집중되어 있고 지수적으로 감쇠하며,
# 얇은 전류 시트의 자기 일관적 운동학적 평형을 반영합니다.
J_z = -np.tanh(Y) / np.cosh(Y)**2

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: In-plane field
ax = axes[0, 0]
ax.streamplot(X, Y, Bx, By, color='black', linewidth=1, density=1.5, arrowsize=1.2)
ax.plot(0, 0, 'rx', markersize=15, markeredgewidth=3)
ax.set_xlabel('$x/d_i$', fontsize=13)
ax.set_ylabel('$y/d_i$', fontsize=13)
ax.set_title('In-Plane Magnetic Field ($B_x$, $B_y$)', fontsize=14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Panel 2: Out-of-plane Hall field
ax = axes[0, 1]
levels = np.linspace(-0.5, 0.5, 21)
contour = ax.contourf(X, Y, Bz, levels=levels, cmap='RdBu_r', extend='both')
ax.contour(X, Y, Bz, levels=levels[::2], colors='black', alpha=0.3, linewidths=0.5)
ax.plot(0, 0, 'kx', markersize=15, markeredgewidth=3)
ax.axhline(0, color='black', linestyle='--', alpha=0.5)
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('$B_z/B_0$', fontsize=13)
ax.set_xlabel('$x/d_i$', fontsize=13)
ax.set_ylabel('$y/d_i$', fontsize=13)
ax.set_title('Out-of-Plane Hall Field ($B_z$)', fontsize=14)
ax.set_aspect('equal')

# Panel 3: Current density
ax = axes[1, 0]
contour = ax.contourf(X, Y, J_z, levels=20, cmap='coolwarm')
ax.contour(X, Y, J_z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('$J_z/J_0$', fontsize=13)
ax.set_xlabel('$x/d_i$', fontsize=13)
ax.set_ylabel('$y/d_i$', fontsize=13)
ax.set_title('Current Density ($J_z$)', fontsize=14)
ax.set_aspect('equal')

# Panel 4: Line plot of Bz along x-axis
ax = axes[1, 1]
y_cuts = [0.5, 1.0, 1.5]
for y_cut in y_cuts:
    idx = np.argmin(np.abs(y - y_cut))
    ax.plot(x, Bz[idx, :], linewidth=2, label=f'$y/d_i = {y_cut}$')
ax.axhline(0, color='black', linestyle='--', alpha=0.5)
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('$x/d_i$', fontsize=13)
ax.set_ylabel('$B_z/B_0$', fontsize=13)
ax.set_title('Hall Field Profile Along $x$ (Quadrupolar Structure)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hall_reconnection_field.png', dpi=150)
plt.show()

# Schematic of two-scale structure
fig, ax = plt.subplots(figsize=(10, 8))

# Outer ion diffusion region
theta = np.linspace(0, 2*np.pi, 100)
x_ion = 2 * np.cos(theta)
y_ion = 1.5 * np.sin(theta)
ax.fill(x_ion, y_ion, color='lightblue', alpha=0.5, label='Ion diffusion region ($\\sim d_i$)')
ax.plot(x_ion, y_ion, 'b-', linewidth=2)

# Inner electron diffusion region
x_elec = 0.5 * np.cos(theta)
y_elec = 0.3 * np.sin(theta)
ax.fill(x_elec, y_elec, color='salmon', alpha=0.5, label='Electron diffusion region ($\\sim d_e$)')
ax.plot(x_elec, y_elec, 'r-', linewidth=2)

# X-point
ax.plot(0, 0, 'kx', markersize=20, markeredgewidth=4)

# Separatrices
ax.plot([-3, 3], [0, 0], 'k--', linewidth=2, alpha=0.7)
ax.plot([0, 0], [-2.5, 2.5], 'k--', linewidth=2, alpha=0.7)

# Inflow/outflow arrows
ax.annotate('', xy=(0, -1.5), xytext=(0, -2.5),
            arrowprops=dict(arrowstyle='->', lw=3, color='green'))
ax.text(0.2, -2, 'Inflow', fontsize=13, color='green')

ax.annotate('', xy=(2.5, 0), xytext=(1.5, 0),
            arrowprops=dict(arrowstyle='->', lw=3, color='orange'))
ax.text(2, 0.2, 'Outflow', fontsize=13, color='orange')

ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$y$', fontsize=14)
ax.set_title('Two-Scale Structure of Hall Reconnection', fontsize=16)
ax.legend(fontsize=12, loc='upper right')
ax.set_xlim(-3, 3)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hall_two_scale_structure.png', dpi=150)
plt.show()
```

### 7.5 재결합률 진화

```python
import numpy as np
import matplotlib.pyplot as plt

# Time array
t = np.linspace(0, 100, 500)

# Sweet-Parker approach to steady state (slow)
M_SP = 0.001 * (1 - np.exp(-t / 30))

# Petschek burst (faster)
M_P = 0.02 * np.exp(-((t - 20) / 10)**2) * (t > 10)

# Hall reconnection (fast, sustained)
M_Hall = 0.1 * (1 - np.exp(-t / 5)) * (t > 15)

# Combined example: onset, burst, quasi-steady
M_combined = M_SP.copy()
M_combined += M_P
M_combined += M_Hall * 0.5

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Panel 1: Individual models
ax = axes[0]
ax.plot(t, M_SP, label='Sweet-Parker (slow)', linewidth=2)
ax.plot(t, M_P, label='Petschek burst', linewidth=2)
ax.plot(t, M_Hall, label='Hall (fast)', linewidth=2)
ax.set_xlabel('Time ($t v_A / L$)', fontsize=13)
ax.set_ylabel('Reconnection rate $M_A$', fontsize=13)
ax.set_title('Idealized Reconnection Rate Profiles', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.15)

# Panel 2: Combined realistic scenario
ax = axes[1]
ax.plot(t, M_combined, linewidth=2.5, color='darkblue')
ax.axhline(0.1, color='red', linestyle='--', alpha=0.6, label='Typical Hall rate (~0.1)')
ax.axhline(0.01, color='orange', linestyle='--', alpha=0.6, label='Typical Petschek rate (~0.01)')
ax.fill_between(t, 0, M_combined, alpha=0.3, color='skyblue')

# Annotate phases
ax.annotate('Onset', xy=(10, 0.005), fontsize=12,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax.annotate('Burst', xy=(25, 0.08), fontsize=12,
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
ax.annotate('Quasi-steady', xy=(60, 0.06), fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax.set_xlabel('Time ($t v_A / L$)', fontsize=13)
ax.set_ylabel('Reconnection rate $M_A$', fontsize=13)
ax.set_title('Realistic Time-Dependent Reconnection', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.12)

plt.tight_layout()
plt.savefig('reconnection_rate_evolution.png', dpi=150)
plt.show()
```

## 요약

자기 재결합은 자기 위상 구조를 변화시키고 자기 에너지를 플라즈마 에너지로 변환하는 기본 과정입니다. 다음을 다루었습니다:

1. **Sweet-Parker 모델**: 길고 얇은 전류 시트에서의 정상 상태 재결합. 속도 $M_A \sim S^{-1/2}$는 $S \sim 10^{14}$일 때 천체물리학 응용에는 너무 느립니다.

2. **Petschek 모델**: 느린 MHD 충격파를 동반한 재결합으로, 속도 $M_A \sim (\ln S)^{-1} \sim 0.01$을 제공합니다. 그러나 이것은 국소화된 저항을 필요로 하며 균일한 $\eta$에서는 불안정합니다.

3. **Hall MHD 재결합**: 무충돌 플라즈마에서 이온은 $\sim d_i$(이온 표피 깊이) 척도에서 분리되어 2-척도 구조로 이어집니다. 재결합률은 빠르며($M_A \sim 0.1$) 저항률과 독립적입니다. 사중극 Hall 자기장은 핵심 관측 신호입니다.

4. **X-점 기하학**: $\mathbf{B} = 0$인 자기 영점으로, 쌍곡선 자기장 선 구조를 가집니다. 분리선은 서로 다른 위상 구조의 영역을 나눕니다. 재결합은 자기장 선 연결성을 변경합니다.

5. **재결합률 측정**: 무차원 속도 $M_A = v_{in}/v_A$가 표준 측정입니다. 태양 플레어 및 자기권 서브스톰에서 관측된 속도는 $\sim 0.01$–$0.1$로, Petschek 및 Hall 재결합과 일치합니다.

재결합률 문제의 해결은 무충돌 효과(Hall 물리학, 운동학적 효과)가 대부분의 자연 플라즈마에서 지배적이며, 고전적 저항과 독립적인 빠른 재결합을 가능하게 한다는 인식에서 나왔습니다. GEM 챌린지는 운동학적 재결합이 일반적으로 $M_A \sim 0.1$을 제공함을 확인했습니다.

## 연습 문제

1. **Sweet-Parker 스케일링**:
   a) 질량 보존, 운동량 균형 및 Ohm 법칙으로부터 Sweet-Parker 재결합률을 유도하세요.
   b) 지구 자기꼬리($L = 10^7$ m, $B = 20$ nT, $n = 10^6$ m⁻³, $\eta = 10^{-2}$ Ω·m)에 대해 $S$와 $M_A$를 계산하세요.
   c) 재결합 시간 척도를 추정하세요. 이것이 관측된 서브스톰 시작 시간(~1시간)과 일치합니까?

2. **Petschek vs Sweet-Parker**:
   a) 어떤 Lundquist 수에서 Petschek 속도가 Sweet-Parker 속도를 초과합니까?
   b) $S = 10^2$에서 $10^{16}$까지 $S$의 함수로 비율 $M_P / M_{SP}$를 플롯하세요.
   c) Petschek이 더 빠른 이유를 물리적으로 설명하세요.

3. **Hall 재결합**:
   a) 태양 코로나 매개변수($n = 10^{16}$ m⁻³)에 대해 이온 표피 깊이 $d_i$를 계산하세요.
   b) 전체 척도가 $L = 10^9$ m이면 척도 분리 $L/d_i$는 얼마입니까?
   c) 2-척도 구조(이온 및 전자 확산 영역)를 스케치하세요.

4. **X-점 자기장**:
   a) 자기장 $\mathbf{B} = B_0 (x \hat{x} - y \hat{y})/L$에 대해 자기장 선($\psi$의 등고선)을 찾으세요.
   b) 위치의 함수로 자기장 강도 $|\mathbf{B}|$를 계산하세요.
   c) $|\mathbf{B}|$가 최대인 곳은? 최소인 곳은?

5. **재결합 전기장**:
   a) $v_{in} = 0.1 v_A$이고 $B_{in} = 0.01$ T이면 재결합 전기장 $E_{rec}$는 얼마입니까?
   b) Alfvén 속도가 $v_A = 10^6$ m/s이면 $E_{rec}$를 V/m로 계산하세요.
   c) $z$ 방향의 1000 km 길이에 걸쳐 초당 얼마나 많은 자기 자속이 재결합됩니까?

6. **사중극 Hall 자기장**:
   a) Hall 항 $\mathbf{J} \times \mathbf{B}/(ne)$가 면외 자기장을 생성하는 이유를 물리적으로 설명하세요.
   b) $xy$-평면의 X-점에 대한 사중극 $B_z$ 구조를 스케치하세요.
   c) 확산 영역을 가로지르는 우주선이 이 자기장을 어떻게 관측할 것입니까?

7. **시뮬레이션 분석**:
   a) 2D MHD 시뮬레이션에서 재결합 단계 동안 $v_{in} = 0.05 v_A$를 측정합니다. $M_A$는 얼마입니까?
   b) 시뮬레이션이 $\eta = 10^{-4}$(코드 단위), $L = 10$, $v_A = 1$을 가지면 $S$를 계산하세요.
   c) 이 재결합률은 Sweet-Parker, Petschek 또는 Hall 재결합과 일치합니까?

8. **에너지 변환**:
   a) 단위 부피당 자기 에너지 유입률은 $\sim v_{in} B^2/(2\mu_0)$입니다. $M_A = 0.1$이면 이것을 $v_A$와 $B$로 표현하세요.
   b) 자기 에너지 유입률을 운동 에너지 유출률 $\sim \rho v_{out}^3 / 2$와 비교하세요.
   c) "누락된" 에너지는 어디로 갑니까?

9. **GEM 챌린지**:
   a) GEM 재결합 챌린지 설정(Harris 시트, 섭동, 경계 조건)을 조사하세요.
   b) 서로 다른 코드의 재결합률에 관한 핵심 발견은 무엇이었습니까?
   c) 저항 MHD 결과가 Hall MHD 및 운동학적 결과와 어떻게 달랐습니까?

10. **관측적 신호**:
    a) 자기꼬리에서 자기 재결합의 세 가지 관측적 신호를 나열하세요.
    b) MMS(Magnetospheric Multiscale) 임무는 Hall 자기장을 어떻게 측정합니까?
    c) 우주선이 이온 확산 영역을 가로지르지만 전자 확산 영역은 가로지르지 않으면 무엇을 관측할 것으로 예상합니까?

## 네비게이션

이전: [전류 구동 불안정성](./04_Current_Driven_Instabilities.md) | 다음: [재결합 응용](./06_Reconnection_Applications.md)
