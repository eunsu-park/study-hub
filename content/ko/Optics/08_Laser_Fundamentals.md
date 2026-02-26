# 08. 레이저 기초

[← 이전: 07. 편광](07_Polarization.md) | [다음: 09. 광섬유 →](09_Fiber_Optics.md)

---

레이저(LASER, Light Amplification by Stimulated Emission of Radiation — 유도 방출에 의한 빛의 증폭)는 20세기에 등장한 광학 발명품 중 가장 변혁적인 것이라 해도 과언이 아니다. 슈퍼마켓의 바코드 스캐너에서 안과 수술까지, 통신에서 중력파 검출(LIGO)까지, 레이저는 어디에나 있다. 레이저를 특별하게 만드는 것은 단순히 빛을 만들어낸다는 사실이 아니라, *어떻게* 빛을 만들어내는지에 있다: 근본적으로 양자역학적 과정을 통해 탁월한 결맞음성(Coherence), 지향성(Directionality), 분광 순도(Spectral Purity)를 지닌 빛을 생성한다.

이 레슨은 레이저의 물리학을 기초부터 쌓아 올린다. 유도 방출에 대한 아인슈타인의 통찰에서 시작하여 레이저 작용에 필요한 조건, 되먹임(Feedback)을 제공하는 광학 공동(Optical Cavity), 주요 레이저 종류를 살펴본다. 이어서 레이저 빛이 공간과 광학계를 전파하는 방식을 설명하는 가우시안 빔(Gaussian Beam) 형식론을 다룬다.

**난이도**: ⭐⭐⭐

## 학습 목표

1. 세 가지 복사 과정(흡수, 자연 방출, 유도 방출)을 설명하고 아인슈타인 관계식을 유도한다
2. 밀도 반전(Population Inversion)을 정의하고 3준위 및 4준위 시스템에서 이를 달성하기 위한 조건을 분석한다
3. 광학 공동의 역할을 설명하고 임계 이득 조건을 유도한다
4. 종방향 모드(Longitudinal Mode)와 횡방향 모드(Transverse Mode)의 물리적 기원을 구분한다
5. 주요 레이저 종류(기체, 고체, 반도체, 광섬유)의 동작 원리, 특성, 응용을 비교한다
6. 레이저의 결맞음성(시간적·공간적)을 정량화하고 선폭(Linewidth) 및 빔 품질(Beam Quality)과 연결한다
7. 가우시안 빔 전파 공식과 ABCD 행렬을 적용하여 광학계를 통한 빔 변화를 예측한다

---

## 목차

1. [빛-물질 상호작용: 세 가지 과정](#1-빛-물질-상호작용-세-가지-과정)
2. [아인슈타인 계수와 관계식](#2-아인슈타인-계수와-관계식)
3. [밀도 반전](#3-밀도-반전)
4. [광학 되먹임: 레이저 공동](#4-광학-되먹임-레이저-공동)
5. [레이저 모드](#5-레이저-모드)
6. [주요 레이저 종류](#6-주요-레이저-종류)
7. [결맞음 특성](#7-결맞음-특성)
8. [가우시안 빔 전파](#8-가우시안-빔-전파)
9. [ABCD 행렬 형식론](#9-abcd-행렬-형식론)
10. [파이썬 예제](#10-파이썬-예제)
11. [요약](#11-요약)
12. [연습 문제](#12-연습-문제)
13. [참고 문헌](#13-참고-문헌)

---

## 1. 빛-물질 상호작용: 세 가지 과정

전자기 복사가 원자(또는 분자, 이온)와 상호작용할 때, 에너지 차이가 $\Delta E = E_2 - E_1 = h\nu$인 두 에너지 준위 $E_1$ (하위)과 $E_2$ (상위) 사이의 전이를 포함하는 세 가지 근본적인 과정이 일어날 수 있다.

### 1.1 흡수(Absorption)

에너지 $h\nu = E_2 - E_1$인 광자가 흡수되어 원자가 $E_1$에서 $E_2$로 전이된다. 광자는 소멸한다. 흡수 속도:

$$\frac{dN_1}{dt}\bigg|_{\text{abs}} = -B_{12}\,\rho(\nu)\,N_1$$

여기서 $N_1$은 준위 1의 밀도, $\rho(\nu)$는 복사장의 분광 에너지 밀도(Spectral Energy Density), $B_{12}$는 흡수에 대한 아인슈타인 B 계수이다.

### 1.2 자연 방출(Spontaneous Emission)

들뜬 상태 $E_2$의 원자가 자발적으로 $E_1$으로 전이하며 에너지 $h\nu$의 광자를 방출한다. 이 과정은 시간과 방향 모두 무작위이다 — 방출된 광자는 무작위 방향으로 무작위 위상을 가지고 방출된다. 속도:

$$\frac{dN_2}{dt}\bigg|_{\text{sp}} = -A_{21}\,N_2$$

여기서 $A_{21}$은 아인슈타인 A 계수이다. 자연 방출의 수명은 $\tau_{\text{sp}} = 1/A_{21}$이다.

### 1.3 유도 방출(Stimulated Emission)

이것이 아인슈타인의 핵심적인 통찰이다(1917). 에너지 $h\nu$의 입사 광자가 $E_2$ 상태의 원자를 $E_1$으로 *유도*하여 첫 번째 광자와 **동일한** 두 번째 광자를 생성한다 — 같은 주파수, 같은 방향, 같은 위상, 같은 편광. 속도:

$$\frac{dN_2}{dt}\bigg|_{\text{st}} = -B_{21}\,\rho(\nu)\,N_2$$

> **비유**: 유도 방출을 완벽하게 맞춰 노래하는 합창단처럼 생각해보자. 한 가수(입사 광자)가 음을 내면, 다른 가수(들뜬 원자)가 완전히 같은 음높이, 박자, 조성으로 합류한다. 결과는 두 개의 완벽하게 동기화된 목소리(두 개의 동일한 광자)이며 — 그 두 개가 더 많은 가수를 유도해 점점 커지고 완벽하게 결맞는 합창이 된다. 이것이 유도 방출에 의한 증폭이다.

### 1.4 유도 방출이 중요한 이유

유도된 광자는 유도를 일으킨 광자의 **클론(Clone)**이다. 즉:
- **같은 주파수** $\nu$ — 분광 순도
- **같은 방향** $\hat{k}$ — 지향성
- **같은 위상** $\phi$ — 결맞음성
- **같은 편광** $\hat{e}$ — 편광 순도

이 네 가지 성질을 동시에 갖는 광자를 생성하는 다른 광원은 없다.

---

## 2. 아인슈타인 계수와 관계식

### 2.1 열평형 논증

열평형에서 상향 전이 속도와 하향 전이 속도는 같다:

$$B_{12}\,\rho(\nu)\,N_1 = A_{21}\,N_2 + B_{21}\,\rho(\nu)\,N_2$$

밀도는 볼츠만 분포(Boltzmann Distribution)를 따른다:

$$\frac{N_2}{N_1} = \frac{g_2}{g_1}\exp\!\left(-\frac{h\nu}{k_BT}\right)$$

여기서 $g_1, g_2$는 두 준위의 축퇴도(Degeneracy)이다.

### 2.2 아인슈타인 관계식 유도

$\rho(\nu)$를 풀고 플랑크의 흑체 복사 공식과 비교하면:

$$\rho(\nu) = \frac{8\pi h\nu^3}{c^3}\frac{1}{e^{h\nu/(k_BT)}-1}$$

**아인슈타인 관계식(Einstein Relations)** 두 개가 도출된다:

$$\boxed{g_1 B_{12} = g_2 B_{21}}$$

$$\boxed{A_{21} = \frac{8\pi h\nu^3}{c^3}B_{21}}$$

첫 번째 관계식은 흡수와 유도 방출이 근본적으로 같은 과정임을 알려준다 (같은 축퇴도일 때 $B_{12} = B_{21}$). 두 번째 관계식은 자연 방출이 주파수에 따라 급격히 증가($\propto \nu^3$)함을 보여주며, 이것이 X선 레이저 제작이 극도로 어려운 이유이다.

### 2.3 물리적 해석

자연 방출과 유도 방출 속도의 비:

$$\frac{A_{21}}{B_{21}\rho(\nu)} = e^{h\nu/(k_BT)} - 1$$

실온($T \approx 300\,\text{K}$)과 가시광 주파수($\nu \sim 5 \times 10^{14}\,\text{Hz}$)에서:

$$\frac{h\nu}{k_BT} \approx \frac{(6.63\times10^{-34})(5\times10^{14})}{(1.38\times10^{-23})(300)} \approx 80$$

따라서 $A_{21}/(B_{21}\rho) \approx e^{80} \approx 10^{35}$. 가시광에 대한 열평형에서는 자연 방출이 압도적으로 우세하다. 이것이 바로 백열등이 비결맞음 빛을 생성하는 이유이며, 레이저 작용을 위해 시스템을 열평형에서 멀리 벗어나게 해야 하는 이유이다.

---

## 3. 밀도 반전

### 3.1 근본적인 요구 조건

유도 방출이 흡수를 압도하려면 상위 준위에 하위 준위보다 더 많은 원자가 있어야 한다 (축퇴도 고려):

$$\frac{N_2}{g_2} > \frac{N_1}{g_1}$$

이 조건 — **밀도 반전(Population Inversion)** — 은 열평형에서는 절대 성립하지 않는다 (볼츠만 인수 $e^{-h\nu/(k_BT)} < 1$이 항상 성립). 밀도 반전은 **펌프(Pump)**라 불리는 외부 에너지원에 의해 인위적으로 만들어져야 한다.

### 3.2 2준위 시스템이 레이저 발진이 불가능한 이유

2준위 시스템을 강하게 펌핑해도 달성 가능한 최대값은 $N_2 = N_1$ (같은 밀도, *포화(Saturation)* 또는 *표백(Bleaching)*이라 함)이다. 펌핑을 강하게 할수록 유도 방출도 증가하므로 $N_2 > N_1$이 되는 것은 불가능하다. 시스템은 투명에 가까워질 뿐 이득을 얻지 못한다.

**결론**: 순수한 2준위 시스템은 밀도 반전을 유지할 수 없다. 최소 3개의 준위가 필요하다.

### 3.3 3준위 시스템

3준위 레이저에서는 (예: 루비 레이저, 1960년 마이만(Maiman)이 최초로 시연):

```
준위 3: --------- 펌프 밴드 (수명 짧음)
           ↑ 펌프    ↓ 빠른 비복사 전이
준위 2: --------- 상위 레이저 준위 (준안정, 긴 τ)
           ↓ 레이저 전이 (느림)
준위 1: --------- 바닥 상태 (하위 레이저 준위)
```

- 원자가 준위 1에서 준위 3으로 펌핑됨
- 빠른 비복사 전이로 준위 2에 도달
- 준위 2는 준안정(Metastable) 상태 (긴 수명 $\tau_2$)로 밀도가 축적됨
- 레이저 전이는 2 → 1에서 발생

**어려움**: 하위 레이저 준위가 바닥 상태이다. 실온에서 대부분의 원자는 준위 1에 있으므로, 투명 상태에 도달하기 전에 절반 이상의 원자를 준위 2로 펌핑해야 한다. 3준위 레이저는 임계 펌프 출력이 크다.

### 3.4 4준위 시스템

대부분의 현대 레이저는 4준위 방식을 사용한다 (예: Nd:YAG):

```
준위 4: --------- 펌프 밴드 (수명 짧음)
           ↑ 펌프    ↓ 빠른 전이
준위 3: --------- 상위 레이저 준위 (준안정)
           ↓ 레이저 전이
준위 2: --------- 하위 레이저 준위 (수명 짧음)
           ↓ 빠른 전이
준위 1: --------- 바닥 상태
```

- 하위 레이저 준위(2)가 준위 1로의 비복사 전이에 의해 빠르게 비워짐
- 준위 3과 2 사이의 밀도 반전이 최소한의 펌핑으로도 달성됨
- 임계값이 3준위 시스템보다 훨씬 낮음

### 3.5 이득 계수

밀도 반전이 있으면 광빔이 지수함수적으로 증폭된다. 소신호 이득 계수(Small-Signal Gain Coefficient):

$$g(\nu) = \frac{c^2}{8\pi\nu^2\tau_{\text{sp}}}\left(\frac{N_2}{g_2} - \frac{N_1}{g_1}\right)g_2\,\mathcal{L}(\nu)$$

여기서 $\mathcal{L}(\nu)$는 정규화된 선형 함수(Lineshape Function)이다. 세기는 다음과 같이 증가한다:

$$I(z) = I_0\,e^{g(\nu)\,z}$$

이것이 **광학적 증폭(Optical Amplification)** — LASER의 "A"이다.

---

## 4. 광학 되먹임: 레이저 공동

### 4.1 증폭기에서 발진기로

이득 매질만으로는 광학 증폭기가 된다 (광섬유 통신의 EDFA와 같이). **발진기(Oscillator)** (레이저)를 만들려면 **양성 되먹임(Positive Feedback)** — 빛을 이득 매질로 반복해서 되돌리는 거울 — 이 필요하다.

가장 단순한 공동은 **패브리-페로 공진기(Fabry-Perot Resonator)**: 거리 $L$로 분리된 두 거울.

```
     거울 1 (R₁)          이득 매질          거울 2 (R₂)
     ┌──────┐         ┌─────────────────┐         ┌──────┐
  ←──│██████│←────────│  ← → ← → ← →  │────────→│██████│──→ 출력
     │██████│  R₁≈1   │      증폭       │         │██████│  R₂<1
     └──────┘         └─────────────────┘         └──────┘
         ←─────────────── L ──────────────────→
```

한쪽 거울(*출력 결합기, Output Coupler*)은 $R_2 < 1$로 빛의 일부가 레이저 빔으로 출력된다.

### 4.2 임계 조건

레이저가 발진하려면 왕복 이득이 왕복 손실과 같거나 커야 한다. 한 번의 왕복($2L$) 후:

$$R_1 R_2\,e^{2(g-\alpha)L} \geq 1$$

여기서 $\alpha$는 단위 길이당 내부 손실 계수(산란, 흡수)이다. 로그를 취하면 **임계 이득(Threshold Gain)**:

$$\boxed{g_{\text{th}} = \alpha + \frac{1}{2L}\ln\!\left(\frac{1}{R_1 R_2}\right)}$$

두 번째 항은 거울 손실을 나타낸다. $g > g_{\text{th}}$이면 광 세기가 지수함수적으로 증가하다가 이득 포화(Gain Saturation)가 $g$를 정상 상태에서 정확히 $g_{\text{th}}$로 되돌린다.

### 4.3 공동 안정성

모든 거울 구성이 안정한 공동을 이루는 것은 아니다. 안정성 파라미터 $g_i = 1 - L/R_i$ ($R_i$는 거울 $i$의 곡률 반지름으로 반사율과 혼동 주의)를 사용하면 **안정성 조건**:

$$\boxed{0 \leq g_1 g_2 \leq 1}$$

주요 안정 구성:
- **평행 평면(Plane-Parallel)** ($R_1 = R_2 = \infty$): $g_1 g_2 = 1$ (경계 안정, 정렬이 어려움)
- **공초점(Confocal)** ($R_1 = R_2 = L$): $g_1 g_2 = 0$ (회절 손실 최소화)
- **반구형(Hemispherical)** ($R_1 = \infty, R_2 = L$): $g_1 g_2 = 0$ (정렬이 쉬움)
- **동심(Concentric)** ($R_1 = R_2 = L/2$): $g_1 g_2 = 1$ (경계 안정, 초점이 좁음)

---

## 5. 레이저 모드

### 5.1 종방향 모드(Longitudinal Mode)

정재파(Standing Wave)는 공동 내에 맞아야 한다. 공명 조건:

$$\nu_q = q\frac{c}{2nL}, \quad q = 1, 2, 3, \ldots$$

인접한 종방향 모드 사이의 주파수 간격인 **자유 분광 범위(FSR, Free Spectral Range)**:

$$\Delta\nu_{\text{FSR}} = \frac{c}{2nL}$$

30 cm 공동, $n = 1$의 경우: $\Delta\nu_{\text{FSR}} = 500\,\text{MHz}$.

레이저는 이득 대역폭 내에 있고 임계값을 초과하는 모든 종방향 모드에서 발진한다. 단일 주파수 동작을 위해 공동 내 에탈론(Etalon)이나 링 공동(Ring Cavity) 등의 기법이 사용된다.

### 5.2 횡방향 모드 (TEM 모드)

레이저 빔의 횡방향 세기 프로파일은 **에르미트-가우시안(Hermite-Gaussian)** (직사각형 대칭) 또는 **라게르-가우시안(Laguerre-Gaussian)** (원통형 대칭) 모드로 표현되며, $\text{TEM}_{mn}$으로 나타낸다.

기본 모드 $\text{TEM}_{00}$는 가우시안 세기 프로파일을 갖는다:

$$I(r) = I_0 \exp\!\left(-\frac{2r^2}{w^2}\right)$$

여기서 $w$는 빔 반지름 (세기가 피크의 $1/e^2$로 감소하는 거리). 이 모드는 회절 손실이 가장 낮고 빔 품질이 가장 높다.

고차 모드 ($\text{TEM}_{10}$, $\text{TEM}_{01}$, $\text{TEM}_{11}$ 등)는 횡방향 프로파일에 마디(Node)가 있고 빔 직경이 크다. 일반적으로 바람직하지 않으며 공동 내 조리개(Aperture)로 억제된다.

### 5.3 $M^2$ 빔 품질 인수

실제 레이저 빔은 빔 품질 인수 $M^2$로 특성화된다:

$$M^2 = \frac{\theta_{\text{actual}} \cdot w_{0,\text{actual}}}{\theta_{\text{Gaussian}} \cdot w_{0,\text{Gaussian}}} \geq 1$$

- $M^2 = 1$: 이상적인 $\text{TEM}_{00}$ 가우시안 빔 (회절 한계)
- $M^2 > 1$: 빔이 회절 한계 빔보다 $M^2$배 빠르게 발산
- 전형적인 값: HeNe $\approx 1.0$, Nd:YAG $\approx 1.1\text{-}1.3$, 고출력 다이오드 바 $\approx 20\text{-}50$

---

## 6. 주요 레이저 종류

### 6.1 기체 레이저(Gas Laser)

**헬륨-네온 (He-Ne)**:
- 파장: 632.8 nm (적색), 543.5 nm (녹색), 1152 nm (적외선)
- 출력: 0.5-50 mW (일반적)
- 메커니즘: 전기 방전이 He 원자를 들뜨게 하고; 충돌(공명 에너지 준위)을 통해 Ne 원자로 에너지 전달
- 특성: 우수한 빔 품질 ($M^2 \approx 1$), 긴 결맞음 길이(~30 cm)
- 응용: 정렬, 간섭 계측, 바코드 스캐너, 교육용

**이산화탄소 (CO$_2$)**:
- 파장: 9.4, 10.6 $\mu$m (중간 적외선)
- 출력: 수십 kW까지 (산업용)
- 메커니즘: CO$_2$ 분자의 진동-회전 전이; N$_2$가 공명 들뜸에 사용
- 특성: 매우 높은 효율(~20%), 고출력
- 응용: 산업용 절단/용접, 수술, 라이다(LIDAR)

**엑시머 레이저(Excimer Laser)** (ArF 193 nm, KrF 248 nm, XeCl 308 nm):
- 자외선 출력, 펄스 동작
- 응용: 반도체 리소그래피(ArF 193 nm이 현대 칩 제조를 견인), 눈 수술(LASIK)

### 6.2 고체 레이저(Solid-State Laser)

**Nd:YAG (네오디뮴 도핑 이트륨 알루미늄 가넷)**:
- 파장: 1064 nm (기본), 2배 주파수 변환으로 532 nm (녹색)
- 4준위 시스템 (낮은 임계값)
- 플래시 램프 또는 다이오드 레이저로 펌핑
- 응용: 재료 가공, 의료 수술, 거리 측정, 과학 연구

**Ti:사파이어 (티타늄 도핑 사파이어)**:
- 파장: 650-1100 nm 가변 (모든 레이저 중 가장 넓은 이득 대역폭)
- 녹색 레이저 (보통 주파수 배가 Nd:YAG)로 펌핑
- 초단 펄스 생성 가능 (모드 잠금, mode-locking, 약 5 fs까지)
- 응용: 초고속 과학, 다광자 현미경, 분광학

### 6.3 반도체 레이저 (레이저 다이오드, Laser Diode)

- p-n 접합의 유도 방출 기반 (직접 밴드갭(Direct Bandgap) 반도체, GaAs, InGaAsP 등)
- 파장은 밴드갭으로 결정: $\lambda = hc/E_g$
- 전류 주입으로 밀도 반전 달성 (외부 펌프 레이저 불필요)
- 매우 소형 (칩 크기), 높은 효율(>50%), GHz 속도 직접 변조 가능
- 종류: 패브리-페로, DFB (분산 되먹임), VCSEL (수직 공동 표면 방출 레이저)
- 응용: 광섬유 통신, 광학 저장 장치 (CD/DVD/블루레이), 레이저 프린터, 레이저 포인터

### 6.4 광섬유 레이저(Fiber Laser)

- 이득 매질이 희토류 이온 (Yb, Er, Tm) 도핑 광섬유
- 매우 긴 이득 경로 (수 미터), 우수한 열 관리 (높은 표면적 대 부피비)
- 고유하게 단일 모드 출력, 높은 빔 품질
- 이터븀(Ytterbium) 광섬유 레이저: 수 kW 연속파(CW) 출력 가능
- 응용: 산업용 절단/용접 (CO$_2$ 레이저 대체), 통신, 방산

### 6.5 비교표

| 파라미터 | He-Ne | CO$_2$ | Nd:YAG | Ti:사파이어 | 다이오드 | 광섬유 |
|----------|-------|--------|--------|------------|--------|--------|
| 파장 | 632.8 nm | 10.6 $\mu$m | 1064 nm | 650-1100 nm | 0.4-2 $\mu$m | 1-2 $\mu$m |
| 출력 | mW | kW | W-kW | W | mW-W | W-kW |
| 효율 | <0.1% | ~20% | ~3% | <0.1% | >50% | >30% |
| 빔 품질 $M^2$ | ~1 | 1-5 | 1-20 | ~1 | 1-50 | ~1 |
| 파장 가변성 | 없음 | 제한적 | 없음 | 광범위 | 제한적 | 제한적 |
| 주요 용도 | 계측 | 절단 | 재료 | 초고속 | 통신 | 산업 |

---

## 7. 결맞음 특성

### 7.1 시간적 결맞음(Temporal Coherence)

**시간적 결맞음**은 빛이 얼마나 오랫동안 예측 가능한 위상 관계를 유지하는지를 나타낸다. **결맞음 시간(Coherence Time)** $\tau_c$와 **결맞음 길이(Coherence Length)** $L_c$로 측정된다:

$$L_c = c\,\tau_c = \frac{c}{\Delta\nu}$$

여기서 $\Delta\nu$는 분광 선폭(Spectral Linewidth)이다. 레이저는 매우 좁은 선폭을 가지므로 긴 결맞음 길이가 나타난다:

| 광원 | 선폭 $\Delta\nu$ | 결맞음 길이 $L_c$ |
|------|----------------|-----------------|
| 백색광 | $\sim 3 \times 10^{14}$ Hz | $\sim 1$ $\mu$m |
| LED | $\sim 10^{13}$ Hz | $\sim 30$ $\mu$m |
| He-Ne (다중 모드) | $\sim 1.5$ GHz | $\sim 20$ cm |
| He-Ne (단일 모드) | $\sim 1$ MHz | $\sim 300$ m |
| 안정화 레이저 | $\sim 1$ Hz | $\sim 3 \times 10^8$ m |

### 7.2 공간적 결맞음(Spatial Coherence)

**공간적 결맞음**은 같은 순간 빔의 서로 다른 횡방향 지점들 사이에서 위상이 얼마나 잘 상관되어 있는지를 나타낸다. $\text{TEM}_{00}$ 레이저 빔은 전체 빔 단면에 걸쳐 사실상 완벽한 공간적 결맞음을 갖는다.

공간적 결맞음 덕분에 레이저 빔이 회절 한계 크기의 초점으로 집속될 수 있고, 넓은 면적에 걸쳐 높은 대비의 간섭 무늬를 만들 수 있다 (홀로그래피에 필수적).

### 7.3 반 시터르트-체르니케 정리(Van Cittert-Zernike Theorem)

부분 결맞음 광원의 경우, 공간적 결맞음도는 **반 시터르트-체르니케 정리**를 통해 광원 형상과 연결된다: 결맞음도의 복소수 값은 광원 세기 분포의 정규화된 푸리에 변환과 같다. 이 정리는 비결맞음 광원(별과 같은)도 충분한 거리에서 간섭 줄무늬를 만들 수 있는 이유를 설명한다.

---

## 8. 가우시안 빔 전파

### 8.1 근축 파동 방정식

헬름홀츠 방정식에서 출발하여 진폭 포락선(Amplitude Envelope)이 천천히 변한다는 가정(근축 근사, Paraxial Approximation)을 취하면:

$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + 2ik\frac{\partial u}{\partial z} = 0$$

기본 해는 **가우시안 빔(Gaussian Beam)**:

$$u(r,z) = \frac{w_0}{w(z)}\exp\!\left(-\frac{r^2}{w^2(z)}\right)\exp\!\left(-i\left[kz + \frac{kr^2}{2R(z)} - \psi(z)\right]\right)$$

### 8.2 가우시안 빔의 주요 파라미터

**빔 반지름** $w(z)$:

$$w(z) = w_0\sqrt{1 + \left(\frac{z}{z_R}\right)^2}$$

**레일리 범위(Rayleigh Range)** $z_R$ (허리에서 빔 면적이 두 배가 되는 거리):

$$z_R = \frac{\pi w_0^2}{\lambda}$$

**곡률 반지름** $R(z)$:

$$R(z) = z\left[1 + \left(\frac{z_R}{z}\right)^2\right]$$

**구이 위상(Gouy Phase)** $\psi(z)$:

$$\psi(z) = \arctan\!\left(\frac{z}{z_R}\right)$$

**원거리 장 발산 반각(Far-Field Divergence Half-Angle)**:

$$\theta = \frac{\lambda}{\pi w_0}$$

> **비유**: 가우시안 빔은 빛으로 만든 모래시계와 같다. 가장 좁은 지점(허리 $w_0$)이 중간에 있고, 빔은 양쪽으로 퍼져 원거리 장에서 원뿔에 가까워진다. 레일리 범위 $z_R$은 모래시계의 목 부분과 같다 — 빔이 거의 평행을 유지하는 영역이다. 허리가 좁을수록 목이 짧아지고 개구각이 넓어지며, 그 반대도 마찬가지이다. 좁은 허리와 작은 발산각을 동시에 가질 수는 없다 — 이것은 불확정성 원리의 작용이다.

### 8.3 중요한 성질

1. **빔 허리-발산각 곱** (가우시안 빔에서 불변):

$$w_0 \cdot \theta = \frac{\lambda}{\pi}$$

이것은 완벽한 $\text{TEM}_{00}$ 빔에서만 달성되는 최솟값이다. 실제 빔은 $w_0 \theta = M^2 \lambda / \pi$이다.

2. **초점 심도(Depth of Focus)** (공초점 파라미터): $b = 2z_R = 2\pi w_0^2 / \lambda$

3. **허리에서의 최대 세기**: 총 출력 $P$에 대해 $I_0 = 2P/(\pi w_0^2)$

4. **빔의 평행 유지** (허리의 $\sqrt{2}$ 이내): 거리 $2z_R$ 범위에서 유지됨.

---

## 9. ABCD 행렬 형식론

### 9.1 광선 전달 행렬(Ray Transfer Matrix)

근축 근사에서 임의의 광학 소자는 $(y, \theta)$ (높이와 각도)로 기술되는 광선을 다음과 같이 변환한다:

$$\begin{pmatrix} y_{\text{out}} \\ \theta_{\text{out}} \end{pmatrix} = \begin{pmatrix} A & B \\ C & D \end{pmatrix}\begin{pmatrix} y_{\text{in}} \\ \theta_{\text{in}} \end{pmatrix}$$

주요 ABCD 행렬:

| 소자 | 행렬 |
|------|------|
| 자유 공간 (길이 $d$) | $\begin{pmatrix} 1 & d \\ 0 & 1 \end{pmatrix}$ |
| 얇은 렌즈 (초점 거리 $f$) | $\begin{pmatrix} 1 & 0 \\ -1/f & 1 \end{pmatrix}$ |
| 곡면 거울 (반지름 $R$) | $\begin{pmatrix} 1 & 0 \\ -2/R & 1 \end{pmatrix}$ |
| 평면 경계 ($n_1 \to n_2$) | $\begin{pmatrix} 1 & 0 \\ 0 & n_1/n_2 \end{pmatrix}$ |

여러 소자의 연속: 오른쪽부터 왼쪽으로 행렬을 곱한다: $M = M_N \cdots M_2 \cdot M_1$.

### 9.2 가우시안 빔 변환

**복소 빔 파라미터(Complex Beam Parameter)** $q(z)$는 곡률 반지름과 빔 반지름 모두를 부호화한다:

$$\frac{1}{q(z)} = \frac{1}{R(z)} - i\frac{\lambda}{\pi w^2(z)}$$

빔 허리에서 ($z = 0$): $q_0 = iz_R$.

ABCD 시스템을 통한 $q$의 변환:

$$\boxed{q_{\text{out}} = \frac{Aq_{\text{in}} + B}{Cq_{\text{in}} + D}}$$

이 단 하나의 공식이 임의의 근축 광학계를 통한 가우시안 빔을 추적한다. 레이저 광학에서 가장 유용한 공식 중 하나이다.

### 9.3 예제: 얇은 렌즈로 가우시안 빔 집속

허리 $w_0$ (렌즈 위치)의 가우시안 빔이 초점 거리 $f$인 얇은 렌즈를 통과한다. 새로운 허리 크기와 위치는?

입력: $q_{\text{in}} = iz_R = i\pi w_0^2/\lambda$

렌즈 행렬: $A = 1, B = 0, C = -1/f, D = 1$

$$q_{\text{out}} = \frac{iz_R}{-iz_R/f + 1} = \frac{iz_R f}{f - iz_R}$$

대수 연산 후 ($f \gg z_R$인 초점면에서의 새로운 허리 반지름):

$$w_0' \approx \frac{f\lambda}{\pi w_0}$$

입력 빔이 클수록($w_0$) 집속 점이 작아진다($w_0'$) — 이것이 레이저 절단이나 광학 포획과 같은 응용에서 레이저 빔을 집속 전에 확대하는 이유이다.

---

## 10. 파이썬 예제

### 10.1 가우시안 빔 전파

```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_beam(z, w0, wavelength):
    """
    Calculate Gaussian beam parameters at position z.

    We compute all key quantities from just the waist size and wavelength.
    The Rayleigh range z_R sets the natural scale: for |z| < z_R the beam
    is quasi-collimated; for |z| >> z_R the beam diverges linearly.
    """
    z_R = np.pi * w0**2 / wavelength  # Rayleigh range: boundary between near and far field
    w = w0 * np.sqrt(1 + (z / z_R)**2)  # Beam radius expands hyperbolically
    R = np.where(
        np.abs(z) > 1e-15,
        z * (1 + (z_R / z)**2),  # Radius of curvature (infinite at waist, minimum at z_R)
        np.inf
    )
    gouy = np.arctan(z / z_R)  # Gouy phase: the pi phase shift through focus
    return w, R, gouy, z_R

# Parameters: a typical He-Ne laser beam
wavelength = 632.8e-9  # 632.8 nm
w0 = 0.5e-3  # 0.5 mm beam waist

z = np.linspace(-0.5, 0.5, 1000)  # propagation axis (meters)
w, R, gouy, z_R = gaussian_beam(z, w0, wavelength)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Beam envelope — the "hourglass" shape
axes[0].fill_between(z * 100, w * 1000, -w * 1000, alpha=0.3, color='red')
axes[0].plot(z * 100, w * 1000, 'r-', linewidth=2)
axes[0].plot(z * 100, -w * 1000, 'r-', linewidth=2)
axes[0].axhline(y=w0 * 1000, color='gray', linestyle='--', alpha=0.5,
                label=f'w₀ = {w0*1e3:.1f} mm')
axes[0].axvline(x=z_R * 100, color='blue', linestyle='--', alpha=0.5,
                label=f'z_R = {z_R*100:.1f} cm')
axes[0].axvline(x=-z_R * 100, color='blue', linestyle='--', alpha=0.5)
axes[0].set_ylabel('w(z) [mm]')
axes[0].set_title(f'Gaussian Beam Propagation (λ = {wavelength*1e9:.1f} nm, w₀ = {w0*1e3:.1f} mm)')
axes[0].legend()

# Radius of curvature
R_display = np.clip(R, -1e3, 1e3)  # Clip to avoid plotting infinity
axes[1].plot(z * 100, R_display, 'b-', linewidth=2)
axes[1].set_ylabel('R(z) [m]')
axes[1].set_ylim(-5, 5)
axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Gouy phase
axes[2].plot(z * 100, np.degrees(gouy), 'g-', linewidth=2)
axes[2].set_ylabel('Gouy phase [deg]')
axes[2].set_xlabel('z [cm]')

plt.tight_layout()
plt.savefig('gaussian_beam_propagation.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Rayleigh range: z_R = {z_R*100:.2f} cm")
print(f"Divergence half-angle: θ = {np.degrees(wavelength/(np.pi*w0)):.4f}°")
print(f"Depth of focus: b = 2z_R = {2*z_R*100:.2f} cm")
```

### 10.2 공동 안정성 선도

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_stability_diagram():
    """
    Plot the stability region for a two-mirror resonator.

    The stability parameters g1, g2 are defined as g_i = 1 - L/R_i.
    Stable cavities satisfy 0 <= g1*g2 <= 1, which defines the
    unshaded region on the plot. Common cavity configurations sit
    at special points on this diagram.
    """
    g1 = np.linspace(-2.5, 2.5, 500)
    g2 = np.linspace(-2.5, 2.5, 500)
    G1, G2 = np.meshgrid(g1, g2)

    # Stability condition: 0 <= g1*g2 <= 1
    stable = (G1 * G2 >= 0) & (G1 * G2 <= 1)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Shade the UNSTABLE regions (so stable regions remain white)
    ax.contourf(G1, G2, ~stable, levels=[0.5, 1.5],
                colors=['lightcoral'], alpha=0.3)
    ax.contour(G1, G2, G1 * G2, levels=[0, 1], colors='black', linewidths=1.5)

    # Mark special configurations
    configs = {
        'Plane-parallel': (1, 1),
        'Confocal': (0, 0),
        'Concentric': (-1, -1),
        'Hemispherical': (0, 1),
    }
    for name, (x, y) in configs.items():
        ax.plot(x, y, 'ko', markersize=8)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(10, 10), fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('g₁ = 1 - L/R₁', fontsize=12)
    ax.set_ylabel('g₂ = 1 - L/R₂', fontsize=12)
    ax.set_title('Laser Cavity Stability Diagram', fontsize=14)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Add label in the stable region
    ax.text(0.5, 0.5, 'STABLE', fontsize=14, ha='center', va='center',
            fontweight='bold', color='green')
    ax.text(-1.5, 1.5, 'UNSTABLE', fontsize=14, ha='center', va='center',
            fontweight='bold', color='red', alpha=0.7)

    plt.tight_layout()
    plt.savefig('cavity_stability.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_stability_diagram()
```

### 10.3 ABCD 행렬 빔 전파

```python
import numpy as np

def abcd_propagate(q_in, M):
    """
    Propagate a Gaussian beam through an ABCD matrix.

    The q-parameter transformation q_out = (A*q + B)/(C*q + D) is the
    master equation of Gaussian beam optics. It elegantly unifies
    geometric ray tracing with diffraction in a single formula.
    """
    A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
    q_out = (A * q_in + B) / (C * q_in + D)
    return q_out

def q_to_params(q, wavelength):
    """Extract beam radius w and radius of curvature R from q-parameter."""
    inv_q = 1.0 / q
    R = 1.0 / np.real(inv_q) if np.abs(np.real(inv_q)) > 1e-15 else np.inf
    # The imaginary part of 1/q gives -lambda/(pi*w^2)
    w = np.sqrt(-wavelength / (np.pi * np.imag(inv_q)))
    return w, R

# --- Example: focus a collimated beam with a lens ---
wavelength = 1064e-9  # Nd:YAG laser
w0_input = 2e-3  # 2 mm input beam waist at the lens

z_R_input = np.pi * w0_input**2 / wavelength
q_input = 1j * z_R_input  # q at the beam waist (which is at the lens)

f = 0.1  # 100 mm focal length lens
M_lens = np.array([[1, 0], [-1/f, 1]])  # Thin lens matrix

q_after_lens = abcd_propagate(q_input, M_lens)

# Propagate through free space to find the focus
distances = np.linspace(0.09, 0.11, 1000)
beam_waists = []

for d in distances:
    M_space = np.array([[1, d], [0, 1]])
    q_final = abcd_propagate(q_after_lens, M_space)
    w, R = q_to_params(q_final, wavelength)
    beam_waists.append(w)

beam_waists = np.array(beam_waists)
min_idx = np.argmin(beam_waists)

print(f"Input beam waist: {w0_input*1e3:.2f} mm")
print(f"Focal length: {f*1e3:.1f} mm")
print(f"Focus position: {distances[min_idx]*1e3:.2f} mm from lens")
print(f"Focused waist: {beam_waists[min_idx]*1e6:.2f} µm")
print(f"Theoretical: {f*wavelength/(np.pi*w0_input)*1e6:.2f} µm")
```

---

## 11. 요약

| 개념 | 핵심 공식 / 아이디어 |
|------|---------------------|
| 유도 방출(Stimulated Emission) | 입사 광자가 동일한 클론을 생성; 레이저 작용의 기초 |
| 아인슈타인 A/B 관계식 | $A_{21} = (8\pi h\nu^3/c^3) B_{21}$ |
| 밀도 반전(Population Inversion) | $N_2/g_2 > N_1/g_1$; 3개 이상의 준위 필요 |
| 임계 이득(Threshold Gain) | $g_{\text{th}} = \alpha + \frac{1}{2L}\ln(1/R_1R_2)$ |
| 공동 안정성(Cavity Stability) | $0 \leq g_1g_2 \leq 1$ ($g_i = 1 - L/R_i$) |
| 종방향 모드 | $\nu_q = qc/(2nL)$; 자유 분광 범위 $= c/(2nL)$ |
| 횡방향 모드 | TEM$_{mn}$ (에르미트-가우시안); TEM$_{00}$이 가우시안 |
| 결맞음 길이 | $L_c = c/\Delta\nu$ |
| 가우시안 빔 반지름 | $w(z) = w_0\sqrt{1+(z/z_R)^2}$ |
| 레일리 범위 | $z_R = \pi w_0^2/\lambda$ |
| 발산각 | $\theta = \lambda/(\pi w_0)$ |
| ABCD 법칙 | $q_{\text{out}} = (Aq_{\text{in}}+B)/(Cq_{\text{in}}+D)$ |
| $M^2$ 인수 | $w_0\theta = M^2\lambda/\pi$ ($M^2=1$이 이상적 가우시안) |

---

## 12. 연습 문제

### 연습 문제 1: 아인슈타인 계수

$\lambda = 694.3\,\text{nm}$ (루비 레이저)에서의 레이저 전이가 자연 방출 수명 $\tau_{\text{sp}} = 3\,\text{ms}$를 갖는다.

(a) 아인슈타인 $A$ 계수를 계산하라.
(b) 아인슈타인 $B$ 계수를 계산하라.
(c) 흑체 복사에 대해 자연 방출과 유도 방출의 속도가 같아지는 온도는? 이것이 열평형 레이저 구현에 대해 무엇을 의미하는가?

### 연습 문제 2: 공동 설계

공동 길이 $L = 20\,\text{cm}$인 Nd:YAG 레이저를 설계하려 한다. 이득 매질의 소신호 이득 계수는 결정 길이 5 cm에 걸쳐 $g_0 = 0.5\,\text{cm}^{-1}$이고, 내부 손실은 $\alpha = 0.01\,\text{cm}^{-1}$이다.

(a) $R_1 = 1.0$ (100% 반사)일 때 발진을 위한 최소 $R_2$는?
(b) 자유 분광 범위(FSR)를 계산하라.
(c) 평면 거울 하나와 곡면 거울 하나를 사용하는 안정한 공동을 설계하라. 곡면 거울의 곡률 반지름은?

### 연습 문제 3: 가우시안 빔

He-Ne 레이저가 $\lambda = 632.8\,\text{nm}$에서 $w_0 = 0.4\,\text{mm}$의 TEM$_{00}$ 빔을 방출한다.

(a) 레일리 범위와 초점 심도를 계산하라.
(b) 허리에서 1 m와 10 m 지점에서의 빔 직경은?
(c) 빔이 허리 위치에 놓인 $f = 50\,\text{mm}$ 렌즈로 집속된다. 집속 점 크기를 계산하라.
(d) $z = 0$부터 $z = 5\,\text{m}$까지 빔 반지름을 그리는 파이썬 스크립트를 작성하라.

### 연습 문제 4: 레이저 비교

He-Ne, CO$_2$, Nd:YAG, Ti:사파이어, 반도체 다이오드 레이저를 다음 파라미터로 비교하는 표를 작성하라: 파장, 펌핑 메커니즘, 효율, 전형적 출력, 결맞음 길이, 주요 응용. 각 레이저에 대해 해당 파장에서 방출하는 물리적 이유를 설명하라.

### 연습 문제 5: ABCD 행렬 연쇄

가우시안 빔 ($w_0 = 1\,\text{mm}$, $\lambda = 1064\,\text{nm}$)이 다음 순서를 통과한다:
1. 30 cm 자유 공간 전파
2. $f = 20\,\text{cm}$ 얇은 렌즈
3. 20 cm 자유 공간 전파

파이썬으로 최종 빔 반지름과 곡률 반지름을 계산하라. 빔이 수렴, 발산, 또는 허리 위치에 있는가?

---

## 13. 참고 문헌

1. Saleh, B. E. A., & Teich, M. C. (2019). *Fundamentals of Photonics* (3rd ed.). Wiley. — Chapters 15-17.
2. Siegman, A. E. (1986). *Lasers*. University Science Books. — 레이저 물리학의 결정판 교재.
3. Svelto, O. (2010). *Principles of Lasers* (5th ed.). Springer.
4. Milonni, P. W., & Eberly, J. H. (2010). *Laser Physics*. Wiley.
5. Einstein, A. (1917). "Zur Quantentheorie der Strahlung." *Physikalische Zeitschrift*, 18, 121-128.

---

[← 이전: 07. 편광](07_Polarization.md) | [다음: 09. 광섬유 →](09_Fiber_Optics.md)
