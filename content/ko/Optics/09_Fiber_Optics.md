# 09. 광섬유

[← 이전: 08. 레이저 기초](08_Laser_Fundamentals.md) | [다음: 10. 푸리에 광학 →](10_Fourier_Optics.md)

---

전 세계 인터넷 — 동영상 스트리밍, 클라우드 컴퓨팅, 금융 거래 — 은 거의 전적으로 머리카락 굵기의 유리 섬유에 의존하며, 이 섬유는 빛 펄스를 대양과 대륙을 가로질러 전송한다. 단 하나의 광섬유(optical fiber)가 수천 킬로미터에 걸쳐 초당 테라비트를 전송할 수 있으며, 이는 수천 가닥의 구리 케이블이 필요한 성능이다. 광섬유는 현대 통신의 근간이며, 그 원리는 센서, 의료 기기, 레이저 전달 시스템까지 확장된다.

이 레슨은 광섬유의 물리를 기초 원리부터 전개한다. 전반사(total internal reflection)에 의해 빛이 어떻게 유도되는지 분석하고, 섬유 유형을 분류하며, 성능을 제한하는 손실 및 분산 메커니즘을 정량화하고, 현대 광섬유 네트워크를 가능하게 하는 핵심 기술 — 증폭기, 파장 다중화, 격자 — 을 살펴본다.

**난이도**: ⭐⭐⭐

## 학습 목표

1. 계단 굴절률 섬유(step-index fiber)의 유도 조건을 유도하고 수치 개구(numerical aperture)를 계산한다
2. 계단 굴절률 섬유와 경사 굴절률 섬유(graded-index fiber)를 구별하고, 경사 굴절률이 모드 분산(modal dispersion)을 줄이는 방법을 설명한다
3. V-수(V-number)를 계산하고 단일 모드 차단 조건을 결정한다
4. 주요 감쇠 메커니즘(흡수, 레일리 산란, 굽힘 손실)을 식별하고 정량화한다
5. 세 가지 분산 유형(모드 분산, 색분산, 편광 모드 분산)과 펄스 확산에 미치는 영향을 분석한다
6. 에르븀 첨가 광섬유 증폭기(EDFA, erbium-doped fiber amplifier)의 동작 원리와 장거리 통신에서의 역할을 설명한다
7. 파장 분할 다중화(WDM, wavelength division multiplexing)와 광섬유 브래그 격자(FBG, fiber Bragg grating)를 설명한다

---

## 목차

1. [광섬유에서의 빛 유도](#1-광섬유에서의-빛-유도)
2. [섬유 유형과 구조](#2-섬유-유형과-구조)
3. [수치 개구와 수용각](#3-수치-개구와-수용각)
4. [광섬유의 모드](#4-광섬유의-모드)
5. [감쇠](#5-감쇠)
6. [분산](#6-분산)
7. [광섬유 증폭기 (EDFA)](#7-광섬유-증폭기-edfa)
8. [파장 분할 다중화 (WDM)](#8-파장-분할-다중화-wdm)
9. [광섬유 브래그 격자 (FBG)](#9-광섬유-브래그-격자-fbg)
10. [Python 예제](#10-python-예제)
11. [요약](#11-요약)
12. [연습 문제](#12-연습-문제)
13. [참고문헌](#13-참고문헌)

---

## 1. 광섬유에서의 빛 유도

### 1.1 전반사(Total Internal Reflection)

광섬유는 **전반사(TIR, total internal reflection)** 를 통해 빛을 유도한다. 빛이 굴절률이 높은 매질 $n_1$ (코어)에서 굴절률이 낮은 매질 $n_2$ (클래딩)으로 진행할 때, 입사각이 임계각(critical angle)을 초과하면:

$$\theta_c = \arcsin\!\left(\frac{n_2}{n_1}\right)$$

빛은 전송 손실 없이 코어로 완전히 반사된다.

> **비유**: 물수제비를 연상해 보라. 돌을 얕은 각도(스치는 입사)로 던지면 수면에서 튀어 계속 나아간다. 가파르게 던지면 물속으로 빠진다. 광섬유의 코어-클래딩 경계면도 빛에 대해 동일하게 작동한다 — 얕은 각도의 빛은 반사되어 유도 상태를 유지하고, 가파른 각도의 빛은 클래딩으로 빠져나간다. 임계각은 반사와 투과의 경계선이다.

### 1.2 기본 섬유 구조

일반적인 광섬유는 세 층으로 구성된다:

```
            단면:
         ┌────────────────────┐
         │   코팅 (250 µm)    │  폴리머 보호층
         │  ┌──────────────┐  │
         │  │ 클래딩        │  │  순수 실리카 (n₂ ≈ 1.444)
         │  │  (125 µm)    │  │
         │  │  ┌────────┐  │  │
         │  │  │  코어  │  │  │  도핑된 실리카 (n₁ ≈ 1.450)
         │  │  │(8-62µm)│  │  │
         │  │  └────────┘  │  │
         │  └──────────────┘  │
         └────────────────────┘
```

굴절률 차이 $\Delta = (n_1 - n_2)/n_1$ 는 매우 작아 — 보통 0.3%에서 2% 수준이다. 이 작은 차이에도 불구하고, 수 킬로미터에 걸쳐 빛을 가두기에 충분하다.

### 1.3 역사적 배경

찰스 카오(Charles Kao)와 조지 호크햄(George Hockham)은 1966년에 손실이 20 dB/km 이하로 줄어든다면 유리 섬유가 정보를 전달할 수 있다고 예측했다. 당시 섬유 손실은 ~1000 dB/km에 달했다. 코닝 글라스웍스(Corning Glass Works)는 1970년에 제어된 도판트를 사용한 융합 실리카로 17 dB/km를 달성했다. 1980년대에는 손실이 1550 nm에서 이론적 최솟값인 ~0.2 dB/km에 도달했다. 카오는 이 연구로 2009년 노벨 물리학상을 수상했다.

---

## 2. 섬유 유형과 구조

### 2.1 계단 굴절률 섬유(Step-Index Fiber)

굴절률 프로파일은 단순한 계단 함수이다:

$$n(r) = \begin{cases} n_1 & r < a \quad (\text{코어}) \\ n_2 & r \geq a \quad (\text{클래딩}) \end{cases}$$

여기서 $a$는 코어 반경이다.

**장점**: 제조가 간단하고 비용이 낮다.
**단점**: 다중 모드 동작 시, 서로 다른 광선 경로가 다른 길이를 가져 **모드 분산(modal dispersion)** 이 발생한다 — 이는 대역폭을 제한하는 지배적 요인이다.

### 2.2 경사 굴절률 섬유(Graded-Index Fiber)

굴절률이 중심에서 바깥쪽으로 점차 감소한다:

$$n(r) = \begin{cases} n_1\sqrt{1 - 2\Delta(r/a)^\alpha} & r < a \\ n_1\sqrt{1 - 2\Delta} \approx n_2 & r \geq a \end{cases}$$

최적 프로파일 지수는 $\alpha \approx 2$ (포물선형)이다. 이 프로파일에서는, 더 긴 경로를 이동하는 광선이 낮은 굴절률 영역을 통과하므로 더 빠르게 이동하여, 축방향 광선과 거의 동시에 도달한다. 이로써 모드 분산이 극적으로 감소한다.

**결과**: 경사 굴절률 섬유는 계단 굴절률 섬유에 비해 모드 분산을 $\sim \Delta / 2$ 배 줄인다 — 나노초/km에서 피코초/km 수준으로.

### 2.3 단일 모드 vs. 다중 모드

| 파라미터 | 단일 모드 (SMF) | 다중 모드 (MMF) |
|----------|-----------------|-----------------|
| 코어 지름 | 8-10 µm | 50 또는 62.5 µm |
| 클래딩 지름 | 125 µm | 125 µm |
| 굴절률 프로파일 | 계단형 | 경사형 (일반적) |
| 모드 수 | 1 (LP$_{01}$) | 수백~수천 |
| 파장 | 1310 nm, 1550 nm | 850 nm, 1300 nm |
| 대역폭 | 매우 높음 (THz·km) | 중간 (MHz·km ~ GHz·km) |
| 결합 | 어려움 (정밀 정렬 필요) | 쉬움 (큰 코어) |
| 전송 거리 | 최대 ~100 km (비증폭) | 최대 ~2 km |
| 미터당 비용 | 낮음 | 낮음 |
| 커넥터 비용 | 높음 | 낮음 |
| 응용 | 통신, 장거리 전송 | LAN, 데이터 센터 |

---

## 3. 수치 개구와 수용각

### 3.1 유도

섬유에 축으로부터 $\theta_a$ 각도로 입사하는 광선을 생각하자. 입사면에서의 스넬 법칙(Snell's law):

$$n_0 \sin\theta_a = n_1 \sin\theta_r$$

여기서 $n_0$는 주변 매질(보통 공기, $n_0 = 1$)의 굴절률이고, $\theta_r$는 굴절각이다. 코어-클래딩 경계면에서 전반사가 일어나려면, 광선이 $\theta_c$ 이상의 각도로 경계면에 입사해야 한다.

기하학적 관계를 정리하면, 최대 수용각이 **수치 개구(NA)**를 결정한다:

$$\boxed{\text{NA} = n_0\sin\theta_a = \sqrt{n_1^2 - n_2^2} \approx n_1\sqrt{2\Delta}}$$

### 3.2 대표적인 값

표준 단일 모드 섬유의 경우: $n_1 = 1.450$, $n_2 = 1.447$이면:

$$\text{NA} = \sqrt{1.450^2 - 1.447^2} = \sqrt{0.008691} \approx 0.093$$

이는 수용 반각 $\theta_a = \arcsin(0.093) \approx 5.3°$에 해당하며 매우 좁은 원추형이다. 다중 모드 섬유의 경우 NA는 보통 0.2~0.3(수용각 약 12~17°)이다.

### 3.3 물리적 의미

NA는 다음을 결정한다:
- 섬유에 빛을 결합하는 용이성 (NA가 클수록 결합이 쉬움)
- 섬유가 지원하는 모드 수 (NA가 클수록 모드가 많음)
- 굽힘에 대한 섬유의 민감도 (NA가 클수록 굽힘 손실이 적음)

---

## 4. 광섬유의 모드

### 4.1 V-수(Normalized Frequency, 정규화 주파수)

**V-수**는 광섬유를 특성화하는 가장 중요한 단일 파라미터이다:

$$\boxed{V = \frac{2\pi a}{\lambda}\,\text{NA} = \frac{2\pi a}{\lambda}\sqrt{n_1^2 - n_2^2}}$$

### 4.2 단일 모드 조건

계단 굴절률 섬유에서, 다음 조건일 때만 기본 모드(LP$_{01}$)만 전파된다:

$$\boxed{V < 2.405}$$

2.405는 베셀 함수(Bessel function) $J_0$의 첫 번째 영점이다. 정확히 $V = 2.405$ (**차단** 파장 $\lambda_c$)에서, 두 번째 모드(LP$_{11}$)가 막 전파되기 시작한다.

$$\lambda_c = \frac{2\pi a\,\text{NA}}{2.405}$$

### 4.3 모드 수

V가 큰 계단 굴절률 다중 모드 섬유의 경우:

$$M \approx \frac{V^2}{2}$$

포물선형 프로파일의 경사 굴절률 섬유의 경우:

$$M \approx \frac{V^2}{4}$$

**예시**: $a = 25\,\mu\text{m}$, NA = 0.2, $\lambda = 850\,\text{nm}$인 다중 모드 계단 굴절률 섬유:

$$V = \frac{2\pi \times 25\times10^{-6}}{850\times10^{-9}} \times 0.2 = 36.9$$

$$M \approx \frac{36.9^2}{2} \approx 681 \text{ 모드}$$

### 4.4 모드 필드 직경(Mode Field Diameter)

단일 모드 섬유에서 기본 모드는 거의 가우시안 강도 프로파일을 가진다. **모드 필드 직경(MFD)**은 필드 진폭이 최댓값의 $1/e$로 감소하는 직경으로 정의된다. 마르쿠제(Marcuse) 근사식은 다음과 같다:

$$\frac{w}{a} \approx 0.65 + \frac{1.619}{V^{3/2}} + \frac{2.879}{V^6}$$

MFD는 코어 직경보다 크며 (필드가 클래딩까지 확장됨), 특히 차단 파장 근처에서 두드러진다.

---

## 5. 감쇠

### 5.1 광섬유의 비어-람베르트 법칙(Beer-Lambert Law)

거리 $L$에서의 광력(power)은:

$$P(L) = P_0 \cdot 10^{-\alpha L / 10}$$

여기서 $\alpha$는 **dB/km** 단위의 감쇠 계수이다. 동등하게:

$$\alpha_{\text{dB/km}} = \frac{10}{L}\log_{10}\!\left(\frac{P_0}{P(L)}\right)$$

### 5.2 감쇠 메커니즘

**레일리 산란(Rayleigh Scattering)**: 유리 내 미시적 밀도 요동에 의한 내인성 산란. $\lambda^{-4}$에 비례하므로, 짧은 파장에서 지배적인 손실 메커니즘이다. 1550 nm에서: ~0.15 dB/km.

$$\alpha_R = \frac{C_R}{\lambda^4}$$

여기서 실리카의 경우 $C_R \approx 0.7\text{-}0.9\;\text{dB}\cdot\mu\text{m}^4/\text{km}$.

**물질 흡수(Material Absorption)**:
- **자외선 흡수**: SiO$_2$의 전자 전이(얼바흐 꼬리, Urbach tail). 파장이 증가할수록 지수적으로 감소.
- **적외선 흡수**: 분자 진동(Si-O 결합). 파장 ~1600 nm 이상에서 지수적으로 증가.
- **OH 흡수**: 수분 불순물($\text{OH}^-$ 이온)이 1383 nm (및 1240 nm, 950 nm 배음)에 강한 피크를 생성. 현대의 "저수분 피크" 섬유(ITU-T G.652.D)는 이 피크를 거의 제거했다.

**굽힘 손실(Bending Loss)**:
- **거시적 굽힘(Macrobending)**: 섬유가 임계 반경 이하로 굽혀지면, 클래딩의 에반에센트 필드(evanescent field)가 $c/n_2$보다 빠르게 이동해야 하는데, 이는 불가능하므로 빛이 방사된다. 임계 굽힘 반경은 차단 파장 근처에서 급격히 증가한다.
- **미세 굽힘(Microbending)**: 소규모의 무작위 변형(제조 또는 케이블 응력에서 발생)이 유도 모드를 방사 모드로 결합시킨다. 적절한 케이블링으로 최소화할 수 있다.

### 5.3 감쇠 스펙트럼

```
감쇠 (dB/km)
    5 │
      │╲
    2 │ ╲    레일리                              IR
      │  ╲   산란                              흡수
    1 │   ╲       ∧                               ╱
      │    ╲     ╱ ╲ OH 피크                    ╱
  0.5 │     ╲   ╱   ╲ (1383 nm)              ╱
      │      ╲ ╱     ╲                      ╱
  0.2 │───────╳───────╲────╲──────────────────╱───
      │              ╲  ╲────────╲──────╱
 0.15 │               최솟값: ~0.17 dB/km @ 1550 nm
      └──────────────────────────────────────────
      800   1000  1200  1400  1550  1600   1800
                    파장 (nm)

      |--O 대역--|--E--|S 대역|C 대역|L 대역|
```

세 가지 주요 통신 창(telecom window):
- **O-대역** (1260-1360 nm): 표준 SMF의 영분산(zero dispersion) 파장
- **C-대역** (1530-1565 nm): 최소 손실, EDFA 이득 창
- **L-대역** (1565-1625 nm): 확장 EDFA 창

---

## 6. 분산

분산(dispersion)은 광 펄스가 전파될수록 퍼지게 만들어, 광섬유 링크의 비트 전송률과 전송 거리를 제한한다. 세 가지 유형이 있다.

### 6.1 모드 분산(Modal Dispersion)

다중 모드 섬유에서 서로 다른 모드는 서로 다른 경로 길이를 이동한다. 계단 굴절률 섬유의 경우:

$$\Delta\tau_{\text{modal}} = \frac{n_1 L}{c}\Delta \approx \frac{n_1^2 L}{n_2 c}\Delta$$

**예시**: $n_1 = 1.48$, $\Delta = 0.01$, $L = 1\,\text{km}$:

$$\Delta\tau = \frac{1.48 \times 1000}{3\times10^8} \times 0.01 \approx 49\,\text{ns/km}$$

이는 대역폭을 대략 $B \approx 1/(2\Delta\tau) \approx 10\,\text{MHz}\cdot\text{km}$으로 제한한다.

경사 굴절률 (포물선형) 섬유: $\Delta\tau \approx \frac{n_1 L}{2c}\Delta^2$ — $\Delta/2$ 배 감소.

**단일 모드 섬유는 모드 분산을 완전히 제거한다** — 모드가 하나뿐이기 때문이다.

### 6.2 색분산(Chromatic Dispersion)

단일 모드 섬유에서도 펄스의 서로 다른 파장 성분은 다른 군속도(group velocity)로 이동한다. 색분산에는 두 가지 기여가 있다:

**물질 분산(Material dispersion)**: 실리카의 굴절률이 파장에 따라 변한다 ($dn/d\lambda \neq 0$).

**도파로 분산(Waveguide dispersion)**: 모드의 유효 굴절률은 필드가 코어와 클래딩에 얼마나 분포하는지에 따라 달라지며, 이는 파장에 따라 변한다.

전체 색분산은 **분산 파라미터(dispersion parameter)**로 특성화된다:

$$D = -\frac{\lambda}{c}\frac{d^2n_{\text{eff}}}{d\lambda^2} \quad [\text{ps/(nm·km)}]$$

표준 SMF-28 섬유의 경우: $D = 0$ at $\lambda_0 \approx 1310\,\text{nm}$, $D \approx +17\,\text{ps/(nm·km)}$ at 1550 nm.

색분산에 의한 **펄스 확산**:

$$\Delta\tau_{\text{chrom}} = |D| \cdot \Delta\lambda \cdot L$$

여기서 $\Delta\lambda$는 광원의 스펙트럼 폭이다.

**예시**: 1550 nm에서 $\Delta\lambda = 0.1\,\text{nm}$인 레이저 다이오드, $L = 100\,\text{km}$:

$$\Delta\tau = 17 \times 0.1 \times 100 = 170\,\text{ps}$$

이는 비트 전송률을 대략 $B < 1/(4\Delta\tau) \approx 1.5\,\text{Gb/s}$로 제한한다.

### 6.3 분산 관리

**분산 이동 섬유(DSF, Dispersion-shifted fiber)**: 도파로 분산을 조절하여 $\lambda_0$를 1550 nm으로 이동시킨다 ($D = 0$ at 1550 nm). 문제: WDM 시스템에서 $D = 0$은 비선형 효과(사파장 혼합, four-wave mixing)를 강화한다.

**비영 분산 이동 섬유(NZ-DSF, Non-zero dispersion-shifted fiber)**: 1550 nm에서 작지만 영이 아닌 $D$ (예: $D \approx 4\,\text{ps/(nm·km)}$). 분산을 줄이면서 비선형 페널티를 회피한다.

**분산 보상 섬유(DCF, Dispersion-compensating fiber)**: 큰 음의 $D$ (예: $-80\,\text{ps/(nm·km)}$)를 가진 섬유로, 표준 섬유에 누적된 분산을 상쇄하는 데 사용된다. 짧은 DCF로 긴 SMF 구간의 분산을 보상할 수 있다.

### 6.4 편광 모드 분산(PMD, Polarization Mode Dispersion)

이상적인 단일 모드 섬유에서는 두 직교 편광 모드(축퇴, degenerate)가 같은 속도로 이동한다. 실제 섬유에서는 결함(코어 타원성, 응력)이 이 축퇴를 깨뜨려 차동 군 지연(differential group delay)이 발생한다:

$$\Delta\tau_{\text{PMD}} = D_{\text{PMD}} \sqrt{L}$$

여기서 $D_{\text{PMD}}$는 PMD 계수로, 현대 섬유에서 보통 0.01~0.1 $\text{ps}/\sqrt{\text{km}}$이다. $\sqrt{L}$ 의존성에 주목하라 — PMD는 확률적 과정이다(모드 결합이 섬유를 따라 변한다).

40 Gb/s 이상에서 PMD는 적응형 보상이 필요한 중요한 손상 요인이 된다.

---

## 7. 광섬유 증폭기 (EDFA)

### 7.1 증폭의 필요성

광증폭기가 없던 시절, 장거리 광섬유 링크는 40~80 km마다 전자식 재생기(regenerator)가 필요했다: 광신호를 수신하여 전기로 변환하고, 재증폭·재타이밍 후 다시 빛으로 전송했다. 이 재생기는 비용이 높고, 파장 특이적이며, 비트 전송률에 의존적이었다.

1980년대 후반 드쉬르비르(Desurvire)와 미어스(Mears)가 시연한 **에르븀 첨가 광섬유 증폭기(EDFA)**는 광섬유 통신에 혁명을 일으켰다. EDFA는 빛을 직접 증폭한다 — C-대역 내 모든 파장을 동시에, 어떤 비트 전송률이든, 어떤 변조 방식이든.

### 7.2 동작 원리

실리카 섬유에 도핑된 에르븀 이온(Er$^{3+}$)은 ~1530~1565 nm(C-대역)에서 3준위 전이를 가진다. 980 nm 또는 1480 nm로 펌핑하면:

```
      ⁴I₁₁/₂ ─────── 980 nm 펌프 대역
                ↓ 빠른 비방사 전이
      ⁴I₁₃/₂ ─────── 준안정 상준위 (τ ≈ 10 ms)
                ↓ 신호 증폭 (1530-1565 nm)
      ⁴I₁₅/₂ ─────── 바닥 상태
```

- 980 nm 펌프: $^4I_{11/2}$ 대역에 흡수되고 $^4I_{13/2}$로 빠르게 완화. 낮은 잡음 지수(~3 dB).
- 1480 nm 펌프: $^4I_{13/2}$로 직접 펌핑. 효율이 높지만 잡음이 약간 높음.
- 신호 광자가 $^4I_{13/2}$에서 $^4I_{15/2}$로의 유도 방출을 자극하여 이득을 제공.

### 7.3 주요 EDFA 파라미터

- **이득**: 20~40 dB (100배~10,000배 증폭)
- **대역폭**: C-대역에서 ~35 nm (L-대역 EDFA 포함 시 ~70 nm로 확장)
- **포화 출력 광력**: 10~25 dBm (10~300 mW)
- **잡음 지수**: 3~6 dB (양자 한계: 3 dB)
- **섬유 길이**: 5~30 m의 에르븀 첨가 섬유

### 7.4 증폭된 자발 방출(ASE, Amplified Spontaneous Emission)

여기된 Er$^{3+}$ 이온으로부터의 자발 방출도 증폭되어 광대역 **ASE 잡음**을 생성한다. 이것이 광증폭 시스템의 근본적인 잡음원이다. 잡음 전력 스펙트럼 밀도는:

$$S_{\text{ASE}} = n_{\text{sp}}(G-1)h\nu$$

여기서 $n_{\text{sp}} \geq 1$은 자발 방출 계수이고 $G$는 이득이다. 잡음 지수는 $\text{NF} = 2n_{\text{sp}}(G-1)/G \approx 2n_{\text{sp}}$ (고이득 시)이다.

> **비유**: EDFA는 빛의 중계 기지국과 같다. 들판을 가로질러 속삭이는 메시지를 전달하는 사람들의 연쇄를 상상해 보라(신호). 각 중계 지점에서 메시지가 증폭되지만(더 크게 외침), 동시에 각 중계 지점은 주변 잡담(ASE 잡음)도 추가한다. 많은 중계 지점을 거치면 메시지는 크지만 잡음도 누적된다. 시스템 설계의 기술은 신호가 잡음보다 크게 유지되도록 중계 간격을 조절하는 것이다.

---

## 8. 파장 분할 다중화 (WDM)

### 8.1 개념

WDM은 라디오 주파수 다중화의 광학적 등가물이다: 서로 다른 파장의 여러 신호가 동일한 광섬유를 공유하며, 각각 독립적인 데이터를 전달한다. 이는 채널 수만큼 섬유의 용량을 곱한 효과를 낸다.

```
   λ₁ ──┐                                     ┌── λ₁
   λ₂ ──┤ MUX ════════ 단일 섬유 ════════ DEMUX ├── λ₂
   λ₃ ──┤      (모든 파장 함께)              ├── λ₃
    ⋮   ┘                                      └──  ⋮
   λₙ ──┘                                     └── λₙ
```

### 8.2 WDM 표준

**CWDM (광역 WDM, Coarse WDM)**: 채널 간격 20 nm, ~18개 채널 (1270~1610 nm). 저비용, 온도 제어 불필요. 단거리 (메트로, 접속 네트워크).

**DWDM (밀집 WDM, Dense WDM)**: 채널 간격 100 GHz (~0.8 nm), 50 GHz (~0.4 nm), 또는 12.5 GHz까지. C-대역에만 40~160개 이상의 채널. 장거리, 해저 케이블.

### 8.3 WDM 시스템 용량

현대 해저 케이블(예: Google의 Dunant, 2020)은:
- 광섬유 쌍당 ~250개의 DWDM 채널
- 각 채널: 200~400 Gb/s (코히런트 변조)
- 케이블당 여러 광섬유 쌍
- 케이블당 총 용량: ~250 Tb/s

### 8.4 핵심 구성 요소

- **다중화기/역다중화기**: 배열 도파로 격자(AWG, arrayed waveguide grating), 박막 필터
- **광 추가/제거 다중화기(OADM)**: 중간 노드에서 개별 파장 삽입/제거
- **재구성 가능한 OADM (ROADM)**: 파장 선택 스위치(WSS, wavelength-selective switch)를 이용한 원격 구성 파장 라우팅

---

## 9. 광섬유 브래그 격자 (FBG)

### 9.1 구조

광섬유 브래그 격자(FBG)는 광섬유 코어 내 굴절률의 주기적 변조이다:

$$n(z) = n_{\text{eff}} + \delta n \cos\!\left(\frac{2\pi z}{\Lambda}\right)$$

여기서 $\Lambda$는 격자 주기(보통 ~500 nm)이고, $\delta n$은 굴절률 변조 진폭(~$10^{-4}$ ~ $10^{-3}$)이다.

FBG는 위상 마스크나 두 빔 간섭을 이용하여 자외선 간섭 패턴을 코어에 조사함으로써 섬유에 기록된다.

### 9.2 브래그 조건(Bragg Condition)

격자는 **브래그 파장**에서 빛을 반사한다:

$$\boxed{\lambda_B = 2n_{\text{eff}}\Lambda}$$

다른 파장의 빛은 영향 없이 통과한다. 반사 대역폭은:

$$\Delta\lambda \approx \lambda_B \frac{\delta n}{n_{\text{eff}}} \quad (\text{약한 격자의 경우})$$

### 9.3 응용

**파장 필터**: WDM 시스템용 초협대역 통과/차단 필터.

**센서**: 브래그 파장은 온도($\sim 10\,\text{pm/°C}$) 및 변형률($\sim 1.2\,\text{pm/}\mu\varepsilon$)에 따라 이동한다. 다른 $\lambda_B$를 가진 여러 FBG를 단일 섬유에 다중화하여, 교량, 파이프라인, 항공기 날개 등의 분산형 센싱에 사용할 수 있다.

**분산 보상**: 처프 FBG(chirped FBG, $\Lambda$가 길이에 따라 변함)는 서로 다른 파장을 서로 다른 위치에서 반사하여, 분산된 펄스를 압축하는 제어된 시간 지연을 도입한다.

**레이저 안정화**: 광섬유 레이저의 FBG 반사경은 레이징 파장을 높은 정밀도와 안정성으로 결정한다.

---

## 10. Python 예제

### 10.1 광섬유 모드 계산

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, kv  # Bessel functions

def compute_modes(n1, n2, a, wavelength):
    """
    Compute guided mode effective indices for a step-index fiber.

    We solve the characteristic equation for LP modes:
    J_l(u) / [u * J_{l-1}(u)] = -K_l(w) / [w * K_{l-1}(w)]
    where u = a*sqrt(k0^2*n1^2 - beta^2), w = a*sqrt(beta^2 - k0^2*n2^2)

    This eigenvalue equation arises from matching the field and its derivative
    at the core-cladding boundary — the same physics as quantum well states.
    """
    k0 = 2 * np.pi / wavelength
    V = k0 * a * np.sqrt(n1**2 - n2**2)

    print(f"V-number: {V:.3f}")
    print(f"NA: {np.sqrt(n1**2 - n2**2):.4f}")
    print(f"Acceptance angle: {np.degrees(np.arcsin(np.sqrt(n1**2 - n2**2))):.2f}°")

    if V < 2.405:
        print("Single-mode fiber (only LP01 propagates)")
    else:
        # Approximate number of modes
        M_step = V**2 / 2
        print(f"Multimode fiber: ~{int(M_step)} modes (step-index)")
        print(f"                 ~{int(V**2/4)} modes (graded-index parabolic)")

    return V

# Standard single-mode fiber (SMF-28)
print("=== SMF-28 at 1310 nm ===")
V_1310 = compute_modes(n1=1.4504, n2=1.4447, a=4.1e-6, wavelength=1310e-9)

print("\n=== SMF-28 at 1550 nm ===")
V_1550 = compute_modes(n1=1.4504, n2=1.4447, a=4.1e-6, wavelength=1550e-9)

print("\n=== Multimode fiber at 850 nm ===")
V_mm = compute_modes(n1=1.480, n2=1.460, a=25e-6, wavelength=850e-9)
```

### 10.2 감쇠 스펙트럼 모델

```python
import numpy as np
import matplotlib.pyplot as plt

def fiber_attenuation(wavelength_nm):
    """
    Model the attenuation spectrum of standard silica fiber.

    Three contributions are modeled:
    1. Rayleigh scattering (∝ λ^-4): dominant below ~1300 nm
    2. IR absorption (exponential rise): dominant above ~1600 nm
    3. OH peak at 1383 nm: from residual water in the glass

    The minimum loss occurs around 1550 nm — the sweet spot for telecom.
    This is not a coincidence: the entire long-haul fiber industry
    operates at 1550 nm because physics dictates it.
    """
    lam = wavelength_nm / 1000.0  # Convert to micrometers

    # Rayleigh scattering: C_R / lambda^4
    rayleigh = 0.8 / lam**4

    # Infrared absorption: exponential tail of molecular resonances
    ir_absorption = 6e11 * np.exp(-48.0 / lam)

    # UV absorption: Urbach tail
    uv_absorption = 1.5e-2 * np.exp(4.6 * (1.0/lam - 1.0/0.16))

    # OH impurity peak at 1383 nm (Gaussian approximation)
    oh_peak = 0.5 * np.exp(-0.5 * ((lam - 1.383) / 0.015)**2)

    total = rayleigh + ir_absorption + uv_absorption + oh_peak
    return total, rayleigh, ir_absorption, oh_peak

wavelengths = np.linspace(800, 1700, 1000)
total, rayleigh, ir, oh = fiber_attenuation(wavelengths)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(wavelengths, total, 'k-', linewidth=2, label='Total')
ax.semilogy(wavelengths, rayleigh, 'b--', linewidth=1, label='Rayleigh scattering')
ax.semilogy(wavelengths, ir, 'r--', linewidth=1, label='IR absorption')
ax.semilogy(wavelengths, oh, 'g--', linewidth=1, label='OH peak (1383 nm)')

# Mark telecom bands
bands = {'O': (1260, 1360), 'E': (1360, 1460), 'S': (1460, 1530),
         'C': (1530, 1565), 'L': (1565, 1625)}
colors = ['#FFE0B2', '#E0F7FA', '#F3E5F5', '#E8F5E9', '#FFF9C4']
for (name, (lo, hi)), color in zip(bands.items(), colors):
    ax.axvspan(lo, hi, alpha=0.3, color=color, label=f'{name}-band')

ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Attenuation (dB/km)', fontsize=12)
ax.set_title('Silica Optical Fiber Attenuation Spectrum', fontsize=14)
ax.set_ylim(0.1, 10)
ax.set_xlim(800, 1700)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fiber_attenuation.png', dpi=150, bbox_inches='tight')
plt.show()

# Find and report the minimum
min_idx = np.argmin(total)
print(f"Minimum attenuation: {total[min_idx]:.3f} dB/km at {wavelengths[min_idx]:.0f} nm")
```

### 10.3 분산 계산

```python
import numpy as np
import matplotlib.pyplot as plt

def sellmeier_silica(wavelength_um):
    """
    Sellmeier equation for fused silica refractive index.

    The Sellmeier coefficients encode the resonance wavelengths
    of the UV and IR absorption bands of SiO2. Between these
    resonances, the material is transparent — the fiber window.
    """
    lam2 = wavelength_um**2
    # Standard Sellmeier coefficients for fused silica
    B = [0.6961663, 0.4079426, 0.8974794]
    C = [0.0684043**2, 0.1162414**2, 9.896161**2]  # C_i = lambda_i^2
    n2 = 1.0
    for bi, ci in zip(B, C):
        n2 += bi * lam2 / (lam2 - ci)
    return np.sqrt(n2)

def material_dispersion(wavelength_um):
    """
    Compute material dispersion D_mat = -(λ/c) * d²n/dλ².

    Material dispersion arises because the glass refractive index
    varies with wavelength. It is zero near 1270 nm and positive
    (anomalous) above that wavelength for silica.
    """
    dlam = 1e-4  # Small step for numerical derivative (in µm)
    lam = wavelength_um

    # Second derivative via central differences (3-point formula)
    n_plus = sellmeier_silica(lam + dlam)
    n_minus = sellmeier_silica(lam - dlam)
    n_center = sellmeier_silica(lam)
    d2n_dlam2 = (n_plus - 2*n_center + n_minus) / dlam**2

    # D = -(λ/c) * d²n/dλ² [ps/(nm·km)]
    c = 3e8  # m/s
    # Convert units: λ in µm → m, d²n/dλ² in 1/µm² → 1/m²
    D = -(lam * 1e-6 / c) * (d2n_dlam2 * 1e12)  # Result in ps/(nm·km)
    return D

wavelengths = np.linspace(1.0, 1.7, 500)  # micrometers
n_vals = sellmeier_silica(wavelengths)
D_vals = material_dispersion(wavelengths)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(wavelengths * 1000, n_vals, 'b-', linewidth=2)
ax1.set_ylabel('Refractive index n', fontsize=12)
ax1.set_title('Silica Refractive Index and Chromatic Dispersion', fontsize=14)
ax1.grid(True, alpha=0.3)

ax2.plot(wavelengths * 1000, D_vals, 'r-', linewidth=2)
ax2.axhline(y=0, color='gray', linestyle='--')
ax2.axvline(x=1310, color='green', linestyle=':', alpha=0.7,
            label='Zero dispersion (~1310 nm)')
ax2.axvline(x=1550, color='orange', linestyle=':', alpha=0.7,
            label='C-band center (1550 nm)')
ax2.set_xlabel('Wavelength (nm)', fontsize=12)
ax2.set_ylabel('D [ps/(nm·km)]', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fiber_dispersion.png', dpi=150, bbox_inches='tight')
plt.show()

# Report D at key wavelengths
for lam in [1.31, 1.55]:
    print(f"D at {lam*1000:.0f} nm: {material_dispersion(lam):.2f} ps/(nm·km)")
```

---

## 11. 요약

| 개념 | 핵심 공식 / 아이디어 |
|------|----------------------|
| 전반사 | 임계각: $\theta_c = \arcsin(n_2/n_1)$ |
| 수치 개구 | $\text{NA} = \sqrt{n_1^2 - n_2^2}$ |
| V-수 | $V = (2\pi a/\lambda)\,\text{NA}$ |
| 단일 모드 조건 | $V < 2.405$ |
| 모드 수 (계단형) | $M \approx V^2/2$ |
| 감쇠 | $P = P_0 \cdot 10^{-\alpha L/10}$; 최솟값 ~0.17 dB/km @ 1550 nm |
| 레일리 산란 | $\propto \lambda^{-4}$ |
| 모드 분산 (계단형) | $\Delta\tau \approx n_1 L \Delta / c$ |
| 색분산 | $\Delta\tau = |D| \Delta\lambda\, L$; $D = 0$ at ~1310 nm (표준 SMF) |
| PMD | $\Delta\tau = D_{\text{PMD}}\sqrt{L}$ |
| EDFA | C-대역 증폭 (1530-1565 nm); 이득 20-40 dB; NF $\geq$ 3 dB |
| 브래그 파장 | $\lambda_B = 2n_{\text{eff}}\Lambda$ |
| WDM | 단일 섬유에 다중 파장; DWDM 간격 ~0.4-0.8 nm |

---

## 12. 연습 문제

### 연습 문제 1: 섬유 설계

$\lambda = 1550\,\text{nm}$에서 동작하는 계단 굴절률 단일 모드 섬유를 $n_1 = 1.450$, $\Delta = 0.3\%$로 설계하라.

(a) $n_2$와 NA를 계산하라.
(b) 단일 모드 동작을 위한 최대 코어 반경 $a$를 구하라.
(c) 마르쿠제 근사를 이용하여 모드 필드 직경을 계산하라.
(d) 차단 파장은 얼마인가?

### 연습 문제 2: 링크 버짓

표준 SMF($\alpha = 0.2\,\text{dB/km}$ at 1550 nm)를 사용하는 광섬유 링크에서, 송신기 출력은 +3 dBm이고 수신기 감도는 -28 dBm이다. 각 커넥터의 손실은 0.5 dB이고, 스플라이스당 커넥터 2개, 스플라이스 5개가 있다.

(a) 총 커넥터/스플라이스 손실을 계산하라.
(b) 3 dB 시스템 마진을 포함했을 때 최대 광섬유 길이는?
(c) 중간 지점에 25 dB 이득의 EDFA를 삽입하면 최대 길이는 얼마나 늘어나는가?

### 연습 문제 3: 분산 제한 거리

10 Gb/s NRZ 시스템이 표준 SMF($D = 17\,\text{ps/(nm·km)}$) 위에서 1550 nm의 DFB 레이저($\Delta\lambda = 0.1\,\text{nm}$)를 사용한다.

(a) km당 펄스 확산을 계산하라.
(b) $\Delta\tau < T_{\text{bit}}/4$ 기준($T_{\text{bit}} = 100\,\text{ps}$)으로 최대 무보상 전송 거리를 구하라.
(c) 표준 섬유 100 km를 보상하려면 DCF($D = -80\,\text{ps/(nm·km)}$)가 얼마나 필요한가?

### 연습 문제 4: FBG 센서

$n_{\text{eff}} = 1.447$인 섬유에 $\Lambda = 535\,\text{nm}$의 FBG가 기록되어 있다.

(a) 브래그 파장을 계산하라.
(b) 온도 민감도가 $10\,\text{pm/°C}$일 때, 50°C 온도 상승에 해당하는 브래그 파장 이동량은?
(c) 변형률 민감도가 $1.2\,\text{pm/}\mu\varepsilon$일 때, (b)와 동일한 이동량을 유발하는 변형률은?

---

## 13. 참고문헌

1. Saleh, B. E. A., & Teich, M. C. (2019). *Fundamentals of Photonics* (3rd ed.). Wiley. — Chapter 10.
2. Agrawal, G. P. (2021). *Fiber-Optic Communication Systems* (5th ed.). Wiley.
3. Okamoto, K. (2022). *Fundamentals of Optical Waveguides* (3rd ed.). Academic Press.
4. Desurvire, E. (2002). *Erbium-Doped Fiber Amplifiers*. Wiley.
5. Kao, C. K. (2009). Nobel Prize Lecture: "Sand from centuries past: Send future voices fast."

---

[← 이전: 08. 레이저 기초](08_Laser_Fundamentals.md) | [다음: 10. 푸리에 광학 →](10_Fourier_Optics.md)
