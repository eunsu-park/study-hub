# 17. 분광학

[← 이전: 16. 적응광학](16_Adaptive_Optics.md) | [개요 →](00_Overview.md)

---

1814년, 요제프 폰 프라운호퍼(Joseph von Fraunhofer)는 프리즘을 통해 망원경으로 태양을 관찰하다가 놀라운 사실을 발견했습니다: 태양빛의 무지개 색깔 띠 위에 특정 파장에서 수백 개의 어두운 선이 교차하고 있었습니다. 이 **프라운호퍼 선(Fraunhofer lines)** — 태양 대기의 원소들이 남긴 흡수 지문 — 은 빛과의 상호작용을 통해 물질을 연구하는 새로운 방법을 열었습니다. 1860년대에 이르러 키르히호프(Kirchhoff)와 분젠(Bunsen)은 모든 원소가 고유한 스펙트럼 선 패턴을 만들어낸다는 사실을 확립했고, 분광학(Spectroscopy)은 화학과 천체물리학 모두에서 가장 강력한 도구가 되었습니다. 오늘날도 마찬가지입니다: 멀리 있는 은하의 구성 성분 결정부터 혈중 산소 수준 측정까지, 반도체 제조의 품질 관리부터 공항에서의 폭발물 탐지까지.

**분광학(Spectroscopy)**은 물질이 파장(혹은 동등하게, 진동수 또는 에너지)에 따라 전자기 복사를 어떻게 흡수·방출·산란하는지를 연구하는 학문입니다. 이 레슨에서는 스펙트럼 선의 물리적 원리, 이를 분해하는 기기, 그리고 스펙트럼에서 정보를 추출하는 분석 기법을 다룹니다. 앞서 학습한 회절(L06), 간섭(L05), 광학 기기(L04) 레슨과 연결하여, 이러한 기본 현상들이 모든 분광기 설계의 근간을 이룬다는 것을 보여줍니다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. 양자화된 에너지 전이로부터 원자 및 분자 스펙트럼의 기원을 설명하고 키르히호프의 분광학 3법칙을 서술한다
2. 복사 전이에 대한 아인슈타인 A 및 B 계수를 유도하고 이를 흡수 및 방출 속도와 연관짓는다
3. 세 가지 주요 스펙트럼 선 넓어짐 메커니즘(자연, 도플러, 압력)을 설명하고 포이그트 프로파일(Voigt profile)을 계산한다
4. 프리즘 및 회절 격자 분광기의 분해능과 자유 스펙트럼 범위(free spectral range)를 분석한다
5. 패브리-페로 간섭계(Fabry-Pérot interferometer)와 푸리에 변환 분광기(Fourier transform spectrometer)의 동작 원리와 성능을 설명한다
6. 비어-람베르트 법칙(Beer-Lambert law)을 정량적 흡수 분광학에 적용하고 흡광도 데이터로부터 농도를 계산한다
7. 형광(fluorescence), 라만(Raman), 레이저 분광학 기법과 그 응용을 설명한다
8. Python으로 스펙트럼 선 피팅, 분광기 시뮬레이션, 패브리-페로 분석을 구현한다

---

## 목차

1. [스펙트럼의 기초](#1-스펙트럼의-기초)
2. [복사 전이](#2-복사-전이)
3. [스펙트럼 선 넓어짐](#3-스펙트럼-선-넓어짐)
4. [분산형 분광기](#4-분산형-분광기)
5. [간섭계형 분광기](#5-간섭계형-분광기)
6. [흡수 분광학](#6-흡수-분광학)
7. [방출 및 형광 분광학](#7-방출-및-형광-분광학)
8. [라만 분광학](#8-라만-분광학)
9. [현대적 기법](#9-현대적-기법)
10. [Python 예제](#10-python-예제)
11. [요약](#11-요약)
12. [연습 문제](#12-연습-문제)
13. [참고문헌](#13-참고문헌)

---

## 1. 스펙트럼의 기초

### 1.1 원자 에너지 준위

원자는 양자역학에 의해 결정된 불연속적인 에너지 준위를 가집니다. 준위 $E_1$과 $E_2$ 사이의 전이에서 방출 또는 흡수되는 광자의 에너지는:

$$h\nu = E_2 - E_1 \qquad \Leftrightarrow \qquad \lambda = \frac{hc}{E_2 - E_1}$$

수소의 에너지 준위는 다음과 같습니다:

$$E_n = -\frac{13.6\,\text{eV}}{n^2}$$

이로부터 잘 알려진 발머 계열(Balmer series, 가시광선), 라이만 계열(Lyman series, UV), 파셴 계열(Paschen series, IR)이 나옵니다.

### 1.2 분자 스펙트럼

분자는 회전과 진동이라는 추가적인 자유도를 가져 훨씬 더 풍부한 스펙트럼을 만들어냅니다:

- **회전 스펙트럼(Rotational spectra)** (마이크로파, 원적외선): $E_J = BJ(J+1)$, 여기서 $B = \hbar^2/(2I)$는 회전 상수
- **진동 스펙트럼(Vibrational spectra)** (적외선): 조화 진동자 근사에서 $E_v = \hbar\omega_0(v + 1/2)$
- **전자 스펙트럼(Electronic spectra)** (가시광선, UV): 전자 상태 사이의 전이로, 보통 진동 구조(전자-진동 대역)가 동반됨

전체 에너지는 근사적으로:

$$E \approx E_{\text{electronic}} + E_{\text{vibrational}} + E_{\text{rotational}}$$

이며, $E_{\text{elec}} \gg E_{\text{vib}} \gg E_{\text{rot}}$ 관계가 성립합니다.

### 1.3 키르히호프의 분광학 법칙

구스타프 키르히호프(Gustav Kirchhoff, 1859)는 세 가지 경험 법칙을 정립했습니다:

1. **뜨겁고 밀도가 높은 물체**(고체, 액체, 또는 고밀도 기체)는 연속 스펙트럼(흑체 복사)을 방출한다
2. **뜨겁고 밀도가 낮은 기체**는 특정 파장에서 빛을 방출한다 — **방출선 스펙트럼(emission line spectrum)**
3. **연속 광원 앞에 있는 차가운 기체**는 동일한 특정 파장에서 빛을 흡수한다 — **흡수선 스펙트럼(absorption line spectrum)**

> **비유**: 스펙트럼 선을 각 원소의 바코드라고 생각하세요. 마트의 스캐너가 고유한 바코드 패턴으로 상품을 식별하듯, 분광학자들은 고유한 스펙트럼 선 패턴으로 원자와 분자를 식별합니다. 모든 원소는 빛을 흡수하거나 방출하는 파장의 집합인 자신만의 "바코드"를 가지며, 이 지문은 인간의 지문만큼 고유합니다.

### 1.4 전자기 스펙트럼과 분광학 영역

| 영역 | 파장 | 전이 | 기법 |
|--------|:----------:|-------------|-----------|
| 감마/X선 | < 10 nm | 내각 전자 | X선 분광학 |
| UV | 10–400 nm | 원자가 전자 | UV-Vis 분광광도법 |
| 가시광선 | 400–700 nm | 원자가 전자 | 광학 분광학 |
| 근적외선 | 0.7–2.5 $\mu$m | 배음 진동 | NIR 분광학 |
| 중적외선 | 2.5–25 $\mu$m | 기본 진동 | FTIR, IR 흡수 |
| 원적외선/THz | 25–1000 $\mu$m | 회전 | THz 분광학 |
| 마이크로파 | 1 mm–1 m | 분자 회전 | 마이크로파 분광학 |
| 전파 | > 1 m | 초미세 구조 (수소 21 cm 선) | 전파 천문학 |

---

## 2. 복사 전이

### 2.1 아인슈타인 계수

1917년, 아인슈타인은 복사와 물질 사이의 상호작용이 세 가지 과정을 포함한다는 것을 보였습니다:

**자발 방출(Spontaneous emission)** (속도 $A_{21}$): 들뜬 상태 2에 있는 원자가 자발적으로 상태 1로 붕괴하면서 광자를 방출합니다. 자발 방출 속도는:

$$\left(\frac{dN_2}{dt}\right)_{\text{spont}} = -A_{21} N_2$$

아인슈타인 A 계수는 s$^{-1}$ 단위를 가지며 복사 수명의 역수입니다: $\tau_{\text{rad}} = 1/A_{21}$.

**유도 방출(Stimulated emission)** (속도 $B_{21}$): 적절한 진동수의 광자가 들뜬 원자를 자극하여 동일한 광자를 방출하게 합니다. 속도는 복사 에너지 밀도 $u(\nu)$에 비례합니다:

$$\left(\frac{dN_2}{dt}\right)_{\text{stim}} = -B_{21} u(\nu) N_2$$

**흡수(Absorption)** (속도 $B_{12}$): 광자가 흡수되어 원자를 상태 1에서 상태 2로 들뜨게 합니다:

$$\left(\frac{dN_1}{dt}\right)_{\text{abs}} = -B_{12} u(\nu) N_1$$

### 2.2 계수들 사이의 관계

열적 평형에서 개체수 비는 볼츠만 분포에 의해 주어집니다:

$$\frac{N_2}{N_1} = \frac{g_2}{g_1}\exp\left(-\frac{h\nu}{k_BT}\right)$$

아인슈타인 관계가 플랑크 흑체 스펙트럼을 재현하도록 요구하면:

$$g_1 B_{12} = g_2 B_{21}$$

$$A_{21} = \frac{8\pi h\nu^3}{c^3} B_{21}$$

여기서 $g_1$, $g_2$는 두 준위의 통계적 가중치(축퇴도)입니다.

### 2.3 진동자 세기

**진동자 세기(oscillator strength)** $f_{12}$는 전이 확률의 무차원 척도입니다:

$$f_{12} = \frac{m_e c}{8\pi^2 e^2 \nu^2} \frac{g_2}{g_1} A_{21}$$

강한 전이는 $f \sim 1$ (예: 나트륨 D 선: $f = 0.65$)을 가집니다. 금지 전이(forbidden transition)는 $f \ll 1$ (예: 성운의 [O III] 선: $f \sim 10^{-8}$)을 가집니다.

### 2.4 선택 규칙

에너지 준위 사이의 모든 전이가 허용되는 것은 아닙니다. 원자에 대한 전기 쌍극자 선택 규칙(electric dipole selection rules)은:

$$\Delta l = \pm 1, \quad \Delta m_l = 0, \pm 1, \quad \Delta S = 0$$

이 규칙을 위반하는 전이는 "금지"되어 있습니다 — 자기 쌍극자(magnetic dipole) 또는 전기 사중극자(electric quadrupole) 상호작용을 통해 여전히 일어나지만, 훨씬 낮은 속도(더 긴 수명)로 발생합니다.

---

## 3. 스펙트럼 선 넓어짐

### 3.1 자연 넓어짐 (로렌츠형)

들뜬 상태의 유한한 복사 수명 $\tau$는 하이젠베르크 불확정성 원리에 의해 에너지 불확정성 $\Delta E \sim \hbar/\tau$를 초래합니다. 이는 **로렌츠형(Lorentzian)** 선 프로파일을 만들어냅니다:

$$\phi_L(\nu) = \frac{1}{\pi} \frac{\gamma/2}{(\nu - \nu_0)^2 + (\gamma/2)^2}$$

여기서 $\gamma = 1/(2\pi\tau)$는 Hz 단위의 반최대폭(HWHM)입니다. 자연 넓어짐은 일반적으로 매우 작으며 ($\Delta\nu \sim 10^7$ Hz, 즉 가시광선 파장에서 $\Delta\lambda \sim 10^{-5}$ nm), 보통 다른 메커니즘에 비해 무시할 수 있습니다.

### 3.2 도플러 넓어짐 (가우스형)

원자의 열운동은 도플러 편이의 분포를 유발합니다. 온도 $T$에서 맥스웰-볼츠만(Maxwell-Boltzmann) 속도 분포에 대해 선 프로파일은 **가우스형(Gaussian)**입니다:

$$\phi_G(\nu) = \frac{1}{\sigma_D\sqrt{2\pi}} \exp\left[-\frac{(\nu - \nu_0)^2}{2\sigma_D^2}\right]$$

여기서 도플러 폭(표준 편차)은:

$$\sigma_D = \frac{\nu_0}{c}\sqrt{\frac{k_BT}{m}}$$

그리고 FWHM은:

$$\Delta\nu_D = 2\sqrt{2\ln 2}\,\sigma_D = \frac{\nu_0}{c}\sqrt{\frac{8k_BT\ln 2}{m}}$$

$T = 5000$ K에서 수소 발머-$\alpha$ 선의 경우: $\Delta\lambda_D \approx 0.04$ nm.

### 3.3 압력 넓어짐 (로렌츠형)

인접 원자 또는 전자와의 충돌이 방출 과정을 방해하여 유효 결맞음 시간을 단축시킵니다. 이는 다음 폭을 가진 또 다른 로렌츠형 프로파일을 만들어냅니다:

$$\gamma_{\text{pressure}} \propto N_{\text{perturber}} \, v_{\text{rel}} \, \sigma_{\text{col}}$$

여기서 $N$은 교란 입자 밀도, $v_{\text{rel}}$은 상대 속도, $\sigma_{\text{col}}$은 충돌 단면적입니다. 압력 넓어짐은 고밀도 플라즈마와 별의 대기에서 지배적입니다.

### 3.4 포이그트 프로파일

도플러 넓어짐과 로렌츠형 넓어짐이 모두 중요할 때, 결과 프로파일은 두 프로파일의 **합성곱(convolution)**입니다:

$$\phi_V(\nu) = \phi_G * \phi_L = \int_{-\infty}^{\infty} \phi_G(\nu') \phi_L(\nu - \nu') \, d\nu'$$

이것이 **포이그트 프로파일(Voigt profile)**로, **파데예바 함수(Faddeeva function)** $w(z)$를 사용하여 표현할 수 있습니다:

$$\phi_V(\nu) = \frac{1}{\sigma_D\sqrt{2\pi}} \text{Re}[w(z)], \qquad z = \frac{(\nu - \nu_0) + i\gamma/2}{\sigma_D\sqrt{2}}$$

여기서 $w(z) = e^{-z^2}\text{erfc}(-iz)$는 복소 오차 함수입니다.

포이그트 프로파일은 가우스형 중심부($\nu_0$ 근처에서 도플러가 지배)와 로렌츠형 날개부($\nu_0$에서 멀어질수록 압력 넓어짐이 지배)를 가집니다.

### 3.5 등가 폭

**등가 폭(equivalent width)** $W_\lambda$는 스펙트럼 선의 전체 흡수량을 측정하며, 연속 강도와 같은 높이를 가지면서 선과 동일한 면적을 갖는 직사각형의 폭으로 정의됩니다:

$$W_\lambda = \int \frac{I_c - I(\lambda)}{I_c} d\lambda$$

여기서 $I_c$는 연속 강도입니다. 등가 폭은 기기 분해능에 독립적이며 흡수체의 열주 밀도(column density)와 직접 관련됩니다(**성장 곡선(curve of growth)**을 통해).

---

## 4. 분산형 분광기

### 4.1 프리즘 분광기

프리즘은 물질 분산을 통해 파장을 분리합니다: 굴절률은 파장에 따라 변합니다(가시광선 영역에서 $dn/d\lambda < 0$, 정상 분산이라고 함).

꼭짓각 $A$인 프리즘의 **각분산(angular dispersion)**:

$$\frac{d\theta}{d\lambda} = \frac{t}{d} \frac{dn}{d\lambda}$$

여기서 $t$는 저면 길이이고 $d$는 출사면에서의 빔 직경입니다. 최소 편향에 대해 동등하게:

$$\frac{d\theta}{d\lambda} = \frac{2\sin(A/2)}{\cos[(\delta_{\min}+A)/2]} \frac{dn}{d\lambda}$$

**분해능(Resolving power)**:

$$R = \frac{\lambda}{\Delta\lambda} = t \frac{dn}{d\lambda}$$

500 nm에서 60 mm BK7 프리즘: $dn/d\lambda \approx -0.04\,\mu\text{m}^{-1}$이므로 $R \approx 2400$.

### 4.2 회절 격자 분광기

회절 격자(diffraction grating)는 현대 분광학의 핵심 도구로, 프리즘보다 높은 분해능과 넓은 파장 커버리지를 제공합니다.

**격자 방정식(grating equation)** (L06에서 복습):

$$d(\sin\theta_i + \sin\theta_m) = m\lambda$$

여기서 $d$는 홈 간격, $\theta_i$는 입사각, $\theta_m$은 $m$차 회절각입니다.

**각분산**:

$$\frac{d\theta_m}{d\lambda} = \frac{m}{d\cos\theta_m}$$

**분해능**:

$$R = mN$$

여기서 $N$은 조명된 홈의 총 수입니다. 1200 홈/mm, 폭 80 mm ($N = 96{,}000$)인 격자를 1차 회절로 사용하면 $R = 96{,}000$ — 많은 스펙트럼 선의 초미세 구조를 분해하기에 충분합니다.

### 4.3 블레이즈 격자

일반 격자는 빛을 여러 차수에 분배하여 대부분의 에너지를 낭비합니다. **블레이즈 격자(blazed grating)**는 톱니 모양의 홈 프로파일을 가져 빛을 특정 차수에 집중시킵니다. **블레이즈 각(blaze angle)** $\theta_B$는 각 홈 면에서의 정반사가 원하는 회절 차수와 일치하도록 선택됩니다:

$$\lambda_{\text{blaze}} = \frac{2d\sin\theta_B}{m}$$

블레이즈 격자는 블레이즈 파장에서 70% 이상의 효율을 달성합니다.

### 4.4 자유 스펙트럼 범위

**자유 스펙트럼 범위(free spectral range, FSR)**는 인접한 회절 차수가 겹치지 않는 파장 범위입니다:

$$\text{FSR} = \frac{\lambda}{m}$$

고차수는 더 좋은 분해능을 제공하지만 FSR은 더 작아집니다. 원하는 차수를 분리하기 위해 차수 분리 필터(order-sorting filter)나 교차 분산기(cross-disperser)를 사용합니다.

### 4.5 분광기 구성

| 구성 | 설명 | 응용 |
|---------------|-------------|-------------|
| 체르니-터너(Czerny-Turner) | 두 개의 오목 거울 + 평면 격자 | 범용 실험실 분광기 |
| 리트로(Littrow) | 격자가 분산기와 역반사기 역할 모두 수행 | 소형 고분해능 |
| 에셸(Echelle) | 높은 블레이즈 각, 고차수 ($m \sim 50$–$100$), 교차 분산 | 고분해능 항성 분광학 |
| 롤랜드 원(Rowland circle) | 오목 격자 (분산기 + 집속기) | X선, EUV 분광학 |

> **비유**: 회절 격자는 빛을 위한 베네치안 블라인드처럼 작동합니다. 블라인드 슬릿을 통과한 햇빛이 반대쪽 벽에 무지개 패턴을 만드는 것은 회절이 흰빛을 색으로 분리하는 것입니다. 격자의 홈이 슬릿 역할을 하지만, 파장을 탁월한 정확도로 분리하기 위해 나노미터 정밀도로 설계되어 있습니다.

---

## 5. 간섭계형 분광기

### 5.1 패브리-페로 간섭계

**패브리-페로 간섭계(Fabry-Pérot interferometer, FPI)**는 간격 $d$로 분리된 두 개의 평행한 부분 반사면으로 구성됩니다. 빛이 표면 사이를 왕복하며, 다음 조건에서 보강 간섭이 발생합니다:

$$2nd\cos\theta = m\lambda \qquad (m = 1, 2, 3, \ldots)$$

여기서 $n$은 간격 매질의 굴절률이고 $\theta$는 입사각입니다.

투과 강도는 **에어리 함수(Airy function)**를 따릅니다:

$$T = \frac{1}{1 + F\sin^2(\delta/2)}$$

여기서:

$$\delta = \frac{4\pi nd\cos\theta}{\lambda}, \qquad F = \frac{4R}{(1-R)^2}$$

이고 $R$은 거울 반사율입니다. $F$는 **세기 파이네스 계수(coefficient of finesse)**라고 합니다(파이네스 자체와 혼동하지 말 것).

### 5.2 파이네스와 분해능

**파이네스(finesse)** $\mathcal{F}$는 FSR과 선폭의 비입니다:

$$\mathcal{F} = \frac{\pi\sqrt{R}}{1-R} = \frac{\pi\sqrt{F}}{2}$$

| 반사율 $R$ | 파이네스 $\mathcal{F}$ |
|:----------------:|:--------------------:|
| 0.5 | 4.4 |
| 0.9 | 30 |
| 0.95 | 61 |
| 0.99 | 313 |

FSR과 분해능은:

$$\text{FSR} = \frac{\lambda^2}{2nd}, \qquad R_{\text{FP}} = \frac{2nd}{\lambda} \mathcal{F} = m\mathcal{F}$$

$d = 5$ mm, $R = 0.95$, $\lambda = 500$ nm인 FPI는 $m = 20{,}000$, $\mathcal{F} = 61$, $R_{\text{FP}} = 1.2 \times 10^6$을 줍니다 — 최고의 에셸 분광기에 필적하는 성능입니다.

### 5.3 푸리에 변환 분광학 (FTS)

**푸리에 변환 분광기(Fourier transform spectrometer)**는 마이컬슨 간섭계(Michelson interferometer)를 기반으로 합니다. 한 거울이 경로차 $\delta$의 범위를 스캔하면서 검출기는 **간섭무늬(interferogram)** — $\delta$의 함수로서의 강도 — 를 기록합니다:

$$I(\delta) = \int_0^{\infty} B(\nu)[1 + \cos(2\pi\nu\delta)] \, d\nu$$

여기서 $B(\nu)$는 스펙트럼 강도입니다. 스펙트럼은 간섭무늬를 푸리에 변환하여 복원됩니다:

$$B(\nu) = 2\int_0^{\infty} [I(\delta) - \langle I \rangle] \cos(2\pi\nu\delta) \, d\delta$$

**FTS의 장점**:
- **펠제트(Fellgett) (다중화) 이점**: 모든 파장을 동시에 측정 (스캐닝 대비 SNR $\propto \sqrt{N}$ 향상)
- **자크기노(Jacquinot) (처리량) 이점**: 원형 조리개를 크게 할 수 있음 (슬릿 불필요)
- **콘느(Connes) 이점**: 기준 레이저(He-Ne)를 통한 파장 보정이 매우 정밀함

**분해능**: 스펙트럼 분해능은 최대 경로차 $\delta_{\max}$에 의해 제한됩니다:

$$\Delta\nu = \frac{1}{2\delta_{\max}}$$

10 cm 경로차를 스캔하는 FTS는 $\Delta\nu = 0.05\,\text{cm}^{-1}$ ($\Delta\lambda \approx 10^{-3}$ nm at 500 nm), 즉 $R \approx 500{,}000$을 달성합니다.

FTS는 중적외선(FTIR 분광학)의 표준 기법으로, 열원이 약하고 다중화 이점이 중요한 영역입니다.

---

## 6. 흡수 분광학

### 6.1 비어-람베르트 법칙

빛이 흡수 매질을 통과할 때 투과 강도는 지수적으로 감소합니다:

$$I(\lambda) = I_0(\lambda) \exp[-\alpha(\lambda) l] = I_0(\lambda) \exp[-\varepsilon(\lambda) c l]$$

여기서:
- $\alpha(\lambda)$는 흡수 계수(m$^{-1}$)
- $\varepsilon(\lambda)$는 몰 흡수 계수(L mol$^{-1}$ cm$^{-1}$)
- $c$는 농도(mol/L)
- $l$은 경로 길이(cm)

### 6.2 흡광도와 투과율

**투과율(transmittance)**과 **흡광도(absorbance)**는:

$$T = \frac{I}{I_0}, \qquad A = -\log_{10}(T) = \varepsilon c l$$

흡광도는 농도에 선형입니다 — 이것이 정량 분석의 기초입니다. UV-Vis 분광광도계는 $A(\lambda)$를 측정하고 검량선을 사용하여 농도를 결정합니다.

### 6.3 비어-람베르트 법칙으로부터의 이탈

이 법칙은 다음을 가정합니다:
- 단색광 (다색광은 높은 흡광도에서 비선형성을 유발함)
- 묽은 용액 (높은 농도에서 분자간 상호작용이 스펙트럼을 이동시킴)
- 산란이나 형광 없음
- 시료를 통한 균일한 경로

높은 농도에서 ($A > 2$, 즉 $T < 1\%$), 분광기의 미광(stray light)이 겉보기 편차를 유발합니다.

### 6.4 차분 광학 흡수 분광학 (DOAS)

DOAS(Differential Optical Absorption Spectroscopy)는 각 흡수체의 좁은 스펙트럼 구조만을 피팅하여 광대역 흡수 특성(레일리 산란, 에어로졸)을 제거합니다. 이 기법은 대기 미량 기체 모니터링(NO$_2$, SO$_2$, O$_3$, HCHO)에 널리 사용됩니다.

---

## 7. 방출 및 형광 분광학

### 7.1 방출 분광학

방출 분광학(emission spectroscopy)에서는 원자나 분자를 (열적, 전기적, 또는 광학적으로) 들뜨게 하고 방출된 빛을 분석합니다. 예시:

- **불꽃 방출(Flame emission)**: 시료를 불꽃에 분무하면 알칼리 금속이 특유의 색을 냅니다 (Na: 노란색, K: 보라색, Li: 빨간색)
- **아크/스파크 방출(Arc/spark emission)**: 금속 및 합금을 위한 고에너지 들뜸
- **ICP-OES**: 유도 결합 플라즈마 광학 방출 분광법(Inductively Coupled Plasma Optical Emission Spectroscopy) — 원소 분석의 핵심 도구

### 7.2 형광 분광학

**형광(fluorescence)**은 전자 들뜸 후에 빛이 방출되는 현상입니다. 이 과정은 **야블론스키 다이어그램(Jablonski diagram)**으로 설명됩니다:

```
        S₂ ────── 빠른 내부 전환
        |
        ▼
S₁ ──── ────── 진동 이완 (ps)
│    ╲
│     ╲ 계간 교차
│      ╲
│       T₁ ──── 인광 (ms-s)
│
▼ 형광 (ns)
S₀
```

주요 성질:

- **스토크스 이동(Stokes shift)**: 형광은 들뜸 빛보다 항상 더 긴 파장(더 낮은 에너지)에서 발생하는데, 이는 들뜬 상태에서의 진동 이완 때문입니다
- **양자 수율(Quantum yield)**: $\Phi = k_r / (k_r + k_{nr})$, 여기서 $k_r$은 복사 속도이고 $k_{nr}$은 모든 비복사 감쇠 채널을 포함합니다
- **형광 수명(Fluorescence lifetime)**: $\tau_f = 1/(k_r + k_{nr})$, 유기 형광단의 경우 보통 1–100 ns

### 7.3 응용

| 기법 | 응용 |
|-----------|-------------|
| 레이저 유도 형광(LIF, Laser-Induced Fluorescence) | 연소 진단, 유동 가시화 |
| 형광 현미경 | 세포 생물학, 의료 영상 |
| 시간 분해 형광(Time-resolved fluorescence) | 단백질 역학, FRET 거리 측정 |
| 형광 상관 분광학(Fluorescence correlation spectroscopy) | 확산 계수, 결합 반응 속도론 |

---

## 8. 라만 분광학

### 8.1 라만 효과

단색광이 분자에서 산란될 때, 대부분의 광자는 **레일리 산란(Rayleigh scattering)**으로 같은 진동수에서 탄성 산란됩니다. 그러나 소수($\sim 10^{-6}$)는 비탄성 산란되어 — 분자 진동을 들뜨게 하거나 가라앉히면서 에너지를 얻거나 잃습니다:

$$\nu_{\text{scattered}} = \nu_0 \pm \nu_{\text{vib}}$$

- **스토크스 선(Stokes lines)** ($\nu_0 - \nu_{\text{vib}}$): 광자가 에너지를 잃음; 분자가 진동 양자를 얻음
- **반스토크스 선(Anti-Stokes lines)** ($\nu_0 + \nu_{\text{vib}}$): 광자가 에너지를 얻음; 분자가 진동 양자를 잃음

반스토크스 대 스토크스 강도 비는 볼츠만 인자에 의해 결정됩니다:

$$\frac{I_{\text{aS}}}{I_{\text{S}}} = \left(\frac{\nu_0 + \nu_{\text{vib}}}{\nu_0 - \nu_{\text{vib}}}\right)^4 \exp\left(-\frac{h\nu_{\text{vib}}}{k_BT}\right)$$

### 8.2 라만 대 IR 분광학

두 기법 모두 분자 진동을 탐색하지만 다른 메커니즘을 통합니다:

| 특성 | 라만 | IR 흡수 |
|----------|:-----:|:-------------:|
| 메커니즘 | 분극률 변화 | 쌍극자 모멘트 변화 |
| 선택 규칙 | $\Delta\alpha \neq 0$ | $\Delta\mu \neq 0$ |
| 물의 간섭 | 최소 | 강함 |
| 시료 준비 | 최소 | 종종 박막/KBr 펠릿 필요 |
| 동핵 분자 | 활성 (O$_2$, N$_2$) | 비활성 |
| 공간 분해능 | $\sim 1\,\mu$m (공초점) | $\sim 10\,\mu$m |

두 기법은 **상보적**입니다: 중심 대칭 분자에서 라만 활성인 모드는 IR 비활성인 경향이 있으며 (상호 배제 규칙(rule of mutual exclusion)).

### 8.3 향상된 라만 기법

- **SERS** (표면 강화 라만 분광학, Surface-Enhanced Raman Spectroscopy): 금속 나노구조가 플라즈몬 강화를 통해 라만 신호를 $10^6$–$10^{10}$배 증폭시킵니다. 단일 분자 검출이 가능합니다.
- **CARS** (결맞음 반스토크스 라만 분광학, Coherent Anti-Stokes Raman Spectroscopy): 두 레이저 빔이 결맞음 반스토크스 신호를 생성합니다. 자발적 라만보다 훨씬 강하며; 연소 진단 및 현미경에 사용됩니다.

---

## 9. 현대적 기법

### 9.1 레이저 분광학

레이저는 기존 광원으로는 불가능한 분광학 기법을 가능하게 합니다:

- **LIBS** (레이저 유도 붕괴 분광학, Laser-Induced Breakdown Spectroscopy): 고출력 레이저 펄스가 시료 표면을 삭마 및 이온화시키고; 결과적인 플라즈마 방출이 원소 구성을 드러냅니다. 원격 분석(화성 탐사선), 미술품 인증, 산업 분류에 사용됩니다.
- **공진기 링다운 분광학(CRDS, Cavity Ring-Down Spectroscopy)**: 레이저 펄스가 고반사율 거울 ($R > 99.99\%$) 사이를 왕복합니다. 감쇠 시간은 공진기 내 흡수에 의존합니다. $10^{-10}$ cm$^{-1}$ 만큼 작은 흡수를 검출 — 미량 기체 감지에 이상적입니다.
- **포화 흡수 분광학(Saturated absorption spectroscopy)**: 강한 "펌프" 빔이 전이를 포화시켜 도플러 넓어짐 프로파일에 좁은 "램 딥(Lamb dip)"을 만듭니다. 주파수 표준을 위한 도플러 이하 분해능을 가능하게 합니다.

### 9.2 시간 분해 분광학

극초단 레이저(펨토초~아토초)는 동역학 연구를 가능하게 합니다:

- **펌프-탐침 분광학(Pump-probe spectroscopy)**: 펌프 펄스가 시료를 들뜨게 하고; 탐침 펄스가 가변 지연에서 흡수를 측정합니다. 시간 분해능은 펄스 지속 시간($\sim 10$ fs)으로 제한됩니다.
- **과도 흡수(Transient absorption)**: 들뜸 후 광학 밀도 변화 $\Delta A(t, \lambda)$를 측정합니다. 들뜬 상태 동역학, 전하 전달, 이완 경로를 파악합니다.

### 9.3 천문 분광학

분광학은 천체물리학의 근간입니다:

- **적색 편이(Redshift)**: 스펙트럼 선의 도플러 편이가 후퇴 속도를 측정합니다: $z = \Delta\lambda/\lambda = v/c$ (비상대론적)
- **항성 분류(Stellar classification)**: 흡수선에 의해 결정되는 스펙트럼 형 (O, B, A, F, G, K, M)
- **시선 속도법(Radial velocity method)**: 주기적 도플러 편이로부터 외계 행성 탐지 (현대 에셸 분광기로 ~1 m/s 정밀도)
- **화학 원소 존재비(Chemical abundances)**: 등가 폭과 스펙트럼 합성이 별의 대기에서 원소 존재비를 결정합니다

### 9.4 영상 분광학

**초분광 영상(Hyperspectral imaging)**은 분광학과 공간 정보를 결합하여 데이터 큐브 $(x, y, \lambda)$를 생성합니다:

- **원격 탐사(Remote sensing)**: 광물 지도 작성, 식생 건강, 대기 모니터링
- **의료 영상(Medical imaging)**: 조직 산소화, 암 탐지
- **미술 보존(Art conservation)**: 시료 채취 없이 안료 식별

---

## 10. Python 예제

### 10.1 스펙트럼 선 프로파일

```python
import numpy as np

def lorentzian(nu: np.ndarray, nu0: float, gamma: float) -> np.ndarray:
    """Lorentzian (natural/pressure broadened) line profile.

    L(nu) = (1/pi) * (gamma/2) / ((nu - nu0)^2 + (gamma/2)^2)

    Parameters
    ----------
    nu : array  — Frequency grid
    nu0 : float  — Line center frequency
    gamma : float  — Full width at half maximum (FWHM)
    """
    return (1 / np.pi) * (gamma / 2) / ((nu - nu0)**2 + (gamma / 2)**2)


def gaussian(nu: np.ndarray, nu0: float, sigma: float) -> np.ndarray:
    """Gaussian (Doppler-broadened) line profile.

    G(nu) = (1/(sigma*sqrt(2*pi))) * exp(-(nu-nu0)^2 / (2*sigma^2))

    Parameters
    ----------
    nu : array  — Frequency grid
    nu0 : float  — Line center frequency
    sigma : float  — Standard deviation (Doppler width)
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((nu - nu0) / sigma)**2
    )


def voigt(nu: np.ndarray, nu0: float, sigma: float,
          gamma: float) -> np.ndarray:
    """Voigt profile via numerical convolution of Gaussian and Lorentzian.

    The Voigt profile is the convolution G * L, representing a line
    broadened by both Doppler (Gaussian) and pressure (Lorentzian)
    mechanisms simultaneously.

    Uses direct numerical convolution (no scipy dependency).
    """
    # Evaluate Lorentzian on the same grid
    L = lorentzian(nu, nu0, gamma)
    # Convolve with Gaussian kernel
    dnu = nu[1] - nu[0]
    # Gaussian kernel centered at zero
    kernel_half = int(5 * sigma / dnu)
    nu_kernel = np.arange(-kernel_half, kernel_half + 1) * dnu
    G_kernel = gaussian(nu_kernel + nu0, nu0, sigma) * dnu
    G_kernel /= G_kernel.sum()  # normalize
    V = np.convolve(L, G_kernel, mode='same')
    return V
```

### 10.2 회절 격자 분광기

```python
def grating_angles(wavelengths: np.ndarray, d: float, theta_i: float,
                   m: int = 1) -> np.ndarray:
    """Compute diffraction angles for a grating spectrometer.

    Grating equation: d * (sin(theta_i) + sin(theta_m)) = m * lambda

    Parameters
    ----------
    wavelengths : array  — Wavelengths in same units as d
    d : float  — Groove spacing
    theta_i : float  — Incidence angle in radians
    m : int  — Diffraction order

    Returns
    -------
    theta_m : array  — Diffraction angles (radians); NaN where no solution exists
    """
    sin_theta_m = m * wavelengths / d - np.sin(theta_i)
    # Physical solutions require |sin(theta_m)| <= 1
    valid = np.abs(sin_theta_m) <= 1.0
    theta_m = np.full_like(wavelengths, np.nan)
    theta_m[valid] = np.arcsin(sin_theta_m[valid])
    return theta_m


def resolving_power_grating(N: int, m: int = 1) -> float:
    """Resolving power of a diffraction grating: R = m * N."""
    return m * N


def focal_plane_positions(wavelengths: np.ndarray, d: float,
                          theta_i: float, f: float,
                          m: int = 1) -> np.ndarray:
    """Map wavelengths to positions on the focal plane of a spectrometer.

    Parameters
    ----------
    f : float  — Focal length of the camera mirror/lens
    Returns x-positions on the detector (same units as f).
    """
    theta_m = grating_angles(wavelengths, d, theta_i, m)
    # Linear dispersion: x = f * theta_m (small angle approx for deviation)
    theta_center = np.nanmedian(theta_m)
    x = f * np.tan(theta_m - theta_center)
    return x
```

### 10.3 패브리-페로 투과

```python
def fabry_perot_transmission(wavelength: np.ndarray, d: float,
                              R: float, n: float = 1.0,
                              theta: float = 0.0) -> np.ndarray:
    """Compute Fabry-Pérot etalon transmission (Airy function).

    T = 1 / (1 + F * sin^2(delta/2))
    delta = 4*pi*n*d*cos(theta) / lambda
    F = 4*R / (1 - R)^2

    Parameters
    ----------
    wavelength : array  — Wavelength(s)
    d : float  — Mirror separation (same units as wavelength)
    R : float  — Mirror reflectance (0 to 1)
    n : float  — Refractive index of gap medium
    theta : float  — Angle of incidence (radians)
    """
    delta = 4 * np.pi * n * d * np.cos(theta) / wavelength
    F = 4 * R / (1 - R)**2
    T = 1.0 / (1.0 + F * np.sin(delta / 2)**2)
    return T


def finesse(R: float) -> float:
    """Reflectance finesse of a Fabry-Pérot interferometer."""
    return np.pi * np.sqrt(R) / (1 - R)
```

### 10.4 비어-람베르트 흡수

```python
def beer_lambert(wavelengths: np.ndarray,
                 epsilon: np.ndarray | float,
                 c: float, l: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute transmittance and absorbance using Beer-Lambert law.

    A = epsilon * c * l
    T = 10^(-A)

    Parameters
    ----------
    wavelengths : array  — Wavelength grid (for labeling; not used in calculation)
    epsilon : array or float  — Molar absorption coefficient(s) (L/(mol*cm))
    c : float  — Concentration (mol/L)
    l : float  — Path length (cm)

    Returns
    -------
    transmittance, absorbance : arrays
    """
    epsilon = np.asarray(epsilon)
    A = epsilon * c * l
    T = 10.0**(-A)
    return T, A
```

---

## 11. 요약

| 개념 | 핵심 공식 / 개념 |
|---------|--------------------|
| 광자 에너지 | $E = h\nu = hc/\lambda$ |
| 키르히호프의 법칙 | 연속, 방출, 흡수 스펙트럼 |
| 아인슈타인 계수 | $A_{21} = (8\pi h\nu^3/c^3)B_{21}$; $g_1 B_{12} = g_2 B_{21}$ |
| 자연 넓어짐 | 로렌츠형: $\gamma = 1/(2\pi\tau)$ |
| 도플러 넓어짐 | 가우스형: $\sigma_D = (\nu_0/c)\sqrt{k_BT/m}$ |
| 포이그트 프로파일 | 가우스형과 로렌츠형의 합성곱 |
| 격자 방정식 | $d(\sin\theta_i + \sin\theta_m) = m\lambda$ |
| 격자 분해능 | $R = mN$ |
| FP 투과 | $T = [1 + F\sin^2(\delta/2)]^{-1}$ |
| FP 파이네스 | $\mathcal{F} = \pi\sqrt{R}/(1-R)$ |
| FTS 분해능 | $\Delta\nu = 1/(2\delta_{\max})$ |
| 비어-람베르트 법칙 | $A = \varepsilon c l$; $T = 10^{-A}$ |
| 라만 편이 | $\nu_{\text{scattered}} = \nu_0 \pm \nu_{\text{vib}}$ |
| 스토크스 이동 | 형광이 들뜸보다 적색 편이됨 |

---

## 12. 연습 문제

### 연습 문제 1: 선 프로파일 분석

수소 발머-$\alpha$ 선 (656.28 nm)이 온도 $T = 10{,}000$ K, 전자 밀도 $n_e = 10^{21}\,\text{m}^{-3}$인 기체 방전에서 관측됩니다. (a) 도플러 폭 $\Delta\lambda_D$를 계산하시오. (b) 이 밀도에서 압력 넓어짐 폭은 대략 $\Delta\lambda_P \approx 0.02$ nm입니다. 진동수 단위의 로렌츠형 FWHM을 계산하시오. (c) 포이그트 프로파일을 그리고 같은 폭의 순수 가우스형 및 순수 로렌츠형 프로파일과 비교하시오. (d) 선 중심에서 얼마나 멀어지면 로렌츠형 날개부가 가우스형을 지배하는가?

### 연습 문제 2: 회절 격자 분광기 설계

나트륨 D 이중선(589.0 nm과 589.6 nm, 간격 0.6 nm)을 분해하기 위한 체르니-터너 분광기를 설계하시오. (a) 필요한 분해능은 얼마인가? (b) 1차 회절에서 1200 홈/mm 격자를 사용할 때, 필요한 최소 조명 폭은 얼마인가? (c) 카메라 초점 거리가 500 mm라면, 검출기에서 두 선의 선형 간격은 얼마인가? (d) 1차 및 2차에서의 자유 스펙트럼 범위는 얼마인가?

### 연습 문제 3: 패브리-페로 분광학

패브리-페로 에탈론이 거울 간격 $d = 10$ mm, 반사율 $R = 0.97$, 간격 굴절률 $n = 1.0$을 가집니다. (a) 파이네스, FSR ($\lambda = 500$ nm에서 nm 단위), 분해능을 계산하시오. (b) 500 nm 주변 1 nm 파장 범위에 걸쳐 투과 함수를 그리시오. (c) 이 범위 내에 투과 봉우리가 몇 개 있는가? (d) 압전 액추에이터를 사용하여 간격 간격을 $\Delta d = 250$ nm만큼 변경하여 에탈론을 스캔하면, 투과 봉우리는 어느 파장 범위를 스캔하는가?

### 연습 문제 4: 정량적 흡수 분석

과망간산칼륨(KMnO$_4$) 용액은 525 nm에서 최대 몰 흡수 계수 $\varepsilon = 2{,}455\,\text{L}\,\text{mol}^{-1}\,\text{cm}^{-1}$를 가집니다. (a) 1 cm 큐벳에서 농도 $c = 0.01, 0.02, 0.05, 0.1, 0.2$ mmol/L에 대해 525 nm에서의 흡광도와 투과율을 그리시오. (b) 측정된 투과율이 35%라면 농도는 얼마인가? (c) 흡광도가 $A = 2$ (투과율 1%)에 도달하는 농도는 얼마인가? (d) 매우 높은 흡광도와 매우 낮은 흡광도에서 농도 측정의 정확도에 어떤 일이 일어나는지 논하시오.

---

## 13. 참고문헌

1. Hecht, E. (2017). *Optics* (5th ed.). Pearson. — Chapter 9 on interference (Fabry-Pérot), Chapter 10 on diffraction (gratings).
2. Demtröder, W. (2015). *Laser Spectroscopy 1: Basic Principles* (5th ed.). Springer. — Comprehensive reference on spectroscopic techniques and laser methods.
3. Banwell, C. N., & McCash, E. M. (1994). *Fundamentals of Molecular Spectroscopy* (4th ed.). McGraw-Hill. — Accessible introduction to rotational, vibrational, and electronic spectroscopy.
4. Hollas, J. M. (2004). *Modern Spectroscopy* (4th ed.). Wiley. — Covers all major spectroscopic techniques with clear explanations.
5. Thorne, A. P., Litzén, U., & Johansson, S. (1999). *Spectrophysics: Principles and Applications*. Springer. — Detailed treatment of spectroscopic instruments and techniques.
6. Griffiths, P. R., & de Haseth, J. A. (2007). *Fourier Transform Infrared Spectrometry* (2nd ed.). Wiley. — The standard FTIR reference.
7. Ferraro, J. R., Nakamoto, K., & Brown, C. W. (2003). *Introductory Raman Spectroscopy* (2nd ed.). Academic Press. — Accessible Raman spectroscopy text.
8. Gray, D. F. (2022). *The Observation and Analysis of Stellar Photospheres* (4th ed.). Cambridge University Press. — Stellar spectroscopy including equivalent widths, curve of growth, and abundance analysis.

---

[← 이전: 16. 적응광학](16_Adaptive_Optics.md) | [개요 →](00_Overview.md)
