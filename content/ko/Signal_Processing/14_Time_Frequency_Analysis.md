# 14. 시간-주파수 분석(Time-Frequency Analysis)

**이전**: [13. 적응 필터](./13_Adaptive_Filters.md) | **다음**: [15. 영상 신호 처리](./15_Image_Signal_Processing.md)

---

푸리에 변환(Fourier Transform)은 신호에 어떤 주파수가 포함되어 있는지를 알려주지만, 언제 발생하는지는 알 수 없습니다. 음악, 음성, 지진 신호, 생체 리듬 등 비정상(non-stationary) 신호에는 스펙트럼 내용이 시간에 따라 어떻게 변하는지를 기술하는 도구가 필요합니다. 이 레슨에서는 시간-주파수 분석의 두 가지 주요 프레임워크인 단시간 푸리에 변환(STFT, Short-Time Fourier Transform)과 웨이블릿 변환(Wavelet Transform)을 다룹니다. 두 방법은 시간 및 주파수 국재화(localization)의 균형을 맞추는 방식에서 근본적으로 다른 접근 방식을 취합니다.

**난이도**: ⭐⭐⭐⭐

**선수 학습**: DFT/FFT, 윈도잉(windowing), 컨볼루션(convolution), 기초 선형 대수

**학습 목표**:
- 비정상 신호에 푸리에 변환이 부적합한 이유 설명하기
- STFT를 유도하고 계산하며 시간-주파수 해상도 트레이드오프 이해하기
- 신호에 대한 하이젠베르크 불확정성 원리 기술 및 해석하기
- 스펙트로그램(spectrogram) 계산 및 해석하기
- 위그너-빌 분포(Wigner-Ville Distribution)와 그 성질 이해하기
- 표준 모 웨이블릿(mother wavelet)을 사용하여 연속 웨이블릿 변환(CWT) 정의 및 계산하기
- 다해상도 분석(MRA, Multiresolution Analysis)을 설명하고 이산 웨이블릿 변환(DWT) 구현하기
- 웨이블릿 분해 및 재구성을 위한 말라 알고리즘(Mallat's Algorithm) 적용하기
- 실제 신호 분석에서 STFT와 웨이블릿 접근법 비교하기

---

## 목차

1. [푸리에 변환의 한계](#1-푸리에-변환의-한계)
2. [단시간 푸리에 변환 (STFT)](#2-단시간-푸리에-변환-stft)
3. [시간-주파수 해상도와 불확정성 원리](#3-시간-주파수-해상도와-불확정성-원리)
4. [스펙트로그램](#4-스펙트로그램)
5. [STFT를 위한 윈도우 선택](#5-stft를-위한-윈도우-선택)
6. [위그너-빌 분포](#6-위그너-빌-분포)
7. [웨이블릿 분석 소개](#7-웨이블릿-분석-소개)
8. [연속 웨이블릿 변환 (CWT)](#8-연속-웨이블릿-변환-cwt)
9. [모 웨이블릿](#9-모-웨이블릿)
10. [다해상도 분석 (MRA)](#10-다해상도-분석-mra)
11. [이산 웨이블릿 변환 (DWT)](#11-이산-웨이블릿-변환-dwt)
12. [말라 알고리즘](#12-말라-알고리즘)
13. [웨이블릿 분해 및 재구성](#13-웨이블릿-분해-및-재구성)
14. [STFT vs 웨이블릿 변환](#14-stft-vs-웨이블릿-변환)
15. [응용 사례](#15-응용-사례)
16. [Python 구현](#16-python-구현)
17. [연습 문제](#17-연습-문제)
18. [요약](#18-요약)
19. [참고 문헌](#19-참고-문헌)

---

## 1. 푸리에 변환의 한계

### 1.1 전역 주파수 분석의 문제점

푸리에 변환은 신호 전체 지속 시간에 걸쳐 주파수 내용을 계산합니다:

$$X(f) = \int_{-\infty}^{\infty} x(t) \, e^{-j2\pi ft} \, dt$$

두 신호를 생각해 봅시다:
1. 2초 동안 지속되는 일정한 440 Hz 음
2. 첫 1초 동안 220 Hz, 다음 1초 동안 440 Hz로 재생되는 신호

두 신호는 동일한 푸리에 크기 스펙트럼을 갖지만(같은 주파수를 같은 총 지속 시간 동안 포함), 지각적으로나 물리적으로 매우 다릅니다. 푸리에 변환은 각 주파수가 **언제** 발생하는지에 대한 모든 시간 정보를 버립니다.

### 1.2 비정상 신호(Non-Stationary Signals)

신호가 **정상(stationary)**이라는 것은 통계적 성질이 시간에 따라 변하지 않는 것을 의미합니다. 대부분의 실제 신호는 비정상입니다:

- **음성**: 음소(phoneme)가 빠르게 변화 (10-50 ms 세그먼트)
- **음악**: 음표, 화음, 강세가 지속적으로 변화
- **지진 신호**: P파, S파, 표면파가 서로 다른 주파수 내용으로 다른 시간에 도착
- **생의학**: 심장 리듬, 뇌파, 근육 신호가 생리적 상태에 따라 변화
- **레이더/소나(Radar/Sonar)**: 처프(chirp) 신호는 시간에 따라 변하는 순간 주파수를 가짐

이러한 신호에는 **공동 시간-주파수 표현(joint time-frequency representation)**이 필요합니다.

### 1.3 순간 주파수(Instantaneous Frequency) 개념

$x(t) = A(t)\cos(\phi(t))$ 형태의 신호에서, **순간 주파수**는:

$$f_i(t) = \frac{1}{2\pi}\frac{d\phi(t)}{dt}$$

선형 처프 $x(t) = \cos(2\pi(f_0 t + \frac{1}{2}\beta t^2))$에서 순간 주파수는 $f_i(t) = f_0 + \beta t$입니다. 푸리에 변환은 이 시간 변화 특성을 포착할 수 없습니다.

---

## 2. 단시간 푸리에 변환(STFT, Short-Time Fourier Transform)

### 2.1 정의

STFT는 시간 $\tau$에 중심을 둔 윈도우 함수 $w(t)$를 적용하여 시간 내에서 푸리에 분석을 국재화합니다:

$$\boxed{X_{STFT}(\tau, f) = \int_{-\infty}^{\infty} x(t) \, w(t - \tau) \, e^{-j2\pi ft} \, dt}$$

아이디어는 간단합니다: $\tau$에 중심을 둔 짧은 윈도우를 신호에 곱하여 국소 세그먼트를 추출한 뒤 푸리에 변환을 취합니다. $\tau$를 신호 전체에 걸쳐 슬라이딩하면 일련의 국소 스펙트럼을 얻습니다.

### 2.2 이산 STFT

길이 $L$의 윈도우 $w[n]$을 가진 이산 시간 신호 $x[n]$에 대해:

$$X_{STFT}[m, k] = \sum_{n=0}^{L-1} x[n + mH] \, w[n] \, e^{-j2\pi kn/N}$$

여기서:
- $m$: 시간 프레임 인덱스
- $k$: 주파수 빈(frequency bin) 인덱스
- $H$: 홉 크기(hop size, 연속 윈도우 간의 이동 간격)
- $N$: FFT 크기 ($N > L$이면 영 패딩)

### 2.3 해석

STFT는 1D 신호를 2D 시간-주파수 평면으로 사상합니다. 각 시간 순간 $\tau$에서 전체 스펙트럼을 얻습니다. STFT는 시간 $\tau$와 주파수 $f$의 두 변수 함수입니다.

```
주파수 ▲
        │  ┌──────┐
        │  │      │   (고주파 이벤트)
        │  │      │
        │  └──────┘
        │           ┌──────────────┐
        │           │              │  (중주파 이벤트)
        │           └──────────────┘
        │  ┌──────────┐
        │  │          │               (저주파 이벤트)
        │  └──────────┘
        └──────────────────────────────────▶ 시간
```

### 2.4 오버랩과 홉 크기

윈도우 길이 $L$ 대비 홉 크기 $H$의 일반적인 선택:
- **$H = L$**: 오버랩 없음, 윈도우 경계에서 이벤트를 놓칠 수 있음
- **$H = L/2$**: 50% 오버랩, 좋은 트레이드오프 (가장 일반적)
- **$H = L/4$**: 75% 오버랩, 더 매끄러운 시간 변화, 계산량 증가

완전 재구성(합성 응용에서 중요)을 위해 윈도우와 홉 크기는 **일정 오버랩-추가(COLA, Constant Overlap-Add) 조건**을 만족해야 합니다:

$$\sum_m w[n - mH] = \text{상수} \quad \forall n$$

---

## 3. 시간-주파수 해상도와 불확정성 원리

### 3.1 시간 및 주파수 해상도

STFT의 시간 및 주파수 해상도는 윈도우에 의해 결정됩니다:

- **시간 해상도**: $\Delta t \approx$ $w(t)$의 유효 폭
- **주파수 해상도**: $\Delta f \approx$ $W(f)$ (윈도우의 푸리에 변환)의 유효 대역폭

짧은 윈도우는 좋은 시간 해상도를 주지만 주파수 해상도가 낮습니다. 긴 윈도우는 좋은 주파수 해상도를 주지만 시간 해상도가 낮습니다.

### 3.2 하이젠베르크-가보 불확정성 원리(Heisenberg-Gabor Uncertainty Principle)

임의의 윈도우 함수 $w(t)$에 대해, 시간-대역폭 곱은 다음을 만족합니다:

$$\boxed{\Delta t \cdot \Delta f \geq \frac{1}{4\pi}}$$

여기서 확산은 다음과 같이 정의됩니다:

$$\Delta t^2 = \frac{\int (t - \bar{t})^2 |w(t)|^2 \, dt}{\int |w(t)|^2 \, dt}, \qquad \Delta f^2 = \frac{\int (f - \bar{f})^2 |W(f)|^2 \, df}{\int |W(f)|^2 \, df}$$

이것은 근본적인 한계입니다: **어떤 윈도우도 시간과 주파수 모두에서 동시에 임의의 정밀도를 달성할 수 없습니다.**

**가우시안 윈도우(Gaussian window)** $w(t) = e^{-\alpha t^2}$는 등호를 달성합니다(최소 불확정성). 이 결과로 얻는 STFT를 **가보 변환(Gabor Transform)**이라 합니다.

### 3.3 시간-주파수 평면의 타일

STFT는 시간-주파수 평면을 **고정 크기** $\Delta t \times \Delta f$의 **타일**로 분할합니다:

```
STFT 타일링 (고정 해상도):

주파수 ▲
        │ ┌──┬──┬──┬──┬──┬──┬──┬──┐
        │ │  │  │  │  │  │  │  │  │
        │ ├──┼──┼──┼──┼──┼──┼──┼──┤  ← 모든 곳에서 동일한 Δt × Δf
        │ │  │  │  │  │  │  │  │  │
        │ ├──┼──┼──┼──┼──┼──┼──┼──┤
        │ │  │  │  │  │  │  │  │  │
        │ ├──┼──┼──┼──┼──┼──┼──┼──┤
        │ │  │  │  │  │  │  │  │  │
        │ └──┴──┴──┴──┴──┴──┴──┴──┘
        └──────────────────────────▶ 시간
```

이 고정 타일링은 STFT의 근본적인 한계입니다: 해상도가 모든 곳에서 동일합니다. 많은 신호에서 고주파에서는 더 나은 시간 해상도를, 저주파에서는 더 나은 주파수 해상도를 원합니다 -- 이것이 바로 웨이블릿이 제공하는 것입니다.

---

## 4. 스펙트로그램(Spectrogram)

### 4.1 정의

**스펙트로그램**은 STFT의 크기의 제곱입니다:

$$S(\tau, f) = |X_{STFT}(\tau, f)|^2$$

시간에 국재화된 **전력 스펙트럼 밀도(power spectral density)**를 나타냅니다. 스펙트로그램은 항상 실수이고 비음수입니다.

### 4.2 성질

1. **에너지 보존** (모얄 공식, Moyal's formula):
$$\int\int |X_{STFT}(\tau,f)|^2 \, d\tau \, df = \int |x(t)|^2 \, dt$$

2. **시간 주변(Time marginal)**: $\int S(\tau,f)\,df$는 시간 $\tau$ 주변의 국소 에너지를 제공 (윈도우에 의존)

3. **주파수 주변(Frequency marginal)**: $\int S(\tau,f)\,d\tau$는 에너지 스펙트럼 밀도를 제공 (윈도우에 의해 스미어됨)

### 4.3 평활화된 위그너-빌 분포로서의 스펙트로그램

스펙트로그램은 위그너-빌 분포(6절 참조)의 2D 평활화임을 보일 수 있습니다:

$$S_x(\tau, f) = \iint W_x(t, \nu) \, W_w(\tau - t, f - \nu) \, dt \, d\nu$$

여기서 $W_x$는 $x$의 위그너-빌 분포이고 $W_w$는 윈도우의 위그너-빌 분포입니다. 윈도우는 2D 평활화 커널로 작용하여, 해상도를 희생하는 대신 교차항(cross-term) 간섭을 줄입니다.

---

## 5. STFT를 위한 윈도우 선택

### 5.1 윈도우 성질

윈도우의 선택은 해상도와 스펙트럼 누설(spectral leakage) 모두에 영향을 미칩니다:

| 윈도우 | 주엽 폭(-3dB) | 사이드로브 레벨 | 최적 용도 |
|---|---|---|---|
| 직사각형(Rectangular) | $0.89/L$ | -13 dB | 최대 주파수 해상도 |
| 한(Hann) | $1.44/L$ | -31 dB | 범용 |
| 해밍(Hamming) | $1.33/L$ | -43 dB | 스펙트럼 분석 |
| 블랙만(Blackman) | $1.68/L$ | -58 dB | 낮은 사이드로브 요구 사항 |
| 가우시안(Gaussian) | $\sim 1.2/L$ | $\sigma$에 따라 다름 | 최적 시간-주파수 곱 |
| 카이저(Kaiser) | 조정 가능 | 조정 가능 | 유연한 설계 |

### 5.2 실용 지침

- **음성**: 윈도우 길이 20-40 ms (피치 주기 포착), 해밍 윈도우, 50% 오버랩
- **음악**: 윈도우 길이 50-100 ms (음악 음표에 대한 더 나은 주파수 해상도), 한 윈도우
- **과도 신호 검출**: 더 나은 시간 해상도를 위한 짧은 윈도우 (1-5 ms)
- **일반 규칙**: 해상하려는 특징의 시간 척도에 맞게 윈도우 길이 선택

### 5.3 영 패딩(Zero-Padding)

FFT 이전에 윈도우 처리된 세그먼트를 영 패딩하는 것($N > L$ 사용)은 주파수 해상도를 **향상시키지 않지만**, 주파수 영역에서 보간을 제공하여 스펙트로그램이 더 매끄럽게 보입니다. 실제 주파수 해상도는 항상 윈도우 길이 $L$에 의해 결정됩니다.

---

## 6. 위그너-빌 분포(Wigner-Ville Distribution)

### 6.1 정의

**위그너-빌 분포(WVD)**는 이차(quadratic) 시간-주파수 분포입니다:

$$W_x(t, f) = \int_{-\infty}^{\infty} x\!\left(t + \frac{\tau}{2}\right) x^*\!\left(t - \frac{\tau}{2}\right) e^{-j2\pi f\tau} \, d\tau$$

### 6.2 성질

1. **우수한 해상도**: WVD는 가능한 최상의 시간-주파수 집중도를 달성
2. **올바른 주변 분포**:
   - $\int W_x(t,f)\,df = |x(t)|^2$ (순간 전력)
   - $\int W_x(t,f)\,dt = |X(f)|^2$ (전력 스펙트럼 밀도)
3. **순간 주파수**: 단성분(mono-component) 신호에서 주파수의 일차 모멘트가 순간 주파수를 제공

### 6.3 교차항(Cross-Term) 문제

WVD는 **이선형(bilinear)** (이차) 표현입니다. 두 성분 $x = x_1 + x_2$를 가진 신호에 대해:

$$W_x = W_{x_1} + W_{x_2} + 2\Re\{W_{x_1,x_2}\}$$

교차항 $W_{x_1,x_2}$는 실제 신호 성분들 사이에 **진동하는 간섭 패턴**을 만듭니다. $N$개 성분에 대해 $N(N-1)/2$개의 교차항이 있어, 실제 신호 내용을 압도할 수 있습니다.

### 6.4 코헨 클래스(Cohen's Class)

**코헨 클래스**는 올바른 주변 분포 성질을 만족하는 시간-주파수 표현의 일반적인 프레임워크를 제공합니다:

$$C_x(t, f) = \iiint e^{-j2\pi(\theta\tau + f\nu - \theta u)} \Phi(\theta, \tau) x\!\left(u+\frac{\tau}{2}\right) x^*\!\left(u-\frac{\tau}{2}\right) du \, d\tau \, d\theta$$

여기서 $\Phi(\theta, \tau)$는 **커널 함수**입니다. 서로 다른 커널은 서로 다른 분포를 제공합니다:
- $\Phi = 1$: 위그너-빌 분포
- $\Phi = $ 가우시안: 평활화된 의사 위그너-빌 (교차항 감소)
- $\Phi = e^{-\sigma^2 \theta^2 \tau^2}$: 최-윌리엄스 분포(Choi-Williams Distribution)

---

## 7. 웨이블릿 분석 소개

### 7.1 동기: 가변 해상도

STFT는 고정 윈도우를 사용하여 모든 주파수에서 동일한 해상도를 제공합니다. 하지만 많은 신호는:
- **시간이 짧은 고주파 이벤트** (과도 신호, 클릭, 에지)
- **시간이 긴 저주파 이벤트** (추세, 기본 음)

를 가집니다.

우리가 원하는 것:
- **고주파**에서의 **짧은 윈도우** (좋은 시간 해상도)
- **저주파**에서의 **긴 윈도우** (좋은 주파수 해상도)

이것이 바로 웨이블릿 변환이 제공하는 것입니다.

### 7.2 STFT에서 웨이블릿으로

STFT에서는 고정 윈도우를 서로 다른 주파수로 변조합니다:

$$\text{STFT}: \quad g_{\tau,f}(t) = w(t-\tau) \, e^{j2\pi ft}$$

웨이블릿 변환에서는 하나의 원형 함수(모 웨이블릿)를 **스케일링**하고 **이동**합니다:

$$\text{CWT}: \quad \psi_{a,b}(t) = \frac{1}{\sqrt{|a|}} \psi\!\left(\frac{t-b}{a}\right)$$

- **이동(Translation)** $b$: 시간에 국재화
- **스케일(Scale)** $a$: 시간-주파수 해상도 제어
  - 작은 $a$: 압축된 웨이블릿 (시간이 짧고, 고주파 포착)
  - 큰 $a$: 늘어난 웨이블릿 (시간이 길고, 저주파 포착)

### 7.3 웨이블릿 타일링

```
웨이블릿 타일링 (가변 해상도):

주파수 ▲
        │ ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐
        │ ├┴┬┴┬┴┬┴┬┴┬┴┬┴┬┴┬┴┬┴┬┴┬┴┬┘  ← 고주파: 좁은 Δt, 넓은 Δf
        │ ├──┬──┬──┬──┬──┬──┬──┬──┤
        │ │  │  │  │  │  │  │  │  │     ← 중주파
        │ ├────┬────┬────┬────┤
        │ │    │    │    │    │          ← 저주파: 넓은 Δt, 좁은 Δf
        │ ├────────┬────────┤
        │ │        │        │
        │ └────────┴────────┘
        └──────────────────────────▶ 시간
```

각 타일은 동일한 면적 $\Delta t \cdot \Delta f = \text{상수}$를 가지지만, 종횡비가 다릅니다: 고주파에서는 키가 크고 좁으며, 저주파에서는 낮고 넓습니다.

---

## 8. 연속 웨이블릿 변환(CWT, Continuous Wavelet Transform)

### 8.1 정의

모 웨이블릿(mother wavelet) $\psi(t)$를 사용한 $x(t)$의 **연속 웨이블릿 변환**은:

$$\boxed{W_x(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \, \psi^*\!\left(\frac{t-b}{a}\right) dt}$$

여기서:
- $a > 0$: **스케일(scale)** 매개변수 (주파수에 반비례)
- $b$: **이동(translation)** (시간 이동) 매개변수
- $\psi^*$: 모 웨이블릿의 복소 켤레
- $1/\sqrt{|a|}$: 스케일에 걸친 에너지 정규화

### 8.2 스케일과 주파수 사이의 관계

중심 주파수 $f_c$를 가진 모 웨이블릿에서, 스케일 $a$에서의 의사 주파수(pseudo-frequency)는:

$$f_a = \frac{f_c}{a}$$

따라서 **큰 스케일**은 **낮은 주파수**에, **작은 스케일**은 **높은 주파수**에 대응합니다.

### 8.3 허용 가능성 조건(Admissibility Condition)

CWT가 가역(invertible)이 되려면, 모 웨이블릿은 **허용 가능성 조건**을 만족해야 합니다:

$$C_\psi = \int_0^{\infty} \frac{|\hat{\Psi}(f)|^2}{f} \, df < \infty$$

여기서 $\hat{\Psi}(f)$는 $\psi(t)$의 푸리에 변환입니다. 이는 $\hat{\Psi}(0) = 0$을 요구하여, 웨이블릿이 **영 평균(zero mean)**을 가져야 함을 의미합니다:

$$\int_{-\infty}^{\infty} \psi(t) \, dt = 0$$

### 8.4 역 CWT

신호는 CWT로부터 재구성할 수 있습니다:

$$x(t) = \frac{1}{C_\psi} \int_0^{\infty} \int_{-\infty}^{\infty} W_x(a,b) \, \frac{1}{\sqrt{a}} \psi\!\left(\frac{t-b}{a}\right) \frac{db \, da}{a^2}$$

### 8.5 스칼로그램(Scalogram)

**스칼로그램**은 스펙트로그램의 웨이블릿 유사체입니다:

$$\text{스칼로그램}(a, b) = |W_x(a, b)|^2$$

---

## 9. 모 웨이블릿(Mother Wavelets)

### 9.1 하르 웨이블릿(Haar Wavelet)

1909년 알프레드 하르(Alfred Haar)가 제안한 가장 단순한 웨이블릿:

$$\psi_{Haar}(t) = \begin{cases} 1 & 0 \leq t < 1/2 \\ -1 & 1/2 \leq t < 1 \\ 0 & \text{그 외} \end{cases}$$

**성질**:
- 유한 지지(compact support, 유한 지속 시간)
- 불연속 (낮은 주파수 국재화)
- 1개의 소실 모멘트(vanishing moment): $\int t^0 \psi(t)\,dt = 0$
- 날카로운 전환 및 에지 검출에 적합

### 9.2 모를레 웨이블릿(Morlet Wavelet)

가우시안으로 변조된 복소 지수로 이루어진 복소 웨이블릿:

$$\psi_{Morlet}(t) = C_\sigma \, \pi^{-1/4} \, e^{j\omega_0 t} \, e^{-t^2/2}$$

여기서 $\omega_0$ (일반적으로 $\omega_0 \approx 5$ 또는 $2\pi$)는 중심 주파수이고 $C_\sigma$는 정규화 상수입니다.

**성질**:
- 복소값 (진폭과 위상 모두 제공)
- 무한 지지 (가우시안 감쇠로 사실상 유한)
- 우수한 시간-주파수 국재화 (가우시안 포락선)
- 유한 지지 없음, 직교성 없음
- 연속 웨이블릿 분석에 가장 많이 사용

### 9.3 멕시칸 햇 웨이블릿(Mexican Hat Wavelet, 리커 웨이블릿)

가우시안의 음의 정규화된 2차 도함수:

$$\psi_{mhat}(t) = \frac{2}{\sqrt{3}\pi^{1/4}} (1 - t^2) \, e^{-t^2/2}$$

**성질**:
- 실수값, 대칭
- 2개의 소실 모멘트: $k = 0, 1$에 대해 $\int t^k \psi(t)\,dt = 0$
- 영상 처리에 사용되는 가우시안의 라플라시안(LoG)과 관련
- 피크와 골짜기 검출에 적합

### 9.4 도비쉐 웨이블릿(Daubechies Wavelets)

잉그리드 도비쉐(Ingrid Daubechies, 1988)는 $N$개의 소실 모멘트와 길이 $2N - 1$의 유한 지지를 가진 웨이블릿 패밀리 $\psi_{Db-N}$을 구성했습니다:

- **Db1** = 하르 웨이블릿
- **Db2**: 지지 길이 3, 2개의 소실 모멘트
- **Db4**: 지지 길이 7, 4개의 소실 모멘트 (일반적으로 사용)
- **Db10**: 지지 길이 19, 10개의 소실 모멘트

**성질**:
- 직교(orthogonal)이고 유한 지지
- 폐쇄형 표현식 없음 (필터 계수로 정의)
- 더 높은 $N$: 더 매끄러운 웨이블릿, 더 긴 지지, 더 나은 주파수 국재화
- 유한 지지와 최대 소실 모멘트를 가진 유일한 직교 웨이블릿

### 9.5 기타 중요 웨이블릿

| 웨이블릿 | 유형 | 주요 특징 |
|---|---|---|
| Symlets (sym$N$) | 직교 | 도비쉐의 근사 대칭 버전 |
| Coiflets (coif$N$) | 직교 | 근사 대칭, $\psi$와 $\phi$ 모두에서 소실 모멘트 |
| Meyer | 직교 | 주파수 영역에서 정의, 무한히 매끄러움 |
| Shannon | 직교 | 이상적 대역 통과, sinc 기반 |
| Gabor | 비직교 | 가우시안 변조 사인파 |
| Paul | 복소 | 해석적, 좋은 시간 국재화 |

---

## 10. 다해상도 분석(MRA, Multiresolution Analysis)

### 10.1 개념

말라(Mallat, 1989)와 마이어(Meyer)가 도입한 다해상도 분석은 DWT의 수학적 프레임워크를 제공합니다. 아이디어는 신호를 서로 다른 해상도의 연속적인 **근사(approximations)**로 분해하고, 각 수준에서 **세부(detail)**를 포착하는 것입니다.

풍경을 바라보는 것을 생각해 보세요:
- **거친 해상도**: 산과 계곡을 봄
- **중간 해상도**: 개별 나무와 건물을 봄
- **세밀한 해상도**: 잎사귀와 벽돌을 봄

### 10.2 중첩된 근사 공간(Nested Approximation Spaces)

MRA는 $L^2(\mathbb{R})$의 닫힌 부분 공간의 중첩된 수열로 구성됩니다:

$$\cdots \subset V_{-2} \subset V_{-1} \subset V_0 \subset V_1 \subset V_2 \subset \cdots$$

다음 조건을 만족합니다:
1. **중첩**: $V_j \subset V_{j+1}$
2. **조밀도**: $\overline{\bigcup_j V_j} = L^2(\mathbb{R})$
3. **분리**: $\bigcap_j V_j = \{0\}$
4. **스케일링**: $f(t) \in V_j \Leftrightarrow f(2t) \in V_{j+1}$
5. **이동 불변성**: 모든 $k \in \mathbb{Z}$에 대해 $f(t) \in V_0 \Leftrightarrow f(t-k) \in V_0$
6. **리즈 기저(Riesz basis)**: $\{\phi(t-k)\}_{k \in \mathbb{Z}}$가 $V_0$의 리즈 기저를 형성하는 **스케일링 함수(scaling function)** $\phi(t)$가 존재

### 10.3 스케일링 함수와 웨이블릿 함수

**스케일링 함수(부 웨이블릿, father wavelet)** $\phi(t)$는 **세분화 방정식(refinement equation, 팽창 방정식)**을 만족합니다:

$$\phi(t) = \sqrt{2} \sum_k h[k] \, \phi(2t - k)$$

여기서 $h[k]$는 **스케일링 계수(scaling coefficients)** (저역 통과 필터)입니다.

**웨이블릿 함수** $\psi(t)$는 다음으로 정의됩니다:

$$\psi(t) = \sqrt{2} \sum_k g[k] \, \phi(2t - k)$$

여기서 $g[k] = (-1)^k h[1-k]$는 **웨이블릿 계수(wavelet coefficients)** (고역 통과 필터, 직교 거울 관계로 $h$와 관련)입니다.

### 10.4 세부 공간(Detail Spaces)

**세부 공간** $W_j$는 $V_{j+1}$에서 $V_j$의 직교 여공간입니다:

$$V_{j+1} = V_j \oplus W_j$$

스케일 $j$에서의 웨이블릿들이 $W_j$를 생성합니다:

$$W_j = \text{span}\{\psi_{j,k}(t) = 2^{j/2}\psi(2^j t - k)\}_{k \in \mathbb{Z}}$$

신호 분해:

$$V_J = V_0 \oplus W_0 \oplus W_1 \oplus \cdots \oplus W_{J-1}$$

$$f(t) = \sum_k c_{0,k}\phi_{0,k}(t) + \sum_{j=0}^{J-1}\sum_k d_{j,k}\psi_{j,k}(t)$$

여기서 $c_{j,k}$는 **근사 계수(approximation coefficients)**, $d_{j,k}$는 **세부 계수(detail coefficients)**입니다.

---

## 11. 이산 웨이블릿 변환(DWT, Discrete Wavelet Transform)

### 11.1 이진 샘플링(Dyadic Sampling)

CWT는 매우 중복적입니다: 모든 스케일 $a$와 이동 $b$에서 계수를 계산합니다. DWT는 **이진 격자(dyadic grid)**에서 샘플링합니다:

$$a = 2^j, \quad b = k \cdot 2^j$$

다음을 얻습니다:

$$W_x(j, k) = 2^{-j/2} \int x(t) \, \psi(2^{-j}t - k) \, dt$$

### 11.2 필터 뱅크와의 연결

DWT는 한 쌍의 필터를 사용하여 효율적으로 계산할 수 있습니다:
- **저역 통과 필터** $h[n]$: 스케일링 함수 $\phi$와 연관
- **고역 통과 필터** $g[n]$: 웨이블릿 $\psi$와 연관

이들은 **직교 거울 필터(QMF, Quadrature Mirror Filter)** 뱅크를 형성합니다:

$$g[n] = (-1)^n h[1-n]$$

**완전 재구성** 조건:
$$H(z)H(z^{-1}) + H(-z)H(-z^{-1}) = 2$$

### 11.3 분석 및 합성

**분석** (분해):
- 근사 계수: $c_{j+1}[k] = \sum_n h[n-2k] \, c_j[n]$ (저역 통과 필터 + 2로 다운샘플)
- 세부 계수: $d_{j+1}[k] = \sum_n g[n-2k] \, c_j[n]$ (고역 통과 필터 + 2로 다운샘플)

**합성** (재구성):
- $c_j[n] = \sum_k c_{j+1}[k] \, h[n-2k] + \sum_k d_{j+1}[k] \, g[n-2k]$ (2로 업샘플 + 필터 + 합산)

---

## 12. 말라 알고리즘(Mallat's Algorithm)

### 12.1 고속 웨이블릿 변환(Fast Wavelet Transform)

말라 알고리즘은 FFT가 DFT를 계산하는 방식과 유사하게, 반복된 필터 뱅크를 사용하여 DWT를 계산합니다. 계산 비용은 $O(N)$ -- FFT보다 더 빠릅니다!

```
말라 알고리즘 (분석/분해):

레벨 0:  c₀[n] = x[n]  (원본 신호, N 샘플)
              │
              ├──── h[n] ──▶ ↓2 ──▶ c₁[k]  (근사, N/2 샘플)
              │                        │
              │                        ├──── h[n] ──▶ ↓2 ──▶ c₂[k]  (N/4)
              │                        │                        │
              │                        │                        ├──── h[n]──▶↓2──▶ c₃
              │                        │                        │
              │                        │                        └──── g[n]──▶↓2──▶ d₃
              │                        │
              │                        └──── g[n] ──▶ ↓2 ──▶ d₂[k]  (세부, N/4)
              │
              └──── g[n] ──▶ ↓2 ──▶ d₁[k]  (세부, N/2 샘플)
```

### 12.2 재구성 (합성)

```
말라 알고리즘 (합성/재구성):

c₃ ──▶ ↑2 ──▶ h̃[n] ──┐
                         ├──▶ + ──▶ c₂ ──▶ ↑2 ──▶ h̃[n] ──┐
d₃ ──▶ ↑2 ──▶ g̃[n] ──┘                                     ├──▶ + ──▶ c₁ ──▶ ...
                                   d₂ ──▶ ↑2 ──▶ g̃[n] ──┘
```

### 12.3 계산 복잡도

길이 $N$의 신호와 $J$ 분해 레벨에 대해:
- 각 레벨: $O(N/2^j)$ 연산 (필터 + 다운샘플)
- 합계: $O(N) + O(N/2) + O(N/4) + \cdots = O(2N) = O(N)$

비교:
- FFT: $O(N \log N)$
- STFT: $O(N \cdot L \cdot \log L)$ (여기서 $L$은 윈도우 길이)

---

## 13. 웨이블릿 분해 및 재구성

### 13.1 다단계 분해(Multi-Level Decomposition)

분해 레벨 $J$에서 신호는 다음과 같이 표현됩니다:

$$x[n] = \underbrace{c_J[k]}_{\text{거친 근사}} + \underbrace{d_J[k]}_{\text{가장 세밀한 세부}} + d_{J-1}[k] + \cdots + d_1[k]$$

각 레벨의 세부 계수는 특정 주파수 대역을 포착합니다:
- $d_1$: 최고 주파수 ($f_s/4$에서 $f_s/2$)
- $d_2$: 다음 대역 ($f_s/8$에서 $f_s/4$)
- $d_j$: 대역 ($f_s/2^{j+1}$에서 $f_s/2^j$)
- $c_J$: 최저 주파수 (0에서 $f_s/2^{J+1}$)

### 13.2 웨이블릿 잡음 제거(Wavelet Denoising, 임계값 처리)

DWT의 가장 강력한 응용 중 하나는 **임계값 처리(thresholding)를 통한 잡음 제거**입니다:

1. 잡음 신호의 DWT 계산
2. 세부 계수에 임계값 적용
3. 역 DWT를 사용하여 재구성

**경(hard) 임계값 처리**:
$$\hat{d}[k] = \begin{cases} d[k] & |d[k]| \geq \lambda \\ 0 & |d[k]| < \lambda \end{cases}$$

**연(soft) 임계값 처리** (수축):
$$\hat{d}[k] = \text{sgn}(d[k]) \max(|d[k]| - \lambda, 0)$$

**범용 임계값(Universal threshold)** (도노호와 존스톤, 1994):
$$\lambda = \sigma \sqrt{2 \ln N}$$

여기서 $\sigma$는 잡음 표준 편차로, 가장 세밀한 세부 레벨에서 추정합니다:
$$\hat{\sigma} = \frac{\text{median}(|d_1|)}{0.6745}$$

### 13.3 레벨 수 선택

길이 $N$의 신호와 필터 길이 $L$에 대한 최대 분해 레벨 수:

$$J_{max} = \lfloor \log_2(N / (L-1)) \rfloor$$

실제로는 대부분의 응용에서 3-6 레벨로 충분합니다.

---

## 14. STFT vs 웨이블릿 변환

### 14.1 해상도 비교

| 특징 | STFT | 웨이블릿 변환 |
|---|---|---|
| 시간-주파수 타일링 | 균일한 직사각형 | 가변적 (이진) |
| 저주파 해상도 | 고주파와 동일 | 더 나은 주파수 해상도 |
| 고주파 해상도 | 저주파와 동일 | 더 나은 시간 해상도 |
| 기저 함수 | 변조된 윈도우 | 스케일/이동된 웨이블릿 |
| 계산 (이산) | 프레임당 $O(N \log N)$ | 합계 $O(N)$ |
| 중복도 (연속) | 높음 (오버랩에 의존) | 높음 (CWT) 또는 임계 (DWT) |
| 위상 정보 | 예 (복소 STFT) | 웨이블릿에 따라 다름 |
| 가역성 | 예 (COLA 조건) | 예 (허용 가능성) |

### 14.2 언제 무엇을 사용할 것인가

**STFT/스펙트로그램 사용 시:**
- 신호가 비교적 균일한 시간-주파수 특성을 가짐
- 모든 주파수에서 일정한 주파수 해상도가 필요
- 음악 음조 분석 (동일한 주파수 빈이 음악적 지각과 일치)
- 신호가 사인파 성분으로 잘 기술됨

**웨이블릿 변환 사용 시:**
- 신호에 과도 및 천천히 변하는 성분이 모두 있음
- 서로 다른 주파수 대역에 서로 다른 시간 해상도가 필요
- 특이점이나 날카로운 전환 검출이 필요
- 잡음 제거가 주요 목표
- 다중 스케일 분석이 필요 (프랙탈, 난류)

### 14.3 처프 신호에 대한 비교

선형 처프 $x(t) = \cos(2\pi(f_0 t + \frac{\beta}{2}t^2))$에 대해:

- **STFT**: 시간-주파수 평면에서 직선을 보여주지만, 선의 폭이 일정 (윈도우에 의해 결정)
- **웨이블릿**: 곡선을 보여주지만 (스케일 $\neq$ 주파수 선형적으로), 고주파에서 더 나은 시간 해상도, 저주파에서 더 나은 주파수 해상도

---

## 15. 응용 사례

### 15.1 음악 및 오디오 분석

- **스펙트로그램**: 멜로디, 하모닉스, 리듬을 보여주는 음악 정보 검색의 기본 도구
- **상수-Q 변환(CQT, Constant-Q Transform)**: 음악적 음계에 맞는 대수적 주파수 간격을 가진 웨이블릿 유사 변환
- **온셋 검출(Onset Detection)**: 웨이블릿 분석이 좋은 시간 정밀도로 음표 온셋(과도 신호)을 검출
- **음조 추적**: 하모닉 분석을 사용한 STFT

### 15.2 생의학 신호 처리

- **ECG 분석**: 웨이블릿 분해로 QRS 복합체(고주파)를 기저선 표류(저주파)에서 분리
- **EEG 스펙트럼 분석**: STFT가 시간에 따른 알파(8-12 Hz), 베타(12-30 Hz), 감마(>30 Hz) 리듬을 드러냄
- **EMG 처리**: 근육 활성화 패턴의 시간-주파수 분석

### 15.3 진동 분석 및 결함 검출

- **베어링 결함 검출**: 웨이블릿 분해가 잡음 속에 숨겨진 주기적 충격 신호를 드러냄
- **회전 기계**: 시간-주파수 분석을 사용한 차수 추적
- **구조 건강 모니터링**: 다리와 건물에서 웨이블릿 기반 손상 검출

### 15.4 지구물리학 및 지진학

- **지진 이벤트 검출**: 다중 스케일 웨이블릿 분석이 로컬 및 원격지진 이벤트를 분리
- **분산 분석**: 웨이블릿 변환을 사용한 군속도 측정
- **시간-주파수 필터링**: 간섭의 특정 시간-주파수 영역 제거

### 15.5 영상 처리

- **웨이블릿 압축**: JPEG2000은 DWT 사용 (도비쉐 9/7 또는 Le Gall 5/3 웨이블릿)
- **에지 검출**: 웨이블릿 최대값이 다른 스케일에서 에지에 대응
- **질감 분석**: 서로 다른 부대역(sub-band)의 웨이블릿 에너지가 질감을 특성화

---

## 16. Python 구현

### 16.1 STFT 및 스펙트로그램

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig


def compute_stft(x, fs, window_length, hop_size, nfft=None, window='hann'):
    """
    Compute the Short-Time Fourier Transform.

    Parameters
    ----------
    x : ndarray
        Input signal
    fs : float
        Sampling frequency
    window_length : int
        Window length in samples
    hop_size : int
        Hop size in samples
    nfft : int
        FFT size (zero-padded if > window_length)
    window : str
        Window type

    Returns
    -------
    t : ndarray
        Time axis
    f : ndarray
        Frequency axis
    Zxx : ndarray
        STFT matrix (frequency x time)
    """
    if nfft is None:
        nfft = window_length

    # Create window
    win = sig.get_window(window, window_length)

    # Number of frames
    n_frames = 1 + (len(x) - window_length) // hop_size

    # STFT matrix
    Zxx = np.zeros((nfft // 2 + 1, n_frames), dtype=complex)

    for m in range(n_frames):
        start = m * hop_size
        segment = x[start:start + window_length] * win
        spectrum = np.fft.rfft(segment, n=nfft)
        Zxx[:, m] = spectrum

    # Time and frequency axes
    t = np.arange(n_frames) * hop_size / fs
    f = np.arange(nfft // 2 + 1) * fs / nfft

    return t, f, Zxx


# Generate test signal: chirp + impulse
fs = 1000  # Sampling frequency
duration = 2.0
t = np.arange(0, duration, 1/fs)
N = len(t)

# Linear chirp from 50 Hz to 400 Hz
chirp = sig.chirp(t, f0=50, f1=400, t1=duration, method='linear')

# Add impulses at specific times
impulse_signal = np.zeros(N)
for t_imp in [0.3, 0.8, 1.5]:
    idx = int(t_imp * fs)
    impulse_signal[idx:idx+10] = 1.0

# Combined signal
x = chirp + 0.5 * impulse_signal + 0.1 * np.random.randn(N)

# Compute spectrograms with different window lengths
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time-domain signal
axes[0, 0].plot(t, x, 'b', alpha=0.7)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].set_title('Signal: Chirp + Impulses')
axes[0, 0].grid(True, alpha=0.3)

# Short window (good time resolution)
window_lengths = [32, 128, 512]
titles = ['Short Window (32 samples)', 'Medium Window (128 samples)',
          'Long Window (512 samples)']

for idx, (wl, title) in enumerate(zip(window_lengths, titles)):
    ax = axes[(idx + 1) // 2, (idx + 1) % 2]
    f_stft, t_stft, Sxx = sig.spectrogram(
        x, fs, window='hann', nperseg=wl, noverlap=wl//2, nfft=max(wl, 512)
    )
    ax.pcolormesh(t_stft, f_stft, 10*np.log10(Sxx + 1e-10),
                  shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Spectrogram: {title}')
    ax.set_ylim([0, 500])

plt.tight_layout()
plt.savefig('stft_resolution_tradeoff.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 16.2 PyWavelets를 사용한 웨이블릿 분석

```python
import pywt
import numpy as np
import matplotlib.pyplot as plt


# Generate a non-stationary signal
fs = 1024
t = np.arange(0, 2, 1/fs)
N = len(t)

# Signal with time-varying frequency content
x = np.zeros(N)
x[:N//4] = np.sin(2 * np.pi * 10 * t[:N//4])           # 10 Hz
x[N//4:N//2] = np.sin(2 * np.pi * 50 * t[N//4:N//2])   # 50 Hz
x[N//2:3*N//4] = np.sin(2 * np.pi * 100 * t[N//2:3*N//4])  # 100 Hz
x[3*N//4:] = np.sin(2 * np.pi * 200 * t[3*N//4:])      # 200 Hz

# Add some noise
x += 0.3 * np.random.randn(N)

# --- Continuous Wavelet Transform (CWT) ---
scales = np.arange(1, 128)
wavelet = 'cmor1.5-1.0'  # Complex Morlet wavelet
coefficients, frequencies = pywt.cwt(x, scales, wavelet, sampling_period=1/fs)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Time domain
axes[0].plot(t, x, 'b', alpha=0.7)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Non-stationary Signal (10→50→100→200 Hz)')
axes[0].grid(True, alpha=0.3)

# CWT scalogram
im = axes[1].pcolormesh(t, frequencies, np.abs(coefficients),
                         shading='gouraud', cmap='jet')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('CWT Scalogram (Morlet Wavelet)')
axes[1].set_ylim([0, 300])
plt.colorbar(im, ax=axes[1], label='|CWT|')

# Compare with STFT spectrogram
from scipy import signal as sig
f_stft, t_stft, Sxx = sig.spectrogram(x, fs, nperseg=128, noverlap=96)
axes[2].pcolormesh(t_stft, f_stft, 10*np.log10(Sxx + 1e-10),
                   shading='gouraud', cmap='jet')
axes[2].set_ylabel('Frequency (Hz)')
axes[2].set_xlabel('Time (s)')
axes[2].set_title('STFT Spectrogram (128-sample Hann window)')
axes[2].set_ylim([0, 300])

plt.tight_layout()
plt.savefig('cwt_vs_stft.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 16.3 이산 웨이블릿 변환(DWT) 및 잡음 제거

```python
import pywt
import numpy as np
import matplotlib.pyplot as plt


# Create a clean signal
fs = 1000
t = np.arange(0, 1, 1/fs)
clean = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*50*t)

# Add noise
np.random.seed(42)
noise_level = 0.8
noisy = clean + noise_level * np.random.randn(len(t))

# DWT decomposition
wavelet = 'db4'
level = 5
coeffs = pywt.wavedec(noisy, wavelet, level=level)

# Estimate noise level from finest detail coefficients
sigma = np.median(np.abs(coeffs[-1])) / 0.6745
print(f"Estimated noise std: {sigma:.3f} (true: {noise_level:.3f})")

# Apply soft thresholding
threshold = sigma * np.sqrt(2 * np.log(len(noisy)))  # Universal threshold
print(f"Threshold: {threshold:.3f}")

coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
for i in range(1, len(coeffs)):
    coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))

# Reconstruct
denoised = pywt.waverec(coeffs_thresh, wavelet)
denoised = denoised[:len(t)]  # Trim to original length

# Plot results
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

axes[0].plot(t, clean, 'g', linewidth=2)
axes[0].set_title('Clean Signal')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, noisy, 'r', alpha=0.7)
axes[1].set_title(f'Noisy Signal (SNR = {10*np.log10(np.var(clean)/noise_level**2):.1f} dB)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, denoised, 'b', linewidth=1.5)
axes[2].set_title('Denoised Signal (Wavelet Soft Thresholding, db4)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(t, clean, 'g--', alpha=0.5, label='Clean')
axes[3].plot(t, denoised, 'b', alpha=0.7, label='Denoised')
axes[3].set_title('Comparison: Clean vs Denoised')
axes[3].set_xlabel('Time (s)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wavelet_denoising.png', dpi=150, bbox_inches='tight')
plt.show()

# Compute SNR improvement
snr_noisy = 10 * np.log10(np.sum(clean**2) / np.sum((noisy - clean)**2))
snr_denoised = 10 * np.log10(np.sum(clean**2) / np.sum((denoised - clean)**2))
print(f"\nSNR (noisy):    {snr_noisy:.1f} dB")
print(f"SNR (denoised): {snr_denoised:.1f} dB")
print(f"SNR improvement: {snr_denoised - snr_noisy:.1f} dB")
```

### 16.4 DWT 분해 시각화

```python
import pywt
import numpy as np
import matplotlib.pyplot as plt


# Generate signal with multi-scale features
fs = 1024
t = np.arange(0, 2, 1/fs)
N = len(t)

# Low frequency trend + medium frequency oscillation + high frequency bursts
x = (0.5 * np.sin(2*np.pi*2*t)                      # 2 Hz trend
     + np.sin(2*np.pi*30*t)                           # 30 Hz oscillation
     + 0.3 * np.sin(2*np.pi*150*t) * (t > 0.5) * (t < 1.0)  # 150 Hz burst
     + 0.2 * np.random.randn(N))                      # noise

# Multi-level DWT decomposition
wavelet = 'db4'
level = 6
coeffs = pywt.wavedec(x, wavelet, level=level)

# Reconstruct individual components
details = []
for i in range(1, level + 1):
    # Zero out all coefficients except the i-th detail
    c_temp = [np.zeros_like(c) for c in coeffs]
    c_temp[i] = coeffs[i].copy()
    detail_i = pywt.waverec(c_temp, wavelet)[:N]
    details.append(detail_i)

# Approximation
c_approx = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
approx = pywt.waverec(c_approx, wavelet)[:N]

# Plot decomposition
fig, axes = plt.subplots(level + 2, 1, figsize=(14, 16), sharex=True)

axes[0].plot(t, x, 'k', alpha=0.7)
axes[0].set_ylabel('Signal')
axes[0].set_title(f'DWT Decomposition ({wavelet}, {level} levels)')

axes[1].plot(t, approx, 'b', alpha=0.7)
axes[1].set_ylabel(f'A{level}')
freq_band = f'0-{fs/2**(level+1):.0f} Hz'
axes[1].annotate(freq_band, xy=(0.98, 0.8), xycoords='axes fraction',
                 ha='right', fontsize=9, color='gray')

for i, detail in enumerate(details):
    ax = axes[i + 2]
    ax.plot(t, detail, 'r' if i == 0 else 'orange' if i < 3 else 'g', alpha=0.7)
    level_idx = level - i
    ax.set_ylabel(f'D{level_idx}')
    low_f = fs / 2**(level_idx + 1)
    high_f = fs / 2**level_idx
    freq_band = f'{low_f:.0f}-{high_f:.0f} Hz'
    ax.annotate(freq_band, xy=(0.98, 0.8), xycoords='axes fraction',
                ha='right', fontsize=9, color='gray')

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('dwt_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 16.5 수동 CWT 구현

```python
import numpy as np
import matplotlib.pyplot as plt


def morlet_wavelet(t, omega0=5.0):
    """Morlet wavelet (simplified, without correction term)."""
    return np.pi**(-0.25) * np.exp(1j * omega0 * t) * np.exp(-t**2 / 2)


def manual_cwt(x, scales, dt=1.0, omega0=5.0):
    """
    Manual CWT implementation using convolution.

    Parameters
    ----------
    x : ndarray
        Input signal
    scales : ndarray
        Array of scales
    dt : float
        Sampling period
    omega0 : float
        Center frequency of Morlet wavelet

    Returns
    -------
    W : ndarray
        CWT coefficient matrix (n_scales x N)
    freqs : ndarray
        Pseudo-frequencies corresponding to each scale
    """
    N = len(x)
    n_scales = len(scales)
    W = np.zeros((n_scales, N), dtype=complex)

    for i, scale in enumerate(scales):
        # Create wavelet at this scale
        # Wavelet support: we need enough points to capture the wavelet
        M = min(10 * int(scale / dt), N)
        t_wav = np.arange(-M, M + 1) * dt
        wavelet = (1.0 / np.sqrt(scale)) * morlet_wavelet(t_wav / scale, omega0)
        wavelet = np.conj(wavelet)  # Conjugate for cross-correlation

        # Convolve (cross-correlate)
        conv_result = np.convolve(x, wavelet, mode='same') * dt
        W[i, :] = conv_result

    # Pseudo-frequencies
    freqs = omega0 / (2 * np.pi * scales * dt)

    return W, freqs


# Test with a chirp signal
fs = 500
t = np.arange(0, 3, 1/fs)
# Chirp from 5 Hz to 100 Hz
x = np.cos(2 * np.pi * (5*t + 47.5*t**2/3))

scales = np.arange(1, 100)
W, freqs = manual_cwt(x, scales, dt=1/fs)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(t, x, 'b', alpha=0.7)
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Chirp Signal (5 Hz to 100 Hz)')
axes[0].grid(True, alpha=0.3)

im = axes[1].pcolormesh(t, freqs, np.abs(W), shading='gouraud', cmap='magma')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('Manual CWT Scalogram (Morlet)')
axes[1].set_ylim([0, 120])
plt.colorbar(im, ax=axes[1], label='|W(a,b)|')

plt.tight_layout()
plt.savefig('manual_cwt.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 17. 연습 문제

### 연습 문제 1: STFT 해상도 트레이드오프

100 Hz와 105 Hz (주파수가 매우 가깝게)의 두 동시 사인파가 각각 0.5초 지속되다가 0.5초의 정적이 이어지는 신호를 생성하시오.

(a) 윈도우 길이 32, 64, 128, 256, 512 샘플로 스펙트로그램을 계산하고 표시하시오 ($f_s = 1000$ Hz).

(b) 각 윈도우 길이에서 두 주파수가 분리되는지(즉, 별개의 피크로 나타나는지) 확인하시오. 필요한 최소 윈도우 길이는 무엇인가?

(c) $t = 0.3$ s에 날카로운 클릭을 추가하시오. 어느 윈도우 길이에서 클릭의 타이밍과 두 주파수 모두를 식별할 수 있는가?

(d) 불확정성 원리를 검증하시오: 각 윈도우에 대해 $\Delta t$와 $\Delta f$를 계산하고 $\Delta t \cdot \Delta f \geq 1/(4\pi)$인지 확인하시오.

### 연습 문제 2: 윈도우 비교

알려진 처프(1초에 걸쳐 50에서 200 Hz, $f_s = 1000$ Hz)를 가진 신호를 사용하여:

(a) 직사각형, 한, 해밍, 블랙만, 가우시안 윈도우(모두 길이 128)를 사용하여 스펙트로그램을 계산하시오.

(b) 각 스펙트로그램에서 볼 수 있는 스펙트럼 누설을 비교하시오. 어느 윈도우가 가장 깨끗한 처프 궤적을 만드는가?

(c) 각 윈도우에 대해 -3 dB 주엽 폭과 최대 사이드로브 레벨을 측정하시오.

### 연습 문제 3: 웨이블릿 패밀리

(a) 하르, db4, db8, sym4, coif2 웨이블릿에 대한 스케일링 함수 $\phi(t)$와 웨이블릿 함수 $\psi(t)$를 도식화하시오 (`pywt.Wavelet(...).wavefun(level=8)` 사용).

(b) 관련 분해 필터 $h[n]$과 $g[n]$의 주파수 응답을 계산하고 도식화하시오.

(c) 각 웨이블릿에 대해 연습 문제 1의 테스트 신호를 분해하고 시간-주파수 해상도를 비교하시오.

(d) 어느 웨이블릿이 두 가까운 사인파를 가장 잘 분리하는가? 어느 웨이블릿이 클릭을 가장 잘 국재화하는가?

### 연습 문제 4: 웨이블릿 잡음 제거

SNR = 5 dB의 가우시안 잡음으로 오염된 신호 $x(t) = \text{sign}(\sin(2\pi \cdot 3 \cdot t))$ (3 Hz의 사각파)를 생성하시오.

(a) 범용 임계값을 사용하여 경(hard) 및 연(soft) 임계값 처리로 웨이블릿 잡음 제거를 적용하시오. 결과를 비교하시오.

(b) 서로 다른 웨이블릿(하르, db4, sym8)을 시도하고 잡음 제거 품질(SNR)을 비교하시오.

(c) 분해 레벨을 1에서 8까지 변화시키시오. 출력 SNR vs 레벨 수를 도식화하시오.

(d) 단순 저역 통과 필터와 비교하시오. 어떤 조건에서 웨이블릿 잡음 제거가 주파수 영역 필터링보다 우수한가?

### 연습 문제 5: CWT vs DWT

다음으로 구성된 신호에 대해:
- 전체 지속 시간 동안 5 Hz 사인파
- $t = 0.3$에서 $t = 0.5$ s 동안의 50 Hz 버스트
- $t = 0.7$에서 $t = 0.72$ s 동안의 200 Hz 버스트 (매우 짧음)

(a) 모를레 웨이블릿을 사용하여 CWT를 계산하고 스칼로그램을 표시하시오.

(b) db4를 사용하여 6-레벨 DWT를 계산하고 분해를 표시하시오.

(c) 어느 표현이 200 Hz 버스트를 시간적으로 더 잘 국재화하는가? 이유는 무엇인가?

(d) 어느 표현이 5 Hz와 50 Hz 성분을 주파수적으로 더 잘 분리하는가? 이유는 무엇인가?

### 연습 문제 6: 음악 분석

짧은 음악 구절을 로드하거나 합성하시오 (예: 피아노 음계: C4, D4, E4, F4, G4, A4, B4, C5, 각 음표 0.25초 지속).

(a) 스펙트로그램을 계산하고 표시하시오. 각 음표와 그 하모닉스를 식별할 수 있는가?

(b) CWT를 계산하고 스칼로그램을 표시하시오. 스펙트로그램과 비교하시오.

(c) 각 시간 프레임에서 기본 주파수를 찾는 간단한 음표 검출기를 구현하시오.

(d) 대수적으로 간격이 맞춰진 주파수 빈을 사용하여 상수-Q 변환(CQT)을 계산하시오. 음악 분석을 위해 선형-주파수 스펙트로그램과 어떻게 비교되는가?

### 연습 문제 7: 위그너-빌 분포

(a) 단성분 선형 처프에 대한 이산 위그너-빌 분포를 구현하시오. WVD가 깨끗하고 좁은 선을 보여주는지 검증하시오.

(b) 두 선형 처프의 합(하나는 주파수 증가, 하나는 감소)에 대해 WVD를 계산하시오. 교차항을 식별하시오. 시간-주파수 평면에서 어디에 나타나는가?

(c) WVD에 평활화 커널을 적용하여(평활화된 의사 위그너-빌) 교차항이 줄어들지만 해상도가 저하됨을 보이시오.

---

## 18. 요약

| 개념 | 핵심 공식 / 아이디어 |
|---|---|
| STFT | $X(\tau,f) = \int x(t)w(t-\tau)e^{-j2\pi ft}dt$ |
| 스펙트로그램 | $S(\tau,f) = |X_{STFT}(\tau,f)|^2$ |
| 불확정성 원리 | $\Delta t \cdot \Delta f \geq 1/(4\pi)$ |
| CWT | $W(a,b) = \frac{1}{\sqrt{a}}\int x(t)\psi^*(\frac{t-b}{a})dt$ |
| 스케일-주파수 관계 | $f_a = f_c / a$ |
| 허용 가능성 | $\int\psi(t)dt = 0$ (영 평균) |
| 세분화 방정식 | $\phi(t) = \sqrt{2}\sum_k h[k]\phi(2t-k)$ |
| QMF 관계 | $g[n] = (-1)^n h[1-n]$ |
| DWT 계산 | 말라 알고리즘: $O(N)$ |
| 잡음 제거 임계값 | $\lambda = \sigma\sqrt{2\ln N}$ (범용) |
| 잡음 추정 | $\hat{\sigma} = \text{median}(|d_1|)/0.6745$ |

**핵심 정리**:
1. 푸리에 변환은 전역 주파수 정보를 제공하며, STFT와 웨이블릿은 시간 국재화를 추가합니다.
2. 불확정성 원리는 공동 시간-주파수 해상도에 근본적인 한계를 설정합니다.
3. STFT는 고정 윈도우를 사용하여 균일한 해상도를 제공합니다 -- 일관된 특성을 가진 신호에 적합합니다.
4. 웨이블릿은 가변 해상도를 사용합니다: 고주파에서 좁은 윈도우, 저주파에서 넓은 윈도우.
5. 말라 알고리즘을 통한 DWT는 계산적으로 효율적이며($O(N)$), 비중복 표현을 생성합니다.
6. 임계값 처리를 통한 웨이블릿 잡음 제거는 잡음 제거를 위한 강력하고 원칙적인 접근 방식입니다.
7. STFT와 웨이블릿 중 선택은 신호 구조와 분석 목표에 따라 달라집니다.

---

## 19. 참고 문헌

1. S. Mallat, *A Wavelet Tour of Signal Processing*, 3rd ed., Academic Press, 2009.
2. I. Daubechies, *Ten Lectures on Wavelets*, SIAM, 1992.
3. L. Cohen, *Time-Frequency Analysis*, Prentice Hall, 1995.
4. C.K. Chui, *An Introduction to Wavelets*, Academic Press, 1992.
5. D.L. Donoho and I.M. Johnstone, "Ideal spatial adaptation by wavelet shrinkage," *Biometrika*, vol. 81, pp. 425-455, 1994.
6. A. Boggess and F.J. Narcowich, *A First Course in Wavelets with Fourier Analysis*, 2nd ed., Wiley, 2009.
7. M. Vetterli and J. Kovacevic, *Wavelets and Subband Coding*, Prentice Hall, 1995.

---

**이전**: [13. 적응 필터](./13_Adaptive_Filters.md) | **다음**: [15. 영상 신호 처리](./15_Image_Signal_Processing.md)
