# 16. 신호 처리의 응용

**이전**: [15. 영상 신호 처리](./15_Image_Signal_Processing.md) | [개요](./00_Overview.md)

---

신호 처리는 단순한 이론적 학문이 아니라, 우리가 매일 사용하는 시스템을 지탱하는 공학적 근간입니다. 이 마지막 레슨에서는 오디오, 통신, 레이더/소나, 생체의학 신호 처리라는 네 가지 주요 응용 분야를 살펴봅니다. 각 분야에 대해 핵심 신호 처리 개념을 전개하고, 수학적 기반을 제공하며, 동작하는 Python 시연을 구현합니다. 목표는 앞선 열다섯 레슨에서 배운 도구들이 실제 시스템에서 어떻게 결합되는지를 보여주는 것입니다.

**난이도**: ⭐⭐⭐

**선수 지식**: DFT/FFT, 필터링(FIR/IIR), 변조 기초, 상관관계, 스펙트럼 분석

**학습 목표**:
- 오디오 신호 표현을 이해하고 디지털 오디오 이펙트를 구현한다
- 자기상관(autocorrelation)과 켑스트럼(cepstrum) 방법을 사용한 피치(pitch) 검출을 구현한다
- 아날로그 및 디지털 변조 방식(AM, FM, ASK, FSK, PSK, QAM)을 설명한다
- 펄스 성형(pulse shaping), 정합 필터링(matched filtering), OFDM 기초를 설명한다
- 레이더 신호 처리에 정합 필터링과 처프(chirp) 압축을 적용한다
- 레이더 파형 설계를 위한 모호 함수(ambiguity function)를 계산하고 해석한다
- 본 과정의 기법을 사용하여 생체의학 신호(ECG, EEG)를 처리한다
- 각 응용의 동작하는 Python 시연을 구현한다

---

## 목차

1. [오디오 신호 처리](#1-오디오-신호-처리)
2. [오디오 표현과 포맷](#2-오디오-표현과-포맷)
3. [디지털 오디오 이펙트](#3-디지털-오디오-이펙트)
4. [피치 검출](#4-피치-검출)
5. [오디오 이퀄라이제이션](#5-오디오-이퀄라이제이션)
6. [음성 코딩: 선형 예측 코딩(LPC)](#6-음성-코딩-선형-예측-코딩lpc)
7. [통신: 아날로그 변조](#7-통신-아날로그-변조)
8. [통신: 디지털 변조](#8-통신-디지털-변조)
9. [펄스 성형과 정합 필터링](#9-펄스-성형과-정합-필터링)
10. [채널 모델과 등화](#10-채널-모델과-등화)
11. [OFDM 기초](#11-ofdm-기초)
12. [레이더 신호 처리](#12-레이더-신호-처리)
13. [처프 신호와 펄스 압축](#13-처프-신호와-펄스-압축)
14. [모호 함수](#14-모호-함수)
15. [생체의학 신호 처리: ECG](#15-생체의학-신호-처리-ecg)
16. [생체의학 신호 처리: EEG](#16-생체의학-신호-처리-eeg)
17. [심박 변이도(HRV)](#17-심박-변이도hrv)
18. [Python 구현](#18-python-구현)
19. [연습 문제](#19-연습-문제)
20. [요약](#20-요약)
21. [참고 문헌](#21-참고-문헌)

---

## 1. 오디오 신호 처리

### 1.1 오디오 신호 체인

```
음원 → 마이크 → ADC → 디지털 처리 → DAC → 앰프 → 스피커
(아날로그)  (변환기)      (DSP 알고리즘)        (변환기)   (아날로그)
```

오디오 신호 처리는 음파의 디지털 표현을 대상으로 동작합니다. 주요 파라미터는 다음과 같습니다:

- **샘플링 레이트**(sampling rate) ($f_s$): 44.1 kHz(CD), 48 kHz(전문가용), 96/192 kHz(고해상도)
- **비트 심도**(bit depth): 16비트(CD), 24비트(전문가용), 32비트 부동소수점(내부 처리)
- **채널**: 모노(1), 스테레오(2), 서라운드(5.1, 7.1), 공간음향(앰비소닉스)

### 1.2 주파수 대역

| 대역 | 주파수 | 음악적 맥락 |
|---|---|---|
| 서브 베이스 | 20-60 Hz | 듣는다기보다 느껴짐 |
| 베이스 | 60-250 Hz | 킥 드럼, 베이스 기타 |
| 저중음 | 250-500 Hz | 따뜻함, 악기 몸통감 |
| 중음 | 500 Hz - 2 kHz | 보컬 명료도, 존재감 |
| 고중음 | 2-4 kHz | 체감 음량, 어택감 |
| 프레젠스 | 4-6 kHz | 정의감, 명확성 |
| 브릴리언스 | 6-20 kHz | 공기감, 반짝임, 치찰음 |

인간 청각은 약 20 Hz에서 20 kHz에 걸쳐 있으며, 외이도 공명으로 인해 2-5 kHz 부근에서 최대 감도를 보입니다.

---

## 2. 오디오 표현과 포맷

### 2.1 펄스 코드 변조(Pulse Code Modulation, PCM)

PCM은 표준 비압축 디지털 오디오 표현입니다:

1. **샘플링**: 아날로그 신호를 $f_s$ 속도로 샘플링
2. **양자화(Quantize)**: 각 샘플을 $2^B$개 레벨 중 하나에 매핑($B$비트 해상도)
3. **부호화(Encode)**: 각 양자화된 샘플의 이진 표현

풀 스케일 정현파에 대한 **신호 대 양자화 잡음비(signal-to-quantization-noise ratio, SQNR)**:

$$\text{SQNR} = 6.02B + 1.76 \text{ dB}$$

16비트 오디오의 경우: $\text{SQNR} \approx 98$ dB. 24비트의 경우: $\text{SQNR} \approx 146$ dB.

### 2.2 데이터 레이트

비압축 데이터 레이트:

$$R = f_s \times B \times C$$

여기서 $C$는 채널 수입니다.

CD 오디오(44.1 kHz, 16비트, 스테레오)의 경우: $R = 44100 \times 16 \times 2 = 1.41$ Mbps.

### 2.3 디더링(Dithering)

비트 심도를 줄일 때 **디더링**은 양자화 전에 소량의 잡음을 추가하여 왜곡(고조파)을 잡음(광대역)으로 변환합니다. 이를 통해 약간의 잡음 바닥(noise floor) 상승을 감수하는 대신 양자화 왜곡 아티팩트를 제거합니다.

디더 잡음은 일반적으로 진폭이 $\pm 1$ LSB인 삼각형 확률 밀도 함수(triangular probability density function, TPDF)를 사용합니다.

---

## 3. 디지털 오디오 이펙트

### 3.1 지연 기반 이펙트

대부분의 오디오 이펙트는 지연 라인(delay line)으로 구성됩니다:

$$y[n] = x[n] + g \cdot x[n - D]$$

여기서 $D$는 샘플 단위 지연이고 $g$는 피드백/피드포워드 이득입니다.

#### 에코/딜레이(Echo/Delay)

피드백이 있는 단순 지연:

$$y[n] = x[n] + g \cdot y[n - D]$$

청취 가능한 에코의 경우: $D > 50$ ms ($D > 0.05 f_s$ 샘플).

#### 리버브(Reverb)

리버브는 방의 음향 응답을 시뮬레이션합니다. 가장 간단한 모델은 빗살 필터(comb filter)와 전역 통과 필터(allpass filter)를 사용합니다(슈뢰더 리버브레이터):

**빗살 필터(피드백)**:
$$y[n] = x[n] + g \cdot y[n - D]$$

**전역 통과 필터(Allpass filter)**:
$$y[n] = -g \cdot x[n] + x[n - D] + g \cdot y[n - D]$$

슈뢰더 리버브레이터는 4개의 병렬 빗살 필터에 이어 2개의 직렬 전역 통과 필터를 연결합니다.

더 정교한 리버브는 **컨볼루션 리버브(convolution reverb)**를 사용합니다: 건식 신호(dry signal)를 측정된 실내 임펄스 응답(room impulse response, RIR)과 컨볼루션합니다.

### 3.2 변조 기반 이펙트

#### 코러스(Chorus)

코러스 이펙트는 지연이 천천히 변조되는 원본 신호와 지연된 복사본을 혼합하여 같은 음표를 연주하는 여러 악기를 시뮬레이션합니다:

$$y[n] = x[n] + g \cdot x[n - D(n)]$$

여기서 $D(n) = D_0 + A \sin(2\pi f_{LFO} n / f_s)$이며, $D_0 \approx 20$-30 ms, $A \approx 1$-5 ms, $f_{LFO} \approx 0.5$-3 Hz입니다.

#### 플랜저(Flanger)

코러스와 유사하지만 더 짧은 지연과 피드백을 사용합니다:

$$y[n] = x[n] + g \cdot y[n - D(n)]$$

$D_0 \approx 1$-10 ms로, 스위핑하는 빗살 필터 이펙트를 만들어냅니다.

#### 비브라토(Vibrato)

순수한 피치 변조(건식 신호 혼합 없음):

$$y[n] = x[n - D(n)]$$

정현파 지연 변조를 사용합니다.

### 3.3 다이나믹스 처리

#### 컴프레서(Compressor)

임계값을 초과하는 신호를 감쇠시켜 다이나믹 레인지를 줄입니다:

$$g_{dB}[n] = \begin{cases} 0 & x_{dB}[n] < T \\ (1 - 1/R)(T - x_{dB}[n]) & x_{dB}[n] \geq T \end{cases}$$

여기서 $T$는 임계값(threshold)이고 $R$은 압축 비율(compression ratio)입니다.

**파라미터**:
- **임계값(Threshold)**: 압축이 시작되는 레벨
- **비율(Ratio)**: 압축량(2:1, 4:1, $\infty$:1 = 리미터)
- **어택(Attack)**: 컴프레서가 반응하는 속도
- **릴리즈(Release)**: 컴프레서가 단위 이득으로 돌아가는 속도
- **니(Knee)**: 임계값에서의 하드 또는 소프트 전환

어택과 릴리즈는 지수 스무딩(exponential smoothing)을 사용하는 엔벨로프 팔로워(envelope follower)로 구현됩니다:

$$x_{env}[n] = \begin{cases} \alpha_a x_{env}[n-1] + (1-\alpha_a)|x[n]| & |x[n]| > x_{env}[n-1] \\ \alpha_r x_{env}[n-1] + (1-\alpha_r)|x[n]| & |x[n]| \leq x_{env}[n-1] \end{cases}$$

여기서 $\alpha_a = e^{-1/(f_s \cdot t_{attack})}$이고 $\alpha_r = e^{-1/(f_s \cdot t_{release})}$입니다.

---

## 4. 피치 검출

### 4.1 자기상관 방법

주기 신호의 기본 주파수(fundamental frequency)는 자기상관 함수(autocorrelation function)로 검출할 수 있습니다:

$$R_{xx}[\tau] = \sum_{n=0}^{N-1} x[n] \cdot x[n + \tau]$$

주기 $T_0$(샘플 단위)를 가진 주기 신호에서 $R_{xx}[\tau]$는 $\tau = 0, T_0, 2T_0, \ldots$에서 피크를 가집니다. 기본 주파수는:

$$f_0 = \frac{f_s}{T_0}$$

**알고리즘**:
1. 신호를 윈도우 처리(해닝 윈도우, 20-50 ms 프레임)
2. 자기상관 계산
3. 원점 이후 첫 번째 유의미한 피크 찾기
4. 피크 위치가 $T_0$를 결정

**피치 범위 제약**: 음성의 경우 $f_0 \in [80, 400]$ Hz이므로 $\tau \in [f_s/400, f_s/80]$에서 피크를 탐색합니다.

### 4.2 켑스트럼 방법

**켑스트럼(cepstrum)**("spectrum"의 철자 재배열)은 다음과 같이 정의됩니다:

$$c[n] = \text{IDFT}\{\log|\text{DFT}\{x[n]\}|\}$$

켑스트럼 영역의 독립 변수 $n$은 **퀘프런시(quefrency)**("frequency"의 철자 재배열)라고 하며 샘플(또는 시간) 단위를 가집니다.

유성음(voiced speech) 신호의 경우 켑스트럼은 **성도 엔벨로프(vocal tract envelope)**(낮은 퀘프런시)와 **피치 여기(pitch excitation)**(피치 주기의 퀘프런시에서 피크)를 분리합니다:

```
Speech spectrum = Vocal tract envelope × Pitch harmonics
    (smooth)              (fine structure)

log(Speech spectrum) = log(Envelope) + log(Harmonics)
    cepstrum domain:    low quefrency   + peak at T₀
```

**알고리즘**:
1. 윈도우 처리된 프레임의 DFT 계산
2. 크기 스펙트럼의 로그 취하기
3. IDFT 계산(켑스트럼)
4. 예상 피치 범위에서 피크 찾기
5. 피크 퀘프런시 = 피치 주기 $T_0$

### 4.3 비교

| 방법 | 장점 | 단점 |
|---|---|---|
| 자기상관 | 강인함, 단순함 | 옥타브 오류, 넓은 피크 |
| 켑스트럼 | 소스/필터 분리 우수 | 잡음 민감성, 해상도 제한 |
| YIN(개선된 자기상관) | 최신 정확도 | 더 복잡함 |
| pYIN | 확률론적, 매우 강인함 | 연산 비용 |

---

## 5. 오디오 이퀄라이제이션

### 5.1 그래픽 이퀄라이저(Graphic Equalizer)

그래픽 이퀄라이저는 고정된 중심 주파수(일반적으로 옥타브 또는 1/3 옥타브 간격)에서 병렬 대역 통과 필터들로 구성됩니다. 각 대역은 조절 가능한 이득을 가집니다.

표준 옥타브 대역 중심 주파수: 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000 Hz.

### 5.2 파라메트릭 이퀄라이저(Parametric Equalizer)

파라메트릭 이퀄라이저는 각 대역에 대해 조절 가능한 **중심 주파수**, **이득**, **대역폭**(Q 팩터)을 제공합니다:

**피킹(벨) 필터**(2차 IIR):

$$H(z) = \frac{1 + \alpha A \cdot z^{-1} \cdot b_1 + \alpha z^{-2}}{1 + \alpha/A \cdot z^{-1} \cdot a_1 + \alpha z^{-2}}$$

여기서 $A = 10^{G_{dB}/40}$, $\omega_0 = 2\pi f_0/f_s$, $\alpha = \sin(\omega_0)/(2Q)$입니다.

피킹 EQ의 쿡북 바이쿼드(biquad) 계수:

$$b_0 = 1 + \alpha A, \quad b_1 = -2\cos\omega_0, \quad b_2 = 1 - \alpha A$$
$$a_0 = 1 + \alpha/A, \quad a_1 = -2\cos\omega_0, \quad a_2 = 1 - \alpha/A$$

### 5.3 쉘빙 필터(Shelving Filter)

- **로우 쉘프(Low shelf)**: $f_0$ 이하의 주파수를 부스트/컷
- **하이 쉘프(High shelf)**: $f_0$ 이상의 주파수를 부스트/컷

오디오 장비의 베이스 및 트레블 컨트롤에 사용됩니다.

---

## 6. 음성 코딩: 선형 예측 코딩(LPC)

### 6.1 소스-필터 모델

음성 생성은 다음과 같이 모델링됩니다:

$$\text{여기 소스} \to \text{성도 필터} \to \text{음성 신호}$$

- **유성음(Voiced sounds)**(모음): 여기 = 피치 주파수의 주기적 펄스열
- **무성음(Unvoiced sounds)**(마찰음): 여기 = 백색 잡음
- **성도(Vocal tract)**: 전극 필터(all-pole filter, 공명 관 모델)

### 6.2 선형 예측

현재 샘플은 과거 샘플들로부터 예측됩니다:

$$\hat{x}[n] = \sum_{k=1}^{p} a_k x[n-k]$$

예측 오차(잔차)는:

$$e[n] = x[n] - \hat{x}[n] = x[n] - \sum_{k=1}^{p} a_k x[n-k]$$

LPC 계수 $\{a_k\}$는 $E[e^2[n]]$을 최소화하도록 선택됩니다.

### 6.3 LPC 계수 계산

예측 오차를 최소화하면 **율-워커(Yule-Walker, 정규) 방정식**이 도출됩니다:

$$\mathbf{R}\mathbf{a} = \mathbf{r}$$

여기서 $\mathbf{R}$은 토플리츠 자기상관 행렬이고 $\mathbf{r}$은 자기상관 벡터입니다.

**레빈슨-더빈 알고리즘(Levinson-Durbin algorithm)**은 토플리츠 구조를 활용하여 $O(p^2)$ 연산으로 이를 풀어냅니다.

### 6.4 LPC 보코더(Vocoder)

LPC 보코더는 다음만을 전송합니다:
1. **LPC 계수** ($\{a_k\}$, 협대역 음성의 경우 일반적으로 $p = 10$-16)
2. **피치 주기** (유성음 프레임의 경우)
3. **유성음/무성음 플래그**
4. **이득** (프레임의 에너지)

이를 통해 음성 품질을 일부 희생하고 매우 낮은 비트율(LPC-10의 경우 2.4 kbps)을 달성합니다.

### 6.5 LPC 스펙트럼

LPC 모델은 성도를 전극 필터로 표현합니다:

$$H(z) = \frac{G}{1 - \sum_{k=1}^{p} a_k z^{-k}} = \frac{G}{A(z)}$$

LPC 스펙트럼 $|H(e^{j\omega})|^2$는 음성 신호의 부드러운 스펙트럼 엔벨로프를 제공합니다. $H(z)$의 극(pole)들은 **포르만트 주파수(formant frequency, 성도의 공명)**에 해당합니다.

---

## 7. 통신: 아날로그 변조

### 7.1 변조가 필요한 이유

변조(modulation)는 기저대역(baseband) 신호를 더 높은 반송파(carrier) 주파수로 이동시킵니다:
1. **안테나 효율**: 안테나 크기 $\sim \lambda/4$; 높은 주파수 = 작은 안테나
2. **주파수 분할 다중화**: 여러 신호가 스펙트럼 공유
3. **잡음 성능**: 일부 변조 방식은 잡음 개선 제공

### 7.2 진폭 변조(Amplitude Modulation, AM)

AM 신호는:

$$x_{AM}(t) = [1 + m \cdot x(t)] \cos(2\pi f_c t)$$

여기서:
- $f_c$는 반송파 주파수
- $m$은 변조 지수(왜곡 없으려면 $0 < m \leq 1$)
- $x(t)$는 정규화된 메시지 신호($|x(t)| \leq 1$)

**스펙트럼**: AM 스펙트럼은 반송파와 상부 및 하부 측대역(sideband)으로 구성됩니다:
$$X_{AM}(f) = \frac{1}{2}\delta(f - f_c) + \frac{m}{4}[X(f - f_c) + X(f + f_c)]$$

**대역폭**: $B_{AM} = 2W$ (여기서 $W$는 메시지 대역폭)

**복조**: 엔벨로프 검파(단순 다이오드 + RC 회로)

### 7.3 주파수 변조(Frequency Modulation, FM)

FM 신호는:

$$x_{FM}(t) = A_c \cos\!\left(2\pi f_c t + 2\pi k_f \int_0^t x(\tau) \, d\tau\right)$$

여기서 $k_f$는 주파수 편이 상수(Hz/V)입니다.

**순시 주파수(instantaneous frequency)**는:

$$f_i(t) = f_c + k_f x(t)$$

**대역폭**(카슨의 법칙, Carson's rule):

$$B_{FM} \approx 2(\Delta f + W) = 2W(\beta + 1)$$

여기서 $\Delta f = k_f \max|x(t)|$는 최대 주파수 편이이고 $\beta = \Delta f / W$는 변조 지수입니다.

**AM 대비 FM의 장점**:
- 일정한 엔벨로프(진폭 변동 없음) -- 비선형 증폭기에 강인함
- 더 나은 잡음 성능(FM은 잡음을 진폭 변동으로 포착하며, 이는 리미팅으로 제거됨)
- 캡처 효과: 강한 신호가 약한 간섭을 억압

### 7.4 위상 변조(Phase Modulation, PM)

$$x_{PM}(t) = A_c \cos\!\left(2\pi f_c t + k_p x(t)\right)$$

PM과 FM은 연관되어 있습니다: $x(t)$의 FM은 $\int x(t)dt$의 PM과 동일합니다.

---

## 8. 통신: 디지털 변조

### 8.1 디지털 변조 개요

디지털 변조(digital modulation)는 이산 심벌을 전송을 위한 아날로그 파형에 매핑합니다:

| 방식 | 변하는 것 | 성상도(Constellation) |
|---|---|---|
| ASK(진폭 편이 변조) | 진폭 | 실수축 위의 점들 |
| FSK(주파수 편이 변조) | 주파수 | 직교 신호들 |
| PSK(위상 편이 변조) | 위상 | 단위원 위의 점들 |
| QAM(직교 진폭 변조) | 진폭 + 위상 | I-Q 평면의 격자 |

### 8.2 이진 위상 편이 변조(Binary Phase Shift Keying, BPSK)

가장 단순한 PSK 방식으로 비트를 대척점 신호에 매핑합니다:

$$s(t) = \begin{cases} +A\cos(2\pi f_c t) & \text{비트 } 1 \\ -A\cos(2\pi f_c t) & \text{비트 } 0 \end{cases}$$

AWGN에서의 **비트 오율(Bit error rate, BER)**:

$$P_b = Q\!\left(\sqrt{\frac{2E_b}{N_0}}\right) = \frac{1}{2}\text{erfc}\!\left(\sqrt{\frac{E_b}{N_0}}\right)$$

여기서 $E_b$는 비트당 에너지이고 $N_0$는 잡음 전력 스펙트럼 밀도입니다.

### 8.3 직교 위상 편이 변조(Quadrature Phase Shift Keying, QPSK)

QPSK는 비트 쌍(다이비트)을 네 개의 위상 상태에 매핑합니다:

$$s_k(t) = A\cos\!\left(2\pi f_c t + \frac{\pi}{4} + \frac{k\pi}{2}\right), \quad k = 0, 1, 2, 3$$

**성상도**: $\{\pi/4, 3\pi/4, 5\pi/4, 7\pi/4\}$ 각도의 4점.

QPSK는 **BPSK와 동일한 BER**을 가지지만 같은 대역폭에서 **두 배의 데이터 레이트**를 전달합니다.

### 8.4 직교 진폭 변조(Quadrature Amplitude Modulation, QAM)

$M$-QAM은 진폭과 위상 모두를 사용하여 심벌당 $\log_2 M$ 비트를 매핑합니다:

$$s(t) = A_I \cos(2\pi f_c t) - A_Q \sin(2\pi f_c t)$$

여기서 $(A_I, A_Q)$는 규칙적인 격자에서 선택됩니다.

**16-QAM**: 심벌당 4비트, $4 \times 4$ 격자의 16개 성상도 점.

**64-QAM**: 심벌당 6비트(WiFi, 케이블 TV에서 사용).

**256-QAM**: 심벌당 8비트(고처리량 WiFi).

트레이드오프: $M$이 높을수록 스펙트럼 효율이 높아지지만 더 높은 SNR이 필요합니다.

### 8.5 I-Q 표현

모든 대역 통과 신호는 다음과 같이 표현할 수 있습니다:

$$s(t) = I(t)\cos(2\pi f_c t) - Q(t)\sin(2\pi f_c t)$$

여기서 $I(t)$는 **동위상(in-phase)** 성분이고 $Q(t)$는 **직교(quadrature)** 성분입니다. 복소 기저대역 등가는:

$$\tilde{s}(t) = I(t) + jQ(t)$$

모든 디지털 변조 방식은 심벌을 $I$-$Q$ 평면의 점에 매핑하는 것으로 설명할 수 있습니다.

---

## 9. 펄스 성형과 정합 필터링

### 9.1 심벌간 간섭(Inter-Symbol Interference, ISI) 문제

디지털 심벌이 펄스로 전송될 때, 한 심벌 펄스의 꼬리가 인접 심벌을 간섭할 수 있습니다:

$$r(t) = \sum_k a_k \, p(t - kT_s)$$

여기서 $a_k$는 심벌 값, $p(t)$는 펄스 형태, $T_s$는 심벌 주기입니다.

### 9.2 나이퀴스트 ISI 기준

샘플링 순간에 펄스 $p(t)$가 ISI를 일으키지 않으려면:

$$p(kT_s) = \begin{cases} 1 & k = 0 \\ 0 & k \neq 0 \end{cases}$$

sinc 펄스 $p(t) = \text{sinc}(t/T_s)$는 최소 대역폭 $W = 1/(2T_s)$로 이를 달성하지만 천천히 감소하여 실용적이지 않습니다.

### 9.3 상승 코사인 펄스(Raised Cosine Pulse)

상승 코사인 펄스는 더 빠른 감소로 ISI를 제로로 만듭니다:

$$P(f) = \begin{cases} T_s & |f| \leq \frac{1-\alpha}{2T_s} \\[6pt] \frac{T_s}{2}\left[1 + \cos\!\left(\frac{\pi T_s}{\alpha}\left(|f| - \frac{1-\alpha}{2T_s}\right)\right)\right] & \frac{1-\alpha}{2T_s} < |f| \leq \frac{1+\alpha}{2T_s} \\[6pt] 0 & |f| > \frac{1+\alpha}{2T_s} \end{cases}$$

여기서 $\alpha \in [0, 1]$은 **롤오프 팩터(rolloff factor)**입니다.

**대역폭**: $W = \frac{1+\alpha}{2T_s}$

실제로는 **루트 상승 코사인(root-raised cosine, RRC)** 필터를 송신기와 수신기에 나누어 사용합니다: $P_{RRC}(f) = \sqrt{P_{RC}(f)}$. 이 방식으로 $P_{TX}(f) \cdot P_{RX}(f) = P_{RC}(f)$가 되며, 수신기 필터는 전송된 펄스의 정합 필터가 됩니다.

### 9.4 정합 필터(Matched Filter)

**정합 필터**는 가법 백색 잡음(additive white noise)에서 알려진 펄스 $s(t)$에 대해 출력 SNR을 최대화합니다:

$$h_{matched}(t) = s^*(T - t)$$

정합 필터는 신호의 시간 반전 후 켤레 복소수를 취하고 $T$만큼 지연한 것입니다.

정합 필터의 **출력 SNR**:

$$\text{SNR}_{max} = \frac{2E_s}{N_0}$$

여기서 $E_s = \int |s(t)|^2 \, dt$는 신호 에너지입니다. 이는 필터에 관계없이 달성 가능한 최대 SNR입니다.

---

## 10. 채널 모델과 등화

### 10.1 가법 백색 가우시안 잡음(Additive White Gaussian Noise, AWGN) 채널

$$r(t) = s(t) + n(t)$$

여기서 $n(t)$는 전력 스펙트럼 밀도 $N_0/2$인 백색 가우시안 잡음입니다.

### 10.2 다중경로 채널(Multipath Channel)

$$r(t) = \sum_{k=0}^{L-1} h_k \, s(t - \tau_k) + n(t)$$

이산 시간 등가:

$$r[n] = \sum_{k=0}^{L-1} h[k] \, s[n-k] + v[n] = (h * s)[n] + v[n]$$

다중경로는 다음을 초래합니다:
- **ISI**: 신호의 지연된 복사본이 현재 심벌을 간섭
- **주파수 선택적 페이딩(Frequency-selective fading)**: 일부 주파수가 다른 주파수보다 더 많이 감쇠

### 10.3 영점 강제(Zero-Forcing, ZF) 등화기

ZF 등화기는 채널을 역변환합니다:

$$W_{ZF}(f) = \frac{1}{H(f)}$$

**문제**: $|H(f)|$가 작은 주파수에서의 잡음 증폭.

### 10.4 MMSE 등화기

$$W_{MMSE}(f) = \frac{H^*(f)}{|H(f)|^2 + N_0/E_s}$$

MMSE 등화기는 ISI 제거와 잡음 억제의 균형을 맞춥니다. $N_0 = 0$일 때 ZF 등화기로 환원됩니다.

### 10.5 적응 등화(Adaptive Equalization)

실제로는 채널이 알려지지 않고 시변합니다. 적응 등화기(레슨 13의 LMS, RLS)가 다음과 함께 사용됩니다:
- **훈련 시퀀스**: 초기 수렴을 위한 알려진 심벌
- **결정 지향 모드**: 수렴 후 검출된 심벌을 기준으로 사용

---

## 11. OFDM 기초

### 11.1 동기

직교 주파수 분할 다중화(Orthogonal Frequency Division Multiplexing, OFDM)는 광대역 무선(WiFi, LTE, 5G) 및 유선(DSL, 케이블) 통신의 지배적인 변조 방식입니다.

핵심 아이디어: 주파수 선택적 채널을 통해 하나의 고속 스트림을 전송하는 대신, 각각 평탄 페이딩(flat fading)을 경험하는 병렬 협대역 부반송파(subcarrier)에서 많은 저속 스트림을 전송합니다.

### 11.2 OFDM 시스템 모델

```
Transmitter:
  Data → S/P → QAM Map → IFFT → Add CP → P/S → DAC → Channel

Receiver:
  ADC → S/P → Remove CP → FFT → QAM Demap → P/S → Data
```

전송된 OFDM 심벌(이산 시간):

$$x[n] = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} X[k] \, e^{j2\pi kn/N}, \quad n = 0, 1, \ldots, N-1$$

이것은 단순히 주파수 영역 데이터 심벌 $X[k]$의 **IFFT**입니다.

### 11.3 순환 전치(Cyclic Prefix, CP)

순환 전치는 OFDM 심벌 시작 부분에 마지막 $L_{CP}$ 샘플을 복사합니다:

$$\tilde{x}[n] = x[n \mod N], \quad n = -L_{CP}, \ldots, N-1$$

목적: 선형 컨볼루션(채널)을 DFT에 의해 대각화되는 **순환 컨볼루션**으로 변환합니다. 이는 OFDM 심벌 간의 심벌간 간섭을 제거합니다.

**조건**: $L_{CP} \geq L_{channel} - 1$ (CP 길이가 채널 임펄스 응답 길이 마이너스 1 이상이어야 함).

### 11.4 단일 탭 등화(One-Tap Equalization)

CP를 제거하고 수신기에서 FFT를 취한 후:

$$Y[k] = H[k] \cdot X[k] + V[k]$$

각 부반송파는 **평탄한**(스칼라) 채널 $H[k]$를 경험합니다. 등화는 간단합니다:

$$\hat{X}[k] = \frac{Y[k]}{H[k]}$$

이것이 OFDM의 엄청난 장점입니다: 복잡한 주파수 선택적 등화 문제를 $N$개의 독립적인 단일 탭 등화로 줄입니다.

### 11.5 OFDM 파라미터(WiFi 예시)

| 파라미터 | 802.11a/g (20 MHz) |
|---|---|
| 부반송파 수 ($N$) | 64 |
| 데이터 부반송파 | 48 |
| 파일럿 부반송파 | 4 |
| 부반송파 간격 | 312.5 kHz |
| 심벌 지속 시간 | 3.2 $\mu$s |
| CP 지속 시간 | 0.8 $\mu$s |
| 전체 심벌 | 4.0 $\mu$s |

---

## 12. 레이더 신호 처리

### 12.1 레이더 기초

**레이더(RADAR)**: 전파 탐지 및 거리 측정(Radio Detection and Ranging). 레이더는 펄스를 전송하고, 표적에서 돌아온 에코는 다음을 알려줍니다:

- **거리**: $R = \frac{c \cdot \tau}{2}$ (여기서 $\tau$는 왕복 지연, $c$는 광속)
- **속도**: 도플러 편이(Doppler shift) $f_d = \frac{2v_r}{\lambda}$ (여기서 $v_r$은 반경 방향 속도, $\lambda$는 파장)

### 12.2 거리 해상도(Range Resolution)

거리 해상도는 전송 펄스의 대역폭으로 결정됩니다:

$$\Delta R = \frac{c}{2B}$$

여기서 $B$는 신호 대역폭입니다.

지속 시간 $\tau_p$의 단순 직사각형 펄스의 경우:
- 대역폭: $B \approx 1/\tau_p$
- 거리 해상도: $\Delta R = c\tau_p/2$

**딜레마**: 좋은 거리 해상도는 짧은 펄스(넓은 대역폭)를 필요로 하지만, 짧은 펄스는 에너지가 낮아 탐지 거리를 제한합니다. 펄스 압축이 이를 해결합니다.

### 12.3 도플러 처리(Doppler Processing)

이동하는 표적에서 수신된 신호는 주파수 편이를 가집니다:

$$f_d = \frac{2v_r f_c}{c} = \frac{2v_r}{\lambda}$$

지속 시간 $T_{CPI}$의 코히어런트 처리 구간(coherent processing interval, CPI)으로부터의 속도 해상도:

$$\Delta v = \frac{\lambda}{2T_{CPI}}$$

### 12.4 레이더용 정합 필터

레이더 수신기는 출력 SNR을 최대화하기 위해 정합 필터를 사용합니다:

$$h_{MF}[n] = s^*[N-1-n]$$

정합 필터 출력은 수신 신호와 전송 파형의 **교차 상관(cross-correlation)**입니다:

$$y[n] = \sum_k r[k] \, s^*[k - n]$$

단순 펄스의 경우 정합 필터 출력은 표적 지연에서 피크를 가지는 삼각형입니다.

---

## 13. 처프 신호와 펄스 압축

### 13.1 선형 FM 처프(Linear FM Chirp)

**처프(chirp)**(선형 주파수 변조 펄스)는 펄스 지속 시간에 걸쳐 주파수를 선형적으로 스윕합니다:

$$s(t) = \text{rect}\!\left(\frac{t}{\tau_p}\right) \exp\!\left(j\pi \frac{B}{\tau_p} t^2\right) \exp(j2\pi f_c t)$$

순시 주파수는 펄스 지속 시간 $\tau_p$ 동안 $f_c - B/2$에서 $f_c + B/2$로 변합니다.

**핵심 특성**: 처프는 대역폭 $B$와 지속 시간 $\tau_p$를 가지므로 **시간-대역폭 곱(time-bandwidth product)**은:

$$\text{TBP} = B \tau_p \gg 1$$

### 13.2 펄스 압축(Pulse Compression)

처프의 정합 필터는 긴 펄스를 짧은 피크로 압축합니다:

- **입력 펄스**: 지속 시간 $\tau_p$, 대역폭 $B$
- **압축된 펄스**: 지속 시간 $\approx 1/B$, 피크 진폭 $\approx \sqrt{B\tau_p}$

**압축비(compression ratio)**는:

$$\text{CR} = B\tau_p$$

**처리 이득(processing gain)**은:

$$G_p = 10\log_{10}(B\tau_p) \text{ dB}$$

예시: $\tau_p = 10$ $\mu$s, $B = 10$ MHz $\Rightarrow$ TBP = 100 $\Rightarrow$ 처리 이득 = 20 dB.

### 13.3 압축 후 거리 해상도

펄스 압축 후 거리 해상도는:

$$\Delta R = \frac{c}{2B}$$

이는 펄스 지속 시간 $\tau_p$가 아닌 대역폭 $B$에 의해 결정됩니다. 긴 처프 펄스는 훨씬 더 많은 에너지를 가지면서도 짧은 연속파(CW) 펄스와 동일한 거리 해상도를 달성합니다.

### 13.4 사이드로브와 윈도잉

압축된 펄스는 사이드로브(sinc 함수의 사이드로브에 유사)를 가집니다. 정합 필터에 적용된 윈도우 함수(해밍, 테일러 등)는 약간 넓어진 메인로브(저하된 해상도)를 대가로 사이드로브를 줄입니다.

---

## 14. 모호 함수

### 14.1 정의

**모호 함수(ambiguity function)**는 레이더 파형이 거리와 속도에서 표적을 동시에 분해하는 능력을 기술합니다:

$$\chi(\tau, f_d) = \int_{-\infty}^{\infty} s(t) \, s^*(t - \tau) \, e^{j2\pi f_d t} \, dt$$

여기서 $\tau$는 시간 지연(거리)이고 $f_d$는 도플러 주파수(속도)입니다.

### 14.2 특성

1. **원점에서 최대값**: $|\chi(0, 0)| = E_s$ (신호 에너지)
2. **부피 불변성(Volume invariance)**: $\iint |\chi(\tau, f_d)|^2 \, d\tau \, df_d = E_s^2$
3. **대칭성**: $|\chi(\tau, f_d)| = |\chi(-\tau, -f_d)|$

부피 불변성은 모호 함수가 고정된 부피 표면임을 의미합니다: 한 차원에서 날카롭게 만들면 다른 차원이 반드시 넓어집니다. 이는 불확정성 원리의 레이더 유사체입니다.

### 14.3 일반적인 파형의 모호 함수

**CW 펄스(직사각형)**:
- $\tau$ 축을 따라 엄지 못(thumbtack) 모양(폭 $\sim \tau_p$)
- $f_d$ 축을 따라 sinc 모양(폭 $\sim 1/\tau_p$)
- 좋은 도플러 해상도, 낮은 거리 해상도(긴 $\tau_p$의 경우)

**선형 FM 처프**:
- 대각선 능선을 따라 좁음(거리-도플러 결합)
- 좋은 거리 해상도($B$에 의해 결정)
- 능선은 도플러 편이가 거리 편이처럼 보인다는 것을 의미 -- 보상이 필요

**위상 코딩 파형**(예: 바커 코드(Barker code)):
- 엄지 못 모양의 모호 함수
- 거리와 도플러 모두에서 낮은 사이드로브
- 특정 코드 길이로 제한

### 14.4 파형 설계

이상적인 모호 함수는 원점의 단일 스파이크(거리와 속도 모두에서 완벽한 해상도)이겠지만, 부피 불변성이 이를 금지합니다. 파형 설계는 응용에 가장 적합한 형태를 선택하는 것입니다:

- **감시 레이더**: 좋은 도플러 해상도 필요 → 긴 CW 펄스
- **추적 레이더**: 좋은 거리 해상도 필요 → 처프
- **펄스-도플러 레이더**: 둘 다 필요 → 처프 펄스의 코히어런트 열

---

## 15. 생체의학 신호 처리: ECG

### 15.1 ECG 신호

심전도(electrocardiogram, ECG 또는 EKG)는 심장의 전기적 활동을 기록합니다. 단일 심박 주기는 특징적인 PQRST 파형을 생성합니다:

```
    R
    ▲
    │╲
    │ ╲
    │  ╲
    │   ╲      T
P   │    ╲    ╱╲
╱╲  │     ╲  ╱  ╲
    │      ╲╱    ╲
────┼──────S──────────── baseline
    Q

P파:    심방 탈분극           (0.08-0.10 s)
QRS:    심실 탈분극           (0.06-0.10 s)
T파:    심실 재분극           (0.16 s)
PR 간격: AV 전도 시간         (0.12-0.20 s)
QT 간격: 심실 활동            (0.30-0.44 s)
```

### 15.2 ECG 신호 특성

| 파라미터 | 값 |
|---|---|
| 진폭 | 0.1 - 5 mV |
| 대역폭 | 0.05 - 150 Hz |
| 일반적인 샘플링 레이트 | 250 - 1000 Hz |
| 심박수 | 60-100 bpm (1-1.67 Hz) |

### 15.3 ECG 잡음 소스

1. **기저선 방랑(Baseline wander)**: 호흡, 신체 움직임으로 인한 저주파 드리프트(< 0.5 Hz)
2. **전원선 간섭**: 50/60 Hz 및 고조파
3. **근육 잡음(EMG)**: 광대역, 20-500 Hz
4. **동작 아티팩트**: 전극 움직임, 광대역

### 15.4 ECG 전처리 파이프라인

```
Raw ECG → Baseline removal → Notch filter → Bandpass filter → QRS detection
           (highpass 0.5 Hz)  (50/60 Hz)    (0.5-40 Hz)      (Pan-Tompkins)
```

**기저선 방랑 제거**: 차단 주파수 0.5 Hz의 고역 통과 필터(또는 200 ms 및 600 ms 윈도우의 중앙값 필터).

**전원선 간섭**: 50 또는 60 Hz에서의 노치 필터(좁은 대역 저지 IIR 필터).

### 15.5 QRS 검출: Pan-Tompkins 알고리즘

Pan-Tompkins 알고리즘(1985)은 표준 QRS 검출기입니다:

1. **대역 통과 필터**(5-15 Hz): QRS 에너지를 최대화하면서 P/T 파와 잡음을 억압
2. **미분**: QRS의 가파른 기울기를 강조: $y[n] = \frac{1}{8}(-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2])$
3. **제곱**: $z[n] = y[n]^2$ (모든 값을 양수로 만들고, 큰 기울기를 강조)
4. **이동 윈도우 적분**: $w[n] = \frac{1}{N}\sum_{k=0}^{N-1} z[n-k]$ ($N \approx 150$ ms)
5. **적응 임계값 처리**: 신호와 잡음 레벨에 적응하는 두 임계값

---

## 16. 생체의학 신호 처리: EEG

### 16.1 EEG 신호

뇌전도(electroencephalogram, EEG)는 두피 전극에서 뇌의 전기적 활동을 기록합니다. EEG 신호는 ECG보다 훨씬 약합니다.

| 파라미터 | 값 |
|---|---|
| 진폭 | 1 - 200 $\mu$V |
| 대역폭 | 0.5 - 100 Hz |
| 샘플링 레이트 | 256 - 1024 Hz |
| 채널 | 1-256 (10-20 시스템) |

### 16.2 EEG 주파수 대역

| 대역 | 주파수 | 상태 |
|---|---|---|
| 델타($\delta$) | 0.5-4 Hz | 깊은 수면 |
| 세타($\theta$) | 4-8 Hz | 졸음, 얕은 수면, 명상 |
| 알파($\alpha$) | 8-13 Hz | 편안함, 눈 감음 |
| 베타($\beta$) | 13-30 Hz | 활발한 사고, 집중 |
| 감마($\gamma$) | 30-100 Hz | 고차 인지 기능, 지각 |

### 16.3 EEG 스펙트럼 분석

EEG의 전력 스펙트럼 밀도(power spectral density, PSD)는 지배적인 뇌 상태를 드러냅니다:

$$S_{xx}(f) = \frac{1}{N}|X(f)|^2$$

**대역 전력(Band power)**은 각 주파수 대역에 걸쳐 PSD를 적분하여 계산됩니다:

$$P_{band} = \int_{f_1}^{f_2} S_{xx}(f) \, df$$

**상대 대역 전력**: $P_{rel} = P_{band} / P_{total}$은 각 대역의 전체 전력 대비 비율을 나타냅니다.

### 16.4 사건 관련 전위(Event-Related Potentials, ERPs)

ERP는 자극에 시간 잠금된 많은 EEG 시행을 평균하여 얻습니다:

$$\text{ERP}[n] = \frac{1}{K}\sum_{k=1}^{K} x_k[n]$$

평균화는 시간 잠금된 반응을 보존하면서 배경 잡음을 $\sqrt{K}$만큼 줄입니다.

### 16.5 EEG를 위한 스펙트럼 분석 방법

- **웰치 방법(Welch's method)**: 강인한 PSD 추정을 위한 평균화된 주기도
- **다중 테이퍼 방법(Multitaper method)**: 분산 감소를 위한 여러 직교 테이퍼
- **단시간 푸리에 변환(Short-Time Fourier Transform)**: 시변 스펙트럼 내용(사건 관련 스펙트럼 섭동)
- **웨이블릿 분석(Wavelet analysis)**: 일시적 뇌 사건의 다중 스케일 시간-주파수 분석

---

## 17. 심박 변이도(HRV)

### 17.1 HRV란?

심박 변이도(Heart Rate Variability, HRV)는 연속 심박 사이의 시간 간격(RR 간격)의 변화입니다. HRV는 자율 신경계 기능의 지표입니다.

### 17.2 시간 영역 측정값

RR 간격 시퀀스 $\{RR_i\}$에서:

| 측정값 | 공식 | 의미 |
|---|---|---|
| SDNN | $\sqrt{\frac{1}{N}\sum(RR_i - \overline{RR})^2}$ | 전체 변이도 |
| RMSSD | $\sqrt{\frac{1}{N-1}\sum(RR_{i+1} - RR_i)^2}$ | 단기 변이도 |
| pNN50 | $\frac{\#\{|RR_{i+1} - RR_i| > 50\text{ms}\}}{N-1} \times 100\%$ | 부교감 신경 활동 |

### 17.3 주파수 영역 측정값

RR 간격 시계열의 PSD는 자율 조절을 드러냅니다:

| 대역 | 주파수 | 기원 |
|---|---|---|
| VLF | 0.003-0.04 Hz | 체온 조절, RAAS |
| LF | 0.04-0.15 Hz | 교감 + 부교감 신경 |
| HF | 0.15-0.40 Hz | 부교감 신경(호흡성 동성 부정맥) |

**LF/HF 비율**은 교감-미주 신경 균형의 (논란이 있는) 지표입니다.

### 17.4 HRV 분석 파이프라인

1. **QRS 검출**: Pan-Tompkins 알고리즘
2. **RR 간격 추출**: 연속 R 피크 사이의 시간
3. **아티팩트 제거**: 이소성 박동 및 놓친 검출 제거
4. **보간**: RR 간격을 균일한 시간 격자로 리샘플링(예: 4 Hz, 큐빅 스플라인)
5. **PSD 추정**: 웰치 방법 또는 AR 모델
6. **대역 전력 계산**: VLF, LF, HF 대역에서 PSD 적분

---

## 18. Python 구현

### 18.1 디지털 오디오 이펙트

```python
import numpy as np
import matplotlib.pyplot as plt


def generate_audio_signal(duration=2.0, fs=44100):
    """Generate a test audio signal (guitar-like pluck)."""
    t = np.arange(0, duration, 1/fs)
    # Karplus-Strong synthesis approximation
    f0 = 220  # A3
    signal = np.zeros(len(t))
    # Harmonics with decay
    for k in range(1, 8):
        decay = np.exp(-k * 1.5 * t)
        signal += (1/k) * np.sin(2*np.pi*k*f0*t) * decay
    signal = signal / np.max(np.abs(signal))
    return t, signal


def delay_effect(x, fs, delay_ms=300, feedback=0.5, mix=0.5):
    """Apply delay/echo effect."""
    delay_samples = int(delay_ms * fs / 1000)
    y = np.zeros(len(x) + delay_samples * 5)
    y[:len(x)] = x.copy()

    for i in range(delay_samples, len(y)):
        y[i] += feedback * y[i - delay_samples]

    return mix * y[:len(x)] + (1 - mix) * x


def chorus_effect(x, fs, depth_ms=3, rate_hz=1.5, mix=0.5):
    """Apply chorus effect with LFO-modulated delay."""
    N = len(x)
    t = np.arange(N) / fs
    base_delay = int(25 * fs / 1000)  # 25 ms base delay
    depth = int(depth_ms * fs / 1000)

    y = np.zeros(N)
    for n in range(base_delay + depth, N):
        # LFO modulates the delay
        mod = depth * np.sin(2 * np.pi * rate_hz * t[n])
        delay = base_delay + int(mod)

        # Linear interpolation for fractional delay
        d_int = int(delay)
        d_frac = delay - d_int

        if n - d_int - 1 >= 0:
            delayed = (1 - d_frac) * x[n - d_int] + d_frac * x[n - d_int - 1]
        else:
            delayed = x[n - d_int]

        y[n] = (1 - mix) * x[n] + mix * delayed

    return y


def compressor(x, fs, threshold_db=-20, ratio=4, attack_ms=5, release_ms=50):
    """Apply dynamic range compression."""
    N = len(x)
    threshold = 10**(threshold_db / 20)
    alpha_a = np.exp(-1 / (fs * attack_ms / 1000))
    alpha_r = np.exp(-1 / (fs * release_ms / 1000))

    env = np.zeros(N)
    gain = np.ones(N)
    y = np.zeros(N)

    for n in range(1, N):
        # Envelope follower
        abs_x = np.abs(x[n])
        if abs_x > env[n-1]:
            env[n] = alpha_a * env[n-1] + (1 - alpha_a) * abs_x
        else:
            env[n] = alpha_r * env[n-1] + (1 - alpha_r) * abs_x

        # Compute gain
        if env[n] > threshold:
            env_db = 20 * np.log10(env[n] + 1e-10)
            thresh_db = 20 * np.log10(threshold)
            gain_db = thresh_db + (env_db - thresh_db) / ratio - env_db
            gain[n] = 10**(gain_db / 20)
        else:
            gain[n] = 1.0

        y[n] = gain[n] * x[n]

    return y, gain


# Demo
fs = 44100
t, x = generate_audio_signal(2.0, fs)

# Apply effects
x_delay = delay_effect(x, fs, delay_ms=200, feedback=0.4)
x_chorus = chorus_effect(x, fs, depth_ms=3, rate_hz=1.5)
x_comp, gain = compressor(x, fs, threshold_db=-12, ratio=4)

# Plot
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

axes[0].plot(t[:4000], x[:4000], 'b', alpha=0.7)
axes[0].set_title('Original Signal')
axes[0].set_ylabel('Amplitude')

axes[1].plot(t[:4000], x_delay[:4000], 'r', alpha=0.7)
axes[1].set_title('Delay Effect (200ms, feedback=0.4)')
axes[1].set_ylabel('Amplitude')

axes[2].plot(t[:4000], x_chorus[:4000], 'g', alpha=0.7)
axes[2].set_title('Chorus Effect')
axes[2].set_ylabel('Amplitude')

axes[3].plot(t[:4000], x[:4000], 'b', alpha=0.3, label='Original')
axes[3].plot(t[:4000], x_comp[:4000], 'r', alpha=0.7, label='Compressed')
axes[3].set_title('Compressor (threshold=-12dB, ratio=4:1)')
axes[3].set_ylabel('Amplitude')
axes[3].set_xlabel('Time (s)')
axes[3].legend()

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('audio_effects.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 18.2 디지털 변조

```python
import numpy as np
import matplotlib.pyplot as plt


def qpsk_modulate(bits, samples_per_symbol=20):
    """QPSK modulation."""
    # Ensure even number of bits
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)

    n_symbols = len(bits) // 2
    symbols = np.zeros(n_symbols, dtype=complex)

    # Gray coding: 00->pi/4, 01->3pi/4, 11->5pi/4, 10->7pi/4
    mapping = {(0, 0): np.exp(1j * np.pi/4),
               (0, 1): np.exp(1j * 3*np.pi/4),
               (1, 1): np.exp(1j * 5*np.pi/4),
               (1, 0): np.exp(1j * 7*np.pi/4)}

    for i in range(n_symbols):
        dibit = (bits[2*i], bits[2*i+1])
        symbols[i] = mapping[dibit]

    # Upsample
    signal = np.zeros(n_symbols * samples_per_symbol, dtype=complex)
    signal[::samples_per_symbol] = symbols

    # Pulse shaping (raised cosine)
    t = np.arange(-4*samples_per_symbol, 4*samples_per_symbol + 1)
    alpha = 0.35
    Ts = samples_per_symbol
    rc_pulse = np.sinc(t/Ts) * np.cos(np.pi*alpha*t/Ts) / (1 - (2*alpha*t/Ts)**2 + 1e-10)
    rc_pulse /= np.sqrt(np.sum(rc_pulse**2))

    signal = np.convolve(signal, rc_pulse, mode='same')
    return symbols, signal


def qam16_constellation():
    """Generate 16-QAM constellation."""
    levels = [-3, -1, 1, 3]
    constellation = []
    for i in levels:
        for q in levels:
            constellation.append(complex(i, q))
    return np.array(constellation) / np.sqrt(10)  # Normalize


# Generate random bits
np.random.seed(42)
bits = np.random.randint(0, 2, 200)

# QPSK modulation
symbols, signal = qpsk_modulate(bits)

# Add AWGN noise at different SNR levels
snr_values = [5, 15, 25]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, snr_db in enumerate(snr_values):
    noise_power = 10**(-snr_db/10)
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(symbols))
                                       + 1j * np.random.randn(len(symbols)))
    received = symbols + noise

    # Constellation diagram
    ax = axes[0, idx]
    ax.scatter(received.real, received.imag, s=10, alpha=0.5, label='Received')
    ax.scatter(symbols.real, symbols.imag, s=100, c='red', marker='x',
               linewidths=2, label='Ideal')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'QPSK: SNR = {snr_db} dB')
    ax.set_xlabel('In-phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.legend(fontsize=8)

# 16-QAM constellation
qam_const = qam16_constellation()
for idx, snr_db in enumerate(snr_values):
    noise_power = 10**(-snr_db/10)
    n_sym = 500
    # Random symbols from constellation
    sym_idx = np.random.randint(0, 16, n_sym)
    tx_symbols = qam_const[sym_idx]
    noise = np.sqrt(noise_power/2) * (np.random.randn(n_sym)
                                       + 1j * np.random.randn(n_sym))
    rx_symbols = tx_symbols + noise

    ax = axes[1, idx]
    ax.scatter(rx_symbols.real, rx_symbols.imag, s=5, alpha=0.3, label='Received')
    ax.scatter(qam_const.real, qam_const.imag, s=100, c='red', marker='x',
               linewidths=2, label='Ideal')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'16-QAM: SNR = {snr_db} dB')
    ax.set_xlabel('In-phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('digital_modulation.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 18.3 레이더: 처프 펄스 압축

```python
import numpy as np
import matplotlib.pyplot as plt


def generate_chirp(tau_p, B, fs):
    """Generate a linear FM chirp pulse."""
    t = np.arange(0, tau_p, 1/fs)
    chirp_rate = B / tau_p
    s = np.exp(1j * np.pi * chirp_rate * t**2)
    return t, s


def matched_filter(received, template):
    """Apply matched filter (cross-correlation)."""
    mf = np.conj(template[::-1])
    output = np.convolve(received, mf, mode='full')
    return output


# Radar parameters
c = 3e8           # Speed of light (m/s)
fc = 10e9         # Carrier frequency (10 GHz, X-band)
tau_p = 10e-6     # Pulse width (10 μs)
B = 5e6           # Bandwidth (5 MHz)
fs = 20e6         # Sampling frequency

# Range resolution
delta_R = c / (2 * B)
print(f"Range resolution: {delta_R:.1f} m")
print(f"Time-bandwidth product: {B * tau_p:.0f}")
print(f"Processing gain: {10*np.log10(B*tau_p):.1f} dB")

# Generate chirp
t_chirp, chirp = generate_chirp(tau_p, B, fs)

# Simulate two targets at different ranges
R1 = 5000    # Target 1 at 5 km
R2 = 5060    # Target 2 at 5.06 km (60 m apart)
A1 = 1.0     # Target 1 amplitude
A2 = 0.5     # Target 2 amplitude (weaker)

# Convert ranges to time delays
tau1 = 2 * R1 / c
tau2 = 2 * R2 / c

# Create received signal
N_total = int(2 * 15000 / c * fs)  # Enough samples for 15 km range
received = np.zeros(N_total, dtype=complex)

# Add noise
noise_level = 0.3
received += noise_level * (np.random.randn(N_total) + 1j * np.random.randn(N_total))

# Add target returns
idx1 = int(tau1 * fs)
idx2 = int(tau2 * fs)
if idx1 + len(chirp) < N_total:
    received[idx1:idx1+len(chirp)] += A1 * chirp
if idx2 + len(chirp) < N_total:
    received[idx2:idx2+len(chirp)] += A2 * chirp

# Matched filter output
mf_output = matched_filter(received, chirp)

# Convert to range
range_axis = np.arange(len(mf_output)) * c / (2 * fs) / 1000  # km

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Chirp waveform
axes[0].plot(t_chirp * 1e6, chirp.real, 'b', alpha=0.7)
axes[0].set_xlabel('Time (μs)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title(f'Linear FM Chirp Pulse (B={B/1e6:.0f} MHz, τ={tau_p*1e6:.0f} μs)')
axes[0].grid(True, alpha=0.3)

# Chirp spectrogram
from scipy import signal
f_stft, t_stft, Sxx = signal.spectrogram(
    chirp.real, fs, nperseg=64, noverlap=56, nfft=256
)
axes[1].pcolormesh(t_stft*1e6, f_stft/1e6, 10*np.log10(Sxx + 1e-10),
                   shading='gouraud', cmap='viridis')
axes[1].set_xlabel('Time (μs)')
axes[1].set_ylabel('Frequency (MHz)')
axes[1].set_title('Chirp Spectrogram')

# Matched filter output
mf_db = 20 * np.log10(np.abs(mf_output) / np.max(np.abs(mf_output)) + 1e-10)
mask = (range_axis > 4.5) & (range_axis < 5.5)
axes[2].plot(range_axis[mask], mf_db[mask], 'b', linewidth=1.5)
axes[2].axvline(x=R1/1000, color='r', linestyle='--', alpha=0.5,
                label=f'Target 1: {R1/1000:.1f} km')
axes[2].axvline(x=R2/1000, color='g', linestyle='--', alpha=0.5,
                label=f'Target 2: {R2/1000:.3f} km')
axes[2].set_xlabel('Range (km)')
axes[2].set_ylabel('Amplitude (dB)')
axes[2].set_title(f'Matched Filter Output (ΔR = {delta_R:.0f} m)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([-40, 5])

plt.tight_layout()
plt.savefig('radar_pulse_compression.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 18.4 생체의학: ECG 처리

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def generate_synthetic_ecg(duration=10, fs=360, heart_rate=72):
    """
    Generate a synthetic ECG signal.

    Parameters
    ----------
    duration : float
        Duration in seconds
    fs : int
        Sampling rate
    heart_rate : int
        Heart rate in bpm

    Returns
    -------
    t : ndarray
        Time axis
    ecg : ndarray
        Synthetic ECG signal
    r_peaks : ndarray
        Indices of R-peaks
    """
    t = np.arange(0, duration, 1/fs)
    N = len(t)
    ecg = np.zeros(N)
    rr_interval = 60 / heart_rate  # seconds
    rr_samples = int(rr_interval * fs)

    r_peaks = []

    # Generate each heartbeat
    beat_start = int(0.1 * fs)
    while beat_start + rr_samples < N:
        # P wave
        p_center = beat_start + int(0.16 * rr_samples)
        p_width = int(0.04 * fs)
        t_local = np.arange(-3*p_width, 3*p_width+1)
        p_wave = 0.15 * np.exp(-t_local**2 / (2*p_width**2))
        start = max(0, p_center - 3*p_width)
        end = min(N, p_center + 3*p_width + 1)
        ecg[start:end] += p_wave[:end-start]

        # QRS complex
        r_center = beat_start + int(0.25 * rr_samples)
        r_peaks.append(r_center)

        # Q wave
        q_center = r_center - int(0.02 * fs)
        q_width = int(0.01 * fs)
        t_local = np.arange(-3*q_width, 3*q_width+1)
        q_wave = -0.1 * np.exp(-t_local**2 / (2*q_width**2))
        start = max(0, q_center - 3*q_width)
        end = min(N, q_center + 3*q_width + 1)
        ecg[start:end] += q_wave[:end-start]

        # R wave
        r_width = int(0.008 * fs)
        t_local = np.arange(-3*r_width, 3*r_width+1)
        r_wave = 1.0 * np.exp(-t_local**2 / (2*r_width**2))
        start = max(0, r_center - 3*r_width)
        end = min(N, r_center + 3*r_width + 1)
        ecg[start:end] += r_wave[:end-start]

        # S wave
        s_center = r_center + int(0.02 * fs)
        s_width = int(0.012 * fs)
        t_local = np.arange(-3*s_width, 3*s_width+1)
        s_wave = -0.2 * np.exp(-t_local**2 / (2*s_width**2))
        start = max(0, s_center - 3*s_width)
        end = min(N, s_center + 3*s_width + 1)
        ecg[start:end] += s_wave[:end-start]

        # T wave
        t_center = beat_start + int(0.5 * rr_samples)
        t_width = int(0.06 * fs)
        t_local = np.arange(-3*t_width, 3*t_width+1)
        t_wave = 0.3 * np.exp(-t_local**2 / (2*t_width**2))
        start = max(0, t_center - 3*t_width)
        end = min(N, t_center + 3*t_width + 1)
        ecg[start:end] += t_wave[:end-start]

        # Add some RR variability
        rr_var = rr_samples + int(np.random.randn() * 0.02 * rr_samples)
        beat_start += rr_var

    return t, ecg, np.array(r_peaks)


def pan_tompkins_qrs(ecg, fs):
    """
    Simplified Pan-Tompkins QRS detector.

    Parameters
    ----------
    ecg : ndarray
        ECG signal
    fs : int
        Sampling rate

    Returns
    -------
    r_peaks : ndarray
        Detected R-peak indices
    """
    # Step 1: Bandpass filter (5-15 Hz)
    nyq = fs / 2
    b_bp, a_bp = signal.butter(2, [5/nyq, 15/nyq], btype='band')
    filtered = signal.filtfilt(b_bp, a_bp, ecg)

    # Step 2: Derivative
    h_diff = np.array([-1, -2, 0, 2, 1]) / 8
    diff = np.convolve(filtered, h_diff, mode='same')

    # Step 3: Squaring
    squared = diff**2

    # Step 4: Moving window integration (150 ms)
    win_size = int(0.15 * fs)
    integrator = np.convolve(squared, np.ones(win_size)/win_size, mode='same')

    # Step 5: Thresholding
    threshold = 0.4 * np.max(integrator)
    peaks_mask = integrator > threshold

    # Find peaks
    r_peaks = []
    min_distance = int(0.3 * fs)  # Minimum 300 ms between beats

    i = 0
    while i < len(peaks_mask):
        if peaks_mask[i]:
            # Find the actual R-peak (maximum in the raw ECG)
            start = max(0, i - win_size//2)
            end = min(len(ecg), i + win_size//2)
            r_idx = start + np.argmax(ecg[start:end])

            if len(r_peaks) == 0 or (r_idx - r_peaks[-1]) > min_distance:
                r_peaks.append(r_idx)
            i = r_idx + min_distance
        else:
            i += 1

    return np.array(r_peaks)


# Generate and process ECG
np.random.seed(42)
fs = 360
t, clean_ecg, true_r_peaks = generate_synthetic_ecg(duration=10, fs=fs, heart_rate=72)

# Add noise
baseline_wander = 0.1 * np.sin(2*np.pi*0.15*t) + 0.05*np.sin(2*np.pi*0.3*t)
powerline = 0.05 * np.sin(2*np.pi*60*t)
muscle_noise = 0.03 * np.random.randn(len(t))
noisy_ecg = clean_ecg + baseline_wander + powerline + muscle_noise

# Preprocessing
# Baseline removal (highpass 0.5 Hz)
b_hp, a_hp = signal.butter(2, 0.5/(fs/2), btype='high')
ecg_no_baseline = signal.filtfilt(b_hp, a_hp, noisy_ecg)

# Notch filter for 60 Hz
b_notch, a_notch = signal.iirnotch(60, 30, fs)
ecg_notched = signal.filtfilt(b_notch, a_notch, ecg_no_baseline)

# Bandpass 0.5-40 Hz
b_bp, a_bp = signal.butter(3, [0.5/(fs/2), 40/(fs/2)], btype='band')
ecg_filtered = signal.filtfilt(b_bp, a_bp, noisy_ecg)

# QRS detection
detected_r_peaks = pan_tompkins_qrs(ecg_filtered, fs)

# Plot results
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

axes[0].plot(t, clean_ecg, 'g', alpha=0.7)
axes[0].plot(t[true_r_peaks], clean_ecg[true_r_peaks], 'rv', markersize=10)
axes[0].set_title('Clean ECG with True R-peaks')
axes[0].set_ylabel('Amplitude (mV)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, noisy_ecg, 'r', alpha=0.5)
axes[1].set_title('Noisy ECG (baseline wander + 60Hz + muscle noise)')
axes[1].set_ylabel('Amplitude (mV)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, ecg_filtered, 'b', alpha=0.7)
axes[2].set_title('Filtered ECG (0.5-40 Hz bandpass)')
axes[2].set_ylabel('Amplitude (mV)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(t, ecg_filtered, 'b', alpha=0.5)
axes[3].plot(t[detected_r_peaks], ecg_filtered[detected_r_peaks],
             'rv', markersize=10, label='Detected R-peaks')
axes[3].set_title(f'QRS Detection (Pan-Tompkins): {len(detected_r_peaks)} beats detected')
axes[3].set_xlabel('Time (s)')
axes[3].set_ylabel('Amplitude (mV)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ecg_processing.png', dpi=150, bbox_inches='tight')
plt.show()

# HRV Analysis
rr_intervals = np.diff(detected_r_peaks) / fs * 1000  # Convert to ms
print(f"\nHRV Time-Domain Measures:")
print(f"  Mean RR: {np.mean(rr_intervals):.1f} ms")
print(f"  SDNN:    {np.std(rr_intervals):.1f} ms")
print(f"  RMSSD:   {np.sqrt(np.mean(np.diff(rr_intervals)**2)):.1f} ms")
print(f"  Mean HR: {60000/np.mean(rr_intervals):.1f} bpm")
```

### 18.5 피치 검출 시연

```python
import numpy as np
import matplotlib.pyplot as plt


def autocorrelation_pitch(frame, fs, fmin=80, fmax=500):
    """
    Pitch detection using autocorrelation.

    Parameters
    ----------
    frame : ndarray
        Signal frame
    fs : float
        Sampling rate
    fmin, fmax : float
        Min/max expected pitch frequency

    Returns
    -------
    f0 : float
        Estimated pitch frequency (0 if unvoiced)
    """
    N = len(frame)
    # Compute autocorrelation
    r = np.correlate(frame, frame, mode='full')
    r = r[N-1:]  # Take positive lags only
    r = r / r[0]  # Normalize

    # Search range
    lag_min = int(fs / fmax)
    lag_max = min(int(fs / fmin), N - 1)

    # Find the first significant peak
    if lag_max >= len(r):
        lag_max = len(r) - 1

    r_search = r[lag_min:lag_max+1]
    if len(r_search) == 0:
        return 0

    peak_idx = np.argmax(r_search) + lag_min

    # Voiced/unvoiced decision
    if r[peak_idx] > 0.3:  # Threshold
        return fs / peak_idx
    else:
        return 0


def cepstrum_pitch(frame, fs, fmin=80, fmax=500):
    """Pitch detection using cepstrum."""
    N = len(frame)
    # Compute cepstrum
    spectrum = np.fft.fft(frame, n=2*N)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.real(np.fft.ifft(log_spectrum))

    # Search range (quefrency = period in samples)
    q_min = int(fs / fmax)
    q_max = min(int(fs / fmin), N - 1)

    cep_search = cepstrum[q_min:q_max+1]
    if len(cep_search) == 0:
        return 0

    peak_idx = np.argmax(cep_search) + q_min

    if cepstrum[peak_idx] > 0.1:
        return fs / peak_idx
    else:
        return 0


# Generate a test signal with known pitch
fs = 16000
duration = 1.0
t = np.arange(0, duration, 1/fs)

# Create a signal with time-varying pitch (glissando)
f0_start = 150
f0_end = 300
phase = 2 * np.pi * (f0_start * t + (f0_end - f0_start) * t**2 / (2 * duration))
signal_test = np.zeros(len(t))
for k in range(1, 6):
    signal_test += (1/k) * np.sin(k * phase)
signal_test = signal_test / np.max(np.abs(signal_test))
signal_test += 0.05 * np.random.randn(len(t))  # Add noise

# Frame-by-frame pitch detection
frame_length = int(0.04 * fs)  # 40 ms
hop = int(0.01 * fs)  # 10 ms
n_frames = (len(signal_test) - frame_length) // hop

pitches_ac = np.zeros(n_frames)
pitches_cep = np.zeros(n_frames)
frame_times = np.zeros(n_frames)

window = np.hanning(frame_length)

for i in range(n_frames):
    start = i * hop
    frame = signal_test[start:start+frame_length] * window
    frame_times[i] = (start + frame_length/2) / fs

    pitches_ac[i] = autocorrelation_pitch(frame, fs)
    pitches_cep[i] = cepstrum_pitch(frame, fs)

# True pitch trajectory
true_pitch = f0_start + (f0_end - f0_start) * frame_times / duration

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(t, signal_test, 'b', alpha=0.5)
axes[0].set_title('Test Signal (Glissando 150-300 Hz)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

axes[1].plot(frame_times, true_pitch, 'k-', linewidth=2, label='True pitch')
axes[1].plot(frame_times, pitches_ac, 'ro', markersize=3, alpha=0.7,
             label='Autocorrelation')
axes[1].plot(frame_times, pitches_cep, 'bx', markersize=3, alpha=0.7,
             label='Cepstrum')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_title('Pitch Detection Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 500])

# Autocorrelation and cepstrum of one frame
frame_idx = n_frames // 2
start = frame_idx * hop
frame = signal_test[start:start+frame_length] * window

r = np.correlate(frame, frame, mode='full')
r = r[len(frame)-1:]
r = r / r[0]

spectrum = np.fft.fft(frame, n=2*frame_length)
cepstrum_vals = np.real(np.fft.ifft(np.log(np.abs(spectrum) + 1e-10)))

lag = np.arange(len(r)) / fs * 1000  # ms
axes[2].plot(lag[:int(fs/80)], r[:int(fs/80)], 'b', alpha=0.7,
             label='Autocorrelation')
axes[2].set_xlabel('Lag (ms)')
axes[2].set_ylabel('Normalized Autocorrelation')
axes[2].set_title('Autocorrelation of Middle Frame')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pitch_detection.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 19. 연습 문제

### 연습 1: 오디오 이펙트 체인

(a) 4개의 병렬 빗살 필터(지연 29.7, 37.1, 41.1, 43.7 ms, 이득 0.742, 0.733, 0.715, 0.697)와 2개의 직렬 전역 통과 필터(지연 5.0, 1.7 ms, 이득 0.7)를 사용하여 슈뢰더 리버브레이터를 구현하십시오. 짧은 임펄스에 적용하고 결과 임펄스 응답을 플로팅하십시오.

(b) 간단한 멜로디(예: 적절한 주파수의 사인파를 사용한 "반짝반짝 작은 별")를 생성하십시오. 리버브를 적용하고 들어보십시오(가능하다면 WAV 파일로 저장).

(c) 플랜저 이펙트를 구현하고, 시변 크기 응답을 플로팅하여 특징적인 빗살 필터 스윕을 시연하십시오.

### 연습 2: 피치 검출 강인성

(a) 220 Hz의 순음(pure tone)을 생성하고 자기상관과 켑스트럼 방법 모두 올바르게 검출하는지 확인하십시오.

(b) 고조파 내용(감소하는 진폭을 가진 1번째부터 5번째 고조파)을 추가하십시오. 피치 검출기가 여전히 기본파를 찾는지 확인하십시오.

(c) SNR = 20, 10, 5, 0 dB에서 잡음을 추가하십시오. 두 방법에 대해 SNR 대 피치 검출 정확도를 플로팅하십시오.

(d) 빠진 기본파를 가진 신호(100 Hz의 고조파 2, 3, 4만 포함)를 만드십시오. 자기상관 방법이 여전히 100 Hz를 검출할 수 있습니까? 켑스트럼은 어떻습니까?

### 연습 3: 디지털 변조 BER

(a) BPSK, QPSK, 16-QAM 변조 및 복조를 구현하십시오.

(b) $E_b/N_0$를 0에서 20 dB까지 AWGN 채널을 통한 전송을 시뮬레이션하십시오. 세 가지 방식 모두의 BER 곡선을 플로팅하십시오.

(c) 이론적 BER과 비교하십시오:
- BPSK: $P_b = Q(\sqrt{2E_b/N_0})$
- QPSK: BPSK와 동일(비트당)
- 16-QAM: $P_b \approx \frac{3}{8}Q(\sqrt{4E_b/(5N_0)})$

(d) 16-QAM에 그레이 코딩(Gray coding)을 구현하고 자연 이진 매핑에 비해 BER이 개선됨을 보이십시오.

### 연습 4: OFDM 시스템

(a) 64개 부반송파, 16-QAM 변조, 길이 16의 순환 전치를 가진 기본 OFDM 송신기와 수신기를 구현하십시오.

(b) AWGN을 포함한 다중경로 채널 $h = [1, 0, 0.5, 0, 0.2]$를 통한 전송을 시뮬레이션하십시오.

(c) 순환 전치가 단순한 단일 탭 등화를 가능하게 함을 보이십시오. 등화 전후의 성상도를 플로팅하십시오.

(d) 순환 전치를 제거하고 결과로 나타나는 ISI를 시연하십시오.

### 연습 5: 레이더 파형 설계

(a) 시간-대역폭 곱 10, 50, 200의 처프 펄스를 생성하십시오. 정합 필터링을 적용하고 압축된 펄스 폭과 사이드로브 레벨을 비교하십시오.

(b) 정합 필터에 해밍 윈도우를 적용하십시오. 메인로브 폭과 사이드로브 레벨에 어떤 영향을 미칩니까?

(c) 10 km와 10.03 km 거리에 두 표적이 있는 시나리오를 시뮬레이션하십시오. 각 TBP에서 표적이 분해되는지 확인하십시오.

(d) TBP = 50인 처프 펄스에 대해 2D 모호 함수 $|\chi(\tau, f_d)|$를 계산하고 플로팅하십시오. 거리-도플러 결합 능선을 식별하십시오.

### 연습 6: ECG 분석 파이프라인

(a) 심박수 75 bpm의 합성 ECG를 생성하고 다음을 추가하십시오:
- 기저선 방랑(0.2 Hz 정현파, 진폭 0.3 mV)
- 50 Hz 전원선 잡음(진폭 0.1 mV)
- 랜덤 잡음(SNR = 20 dB)

(b) 전체 전처리 파이프라인을 구현하십시오: 고역 통과 필터(0.5 Hz), 노치 필터(50 Hz), 대역 통과(1-40 Hz). 각 단계 후의 신호를 보이십시오.

(c) Pan-Tompkins QRS 검출기를 구현하십시오. 검출된 박동 수를 세고 실제 값과 비교하십시오.

(d) RR 간격을 추출하고 SDNN, RMSSD, pNN50을 계산하십시오. 알려진 심박 변이도와 비교하십시오.

(e) HRV 전력 스펙트럼을 계산하고 LF 및 HF 대역을 식별하십시오.

### 연습 7: EEG 스펙트럼 분석

(a) EEG 신호를 다음의 합으로 시뮬레이션하십시오:
- 알파 대역(10 Hz): 눈을 감았을 때 지배적
- 베타 대역(20 Hz): 집중 시 존재
- 배경 잡음(1/f 스펙트럼)

(b) 웰치 방법(2초 윈도우, 50% 오버랩)을 사용하여 전력 스펙트럼 밀도를 계산하십시오. 알파 및 베타 피크를 식별하십시오.

(c) "눈 뜨기/감기" 실험을 시뮬레이션하십시오: 알파 전력이 $t = 2$-4 s(눈 감음) 동안 증가하고 다른 곳에서는 감소합니다. STFT를 사용하여 시변 알파 대역 전력을 보이십시오.

(d) 슬라이딩 윈도우를 사용하여 시간에 따른 상대 대역 전력(알파/전체, 베타/전체)을 계산하십시오.

---

## 20. 요약

| 분야 | 핵심 기법 | 신호 처리 도구 |
|---|---|---|
| 오디오 | 지연 이펙트, 리버브 | 지연 라인, 빗살/전역 통과 필터 |
| 오디오 | 피치 검출 | 자기상관, 켑스트럼 |
| 오디오 | 이퀄라이제이션 | 파라메트릭 바이쿼드 필터 |
| 오디오 | 음성 코딩 | 선형 예측(LPC) |
| 통신 | 아날로그 변조 | AM, FM, PM |
| 통신 | 디지털 변조 | PSK, QAM(I-Q 처리) |
| 통신 | 펄스 성형 | 상승 코사인, 정합 필터 |
| 통신 | OFDM | FFT/IFFT, 순환 전치 |
| 통신 | 등화 | ZF, MMSE, 적응(LMS/RLS) |
| 레이더 | 거리 검출 | 정합 필터 |
| 레이더 | 펄스 압축 | 처프, 정합 필터 |
| 레이더 | 파형 설계 | 모호 함수 |
| 생체의학 | ECG 전처리 | 대역 통과, 노치 필터 |
| 생체의학 | QRS 검출 | Pan-Tompkins 알고리즘 |
| 생체의학 | EEG 분석 | PSD, 대역 전력, STFT |
| 생체의학 | HRV | 시간/주파수 영역 측정값 |

**핵심 요점**:
1. 오디오 이펙트는 기본 빌딩 블록인 지연 라인, 필터, 변조기로 구성됩니다.
2. 피치 검출은 자기상관을 통한 주기성 찾기 또는 켑스트럼을 통한 스펙트럼 분석으로 귀결됩니다.
3. 디지털 변조는 비트를 I-Q 평면의 점에 매핑합니다; 고차 방식은 스펙트럼 효율을 위해 SNR을 교환합니다.
4. OFDM은 FFT를 사용하여 주파수 선택적 채널을 병렬 평탄 채널로 변환합니다.
5. 레이더 펄스 압축은 높은 에너지(긴 펄스)와 미세한 거리 해상도(넓은 대역폭) 모두를 달성합니다.
6. 모호 함수는 레이더 파형의 거리-속도 해상도를 완전히 특성화합니다.
7. ECG 처리는 신뢰할 수 있는 QRS 검출을 위해 대역 통과 필터링, 노치 필터링, Pan-Tompkins 알고리즘을 결합합니다.
8. EEG 스펙트럼 분석은 주파수 대역에 걸친 전력 분포를 통해 뇌 상태를 드러냅니다.
9. 이 레슨의 모든 응용은 레슨 1-15의 기초 위에 직접 구축됩니다.

---

## 21. 참고 문헌

1. J.O. Smith III, *Physical Audio Signal Processing*, W3K Publishing, 2010. (온라인 제공)
2. J.G. Proakis and M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008.
3. M.A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., McGraw-Hill, 2014.
4. M.I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill, 2001.
5. J. Pan and W.J. Tompkins, "A real-time QRS detection algorithm," *IEEE Trans. Biomedical Engineering*, vol. BME-32, no. 3, pp. 230-236, 1985.
6. R. Rangayyan, *Biomedical Signal Analysis*, 2nd ed., Wiley-IEEE Press, 2015.
7. S. Haykin and M. Moher, *Communication Systems*, 5th ed., Wiley, 2009.
8. A.V. Oppenheim and R.W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010.
9. U.R. Acharya et al., "Heart rate variability: a review," *Medical and Biological Engineering and Computing*, vol. 44, pp. 1031-1051, 2006.

---

**이전**: [15. 영상 신호 처리](./15_Image_Signal_Processing.md) | [개요](./00_Overview.md)
