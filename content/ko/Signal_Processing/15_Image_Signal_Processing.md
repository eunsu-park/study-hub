# 15. 영상 신호 처리(Image Signal Processing)

**이전**: [14. 시간-주파수 분석](./14_Time_Frequency_Analysis.md) | **다음**: [16. 응용](./16_Applications.md)

---

영상(Image)은 2차원 신호입니다. 시간이 아닌 공간 좌표의 함수로, 1D 신호 처리의 거의 모든 개념 — 합성곱(Convolution), 푸리에 변환(Fourier Transform), 필터링(Filtering), 샘플링(Sampling) — 은 자연스럽게 2D로 확장됩니다. 이 레슨은 영상 분석의 신호 처리 기초를 다루며, 공간 및 주파수 영역 연산, 에지 검출(Edge Detection), 화질 향상(Enhancement), 압축(Compression)을 포함합니다. NumPy와 SciPy를 사용하며, 고수준 컴퓨터 비전 API보다 신호 처리 원리에 집중합니다.

**난이도**: ⭐⭐⭐

**선수 과목**: 1D DFT, 합성곱, FIR 필터 설계, 샘플링 정리

**학습 목표**:
- 영상을 2D 이산 신호로 표현하고 샘플링 구조를 이해하기
- 2D 이산 푸리에 변환(DFT)을 계산하고 해석하기
- 공간 영역 필터링(평활화, 선명화, 중앙값)을 위한 2D 합성곱 적용하기
- 주파수 영역 필터(저역통과, 고역통과, 대역통과) 설계 및 적용하기
- 캐니(Canny) 알고리즘을 포함한 그래디언트 기반 및 라플라시안(Laplacian) 에지 검출기 구현하기
- 영상 향상을 위한 히스토그램 평활화(Histogram Equalization) 및 대비 스트레칭(Contrast Stretching) 적용하기
- DCT 기반 JPEG 압축 파이프라인 이해하기
- 2D에서의 샘플링, 해상도, 나이퀴스트(Nyquist) 기준 설명하기

---

## 목차

1. [영상을 2D 신호로](#1-영상을-2d-신호로)
2. [2D 이산 푸리에 변환](#2-2d-이산-푸리에-변환)
3. [2D 합성곱](#3-2d-합성곱)
4. [공간 영역 필터링: 평활화](#4-공간-영역-필터링-평활화)
5. [공간 영역 필터링: 선명화](#5-공간-영역-필터링-선명화)
6. [중앙값 필터링](#6-중앙값-필터링)
7. [주파수 영역 필터링: 저역통과](#7-주파수-영역-필터링-저역통과)
8. [주파수 영역 필터링: 고역통과 및 대역통과](#8-주파수-영역-필터링-고역통과-및-대역통과)
9. [에지 검출: 그래디언트 연산자](#9-에지-검출-그래디언트-연산자)
10. [에지 검출: 라플라시안과 LoG](#10-에지-검출-라플라시안과-log)
11. [캐니 에지 검출기](#11-캐니-에지-검출기)
12. [영상 향상](#12-영상-향상)
13. [영상 압축: JPEG 개요](#13-영상-압축-jpeg-개요)
14. [2D 샘플링](#14-2d-샘플링)
15. [Python 구현](#15-python-구현)
16. [연습 문제](#16-연습-문제)
17. [요약](#17-요약)
18. [참고 문헌](#18-참고-문헌)

---

## 1. 영상을 2D 신호로

### 1.1 디지털 영상 표현

그레이스케일(Grayscale) 디지털 영상은 2D 함수 $f[m, n]$이며, 여기서:
- $m \in \{0, 1, \ldots, M-1\}$은 행 인덱스(수직 좌표)
- $n \in \{0, 1, \ldots, N-1\}$은 열 인덱스(수평 좌표)
- $f[m, n] \in [0, 255]$ (8비트 영상의 경우, 0 = 검정, 255 = 흰색)

컬러 영상은 세 개의 채널을 가집니다: $f_R[m,n]$, $f_G[m,n]$, $f_B[m,n]$.

### 1.2 영상 형성

영상은 연속 광도 필드 $f(x, y)$를 샘플링한 결과입니다:

$$f[m, n] = f(m \Delta x, n \Delta y)$$

여기서 $\Delta x$와 $\Delta y$는 공간 샘플링 간격(Spatial Sampling Interval)입니다. 역수 $1/\Delta x$와 $1/\Delta y$는 공간 샘플링 주파수입니다.

### 1.3 신호 처리 관점에서 본 영상 특성

| 특성 | 1D 신호 | 2D 영상 |
|---|---|---|
| 독립 변수 | 시간 $t$ | 공간 좌표 $(x, y)$ |
| 샘플링 | 시간적 샘플링률 $f_s$ | 공간 해상도(단위 길이당 픽셀 수) |
| 주파수 | 시간 주파수(Hz) | 공간 주파수(cycles/pixel 또는 cycles/mm) |
| 필터링 | 시간 필터 | 공간 필터(커널) |
| 변환 | 1D DFT | 2D DFT |

### 1.4 공간 주파수

공간 주파수(Spatial Frequency)는 영상에서 픽셀 값이 얼마나 빠르게 변하는지를 나타냅니다:
- **낮은 공간 주파수**: 매끄러운 영역, 점진적인 밝기 변화(하늘, 벽)
- **높은 공간 주파수**: 에지, 텍스처(Texture), 세밀한 디테일(머리카락, 텍스트, 잡음)

영상에서 사인파 패턴: $f[m,n] = \cos(2\pi u_0 m + 2\pi v_0 n)$은 공간 주파수 $u_0$ (수직, cycles/pixel)와 $v_0$ (수평, cycles/pixel)를 가집니다.

---

## 2. 2D 이산 푸리에 변환

### 2.1 정의

$M \times N$ 영상 $f[m, n]$의 **2D DFT**는:

$$\boxed{F[k, l] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} f[m, n] \, e^{-j2\pi(km/M + ln/N)}}$$

여기서 $k \in \{0, \ldots, M-1\}$이고 $l \in \{0, \ldots, N-1\}$입니다.

**역 2D DFT**는:

$$f[m, n] = \frac{1}{MN} \sum_{k=0}^{M-1} \sum_{l=0}^{N-1} F[k, l] \, e^{j2\pi(km/M + ln/N)}$$

### 2.2 분리성(Separability)

2D DFT는 **분리 가능**합니다. 행 방향 1D DFT에 이어 열 방향 1D DFT(또는 그 반대)로 계산할 수 있습니다:

$$F[k, l] = \sum_{m=0}^{M-1} \left(\sum_{n=0}^{N-1} f[m,n] \, e^{-j2\pi ln/N}\right) e^{-j2\pi km/M}$$

이를 통해 FFT를 사용하면 복잡도가 $O(M^2 N^2)$에서 $O(MN(\log M + \log N))$으로 줄어듭니다.

### 2.3 2D 스펙트럼 해석

크기 $|F[k,l]|$는 각 공간 주파수 성분의 강도를 나타냅니다:
- **중심** ($k = 0, l = 0$): DC 성분(평균 밝기)
- **수평축** ($k = 0$): 수평 주파수 성분
- **수직축** ($l = 0$): 수직 주파수 성분
- **대각선**: 대각 주파수 성분
- **중심으로부터의 거리**: 전체적인 공간 주파수 크기

위상 $\angle F[k,l]$은 특징의 공간 위치를 인코딩합니다. 놀랍게도, 위상이 크기보다 더 많은 지각 정보를 담고 있는 경우가 많습니다.

### 2.4 2D DFT의 성질

| 성질 | 공간 영역 | 주파수 영역 |
|---|---|---|
| 선형성 | $af_1 + bf_2$ | $aF_1 + bF_2$ |
| 이동 | $f[m-m_0, n-n_0]$ | $F[k,l] \, e^{-j2\pi(km_0/M + ln_0/N)}$ |
| 합성곱 | $f * g$ | $F \cdot G$ |
| 상관 | $f \star g$ | $F^* \cdot G$ |
| 파스발 정리 | $\sum|f|^2 = \frac{1}{MN}\sum|F|^2$ | (에너지 보존) |
| 켤레 대칭 | $f$ 실수 | $F[k,l] = F^*[-k,-l]$ |

### 2.5 스펙트럼 중심화

기본적으로 `np.fft.fft2`는 DC 성분을 모서리 $(0,0)$에 배치합니다. 시각화를 위해 중심화하려면 `np.fft.fftshift`를 사용하여 사분면을 교환합니다. 동등하게, DFT 전에 공간 영상에 $(-1)^{m+n}$을 곱하는 방법도 있습니다.

---

## 3. 2D 합성곱

### 3.1 정의

영상 $f[m,n]$과 커널 $h[m,n]$ (크기 $K \times L$)의 2D 이산 합성곱은:

$$\boxed{g[m, n] = (f * h)[m, n] = \sum_{i=0}^{K-1} \sum_{j=0}^{L-1} f[m-i, n-j] \, h[i, j]}$$

실제로는 대칭 커널에 대해 합성곱과 동일한 **상관(Correlation)**을 주로 사용합니다:

$$(f \star h)[m, n] = \sum_{i=0}^{K-1} \sum_{j=0}^{L-1} f[m+i-K/2, n+j-L/2] \, h[i, j]$$

### 3.2 경계 처리(Boundary Handling)

커널이 영상 경계를 벗어날 때 에지를 처리해야 합니다:

| 방법 | 설명 | 효과 |
|---|---|---|
| 제로 패딩(Zero-padding) | 외부를 0으로 가정 | 어두운 경계 |
| 반사(Reflect) | 영상을 거울 반사 | 자연스러운 모양 |
| 감싸기(Wrap) | 주기적 확장 | 원형 합성곱 |
| 복제(Replicate) | 에지 픽셀 확장 | 아티팩트 최소화 |
| 자르기(Crop) | 유효 영역만 출력 | 작은 출력 |

### 3.3 2D 합성곱 정리

$$f * h \xleftrightarrow{\text{DFT}} F \cdot H$$

주파수 영역 필터링은 큰 커널에 대해 효율적입니다:
- **공간 합성곱**: $O(MN \cdot KL)$ 연산
- **주파수 영역**: $O(MN \log(MN))$ (FFT + 점별 곱 + IFFT)

$KL > \log(MN)$이면 주파수 영역 필터링이 더 빠릅니다.

---

## 4. 공간 영역 필터링: 평활화

### 4.1 평균(박스) 필터

가장 단순한 평활화 필터는 각 픽셀을 이웃의 평균으로 대체합니다:

$$h_{avg} = \frac{1}{K^2} \begin{bmatrix} 1 & 1 & \cdots & 1 \\ 1 & 1 & \cdots & 1 \\ \vdots & & & \vdots \\ 1 & 1 & \cdots & 1 \end{bmatrix}_{K \times K}$$

$3 \times 3$의 경우:
$$h_{3\times3} = \frac{1}{9}\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

**효과**: 영상을 흐릿하게(블러) 만들고, 잡음을 줄이지만 에지도 흐릿해집니다.

**주파수 응답**: 2D 박스 필터는 주파수 영역에서 2D sinc 함수로, 상당한 사이드로브(Sidelobe) 누설이 있습니다.

### 4.2 가우시안(Gaussian) 필터

더 나은 평활화 필터는 2D 가우시안 커널을 사용합니다:

$$h_{Gauss}[m, n] = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{m^2 + n^2}{2\sigma^2}\right)$$

가우시안 필터의 특성:
- **분리 가능**: $h[m,n] = g[m] \cdot g[n]$, 여기서 $g[k] = \exp(-k^2/(2\sigma^2))$
- **등방성(Isotropic)**: 모든 방향에서 동일한 평활화
- **링잉 없음**: 주파수 영역에서도 가우시안(사이드로브 없음)
- **최적**: 공간-주파수 불확정성 곱을 최소화

**분리 가능성**으로 픽셀당 계산이 $O(K^2)$에서 $O(2K)$로 감소합니다.

일반적인 커널 크기: $\sigma = 1 \to 5 \times 5$, $\sigma = 2 \to 9 \times 9$ (크기 $\geq 6\sigma + 1$ 선택).

### 4.3 가중 평균 필터

박스와 가우시안의 절충안:

$$h_{weighted} = \frac{1}{16}\begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$

이는 $\sigma \approx 0.85$인 가우시안의 근사입니다.

---

## 5. 공간 영역 필터링: 선명화

### 5.1 라플라시안(Laplacian) 연산자

**라플라시안**은 밝기 변화가 급격한 영역을 강조하는 2차 미분 연산자입니다:

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

이산 근사(4방향 연결):
$$\nabla^2 f[m,n] \approx f[m+1,n] + f[m-1,n] + f[m,n+1] + f[m,n-1] - 4f[m,n]$$

커널:
$$h_{Lap4} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

대각선 포함(8방향 연결):
$$h_{Lap8} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & -8 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

### 5.2 라플라시안 선명화

원본 영상에서 라플라시안을 빼서 선명화를 달성합니다:

$$g[m,n] = f[m,n] - c \cdot \nabla^2 f[m,n]$$

여기서 $c > 0$는 선명화 강도를 제어합니다. $c = 1$ (4방향 라플라시안)에서의 결합 커널:

$$h_{sharp} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

### 5.3 언샤프 마스킹(Unsharp Masking)

사진에서 사용되던 고전적인 기법:

1. 영상 블러 처리: $f_{blur} = f * h_{Gauss}$
2. "언샤프 마스크" 계산: $\text{mask} = f - f_{blur}$
3. 스케일링된 마스크 추가: $g = f + k \cdot \text{mask}$

여기서 $k > 0$은 선명화 계수입니다.

$$g = (1 + k) f - k (f * h_{Gauss}) = f + k(f - f * h_{Gauss})$$

주파수 영역에서: $G = F + k(F - F \cdot H_{Gauss}) = F(1 + k - kH_{Gauss})$

이는 **고역 강조 필터(Highpass Emphasis Filter)**입니다: DC 성분은 그대로 두면서 고주파를 증폭합니다.

---

## 6. 중앙값 필터링

### 6.1 정의

**중앙값 필터(Median Filter)**는 각 픽셀을 지역 이웃의 픽셀 값들의 중앙값으로 대체합니다:

$$g[m,n] = \text{median}\{f[m+i, n+j] : (i,j) \in \mathcal{W}\}$$

여기서 $\mathcal{W}$는 필터 윈도우(일반적으로 $3 \times 3$ 또는 $5 \times 5$)입니다.

### 6.2 특성

- **비선형**: 합성곱으로 표현 불가
- **에지 보존**: 날카로운 에지를 유지하면서 잡음 평활화
- **임펄스 잡음에 탁월**: 소금-후추(Salt-and-Pepper) 잡음을 완전히 제거
- **순위 순서 필터(Rank-Order Filter)**: 순서 통계량 필터의 특수한 경우

### 6.3 선형 필터와 비교

| 특성 | 가우시안 필터 | 중앙값 필터 |
|---|---|---|
| 유형 | 선형 | 비선형 |
| 가우시안 잡음 | 우수 | 보통 |
| 소금-후추 잡음 | 불량(스파이크를 흐리게) | 탁월(스파이크 제거) |
| 에지 보존 | 에지 흐릿화 | 에지 보존 |
| 계산량 | 픽셀당 $O(K^2)$ | 픽셀당 $O(K^2 \log K)$ (정렬) |
| 주파수 영역 | 저역통과(가우시안) | 단순한 주파수 해석 없음 |

---

## 7. 주파수 영역 필터링: 저역통과

### 7.1 일반 절차

1. 2D DFT 계산: $F = \text{FFT2}(f)$
2. 스펙트럼 중심화: $F_c = \text{fftshift}(F)$
3. 필터 곱: $G_c = F_c \cdot H$
4. 중심화 해제: $G = \text{ifftshift}(G_c)$
5. 역 DFT: $g = \text{IFFT2}(G)$

### 7.2 이상적 저역통과 필터

$$H_{ideal}[k, l] = \begin{cases} 1 & \sqrt{k^2 + l^2} \leq D_0 \\ 0 & \text{otherwise} \end{cases}$$

여기서 $D_0$는 차단 주파수(중심에서 픽셀 단위)입니다.

**문제**: 이상적 필터는 무한한 공간 범위(2D sinc)를 가져 **링잉 아티팩트(Ringing Artifact)**(깁스 현상, Gibbs Phenomenon)를 유발합니다.

### 7.3 버터워스(Butterworth) 저역통과 필터

$$H_{Butter}[k, l] = \frac{1}{1 + \left(\frac{D(k,l)}{D_0}\right)^{2n}}$$

여기서 $D(k,l) = \sqrt{k^2 + l^2}$는 중심으로부터의 거리이고 $n$은 필터 차수입니다.

- $n = 1$: 완만한 롤오프(Rolloff), 링잉 최소
- $n = 2$: 좋은 균형(일반적으로 사용)
- $n \to \infty$: 이상적 필터에 근접

### 7.4 가우시안 저역통과 필터

$$H_{Gauss}[k, l] = \exp\!\left(-\frac{D(k,l)^2}{2D_0^2}\right)$$

**특성**:
- 링잉 없음(주파수에서 가우시안 → 공간에서 가우시안)
- 부드러운 롤오프
- $D_0$는 $H = e^{-1/2} \approx 0.607$에 해당하는 주파수

### 7.5 저역통과 필터 비교

```
|H(D)|
  1 ─── ─────────┐
  │    Ideal ──────┘
  │    Butter(n=2) ─────╲
  │    Gaussian ──────────╲
  │                         ╲
  0 ────────────────────────────▶ D
                   D₀
```

가우시안 필터는 가장 부드러운 공간 응답(링잉 없음)을 제공하지만 가장 덜 날카로운 주파수 차단을 갖습니다.

---

## 8. 주파수 영역 필터링: 고역통과 및 대역통과

### 8.1 고역통과 필터(Highpass Filter)

고역통과 필터는 저역통과 필터의 보완입니다:

$$H_{HP}(k, l) = 1 - H_{LP}(k, l)$$

고역통과 필터링은 에지와 세밀한 디테일을 보존하면서 부드럽게 천천히 변하는 영역을 제거합니다.

**이상적 고역통과**:
$$H_{ideal,HP}[k,l] = \begin{cases} 0 & D(k,l) \leq D_0 \\ 1 & D(k,l) > D_0 \end{cases}$$

**가우시안 고역통과**:
$$H_{Gauss,HP}[k,l] = 1 - \exp\!\left(-\frac{D(k,l)^2}{2D_0^2}\right)$$

### 8.2 고부스트 필터(High-Boost Filter, 고역 강조)

저주파 성분의 일부를 유지하기 위해:

$$H_{boost} = a + (1-a) \cdot H_{HP} = a \cdot H_{LP} + H_{HP}$$

$a > 1$이면: 전반적인 영상을 보존하면서 고주파를 강조합니다.

이는 주파수 영역에서 언샤프 마스킹과 동일합니다.

### 8.3 대역통과(Bandpass) 및 대역저지(Bandstop, 노치) 필터

**대역통과 필터**는 $D_1$과 $D_2$ 사이의 링 영역 주파수를 통과시킵니다:

$$H_{BP}[k,l] = \begin{cases} 1 & D_1 \leq D(k,l) \leq D_2 \\ 0 & \text{otherwise} \end{cases}$$

**노치 필터(Notch Filter)**는 특정 주파수 성분을 제거합니다(주기적 잡음 패턴 제거에 유용):

$$H_{notch}[k,l] = \prod_{i=1}^{Q} \frac{1}{1 + \left(\frac{D_{0i}}{D_i(k,l)}\right)^{2n}}$$

여기서 $D_i(k,l)$는 $i$번째 노치 중심으로부터의 거리입니다.

**응용**: 주기적 간섭 패턴 제거(예: 모아레(Moire) 패턴, 스캔 아티팩트).

---

## 9. 에지 검출: 그래디언트 연산자

### 9.1 영상 그래디언트(Gradient)

2D 영상 $f(x,y)$의 그래디언트는:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} G_x \\ G_y \end{bmatrix}$$

**그래디언트 크기**는 에지 강도를 나타냅니다:
$$|\nabla f| = \sqrt{G_x^2 + G_y^2} \approx |G_x| + |G_y|$$

**그래디언트 방향**은 에지 방향을 나타냅니다:
$$\theta = \arctan\!\left(\frac{G_y}{G_x}\right)$$

### 9.2 소벨(Sobel) 연산자

소벨 연산자는 평활화된 그래디언트 추정치를 계산합니다:

$$G_x = h_x * f = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * f$$

$$G_y = h_y * f = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * f$$

**설계 근거**: 소벨 커널은 분리 가능합니다: $h_x = \begin{bmatrix}1\\2\\1\end{bmatrix} \begin{bmatrix}-1&0&1\end{bmatrix}$. 첫 번째 인자는 $y$ 방향으로 평활화하고, 두 번째는 $x$ 방향으로 미분을 계산합니다. 평활화는 잡음 민감도를 줄입니다.

### 9.3 프레윗(Prewitt) 연산자

소벨과 유사하지만 균일한 가중치를 사용합니다:

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix} * f, \qquad G_y = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix} * f$$

### 9.4 샤르(Scharr) 연산자

더 나은 회전 대칭성을 가진 개선된 그래디언트 근사:

$$G_x = \begin{bmatrix} -3 & 0 & 3 \\ -10 & 0 & 10 \\ -3 & 0 & 3 \end{bmatrix} * f$$

### 9.5 로버츠 크로스(Roberts Cross) 연산자

$2 \times 2$ 커널을 사용하는 가장 단순한 그래디언트 연산자:

$$G_x = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} * f, \qquad G_y = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} * f$$

작은 커널 크기로 인해 잡음에 민감합니다.

---

## 10. 에지 검출: 라플라시안과 LoG

### 10.1 라플라시안 에지 검출

라플라시안은 2차 미분의 **영교차(Zero Crossing)**로 에지를 검출합니다:

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

에지에서: 1차 미분은 피크를 가지고, 2차 미분은 0을 교차합니다.

**장점**: 등방성(모든 방향에서 동일하게 에지 검출).

**단점**: 잡음에 매우 민감(1차 연산자보다 고주파 잡음을 더 증폭).

### 10.2 가우시안의 라플라시안(LoG, Laplacian of Gaussian)

잡음 민감도를 줄이기 위해, 먼저 가우시안으로 평활화한 후 라플라시안을 적용합니다:

$$\text{LoG}(x, y) = \nabla^2 [G_\sigma * f] = [\nabla^2 G_\sigma] * f$$

LoG 커널(1D에서 "멕시코 모자(Mexican Hat)"라고도 불림)은:

$$\text{LoG}(x, y) = -\frac{1}{\pi\sigma^4}\left[1 - \frac{x^2 + y^2}{2\sigma^2}\right] \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

**마-힐드레스 에지 검출기(Marr-Hildreth Edge Detector)**: $\text{LoG} * f$의 영교차를 찾습니다.

### 10.3 가우시안 차분(DoG, Difference of Gaussians)

LoG의 효율적인 근사:

$$\text{DoG}(x,y) = G_{\sigma_1}(x,y) - G_{\sigma_2}(x,y) \approx (\sigma_2 - \sigma_1) \cdot \nabla^2 G_{\bar{\sigma}}$$

일반적으로 $\sigma_2 / \sigma_1 \approx 1.6$. DoG는 SIFT 특징 검출기의 기반입니다.

---

## 11. 캐니 에지 검출기

### 11.1 설계 기준

존 캐니(John Canny)(1986)는 세 가지 기준으로 에지 검출을 최적화 문제로 공식화했습니다:

1. **좋은 검출(Good Detection)**: 실제 에지를 놓칠 확률이 낮고 거짓 에지가 발생할 확률이 낮음
2. **좋은 위치 정확도(Good Localization)**: 검출된 에지가 실제 에지에 최대한 가까워야 함
3. **단일 응답(Single Response)**: 에지당 하나의 응답만(중복 검출 없음)

### 11.2 알고리즘 단계

```
Step 1: Gaussian smoothing
────────────────────────
f_smooth = f * G_σ

Step 2: Gradient computation (Sobel)
────────────────────────────────────
G_x = h_x * f_smooth
G_y = h_y * f_smooth
Magnitude = sqrt(G_x² + G_y²)
Direction θ = atan2(G_y, G_x)

Step 3: Non-maximum suppression
───────────────────────────────
For each pixel:
  - Quantize θ to 0°, 45°, 90°, 135°
  - Compare magnitude with neighbors along gradient direction
  - Suppress (set to 0) if not a local maximum

Step 4: Double thresholding
───────────────────────────
Strong edges:  magnitude > T_high
Weak edges:    T_low < magnitude ≤ T_high
Non-edges:     magnitude ≤ T_low

Step 5: Edge tracking by hysteresis
────────────────────────────────────
- Strong edges are always edges
- Weak edges are edges only if connected to a strong edge
- Non-edges are discarded
```

### 11.3 비최대 억제(Non-Maximum Suppression)

이 단계는 그래디언트 방향을 따라 지역 최대값이 아닌 픽셀을 억제하여 에지를 1픽셀 너비로 세밀화합니다:

```
Gradient direction:          Neighbors to compare:
       0° (horizontal)       left, right
      45° (diagonal)         upper-right, lower-left
      90° (vertical)         above, below
     135° (anti-diagonal)    upper-left, lower-right
```

### 11.4 이력 임계값(Hysteresis Thresholding)

두 개의 임계값($T_{low}$와 $T_{high}$, 일반적으로 $T_{low} = 0.4 T_{high}$)을 사용하면 다음을 방지합니다:
- 끊어진 에지(단일 높은 임계값이 유발하는)
- 과도한 거짓 에지(단일 낮은 임계값이 유발하는)

이력 단계는 연결성 분석(예: BFS 또는 DFS)을 사용하여 약한 에지 픽셀을 강한 에지 픽셀에 연결합니다.

### 11.5 매개변수 선택

| 매개변수 | 효과 | 일반적인 범위 |
|---|---|---|
| $\sigma$ (가우시안) | 잡음 억제 대 에지 위치 정확도 | 1.0 - 3.0 |
| $T_{high}$ | 강한 에지 임계값 | 그래디언트의 70번째-90번째 백분위수 |
| $T_{low}$ | 약한 에지 임계값 | $0.3 T_{high}$ ~ $0.5 T_{high}$ |

---

## 12. 영상 향상

### 12.1 히스토그램(Histogram)

영상의 **히스토그램**은 각 밝기 레벨의 픽셀 수를 셉니다:

$$h[k] = \text{밝기 } k \text{를 가진 픽셀의 수}, \quad k = 0, 1, \ldots, L-1$$

**정규화된 히스토그램**(확률 질량 함수, PMF):

$$p[k] = \frac{h[k]}{M \cdot N}$$

### 12.2 히스토그램 평활화(Histogram Equalization)

히스토그램 평활화는 픽셀 밝기를 재분배하여 (근사적으로) 균일한 히스토그램을 달성하고 대비를 최대화합니다.

**연속 경우**: 균일 분포를 만드는 변환 $s = T(r)$은:

$$s = T(r) = \int_0^r p_r(\rho) \, d\rho = \text{CDF}(r)$$

**이산 경우**: $L$-레벨 영상에 대해:

$$s_k = T(r_k) = (L-1) \sum_{j=0}^{k} p[j] = (L-1) \cdot \text{CDF}[k]$$

**알고리즘**:
1. 히스토그램 $h[k]$ 계산
2. CDF 계산: $\text{CDF}[k] = \sum_{j=0}^{k} p[j]$
3. 각 픽셀 매핑: $f_{eq}[m,n] = \text{round}((L-1) \cdot \text{CDF}[f[m,n]])$

### 12.3 대비 제한 적응형 히스토그램 평활화(CLAHE, Contrast Limited Adaptive Histogram Equalization)

전역 히스토그램 평활화는 부드러운 영역에서 잡음을 과도하게 강조할 수 있습니다. CLAHE는 다음과 같이 해결합니다:

1. 영상을 작은 타일(예: $8 \times 8$)로 분할
2. 히스토그램을 제한하는 **클립 한계(Clip Limit)**를 적용하여 각 타일에 히스토그램 평활화 적용(초과 카운트 재분배)
3. 타일 간 자연스러운 결과를 위해 보간

### 12.4 대비 스트레칭(Contrast Stretching)

**선형 대비 스트레칭**은 밝기 범위 $[r_{min}, r_{max}]$를 전체 범위 $[0, L-1]$으로 매핑합니다:

$$s = \frac{L-1}{r_{max} - r_{min}} (r - r_{min})$$

**백분위수 기반 스트레칭**: 이상값 영향을 피하기 위해 최소/최대 대신 2번째와 98번째 백분위수 사용.

### 12.5 감마 보정(Gamma Correction)

거듭제곱 변환:

$$s = c \cdot r^\gamma$$

- $\gamma < 1$: 어두운 영역 밝게(어두운 톤 확장)
- $\gamma = 1$: 변화 없음
- $\gamma > 1$: 밝은 영역 어둡게(밝은 톤 확장)

---

## 13. 영상 압축: JPEG 개요

### 13.1 JPEG 파이프라인

JPEG(Joint Photographic Experts Group)은 가장 널리 사용되는 영상 압축 표준입니다. 두 가지 특성을 활용합니다:
1. **공간 중복성(Spatial Redundancy)**: 인접 픽셀들은 상관 관계가 있음
2. **심리시각적 중복성(Psychovisual Redundancy)**: 인간은 고주파 디테일에 덜 민감함

```
JPEG Encoding Pipeline:
────────────────────────────────────────────────────────────

RGB → YCbCr → Chroma    → 8×8 Block → DCT → Quantize → Entropy
              Subsampling  Partition              ↓       Encoding
                                             Q-Table     (Huffman
                                                          or
                                                          Arithmetic)
```

### 13.2 색 공간 변환

RGB에서 YCbCr로 변환:
- **Y**: 루미넌스(Luminance, 밝기) — 가장 중요한 성분
- **Cb**: 청색 차이 색도(Blue-difference Chrominance)
- **Cr**: 적색 차이 색도(Red-difference Chrominance)

$$\begin{bmatrix} Y \\ C_b \\ C_r \end{bmatrix} = \begin{bmatrix} 0.299 & 0.587 & 0.114 \\ -0.169 & -0.331 & 0.500 \\ 0.500 & -0.419 & -0.081 \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix} + \begin{bmatrix} 0 \\ 128 \\ 128 \end{bmatrix}$$

### 13.3 크로마 서브샘플링(Chroma Subsampling)

인간 시각은 밝기보다 색상에 대한 공간 해상도가 낮습니다. JPEG는 색도 채널을 서브샘플링하여 이를 활용합니다:
- **4:4:4**: 서브샘플링 없음(전체 해상도)
- **4:2:2**: 수평으로 2배 서브샘플링
- **4:2:0**: 수평 및 수직 모두 2배 서브샘플링(가장 일반적)

### 13.4 2D 이산 코사인 변환(DCT, Discrete Cosine Transform)

각 $8 \times 8$ 블록은 Type-II DCT를 사용하여 변환됩니다:

$$F[u, v] = \frac{1}{4} C_u C_v \sum_{m=0}^{7} \sum_{n=0}^{7} f[m,n] \cos\!\left(\frac{(2m+1)u\pi}{16}\right) \cos\!\left(\frac{(2n+1)v\pi}{16}\right)$$

여기서 $C_u = 1/\sqrt{2}$ ($u = 0$일 때), 나머지는 $C_u = 1$.

DFT보다 DCT를 선호하는 이유:
- 실수값(복소 연산 불필요)
- 더 나은 에너지 집중(더 적은 계수에 더 많은 에너지)
- 블록 경계에서 불연속 아티팩트 없음(DCT는 암묵적으로 짝수 확장을 가정)

### 13.5 양자화(Quantization)

DCT 계수를 **양자화 행렬(Quantization Matrix)** $Q[u,v]$로 나누고 반올림합니다:

$$F_Q[u,v] = \text{round}\!\left(\frac{F[u,v]}{Q[u,v]}\right)$$

양자화 행렬은 고주파 성분에 더 큰 값을 가집니다(인간이 덜 민감한 주파수를 더 공격적으로 압축). 이것이 **손실(Lossy)** 단계 — 정보가 비가역적으로 버려집니다.

**품질 계수(Quality Factor)**: $Q$ 행렬을 스케일링하여 품질-크기 균형을 조절합니다.

### 13.6 엔트로피 코딩(Entropy Coding)

양자화된 계수는 다음을 사용하여 인코딩됩니다:
1. **지그재그 스캔(Zigzag Scanning)**: $8 \times 8$ 블록을 지그재그 패턴으로 읽어 저주파(큰 값) 계수를 먼저, 고주파(종종 0) 계수를 마지막에 배치
2. **런-길이 인코딩(RLE, Run-Length Encoding)**: 0의 연속을 효율적으로 인코딩
3. **허프만 코딩(Huffman Coding)** (또는 산술 코딩): 추가 압축을 위한 가변 길이 코딩

### 13.7 JPEG 아티팩트

높은 압축률에서 JPEG는 특징적인 아티팩트를 생성합니다:
- **블로킹(Blocking)**: 가시적인 $8 \times 8$ 그리드 경계
- **모기 잡음(Mosquito Noise)**: 날카로운 에지 주변의 링잉
- **색 번짐(Color Bleeding)**: 고대비 에지 근처의 색도 아티팩트

---

## 14. 2D 샘플링

### 14.1 2D 샘플링 정리

$[-u_{max}, u_{max}] \times [-v_{max}, v_{max}]$로 대역 제한된 2D 연속 신호 $f(x, y)$는 다음 조건을 만족할 때 샘플로부터 완벽하게 재구성될 수 있습니다:

$$\frac{1}{\Delta x} > 2u_{max}, \quad \frac{1}{\Delta y} > 2v_{max}$$

여기서 $\Delta x$와 $\Delta y$는 샘플링 간격입니다.

### 14.2 2D 에일리어싱(Aliasing)

샘플링 조건이 위반되면 고주파 콘텐츠가 저주파 콘텐츠와 겹치게 됩니다. 영상에서 에일리어싱은 다음으로 나타납니다:
- **모아레 패턴**: 실제 장면에 존재하지 않는 규칙적인 패턴
- **들쭉날쭉한 에지(계단 현상, Staircasing)**: 대각선이 계단 형태로 나타남
- **마차 바퀴 효과(Wagon Wheel Effect)**: 비디오에서 회전하는 바퀴가 역방향으로 보임

### 14.3 안티에일리어싱(Anti-Aliasing)

영상을 다운샘플링하기 전에 새로운 나이퀴스트 주파수 이상의 주파수를 제거하는 저역통과 필터를 적용합니다:

1. 필터링: $f_{filtered} = f * h_{LP}$
2. 다운샘플링: $f_{down}[m,n] = f_{filtered}[Dm, Dn]$

여기서 $D$는 다운샘플링 계수이고 $h_{LP}$의 차단 주파수는 $\pi/D$입니다.

### 14.4 영상 보간(Interpolation, 업샘플링)

영상 크기 조정을 위한 일반적인 보간 방법:
- **최근접 이웃(Nearest Neighbor)**: 블록 아티팩트, 날카로움
- **쌍선형(Bilinear)**: 부드러움, 약간의 블러
- **쌍삼차(Bicubic)**: 쌍선형보다 날카롭고 약간의 링잉 가능
- **sinc(이상적)**: 완벽한 재구성이지만 무한한 지원

쌍선형 보간:

$$f(x, y) = (1-a)(1-b)f[m,n] + a(1-b)f[m,n+1] + (1-a)b f[m+1,n] + ab \, f[m+1,n+1]$$

여기서 $a = x - m$과 $b = y - n$은 소수 부분입니다.

---

## 15. Python 구현

### 15.1 2D DFT 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def create_test_image(size=256):
    """Create a test image with various spatial frequencies."""
    img = np.zeros((size, size))

    # Horizontal bars (vertical frequency)
    y = np.arange(size)
    for freq in [5, 15, 40]:
        stripe = np.sin(2 * np.pi * freq * y / size)
        img[:, size//6:size//3] += stripe[:, np.newaxis] * (freq == 5)
        img[:, size//3:size//2] += stripe[:, np.newaxis] * (freq == 15)
        img[:, size//2:2*size//3] += stripe[:, np.newaxis] * (freq == 40)

    # Add a rectangle
    img[size//4:3*size//4, 3*size//4:7*size//8] = 1.0

    # Add a circle
    yy, xx = np.mgrid[:size, :size]
    circle = ((xx - size//8)**2 + (yy - 3*size//4)**2) < (size//10)**2
    img[circle] = 1.0

    return img


# Create test image
img = create_test_image(256)

# Compute 2D DFT
F = np.fft.fft2(img)
F_shifted = np.fft.fftshift(F)
magnitude = np.log1p(np.abs(F_shifted))  # Log scale for visualization
phase = np.angle(F_shifted)

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(magnitude, cmap='hot')
axes[1].set_title('Magnitude Spectrum (log scale)')
axes[1].axis('off')

axes[2].imshow(phase, cmap='hsv')
axes[2].set_title('Phase Spectrum')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('2d_dft_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 15.2 공간 필터링: 평활화 및 선명화

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal


def generate_noisy_image(size=256, noise_std=20):
    """Generate a test image with noise."""
    # Create a clean image with geometric shapes
    img = np.zeros((size, size))

    # Rectangle
    img[50:200, 30:100] = 180

    # Circle
    yy, xx = np.mgrid[:size, :size]
    circle = ((xx - 180)**2 + (yy - 130)**2) < 50**2
    img[circle] = 220

    # Triangle
    for i in range(80):
        img[170+i, 130+i//2:230-i//2] = 140

    # Add Gaussian noise
    noisy = img + noise_std * np.random.randn(size, size)
    return img.clip(0, 255), noisy.clip(0, 255)


np.random.seed(42)
clean, noisy = generate_noisy_image()

# Define kernels
box_3x3 = np.ones((3, 3)) / 9
box_5x5 = np.ones((5, 5)) / 25

# Gaussian kernel
sigma = 1.5
size_g = int(6*sigma + 1) | 1  # Ensure odd
ax = np.arange(-size_g//2 + 1, size_g//2 + 1)
xx, yy = np.meshgrid(ax, ax)
gaussian_kernel = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
gaussian_kernel /= gaussian_kernel.sum()

# Laplacian sharpening kernel
laplacian = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

# Apply filters
smoothed_box = ndimage.convolve(noisy, box_5x5)
smoothed_gauss = ndimage.convolve(noisy, gaussian_kernel)
sharpened = ndimage.convolve(clean, laplacian)  # Sharpen the clean image

# Unsharp mask
blurred = ndimage.gaussian_filter(clean, sigma=2.0)
unsharp = clean + 1.5 * (clean - blurred)
unsharp = np.clip(unsharp, 0, 255)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(clean, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Clean Image')

axes[0, 1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('Noisy Image (σ=20)')

axes[0, 2].imshow(smoothed_box, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title('Box Filter 5×5')

axes[1, 0].imshow(smoothed_gauss, cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title(f'Gaussian Filter (σ={sigma})')

axes[1, 1].imshow(np.clip(sharpened, 0, 255), cmap='gray', vmin=0, vmax=255)
axes[1, 1].set_title('Laplacian Sharpening')

axes[1, 2].imshow(unsharp, cmap='gray', vmin=0, vmax=255)
axes[1, 2].set_title('Unsharp Masking (k=1.5)')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('spatial_filtering.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 15.3 주파수 영역 필터링

```python
import numpy as np
import matplotlib.pyplot as plt


def frequency_filter(img, filter_func, *args):
    """Apply a frequency domain filter to an image."""
    M, N = img.shape
    # Compute centered 2D DFT
    F = np.fft.fftshift(np.fft.fft2(img))

    # Create frequency grid
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)

    # Apply filter
    H = filter_func(D, *args)
    G = F * H

    # Inverse DFT
    g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    return g, H


def ideal_lowpass(D, D0):
    return (D <= D0).astype(float)

def butterworth_lowpass(D, D0, n=2):
    return 1.0 / (1 + (D / D0)**(2*n))

def gaussian_lowpass(D, D0):
    return np.exp(-D**2 / (2 * D0**2))


# Create test image
np.random.seed(42)
size = 256
img = np.zeros((size, size))
img[80:180, 60:200] = 200
yy, xx = np.mgrid[:size, :size]
circle = ((xx - 128)**2 + (yy - 128)**2) < 40**2
img[circle] = 255
img += 15 * np.random.randn(size, size)
img = np.clip(img, 0, 255)

# Apply different lowpass filters
D0 = 30  # Cutoff frequency

filtered_ideal, H_ideal = frequency_filter(img, ideal_lowpass, D0)
filtered_butter, H_butter = frequency_filter(img, butterworth_lowpass, D0, 2)
filtered_gauss, H_gauss = frequency_filter(img, gaussian_lowpass, D0)

# Highpass (complement of Gaussian lowpass)
filtered_hp, H_hp = frequency_filter(
    img, lambda D, D0: 1 - gaussian_lowpass(D, D0), D0
)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Top row: filters
axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original')

axes[0, 1].imshow(H_ideal, cmap='gray')
axes[0, 1].set_title(f'Ideal LP (D₀={D0})')

axes[0, 2].imshow(H_butter, cmap='gray')
axes[0, 2].set_title(f'Butterworth LP (D₀={D0}, n=2)')

axes[0, 3].imshow(H_gauss, cmap='gray')
axes[0, 3].set_title(f'Gaussian LP (D₀={D0})')

# Bottom row: filtered images
axes[1, 0].imshow(np.clip(filtered_ideal, 0, 255), cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title('Ideal LP (ringing)')

axes[1, 1].imshow(np.clip(filtered_butter, 0, 255), cmap='gray', vmin=0, vmax=255)
axes[1, 1].set_title('Butterworth LP')

axes[1, 2].imshow(np.clip(filtered_gauss, 0, 255), cmap='gray', vmin=0, vmax=255)
axes[1, 2].set_title('Gaussian LP (smooth)')

axes[1, 3].imshow(np.clip(filtered_hp + 128, 0, 255), cmap='gray')
axes[1, 3].set_title('Gaussian HP')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('frequency_domain_filtering.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 15.4 에지 검출

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def sobel_edge_detection(img):
    """Manual Sobel edge detection."""
    # Sobel kernels
    Kx = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)
    Ky = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]], dtype=float)

    Gx = ndimage.convolve(img, Kx)
    Gy = ndimage.convolve(img, Ky)

    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.arctan2(Gy, Gx)

    return magnitude, direction, Gx, Gy


def canny_edge_detector(img, sigma=1.0, low_thresh=0.1, high_thresh=0.3):
    """
    Manual Canny edge detector implementation.

    Parameters
    ----------
    img : ndarray
        Input grayscale image (float, 0-1 range)
    sigma : float
        Gaussian smoothing parameter
    low_thresh : float
        Low threshold (fraction of max gradient)
    high_thresh : float
        High threshold (fraction of max gradient)

    Returns
    -------
    edges : ndarray
        Binary edge map
    """
    # Step 1: Gaussian smoothing
    smoothed = ndimage.gaussian_filter(img, sigma)

    # Step 2: Gradient computation (Sobel)
    magnitude, direction, _, _ = sobel_edge_detection(smoothed)

    # Normalize
    magnitude = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude

    # Step 3: Non-maximum suppression
    M, N = magnitude.shape
    nms = np.zeros_like(magnitude)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # Determine neighbor direction
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = magnitude[i, j-1], magnitude[i, j+1]
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[i-1, j+1], magnitude[i+1, j-1]
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[i-1, j], magnitude[i+1, j]
            else:
                n1, n2 = magnitude[i-1, j-1], magnitude[i+1, j+1]

            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                nms[i, j] = magnitude[i, j]

    # Step 4: Double thresholding
    strong = nms >= high_thresh
    weak = (nms >= low_thresh) & (nms < high_thresh)

    # Step 5: Edge tracking by hysteresis (simplified using dilation)
    edges = strong.copy()
    # Connect weak edges adjacent to strong edges
    for _ in range(10):  # Iterate to propagate connectivity
        dilated = ndimage.binary_dilation(edges)
        edges = edges | (weak & dilated)

    return edges.astype(float)


# Create a test image
np.random.seed(42)
size = 256
img = np.zeros((size, size))
# Rectangle
img[60:200, 40:220] = 0.8
# Inner rectangle
img[90:170, 70:190] = 0.3
# Circle
yy, xx = np.mgrid[:size, :size]
circle = ((xx - 128)**2 + (yy - 128)**2) < 30**2
img[circle] = 1.0
# Add mild noise
img += 0.02 * np.random.randn(size, size)
img = np.clip(img, 0, 1)

# Edge detection methods
magnitude, direction, Gx, Gy = sobel_edge_detection(img)

# Laplacian
laplacian = ndimage.laplace(ndimage.gaussian_filter(img, 1.0))

# LoG
log_result = ndimage.gaussian_laplace(img, sigma=2.0)

# Canny
canny_edges = canny_edge_detector(img, sigma=1.5, low_thresh=0.05, high_thresh=0.15)

# Display
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')

axes[0, 1].imshow(magnitude, cmap='gray')
axes[0, 1].set_title('Sobel Magnitude')

axes[0, 2].imshow(direction, cmap='hsv')
axes[0, 2].set_title('Sobel Direction')

axes[1, 0].imshow(np.abs(laplacian), cmap='gray')
axes[1, 0].set_title('Laplacian (after Gaussian)')

axes[1, 1].imshow(np.abs(log_result), cmap='gray')
axes[1, 1].set_title('Laplacian of Gaussian (σ=2)')

axes[1, 2].imshow(canny_edges, cmap='gray')
axes[1, 2].set_title('Canny Edge Detector')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('edge_detection.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 15.5 히스토그램 평활화

```python
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(img):
    """
    Apply histogram equalization to an 8-bit grayscale image.

    Parameters
    ----------
    img : ndarray
        Input image (uint8 or float [0, 255])

    Returns
    -------
    equalized : ndarray
        Histogram-equalized image
    """
    img_int = img.astype(np.uint8)
    L = 256

    # Compute histogram
    hist, bins = np.histogram(img_int, bins=L, range=(0, L))

    # Compute CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalize to [0, 1]

    # Map pixel values
    equalized = np.round((L - 1) * cdf_normalized[img_int]).astype(np.uint8)

    return equalized


# Create a low-contrast test image
np.random.seed(42)
size = 256
yy, xx = np.mgrid[:size, :size]

# Low-contrast image (values concentrated in [80, 170])
img = 80 + 90 * (np.sin(2*np.pi*3*xx/size) * np.cos(2*np.pi*2*yy/size) + 1) / 2
circle = ((xx - 128)**2 + (yy - 128)**2) < 60**2
img[circle] = 150
img += 5 * np.random.randn(size, size)
img = np.clip(img, 0, 255).astype(np.uint8)

# Apply histogram equalization
equalized = histogram_equalization(img)

# Gamma correction examples
gamma_low = np.clip(255 * (img / 255.0)**0.5, 0, 255).astype(np.uint8)   # Brighten
gamma_high = np.clip(255 * (img / 255.0)**2.0, 0, 255).astype(np.uint8)  # Darken

# Plot
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Images
axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original (low contrast)')

axes[0, 1].imshow(equalized, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('Histogram Equalized')

axes[0, 2].imshow(gamma_low, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title('Gamma = 0.5 (brighten)')

axes[0, 3].imshow(gamma_high, cmap='gray', vmin=0, vmax=255)
axes[0, 3].set_title('Gamma = 2.0 (darken)')

# Histograms
axes[1, 0].hist(img.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.7)
axes[1, 0].set_title('Original Histogram')
axes[1, 0].set_xlim([0, 255])

axes[1, 1].hist(equalized.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
axes[1, 1].set_title('Equalized Histogram')
axes[1, 1].set_xlim([0, 255])

axes[1, 2].hist(gamma_low.ravel(), bins=256, range=(0, 255), color='orange', alpha=0.7)
axes[1, 2].set_title('Gamma 0.5 Histogram')
axes[1, 2].set_xlim([0, 255])

axes[1, 3].hist(gamma_high.ravel(), bins=256, range=(0, 255), color='red', alpha=0.7)
axes[1, 3].set_title('Gamma 2.0 Histogram')
axes[1, 3].set_xlim([0, 255])

for ax in axes[0]:
    ax.axis('off')

plt.tight_layout()
plt.savefig('histogram_equalization.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 15.6 DCT 기반 압축 데모

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn


def jpeg_compress_block(block, quality=50):
    """
    Simulate JPEG compression on a single 8x8 block.

    Parameters
    ----------
    block : ndarray
        8x8 pixel block
    quality : int
        Quality factor (1-100)

    Returns
    -------
    reconstructed : ndarray
        Reconstructed block after compression
    n_nonzero : int
        Number of non-zero quantized coefficients
    """
    # Standard JPEG luminance quantization table
    Q_base = np.array([
        [16, 11, 10, 16,  24,  40,  51,  61],
        [12, 12, 14, 19,  26,  58,  60,  55],
        [14, 13, 16, 24,  40,  57,  69,  56],
        [14, 17, 22, 29,  51,  87,  80,  62],
        [18, 22, 37, 56,  68, 109, 103,  77],
        [24, 35, 55, 64,  81, 104, 113,  92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103,  99]
    ])

    # Scale quantization table by quality
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    Q = np.clip(np.round(Q_base * scale / 100), 1, 255)

    # Shift to center around 0
    block_shifted = block.astype(float) - 128

    # Forward DCT
    dct_block = dctn(block_shifted, type=2, norm='ortho')

    # Quantize
    quantized = np.round(dct_block / Q)
    n_nonzero = np.count_nonzero(quantized)

    # Dequantize
    dequantized = quantized * Q

    # Inverse DCT
    reconstructed = idctn(dequantized, type=2, norm='ortho') + 128

    return np.clip(reconstructed, 0, 255), n_nonzero


def jpeg_compress_image(img, quality=50):
    """Simulate JPEG compression on a full image."""
    M, N = img.shape
    # Pad to multiple of 8
    M_pad = int(np.ceil(M / 8)) * 8
    N_pad = int(np.ceil(N / 8)) * 8
    padded = np.zeros((M_pad, N_pad))
    padded[:M, :N] = img

    result = np.zeros_like(padded)
    total_nonzero = 0

    for i in range(0, M_pad, 8):
        for j in range(0, N_pad, 8):
            block = padded[i:i+8, j:j+8]
            result[i:i+8, j:j+8], nz = jpeg_compress_block(block, quality)
            total_nonzero += nz

    total_coeffs = (M_pad // 8) * (N_pad // 8) * 64
    return result[:M, :N], total_nonzero / total_coeffs


# Create test image
np.random.seed(42)
size = 256
yy, xx = np.mgrid[:size, :size]
img = np.zeros((size, size))
img[40:220, 30:230] = 180
circle = ((xx - 128)**2 + (yy - 128)**2) < 50**2
img[circle] = 240
img += np.sin(2*np.pi*5*xx/size) * 30
img = np.clip(img + 10*np.random.randn(size, size), 0, 255)

# Compress at different quality levels
qualities = [5, 20, 50, 90]
fig, axes = plt.subplots(2, len(qualities)+1, figsize=(18, 8))

axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# Show DCT of one block
block = img[40:48, 30:38]
dct_block = dctn(block - 128, type=2, norm='ortho')
axes[1, 0].imshow(np.log1p(np.abs(dct_block)), cmap='hot')
axes[1, 0].set_title('DCT of 8x8 block\n(log magnitude)')
axes[1, 0].axis('off')

for idx, q in enumerate(qualities):
    compressed, ratio = jpeg_compress_image(img, quality=q)
    psnr = 10 * np.log10(255**2 / np.mean((img - compressed)**2))
    error = np.abs(img - compressed)

    axes[0, idx+1].imshow(compressed, cmap='gray', vmin=0, vmax=255)
    axes[0, idx+1].set_title(f'Q={q}\nPSNR={psnr:.1f} dB')
    axes[0, idx+1].axis('off')

    axes[1, idx+1].imshow(error, cmap='hot', vmin=0, vmax=50)
    axes[1, idx+1].set_title(f'Error (non-zero: {ratio:.1%})')
    axes[1, idx+1].axis('off')

plt.suptitle('JPEG Compression Simulation (DCT-based)', fontsize=14)
plt.tight_layout()
plt.savefig('jpeg_compression.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 16. 연습 문제

### 연습 1: 2D DFT 성질

(a) 중앙에 흰색 직사각형(64x32 픽셀)이 있는 $256 \times 256$ 영상을 만드세요. 2D DFT 크기 스펙트럼을 계산하고 표시하세요. 2D sinc 함수 관점에서 패턴을 설명하세요.

(b) 직사각형을 오른쪽으로 50픽셀 이동하세요. 새로운 크기 및 위상 스펙트럼을 계산하세요. 무엇이 변하고 무엇이 동일합니까? 이동 정리를 검증하세요.

(c) 직사각형을 45도 회전하고 크기 스펙트럼을 계산하세요. 공간 영역에서의 회전이 주파수 영역에 어떤 영향을 미칩니까?

(d) 한 영상의 크기와 다른 영상의 위상을 교환하세요. 결과를 표시하세요. 크기와 위상 중 어느 것이 더 많은 시각 정보를 담고 있습니까?

### 연습 2: 평활화 필터 비교

다음을 포함한 테스트 영상을 생성하세요:
- 가우시안 잡음이 있는 영역 ($\sigma = 30$)
- 소금-후추 잡음이 있는 영역(5% 확률)

(a) 크기 $3 \times 3$, $5 \times 5$, $7 \times 7$의 박스 필터를 적용하세요. 각각의 PSNR을 측정하세요.

(b) $\sigma = 1, 2, 3$인 가우시안 필터를 적용하세요. 박스 필터와 비교하세요.

(c) 크기 $3 \times 3$과 $5 \times 5$의 중앙값 필터를 적용하세요. 어떤 잡음 유형이 중앙값 필터링에서 가장 많은 혜택을 받습니까?

(d) 각 필터에 대해 에지 보존 지표를 계산하세요: 필터링된 영상의 그래디언트 에너지 대 원본 깨끗한 영상의 그래디언트 에너지 비율.

### 연습 3: 주파수 영역 설계

(a) 중심에서 $D_1 = 20$과 $D_2 = 60$ 픽셀 사이의 공간 주파수를 통과시키는 이상적 대역통과 필터를 설계하고 구현하세요. 테스트 영상에 적용하고 결과를 관찰하세요.

(b) 사진에 주기적 수직 줄무늬(잡음)가 추가된 영상을 만드세요. 기저 영상에 영향을 주지 않고 줄무늬를 제거하는 노치 필터를 설계하세요.

(c) 동형 필터(Homomorphic Filter)를 구현하세요: 영상의 로그를 취하고, 주파수 영역에서 고역통과 필터를 적용한 후, 지수화합니다. 이는 조명을 정규화하여 지역 대비를 향상시킵니다. 히스토그램 평활화와 비교하세요.

### 연습 4: 에지 검출 비교

다음을 포함한 테스트 영상을 만드세요:
- 강한 에지(고대비)
- 약한 에지(저대비)
- 텍스처 영역
- 가우시안 잡음 ($\sigma = 10$)

(a) 소벨, 프레윗, 샤르 연산자를 적용하세요. 에지 맵과 그래디언트 크기를 비교하세요.

(b) $\sigma = 1, 2, 3$인 라플라시안과 LoG를 적용하세요. $\sigma$가 검출된 에지에 어떤 영향을 미칩니까?

(c) 완전한 캐니 에지 검출기(5단계 모두)를 구현하세요. 임계값과 가우시안 $\sigma$를 변화시켜 효과를 확인하세요.

(d) 정밀도-재현율(Precision-Recall) 곡선을 사용하여 정답 에지 맵과 비교해 검출기를 정량적으로 평가하세요.

### 연습 5: 히스토그램 처리

(a) 어두운 영상(히스토그램이 [0, 50]에 집중)과 밝은 영상(히스토그램이 [200, 255]에 집중)을 생성하세요. 두 영상에 히스토그램 평활화를 적용하고 비교하세요.

(b) CLAHE를 구현하세요: 영상을 $8 \times 8$ 타일로 나누고, 각 타일에 대해 클리핑된 히스토그램을 계산하고, 부드러운 전환을 위해 쌍선형 보간을 사용하세요. 전역 히스토그램 평활화와 비교하세요.

(c) 히스토그램 매칭(명세화, Histogram Specification)을 구현하세요: 목표 히스토그램 형태가 주어졌을 때, 영상의 히스토그램이 목표에 근사하도록 변환하세요.

### 연습 6: JPEG 압축 분석

(a) 완전한 JPEG 압축 시뮬레이터를 구현하세요: 색 공간 변환(컬러 영상의 경우), $8 \times 8$ 블록 DCT, 양자화, 지그재그 스캔, RLE. 품질 계수 10, 30, 50, 70, 90에서 압축률과 PSNR을 측정하세요.

(b) PSNR 대 압축률 곡선(율-왜곡 곡선, Rate-Distortion Curve)을 플롯하세요.

(c) 낮은 품질에서 블로킹 아티팩트를 시각화하세요. 간단한 디블로킹 필터(블록 경계에만 적용된 저역통과 필터)를 구현하세요.

(d) DCT 압축과 웨이블릿(Wavelet) 압축을 비교하세요: 2D DWT를 적용하고, 작은 계수를 임계값으로 제거하고, 재구성합니다. 두 방법의 율-왜곡 곡선을 플롯하세요.

### 연습 7: 2D 샘플링과 에일리어싱

(a) 영역판 패턴(Zone Plate Pattern)을 가진 고해상도 ($512 \times 512$) 영상을 만드세요: $f[m,n] = \cos(\alpha(m^2 + n^2))$. 2, 4, 8배로 다운샘플링하세요(안티에일리어싱 유무). 에일리어싱 아티팩트를 확인하세요.

(b) 쌍선형 및 쌍삼차 보간을 처음부터 구현하세요. 작은 ($32 \times 32$) 영상을 최근접 이웃, 쌍선형, 쌍삼차 방법으로 8배 업샘플링하세요. 결과를 비교하세요.

(c) 자연 영상이 주어졌을 때, 2D 파워 스펙트럼을 계산하고 유효 대역폭을 추정하세요. 에일리어싱을 피하기 위해 필요한 최소 샘플링률을 결정하세요.

---

## 17. 요약

| 개념 | 핵심 공식 / 아이디어 |
|---|---|
| 2D DFT | $F[k,l] = \sum_m\sum_n f[m,n]e^{-j2\pi(km/M + ln/N)}$ |
| 2D 합성곱 | $g = f * h$ (공간) $\Leftrightarrow$ $G = F \cdot H$ (주파수) |
| 가우시안 필터 | $h[m,n] = \frac{1}{2\pi\sigma^2}e^{-(m^2+n^2)/(2\sigma^2)}$; 분리 가능 |
| 소벨 연산자 | 평활화된 그래디언트: $|\nabla f| = \sqrt{G_x^2 + G_y^2}$ |
| 라플라시안 | $\nabla^2 f$; 영교차로 에지 검출 |
| LoG | $\nabla^2(G_\sigma * f) = (\nabla^2 G_\sigma) * f$ |
| 캐니 | 평활화 $\to$ 그래디언트 $\to$ 비최대 억제 $\to$ 이력 |
| 히스토그램 평활화 | $s_k = (L-1)\sum_{j=0}^k p[j]$ (CDF 매핑) |
| DCT (JPEG) | 실수값, DFT보다 더 나은 에너지 집중 |
| JPEG 양자화 | $F_Q = \text{round}(F/Q)$ (손실 단계) |
| 2D 나이퀴스트 | $1/\Delta x > 2u_{max}$, $1/\Delta y > 2v_{max}$ |

**핵심 요점**:
1. 영상은 2D 신호이며, 1D 신호 처리의 모든 개념이 자연스럽게 2D로 확장됩니다.
2. 2D DFT는 분리 가능하여 행-열 FFT를 통한 효율적인 계산이 가능합니다.
3. 공간 필터링(커널과의 합성곱)은 평활화, 선명화, 에지 검출을 처리합니다.
4. 주파수 영역 필터링은 큰 커널에 효율적이고 직관적인 설계를 제공합니다(저역통과, 고역통과, 노치).
5. 에지 검출은 미분(그래디언트 또는 라플라시안)과 잡음 억제(가우시안 평활화)를 결합합니다.
6. 캐니 에지 검출기는 검출 정확도, 위치 정확도, 단일 응답 기준을 최적으로 균형 잡습니다.
7. 히스토그램 평활화는 강력하고 자동적인 대비 향상 기법입니다.
8. JPEG 압축은 DCT의 에너지 집중과 고주파 디테일에 대한 인간 시각의 낮은 민감도를 활용합니다.
9. 안티에일리어싱(다운샘플링 전 저역통과 필터)은 아티팩트 없는 영상 크기 조정에 중요합니다.

---

## 18. 참고 문헌

1. R.C. Gonzalez and R.E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018.
2. A.K. Jain, *Fundamentals of Digital Image Processing*, Prentice Hall, 1989.
3. W.K. Pratt, *Digital Image Processing*, 4th ed., Wiley, 2007.
4. J.F. Canny, "A computational approach to edge detection," *IEEE Trans. PAMI*, vol. 8, no. 6, pp. 679-698, 1986.
5. G.K. Wallace, "The JPEG still picture compression standard," *IEEE Trans. Consumer Electronics*, vol. 38, no. 1, pp. xviii-xxxiv, 1992.
6. A.V. Oppenheim and R.W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010 (Chapter 8: The Discrete Fourier Transform).

---

**이전**: [14. 시간-주파수 분석](./14_Time_Frequency_Analysis.md) | **다음**: [16. 응용](./16_Applications.md)
