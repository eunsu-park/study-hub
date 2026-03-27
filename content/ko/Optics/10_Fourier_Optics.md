# 10. 푸리에 광학

[← 이전: 09. 광섬유](09_Fiber_Optics.md) | [다음: 11. 홀로그래피 →](11_Holography.md)

---

푸리에 광학(Fourier optics)은 단순한 유리 렌즈가 빛의 속도로 푸리에 변환(Fourier transform)이라는 수학적 연산을 수행한다는 놀라운 인식이다 — 공간 패턴을 공간 주파수로, 또는 그 반대로 변환한다. 물리 광학과 선형 시스템 이론을 연결하는 이 통찰은 광학 기기 설계, 영상 처리, 그리고 현미경과 카메라의 근본적인 작동 방식에 대한 이해를 혁신했다.

에른스트 아베(Ernst Abbe)가 1870년대에 현미경 결상 이론을 정립했을 때, 그는 분해능이 렌즈 품질만으로 결정되는 것이 아니라 렌즈가 수집할 수 있는 회절 차수(공간 주파수)의 수에 의해 결정된다는 것을 보였다. 프리츠 제르니케(Frits Zernike)가 1930년대에 위상차 현미경(phase contrast microscopy)을 발명했을 때(1953년 노벨상), 그는 푸리에 광학 원리를 이용하여 투명한 생물 시편을 가시화했다. 오늘날, 푸리에 광학은 천문 망원경의 적응 광학(adaptive optics)부터 스마트폰의 계산 영상(computational imaging)까지 모든 분야의 기반이 된다.

이 레슨은 회절과 전파 간의 푸리에 변환 관계를 전개하고, 렌즈가 어떻게 푸리에 변환을 구현하는지 보여주며, 공간 필터링(spatial filtering), 광학 전달 함수(optical transfer function), 그리고 코히런트 결상 이론을 탐구한다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. 스칼라 회절을 선형 시불변(LSI, linear, shift-invariant) 시스템으로 기술하고, 임펄스 응답과 전달 함수를 식별한다
2. 각 스펙트럼(angular spectrum) 표현을 유도하고 자유 공간 전파의 전달 함수를 구한다
3. 얇은 렌즈가 후초점면(back focal plane)에서 푸리에 변환을 수행함을 증명한다
4. 4f 공간 필터링 시스템을 설계하고 다양한 필터 마스크의 효과를 예측한다
5. 점 확산 함수(PSF, point spread function), 광학 전달 함수(OTF, optical transfer function), 변조 전달 함수(MTF, modulation transfer function)를 정의하고 상호 관계를 파악한다
6. 코히런트(coherent) 및 비코히런트(incoherent) 결상에서의 아베 결상 이론과 분해능 한계를 설명한다
7. 위상차 현미경의 원리를 공간 필터링 연산으로 설명한다

---

## 목차

1. [선형 시스템으로서의 회절](#1-선형-시스템으로서의-회절)
2. [각 스펙트럼 표현](#2-각-스펙트럼-표현)
3. [프레넬 및 프라운호퍼 회절 재조명](#3-프레넬-및-프라운호퍼-회절-재조명)
4. [푸리에 변환기로서의 렌즈](#4-푸리에-변환기로서의-렌즈)
5. [4f 시스템과 공간 필터링](#5-4f-시스템과-공간-필터링)
6. [광학 전달 함수](#6-광학-전달-함수)
7. [결상의 아베 이론](#7-결상의-아베-이론)
8. [위상차 현미경](#8-위상차-현미경)
9. [Python 예제](#9-python-예제)
10. [요약](#10-요약)
11. [연습 문제](#11-연습-문제)
12. [참고문헌](#12-참고문헌)

---

## 1. 선형 시스템으로서의 회절

### 1.1 선형 시스템 관점

푸리에 광학의 핵심 통찰은 이것이다: **자유 공간에서 빛의 전파는 선형 시불변(LSI) 시스템이다**. 이는 임펄스 응답 $h(x, y)$ 또는 동등하게 전달 함수 $H(f_x, f_y)$로 완전히 특성화할 수 있음을 의미한다.

$z = 0$ 평면에서의 입력 필드 $U_{\text{in}}(x, y)$가 주어지면, $z = d$ 평면에서의 출력 필드는:

$$U_{\text{out}}(x, y) = U_{\text{in}}(x, y) * h(x, y; d)$$

여기서 $*$는 2D 합성곱(convolution)을 의미하고, $h$는 전파 임펄스 응답(점 광원이 만들어 내는 필드)이다.

주파수 영역에서:

$$\tilde{U}_{\text{out}}(f_x, f_y) = \tilde{U}_{\text{in}}(f_x, f_y) \cdot H(f_x, f_y; d)$$

> **비유**: 복잡한 장면을 음악의 화음으로 생각하라 — 많은 순음(공간 주파수)의 중첩. 자유 공간 전파는 오디오 이퀄라이저와 같다: 전달 함수 $H$에 따라 각 주파수 성분을 독립적으로 수정하되, 새로운 주파수를 생성하지는 않는다. 출력은 각 공간 주파수의 진폭과 위상이 전파 방식에 따라 조정된 입력일 뿐이다. 렌즈는 이 비유에서 스펙트럼 분석기와 같다 — 모든 음을 분리하여 나란히 표시한다.

### 1.2 스칼라 회절 이론(Scalar Diffraction Theory)

우리는 **스칼라 근사(scalar approximation)** 내에서 작업한다: 빛을 스칼라 복소 필드 $U(\mathbf{r}) = |U|e^{i\phi}$로 취급하고 편광을 무시한다. 이는 다음 조건에서 유효하다:
- 형상 크기가 파장보다 훨씬 큰 경우
- 근축(paraxial) 영역에서 작업하는 경우 (각도 < ~20°)
- 날카로운 물질 경계 근처가 아닌 경우

스칼라 필드는 헬름홀츠 방정식(Helmholtz equation)을 만족한다:

$$(\nabla^2 + k^2)U = 0, \quad k = \frac{2\pi}{\lambda}$$

### 1.3 하위헌스-프레넬 원리 (형식화)

파면의 모든 점은 2차 광원으로 작용한다. 레일리-좀머펠트(Rayleigh-Sommerfeld) 회절 적분이 이를 형식화한다:

$$U(x, y, d) = \frac{1}{i\lambda}\iint U(x', y', 0)\frac{e^{ikr}}{r}\cos\theta\,dx'dy'$$

여기서 $r = \sqrt{(x-x')^2 + (y-y')^2 + d^2}$는 광원에서 관측점까지의 거리이다.

---

## 2. 각 스펙트럼 표현

### 2.1 평면파로의 분해

임의의 단색 필드 $U(x, y, 0)$는 2D 푸리에 변환을 통해 평면파로 분해될 수 있다:

$$\tilde{U}(f_x, f_y; 0) = \iint U(x, y, 0)\,e^{-i2\pi(f_x x + f_y y)}\,dx\,dy$$

각 공간 주파수 $(f_x, f_y)$는 다음 각도로 전파하는 평면파에 해당한다:

$$\sin\alpha = \lambda f_x, \quad \sin\beta = \lambda f_y$$

여기서 $\alpha$와 $\beta$는 $xz$ 및 $yz$ 평면에서 $z$축과 이루는 각도이다.

### 2.2 자유 공간의 전달 함수

각 평면파 성분은 거리 $d$를 전파하면서 위상을 획득한다:

$$H(f_x, f_y; d) = \exp\!\left(i2\pi d\sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2}\right)$$

이는 (스칼라 근사 내에서) 정확한 식이다. 두 가지 중요한 영역:

**전파파(Propagating waves)**: $f_x^2 + f_y^2 < 1/\lambda^2$이면, $H$는 순수한 위상 인수이다 — 파는 손실 없이 전파된다.

**에반에센트파(Evanescent waves)**: $f_x^2 + f_y^2 > 1/\lambda^2$이면, 제곱근이 허수가 되어 $H$가 지수적으로 감쇠한다:

$$H \propto \exp\!\left(-2\pi d\sqrt{f_x^2 + f_y^2 - 1/\lambda^2}\right)$$

이 고주파 성분은 파장 이하의 세부 정보를 담고 있지만 파장의 일부 거리 내에서 소멸된다 — 이것이 통상의 광학이 $\sim\lambda/2$보다 작은 형상을 분해할 수 없는 근본적인 이유이다.

### 2.3 근축 (프레넬) 근사

작은 각도의 경우 ($f_x^2 + f_y^2 \ll 1/\lambda^2$), 제곱근을 테일러 전개하면:

$$H(f_x, f_y; d) \approx e^{ikd}\exp\!\left(-i\pi\lambda d(f_x^2 + f_y^2)\right)$$

이것이 **프레넬 전달 함수(Fresnel transfer function)**이다 — 주파수 공간에서의 이차 위상. 이는 공간 영역의 이차 위상(프레넬 전파 커널)에 해당한다.

---

## 3. 프레넬 및 프라운호퍼 회절 재조명

### 3.1 프레넬 회절(Fresnel Diffraction)

근축 근사 하에서, 전파된 필드는:

$$U(x, y, d) = \frac{e^{ikd}}{i\lambda d}\iint U(x', y', 0)\,\exp\!\left(\frac{ik}{2d}\left[(x-x')^2 + (y-y')^2\right]\right)dx'dy'$$

이는 프레넬 커널 $h_F(x,y) = \frac{e^{ikd}}{i\lambda d}\exp\!\left(\frac{ik(x^2+y^2)}{2d}\right)$와의 합성곱이다.

### 3.2 프라운호퍼 회절(Fraunhofer Diffraction)

관측 거리가 매우 클 때 ($d \gg \frac{k(x'^2 + y'^2)_{\max}}{2}$, 즉 $d \gg a^2/\lambda$ ($a$는 조리개 크기)), 적분 내 이차 위상이 무시 가능해지고:

$$U(x, y, d) = \frac{e^{ikd}}{i\lambda d}e^{\frac{ik(x^2+y^2)}{2d}}\tilde{U}\!\left(\frac{x}{\lambda d}, \frac{y}{\lambda d}\right)$$

원거리 패턴은 공간 주파수 $f_x = x/(\lambda d)$, $f_y = y/(\lambda d)$에서 평가된 조리개 필드의 **푸리에 변환**이다.

**핵심 통찰**: 프라운호퍼 회절은 물리적 푸리에 변환이며, 변환 변수는 관측 각도이다.

### 3.3 프레넬 수(Fresnel Number)

프레넬 수 $N_F = a^2/(\lambda d)$는 회절 영역을 분류한다:
- $N_F \gg 1$: 근거리장 (프레넬 회절)
- $N_F \ll 1$: 원거리장 (프라운호퍼 회절)
- $N_F \sim 1$: 전환 영역

---

## 4. 푸리에 변환기로서의 렌즈

### 4.1 얇은 렌즈의 위상 변환

초점 거리 $f$의 얇은 렌즈는 이차 위상을 도입한다:

$$t_{\text{lens}}(x, y) = \exp\!\left(-\frac{ik(x^2 + y^2)}{2f}\right)$$

이는 프레넬 전파 커널과 같은 수학적 형태이지만 부호가 반대이다. 렌즈는 거리 $f$를 전파하는 동안 누적된 이차 위상을 정확히 상쇄한다.

### 4.2 푸리에 변환 특성

렌즈의 **전초점면(front focal plane)** ($z = -f$)에 입력 투명체 $U_{\text{in}}(x, y)$를 놓는다. **후초점면(back focal plane)** ($z = +f$)에서의 필드는:

$$\boxed{U_f(x, y) = \frac{1}{i\lambda f}\tilde{U}_{\text{in}}\!\left(\frac{x}{\lambda f}, \frac{y}{\lambda f}\right)}$$

이는 이차 위상 앞인수(prefactor)가 없는 **정확한** 푸리에 변환이다(근축 근사 내에서). 렌즈는 프라운호퍼 원거리장을 무한대에서 초점면으로 가져온다.

**물리적 의미**: 초점면의 각 점 $(x, y)$는 입력의 공간 주파수 성분에 해당한다:

$$f_x = \frac{x}{\lambda f}, \quad f_y = \frac{y}{\lambda f}$$

### 4.3 증명 개요

입력면 직후의 필드가 거리 $f$를 전파하여 렌즈에 도달하고(프레넬 전파), 렌즈를 통과하며(위상 곱), 다시 거리 $f$를 전파하여 출력면에 도달한다(프레넬 전파). 세 개의 이차 위상이 완벽하게 상쇄되어 순수한 푸리에 적분만 남는다. 이 상쇄가 초점면 결과가 깔끔한 푸리에 변환이 되는 이유이다.

### 4.4 스케일링 특성

푸리에 면에서의 공간 주파수 분해능은:

$$\delta x = \lambda f \cdot \delta f_x$$

따라서 긴 초점 거리 $f$는 스펙트럼을 더 넓은 영역에 펼쳐 높은 스펙트럼 분해능을 제공하고, 짧은 $f$는 이를 압축한다(낮은 분해능이지만 더 넓은 시야각).

---

## 5. 4f 시스템과 공간 필터링

### 5.1 4f 구성

**4f 시스템**은 두 렌즈가 각 초점 거리의 합만큼 떨어진 구성이며, 총 길이는 $4f$이다(동일한 렌즈를 가정할 때 이름의 유래):

```
입력       렌즈 1      푸리에       렌즈 2      출력
면           f          면           f           면
  │           │            │            │            │
  │     f     │     f      │     f      │     f      │
  │←────────→│←─────────→│←─────────→│←──────────→│
  │           │            │            │            │
  U_in      FT           F(fx,fy)     FT^(-1)     U_out
```

- 렌즈 1이 푸리에 면에서 $U_{\text{in}}$의 푸리에 변환을 계산
- 공간 필터(마스크) $H(f_x, f_y)$를 푸리에 면에 배치
- 렌즈 2가 역 푸리에 변환을 계산하여 필터링된 출력 생성

출력은:

$$U_{\text{out}}(x, y) = \mathcal{F}^{-1}\!\left\{\tilde{U}_{\text{in}}(f_x, f_y) \cdot H(f_x, f_y)\right\}$$

이것이 **광학적 합성곱(optical convolution)** 이다 — 선형 공간 필터의 광학적 구현. 전체 연산이 빛의 속도로, 모든 픽셀에 걸쳐 동시에 수행된다.

### 5.2 공간 필터링 예시

**저역 통과 필터(Low-pass filter)** (푸리에 면의 핀홀): 고공간 주파수를 차단하여 영상을 부드럽게 한다. 날카로운 에지와 노이즈를 제거.

$$H(f_x, f_y) = \begin{cases} 1 & \sqrt{f_x^2 + f_y^2} \leq f_c \\ 0 & \text{그 외} \end{cases}$$

**고역 통과 필터(High-pass filter)** (푸리에 면의 불투명 디스크): 저공간 주파수를 차단하여 에지와 세부 정보를 강조.

**대역 통과 필터(Band-pass filter)** (환형 개구): 특정 범위의 공간 주파수만 통과.

**방향성 필터(Directional filter)** (푸리에 면의 슬릿): 한 방향의 주파수만 통과시켜 특정 방향성을 가진 형상을 제거(예: 수평 스캔 라인 제거).

> **비유**: 4f 시스템은 영상을 위한 그래픽 이퀄라이저와 같다. 음악 이퀄라이저가 오디오 스펙트럼을 슬라이더 배열로 표시하여 저음(저주파)이나 고음(고주파)을 조절할 수 있게 하는 것처럼, 4f 시스템은 푸리에 면에 영상의 공간 스펙트럼을 표시하여 마스크로 어떤 공간 주파수 대역이든 물리적으로 차단하거나 수정할 수 있게 한다.

### 5.3 역사적 주석: 아베-포터 실험

1906년 앨버트 포터(Albert Porter)는 현미경의 푸리에 면에 다양한 마스크를 배치하여 아베의 이론을 시연했다. 격자 물체는 푸리에 면에 규칙적인 점 배열(회절 차수)을 만들었다. 특정 점을 차단함으로써, 그는 영상에서 수평 또는 수직 선을 사라지게 만들 수 있었다 — 영상이 공간 주파수 성분으로부터 조립된다는 것의 극적인 시연이었다.

---

## 6. 광학 전달 함수

### 6.1 코히런트 대 비코히런트 결상

결상 이론은 조명이 **코히런트(coherent)** 인지 **비코히런트(incoherent)** 인지에 따라 근본적으로 달라진다:

**코히런트 결상(Coherent imaging)** (예: 레이저 조명): 광학 시스템이 복소 필드 진폭에 대해 선형이다.

$$U_{\text{img}}(x, y) = h(x, y) * U_{\text{obj}}(x, y)$$

여기서 $h$는 진폭 점 확산 함수(코히런트 PSF)이다.

**비코히런트 결상(Incoherent imaging)** (예: 자연광): 시스템이 강도(intensity)에 대해 선형이다.

$$I_{\text{img}}(x, y) = |h(x, y)|^2 * I_{\text{obj}}(x, y)$$

### 6.2 점 확산 함수(PSF, Point Spread Function)

**PSF**는 점 광원의 상 — 결상 시스템의 임펄스 응답이다.

지름 $D$의 원형 조리개에서, 코히런트 PSF는 에어리 패턴(Airy pattern)이다:

$$h(r) \propto \frac{2J_1(\pi Dr/(\lambda f))}{\pi Dr/(\lambda f)}$$

강도 PSF(비코히런트)는 $|h(r)|^2$, 즉 에어리 원반(Airy disk)이다. 첫 번째 영점은:

$$r_{\text{Airy}} = 1.22\frac{\lambda f}{D}$$

이것이 **레일리 분해능 기준(Rayleigh resolution criterion)**이다: 한 점 광원의 최댓값이 다른 점 광원의 첫 번째 최솟값과 일치할 때 두 점 광원이 가까스로 분해된다.

### 6.3 코히런트 전달 함수(CTF, Coherent Transfer Function)

코히런트 결상에서, 주파수 영역의 전달 함수는:

$$H_{\text{coh}}(f_x, f_y) = P(\lambda f f_x, \lambda f f_y)$$

여기서 $P$는 동공 함수(pupil function)이다. 반경 $a$의 원형 조리개의 경우:

$$H_{\text{coh}}(f_r) = \begin{cases} 1 & f_r \leq f_c = a/(\lambda f) \\ 0 & f_r > f_c \end{cases}$$

코히런트 시스템은 $f_c$에서 급격한 차단을 가진 완벽한 저역 통과 필터이다.

### 6.4 광학 전달 함수(OTF) — 비코히런트

비코히런트 결상에서, 전달 함수는 **OTF**로 정의된다:

$$\text{OTF}(f_x, f_y) = \frac{\iint P(\xi, \eta)\,P^*(\xi - \lambda f f_x, \eta - \lambda f f_y)\,d\xi\,d\eta}{\iint |P(\xi, \eta)|^2\,d\xi\,d\eta}$$

이는 동공 함수의 **자기상관(autocorrelation)**으로, 원점에서 1로 정규화된다.

### 6.5 변조 전달 함수(MTF, Modulation Transfer Function)

**MTF**는 OTF의 크기이다:

$$\text{MTF}(f_x, f_y) = |\text{OTF}(f_x, f_y)|$$

각 공간 주파수에서 대비(contrast)가 얼마나 유지되는지를 기술한다. MTF = 1은 완전한 대비; MTF = 0은 해당 공간 주파수가 완전히 손실됨을 의미한다.

원형 조리개(수차 없음)의 경우, OTF/MTF는 다음 형태를 가진다:

$$\text{MTF}(f_r) = \frac{2}{\pi}\left[\arccos\!\left(\frac{f_r}{2f_c}\right) - \frac{f_r}{2f_c}\sqrt{1 - \left(\frac{f_r}{2f_c}\right)^2}\right]$$

$f_r \leq 2f_c$에서 유효하고, 그 외에서는 0이다. 비코히런트 차단 주파수는 코히런트 차단 주파수의 **두 배**이다: $f_{\text{max}} = 2f_c = D/(\lambda f)$.

### 6.6 코히런트 대 비코히런트 분해능

| 특성 | 코히런트 | 비코히런트 |
|------|----------|------------|
| 선형성의 기준 | 진폭 $U$ | 강도 $I$ |
| 전달 함수 | CTF (동공) | OTF (동공의 자기상관) |
| 차단 주파수 | $f_c = a/(\lambda f)$ | $2f_c = D/(\lambda f)$ |
| 위상 응답 | 있음 | 있음 (위상 전달 함수, PTF) |
| 에지 링잉(Edge ringing) | 있음 (코히런트 아티팩트) | 없음 (더 부드러움) |

비코히런트 시스템은 $2f_c$까지 주파수를 통과시켜 코히런트 차단 주파수의 두 배이지만, 대비는 감소한다. 어느 것이 "더 나은지"는 응용에 따라 다르다: 비코히런트 결상은 차단 주파수가 높지만 위상 정보가 없고, 코히런트 결상은 위상을 보존하지만 스페클(speckle)로 인한 문제가 있다.

---

## 7. 결상의 아베 이론

### 7.1 아베의 2단계 모델

에른스트 아베(Ernst Abbe, 1873)는 현미경 결상을 **2단계 과정**으로 기술했다:

1. **1단계**: 물체가 조명 빔을 회절시켜 회절 차수를 생성한다. 각 차수는 물체의 공간 주파수에 해당한다.

2. **2단계**: 현미경 대물렌즈가 이 회절 차수들을 수집하여 재결합하여 상을 형성한다. 렌즈에 입사한 차수만 상에 기여한다.

```
  조명       물체       대물        후초점       상
  면         (격자)     렌즈         면           면
     │              │           │            │            │
   ──┼──────────→   │ ↗ +1     │            │            │
     │              │ → 0      │        · +1 │            │
   ──┼──────────→   │ ↘ -1     │        · 0  │            │
     │              │           │        · -1 │            │
     │              │           │            │            │
     코히런트    회절 차수      렌즈      푸리에      재결합된
     평면파                    수집       면         차수가 상
                                                    형성
```

### 7.2 분해능 한계

주기 $d$의 격자에서, 1차 회절 각도는 $\sin\theta_1 = \lambda/d$이다. 대물렌즈가 이 차수를 수집하려면 $\sin\theta_1 \leq \text{NA}$이어야 한다:

$$d_{\min} = \frac{\lambda}{\text{NA}} \quad (\text{코히런트, 수직 입사})$$

사각 조명(컨덴서 NA를 가득 채움)으로 한계가 향상된다:

$$\boxed{d_{\min} = \frac{\lambda}{2\,\text{NA}}} \quad (\text{아베 분해능 한계})$$

이것이 원거리장 광학 현미경의 근본적인 분해능 한계이다. 가시광선($\lambda \approx 500\,\text{nm}$)과 고NA 오일 침지 대물렌즈($\text{NA} = 1.4$)의 경우:

$$d_{\min} = \frac{500}{2 \times 1.4} \approx 180\,\text{nm}$$

### 7.3 아베 한계를 넘어서

현대 초분해능 기법(STED, PALM, STORM)은 비선형 형광 반응이나 확률론적 단일 분자 위치 결정을 이용하여 이 한계를 우회한다. 이 방법들은 2014년 노벨 화학상을 수상했다(베치그(Betzig), 헬(Hell), 모너(Moerner)).

---

## 8. 위상차 현미경

### 8.1 문제: 위상 물체(Phase Objects)

많은 생물 시편(세포, 박테리아)은 거의 투명하여 — 빛을 거의 흡수하지 않지만 서로 다른 굴절률 영역을 통과하면서 위상을 변화시킨다. 표준 명시야(bright-field) 현미경은 위상이 아닌 강도를 검출하므로, 이런 시편은 거의 보이지 않는다.

수학적으로, 얇은 위상 물체의 투과율은:

$$t(x, y) = e^{i\phi(x, y)} \approx 1 + i\phi(x, y) \quad (\text{작은 } \phi \text{의 경우})$$

강도는 $|t|^2 = 1 + \phi^2 \approx 1$ — 위상 정보가 손실된다.

### 8.2 제르니케의 해결책

프리츠 제르니케(Frits Zernike, 1934)는 이 문제가 공간 필터링의 문제임을 깨달았다. 푸리에 면에서:

- "1" (DC 성분, 회절되지 않은 빛)은 중앙에 밝은 점이 됨
- "$i\phi$" 항 (위상 정보를 담은 회절광)은 면 전체로 퍼짐

회절광은 DC 성분과 $\pi/2$ 위상차를 가진다($i$ 인수). 이들의 간섭은 위상 이동이 $\pi/2$이기 때문에 강도 변화를 만들지 않는다(예를 들어: $|1 + i\epsilon|^2 \approx 1$).

제르니케의 통찰: 푸리에 면 중앙의 소형 위상판(위상 링)을 이용하여 **DC 성분에 추가적인 $\pm\pi/2$를 이동**시키는 것이다:

$$\text{위상판}: \quad H(0,0) = e^{\pm i\pi/2} = \pm i$$

이제 전체 필드는:

$$U_{\text{out}} \propto \pm i + i\phi = i(\pm 1 + \phi)$$

강도는:

$$I \propto |(\pm 1 + \phi)|^2 = 1 \pm 2\phi + \phi^2 \approx 1 \pm 2\phi$$

위상 $\phi$가 이제 **강도에 선형으로** 나타난다 — 위상 물체가 가시화된다!

- **양의 위상차(Positive phase contrast)** ($+\pi/2$ 이동): 위상이 앞선 영역이 회색 배경에서 밝게 보임
- **음의 위상차(Negative phase contrast)** ($-\pi/2$ 이동): 위상이 앞선 영역이 어둡게 보임

### 8.3 실용적 구현

현미경에서:
1. 환형 컨덴서 조리개가 중공 원추형 조명을 생성
2. 회절되지 않은 빛이 대물렌즈 후초점면의 일치하는 환형 위상 링을 통과
3. 회절광은 대부분 링을 빗나가 영향 없이 통과
4. 링의 $\pi/2$ 위상 이동이 위상 차이를 강도 차이로 변환

이는 염색이나 처리 없이 — 살아있는 세포를 실시간으로 관찰할 수 있다.

---

## 9. Python 예제

### 9.1 프라운호퍼 회절과 푸리에 변환

```python
import numpy as np
import matplotlib.pyplot as plt

def fraunhofer_diffraction(aperture, wavelength, z, pixel_size):
    """
    Compute the Fraunhofer diffraction pattern of an aperture.

    In the far field, the diffracted field is the Fourier transform
    of the aperture function. We use FFT to compute this numerically.
    The physical coordinates in the observation plane are related to
    spatial frequencies by (x, y) = lambda * z * (fx, fy).
    """
    N = aperture.shape[0]

    # Compute 2D FFT — this IS the Fraunhofer integral
    # fftshift centers the zero-frequency component
    U_far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture)))

    # Intensity pattern
    I_far = np.abs(U_far)**2
    I_far /= I_far.max()  # Normalize to peak

    # Physical coordinates in observation plane
    df = 1.0 / (N * pixel_size)  # Frequency spacing
    fx = np.arange(-N//2, N//2) * df
    x_obs = wavelength * z * fx  # Physical coordinates

    return I_far, x_obs

# --- Simulate diffraction from different apertures ---
N = 1024
pixel_size = 1e-6  # 1 µm per pixel
wavelength = 500e-9  # 500 nm (green light)
z = 1.0  # 1 meter propagation

# Coordinate grid
x = np.arange(-N//2, N//2) * pixel_size
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)

# Circular aperture (diameter = 200 µm)
D = 200e-6
aperture_circ = (R <= D/2).astype(float)

# Square aperture (side = 200 µm)
a = 200e-6
aperture_sq = ((np.abs(X) <= a/2) & (np.abs(Y) <= a/2)).astype(float)

# Double slit (slit width = 50 µm, separation = 200 µm)
slit_w = 50e-6
slit_sep = 200e-6
slit_h = 400e-6
aperture_ds = (
    ((np.abs(X - slit_sep/2) <= slit_w/2) | (np.abs(X + slit_sep/2) <= slit_w/2))
    & (np.abs(Y) <= slit_h/2)
).astype(float)

fig, axes = plt.subplots(3, 2, figsize=(12, 14))
apertures = [aperture_circ, aperture_sq, aperture_ds]
titles = ['Circular Aperture', 'Square Aperture', 'Double Slit']

for i, (ap, title) in enumerate(zip(apertures, titles)):
    # Aperture
    extent_ap = [x[0]*1e3, x[-1]*1e3, x[0]*1e3, x[-1]*1e3]
    axes[i, 0].imshow(ap, cmap='gray', extent=extent_ap)
    axes[i, 0].set_title(f'{title} (Object Plane)', fontsize=11)
    axes[i, 0].set_xlabel('x [mm]')
    axes[i, 0].set_ylabel('y [mm]')

    # Diffraction pattern (log scale for visibility)
    I_far, x_obs = fraunhofer_diffraction(ap, wavelength, z, pixel_size)
    extent_diff = [x_obs[0]*1e3, x_obs[-1]*1e3, x_obs[0]*1e3, x_obs[-1]*1e3]
    axes[i, 1].imshow(np.log10(I_far + 1e-6), cmap='inferno',
                       extent=extent_diff, vmin=-4, vmax=0)
    axes[i, 1].set_title(f'{title} (Fraunhofer Pattern, log)', fontsize=11)
    axes[i, 1].set_xlabel('x [mm]')
    axes[i, 1].set_ylabel('y [mm]')
    axes[i, 1].set_xlim(-5, 5)
    axes[i, 1].set_ylim(-5, 5)

plt.tight_layout()
plt.savefig('fraunhofer_diffraction.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.2 4f 공간 필터링 시스템

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import face  # Test image

def spatial_filter_4f(image, filter_mask):
    """
    Simulate a 4f spatial filtering system.

    Steps:
    1. Lens 1: Fourier transform the input image
    2. Multiply by the filter mask in the Fourier plane
    3. Lens 2: inverse Fourier transform to get the filtered output

    This is optically equivalent to convolution with the inverse FT of the mask.
    """
    # Fourier transform (Lens 1)
    spectrum = np.fft.fftshift(np.fft.fft2(image))

    # Apply filter in Fourier plane
    filtered_spectrum = spectrum * filter_mask

    # Inverse Fourier transform (Lens 2)
    output = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum))

    return np.abs(output), np.abs(spectrum), np.abs(filtered_spectrum)

# Load and prepare test image (grayscale)
img = face(gray=True).astype(float)
img = img[:512, :512]  # Crop to square
img /= img.max()

Ny, Nx = img.shape
fy = np.fft.fftshift(np.fft.fftfreq(Ny))
fx = np.fft.fftshift(np.fft.fftfreq(Nx))
FX, FY = np.meshgrid(fx, fy)
FR = np.sqrt(FX**2 + FY**2)

# Define filter masks
# 1. Low-pass: keep only low spatial frequencies (blur)
cutoff_low = 0.05  # Normalized frequency
mask_lowpass = (FR <= cutoff_low).astype(float)

# 2. High-pass: remove low frequencies (edge detection)
cutoff_high = 0.02
mask_highpass = (FR > cutoff_high).astype(float)

# 3. Band-pass: keep middle frequencies
mask_bandpass = ((FR > 0.02) & (FR < 0.1)).astype(float)

# 4. Directional: vertical slit (pass horizontal details only)
mask_directional = (np.abs(FX) < 0.01).astype(float)

filters = [mask_lowpass, mask_highpass, mask_bandpass, mask_directional]
names = ['Low-pass (blur)', 'High-pass (edges)', 'Band-pass', 'Vertical slit']

fig, axes = plt.subplots(4, 3, figsize=(14, 16))
fig.suptitle('4f Spatial Filtering System', fontsize=16, y=1.01)

for i, (mask, name) in enumerate(zip(filters, names)):
    output, spectrum, filt_spectrum = spatial_filter_4f(img, mask)

    # Filter mask
    axes[i, 0].imshow(mask, cmap='gray', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[i, 0].set_title(f'Filter: {name}', fontsize=10)
    axes[i, 0].set_xlabel('fx')
    axes[i, 0].set_ylabel('fy')

    # Filtered spectrum
    axes[i, 1].imshow(np.log10(filt_spectrum + 1), cmap='inferno')
    axes[i, 1].set_title('Filtered Spectrum (log)', fontsize=10)
    axes[i, 1].axis('off')

    # Output image
    axes[i, 2].imshow(output, cmap='gray')
    axes[i, 2].set_title('Output Image', fontsize=10)
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig('4f_spatial_filtering.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 9.3 원형 조리개의 MTF

```python
import numpy as np
import matplotlib.pyplot as plt

def mtf_circular(f_norm):
    """
    Analytical MTF for a diffraction-limited circular aperture.

    The MTF is the autocorrelation of the circular pupil, normalized
    to unity at zero frequency. For an aberration-free system, this
    represents the theoretical maximum contrast at each spatial frequency.
    Below this curve, you can never do better without super-resolution tricks.
    """
    # Normalized frequency: f_norm = f / f_cutoff, range [0, 1]
    f = np.clip(f_norm, 0, 1)
    mtf = (2/np.pi) * (np.arccos(f) - f * np.sqrt(1 - f**2))
    return mtf

f = np.linspace(0, 1, 500)
mtf_vals = mtf_circular(f)

# Compare with coherent transfer function (CTF)
ctf = np.where(f <= 0.5, 1.0, 0.0)  # CTF cuts off at f_c = f_max/2

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(f, mtf_vals, 'b-', linewidth=2, label='Incoherent MTF')
ax.plot(f, ctf, 'r--', linewidth=2, label='Coherent CTF (at f/f_inc_cutoff)')
ax.fill_between(f, 0, mtf_vals, alpha=0.1, color='blue')

ax.set_xlabel('Normalized spatial frequency (f / f_cutoff)', fontsize=12)
ax.set_ylabel('Transfer function value', fontsize=12)
ax.set_title('MTF and CTF of a Circular Aperture (Aberration-Free)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1.05)
ax.set_ylim(-0.05, 1.1)

# Annotate
ax.annotate('Incoherent cutoff\n(f = D/λf)', xy=(1.0, 0), fontsize=9,
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color='blue'),
            xytext=(0.85, 0.15))
ax.annotate('Coherent cutoff\n(f = D/2λf)', xy=(0.5, 0), fontsize=9,
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color='red'),
            xytext=(0.35, 0.15))

plt.tight_layout()
plt.savefig('mtf_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 10. 요약

| 개념 | 핵심 공식 / 아이디어 |
|------|----------------------|
| 회절의 LSI 시스템 | $\tilde{U}_{\text{out}} = \tilde{U}_{\text{in}} \cdot H(f_x, f_y)$ |
| 자유 공간 전달 함수 | $H = \exp\!\left(i2\pi d\sqrt{1/\lambda^2 - f_x^2 - f_y^2}\right)$ |
| 에반에센트파 | $f_x^2 + f_y^2 > 1/\lambda^2 \Rightarrow$ 지수적 감쇠 |
| 프레넬 근사 | $H \approx e^{ikd}\exp(-i\pi\lambda d(f_x^2+f_y^2))$ |
| 프라운호퍼 회절 | 원거리장 $\propto$ 조리개의 푸리에 변환 |
| 렌즈의 FT 특성 | $U_f = \frac{1}{i\lambda f}\tilde{U}_{\text{in}}(x/\lambda f, y/\lambda f)$ |
| 4f 시스템 | 공간 필터링: FT → 필터 → 역 FT |
| PSF (원형 조리개) | 에어리 원반; 첫 번째 영점 $r = 1.22\lambda f/D$ |
| 코히런트 차단 주파수 | $f_c = a/(\lambda f) = D/(2\lambda f)$ |
| 비코히런트 차단 주파수 | $2f_c = D/(\lambda f)$ — 코히런트의 두 배 |
| MTF | $|\text{OTF}|$; 공간 주파수별 대비를 기술 |
| 아베 분해능 | $d_{\min} = \lambda/(2\,\text{NA})$ |
| 위상차 현미경 | DC의 $\pi/2$ 위상 이동으로 위상 → 강도 변환 |

---

## 11. 연습 문제

### 연습 문제 1: 각 스펙트럼 전파

균일한 평면파($\lambda = 633\,\text{nm}$)가 폭 $a = 100\,\mu\text{m}$의 슬릿을 조명한다.

(a) 슬릿 투과율의 각 스펙트럼(푸리에 변환)을 쓰라.
(b) 에반에센트파가 시작하는 공간 주파수는?
(c) Python을 이용하여 $d = 1\,\text{cm}$와 $d = 1\,\text{m}$을 통해 각 스펙트럼을 전파하라. 결과를 비교하고 어느 것이 프레넬 영역이고 어느 것이 프라운호퍼 영역인지 식별하라.

### 연습 문제 2: 4f 시스템 설계

$\lambda = 532\,\text{nm}$에서 $f = 200\,\text{mm}$의 렌즈를 사용하는 4f 시스템을 설계하라.

(a) 물체가 0에서 50 선/mm까지의 공간 주파수 성분을 가진다. 필터 면에서 푸리에 스펙트럼의 물리적 크기는?
(b) 공간 주파수를 10 선/mm까지만 통과시키는 저역 통과 필터를 만들려면, 필요한 핀홀 지름은?
(c) Python으로 이 필터를 구현하고, 격자 패턴을 포함한 테스트 영상에 미치는 효과를 보여라.

### 연습 문제 3: MTF 분석

카메라 렌즈가 $f = 50\,\text{mm}$, $f/\# = 2.8$ (조리개 $D = 17.9\,\text{mm}$)이고, $\lambda = 550\,\text{nm}$에서 사용된다.

(a) 회절 한계 MTF 차단 주파수(비코히런트)를 계산하라.
(b) 회절 한계 MTF를 그래프로 그려라.
(c) 검출기의 픽셀 피치가 5 $\mu\text{m}$이다. 검출기의 나이퀴스트(Nyquist) 한계는 어떤 공간 주파수에서 발생하는가? 이 시스템은 회절 한계인가, 아니면 검출기 한계인가?

### 연습 문제 4: 위상차 현미경

제르니케 위상차 현미경을 시뮬레이션하라:

(a) 2D 위상 물체를 생성하라: $t(x,y) = \exp(i\phi(x,y))$, 여기서 $\phi$는 배경이 $\phi = 0$인 상태에서 $\phi = 0.3$ rad의 원형 영역을 가진다.
(b) 명시야 강도 영상(필터 없음)을 계산하고 표시하라. 거의 균일함을 확인하라.
(c) 푸리에 면의 DC 성분에 $\pi/2$ 위상 링 필터를 적용하라. 결과 영상을 표시하고 위상 물체가 이제 가시화되었음을 확인하라.

---

## 12. 참고문헌

1. Goodman, J. W. (2017). *Introduction to Fourier Optics* (4th ed.). W. H. Freeman. — 푸리에 광학의 표준 참고서.
2. Saleh, B. E. A., & Teich, M. C. (2019). *Fundamentals of Photonics* (3rd ed.). Wiley. — Chapters 4-5.
3. Hecht, E. (2017). *Optics* (5th ed.). Pearson. — Chapter 11.
4. Born, M., & Wolf, E. (2019). *Principles of Optics* (7th expanded ed.). Cambridge University Press.
5. Abbe, E. (1873). "Beitrage zur Theorie des Mikroskops und der mikroskopischen Wahrnehmung." *Archiv fur Mikroskopische Anatomie*, 9, 413-468.
6. Zernike, F. (1955). Nobel Lecture: "How I Discovered Phase Contrast."

---

[← 이전: 09. 광섬유](09_Fiber_Optics.md) | [다음: 11. 홀로그래피 →](11_Holography.md)
