# 15. 체르니케 다항식(Zernike Polynomials)

[← 이전: 14. 전산 광학](14_Computational_Optics.md) | [다음: 16. 적응광학 →](16_Adaptive_Optics.md)

---

광학 시스템이 완벽한 상을 만들어 내지 못할 때, 그 원인은 거의 항상 왜곡된 파면(wavefront)에 있습니다. 약간 정렬이 틀어진 망원경 거울, 제조 불량이 있는 렌즈, 혹은 관측소 상공의 난류 대기 중 어느 것이 원인이든 간에, 결과적으로 발생하는 파면 오차(wavefront error) — 이상적인 구면으로부터의 편차 — 는 상의 질을 저하시킵니다. 이 오차를 진단하고 보정하기 위해서는 원형 개구(circular aperture) 위의 파면 형상을 정밀하게 기술하는 수학적 언어가 필요합니다.

체르니케 다항식(Zernike polynomials)은 1930년대 프리츠 체르니케(Frits Zernike, 위상차 현미경을 발명한 바로 그 물리학자)에 의해 도입되어, 정확히 이 역할을 수행합니다. 이 다항식들은 단위 원(unit circle) 위에 정의된 완전 정규직교(complete orthonormal) 기저 집합을 이루며, 대부분의 광학 개구가 갖는 자연스러운 기하 형태에 부합합니다. 각 체르니케 모드는 기울기(tilt), 초점이탈(defocus), 코마(coma), 비점수차(astigmatism), 구면수차(spherical aberration)와 같이 인식 가능한 광학 수차에 대응하므로, 푸리에 모드(Fourier modes)와 같은 다른 기저계보다 물리적으로 훨씬 직관적입니다. 오늘날 체르니케 다항식은 광학 시험, 적응 광학(adaptive optics), 안과학(ophthalmology), 광학 설계에서 파면 분석의 표준 도구입니다.

이 레슨은 체르니케 다항식을 수학적 정의로부터 실용적인 파면 분석과 대기 난류 모델링까지 전개하며, 레슨 14(전산 광학, §5.3)에서의 간략한 소개를 바탕으로 합니다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. 단위 원 위에서 체르니케 다항식을 정의하고 반경 및 각도 성분을 유도한다
2. 놀 인덱싱 규약(Noll indexing convention)을 적용하여 단일 인덱스로 체르니케 모드를 열거한다
3. 단위 원 위에서 체르니케 다항식의 정규직교성을 증명하고 파면 분해에 대한 의의를 설명한다
4. 처음 21개 체르니케 모드의 물리적 의미와 점퍼짐함수(PSF)에 미치는 영향을 파악한다
5. 내적과 기울기 측정으로부터의 최소 제곱 피팅을 이용하여 파면 분해를 수행한다
6. 콜모고로프 난류 통계(Kolmogorov turbulence statistics)를 기술하고 대기 위상 스크린을 위한 놀 공분산 행렬을 유도한다
7. 체르니케 계수로부터 RMS 파면 오차를 계산하고 마레샬 근사(Maréchal approximation)를 통해 스트렐 비율(Strehl ratio)을 추정한다
8. 파이썬(Python)으로 체르니케 모드 생성, 파면 피팅, 난류 위상 스크린 시뮬레이션을 구현한다

---

## 목차

1. [파면 수차 개요](#1-파면-수차-개요)
2. [체르니케 다항식의 정의](#2-체르니케-다항식의-정의)
3. [놀 인덱싱 규약](#3-놀-인덱싱-규약)
4. [직교성과 완전성](#4-직교성과-완전성)
5. [모드의 물리적 해석](#5-모드의-물리적-해석)
6. [파면 분해와 피팅](#6-파면-분해와-피팅)
7. [대기 난류와 콜모고로프 통계](#7-대기-난류와-콜모고로프-통계)
8. [RMS 파면 오차와 스트렐 비율](#8-rms-파면-오차와-스트렐-비율)
9. [파이썬 예제](#9-파이썬-예제)
10. [요약](#10-요약)
11. [연습 문제](#11-연습-문제)
12. [참고문헌](#12-참고문헌)

---

## 1. 파면 수차 개요

### 1.1 이상적인 파면

이상적인 결상 시스템은 점 광원을 상 평면(image plane)에서 회절 한계 점으로 수렴하는 구면 파면으로 변환합니다. 파면 — 등위상면(surface of constant optical phase) — 은 상점(image point)을 중심으로 하는 완전한 구면입니다. 이로부터 얻어지는 상은 에어리 패턴(Airy pattern)으로, 회절이 허용하는 최선의 결과입니다.

### 1.2 광로차와 파면 오차

실제로는 모든 광학 요소가 이상적인 구면으로부터의 편차를 유발합니다. **파면 오차(wavefront error)** $W(\rho, \theta)$를 파면 수 또는 마이크로미터로 측정한, 실제 파면과 이상적인 기준 구면 사이의 광로차(OPD, Optical Path Difference)로 정의합니다:

$$W(\rho, \theta) = \text{OPD}(\rho, \theta) = n \cdot \Delta z(\rho, \theta)$$

여기서 $\rho$와 $\theta$는 사출 동공(exit pupil) 위의 극좌표($\rho \in [0, 1]$로 정규화), $n$은 굴절률, $\Delta z$는 물리적 표면 편차입니다.

### 1.3 원형 기저가 필요한 이유

대부분의 광학 시스템은 원형 개구를 가집니다 — 망원경의 주경, 카메라 렌즈, 인간의 눈. 원 위에 정의된 파면 기저는 이러한 기하 형태에 자연스럽게 대응합니다. 푸리에 모드는 직사각형 영역에서, 르장드르 다항식(Legendre polynomials)은 구간(interval) 위에서 작동하지만, **체르니케 다항식은 단위 원판(unit disk) 위에서 직교하며 반경 인수와 각도 인수로 깔끔하게 분리되는 유일한 다항식 집합입니다**.

> **유비**: 푸리에 급수가 시간 신호를 각각 명확한 주파수 해석을 갖는 사인 및 코사인 조화파들로 분해하듯이, 체르니케 다항식은 원형 개구 위의 파면을 "수차 조화파들"로 분해합니다. 각 모드는 명확한 광학적 해석을 가집니다: "기본" 모드는 기울기와 초점이탈이고, 고차 모드는 코마와 구면수차 같이 점점 더 세밀한 파면 구조를 나타냅니다.

---

## 2. 체르니케 다항식의 정의

### 2.1 반경 다항식

체르니케 다항식은 단위 원 $\rho \in [0, 1]$, $\theta \in [0, 2\pi)$ 위에서 반경 부분과 각도 부분의 곱으로 정의됩니다. 차수 $n$과 방위 주파수 $m$을 갖는 반경 다항식은 다음과 같습니다:

$$R_n^{|m|}(\rho) = \sum_{s=0}^{(n-|m|)/2} \frac{(-1)^s (n-s)!}{s! \left(\frac{n+|m|}{2}-s\right)! \left(\frac{n-|m|}{2}-s\right)!} \rho^{n-2s}$$

인덱스는 두 가지 조건을 만족해야 합니다:
- $n \geq 0$은 반경 차수(radial order)
- $|m| \leq n$이고 $n - |m|$은 짝수 ($(n - |m|)/2$가 음이 아닌 정수가 되도록)

예를 들면:
- $R_0^0(\rho) = 1$ (상수)
- $R_1^1(\rho) = \rho$ (선형)
- $R_2^0(\rho) = 2\rho^2 - 1$ (이차, 초점이탈과 관련)
- $R_2^2(\rho) = \rho^2$ (이차, 비점수차와 관련)
- $R_3^1(\rho) = 3\rho^3 - 2\rho$ (삼차, 코마와 관련)
- $R_4^0(\rho) = 6\rho^4 - 6\rho^2 + 1$ (사차, 구면수차와 관련)

### 2.2 완전한 체르니케 다항식

완전한 체르니케 다항식은 반경 함수와 삼각 함수 형태의 각도 의존성을 결합합니다:

$$Z_n^m(\rho, \theta) = \begin{cases} N_n^m R_n^{|m|}(\rho) \cos(m\theta) & \text{if } m \geq 0 \\ N_n^{|m|} R_n^{|m|}(\rho) \sin(|m|\theta) & \text{if } m < 0 \end{cases}$$

여기서 정규직교성을 보장하는 정규화 인수는 다음과 같습니다:

$$N_n^m = \sqrt{\frac{2(n+1)}{1 + \delta_{m0}}}$$

여기서 $\delta_{m0}$는 크로네커 델타(Kronecker delta) ($m = 0$이면 $\delta_{m0} = 1$, 아니면 $0$)입니다. $2(n+1)$ 인수는 반경 적분으로부터 오고, $(1 + \delta_{m0})$는 각도 평균을 처리합니다 (코사인 및 사인 모드는 $\pi$로 적분되지만, $m = 0$인 경우는 $2\pi$로 적분됩니다).

### 2.3 인덱스 범위와 모드 수

주어진 최대 반경 차수 $n_{\max}$에 대해 체르니케 모드의 총 수는:

$$N_{\text{modes}} = \frac{(n_{\max}+1)(n_{\max}+2)}{2}$$

| $n_{\max}$ | 모드 수 | 포함되는 수차 |
|:-----------:|:-----:|----------|
| 1 | 3 | 피스톤(Piston), 팁(Tip), 틸트(Tilt) |
| 2 | 6 | + 초점이탈(Defocus), 비점수차(Astigmatism) |
| 3 | 10 | + 코마(Coma), 세잎수차(Trefoil) |
| 4 | 15 | + 구면수차(Spherical), 2차 비점수차, 네잎수차(Quadrafoil) |
| 5 | 21 | + 2차 코마, 2차 세잎수차, 다섯잎수차(Pentafoil) |
| 6 | 28 | + 2차 구면수차, ... |

---

## 3. 놀 인덱싱 규약

### 3.1 단일 인덱스 문제

이중 인덱스 $(n, m)$ 표기법은 수학적으로 자연스럽지만, 벡터 안에서 모드를 순서짓는 데는 불편합니다. 여러 단일 인덱스 방식이 존재하며, 가장 널리 사용되는 것은 각 모드에 단일 인덱스 $j = 1, 2, 3, \ldots$를 할당하는 **놀 규약(Noll convention)** (Noll, 1976)입니다.

### 3.2 놀 정렬 규칙

놀의 방식은 증가하는 반경 차수 $n$ 순서로 모드를 정렬하며, 같은 차수 내에서 다음 규칙을 따릅니다:

1. 같은 $|m|$에 대해 짝수 $m$ (코사인) 모드가 홀수 $m$ (사인) 모드보다 앞에 옵니다
2. 같은 $n$ 내에서, 모드는 $|m|$가 증가하는 순서로 정렬됩니다
3. 같은 $n$과 $|m|$에 대해, 코사인 항(짝수 $j$)이 사인 항(홀수 $j$)보다 앞에 옵니다

처음 21개 모드에 대한 $j$에서 $(n, m)$으로의 대응표:

| $j$ | $n$ | $m$ | 이름 | 식 |
|:---:|:---:|:---:|------|------------|
| 1 | 0 | 0 | 피스톤(Piston) | $1$ |
| 2 | 1 | 1 | 틸트 (x) | $2\rho\cos\theta$ |
| 3 | 1 | −1 | 틸트 (y) | $2\rho\sin\theta$ |
| 4 | 2 | 0 | 초점이탈(Defocus) | $\sqrt{3}(2\rho^2 - 1)$ |
| 5 | 2 | −2 | 비점수차 (사선) | $\sqrt{6}\rho^2\sin 2\theta$ |
| 6 | 2 | 2 | 비점수차 (수직) | $\sqrt{6}\rho^2\cos 2\theta$ |
| 7 | 3 | −1 | 코마 (수직) | $\sqrt{8}(3\rho^3 - 2\rho)\sin\theta$ |
| 8 | 3 | 1 | 코마 (수평) | $\sqrt{8}(3\rho^3 - 2\rho)\cos\theta$ |
| 9 | 3 | −3 | 세잎수차 (수직) | $\sqrt{8}\rho^3\sin 3\theta$ |
| 10 | 3 | 3 | 세잎수차 (사선) | $\sqrt{8}\rho^3\cos 3\theta$ |
| 11 | 4 | 0 | 구면수차(Spherical) | $\sqrt{5}(6\rho^4 - 6\rho^2 + 1)$ |
| 12 | 4 | 2 | 2차 비점수차 (수직) | $\sqrt{10}(4\rho^4 - 3\rho^2)\cos 2\theta$ |
| 13 | 4 | −2 | 2차 비점수차 (사선) | $\sqrt{10}(4\rho^4 - 3\rho^2)\sin 2\theta$ |
| 14 | 4 | 4 | 네잎수차 (수직) | $\sqrt{10}\rho^4\cos 4\theta$ |
| 15 | 4 | −4 | 네잎수차 (사선) | $\sqrt{10}\rho^4\sin 4\theta$ |
| 16 | 5 | 1 | 2차 코마 (수평) | $\sqrt{12}(10\rho^5 - 12\rho^3 + 3\rho)\cos\theta$ |
| 17 | 5 | −1 | 2차 코마 (수직) | $\sqrt{12}(10\rho^5 - 12\rho^3 + 3\rho)\sin\theta$ |
| 18 | 5 | 3 | 2차 세잎수차 (사선) | $\sqrt{12}(5\rho^5 - 4\rho^3)\cos 3\theta$ |
| 19 | 5 | −3 | 2차 세잎수차 (수직) | $\sqrt{12}(5\rho^5 - 4\rho^3)\sin 3\theta$ |
| 20 | 5 | 5 | 다섯잎수차 (수평) | $\sqrt{12}\rho^5\cos 5\theta$ |
| 21 | 5 | −5 | 다섯잎수차 (수직) | $\sqrt{12}\rho^5\sin 5\theta$ |

### 3.3 변환 알고리즘

놀 인덱스 $j$에서 $(n, m)$으로의 변환은 다음 단계를 따릅니다:

1. $n(n+1)/2 < j \leq (n+1)(n+2)/2$를 만족하는 $n$을 구한다
2. 해당 행 내의 위치를 계산한다: $k = j - n(n+1)/2$
3. $k$와 $n$의 홀짝성으로부터 $|m|$을 결정한다
4. $j$가 짝수이면 코사인($m > 0$), 홀수이면 사인($m < 0$)에 따라 $m$의 부호를 할당한다 — 단, $m = 0$ 모드는 $j$가 짝수인 예외가 있다

> **참고**: 다른 인덱싱 규약도 존재합니다. **ANSI/OSA 표준** (Thibos 등, 2002)은 각 $n$ 내에서 $|m|$ 기준으로 정렬하고, 음수 $m$이 양수 $m$보다 앞에 오는 다른 순서를 사용합니다. 논문을 읽거나 광학 소프트웨어와 연동할 때는 항상 어느 규약이 사용되고 있는지 확인하십시오.

---

## 4. 직교성과 완전성

### 4.1 정규직교성 관계

체르니케 다항식의 핵심 특성은 단위 원판 위에서의 정규직교성입니다:

$$\int_0^1 \int_0^{2\pi} Z_j(\rho, \theta) Z_{j'}(\rho, \theta) \, \rho \, d\rho \, d\theta = \pi \, \delta_{jj'}$$

이는 다음을 의미합니다:
- 서로 다른 모드는 직교합니다: 겹침 적분(overlap integral)이 0
- 각 모드는 $\pi$ (단위 원판의 넓이)로 정규화됩니다

일부 참고문헌에서는 $\pi$로 나누어 $\langle Z_j, Z_{j'} \rangle = \delta_{jj'}$로 단위 정규화합니다.

### 4.2 증명 개요

직교성은 두 개의 독립적인 인수로부터 유도됩니다:

**각도 직교성(Angular orthogonality)**: 삼각 함수들은 다음을 만족합니다:
$$\int_0^{2\pi} \cos(m\theta)\cos(m'\theta)\,d\theta = \pi(1+\delta_{m0})\delta_{mm'}$$
$$\int_0^{2\pi} \sin(m\theta)\sin(m'\theta)\,d\theta = \pi\delta_{mm'} \quad (m, m' \neq 0)$$
$$\int_0^{2\pi} \cos(m\theta)\sin(m'\theta)\,d\theta = 0$$

**반경 직교성(Radial orthogonality)**: 같은 $|m|$을 갖는 다항식들에 대해:
$$\int_0^1 R_n^{|m|}(\rho) R_{n'}^{|m|}(\rho) \, \rho \, d\rho = \frac{\delta_{nn'}}{2(n+1)}$$

이 결과는 자명하지 않으며, 체르니케 반경 다항식과 야코비 다항식(Jacobi polynomials) 사이의 연결로부터 도출됩니다:

$$R_n^{|m|}(\rho) = (-1)^{(n-|m|)/2} P_{(n-|m|)/2}^{(0, |m|)}(1 - 2\rho^2)$$

여기서 $P_k^{(\alpha, \beta)}$는 가중치 $(1-x)^\alpha(1+x)^\beta$에 대해 직교하는 야코비 다항식입니다. 변수 치환 $x = 1 - 2\rho^2$을 통해 반경 적분이 표준 야코비 직교성 관계로 변환됩니다.

### 4.3 완전성

체르니케 다항식은 단위 원판 위의 제곱 적분 가능한 함수에 대한 **완전한(complete)** 기저를 형성합니다. $\int\!\int |W|^2 \rho\,d\rho\,d\theta < \infty$를 만족하는 임의의 파면 $W(\rho, \theta)$는 다음과 같이 전개될 수 있습니다:

$$W(\rho, \theta) = \sum_{j=1}^{\infty} a_j Z_j(\rho, \theta)$$

실제로는 어떤 최대 차수 $j_{\max}$에서 절단하며, $j_{\max}$가 증가할수록 절단 오차는 감소합니다.

### 4.4 왜 푸리에 모드를 쓰지 않는가?

푸리에 모드(직교 좌표계의 사인과 코사인)는 *직사각형* 위에서 직교하지만, 원형 영역으로 제한하면 직교성을 잃습니다. 이는 원형 개구 위에서의 푸리에 계수들이 결합(coupled)되어 있음을 의미합니다 — 하나의 계수를 변경하면 다른 계수들의 해석에 영향을 줍니다. 체르니케 다항식은 원형 기하에 맞게 특별히 설계되었기 때문에 이 문제를 완전히 피할 수 있습니다.

| 속성 | 체르니케(Zernike) | 푸리에(Fourier) |
|----------|---------|---------|
| 정의 영역 | 단위 원 | 직사각형 |
| 원 위에서의 직교성 | 있음 | 없음 |
| 물리적 해석 | 각 모드 = 이름 있는 수차 | 직접적인 수차 의미 없음 |
| 계산 비용 | 전 모드에 $O(j_{\max}^2)$ | FFT를 통해 $O(N \log N)$ |
| 환형 개구(Annular apertures) | 수정된 체르니케 필요 | 여전히 작동 |

---

## 5. 모드의 물리적 해석

### 5.1 저차 수차

각 체르니케 모드는 고전적인 광학 수차에 대응합니다. 이 모드들을 이해하는 것은 광학 엔지니어에게 필수적이며, 수학적 계수를 상질에 대한 물리적 효과와 연결시켜 줍니다.

**피스톤(Piston)** ($j = 1$, $Z_1 = 1$): 개구 전체에 걸친 일정한 위상 오프셋. 이것은 상질에 영향을 미치지 않으며(전체 파면의 위상을 균일하게 이동시킬 뿐), 보통 무시됩니다. 그러나 간섭 측정에서는 피스톤이 중요한데, 줄무늬(fringe)의 위치를 결정하기 때문입니다.

**팁과 틸트(Tip and Tilt)** ($j = 2, 3$): 이 선형 모드들은 상의 형태를 바꾸지 않고 초점면에서 상을 측방향으로 이동시킵니다. 팁($\rho\cos\theta$)은 수평으로, 틸트($\rho\sin\theta$)는 수직으로 이동합니다. 천문 관측에서 대기의 팁-틸트는 별 상의 "흔들림"을 유발하며, 장 노출 시 상 열화의 주된 원인입니다.

**초점이탈(Defocus)** ($j = 4$, $Z_4 = \sqrt{3}(2\rho^2 - 1)$): 최적 초점을 축 방향으로 이동시키는 이차 파면 곡률. 상이 대칭적으로 흐려지며 PSF가 균일한 원판으로 확대됩니다. 초점 위치를 조정하여 보정합니다.

**비점수차(Astigmatism)** ($j = 5, 6$): 파면이 두 수직 축을 따라 서로 다른 곡률을 가져 십자 형태의 PSF를 만들어 냅니다. 하나의 최적 초점이 존재하지 않으며 — 수평선과 수직선의 상이 서로 다른 거리에서 초점을 맺습니다. 인간의 눈에서 흔히 나타납니다.

### 5.2 3차(자이델) 수차

**코마(Coma)** ($j = 7, 8$, $n = 3$, $|m| = 1$): 일반적으로 축외(off-axis) 결상에 의해 발생하는 혜성 형태의 PSF 번짐. 파면 오차가 $\rho^3$으로 변화하여, 밝은 핵과 광축으로부터 멀어지는 방향으로 향하는 퍼진 꼬리를 가진 비대칭적인 상을 만들어 냅니다.

**세잎수차(Trefoil)** ($j = 9, 10$, $n = 3$, $|m| = 3$): 삼각형 PSF를 생성하는 3중 대칭 수차. 단순한 광학 시스템에서는 덜 흔하게 나타나지만 분할 거울 망원경(segmented-mirror telescopes)에서 중요합니다 (경면 정렬 오차로 인해).

**구면수차(Spherical Aberration)** ($j = 11$, $n = 4$, $m = 0$): 초점이탈 다음으로 가장 중요한 회전 대칭 수차. 개구 가장자리의 광선이 근축 광선(paraxial ray)과 다른 거리에서 초점을 맺어, PSF 핵 주위에 헤일로(halo)를 형성합니다. 이것은 COSTAR 보정 광학이 설치되기 전 허블 우주 망원경(Hubble Space Telescope)을 괴롭혔던 수차입니다.

### 5.3 고차 모드

| 차수 $n$ | 모드 | 물리적 의의 |
|:---------:|:-----:|----------------------|
| 5 | 2차 코마, 2차 세잎수차, 다섯잎수차 | 미세 구조 보정; 대형 망원경에서 중요 |
| 6 | 2차 구면수차, ... | 저차 AO 보정 후 대기 난류 잔류 오차 |
| 7–10 | 3차 수차 | 8–30 m 망원경의 익스트림 AO 시스템에서 관련 |
| >10 | 고주파 구조 | "신틸레이션(Scintillation)" 영역; 보정 어려움 |

> **유비**: 체르니케 모드를 원형 북면의 배음(harmonics)이라고 생각하세요. 최저 모드($j = 1$, 피스톤)는 정지된 북. 팁과 틸트($j = 2, 3$)는 시소처럼 기우는 북. 초점이탈($j = 4$)은 중심이 위로 불룩 솟으면서 가장자리가 내려가는(또는 그 반대) 북. 고차 모드들은 점점 더 복잡한 절선 패턴을 가집니다 — 클라드니 판(Chladni plate)의 모래로 시각화할 수 있는 진동 모드와 정확히 같지만, 정사각형이 아닌 원형 판에서의 경우입니다.

### 5.4 PSF에 미치는 영향

파면 오차 $W(\rho, \theta)$는 복소 동공 함수(complex pupil function)를 다음과 같이 변화시킵니다:

$$P(\rho, \theta) = A(\rho, \theta) \exp\left[\frac{2\pi i}{\lambda} W(\rho, \theta)\right]$$

여기서 $A$는 개구 진폭입니다. PSF는 $P$의 푸리에 변환의 제곱 모듈러스입니다:

$$\text{PSF}(\mathbf{u}) = \left|\mathcal{F}\{P(\rho, \theta)\}\right|^2$$

각 체르니케 모드는 특징적인 PSF 왜곡을 만들어 냅니다:

| 모드 | PSF 효과 |
|------|-----------|
| 초점이탈 | 대칭적 확대 (큰 초점이탈에서 도넛 형태) |
| 비점수차 | 한 축 방향으로의 늘어남; 큰 진폭에서 십자 패턴 |
| 코마 | 혜성 형태 꼬리; 밝은 핵 + 퍼진 부채꼴 |
| 세잎수차 | 삼중 대칭; 세 꼭짓점을 가진 별 모양 |
| 구면수차 | 밝은 중심 핵 + 퍼진 헤일로 |

---

## 6. 파면 분해와 피팅

### 6.1 내적을 통한 모드 분해

단위 원판 위에서 측정된 파면 $W(\rho, \theta)$가 주어질 때, 모드 $j$에 대한 체르니케 계수는 기저 위로의 투영(projection)으로 구합니다:

$$a_j = \frac{1}{\pi} \int_0^1 \int_0^{2\pi} W(\rho, \theta) Z_j(\rho, \theta) \, \rho \, d\rho \, d\theta$$

$N \times N$ 격자 위의 이산 데이터에 대해서는 가중 합이 됩니다:

$$a_j = \frac{1}{\sum_k w_k} \sum_{k \in \text{pupil}} w_k W_k Z_j(\rho_k, \theta_k)$$

여기서 $w_k$는 구적 가중치(quadrature weights)로, 원형 개구 내의 등간격 픽셀에 대해서는 흔히 균일하게 설정합니다.

### 6.2 최소 제곱 피팅

실제로, 파면 센서는 파면 자체보다 **기울기(slopes)** — 국소 파면 기울기(gradient) — 를 측정하는 경우가 많습니다. 예를 들어 샥-하트만 센서(Shack-Hartmann sensor)는 서브 개구(subaperture) 격자에서 $\partial W / \partial x$와 $\partial W / \partial y$를 측정합니다.

피팅 문제는 다음과 같이 됩니다: 기울기 측정값 $\mathbf{s} = [s_{x,1}, s_{y,1}, s_{x,2}, s_{y,2}, \ldots]^T$가 주어질 때, 다음을 최소화하는 체르니케 계수 $\mathbf{a} = [a_1, a_2, \ldots, a_J]^T$를 구합니다:

$$\|\mathbf{s} - \mathbf{D}\mathbf{a}\|^2$$

여기서 $\mathbf{D}$는 다음 원소를 갖는 **미분 행렬(derivative matrix)**입니다:

$$D_{2k-1, j} = \frac{\partial Z_j}{\partial x}\bigg|_{(\rho_k, \theta_k)}, \quad D_{2k, j} = \frac{\partial Z_j}{\partial y}\bigg|_{(\rho_k, \theta_k)}$$

최소 제곱 해는:

$$\hat{\mathbf{a}} = (\mathbf{D}^T \mathbf{D})^{-1} \mathbf{D}^T \mathbf{s} = \mathbf{D}^+ \mathbf{s}$$

여기서 $\mathbf{D}^+$는 무어-펜로즈 유사역행렬(Moore-Penrose pseudo-inverse)입니다. 실제로는 특잇값 분해(SVD, Singular Value Decomposition)를 사용하여 $\mathbf{D}^+$를 강건하게 계산하며, 잡음 임계값 이하의 특잇값은 버립니다.

### 6.3 피팅 오차와 모드 선택

$J$개의 모드를 피팅한 후의 잔류 파면 오차는:

$$W_{\text{res}}(\rho, \theta) = W(\rho, \theta) - \sum_{j=1}^{J} a_j Z_j(\rho, \theta)$$

RMS 잔류 오차는 $J$가 증가할수록 감소하지만, 잡음 증폭도 함께 증가합니다 (고차 모드는 측정 잡음에 더 민감합니다). 최적 모드 수는 수차 피팅과 잡음 전파 사이의 균형을 맞추는 것으로 — 편향-분산 트레이드오프(bias-variance tradeoff)의 전형입니다.

### 6.4 체르니케 미분

기울기 기반 피팅을 위해서는 직교 좌표에 대한 체르니케 다항식의 편미분이 필요합니다. 이는 해석적으로 또는 점화 관계(recurrence relations)를 통해 계산할 수 있습니다. 놀 인덱싱된 모드 $Z_j(\rho, \theta)$에 대해:

$$\frac{\partial Z_j}{\partial x} = \frac{\partial Z_j}{\partial \rho}\cos\theta - \frac{1}{\rho}\frac{\partial Z_j}{\partial \theta}\sin\theta$$

$$\frac{\partial Z_j}{\partial y} = \frac{\partial Z_j}{\partial \rho}\sin\theta + \frac{1}{\rho}\frac{\partial Z_j}{\partial \theta}\cos\theta$$

반경 다항식 $R_n^{|m|}$과 삼각 인수의 미분은 표준 미적분 규칙을 따릅니다.

---

## 7. 대기 난류와 콜모고로프 통계

### 7.1 콜모고로프 모델

대기 난류는 굴절률 $n(\mathbf{r})$의 무작위 변동을 만들어 내는 온도 요동에 의해 구동됩니다. 콜모고로프(Kolmogorov, 1941)는 **내부 스케일(inner scale)** $l_0$ (점성 소산이 우세한 수 밀리미터)과 **외부 스케일(outer scale)** $L_0$ (에너지 주입 스케일인 수십 미터) 사이의 규모에서, 굴절률 구조 함수(structure function)가 멱 법칙을 따름을 보였습니다:

$$D_n(r) = \langle [n(\mathbf{r}') - n(\mathbf{r}' + \mathbf{r})]^2 \rangle = C_n^2 r^{2/3}$$

여기서 $C_n^2$은 굴절률 구조 상수(단위: $\text{m}^{-2/3}$)로, 난류의 강도를 나타내며 고도에 따라 변합니다.

### 7.2 프리드 파라미터

**프리드 파라미터(Fried parameter)** $r_0$ (결합 길이(coherence length)라고도 함)는 대기 분해능이 회절 한계와 같아지는 개구 직경입니다:

$$r_0 = \left[0.423 k^2 \sec(\gamma) \int_0^{\infty} C_n^2(h) \, dh\right]^{-3/5}$$

여기서 $k = 2\pi/\lambda$이고 $\gamma$는 천정 거리각(zenith angle)입니다. 대표적인 값:

| 조건 | 500 nm에서의 $r_0$ |
|-----------|:---------------:|
| 최상의 관측지 (마우나케아) | 20–30 cm |
| 좋은 관측지 | 10–15 cm |
| 보통 관측지 | 5–10 cm |
| 불량 시상(Poor seeing) | < 5 cm |

대기 분해능(시상)은 $\theta_{\text{seeing}} \approx 0.98 \lambda / r_0$이고, $D \gg r_0$인 직경 $D$의 망원경은 분해능이 $\lambda/D$가 아닌 $\lambda/r_0$로 제한됩니다.

### 7.3 파면 위상 구조 함수

콜모고로프 난류를 통해 전파하는 평면파에 대한 위상 구조 함수는:

$$D_\phi(r) = 6.88 \left(\frac{r}{r_0}\right)^{5/3}$$

이는 위상 파워 스펙트럼 밀도(PSD, Power Spectral Density)로 표현할 수 있습니다:

$$\Phi_\phi(\kappa) = 0.023 \, r_0^{-5/3} \kappa^{-11/3}$$

여기서 $\kappa$는 공간 주파수입니다. 이 $-11/3$ 멱 법칙은 콜모고로프 난류의 특징입니다.

### 7.4 놀 공분산 행렬

대기 파면이 체르니케 모드로 분해될 때, 계수 $a_j$와 $a_{j'}$ 사이의 공분산은 (Noll, 1976):

$$\langle a_j a_{j'} \rangle = K_{jj'} \left(\frac{D}{r_0}\right)^{5/3}$$

여기서 $D$는 망원경 직경이고 $K_{jj'}$는 모드 인덱스에만 의존하는 행렬입니다. 주요 특성:

- **대각 우세(Diagonal dominance)**: 분산의 대부분이 저차 모드에 집중됩니다 (팁-틸트만으로도 전체 분산의 약 87%를 차지)
- **비대각 결합(Off-diagonal coupling)**: 같은 $|m|$과 같은 $n$의 홀짝성을 갖는 모드들이 상관됩니다
- **멱 법칙 감소(Power-law decay)**: 모드 $j$의 분산은 큰 $j$에 대해 대략 $j^{-11/6}$으로 감소

총 파면 분산은:

$$\sigma_\phi^2 = 1.0299 \left(\frac{D}{r_0}\right)^{5/3} \quad [\text{rad}^2]$$

처음 $J$개의 체르니케 모드를 제거한 후(예: 적응 광학을 통해)의 잔류 분산은:

$$\sigma_J^2 \approx 0.2944 \, J^{-\sqrt{3}/2} \left(\frac{D}{r_0}\right)^{5/3} \quad [\text{rad}^2] \quad \text{(for large } J\text{)}$$

### 7.5 위상 스크린 생성

대기 난류를 시뮬레이션하기 위해, 콜모고로프 파워 스펙트럼을 갖는 무작위 위상 스크린을 생성합니다. **FFT 방법**은 다음과 같이 작동합니다:

1. 복소 가우스 난수 격자 $\hat{c}(\kappa_x, \kappa_y)$를 생성합니다
2. 콜모고로프 파워 스펙트럼의 제곱근으로 필터링합니다: $\hat{\phi}(\boldsymbol{\kappa}) = \hat{c}(\boldsymbol{\kappa}) \sqrt{\Phi_\phi(\kappa)}$
3. 역 FFT를 적용하여 $\phi(x, y)$를 얻습니다

결과 위상 스크린은 올바른 구조 함수 통계를 가집니다. 더 정확한 저주파 성분을 위해 **부조화파 추가(subharmonic addition)** 방법을 사용할 수 있습니다 (Lane 등, 1992).

---

## 8. RMS 파면 오차와 스트렐 비율

### 8.1 체르니케 계수로부터의 RMS

정규직교성 덕분에, RMS 파면 오차는 체르니케 계수로 간단하게 표현됩니다:

$$\sigma_W = \sqrt{\frac{1}{\pi}\int\!\!\int_{\text{pupil}} W^2 \rho\,d\rho\,d\theta} = \sqrt{\sum_{j=2}^{J} a_j^2}$$

피스톤($j = 1$)은 상질에 영향을 주지 않으므로 제외합니다. 이것이 체르니케 기저의 큰 장점 중 하나입니다: RMS는 단순히 계수들의 제곱합의 제곱근입니다.

### 8.2 마레샬 근사

작은 파면 오차($\sigma_W \ll \lambda$)에 대해, **스트렐 비율(Strehl ratio)** (회절 한계 대비 최대 세기)은 다음으로 근사됩니다:

$$S \approx \exp\left[-(2\pi\sigma_W)^2\right] = \exp\left[-\left(\frac{2\pi\sigma_W}{\lambda}\right)^2\right]$$

여기서 $\sigma_W$는 파면 수(waves) 단위입니다. 이는 $S \gtrsim 0.1$ (즉, $\sigma_W \lesssim \lambda/4$)에서 유효합니다.

| $\sigma_W$ (파면 수) | $\sigma_W$ (550 nm에서 nm) | 스트렐 비율 |
|:---:|:---:|:---:|
| 0 | 0 | 1.000 |
| $\lambda/20$ | 27.5 | 0.905 |
| $\lambda/14$ | 39.3 | 0.815 |
| $\lambda/10$ | 55.0 | 0.674 |
| $\lambda/7$ | 78.6 | 0.444 |
| $\lambda/4$ | 137.5 | 0.081 |

> **레일리 기준(Rayleigh criterion)** ($\lambda/4$ 최대-최솟값 OPD, 초점이탈에 대해 $\sigma_W \approx \lambda/14$ RMS에 해당)은 스트렐 비율 약 0.8을 주며, 전통적으로 "회절 한계" 시스템의 임계값으로 간주됩니다.

### 8.3 부분 보정 분석

적응 광학 시스템이 처음 $J$개의 체르니케 모드를 보정하면, 잔류 스트렐 비율은 보정되지 않은 모드에만 의존합니다:

$$S_J \approx \exp\left[-(2\pi)^2 \sum_{j=J+1}^{\infty} \langle a_j^2 \rangle\right]$$

놀 잔류 분산 공식을 사용하면:

| 보정된 모드 수 | 제거된 수차 | 잔류 분산 (rad²) | 개선 배율 |
|:---:|---|:---:|:---:|
| 1 (피스톤) | — | $1.030(D/r_0)^{5/3}$ | 1.0× |
| 3 (팁-틸트) | 팁, 틸트 | $0.134(D/r_0)^{5/3}$ | 7.7× |
| 6 | + 초점이탈, 비점수차 | $0.058(D/r_0)^{5/3}$ | 17.8× |
| 10 | + 코마, 세잎수차 | $0.034(D/r_0)^{5/3}$ | 30.3× |
| 21 | 5차까지 | $0.016(D/r_0)^{5/3}$ | 64.4× |

이 표는 처음 몇 개의 모드만 보정해도 극적인 개선이 이루어짐을 보여줍니다 — 적응 광학의 핵심 동기입니다.

---

## 9. 파이썬 예제

### 9.1 체르니케 모드 생성

```python
import numpy as np

def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index j (starting at 1) to radial order n and
    azimuthal frequency m.

    The Noll convention orders modes by increasing n, with even-j
    assigned to cosine (m >= 0) and odd-j to sine (m < 0) terms.
    """
    # Find radial order n: j falls in the range (n(n+1)/2, (n+1)(n+2)/2]
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    # Position within this order
    k = j - n * (n + 1) // 2
    # Determine |m|
    if n % 2 == 0:
        m_abs = 2 * ((k + 1) // 2)
    else:
        m_abs = 2 * (k // 2) + 1
    # Sign convention: even j -> cosine (m >= 0), odd j -> sine (m < 0)
    if m_abs == 0:
        m = 0
    elif j % 2 == 0:
        m = m_abs
    else:
        m = -m_abs
    return n, m


def zernike_radial(n: int, m_abs: int, rho: np.ndarray) -> np.ndarray:
    """Compute radial Zernike polynomial R_n^|m|(rho).

    Uses the explicit factorial formula. The polynomial is zero outside
    the unit circle.

    Parameters
    ----------
    n : int  — Radial order (>= 0)
    m_abs : int  — Absolute azimuthal frequency (0 <= m_abs <= n, n - m_abs even)
    rho : ndarray  — Radial coordinate(s), 0 <= rho <= 1
    """
    R = np.zeros_like(rho, dtype=float)
    for s in range((n - m_abs) // 2 + 1):
        coeff = ((-1) ** s * np.math.factorial(n - s)
                 / (np.math.factorial(s)
                    * np.math.factorial((n + m_abs) // 2 - s)
                    * np.math.factorial((n - m_abs) // 2 - s)))
        R += coeff * rho ** (n - 2 * s)
    return R


def zernike(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate the Noll-indexed Zernike polynomial Z_j at (rho, theta).

    Returns the normalized polynomial value. Points outside the unit
    circle are set to NaN.
    """
    n, m = noll_to_nm(j)
    m_abs = abs(m)
    # Normalization factor
    if m == 0:
        norm = np.sqrt(n + 1)
    else:
        norm = np.sqrt(2 * (n + 1))
    R = zernike_radial(n, m_abs, rho)
    if m >= 0:
        Z = norm * R * np.cos(m_abs * theta)
    else:
        Z = norm * R * np.sin(m_abs * theta)
    # Mask outside unit circle
    Z[rho > 1.0] = np.nan
    return Z
```

### 9.2 이산 데이터로부터의 파면 피팅

```python
def zernike_fit(wavefront: np.ndarray, n_modes: int,
                rho: np.ndarray, theta: np.ndarray,
                mask: np.ndarray) -> np.ndarray:
    """Fit Zernike coefficients to a wavefront on a discrete grid.

    This builds the design matrix [Z_1, Z_2, ..., Z_J] at the valid
    (in-pupil) pixel locations and solves the normal equations via SVD.

    Parameters
    ----------
    wavefront : 2D array  — Measured wavefront (same shape as rho, theta)
    n_modes : int  — Number of Zernike modes to fit (j = 1..n_modes)
    rho, theta : 2D arrays  — Polar coordinates at each pixel
    mask : 2D bool array  — True inside the pupil

    Returns
    -------
    coeffs : 1D array of length n_modes  — Fitted Zernike coefficients
    """
    # Flatten valid pixels
    w = wavefront[mask].ravel()
    r = rho[mask].ravel()
    t = theta[mask].ravel()
    # Build design matrix: each column is Z_j evaluated at valid pixels
    A = np.column_stack([zernike(j, r, t) for j in range(1, n_modes + 1)])
    # Solve via least squares (SVD-based)
    coeffs, _, _, _ = np.linalg.lstsq(A, w, rcond=None)
    return coeffs
```

### 9.3 콜모고로프 위상 스크린 (FFT 방법)

```python
def kolmogorov_phase_screen(N: int, r0: float, L: float,
                            seed: int | None = None) -> np.ndarray:
    """Generate a Kolmogorov turbulence phase screen using the FFT method.

    The screen has the correct D_phi(r) = 6.88 (r/r0)^(5/3) structure
    function for separations between the grid spacing and the screen size.

    Parameters
    ----------
    N : int  — Grid size (NxN pixels)
    r0 : float  — Fried parameter in physical units (e.g., meters)
    L : float  — Physical side length of the screen (same units as r0)
    seed : int or None  — Random seed for reproducibility

    Returns
    -------
    phi : 2D array (N, N)  — Phase screen in radians
    """
    rng = np.random.default_rng(seed)
    # Spatial frequency grid
    df = 1.0 / L  # frequency spacing
    fx = np.fft.fftfreq(N, d=L / N)
    fy = np.fft.fftfreq(N, d=L / N)
    Fx, Fy = np.meshgrid(fx, fy)
    f_mag = np.sqrt(Fx**2 + Fy**2)
    f_mag[0, 0] = 1.0  # avoid division by zero; DC is set to zero later
    # Kolmogorov PSD: Phi(f) = 0.023 * r0^(-5/3) * (2*pi*f)^(-11/3)
    # In terms of spatial frequency f (not angular frequency kappa):
    psd = 0.023 * r0**(-5.0/3) * (2 * np.pi * f_mag)**(-11.0/3)
    psd[0, 0] = 0.0  # remove piston
    # Generate random complex field weighted by sqrt(PSD)
    # The factor accounts for the discrete FT normalization
    cn = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    cn *= np.sqrt(psd) * (2 * np.pi / L)
    # Inverse FFT to get phase screen
    phi = np.real(np.fft.ifft2(cn)) * N**2
    return phi
```

### 9.4 스트렐 비율 계산기

```python
def strehl_marechal(rms_waves: float) -> float:
    """Estimate Strehl ratio using the Maréchal approximation.

    S ≈ exp(-(2π σ)²), valid for σ ≲ λ/4 (Strehl ≳ 0.1).

    Parameters
    ----------
    rms_waves : float  — RMS wavefront error in units of waves (λ)

    Returns
    -------
    S : float  — Estimated Strehl ratio (0 to 1)
    """
    return np.exp(-(2 * np.pi * rms_waves)**2)


def rms_from_zernike(coeffs: np.ndarray, exclude_piston: bool = True) -> float:
    """Compute RMS wavefront error from Zernike coefficients.

    Thanks to orthonormality: σ = sqrt(Σ a_j²), excluding piston.
    """
    start = 1 if exclude_piston else 0  # coeffs[0] = a_1 (piston)
    return np.sqrt(np.sum(coeffs[start:]**2))
```

---

## 10. 요약

| 개념 | 핵심 공식 / 아이디어 |
|---------|--------------------|
| 체르니케 다항식 | $Z_n^m(\rho, \theta) = N_n^m R_n^{|m|}(\rho) \times \{\cos, \sin\}(|m|\theta)$ |
| 반경 다항식 | $R_n^{|m|}(\rho) = \sum_s \frac{(-1)^s(n-s)!}{s!(\ldots)!(\ldots)!}\rho^{n-2s}$ |
| 정규화 | $N_n^m = \sqrt{2(n+1)/(1+\delta_{m0})}$ |
| 정규직교성 | $\langle Z_j, Z_{j'} \rangle = \pi\delta_{jj'}$ |
| $n$차까지의 모드 수 | $(n+1)(n+2)/2$ |
| 놀 인덱싱 | $j = 1, 2, 3, \ldots$를 Noll (1976)에 따라 $(n, m)$으로 대응 |
| 파면 분해 | $a_j = \frac{1}{\pi}\int\!\int W Z_j \rho\,d\rho\,d\theta$ |
| RMS 파면 오차 | $\sigma_W = \sqrt{\sum_{j=2}^J a_j^2}$ |
| 콜모고로프 PSD | $\Phi_\phi(\kappa) = 0.023\,r_0^{-5/3}\kappa^{-11/3}$ |
| 프리드 파라미터 | $r_0 = [0.423 k^2 \sec\gamma \int C_n^2\,dh]^{-3/5}$ |
| 마레샬 근사 | $S \approx e^{-(2\pi\sigma_W)^2}$ |
| $J$개 모드 후 잔류 오차 | $\sigma_J^2 \approx 0.2944\,J^{-\sqrt{3}/2}(D/r_0)^{5/3}$ |

---

## 11. 연습 문제

### 연습 문제 1: 반경 다항식 검증

(a) 명시적인 체르니케 반경 다항식 공식을 사용하여 $R_4^0(\rho) = 6\rho^4 - 6\rho^2 + 1$임을 직접 계산으로 보이시오. (b) 수치적으로 직교성을 검증하시오: 조밀한 격자에서 $\int_0^1 R_2^0(\rho) R_4^0(\rho) \rho\,d\rho$를 계산하고 0임을 확인하시오. (c) $R_3^1$과 $R_5^1$에 대해 반복하시오. (d) $R_6^0(\rho)$를 계산하고 도식화하시오. 이 모드의 물리적 해석은 무엇인가?

### 연습 문제 2: 파면 분석

파면 센서가 다음의 놀 체르니케 계수를 반환합니다(파면 수 단위): $a_2 = 0.05$, $a_3 = -0.08$, $a_4 = 0.30$, $a_5 = -0.15$, $a_6 = 0.10$, $a_7 = 0.12$, $a_8 = -0.06$, $a_{11} = 0.20$. (a) 총 RMS 파면 오차를 계산하시오. (b) 어느 단일 모드가 RMS에 가장 많이 기여하는가? (c) 마레샬 근사를 사용하여 스트렐 비율을 추정하시오. (d) 가장 나쁜 세 모드를 완벽하게 보정한다면 새로운 스트렐 비율은 얼마인가? (e) 파면 오차 지도와 대응하는 PSF를 도식화하시오 (힌트: 동공 함수의 FFT 사용).

### 연습 문제 3: 대기 위상 스크린

$\lambda = 500$ nm에서 $r_0 = 15$ cm인 4미터 망원경에 대해 512×512 격자에서 콜모고로프 위상 스크린을 생성하시오. (a) 생성된 스크린으로부터 위상 구조 함수 $D_\phi(r)$를 계산하고 도식화하여 이론값 $6.88(r/r_0)^{5/3}$과 비교하시오. (b) 처음 36개의 체르니케 모드를 피팅하고 계수 크기를 표시하시오. (c) 팁-틸트 모드가 총 분산을 지배함을 검증하시오. (d) 10개와 36개의 모드를 제거한 후 잔류 분산을 놀 공식과 비교하시오.

### 연습 문제 4: 환형 개구 확장

많은 망원경에는 중앙 차폐(secondary mirror)가 있습니다. (a) 차폐 비율 $\epsilon = 0.3$인 환형 개구를 나타내는 격자에서 처음 15개의 표준 체르니케 모드를 계산하고, 이것이 **직교하지 않음을**을 검증하시오 (그람 행렬(Gram matrix) $G_{jj'} = \langle Z_j, Z_{j'} \rangle_\text{annulus}$를 계산하시오). (b) 그람-슈미트 직교화(Gram-Schmidt orthogonalization)를 사용하여 처음 10개의 환형 체르니케 다항식을 구성하시오. (c) 합성 파면(코마 + 구면수차)을 표준 및 환형 기저 양쪽에서 분해하시오. 계수가 어떻게 다른가? (d) 환형 체르니케 다항식이 필요한 경우와 전체 원에 대한 표준 체르니케 다항식이 충분한 근사인 경우를 논의하시오.

---

## 12. 참고문헌

1. Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence." *Journal of the Optical Society of America*, 66(3), 207–211. — 놀 인덱싱 및 대기 체르니케 통계의 표준 참고문헌.
2. Born, M., & Wolf, E. (2019). *Principles of Optics* (7th expanded ed.). Cambridge University Press. — 수차에 관한 9장; 체르니케 다항식에 관한 부록 VII.
3. Mahajan, V. N. (2013). *Optical Imaging and Aberrations, Part III: Wavefront Analysis*. SPIE Press. — 체르니케 기반 파면 분석의 포괄적 처리.
4. Hardy, J. W. (1998). *Adaptive Optics for Astronomical Telescopes*. Oxford University Press. — 대기 난류와 체르니케 분해에 관한 3–4장.
5. Roddier, F. (Ed.) (1999). *Adaptive Optics in Astronomy*. Cambridge University Press. — 대기 난류 이론에 관한 2장.
6. Thibos, L. N., Applegate, R. A., Schwiegerling, J. T., & Webb, R. (2002). "Standards for reporting the optical aberrations of eyes." *Journal of Refractive Surgery*, 18(5), S652–S660. — 안과학을 위한 ANSI/OSA 체르니케 표준.
7. Lane, R. G., Glindemann, A., & Dainty, J. C. (1992). "Simulation of a Kolmogorov phase screen." *Waves in Random Media*, 2(3), 209–224. — 부조화파를 포함한 FFT 기반 위상 스크린 생성.

---

[← 이전: 14. 전산 광학](14_Computational_Optics.md) | [다음: 16. 적응광학 →](16_Adaptive_Optics.md)
