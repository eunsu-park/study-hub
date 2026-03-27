# 16. 적응광학(Adaptive Optics)

[← 이전: 15. 체르니케 다항식](15_Zernike_Polynomials.md) | [다음: 17. 분광학 →](17_Spectroscopy.md)

---

1990년, 허블 우주 망원경은 잘못된 모양으로 연마된 주경을 달고 발사되었다. 2.2 마이크로미터의 구면 수차(spherical aberration)는 육안으로는 거의 감지할 수 없었지만 천문 관측 영상에는 치명적이었다. 15억 달러짜리 장비가 거의 장님이 된 것이다. 3년 후, 우주비행사들이 COSTAR라는 보정 광학 장치를 설치하여 허블의 시력을 회절 한계(diffraction limit) 수준으로 회복시켰다. 이 극적인 수리가 가능했던 것은 수차가 정적이고 알려진 형태였기 때문이다. 그런데 지상 망원경의 경우에는 어떻게 해야 할까? 대기가 초당 수백 번씩 파면(wavefront)을 무작위로, 끊임없이 변하는 왜곡으로 뒤흔드는 환경에서는 말이다.

**적응광학(Adaptive Optics, AO)**은 이 도전에 응답하는 기술이다. 1953년 호레이스 배브콕(Horace Babcock)이 처음 제안한 AO 시스템은 파면 센서(wavefront sensor)를 사용해 대기 파면 왜곡을 실시간으로 측정하고, 빠른 재구성기(reconstructor)로 보정값을 계산한 다음, 변형 거울(deformable mirror)로 적용한다 — 이 모든 과정이 수 밀리초 안에 이루어진다. 그 결과 지름 8~10미터의 지상 망원경이 회절 한계에 근접한 분해능을 달성하여 특정 관측에서는 허블에 필적하거나 능가할 수 있다.

이 단원은 대기 난류(atmospheric turbulence) 특성화부터 파면 센싱(wavefront sensing), 재구성(reconstruction), 보정(correction)까지 완전한 AO 시스템을 다루며, 15단원의 체르니케 다항식(Zernike polynomial) 기초를 토대로 한다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. 대기 난류가 지상 망원경의 분해능을 제한하는 이유를 설명하고, 프리드 매개변수(Fried parameter)를 사용해 시잉 한계(seeing limit)를 정량화할 수 있다
2. 주요 대기 매개변수(프리드 매개변수 $r_0$, 그린우드 주파수(Greenwood frequency) $f_G$, 등플라나틱 각도(isoplanatic angle) $\theta_0$)와 이들의 파장 및 천정각 의존성을 설명할 수 있다
3. 고전적 AO 시스템의 블록 다이어그램(안내성(guide star), 파면 센서, 재구성기, 변형 거울, 과학 카메라)을 그리고 설명할 수 있다
4. 샤크-하트만(Shack-Hartmann), 곡률(curvature), 피라미드(pyramid) 파면 센서의 작동 원리를 설명할 수 있다
5. 기울기 측정으로부터 최소제곱 파면 재구성을 유도하고, 구역(zonal) 방식과 모드(modal) 방식의 차이를 설명할 수 있다
6. 변형 거울 기술, 영향 함수(influence function), 구동기-파면 매핑을 설명할 수 있다
7. 폐루프(closed-loop) AO 제어(적분기 이득(integrator gain), 시간 오차(temporal error), AO 오차 예산(error budget))를 분석할 수 있다
8. 레이저 안내성(laser guide star), 다중 켤레 AO(multi-conjugate AO), 외계행성 검출을 위한 극한 AO(extreme AO) 등 고급 AO 개념을 설명할 수 있다

---

## 목차

1. [적응광학이 왜 필요한가?](#1-적응광학이-왜-필요한가)
2. [대기 난류 매개변수](#2-대기-난류-매개변수)
3. [AO 시스템 구조](#3-ao-시스템-구조)
4. [파면 센서](#4-파면-센서)
5. [파면 재구성](#5-파면-재구성)
6. [변형 거울](#6-변형-거울)
7. [폐루프 제어](#7-폐루프-제어)
8. [성능 지표와 오차 예산](#8-성능-지표와-오차-예산)
9. [고급 AO 개념](#9-고급-ao-개념)
10. [파이썬 예제](#10-파이썬-예제)
11. [요약](#11-요약)
12. [연습 문제](#12-연습-문제)
13. [참고문헌](#13-참고문헌)

---

## 1. 적응광학이 왜 필요한가?

### 1.1 시잉 한계

이상적인 지름 $D$의 망원경은 다음만큼 작은 각도를 분해해야 한다:

$$\theta_{\text{diff}} = 1.22 \frac{\lambda}{D}$$

$\lambda = 500$ nm에서 8미터 망원경의 경우, $\theta_{\text{diff}} \approx 0.016$ 각초(arcseconds)가 된다 — 30 km 거리에서 신문 헤드라인을 읽을 수 있을 만큼 선명한 수준이다. 그러나 대기 난류는 일반적으로 분해능을 다음으로 제한한다:

$$\theta_{\text{seeing}} \approx 0.98 \frac{\lambda}{r_0} \approx 0.5\text{--}2\,\text{arcsec}$$

500 nm에서 $r_0 = 10$ cm일 때 시잉(seeing)은 약 1 각초다 — 회절 한계보다 **50배 나쁜** 수준이다. 대기 왜곡이 보정되지 않는 한 대형 주경에 대한 모든 투자가 낭비된다.

### 1.2 짧은 노출 대 긴 노출

단일 짧은 노출(~10 ms)에서 대기 파면은 대략 "정지"된 상태이며, 영상은 스페클 패턴(speckle pattern) — 각각의 크기가 회절 한계인 밝은 점들의 무작위 집합 — 으로 구성된다. 긴 노출(수 초에서 수 분)에 걸쳐 이 스페클들은 평균화되어 익숙한 시잉 제한 얼룩(blob)을 만들어낸다.

> **비유**: 수영장 바닥의 동전을 바라본다고 상상해보라. 수면의 잔물결이 영상을 왜곡하여 동전이 춤추고 흔들리는 것처럼 보인다. 만약 수면을 순간 정지시킨 다음 유연한 틀로 평평하게 밀어낼 수 있다면, 동전은 선명하고 고요하게 보일 것이다. 적응광학은 바로 이것을 한다: 초당 수백 번, "잔물결"(대기 난류)을 측정하고 변형 거울로 평탄화한다.

### 1.3 역사적 발전

| 연도 | 이정표 |
|------|--------|
| 1953 | 배브콕(Babcock)이 AO 개념 제안 |
| 1970년대 | 미국 군이 위성 영상용 기밀 AO 개발 |
| 1989 | ESO COME-ON: 최초의 천문 AO 시스템 |
| 1994 | 레이저 안내성 시연 |
| 2002 | 레이저 안내성을 갖춘 켁(Keck) AO 운영 |
| 2010년대 | 극한 AO 시스템(GPI, SPHERE)이 외계행성 직접 촬영 |
| 2020년대 | ELT(39 m), TMT(30 m), GMT(25 m) 모두 MCAO를 중심으로 설계 |

---

## 2. 대기 난류 매개변수

### 2.1 프리드 매개변수 $r_0$

프리드 매개변수(15단원 §7.2에서 소개)는 대기 난류를 특성화하는 가장 중요한 단일 수치다. 유효 간섭성 구경 지름(effective coherent aperture diameter)을 나타낸다:

$$r_0 = \left[0.423 k^2 \sec\gamma \int_0^\infty C_n^2(h)\,dh\right]^{-3/5}$$

주요 스케일링 법칙:

$$r_0 \propto \lambda^{6/5} \qquad r_0 \propto (\cos\gamma)^{3/5}$$

좋은 관측지에서 $r_0$는 500 nm에서 약 15 cm다. 적외선 파장($K$-band, 2.2 $\mu$m)에서는 $r_0$가 ~75 cm까지 증가하여 AO 보정이 훨씬 쉬워진다.

### 2.2 그린우드 주파수 $f_G$

대기 난류는 정적이지 않다 — 바람이 난류층을 망원경 구경에 걸쳐 이동시킨다. **그린우드 주파수(Greenwood frequency)**는 AO 보정에 필요한 시간 대역폭(temporal bandwidth)을 특성화한다:

$$f_G = 0.427 \frac{v_{\text{eff}}}{r_0}$$

여기서 $v_{\text{eff}}$는 유효 풍속(일반적으로 가장 강한 난류층의 바람)이다. 가시광선 파장에서 전형적인 값은 $f_G \approx 20$~$50$ Hz다. AO 시스템은 대기를 따라가기 위해 적어도 $f_G$ 이상의 폐루프 대역폭으로 작동해야 한다.

더 정확하게는, AO 시스템의 $-3\,\text{dB}$ 대역폭이 다음을 만족해야 한다:

$$f_{3\text{dB}} \gtrsim f_G$$

대역폭이 $f_G$보다 낮아지면, 시간 오차가 오차 예산을 지배하게 된다.

### 2.3 등플라나틱 각도 $\theta_0$

AO 보정은 안내성 주변의 제한된 시야 내에서만 유효하다. **등플라나틱 각도(isoplanatic angle)**는 파면이 상관관계를 갖는 각도 반경이다:

$$\theta_0 = 0.314 \frac{r_0}{\bar{h}}$$

여기서 $\bar{h}$는 ($C_n^2$로 가중된) 난류의 유효 고도다. 전형적인 값: 가시광선에서 $\theta_0 \approx 2$~$5$ 각초, $K$-band에서 $10$~$20$ 각초. 효과적인 보정을 위해 과학 목표물은 안내성으로부터 $\theta_0$ 이내에 있어야 한다.

### 2.4 $C_n^2$ 프로필

굴절률 구조 상수(refractive index structure constant) $C_n^2(h)$는 고도에 따라 변한다. 전형적인 프로필은 다음을 갖는다:

- **지표층(surface layer)** (0~1 km): 지면 가열로 인한 강한 난류
- **자유 대기(free atmosphere)** (1~10 km): 약하고 매끄러운 난류
- **대류권계면(tropopause)** (~10~12 km): 강한 전단층(shear layer) (제트 기류)

| 층 | 전형적 $C_n^2$ (m$^{-2/3}$) | 시잉 기여도 |
|-------|:---------------------------:|----------------------|
| 지표 (0~500 m) | $10^{-14}$ ~ $10^{-13}$ | 50~80% |
| 자유 대기 | $10^{-17}$ ~ $10^{-16}$ | 10~30% |
| 대류권계면 | $10^{-16}$ ~ $10^{-15}$ | 10~30% |

난류의 다층 구조는 다중 켤레 AO(MCAO, multi-conjugate AO)와 같은 고급 AO 개념에 매우 중요하며, MCAO는 지배적인 난류층에 켤레인 변형 거울들을 배치한다.

---

## 3. AO 시스템 구조

### 3.1 블록 다이어그램

고전적인 단일 켤레(single-conjugate) AO 시스템은 피드백 루프를 이루는 네 가지 주요 구성 요소로 이루어진다:

```
                    ┌──────────────┐
                    │  Guide Star  │
                    └──────┬───────┘
                           │ turbulent wavefront
                    ┌──────▼───────┐
                    │  Deformable  │◄─── DM commands
                    │    Mirror    │
                    └──────┬───────┘
                           │ corrected wavefront
              ┌────────────┼────────────┐
              │                         │
       ┌──────▼───────┐         ┌──────▼───────┐
       │   Wavefront   │         │   Science    │
       │    Sensor     │         │   Camera     │
       └──────┬────────┘         └──────────────┘
              │ slope measurements
       ┌──────▼────────┐
       │ Reconstructor │
       │  & Controller │
       └──────┬────────┘
              │ DM commands
              └─────────────────────────────────►
```

안내성(자연 또는 레이저)으로부터 나온 빛이 대기를 통과하여 변형 거울에서 반사된 후, 파면 센서와 과학 카메라로 분리된다. 파면 센서는 잔류 수차를 측정하고, 재구성기는 보정값을 계산하며, 변형 거울이 이를 적용한다 — 수백 Hz로 루프를 닫는다.

### 3.2 광학 경로

망원경에서 나온 빔은 먼저 **변형 거울(DM)**에서 반사되어 보정이 적용된다. 그런 다음 **이색성 빔 분리기(dichroic beamsplitter)**가 빛을 분리한다:
- 안내성 파장은 **파면 센서(WFS)**로 전달된다
- 과학 파장은 **과학 카메라**로 통과된다

이 배치는 WFS가 DM 보정 *이후* 잔류 파면 오차를 감지하도록 하여 폐루프 피드백 시스템을 가능하게 한다.

### 3.3 타이밍 요구사항

측정에서 보정까지의 폐루프 지연이 대기 간섭 시간(atmospheric coherence time)보다 짧아야 한다:

$$\tau_0 = 0.314 \frac{r_0}{v_{\text{eff}}} = \frac{0.134}{f_G}$$

$f_G = 30$ Hz에서 $\tau_0 \approx 4.5$ ms다. CCD 판독 + 계산 + DM 안정화를 포함한 총 루프 지연이 이보다 훨씬 짧아야 한다. 현대 AO 시스템은 500~2000 Hz의 루프 속도를 달성한다.

---

## 4. 파면 센서

### 4.1 샤크-하트만 센서

천문학에서 가장 널리 사용되는 WFS. **마이크로렌즈 어레이(microlens array, lenslet array)**가 동공(pupil)을 부분 구경(subaperture)으로 나눈다(일반적으로 크기 $d \sim r_0$). 각 렌즐릿(lenslet)은 안내성 영상을 형성하며, 이 점의 기준 위치에서의 변위는 해당 부분 구경에 걸친 평균 파면 기울기에 비례한다:

$$s_x = \frac{\partial W}{\partial x}\bigg|_{\text{avg}}, \qquad s_y = \frac{\partial W}{\partial y}\bigg|_{\text{avg}}$$

**중심 찾기 알고리즘(centroiding algorithm)**이 점의 변위를 측정한다:

**무게 중심(Center of gravity, CoG)**:
$$s_x = \frac{\sum_i x_i I_i}{\sum_i I_i}, \qquad s_y = \frac{\sum_i y_i I_i}{\sum_i I_i}$$

여기서 $I_i$는 픽셀 $i$의 강도다. 간단하고 빠르지만 배경 잡음에 민감하다.

**임계값 적용 CoG(Thresholded CoG)**: CoG 계산 전에 임계값 이하의 픽셀을 0으로 설정한다. 잡음을 줄이지만 편향(bias)이 생긴다.

**상관 관계(Correlation)**: 각 부분 구경 영상을 기준 PSF(Point Spread Function)와 교차 상관한다. 잡음에 더 강건하지만 계산 비용이 많이 든다.

| 매개변수 | 전형적 값 |
|-----------|:-------------:|
| 부분 구경 크기 | $r_0$ ~ $2r_0$ |
| 부분 구경 수 | 10×10 ~ 80×80 |
| 부분 구경당 픽셀 | 4×4 ~ 16×16 |
| 프레임 속도 | 500~3000 Hz |

### 4.2 곡률 센서

로디에(Roddier, 1988)가 제안한 곡률 센서(curvature sensor)는 두 초점 이탈 평면 간의 강도 차이를 측정한다:

$$\frac{I_1(\mathbf{r}) - I_2(\mathbf{r})}{I_1(\mathbf{r}) + I_2(\mathbf{r})} \propto \nabla^2 W(\mathbf{r}) + \frac{\partial W}{\partial n}\bigg|_{\text{edge}}$$

여기서 $I_1$과 $I_2$는 초점 내(intra-focal)와 초점 외(extra-focal) 강도이며, 에지 항(edge term)은 동공 경계에서의 파면 기울기를 설명한다.

**장점**: 파면의 라플라시안(Laplacian, 2차 도함수)을 직접 측정하며, 이는 바이모프 변형 거울(bimorph deformable mirror)과 자연스럽게 매핑된다. 샤크-하트만보다 간단한 광학 구성.

**단점**: 초점면 사이를 진동하는 진동막 거울(vibrating membrane mirror)이 필요하며, 공간 분해능이 낮다.

### 4.3 피라미드 센서

라가조니(Ragazzoni, 1996)가 개발한 피라미드 센서(pyramid sensor)는 초점면에 놓인 유리 피라미드를 사용하여 빔을 동공 영상의 네 복사본으로 분리한다:

$$s_x \propto \frac{(I_1 + I_2) - (I_3 + I_4)}{I_1 + I_2 + I_3 + I_4}$$

피라미드는 변조(oscillation, 초점 주위를 회전)되어 감도 범위를 조정할 수 있다 — 다른 센서에 없는 고유한 장점이다.

**장점**: 밝은 안내성에서 샤크-하트만보다 높은 감도; 변조를 통한 조정 가능한 동적 범위.

**단점**: 큰 수차에서 비선형 응답; 넓은 범위를 위해 변조 필요.

### 4.4 비교

| 특성 | 샤크-하트만 | 곡률 | 피라미드 |
|----------|:--------------:|:---------:|:-------:|
| 측정량 | 기울기($\nabla W$) | 라플라시안($\nabla^2 W$) | 기울기($\nabla W$) |
| 선형성 | 부분 구경당 $\pm \lambda/2$까지 선형 | 선형 | 변조로 선형 |
| 감도 (밝은 별) | 보통 | 보통 | 높음 |
| 감도 (어두운 별) | 좋음 | 좋음 | 보통 |
| 공간 샘플링 | 렌즐릿에 고정 | 초점 이탈 거리에 고정 | 조정 가능(변조) |
| 복잡도 | 간단, 강건 | 보통 | 복잡한 광학 |
| 사용 사례 | 대부분의 AO 시스템 | 초기 AO 시스템, ESO | ELT(MAORY), TMT |

---

## 5. 파면 재구성

### 5.1 재구성 문제

WFS로부터 기울기 측정값 $\mathbf{s} = [s_{x,1}, s_{y,1}, s_{x,2}, s_{y,2}, \ldots]^T$가 주어지면, 파면 $W(\mathbf{r})$를 추정하거나 이를 평탄화할 DM 명령 $\mathbf{c}$를 추정해야 한다.

### 5.2 구역 재구성

**구역(zonal)** 방식에서는 파면이 각 부분 구경 중심의 위상값 격자로 표현된다. 인접한 격자점 사이의 기울기는:

$$s_x \approx \frac{W_{i+1,j} - W_{i,j}}{\Delta x}$$

이는 희소 선형 시스템(sparse linear system) $\mathbf{s} = \mathbf{G}\mathbf{w}$를 만들며, 여기서 $\mathbf{G}$는 **기하 행렬(geometry matrix)**이다(희소, 성분 $\pm 1/\Delta x$). 최소제곱 해는:

$$\hat{\mathbf{w}} = (\mathbf{G}^T\mathbf{G})^{-1}\mathbf{G}^T\mathbf{s}$$

행렬 $\mathbf{G}^T\mathbf{G}$는 이산 라플라시안(discrete Laplacian)이므로 푸리에 방법이나 반복 솔버(가우스-자이델, 켤레 기울기)를 사용하여 효율적으로 재구성할 수 있다.

### 5.3 모드 재구성

**모드(modal)** 방식에서는 파면이 체르니케(또는 다른) 모드로 전개된다:

$$W(\rho, \theta) = \sum_{j=1}^{J} a_j Z_j(\rho, \theta)$$

각 부분 구경에서의 기울기는 계수의 선형 함수다:

$$\mathbf{s} = \mathbf{D}\mathbf{a}$$

여기서 $\mathbf{D}$는 **상호작용 행렬(interaction matrix)**이다(각 부분 구경에서 각 체르니케 모드의 도함수). 최소제곱 재구성기는:

$$\hat{\mathbf{a}} = \mathbf{D}^+ \mathbf{s} = (\mathbf{D}^T\mathbf{D})^{-1}\mathbf{D}^T\mathbf{s}$$

**장점**: 모드를 필터링할 수 있다(예: 피스톤(piston) 제외); 콜모고로프(Kolmogorov) 통계와 자연스러운 연결; 고차 모드의 잡음 억제 가능. **단점**: 체르니케 도함수 계산 필요; 선택된 모드 수에 의해 제한된다.

### 5.4 상호작용 행렬 (보정)

실제로는 $\mathbf{D}$를 경험적으로 측정한다:

1. DM 구동기(또는 각 모드)를 하나씩 작동시킨다
2. 각 명령 $\mathbf{c}_j$에 대한 WFS 응답 $\mathbf{s}_j$를 기록한다
3. 상호작용 행렬은 $\mathbf{M} = [\mathbf{s}_1 | \mathbf{s}_2 | \ldots]$
4. 명령 행렬(재구성기)은 $\mathbf{R} = \mathbf{M}^+$ (SVD를 통한 유사 역행렬(pseudo-inverse))

이 보정 과정은 WFS 기하학, DM 영향 함수, 광학 정렬 오차 등 모든 실제 효과를 반영한다.

### 5.5 잡음 전파

재구성된 파면은 WFS 측정 잡음을 증폭한다. **잡음 전파 계수(noise propagation coefficient)** $\eta$는 다음과 같이 정의된다:

$$\sigma_{\text{recon}}^2 = \eta \, \sigma_{\text{meas}}^2$$

구역 재구성기의 경우 $\eta$는 기하학에 따라 달라지며 일반적으로 0.2~1.0이다. 모드 재구성기의 경우 잡음 전파는 모드 번호가 높아질수록 증가하므로, 대기 신호가 잡음 바닥 아래로 떨어지는 수준에서 모드를 잘라내도록 동기를 부여한다.

---

## 6. 변형 거울

### 6.1 변형 거울의 종류

**분절 DM(Segmented DM)**: 피스톤, 팁, 틸트(tip-tilt) 구동기를 각각 갖는 평면 거울 분절 어레이. 분절 거울 망원경(켁(Keck), ELT)에 사용된다. 간극 회절(gap diffraction)이 단점이다.

**연속 면판 DM(Continuous facesheet DM)**: 아래쪽 구동기 어레이로 지지되는 얇은 반사막. 천문 AO에서 가장 일반적인 유형. 매끄럽고 연속적인 보정을 제공한다.

**바이모프 DM(Bimorph DM)**: 두 압전(piezoelectric) 층을 접합; 전압을 가하면 거울이 구부러진다. 곡률 센서와 자연스러운 매칭(둘 다 파면의 라플라시안을 다룬다).

**MEMS DM**: 칩 규모 거울에 수천 개의 구동기를 갖는 미세전자기계 시스템(Micro-electro-mechanical systems). 극한 AO 및 실험실 시스템에 사용된다.

### 6.2 영향 함수

**영향 함수(influence function)** $\phi_k(\mathbf{r})$는 구동기 $k$에 단위 명령이 가해졌을 때의 거울 표면 모양을 나타낸다:

$$W_{\text{DM}}(\mathbf{r}) = \sum_{k=1}^{K} c_k \phi_k(\mathbf{r})$$

연속 면판 DM의 경우, 영향 함수는 대략 가우시안(Gaussian)이다:

$$\phi_k(\mathbf{r}) \approx \exp\left(-\ln 2 \frac{|\mathbf{r} - \mathbf{r}_k|^2}{w^2}\right)$$

여기서 $w$는 영향 함수 폭(구동기 간 결합(coupling)과 관련)이다. **결합(coupling)**은 인접 구동기가 느끼는 행정(stroke)의 비율로, 압전 스택 DM에서 일반적으로 10~15%다.

### 6.3 주요 DM 매개변수

| 매개변수 | 설명 | 전형적 값 |
|-----------|-------------|:--------------:|
| 구동기 수 | 공간 자유도 | 100~10,000 |
| 구동기 간격 | 구동기 간 거리 | 0.3~10 mm |
| 행정(Stroke) | 최대 표면 변위 | 2~10 $\mu$m |
| 결합 | 인접 응답 비율 | 10~15% |
| 대역폭 | 기계적 공진 한계 | 1~10 kHz |
| 평탄도 | 모든 구동기가 0일 때 RMS 잔류값 | 5~30 nm |

### 6.4 피팅 오차

DM은 구동기 간격 $d$보다 작은 파면 특징을 재현할 수 없다. **피팅 오차(fitting error)**는 많은 AO 시스템에서 지배적인 오차 항이다:

$$\sigma_{\text{fit}}^2 = \alpha \left(\frac{d}{r_0}\right)^{5/3}$$

여기서 $\alpha \approx 0.23$은 연속 면판 DM이고 $\alpha \approx 0.28$은 분절 DM이다. 파장 $\lambda$에서 스트렐 비(Strehl ratio) $S$를 달성하려면 구동기 간격이 다음을 만족해야 한다:

$$d \lesssim r_0 \left(\frac{-\ln S}{(2\pi \alpha)^{2}}\right)^{3/10}$$

---

## 7. 폐루프 제어

### 7.1 AO 시간 루프

각 시간 단계 $t_k$에서:

1. **측정**: WFS가 잔류 파면에서 기울기 $\mathbf{s}_k$를 읽는다
2. **재구성**: 명령 행렬을 사용하여 잔류값 $\hat{\mathbf{w}}_k = \mathbf{R} \mathbf{s}_k$를 계산한다
3. **제어**: DM 명령 $\mathbf{c}_{k+1} = \mathbf{c}_k + g \hat{\mathbf{w}}_k$를 업데이트한다
4. **적용**: DM에 명령을 전송한다

이는 이득 $g \in (0, 1]$을 갖는 **적분기(integrator)** 제어기다.

### 7.2 적분기 제어

파면 오차는 루프 속도에 비해 천천히 변하기 때문에, 적분기는 시간에 걸쳐 보정을 축적하며 이는 적절한 방식이다. 적분기의 전달 함수(transfer function)는:

$$H_{\text{rej}}(f) = \frac{1}{1 + g \frac{f_s}{2\pi i f} e^{-i2\pi f \tau}}$$

여기서 $f_s$는 루프 주파수이고 $\tau$는 총 지연(일반적으로 1~2 프레임)이다. 거부 대역폭(rejection bandwidth, $|H_{\text{rej}}| = 0.5$인 지점)은:

$$f_{-3\text{dB}} \approx \frac{g f_s}{2\pi}$$

**이득 선택**: 이득이 너무 낮으면 대역폭이 좋지 않아(시간 오차) 나쁜 결과가 나온다. 이득이 너무 높으면 잡음이 증폭(잡음 오차)되고 루프가 불안정해질 수 있다. 최적 이득은 이 둘을 균형잡는다:

$$g_{\text{opt}} = 1 - \exp(-2\pi f_G \tau)$$

### 7.3 시간 오차

AO 대역폭이 대기를 추적하기에 불충분할 때, 잔류 시간 오차는:

$$\sigma_{\text{temp}}^2 = \left(\frac{f_G}{f_{-3\text{dB}}}\right)^{5/3}$$

이것은 "대역폭 부족(bandwidth-starved)" 시스템(예: 높은 그린우드 주파수를 가진 가시광선 AO)에서 지배적인 오차 항이다.

### 7.4 고급 제어기

단순 적분기는 더 정교한 제어 법칙으로 개선할 수 있다:

- **비례-적분(PI, Proportional-Integral)**: 더 빠른 응답을 위해 비례 항을 추가
- **선형 이차 가우시안(LQG, Linear Quadratic Gaussian)**: 시간 및 공간 상관 관계를 기반으로 파면을 예측하는 칼만 필터(Kalman filter)를 사용하는 최적 제어기
- **예측 제어(Predictive control)**: 풍속과 방향을 사용하여 파면 진화를 예측

---

## 8. 성능 지표와 오차 예산

### 8.1 스트렐 비

스트렐 비(Strehl ratio)는 AO 성능의 주요 지표다:

$$S = \frac{I_{\text{peak}}}{I_{\text{diffraction limit}}} \approx \exp\left[-(2\pi\sigma_{\text{total}})^2\right]$$

여기서 $\sigma_{\text{total}}$은 파수(waves) 단위의 총 잔류 파면 오차다.

### 8.2 오차 예산

총 파면 분산(wavefront variance)은 독립적인 오차 항들의 합이다:

$$\sigma_{\text{total}}^2 = \sigma_{\text{fit}}^2 + \sigma_{\text{temp}}^2 + \sigma_{\text{noise}}^2 + \sigma_{\text{alias}}^2 + \sigma_{\text{aniso}}^2 + \sigma_{\text{other}}^2$$

| 오차 항 | 원인 | 공식 |
|------------|--------|---------|
| 피팅(Fitting) | DM 구동기 밀도 | $\alpha (d/r_0)^{5/3}$ |
| 시간(Temporal) | 대역폭 대 그린우드 주파수 | $(f_G / f_{-3\text{dB}})^{5/3}$ |
| 잡음(Noise) | WFS 광자 + 판독 잡음 | $\propto 1/\text{SNR}^2$ |
| 에일리어싱(Aliasing) | 고차 모드가 저차 모드로 접힘 | $\approx 0.08 (d/r_0)^{5/3}$ |
| 등플라나틱 이탈(Anisoplanatism) | 안내성으로부터의 각도 분리 | $(\theta/\theta_0)^{5/3}$ |

> **비유**: AO 오차 예산을 여러 고리로 이루어진 사슬로 생각하라. 전체 성능(스트렐 비)은 가장 약한 고리에 의해 제한된다. 뛰어난 DM(작은 피팅 오차)을 가지고 있어도 느린 제어 루프(큰 시간 오차)가 있으면 여전히 성능이 나쁠 것이다. 좋은 AO 공학이란 어느 한 항이 지배적이지 않도록 모든 오차 항을 균형잡는 것을 의미한다.

### 8.3 하늘 커버리지

자연 안내성(NGS, Natural Guide Star) AO는 등플라나틱 각도 내에서 충분히 밝은 별이 필요하다. 그러한 별을 찾을 확률이 **하늘 커버리지(sky coverage)**다. 제한 등급 $m_V = 14$이고 $\theta_0 = 5''$일 때:

$$\text{하늘 커버리지} \approx 1\%$$

이 극히 낮은 커버리지가 레이저 안내성(§9 참조) 개발의 동기가 된다.

---

## 9. 고급 AO 개념

### 9.1 레이저 안내성 (LGS)

밝은 자연 안내성이 드물기 때문에, AO 시스템은 레이저를 사용하여 인공 신호원(beacon)을 만든다:

**레일리 LGS(Rayleigh LGS)**: 10~20 km 고도에서 분자에 의해 산란되는 펄스 레이저(일반적으로 532 nm). 시간 게이팅(time-gated) 검출이 원하는 고도를 분리한다. **원뿔 효과(cone effect)**에 의해 제한된다 — 레이저 빔이 원통이 아닌 원뿔을 샘플링하므로 고고도 난류가 불충분하게 샘플링된다.

**나트륨 LGS(Sodium LGS)**: 나트륨 D2 라인(589 nm)에 동조된 CW 또는 펄스 레이저가 ~90 km 고도의 중간권(mesospheric) 나트륨층을 여기시킨다. 더 높은 신호원 고도가 원뿔 효과를 줄이지만 더 비싼 레이저 기술이 필요하다.

**한계**:
- LGS는 팁-틸트(tip-tilt)를 측정할 수 없다(레이저가 올라갔다가 같은 대기를 통해 내려오므로 — 틸트는 상반적(reciprocal)이어서 상쇄된다). 팁-틸트를 위해 여전히 어두운 NGS가 필요하다.
- 원뿔 효과(초점 등플라나틱 이탈, focus anisoplanatism)가 단일 LGS 시스템의 보정 가능한 시야를 제한한다.

### 9.2 다중 켤레 AO (MCAO)

MCAO는 여러 안내성과 여러 변형 거울을 사용하며, 각 거울은 다른 난류층에 켤레이어서 더 넓은 시야를 보정한다:

```
Atmosphere:  Layer 1 (ground)    Layer 2 (high)
                │                    │
DM1 ◄──────────┘                    │
(conjugate to ground)                │
                                     │
DM2 ◄───────────────────────────────┘
(conjugate to 10 km)
```

시야에 걸쳐 퍼진 3~5개의 레이저 안내성과 2~3개의 DM을 사용하여, MCAO는 직경 1~2 각분에 걸쳐 균일한 보정을 달성한다 — 등플라나틱 각도보다 훨씬 크다.

**예시**: 제미니 사우스(Gemini South)의 GeMS 시스템은 5개의 나트륨 LGS와 2개의 DM을 사용하여 85 각초 시야를 보정한다.

### 9.3 지표층 AO (GLAO)

GLAO는 지면에 켤레인 단일 DM을 사용하여 지표층 난류만을 보정한다(전체 시잉의 50~80%에 기여). 이는 적당한 개선(50% 시잉 감소)을 제공하지만 매우 넓은 시야(각분에서 도까지)에 걸쳐 이루어진다.

### 9.4 극한 AO (ExAO)

극한 AO 시스템은 외계행성과 원반 주변(circumstellar disk)의 고대비 영상을 위해 설계되었다. 다음과 같은 특징이 있다:

- 매우 많은 구동기 수 (3000~10,000)
- 매우 빠른 루프 속도 (1~3 kHz)
- 항성광 억제를 위한 코로나그래프(coronagraph)
- 사후 처리 스페클 제거(ADI, SDI)
- 근적외선에서 스트렐 비 > 90% 목표

**예시**: 제미니 행성 촬영기(GPI, Gemini Planet Imager), VLT/SPHERE, 스바루(Subaru)/SCExAO.

### 9.5 초대형 망원경을 위한 AO

차세대 ELT(지름 25~39 m)는 처음부터 AO를 중심으로 설계되었다:

| 망원경 | 지름 | AO 시스템 | DM 구동기 수 | LGS |
|-----------|:--------:|-----------|:------------:|:---:|
| ELT (ESO) | 39 m | MAORY (MCAO) | ~5000 (M4) | 나트륨 6개 |
| TMT | 30 m | NFIRAOS (MCAO) | ~5000 | 나트륨 6개 |
| GMT | 25 m | LTAO/GLAO | 분절당 ~3500 | 나트륨 6개 |

---

## 10. 파이썬 예제

### 10.1 샤크-하트만 점 시뮬레이션

```python
import numpy as np

def shack_hartmann_spots(wavefront: np.ndarray, n_sub: int,
                         pupil_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Shack-Hartmann spot positions from a wavefront.

    The wavefront is divided into n_sub × n_sub subapertures. For each
    subaperture, the average slope is computed from finite differences.

    Parameters
    ----------
    wavefront : 2D array  — Wavefront phase (radians) on a square grid
    n_sub : int  — Number of subapertures across the pupil diameter
    pupil_radius : float  — Physical radius of the pupil (same units as wavefront grid)

    Returns
    -------
    sx, sy : arrays of shape (n_sub, n_sub)  — Slope measurements (rad/m)
    """
    N = wavefront.shape[0]
    sub_size = N // n_sub
    dx = 2 * pupil_radius / N  # pixel size in physical units

    sx = np.zeros((n_sub, n_sub))
    sy = np.zeros((n_sub, n_sub))

    for i in range(n_sub):
        for j in range(n_sub):
            # Extract subaperture
            sub = wavefront[i*sub_size:(i+1)*sub_size,
                            j*sub_size:(j+1)*sub_size]
            # Average x-slope from finite differences
            dWdx = np.diff(sub, axis=1) / dx
            sx[i, j] = np.mean(dWdx)
            # Average y-slope
            dWdy = np.diff(sub, axis=0) / dx
            sy[i, j] = np.mean(dWdy)

    return sx, sy
```

### 10.2 영향 행렬과 재구성기

```python
def build_influence_matrix(n_act: int, n_sub: int,
                           coupling: float = 0.15) -> np.ndarray:
    """Build a simplified influence matrix for a DM with Gaussian
    influence functions, sensed by a Shack-Hartmann WFS.

    The influence matrix M maps DM commands to WFS slopes:
        s = M @ c

    The command matrix (reconstructor) is R = pinv(M).

    Parameters
    ----------
    n_act : int  — Number of actuators across the pupil
    n_sub : int  — Number of WFS subapertures across the pupil
    coupling : float  — Inter-actuator coupling (0 to 1)

    Returns
    -------
    M : 2D array of shape (2*n_sub^2, n_act^2)  — Influence matrix
    """
    # Actuator positions (normalized to [0, 1])
    act_pos = np.linspace(0, 1, n_act)
    ax, ay = np.meshgrid(act_pos, act_pos)
    act_x = ax.ravel()
    act_y = ay.ravel()

    # Subaperture center positions
    sub_pos = np.linspace(0.5/n_sub, 1 - 0.5/n_sub, n_sub)
    sx_grid, sy_grid = np.meshgrid(sub_pos, sub_pos)
    sub_x = sx_grid.ravel()
    sub_y = sy_grid.ravel()

    n_slopes = 2 * len(sub_x)
    n_actuators = len(act_x)
    M = np.zeros((n_slopes, n_actuators))

    # Width of influence function from coupling
    # coupling = exp(-ln2 * (pitch/w)^2), so w = pitch / sqrt(-ln(coupling)/ln2)
    pitch = 1.0 / (n_act - 1) if n_act > 1 else 1.0
    w = pitch / np.sqrt(-np.log(coupling) / np.log(2))

    for k in range(n_actuators):
        # Distance from subaperture centers to actuator k
        dx = sub_x - act_x[k]
        dy = sub_y - act_y[k]
        r2 = dx**2 + dy**2
        # Gaussian influence
        phi = np.exp(-np.log(2) * r2 / w**2)
        # Slopes are derivatives of the influence function
        # d(phi)/dx = phi * (-2*ln2*dx/w^2)
        dphi_dx = phi * (-2 * np.log(2) * dx / w**2)
        dphi_dy = phi * (-2 * np.log(2) * dy / w**2)
        M[:len(sub_x), k] = dphi_dx
        M[len(sub_x):, k] = dphi_dy

    return M
```

### 10.3 폐루프 AO 시뮬레이션

```python
def ao_closed_loop(phase_screens: list[np.ndarray],
                   n_sub: int, n_act: int,
                   gain: float = 0.5,
                   coupling: float = 0.15,
                   pupil_radius: float = 1.0) -> dict:
    """Run a closed-loop AO simulation over a sequence of phase screens.

    Each phase screen represents the atmospheric wavefront at one time step.
    The AO system measures slopes, reconstructs, and applies correction.

    Returns a dict with residual wavefronts and Strehl ratios per step.
    """
    N = phase_screens[0].shape[0]

    # Build interaction matrix and reconstructor
    M = build_influence_matrix(n_act, n_sub, coupling)
    # Pseudo-inverse via SVD (truncate small singular values)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_inv = np.where(s > 0.01 * s[0], 1.0 / s, 0.0)
    R = (Vt.T * s_inv) @ U.T  # command matrix

    # Initialize DM commands
    dm_commands = np.zeros(n_act * n_act)
    strehls = []
    rms_list = []

    for screen in phase_screens:
        # Apply current DM correction
        # Reconstruct DM surface from commands (simplified: nearest-neighbor)
        dm_surface = _commands_to_surface(dm_commands, n_act, N, coupling)
        residual = screen - dm_surface

        # Measure slopes
        sx, sy = shack_hartmann_spots(residual, n_sub, pupil_radius)
        slopes = np.concatenate([sx.ravel(), sy.ravel()])

        # Reconstruct and update commands
        correction = R @ slopes
        dm_commands += gain * correction

        # Compute metrics
        pupil_mask = _circular_mask(N)
        rms = np.std(residual[pupil_mask])
        strehl = np.exp(-(2 * np.pi * rms)**2) if rms < 0.5 else 0.0
        strehls.append(strehl)
        rms_list.append(rms)

    return {'strehls': np.array(strehls), 'rms': np.array(rms_list)}


def _commands_to_surface(commands: np.ndarray, n_act: int, N: int,
                         coupling: float) -> np.ndarray:
    """Convert DM actuator commands to a wavefront surface on an NxN grid."""
    surface = np.zeros((N, N))
    act_pos = np.linspace(0, 1, n_act)
    grid = np.linspace(0, 1, N)
    X, Y = np.meshgrid(grid, grid)

    pitch = 1.0 / (n_act - 1) if n_act > 1 else 1.0
    w = pitch / np.sqrt(-np.log(coupling) / np.log(2))

    for idx, cmd in enumerate(commands):
        if abs(cmd) < 1e-12:
            continue
        i, j = divmod(idx, n_act)
        r2 = (X - act_pos[j])**2 + (Y - act_pos[i])**2
        surface += cmd * np.exp(-np.log(2) * r2 / w**2)
    return surface


def _circular_mask(N: int) -> np.ndarray:
    """Create a circular boolean mask on an NxN grid."""
    y, x = np.mgrid[-1:1:complex(N), -1:1:complex(N)]
    return (x**2 + y**2) <= 1.0
```

---

## 11. 요약

| 개념 | 핵심 공식 / 아이디어 |
|---------|--------------------|
| 시잉 한계 | $\theta_{\text{seeing}} \approx 0.98\lambda/r_0$ |
| 프리드 매개변수 | $r_0 \propto \lambda^{6/5}$, 500 nm에서 일반적으로 10~20 cm |
| 그린우드 주파수 | $f_G = 0.427 v_{\text{eff}}/r_0$, 일반적으로 20~50 Hz |
| 등플라나틱 각도 | $\theta_0 = 0.314 r_0/\bar{h}$, 일반적으로 2~5 각초 |
| 간섭 시간 | $\tau_0 = 0.314 r_0/v_{\text{eff}}$ |
| 샤크-하트만 | 마이크로렌즈 어레이가 국소 기울기 $\nabla W$ 측정 |
| 최소제곱 재구성 | $\hat{\mathbf{a}} = \mathbf{D}^+\mathbf{s}$ (모드) 또는 $\hat{\mathbf{w}} = \mathbf{G}^+\mathbf{s}$ (구역) |
| 영향 함수 | $W_{\text{DM}} = \sum_k c_k \phi_k(\mathbf{r})$ |
| 피팅 오차 | $\sigma_{\text{fit}}^2 = \alpha(d/r_0)^{5/3}$ |
| 시간 오차 | $\sigma_{\text{temp}}^2 = (f_G/f_{-3\text{dB}})^{5/3}$ |
| 적분기 이득 | $\mathbf{c}_{k+1} = \mathbf{c}_k + g\hat{\mathbf{w}}_k$ |
| 스트렐 비 | $S \approx e^{-(2\pi\sigma)^2}$ |
| 하늘 커버리지 (NGS) | 가시광선 파장에서 ~1% |
| 레이저 안내성 | 10~90 km의 인공 신호원; 팁-틸트 측정 불가 |
| MCAO | 난류층에 켤레인 다중 DM; 넓은 시야 |

---

## 12. 연습 문제

### 연습 문제 1: AO 시스템 설계

$r_0 = 12$ cm (500 nm 기준)이고 $v_{\text{eff}} = 15$ m/s인 관측지에 있는 4미터 망원경을 위한 샤크-하트만 AO 시스템을 설계하고 있다. 과학 파장은 $K$-band (2.2 $\mu$m)이다. (a) $K$-band에서 $r_0$, $f_G$, $\theta_0$, $\tau_0$를 계산하라. (b) 부분 구경 수와 DM 구동기 수를 선택하라. (c) 시간 오차가 50 nm RMS 이하가 되려면 어떤 루프 속도가 필요한가? (d) 피팅, 시간, 30 nm 잡음 오차 항을 가정하여 총 스트렐 비를 추정하라.

### 연습 문제 2: 파면 재구성

$D/r_0 = 10$인 무작위 대기 파면을 생성하라(15단원의 콜모고로프(Kolmogorov) 위상 스크린 함수 사용). (a) 16×16 샤크-하트만 측정을 시뮬레이션하라(각 부분 구경에서 평균 기울기 계산). (b) 구역 방식과 모드(체르니케, 36 모드) 방식 모두로 파면을 재구성하라. (c) 재구성 오차 맵을 비교하라. (d) 측정 잡음(기울기에 SNR = 10, 20, 50의 가우시안 잡음 추가)에 따라 재구성 오차가 어떻게 변하는가?

### 연습 문제 3: 폐루프 시뮬레이션

§10.3에 제공된 `ao_closed_loop` 함수(또는 직접 구현)를 사용하여, $D/r_0 = 8$인 콜모고로프 위상 스크린에서 10×10 WFS를 가진 12×12 구동기 AO 시스템의 100 시간 단계를 시뮬레이션하라. (a) 이득 $g = 0.3, 0.5, 0.7, 0.9$에 대해 시간 단계별 스트렐 비를 그래프로 나타내라. (b) 어떤 이득이 최고 평균 스트렐 비를 제공하는가? (c) 최선의 경우에 대해 보정 전/후 파면을 나란히 보여라. (d) 보정된 경우의 PSF를 계산하고 회절 한계 PSF와 비교하라.

### 연습 문제 4: 레이저 안내성 한계

나트륨 레이저 안내성이 $r_0 = 15$ cm (500 nm 기준)인 10미터 망원경 위 $h = 90$ km 고도에 생성된다. (a) 원뿔 효과의 각도 범위(LGS 빔과 평면파 사이의 차이)를 계산하라. (b) $\sigma_{\text{cone}}^2 = (D/d_0)^{5/3}$ 공식(여기서 $d_0 \approx 2.91 r_0 (h/\bar{h})^{6/5}$이고 $\bar{h} = 5$ km)을 사용하여 초점 등플라나틱 이탈 오차를 추정하라. (c) 원뿔 효과 오차가 20×20 구동기 시스템의 피팅 오차와 같아지는 망원경 지름은 얼마인가? (d) MCAO는 원뿔 효과를 어떻게 완화하는가?

---

## 13. 참고문헌

1. Hardy, J. W. (1998). *Adaptive Optics for Astronomical Telescopes*. Oxford University Press. — 천문 AO의 기초 교재.
2. Roddier, F. (Ed.) (1999). *Adaptive Optics in Astronomy*. Cambridge University Press. — AO의 모든 측면을 다루는 포괄적인 편집 서적.
3. Tyson, R. K. (2015). *Principles of Adaptive Optics* (4th ed.). CRC Press. — 시스템 공학을 강조한 접근하기 쉬운 입문서.
4. Davies, R., & Kasper, M. (2012). "Adaptive optics for astronomy." *Annual Review of Astronomy and Astrophysics*, 50, 305–351. — 훌륭한 현대적 리뷰.
5. Roddier, F. (1988). "Curvature sensing and compensation: a new concept in adaptive optics." *Applied Optics*, 27(7), 1223–1225. — 곡률 센서 원본 논문.
6. Ragazzoni, R. (1996). "Pupil plane wavefront sensing with an oscillating prism." *Journal of Modern Optics*, 43(2), 289–293. — 피라미드 센서.
7. Rigaut, F. (2015). "Astronomical adaptive optics." *Publications of the Astronomical Society of the Pacific*, 127(958), 1197–1203. — AO 시스템과 성능 개요.
8. Guyon, O. (2005). "Limits of adaptive optics for high-contrast imaging." *The Astrophysical Journal*, 629, 592–614. — 외계행성 영상을 위한 극한 AO 이론.

---

[← 이전: 15. 체르니케 다항식](15_Zernike_Polynomials.md) | [다음: 17. 분광학 →](17_Spectroscopy.md)
