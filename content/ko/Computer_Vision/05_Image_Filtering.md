# 이미지 필터링

## 개요

이미지 필터링(Image Filtering)은 이미지의 픽셀 값을 주변 픽셀을 고려하여 변환하는 작업입니다. 노이즈 제거, 블러, 샤프닝 등 다양한 효과를 낼 수 있습니다. 이 문서에서는 커널과 컨볼루션의 개념부터 OpenCV의 다양한 필터 함수까지 학습합니다.

**난이도**: ⭐⭐ (초급-중급)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 커널(Kernel)과 컨볼루션(Convolution) 개념 이해
2. 다양한 블러 필터 (`blur`, `GaussianBlur`, `medianBlur`, `bilateralFilter`)
3. 엣지 보존 스무딩
4. 커스텀 필터와 샤프닝 구현

---

## 목차

OpenCV 함수 참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 컨볼루션 수학, 주파수 영역 관점, 그리고 선형 필터가 경계를 흐리는 반면 양방향 필터 같은 비선형 필터가 경계를 보존하는 이유를 다룹니다.

1. [커널과 컨볼루션](#1-커널과-컨볼루션)
2. [평균 블러 - blur()](#2-평균-블러---blur)
3. [가우시안 블러 - GaussianBlur()](#3-가우시안-블러---gaussianblur)
4. [중앙값 블러 - medianBlur()](#4-중앙값-블러---medianblur)
5. [양방향 필터 - bilateralFilter()](#5-양방향-필터---bilateralfilter)
6. [커스텀 필터 - filter2D()](#6-커스텀-필터---filter2d)
7. [샤프닝 필터](#7-샤프닝-필터)
8. [연습 문제](#8-연습-문제)
9. [다음 단계](#9-다음-단계)
10. [참고 자료](#10-참고-자료)

---

## 이론과 원리

이미지 필터링은 그 본질이 **2D 신호에 적용되는 수학적 컨볼루션(convolution)**입니다. 이 레슨의 뒤에 나오는 모든 선형 필터 — 평균 블러, 가우시안 블러, Sobel, Laplacian, 샤프닝, 엠보스, `filter2D`의 커스텀 커널 — 는 같은 연산에 서로 다른 커널을 꽂은 것뿐입니다. 수학을 이해하면 커널을 실행해 보기 전에 그 효과를 예측할 수 있고, 특정 목적에 맞는 새 커널을 설계할 수 있으며, 몇몇 필터(중앙값, 양방향)가 왜 이 틀 바깥에 있어 다르게 분석해야 하는지가 명확해집니다.

이 섹션은 다음을 다룹니다:

- **(A) 컨볼루션** — 수학적 정의, 구현이 실제로는 *교차 상관(cross-correlation)*을 계산하는 이유, 선형 평행이동 불변성(LSI) 속성, 분리 가능 커널이 빠른 이유.
- **(B) 주파수 영역 관점** — "블러 커널 = 저역 통과 필터"가 문자 그대로 참인 이유, 같은 원리로 블러링·샤프닝·에지 검출이 통일되는 방식.
- **(C) 가우시안 커널** 심층 — `σ`와 커널 크기의 관계, 가우시안이 그토록 많은 성질을 동시에 만족하는 *유일한* 블러 커널인 이유, 확산(diffusion)과의 연결.
- **(D) 양방향 필터(Bilateral filter)** — *range 커널*을 도입해 의도적으로 선형성을 깨고 경계를 보존하는 방법.
- **(E) 언샤프 마스킹(Unsharp masking)** — "샤프닝 = 원본 + α·(원본 − 블러)" 공식이 주파수 분해로부터 자연스럽게 유도되는 과정.
- **(F) 경계 처리** — 커널이 이미지 가장자리를 넘어갈 때 각 `BORDER_*` 모드가 실제로 의미하는 것.

### A. 컨볼루션: 수학적 핵심

#### A.1 정의

1D 연속 신호 `f(t)`와 커널 `h(t)`의 컨볼루션은

```
(f * h)(t) = ∫ f(τ) · h(t - τ) dτ
```

이산 2D 이미지에서의 대응은

```
(I * K)(x, y) = Σᵢ Σⱼ  I(x - i, y - j) · K(i, j)
```

합은 `K`가 0이 아닌 모든 오프셋 `(i, j)`에 대해 수행됩니다. **각 출력 픽셀은 국소 이웃 픽셀들의 가중합**이며, 가중치는 커널이 결정합니다. 서로 다른 필터 = 서로 다른 가중치 패턴.

#### A.2 교차 상관 vs 진짜 컨볼루션

위의 교과서적 정의는 커널을 뒤집습니다 — `K(i, j)`가 `I(x - i, y - j)`에 쓰이는데, 이는 합산 전에 커널을 180° 회전시키는 것과 같습니다. 반면 `cv2.filter2D`를 비롯한 이 레슨의 모든 선형 필터가 실제 계산하는 것은 **교차 상관(cross-correlation)** — 뒤집기 없음:

```
(I ⋆ K)(x, y) = Σᵢ Σⱼ  I(x + i, y + j) · K(i, j)
```

**대칭 커널** — 평균, 가우시안, Laplacian, 대부분의 샤프닝, 대부분의 엠보스 — 에서는 뒤집기가 아무 효과가 없으므로 교차 상관과 컨볼루션이 같은 결과를 냅니다. **비대칭 커널**(대표적으로 Sobel)에서는 180° 회전만큼 다릅니다. 실용 규칙: 비대칭 커널로 교과서적 컨볼루션 결과를 원한다면 `cv2.flip(kernel, -1)`로 먼저 뒤집고 `filter2D`에 넘기세요. 이 레슨에서는 기술적으로 교차 상관인 경우에도 "컨볼루션"이라는 용어를 씁니다. 실무에서는 두 용어가 혼용됩니다.

#### A.3 선형성과 평행이동 불변성 (LSI)

고정된 커널과의 컨볼루션은 **선형 평행이동 불변(Linear Shift-Invariant)** 연산입니다:

- **선형**: `I1 + I2`에 필터를 적용한 결과 = 각각에 필터를 적용해 더한 결과. 입력을 2배 하면 출력도 2배.
- **평행이동 불변**: 입력을 `(a, b)`만큼 이동하면 출력도 정확히 `(a, b)`만큼 이동. 같은 필터가 이미지 전체에 동일하게 적용됨.

`cv2.filter2D`로 만든 모든 필터 — 평균, 가우시안, Sobel, Laplacian, 샤프닝, 엠보스, §6-7의 모든 커스텀 커널 — 는 LSI입니다. **중앙값 필터**(§4)와 **양방향 필터**(§5)는 선형이 *아닙니다*. 바로 이 비선형성 덕분에 선형 필터로는 불가능한 일(이상치 제거, 경계 보존)이 가능해집니다. §B의 주파수 영역 도구들은 LSI 필터에만 직접 적용됩니다.

#### A.4 분리 가능 커널

2D 커널 `K`가 두 1D 커널의 외적으로 분해되면 — `K(i, j) = u(i) · v(j)` — 그 커널은 **분리 가능(separable)** 합니다. 이 성질이 성립하면 2D 컨볼루션은 1D 컨볼루션 두 번(행 방향 한 번, 열 방향 한 번)과 같습니다:

```
I * K  =  (I *row u) *col v
```

`k × k` 커널에 대해 출력 픽셀당 `k²`회가 아니라 `2k`회의 곱셈-덧셈이 됩니다. `k = 9`이면 18 vs 81 — 공짜로 4.5배 가속. 가우시안, 평균/박스, Sobel 커널은 모두 분리 가능하며, OpenCV의 `GaussianBlur`, `boxFilter`, `Sobel`은 내부적으로 이를 이용합니다. 직접 분리 가능한 커스텀 커널을 만들 때는 `cv2.sepFilter2D`를 쓸 수 있습니다.

### B. 주파수 영역 관점

어떠한 이미지도 2D 푸리에 변환을 통해 서로 다른 공간 주파수의 사인파의 합으로 분해됩니다. 저주파는 부드럽고 천천히 변하는 내용(넓은 영역, 그라데이션)에, 고주파는 급격한 변화(경계, 텍스처, 노이즈)에 대응합니다.

#### B.1 컨볼루션 정리

필터링에 적용된 푸리에 분석의 중심 결과:

```
공간 영역의 컨볼루션  ⇔  주파수 영역의 곱셈

F(I * K) = F(I) · F(K)
```

즉, **커널의 효과는 그 푸리에 변환 `F(K)`에 의해 완전히 기술됩니다**. `F(K)`가 1에 가까운 주파수는 거의 통과되고, 0에 가까운 주파수는 억제됩니다. 이것이 모든 선형 필터를 *통과시키는 성분*에 따라 분류하는 이유입니다:

- **저역 통과(Low-pass)**: `F(K)`가 0 주파수 근처에서 크고 고주파에서 작다. 부드러운 내용을 통과시키고 급격한 변화를 막는다. 이것이 블러. (평균, 가우시안.)
- **고역 통과(High-pass)**: `F(K)`가 0 주파수에서 ~0이고 고주파에서 크다. 부드러운 내용을 막고 경계와 텍스처를 강조한다. (Laplacian.)
- **대역 통과(Band-pass)**: 특정 주파수 링만 통과시킨다. (Difference-of-Gaussians, §8 에지 검출에서 사용.)

#### B.2 블러링이 노이즈를 제거하는 이유 (그리고 경계도 흐리는 이유)

센서 노이즈와 소금-후추 노이즈는 픽셀 간 급격한 변화로 나타납니다 — 거의 정의상 고주파 성분입니다. 실제 장면의 특징(객체, 음영 그라데이션)은 더 천천히 변하고 저주파 대역에 집중되어 있습니다. 저역 통과 커널은 노이즈가 사는 고주파 대역을 억제하면서 신호가 사는 저주파 대역을 보존합니다.

걸림돌: **날카로운 경계도 고주파 성분입니다**. 선형 저역 통과 필터는 "노이즈 때문에 생긴 고주파"와 "경계 때문에 생긴 고주파"를 구별할 방법이 없습니다. 둘 다 똑같이 약화시킵니다. 그래서 모든 선형 블러는 경계까지 흐리게 만듭니다. 진짜 경계 보존 평활화를 하려면 비선형으로 가야 합니다(§D).

#### B.3 샤프닝 = 고주파 증폭

"샤프닝 = 원본 + α · (원본 − 블러)" 같은 공식은 주파수 영역에서 보면 자연스럽습니다. `(원본 − 블러)`는 이미지의 고주파 부분을 분리하고, `α > 0`으로 다시 더하면 블러가 억제했을 주파수가 정확히 증폭됩니다. §E가 이를 엄밀하게 유도합니다.

### C. 가우시안 커널, 심층

#### C.1 공식

```
G(x, y; σ) = (1 / (2π σ²)) · exp( -(x² + y²) / (2σ²) )
```

각 기호:

- `(x, y)`는 커널 중심으로부터의 오프셋. 최댓값은 `(0, 0)`.
- `σ`는 픽셀 단위의 표준편차. 종 모양의 폭을 결정합니다 — `σ`가 크면 가중치가 멀리 퍼지고 블러가 강해집니다.
- `1 / (2π σ²)`는 2D 가우시안의 정규화. 연속 적분이 1이 되도록 해 전체 밝기를 보존합니다. OpenCV는 연속 함수를 격자 위에서 샘플링한 뒤 이산 합을 다시 1로 정규화해 샘플링 오차를 제거합니다.

#### C.2 `σ`와 커널 크기의 관계

`σ`는 블러 강도를 설정합니다. 커널 크기는 계산이 얼마나 넓은 이웃을 볼지를 설정합니다. 둘은 일관되어야 합니다. `ksize`가 `σ`에 비해 너무 작으면 꼬리가 잘려 더 이상 진짜 가우시안이 아닙니다 — 의도보다 약하고 덜 등방적인 블러가 됩니다.

표준 규칙은 **`ksize ≥ 6σ + 1`** — 질량의 99.7% 이상(±3σ)을 담습니다. 둘 중 하나를 `0`으로 두면 OpenCV가 나머지로부터 유도합니다:

```
σ = 0.3 · ((ksize - 1) · 0.5 - 1) + 0.8      # sigma=0, ksize 지정일 때
```

대략 역으로 하면 `ksize ≈ 2·⌈3σ⌉ + 1`입니다. 블러 강도가 관심사라면 `σ`를 명시적으로 지정하고 `ksize=(0, 0)`을 넘기세요 — OpenCV가 충분히 큰 커널을 골라줍니다.

#### C.3 가우시안이 *그* 등방 블러인 이유

가우시안은 다음 모든 성질을 동시에 만족하는 (스케일 차이를 제외하면) **유일한** 커널입니다:

- **등방(Isotropic)**: 반지름 대칭. 블러된 점이 어느 방향에서 봐도 같아 보인다.
- **분리 가능(Separable)**: `G(x, y; σ) = G(x; σ) · G(y; σ)`. 2D 블러 = 1D 블러 두 번 — §A.4의 `O(k²) → O(2k)` 속도 향상이 공짜.
- **비음수(Non-negative)**: 음수 가중치가 없어 모든 출력 값이 입력들의 진짜 가중 평균 — 값이 입력 범위 밖으로 튈 수 없다.
- **자기 컨볼루션에 닫힘**: 가우시안을 다른 가우시안으로 블러하면 다시 가우시안. `GaussianBlur(σ₁)` 뒤 `GaussianBlur(σ₂)` = 한 번의 `GaussianBlur(√(σ₁² + σ₂²))`. 이미지 피라미드와 스케일-공간 표현이 이 성질에 기반합니다.

이 네 가지를 동시에 만족하는 다른 커널은 없습니다. 이 유일성이 컴퓨터 비전 전반(SIFT의 스케일-공간 §13, Canny의 전처리 §8, 특징 검출, 신경망의 암묵적 정칙화 사전분포 등)에서 가우시안이 계속 등장하는 이유입니다.

#### C.4 확산과의 연결

분산 `σ²`인 `GaussianBlur` 적용은 수학적으로 2D **열 방정식** `∂I/∂t = ∇²I`를 시간 `t = σ²/2`만큼 전진시키는 것과 같습니다. 블러링은 말 그대로 픽셀 값이 서로에게 퍼져나가는 것이며, 금속판 속에서 열이 확산되는 것과 같은 현상입니다. 이 관점이 스케일-공간 방법과 *비등방 확산(anisotropic diffusion)*의 근거입니다. 후자는 확산 속도가 국소 기울기에 의존하는 경계 보존 평활화 방법입니다.

### D. 양방향 필터: 선형성을 깨서 경계를 보존하기

선형 저역 통과 필터는 "노이즈로 인한 고주파"와 "경계로 인한 고주파"를 구별할 수 없습니다 — `F(K)`에게 둘은 동일합니다. 양방향 필터는 커널을 **이웃 픽셀들의 값에 의존하게 만들어** 이 문제를 우회합니다. 이 데이터 의존성이 선형성을 깨뜨리며, 그것이 바로 목표입니다.

#### D.1 공식

```
BF[I](x) = (1 / W(x)) · Σ_{y ∈ Ω}   Gσ_s(‖x - y‖) · Gσ_r(|I(x) - I(y)|) · I(y)
```

기호 의미:

- **Gσ_s** (공간 커널, spatial kernel)는 이웃을 *거리*로 가중치 부여 — 가까운 픽셀이 더 기여. 일반 가우시안 블러의 커널과 같은 역할. OpenCV의 `sigmaSpace`로 제어.
- **Gσ_r** (range 커널)은 이웃을 *밝기 유사도*로 가중치 부여 — 중심과 색이 비슷한 픽셀이 더 기여하고, 경계 건너편 픽셀은 거의 무시. `sigmaColor`로 제어.
- **W(x)**는 픽셀별 정규화로 가중치의 합을 1로 만듭니다. `x`에 의존한다는 점이 필터가 비선형인 이유의 일부입니다.

#### D.2 경계가 살아남는 이유

평탄한 영역 내부에서는 모든 이웃이 밝기가 비슷하므로 `Gσ_r ≈ 1` — 양방향 필터는 평범한 공간 가우시안으로 퇴화합니다. 경계 근처에서는 반대편 픽셀의 밝기가 매우 다르므로 그들의 range 가중치가 ≈ 0 — 사실상 무시됩니다. 평균은 중심 픽셀이 이미 속한 경계의 한쪽 면 내에서만 이뤄집니다. 평탄 영역은 평활화되고, 경계 전이부는 보존됩니다.

대가는 계산 비용과 분석 용이성입니다. 필터가 데이터에 의존하므로 분리 가능성(§A.4)이 성립하지 않고, §B의 푸리에 도구도 직접 적용되지 않습니다. 소박한 구현은 픽셀당 `O(k²)`이며, 빠른 근사(예: permutohedral lattice)가 실시간 사용을 가능하게 합니다.

### E. 언샤프 마스킹, 주파수 분해에서 유도하기

어떤 이미지든 저주파 부분과 고주파 부분으로 분해할 수 있습니다: `I = I_low + I_high`, 여기서 `I_low = GaussianBlur(I)`이고 `I_high = I - I_low`. "샤프닝"은 고주파 부분을 증폭하는 것:

```
I_sharp = I_low + (1 + α) · I_high
        = I_low + (1 + α) · (I - I_low)
        = I + α · (I - I_low)
        = I + α · (I - GaussianBlur(I))
```

마지막 식은 §7의 `unsharp_mask`가 구현하는 공식과 정확히 같습니다. `α`는 `amount` 파라미터 — 값이 클수록 고주파 성분이 더 강하게 증폭됩니다.

**threshold** 파라미터는 평탄 영역을 보호합니다. 어떤 `|I - I_low|` 이하의 차이는 실제 디테일이 아니라 노이즈로 간주되어 원본 픽셀이 유지됩니다. 이것이 없으면 원래 평탄해야 할 영역(피부, 하늘 등)에서 센서 노이즈가 증폭되어 거칠고 알갱이진 모습이 나타납니다.

같은 효과를 한 번의 연산으로 내는 샤프닝 커널 `[[0,-1,0],[-1,5,-1],[0,-1,0]]`도 바로 이 유도의 결과입니다. 이는 `I + 1·(I - Laplacian 기반 블러)`를 단일 3×3 컨볼루션으로 압축한 것입니다. 중심의 `5`는 `1 (원본) + 4 (증폭) = 5`이고, 네 개의 `-1`을 합하면 `-4` — 커널의 총합은 `1`이 되어 전체 밝기가 보존됩니다.

### F. 경계 처리

커널이 이미지 가장자리를 넘어가면 존재하지 않는 픽셀을 참조하게 됩니다. OpenCV의 모든 필터는 이 "가상 픽셀"을 어떤 값으로 채울지 제어하는 `borderType` 파라미터를 제공합니다. 각 모드의 수학적 의미 (1D 행 `abcdefgh` 양쪽에 가상 픽셀 3개씩):

```
BORDER_CONSTANT     |  0  0  0 | a b c d e f g h |  0  0  0 |   (상수 채움, 기본은 0)
BORDER_REPLICATE    |  a  a  a | a b c d e f g h |  h  h  h |   (가장자리 픽셀 복제)
BORDER_REFLECT      |  c  b  a | a b c d e f g h |  h  g  f |   (가장자리 픽셀 포함 반사)
BORDER_REFLECT_101  |  d  c  b | a b c d e f g h |  g  f  e |   (가장자리 픽셀 제외 반사) ← OpenCV 기본
BORDER_WRAP         |  f  g  h | a b c d e f g h |  a  b  c |   (주기적 — 이미지가 반복)
```

효과:

- **BORDER_CONSTANT**는 대칭·0 평균 커널(기울기 연산자)에는 괜찮지만, 평균·가우시안 커널에는 경계에서 어두운 후광을 만듭니다. 0과의 평균이 값을 끌어내리기 때문.
- **BORDER_REPLICATE**는 가장 저렴하고 자연 이미지에 대개 안전한 기본값 — 새로운 값을 도입하지 않습니다.
- **BORDER_REFLECT_101** (대부분 필터의 OpenCV 기본값)은 경계를 넘어 미분이 연속적이 되도록 가장 매끄럽게 확장합니다 — 이미지 가장자리에서 에지 검출 커널이 만들어내는 가짜 강한 응답이 없습니다. Sobel, Laplacian, Canny 전처리, 다중 스케일 피라미드에 선호.
- **BORDER_WRAP**은 이미 주기성을 가정하는 DFT 기반 주파수 영역 연산(§B)에서 의미가 있습니다.

### 이론에서 아래 함수들로

이 레슨의 뒤에 나오는 각 섹션은 위 틀의 구체적 구현입니다:

- `cv2.filter2D(..., kernel, borderType=...)` — 사용자 정의 커널로 2D 교차 상관(§A.2), 경계 모드는 §F.
- `cv2.sepFilter2D` — §A.4를 활용한 분리 가능 2D 컨볼루션.
- `cv2.blur(ksize)` / `cv2.boxFilter` — 크기 `ksize`의 평평한 저역 통과 커널. 분리 가능. 픽셀 공간에서는 등방이지만 주파수 공간에서는 *그렇지 않음*(사이드로브 존재 — 그래서 대부분의 용도에서 가우시안보다 열등).
- `cv2.GaussianBlur(ksize, sigma)` — 분리 가능 가우시안 저역 통과 필터(§C). 범용 블러의 최선택.
- `cv2.medianBlur(ksize)` — 비선형 순서 통계 필터. 소금-후추 노이즈에 최적 — 한 개의 이상치는 중앙값에 도달할 수 없지만, 어떠한 선형 필터의 가중합에는 항상 영향을 주기 때문.
- `cv2.bilateralFilter(d, sigmaColor, sigmaSpace)` — §D.1을 구현한 비선형 경계 보존 평활화.
- **샤프닝 커널 `[[0,-1,0],[-1,5,-1],[0,-1,0]]`** — `amount = 1`인 §E를 한 번의 연산으로 구현.
- `unsharp_mask(amount, threshold)` — 노이즈 방지 임계값을 가진 §E의 명시적 구현.

---

## 1. 커널과 컨볼루션

### 커널(Kernel)이란?

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kernel                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   A kernel (or filter, mask) is a small matrix that defines    │
│   the operation to apply to an image. Typically 3x3, 5x5, 7x7. │
│                                                                 │
│   Example: 3x3 average filter kernel                            │
│                                                                 │
│        1/9   1/9   1/9         ┌───┬───┬───┐                   │
│                                │1/9│1/9│1/9│                   │
│        1/9   1/9   1/9    =    ├───┼───┼───┤                   │
│                                │1/9│1/9│1/9│                   │
│        1/9   1/9   1/9         ├───┼───┼───┤                   │
│                                │1/9│1/9│1/9│                   │
│                                └───┴───┴───┘                   │
│                                                                 │
│   Kernel size meaning:                                          │
│   - Larger size considers wider area                            │
│   - Large kernel = strong effect, slow processing              │
│   - Small kernel = weak effect, fast processing                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 컨볼루션(Convolution) 연산

```
┌─────────────────────────────────────────────────────────────────┐
│                      Convolution Operation                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Apply kernel to each pixel of input image to calculate new value│
│                                                                 │
│   Input image          3x3 kernel           Output              │
│   ┌───┬───┬───┬───┐   ┌───┬───┬───┐                           │
│   │ 1 │ 2 │ 3 │ 4 │   │1/9│1/9│1/9│                           │
│   ├───┼───┼───┼───┤   ├───┼───┼───┤      Result pixel:         │
│   │ 5 │ 6 │ 7 │ 8 │   │1/9│1/9│1/9│   (1+2+3+5+6+7+9+10+11)/9 │
│   ├───┼───┼───┼───┤   ├───┼───┼───┤      = 54/9 = 6            │
│   │ 9 │10 │11 │12 │   │1/9│1/9│1/9│                           │
│   ├───┼───┼───┼───┤   └───┴───┴───┘                           │
│   │13 │14 │15 │16 │                                            │
│   └───┴───┴───┴───┘                                            │
│                                                                 │
│   Process:                                                      │
│   1. Place kernel over image                                    │
│   2. Multiply corresponding pixels                              │
│   3. Sum all results                                            │
│   4. Move to next pixel and repeat                              │
│                                                                 │
│   Border handling:                                              │
│   - BORDER_CONSTANT: Fill with constant value (default 0)       │
│   - BORDER_REPLICATE: Replicate border pixels                   │
│   - BORDER_REFLECT: Reflect at border                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 컨볼루션 시각화

```python
import cv2
import numpy as np

def visualize_convolution(img, kernel):
    """Visualize convolution process (for learning)"""
    h, w = img.shape
    kh, kw = kernel.shape
    pad = kh // 2  # Why pad = kh//2: ensures the output has the same size as the input

    # Why zero-padding: border pixels need neighbors; padding with 0 is neutral for average kernels
    padded = np.pad(img, pad, mode='constant', constant_values=0)

    # Why float64: intermediate sums can exceed uint8 range (0-255); promotes before clipping
    result = np.zeros_like(img, dtype=np.float64)

    # Slow explicit loop — used here to make each step visible; use cv2.filter2D in production
    for y in range(h):
        for x in range(w):
            region = padded[y:y+kh, x:x+kw]
            result[y, x] = np.sum(region * kernel)

    return result


# Example
img = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=np.float64)

kernel = np.ones((3, 3)) / 9  # Average filter: weights sum to 1 to preserve overall brightness

result = visualize_convolution(img, kernel)
print("Input:\n", img)
print("\nResult:\n", result)
```

**블러링이 작동하는 이유는?** 이미지의 노이즈(noise)는 실제 장면 내용을 반영하지 않으면서 인접 픽셀이 급격히 다른, 빠른 고주파 픽셀 변동으로 나타납니다. 블러 커널이 이웃 픽셀을 평균하면 이러한 급격한 무작위 변동이 상쇄되는 반면, 실제 이미지 특징(여러 픽셀에 걸쳐 점진적으로 변하는)은 보존됩니다. 신호 처리 관점에서 블러 커널은 고주파 성분을 억제하는 **저역 통과 필터(low-pass filter)**입니다. 이것이 가우시안 블러가 엣지 검출 전처리 단계로 선호되는 이유이기도 합니다: 노이즈로 인한 거짓 엣지를 제거하면서 실제 구조적 경계는 보존합니다.

---

## 2. 평균 블러 - blur()

### 기본 사용법

평균 블러는 가장 단순한 블러 필터로, 커널 영역의 평균값을 사용합니다.

```python
import cv2

img = cv2.imread('image.jpg')

# blur(src, ksize)
# ksize: kernel size in (width, height) format

blur_3x3 = cv2.blur(img, (3, 3))
blur_5x5 = cv2.blur(img, (5, 5))
blur_7x7 = cv2.blur(img, (7, 7))
blur_15x15 = cv2.blur(img, (15, 15))

cv2.imshow('Original', img)
cv2.imshow('3x3 Blur', blur_3x3)
cv2.imshow('5x5 Blur', blur_5x5)
cv2.imshow('15x15 Blur', blur_15x15)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 평균 블러 커널

```
┌─────────────────────────────────────────────────────────────────┐
│                      Average Blur Kernel                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   3x3 average kernel:                                           │
│   ┌─────┬─────┬─────┐                                          │
│   │ 1/9 │ 1/9 │ 1/9 │                                          │
│   ├─────┼─────┼─────┤                                          │
│   │ 1/9 │ 1/9 │ 1/9 │  =  1/9 × [[1, 1, 1],                   │
│   ├─────┼─────┼─────┤           [1, 1, 1],                    │
│   │ 1/9 │ 1/9 │ 1/9 │           [1, 1, 1]]                    │
│   └─────┴─────┴─────┘                                          │
│                                                                 │
│   5x5 average kernel:                                           │
│   All values are 1/25                                           │
│                                                                 │
│   Features:                                                     │
│   - Simple and fast                                             │
│   - Edges also get blurred                                      │
│   - Effective for uniform noise removal                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### boxFilter()

`blur()`의 일반화된 버전입니다.

```python
import cv2

img = cv2.imread('image.jpg')

# normalize=True (default): Normalize kernel (average filter)
# normalize=False: Sum filter
blur_normalized = cv2.boxFilter(img, -1, (5, 5), normalize=True)
sum_filter = cv2.boxFilter(img, -1, (5, 5), normalize=False)

# Same as blur(img, (5, 5))
print(f"Difference: {np.sum(np.abs(cv2.blur(img, (5, 5)) - blur_normalized))}")  # 0
```

---

## 3. 가우시안 블러 - GaussianBlur()

### 가우시안 필터란?

가우시안 필터는 중심에 더 큰 가중치를 주는 블러 필터입니다. 자연스러운 블러 효과를 만들어냅니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Gaussian Kernel                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Gaussian distribution (normal distribution, bell shape):      │
│                                                                 │
│          ▲                                                      │
│          │     ████                                             │
│          │   ████████                                           │
│          │  ██████████                                          │
│          │ ████████████                                         │
│          │██████████████                                        │
│          └──────────────────▶                                   │
│                   Weight decreases away from center             │
│                                                                 │
│   3x3 Gaussian kernel (approximate):                            │
│   ┌─────┬─────┬─────┐                                          │
│   │ 1   │ 2   │ 1   │                                          │
│   ├─────┼─────┼─────┤  ×  1/16                                 │
│   │ 2   │ 4   │ 2   │                                          │
│   ├─────┼─────┼─────┤                                          │
│   │ 1   │ 2   │ 1   │                                          │
│   └─────┴─────┴─────┘                                          │
│                                                                 │
│   Features:                                                     │
│   - More natural result than average blur                       │
│   - Often used for edge detection preprocessing                │
│   - Control blur strength with sigma (σ) value                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 사용법

```python
import cv2

img = cv2.imread('image.jpg')

# GaussianBlur(src, ksize, sigmaX, sigmaY=0)
# ksize: Kernel size (must be odd)
# sigmaX: Standard deviation in X direction (0 = auto-calculate from kernel size)
# sigmaY: Standard deviation in Y direction (0 = same as sigmaX)

# Why ksize=(5,5) and sigmaX=0: letting OpenCV derive sigma from kernel size is the
# recommended default — it ties blur strength to a single intuitive parameter (kernel size)
# rather than requiring you to keep ksize and sigma in sync manually
blur1 = cv2.GaussianBlur(img, (5, 5), 0)

# Why (0,0) with explicit sigma: when you reason in terms of sigma (e.g., "blur ~3 pixels"),
# letting OpenCV pick the minimal sufficient kernel size avoids unnecessary computation
blur2 = cv2.GaussianBlur(img, (0, 0), 3)  # sigma=3

# Specify both kernel size and sigma
blur3 = cv2.GaussianBlur(img, (7, 7), 1.5)
```

### sigma와 커널 크기의 관계

```python
import cv2
import numpy as np

# Generate Gaussian kernel directly to check
def show_gaussian_kernel(ksize, sigma):
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel_2d = kernel @ kernel.T  # 1D to 2D
    print(f"Kernel ({ksize}x{ksize}, sigma={sigma}):")
    print(np.round(kernel_2d, 4))
    print(f"Sum: {np.sum(kernel_2d):.4f}\n")


show_gaussian_kernel(3, 0)   # sigma auto-calculated
show_gaussian_kernel(5, 0)
show_gaussian_kernel(5, 1.0)
show_gaussian_kernel(5, 2.0)

# Recommended: sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
```

### 평균 블러 vs 가우시안 블러

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Compare with same kernel size
ksize = 15
avg_blur = cv2.blur(img, (ksize, ksize))
gauss_blur = cv2.GaussianBlur(img, (ksize, ksize), 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_rgb)
axes[0].set_title('Original')

axes[1].imshow(cv2.cvtColor(avg_blur, cv2.COLOR_BGR2RGB))
axes[1].set_title('Average Blur')

axes[2].imshow(cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2RGB))
axes[2].set_title('Gaussian Blur')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 4. 중앙값 블러 - medianBlur()

### 중앙값 필터란?

중앙값 필터는 커널 영역의 중앙값(median)을 사용합니다. Salt-and-pepper 노이즈 제거에 매우 효과적입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Median Filter Operation                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input region:                                                 │
│   ┌────┬────┬────┐                                             │
│   │ 10 │ 20 │ 30 │                                             │
│   ├────┼────┼────┤                                             │
│   │ 40 │255 │ 60 │   ← Center 255 is noise (salt)              │
│   ├────┼────┼────┤                                             │
│   │ 70 │ 80 │ 90 │                                             │
│   └────┴────┴────┘                                             │
│                                                                 │
│   Sort values: 10, 20, 30, 40, 60, 70, 80, 90, 255             │
│   Median: 60 (5th value)                                        │
│                                                                 │
│   Result:                                                       │
│   ┌────┬────┬────┐                                             │
│   │    │    │    │                                             │
│   ├────┼────┼────┤                                             │
│   │    │ 60 │    │   ← Noise removed                           │
│   ├────┼────┼────┤                                             │
│   │    │    │    │                                             │
│   └────┴────┴────┘                                             │
│                                                                 │
│   Features:                                                     │
│   - Very effective for salt-and-pepper noise                   │
│   - Preserves edges relatively well                            │
│   - Slower than average/Gaussian                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 사용법

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Add salt-and-pepper noise (for testing)
def add_salt_pepper_noise(img, amount=0.05):
    noisy = img.copy()
    h, w = img.shape[:2]
    num_pixels = int(amount * h * w)

    # Salt (white)
    for _ in range(num_pixels):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        noisy[y, x] = 255

    # Pepper (black)
    for _ in range(num_pixels):
        y, x = np.random.randint(0, h), np.random.randint(0, w)
        noisy[y, x] = 0

    return noisy


noisy_img = add_salt_pepper_noise(img, 0.02)

# medianBlur(src, ksize)
# ksize: Only odd numbers allowed (3, 5, 7, ...)
median_3 = cv2.medianBlur(noisy_img, 3)
median_5 = cv2.medianBlur(noisy_img, 5)

# Compare: average blur, Gaussian blur
avg_blur = cv2.blur(noisy_img, (5, 5))
gauss_blur = cv2.GaussianBlur(noisy_img, (5, 5), 0)

cv2.imshow('Noisy', noisy_img)
cv2.imshow('Average Blur', avg_blur)
cv2.imshow('Gaussian Blur', gauss_blur)
cv2.imshow('Median Blur', median_5)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 5. 양방향 필터 - bilateralFilter()

### 양방향 필터란?

양방향 필터(Bilateral Filter)는 엣지를 보존하면서 스무딩하는 필터입니다. 피부 보정, 그림 효과 등에 사용됩니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Bilateral Filter Principle                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Regular Gaussian filter:                                      │
│   - Only considers distance → edges also blurred                │
│                                                                 │
│   Bilateral filter:                                             │
│   - Considers both distance (spatial) + color difference        │
│   - Only includes similar-colored pixels in average             │
│   - Preserves edges (where color difference is large)           │
│                                                                 │
│   Example:                                                      │
│   ┌─────────────────────────────────────────┐                   │
│   │ 100  100  100 │ 200  200  200 │          │                   │
│   │ 100  100  100 │ 200  200  200 │  ← Edge  │                   │
│   │ 100  100  100 │ 200  200  200 │          │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
│   Gaussian: 100 and 200 mix to around 150                       │
│   Bilateral: 100 area stays 100, 200 area stays 200             │
│                                                                 │
│   Weight = spatial Gaussian × color Gaussian                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 사용법

```python
import cv2

img = cv2.imread('portrait.jpg')

# bilateralFilter(src, d, sigmaColor, sigmaSpace)
# d: Filter size (-1 = auto-calculate from sigmaSpace)
# sigmaColor: Sigma in color space (higher = average wider color range)
# sigmaSpace: Sigma in coordinate space (higher = consider wider area)

# Weak effect
bilateral_weak = cv2.bilateralFilter(img, 9, 50, 50)

# Medium effect
bilateral_medium = cv2.bilateralFilter(img, 9, 75, 75)

# Strong effect (painting-like)
bilateral_strong = cv2.bilateralFilter(img, 15, 100, 100)

# Very strong effect
bilateral_extreme = cv2.bilateralFilter(img, 15, 150, 150)
```

### 피부 스무딩 예제

```python
import cv2
import numpy as np

def skin_smoothing(img, strength='medium'):
    """Skin smoothing effect"""
    params = {
        'weak': (5, 30, 30),
        'medium': (9, 75, 75),
        'strong': (15, 100, 100),
        'extreme': (20, 150, 150)
    }

    d, sigmaColor, sigmaSpace = params.get(strength, params['medium'])

    # Apply bilateral filter
    smooth = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

    # Blend with original (natural effect)
    alpha = 0.7  # Blending ratio
    result = cv2.addWeighted(smooth, alpha, img, 1 - alpha, 0)

    return result


img = cv2.imread('portrait.jpg')
result = skin_smoothing(img, 'medium')

cv2.imshow('Original', img)
cv2.imshow('Smoothed', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 블러 필터 비교

```python
import cv2
import time
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

# Compare processing time
filters = []

start = time.time()
avg = cv2.blur(img, (9, 9))
filters.append(('Average', avg, time.time() - start))

start = time.time()
gauss = cv2.GaussianBlur(img, (9, 9), 0)
filters.append(('Gaussian', gauss, time.time() - start))

start = time.time()
median = cv2.medianBlur(img, 9)
filters.append(('Median', median, time.time() - start))

start = time.time()
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
filters.append(('Bilateral', bilateral, time.time() - start))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for ax, (name, result, elapsed) in zip(axes, filters):
    ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax.set_title(f'{name} ({elapsed*1000:.1f}ms)')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 6. 커스텀 필터 - filter2D()

### filter2D() 사용법

`filter2D()`를 사용하면 직접 정의한 커널로 컨볼루션을 수행할 수 있습니다.

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# filter2D(src, ddepth, kernel)
# ddepth: Output image depth (-1 = same as input)
# kernel: User-defined kernel

# Create and apply average filter manually
kernel_avg = np.ones((5, 5), np.float32) / 25
avg_custom = cv2.filter2D(img, -1, kernel_avg)

# Same result as blur()
avg_builtin = cv2.blur(img, (5, 5))
print(f"Difference: {np.sum(np.abs(avg_custom - avg_builtin))}")  # 0
```

### 다양한 커스텀 커널

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 1. Emboss effect
kernel_emboss = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2]
])
emboss = cv2.filter2D(img, -1, kernel_emboss) + 128

# 2. Edge detection (Laplacian)
kernel_laplacian = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
])
laplacian = cv2.filter2D(img, -1, kernel_laplacian)

# 3. Sobel X (vertical edges)
kernel_sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
sobel_x = cv2.filter2D(img, -1, kernel_sobel_x)

# 4. Sobel Y (horizontal edges)
kernel_sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])
sobel_y = cv2.filter2D(img, -1, kernel_sobel_y)
```

### 커널 시각화 도구

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_and_show_kernel(img, kernel, title):
    """Visualize kernel application result and kernel"""
    result = cv2.filter2D(img, -1, kernel)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Kernel visualization
    im = axes[1].imshow(kernel, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1].set_title(f'Kernel ({kernel.shape[0]}x{kernel.shape[1]})')
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            axes[1].text(j, i, f'{kernel[i,j]:.1f}',
                        ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=axes[1])

    # Result
    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2].set_title(title)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


img = cv2.imread('image.jpg')

# Example: Emboss kernel
kernel_emboss = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2]
], dtype=np.float32)

apply_and_show_kernel(img, kernel_emboss, 'Emboss')
```

---

## 7. 샤프닝 필터

### 샤프닝 원리

```
┌─────────────────────────────────────────────────────────────────┐
│                      Sharpening Principle                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Sharpening = Original + (Original - Blur)                     │
│              = Original + High-frequency component              │
│              = Edge enhancement                                 │
│                                                                 │
│   Or directly with kernel:                                      │
│                                                                 │
│   Basic sharpening kernel:                                      │
│   ┌────┬────┬────┐                                             │
│   │  0 │ -1 │  0 │                                             │
│   ├────┼────┼────┤                                             │
│   │ -1 │  5 │ -1 │   Center = 5 (original weight)              │
│   ├────┼────┼────┤   Surrounding = -1 (subtract blur)          │
│   │  0 │ -1 │  0 │   Sum = 1 (preserve brightness)             │
│   └────┴────┴────┘                                             │
│                                                                 │
│   Strong sharpening kernel:                                     │
│   ┌────┬────┬────┐                                             │
│   │ -1 │ -1 │ -1 │                                             │
│   ├────┼────┼────┤                                             │
│   │ -1 │  9 │ -1 │   Center = 9                                │
│   ├────┼────┼────┤   Surrounding = -1 × 8 = -8                │
│   │ -1 │ -1 │ -1 │   Sum = 1                                   │
│   └────┴────┴────┘                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 샤프닝 구현

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Method 1: Using kernel
kernel_sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
sharpened1 = cv2.filter2D(img, -1, kernel_sharpen)

# Method 2: Strong sharpening kernel
kernel_sharpen_strong = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])
sharpened2 = cv2.filter2D(img, -1, kernel_sharpen_strong)

# Method 3: Unsharp Masking
def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Sharpening with unsharp masking

    amount: Sharpening strength (1.0 = standard)
    threshold: Edge detection threshold (noise prevention)
    """
    # Why Gaussian blur here: isolates low-frequency content; subtracting it leaves only
    # high-frequency detail (edges, texture) which we then amplify
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)

    # Original - Blur = Edges/Details
    # sharpened = Original + amount × (Original - Blur)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

    if threshold > 0:
        # Why threshold: prevents amplifying flat-region noise — only sharpen where
        # there is already a meaningful intensity difference between original and blur
        diff = cv2.absdiff(img, blurred)
        mask = (diff < threshold).astype(np.uint8) * 255
        sharpened = np.where(mask == 255, img, sharpened)

    return sharpened


sharpened3 = unsharp_mask(img, amount=1.5)
```

### 적응형 샤프닝

```python
import cv2
import numpy as np

def adaptive_sharpening(img, amount=1.0):
    """
    Adaptive sharpening - apply sharpening only to edge regions
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    # Why dilate edges: the 1-pixel Canny edge is too narrow; dilation creates a soft
    # transition zone so sharpening doesn't produce hard halos at region boundaries
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Why blur before sharpening: we need the low-frequency baseline to subtract from
    blurred = cv2.GaussianBlur(img, (5, 5), 1)

    # Sharpening
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

    # Why blend instead of hard mask: keeps flat areas completely unchanged while
    # concentrating sharpening where edges already exist, avoiding noise amplification
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) / 255.0
    result = (sharpened * edges_3ch + img * (1 - edges_3ch)).astype(np.uint8)

    return result


img = cv2.imread('image.jpg')
result = adaptive_sharpening(img, amount=2.0)
```

---

## 8. 연습 문제

### 연습 1: 노이즈 제거 비교

가우시안 노이즈와 Salt-and-pepper 노이즈를 각각 생성하고, 세 가지 블러 필터(평균, 가우시안, 중앙값)로 제거 효과를 비교하세요. PSNR 값으로 정량적 비교도 수행하세요.

```python
# Hint: Add Gaussian noise
def add_gaussian_noise(img, mean=0, var=100):
    noise = np.random.normal(mean, var**0.5, img.shape)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy
```

### 연습 2: 실시간 블러 강도 조절

웹캠 영상에 트랙바로 블러 강도(커널 크기)를 조절할 수 있는 프로그램을 작성하세요. 가우시안 블러와 양방향 필터 중 선택할 수 있게 하세요.

### 연습 3: 커스텀 엠보스 방향

8방향(상, 하, 좌, 우, 대각선 4방향)으로 다른 엠보스 효과를 내는 커널들을 설계하고 테스트하세요.

### 연습 4: 고급 샤프닝

다음 기능을 가진 고급 샤프닝 함수를 구현하세요:
1. 샤프닝 강도 조절 (amount)
2. 블러 반경 조절 (radius)
3. 임계값 적용 (threshold) - 작은 변화는 무시
4. 하이라이트/섀도우 별도 처리

### 연습 5: 미니어처 효과 (틸트 시프트)

가우시안 블러와 마스크를 사용하여 틸트 시프트(tilt-shift) 미니어처 효과를 구현하세요. 이미지 중앙 부분은 선명하게, 위아래는 점진적으로 블러 처리합니다.

```python
# Hint
def tilt_shift(img, focus_y, focus_height, blur_amount):
    # Create gradient mask
    # Blend blurred and original images using mask
    pass
```

---

## 9. 다음 단계

[모폴로지 연산](./06_Morphology.md)에서 침식, 팽창, 열기/닫기 등 형태학적 연산을 학습합니다!

**다음에 배울 내용**:
- 구조 요소 (Structuring Element)
- 침식 (Erosion)과 팽창 (Dilation)
- 열기 (Opening)와 닫기 (Closing)
- 노이즈 제거와 객체 분리

---

## 10. 참고 자료

### 공식 문서

- [blur() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37)
- [GaussianBlur() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1)
- [medianBlur() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9)
- [bilateralFilter() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed)

### 관련 학습 자료

| 폴더 | 관련 내용 |
|------|----------|
| [기하학적 변환](./04_Geometric_Transforms.md) | 이미지 전처리 |
| [엣지 검출 (Edge Detection)](./08_Edge_Detection.md) | 필터링 후 엣지 검출 |

### 추가 참고

- [이미지 필터링 이론](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
- [컨볼루션 시각화](https://setosa.io/ev/image-kernels/)

