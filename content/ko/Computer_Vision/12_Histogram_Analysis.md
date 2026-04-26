# 히스토그램 분석 (Histogram Analysis)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 이미지 히스토그램(image histogram)이 무엇이며 픽셀 밝기 분포를 어떻게 나타내는지 설명할 수 있다
2. OpenCV를 사용하여 그레이스케일(grayscale) 및 다채널 이미지의 히스토그램을 계산하고 시각화할 수 있다
3. 히스토그램 균등화(histogram equalization)와 CLAHE를 구현하여 이미지 대비(contrast)를 향상시킬 수 있다
4. 다양한 거리 메트릭(distance metrics)으로 히스토그램을 비교하여 이미지 유사도를 측정할 수 있다
5. 색상 분포(color distribution)를 기반으로 관심 영역(region of interest)을 찾기 위해 히스토그램 역투영(histogram backprojection)을 적용할 수 있다
6. 히스토그램 형태를 분석하여 과노출(overexposure)이나 낮은 대비 등 이미지 품질 문제를 진단할 수 있다

---

## 개요

히스토그램은 이미지의 밝기 분포를 나타내는 그래프입니다. 이미지 분석, 대비 향상, 색상 비교 등에 활용됩니다. 이 레슨에서는 히스토그램 계산, 균등화, CLAHE, 비교, 역투영 등을 학습합니다.

---

## 목차

OpenCV 함수 참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 히스토그램을 밝기 확률 분포의 추정치로 보는 관점, CDF 기반 히스토그램 균등화의 유도, 그리고 CLAHE가 전역 균등화의 국소 대비 실패를 어떻게 해결하는지를 다룹니다.

1. [히스토그램 기초](#1-히스토그램-기초)
2. [히스토그램 계산](#2-히스토그램-계산)
3. [히스토그램 균등화](#3-히스토그램-균등화)
4. [CLAHE](#4-clahe)
5. [히스토그램 비교](#5-히스토그램-비교)
6. [역투영](#6-역투영)
7. [연습 문제](#7-연습-문제)

---

## 이론과 원리

이미지 히스토그램은 이미지 내 밝기의 **확률 분포**에 대한 이산 추정치입니다. 히스토그램으로 하는 모든 유용한 작업 — 분석, 균등화, 비교, 역투영 — 은 그 분포에 대한 연산입니다. 확률적 기초를 이해하면 각 작업에 맞는 도구를 고를 수 있고, 각 알고리즘이 내 이미지에 무슨 일을 할지 예측할 수 있습니다.

이 섹션은 다음을 다룹니다:

- **(A) 확률 분포로서의 히스토그램** — 정의, 정규화, 이미지 대비 잃어버리는 것.
- **(B) 히스토그램 균등화** — CDF로부터의 유도, 모든 분포를 균등 분포로 매핑하는 이유, 실패 양상.
- **(C) CLAHE** — 대비 제한 적응형 히스토그램 균등화, "적응형"과 "대비 제한"이 실제 의미하는 것.
- **(D) 히스토그램 비교** — 흔한 거리 메트릭과 각각이 측정하는 것.
- **(E) 역투영(Backprojection)** — 히스토그램을 객체 위치 추정용 우도 맵으로 전환.

### A. 확률 분포로서의 히스토그램

`N` 픽셀과 밝기 범위 `[0, L-1]`(보통 `L = 256`)을 가진 이미지 `I`에서, 원시 히스토그램 `h(r)`은 각 밝기의 픽셀 수를 셉니다:

```
h(r) = |{ (x, y) : I(x, y) = r }|       for r = 0, 1, ..., L-1
```

**정규화된 히스토그램** `p(r) = h(r) / N`은 올바른 확률 질량 함수입니다:

```
Σ_r  p(r) = 1
```

"무작위로 뽑은 픽셀의 밝기가 `r`일 확률"로 해석할 수 있습니다. 히스토그램은 모든 공간 정보를 버립니다 — 픽셀의 공간적 재배열은 같은 히스토그램을 낳습니다 — 따라서 이미지의 톤 특성만 포착할 뿐, 내용은 포착하지 않습니다. 이것이 강점(공간 이동에 대한 불변성)이자 한계(비슷한 톤 분포를 가진 두 서로 다른 장면을 구별할 수 없음)입니다.

#### A.1 히스토그램 형태가 알려주는 것

히스토그램 형태를 읽으면 노출 문제를 한눈에 진단할 수 있습니다:

- 대부분 질량이 왼쪽에 몰림 → **노출 부족**(너무 어두움).
- 대부분 질량이 오른쪽에 몰림 → **노출 과다**(너무 밝음).
- 0 또는 255에서 질량이 잘림 → **하드 클리핑**, 디테일이 복구 불가능하게 손실.
- 넓고 잘 퍼진 질량이 `[0, L-1]` 전체 범위를 사용 → **좋은 동적 범위**.
- 타이트하고 좁은 질량 → **낮은 대비**, 이미지가 톤 범위의 일부만 사용.

모든 히스토그램 향상 알고리즘은 이 증상 중 하나를 겨냥합니다.

### B. 히스토그램 균등화

**목표**: 출력 히스토그램이 (근사적으로) **균등** 분포가 되도록 픽셀 밝기를 재분배 — 모든 밝기 레벨이 동등하게 자주 쓰이도록. 균등 분포는 유계 범위에서 최대 엔트로피 분포이며, 이는 "픽셀당 최대 정보"에 대응하고, 실전에서는 최대 대비에 해당합니다.

#### B.1 유도

`r`을 원래 밝기(pdf `p_r(r)`를 가진 확률 변수), `s`를 균등화된 밝기라 합시다. 어떤 변환 `T`에 대해 `s = T(r)`이고 `p_s(s)`가 `[0, L-1]`에서 균등이 되기를 원합니다. 확률론의 한 정리는, `T(r)`이 `r`의 **누적분포함수**(CDF)와 같다면(`[0, L-1]`로 스케일됨), `T(r)`이 균등 분포를 가진다고 말합니다:

```
s = T(r) = (L - 1) · ∫₀ʳ p_r(w) dw
```

히스토그램 `h(r)`을 가진 이산 경우:

```
s_k = round( (L - 1) · (1/N) · Σᵢ₌₀ᵏ  h(i) )  =  round( (L - 1) · CDF(k) )
```

즉 균등화 절차는 세 단계입니다:

1. 히스토그램 계산.
2. CDF 계산(누적 합, 그 다음 `CDF(L-1) = 1`이 되도록 스케일).
3. `(L - 1) · CDF`를 룩업 테이블로 사용해 각 픽셀을 재매핑.

#### B.2 작동 이유와 실패 경우

CDF가 비감소이므로 변환은 순서 보존 — 균등화 후에도 픽셀 A가 B보다 밝으면 여전히 더 밝습니다. CDF가 밀도 높은 영역을 늘리고 밀도 낮은 영역을 압축하므로, 인구가 많은 톤 범위가 퍼지고 희소한 범위가 합쳐집니다. 픽셀이 가장 많은 히스토그램 부분의 대비가 향상됩니다.

**실패 양상**:

- **증폭된 잡음.** 평탄 영역에는 낮은 진폭의 센서 잡음이 있는데, 균등화가 그 잡음을 가시적 패턴으로 늘릴 수 있습니다.
- **전역 변환만.** 단일 룩업 테이블이 모든 픽셀에 적용되므로, 국소 대비 문제는 무시됩니다. 이미지의 일부가 과노출이고 다른 부분이 노출 부족이면, 균등화는 둘 다 해결하지 못하고 — 평균을 최적화할 수밖에 없습니다.
- **절대 밝기 파괴.** 특정 밝기 값이 중요한 작업(색상학, 측정)에서는 균등화가 그 정보를 파괴합니다.

### C. CLAHE: 대비 제한 적응형 히스토그램 균등화

CLAHE는 전역 균등화의 두 주요 실패를 모두 해결합니다:

#### C.1 "적응형" — 국소 히스토그램

이미지를 **타일** 격자(보통 8×8)로 나눕니다. 각 타일마다 별도의 히스토그램과 균등화 LUT를 계산합니다. 각 출력 픽셀에 대해 그 타일의 LUT를 사용 — **다만 타일 경계에서 가시적인 경계가 생기지 않도록 네 인접 타일 LUT들 사이를 bilinear 보간**합니다. 이렇게 하면 국소 대비가 각 영역마다 별도로 최적화됩니다. 어두운 구석은 자기만의 스트레치, 밝은 중앙은 자기만의 스트레치.

#### C.2 "대비 제한" — 히스토그램 클리핑

각 타일의 히스토그램 CDF를 계산하기 전에 임계값(`clipLimit` 파라미터)을 초과하는 빈을 **클립**하고, 클리핑된 초과분을 모든 빈에 균등 재분배합니다. 이게 중요한 이유: 거의 평탄한 영역에서는 히스토그램이 매우 큰 스파이크를 가지는데, 이것이 매우 가파른 국소 변환을 만들어 잡음을 극적으로 증폭하기 때문입니다. 스파이크를 클립하면 CDF 기울기가 유계가 되어 잡음 증폭도 유계가 됩니다.

수식으로, 원본 히스토그램이 총합 `N_tile`의 `h(r)`이면, 클립된 버전은

```
h_clipped(r) = min(h(r), clipLimit · N_tile / L)
excess       = Σ max(0, h(r) - clipLimit · N_tile / L)
h_final(r)   = h_clipped(r) + excess / L
```

그 다음 CDF와 LUT를 `h_final`에서 계산. 전형적 `clipLimit = 2.0–4.0`.

결과: 평범한 균등화의 특징적 잡음 증폭 아티팩트 없이 강한 국소 대비 향상. CLAHE는 의료 영상, 번호판, 그리고 전역 톤보다 국소 디테일이 중요한 모든 콘텐츠의 전처리 기본 선택입니다.

### D. 히스토그램 비교

두 정규화 히스토그램 `p`, `q`가 있으면 거리 또는 유사도를 계산해 이미지들을 비교할 수 있습니다. 서로 다른 메트릭이 서로 다른 질문에 답합니다:

- **상관(Correlation)** (`CV_COMP_CORREL`). 두 히스토그램 사이의 Pearson 상관계수. 범위 `[-1, 1]`, 1 = 완벽한 일치, 0 = 무상관. 스케일에 불변인 형태 유사도를 측정.

  ```
  d(p, q) = Σ (p(i) - p̄)(q(i) - q̄)  /  √(Σ(p(i) - p̄)² · Σ(q(i) - q̄)²)
  ```

- **카이제곱(Chi-square)** (`CV_COMP_CHISQR`). 비대칭 거리 — 기준 `p`에 질량이 있는데 시험 `q`에는 없는 빈에 페널티. 작을수록 더 유사. `p`가 템플릿일 때 유용.

  ```
  d(p, q) = Σ (p(i) - q(i))² / p(i)
  ```

- **교집합(Intersection)** (`CV_COMP_INTERSECT`). 빈당 공유 질량의 최솟값. 클수록 더 유사.

  ```
  d(p, q) = Σ min(p(i), q(i))
  ```

- **바타차야(Bhattacharyya) 거리** (`CV_COMP_BHATTACHARYYA`). 두 확률 분포의 겹침을 측정. 0 = 동일, 1 = 분리.

  ```
  d(p, q) = √(1 - (1/√(p̄ · q̄ · N²)) · Σ √(p(i) · q(i)))
  ```

범용 형태 매칭에는 상관, 알려진 템플릿과 비교할 때는 카이제곱, 빠른 근사 매칭에는 교집합, 원칙적 확률적 거리에는 바타차야를 선택하세요.

### E. 역투영: 우도 맵으로서의 히스토그램

순방향: 이미지가 히스토그램을 만듭니다. **역투영은 이를 뒤집습니다**: 기준 히스토그램(예: 찾으려는 객체의 색상 히스토그램)이 주어지면, 각 픽셀 값이 그 분포 아래에서 얼마나 가능성 있는지의 픽셀별 맵을 만듭니다.

절차:

1. 목표를 포함하는 샘플 영역(예: 객체의 피부색)으로부터 기준 히스토그램 `p(r)` 구축.
2. 질의 이미지의 각 픽셀에 대해 `p(I(x, y))`를 찾아봄 — 이 값이 "이 픽셀이 목표 분포에서 나왔을 확률".
3. 출력은 확률 맵. 보통 임계처리되거나 mean-shift 추적에 넘겨져 객체를 국소화.

이것이 색상 기반 객체 추적의 핵심(예: CAMShift 추적기). 보통 HSV 색공간의 2D `(H, S)` 히스토그램을 써서 조명을 상쇄(§03.C). 결과: 밝은 영역 = 가능성 있는 목표 위치인 히트맵.

### 이론에서 아래 함수들로

- `cv2.calcHist(images, channels, mask, histSize, ranges)` — 이미지 또는 ROI에 대한 `h(r)` 계산(§A). `mask`로 하위 영역에 계산을 제한 가능.
- `cv2.equalizeHist(img)` — §B 적용. 그레이스케일 이미지만. 컬러의 경우 HSV 또는 YCrCb로 변환 후 `V` 또는 `Y` 채널에 적용.
- `cv2.createCLAHE(clipLimit, tileGridSize).apply(img)` — §C 적용. 두 파라미터가 §C.1과 §C.2에 직접 대응.
- `cv2.compareHist(hist1, hist2, method)` — 히스토그램 비교(§D). `method` 플래그가 메트릭 선택.
- `cv2.calcBackProject(images, channels, hist, ranges, scale)` — 역투영(§E), 픽셀별 우도 맵 생성.
- `cv2.normalize(hist, ..., norm_type=cv2.NORM_MINMAX)` — 흔한 전처리: 비교 전에 히스토그램을 `[0, 1]` 또는 다른 범위로 스케일.

---

## 1. 히스토그램 기초

### 히스토그램이란?

```
Histogram:
A graph representing the distribution of pixel brightness values in an image

X-axis: Brightness value (0-255)
Y-axis: Number of pixels with that brightness value

Dark Image                Bright Image            High Contrast Image
    │                        │                      │
Freq│█                       │       █              │   █   █
uenc│██                      │      ██              │  ███ ███
y   │███                     │     ███              │ █████████
    └────────────           └────────────          └────────────
    0          255          0          255         0          255
     Brightness               Brightness              Brightness
```

히스토그램은 전체 이미지를 밝기 분포의 간결한 통계적 요약으로 축소하며, 공간적 배치와는 독립적입니다. 이로 인해 이미지의 강력한 "지문(fingerprint)" 역할을 합니다. 동일한 장면을 같은 조명 아래서 촬영한 두 사진은, 약간 이동하거나 회전되어 있더라도 매우 유사한 히스토그램을 공유합니다. 반면 전혀 다른 장면은 뚜렷하게 다른 히스토그램 모양을 가집니다. 이 특성 덕분에 히스토그램은 빠른 이미지 검색, 노출 진단, 색상 기반 객체 추적에 유용합니다.

### 히스토그램의 활용

```
1. Image Analysis
   - Check exposure status (overexposed, underexposed)
   - Assess contrast level

2. Image Enhancement
   - Histogram equalization
   - Contrast adjustment

3. Image Comparison
   - Similar image search
   - Color-based matching

4. Object Tracking
   - Color histogram backprojection
   - CamShift/MeanShift algorithms
```

---

## 2. 히스토그램 계산

### cv2.calcHist() 함수

```python
hist = cv2.calcHist(images, channels, mask, histSize, ranges)
```

| 파라미터 | 설명 |
|----------|------|
| images | 입력 이미지 리스트 [img] |
| channels | 채널 인덱스 [0], [1], [2] 또는 [0, 1] 등 |
| mask | 마스크 (None = 전체 이미지) |
| histSize | 빈(bin) 개수 [256] |
| ranges | 값 범위 [0, 256] |

### 그레이스케일 히스토그램

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calc_gray_histogram(image_path):
    """Calculate and visualize grayscale histogram"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate histogram
    hist = cv2.calcHist(
        [img],           # Image (passed as list)
        [0],             # Channel (0 for grayscale)
        None,            # Mask (entire image)
        [256],           # 256 bins — one per possible 8-bit intensity level (0–255).
                         # This gives maximum precision. Fewer bins (e.g., 64) would
                         # merge adjacent intensities, speeding up comparison at the
                         # cost of discriminative power; useful for retrieval at scale.
        [0, 256]         # Value range (upper bound is exclusive, so this covers 0–255)
    )

    # Visualize with Matplotlib
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.plot(hist, color='black')
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

    return hist

hist = calc_gray_histogram('image.jpg')
```

### 컬러 히스토그램

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calc_color_histogram(image_path):
    """RGB channel-wise histogram"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    colors = ('r', 'g', 'b')
    channel_names = ('Red', 'Green', 'Blue')

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        # BGR order, so adjust index: R=2, G=1, B=0
        channel_idx = 2 - i
        hist = cv2.calcHist([img], [channel_idx], None, [256], [0, 256])
        plt.plot(hist, color=color, label=name)

    plt.title('Color Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.legend()

    plt.tight_layout()
    plt.show()

calc_color_histogram('colorful.jpg')
```

### 2D 히스토그램 (Hue-Saturation)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calc_2d_histogram(image_path):
    """Hue-Saturation 2D histogram"""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # H: 0-180, S: 0-256
    hist = cv2.calcHist(
        [hsv],
        [0, 1],          # H and S channels
        None,
        [30, 32],        # Number of bins (H: 30, S: 32)
        [0, 180, 0, 256] # Ranges (H: 0-180, S: 0-256)
    )

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(hist, interpolation='nearest')
    plt.title('2D Histogram (H-S)')
    plt.xlabel('Saturation')
    plt.ylabel('Hue')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    return hist

hist_2d = calc_2d_histogram('colorful.jpg')
```

### 마스크를 사용한 히스토그램

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_with_mask(image_path):
    """Calculate histogram for specific region only"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    # Create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), min(h, w)//3, 255, -1)

    # Full histogram
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Masked region histogram
    hist_masked = cv2.calcHist([img], [0], mask, [256], [0, 256])

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.plot(hist_full, label='Full', alpha=0.7)
    plt.plot(hist_masked, label='Masked', alpha=0.7)
    plt.legend()
    plt.title('Histograms')

    plt.tight_layout()
    plt.show()

histogram_with_mask('image.jpg')
```

---

## 3. 히스토그램 균등화

### 개념

```
Histogram Equalization:
Makes the brightness distribution of an image uniform to enhance contrast

Original Histogram            Equalized Histogram
    │                              │
    │█                             │   █ █ █
    │███                           │ █ █ █ █ █
    │█████                         │█ █ █ █ █ █ █
    └────────────                  └────────────────
    0          255                 0              255

Transformation Process:
1. Calculate histogram
2. Calculate cumulative distribution function (CDF)
3. Normalize CDF
4. Map pixel values
```

CDF 매핑은 간단한 기하학적 원리로 작동합니다. 픽셀의 누적 개수가 0~255 전체에 고르게 분포하도록 이미지의 밝기 범위를 늘리면, 출력 히스토그램이 최대한 평탄(균일)해집니다. 수학적으로 매핑은 `out = round(CDF(in) × 255)`입니다. 많은 픽셀이 집중된 밝기값(히스토그램의 높은 스파이크)은 넓은 출력 범위로 펼쳐져, 이전에는 구분할 수 없었던 회색 레벨들을 분리하여 숨겨진 디테일을 드러냅니다.

### cv2.equalizeHist()

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize_histogram_demo(image_path):
    """Histogram equalization demo"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Histogram equalization
    equalized = cv2.equalizeHist(img)

    # Calculate histograms
    hist_before = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_after = cv2.calcHist([equalized], [0], None, [256], [0, 256])

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(equalized, cmap='gray')
    axes[0, 1].set_title('Equalized')
    axes[0, 1].axis('off')

    axes[1, 0].plot(hist_before)
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlim([0, 256])

    axes[1, 1].plot(hist_after)
    axes[1, 1].set_title('Equalized Histogram')
    axes[1, 1].set_xlim([0, 256])

    plt.tight_layout()
    plt.show()

    return equalized

equalized = equalize_histogram_demo('dark_image.jpg')
```

### 컬러 이미지 균등화

```python
import cv2
import numpy as np

def equalize_color_image(image_path):
    """Histogram equalization for color images"""
    img = cv2.imread(image_path)

    # Method 1: Use YCrCb color space (recommended)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])  # Equalize Y channel only
    result_ycrcb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Method 2: Use HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])  # Equalize V channel only
    result_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Method 3: Equalize each channel individually (may cause color distortion)
    b, g, r = cv2.split(img)
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    result_rgb = cv2.merge([b_eq, g_eq, r_eq])

    cv2.imshow('Original', img)
    cv2.imshow('YCrCb Equalization', result_ycrcb)
    cv2.imshow('HSV Equalization', result_hsv)
    cv2.imshow('RGB Equalization', result_rgb)
    cv2.waitKey(0)

    return result_ycrcb

equalize_color_image('dark_color.jpg')
```

---

## 4. CLAHE

### 개념

```
CLAHE (Contrast Limited Adaptive Histogram Equalization):
Adaptive histogram equalization

Problem: Global equalization can amplify noise
Solution: Divide image into tiles and equalize locally

┌────┬────┬────┬────┐
│    │    │    │    │
│ T1 │ T2 │ T3 │ T4 │   Apply equalization
├────┼────┼────┼────┤   to each tile
│    │    │    │    │
│ T5 │ T6 │ T7 │ T8 │   Smooth boundaries
├────┼────┼────┼────┤   with interpolation
│ T9 │T10 │T11 │T12 │
└────┴────┴────┴────┘

Features:
- clipLimit: Contrast limit (higher = stronger contrast)
- tileGridSize: Tile size (smaller = more detailed)
```

### cv2.createCLAHE()

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def clahe_demo(image_path):
    """CLAHE application demo"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Standard equalization
    equalized = cv2.equalizeHist(img)

    # Create and apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,      # Contrast amplification cap per tile. When any histogram
                            # bin would exceed this limit after equalization, the excess
                            # votes are redistributed uniformly — this prevents runaway
                            # noise amplification in flat (low-texture) regions.
                            # 2.0 is a conservative default; raise to 4–8 for very dark
                            # medical images, but expect more noise at high values.
        tileGridSize=(8, 8) # Divides the image into an 8×8 grid of tiles; equalization
                            # is applied independently within each tile, then boundaries
                            # are blended with bilinear interpolation. Smaller tiles
                            # (e.g., 4×4) enhance local detail more aggressively;
                            # larger tiles (16×16) behave closer to global equalization.
    )
    clahe_result = clahe.apply(img)

    # Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(equalized, cmap='gray')
    axes[1].set_title('Standard Equalization')
    axes[1].axis('off')

    axes[2].imshow(clahe_result, cmap='gray')
    axes[2].set_title('CLAHE')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return clahe_result

clahe_demo('low_contrast.jpg')
```

### CLAHE 파라미터 비교

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_clahe_params(image_path):
    """Compare CLAHE with different parameters"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    clip_limits = [1.0, 2.0, 4.0, 8.0]
    tile_sizes = [(4, 4), (8, 8), (16, 16)]

    fig, axes = plt.subplots(len(tile_sizes), len(clip_limits) + 1,
                              figsize=(15, 10))

    for i, tile_size in enumerate(tile_sizes):
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Original\nTile: {tile_size}')
        axes[i, 0].axis('off')

        for j, clip_limit in enumerate(clip_limits):
            clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                     tileGridSize=tile_size)
            result = clahe.apply(img)

            axes[i, j + 1].imshow(result, cmap='gray')
            axes[i, j + 1].set_title(f'clip={clip_limit}')
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.show()

compare_clahe_params('low_contrast.jpg')
```

### 컬러 이미지에 CLAHE 적용

```python
import cv2
import numpy as np

def clahe_color(image_path, clip_limit=2.0, tile_size=(8, 8)):
    """Apply CLAHE to color image"""
    img = cv2.imread(image_path)

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to BGR
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    cv2.imshow('Original', img)
    cv2.imshow('CLAHE', result)
    cv2.waitKey(0)

    return result

clahe_color('dark_scene.jpg')
```

---

## 5. 히스토그램 비교

### cv2.compareHist()

```python
similarity = cv2.compareHist(hist1, hist2, method)
```

| 방법 | 설명 | 범위 | 해석 |
|------|------|------|------|
| cv2.HISTCMP_CORREL | 상관관계 | -1 ~ 1 | 1: 완전 일치 |
| cv2.HISTCMP_CHISQR | 카이제곱 | 0 ~ ∞ | 0: 완전 일치 |
| cv2.HISTCMP_INTERSECT | 교차 | 0 ~ min(sum) | 높을수록 유사 |
| cv2.HISTCMP_BHATTACHARYYA | 바타차리아 거리 | 0 ~ 1 | 0: 완전 일치 |

### 히스토그램 비교 예제

```python
import cv2
import numpy as np

def compare_histograms(image_paths):
    """Compare histograms of multiple images"""
    # Base image
    base_img = cv2.imread(image_paths[0])
    base_hsv = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV)

    # Calculate histogram (H-S 2D)
    base_hist = cv2.calcHist(
        [base_hsv], [0, 1], None,
        [50, 60], [0, 180, 0, 256]
    )
    cv2.normalize(base_hist, base_hist, 0, 1, cv2.NORM_MINMAX)

    print(f"Base image: {image_paths[0]}")
    print("-" * 50)

    methods = [
        (cv2.HISTCMP_CORREL, 'Correlation'),
        (cv2.HISTCMP_CHISQR, 'Chi-Square'),
        (cv2.HISTCMP_INTERSECT, 'Intersection'),
        (cv2.HISTCMP_BHATTACHARYYA, 'Bhattacharyya')
    ]

    for path in image_paths[1:]:
        img = cv2.imread(path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [50, 60], [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        print(f"\nComparing: {path}")
        for method, name in methods:
            result = cv2.compareHist(base_hist, hist, method)
            print(f"  {name}: {result:.4f}")

# Usage example
image_files = ['ref.jpg', 'similar1.jpg', 'similar2.jpg', 'different.jpg']
compare_histograms(image_files)
```

### 유사 이미지 검색

```python
import cv2
import numpy as np
import os

def find_similar_images(query_path, search_dir, top_k=5):
    """Histogram-based similar image search"""
    # Query image histogram
    query = cv2.imread(query_path)
    query_hsv = cv2.cvtColor(query, cv2.COLOR_BGR2HSV)
    query_hist = cv2.calcHist([query_hsv], [0, 1], None,
                               [50, 60], [0, 180, 0, 256])
    cv2.normalize(query_hist, query_hist, 0, 1, cv2.NORM_MINMAX)

    results = []

    # Compare with all images in search directory
    for filename in os.listdir(search_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        filepath = os.path.join(search_dir, filename)
        img = cv2.imread(filepath)
        if img is None:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None,
                             [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        # Calculate correlation (higher = more similar)
        similarity = cv2.compareHist(query_hist, hist, cv2.HISTCMP_CORREL)
        results.append((filename, similarity))

    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"Query: {query_path}")
    print(f"\nTop {top_k} similar images:")
    for filename, sim in results[:top_k]:
        print(f"  {filename}: {sim:.4f}")

    return results[:top_k]

# Usage example
find_similar_images('query.jpg', './image_database/', top_k=5)
```

---

## 6. 역투영

### 개념

```
Backprojection:
Detect specific color regions using histograms

Process:
1. Calculate color histogram of object of interest (ROI)
2. Replace each pixel in the entire image with its histogram value
3. High value = similar to color of interest

Applications:
- Color-based object tracking
- Core of CamShift/MeanShift algorithms

Example:
┌─────────────┐       ┌─────────────┐
│   🟡 ROI    │       │ ■ ■ □ □ □ │
│  (Yellow)   │  ──▶  │ ■ ■ ■ □ □ │  High value = Yellow
│             │       │ □ ■ ■ ■ □ │
└─────────────┘       └─────────────┘
  Color Histogram      Backprojection Result
```

### cv2.calcBackProject()

```python
import cv2
import numpy as np

def backprojection_demo(image_path, roi_coords):
    """Backprojection demo"""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set ROI region
    x, y, w, h = roi_coords
    roi = hsv[y:y+h, x:x+w]

    # Calculate ROI histogram
    roi_hist = cv2.calcHist(
        [roi], [0, 1], None,
        [180, 256], [0, 180, 0, 256]
    )
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Backprojection
    backproj = cv2.calcBackProject(
        [hsv], [0, 1], roi_hist,
        [0, 180, 0, 256], 1
    )

    # Remove noise with filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(backproj, -1, kernel, backproj)
    _, backproj = cv2.threshold(backproj, 50, 255, cv2.THRESH_BINARY)

    # Visualization
    result = img.copy()
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mask detected region
    mask = cv2.merge([backproj, backproj, backproj])
    detected = cv2.bitwise_and(img, mask)

    cv2.imshow('Original with ROI', result)
    cv2.imshow('Back Projection', backproj)
    cv2.imshow('Detected', detected)
    cv2.waitKey(0)

    return backproj

# Usage example (x, y, width, height)
backprojection_demo('scene.jpg', (100, 100, 50, 50))
```

### 피부색 검출

```python
import cv2
import numpy as np

def detect_skin(image_path):
    """Skin color detection (using backprojection)"""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Skin color range (HSV)
    # H: 0-20, S: 48-255, V: 80-255 (typical skin color)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Skin color mask
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Generate histogram of skin region
    skin_region = cv2.bitwise_and(hsv, hsv, mask=skin_mask)
    skin_hist = cv2.calcHist([skin_region], [0, 1], skin_mask,
                              [180, 256], [0, 180, 0, 256])
    cv2.normalize(skin_hist, skin_hist, 0, 255, cv2.NORM_MINMAX)

    # Backprojection
    backproj = cv2.calcBackProject([hsv], [0, 1], skin_hist,
                                    [0, 180, 0, 256], 1)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    backproj = cv2.morphologyEx(backproj, cv2.MORPH_OPEN, kernel)
    backproj = cv2.morphologyEx(backproj, cv2.MORPH_CLOSE, kernel)

    # Result
    result = cv2.bitwise_and(img, img, mask=backproj)

    cv2.imshow('Original', img)
    cv2.imshow('Skin Mask', backproj)
    cv2.imshow('Detected Skin', result)
    cv2.waitKey(0)

    return backproj

detect_skin('person.jpg')
```

### CamShift를 이용한 객체 추적

```python
import cv2
import numpy as np

def camshift_tracking(video_path):
    """Object tracking using CamShift"""
    cap = cv2.VideoCapture(video_path)

    # Select ROI from first frame
    ret, frame = cap.read()
    if not ret:
        return

    # Select ROI (select with mouse or specify directly)
    roi = cv2.selectROI('Select ROI', frame, False)
    cv2.destroyWindow('Select ROI')

    x, y, w, h = roi
    track_window = (x, y, w, h)

    # Calculate ROI histogram
    roi_frame = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]),
                       np.array([180, 255, 255]))

    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # CamShift termination criteria
    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Backprojection
        backproj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply CamShift
        ret, track_window = cv2.CamShift(backproj, track_window, term_criteria)

        # Draw result (rotated rectangle)
        pts = cv2.boxPoints(ret)
        pts = np.int_(pts)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        cv2.imshow('CamShift Tracking', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# camshift_tracking('video.mp4')
```

---

## 7. 연습 문제

### 문제 1: 자동 대비 조정

이미지의 히스토그램을 분석하여 자동으로 최적의 대비 조정을 수행하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def auto_contrast(image):
    """Automatic contrast adjustment (histogram stretching)"""
    if len(image.shape) == 3:
        # Color image: LAB conversion
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Histogram stretching on L channel
        l_min = np.min(l)
        l_max = np.max(l)
        l_stretched = ((l - l_min) * 255 / (l_max - l_min)).astype(np.uint8)

        lab_stretched = cv2.merge([l_stretched, a, b])
        result = cv2.cvtColor(lab_stretched, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale
        img_min = np.min(image)
        img_max = np.max(image)
        result = ((image - img_min) * 255 / (img_max - img_min)).astype(np.uint8)

    return result

# Test
img = cv2.imread('low_contrast.jpg')
result = auto_contrast(img)
cv2.imshow('Original', img)
cv2.imshow('Auto Contrast', result)
cv2.waitKey(0)
```

</details>

### 문제 2: 색상 분포 분석

이미지의 주요 색상 3가지를 추출하세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np
from collections import Counter

def find_dominant_colors(image, k=3):
    """Extract dominant colors using K-means"""
    # Convert image to 1D array
    pixels = image.reshape(-1, 3).astype(np.float32)

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Count pixels in each cluster
    label_counts = Counter(labels.flatten())

    # Return colors and ratios
    colors = []
    total = len(labels)
    for idx, count in label_counts.most_common(k):
        color = centers[idx].astype(int)
        percentage = count / total * 100
        colors.append((color, percentage))

    # Visualize results
    result = np.zeros((100, 300, 3), dtype=np.uint8)
    x = 0
    for color, pct in colors:
        width = int(pct * 3)
        result[:, x:x+width] = color
        x += width
        print(f"BGR: {color}, Ratio: {pct:.1f}%")

    cv2.imshow('Dominant Colors', result)
    cv2.waitKey(0)

    return colors

# Test
img = cv2.imread('colorful.jpg')
colors = find_dominant_colors(img, k=5)
```

</details>

### 문제 3: 조명 균일화

조명이 불균일한 문서 이미지를 균일하게 만드세요.

<details>
<summary>정답 코드</summary>

```python
import cv2
import numpy as np

def normalize_illumination(image):
    """Illumination normalization"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Estimate background (large blur)
    background = cv2.GaussianBlur(gray, (101, 101), 0)

    # Remove background (original / background)
    normalized = cv2.divide(gray, background, scale=255)

    # Apply additional CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    cv2.imshow('Original', gray)
    cv2.imshow('Background', background)
    cv2.imshow('Normalized', normalized)
    cv2.imshow('Enhanced', enhanced)
    cv2.waitKey(0)

    return enhanced

# Test
img = cv2.imread('uneven_document.jpg')
result = normalize_illumination(img)
```

</details>

### 추천 문제

| 난이도 | 주제 | 설명 |
|--------|------|------|
| ⭐ | 히스토그램 그리기 | RGB 채널별 히스토그램 시각화 |
| ⭐⭐ | 대비 향상 | equalizeHist vs CLAHE 비교 |
| ⭐⭐ | 이미지 유사도 | 히스토그램으로 유사 이미지 찾기 |
| ⭐⭐⭐ | 객체 추적 | CamShift로 색상 객체 추적 |
| ⭐⭐⭐ | HDR 톤맵핑 | 다중 노출 이미지 합성 |

---

## 다음 단계

- [특징점 검출 (Feature Detection)](./13_Feature_Detection.md) - Harris, FAST, SIFT, ORB

---

## 참고 자료

- [OpenCV Histograms](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html)
- [Histogram Equalization](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
- [Histogram Backprojection](https://docs.opencv.org/4.x/dc/df6/tutorial_py_histogram_backprojection.html)
