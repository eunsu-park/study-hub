# 색상 공간

## 개요

컴퓨터 비전에서 색상 공간(Color Space)은 색상을 표현하는 방법입니다. OpenCV는 기본적으로 BGR 색상 공간을 사용하지만, 특정 작업에는 HSV, LAB 등 다른 색상 공간이 더 효과적입니다. 이 문서에서는 다양한 색상 공간의 특성과 변환 방법, 그리고 색상 기반 객체 추적을 학습합니다.

**난이도**: ⭐⭐ (초급-중급)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. BGR과 RGB의 차이 이해
2. HSV 색상 공간의 원리와 활용
3. `cv2.cvtColor()`를 사용한 색상 공간 변환
4. 채널 분리/병합
5. 색상 기반 객체 추적 구현

---

## 목차

1. [BGR vs RGB](#1-bgr-vs-rgb)
2. [cv2.cvtColor()와 색상 변환 상수](#2-cv2cvtcolor와-색상-변환-상수)
3. [HSV 색상 공간](#3-hsv-색상-공간)
4. [LAB 색상 공간](#4-lab-색상-공간)
5. [그레이스케일 변환](#5-그레이스케일-변환)
6. [채널 분리와 병합](#6-채널-분리와-병합)
7. [색상 기반 객체 추적](#7-색상-기반-객체-추적)
8. [연습 문제](#8-연습-문제)
9. [다음 단계](#9-다음-단계)
10. [참고 자료](#10-참고-자료)

---

## 1. BGR vs RGB

### 이론: 왜 여러 색상 공간이 존재하는가

인간의 망막에는 세 종류의 원뿔세포 — S, M, L — 가 있어 짧은, 중간, 긴 파장에 각각 민감합니다. 뇌는 각 원뿔세포 유형의 반응만 알 뿐, 들어오는 스펙트럼 자체를 직접 보지는 못합니다. 이것이 색 시각이 근본적으로 **3차원적**인 이유이며, 두 서로 다른 빛 스펙트럼이 같은 S/M/L 반응을 내면 동일하게 보이는 이유입니다(이 현상을 *메타메리즘(metamerism)*이라고 합니다).

이를 바탕으로 색은 어떠한 3차원 좌표계로도 매개변수화할 수 있습니다. 선택은 무엇을 하려느냐에 따라 결정됩니다:

| 필요 | 적절한 공간 |
|------|------------|
| R, G, B 서브픽셀을 가진 디스플레이 하드웨어 구동 | RGB |
| 조명과 무관하게 "이 픽셀이 빨간색인가?" 묻기 | HSV (hue가 답) |
| 지각 색 거리 측정 (색 A가 B에 가까운가 C에 가까운가?) | CIE LAB |
| 색을 싸게 압축 (낮은 chroma 민감도 활용) | YCbCr / YUV (JPEG, MPEG에서 사용) |
| 인쇄 (감산 혼합) | CMYK |

공간 간 변환은 **경계에서만 손실이 있습니다** — 둘의 색역(gamut) 교집합 내부에서는 부동소수 정밀도 수준까지 가역적입니다. `cv2.cvtColor`는 실용 도구이지만, 함수 시그니처를 아는 것보다 **어떤 목적 공간이 내 작업에 맞는지**를 아는 것이 더 중요합니다.

### 이론: RGB: 가산 원색

RGB는 색을 `(R, G, B) ∈ [0, 1]³` 정육면체 내의 한 점에 배치하며, 각 축은 빨강, 초록, 파랑 원색광의 강도에 대응합니다. 검정은 `(0, 0, 0)`, 흰색은 `(1, 1, 1)`, 회색 대각선은 검정에서 흰색까지 이어집니다.

이 모델은 디스플레이 하드웨어에서 직접 나옵니다. LCD나 OLED 픽셀은 그 세 가지 색의 세 서브 방출체로 구성되며, 보이는 것은 그 강도의 **가산** 혼합입니다. 그래서 RGB는 출력 렌더링과 합성에는 맞는 공간이지만, 이미지 내용을 *분석*하기에는 거의 항상 잘못된 공간입니다.

RGB가 분석 공간으로 나쁜 이유:

1. **지각적으로 균일하지 않음**. 고정된 유클리드 거리 `Δ = ‖(R₁,G₁,B₁) - (R₂,G₂,B₂)‖`는 고정된 지각 색 차이에 대응하지 않습니다. 초록 영역에서 눈은 지나치게 민감하고, 파랑에서는 둔합니다. §4(LAB 색상 공간)가 이를 해결하는 방법을 보여줍니다.
2. **채널들이 높게 상관되어 있음.** 대부분의 자연 이미지에서 `R`, `G`, `B` 값이 함께 변합니다 — 장면을 밝히면 세 채널 모두 올라갑니다. "색"을 "밝기"와 깔끔하게 분리하고 싶은 연산은 RGB에서 수행할 수 없습니다.
3. **밝기가 얽혀 있음.** "이 픽셀이 빨간색인가?"에 대한 답은 조명 불변이어야 하지만, 그림자 속 빨간 공은 밝은 빛 아래 빨간 공보다 `R, G, B`가 모두 작습니다. 어떠한 RGB 임계값도 조명 조건에 걸쳐 빨강과 빨강 아닌 것을 안정적으로 구별하지 못합니다.

### 이론: 감마 인코딩: 숨겨진 비선형성

일반적인 JPEG나 PNG에 저장된 8비트 `R, G, B` 값은 선형 밝기가 *아닙니다*. 양자화 전에 대략 `x^(1/2.2)` 곡선을 거친 **감마 인코딩** 상태입니다. 원래는 CRT 전자총의 비선형성을 보상하기 위한 것이었으나, 인간의 밝기 감도에도 우연히 일치하기 때문에 살아남았습니다. 인간의 눈은 대체로 로그 반응을 하므로, 감마 인코딩된 8비트 채널은 256레벨을 효율적으로 씁니다(필요한 어두운 영역에 더 많은 레벨).

sRGB 표준은 두 조각 인코딩을 씁니다:

```
s_encoded = { 12.92 · s                    if s ≤ 0.0031308
            { 1.055 · s^(1/2.4) - 0.055    otherwise
```

여기서 `s`는 `[0, 1]` 범위의 선형 밝기. 디코딩(sRGB → linear)은 이 함수의 역입니다.

왜 이것이 중요한가:

- **블렌딩, 평균, 블러링은 선형 공간에서 수행해야** 물리적인 빛의 결합과 일치합니다. 감마 인코딩된 두 값을 평균하면 두 값의 물리적 중간점보다 어두운 결과가 나옵니다 — 순진하게 리사이즈된 이미지에서 흔히 보이는 고전적 아티팩트.
- **OpenCV의 표준 함수들은 감마 인코딩된 `uint8` 값에 그대로 작용합니다** — 결과의 블러, 블렌드, 믹스는 근사일 뿐 기술적으로는 부정확합니다. 픽셀 단위로 정확한 색 과학을 위해서는 연산 전에 선형으로 변환(`x / 255`, 그 다음 `x > 0.04045`이면 `((x + 0.055) / 1.055) ** 2.4`, 아니면 `x / 12.92`)하고 표시용으로 다시 인코딩하세요.
- 앤티앨리어싱, 알파 합성, 물리 기반 렌더링 파이프라인은 모두 올바르게 보이려면 선형 공간 수학이 필요합니다.

일반적인 컴퓨터 비전 작업(분할, 검출, 인식)에서는 감마 곡선이 대개 무시됩니다 — 알고리즘이 수용하도록 학습하는 작은 체계적 왜곡입니다. 색 과학, HDR, 광도 측정 작업에서는 감마를 올바르게 다루는 것이 필수입니다.

### OpenCV의 기본 색상 순서

```
┌─────────────────────────────────────────────────────────────────┐
│                    BGR vs RGB Comparison                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   OpenCV (BGR)                 Most Libraries (RGB)             │
│   ┌─────────────┐              ┌─────────────┐                 │
│   │ B │ G │ R │               │ R │ G │ B │                   │
│   │[0]│[1]│[2]│               │[0]│[1]│[2]│                   │
│   └─────────────┘              └─────────────┘                 │
│                                                                 │
│   Pure red:                    Pure red:                        │
│   [0, 0, 255]                  [255, 0, 0]                      │
│                                                                 │
│   Pure blue:                   Pure blue:                       │
│   [255, 0, 0]                  [0, 0, 255]                      │
│                                                                 │
│   OpenCV libraries:            RGB libraries:                   │
│   - cv2.imread()               - matplotlib                     │
│   - cv2.imshow()               - PIL/Pillow                     │
│   - cv2.imwrite()              - Tkinter                        │
│                                - Web browsers (CSS/HTML)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### BGR을 사용하는 이유

역사적인 이유입니다. 초기 카메라와 디스플레이 하드웨어가 BGR 순서로 데이터를 저장했고, OpenCV는 이 관례를 따랐습니다.

### BGR ↔ RGB 변환

```python
import cv2
import numpy as np

img_bgr = cv2.imread('image.jpg')

# cvtColor is the safest and most readable approach — explicitly declares intent
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

img_bgr_back = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# [:, :, ::-1] reverses the channel axis in-place (zero-copy view) — faster
# than cvtColor but less readable; use when performance matters
img_rgb_np = img_bgr[:, :, ::-1]  # Reverse channel order
img_rgb_np = img_bgr[..., ::-1]   # Same result

# cv2.split + cv2.merge is slower than slicing but makes the intent explicit
# and is easier to extend (e.g., inserting a new channel between them)
b, g, r = cv2.split(img_bgr)
img_rgb_split = cv2.merge([r, g, b])
```

### matplotlib과 함께 사용하기

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

# Wrong display (BGR as-is → colors are swapped)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)  # BGR as-is → red and blue swapped
plt.title('Wrong (BGR)')
plt.axis('off')

# Correct display (convert to RGB)
plt.subplot(1, 3, 2)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('Correct (RGB)')
plt.axis('off')

# Grayscale
plt.subplot(1, 3, 3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.tight_layout()
plt.show()
```

---

## 2. cv2.cvtColor()와 색상 변환 상수

### 기본 사용법

```python
import cv2

img = cv2.imread('image.jpg')

# cv2.cvtColor(src, code) - color space conversion
dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### 주요 변환 코드

```
┌─────────────────────────────────────────────────────────────────┐
│                     Major Color Conversion Codes                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   BGR ↔ Other Color Spaces                                      │
│   ├── COLOR_BGR2RGB / COLOR_RGB2BGR                             │
│   ├── COLOR_BGR2GRAY / COLOR_GRAY2BGR                           │
│   ├── COLOR_BGR2HSV / COLOR_HSV2BGR                             │
│   ├── COLOR_BGR2LAB / COLOR_LAB2BGR                             │
│   ├── COLOR_BGR2YCrCb / COLOR_YCrCb2BGR                         │
│   └── COLOR_BGR2HLS / COLOR_HLS2BGR                             │
│                                                                 │
│   RGB ↔ Other Color Spaces                                      │
│   ├── COLOR_RGB2GRAY / COLOR_GRAY2RGB                           │
│   ├── COLOR_RGB2HSV / COLOR_HSV2RGB                             │
│   ├── COLOR_RGB2LAB / COLOR_LAB2RGB                             │
│   └── COLOR_RGB2HLS / COLOR_HLS2RGB                             │
│                                                                 │
│   Special Conversions                                           │
│   ├── COLOR_BGR2HSV_FULL  (H: 0-255)                            │
│   ├── COLOR_BGR2HSV       (H: 0-179)                            │
│   └── COLOR_BayerBG2BGR   (Bayer → BGR)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 변환 예시

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to various color spaces
conversions = {
    'Original (RGB)': img_rgb,
    'Grayscale': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
    'HSV': cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
    'LAB': cv2.cvtColor(img, cv2.COLOR_BGR2LAB),
    'YCrCb': cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb),
    'HLS': cv2.cvtColor(img, cv2.COLOR_BGR2HLS),
}

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for ax, (name, converted) in zip(axes, conversions.items()):
    if len(converted.shape) == 2:
        ax.imshow(converted, cmap='gray')
    else:
        ax.imshow(converted)
    ax.set_title(name)
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 3. HSV 색상 공간

RGB와 BGR은 색상과 밝기가 섞여 있어 조명이 변할 때 특정 색상을 분리하기 어렵습니다. HSV는 이 두 요소를 분리합니다. Hue 채널 하나만으로 색상을 나타낼 수 있으므로, 밝은 곳이든 어두운 곳이든 간단한 범위 임계처리만으로 "빨간 물체"를 감지할 수 있습니다.

### 이론: HSV: 색상과 밝기를 분리하기

HSV(Hue, Saturation, Value)는 RGB 정육면체를 사람이 색을 묘사하는 방식을 반영한 원통형으로 재매개변수화합니다:

- **Hue H** (각도 0°–360°) — 어떤 색인가. 0°에서 빨강, 120°에서 초록, 240°에서 파랑, 다시 돌아옴.
- **Saturation S** (분수 0–1) — 순수한 색 vs 흐릿한 색. 0 = 회색, 1 = 완전히 선명.
- **Value V** (분수 0–1) — 얼마나 밝은가. 0 = 검정, 1 = 해당 색상의 최대 밝기.

#### HSV 변환 공식

`R, G, B ∈ [0, 1]`이고 `M = max(R, G, B)`, `m = min(R, G, B)`, `Δ = M - m`이라 하면

```
V = M

      ⎧  0                           if Δ = 0     (색이 없음, S = 0)
S = ⎨
      ⎩  Δ / M                       otherwise

      ⎧  undefined                   if Δ = 0
      ⎪  60° · ((G - B) / Δ) mod 6   if M = R
H = ⎨
      ⎪  60° · ((B - R) / Δ + 2)     if M = G
      ⎩  60° · ((R - G) / Δ + 4)     if M = B
```

`max - min` 값 `Δ`는 RGB 점이 얼마나 대각선에서 "벗어나 있는지"를 측정합니다 — `R = G = B`(색상 미정의 회색)이면 0, 색이 더 선명해질수록 커집니다. Hue 공식은 어느 채널이 지배적인지에 따라 섹터를 고르고 양쪽 원색 사이를 선형 보간합니다.

#### 이것이 컴퓨터 비전에 중요한 이유

HSV에서 밝기는 `V`에 격리됩니다. 그림자 속과 햇빛 속에서 촬영된 빨간 물체는 비슷한 `H`(빨강은 여전히 빨강)와 비슷한 `S`(물체는 여전히 선명함)를 가지지만 `V`는 매우 다릅니다. `H`와 `S`만으로 — `V`를 무시하고 — 픽셀을 선택하는 필터는 조명 변화에 강건해집니다. 이것이 색상 기반 분할의 교과서적 레시피입니다:

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (100, 150, 50), (130, 255, 255))   # blue-ish, any brightness
```

#### OpenCV의 `uint8` 스케일링

OpenCV는 HSV를 `uint8`로 저장하는데, 이는 256레벨만 가집니다. Hue를 각도(도)로 하려면 ≥ 360레벨이 필요하므로 OpenCV는 절반으로 축소합니다: **H ∈ [0, 180], [0, 360]이 아님**. Saturation과 Value는 `[0, 255]`로 스케일됩니다. 흔한 버그: `[0, 360]`을 쓰는 튜토리얼에서 HSV 범위를 복사해 결과가 두 배 어긋나는 경우. OpenCV에서 범위를 생각할 때는 모든 각도를 **2로 나누세요**.

### HSV란?

HSV는 색상(Hue), 채도(Saturation), 명도(Value)로 색을 표현합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      HSV Color Space                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   H (Hue) - Color                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0°    60°   120°   180°   240°   300°   360°          │   │
│   │  Red   Yellow Green  Cyan   Blue  Magenta Red          │   │
│   │  ├──────┼──────┼──────┼──────┼──────┼──────┤            │   │
│   │  0     30     60     90    120    150    179            │   │
│   │      (OpenCV H range: 0-179)                            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   S (Saturation) - Saturation (0-255)                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (grayscale/gray)  ──────────────▶  255 (pure color)  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   V (Value) - Brightness (0-255)                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (black)  ──────────────────▶  255 (bright)           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│                        V (Brightness)                           │
│                          ▲                                       │
│                          │    White                              │
│                          │   /                                   │
│                          │  /                                    │
│                          │ /     Pure color                      │
│                          │/───────●                              │
│                          │        ╲                              │
│                          │         ╲  S (Saturation)             │
│                          │          ╲                            │
│                          ●───────────╲───▶ H (Hue, circular)     │
│                        Black                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### HSV 변환 및 채널 확인

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

# BGR → HSV conversion
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split channels
h, s, v = cv2.split(hsv)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')

axes[0, 1].imshow(h, cmap='hsv')  # Use hsv colormap for Hue
axes[0, 1].set_title('H (Hue)')

axes[1, 0].imshow(s, cmap='gray')
axes[1, 0].set_title('S (Saturation)')

axes[1, 1].imshow(v, cmap='gray')
axes[1, 1].set_title('V (Value)')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### HSV의 장점

```python
import cv2
import numpy as np

# In HSV, lighting changes mainly affect V (brightness); H stays stable.
# That's why HSV works far better than BGR for robust color detection.

img = cv2.imread('red_objects.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Red wraps around the Hue circle: it appears near H=0 AND near H=180.
# Two separate ranges are needed because OpenCV's H axis is 0-179, not circular.
lower_red1 = np.array([0, 100, 100])    # S>100 and V>100 exclude near-gray pixels
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

# Bitwise OR merges both masks into one — pixels belonging to either range pass
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 | mask2

# bitwise_and zeroes out pixels where mask=0, keeping only the detected color
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('Original', img)
cv2.imshow('Mask', mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 주요 색상의 HSV 범위

```
┌─────────────────────────────────────────────────────────────────┐
│                    Common Color HSV Ranges (OpenCV)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Color      H (Hue)        S (Saturation)   V (Value)          │
│   ──────────────────────────────────────────────────────────    │
│   Red        0-10, 160-179   100-255         100-255            │
│   Orange     10-25           100-255         100-255            │
│   Yellow     25-35           100-255         100-255            │
│   Green      35-85           100-255         100-255            │
│   Cyan       85-95           100-255         100-255            │
│   Blue       95-130          100-255         100-255            │
│   Magenta    130-160         100-255         100-255            │
│                                                                 │
│   White      0-179           0-30            200-255            │
│   Black      0-179           0-255           0-50               │
│   Gray       0-179           0-30            50-200             │
│                                                                 │
│   Note: Ranges need adjustment based on lighting conditions     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. LAB 색상 공간

LAB은 RGB와 HSV가 공통으로 가진 문제를 해결합니다. 즉, 수치상의 동일한 차이가 인간이 느끼는 동일한 차이와 일치하지 않는다는 문제입니다. LAB에서는 두 색상 벡터 간의 유클리드 거리가 인간의 눈에 보이는 색상 차이와 근접하게 대응합니다. 따라서 지각적 색상 비교와 전문적인 색상 보정에 이상적인 색상 공간입니다.

### 이론: CIE LAB: 지각적으로 균일한 색

CIE LAB 공간(CIELAB 또는 L*a*b*로도 불림)은 1976년에 수치 거리가 지각 색 차이와 일치하도록 설계되었습니다. 축:

- **L*** — 지각 밝기, `[0, 100]`. 0 = 검정, 100 = 확산 흰색.
- **a*** — 녹색-적색 축. 음수 = 녹색, 양수 = 적색.
- **b*** — 청색-황색 축. 음수 = 청색, 양수 = 황색.

핵심 보장은 유클리드 거리

```
ΔE = √((L₁ - L₂)² + (a₁ - a₂)² + (b₁ - b₂)²)
```

가 색역 전반에 걸쳐 동등하게 달라 보이는 색들에 대해 근사적으로 일정하다는 것입니다. 대략 `ΔE ≈ 2.3`이 평균 식별 임계값(just-noticeable difference)이고, `ΔE = 10`은 확실히 다른 색입니다.

이 공간이 필요한 경우:

- 지각적 유사도로 색을 매칭하거나 순위 매기기(팔레트 양자화, 색 검색).
- 조명을 무시하고 "색만의" 기울기 계산(a-b 평면 거리).
- 색 충실도를 보존해야 하는 인쇄 또는 렌더링 파이프라인.

sRGB에서 CIE LAB으로의 변환은 비선형입니다(CIE XYZ 3자극 공간과 세제곱근 함수를 거칩니다). OpenCV는 이를 `cv2.cvtColor(img, cv2.COLOR_BGR2LAB)` 뒤에 숨기고, `L ∈ [0, 100]`을 `uint8` 이미지에 대해 `[0, 255]`로 스케일하여 저장합니다(`a, b`는 0점이 128에 오도록 128만큼 오프셋됩니다).

### LAB이란?

LAB(또는 CIELAB)은 인간의 색상 인지에 기반한 색상 공간입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      LAB Color Space                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   L (Lightness) - Brightness                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (black)  ──────────────────────▶  255 (white)        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   A - Green(-) ↔ Red(+)                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (green)  ────── 128 (neutral) ──────  255 (red)      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   B - Blue(-) ↔ Yellow(+)                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0 (blue)  ────── 128 (neutral) ──────  255 (yellow)    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│                     +B (Yellow)                                  │
│                        ▲                                        │
│                        │                                        │
│            -A ◀────────┼────────▶ +A                            │
│          (Green)       │        (Red)                           │
│                        │                                        │
│                        ▼                                        │
│                     -B (Blue)                                    │
│                                                                 │
│   Advantages:                                                   │
│   - Color distance calculation similar to human vision          │
│   - Brightness and color are separated                          │
│   - Useful for color correction and color transfer              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### LAB 변환 및 활용

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab)

# Modifying only L leaves the color (a, b) untouched — this is the key advantage
# over adjusting brightness in BGR, where adding a constant shifts all three channels
# and inadvertently changes the hue
l_adjusted = cv2.add(l, 30)  # cv2.add saturates at 255, avoiding overflow wrapping
l_adjusted = np.clip(l_adjusted, 0, 255).astype(np.uint8)

# Reassemble: a and b unchanged, so colors remain perceptually identical to the original
lab_adjusted = cv2.merge([l_adjusted, a, b])
result = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')

axes[0, 1].imshow(l, cmap='gray')
axes[0, 1].set_title('L (Lightness)')

axes[0, 2].imshow(a, cmap='RdYlGn_r')
axes[0, 2].set_title('A (Green-Red)')

axes[1, 0].imshow(b, cmap='YlGnBu_r')
axes[1, 0].set_title('B (Blue-Yellow)')

axes[1, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Brightness Adjusted')

for ax in axes.flatten():
    ax.axis('off')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
```

### CLAHE로 LAB 밝기 보정

```python
import cv2

img = cv2.imread('dark_image.jpg')

# Working in LAB is crucial here: CLAHE must be applied only to lightness (L),
# not to color channels — otherwise it would create color distortions
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# CLAHE enhances local contrast adaptively per tile rather than globally,
# preventing over-brightening bright regions while lifting dark ones.
# clipLimit=2.0 caps the amplification to avoid amplifying noise.
# tileGridSize=(8,8) is a good balance: coarser → more global; finer → more local
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

# a and b carry the color; only L was modified, so hues are preserved
lab_clahe = cv2.merge([l_clahe, a, b])
result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

cv2.imshow('Original', img)
cv2.imshow('CLAHE Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 5. 그레이스케일 변환

### 이론: 그레이스케일: 휘도 가중 변환

컬러 이미지를 그레이스케일로 변환하는 것은 `R`, `G`, `B`의 단순 평균이 *아닙니다* — 세 채널은 지각 밝기에 다른 양으로 기여합니다. 눈은 초록빛에 가장 민감하고 파랑에 가장 둔감하기 때문입니다. 표준 ITU-R BT.601 공식(OpenCV의 `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`가 사용):

```
Y = 0.299 · R + 0.587 · G + 0.114 · B
```

직접 평균 `(R + G + B) / 3`은 눈에 띄게 "잘못"되어 보입니다 — 초록 나뭇잎이 너무 어둡고 파란 하늘이 너무 밝게 보입니다. 가중치가 채널별 밝기 기여도를 반영하지 않기 때문입니다.

BT.709(HDTV)는 약간 다른 가중치를 씁니다: `Y = 0.2126 R + 0.7152 G + 0.0722 B`. 수치 차이는 방송 품질 작업에 중요하며, 일반 컴퓨터 비전 작업에서는 어느 쪽이든 괜찮습니다.

### 변환 원리

```
┌─────────────────────────────────────────────────────────────────┐
│                   Grayscale Conversion Principle                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   BGR → Grayscale conversion formula:                           │
│                                                                 │
│   Gray = 0.114 × B + 0.587 × G + 0.299 × R                     │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   Why not simple average?                               │   │
│   │                                                         │   │
│   │   Human eyes are most sensitive to green and least to blue │
│   │   Therefore, green (G) has the highest weight (0.587)  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Color image                     Grayscale                     │
│   ┌───────────────┐              ┌───────────────┐             │
│   │ B │ G │ R │               │     Gray      │             │
│   │200│100│ 50│    ───▶       │      121      │             │
│   └───────────────┘              └───────────────┘             │
│   0.114×200 + 0.587×100 + 0.299×50 = 121.45                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

`Gray = 0.114·B + 0.587·G + 0.299·R` 공식은 인간의 광수용체 민감도에 따라 채널에 가중치를 부여합니다. 눈은 녹색에 가장 민감(~55%)하고, 빨간색에 중간 정도(~30%), 파란색에 가장 덜 민감(~11%)합니다. 단순 평균(각 0.333)을 사용하면 파란색 영역이 너무 밝고 녹색 영역이 너무 어두운 그레이스케일이 만들어집니다.

### 그레이스케일 변환 방법

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# cvtColor uses the luminosity-weighted formula above — preferred over imread grayscale
# because it works on an already-loaded image without re-reading from disk
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Method 2: Read directly with imread
gray2 = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Method 3: Manual calculation with NumPy (for learning)
b, g, r = cv2.split(img)
gray3 = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)

# Method 4: Simple average (not recommended - visually unnatural)
gray4 = np.mean(img, axis=2).astype(np.uint8)

# Compare results
print(f"cvtColor result: {gray1.shape}")
print(f"Manual calculation result: {gray3.shape}")
print(f"Max difference: {np.max(np.abs(gray1.astype(int) - gray3.astype(int)))}")
```

### 그레이스케일 → 컬러 (의사 컬러)

```python
import cv2

gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Grayscale → 3 channels (still grayscale)
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Apply colormap (heatmap, etc.)
# COLORMAP_JET, COLORMAP_HOT, COLORMAP_RAINBOW, etc.
colormap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

cv2.imshow('Grayscale', gray)
cv2.imshow('Colormap', colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 6. 채널 분리와 병합

### cv2.split()과 cv2.merge()

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Split channels
b, g, r = cv2.split(img)

# Or use NumPy indexing (faster)
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]

# Merge channels
merged = cv2.merge([b, g, r])  # BGR order

# Change channel order when merging (BGR → RGB)
rgb = cv2.merge([r, g, b])

# Combine with empty channels (display single channel only)
zeros = np.zeros_like(b)
only_blue = cv2.merge([b, zeros, zeros])
only_green = cv2.merge([zeros, g, zeros])
only_red = cv2.merge([zeros, zeros, r])
```

### 채널별 시각화

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
b, g, r = cv2.split(img)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Original
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')

# Each channel (as grayscale)
axes[0, 1].imshow(r, cmap='gray')
axes[0, 1].set_title('Red Channel')

axes[0, 2].imshow(g, cmap='gray')
axes[0, 2].set_title('Green Channel')

axes[1, 0].imshow(b, cmap='gray')
axes[1, 0].set_title('Blue Channel')

# Each channel (in color)
zeros = np.zeros_like(b)
axes[1, 1].imshow(cv2.merge([zeros, zeros, r]))  # RGB order
axes[1, 1].set_title('Red Only')

axes[1, 2].imshow(cv2.merge([zeros, g, zeros]))
axes[1, 2].set_title('Green Only')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
```

### 채널 조작 예제

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 1. Boost red channel: cast to int16 first to avoid uint8 overflow, then clip
b, g, r = cv2.split(img)
r_boost = np.clip(r.astype(np.int16) + 50, 0, 255).astype(np.uint8)
warm = cv2.merge([b, g, r_boost])  # Higher R relative to B gives a warm/sunset feel

# 2. Swapping R and B produces a "cool" or infrared-like look — useful for artistic effects
b, g, r = cv2.split(img)
swapped = cv2.merge([r, g, b])

# 3. Simple average is visually inaccurate (ignores perceptual weights) but useful
# as a fast approximation when exact luminance doesn't matter
b, g, r = cv2.split(img)
gray_avg = ((b.astype(np.int16) + g + r) // 3).astype(np.uint8)

# 4. zeros_like preserves the same shape and dtype as b — safer than np.zeros((h,w))
b, g, r = cv2.split(img)
only_r = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
```

---

## 7. 색상 기반 객체 추적

### inRange()를 사용한 색상 필터링

```
┌─────────────────────────────────────────────────────────────────┐
│                   Color-Based Object Tracking Pipeline          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input image (BGR)                                             │
│        │                                                        │
│        ▼                                                        │
│   HSV conversion                                                │
│        │                                                        │
│        ▼                                                        │
│   cv2.inRange(hsv, lower, upper) ──▶ Binary mask               │
│        │                                                        │
│        ▼                                                        │
│   Noise removal (morphological operations)                      │
│        │                                                        │
│        ▼                                                        │
│   Contour detection                                             │
│        │                                                        │
│        ▼                                                        │
│   Extract object position/size                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 색상 추적 구현

```python
import cv2
import numpy as np

def track_color(img, lower_hsv, upper_hsv):
    """Track objects in a specific color range"""
    # HSV conversion
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Detect contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Draw results
    result = img.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area filter
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Center point
            cx, cy = x + w//2, y + h//2
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)

    return result, mask


# Example usage: Track blue
img = cv2.imread('blue_objects.jpg')

lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

result, mask = track_color(img, lower_blue, upper_blue)

cv2.imshow('Original', img)
cv2.imshow('Mask', mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 실시간 색상 추적 (웹캠)

```python
import cv2
import numpy as np

def nothing(x):
    pass

# Create trackbars
cv2.namedWindow('Trackbars')
cv2.createTrackbar('H_Low', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('H_High', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('S_Low', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('S_High', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V_Low', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('V_High', 'Trackbars', 255, 255, nothing)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Read trackbar values
    h_low = cv2.getTrackbarPos('H_Low', 'Trackbars')
    h_high = cv2.getTrackbarPos('H_High', 'Trackbars')
    s_low = cv2.getTrackbarPos('S_Low', 'Trackbars')
    s_high = cv2.getTrackbarPos('S_High', 'Trackbars')
    v_low = cv2.getTrackbarPos('V_Low', 'Trackbars')
    v_high = cv2.getTrackbarPos('V_High', 'Trackbars')

    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])

    # HSV conversion and mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 다중 색상 추적

```python
import cv2
import numpy as np

# Define multiple colors
colors = {
    'red': {
        'lower1': np.array([0, 100, 100]),
        'upper1': np.array([10, 255, 255]),
        'lower2': np.array([160, 100, 100]),
        'upper2': np.array([179, 255, 255]),
        'color': (0, 0, 255)
    },
    'green': {
        'lower': np.array([35, 100, 100]),
        'upper': np.array([85, 255, 255]),
        'color': (0, 255, 0)
    },
    'blue': {
        'lower': np.array([100, 100, 100]),
        'upper': np.array([130, 255, 255]),
        'color': (255, 0, 0)
    }
}

def track_multiple_colors(img, colors):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result = img.copy()

    for name, params in colors.items():
        # Create mask
        if 'lower1' in params:  # For colors like red with two ranges
            mask1 = cv2.inRange(hsv, params['lower1'], params['upper1'])
            mask2 = cv2.inRange(hsv, params['lower2'], params['upper2'])
            mask = mask1 | mask2
        else:
            mask = cv2.inRange(hsv, params['lower'], params['upper'])

        # Detect contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x+w, y+h), params['color'], 2)
                cv2.putText(result, name, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, params['color'], 2)

    return result
```

---

## 8. 연습 문제

### 연습 1: 색상 팔레트 생성

16가지 주요 색상(빨강, 주황, 노랑, 초록, 청록, 파랑, 보라, 분홍, 흰색, 검정, 회색 등)을 BGR 값으로 정의하고, 100x100 크기의 색상 칩을 4x4 격자로 배치한 팔레트 이미지를 생성하세요.

### 연습 2: HSV 색상 선택기

마우스로 이미지를 클릭하면 해당 픽셀의 HSV 값을 출력하고, 그 색상과 유사한 모든 영역을 하이라이트하는 프로그램을 작성하세요.

```python
# Hint: use cv2.setMouseCallback()
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Output HSV value of clicked position
        pass
```

### 연습 3: 채널 스왑 효과

이미지의 채널을 다양하게 조합하여 6가지 효과(BGR, BRG, GBR, GRB, RBG, RGB)를 만들고 비교하세요.

### 연습 4: 피부색 검출

HSV와 YCrCb 색상 공간을 사용하여 이미지에서 피부색 영역을 검출하세요. 두 방법의 결과를 비교하세요.

```python
# Example HSV ranges for skin color
# H: 0-50, S: 20-150, V: 70-255

# Example YCrCb ranges for skin color
# Y: 0-255, Cr: 135-180, Cb: 85-135
```

### 연습 5: 색상 전이 애니메이션

H 채널을 점진적으로 증가시켜 이미지의 색상이 무지개처럼 변하는 애니메이션을 만드세요.

```python
# Hint
for h_shift in range(0, 180, 5):
    h_channel = (original_h + h_shift) % 180
    # ...
```

---

## 9. 다음 단계

[기하학적 변환](./04_Geometric_Transforms.md)에서 이미지 크기 조절, 회전, 뒤집기, 어파인/원근 변환 등을 학습합니다!

**다음에 배울 내용**:
- `cv2.resize()`와 보간법
- 회전, 뒤집기 함수
- 어파인 변환 (이동, 회전, 스케일)
- 원근 변환 (문서 스캔)

---

## 10. 참고 자료

### 공식 문서

- [cvtColor() 문서](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)
- [색상 공간 변환](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)
- [inRange() 문서](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html)

### 관련 학습 자료

| 폴더 | 관련 내용 |
|------|----------|
| [이미지 기초 연산](./02_Image_Basics.md) | 이미지 읽기, 픽셀 접근 |
| [이진화 및 임계처리](./07_Thresholding.md) | HSV 기반 임계처리 |

### 색상 공간 참고

- [색상 공간 위키피디아](https://en.wikipedia.org/wiki/Color_space)
- [HSV 색상 모델](https://en.wikipedia.org/wiki/HSL_and_HSV)
- [CIELAB 색상 공간](https://en.wikipedia.org/wiki/CIELAB_color_space)

