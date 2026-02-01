# 히스토그램 분석 (Histogram Analysis)

## 개요

히스토그램은 이미지의 밝기 분포를 나타내는 그래프입니다. 이미지 분석, 대비 향상, 색상 비교 등에 활용됩니다. 이 레슨에서는 히스토그램 계산, 균등화, CLAHE, 비교, 역투영 등을 학습합니다.

---

## 목차

1. [히스토그램 기초](#1-히스토그램-기초)
2. [히스토그램 계산](#2-히스토그램-계산)
3. [히스토그램 균등화](#3-히스토그램-균등화)
4. [CLAHE](#4-clahe)
5. [히스토그램 비교](#5-히스토그램-비교)
6. [역투영](#6-역투영)
7. [연습 문제](#7-연습-문제)

---

## 1. 히스토그램 기초

### 히스토그램이란?

```
히스토그램 (Histogram):
이미지 픽셀 밝기값의 분포를 나타내는 그래프

X축: 밝기값 (0-255)
Y축: 해당 밝기값을 가진 픽셀 수

어두운 이미지            밝은 이미지           대비 좋은 이미지
    │                        │                      │
 빈 │█                       │       █              │   █   █
 도 │██                      │      ██              │  ███ ███
 수 │███                     │     ███              │ █████████
    └────────────           └────────────          └────────────
    0          255          0          255         0          255
      밝기값                   밝기값                  밝기값
```

### 히스토그램의 활용

```
1. 이미지 분석
   - 노출 상태 확인 (과노출, 저노출)
   - 대비 수준 파악

2. 이미지 향상
   - 히스토그램 균등화
   - 대비 조정

3. 이미지 비교
   - 유사 이미지 검색
   - 색상 기반 매칭

4. 객체 추적
   - 색상 히스토그램 역투영
   - CamShift/MeanShift 알고리즘
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
    """그레이스케일 히스토그램 계산 및 시각화"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 히스토그램 계산
    hist = cv2.calcHist(
        [img],           # 이미지 (리스트로 전달)
        [0],             # 채널 (그레이스케일은 0)
        None,            # 마스크 (전체 이미지)
        [256],           # 빈 개수 (0-255: 256개)
        [0, 256]         # 값 범위
    )

    # Matplotlib으로 시각화
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
    """RGB 채널별 히스토그램"""
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
        # BGR 순서이므로 인덱스 조정: R=2, G=1, B=0
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
    """Hue-Saturation 2D 히스토그램"""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # H: 0-180, S: 0-256
    hist = cv2.calcHist(
        [hsv],
        [0, 1],          # H와 S 채널
        None,
        [30, 32],        # 빈 개수 (H: 30, S: 32)
        [0, 180, 0, 256] # 범위 (H: 0-180, S: 0-256)
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
    """특정 영역만 히스토그램 계산"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    # 원형 마스크 생성
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), min(h, w)//3, 255, -1)

    # 전체 히스토그램
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])

    # 마스크 영역만 히스토그램
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
히스토그램 균등화 (Histogram Equalization):
이미지의 밝기 분포를 균일하게 만들어 대비 향상

원본 히스토그램               균등화된 히스토그램
    │                              │
    │█                             │   █ █ █
    │███                           │ █ █ █ █ █
    │█████                         │█ █ █ █ █ █ █
    └────────────                  └────────────────
    0          255                 0              255

변환 과정:
1. 히스토그램 계산
2. 누적 분포 함수 (CDF) 계산
3. CDF 정규화
4. 픽셀값 매핑
```

### cv2.equalizeHist()

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize_histogram_demo(image_path):
    """히스토그램 균등화 데모"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 히스토그램 균등화
    equalized = cv2.equalizeHist(img)

    # 히스토그램 계산
    hist_before = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_after = cv2.calcHist([equalized], [0], None, [256], [0, 256])

    # 시각화
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
    """컬러 이미지 히스토그램 균등화"""
    img = cv2.imread(image_path)

    # 방법 1: YCrCb 색공간 사용 (권장)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])  # Y 채널만 균등화
    result_ycrcb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # 방법 2: HSV 색공간 사용
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])  # V 채널만 균등화
    result_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 방법 3: 각 채널 개별 균등화 (색상 왜곡 가능)
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
적응형 히스토그램 균등화

문제점: 전역 균등화는 노이즈 증폭 가능
해결: 이미지를 타일로 나누어 지역적으로 균등화

┌────┬────┬────┬────┐
│    │    │    │    │
│ T1 │ T2 │ T3 │ T4 │   각 타일(Tile)별로
├────┼────┼────┼────┤   균등화 적용
│    │    │    │    │
│ T5 │ T6 │ T7 │ T8 │   경계는 보간으로
├────┼────┼────┼────┤   부드럽게 연결
│ T9 │T10 │T11 │T12 │
└────┴────┴────┴────┘

특징:
- clipLimit: 대비 제한 (높을수록 강한 대비)
- tileGridSize: 타일 크기 (작을수록 세밀)
```

### cv2.createCLAHE()

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def clahe_demo(image_path):
    """CLAHE 적용 데모"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 일반 균등화
    equalized = cv2.equalizeHist(img)

    # CLAHE 생성 및 적용
    clahe = cv2.createCLAHE(
        clipLimit=2.0,      # 대비 제한 (1.0 ~ 4.0 권장)
        tileGridSize=(8, 8) # 타일 크기
    )
    clahe_result = clahe.apply(img)

    # 비교
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
    """CLAHE 파라미터별 비교"""
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
    """컬러 이미지에 CLAHE 적용"""
    img = cv2.imread(image_path)

    # LAB 색공간 변환
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # L 채널에 CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # BGR로 변환
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
    """여러 이미지의 히스토그램 비교"""
    # 기준 이미지
    base_img = cv2.imread(image_paths[0])
    base_hsv = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV)

    # 히스토그램 계산 (H-S 2D)
    base_hist = cv2.calcHist(
        [base_hsv], [0, 1], None,
        [50, 60], [0, 180, 0, 256]
    )
    cv2.normalize(base_hist, base_hist, 0, 1, cv2.NORM_MINMAX)

    print(f"기준 이미지: {image_paths[0]}")
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

        print(f"\n비교: {path}")
        for method, name in methods:
            result = cv2.compareHist(base_hist, hist, method)
            print(f"  {name}: {result:.4f}")

# 사용 예
image_files = ['ref.jpg', 'similar1.jpg', 'similar2.jpg', 'different.jpg']
compare_histograms(image_files)
```

### 유사 이미지 검색

```python
import cv2
import numpy as np
import os

def find_similar_images(query_path, search_dir, top_k=5):
    """히스토그램 기반 유사 이미지 검색"""
    # 쿼리 이미지 히스토그램
    query = cv2.imread(query_path)
    query_hsv = cv2.cvtColor(query, cv2.COLOR_BGR2HSV)
    query_hist = cv2.calcHist([query_hsv], [0, 1], None,
                               [50, 60], [0, 180, 0, 256])
    cv2.normalize(query_hist, query_hist, 0, 1, cv2.NORM_MINMAX)

    results = []

    # 검색 디렉토리의 모든 이미지와 비교
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

        # 상관관계 계산 (높을수록 유사)
        similarity = cv2.compareHist(query_hist, hist, cv2.HISTCMP_CORREL)
        results.append((filename, similarity))

    # 유사도순 정렬
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"쿼리: {query_path}")
    print(f"\nTop {top_k} 유사 이미지:")
    for filename, sim in results[:top_k]:
        print(f"  {filename}: {sim:.4f}")

    return results[:top_k]

# 사용 예
find_similar_images('query.jpg', './image_database/', top_k=5)
```

---

## 6. 역투영

### 개념

```
역투영 (Backprojection):
히스토그램을 이용해 특정 색상 영역 검출

과정:
1. 관심 객체(ROI)의 색상 히스토그램 계산
2. 전체 이미지에서 각 픽셀의 히스토그램 값으로 대체
3. 높은 값 = 관심 색상과 유사

활용:
- 색상 기반 객체 추적
- CamShift/MeanShift 알고리즘의 핵심

예시:
┌─────────────┐       ┌─────────────┐
│   🟡 ROI    │       │ ■ ■ □ □ □ │
│  (노란색)   │  ──▶  │ ■ ■ ■ □ □ │  높은 값 = 노란색
│             │       │ □ ■ ■ ■ □ │
└─────────────┘       └─────────────┘
  색상 히스토그램        역투영 결과
```

### cv2.calcBackProject()

```python
import cv2
import numpy as np

def backprojection_demo(image_path, roi_coords):
    """역투영 데모"""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ROI 영역 설정
    x, y, w, h = roi_coords
    roi = hsv[y:y+h, x:x+w]

    # ROI의 히스토그램 계산
    roi_hist = cv2.calcHist(
        [roi], [0, 1], None,
        [180, 256], [0, 180, 0, 256]
    )
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # 역투영
    backproj = cv2.calcBackProject(
        [hsv], [0, 1], roi_hist,
        [0, 180, 0, 256], 1
    )

    # 필터링으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(backproj, -1, kernel, backproj)
    _, backproj = cv2.threshold(backproj, 50, 255, cv2.THRESH_BINARY)

    # 시각화
    result = img.copy()
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 검출된 영역 마스킹
    mask = cv2.merge([backproj, backproj, backproj])
    detected = cv2.bitwise_and(img, mask)

    cv2.imshow('Original with ROI', result)
    cv2.imshow('Back Projection', backproj)
    cv2.imshow('Detected', detected)
    cv2.waitKey(0)

    return backproj

# 사용 예 (x, y, width, height)
backprojection_demo('scene.jpg', (100, 100, 50, 50))
```

### 피부색 검출

```python
import cv2
import numpy as np

def detect_skin(image_path):
    """피부색 검출 (역투영 활용)"""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 피부색 범위 (HSV)
    # H: 0-20, S: 48-255, V: 80-255 (일반적인 피부색)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # 피부색 마스크
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 피부색 영역의 히스토그램 생성
    skin_region = cv2.bitwise_and(hsv, hsv, mask=skin_mask)
    skin_hist = cv2.calcHist([skin_region], [0, 1], skin_mask,
                              [180, 256], [0, 180, 0, 256])
    cv2.normalize(skin_hist, skin_hist, 0, 255, cv2.NORM_MINMAX)

    # 역투영
    backproj = cv2.calcBackProject([hsv], [0, 1], skin_hist,
                                    [0, 180, 0, 256], 1)

    # 모폴로지 연산
    kernel = np.ones((5, 5), np.uint8)
    backproj = cv2.morphologyEx(backproj, cv2.MORPH_OPEN, kernel)
    backproj = cv2.morphologyEx(backproj, cv2.MORPH_CLOSE, kernel)

    # 결과
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
    """CamShift를 이용한 객체 추적"""
    cap = cv2.VideoCapture(video_path)

    # 첫 프레임에서 ROI 선택
    ret, frame = cap.read()
    if not ret:
        return

    # ROI 선택 (마우스로 선택하거나 직접 지정)
    roi = cv2.selectROI('Select ROI', frame, False)
    cv2.destroyWindow('Select ROI')

    x, y, w, h = roi
    track_window = (x, y, w, h)

    # ROI의 히스토그램 계산
    roi_frame = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]),
                       np.array([180, 255, 255]))

    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # CamShift 종료 조건
    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 역투영
        backproj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # CamShift 적용
        ret, track_window = cv2.CamShift(backproj, track_window, term_criteria)

        # 결과 그리기 (회전된 사각형)
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
    """자동 대비 조정 (히스토그램 스트레칭)"""
    if len(image.shape) == 3:
        # 컬러 이미지: LAB 변환
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # L 채널에 히스토그램 스트레칭
        l_min = np.min(l)
        l_max = np.max(l)
        l_stretched = ((l - l_min) * 255 / (l_max - l_min)).astype(np.uint8)

        lab_stretched = cv2.merge([l_stretched, a, b])
        result = cv2.cvtColor(lab_stretched, cv2.COLOR_LAB2BGR)
    else:
        # 그레이스케일
        img_min = np.min(image)
        img_max = np.max(image)
        result = ((image - img_min) * 255 / (img_max - img_min)).astype(np.uint8)

    return result

# 테스트
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
    """K-means로 주요 색상 추출"""
    # 이미지를 1D 배열로 변환
    pixels = image.reshape(-1, 3).astype(np.float32)

    # K-means 클러스터링
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # 각 클러스터의 픽셀 수 계산
    label_counts = Counter(labels.flatten())

    # 색상과 비율 반환
    colors = []
    total = len(labels)
    for idx, count in label_counts.most_common(k):
        color = centers[idx].astype(int)
        percentage = count / total * 100
        colors.append((color, percentage))

    # 결과 시각화
    result = np.zeros((100, 300, 3), dtype=np.uint8)
    x = 0
    for color, pct in colors:
        width = int(pct * 3)
        result[:, x:x+width] = color
        x += width
        print(f"BGR: {color}, 비율: {pct:.1f}%")

    cv2.imshow('Dominant Colors', result)
    cv2.waitKey(0)

    return colors

# 테스트
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
    """조명 균일화"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 배경 추정 (큰 블러)
    background = cv2.GaussianBlur(gray, (101, 101), 0)

    # 배경 제거 (원본 / 배경)
    normalized = cv2.divide(gray, background, scale=255)

    # CLAHE 추가 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    cv2.imshow('Original', gray)
    cv2.imshow('Background', background)
    cv2.imshow('Normalized', normalized)
    cv2.imshow('Enhanced', enhanced)
    cv2.waitKey(0)

    return enhanced

# 테스트
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

- [13_Feature_Detection.md](./13_Feature_Detection.md) - Harris, FAST, SIFT, ORB

---

## 참고 자료

- [OpenCV Histograms](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html)
- [Histogram Equalization](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
- [Histogram Backprojection](https://docs.opencv.org/4.x/dc/df6/tutorial_py_histogram_backprojection.html)
