# 모폴로지 연산

## 개요

모폴로지 연산(Morphological Operations)은 이진 이미지나 그레이스케일 이미지의 형태를 기반으로 하는 연산입니다. 주로 노이즈 제거, 객체 분리, 홀 채우기 등에 사용됩니다. 이 문서에서는 구조 요소의 개념부터 다양한 모폴로지 연산의 활용까지 학습합니다.

**난이도**: ⭐⭐ (초급-중급)

**학습 목표**:
- 구조 요소(Structuring Element) 이해
- 침식(Erosion)과 팽창(Dilation) 연산
- 열기(Opening)와 닫기(Closing) 연산
- 그래디언트, 탑햇, 블랙햇 연산
- 노이즈 제거 및 객체 분리 응용

---

## 목차

1. [모폴로지 연산 개요](#1-모폴로지-연산-개요)
2. [구조 요소 - getStructuringElement()](#2-구조-요소---getstructuringelement)
3. [침식 - erode()](#3-침식---erode)
4. [팽창 - dilate()](#4-팽창---dilate)
5. [열기와 닫기 - morphologyEx()](#5-열기와-닫기---morphologyex)
6. [그래디언트, 탑햇, 블랙햇](#6-그래디언트-탑햇-블랙햇)
7. [실전 응용](#7-실전-응용)
8. [연습 문제](#8-연습-문제)
9. [다음 단계](#9-다음-단계)
10. [참고 자료](#10-참고-자료)

---

## 1. 모폴로지 연산 개요

### 모폴로지란?

```
┌─────────────────────────────────────────────────────────────────┐
│                     모폴로지 연산 개요                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   모폴로지(Morphology) = 형태학                                  │
│   이미지의 형태(shape)를 기반으로 하는 연산                       │
│                                                                 │
│   주요 용도:                                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  1. 노이즈 제거     - 작은 점 노이즈 제거                 │   │
│   │  2. 홀 채우기       - 객체 내부의 구멍 메우기              │   │
│   │  3. 객체 분리       - 붙어있는 객체들 분리                 │   │
│   │  4. 객체 연결       - 떨어진 부분들 연결                   │   │
│   │  5. 엣지 검출       - 모폴로지 그래디언트                  │   │
│   │  6. 스켈레톤화     - 객체의 뼈대 추출                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   기본 연산:                                                     │
│   - 침식 (Erosion): 객체 축소                                   │
│   - 팽창 (Dilation): 객체 확장                                  │
│                                                                 │
│   조합 연산:                                                     │
│   - 열기 (Opening) = 침식 → 팽창                                │
│   - 닫기 (Closing) = 팽창 → 침식                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 작동 원리

모폴로지 연산은 **구조 요소(Structuring Element)**라는 작은 마스크를 이미지 위로 이동시키며 픽셀 값을 결정합니다.

---

## 2. 구조 요소 - getStructuringElement()

### 구조 요소란?

```
┌─────────────────────────────────────────────────────────────────┐
│                        구조 요소                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   구조 요소 = 커널처럼 연산에 사용되는 작은 이진 행렬              │
│                                                                 │
│   주요 형태:                                                     │
│                                                                 │
│   MORPH_RECT (직사각형)    MORPH_CROSS (십자)    MORPH_ELLIPSE  │
│   ┌───┬───┬───┐           ┌───┬───┬───┐        ┌───┬───┬───┐  │
│   │ 1 │ 1 │ 1 │           │ 0 │ 1 │ 0 │        │ 0 │ 1 │ 0 │  │
│   ├───┼───┼───┤           ├───┼───┼───┤        ├───┼───┼───┤  │
│   │ 1 │ 1 │ 1 │           │ 1 │ 1 │ 1 │        │ 1 │ 1 │ 1 │  │
│   ├───┼───┼───┤           ├───┼───┼───┤        ├───┼───┼───┤  │
│   │ 1 │ 1 │ 1 │           │ 0 │ 1 │ 0 │        │ 0 │ 1 │ 0 │  │
│   └───┴───┴───┘           └───┴───┴───┘        └───┴───┴───┘  │
│   모든 방향 영향          수직/수평만 영향      타원형 영향      │
│                                                                 │
│   크기에 따른 효과:                                              │
│   - 작은 크기 (3x3): 세밀한 처리                                │
│   - 큰 크기 (7x7, 9x9): 강한 효과                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 구조 요소 생성

```python
import cv2
import numpy as np

# getStructuringElement(shape, ksize, anchor=(-1,-1))
# shape: 구조 요소 형태
# ksize: (width, height) 크기
# anchor: 기준점 (기본값: 중심)

# 직사각형
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
print("RECT (5x5):\n", rect_kernel)

# 십자형
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
print("\nCROSS (5x5):\n", cross_kernel)

# 타원형
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
print("\nELLIPSE (5x5):\n", ellipse_kernel)

# 커스텀 구조 요소
custom_kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)
```

### 구조 요소 시각화

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

shapes = [
    ('RECT', cv2.MORPH_RECT),
    ('CROSS', cv2.MORPH_CROSS),
    ('ELLIPSE', cv2.MORPH_ELLIPSE)
]

sizes = [(5, 5), (7, 7), (11, 11)]

fig, axes = plt.subplots(len(shapes), len(sizes), figsize=(12, 10))

for i, (name, shape) in enumerate(shapes):
    for j, size in enumerate(sizes):
        kernel = cv2.getStructuringElement(shape, size)
        axes[i, j].imshow(kernel, cmap='gray')
        axes[i, j].set_title(f'{name} {size}')
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()
```

---

## 3. 침식 - erode()

### 침식 연산 원리

```
┌─────────────────────────────────────────────────────────────────┐
│                         침식 (Erosion)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   원리:                                                         │
│   - 구조 요소를 이미지 위로 이동                                 │
│   - 구조 요소 아래의 모든 픽셀이 1일 때만 중심 픽셀을 1로         │
│   - 하나라도 0이면 중심 픽셀은 0                                 │
│                                                                 │
│   효과:                                                         │
│   - 전경(흰색) 영역 축소                                        │
│   - 작은 노이즈 제거                                            │
│   - 연결된 객체 분리                                            │
│   - 경계 부드럽게                                               │
│                                                                 │
│   예시:                                                         │
│   원본:               침식 후 (3x3):                            │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │ ████████████│     │   ████████  │                          │
│   │ ████████████│ ──▶ │   ████████  │                          │
│   │ ████████████│     │   ████████  │                          │
│   │ ████████████│     │             │                          │
│   └─────────────┘     └─────────────┘                          │
│   테두리 1픽셀씩 축소                                            │
│                                                                 │
│   노이즈 제거:                                                   │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │ ██  ■  ████ │     │ ██     ███  │                          │
│   │ ████  ████  │ ──▶ │  ██    ██   │  작은 점(■) 제거         │
│   │    ■  ████  │     │       ███   │                          │
│   └─────────────┘     └─────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 침식 사용법

```python
import cv2
import numpy as np

# 이진 이미지 준비
img = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 구조 요소 생성
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# erode(src, kernel, iterations=1)
# iterations: 반복 횟수 (기본값 1)

eroded_1 = cv2.erode(binary, kernel, iterations=1)
eroded_2 = cv2.erode(binary, kernel, iterations=2)
eroded_3 = cv2.erode(binary, kernel, iterations=3)

cv2.imshow('Original', binary)
cv2.imshow('Eroded 1x', eroded_1)
cv2.imshow('Eroded 2x', eroded_2)
cv2.imshow('Eroded 3x', eroded_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 침식 테스트 이미지 생성

```python
import cv2
import numpy as np

# 테스트 이미지 생성
img = np.zeros((300, 400), dtype=np.uint8)

# 큰 사각형
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

# 작은 노이즈 점들
for _ in range(50):
    x, y = np.random.randint(200, 350), np.random.randint(50, 250)
    cv2.circle(img, (x, y), 2, 255, -1)

# 연결된 원
cv2.circle(img, (280, 150), 40, 255, -1)
cv2.circle(img, (320, 150), 40, 255, -1)

# 침식 적용
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
eroded = cv2.erode(img, kernel, iterations=1)

cv2.imshow('Original', img)
cv2.imshow('Eroded', eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 4. 팽창 - dilate()

### 팽창 연산 원리

```
┌─────────────────────────────────────────────────────────────────┐
│                         팽창 (Dilation)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   원리:                                                         │
│   - 구조 요소를 이미지 위로 이동                                 │
│   - 구조 요소 아래의 픽셀 중 하나라도 1이면 중심 픽셀을 1로       │
│   - 침식의 반대 연산                                            │
│                                                                 │
│   효과:                                                         │
│   - 전경(흰색) 영역 확장                                        │
│   - 홀(구멍) 채우기                                             │
│   - 끊어진 부분 연결                                            │
│   - 객체 강조                                                   │
│                                                                 │
│   예시:                                                         │
│   원본:               팽창 후 (3x3):                            │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │   ██████    │     │ ████████████│                          │
│   │   ██████    │ ──▶ │ ████████████│                          │
│   │   ██████    │     │ ████████████│                          │
│   └─────────────┘     └─────────────┘                          │
│   테두리 1픽셀씩 확장                                            │
│                                                                 │
│   끊어진 부분 연결:                                              │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │ ██      ██  │     │ ████    ████│                          │
│   │ ██  ..  ██  │ ──▶ │ ██████████  │  점선이 연결됨           │
│   │ ██      ██  │     │ ████    ████│                          │
│   └─────────────┘     └─────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 팽창 사용법

```python
import cv2
import numpy as np

# 이진 이미지 준비
img = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# dilate(src, kernel, iterations=1)
dilated_1 = cv2.dilate(binary, kernel, iterations=1)
dilated_2 = cv2.dilate(binary, kernel, iterations=2)
dilated_3 = cv2.dilate(binary, kernel, iterations=3)

cv2.imshow('Original', binary)
cv2.imshow('Dilated 1x', dilated_1)
cv2.imshow('Dilated 2x', dilated_2)
cv2.imshow('Dilated 3x', dilated_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 침식과 팽창 비교

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 테스트 이미지
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
cv2.circle(img, (100, 100), 20, 0, -1)  # 내부 구멍

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

eroded = cv2.erode(img, kernel, iterations=1)
dilated = cv2.dilate(img, kernel, iterations=1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')

axes[1].imshow(eroded, cmap='gray')
axes[1].set_title('Eroded (Shrink)')

axes[2].imshow(dilated, cmap='gray')
axes[2].set_title('Dilated (Expand)')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 5. 열기와 닫기 - morphologyEx()

### 열기 (Opening)

```
┌─────────────────────────────────────────────────────────────────┐
│                      열기 (Opening)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   열기 = 침식 → 팽창                                            │
│                                                                 │
│   효과:                                                         │
│   - 작은 노이즈(점) 제거                                        │
│   - 객체의 전체 크기는 대략 유지                                 │
│   - 가느다란 연결부 끊기                                        │
│                                                                 │
│   원본        침식         팽창 (열기 결과)                      │
│   ┌──────┐    ┌──────┐    ┌──────┐                              │
│   │██ ■ █│    │█     │    │██   █│                              │
│   │██████│ ─▶ │ ████ │ ─▶ │██████│                              │
│   │  ■ ██│    │    █ │    │    ██│                              │
│   └──────┘    └──────┘    └──────┘                              │
│   작은 점(■) 제거됨                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 닫기 (Closing)

```
┌─────────────────────────────────────────────────────────────────┐
│                      닫기 (Closing)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   닫기 = 팽창 → 침식                                            │
│                                                                 │
│   효과:                                                         │
│   - 작은 구멍(홀) 채우기                                        │
│   - 객체의 전체 크기는 대략 유지                                 │
│   - 끊어진 부분 연결                                            │
│                                                                 │
│   원본        팽창         침식 (닫기 결과)                      │
│   ┌──────┐    ┌──────┐    ┌──────┐                              │
│   │██████│    │██████│    │██████│                              │
│   │██○ ██│ ─▶ │██████│ ─▶ │██████│                              │
│   │██████│    │██████│    │██████│                              │
│   └──────┘    └──────┘    └──────┘                              │
│   내부 구멍(○) 채워짐                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### morphologyEx() 사용법

```python
import cv2
import numpy as np

img = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# morphologyEx(src, op, kernel, iterations=1)
# op: 연산 종류

# 열기 (Opening): 노이즈 제거
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# 닫기 (Closing): 홀 채우기
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 열기 후 닫기 (노이즈 제거 + 홀 채우기)
clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original', binary)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.imshow('Open + Close', clean)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 열기와 닫기 비교 테스트

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 테스트 이미지: 노이즈 + 구멍이 있는 사각형
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

# 노이즈 추가 (작은 점들)
noise = img.copy()
for _ in range(30):
    x, y = np.random.randint(10, 45), np.random.randint(10, 190)
    cv2.circle(noise, (x, y), 2, 255, -1)
for _ in range(30):
    x, y = np.random.randint(155, 190), np.random.randint(10, 190)
    cv2.circle(noise, (x, y), 2, 255, -1)

# 구멍 추가 (객체 내부)
holes = noise.copy()
for _ in range(10):
    x, y = np.random.randint(60, 140), np.random.randint(60, 140)
    cv2.circle(holes, (x, y), 3, 0, -1)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

opening = cv2.morphologyEx(holes, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, kernel)
both = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(holes, cmap='gray')
axes[0, 0].set_title('Original (Noise + Holes)')

axes[0, 1].imshow(opening, cmap='gray')
axes[0, 1].set_title('Opening (Noise Removed)')

axes[1, 0].imshow(closing, cmap='gray')
axes[1, 0].set_title('Closing (Holes Filled)')

axes[1, 1].imshow(both, cmap='gray')
axes[1, 1].set_title('Open + Close')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 6. 그래디언트, 탑햇, 블랙햇

### 모폴로지 그래디언트

```
┌─────────────────────────────────────────────────────────────────┐
│                   모폴로지 그래디언트                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   그래디언트 = 팽창 - 침식                                       │
│                                                                 │
│   효과: 객체의 윤곽선(경계) 추출                                 │
│                                                                 │
│   원본              팽창              침식                       │
│   ┌──────┐         ┌──────┐         ┌──────┐                   │
│   │ ████ │         │██████│         │  ██  │                   │
│   │ ████ │    -    │██████│    =    │  ██  │                   │
│   │ ████ │         │██████│         │  ██  │                   │
│   └──────┘         └──────┘         └──────┘                   │
│                                                                 │
│   그래디언트 결과:                                               │
│   ┌──────┐                                                      │
│   │ ████ │  → 외곽선만 남음                                     │
│   │ █  █ │                                                      │
│   │ ████ │                                                      │
│   └──────┘                                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 탑햇과 블랙햇

```
┌─────────────────────────────────────────────────────────────────┐
│                    탑햇 / 블랙햇                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   탑햇 (Top Hat) = 원본 - 열기                                   │
│   - 밝은 영역에서 작은 밝은 부분 추출                            │
│   - 배경보다 밝은 작은 객체 검출                                 │
│                                                                 │
│   블랙햇 (Black Hat) = 닫기 - 원본                               │
│   - 어두운 영역에서 작은 어두운 부분 추출                        │
│   - 배경보다 어두운 작은 구멍/객체 검출                          │
│                                                                 │
│   활용:                                                         │
│   - 조명이 불균일한 이미지 보정                                  │
│   - 문서 이미지의 그림자 제거                                    │
│   - 작은 결함 검출                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 구현 및 사용

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

# 모폴로지 그래디언트
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# 탑햇
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# 블랙햇
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# 수동 계산 (확인용)
dilated = cv2.dilate(img, kernel)
eroded = cv2.erode(img, kernel)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

gradient_manual = dilated - eroded
tophat_manual = img - opening
blackhat_manual = closing - img

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')

axes[0, 1].imshow(gradient, cmap='gray')
axes[0, 1].set_title('Gradient (Edge)')

axes[0, 2].imshow(tophat, cmap='gray')
axes[0, 2].set_title('Top Hat (Bright spots)')

axes[1, 0].imshow(blackhat, cmap='gray')
axes[1, 0].set_title('Black Hat (Dark spots)')

# 탑햇 + 블랙햇으로 대비 향상
enhanced = cv2.add(img, tophat)
enhanced = cv2.subtract(enhanced, blackhat)
axes[1, 1].imshow(enhanced, cmap='gray')
axes[1, 1].set_title('Enhanced (Top+Black Hat)')

for ax in axes.flatten():
    ax.axis('off')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
```

### 모든 모폴로지 연산 정리

```python
import cv2

# morphologyEx()에서 사용 가능한 연산 목록
operations = {
    cv2.MORPH_ERODE: "침식 (Erode)",
    cv2.MORPH_DILATE: "팽창 (Dilate)",
    cv2.MORPH_OPEN: "열기 (Open = Erode + Dilate)",
    cv2.MORPH_CLOSE: "닫기 (Close = Dilate + Erode)",
    cv2.MORPH_GRADIENT: "그래디언트 (Dilate - Erode)",
    cv2.MORPH_TOPHAT: "탑햇 (Src - Open)",
    cv2.MORPH_BLACKHAT: "블랙햇 (Close - Src)",
    cv2.MORPH_HITMISS: "히트미스 (패턴 매칭)"
}

for op, name in operations.items():
    print(f"{op}: {name}")
```

---

## 7. 실전 응용

### 노이즈 제거 파이프라인

```python
import cv2
import numpy as np

def remove_noise_morphology(binary_img, noise_size=3):
    """
    모폴로지 연산으로 노이즈 제거

    Parameters:
    - binary_img: 이진 이미지
    - noise_size: 제거할 노이즈의 최대 크기
    """
    # 커널 크기 = 노이즈 크기 * 2 + 1
    kernel_size = noise_size * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    # 열기로 작은 점 노이즈 제거
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # 닫기로 작은 구멍 채우기
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


# 사용 예
img = cv2.imread('noisy_document.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
cleaned = remove_noise_morphology(binary, noise_size=2)
```

### 객체 분리

```python
import cv2
import numpy as np

def separate_objects(binary_img, erosion_iterations=3):
    """
    붙어있는 객체들을 분리
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 침식으로 객체 축소 (연결부 끊기)
    eroded = cv2.erode(binary_img, kernel, iterations=erosion_iterations)

    # 거리 변환으로 중심점 찾기 (선택적)
    dist_transform = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)

    return eroded, sure_fg


# 사용 예
img = cv2.imread('connected_circles.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
separated, centers = separate_objects(binary)
```

### 문서 이미지 전처리

```python
import cv2
import numpy as np

def preprocess_document(img):
    """
    문서 이미지 전처리 (그림자 제거 + 이진화)
    """
    # 그레이스케일 변환
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 탑햇으로 밝은 배경 추출
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    # 블랙햇으로 그림자/어두운 부분 보정
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 원본에서 블랙햇 빼기 (그림자 제거 효과)
    no_shadow = cv2.add(gray, blackhat)

    # 적응형 이진화
    binary = cv2.adaptiveThreshold(
        no_shadow, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 15
    )

    # 노이즈 제거
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)

    return binary


# 사용 예
img = cv2.imread('document_with_shadow.jpg')
result = preprocess_document(img)
```

### 스켈레톤화 (Skeletonization)

```python
import cv2
import numpy as np

def skeletonize(img):
    """
    모폴로지 연산으로 스켈레톤(뼈대) 추출
    """
    skeleton = np.zeros_like(img)
    temp = img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # 열기 연산
        opened = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)

        # 차이 계산
        diff = cv2.subtract(temp, opened)

        # 침식
        temp = cv2.erode(temp, kernel)

        # 스켈레톤에 추가
        skeleton = cv2.bitwise_or(skeleton, diff)

        # 더 이상 흰색 픽셀이 없으면 종료
        if cv2.countNonZero(temp) == 0:
            break

    return skeleton


# 사용 예
img = cv2.imread('character.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
skeleton = skeletonize(binary)
```

---

## 8. 연습 문제

### 연습 1: 구조 요소 효과 비교

동일한 이진 이미지에 대해 세 가지 구조 요소(RECT, CROSS, ELLIPSE)를 사용하여 침식과 팽창을 적용하고, 결과의 차이를 분석하세요.

### 연습 2: 문자 두께 조절

손글씨 이미지에서 문자의 두께를 조절하는 함수를 작성하세요:
- 양수 값: 팽창으로 두껍게
- 음수 값: 침식으로 얇게

```python
def adjust_stroke_width(img, amount):
    """
    amount > 0: 두껍게
    amount < 0: 얇게
    """
    pass
```

### 연습 3: 경계 추출 비교

다음 세 가지 방법으로 객체의 경계를 추출하고 비교하세요:
1. 모폴로지 그래디언트
2. Canny 엣지 검출
3. findContours

### 연습 4: 점자 인식 전처리

점자 이미지에서 각 점을 개별적으로 검출하기 위한 전처리 파이프라인을 설계하세요. (힌트: 침식으로 점들을 분리)

### 연습 5: 세포 분리 (Watershed 전처리)

현미경 세포 이미지에서 붙어있는 세포들을 분리하기 위한 전처리를 구현하세요:
1. 이진화
2. 노이즈 제거 (열기/닫기)
3. 확실한 배경 영역 찾기 (팽창)
4. 확실한 전경 영역 찾기 (거리 변환 + 임계값)

---

## 9. 다음 단계

[07_Thresholding.md](./07_Thresholding.md)에서 다양한 이진화 방법과 임계처리 기법을 학습합니다!

**다음에 배울 내용**:
- 전역 임계처리 (`cv2.threshold`)
- OTSU 자동 임계값
- 적응형 임계처리
- HSV 기반 임계처리

---

## 10. 참고 자료

### 공식 문서

- [erode() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511f2fb1c)
- [dilate() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c)
- [morphologyEx() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f)
- [getStructuringElement() 문서](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac342a1bb6eabf6f55c803b09268e36dc)

### 관련 학습 자료

| 폴더 | 관련 내용 |
|------|----------|
| [05_Image_Filtering.md](./05_Image_Filtering.md) | 필터링 기초 |
| [09_Contours.md](./09_Contours.md) | 전처리 후 윤곽선 검출 |

### 추가 참고

- [모폴로지 연산 튜토리얼](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [수학적 모폴로지 이론](https://en.wikipedia.org/wiki/Mathematical_morphology)

