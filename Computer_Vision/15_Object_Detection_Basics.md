# 객체 검출 기초 (Object Detection Basics)

## 개요

이미지에서 특정 객체를 찾아내는 객체 검출의 기초 방법들을 학습합니다. 템플릿 매칭, Haar Cascade, HOG+SVM 등 전통적인 객체 검출 기법의 원리와 구현 방법을 익힙니다.

**난이도**: ⭐⭐⭐

**선수 지식**: 이미지 필터링, 엣지 검출, 특징점 검출

---

## 목차

1. [템플릿 매칭 (Template Matching)](#1-템플릿-매칭-template-matching)
2. [템플릿 매칭 방법 비교](#2-템플릿-매칭-방법-비교)
3. [다중 스케일 템플릿 매칭](#3-다중-스케일-템플릿-매칭)
4. [Haar Cascade 분류기](#4-haar-cascade-분류기)
5. [CascadeClassifier 사용법](#5-cascadeclassifier-사용법)
6. [HOG + SVM 보행자 검출](#6-hog--svm-보행자-검출)
7. [연습 문제](#7-연습-문제)

---

## 1. 템플릿 매칭 (Template Matching)

### 기본 개념

```
템플릿 매칭: 작은 템플릿 이미지를 큰 이미지 위에서
            슬라이딩하며 유사도를 계산하는 방법

┌─────────────────────────────────┐
│  원본 이미지                     │
│    ┌─────────────────────┐      │
│    │                     │      │
│    │    ┌────┐           │      │
│    │    │ T  │ ← 템플릿  │      │
│    │    └────┘   위치 탐색│      │
│    │                     │      │
│    └─────────────────────┘      │
│                                 │
│  결과: 각 위치에서의 유사도 맵   │
└─────────────────────────────────┘
```

### matchTemplate() 기본 사용

```python
import cv2
import numpy as np

# 이미지와 템플릿 로드
img = cv2.imread('image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)

# 템플릿 크기
h, w = template.shape

# 템플릿 매칭 수행
result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

# 최대/최소 위치 찾기
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# TM_CCOEFF_NORMED는 최댓값이 최적 매칭
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# 결과 시각화
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
cv2.imshow('Detected', img)
cv2.waitKey(0)
```

### 템플릿 매칭 결과 이해

```
원본 이미지 (W x H)     템플릿 (w x h)     결과 이미지
┌───────────────┐       ┌───┐            ┌───────────┐
│               │       │ T │            │           │
│       W       │   +   │w×h│     =      │ (W-w+1)   │
│               │       └───┘            │   ×       │
│       H       │                        │ (H-h+1)   │
│               │                        │           │
└───────────────┘                        └───────────┘

결과 이미지의 각 픽셀 = 해당 위치에서의 매칭 점수
```

---

## 2. 템플릿 매칭 방법 비교

### 매칭 방법 종류

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)

# 6가지 매칭 방법
methods = [
    ('TM_SQDIFF', cv2.TM_SQDIFF),           # 제곱 차이
    ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED),  # 정규화된 제곱 차이
    ('TM_CCORR', cv2.TM_CCORR),             # 상관관계
    ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),   # 정규화된 상관관계
    ('TM_CCOEFF', cv2.TM_CCOEFF),           # 상관계수
    ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED)  # 정규화된 상관계수
]

h, w = template.shape

for name, method in methods:
    result = cv2.matchTemplate(img, template, method)

    # SQDIFF는 최솟값이 최적, 나머지는 최댓값이 최적
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = min_loc
    else:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc

    print(f"{name}: 위치={top_left}, 점수={max_val:.4f}")
```

### 방법별 특징

```
┌────────────────────┬─────────────────────────────────────────┐
│      방법          │                  특징                    │
├────────────────────┼─────────────────────────────────────────┤
│ TM_SQDIFF          │ 제곱 차이 합. 0에 가까울수록 좋음         │
│                    │ 조명 변화에 민감                         │
├────────────────────┼─────────────────────────────────────────┤
│ TM_SQDIFF_NORMED   │ 정규화된 제곱 차이. 0~1 범위             │
│                    │ 0에 가까울수록 좋음                      │
├────────────────────┼─────────────────────────────────────────┤
│ TM_CCORR           │ 상관관계. 값이 클수록 좋음                │
│                    │ 밝은 영역에 편향될 수 있음               │
├────────────────────┼─────────────────────────────────────────┤
│ TM_CCORR_NORMED    │ 정규화된 상관관계. 0~1 범위              │
│                    │ 값이 클수록 좋음                         │
├────────────────────┼─────────────────────────────────────────┤
│ TM_CCOEFF          │ 상관계수. 평균을 빼서 조명 변화에 강함    │
│                    │ 값이 클수록 좋음                         │
├────────────────────┼─────────────────────────────────────────┤
│ TM_CCOEFF_NORMED   │ 정규화된 상관계수. -1~1 범위             │
│                    │ 1에 가까울수록 좋음. 가장 널리 사용       │
└────────────────────┴─────────────────────────────────────────┘
```

### 다중 객체 검출

```python
import cv2
import numpy as np

def find_multiple_matches(img, template, threshold=0.8):
    """여러 개의 동일한 객체 검출"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) \
                    if len(template.shape) == 3 else template

    h, w = template_gray.shape

    # 템플릿 매칭
    result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # 임계값 이상인 위치 찾기
    locations = np.where(result >= threshold)

    # 결과 그리기
    img_result = img.copy()
    matches = []

    for pt in zip(*locations[::-1]):  # x, y 순서로 변환
        # Non-Maximum Suppression (간단 버전)
        is_new = True
        for existing in matches:
            if abs(pt[0] - existing[0]) < w//2 and abs(pt[1] - existing[1]) < h//2:
                is_new = False
                break

        if is_new:
            matches.append(pt)
            cv2.rectangle(img_result, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    print(f"검출된 객체 수: {len(matches)}")
    return img_result, matches

# 사용 예
img = cv2.imread('coins.jpg')
template = cv2.imread('coin_template.jpg')
result, locations = find_multiple_matches(img, template, threshold=0.85)
```

---

## 3. 다중 스케일 템플릿 매칭

### 문제점과 해결책

```
문제: 템플릿 매칭은 크기 변화에 취약
     원본과 템플릿의 크기가 다르면 검출 실패

해결: 다양한 스케일에서 매칭 수행

원본 이미지       다양한 크기의 템플릿
┌─────────┐       ┌──┐  ┌───┐  ┌────┐
│   ?     │       │T │  │ T │  │ T  │
│         │   ×   └──┘  └───┘  └────┘
│         │       작음   중간   크기
└─────────┘

또는

다양한 크기의 원본   템플릿
┌─────────┐
│         │
│         │         ┌───┐
└─────────┘         │ T │
┌───────┐    ×     └───┘
│       │
└───────┘
```

### 다중 스케일 매칭 구현

```python
import cv2
import numpy as np

def multi_scale_template_matching(img, template, scale_range=(0.5, 1.5),
                                  scale_step=0.1, method=cv2.TM_CCOEFF_NORMED):
    """다중 스케일 템플릿 매칭"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) \
                    if len(template.shape) == 3 else template

    best_match = None
    best_val = -1
    best_scale = 1.0

    th, tw = template_gray.shape

    # 다양한 스케일에서 매칭
    for scale in np.arange(scale_range[0], scale_range[1] + scale_step, scale_step):
        # 템플릿 크기 조정
        new_w = int(tw * scale)
        new_h = int(th * scale)

        # 이미지보다 큰 템플릿은 스킵
        if new_w > img_gray.shape[1] or new_h > img_gray.shape[0]:
            continue

        scaled_template = cv2.resize(template_gray, (new_w, new_h))

        # 템플릿 매칭
        result = cv2.matchTemplate(img_gray, scaled_template, method)

        # 최댓값 찾기
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            if best_match is None or max_val < best_val:
                best_val = max_val
                best_match = min_loc
                best_scale = scale
        else:
            if max_val > best_val:
                best_val = max_val
                best_match = max_loc
                best_scale = scale

    # 결과 시각화
    if best_match is not None:
        result_img = img.copy()
        top_left = best_match
        bottom_right = (int(top_left[0] + tw * best_scale),
                       int(top_left[1] + th * best_scale))
        cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)

        print(f"최적 스케일: {best_scale:.2f}")
        print(f"매칭 점수: {best_val:.4f}")
        print(f"위치: {top_left}")

        return result_img, best_match, best_scale, best_val

    return img, None, None, None

# 사용 예
img = cv2.imread('scene.jpg')
template = cv2.imread('object.jpg')
result, loc, scale, score = multi_scale_template_matching(
    img, template,
    scale_range=(0.3, 2.0),
    scale_step=0.05
)
```

### 피라미드 기반 다중 스케일 매칭

```python
def pyramid_template_matching(img, template, levels=5, scale_factor=0.75):
    """이미지 피라미드를 이용한 다중 스케일 매칭"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) \
                    if len(template.shape) == 3 else template

    best_result = {
        'location': None,
        'value': -1,
        'scale': 1.0,
        'size': template_gray.shape
    }

    current_scale = 1.0

    for level in range(levels):
        # 현재 스케일에서의 이미지 크기
        scaled_img = cv2.resize(img_gray, None,
                                fx=current_scale, fy=current_scale)

        # 템플릿이 이미지보다 크면 중단
        if (scaled_img.shape[0] < template_gray.shape[0] or
            scaled_img.shape[1] < template_gray.shape[1]):
            break

        # 템플릿 매칭
        result = cv2.matchTemplate(scaled_img, template_gray,
                                   cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_result['value']:
            # 원본 이미지 좌표로 변환
            orig_loc = (int(max_loc[0] / current_scale),
                       int(max_loc[1] / current_scale))
            best_result = {
                'location': orig_loc,
                'value': max_val,
                'scale': current_scale,
                'size': (int(template_gray.shape[1] / current_scale),
                        int(template_gray.shape[0] / current_scale))
            }

        current_scale *= scale_factor

    return best_result

# 사용 예
img = cv2.imread('scene.jpg')
template = cv2.imread('object.jpg')
result = pyramid_template_matching(img, template, levels=8)

if result['location']:
    img_result = img.copy()
    x, y = result['location']
    w, h = result['size']
    cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(f"검출 위치: {result['location']}")
    print(f"검출 스케일: {result['scale']:.3f}")
    print(f"매칭 점수: {result['value']:.4f}")
```

---

## 4. Haar Cascade 분류기

### Haar 특징 이해

```
Haar-like 특징: 밝은 영역과 어두운 영역의 차이를 이용

기본 Haar 특징들:
┌───────────────────────────────────────────────────────┐
│                                                       │
│   Edge features (엣지 특징)                           │
│   ┌────┬────┐    ┌────┐                              │
│   │████│    │    │████│                              │
│   │████│    │    ├────┤                              │
│   └────┴────┘    │    │                              │
│                  └────┘                              │
│                                                       │
│   Line features (선 특징)                             │
│   ┌────┬────┬────┐    ┌────┐                         │
│   │████│    │████│    │████│                         │
│   └────┴────┴────┘    ├────┤                         │
│                       │    │                         │
│                       ├────┤                         │
│                       │████│                         │
│                       └────┘                         │
│                                                       │
│   Center-surround features (중심-주변 특징)          │
│   ┌────┬────┬────┐                                   │
│   │████│    │████│                                   │
│   ├────┼────┼────┤                                   │
│   │████│    │████│                                   │
│   └────┴────┴────┘                                   │
│                                                       │
│   ████ = 검은 영역 (합산 후 빼기)                    │
│   빈칸 = 흰 영역 (합산)                              │
│                                                       │
│   특징값 = Σ(흰 영역) - Σ(검은 영역)                 │
└───────────────────────────────────────────────────────┘
```

### Integral Image (적분 이미지)

```
적분 이미지: 특징 계산을 O(1)로 만드는 기법

원본 이미지           적분 이미지
┌───┬───┬───┐        ┌───┬───┬───┐
│ 1 │ 2 │ 3 │        │ 1 │ 3 │ 6 │
├───┼───┼───┤   →    ├───┼───┼───┤
│ 4 │ 5 │ 6 │        │ 5 │12 │21 │
├───┼───┼───┤        ├───┼───┼───┤
│ 7 │ 8 │ 9 │        │12 │27 │45 │
└───┴───┴───┘        └───┴───┴───┘

적분 이미지 계산:
ii(x,y) = Σ i(x',y')  for x'≤x, y'≤y

영역 합 계산 (4번의 배열 접근으로 가능):
A ───── B
│       │
│  영역 │
│       │
C ───── D

영역 합 = ii(D) - ii(B) - ii(C) + ii(A)
```

### Cascade 구조

```
Cascade (캐스케이드): 단계적 분류기

이미지 윈도우
    │
    ▼
┌─────────┐    NO (빠른 거부)
│ Stage 1 │ ──────────────────→ 비객체
│ (간단)  │
└────┬────┘
     │ YES
     ▼
┌─────────┐    NO
│ Stage 2 │ ──────────────────→ 비객체
│         │
└────┬────┘
     │ YES
     ▼
    ...
     │
     ▼
┌─────────┐    NO
│ Stage N │ ──────────────────→ 비객체
│ (복잡)  │
└────┬────┘
     │ YES
     ▼
   객체!

장점: 대부분의 비객체를 초기 단계에서 빠르게 제거
```

---

## 5. CascadeClassifier 사용법

### 기본 사용법

```python
import cv2

# Haar Cascade 분류기 로드
# OpenCV에 포함된 사전 학습된 분류기 사용
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# 이미지 로드
img = cv2.imread('people.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = face_cascade.detectMultiScale(
    gray,           # 입력 이미지 (그레이스케일)
    scaleFactor=1.1, # 이미지 축소 비율
    minNeighbors=5,  # 최소 이웃 수 (높을수록 엄격)
    minSize=(30, 30), # 최소 객체 크기
    maxSize=(300, 300) # 최대 객체 크기
)

# 검출 결과 그리기
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 얼굴 영역에서 눈 검출
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

print(f"검출된 얼굴 수: {len(faces)}")
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
```

### detectMultiScale 매개변수

```
detectMultiScale(image, scaleFactor, minNeighbors, ...)

┌─────────────────────────────────────────────────────────────┐
│ scaleFactor: 각 스케일에서의 이미지 축소 비율               │
│                                                             │
│   scaleFactor = 1.1 (기본값)                               │
│   ┌─────────┐                                              │
│   │ 100x100 │ → 91x91 → 83x83 → 75x75 → ...               │
│   └─────────┘                                              │
│   작을수록 더 정밀하지만 느림                               │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ minNeighbors: 객체로 인정하기 위한 최소 이웃 검출 수        │
│                                                             │
│   minNeighbors = 3                                         │
│   ┌───────────────┐                                        │
│   │   ┌─┐ ┌─┐     │ → 2개 검출 → 무시 (< 3)              │
│   │   └─┘ └─┘     │                                        │
│   └───────────────┘                                        │
│   높을수록 오검출 감소, 미검출 증가                         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ minSize, maxSize: 검출할 객체의 크기 범위                   │
│                                                             │
│   minSize=(30, 30)  maxSize=(300, 300)                     │
│   30x30 픽셀 미만이나 300x300 픽셀 초과는 무시              │
└─────────────────────────────────────────────────────────────┘
```

### 사용 가능한 Cascade 파일

```python
import cv2
import os

# 사용 가능한 Haar Cascade 파일 목록
cascade_dir = cv2.data.haarcascades
print("사용 가능한 Cascade 파일:")
for f in sorted(os.listdir(cascade_dir)):
    if f.endswith('.xml'):
        print(f"  - {f}")

# 주요 Cascade 파일:
# haarcascade_frontalface_default.xml  - 정면 얼굴
# haarcascade_frontalface_alt.xml      - 정면 얼굴 (대안)
# haarcascade_frontalface_alt2.xml     - 정면 얼굴 (대안 2)
# haarcascade_profileface.xml          - 측면 얼굴
# haarcascade_eye.xml                  - 눈
# haarcascade_eye_tree_eyeglasses.xml  - 안경 낀 눈
# haarcascade_smile.xml                - 웃음
# haarcascade_fullbody.xml             - 전신
# haarcascade_upperbody.xml            - 상체
# haarcascade_lowerbody.xml            - 하체
# haarcascade_frontalcatface.xml       - 고양이 얼굴
# haarcascade_russian_plate_number.xml - 러시아 차량 번호판
```

### 다중 Cascade 조합

```python
import cv2

class FaceFeatureDetector:
    """얼굴 특징 검출기"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml')

    def detect(self, img):
        """얼굴, 눈, 웃음 검출"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # 히스토그램 평활화

        results = []

        # 얼굴 검출
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5,
                                                    minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi_gray = gray[y:y+h, x:x+w]

            face_data = {
                'bbox': (x, y, w, h),
                'eyes': [],
                'smiling': False
            }

            # 얼굴 상단 절반에서 눈 검출
            eye_roi = face_roi_gray[0:h//2, :]
            eyes = self.eye_cascade.detectMultiScale(eye_roi, 1.1, 3,
                                                      minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                face_data['eyes'].append((x + ex, y + ey, ew, eh))

            # 얼굴 하단에서 웃음 검출
            smile_roi = face_roi_gray[h//2:, :]
            smiles = self.smile_cascade.detectMultiScale(smile_roi, 1.7, 20,
                                                          minSize=(25, 25))
            face_data['smiling'] = len(smiles) > 0

            results.append(face_data)

        return results

    def draw_results(self, img, results):
        """결과 시각화"""
        output = img.copy()

        for face in results:
            x, y, w, h = face['bbox']

            # 얼굴 사각형
            color = (0, 255, 0) if face['smiling'] else (255, 0, 0)
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)

            # 눈 원
            for (ex, ey, ew, eh) in face['eyes']:
                center = (ex + ew//2, ey + eh//2)
                radius = min(ew, eh) // 2
                cv2.circle(output, center, radius, (0, 255, 255), 2)

            # 웃음 상태 표시
            label = "Smiling :)" if face['smiling'] else "Neutral"
            cv2.putText(output, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return output

# 사용 예
detector = FaceFeatureDetector()
img = cv2.imread('group_photo.jpg')
results = detector.detect(img)
output = detector.draw_results(img, results)
cv2.imshow('Face Features', output)
```

---

## 6. HOG + SVM 보행자 검출

### HOG (Histogram of Oriented Gradients) 이해

```
HOG: 국소 영역의 기울기(gradient) 방향 분포를 특징으로 사용

1. 그레이스케일 변환

2. 기울기 계산
   ┌───────────────────────────────────────────┐
   │  Gx = 수평 기울기 (Sobel x)               │
   │  Gy = 수직 기울기 (Sobel y)               │
   │                                           │
   │  크기: G = √(Gx² + Gy²)                  │
   │  방향: θ = arctan(Gy/Gx)                 │
   └───────────────────────────────────────────┘

3. 셀 단위로 기울기 히스토그램 계산
   ┌─────────────────────────────────────────┐
   │  이미지를 8x8 픽셀 셀로 분할            │
   │  각 셀에서 방향 히스토그램 (9개 빈)     │
   │                                         │
   │  0°  20° 40° 60° 80° 100°120°140°160°  │
   │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐ │
   │  │   │███│   │   │█████│   │   │   │ │
   │  └───┴───┴───┴───┴───┴───┴───┴───┴───┘ │
   └─────────────────────────────────────────┘

4. 블록 정규화
   ┌─────────────────────────────────────────┐
   │  2x2 셀 = 1 블록                        │
   │  블록 내 히스토그램을 연결 후 정규화    │
   │                                         │
   │  ┌────┬────┐                            │
   │  │cell│cell│ → [36차원 특징 벡터]       │
   │  ├────┼────┤     (9 × 4 = 36)          │
   │  │cell│cell│                            │
   │  └────┴────┘                            │
   └─────────────────────────────────────────┘

5. 모든 블록의 특징을 연결하여 최종 HOG 디스크립터 생성
```

### HOG 보행자 검출기 사용

```python
import cv2
import numpy as np

# HOG 디스크립터 + SVM 분류기
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 이미지 로드
img = cv2.imread('street.jpg')
img = cv2.resize(img, None, fx=0.5, fy=0.5)  # 속도를 위해 축소

# 보행자 검출
# detectMultiScale 반환: (검출 영역, 신뢰도 가중치)
boxes, weights = hog.detectMultiScale(
    img,
    winStride=(8, 8),    # 윈도우 이동 간격
    padding=(4, 4),       # 패딩
    scale=1.05,           # 스케일 팩터
    hitThreshold=0,       # SVM 임계값
    finalThreshold=2.0    # 최종 그룹화 임계값
)

# 결과 그리기
for (x, y, w, h), weight in zip(boxes, weights):
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, f'{weight[0]:.2f}', (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

print(f"검출된 보행자 수: {len(boxes)}")
cv2.imshow('Pedestrian Detection', img)
cv2.waitKey(0)
```

### Non-Maximum Suppression (NMS)

```python
import cv2
import numpy as np

def non_max_suppression(boxes, scores, threshold=0.5):
    """Non-Maximum Suppression 구현"""
    if len(boxes) == 0:
        return []

    # 좌표를 float로 변환
    boxes = boxes.astype(np.float32)

    # 좌표 분리
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # 면적 계산
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 점수로 정렬 (내림차순)
    order = scores.flatten().argsort()[::-1]

    keep = []
    while order.size > 0:
        # 가장 높은 점수의 박스 선택
        i = order[0]
        keep.append(i)

        # 나머지 박스들과의 IoU 계산
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # IoU가 임계값보다 작은 박스만 유지
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep

# HOG 검출과 NMS 적용
def detect_pedestrians_with_nms(img, nms_threshold=0.3):
    """NMS를 적용한 보행자 검출"""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # 검출
    boxes, weights = hog.detectMultiScale(img, winStride=(8, 8),
                                          padding=(4, 4), scale=1.05)

    if len(boxes) == 0:
        return img, []

    # NMS 적용
    boxes = np.array(boxes)
    weights = np.array(weights)
    keep = non_max_suppression(boxes, weights, nms_threshold)

    # 결과 그리기
    result = img.copy()
    final_boxes = []

    for i in keep:
        x, y, w, h = boxes[i]
        final_boxes.append((x, y, w, h))
        cv2.rectangle(result, (int(x), int(y)), (int(x+w), int(y+h)),
                     (0, 255, 0), 2)

    return result, final_boxes

# 사용 예
img = cv2.imread('crowd.jpg')
result, detections = detect_pedestrians_with_nms(img)
print(f"NMS 후 검출 수: {len(detections)}")
```

### HOG 특징 시각화

```python
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure

def visualize_hog(img):
    """HOG 특징 시각화"""
    # 그레이스케일 변환
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 크기 조정 (64x128 - HOG 보행자 검출 표준 크기)
    resized = cv2.resize(gray, (64, 128))

    # scikit-image의 hog 사용 (시각화 포함)
    features, hog_image = hog(
        resized,
        orientations=9,        # 기울기 방향 빈 수
        pixels_per_cell=(8, 8),  # 셀 크기
        cells_per_block=(2, 2),  # 블록 내 셀 수
        visualize=True,
        block_norm='L2-Hys'
    )

    # 시각화를 위한 rescale
    hog_image_rescaled = exposure.rescale_intensity(hog_image,
                                                     out_range=(0, 255))
    hog_image_rescaled = hog_image_rescaled.astype(np.uint8)

    print(f"HOG 특징 벡터 크기: {features.shape[0]}")

    return hog_image_rescaled, features

# 사용 예 (scikit-image 설치 필요: pip install scikit-image)
# img = cv2.imread('person.jpg')
# hog_vis, features = visualize_hog(img)
# cv2.imshow('HOG Visualization', hog_vis)
```

### 커스텀 HOG + SVM 학습 (개념)

```python
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

def train_hog_svm_classifier(positive_samples, negative_samples):
    """HOG + SVM 분류기 학습 (개념 예제)"""

    # HOG 디스크립터 설정
    win_size = (64, 128)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                            cell_size, nbins)

    # 특징 추출
    features = []
    labels = []

    # Positive 샘플 (객체가 있는 이미지)
    for img in positive_samples:
        img_resized = cv2.resize(img, win_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        h = hog.compute(gray)
        features.append(h.flatten())
        labels.append(1)

    # Negative 샘플 (객체가 없는 이미지)
    for img in negative_samples:
        img_resized = cv2.resize(img, win_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        h = hog.compute(gray)
        features.append(h.flatten())
        labels.append(0)

    X = np.array(features)
    y = np.array(labels)

    # 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # SVM 학습
    clf = svm.LinearSVC(C=0.01)
    clf.fit(X_train, y_train)

    # 정확도 출력
    accuracy = clf.score(X_test, y_test)
    print(f"테스트 정확도: {accuracy:.4f}")

    return hog, clf

# 학습된 SVM을 HOGDescriptor에 설정하는 방법
def set_svm_detector(hog, clf):
    """학습된 SVM을 HOG 검출기에 설정"""
    # LinearSVC의 계수와 절편을 추출
    sv = clf.coef_.flatten()
    rho = -clf.intercept_[0]

    # HOG 디스크립터가 기대하는 형식으로 변환
    detector = np.append(sv, rho)

    hog.setSVMDetector(detector)
    return hog
```

---

## 7. 연습 문제

### 문제 1: 다중 템플릿 매칭

여러 종류의 템플릿을 동시에 매칭하는 프로그램을 작성하세요.

**요구사항**:
- 3개 이상의 서로 다른 템플릿 이미지 사용
- 각 템플릿에 대해 다른 색상으로 검출 결과 표시
- 각 템플릿의 매칭 점수 출력

<details>
<summary>힌트</summary>

```python
templates = [
    ('template1.jpg', (255, 0, 0)),   # 파란색
    ('template2.jpg', (0, 255, 0)),   # 녹색
    ('template3.jpg', (0, 0, 255))    # 빨간색
]

for template_path, color in templates:
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    # ... 매칭 및 그리기
```

</details>

### 문제 2: 회전 불변 템플릿 매칭

템플릿을 다양한 각도로 회전시켜 매칭하는 프로그램을 구현하세요.

**요구사항**:
- 템플릿을 0도부터 360도까지 10도 간격으로 회전
- 각 회전 각도에서 가장 높은 매칭 점수 기록
- 최적의 회전 각도와 위치 출력

<details>
<summary>힌트</summary>

```python
def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

for angle in range(0, 360, 10):
    rotated_template = rotate_image(template, angle)
    # 템플릿 매칭 수행
```

</details>

### 문제 3: 실시간 얼굴 검출 최적화

웹캠에서 실시간으로 얼굴을 검출하되, 30 FPS 이상을 유지하도록 최적화하세요.

**요구사항**:
- 프레임 크기 조절
- detectMultiScale 매개변수 최적화
- FPS 표시

<details>
<summary>힌트</summary>

```python
# 최적화 팁:
# 1. 프레임을 절반 크기로 축소
# 2. scaleFactor를 1.2~1.3으로 증가
# 3. minNeighbors를 3으로 낮춤
# 4. minSize를 적절히 설정

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    # 검출 후 좌표를 2배로 스케일
```

</details>

### 문제 4: HOG 시각화 도구

이미지의 HOG 특징을 실시간으로 시각화하는 프로그램을 작성하세요.

**요구사항**:
- 트랙바로 HOG 파라미터 조절 (cell_size, nbins)
- 원본 이미지와 HOG 시각화를 나란히 표시
- 특징 벡터의 차원 표시

<details>
<summary>힌트</summary>

```python
def on_trackbar(val):
    cell_size = cv2.getTrackbarPos('Cell Size', 'HOG')
    if cell_size < 4:
        cell_size = 4
    # HOG 재계산 및 시각화
```

</details>

### 문제 5: 자동차 번호판 검출기

Haar Cascade 또는 템플릿 매칭을 사용하여 자동차 번호판을 검출하는 프로그램을 구현하세요.

**요구사항**:
- 번호판 영역 검출
- 검출된 영역 크롭 및 저장
- 신뢰도 점수 표시

<details>
<summary>힌트</summary>

```python
# haarcascade_russian_plate_number.xml 또는
# 직접 학습한 cascade 사용

# 또는 번호판 특성을 이용한 검출:
# 1. 엣지 검출
# 2. 직사각형 윤곽선 검출
# 3. 가로세로 비율 필터링 (번호판은 보통 4:1 ~ 5:1)
```

</details>

---

## 다음 단계

- [16_Face_Detection.md](./16_Face_Detection.md) - dlib, face_recognition, 실시간 얼굴 인식

---

## 참고 자료

- [OpenCV Template Matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
- [OpenCV Cascade Classifier](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [HOG Tutorial - Learn OpenCV](https://learnopencv.com/histogram-of-oriented-gradients/)
- Dalal, N., & Triggs, B. (2005). "Histograms of Oriented Gradients for Human Detection"
- Viola, P., & Jones, M. (2001). "Rapid Object Detection using a Boosted Cascade of Simple Features"
