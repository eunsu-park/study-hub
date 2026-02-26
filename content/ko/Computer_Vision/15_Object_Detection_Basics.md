# 객체 검출 기초 (Object Detection Basics)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 템플릿 매칭(template matching)의 원리를 설명하고 OpenCV를 사용하여 다양한 유사도 메트릭(similarity metrics)으로 구현할 수 있다
2. 이미지에서 다양한 크기의 객체를 처리하기 위해 다중 스케일 템플릿 매칭(multi-scale template matching)을 적용할 수 있다
3. 하르 캐스케이드(Haar Cascade) 분류기의 동작 원리를 설명하고 사전 학습된 캐스케이드를 사용하여 객체를 검출할 수 있다
4. HOG(Histogram of Oriented Gradients) 특징 추출을 구현하고 보행자 검출을 위해 SVM 분류기와 결합할 수 있다
5. 템플릿 매칭, 하르 캐스케이드, HOG+SVM 접근 방법의 장단점을 비교할 수 있다
6. 주어진 검출 작업에 가장 적합한 고전적 방법을 선택하는 검출 파이프라인(detection pipeline)을 설계할 수 있다

---

## 개요

이미지에서 특정 객체를 찾아내는 객체 검출의 기초 방법들을 학습합니다. 템플릿 매칭, Haar Cascade, HOG+SVM 등 전통적인 객체 검출 기법의 원리와 구현 방법을 익힙니다.

고전적 검출 방법(classical detection methods)은 딥러닝과 함께 여전히 유효합니다. 학습 데이터가 필요 없고, 연산 자원이 제한된 엣지 디바이스(edge device)에서 효율적으로 동작하며, 해석이 가능(interpretable)합니다. 목표 외관이 명확히 정의되어 있고 신경망의 비용을 감당하기 어려울 때 올바른 선택입니다.

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
Template Matching: A method that slides a small template image
                  over a larger image and computes similarity

+-------------------------------+
|  Source Image                 |
|    +---------------------+    |
|    |                     |    |
|    |    +----+           |    |
|    |    | T  | <- Template|   |
|    |    +----+   position |   |
|    |           search     |   |
|    +---------------------+    |
|                               |
|  Result: Similarity map at    |
|          each position        |
+-------------------------------+
```

### matchTemplate() 기본 사용

```python
import cv2
import numpy as np

# Load image and template
img = cv2.imread('image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)

# Template size
h, w = template.shape

# TM_CCOEFF_NORMED is preferred: subtracting the mean makes it invariant to
# additive lighting changes, and normalization constrains scores to [-1, 1]
# so a threshold of 0.8 has consistent meaning regardless of image brightness
result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

# Find min/max locations
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# For TM_CCOEFF_NORMED, maximum value is best match
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# Visualize result
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
cv2.imshow('Detected', img)
cv2.waitKey(0)
```

### 템플릿 매칭 결과 이해

결과 맵(result map)의 크기가 `(W-w+1) × (H-h+1)`인 이유는, 템플릿이 원본 이미지 내에 완전히 들어가야 하기 때문입니다. 템플릿의 중심이 어떤 가장자리로부터 `w/2` 픽셀보다 가까이 배치될 수 없으므로, 가로로 유효한 위치는 정확히 `W-w+1`개, 세로로 유효한 위치는 `H-h+1`개가 됩니다.

```
Source Image (W x H)     Template (w x h)     Result Image
+---------------+       +---+            +-----------+
|               |       | T |            |           |
|       W       |   +   |w*h|     =      | (W-w+1)   |
|               |       +---+            |   x       |
|       H       |                        | (H-h+1)   |
|               |                        |           |
+---------------+                        +-----------+

Each pixel in result image = matching score at that location
```

---

## 2. 템플릿 매칭 방법 비교

### 매칭 방법 종류

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)

# 6 matching methods
methods = [
    ('TM_SQDIFF', cv2.TM_SQDIFF),           # Squared difference
    ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED),  # Normalized squared difference
    ('TM_CCORR', cv2.TM_CCORR),             # Cross correlation
    ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),   # Normalized cross correlation
    ('TM_CCOEFF', cv2.TM_CCOEFF),           # Correlation coefficient
    ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED)  # Normalized correlation coefficient
]

h, w = template.shape

for name, method in methods:
    result = cv2.matchTemplate(img, template, method)

    # SQDIFF uses minimum as best, others use maximum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = min_loc
    else:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc

    print(f"{name}: location={top_left}, score={max_val:.4f}")
```

### 방법별 특징

```
+--------------------+-----------------------------------------+
|      Method        |                  Characteristics         |
+--------------------+-----------------------------------------+
| TM_SQDIFF          | Sum of squared differences. Closer to   |
|                    | 0 is better. Sensitive to lighting      |
+--------------------+-----------------------------------------+
| TM_SQDIFF_NORMED   | Normalized squared difference. 0-1 range|
|                    | Closer to 0 is better                   |
+--------------------+-----------------------------------------+
| TM_CCORR           | Cross correlation. Higher is better     |
|                    | Can be biased towards bright regions    |
+--------------------+-----------------------------------------+
| TM_CCORR_NORMED    | Normalized cross correlation. 0-1 range |
|                    | Higher is better                        |
+--------------------+-----------------------------------------+
| TM_CCOEFF          | Correlation coefficient. Subtracts mean |
|                    | to handle lighting. Higher is better    |
+--------------------+-----------------------------------------+
| TM_CCOEFF_NORMED   | Normalized correlation coefficient.     |
|                    | -1 to 1 range. Closer to 1 is better.   |
|                    | Most widely used                        |
+--------------------+-----------------------------------------+
```

### 다중 객체 검출

```python
import cv2
import numpy as np

def find_multiple_matches(img, template, threshold=0.8):
    """Detect multiple identical objects"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) \
                    if len(template.shape) == 3 else template

    h, w = template_gray.shape

    # Template matching
    result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Threshold at 0.8+ selects only high-confidence hits; lower values
    # flood-fill the area around each true match with many overlapping detections
    locations = np.where(result >= threshold)

    # Draw results
    img_result = img.copy()
    matches = []

    for pt in zip(*locations[::-1]):  # Convert to x, y order
        # Greedy NMS: suppress any new candidate whose top-left corner falls
        # within half the template size of an already-accepted match,
        # preventing the same object from being counted multiple times
        is_new = True
        for existing in matches:
            if abs(pt[0] - existing[0]) < w//2 and abs(pt[1] - existing[1]) < h//2:
                is_new = False
                break

        if is_new:
            matches.append(pt)
            cv2.rectangle(img_result, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    print(f"Objects detected: {len(matches)}")
    return img_result, matches

# Usage example
img = cv2.imread('coins.jpg')
template = cv2.imread('coin_template.jpg')
result, locations = find_multiple_matches(img, template, threshold=0.85)
```

---

## 3. 다중 스케일 템플릿 매칭

### 문제점과 해결책

```
Problem: Template matching is vulnerable to scale changes
        Detection fails if source and template sizes differ

Solution: Perform matching at various scales

Source Image       Templates at various sizes
+---------+       +--+  +---+  +----+
|   ?     |       |T |  | T |  | T  |
|         |   x   +--+  +---+  +----+
|         |       small medium large
+---------+

Or

Source at various sizes   Template
+---------+
|         |
|         |         +---+
+---------+         | T |
+-------+    x     +---+
|       |
+-------+
```

### 다중 스케일 매칭 구현

```python
import cv2
import numpy as np

def multi_scale_template_matching(img, template, scale_range=(0.5, 1.5),
                                  scale_step=0.1, method=cv2.TM_CCOEFF_NORMED):
    """Multi-scale template matching"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) \
                    if len(template.shape) == 3 else template

    best_match = None
    best_val = -1
    best_scale = 1.0

    th, tw = template_gray.shape

    # Match at various scales
    for scale in np.arange(scale_range[0], scale_range[1] + scale_step, scale_step):
        # Resize the template rather than the source image so that
        # coordinates in the result map remain in the original image space,
        # avoiding a second coordinate transform after finding the best match
        new_w = int(tw * scale)
        new_h = int(th * scale)

        # Skip if template is larger than image
        if new_w > img_gray.shape[1] or new_h > img_gray.shape[0]:
            continue

        scaled_template = cv2.resize(template_gray, (new_w, new_h))

        # Template matching
        result = cv2.matchTemplate(img_gray, scaled_template, method)

        # Find maximum
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

    # Visualize result
    if best_match is not None:
        result_img = img.copy()
        top_left = best_match
        bottom_right = (int(top_left[0] + tw * best_scale),
                       int(top_left[1] + th * best_scale))
        cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)

        print(f"Optimal scale: {best_scale:.2f}")
        print(f"Match score: {best_val:.4f}")
        print(f"Location: {top_left}")

        return result_img, best_match, best_scale, best_val

    return img, None, None, None

# Usage example
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
    """Multi-scale matching using image pyramid"""
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
        # Image size at current scale
        scaled_img = cv2.resize(img_gray, None,
                                fx=current_scale, fy=current_scale)

        # Stop if template is larger than image
        if (scaled_img.shape[0] < template_gray.shape[0] or
            scaled_img.shape[1] < template_gray.shape[1]):
            break

        # Template matching
        result = cv2.matchTemplate(scaled_img, template_gray,
                                   cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_result['value']:
            # Convert to original image coordinates
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

# Usage example
img = cv2.imread('scene.jpg')
template = cv2.imread('object.jpg')
result = pyramid_template_matching(img, template, levels=8)

if result['location']:
    img_result = img.copy()
    x, y = result['location']
    w, h = result['size']
    cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(f"Detection location: {result['location']}")
    print(f"Detection scale: {result['scale']:.3f}")
    print(f"Match score: {result['value']:.4f}")
```

---

## 4. Haar Cascade 분류기

### Haar 특징 이해

```
Haar-like Features: Use difference between bright and dark regions

Basic Haar Features:
+-------------------------------------------------------+
|                                                       |
|   Edge features                                       |
|   +----+----+    +----+                               |
|   |####|    |    |####|                               |
|   |####|    |    +----+                               |
|   +----+----+    |    |                               |
|                  +----+                               |
|                                                       |
|   Line features                                       |
|   +----+----+----+    +----+                          |
|   |####|    |####|    |####|                          |
|   +----+----+----+    +----+                          |
|                       |    |                          |
|                       +----+                          |
|                       |####|                          |
|                       +----+                          |
|                                                       |
|   Center-surround features                            |
|   +----+----+----+                                    |
|   |####|    |####|                                    |
|   +----+----+----+                                    |
|   |####|    |####|                                    |
|   +----+----+----+                                    |
|                                                       |
|   #### = Black region (sum then subtract)             |
|   blank = White region (sum)                          |
|                                                       |
|   Feature value = sum(white regions) - sum(black)     |
+-------------------------------------------------------+
```

### Integral Image (적분 이미지)

```
Integral Image: Technique for O(1) feature computation

Original Image           Integral Image
+---+---+---+        +---+---+---+
| 1 | 2 | 3 |        | 1 | 3 | 6 |
+---+---+---+   ->   +---+---+---+
| 4 | 5 | 6 |        | 5 |12 |21 |
+---+---+---+        +---+---+---+
| 7 | 8 | 9 |        |12 |27 |45 |
+---+---+---+        +---+---+---+

Integral image computation:
ii(x,y) = sum i(x',y')  for x'<=x, y'<=y

Region sum (only 4 array accesses needed):
A ----- B
|       |
| region|
|       |
C ----- D

Region sum = ii(D) - ii(B) - ii(C) + ii(A)
```

### Cascade 구조

```
Cascade: Staged classifier

Image window
    |
    v
+---------+    NO (fast reject)
| Stage 1 | ------------------> Not object
| (simple)|
+----+----+
     | YES
     v
+---------+    NO
| Stage 2 | ------------------> Not object
|         |
+----+----+
     | YES
     v
    ...
     |
     v
+---------+    NO
| Stage N | ------------------> Not object
| (complex)|
+----+----+
     | YES
     v
   Object!

Advantage: Most non-objects rejected quickly at early stages
```

---

## 5. CascadeClassifier 사용법

### 기본 사용법

```python
import cv2

# Load Haar Cascade classifier
# Use pre-trained classifiers included with OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# Load image
img = cv2.imread('people.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,           # Input image (grayscale)
    scaleFactor=1.1, # Each pyramid level is 10% smaller; lower (1.05) is more thorough but slower
    minNeighbors=5,  # Higher values reduce false positives at the cost of missing real detections
    minSize=(30, 30),  # Skipping windows smaller than 30×30 avoids wasting time
                       # on regions too small to be a face in the expected scene scale
    maxSize=(300, 300) # Upper bound prevents the cascade from producing giant false
                       # positives when a high-contrast region spans the whole frame
)

# Draw detection results
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Eye detection runs only inside the already-confirmed face ROI, not the full image.
    # This dramatically reduces the search space and eliminates false positives from
    # eye-like patterns (buttons, circles) outside face regions.
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

print(f"Faces detected: {len(faces)}")
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
```

### detectMultiScale 매개변수

```
detectMultiScale(image, scaleFactor, minNeighbors, ...)

+-------------------------------------------------------------+
| scaleFactor: Image reduction ratio at each scale            |
|                                                             |
|   scaleFactor = 1.1 (default)                               |
|   +---------+                                               |
|   | 100x100 | -> 91x91 -> 83x83 -> 75x75 -> ...            |
|   +---------+                                               |
|   Smaller = more precise but slower                         |
|                                                             |
+-------------------------------------------------------------+
| minNeighbors: Minimum detection count to accept as object   |
|                                                             |
|   minNeighbors = 3                                          |
|   +---------------+                                         |
|   |   +-+ +-+     | -> 2 detections -> Ignore (< 3)        |
|   |   +-+ +-+     |                                         |
|   +---------------+                                         |
|   Higher = fewer false positives, more missed detections    |
|                                                             |
+-------------------------------------------------------------+
| minSize, maxSize: Object size range to detect               |
|                                                             |
|   minSize=(30, 30)  maxSize=(300, 300)                      |
|   Ignore below 30x30 pixels or above 300x300 pixels         |
+-------------------------------------------------------------+
```

### 사용 가능한 Cascade 파일

```python
import cv2
import os

# List available Haar Cascade files
cascade_dir = cv2.data.haarcascades
print("Available Cascade files:")
for f in sorted(os.listdir(cascade_dir)):
    if f.endswith('.xml'):
        print(f"  - {f}")

# Major Cascade files:
# haarcascade_frontalface_default.xml  - Frontal face
# haarcascade_frontalface_alt.xml      - Frontal face (alternative)
# haarcascade_frontalface_alt2.xml     - Frontal face (alternative 2)
# haarcascade_profileface.xml          - Profile face
# haarcascade_eye.xml                  - Eyes
# haarcascade_eye_tree_eyeglasses.xml  - Eyes with glasses
# haarcascade_smile.xml                - Smile
# haarcascade_fullbody.xml             - Full body
# haarcascade_upperbody.xml            - Upper body
# haarcascade_lowerbody.xml            - Lower body
# haarcascade_frontalcatface.xml       - Cat face
# haarcascade_russian_plate_number.xml - Russian license plate
```

### 다중 Cascade 조합

```python
import cv2

class FaceFeatureDetector:
    """Face feature detector"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml')

    def detect(self, img):
        """Detect face, eyes, and smile"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Histogram equalization

        results = []

        # Detect faces first: the cascade order matters because eye and smile
        # detectors produce too many false positives on arbitrary image regions.
        # Running them only inside confirmed face bounding boxes (ROI) constrains
        # the problem and makes the pipeline both faster and more accurate.
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5,
                                                    minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi_gray = gray[y:y+h, x:x+w]

            face_data = {
                'bbox': (x, y, w, h),
                'eyes': [],
                'smiling': False
            }

            # Detect eyes in top half of face
            eye_roi = face_roi_gray[0:h//2, :]
            eyes = self.eye_cascade.detectMultiScale(eye_roi, 1.1, 3,
                                                      minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                face_data['eyes'].append((x + ex, y + ey, ew, eh))

            # Detect smile in bottom half of face
            smile_roi = face_roi_gray[h//2:, :]
            # minNeighbors=20 is intentionally high for smiles: the smile cascade
            # is noisier than the face cascade, so a strict consensus requirement
            # prevents every open mouth or shadow from being flagged as a smile
            smiles = self.smile_cascade.detectMultiScale(smile_roi, 1.7, 20,
                                                          minSize=(25, 25))
            face_data['smiling'] = len(smiles) > 0

            results.append(face_data)

        return results

    def draw_results(self, img, results):
        """Visualize results"""
        output = img.copy()

        for face in results:
            x, y, w, h = face['bbox']

            # Face rectangle
            color = (0, 255, 0) if face['smiling'] else (255, 0, 0)
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)

            # Eye circles
            for (ex, ey, ew, eh) in face['eyes']:
                center = (ex + ew//2, ey + eh//2)
                radius = min(ew, eh) // 2
                cv2.circle(output, center, radius, (0, 255, 255), 2)

            # Smile status
            label = "Smiling :)" if face['smiling'] else "Neutral"
            cv2.putText(output, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return output

# Usage example
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
HOG: Uses gradient direction distribution in local regions as features

1. Grayscale conversion

2. Gradient computation
   +-------------------------------------------+
   |  Gx = Horizontal gradient (Sobel x)       |
   |  Gy = Vertical gradient (Sobel y)         |
   |                                           |
   |  Magnitude: G = sqrt(Gx^2 + Gy^2)         |
   |  Direction: theta = arctan(Gy/Gx)         |
   +-------------------------------------------+

3. Compute gradient histogram per cell
   +-----------------------------------------+
   |  Divide image into 8x8 pixel cells      |
   |  Direction histogram per cell (9 bins)  |
   |                                         |
   |  0   20  40  60  80 100 120 140 160     |
   |  +---+---+---+---+---+---+---+---+---+  |
   |  |   |###|   |   |#####|   |   |   |  |
   |  +---+---+---+---+---+---+---+---+---+  |
   +-----------------------------------------+

4. Block normalization
   +-----------------------------------------+
   |  2x2 cells = 1 block                    |
   |  Concatenate histograms in block then   |
   |  normalize                              |
   |                                         |
   |  +----+----+                            |
   |  |cell|cell| -> [36-dim feature vector] |
   |  +----+----+     (9 x 4 = 36)           |
   |  |cell|cell|                            |
   |  +----+----+                            |
   +-----------------------------------------+

5. Concatenate all block features for final HOG descriptor
```

### HOG 보행자 검출기 사용

```python
import cv2
import numpy as np

# HOG descriptor + SVM classifier
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load image
img = cv2.imread('street.jpg')
img = cv2.resize(img, None, fx=0.5, fy=0.5)  # Resize for speed

# Pedestrian detection
# detectMultiScale returns: (detected regions, confidence weights)
boxes, weights = hog.detectMultiScale(
    img,
    winStride=(8, 8),    # Window stride
    padding=(4, 4),       # Padding
    scale=1.05,           # Scale factor
    hitThreshold=0,       # SVM threshold
    finalThreshold=2.0    # Final grouping threshold
)

# Draw results
for (x, y, w, h), weight in zip(boxes, weights):
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, f'{weight[0]:.2f}', (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

print(f"Pedestrians detected: {len(boxes)}")
cv2.imshow('Pedestrian Detection', img)
cv2.waitKey(0)
```

### Non-Maximum Suppression (NMS)

```python
import cv2
import numpy as np

def non_max_suppression(boxes, scores, threshold=0.5):
    """Non-Maximum Suppression implementation"""
    if len(boxes) == 0:
        return []

    # Convert coordinates to float
    boxes = boxes.astype(np.float32)

    # Separate coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # Compute areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort by scores (descending)
    order = scores.flatten().argsort()[::-1]

    keep = []
    while order.size > 0:
        # Select box with highest score
        i = order[0]
        keep.append(i)

        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # Keep only boxes with IoU below threshold
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep

# HOG detection with NMS
def detect_pedestrians_with_nms(img, nms_threshold=0.3):
    """Pedestrian detection with NMS"""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detection
    boxes, weights = hog.detectMultiScale(img, winStride=(8, 8),
                                          padding=(4, 4), scale=1.05)

    if len(boxes) == 0:
        return img, []

    # Apply NMS
    boxes = np.array(boxes)
    weights = np.array(weights)
    keep = non_max_suppression(boxes, weights, nms_threshold)

    # Draw results
    result = img.copy()
    final_boxes = []

    for i in keep:
        x, y, w, h = boxes[i]
        final_boxes.append((x, y, w, h))
        cv2.rectangle(result, (int(x), int(y)), (int(x+w), int(y+h)),
                     (0, 255, 0), 2)

    return result, final_boxes

# Usage example
img = cv2.imread('crowd.jpg')
result, detections = detect_pedestrians_with_nms(img)
print(f"Detections after NMS: {len(detections)}")
```

### HOG 특징 시각화

```python
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure

def visualize_hog(img):
    """HOG feature visualization"""
    # Grayscale conversion
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Resize (64x128 - HOG pedestrian detection standard size)
    resized = cv2.resize(gray, (64, 128))

    # Use scikit-image's hog (includes visualization)
    features, hog_image = hog(
        resized,
        orientations=9,          # 9 bins cover 0-180° in 20° steps; coarse enough
                                 # to be robust to small pose changes, fine enough
                                 # to distinguish different edge orientations
        pixels_per_cell=(8, 8),  # 8×8 cells capture local texture at pedestrian scale;
                                 # smaller cells are noisier, larger cells lose spatial detail
        cells_per_block=(2, 2),  # 2×2 block normalization reduces sensitivity to local
                                 # illumination variation — the key advantage of HOG over raw gradients
        visualize=True,
        block_norm='L2-Hys'      # Clipped L2 norm prevents a single large gradient from
                                 # dominating the descriptor (more stable than plain L2)
    )

    # Rescale for visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image,
                                                     out_range=(0, 255))
    hog_image_rescaled = hog_image_rescaled.astype(np.uint8)

    print(f"HOG feature vector size: {features.shape[0]}")

    return hog_image_rescaled, features

# Usage example (requires scikit-image: pip install scikit-image)
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
    """HOG + SVM classifier training (conceptual example)"""

    # HOG descriptor setup
    win_size = (64, 128)    # 1:2 aspect ratio matches the typical standing-person bounding box;
                            # Dalal & Triggs used this exact size in the original 2005 paper
    block_size = (16, 16)   # Two cells per side: large enough for meaningful normalization context
    block_stride = (8, 8)   # 50% overlap between adjacent blocks provides redundant coverage
                            # that improves robustness at the cost of a larger descriptor
    cell_size = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                            cell_size, nbins)

    # Feature extraction
    features = []
    labels = []

    # Positive samples (images with object)
    for img in positive_samples:
        img_resized = cv2.resize(img, win_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        h = hog.compute(gray)
        features.append(h.flatten())
        labels.append(1)

    # Negative samples (images without object)
    for img in negative_samples:
        img_resized = cv2.resize(img, win_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        h = hog.compute(gray)
        features.append(h.flatten())
        labels.append(0)

    X = np.array(features)
    y = np.array(labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # SVM training
    # C=0.01 (small regularization) encourages a wider margin, which generalizes
    # better on HOG features that already capture appearance structure; high C
    # risks overfitting to training-set lighting and pose variations
    clf = svm.LinearSVC(C=0.01)
    clf.fit(X_train, y_train)

    # Print accuracy
    accuracy = clf.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    return hog, clf

# Method to set trained SVM to HOGDescriptor
def set_svm_detector(hog, clf):
    """Set trained SVM to HOG detector"""
    # Extract LinearSVC coefficients and intercept
    sv = clf.coef_.flatten()
    rho = -clf.intercept_[0]

    # Convert to format expected by HOG descriptor
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
    ('template1.jpg', (255, 0, 0)),   # Blue
    ('template2.jpg', (0, 255, 0)),   # Green
    ('template3.jpg', (0, 0, 255))    # Red
]

for template_path, color in templates:
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    # ... matching and drawing
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
    # Perform template matching
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
# Optimization tips:
# 1. Reduce frame to half size
# 2. Increase scaleFactor to 1.2~1.3
# 3. Lower minNeighbors to 3
# 4. Set appropriate minSize

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    # Detect then scale coordinates by 2
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
    # Recompute and visualize HOG
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
# haarcascade_russian_plate_number.xml or
# Use custom trained cascade

# Or detection using license plate characteristics:
# 1. Edge detection
# 2. Rectangular contour detection
# 3. Aspect ratio filtering (plates are typically 4:1 ~ 5:1)
```

</details>

---

## 다음 단계

- [얼굴 검출 및 인식 (Face Detection and Recognition)](./16_Face_Detection.md) - dlib, face_recognition, 실시간 얼굴 인식

---

## 참고 자료

- [OpenCV Template Matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
- [OpenCV Cascade Classifier](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [HOG Tutorial - Learn OpenCV](https://learnopencv.com/histogram-of-oriented-gradients/)
- Dalal, N., & Triggs, B. (2005). "Histograms of Oriented Gradients for Human Detection"
- Viola, P., & Jones, M. (2001). "Rapid Object Detection using a Boosted Cascade of Simple Features"
