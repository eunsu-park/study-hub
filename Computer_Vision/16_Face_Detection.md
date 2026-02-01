# 얼굴 검출 및 인식 (Face Detection and Recognition)

## 개요

얼굴 검출과 인식은 컴퓨터 비전의 가장 실용적인 응용 분야입니다. Haar Cascade, dlib, face_recognition 라이브러리를 활용한 다양한 얼굴 처리 기술을 학습합니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: 객체 검출 기초, 특징점 검출, 이미지 변환

---

## 목차

1. [Haar Cascade 얼굴/눈 검출](#1-haar-cascade-얼굴눈-검출)
2. [dlib 얼굴 검출기 (HOG-based)](#2-dlib-얼굴-검출기-hog-based)
3. [dlib 얼굴 랜드마크 (68 포인트)](#3-dlib-얼굴-랜드마크-68-포인트)
4. [LBPH 얼굴 인식](#4-lbph-얼굴-인식)
5. [face_recognition 라이브러리](#5-face_recognition-라이브러리)
6. [실시간 얼굴 검출](#6-실시간-얼굴-검출)
7. [연습 문제](#7-연습-문제)

---

## 1. Haar Cascade 얼굴/눈 검출

### 얼굴 검출 원리

```
Haar Cascade 얼굴 검출 과정:

1. 적분 이미지 계산
   ┌─────────────────┐
   │ 원본 이미지     │ → 적분 이미지 (빠른 특징 계산)
   └─────────────────┘

2. 다양한 크기의 윈도우로 스캔
   ┌─────────────────────────────┐
   │  ┌──┐                       │
   │  │  │  → 작은 윈도우       │
   │  └──┘                       │
   │     ┌─────┐                 │
   │     │     │ → 중간 윈도우   │
   │     └─────┘                 │
   │        ┌────────┐           │
   │        │        │ → 큰 윈도우│
   │        └────────┘           │
   └─────────────────────────────┘

3. 각 윈도우에서 Cascade 분류기 적용

   윈도우 → Stage 1 → Stage 2 → ... → Stage N
           (얼굴?)   (얼굴?)         (얼굴!)

4. 검출 결과 그룹화 (중복 제거)
```

### 기본 얼굴 검출

```python
import cv2
import numpy as np

def detect_faces_haar(img, scale_factor=1.1, min_neighbors=5,
                      min_size=(30, 30)):
    """Haar Cascade를 이용한 얼굴 검출"""

    # Cascade 분류기 로드
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # 그레이스케일 변환
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 히스토그램 평활화 (조명 보정)
    gray = cv2.equalizeHist(gray)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces

# 사용 예
img = cv2.imread('photo.jpg')
faces = detect_faces_haar(img)

# 결과 그리기
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # 얼굴 중심점
    center = (x + w//2, y + h//2)
    cv2.circle(img, center, 3, (0, 0, 255), -1)

print(f"검출된 얼굴 수: {len(faces)}")
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
```

### 얼굴 및 눈 동시 검출

```python
import cv2

class HaarFaceEyeDetector:
    """Haar Cascade 기반 얼굴/눈 검출기"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.eye_glasses_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )

    def detect(self, img, detect_eyes=True):
        """얼굴과 눈 검출"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        results = []

        # 얼굴 검출
        faces = self.face_cascade.detectMultiScale(
            gray, 1.1, 5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_data = {
                'face_rect': (x, y, w, h),
                'eyes': []
            }

            if detect_eyes:
                # 얼굴 상단 50%에서 눈 검출
                roi_gray = gray[y:y+h//2, x:x+w]

                # 일반 눈 검출 시도
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray, 1.1, 3, minSize=(20, 20)
                )

                # 안경 착용자용 검출 시도 (일반 검출 실패 시)
                if len(eyes) < 2:
                    eyes = self.eye_glasses_cascade.detectMultiScale(
                        roi_gray, 1.1, 3, minSize=(20, 20)
                    )

                # 가장 그럴듯한 두 눈 선택
                eyes = self._select_best_eyes(eyes, w)

                for (ex, ey, ew, eh) in eyes:
                    face_data['eyes'].append((x + ex, y + ey, ew, eh))

            results.append(face_data)

        return results

    def _select_best_eyes(self, eyes, face_width):
        """가장 좋은 두 눈 선택"""
        if len(eyes) <= 2:
            return eyes

        # 눈 크기와 y 좌표로 필터링
        eyes = sorted(eyes, key=lambda e: e[1])  # y 좌표로 정렬

        # 상위 4개 후보에서 선택
        candidates = eyes[:4]

        # x 좌표 기준 왼쪽/오른쪽 분리
        mid_x = face_width // 2
        left_eyes = [e for e in candidates if e[0] + e[2]//2 < mid_x]
        right_eyes = [e for e in candidates if e[0] + e[2]//2 >= mid_x]

        result = []
        if left_eyes:
            result.append(left_eyes[0])
        if right_eyes:
            result.append(right_eyes[0])

        return result

    def draw_results(self, img, results):
        """결과 시각화"""
        output = img.copy()

        for face_data in results:
            x, y, w, h = face_data['face_rect']

            # 얼굴 사각형
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 눈
            for (ex, ey, ew, eh) in face_data['eyes']:
                center = (ex + ew//2, ey + eh//2)
                radius = (ew + eh) // 4
                cv2.circle(output, center, radius, (255, 0, 0), 2)

        return output

# 사용 예
detector = HaarFaceEyeDetector()
img = cv2.imread('portrait.jpg')
results = detector.detect(img)
output = detector.draw_results(img, results)
cv2.imshow('Detection', output)
```

---

## 2. dlib 얼굴 검출기 (HOG-based)

### dlib 설치

```bash
# dlib 설치 (C++ 컴파일러 필요)
pip install dlib

# 또는 conda 사용 (더 쉬움)
conda install -c conda-forge dlib
```

### HOG 기반 검출기

```python
import cv2
import dlib
import numpy as np

# HOG 기반 얼굴 검출기
detector = dlib.get_frontal_face_detector()

# 이미지 로드
img = cv2.imread('photo.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 얼굴 검출
# 두 번째 인자: 업샘플링 횟수 (0=원본, 1=2배, 2=4배)
faces = detector(rgb, 1)

print(f"검출된 얼굴 수: {len(faces)}")

# 결과 시각화
for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 신뢰도 점수 (dlib 검출기는 기본적으로 점수 제공)
    # detector.run()을 사용하면 점수도 얻을 수 있음

cv2.imshow('dlib HOG Detection', img)
cv2.waitKey(0)
```

### CNN 기반 검출기 (더 정확)

```python
import cv2
import dlib

# CNN 얼굴 검출기 (모델 파일 필요)
# 다운로드: http://dlib.net/files/mmod_human_face_detector.dat.bz2
cnn_detector = dlib.cnn_face_detection_model_v1(
    'mmod_human_face_detector.dat'
)

img = cv2.imread('photo.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# CNN 검출
detections = cnn_detector(rgb, 1)

for d in detections:
    x1 = d.rect.left()
    y1 = d.rect.top()
    x2 = d.rect.right()
    y2 = d.rect.bottom()
    confidence = d.confidence

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'{confidence:.2f}', (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
```

### Haar vs dlib 비교

```
┌────────────────┬──────────────────┬──────────────────┐
│     항목       │   Haar Cascade   │   dlib HOG       │
├────────────────┼──────────────────┼──────────────────┤
│ 속도           │ 빠름             │ 중간             │
│ 정확도         │ 중간             │ 높음             │
│ 오검출         │ 많음             │ 적음             │
│ 측면 얼굴      │ 별도 모델 필요   │ 지원 안 함       │
│ 작은 얼굴      │ 잘 검출          │ 검출 어려움      │
│ 설치 용이성    │ OpenCV 기본 포함 │ 별도 설치 필요   │
│ 메모리 사용    │ 적음             │ 중간             │
└────────────────┴──────────────────┴──────────────────┘

dlib CNN 검출기:
- 가장 정확하지만 GPU 없이는 느림
- 측면 얼굴도 잘 검출
- 작은 얼굴 검출 우수
```

---

## 3. dlib 얼굴 랜드마크 (68 포인트)

### 68 포인트 랜드마크 구조

```
얼굴 랜드마크 68 포인트:

        17-21    22-26
         ____     ____
    0   /    \   /    \   16
    |  │ 36-41│ │42-47 │  |
    |   \____/   \____/   |
    |      48-67          |
    |      /    \         |
    |     /      \        |
   8     \________/

포인트 그룹:
- 0-16:   턱 라인 (jaw)
- 17-21:  왼쪽 눈썹 (left eyebrow)
- 22-26:  오른쪽 눈썹 (right eyebrow)
- 27-35:  코 (nose)
- 36-41:  왼쪽 눈 (left eye)
- 42-47:  오른쪽 눈 (right eye)
- 48-67:  입 (mouth)
  - 48-59: 외부 입술 (outer lip)
  - 60-67: 내부 입술 (inner lip)
```

### 랜드마크 검출

```python
import cv2
import dlib
import numpy as np

# 검출기와 예측기 로드
detector = dlib.get_frontal_face_detector()
# 다운로드: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_landmarks(img):
    """얼굴 랜드마크 검출"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 얼굴 검출
    faces = detector(rgb, 1)

    all_landmarks = []

    for face in faces:
        # 랜드마크 예측
        shape = predictor(rgb, face)

        # dlib shape를 numpy 배열로 변환
        landmarks = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)

        all_landmarks.append({
            'face_rect': (face.left(), face.top(),
                         face.right(), face.bottom()),
            'landmarks': landmarks
        })

    return all_landmarks

def draw_landmarks(img, landmarks_data, draw_indices=False):
    """랜드마크 시각화"""
    output = img.copy()

    for data in landmarks_data:
        landmarks = data['landmarks']

        # 모든 포인트 그리기
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(output, (x, y), 2, (0, 255, 0), -1)
            if draw_indices:
                cv2.putText(output, str(i), (x-5, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        # 연결선 그리기
        # 턱 라인
        for i in range(16):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (255, 255, 0), 1)

        # 눈썹
        for i in range(17, 21):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (255, 255, 0), 1)
        for i in range(22, 26):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (255, 255, 0), 1)

        # 코
        for i in range(27, 30):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (255, 255, 0), 1)
        for i in range(31, 35):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (255, 255, 0), 1)

        # 눈
        eye_indices = [(36,41), (42,47)]
        for start, end in eye_indices:
            for i in range(start, end):
                cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                        (0, 255, 255), 1)
            cv2.line(output, tuple(landmarks[end]), tuple(landmarks[start]),
                    (0, 255, 255), 1)

        # 입
        for i in range(48, 59):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (0, 0, 255), 1)
        cv2.line(output, tuple(landmarks[59]), tuple(landmarks[48]),
                (0, 0, 255), 1)

        for i in range(60, 67):
            cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]),
                    (0, 0, 255), 1)
        cv2.line(output, tuple(landmarks[67]), tuple(landmarks[60]),
                (0, 0, 255), 1)

    return output

# 사용 예
img = cv2.imread('face.jpg')
landmarks_data = get_landmarks(img)
result = draw_landmarks(img, landmarks_data)
cv2.imshow('Landmarks', result)
```

### 랜드마크 활용

```python
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

class FaceLandmarkAnalyzer:
    """얼굴 랜드마크 분석기"""

    # 랜드마크 인덱스 정의
    JAWLINE = list(range(0, 17))
    LEFT_EYEBROW = list(range(17, 22))
    RIGHT_EYEBROW = list(range(22, 27))
    NOSE_BRIDGE = list(range(27, 31))
    NOSE_TIP = list(range(31, 36))
    LEFT_EYE = list(range(36, 42))
    RIGHT_EYE = list(range(42, 48))
    OUTER_LIP = list(range(48, 60))
    INNER_LIP = list(range(60, 68))

    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def get_landmarks(self, img):
        """랜드마크 추출"""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb, 0)

        if len(faces) == 0:
            return None

        shape = self.predictor(rgb, faces[0])
        landmarks = np.array([[shape.part(i).x, shape.part(i).y]
                             for i in range(68)])
        return landmarks

    def eye_aspect_ratio(self, eye_points):
        """눈 종횡비 (EAR) 계산 - 졸음 감지에 사용"""
        # 수직 거리
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        # 수평 거리
        C = dist.euclidean(eye_points[0], eye_points[3])

        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth_points):
        """입 종횡비 (MAR) 계산 - 하품 감지에 사용"""
        # 수직 거리
        A = dist.euclidean(mouth_points[2], mouth_points[10])  # 51, 59
        B = dist.euclidean(mouth_points[4], mouth_points[8])   # 53, 57
        # 수평 거리
        C = dist.euclidean(mouth_points[0], mouth_points[6])   # 49, 55

        mar = (A + B) / (2.0 * C)
        return mar

    def get_face_angle(self, landmarks):
        """얼굴 기울기 각도 계산"""
        # 양쪽 눈 중심점 사용
        left_eye_center = landmarks[self.LEFT_EYE].mean(axis=0)
        right_eye_center = landmarks[self.RIGHT_EYE].mean(axis=0)

        # 각도 계산
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        return angle

    def analyze_face(self, img):
        """얼굴 종합 분석"""
        landmarks = self.get_landmarks(img)
        if landmarks is None:
            return None

        # 눈 분석
        left_eye = landmarks[self.LEFT_EYE]
        right_eye = landmarks[self.RIGHT_EYE]
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # 입 분석
        outer_lip = landmarks[self.OUTER_LIP]
        mar = self.mouth_aspect_ratio(outer_lip)

        # 얼굴 각도
        angle = self.get_face_angle(landmarks)

        return {
            'landmarks': landmarks,
            'eye_aspect_ratio': avg_ear,
            'mouth_aspect_ratio': mar,
            'face_angle': angle,
            'eyes_closed': avg_ear < 0.2,  # 임계값 기반
            'mouth_open': mar > 0.5
        }

# 사용 예
analyzer = FaceLandmarkAnalyzer('shape_predictor_68_face_landmarks.dat')
img = cv2.imread('face.jpg')
analysis = analyzer.analyze_face(img)

if analysis:
    print(f"눈 종횡비 (EAR): {analysis['eye_aspect_ratio']:.3f}")
    print(f"입 종횡비 (MAR): {analysis['mouth_aspect_ratio']:.3f}")
    print(f"얼굴 기울기: {analysis['face_angle']:.1f}도")
    print(f"눈 감음: {analysis['eyes_closed']}")
    print(f"입 벌림: {analysis['mouth_open']}")
```

---

## 4. LBPH 얼굴 인식

### LBP (Local Binary Patterns) 이해

```
LBP: 각 픽셀 주변의 패턴을 이진 코드로 표현

   주변 픽셀       비교 (> 중심?)     이진 코드
   ┌───┬───┬───┐    ┌───┬───┬───┐
   │ 6 │ 5 │ 2 │    │ 1 │ 1 │ 0 │    11000011
   ├───┼───┼───┤    ├───┼───┼───┤    = 195
   │ 7 │[4]│ 1 │    │ 1 │   │ 0 │
   ├───┼───┼───┤    ├───┼───┼───┤
   │ 8 │ 3 │ 2 │    │ 1 │ 0 │ 0 │
   └───┴───┴───┘    └───┴───┴───┘

   중심 픽셀(4)과 주변 비교:
   6>4=1, 5>4=1, 2<4=0, 1<4=0, 2<4=0, 3<4=0, 8>4=1, 7>4=1

LBPH (LBP Histogram):
- 이미지를 여러 셀로 분할
- 각 셀에서 LBP 히스토그램 계산
- 모든 히스토그램을 연결하여 특징 벡터 생성
```

### LBPH 얼굴 인식기

```python
import cv2
import numpy as np
import os

class LBPHFaceRecognizer:
    """LBPH 기반 얼굴 인식기"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,        # LBP 반경
            neighbors=8,     # 이웃 수
            grid_x=8,        # x 방향 셀 수
            grid_y=8         # y 방향 셀 수
        )
        self.label_names = {}

    def prepare_training_data(self, data_dir):
        """학습 데이터 준비"""
        faces = []
        labels = []

        for label_id, person_name in enumerate(os.listdir(data_dir)):
            person_dir = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            self.label_names[label_id] = person_name

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                # 얼굴 검출
                detected_faces = self.face_cascade.detectMultiScale(
                    img, 1.1, 5, minSize=(50, 50)
                )

                for (x, y, w, h) in detected_faces:
                    face_roi = img[y:y+h, x:x+w]
                    # 크기 정규화
                    face_roi = cv2.resize(face_roi, (100, 100))
                    faces.append(face_roi)
                    labels.append(label_id)

        return faces, labels

    def train(self, faces, labels):
        """모델 학습"""
        self.recognizer.train(faces, np.array(labels))
        print(f"학습 완료: {len(set(labels))}명, {len(faces)}개 이미지")

    def save_model(self, path):
        """모델 저장"""
        self.recognizer.save(path)
        # 라벨 이름도 저장
        np.save(path + '_labels.npy', self.label_names)

    def load_model(self, path):
        """모델 로드"""
        self.recognizer.read(path)
        self.label_names = np.load(path + '_labels.npy',
                                   allow_pickle=True).item()

    def predict(self, img):
        """얼굴 인식"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))

            label, confidence = self.recognizer.predict(face_roi)

            # confidence가 낮을수록 좋음 (LBPH)
            # 일반적으로 50 이하면 매우 좋은 매칭
            name = self.label_names.get(label, "Unknown")
            if confidence > 100:
                name = "Unknown"

            results.append({
                'rect': (x, y, w, h),
                'name': name,
                'confidence': confidence
            })

        return results

    def draw_results(self, img, results):
        """결과 시각화"""
        output = img.copy()

        for result in results:
            x, y, w, h = result['rect']
            name = result['name']
            conf = result['confidence']

            # 색상: 인식 성공(녹색), 실패(빨간색)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)

            label = f"{name} ({conf:.1f})"
            cv2.putText(output, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return output

# 사용 예
"""
데이터 디렉토리 구조:
faces/
    person1/
        img1.jpg
        img2.jpg
    person2/
        img1.jpg
        img2.jpg
"""

# 학습
recognizer = LBPHFaceRecognizer()
faces, labels = recognizer.prepare_training_data('faces')
recognizer.train(faces, labels)
recognizer.save_model('face_model.yml')

# 인식
recognizer.load_model('face_model.yml')
test_img = cv2.imread('test.jpg')
results = recognizer.predict(test_img)
output = recognizer.draw_results(test_img, results)
cv2.imshow('Recognition', output)
```

---

## 5. face_recognition 라이브러리

### 설치

```bash
pip install face_recognition
```

### 기본 사용법

```python
import face_recognition
import cv2
import numpy as np

# 이미지 로드
img = face_recognition.load_image_file('photo.jpg')

# 얼굴 위치 검출
face_locations = face_recognition.face_locations(img)
# 또는 CNN 모델 사용 (더 정확)
# face_locations = face_recognition.face_locations(img, model='cnn')

print(f"검출된 얼굴 수: {len(face_locations)}")

# 얼굴 인코딩 (128차원 특징 벡터)
face_encodings = face_recognition.face_encodings(img, face_locations)

# 얼굴 랜드마크
face_landmarks = face_recognition.face_landmarks(img, face_locations)

# 결과 시각화
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(img_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

cv2.imshow('Detection', img_bgr)
```

### 얼굴 비교 및 인식

```python
import face_recognition
import cv2
import numpy as np
import os

class FaceRecognitionSystem:
    """face_recognition 기반 얼굴 인식 시스템"""

    def __init__(self):
        self.known_encodings = []
        self.known_names = []

    def add_face(self, img_path, name):
        """알려진 얼굴 추가"""
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:
            self.known_encodings.append(encodings[0])
            self.known_names.append(name)
            print(f"'{name}' 얼굴 등록 완료")
            return True
        else:
            print(f"'{img_path}'에서 얼굴을 찾을 수 없습니다")
            return False

    def load_faces_from_directory(self, data_dir):
        """디렉토리에서 얼굴 로드"""
        for person_name in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                self.add_face(img_path, person_name)

        print(f"총 {len(self.known_encodings)}개 얼굴 로드 완료")

    def recognize(self, img, tolerance=0.6):
        """얼굴 인식"""
        # RGB 변환
        if isinstance(img, str):
            img = face_recognition.load_image_file(img)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 얼굴 검출 및 인코딩
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        results = []

        for (top, right, bottom, left), encoding in zip(face_locations,
                                                         face_encodings):
            # 알려진 얼굴들과 비교
            matches = face_recognition.compare_faces(
                self.known_encodings, encoding, tolerance=tolerance
            )

            # 거리 계산 (낮을수록 유사)
            distances = face_recognition.face_distance(
                self.known_encodings, encoding
            )

            name = "Unknown"
            confidence = 0.0

            if True in matches:
                # 가장 가까운 매칭 찾기
                best_match_idx = np.argmin(distances)
                if matches[best_match_idx]:
                    name = self.known_names[best_match_idx]
                    confidence = 1 - distances[best_match_idx]

            results.append({
                'location': (top, right, bottom, left),
                'name': name,
                'confidence': confidence
            })

        return results

    def save_encodings(self, path):
        """인코딩 저장"""
        data = {
            'encodings': self.known_encodings,
            'names': self.known_names
        }
        np.save(path, data)

    def load_encodings(self, path):
        """인코딩 로드"""
        data = np.load(path, allow_pickle=True).item()
        self.known_encodings = data['encodings']
        self.known_names = data['names']

# 사용 예
system = FaceRecognitionSystem()

# 알려진 얼굴 등록
system.add_face('known_faces/person1.jpg', 'Alice')
system.add_face('known_faces/person2.jpg', 'Bob')
# 또는 디렉토리에서 로드
# system.load_faces_from_directory('known_faces')

# 인식
test_img = cv2.imread('test.jpg')
results = system.recognize(test_img)

# 시각화
for result in results:
    top, right, bottom, left = result['location']
    name = result['name']
    conf = result['confidence']

    cv2.rectangle(test_img, (left, top), (right, bottom), (0, 255, 0), 2)
    label = f"{name} ({conf:.2%})"
    cv2.putText(test_img, label, (left, top - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow('Recognition', test_img)
```

### 얼굴 클러스터링

```python
import face_recognition
from sklearn.cluster import DBSCAN
import numpy as np
import os

def cluster_faces(image_dir, output_dir='clustered'):
    """유사한 얼굴끼리 그룹화"""

    encodings = []
    image_paths = []
    face_locations_list = []

    # 모든 이미지에서 얼굴 인코딩 추출
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = face_recognition.load_image_file(img_path)

        locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, locations)

        for encoding, location in zip(face_encodings, locations):
            encodings.append(encoding)
            image_paths.append(img_path)
            face_locations_list.append(location)

    # DBSCAN 클러스터링
    # eps: 동일 클러스터로 간주할 거리 임계값
    # min_samples: 클러스터 형성에 필요한 최소 샘플 수
    clt = DBSCAN(metric='euclidean', eps=0.5, min_samples=2)
    clt.fit(encodings)

    # 클러스터별 결과 정리
    label_ids = np.unique(clt.labels_)
    num_unique = len(label_ids[label_ids > -1])  # -1은 노이즈

    print(f"발견된 고유 인물 수: {num_unique}")

    # 클러스터별로 이미지 저장
    os.makedirs(output_dir, exist_ok=True)

    for label_id in label_ids:
        indices = np.where(clt.labels_ == label_id)[0]

        if label_id == -1:
            folder = os.path.join(output_dir, 'unknown')
        else:
            folder = os.path.join(output_dir, f'person_{label_id}')

        os.makedirs(folder, exist_ok=True)
        print(f"클러스터 {label_id}: {len(indices)}개 얼굴")

    return clt.labels_, image_paths, face_locations_list
```

---

## 6. 실시간 얼굴 검출

### 기본 실시간 검출

```python
import cv2
import time

def realtime_face_detection():
    """실시간 얼굴 검출 (Haar Cascade)"""

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_counter = 0
    fps = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 축소 (속도 향상)
        small_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,  # 빠른 검출을 위해 증가
            minNeighbors=4,
            minSize=(30, 30)
        )

        # 좌표 스케일 복원
        for (x, y, w, h) in faces:
            x, y, w, h = x*2, y*2, w*2, h*2
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # FPS 계산
        fps_counter += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            start_time = time.time()

        # FPS 표시
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

realtime_face_detection()
```

### 실시간 얼굴 인식

```python
import cv2
import face_recognition
import numpy as np
import time

class RealtimeFaceRecognition:
    """실시간 얼굴 인식 시스템"""

    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.process_every_n_frames = 3  # 매 n번째 프레임만 처리

    def add_known_face(self, img_path, name):
        """알려진 얼굴 추가"""
        img = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(img)

        if encoding:
            self.known_encodings.append(encoding[0])
            self.known_names.append(name)

    def run(self, camera_id=0):
        """실시간 인식 실행"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        face_locations = []
        face_names = []

        fps_time = time.time()
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 크기 축소 (속도 향상)
            small_frame = cv2.resize(frame, None, fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # 매 n번째 프레임만 처리
            if frame_count % self.process_every_n_frames == 0:
                # 얼굴 검출
                face_locations = face_recognition.face_locations(rgb_small)
                face_encodings = face_recognition.face_encodings(
                    rgb_small, face_locations
                )

                face_names = []
                for encoding in face_encodings:
                    name = "Unknown"

                    if self.known_encodings:
                        matches = face_recognition.compare_faces(
                            self.known_encodings, encoding, tolerance=0.6
                        )
                        distances = face_recognition.face_distance(
                            self.known_encodings, encoding
                        )

                        if len(distances) > 0:
                            best_idx = np.argmin(distances)
                            if matches[best_idx]:
                                name = self.known_names[best_idx]

                    face_names.append(name)

            # 결과 표시 (좌표 스케일 복원)
            for (top, right, bottom, left), name in zip(face_locations,
                                                         face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # 이름 배경
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom),
                             color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # FPS 계산 및 표시
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()

            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Face Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 's' 키로 현재 프레임 저장
                cv2.imwrite(f'capture_{frame_count}.jpg', frame)

        cap.release()
        cv2.destroyAllWindows()

# 사용 예
system = RealtimeFaceRecognition()
system.add_known_face('alice.jpg', 'Alice')
system.add_known_face('bob.jpg', 'Bob')
system.run()
```

### 성능 최적화 팁

```
┌─────────────────────────────────────────────────────────────┐
│                   실시간 처리 최적화 전략                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. 프레임 크기 축소                                         │
│    - 1/4 크기로 축소하면 처리량 16배 감소                   │
│    - small = cv2.resize(frame, None, fx=0.25, fy=0.25)     │
│                                                             │
│ 2. 프레임 스킵                                              │
│    - 매 프레임 처리 불필요                                  │
│    - 2-5 프레임마다 검출 수행                               │
│    - 중간 프레임은 이전 결과 사용                           │
│                                                             │
│ 3. ROI 기반 처리                                            │
│    - 이전 검출 위치 주변만 검색                             │
│    - 트래킹과 검출 조합                                     │
│                                                             │
│ 4. 모델 선택                                                │
│    - Haar: 가장 빠름, 정확도 낮음                          │
│    - dlib HOG: 중간                                         │
│    - dlib CNN: 느림, GPU 권장                               │
│    - face_recognition: dlib 기반                            │
│                                                             │
│ 5. 멀티스레딩                                               │
│    - 검출과 표시를 별도 스레드로                            │
│    - Queue로 프레임 전달                                    │
│                                                             │
│ 6. GPU 가속                                                 │
│    - dlib CUDA 빌드                                         │
│    - OpenCV DNN (CUDA backend)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 연습 문제

### 문제 1: 출석 체크 시스템

얼굴 인식 기반 출석 체크 시스템을 구현하세요.

**요구사항**:
- 등록된 사용자 얼굴 DB 관리
- 실시간 웹캠 인식
- 출석 시간 기록 (CSV 또는 DB)
- 중복 출석 방지 (일정 시간 내)

<details>
<summary>힌트</summary>

```python
import datetime
import csv

class AttendanceSystem:
    def __init__(self):
        self.attendance_log = {}  # {name: last_check_time}
        self.cooldown = 3600  # 1시간

    def mark_attendance(self, name):
        now = datetime.datetime.now()
        last_check = self.attendance_log.get(name)

        if last_check is None or (now - last_check).seconds > self.cooldown:
            self.attendance_log[name] = now
            self.save_to_csv(name, now)
            return True
        return False
```

</details>

### 문제 2: 졸음 감지 시스템

눈 종횡비(EAR)를 이용한 졸음 감지 시스템을 구현하세요.

**요구사항**:
- 실시간 눈 상태 모니터링
- EAR이 임계값 이하로 일정 시간 유지되면 경고
- 경고음 또는 시각적 알림
- 현재 EAR 값 표시

<details>
<summary>힌트</summary>

```python
import dlib
from scipy.spatial import distance as dist
import pygame  # 경고음용

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20  # 연속 프레임 수

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

counter = 0  # 연속 프레임 카운터
# EAR < THRESHOLD가 CONSEC_FRAMES 이상 지속되면 경고
```

</details>

### 문제 3: 얼굴 모자이크 처리

특정 인물의 얼굴만 모자이크 처리하는 프로그램을 작성하세요.

**요구사항**:
- 지정된 인물 외 모든 얼굴 모자이크
- 또는 지정된 인물만 모자이크
- 모자이크 강도 조절 가능

<details>
<summary>힌트</summary>

```python
def mosaic_face(img, rect, scale=0.1):
    x, y, w, h = rect
    roi = img[y:y+h, x:x+w]

    # 축소 후 확대 (모자이크 효과)
    small = cv2.resize(roi, None, fx=scale, fy=scale)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    img[y:y+h, x:x+w] = mosaic
    return img
```

</details>

### 문제 4: 얼굴 정렬 (Face Alignment)

눈 위치를 기준으로 얼굴을 정렬하는 프로그램을 구현하세요.

**요구사항**:
- 양쪽 눈 위치 검출
- 눈을 수평으로 맞추도록 회전
- 얼굴 영역 크롭 및 크기 정규화

<details>
<summary>힌트</summary>

```python
import numpy as np

def align_face(img, left_eye, right_eye, desired_size=(256, 256)):
    # 눈 사이 각도 계산
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # 회전 중심 (양 눈 중간점)
    center = ((left_eye[0] + right_eye[0]) // 2,
              (left_eye[1] + right_eye[1]) // 2)

    # 회전 행렬
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 회전 적용
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return aligned
```

</details>

### 문제 5: 감정 분석 시스템

얼굴 랜드마크를 이용한 간단한 감정 분석 시스템을 구현하세요.

**요구사항**:
- 눈, 눈썹, 입 모양 분석
- 기본 감정 분류 (행복, 슬픔, 놀람, 무표정)
- 실시간 감정 표시

<details>
<summary>힌트</summary>

```python
# 감정 판단 기준 예시:
# - 행복: 입꼬리가 올라감 (입 양끝 y좌표 < 입 중앙 y좌표)
# - 놀람: 눈과 입이 크게 열림 (EAR 높음, MAR 높음)
# - 슬픔: 눈썹이 처짐, 입꼬리가 내려감
# - 무표정: 변화가 적음

def analyze_emotion(landmarks):
    # 입 분석
    mouth = landmarks[48:68]
    mouth_height = mouth[14][1] - mouth[10][1]  # 입 높이

    # 눈 분석
    left_eye = landmarks[36:42]
    ear = eye_aspect_ratio(left_eye)

    # 규칙 기반 판단
    if mouth_height > threshold and ear > threshold:
        return "Surprised"
    # ...
```

</details>

---

## 다음 단계

- [17_Video_Processing.md](./17_Video_Processing.md) - VideoCapture, 배경 차분, 옵티컬 플로우

---

## 참고 자료

- [dlib Documentation](http://dlib.net/python/index.html)
- [face_recognition GitHub](https://github.com/ageitgey/face_recognition)
- [OpenCV Face Recognition](https://docs.opencv.org/4.x/da/d60/tutorial_face_main.html)
- [68 Face Landmarks](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
- Kazemi, V., & Sullivan, J. (2014). "One Millisecond Face Alignment with an Ensemble of Regression Trees"
