# 카메라 캘리브레이션 (Camera Calibration)

## 개요

카메라 캘리브레이션은 카메라의 내부 파라미터와 렌즈 왜곡을 측정하는 과정입니다. 정확한 3D 복원, 증강현실, 로봇 비전 등에서 필수적인 단계입니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: 기하학적 변환, 선형대수 기초, 이미지 좌표계

---

## 목차

1. [카메라 내부 파라미터](#1-카메라-내부-파라미터)
2. [렌즈 왜곡](#2-렌즈-왜곡)
3. [findChessboardCorners()](#3-findchessboardcorners)
4. [calibrateCamera()](#4-calibratecamera)
5. [undistort(): 왜곡 보정](#5-undistort-왜곡-보정)
6. [재투영 오차](#6-재투영-오차)
7. [연습 문제](#7-연습-문제)

---

## 1. 카메라 내부 파라미터

### 핀홀 카메라 모델

```
핀홀 카메라 모델:
3D 세계 좌표를 2D 이미지 좌표로 투영

        3D 세계
           P(X, Y, Z)
              │
              │
              ▼
       ┌──────────────┐
       │    렌즈      │  ← 카메라
       └──────────────┘
              │
              │  초점 거리 f
              ▼
       ┌──────────────┐
       │ 이미지 평면  │  → p(u, v)
       │      ●       │
       └──────────────┘

투영 공식:
u = fx * (X/Z) + cx
v = fy * (Y/Z) + cy

- (X, Y, Z): 3D 점의 카메라 좌표
- (u, v): 2D 이미지 좌표 (픽셀)
- fx, fy: 초점 거리 (픽셀 단위)
- (cx, cy): 주점 (principal point)
```

### 카메라 행렬 (Intrinsic Matrix)

```
카메라 내부 파라미터 행렬 K:

     ┌             ┐
     │ fx   0   cx │
K =  │  0  fy   cy │
     │  0   0    1 │
     └             ┘

파라미터 설명:
┌────────────────────────────────────────────────────────────┐
│ fx, fy: 초점 거리 (focal length)                           │
│         - 픽셀 단위                                        │
│         - fx = f / pixel_width                             │
│         - fy = f / pixel_height                            │
│         - 일반적으로 fx ≈ fy (정사각 픽셀)                 │
│                                                            │
│ cx, cy: 주점 (principal point)                             │
│         - 광축이 이미지 평면과 만나는 점                   │
│         - 이상적으로는 이미지 중심 (width/2, height/2)     │
│         - 실제로는 약간의 오프셋 존재                      │
│                                                            │
│ skew: 비대칭 계수 (보통 0)                                 │
│         - 행렬의 (0,1) 위치                                │
│         - 픽셀의 비직각성                                  │
└────────────────────────────────────────────────────────────┘

예시 (일반적인 웹캠):
     ┌                    ┐
     │ 800    0    320    │
K =  │   0  800    240    │   (640x480 해상도)
     │   0    0      1    │
     └                    ┘
```

### 외부 파라미터

```
카메라 외부 파라미터 (Extrinsic Parameters):
카메라 좌표계와 세계 좌표계 사이의 변환

세계 좌표 → 카메라 좌표:

[X_cam]       [X_world]
[Y_cam] = R * [Y_world] + t
[Z_cam]       [Z_world]

R: 3x3 회전 행렬 (rotation)
t: 3x1 이동 벡터 (translation)

전체 투영:

     ┌   ┐       ┌             ┐   ┌       ┐   ┌   ┐
s *  │ u │   =   │ fx   0   cx │ * │ R | t │ * │ X │
     │ v │       │  0  fy   cy │   │   |   │   │ Y │
     │ 1 │       │  0   0    1 │   │   |   │   │ Z │
     └   ┘       └             ┘   └       ┘   │ 1 │
                                               └   ┘
    이미지         카메라 행렬      외부 행렬    세계 좌표
```

---

## 2. 렌즈 왜곡

### 왜곡 종류

```
렌즈 왜곡 (Lens Distortion):

1. 방사 왜곡 (Radial Distortion)
   - 렌즈의 곡면으로 인해 발생
   - 배럴 (Barrel): 볼록하게 왜곡 (광각 렌즈)
   - 핀쿠션 (Pincushion): 오목하게 왜곡 (망원 렌즈)

   원본          배럴 왜곡       핀쿠션 왜곡
   ┌───────┐    ╭───────╮      ┌───────┐
   │       │    │       │      ╰       ╯
   │       │    │       │      │       │
   │       │    │       │      ╭       ╮
   └───────┘    ╰───────╯      └───────┘

2. 접선 왜곡 (Tangential Distortion)
   - 렌즈와 이미지 센서의 정렬 오차로 인해 발생
   - 이미지가 기울어지거나 비틀어짐

   ┌───────┐      ┌───────┐
   │       │      │╲      │
   │       │  →   │ ╲     │
   │       │      │  ╲    │
   └───────┘      └───────┘
```

### 왜곡 모델 수식

```
방사 왜곡 (Radial Distortion):

x_distorted = x * (1 + k1*r² + k2*r⁴ + k3*r⁶)
y_distorted = y * (1 + k1*r² + k2*r⁴ + k3*r⁶)

여기서:
- r² = x² + y² (정규화된 이미지 좌표에서의 거리)
- k1, k2, k3: 방사 왜곡 계수

접선 왜곡 (Tangential Distortion):

x_distorted = x + [2*p1*x*y + p2*(r² + 2*x²)]
y_distorted = y + [p1*(r² + 2*y²) + 2*p2*x*y]

여기서:
- p1, p2: 접선 왜곡 계수

왜곡 계수 벡터:
distCoeffs = [k1, k2, p1, p2, k3]

(일부 모델에서는 k4, k5, k6 추가)
```

### 왜곡의 영향

```
왜곡이 심한 경우의 영향:

1. 직선이 곡선으로 보임
   실제: ───────────
   왜곡: ╭─────────╮

2. 거리 측정 오류
   - 이미지 가장자리로 갈수록 오류 증가
   - 정밀 측정 불가능

3. 3D 복원 오류
   - 스테레오 비전에서 깊이 오류
   - AR 마커 위치 오류

4. 객체 인식 성능 저하
   - 템플릿 매칭 실패
   - 특징점 매칭 정확도 저하
```

---

## 3. findChessboardCorners()

### 체스보드 패턴

```
체스보드 패턴을 사용하는 이유:

1. 코너 검출이 정확함
2. 제작이 쉬움 (프린트 가능)
3. 평면 패턴으로 캘리브레이션 용이

체스보드 크기 정의:
┌───┬───┬───┬───┬───┬───┬───┬───┐
│   │███│   │███│   │███│   │███│
├───┼───┼───┼───┼───┼───┼───┼───┤
│███│   │███│   │███│   │███│   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │███│   │███│   │███│   │███│
├───┼───┼───┼───┼───┼───┼───┼───┤
│███│   │███│   │███│   │███│   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │███│   │███│   │███│   │███│
├───┼───┼───┼───┼───┼───┼───┼───┤
│███│   │███│   │███│   │███│   │
└───┴───┴───┴───┴───┴───┴───┴───┘

내부 코너 수: (7, 5)
- 가로 7개, 세로 5개의 내부 코너
- 총 35개 코너점

주의: 체스보드 크기는 "내부 코너 수"
      칸 수가 아님!
```

### 코너 검출

```python
import cv2
import numpy as np

# 체스보드 내부 코너 수
CHECKERBOARD = (7, 5)

# 이미지 로드
img = cv2.imread('chessboard.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 체스보드 코너 검출
ret, corners = cv2.findChessboardCorners(
    gray,
    CHECKERBOARD,
    flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
          cv2.CALIB_CB_FAST_CHECK +
          cv2.CALIB_CB_NORMALIZE_IMAGE
)

if ret:
    print(f"코너 검출 성공: {corners.shape[0]}개")

    # 서브픽셀 정밀도로 코너 위치 개선
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 코너 시각화
    img_corners = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
    cv2.imshow('Corners', img_corners)
    cv2.waitKey(0)
else:
    print("코너 검출 실패")
```

### 검출 플래그 옵션

```
findChessboardCorners 플래그:

┌────────────────────────────────┬─────────────────────────────────┐
│ 플래그                         │ 설명                            │
├────────────────────────────────┼─────────────────────────────────┤
│ CALIB_CB_ADAPTIVE_THRESH       │ 적응형 이진화 사용              │
│                                │ (조명 변화에 강건)              │
├────────────────────────────────┼─────────────────────────────────┤
│ CALIB_CB_NORMALIZE_IMAGE       │ 이미지 정규화                   │
│                                │ (대비 개선)                     │
├────────────────────────────────┼─────────────────────────────────┤
│ CALIB_CB_FILTER_QUADS          │ 잘못된 사각형 필터링            │
│                                │ (오검출 감소)                   │
├────────────────────────────────┼─────────────────────────────────┤
│ CALIB_CB_FAST_CHECK            │ 빠른 검사로 실패 조기 판단      │
│                                │ (속도 향상)                     │
└────────────────────────────────┴─────────────────────────────────┘

권장 조합:
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
```

### 여러 이미지에서 코너 수집

```python
import cv2
import numpy as np
import glob

def collect_calibration_points(image_paths, checkerboard_size):
    """여러 이미지에서 캘리브레이션 포인트 수집"""

    # 3D 점 (세계 좌표): z=0인 평면
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3),
                    np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0],
                           0:checkerboard_size[1]].T.reshape(-1, 2)

    # 실제 크기 적용 (예: 각 칸이 25mm)
    square_size = 25.0  # mm
    objp *= square_size

    obj_points = []  # 3D 점들
    img_points = []  # 2D 점들
    img_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.001)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        # 체스보드 코너 검출
        ret, corners = cv2.findChessboardCorners(
            gray, checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # 서브픽셀 정밀도 개선
            corners = cv2.cornerSubPix(gray, corners, (11, 11),
                                       (-1, -1), criteria)

            obj_points.append(objp)
            img_points.append(corners)

            print(f"성공: {img_path}")
        else:
            print(f"실패: {img_path}")

    print(f"\n총 {len(obj_points)}/{len(image_paths)} 이미지 사용")
    return obj_points, img_points, img_size

# 사용 예
images = glob.glob('calibration_images/*.jpg')
obj_points, img_points, img_size = collect_calibration_points(
    images, (7, 5)
)
```

---

## 4. calibrateCamera()

### 카메라 캘리브레이션 수행

```python
import cv2
import numpy as np
import glob

def calibrate_camera(image_folder, checkerboard_size=(7, 5),
                     square_size=25.0):
    """카메라 캘리브레이션 수행"""

    # 3D 객체 점
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3),
                    np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0],
                           0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []
    img_points = []

    images = glob.glob(f'{image_folder}/*.jpg')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.001)

    img_size = None
    valid_images = []

    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray, checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11),
                                       (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
            valid_images.append(img_path)

    if len(obj_points) < 10:
        print(f"경고: 이미지 수가 적습니다 ({len(obj_points)}개)")

    # 캘리브레이션 수행
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,     # 3D 점들
        img_points,     # 2D 점들
        img_size,       # 이미지 크기
        None,           # 초기 카메라 행렬 (None이면 자동 계산)
        None,           # 초기 왜곡 계수
        flags=cv2.CALIB_FIX_K3  # k3 고정 (선택사항)
    )

    print(f"\n캘리브레이션 완료")
    print(f"재투영 오차: {ret:.4f} 픽셀")
    print(f"\n카메라 행렬:\n{camera_matrix}")
    print(f"\n왜곡 계수:\n{dist_coeffs.ravel()}")

    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'reprojection_error': ret,
        'valid_images': valid_images
    }

# 사용 예
result = calibrate_camera('calibration_images', (7, 5), 25.0)
```

### 캘리브레이션 결과 저장/로드

```python
import cv2
import numpy as np
import json

def save_calibration(filepath, camera_matrix, dist_coeffs):
    """캘리브레이션 결과 저장"""

    # NumPy 배열을 리스트로 변환 (JSON 호환)
    data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist()
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"저장됨: {filepath}")

def load_calibration(filepath):
    """캘리브레이션 결과 로드"""

    with open(filepath, 'r') as f:
        data = json.load(f)

    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeffs'])

    return camera_matrix, dist_coeffs

# 또는 OpenCV FileStorage 사용
def save_calibration_yaml(filepath, camera_matrix, dist_coeffs):
    """YAML 형식으로 저장"""
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
    fs.write('camera_matrix', camera_matrix)
    fs.write('dist_coeffs', dist_coeffs)
    fs.release()

def load_calibration_yaml(filepath):
    """YAML 형식에서 로드"""
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('camera_matrix').mat()
    dist_coeffs = fs.getNode('dist_coeffs').mat()
    fs.release()
    return camera_matrix, dist_coeffs

# 사용 예
save_calibration('camera_calib.json', result['camera_matrix'],
                 result['dist_coeffs'])
camera_matrix, dist_coeffs = load_calibration('camera_calib.json')
```

### 캘리브레이션 플래그

```
calibrateCamera 플래그 옵션:

┌──────────────────────────────┬──────────────────────────────────┐
│ 플래그                       │ 설명                             │
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_USE_INTRINSIC_GUESS    │ 초기 카메라 행렬 사용            │
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_FIX_PRINCIPAL_POINT    │ 주점 고정                        │
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_FIX_ASPECT_RATIO       │ fx/fy 비율 고정                  │
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_ZERO_TANGENT_DIST      │ 접선 왜곡 = 0으로 고정           │
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_FIX_K1, K2, K3, ...    │ 특정 왜곡 계수 고정              │
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_RATIONAL_MODEL         │ 고차 왜곡 모델 사용 (k4,k5,k6)   │
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_FIX_S1_S2_S3_S4        │ 얇은 렌즈 왜곡 계수 고정         │
└──────────────────────────────┴──────────────────────────────────┘

일반적인 조합:
# 기본 캘리브레이션
flags = 0

# 간단한 왜곡 모델 (k1, k2만)
flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST

# 고정밀 캘리브레이션
flags = cv2.CALIB_RATIONAL_MODEL
```

---

## 5. undistort(): 왜곡 보정

### 기본 왜곡 보정

```python
import cv2
import numpy as np

def undistort_image(img, camera_matrix, dist_coeffs):
    """이미지 왜곡 보정"""

    h, w = img.shape[:2]

    # 새 카메라 행렬 계산 (최적화된 영역)
    # alpha: 0=모든 왜곡 픽셀 제거, 1=모든 원본 픽셀 유지
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
    )

    # 왜곡 보정
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs,
                                 None, new_camera_matrix)

    # ROI로 크롭 (선택사항)
    x, y, w, h = roi
    if all([x, y, w, h]):  # ROI가 유효한 경우
        undistorted_cropped = undistorted[y:y+h, x:x+w]
        return undistorted, undistorted_cropped

    return undistorted, undistorted

# 사용 예
img = cv2.imread('distorted.jpg')
camera_matrix, dist_coeffs = load_calibration('camera_calib.json')
undistorted, cropped = undistort_image(img, camera_matrix, dist_coeffs)

cv2.imshow('Original', img)
cv2.imshow('Undistorted', undistorted)
cv2.imshow('Cropped', cropped)
cv2.waitKey(0)
```

### 리맵핑 방식 (더 효율적)

```python
import cv2
import numpy as np

class UndistortMapper:
    """리맵핑 기반 왜곡 보정기 (비디오용)"""

    def __init__(self, camera_matrix, dist_coeffs, img_size, alpha=1):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        w, h = img_size

        # 새 카메라 행렬
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
        )

        # 리맵핑 맵 계산 (한 번만)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None,
            self.new_camera_matrix, (w, h), cv2.CV_32FC1
        )

    def undistort(self, img, crop=True):
        """빠른 왜곡 보정 (리맵핑 사용)"""
        undistorted = cv2.remap(img, self.mapx, self.mapy,
                                cv2.INTER_LINEAR)

        if crop and all(self.roi):
            x, y, w, h = self.roi
            return undistorted[y:y+h, x:x+w]

        return undistorted

# 비디오 처리 예
cap = cv2.VideoCapture(0)

# 첫 프레임으로 크기 확인
ret, frame = cap.read()
h, w = frame.shape[:2]

# 리맵퍼 초기화 (한 번만)
camera_matrix, dist_coeffs = load_calibration('camera_calib.json')
mapper = UndistortMapper(camera_matrix, dist_coeffs, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 빠른 왜곡 보정
    undistorted = mapper.undistort(frame)

    cv2.imshow('Original', frame)
    cv2.imshow('Undistorted', undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### 왜곡 보정 시각화

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_undistortion(img, camera_matrix, dist_coeffs):
    """왜곡 보정 전후 비교 시각화"""

    h, w = img.shape[:2]

    # 왜곡 보정
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)

    # 격자 오버레이 생성
    def add_grid(image, step=50):
        result = image.copy()
        for i in range(0, image.shape[1], step):
            cv2.line(result, (i, 0), (i, image.shape[0]), (0, 255, 0), 1)
        for i in range(0, image.shape[0], step):
            cv2.line(result, (0, i), (image.shape[1], i), (0, 255, 0), 1)
        return result

    img_grid = add_grid(img)
    undistorted_grid = add_grid(undistorted)

    # 나란히 표시
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original (Distorted)')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(undistorted_grid, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Undistorted')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    return undistorted

# 사용 예
img = cv2.imread('distorted.jpg')
undistorted = visualize_undistortion(img, camera_matrix, dist_coeffs)
```

---

## 6. 재투영 오차

### 재투영 오차 계산

```
재투영 오차 (Reprojection Error):
캘리브레이션 품질을 나타내는 지표

과정:
1. 알려진 3D 점을 캘리브레이션 결과로 2D로 투영
2. 검출된 2D 코너와의 거리 계산
3. 모든 점에 대한 평균 거리

    실제 검출 위치 ●────────● 재투영 위치
                   │ 오차   │
                   └────────┘

좋은 캘리브레이션: 재투영 오차 < 0.5 픽셀
보통: 0.5 ~ 1.0 픽셀
나쁨: > 1.0 픽셀
```

### 재투영 오차 상세 분석

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_reprojection_error(obj_points, img_points,
                                  rvecs, tvecs,
                                  camera_matrix, dist_coeffs):
    """상세 재투영 오차 계산"""

    errors = []
    per_image_errors = []

    for i in range(len(obj_points)):
        # 3D 점을 2D로 재투영
        projected_points, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i],
            camera_matrix, dist_coeffs
        )

        # 오차 계산
        error = cv2.norm(img_points[i], projected_points, cv2.NORM_L2)
        error /= len(projected_points)

        per_image_errors.append(error)

        # 각 점별 오차
        for j in range(len(projected_points)):
            pt_error = np.linalg.norm(
                img_points[i][j] - projected_points[j]
            )
            errors.append(pt_error)

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)

    print(f"재투영 오차 통계:")
    print(f"  평균: {mean_error:.4f} 픽셀")
    print(f"  표준편차: {std_error:.4f}")
    print(f"  최대: {max_error:.4f}")

    return {
        'mean': mean_error,
        'std': std_error,
        'max': max_error,
        'per_point': errors,
        'per_image': per_image_errors
    }

def visualize_reprojection_error(error_data):
    """재투영 오차 시각화"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 포인트별 오차 히스토그램
    axes[0].hist(error_data['per_point'], bins=50, edgecolor='black')
    axes[0].axvline(error_data['mean'], color='r', linestyle='--',
                    label=f"Mean: {error_data['mean']:.3f}")
    axes[0].set_xlabel('Reprojection Error (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].legend()

    # 이미지별 오차
    axes[1].bar(range(len(error_data['per_image'])),
                error_data['per_image'])
    axes[1].axhline(error_data['mean'], color='r', linestyle='--')
    axes[1].set_xlabel('Image Index')
    axes[1].set_ylabel('Mean Error (pixels)')
    axes[1].set_title('Per-Image Error')

    plt.tight_layout()
    plt.show()

# 사용 예
# calibration 결과에서
error_data = calculate_reprojection_error(
    obj_points, img_points,
    result['rvecs'], result['tvecs'],
    result['camera_matrix'], result['dist_coeffs']
)
visualize_reprojection_error(error_data)
```

### 캘리브레이션 품질 개선

```python
def improve_calibration(obj_points, img_points, img_size,
                        camera_matrix, dist_coeffs,
                        rvecs, tvecs, threshold=1.0):
    """높은 오차의 이미지를 제거하여 캘리브레이션 개선"""

    # 각 이미지의 재투영 오차 계산
    per_image_errors = []

    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i],
            camera_matrix, dist_coeffs
        )
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2)
        error /= len(projected)
        per_image_errors.append(error)

    # 임계값 이하의 이미지만 선택
    good_indices = [i for i, e in enumerate(per_image_errors)
                    if e < threshold]

    if len(good_indices) < 5:
        print("경고: 좋은 이미지가 너무 적습니다")
        return None

    # 선택된 이미지로 재캘리브레이션
    good_obj = [obj_points[i] for i in good_indices]
    good_img = [img_points[i] for i in good_indices]

    ret, new_camera_matrix, new_dist_coeffs, new_rvecs, new_tvecs = \
        cv2.calibrateCamera(good_obj, good_img, img_size, None, None)

    print(f"제거된 이미지: {len(obj_points) - len(good_indices)}")
    print(f"새 재투영 오차: {ret:.4f}")

    return {
        'camera_matrix': new_camera_matrix,
        'dist_coeffs': new_dist_coeffs,
        'reprojection_error': ret,
        'used_images': len(good_indices)
    }
```

### 실시간 캘리브레이션

```python
import cv2
import numpy as np

class RealtimeCalibrator:
    """실시간 카메라 캘리브레이션"""

    def __init__(self, checkerboard_size=(7, 5), square_size=25.0,
                 min_images=15):
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.min_images = min_images

        # 3D 객체 점
        self.objp = np.zeros(
            (checkerboard_size[0] * checkerboard_size[1], 3),
            np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0:checkerboard_size[0],
            0:checkerboard_size[1]
        ].T.reshape(-1, 2)
        self.objp *= square_size

        self.obj_points = []
        self.img_points = []
        self.img_size = None

        self.calibrated = False
        self.camera_matrix = None
        self.dist_coeffs = None

        self.criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def process_frame(self, frame):
        """프레임 처리 및 코너 검출"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.img_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray, self.checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        display = frame.copy()

        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11),
                                       (-1, -1), self.criteria)
            cv2.drawChessboardCorners(display, self.checkerboard_size,
                                       corners, ret)

        # 상태 표시
        status = f"Images: {len(self.obj_points)}/{self.min_images}"
        cv2.putText(display, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.calibrated:
            cv2.putText(display, "CALIBRATED", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return display, ret, corners

    def capture(self, corners):
        """캘리브레이션용 프레임 캡처"""
        if corners is not None:
            self.obj_points.append(self.objp)
            self.img_points.append(corners)
            print(f"캡처: {len(self.obj_points)}번째")

            # 충분한 이미지가 모이면 자동 캘리브레이션
            if len(self.obj_points) >= self.min_images and not self.calibrated:
                self.calibrate()

    def calibrate(self):
        """캘리브레이션 수행"""
        if len(self.obj_points) < self.min_images:
            print(f"이미지가 부족합니다: {len(self.obj_points)}/{self.min_images}")
            return False

        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = \
            cv2.calibrateCamera(
                self.obj_points, self.img_points,
                self.img_size, None, None
            )

        self.calibrated = True
        print(f"\n캘리브레이션 완료!")
        print(f"재투영 오차: {ret:.4f}")
        print(f"카메라 행렬:\n{self.camera_matrix}")

        return True

    def undistort(self, frame):
        """왜곡 보정"""
        if not self.calibrated:
            return frame

        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

# 사용 예
cap = cv2.VideoCapture(0)
calibrator = RealtimeCalibrator(min_images=15)

print("스페이스바: 캡처, c: 캘리브레이션, u: 왜곡 보정 토글, q: 종료")

show_undistorted = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display, found, corners = calibrator.process_frame(frame)

    if show_undistorted and calibrator.calibrated:
        display = calibrator.undistort(display)
        cv2.putText(display, "UNDISTORTED", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Calibration', display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and found:
        calibrator.capture(corners)
    elif key == ord('c'):
        calibrator.calibrate()
    elif key == ord('u'):
        show_undistorted = not show_undistorted

cap.release()
cv2.destroyAllWindows()

# 결과 저장
if calibrator.calibrated:
    save_calibration('camera_calib.json',
                    calibrator.camera_matrix,
                    calibrator.dist_coeffs)
```

---

## 7. 연습 문제

### 문제 1: 캘리브레이션 이미지 자동 수집

웹캠에서 자동으로 캘리브레이션 이미지를 수집하는 프로그램을 작성하세요.

**요구사항**:
- 체스보드 검출 시 자동 캡처 (일정 시간 간격)
- 다양한 각도/위치에서 캡처되도록 안내
- 수집된 이미지 품질 확인 (블러 제거)
- 최소 15-20장 수집

<details>
<summary>힌트</summary>

```python
import time

last_capture_time = 0
min_interval = 2.0  # 최소 캡처 간격 (초)

# 블러 검출
def is_blurry(img, threshold=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# 자동 캡처 조건
if (found and
    time.time() - last_capture_time > min_interval and
    not is_blurry(frame)):
    # 캡처
```

</details>

### 문제 2: 어안 렌즈 캘리브레이션

어안 (fisheye) 렌즈 카메라를 캘리브레이션하세요.

**요구사항**:
- cv2.fisheye 모듈 사용
- 어안 특유의 극심한 왜곡 보정
- 일반 모델과 결과 비교

<details>
<summary>힌트</summary>

```python
# 어안 캘리브레이션
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_points]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_points]

flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
         cv2.fisheye.CALIB_FIX_SKEW)

ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    obj_points, img_points, img_size, K, D,
    rvecs, tvecs, flags
)

# 어안 왜곡 보정
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), K, img_size, cv2.CV_16SC2
)
```

</details>

### 문제 3: 스테레오 캘리브레이션

두 대의 카메라를 동시에 캘리브레이션하세요.

**요구사항**:
- 각 카메라 개별 캘리브레이션
- 스테레오 캘리브레이션 (상대 위치 계산)
- 스테레오 정류 (rectification)

<details>
<summary>힌트</summary>

```python
# 스테레오 캘리브레이션
ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
    obj_points,
    img_points_left, img_points_right,
    camera_matrix_left, dist_coeffs_left,
    camera_matrix_right, dist_coeffs_right,
    img_size,
    flags=cv2.CALIB_FIX_INTRINSIC
)

# 스테레오 정류
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    M1, d1, M2, d2, img_size, R, T
)
```

</details>

### 문제 4: 원형 패턴 캘리브레이션

체스보드 대신 원형 패턴을 사용한 캘리브레이션을 구현하세요.

**요구사항**:
- cv2.findCirclesGrid() 사용
- 대칭/비대칭 원형 그리드 지원
- 체스보드 결과와 비교

<details>
<summary>힌트</summary>

```python
# 원형 그리드 검출
# 대칭 그리드
ret, centers = cv2.findCirclesGrid(
    gray, (4, 11),
    flags=cv2.CALIB_CB_SYMMETRIC_GRID
)

# 비대칭 그리드 (더 정확)
ret, centers = cv2.findCirclesGrid(
    gray, (4, 11),
    flags=cv2.CALIB_CB_ASYMMETRIC_GRID
)

# 비대칭 그리드의 3D 점
objp = np.zeros((4*11, 3), np.float32)
for i in range(11):
    for j in range(4):
        objp[i*4 + j] = [j*2 + (i%2), i, 0]
```

</details>

### 문제 5: 캘리브레이션 품질 평가 도구

캘리브레이션 결과의 품질을 종합적으로 평가하는 도구를 만드세요.

**요구사항**:
- 재투영 오차 분포 시각화
- 왜곡 계수 분석
- 이상치 이미지 검출 및 제거
- 캘리브레이션 신뢰도 점수

<details>
<summary>힌트</summary>

```python
class CalibrationEvaluator:
    def evaluate(self, result, obj_points, img_points):
        # 재투영 오차 분포
        errors = self.compute_per_point_errors(...)

        # 이상치 검출 (2 표준편차 이상)
        outliers = errors > np.mean(errors) + 2*np.std(errors)

        # 왜곡 계수 분석
        k1, k2, p1, p2, k3 = result['dist_coeffs'].ravel()

        # 신뢰도 점수
        score = 100
        score -= min(50, result['reprojection_error'] * 50)  # 오차 페널티
        score -= min(30, outlier_ratio * 100)  # 이상치 페널티

        return {'score': score, ...}
```

</details>

---

## 다음 단계

- [19_DNN_Module.md](./19_DNN_Module.md) - cv2.dnn, YOLO, SSD

---

## 참고 자료

- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- Zhang, Z. (2000). "A Flexible New Technique for Camera Calibration"
- [OpenCV Fisheye Module](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)
- [Calibration Pattern Generator](https://calib.io/pages/camera-calibration-pattern-generator)
