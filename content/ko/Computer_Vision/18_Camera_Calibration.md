# 카메라 캘리브레이션 (Camera Calibration)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 핀홀 카메라 모델(Pinhole Camera Model)과 내부 파라미터(Intrinsic Parameters)(초점 거리, 주점, 비틀림)를 설명할 수 있습니다.
2. 렌즈 왜곡(Lens Distortion)의 종류(방사형, 접선형)와 수학적 모델을 기술할 수 있습니다.
3. `findChessboardCorners()`와 `calibrateCamera()`를 사용하여 체스보드 패턴으로 카메라 캘리브레이션을 구현할 수 있습니다.
4. 캘리브레이션 결과를 활용해 `undistort()`로 이미지의 렌즈 왜곡을 보정할 수 있습니다.
5. 재투영 오차(Reprojection Error)를 계산하고 해석하여 캘리브레이션 품질을 평가할 수 있습니다.

---

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
Pinhole Camera Model:
Projects 3D world coordinates to 2D image coordinates

        3D World
           P(X, Y, Z)
              │
              │
              ▼
       ┌──────────────┐
       │    Lens      │  ← Camera
       └──────────────┘
              │
              │  Focal length f
              ▼
       ┌──────────────┐
       │ Image Plane  │  → p(u, v)
       │      ●       │
       └──────────────┘

Projection formula:
u = fx * (X/Z) + cx
v = fy * (Y/Z) + cy

- (X, Y, Z): 3D point in camera coordinates
- (u, v): 2D image coordinates (pixels)
- fx, fy: Focal length (in pixels)
- (cx, cy): Principal point

Derivation via similar triangles: Consider a side view where
a 3D point at position (X, Z) projects through the pinhole
onto the image plane at distance f. The similar triangles
formed by (0,0)-(0,f)-(x_img,f) and (0,0)-(0,Z)-(X,Z) give
x_img/f = X/Z, so x_img = f·X/Z. Converting to pixel
coordinates with focal length in pixels (fx) and adding the
principal point offset (cx) yields u = fx·(X/Z) + cx.
```

### 카메라 행렬 (Intrinsic Matrix)

```
Camera Intrinsic Parameter Matrix K:

     ┌             ┐
     │ fx   0   cx │
K =  │  0  fy   cy │
     │  0   0    1 │
     └             ┘

Parameter Description:
┌────────────────────────────────────────────────────────────┐
│ fx, fy: Focal length                                       │
│         - In pixel units                                   │
│         - fx = f / pixel_width                             │
│         - fy = f / pixel_height                            │
│         - Generally fx ≈ fy (square pixels)                │
│                                                            │
│ cx, cy: Principal point                                    │
│         - Point where optical axis meets image plane       │
│         - Ideally at image center (width/2, height/2)      │
│         - In practice, slight offset exists                │
│                                                            │
│ skew: Skew coefficient (usually 0)                         │
│         - At position (0,1) in matrix                      │
│         - Non-orthogonality of pixels                      │
└────────────────────────────────────────────────────────────┘

Example (typical webcam):
     ┌                    ┐
     │ 800    0    320    │
K =  │   0  800    240    │   (640x480 resolution)
     │   0    0      1    │
     └                    ┘
```

### 외부 파라미터

```
Camera Extrinsic Parameters:
Transformation between camera coordinate system and world coordinate system

World coordinates → Camera coordinates:

[X_cam]       [X_world]
[Y_cam] = R * [Y_world] + t
[Z_cam]       [Z_world]

R: 3x3 rotation matrix
t: 3x1 translation vector

Complete projection:

     ┌   ┐       ┌             ┐   ┌       ┐   ┌   ┐
s *  │ u │   =   │ fx   0   cx │ * │ R | t │ * │ X │
     │ v │       │  0  fy   cy │   │   |   │   │ Y │
     │ 1 │       │  0   0    1 │   │   |   │   │ Z │
     └   ┘       └             ┘   └       ┘   │ 1 │
                                               └   ┘
    Image         Camera Matrix    Extrinsic    World
                                   Matrix       Coordinates
```

---

## 2. 렌즈 왜곡

### 왜곡 종류

```
Lens Distortion:

1. Radial Distortion
   - Caused by lens curvature
   - Barrel: Convex distortion (wide-angle lens)
   - Pincushion: Concave distortion (telephoto lens)

   Original       Barrel          Pincushion
   ┌───────┐    ╭───────╮      ┌───────┐
   │       │    │       │      ╰       ╯
   │       │    │       │      │       │
   │       │    │       │      ╭       ╮
   └───────┘    ╰───────╯      └───────┘

2. Tangential Distortion
   - Caused by lens and image sensor misalignment
   - Image appears tilted or twisted

   ┌───────┐      ┌───────┐
   │       │      │╲      │
   │       │  →   │ ╲     │
   │       │      │  ╲    │
   └───────┘      └───────┘
```

### 왜곡 모델 수식

```
Radial Distortion:

x_distorted = x * (1 + k1*r² + k2*r⁴ + k3*r⁶)
y_distorted = y * (1 + k1*r² + k2*r⁴ + k3*r⁶)

Where:
- r² = x² + y² (distance in normalized image coordinates)
- k1, k2, k3: Radial distortion coefficients

Intuition: The polynomial in r acts as a radial scaling factor.
k1 dominates near the image center; k2 and k3 only matter at the edges
where r is large. A negative k1 produces barrel distortion (center
expands outward), positive k1 produces pincushion (edges pulled in).
In practice, k1 alone often captures 95% of radial distortion.

Tangential Distortion:

x_distorted = x + [2*p1*x*y + p2*(r² + 2*x²)]
y_distorted = y + [p1*(r² + 2*y²) + 2*p2*x*y]

Where:
- p1, p2: Tangential distortion coefficients

Intuition: Tangential terms are cross-products of x and y, so they
shift pixels in a direction perpendicular to the radial direction —
geometrically equivalent to slightly tilting the sensor relative to
the lens. For most modern cameras p1, p2 are very small (<0.01).

Distortion coefficient vector:
distCoeffs = [k1, k2, p1, p2, k3]

(Some models add k4, k5, k6)
```

### 왜곡의 영향

```
Impact of severe distortion:

1. Straight lines appear curved
   Actual: ───────────
   Distorted: ╭─────────╮

2. Distance measurement errors
   - Errors increase toward image edges
   - Precise measurement impossible

3. 3D reconstruction errors
   - Depth errors in stereo vision
   - AR marker position errors

4. Object recognition degradation
   - Template matching failures
   - Feature matching accuracy decline
```

---

## 3. findChessboardCorners()

### 체스보드 패턴

```
Why use a chessboard pattern:

1. Accurate corner detection — black/white corners are saddle points in
   intensity, giving a unique sub-pixel localizable feature regardless of
   viewing angle or distance (unlike circle centers, which shift under perspective)
2. Easy to create (printable)
3. Planar pattern facilitates calibration — knowing all corners lie on a
   single flat plane (Z=0) provides strong geometric constraints that allow
   computing both intrinsic and extrinsic parameters from a single view

Chessboard size definition:
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

Internal corner count: (7, 5)
- 7 horizontal, 5 vertical internal corners
- Total 35 corner points

Note: Chessboard size is "internal corner count"
      Not the number of squares!
```

### 코너 검출

```python
import cv2
import numpy as np

# Chessboard internal corner count
CHECKERBOARD = (7, 5)

# Load image
img = cv2.imread('chessboard.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect chessboard corners
# ADAPTIVE_THRESH + NORMALIZE_IMAGE improve robustness to uneven lighting —
# the board is often lit differently across its surface
ret, corners = cv2.findChessboardCorners(
    gray,
    CHECKERBOARD,
    flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
          cv2.CALIB_CB_FAST_CHECK +    # Reject images quickly if no checkerboard
                                        # pattern is even plausible — saves ~80% time
                                        # on frames without the target
          cv2.CALIB_CB_NORMALIZE_IMAGE
)

if ret:
    print(f"Corner detection successful: {corners.shape[0]} corners")

    # Refine corner positions to subpixel accuracy —
    # findChessboardCorners returns integer-pixel corners; cornerSubPix
    # iteratively minimizes gradient direction in a (11,11) window to
    # achieve ~0.1-pixel precision, which is essential for low reprojection error
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Visualize corners
    img_corners = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
    cv2.imshow('Corners', img_corners)
    cv2.waitKey(0)
else:
    print("Corner detection failed")
```

### 검출 플래그 옵션

```
findChessboardCorners flags:

┌────────────────────────────────┬─────────────────────────────────┐
│ Flag                           │ Description                     │
├────────────────────────────────┼─────────────────────────────────┤
│ CALIB_CB_ADAPTIVE_THRESH       │ Use adaptive binarization       │
│                                │ (robust to illumination changes)│
├────────────────────────────────┼─────────────────────────────────┤
│ CALIB_CB_NORMALIZE_IMAGE       │ Normalize image                 │
│                                │ (improve contrast)              │
├────────────────────────────────┼─────────────────────────────────┤
│ CALIB_CB_FILTER_QUADS          │ Filter incorrect quadrilaterals │
│                                │ (reduce false detections)       │
├────────────────────────────────┼─────────────────────────────────┤
│ CALIB_CB_FAST_CHECK            │ Fast check for early failure    │
│                                │ (speed improvement)             │
└────────────────────────────────┴─────────────────────────────────┘

Recommended combination:
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
```

### 여러 이미지에서 코너 수집

```python
import cv2
import numpy as np
import glob

def collect_calibration_points(image_paths, checkerboard_size):
    """Collect calibration points from multiple images.
    Multiple views are required because a single view only determines
    the homography — you need diverse angles/distances so that different
    distortion amounts are observed, making k1..k3 well-constrained.
    """

    # 3D points (world coordinates): z=0 plane
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3),
                    np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0],
                           0:checkerboard_size[1]].T.reshape(-1, 2)

    # Apply actual size (e.g., each square is 25mm)
    # Scaling objp by the physical square size makes tvecs (translation vectors)
    # come out in mm, enabling real-world distance measurement later
    square_size = 25.0  # mm
    objp *= square_size

    obj_points = []  # 3D points (same for every image — pattern doesn't change)
    img_points = []  # 2D points (different per image — perspective changes)
    img_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.001)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        # Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Refine to subpixel accuracy
            corners = cv2.cornerSubPix(gray, corners, (11, 11),
                                       (-1, -1), criteria)

            obj_points.append(objp)
            img_points.append(corners)

            print(f"Success: {img_path}")
        else:
            print(f"Failed: {img_path}")

    print(f"\nUsing {len(obj_points)}/{len(image_paths)} images total")
    return obj_points, img_points, img_size

# Usage example
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
    """Perform camera calibration"""

    # 3D object points
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

    # Calibration needs enough views to constrain all unknowns:
    # 5 intrinsic + 5 distortion params = 10 unknowns, so ≥10 images
    # from diverse angles is the practical minimum for stable results
    if len(obj_points) < 10:
        print(f"Warning: Low number of images ({len(obj_points)})")

    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,     # 3D points
        img_points,     # 2D points
        img_size,       # Image size
        None,           # Initial camera matrix (None for automatic calculation)
        None,           # Initial distortion coefficients
        flags=cv2.CALIB_FIX_K3  # k3 models extreme distortion at image corners;
                                 # fixing it at 0 prevents overfitting when the
                                 # calibration board doesn't reach the frame edges
    )

    print(f"\nCalibration complete")
    print(f"Reprojection error: {ret:.4f} pixels")
    print(f"\nCamera matrix:\n{camera_matrix}")
    print(f"\nDistortion coefficients:\n{dist_coeffs.ravel()}")

    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'reprojection_error': ret,
        'valid_images': valid_images
    }

# Usage example
result = calibrate_camera('calibration_images', (7, 5), 25.0)
```

### 캘리브레이션 결과 저장/로드

```python
import cv2
import numpy as np
import json

def save_calibration(filepath, camera_matrix, dist_coeffs):
    """Save calibration results"""

    # Convert NumPy arrays to lists (JSON compatible)
    data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist()
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {filepath}")

def load_calibration(filepath):
    """Load calibration results"""

    with open(filepath, 'r') as f:
        data = json.load(f)

    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeffs'])

    return camera_matrix, dist_coeffs

# Or use OpenCV FileStorage
def save_calibration_yaml(filepath, camera_matrix, dist_coeffs):
    """Save in YAML format"""
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
    fs.write('camera_matrix', camera_matrix)
    fs.write('dist_coeffs', dist_coeffs)
    fs.release()

def load_calibration_yaml(filepath):
    """Load from YAML format"""
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('camera_matrix').mat()
    dist_coeffs = fs.getNode('dist_coeffs').mat()
    fs.release()
    return camera_matrix, dist_coeffs

# Usage example
save_calibration('camera_calib.json', result['camera_matrix'],
                 result['dist_coeffs'])
camera_matrix, dist_coeffs = load_calibration('camera_calib.json')
```

### 캘리브레이션 플래그

```
calibrateCamera flag options:

┌──────────────────────────────┬──────────────────────────────────┐
│ Flag                         │ Description                      │
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_USE_INTRINSIC_GUESS    │ Use initial camera matrix        │
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_FIX_PRINCIPAL_POINT    │ Fix principal point              │
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_FIX_ASPECT_RATIO       │ Fix fx/fy aspect ratio           │
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_ZERO_TANGENT_DIST      │ Fix tangential distortion = 0    │
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_FIX_K1, K2, K3, ...    │ Fix specific distortion coeffs   │
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_RATIONAL_MODEL         │ Use higher-order model (k4,k5,k6)│
├──────────────────────────────┼──────────────────────────────────┤
│ CALIB_FIX_S1_S2_S3_S4        │ Fix thin prism distortion coeffs │
└──────────────────────────────┴──────────────────────────────────┘

Common combinations:
# Basic calibration
flags = 0

# Simple distortion model (k1, k2 only)
flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST

# High precision calibration
flags = cv2.CALIB_RATIONAL_MODEL
```

---

## 5. undistort(): 왜곡 보정

### 기본 왜곡 보정

```python
import cv2
import numpy as np

def undistort_image(img, camera_matrix, dist_coeffs):
    """Image distortion correction"""

    h, w = img.shape[:2]

    # Compute new camera matrix (optimized region)
    # alpha controls the trade-off between field-of-view and black border removal:
    # alpha=0 crops away all black border pixels (smaller FoV, no wasted pixels)
    # alpha=1 preserves all original pixels (full FoV but black borders remain)
    # For measurement tasks use alpha=1; for display use alpha=0
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
    )

    # Undistort
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs,
                                 None, new_camera_matrix)

    # Crop to ROI (optional)
    x, y, w, h = roi
    if all([x, y, w, h]):  # If ROI is valid
        undistorted_cropped = undistorted[y:y+h, x:x+w]
        return undistorted, undistorted_cropped

    return undistorted, undistorted

# Usage example
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
    """Remapping-based undistortion (for video)"""

    def __init__(self, camera_matrix, dist_coeffs, img_size, alpha=1):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        w, h = img_size

        # New camera matrix
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
        )

        # Compute remapping maps once and reuse every frame —
        # cv2.undistort() recomputes the map internally each call (O(W×H) work),
        # so pre-computing with initUndistortRectifyMap saves ~30-40% CPU for
        # video streams where the camera parameters don't change between frames
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None,
            self.new_camera_matrix, (w, h), cv2.CV_32FC1
        )

    def undistort(self, img, crop=True):
        """Fast undistortion (using remapping)"""
        # cv2.remap applies precomputed pixel mappings with bilinear interpolation —
        # INTER_LINEAR gives smooth results for non-integer remapped coordinates
        undistorted = cv2.remap(img, self.mapx, self.mapy,
                                cv2.INTER_LINEAR)

        if crop and all(self.roi):
            x, y, w, h = self.roi
            return undistorted[y:y+h, x:x+w]

        return undistorted

# Video processing example
cap = cv2.VideoCapture(0)

# Check size with first frame
ret, frame = cap.read()
h, w = frame.shape[:2]

# Initialize remapper (only once)
camera_matrix, dist_coeffs = load_calibration('camera_calib.json')
mapper = UndistortMapper(camera_matrix, dist_coeffs, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Fast undistortion
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
    """Visualize before/after undistortion comparison"""

    h, w = img.shape[:2]

    # Undistort
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)

    # Create grid overlay
    def add_grid(image, step=50):
        result = image.copy()
        for i in range(0, image.shape[1], step):
            cv2.line(result, (i, 0), (i, image.shape[0]), (0, 255, 0), 1)
        for i in range(0, image.shape[0], step):
            cv2.line(result, (0, i), (image.shape[1], i), (0, 255, 0), 1)
        return result

    img_grid = add_grid(img)
    undistorted_grid = add_grid(undistorted)

    # Display side by side
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

# Usage example
img = cv2.imread('distorted.jpg')
undistorted = visualize_undistortion(img, camera_matrix, dist_coeffs)
```

---

## 6. 재투영 오차

### 재투영 오차 계산

```
Reprojection Error:
Metric indicating calibration quality

Process:
1. Project known 3D points to 2D using calibration results
2. Calculate distance to detected 2D corners
3. Average distance over all points

    Actual detected position ●────────● Reprojected position
                             │ Error  │
                             └────────┘

Good calibration: Reprojection error < 0.5 pixels
Acceptable: 0.5 ~ 1.0 pixels
Poor: > 1.0 pixels

Why 0.5 pixels is the threshold: a well-focused camera with sub-pixel
corner refinement has detection noise of ~0.1–0.2 pixels, so any
additional reprojection error above 0.5 pixels indicates the calibration
model itself is introducing errors (too few views, poor coverage, or
a board that wasn't held flat during capture).

Why analyze per-image errors: a single high-error image (blurred, board
not fully visible, or wrong pose) can inflate the overall mean. Identifying
and removing that image often reduces total error significantly without
recapturing new data.
```

### 재투영 오차 상세 분석

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_reprojection_error(obj_points, img_points,
                                  rvecs, tvecs,
                                  camera_matrix, dist_coeffs):
    """Detailed reprojection error calculation"""

    errors = []
    per_image_errors = []

    for i in range(len(obj_points)):
        # Reproject 3D points to 2D
        projected_points, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i],
            camera_matrix, dist_coeffs
        )

        # Calculate error
        error = cv2.norm(img_points[i], projected_points, cv2.NORM_L2)
        error /= len(projected_points)

        per_image_errors.append(error)

        # Per-point error
        for j in range(len(projected_points)):
            pt_error = np.linalg.norm(
                img_points[i][j] - projected_points[j]
            )
            errors.append(pt_error)

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)

    print(f"Reprojection error statistics:")
    print(f"  Mean: {mean_error:.4f} pixels")
    print(f"  Std Dev: {std_error:.4f}")
    print(f"  Max: {max_error:.4f}")

    return {
        'mean': mean_error,
        'std': std_error,
        'max': max_error,
        'per_point': errors,
        'per_image': per_image_errors
    }

def visualize_reprojection_error(error_data):
    """Visualize reprojection error"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Per-point error histogram
    axes[0].hist(error_data['per_point'], bins=50, edgecolor='black')
    axes[0].axvline(error_data['mean'], color='r', linestyle='--',
                    label=f"Mean: {error_data['mean']:.3f}")
    axes[0].set_xlabel('Reprojection Error (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].legend()

    # Per-image error
    axes[1].bar(range(len(error_data['per_image'])),
                error_data['per_image'])
    axes[1].axhline(error_data['mean'], color='r', linestyle='--')
    axes[1].set_xlabel('Image Index')
    axes[1].set_ylabel('Mean Error (pixels)')
    axes[1].set_title('Per-Image Error')

    plt.tight_layout()
    plt.show()

# Usage example
# From calibration results
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
    """Improve calibration by removing high-error images"""

    # Calculate reprojection error for each image
    per_image_errors = []

    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i],
            camera_matrix, dist_coeffs
        )
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2)
        error /= len(projected)
        per_image_errors.append(error)

    # Select only images below threshold
    good_indices = [i for i, e in enumerate(per_image_errors)
                    if e < threshold]

    if len(good_indices) < 5:
        print("Warning: Too few good images")
        return None

    # Re-calibrate with selected images
    good_obj = [obj_points[i] for i in good_indices]
    good_img = [img_points[i] for i in good_indices]

    ret, new_camera_matrix, new_dist_coeffs, new_rvecs, new_tvecs = \
        cv2.calibrateCamera(good_obj, good_img, img_size, None, None)

    print(f"Removed images: {len(obj_points) - len(good_indices)}")
    print(f"New reprojection error: {ret:.4f}")

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
    """Real-time camera calibration"""

    def __init__(self, checkerboard_size=(7, 5), square_size=25.0,
                 min_images=15):
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.min_images = min_images

        # 3D object points
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
        """Process frame and detect corners"""
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

        # Display status
        status = f"Images: {len(self.obj_points)}/{self.min_images}"
        cv2.putText(display, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.calibrated:
            cv2.putText(display, "CALIBRATED", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return display, ret, corners

    def capture(self, corners):
        """Capture frame for calibration"""
        if corners is not None:
            self.obj_points.append(self.objp)
            self.img_points.append(corners)
            print(f"Captured: #{len(self.obj_points)}")

            # Auto-calibrate when enough images collected
            if len(self.obj_points) >= self.min_images and not self.calibrated:
                self.calibrate()

    def calibrate(self):
        """Perform calibration"""
        if len(self.obj_points) < self.min_images:
            print(f"Insufficient images: {len(self.obj_points)}/{self.min_images}")
            return False

        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = \
            cv2.calibrateCamera(
                self.obj_points, self.img_points,
                self.img_size, None, None
            )

        self.calibrated = True
        print(f"\nCalibration complete!")
        print(f"Reprojection error: {ret:.4f}")
        print(f"Camera matrix:\n{self.camera_matrix}")

        return True

    def undistort(self, frame):
        """Undistort frame"""
        if not self.calibrated:
            return frame

        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

# Usage example
cap = cv2.VideoCapture(0)
calibrator = RealtimeCalibrator(min_images=15)

print("Space: Capture, c: Calibrate, u: Toggle undistortion, q: Quit")

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

# Save results
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
min_interval = 2.0  # Minimum capture interval (seconds)

# Blur detection
def is_blurry(img, threshold=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# Auto-capture conditions
if (found and
    time.time() - last_capture_time > min_interval and
    not is_blurry(frame)):
    # Capture
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
# Fisheye calibration
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

# Fisheye undistortion
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
# Stereo calibration
ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
    obj_points,
    img_points_left, img_points_right,
    camera_matrix_left, dist_coeffs_left,
    camera_matrix_right, dist_coeffs_right,
    img_size,
    flags=cv2.CALIB_FIX_INTRINSIC
)

# Stereo rectification
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
# Circular grid detection
# Symmetric grid
ret, centers = cv2.findCirclesGrid(
    gray, (4, 11),
    flags=cv2.CALIB_CB_SYMMETRIC_GRID
)

# Asymmetric grid (more accurate)
ret, centers = cv2.findCirclesGrid(
    gray, (4, 11),
    flags=cv2.CALIB_CB_ASYMMETRIC_GRID
)

# 3D points for asymmetric grid
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
        # Reprojection error distribution
        errors = self.compute_per_point_errors(...)

        # Outlier detection (beyond 2 standard deviations)
        outliers = errors > np.mean(errors) + 2*np.std(errors)

        # Distortion coefficient analysis
        k1, k2, p1, p2, k3 = result['dist_coeffs'].ravel()

        # Confidence score
        score = 100
        score -= min(50, result['reprojection_error'] * 50)  # Error penalty
        score -= min(30, outlier_ratio * 100)  # Outlier penalty

        return {'score': score, ...}
```

</details>

---

## 다음 단계

- [딥러닝 DNN 모듈 (Deep Neural Network Module)](./19_DNN_Module.md) - cv2.dnn, YOLO, SSD

---

## 참고 자료

- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- Zhang, Z. (2000). "A Flexible New Technique for Camera Calibration"
- [OpenCV Fisheye Module](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)
- [Calibration Pattern Generator](https://calib.io/pages/camera-calibration-pattern-generator)
