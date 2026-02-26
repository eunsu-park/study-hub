# 환경 설정 및 기초

## 개요

OpenCV(Open Source Computer Vision Library)는 실시간 컴퓨터 비전을 위한 오픈소스 라이브러리입니다. 이 문서에서는 OpenCV 설치부터 첫 프로그램 실행, 그리고 이미지 데이터의 기본 구조를 학습합니다.

**난이도**: ⭐ (입문)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. OpenCV 설치 및 개발 환경 구성
2. 버전 확인 및 첫 번째 프로그램 작성
3. OpenCV와 NumPy의 관계 이해
4. 이미지가 ndarray로 표현되는 개념 이해

---

## 목차

1. [OpenCV 소개](#1-opencv-소개)
2. [설치 방법](#2-설치-방법)
3. [개발 환경 설정](#3-개발-환경-설정)
4. [버전 확인 및 첫 프로그램](#4-버전-확인-및-첫-프로그램)
5. [OpenCV와 NumPy 관계](#5-opencv와-numpy-관계)
6. [이미지는 ndarray](#6-이미지는-ndarray)
7. [연습 문제](#7-연습-문제)
8. [다음 단계](#8-다음-단계)
9. [참고 자료](#9-참고-자료)

---

## 1. OpenCV 소개

### OpenCV란?

OpenCV는 Intel에서 시작하여 현재는 오픈소스로 관리되는 컴퓨터 비전 라이브러리입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenCV Application Areas                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │  Image      │   │  Object     │   │  Face       │          │
│   │  Processing │   │  Detection  │   │  Recognition│          │
│   │  Filtering  │   │  YOLO/SSD   │   │  Auth Systems│         │
│   │  Transform  │   │  Tracking   │   │  Emotion    │          │
│   └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                 │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │  Medical    │   │  Autonomous │   │  AR/VR      │          │
│   │  Imaging    │   │  Driving    │   │  Marker     │          │
│   │  CT/MRI     │   │  Lane Detect│   │  Recognition│          │
│   │  Diagnosis  │   │  Obstacles  │   │  3D Recon   │          │
│   └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### OpenCV의 특징

| 특징 | 설명 |
|------|------|
| **크로스 플랫폼** | Windows, macOS, Linux, Android, iOS 지원 |
| **다국어 지원** | C++, Python, Java 등 다양한 언어 바인딩 |
| **실시간 처리** | 최적화된 알고리즘으로 실시간 영상 처리 가능 |
| **풍부한 기능** | 2500개 이상의 최적화된 알고리즘 |
| **활발한 커뮤니티** | 방대한 문서와 예제, 활발한 개발 |

---

## 2. 설치 방법

### opencv-python vs opencv-contrib-python

```
┌────────────────────────────────────────────────────────────────┐
│                     OpenCV Python Packages                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   opencv-python                opencv-contrib-python           │
│   ┌──────────────────┐        ┌──────────────────────────┐    │
│   │  Main modules    │        │  Main modules            │    │
│   │  - core          │        │  - core                  │    │
│   │  - imgproc       │        │  - imgproc               │    │
│   │  - video         │        │  - video                 │    │
│   │  - highgui       │   ⊂    │  + Extra modules         │    │
│   │  - calib3d       │        │    - SIFT, SURF          │    │
│   │  - features2d    │        │    - xfeatures2d         │    │
│   │  - objdetect     │        │    - tracking            │    │
│   │  - dnn           │        │    - aruco               │    │
│   │  - ml            │        │    - face                │    │
│   └──────────────────┘        └──────────────────────────┘    │
│                                                                │
│   → Covers most features        → For additional algorithms   │
│   → Quick installation           → Includes patented/research │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### pip를 이용한 설치

```bash
# Basic installation: covers ~95% of use cases (filtering, detection, video I/O)
# Choose this when you don't need patented algorithms like SIFT/SURF
pip install opencv-python

# contrib adds extra modules including SIFT, SURF, ArUco, and face recognition
# Use this when you need research-grade features not yet in the main module
pip install opencv-contrib-python

# Install with NumPy and matplotlib (recommended for development)
# NumPy is required at runtime; matplotlib makes Jupyter-based visualization easy
pip install opencv-python numpy matplotlib

# Install specific version (useful when pinning for reproducible environments)
pip install opencv-python==4.8.0.76

# Upgrade
pip install --upgrade opencv-python
```

**주의사항**: `opencv-python`과 `opencv-contrib-python`을 동시에 설치하지 마세요. 충돌이 발생할 수 있습니다.

```bash
# Wrong (causes conflicts)
pip install opencv-python opencv-contrib-python  # ✗

# Correct (choose one)
pip install opencv-contrib-python  # ✓ (contrib includes basic features)
```

### 가상환경 사용 (권장)

```bash
# Create virtual environment
python -m venv opencv_env

# Activate (Windows)
opencv_env\Scripts\activate

# Activate (macOS/Linux)
source opencv_env/bin/activate

# Install packages
pip install opencv-contrib-python numpy matplotlib

# Deactivate
deactivate
```

---

## 3. 개발 환경 설정

### VSCode 설정

```
┌─────────────────────────────────────────────────────────────┐
│                   VSCode Recommended Settings                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Essential Extensions:                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  1. Python (Microsoft)        - Python support      │   │
│   │  2. Pylance                   - Code analysis       │   │
│   │  3. Jupyter                   - Notebook support    │   │
│   │  4. Python Image Preview      - Image preview       │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   Recommended Extensions:                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  5. Image Preview             - Image file preview  │   │
│   │  6. Rainbow CSV               - CSV readability     │   │
│   │  7. GitLens                   - Git history         │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**settings.json 권장 설정**:

```json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,
    "[python]": {
        "editor.formatOnSave": true
    }
}
```

### PyCharm 설정

1. **프로젝트 생성**: File → New Project → Pure Python
2. **인터프리터 설정**: Settings → Project → Python Interpreter
3. **패키지 설치**: + 버튼 → opencv-contrib-python 검색 → Install

### Jupyter Notebook

```bash
# Install Jupyter
pip install jupyter

# Run
jupyter notebook

# Or JupyterLab
pip install jupyterlab
jupyter lab
```

Jupyter에서 이미지 표시:

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
# BGR → RGB conversion (matplotlib uses RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis('off')
plt.show()
```

---

## 4. 버전 확인 및 첫 프로그램

### 설치 확인

```python
import cv2
import numpy as np

# Check OpenCV version — important because APIs change between major versions
print(f"OpenCV version: {cv2.__version__}")
# Output example: OpenCV version: 4.8.0

# NumPy version matters: OpenCV relies on NumPy's array layout (C-contiguous)
print(f"NumPy version: {np.__version__}")
# Output example: NumPy version: 1.24.3

# getBuildInformation() reveals whether GPU (CUDA/OpenCL) and optimized BLAS
# libraries are available — useful for diagnosing performance bottlenecks
print(cv2.getBuildInformation())
```

### 첫 번째 프로그램: 이미지 읽기와 표시

```python
import cv2

# Read image
img = cv2.imread('sample.jpg')

# Check if image was read successfully
if img is None:
    print("Cannot read image!")
else:
    print(f"Image size: {img.shape}")

    # Display image in window
    cv2.imshow('My First OpenCV', img)

    # Wait for key press (0 = wait indefinitely)
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()
```

### 이미지가 없을 때 테스트

```python
import cv2
import numpy as np

# np.zeros creates a black canvas: dtype=uint8 gives the [0, 255] range
# that OpenCV functions expect — using float here would cause display issues
img = np.zeros((300, 400, 3), dtype=np.uint8)

# Add text — useful as a quick smoke test without needing an image file on disk
cv2.putText(img, 'Hello OpenCV!', (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Draw circle: thickness=2 (outline); use -1 for a filled circle
cv2.circle(img, (200, 200), 50, (0, 255, 0), 2)

# Display
cv2.imshow('Test Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 5. OpenCV와 NumPy 관계

### NumPy 기반 구조

OpenCV-Python에서 이미지는 NumPy 배열(ndarray)로 표현됩니다.

```
┌─────────────────────────────────────────────────────────────────┐
│               Relationship between OpenCV and NumPy             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   cv2.imread()                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────────────────────────────────────┐          │
│   │              numpy.ndarray                       │          │
│   │  ┌─────────────────────────────────────────┐    │          │
│   │  │  shape: (height, width, channels)       │    │          │
│   │  │  dtype: uint8 (0-255)                   │    │          │
│   │  │  data: actual pixel values              │    │          │
│   │  └─────────────────────────────────────────┘    │          │
│   └─────────────────────────────────────────────────┘          │
│        │                                                        │
│        ▼                                                        │
│   NumPy operations available:                                  │
│   - Slicing: img[100:200, 50:150]                              │
│   - Operations: img + 50, img * 1.5                            │
│   - Functions: np.mean(img), np.max(img)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### NumPy 연산 활용 예시

```python
import cv2
import numpy as np

img = cv2.imread('sample.jpg')

# Use NumPy functions
print(f"Average brightness: {np.mean(img):.2f}")
print(f"Maximum: {np.max(img)}")
print(f"Minimum: {np.min(img)}")

# np.clip prevents uint8 overflow: without it, 255 + 1 wraps around to 0
# (silent integer overflow), which creates dark spots instead of bright ones
brighter = np.clip(img + 50, 0, 255).astype(np.uint8)
darker = np.clip(img - 50, 0, 255).astype(np.uint8)

# Boolean array: True/False mask for downstream masking or counting pixels
bright_pixels = img > 200

# Statistics
print(f"Standard deviation: {np.std(img):.2f}")
```

### OpenCV 함수 vs NumPy 연산

```python
import cv2
import numpy as np

img = cv2.imread('sample.jpg')

# Method 1: Using OpenCV functions
mean_cv = cv2.mean(img)
print(f"OpenCV mean: {mean_cv}")  # (B_avg, G_avg, R_avg, 0)

# Method 2: Using NumPy
mean_np = np.mean(img, axis=(0, 1))
print(f"NumPy mean: {mean_np}")  # [B_avg, G_avg, R_avg]

# Performance comparison (OpenCV is usually faster)
import time

# Gaussian blur comparison
img_large = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)

start = time.time()
blur_cv = cv2.GaussianBlur(img_large, (5, 5), 0)
print(f"OpenCV: {time.time() - start:.4f}s")
```

---

## 6. 이미지는 ndarray

### 이미지 데이터 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    Image = 3D Array                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   img.shape = (height, width, channels)                         │
│                                                                 │
│   e.g., (480, 640, 3) → 480 rows × 640 cols × 3 channels (BGR)  │
│                                                                 │
│         width (columns, x-axis)                                 │
│       ←───────────────→                                         │
│      ┌─────────────────┐  ↑                                     │
│      │ B G R │ B G R │ │  │                                     │
│      │ pixel │ pixel │ │  │ height                              │
│      ├───────┼───────┤ │  │ (rows, y-axis)                      │
│      │ B G R │ B G R │ │  │                                     │
│      │ pixel │ pixel │ │  │                                     │
│      └─────────────────┘  ↓                                     │
│                                                                 │
│   Access: img[y, x] or img[y, x, channel]                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 데이터 타입 (dtype)

```python
import cv2
import numpy as np

img = cv2.imread('sample.jpg')

# Basic data type
print(f"Data type: {img.dtype}")  # uint8

# Common data types
# uint8:  0 ~ 255 (most common)
# float32: 0.0 ~ 1.0 (deep learning, precision calculations)
# float64: 0.0 ~ 1.0 (scientific computing)

# Type conversion
img_float = img.astype(np.float32) / 255.0
print(f"After conversion: {img_float.dtype}, range: {img_float.min():.2f} ~ {img_float.max():.2f}")

# Back to uint8 (for saving/display)
img_back = (img_float * 255).astype(np.uint8)
```

### 다양한 이미지 형태

```python
import cv2
import numpy as np

# Color image (3 channels)
color_img = cv2.imread('sample.jpg', cv2.IMREAD_COLOR)
print(f"Color: {color_img.shape}")  # (H, W, 3)

# Grayscale (1 channel, 2D)
gray_img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
print(f"Gray: {gray_img.shape}")  # (H, W)

# With alpha channel (4 channels)
alpha_img = cv2.imread('sample.png', cv2.IMREAD_UNCHANGED)
if alpha_img is not None and alpha_img.shape[2] == 4:
    print(f"With alpha: {alpha_img.shape}")  # (H, W, 4)

# Create new images
blank_color = np.zeros((300, 400, 3), dtype=np.uint8)  # Black color
blank_gray = np.zeros((300, 400), dtype=np.uint8)       # Black gray
white_img = np.ones((300, 400, 3), dtype=np.uint8) * 255  # White
```

### 이미지 속성 확인

```python
import cv2

img = cv2.imread('sample.jpg')

if img is not None:
    # Basic properties
    print(f"Shape (H, W, C): {img.shape}")
    print(f"Height: {img.shape[0]}px")
    print(f"Width: {img.shape[1]}px")
    print(f"Channels: {img.shape[2]}")

    # Data properties
    print(f"Data type: {img.dtype}")
    print(f"Total pixels: {img.size}")  # H * W * C
    print(f"Memory size: {img.nbytes} bytes")

    # Dimensions
    print(f"Dimensions: {img.ndim}")  # Color=3, Gray=2
```

---

## 7. 연습 문제

### 연습 1: 환경 확인 스크립트

다음 정보를 출력하는 스크립트를 작성하세요:
- OpenCV 버전
- NumPy 버전
- Python 버전
- 사용 가능한 GPU 가속 여부 (`cv2.cuda.getCudaEnabledDeviceCount()`)

```python
# Hint
import cv2
import numpy as np
import sys

# Write your code here
```

### 연습 2: 이미지 정보 출력기

주어진 이미지 파일의 모든 속성을 출력하는 함수를 작성하세요:

```python
def print_image_info(filepath):
    """
    Prints detailed information about an image file.

    Output items:
    - File path
    - Load success status
    - Image size (width x height)
    - Number of channels
    - Data type
    - Memory usage
    - Pixel value range (min, max)
    - Average brightness
    """
    # Write your code here
    pass
```

### 연습 3: 빈 캔버스 생성

다음 조건의 이미지들을 생성하고 저장하세요:

1. 800x600 검은색 이미지
2. 800x600 흰색 이미지
3. 800x600 빨간색 이미지 (BGR에서 빨간색은?)
4. 400x400 체크무늬 패턴 (50px 단위)

### 연습 4: NumPy 연산 실습

이미지를 로드한 후 다음 작업을 수행하세요:

```python
# 1. Increase brightness by 50 (apply clipping)
# 2. Decrease brightness by 50 (apply clipping)
# 3. Increase contrast by 1.5x
# 4. Invert image (255 - img)
```

### 연습 5: 채널 분리 미리보기

컬러 이미지를 BGR 채널별로 분리하여 각각을 그레이스케일로 표시하는 코드를 작성하세요. NumPy 인덱싱을 사용하세요.

---

## 8. 다음 단계

[이미지 기초 연산](./02_Image_Basics.md)에서 이미지 읽기/쓰기, 픽셀 접근, ROI 설정 등 기본적인 이미지 연산을 학습합니다!

**다음에 배울 내용**:
- `cv2.imread()`, `cv2.imshow()`, `cv2.imwrite()` 상세
- 픽셀 단위 접근과 수정
- 관심 영역(ROI) 설정
- 이미지 복사와 참조

---

## 9. 참고 자료

### 공식 문서

- [OpenCV 공식 문서](https://docs.opencv.org/)
- [OpenCV-Python 튜토리얼](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [NumPy 공식 문서](https://numpy.org/doc/)

### 유용한 링크

- [PyImageSearch](https://pyimagesearch.com/) - 실전 예제 다수
- [Learn OpenCV](https://learnopencv.com/) - 고급 튜토리얼
- [OpenCV GitHub](https://github.com/opencv/opencv)

### 관련 학습 자료

| 폴더 | 관련 내용 |
|------|----------|
| [Python/](../Python/) | NumPy 배열 연산, 타입 힌트 |
| [Linux/](../Linux/) | 개발 환경, 터미널 사용 |

