# 환경 설정 및 기초

## 개요

OpenCV(Open Source Computer Vision Library)는 실시간 컴퓨터 비전을 위한 오픈소스 라이브러리입니다. 이 문서에서는 OpenCV 설치부터 첫 프로그램 실행, 그리고 이미지 데이터의 기본 구조를 학습합니다.

**난이도**: ⭐ (입문)

**학습 목표**:
- OpenCV 설치 및 개발 환경 구성
- 버전 확인 및 첫 번째 프로그램 작성
- OpenCV와 NumPy의 관계 이해
- 이미지가 ndarray로 표현되는 개념 이해

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
│                        OpenCV 활용 분야                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │  영상 처리   │   │  객체 검출   │   │  얼굴 인식   │          │
│   │  필터링     │   │  YOLO/SSD   │   │  인증 시스템  │          │
│   │  변환       │   │  추적       │   │  감정 분석   │          │
│   └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                 │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │  의료 영상   │   │  자율 주행   │   │  AR/VR      │          │
│   │  CT/MRI     │   │  차선 검출   │   │  마커 인식   │          │
│   │  진단 보조   │   │  장애물 인식  │   │  3D 재구성   │          │
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
│                      OpenCV Python 패키지                       │
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
│   → 대부분의 기능 사용 가능     → SIFT 등 추가 알고리즘 필요시    │
│   → 빠른 설치                 → 특허/연구 알고리즘 포함         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### pip를 이용한 설치

```bash
# 기본 설치 (대부분의 경우 충분)
pip install opencv-python

# 추가 기능 포함 설치 (SIFT, SURF 등)
pip install opencv-contrib-python

# NumPy와 matplotlib도 함께 설치 (권장)
pip install opencv-python numpy matplotlib

# 버전 지정 설치
pip install opencv-python==4.8.0.76

# 업그레이드
pip install --upgrade opencv-python
```

**주의사항**: `opencv-python`과 `opencv-contrib-python`을 동시에 설치하지 마세요. 충돌이 발생할 수 있습니다.

```bash
# 잘못된 예 (충돌 발생)
pip install opencv-python opencv-contrib-python  # ✗

# 올바른 예 (둘 중 하나만)
pip install opencv-contrib-python  # ✓ (contrib에 기본 기능 포함)
```

### 가상환경 사용 (권장)

```bash
# 가상환경 생성
python -m venv opencv_env

# 활성화 (Windows)
opencv_env\Scripts\activate

# 활성화 (macOS/Linux)
source opencv_env/bin/activate

# 패키지 설치
pip install opencv-contrib-python numpy matplotlib

# 비활성화
deactivate
```

---

## 3. 개발 환경 설정

### VSCode 설정

```
┌─────────────────────────────────────────────────────────────┐
│                     VSCode 권장 설정                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   필수 확장 프로그램:                                        │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  1. Python (Microsoft)        - Python 지원         │   │
│   │  2. Pylance                   - 코드 분석, 자동완성  │   │
│   │  3. Jupyter                   - 노트북 지원         │   │
│   │  4. Python Image Preview      - 이미지 미리보기     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   권장 확장 프로그램:                                        │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  5. Image Preview             - 이미지 파일 미리보기 │   │
│   │  6. Rainbow CSV               - CSV 파일 가독성     │   │
│   │  7. GitLens                   - Git 히스토리 확인   │   │
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
# Jupyter 설치
pip install jupyter

# 실행
jupyter notebook

# 또는 JupyterLab
pip install jupyterlab
jupyter lab
```

Jupyter에서 이미지 표시:

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
# BGR → RGB 변환 (matplotlib은 RGB 사용)
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

# OpenCV 버전 확인
print(f"OpenCV 버전: {cv2.__version__}")
# 출력 예: OpenCV 버전: 4.8.0

# NumPy 버전 확인
print(f"NumPy 버전: {np.__version__}")
# 출력 예: NumPy 버전: 1.24.3

# 빌드 정보 확인 (상세)
print(cv2.getBuildInformation())
```

### 첫 번째 프로그램: 이미지 읽기와 표시

```python
import cv2

# 이미지 읽기
img = cv2.imread('sample.jpg')

# 이미지가 제대로 읽혔는지 확인
if img is None:
    print("이미지를 읽을 수 없습니다!")
else:
    print(f"이미지 크기: {img.shape}")

    # 창에 이미지 표시
    cv2.imshow('My First OpenCV', img)

    # 키 입력 대기 (0 = 무한 대기)
    cv2.waitKey(0)

    # 모든 창 닫기
    cv2.destroyAllWindows()
```

### 이미지가 없을 때 테스트

```python
import cv2
import numpy as np

# 검은색 이미지 생성 (300x400, 3채널)
img = np.zeros((300, 400, 3), dtype=np.uint8)

# 텍스트 추가
cv2.putText(img, 'Hello OpenCV!', (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 원 그리기 (중심, 반지름, 색상, 두께)
cv2.circle(img, (200, 200), 50, (0, 255, 0), 2)

# 표시
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
│                  OpenCV와 NumPy의 관계                          │
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
│   │  │  data: 실제 픽셀 값                      │    │          │
│   │  └─────────────────────────────────────────┘    │          │
│   └─────────────────────────────────────────────────┘          │
│        │                                                        │
│        ▼                                                        │
│   NumPy 연산 사용 가능:                                          │
│   - 슬라이싱: img[100:200, 50:150]                              │
│   - 연산: img + 50, img * 1.5                                   │
│   - 함수: np.mean(img), np.max(img)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### NumPy 연산 활용 예시

```python
import cv2
import numpy as np

img = cv2.imread('sample.jpg')

# NumPy 함수 사용
print(f"평균 밝기: {np.mean(img):.2f}")
print(f"최대값: {np.max(img)}")
print(f"최소값: {np.min(img)}")

# 배열 연산으로 밝기 조절
brighter = np.clip(img + 50, 0, 255).astype(np.uint8)
darker = np.clip(img - 50, 0, 255).astype(np.uint8)

# 비교 연산
bright_pixels = img > 200  # Boolean 배열

# 통계
print(f"표준편차: {np.std(img):.2f}")
```

### OpenCV 함수 vs NumPy 연산

```python
import cv2
import numpy as np

img = cv2.imread('sample.jpg')

# 방법 1: OpenCV 함수 사용
mean_cv = cv2.mean(img)
print(f"OpenCV mean: {mean_cv}")  # (B평균, G평균, R평균, 0)

# 방법 2: NumPy 사용
mean_np = np.mean(img, axis=(0, 1))
print(f"NumPy mean: {mean_np}")  # [B평균, G평균, R평균]

# 성능 비교 (대부분의 경우 OpenCV가 빠름)
import time

# 가우시안 블러 비교
img_large = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)

start = time.time()
blur_cv = cv2.GaussianBlur(img_large, (5, 5), 0)
print(f"OpenCV: {time.time() - start:.4f}초")
```

---

## 6. 이미지는 ndarray

### 이미지 데이터 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    이미지 = 3차원 배열                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   img.shape = (height, width, channels)                         │
│                                                                 │
│   예: (480, 640, 3) → 480행 × 640열 × 3채널(BGR)                │
│                                                                 │
│         width (열, x축)                                         │
│       ←───────────────→                                         │
│      ┌─────────────────┐  ↑                                     │
│      │ B G R │ B G R │ │  │                                     │
│      │ 픽셀  │ 픽셀  │ │  │ height                              │
│      ├───────┼───────┤ │  │ (행, y축)                           │
│      │ B G R │ B G R │ │  │                                     │
│      │ 픽셀  │ 픽셀  │ │  │                                     │
│      └─────────────────┘  ↓                                     │
│                                                                 │
│   접근: img[y, x] 또는 img[y, x, channel]                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 데이터 타입 (dtype)

```python
import cv2
import numpy as np

img = cv2.imread('sample.jpg')

# 기본 데이터 타입
print(f"데이터 타입: {img.dtype}")  # uint8

# 주요 데이터 타입
# uint8:  0 ~ 255 (가장 일반적)
# float32: 0.0 ~ 1.0 (딥러닝, 정밀 계산)
# float64: 0.0 ~ 1.0 (과학적 계산)

# 타입 변환
img_float = img.astype(np.float32) / 255.0
print(f"변환 후: {img_float.dtype}, 범위: {img_float.min():.2f} ~ {img_float.max():.2f}")

# 다시 uint8로 (저장/표시용)
img_back = (img_float * 255).astype(np.uint8)
```

### 다양한 이미지 형태

```python
import cv2
import numpy as np

# 컬러 이미지 (3채널)
color_img = cv2.imread('sample.jpg', cv2.IMREAD_COLOR)
print(f"컬러: {color_img.shape}")  # (H, W, 3)

# 그레이스케일 (1채널, 2차원)
gray_img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
print(f"그레이: {gray_img.shape}")  # (H, W)

# 알파 채널 포함 (4채널)
alpha_img = cv2.imread('sample.png', cv2.IMREAD_UNCHANGED)
if alpha_img is not None and alpha_img.shape[2] == 4:
    print(f"알파 포함: {alpha_img.shape}")  # (H, W, 4)

# 새 이미지 생성
blank_color = np.zeros((300, 400, 3), dtype=np.uint8)  # 검은 컬러
blank_gray = np.zeros((300, 400), dtype=np.uint8)       # 검은 그레이
white_img = np.ones((300, 400, 3), dtype=np.uint8) * 255  # 흰색
```

### 이미지 속성 확인

```python
import cv2

img = cv2.imread('sample.jpg')

if img is not None:
    # 기본 속성
    print(f"형태 (H, W, C): {img.shape}")
    print(f"높이: {img.shape[0]}px")
    print(f"너비: {img.shape[1]}px")
    print(f"채널 수: {img.shape[2]}")

    # 데이터 속성
    print(f"데이터 타입: {img.dtype}")
    print(f"총 픽셀 수: {img.size}")  # H * W * C
    print(f"메모리 크기: {img.nbytes} bytes")

    # 차원
    print(f"차원: {img.ndim}")  # 컬러=3, 그레이=2
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
# 힌트
import cv2
import numpy as np
import sys

# 여기에 코드 작성
```

### 연습 2: 이미지 정보 출력기

주어진 이미지 파일의 모든 속성을 출력하는 함수를 작성하세요:

```python
def print_image_info(filepath):
    """
    이미지 파일의 상세 정보를 출력합니다.

    출력 항목:
    - 파일 경로
    - 로드 성공 여부
    - 이미지 크기 (너비 x 높이)
    - 채널 수
    - 데이터 타입
    - 메모리 사용량
    - 픽셀 값 범위 (최소, 최대)
    - 평균 밝기
    """
    # 여기에 코드 작성
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
# 1. 밝기 50 증가 (클리핑 적용)
# 2. 밝기 50 감소 (클리핑 적용)
# 3. 대비 1.5배 증가
# 4. 이미지 반전 (255 - img)
```

### 연습 5: 채널 분리 미리보기

컬러 이미지를 BGR 채널별로 분리하여 각각을 그레이스케일로 표시하는 코드를 작성하세요. NumPy 인덱싱을 사용하세요.

---

## 8. 다음 단계

[02_Image_Basics.md](./02_Image_Basics.md)에서 이미지 읽기/쓰기, 픽셀 접근, ROI 설정 등 기본적인 이미지 연산을 학습합니다!

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

