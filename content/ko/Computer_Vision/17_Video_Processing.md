# 비디오 처리 (Video Processing)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. OpenCV에서 VideoCapture와 VideoWriter가 비디오 파일 및 카메라 스트림을 읽고 쓰는 방법을 설명할 수 있습니다.
2. 정확한 FPS 측정과 함께 프레임별(frame-by-frame) 비디오 처리 파이프라인을 구현할 수 있습니다.
3. 배경 차분(Background Subtraction) 알고리즘(MOG2, KNN)을 적용하여 비디오에서 움직이는 물체를 검출할 수 있습니다.
4. 옵티컬 플로우(Optical Flow) 기법(Lucas-Kanade, Farneback)을 구현하여 프레임 간 움직임을 분석할 수 있습니다.
5. 다양한 비디오 분석 과제에 적합한 객체 추적(Object Tracking) 알고리즘을 비교하고 선택할 수 있습니다.

---

## 개요

비디오는 연속된 이미지 프레임의 시퀀스입니다. OpenCV를 사용하여 비디오 파일과 카메라 스트림을 처리하고, 배경 차분과 옵티컬 플로우를 이용한 동작 분석 방법을 학습합니다.

단일 이미지 처리와 달리, 비디오는 시간적 차원(temporal dimension)을 도입합니다. 각 프레임에는 이전 프레임과 다음 프레임이 있으므로, 알고리즘은 정지 이미지에서는 얻을 수 없는 동작 단서를 활용할 수 있습니다. 또한 실시간 제약이 추가됩니다 — 프레임당 50ms가 걸리는 처리 파이프라인은 처리량을 20 FPS로 제한하므로, 비디오 작업에서는 성능 인식(performance awareness)이 필수적입니다.

**난이도**: ⭐⭐⭐

**선수 지식**: 이미지 기초 연산, 필터링, 객체 탐지(Object Detection)

---

## 목차

OpenCV 함수 참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 비디오 추상화, 코덱과 컨테이너의 분리, 픽셀별 밀도 추정으로서의 배경 모델링, 그리고 추적 vs 검출의 트레이드오프를 다룹니다.

1. [VideoCapture: 파일과 카메라](#1-videocapture-파일과-카메라)
2. [VideoWriter: 비디오 저장](#2-videowriter-비디오-저장)
3. [프레임 단위 처리](#3-프레임-단위-처리)
4. [FPS 계산](#4-fps-계산)
5. [배경 차분 (MOG2, KNN)](#5-배경-차분-mog2-knn)
6. [옵티컬 플로우](#6-옵티컬-플로우)
7. [객체 추적](#7-객체-추적)
8. [연습 문제](#8-연습-문제)

---

## 이론과 원리

비디오는 대략 규칙적 시간 간격으로 캡처된 이미지 수열 `I_1, I_2, ..., I_T`입니다. 단일 이미지 처리에 비해 수학적 추가는 작습니다 — 단지 시간 인덱스 하나 — 하지만 질적으로 다른 부류의 알고리즘을 가능하게 합니다: **동작 추정**, **배경 모델링**, **추적**, **시간적 일관성 강제**. 모두 같은 기저 가정 위에 구축됩니다: 연속 프레임은 고도로 상관되어 있다.

이 섹션은 다음을 다룹니다:

- **(A) 비디오 추상화** — 비디오 파일이 실제로 무엇을 담고 있는지, 그리고 코덱/컨테이너 구분.
- **(B) FPS, 타임스탬프, 동기화** — "30 fps"가 종종 근사인 이유, 그리고 임의 속도로 비디오 처리하는 법.
- **(C) 배경 차분** — 전경 검출로서의 픽셀별 밀도 모델링.
- **(D) 객체 추적** — 검출 + 연관 패러다임과 추적이 어려운 이유.
- **(E) 시간적 일관성과 플리커 회피** — 단일 이미지 알고리즘을 프레임별로 적용하면 종종 이상해 보이는 이유.

### A. 비디오 추상화

비디오 파일(`.mp4`, `.avi`, `.mkv`)은 하나 이상의 **트랙**(비디오, 오디오, 자막)을 담은 **컨테이너**로 구조화되며, 각 트랙은 특정 **코덱**으로 인코딩됩니다.

- **컨테이너**(파일 포맷): 바이트가 디스크에 어떻게 배치되고 트랙이 어떻게 교차 배열되는지 정의. MP4, AVI, MKV, WebM이 컨테이너.
- **코덱**(인코더/디코더): 픽셀 데이터가 어떻게 압축되는지 정의. H.264, H.265/HEVC, VP9, AV1이 코덱. OpenCV는 이들을 디코딩하기 위해 FFmpeg나 GStreamer를 사용.

구분이 중요한 이유: `.mp4`로 끝나는 파일은 시스템이 디코드할 수 있는지 아무것도 말해주지 않기 때문 — 코덱이 중요. OpenCV의 코덱 지원은 컴파일된 FFmpeg 빌드에 따라 다르며, 같은 `VideoCapture` 코드가 한 머신에서는 실패하고 다른 머신에서는 작동할 수 있는 이유.

조회 가능한 주요 속성(`cap.get(cv2.CAP_PROP_*)`):

- `CAP_PROP_FRAME_WIDTH`, `CAP_PROP_FRAME_HEIGHT` — 프레임별 이미지 차원.
- `CAP_PROP_FPS` — 공칭 프레임 레이트.
- `CAP_PROP_FRAME_COUNT` — 전체 프레임 수(스트림에서는 0일 수 있고, 가변 프레임 레이트 비디오에서는 부정확할 수 있음).
- `CAP_PROP_POS_FRAMES`, `CAP_PROP_POS_MSEC` — 현재 읽기 위치.
- `CAP_PROP_FOURCC` — 4문자 코덱 식별자.

### B. 프레임 레이트, 타임스탬프, 동기화

"30 FPS"는 거의 항상 근사. 실제 비디오는 **가변 프레임 레이트(VFR)** — 프레임이 개별적으로 타임스탬프되고, 그들 사이 간격이 변합니다. 카메라, 화면 녹화기, 현대 코덱 모두 VFR 비디오를 출력.

두 결과:

- **`CAP_PROP_FPS`로부터의 FPS**는 평균 또는 공칭 레이트; `frame_count / fps`로 "몇 초 지났는지" 계산하지 마세요. 실제 타이밍은 `CAP_PROP_POS_MSEC` 사용.
- **실시간 처리**는 루프의 벽시계가 비디오의 표시 시간과 일치함을 의미. 더 빨리 처리하면 스로틀(`time.sleep`) 필요. 더 느리면 프레임을 버리거나 뒤처짐.

파이프라인의 FPS 측정: 한 프레임 처리 전후의 벽시계 시간을 기록해 `1 / elapsed` 계산. 단일 프레임 딸꾹질에서 스파이크하지 않는 rolling average를 위해 최근 10-30 프레임에 걸쳐 평균.

### C. 배경 차분: 픽셀별 밀도 모델링

**문제**: 고정된 카메라와 가끔 움직이는 객체(사람, 차)가 있는 장면이 주어지면, 각 프레임을 전경(움직이는) 픽셀과 배경(정적) 픽셀로 분리.

**아이디어**: **픽셀별 확률 분포**를 색상에 대해 구축. 각 픽셀 위치 `(x, y)`에서 많은 프레임에 걸쳐 색상을 관찰하고 그 분포를 모델링. 새 관찰이 이 분포 아래에서 가능성이 낮으면 **전경**.

시간이 지남에 따라 배경은 천천히 바뀔 수 있음(움직이는 그림자, 조명 변화), 따라서 모델은 업데이트해야 함. 이것이 `cv2.createBackgroundSubtractorMOG2`가 구현하는 핵심 알고리즘:

#### C.1 MOG2: 가우시안 혼합

각 픽셀에 대해 여러(보통 3-5개) 구성 요소를 가진 **가우시안 혼합 모델** 유지. 각 구성 요소는 평균 색상, 공분산, 그리고 이 픽셀에서 그 색상이 얼마나 자주 관찰됐는지를 나타내는 가중치를 가집니다.

- 새 관찰은 높은 가중치 구성 요소 중 어느 것에도 맞지 않으면(즉, 모든 구성 요소에서 제곱 Mahalanobis 거리가 임계값을 초과하면) 전경.
- 그렇지 않으면, 매치된 구성 요소의 평균, 공분산, 가중치를 지수 이동 평균(학습률 `α`)으로 업데이트.

**왜 혼합**: 픽셀이 합법적으로 여러 "배경" 색상을 취할 수 있음 — 흔들리는 나뭇잎이 녹색과 그 뒤 하늘 사이를 오가고, 단일 가우시안은 둘 다 포착할 수 없음. 혼합은 관찰된 실제 색상 분포에 맞추기 위해 구성 요소를 늘리고 줄임.

MOG2는 **그림자 검출**도 지원: 배경 구성 요소의 더 어두운 버전(낮은 밝기, 같은 chromaticity)인 픽셀은 전경(`255`) 대신 그림자(마스크에서 회색, `127`)로 표시.

#### C.2 KNN 배경 차분기

K-최근접-이웃 기반 대안. 각 픽셀에 대해 최근 관찰의 이력 유지. 최근 `N` 관찰 중 `K`개 미만이 새 픽셀에 가까우면 전경. 개념적으로 MOG2보다 단순, 경험적으로 종종 비슷, 약간 더 많은 메모리.

#### C.3 한계

두 알고리즘 모두 고정 카메라를 가정. 카메라가 움직이면(손에 들고, 팬) 모든 픽셀이 전경처럼 보이고 방법이 실패. 움직이는 카메라의 경우, 먼저 optical flow(§31), 특징 추적, 또는 안정화 사용.

### D. 객체 추적

**추적**은 시간에 걸쳐 객체의 신원을 유지하는 작업: 프레임 `t`의 위치가 주어지면 프레임 `t+1`에서 찾기. 검출(프레임별 독립적으로 객체 찾기)과의 차이는 추적이 **시간적 신원**을 강제한다는 것: 검출이 두 프레임에서 같은 객체를 찾을 수도 있지만 같은 객체라고 말하지는 않음.

#### D.1 Tracking-by-detection 패러다임

현대 실천: **모든 프레임에 검출기를 실행하고 프레임 전반에 걸쳐 검출을 연관**. 연관은 동작 예측(검출 위치가 예측한 곳 근처여야 함)과 외관 특징(검출된 객체의 디스크립터가 추적 객체의 디스크립터와 매치해야 함)을 사용.

SORT(Simple Online and Realtime Tracking)와 DeepSORT가 정식 구현: Kalman 필터 동작 예측 + 검출-트랙 연관을 위한 Hungarian 알고리즘, DeepSORT는 객체가 잠시 사라질 때 도움이 되도록 외관 임베딩 추가.

#### D.2 온라인 추적기(단일 객체)

미리 지정된 한 객체 추적의 단순한 경우: **correlation-filter 추적기**(KCF, CSRT, MOSSE)는 객체 위치에서 피크 응답을 생성하는 필터를 학습하고, 다음 프레임에서 그 피크를 스캔.

- **KCF** (Kernelized Correlation Filter): 빠르고, 짧은 수열에 정확, 스케일 변화에 약함.
- **CSRT** (Channel and Spatial Reliability Tracker): KCF보다 정확, 더 느림, 스케일 변화와 부분 가림을 더 잘 처리.
- **MOSSE** (Minimum Output Sum of Squared Error): 가장 빠름, 가장 덜 정확.
- **MedianFlow**: 실패를 검출하기 위해 앞뒤로 추적.
- **GOTURN / DaSiamRPN / SiamRPN**: 딥러닝 기반 Siamese 추적기, 최고 정확도지만 GPU 필요.

이들 중 어느 것도 객체를 초기에 검출하지 않음 — 첫 프레임에서 경계 상자를 지정하고 추적기가 따라감.

#### D.3 추적이 근본적으로 어려운 이유

추적은 다음을 처리해야 함:

- **외관 변화**: 객체가 회전, 변형, 조명이 바뀜.
- **가림**: 객체가 무언가 뒤에 잠시 가려짐; 다시 나타나면 같은 객체인가?
- **스케일과 포즈 변화**: 객체가 접근/멀어짐에 따라 커지고/작아지고, 3D에서 회전.
- **Drift**: 작은 프레임별 오류가 누적, 결국 객체를 잃음.
- **신원 전환**: 두 객체가 교차; 이후 어느 것이 어느 것?

이것들이 추적이 "단지 프레임별 검출"이 아닌 자체 연구 분야를 가지는 이유. 현대 시스템은 검출, 추적, 그리고 재식별(긴 시간 간격에 걸친 Re-ID 특징 매칭)의 조합 사용.

### E. 시간적 일관성과 플리커

프레임별 알고리즘을 독립적으로 적용하면 종종 **플리커** — 입력의 작은 프레임 간 잡음이 출력의 큰 변화를 만들어 출력이 흔들림. 플리커 원인:

- 분할 모델이 프레임당 약간 다른 마스크 생성.
- 색상 조정 알고리즘의 동작이 프레임당 변하는 전역 통계에 의존.
- 정지 객체에서도 검출기의 경계 상자가 몇 픽셀 떨림.

완화:

- 출력의 **지수 이동 평균**: `result_smooth(t) = α · result_raw(t) + (1-α) · result_smooth(t-1)` (`α = 0.3–0.5`).
- 독립적으로 재검출하기보다 프레임 전반에 걸쳐 경계 상자를 **추적하고 안정화**.
- 모델 아키텍처의 **시간적 필터링**: 3D 컨볼루션(§30) 또는 프레임 간 상태를 공유하는 recurrent 컴포넌트 사용.

지연의 대가로 시간적 필터링을 하는 것을 조심: 완벽한 smoother는 미래 프레임을 봐야 함. 실시간 비디오에서는 과거 방향으로만 smooth 가능.

### 이론에서 아래 함수들로

- `cv2.VideoCapture(source)` — 파일/카메라/RTSP 스트림 리더(§A). `source` = 경로, 정수(카메라 인덱스) 또는 URL.
- `cv2.VideoWriter(path, fourcc, fps, size)` — 인코딩된 비디오 작성자. `fourcc` 코덱 코드가 압축 선택.
- `cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)` — §C.1 MOG2.
- `cv2.createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows)` — §C.2 KNN.
- `cv2.TrackerKCF_create()`, `cv2.TrackerCSRT_create()`, `cv2.legacy.TrackerMOSSE_create()` — §D.2 단일 객체 추적기.
- `cv2.calcOpticalFlowPyrLK` / `cv2.calcOpticalFlowFarneback` — 동작 분석을 위한 31 레슨 optical flow.

---

## 1. VideoCapture: 파일과 카메라

### 비디오 구조 이해

```
Video = Sequence of continuous image frames

Time ------------------------------------------>
    +-----++-----++-----++-----++-----+
    |Frame||Frame||Frame||Frame||Frame| ...
    |  1  ||  2  ||  3  ||  4  ||  5  |
    +-----++-----++-----++-----++-----+

FPS (Frames Per Second): Number of frames per second
- 24 FPS: Movie standard
- 30 FPS: General video
- 60 FPS: Gaming, sports
- 120+ FPS: Slow motion

Resolution: Size of each frame
- 640x480: VGA
- 1280x720: HD (720p)
- 1920x1080: Full HD (1080p)
- 3840x2160: 4K
```

### 비디오 파일 읽기

```python
import cv2

# Open video file
cap = cv2.VideoCapture('video.mp4')

# Check if opened successfully
if not cap.isOpened():
    print("Cannot open video")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total frames: {frame_count}")
print(f"Duration: {duration:.2f} seconds")

# Frame reading loop
while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error")
        break

    # Frame processing
    cv2.imshow('Video', frame)

    # Exit with 'q' key, wait 1ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

### 카메라 입력

```python
import cv2

# Open camera (device ID: 0=default camera)
cap = cv2.VideoCapture(0)

# If camera fails to open
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Set camera properties — explicitly request resolution and FPS because
# cameras often default to a lower mode; requesting forces negotiation
# with the driver (actual values may still differ, always verify after setting)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# BUFFERSIZE=1 keeps only the most recent frame in the driver buffer,
# trading throughput for latency — critical for real-time applications
# where a stale frame is worse than a dropped one
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    # Horizontal flip (mirror effect)
    frame = cv2.flip(frame, 1)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 주요 VideoCapture 속성

```python
import cv2

cap = cv2.VideoCapture('video.mp4')

# Read properties
properties = {
    'CAP_PROP_FRAME_WIDTH': cv2.CAP_PROP_FRAME_WIDTH,    # Frame width
    'CAP_PROP_FRAME_HEIGHT': cv2.CAP_PROP_FRAME_HEIGHT,  # Frame height
    'CAP_PROP_FPS': cv2.CAP_PROP_FPS,                    # FPS
    'CAP_PROP_FRAME_COUNT': cv2.CAP_PROP_FRAME_COUNT,    # Total frame count
    'CAP_PROP_POS_FRAMES': cv2.CAP_PROP_POS_FRAMES,      # Current frame position
    'CAP_PROP_POS_MSEC': cv2.CAP_PROP_POS_MSEC,          # Current position (ms)
    'CAP_PROP_FOURCC': cv2.CAP_PROP_FOURCC,              # Codec 4-char code
    'CAP_PROP_BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,      # Brightness (camera)
    'CAP_PROP_CONTRAST': cv2.CAP_PROP_CONTRAST,          # Contrast (camera)
}

for name, prop in properties.items():
    value = cap.get(prop)
    print(f"{name}: {value}")

# Seek to specific frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # Go to frame 100

# Seek to specific time (milliseconds)
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)  # Go to 5 seconds

cap.release()
```

---

## 2. VideoWriter: 비디오 저장

### 기본 비디오 저장

```python
import cv2

# Video capture setup
cap = cv2.VideoCapture(0)

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30.0

# Codec setup (4-character code)
# 'XVID': for AVI container — widely compatible, moderate compression
# 'mp4v': for MP4 container — good balance of compatibility and file size
# 'MJPG': Motion JPEG — fast but large files (each frame independently compressed)
# 'avc1'/'X264': H.264 — highest compression ratio but requires codec install
# Choose mp4v when portability matters; use XVID when H.264 is unavailable
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create VideoWriter
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

print("Recording started... Press 'q' to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame
    out.write(frame)

    # Recording indicator
    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red circle
    cv2.putText(frame, 'REC', (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Recording', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Recording complete: output.mp4")
```

### 주요 코덱

```
+-----------+-------------+------------------------+
|   Codec   |  Container  |      Characteristics   |
+-----------+-------------+------------------------+
| 'XVID'    | .avi        | Widely supported,      |
|           |             | decent compression     |
| 'MJPG'    | .avi        | Motion JPEG, fast      |
| 'mp4v'    | .mp4        | MPEG-4, good compat    |
| 'avc1'    | .mp4        | H.264, high compression|
| 'X264'    | .mp4        | H.264 (requirements)   |
| 'VP80'    | .webm       | VP8, for web           |
| 'VP90'    | .webm       | VP9, high efficiency   |
+-----------+-------------+------------------------+

# Codec test
def test_codec(codec_str, extension):
    fourcc = cv2.VideoWriter_fourcc(*codec_str)
    out = cv2.VideoWriter(f'test.{extension}', fourcc, 30, (640, 480))
    if out.isOpened():
        print(f"{codec_str}: Supported")
        out.release()
        return True
    else:
        print(f"{codec_str}: Not supported")
        return False
```

### 처리된 비디오 저장

```python
import cv2

def process_and_save_video(input_path, output_path, process_func):
    """Process video and save"""

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed = process_func(frame)

        # Save
        out.write(processed)

        # Progress display
        frame_num += 1
        progress = (frame_num / total_frames) * 100
        print(f"\rProcessing: {progress:.1f}%", end='')

    print("\nComplete!")

    cap.release()
    out.release()

# Usage example: Grayscale conversion and edge detection
def edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Convert to 3 channels (VideoWriter is set for color video)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

process_and_save_video('input.mp4', 'edges.mp4', edge_detection)
```

---

## 3. 프레임 단위 처리

### 프레임 처리 파이프라인

```
Frame Processing Pipeline:

Input --> Preprocessing --> Analysis --> Postprocessing --> Output
              |              |              |
              v              v              v
          - Resize       - Detection    - Visualization
          - Color conv   - Tracking     - Filtering
          - Noise        - Recognition  - Compositing
            removal
```

### 다중 처리 예제

```python
import cv2
import numpy as np

class VideoProcessor:
    """Video frame processor"""

    def __init__(self):
        self.processors = []

    def add_processor(self, name, func):
        """Add processing function"""
        self.processors.append((name, func))

    def process_frame(self, frame):
        """Apply all processing functions"""
        result = frame.copy()
        for name, func in self.processors:
            result = func(result)
        return result

    def process_video(self, input_source, output_path=None, display=True):
        """Process video"""
        cap = cv2.VideoCapture(input_source)

        out = None
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process
            processed = self.process_frame(frame)

            # Save
            if out:
                out.write(processed)

            # Display
            if display:
                cv2.imshow('Processed', processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

# Usage example
processor = VideoProcessor()

# Add processing functions
processor.add_processor('blur', lambda f: cv2.GaussianBlur(f, (5, 5), 0))
processor.add_processor('edge', lambda f: cv2.Canny(f, 50, 150))

def add_timestamp(frame):
    import datetime
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, now, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

processor.add_processor('timestamp', add_timestamp)

# Process webcam
processor.process_video(0, output_path='recorded.mp4')
```

### 프레임 건너뛰기와 버퍼링

```python
import cv2
import time

def skip_frames_processing(video_path, skip=5):
    """Frame skipping (speed improvement)"""

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every skip frames
        if frame_count % skip != 0:
            continue

        # Perform heavy processing
        processed = heavy_processing(frame)

        cv2.imshow('Skipped Processing', processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

def buffered_reading(video_path, buffer_size=10):
    """Frame buffering (smooth playback)"""
    from collections import deque
    from threading import Thread

    cap = cv2.VideoCapture(video_path)
    buffer = deque(maxlen=buffer_size)
    stop_flag = False

    def read_frames():
        while not stop_flag:
            ret, frame = cap.read()
            if not ret:
                break
            if len(buffer) < buffer_size:
                buffer.append(frame)

    # Start reading thread
    thread = Thread(target=read_frames)
    thread.start()

    # Wait for initial buffer fill
    time.sleep(0.5)

    while True:
        if len(buffer) > 0:
            frame = buffer.popleft()
            cv2.imshow('Buffered', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    stop_flag = True
    thread.join()
    cap.release()
```

---

## 4. FPS 계산

### FPS 측정 방법

```python
import cv2
import time

class FPSCounter:
    """FPS measurement class"""

    def __init__(self, avg_frames=30):
        self.frame_times = []
        # avg_frames=30 gives a ~1-second rolling window at 30 FPS —
        # large enough to smooth out single-frame spikes, small enough
        # to respond to genuine performance changes within seconds
        self.avg_frames = avg_frames
        self.last_time = time.time()

    def update(self):
        """Call after processing each frame"""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time

        # Sliding window: discard the oldest sample so the average
        # reflects recent performance rather than startup conditions
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)

    def get_fps(self):
        """Return current FPS"""
        if len(self.frame_times) == 0:
            return 0
        # Averaging inter-frame intervals then inverting is more stable
        # than counting frames in a fixed time window, because it handles
        # irregular processing times without a separate timer thread
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0

# Usage example
cap = cv2.VideoCapture(0)
fps_counter = FPSCounter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame processing
    # ...

    fps_counter.update()
    fps = fps_counter.get_fps()

    # Display FPS
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('FPS', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### 처리 시간 분석

```python
import cv2
import time

class PerformanceMonitor:
    """Performance monitoring"""

    def __init__(self):
        self.timings = {}

    def start(self, name):
        """Start timing"""
        self.timings[name] = {'start': time.time()}

    def stop(self, name):
        """Stop timing"""
        if name in self.timings:
            elapsed = time.time() - self.timings[name]['start']
            self.timings[name]['elapsed'] = elapsed
            return elapsed
        return 0

    def get_report(self):
        """Performance report"""
        report = []
        for name, data in self.timings.items():
            if 'elapsed' in data:
                report.append(f"{name}: {data['elapsed']*1000:.2f}ms")
        return '\n'.join(report)

# Usage example
monitor = PerformanceMonitor()

cap = cv2.VideoCapture(0)

while True:
    # Measure total frame time
    monitor.start('total')

    ret, frame = cap.read()
    if not ret:
        break

    # Measure preprocessing time
    monitor.start('preprocess')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    monitor.stop('preprocess')

    # Measure detection time
    monitor.start('detection')
    edges = cv2.Canny(blur, 50, 150)
    monitor.stop('detection')

    monitor.stop('total')

    # Display performance
    y = 30
    for line in monitor.get_report().split('\n'):
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 20

    cv2.imshow('Performance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

---

## 5. 배경 차분 (MOG2, KNN)

### 배경 차분 원리

```
Background Subtraction:
Separate moving foreground objects from stationary background

+-----------------+     +-----------------+     +-----------------+
| Current frame   |  -  | Background model|  =  | Foreground mask |
|                 |     |                 |     |                 |
|    +---+        |     |                 |     |    +---+        |
|    | * | (person)|    |   (empty room)  |     |    |###|        |
|    +---+        |     |                 |     |    +---+        |
|                 |     |                 |     |                 |
+-----------------+     +-----------------+     +-----------------+

Background model learning:
- Analyze multiple frames to learn background statistics
- Handle lighting changes, shadows, etc.
- Adapt to dynamic backgrounds (tree leaves, etc.)
```

### MOG2 (Mixture of Gaussians)

```python
import cv2
import numpy as np

# Create MOG2 background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(
    history=500,          # Frames used to build background model —
                          # larger = slower adaptation to scene changes
                          # (e.g., 500 frames at 30 FPS ≈ 16 seconds of memory)
    varThreshold=16,      # Mahalanobis distance threshold for classifying a pixel
                          # as foreground; lower = more sensitive but more noise
    detectShadows=True    # Marks shadows as 127 (gray) instead of 255 (white),
                          # letting you remove them separately to avoid false positives
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    # fgMask: foreground=255, background=0, shadow=127
    fgMask = backSub.apply(frame)

    # Remove shadows (127 -> 0)
    fgMask_no_shadow = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)[1]

    # Remove noise with morphological operations:
    # OPEN (erode then dilate) removes small speckles/salt noise
    # CLOSE (dilate then erode) fills holes inside detected objects
    # ELLIPSE kernel is rotationally symmetric — better for blob-shaped objects
    # than RECT, which leaves corner artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgMask_clean = cv2.morphologyEx(fgMask_no_shadow, cv2.MORPH_OPEN, kernel)
    fgMask_clean = cv2.morphologyEx(fgMask_clean, cv2.MORPH_CLOSE, kernel)

    # Extract foreground
    foreground = cv2.bitwise_and(frame, frame, mask=fgMask_clean)

    # Display results
    cv2.imshow('Original', frame)
    cv2.imshow('FG Mask', fgMask)
    cv2.imshow('Cleaned Mask', fgMask_clean)
    cv2.imshow('Foreground', foreground)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### KNN 배경 차분

```python
import cv2

# Create KNN background subtractor
backSub = cv2.createBackgroundSubtractorKNN(
    history=500,          # Background learning frame count
    dist2Threshold=400.0, # Distance threshold
    detectShadows=True    # Shadow detection
)

cap = cv2.VideoCapture('traffic.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Background subtraction
    fgMask = backSub.apply(frame)

    # Remove noise
    fgMask = cv2.medianBlur(fgMask, 5)

    # Contour detection
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Mark moving objects
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area filter
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Motion Detection', frame)
    cv2.imshow('Mask', fgMask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
```

### MOG2 vs KNN 비교

```
+----------------+----------------------+----------------------+
|     Item       |        MOG2          |        KNN           |
+----------------+----------------------+----------------------+
| Algorithm      | Gaussian Mixture Model| K-Nearest Neighbors |
| Speed          | Fast                 | Medium               |
| Memory         | Low                  | High                 |
| Dynamic BG     | Medium               | Good                 |
| Lighting Change| Medium               | Good                 |
| Noise          | Sensitive            | Robust               |
| Recommended    | Static scenes,       | Complex scenes       |
|                | real-time            |                      |
+----------------+----------------------+----------------------+
```

---

## 6. 옵티컬 플로우

### 옵티컬 플로우 개념

```
Optical Flow:
Estimate pixel movement between consecutive frames

Frame t                    Frame t+1
+-----------------+        +-----------------+
|                 |        |                 |
|    *            |   ->   |        *        |
|                 |        |                 |
+-----------------+        +-----------------+

Velocity vector (u, v):
- Pixel (x, y) moves to (x+u, y+v) in next frame
- I(x, y, t) = I(x+u, y+v, t+1) (brightness constancy assumption)

Types:
1. Sparse: Only compute movement for specific points (Lucas-Kanade)
2. Dense: Compute movement for all pixels (Farneback)
```

### Lucas-Kanade 옵티컬 플로우

```python
import cv2
import numpy as np

# Lucas-Kanade parameters
lk_params = dict(
    winSize=(15, 15),      # Search window: larger = handles bigger motion but slower
                           # and prone to aperture problem on textureless regions
    maxLevel=2,            # Image pyramid levels: pyramid lets LK handle fast motion
                           # by first estimating flow on a downsampled image, then
                           # refining on higher resolution (maxLevel=2 → 3 scales)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    # Stop iterating when error < 0.03 OR after 10 iterations — combining both
    # prevents wasting time on converged estimates and limits worst-case cost
)

# Feature detection parameters
feature_params = dict(
    maxCorners=100,        # Cap at 100 to keep tracking computationally feasible
    qualityLevel=0.3,      # Keep only corners scoring ≥ 30% of the strongest one,
                           # filtering weak features that would drift under noise
    minDistance=7,         # Enforce spatial spread so features cover the whole frame,
                           # not just one high-contrast region
    blockSize=7
)

cap = cv2.VideoCapture(0)

# Read first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect features
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# For trajectory visualization
mask = np.zeros_like(old_frame)

# Colors
colors = np.random.randint(0, 255, (100, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None and len(p0) > 0:
        # Compute optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        if p1 is not None:
            # Select good points only
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Visualize movement
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)

                # Trajectory line
                mask = cv2.line(mask, (a, b), (c, d),
                               colors[i % 100].tolist(), 2)
                # Current position point
                frame = cv2.circle(frame, (a, b), 5,
                                   colors[i % 100].tolist(), -1)

            # Update for next frame
            p0 = good_new.reshape(-1, 1, 2)

    # Combine trajectory
    img = cv2.add(frame, mask)

    cv2.imshow('Lucas-Kanade', img)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Re-detect features with 'r' key
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)

    old_gray = frame_gray.copy()

cap.release()
cv2.destroyAllWindows()
```

### Farneback 밀집 옵티컬 플로우

```python
import cv2
import numpy as np

def draw_flow(img, flow, step=16):
    """Visualize flow vectors"""
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
    fx, fy = flow[y, x].T

    # Draw lines
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    vis = img.copy()
    cv2.polylines(vis, lines, 0, (0, 255, 0))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 2, (0, 255, 0), -1)

    return vis

def flow_to_hsv(flow):
    """Convert flow to HSV color"""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Direction -> Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Magnitude -> Value

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prvs, next_gray,
        None,           # Initial flow (None = start from zero displacement)
        pyr_scale=0.5,  # Each pyramid level is half the previous resolution —
                        # 0.5 is the standard choice; lower values handle larger
                        # motions but increase computation
        levels=3,       # 3 pyramid levels cover displacements up to ~8× winsize
        winsize=15,     # Neighborhood for polynomial expansion; larger = smoother
                        # flow but blurs motion boundaries
        iterations=3,   # Refinement passes per pyramid level; 3 is enough for
                        # typical video, more iterations rarely improve quality
        poly_n=5,       # Pixel neighborhood size for polynomial fit (5 or 7)
        poly_sigma=1.2, # Gaussian weighting of the neighborhood; must match poly_n
                        # (use 1.1 for poly_n=5, 1.5 for poly_n=7)
        flags=0
    )

    # Visualization
    flow_vis = draw_flow(frame2, flow)
    hsv_vis = flow_to_hsv(flow)

    cv2.imshow('Flow Vectors', flow_vis)
    cv2.imshow('Flow HSV', hsv_vis)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    prvs = next_gray

cap.release()
cv2.destroyAllWindows()
```

---

## 7. 객체 추적

### OpenCV 내장 트래커

```python
import cv2

# Tracker types
TRACKERS = {
    'BOOSTING': cv2.legacy.TrackerBoosting_create,
    'MIL': cv2.TrackerMIL_create,
    'KCF': cv2.TrackerKCF_create,
    'CSRT': cv2.TrackerCSRT_create,
    'MOSSE': cv2.legacy.TrackerMOSSE_create
}

def track_object(video_path, tracker_type='CSRT'):
    """Single object tracking"""

    # Create tracker
    tracker = TRACKERS[tracker_type]()

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    # Select object to track (mouse drag)
    bbox = cv2.selectROI('Select Object', frame, False)
    cv2.destroyWindow('Select Object')

    # Initialize tracker
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracking
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, tracker_type, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Tracking Failed', (100, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Tracking', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Usage example
track_object('video.mp4', 'CSRT')
```

### 다중 객체 추적

```python
import cv2

class MultiObjectTracker:
    """Multi-object tracker"""

    def __init__(self, tracker_type='CSRT'):
        self.tracker_type = tracker_type
        self.trackers = []
        self.colors = []

    def add_tracker(self, frame, bbox):
        """Add new tracker"""
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        self.trackers.append(tracker)
        self.colors.append((
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        ))

    def update(self, frame):
        """Update all trackers"""
        results = []

        for i, tracker in enumerate(self.trackers):
            success, bbox = tracker.update(frame)
            if success:
                results.append({
                    'id': i,
                    'bbox': bbox,
                    'color': self.colors[i]
                })

        return results

    def draw(self, frame, results):
        """Visualize results"""
        for r in results:
            x, y, w, h = [int(v) for v in r['bbox']]
            cv2.rectangle(frame, (x, y), (x+w, y+h), r['color'], 2)
            cv2.putText(frame, f"ID: {r['id']}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, r['color'], 2)
        return frame

# Usage example
import numpy as np

cap = cv2.VideoCapture(0)
multi_tracker = MultiObjectTracker()

ret, frame = cap.read()

# Select multiple objects (ESC to finish)
while True:
    bbox = cv2.selectROI('Select Objects (Press ESC when done)', frame, False)
    if bbox == (0, 0, 0, 0):  # ESC pressed
        break
    multi_tracker.add_tracker(frame, bbox)

cv2.destroyWindow('Select Objects (Press ESC when done)')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = multi_tracker.update(frame)
    frame = multi_tracker.draw(frame, results)

    cv2.imshow('Multi Tracking', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
```

### 배경 차분 + 추적 결합

```python
import cv2
import numpy as np

class MotionTracker:
    """Background subtraction-based motion tracking"""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.tracks = {}  # {id: {'centroid': (x,y), 'frames': count}}
        self.next_id = 0
        self.max_distance = 50  # Distance for same object judgment

    def process(self, frame):
        """Process frame"""
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        # Remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        # Contour detection
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # Current frame's objects
        current_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                centroid = (x + w//2, y + h//2)
                current_objects.append({
                    'centroid': centroid,
                    'bbox': (x, y, w, h)
                })

        # Match with existing tracks
        self._match_tracks(current_objects)

        return fg_mask, current_objects

    def _match_tracks(self, current_objects):
        """Match current objects with existing tracks"""
        matched = set()

        for obj in current_objects:
            cx, cy = obj['centroid']
            best_match = None
            best_dist = float('inf')

            # Find closest existing track
            for track_id, track in self.tracks.items():
                tx, ty = track['centroid']
                dist = np.sqrt((cx-tx)**2 + (cy-ty)**2)

                if dist < self.max_distance and dist < best_dist:
                    best_dist = dist
                    best_match = track_id

            if best_match is not None:
                # Update existing track
                self.tracks[best_match]['centroid'] = obj['centroid']
                self.tracks[best_match]['bbox'] = obj['bbox']
                self.tracks[best_match]['frames'] += 1
                obj['id'] = best_match
                matched.add(best_match)
            else:
                # Create new track
                obj['id'] = self.next_id
                self.tracks[self.next_id] = {
                    'centroid': obj['centroid'],
                    'bbox': obj['bbox'],
                    'frames': 1
                }
                self.next_id += 1

        # Remove old tracks
        to_remove = [tid for tid in self.tracks if tid not in matched]
        for tid in to_remove:
            if self.tracks[tid]['frames'] < 10:  # Remove short tracks immediately
                del self.tracks[tid]

    def draw(self, frame, objects):
        """Visualize"""
        for obj in objects:
            x, y, w, h = obj['bbox']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if 'id' in obj:
                cv2.putText(frame, f"ID: {obj['id']}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

# Usage example
cap = cv2.VideoCapture(0)
tracker = MotionTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask, objects = tracker.process(frame)
    output = tracker.draw(frame, objects)

    cv2.imshow('Motion Tracking', output)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
```

---

## 8. 연습 문제

### 문제 1: 비디오 플레이어

기본적인 비디오 플레이어를 구현하세요.

**요구사항**:
- 재생/일시정지 토글 (스페이스바)
- 앞으로/뒤로 건너뛰기 (방향키)
- 프레임 단위 이동 (./,)
- 현재 시간/총 시간 표시
- 프로그레스 바

<details>
<summary>힌트</summary>

```python
# Frame navigation
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

# Key handling
key = cv2.waitKey(delay) & 0xFF
if key == ord(' '):  # Spacebar
    paused = not paused
elif key == 83:  # Right arrow
    skip_forward()
```

</details>

### 문제 2: 움직임 히트맵

비디오에서 움직임이 많은 영역을 히트맵으로 시각화하세요.

**요구사항**:
- 배경 차분으로 움직임 검출
- 누적 움직임 맵 생성
- 컬러맵 적용 (COLORMAP_JET)
- 원본과 히트맵 블렌딩

<details>
<summary>힌트</summary>

```python
# Initialize accumulation map
accumulator = np.zeros((height, width), dtype=np.float32)

# Accumulate per frame
accumulator += fg_mask.astype(np.float32) / 255.0

# Normalize and apply colormap
normalized = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX)
heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
```

</details>

### 문제 3: 속도 측정

옵티컬 플로우를 이용해 객체의 이동 속도를 측정하세요.

**요구사항**:
- 특정 ROI 내 평균 플로우 계산
- 픽셀 속도를 실제 속도로 변환 (캘리브레이션 필요)
- 속도 그래프 실시간 표시

<details>
<summary>힌트</summary>

```python
# Average flow in ROI
roi_flow = flow[y:y+h, x:x+w]
avg_flow = np.mean(roi_flow, axis=(0, 1))

# Speed calculation (pixels/frame)
speed = np.sqrt(avg_flow[0]**2 + avg_flow[1]**2)

# Convert to actual speed (e.g., 1 pixel = 1cm, 30fps)
real_speed = speed * pixels_to_cm * fps  # cm/s
```

</details>

### 문제 4: 차량 계수기

도로 비디오에서 통과하는 차량을 계수하세요.

**요구사항**:
- 배경 차분으로 차량 검출
- 가상 선 설정 (계수 라인)
- 선을 통과하는 객체 계수
- 진입/퇴장 방향 구분

<details>
<summary>힌트</summary>

```python
# Define virtual line
line_y = height // 2

# Check if object crossed line
def crossed_line(prev_y, curr_y, line_y):
    # Top to bottom
    if prev_y < line_y and curr_y >= line_y:
        return 'down'
    # Bottom to top
    if prev_y > line_y and curr_y <= line_y:
        return 'up'
    return None
```

</details>

### 문제 5: 동작 인식

옵티컬 플로우 패턴을 분석하여 간단한 동작(손 흔들기, 원 그리기)을 인식하세요.

**요구사항**:
- 손 영역 검출 (피부색 기반)
- 움직임 패턴 추적
- 패턴 분류 (규칙 기반 또는 템플릿 매칭)
- 인식된 동작 표시

<details>
<summary>힌트</summary>

```python
# Skin color detection (HSV)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_skin = np.array([0, 20, 70])
upper_skin = np.array([20, 255, 255])
mask = cv2.inRange(hsv, lower_skin, upper_skin)

# Store movement trajectory
trajectory = []
trajectory.append(centroid)

# Trajectory analysis
# Hand waving: oscillation in x direction
# Circle drawing: start and end points close + certain area
```

</details>

---

## 다음 단계

- [카메라 캘리브레이션 (Camera Calibration)](./18_Camera_Calibration.md) - 카메라 행렬, 왜곡 보정

---

## 참고 자료

- [OpenCV Video I/O](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)
- [Background Subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
- [Optical Flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [Object Tracking](https://docs.opencv.org/4.x/d9/df8/group__tracking.html)
- Horn, B. K., & Schunck, B. G. (1981). "Determining Optical Flow"
- Lucas, B. D., & Kanade, T. (1981). "An Iterative Image Registration Technique"
