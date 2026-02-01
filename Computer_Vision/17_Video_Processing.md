# 비디오 처리 (Video Processing)

## 개요

비디오는 연속된 이미지 프레임의 시퀀스입니다. OpenCV를 사용하여 비디오 파일과 카메라 스트림을 처리하고, 배경 차분과 옵티컬 플로우를 이용한 동작 분석 방법을 학습합니다.

**난이도**: ⭐⭐⭐

**선수 지식**: 이미지 기초 연산, 필터링, 객체 검출

---

## 목차

1. [VideoCapture: 파일과 카메라](#1-videocapture-파일과-카메라)
2. [VideoWriter: 비디오 저장](#2-videowriter-비디오-저장)
3. [프레임 단위 처리](#3-프레임-단위-처리)
4. [FPS 계산](#4-fps-계산)
5. [배경 차분 (MOG2, KNN)](#5-배경-차분-mog2-knn)
6. [옵티컬 플로우](#6-옵티컬-플로우)
7. [객체 추적](#7-객체-추적)
8. [연습 문제](#8-연습-문제)

---

## 1. VideoCapture: 파일과 카메라

### 비디오 구조 이해

```
비디오 = 연속된 이미지 프레임

시간 ──────────────────────────────────────────▶
    ┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐
    │Frame││Frame││Frame││Frame││Frame│ ...
    │  1  ││  2  ││  3  ││  4  ││  5  │
    └─────┘└─────┘└─────┘└─────┘└─────┘

FPS (Frames Per Second): 초당 프레임 수
- 24 FPS: 영화 표준
- 30 FPS: 일반 비디오
- 60 FPS: 게임, 스포츠
- 120+ FPS: 슬로모션

해상도: 각 프레임의 크기
- 640x480: VGA
- 1280x720: HD (720p)
- 1920x1080: Full HD (1080p)
- 3840x2160: 4K
```

### 비디오 파일 읽기

```python
import cv2

# 비디오 파일 열기
cap = cv2.VideoCapture('video.mp4')

# 열기 성공 확인
if not cap.isOpened():
    print("비디오를 열 수 없습니다")
    exit()

# 비디오 속성 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"해상도: {width}x{height}")
print(f"FPS: {fps}")
print(f"총 프레임: {frame_count}")
print(f"재생 시간: {duration:.2f}초")

# 프레임 읽기 루프
while True:
    ret, frame = cap.read()

    if not ret:
        print("비디오 끝 또는 에러")
        break

    # 프레임 처리
    cv2.imshow('Video', frame)

    # 'q' 키로 종료, 1ms 대기
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
```

### 카메라 입력

```python
import cv2

# 카메라 열기 (장치 ID: 0=기본 카메라)
cap = cv2.VideoCapture(0)

# 카메라 열기 실패 시
if not cap.isOpened():
    print("카메라를 열 수 없습니다")
    exit()

# 카메라 속성 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# 버퍼 크기 설정 (지연 감소)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(f"카메라 해상도: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    # 좌우 반전 (거울 효과)
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

# 읽기 속성
properties = {
    'CAP_PROP_FRAME_WIDTH': cv2.CAP_PROP_FRAME_WIDTH,    # 프레임 너비
    'CAP_PROP_FRAME_HEIGHT': cv2.CAP_PROP_FRAME_HEIGHT,  # 프레임 높이
    'CAP_PROP_FPS': cv2.CAP_PROP_FPS,                    # FPS
    'CAP_PROP_FRAME_COUNT': cv2.CAP_PROP_FRAME_COUNT,    # 총 프레임 수
    'CAP_PROP_POS_FRAMES': cv2.CAP_PROP_POS_FRAMES,      # 현재 프레임 위치
    'CAP_PROP_POS_MSEC': cv2.CAP_PROP_POS_MSEC,          # 현재 위치 (밀리초)
    'CAP_PROP_FOURCC': cv2.CAP_PROP_FOURCC,              # 코덱 4문자 코드
    'CAP_PROP_BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,      # 밝기 (카메라)
    'CAP_PROP_CONTRAST': cv2.CAP_PROP_CONTRAST,          # 대비 (카메라)
}

for name, prop in properties.items():
    value = cap.get(prop)
    print(f"{name}: {value}")

# 특정 프레임으로 이동
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # 100번째 프레임으로

# 특정 시간으로 이동 (밀리초)
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)  # 5초 위치로

cap.release()
```

---

## 2. VideoWriter: 비디오 저장

### 기본 비디오 저장

```python
import cv2

# 비디오 캡처 설정
cap = cv2.VideoCapture(0)

# 비디오 속성
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30.0

# 코덱 설정 (4문자 코드)
# 'XVID': AVI 컨테이너용
# 'mp4v': MP4 컨테이너용
# 'MJPG': Motion JPEG
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# VideoWriter 생성
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

print("녹화 시작... 'q'를 눌러 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 저장
    out.write(frame)

    # 녹화 표시
    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # 빨간 원
    cv2.putText(frame, 'REC', (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Recording', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
print("녹화 완료: output.mp4")
```

### 주요 코덱

```
┌─────────────┬─────────────┬────────────────────────┐
│   코덱      │  컨테이너   │         특징           │
├─────────────┼─────────────┼────────────────────────┤
│ 'XVID'      │ .avi        │ 널리 지원, 적당한 압축 │
│ 'MJPG'      │ .avi        │ Motion JPEG, 빠름      │
│ 'mp4v'      │ .mp4        │ MPEG-4, 호환성 좋음    │
│ 'avc1'      │ .mp4        │ H.264, 고압축률        │
│ 'X264'      │ .mp4        │ H.264 (요구사항 있음)  │
│ 'VP80'      │ .webm       │ VP8, 웹용              │
│ 'VP90'      │ .webm       │ VP9, 고효율            │
└─────────────┴─────────────┴────────────────────────┘

# 코덱 테스트
def test_codec(codec_str, extension):
    fourcc = cv2.VideoWriter_fourcc(*codec_str)
    out = cv2.VideoWriter(f'test.{extension}', fourcc, 30, (640, 480))
    if out.isOpened():
        print(f"{codec_str}: 지원됨")
        out.release()
        return True
    else:
        print(f"{codec_str}: 지원 안 됨")
        return False
```

### 처리된 비디오 저장

```python
import cv2

def process_and_save_video(input_path, output_path, process_func):
    """비디오 처리 후 저장"""

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

        # 프레임 처리
        processed = process_func(frame)

        # 저장
        out.write(processed)

        # 진행률 표시
        frame_num += 1
        progress = (frame_num / total_frames) * 100
        print(f"\r처리 중: {progress:.1f}%", end='')

    print("\n완료!")

    cap.release()
    out.release()

# 사용 예: 그레이스케일 변환 및 엣지 검출
def edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # 3채널로 변환 (VideoWriter는 컬러 비디오로 설정됨)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

process_and_save_video('input.mp4', 'edges.mp4', edge_detection)
```

---

## 3. 프레임 단위 처리

### 프레임 처리 파이프라인

```
프레임 처리 파이프라인:

입력 ──▶ 전처리 ──▶ 분석 ──▶ 후처리 ──▶ 출력
         │          │         │
         ▼          ▼         ▼
      - 리사이즈  - 검출    - 시각화
      - 색 변환  - 추적    - 필터링
      - 노이즈   - 인식    - 합성
        제거
```

### 다중 처리 예제

```python
import cv2
import numpy as np

class VideoProcessor:
    """비디오 프레임 처리기"""

    def __init__(self):
        self.processors = []

    def add_processor(self, name, func):
        """처리 함수 추가"""
        self.processors.append((name, func))

    def process_frame(self, frame):
        """모든 처리 함수 적용"""
        result = frame.copy()
        for name, func in self.processors:
            result = func(result)
        return result

    def process_video(self, input_source, output_path=None, display=True):
        """비디오 처리"""
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

            # 처리
            processed = self.process_frame(frame)

            # 저장
            if out:
                out.write(processed)

            # 표시
            if display:
                cv2.imshow('Processed', processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

# 사용 예
processor = VideoProcessor()

# 처리 함수들 추가
processor.add_processor('blur', lambda f: cv2.GaussianBlur(f, (5, 5), 0))
processor.add_processor('edge', lambda f: cv2.Canny(f, 50, 150))

def add_timestamp(frame):
    import datetime
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, now, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

processor.add_processor('timestamp', add_timestamp)

# 웹캠 처리
processor.process_video(0, output_path='recorded.mp4')
```

### 프레임 건너뛰기와 버퍼링

```python
import cv2
import time

def skip_frames_processing(video_path, skip=5):
    """프레임 건너뛰기 (속도 향상)"""

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # skip 프레임마다 처리
        if frame_count % skip != 0:
            continue

        # 무거운 처리 수행
        processed = heavy_processing(frame)

        cv2.imshow('Skipped Processing', processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

def buffered_reading(video_path, buffer_size=10):
    """프레임 버퍼링 (스무스한 재생)"""
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

    # 읽기 스레드 시작
    thread = Thread(target=read_frames)
    thread.start()

    # 초기 버퍼 채우기 대기
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
    """FPS 측정 클래스"""

    def __init__(self, avg_frames=30):
        self.frame_times = []
        self.avg_frames = avg_frames
        self.last_time = time.time()

    def update(self):
        """프레임 처리 후 호출"""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time

        # 최근 N개 프레임만 유지
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)

    def get_fps(self):
        """현재 FPS 반환"""
        if len(self.frame_times) == 0:
            return 0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0

# 사용 예
cap = cv2.VideoCapture(0)
fps_counter = FPSCounter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 처리
    # ...

    fps_counter.update()
    fps = fps_counter.get_fps()

    # FPS 표시
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
    """성능 모니터링"""

    def __init__(self):
        self.timings = {}

    def start(self, name):
        """타이밍 시작"""
        self.timings[name] = {'start': time.time()}

    def stop(self, name):
        """타이밍 종료"""
        if name in self.timings:
            elapsed = time.time() - self.timings[name]['start']
            self.timings[name]['elapsed'] = elapsed
            return elapsed
        return 0

    def get_report(self):
        """성능 리포트"""
        report = []
        for name, data in self.timings.items():
            if 'elapsed' in data:
                report.append(f"{name}: {data['elapsed']*1000:.2f}ms")
        return '\n'.join(report)

# 사용 예
monitor = PerformanceMonitor()

cap = cv2.VideoCapture(0)

while True:
    # 전체 프레임 시간 측정
    monitor.start('total')

    ret, frame = cap.read()
    if not ret:
        break

    # 전처리 시간 측정
    monitor.start('preprocess')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    monitor.stop('preprocess')

    # 검출 시간 측정
    monitor.start('detection')
    edges = cv2.Canny(blur, 50, 150)
    monitor.stop('detection')

    monitor.stop('total')

    # 성능 표시
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
배경 차분 (Background Subtraction):
움직이는 전경 객체를 정지된 배경으로부터 분리

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 현재 프레임     │  -  │   배경 모델     │  =  │   전경 마스크   │
│                 │     │                 │     │                 │
│    ┌───┐        │     │                 │     │    ┌───┐        │
│    │ ● │ (사람) │     │   (빈 방)       │     │    │███│        │
│    └───┘        │     │                 │     │    └───┘        │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘

배경 모델 학습:
- 여러 프레임을 분석하여 배경 통계 학습
- 조명 변화, 그림자 등 처리
- 동적 배경 (나뭇잎 흔들림 등) 대응
```

### MOG2 (Mixture of Gaussians)

```python
import cv2
import numpy as np

# MOG2 배경 차분기 생성
backSub = cv2.createBackgroundSubtractorMOG2(
    history=500,          # 배경 학습에 사용할 프레임 수
    varThreshold=16,      # 픽셀이 배경으로 판단되는 분산 임계값
    detectShadows=True    # 그림자 검출 여부
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 배경 차분 적용
    # fgMask: 전경=255, 배경=0, 그림자=127
    fgMask = backSub.apply(frame)

    # 그림자 제거 (127 -> 0)
    fgMask_no_shadow = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)[1]

    # 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgMask_clean = cv2.morphologyEx(fgMask_no_shadow, cv2.MORPH_OPEN, kernel)
    fgMask_clean = cv2.morphologyEx(fgMask_clean, cv2.MORPH_CLOSE, kernel)

    # 전경 추출
    foreground = cv2.bitwise_and(frame, frame, mask=fgMask_clean)

    # 결과 표시
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

# KNN 배경 차분기 생성
backSub = cv2.createBackgroundSubtractorKNN(
    history=500,          # 배경 학습 프레임 수
    dist2Threshold=400.0, # 거리 임계값
    detectShadows=True    # 그림자 검출
)

cap = cv2.VideoCapture('traffic.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 배경 차분
    fgMask = backSub.apply(frame)

    # 노이즈 제거
    fgMask = cv2.medianBlur(fgMask, 5)

    # 윤곽선 검출
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # 움직이는 객체 표시
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # 최소 면적 필터
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
┌────────────────┬──────────────────────┬──────────────────────┐
│     항목       │        MOG2          │        KNN           │
├────────────────┼──────────────────────┼──────────────────────┤
│ 알고리즘       │ 가우시안 혼합 모델   │ K-최근접 이웃        │
│ 속도           │ 빠름                 │ 중간                 │
│ 메모리         │ 적음                 │ 많음                 │
│ 동적 배경      │ 보통                 │ 좋음                 │
│ 조명 변화      │ 보통                 │ 좋음                 │
│ 노이즈         │ 민감                 │ 강건                 │
│ 추천 상황      │ 정적 장면, 실시간    │ 복잡한 장면          │
└────────────────┴──────────────────────┴──────────────────────┘
```

---

## 6. 옵티컬 플로우

### 옵티컬 플로우 개념

```
옵티컬 플로우 (Optical Flow):
연속된 프레임 사이의 픽셀 움직임을 추정

프레임 t                    프레임 t+1
┌─────────────────┐        ┌─────────────────┐
│                 │        │                 │
│    ●            │   →    │        ●        │
│                 │        │                 │
└─────────────────┘        └─────────────────┘

속도 벡터 (u, v):
- 픽셀 (x, y)가 다음 프레임에서 (x+u, y+v)로 이동
- I(x, y, t) = I(x+u, y+v, t+1) (밝기 항상성 가정)

종류:
1. Sparse (희소): 특정 점들의 움직임만 계산 (Lucas-Kanade)
2. Dense (밀집): 모든 픽셀의 움직임 계산 (Farneback)
```

### Lucas-Kanade 옵티컬 플로우

```python
import cv2
import numpy as np

# Lucas-Kanade 파라미터
lk_params = dict(
    winSize=(15, 15),      # 검색 윈도우 크기
    maxLevel=2,            # 피라미드 레벨
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# 특징점 검출 파라미터
feature_params = dict(
    maxCorners=100,        # 최대 특징점 수
    qualityLevel=0.3,      # 품질 수준
    minDistance=7,         # 최소 거리
    blockSize=7
)

cap = cv2.VideoCapture(0)

# 첫 프레임 읽기
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# 특징점 검출
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 추적 궤적 시각화용
mask = np.zeros_like(old_frame)

# 색상
colors = np.random.randint(0, 255, (100, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None and len(p0) > 0:
        # 옵티컬 플로우 계산
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        if p1 is not None:
            # 좋은 점만 선택
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # 움직임 시각화
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)

                # 궤적 선
                mask = cv2.line(mask, (a, b), (c, d),
                               colors[i % 100].tolist(), 2)
                # 현재 위치 점
                frame = cv2.circle(frame, (a, b), 5,
                                   colors[i % 100].tolist(), -1)

            # 다음 프레임을 위한 업데이트
            p0 = good_new.reshape(-1, 1, 2)

    # 궤적 합성
    img = cv2.add(frame, mask)

    cv2.imshow('Lucas-Kanade', img)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # 'r' 키로 특징점 재검출
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
    """플로우 벡터 시각화"""
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
    fx, fy = flow[y, x].T

    # 선 그리기
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    vis = img.copy()
    cv2.polylines(vis, lines, 0, (0, 255, 0))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 2, (0, 255, 0), -1)

    return vis

def flow_to_hsv(flow):
    """플로우를 HSV 색상으로 변환"""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # 방향 -> Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 크기 -> Value

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Farneback 옵티컬 플로우
    flow = cv2.calcOpticalFlowFarneback(
        prvs, next_gray,
        None,           # 초기 플로우
        pyr_scale=0.5,  # 피라미드 스케일
        levels=3,       # 피라미드 레벨
        winsize=15,     # 윈도우 크기
        iterations=3,   # 반복 횟수
        poly_n=5,       # 다항식 크기
        poly_sigma=1.2, # 가우시안 시그마
        flags=0
    )

    # 시각화
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

# 트래커 종류
TRACKERS = {
    'BOOSTING': cv2.legacy.TrackerBoosting_create,
    'MIL': cv2.TrackerMIL_create,
    'KCF': cv2.TrackerKCF_create,
    'CSRT': cv2.TrackerCSRT_create,
    'MOSSE': cv2.legacy.TrackerMOSSE_create
}

def track_object(video_path, tracker_type='CSRT'):
    """단일 객체 추적"""

    # 트래커 생성
    tracker = TRACKERS[tracker_type]()

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    # 추적할 객체 선택 (마우스 드래그)
    bbox = cv2.selectROI('Select Object', frame, False)
    cv2.destroyWindow('Select Object')

    # 트래커 초기화
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 추적 업데이트
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

# 사용 예
track_object('video.mp4', 'CSRT')
```

### 다중 객체 추적

```python
import cv2

class MultiObjectTracker:
    """다중 객체 추적기"""

    def __init__(self, tracker_type='CSRT'):
        self.tracker_type = tracker_type
        self.trackers = []
        self.colors = []

    def add_tracker(self, frame, bbox):
        """새 트래커 추가"""
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        self.trackers.append(tracker)
        self.colors.append((
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        ))

    def update(self, frame):
        """모든 트래커 업데이트"""
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
        """결과 시각화"""
        for r in results:
            x, y, w, h = [int(v) for v in r['bbox']]
            cv2.rectangle(frame, (x, y), (x+w, y+h), r['color'], 2)
            cv2.putText(frame, f"ID: {r['id']}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, r['color'], 2)
        return frame

# 사용 예
import numpy as np

cap = cv2.VideoCapture(0)
multi_tracker = MultiObjectTracker()

ret, frame = cap.read()

# 여러 객체 선택 (ESC로 종료)
while True:
    bbox = cv2.selectROI('Select Objects (Press ESC when done)', frame, False)
    if bbox == (0, 0, 0, 0):  # ESC 누름
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
    """배경 차분 기반 움직임 추적"""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.tracks = {}  # {id: {'centroid': (x,y), 'frames': count}}
        self.next_id = 0
        self.max_distance = 50  # 동일 객체 판단 거리

    def process(self, frame):
        """프레임 처리"""
        # 배경 차분
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        # 노이즈 제거
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        # 윤곽선 검출
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # 현재 프레임의 객체들
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

        # 기존 트랙과 매칭
        self._match_tracks(current_objects)

        return fg_mask, current_objects

    def _match_tracks(self, current_objects):
        """현재 객체와 기존 트랙 매칭"""
        matched = set()

        for obj in current_objects:
            cx, cy = obj['centroid']
            best_match = None
            best_dist = float('inf')

            # 가장 가까운 기존 트랙 찾기
            for track_id, track in self.tracks.items():
                tx, ty = track['centroid']
                dist = np.sqrt((cx-tx)**2 + (cy-ty)**2)

                if dist < self.max_distance and dist < best_dist:
                    best_dist = dist
                    best_match = track_id

            if best_match is not None:
                # 기존 트랙 업데이트
                self.tracks[best_match]['centroid'] = obj['centroid']
                self.tracks[best_match]['bbox'] = obj['bbox']
                self.tracks[best_match]['frames'] += 1
                obj['id'] = best_match
                matched.add(best_match)
            else:
                # 새 트랙 생성
                obj['id'] = self.next_id
                self.tracks[self.next_id] = {
                    'centroid': obj['centroid'],
                    'bbox': obj['bbox'],
                    'frames': 1
                }
                self.next_id += 1

        # 오래된 트랙 제거
        to_remove = [tid for tid in self.tracks if tid not in matched]
        for tid in to_remove:
            if self.tracks[tid]['frames'] < 10:  # 짧은 트랙은 바로 제거
                del self.tracks[tid]

    def draw(self, frame, objects):
        """시각화"""
        for obj in objects:
            x, y, w, h = obj['bbox']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if 'id' in obj:
                cv2.putText(frame, f"ID: {obj['id']}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

# 사용 예
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
# 프레임 이동
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

# 키 처리
key = cv2.waitKey(delay) & 0xFF
if key == ord(' '):  # 스페이스바
    paused = not paused
elif key == 83:  # 오른쪽 화살표
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
# 누적 맵 초기화
accumulator = np.zeros((height, width), dtype=np.float32)

# 프레임마다 누적
accumulator += fg_mask.astype(np.float32) / 255.0

# 정규화 및 컬러맵 적용
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
# ROI 내 평균 플로우
roi_flow = flow[y:y+h, x:x+w]
avg_flow = np.mean(roi_flow, axis=(0, 1))

# 속도 계산 (픽셀/프레임)
speed = np.sqrt(avg_flow[0]**2 + avg_flow[1]**2)

# 실제 속도 변환 (예: 1픽셀 = 1cm, 30fps)
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
# 가상 선 정의
line_y = height // 2

# 객체가 선을 통과했는지 확인
def crossed_line(prev_y, curr_y, line_y):
    # 위에서 아래로
    if prev_y < line_y and curr_y >= line_y:
        return 'down'
    # 아래에서 위로
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
# 피부색 검출 (HSV)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_skin = np.array([0, 20, 70])
upper_skin = np.array([20, 255, 255])
mask = cv2.inRange(hsv, lower_skin, upper_skin)

# 움직임 궤적 저장
trajectory = []
trajectory.append(centroid)

# 궤적 분석
# 손 흔들기: x 방향 진동
# 원 그리기: 시작점과 끝점이 가까움 + 일정 면적
```

</details>

---

## 다음 단계

- [18_Camera_Calibration.md](./18_Camera_Calibration.md) - 카메라 행렬, 왜곡 보정

---

## 참고 자료

- [OpenCV Video I/O](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)
- [Background Subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
- [Optical Flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [Object Tracking](https://docs.opencv.org/4.x/d9/df8/group__tracking.html)
- Horn, B. K., & Schunck, B. G. (1981). "Determining Optical Flow"
- Lucas, B. D., & Kanade, T. (1981). "An Iterative Image Registration Technique"
