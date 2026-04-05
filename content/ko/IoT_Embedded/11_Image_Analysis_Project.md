# 11. 영상 분석 프로젝트

**이전**: [홈 자동화 프로젝트](./10_Home_Automation_Project.md) | **다음**: [클라우드 IoT 통합](./12_Cloud_IoT_Integration.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. Pi Camera 하드웨어를 설정하고 이미지 캡처를 위한 picamera2를 구성할 수 있다
2. 프로그래밍으로 정지 이미지를 캡처하고 동영상을 녹화할 수 있다
3. MJPEG 및 WebSocket 기반 실시간 영상 스트리밍 서버를 구현할 수 있다
4. 카메라 프레임에서 TFLite를 사용하여 실시간 객체 검출을 수행할 수 있다
5. MQTT 알림이 포함된 프레임 차분(frame-differencing) 모션 감지 시스템을 구축할 수 있다
6. 모션이 감지되면 자동으로 동영상 녹화를 시작하도록 구성할 수 있다

---

IoT 시스템에 카메라를 추가하면 감지의 차원이 완전히 달라집니다. 온도와 같은 단일 변수를 측정하는 대신, 카메라는 사람을 감지하고, 물체를 식별하고, 번호판을 읽거나, 방 전체를 모니터링할 수 있습니다. 이 프로젝트는 앞선 레슨의 Edge AI 기술과 Pi Camera를 결합하여, 로컬 하드웨어에서 완전히 실행되는 실용적인 비전 시스템을 구축합니다.

---

## 1. Pi Camera 설정

### 1.1 하드웨어 연결

```
┌─────────────────────────────────────────────────────────────┐
│                    Pi Camera 연결                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. 라즈베리파이 전원 OFF                                   │
│   2. CSI 포트 커넥터 열기 (살짝 당김)                        │
│   3. 리본 케이블 삽입 (파란색 면이 이더넷 포트 방향)         │
│   4. 커넥터 닫기                                             │
│   5. 전원 ON                                                 │
│                                                              │
│   ┌─────────────────────────┐                               │
│   │      Raspberry Pi       │                               │
│   │   ┌─────────────────┐   │                               │
│   │   │    CSI Port     │   │                               │
│   │   │  [▓▓▓▓▓▓▓▓▓▓]  │←─ 카메라 리본 케이블              │
│   │   └─────────────────┘   │                               │
│   │                         │                               │
│   └─────────────────────────┘                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 소프트웨어 설정

```bash
# 카메라 활성화 (raspi-config)
sudo raspi-config
# Interface Options > Legacy Camera > Disable
# (picamera2는 레거시 카메라 비활성화 필요)

# picamera2 설치
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera

# 추가 패키지
pip install numpy pillow opencv-python

# 카메라 테스트
libcamera-hello --list-cameras
```

### 1.3 카메라 확인

```python
#!/usr/bin/env python3
"""Pi Camera 확인"""

from picamera2 import Picamera2
import time

def test_camera():
    """카메라 테스트"""
    picam2 = Picamera2()

    # 카메라 정보
    camera_info = picam2.camera_properties
    print("=== 카메라 정보 ===")
    print(f"모델: {camera_info.get('Model', 'Unknown')}")
    print(f"픽셀 크기: {camera_info.get('PixelArraySize', 'Unknown')}")

    # 설정
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)

    # 이미지 캡처
    picam2.start()
    time.sleep(2)  # 센서 안정화

    frame = picam2.capture_array()
    print(f"\n캡처된 이미지: {frame.shape}")

    # 저장
    from PIL import Image
    img = Image.fromarray(frame)
    img.save("test_capture.jpg")
    print("이미지 저장: test_capture.jpg")

    picam2.stop()

if __name__ == "__main__":
    test_camera()
```

---

## 2. picamera2 라이브러리

### 2.1 기본 사용법

```python
#!/usr/bin/env python3
"""picamera2 기본 사용법"""

from picamera2 import Picamera2
import time

class CameraHandler:
    """Pi Camera 핸들러"""

    def __init__(self, resolution: tuple = (640, 480)):
        self.picam2 = Picamera2()
        self.resolution = resolution

        # 프리뷰 설정
        self.preview_config = self.picam2.create_preview_configuration(
            main={"size": resolution, "format": "RGB888"}
        )

        # 정지화 설정 (고해상도)
        self.still_config = self.picam2.create_still_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}
        )

    def start(self, mode: str = "preview"):
        """카메라 시작"""
        if mode == "preview":
            self.picam2.configure(self.preview_config)
        else:
            self.picam2.configure(self.still_config)

        self.picam2.start()
        time.sleep(0.5)  # 안정화

    def stop(self):
        """카메라 중지"""
        self.picam2.stop()

    def capture_frame(self):
        """프레임 캡처"""
        return self.picam2.capture_array()

    def capture_image(self, filename: str):
        """이미지 파일 저장"""
        self.picam2.capture_file(filename)
        print(f"이미지 저장: {filename}")

    def set_controls(self, **kwargs):
        """카메라 제어 설정"""
        # 예: brightness, contrast, exposure_time, analogue_gain
        self.picam2.set_controls(kwargs)

# 사용 예
if __name__ == "__main__":
    camera = CameraHandler((640, 480))
    camera.start()

    # 프레임 캡처
    frame = camera.capture_frame()
    print(f"프레임 크기: {frame.shape}")

    # 이미지 저장
    camera.capture_image("snapshot.jpg")

    camera.stop()
```

### 2.2 비디오 녹화

```python
#!/usr/bin/env python3
"""비디오 녹화"""

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import time

def record_video(filename: str, duration: int = 10):
    """비디오 녹화"""
    picam2 = Picamera2()

    # 비디오 설정
    video_config = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(video_config)

    # 인코더 및 출력 설정
    encoder = H264Encoder(bitrate=10000000)
    output = FfmpegOutput(filename)

    # 녹화 시작
    picam2.start_recording(encoder, output)
    print(f"녹화 시작: {filename}")

    time.sleep(duration)

    # 녹화 중지
    picam2.stop_recording()
    print(f"녹화 완료: {duration}초")

if __name__ == "__main__":
    record_video("video.mp4", duration=10)
```

---

## 3. 실시간 영상 스트리밍

### 3.1 MJPEG 스트리밍 서버

```python
#!/usr/bin/env python3
"""MJPEG 스트리밍 서버"""

from flask import Flask, Response
from picamera2 import Picamera2
import io
from PIL import Image
import threading
import time

app = Flask(__name__)

class StreamingServer:
    """MJPEG 스트리밍 서버"""

    def __init__(self):
        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(self.config)
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        """스트리밍 시작"""
        self.picam2.start()
        self.running = True

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """캡처 루프"""
        while self.running:
            frame = self.picam2.capture_array()

            # JPEG으로 인코딩
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=80)
            jpeg_data = buffer.getvalue()

            with self.lock:
                self.frame = jpeg_data

            time.sleep(0.033)  # ~30 FPS

    def get_frame(self):
        """현재 프레임 반환"""
        with self.lock:
            return self.frame

    def stop(self):
        """스트리밍 중지"""
        self.running = False
        self.picam2.stop()

# 전역 스트리밍 서버
streamer = StreamingServer()

def generate_frames():
    """프레임 제너레이터"""
    while True:
        frame = streamer.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)

@app.route('/')
def index():
    return '''
    <html>
    <head><title>Pi Camera 스트림</title></head>
    <body>
        <h1>Pi Camera 실시간 스트림</h1>
        <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    streamer.start()
    try:
        app.run(host='0.0.0.0', port=8080, threaded=True)
    finally:
        streamer.stop()
```

### 3.2 WebSocket 스트리밍

```python
#!/usr/bin/env python3
"""WebSocket 영상 스트리밍"""

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from picamera2 import Picamera2
import base64
import io
from PIL import Image
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class WebSocketStreamer:
    """WebSocket 스트리머"""

    def __init__(self):
        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(self.config)
        self.running = False

    def start_streaming(self):
        """스트리밍 시작"""
        self.picam2.start()
        self.running = True

        while self.running:
            frame = self.picam2.capture_array()

            # Base64 인코딩
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=70)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # 클라이언트에 전송
            socketio.emit('frame', {'image': img_str})

            time.sleep(0.05)  # ~20 FPS

    def stop(self):
        self.running = False
        self.picam2.stop()

streamer = WebSocketStreamer()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebSocket Stream</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    </head>
    <body>
        <h1>WebSocket 영상 스트림</h1>
        <canvas id="canvas" width="640" height="480"></canvas>
        <script>
            const socket = io();
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            socket.on('frame', function(data) {
                const img = new Image();
                img.onload = function() {
                    ctx.drawImage(img, 0, 0);
                };
                img.src = 'data:image/jpeg;base64,' + data.image;
            });
        </script>
    </body>
    </html>
    '''

@socketio.on('connect')
def handle_connect():
    print('클라이언트 연결됨')

if __name__ == "__main__":
    stream_thread = threading.Thread(target=streamer.start_streaming, daemon=True)
    stream_thread.start()

    try:
        socketio.run(app, host='0.0.0.0', port=8080)
    finally:
        streamer.stop()
```

---

## 4. 객체 검출 (TFLite)

### 4.1 실시간 객체 검출

```python
#!/usr/bin/env python3
"""실시간 객체 검출 (TFLite + Pi Camera)"""

from picamera2 import Picamera2
import numpy as np
import cv2
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

class RealtimeDetector:
    """실시간 객체 검출기"""

    COCO_LABELS = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        # ... (전체 COCO 라벨)
    ]

    def __init__(self, model_path: str, threshold: float = 0.5):
        # TFLite 모델 로드
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        self.threshold = threshold

        # 카메라 초기화
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """전처리"""
        # 리사이즈
        resized = cv2.resize(frame, (self.input_width, self.input_height))

        # 정규화
        input_data = resized.astype(np.float32) / 255.0

        # 배치 차원 추가
        input_data = np.expand_dims(input_data, axis=0)

        return input_data

    def detect(self, frame: np.ndarray) -> list:
        """객체 검출"""
        input_data = self.preprocess(frame)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # 출력 파싱 (모델에 따라 다름)
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        detections = []
        h, w = frame.shape[:2]

        for i in range(len(scores)):
            if scores[i] > self.threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                detections.append({
                    'class_id': int(classes[i]),
                    'class_name': self.COCO_LABELS[int(classes[i])] if int(classes[i]) < len(self.COCO_LABELS) else 'unknown',
                    'score': float(scores[i]),
                    'box': [int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)]
                })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """검출 결과 시각화"""
        result = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['class_name']}: {det['score']:.2f}"

            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return result

    def run(self, duration: float = 60, display: bool = False):
        """실시간 검출 실행"""
        self.picam2.start()
        print(f"실시간 검출 시작 ({duration}초)")

        start_time = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < duration:
                frame = self.picam2.capture_array()

                # 검출
                detections = self.detect(frame)
                frame_count += 1

                if detections:
                    print(f"[{frame_count}] 검출: {len(detections)}개")
                    for det in detections:
                        print(f"  - {det['class_name']}: {det['score']:.2f}")

                if display:
                    result = self.draw_detections(frame, detections)
                    cv2.imshow("Detection", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            self.picam2.stop()
            if display:
                cv2.destroyAllWindows()

            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"\nFPS: {fps:.1f}")

if __name__ == "__main__":
    detector = RealtimeDetector("ssd_mobilenet.tflite", threshold=0.5)
    detector.run(duration=30, display=False)
```

### 4.2 검출 결과 스트리밍

```python
#!/usr/bin/env python3
"""객체 검출 결과 스트리밍"""

from flask import Flask, Response, jsonify
from picamera2 import Picamera2
import numpy as np
import cv2
import io
from PIL import Image
import threading
import time
import json

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

class DetectionStreamer:
    """검출 결과 스트리머"""

    def __init__(self, model_path: str):
        # 모델 로드
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 카메라
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)

        self.frame = None
        self.detections = []
        self.lock = threading.Lock()

    def start(self):
        self.picam2.start()
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _process_loop(self):
        while self.running:
            frame = self.picam2.capture_array()
            detections = self._detect(frame)

            # 결과 그리기
            result = self._draw(frame, detections)

            # JPEG 인코딩
            img = Image.fromarray(result)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=80)

            with self.lock:
                self.frame = buffer.getvalue()
                self.detections = detections

            time.sleep(0.05)

    def _detect(self, frame: np.ndarray) -> list:
        # 간략화된 검출 로직
        input_size = (self.input_details[0]['shape'][1], self.input_details[0]['shape'][2])
        resized = cv2.resize(frame, input_size)
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # 출력 파싱 (모델에 따라 조정 필요)
        return []  # 실제 구현 필요

    def _draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        result = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.get('box', [0, 0, 0, 0])
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return result

    def get_frame(self):
        with self.lock:
            return self.frame

    def get_detections(self):
        with self.lock:
            return self.detections

    def stop(self):
        self.running = False
        self.picam2.stop()

streamer = DetectionStreamer("model.tflite")

def generate():
    while True:
        frame = streamer.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def detections():
    return jsonify(streamer.get_detections())

if __name__ == "__main__":
    streamer.start()
    try:
        app.run(host='0.0.0.0', port=8080)
    finally:
        streamer.stop()
```

---

## 5. 모션 감지

### 5.1 프레임 차이 기반 모션 감지

```python
#!/usr/bin/env python3
"""모션 감지 시스템"""

from picamera2 import Picamera2
import numpy as np
import cv2
import time
from datetime import datetime
import paho.mqtt.client as mqtt
import json

class MotionDetector:
    """모션 감지기"""

    def __init__(self, threshold: int = 30, min_area: int = 500):
        self.threshold = threshold
        self.min_area = min_area

        # 카메라
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)

        self.prev_frame = None
        self.motion_detected = False

        # MQTT (선택)
        self.mqtt_client = mqtt.Client()
        try:
            self.mqtt_client.connect("localhost", 1883)
            self.mqtt_client.loop_start()
            self.mqtt_enabled = True
        except:
            self.mqtt_enabled = False

    def detect_motion(self, frame: np.ndarray) -> tuple:
        """모션 감지"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, frame, []

        # 프레임 차이 계산
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]

        # 노이즈 제거
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_regions = []
        result = frame.copy()

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            motion_regions.append({'x': x, 'y': y, 'w': w, 'h': h})
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.prev_frame = gray
        motion_detected = len(motion_regions) > 0

        return motion_detected, result, motion_regions

    def on_motion(self, regions: list):
        """모션 감지 이벤트"""
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] 모션 감지! 영역: {len(regions)}개")

        # MQTT 발행
        if self.mqtt_enabled:
            data = {
                "event": "motion_detected",
                "regions": regions,
                "timestamp": timestamp
            }
            self.mqtt_client.publish("camera/motion", json.dumps(data))

        # 스냅샷 저장
        # self.save_snapshot(f"motion_{timestamp}.jpg")

    def run(self, duration: float = 60, cooldown: float = 5):
        """모션 감지 실행"""
        self.picam2.start()
        print(f"모션 감지 시작 ({duration}초)")

        start_time = time.time()
        last_motion_time = 0

        try:
            while time.time() - start_time < duration:
                frame = self.picam2.capture_array()

                detected, result, regions = self.detect_motion(frame)

                if detected:
                    current_time = time.time()
                    if current_time - last_motion_time > cooldown:
                        self.on_motion(regions)
                        last_motion_time = current_time

                time.sleep(0.1)

        except KeyboardInterrupt:
            pass
        finally:
            self.picam2.stop()
            if self.mqtt_enabled:
                self.mqtt_client.loop_stop()
            print("모션 감지 종료")

if __name__ == "__main__":
    detector = MotionDetector(threshold=30, min_area=500)
    detector.run(duration=120, cooldown=5)
```

### 5.2 모션 감지 + 녹화

```python
#!/usr/bin/env python3
"""모션 감지 시 녹화"""

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import numpy as np
import cv2
import time
from datetime import datetime
import threading
import os

class MotionRecorder:
    """모션 감지 녹화기"""

    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.picam2 = Picamera2()
        self.video_config = self.picam2.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
        self.picam2.configure(self.video_config)

        self.encoder = H264Encoder()
        self.is_recording = False
        self.prev_frame = None
        self.threshold = 30
        self.min_area = 500

    def detect_motion(self, frame: np.ndarray) -> bool:
        """모션 감지"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.prev_frame = gray

        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                return True

        return False

    def start_recording(self):
        """녹화 시작"""
        if self.is_recording:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"motion_{timestamp}.h264")

        output = FileOutput(filename)
        self.picam2.start_recording(self.encoder, output)
        self.is_recording = True
        self.recording_start = time.time()
        print(f"녹화 시작: {filename}")

    def stop_recording(self):
        """녹화 중지"""
        if not self.is_recording:
            return

        self.picam2.stop_recording()
        self.is_recording = False
        duration = time.time() - self.recording_start
        print(f"녹화 완료: {duration:.1f}초")

    def run(self, pre_record: float = 2, post_record: float = 5):
        """모션 감지 녹화 실행"""
        self.picam2.start()
        print("모션 감지 녹화 시스템 시작")

        last_motion_time = 0

        try:
            while True:
                frame = self.picam2.capture_array()
                motion = self.detect_motion(frame)

                if motion:
                    last_motion_time = time.time()
                    if not self.is_recording:
                        self.start_recording()

                # 모션 없으면 post_record 후 녹화 중지
                if self.is_recording:
                    if time.time() - last_motion_time > post_record:
                        self.stop_recording()

                time.sleep(0.1)

        except KeyboardInterrupt:
            pass
        finally:
            if self.is_recording:
                self.stop_recording()
            self.picam2.stop()

if __name__ == "__main__":
    recorder = MotionRecorder("recordings")
    recorder.run(post_record=5)
```

---

## 연습 문제

### 문제 1: 타임랩스
1. 일정 간격으로 이미지를 캡처하는 타임랩스 시스템을 구현하세요.
2. 캡처된 이미지를 비디오로 변환하세요.

### 문제 2: 얼굴 검출
1. OpenCV 또는 TFLite로 얼굴 검출을 구현하세요.
2. 얼굴 검출 시 알림을 보내세요.

### 문제 3: 클라우드 연동
1. 모션 감지 시 이미지를 클라우드에 업로드하세요.
2. 검출 결과를 데이터베이스에 저장하세요.

---

## 다음 단계

- [클라우드 IoT 통합](12_Cloud_IoT_Integration.md): 영상 데이터 클라우드 연동

---

*최종 업데이트: 2026-02-01*
