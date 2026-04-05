# 09. Edge AI - ONNX Runtime

**이전**: [Edge AI (TFLite)](./08_Edge_AI_TFLite.md) | **다음**: [홈 자동화 프로젝트](./10_Home_Automation_Project.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. ONNX 오픈 포맷과 크로스 프레임워크 모델 이식성에서의 역할을 설명할 수 있다
2. 엣지 배포를 위한 ONNX Runtime과 TensorFlow Lite를 비교할 수 있다
3. PyTorch 및 TensorFlow 모델을 ONNX 포맷으로 변환할 수 있다
4. 엣지 성능을 위해 ONNX 모델을 검증하고, 단순화하고, 양자화할 수 있다
5. Raspberry Pi에서 ONNX Runtime으로 추론을 실행할 수 있다
6. YOLO ONNX 모델로 객체 검출을 구현할 수 있다

---

모든 머신러닝 모델이 TensorFlow로 학습되는 것은 아닙니다. 팀에서 PyTorch, scikit-learn, 또는 다른 프레임워크를 사용한다면, ONNX는 엣지 장치에 모델을 배포할 수 있는 범용 교환 포맷을 제공합니다. ONNX Runtime은 TFLite와 함께 고성능 추론 엔진을 제공하여, IoT AI 워크로드에서 특정 프레임워크에 종속되지 않도록 해줍니다.

---

## 1. ONNX 개요

### 1.1 ONNX란?

**ONNX(Open Neural Network Exchange)**는 다양한 ML 프레임워크 간 모델 호환성을 제공하는 오픈 포맷입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    ONNX 생태계                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   훈련 프레임워크                         추론 엔진          │
│   ┌──────────┐                                              │
│   │ PyTorch  │────┐                                         │
│   └──────────┘    │                                         │
│   ┌──────────┐    │        ┌──────────┐    ┌──────────────┐ │
│   │TensorFlow│────┼──────▶│  ONNX    │───▶│ONNX Runtime  │ │
│   └──────────┘    │        │ (.onnx)  │    │(크로스플랫폼)│ │
│   ┌──────────┐    │        └──────────┘    └──────────────┘ │
│   │  Keras   │────┤                               │         │
│   └──────────┘    │                               ▼         │
│   ┌──────────┐    │                        ┌──────────────┐ │
│   │ Sklearn  │────┘                        │ 배포 대상    │ │
│   └──────────┘                             │ • 라즈베리파이│ │
│                                            │ • Windows    │ │
│                                            │ • Android    │ │
│                                            │ • iOS        │ │
│                                            └──────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 ONNX vs TFLite

| 특성 | ONNX | TFLite |
|------|------|--------|
| **개발사** | Microsoft + 파트너 | Google |
| **프레임워크 지원** | PyTorch, TF, Sklearn 등 | TensorFlow/Keras |
| **포맷** | .onnx (Protobuf) | .tflite (FlatBuffer) |
| **최적화** | ONNX Runtime | TF Lite Interpreter |
| **양자화** | 지원 | 지원 |
| **하드웨어** | CPU, GPU, NPU | CPU, GPU, Edge TPU |

### 1.3 ONNX Runtime 특징

```python
# ONNX Runtime 특징
onnx_runtime_features = {
    "크로스플랫폼": "Windows, Linux, macOS, Android, iOS",
    "하드웨어 가속": "CPU, CUDA, TensorRT, DirectML, OpenVINO",
    "다중 언어": "Python, C++, C#, Java, JavaScript",
    "최적화": "그래프 최적화, 양자화, 연산자 퓨전",
    "유연성": "다양한 프레임워크에서 변환된 모델 실행"
}
```

---

## 2. ONNX Runtime 설치

### 2.1 라즈베리파이 설치

```bash
# 기본 ONNX Runtime (CPU)
pip install onnxruntime

# ARM64 최적화 버전 (라즈베리파이 OS 64bit)
# pip install onnxruntime --extra-index-url https://aiinfra.pkgs.visualstudio.com/...

# 추가 패키지
pip install numpy pillow onnx

# 모델 변환용 (PC에서)
pip install tf2onnx torch onnx-simplifier
```

### 2.2 설치 확인

```python
#!/usr/bin/env python3
"""ONNX Runtime 설치 확인"""

import onnxruntime as ort
import numpy as np

# 버전 확인
print(f"ONNX Runtime 버전: {ort.__version__}")

# 사용 가능한 프로바이더 (실행 백엔드)
providers = ort.get_available_providers()
print(f"사용 가능한 프로바이더: {providers}")

# 간단한 테스트
# 더미 모델 실행
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

print("\nONNX Runtime 정상 동작 확인!")
```

---

## 3. 모델 변환

### 3.1 PyTorch to ONNX

```python
#!/usr/bin/env python3
"""PyTorch 모델을 ONNX로 변환"""

import torch
import torch.nn as nn

# 예시 모델
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def export_to_onnx(model, output_path: str, input_shape: tuple):
    """PyTorch 모델을 ONNX로 내보내기"""
    model.eval()

    # 더미 입력
    dummy_input = torch.randn(*input_shape)

    # ONNX 내보내기
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=13
    )

    print(f"ONNX 모델 저장: {output_path}")

# 사용 예
if __name__ == "__main__":
    model = SimpleNet()
    export_to_onnx(model, "simple_net.onnx", (1, 10))
```

### 3.2 TensorFlow/Keras to ONNX

```bash
# tf2onnx 사용 (커맨드라인)
python -m tf2onnx.convert \
    --saved-model tensorflow_model/ \
    --output model.onnx \
    --opset 13
```

```python
#!/usr/bin/env python3
"""TensorFlow/Keras 모델을 ONNX로 변환"""

import tensorflow as tf
import tf2onnx
import onnx

def keras_to_onnx(model_path: str, output_path: str):
    """Keras 모델을 ONNX로 변환"""
    # Keras 모델 로드
    model = tf.keras.models.load_model(model_path)

    # ONNX로 변환
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        opset=13,
        output_path=output_path
    )

    print(f"변환 완료: {output_path}")

# 사용 예
keras_to_onnx("my_model.h5", "my_model.onnx")
```

### 3.3 모델 검증 및 단순화

```python
#!/usr/bin/env python3
"""ONNX 모델 검증 및 단순화"""

import onnx
from onnxsim import simplify

def validate_and_simplify(model_path: str, output_path: str = None):
    """ONNX 모델 검증 및 최적화"""
    # 모델 로드
    model = onnx.load(model_path)

    # 검증
    try:
        onnx.checker.check_model(model)
        print("모델 검증 통과")
    except Exception as e:
        print(f"검증 실패: {e}")
        return

    # 모델 정보 출력
    print(f"\n모델 정보:")
    print(f"  IR 버전: {model.ir_version}")
    print(f"  Opset: {model.opset_import[0].version}")
    print(f"  그래프 이름: {model.graph.name}")

    # 입출력 정보
    print(f"\n입력:")
    for input in model.graph.input:
        print(f"  {input.name}: {input.type}")

    print(f"\n출력:")
    for output in model.graph.output:
        print(f"  {output.name}: {output.type}")

    # 단순화 (중복 연산 제거, 그래프 최적화)
    simplified_model, check = simplify(model)

    if check:
        print("\n단순화 성공")

        if output_path:
            onnx.save(simplified_model, output_path)
            print(f"저장: {output_path}")

            # 크기 비교
            import os
            orig_size = os.path.getsize(model_path) / 1024
            new_size = os.path.getsize(output_path) / 1024
            print(f"\n크기: {orig_size:.1f}KB -> {new_size:.1f}KB")

        return simplified_model
    else:
        print("단순화 실패")
        return model

# 사용 예
if __name__ == "__main__":
    validate_and_simplify("model.onnx", "model_simplified.onnx")
```

---

## 4. 추론 수행

### 4.1 기본 추론

```python
#!/usr/bin/env python3
"""ONNX Runtime 기본 추론"""

import onnxruntime as ort
import numpy as np

class ONNXModel:
    """ONNX 모델 래퍼"""

    def __init__(self, model_path: str, providers: list = None):
        if providers is None:
            providers = ['CPUExecutionProvider']

        # 세션 옵션
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        # 세션 생성
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        # 입출력 정보
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

    def get_input_shape(self):
        return self.input_shape

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """추론 수행"""
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        return outputs[0]

# 사용 예
if __name__ == "__main__":
    model = ONNXModel("model.onnx")

    print(f"입력 형태: {model.get_input_shape()}")

    # 더미 입력
    input_data = np.random.randn(1, 10).astype(np.float32)
    output = model.predict(input_data)

    print(f"출력 형태: {output.shape}")
    print(f"출력 값: {output}")
```

### 4.2 배치 추론

```python
#!/usr/bin/env python3
"""ONNX Runtime 배치 추론"""

import onnxruntime as ort
import numpy as np
import time

def batch_inference(model_path: str, data: np.ndarray,
                    batch_size: int = 32) -> np.ndarray:
    """배치 추론 수행"""
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    results = []
    num_samples = len(data)

    for i in range(0, num_samples, batch_size):
        batch = data[i:i + batch_size]
        output = session.run([output_name], {input_name: batch})[0]
        results.append(output)

    return np.concatenate(results, axis=0)

# 성능 측정
def benchmark_batch_sizes(model_path: str, input_shape: tuple):
    """배치 크기별 성능 비교"""
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name

    total_samples = 1000

    for batch_size in [1, 8, 16, 32, 64]:
        data = np.random.randn(total_samples, *input_shape[1:]).astype(np.float32)

        start = time.perf_counter()

        for i in range(0, total_samples, batch_size):
            batch = data[i:i + batch_size]
            _ = session.run(None, {input_name: batch})

        elapsed = time.perf_counter() - start
        throughput = total_samples / elapsed

        print(f"배치 크기 {batch_size:2d}: {throughput:.1f} samples/sec")
```

### 4.3 양자화 추론

```python
#!/usr/bin/env python3
"""ONNX Runtime 양자화"""

import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(model_path: str, output_path: str):
    """동적 양자화 적용"""
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )

    import os
    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"원본: {orig_size:.2f} MB")
    print(f"양자화: {new_size:.2f} MB")
    print(f"압축률: {orig_size / new_size:.1f}x")

# 사용 예
quantize_model("model.onnx", "model_quantized.onnx")
```

---

## 5. 객체 검출 예제

### 5.1 YOLO ONNX 모델 사용

```python
#!/usr/bin/env python3
"""YOLOv5 ONNX 객체 검출"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

class YOLODetector:
    """YOLOv5 ONNX 객체 검출기"""

    # COCO 클래스
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 입력 정보
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def preprocess(self, image: np.ndarray) -> tuple:
        """이미지 전처리"""
        orig_height, orig_width = image.shape[:2]

        # 리사이즈
        resized = cv2.resize(image, (self.input_width, self.input_height))

        # BGR to RGB, HWC to CHW
        input_data = resized[:, :, ::-1].transpose(2, 0, 1)

        # 정규화 (0-1)
        input_data = input_data.astype(np.float32) / 255.0

        # 배치 차원 추가
        input_data = np.expand_dims(input_data, axis=0)

        # 스케일 비율 저장
        scale = (orig_width / self.input_width, orig_height / self.input_height)

        return input_data, scale

    def postprocess(self, output: np.ndarray, scale: tuple) -> list:
        """출력 후처리"""
        predictions = output[0]

        boxes = []
        scores = []
        class_ids = []

        for pred in predictions:
            confidence = pred[4]

            if confidence > self.conf_threshold:
                class_probs = pred[5:]
                class_id = np.argmax(class_probs)
                class_score = class_probs[class_id]

                if class_score > self.conf_threshold:
                    # 박스 좌표 (center_x, center_y, width, height)
                    cx, cy, w, h = pred[:4]

                    # 원본 스케일로 변환
                    x1 = int((cx - w / 2) * scale[0])
                    y1 = int((cy - h / 2) * scale[1])
                    x2 = int((cx + w / 2) * scale[0])
                    y2 = int((cy + h / 2) * scale[1])

                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(confidence * class_score))
                    class_ids.append(int(class_id))

        # NMS (Non-Maximum Suppression)
        if boxes:
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.conf_threshold, self.iou_threshold
            )

            results = []
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                results.append({
                    'box': boxes[idx],
                    'score': scores[idx],
                    'class_id': class_ids[idx],
                    'class_name': self.CLASSES[class_ids[idx]]
                })

            return results

        return []

    def detect(self, image: np.ndarray) -> list:
        """객체 검출"""
        input_data, scale = self.preprocess(image)

        outputs = self.session.run(None, {self.input_name: input_data})

        detections = self.postprocess(outputs[0], scale)

        return detections

    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """검출 결과 시각화"""
        result = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['class_name']}: {det['score']:.2f}"

            # 박스
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 라벨
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result

# 사용 예
if __name__ == "__main__":
    detector = YOLODetector("yolov5s.onnx")

    # 이미지 로드
    image = cv2.imread("test_image.jpg")

    # 검출
    detections = detector.detect(image)

    print(f"검출된 객체: {len(detections)}개")
    for det in detections:
        print(f"  {det['class_name']}: {det['score']:.2f}")

    # 결과 저장
    result_image = detector.draw_detections(image, detections)
    cv2.imwrite("result.jpg", result_image)
```

### 5.2 실시간 객체 검출

```python
#!/usr/bin/env python3
"""실시간 객체 검출 (Pi Camera + ONNX)"""

import numpy as np
import cv2
import time

try:
    from picamera2 import Picamera2
    HAS_CAMERA = True
except ImportError:
    HAS_CAMERA = False

# YOLODetector 클래스는 위와 동일

class RealtimeDetector:
    """실시간 객체 검출기"""

    def __init__(self, model_path: str):
        self.detector = YOLODetector(model_path)

        if HAS_CAMERA:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.camera.configure(config)

    def run(self, duration: float = 60, display: bool = False):
        """실시간 검출 실행"""
        if not HAS_CAMERA:
            print("카메라 없음")
            return

        self.camera.start()
        print(f"실시간 검출 시작 ({duration}초)")

        start_time = time.time()
        frame_count = 0
        fps_time = time.time()

        try:
            while time.time() - start_time < duration:
                # 프레임 캡처
                frame = self.camera.capture_array()

                # BGR 변환 (OpenCV 형식)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 검출
                detections = self.detector.detect(frame_bgr)

                frame_count += 1

                # FPS 계산
                if frame_count % 10 == 0:
                    elapsed = time.time() - fps_time
                    fps = 10 / elapsed
                    fps_time = time.time()

                    print(f"\rFPS: {fps:.1f}, 검출: {len(detections)}개", end="")

                    for det in detections:
                        print(f" | {det['class_name']}", end="")

                # 디스플레이 (선택)
                if display:
                    result = self.detector.draw_detections(frame_bgr, detections)
                    cv2.imshow("Detection", result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            self.camera.stop()
            if display:
                cv2.destroyAllWindows()

            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"\n\n평균 FPS: {avg_fps:.1f}")

if __name__ == "__main__":
    detector = RealtimeDetector("yolov5s.onnx")
    detector.run(duration=30, display=False)
```

### 5.3 검출 결과 MQTT 발행

```python
#!/usr/bin/env python3
"""객체 검출 결과 MQTT 발행"""

import paho.mqtt.client as mqtt
import json
import time

class DetectionPublisher:
    """검출 결과 MQTT 발행기"""

    def __init__(self, model_path: str, mqtt_broker: str = "localhost"):
        self.detector = YOLODetector(model_path)

        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(mqtt_broker, 1883)
        self.mqtt_client.loop_start()

        self.node_id = "detector_01"

    def process_and_publish(self, image_path: str):
        """이미지 처리 및 결과 발행"""
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지 로드 실패: {image_path}")
            return

        # 검출
        start = time.perf_counter()
        detections = self.detector.detect(image)
        inference_time = (time.perf_counter() - start) * 1000

        # 결과 생성
        result = {
            "node_id": self.node_id,
            "image": image_path,
            "detections": [
                {
                    "class": det['class_name'],
                    "score": round(det['score'], 3),
                    "box": det['box']
                }
                for det in detections
            ],
            "count": len(detections),
            "inference_time_ms": round(inference_time, 2),
            "timestamp": time.time()
        }

        # MQTT 발행
        topic = f"edge/{self.node_id}/detection"
        self.mqtt_client.publish(topic, json.dumps(result))

        print(f"발행: {topic}")
        print(f"  검출: {len(detections)}개, 시간: {inference_time:.1f}ms")

        return result

    def shutdown(self):
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

if __name__ == "__main__":
    publisher = DetectionPublisher("yolov5s.onnx")

    try:
        publisher.process_and_publish("test_image.jpg")
    finally:
        publisher.shutdown()
```

---

## 연습 문제

### 문제 1: 모델 변환
1. PyTorch 이미지 분류 모델을 ONNX로 변환하세요.
2. 변환된 모델을 검증하고 단순화하세요.

### 문제 2: 성능 비교
1. TFLite와 ONNX Runtime의 추론 속도를 비교하세요.
2. 배치 크기별 처리량을 측정하세요.

### 문제 3: 실시간 검출
1. YOLO 모델로 실시간 객체 검출을 구현하세요.
2. 검출 결과를 MQTT로 발행하세요.

---

## 다음 단계

- [홈 자동화 프로젝트](10_Home_Automation_Project.md): AI 기반 스마트홈
- [영상 분석 프로젝트](11_Image_Analysis_Project.md): 영상 분석 프로젝트

---

*최종 업데이트: 2026-02-01*
