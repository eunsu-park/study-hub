# 08. Edge AI - TensorFlow Lite

**이전**: [HTTP REST for IoT](./07_HTTP_REST_for_IoT.md) | **다음**: [Edge AI (ONNX)](./09_Edge_AI_ONNX.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. Edge AI 개념을 설명하고 클라우드 기반 AI 추론과 비교할 수 있다
2. 모델 학습부터 엣지 배포까지 TensorFlow Lite 워크플로우를 설명할 수 있다
3. 다양한 양자화(quantization) 옵션으로 Keras 모델을 TFLite 포맷으로 변환할 수 있다
4. TFLite 인터프리터를 사용하여 Raspberry Pi에서 추론을 실행할 수 있다
5. 엣지 하드웨어에서 로컬로 실행되는 이미지 분류기를 구축할 수 있다
6. Edge AI 결과를 MQTT와 통합하여 IoT 데이터 파이프라인을 구성할 수 있다

---

클라우드가 아닌 엣지 장치에서 AI 모델을 처리하면 네트워크 지연을 없애고, 개인정보를 보호하며, 인터넷이 끊겨도 시스템이 동작하게 합니다. 공장 바닥에서 실시간 결함 감지나 보안 카메라의 인물 감지와 같은 IoT 애플리케이션에서는, 클라우드 왕복에 수백 밀리초를 기다리는 것은 선택지가 될 수 없습니다. TensorFlow Lite는 Raspberry Pi만큼 작은 장치에서도 신경망을 실행할 수 있게 해줍니다.

> **비유: 지점 사무소** -- 클라우드 AI는 모든 결정을 본사에 보내는 것과 같습니다 — 정확하지만 느리고 비용이 많이 듭니다. Edge AI는 일상적인 결정(이것이 사람인가? 이 온도가 비정상인가?)을 현장에서 밀리초 단위로 처리하고, 비정상적인 경우에만 본사에 보고하는 지역 지점 사무소입니다. TensorFlow Lite는 지점 사무소에 딱 맞는 소형 툴킷입니다.

## 1. Edge AI 개념

### 1.1 Edge AI란?

**Edge AI**는 클라우드가 아닌 엣지 디바이스(라즈베리파이, 스마트폰 등)에서 직접 AI 추론을 수행하는 것입니다.

```
┌─────────────────────────────────────────────────────────────┐
│              클라우드 AI vs Edge AI                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   클라우드 AI                        Edge AI                 │
│   ┌─────────┐                       ┌─────────┐             │
│   │  센서   │                       │  센서   │             │
│   └────┬────┘                       └────┬────┘             │
│        │ 데이터                          │ 데이터            │
│        ▼                                 ▼                  │
│   ┌─────────┐                       ┌─────────┐             │
│   │ 네트워크│                       │   Edge  │             │
│   └────┬────┘                       │   AI    │             │
│        │                            └────┬────┘             │
│        ▼                                 │ 결과              │
│   ┌─────────┐                            ▼                  │
│   │ 클라우드│                       ┌─────────┐             │
│   │   AI    │                       │  Action │             │
│   └────┬────┘                       └─────────┘             │
│        │                                                    │
│        ▼                            장점:                    │
│   ┌─────────┐                       • 저지연 (< 50ms)       │
│   │  Action │                       • 오프라인 동작         │
│   └─────────┘                       • 프라이버시            │
│                                     • 비용 절감             │
│   단점:                                                     │
│   • 지연 (100ms+)                                           │
│   • 네트워크 의존                                           │
│   • 데이터 전송 비용                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Edge AI 활용 사례

| 분야 | 활용 예시 |
|------|-----------|
| **스마트홈** | 얼굴 인식 도어락, 음성 인식 |
| **산업** | 불량품 검출, 예측 정비 |
| **헬스케어** | 웨어러블 건강 모니터링 |
| **농업** | 작물 질병 감지, 해충 식별 |
| **자동차** | ADAS, 보행자 감지 |

### 1.3 Edge AI 프레임워크 비교

| 프레임워크 | 개발사 | 특징 | 하드웨어 지원 |
|-----------|--------|------|--------------|
| **TensorFlow Lite** | Google | 범용, 생태계 풍부 | CPU, GPU, Edge TPU |
| **ONNX Runtime** | Microsoft | 다양한 프레임워크 호환 | CPU, GPU |
| **OpenVINO** | Intel | Intel 최적화 | Intel CPU/GPU |
| **TensorRT** | NVIDIA | NVIDIA GPU 최적화 | NVIDIA GPU |

---

## 2. TensorFlow Lite 개요

### 2.1 TFLite 특징

```python
# TensorFlow Lite 특징
tflite_features = {
    "경량화": "모델 크기 감소 (양자화로 1/4)",
    "최적화": "모바일/임베디드 추론 최적화",
    "하드웨어 가속": "GPU, Edge TPU, DSP 지원",
    "크로스 플랫폼": "Android, iOS, Linux, MCU",
    "연산자": "TF 연산자 서브셋 지원"
}
```

### 2.2 라즈베리파이에서 설치

```bash
# 방법 1: tflite-runtime (권장, 경량)
pip install tflite-runtime

# 방법 2: 전체 TensorFlow (무거움)
# pip install tensorflow

# 추가 패키지
pip install numpy pillow
```

### 2.3 TFLite 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│                TensorFlow Lite 워크플로우                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. 모델 훈련 (PC/클라우드)                                │
│      ┌────────────────────┐                                 │
│      │  TensorFlow/Keras  │                                 │
│      │     모델 (.h5)     │                                 │
│      └─────────┬──────────┘                                 │
│                │                                             │
│   2. 모델 변환                                               │
│      ┌─────────▼──────────┐                                 │
│      │   TFLite Converter │                                 │
│      │   (양자화 옵션)    │                                 │
│      └─────────┬──────────┘                                 │
│                │                                             │
│   3. 최적화된 모델                                          │
│      ┌─────────▼──────────┐                                 │
│      │   model.tflite     │                                 │
│      │   (경량화된 모델)   │                                 │
│      └─────────┬──────────┘                                 │
│                │                                             │
│   4. 엣지 배포                                               │
│      ┌─────────▼──────────┐                                 │
│      │  TFLite Runtime    │                                 │
│      │  (라즈베리파이)    │                                 │
│      └────────────────────┘                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 모델 변환 (.tflite)

### 3.1 기본 변환

```python
#!/usr/bin/env python3
"""TensorFlow 모델을 TFLite로 변환"""

import tensorflow as tf

# 기존 Keras 모델 로드
model = tf.keras.models.load_model('my_model.h5')

# 변환기 생성
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 변환
tflite_model = converter.convert()

# 저장
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"모델 크기: {len(tflite_model) / 1024:.2f} KB")
```

### 3.2 양자화 (Quantization)

```python
#!/usr/bin/env python3
"""양자화를 통한 모델 최적화"""

import tensorflow as tf
import numpy as np

def load_model():
    return tf.keras.models.load_model('my_model.h5')

def convert_to_tflite(model, quantization='none'):
    """
    양자화 옵션:
    - 'none': 기본 (float32)
    - 'dynamic': 동적 범위 양자화 (가중치만)
    - 'float16': Float16 양자화
    - 'int8': 전체 정수 양자화 (대표 데이터셋 필요)
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == 'dynamic':
        # 동적 범위 양자화 (가장 쉬움)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif quantization == 'float16':
        # Float16 양자화
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quantization == 'int8':
        # 전체 정수 양자화 (최대 최적화)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # 대표 데이터셋 제공 필요
        def representative_dataset():
            for _ in range(100):
                yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

        converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()
    return tflite_model

# 변환 및 크기 비교
model = load_model()

for quant in ['none', 'dynamic', 'float16']:
    tflite_model = convert_to_tflite(model, quant)
    size_kb = len(tflite_model) / 1024

    with open(f'model_{quant}.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f"{quant}: {size_kb:.2f} KB")
```

### 3.3 사전 훈련 모델 변환

```python
#!/usr/bin/env python3
"""MobileNet을 TFLite로 변환"""

import tensorflow as tf

# MobileNetV2 로드
model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=True
)

# 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('mobilenet_v2.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"변환 완료: {len(tflite_model) / (1024*1024):.2f} MB")
```

---

## 4. 라즈베리파이에서 추론

### 4.1 TFLite 인터프리터 기본

```python
#!/usr/bin/env python3
"""TFLite 추론 기본"""

import numpy as np

# tflite-runtime 사용 (라즈베리파이)
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

class TFLiteModel:
    """TFLite 모델 래퍼"""

    def __init__(self, model_path: str):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # 입출력 정보
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 입력 형태
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']

    def get_input_shape(self):
        """입력 형태 반환"""
        return self.input_shape

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """추론 수행"""
        # 입력 설정
        # astype(self.input_dtype) 중요 이유: INT8 양자화 모델은 int8 입력을
        # 기대함. float32 데이터를 INT8 모델에 전달하면 TFLite가 오류를
        # 발생시키지 않음 -- 바이트를 조용히 재해석하여 유효해 보이지만
        # 잘못된 예측 결과(garbage output)를 생성. 항상 모델이 기대하는
        # dtype으로 변환해야 함. 예상치 못한 분류 결과가 나올 때
        # input_details[0]['dtype']을 확인할 것.
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            input_data.astype(self.input_dtype)
        )

        # 추론
        self.interpreter.invoke()

        # 출력 가져오기
        output = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )

        return output

# 사용 예
if __name__ == "__main__":
    model = TFLiteModel("model.tflite")
    print(f"입력 형태: {model.get_input_shape()}")

    # 더미 입력
    input_data = np.random.rand(*model.get_input_shape()).astype(np.float32)

    output = model.predict(input_data)
    print(f"출력 형태: {output.shape}")
```

### 4.2 성능 측정

```python
#!/usr/bin/env python3
"""TFLite 추론 성능 측정"""

import numpy as np
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

def benchmark_model(model_path: str, num_runs: int = 100):
    """모델 성능 벤치마크"""
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # 워밍업
    dummy_input = np.random.rand(*input_shape).astype(input_dtype)
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

    # 벤치마크
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()

        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / avg_time

    print(f"=== {model_path} ===")
    print(f"평균 추론 시간: {avg_time:.2f} ms (+/- {std_time:.2f})")
    print(f"FPS: {fps:.1f}")
    print(f"입력 형태: {input_shape}")

    return avg_time

# 여러 모델 비교
if __name__ == "__main__":
    models = [
        "model_none.tflite",
        "model_dynamic.tflite",
        "model_float16.tflite"
    ]

    for model in models:
        try:
            benchmark_model(model)
            print()
        except Exception as e:
            print(f"{model}: 오류 - {e}")
```

---

## 5. 이미지 분류 예제

### 5.1 ImageNet 분류

```python
#!/usr/bin/env python3
"""TFLite 이미지 분류 (MobileNet)"""

import numpy as np
from PIL import Image
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

class ImageClassifier:
    """TFLite 이미지 분류기"""

    def __init__(self, model_path: str, labels_path: str = None):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 입력 크기 확인
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

        # 라벨 로드
        self.labels = []
        if labels_path:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """이미지 전처리"""
        # 리사이즈
        image = image.resize((self.input_width, self.input_height))

        # NumPy 배열로 변환
        input_data = np.array(image, dtype=np.float32)

        # [-1, 1] 정규화 이유: MobileNet이 정확히 이 공식으로 [-1, 1] 범위로
        # 스케일링된 픽셀값으로 훈련되었음. 다른 정규화 방식(예: /255.0은
        # [0, 1] 범위)을 사용하면 오류 없이 잘못된 결과를 생성함 -- 모델이
        # 여전히 클래스 확률을 출력하지만, 입력 분포가 모델이 학습한 것과
        # 일치하지 않아 의미 없는 값이 됨. 항상 모델의 훈련 파이프라인과
        # 동일한 전처리를 사용해야 함.
        input_data = (input_data - 127.5) / 127.5

        # 배치 차원 추가
        input_data = np.expand_dims(input_data, axis=0)

        return input_data

    def classify(self, image_path: str, top_k: int = 5) -> list:
        """이미지 분류"""
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        input_data = self.preprocess(image)

        # 추론
        start = time.perf_counter()

        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            input_data
        )
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0]

        inference_time = (time.perf_counter() - start) * 1000

        # Top-K 결과
        top_indices = output.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            score = float(output[idx])
            results.append({
                "class_id": int(idx),
                "label": label,
                "score": score
            })

        return {
            "results": results,
            "inference_time_ms": inference_time
        }

# 사용 예
if __name__ == "__main__":
    classifier = ImageClassifier(
        model_path="mobilenet_v2.tflite",
        labels_path="imagenet_labels.txt"
    )

    result = classifier.classify("test_image.jpg")

    print(f"추론 시간: {result['inference_time_ms']:.2f} ms")
    print("\n분류 결과:")
    for r in result['results']:
        print(f"  {r['label']}: {r['score']:.4f}")
```

### 5.2 실시간 카메라 분류

```python
#!/usr/bin/env python3
"""Pi Camera를 이용한 실시간 이미지 분류"""

import numpy as np
from PIL import Image
import time
import io

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

try:
    from picamera2 import Picamera2
    HAS_CAMERA = True
except ImportError:
    HAS_CAMERA = False
    print("picamera2 없음: 시뮬레이션 모드")

class RealtimeClassifier:
    """실시간 이미지 분류기"""

    def __init__(self, model_path: str, labels_path: str):
        # 모델 로드
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

        # 라벨
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # 카메라 초기화
        if HAS_CAMERA:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.camera.configure(config)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리"""
        image = Image.fromarray(frame)
        image = image.resize((self.input_width, self.input_height))

        input_data = np.array(image, dtype=np.float32)
        # [-1, 1] 범위 이유: MobileNet 훈련 시 사용된 정규화와 일치해야 함 (5.1 참조)
        input_data = (input_data - 127.5) / 127.5
        input_data = np.expand_dims(input_data, axis=0)

        return input_data

    def classify_frame(self, frame: np.ndarray) -> dict:
        """단일 프레임 분류"""
        input_data = self.preprocess(frame)

        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            input_data
        )
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0]

        top_idx = output.argmax()

        return {
            "label": self.labels[top_idx] if top_idx < len(self.labels) else "unknown",
            "score": float(output[top_idx])
        }

    def run(self, duration: float = 30):
        """실시간 분류 실행"""
        if not HAS_CAMERA:
            print("카메라 없음")
            return

        self.camera.start()
        print(f"실시간 분류 시작 ({duration}초)")

        start_time = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < duration:
                frame = self.camera.capture_array()
                result = self.classify_frame(frame)

                frame_count += 1
                print(f"\r[{frame_count}] {result['label']}: {result['score']:.2f}", end="")

        except KeyboardInterrupt:
            pass
        finally:
            self.camera.stop()
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"\n\nFPS: {fps:.1f}")

if __name__ == "__main__":
    classifier = RealtimeClassifier(
        model_path="mobilenet_v2.tflite",
        labels_path="imagenet_labels.txt"
    )
    classifier.run(duration=60)
```

### 5.3 IoT 통합 예제

```python
#!/usr/bin/env python3
"""TFLite + MQTT: 이미지 분류 결과 발행"""

import numpy as np
from PIL import Image
import json
import time
import paho.mqtt.client as mqtt

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

class AIEdgeNode:
    """AI 엣지 노드"""

    def __init__(self, model_path: str, labels_path: str,
                 mqtt_broker: str = "localhost"):
        # TFLite 모델
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # MQTT 클라이언트
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(mqtt_broker, 1883)
        self.mqtt_client.loop_start()

        self.node_id = "edge_ai_01"

    def classify_and_publish(self, image_path: str):
        """분류하고 결과를 MQTT로 발행"""
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.input_shape[2], self.input_shape[1]))

        input_data = np.array(image, dtype=np.float32)
        # [-1, 1] 범위 이유: MobileNet 훈련 시 사용된 정규화와 일치해야 함 (5.1 참조)
        input_data = (input_data - 127.5) / 127.5
        input_data = np.expand_dims(input_data, axis=0)

        # 추론
        start = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        inference_time = (time.perf_counter() - start) * 1000

        # 결과 생성
        top_idx = output.argsort()[-3:][::-1]
        predictions = [
            {
                "label": self.labels[idx] if idx < len(self.labels) else "unknown",
                "score": float(output[idx])
            }
            for idx in top_idx
        ]

        result = {
            "node_id": self.node_id,
            "image": image_path,
            "predictions": predictions,
            "inference_time_ms": round(inference_time, 2),
            "timestamp": time.time()
        }

        # MQTT 발행
        topic = f"edge/{self.node_id}/classification"
        self.mqtt_client.publish(topic, json.dumps(result))

        print(f"발행: {topic}")
        print(f"  Top-1: {predictions[0]['label']} ({predictions[0]['score']:.2f})")

        return result

    def shutdown(self):
        """정리"""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

# 사용 예
if __name__ == "__main__":
    node = AIEdgeNode(
        model_path="mobilenet_v2.tflite",
        labels_path="imagenet_labels.txt",
        mqtt_broker="localhost"
    )

    try:
        # 테스트 이미지 분류
        node.classify_and_publish("test_image.jpg")
    finally:
        node.shutdown()
```

---

## 연습 문제

### 문제 1: 모델 변환
1. Keras 모델을 TFLite로 변환하세요.
2. 동적 양자화를 적용하고 크기를 비교하세요.

### 문제 2: 성능 최적화
1. 동일 모델의 FP32, FP16, INT8 버전 성능을 비교하세요.
2. 라즈베리파이에서 FPS를 측정하세요.

### 문제 3: 실시간 분류
1. Pi Camera로 실시간 이미지 분류를 구현하세요.
2. 분류 결과를 MQTT로 발행하세요.

---

## 다음 단계

- [Edge AI - ONNX Runtime](09_Edge_AI_ONNX.md): ONNX Runtime으로 Edge AI
- [영상 분석 프로젝트](11_Image_Analysis_Project.md): 영상 분석 프로젝트

---

*최종 업데이트: 2026-02-01*
