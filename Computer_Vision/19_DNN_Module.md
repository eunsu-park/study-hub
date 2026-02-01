# 딥러닝 DNN 모듈 (Deep Neural Network Module)

## 개요

OpenCV의 DNN 모듈은 사전 학습된 딥러닝 모델을 로드하고 추론하는 기능을 제공합니다. TensorFlow, Caffe, Darknet, ONNX 등 다양한 프레임워크의 모델을 지원하며, CPU와 GPU에서 효율적으로 실행할 수 있습니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: 딥러닝 기초 개념, 객체 검출, 이미지 분류

---

## 목차

1. [cv2.dnn 모듈 개요](#1-cv2dnn-모듈-개요)
2. [readNet(): 모델 로딩](#2-readnet-모델-로딩)
3. [blobFromImage(): 전처리](#3-blobfromimage-전처리)
4. [YOLO 객체 검출](#4-yolo-객체-검출)
5. [SSD (Single Shot Detector)](#5-ssd-single-shot-detector)
6. [DNN 얼굴 검출](#6-dnn-얼굴-검출)
7. [연습 문제](#7-연습-문제)

---

## 1. cv2.dnn 모듈 개요

### DNN 모듈의 특징

```
OpenCV DNN 모듈:

┌─────────────────────────────────────────────────────────────────┐
│                        cv2.dnn                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  지원 프레임워크:                                               │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐     │
│  │   Caffe     │ TensorFlow  │   Darknet   │    ONNX     │     │
│  │  (.caffemodel,│ (.pb)     │  (.weights, │  (.onnx)    │     │
│  │   .prototxt)│             │   .cfg)     │             │     │
│  └─────────────┴─────────────┴─────────────┴─────────────┘     │
│                                                                 │
│  실행 백엔드:                                                   │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐     │
│  │    CPU      │   OpenCL    │    CUDA     │   Vulkan    │     │
│  │  (기본)     │   (GPU)     │   (NVIDIA)  │   (다중GPU) │     │
│  └─────────────┴─────────────┴─────────────┴─────────────┘     │
│                                                                 │
│  특징:                                                          │
│  - 추론 전용 (학습 불가)                                        │
│  - 최적화된 연산                                                │
│  - 다양한 하드웨어 지원                                         │
│  - 간단한 API                                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 기본 워크플로우

```
DNN 추론 워크플로우:

1. 모델 로딩
   ┌──────────────────┐
   │  readNet()       │ → 모델 파일 로드
   └────────┬─────────┘
            │
            ▼
2. 백엔드/타겟 설정
   ┌──────────────────┐
   │ setPreferableBackend()│ → CPU/CUDA/OpenCL
   │ setPreferableTarget() │
   └────────┬─────────┘
            │
            ▼
3. 입력 전처리
   ┌──────────────────┐
   │ blobFromImage()  │ → 이미지 → Blob
   └────────┬─────────┘
            │
            ▼
4. 추론 실행
   ┌──────────────────┐
   │ net.setInput()   │
   │ net.forward()    │ → 추론 결과
   └────────┬─────────┘
            │
            ▼
5. 결과 후처리
   ┌──────────────────┐
   │ NMS, 시각화 등   │
   └──────────────────┘
```

---

## 2. readNet(): 모델 로딩

### 다양한 모델 형식 로딩

```python
import cv2

# Caffe 모델 로딩
net_caffe = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',      # 네트워크 구조
    'model.caffemodel'      # 가중치
)

# TensorFlow 모델 로딩
net_tf = cv2.dnn.readNetFromTensorflow(
    'frozen_inference_graph.pb',  # 동결된 그래프
    'graph.pbtxt'                 # 텍스트 그래프 (선택)
)

# Darknet (YOLO) 모델 로딩
net_darknet = cv2.dnn.readNetFromDarknet(
    'yolov3.cfg',           # 설정 파일
    'yolov3.weights'        # 가중치
)

# ONNX 모델 로딩
net_onnx = cv2.dnn.readNetFromONNX('model.onnx')

# 범용 함수 (자동 감지)
net = cv2.dnn.readNet('model.weights', 'model.cfg')
```

### 백엔드 및 타겟 설정

```python
import cv2

net = cv2.dnn.readNet('model.weights', 'model.cfg')

# 백엔드 설정
# - cv2.dnn.DNN_BACKEND_OPENCV: OpenCV 내장 (기본)
# - cv2.dnn.DNN_BACKEND_CUDA: NVIDIA CUDA
# - cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE: Intel OpenVINO

# 타겟 설정
# - cv2.dnn.DNN_TARGET_CPU: CPU (기본)
# - cv2.dnn.DNN_TARGET_OPENCL: OpenCL GPU
# - cv2.dnn.DNN_TARGET_CUDA: NVIDIA GPU
# - cv2.dnn.DNN_TARGET_CUDA_FP16: NVIDIA GPU (반정밀도)

# CPU 실행 (기본)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# CUDA GPU 실행 (OpenCV가 CUDA 지원으로 빌드된 경우)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 사용 가능한 백엔드 확인
print("Available backends:", cv2.dnn.getAvailableBackends())
```

### 레이어 정보 확인

```python
import cv2

net = cv2.dnn.readNet('yolov3.cfg', 'yolov3.weights')

# 레이어 이름 목록
layer_names = net.getLayerNames()
print(f"총 레이어 수: {len(layer_names)}")
print("레이어 목록 (일부):", layer_names[:10])

# 출력 레이어 (연결되지 않은 레이어)
output_layers = net.getUnconnectedOutLayers()
output_layer_names = [layer_names[i - 1] for i in output_layers]
print("출력 레이어:", output_layer_names)

# 특정 레이어 정보
layer = net.getLayer(0)
print(f"레이어 타입: {layer.type}")
```

---

## 3. blobFromImage(): 전처리

### Blob 개념

```
Blob (Binary Large Object):
DNN 모델 입력을 위한 4차원 텐서

차원 구조:
┌─────────────────────────────────────────────────────────────┐
│  Blob Shape: (N, C, H, W)                                   │
│                                                             │
│  N: 배치 크기 (Batch Size)                                  │
│     - 한 번에 처리할 이미지 수                              │
│                                                             │
│  C: 채널 수 (Channels)                                      │
│     - RGB: 3, 그레이스케일: 1                               │
│                                                             │
│  H: 높이 (Height)                                           │
│     - 모델이 요구하는 입력 높이                             │
│                                                             │
│  W: 너비 (Width)                                            │
│     - 모델이 요구하는 입력 너비                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

예시: YOLO (416x416)
blob.shape = (1, 3, 416, 416)
             │  │   │    │
             │  │   │    └── 너비 416
             │  │   └── 높이 416
             │  └── RGB 3채널
             └── 이미지 1장
```

### blobFromImage 사용법

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 기본 Blob 생성
blob = cv2.dnn.blobFromImage(
    img,                    # 입력 이미지
    scalefactor=1/255.0,    # 스케일 팩터 (정규화)
    size=(416, 416),        # 목표 크기
    mean=(0, 0, 0),         # 평균값 빼기
    swapRB=True,            # BGR → RGB
    crop=False              # 크롭 여부
)

print(f"Blob shape: {blob.shape}")  # (1, 3, 416, 416)

# 다양한 전처리 옵션

# 1. ImageNet 스타일 (mean subtraction)
blob_imagenet = cv2.dnn.blobFromImage(
    img,
    scalefactor=1.0,
    size=(224, 224),
    mean=(104.0, 117.0, 123.0),  # ImageNet 평균
    swapRB=True
)

# 2. YOLO 스타일 (0-1 정규화)
blob_yolo = cv2.dnn.blobFromImage(
    img,
    scalefactor=1/255.0,
    size=(416, 416),
    mean=(0, 0, 0),
    swapRB=True
)

# 3. SSD 스타일
blob_ssd = cv2.dnn.blobFromImage(
    img,
    scalefactor=1.0,
    size=(300, 300),
    mean=(104.0, 177.0, 123.0),
    swapRB=True
)
```

### 여러 이미지 처리

```python
import cv2
import numpy as np

def prepare_batch(images, size=(416, 416)):
    """여러 이미지를 배치로 처리"""

    # 방법 1: blobFromImages 사용
    blob = cv2.dnn.blobFromImages(
        images,
        scalefactor=1/255.0,
        size=size,
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )

    return blob

# 사용 예
images = [cv2.imread(f'image{i}.jpg') for i in range(4)]
batch_blob = prepare_batch(images)
print(f"Batch blob shape: {batch_blob.shape}")  # (4, 3, 416, 416)

# 네트워크에 입력
net.setInput(batch_blob)
outputs = net.forward()
```

### 전처리 시각화

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_blob(blob):
    """Blob 내용 시각화"""

    # blob shape: (N, C, H, W)
    n, c, h, w = blob.shape

    for i in range(n):
        fig, axes = plt.subplots(1, c+1, figsize=(15, 4))

        # 각 채널 표시
        for j in range(c):
            channel = blob[i, j, :, :]
            axes[j].imshow(channel, cmap='gray')
            axes[j].set_title(f'Channel {j}')
            axes[j].axis('off')

        # 합성 이미지 (RGB로 재조합)
        if c == 3:
            combined = np.transpose(blob[i], (1, 2, 0))  # CHW → HWC
            combined = (combined * 255).astype(np.uint8)
            axes[c].imshow(combined)
            axes[c].set_title('Combined (RGB)')
            axes[c].axis('off')

        plt.suptitle(f'Image {i}')
        plt.tight_layout()
        plt.show()

# 사용 예
img = cv2.imread('image.jpg')
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True)
visualize_blob(blob)
```

---

## 4. YOLO 객체 검출

### YOLO 개요

```
YOLO (You Only Look Once):
실시간 객체 검출 알고리즘

특징:
- 단일 패스로 검출 (End-to-End)
- 빠른 속도 (실시간 가능)
- 전체 이미지 컨텍스트 사용

출력 구조:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  각 검출에 대해:                                            │
│  [center_x, center_y, width, height, confidence, class_scores...]│
│                                                             │
│  - center_x, center_y: 바운딩 박스 중심 (0-1 정규화)        │
│  - width, height: 박스 크기 (0-1 정규화)                    │
│  - confidence: 객체 존재 확률                               │
│  - class_scores: 각 클래스별 확률 (80개)                    │
│                                                             │
│  예: COCO 데이터셋 (80 클래스)                              │
│  출력 벡터 길이 = 4 + 1 + 80 = 85                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

YOLO 버전별 특징:
┌─────────┬──────────┬──────────┬───────────────────┐
│ 버전    │ 입력크기 │  mAP     │ 속도 (FPS)        │
├─────────┼──────────┼──────────┼───────────────────┤
│ YOLOv3  │ 416x416  │ 33.0     │ ~35 (GPU)         │
│ YOLOv3-tiny│ 416x416│ 15.0    │ ~220 (GPU)        │
│ YOLOv4  │ 416x416  │ 43.5     │ ~38 (GPU)         │
│ YOLOv4-tiny│ 416x416│ 21.7    │ ~371 (GPU)        │
└─────────┴──────────┴──────────┴───────────────────┘
```

### YOLOv3 구현

```python
import cv2
import numpy as np

class YOLODetector:
    """YOLOv3 객체 검출기"""

    def __init__(self, config_path, weights_path, names_path,
                 conf_threshold=0.5, nms_threshold=0.4):
        # 모델 로딩
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 클래스 이름 로딩
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # 출력 레이어 이름
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1]
                              for i in self.net.getUnconnectedOutLayers()]

        # 임계값
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # 색상 (클래스별)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3),
                                        dtype=np.uint8)

    def detect(self, img):
        """객체 검출"""
        height, width = img.shape[:2]

        # Blob 생성
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416),
                                      swapRB=True, crop=False)

        # 추론
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # 결과 파싱
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:
                    # 바운딩 박스 좌표 (정규화됨)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # 좌상단 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                    self.conf_threshold, self.nms_threshold)

        results = []
        for i in indices:
            box = boxes[i]
            results.append({
                'box': box,
                'confidence': confidences[i],
                'class_id': class_ids[i],
                'class_name': self.classes[class_ids[i]]
            })

        return results

    def draw(self, img, results):
        """결과 시각화"""
        for det in results:
            x, y, w, h = det['box']
            color = [int(c) for c in self.colors[det['class_id']]]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(img, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img

# 사용 예
# 모델 파일 다운로드 필요:
# - yolov3.cfg: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
# - yolov3.weights: https://pjreddie.com/media/files/yolov3.weights
# - coco.names: https://github.com/pjreddie/darknet/blob/master/data/coco.names

detector = YOLODetector(
    'yolov3.cfg',
    'yolov3.weights',
    'coco.names'
)

img = cv2.imread('street.jpg')
results = detector.detect(img)
output = detector.draw(img, results)

print(f"검출된 객체 수: {len(results)}")
for r in results:
    print(f"  - {r['class_name']}: {r['confidence']:.2%}")

cv2.imshow('YOLO Detection', output)
cv2.waitKey(0)
```

### YOLOv3-tiny (경량 버전)

```python
import cv2
import numpy as np

def yolo_tiny_detect(img, conf_threshold=0.5):
    """YOLOv3-tiny를 이용한 빠른 검출"""

    # YOLOv3-tiny 로딩
    net = cv2.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')

    # 클래스 이름
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # 출력 레이어
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    height, width = img.shape[:2]

    # Blob 및 추론
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # 결과 파싱
    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

    # 결과 그리기
    for i in indices:
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img
```

---

## 5. SSD (Single Shot Detector)

### SSD 개요

```
SSD (Single Shot MultiBox Detector):
다중 스케일 특징맵을 이용한 객체 검출

특징:
- 다양한 크기의 객체 검출에 강함
- YOLO보다 작은 객체 검출 우수
- 다양한 백본 네트워크 사용 가능 (VGG, MobileNet 등)

아키텍처:
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  입력 이미지 (300x300)                                     │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────┐                                              │
│  │ 백본     │ VGG16, MobileNet 등                         │
│  │ 네트워크 │                                              │
│  └────┬─────┘                                              │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ 38x38    │  │ 19x19    │  │ 10x10    │  │ 5x5 ...  │   │
│  │ 특징맵   │  │ 특징맵   │  │ 특징맵   │  │ 특징맵   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │          │
│       └─────────────┴─────────────┴─────────────┘          │
│                         │                                  │
│                         ▼                                  │
│                    NMS → 최종 검출                         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### MobileNet SSD 구현

```python
import cv2
import numpy as np

class SSDDetector:
    """MobileNet SSD 객체 검출기"""

    # COCO 클래스 (MobileNet SSD v2)
    CLASSES = ["background", "person", "bicycle", "car", "motorcycle",
               "airplane", "bus", "train", "truck", "boat", "traffic light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird",
               "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
               "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
               "suitcase", "frisbee", "skis", "snowboard", "sports ball",
               "kite", "baseball bat", "baseball glove", "skateboard",
               "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
               "donut", "cake", "chair", "couch", "potted plant", "bed",
               "dining table", "toilet", "tv", "laptop", "mouse", "remote",
               "keyboard", "cell phone", "microwave", "oven", "toaster",
               "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"]

    def __init__(self, config_path, weights_path, conf_threshold=0.5):
        self.net = cv2.dnn.readNetFromTensorflow(weights_path, config_path)
        self.conf_threshold = conf_threshold

        np.random.seed(42)
        self.colors = np.random.randint(0, 255,
                                        size=(len(self.CLASSES), 3),
                                        dtype=np.uint8)

    def detect(self, img):
        """객체 검출"""
        height, width = img.shape[:2]

        # Blob 생성 (SSD는 300x300 또는 512x512 입력)
        blob = cv2.dnn.blobFromImage(img, size=(300, 300),
                                      mean=(127.5, 127.5, 127.5),
                                      scalefactor=1/127.5,
                                      swapRB=True)

        self.net.setInput(blob)
        detections = self.net.forward()

        results = []

        # 출력 shape: (1, 1, N, 7)
        # 각 검출: [batch_id, class_id, confidence, x1, y1, x2, y2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.conf_threshold:
                class_id = int(detections[0, 0, i, 1])

                # 좌표 (정규화됨 → 픽셀)
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)

                results.append({
                    'box': [x1, y1, x2 - x1, y2 - y1],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.CLASSES[class_id]
                })

        return results

    def draw(self, img, results):
        """결과 시각화"""
        for det in results:
            x, y, w, h = det['box']
            color = [int(c) for c in self.colors[det['class_id']]]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(img, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img

# 사용 예
# 모델 다운로드:
# ssd_mobilenet_v2_coco_2018_03_29.pbtxt
# frozen_inference_graph.pb

# detector = SSDDetector(
#     'ssd_mobilenet_v2_coco.pbtxt',
#     'frozen_inference_graph.pb'
# )
```

### Caffe SSD (경량)

```python
import cv2
import numpy as np

def ssd_caffe_detect(img, prototxt, caffemodel, conf_threshold=0.5):
    """Caffe SSD 검출 (MobileNet 백본)"""

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    height, width = img.shape[:2]

    # Blob 생성
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        0.007843,  # 1/127.5
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            class_id = int(detections[0, 0, i, 1])

            box = detections[0, 0, i, 3:7] * np.array([width, height,
                                                       width, height])
            x1, y1, x2, y2 = box.astype(int)

            results.append({
                'box': [x1, y1, x2 - x1, y2 - y1],
                'confidence': float(confidence),
                'class_id': class_id,
                'class_name': CLASSES[class_id]
            })

    return results
```

---

## 6. DNN 얼굴 검출

### OpenCV DNN 얼굴 검출기

```python
import cv2
import numpy as np

class DNNFaceDetector:
    """DNN 기반 얼굴 검출기 (res10_300x300)"""

    def __init__(self, model_path, config_path=None, conf_threshold=0.5):
        """
        모델 다운로드:
        - deploy.prototxt: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
        - res10_300x300_ssd_iter_140000.caffemodel
        """

        if config_path:
            # Caffe 모델
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        else:
            # TensorFlow 모델
            self.net = cv2.dnn.readNetFromTensorflow(model_path)

        self.conf_threshold = conf_threshold

    def detect(self, img):
        """얼굴 검출"""
        height, width = img.shape[:2]

        # Blob 생성
        blob = cv2.dnn.blobFromImage(
            img, 1.0, (300, 300),
            (104.0, 177.0, 123.0),  # 평균값
            swapRB=False,
            crop=False
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array(
                    [width, height, width, height]
                )
                x1, y1, x2, y2 = box.astype(int)

                # 경계 체크
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                faces.append({
                    'box': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': float(confidence)
                })

        return faces

    def draw(self, img, faces):
        """결과 시각화"""
        for face in faces:
            x, y, w, h = face['box']
            conf = face['confidence']

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Face: {conf:.2f}"
            cv2.putText(img, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

# 사용 예
detector = DNNFaceDetector(
    'res10_300x300_ssd_iter_140000.caffemodel',
    'deploy.prototxt'
)

img = cv2.imread('group_photo.jpg')
faces = detector.detect(img)
output = detector.draw(img, faces)

print(f"검출된 얼굴 수: {len(faces)}")
cv2.imshow('DNN Face Detection', output)
cv2.waitKey(0)
```

### 실시간 DNN 얼굴 검출

```python
import cv2
import time

def realtime_dnn_face_detection():
    """실시간 DNN 얼굴 검출"""

    # 모델 로딩
    net = cv2.dnn.readNetFromCaffe(
        'deploy.prototxt',
        'res10_300x300_ssd_iter_140000.caffemodel'
    )

    cap = cv2.VideoCapture(0)

    fps_time = time.time()
    fps = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Blob 생성
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0), False, False
        )

        net.setInput(blob)
        detections = net.forward()

        # 검출 결과 처리
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{confidence:.2%}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # FPS 계산
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()

        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('DNN Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행
realtime_dnn_face_detection()
```

### Haar vs DNN 얼굴 검출 비교

```python
import cv2
import time
import numpy as np

def compare_face_detectors(img):
    """Haar와 DNN 얼굴 검출 비교"""

    # Haar Cascade
    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # DNN
    dnn_net = cv2.dnn.readNetFromCaffe(
        'deploy.prototxt',
        'res10_300x300_ssd_iter_140000.caffemodel'
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # Haar 검출
    start = time.time()
    haar_faces = haar.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    haar_time = time.time() - start

    # DNN 검출
    start = time.time()
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0), False, False)
    dnn_net.setInput(blob)
    dnn_detections = dnn_net.forward()
    dnn_time = time.time() - start

    # 결과 시각화
    img_haar = img.copy()
    img_dnn = img.copy()

    for (x, y, w, h) in haar_faces:
        cv2.rectangle(img_haar, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for i in range(dnn_detections.shape[2]):
        conf = dnn_detections[0, 0, i, 2]
        if conf > 0.5:
            box = dnn_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_dnn, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(img_haar, f"Haar: {len(haar_faces)} faces, {haar_time*1000:.1f}ms",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    dnn_count = sum(1 for i in range(dnn_detections.shape[2])
                    if dnn_detections[0, 0, i, 2] > 0.5)
    cv2.putText(img_dnn, f"DNN: {dnn_count} faces, {dnn_time*1000:.1f}ms",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 결과 표시
    combined = np.hstack([img_haar, img_dnn])
    cv2.imshow('Haar vs DNN', combined)
    cv2.waitKey(0)

    return {
        'haar': {'count': len(haar_faces), 'time': haar_time},
        'dnn': {'count': dnn_count, 'time': dnn_time}
    }

# 비교 실행
# result = compare_face_detectors(cv2.imread('group.jpg'))
```

---

## 7. 연습 문제

### 문제 1: 객체 검출 성능 비교

YOLO와 SSD의 성능을 비교하는 프로그램을 작성하세요.

**요구사항**:
- 동일한 테스트 이미지 세트 사용
- 검출 속도 측정
- 검출 정확도 비교 (IoU 기반)
- 결과 시각화

<details>
<summary>힌트</summary>

```python
def compare_detectors(img, yolo_detector, ssd_detector):
    # YOLO 검출
    yolo_start = time.time()
    yolo_results = yolo_detector.detect(img)
    yolo_time = time.time() - yolo_start

    # SSD 검출
    ssd_start = time.time()
    ssd_results = ssd_detector.detect(img)
    ssd_time = time.time() - ssd_start

    return {
        'yolo': {'results': yolo_results, 'time': yolo_time},
        'ssd': {'results': ssd_results, 'time': ssd_time}
    }
```

</details>

### 문제 2: 커스텀 클래스 필터링

특정 클래스만 검출하는 필터링 기능을 구현하세요.

**요구사항**:
- 검출할 클래스 목록 지정
- 다른 클래스 무시
- 클래스별 색상 지정
- 클래스별 신뢰도 임계값 설정

<details>
<summary>힌트</summary>

```python
class FilteredDetector:
    def __init__(self, base_detector, target_classes):
        self.detector = base_detector
        self.target_classes = target_classes
        self.class_thresholds = {cls: 0.5 for cls in target_classes}

    def detect(self, img):
        all_results = self.detector.detect(img)
        filtered = [r for r in all_results
                    if r['class_name'] in self.target_classes and
                    r['confidence'] > self.class_thresholds[r['class_name']]]
        return filtered
```

</details>

### 문제 3: 비디오 객체 추적 + 검출

검출과 추적을 결합하여 안정적인 비디오 객체 인식을 구현하세요.

**요구사항**:
- N 프레임마다 검출 수행
- 중간 프레임은 추적으로 대체
- ID 할당 및 유지
- 추적 실패 시 재검출

<details>
<summary>힌트</summary>

```python
class DetectionTracker:
    def __init__(self, detector, detect_every_n=5):
        self.detector = detector
        self.detect_every_n = detect_every_n
        self.trackers = {}
        self.frame_count = 0

    def process(self, frame):
        if self.frame_count % self.detect_every_n == 0:
            # 새로운 검출
            detections = self.detector.detect(frame)
            self.update_trackers(frame, detections)
        else:
            # 기존 트래커 업데이트
            self.update_existing_trackers(frame)

        self.frame_count += 1
        return self.get_current_positions()
```

</details>

### 문제 4: 모델 앙상블

여러 모델의 결과를 결합하여 정확도를 높이세요.

**요구사항**:
- 2개 이상의 모델 사용
- 검출 결과 병합 (Weighted NMS)
- 신뢰도 보정
- 최종 결과 출력

<details>
<summary>힌트</summary>

```python
def ensemble_detection(img, detectors, weights=None):
    all_boxes = []
    all_scores = []
    all_classes = []

    for i, detector in enumerate(detectors):
        results = detector.detect(img)
        weight = weights[i] if weights else 1.0

        for r in results:
            all_boxes.append(r['box'])
            all_scores.append(r['confidence'] * weight)
            all_classes.append(r['class_id'])

    # Soft-NMS 또는 Weighted Box Fusion
    final_results = weighted_nms(all_boxes, all_scores, all_classes)
    return final_results
```

</details>

### 문제 5: 실시간 객체 계수 시스템

비디오에서 특정 객체(예: 사람, 차량)를 계수하는 시스템을 구현하세요.

**요구사항**:
- 실시간 검출
- 계수 라인/영역 설정
- 진입/퇴장 구분
- 통계 표시 (시간별, 누적)

<details>
<summary>힌트</summary>

```python
class ObjectCounter:
    def __init__(self, detector, count_line_y):
        self.detector = detector
        self.count_line_y = count_line_y
        self.tracked_objects = {}  # {id: previous_y}
        self.count_in = 0
        self.count_out = 0

    def process(self, frame):
        results = self.detector.detect(frame)

        for obj in results:
            # 객체 중심 y 좌표
            _, y, _, h = obj['box']
            center_y = y + h // 2

            # 이전 위치와 비교하여 라인 통과 확인
            if obj['id'] in self.tracked_objects:
                prev_y = self.tracked_objects[obj['id']]
                if prev_y < self.count_line_y <= center_y:
                    self.count_out += 1
                elif prev_y > self.count_line_y >= center_y:
                    self.count_in += 1

            self.tracked_objects[obj['id']] = center_y
```

</details>

---

## 다음 단계

- [20_Practical_Projects.md](./20_Practical_Projects.md) - 문서 스캐너, 차선 검출, AR 마커, 얼굴 필터

---

## 참고 자료

- [OpenCV DNN Module](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [SSD Paper](https://arxiv.org/abs/1512.02325)
- [OpenCV Model Zoo](https://github.com/opencv/opencv/tree/master/samples/dnn)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
