# Deep Neural Network Module (DNN Module)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how OpenCV's DNN module loads and runs inference with pre-trained models from multiple frameworks.
2. Implement model loading with `readNet()` and input preprocessing with `blobFromImage()`.
3. Apply the DNN module to perform YOLO-based object detection on images and video streams.
4. Compare SSD and YOLO architectures for real-time object detection tasks.
5. Implement DNN-based face detection and modern ONNX model inference in a complete pipeline.

---

## Overview

OpenCV's DNN module provides functionality to load and run inference with pre-trained deep learning models. It supports models from various frameworks including TensorFlow, Caffe, Darknet, and ONNX, and can run efficiently on both CPU and GPU.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: Deep learning fundamentals, object detection, image classification

---

## Table of Contents

1. [cv2.dnn Module Overview](#1-cv2dnn-module-overview)
2. [readNet(): Model Loading](#2-readnet-model-loading)
3. [blobFromImage(): Preprocessing](#3-blobfromimage-preprocessing)
4. [YOLO Object Detection](#4-yolo-object-detection)
5. [SSD (Single Shot Detector)](#5-ssd-single-shot-detector)
6. [DNN Face Detection](#6-dnn-face-detection)
7. [Modern Object Detection with ONNX](#7-modern-object-detection-with-onnx)
8. [Exercises](#8-exercises)

---

## 1. cv2.dnn Module Overview

The OpenCV DNN module solves a real deployment problem: you trained a model in PyTorch or TensorFlow, but shipping those heavy frameworks to an edge device or embedding them into a C++ application is impractical. The DNN module lets you run inference on models from any major framework using only OpenCV, with no additional runtime dependency. This matters especially for embedded systems, mobile apps, and production pipelines where minimizing dependencies is critical.

### DNN Module Features

```
OpenCV DNN Module:

┌─────────────────────────────────────────────────────────────────┐
│                        cv2.dnn                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Supported Frameworks:                                          │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐     │
│  │   Caffe     │ TensorFlow  │   Darknet   │    ONNX     │     │
│  │  (.caffemodel,│ (.pb)     │  (.weights, │  (.onnx)    │     │
│  │   .prototxt)│             │   .cfg)     │             │     │
│  └─────────────┴─────────────┴─────────────┴─────────────┘     │
│                                                                 │
│  Execution Backends:                                            │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐     │
│  │    CPU      │   OpenCL    │    CUDA     │   Vulkan    │     │
│  │  (default)  │   (GPU)     │   (NVIDIA)  │  (multi-GPU)│     │
│  └─────────────┴─────────────┴─────────────┴─────────────┘     │
│                                                                 │
│  Features:                                                      │
│  - Inference only (no training)                                 │
│  - Optimized operations                                         │
│  - Multiple hardware support                                    │
│  - Simple API                                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Basic Workflow

```
DNN Inference Workflow:

1. Model Loading
   ┌──────────────────┐
   │  readNet()       │ → Load model file
   └────────┬─────────┘
            │
            ▼
2. Backend/Target Setup
   ┌──────────────────┐
   │ setPreferableBackend()│ → CPU/CUDA/OpenCL
   │ setPreferableTarget() │
   └────────┬─────────┘
            │
            ▼
3. Input Preprocessing
   ┌──────────────────┐
   │ blobFromImage()  │ → Image → Blob
   └────────┬─────────┘
            │
            ▼
4. Run Inference
   ┌──────────────────┐
   │ net.setInput()   │
   │ net.forward()    │ → Inference result
   └────────┬─────────┘
            │
            ▼
5. Post-processing
   ┌──────────────────┐
   │ NMS, visualization, etc. │
   └──────────────────┘
```

---

## 2. readNet(): Model Loading

### Loading Various Model Formats

```python
import cv2

# Loading Caffe model
net_caffe = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',      # Network structure
    'model.caffemodel'      # Weights
)

# Loading TensorFlow model
net_tf = cv2.dnn.readNetFromTensorflow(
    'frozen_inference_graph.pb',  # Frozen graph
    'graph.pbtxt'                 # Text graph (optional)
)

# Loading Darknet (YOLO) model
net_darknet = cv2.dnn.readNetFromDarknet(
    'yolov3.cfg',           # Config file
    'yolov3.weights'        # Weights
)

# Loading ONNX model
net_onnx = cv2.dnn.readNetFromONNX('model.onnx')

# Generic function (auto-detect)
net = cv2.dnn.readNet('model.weights', 'model.cfg')
```

### Backend and Target Configuration

```python
import cv2

net = cv2.dnn.readNet('model.weights', 'model.cfg')

# Backend options
# - cv2.dnn.DNN_BACKEND_OPENCV: OpenCV built-in (default)
# - cv2.dnn.DNN_BACKEND_CUDA: NVIDIA CUDA
# - cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE: Intel OpenVINO

# Target options
# - cv2.dnn.DNN_TARGET_CPU: CPU (default)
# - cv2.dnn.DNN_TARGET_OPENCL: OpenCL GPU
# - cv2.dnn.DNN_TARGET_CUDA: NVIDIA GPU
# - cv2.dnn.DNN_TARGET_CUDA_FP16: NVIDIA GPU (half precision)

# CPU execution (default)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# CUDA GPU execution (if OpenCV is built with CUDA support)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Check available backends
print("Available backends:", cv2.dnn.getAvailableBackends())
```

### Checking Layer Information

```python
import cv2

net = cv2.dnn.readNet('yolov3.cfg', 'yolov3.weights')

# Layer names list
layer_names = net.getLayerNames()
print(f"Total layers: {len(layer_names)}")
print("Layer list (partial):", layer_names[:10])

# Output layers (unconnected layers)
output_layers = net.getUnconnectedOutLayers()
output_layer_names = [layer_names[i - 1] for i in output_layers]
print("Output layers:", output_layer_names)

# Specific layer info
layer = net.getLayer(0)
print(f"Layer type: {layer.type}")
```

---

## 3. blobFromImage(): Preprocessing

### Blob Concept

```
Blob (Binary Large Object):
4-dimensional tensor for DNN model input

Dimension structure:
┌─────────────────────────────────────────────────────────────┐
│  Blob Shape: (N, C, H, W)                                   │
│                                                             │
│  N: Batch Size                                              │
│     - Number of images to process at once                   │
│                                                             │
│  C: Number of Channels                                      │
│     - RGB: 3, Grayscale: 1                                  │
│                                                             │
│  H: Height                                                  │
│     - Input height required by model                        │
│                                                             │
│  W: Width                                                   │
│     - Input width required by model                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Example: YOLO (416x416)
blob.shape = (1, 3, 416, 416)
             │  │   │    │
             │  │   │    └── Width 416
             │  │   └── Height 416
             │  └── RGB 3 channels
             └── 1 image
```

### Using blobFromImage

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Basic Blob creation
blob = cv2.dnn.blobFromImage(
    img,                    # Input image
    scalefactor=1/255.0,    # Divide by 255 to map [0,255]→[0,1]; the model was trained on this range
    size=(416, 416),        # Target size
    mean=(0, 0, 0),         # Mean subtraction — set to (0,0,0) for YOLO since it uses pure [0,1] normalization;
                            #   ImageNet-trained models use per-channel means instead (e.g. 104,117,123)
    swapRB=True,            # OpenCV loads images as BGR; most DNN models expect RGB — this fixes the mismatch
    crop=False              # False = stretch to fit; True = center-crop (changes aspect ratio, hurts accuracy)
)

print(f"Blob shape: {blob.shape}")  # (1, 3, 416, 416)

# Various preprocessing options

# 1. ImageNet style (mean subtraction)
blob_imagenet = cv2.dnn.blobFromImage(
    img,
    scalefactor=1.0,         # No rescaling — the model was trained on raw [0,255] with only mean subtracted
    size=(224, 224),
    mean=(104.0, 117.0, 123.0),  # Per-channel ImageNet mean (BGR order); subtracting removes dataset-wide color bias
    swapRB=True
)

# 2. YOLO style (0-1 normalization)
blob_yolo = cv2.dnn.blobFromImage(
    img,
    scalefactor=1/255.0,
    size=(416, 416),
    mean=(0, 0, 0),
    swapRB=True
)

# 3. SSD style
blob_ssd = cv2.dnn.blobFromImage(
    img,
    scalefactor=1.0,
    size=(300, 300),
    mean=(104.0, 177.0, 123.0),
    swapRB=True
)
```

### Processing Multiple Images

```python
import cv2
import numpy as np

def prepare_batch(images, size=(416, 416)):
    """Process multiple images as a batch"""

    # Method 1: Using blobFromImages
    blob = cv2.dnn.blobFromImages(
        images,
        scalefactor=1/255.0,
        size=size,
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )

    return blob

# Usage example
images = [cv2.imread(f'image{i}.jpg') for i in range(4)]
batch_blob = prepare_batch(images)
print(f"Batch blob shape: {batch_blob.shape}")  # (4, 3, 416, 416)

# Input to network
net.setInput(batch_blob)
outputs = net.forward()
```

### Visualizing Preprocessing

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_blob(blob):
    """Visualize blob contents"""

    # blob shape: (N, C, H, W)
    n, c, h, w = blob.shape

    for i in range(n):
        fig, axes = plt.subplots(1, c+1, figsize=(15, 4))

        # Display each channel
        for j in range(c):
            channel = blob[i, j, :, :]
            axes[j].imshow(channel, cmap='gray')
            axes[j].set_title(f'Channel {j}')
            axes[j].axis('off')

        # Combined image (recombine as RGB)
        if c == 3:
            combined = np.transpose(blob[i], (1, 2, 0))  # CHW → HWC
            combined = (combined * 255).astype(np.uint8)
            axes[c].imshow(combined)
            axes[c].set_title('Combined (RGB)')
            axes[c].axis('off')

        plt.suptitle(f'Image {i}')
        plt.tight_layout()
        plt.show()

# Usage example
img = cv2.imread('image.jpg')
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True)
visualize_blob(blob)
```

---

## 4. YOLO Object Detection

### YOLO Overview

```
YOLO (You Only Look Once):
Real-time object detection algorithm

Features:
- Single pass detection (End-to-End)
- Fast speed (real-time capable)
- Uses full image context

Output structure:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  For each detection:                                        │
│  [center_x, center_y, width, height, confidence, class_scores...]│
│                                                             │
│  - center_x, center_y: Bounding box center (0-1 normalized) │
│  - width, height: Box size (0-1 normalized)                 │
│  - confidence: Object presence probability                  │
│  - class_scores: Probability for each class (80 classes)    │
│                                                             │
│  Example: COCO dataset (80 classes)                         │
│  Output vector length = 4 + 1 + 80 = 85                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

YOLO version comparison:
┌─────────┬──────────┬──────────┬───────────────────┐
│ Version │ Input    │  mAP     │ Speed (FPS)       │
├─────────┼──────────┼──────────┼───────────────────┤
│ YOLOv3  │ 416x416  │ 33.0     │ ~35 (GPU)         │
│ YOLOv3-tiny│ 416x416│ 15.0    │ ~220 (GPU)        │
│ YOLOv4  │ 416x416  │ 43.5     │ ~38 (GPU)         │
│ YOLOv4-tiny│ 416x416│ 21.7    │ ~371 (GPU)        │
└─────────┴──────────┴──────────┴───────────────────┘
```

### YOLOv3 Implementation

```python
import cv2
import numpy as np

class YOLODetector:
    """YOLOv3 Object Detector"""

    def __init__(self, config_path, weights_path, names_path,
                 conf_threshold=0.5, nms_threshold=0.4):
        # Load model
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Load class names
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Output layer names — YOLO has 3 detection heads (at different scales);
        # getUnconnectedOutLayers() finds them automatically regardless of model variant
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1]
                              for i in self.net.getUnconnectedOutLayers()]

        # conf_threshold=0.5: only keep detections with >50% class confidence;
        #   lower values increase recall but add false positives
        # nms_threshold=0.4: during NMS, boxes with IoU > 0.4 are suppressed;
        #   lower = more aggressive suppression (fewer duplicate boxes)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # Seed ensures consistent per-class colors across runs for easy visual tracking
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3),
                                        dtype=np.uint8)

    def detect(self, img):
        """Object detection"""
        height, width = img.shape[:2]

        # Create blob
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416),
                                      swapRB=True, crop=False)

        # Inference
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # Parse results
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                # detection[0:4] = bbox, detection[4] = objectness (YOLOv3), detection[5:] = class probs
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:
                    # YOLO outputs coordinates normalized to [0,1]; multiply by image dims to get pixels
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Convert from center format (cx,cy,w,h) to top-left format (x,y,w,h) for NMSBoxes
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # NMS removes redundant overlapping boxes — YOLO generates many candidates per object,
        # NMS keeps only the highest-confidence box when boxes overlap by more than nms_threshold IoU
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
        """Visualize results"""
        for det in results:
            x, y, w, h = det['box']
            color = [int(c) for c in self.colors[det['class_id']]]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(img, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img

# Usage example
# Model files download required:
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

print(f"Detected objects: {len(results)}")
for r in results:
    print(f"  - {r['class_name']}: {r['confidence']:.2%}")

cv2.imshow('YOLO Detection', output)
cv2.waitKey(0)
```

### YOLOv3-tiny (Lightweight Version)

```python
import cv2
import numpy as np

def yolo_tiny_detect(img, conf_threshold=0.5):
    """Fast detection using YOLOv3-tiny"""

    # Load YOLOv3-tiny
    net = cv2.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')

    # Class names
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    height, width = img.shape[:2]

    # Blob and inference
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Parse results
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

    # Draw results
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

### SSD Overview

```
SSD (Single Shot MultiBox Detector):
Object detection using multi-scale feature maps

Features:
- Strong at detecting objects of various sizes
- Better at detecting small objects than YOLO
- Various backbone networks available (VGG, MobileNet, etc.)

Architecture:
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  Input Image (300x300)                                     │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────┐                                              │
│  │ Backbone │ VGG16, MobileNet, etc.                      │
│  │ Network  │                                              │
│  └────┬─────┘                                              │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ 38x38    │  │ 19x19    │  │ 10x10    │  │ 5x5 ...  │   │
│  │ Feature  │  │ Feature  │  │ Feature  │  │ Feature  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │          │
│       └─────────────┴─────────────┴─────────────┘          │
│                         │                                  │
│                         ▼                                  │
│                    NMS → Final Detection                   │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### MobileNet SSD Implementation

```python
import cv2
import numpy as np

class SSDDetector:
    """MobileNet SSD Object Detector"""

    # COCO classes (MobileNet SSD v2)
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
        """Object detection"""
        height, width = img.shape[:2]

        # Create blob (SSD uses 300x300 or 512x512 input)
        blob = cv2.dnn.blobFromImage(img, size=(300, 300),
                                      mean=(127.5, 127.5, 127.5),
                                      scalefactor=1/127.5,
                                      swapRB=True)

        self.net.setInput(blob)
        detections = self.net.forward()

        results = []

        # Output shape: (1, 1, N, 7)
        # Each detection: [batch_id, class_id, confidence, x1, y1, x2, y2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.conf_threshold:
                class_id = int(detections[0, 0, i, 1])

                # Coordinates (normalized → pixels)
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
        """Visualize results"""
        for det in results:
            x, y, w, h = det['box']
            color = [int(c) for c in self.colors[det['class_id']]]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(img, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img

# Usage example
# Model download:
# ssd_mobilenet_v2_coco_2018_03_29.pbtxt
# frozen_inference_graph.pb

# detector = SSDDetector(
#     'ssd_mobilenet_v2_coco.pbtxt',
#     'frozen_inference_graph.pb'
# )
```

### Caffe SSD (Lightweight)

```python
import cv2
import numpy as np

def ssd_caffe_detect(img, prototxt, caffemodel, conf_threshold=0.5):
    """Caffe SSD detection (MobileNet backbone)"""

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    height, width = img.shape[:2]

    # Create blob
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

## 6. DNN Face Detection

### OpenCV DNN Face Detector

```python
import cv2
import numpy as np

class DNNFaceDetector:
    """DNN-based Face Detector (res10_300x300)"""

    def __init__(self, model_path, config_path=None, conf_threshold=0.5):
        """
        Model download:
        - deploy.prototxt: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
        - res10_300x300_ssd_iter_140000.caffemodel
        """

        if config_path:
            # Caffe model
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        else:
            # TensorFlow model
            self.net = cv2.dnn.readNetFromTensorflow(model_path)

        self.conf_threshold = conf_threshold

    def detect(self, img):
        """Face detection"""
        height, width = img.shape[:2]

        # Create blob
        blob = cv2.dnn.blobFromImage(
            img, 1.0, (300, 300),
            (104.0, 177.0, 123.0),  # Per-channel means the res10 model was trained with (BGR order);
                                    # using the training-time means keeps pixel distributions matched
            swapRB=False,           # The Caffe res10 model was trained on BGR, matching OpenCV's native order
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

                # Boundary check
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
        """Visualize results"""
        for face in faces:
            x, y, w, h = face['box']
            conf = face['confidence']

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Face: {conf:.2f}"
            cv2.putText(img, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

# Usage example
detector = DNNFaceDetector(
    'res10_300x300_ssd_iter_140000.caffemodel',
    'deploy.prototxt'
)

img = cv2.imread('group_photo.jpg')
faces = detector.detect(img)
output = detector.draw(img, faces)

print(f"Detected faces: {len(faces)}")
cv2.imshow('DNN Face Detection', output)
cv2.waitKey(0)
```

### Real-time DNN Face Detection

```python
import cv2
import time

def realtime_dnn_face_detection():
    """Real-time DNN face detection"""

    # Load model
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

        # Create blob
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0), False, False
        )

        net.setInput(blob)
        detections = net.forward()

        # Process detection results
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # 0.5 threshold balances precision and recall for real-time use;
            # lower values catch more faces but add false positives that hurt UX
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{confidence:.2%}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate FPS
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

# Run
realtime_dnn_face_detection()
```

### Haar vs DNN Face Detection Comparison

```python
import cv2
import time
import numpy as np

def compare_face_detectors(img):
    """Compare Haar and DNN face detection"""

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

    # Haar detection
    start = time.time()
    haar_faces = haar.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    haar_time = time.time() - start

    # DNN detection
    start = time.time()
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0), False, False)
    dnn_net.setInput(blob)
    dnn_detections = dnn_net.forward()
    dnn_time = time.time() - start

    # Visualize results
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

    # Display results
    combined = np.hstack([img_haar, img_dnn])
    cv2.imshow('Haar vs DNN', combined)
    cv2.waitKey(0)

    return {
        'haar': {'count': len(haar_faces), 'time': haar_time},
        'dnn': {'count': dnn_count, 'time': dnn_time}
    }

# Run comparison
# result = compare_face_detectors(cv2.imread('group.jpg'))
```

---

## 7. Modern Object Detection with ONNX

### 7.1 Running YOLOv8 with OpenCV DNN

YOLOv8, released by Ultralytics in 2023, represents a significant advancement in the YOLO family. It can be exported to ONNX format and run efficiently with OpenCV's DNN module.

```python
import cv2
import numpy as np

class YOLOv8Detector:
    """YOLOv8 ONNX Object Detector"""

    def __init__(self, onnx_model_path, conf_threshold=0.5, iou_threshold=0.4):
        """
        Initialize YOLOv8 detector

        To export YOLOv8 to ONNX:
        pip install ultralytics
        yolo export model=yolov8n.pt format=onnx
        """
        self.net = cv2.dnn.readNetFromONNX(onnx_model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # COCO class names (80 classes)
        self.classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
            "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]

        # Generate random colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3),
                                        dtype=np.uint8)

    def detect(self, img):
        """
        YOLOv8 object detection

        YOLOv8 output format (different from YOLOv5):
        - Shape: (1, 84, 8400) for 640x640 input
        - First 4 rows: [x_center, y_center, width, height]
        - Rows 4-83: class probabilities (80 classes)
        - No objectness score (unlike YOLOv5)
        """
        height, width = img.shape[:2]

        # Preprocessing: letterbox resize
        input_size = 640  # YOLOv8 was trained at 640×640; using a different size degrades accuracy
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1/255.0,  # Normalize to [0,1] — YOLOv8 training uses this range (no per-channel mean)
            size=(input_size, input_size),
            mean=(0, 0, 0),       # No mean subtraction for YOLO-family models; they rely on [0,1] scaling alone
            swapRB=True,          # Convert BGR→RGB to match YOLOv8's training data format
            crop=False            # Preserve aspect ratio via padding rather than cropping to avoid distortion
        )

        # Inference
        self.net.setInput(blob)
        outputs = self.net.forward()

        # YOLOv8 output shape: (1, 84, 8400)
        # Transpose to (8400, 84) for easier processing
        outputs = outputs[0].transpose()  # (8400, 84)

        boxes = []
        confidences = []
        class_ids = []

        # Scale factors for coordinate conversion
        x_scale = width / input_size
        y_scale = height / input_size

        for detection in outputs:
            # Extract class scores (rows 4-83)
            class_scores = detection[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence > self.conf_threshold:
                # Extract bounding box (rows 0-3: cx, cy, w, h)
                cx = detection[0] * x_scale
                cy = detection[1] * y_scale
                w = detection[2] * x_scale
                h = detection[3] * y_scale

                # Convert to top-left corner format
                x = int(cx - w / 2)
                y = int(cy - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            self.conf_threshold, self.iou_threshold
        )

        results = []
        for i in indices:
            results.append({
                'box': boxes[i],
                'confidence': confidences[i],
                'class_id': class_ids[i],
                'class_name': self.classes[class_ids[i]]
            })

        return results

    def draw(self, img, results):
        """Visualize detection results"""
        for det in results:
            x, y, w, h = det['box']
            class_id = det['class_id']
            confidence = det['confidence']

            color = [int(c) for c in self.colors[class_id]]

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label = f"{det['class_name']}: {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(img, (x, y - label_h - 10), (x + label_w, y), color, -1)
            cv2.putText(
                img, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return img

# Usage example
# First, export YOLOv8 model:
# pip install ultralytics
# yolo export model=yolov8n.pt format=onnx
# This creates yolov8n.onnx

detector = YOLOv8Detector('yolov8n.onnx', conf_threshold=0.5)
img = cv2.imread('street.jpg')
results = detector.detect(img)
output = detector.draw(img, results)

print(f"Detected {len(results)} objects:")
for r in results:
    print(f"  - {r['class_name']}: {r['confidence']:.2%}")

cv2.imshow('YOLOv8 Detection', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 7.2 SAM (Segment Anything Model) with ONNX

Meta's Segment Anything Model (SAM) enables powerful image segmentation. While SAM is typically used with PyTorch, the encoder and decoder can be exported to ONNX and run with OpenCV DNN for inference.

```python
import cv2
import numpy as np

class SAMONNXDetector:
    """
    Simplified SAM ONNX inference with OpenCV

    SAM consists of two components:
    1. Image Encoder: Encodes input image to embeddings
    2. Mask Decoder: Generates masks from embeddings and prompts

    To export SAM to ONNX, see:
    https://github.com/facebookresearch/segment-anything
    """

    def __init__(self, encoder_path, decoder_path):
        """
        Initialize SAM with ONNX models

        Args:
            encoder_path: Path to encoder ONNX model
            decoder_path: Path to decoder ONNX model
        """
        self.encoder = cv2.dnn.readNetFromONNX(encoder_path)
        self.decoder = cv2.dnn.readNetFromONNX(decoder_path)
        self.image_size = 1024  # SAM default size

    def preprocess(self, img):
        """Preprocess image for SAM encoder"""
        # Resize to 1024x1024
        h, w = img.shape[:2]
        scale = self.image_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(img, (new_w, new_h))

        # Pad to square
        padded = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Normalize (ImageNet style)
        normalized = padded.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std

        # Convert to blob (1, 3, 1024, 1024)
        blob = cv2.dnn.blobFromImage(normalized, 1.0, swapRB=True)

        return blob, scale

    def encode_image(self, img):
        """Generate image embeddings using encoder"""
        blob, scale = self.preprocess(img)
        self.encoder.setInput(blob)
        embeddings = self.encoder.forward()
        return embeddings, scale

    def segment_with_point(self, img, point_coords, point_labels):
        """
        Segment image with point prompts

        Args:
            img: Input image
            point_coords: List of (x, y) coordinates
            point_labels: List of labels (1=foreground, 0=background)

        Returns:
            Segmentation mask
        """
        # Get image embeddings
        embeddings, scale = self.encode_image(img)

        # Scale point coordinates
        scaled_coords = np.array(point_coords) * scale

        # Prepare decoder inputs
        point_coords_input = scaled_coords.reshape(1, -1, 2).astype(np.float32)
        point_labels_input = np.array(point_labels).reshape(1, -1).astype(np.float32)

        # Run decoder
        # Note: Actual SAM ONNX decoder has specific input format
        # This is simplified - refer to official SAM ONNX export
        self.decoder.setInput(embeddings, 'image_embeddings')
        # Additional inputs for decoder would be set here

        mask = self.decoder.forward()

        return mask

# Conceptual usage (requires actual SAM ONNX models)
# sam = SAMONNXDetector('sam_vit_h_encoder.onnx', 'sam_vit_h_decoder.onnx')
# img = cv2.imread('image.jpg')
#
# # Segment with point prompt
# point_coords = [(100, 150)]  # Click location
# point_labels = [1]  # Foreground point
# mask = sam.segment_with_point(img, point_coords, point_labels)
```

**Note**: Running SAM with OpenCV DNN is complex due to its architecture. For production use, consider using the official SAM implementation or pre-built inference servers.

### 7.3 Model Zoo and Current Landscape (2025)

The object detection and segmentation landscape has evolved significantly. Here's an overview of popular models and their ONNX availability:

| Model Family | Latest Version | ONNX Support | Use Case | Performance |
|--------------|----------------|--------------|----------|-------------|
| **YOLOv8** | v8.1 (2024) | ✅ Native | Real-time detection | mAP 53.9 (YOLOv8x) |
| **YOLOv9** | v9.0 (2024) | ✅ Export | Improved accuracy | mAP 55.6 |
| **YOLOv10** | v10.0 (2024) | ✅ Export | NMS-free YOLO | Faster inference |
| **YOLOv11** | v11.0 (2024) | ✅ Export | Latest Ultralytics | State-of-the-art |
| **RT-DETR** | v2 (2024) | ✅ Export | Transformer detector | mAP 53.1, no NMS |
| **SAM** | v1.0 (2023) | ✅ Complex | Universal segmentation | Zero-shot capable |
| **SAM 2** | v2.0 (2024) | ✅ Complex | Video segmentation | Temporal tracking |
| **Depth Anything** | v2 (2024) | ✅ Export | Monocular depth | Fast, accurate |
| **GroundingDINO** | v1.5 (2024) | ⚠️ Limited | Text-prompted detection | Open-vocabulary |
| **DINO v2** | v2.0 (2024) | ✅ Export | Self-supervised features | Strong backbone |

#### 7.3.1 Quick Start: YOLOv11 with ONNX

```python
# Install Ultralytics
# pip install ultralytics

# Export YOLOv11 to ONNX (Python)
from ultralytics import YOLO

model = YOLO('yolov11n.pt')  # n, s, m, l, x variants
model.export(format='onnx', dynamic=False)  # Creates yolov11n.onnx

# Then use with OpenCV (same as YOLOv8 example above)
# detector = YOLOv8Detector('yolov11n.onnx')  # API compatible
```

#### 7.3.2 RT-DETR: Transformer-based Detection

RT-DETR (Real-Time DEtection TRansformer) eliminates the need for NMS by using a transformer architecture:

```python
import cv2
import numpy as np

class RTDETRDetector:
    """RT-DETR ONNX Detector (NMS-free)"""

    def __init__(self, onnx_path, conf_threshold=0.5):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.conf_threshold = conf_threshold

        # COCO classes (same as YOLO)
        self.classes = ["person", "bicycle", "car", ...]  # 80 classes

    def detect(self, img):
        """
        RT-DETR detection (no NMS required)

        Output format: Direct bounding boxes and scores
        Shape: (1, 300, 6) - top 300 detections
        Each detection: [x1, y1, x2, y2, confidence, class_id]
        """
        height, width = img.shape[:2]

        # Preprocessing (RT-DETR uses 640x640)
        blob = cv2.dnn.blobFromImage(
            img, 1/255.0, (640, 640),
            mean=(0, 0, 0), swapRB=True, crop=False
        )

        self.net.setInput(blob)
        outputs = self.net.forward()

        # Parse outputs (already NMS-filtered by model)
        results = []
        for detection in outputs[0]:  # (300, 6)
            confidence = detection[4]
            if confidence > self.conf_threshold:
                class_id = int(detection[5])

                # Scale coordinates
                x1 = int(detection[0] * width)
                y1 = int(detection[1] * height)
                x2 = int(detection[2] * width)
                y2 = int(detection[3] * height)

                results.append({
                    'box': [x1, y1, x2-x1, y2-y1],
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': self.classes[class_id]
                })

        return results

# Export RT-DETR to ONNX:
# pip install ultralytics
# yolo export model=rtdetr-l.pt format=onnx
```

#### 7.3.3 Model Selection Guide

**For Real-time Applications (Edge Devices, Mobile)**:
- YOLOv8n/s: Fastest, good for embedded systems
- YOLOv10n: No NMS overhead, even faster
- RT-DETR-s: Best accuracy/speed trade-off

**For High Accuracy (Server, Cloud)**:
- YOLOv11x: State-of-the-art detection
- RT-DETR-x: Transformer advantages
- Ensemble multiple models

**For Segmentation**:
- YOLOv8-seg: Instance segmentation (ONNX native)
- SAM/SAM2: Universal segmentation (prompt-based)
- Depth Anything: Depth estimation

**OpenCV DNN Compatibility**:
- ✅ Full support: YOLOv8-11, RT-DETR, MobileNet-SSD
- ⚠️ Partial: SAM (complex multi-stage pipeline)
- ❌ Limited: Models requiring custom ops (GroundingDINO)

**Performance Benchmarks (on RTX 4090, 2024)**:

| Model | Input Size | FPS (CUDA) | mAP | OpenCV DNN |
|-------|------------|------------|-----|------------|
| YOLOv8n | 640 | 450 | 37.3 | ✅ Good |
| YOLOv11m | 640 | 200 | 51.5 | ✅ Good |
| RT-DETR-l | 640 | 110 | 53.1 | ✅ Excellent |
| YOLOv8x | 640 | 80 | 53.9 | ✅ Good |

---

## 8. Exercises

### Exercise 1: Object Detection Performance Comparison

Write a program that compares the performance of YOLO and SSD.

**Requirements**:
- Use the same test image set
- Measure detection speed
- Compare detection accuracy (IoU-based)
- Visualize results

<details>
<summary>Hint</summary>

```python
def compare_detectors(img, yolo_detector, ssd_detector):
    # YOLO detection
    yolo_start = time.time()
    yolo_results = yolo_detector.detect(img)
    yolo_time = time.time() - yolo_start

    # SSD detection
    ssd_start = time.time()
    ssd_results = ssd_detector.detect(img)
    ssd_time = time.time() - ssd_start

    return {
        'yolo': {'results': yolo_results, 'time': yolo_time},
        'ssd': {'results': ssd_results, 'time': ssd_time}
    }
```

</details>

### Exercise 2: Custom Class Filtering

Implement filtering functionality to detect only specific classes.

**Requirements**:
- Specify list of classes to detect
- Ignore other classes
- Set class-specific colors
- Set class-specific confidence thresholds

<details>
<summary>Hint</summary>

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

### Exercise 3: Video Object Tracking + Detection

Implement stable video object recognition by combining detection and tracking.

**Requirements**:
- Perform detection every N frames
- Use tracking for intermediate frames
- Assign and maintain IDs
- Re-detect on tracking failure

<details>
<summary>Hint</summary>

```python
class DetectionTracker:
    def __init__(self, detector, detect_every_n=5):
        self.detector = detector
        self.detect_every_n = detect_every_n
        self.trackers = {}
        self.frame_count = 0

    def process(self, frame):
        if self.frame_count % self.detect_every_n == 0:
            # New detection
            detections = self.detector.detect(frame)
            self.update_trackers(frame, detections)
        else:
            # Update existing trackers
            self.update_existing_trackers(frame)

        self.frame_count += 1
        return self.get_current_positions()
```

</details>

### Exercise 4: Model Ensemble

Improve accuracy by combining results from multiple models.

**Requirements**:
- Use 2 or more models
- Merge detection results (Weighted NMS)
- Calibrate confidence scores
- Output final results

<details>
<summary>Hint</summary>

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

    # Soft-NMS or Weighted Box Fusion
    final_results = weighted_nms(all_boxes, all_scores, all_classes)
    return final_results
```

</details>

### Exercise 5: Real-time Object Counting System

Implement a system that counts specific objects (e.g., people, vehicles) in video.

**Requirements**:
- Real-time detection
- Set counting line/region
- Distinguish entry/exit
- Display statistics (hourly, cumulative)

<details>
<summary>Hint</summary>

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
            # Object center y coordinate
            _, y, _, h = obj['box']
            center_y = y + h // 2

            # Compare with previous position to check line crossing
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

## Next Steps

- [20_Practical_Projects.md](./20_Practical_Projects.md) - Document scanner, lane detection, AR markers, face filters

---

## References

- [OpenCV DNN Module](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [SSD Paper](https://arxiv.org/abs/1512.02325)
- [OpenCV Model Zoo](https://github.com/opencv/opencv/tree/master/samples/dnn)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
